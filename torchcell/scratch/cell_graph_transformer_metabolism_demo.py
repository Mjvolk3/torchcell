# torchcell/scratch/cell_graph_transformer_metabolism_demo
# [[torchcell.scratch.cell_graph_transformer_metabolism_demo]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/scratch/cell_graph_transformer_metabolism_demo
"""Run ONE real batch through the model and report what each head actually sees.

Moved verbatim from torchcell/models/cell_graph_transformer_metabolism.py on
2026-10-06 (test campaign Phase 23) because it is a demo and not library code, together
with the module's worktree import bootstrap, which only mattered for direct execution.
The main's own docstring still names the old file path; it is moved unedited.

Run from the repo root:

    PYTHONPATH=$PWD python torchcell/scratch/cell_graph_transformer_metabolism_demo.py

Needs DATA_ROOT and EXPERIMENT_ROOT in .env, the 019-simb-multimodal
fig6_pigment_transfer dataset (and its query file) and the genome, GO and network
data under DATA_ROOT. It runs on the CPU.
"""

if __name__ == "__main__":
    # WORKTREE IMPORT BOOTSTRAP -- must run BEFORE any `torchcell` import below.
    #
    # Running this file directly (python .../cell_graph_transformer_metabolism.py, or a
    # VS Code debug session) puts THIS file's directory on sys.path, but the `torchcell`
    # PACKAGE still resolves through the editable install, which points at the PRIMARY
    # checkout. You then execute the worktree's model file against main's library and get
    # errors like `cannot import name 'DeletionKeyedGenotypeAggregator' from
    # 'torchcell.data'` -- a symbol that exists only on this branch.
    #
    # Prepending the worktree root (three levels up: torchcell/models/<file>) makes the
    # sibling `torchcell` package win, so the file is always debugged against the code it
    # actually lives in. Only applies to direct execution; imports are unaffected.
    import os.path as _osp
    import sys as _sys

    _sys.path.insert(
        0, _osp.dirname(_osp.dirname(_osp.dirname(_osp.abspath(__file__))))
    )

from typing import Any, cast

import torch

from torchcell.models.cell_graph_transformer_metabolism import (
    CellGraphTransformerMetabolism,
    perturbed_gene_pool,
)


def main() -> None:
    r"""Run ONE real batch through the model and report what each head actually sees.

    Run from the repo/worktree root::

        PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \\
            torchcell/models/cell_graph_transformer_metabolism.py

    The point of this entry is the HEAD-INPUT VARIANCE table. A readout can only
    distinguish two strains through terms that DIFFER between them, and on the
    ``fig6_pigment_transfer`` build the three candidate terms differ by orders of
    magnitude:

    * ``h_CLS`` -- the encoding of the unperturbed reference graph, broadcast across the
      batch. Across-batch std is EXACTLY 0: it cannot carry genotype.
    * ``mean over all N=6607 gene tokens`` -- one token differs per single-KO strain, so
      the across-batch std is ~1/6607 of the per-token scale.
    * ``mean over the perturbed gene tokens only`` -- the genotype itself.

    Reading those three numbers side by side is what turns "the model predicts a
    constant" into a diagnosis. It is the check that would have caught the pooling
    defect immediately, so it lives with the model rather than in a notebook.
    """
    import os
    import os.path as osp

    from dotenv import load_dotenv
    from torch_geometric.loader import DataLoader

    import torchcell
    from torchcell.data import (
        DeletionKeyedGenotypeAggregator,
        MeanExperimentDeduplicator,
    )
    from torchcell.data.graph_processor import Perturbation
    from torchcell.data.neo4j_cell import Neo4jCellDataset
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.graph.graph import build_gene_multigraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    # Print WHICH torchcell won. If this is not the tree containing this file, the run is
    # meaningless -- see the worktree import bootstrap at the top of the module.
    print(f"torchcell package: {osp.dirname(osp.abspath(torchcell.__file__))}")
    print(f"this model file  : {osp.abspath(__file__)}")

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    experiment_root = os.environ["EXPERIMENT_ROOT"]
    query_path = osp.join(
        experiment_root, "019-simb-multimodal/queries/fig6_pigment_transfer.cql"
    )
    with open(query_path) as f:
        query = f.read()

    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(data_root, "data/sgd/genome"),
        string_root=osp.join(data_root, "data/string"),
        tflink_root=osp.join(data_root, "data/tflink"),
        genome=genome,
    )
    dataset = Neo4jCellDataset(
        root=osp.join(
            data_root,
            "data/torchcell/experiments/019-simb-multimodal/fig6_pigment_transfer",
        ),
        query=query,
        gene_set=genome.gene_set,
        # Connection comes from NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD in the environment
        # (defaulting to the locally served instance). The radiant host this previously
        # hardcoded no longer serves the database.
        graphs=build_gene_multigraph(
            graph=graph, graph_names=["physical", "regulatory"]
        ),
        incidence_graphs=None,
        node_embeddings={},
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=DeletionKeyedGenotypeAggregator,
        graph_processor=Perturbation(),
    )
    print(f"dataset: {len(dataset)} aggregated genotypes")

    # follow_batch is what creates `perturbation_indices_batch`, which the perturbed-gene
    # pool needs to attribute each perturbed gene to its sample.
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        follow_batch=["perturbation_indices", "phenotype_values"],
    )
    batch = next(iter(loader))

    heads_config: dict[str, Any] = {
        "betaxanthin": {"kind": "scalar", "output_dim": 1},
        "beta_carotene": {"kind": "scalar", "output_dim": 1},
        "mulleder19": {"kind": "vector", "output_dim": 19},
    }
    torch.manual_seed(42)
    model = CellGraphTransformerMetabolism(
        cell_graph=dataset.cell_graph,
        gene_num=6607,
        hidden_channels=32,
        num_transformer_layers=2,
        num_attention_heads=4,
        dropout=0.1,
        heads_config=heads_config,
    )
    model.eval()

    print("\nparameters:")
    for k, v in model.num_parameters.items():
        print(f"  {k:28s} {v:>9,}")

    with torch.no_grad():
        _, reps = model(dataset.cell_graph, batch)
        h_CLS = reps["h_CLS"]
        H_genes_pert = reps["H_genes_pert"]
        pert_pool = perturbed_gene_pool(
            H_genes_pert,
            batch["gene"].perturbation_indices,
            batch["gene"].perturbation_indices_batch,
        )
        b = H_genes_pert.shape[0]
        cls_b = h_CLS.unsqueeze(0).expand(b, -1)
        gene_pool = H_genes_pert.mean(dim=1)

        print(f"\nbatch: {b} strains, H_genes_pert {tuple(H_genes_pert.shape)}")
        print("\nHEAD-INPUT VARIANCE (std ACROSS the batch, averaged over dims):")
        print(f"  {'term':<34}{'across-batch std':>18}")
        for label, t in (
            ("h_CLS (reference cell)", cls_b),
            ("mean over ALL 6607 gene tokens", gene_pool),
            ("mean over PERTURBED genes only", pert_pool),
        ):
            print(f"  {label:<34}{t.std(dim=0).mean().item():>18.3e}")
        ratio = pert_pool.std(dim=0).mean() / gene_pool.std(dim=0).mean().clamp(
            min=1e-12
        )
        print(f"\n  perturbed-pool signal is {ratio.item():.1f}x the genome-wide pool")

        print("\nhead outputs:")
        for name, out in cast(dict[str, torch.Tensor], reps["head_outputs"]).items():
            print(f"  {name:<20}{str(tuple(out.shape)):>16}")


if __name__ == "__main__":
    main()
