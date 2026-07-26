# torchcell/models/cell_graph_transformer_metabolism
# [[torchcell.models.cell_graph_transformer_metabolism]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/cell_graph_transformer_metabolism
# Test file: tests/torchcell/models/test_cell_graph_transformer_metabolism.py

"""CGT-Metabolism: the Cell Graph Transformer with production / metabolome readouts.

This is the model class the metabolism work (pigment production, metabolome transfer,
and later the enzyme-constrained flux layer) is built inside. Track A -- the SIMB
demonstration -- activates only the readout heads; no flux layer is involved.

WHAT IS REUSED, AND WHY IT IS REUSED RATHER THAN COPIED
-------------------------------------------------------
The encoder (learnable gene embeddings + CLS token + the graph-regularized transformer
layers), the graph-regularization KL term, and the EQUIVARIANT perturbation operator
(``EquivariantPerturbationTransform``) plus the Type-II ``PerturbationHead`` are taken
from :mod:`torchcell.models.equivariant_cell_graph_transformer` **unchanged**, by
subclassing :class:`~torchcell.models.equivariant_cell_graph_transformer.CellGraphTransformer`
and delegating the whole forward to it.

That is a stronger guarantee than transcribing those ~900 lines into this file: a
transcription can silently drift from the original, whereas an inherited implementation
IS the original. What subclassing does NOT make free is initialization parity -- new
parameters created in a subclass ``__init__`` consume the global RNG stream, so a head
constructed in the wrong place would change every encoder weight for a given seed and
quietly invalidate any comparison against a Fig-3 run. Every metabolism head is therefore
constructed strictly AFTER ``super().__init__()`` returns, and
``tests/torchcell/models/test_cell_graph_transformer_metabolism.py`` asserts allclose
outputs against the parent at a fixed seed with the metabolism heads disabled.

THE THREE HEADS (Track A)
-------------------------
Declared through ``heads_config`` alongside the inherited ``global`` / ``per_gene`` /
``per_metabolite`` entries. Each extra entry needs a ``kind``:

* ``betaxanthin``   -- ``kind: scalar``. Cachera 2023 CRI-SPA corrected fluorescence,
  population-centred (so it can be negative), one value per strain.
* ``beta_carotene`` -- ``kind: scalar``. Ozaydin 2013 colony-colour score, a SUBJECTIVE
  ORDINAL on -5..+5. Report Spearman for this target, not Pearson.
* ``mulleder19``    -- ``kind: vector``, ``output_dim: 19``. Mulleder 2016 intracellular
  amino-acid concentrations in mM over 19 FIXED key-sorted columns
  (``alanine ... valine``). Deliberately NOT aligned to Yeast9 metabolite ids: the 19
  columns are dense in every record, so Track A needs no ``s_NNNN`` mapping at all.

**They must not share a head.** The three targets are a centred fluorescence, an ordinal
in -5..+5, and an absolute mM concentration -- mutually incomparable units. One shared
scalar head would be asked to regress three different scales onto one output, and the
largest-magnitude target would dominate the loss. Separate heads, separate losses, one
shared encoder: that separation IS the multitask claim, and it costs three small MLPs.

TRACK B (not implemented here)
------------------------------
The metabolite/enzyme entity layers, the flux readout with the box satisfied by
reparameterization, the soft ``Sv = 0`` / capacity / budget / thermodynamic terms, and
the GPR-derived availability chain all attach at the same place these heads do -- the
``(h_CLS, H_genes_pert)`` pair returned by the inherited forward. Nothing in Track A
forecloses them; nothing in Track A pays for them either.
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
import torch.nn as nn
from torch_geometric.data import HeteroData

from torchcell.models.equivariant_cell_graph_transformer import CellGraphTransformer

#: heads_config keys handled by the parent class; everything else is a metabolism head.
INHERITED_HEADS = frozenset({"global", "per_gene", "per_metabolite"})


class ProductScalarHead(nn.Module):
    """Whole-cell readout for a SCALAR production phenotype (one value per strain).

    Reads the CLS token, optionally concatenated with a mean pool over the equivariant
    perturbed gene embeddings, and emits exactly one value. Structurally this is the S1
    pooled readout; it is a distinct class from the vector head so that "this head is
    scalar" is a property of the module rather than of a config string that some other
    code has to agree with. ``output_dim`` is fixed at 1 by construction -- a scalar
    target assigned into a wider head broadcasts silently, and the cheapest place to
    make that impossible is here.
    """

    def __init__(
        self,
        hidden_dim: int,
        use_gene_pool: bool = True,
        use_pert_pool: bool = True,
        dropout: float = 0.1,
        param_dim: int = 1,
    ) -> None:
        """Build the MLP mapping ``[h_CLS (|| mean_i h_i_pert)]`` to one value.

        Args:
            hidden_dim: Model hidden dimension.
            use_gene_pool: Concatenate a mean pool over ALL gene tokens of the perturbed
                representation. Measured across-batch std 1.3e-2 -- the perturbation
                operator cross-attends, so this is NOT as diluted as "one of 6,607 tokens
                differs" would suggest (see ``perturbed_gene_pool``).
            use_pert_pool: Concatenate a mean pool over ONLY the perturbed gene tokens.
                Measured across-batch std 3.2e-2, i.e. ~2.5x the genome-wide pool. For a
                single-KO strain this is the genotype itself, so keep it on; it is a
                modest signal-to-background gain, not a cure for the observed nulls.
            dropout: Dropout probability.
            param_dim: Distributional params for the single feature (1 point, 2
                gaussian, K quantile). ``output_dim`` stays 1 either way.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = 1
        self.use_gene_pool = use_gene_pool
        self.use_pert_pool = use_pert_pool
        self.param_dim = param_dim
        in_dim = hidden_dim * (1 + int(use_gene_pool) + int(use_pert_pool))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, param_dim),
        )

    def forward(
        self,
        h_CLS: torch.Tensor,
        H_genes_pert: torch.Tensor,
        pert_pool: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``[batch, 1]`` (point) or ``[batch, 1, param_dim]`` (distributional)."""
        batch_size = H_genes_pert.shape[0]
        h = h_CLS.unsqueeze(0).expand(batch_size, -1)
        if self.use_gene_pool:
            h = torch.cat([h, H_genes_pert.mean(dim=1)], dim=-1)
        if self.use_pert_pool:
            if pert_pool is None:
                raise ValueError(
                    "use_pert_pool=True but pert_pool was not supplied; the model's "
                    "forward must pass perturbed_gene_pool(...)."
                )
            h = torch.cat([h, pert_pool], dim=-1)
        out = self.mlp(h)
        if self.param_dim > 1:
            out = out.view(batch_size, 1, self.param_dim)
        return cast(torch.Tensor, out)


def perturbed_gene_pool(
    H_genes_pert: torch.Tensor,
    perturbation_indices: torch.Tensor,
    perturbation_indices_batch: torch.Tensor,
) -> torch.Tensor:
    """Mean over ONLY the perturbed gene tokens, per sample. Returns ``[batch, d]``.

    WHY THIS EXISTS, with the numbers MEASURED rather than assumed (run ``main()`` in this
    file to reproduce them on a real batch):

    ======================================  ==================
    head input term                         across-batch std
    ======================================  ==================
    ``h_CLS``                               **0.000e+00**
    mean over ALL 6607 gene tokens          1.309e-02
    mean over PERTURBED genes only          3.217e-02
    ======================================  ==================

    Two conclusions, one of which corrects an earlier guess:

    1. ``h_CLS`` is the encoding of the UNPERTURBED reference graph (the parent encodes
       ``G`` once at batch size 1) and is broadcast identically across the batch. Its
       across-batch variance is EXACTLY zero -- it cannot distinguish two strains, yet it
       occupies a full ``hidden_dim`` of the head's input width. That part is confirmed.
    2. The genome-wide mean is NOT attenuated ~1/6607, as a naive "only one of 6,607
       tokens differs" argument predicts. The equivariant perturbation operator
       cross-attends, so it has already propagated the perturbation into many gene
       representations before any pooling happens. The perturbed-gene pool carries only
       **~2.5x** the across-batch signal of the genome-wide pool, not ~6607x.

    So this pool is a real but MODEST improvement in signal-to-background, and it is NOT
    a sufficient explanation for the observed mean collapse (job 1331: val pearson exactly
    0.00000, val loss ~0.97 = the variance of the standardized target). The genotype
    signal reaching the head is measurable either way; the model's failure to map it onto
    betaxanthin is a separate problem.

    Keeping it is still justified -- for a single-KO strain the perturbed gene's
    representation IS the genotype, which is the S0 argument from the decoder note applied
    to a scalar target -- but do not cite it as the fix for the nulls.

    Index convention matches the parent operator
    (``equivariant_cell_graph_transformer.py:1649``): ``perturbation_indices`` is a flat
    ``[total_pert_genes]`` tensor of gene indices and ``perturbation_indices_batch``
    assigns each entry to its sample.
    """
    batch_size, _, d = H_genes_pert.shape
    gathered = H_genes_pert[perturbation_indices_batch, perturbation_indices]
    pooled = H_genes_pert.new_zeros(batch_size, d)
    pooled.index_add_(0, perturbation_indices_batch, gathered)
    counts = H_genes_pert.new_zeros(batch_size)
    counts.index_add_(
        0,
        perturbation_indices_batch,
        torch.ones_like(counts[perturbation_indices_batch]),
    )
    # A sample with no perturbation (wild type) keeps a zero vector rather than NaN.
    return pooled / counts.clamp(min=1.0).unsqueeze(-1)


class MetabolomeVectorHead(nn.Module):
    """Whole-cell readout for a FIXED-COLUMN metabolome vector (Mulleder's 19 AAs).

    The columns are the measured metabolite keys in ``sorted()`` order -- the same order
    the ``Perturbation`` graph processor uses when it flattens the ``metabolite_level``
    dict -- so column ``k`` of the output is always the same amino acid. This head does
    NOT read the Yeast9 metabolite node set and needs no ``s_NNNN`` mapping: the
    measured set is small, dense, and fixed, which is precisely why Track A can skip the
    metabolite-id alignment that a general per-metabolite readout would require.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        use_gene_pool: bool = True,
        use_pert_pool: bool = True,
        dropout: float = 0.1,
        param_dim: int = 1,
    ) -> None:
        """Build the MLP mapping the pooled representation to ``output_dim`` columns.

        Args:
            hidden_dim: Model hidden dimension.
            output_dim: Number of measured metabolite columns F (19 for Mulleder).
            use_gene_pool: Concatenate a mean pool over ALL gene tokens of the perturbed
                representation. Measured across-batch std 1.3e-2 -- the perturbation
                operator cross-attends, so this is NOT as diluted as "one of 6,607 tokens
                differs" would suggest (see ``perturbed_gene_pool``).
            use_pert_pool: Concatenate a mean pool over ONLY the perturbed gene tokens.
                Measured across-batch std 3.2e-2, i.e. ~2.5x the genome-wide pool. For a
                single-KO strain this is the genotype itself, so keep it on; it is a
                modest signal-to-background gain, not a cure for the observed nulls.
            dropout: Dropout probability.
            param_dim: Distributional params PER COLUMN; ``output_dim`` stays F.
        """
        super().__init__()
        if output_dim < 2:
            raise ValueError(
                f"MetabolomeVectorHead output_dim must be >= 2, got {output_dim}; "
                "use ProductScalarHead for a single-value phenotype."
            )
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_gene_pool = use_gene_pool
        self.use_pert_pool = use_pert_pool
        self.param_dim = param_dim
        in_dim = hidden_dim * (1 + int(use_gene_pool) + int(use_pert_pool))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim * param_dim),
        )

    def forward(
        self,
        h_CLS: torch.Tensor,
        H_genes_pert: torch.Tensor,
        pert_pool: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``[batch, F]`` (point) or ``[batch, F, param_dim]``."""
        batch_size = H_genes_pert.shape[0]
        h = h_CLS.unsqueeze(0).expand(batch_size, -1)
        if self.use_gene_pool:
            h = torch.cat([h, H_genes_pert.mean(dim=1)], dim=-1)
        if self.use_pert_pool:
            if pert_pool is None:
                raise ValueError(
                    "use_pert_pool=True but pert_pool was not supplied; the model's "
                    "forward must pass perturbed_gene_pool(...)."
                )
            h = torch.cat([h, pert_pool], dim=-1)
        out = self.mlp(h)
        if self.param_dim > 1:
            out = out.view(batch_size, self.output_dim, self.param_dim)
        return cast(torch.Tensor, out)


class CellGraphTransformerMetabolism(CellGraphTransformer):
    """CellGraphTransformer + named production / metabolome readout heads.

    Every extra ``heads_config`` entry (any key outside :data:`INHERITED_HEADS`) becomes
    one head, registered as the submodule ``f"{name}_head"`` and emitted into
    ``reps["head_outputs"][name]``. The spec keys are:

    * ``kind`` -- ``"scalar"`` (:class:`ProductScalarHead`) or ``"vector"``
      (:class:`MetabolomeVectorHead`). Required; there is no default, because guessing
      it is exactly the mistake that lets a scalar target broadcast across a vector head.
    * ``output_dim`` -- feature count F (must be 1 for ``scalar``).
    * ``use_gene_pool`` / ``dropout`` / ``param_dim`` -- as for the inherited heads.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Build the inherited model, then the metabolism heads.

        The metabolism heads are constructed strictly after ``super().__init__()`` so the
        encoder + PERT parameters consume the RNG stream in exactly the parent's order
        and are bit-identical to the parent's at the same seed (parity-tested).
        """
        super().__init__(*args, **kwargs)
        self.metabolism_head_names: list[str] = []
        for name, raw_spec in self.heads_config.items():
            if name in INHERITED_HEADS:
                continue
            spec = dict(raw_spec or {})
            kind = spec.get("kind")
            output_dim = int(spec.get("output_dim", 1))
            head: nn.Module
            if kind == "scalar":
                if output_dim != 1:
                    raise ValueError(
                        f"head '{name}' is kind='scalar' but output_dim={output_dim}; "
                        "a scalar head emits exactly one value."
                    )
                head = ProductScalarHead(
                    hidden_dim=self.hidden_channels,
                    use_gene_pool=bool(spec.get("use_gene_pool", True)),
                    use_pert_pool=bool(spec.get("use_pert_pool", True)),
                    dropout=float(spec.get("dropout", 0.1)),
                    param_dim=int(spec.get("param_dim", 1)),
                )
            elif kind == "vector":
                head = MetabolomeVectorHead(
                    hidden_dim=self.hidden_channels,
                    output_dim=output_dim,
                    use_gene_pool=bool(spec.get("use_gene_pool", True)),
                    use_pert_pool=bool(spec.get("use_pert_pool", True)),
                    dropout=float(spec.get("dropout", 0.1)),
                    param_dim=int(spec.get("param_dim", 1)),
                )
            else:
                raise ValueError(
                    f"metabolism head '{name}' needs kind='scalar' or kind='vector' "
                    f"in its heads_config spec, got {kind!r}."
                )
            self.add_module(f"{name}_head", head)
            self.metabolism_head_names.append(name)

    def forward(
        self, cell_graph: HeteroData, batch: HeteroData, return_attention: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run the inherited forward, then append the metabolism head outputs.

        The encoder, graph-regularization loss and equivariant perturbation operator are
        the parent's, invoked unchanged; the heads read the ``h_CLS`` /
        ``H_genes_pert`` pair the parent already returns.
        """
        predictions, reps = super().forward(cell_graph, batch, return_attention)
        h_CLS = reps["h_CLS"]
        H_genes_pert = reps["H_genes_pert"]
        head_outputs = cast(dict[str, torch.Tensor], reps["head_outputs"])
        pert_pool = perturbed_gene_pool(
            H_genes_pert,
            batch["gene"].perturbation_indices,
            batch["gene"].perturbation_indices_batch,
        )
        for name in self.metabolism_head_names:
            head = cast(nn.Module, getattr(self, f"{name}_head"))
            head_outputs[name] = head(h_CLS, H_genes_pert, pert_pool)
        return predictions, reps

    @property
    def num_parameters(self) -> dict[str, int]:
        """Parameter counts, with one entry per metabolism head."""
        counts = super().num_parameters
        total = counts.pop("total")
        for name in self.metabolism_head_names:
            head = cast(nn.Module, getattr(self, f"{name}_head"))
            n = sum(p.numel() for p in head.parameters() if p.requires_grad)
            counts[f"{name}_head"] = n
            total += n
        counts["total"] = total
        return counts


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
        uri="neo4j+s://torchcell-database.ncsa.illinois.edu:7687",
        username="readonly",
        password="ReadOnly",
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
