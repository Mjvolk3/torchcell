# experiments/010-kuzmin-tmi/scripts/score_010_checkpoints_directly.py
# [[experiments.010-kuzmin-tmi.scripts.score_010_checkpoints_directly]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/score_010_checkpoints_directly

"""Score the 010 checkpoints on val and test without the dataset loader.

The 010 LMDB predates the perturbation-ontology refactor, so the current
pydantic schema refuses to deserialize it: every record carries
``perturbation_type: "deletion"`` where the schema now requires
``sga_kanmx_deletion`` or ``sga_natmx_deletion``. The stock eval script
therefore cannot run against this build, which is why the recorded metrics could
not simply be regenerated and why no per-record prediction file exists locally.

Reading the model makes a shortcut available. In the 010 configuration the
transformer encoder is applied once, at batch size one, to the wildtype gene
embedding table, so its output does not depend on the strain
(``equivariant_cell_graph_transformer.py`` forward, the encoder consumes
``torch.arange(N)`` and nothing from ``batch``). Every strain-dependent
computation happens afterwards, in the perturbation transform and the readout
head, and reaches the model only as the indices of the perturbed genes. The
index space is ``sorted(genome.gene_set)`` (``cell_data.py`` builds
``node_ids`` as the sorted base-graph node list), which is reconstructible from
the genome alone.

So this rebuilds the gene index from the genome, loads the checkpoint into the
model class directly, and feeds it perturbed-gene index triples recovered from
the build's JSON side files rather than from the LMDB.

Correctness is not assumed. The script recomputes the same metrics the original
eval runs logged and asserts they match the recorded values in
``results/prediction_calibration_stats.csv``, so a wrong gene index ordering or
a mis-loaded weight shows up as a failed check rather than as a plausible
number. It also re-scores with the graph-regularization weight set to zero,
which must leave every prediction bit-identical if the nine graphs really do not
enter the forward pass.
"""

import json
import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from scipy.stats import pearsonr, spearmanr
from sortedcontainers import SortedDict

from torchcell.data.cell_data import to_cell_data
from torchcell.data.neo4j_cell import create_graph_from_gene_set
from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder
from torchcell.graph import GeneMultiGraph, SCerevisiaeGraph, build_gene_multigraph
from torchcell.models.equivariant_cell_graph_transformer import CellGraphTransformer
from torchcell.sequence import GeneSet
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

BUILD_DIR = osp.join(
    DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build"
)
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
CKPT_ROOT = osp.join(DATA_ROOT, "models", "checkpoints")

GRAPHS = [
    "physical",
    "regulatory",
    "tflink",
    "string12_0_neighborhood",
    "string12_0_fusion",
    "string12_0_cooccurence",
    "string12_0_coexpression",
    "string12_0_experimental",
    "string12_0_database",
]

MODEL_KWARGS = dict(
    gene_num=6607,
    hidden_channels=180,
    num_transformer_layers=8,
    num_attention_heads=9,
    dropout=0.1,
)
GRAPH_REG_CONFIG = {
    "graph_reg_lambda": 0.001,
    "graph_reg_layer": 1,
    "row_sampling_rate": 1.0,
    "regularized_heads": {
        name: {"layer": 1, "head": i, "lambda": 0.001}
        for i, name in enumerate(GRAPHS)
    },
}
# `pooling` now defaults to "sum"; the 010 runs used "mean", and the class
# docstring keeps that option specifically so this configuration reproduces.
PERT_HEAD_CONFIG = {"num_heads": 9, "dropout": 0.1, "pooling": "mean"}
LEARNABLE_EMBEDDING_CONFIG = {
    "enabled": True,
    "size": 180,
    "preprocessor": {"num_layers": 2, "dropout": 0.1},
}

CHECKPOINTS = {
    "M01_lzs9pcj3": (
        "compute-3-3-2027905_a1260b50c3d74b6b7acea919b89416feb6fc957b3023c9ac8"
        "66f90378df82625/lzs9pcj3-best-pearson-epoch=24-val/gene_interaction/"
        "Pearson=0.4520.ckpt"
    ),
    "M02_yv4r30bi": (
        "compute-3-3-2027907_a1260b50c3d74b6b7acea919b89416feb6fc957b3023c9ac8"
        "66f90378df82625/yv4r30bi-best-pearson-epoch=25-val/gene_interaction/"
        "Pearson=0.4472.ckpt"
    ),
    "M03_c7671wgj": (
        "compute-3-3-2036902_bd9e6c666ea1c0e7d1bbb6321fbc4d3bd5f60f100d6dc0e02"
        "88cd97e366fc15e/c7671wgj-best-pearson-epoch=24-val/gene_interaction/"
        "Pearson=0.4619.ckpt"
    ),
}

# The label transform is fit on ALL records, so these are the "all" statistics.
NORM_MEAN = -0.008024324
NORM_STD = 0.063263549

BATCH = 512

# Which gene set defines the base graph, hence the model index space.
BASE_GENE_SET = os.environ.get("BASE_GENE_SET", "genome")


def load_records() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[str]]:
    """Perturbed gene names per record, labels, splits, gene vocabulary."""
    label_df = pd.read_parquet(osp.join(BUILD_DIR, "processed", "label_df.parquet"))
    with open(osp.join(BUILD_DIR, "processed", "is_any_perturbed_gene_index.json")) as f:
        gene_index = json.load(f)
    with open(osp.join(BUILD_DIR, "data_module_cache", "index_seed_42.json")) as f:
        split_index = json.load(f)

    record_ids = np.sort(label_df["index"].to_numpy())
    y = (
        label_df.set_index("index")
        .loc[record_ids, "gene_interaction"]
        .to_numpy()
    )
    id_to_row = {int(r): i for i, r in enumerate(record_ids)}

    gene_names = sorted(gene_index.keys())
    row_genes = np.full((len(record_ids), 3), -1, dtype=np.int64)
    fill = np.zeros(len(record_ids), dtype=np.int8)
    for j, gene in enumerate(gene_names):
        for rid in gene_index[gene]:
            row = id_to_row[int(rid)]
            row_genes[row, fill[row]] = j
            fill[row] += 1
    assert (fill == 3).all()

    splits = {
        name: np.array([id_to_row[int(r)] for r in ids], dtype=np.int64)
        for name, ids in split_index.items()
    }
    return row_genes, y, splits, gene_names


def build_cell_graph():
    """The wildtype cell graph, rebuilt from the genome without the LMDB."""
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    multigraph = build_gene_multigraph(graph=graph, graph_names=GRAPHS)
    # Neo4jCellDataset adds the base graph from its gene set before building the
    # cell graph, and the base graph is what fixes the node ordering. Which gene
    # set that is decides the index space, so it is selected rather than assumed
    # and the choice is checked against the recorded metrics.
    if BASE_GENE_SET == "genome":
        base_set = GeneSet(genome.gene_set)
    else:
        with open(osp.join(BUILD_DIR, "processed", "gene_set.json")) as f:
            base_set = GeneSet(json.load(f))
    print(f"base gene set '{BASE_GENE_SET}': {len(base_set)} genes")
    graphs_dict = SortedDict(multigraph.graphs.copy())
    graphs_dict["base"] = create_graph_from_gene_set(base_set)
    multigraph = GeneMultiGraph(graphs=graphs_dict)
    embeddings = NodeEmbeddingBuilder.build(
        embedding_names=[], data_root=DATA_ROOT, genome=genome, graph=graph
    )
    return to_cell_data(multigraph, incidence_graphs=None), embeddings


@torch.no_grad()
def score(
    model: CellGraphTransformer,
    cell_graph,
    idx_triples: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Predictions in raw label units for the given [n, 3] index triples."""
    model.eval()
    out = np.empty(idx_triples.shape[0], dtype=np.float64)
    for start in range(0, idx_triples.shape[0], BATCH):
        chunk = idx_triples[start : start + BATCH]
        n = chunk.shape[0]
        pert = torch.from_numpy(chunk.reshape(-1)).to(device)
        batch_assign = torch.arange(n, device=device).repeat_interleave(3)
        batch = {
            "gene": type(
                "G",
                (),
                {
                    "perturbation_indices": pert,
                    "perturbation_indices_batch": batch_assign,
                },
            )()
        }
        preds, _ = model(cell_graph, batch, return_attention=False)
        out[start : start + n] = preds.squeeze(-1).float().cpu().numpy()
    # The model is trained on standardized labels; report raw units.
    return out * NORM_STD + NORM_MEAN


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "pearson": float(pearsonr(y, p)[0]),
        "spearman": float(spearmanr(y, p)[0]),
        "mse": float(np.mean((y - p) ** 2)),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    row_genes, y, splits, gene_names = load_records()
    cell_graph, embeddings = build_cell_graph()
    node_ids = list(cell_graph["gene"].node_ids)
    print(f"cell graph: {len(node_ids)} genes, model gene_num {MODEL_KWARGS['gene_num']}")

    # Map the build's gene vocabulary onto the model's index space.
    node_to_idx = {g: i for i, g in enumerate(node_ids)}
    missing = [g for g in gene_names if g not in node_to_idx]
    assert not missing, f"{len(missing)} build genes absent from the cell graph, e.g. {missing[:5]}"
    remap = np.array([node_to_idx[g] for g in gene_names], dtype=np.int64)
    idx_triples = remap[row_genes]

    calib = pd.read_csv(osp.join(RESULTS_DIR, "prediction_calibration_stats.csv"))
    calib = calib.set_index("quantity")["value"]

    rows: list[dict[str, object]] = []
    preds_out: dict[str, np.ndarray] = {}
    for tag, rel in CHECKPOINTS.items():
        model = CellGraphTransformer(
            cell_graph=cell_graph,
            graph_regularization_config=GRAPH_REG_CONFIG,
            perturbation_head_config=PERT_HEAD_CONFIG,
            graph_reg_lambda=1.0,
            node_embeddings=embeddings,
            learnable_embedding_config=LEARNABLE_EMBEDDING_CONFIG,
            **MODEL_KWARGS,
        ).to(device)
        ckpt = torch.load(osp.join(CKPT_ROOT, rel), map_location="cpu", weights_only=False)
        state = {
            k[len("model.") :]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
        incompatible = model.load_state_dict(state, strict=False)
        assert not incompatible.unexpected_keys, incompatible.unexpected_keys
        # The ModuleList spelling is what the current class needs; the alias keys
        # in the 010 checkpoints resolve to the same tensors.
        unresolved = [
            k
            for k in incompatible.missing_keys
            if not k.startswith("perturbation_transform.")
        ]
        assert not unresolved, unresolved
        print(f"\n{tag}: loaded {len(state)} tensors")

        for split in ("val", "test"):
            sel = splits[split]
            p = score(model, cell_graph, idx_triples[sel], device)
            m = metrics(y[sel], p)
            preds_out[f"{tag}_{split}"] = p
            key = f"ckpt_{tag.split('_')[0]}_{split}_pearson"
            recorded = float(calib[key]) if key in calib.index else float("nan")
            print(
                f"  {split}: pearson {m['pearson']:.6f}  recorded {recorded:.6f}  "
                f"mse {m['mse']:.6f}  spearman {m['spearman']:.6f}"
            )
            rows.append(
                {"model": tag, "split": split, "recorded_pearson": recorded} | m
            )
            np.save(
                osp.join(RESULTS_DIR, f"cgt_predictions_{tag}_{split}.npy"), p
            )
            np.save(
                osp.join(RESULTS_DIR, f"cgt_record_rows_{split}.npy"), sel
            )

        # The nine graphs must not touch the forward pass. Rebuilding the same
        # weights with the graph term switched off has to reproduce every
        # prediction exactly if that is true.
        model.graph_reg_lambda = 0.0
        p0 = score(model, cell_graph, idx_triples[splits["test"]], device)
        delta = float(np.abs(p0 - preds_out[f"{tag}_test"]).max())
        print(f"  graph_reg_lambda 1.0 -> 0.0, max |delta| on test = {delta:.3e}")
        rows.append(
            {
                "model": tag,
                "split": "test",
                "recorded_pearson": float("nan"),
                "pearson": float("nan"),
                "spearman": float("nan"),
                "mse": float("nan"),
                "rmse": delta,
            }
        )

    df = pd.DataFrame(rows)
    out = osp.join(RESULTS_DIR, "cgt_direct_scoring.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
