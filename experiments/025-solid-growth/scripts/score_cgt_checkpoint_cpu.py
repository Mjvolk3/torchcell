# experiments/025-solid-growth/scripts/score_cgt_checkpoint_cpu.py
# [[experiments.025-solid-growth.scripts.score_cgt_checkpoint_cpu]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/score_cgt_checkpoint_cpu
"""Score a 025 cell-graph-transformer checkpoint on an arm's held-out parts, on CPU.

Job 1640 (W&B 327csnlk, ``cgt_s0_q_kl_004``, arm Q query-pair-disjoint) hit its 12 hour
wall clock before any test evaluation, so the disjoint transformer-versus-null comparison
has a validation maximum and no test number. This closes that without a GPU and without
the 3.2 TB LMDB, the way ``experiments/010-kuzmin-tmi/scripts/score_010_checkpoints_directly.py``
scored the 010 checkpoints: in this configuration the encoder runs once on the wildtype
gene table and depends on nothing in the batch, so a record reaches the model only as the
indices of its three perturbed genes. The encoder is evaluated ONCE here and the
perturbation transform plus readout head are then applied per batch of index triples.

Two traps the 010 scoring recorded, and one more for 025:

1. The index space is ``sorted(genome.gene_set)`` with the genome the trainer used, 6,607
   genes. The build's own gene set (4,352 perturbed genes) is NOT the index space; using it
   gives a plausible number near zero.
2. The readout pooling. The 010 checkpoints pooled by MEAN; ``PerturbationHead`` now
   defaults to SUM and the 025 configs set no ``pooling`` key, so a 025 checkpoint pooled by
   SUM. The value is a CLI argument and the reproduction check below is what validates it.
3. The label normalizer. ``transforms.fit_on: train`` fits the z-score on the arm's training
   records only, so the inverse transform uses the ``train_<arm>`` constants of
   ``results/label_normalization_constants.json``, not the all-record constants 010 used.

Correctness is checked, not assumed: for the validation part the recomputed Pearson is
compared to the run's logged value at the checkpoint's epoch
(``results/additive_baselines_025_arm_val_history.csv``). Training ran bf16-mixed and this
scores in fp32, so agreement is expected at the third to fourth decimal, as it was for 010.

Reads
    $DATA_ROOT/.../025-solid-growth/001-full-build/processed/label_df.parquet
    $DATA_ROOT/.../025-solid-growth/recapitulation/recapitulation_per_triple.csv.gz
    results/subset_S0_indices.json.gz, results/<split artifact of the arm>
    results/label_normalization_constants.json
    the checkpoint

Writes
    results/cgt_checkpoint_scores_<run-id>.json
    results/cgt_checkpoint_pred_<run-id>_<part>.npy and ..._<part>_ids.npy
        (predictions in raw tau units, aligned to the sorted record ids of the part)

Run from the repo root, for example:
    python experiments/025-solid-growth/scripts/score_cgt_checkpoint_cpu.py \
        --checkpoint "$DATA_ROOT/models/checkpoints/gilahyper-1640_.../327csnlk-best-pearson-epoch=07-val/gene_interaction/Pearson=0.1993.ckpt" \
        --run-id 327csnlk --arm Q --parts val,test
"""

import argparse
import gzip
import hashlib
import json
import os
import os.path as osp
import time

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

BUILD = osp.join(DATA_ROOT, "data/torchcell/experiments/025-solid-growth/001-full-build")
RECAP = osp.join(
    DATA_ROOT,
    "data/torchcell/experiments/025-solid-growth/recapitulation/recapitulation_per_triple.csv.gz",
)
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results")
SUBSET = "subset_S0_indices.json.gz"
ARMS = {
    "R": ("pinned_splits_from_010_seed_42.json.gz", "pinned"),
    "Q": ("query_pair_disjoint_splits_025.json.gz", "splits"),
}
VAL_HISTORY_CSV = "additive_baselines_025_arm_val_history.csv"
VAL_KEY = "val/gene_interaction/Pearson"

# The model half of cgt_s0_q_kl_004.yaml, which is cgt_s0_r_kl_000's, which is 010's.
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
        name: {"layer": 1, "head": i, "lambda": 0.001} for i, name in enumerate(GRAPHS)
    },
}
LEARNABLE_EMBEDDING_CONFIG = {
    "enabled": True,
    "size": 180,
    "preprocessor": {"num_layers": 2, "dropout": 0.1},
}


def load_gz(name: str):
    with gzip.open(osp.join(RESULTS_DIR, name), "rt") as f:
        return json.load(f)


def load_records() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(sorted S0 record ids, labels in raw tau, [n, 3] systematic gene names)."""
    subset = np.array(sorted(load_gz(SUBSET)), dtype=np.int64)
    label_df = pd.read_parquet(
        osp.join(BUILD, "processed", "label_df.parquet"),
        columns=["index", "gene_interaction"],
    ).set_index("index")
    y = label_df.loc[subset, "gene_interaction"].to_numpy(dtype=np.float64)
    assert np.isfinite(y).all(), "a subset record carries no gene_interaction label"
    recap = pd.read_csv(RECAP, usecols=["idx_025", "gene_a", "gene_b", "gene_c"]).set_index(
        "idx_025"
    )
    missing = np.setdiff1d(subset, recap.index.to_numpy())
    assert missing.size == 0, f"{missing.size} subset records absent from the recap table"
    genes = recap.loc[subset, ["gene_a", "gene_b", "gene_c"]].to_numpy()
    return subset, y, genes


def part_rows(record_ids: np.ndarray, arm: str, part: str) -> np.ndarray:
    fname, key = ARMS[arm]
    ids = load_gz(fname)[key][part]
    id_to_row = {int(r): i for i, r in enumerate(record_ids)}
    rows = np.array(sorted(id_to_row[int(r)] for r in ids), dtype=np.int64)
    assert rows.size == len(ids), f"{arm}/{part}: a pinned index lies outside S0"
    return rows


def build_cell_graph():
    """The wildtype cell graph the trainer built, from the genome, without the LMDB."""
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
    # Neo4jCellDataset(gene_set=genome.gene_set) adds the base graph from the genome gene
    # set, and the base graph fixes the node ordering: sorted(genome.gene_set).
    graphs_dict = SortedDict(multigraph.graphs.copy())
    graphs_dict["base"] = create_graph_from_gene_set(GeneSet(genome.gene_set))
    multigraph = GeneMultiGraph(graphs=graphs_dict)
    embeddings = NodeEmbeddingBuilder.build(
        embedding_names=[], data_root=DATA_ROOT, genome=genome, graph=graph
    )
    return to_cell_data(multigraph, incidence_graphs=None), embeddings


def load_model(checkpoint: str, cell_graph, embeddings, pooling: str, device) -> CellGraphTransformer:
    model = CellGraphTransformer(
        cell_graph=cell_graph,
        graph_regularization_config=GRAPH_REG_CONFIG,
        perturbation_head_config={"num_heads": 9, "dropout": 0.1, "pooling": pooling},
        graph_reg_lambda=0.0,  # scoring: never materialize the attention matrices
        node_embeddings=embeddings,
        learnable_embedding_config=LEARNABLE_EMBEDDING_CONFIG,
        **MODEL_KWARGS,
    ).to(device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {k[len("model.") :]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    incompatible = model.load_state_dict(state, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    n_rows = model.gene_embedding.weight.shape[0]
    assert n_rows == MODEL_KWARGS["gene_num"] == int(cell_graph["gene"].num_nodes), (
        n_rows,
        MODEL_KWARGS["gene_num"],
        int(cell_graph["gene"].num_nodes),
    )
    for name in ("perturbation_propagation", "observed_label_encoder", "cross_gene_mixing", "post_perturbation_mixing"):
        assert getattr(model, name) is None, f"{name} is set; this scorer covers the 010 configuration only"
    model.eval()
    print(f"loaded {len(state)} tensors from epoch {ckpt['epoch']} of {osp.basename(checkpoint)}")
    return model


@torch.no_grad()
def encode_once(model: CellGraphTransformer) -> tuple[torch.Tensor, torch.Tensor]:
    """Steps 1 to 4 of CellGraphTransformer.forward: the strain-independent encoder, once."""
    device = model.gene_embedding.weight.device
    gene_embs = model.gene_embedding(torch.arange(model.gene_num, device=device))
    # A learnable table already at hidden width has no preprocessor in this configuration
    # (the checkpoint's 158 tensors load strictly either way); the forward applies it only
    # when the model built one.
    if model.embedding_preprocessor is not None:
        gene_embs = model.embedding_preprocessor(gene_embs)
    H = torch.cat([model.cls_token, gene_embs], dim=0).unsqueeze(0)
    for layer in model.transformer_layers:
        H, _ = layer(H, return_attention=False, head_mask=None)
    H = H.squeeze(0)
    return H[0], H[1:]


@torch.no_grad()
def score(model, h_cls, H_genes, idx_triples: np.ndarray, batch_size: int, device) -> np.ndarray:
    """Standardized predictions for [n, 3] index triples, through the transform and head."""
    out = np.empty(idx_triples.shape[0], dtype=np.float64)
    for start in range(0, idx_triples.shape[0], batch_size):
        chunk = idx_triples[start : start + batch_size]
        n = chunk.shape[0]
        pert = torch.from_numpy(chunk.reshape(-1)).to(device)
        assign = torch.arange(n, device=device).repeat_interleave(3)
        H_pert, _ = model.perturbation_transform(H_genes, pert, assign)
        preds = model.perturbation_head(h_cls, H_pert, pert, assign)
        out[start : start + n] = preds.squeeze(-1).float().cpu().numpy()
    return out


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "pearson": float(pearsonr(y, p)[0]),
        "spearman": float(spearmanr(y, p)[0]),
        "mse": float(np.mean((y - p) ** 2)),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
    }


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--arm", default="Q", choices=list(ARMS))
    ap.add_argument("--parts", default="val,test")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N records of each part")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--pooling", default="sum", choices=["sum", "mean"], help="readout pooling the checkpoint trained with")
    args = ap.parse_args()

    n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", torch.get_num_threads()))
    torch.set_num_threads(n_threads)
    device = torch.device(args.device)
    print(f"device {device}, {n_threads} threads")

    record_ids, y, genes = load_records()
    cell_graph, embeddings = build_cell_graph()
    node_ids = list(cell_graph["gene"].node_ids)
    node_to_idx = {g: i for i, g in enumerate(node_ids)}
    vocab = sorted(set(genes.ravel().tolist()))
    missing = [g for g in vocab if g not in node_to_idx]
    assert not missing, f"{len(missing)} build genes absent from the cell graph, e.g. {missing[:5]}"
    idx_triples = np.vectorize(node_to_idx.__getitem__)(genes).astype(np.int64)
    print(f"cell graph {len(node_ids)} genes; S0 {record_ids.size} records over {len(vocab)} genes")

    with open(osp.join(RESULTS_DIR, "label_normalization_constants.json")) as f:
        norm = json.load(f)[f"train_{args.arm}"]
    print(f"normalizer train_{args.arm}: mean {norm['mean']:.9f} sd {norm['sd']:.9f} over {norm['n']} records")

    model = load_model(args.checkpoint, cell_graph, embeddings, args.pooling, device)
    ckpt_epoch = int(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["epoch"])
    t0 = time.time()
    h_cls, H_genes = encode_once(model)
    print(f"encoder once: {time.time() - t0:.1f} s")

    history = pd.read_csv(osp.join(RESULTS_DIR, VAL_HISTORY_CSV))
    logged = history[(history["run"] == args.run_id) & (history["epoch"] == ckpt_epoch)]

    out: dict[str, object] = {
        "run": args.run_id,
        "arm": args.arm,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256(args.checkpoint),
        "epoch": ckpt_epoch,
        "pooling": args.pooling,
        "normalizer": {"fit_on": f"train_{args.arm}", **norm},
        "limit": args.limit,
        "parts": {},
    }
    for part in args.parts.split(","):
        rows = part_rows(record_ids, args.arm, part)
        if args.limit is not None:
            rows = rows[: args.limit]
        t0 = time.time()
        p = score(model, h_cls, H_genes, idx_triples[rows], args.batch_size, device) * norm["sd"] + norm["mean"]
        dt = time.time() - t0
        m = metrics(y[rows], p)
        m |= {"n": int(rows.size), "seconds": round(dt, 1), "seconds_per_1000": round(1000 * dt / rows.size, 2)}
        if part == "val" and len(logged) == 1:
            m["logged_val_pearson_at_epoch"] = float(logged[VAL_KEY].iloc[0])
            m["pearson_minus_logged"] = m["pearson"] - m["logged_val_pearson_at_epoch"]
        out["parts"][part] = m
        print(f"{part}: n {rows.size} pearson {m['pearson']:.6f} spearman {m['spearman']:.6f} mse {m['mse']:.6e} ({m['seconds_per_1000']} s per 1000)")
        if "logged_val_pearson_at_epoch" in m:
            print(f"  logged at epoch {ckpt_epoch}: {m['logged_val_pearson_at_epoch']:.6f}, difference {m['pearson_minus_logged']:+.6f}")
        suffix = "" if args.limit is None else f"_limit{args.limit}"
        np.save(osp.join(RESULTS_DIR, f"cgt_checkpoint_pred_{args.run_id}_{part}{suffix}.npy"), p)
        np.save(osp.join(RESULTS_DIR, f"cgt_checkpoint_pred_{args.run_id}_{part}{suffix}_ids.npy"), record_ids[rows])

    suffix = "" if args.limit is None else f"_limit{args.limit}"
    path = osp.join(RESULTS_DIR, f"cgt_checkpoint_scores_{args.run_id}{suffix}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
