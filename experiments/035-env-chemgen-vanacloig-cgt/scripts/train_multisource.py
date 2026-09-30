# experiments/035-env-chemgen-vanacloig-cgt/scripts/train_multisource.py
# [[experiments.035-env-chemgen-vanacloig-cgt.scripts.train_multisource]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/035-env-chemgen-vanacloig-cgt/scripts/train_multisource
"""Does training the compound encoder on other chemogenomic stores help Vanacloig?

The factorized model of ``train_factorized.py`` sees 28 Vanacloig compounds per fold. The
other stores (``aux_matrices.py``) hold thousands more compounds, with every Vanacloig
compound removed, measured on largely the same deletion genes. Here one model is trained
on Vanacloig and the chosen other sources at once:

    y_s(g, c) = b_s(g) + a_s(u_c) + < z_g , A_s u_c > / sqrt(dim)

``u_c``   the SHARED compound encoder over the compound's embedding (as in
          ``train_factorized.py``).
``z_g``   a SHARED gene table over the union of the sources' genes.
``A_s``   a per-source linear map (initialized to the identity), ``b_s`` a per-source
          gene bias and ``a_s`` a per-source compound offset.

``graph_smooth`` adds a Laplacian penalty on the gene table over the union of the nine gene
networks, so genes measured only in some sources (Wildenhain covers 242) can inform their
network neighbors (hypothesis under test).

A step takes the Vanacloig training block (every gene, the fold's training compounds)
and, from each other source, a random batch of its compounds with every gene. Each
source's loss is a masked MSE on its standardized values, the other sources weighted by
``aux_weight``.

SELECTION AND SCORE are exactly those of ``train_factorized.py``: the step with the best
mean centered Spearman on the fold's 4 validation compounds, test compounds scored
centered on the fold's non-test compounds, plus the seed ensemble.

Writes ``results/factorized/<sweep>/<name>_scores.csv`` so ``compare_models.py`` reads
it beside the single-source runs.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
import yaml
from dotenv import load_dotenv
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

sys.path.insert(0, osp.dirname(__file__))
from train_factorized import (  # noqa: E402
    CELL_TABLE,
    COUNT_FINGERPRINTS,
    EMBEDDING_DIR,
    PREDICTIONS,
    centered_val_score,
)
from vanacloig_data import Fold, load_cells, make_folds, score_compounds  # noqa: E402

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
WANDB_MODE = os.getenv("WANDB_MODE")
EXPERIMENT = osp.join(EXPERIMENT_ROOT, "035-env-chemgen-vanacloig-cgt")
AUX_DIR = osp.join(DATA_ROOT, "experiments", "035-env-chemgen-vanacloig-cgt", "aux")


class MultiSourceConfig(BaseModel):
    """One multi-source configuration, trained on every fold and seed it names."""

    model_config = ConfigDict(extra="forbid")

    name: str
    aux_sources: list[str]
    aux_weight: float = 1.0
    aux_compounds: int = 128  # compounds drawn per other source per step
    embeddings: list[str] = ["fcfp4_count"]
    pca_dim: int | None = 64
    dim: int = 64
    hidden: int = 256
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    steps: int = 3000
    eval_every: int = 50
    seeds: list[int] = [0, 1, 2]
    fold_seed: int = 0
    n_folds: int = 5
    n_val: int = 4
    folds: list[int] | None = None
    # Laplacian smoothness of the shared gene table over the union of the nine gene
    # networks of ``conf/default.yaml``: penalty * mean over edges of ||z_i - z_j||^2
    graph_smooth: float = 0.0


class Source(BaseModel):
    """One matrix: its genes as rows of the shared table, compounds as feature rows."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    gene_rows: NDArray[np.int64]  # [genes] index into the shared gene table
    compound_rows: NDArray[np.int64]  # [compounds] index into the feature matrix
    y: NDArray[np.float32]  # genes x compounds, standardized, NaN unmeasured


def library_features(
    names: list[str], pca_dim: int | None
) -> tuple[list[str], NDArray]:
    """Every embedded compound's concatenated representation, keyed by InChIKey."""
    keys: list[str] | None = None
    blocks = []
    for name in names:
        data = np.load(osp.join(EMBEDDING_DIR, f"{name}.npz"), allow_pickle=True)
        x = data["X"].astype(np.float64)
        x = x[:, np.isfinite(x).all(axis=0)]
        if name in COUNT_FINGERPRINTS:
            x = np.log1p(x)
        order = list(data["inchikey"])
        if keys is None:
            keys = order
        else:
            position = {k: i for i, k in enumerate(order)}
            keep = [k for k in keys if k in position]
            blocks = [b[[keys.index(k) for k in keep]] for b in blocks]
            keys = keep
            x = x[[position[k] for k in keys]]
        if pca_dim is not None and pca_dim < x.shape[1]:
            sd = x.std(0)
            z = (x - x.mean(0)) / np.where(sd > 0, sd, 1.0)
            _, _, vt = np.linalg.svd(z, full_matrices=False)
            x = z @ vt[:pca_dim].T
        blocks.append(x)
    assert keys is not None
    return keys, np.concatenate(blocks, axis=1)


def network_edges(all_genes: list[str]) -> torch.Tensor:
    """[2, E] undirected, deduplicated edges of the nine networks over ``all_genes``."""
    from train_vanacloig_cgt import build_cell_graph

    raw = yaml.safe_load(open(osp.join(EXPERIMENT, "conf", "default.yaml")))
    graph = build_cell_graph(list(raw["cell_dataset"]["graphs"]))
    position = {g: i for i, g in enumerate(all_genes)}
    node_ids = list(graph["gene"].node_ids)
    lookup = torch.tensor([position.get(g, -1) for g in node_ids], dtype=torch.long)
    parts = []
    for edge_type in graph.edge_types:
        if edge_type[0] != "gene" or edge_type[2] != "gene":
            continue
        e = lookup[graph[edge_type].edge_index]
        keep = (e[0] >= 0) & (e[1] >= 0) & (e[0] != e[1])
        parts.append(e[:, keep])
    edges = torch.cat(parts, dim=1)
    edges = torch.sort(edges, dim=0).values
    return torch.unique(edges, dim=1)


class MultiSource(nn.Module):
    """Shared gene table and compound encoder, per-source bias, offset and map."""

    def __init__(
        self, cfg: MultiSourceConfig, n_genes: int, feature_dim: int, n_sources: int
    ) -> None:
        """Source 0 is Vanacloig."""
        super().__init__()
        self.dim = cfg.dim
        self.table = nn.Embedding(n_genes, cfg.dim)
        nn.init.normal_(self.table.weight, std=0.1)
        self.bias = nn.Parameter(torch.zeros(n_sources, n_genes))
        self.compound = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(feature_dim, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.dim),
        )
        self.offset = nn.ModuleList(nn.Linear(cfg.dim, 1) for _ in range(n_sources))
        self.maps = nn.Parameter(torch.eye(cfg.dim).repeat(n_sources, 1, 1))

    def forward(
        self, source: int, genes: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """[genes, compounds] prediction for one source."""
        u = self.compound(x)
        v = u @ self.maps[source].T
        z = self.table(genes)
        return (
            self.bias[source, genes][:, None]
            + self.offset[source](u).T
            + z @ v.T / np.sqrt(self.dim)
        )


def train_seed(
    cfg: MultiSourceConfig,
    vanacloig: Source,
    aux: list[Source],
    features: NDArray,
    y_raw: NDArray,
    fold: Fold,
    seed: int,
    device: torch.device,
    edges: torch.Tensor | None = None,
) -> tuple[NDArray, pd.DataFrame]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_rows = vanacloig.compound_rows[fold.train]
    fit_rows = np.unique(np.concatenate([train_rows] + [s.compound_rows for s in aux]))
    mu, sd = features[fit_rows].mean(0), features[fit_rows].std(0)
    keep = sd > 1e-8
    x = torch.tensor(
        (features[:, keep] - mu[keep]) / sd[keep], dtype=torch.float32, device=device
    )

    y_train = y_raw[:, fold.train]
    y_mean, y_sd = float(np.nanmean(y_train)), float(np.nanstd(y_train))
    v_target = torch.tensor((y_train - y_mean) / y_sd, dtype=torch.float32)
    v_mask = torch.isfinite(v_target).to(device)
    v_target = torch.nan_to_num(v_target).to(device)
    v_genes = torch.tensor(vanacloig.gene_rows, device=device)
    aux_t = [
        (
            torch.tensor(s.gene_rows, device=device),
            torch.nan_to_num(torch.tensor(s.y)).to(device),
            torch.isfinite(torch.tensor(s.y)).to(device),
            torch.tensor(s.compound_rows, device=device),
        )
        for s in aux
    ]

    n_genes = (
        int(max([vanacloig.gene_rows.max()] + [s.gene_rows.max() for s in aux])) + 1
    )
    model = MultiSource(cfg, n_genes, x.shape[1], 1 + len(aux)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=cfg.steps, pct_start=0.05
    )
    x_all_v = x[torch.tensor(vanacloig.compound_rows, device=device)]
    x_train_v = x_all_v[torch.tensor(fold.train, device=device)]

    history, best, best_pred = [], (-np.inf, -1), None
    started = time.time()
    for step in range(cfg.steps):
        model.train()
        pred = model(0, v_genes, x_train_v)
        loss_v = (((pred - v_target) ** 2) * v_mask).sum() / v_mask.sum()
        loss_aux = torch.zeros((), device=device)
        for k, (genes, target, mask, rows) in enumerate(aux_t):
            pick = torch.tensor(
                rng.choice(
                    len(rows), size=min(cfg.aux_compounds, len(rows)), replace=False
                ),
                device=device,
            )
            p = model(k + 1, genes, x[rows[pick]])
            m = mask[:, pick]
            loss_aux = loss_aux + (
                ((p - target[:, pick]) ** 2) * m
            ).sum() / m.sum().clamp(min=1)
        loss = loss_v + cfg.aux_weight * loss_aux / max(len(aux_t), 1)
        if edges is not None:
            # a random 200,000 of the edges per step; the mean is unbiased
            pick = torch.randint(
                edges.shape[1], (min(200_000, edges.shape[1]),), device=device
            )
            e = edges[:, pick]
            z = model.table.weight
            smooth = ((z[e[0]] - z[e[1]]) ** 2).sum(1).mean()
            loss = loss + cfg.graph_smooth * smooth
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()
        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
            model.eval()
            with torch.no_grad():
                full = model(0, v_genes, x_all_v).cpu().numpy() * y_sd + y_mean
            val = centered_val_score(y_raw, full, fold.train, fold.val)
            history.append(
                {
                    "step": step + 1,
                    "loss_vanacloig": float(loss_v),
                    "loss_aux": float(loss_aux),
                    "val_centered_mean": val,
                    "seconds": time.time() - started,
                }
            )
            if np.isfinite(val) and val > best[0]:
                best, best_pred = (val, step + 1), full
    assert best_pred is not None
    return best_pred, pd.DataFrame(history).assign(seed=seed, selected_step=best[1])


def run(cfg: MultiSourceConfig, sweep: str, device: torch.device) -> pd.DataFrame:
    cells = load_cells(CELL_TABLE, osp.join(EMBEDDING_DIR, "fcfp4_count.npz"))
    y_raw = cells.matrix(cells.response)
    keys, features = library_features(cfg.embeddings, cfg.pca_dim)
    key_row = {k: i for i, k in enumerate(keys)}

    aux_raw = [np.load(osp.join(AUX_DIR, f"{s}.npz")) for s in cfg.aux_sources]
    all_genes = sorted(set(cells.genes).union(*[set(a["genes"]) for a in aux_raw]))
    gene_row = {g: i for i, g in enumerate(all_genes)}
    vanacloig = Source(
        name="vanacloig",
        gene_rows=np.array([gene_row[g] for g in cells.genes]),
        compound_rows=np.array([key_row[k] for k in cells.inchikeys]),
        y=y_raw.astype(np.float32),
    )
    aux = []
    for name, a in zip(cfg.aux_sources, aux_raw, strict=True):
        embedded = np.array([k in key_row for k in a["inchikeys"]])
        aux.append(
            Source(
                name=name,
                gene_rows=np.array([gene_row[g] for g in a["genes"]]),
                compound_rows=np.array([key_row[k] for k in a["inchikeys"][embedded]]),
                y=a["Y"][:, embedded],
            )
        )
        print(
            f"{name}: {len(a['genes'])} genes, {embedded.sum()} of {len(embedded)} "
            f"compounds embedded",
            flush=True,
        )

    out_dir = osp.join(EXPERIMENT, "results", "factorized", sweep)
    os.makedirs(out_dir, exist_ok=True)
    pred_dir = osp.join(PREDICTIONS, sweep)
    os.makedirs(pred_dir, exist_ok=True)
    run_wandb = wandb.init(
        mode=WANDB_MODE,
        project="torchcell_035-env-chemgen-vanacloig-cgt",
        group=f"{sweep}/{cfg.name}",
        name=cfg.name,
        tags=["multisource", sweep],
        config=cfg.model_dump(),
        dir=osp.join(DATA_ROOT, "wandb-experiments", "035-env-chemgen-vanacloig-cgt"),
        reinit=True,
    )
    edges = None
    if cfg.graph_smooth > 0:
        edges = network_edges(all_genes).to(device)
        print(f"graph smoothness over {edges.shape[1]:,} edges", flush=True)
    frames, histories = [], []
    for fold in make_folds(len(cells.compounds), cfg.n_folds, cfg.n_val, cfg.fold_seed):
        if cfg.folds is not None and fold.fold not in cfg.folds:
            continue
        pool = sorted(fold.train + fold.val)
        preds = []
        for seed in cfg.seeds:
            pred, hist = train_seed(
                cfg, vanacloig, aux, features, y_raw, fold, seed, device
            )
            preds.append(pred)
            histories.append(hist.assign(fold=fold.fold))
            step = int(hist["selected_step"].iloc[0])
            frames.append(
                score_compounds(cells, pred, pool, fold.test).assign(
                    member=f"seed{seed}", selected_step=step, fold=fold.fold
                )
            )
            print(
                f"{cfg.name} fold {fold.fold} seed {seed}: selected step {step}, "
                f"val {hist['val_centered_mean'].max():.3f}, "
                f"{hist['seconds'].iloc[-1]:.0f} s",
                flush=True,
            )
        ensemble = np.mean(preds, axis=0)
        np.save(
            osp.join(pred_dir, f"{cfg.name}_fold{fold.fold}_seed{cfg.fold_seed}.npy"),
            ensemble.astype(np.float32),
        )
        frames.append(
            score_compounds(cells, ensemble, pool, fold.test).assign(
                member="ensemble", selected_step=-1, fold=fold.fold
            )
        )
    scores = pd.concat(frames, ignore_index=True).assign(
        name=cfg.name, fold_seed=cfg.fold_seed
    )
    scores.to_csv(osp.join(out_dir, f"{cfg.name}_scores.csv"), index=False)
    pd.concat(histories).to_csv(
        osp.join(out_dir, f"{cfg.name}_history.csv"), index=False
    )
    summary = {
        f"test/{member}_{target}_median": float(g["spearman"].median())
        for (member, target), g in scores.groupby(["member", "target"])
    }
    wandb.log(summary)
    print(f"{cfg.name}: {summary}", flush=True)
    run_wandb.finish()
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    sweep = osp.splitext(osp.basename(args.sweep))[0]
    with open(args.sweep) as f:
        configs = [MultiSourceConfig(**c) for c in yaml.safe_load(f)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for cfg in configs:
        if args.only and cfg.name not in args.only:
            continue
        done = osp.join(
            EXPERIMENT, "results", "factorized", sweep, f"{cfg.name}_scores.csv"
        )
        if osp.exists(done):
            print(f"{cfg.name}: already scored, skipped", flush=True)
            continue
        run(cfg, sweep, device)


if __name__ == "__main__":
    main()
