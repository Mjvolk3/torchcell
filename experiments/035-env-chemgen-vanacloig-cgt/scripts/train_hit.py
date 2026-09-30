# experiments/035-env-chemgen-vanacloig-cgt/scripts/train_hit.py
# [[experiments.035-env-chemgen-vanacloig-cgt.scripts.train_hit]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/035-env-chemgen-vanacloig-cgt/scripts/train_hit
"""Where does the molecule enter the cell? Four ways of mixing a compound with the genes.

Every model before this one met the compound only at the readout: a vector from its
fingerprint, dotted with the strain's vector. The molecule never touched a gene node and
nothing propagated. Here the gene side is a table over the 6,607 cell-graph genes,
optionally passed through ``gcn_layers`` of message passing over the nine gene networks,
and the compound is mixed in one of four ways (``mix``):

``readout``      ``y = b_g + a(u_c) + MLP([h_g ; u_c ; h_g * W u_c])``. The reference.
``hit``          the compound ATTENDS OVER ALL GENE NODES: ``heads`` heads of attention
                 from ``u_c`` to the gene tokens give a hit distribution ``s_c`` over
                 genes (which genes the molecule reaches) and a hit vector ``m_c``. The
                 readout adds ``m_c`` and the strain's own hit strength ``s_c[g]``.
``hit_prop``     ``hit`` plus propagation: the hit distribution is spread ``hops`` steps
                 outward over each network (row-normalized adjacency, as
                 ``PerturbationGraphPropagation`` does for deletions), and the reach of
                 the molecule at the deleted gene ``g`` along each graph and hop is a
                 feature. This is the ego net around the molecule's targets.
``gene_attend``  the deleted gene attends over {itself, the compound, a null sink}, the
                 form of the cell graph transformer's perturbation operator with the
                 compound as one more perturbation token; the readout reads the
                 perturbed gene.

Trained full batch (every strain against every training compound each step), selected
on the fold's validation compounds, scored on the test compounds centered by the
non-test mean, seeds ensembled: the protocol of ``train_factorized.py``. Writes
``results/factorized/<sweep>/<name>_scores.csv`` and the ensemble prediction to
``$DATA_ROOT/.../predictions/<sweep>/``.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
import time
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
import yaml
from dotenv import load_dotenv
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from torch_geometric.nn import SAGEConv

sys.path.insert(0, osp.dirname(__file__))
from train_factorized import (  # noqa: E402
    CELL_TABLE,
    EMBEDDING_DIR,
    PREDICTIONS,
    centered_val_score,
    compound_input,
)
from train_vanacloig_cgt import build_cell_graph  # noqa: E402
from vanacloig_data import Fold, load_cells, make_folds, score_compounds  # noqa: E402

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
WANDB_MODE = os.getenv("WANDB_MODE")
EXPERIMENT = osp.join(EXPERIMENT_ROOT, "035-env-chemgen-vanacloig-cgt")
GRAPHS = (
    "physical",
    "regulatory",
    "tflink",
    "string12_0_neighborhood",
    "string12_0_fusion",
    "string12_0_cooccurence",
    "string12_0_coexpression",
    "string12_0_experimental",
    "string12_0_database",
)


class HitConfig(BaseModel):
    """One mixing arm at one size, trained on every fold and seed it names."""

    model_config = ConfigDict(extra="forbid")

    name: str
    mix: Literal["readout", "hit", "hit_prop", "gene_attend"] = "readout"
    dim: int = 64
    hidden: int = 256
    gcn_layers: int = 0
    gcn_kind: Literal["union", "relational"] = "union"
    heads: int = 4
    hops: int = 2
    hit_temperature: float = 1.0
    prop_graphs: list[str] = list(GRAPHS)
    embeddings: list[str] = ["fcfp4_count"]
    pca_dim: int | None = None
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    steps: int = 2000
    eval_every: int = 50
    seeds: list[int] = [0, 1, 2]
    fold_seed: int = 0
    n_folds: int = 5
    n_val: int = 4
    folds: list[int] | None = None


class Networks(BaseModel):
    """The nine gene networks over the cell graph's genes, as the model consumes them."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    n_genes: int
    node_ids: list[str]
    per_graph: dict[str, torch.Tensor]  # name -> [2, E] undirected edges
    union: torch.Tensor  # [2, E]
    adjacency_t: dict[str, torch.Tensor]  # name -> sparse row-normalized, transposed


def load_networks(device: torch.device) -> Networks:
    graph = build_cell_graph(list(GRAPHS))
    n = int(graph["gene"].num_nodes)
    per_graph: dict[str, torch.Tensor] = {}
    for edge_type in graph.edge_types:
        if edge_type[0] != "gene" or edge_type[2] != "gene":
            continue
        e = graph[edge_type].edge_index
        e = e[:, e[0] != e[1]]
        e = torch.cat([e, e.flip(0)], dim=1)
        per_graph[edge_type[1]] = torch.unique(e, dim=1)
    union = torch.unique(torch.cat(list(per_graph.values()), dim=1), dim=1)
    adjacency_t = {}
    for name, e in per_graph.items():
        degree = torch.zeros(n).scatter_add_(0, e[0], torch.ones(e.shape[1]))
        weight = 1.0 / degree[e[0]]
        # A_T[j, i] = A[i, j] / deg(i): A_T @ x spreads mass outward from i
        adjacency_t[name] = (
            torch.sparse_coo_tensor(torch.stack([e[1], e[0]]), weight, (n, n))
            .coalesce()
            .to(device)
        )
    return Networks(
        n_genes=n,
        node_ids=list(graph["gene"].node_ids),
        per_graph={k: v.to(device) for k, v in per_graph.items()},
        union=union.to(device),
        adjacency_t=adjacency_t,
    )


class HitModel(nn.Module):
    """Gene table (+ message passing) and one of four compound mixings."""

    def __init__(self, cfg: HitConfig, nets: Networks, feature_dim: int) -> None:
        """``nets`` supplies the edges for message passing and propagation."""
        super().__init__()
        self.cfg = cfg
        self.nets = nets
        d = cfg.dim
        self.table = nn.Embedding(nets.n_genes, d)
        nn.init.normal_(self.table.weight, std=0.1)
        self.gene_bias = nn.Embedding(nets.n_genes, 1)
        nn.init.zeros_(self.gene_bias.weight)
        if cfg.gcn_kind == "union":
            self.convs = nn.ModuleList(SAGEConv(d, d) for _ in range(cfg.gcn_layers))
        else:
            self.convs = nn.ModuleList(
                nn.ModuleDict({g: SAGEConv(d, d) for g in nets.per_graph})
                for _ in range(cfg.gcn_layers)
            )
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(cfg.gcn_layers))
        self.compound = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(feature_dim, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, d),
        )
        self.offset = nn.Linear(d, 1)
        self.pair = nn.Linear(d, d, bias=False)
        parts = 3
        if cfg.mix in ("hit", "hit_prop"):
            assert d % cfg.heads == 0
            self.q = nn.Linear(d, d)
            self.k = nn.Linear(d, d)
            self.v = nn.Linear(d, d)
            self.hit_scale = nn.Linear(1, d)
            parts += 2
        if cfg.mix == "hit_prop":
            n_feat = len(cfg.prop_graphs) * cfg.hops + 1
            self.prop = nn.Sequential(
                nn.Linear(n_feat, d),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(d, d),
            )
            parts += 1
        if cfg.mix == "gene_attend":
            self.attend = nn.MultiheadAttention(
                d, cfg.heads, dropout=cfg.dropout, batch_first=True
            )
            self.null_bias = nn.Parameter(torch.zeros(1))
            self.attend_norm = nn.LayerNorm(d)
            parts = 2
        self.head = nn.Sequential(
            nn.Linear(parts * d, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, 1),
        )

    def genes(self) -> torch.Tensor:
        """[N, d] gene tokens after message passing."""
        h = self.table.weight
        for layer, norm in zip(self.convs, self.norms, strict=True):
            if self.cfg.gcn_kind == "union":
                update = layer(h, self.nets.union)
            else:
                update = sum(
                    conv(h, self.nets.per_graph[g]) for g, conv in layer.items()
                ) / len(layer)
            h = norm(h + nn.functional.gelu(update))
        return h

    def hit(
        self, h: torch.Tensor, u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hit distribution [C, N] over genes and hit vector [C, d] per compound."""
        c, n, d, heads = u.shape[0], h.shape[0], self.cfg.dim, self.cfg.heads
        dh = d // heads
        q = self.q(u).view(c, heads, dh)
        k = self.k(h).view(n, heads, dh)
        v = self.v(h).view(n, heads, dh)
        logits = torch.einsum("chd,nhd->chn", q, k) / (
            dh**0.5 * self.cfg.hit_temperature
        )
        a = torch.softmax(logits, dim=-1)  # [C, heads, N]
        m = torch.einsum("chn,nhd->chd", a, v).reshape(c, d)
        return a.mean(1), m

    def propagate(self, s: torch.Tensor) -> torch.Tensor:
        """[N, C, G * hops + 1] reach of each compound's hit mass at every gene."""
        n = s.shape[1]
        with torch.autocast(device_type=s.device.type, enabled=False):
            x0 = s.T.float()  # [N, C]
            feats = [x0]
            for g in self.cfg.prop_graphs:
                x = x0
                for _ in range(self.cfg.hops):
                    x = torch.sparse.mm(self.nets.adjacency_t[g], x)
                    feats.append(x)
            return torch.log1p(torch.stack(feats, dim=-1) * n)

    def forward(self, gene_idx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """[B, C] predictions for strains deleting ``gene_idx`` under compounds ``x``."""
        h_all = self.genes()
        h = h_all[gene_idx]  # [B, d]
        u = self.compound(x)  # [C, d]
        b, c, d = h.shape[0], u.shape[0], self.cfg.dim
        base = self.gene_bias(gene_idx) + self.offset(u).T  # [B, C]
        hh = h[:, None, :].expand(b, c, d)
        uu = u[None, :, :].expand(b, c, d)
        if self.cfg.mix == "gene_attend":
            # keys: the deleted gene, the compound, and a zero null sink whose logit
            # carries a learned bias so the attention weight depends on the query
            keys = torch.stack([hh, uu, torch.zeros_like(hh)], dim=2).reshape(
                b * c, 3, d
            )
            query = hh.reshape(b * c, 1, d)
            mask = torch.cat(
                [torch.zeros(1, 2, device=h.device), self.null_bias.view(1, 1)], dim=1
            ).expand(b * c, 3)
            attended, _ = self.attend(
                query, keys, keys, attn_mask=mask.unsqueeze(1), need_weights=False
            )
            out = self.attend_norm(query + attended).reshape(b, c, d)
            return base + self.head(torch.cat([out, uu], -1)).squeeze(-1)
        parts = [hh, uu, hh * self.pair(u)[None]]
        if self.cfg.mix in ("hit", "hit_prop"):
            s, m = self.hit(h_all, u)  # [C, N], [C, d]
            parts.append(m[None].expand(b, c, d))
            strength = torch.log1p(s[:, gene_idx].T * h_all.shape[0])  # [B, C]
            parts.append(self.hit_scale(strength.unsqueeze(-1)))
        if self.cfg.mix == "hit_prop":
            reach = self.propagate(s)[gene_idx]  # [B, C, F]
            parts.append(self.prop(reach.to(hh.dtype)))
        return base + self.head(torch.cat(parts, -1)).squeeze(-1)


def train_seed(
    cfg: HitConfig,
    nets: Networks,
    y: NDArray[np.float64],
    gene_idx: torch.Tensor,
    x_raw: NDArray[np.float64],
    fold: Fold,
    seed: int,
    device: torch.device,
) -> tuple[NDArray[np.float64], pd.DataFrame]:
    torch.manual_seed(seed)
    train = fold.train
    mu, sd = x_raw[train].mean(0), x_raw[train].std(0)
    keep = sd > 1e-8
    x = torch.tensor(
        (x_raw[:, keep] - mu[keep]) / sd[keep], dtype=torch.float32, device=device
    )
    y_train = y[:, train]
    y_mean, y_sd = float(np.nanmean(y_train)), float(np.nanstd(y_train))
    target = torch.tensor((y_train - y_mean) / y_sd, dtype=torch.float32)
    mask = torch.isfinite(target).to(device)
    target = torch.nan_to_num(target).to(device)

    model = HitModel(cfg, nets, x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=cfg.steps, pct_start=0.05
    )
    x_train = x[train]
    history, best, best_pred = [], (-np.inf, -1), None
    started = time.time()
    for step in range(cfg.steps):
        model.train()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            pred = model(gene_idx, x_train).float()
        loss = (((pred - target) ** 2) * mask).sum() / mask.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()
        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
            model.eval()
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type, dtype=torch.bfloat16),
            ):
                full = model(gene_idx, x).float().cpu().numpy() * y_sd + y_mean
            val = centered_val_score(y, full, train, fold.val)
            history.append(
                {
                    "step": step + 1,
                    "train_loss": float(loss),
                    "val_centered_mean": val,
                    "seconds": time.time() - started,
                }
            )
            if np.isfinite(val) and val > best[0]:
                best, best_pred = (val, step + 1), full
    assert best_pred is not None
    return best_pred, pd.DataFrame(history).assign(seed=seed, selected_step=best[1])


def run(cfg: HitConfig, sweep: str, device: torch.device, nets: Networks) -> None:
    cells = load_cells(CELL_TABLE, osp.join(EMBEDDING_DIR, "fcfp4_count.npz"))
    y = cells.matrix(cells.response)
    position = {g: i for i, g in enumerate(nets.node_ids)}
    gene_idx = torch.tensor([position[g] for g in cells.genes], device=device)
    x_raw = compound_input(cells, cfg.embeddings, cfg.pca_dim)

    out_dir = osp.join(EXPERIMENT, "results", "factorized", sweep)
    pred_dir = osp.join(PREDICTIONS, sweep)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    run_wandb = wandb.init(
        mode=WANDB_MODE,
        project="torchcell_035-env-chemgen-vanacloig-cgt",
        group=f"{sweep}/{cfg.name}",
        name=cfg.name,
        tags=["hit", sweep],
        config=cfg.model_dump(),
        dir=osp.join(DATA_ROOT, "wandb-experiments", "035-env-chemgen-vanacloig-cgt"),
        reinit=True,
    )
    frames, histories = [], []
    for fold in make_folds(len(cells.compounds), cfg.n_folds, cfg.n_val, cfg.fold_seed):
        if cfg.folds is not None and fold.fold not in cfg.folds:
            continue
        pool = sorted(fold.train + fold.val)
        preds = []
        for seed in cfg.seeds:
            pred, hist = train_seed(cfg, nets, y, gene_idx, x_raw, fold, seed, device)
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
                f"val {hist['val_centered_mean'].max():.3f}, {hist['seconds'].iloc[-1]:.0f} s",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    sweep = osp.splitext(osp.basename(args.sweep))[0]
    with open(args.sweep) as f:
        configs = [HitConfig(**c) for c in yaml.safe_load(f)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = load_networks(device)
    print(
        f"networks: {nets.n_genes} genes, union {nets.union.shape[1]:,} edges, "
        + ", ".join(f"{k} {v.shape[1]:,}" for k, v in nets.per_graph.items()),
        flush=True,
    )
    for cfg in configs:
        if args.only and cfg.name not in args.only:
            continue
        done = osp.join(
            EXPERIMENT, "results", "factorized", sweep, f"{cfg.name}_scores.csv"
        )
        if osp.exists(done):
            print(f"{cfg.name}: already scored, skipped", flush=True)
            continue
        run(cfg, sweep, device, nets)


if __name__ == "__main__":
    main()
