# experiments/035-env-chemgen-vanacloig-cgt/scripts/train_factorized.py
# [[experiments.035-env-chemgen-vanacloig-cgt.scripts.train_factorized]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/035-env-chemgen-vanacloig-cgt/scripts/train_factorized
"""Gene representation times compound encoder, trained on the whole Vanacloig matrix.

Every cell is one strain under one compound, and the strain set is fixed, so a model is
two maps: one from a strain to a vector ``z_g`` and one from a compound's structure to a
vector ``u_c``. The compound map sees only the molecule's embedding, so a compound never
trained on is predicted from its structure: the inductive case.

GENE ENCODERS (``gene_encoder``):

``table``  a learned embedding per queried gene.
``cgt``    the cell graph transformer of ``train_vanacloig_cgt.py``, unchanged: ``z_g``
           is the sum of the perturbed embeddings of the strain's four deleted genes,
           projected to ``dim``; optionally with the identity skip. A step encodes a
           batch of STRAINS once and scores each against every training compound, so the
           encoder runs once per strain rather than once per cell.

HEADS (``head``), both with a per-gene bias ``b_g`` and a per-compound offset ``a(u_c)``:

``bilinear``  ``y = b_g + a(u_c) + <z_g, u_c>``: a low-rank, inductive matrix completion.
``mlp``       ``y = b_g + a(u_c) + MLP([z_g ; u_c ; z_g * u_c])``.

COMPOUND INPUT: the named embeddings of ``031 embed_compounds.py``, each optionally
reduced by a PCA fit on the whole embedded library (5,472 compounds, no labels), then
concatenated and standardized over the fold's training compounds.

SELECTION: each seed trains on the fold's training compounds and keeps the step with the
best mean centered Spearman on the fold's validation compounds. The test compounds are
scored with that step's prediction, and the seeds' selected predictions are also averaged
into an ``ensemble`` row. Test scores are centered by the mean over the fold's NON-TEST
compounds (training plus validation), as in ``baseline_ladder.py``, so the two are
scored against the same reference.

Run with a sweep file (a YAML list of ``FactorizedConfig`` dicts):

    python train_factorized.py --sweep conf/factorized/round2.yaml

Writes ``results/factorized/<sweep>/<name>_scores.csv`` and ``..._history.csv``.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
import time
from types import SimpleNamespace
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
from scipy.stats import rankdata

sys.path.insert(0, osp.dirname(__file__))
from vanacloig_data import (  # noqa: E402
    Fold,
    VanacloigCells,
    load_cells,
    make_folds,
    score_compounds,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
WANDB_MODE = os.getenv("WANDB_MODE")
EXPERIMENT = osp.join(EXPERIMENT_ROOT, "035-env-chemgen-vanacloig-cgt")
CELL_TABLE = (
    "/scratch/projects/torchcell-scratch/experiments/033-env-chemgen-pooled/"
    "cell_table/cell_table.parquet"
)
EMBEDDING_DIR = (
    "/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/"
    "031-env-chemgen-vanacloig-hillenmeyer/experiments/"
    "031-env-chemgen-inhibitor-tolerance/results/embeddings"
)
COUNT_FINGERPRINTS = ("fcfp4_count", "ecfp4_count")
PREDICTIONS = osp.join(
    DATA_ROOT, "experiments", "035-env-chemgen-vanacloig-cgt", "predictions"
)


class FactorizedConfig(BaseModel):
    """One model configuration, trained on every fold and seed it names."""

    model_config = ConfigDict(extra="forbid")

    name: str
    gene_encoder: Literal["table", "cgt"] = "table"
    # operator: the compound acts on every gene of the strain's perturbed state (cgt only)
    head: Literal["bilinear", "mlp", "operator"] = "bilinear"
    embeddings: list[str] = ["fcfp4_count"]
    pca_dim: int | None = 64
    dim: int = 64
    hidden: int = 256
    dropout: float = 0.2
    input_noise: float = 0.0  # gaussian noise on the standardized compound input
    lr: float = 1e-3
    weight_decay: float = 1e-4
    steps: int = 3000
    eval_every: int = 50
    # epochs, when given, set steps = epochs passes over the strains and the eval
    # cadence to every eval_epochs passes; one pass is ceil(n_genes / gene_batch) steps
    epochs: int | None = None
    eval_epochs: int = 1
    # select "val" keeps the step with the best validation score; "last" keeps the final
    # step of a fixed budget. fit_on "pool" trains on the validation compounds too (ridge
    # fits on the whole non-test pool), which only makes sense with select "last"; the
    # logged validation score is then in-sample.
    select: Literal["val", "last"] = "val"
    fit_on: Literal["train", "pool"] = "train"
    gene_batch: int = 0  # 0 = every gene in one step
    se_weight: float | None = None  # weight 1 / (se^2 + s0^2) with s0 this value
    huber: float | None = None  # Huber delta on the standardized target
    seeds: list[int] = [0, 1, 2]
    fold_seed: int = 0
    n_folds: int = 5
    n_val: int = 4
    folds: list[int] | None = None
    # cgt only
    cgt_lambda: float = 1.0
    cgt_layers: int = 8
    cgt_heads: int = 9
    cgt_dim: int = 180  # hidden_channels, must be divisible by cgt_heads
    cgt_lr: float = 5e-4
    identity_skip: bool = False


# ---- data ------------------------------------------------------------------ #
def compound_input(
    cells: VanacloigCells, names: list[str], pca_dim: int | None
) -> NDArray[np.float64]:
    """[41, d] concatenated compound representation, before fold standardization."""
    blocks = []
    for name in names:
        data = np.load(osp.join(EMBEDDING_DIR, f"{name}.npz"), allow_pickle=True)
        x_all = data["X"].astype(np.float64)
        x_all = x_all[:, np.isfinite(x_all).all(axis=0)]
        if name in COUNT_FINGERPRINTS:
            x_all = np.log1p(x_all)
        row = {key: i for i, key in enumerate(data["inchikey"])}
        missing = [k for k in cells.inchikeys if k not in row]
        assert not missing, f"{name} has no embedding for {missing}"
        if pca_dim is not None and pca_dim < x_all.shape[1]:
            mu, sd = x_all.mean(0), x_all.std(0)
            z = (x_all - mu) / np.where(sd > 0, sd, 1.0)
            _, _, vt = np.linalg.svd(z - z.mean(0), full_matrices=False)
            x_all = z @ vt[:pca_dim].T
        blocks.append(np.stack([x_all[row[k]] for k in cells.inchikeys]))
    return np.concatenate(blocks, axis=1)


def fast_spearman(pred: NDArray[np.float64], obs: NDArray[np.float64]) -> float:
    ok = np.isfinite(pred) & np.isfinite(obs)
    if ok.sum() < 10 or np.std(pred[ok]) < 1e-10:
        return float("nan")
    return float(np.corrcoef(rankdata(pred[ok]), rankdata(obs[ok]))[0, 1])


def centered_val_score(
    y: NDArray[np.float64], pred: NDArray[np.float64], train: list[int], held: list[int]
) -> float:
    measured_mean = np.nanmean(y[:, train], axis=1)
    predicted_mean = pred[:, train].mean(axis=1)
    return float(
        np.nanmean(
            [
                fast_spearman(pred[:, j] - predicted_mean, y[:, j] - measured_mean)
                for j in held
            ]
        )
    )


# ---- model ----------------------------------------------------------------- #
class Factorized(nn.Module):
    """Per-gene bias, compound offset, and a gene-by-compound interaction."""

    def __init__(
        self, cfg: FactorizedConfig, n_genes: int, feature_dim: int, encoder=None
    ) -> None:
        """``encoder`` None selects the gene table."""
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        if encoder is None:
            self.table = nn.Embedding(n_genes, cfg.dim)
            nn.init.normal_(self.table.weight, std=0.1)
        else:
            width = cfg.cgt_dim * (2 if cfg.identity_skip else 1)
            self.project = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, cfg.dim))
        self.gene_bias = nn.Embedding(n_genes, 1)
        nn.init.zeros_(self.gene_bias.weight)
        self.compound = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(feature_dim, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.dim),
        )
        self.offset = nn.Linear(cfg.dim, 1)
        if cfg.head == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(3 * cfg.dim, cfg.hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden, 1),
            )

    def genes(
        self, gene_idx: torch.Tensor, strain: torch.Tensor | None, cell_graph=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, dim] gene vectors and the encoder's graph penalty."""
        if self.encoder is None:
            return self.table(gene_idx), torch.zeros((), device=gene_idx.device)
        assert strain is not None
        size, order = strain.shape
        batch = {
            "gene": SimpleNamespace(
                perturbation_indices=strain.reshape(-1),
                perturbation_indices_batch=torch.arange(
                    size, device=strain.device
                ).repeat_interleave(order),
            )
        }
        _, out = self.encoder(cell_graph, batch)
        rows = torch.arange(size, device=strain.device).unsqueeze(1)
        parts = [out["H_genes_pert"][rows, strain].sum(dim=1)]
        if self.cfg.identity_skip:
            parts.append(self.encoder.gene_embedding(strain).sum(dim=1))
        return self.project(torch.cat(parts, -1).float()), out["graph_reg_loss"]

    def forward(
        self,
        z: torch.Tensor,
        gene_idx: torch.Tensor,
        x: torch.Tensor,
        strain: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """[B, C] predictions for gene vectors ``z`` [B, dim] and compounds ``x``."""
        u = self.compound(x)  # [C, dim]
        base = self.gene_bias(gene_idx) + self.offset(u).T  # [B, C]
        if self.cfg.head == "bilinear":
            return base + z @ u.T / np.sqrt(self.cfg.dim)
        b, c = z.shape[0], u.shape[0]
        zz = z[:, None, :].expand(b, c, -1)
        uu = u[None, :, :].expand(b, c, -1)
        return base + self.mlp(torch.cat([zz, uu, zz * uu], -1)).squeeze(-1)


class EnvironmentOperator(nn.Module):
    """The compound as a Type I operator on the strain's perturbed cell state.

    The transformer runs once on the wildtype graph and the deletion operator gives the
    strain's state ``H`` [B, N, d]. A compound token ``e_c`` from the fingerprint then
    acts on every gene: per gene and per head a sigmoid gate ``a`` says how much of the
    compound's value ``v_c`` that gene takes, and the state becomes
    ``H + beta * a * v_c`` with ``beta`` a ReZero scalar at zero, so the model starts as
    the identity on the strain state. The readout is invariant over the genome: the
    deleted genes' rows summed, the mean over all genes, and the token itself, through an
    MLP, beside the gene bias and compound offset. Both readouts are linear in the
    update, so the pooled term is ``mean_i a`` times ``v_c`` and the ``B x C x N x d``
    state is never materialized; only the gates ``[B, C, N, heads]`` are.
    """

    def __init__(
        self, cfg: FactorizedConfig, n_genes: int, feature_dim: int, encoder
    ) -> None:
        """``encoder`` is the cell graph transformer; its width is the operator's."""
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        d, heads = cfg.cgt_dim, cfg.cgt_heads
        self.gene_bias = nn.Embedding(n_genes, 1)
        nn.init.zeros_(self.gene_bias.weight)
        self.compound = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(feature_dim, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, d),
        )
        self.offset = nn.Linear(d, 1)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.gate_bias = nn.Parameter(torch.zeros(heads))
        self.beta = nn.Parameter(torch.zeros(()))
        self.norm = nn.LayerNorm(3 * d)
        self.head = nn.Sequential(
            nn.Linear(3 * d, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, 1),
        )

    def genes(
        self, gene_idx: torch.Tensor, strain: torch.Tensor | None, cell_graph=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, N, d] strain states after the deletion operator, and the graph penalty."""
        assert strain is not None
        size, order = strain.shape
        batch = {
            "gene": SimpleNamespace(
                perturbation_indices=strain.reshape(-1),
                perturbation_indices_batch=torch.arange(
                    size, device=strain.device
                ).repeat_interleave(order),
            )
        }
        _, out = self.encoder(cell_graph, batch)
        return out["H_genes_pert"], out["graph_reg_loss"]

    def forward(
        self,
        z: torch.Tensor,
        gene_idx: torch.Tensor,
        x: torch.Tensor,
        strain: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """[B, C] predictions from strain states ``z`` [B, N, d] under compounds ``x``."""
        assert strain is not None
        b, n, d = z.shape
        heads = self.cfg.cgt_heads
        dh = d // heads
        e = self.compound(x)  # [C, d]
        c = e.shape[0]
        q = self.q(z).view(b, n, heads, dh)
        k = self.k(e).view(c, heads, dh)
        v = self.v(e).view(c, heads, dh)
        # how much of the compound each gene takes, per head
        a = torch.sigmoid(
            torch.einsum("bnhd,chd->bcnh", q, k) / dh**0.5 + self.gate_bias
        )  # [B, C, N, heads]
        s = strain.shape[1]
        rows = torch.arange(b, device=z.device)[:, None]
        h_del = z[rows, strain].sum(1)  # [B, d]
        a_del = torch.gather(
            a, 2, strain[:, None, :, None].expand(b, c, s, heads)
        )  # [B, C, S, heads]
        upd_del = (a_del.sum(2).unsqueeze(-1) * v[None]).reshape(b, c, d)
        upd_pool = (a.mean(2).unsqueeze(-1) * v[None]).reshape(b, c, d)
        z_del = h_del[:, None] + self.beta * upd_del  # [B, C, d]
        z_pool = z.mean(1)[:, None] + self.beta * upd_pool  # [B, C, d]
        ee = e[None].expand(b, c, d)
        base = self.gene_bias(gene_idx) + self.offset(e).T  # [B, C]
        state = self.norm(torch.cat([z_del, z_pool, ee], -1).float())
        return base + self.head(state).squeeze(-1)


# ---- training -------------------------------------------------------------- #
class Context(BaseModel):
    """The data every seed of a config shares."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cells: VanacloigCells
    y: NDArray[np.float64]  # genes x compounds, NaN where unmeasured
    se: NDArray[np.float64]
    x_raw: NDArray[np.float64]  # 41 x d
    strain: torch.Tensor | None = None
    cell_graph: object | None = None


def build_encoder(cfg: FactorizedConfig, cell_graph):
    from omegaconf import OmegaConf

    from torchcell.models.equivariant_cell_graph_transformer import CellGraphTransformer

    raw = OmegaConf.load(osp.join(EXPERIMENT, "conf", "default.yaml"))
    raw.pop("hydra")
    model_cfg = OmegaConf.to_container(raw, resolve=True)["model"]  # type: ignore[index]
    assert cfg.cgt_dim % cfg.cgt_heads == 0, "cgt_dim must be divisible by cgt_heads"
    model_cfg["learnable_embedding"]["size"] = cfg.cgt_dim
    # the prior sits on layer 1 (python index) in default.yaml; a one-layer encoder
    # has no layer 1, so without this it would train with no prior at all. Each head's
    # own lambda is 1 and the single scale is cfg.cgt_lambda in train_seed.
    reg = model_cfg["graph_regularization"]
    reg["graph_reg_layer"] = min(1, cfg.cgt_layers - 1)
    for head in reg["regularized_heads"].values():
        head["layer"] = reg["graph_reg_layer"]
        head["lambda"] = 1.0
    assert len(reg["regularized_heads"]) <= cfg.cgt_heads, "fewer heads than graphs"
    return CellGraphTransformer(
        gene_num=model_cfg["gene_num"],
        hidden_channels=cfg.cgt_dim,
        num_transformer_layers=cfg.cgt_layers,
        num_attention_heads=cfg.cgt_heads,
        cell_graph=cell_graph,
        graph_regularization_config=model_cfg["graph_regularization"],
        perturbation_head_config=model_cfg["perturbation_head"],
        dropout=model_cfg["dropout"],
        graph_reg_lambda=cfg.cgt_lambda,
        node_embeddings=None,
        learnable_embedding_config=model_cfg["learnable_embedding"],
    )


@torch.no_grad()
def predict_all(
    model: nn.Module, ctx: Context, x: torch.Tensor, device: torch.device, batch: int
) -> NDArray[np.float64]:
    model.eval()
    n = ctx.y.shape[0]
    out = []
    for start in range(0, n, batch):
        idx = torch.arange(start, min(n, start + batch), device=device)
        strain = None if ctx.strain is None else ctx.strain[idx.cpu()].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            z, _ = model.genes(idx, strain, ctx.cell_graph)
            pred = model(z.float(), idx, x, strain)
        out.append(pred.float().cpu().numpy())
    return np.concatenate(out)


def train_seed(
    cfg: FactorizedConfig, ctx: Context, fold: Fold, seed: int, device: torch.device
) -> tuple[NDArray[np.float64], pd.DataFrame]:
    """The selected full-matrix prediction (response units) and the step history."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = fold.train if cfg.fit_on == "train" else sorted(fold.train + fold.val)
    x_fit = ctx.x_raw[train]
    mu, sd = x_fit.mean(0), x_fit.std(0)
    keep = sd > 1e-8
    x_std = (ctx.x_raw[:, keep] - mu[keep]) / sd[keep]
    x = torch.tensor(x_std, dtype=torch.float32, device=device)

    y_train = ctx.y[:, train]
    y_mean, y_sd = float(np.nanmean(y_train)), float(np.nanstd(y_train))
    target = torch.tensor((y_train - y_mean) / y_sd, dtype=torch.float32)
    mask = torch.isfinite(target)
    target = torch.nan_to_num(target).to(device)
    mask = mask.to(device)
    weight = torch.ones_like(target)
    if cfg.se_weight is not None:
        se = np.nan_to_num(ctx.se[:, train] / y_sd, nan=1.0)
        w = 1.0 / (se**2 + cfg.se_weight**2)
        weight = torch.tensor(w / w[np.isfinite(y_train)].mean(), device=device).float()

    encoder = None
    if cfg.gene_encoder == "cgt":
        encoder = build_encoder(cfg, ctx.cell_graph)
    model: nn.Module
    if cfg.head == "operator":
        assert encoder is not None, "the operator head needs the cgt gene encoder"
        model = EnvironmentOperator(cfg, ctx.y.shape[0], x.shape[1], encoder)
    else:
        model = Factorized(cfg, ctx.y.shape[0], x.shape[1], encoder)
    model = model.to(device)
    groups = [
        {
            "params": [
                p for n, p in model.named_parameters() if not n.startswith("encoder.")
            ],
            "lr": cfg.lr,
        }
    ]
    if encoder is not None:
        groups.append({"params": list(model.encoder.parameters()), "lr": cfg.cgt_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    n_genes = ctx.y.shape[0]
    gene_batch = cfg.gene_batch or n_genes
    steps_per_epoch = -(-n_genes // gene_batch)
    steps, eval_every = cfg.steps, cfg.eval_every
    if cfg.epochs is not None:
        steps = cfg.epochs * steps_per_epoch
        eval_every = cfg.eval_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.05
    )
    pool = sorted(train + fold.val)
    tag = f"fold{fold.fold}_seed{seed}"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n_genes, generator=generator)
    cursor = 0

    x_train = x[train]
    history = []
    best = (-np.inf, -1)
    best_pred: NDArray[np.float64] | None = None
    started = time.time()
    for step in range(steps):
        model.train()
        if cursor + gene_batch > n_genes:
            order = torch.randperm(n_genes, generator=generator)
            cursor = 0
        idx_cpu = order[cursor : cursor + gene_batch]
        cursor += gene_batch
        idx = idx_cpu.to(device)
        strain = None if ctx.strain is None else ctx.strain[idx_cpu].to(device)
        xin = x_train
        if cfg.input_noise > 0:
            xin = xin + cfg.input_noise * torch.randn_like(xin)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            z, penalty = model.genes(idx, strain, ctx.cell_graph)
            pred = model(z.float(), idx, xin, strain).float()
        t, m, w = target[idx], mask[idx], weight[idx]
        if cfg.huber is None:
            err = (pred - t) ** 2
        else:
            err = nn.functional.huber_loss(pred, t, delta=cfg.huber, reduction="none")
        loss = (err * w * m).sum() / m.sum()
        total = loss + (cfg.cgt_lambda * penalty if encoder is not None else 0.0)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % eval_every == 0 or step + 1 == steps:
            full = predict_all(model, ctx, x, device, 256 if encoder else n_genes)
            full = full * y_sd + y_mean
            val = centered_val_score(ctx.y, full, train, fold.val)
            # the held-out compounds at this step, logged for visibility only; the
            # selected step is still chosen on the validation compounds
            test = centered_val_score(ctx.y, full, pool, fold.test)
            record = {
                "step": step + 1,
                "epoch": (step + 1) / steps_per_epoch,
                "train_loss": float(loss.detach()),
                "penalty": float(penalty),
                "grad_norm": float(grad_norm),
                "lr": float(scheduler.get_last_lr()[0]),
                "val_centered_mean": val,
                "test_centered_mean": test,
                "seconds": time.time() - started,
                "gpu_peak_gb": (
                    torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda"
                    else 0.0
                ),
            }
            history.append(record)
            if wandb.run is not None:
                measured_mean = np.nanmean(ctx.y[:, train], axis=1)
                predicted_mean = full[:, train].mean(axis=1)
                per_compound = {
                    f"{tag}/val_spearman/{ctx.cells.compounds[j]}": fast_spearman(
                        full[:, j] - predicted_mean, ctx.y[:, j] - measured_mean
                    )
                    for j in fold.val
                }
                pool_mean = np.nanmean(ctx.y[:, pool], axis=1)
                pool_pred = full[:, pool].mean(axis=1)
                per_compound |= {
                    f"{tag}/test_spearman/{ctx.cells.compounds[j]}": fast_spearman(
                        full[:, j] - pool_pred, ctx.y[:, j] - pool_mean
                    )
                    for j in fold.test
                }
                wandb.log(
                    {f"{tag}/{k}": v for k, v in record.items() if k != "step"}
                    | per_compound
                    | {"step": step + 1}
                )
            if cfg.select == "last":
                best, best_pred = (val, step + 1), full
            elif np.isfinite(val) and val > best[0]:
                best = (val, step + 1)
                best_pred = full
    assert best_pred is not None, "no finite validation score"
    hist = pd.DataFrame(history).assign(seed=seed, selected_step=best[1])
    return best_pred, hist


def run(cfg: FactorizedConfig, sweep: str, device: torch.device) -> pd.DataFrame:
    cells = load_cells(CELL_TABLE, osp.join(EMBEDDING_DIR, "fcfp4_count.npz"))
    ctx = Context(
        cells=cells,
        y=cells.matrix(cells.response),
        se=cells.matrix(cells.response_se),
        x_raw=compound_input(cells, cfg.embeddings, cfg.pca_dim),
    )
    if cfg.gene_encoder == "cgt":
        from train_vanacloig_cgt import build_cell_graph, strain_indices

        raw = yaml.safe_load(open(osp.join(EXPERIMENT, "conf", "default.yaml")))
        cell_graph = build_cell_graph(list(raw["cell_dataset"]["graphs"]))
        node_ids = list(cell_graph["gene"].node_ids)
        # one strain per queried gene, in ``cells.genes`` order
        per_gene = VanacloigCells(
            **(
                cells.model_dump()
                | {
                    "gene_of_cell": np.arange(len(cells.genes)),
                    "compound_of_cell": np.zeros(len(cells.genes), dtype=np.int64),
                    "response": np.zeros(len(cells.genes)),
                    "response_se": np.zeros(len(cells.genes)),
                }
            )
        )
        ctx.strain = strain_indices(per_gene, node_ids)
        ctx.cell_graph = cell_graph.to(device)

    out_dir = osp.join(EXPERIMENT, "results", "factorized", sweep)
    os.makedirs(out_dir, exist_ok=True)
    # the seed-ensemble prediction for every gene and compound, for stacking offline
    pred_dir = osp.join(PREDICTIONS, sweep)
    os.makedirs(pred_dir, exist_ok=True)
    run_wandb = wandb.init(
        mode=WANDB_MODE,
        project="torchcell_035-env-chemgen-vanacloig-cgt",
        group=f"{sweep}/{cfg.name}",
        name=cfg.name,
        tags=["factorized", sweep],
        config=cfg.model_dump(),
        dir=osp.join(DATA_ROOT, "wandb-experiments", "035-env-chemgen-vanacloig-cgt"),
        reinit=True,
    )
    # every per-fold curve is keyed by the training step, so folds overlay on one page
    run_wandb.define_metric("step")
    run_wandb.define_metric("*", step_metric="step")
    folds = make_folds(len(cells.compounds), cfg.n_folds, cfg.n_val, cfg.fold_seed)
    frames, histories = [], []
    for fold in folds:
        if cfg.folds is not None and fold.fold not in cfg.folds:
            continue
        pool = sorted(fold.train + fold.val)
        preds = []
        for seed in cfg.seeds:
            pred, hist = train_seed(cfg, ctx, fold, seed, device)
            preds.append(pred)
            histories.append(hist.assign(fold=fold.fold))
            frames.append(
                score_compounds(cells, pred, pool, fold.test)
                .assign(
                    member=f"seed{seed}",
                    selected_step=int(hist["selected_step"].iloc[0]),
                )
                .assign(fold=fold.fold)
            )
            print(
                f"{cfg.name} fold {fold.fold} seed {seed}: selected step "
                f"{int(hist['selected_step'].iloc[0])}, val {hist['val_centered_mean'].max():.3f}, "
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
    run_wandb.summary.update(summary)
    print(f"{cfg.name}: {summary}", flush=True)
    run_wandb.finish()
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True, help="YAML list of configs")
    parser.add_argument("--only", nargs="*", default=None, help="config names to run")
    args = parser.parse_args()
    sweep = osp.splitext(osp.basename(args.sweep))[0]
    with open(args.sweep) as f:
        configs = [FactorizedConfig(**c) for c in yaml.safe_load(f)]
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
