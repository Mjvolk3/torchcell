# experiments/043-expression-proteome-morphology/scripts/diagnostics_043.py
# [[experiments.043-expression-proteome-morphology.scripts.diagnostics_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/diagnostics_043.py
"""Two diagnostics of the trained 043 joint model: double composition and the isolate path.

Context (2026-10-26, ``baselines_043.json``): the joint model ``gs1024_w4_bs16`` (seeds
0-2) is below ridge on Caudal (0.160 vs 0.307) and Muenzner protein (0.199 vs 0.402) and
below the additive-singles null on the Sameith doubles (0.199 vs 0.440). These two
diagnostics locate where the model loses the signal. The model is built exactly as
``train_043`` builds it (``checkpoint_config`` -> ``PreparedStore043`` -> ``build_view`` ->
``h5_vector_source`` -> ``build_model``, weights loaded strictly, the
``morphology_probe_043`` code path), run on CPU in eval mode, and read with forward
hooks on the trunk and the operator (the forward is not changed).

D1, composition on the Sameith doubles (split B, 72 doubles read through the Kemmeren
head; ``TrainingView.doubles`` and the Kemmeren single of each gene,
``doubles_singles``). For each checkpoint, ``train_043.predict`` on the doubles and their
singles (the call ``evaluate_doubles`` makes), predictions inverted to released Kemmeren
units, then, with the trainer's centered per-double Pearson (``column_pearson`` over
genes of ``a - m`` against ``b - m``, ``m`` = Kemmeren's training-mean profile, median
over doubles):

* model double vs measured double (``val/doubles/pearson_per_strain_centered_median``);
* sum of model singles vs measured double (``.../additive_model_...``);
* model double vs sum of model singles (the composition statistic);
* sum of measured singles vs measured double (``.../additive_measured_...``, 0.4397);
* model single vs measured single (each single, centered the same way; in-sample fit);
* residual: ``(p_xy - p_x - p_y)`` vs ``(m_xy - m_x - m_y)`` per double over genes
  (uncentered: the two residuals differ by no shared offset), and per gene across the 72
  doubles. The raw residual is shared by any sub-additive composition (a double that is
  ``b (x + y)`` has residual ``-(1 - b)(x + y)``, which tracks the measured residual
  whenever the model singles track the measured singles), so three controls go beside
  it: ``-(p_x + p_y)`` against the measured residual; the orthogonalized residual (per
  double, ``p_xy`` regressed on ``[1, p_x, p_y]`` over genes and ``m_xy`` on
  ``[1, m_x, m_y]``, residual against residual); and the model double, or the sum of
  model singles, against the measured double's part beyond the measured singles;
* slope of ``p_xy`` on ``p_x + p_y`` per double over genes (released units, 0 = the
  reference strain; 1 = additive, below 1 = sub-additive) and the norm ratio
  ``|p_xy| / |p_x + p_y|``; the same for the measured double against the measured sum.

Checkpoints: ``best.pt`` (minimum val/loss; the primary row), plus ``epoch_0050.pt`` and
``last.pt`` where present, of bs16 seeds 0-2 and ``gs1024_w5_onto_inject_op`` seeds 0-1.

D2, the isolate path (Caudal 2024 and Muenzner 2024 protein; one context per isolate,
plus 35 to 91 off-graph marks per record). From the bs16 ``best.pt`` checkpoints, per
record (training and validation records of the dataset):

* context level: ``input_mean`` (mean over genes of the model's input tokens, composite
  projection plus position, before the trunk), ``trunk_mean`` / ``trunk_cls`` (trunk
  output), ``op_cls`` / ``op_mean`` (operator output);
* per gene: the gene's input token, trunk token and post-operator token (``H_pert``,
  what the per-gene head reads together with the ontology token);
* the model's own head output (point prediction in z).

Frozen references (no learned weight): ``frozen_indicators`` = the ``baselines_043``
ridge input for the isolates (per-gene variant, loss-of-function and absence calls, the
GBLUP kinship input; it reproduces that script's 0.307 / 0.402), and
``frozen_ctx_mean`` = the mean over genes of the frozen composite vectors of the isolate
context (each block LayerNorm-standardized without affine, absent rows 0; 3,328), the
frozen analog of ``trunk_mean``.

Every context-level representation goes through the ``baselines_043`` ridge (the kernel
ridge of ``linear_evals_043.ridge_fold`` on the training records, penalty
``(trace K / n) x 10^g``, g in -5..5 by 0.5, chosen on the inner split of 20% of training
units, rng seed 0, by median per-gene Pearson; refit on all training records). Per-gene
ridge: for each gene, ridge from that gene's own token (144) to that gene's target
across training isolates, with one penalty shared by all genes chosen on the same inner
split. Scoring: ``gene_level_metrics`` on the trainer's validation records (all of them;
both datasets are below the 2,000-record cap, so the subset does not move with the
seed), i.e. ``val/<ds>/pearson_per_gene_median`` (genes with >= 3 finite pairs; a
constant prediction is undefined and not in the median).

Stages: ``d1`` and ``d2`` (CPU, slurm: ``gh_diagnostics_043.slurm``) write
``results/diagnostics_043.json``; ``wandb`` adds the logged values at each D1
checkpoint's epoch (light, network); ``report`` draws the figure and prints the tables.

    python experiments/043-expression-proteome-morphology/scripts/diagnostics_043.py --stage d1
    python experiments/043-expression-proteome-morphology/scripts/diagnostics_043.py --stage d2
    python experiments/043-expression-proteome-morphology/scripts/diagnostics_043.py --stage wandb
    python experiments/043-expression-proteome-morphology/scripts/diagnostics_043.py --stage report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import pickle
import subprocess
import sys
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

SCRIPTS = osp.dirname(osp.abspath(__file__))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
from baselines_043 import RIDGE_GRID, inner_split  # noqa: E402
from global_split_043 import Role  # noqa: E402
from linear_evals_043 import ALL as RAW_NAMES  # noqa: E402
from linear_evals_043 import ridge_fold  # noqa: E402
from store_version_043 import store_paths  # noqa: E402
from train_043 import (  # noqa: E402
    ROLE_CODE,
    PreparedStore043,
    TrainingView,
    _nanmedian,
    build_model,
    build_view,
    checkpoint_config,
    column_pearson,
    eval_positions,
    gene_level_metrics,
    h5_vector_source,
    make_batch,
    predict,
)

from torchcell.losses.readout_family_043 import ReadoutLossFamily043  # noqa: E402
from torchcell.models.cell_model_043 import CellModel043  # noqa: E402

EXPERIMENT = "043-expression-proteome-morphology"
STORE_VERSION = "v2"
CKPT_ROOT = "experiments/043-expression-proteome-morphology/checkpoints"
#: label -> run directory under $DATA_ROOT/<CKPT_ROOT>; the leaf is the W&B run id
RUNS: dict[str, str] = {
    "bs16_s0": "all_onto_hlgauss_equal_s0/gs1024_w4_bs16",
    "bs16_s1": "all_onto_hlgauss_equal_s1/gs1024_w4_bs16_s1",
    "bs16_s2": "all_onto_hlgauss_equal_s2/gs1024_w4_bs16_s2",
    "injop_s0": "all_onto_injop_hlgauss_equal_s0/gs1024_w5_onto_inject_op",
    "injop_s1": "all_onto_injop_hlgauss_equal_s1/gs1024_w5_onto_inject_op_s1",
}
D1_FILES = ("best.pt", "epoch_0050.pt", "last.pt")
D2_RUNS = ("bs16_s0", "bs16_s1", "bs16_s2")
ISOLATES = ("caudal", "muenzner")
SEEDS = (0, 1, 2)
VAL_CAP = 2000
MIN_STRAINS = 3
MIN_FIT_PER_GENE = 10
CTX_REPS = ("input_mean", "trunk_mean", "trunk_cls", "op_cls", "op_mean")
GENE_REPS = ("input", "trunk", "op")
WANDB_ENTITY = "zhao-group"
WANDB_PROJECT = "torchcell_043-expression-proteome-morphology"


# ============================================================================ model
class Loaded:
    """A checkpoint built as the trainer builds it (store cached across checkpoints)."""

    _stores: dict[str, PreparedStore043] = {}

    def __init__(self, data_root: str, path: str) -> None:
        """Load ``path`` (sha256 recorded), its prepared store, view and model."""
        with open(path, "rb") as fh:
            self.sha256 = hashlib.sha256(fh.read()).hexdigest()
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.cfg = checkpoint_config(state)
        gen = store_paths(self.cfg.data.store_version)
        if gen.version != STORE_VERSION:
            raise ValueError(f"{path}: store {gen.version}, expected {STORE_VERSION}")
        prep = self.cfg.paths.resolved_prep_dir(
            self.cfg.data.route, gen, self.cfg.data.prep_version
        )
        if prep not in Loaded._stores:
            Loaded._stores[prep] = PreparedStore043.load(prep)
        self.store = Loaded._stores[prep]
        self.view: TrainingView = build_view(
            self.store,
            self.cfg.arm,
            self.cfg.data.mark_cap,
            self.cfg.data.target_transform,
            self.cfg.data.target_stats_by,
            self.cfg.weighting.by,
        )
        source = h5_vector_source(self.cfg.paths, gen)
        tables = [
            source(e, self.view.used_ids[e]) for e in range(len(self.view.used_ids))
        ]
        self.model: CellModel043 = build_model(self.cfg, self.view, tables)
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()
        self.objective = ReadoutLossFamily043(self.model.config, None)
        if self.cfg.loss.kind == "binned_quantile":
            raise ValueError("binned_quantile checkpoints need the training bins")
        self.meta = {
            "checkpoint": path,
            "checkpoint_sha256": self.sha256,
            "epoch": int(state["epoch"]),
            "step": int(state["step"]),
            "best_val_loss": state.get("best_val_loss"),
            "best_val_epoch": state.get("best_val_epoch"),
            "ontology_inject": self.model.config.ontology_inject,
            "hidden_dim": self.model.config.hidden_dim,
            "target_transform": self.cfg.data.target_transform,
            "prep_version": self.store.meta.prep_version,
        }


# ============================================================================ D1
def _centered(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Per-row Pearson over genes of ``a - m`` against ``b - m`` (evaluate_doubles)."""
    return column_pearson((a - m).T, (b - m).T, 3)


def _row_slope_and_ratio(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the OLS slope of ``y`` on ``x`` over finite genes and ``|y| / |x|``."""
    slope = np.full(len(y), np.nan)
    ratio = np.full(len(y), np.nan)
    for i in range(len(y)):
        ok = np.isfinite(y[i]) & np.isfinite(x[i])
        xi, yi = x[i, ok], y[i, ok]
        xc = xi - xi.mean()
        slope[i] = float((xc * (yi - yi.mean())).sum() / (xc * xc).sum())
        ratio[i] = float(np.linalg.norm(yi) / np.linalg.norm(xi))
    return slope, ratio


def _row_residual(y: np.ndarray, *xs: np.ndarray) -> np.ndarray:
    """Per row, the residual of ``y`` regressed on ``[1, *xs]`` over finite genes (NaN
    elsewhere): the part of ``y`` no per-row linear mixture of ``xs`` explains.
    """
    out = np.full(y.shape, np.nan)
    for i in range(len(y)):
        ok = np.isfinite(y[i])
        for x in xs:
            ok &= np.isfinite(x[i])
        A = np.stack([np.ones(int(ok.sum()))] + [x[i, ok] for x in xs], 1)
        coef, *_ = np.linalg.lstsq(A, y[i, ok], rcond=None)
        out[i, ok] = y[i, ok] - A @ coef
    return out


def d1_one(ck: Loaded) -> dict[str, Any]:
    """The D1 statistics of one checkpoint (module docstring)."""
    v, s = ck.view, ck.store
    singles = v.doubles_singles
    n = len(v.doubles)
    pos = np.concatenate([v.doubles, singles[:, 0], singles[:, 1]])
    t0 = time.time()
    pred = predict(
        ck.model, ck.objective, v, pos, ck.cfg.data.batch_size, torch.device("cpu"), 0
    )
    st = v.stats.get("kemmeren")
    released = st.invert(pred["gene"]).astype(np.float64)
    measured = st.invert(v.Z[s.rec_yrow[v.rec[pos]]]).astype(np.float64)
    mp = st.invert(v.train_mean_z["kemmeren"]).astype(np.float64)
    p_xy, p_x, p_y = released[:n], released[n : 2 * n], released[2 * n :]
    m_xy, m_x, m_y = measured[:n], measured[n : 2 * n], measured[2 * n :]
    r = {
        "model_double_vs_measured_double": _centered(p_xy, m_xy, mp),
        "model_singles_sum_vs_measured_double": _centered(p_x + p_y, m_xy, mp),
        "model_double_vs_model_singles_sum": _centered(p_xy, p_x + p_y, mp),
        "measured_singles_sum_vs_measured_double": _centered(m_x + m_y, m_xy, mp),
        "model_single_vs_measured_single": np.concatenate(
            [_centered(p_x, m_x, mp), _centered(p_y, m_y, mp)]
        ),
        "residual_model_vs_measured_per_double": column_pearson(
            (p_xy - p_x - p_y).T, (m_xy - m_x - m_y).T, 3
        ),
        "residual_model_vs_measured_per_gene": column_pearson(
            p_xy - p_x - p_y, m_xy - m_x - m_y, 3
        ),
        # controls for the raw residual: a residual of the form -(1 - b)(x + y) is
        # shared by any sub-additive composition, so these remove it
        "control_minus_model_singles_sum_vs_measured_residual": column_pearson(
            -(p_x + p_y).T, (m_xy - m_x - m_y).T, 3
        ),
        "orth_residual_model_vs_measured": column_pearson(
            _row_residual(p_xy, p_x, p_y).T, _row_residual(m_xy, m_x, m_y).T, 3
        ),
        "model_double_vs_measured_double_beyond_measured_singles": column_pearson(
            p_xy.T, _row_residual(m_xy, m_x, m_y).T, 3
        ),
        "model_singles_sum_vs_measured_double_beyond_measured_singles": column_pearson(
            (p_x + p_y).T, _row_residual(m_xy, m_x, m_y).T, 3
        ),
    }
    slope_m, ratio_m = _row_slope_and_ratio(p_xy, p_x + p_y)
    slope_t, ratio_t = _row_slope_and_ratio(m_xy, m_x + m_y)
    out: dict[str, Any] = {
        **ck.meta,
        "n_doubles": int(n),
        "roles": {
            "doubles": sorted({int(x) for x in v.role[v.doubles]}),
            "singles": sorted({int(x) for x in v.role[singles.ravel()]}),
        },
        "seconds": round(time.time() - t0, 1),
    }
    for k, x in r.items():
        out[k] = {
            "median": _nanmedian(x),
            "mean": float(np.nanmean(x)),
            "n_scored": int(np.isfinite(x).sum()),
        }
        if k != "residual_model_vs_measured_per_gene":
            out[k]["values"] = [float(y) for y in x]
    out["slope_model_double_on_model_singles_sum"] = {
        "median": _nanmedian(slope_m),
        "values": [float(y) for y in slope_m],
    }
    out["norm_ratio_model_double_to_model_singles_sum"] = {
        "median": _nanmedian(ratio_m)
    }
    out["slope_measured_double_on_measured_singles_sum"] = {
        "median": _nanmedian(slope_t),
        "values": [float(y) for y in slope_t],
    }
    out["norm_ratio_measured_double_to_measured_singles_sum"] = {
        "median": _nanmedian(ratio_t)
    }
    # scale of the model's own double and singles against the measured ones
    _, ratio_pm = _row_slope_and_ratio(p_xy, m_xy)
    out["norm_ratio_model_double_to_measured_double"] = {"median": _nanmedian(ratio_pm)}
    _, ratio_s = _row_slope_and_ratio(
        np.concatenate([p_x, p_y]), np.concatenate([m_x, m_y])
    )
    out["norm_ratio_model_single_to_measured_single"] = {"median": _nanmedian(ratio_s)}
    print(
        f"[d1] {ck.meta['checkpoint']} epoch {ck.meta['epoch']}: double "
        f"{out['model_double_vs_measured_double']['median']:.3f}, model sum "
        f"{out['model_singles_sum_vs_measured_double']['median']:.3f}, double vs sum "
        f"{out['model_double_vs_model_singles_sum']['median']:.3f}, measured sum "
        f"{out['measured_singles_sum_vs_measured_double']['median']:.4f}, residual "
        f"{out['residual_model_vs_measured_per_double']['median']:.3f}, slope "
        f"{out['slope_model_double_on_model_singles_sum']['median']:.3f} (measured "
        f"{out['slope_measured_double_on_measured_singles_sum']['median']:.3f})",
        flush=True,
    )
    return out


def stage_d1(data_root: str, payload: dict[str, Any]) -> None:
    res: dict[str, Any] = {}
    for label, rel in RUNS.items():
        for fn in D1_FILES:
            path = osp.join(data_root, CKPT_ROOT, rel, fn)
            if not osp.exists(path):
                res[f"{label}/{fn}"] = {"absent": path}
                continue
            res[f"{label}/{fn}"] = d1_one(Loaded(data_root, path))
    payload["d1"] = {
        "statistic": "median over the 72 doubles of the per-double Pearson r over genes "
        "(centered on Kemmeren's training-mean profile, released units; residual rows "
        "uncentered; per-gene residual = median over genes of the Pearson r across the 72 "
        "doubles)",
        "checkpoints": res,
    }


# ============================================================================ D2
def _ctx_frozen_mean(model: CellModel043, key_rows: torch.Tensor) -> torch.Tensor:
    """``[C, 3328]``: mean over genes of each block's frozen vector, LayerNorm without
    affine, absent rows 0 (the frozen composite context summary).
    """
    parts = []
    comp = model.composite
    for e in range(comp.n_blocks):
        rows = key_rows[..., e]
        have = rows >= 0
        tab = comp.table(e)
        vec = F.layer_norm(tab[rows.clamp_min(0)].float(), (tab.shape[1],))
        vec = vec * have.unsqueeze(-1).to(vec.dtype)
        parts.append(vec.mean(1))
    return torch.cat(parts, -1)


@torch.no_grad()
def extract(
    ck: Loaded, pos: np.ndarray, cols: np.ndarray, batch_size: int, want_frozen: bool
) -> dict[str, np.ndarray]:
    """Context-level, per-gene (``cols`` only, fp16) and head read-outs of ``pos``."""
    model = ck.model
    if model.config.ontology_inject != "head":
        raise ValueError(
            "D2 reads the trunk output as the operator input (inject=head)"
        )
    cap: dict[str, torch.Tensor] = {}

    def trunk_hook(_m: Any, a: tuple[torch.Tensor, ...], o: torch.Tensor) -> None:
        cap["x"], cap["trunk"] = a[0], o

    def op_hook(_m: Any, a: tuple[torch.Tensor, ...], o: torch.Tensor) -> None:
        cap["h_in"], cap["h_out"] = a[0], o

    handles = [
        model.trunk.register_forward_hook(trunk_hook),
        model.operator.register_forward_hook(op_hook),
    ]
    d = model.config.hidden_dim
    n, gc = len(pos), len(cols)
    G = ck.store.Y.shape[1]
    out: dict[str, np.ndarray] = {k: np.zeros((n, d), np.float32) for k in CTX_REPS}
    for k in GENE_REPS:
        out[f"gene_{k}"] = np.zeros((n, gc, d), np.float16)
    out["head"] = np.full((n, G), np.nan, np.float32)
    if want_frozen:
        out["frozen_ctx_mean"] = np.zeros(
            (
                n,
                sum(
                    model.composite.table(e).shape[1]
                    for e in range(model.composite.n_blocks)
                ),
            ),
            np.float32,
        )
    cg = torch.from_numpy(cols.astype(np.int64))
    t0 = time.time()
    max_in_diff = 0.0
    for lo in range(0, n, batch_size):
        p = pos[lo : lo + batch_size]
        batch, tg = make_batch(ck.view, p.tolist())
        res = model(batch, tg.gene_rows, tg.record_rows)
        if len(tg.gene_rows) != len(p):
            raise ValueError("an isolate record is not gene level")
        rc = batch.record_context
        x, trunk, h_in, h_out = cap["x"], cap["trunk"], cap["h_in"], cap["h_out"]
        max_in_diff = max(max_in_diff, float((h_in - trunk[rc]).abs().max()))
        rows = slice(lo, lo + len(p))
        out["input_mean"][rows] = x[rc].mean(1).numpy()
        out["trunk_mean"][rows] = trunk[rc, 1:].mean(1).numpy()
        out["trunk_cls"][rows] = trunk[rc, 0].numpy()
        out["op_cls"][rows] = h_out[:, 0].numpy()
        out["op_mean"][rows] = h_out[:, 1:].mean(1).numpy()
        out["gene_input"][rows] = x[rc][:, cg].numpy().astype(np.float16)
        out["gene_trunk"][rows] = trunk[rc][:, cg + 1].numpy().astype(np.float16)
        out["gene_op"][rows] = h_out[:, cg + 1].numpy().astype(np.float16)
        pt = ck.objective.point(res.gene).float().numpy()
        out["head"][lo + tg.gene_rows.numpy()] = pt
        if want_frozen:
            out["frozen_ctx_mean"][rows] = _ctx_frozen_mean(
                model, batch.contexts.key_rows
            )[rc].numpy()
        if (lo // batch_size) % 10 == 0:
            print(f"  {lo + len(p)}/{n} records, {time.time() - t0:.0f} s", flush=True)
    for h in handles:
        h.remove()
    if max_in_diff > 0:
        raise ValueError(
            f"operator input differs from the trunk output by {max_in_diff}"
        )
    if ck.view.target_map is not None:
        raise ValueError(
            "D2 scores the head in legacy z; this checkpoint has a target map"
        )
    return out


def _score(ck: Loaded, d: str, P_val: np.ndarray, Y_val: np.ndarray) -> dict[str, Any]:
    """``gene_level_metrics`` on the validation records, plus the genes scored."""
    m = gene_level_metrics(P_val, Y_val, ck.view.train_mean_z[d], MIN_STRAINS)
    r = column_pearson(P_val, Y_val, MIN_STRAINS)
    m["n_genes_scored"] = float(np.isfinite(r).sum())
    return m


def ridge_multi(
    F_all: np.ndarray, Ytr: np.ndarray, n_tr: int, units: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """The ``baselines_043`` ridge: predictions for rows ``n_tr:`` (validation)."""
    keep = F_all[:n_tr].std(0) > 0
    Fk = F_all[:, keep].astype(np.float64)
    m, sd = Fk[:n_tr].mean(0), Fk[:n_tr].std(0)
    Fz = (Fk - m) / sd
    Yall = np.full((len(Fk), Ytr.shape[1]), np.nan)
    Yall[:n_tr] = Ytr
    va_i = inner_split(units)
    P, best, scores = ridge_fold(
        Fz, None, Yall, np.arange(n_tr), va_i, np.arange(n_tr, len(Fk)), RIDGE_GRID
    )
    return P, {
        "log10_lambda_rel": best,
        "grid_edge": bool(best in (RIDGE_GRID[0], RIDGE_GRID[-1])),
        "inner_scores_median_per_gene_pearson": scores,
        "n_feature_columns": int(keep.sum()),
        "n_inner_heldout_records": int(len(va_i)),
    }


def ridge_per_gene(
    X: np.ndarray, Y: np.ndarray, n_tr: int, units: np.ndarray, chunk: int = 256
) -> tuple[np.ndarray, dict[str, Any]]:
    """Per gene ``g``: ridge from ``X[:, g]`` (d) to ``Y[:, g]`` over training rows with a
    finite target; one penalty for all genes, chosen on the inner split by the median
    per-gene Pearson; predictions for rows ``n_tr:``. ``X`` z-scored per gene on the
    fitting rows, lambda = (trace K / n) x 10^g = (non-constant dims) x 10^g.
    """
    va_i = inner_split(units)
    tr_i = np.arange(n_tr)
    inner = np.setdiff1d(tr_i, va_i)
    te_i = np.arange(n_tr, len(X))

    def fit_predict(
        fit: np.ndarray, out_rows: np.ndarray, grid: list[float]
    ) -> list[np.ndarray]:
        preds = [np.full((len(out_rows), X.shape[1]), np.nan) for _ in grid]
        for c0 in range(0, X.shape[1], chunk):
            c = slice(c0, min(c0 + chunk, X.shape[1]))
            Xf = torch.from_numpy(X[fit, c].astype(np.float64))  # [n, g, d]
            Xo = torch.from_numpy(X[out_rows, c].astype(np.float64))
            Yf = torch.from_numpy(Y[fit, c].astype(np.float64))
            msk = torch.isfinite(Yf).to(torch.float64)
            cnt = msk.sum(0)
            ok = cnt >= MIN_FIT_PER_GENE
            cs = cnt.clamp_min(1.0)
            xm = torch.einsum("ngd,ng->gd", Xf, msk) / cs[:, None]
            Xc = (Xf - xm) * msk[..., None]
            xs = torch.sqrt(torch.einsum("ngd,ngd->gd", Xc, Xc) / cs[:, None])
            nz = (xs > 1e-6).to(torch.float64)
            xs = torch.where(xs > 1e-6, xs, torch.ones_like(xs))
            Xs = Xc / xs * nz
            ym = torch.nan_to_num(Yf).mul(msk).sum(0) / cs
            yc = (torch.nan_to_num(Yf) - ym) * msk
            gram = torch.einsum("ngd,nge->gde", Xs, Xs)
            b = torch.einsum("ngd,ng->gd", Xs, yc)
            w, V = torch.linalg.eigh(gram)
            Vb = torch.einsum("gde,gd->ge", V, b)
            Xso = (Xo - xm) / xs * nz
            scale = nz.sum(-1).clamp_min(1.0)
            for j, g in enumerate(grid):
                lam = scale * 10.0**g
                beta = torch.einsum(
                    "gde,ge->gd", V, Vb / (w.clamp_min(0) + lam[:, None])
                )
                p = torch.einsum("ngd,gd->ng", Xso, beta) + ym
                p[:, ~ok] = float("nan")
                preds[j][:, c] = p.numpy()
        return preds

    inner_preds = fit_predict(inner, va_i, RIDGE_GRID)
    scores = {
        str(g): _nanmedian(column_pearson(p, Y[va_i], MIN_STRAINS))
        for g, p in zip(RIDGE_GRID, inner_preds, strict=True)
    }
    best = max(RIDGE_GRID, key=lambda g: (np.nan_to_num(scores[str(g)], nan=-9), -g))
    P = fit_predict(tr_i, te_i, [best])[0]
    return P, {
        "log10_lambda_rel": best,
        "grid_edge": bool(best in (RIDGE_GRID[0], RIDGE_GRID[-1])),
        "inner_scores_median_per_gene_pearson": scores,
        "n_inner_heldout_records": int(len(va_i)),
        "min_finite_training_records_per_gene": MIN_FIT_PER_GENE,
    }


def indicator_features(
    ck: Loaded, d: str, pos: np.ndarray, data_root: str
) -> np.ndarray:
    """``baselines_043.run_gene_dataset`` isolate features: per-gene calls (variant, loss
    of function, absence) from the linear-evals cache, 3 x 6,607 columns.
    """
    gen = store_paths(STORE_VERSION)
    cache = osp.join(
        data_root, "data/torchcell/experiments", EXPERIMENT, gen.linear_evals_cache
    )
    with open(osp.join(cache, f"{RAW_NAMES[d]}.pkl"), "rb") as fh:
        rows = {int(r["raw_index"]): r for r in pickle.load(fh)}
    s, v = ck.store, ck.view
    G = s.Y.shape[1]
    Z = np.zeros((len(pos), 3 * G))
    for i, p in enumerate(pos):
        gi, pt = rows[int(s.rec_raw_index[v.rec[p]])]["marked"]
        for t, off in ((1, 0), (2, G), (3, 2 * G)):
            Z[i, off + gi[pt == t]] = 1.0
    return Z


def stage_d2(
    data_root: str, payload: dict[str, Any], batch_size: int, smoke: bool
) -> None:
    res: dict[str, Any] = {"runs": {}}
    frozen_done: dict[str, dict[str, Any]] = {}
    for label in D2_RUNS if not smoke else D2_RUNS[:1]:
        path = osp.join(data_root, CKPT_ROOT, RUNS[label], "best.pt")
        ck = Loaded(data_root, path)
        v, s = ck.view, ck.store
        run_out: dict[str, Any] = {"meta": ck.meta, "datasets": {}}
        for d in ISOLATES:
            tr = v.positions(ROLE_CODE[Role.TRAIN], d)
            val = v.positions(ROLE_CODE[Role.VAL], d)
            for seed in SEEDS:
                sub = eval_positions(v, ROLE_CODE[Role.VAL], VAL_CAP, seed)
                sub = sub[v.local_dataset[sub] == v.datasets.index(d)]
                if not np.array_equal(sub, val):
                    raise ValueError(
                        f"{d}: seed {seed} validation subset is not every record"
                    )
            if smoke:
                tr, val = tr[:60], val[:20]
            pos = np.concatenate([tr, val])
            n_tr = len(tr)
            Ytr = v.Z[s.rec_yrow[v.rec[tr]]].astype(np.float64)
            Yval = v.Z[s.rec_yrow[v.rec[val]]].astype(np.float64)
            cols = np.nonzero(np.isfinite(Ytr).any(0))[0]
            units = s.rec_unit[v.rec[tr]]
            G = Ytr.shape[1]

            def expand(Pc: np.ndarray) -> np.ndarray:
                P = np.full((Pc.shape[0], G), np.nan)
                P[:, cols] = Pc
                return P

            print(
                f"[d2] {label} {d}: {n_tr} train + {len(val)} val records, {len(cols)} target genes",
                flush=True,
            )
            t0 = time.time()
            reps = extract(ck, pos, cols, batch_size, want_frozen=d not in frozen_done)
            ex_s = time.time() - t0
            dres: dict[str, Any] = {
                "n_train_records": int(n_tr),
                "n_val_records": int(len(val)),
                "n_target_genes": int(len(cols)),
                "extract_seconds": round(ex_s, 1),
                "representations": {},
            }
            R = dres["representations"]
            R["head_output"] = {
                **_score(ck, d, reps["head"][n_tr:].astype(np.float64), Yval),
                "fit": None,
            }
            if d not in frozen_done:
                fz: dict[str, Any] = {}
                Fi = indicator_features(ck, d, pos, data_root)
                P, info = ridge_multi(Fi, Ytr[:, cols], n_tr, units)
                fz["frozen_indicators"] = {
                    **_score(ck, d, expand(P), Yval),
                    "fit": info,
                }
                P, info = ridge_multi(
                    reps["frozen_ctx_mean"], Ytr[:, cols], n_tr, units
                )
                fz["frozen_ctx_mean"] = {**_score(ck, d, expand(P), Yval), "fit": info}
                frozen_done[d] = fz
            for k in CTX_REPS:
                P, info = ridge_multi(reps[k], Ytr[:, cols], n_tr, units)
                R[k] = {**_score(ck, d, expand(P), Yval), "fit": info}
                print(
                    f"  {k}: {R[k]['pearson_per_gene_median']:.4f} (g={info['log10_lambda_rel']})",
                    flush=True,
                )
            Ycols = np.concatenate([Ytr[:, cols], Yval[:, cols]])
            for k in GENE_REPS:
                t1 = time.time()
                P, info = ridge_per_gene(reps[f"gene_{k}"], Ycols, n_tr, units)
                R[f"per_gene_{k}"] = {
                    **_score(ck, d, expand(P), Yval),
                    "fit": {**info, "seconds": round(time.time() - t1, 1)},
                }
                print(
                    f"  per_gene_{k}: {R[f'per_gene_{k}']['pearson_per_gene_median']:.4f} (g={info['log10_lambda_rel']})",
                    flush=True,
                )
            print(
                f"  head: {R['head_output']['pearson_per_gene_median']:.4f}; frozen indicators "
                f"{frozen_done[d]['frozen_indicators']['pearson_per_gene_median']:.4f}, frozen ctx mean "
                f"{frozen_done[d]['frozen_ctx_mean']['pearson_per_gene_median']:.4f}",
                flush=True,
            )
            run_out["datasets"][d] = dres
            del reps
        res["runs"][label] = run_out
    res["frozen"] = frozen_done
    res["statistic"] = (
        "val/<ds>/pearson_per_gene_median: median over genes of the Pearson r across the "
        "validation records (gene_level_metrics, genes with >= 3 finite pairs); "
        "pearson_per_strain_centered_median also reported"
    )
    payload["d2"] = res


# ============================================================================ wandb
def stage_wandb(payload: dict[str, Any]) -> None:
    """Logged values at each D1 checkpoint's epoch (the trainer's own evaluation)."""
    import wandb

    api = wandb.Api(timeout=120)
    keys = [
        "val/doubles/pearson_per_strain_centered_median",
        "val/doubles/additive_model_pearson_per_strain_centered_median",
        "val/doubles/additive_measured_pearson_per_strain_centered_median",
        "val/caudal/pearson_per_gene_median",
        "val/muenzner/pearson_per_gene_median",
    ]
    hist: dict[str, dict[int, dict[str, float]]] = {}
    for label, rel in RUNS.items():
        rid = osp.basename(rel)
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{rid}")
        rows = run.scan_history(keys=["epoch", *keys], page_size=5000)
        hist[label] = {
            int(r["epoch"]): {k: r.get(k) for k in keys}
            for r in rows
            if r.get("epoch") is not None and r.get(keys[0]) is not None
        }
    for key, c in payload["d1"]["checkpoints"].items():
        if "absent" in c:
            continue
        label = key.split("/")[0]
        rid = osp.basename(RUNS[label])
        c["wandb_logged_at_epoch"] = hist[label].get(c["epoch"])
        c["wandb_run_url"] = (
            f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{rid}"
        )
    for label, r in payload.get("d2", {}).get("runs", {}).items():
        r["wandb_logged_at_epoch"] = hist[label].get(r["meta"]["epoch"])


# ============================================================================ report
D1_STATS = (
    ("model_double_vs_measured_double", "model double vs measured double"),
    ("model_singles_sum_vs_measured_double", "sum of model singles vs measured double"),
    ("model_double_vs_model_singles_sum", "model double vs sum of model singles"),
    ("residual_model_vs_measured_per_double", "raw residual vs measured residual"),
    ("orth_residual_model_vs_measured", "orthogonalized residual vs measured"),
    ("model_single_vs_measured_single", "model single vs measured single (training)"),
)
D2_ROWS = (
    ("head_output", "model head output"),
    ("frozen_indicators", "frozen per-gene calls (baseline ridge)"),
    ("frozen_ctx_mean", "frozen composite, mean over genes"),
    ("input_mean", "input tokens, mean over genes"),
    ("trunk_mean", "trunk, mean over genes"),
    ("trunk_cls", "trunk, CLS"),
    ("op_cls", "operator, CLS"),
    ("op_mean", "operator, mean over genes"),
    ("per_gene_input", "per gene: own input token"),
    ("per_gene_trunk", "per gene: own trunk token"),
    ("per_gene_op", "per gene: own post-operator token"),
)
DS_LABEL = {"caudal": "Caudal 2024", "muenzner": "Muenzner 2024 protein"}


def _d2_values(payload: dict[str, Any], d: str, k: str, stat: str) -> list[float]:
    d2 = payload["d2"]
    if k.startswith("frozen_"):
        return [float(d2["frozen"][d][k][stat])]
    return [
        float(r["datasets"][d]["representations"][k][stat]) for r in d2["runs"].values()
    ]


def stage_report(payload: dict[str, Any]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    from torchcell.timestamp import timestamp
    from torchcell.utils.utils import (
        PANEL_WIDTHS_MM,
        PLOT_PALETTE,
        mm_to_in,
        savefig_true_size_svg,
    )

    ck = payload["d1"]["checkpoints"]
    # ---- D1 table
    hdr = (
        "| checkpoint | epoch | model double vs measured double, median centered r | "
        "sum of model singles vs measured double, median centered r | model double vs sum of "
        "model singles, median centered r | measured singles sum vs measured double, median "
        "centered r | model single vs measured single (144), median centered r | residual "
        "model vs measured, median per-double r | control: minus model singles sum vs "
        "measured residual, median per-double r | orthogonalized residual model vs "
        "measured, median per-double r | model double vs measured double beyond measured "
        "singles, median per-double r | sum of model singles vs the same, median per-double "
        "r | residual, median per-gene r | slope of model double on model singles sum, "
        "median | slope of measured double on measured singles sum, median | norm ratio "
        "model double to model singles sum, median | norm ratio measured double to measured "
        "singles sum, median | logged val/doubles at this epoch, r |"
    )
    lines = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for key, c in ck.items():
        if "absent" in c:
            continue
        lg = (c.get("wandb_logged_at_epoch") or {}).get(
            "val/doubles/pearson_per_strain_centered_median"
        )
        lines.append(
            f"| {key} | {c['epoch']} | {c['model_double_vs_measured_double']['median']:.3f} | "
            f"{c['model_singles_sum_vs_measured_double']['median']:.3f} | "
            f"{c['model_double_vs_model_singles_sum']['median']:.3f} | "
            f"{c['measured_singles_sum_vs_measured_double']['median']:.4f} | "
            f"{c['model_single_vs_measured_single']['median']:.3f} | "
            f"{c['residual_model_vs_measured_per_double']['median']:.3f} | "
            f"{c['control_minus_model_singles_sum_vs_measured_residual']['median']:.3f} | "
            f"{c['orth_residual_model_vs_measured']['median']:.3f} | "
            f"{c['model_double_vs_measured_double_beyond_measured_singles']['median']:.3f} | "
            f"{c['model_singles_sum_vs_measured_double_beyond_measured_singles']['median']:.3f} | "
            f"{c['residual_model_vs_measured_per_gene']['median']:.3f} | "
            f"{c['slope_model_double_on_model_singles_sum']['median']:.3f} | "
            f"{c['slope_measured_double_on_measured_singles_sum']['median']:.3f} | "
            f"{c['norm_ratio_model_double_to_model_singles_sum']['median']:.3f} | "
            f"{c['norm_ratio_measured_double_to_measured_singles_sum']['median']:.3f} | "
            + ("--" if lg is None else f"{lg:.3f}")
            + " |"
        )
    t1 = "\n".join(lines)
    # ---- D2 table
    hdr2 = (
        "| representation | Caudal 2024 val median per-gene Pearson r, mean (sd) of 3 seeds | "
        "Caudal genes scored, n | Muenzner 2024 protein val median per-gene Pearson r, mean "
        "(sd) of 3 seeds | Muenzner genes scored, n |"
    )
    lines2 = [hdr2, "|---|---|---|---|---|"]
    for k, lab in D2_ROWS:
        cells = []
        for d in ISOLATES:
            x = _d2_values(payload, d, k, "pearson_per_gene_median")
            ng = _d2_values(payload, d, k, "n_genes_scored")
            cells.append(
                f"{x[0]:.3f} (deterministic)"
                if len(x) == 1
                else f"{np.mean(x):.3f} ({np.std(x, ddof=1):.3f})"
            )
            cells.append(f"{int(np.mean(ng)):,}")
        lines2.append(f"| {lab} | " + " | ".join(cells) + " |")
    t2 = "\n".join(lines2)
    print(t1)
    print()
    print(t2)
    payload["report_tables_markdown"] = {"d1": t1, "d2": t2}
    # ---- figure
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }
    )
    fig, (a1, a2) = plt.subplots(
        1,
        2,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(88)),
        gridspec_kw={"width_ratios": [1.0, 1.25]},
    )
    # a: D1, best.pt of each run
    runs = [k for k in ck if k.endswith("/best.pt") and "absent" not in ck[k]]
    w = 0.8 / len(D1_STATS)
    for si, (st, lab) in enumerate(D1_STATS):
        for ri, k in enumerate(runs):
            a1.bar(
                ri - 0.4 + w * (si + 0.5),
                ck[k][st]["median"],
                w,
                color=PLOT_PALETTE[si],
                edgecolor="black",
                linewidth=0.3,
                label=lab if ri == 0 else None,
                zorder=3,
            )
    add = ck[runs[0]]["measured_singles_sum_vs_measured_double"]["median"]
    a1.axhline(
        add,
        color="black",
        linestyle="--",
        linewidth=0.6,
        zorder=4,
        label=f"sum of measured singles ({add:.3f})",
    )
    a1.set_xticks(range(len(runs)))
    a1.set_xticklabels(
        [
            k.split("/")[0]
            .replace("bs16_s", "bs16\nseed ")
            .replace("injop_s", "inject op\nseed ")
            + f"\nep {ck[k]['epoch']}"
            for k in runs
        ]
    )
    a1.set_ylabel("median over 72 doubles of per-double Pearson r over genes")
    a1.set_title(
        "Sameith doubles: does the operator compose two marks?", loc="left", pad=40
    )
    a1.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=2,
        handlelength=1.0,
        columnspacing=0.8,
        borderaxespad=0.3,
    )
    a1.set_ylim(-0.2, 1.0)
    # b: D2, 3-seed mean +- sd
    rows = list(D2_ROWS)
    y = np.arange(len(rows))[::-1]
    h = 0.38
    for di, d in enumerate(ISOLATES):
        for yi, (k, _) in zip(y, rows, strict=True):
            x = _d2_values(payload, d, k, "pearson_per_gene_median")
            a2.barh(
                yi + (h / 2 if di == 0 else -h / 2),
                float(np.mean(x)),
                h,
                xerr=None if len(x) == 1 else float(np.std(x, ddof=1)),
                color=PLOT_PALETTE[di],
                edgecolor="black",
                linewidth=0.3,
                error_kw={"elinewidth": 0.5, "capsize": 1.0, "capthick": 0.5},
                label=DS_LABEL[d] if yi == y[0] else None,
                zorder=3,
            )
    a2.set_yticks(y)
    a2.set_yticklabels([lab for _, lab in rows])
    a2.set_xlim(-0.1, 0.7)
    a2.set_xlabel(
        "validation median per-gene Pearson r (ridge fit on training isolates;\nbars = mean of bs16 seeds 0-2, whiskers = sd; frozen rows deterministic)"
    )
    a2.set_title("Isolates: where is the genotype signal lost?", loc="left", pad=40)
    a2.legend(frameon=False, loc="lower right", handlelength=1.0)
    for ax, axis in ((a1, "y"), (a2, "x")):
        loc = ax.yaxis if axis == "y" else ax.xaxis
        loc.set_major_locator(MultipleLocator(0.2))
        loc.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(which="minor", length=0)
        ax.grid(axis=axis, which="both", linewidth=0.3, color="#DDDDDD", zorder=0)
        (ax.axhline if axis == "y" else ax.axvline)(
            0, color="black", linewidth=0.5, zorder=1
        )
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.5)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.80, bottom=0.15, wspace=0.62)
    for x0, letter in ((0.008, "a"), (0.385, "b")):
        fig.text(x0, 0.985, letter, fontsize=8, fontweight="bold", va="top")
    d_out = osp.join(os.environ["ASSET_IMAGES_DIR"], EXPERIMENT)
    os.makedirs(d_out, exist_ok=True)
    base = osp.join(d_out, f"diagnostics_043_{timestamp()}")
    fig.savefig(base + ".png", dpi=300)
    savefig_true_size_svg(fig, base + ".svg")
    plt.close(fig)
    payload["figure"] = base + ".svg"
    print(base + ".svg")
    return base + ".svg"


# ============================================================================ main
def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=SCRIPTS,
    ).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True, choices=["d1", "d2", "wandb", "report"])
    ap.add_argument("--batch-size", type=int, default=8, help="D2 records per forward")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="D2 on 60 + 20 records, one checkpoint, json to $SMOKE_JSON",
    )
    args = ap.parse_args()
    load_dotenv()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    exp_dir = osp.join(os.environ["EXPERIMENT_ROOT"], EXPERIMENT)
    out_path = (
        os.environ["SMOKE_JSON"]
        if args.smoke
        else osp.join(exp_dir, "results", "diagnostics_043.json")
    )
    payload: dict[str, Any] = {}
    if osp.exists(out_path):
        with open(out_path) as f:
            payload = json.load(f)
    payload.update(
        {
            "generated_by": f"experiments/{EXPERIMENT}/scripts/diagnostics_043.py",
            "store_version": STORE_VERSION,
        }
    )
    stamp = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "torchcell_commit": _git_commit(),
    }
    payload.setdefault("stages", {})[args.stage] = stamp
    data_root = os.environ["DATA_ROOT"]
    if args.stage == "d1":
        stage_d1(data_root, payload)
    elif args.stage == "d2":
        stage_d2(data_root, payload, args.batch_size, args.smoke)
    elif args.stage == "wandb":
        stage_wandb(payload)
    else:
        stage_report(payload)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=1, allow_nan=True)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
