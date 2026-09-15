# experiments/027-betaxanthin-metabolic/scripts/analyze_projection.py
# [[experiments.027-betaxanthin-metabolic.projection-head]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/analyze_projection.py

r"""Read the wave-6 projection cells into one table, one figure, and one sampler audit.

Every number the projection note reports comes from here. Three passes, each optional so a
partial night still produces the part it has evidence for:

1. **The curve.** Per cell, from its slurm log, the escape epoch under the wave-5 rule
   (three consecutive epochs with ``val_pred_sd_ratio`` above 0.05), whether the escape was
   SUSTAINED to the end of the run, and the constraint scale and loss at the escape. A run
   that escaped and then went back to a constant is the failure mode job 1678 showed, and
   the sustained flag is what separates it from an escape that held.

2. **The endpoint.** From the result file, the validation-selected epoch, the pinned test
   Spearman on the 629 Merzbacher genes, and, when the trainer's audit landed, the whole
   ``feasibility`` block, reported at the top-100 group because that is the group the
   Genesis metabolic-grounding target is defined on.

3. **The sampler.** When ``$DATA_ROOT/checkpoints-027/<stem>.pt`` exists, the model is
   rebuilt with ``build_model_for_cell``, the saved weights are copied in exactly as
   ``audit_top_strain_feasibility.py`` does it, the flux layer's sampler is switched on, and
   :meth:`FluxLayer.sample` draws ``--draws`` fluxes for the top ``--draw-genotypes``
   predicted strains. Reported: the fraction of DRAWS whose balance ratio median is at or
   below ``--balance-tol`` and whose uphill fraction is exactly zero, and the fraction of
   GENOTYPES for which EVERY draw meets that. The two are different claims and both are in
   the JSON.

Tolerances are written into the output rather than left in this file's source, because a
feasibility fraction without its tolerance is not a number.

Run from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/analyze_projection.py

Writes ``results/wave6_projection_summary.json`` and a figure under
``$ASSET_IMAGES_DIR/027-betaxanthin-metabolic/``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from torchcell.metabolism.constraints import ThermoMode  # noqa: E402
from torchcell.models.cell_graph_transformer_metabolism import (  # noqa: E402
    perturbed_gene_pool,
)
from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import PLOT_PALETTE, mm_to_in  # noqa: E402

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
DATA_ROOT = os.environ["DATA_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXP_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic")
SLURM_OUT = osp.join(EXP_DIR, "slurm", "output")
RESULTS = osp.join(EXP_DIR, "results")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "027-betaxanthin-metabolic")

#: The wave-5 escape rule, reused verbatim so wave 6 is comparable to wave 5.
ESCAPE_SD_RATIO = 0.05
ESCAPE_CONSECUTIVE_EPOCHS = 3

#: The paired comparators, at the seeds the wave-6 cells run. Pooled is the no-module
#: control and `nocstr` is the BOX-parameterized flux head at a constraint scale of zero;
#: both are test Spearman on the same 629 pinned genes at the validation-selected epoch.
BASELINES: dict[int, dict[str, Any]] = {
    2700: {
        "pooled": 0.0717,
        "pooled_job": "1674",
        "nocstr": 0.0747,
        "nocstr_job": "1673",
    },
    2701: {
        "pooled": 0.1058,
        "pooled_job": "1680",
        "nocstr": 0.1185,
        "nocstr_job": "1679",
    },
}

EPOCH_RE = re.compile(
    r"\[(?P<arm>\S+) s(?P<seed>\d+)\] ep\s+(?P<epoch>\d+) loss (?P<loss>\S+) "
    r"data (?P<data>\S+) cstr (?P<cstr>\S+) scale (?P<scale>\S+) "
    r"val_loss (?P<val_loss>\S+) val_r (?P<val_r>\S+) sd_ratio (?P<sd>\S+)"
)


class EpochRow(BaseModel):
    """One epoch as the log printed it."""

    model_config = ConfigDict(extra="forbid")

    epoch: int
    train_loss: float
    train_data_loss: float
    train_constraint_loss: float
    constraint_scale: float
    val_loss: float
    val_pearson: float
    val_pred_sd_ratio: float


class DrawAudit(BaseModel):
    """The sampler read on the top-ranked genotypes, with its tolerances attached.

    ``thermo_available`` is False for an arm whose ``thermo_mode`` is OFF, and then every
    thermodynamic field is **null rather than 1.0**. This is not fastidiousness: the first
    run of this audit, on job 1689, reported ``frac_draws_downhill = 1.0`` for an arm that
    has no thermodynamic model at all, because :meth:`FluxLayer.feasibility` zero-fills a
    term that does not apply and a zero violation fraction reads as perfect compliance. An
    arm that makes no thermodynamic claim must not be scored as satisfying one.
    """

    model_config = ConfigDict(extra="forbid")

    n_genotypes: int
    n_draws: int
    thermo_available: bool
    balance_tolerance: float
    uphill_tolerance: float
    box_tolerance: float
    projection_steps: int
    projection_end: str
    thermo_box: bool
    frac_draws_balanced: float
    frac_draws_downhill: float | None
    frac_draws_feasible: float | None
    frac_genotypes_all_draws_feasible: float | None
    frac_genotypes_any_draw_feasible: float | None
    balance_ratio_median_p50: float
    balance_ratio_median_p90: float
    box_violation_frac_mean: float
    uphill_frac_mean: float | None


class BoxViolationBreakdown(BaseModel):
    r"""What the box-violation FRACTION is actually made of, on the top-ranked strains.

    A fraction of 0.61 can mean two completely different things and the Genesis criterion
    cares which. It can be 61 percent of reactions overshooting a real bound by a real
    amount, or it can be 61 percent of reactions sitting a hair outside a box that has
    **collapsed to the single point zero** because their gene is deleted or their enzyme
    capacity is zero, where the balance projection then moves them off zero by an epsilon
    and every epsilon counts as a violation at a tolerance of 1e-4.

    The split is ``zero_width``: reactions with ``ub - lb <= 1e-12``, which carry no
    information about whether a bound is being respected, only about how exactly the last
    projection hit zero. ``positive_width`` is the genuine overshoot.
    """

    model_config = ConfigDict(extra="forbid")

    n_genotypes: int
    n_reactions: int
    tolerance: float
    flux_scale: float
    frac_reactions_violating: float
    frac_reactions_zero_width_box: float
    frac_violations_from_zero_width_box: float
    frac_reactions_violating_positive_width: float
    overshoot_over_flux_scale_median: float
    overshoot_over_flux_scale_p90: float
    overshoot_over_flux_scale_max: float
    overshoot_over_box_width_median: float
    overshoot_over_box_width_p90: float
    frac_violations_under_1e_3_of_flux_scale: float
    note: str


#: The three cytosolic species the Cachera cassette's route runs through, from 026's
#: BETAXANTHIN_PRECURSORS: tyrosine, and the two aromatic-pathway intermediates its
#: feedback-resistant ARO4 and ARO7 alleles act on.
BETAXANTHIN_PRECURSOR_METABOLITES = ["s_1051", "s_0188", "s_1032"]


class FluxDiagnosis(BaseModel):
    r"""Why a feasible head might carry no phenotype signal.

    The hypothesis this is built to test, stated as one: the learned sign pattern plus mass
    balance admits only near-zero flux, so ``v`` is nearly the same vector for every
    genotype and the flux-scalar head reads nothing. Three things would have to be true
    together, and each has its own field: the flux is small, most of it is exactly zero, and
    what is left barely moves between strains.

    ``v_spread_over_scale`` is the decisive one. It is the standard deviation of each
    reaction's flux ACROSS genotypes, averaged over reactions and divided by the mean
    absolute flux, so it asks how much of the flux vector is genotype-specific rather than
    shared. A value near zero says the head is reading a constant.
    """

    model_config = ConfigDict(extra="forbid")

    n_genotypes: int
    n_reactions: int
    parameterization_raw: str
    thermo_box: bool
    abs_v_median: float
    abs_v_mean: float
    frac_reactions_below_1e_6: float
    v_spread_across_genotypes_mean: float
    v_spread_over_scale: float
    n_reactions_identical_across_genotypes: int
    frac_boxes_collapsed_to_zero: float
    n_betaxanthin_route_reactions: int
    n_betaxanthin_route_collapsed: int
    frac_betaxanthin_route_collapsed: float
    projection_residual_trace: list[float]
    projection_box_gap_trace: list[float]
    trace_note: str


class CellSummary(BaseModel):
    """One wave-6 projection cell, from whatever evidence exists for it."""

    model_config = ConfigDict(extra="forbid")

    cell: str
    arm: str | None = None
    seed: int | None = None
    job_id: str | None = None
    log_path: str | None = None
    result_path: str | None = None
    constraint_scale: float | None = None
    epochs_completed: int = 0
    finished: bool = False
    escape_epoch: int | None = None
    escape_sustained: bool | None = None
    constraint_scale_at_escape: float | None = None
    constraint_loss_at_escape: float | None = None
    data_loss_at_escape: float | None = None
    val_pred_sd_ratio_last: float | None = None
    val_pred_sd_ratio_max: float | None = None
    val_pearson_last: float | None = None
    select_epoch: int | None = None
    test_spearman_pinned: float | None = None
    test_pearson_pinned: float | None = None
    n_test_pinned: int | None = None
    delta_vs_pooled: float | None = None
    delta_vs_box_nocstr: float | None = None
    projection: dict[str, Any] | None = None
    kcat_tier_counts: dict[str, int] | None = None
    kcat_table_label: str | None = None
    feas_last_epoch: dict[str, float] = Field(default_factory=dict)
    feasibility_top100: dict[str, float] | None = None
    feasibility_test: dict[str, float] | None = None
    draw_audit: DrawAudit | None = None
    box_breakdown: BoxViolationBreakdown | None = None
    flux_diagnosis: FluxDiagnosis | None = None
    constraint_terms_at_select: dict[str, float] = Field(default_factory=dict)
    sd_ratio_curve: list[float] = Field(default_factory=list)


class ProjectionSummary(BaseModel):
    """The whole wave-6 read."""

    model_config = ConfigDict(extra="forbid")

    escape_rule: str
    scoring_note: str
    baselines: dict[str, dict[str, Any]]
    cells: list[CellSummary]
    figure_path: str | None = None


def parse_log(path: str) -> list[EpochRow]:
    rows: list[EpochRow] = []
    with open(path, errors="replace") as handle:
        for line in handle:
            match = EPOCH_RE.search(line)
            if match is None:
                continue
            rows.append(
                EpochRow(
                    epoch=int(match["epoch"]),
                    train_loss=float(match["loss"]),
                    train_data_loss=float(match["data"]),
                    train_constraint_loss=float(match["cstr"]),
                    constraint_scale=float(match["scale"]),
                    val_loss=float(match["val_loss"]),
                    val_pearson=float(match["val_r"]),
                    val_pred_sd_ratio=float(match["sd"]),
                )
            )
    return rows


def newest_log(job_name: str) -> tuple[str | None, str | None]:
    """The most recent slurm log for a job name, and the job id in its filename."""
    matches = sorted(glob.glob(osp.join(SLURM_OUT, f"{job_name}_*.out")))
    if not matches:
        return None, None
    path = matches[-1]
    return path, osp.splitext(osp.basename(path))[0].rsplit("_", 1)[-1]


def read_curve(summary: CellSummary, rows: list[EpochRow]) -> None:
    """Fill in the escape read from the per-epoch log rows."""
    summary.epochs_completed = len(rows)
    if not rows:
        return
    summary.sd_ratio_curve = [r.val_pred_sd_ratio for r in rows]
    summary.val_pred_sd_ratio_last = rows[-1].val_pred_sd_ratio
    summary.val_pred_sd_ratio_max = max(r.val_pred_sd_ratio for r in rows)
    summary.val_pearson_last = rows[-1].val_pearson
    need = ESCAPE_CONSECUTIVE_EPOCHS
    for i in range(len(rows) - need + 1):
        window = rows[i : i + need]
        if all(r.val_pred_sd_ratio > ESCAPE_SD_RATIO for r in window):
            summary.escape_epoch = window[0].epoch
            summary.constraint_scale_at_escape = window[0].constraint_scale
            summary.constraint_loss_at_escape = window[0].train_constraint_loss
            summary.data_loss_at_escape = window[0].train_data_loss
            break
    if summary.escape_epoch is not None:
        # SUSTAINED, which is the question the everything-on cells are asking: job 1678
        # escaped at epoch 41 and then collapsed back to a constant. The last three epochs
        # of the run are the cheapest statement of whether the escape held.
        tail = rows[-need:]
        summary.escape_sustained = all(
            r.val_pred_sd_ratio > ESCAPE_SD_RATIO for r in tail
        )


def read_result(summary: CellSummary, payload: dict[str, Any]) -> None:
    """Fill in the endpoint and the trainer's feasibility block from the result file."""
    summary.finished = True
    summary.arm = payload["cell"]["arm"]
    summary.seed = payload["cell"]["seed"]
    summary.constraint_scale = payload["cell"].get("constraint_scale")
    summary.select_epoch = payload["select_epoch"]
    pinned = payload["test"]["pinned"]
    summary.test_spearman_pinned = pinned["spearman"]
    summary.test_pearson_pinned = pinned["pearson"]
    summary.n_test_pinned = pinned["n"]
    base = BASELINES.get(int(summary.seed or -1))
    if base is not None and summary.test_spearman_pinned is not None:
        summary.delta_vs_pooled = summary.test_spearman_pinned - base["pooled"]
        summary.delta_vs_box_nocstr = summary.test_spearman_pinned - base["nocstr"]
    coverage = (payload.get("flux_meta") or {}).get("coverage") or {}
    summary.projection = {
        k: coverage.get(k)
        for k in (
            "parameterization",
            "parameterization_raw",
            "projection_steps",
            "projection_end",
            "projection_method",
            "thermo_box",
            "thermo_box_margin_kj",
            "n_latent",
            "null_basis_orthonormality_error",
            "null_basis_balance_error",
            "latent_stochastic",
            "kl_weight",
            "thermo_uncertainty",
        )
    }
    # THE KINETIC TABLE VERSION, PER CELL AND NEVER ASSUMED. The kcat tables were revised at
    # 00:30 on 2026-09-11 and a homolog tier was added, so two cells of the same arm run
    # hours apart can differ in the capacity bound as well as in the thing being varied.
    # The label is built from the counts the run itself recorded rather than from a version
    # name, so a difference between two cells is visible without knowing when they ran, and
    # a scale contrast across two labels is a confounded contrast.
    meta = payload.get("flux_meta") or {}
    counts = meta.get("kcat_tier_counts") or coverage.get("kcat_tier_counts")
    if counts is not None:
        summary.kcat_tier_counts = counts
        summary.kcat_table_label = "/".join(
            f"{k[0]}{v}" for k, v in sorted(counts.items()) if v
        )
    history = payload.get("history") or []
    # WHICH TERM IS ACTUALLY NONZERO at the epoch the cell was scored on. A constraint sum
    # of 0.04 says nothing about whether it is parsimony, the budget, the dissipation hinge
    # or the thermodynamic prior doing the pulling.
    sel = summary.select_epoch
    if history and sel is not None and sel < len(history):
        summary.constraint_terms_at_select = {
            k: float(v)
            for k, v in history[sel].items()
            if k.startswith("c_") and isinstance(v, (int, float))
        }
    if history:
        summary.feas_last_epoch = {
            k: float(v)
            for k, v in history[-1].items()
            if k.startswith(("feas_", "c_")) and isinstance(v, (int, float))
        }
    block = payload.get("feasibility")
    if block is not None and block.get("available"):
        groups = block.get("groups", {})
        for name, payload_group in groups.items():
            if "top" in name and "100" in name:
                summary.feasibility_top100 = payload_group.get("fractions")
            elif name in {"test_pinned", "test"}:
                summary.feasibility_test = payload_group.get("fractions")


def _box_breakdown(
    parts: list[dict[str, np.ndarray]], tolerance: float, flux_scale: float
) -> BoxViolationBreakdown:
    """Assemble :class:`BoxViolationBreakdown` from per-batch (v, lb, ub) slices."""
    v = np.concatenate([p["v"] for p in parts], axis=0)
    lb = np.concatenate([p["lb"] for p in parts], axis=0)
    ub = np.concatenate([p["ub"] for p in parts], axis=0)
    width = ub - lb
    zero_width = width <= 1e-12
    overshoot = np.maximum(v - ub, 0.0) + np.maximum(lb - v, 0.0)
    violating = overshoot > tolerance
    n_viol = int(violating.sum())
    over_scale = overshoot[violating] / flux_scale
    positive = violating & ~zero_width
    over_width = (
        overshoot[positive] / width[positive]
        if int(positive.sum())
        else np.zeros(1, dtype=float)
    )
    empty = np.zeros(1, dtype=float)
    return BoxViolationBreakdown(
        n_genotypes=int(v.shape[0]),
        n_reactions=int(v.shape[1]),
        tolerance=tolerance,
        flux_scale=flux_scale,
        frac_reactions_violating=float(violating.mean()),
        frac_reactions_zero_width_box=float(zero_width.mean()),
        frac_violations_from_zero_width_box=(
            float((violating & zero_width).sum() / n_viol) if n_viol else 0.0
        ),
        frac_reactions_violating_positive_width=float(positive.mean()),
        overshoot_over_flux_scale_median=float(
            np.median(over_scale if over_scale.size else empty)
        ),
        overshoot_over_flux_scale_p90=float(
            np.percentile(over_scale if over_scale.size else empty, 90)
        ),
        overshoot_over_flux_scale_max=float(
            (over_scale if over_scale.size else empty).max()
        ),
        overshoot_over_box_width_median=float(np.median(over_width)),
        overshoot_over_box_width_p90=float(np.percentile(over_width, 90)),
        frac_violations_under_1e_3_of_flux_scale=(
            float((over_scale < 1e-3).mean()) if over_scale.size else 0.0
        ),
        note=(
            "A reaction whose box has collapsed to the single point zero (a deleted gene, "
            "or zero enzyme capacity) is counted as violating by any epsilon the balance "
            "projection moves it off zero, which is a statement about projection precision "
            "rather than about a bound being broken. frac_violations_from_zero_width_box "
            "is how much of the headline fraction is that."
        ),
    )


def _diagnose(
    layer: Any,
    v: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    trace: dict[str, list[float]],
) -> FluxDiagnosis:
    """Assemble :class:`FluxDiagnosis` from one batch of top-ranked genotypes."""
    abs_v = np.abs(v)
    spread = v.std(axis=0)
    scale = float(abs_v.mean())
    collapsed = (ub - lb) <= 1e-12
    # Reactions that touch any betaxanthin precursor, read off the layer's own S.
    met_pos = {m: i for i, m in enumerate(layer.met_ids)}
    rows = [met_pos[m] for m in BETAXANTHIN_PRECURSOR_METABOLITES if m in met_pos]
    idx = layer.s_indices.cpu().numpy()
    route = np.unique(idx[1][np.isin(idx[0], rows)])
    route_collapsed = collapsed[:, route]
    return FluxDiagnosis(
        n_genotypes=int(v.shape[0]),
        n_reactions=int(v.shape[1]),
        parameterization_raw=str(layer.config.parameterization),
        thermo_box=bool(layer.thermo_box),
        abs_v_median=float(np.median(abs_v)),
        abs_v_mean=scale,
        frac_reactions_below_1e_6=float((abs_v < 1e-6).mean()),
        v_spread_across_genotypes_mean=float(spread.mean()),
        v_spread_over_scale=float(spread.mean() / scale) if scale > 0 else 0.0,
        n_reactions_identical_across_genotypes=int((spread < 1e-9).sum()),
        frac_boxes_collapsed_to_zero=float(collapsed.mean()),
        n_betaxanthin_route_reactions=int(route.size),
        n_betaxanthin_route_collapsed=int(route_collapsed.mean(axis=0).sum()),
        frac_betaxanthin_route_collapsed=float(route_collapsed.mean()),
        projection_residual_trace=trace["balance"],
        projection_box_gap_trace=trace["box"],
        trace_note=(
            "Per alternation k: 'balance' is max |S v| / max |v| after step k, 'box' is the "
            "mean out-of-box overshoot over flux_scale. A trace that stops falling and "
            "alternates between two levels says the tightened box and ker S meet only near "
            "the origin for the learned sign pattern, which is the degeneracy the "
            "near-zero-flux hypothesis predicts."
        ),
    )


def _trace_projection(
    layer: Any, context: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor, steps: int
) -> dict[str, list[float]]:
    """Re-run the alternating projection outside the layer, recording both residuals.

    Reuses the layer's own `_latent`, `_project_balanced` and the ALREADY-TIGHTENED box from
    its forward pass, so this is the same sequence `forward` ran, not a reimplementation of
    it. Done here rather than by instrumenting `forward` because `forward` runs on every
    training step and must not carry a diagnostic.
    """
    with torch.no_grad():
        latent, _ = layer._latent(context)
        v = (latent * layer.config.flux_scale) @ layer.null_basis.t()
        trace: dict[str, list[float]] = {"balance": [], "box": []}
        end_on_box = layer.projection_end == "box"
        for k in range(steps):
            v = torch.minimum(torch.maximum(v, lb), ub)
            if not (end_on_box and k == steps - 1):
                v = layer._project_balanced(v)
            scale = float(v.abs().max())
            trace["balance"].append(
                float(layer._s_matmul(v).abs().max()) / max(scale, 1e-12)
            )
            gap = torch.relu(v - ub) + torch.relu(lb - v)
            trace["box"].append(float(gap.mean()) / float(layer.config.flux_scale))
    return trace


def audit_draws(
    result_name: str,
    n_draws: int,
    n_genotypes: int,
    balance_tol: float,
    uphill_tol: float,
    box_tol: float,
) -> tuple[DrawAudit | None, BoxViolationBreakdown | None, FluxDiagnosis | None]:
    """Rebuild one cell from its checkpoint and price ``n_draws`` sampled fluxes.

    Imported lazily, because the trainer owns ``train_bx.py`` and importing it pulls in the
    whole dataset stack; a summary of the curves must not depend on that.
    """
    import importlib.util

    stem = osp.splitext(result_name)[0]
    ckpt_path = osp.join(
        os.getenv("TC_CHECKPOINT_DIR", osp.join(DATA_ROOT, "checkpoints-027")),
        f"{stem}.pt",
    )
    if not osp.isfile(ckpt_path):
        return None, None, None

    spec = importlib.util.spec_from_file_location(
        "train_bx_for_audit", osp.join(EXP_DIR, "scripts", "train_bx.py")
    )
    assert spec is not None and spec.loader is not None
    tbx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tbx)

    with open(osp.join(RESULTS, result_name)) as handle:
        result = json.load(handle)
    cell = tbx.BxCell(**result["cell"])
    cfg = tbx.load_config()
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert checkpoint["cell"]["seed"] == cell.seed, (
        f"checkpoint is for seed {checkpoint['cell']['seed']} and the result file for "
        f"{cell.seed}; one of the two is the wrong file."
    )

    dataset = tbx.build_dataset()
    pinned, _ = tbx.resolve_pinned_test(dataset, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, main_head, _ = tbx.build_model_for_cell(cell, cfg, dataset, device)
    layer = model.flux_layer
    assert layer is not None, f"arm {cell.arm!r} has no flux layer to sample from."
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(checkpoint["parameters"][name].to(param.device))
    mean = checkpoint["target_mean"].to(device)
    sd = checkpoint["target_sd"].to(device)

    dm = tbx.CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(DATA_ROOT, cfg.data.dataset_subdir, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=cell.batch_size,
        random_seed=cell.split_seed,
        # SINGLE PROCESS, deliberately, whatever the cell trained with. This is one
        # inference pass, so worker processes buy nothing, and a spawned worker re-imports
        # torchcell on every start: that is what killed jobs 1681 and 1684, and an audit
        # that hangs at 1.8 percent CPU waiting on a worker is the same failure wearing a
        # different hat. Observed here on 2026-09-11 with cell.num_workers = 2.
        num_workers=0,
        pin_memory=True,
        prefetch=False,
        follow_batch=["perturbation_indices", "deletion_indices", "phenotype_values"],
        pinned_test_indices=pinned,
    )
    dm.setup()

    # THE SAMPLER, switched on after the weights are loaded, exactly as
    # audit_top_strain_feasibility.py does it. The projection head builds its spread head
    # unconditionally so this flip needs no parameter the checkpoint lacks.
    layer.config.stochastic = True
    model.eval()
    assert layer.latent_stochastic, (
        "the flux layer did not become a sampler when config.stochastic was set. Only the "
        "projection parameterization honors the runtime flip; a box or nullspace cell has "
        "to be trained with latent_stochastic from the start."
    )

    # PASS ONE: rank the whole dataset by predicted betaxanthin. The loader is walked twice
    # rather than held in memory, because the second pass needs the same order and a
    # retained list of 39 batches of graph data on the GPU is a needless allocation. Both
    # passes are in eval order, which is deterministic; the trainer's own feasibility pass
    # relies on the same property to align its masks.
    cell_graph = dataset.cell_graph.to(device)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dm.all_dataloader():
            batch = batch.to(device)
            _, reps = model(cell_graph, batch)
            pred = (
                tbx._head_params(reps["head_outputs"][main_head])[:, 0] * sd[0]
                + mean[0]
            )
            preds.append(pred.detach().cpu().numpy())
    prediction = np.concatenate(preds)
    wanted = set(np.argsort(-prediction, kind="stable")[:n_genotypes].tolist())

    # PASS TWO: draw for the selected genotypes only.
    bal: list[np.ndarray] = []
    box: list[np.ndarray] = []
    uphill: list[np.ndarray] = []
    box_parts: list[dict[str, np.ndarray]] = []
    diagnosis: FluxDiagnosis | None = None
    offset = 0
    with torch.no_grad():
        for batch in dm.all_dataloader():
            batch = batch.to(device)
            size = int(batch.num_graphs)
            keep = [i for i in range(size) if (offset + i) in wanted]
            offset += size
            if not keep:
                continue
            _, reps = model(cell_graph, batch)
            # The DETERMINISTIC forward's box residual, on the same genotypes. Kept apart
            # from the draws because the question it answers is about the head that was
            # scored, not about the sampler.
            rows = torch.tensor(keep, device=reps["flux"]["v"].device)
            box_parts.append(
                {
                    k: reps["flux"][k][rows].detach().cpu().numpy()
                    for k in ("v", "lb", "ub")
                }
            )
            h_genes = reps["H_genes_pert"]
            pert = batch["gene"].perturbation_indices
            pert_batch = batch["gene"].perturbation_indices_batch
            context = perturbed_gene_pool(h_genes, pert, pert_batch)
            if diagnosis is None and layer.parameterization == "projection":
                diagnosis = _diagnose(
                    layer,
                    reps["flux"]["v"][rows].cpu().numpy(),
                    reps["flux"]["lb"][rows].cpu().numpy(),
                    reps["flux"]["ub"][rows].cpu().numpy(),
                    _trace_projection(
                        layer,
                        context,
                        reps["flux"]["lb"],
                        reps["flux"]["ub"],
                        layer.projection_steps,
                    ),
                )
            drawn = layer.sample(h_genes, context, pert, pert_batch, n_draws)
            idx = torch.tensor(keep, device=drawn.v.device)
            bal.append(drawn.balance_ratio_median[:, idx].cpu().numpy())
            box.append(drawn.box_violation_frac[:, idx].cpu().numpy())
            uphill.append(drawn.thermo_uphill_frac[:, idx].cpu().numpy())
    balance = np.concatenate(bal, axis=1)
    boxes = np.concatenate(box, axis=1)
    ups = np.concatenate(uphill, axis=1)
    balanced = balance <= balance_tol
    # An arm with thermo_mode OFF has no delta_r_G, so feasibility() zero-fills the uphill
    # fraction and every draw would score as downhill. That is an absence of a claim, not a
    # satisfied one, so the whole thermodynamic half is reported as null instead.
    thermo_available = layer.thermo_mode is not ThermoMode.OFF
    downhill = ups <= uphill_tol
    feasible = balanced & downhill
    breakdown = _box_breakdown(box_parts, box_tol, float(layer.config.flux_scale))
    audit = DrawAudit(
        n_genotypes=int(balance.shape[1]),
        n_draws=n_draws,
        balance_tolerance=balance_tol,
        uphill_tolerance=uphill_tol,
        box_tolerance=box_tol,
        thermo_available=thermo_available,
        projection_steps=layer.projection_steps,
        projection_end=layer.projection_end,
        thermo_box=bool(layer.thermo_box),
        frac_draws_balanced=float(balanced.mean()),
        frac_draws_downhill=float(downhill.mean()) if thermo_available else None,
        frac_draws_feasible=float(feasible.mean()) if thermo_available else None,
        frac_genotypes_all_draws_feasible=(
            float(feasible.all(axis=0).mean()) if thermo_available else None
        ),
        frac_genotypes_any_draw_feasible=(
            float(feasible.any(axis=0).mean()) if thermo_available else None
        ),
        balance_ratio_median_p50=float(np.median(balance)),
        balance_ratio_median_p90=float(np.percentile(balance, 90)),
        box_violation_frac_mean=float(boxes.mean()),
        uphill_frac_mean=float(ups.mean()) if thermo_available else None,
    )
    return audit, breakdown, diagnosis


#: The three families the wave-6 cells fall into, and the ONE thing that separates them:
#: which constraint the projection loop makes exact. Colored by family rather than by cell
#: because twelve distinct colors would encode nothing, and the family is the finding.
FAMILIES: list[tuple[str, str]] = [
    ("balance end", "mass balance exact"),
    ("box end", "bounds exact"),
    ("tdg", "bounds and second law exact"),
]


def _family(cell: CellSummary) -> str:
    """Which exactness family a cell belongs to, from what its layer actually ran."""
    proj = cell.projection or {}
    if proj.get("thermo_box"):
        return "tdg"
    if proj.get("projection_end") == "box":
        return "box end"
    return "balance end"


def make_figure(cells: list[CellSummary]) -> str:
    """Two panels: the escape curve by family, and the endpoint against the pooled control.

    Panel a is the prediction spread over epochs, the quantity that says whether a cell left
    the betaxanthin mean plateau and whether it stayed off it, with one line per cell colored
    by which constraint its projection loop makes exact. Panel b is the endpoint each cell is
    scored on beside the pooled control at its own seed.
    """
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
        }
    )
    fig, axes = plt.subplots(
        2, 1, figsize=(mm_to_in(179.0), mm_to_in(112.0)), constrained_layout=True
    )
    color = {name: PLOT_PALETTE[i] for i, (name, _) in enumerate(FAMILIES)}

    ax = axes[0]
    seen: set[str] = set()
    for cell in [c for c in cells if c.sd_ratio_curve]:
        fam = _family(cell)
        ax.plot(
            range(len(cell.sd_ratio_curve)),
            cell.sd_ratio_curve,
            color=color[fam],
            linewidth=0.8,
            label=None if fam in seen else f"{fam} ({dict(FAMILIES)[fam]})",
        )
        seen.add(fam)
    ax.axhline(ESCAPE_SD_RATIO, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation prediction sd / target sd")
    ax.set_ylim(0.0, 0.6)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(which="both", axis="y", linewidth=0.3, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=3,
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fancybox=False,
    )

    ax = axes[1]
    scored = [c for c in cells if c.test_spearman_pinned is not None]
    positions = np.arange(len(scored), dtype=float)
    width = 0.38
    for i, cell in enumerate(scored):
        ax.bar(
            positions[i] - width / 2,
            cell.test_spearman_pinned or 0.0,
            width=width,
            color=color[_family(cell)],
            edgecolor="black",
            linewidth=0.5,
        )
    ax.bar(
        positions + width / 2,
        [BASELINES.get(int(c.seed or -1), {}).get("pooled", np.nan) for c in scored],
        width=width,
        color="white",
        edgecolor="black",
        linewidth=0.5,
        hatch="///",
        label="pooled control, same seed",
    )
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([c.cell.split()[0] for c in scored])
    ax.set_ylabel("test Spearman, 629 pinned genes")
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.grid(which="major", axis="y", linewidth=0.3, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=1,
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fancybox=False,
    )

    for ax, letter in zip(axes, "ab", strict=False):
        ax.text(
            -0.06,
            1.30,
            letter,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="top",
            ha="left",
        )

    os.makedirs(IMAGE_DIR, exist_ok=True)
    stamp = timestamp()
    png = osp.join(IMAGE_DIR, f"wave6_projection_{stamp}.png")
    svg = osp.join(IMAGE_DIR, f"wave6_projection_{stamp}.svg")
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    plt.close(fig)
    return svg


#: cell label -> (slurm job name, result file name).
CELLS: dict[str, tuple[str, str]] = {
    "P1 projection scale0 s2700": (
        "027-w6-proj-nocstr-s2700",
        "wave6_proj_flux_projection_s2700.json",
    ),
    "P2 projection scale0 s2701": (
        "027-w6-proj-nocstr-s2701",
        "wave6_proj_flux_projection_s2701.json",
    ),
    "P3 everything-on scale0.1 s2700": (
        "027-w6-proj-anchored-s2700",
        "wave6_proj_flux_projection_anchored_s2700.json",
    ),
    "P4 everything-on scale0.1 s2701": (
        "027-w6-proj-anchored-s2701",
        "wave6_proj_flux_projection_anchored_s2701.json",
    ),
    # The second-law-as-a-box cells. `tdg` is the directional box, `ue` the union_external
    # deltaG route, and all four end on the clip at K = 20.
    "P5 tdg scale0.1 s2700": (
        "027-w6-proj-tdg-s2700",
        "wave6_proj_flux_projection_anchored_tdg_ue_s2700.json",
    ),
    "P6 tdg scale0.1 s2701": (
        "027-w6-proj-tdg-s2701",
        "wave6_proj_flux_projection_anchored_tdg_ue_s2701.json",
    ),
    "P7 tdg scale0 s2700": (
        "027-w6-proj-tdg-nocstr-s2700",
        "wave6_proj_flux_projection_anchored_tdg_ue_nocstr_s2700.json",
    ),
    "P8 tdg stochastic scale0.1 s2700": (
        "027-w6-proj-tdg-stoch-s2700",
        "wave6_proj_flux_projection_anchored_tdg_stoch_ue_s2700.json",
    ),
    # The two discriminating controls. P9 and P10 repeat the tdg arm at further seeds, to
    # say whether its zero signal is the head or one seed; P11 and P12 end on the clip with
    # NO sign tightening, to say whether respecting the bounds alone costs the signal.
    "P9 tdg scale0 s2701": (
        "027-w6-proj-tdg-nocstr-s2701",
        "wave6_proj_flux_projection_anchored_tdg_ue_nocstr_s2701.json",
    ),
    "P10 tdg scale0 s2702": (
        "027-w6-proj-tdg-nocstr-s2702",
        "wave6_proj_flux_projection_anchored_tdg_ue_nocstr_s2702.json",
    ),
    "P11 box-end scale0 s2700": (
        "027-w6-proj-boxend-nocstr-s2700",
        "wave6_proj_flux_projection_anchored_k20_boxend_ue_nocstr_s2700.json",
    ),
    "P12 box-end scale0 s2701": (
        "027-w6-proj-boxend-nocstr-s2701",
        "wave6_proj_flux_projection_anchored_k20_boxend_ue_nocstr_s2701.json",
    ),
    # Third seeds for the two arms that keep the signal, and the first cells for the
    # 5 kJ/mol margin, which asks whether the sign tightening can be kept at a strength
    # that leaves the head somewhere to move.
    "P13 tdgm5 scale0 s2700": (
        "027-w6-proj-tdgm5-nocstr-s2700",
        "wave6_proj_flux_projection_anchored_tdgm5_ue_nocstr_s2700.json",
    ),
    "P14 tdgm5 scale0 s2701": (
        "027-w6-proj-tdgm5-nocstr-s2701",
        "wave6_proj_flux_projection_anchored_tdgm5_ue_nocstr_s2701.json",
    ),
    "P15 projection scale0 s2702": (
        "027-w6-proj-nocstr-s2702",
        "wave6_proj_flux_projection_s2702.json",
    ),
    "P16 box-end scale0 s2702": (
        "027-w6-proj-boxend-nocstr-s2702",
        "wave6_proj_flux_projection_anchored_k20_boxend_ue_nocstr_s2702.json",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--draws",
        type=int,
        default=64,
        help="Sampled fluxes per genotype in the checkpoint audit.",
    )
    parser.add_argument(
        "--draw-genotypes",
        type=int,
        default=100,
        help="How many top-predicted strains to draw for. 100 is the Genesis group.",
    )
    parser.add_argument("--balance-tol", type=float, default=1e-2)
    parser.add_argument("--uphill-tol", type=float, default=0.0)
    parser.add_argument("--box-tol", type=float, default=1e-4)
    parser.add_argument(
        "--no-draws",
        action="store_true",
        help="Skip the checkpoint audit. It needs the dataset and a GPU is worth having.",
    )
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--out", default="wave6_projection_summary.json")
    args = parser.parse_args()

    cells: list[CellSummary] = []
    for label, (job_name, result_name) in CELLS.items():
        log_path, job_id = newest_log(job_name)
        summary = CellSummary(cell=label, job_id=job_id, log_path=log_path)
        if log_path is not None:
            read_curve(summary, parse_log(log_path))
        result_path = osp.join(RESULTS, result_name)
        if osp.exists(result_path):
            summary.result_path = result_path
            with open(result_path) as handle:
                read_result(summary, json.load(handle))
            if not args.no_draws:
                (summary.draw_audit, summary.box_breakdown, summary.flux_diagnosis) = (
                    audit_draws(
                        result_name,
                        args.draws,
                        args.draw_genotypes,
                        args.balance_tol,
                        args.uphill_tol,
                        args.box_tol,
                    )
                )
        if log_path is None and summary.result_path is None:
            continue
        cells.append(summary)

    figure_path = None if args.no_figure or not cells else make_figure(cells)
    summary_all = ProjectionSummary(
        escape_rule=(
            "escape_epoch is the first epoch beginning a run of "
            f"{ESCAPE_CONSECUTIVE_EPOCHS} consecutive epochs with val_pred_sd_ratio > "
            f"{ESCAPE_SD_RATIO}, the wave-5 rule verbatim. escape_sustained is whether the "
            "LAST three epochs of the run are also above it, which is what separates an "
            "escape that held from job 1678's escape-then-collapse."
        ),
        scoring_note=(
            "test_spearman_pinned is read at the validation-selected epoch, never as a "
            "maximum over epochs. The 019 replicate study puts the between-seed spread near "
            "0.04 in Pearson, so a paired difference under that is within noise at one seed. "
            "Draw fractions carry their tolerances in the same record."
        ),
        baselines={str(k): v for k, v in BASELINES.items()},
        cells=cells,
        figure_path=figure_path,
    )
    out_path = osp.join(RESULTS, args.out)
    with open(out_path, "w") as handle:
        json.dump(summary_all.model_dump(), handle, indent=1)
    print(f"[analyze_projection] wrote {out_path}")
    if figure_path:
        print(f"[analyze_projection] wrote {figure_path}")
    for cell in cells:
        print(
            f"  {cell.cell:34s} job {cell.job_id or '-':>5s} "
            f"ep {cell.epochs_completed:3d} escape {str(cell.escape_epoch):>5s} "
            f"sustained {str(cell.escape_sustained):>5s} "
            f"sel {str(cell.select_epoch):>4s} "
            f"testSp {cell.test_spearman_pinned if cell.test_spearman_pinned is None else round(cell.test_spearman_pinned, 4)}"
        )


if __name__ == "__main__":
    main()
