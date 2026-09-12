# experiments/027-betaxanthin-metabolic/scripts/analyze_wave7_aa.py
# [[experiments.027-betaxanthin-metabolic.wave7-amino-acid-concentration]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/analyze_wave7_aa.py

r"""Aggregate the wave-7 amino-acid cells: does supervising the CONCENTRATION help?

WHAT WAVE 7 CONTRASTS. Four arms on ``--head mulleder19``, three seeds each, paired:

* ``pooled``    -- no metabolic module, the control.
* ``turnover``  -- wave 6's ``flux_metabolite`` readout, ``a_k log(1 + omega_k) + b_k`` on
  the turnover of each amino acid's GEM metabolite, everything on at constraint scale 0.1.
* ``conc``      -- the same arm and scale with ``flux_concentration``: the prediction IS the
  layer's learned log10 molar concentration of that metabolite, no parameters on the mean.
* ``conc_nocstr`` -- the concentration readout with the constraint scale at exactly 0, which
  separates the readout from the cost the physics penalty charges.

WHAT IT READS, and nothing else: finished result files under ``results/``. A cell with no
result file is listed by name under ``missing_files`` rather than dropped, so the script is
safe to re-run while the queue drains.

THREE THINGS IT REPORTS
-----------------------
1. **The paired phenotype contrast**: mean per-feature test Spearman over the 19 columns on
   the pinned Merzbacher test genes at the validation-selected epoch, each flux arm minus
   pooled at the same seed. Never a maximum over epochs.
2. **The mechanistic readout**: per amino acid, the Spearman and Pearson between the
   layer's LEARNED log10 concentration and the measured one on the held-out genes, for
   every arm whose layer is anchored, supervised or not, plus the fraction of predicted
   values pinned at the concentration window's edge.
3. **Feasibility**: the balance, box and second-law fractions and the balance residual, so
   a readout change can be read against what it did to the physics.

THE PHYSICS GUARD. Every wave-7 flux cell must carry the same kcat table (97 measured / 782
homolog / 2,831 predicted / 18 default); a cell built from a different one is not a
replicate and is reported under ``physics_mismatch`` rather than averaged in.

Run from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/analyze_wave7_aa.py

Writes ``results/wave7_aa_summary.json`` and one timestamped figure into
``$ASSET_IMAGES_DIR/027-betaxanthin-metabolic/``.
"""

import json
import os
import os.path as osp
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator
from pydantic import BaseModel, ConfigDict, Field

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXP_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic")
RESULTS = osp.join(EXP_DIR, "results")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "027-betaxanthin-metabolic")

SEEDS = (2700, 2701, 2702)
#: THE PRIMARY BUDGET IS 200 EPOCHS, and the reason is measured rather than chosen. On the
#: 80-epoch cells the concentration head was still rising at the last epoch (job 1746, seed
#: 2700: validation Pearson max 0.1312 at epoch 79 with the validation loss still falling;
#: job 1747, seed 2701: 0.1407 at epoch 75), while the pooled control peaks at epoch 23 to
#: 27 and overfits after. Selecting an epoch inside a budget one arm has not finished using
#: biases the contrast against that arm, so the 200-epoch cells give every arm the same
#: budget and the 80-epoch cells are kept as the shorter-budget read of the same arms.
BUDGETS: dict[str, str] = {"e200": "_e200", "e80": ""}
PRIMARY_BUDGET = "e200"
ARM_STEMS: tuple[str, ...] = ("pooled", "pooledq", "turnover", "conc", "conc_nocstr")


def arms_for(budget: str) -> dict[str, str]:
    """Result-file stems for one epoch budget, keyed by arm."""
    tag = BUDGETS[budget]
    return {arm: f"wave7_aa_{arm}{tag}_s{{seed}}.json" for arm in ARM_STEMS}


ARMS: dict[str, str] = arms_for(PRIMARY_BUDGET)
#: Axis labels, so no panel ever shows a result-file stem.
ARM_LABELS: dict[str, str] = {
    "pooled": "pooled",
    "pooledq": "pooled,\nquantile",
    "turnover": "turnover",
    "conc": "concentration",
    "conc_nocstr": "concentration,\nno penalty",
}
#: The kcat table every wave-7 flux cell is required to have resolved.
EXPECTED_KCAT_TIERS: dict[str, int] = {
    "measured": 97,
    "homolog": 782,
    "predicted": 2831,
    "default": 18,
}
#: The three component criteria of the Genesis metabolic-grounding test.
CRITERIA = (
    ("balance_median_ratio_le_1e-02", "mass balance within 1%"),
    ("box_violation_frac_eq_0", "inside the flux box"),
    ("thermo_model_violation_frac_eq_0", "no uphill reaction"),
)


class PairedContrast(BaseModel):
    """One arm-versus-arm contrast, paired by seed."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    arm_a: str
    arm_b: str
    seeds: list[int]
    a_values: list[float]
    b_values: list[float]
    differences: list[float]
    n_pairs: int
    mean_difference: float
    sd_difference: float
    t_statistic: float | None
    t_statistic_note: str
    a_mean: float
    a_sd: float
    b_mean: float
    b_sd: float
    missing: list[str] = Field(default_factory=list)


class ConcentrationRow(BaseModel):
    """The learned concentration against Mulleder, for one cell."""

    model_config = ConfigDict(extra="forbid")

    cell: str
    arm: str
    #: The wave-7 arm the cell belongs to ('turnover', 'conc', 'conc_nocstr'), taken from
    #: the result stem: all three flux arms share one `arm` name and differ in the readout
    #: and the constraint scale, so `arm` alone prints the same string three times.
    label: str
    seed: int
    supervised: bool
    conc_lower_m: float
    conc_upper_m: float
    n_records: int
    n_measurements: int
    mean_spearman: float
    mean_pearson: float
    columns: list[str]
    spearman_per_feature: list[float]
    pearson_per_feature: list[float]
    median_abs_error_log10: list[float]
    frac_pred_at_ceiling: float
    frac_pred_at_floor: float


class CalibrationRow(BaseModel):
    """How calibrated one cell's predictive distribution is on the pinned test genes."""

    model_config = ConfigDict(extra="forbid")

    cell: str
    arm: str
    seed: int
    dist: str
    n_measurements: int
    coverage: dict[str, float]
    pit_ks: float
    pit_mean: float


class CellRow(BaseModel):
    """One finished cell, everything the note quotes about it."""

    model_config = ConfigDict(extra="forbid")

    cell: str
    arm: str
    seed: int
    select_epoch: int
    select_rule: str
    epochs_run: int
    score_mean_spearman: float
    score_mean_pearson: float
    n_pinned_measurements: int
    n_pinned_records: int
    epochs_configured: int
    final_val_pred_sd_ratio: float
    selected_val_pred_sd_ratio: float
    #: The centered 5-epoch moving average of validation Pearson, the curve the epoch is
    #: selected on, written out in full so a note can plot or re-read it.
    val_pearson_ma: list[float]
    val_pearson_ma_epochs: list[int]
    #: The MAXIMUM of that curve. An upward-biased order statistic whose bias grows with
    #: the budget, reported as a DIAGNOSTIC of how much was left on the table, never as the
    #: score. The score is `score_mean_spearman`, the test Spearman at the selected epoch.
    max_val_pearson_ma_diagnostic: float
    argmax_val_pearson_ma_epoch: int
    #: True when the selected epoch falls in the last tenth of the run, i.e. the curve had
    #: not turned over and the budget, not the model, ended the run.
    still_rising: bool
    kcat_tier_counts: dict[str, int] | None
    thermo_source: str | None
    conc_upper_m: float
    aa_readout: str
    target_log10_molar: bool
    feasibility: dict[str, float] = Field(default_factory=dict)
    balance_median: float | None = None
    balance_max: float | None = None
    wandb_url: str | None = None
    checkpoint_path: str | None = None
    physics_mismatch: str | None = None


class Wave7Summary(BaseModel):
    """Everything the wave-7 note reports."""

    model_config = ConfigDict(extra="forbid")

    generated_at: str
    expected_kcat_tier_counts: dict[str, int]
    primary_budget: str
    #: Contrasts keyed by "<budget>/<contrast>", so the 200-epoch and 80-epoch reads of the
    #: same arm pair are never mixed into one number.
    contrasts: dict[str, PairedContrast]
    cells: list[CellRow]
    calibration: list[CalibrationRow]
    concentration: list[ConcentrationRow]
    missing_files: list[str]
    figure_path: str | None = None


def _load(stem: str) -> dict[str, Any] | None:
    path = osp.join(RESULTS, stem)
    if not osp.isfile(path):
        return None
    with open(path) as fh:
        return cast(dict[str, Any], json.load(fh))


def _score(result: dict[str, Any]) -> float:
    """Mean per-feature Spearman on the pinned test genes at the selected epoch."""
    return float(result["test"]["pinned"]["mean_spearman"])


def _contrast(
    metric: str,
    loaded: dict[str, dict[str, Any]],
    a_name: str,
    b_name: str,
    missing: list[str],
    arms: dict[str, str],
) -> PairedContrast:
    seeds: list[int] = []
    a_vals: list[float] = []
    b_vals: list[float] = []
    for seed in SEEDS:
        a = loaded.get(arms[a_name].format(seed=seed))
        b = loaded.get(arms[b_name].format(seed=seed))
        if a is None or b is None:
            continue
        seeds.append(seed)
        a_vals.append(_score(a))
        b_vals.append(_score(b))
    diffs = [a - b for a, b in zip(a_vals, b_vals, strict=True)]
    n = len(diffs)
    # ddof=1: the sample SD of the differences, the denominator of the paired t.
    sd = float(np.std(diffs, ddof=1)) if n > 1 else float("nan")
    mean = float(np.mean(diffs)) if n else float("nan")
    return PairedContrast(
        metric=metric,
        arm_a=a_name,
        arm_b=b_name,
        seeds=seeds,
        a_values=a_vals,
        b_values=b_vals,
        differences=diffs,
        n_pairs=n,
        mean_difference=mean,
        sd_difference=sd,
        t_statistic=(float(mean / (sd / np.sqrt(n))) if n > 1 and sd > 0 else None),
        t_statistic_note=(
            f"paired t = mean(difference) / (sd(difference) / sqrt(n)) on n={n} seeds, "
            f"df={max(n - 1, 0)}. Read for sign and magnitude; at this n it summarizes "
            "the spread and is not a significance claim."
        ),
        a_mean=float(np.mean(a_vals)) if a_vals else float("nan"),
        a_sd=float(np.std(a_vals, ddof=1)) if len(a_vals) > 1 else float("nan"),
        b_mean=float(np.mean(b_vals)) if b_vals else float("nan"),
        b_sd=float(np.std(b_vals, ddof=1)) if len(b_vals) > 1 else float("nan"),
        missing=missing,
    )


def _physics_mismatch(result: dict[str, Any]) -> str | None:
    """Whether this cell's kcat table is the one every wave-7 flux cell must carry."""
    meta = result.get("flux_meta") or {}
    if not meta:
        return None
    counts = meta.get("kcat_tier_counts") or {}
    got = {k: int(v) for k, v in counts.items() if k in EXPECTED_KCAT_TIERS}
    if got == EXPECTED_KCAT_TIERS:
        return None
    return (
        f"kcat tiers {got} against the required {EXPECTED_KCAT_TIERS}; this cell was built "
        "from a different parameter table and is not a replicate of the others."
    )


def _cell_row(stem: str, result: dict[str, Any]) -> CellRow:
    cfg = result["cell"]
    history = result["history"]
    block = result.get("feasibility") or {"available": False}
    fractions: dict[str, float] = {}
    balance_median: float | None = None
    balance_max: float | None = None
    if block.get("available"):
        pop = block["groups"]["all"]
        fractions = {k: float(v) for k, v in pop["fractions"].items()}
        balance_median = float(pop["stats"]["balance_median_ratio"]["median"])
        balance_max = float(pop["stats"]["balance_max_ratio"]["median"])
    sel = int(result["select_epoch"])
    meta = result.get("flux_meta") or {}
    curve = [float(r["val_pearson"]) for r in history]
    window = int(cfg.get("select_smoothing_window", 5))
    half = window // 2
    ma = [
        float(np.mean(curve[i - half : i + half + 1]))
        for i in range(half, len(curve) - half)
    ]
    ma_epochs = list(range(half, len(curve) - half))
    argmax_ma = int(np.argmax(ma)) + half if ma else sel
    return CellRow(
        cell=stem,
        arm=cfg["arm"],
        seed=int(cfg["seed"]),
        select_epoch=sel,
        select_rule=result["select_rule"],
        epochs_run=len(history),
        score_mean_spearman=_score(result),
        score_mean_pearson=float(result["test"]["pinned"]["mean_pearson"]),
        n_pinned_measurements=int(result["test"]["pinned"]["n_measurements"]),
        n_pinned_records=int(result["test"]["pinned"]["n_records_with_any_feature"]),
        epochs_configured=int(cfg["epochs"]),
        final_val_pred_sd_ratio=float(history[-1]["val_pred_sd_ratio"]),
        selected_val_pred_sd_ratio=float(history[sel]["val_pred_sd_ratio"]),
        val_pearson_ma=ma,
        val_pearson_ma_epochs=ma_epochs,
        max_val_pearson_ma_diagnostic=float(max(ma)) if ma else float("nan"),
        argmax_val_pearson_ma_epoch=argmax_ma,
        still_rising=sel >= 0.9 * (len(history) - 1),
        kcat_tier_counts=(
            {k: int(v) for k, v in (meta.get("kcat_tier_counts") or {}).items()}
            if meta
            else None
        ),
        thermo_source=meta.get("thermo_source"),
        conc_upper_m=float(cfg.get("conc_upper_m", 1.0e-2)),
        aa_readout=str(cfg.get("aa_readout", "flux_metabolite")),
        target_log10_molar=bool(cfg.get("target_log10_molar", False)),
        feasibility=fractions,
        balance_median=balance_median,
        balance_max=balance_max,
        wandb_url=result.get("wandb_url"),
        checkpoint_path=result.get("checkpoint_path"),
        physics_mismatch=_physics_mismatch(result),
    )


def _calibration_row(stem: str, result: dict[str, Any]) -> CalibrationRow | None:
    block = result.get("calibration")
    if block is None or not block["available"]:
        return None
    cfg = result["cell"]
    return CalibrationRow(
        cell=stem,
        arm=cfg["arm"],
        seed=int(cfg["seed"]),
        dist=str(cfg.get("dist", "crps")),
        n_measurements=int(block["n_measurements"]),
        coverage={k: float(v) for k, v in block["coverage"].items()},
        pit_ks=float(block["pit_ks"]),
        pit_mean=float(block["pit_mean"]),
    )


def _concentration_row(stem: str, result: dict[str, Any]) -> ConcentrationRow | None:
    audit = result.get("concentration_audit")
    if audit is None or not audit["available"]:
        return None
    group = audit["groups"]["pinned_test"]
    cfg = result["cell"]
    return ConcentrationRow(
        cell=stem,
        arm=cfg["arm"],
        label=stem.removeprefix("wave7_aa_").removesuffix(".json").rsplit("_s", 1)[0],
        seed=int(cfg["seed"]),
        supervised=bool(audit["supervised"]),
        conc_lower_m=float(audit["conc_lower_m"]),
        conc_upper_m=float(audit["conc_upper_m"]),
        n_records=int(group["n_records"]),
        n_measurements=int(group["n_measurements"]),
        mean_spearman=float(group["mean_spearman"]),
        mean_pearson=float(group["mean_pearson"]),
        columns=list(group["columns"]),
        spearman_per_feature=[float(x) for x in group["spearman_per_feature"]],
        pearson_per_feature=[float(x) for x in group["pearson_per_feature"]],
        median_abs_error_log10=[float(x) for x in group["median_abs_error_log10"]],
        frac_pred_at_ceiling=float(group["frac_pred_at_ceiling"]),
        frac_pred_at_floor=float(group["frac_pred_at_floor"]),
    )


def _panel_arms(
    ax: Any, loaded: dict[str, dict[str, Any]], budget: str = PRIMARY_BUDGET
) -> None:
    """One line per seed across the four arms, so the pairing is read vertically."""
    ARMS = arms_for(budget)
    live = [a for a in ARMS if any(loaded.get(ARMS[a].format(seed=s)) for s in SEEDS)]
    xs = list(range(len(live)))
    drawn = 0
    for i, seed in enumerate(SEEDS):
        res = [loaded.get(ARMS[a].format(seed=seed)) for a in live]
        # A seed missing one arm still shows the arms it HAS: dropping the whole seed
        # would hide two measured cells because a third has not run.
        present = [(x, r) for x, r in zip(xs, res, strict=True) if r is not None]
        if not present:
            continue
        ax.plot(
            [x for x, _ in present],
            [_score(r) for _, r in present],
            marker="o",
            markersize=2.0,
            linewidth=0.6,
            color=PLOT_PALETTE[i % len(PLOT_PALETTE)],
            label=f"seed {seed}",
        )
        drawn += 1
    ax.set_xticks(xs)
    ax.set_xticklabels([ARM_LABELS[a] for a in live])
    ax.set_xlim(-0.35, max(len(live) - 1, 1) - 0.65 + 0.35 + 0.65)
    ax.set_ylabel("mean per-feature test Spearman")
    if drawn:
        lo, hi = ax.get_ylim()
        ax.yaxis.set_minor_locator(MultipleLocator(max((hi - lo) / 10.0, 1e-6)))
        ax.tick_params(which="minor", length=0)
        ax.grid(True, which="both", axis="y", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)


def _panel_per_amino_acid(ax: Any, rows: list[ConcentrationRow]) -> None:
    """Per amino acid, the learned concentration against the measured one, by arm.

    Averaged over seeds within an arm, because the question is about the arm and three
    lines per arm would be nineteen groups of twelve bars.
    """
    by_arm: dict[str, list[ConcentrationRow]] = {}
    for row in rows:
        by_arm.setdefault(row.label, []).append(row)
    if not by_arm:
        ax.set_xticks([])
        ax.set_ylabel("Spearman, learned vs measured")
        return
    columns = rows[0].columns
    idx = np.arange(len(columns), dtype=float)
    keys = sorted(by_arm)
    width = 0.8 / len(keys)
    for i, key in enumerate(keys):
        stack = np.asarray([r.spearman_per_feature for r in by_arm[key]], dtype=float)
        ax.bar(
            idx + (i - (len(keys) - 1) / 2) * width,
            stack.mean(axis=0),
            width,
            color=PLOT_PALETTE[i % len(PLOT_PALETTE)],
            edgecolor="black",
            linewidth=0.3,
            label=f"{key} (n={stack.shape[0]})",
        )
    ax.axhline(0.0, color="black", linewidth=0.4)
    ax.set_xticks(idx)
    ax.set_xticklabels(columns, rotation=90)
    ax.set_ylabel("Spearman, learned vs measured")
    ax.grid(True, which="both", axis="y", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)


def _panel_escape(ax: Any, loaded: dict[str, dict[str, Any]]) -> None:
    """Validation prediction spread over the target's, per epoch, one line per cell."""
    for i, (stem, result) in enumerate(sorted(loaded.items())):
        history = result["history"]
        ax.plot(
            [row["epoch"] for row in history],
            [row["val_pred_sd_ratio"] for row in history],
            linewidth=0.5,
            color=PLOT_PALETTE[i % len(PLOT_PALETTE)],
            label=stem.removeprefix("wave7_aa_").removesuffix(".json"),
        )
    ax.axhline(0.05, color="black", linewidth=0.4, linestyle="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation prediction spread ratio")
    ax.grid(True, which="both", axis="y", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)


def _panel_curves(ax: Any, cells: list[CellRow]) -> None:
    """The curve the epoch is selected on, with the selected epoch marked.

    This is what says whether a budget ended a run before its curve turned over: a marker
    sitting at the right edge is a cell whose selection was made by the wall, not by the
    model. The marker is the datum, not an annotation.
    """
    for i, row in enumerate(cells):
        if not row.val_pearson_ma:
            continue
        color = PLOT_PALETTE[i % len(PLOT_PALETTE)]
        ax.plot(
            row.val_pearson_ma_epochs,
            row.val_pearson_ma,
            linewidth=0.5,
            color=color,
            label=row.cell.removeprefix("wave7_aa_").removesuffix(".json"),
        )
        ax.plot(
            [row.select_epoch],
            [row.val_pearson_ma[row.val_pearson_ma_epochs.index(row.select_epoch)]]
            if row.select_epoch in row.val_pearson_ma_epochs
            else [row.val_pearson_ma[-1]],
            marker="o",
            markersize=2.5,
            linestyle="none",
            color=color,
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation Pearson (5-epoch mean)")
    ax.grid(True, which="both", axis="y", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)


def _panel_criteria(ax: Any, cells: list[CellRow]) -> None:
    """Fraction of all genotypes meeting each component criterion, per flux cell."""
    usable = [c for c in cells if c.feasibility]
    if not usable:
        ax.set_xticks([])
        ax.set_ylabel("fraction of genotypes")
        return
    idx = np.arange(len(usable), dtype=float)
    width = 0.27
    for i, (key, label) in enumerate(CRITERIA):
        ax.bar(
            idx + (i - 1) * width,
            [c.feasibility.get(key, np.nan) for c in usable],
            width,
            color=PLOT_PALETTE[i],
            edgecolor="black",
            linewidth=0.4,
            label=label,
        )
    ax.set_xticks(idx)
    ax.set_xticklabels(
        [f"{c.cell.removeprefix('wave7_aa_').removesuffix('.json')}" for c in usable],
        rotation=90,
    )
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.set_ylabel("fraction of genotypes")
    ax.grid(True, which="both", axis="y", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)


def make_figure(
    loaded: dict[str, dict[str, Any]],
    cells: list[CellRow],
    conc: list[ConcentrationRow],
    stamp: str,
) -> str:
    """The four panels the note embeds."""
    apply_paper_style()
    fig, grid = plt.subplots(
        3,
        2,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(168.0)),
        constrained_layout=True,
    )
    axes = grid.ravel()
    _panel_arms(axes[0], loaded, PRIMARY_BUDGET)
    _panel_per_amino_acid(axes[1], conc)
    _panel_curves(axes[2], [c for c in cells if c.epochs_configured >= 200])
    _panel_escape(axes[3], loaded)
    _panel_criteria(axes[4], cells)
    _panel_arms(axes[5], loaded, "e80")
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
    for ax, ncol in (
        (axes[0], 3),
        (axes[1], 3),
        (axes[2], 2),
        (axes[3], 2),
        (axes[5], 3),
    ):
        if ax.get_legend_handles_labels()[0]:
            ax.legend(
                loc="lower left",
                bbox_to_anchor=(0, 1.02),
                ncol=ncol,
                frameon=True,
                fontsize=5.0,
                handlelength=1.2,
                borderpad=0.3,
            )
    for ax, letter in zip(axes, "abcdef", strict=True):
        ax.text(
            -0.16,
            1.12,
            letter,
            transform=ax.transAxes,
            fontsize=8.0,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
    os.makedirs(IMAGE_DIR, exist_ok=True)
    base = osp.join(IMAGE_DIR, f"wave7_aa_concentration_{stamp}")
    fig.savefig(f"{base}.png", dpi=300)
    savefig_true_size_svg(fig, f"{base}.svg")
    plt.close(fig)
    return f"{base}.svg"


def main() -> None:
    stamp = timestamp()
    loaded: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for budget in BUDGETS:
        for pattern in arms_for(budget).values():
            for seed in SEEDS:
                stem = pattern.format(seed=seed)
                result = _load(stem)
                if result is None:
                    missing.append(stem)
                else:
                    loaded[stem] = result

    #: Every contrast is computed WITHIN one epoch budget. An arm scored at 80 epochs and
    #: one scored at 200 are not paired observations: the selection rule searches a longer
    #: curve in the second, and the concentration head was still rising when the 80-epoch
    #: budget ended.
    pairs = (
        ("turnover_vs_pooled", "turnover", "pooled", "turnover readout vs pooled"),
        ("conc_vs_pooled", "conc", "pooled", "concentration readout vs pooled"),
        (
            "conc_nocstr_vs_pooled",
            "conc_nocstr",
            "pooled",
            "concentration readout at constraint scale 0 vs pooled",
        ),
        ("conc_vs_turnover", "conc", "turnover", "concentration vs turnover readout"),
        (
            "conc_vs_conc_nocstr",
            "conc",
            "conc_nocstr",
            "concentration readout with the physics penalty vs without it",
        ),
        (
            "pooledq_vs_pooled",
            "pooledq",
            "pooled",
            "pooled head under the pinball loss vs under the Gaussian CRPS",
        ),
        (
            "conc_vs_pooledq",
            "conc",
            "pooledq",
            "concentration readout vs the pinball-scored pooled control",
        ),
    )
    contrasts: dict[str, PairedContrast] = {}
    for budget in BUDGETS:
        arms = arms_for(budget)
        for key, a_name, b_name, what in pairs:
            contrasts[f"{budget}/{key}"] = _contrast(
                f"mean per-feature test Spearman (mulleder19), {what}, {budget}",
                loaded,
                a_name,
                b_name,
                missing,
                arms,
            )
    cells = [_cell_row(stem, result) for stem, result in sorted(loaded.items())]
    conc_rows = [
        row
        for stem, result in sorted(loaded.items())
        if (row := _concentration_row(stem, result)) is not None
    ]
    calib_rows = [
        crow
        for stem, result in sorted(loaded.items())
        if (crow := _calibration_row(stem, result)) is not None
    ]
    figure = make_figure(loaded, cells, conc_rows, stamp)
    summary = Wave7Summary(
        generated_at=stamp,
        expected_kcat_tier_counts=EXPECTED_KCAT_TIERS,
        primary_budget=PRIMARY_BUDGET,
        contrasts=contrasts,
        cells=cells,
        calibration=calib_rows,
        concentration=conc_rows,
        missing_files=sorted(set(missing)),
        figure_path=figure,
    )
    os.makedirs(RESULTS, exist_ok=True)
    path = osp.join(RESULTS, "wave7_aa_summary.json")
    with open(path, "w") as fh:
        json.dump(summary.model_dump(), fh, indent=2)

    print(f"wrote {path}\nfigure {figure}")
    print("\n| contrast | n | mean difference | sd | t |")
    print("|---|---:|---:|---:|---:|")
    for name, c in contrasts.items():
        t = "n/a" if c.t_statistic is None else f"{c.t_statistic:.2f}"
        print(
            f"| {name} | {c.n_pairs} | {c.mean_difference:+.4f} | "
            f"{c.sd_difference:.4f} | {t} |"
        )
    print(
        "\n| cell | epochs | sel epoch | mean spearman | sd ratio | still rising | "
        "max MA (biased) | kcat tiers | mismatch |"
    )
    print("|---|---:|---:|---:|---:|---|---:|---|---|")
    for cell_row in cells:
        tiers = (
            "none"
            if cell_row.kcat_tier_counts is None
            else str(cell_row.kcat_tier_counts)
        )
        print(
            f"| {cell_row.cell} | {cell_row.epochs_run} | {cell_row.select_epoch} | "
            f"{cell_row.score_mean_spearman:+.4f} | "
            f"{cell_row.final_val_pred_sd_ratio:.3f} | "
            f"{'yes' if cell_row.still_rising else 'no'} | "
            f"{cell_row.max_val_pearson_ma_diagnostic:+.4f} at "
            f"{cell_row.argmax_val_pearson_ma_epoch} | {tiers} | "
            f"{'OK' if cell_row.physics_mismatch is None else cell_row.physics_mismatch} |"
        )
    print("\n| cell | supervised | conc mean spearman | mean pearson | at ceiling |")
    print("|---|---|---:|---:|---:|")
    for r in conc_rows:
        print(
            f"| {r.cell} | {r.supervised} | {r.mean_spearman:+.4f} | "
            f"{r.mean_pearson:+.4f} | {r.frac_pred_at_ceiling:.3f} |"
        )
    if calib_rows:
        print("\n| cell | loss | n | cov 0.5 | cov 0.8 | cov 0.9 | PIT KS | PIT mean |")
        print("|---|---|---:|---:|---:|---:|---:|---:|")
        for crow in calib_rows:
            print(
                f"| {crow.cell} | {crow.dist} | {crow.n_measurements} | "
                f"{crow.coverage.get('alpha_0.50', float('nan')):.3f} | "
                f"{crow.coverage.get('alpha_0.80', float('nan')):.3f} | "
                f"{crow.coverage.get('alpha_0.90', float('nan')):.3f} | "
                f"{crow.pit_ks:.4f} | {crow.pit_mean:.3f} |"
            )
    if missing:
        print(f"\nmissing ({len(missing)}): {', '.join(sorted(set(missing)))}")


if __name__ == "__main__":
    main()
