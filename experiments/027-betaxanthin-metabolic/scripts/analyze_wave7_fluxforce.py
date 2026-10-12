# experiments/027-betaxanthin-metabolic/scripts/analyze_wave7_fluxforce.py
# [[experiments.027-betaxanthin-metabolic.wave7-flux-force-head]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/analyze_wave7_fluxforce
"""Wave 7: the flux-force channel against the sign clamp and against no thermodynamics.

THREE ARMS, PAIRED BY SEED, EVERYTHING ELSE HELD FIXED. All three are twenty alternations
ending on the clip, everything on, constraint scale 0.0, the union_external deltaG table and
the shipped verified kcat table (97/782/2831/18); they differ in the parameterization string
and nothing else.

``fluxforce_k20_box``
    The second law as a smooth coupling: ``v = m f(dG)`` with ``f(x) = -tanh(x / 2RT)`` and
    ``m >= 0`` read off the genotype box, then the clip and the projection.
``projection_k20_box_tdg``
    The sign clamp. ``dG > eps`` forces ``ub <- 0`` inside a ``torch.where``, which passes no
    gradient to the learned concentrations at all.
``projection_k20_box``
    The same projection with no thermodynamics in the box, which is what says how much of
    any difference is the second law rather than the clip.

WHAT IS READ, AND THE SCORING RULE IS NAMED BECAUSE IT IS NOT A MAXIMUM.

* **Signal.** The pinned Merzbacher test Spearman at the validation-selected epoch, which
  the trainer picks by a centered 5-epoch moving average of the validation Pearson. A max
  over 80 noisy epochs is an upward-biased order statistic; this is not that.
* **Escape.** The first epoch beginning three consecutive epochs with
  ``val_pred_sd_ratio`` above 0.05, and whether the last three epochs are still above it.
  A run that escapes the mean plateau and falls back is not an escape.
* **Feasibility.** The trainer's own block, at the ``all_top100`` group, because the Genesis
  metabolic-grounding target is a statement about the strains a model NOMINATES. The joint
  key ``both_balance_le_1e-02_and_thermo_model_eq_0`` is the Genesis number;
  ``pinned_test_top100`` is carried alongside it because the two groups have disagreed.
* **The projection's own cost.** ``feas_sign_flip_frac``, the fraction of flux-carrying
  priced reactions whose sign the projection reversed. The flux-force channel satisfies the
  second law exactly BEFORE the projection, so this is the whole of the violation it
  produces, and it is reported rather than inferred.

Run from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/analyze_wave7_fluxforce.py

Writes ``results/wave7_fluxforce_summary.json`` and a timestamped SVG and PNG under
``$ASSET_IMAGES_DIR/027-betaxanthin-metabolic/``.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
EXP_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic")
RESULTS_DIR = osp.join(EXP_DIR, "results")
SLURM_OUT_DIR = osp.join(EXP_DIR, "slurm", "output")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "027-betaxanthin-metabolic")

SEEDS = (2700, 2701, 2702)

#: arm label -> the result stem per seed. The two controls are wave 6 cells reused at the
#: same seeds and on the same kcat table, which is checked rather than assumed.
ARM_STEMS: dict[str, str] = {
    "fluxforce": "wave7_ff_fluxforce_k20_box_ue_nocstr_s{seed}",
    "tdg": "wave6_proj_flux_projection_anchored_tdg_ue_nocstr_s{seed}",
    "projection": "wave6_proj_flux_projection_anchored_k20_boxend_ue_nocstr_s{seed}",
}
ARM_CHANNEL: dict[str, str] = {
    "fluxforce": "fluxforce_k20_box",
    "tdg": "projection_k20_box_tdg",
    "projection": "projection_k20_box",
    "fluxforce_force_e200": "fluxforce_k20_force",
    "fluxforce_boxforce": "fluxforce_k100_boxforce",
}
#: Cells that are not a per-seed template. The 200-epoch pair at seed 2700 asks a different
#: question from the three-seed arms above: whether ending on the direction clip rather than
#: on the box clip recovers the second law the projection undoes, at a budget long enough
#: that neither cell is scored while its validation curve is still climbing.
EXTRA_STEMS: dict[str, dict[int, str]] = {
    "fluxforce_force_e200": {
        seed: f"wave7_ff_fluxforce_k20_force_e200_s{seed}" for seed in SEEDS
    },
    "fluxforce_boxforce": {
        seed: f"wave7_ff_fluxforce_k100_boxforce_e80_s{seed}" for seed in SEEDS
    },
}
#: The one kcat table every cell in this wave is allowed to have been built on.
REQUIRED_KCAT_TIER_COUNTS = {
    "measured": 97,
    "homolog": 782,
    "predicted": 2831,
    "default": 18,
}
ESCAPE_SD_RATIO = 0.05
ESCAPE_RUN_EPOCHS = 3
GENESIS_KEY = "both_balance_le_1e-02_and_thermo_model_eq_0"
TOP_GROUP = "all_top100"
ALT_GROUP = "pinned_test_top100"


class Cell(BaseModel):
    """One training run, reduced to the numbers this wave is decided on."""

    model_config = ConfigDict(extra="forbid")

    arm: str
    channel: str
    seed: int
    stem: str
    job_id: int | None = Field(description="From the slurm log that names this result.")
    wandb_run_id: str | None
    epochs_run: int
    select_epoch: int
    select_rule: str
    test_spearman_pinned: float
    n_pinned: int
    escape_epoch: int | None
    escape_sustained: bool
    val_pred_sd_ratio_last: float
    val_pred_sd_ratio_max: float
    val_pearson_ma5_max: float
    val_pearson_ma5_last: float
    still_rising: bool = Field(
        description=(
            "The validation moving average peaked inside the last ten epochs, so the "
            "budget, not the model, is what ended the run."
        )
    )
    kcat_tier_counts: dict[str, int]
    thermo_source: str
    parameterization_raw: str
    sign_flip_frac_last: float | None = Field(
        description="Mean over the last five epochs; None on an arm that has no such key."
    )
    g_diss_mean_last: float | None
    top100: dict[str, float]
    pinned_test_top100: dict[str, float]
    thermo_model_violation_median: float | None
    thermo_standard_state_violation_median: float | None
    balance_median_top100: float | None


class ArmSummary(BaseModel):
    """One arm over its seeds. n, mean and sd, never a single number on its own."""

    model_config = ConfigDict(extra="forbid")

    arm: str
    channel: str
    n: int
    seeds: list[int]
    job_ids: list[int | None]
    test_spearman_mean: float
    test_spearman_sd: float
    test_spearman_by_seed: dict[str, float]
    escaped: int
    escaped_sustained: int
    genesis_top100_mean: float
    balance_le_1e_02_top100_mean: float
    thermo_model_eq_0_top100_mean: float
    thermo_standard_state_eq_0_top100_mean: float
    sign_flip_frac_mean: float | None


class PairedContrast(BaseModel):
    """A paired difference over the seeds both arms ran, which is the only honest read."""

    model_config = ConfigDict(extra="forbid")

    arm: str
    against: str
    n_pairs: int
    seeds: list[int]
    mean_diff: float
    sd_diff: float
    diffs_by_seed: dict[str, float]


class Wave7Summary(BaseModel):
    """The whole wave, as it is written to disk."""

    model_config = ConfigDict(extra="forbid")

    built_at: str
    seeds: list[int]
    scoring: dict[str, str]
    required_kcat_tier_counts: dict[str, int]
    missing: list[str]
    cells: list[Cell]
    arms: list[ArmSummary]
    contrasts: list[PairedContrast]
    figure_svg: str
    figure_png: str


def _job_id_for(stem: str) -> int | None:
    """The slurm job whose log names this result file, or None if no log names it.

    Read from the logs rather than typed in, so a number in this summary cannot drift from
    the run that produced it.
    """
    if not osp.isdir(SLURM_OUT_DIR):
        return None
    needle = f"{stem}.json"
    best: int | None = None
    for name in os.listdir(SLURM_OUT_DIR):
        match = re.match(r"^.*_(\d+)\.out$", name)
        if match is None:
            continue
        with open(osp.join(SLURM_OUT_DIR, name), errors="replace") as handle:
            if needle in handle.read():
                job = int(match.group(1))
                best = job if best is None else max(best, job)
    return best


def _escape(history: list[dict[str, Any]]) -> tuple[int | None, bool]:
    """First epoch beginning a run of three above the threshold, and whether it held."""
    ratios = [float(row["val_pred_sd_ratio"]) for row in history]
    epoch: int | None = None
    for i in range(len(ratios) - ESCAPE_RUN_EPOCHS + 1):
        window = ratios[i : i + ESCAPE_RUN_EPOCHS]
        if all(r > ESCAPE_SD_RATIO for r in window):
            epoch = int(history[i]["epoch"])
            break
    tail = ratios[-ESCAPE_RUN_EPOCHS:]
    sustained = epoch is not None and all(r > ESCAPE_SD_RATIO for r in tail)
    return epoch, sustained


def _ma5(history: list[dict[str, Any]]) -> list[float]:
    """Centered 5-epoch moving average of the validation Pearson, the selection rule."""
    values = np.array([float(row["val_pearson"]) for row in history], dtype=float)
    out: list[float] = []
    for i in range(len(values)):
        lo, hi = max(0, i - 2), min(len(values), i + 3)
        out.append(float(values[lo:hi].mean()))
    return out


def _tail_mean(history: list[dict[str, Any]], key: str, n: int = 5) -> float | None:
    values = [float(row[key]) for row in history if key in row]
    if not values:
        return None
    return float(np.mean(values[-n:]))


def _stat(block: dict[str, Any], group: str, name: str) -> float | None:
    stats = block.get("groups", {}).get(group, {}).get("stats", {})
    entry = stats.get(name)
    if entry is None:
        return None
    return float(entry["median"]) if isinstance(entry, dict) else float(entry)


def _load(stem: str) -> dict[str, Any] | None:
    path = osp.join(RESULTS_DIR, f"{stem}.json")
    if not osp.isfile(path):
        return None
    with open(path) as handle:
        result: dict[str, Any] = json.load(handle)
    return result


def _cell(arm: str, seed: int, stem: str, result: dict[str, Any]) -> Cell:
    flux_meta = result["flux_meta"]
    counts = dict(flux_meta["kcat_tier_counts"])
    if counts != REQUIRED_KCAT_TIER_COUNTS:
        raise ValueError(
            f"{stem} was built on kcat tier counts {counts}, not the wave's table "
            f"{REQUIRED_KCAT_TIER_COUNTS}. Cells on different parameter tables are not "
            "comparable and are not pooled."
        )
    history = result["history"]
    escape_epoch, sustained = _escape(history)
    ma5 = _ma5(history)
    feasibility = result.get("feasibility", {})
    groups = feasibility.get("groups", {})
    return Cell(
        arm=arm,
        channel=ARM_CHANNEL[arm],
        seed=seed,
        stem=stem,
        job_id=_job_id_for(stem),
        wandb_run_id=result.get("wandb_run_id"),
        epochs_run=len(history),
        select_epoch=int(result["select_epoch"]),
        select_rule=str(result["select_rule"]),
        test_spearman_pinned=float(result["test"]["pinned"]["spearman"]),
        n_pinned=int(result["test"]["pinned"]["n"]),
        escape_epoch=escape_epoch,
        escape_sustained=sustained,
        val_pred_sd_ratio_last=float(history[-1]["val_pred_sd_ratio"]),
        val_pred_sd_ratio_max=max(float(r["val_pred_sd_ratio"]) for r in history),
        val_pearson_ma5_max=max(ma5),
        val_pearson_ma5_last=ma5[-1],
        still_rising=int(np.argmax(ma5)) >= len(ma5) - 10,
        kcat_tier_counts=counts,
        thermo_source=str(flux_meta["thermo_source"]),
        parameterization_raw=str(flux_meta["coverage"]["parameterization_raw"]),
        sign_flip_frac_last=_tail_mean(history, "feas_sign_flip_frac"),
        g_diss_mean_last=_tail_mean(history, "g_diss_mean")
        or _tail_mean(history, "g_diss"),
        top100=dict(groups.get(TOP_GROUP, {}).get("fractions", {})),
        pinned_test_top100=dict(groups.get(ALT_GROUP, {}).get("fractions", {})),
        thermo_model_violation_median=_stat(
            feasibility, TOP_GROUP, "thermo_model_violation_frac"
        ),
        thermo_standard_state_violation_median=_stat(
            feasibility, TOP_GROUP, "thermo_standard_state_violation_frac"
        ),
        balance_median_top100=_stat(feasibility, TOP_GROUP, "balance_median_ratio"),
    )


def _arm_summary(arm: str, cells: list[Cell]) -> ArmSummary:
    scores = np.array([c.test_spearman_pinned for c in cells], dtype=float)
    flips = [c.sign_flip_frac_last for c in cells if c.sign_flip_frac_last is not None]
    return ArmSummary(
        arm=arm,
        channel=ARM_CHANNEL[arm],
        n=len(cells),
        seeds=[c.seed for c in cells],
        job_ids=[c.job_id for c in cells],
        test_spearman_mean=float(scores.mean()),
        # The sample sd, ddof 1: three seeds are a sample, not the population.
        test_spearman_sd=float(scores.std(ddof=1)) if len(scores) > 1 else float("nan"),
        test_spearman_by_seed={str(c.seed): c.test_spearman_pinned for c in cells},
        escaped=sum(1 for c in cells if c.escape_epoch is not None),
        escaped_sustained=sum(1 for c in cells if c.escape_sustained),
        genesis_top100_mean=float(
            np.mean([c.top100.get(GENESIS_KEY, float("nan")) for c in cells])
        ),
        balance_le_1e_02_top100_mean=float(
            np.mean(
                [
                    c.top100.get("balance_median_ratio_le_1e-02", float("nan"))
                    for c in cells
                ]
            )
        ),
        thermo_model_eq_0_top100_mean=float(
            np.mean(
                [
                    c.top100.get("thermo_model_violation_frac_eq_0", float("nan"))
                    for c in cells
                ]
            )
        ),
        thermo_standard_state_eq_0_top100_mean=float(
            np.mean(
                [
                    c.top100.get(
                        "thermo_standard_state_violation_frac_eq_0", float("nan")
                    )
                    for c in cells
                ]
            )
        ),
        sign_flip_frac_mean=float(np.mean(flips)) if flips else None,
    )


def _contrast(arm: str, against: str, by_arm: dict[str, list[Cell]]) -> PairedContrast:
    left = {c.seed: c.test_spearman_pinned for c in by_arm[arm]}
    right = {c.seed: c.test_spearman_pinned for c in by_arm[against]}
    seeds = sorted(set(left) & set(right))
    diffs = np.array([left[s] - right[s] for s in seeds], dtype=float)
    return PairedContrast(
        arm=arm,
        against=against,
        n_pairs=len(seeds),
        seeds=seeds,
        mean_diff=float(diffs.mean()) if len(diffs) else float("nan"),
        sd_diff=float(diffs.std(ddof=1)) if len(diffs) > 1 else float("nan"),
        diffs_by_seed={str(s): float(d) for s, d in zip(seeds, diffs, strict=True)},
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }
    )


def _box(ax: Any) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)


def make_figure(cells: list[Cell], arms: list[ArmSummary]) -> tuple[str, str]:
    """Four panels: signal, escape, the Genesis criteria, and what the projection undoes."""
    _style()
    order = [a.arm for a in arms]
    colors = {arm: PLOT_PALETTE[i] for i, arm in enumerate(order)}
    width_mm = PANEL_WIDTHS_MM["full"]
    fig, axes = plt.subplots(
        2, 2, figsize=(mm_to_in(width_mm), mm_to_in(110.0)), constrained_layout=True
    )

    ax = axes[0][0]
    bar_w = 0.8 / max(len(order), 1)
    for i, arm in enumerate(order):
        by_seed = {c.seed: c.test_spearman_pinned for c in cells if c.arm == arm}
        xs = np.arange(len(SEEDS)) + i * bar_w - 0.4 + bar_w / 2
        ys = [by_seed.get(s, np.nan) for s in SEEDS]
        ax.bar(
            xs,
            ys,
            width=bar_w,
            color=colors[arm],
            edgecolor="black",
            linewidth=0.5,
            label=arm,
        )
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xticks(np.arange(len(SEEDS)))
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel("seed")
    ax.set_ylabel("test Spearman, 629 pinned genes")
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=len(order),
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
    )
    _box(ax)

    ax = axes[0][1]
    for i, arm in enumerate(order):
        cell_by_seed = {c.seed: c for c in cells if c.arm == arm}
        xs = np.arange(len(SEEDS)) + i * bar_w - 0.4 + bar_w / 2
        heights = [
            (
                cell_by_seed[s].escape_epoch
                if s in cell_by_seed and cell_by_seed[s].escape_epoch is not None
                else 0
            )
            for s in SEEDS
        ]
        # Hatched where the escape did not hold to the last three epochs: a run that
        # escapes the mean plateau and falls back has not escaped it.
        hatches = [
            "" if (s in cell_by_seed and cell_by_seed[s].escape_sustained) else "///"
            for s in SEEDS
        ]
        for x, h, hatch in zip(xs, heights, hatches, strict=True):
            ax.bar(
                x,
                h,
                width=bar_w,
                color=colors[arm],
                edgecolor="black",
                linewidth=0.5,
                hatch=hatch,
            )
    ax.set_xticks(np.arange(len(SEEDS)))
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel("seed")
    ax.set_ylabel("escape epoch (hatched: not sustained)")
    _box(ax)

    ax = axes[1][0]
    criteria = [
        ("balance_median_ratio_le_1e-02", "balance <= 1e-2"),
        ("thermo_model_violation_frac_eq_0", "second law = 0"),
        (GENESIS_KEY, "both (Genesis)"),
    ]
    for i, arm in enumerate(order):
        values = [
            float(np.mean([c.top100.get(k, np.nan) for c in cells if c.arm == arm]))
            for k, _ in criteria
        ]
        xs = np.arange(len(criteria)) + i * bar_w - 0.4 + bar_w / 2
        ax.bar(
            xs, values, width=bar_w, color=colors[arm], edgecolor="black", linewidth=0.5
        )
    ax.set_xticks(np.arange(len(criteria)))
    ax.set_xticklabels([label for _, label in criteria])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.set_ylabel("fraction of the top 100 predicted strains")
    _box(ax)

    ax = axes[1][1]
    for i, arm in enumerate(order):
        flips = {
            c.seed: c.sign_flip_frac_last
            for c in cells
            if c.arm == arm and c.sign_flip_frac_last is not None
        }
        if not flips:
            continue
        xs = np.arange(len(SEEDS)) + i * bar_w - 0.4 + bar_w / 2
        ax.bar(
            xs,
            [flips.get(s, np.nan) for s in SEEDS],
            width=bar_w,
            color=colors[arm],
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(np.arange(len(SEEDS)))
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel("seed")
    ax.set_ylabel("sign flips introduced by the projection")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    _box(ax)

    for letter, panel in zip("abcd", axes.ravel(), strict=True):
        panel.text(
            -0.16,
            1.04,
            letter,
            transform=panel.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    stamp = timestamp()
    os.makedirs(IMAGE_DIR, exist_ok=True)
    svg = osp.join(IMAGE_DIR, f"wave7_fluxforce_{stamp}.svg")
    png = osp.join(IMAGE_DIR, f"wave7_fluxforce_{stamp}.png")
    savefig_true_size_svg(fig, svg)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return svg, png


def main() -> None:
    by_arm: dict[str, list[Cell]] = {}
    cells: list[Cell] = []
    missing: list[str] = []
    plan: list[tuple[str, int, str]] = [
        (arm, seed, template.format(seed=seed))
        for arm, template in ARM_STEMS.items()
        for seed in SEEDS
    ]
    plan += [
        (arm, seed, stem)
        for arm, per_seed in EXTRA_STEMS.items()
        for seed, stem in per_seed.items()
    ]
    for arm, seed, stem in [(a, s, st) for a, s, st in plan]:
        if True:
            result = _load(stem)
            if result is None:
                missing.append(stem)
                continue
            cell = _cell(arm, seed, stem, result)
            cells.append(cell)
            by_arm.setdefault(arm, []).append(cell)
    if not cells:
        raise SystemExit(f"no result files found; missing {missing}")

    arms = [
        _arm_summary(arm, by_arm[arm])
        for arm in list(ARM_STEMS) + list(EXTRA_STEMS)
        if arm in by_arm
    ]
    contrasts = [
        _contrast(a, b, by_arm)
        for a, b in (
            ("fluxforce", "tdg"),
            ("fluxforce", "projection"),
            ("tdg", "projection"),
            # The end-on-clip variant against the box end, against the sign clamp whose
            # exactness it matches, and against the plain projection whose escape it keeps.
            ("fluxforce_force_e200", "fluxforce"),
            ("fluxforce_force_e200", "tdg"),
            ("fluxforce_force_e200", "projection"),
            ("fluxforce_boxforce", "fluxforce_force_e200"),
            ("fluxforce_boxforce", "tdg"),
        )
        if a in by_arm and b in by_arm
    ]
    svg, png = make_figure(cells, arms)
    summary = Wave7Summary(
        built_at=timestamp(),
        seeds=list(SEEDS),
        scoring={
            "signal": (
                "test Spearman on the 629 pinned Merzbacher genes at the epoch the "
                "trainer selected by a centered 5-epoch moving average of the validation "
                "Pearson; never a maximum over epochs"
            ),
            "escape": (
                f"first epoch beginning {ESCAPE_RUN_EPOCHS} consecutive epochs with "
                f"val_pred_sd_ratio above {ESCAPE_SD_RATIO}; sustained means the last "
                f"{ESCAPE_RUN_EPOCHS} epochs are also above it"
            ),
            "feasibility": (
                f"the trainer's own block at the {TOP_GROUP} group; the Genesis key is "
                f"{GENESIS_KEY}"
            ),
            "sign_flips": (
                "mean over the last five epochs of feas_sign_flip_frac, the fraction of "
                "flux-carrying priced reactions whose sign the projection reversed"
            ),
        },
        required_kcat_tier_counts=REQUIRED_KCAT_TIER_COUNTS,
        missing=missing,
        cells=cells,
        arms=arms,
        contrasts=contrasts,
        figure_svg=svg,
        figure_png=png,
    )
    out = osp.join(RESULTS_DIR, "wave7_fluxforce_summary.json")
    with open(out, "w") as handle:
        handle.write(summary.model_dump_json(indent=2))

    print(
        f"{'arm':>12} {'n':>2} {'mean':>8} {'sd':>7} {'esc':>5} {'sust':>5} "
        f"{'genesis':>8} {'bal':>6} {'2law':>6} {'flip':>6}"
    )
    for a in arms:
        flip = (
            "n/a" if a.sign_flip_frac_mean is None else f"{a.sign_flip_frac_mean:.3f}"
        )
        print(
            f"{a.arm:>12} {a.n:>2} {a.test_spearman_mean:>8.4f} "
            f"{a.test_spearman_sd:>7.4f} {a.escaped:>5} {a.escaped_sustained:>5} "
            f"{a.genesis_top100_mean:>8.2f} {a.balance_le_1e_02_top100_mean:>6.2f} "
            f"{a.thermo_model_eq_0_top100_mean:>6.2f} {flip:>6}"
        )
    for c in contrasts:
        print(
            f"paired {c.arm} - {c.against}: n={c.n_pairs} mean={c.mean_diff:+.4f} "
            f"sd={c.sd_diff:.4f}"
        )
    if missing:
        print("missing:", ", ".join(missing))
    print("wrote", out)
    print("figure", svg)


if __name__ == "__main__":
    main()
