# experiments/027-betaxanthin-metabolic/scripts/analyze_wave8_density.py
# [[experiments.027-betaxanthin-metabolic.wave7-flux-force-head]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/analyze_wave8_density
"""Wave 8 block 1: does pricing dead rows buy a Genesis pass that is not about zeros?

THE ARM AND ITS CONTROL. `fluxforce_k100_boxforce` at density weight 0.5 and 2.0 against
the same channel at weight 0.0 (jobs 1767, 1770, 1771), paired by seed, everything else
held: 80 epochs, constraint scale 0.0, union_external deltaG, the shipped verified kcat
table (97/782/2831/18).

ALL THREE BALANCE RULES ARE REPORTED, because the wave-7 verdict turned on which one is
asked. The trainer's own block gives the MEDIAN rule; the MAX rule and the
TURNOVER-THRESHOLDED rule come from `audit_top_strain_feasibility.py`'s strict block, so a
cell contributes those two only once its audit has been run. The audit is also the only
place `flux_active_frac` and `balance_active_rows` are computed, and a rule evaluated over
a handful of rows is not a statement about a flux vector, so the row count travels with
every thresholded ratio.

Run from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/analyze_wave8_density.py

Writes ``results/wave8_density_summary.json`` and a timestamped SVG and PNG under
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
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
EXP_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic")
RESULTS_DIR = osp.join(EXP_DIR, "results")
SLURM_OUT_DIR = osp.join(EXP_DIR, "slurm", "output")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "027-betaxanthin-metabolic")

SEEDS = (2700, 2701, 2702)
REQUIRED_KCAT_TIER_COUNTS = {
    "measured": 97,
    "homolog": 782,
    "predicted": 2831,
    "default": 18,
}
#: arm label -> (result stem template, audit stem template, density weight). The audit is
#: optional: a cell without one reports the median rule and says the other two are absent
#: rather than reporting them as failures.
ARMS: dict[str, tuple[str, str, float]] = {
    "density_0.0": (
        "wave7_ff_fluxforce_k100_boxforce_e80_s{seed}",
        "audit_wave7_ff_boxforce_k100_s{seed}_activerule",
        0.0,
    ),
    "density_0.5": (
        "wave8_dens_d0p5_boxforce_k100_e80_s{seed}",
        "audit_wave8_dens_d0p5_s{seed}",
        0.5,
    ),
    "density_2.0": (
        "wave8_dens_d2p0_boxforce_k100_e80_s{seed}",
        "audit_wave8_dens_d2p0_s{seed}",
        2.0,
    ),
}
ESCAPE_SD_RATIO = 0.05
ESCAPE_RUN_EPOCHS = 3
MEDIAN_RULE = "genesis_median_rule_thermo_model"
MAX_RULE = "genesis_max_rule_thermo_model"
ACTIVE_RULE = "genesis_active_max_1e-03_rule_thermo_model"
TRAINER_MEDIAN_KEY = "both_balance_le_1e-02_and_thermo_model_eq_0"
TOP_GROUP = "all_top100"


class Cell(BaseModel):
    """One density cell, under every rule it has evidence for."""

    model_config = ConfigDict(extra="forbid")

    arm: str
    density_weight: float
    seed: int
    stem: str
    job_id: int | None
    select_epoch: int
    epochs_run: int
    test_spearman_pinned: float
    escape_epoch: int | None
    escape_sustained: bool
    val_pred_sd_ratio_max: float
    c_density_first: float | None
    c_density_last: float | None
    active_row_frac_first: float | None
    active_row_frac_last: float | None
    genesis_median_rule: float
    genesis_max_rule: float | None = Field(
        description="From the audit's strict block; None when no audit has been run."
    )
    genesis_active_rule: float | None
    flux_active_frac: float | None
    balance_active_rows_1e_03: float | None
    balance_active_max_ratio_1e_03: float | None
    balance_active_p95_ratio_1e_03: float | None
    balance_max_ratio: float | None
    second_law_eq_0_frac: float
    sign_flip_frac_last: float | None


class ArmSummary(BaseModel):
    """One density weight over its seeds."""

    model_config = ConfigDict(extra="forbid")

    arm: str
    density_weight: float
    n: int
    seeds: list[int]
    job_ids: list[int | None]
    test_spearman_mean: float
    test_spearman_sd: float
    test_spearman_by_seed: dict[str, float]
    escaped_sustained: int
    genesis_median_rule_mean: float
    genesis_max_rule_mean: float | None
    genesis_active_rule_mean: float | None
    flux_active_frac_mean: float | None
    active_row_frac_last_mean: float | None


class Contrast(BaseModel):
    """A paired difference against the density-0 control."""

    model_config = ConfigDict(extra="forbid")

    arm: str
    against: str
    n_pairs: int
    seeds: list[int]
    mean_diff: float
    sd_diff: float
    diffs_by_seed: dict[str, float]


class Wave8Summary(BaseModel):
    """The block, as written to disk."""

    model_config = ConfigDict(extra="forbid")

    built_at: str
    seeds: list[int]
    rules: dict[str, str]
    missing_results: list[str]
    missing_audits: list[str]
    cells: list[Cell]
    arms: list[ArmSummary]
    contrasts: list[Contrast]
    figure_svg: str
    figure_png: str


def _job_id_for(stem: str) -> int | None:
    """The slurm job whose log names this result, read from the logs rather than typed."""
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
    ratios = [float(row["val_pred_sd_ratio"]) for row in history]
    epoch: int | None = None
    for i in range(len(ratios) - ESCAPE_RUN_EPOCHS + 1):
        if all(r > ESCAPE_SD_RATIO for r in ratios[i : i + ESCAPE_RUN_EPOCHS]):
            epoch = int(history[i]["epoch"])
            break
    sustained = epoch is not None and all(
        r > ESCAPE_SD_RATIO for r in ratios[-ESCAPE_RUN_EPOCHS:]
    )
    return epoch, sustained


def _series(
    history: list[dict[str, Any]], key: str
) -> tuple[float | None, float | None]:
    values = [float(row[key]) for row in history if key in row]
    if not values:
        return None, None
    return values[0], values[-1]


def _load(stem: str) -> dict[str, Any] | None:
    path = osp.join(RESULTS_DIR, f"{stem}.json")
    if not osp.isfile(path):
        return None
    with open(path) as handle:
        payload: dict[str, Any] = json.load(handle)
    return payload


def _strict(audit: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if audit is None:
        return {}, {}
    group = audit["strict_feasibility"]["groups"][TOP_GROUP]
    return group["fractions"], group["stats"]


def _median_stat(stats: dict[str, Any], key: str) -> float | None:
    entry = stats.get(key)
    return None if entry is None else float(entry["median"])


def _cell(
    arm: str, weight: float, seed: int, stem: str, audit_stem: str
) -> Cell | None:
    result = _load(stem)
    if result is None:
        return None
    counts = dict(result["flux_meta"]["kcat_tier_counts"])
    if counts != REQUIRED_KCAT_TIER_COUNTS:
        raise ValueError(
            f"{stem} was built on kcat tier counts {counts}, not the wave's table "
            f"{REQUIRED_KCAT_TIER_COUNTS}; cells on different tables are not pooled."
        )
    history = result["history"]
    escape_epoch, sustained = _escape(history)
    fractions, stats = _strict(_load(audit_stem))
    trainer = result["feasibility"]["groups"][TOP_GROUP]["fractions"]
    c_first, c_last = _series(history, "c_density")
    a_first, a_last = _series(history, "feas_active_row_frac")
    _, flip_last = _series(history, "feas_sign_flip_frac")
    return Cell(
        arm=arm,
        density_weight=weight,
        seed=seed,
        stem=stem,
        job_id=_job_id_for(stem),
        select_epoch=int(result["select_epoch"]),
        epochs_run=len(history),
        test_spearman_pinned=float(result["test"]["pinned"]["spearman"]),
        escape_epoch=escape_epoch,
        escape_sustained=sustained,
        val_pred_sd_ratio_max=max(float(r["val_pred_sd_ratio"]) for r in history),
        c_density_first=c_first,
        c_density_last=c_last,
        active_row_frac_first=a_first,
        active_row_frac_last=a_last,
        genesis_median_rule=float(trainer[TRAINER_MEDIAN_KEY]),
        genesis_max_rule=(
            None if MAX_RULE not in fractions else float(fractions[MAX_RULE])
        ),
        genesis_active_rule=(
            None if ACTIVE_RULE not in fractions else float(fractions[ACTIVE_RULE])
        ),
        flux_active_frac=_median_stat(stats, "flux_active_frac"),
        balance_active_rows_1e_03=_median_stat(stats, "balance_active_rows_1e-03"),
        balance_active_max_ratio_1e_03=_median_stat(
            stats, "balance_active_max_ratio_1e-03"
        ),
        balance_active_p95_ratio_1e_03=_median_stat(
            stats, "balance_active_p95_ratio_1e-03"
        ),
        balance_max_ratio=_median_stat(stats, "balance_max_ratio"),
        second_law_eq_0_frac=float(trainer["thermo_model_violation_frac_eq_0"]),
        sign_flip_frac_last=flip_last,
    )


def _mean_or_none(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None


def _arm_summary(arm: str, weight: float, cells: list[Cell]) -> ArmSummary:
    scores = np.array([c.test_spearman_pinned for c in cells], dtype=float)
    return ArmSummary(
        arm=arm,
        density_weight=weight,
        n=len(cells),
        seeds=[c.seed for c in cells],
        job_ids=[c.job_id for c in cells],
        test_spearman_mean=float(scores.mean()),
        test_spearman_sd=float(scores.std(ddof=1)) if len(scores) > 1 else float("nan"),
        test_spearman_by_seed={str(c.seed): c.test_spearman_pinned for c in cells},
        escaped_sustained=sum(1 for c in cells if c.escape_sustained),
        genesis_median_rule_mean=float(np.mean([c.genesis_median_rule for c in cells])),
        genesis_max_rule_mean=_mean_or_none([c.genesis_max_rule for c in cells]),
        genesis_active_rule_mean=_mean_or_none([c.genesis_active_rule for c in cells]),
        flux_active_frac_mean=_mean_or_none([c.flux_active_frac for c in cells]),
        active_row_frac_last_mean=_mean_or_none(
            [c.active_row_frac_last for c in cells]
        ),
    )


def _contrast(arm: str, against: str, by_arm: dict[str, list[Cell]]) -> Contrast:
    left = {c.seed: c.test_spearman_pinned for c in by_arm[arm]}
    right = {c.seed: c.test_spearman_pinned for c in by_arm[against]}
    seeds = sorted(set(left) & set(right))
    diffs = np.array([left[s] - right[s] for s in seeds], dtype=float)
    return Contrast(
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
    """Signal, the three rules, and what the term actually moved."""
    _style()
    order = [a.arm for a in arms]
    colors = {arm: PLOT_PALETTE[i] for i, arm in enumerate(order)}
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(52.0)),
        constrained_layout=True,
    )
    bar_w = 0.8 / max(len(order), 1)

    ax = axes[0]
    for i, arm in enumerate(order):
        by_seed = {c.seed: c.test_spearman_pinned for c in cells if c.arm == arm}
        xs = np.arange(len(SEEDS)) + i * bar_w - 0.4 + bar_w / 2
        ax.bar(
            xs,
            [by_seed.get(s, np.nan) for s in SEEDS],
            width=bar_w,
            color=colors[arm],
            edgecolor="black",
            linewidth=0.5,
            label=arm.replace("density_", "weight "),
        )
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xticks(np.arange(len(SEEDS)))
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel("seed")
    ax.set_ylabel("test Spearman, 629 pinned genes")
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

    ax = axes[1]
    rules = [
        ("genesis_median_rule_mean", "median"),
        ("genesis_active_rule_mean", "turnover 1e-3"),
        ("genesis_max_rule_mean", "max"),
    ]
    for i, arm in enumerate(order):
        summary = next(a for a in arms if a.arm == arm)
        values = [getattr(summary, key) for key, _ in rules]
        xs = np.arange(len(rules)) + i * bar_w - 0.4 + bar_w / 2
        ax.bar(
            xs,
            [np.nan if v is None else v for v in values],
            width=bar_w,
            color=colors[arm],
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(np.arange(len(rules)))
    ax.set_xticklabels([label for _, label in rules])
    ax.set_xlabel("balance rule")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.set_ylabel("Genesis pass, top 100")
    _box(ax)

    ax = axes[2]
    for i, arm in enumerate(order):
        by_seed = {
            c.seed: c.active_row_frac_last
            for c in cells
            if c.arm == arm and c.active_row_frac_last is not None
        }
        if not by_seed:
            continue
        xs = np.arange(len(SEEDS)) + i * bar_w - 0.4 + bar_w / 2
        ax.bar(
            xs,
            [by_seed.get(s, np.nan) for s in SEEDS],
            width=bar_w,
            color=colors[arm],
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(np.arange(len(SEEDS)))
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel("seed")
    ax.set_ylabel("independent rows turning over")
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    _box(ax)

    for letter, panel in zip("abc", axes, strict=True):
        panel.text(
            -0.22,
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
    svg = osp.join(IMAGE_DIR, f"wave8_density_{stamp}.svg")
    png = osp.join(IMAGE_DIR, f"wave8_density_{stamp}.png")
    savefig_true_size_svg(fig, svg)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return svg, png


def main() -> None:
    cells: list[Cell] = []
    by_arm: dict[str, list[Cell]] = {}
    missing_results: list[str] = []
    missing_audits: list[str] = []
    for arm, (stem_tpl, audit_tpl, weight) in ARMS.items():
        for seed in SEEDS:
            stem = stem_tpl.format(seed=seed)
            audit_stem = audit_tpl.format(seed=seed)
            cell = _cell(arm, weight, seed, stem, audit_stem)
            if cell is None:
                missing_results.append(stem)
                continue
            if cell.genesis_active_rule is None:
                missing_audits.append(audit_stem)
            cells.append(cell)
            by_arm.setdefault(arm, []).append(cell)
    if not cells:
        raise SystemExit(f"no result files found; missing {missing_results}")

    arms = [
        _arm_summary(arm, ARMS[arm][2], by_arm[arm]) for arm in ARMS if arm in by_arm
    ]
    contrasts = [
        _contrast(arm, "density_0.0", by_arm)
        for arm in ("density_0.5", "density_2.0")
        if arm in by_arm and "density_0.0" in by_arm
    ]
    svg, png = make_figure(cells, arms)
    summary = Wave8Summary(
        built_at=timestamp(),
        seeds=list(SEEDS),
        rules={
            "median": (
                "the trainer's own criterion, balance_median_ratio <= 1e-2 and no active "
                "reaction uphill; a mostly-off flux passes it on the zeros"
            ),
            "max": (
                "balance_max_ratio <= 1e-2, every independent row; reads 2.0, the "
                "statistic's largest value, on every head measured in waves 6 to 8"
            ),
            "turnover_1e-3": (
                "the same ratio over the rows whose turnover exceeds 1e-3 * flux_scale, "
                "reported beside the count of rows that clear the threshold"
            ),
            "scoring": (
                "test Spearman on the 629 pinned Merzbacher genes at the epoch the "
                "trainer selected by a centered 5-epoch moving average of the validation "
                "Pearson, never a maximum over epochs"
            ),
        },
        missing_results=missing_results,
        missing_audits=missing_audits,
        cells=cells,
        arms=arms,
        contrasts=contrasts,
        figure_svg=svg,
        figure_png=png,
    )
    out = osp.join(RESULTS_DIR, "wave8_density_summary.json")
    with open(out, "w") as handle:
        handle.write(summary.model_dump_json(indent=2))

    print(
        f"{'arm':>12} {'n':>2} {'mean':>8} {'sd':>7} {'sust':>5} "
        f"{'median':>7} {'active':>7} {'max':>5} {'rows%':>7} {'flux%':>7}"
    )
    for a in arms:

        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.2f}"

        print(
            f"{a.arm:>12} {a.n:>2} {a.test_spearman_mean:>8.4f} "
            f"{a.test_spearman_sd:>7.4f} {a.escaped_sustained:>5} "
            f"{a.genesis_median_rule_mean:>7.2f} {fmt(a.genesis_active_rule_mean):>7} "
            f"{fmt(a.genesis_max_rule_mean):>5} "
            f"{fmt(a.active_row_frac_last_mean):>7} {fmt(a.flux_active_frac_mean):>7}"
        )
    for c in contrasts:
        print(
            f"paired {c.arm} - {c.against}: n={c.n_pairs} mean={c.mean_diff:+.4f} "
            f"sd={c.sd_diff:.4f}"
        )
    if missing_results:
        print("missing results:", ", ".join(missing_results))
    if missing_audits:
        print(
            "missing audits (max and turnover rules absent):", ", ".join(missing_audits)
        )
    print("wrote", out)
    print("figure", svg)


if __name__ == "__main__":
    main()
