# experiments/027-betaxanthin-metabolic/scripts/analyze_wave7_parity.py
# [[experiments.027-betaxanthin-metabolic.wave7-parity-one-table]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/analyze_wave7_parity.py

r"""The betaxanthin parity contrast on ONE kcat table, and the honest feasibility rules.

WHY WAVE 7. Wave 6 measured the box flux head at constraint scale 0 against the pooled
control over six seeds and read -0.017, sd 0.046. Three different kcat tables sat under
those six cells, because the resolver reads the predicted-parameter mirror off ``$DATA_ROOT``
and that mirror moved while the queue drained. None of the three is the shipped verified
table (97 measured / 782 homolog / 2,831 predicted / 18 default), so the wave-6 "same table"
subset is a subset of the WAVE-5 table, not of the shipped one. Wave 7 reruns the same
recipe at the same seeds with every flux cell on the shipped table, verified before
submission by ``verify_kcat_table.py`` and checked again here against the tier counts each
cell recorded for itself.

THE SCORING RULE, fixed before the cells ran and identical to wave 6: the pinned Merzbacher
test Spearman at the epoch chosen by the centered five-epoch moving average of validation
Pearson. Never a maximum over epochs.

TWO FEASIBILITY RULES. The trainer's Genesis criterion uses the MEDIAN relative balance
residual over the independent rows of S, which a flux vector that is off on most reactions
passes on its zeros. ``audit_top_strain_feasibility.py`` now also reports the max over the
same rows and the fraction of reactions carrying flux; this script tabulates both from the
audit files, so the "met once" result can be read under either rule.

Run from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/analyze_wave7_parity.py

Writes ``results/wave7_parity_summary.json`` and one timestamped figure into
``$ASSET_IMAGES_DIR/027-betaxanthin-metabolic/``. Only finished result files are read; a
cell that has not written one is listed under ``missing_files`` by name.
"""

import json
import os
import os.path as osp
import re
import sys
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

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from analyze_wave6 import physics_key  # noqa: E402

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXP_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic")
RESULTS = osp.join(EXP_DIR, "results")
SLURM_OUT = osp.join(EXP_DIR, "slurm", "output")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "027-betaxanthin-metabolic")

SEEDS = (2700, 2701, 2702, 2703, 2704, 2705)
WAVE7_ARMS = {
    "nocstr": "wave7_parity_nocstr_s{seed}.json",
    "pooled": "wave7_parity_pooled_s{seed}.json",
}
WAVE6_ARMS = {
    "nocstr": "wave5_g7_flux_capacity_nocstr_s{seed}.json",
    "pooled": "wave5_g8_pooled_s{seed}.json",
}

#: The shipped verified kcat table every wave-7 flux cell must carry.
SHIPPED_TIER_COUNTS = {"measured": 97, "homolog": 782, "predicted": 2831, "default": 18}

#: The wave-6 projection cells the feasibility rules are re-read on: the plain projection,
#: the sign-tightened one (tdg), and the 5 kJ/mol margin (tdgm5). The stochastic cell is the
#: one that met the Genesis criterion under the median rule (job 1725).
AUDIT_CELLS: tuple[str, ...] = (
    "wave6_proj_flux_projection_s2700",
    "wave6_proj_flux_projection_s2701",
    "wave6_proj_flux_projection_s2702",
    "wave6_proj_flux_projection_anchored_tdg_ue_nocstr_s2700",
    "wave6_proj_flux_projection_anchored_tdg_ue_nocstr_s2701",
    "wave6_proj_flux_projection_anchored_tdg_ue_nocstr_s2702",
    "wave6_proj_flux_projection_anchored_tdg_stoch_ue_s2700",
    "wave6_proj_flux_projection_anchored_tdgm5_ue_nocstr_s2700",
    "wave6_proj_flux_projection_anchored_tdgm5_ue_nocstr_s2701",
    "wave7_parity_nocstr_s2700",
    "wave7_parity_nocstr_s2701",
    "wave7_parity_nocstr_s2702",
    "wave7_parity_nocstr_s2703",
    "wave7_parity_nocstr_s2704",
    "wave7_parity_nocstr_s2705",
    "wave7_ff_fluxforce_k100_boxforce_e80_s2700",
    "wave7_ff_fluxforce_k100_boxforce_e80_s2701",
    "wave7_ff_fluxforce_k100_boxforce_e80_s2702",
    "wave7_ff_fluxforce_k20_force_e200_s2700",
    "wave7_ff_fluxforce_k20_force_e200_s2701",
    "wave7_ff_fluxforce_k20_force_e200_s2702",
)

#: How each audited cell is named in the table and on the axis.
AUDIT_LABELS: dict[str, str] = {
    "wave6_proj_flux_projection_s2700": "proj plain s2700",
    "wave6_proj_flux_projection_s2701": "proj plain s2701",
    "wave6_proj_flux_projection_s2702": "proj plain s2702",
    "wave6_proj_flux_projection_anchored_tdg_ue_nocstr_s2700": "proj tdg s2700",
    "wave6_proj_flux_projection_anchored_tdg_ue_nocstr_s2701": "proj tdg s2701",
    "wave6_proj_flux_projection_anchored_tdg_ue_nocstr_s2702": "proj tdg s2702",
    "wave6_proj_flux_projection_anchored_tdg_stoch_ue_s2700": "proj tdg stoch s2700",
    "wave6_proj_flux_projection_anchored_tdgm5_ue_nocstr_s2700": "proj tdgm5 s2700",
    "wave6_proj_flux_projection_anchored_tdgm5_ue_nocstr_s2701": "proj tdgm5 s2701",
    "wave7_parity_nocstr_s2700": "w7 box s2700",
    "wave7_parity_nocstr_s2701": "w7 box s2701",
    "wave7_parity_nocstr_s2702": "w7 box s2702",
    "wave7_parity_nocstr_s2703": "w7 box s2703",
    "wave7_parity_nocstr_s2704": "w7 box s2704",
    "wave7_parity_nocstr_s2705": "w7 box s2705",
    "wave7_ff_fluxforce_k100_boxforce_e80_s2700": "boxforce s2700",
    "wave7_ff_fluxforce_k100_boxforce_e80_s2701": "boxforce s2701",
    "wave7_ff_fluxforce_k100_boxforce_e80_s2702": "boxforce s2702",
    "wave7_ff_fluxforce_k20_force_e200_s2700": "force end s2700",
    "wave7_ff_fluxforce_k20_force_e200_s2701": "force end s2701",
    "wave7_ff_fluxforce_k20_force_e200_s2702": "force end s2702",
}

#: The turnover threshold whose thresholded balance rule is the primary one, matching
#: ``audit_top_strain_feasibility.PRIMARY_TURNOVER_THRESHOLD``. The other two thresholds are
#: carried through into the JSON as the sensitivity.
PRIMARY_TURNOVER_TAG = "1e-03"
TURNOVER_TAGS = ("1e-02", "1e-03", "1e-04")


class CellRecord(BaseModel):
    """One finished cell, with what it was built from and what it scored."""

    model_config = ConfigDict(extra="forbid")

    cell: str
    wave: str
    arm: str
    seed: int
    select_epoch: int
    select_rule: str
    pinned_test_spearman: float
    pinned_test_pearson: float
    n_pinned: int
    final_val_pred_sd_ratio: float
    epochs_run: int
    kcat_tier_counts: dict[str, int] | None
    on_shipped_table: bool | None
    physics_key: str | None
    code_provenance: dict[str, str] | None
    checkpoint_path: str | None
    wandb_url: str | None
    slurm_job_ids: list[int] = Field(default_factory=list)


class PairedContrast(BaseModel):
    """One paired-by-seed contrast of the pinned test Spearman."""

    model_config = ConfigDict(extra="forbid")

    name: str
    scoring_rule: str = (
        "pinned Merzbacher test Spearman at the epoch selected by the centered 5-epoch "
        "moving average of validation Pearson; never a maximum over epochs."
    )
    seeds: list[int]
    flux_values: list[float]
    pooled_values: list[float]
    differences: list[float]
    n_pairs: int
    mean_difference: float
    sd_difference: float
    t_statistic: float | None
    flux_mean: float
    flux_sd: float
    pooled_mean: float
    pooled_sd: float
    note: str = ""


class FeasibilityRuleRow(BaseModel):
    """One audited cell under the median rule and the max rule."""

    model_config = ConfigDict(extra="forbid")

    cell: str
    label: str
    slurm_job_ids: list[int] = Field(default_factory=list)
    group: str
    n: int
    balance_median_ratio_median: float
    balance_max_ratio_median: float
    flux_active_frac_median: float
    flux_active_frac_p90: float
    balance_median_le_1e_2: float
    balance_max_le_1e_2: float
    thermo_standard_state_no_uphill: float
    genesis_median_rule: float
    genesis_max_rule: float
    genesis_median_rule_model_thermo: float | None = None
    genesis_max_rule_model_thermo: float | None = None
    #: The turnover-thresholded rule at the primary threshold: the ratio restricted to the
    #: independent rows whose turnover exceeds 1e-3 of the flux scale.
    balance_active_max_ratio_median: float | None = None
    balance_active_p95_ratio_median: float | None = None
    balance_active_rows_median: float | None = None
    balance_active_max_le_1e_2: float | None = None
    genesis_active_rule: float | None = None
    genesis_active_rule_model_thermo: float | None = None
    #: Every threshold, so the primary is readable against its sensitivity: tag -> value.
    balance_active_max_ratio_by_threshold: dict[str, float] = Field(
        default_factory=dict
    )
    balance_active_p95_ratio_by_threshold: dict[str, float] = Field(
        default_factory=dict
    )
    balance_active_rows_by_threshold: dict[str, float] = Field(default_factory=dict)
    genesis_active_by_threshold: dict[str, float] = Field(default_factory=dict)
    agreement_max_abs_delta: float | None = None


class Wave7Summary(BaseModel):
    """Everything the wave-7 note reports."""

    model_config = ConfigDict(extra="forbid")

    generated_at: str
    kcat_table_verification: dict[str, Any] | None
    expected_tier_counts: dict[str, int]
    wave7: PairedContrast
    wave7_off_table_seeds: dict[int, str] = Field(default_factory=dict)
    #: The wave-6 pairs whose FLUX cell also carries the shipped table, and the pooled
    #: contrast over wave 7 plus those. Empty when no wave-6 cell is on the shipped table,
    #: which is what the wave-6 result files say.
    wave6_shipped_table_seeds: list[int] = Field(default_factory=list)
    combined: PairedContrast
    #: Wave 7 pooled with the wave-6 seeds that share ONE table with each other. That table
    #: is the wave-5 one, not the shipped one, so this contrast spans two parameter tables
    #: and is reported as such rather than as a six-seed same-table result.
    combined_with_wave6_same_table_group: PairedContrast
    wave6_reference: PairedContrast
    wave6_by_table: dict[str, PairedContrast] = Field(default_factory=dict)
    #: Same seed, wave 6's table against wave 7's shipped table, flux arm and pooled arm.
    same_seed_table_shift: dict[int, dict[str, float]] = Field(default_factory=dict)
    cells: list[CellRecord] = Field(default_factory=list)
    feasibility: list[FeasibilityRuleRow] = Field(default_factory=list)
    missing_files: list[str] = Field(default_factory=list)
    missing_audits: list[str] = Field(default_factory=list)
    figure_path: str | None = None


def _load(stem: str) -> dict[str, Any] | None:
    path = osp.join(RESULTS, stem)
    if not osp.isfile(path):
        return None
    with open(path) as fh:
        return cast(dict[str, Any], json.load(fh))


def _slurm_jobs() -> dict[str, list[int]]:
    """Job ids per result file, read out of the slurm logs rather than off a name.

    The trainer prints the absolute path of the result file it wrote, so the log itself says
    which job produced which cell. Deriving it from the job name instead gets it wrong: the
    wave-6 projection cells are named `027-w6-proj-tdg-nocstr-s2700` and
    `027-w6-proj-tdg-stoch-s2700` and write result stems that share no substring with either.
    """
    jobs: dict[str, list[int]] = {}
    if not osp.isdir(SLURM_OUT):
        return jobs
    name_pattern = re.compile(r"^(?P<name>.+)_(?P<job>\d+)\.out$")
    result_pattern = re.compile(r"results/([A-Za-z0-9_]+)\.json")
    for entry in sorted(os.listdir(SLURM_OUT)):
        match = name_pattern.match(entry)
        if not match:
            continue
        with open(osp.join(SLURM_OUT, entry), errors="replace") as fh:
            written = result_pattern.findall(fh.read())
        for stem in set(written):
            jobs.setdefault(f"{stem}.json", []).append(int(match["job"]))
    return {stem: sorted(ids) for stem, ids in jobs.items()}


def _record(
    stem: str, wave: str, result: dict[str, Any], jobs: dict[str, list[int]]
) -> CellRecord:
    cell = result["cell"]
    pinned = result["test"]["pinned"]
    counts = (result.get("flux_meta") or {}).get("kcat_tier_counts")
    return CellRecord(
        cell=stem,
        wave=wave,
        arm=cell["arm"],
        seed=int(cell["seed"]),
        select_epoch=int(result["select_epoch"]),
        select_rule=str(result["select_rule"]),
        pinned_test_spearman=float(pinned["spearman"]),
        pinned_test_pearson=float(pinned["pearson"]),
        n_pinned=int(pinned["n"]),
        final_val_pred_sd_ratio=float(result["history"][-1]["val_pred_sd_ratio"]),
        epochs_run=len(result["history"]),
        kcat_tier_counts=counts,
        on_shipped_table=None if counts is None else counts == SHIPPED_TIER_COUNTS,
        physics_key=physics_key(result),
        code_provenance=result.get("code_provenance"),
        checkpoint_path=result.get("checkpoint_path"),
        wandb_url=result.get("wandb_url"),
        slurm_job_ids=jobs.get(stem, []),
    )


def _contrast(
    name: str, pairs: list[tuple[int, float, float]], note: str = ""
) -> PairedContrast:
    seeds = [s for s, _, _ in pairs]
    flux = [f for _, f, _ in pairs]
    pooled = [p for _, _, p in pairs]
    diffs = [f - p for f, p in zip(flux, pooled, strict=True)]
    n = len(diffs)
    sd = float(np.std(diffs, ddof=1)) if n > 1 else float("nan")
    mean = float(np.mean(diffs)) if n else float("nan")
    return PairedContrast(
        name=name,
        seeds=seeds,
        flux_values=flux,
        pooled_values=pooled,
        differences=diffs,
        n_pairs=n,
        mean_difference=mean,
        sd_difference=sd,
        t_statistic=(float(mean / (sd / np.sqrt(n))) if n > 1 and sd > 0 else None),
        flux_mean=float(np.mean(flux)) if flux else float("nan"),
        flux_sd=float(np.std(flux, ddof=1)) if len(flux) > 1 else float("nan"),
        pooled_mean=float(np.mean(pooled)) if pooled else float("nan"),
        pooled_sd=float(np.std(pooled, ddof=1)) if len(pooled) > 1 else float("nan"),
        note=note,
    )


def _stat(stats: dict[str, Any], key: str) -> float | None:
    """One group median out of an audit's stats block, or None when the key is absent."""
    payload = stats.get(key)
    return None if payload is None else float(payload["median"])


def _feasibility_rows(
    group: str, jobs: dict[str, list[int]]
) -> tuple[list[FeasibilityRuleRow], list[str]]:
    rows: list[FeasibilityRuleRow] = []
    missing: list[str] = []
    for stem in AUDIT_CELLS:
        audit = _load(f"audit_{stem}.json")
        if audit is None:
            missing.append(f"audit_{stem}.json")
            continue
        strict = audit["strict_feasibility"]
        if group not in strict["groups"]:
            missing.append(f"audit_{stem}.json[{group}]")
            continue
        payload = strict["groups"][group]
        fractions = payload["fractions"]
        stats = payload["stats"]
        rows.append(
            FeasibilityRuleRow(
                cell=stem,
                label=AUDIT_LABELS.get(stem, stem),
                slurm_job_ids=jobs.get(f"{stem}.json", []),
                group=group,
                n=int(payload["n"]),
                balance_median_ratio_median=float(
                    stats["balance_median_ratio"]["median"]
                ),
                balance_max_ratio_median=float(stats["balance_max_ratio"]["median"]),
                flux_active_frac_median=float(stats["flux_active_frac"]["median"]),
                flux_active_frac_p90=float(stats["flux_active_frac"]["p90"]),
                balance_median_le_1e_2=float(
                    fractions["balance_median_ratio_le_1e-02"]
                ),
                balance_max_le_1e_2=float(fractions["balance_max_ratio_le_1e-02"]),
                thermo_standard_state_no_uphill=float(
                    fractions["thermo_standard_state_violation_frac_eq_0"]
                ),
                genesis_median_rule=float(
                    fractions["genesis_median_rule_thermo_standard_state"]
                ),
                genesis_max_rule=float(
                    fractions["genesis_max_rule_thermo_standard_state"]
                ),
                genesis_median_rule_model_thermo=fractions.get(
                    "genesis_median_rule_thermo_model"
                ),
                genesis_max_rule_model_thermo=fractions.get(
                    "genesis_max_rule_thermo_model"
                ),
                balance_active_max_ratio_median=_stat(
                    stats, f"balance_active_max_ratio_{PRIMARY_TURNOVER_TAG}"
                ),
                balance_active_p95_ratio_median=_stat(
                    stats, f"balance_active_p95_ratio_{PRIMARY_TURNOVER_TAG}"
                ),
                balance_active_rows_median=_stat(
                    stats, f"balance_active_rows_{PRIMARY_TURNOVER_TAG}"
                ),
                balance_active_max_le_1e_2=fractions.get(
                    f"balance_active_max_ratio_{PRIMARY_TURNOVER_TAG}_le_1e-02"
                ),
                genesis_active_rule=fractions.get(
                    f"genesis_active_max_{PRIMARY_TURNOVER_TAG}_rule_"
                    "thermo_standard_state"
                ),
                genesis_active_rule_model_thermo=fractions.get(
                    f"genesis_active_max_{PRIMARY_TURNOVER_TAG}_rule_thermo_model"
                ),
                balance_active_max_ratio_by_threshold={
                    tag: value
                    for tag in TURNOVER_TAGS
                    if (value := _stat(stats, f"balance_active_max_ratio_{tag}"))
                    is not None
                },
                balance_active_p95_ratio_by_threshold={
                    tag: value
                    for tag in TURNOVER_TAGS
                    if (value := _stat(stats, f"balance_active_p95_ratio_{tag}"))
                    is not None
                },
                balance_active_rows_by_threshold={
                    tag: value
                    for tag in TURNOVER_TAGS
                    if (value := _stat(stats, f"balance_active_rows_{tag}")) is not None
                },
                genesis_active_by_threshold={
                    tag: fractions[key]
                    for tag in TURNOVER_TAGS
                    for key in [
                        f"genesis_active_max_{tag}_rule_thermo_model",
                        f"genesis_active_max_{tag}_rule_thermo_standard_state",
                    ]
                    if key in fractions
                },
                agreement_max_abs_delta=(
                    audit["agreement_with_trainer"].get("max_abs_delta")
                    if audit["agreement_with_trainer"].get("checked")
                    else None
                ),
            )
        )
    return rows, missing


def _panel_paired(ax: Any, contrast: PairedContrast) -> None:
    """One line per seed across the two arms, so the pairing is read vertically."""
    xs = [0, 1]
    for i, seed in enumerate(contrast.seeds):
        ax.plot(
            xs,
            [contrast.flux_values[i], contrast.pooled_values[i]],
            marker="o",
            markersize=2.0,
            linewidth=0.6,
            color=PLOT_PALETTE[i % len(PLOT_PALETTE)],
            label=f"seed {seed}",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(["flux, box", "pooled"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("pinned test Spearman")
    if contrast.seeds:
        lo, hi = ax.get_ylim()
        ax.yaxis.set_minor_locator(MultipleLocator(max((hi - lo) / 10.0, 1e-6)))
        ax.tick_params(which="minor", length=0)
        ax.grid(True, which="both", axis="y", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)


def _panel_differences(ax: Any, wave7: PairedContrast, wave6: PairedContrast) -> None:
    """The paired difference per seed, wave 6's mixed tables against wave 7's one table."""
    seeds = sorted(set(wave7.seeds) | set(wave6.seeds))
    if not seeds:
        ax.set_xticks([])
        ax.set_ylabel("flux minus pooled")
        return
    idx = np.arange(len(seeds), dtype=float)
    width = 0.38
    for offset, contrast, color, label in (
        (-width / 2, wave6, PLOT_PALETTE[0], "wave 6, three tables"),
        (width / 2, wave7, PLOT_PALETTE[1], "wave 7, shipped table"),
    ):
        values = [
            contrast.differences[contrast.seeds.index(s)]
            if s in contrast.seeds
            else np.nan
            for s in seeds
        ]
        ax.bar(
            idx + offset,
            values,
            width,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            label=label,
        )
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"s{s}" for s in seeds])
    ax.set_ylabel("flux minus pooled")
    ax.grid(True, which="both", axis="y", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)


def _panel_rules(ax: Any, rows: list[FeasibilityRuleRow]) -> None:
    """Mass balance within 1 percent under each of the three balance rules.

    The BALANCE half of the Genesis criterion rather than the criterion itself, because the
    criterion is 0.00 in every audited cell under the standard-state second law and a panel
    of empty bars carries nothing. What the rules change is exactly this half, and the
    Genesis fractions under each rule are in the note's table.
    """
    if not rows:
        ax.set_xticks([])
        ax.set_ylabel("fraction of the top 100 balanced")
        return
    idx = np.arange(len(rows), dtype=float)
    width = 0.27
    for offset, values, color, label in (
        (
            -width,
            [r.balance_median_le_1e_2 for r in rows],
            PLOT_PALETTE[0],
            "median rule",
        ),
        (0.0, [r.balance_max_le_1e_2 for r in rows], PLOT_PALETTE[1], "max rule"),
        (
            width,
            [
                np.nan
                if r.balance_active_max_le_1e_2 is None
                else r.balance_active_max_le_1e_2
                for r in rows
            ],
            PLOT_PALETTE[2],
            "max over rows that turn over",
        ),
    ):
        ax.bar(
            idx + offset,
            values,
            width,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            label=label,
        )
    ax.set_xticks(idx)
    ax.set_xticklabels([r.label for r in rows], rotation=90)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.set_ylabel("fraction of the top 100 balanced")
    ax.grid(True, which="both", axis="y", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)


def _panel_active(ax: Any, rows: list[FeasibilityRuleRow]) -> None:
    """The fraction of reactions carrying flux, which is what makes a median pass cheap."""
    if not rows:
        ax.set_xticks([])
        ax.set_ylabel("fraction of reactions carrying flux")
        return
    idx = np.arange(len(rows), dtype=float)
    ax.bar(
        idx,
        [r.flux_active_frac_median for r in rows],
        0.6,
        color=PLOT_PALETTE[2],
        edgecolor="black",
        linewidth=0.4,
        label="median genotype",
    )
    ax.set_xticks(idx)
    ax.set_xticklabels([r.label for r in rows], rotation=90)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.set_ylabel("fraction of reactions carrying flux")
    ax.grid(True, which="both", axis="y", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)


def make_figure(
    wave7: PairedContrast,
    wave6: PairedContrast,
    rows: list[FeasibilityRuleRow],
    stamp: str,
) -> str:
    apply_paper_style()
    fig, grid = plt.subplots(
        2,
        2,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(135.0)),
        constrained_layout=True,
    )
    axes = grid.ravel()
    _panel_paired(axes[0], wave7)
    _panel_differences(axes[1], wave7, wave6)
    _panel_rules(axes[2], rows)
    _panel_active(axes[3], rows)
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
    for ax, ncol in ((axes[0], 3), (axes[1], 2), (axes[2], 3), (axes[3], 1)):
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
    for ax, letter in zip(axes, "abcd", strict=True):
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
    base = osp.join(IMAGE_DIR, f"wave7_parity_one_table_{stamp}")
    fig.savefig(f"{base}.png", dpi=300)
    savefig_true_size_svg(fig, f"{base}.svg")
    plt.close(fig)
    return f"{base}.svg"


def main() -> None:
    stamp = timestamp()
    jobs = _slurm_jobs()
    cells: list[CellRecord] = []
    missing: list[str] = []

    loaded: dict[str, dict[str, dict[str, Any]]] = {"wave7": {}, "wave6": {}}
    for wave, arms in (("wave7", WAVE7_ARMS), ("wave6", WAVE6_ARMS)):
        for seed in SEEDS:
            for arm, template in arms.items():
                stem = template.format(seed=seed)
                result = _load(stem)
                if result is None:
                    missing.append(stem)
                    continue
                loaded[wave][f"{arm}_{seed}"] = result
                cells.append(_record(stem, wave, result, jobs))

    # WAVE 7, and the table guard. A flux cell that did not resolve the shipped table is
    # not a replicate of the others and is reported by name rather than averaged in.
    off_table: dict[int, str] = {}
    wave7_pairs: list[tuple[int, float, float]] = []
    for seed in SEEDS:
        flux = loaded["wave7"].get(f"nocstr_{seed}")
        pooled = loaded["wave7"].get(f"pooled_{seed}")
        if flux is None or pooled is None:
            continue
        counts = (flux.get("flux_meta") or {}).get("kcat_tier_counts")
        if counts != SHIPPED_TIER_COUNTS:
            off_table[seed] = (
                f"resolved {counts}, not the shipped {SHIPPED_TIER_COUNTS}"
            )
            continue
        wave7_pairs.append(
            (
                seed,
                float(flux["test"]["pinned"]["spearman"]),
                float(pooled["test"]["pinned"]["spearman"]),
            )
        )
    wave7 = _contrast(
        "betaxanthin parity, wave 7, every flux cell on the shipped verified kcat table",
        wave7_pairs,
    )

    # WAVE 6, for reference and per table.
    wave6_pairs: list[tuple[int, float, float]] = []
    by_table: dict[str, list[tuple[int, float, float]]] = {}
    shipped_wave6: list[int] = []
    for seed in SEEDS:
        flux = loaded["wave6"].get(f"nocstr_{seed}")
        pooled = loaded["wave6"].get(f"pooled_{seed}")
        if flux is None or pooled is None:
            continue
        entry = (
            seed,
            float(flux["test"]["pinned"]["spearman"]),
            float(pooled["test"]["pinned"]["spearman"]),
        )
        wave6_pairs.append(entry)
        counts = (flux.get("flux_meta") or {}).get("kcat_tier_counts") or {}
        key = "/".join(f"{k}={counts[k]}" for k in sorted(counts))
        by_table.setdefault(key, []).append(entry)
        if counts == SHIPPED_TIER_COUNTS:
            shipped_wave6.append(seed)
    wave6 = _contrast(
        "betaxanthin parity, wave 6, three kcat tables pooled (NOT the number to quote)",
        wave6_pairs,
        note=(
            "reproduced here from the wave-6 result files so the two waves are computed by "
            "one script under one scoring rule."
        ),
    )

    combined = _contrast(
        "betaxanthin parity, every pair on the shipped verified kcat table",
        wave7_pairs + [p for p in wave6_pairs if p[0] in shipped_wave6],
        note=(
            "wave 7 plus every wave-6 pair whose flux cell also carries the shipped table. "
            f"Wave-6 pairs on the shipped table: {shipped_wave6 or 'none'}. The wave-6 "
            "'same table' subset (seeds 2700, 2701, 2703) is the WAVE-5 table "
            "148/0/3410/170, not the shipped one, so it cannot be pooled with wave 7."
        ),
    )

    shift: dict[int, dict[str, float]] = {}
    for seed in SEEDS:
        w7f = loaded["wave7"].get(f"nocstr_{seed}")
        w6f = loaded["wave6"].get(f"nocstr_{seed}")
        w7p = loaded["wave7"].get(f"pooled_{seed}")
        w6p = loaded["wave6"].get(f"pooled_{seed}")
        if w7f is None or w6f is None or w7p is None or w6p is None:
            continue
        shift[seed] = {
            "flux_wave6": float(w6f["test"]["pinned"]["spearman"]),
            "flux_wave7": float(w7f["test"]["pinned"]["spearman"]),
            "pooled_wave6": float(w6p["test"]["pinned"]["spearman"]),
            "pooled_wave7": float(w7p["test"]["pinned"]["spearman"]),
        }

    # The wave-6 seeds that share one table WITH EACH OTHER, which is the largest such
    # group and the one the wave-6 note quotes. It is the wave-5 table, so pooling it with
    # wave 7 crosses two tables; the contrast is built because it was asked for and is
    # labeled with what it spans.
    largest_wave6_group = max(
        by_table.items(), key=lambda item: len(item[1]), default=("", [])
    )
    combined_two_tables = _contrast(
        "betaxanthin parity, wave 7 pooled with the largest wave-6 same-table group "
        "(TWO tables, not one)",
        wave7_pairs + list(largest_wave6_group[1]),
        note=(
            "wave 7 seeds are on the shipped verified table "
            "(97/782/2831/18); the wave-6 seeds pooled in here are on "
            f"{largest_wave6_group[0] or 'no table'}. Reported because the six-seed "
            "combination was asked for, and labeled because the two halves are not "
            "replicates of one another."
        ),
    )

    rows, missing_audits = _feasibility_rows("all_top100", jobs)
    verification = _load("wave7_kcat_table_verification.json")
    figure = make_figure(wave7, wave6, rows, stamp)
    summary = Wave7Summary(
        generated_at=stamp,
        kcat_table_verification=verification,
        expected_tier_counts=SHIPPED_TIER_COUNTS,
        wave7=wave7,
        wave7_off_table_seeds=off_table,
        wave6_shipped_table_seeds=shipped_wave6,
        combined=combined,
        combined_with_wave6_same_table_group=combined_two_tables,
        wave6_reference=wave6,
        wave6_by_table={
            key: _contrast(f"wave 6, kcat {key}", pairs)
            for key, pairs in by_table.items()
        },
        same_seed_table_shift=shift,
        cells=cells,
        feasibility=rows,
        missing_files=sorted(set(missing)),
        missing_audits=missing_audits,
        figure_path=figure,
    )
    out = osp.join(RESULTS, "wave7_parity_summary.json")
    with open(out, "w") as fh:
        fh.write(summary.model_dump_json(indent=2))
    print(f"wrote {out}")
    print(f"figure {figure}")
    for contrast in (wave7, combined, combined_two_tables, wave6):
        print(
            f"\n{contrast.name}\n"
            f"  seeds {contrast.seeds} n={contrast.n_pairs}\n"
            f"  flux   mean {contrast.flux_mean:+.4f} sd {contrast.flux_sd:.4f}\n"
            f"  pooled mean {contrast.pooled_mean:+.4f} sd {contrast.pooled_sd:.4f}\n"
            f"  paired difference {contrast.mean_difference:+.4f} "
            f"sd {contrast.sd_difference:.4f} t {contrast.t_statistic}"
        )
    if off_table:
        print("\nOFF-TABLE flux cells, excluded from the wave-7 contrast:")
        for seed, why in off_table.items():
            print(f"  seed {seed}: {why}")
    if rows:
        print(
            "\nfeasibility of the top 100; genesis is median / max / turnover-thresholded"
        )
        for row in rows:
            active = row.balance_active_max_ratio_median
            active_rows = row.balance_active_rows_median
            genesis_active = row.genesis_active_rule_model_thermo
            if genesis_active is None:
                genesis_active = row.genesis_active_rule
            genesis_median = row.genesis_median_rule_model_thermo
            if genesis_median is None:
                genesis_median = row.genesis_median_rule
            genesis_max = row.genesis_max_rule_model_thermo
            if genesis_max is None:
                genesis_max = row.genesis_max_rule
            print(
                f"  {row.label:22s} job {str(row.slurm_job_ids):>8s} "
                f"bal med {row.balance_median_ratio_median:.2e} "
                f"max {row.balance_max_ratio_median:.2e} "
                f"active-max {'n/a' if active is None else f'{active:.2e}'} "
                f"rows {'n/a' if active_rows is None else f'{active_rows:.0f}'} "
                f"carrying {row.flux_active_frac_median:.3f} | genesis "
                f"{genesis_median:.2f} / {genesis_max:.2f} / "
                f"{'n/a' if genesis_active is None else f'{genesis_active:.2f}'}"
            )
    if summary.missing_files:
        print("\nMISSING result files:")
        for name in summary.missing_files:
            print(f"  {name}")
    if summary.missing_audits:
        print("\nMISSING audits:")
        for name in summary.missing_audits:
            print(f"  {name}")


if __name__ == "__main__":
    main()
