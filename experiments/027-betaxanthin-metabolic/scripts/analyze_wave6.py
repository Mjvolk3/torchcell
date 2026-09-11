# experiments/027-betaxanthin-metabolic/scripts/analyze_wave6.py
# [[experiments.027-betaxanthin-metabolic.wave6]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/analyze_wave6.py

r"""Aggregate every wave-5 and wave-6 cell into one table, one figure and one JSON.

WHAT IT READS, and nothing else. Only finished result files under ``results/``. A cell
whose file does not exist is reported as MISSING by name rather than dropped, and a cell
that is still running has no result file, so it can never contribute a number here. The
script is therefore safe to re-run while the queue drains: it summarizes what exists.

THREE TABLES
------------
1. **The betaxanthin parity contrast**, paired by seed. Each seed contributes one
   difference, ``flux_capacity nocstr headfix`` minus ``pooled``, of the pinned-test
   Spearman at the validation-selected epoch. Reported as the mean difference, its sample
   SD, the number of seeds, and a paired t statistic. The t statistic is a summary of the
   spread, not a significance claim: at five or six seeds it is read for its sign and
   magnitude and the SD is what carries the uncertainty.
2. **The amino-acid cells**, the same way, on the MEAN per-feature Spearman over the 19
   Mulleder columns.
3. **The feasibility table**: for every cell that carries a flux layer, the fraction of the
   top 100 predicted producers meeting each tolerance, beside the same fraction over the
   whole population, so a top-100 number can be read against its base rate.

Run from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/analyze_wave6.py

Writes ``results/wave6_summary.json`` and one timestamped figure into
``$ASSET_IMAGES_DIR/027-betaxanthin-metabolic/``.
"""

import json
import os
import os.path as osp
from collections import Counter
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

EXPERIMENT_ROOT = cast(str, os.environ["EXPERIMENT_ROOT"])
ASSET_IMAGES_DIR = cast(str, os.environ["ASSET_IMAGES_DIR"])
EXP_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic")
RESULTS = osp.join(EXP_DIR, "results")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "027-betaxanthin-metabolic")

#: The betaxanthin parity pair, by result-file stem. The wave-5 stems are kept for the
#: wave-6 seeds so the six seeds of one arm are one series of files.
PARITY_SEEDS = (2700, 2701, 2702, 2703, 2704, 2705)
PARITY_ARMS: dict[str, str] = {
    "nocstr": "wave5_g7_flux_capacity_nocstr_s{seed}.json",
    "pooled": "wave5_g8_pooled_s{seed}.json",
}

#: The amino-acid cells. `allon` is a single constrained cell and has no pair, so it is
#: summarized per arm and never enters a paired contrast.
AA_SEEDS = (2700, 2701, 2702)
AA_ARMS: dict[str, str] = {
    "nocstr": "wave6_aa_nocstr_s{seed}.json",
    "pooled": "wave6_aa_pooled_s{seed}.json",
}
AA_EXTRA: dict[str, str] = {"allon_s2700": "wave6_aa_allon_s2700.json"}

#: Which feasibility group the headline number comes from, and the criteria to tabulate.
TOP_GROUP = "all_top100"
POP_GROUP = "all"


def physics_key(result: dict[str, Any]) -> str | None:
    """What the flux layer was actually built from, as one comparable string.

    NOT a code hash. `train_flux.py`'s ``kcat_homolog_tier`` was switched ON by default on
    2026-09-11, which changes the resolved kcat table and therefore the enzyme-capacity
    bound, i.e. the flux box itself. Two cells built that way and the old way are not
    replicates of one another, and pooling them into one paired contrast would average two
    different models. `flux_layer.py` is byte-identical across the change, so a hash of the
    layer would NOT catch it; what catches it is the tier counts the run recorded for
    itself.

    Returns ``None`` for an arm with no flux layer, which has no physics to match.
    """
    meta = result.get("flux_meta") or {}
    if not meta:
        return None
    counts = meta.get("kcat_tier_counts") or {}
    return "|".join(
        [
            str(meta.get("kcat_tier")),
            ",".join(f"{k}={counts[k]}" for k in sorted(counts)),
            str(meta.get("thermo_source")),
            str(meta.get("thermo_sha256")),
            ",".join(
                f"{k}={v}" for k, v in sorted((meta.get("km_tier_counts") or {}).items())
            ),
        ]
    )


class PairedContrast(BaseModel):
    """One arm-vs-arm contrast, paired by seed."""

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
    #: Seeds dropped from the contrast because the flux arm's layer was built from a
    #: different parameter table than the majority of the cells, keyed by seed.
    excluded_physics_mismatch: dict[int, str] = Field(default_factory=dict)
    physics_key_used: str | None = None


class FeasibilityRow(BaseModel):
    """The feasibility of one cell, top 100 against the whole population."""

    model_config = ConfigDict(extra="forbid")

    cell: str
    arm: str
    seed: int
    head: str
    select_epoch: int
    available: bool
    reason: str | None = None
    n_top: int | None = None
    n_population: int | None = None
    top_fractions: dict[str, float] = Field(default_factory=dict)
    population_fractions: dict[str, float] = Field(default_factory=dict)
    #: The MEDIAN over the group of each per-genotype feasibility scalar.
    top_medians: dict[str, float] = Field(default_factory=dict)
    population_medians: dict[str, float] = Field(default_factory=dict)

    def stats_available(self, stat: str) -> bool:
        """Whether this cell reported the named per-genotype scalar at all."""
        return stat in self.population_medians

    def stat(self, stat: str, group: str) -> float:
        """The group median of one per-genotype scalar; NaN when the cell lacks it."""
        source = self.top_medians if group == "top" else self.population_medians
        return source.get(stat, float("nan"))

    def short_label(self) -> str:
        """Arm, phenotype and seed in the few characters an axis tick can hold."""
        arm = "flux" if self.arm != "pooled" else "pool"
        phen = "bx" if self.head == "betaxanthin" else "aa"
        return f"{arm} {phen} s{self.seed}"


class Wave6Summary(BaseModel):
    """Everything the wave-6 note reports."""

    model_config = ConfigDict(extra="forbid")

    generated_at: str
    betaxanthin_parity: PairedContrast
    amino_acid_parity: PairedContrast
    amino_acid_extra: dict[str, dict[str, Any]] = Field(default_factory=dict)
    feasibility: list[FeasibilityRow] = Field(default_factory=list)
    per_cell: dict[str, dict[str, Any]] = Field(default_factory=dict)
    missing_files: list[str] = Field(default_factory=list)
    figure_path: str | None = None


def _load(stem: str) -> dict[str, Any] | None:
    path = osp.join(RESULTS, stem)
    if not osp.isfile(path):
        return None
    with open(path) as fh:
        return cast(dict[str, Any], json.load(fh))


def _score(result: dict[str, Any]) -> float:
    """The cell's headline held-out number, whichever head it trained."""
    pinned = result["test"]["pinned"]
    return float(
        pinned["spearman"] if result["cell"].get("head", "betaxanthin") == "betaxanthin"
        else pinned["mean_spearman"]
    )


def _paired(
    metric: str, arms: dict[str, str], seeds: tuple[int, ...]
) -> tuple[PairedContrast, dict[str, dict[str, Any]]]:
    a_name, b_name = "nocstr", "pooled"
    missing: list[str] = []
    loaded: dict[str, dict[str, Any]] = {}
    pairs: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for seed in seeds:
        stems = {arm: arms[arm].format(seed=seed) for arm in (a_name, b_name)}
        results = {arm: _load(stem) for arm, stem in stems.items()}
        for arm, res in results.items():
            if res is None:
                missing.append(stems[arm])
            else:
                loaded[stems[arm]] = res
        if results[a_name] is None or results[b_name] is None:
            continue
        pairs.append(
            (
                seed,
                cast(dict[str, Any], results[a_name]),
                cast(dict[str, Any], results[b_name]),
            )
        )

    # THE PHYSICS GUARD. Only cells whose flux layer was built from the same parameter
    # tables are replicates of one another; the majority key is what the contrast is
    # computed on, and every seed outside it is dropped BY NAME rather than averaged in.
    keys = [physics_key(a) for _, a, _ in pairs]
    present = [k for k in keys if k is not None]
    modal = Counter(present).most_common(1)[0][0] if present else None
    kept_seeds: list[int] = []
    a_vals: list[float] = []
    b_vals: list[float] = []
    excluded: dict[int, str] = {}
    for (seed, a_res, b_res), key in zip(pairs, keys, strict=True):
        if modal is not None and key is not None and key != modal:
            excluded[seed] = (
                f"the flux arm's layer was built from a different parameter table "
                f"({key}) than the other cells ({modal}), so it is not a replicate of them."
            )
            continue
        kept_seeds.append(seed)
        a_vals.append(_score(a_res))
        b_vals.append(_score(b_res))
    diffs = [a - b for a, b in zip(a_vals, b_vals, strict=True)]
    n = len(diffs)
    # ddof=1: the sample SD of the differences, which is what the paired t divides by.
    sd = float(np.std(diffs, ddof=1)) if n > 1 else float("nan")
    mean = float(np.mean(diffs)) if n else float("nan")
    t = float(mean / (sd / np.sqrt(n))) if n > 1 and sd > 0 else None
    return (
        PairedContrast(
            metric=metric,
            arm_a=a_name,
            arm_b=b_name,
            seeds=kept_seeds,
            a_values=a_vals,
            b_values=b_vals,
            differences=diffs,
            n_pairs=n,
            mean_difference=mean,
            sd_difference=sd,
            t_statistic=t,
            t_statistic_note=(
                "paired t = mean(difference) / (sd(difference) / sqrt(n)) on "
                f"n={n} seeds, df={max(n - 1, 0)}. Reported for its sign and magnitude; "
                "at this n it is a summary of the spread, not a significance claim."
            ),
            a_mean=float(np.mean(a_vals)) if a_vals else float("nan"),
            a_sd=float(np.std(a_vals, ddof=1)) if len(a_vals) > 1 else float("nan"),
            b_mean=float(np.mean(b_vals)) if b_vals else float("nan"),
            b_sd=float(np.std(b_vals, ddof=1)) if len(b_vals) > 1 else float("nan"),
            missing=missing,
            excluded_physics_mismatch=excluded,
            physics_key_used=modal,
        ),
        loaded,
    )


def _feasibility_rows(loaded: dict[str, dict[str, Any]]) -> list[FeasibilityRow]:
    rows: list[FeasibilityRow] = []
    for stem, result in sorted(loaded.items()):
        block = result.get("feasibility")
        cellcfg = result["cell"]
        if block is None:
            rows.append(
                FeasibilityRow(
                    cell=stem,
                    arm=cellcfg["arm"],
                    seed=int(cellcfg["seed"]),
                    head=cellcfg.get("head", "betaxanthin"),
                    select_epoch=int(result["select_epoch"]),
                    available=False,
                    reason=(
                        "the cell predates the wave-6 feasibility block; its result file "
                        "carries no 'feasibility' key."
                    ),
                )
            )
            continue
        if not block["available"]:
            rows.append(
                FeasibilityRow(
                    cell=stem,
                    arm=cellcfg["arm"],
                    seed=int(cellcfg["seed"]),
                    head=cellcfg.get("head", "betaxanthin"),
                    select_epoch=int(result["select_epoch"]),
                    available=False,
                    reason=block["reason"],
                )
            )
            continue
        groups = block["groups"]
        top = groups.get(TOP_GROUP)
        pop = groups[POP_GROUP]
        rows.append(
            FeasibilityRow(
                cell=stem,
                arm=cellcfg["arm"],
                seed=int(cellcfg["seed"]),
                head=cellcfg.get("head", "betaxanthin"),
                select_epoch=int(result["select_epoch"]),
                available=True,
                n_top=None if top is None else int(top["n"]),
                n_population=int(pop["n"]),
                top_fractions={} if top is None else dict(top["fractions"]),
                population_fractions=dict(pop["fractions"]),
                top_medians=(
                    {}
                    if top is None
                    else {k: float(v["median"]) for k, v in top["stats"].items()}
                ),
                population_medians={
                    k: float(v["median"]) for k, v in pop["stats"].items()
                },
            )
        )
    return rows


def _panel_paired(ax: Any, contrast: PairedContrast, ylabel: str) -> None:
    """Per-seed paired points, one line per seed, arm on the x axis."""
    xs = [0, 1]
    for i, seed in enumerate(contrast.seeds):
        color = PLOT_PALETTE[i % len(PLOT_PALETTE)]
        ax.plot(
            xs,
            [contrast.a_values[i], contrast.b_values[i]],
            marker="o",
            markersize=2.0,
            linewidth=0.6,
            color=color,
            label=f"seed {seed}",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(["flux, no constraint", "pooled"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel(ylabel)
    if contrast.seeds:
        # The minor step follows the DATA range. A fixed 0.01 step is right for a span of
        # 0.05 and draws a hundred gridlines on the empty 0-to-1 default, which is what the
        # panel shows before its cells have finished.
        lo, hi = ax.get_ylim()
        ax.yaxis.set_minor_locator(MultipleLocator(max((hi - lo) / 10.0, 1e-6)))
        ax.tick_params(which="minor", length=0)
        ax.grid(True, which="both", axis="y", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)


def _panel_violation(
    ax: Any, rows: list[FeasibilityRow], stat: str, ylabel: str, ymax: float
) -> None:
    """Median per-genotype violation, top 100 against the whole population, per flux cell.

    The violation itself rather than the fraction meeting a tolerance: on an arm whose
    constraint weight is zero every tolerance in the ladder is missed by every genotype, so
    a panel of fractions would be a row of empty bars, and the magnitude of the violation is
    the thing that distinguishes one cell from another.
    """
    usable = [r for r in rows if r.available and r.stats_available(stat)]
    if not usable:
        ax.set_xticks([])
        ax.set_ylabel(ylabel)
        return
    idx = np.arange(len(usable), dtype=float)
    width = 0.38
    ax.bar(
        idx - width / 2,
        [r.stat(stat, "top") for r in usable],
        width,
        color=PLOT_PALETTE[0],
        edgecolor="black",
        linewidth=0.4,
        label="top 100 predicted",
    )
    ax.bar(
        idx + width / 2,
        [r.stat(stat, "population") for r in usable],
        width,
        color=PLOT_PALETTE[1],
        edgecolor="black",
        linewidth=0.4,
        label="all genotypes",
    )
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{r.short_label()}" for r in usable], rotation=90)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(MultipleLocator(ymax / 5))
    ax.yaxis.set_minor_locator(MultipleLocator(ymax / 10))
    ax.tick_params(which="minor", length=0)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", axis="y", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)


def make_figure(
    bx: PairedContrast, aa: PairedContrast, rows: list[FeasibilityRow], stamp: str
) -> str:
    apply_paper_style()
    width_mm = PANEL_WIDTHS_MM["full"]
    fig, grid = plt.subplots(
        2, 2, figsize=(mm_to_in(width_mm), mm_to_in(130.0)), constrained_layout=True
    )
    axes = grid.ravel()
    _panel_paired(axes[0], bx, "pinned test Spearman")
    _panel_paired(axes[1], aa, "mean per-feature Spearman")
    # The theoretical ceiling of the balance ratio is 2: |sum_j S_ij v_j| can at most equal
    # sum_j |S_ij v_j|, and omega is half of that. A bar at 2 is a flux vector in which
    # nothing a metabolite produces is consumed.
    _panel_violation(
        axes[2],
        rows,
        "balance_median_ratio",
        "median balance residual ratio",
        2.0,
    )
    _panel_violation(
        axes[3],
        rows,
        "thermo_standard_state_violation_frac",
        "median uphill reaction fraction",
        1.0,
    )
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
    if bx.seeds:
        axes[0].legend(
            loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=True,
            fontsize=5.0, handlelength=1.2, borderpad=0.3,
        )
    for ax in (axes[2], axes[3]):
        if ax.get_legend_handles_labels()[0]:
            ax.legend(
                loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=True,
                fontsize=5.0, handlelength=1.2, borderpad=0.3,
            )
    for ax, letter in zip(axes, "abcd", strict=True):
        ax.text(
            -0.16, 1.12, letter, transform=ax.transAxes,
            fontsize=8.0, fontweight="bold", va="bottom", ha="left",
        )
    os.makedirs(IMAGE_DIR, exist_ok=True)
    base = osp.join(IMAGE_DIR, f"wave6_parity_and_feasibility_{stamp}")
    fig.savefig(f"{base}.png", dpi=300)
    savefig_true_size_svg(fig, f"{base}.svg")
    plt.close(fig)
    return f"{base}.svg"


def main() -> None:
    stamp = timestamp()
    bx, loaded_bx = _paired("pinned test Spearman (betaxanthin)", PARITY_ARMS, PARITY_SEEDS)
    aa, loaded_aa = _paired(
        "mean per-feature test Spearman (mulleder19)", AA_ARMS, AA_SEEDS
    )
    loaded: dict[str, dict[str, Any]] = {**loaded_bx, **loaded_aa}
    extra: dict[str, dict[str, Any]] = {}
    for name, stem in AA_EXTRA.items():
        result = _load(stem)
        if result is None:
            aa.missing.append(stem)
            continue
        loaded[stem] = result
        extra[name] = {
            "cell": stem,
            "select_epoch": result["select_epoch"],
            "mean_spearman_pinned": result["test"]["pinned"]["mean_spearman"],
            "mean_pearson_pinned": result["test"]["pinned"]["mean_pearson"],
            "mean_spearman_all_test": result["test"]["all"]["mean_spearman"],
        }

    rows = _feasibility_rows(loaded)
    per_cell = {
        stem: {
            "arm": r["cell"]["arm"],
            "seed": r["cell"]["seed"],
            "head": r["cell"].get("head", "betaxanthin"),
            "select_epoch": r["select_epoch"],
            "select_rule": r["select_rule"],
            "select_val_pearson": r["select_val_pearson"],
            "score": _score(r),
            "n_pinned": (
                r["test"]["pinned"].get("n")
                or r["test"]["pinned"].get("n_measurements")
            ),
            "epochs_run": len(r["history"]),
            "epochs_configured": r["cell"]["epochs"],
            "final_val_pred_sd_ratio": r["history"][-1]["val_pred_sd_ratio"],
            "wandb_url": r.get("wandb_url"),
            "code_provenance": r.get("code_provenance"),
            "checkpoint_path": r.get("checkpoint_path"),
            "physics_key": physics_key(r),
        }
        for stem, r in sorted(loaded.items())
    }

    figure = make_figure(bx, aa, rows, stamp)
    summary = Wave6Summary(
        generated_at=stamp,
        betaxanthin_parity=bx,
        amino_acid_parity=aa,
        amino_acid_extra=extra,
        feasibility=rows,
        per_cell=per_cell,
        missing_files=sorted(set(bx.missing) | set(aa.missing)),
        figure_path=figure,
    )
    out = osp.join(RESULTS, "wave6_summary.json")
    with open(out, "w") as fh:
        fh.write(summary.model_dump_json(indent=2))

    print(f"wrote {out}")
    print(f"figure {figure}")
    print(
        f"\nbetaxanthin parity, n={bx.n_pairs} seeds {bx.seeds}\n"
        f"  nocstr mean {bx.a_mean:+.4f} sd {bx.a_sd:.4f}\n"
        f"  pooled mean {bx.b_mean:+.4f} sd {bx.b_sd:.4f}\n"
        f"  paired difference {bx.mean_difference:+.4f} sd {bx.sd_difference:.4f} "
        f"t {bx.t_statistic}"
    )
    print(
        f"\namino-acid parity, n={aa.n_pairs} seeds {aa.seeds}\n"
        f"  nocstr mean {aa.a_mean:+.4f} sd {aa.a_sd:.4f}\n"
        f"  pooled mean {aa.b_mean:+.4f} sd {aa.b_sd:.4f}\n"
        f"  paired difference {aa.mean_difference:+.4f} sd {aa.sd_difference:.4f} "
        f"t {aa.t_statistic}"
    )
    print("\nfeasibility (top 100 predicted / all genotypes)")
    for row in rows:
        if not row.available:
            print(f"  {row.cell}: null -- {row.reason}")
            continue
        for key in sorted(row.population_fractions):
            top = row.top_fractions.get(key)
            top_txt = "n/a" if top is None else f"{top:.3f}"
            print(f"  {row.cell} {key}: top {top_txt} / all "
                  f"{row.population_fractions[key]:.3f}")
    for label, contrast in (("betaxanthin", bx), ("amino acid", aa)):
        if contrast.excluded_physics_mismatch:
            print(f"\nEXCLUDED from the {label} contrast, physics mismatch:")
            for seed, why in contrast.excluded_physics_mismatch.items():
                print(f"  seed {seed}: {why}")
    if summary.missing_files:
        print("\nMISSING (not yet finished or never launched):")
        for name in summary.missing_files:
            print(f"  {name}")


if __name__ == "__main__":
    main()
