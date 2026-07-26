# experiments/019-echo-crispr-array/scripts/ladder_feasibility.py
# [[experiments.019-echo-crispr-array.scripts.ladder_feasibility]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/ladder_feasibility
"""Is "stepping up the fitness ladder" reachable with a deletion panel?

The next round pairs the 12 assayed single knockouts with ~14 double knockouts and asks
whether stacking deletions walks fitness UPWARD. That is only worth doing if upward
headroom exists, so this script measures the headroom from published data:

  A. Our 12-strain panel -- run-3 measurements and the published Costanzo SMF, each
     tested one-sided against the on-plate wild type (fitness = 1).
  B. Genome-wide single deletions (Costanzo 2016): what fraction of the ~7,700 deletion
     strains sit significantly ABOVE wild type, and where the ceiling is.
  C. Double deletions (Costanzo 2016 subset): fraction beating wild type, and fraction
     beating BOTH of their own singles -- the actual "rung" event.
  D. The Kuzmin 2018 ladder (singles/doubles/triples from ONE screen at 26 C): does the
     upper tail move as deletions are stacked?

Multiplicity is controlled with Benjamini-Hochberg FDR at 0.05; every test is one-sided
because only improvement counts.

CAVEAT recorded in the output table: the double/triple uncertainties are `sample_sd`
over 4 colonies within one screen, so they omit the plate/screen term and are
OPTIMISTIC. That biases the exceedance counts UP, which makes the "no headroom"
conclusion conservative.
"""

import json
import os
import os.path as osp
import pickle
from typing import Any

import lmdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator
from scipy import stats

from torchcell.data.experiment_dataset import resolve_interned
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]

EXP_DIR = osp.join(EXPERIMENT_ROOT, "019-echo-crispr-array")
RESULTS = osp.join(EXP_DIR, "results")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "019-echo-crispr-array")

DELETION_TYPES = {"sga_kanmx_deletion", "sga_natmx_deletion"}
FDR = 0.05
N_DOUBLES_NEXT_ROUND = 14


# --------------------------------------------------------------------------- io
def _load_interned(base: str) -> dict[str, Any]:
    """Load the sibling `interned` env that record dicts reference by `$ref`."""
    d = osp.join(base, "processed/interned")
    out: dict[str, Any] = {}
    if osp.isdir(d):
        env = lmdb.open(d, readonly=True, lock=False, readahead=False)
        with env.begin() as txn:
            for k, v in txn.cursor():
                out[k.decode()] = pickle.loads(v)
        env.close()
    return out


def read_fitness_lmdb(name: str) -> pd.DataFrame:
    """Stream a fitness LMDB into a flat frame of genes + fitness + SE."""
    base = osp.join(DATA_ROOT, "data/torchcell", name)
    interned = _load_interned(base)
    env = lmdb.open(
        osp.join(base, "processed/lmdb"), readonly=True, lock=False, readahead=False
    )
    rows: list[dict[str, Any]] = []
    with env.begin() as txn:
        for _, v in txn.cursor():
            e = resolve_interned(pickle.loads(v), interned)["experiment"]
            ph = e["phenotype"]
            row: dict[str, Any] = {
                "temperature": e["environment"]["temperature"]["value"],
                "fitness": ph["fitness"],
                "fitness_se": ph["fitness_se"],
                "n_samples": ph["n_samples"],
            }
            for i, p in enumerate(e["genotype"]["perturbations"]):
                row[f"gene{i}"] = p["systematic_gene_name"]
                row[f"ptype{i}"] = p["perturbation_type"]
            rows.append(row)
    env.close()
    return pd.DataFrame(rows)


def deletions_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep records whose every perturbation is a kanMX/natMX deletion.

    Also drops the handful of records with a non-positive SE: a zero SE would send the
    one-sided z to +/-inf and force a spurious rejection.
    """
    cols = [c for c in df.columns if c.startswith("ptype")]
    mask = np.ones(len(df), dtype=bool)
    for c in cols:
        mask &= df[c].isin(DELETION_TYPES).to_numpy()
    out = df[mask]
    if out.fitness_se.notna().any():
        bad = int((out.fitness_se.notna() & (out.fitness_se <= 0)).sum())
        if bad:
            print(f"  dropped {bad} record(s) with SE <= 0")
            out = out[out.fitness_se.isna() | (out.fitness_se > 0)]
    return out


# ------------------------------------------------------------------- statistics
def bh_reject(p: np.ndarray, alpha: float = FDR) -> np.ndarray:
    """Benjamini-Hochberg step-up: boolean mask of rejected hypotheses."""
    n = len(p)
    order = np.argsort(p)
    passed = p[order] <= alpha * (np.arange(1, n + 1) / n)
    out = np.zeros(n, dtype=bool)
    if passed.any():
        out[order[: int(np.max(np.flatnonzero(passed))) + 1]] = True
    return out


def p_greater(value: np.ndarray, ref: float | np.ndarray, se: np.ndarray) -> np.ndarray:
    """One-sided p-value for H1: value > ref (normal approximation)."""
    return np.asarray(stats.norm.sf((value - np.asarray(ref)) / se))


def summarise(
    label: str,
    value: np.ndarray,
    se: np.ndarray,
    ref: float | np.ndarray,
    se_note: str,
) -> dict[str, Any]:
    """Count point-estimate and FDR-significant exceedances of `ref`."""
    p = p_greater(value, ref, se)
    rej = bh_reject(p)
    excess = value - np.asarray(ref)
    return {
        "comparison": label,
        "n": int(len(value)),
        "n_above": int((excess > 0).sum()),
        "frac_above": float((excess > 0).mean()),
        "n_sig_above": int(rej.sum()),
        "frac_sig_above": float(rej.mean()),
        "median_excess_of_sig": float(np.median(excess[rej])) if rej.any() else np.nan,
        "max_value": float(np.max(value)),
        "q99_value": float(np.quantile(value, 0.99)),
        "se_note": se_note,
    }


BOOT_SE = "bootstrap SE over 17 screens (honest)"
COL_SE = "sample SD / sqrt(4 colonies), within one screen (optimistic)"


# ------------------------------------------------------------------------ parts
def part_a() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Our 12-strain panel: measured and published, each tested against WT = 1."""
    df = pd.read_csv(osp.join(RESULTS, "run3_vs_reference.csv"))
    df["p_measured"] = p_greater(df.boot_fitness.to_numpy(), 1.0, df.boot_se.to_numpy())
    df["p_published"] = p_greater(
        df.costanzo_smf.to_numpy(), 1.0, df.costanzo_se.to_numpy()
    )
    df["sig_measured"] = bh_reject(df.p_measured.to_numpy())
    df["sig_published"] = bh_reject(df.p_published.to_numpy())
    rows = [
        summarise(
            "A. our panel, run 3, vs WT",
            df.boot_fitness.to_numpy(),
            df.boot_se.to_numpy(),
            1.0,
            "bootstrap SE over 3 plates",
        ),
        summarise(
            "A. our panel, Costanzo published, vs WT",
            df.costanzo_smf.to_numpy(),
            df.costanzo_se.to_numpy(),
            1.0,
            BOOT_SE,
        ),
    ]
    return df, rows


def part_b() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Genome-wide single-deletion headroom above WT (Costanzo 2016).

    The SMF table is emitted once per screen temperature with byte-identical values
    (verified r = 1.0), so it is deduplicated to one row per gene x allele -- otherwise
    the FDR would run on twice the true number of hypotheses.
    """
    smf = deletions_only(read_fitness_lmdb("smf_costanzo2016")).dropna(
        subset=["fitness", "fitness_se"]
    )
    smf = smf.drop_duplicates(subset=["gene0", "ptype0", "fitness"])
    rows = [
        summarise(
            "B. genome-wide single deletions vs WT",
            smf.fitness.to_numpy(),
            smf.fitness_se.to_numpy(),
            1.0,
            BOOT_SE,
        )
    ]
    return smf, rows


def part_c(smf: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Double-deletion headroom: vs WT, and vs the better of its own two singles."""
    dmf = deletions_only(read_fitness_lmdb("dmf_costanzo2016_5e5")).dropna(
        subset=["fitness", "fitness_se"]
    )
    key = smf.groupby("gene0").agg(
        smf_fitness=("fitness", "mean"), smf_se=("fitness_se", "mean")
    )
    for i in (0, 1):
        dmf = dmf.merge(
            key.rename(columns={"smf_fitness": f"smf{i}", "smf_se": f"smf{i}_se"}),
            left_on=f"gene{i}",
            right_index=True,
            how="left",
        )
    j = dmf.dropna(subset=["smf0", "smf1"]).copy()
    j["smf_best"] = j[["smf0", "smf1"]].max(axis=1)
    j["smf_best_se"] = np.where(j.smf0 >= j.smf1, j.smf0_se, j.smf1_se)
    j["se_diff"] = np.hypot(j.fitness_se, j.smf_best_se)

    rows = [
        summarise(
            "C. genome-wide double deletions vs WT",
            dmf.fitness.to_numpy(),
            dmf.fitness_se.to_numpy(),
            1.0,
            COL_SE,
        ),
        summarise(
            "C. genome-wide double deletions vs best own single",
            j.fitness.to_numpy(),
            j.se_diff.to_numpy(),
            j.smf_best.to_numpy(),
            COL_SE + " + " + BOOT_SE,
        ),
    ]
    # the actual rung: beats its best single AND beats WT (both must hold)
    both = (j.fitness > j.smf_best) & (j.fitness > 1.0)
    p_rung = np.maximum(
        p_greater(j.fitness.to_numpy(), j.smf_best.to_numpy(), j.se_diff.to_numpy()),
        p_greater(j.fitness.to_numpy(), 1.0, j.fitness_se.to_numpy()),
    )
    rej = bh_reject(p_rung)
    excess = (j.fitness - np.maximum(j.smf_best, 1.0)).to_numpy()
    rows.append(
        {
            "comparison": "C. RUNG: double > best own single AND > WT",
            "n": int(len(j)),
            "n_above": int(both.sum()),
            "frac_above": float(both.mean()),
            "n_sig_above": int(rej.sum()),
            "frac_sig_above": float(rej.mean()),
            "median_excess_of_sig": (
                float(np.median(excess[rej])) if rej.any() else np.nan
            ),
            "max_value": float(j.fitness.max()),
            "q99_value": float(j.fitness.quantile(0.99)),
            "se_note": COL_SE + " + " + BOOT_SE,
        }
    )
    return j, rows


def part_d() -> tuple[dict[int, pd.DataFrame], list[dict[str, Any]]]:
    """The Kuzmin 2018 ladder: singles, doubles and triples from one 26 C screen."""
    orders: dict[int, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for order, name in ((1, "smf_kuzmin2018"), (2, "dmf_kuzmin2018"), (3, "tmf_kuzmin2018")):
        df = deletions_only(read_fitness_lmdb(name)).dropna(subset=["fitness"])
        orders[order] = df
        if df.fitness_se.notna().all():
            rows.append(
                summarise(
                    f"D. Kuzmin 2018 order-{order} deletions vs WT",
                    df.fitness.to_numpy(),
                    df.fitness_se.to_numpy(),
                    1.0,
                    COL_SE,
                )
            )
        else:
            # Kuzmin releases singles without an uncertainty column -> point estimates only
            rows.append(
                {
                    "comparison": f"D. Kuzmin 2018 order-{order} deletions vs WT",
                    "n": int(len(df)),
                    "n_above": int((df.fitness > 1.0).sum()),
                    "frac_above": float((df.fitness > 1.0).mean()),
                    "n_sig_above": -1,
                    "frac_sig_above": np.nan,
                    "median_excess_of_sig": np.nan,
                    "max_value": float(df.fitness.max()),
                    "q99_value": float(df.fitness.quantile(0.99)),
                    "se_note": "no uncertainty released for Kuzmin singles",
                }
            )
    return orders, rows


# ----------------------------------------------------------------------- figure
def make_figure(
    panel: pd.DataFrame,
    smf: pd.DataFrame,
    dbl: pd.DataFrame,
    orders: dict[int, pd.DataFrame],
    out_stem: str,
) -> None:
    """Four panels: our panel, genome-wide singles, doubles, and the Kuzmin ladder."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            # set explicitly: `medium` resolves against a style that is not font.size here
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "legend.fontsize": 5,
        }
    )
    fig, axes = plt.subplots(
        2, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(104.0))
    )

    # (a) our 12 strains: measured vs published
    ax = axes[0, 0]
    p = panel.sort_values("boot_fitness").reset_index(drop=True)
    y = np.arange(len(p))
    ax.errorbar(
        p.costanzo_smf, y + 0.18, xerr=p.costanzo_se, fmt="s", ms=2.2, lw=0.6,
        color=PLOT_PALETTE[4], label="Costanzo 2016", capsize=1.2,
    )
    ax.errorbar(
        p.boot_fitness, y - 0.18, xerr=p.boot_se, fmt="o", ms=2.2, lw=0.6,
        color=PLOT_PALETTE[0], label="this assay (run 3)", capsize=1.2,
    )
    ax.axvline(1.0, color="black", lw=0.6, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(p.strain, fontsize=5)
    ax.set_xlabel("fitness (WT = 1)")
    ax.set_title("a  12-strain panel: none above WT", fontsize=6, loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0.4, 1.3)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))

    # (b) genome-wide single deletions
    ax = axes[0, 1]
    ax.hist(smf.fitness.to_numpy(), bins=np.arange(0.0, 1.62, 0.02),
            color=PLOT_PALETTE[1], lw=0)
    ax.axvline(1.0, color="black", lw=0.6, ls="--")
    ax.annotate(
        f"ceiling {smf.fitness.max():.3f}", xy=(smf.fitness.max(), 0),
        xytext=(1.18, ax.get_ylim()[1] * 0.55), fontsize=5,
        arrowprops={"arrowstyle": "->", "lw": 0.5},
    )
    ax.set_xlabel("single-deletion fitness")
    ax.set_ylabel("strains")
    ax.set_title(f"b  genome-wide singles (n = {len(smf):,})", fontsize=6, loc="left")
    ax.set_xlim(0.0, 1.6)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))

    # (c) genome-wide doubles, highlighting those beating their best single
    ax = axes[1, 0]
    bins = np.arange(0.0, 1.62, 0.02)
    ax.hist(dbl.fitness.to_numpy(), bins=bins, color=PLOT_PALETTE[5], lw=0,
            label="all doubles")
    ax.hist(dbl.fitness[dbl.fitness > dbl.smf_best].to_numpy(), bins=bins,
            color=PLOT_PALETTE[2], lw=0, label="> best own single")
    ax.axvline(1.0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("double-deletion fitness")
    ax.set_ylabel("strains")
    ax.set_title(f"c  genome-wide doubles (n = {len(dbl):,})", fontsize=6, loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(0.0, 1.6)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))

    # (d) the ladder: stacking deletions moves the population AWAY from WT.
    # `max` is one outlier colony, so the honest statistic is the FRACTION above WT.
    ax = axes[1, 1]
    xs = sorted(orders)
    frac = [float((orders[o].fitness > 1.0).mean()) for o in xs]
    frac_sig = [
        float(bh_reject(p_greater(orders[o].fitness.to_numpy(), 1.0, orders[o].fitness_se.to_numpy())).mean())
        if orders[o].fitness_se.notna().all()
        else np.nan
        for o in xs
    ]
    idx = np.arange(len(xs))
    ax.bar(idx - 0.19, frac, 0.36, color=PLOT_PALETTE_FILL[0], edgecolor="black",
           lw=0.4, label="above WT (point estimate)")
    ax.bar(idx + 0.19, np.nan_to_num(frac_sig), 0.36, color=PLOT_PALETTE[0],
           edgecolor="black", lw=0.4, label="above WT (FDR 0.05)")
    for i, (f, s, o) in enumerate(zip(frac, frac_sig, xs)):
        ax.annotate(f"{f:.1%}", (i - 0.19, f), textcoords="offset points",
                    xytext=(0, 2), ha="center", fontsize=5)
        ax.annotate(
            "no SE\nreleased" if np.isnan(s) else f"{s:.1%}",
            (i + 0.19, 0 if np.isnan(s) else s),
            textcoords="offset points", xytext=(0, 2), ha="center", fontsize=5,
        )
        ax.annotate(f"q99 = {orders[o].fitness.quantile(0.99):.2f}", (i, 0.30),
                    ha="center", fontsize=5)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{o}\n(n = {len(orders[o]):,})" for o in xs])
    ax.set_xlabel("deletions stacked")
    ax.set_ylabel("fraction of strains above WT")
    ax.set_title("d  Kuzmin 2018 ladder, one 26 C screen", fontsize=6, loc="left")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0.0, 0.40)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    for ax in axes.ravel():
        for s in ax.spines.values():
            s.set_visible(True)
        ax.tick_params(width=0.5, length=2)
        ax.tick_params(which="minor", length=0)

    fig.tight_layout(pad=0.4)
    fig.savefig(out_stem + ".png", dpi=300)
    savefig_true_size_svg(fig, out_stem + ".svg")
    plt.close(fig)


# ------------------------------------------------------------------------- main
def main() -> None:
    """Run every headroom analysis, then write the table, figure and JSON summary."""
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    panel, rows_a = part_a()
    smf, rows_b = part_b()
    dbl, rows_c = part_c(smf)
    orders, rows_d = part_d()

    table = pd.DataFrame(rows_a + rows_b + rows_c + rows_d)
    table.to_csv(osp.join(RESULTS, "run3_ladder_feasibility.csv"), index=False)
    panel.to_csv(osp.join(RESULTS, "run3_panel_vs_wt_tests.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 44):
        print(table.drop(columns=["se_note"]).to_string(index=False))
    print()

    rung = table.loc[table.comparison.str.startswith("C. RUNG")].iloc[0]
    summary = {
        "n_doubles_next_round": N_DOUBLES_NEXT_ROUND,
        "rung_rate_point_estimate": float(rung.frac_above),
        "rung_rate_fdr_significant": float(rung.frac_sig_above),
        "expected_rungs_if_random_point": float(N_DOUBLES_NEXT_ROUND * rung.frac_above),
        "expected_rungs_if_random_significant": float(
            N_DOUBLES_NEXT_ROUND * rung.frac_sig_above
        ),
        "p_zero_significant_rungs_if_random": float(
            (1 - rung.frac_sig_above) ** N_DOUBLES_NEXT_ROUND
        ),
        "enrichment_needed_for_one_expected_rung": (
            float(1.0 / (N_DOUBLES_NEXT_ROUND * rung.frac_sig_above))
            if rung.frac_sig_above > 0
            else None
        ),
        "single_deletion_ceiling": float(smf.fitness.max()),
        "double_deletion_ceiling": float(dbl.fitness.max()),
        "triple_deletion_ceiling": float(orders[3].fitness.max()),
    }
    print(json.dumps(summary, indent=2))
    with open(osp.join(RESULTS, "run3_ladder_feasibility_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    make_figure(panel, smf, dbl, orders, osp.join(IMG_DIR, "ladder_feasibility"))
    print("\nfigure:", osp.join(IMG_DIR, "ladder_feasibility.svg"))


if __name__ == "__main__":
    main()
