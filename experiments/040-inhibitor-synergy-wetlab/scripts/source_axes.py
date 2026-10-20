# experiments/040-inhibitor-synergy-wetlab/scripts/source_axes.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.source_axes]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/source_axes
"""What the four pooled chemogenomic sources share, and on what axis they can be compared.

A joint model over Hoepfner 2014, Hillenmeyer 2008 HET, Wildenhain 2015 and Vanacloig 2022
has to be judged per source, because the four do not measure one quantity: different
ploidy, perturbation, medium, label and assay, and a shared (gene, compound) pair agrees
across sources barely above zero. This script fixes the comparison basis and measures the
one axis that might transfer, the dose.

FOUR MEASUREMENTS, each written to its own table.

1. ``source_basis.csv`` -- THE COMMON AXIS. Per source: the best simple compound-cold
   baseline (ridge or k-nearest-neighbor over eleven molecule encoders, experiment 031's
   ``per_dataset_baselines.py``), that source's reliability ceiling, and the ratio. The
   ratio is the axis a joint model is read on, because a source whose simple baseline is
   already near its ceiling has no headroom for pooling to recover, and a source with no
   served uncertainty has no ceiling and is reported without one.

2. ``source_agreement.csv`` -- WHY POOLING IS NOT AVERAGING. The oriented Spearman of each
   pair of sources over their shared (gene, compound) cells (experiment 033's
   ``store_against_plan.py``), beside the count of shared pairs. A joint model that treats
   the label as one quantity is asserting these are high.

3. ``dose_structure.csv`` -- THE DOSE AXIS, measured here. Hoepfner serves 83 compounds at
   two or more molar doses and Hillenmeyer HET 66. For every compound served at three or
   more, with the gene-by-dose matrix complete: the Spearman of each gene's response
   against dose, the correlation of profiles at ADJACENT doses, the correlation of the
   profiles at the extreme doses, and the Spearman of the per-gene spread against dose. A
   conserved profile whose spread grows with dose is a gain, which is what a FiLM scale on
   a compound token can express; a profile that reorders with dose is not.

4. ``dose_reference.csv`` -- WHAT TO COMPARE THAT WITH. The same profile correlation
   between two SCREENS at one dose (experiment 033's ``within_cell_reliability.csv``), so
   the adjacent-dose correlation is read against the repeat-measurement correlation rather
   than against one.

Run from the repository root:

    python experiments/040-inhibitor-synergy-wetlab/scripts/source_axes.py

Writes the four tables to ``results/`` and the figure ``source_axes.{svg,png}`` to
``ASSET_IMAGES_DIR/040-inhibitor-synergy-wetlab/``.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import spearmanr

sys.path.insert(0, osp.dirname(__file__))
from mixture_data import SPECS, STORE, load_cells  # noqa: E402
from mixture_rules import box, panel_letters, save, tenth_grid  # noqa: E402

from torchcell.utils import PLOT_PALETTE  # noqa: E402

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "040-inhibitor-synergy-wetlab", "results")

#: Experiment 031's per-dataset baselines and 033's pooled axes, by absolute path because
#: neither branch is landed. Every path read is recorded in ``source_axes_sources.csv``.
WORKTREES = "/home/michaelvolk/Documents/projects/torchcell.worktrees"
BASELINES_CSV = osp.join(
    WORKTREES,
    "exp/031-env-chemgen-vanacloig-hillenmeyer/experiments",
    "031-env-chemgen-inhibitor-tolerance/results/per_dataset_baselines_summary.csv",
)
POOLED = osp.join(WORKTREES, "033-build-002/experiments/033-env-chemgen-pooled/results")
AGREEMENT_CSV = osp.join(POOLED, "cross_source_cells.csv")
REPEAT_CSV = osp.join(POOLED, "within_cell_reliability.csv")
AXES_CSV = osp.join(POOLED, "store_axes.csv")

#: The 031 dataset names, the 033 names, and the label this experiment plots.
SOURCES: dict[str, dict[str, str]] = {
    "hoepfner2014": {"pooled": "Hoepfner 2014", "label": "Hoepfner"},
    "hillenmeyer2008_het": {"pooled": "Hillenmeyer HET", "label": "Hillenmeyer HET"},
    "wildenhain2015": {"pooled": "Wildenhain 2015", "label": "Wildenhain"},
    "vanacloig2022": {"pooled": "Vanacloig 2022", "label": "Vanacloig"},
}
#: A dose series needs at least this many doses to say anything about shape, and the
#: matrix is kept only where this many genes are measured at every one of them.
MIN_DOSES = 3
MIN_GENES = 500
#: The target a joint model is scored on: the compound-specific part of a profile, the
#: gene main effect removed. The raw target is dominated by the gene mean.
TARGET = "centered"


def common_basis() -> pd.DataFrame:
    """Per source: its best simple compound-cold baseline against its own ceiling."""
    baselines = pd.read_csv(BASELINES_CSV)
    baselines = baselines[baselines["target"] == TARGET]
    rows = []
    for name, meta in SOURCES.items():
        block = baselines[baselines["dataset"] == name]
        assert len(block), f"{name} has no baseline row in {BASELINES_CSV}"
        best = block.loc[block["spearman_median"].idxmax()]
        null = block[block["model"] == "random_neighbor"]
        rows.append(
            {
                "source": meta["label"],
                "scheme": best["scheme"],
                "folds": int(best["folds"]),
                "best_features": best["features"],
                "best_model": best["model"],
                "spearman_median": float(best["spearman_median"]),
                "ceiling_median": float(best["ceiling_median"]),
                "fraction_of_ceiling": float(best["frac_of_ceiling"]),
                "null_random_neighbor": (
                    float(null["spearman_median"].iloc[0])
                    if len(null)
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def agreement() -> pd.DataFrame:
    """The oriented Spearman between each pair of sources over their shared cells."""
    frame = pd.read_csv(AGREEMENT_CSV)
    label = {meta["pooled"]: meta["label"] for meta in SOURCES.values()}
    return pd.DataFrame(
        {
            "a": frame["a"].map(label),
            "b": frame["b"].map(label),
            "shared_genes": frame["shared_genes"],
            "shared_compounds": frame["shared_compounds"],
            "shared_pairs": frame["shared_pairs"],
            "spearman_oriented": frame["spearman_oriented"],
        }
    ).sort_values("shared_pairs", ascending=False, ignore_index=True)


def dose_structure(store: str) -> pd.DataFrame:
    """One row per compound served at ``MIN_DOSES`` or more molar doses in one source.

    The matrix is genes by doses, kept only for genes measured at every dose, so the
    correlations compare the same gene set across doses. Read from the 040 store, one
    row per served record; repeated measurements of a cell average in the pivot.
    """
    cells = load_cells(store, frozenset(spec.dataset for spec in SPECS.values()))
    frame = cells[cells["n_compounds"] == 1]
    frame = frame[np.isfinite(frame["log10_molar"])].copy()
    frame["y"] = frame["response"].astype(float)
    rows = []
    for (dataset, arm), block in frame.groupby(["dataset", "perturbation_type"]):
        doses_per_compound = block.groupby("inchikeys")["log10_molar"].nunique()
        for inchikey in doses_per_compound[doses_per_compound >= MIN_DOSES].index:
            cells = block[block["inchikeys"] == inchikey]
            matrix = cells.pivot_table(
                index="genes", columns="log10_molar", values="y", aggfunc="mean"
            )
            matrix = matrix.dropna(thresh=matrix.shape[1])
            if len(matrix) < MIN_GENES:
                continue
            doses = np.asarray(matrix.columns, dtype=float)
            values = matrix.to_numpy()
            per_gene = np.array([spearmanr(doses, row).statistic for row in values])
            adjacent = [
                spearmanr(values[:, j], values[:, j + 1]).statistic
                for j in range(len(doses) - 1)
            ]
            spread = values.std(axis=0)
            rows.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "compound": cells["compound_names"].iloc[0],
                    "inchikey": inchikey,
                    "n_doses": len(doses),
                    "log10_dose_span": float(doses.max() - doses.min()),
                    "genes": int(len(matrix)),
                    "median_gene_dose_spearman": float(np.median(per_gene)),
                    "frac_genes_falling_with_dose": float((per_gene < -0.5).mean()),
                    "adjacent_dose_profile_spearman": float(np.median(adjacent)),
                    "extreme_dose_profile_spearman": float(
                        spearmanr(values[:, 0], values[:, -1]).statistic
                    ),
                    "spread_vs_dose_spearman": float(
                        spearmanr(doses, spread).statistic
                    ),
                }
            )
    assert rows, "no compound carried a usable dose series"
    return pd.DataFrame(rows).sort_values(
        ["dataset", "arm", "compound"], ignore_index=True
    )


def dose_reference() -> pd.DataFrame:
    """The profile correlation between two SCREENS at one dose, the repeat reference."""
    frame = pd.read_csv(REPEAT_CSV)
    return frame[
        [
            "dataset",
            "functional_dose",
            "environments",
            "screen_pairs",
            "spearman_median",
            "pearson_median",
            "ceiling_r_truth",
        ]
    ].copy()


def plot(
    basis: pd.DataFrame, pairs: pd.DataFrame, doses: pd.DataFrame, repeats: pd.DataFrame
) -> list[str]:
    """Four panels: the common axis, the label problem, and the dose axis twice."""
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
    from torchcell.utils import PANEL_WIDTHS_MM, mm_to_in

    width = PANEL_WIDTHS_MM["full"]
    # NOT constrained_layout: panel_letters shrinks the top margin with
    # subplots_adjust, which a constrained layout overrides, and the letters are then
    # clipped by the true-size export without a warning
    # two rows rather than four across: the source names and the axis labels are long,
    # and at a quarter of 179 mm each they collide with the next panel
    fig, axes = plt.subplots(2, 2, figsize=(mm_to_in(width), mm_to_in(108)))
    axes = np.ravel(axes)

    # a: the best simple baseline and its ceiling, per source
    ax = axes[0]
    order = basis.sort_values("spearman_median", ascending=False, ignore_index=True)
    x = np.arange(len(order))
    ax.bar(
        x,
        order["spearman_median"],
        color=[PLOT_PALETTE[i] for i in range(len(order))],
        edgecolor="black",
        linewidth=0.5,
        label="best ridge or kNN",
    )
    ceiling = order["ceiling_median"].to_numpy()
    for i, value in enumerate(ceiling):
        if np.isfinite(value):
            ax.plot(
                [i - 0.42, i + 0.42],
                [value, value],
                color="black",
                linewidth=0.8,
                label="reliability ceiling" if i == 0 else None,
            )
    ax.plot(
        x,
        order["null_random_neighbor"],
        linestyle="none",
        marker="_",
        color=PLOT_PALETTE[5],
        markersize=4,
        label="random-neighbor null",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(order["source"], rotation=20, ha="right")
    ax.set_ylabel("compound-cold Spearman, centered")
    ax.set_ylim(0, 1)
    tenth_grid(ax, "y")
    # center right, not upper: the ceiling lines run the full panel width near the top
    ax.legend(loc="center right", frameon=False, handlelength=1.2)
    box(ax)

    # b: a shared (gene, compound) pair does not agree across sources
    ax = axes[1]
    y = np.arange(len(pairs))
    ax.barh(
        y,
        pairs["spearman_oriented"],
        color=PLOT_PALETTE[1],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{a} / {b}" for a, b in zip(pairs["a"], pairs["b"], strict=True)]
    )
    ax.set_xlabel("oriented Spearman over shared cells")
    ax.set_xlim(0, 0.5)
    ax.invert_yaxis()
    for i, (n, value) in enumerate(
        zip(pairs["shared_pairs"], pairs["spearman_oriented"], strict=True)
    ):
        ax.text(
            value + 0.012,
            i,
            f"{n:,} pairs",
            va="center",
            ha="left",
            fontsize=5,
            color="black",
        )
    box(ax)

    # c: the profile is conserved across dose, against the repeat-screen reference
    ax = axes[2]
    labels, positions = [], []
    for i, (name, block) in enumerate(doses.groupby("dataset")):
        values = block["adjacent_dose_profile_spearman"].dropna()
        ax.boxplot(
            values,
            positions=[i],
            widths=0.6,
            patch_artist=True,
            boxprops={"facecolor": PLOT_PALETTE[i], "linewidth": 0.5},
            medianprops={"color": "black", "linewidth": 0.8},
            whiskerprops={"linewidth": 0.5},
            capprops={"linewidth": 0.5},
            flierprops={"marker": ".", "markersize": 1.5},
        )
        labels.append(short(name))
        positions.append(i)
    for _, row in repeats.iterrows():
        match = [
            i
            for i, name in enumerate(sorted(doses["dataset"].unique()))
            if short(name) in row["dataset"] or row["dataset"] in short(name)
        ]
        if match:
            ax.plot(
                [match[0] - 0.3, match[0] + 0.3],
                [row["spearman_median"]] * 2,
                color="black",
                linewidth=0.8,
                linestyle="--",
            )
    ax.plot([], [], color="black", linewidth=0.8, linestyle="--", label="repeat screen")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("adjacent-dose profile Spearman")
    ax.set_ylim(-0.2, 1)
    ax.legend(loc="upper right", frameon=False, handlelength=1.4)
    box(ax)

    # d: the spread grows with dose, compound by compound
    ax = axes[3]
    for i, (name, block) in enumerate(doses.groupby("dataset")):
        ax.plot(
            block["log10_dose_span"],
            block["spread_vs_dose_spearman"],
            linestyle="none",
            marker="o",
            markersize=2,
            markerfacecolor=PLOT_PALETTE[i],
            markeredgecolor="black",
            markeredgewidth=0.2,
            label=short(name),
        )
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("log10 span of the dose series")
    ax.set_ylabel("spread against dose, Spearman")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="lower right", frameon=False, handlelength=0.8)
    box(ax)

    fig.tight_layout()
    panel_letters(fig, axes)
    return save(fig, "source_axes")


def short(dataset: str) -> str:
    """``EnvChemgenHoepfner2014Dataset`` as ``Hoepfner``."""
    for meta in SOURCES.values():
        stem = meta["label"].split()[0]
        if stem.lower() in dataset.lower():
            return meta["label"]
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default=STORE)
    args = parser.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    basis = common_basis()
    pairs = agreement()
    repeats = dose_reference()
    doses = dose_structure(args.store)
    for frame, name in (
        (basis, "source_basis"),
        (pairs, "source_agreement"),
        (doses, "dose_structure"),
        (repeats, "dose_reference"),
    ):
        frame.to_csv(osp.join(RESULTS, f"{name}.csv"), index=False)
        print(f"\n{name}")
        print(frame.to_string(index=False), flush=True)
    pd.DataFrame(
        [
            {"table": "source_basis", "read": BASELINES_CSV},
            {"table": "source_agreement", "read": AGREEMENT_CSV},
            {"table": "dose_reference", "read": REPEAT_CSV},
            {"table": "dose_structure", "read": args.store},
            {
                "table": "store_axes (dose basis quoted in mixture_data)",
                "read": AXES_CSV,
            },
        ]
    ).to_csv(osp.join(RESULTS, "source_axes_sources.csv"), index=False)

    summary = (
        doses.groupby("dataset")
        .agg(
            compounds=("compound", "size"),
            median_doses=("n_doses", "median"),
            adjacent=("adjacent_dose_profile_spearman", "median"),
            extreme=("extreme_dose_profile_spearman", "median"),
            spread_vs_dose=("spread_vs_dose_spearman", "median"),
            median_gene_dose_spearman=("median_gene_dose_spearman", "median"),
        )
        .round(3)
    )
    print("\ndose_structure, median over compounds")
    print(summary.to_string(), flush=True)
    for path in plot(basis, pairs, doses, repeats):
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
