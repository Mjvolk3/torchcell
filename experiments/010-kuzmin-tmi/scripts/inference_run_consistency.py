# experiments/010-kuzmin-tmi/scripts/inference_run_consistency.py
# [[experiments.010-kuzmin-tmi.scripts.inference_run_consistency]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/inference_run_consistency

"""Is one 010 checkpoint's prediction for a triple the same across inference runs?

Checkpoint ``c7671wgj`` scored three separate 010 inference spaces:

    inference_1     4,370,595 triples over a 526-gene panel
    inference_2       479,195 triples
    inference_3   465,735,532 triples, genome-wide; this is the space the
                  panel-12 and panel-24 selections consumed

The model is a deterministic function of the perturbed gene set, so a triple
scored in two runs must receive the same prediction, and a gene's average
predicted effect must be the same in both. This script checks that, three ways:

  * direct comparison of the prediction for triples that appear in two runs,
  * correlation of per-gene mean predictions between runs, which averages over
    hundreds of triples per gene and so is not sampling-noise limited,
  * correlation of each run's per-gene mean prediction with the per-gene
    coefficient of the additive ridge from
    ``additive_baseline_gene_interaction.py``. That coefficient is fit on the
    measured training split and keyed on true systematic gene name, so it is an
    external anchor on gene identity: a run that handles gene identity
    correctly should track it, and one that does not, should not.
"""

import glob
import os
import os.path as osp
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv

from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
INFERENCE_ROOT = osp.join(DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi")
COEF_NPZ = osp.join(RESULTS_DIR, "additive_baseline_B1_coefficients.npz")

RUN = "c7671wgj"
RUNS = ["inference_1", "inference_2", "inference_3"]
# inference_3 is 4 GB; every Nth row group is a stratified sample of it.
ROW_GROUP_STRIDE = 15
MIN_GENE_OBS = 200


def parquet_for(run: str) -> str:
    matches = glob.glob(
        osp.join(INFERENCE_ROOT, run, "inferred", f"*-{RUN}-best-pearson-*.parquet")
    )
    assert len(matches) == 1, f"expected one {RUN} parquet in {run}, got {matches}"
    return matches[0]


def load_predictions(run: str) -> pd.DataFrame:
    path = parquet_for(run)
    pf = pq.ParquetFile(path)
    cols = ["gene1", "gene2", "gene3", "prediction"]
    if pf.num_row_groups <= ROW_GROUP_STRIDE:
        df = pq.read_table(path, columns=cols).to_pandas()
    else:
        df = pd.concat(
            [
                pf.read_row_group(gi, columns=cols).to_pandas()
                for gi in range(0, pf.num_row_groups, ROW_GROUP_STRIDE)
            ],
            ignore_index=True,
        )
    print(
        f"{run}: {pf.metadata.num_rows} rows on disk, {len(df)} loaded "
        f"({pf.num_row_groups} row groups)"
    )
    return df


def gene_means(df: pd.DataFrame) -> pd.DataFrame:
    long = pd.concat(
        [
            df[[c, "prediction"]].rename(columns={c: "gene"})
            for c in ("gene1", "gene2", "gene3")
        ]
    )
    return long.groupby("gene")["prediction"].agg(["mean", "count"])


def main() -> None:
    preds = {run: load_predictions(run) for run in RUNS}
    means = {run: gene_means(df) for run, df in preds.items()}
    rows: list[dict[str, object]] = []

    print("\nprediction for the SAME triple, across runs")
    for a, b in combinations(RUNS, 2):
        da = preds[a].assign(key=lambda d: d.gene1 + "|" + d.gene2 + "|" + d.gene3)
        db = preds[b].assign(key=lambda d: d.gene1 + "|" + d.gene2 + "|" + d.gene3)
        m = da[["key", "prediction"]].merge(
            db[["key", "prediction"]], on="key", suffixes=("_a", "_b")
        )
        if len(m) < 100:
            print(f"  {a} vs {b}: only {len(m)} shared triples, skipped")
            continue
        r = float(np.corrcoef(m.prediction_a, m.prediction_b)[0, 1])
        mad = float(np.abs(m.prediction_a - m.prediction_b).mean())
        print(f"  {a} vs {b}: n={len(m)}  Pearson {r: .4f}  mean|diff| {mad:.6f}")
        rows.append({"comparison": f"{a}|{b}", "quantity": "triple_pearson", "value": r})
        rows.append({"comparison": f"{a}|{b}", "quantity": "triple_n", "value": len(m)})
        rows.append(
            {"comparison": f"{a}|{b}", "quantity": "triple_mean_abs_diff", "value": mad}
        )

    print(f"\nper-gene mean prediction, across runs (genes with >={MIN_GENE_OBS} obs)")
    for a, b in combinations(RUNS, 2):
        j = means[a].join(means[b], lsuffix="_a", rsuffix="_b", how="inner")
        j = j[(j["count_a"] >= MIN_GENE_OBS) & (j["count_b"] >= MIN_GENE_OBS)]
        r = float(np.corrcoef(j["mean_a"], j["mean_b"])[0, 1])
        print(f"  {a} vs {b}: {len(j)} genes  Pearson {r: .4f}")
        rows.append(
            {"comparison": f"{a}|{b}", "quantity": "gene_mean_pearson", "value": r}
        )
        rows.append({"comparison": f"{a}|{b}", "quantity": "gene_mean_n", "value": len(j)})

    coef = np.load(COEF_NPZ, allow_pickle=True)
    beta = pd.Series(coef["beta"], index=list(coef["gene_names"]), name="beta")
    print("\nper-gene mean prediction vs the additive ridge coefficient")
    anchor = {}
    for run in RUNS:
        j = means[run].join(beta, how="inner")
        j = j[j["count"] >= MIN_GENE_OBS]
        r = float(np.corrcoef(j["mean"], j["beta"])[0, 1])
        anchor[run] = (j, r)
        print(f"  {run}: {len(j)} genes  Pearson {r: .4f}")
        rows.append(
            {"comparison": f"{run}|additive_beta", "quantity": "pearson", "value": r}
        )
        rows.append(
            {"comparison": f"{run}|additive_beta", "quantity": "n_genes", "value": len(j)}
        )

    # The three runs cover different gene sets, and inference_3 reaches many
    # barely-measured genes whose ridge coefficient is heavily shrunk. Repeat
    # the anchor test on the genes COMMON to all three runs so that the
    # comparison cannot be explained by which genes each run happened to score.
    common = set.intersection(
        *[
            set(means[run][means[run]["count"] >= MIN_GENE_OBS].index)
            for run in RUNS
        ]
    ) & set(beta.index)
    common = sorted(common)
    print(f"\nanchor test restricted to the {len(common)} genes common to all runs")
    for run in RUNS:
        sub = means[run].loc[common]
        r = float(np.corrcoef(sub["mean"], beta.loc[common])[0, 1])
        print(f"  {run}: Pearson {r: .4f}")
        rows.append(
            {
                "comparison": f"{run}|additive_beta_common_genes",
                "quantity": "pearson",
                "value": r,
            }
        )
    rows.append(
        {
            "comparison": "common_genes",
            "quantity": "n_genes",
            "value": len(common),
        }
    )

    out = osp.join(RESULTS_DIR, "inference_run_consistency.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")
    plot(anchor)


def plot(anchor: dict[str, tuple[pd.DataFrame, float]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(
        1,
        len(anchor),
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(52.0)),
        sharey=True,
    )
    for ax, (run, (j, r)) in zip(axes, anchor.items()):
        ax.scatter(
            j["beta"],
            j["mean"],
            s=3,
            color=PLOT_PALETTE[0],
            edgecolor="black",
            linewidth=0.2,
        )
        ax.set_xlabel("Additive ridge coefficient")
        ax.set_title(f"{run.replace('_', ' ')}, r = {r:.3f}", fontsize=6)
        for spine in ax.spines.values():
            spine.set_visible(True)
    axes[0].set_ylabel("Mean predicted interaction for the gene")
    fig.tight_layout()

    stem = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi", "inference_run_consistency")
    os.makedirs(osp.dirname(stem), exist_ok=True)
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"wrote {stem}.svg")


if __name__ == "__main__":
    main()
