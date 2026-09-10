#!/usr/bin/env python
# experiments/010-kuzmin-tmi/scripts/inference_4_measured_diagnostic.py
# [[experiments.010-kuzmin-tmi.scripts.inference_4_measured_diagnostic]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/inference_4_measured_diagnostic
"""The 4,877 inference_4 triples that the 010 build already measured, used as a probe.

WHY NOT JUST DROP THEM. The obvious response to finding measured combinations inside a
nominated space is to filter them out. That throws away the only ground truth that
exists INSIDE this space. Labeling them by 010 split turns the same overlap into the
diagnostic the space otherwise has no way to run:

  train  the model fit these. Agreement here is a SANITY check, not evidence. If it is
         poor, something is wrong with the scoring path rather than with the biology.
  val    held out. Honest.
  test   held out. Honest.

The reading is the CONTRAST. If train agrees and val/test do not, the ranking is
recalling rather than generalizing, and the 99.99 percent of the space with no
measurement should be trusted less in proportion. If val and test track train, the tail
is doing something real.

ZOOM ON THE TAIL. Aggregate agreement over all 4,877 is dominated by the bulk near zero,
which is not where nominations come from. So every statistic is also computed inside
rank windows of the ensemble ranking, which is the population a panel is actually drawn
from.

NO GPU, NO INFERENCE. Predictions for the whole 41,877,232-triple space already exist as
the four Parquet shards per checkpoint. This is a join, so it costs a Parquet read and an
LMDB pass. Run it on CPU while the GPUs are training.

Run from repo root, or via
  sbatch experiments/010-kuzmin-tmi/scripts/gh_inference_4_measured_diagnostic.slurm

Outputs, under results/inference_4/:
  measured_in_space.csv            every measured triple, split, measured tau, predictions
  measured_agreement_by_split.csv  agreement statistics per split, whole set and by window
  measured_diagnostic.json         the counts quoted in prose
  $ASSET_IMAGES_DIR/010-kuzmin-tmi/inference_4_measured_diagnostic.{png,svg}
"""

import glob
import json
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")

import lmdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from scipy import stats as sps
from tqdm import tqdm

from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results", "inference_4")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")
BUILD_010 = osp.join(
    DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build"
)
BASE = osp.join(DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/inference_4")

CHECKPOINTS = {"lzs9pcj3": "M01", "yv4r30bi": "M02", "c7671wgj": "M03"}
SPLITS = ("train", "val", "test")
SPLIT_COLOR = {"train": PLOT_PALETTE[0], "val": PLOT_PALETTE[1], "test": PLOT_PALETTE[2]}
# Rank windows of the ensemble ranking. A panel is drawn from the first of these, so an
# aggregate over the whole space would answer a question nobody asked.
WINDOWS = [1_000, 10_000, 100_000, 1_000_000, None]
POSITIVE_CALL, P_CUT = 0.08, 0.05
LABEL_SD = 0.06326
# A correlation on fewer than this many points is not reportable; the tail windows hold
# only a handful of held-out triples and a line through them would invent a trend.
MIN_N_FOR_R = 10


def set_plot_style():
    plt.rcParams.update(
        {
            "font.family": "Arial", "font.size": 6, "axes.labelsize": 6,
            "axes.titlesize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
            "legend.fontsize": 5, "legend.title_fontsize": 5, "figure.titlesize": 6,
            "svg.fonttype": "none", "axes.linewidth": 0.5,
            "savefig.bbox": None, "savefig.pad_inches": 0.0,
        }
    )


def load_010():
    """Identity -> (split, measured tau, p-value)."""
    with open(osp.join(BUILD_010, "data_module_cache/index_seed_42.json")) as f:
        index = json.load(f)
    split_of = {i: s for s in SPLITS for i in index[s]}
    out: dict[tuple[str, ...], tuple[str, float, float]] = {}
    env = lmdb.open(osp.join(BUILD_010, "processed/lmdb"), readonly=True, lock=False,
                    readahead=False)
    with env.begin() as txn:
        n = txn.stat()["entries"]
        for i in tqdm(range(n), desc="reading the 010 build"):
            rec = json.loads(txn.get(str(i).encode()))
            exp = rec[0]["experiment"]
            ident = tuple(sorted({p["systematic_gene_name"]
                                  for p in exp["genotype"]["perturbations"]}))
            phen = exp["phenotype"]
            out[ident] = (split_of[i], phen.get("gene_interaction"),
                          phen.get("gene_interaction_p_value"))
    env.close()
    print(f"010: {n:,} records, {len(out):,} identities, "
          + ", ".join(f"{s} {sum(1 for v in out.values() if v[0] == s):,}"
                      for s in SPLITS))
    return out


def load_checkpoint(tag: str) -> np.ndarray:
    files = sorted(glob.glob(osp.join(BASE, "inferred", f"*{tag}*shard*.parquet")))
    if len(files) != 4:
        raise SystemExit(f"{tag}: expected 4 shard files, found {len(files)}")
    parts, cursor = [], 0
    for f in files:
        t = pq.read_table(f, columns=["index", "prediction"])
        i = t["index"].to_numpy()
        if i[0] != cursor or not np.all(np.diff(i) == 1):
            raise SystemExit(f"{tag}: shard {f} is not contiguous from {cursor}")
        cursor = int(i[-1]) + 1
        parts.append(t["prediction"].to_numpy())
    return np.concatenate(parts).astype(np.float32)


def agreement(sub: pd.DataFrame, label: str) -> dict:
    """Agreement between prediction and measurement on one subset."""
    row = {"subset": label, "n": int(len(sub))}
    if len(sub) < 3:
        return row
    x = sub["measured_tau"].to_numpy(dtype=float)
    y = sub["pred_mean"].to_numpy(dtype=float)
    row["pearson_r"] = float(sps.pearsonr(x, y)[0])
    row["pearson_p"] = float(sps.pearsonr(x, y)[1])
    row["spearman_rho"] = float(sps.spearmanr(x, y)[0])
    row["mean_measured"] = float(x.mean())
    row["mean_predicted"] = float(y.mean())
    row["mean_signed_error"] = float((y - x).mean())
    row["n_over_predicted"] = int((y > x).sum())
    # Regress MEASURED on PREDICTED: slope 1 is calibrated, slope < 1 is inflation.
    row["calibration_slope"] = float(np.polyfit(y, x, 1)[0])
    called = (x > POSITIVE_CALL)
    if sub["measured_p"].notna().all():
        called = called & (sub["measured_p"].to_numpy(dtype=float) < P_CUT)
    row["n_measured_positive"] = int(called.sum())
    row["precision_measured_positive"] = float(called.mean())
    return row


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    known = load_010()

    print("loading the space ...")
    preds = {t: load_checkpoint(t) for t in CHECKPOINTS}
    stack = np.stack([preds[t] for t in CHECKPOINTS], axis=1)
    mean, worst = stack.mean(axis=1), stack.min(axis=1)

    idx = pq.read_table(osp.join(BASE, "triple_index.parquet"),
                        columns=["gene1", "gene2", "gene3"])
    g1 = idx["gene1"].to_numpy(zero_copy_only=False)
    g2 = idx["gene2"].to_numpy(zero_copy_only=False)
    g3 = idx["gene3"].to_numpy(zero_copy_only=False)
    n_space = len(g1)
    if n_space != len(mean):
        raise SystemExit("triple_index does not align with the predictions")

    # Rank of every triple in the ensemble ranking, 1 = most positive. Needed so each
    # measured triple can be placed inside the tail rather than only in the bulk.
    order = np.argsort(mean)[::-1]
    rank = np.empty(n_space, dtype=np.int64)
    rank[order] = np.arange(1, n_space + 1)

    # Order-free int64 key over the 010 vocabulary; a gene 010 never saw cannot match.
    vocab = sorted({g for ident in known for g in ident})
    code = {g: i for i, g in enumerate(vocab)}
    N = len(code) + 1

    def enc(arr):
        return np.array([code.get(g, -1) for g in arr], dtype=np.int64)

    c = np.sort(np.stack([enc(g1), enc(g2), enc(g3)], axis=1), axis=1)
    ok = c[:, 0] >= 0
    keys = c[:, 0] * N * N + c[:, 1] * N + c[:, 2]

    key_to_ident = {}
    for ident in known:
        cs = sorted(code[g] for g in ident)
        if len(cs) == 3:
            key_to_ident[cs[0] * N * N + cs[1] * N + cs[2]] = ident
    member = np.fromiter(key_to_ident.keys(), dtype=np.int64, count=len(key_to_ident))
    hit = ok & np.isin(keys, member)
    ii = np.where(hit)[0]
    print(f"\nmeasured triples inside the space: {len(ii):,} of {n_space:,}")

    idents = [key_to_ident[int(k)] for k in keys[ii]]
    df = pd.DataFrame({
        "index": ii,
        "gene1": g1[ii], "gene2": g2[ii], "gene3": g3[ii],
        "split": [known[t][0] for t in idents],
        "measured_tau": [known[t][1] for t in idents],
        "measured_p": [known[t][2] for t in idents],
        **{CHECKPOINTS[t]: preds[t][ii] for t in CHECKPOINTS},
        "pred_mean": mean[ii], "pred_worst": worst[ii], "rank": rank[ii],
    }).sort_values("rank").reset_index(drop=True)
    df.to_csv(osp.join(RESULTS_DIR, "measured_in_space.csv"), index=False)
    print(df["split"].value_counts().to_string())

    rows = []
    for s in SPLITS:
        rows.append({**agreement(df[df["split"] == s], f"all | {s}"), "window": "all",
                     "split": s})
    rows.append({**agreement(df, "all | pooled"), "window": "all", "split": "pooled"})
    for w in WINDOWS:
        sub_w = df if w is None else df[df["rank"] <= w]
        lab = "all" if w is None else f"top {w:,}"
        for s in SPLITS:
            rows.append({**agreement(sub_w[sub_w["split"] == s], f"{lab} | {s}"),
                         "window": lab, "split": s})
        rows.append({**agreement(sub_w[sub_w["split"] != "train"], f"{lab} | held out"),
                     "window": lab, "split": "held out"})
    agree = pd.DataFrame(rows).drop_duplicates(subset=["subset"])
    # Enrichment of real positive calls over each split's OWN base rate. This is the
    # number that answers whether the ranking generalizes: a held-out enrichment well
    # above 1 means the ordering finds real interactions it was not trained on.
    base = {r["split"]: r.get("precision_measured_positive")
            for r in rows if r.get("window") == "all"}
    agree["base_rate"] = agree["split"].map(base)
    agree["positive_enrichment"] = (
        agree["precision_measured_positive"] / agree["base_rate"]
    )
    agree.to_csv(osp.join(RESULTS_DIR, "measured_agreement_by_split.csv"), index=False)

    print("\n=== agreement between prediction and measurement ===")
    cols = ["subset", "n", "pearson_r", "spearman_rho", "calibration_slope",
            "mean_signed_error", "n_measured_positive", "precision_measured_positive",
            "positive_enrichment"]
    print(agree[[c for c in cols if c in agree.columns]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # RANGE RESTRICTION. The roster drops essential genes, genes below 0.9 single-mutant
    # fitness, and mitochondrial genes, so interactions among what survives are milder
    # than in the full build. A correlation is attenuated by exactly that compression,
    # so the training r here is not comparable to the model's headline validation
    # Pearson unless the spread is quoted beside it.
    sd_here = float(df["measured_tau"].std())
    range_restriction = {
        "measured_tau_sd_in_space": sd_here,
        "label_sd_full_build": LABEL_SD,
        "ratio": float(sd_here / LABEL_SD),
    }
    print(f"\nrange restriction: measured tau SD is {sd_here:.4f} inside the space "
          f"against {LABEL_SD:.4f} over the whole build, a ratio of "
          f"{sd_here / LABEL_SD:.2f}")

    summary = {
        "n_space": int(n_space),
        "n_measured_in_space": int(len(df)),
        "range_restriction": range_restriction,
        "share_measured": float(len(df) / n_space),
        "by_split": {s: int((df["split"] == s).sum()) for s in SPLITS},
        "n_held_out": int((df["split"] != "train").sum()),
        "positive_call": POSITIVE_CALL,
        "label_sd": LABEL_SD,
        "agreement": agree.to_dict(orient="records"),
        "top_measured": df.head(25)[
            ["rank", "gene1", "gene2", "gene3", "split", "measured_tau",
             "measured_p", "pred_mean"]
        ].to_dict(orient="records"),
    }
    with open(osp.join(RESULTS_DIR, "measured_diagnostic.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    plot(df, agree, summary, osp.join(IMAGES_DIR, "inference_4_measured_diagnostic"))
    print(f"\nwrote {RESULTS_DIR} and figures to {IMAGES_DIR}")


def _letter(ax, letter):
    ax.text(-0.17, 1.06, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left")


def plot(df, agree, summary, out_stem):
    set_plot_style()
    fig, axes2 = plt.subplots(
        2, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(122.0))
    )
    axes = axes2.ravel()

    # a: predicted against measured, every measured triple, by split.
    ax = axes[0]
    for s in SPLITS:
        sub = df[df["split"] == s]
        r = agree.loc[agree["subset"] == f"all | {s}", "pearson_r"]
        lab = f"{s} ({len(sub):,}" + (f", r = {r.iloc[0]:+.2f})" if len(r) else ")")
        ax.scatter(sub["measured_tau"], sub["pred_mean"], s=2.0,
                   color=SPLIT_COLOR[s], linewidths=0, alpha=0.55, zorder=3, label=lab)
    lim = np.array([min(df["measured_tau"].min(), df["pred_mean"].min()) - 0.05,
                    max(df["measured_tau"].max(), df["pred_mean"].max()) + 0.05])
    ax.plot(lim, lim, color="black", linewidth=0.5, linestyle="--", zorder=4)
    ax.axhline(0, color="0.75", linewidth=0.4, zorder=1)
    ax.axvline(0, color="0.75", linewidth=0.4, zorder=1)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("Measured $\\tau$ in the 010 build")
    ax.set_ylabel("Predicted $\\tau$, ensemble mean")
    ax.set_title(f"All {len(df):,} measured triples inside the space\n"
                 f"dashed line is equality", fontsize=6, loc="left", pad=3)
    ax.legend(loc="upper left", frameon=True, fontsize=5, handlelength=1.0,
              labelspacing=0.25, borderpad=0.3, markerscale=3)

    # b: does agreement hold as the ranking is zoomed into its tail?
    ax = axes[1]
    wins = [w for w in agree["window"].unique() if w != "all"] + ["all"]
    xs = np.arange(len(wins))
    for s, style in (("train", "-o"), ("held out", "--s")):
        ys, ns = [], []
        for w in wins:
            row = agree[(agree["window"] == w) & (agree["split"] == s)]
            ys.append(row["pearson_r"].iloc[0] if len(row) and "pearson_r" in row
                      and pd.notna(row["pearson_r"].iloc[0]) else np.nan)
            ns.append(int(row["n"].iloc[0]) if len(row) else 0)
        col = SPLIT_COLOR["train"] if s == "train" else PLOT_PALETTE[4]
        ax.plot(xs, ys, style[0], color=col, linewidth=0.9, zorder=4, label=s)
        # A correlation on fewer than 10 points is noise. Draw those hollow so the
        # line is not read as a trend through them.
        for x, y, nn in zip(xs, ys, ns):
            if not np.isfinite(y):
                continue
            solid = nn >= MIN_N_FOR_R
            ax.plot([x], [y], style[1], color=col if solid else "white",
                    markersize=3.4, markeredgecolor=col if solid else "0.5",
                    markeredgewidth=0.5, zorder=5)
            ax.text(x, y + 0.035, f"n={nn}", ha="center", fontsize=4.2,
                    color="black" if solid else "0.45")
    ax.axhline(0, color="black", linewidth=0.5, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(wins, fontsize=5)
    ax.set_ylim(-0.05, 1.12)
    ax.set_ylabel("Pearson $r$, predicted against measured")
    ax.set_title(f"Zooming into the tail of the ranking\n"
                 f"hollow marks are $n < {MIN_N_FOR_R}$ and carry no weight",
                 fontsize=6, loc="left", pad=3)
    ax.legend(loc="lower right", frameon=True, fontsize=5, handlelength=1.4,
              labelspacing=0.25, borderpad=0.3)

    # c: calibration. Binned predicted against mean measured, per split.
    ax = axes[2]
    edges = np.quantile(df["pred_mean"], np.linspace(0, 1, 9))
    edges = np.unique(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for s in SPLITS:
        sub = df[df["split"] == s]
        if len(sub) < 10:
            continue
        b = np.digitize(sub["pred_mean"], edges[1:-1])
        m = [sub["measured_tau"].to_numpy()[b == k].mean() if (b == k).sum() else np.nan
             for k in range(len(centers))]
        ax.plot(centers, m, "-o", color=SPLIT_COLOR[s], linewidth=0.9, markersize=2.6,
                markeredgecolor="black", markeredgewidth=0.3, zorder=4, label=s)
    ax.plot(centers, centers, color="black", linewidth=0.5, linestyle="--", zorder=2)
    ax.axhline(POSITIVE_CALL, color="0.5", linewidth=0.5, linestyle=":", zorder=2)
    ax.axhline(0, color="0.75", linewidth=0.4, zorder=1)
    # Scale to the MEASURED range, not to the equality line. Measured tau moves by a
    # few hundredths across the whole predicted range, and on a shared scale with the
    # identity line that movement is invisible, which is itself the finding.
    lo = float(np.nanmin(df["measured_tau"].groupby(
        np.digitize(df["pred_mean"], edges[1:-1])).mean()))
    hi = float(np.nanmax(df["measured_tau"].groupby(
        np.digitize(df["pred_mean"], edges[1:-1])).mean()))
    pad = max(0.02, 0.35 * (hi - lo))
    ax.set_ylim(min(lo - pad, -0.02), max(hi + pad, POSITIVE_CALL * 1.25))
    ax.set_xlabel("Predicted $\\tau$, octile center")
    ax.set_ylabel("Mean measured $\\tau$")
    slope = agree.loc[agree["subset"] == "all | pooled", "calibration_slope"]
    ax.set_title(f"Calibration inside the space, y scaled to the data\n"
                 f"equality leaves the frame; pooled slope is "
                 f"{slope.iloc[0]:.2f}" if len(slope) else "Calibration inside the space",
                 fontsize=6, loc="left", pad=3)
    ax.legend(loc="upper left", frameon=True, fontsize=5, handlelength=1.4,
              labelspacing=0.25, borderpad=0.3)

    # d: of the measured triples the ranking puts highest, how many are real calls?
    ax = axes[3]
    wins2 = [w for w in wins if w != "all"]
    xs = np.arange(len(wins2))
    width = 0.38
    base_rate = {s: agree.loc[(agree["window"] == "all") & (agree["split"] == s),
                              "precision_measured_positive"].iloc[0]
                 for s in ("train", "held out")
                 if len(agree[(agree["window"] == "all") & (agree["split"] == s)])}
    for j, s in enumerate(("train", "held out")):
        vals, ns, enr = [], [], []
        for w in wins2:
            row = agree[(agree["window"] == w) & (agree["split"] == s)]
            vals.append(row["precision_measured_positive"].iloc[0]
                        if len(row) and "precision_measured_positive" in row
                        and pd.notna(row["precision_measured_positive"].iloc[0])
                        else 0.0)
            ns.append(int(row["n"].iloc[0]) if len(row) else 0)
            enr.append(row["positive_enrichment"].iloc[0]
                       if len(row) and "positive_enrichment" in row
                       and pd.notna(row["positive_enrichment"].iloc[0]) else np.nan)
        col = SPLIT_COLOR["train"] if s == "train" else PLOT_PALETTE[4]
        ax.bar(xs + (j - 0.5) * width, vals, width, color=col, edgecolor="black",
               linewidth=0.4, zorder=3, label=s)
        for x, v, nn, e in zip(xs + (j - 0.5) * width, vals, ns, enr):
            tag = f"{v:.0%}\nn={nn}" + (f"\n{e:.0f}$\\times$" if np.isfinite(e) and e > 0
                                        else "")
            ax.text(x, v + 0.008, tag, ha="center", va="bottom", fontsize=4.2)
        base_v = base_rate.get(s)
        if base_v:
            ax.axhline(base_v, color=col, linewidth=0.6, linestyle=":", zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(wins2, fontsize=5)
    ax.set_ylabel(f"Share with measured $\\tau > {POSITIVE_CALL:+.2f}$")
    ax.set_ylim(0, 0.30)
    ax.set_title("Real positives where they can be checked\n"
                 "dotted lines are each split's own base rate; labels give enrichment",
                 fontsize=6, loc="left", pad=3)
    ax.legend(loc="upper right", frameon=True, fontsize=5, handlelength=1.0,
              labelspacing=0.25, borderpad=0.3)

    for ax in axes:
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.5)
            sp.set_color("black")
        ax.grid(axis="y", which="major", color="0.85", linewidth=0.3, zorder=0)
        ax.set_axisbelow(True)
    for ax, letter in zip(axes, "abcd"):
        _letter(ax, letter)

    fig.suptitle(
        f"The {len(df):,} inference_4 triples the 010 build already measured, labeled by "
        f"split. Train is a sanity check; val and test are the honest read.",
        fontsize=6, y=0.995,
    )
    fig.tight_layout(rect=(0.01, 0, 1, 0.965))
    fig.savefig(f"{out_stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{out_stem}.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
