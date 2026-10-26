# experiments/010-kuzmin-tmi/scripts/dango_full_dataset_panels_efg.py
# [[experiments.010-kuzmin-tmi.scripts.dango_full_dataset_panels_efg]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/dango_full_dataset_panels_efg
"""Panels e, f, g of FigS-dango-full-dataset, from the frozen inference outputs.

Inputs are the files ``dango_full_dataset_inference.py`` (this folder) wrote to
``experiments/010-kuzmin-tmi/results/`` after checking its validation metrics against the
logged ones, plus the CGT validation predictions of ``score_010_checkpoints_directly.py``:

- (e) predicted against measured trigenic interaction tau on the experiment-006
  validation split for DANGO run 014mprap (STRING v9.1), minimum-validation-MSE checkpoint
  (epoch 81); hexagonal bins, log count scale, dashed identity line.
- (f) absolute error |prediction - tau| against measured |tau|, fixed-width bins of 0.02
  up to 0.22 (about the 99th percentile of |tau| in both splits), median line and
  interquartile band per bin, DANGO 014mprap on the experiment-006 validation split beside
  CGT run c7671wgj (best-Pearson checkpoint, epoch 24) on the experiment-010 validation
  split. The two splits are DIFFERENT records (different builds); each curve is computed
  on its own split and nothing is joined row by row. The dashed diagonal is the error of
  predicting 0, which is |tau| itself.
- (g) per-gene softmax weight over the six STRING channels of DANGO's meta-embedding,
  STRING v9.1 (014mprap) against v12.0 (9jpfy547), both at the minimum-validation-MSE
  checkpoint, over all 6,607 genes of the index space; box = interquartile range, line =
  median, open circle = mean, whiskers = 5th to 95th percentile, no fliers; dashed line at
  1/6 (uniform).

Outputs
-------
panels  : $ASSET_IMAGES_DIR/010-kuzmin-tmi/dango_full_dataset_{pred_vs_measured,error_vs_tau,channel_weights}.{svg,png}
results : experiments/010-kuzmin-tmi/results/dango_full_dataset_panel_e_summary.csv
          experiments/010-kuzmin-tmi/results/dango_full_dataset_panel_f_bins.csv
          experiments/010-kuzmin-tmi/results/dango_full_dataset_panel_g_summary.csv

Panel sizes: e is third width at the height of the convergence panel (52 mm) so it closes
row 2 of the composed figure; f and g are half width at 50 mm, which keeps the composed
figure under the 170 mm Nature height.

Run from the repo root:
    python experiments/010-kuzmin-tmi/scripts/dango_full_dataset_panels_efg.py
"""

import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator
from scipy.stats import pearsonr

from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    mm_to_in,
    savefig_true_size_svg,
)

# Set AFTER the torchcell imports: torchcell.graph applies the repo mplstyle on import.
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6,
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.5,
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.01,
        # $\tau$ in Arial, not DejaVu Sans (torchcell.utils.PAPER_RC).
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.default": "it",
    }
)

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
DATA_ROOT = os.environ["DATA_ROOT"]

RESULTS_DIR = "experiments/010-kuzmin-tmi/results"
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")
CGT_LABEL_PARQUET = osp.join(
    DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build/processed/label_df.parquet"
)
PANEL_E_CSV = osp.join(RESULTS_DIR, "dango_full_dataset_panel_e_summary.csv")
PANEL_F_CSV = osp.join(RESULTS_DIR, "dango_full_dataset_panel_f_bins.csv")
PANEL_G_CSV = osp.join(RESULTS_DIR, "dango_full_dataset_panel_g_summary.csv")

# Colors keep the meaning they have in panels a to d (dango_full_dataset_si.py GROUP_COLOR):
# DANGO STRING v9.1 = amber, v12.0 = lilac, CGT = wheat.
COLOR_V91 = PLOT_PALETTE[0]
COLOR_V120 = PLOT_PALETTE[2]
COLOR_CGT = PLOT_PALETTE[3]

TAU_LIM = 0.5  # panel e axes span [-TAU_LIM, TAU_LIM] on both axes
BIN_WIDTH = 0.02
BIN_TOP = 0.22  # about the 99th percentile of |tau| (0.210 in the 006 split, 0.218 in the 010 split)
CHANNELS = ["neighborhood", "fusion", "cooccurence", "coexpression", "experimental", "database"]
CHANNEL_LABEL = {
    "neighborhood": "neighborhood",
    "fusion": "fusion",
    "cooccurence": "co-\noccurrence",
    "coexpression": "co-\nexpression",
    "experimental": "experimental",
    "database": "database",
}
RELEASES = [("v9.1", "014mprap", COLOR_V91), ("v12.0", "9jpfy547", COLOR_V120)]


def style_axes(ax):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)
        sp.set_color("black")
    ax.tick_params(length=2, width=0.5)


def save(fig, stem: str):
    os.makedirs(IMG_DIR, exist_ok=True)
    svg = osp.join(IMG_DIR, f"{stem}.svg")
    savefig_true_size_svg(fig, svg)
    fig.savefig(osp.join(IMG_DIR, f"{stem}.png"), dpi=300)
    plt.close(fig)
    print(f"  wrote {svg}")


def load_dango(tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Validation labels and predictions on the experiment-006 split, checked against the
    Pearson the inference script recorded (which itself matched the logged value).
    """
    y = np.load(osp.join(RESULTS_DIR, "dango_labels_val.npy"))
    p = np.load(osp.join(RESULTS_DIR, f"dango_predictions_{tag}_val.npy"))
    if y.shape != p.shape:
        raise ValueError(f"{tag}: {y.shape} labels vs {p.shape} predictions")
    check = pd.read_csv(osp.join(RESULTS_DIR, "dango_full_dataset_inference_check.csv"))
    run, ckpt = tag.split("_")
    rec = check[(check["run"] == run) & (check["ckpt"] == ckpt) & (check["split"] == "val")].iloc[0]
    r = pearsonr(y, p)[0]
    if abs(r - rec["pearson"]) > 1e-9:
        raise ValueError(f"{tag}: Pearson {r} differs from the recorded {rec['pearson']}")
    return y, p


def load_cgt() -> tuple[np.ndarray, np.ndarray]:
    """CGT c7671wgj validation labels and predictions on the experiment-010 split. Labels
    come from the 010 build's label table, rows in sorted record-id order (the order
    score_010_checkpoints_directly.load_records uses for cgt_record_rows_val.npy).
    """
    label_df = pd.read_parquet(CGT_LABEL_PARQUET)
    record_ids = np.sort(label_df["index"].to_numpy())
    y_all = label_df.set_index("index").loc[record_ids, "gene_interaction"].to_numpy()
    rows = np.load(osp.join(RESULTS_DIR, "cgt_record_rows_val.npy"))
    p = np.load(osp.join(RESULTS_DIR, "cgt_predictions_M03_c7671wgj_val.npy"))
    y = y_all[rows]
    scoring = pd.read_csv(osp.join(RESULTS_DIR, "cgt_direct_scoring.csv"))
    rec = scoring[(scoring["model"] == "M03_c7671wgj") & (scoring["split"] == "val")].iloc[0]
    r = pearsonr(y, p)[0]
    if abs(r - rec["pearson"]) > 1e-9:
        raise ValueError(f"CGT: Pearson {r} differs from the recorded {rec['pearson']}")
    return y, p


def panel_pred_vs_measured(y: np.ndarray, p: np.ndarray):
    """(e) Hexbin of predicted against measured tau, DANGO 014mprap min-MSE checkpoint."""
    w, h = PANEL_WIDTHS_MM["third"], 52.0
    fig, ax = plt.subplots(figsize=(mm_to_in(w), mm_to_in(h)))
    fig.subplots_adjust(left=0.2, right=0.97, top=0.97, bottom=0.17)
    cmap = LinearSegmentedColormap.from_list(
        "amber", [PLOT_PALETTE_FILL[0], PLOT_PALETTE[0], PLOT_PALETTE[6]]
    )
    hb = ax.hexbin(
        y, p, gridsize=48, extent=(-TAU_LIM, TAU_LIM, -TAU_LIM, TAU_LIM), mincnt=1,
        cmap=cmap, norm=LogNorm(vmin=1, vmax=1e4), linewidths=0.0, rasterized=False,
    )
    ax.plot([-TAU_LIM, TAU_LIM], [-TAU_LIM, TAU_LIM], color="black", linewidth=0.6,
            linestyle=(0, (3, 2)), label="identity")
    ax.set_xlim(-TAU_LIM, TAU_LIM)
    ax.set_ylim(-TAU_LIM, TAU_LIM)
    ticks = [-0.4, -0.2, 0.0, 0.2, 0.4]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xlabel(r"Measured $\tau$")
    ax.set_ylabel(r"Predicted $\tau$")
    r = pearsonr(y, p)[0]
    n_out = int(((np.abs(y) > TAU_LIM) | (np.abs(p) > TAU_LIM)).sum())
    ax.text(0.96, 0.04, f"Pearson r = {r:.3f}\nn = {len(y):,}",
            transform=ax.transAxes, ha="right", va="bottom", linespacing=1.25)
    ax.text(0.04, 0.63, "DANGO, STRING v9.1", transform=ax.transAxes, ha="left", va="top")
    cax = ax.inset_axes([0.05, 0.86, 0.42, 0.045])
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal")
    cb.set_ticks([1, 10, 100, 1000, 10000])
    cb.set_ticklabels([r"$\mathregular{1}$", r"$\mathregular{10}$", r"$\mathregular{10^{2}}$",
                     r"$\mathregular{10^{3}}$", r"$\mathregular{10^{4}}$"])
    cb.ax.tick_params(length=1.5, width=0.4, pad=1)
    cb.ax.xaxis.set_ticks_position("bottom")
    cb.ax.xaxis.set_label_position("top")
    cb.outline.set_linewidth(0.4)
    cb.set_label("Records per hexagon", labelpad=1.0)
    cb.ax.minorticks_off()
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.76), handlelength=1.6,
              borderaxespad=0.3)
    style_axes(ax)
    save(fig, "dango_full_dataset_pred_vs_measured")
    return {"model": "DANGO 014mprap minmse (STRING v9.1)", "split": "006 validation",
            "n": len(y), "pearson": r, "axis_limit": TAU_LIM, "n_outside_axes": n_out,
            "n_measured_outside": int((np.abs(y) > TAU_LIM).sum()),
            "n_predicted_outside": int((np.abs(p) > TAU_LIM).sum()),
            "pred_min": float(p.min()), "pred_max": float(p.max()),
            "measured_min": float(y.min()), "measured_max": float(y.max()),
            "pred_sd": float(p.std(ddof=1)), "measured_sd": float(y.std(ddof=1))}


def error_bins(y: np.ndarray, p: np.ndarray, model: str, split: str) -> pd.DataFrame:
    """Median and quartiles of |p - y| in fixed-width bins of |y|, plus an all-records row."""
    edges = np.round(np.arange(0.0, BIN_TOP + BIN_WIDTH / 2, BIN_WIDTH), 10)
    a = np.abs(y)
    err = np.abs(p - y)
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel = (a >= lo) & (a < hi)
        e = err[sel]
        rows.append({"model": model, "split": split, "bin": i, "bin_lo": lo, "bin_hi": hi,
                     "n": int(sel.sum()), "frac_of_split": float(sel.mean()),
                     "median": float(np.median(e)), "q25": float(np.quantile(e, 0.25)),
                     "q75": float(np.quantile(e, 0.75)), "mean": float(e.mean()),
                     "median_predict_zero": float(np.median(a[sel]))})
    above = a >= BIN_TOP
    rows.append({"model": model, "split": split, "bin": "above_top", "bin_lo": BIN_TOP, "bin_hi": np.inf,
                 "n": int(above.sum()), "frac_of_split": float(above.mean()),
                 "median": float(np.median(err[above])), "q25": float(np.quantile(err[above], 0.25)),
                 "q75": float(np.quantile(err[above], 0.75)), "mean": float(err[above].mean()),
                 "median_predict_zero": float(np.median(a[above]))})
    rows.append({"model": model, "split": split, "bin": "all", "bin_lo": 0.0, "bin_hi": np.inf,
                 "n": len(y), "frac_of_split": 1.0, "median": float(np.median(err)),
                 "q25": float(np.quantile(err, 0.25)), "q75": float(np.quantile(err, 0.75)),
                 "mean": float(err.mean()), "median_predict_zero": float(np.median(a))})
    df = pd.DataFrame(rows)
    # Median error relative to the error of predicting 0 in the same bin (median |tau|).
    df["median_over_predict_zero"] = df["median"] / df["median_predict_zero"]
    return df


def panel_error_vs_tau(bins: pd.DataFrame):
    """(f) Median |error| (line) and interquartile band against |tau|, two models, each on
    its own validation split; dashed diagonal = predicting 0.
    """
    w, h = PANEL_WIDTHS_MM["half"], 50.0
    fig, ax = plt.subplots(figsize=(mm_to_in(w), mm_to_in(h)))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.18)
    series = [
        ("DANGO", "DANGO (STRING v9.1), 006 validation", COLOR_V91, "o"),
        ("CGT", "CGT, 010 validation", COLOR_CGT, "s"),
    ]
    for model, label, color, marker in series:
        b = bins[(bins["model"] == model) & bins["bin"].apply(lambda v: isinstance(v, (int, np.integer)))]
        x = (b["bin_lo"] + b["bin_hi"]).to_numpy() / 2
        ax.fill_between(x, b["q25"], b["q75"], facecolor=to_rgba(color, 0.28), edgecolor="none")
        ax.plot(x, b["median"], color=color, linewidth=0.9, marker=marker, markersize=2.4,
                markeredgecolor="black", markeredgewidth=0.3, label=label)
    ax.plot([0, BIN_TOP], [0, BIN_TOP], color="black", linewidth=0.6, linestyle=(0, (3, 2)),
            label=r"predicting 0 (error = $|\tau|$)")
    ax.set_xlim(0, BIN_TOP)
    ax.set_ylim(0, BIN_TOP)
    ax.xaxis.set_major_locator(FixedLocator(np.round(np.arange(0, BIN_TOP + 1e-9, 0.04), 2)))
    ax.xaxis.set_minor_locator(FixedLocator(np.round(np.arange(0.02, BIN_TOP, 0.04), 2)))
    ax.yaxis.set_major_locator(FixedLocator(np.round(np.arange(0, BIN_TOP + 1e-9, 0.04), 2)))
    ax.yaxis.set_minor_locator(FixedLocator(np.round(np.arange(0.02, BIN_TOP, 0.04), 2)))
    ax.tick_params(which="minor", length=0)
    ax.grid(which="both", color="#CACACA", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.set_xlabel(r"Measured $|\tau|$ (bins of 0.02)")
    ax.set_ylabel(r"Absolute error of predicted $\tau$")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=to_rgba("#666666", 0.28), edgecolor="none"))
    labels.append("interquartile range")
    ax.legend(handles, labels, frameon=False, loc="upper left", handlelength=1.6,
              borderaxespad=0.3, labelspacing=0.3)
    style_axes(ax)
    save(fig, "dango_full_dataset_error_vs_tau")


def channel_summary() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    weights, rows = {}, []
    for release, run, _ in RELEASES:
        w = pd.read_csv(osp.join(RESULTS_DIR, f"dango_meta_embedding_weights_{run}_minmse.csv"))
        if len(w) != 6607 or not np.allclose(w[CHANNELS].sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"{run}: expected 6,607 genes with weights summing to 1")
        weights[release] = w
        for c in CHANNELS:
            v = w[c].to_numpy()
            rows.append({"release": release, "run": run, "channel": c, "n_genes": len(v),
                         "mean": v.mean(), "median": np.median(v), "q25": np.quantile(v, 0.25),
                         "q75": np.quantile(v, 0.75), "p05": np.quantile(v, 0.05),
                         "p95": np.quantile(v, 0.95), "min": v.min(), "max": v.max(),
                         "frac_above_uniform": float((v > 1 / 6).mean())})
    return pd.DataFrame(rows), weights


def panel_channel_weights(weights: dict[str, pd.DataFrame]):
    """(g) Grouped boxplots of the per-gene channel weight, v9.1 against v12.0."""
    w, h = PANEL_WIDTHS_MM["half"], 50.0
    fig, ax = plt.subplots(figsize=(mm_to_in(w), mm_to_in(h)))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.235)
    x = np.arange(len(CHANNELS))
    offset, width = 0.19, 0.32
    for k, (release, run, color) in enumerate(RELEASES):
        data = [weights[release][c].to_numpy() for c in CHANNELS]
        ax.boxplot(
            data, positions=x + (k - 0.5) * 2 * offset, widths=width, whis=(5, 95),
            showfliers=False, patch_artist=True, showmeans=True,
            meanprops={"marker": "o", "markersize": 2.2, "markerfacecolor": "white",
                       "markeredgecolor": "black", "markeredgewidth": 0.4},
            boxprops={"facecolor": color, "edgecolor": "black", "linewidth": 0.4},
            medianprops={"color": "black", "linewidth": 0.7},
            whiskerprops={"color": "black", "linewidth": 0.4},
            capprops={"color": "black", "linewidth": 0.4},
        )
    ax.axhline(1 / 6, color="black", linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([CHANNEL_LABEL[c] for c in CHANNELS])
    ax.set_xlim(-0.55, len(CHANNELS) - 0.45)
    ax.set_xlabel("STRING channel")
    ax.set_ylabel("Meta-embedding weight per gene")
    ax.set_ylim(0, 0.6)
    ax.yaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6]))
    ax.yaxis.set_minor_locator(FixedLocator([0.1, 0.3, 0.5]))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="both", color="#CACACA", linewidth=0.4)
    ax.set_axisbelow(True)
    handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.4) for _, _, c in RELEASES]
    labels = [f"STRING {rel}" for rel, _, _ in RELEASES]
    handles.append(Line2D([], [], linestyle="none", marker="o", markersize=2.2, markerfacecolor="white",
                          markeredgecolor="black", markeredgewidth=0.4))
    labels.append("mean")
    handles.append(Line2D([], [], color="black", linewidth=0.6, linestyle=(0, (3, 2))))
    labels.append("uniform (1/6)")
    ax.legend(handles, labels, frameon=False, loc="upper left", ncol=4, handlelength=1.4,
              handletextpad=0.4, columnspacing=0.9, borderaxespad=0.3)
    style_axes(ax)
    save(fig, "dango_full_dataset_channel_weights")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    y_d, p_d = load_dango("014mprap_minmse")
    y_c, p_c = load_cgt()
    print(f"DANGO 014mprap minmse, 006 val: n {len(y_d):,}, Pearson {pearsonr(y_d, p_d)[0]:.4f}")
    print(f"CGT c7671wgj, 010 val: n {len(y_c):,}, Pearson {pearsonr(y_c, p_c)[0]:.4f}")

    e = panel_pred_vs_measured(y_d, p_d)
    pd.DataFrame([e]).to_csv(PANEL_E_CSV, index=False)
    print(pd.DataFrame([e]).T.to_string(header=False))

    bins = pd.concat(
        [error_bins(y_d, p_d, "DANGO", "006 validation (014mprap minmse, STRING v9.1)"),
         error_bins(y_c, p_c, "CGT", "010 validation (c7671wgj best-Pearson, epoch 24)")],
        ignore_index=True,
    )
    bins.to_csv(PANEL_F_CSV, index=False)
    print(bins.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    panel_error_vs_tau(bins)

    summary, weights = channel_summary()
    summary.to_csv(PANEL_G_CSV, index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    panel_channel_weights(weights)
    print(f"  wrote {PANEL_E_CSV}, {PANEL_F_CSV}, {PANEL_G_CSV}")


if __name__ == "__main__":
    main()
