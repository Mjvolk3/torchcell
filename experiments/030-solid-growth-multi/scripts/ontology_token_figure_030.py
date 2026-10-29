# experiments/030-solid-growth-multi/scripts/ontology_token_figure_030.py
# [[experiments.030-solid-growth-multi.scripts.ontology_token_figure_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/ontology_token_figure_030.py
"""The data panels of the dissertation's "experiment ontology as model input" figure
(Section 3.7), a 2 x 3 sheet on the random split. Panels b (the shared-bias idea) and c (the
annotation, the encoder, the two entry points) are drawn in draw.io by the thesis repository's
``notes/assets/drawio/fig-ch03-ontology-input.gen.py``, which lays this sheet into the figure
with slots b and c left empty here.

a, the screen offset: histograms of the measured trigenic interaction of the held-out
   validation triples of each Kuzmin screen, with the two means. The 2018 screen sits below
   zero and the 2020 screen at zero, so a predictor that knows the screen scores a Pearson
   correlation from the offset alone (the floor of d).
d, validation Pearson of the dataset-name token arm against the ontology-affine arm: the
   trigenic interaction pooled over the validation triples, within the Kuzmin 2018 triples,
   within the Kuzmin 2020 triples, and fitness. Each bar is the mean of the run's last ten
   logged epochs (the name-token arm ends at epoch 183 after two checkpoint continuations, the
   ontology arm at 199). The dashed segment over the pooled group is the floor of a predictor
   that outputs each screen's mean, computed on the same genotypes.
e, the fixed-token swap: the mean predicted tau of each screen's validation triples under
   the record's own token and under the other Kuzmin screen's token, for both arms, with the
   measured per-screen means as black ticks. The ontology annotates the two screens
   identically, so the swap is the identity there.
f, the offset the model carries over training: pooled validation Pearson minus the mean of the
   two within-screen Pearsons, per epoch, for both arms. The name token starts at the floor
   (the token alone names the screen at epoch 0); from epoch 10 the two arms carry the same
   gap, which the genes supply.

Reads the two runs' W&B histories once into ``results/ontology_token_figure_030_history.csv``
(``--refresh`` pulls again) and the fixed-token readouts
``results/fixed_token_readout_030_r_s0_ep183{.json,_rows.parquet}`` (name token, epoch 183)
and ``results/fixed_token_readout_030_r_onto_aff_s0_ep199{.json,_rows.parquet}`` (ontology,
epoch 199). Genotypes are reduced as the readout reduces them (``policy_reduce``); the
histograms of a are over the readout's validation rows.

Writes, true size at the dissertation's text width (164 mm, so the ink-cropped export is 165):
  $ASSET_IMAGES_DIR/030-solid-growth-multi/ontology_token_figure_030.svg  (+ .png)
  $ASSET_IMAGES_DIR/030-solid-growth-multi/ontology_token_figure_030_slots.json  (slots b, c in mm)
  results/ontology_token_figure_030_numbers.json   every number the caption quotes

    PYTHONPATH=$PWD python experiments/030-solid-growth-multi/scripts/ontology_token_figure_030.py [--refresh]
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from torchcell.utils import PLOT_PALETTE, PLOT_PALETTE_FILL, mm_to_in, savefig_true_size_svg  # noqa: E402

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from fixed_token_readout_030 import policy_reduce  # noqa: E402

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS = osp.join(EXPERIMENT_ROOT, "030-solid-growth-multi", "results")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "030-solid-growth-multi")
STEM = osp.join(IMG_DIR, "ontology_token_figure_030")
HISTORY_CSV = osp.join(RESULTS, "ontology_token_figure_030_history.csv")
NUMBERS_JSON = osp.join(RESULTS, "ontology_token_figure_030_numbers.json")

ENTITY = "zhao-group"
PROJECT = "torchcell_030-solid-growth-multi_equivariant_cell_graph_transformer"
# The two arms on the random split, seed 0, IGB mmli (four A100). The name-token run trained
# in three pieces (the 100-epoch job, then two checkpoint continuations), one W&B run each.
# One hue per thing: the dataset name gray, the ontology purple, the 2018 screen orange, the
# 2020 screen blue (red sat too close to orange in the overlapping histograms; the sheets of
# 025 draw overlapping distributions as step outlines, s3_closure_plots.py).
ARMS = {
    "name": {"label": "dataset-name token", "arm": "s3_composite_token", "color": PLOT_PALETTE[5], "fill": PLOT_PALETTE_FILL[5]},
    "ontology": {"label": "ontology token", "arm": "s3_onto_affine", "color": PLOT_PALETTE[2], "fill": PLOT_PALETTE_FILL[2]},
}
SCREEN_COLOR = {"2018": PLOT_PALETTE[0], "2020": PLOT_PALETTE[4]}
READOUT = {
    "name": "fixed_token_readout_030_r_s0_ep183",
    "ontology": "fixed_token_readout_030_r_onto_aff_s0_ep199",
}
KEYS = {
    "pooled": "val/gene_interaction/Pearson",
    "within_2018": "val/token/TmiKuzmin2018Dataset/gene_interaction/Pearson",
    "within_2020": "val/token/TmiKuzmin2020Dataset/gene_interaction/Pearson",
    "fitness": "val/fitness/Pearson",
}
SCREENS = {"TmiKuzmin2018Dataset": "2018", "TmiKuzmin2020Dataset": "2020"}
WINDOW = 10  # epochs averaged at the end of each run
F_XMAX = 40  # panel f shows the epochs in which the offset is learned; the plateau to 200 is in the numbers file

# ------------------------------------------------------------------ sheet geometry, mm
W_MM = 164.0  # the thesis text block is 165 mm; the ink-cropped export adds 1 pt a side
H_MM = 113.0
ROW1_TOP, ROW1_H = 8.5, 40.0
ROW2_TOP, ROW2_H = 64.0, 36.0
A_X, A_W = 10.0, 33.0
SLOT_B = (47.0, 7.0, 48.0, 52.0)  # x, y, w, h of the shared-bias panel, drawn by the thesis generator
C_X, C_W = 113.5, 50.5  # the identity within screens against the merged pool
SLOT_D = (0.7, ROW2_TOP - 1.5, 66.0, 52.0)  # the annotation and encoder panel, drawn by the thesis generator
E_X, E_W = 78.0, 34.0
F_X, F_W = 126.0, 38.0
R025 = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results")  # the closure recompute of c
GRID = "#E3E3E3"

plt.rcParams.update(
    {
        # The readout import leaves savefig.bbox at "tight", which recrops the sheet to its
        # ink and loses the slots; the sheet is a fixed canvas.
        "savefig.bbox": "standard",
        "font.family": "Arial",
        "font.size": 6,
        "axes.labelsize": 6,
        "axes.titlesize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5,
        "svg.fonttype": "none",
        "axes.linewidth": 0.5,
        "legend.edgecolor": "none",
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "legend.facecolor": "white",
        "patch.linewidth": 0.5,
        "hatch.linewidth": 0.6,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    }
)


# ------------------------------------------------------------------ data
def pull_history() -> pd.DataFrame:
    """Every validation row of the two arms' rank-0 runs, one row per (arm, run, epoch)."""
    import wandb

    api = wandb.Api()
    frames = []
    for key, arm in ARMS.items():
        runs = list(
            api.runs(
                f"{ENTITY}/{PROJECT}",
                filters={"config.arm": arm["arm"], "config.split": "R", "config.rank0": True},
            )
        )
        if not runs:
            raise RuntimeError(f"no rank-0 R-split run labeled {arm['arm']} (run wandb_view_030.py first)")
        for run in runs:
            # history(keys=...) returns every step at which ALL the keys were logged (the
            # end-of-epoch validation rows); scan_history over the 200k training steps came
            # back truncated (120 of 200 epochs, 2026-10-28).
            h = run.history(keys=["epoch", *KEYS.values()], samples=100000, pandas=True)
            rows = pd.DataFrame({"arm": key, "run_id": run.id, "run_name": run.name,
                                 "epoch": h["epoch"].astype(int),
                                 **{k: pd.to_numeric(h[v], errors="coerce") for k, v in KEYS.items()}})
            frames.append(rows)
            print(f"{key}: {run.name} ({run.id}) {len(rows)} validation epochs, {int(rows.epoch.min())}-{int(rows.epoch.max())}")
    hist = pd.concat(frames, ignore_index=True)
    return hist.sort_values(["arm", "epoch", "run_id"]).groupby(["arm", "epoch"], as_index=False).last()


def history(refresh: bool) -> pd.DataFrame:
    if refresh or not osp.exists(HISTORY_CSV):
        hist = pull_history()
        hist.to_csv(HISTORY_CSV, index=False)
        print("wrote", HISTORY_CSV)
    return pd.read_csv(HISTORY_CSV)


def last_window(hist: pd.DataFrame, arm: str) -> tuple[pd.Series, int, int]:
    h = hist[hist.arm == arm].sort_values("epoch")
    tail = h.tail(WINDOW)
    return tail[list(KEYS)].mean(), int(tail.epoch.iloc[0]), int(tail.epoch.iloc[-1])


def readout(arm: str) -> tuple[dict, pd.DataFrame]:
    with open(osp.join(RESULTS, READOUT[arm] + ".json")) as fh:
        summary = json.load(fh)
    rows = pd.read_parquet(osp.join(RESULTS, READOUT[arm] + "_rows.parquet"))
    return summary, rows


def swap_column(arm: str, summary: dict, rows: pd.DataFrame) -> pd.Series:
    """The prediction of every row under the OTHER Kuzmin screen's token (or profile)."""
    if arm == "name":
        col = {"TmiKuzmin2018Dataset": "pred_TmiKuzmin2020Dataset", "TmiKuzmin2020Dataset": "pred_TmiKuzmin2018Dataset"}
    else:
        # Both screens annotate identically: one profile holds all four Kuzmin interaction
        # sources, so the other screen's profile is the record's own.
        profiles = {label: set(names) for label, names in summary["fixed_profiles"].items()}
        label = next(l for l, names in profiles.items() if {"TmiKuzmin2018Dataset", "TmiKuzmin2020Dataset"} <= names)
        col = {"TmiKuzmin2018Dataset": f"pred_{label}", "TmiKuzmin2020Dataset": f"pred_{label}"}
    vocabulary = summary["dataset_vocabulary"]  # the rows' token column indexes it
    out = pd.Series(np.nan, index=rows.index, dtype=float)
    for token, c in col.items():
        m = rows.token == vocabulary.index(token)
        out[m] = rows.loc[m, c].to_numpy(dtype=float)
    assert not out.isna().any(), "a validation row is in neither Kuzmin screen"
    return out


def reduced_read(rows: pd.DataFrame, pred: pd.Series, vocabulary: list[str]) -> dict[str, float]:
    """Pooled Pearson and per-screen mean prediction of ``pred`` after the readout's genotype reduction."""
    df = rows[["genotype", "token", "order", "target"]].copy()
    df["pred_own"] = pred.to_numpy(dtype=float)
    red = policy_reduce(df, vocabulary)
    out = {"pearson": float(np.corrcoef(red.target, red.pred_own)[0, 1])}
    for token, screen in SCREENS.items():
        m = red.screen == f"kuzmin{screen}"
        out[f"mean_pred_{screen}"] = float(red.loc[m, "pred_own"].mean())
        out[f"mean_target_{screen}"] = float(red.loc[m, "target"].mean())
        out[f"n_{screen}"] = int(m.sum())
    return out


def floor_pearson(rows: pd.DataFrame, vocabulary: list[str]) -> float:
    """A predictor that outputs each screen's mean measured tau, scored on the reduced genotypes."""
    red = policy_reduce(rows[["genotype", "token", "order", "target", "pred_own"]].copy(), vocabulary)
    means = red.groupby("token")["target"].transform("mean")
    return float(np.corrcoef(red.target, means)[0, 1])


# ------------------------------------------------------------------ drawing
def axes_mm(fig: Figure, x: float, y: float, w: float, h: float) -> Axes:
    return fig.add_axes((x / W_MM, 1 - (y + h) / H_MM, w / W_MM, h / H_MM))


def box(ax: Axes) -> None:
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.3)
    ax.set_axisbelow(True)


def letter(fig: Figure, ch: str, x_mm: float, y_mm: float) -> None:
    fig.text(x_mm / W_MM, 1 - y_mm / H_MM, ch, fontsize=8, fontweight="bold", ha="left", va="top")


def title(ax: Axes, words: str, y: float = 1.03) -> None:
    ax.text(0.5, y, words, transform=ax.transAxes, fontsize=6, ha="center", va="bottom")


def legend(ax: Axes, handles: list, loc: str, anchor: tuple[float, float]) -> None:
    ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, handlelength=1.0, handletextpad=0.3,
              borderpad=0.2, labelspacing=0.15, frameon=True)


def panel_a(ax: Axes, rows: pd.DataFrame, vocabulary: list[str]) -> dict[str, float]:
    """Histograms of the measured tau of each screen's validation rows."""
    bins = np.arange(-0.3, 0.3001, 0.01)
    out: dict[str, float] = {}
    for token, screen in SCREENS.items():
        t = rows.loc[rows.token == vocabulary.index(token), "target"].to_numpy(dtype=float)
        ax.hist(t, bins=bins, density=True, histtype="step", color=SCREEN_COLOR[screen], linewidth=0.8,
                zorder=3, label=f"Kuzmin {screen}")
        ax.axvline(t.mean(), color=SCREEN_COLOR[screen], lw=0.6, ls=(0, (2, 1.2)), zorder=4)
        out[f"mean_{screen}"] = float(t.mean())
        out[f"sd_{screen}"] = float(t.std(ddof=1))
        out[f"n_rows_{screen}"] = int(len(t))
        out[f"frac_positive_{screen}"] = float((t > 0).mean())
    ax.set_xlim(-0.3, 0.3)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlabel(r"measured $\tau_{ijk}$")
    ax.set_ylabel("density")
    title(ax, "Kuzmin 2018 is all negative interactions")
    handles = [Line2D([0], [0], color=SCREEN_COLOR[s], lw=0.8, label=f"Kuzmin {s}") for s in ("2018", "2020")]
    handles.append(Line2D([0], [0], color="black", lw=0.6, ls=(0, (2, 1.2)), label="screen mean"))
    legend(ax, handles, "upper right", (1.02, 1.0))
    box(ax)
    ax.grid(False)
    return out


def panel_c(ax: Axes) -> dict[str, float]:
    """The interaction identity, stored score against the score recomputed from fitness: every
    record kept and read within its own screen (purple, one bar per screen) against the records
    merged into one fitness per genotype across sources (gray, the 025 training pool). From the
    025 closure recompute (s3_closure_recompute_summary.json, s3_closure_trigenic_within_screen.json)."""
    with open(osp.join(R025, "s3_closure_recompute_summary.json")) as fh:
        rec = json.load(fh)
    with open(osp.join(R025, "s3_closure_trigenic_within_screen.json")) as fh:
        tri = json.load(fh)
    bars = [  # (x, label, value, kept-within-screen?)
        (0.0, "Costanzo\n2016", rec["within_screen_control"]["costanzo2016"]["pearson"], True),
        (1.0, "Kuzmin\n2018", rec["within_screen_control"]["kuzmin2018"]["pearson"], True),
        (2.0, "Kuzmin\n2020", rec["within_screen_control"]["kuzmin2020"]["pearson"], True),
        (3.0, "merged", rec["digenic_strength"]["all"]["pearson"], False),
        (4.6, "Kuzmin\n2018", tri["kuzmin2018"]["tau_within_screen"]["pearson"], True),
        (5.6, "Kuzmin\n2020", tri["kuzmin2020"]["tau_within_screen"]["pearson"], True),
        (6.6, "merged", rec["trigenic_strength"]["all"]["pearson"], False),
    ]
    out: dict[str, float] = {}
    for x, label, v, kept in bars:
        ax.bar(x, v, width=0.8, color=ARMS["ontology"]["color"] if kept else ARMS["name"]["color"],
               edgecolor="black", linewidth=0.4, zorder=3)
        ax.text(x, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=5)
        out[f"{'digenic' if x < 4 else 'trigenic'}/{label.replace(chr(10), ' ')}"] = float(v)
    ax.set_xticks([b[0] for b in bars])
    ax.set_xticklabels([b[1] for b in bars], fontsize=5)
    ax.text(1.5, 1.16, "digenic", ha="center", va="bottom", fontsize=6)
    ax.text(5.6, 1.16, "trigenic", ha="center", va="bottom", fontsize=6)
    ax.axvline(3.8, color=GRID, lw=0.5)
    ax.set_xlim(-0.6, 7.2)
    ax.set_ylim(0, 1.5)  # the top third holds the key, above the bars
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.set_ylabel("Pearson, stored vs recomputed")
    title(ax, "kept per screen the identity holds; merged it breaks")
    handles = [Patch(facecolor=ARMS["ontology"]["color"], edgecolor="black", linewidth=0.4, label="every record kept, read within its screen"),
               Patch(facecolor=ARMS["name"]["color"], edgecolor="black", linewidth=0.4, label="one fitness per genotype across sources")]
    legend(ax, handles, "upper center", (0.5, 1.0))
    box(ax)
    return out


def panel_e(ax: Axes, windows: dict[str, pd.Series], floor: float) -> None:
    groups = [("pooled", "pooled"), ("within_2018", "within\n2018"), ("within_2020", "within\n2020"), ("fitness", "fitness")]
    w = 0.36
    for g, (key, _) in enumerate(groups):
        for k, arm in enumerate(ARMS):
            x = g + (k - 0.5) * w
            ax.bar(x, float(windows[arm][key]), width=w * 0.92, color=ARMS[arm]["color"], edgecolor="black", linewidth=0.4, zorder=3)
    ax.plot([-0.5 * w - 0.2, 0.5 * w + 0.2], [floor, floor], color="black", lw=0.6, ls=(0, (2, 1.2)), zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[1] for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.set_ylabel("validation Pearson")
    title(ax, "both arms score the triples alike")
    handles = [Patch(facecolor=ARMS[a]["color"], edgecolor="black", linewidth=0.4, label=ARMS[a]["label"]) for a in ARMS]
    handles.append(Line2D([0], [0], color="black", lw=0.6, ls=(0, (2, 1.2)), label=f"screen-mean floor, {floor:.2f}"))
    legend(ax, handles, "upper left", (-0.02, 1.0))
    box(ax)


def panel_f(ax: Axes, swap: dict[str, dict[str, dict[str, float]]]) -> None:
    """swap[arm][read] -> {mean_pred_2018, mean_pred_2020, mean_target_2018, mean_target_2020}, read in (own, other)."""
    w = 0.3
    xs: list[float] = []
    labels: list[str] = []
    for a, arm in enumerate(ARMS):
        for s, screen in enumerate(("2018", "2020")):
            g = a * 2.4 + s
            xs.append(g)
            labels.append(f"{screen}\ntriples")
            for k, read in enumerate(("own", "other")):
                x = g + (k - 0.5) * w
                v = swap[arm][read][f"mean_pred_{screen}"]
                # the swapped read on the arm's light fill, so the black hatch stays visible
                ax.bar(x, v, width=w * 0.92, color=ARMS[arm]["fill" if read == "other" else "color"],
                       edgecolor="black", linewidth=0.4, hatch="///" if read == "other" else None, zorder=3)
            t = swap[arm]["own"][f"mean_target_{screen}"]
            ax.plot([g - w - 0.05, g + w + 0.05], [t, t], color="black", lw=0.8, zorder=5)
        ax.text(a * 2.4 + 0.5, -0.19, ARMS[arm]["label"], ha="center", va="top", fontsize=6,
                transform=ax.get_xaxis_transform(),
                bbox=dict(boxstyle="round,pad=0.3", facecolor=ARMS[arm]["fill"], edgecolor=ARMS[arm]["color"], linewidth=0.5))
    ax.axhline(0, color="black", lw=0.4, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.7, 2.4 + 1.7)
    ax.set_ylim(-0.07, 0.03)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.set_ylabel(r"mean predicted $\tau_{ijk}$")
    title(ax, "the name token carries the offset")
    handles = [
        Patch(facecolor="white", edgecolor="black", linewidth=0.4, label="own token"),
        Patch(facecolor="white", edgecolor="black", linewidth=0.4, hatch="///", label="other screen's token"),
        Line2D([0], [0], color="black", lw=0.8, label="measured mean"),
    ]
    legend(ax, handles, "upper right", (1.02, 1.0))
    box(ax)
    ax.grid(False)


def panel_epochs(ax: Axes, hist: pd.DataFrame, floor: float) -> dict[str, float]:
    """Pooled (solid) and mean within-screen (dashed) validation Pearson per epoch, both arms,
    over the first F_XMAX epochs, with the screen-mean floor."""
    out: dict[str, float] = {}
    for arm in ARMS:
        h = hist[hist.arm == arm].sort_values("epoch")
        within = 0.5 * (h.within_2018 + h.within_2020)
        ax.plot(h.epoch, h.pooled, color=ARMS[arm]["color"], lw=0.8, zorder=3)
        ax.plot(h.epoch, within, color=ARMS[arm]["color"], lw=0.8, ls=(0, (2.5, 1.5)), zorder=3)
        out[f"pooled_epoch0_{arm}"] = float(h.pooled.iloc[0])
        out[f"within_epoch0_{arm}"] = float(within.iloc[0])
        out[f"pooled_epoch10_{arm}"] = float(h.pooled[h.epoch == 10].iloc[0])
        out[f"within_epoch10_{arm}"] = float(within[h.epoch == 10].iloc[0])
    ax.axhline(floor, color="black", lw=0.6, ls=(0, (2, 1.2)), zorder=2)
    ax.set_xlim(0, F_XMAX)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_ylim(0, 0.6)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation Pearson")
    title(ax, "the name token starts at the floor")
    # the arm colors are keyed in d; this key names the line styles only
    handles = [Line2D([0], [0], color="black", lw=0.8, label="pooled"),
                Line2D([0], [0], color="black", lw=0.8, ls=(0, (2.5, 1.5)), label="mean within screen"),
                Line2D([0], [0], color="black", lw=0.6, ls=(0, (2, 1.2)), label=f"screen-mean floor, {floor:.2f}")]
    legend(ax, handles, "lower right", (1.02, -0.02))
    box(ax)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true", help="pull the W&B histories again")
    args = ap.parse_args()
    os.makedirs(IMG_DIR, exist_ok=True)

    hist = history(args.refresh)
    windows: dict[str, pd.Series] = {}
    numbers: dict = {"panel_a": {}, "panel_c": {}, "panel_e": {}, "panel_f": {}, "window_epochs": WINDOW}
    for arm in ARMS:
        win, e0, e1 = last_window(hist, arm)
        windows[arm] = win
        numbers["panel_e"][arm] = {"epochs": [e0, e1], **{k: round(float(win[k]), 4) for k in KEYS}}
        print(f"e {arm:9s} epochs {e0}-{e1}: " + " ".join(f"{k} {win[k]:.3f}" for k in KEYS))

    swap: dict[str, dict[str, dict[str, float]]] = {}
    floor = None
    name_rows = None
    name_vocab: list[str] = []
    for arm in ARMS:
        summary, rows = readout(arm)
        vocab = summary["dataset_vocabulary"]
        own = reduced_read(rows, rows.pred_own, vocab)
        other = reduced_read(rows, swap_column(arm, summary, rows), vocab)
        swap[arm] = {"own": own, "other": other}
        if floor is None:
            floor = floor_pearson(rows, vocab)
            name_rows, name_vocab = rows, vocab
        numbers["panel_f"][arm] = {
            "checkpoint_epoch": summary["checkpoint_epoch"],
            "own": {k: round(v, 4) for k, v in own.items()},
            "other_screen_token": {k: round(v, 4) for k, v in other.items()},
        }
        print(f"f {arm:9s} ep {summary['checkpoint_epoch']}: own r {own['pearson']:.3f} "
              f"means {own['mean_pred_2018']:+.4f}/{own['mean_pred_2020']:+.4f}; "
              f"other r {other['pearson']:.3f} means {other['mean_pred_2018']:+.4f}/{other['mean_pred_2020']:+.4f}")
    assert floor is not None and name_rows is not None
    numbers["screen_mean_floor_pearson"] = round(floor, 4)
    numbers["measured_means"] = {s: round(swap["name"]["own"][f"mean_target_{s}"], 4) for s in ("2018", "2020")}
    print(f"screen-mean floor r {floor:.3f}")

    fig = plt.figure(figsize=(mm_to_in(W_MM), mm_to_in(H_MM)))
    ax_a = axes_mm(fig, A_X, ROW1_TOP, A_W, ROW1_H)
    ax_c = axes_mm(fig, C_X, ROW1_TOP, C_W, ROW1_H)
    ax_e = axes_mm(fig, E_X, ROW2_TOP, E_W, ROW2_H)
    ax_f = axes_mm(fig, F_X, ROW2_TOP, F_W, ROW2_H)
    numbers["panel_a"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in panel_a(ax_a, name_rows, name_vocab).items()}
    numbers["panel_c"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in panel_c(ax_c).items()}
    panel_e(ax_e, windows, floor)
    panel_f(ax_f, swap)
    print("a " + " ".join(f"{k} {v}" for k, v in numbers["panel_a"].items()))
    letter(fig, "a", 0.7, 1.2)
    letter(fig, "c", C_X - 16.0, 1.2)
    letter(fig, "e", E_X - 9.5, ROW2_TOP - 7.3)
    letter(fig, "f", F_X - 9.5, ROW2_TOP - 7.3)
    savefig_true_size_svg(fig, STEM + ".svg")
    fig.savefig(STEM + ".png", dpi=300)
    with open(STEM + "_slots.json", "w") as fh:
        json.dump({"page_width_mm": W_MM, "page_height_mm": H_MM,
                   "b": dict(zip(("x_mm", "y_mm", "width_mm", "height_mm"), SLOT_B)),
                   "d": dict(zip(("x_mm", "y_mm", "width_mm", "height_mm"), SLOT_D)),
                   "row2_axes_bottom_mm": ROW2_TOP + ROW2_H, "row1_axes_bottom_mm": ROW1_TOP + ROW1_H}, fh, indent=2)
    # The epoch curves (pooled and within-screen Pearson over the first forty epochs) left the
    # figure on the author's review of 2026-10-29; they stay as a standalone sheet for the
    # script's note, with their numbers in the numbers file.
    fig2 = plt.figure(figsize=(mm_to_in(60.0), mm_to_in(48.0)))
    ax2 = fig2.add_axes((12.0 / 60.0, 8.0 / 48.0, 45.0 / 60.0, 32.0 / 48.0))
    numbers["epoch_curves"] = {k: round(v, 4) for k, v in panel_epochs(ax2, hist, floor).items()}
    savefig_true_size_svg(fig2, STEM + "_epochs.svg")
    fig2.savefig(STEM + "_epochs.png", dpi=300)
    print("epochs " + " ".join(f"{k} {v:.3f}" for k, v in numbers["epoch_curves"].items()))
    with open(NUMBERS_JSON, "w") as fh:
        json.dump(numbers, fh, indent=2)
    print("wrote", STEM + ".svg", STEM + ".png", NUMBERS_JSON, sep="\n  ")


if __name__ == "__main__":
    main()
