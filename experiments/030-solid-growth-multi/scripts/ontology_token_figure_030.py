# experiments/030-solid-growth-multi/scripts/ontology_token_figure_030.py
# [[experiments.030-solid-growth-multi.scripts.ontology_token_figure_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/ontology_token_figure_030.py
"""Panels b and c of the dissertation's "experiment ontology as model input" figure
(Section 3.7), on the random split: the dataset-name token arm against the ontology-affine
arm, performance then bias. Panel a (the annotation, the encoder, the two entry points) is
drawn in draw.io by the thesis repository's ``notes/assets/drawio/fig-ch03-ontology-input.gen.py``,
which lays this sheet into the figure with slot a left empty here.

b, validation Pearson of the two arms: the trigenic interaction pooled over the pinned
   validation triples, within the Kuzmin 2018 triples, within the Kuzmin 2020 triples, and
   fitness. Each bar is the mean of the run's last ten logged epochs (the name-token arm
   ends at epoch 183 after two checkpoint continuations, the ontology arm at 199). The dashed
   segment over the pooled group is the floor a predictor of the two screens' mean tau scores,
   computed on the same genotypes.
c, the fixed-token swap: the mean predicted tau of each screen's validation triples under the
   record's own token and under the other Kuzmin screen's token, for both arms, with the
   measured per-screen means as black ticks. The ontology annotates the two screens
   identically, so the swap is the identity there. The pooled Pearson under the swap is in the
   numbers file and the caption.

Reads the two runs' W&B histories once into ``results/ontology_token_figure_030_history.csv``
(``--refresh`` pulls again) and the fixed-token readouts
``results/fixed_token_readout_030_r_s0_ep183{.json,_rows.parquet}`` (name token, epoch 183)
and ``results/fixed_token_readout_030_r_onto_aff_s0_ep199{.json,_rows.parquet}`` (ontology,
epoch 199). Genotypes are reduced as the readout reduces them (``policy_reduce``).

Writes, true size at the dissertation's 165 mm text width:
  $ASSET_IMAGES_DIR/030-solid-growth-multi/ontology_token_figure_030.svg  (+ .png)
  $ASSET_IMAGES_DIR/030-solid-growth-multi/ontology_token_figure_030_slot_a.json
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

from torchcell.utils import PLOT_PALETTE, mm_to_in, savefig_true_size_svg  # noqa: E402

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
ARMS = {
    "name": {"label": "dataset-name token", "arm": "s3_composite_token", "color": PLOT_PALETTE[5]},
    "ontology": {"label": "ontology token", "arm": "s3_onto_affine", "color": PLOT_PALETTE[2]},
}
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

# ------------------------------------------------------------------ sheet geometry, mm
W_MM = 165.0
H_MM = 60.0
SLOT = (0.7, 7.0, 66.0, 52.0)  # x, y, w, h of panel a, drawn by the thesis generator
ROW_TOP, ROW_H = 8.5, 36.0
B_X, B_W = 80.0, 32.0
C_X, C_W = 127.5, 36.5
GRID = "#E3E3E3"

plt.rcParams.update(
    {
        # The readout import leaves savefig.bbox at "tight", which recrops the sheet to its
        # ink and loses slot a; the sheet is a fixed canvas.
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
        "hatch.linewidth": 0.4,
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


def panel_b(ax: Axes, windows: dict[str, pd.Series], floor: float) -> None:
    groups = [("pooled", "pooled"), ("within_2018", "within\n2018"), ("within_2020", "within\n2020"), ("fitness", "fitness")]
    w = 0.36
    for g, (key, _) in enumerate(groups):
        for k, arm in enumerate(ARMS):
            x = g + (k - 0.5) * w
            v = float(windows[arm][key])
            ax.bar(x, v, width=w * 0.92, color=ARMS[arm]["color"], edgecolor="black", linewidth=0.4, zorder=3)
    ax.plot([-0.5 * w - 0.2, 0.5 * w + 0.2], [floor, floor], color="black", lw=0.6, ls=(0, (2, 1.2)), zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[1] for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.set_ylabel("validation Pearson")
    ax.text(0.5, 1.03, "both arms score the triples alike", transform=ax.transAxes, fontsize=6, ha="center", va="bottom")
    handles = [Patch(facecolor=ARMS[a]["color"], edgecolor="black", linewidth=0.4, label=ARMS[a]["label"]) for a in ARMS]
    handles.append(Line2D([0], [0], color="black", lw=0.6, ls=(0, (2, 1.2)), label=f"screen-mean floor, {floor:.2f}"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, 1.0), handlelength=1.0, handletextpad=0.3, borderpad=0.2, labelspacing=0.15, frameon=True)
    box(ax)


def panel_c(ax: Axes, swap: dict[str, dict[str, dict[str, float]]]) -> None:
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
                ax.bar(x, v, width=w * 0.92, color=ARMS[arm]["color"], edgecolor="black", linewidth=0.4,
                       hatch="////" if read == "other" else None, zorder=3)
            t = swap[arm]["own"][f"mean_target_{screen}"]
            ax.plot([g - w - 0.05, g + w + 0.05], [t, t], color="black", lw=0.8, zorder=5)
        ax.text(a * 2.4 + 0.5, -0.30, ARMS[arm]["label"], ha="center", va="top", fontsize=6,
                transform=ax.get_xaxis_transform())
    ax.axhline(0, color="black", lw=0.4, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.7, 2.4 + 1.7)
    ax.set_ylim(-0.07, 0.03)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.set_ylabel(r"mean predicted $\tau_{ijk}$")
    ax.text(0.5, 1.03, "the name token carries the screen offset", transform=ax.transAxes, fontsize=6, ha="center", va="bottom")
    handles = [
        Patch(facecolor="white", edgecolor="black", linewidth=0.4, label="own token"),
        Patch(facecolor="white", edgecolor="black", linewidth=0.4, hatch="////", label="other screen's token"),
        Line2D([0], [0], color="black", lw=0.8, label="measured mean"),
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.02, 1.0), handlelength=0.9, handletextpad=0.3, borderpad=0.2, labelspacing=0.15, frameon=True)
    box(ax)
    ax.grid(False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true", help="pull the W&B histories again")
    args = ap.parse_args()
    os.makedirs(IMG_DIR, exist_ok=True)

    hist = history(args.refresh)
    windows: dict[str, pd.Series] = {}
    numbers: dict = {"panel_b": {}, "panel_c": {}, "window_epochs": WINDOW}
    for arm in ARMS:
        win, e0, e1 = last_window(hist, arm)
        windows[arm] = win
        numbers["panel_b"][arm] = {"epochs": [e0, e1], **{k: round(float(win[k]), 4) for k in KEYS}}
        print(f"b {arm:9s} epochs {e0}-{e1}: " + " ".join(f"{k} {win[k]:.3f}" for k in KEYS))

    swap: dict[str, dict[str, dict[str, float]]] = {}
    floor = None
    for arm in ARMS:
        summary, rows = readout(arm)
        vocab = summary["dataset_vocabulary"]
        own = reduced_read(rows, rows.pred_own, vocab)
        other = reduced_read(rows, swap_column(arm, summary, rows), vocab)
        swap[arm] = {"own": own, "other": other}
        if floor is None:
            floor = floor_pearson(rows, vocab)
        numbers["panel_c"][arm] = {
            "checkpoint_epoch": summary["checkpoint_epoch"],
            "own": {k: round(v, 4) for k, v in own.items()},
            "other_screen_token": {k: round(v, 4) for k, v in other.items()},
        }
        print(f"c {arm:9s} ep {summary['checkpoint_epoch']}: own r {own['pearson']:.3f} "
              f"means {own['mean_pred_2018']:+.4f}/{own['mean_pred_2020']:+.4f}; "
              f"other r {other['pearson']:.3f} means {other['mean_pred_2018']:+.4f}/{other['mean_pred_2020']:+.4f}")
    assert floor is not None
    numbers["screen_mean_floor_pearson"] = round(floor, 4)
    numbers["measured_means"] = {s: round(swap["name"]["own"][f"mean_target_{s}"], 4) for s in ("2018", "2020")}
    print(f"screen-mean floor r {floor:.3f}")

    fig = plt.figure(figsize=(mm_to_in(W_MM), mm_to_in(H_MM)))
    ax_b = axes_mm(fig, B_X, ROW_TOP, B_W, ROW_H)
    ax_c = axes_mm(fig, C_X, ROW_TOP, C_W, ROW_H)
    panel_b(ax_b, windows, floor)
    panel_c(ax_c, swap)
    letter(fig, "b", B_X - 9.5, 1.2)
    letter(fig, "c", C_X - 9.5, 1.2)
    savefig_true_size_svg(fig, STEM + ".svg")
    fig.savefig(STEM + ".png", dpi=300)
    with open(STEM + "_slot_a.json", "w") as fh:
        json.dump({"x_mm": SLOT[0], "y_mm": SLOT[1], "width_mm": SLOT[2], "height_mm": SLOT[3],
                   "page_width_mm": W_MM, "page_height_mm": H_MM}, fh, indent=2)
    with open(NUMBERS_JSON, "w") as fh:
        json.dump(numbers, fh, indent=2)
    print("wrote", STEM + ".svg", STEM + ".png", NUMBERS_JSON, sep="\n  ")


if __name__ == "__main__":
    main()
