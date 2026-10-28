# experiments/006-kuzmin-tmi/scripts/dango_ablation_si.py
# [[experiments.006-kuzmin-tmi.scripts.dango_ablation_si]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/006-kuzmin-tmi/scripts/dango_ablation_si
"""DANGO ablation arms: frozen curves, per-arm table, three SI panels.

Reads every run of ``zhao-group/torchcell_006-kuzmin2018-tmi_dango_ablation``, the W&B
project that ``dango_ablation.py`` (this folder) trains into: one run per arm of a 2 x 2
ablation of the STRING v9.1 DANGO configuration on the experiment-006 splits, 150 epochs,
batch 64 on one GPU, seed 42. The arms (``conf/dango_ablation_*.yaml``):

* ``baseline``: the configuration of the five v9.1 runs of ``dango_full_dataset_si.py``
  (LinearUntilUniform, transition epoch 10: the reconstruction loss weight alpha goes from 1
  to 0.5 over epochs 0 to 10; ReZero gates beta_l initialized at 0.01);
* ``no_pretrain``: ``PreThenPost(transition_epoch=0)``, alpha = 0 from the first step, so the
  model trains on the interaction loss alone; the reconstruction loss is still computed and
  logged, unweighted, but sends no gradient. This removes an auxiliary loss trained jointly
  with the interaction loss, not the DANGO paper's staged pretraining;
* ``no_rezero``: ``model.residual: plain``, x + O(attn(x)) with no beta parameter;
* ``no_pretrain_no_rezero``: both. Absent from the outputs until it has a run.

Scoring rule: best validation Pearson r is the MAXIMUM over epochs of
``val/gene_interaction/Pearson``, an upward-biased order statistic. The trainer keeps the
minimum-validation-MSE checkpoint and scores it once on test
(``best_mse_ckpt/test/gene_interaction/Pearson`` in the run summary).

For scale, panels a and b set the arms beside the five STRING v9.1 runs of
``experiments/010-kuzmin-tmi/scripts/dango_full_dataset_si.py``, read from that script's
frozen ``dango_full_dataset_{runs,history}.csv`` (not re-pulled). Those runs used batch 64
per GPU on two or four GPUs, so 2 to 4 times fewer optimizer steps per epoch than the arms;
their spread is a scale for seed-to-seed variation, not a matched control. Their maxima all
fall at epochs 98 to 140, so a 150-epoch cut leaves their best r unchanged.

Outputs
-------
results : experiments/006-kuzmin-tmi/results/dango_ablation_curves.csv  (per arm x epoch)
          experiments/006-kuzmin-tmi/results/dango_ablation_arms.csv    (one row per arm)
panels  : $ASSET_IMAGES_DIR/006-kuzmin-tmi/dango_ablation_{curves,scores,reconstruction}.{svg,png}
table   : paper/nature-biotech/sections/tab-dango-ablation-arms.tex

Run from the repo root (``--from-csv`` re-renders offline from the frozen CSVs):
    python experiments/006-kuzmin-tmi/scripts/dango_ablation_si.py [--from-csv]
"""

import argparse
import os
import os.path as osp
import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from matplotlib.ticker import (
    FixedLocator,
    FormatStrFormatter,
    LogLocator,
    MultipleLocator,
    NullFormatter,
)

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
        "hatch.linewidth": 0.4,
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.01,
    }
)

load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

PROJECT = "zhao-group/torchcell_006-kuzmin2018-tmi_dango_ablation"
ARMS = ["baseline", "no_pretrain", "no_rezero", "no_pretrain_no_rezero"]
ARM_LABEL = {
    "baseline": "Baseline",
    "no_pretrain": "No reconstruction loss",
    "no_rezero": "No ReZero gate",
    "no_pretrain_no_rezero": "Neither",
}
# Tick labels of panel b, broken to fit a third-width panel.
ARM_TICK = {
    "baseline": "Baseline",
    "no_pretrain": "No\nreconstruction\nloss",
    "no_rezero": "No ReZero\ngate",
    "no_pretrain_no_rezero": "Neither",
}
# The table's definition column: what each arm changes, in words.
ARM_DEFINITION = {
    "baseline": (
        "None: reconstruction loss at weight $\\alpha_e$, 1 to 0.5 over epochs 0 to 10, "
        "then 0.5; ReZero gates $\\beta_l$ from 0.01"
    ),
    "no_pretrain": (
        "Interaction loss only, $\\alpha_e=0$ from the first step; "
        "$\\mathcal{L}_{\\mathrm{rec}}$ logged, not trained"
    ),
    "no_rezero": "Plain residual $x+O_l(\\mathrm{attn}(x))$ in both layers, no $\\beta_l$",
    "no_pretrain_no_rezero": "Both changes",
}
ARM_COLOR = {a: PLOT_PALETTE[i] for i, a in enumerate(ARMS)}
ARM_FILL = {a: PLOT_PALETTE_FILL[i] for i, a in enumerate(ARMS)}
REF_COLOR = PLOT_PALETTE[5]  # gray: the five earlier v9.1 runs, for scale

VAL_R = "val/gene_interaction/Pearson"
VAL_MSE = "val/gene_interaction/MSE"
VAL_REC = "val/reconstruction_loss"
VAL_INT = "val/interaction_loss"
VAL_ALPHA = "val/alpha"
TRAIN_R = "train/gene_interaction/Pearson"
BETA1, BETA2 = "model/beta1", "model/beta2"
EPOCH_WALL = "perf/epoch_wall_s"
TEST_R = "best_mse_ckpt/test/gene_interaction/Pearson"
TEST_MSE = "best_mse_ckpt/test/gene_interaction/MSE"
KEYS = ["epoch", VAL_R, VAL_MSE, VAL_REC, VAL_INT, VAL_ALPHA, TRAIN_R, BETA1, BETA2, EPOCH_WALL, "_runtime", "_step"]
JOB_RE = re.compile(r"^run_[^_]+?-(\d+)_[0-9a-f]{64}$")  # run_<host>-<slurm job>_<config sha256>

LEVEL = 0.35  # the fixed validation Pearson level of dango_full_dataset_si.py
LEVEL2 = 0.36  # a second level, above which the arms separate (time to the plateau)
CHECK_EPOCHS = (50, 100)  # validation r is also recorded at these epochs
MAX_EPOCHS = 150

RESULTS_DIR = "experiments/006-kuzmin-tmi/results"
CURVES_CSV = osp.join(RESULTS_DIR, "dango_ablation_curves.csv")
ARMS_CSV = osp.join(RESULTS_DIR, "dango_ablation_arms.csv")
REF_RUNS_CSV = "experiments/010-kuzmin-tmi/results/dango_full_dataset_runs.csv"
REF_HIST_CSV = "experiments/010-kuzmin-tmi/results/dango_full_dataset_history.csv"
REF_GROUP = "9_1"
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "006-kuzmin-tmi")
TEX_PATH = "paper/nature-biotech/sections/tab-dango-ablation-arms.tex"
PANEL_H_MM = 52.0


def epoch_frame(hist: pd.DataFrame, key: str, cols: list[str]) -> pd.DataFrame:
    """Rows that carry ``key``, one per epoch (the history interleaves step and epoch rows)."""
    sub = hist.dropna(subset=[key]).drop_duplicates("epoch", keep="last")
    return sub[["epoch"] + cols].astype({"epoch": int})


def pull() -> pd.DataFrame:
    import wandb

    api = wandb.Api(timeout=300)
    runs = list(api.runs(PROJECT, per_page=100))
    arms = [r.config["arm"] for r in runs]
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise ValueError(f"runs with arms outside {ARMS}: {sorted(unknown)}")
    repeated = [a for a, n in Counter(arms).items() if n != 1]
    if repeated:
        raise ValueError(f"arms with more than one run: {repeated}")

    frames = []
    for run in runs:
        if run.state != "finished":
            raise ValueError(f"{run.id} ({run.config['arm']}) is {run.state}, not finished")
        job = JOB_RE.match(run.name)
        if job is None:
            raise ValueError(f"run name {run.name} does not carry a slurm job id")
        hist = pd.DataFrame(list(run.scan_history(keys=KEYS, page_size=10000)))
        for k in KEYS:
            if k not in hist:
                hist[k] = np.nan  # the plain-residual arm logs no beta
        val = epoch_frame(hist, VAL_R, [VAL_R, VAL_MSE, VAL_REC, VAL_INT, VAL_ALPHA, "_runtime"])
        end = epoch_frame(hist, EPOCH_WALL, [BETA1, BETA2, EPOCH_WALL])
        train = epoch_frame(hist, TRAIN_R, [TRAIN_R])
        if not (len(val) == len(end) == len(train) == MAX_EPOCHS):
            raise ValueError(f"{run.id}: {len(val)} val, {len(end)} epoch-end, {len(train)} train epochs")
        curve = val.merge(end, on="epoch", validate="1:1").merge(train, on="epoch", validate="1:1")
        curve.insert(0, "arm", run.config["arm"])
        curve.insert(1, "run_id", run.id)
        curve.insert(2, "job_id", int(job.group(1)))
        curve["test_pearson_at_min_mse"] = float(run.summary[TEST_R])
        curve["test_mse_at_min_mse"] = float(run.summary[TEST_MSE])
        frames.append(curve)
        print(f"  {run.id} {run.config['arm']:12s} job {job.group(1)}: {len(curve)} epochs")
    curves = pd.concat(frames, ignore_index=True).rename(
        columns={
            VAL_R: "val_pearson",
            VAL_MSE: "val_mse",
            VAL_REC: "val_reconstruction_loss",
            VAL_INT: "val_interaction_loss",
            VAL_ALPHA: "val_alpha",
            TRAIN_R: "train_pearson",
            BETA1: "beta1",
            BETA2: "beta2",
            EPOCH_WALL: "epoch_wall_s",
            "_runtime": "runtime_s",
        }
    )
    curves["arm"] = pd.Categorical(curves["arm"], ARMS, ordered=True)
    return curves.sort_values(["arm", "epoch"]).reset_index(drop=True)


def summarize(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, c in curves.groupby("arm", observed=True):
        c = c.sort_values("epoch")
        best = c.loc[c["val_pearson"].idxmax()]
        min_mse = c.loc[c["val_mse"].idxmin()]
        hit = c.loc[c["val_pearson"] >= LEVEL, "epoch"]
        hit2 = c.loc[c["val_pearson"] >= LEVEL2, "epoch"]
        if hit.empty or hit2.empty:
            raise ValueError(f"{arm} never reaches validation r >= {LEVEL} or >= {LEVEL2}")
        by_epoch = c.set_index("epoch")
        last = c.iloc[-1]
        rows.append(
            {
                "arm": arm,
                "run_id": c["run_id"].iloc[0],
                "job_id": int(c["job_id"].iloc[0]),
                "epochs_logged": len(c),
                "best_val_pearson": float(best["val_pearson"]),
                "best_epoch": int(best["epoch"]),
                "min_mse_epoch": int(min_mse["epoch"]),
                "val_pearson_at_min_mse": float(min_mse["val_pearson"]),
                "test_pearson_at_min_mse": float(c["test_pearson_at_min_mse"].iloc[0]),
                "test_mse_at_min_mse": float(c["test_mse_at_min_mse"].iloc[0]),
                "epoch_to_level": int(hit.min()),
                "epoch_to_level2": int(hit2.min()),
                "hours_to_level2": float(by_epoch.loc[int(hit2.min()), "runtime_s"]) / 3600.0,
                **{f"val_pearson_epoch_{e}": float(by_epoch.loc[e, "val_pearson"]) for e in CHECK_EPOCHS},
                "beta1_final": float(last["beta1"]) if pd.notna(last["beta1"]) else np.nan,
                "beta2_final": float(last["beta2"]) if pd.notna(last["beta2"]) else np.nan,
                "val_reconstruction_first": float(c["val_reconstruction_loss"].iloc[0]),
                "val_reconstruction_last": float(last["val_reconstruction_loss"]),
                "epoch_wall_s_median": float(c["epoch_wall_s"].median()),
                "wall_h": float(c["runtime_s"].max()) / 3600.0,
            }
        )
    return pd.DataFrame(rows)


def reference_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The five STRING v9.1 runs of dango_full_dataset_si.py: their first MAX_EPOCHS epochs,
    and their best validation r (which all occur inside that window).
    """
    runs = pd.read_csv(REF_RUNS_CSV)
    runs = runs[runs["group"] == REF_GROUP]
    hist = pd.read_csv(REF_HIST_CSV)
    hist = hist[hist["run_id"].isin(runs["run_id"]) & (hist["epoch"] < MAX_EPOCHS)]
    if not (runs["best_epoch"] < MAX_EPOCHS).all():
        raise ValueError("a reference run peaks after the ablation window; its full best r is not comparable")
    windowed = hist.groupby("run_id")["val_pearson"].max()
    if not np.allclose(windowed.loc[runs["run_id"]].to_numpy(), runs["best_val_pearson"].to_numpy()):
        raise ValueError("reference best r inside the window differs from the full-run best r")
    return runs, hist


def reference_summary(ref_runs: pd.DataFrame, ref_hist: pd.DataFrame, batch: int) -> dict:
    """Spread of the reference runs by the same rules, and their optimizer-step ratio: a run
    on g GPUs at batch b per GPU takes g * b / ``batch`` times fewer steps per epoch.
    """
    level = ref_hist[ref_hist["val_pearson"] >= LEVEL].groupby("run_id")["epoch"].min()
    level2 = ref_hist[ref_hist["val_pearson"] >= LEVEL2].groupby("run_id")["epoch"].min()
    if set(level.index) != set(ref_runs["run_id"]) or set(level2.index) != set(ref_runs["run_id"]):
        raise ValueError("a reference run never reaches a level inside the window")
    ratio = ref_runs["n_ranks"] * ref_runs["batch_size"] / batch
    return {
        "n": NUMBER_WORD[len(ref_runs)],
        "best_lo": float(ref_runs["best_val_pearson"].min()),
        "best_hi": float(ref_runs["best_val_pearson"].max()),
        "level_lo": int(level.min()),
        "level_hi": int(level.max()),
        "level2_lo": int(level2.min()),
        "level2_hi": int(level2.max()),
        "steps_lo": NUMBER_WORD[int(ratio.min())],
        "steps_hi": NUMBER_WORD[int(ratio.max())],
    }


NUMBER_WORD = {2: "two", 3: "three", 4: "four", 5: "five"}
ARM_BATCH = 64  # data_module.batch_size of every arm, on one GPU


def style_axes(ax):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)
        sp.set_color("black")
    ax.tick_params(length=2, width=0.5)


def framed_legend(ax, **kw):
    """Legend on an opaque white box with no edge (figure-workflow rule, 2026-10-27)."""
    opts = {"handlelength": 1.2, "handletextpad": 0.4, "labelspacing": 0.25,
            "borderaxespad": 0.3, "borderpad": 0.3} | kw
    return ax.legend(frameon=True, framealpha=1, edgecolor="none", facecolor="white",
                     fancybox=False, **opts)


def new_panel():
    w = PANEL_WIDTHS_MM["third"]
    fig, ax = plt.subplots(figsize=(mm_to_in(w), mm_to_in(PANEL_H_MM)))
    return fig, ax


def panel_curves(curves: pd.DataFrame, arms: list[str], ref_hist: pd.DataFrame):
    """a: validation Pearson r against epoch, one line per arm over the five earlier v9.1 runs
    in thin gray. Linear epoch axis 0 to 150; y zoomed to the rise and the plateau, with the
    0.35 level dashed.
    """
    fig, ax = new_panel()
    fig.subplots_adjust(left=0.2, right=0.96, top=0.97, bottom=0.17)
    for i, (rid, hh) in enumerate(ref_hist.groupby("run_id")):
        hh = hh.sort_values("epoch")
        ax.plot(hh["epoch"], hh["val_pearson"], color=REF_COLOR, linewidth=0.35, alpha=0.6,
                zorder=1, label="Five earlier v9.1 runs" if i == 0 else None)
    for arm in arms:
        c = curves[curves["arm"] == arm].sort_values("epoch")
        ax.plot(c["epoch"], c["val_pearson"], color=ARM_COLOR[arm], linewidth=0.8, zorder=3,
                label=ARM_LABEL[arm])
    ax.axhline(LEVEL, color="black", linewidth=0.5, linestyle=(0, (3, 2)), zorder=2)
    ax.set_xlim(0, MAX_EPOCHS)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Pearson r")
    ax.set_ylim(0.2, 0.4)
    ax.yaxis.set_major_locator(FixedLocator([0.2, 0.3, 0.4]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_minor_locator(FixedLocator([0.25, 0.35]))
    ax.tick_params(which="minor", length=1.2, width=0.5)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.grid(axis="y", which="major", color="#CACACA", linewidth=0.4)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(ARM_LABEL[a]) for a in arms] + [labels.index("Five earlier v9.1 runs")]
    leg = framed_legend(ax, handles=[handles[k] for k in order], labels=[labels[k] for k in order],
                        loc="lower right")
    for line in leg.get_lines():
        line.set_linewidth(1.0)
        line.set_alpha(1.0)
    style_axes(ax)
    save(fig, "dango_ablation_curves")


def panel_scores(arms_df: pd.DataFrame, arms: list[str], ref_runs: pd.DataFrame):
    """b: per arm, best validation r (dark bar, line color) beside test r at the minimum-
    validation-MSE checkpoint (light fill, black hatch); gray band = range of best validation r
    over the five earlier v9.1 runs. One run per arm, so no whisker.
    """
    fig, ax = new_panel()
    fig.subplots_adjust(left=0.2, right=0.96, top=0.97, bottom=0.2)
    lo, hi = ref_runs["best_val_pearson"].min(), ref_runs["best_val_pearson"].max()
    ax.axhspan(lo, hi, color=to_rgba(REF_COLOR, 0.25), zorder=0, linewidth=0)
    ax.axhline(lo, color=REF_COLOR, linewidth=0.4, zorder=0.5)
    ax.axhline(hi, color=REF_COLOR, linewidth=0.4, zorder=0.5)
    bw = 0.34
    for i, arm in enumerate(arms):
        r = arms_df[arms_df["arm"] == arm].iloc[0]
        ax.bar(i - bw / 2, r["best_val_pearson"], bw, color=ARM_COLOR[arm], edgecolor="black",
               linewidth=0.4, zorder=2)
        ax.bar(i + bw / 2, r["test_pearson_at_min_mse"], bw, color=ARM_FILL[arm], edgecolor="black",
               linewidth=0.4, hatch="//////", zorder=2)
    ax.set_xticks(np.arange(len(arms)))
    ax.set_xticklabels([ARM_TICK[a] for a in arms])
    ax.set_xlim(-0.55, len(arms) - 0.45)
    ax.set_ylabel("Pearson r")
    ax.set_ylim(0.34, 0.40)
    ax.yaxis.set_major_locator(FixedLocator([0.34, 0.36, 0.38, 0.40]))
    ax.yaxis.set_minor_locator(FixedLocator([0.35, 0.37, 0.39]))
    ax.tick_params(which="minor", length=1.2, width=0.5)
    ax.tick_params(axis="x", which="both", length=0)
    ax.grid(axis="y", which="major", color="#CACACA", linewidth=0.4)
    ax.set_axisbelow(True)
    handles = [
        Patch(facecolor=REF_COLOR, edgecolor="black", linewidth=0.4, label="Best validation"),
        Patch(facecolor=PLOT_PALETTE_FILL[5], edgecolor="black", linewidth=0.4, hatch="//////",
              label="Test at min. val. MSE"),
        Patch(facecolor=to_rgba(REF_COLOR, 0.25), edgecolor=REF_COLOR, linewidth=0.4,
              label="Earlier v9.1 runs, val."),
    ]
    framed_legend(ax, handles=handles, loc="upper left", handlelength=1.0)
    style_axes(ax)
    save(fig, "dango_ablation_scores")


def panel_reconstruction(curves: pd.DataFrame, arms: list[str]):
    """c: validation reconstruction loss (unweighted) against epoch on a log axis. It falls in
    every arm that trains on it and rises in the arm that never does.
    """
    fig, ax = new_panel()
    fig.subplots_adjust(left=0.2, right=0.96, top=0.97, bottom=0.17)
    for arm in arms:
        c = curves[curves["arm"] == arm].sort_values("epoch")
        ax.plot(c["epoch"], c["val_reconstruction_loss"], color=ARM_COLOR[arm], linewidth=0.8,
                label=ARM_LABEL[arm])
    ax.set_yscale("log")
    ax.set_ylim(5e-4, 5e-2)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0, MAX_EPOCHS)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation reconstruction loss (log scale)")
    ax.grid(axis="y", which="major", color="#CACACA", linewidth=0.4)
    ax.set_axisbelow(True)
    leg = framed_legend(ax, loc="upper right")
    for line in leg.get_lines():
        line.set_linewidth(1.0)
    style_axes(ax)
    save(fig, "dango_ablation_reconstruction")


def save(fig, stem: str):
    os.makedirs(IMG_DIR, exist_ok=True)
    svg = osp.join(IMG_DIR, f"{stem}.svg")
    savefig_true_size_svg(fig, svg)
    fig.savefig(osp.join(IMG_DIR, f"{stem}.png"), dpi=300)
    plt.close(fig)
    print(f"  wrote {svg}")


def write_tex(arms_df: pd.DataFrame, ref: dict):
    src = "experiments/006-kuzmin-tmi/scripts/dango_ablation_si.py"

    def head(*lines: str) -> str:
        body = "\\\\ ".join(lines)
        return f"\\textbf{{\\begin{{tabular}}[b]{{@{{}}l@{{}}}}{body}\\end{{tabular}}}}"

    missing = [a for a in ARMS if a not in set(arms_df["arm"])]
    missing_txt = (
        "The arm with both changes has not been run."
        if missing == ["no_pretrain_no_rezero"]
        else ("Arms without a run: " + ", ".join(ARM_LABEL[a] for a in missing) + "." if missing else "")
    )
    lines = [
        f"%% SOURCE: {src} -- AUTO-GENERATED from wandb {PROJECT}; do not hand-edit.",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\caption{The DANGO ablation arms, one run each. Every arm trains the STRING v9.1",
        "configuration of \\supptab{tab:dango-full-runs} (hidden width 64, four heads, AdamW at",
        "learning rate $10^{-5}$) on the experiment-006 splits for 150 epochs at batch 64 on one",
        "GPU, with seed 42 for initialization and shuffling; it changes only what the second",
        f"column states. {missing_txt} Best validation $r$ is the maximum over epochs of",
        "validation Pearson $r$, an upward-biased order statistic, with the epoch it occurs at;",
        "validation and test Pearson $r$ at minimum validation MSE are read at the checkpoint the",
        "trainer keeps, the only one scored on test. Epochs count from 0. $\\beta_1,\\beta_2$ are",
        "the two ReZero gates after the last epoch (initialized at 0.01; -- where the arm has",
        "none). $\\mathcal{L}_{\\mathrm{rec}}$ is the unweighted validation reconstruction loss",
        "(Eq.~\\ref{eq:dango-rec}) after the first and the last epoch. Wall time is the median",
        "over epochs of one training epoch plus its validation, on one NVIDIA RTX 6000 Ada GPU.",
        f"For scale, the {ref['n']} v9.1 runs of \\supptab{{tab:dango-full-runs}} reach best",
        f"validation $r$ {ref['best_lo']:.4f} to {ref['best_hi']:.4f}, $r\\ge{LEVEL:.2f}$ at epochs",
        f"{ref['level_lo']} to {ref['level_hi']} and $r\\ge{LEVEL2:.2f}$ at epochs {ref['level2_lo']} to",
        f"{ref['level2_hi']}, at {ref['steps_lo']} to {ref['steps_hi']} times fewer optimizer steps per epoch.}}",
        "\\label{tab:dango-ablation-arms}",
        # No array package in the class: ragged p cells, rows ended by \\tabularnewline.
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{@{}p{1.65cm} p{3.0cm} l l l l l l r@{}}",
        "\\toprule",
        " & ".join(
            [
                head("Arm, run"),
                head("Change from", "the baseline"),
                head("Best val.", "Pearson $r$", "(epoch)"),
                head("Val. Pearson", "$r$ at min.", "val. MSE", "(epoch)"),
                head("Test Pearson", "$r$ at min.", "val. MSE"),
                head("First epoch,", "val. Pearson", f"$r \\ge {LEVEL:.2f}$; ${LEVEL2:.2f}$"),
                head("$\\beta_1,\\beta_2$", "after", f"epoch {MAX_EPOCHS - 1}"),
                head("Val. $\\mathcal{L}_{\\mathrm{rec}}$,", f"epoch 0 $\\to$ {MAX_EPOCHS - 1}"),
                head("Median", "wall per", "epoch (s)"),
            ]
        )
        + "\\tabularnewline",
        "\\midrule",
    ]
    for _, r in arms_df.iterrows():
        betas = "--" if pd.isna(r["beta1_final"]) else f"{r['beta1_final']:.3f}, {r['beta2_final']:.3f}"
        lines.append(
            f"\\raggedright {ARM_LABEL[r['arm']]}\\newline\\texttt{{{r['run_id']}}} & "
            f"\\raggedright {ARM_DEFINITION[r['arm']]} & "
            f"{r['best_val_pearson']:.4f} ({int(r['best_epoch'])}) & "
            f"{r['val_pearson_at_min_mse']:.4f} ({int(r['min_mse_epoch'])}) & "
            f"{r['test_pearson_at_min_mse']:.4f} & {int(r['epoch_to_level'])}; {int(r['epoch_to_level2'])} & {betas} & "
            f"{r['val_reconstruction_first']:.4f} $\\to$ {r['val_reconstruction_last']:.4f} & "
            f"{r['epoch_wall_s_median']:.0f}\\tabularnewline"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    with open(TEX_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {TEX_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true", help="re-render from the frozen CSVs")
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.from_csv:
        curves = pd.read_csv(CURVES_CSV)
        curves["arm"] = pd.Categorical(curves["arm"], ARMS, ordered=True)
    else:
        curves = pull()
        curves.to_csv(CURVES_CSV, index=False)
        print(f"  wrote {CURVES_CSV} ({len(curves)} rows)")
    arms_df = summarize(curves)
    arms_df.to_csv(ARMS_CSV, index=False)
    print(f"  wrote {ARMS_CSV}")
    print(arms_df.to_string(index=False))
    arms = [a for a in ARMS if a in set(arms_df["arm"])]
    ref_runs, ref_hist = reference_runs()
    ref = reference_summary(ref_runs, ref_hist, ARM_BATCH)
    print(f"  reference v9.1 runs: {ref}")
    panel_curves(curves, arms, ref_hist)
    panel_scores(arms_df, arms, ref_runs)
    panel_reconstruction(curves, arms)
    write_tex(arms_df, ref)


if __name__ == "__main__":
    main()
