# experiments/005-kuzmin2018-tmi/scripts/dango_string_version_sweep.py
# [[experiments.005-kuzmin2018-tmi.scripts.dango_string_version_sweep]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/005-kuzmin2018-tmi/scripts/dango_string_version_sweep
"""DANGO replication in TorchCell: validation Pearson by STRING release and loss schedule.

The DANGO replication (``experiments/005-kuzmin2018-tmi/scripts/dango.py``, wandb project
``zhao-group/torchcell_005-kuzmin2018-tmi_dango``) was trained on the Kuzmin 2018 trigenic
interaction set with the six STRING evidence channels swapped between releases v9.1 (the
DANGO paper's graphs), v11.0, and v12.0, under each of the three pretraining-to-main loss
schedules (``regression_task.loss_scheduler.type``). The results note recorded these runs by
reading values off the wandb charts; this script pulls the run histories through the wandb
API so the figure and table come from the logged data.

Scoring rule: for each run the statistic is the MAXIMUM over epochs of the logged
``val/gene_interaction/Pearson`` (the checkpoint-selection criterion in the trainer). A max
over epochs is an upward-biased order statistic, so the epoch count of each run is written
next to it. Runs are kept when they logged at least ``MIN_EPOCHS`` epochs. The trainer never
ran ``trainer.test``, so no test-split metric exists for these runs.

What the runs logged (``scan_history`` columns): per epoch on both splits, Pearson, MSE,
RMSE (and ``transformed/`` duplicates), the total loss, the reconstruction and interaction
losses and their alpha-weighted versions, the integrated-embedding norm; per step, the same
training losses, ``train/alpha``, the (constant) learning rate; every other epoch on a
subsample, ``*_sample/`` MAE, Spearman, JS divergence, Wasserstein, and image panels
(correlation scatter, distribution, box plot, oversmoothing). No gradient norm. The
per-epoch curves frozen here are the ``VAL_KEYS`` + ``TRAIN_EPOCH_KEYS`` + ``TRAIN_STEP_KEYS``
columns below.

Outputs
-------
results : experiments/005-kuzmin2018-tmi/results/dango_string_version_sweep.csv   (one row per run)
          experiments/005-kuzmin2018-tmi/results/dango_string_version_summary.csv (per version x schedule)
          experiments/005-kuzmin2018-tmi/results/dango_string_version_curves.csv  (per run x epoch)
panels  : $ASSET_IMAGES_DIR/005-kuzmin2018-tmi/dango_string_version_sweep.{svg,png}
          $ASSET_IMAGES_DIR/005-kuzmin2018-tmi/dango_string_version_curves.{svg,png}
            (2 x 3 small multiples: alpha_e, validation reconstruction and interaction loss;
             train Pearson, validation Pearson, validation MSE)
table   : paper/nature-biotech/sections/tab-dango-string-versions.tex

Run from the repo root (``--from-csv`` re-renders offline from the frozen tables):
    python experiments/005-kuzmin2018-tmi/scripts/dango_string_version_sweep.py [--from-csv]
"""

import argparse
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    }
)

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogLocator, MultipleLocator, NullFormatter

from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in, savefig_true_size_svg

load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

ENTITY_PROJECT = "zhao-group/torchcell_005-kuzmin2018-tmi_dango"
RESULTS_DIR = "experiments/005-kuzmin2018-tmi/results"
RUNS_CSV = osp.join(RESULTS_DIR, "dango_string_version_sweep.csv")
SUMMARY_CSV = osp.join(RESULTS_DIR, "dango_string_version_summary.csv")
CURVES_CSV = osp.join(RESULTS_DIR, "dango_string_version_curves.csv")
SPLIT_CSV = osp.join(RESULTS_DIR, "dango_dataset_split.csv")  # label SD, from dango_construction_si.py
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "005-kuzmin2018-tmi")
TEX_PATH = "paper/nature-biotech/sections/tab-dango-string-versions.tex"

METRIC = "val/gene_interaction/Pearson"
TRAIN_METRIC = "train/gene_interaction/Pearson"
# Per-epoch curves frozen to CURVES_CSV, in three groups that are logged together so each
# group is one ``scan_history`` call per run: logged key -> column name. Epoch-end keys take
# the last value logged in the epoch; per-step keys take the mean over the epoch's steps
# except ``alpha`` and the learning rate, which are constant within an epoch (last value).
# Also logged but not frozen: RMSE, the ``transformed/`` duplicates of the metrics, the
# ``*_sample/`` metrics and images (every other epoch, on a subsample), and the parameter
# counts (summary). No gradient norm is logged.
VAL_KEYS = {
    METRIC: "val_pearson",
    "val/gene_interaction/MSE": "val_mse",
    "val/loss": "val_loss",
    "val/reconstruction_loss": "val_reconstruction_loss",
    "val/interaction_loss": "val_interaction_loss",
    "val/weighted_reconstruction_loss": "val_weighted_reconstruction_loss",
    "val/weighted_interaction_loss": "val_weighted_interaction_loss",
    "val/integrated_embeddings_norm": "val_embedding_norm",
}
TRAIN_EPOCH_KEYS = {TRAIN_METRIC: "train_pearson", "train/gene_interaction/MSE": "train_mse"}
TRAIN_STEP_KEYS = {
    "train/alpha": "alpha",
    "learning_rate": "learning_rate",
    "train/loss": "train_loss",
    "train/reconstruction_loss": "train_reconstruction_loss",
    "train/interaction_loss": "train_interaction_loss",
    "train/weighted_reconstruction_loss": "train_weighted_reconstruction_loss",
    "train/weighted_interaction_loss": "train_weighted_interaction_loss",
    "train/integrated_embeddings_norm": "train_embedding_norm",
}
STEP_LAST = {"alpha", "learning_rate"}
MIN_EPOCHS = 100
VERSIONS = ["9_1", "11_0", "12_0"]
VERSION_LABEL = {"9_1": "v9.1", "11_0": "v11.0", "12_0": "v12.0"}
SCHEDULES = ["PreThenPost", "LinearUntilUniform", "LinearUntilFlipped"]
SCHEDULE_LABEL = {
    "PreThenPost": "pretrain then main",
    "LinearUntilUniform": "linear to uniform",
    "LinearUntilFlipped": "linear to flipped",
}
SCHEDULE_STYLE = {"PreThenPost": "-", "LinearUntilUniform": (0, (4, 1.5)), "LinearUntilFlipped": (0, (1, 1))}
VERSION_COLOR = {v: PLOT_PALETTE[i] for i, v in enumerate(VERSIONS)}


def string_version(graphs: list[str]) -> str:
    versions = {g.split("_")[0].replace("string", "") + "_" + g.split("_")[1] for g in graphs}
    if len(versions) != 1:
        raise ValueError(f"run mixes STRING versions: {graphs}")
    return versions.pop()


def epoch_frame(run, keys: dict[str, str], mean_cols: set[str] = frozenset()) -> pd.DataFrame:
    """One row per epoch for a group of keys logged together: the last value in the epoch, or
    the mean over the epoch's rows for ``mean_cols``. Empty when the run never logged one of
    the keys (the smoke tests predate the weighted-loss keys); ``scan_history`` does not
    filter those out itself."""
    rows = list(run.scan_history(keys=["epoch", *keys], page_size=5000))
    df = pd.DataFrame(rows).rename(columns=keys)
    if not rows or not set(keys.values()) <= set(df.columns):
        return pd.DataFrame(columns=list(keys.values()))
    g = df.groupby("epoch")
    out = {col: (g[col].mean() if col in mean_cols else g[col].last()) for col in keys.values()}
    return pd.DataFrame(out)


def pull_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    import wandb

    api = wandb.Api(timeout=120)
    rows, curves = [], []
    for run in api.runs(ENTITY_PROJECT):
        cfg = run.config
        graphs = cfg["cell_dataset"]["graphs"]
        val_frame = epoch_frame(run, VAL_KEYS)
        # Epochs with a validation Pearson; a run killed mid-epoch leaves a last epoch with
        # per-step keys only, which must not count as logged.
        val = val_frame["val_pearson"].dropna() if len(val_frame) else pd.Series(dtype=float)
        if val.empty or val.index.max() < MIN_EPOCHS:
            # Drops the smoke tests that predate the scheduler key or stopped within a few epochs.
            print(f"  skip {run.id} ({run.state}, {0 if val.empty else int(val.index.max())} epochs)")
            continue
        schedule = cfg["regression_task"]["loss_scheduler"]["type"]
        train_frame = epoch_frame(run, TRAIN_EPOCH_KEYS)
        step_frame = epoch_frame(run, TRAIN_STEP_KEYS, mean_cols=set(TRAIN_STEP_KEYS.values()) - STEP_LAST)
        if train_frame.empty or step_frame.empty:
            raise ValueError(f"run {run.id} lacks a logged key group: train={len(train_frame)} step={len(step_frame)}")
        curve = pd.concat([val_frame, train_frame, step_frame], axis=1)
        curve.index.name = "epoch"
        curve = curve.reset_index()
        curve.insert(0, "run_id", run.id)
        curves.append(curve)
        best_epoch = int(val.idxmax())
        train = train_frame["train_pearson"].dropna()
        rows.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "string_version": string_version(graphs),
                "loss_schedule": schedule,
                "transition_epoch": cfg["regression_task"]["loss_scheduler"].get("transition_epoch"),
                "max_epochs": cfg["trainer"]["max_epochs"],
                "batch_size": cfg["data_module"]["batch_size"],
                "lr": cfg["regression_task"]["optimizer"]["lr"],
                "params_total": run.summary.get("model/params_total"),
                "epochs_logged": int(val.index.max()) + 1,
                "best_val_pearson": float(val.max()),
                "best_epoch": best_epoch,
                "train_pearson_at_best": float(train.get(best_epoch, np.nan)),
                "final_val_pearson": float(val.iloc[-1]),
                "final_train_pearson": float(train.iloc[-1]) if len(train) else np.nan,
            }
        )
        r = rows[-1]
        print(f"  {run.id} {r['string_version']:5s} {schedule:20s} best={r['best_val_pearson']:.4f} "
              f"@ {best_epoch} / {r['epochs_logged']}  train@best={r['train_pearson_at_best']:.4f}")
    runs = pd.DataFrame(rows).sort_values(["string_version", "loss_schedule", "created_at"])
    return runs, pd.concat(curves, ignore_index=True)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby(["string_version", "loss_schedule"])["best_val_pearson"]
    s = g.agg(n="count", mean="mean", sd=lambda x: x.std(ddof=1) if len(x) > 1 else np.nan).reset_index()
    s["sem"] = s["sd"] / np.sqrt(s["n"])
    s["string_version"] = pd.Categorical(s["string_version"], VERSIONS, ordered=True)
    s["loss_schedule"] = pd.Categorical(s["loss_schedule"], SCHEDULES, ordered=True)
    return s.sort_values(["string_version", "loss_schedule"]).reset_index(drop=True)


def box_axes(ax):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_color("black")
    ax.tick_params(length=2, width=0.5)


def panel(runs: pd.DataFrame, summary: pd.DataFrame):
    """Half-width panel: best validation Pearson per run, grouped by loss schedule, colored by
    STRING release; bar = mean over runs, whisker = SEM when n > 1, marker = each run. 46 mm
    tall (with the decreased-zeros panel beside it) so the 2 x 3 curves panel below fits
    the 170 mm figure."""
    w, h = PANEL_WIDTHS_MM["half"], 46.0
    fig, ax = plt.subplots(figsize=(mm_to_in(w), mm_to_in(h)))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.17)
    x = np.arange(len(SCHEDULES))
    bw = 0.26
    rng = np.random.default_rng(0)
    for i, v in enumerate(VERSIONS):
        xs = x + (i - 1) * bw
        means, sems = [], []
        for sch in SCHEDULES:
            row = summary[(summary["string_version"] == v) & (summary["loss_schedule"] == sch)]
            means.append(row["mean"].item() if len(row) else np.nan)
            sems.append(row["sem"].item() if len(row) else np.nan)
        ax.bar(xs, means, bw, color=VERSION_COLOR[v], edgecolor="black", linewidth=0.4, label=f"STRING {VERSION_LABEL[v]}")
        ax.errorbar(xs, means, yerr=sems, fmt="none", ecolor="black", elinewidth=0.6, capsize=1.5, capthick=0.6)
        for xi, sch in zip(xs, SCHEDULES):
            pts = runs[(runs["string_version"] == v) & (runs["loss_schedule"] == sch)]["best_val_pearson"]
            ax.scatter(xi + rng.uniform(-0.06, 0.06, len(pts)), pts, s=5, facecolor="white",
                       edgecolor="black", linewidth=0.4, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([SCHEDULE_LABEL[s] for s in SCHEDULES])
    ax.set_xlabel("Loss schedule (pretraining to main task)")
    ax.set_ylabel("Best validation Pearson r")
    ax.set_ylim(0, 0.56)
    ax.yaxis.set_major_locator(FixedLocator([0, 0.1, 0.2, 0.3, 0.4, 0.5]))
    ax.yaxis.set_minor_locator(FixedLocator(np.arange(0.05, 0.56, 0.1)))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="both", color="#CACACA", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", ncol=3, handlelength=1.0, columnspacing=1.0,
              handletextpad=0.4)
    box_axes(ax)
    os.makedirs(IMG_DIR, exist_ok=True)
    svg = osp.join(IMG_DIR, "dango_string_version_sweep.svg")
    savefig_true_size_svg(fig, svg)
    fig.savefig(osp.join(IMG_DIR, "dango_string_version_sweep.png"), dpi=300)
    plt.close(fig)
    print(f"  wrote {svg}")


def curves_panel(runs: pd.DataFrame, curves: pd.DataFrame):
    """Full-width 2 x 3 small multiples over epochs for every kept run; color = STRING
    release, line style = loss schedule. Top row, what the schedules did: the pretraining
    weight alpha_e (identical for every run of a schedule, drawn once per schedule), the
    validation reconstruction loss, and the validation interaction loss. Bottom row, the
    accuracy: training Pearson, validation Pearson, validation MSE with the label variance
    over all records (the MSE of predicting the mean) as a dashed reference. The schedule
    legend sits in the alpha subplot, the release legend in the interaction-loss subplot."""
    w, h = PANEL_WIDTHS_MM["full"], 56.0
    fig, axes = plt.subplots(2, 3, figsize=(mm_to_in(w), mm_to_in(h)))
    # right = 0.985 keeps the last "1000" tick label inside the canvas (it was clipped at 0.99);
    # top = 0.98 gives the panel the same ink gutter below row 2 as row 2 has below row 1.
    fig.subplots_adjust(left=0.062, right=0.985, top=0.98, bottom=0.15, wspace=0.34, hspace=0.5)
    meta = runs.set_index("run_id")
    x_max = int(curves["epoch"].max()) + 1
    label_var = float(pd.read_csv(SPLIT_CSV)["label_sd"].item()) ** 2

    def log_epoch_axis(ax):
        """The top row plots epoch + 1 on a log axis so the ten-epoch schedule window is
        visible next to the 1,000-epoch drift; the bottom row is linear in the epoch."""
        ax.set_xscale("log")
        ax.set_xlim(1, 1100)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Epoch (log scale)")

    def draw(ax, col, log_x=False):
        for rid, c in curves.groupby("run_id"):
            c = c.dropna(subset=[col]).sort_values("epoch")
            ax.plot(c["epoch"] + (1 if log_x else 0), c[col], color=VERSION_COLOR[meta.loc[rid, "string_version"]],
                    linestyle=SCHEDULE_STYLE[meta.loc[rid, "loss_schedule"]], linewidth=0.6, alpha=0.9)
        if log_x:
            log_epoch_axis(ax)
        else:
            ax.set_xlim(0, x_max)
            ax.set_xlabel("Epoch")

    # (0, 0) the schedules: every run of a schedule logs the same alpha_e, so draw it once.
    ax = axes[0, 0]
    for s in SCHEDULES:
        ids = meta.index[meta["loss_schedule"] == s]
        a = curves[curves["run_id"].isin(ids)].pivot(index="epoch", columns="run_id", values="alpha")
        if (a.max(axis=1) - a.min(axis=1)).max() > 0:
            raise ValueError(f"alpha differs between runs of schedule {s}")
        ax.plot(a.index + 1, a.iloc[:, 0], color="black", linestyle=SCHEDULE_STYLE[s], linewidth=0.8,
                label=SCHEDULE_LABEL[s])
    log_epoch_axis(ax)
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_ylabel(r"Pretraining weight $\alpha_e$")
    ax.legend(frameon=False, loc="upper right", handlelength=2.0, handletextpad=0.5, labelspacing=0.3)
    # (0, 1) and (0, 2) the two loss components on the validation split.
    ax = axes[0, 1]
    draw(ax, "val_reconstruction_loss", log_x=True)
    ax.set_ylim(0, 0.016)
    ax.yaxis.set_major_locator(MultipleLocator(0.004))
    ax.set_ylabel("Reconstruction loss\n(validation)")
    ax = axes[0, 2]
    draw(ax, "val_interaction_loss", log_x=True)
    ax.set_ylim(0.001, 0.004)
    ax.yaxis.set_major_locator(MultipleLocator(0.001))
    ax.set_ylabel("Interaction loss\n(validation)")
    handles = [Line2D([], [], color=VERSION_COLOR[v], linewidth=1.0, label=f"STRING {VERSION_LABEL[v]}") for v in VERSIONS]
    ax.legend(handles=handles, frameon=False, loc="upper right", handlelength=1.6, handletextpad=0.5, labelspacing=0.3)
    # (1, 0) to (1, 2) the accuracy curves.
    for ax, col, ylabel in zip(axes[1, :2], ["train_pearson", "val_pearson"], ["Train Pearson r", "Validation Pearson r"]):
        draw(ax, col)
        ax.set_ylim(0, 0.8)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.set_ylabel(ylabel)
    ax = axes[1, 2]
    draw(ax, "val_mse")
    ax.axhline(label_var, color="black", linestyle=(0, (3, 2)), linewidth=0.6)
    ax.set_ylim(0.002, 0.005)
    ax.yaxis.set_major_locator(MultipleLocator(0.001))
    ax.set_ylabel("Validation MSE")
    for ax in axes.ravel():
        ax.tick_params(which="minor", length=0)
        ax.grid(axis="y", which="both", color="#CACACA", linewidth=0.4)
        ax.set_axisbelow(True)
        box_axes(ax)
    svg = osp.join(IMG_DIR, "dango_string_version_curves.svg")
    savefig_true_size_svg(fig, svg)
    fig.savefig(osp.join(IMG_DIR, "dango_string_version_curves.png"), dpi=300)
    plt.close(fig)
    print(f"  wrote {svg}")


def write_tex(runs: pd.DataFrame, summary: pd.DataFrame):
    src = "experiments/005-kuzmin2018-tmi/scripts/dango_string_version_sweep.py"
    lines = [
        f"%% SOURCE: {src} -- AUTO-GENERATED from wandb {ENTITY_PROJECT}; do not hand-edit.",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\caption{DANGO replication in TorchCell on the Kuzmin 2018 trigenic interactions: best",
        "validation Pearson $r$ (maximum over epochs of the logged validation Pearson, the",
        "checkpoint-selection rule) by STRING release and loss schedule, with the training Pearson",
        "$r$ at that epoch. Mean $\\pm$ SEM over $n$ runs; a single run reports its value with no",
        "whisker. Epochs is the range of epochs logged across those runs. Every run used AdamW",
        "(learning rate $10^{-5}$, weight decay $10^{-6}$, batch 32), hidden width $d=64$, and four",
        "attention heads, on the seed-split 72,841 / 9,105 / 9,104 records (train / validation /",
        "test). No test-split metric was logged for these runs.}",
        "\\label{tab:dango-string-versions}",
        "\\begin{tabular}{@{}l l r l l l@{}}",
        "\\toprule",
        "\\textbf{STRING} & \\textbf{Loss schedule} & $n$ & \\textbf{Val.\\ Pearson $r$} & \\textbf{Train $r$ at best} & \\textbf{Epochs}\\\\",
        "\\midrule",
    ]
    for _, r in summary.iterrows():
        sub = runs[(runs["string_version"] == r["string_version"]) & (runs["loss_schedule"] == r["loss_schedule"])]
        lo, hi = sub["epochs_logged"].min(), sub["epochs_logged"].max()
        ep = f"{lo}" if lo == hi else f"{lo}--{hi}"
        val = f"{r['mean']:.3f} $\\pm$ {r['sem']:.3f}" if r["n"] > 1 else f"{r['mean']:.3f}"
        tr = sub["train_pearson_at_best"]
        train = f"{tr.mean():.3f} $\\pm$ {tr.std(ddof=1) / np.sqrt(len(tr)):.3f}" if len(tr) > 1 else f"{tr.mean():.3f}"
        lines.append(f"{VERSION_LABEL[r['string_version']]} & {SCHEDULE_LABEL[r['loss_schedule']]} & {int(r['n'])} & {val} & {train} & {ep}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    with open(TEX_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {TEX_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true", help="re-render from the frozen run tables")
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.from_csv:
        runs = pd.read_csv(RUNS_CSV)
        curves = pd.read_csv(CURVES_CSV)
    else:
        runs, curves = pull_runs()
        runs.to_csv(RUNS_CSV, index=False)
        curves.to_csv(CURVES_CSV, index=False)
        print(f"wrote {RUNS_CSV} ({len(runs)} runs) and {CURVES_CSV} ({len(curves)} run-epochs)")
    summary = summarize(runs)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(summary.to_string(index=False))
    print(f"best val Pearson over runs: {runs['best_val_pearson'].min():.4f} to {runs['best_val_pearson'].max():.4f}; "
          f"train at best: {runs['train_pearson_at_best'].min():.4f} to {runs['train_pearson_at_best'].max():.4f}")
    panel(runs, summary)
    curves_panel(runs, curves)
    write_tex(runs, summary)


if __name__ == "__main__":
    main()
