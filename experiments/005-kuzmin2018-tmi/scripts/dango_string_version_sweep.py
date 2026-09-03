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
next to it. Runs are kept when they logged at least ``MIN_EPOCHS`` epochs.

Outputs
-------
results : experiments/005-kuzmin2018-tmi/results/dango_string_version_sweep.csv   (one row per run)
          experiments/005-kuzmin2018-tmi/results/dango_string_version_summary.csv (per version x schedule)
panel   : $ASSET_IMAGES_DIR/005-kuzmin2018-tmi/dango_string_version_sweep.{svg,png}
table   : paper/nature-biotech/sections/tab-dango-string-versions.tex

Run from the repo root (``--from-csv`` re-renders offline from the frozen run table):
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
    }
)

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import FixedLocator

from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in, savefig_true_size_svg

load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

ENTITY_PROJECT = "zhao-group/torchcell_005-kuzmin2018-tmi_dango"
RESULTS_DIR = "experiments/005-kuzmin2018-tmi/results"
RUNS_CSV = osp.join(RESULTS_DIR, "dango_string_version_sweep.csv")
SUMMARY_CSV = osp.join(RESULTS_DIR, "dango_string_version_summary.csv")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "005-kuzmin2018-tmi")
TEX_PATH = "paper/nature-biotech/sections/tab-dango-string-versions.tex"

METRIC = "val/gene_interaction/Pearson"
MIN_EPOCHS = 100
VERSIONS = ["9_1", "11_0", "12_0"]
VERSION_LABEL = {"9_1": "v9.1", "11_0": "v11.0", "12_0": "v12.0"}
SCHEDULES = ["PreThenPost", "LinearUntilUniform", "LinearUntilFlipped"]
SCHEDULE_LABEL = {
    "PreThenPost": "pretrain then main",
    "LinearUntilUniform": "linear to uniform",
    "LinearUntilFlipped": "linear to flipped",
}
VERSION_COLOR = {v: PLOT_PALETTE[i] for i, v in enumerate(VERSIONS)}


def string_version(graphs: list[str]) -> str:
    versions = {g.split("_")[0].replace("string", "") + "_" + g.split("_")[1] for g in graphs}
    if len(versions) != 1:
        raise ValueError(f"run mixes STRING versions: {graphs}")
    return versions.pop()


def pull_runs() -> pd.DataFrame:
    import wandb

    api = wandb.Api(timeout=120)
    rows = []
    for run in api.runs(ENTITY_PROJECT):
        cfg = run.config
        graphs = cfg["cell_dataset"]["graphs"]
        hist = pd.DataFrame(list(run.scan_history(keys=["epoch", METRIC])))
        if hist.empty or hist["epoch"].max() < MIN_EPOCHS:
            # The one run this drops is the 9-epoch smoke test that predates the scheduler key.
            print(f"  skip {run.id} ({run.state}, {0 if hist.empty else int(hist['epoch'].max())} epochs)")
            continue
        schedule = cfg["regression_task"]["loss_scheduler"]["type"]
        best = hist.loc[hist[METRIC].idxmax()]
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
                "epochs_logged": int(hist["epoch"].max()) + 1,
                "best_val_pearson": float(best[METRIC]),
                "best_epoch": int(best["epoch"]),
                "final_val_pearson": float(hist.sort_values("epoch")[METRIC].iloc[-1]),
            }
        )
        print(f"  {run.id} {rows[-1]['string_version']:5s} {schedule:20s} best={rows[-1]['best_val_pearson']:.4f} @ {rows[-1]['best_epoch']} / {rows[-1]['epochs_logged']}")
    return pd.DataFrame(rows).sort_values(["string_version", "loss_schedule", "created_at"])


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby(["string_version", "loss_schedule"])["best_val_pearson"]
    s = g.agg(n="count", mean="mean", sd=lambda x: x.std(ddof=1) if len(x) > 1 else np.nan).reset_index()
    s["sem"] = s["sd"] / np.sqrt(s["n"])
    s["string_version"] = pd.Categorical(s["string_version"], VERSIONS, ordered=True)
    s["loss_schedule"] = pd.Categorical(s["loss_schedule"], SCHEDULES, ordered=True)
    return s.sort_values(["string_version", "loss_schedule"]).reset_index(drop=True)


def panel(runs: pd.DataFrame, summary: pd.DataFrame):
    """Half-width panel: best validation Pearson per run, grouped by loss schedule, colored by
    STRING release; bar = mean over runs, whisker = SEM when n > 1, marker = each run."""
    w, h = PANEL_WIDTHS_MM["half"], 52.0
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
    ax.tick_params(length=2, width=0.5)
    ax.legend(frameon=False, loc="upper center", ncol=3, handlelength=1.0, columnspacing=1.0,
              handletextpad=0.4)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_color("black")
    os.makedirs(IMG_DIR, exist_ok=True)
    svg = osp.join(IMG_DIR, "dango_string_version_sweep.svg")
    savefig_true_size_svg(fig, svg)
    fig.savefig(osp.join(IMG_DIR, "dango_string_version_sweep.png"), dpi=300)
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
        "checkpoint-selection rule) by STRING release and loss schedule. Mean $\\pm$ SEM over $n$",
        "runs; a single run reports its value with no whisker. Epochs is the range of epochs logged",
        "across those runs.}",
        "\\label{tab:dango-string-versions}",
        "\\begin{tabular}{@{}l l r l l@{}}",
        "\\toprule",
        "\\textbf{STRING} & \\textbf{Loss schedule} & $n$ & \\textbf{Val.\\ Pearson $r$} & \\textbf{Epochs}\\\\",
        "\\midrule",
    ]
    for _, r in summary.iterrows():
        sub = runs[(runs["string_version"] == r["string_version"]) & (runs["loss_schedule"] == r["loss_schedule"])]
        lo, hi = sub["epochs_logged"].min(), sub["epochs_logged"].max()
        ep = f"{lo}" if lo == hi else f"{lo}--{hi}"
        val = f"{r['mean']:.3f} $\\pm$ {r['sem']:.3f}" if r["n"] > 1 else f"{r['mean']:.3f}"
        lines.append(f"{VERSION_LABEL[r['string_version']]} & {SCHEDULE_LABEL[r['loss_schedule']]} & {int(r['n'])} & {val} & {ep}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    with open(TEX_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {TEX_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true", help="re-render from the frozen run CSV")
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.from_csv:
        runs = pd.read_csv(RUNS_CSV)
    else:
        runs = pull_runs()
        runs.to_csv(RUNS_CSV, index=False)
        print(f"wrote {RUNS_CSV} ({len(runs)} runs)")
    summary = summarize(runs)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(summary.to_string(index=False))
    panel(runs, summary)
    write_tex(runs, summary)


if __name__ == "__main__":
    main()
