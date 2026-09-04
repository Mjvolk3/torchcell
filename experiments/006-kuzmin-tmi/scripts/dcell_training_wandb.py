# experiments/006-kuzmin-tmi/scripts/dcell_training_wandb.py
# [[experiments.006-kuzmin-tmi.scripts.dcell_training_wandb]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/006-kuzmin-tmi/scripts/dcell_training_wandb
"""DCell training cost, instability, and the checkpoint-based Fig. 2 baseline.

The DCell value in Fig. 2d (0.157 +/- 0.009) is computed over three validation
evaluations of ONE training run, not over replicate runs. This script freezes that run,
recomputes the statistic from the logged history, and puts its cost next to the CGT and
DANGO runs on the same task.

wandb sources (entity ``zhao-group``; every run id below is recorded in ``runs.csv``):

* DCell, experiment 006 build ``torchcell_006-kuzmin-tmi_dcell``:
  SLURM job 1922684 (config ``dcell_kuzmin2018_tmi_mmli_001``, auxiliary losses on),
  rank 0 = ``eni948by``; job 1921740 (``mmli_000``, auxiliary losses off, stopped after
  10 epochs), rank 0 = ``x69dyevg``. Each DDP rank is a separate wandb run; only rank 0
  logs the validation metrics.
* DCell, experiment 005 build ``torchcell_005-kuzmin2018-tmi_dcell``: job 1811673,
  rank 0 = ``biucpv7p`` (earlier dataset build, context only).
* CGT, ``torchcell_010-kuzmin-tmi_equivariant_cell_graph_transformer``: the three
  training jobs whose best-Pearson checkpoints were re-evaluated for Fig. 2 (eval runs
  tagged ``evaluation``/``full_dataset`` record the checkpoint path).
* DANGO, ``torchcell_006-kuzmin2018-tmi_dango``: the two 1000-epoch training runs.

Outputs
-------
results : experiments/006-kuzmin-tmi/results/dcell_training/
          runs.csv, history_<run>.csv (epoch level), history_full_eni948by.csv,
          checkpoints.csv, cost.csv, speedup_stages.csv
panels  : $ASSET_IMAGES_DIR/006-kuzmin-tmi/dcell_training_{val_pearson,loss,cost,stages}.{svg,png}
tables  : paper/nature-biotech/sections/tab-dcell-training-checkpoints.tex,
          tab-dcell-training-cost.tex

Run from the repo root:
    python experiments/006-kuzmin-tmi/scripts/dcell_training_wandb.py            # pull + render
    python experiments/006-kuzmin-tmi/scripts/dcell_training_wandb.py --from-csv # re-render only
"""

import argparse
import glob
import os
import os.path as osp
import re
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in, savefig_true_size_svg

# Set AFTER the torchcell imports: the repo mplstyle is applied on import by some
# torchcell modules and would override these.
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
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "lines.linewidth": 0.8,
        "svg.fonttype": "none",
        "savefig.bbox": None,
    }
)

load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
assert ASSET_IMAGES_DIR is not None, "ASSET_IMAGES_DIR must be set in the environment"

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
EXP_DIR = osp.dirname(SCRIPT_DIR)
REPO_ROOT = osp.dirname(osp.dirname(EXP_DIR))
RESULTS = osp.join(EXP_DIR, "results", "dcell_training")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "006-kuzmin-tmi")
TEX_DIR = osp.join(REPO_ROOT, "paper", "nature-biotech", "sections")
SPEEDUP_NOTE = osp.join(REPO_ROOT, "notes", "experiments.006-kuzmin-tmi.dcell-speed-up.md")
ENTITY = "zhao-group"

# Fig. 2 colors (Yeast9 yellow, DCell purple, DANGO orange, TorchCell red), from PLOT_PALETTE.
ORANGE, RED, PURPLE, YELLOW, BLUE, GRAY = PLOT_PALETTE[:6]
MODEL_COLOR = {"DCell": PURPLE, "CGT": RED, "DANGO": ORANGE}

# Dataset builds. Record counts from
# notes/experiments.011-kuzmin-tmi.scripts.query-comparison-006-009-010-011.md (006: 332,313;
# 010: 376,732) and notes/experiments.010-kuzmin-tmi.performance-diff-010-009.md
# (010 train split 301,386). Samples per epoch below are MEASURED from optimizer steps.
BUILD_RECORDS = {"006": 332_313, "010": 376_732}

# The three DCell validation Pearson values hardcoded in
# experiments/010-kuzmin-tmi/scripts/trigenic_tau_model_comparison.py (Fig. 2d).
FIG2_DCELL_VALUES = [0.17321017384529114, 0.1550033837556839, 0.14192065596580505]

RUNS = [
    # model, project, run id, slurm job, rank, role, dataset build
    ("DCell", "torchcell_006-kuzmin-tmi_dcell", "eni948by", 1922684, 0, "train", "006"),
    ("DCell", "torchcell_006-kuzmin-tmi_dcell", "lrudrans", 1922684, None, "train", "006"),
    ("DCell", "torchcell_006-kuzmin-tmi_dcell", "fy516knk", 1922684, None, "train", "006"),
    ("DCell", "torchcell_006-kuzmin-tmi_dcell", "c7248f86", 1922684, None, "train", "006"),
    ("DCell", "torchcell_006-kuzmin-tmi_dcell", "x69dyevg", 1921740, 0, "train", "006"),
    ("DCell", "torchcell_006-kuzmin-tmi_dcell", "tjmmnyoy", 1921740, None, "train", "006"),
    ("DCell", "torchcell_006-kuzmin-tmi_dcell", "obr9nmy9", 1921740, None, "train", "006"),
    ("DCell", "torchcell_006-kuzmin-tmi_dcell", "dttu9dx2", 1921740, None, "train", "006"),
    ("DCell", "torchcell_005-kuzmin2018-tmi_dcell", "biucpv7p", 1811673, 0, "train", "005"),
    ("CGT", "torchcell_010-kuzmin-tmi_equivariant_cell_graph_transformer", "lzs9pcj3", 2027905, 0, "train", "010"),
    ("CGT", "torchcell_010-kuzmin-tmi_equivariant_cell_graph_transformer", "yv4r30bi", 2027907, 0, "train", "010"),
    ("CGT", "torchcell_010-kuzmin-tmi_equivariant_cell_graph_transformer", "c7671wgj", 2036902, 0, "train", "010"),
    ("CGT", "torchcell_010-kuzmin-tmi_equivariant_cell_graph_transformer", "leodrxht", 2027905, 0, "eval", "010"),
    ("CGT", "torchcell_010-kuzmin-tmi_equivariant_cell_graph_transformer", "cvu2ryfw", 2027907, 0, "eval", "010"),
    ("CGT", "torchcell_010-kuzmin-tmi_equivariant_cell_graph_transformer", "0psour3n", 2036902, 0, "eval", "010"),
    ("DANGO", "torchcell_006-kuzmin2018-tmi_dango", "014mprap", 1941704, 0, "train", "006"),
    ("DANGO", "torchcell_006-kuzmin2018-tmi_dango", "9jpfy547", 1940775, 0, "train", "006"),
]
HISTORY_RUNS = ["eni948by", "x69dyevg", "biucpv7p", "lzs9pcj3", "yv4r30bi", "c7671wgj", "014mprap", "9jpfy547"]
# DANGO histories are 214k rows (1000 epochs); the wandb history artifact (parquet) is the
# fast path, scan_history times out.
ARTIFACT_HISTORY_RUNS = {"014mprap", "9jpfy547"}

HIST_KEYS = [
    "epoch",
    "_step",
    "_runtime",
    "_timestamp",
    "trainer/global_step",
    "learning_rate",
    "train/loss",
    "train/primary_loss",
    "train/auxiliary_loss",
    "val/loss",
    "val/gene_interaction/Pearson",
    "val/gene_interaction/MSE",
    "val_sample/Pearson_target_0",
]


# ----------------------------------------------------------------------------- wandb pull
def pull_runs(api) -> pd.DataFrame:
    rows = []
    for model, project, rid, job, rank, role, build in RUNS:
        r = api.run(f"{ENTITY}/{project}/{rid}")
        c, s = r.config, r.summary
        tr, dm = c.get("trainer", {}), c.get("data_module", {})
        rows.append(
            {
                "model": model,
                "project": project,
                "run_id": rid,
                "slurm_job": job,
                "ddp_rank": rank,
                "role": role,
                "dataset_build": build,
                "state": r.state,
                "created_at": r.created_at,
                "runtime_s": s.get("_runtime"),
                "last_step": s.get("_step"),
                "devices": tr.get("devices"),
                "strategy": tr.get("strategy"),
                "precision": tr.get("precision", "32-true"),
                "max_epochs": tr.get("max_epochs"),
                "batch_size_per_gpu": dm.get("batch_size"),
                "num_workers": dm.get("num_workers"),
                "lr": c.get("regression_task", {}).get("optimizer", {}).get("lr"),
                "lr_scheduler_min_lr": c.get("regression_task", {}).get("lr_scheduler", {}).get("min_lr"),
                "use_auxiliary_losses": c.get("regression_task", {}).get("dcell_loss", {}).get("use_auxiliary_losses"),
                "aux_alpha": c.get("regression_task", {}).get("dcell_loss", {}).get("alpha"),
                "checkpoint_path": c.get("model", {}).get("checkpoint_path"),
                "params_total": s.get("model/params_total"),
                "num_go_terms": s.get("model/num_go_terms"),
                "summary_val_pearson": s.get("val/gene_interaction/Pearson"),
                "summary_test_pearson": s.get("test/gene_interaction/Pearson"),
                "tags": ";".join(r.tags),
            }
        )
    return pd.DataFrame(rows)


def pull_history(api, project: str, rid: str) -> pd.DataFrame:
    r = api.run(f"{ENTITY}/{project}/{rid}")
    if rid in ARTIFACT_HISTORY_RUNS:
        art = [a for a in r.logged_artifacts() if a.type == "wandb-history"][0]
        with tempfile.TemporaryDirectory() as tmp:
            d = art.download(root=tmp)
            df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{d}/**/*.parquet", recursive=True))])
    else:
        df = pd.DataFrame(list(r.scan_history()))
    cols = [k for k in HIST_KEYS if k in df.columns]
    return df[cols].sort_values("_step").reset_index(drop=True)


def epoch_level(full: pd.DataFrame) -> pd.DataFrame:
    """One row per validation evaluation, plus the mean train loss over that epoch."""
    v = full.dropna(subset=["val/gene_interaction/Pearson"]).copy()
    if "train/loss" in full.columns:
        tl = full.dropna(subset=["train/loss"]).groupby("epoch")["train/loss"].mean()
        v["train_loss_epoch_mean"] = v["epoch"].map(tl)
    keep = [c for c in v.columns if c not in ("train/loss", "train/primary_loss", "train/auxiliary_loss")]
    return v[keep].reset_index(drop=True)


# ----------------------------------------------------------------------------- speed-up note
def parse_speedup_note(path: str) -> pd.DataFrame:
    """Seconds per optimizer step at each stage of the DCell speed-up work on gilahyper.

    Each ``##`` section of the note is a pasted training log. The progress bar's last
    logged line gives elapsed wall-clock and iterations completed; elapsed / iterations is
    the mean step time INCLUDING the first-step warmup, so it is an upper bound on the
    steady-state step time. The section's ``wandb_cfg`` line gives the configuration.
    """
    txt = open(path, encoding="utf-8").read()
    rows = []
    for sec in re.split(r"^## ", txt, flags=re.M)[1:]:
        title, body = sec.split("\n", 1)
        bars = re.findall(
            r"Epoch (\d+):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+) \[(\d+):(\d+)(?::(\d+))?<", body
        )
        if not bars:
            continue
        ep, it, total, a, b, c = bars[-1]
        elapsed = int(a) * 3600 + int(b) * 60 + int(c) if c else int(a) * 60 + int(b)
        cfg = re.search(r"wandb_cfg (\{.*\})", body)
        cfg = cfg.group(1) if cfg else ""

        def grab(key, default):
            m = re.search(rf"'{key}': '?([^,'}}]+)'?", cfg)
            return m.group(1) if m else default

        rows.append(
            {
                "stage": title.strip(),
                "batch_size_per_gpu": int(grab("batch_size", 0)),
                "num_workers": int(grab("num_workers", 0)),
                "precision": grab("precision", "32-true"),
                "compile_mode": grab("compile_mode", "None"),
                "devices": int(grab("devices", 0)),
                "steps_done": int(it),
                "steps_per_epoch": int(total),
                "elapsed_s": elapsed,
                "s_per_step": elapsed / int(it),
            }
        )
    df = pd.DataFrame(rows)
    df["samples_per_s"] = df["batch_size_per_gpu"] * df["devices"] / df["s_per_step"]
    return df


# ----------------------------------------------------------------------------- statistics
def run_cost(hist: pd.DataFrame, meta: pd.Series, samples_per_epoch: float) -> dict:
    v = hist.sort_values("epoch")
    best = v.loc[v["val/gene_interaction/Pearson"].idxmax()]
    dt = np.diff(v["_runtime"].values)
    return {
        "model": meta["model"],
        "run_id": meta["run_id"],
        "slurm_job": meta["slurm_job"],
        "dataset_build": meta["dataset_build"],
        "build_records": BUILD_RECORDS.get(meta["dataset_build"]),
        "gpus": int(meta["devices"]),
        "global_batch": int(meta["batch_size_per_gpu"]) * int(meta["devices"]),
        "steps_per_epoch": v["trainer/global_step"].max() / (v["epoch"].max() + 1),
        "samples_per_epoch": samples_per_epoch,
        "epochs_logged": int(v["epoch"].max()) + 1,
        "max_epochs": int(meta["max_epochs"]),
        "epoch_time_h_median": float(np.median(dt)) / 3600,
        "samples_per_s": samples_per_epoch / float(np.median(dt)),
        "best_val_pearson": float(best["val/gene_interaction/Pearson"]),
        "best_epoch": int(best["epoch"]),
        "hours_to_best": float(best["_runtime"]) / 3600,
        "gpu_hours_to_best": float(best["_runtime"]) / 3600 * int(meta["devices"]),
        "total_hours": float(v["_runtime"].max()) / 3600,
        "total_gpu_hours": float(v["_runtime"].max()) / 3600 * int(meta["devices"]),
        "final_val_pearson": float(v["val/gene_interaction/Pearson"].iloc[-1]),
    }


def checkpoint_table(hist: pd.DataFrame) -> pd.DataFrame:
    v = hist.sort_values("epoch").reset_index(drop=True)
    v["rank_desc"] = v["val/gene_interaction/Pearson"].rank(ascending=False, method="min").astype(int)
    rows = []
    for val in FIG2_DCELL_VALUES:
        i = (v["val/gene_interaction/Pearson"] - val).abs().idxmin()
        r = v.loc[i]
        assert abs(r["val/gene_interaction/Pearson"] - val) < 1e-9, "Fig. 2 value not found in history"
        rows.append(
            {
                "epoch": int(r["epoch"]),
                "val_pearson": float(r["val/gene_interaction/Pearson"]),
                "val_loss": float(r["val/loss"]),
                "wall_clock_days": float(r["_runtime"]) / 86400,
                "rank_among_evals": int(r["rank_desc"]),
                "n_evals": len(v),
            }
        )
    return pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)


# ----------------------------------------------------------------------------- tables
def fmt(x, nd=3):
    return f"{x:.{nd}f}"


def write_checkpoint_tex(ck: pd.DataFrame, hist: pd.DataFrame, cost_row: dict, path: str):
    vals = ck["val_pearson"].values
    mean, sd, sem = vals.mean(), vals.std(ddof=1), vals.std(ddof=1) / np.sqrt(len(vals))
    v = hist.sort_values("epoch")
    last = v.iloc[-1]
    # The training script's ModelCheckpoint monitors val/gene_interaction/MSE (mode=min).
    mse_best = v.loc[v["val/gene_interaction/MSE"].idxmin()]
    src = "experiments/006-kuzmin-tmi/scripts/dcell_training_wandb.py"
    lines = [
        f"%% SOURCE: {src} -- AUTO-GENERATED, do not hand-edit; rerun the script.",
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{The DCell value in Fig.~\ref{fig:ggi}d and the run it comes from. All values are",
        r"the validation Pearson~$r$ logged by the single DCell training run (wandb",
        f"\\texttt{{{cost_row['run_id']}}}, SLURM job {cost_row['slurm_job']}, {cost_row['epochs_logged']} epochs, "
        f"{cost_row['total_hours']/24:.1f} days on {cost_row['gpus']} GPUs). \\emph{{Rank}} is the value's rank among",
        f"the {int(ck['n_evals'].iloc[0])} per-epoch validation evaluations of that run. The three epochs are",
        r"evaluations of one run, so their spread is checkpoint-to-checkpoint variation, not replicate",
        r"variance, and the reported SEM is not a replicate SEM.}",
        r"\label{tab:dcell-training-checkpoints}",
        r"\begin{tabular}{@{}l r r r r@{}}",
        r"\toprule",
        r"\textbf{Evaluation} & \textbf{Epoch} & \textbf{Wall-clock (d)} & \textbf{Val.\ Pearson $r$} & \textbf{Rank}\\",
        r"\midrule",
    ]
    for _, r in ck.iterrows():
        lines.append(
            f"Fig.~\\ref{{fig:ggi}}d checkpoint & {int(r['epoch'])} & {r['wall_clock_days']:.1f} & {fmt(r['val_pearson'])} & {int(r['rank_among_evals'])}\\\\"
        )
    lines += [
        r"\midrule",
        f"Mean of the three (Fig.~\\ref{{fig:ggi}}d) & & & {fmt(mean)} & \\\\",
        f"SD over the three (ddof$=$1) & & & {fmt(sd)} & \\\\",
        f"SEM $=$ SD$/\\sqrt{{3}}$ (Fig.~\\ref{{fig:ggi}}d error bar) & & & {fmt(sem, 4)} & \\\\",
        r"\midrule",
        f"Maximum over all epochs & {cost_row['best_epoch']} & {cost_row['hours_to_best']/24:.1f} & {fmt(cost_row['best_val_pearson'])} & 1\\\\",
        f"Minimum validation MSE (the saved checkpoint) & {int(mse_best['epoch'])} & {mse_best['_runtime']/86400:.1f} & {fmt(mse_best['val/gene_interaction/Pearson'])} & {int(mse_best['rank_desc'])}\\\\",
        f"Final epoch & {int(last['epoch'])} & {last['_runtime']/86400:.1f} & {fmt(last['val/gene_interaction/Pearson'])} & {int(last['rank_desc']) if 'rank_desc' in last else ''}\\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def write_cost_tex(cost: pd.DataFrame, path: str):
    src = "experiments/006-kuzmin-tmi/scripts/dcell_training_wandb.py"
    lines = [
        f"%% SOURCE: {src} -- AUTO-GENERATED, do not hand-edit; rerun the script.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Training cost of the three learned models compared in Fig.~\ref{fig:ggi}d, from the",
        r"wandb histories of the runs listed in \texttt{results/dcell\_training/runs.csv}. Samples per",
        r"epoch is optimizer steps per epoch times the global batch. \emph{Best} is the epoch of the",
        r"maximum validation Pearson~$r$ over the run; the wall-clock and GPU-hours to reach it are the",
        r"logged run time at that epoch. CGT rows are the three replicate training jobs whose",
        r"best-Pearson checkpoints give the Fig.~\ref{fig:ggi}d value; DANGO rows are the two 1{,}000-epoch training",
        r"runs on the same build as DCell. DCell and DANGO trained on the experiment-006 build",
        r"(332,313 records); the CGT replicates trained on the experiment-010 build (376,732 records).}",
        r"\label{tab:dcell-training-cost}",
        r"\begin{tabular}{@{}l l l r r r r r r r r@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Run} & \textbf{Build} & \textbf{GPUs} & \textbf{Samples/epoch} & \textbf{h/epoch} & \textbf{Samples/s} & \textbf{Best epoch} & \textbf{Best $r$} & \textbf{h to best} & \textbf{GPU-h to best}\\",
        r"\midrule",
    ]
    for _, r in cost.iterrows():
        lines.append(
            f"{r['model']} & \\texttt{{{r['run_id']}}} & {r['dataset_build']} & {r['gpus']} & {r['samples_per_epoch']:,.0f} & "
            f"{r['epoch_time_h_median']:.2f} & {r['samples_per_s']:,.0f} & {r['best_epoch']} & {fmt(r['best_val_pearson'])} & "
            f"{r['hours_to_best']:.1f} & {r['gpu_hours_to_best']:,.0f}\\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"wrote {path}")


# ----------------------------------------------------------------------------- panels
def box(ax):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_edgecolor("black")


def save(fig, name):
    os.makedirs(IMG_DIR, exist_ok=True)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, f"{name}.svg"))
    fig.savefig(osp.join(IMG_DIR, f"{name}.png"), dpi=300)
    plt.close(fig)
    print(f"saved {osp.join(IMG_DIR, name)}.{{svg,png}}")


def panel_val_pearson(hist: pd.DataFrame, ck: pd.DataFrame, hist_aux_off: pd.DataFrame):
    w = mm_to_in(PANEL_WIDTHS_MM["half"])
    fig, ax = plt.subplots(figsize=(w, mm_to_in(52)))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.17, top=0.84)
    v = hist.sort_values("epoch")
    h_per_epoch = np.median(np.diff(v["_runtime"].values)) / 3600
    ax.plot(v["epoch"], v["val/gene_interaction/Pearson"], color=PURPLE, lw=0.8, label="job 1922684, aux. losses on")
    a = hist_aux_off.sort_values("epoch")
    ax.plot(a["epoch"], a["val/gene_interaction/Pearson"], color=GRAY, lw=0.8, ls="--", label=f"job 1921740, aux. losses off ({int(a['epoch'].max()) + 1} epochs, stopped)")
    ax.scatter(ck["epoch"], ck["val_pearson"], s=12, marker="o", facecolor="none", edgecolor=RED, lw=0.7, zorder=5, label="Fig. 2d checkpoints")
    ax.axhline(ck["val_pearson"].mean(), color=RED, lw=0.5, ls=":")
    ax.text(v["epoch"].max(), ck["val_pearson"].mean() + 0.004, f"mean {ck['val_pearson'].mean():.3f}", ha="right", va="bottom", color=RED, fontsize=6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Pearson r")
    ax.set_xlim(0, v["epoch"].max() + 1)
    ax.set_ylim(-0.05, 0.2)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.grid(axis="y", color="#D0D0D0", lw=0.4)
    ax.set_axisbelow(True)
    sec = ax.secondary_xaxis("top", functions=(lambda e: e * h_per_epoch / 24, lambda d: d * 24 / h_per_epoch))
    sec.set_xlabel("Wall-clock (days, 4 GPUs)")
    sec.xaxis.set_major_locator(MultipleLocator(5))
    ax.legend(loc="lower right", frameon=False, handlelength=1.6, borderpad=0.2, labelspacing=0.3)
    box(ax)
    save(fig, "dcell_training_val_pearson")


def panel_loss(hist: pd.DataFrame, full: pd.DataFrame, ck: pd.DataFrame):
    w = mm_to_in(PANEL_WIDTHS_MM["half"])
    fig, ax = plt.subplots(figsize=(w, mm_to_in(52)))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.17, top=0.84)
    steps = full.dropna(subset=["train/loss"])
    n_step = steps["trainer/global_step"].max() / (steps["epoch"].max() + 1)
    ax.plot(steps["trainer/global_step"] / n_step, steps["train/loss"], color=YELLOW, lw=0.4, label="train loss (per 10 steps)")
    v = hist.sort_values("epoch")
    ax.plot(v["epoch"], v["val/loss"], color=PURPLE, lw=0.8, label="validation loss (per epoch)")
    for e in ck["epoch"]:
        ax.axvline(e, color=RED, lw=0.4, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("DCell loss (log scale)")
    ax.set_xlim(0, v["epoch"].max() + 1)
    ax.grid(axis="y", color="#D0D0D0", lw=0.4, which="major")
    ax.set_axisbelow(True)
    lr = full["learning_rate"].dropna()
    assert lr.min() == lr.max(), "learning rate was reduced during the run; update the label"
    ax.text(0.02, 0.04, f"learning rate constant at {lr.max():.0e} (min_lr = lr)", transform=ax.transAxes, fontsize=6, va="bottom")
    ax.legend(loc="upper right", frameon=False, handlelength=1.6, borderpad=0.2, labelspacing=0.3)
    box(ax)
    save(fig, "dcell_training_loss")


def panel_cost(cost: pd.DataFrame):
    w = mm_to_in(PANEL_WIDTHS_MM["half"])
    fig, axes = plt.subplots(1, 2, figsize=(w, mm_to_in(52)))
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.17, top=0.9, wspace=0.55)
    order = ["DCell", "DANGO", "CGT"]
    labels = {"DCell": "DCell", "DANGO": "DANGO", "CGT": "TorchCell\n(CGT)"}
    for ax, col, ylab in [
        (axes[0], "gpu_hours_to_best", "GPU-hours to best validation epoch"),
        (axes[1], "samples_per_s", "Training samples per second"),
    ]:
        for i, m in enumerate(order):
            sub = cost[cost["model"] == m]
            ax.bar(i, sub[col].mean(), width=0.6, color=MODEL_COLOR[m], edgecolor="black", lw=0.5, zorder=3)
            ax.scatter([i] * len(sub), sub[col], s=6, color="black", zorder=4)
            top = sub[col].max()
            ax.text(i, top * 1.25, f"{sub[col].mean():,.0f}", ha="center", va="bottom", fontsize=6)
        ax.set_yscale("log")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([labels[m] for m in order])
        ax.set_ylabel(ylab)
        ax.grid(axis="y", color="#D0D0D0", lw=0.4, which="major")
        ax.set_axisbelow(True)
        box(ax)
    axes[0].set_ylim(1, 1e4)
    axes[1].set_ylim(10, 1e4)
    save(fig, "dcell_training_cost")


def panel_stages(st: pd.DataFrame):
    w = mm_to_in(PANEL_WIDTHS_MM["half"])
    fig, ax = plt.subplots(figsize=(w, mm_to_in(52)))
    fig.subplots_adjust(left=0.5, right=0.96, bottom=0.17, top=0.95)
    short = {
        "Before Duplicate Forward": "baseline (duplicate forward)",
        "After Removing Duplicate Forward": "single forward",
        "After Caching GO Strata": "+ cached GO strata",
        "16-Mixed Precision": "+ fp16-mixed",
        "BF16-Mixed Precision": "+ bf16-mixed",
        "BF16-Mixed Precision Regress": "bf16-mixed, rerun",
        "BF16-Mixed Precison - 12 workers": "bf16-mixed, 12 workers",
        "BF16-Mixed Precison - 8 Workers - Compile - batch size 500": "+ torch.compile, batch 500",
        "BF16-Mixed Precison - 8 Workers - Compile - batch size 600": "+ torch.compile, batch 600",
    }
    y = np.arange(len(st))[::-1]
    ax.barh(y, st["samples_per_s"], color=PURPLE, edgecolor="black", lw=0.5, height=0.65, zorder=3)
    for yi, (_, r) in zip(y, st.iterrows()):
        ax.text(r["samples_per_s"] * 1.08, yi, f"{r['s_per_step']:.0f} s/step", va="center", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels([short.get(s, s) for s in st["stage"]])
    ax.set_xscale("log")
    ax.set_xlim(3, 300)
    ax.set_xlabel("Training samples per second (4 GPUs, gilahyper)")
    ax.grid(axis="x", color="#D0D0D0", lw=0.4, which="major")
    ax.set_axisbelow(True)
    box(ax)
    save(fig, "dcell_training_stages")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true", help="re-render from the frozen CSVs, no wandb access")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    if not args.from_csv:
        import wandb

        api = wandb.Api()
        runs = pull_runs(api)
        runs.to_csv(osp.join(RESULTS, "runs.csv"), index=False)
        for rid in HISTORY_RUNS:
            project = runs.loc[runs["run_id"] == rid, "project"].iloc[0]
            full = pull_history(api, project, rid)
            if rid == "eni948by":
                full.to_csv(osp.join(RESULTS, "history_full_eni948by.csv"), index=False)
            epoch_level(full).to_csv(osp.join(RESULTS, f"history_{rid}.csv"), index=False)
            print(f"froze history for {rid}: {len(full)} rows")
        parse_speedup_note(SPEEDUP_NOTE).to_csv(osp.join(RESULTS, "speedup_stages.csv"), index=False)

    runs = pd.read_csv(osp.join(RESULTS, "runs.csv"), dtype={"dataset_build": str})
    hist = {rid: pd.read_csv(osp.join(RESULTS, f"history_{rid}.csv")) for rid in HISTORY_RUNS}
    full = pd.read_csv(osp.join(RESULTS, "history_full_eni948by.csv"))
    stages = pd.read_csv(osp.join(RESULTS, "speedup_stages.csv"))

    # Cost per run: samples per epoch from optimizer steps (measured), not from the split size.
    cost_rows = []
    for rid in ["eni948by", "lzs9pcj3", "yv4r30bi", "c7671wgj", "014mprap", "9jpfy547"]:
        meta = runs[runs["run_id"] == rid].iloc[0]
        v = hist[rid]
        steps_per_epoch = v["trainer/global_step"].max() / (v["epoch"].max() + 1)
        spe = steps_per_epoch * meta["batch_size_per_gpu"] * meta["devices"]
        cost_rows.append(run_cost(v, meta, spe))
    cost = pd.DataFrame(cost_rows)
    cost.to_csv(osp.join(RESULTS, "cost.csv"), index=False)

    ck = checkpoint_table(hist["eni948by"])
    ck.to_csv(osp.join(RESULTS, "checkpoints.csv"), index=False)
    vals = ck["val_pearson"].values
    print(ck.to_string())
    print(f"Fig. 2d DCell statistic over {len(vals)} evaluations of one run: mean {vals.mean():.4f}, "
          f"SD {vals.std(ddof=1):.4f}, SEM {vals.std(ddof=1)/np.sqrt(len(vals)):.4f}")
    print(cost.to_string())
    print(stages.to_string())

    dcell_cost = cost[cost["run_id"] == "eni948by"].iloc[0].to_dict()
    h = hist["eni948by"].sort_values("epoch").reset_index(drop=True)
    h["rank_desc"] = h["val/gene_interaction/Pearson"].rank(ascending=False, method="min").astype(int)
    write_checkpoint_tex(ck, h, dcell_cost, osp.join(TEX_DIR, "tab-dcell-training-checkpoints.tex"))
    write_cost_tex(cost, osp.join(TEX_DIR, "tab-dcell-training-cost.tex"))

    panel_val_pearson(hist["eni948by"], ck, hist["x69dyevg"])
    panel_loss(hist["eni948by"], full, ck)
    panel_cost(cost)
    panel_stages(stages)


if __name__ == "__main__":
    main()
