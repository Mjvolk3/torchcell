# experiments/010-kuzmin-tmi/scripts/dango_full_dataset_si.py
# [[experiments.010-kuzmin-tmi.scripts.dango_full_dataset_si]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/dango_full_dataset_si
"""DANGO on the full trigenic dataset: the runs behind the main-text baseline.

The main text (Fig. 2d) reports DANGO at Pearson r = 0.367 against the Cell Graph
Transformer (CGT) at 0.454. The three DANGO replicate values hardcoded in
``trigenic_tau_model_comparison.py`` (0.36759, 0.36708, 0.36637) are matched here to the
wandb runs that produced them. Every DANGO run trained on the full trigenic dataset lives
in ``zhao-group/torchcell_006-kuzmin2018-tmi_dango`` (script
``experiments/006-kuzmin-tmi/scripts/dango.py``, query
``experiments/006-kuzmin-tmi/queries/001_small_build.cql``); the CGT replicates live in
``zhao-group/torchcell_010-kuzmin-tmi_equivariant_cell_graph_transformer``.

Scoring rule: for each run the statistic is the MAXIMUM over epochs of the logged
``val/gene_interaction/Pearson`` (torchmetrics ``PearsonCorrCoef`` over the whole
validation split, synchronized across DDP ranks). A max over epochs is an upward-biased
order statistic, so the epoch count of each run is reported next to it. The trainer's
checkpoint callback monitors validation MSE, not Pearson, so the Pearson at the
minimum-MSE epoch is reported as well. A DDP job appears on wandb as one run per rank;
only the rank that logs the synchronized validation metrics is kept, and the number of
runs sharing a name gives the GPU count of the job.

Outputs
-------
results : experiments/010-kuzmin-tmi/results/dango_full_dataset_runs.csv         (one row per run)
          experiments/010-kuzmin-tmi/results/dango_full_dataset_history.csv      (val metrics per epoch)
          experiments/010-kuzmin-tmi/results/dango_full_dataset_summary.csv      (per group)
          experiments/010-kuzmin-tmi/results/dango_full_dataset_data_effect.csv  (per build x release)
panels  : $ASSET_IMAGES_DIR/010-kuzmin-tmi/dango_full_dataset_{curves,best,convergence,data_effect}.{svg,png}
table   : paper/nature-biotech/sections/tab-dango-full-runs.tex

The data-effect panel sets these runs beside the Kuzmin 2018-only replication of
``experiments/005-kuzmin2018-tmi`` (wandb ``zhao-group/torchcell_005-kuzmin2018-tmi_dango``,
frozen in ``experiments/005-kuzmin2018-tmi/results/dango_string_version_sweep.csv`` by
``dango_string_version_sweep.py``): the same implementation and scoring rule on 91,050 records
against 332,313. The two builds are split separately (same splitter and seed, different
membership), the 005 runs pool the three loss schedules (which do not separate there) at batch
32 on one GPU while the 006 runs use linear-to-uniform at batch 64 per GPU on 2 to 4 GPUs, so
this is a comparison of two training campaigns, not a controlled data-size ablation. It is the
only measured data-size comparison for DANGO.

Run from the repo root (``--from-csv`` re-renders offline from the frozen CSVs):
    python experiments/010-kuzmin-tmi/scripts/dango_full_dataset_si.py [--from-csv]
"""

import argparse
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in, savefig_true_size_svg

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
    }
)

load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

DANGO_PROJECT = "zhao-group/torchcell_006-kuzmin2018-tmi_dango"
CGT_PROJECT = "zhao-group/torchcell_010-kuzmin-tmi_equivariant_cell_graph_transformer"
# The three CGT training runs whose best-Pearson checkpoints were re-evaluated as the
# main-text replicates (evaluation runs leodrxht, 0psour3n, cvu2ryfw; tags
# `evaluation, full_dataset`; config `model.checkpoint_path` names the training run).
CGT_REPLICATES = ["lzs9pcj3", "c7671wgj", "yv4r30bi"]
# The three values hardcoded as "DANGO (repro best)" in trigenic_tau_model_comparison.py.
MAIN_TEXT_DANGO = [0.36759, 0.36708, 0.36637]

RESULTS_DIR = "experiments/010-kuzmin-tmi/results"
RUNS_CSV = osp.join(RESULTS_DIR, "dango_full_dataset_runs.csv")
HIST_CSV = osp.join(RESULTS_DIR, "dango_full_dataset_history.csv")
SUMMARY_CSV = osp.join(RESULTS_DIR, "dango_full_dataset_summary.csv")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")
TEX_PATH = "paper/nature-biotech/sections/tab-dango-full-runs.tex"

# The Kuzmin 2018-only replication (experiment 005), frozen by dango_string_version_sweep.py.
SWEEP_005_CSV = "experiments/005-kuzmin2018-tmi/results/dango_string_version_sweep.csv"
SPLIT_005_CSV = "experiments/005-kuzmin2018-tmi/results/dango_dataset_split.csv"
DATA_EFFECT_CSV = osp.join(RESULTS_DIR, "dango_full_dataset_data_effect.csv")
# Records in the experiment-006 build (TmiKuzmin2018, any perturbation type, + TmiKuzmin2020
# deletions), from notes/experiments.011-kuzmin-tmi.scripts.query-comparison-006-009-010-011.md;
# the 005 count is read from SPLIT_005_CSV.
RECORDS_006 = 332_313
BUILD_LABEL = {"005": "Kuzmin 2018", "006": "Kuzmin 2018 + Kuzmin 2020 deletions"}

METRIC = "val/gene_interaction/Pearson"
MSE = "val/gene_interaction/MSE"
MIN_EPOCHS = 100  # drops the smoke tests (1 to 6 epochs) and the one-epoch profile run
CONVERGENCE_FRACTION = 0.99  # epoch at which a run first reaches this fraction of its best
LEVEL = 0.35  # fixed validation Pearson level every DANGO run reaches; epoch of first crossing

GROUPS = ["9_1", "11_0", "12_0", "cgt"]
GROUP_LABEL = {
    "9_1": "DANGO, STRING v9.1",
    "11_0": "DANGO, STRING v11.0",
    "12_0": "DANGO, STRING v12.0",
    "cgt": "CGT (main text)",
}
GROUP_COLOR = {g: PLOT_PALETTE[i] for i, g in enumerate(GROUPS)}


def string_version(graphs: list[str]) -> str:
    versions = {g.split("_")[0].replace("string", "") + "_" + g.split("_")[1] for g in graphs}
    if len(versions) != 1:
        raise ValueError(f"run mixes STRING versions: {graphs}")
    return versions.pop()


def cluster_of(run_name: str) -> str:
    host = run_name.removeprefix("run_").split("_")[0]
    if "delta" in host:
        return "Delta (4 x A40)"
    if host.startswith("compute-"):
        return "IGB"
    return host


def val_history(run, keys: list[str]) -> pd.DataFrame:
    """Per-epoch rows that carry the synchronized validation metric."""
    hist = pd.DataFrame(list(run.scan_history(keys=keys, page_size=5000)))
    if hist.empty or METRIC not in hist:
        return pd.DataFrame()
    hist = hist.dropna(subset=[METRIC]).sort_values("_step")
    if MSE not in hist:
        hist[MSE] = np.nan
    return hist


def run_row(run, hist: pd.DataFrame, group: str, n_ranks: int) -> dict:
    cfg = run.config
    best = hist.loc[hist[METRIC].idxmax()]
    threshold = CONVERGENCE_FRACTION * best[METRIC]
    epoch_conv = int(hist.loc[hist[METRIC] >= threshold, "epoch"].min())
    row = {
        "group": group,
        "run_id": run.id,
        "run_name": run.name,
        "cluster": cluster_of(run.name),
        "state": run.state,
        "created_at": str(run.created_at),
        "n_ranks": n_ranks,
        "devices_cfg": cfg["trainer"]["devices"],
        "max_epochs": cfg["trainer"]["max_epochs"],
        "batch_size": cfg["data_module"]["batch_size"],
        "epochs_logged": int(len(hist)),
        "last_epoch": int(hist["epoch"].max()),
        "wall_h": float(hist["_runtime"].max()) / 3600.0,
        "best_val_pearson": float(best[METRIC]),
        "best_epoch": int(best["epoch"]),
        "epoch_to_99pct": epoch_conv,
        "hours_to_99pct": float(hist.loc[hist["epoch"] == epoch_conv, "_runtime"].iloc[0]) / 3600.0,
        "final_val_pearson": float(hist[METRIC].iloc[-1]),
    }
    if MSE in hist and hist[MSE].notna().any():
        best_mse = hist.loc[hist[MSE].idxmin()]
        row["best_mse_epoch"] = int(best_mse["epoch"])
        row["val_pearson_at_best_mse"] = float(best_mse[METRIC])
    if group == "cgt":
        row.update({"lr": cfg["regression_task"]["optimizer"]["lr"], "loss_schedule": "", "transition_epoch": ""})
    else:
        row.update(
            {
                "lr": cfg["regression_task"]["optimizer"]["lr"],
                "loss_schedule": cfg["regression_task"]["loss_scheduler"]["type"],
                "transition_epoch": cfg["regression_task"]["loss_scheduler"]["transition_epoch"],
                "hidden_channels": cfg["model"]["hidden_channels"],
                "num_heads": cfg["model"]["num_heads"],
            }
        )
    return row


def pull() -> tuple[pd.DataFrame, pd.DataFrame]:
    import wandb

    api = wandb.Api(timeout=300)
    keys = ["epoch", METRIC, MSE, "_runtime", "_step"]
    rows, hists = [], []

    dango_runs = list(api.runs(DANGO_PROJECT, per_page=200))
    ranks_per_name = pd.Series([r.name for r in dango_runs]).value_counts()
    for run in dango_runs:
        if METRIC not in run.summary._json_dict:
            continue  # non-zero DDP ranks log only the per-rank sample statistics
        hist = val_history(run, keys)
        if hist.empty or len(hist) < MIN_EPOCHS:
            print(f"  skip {run.id} ({run.state}, {len(hist)} validation epochs)")
            continue
        group = string_version(run.config["cell_dataset"]["graphs"])
        rows.append(run_row(run, hist, group, int(ranks_per_name[run.name])))
        hist = hist.assign(run_id=run.id, group=group)
        hists.append(hist[["run_id", "group", "epoch", METRIC, MSE, "_runtime"]])
        print(f"  {run.id} {group:5s} best={rows[-1]['best_val_pearson']:.5f} @ {rows[-1]['best_epoch']} / {rows[-1]['epochs_logged']}")

    for rid in CGT_REPLICATES:
        run = api.run(f"{CGT_PROJECT}/{rid}")
        hist = val_history(run, keys)
        rows.append(run_row(run, hist, "cgt", int(run.config["trainer"]["devices"])))
        hist = hist.assign(run_id=run.id, group="cgt")
        hists.append(hist[["run_id", "group", "epoch", METRIC, MSE, "_runtime"]])
        print(f"  {run.id} cgt   best={rows[-1]['best_val_pearson']:.5f} @ {rows[-1]['best_epoch']} / {rows[-1]['epochs_logged']}")

    runs = pd.DataFrame(rows)
    runs["main_text"] = runs["best_val_pearson"].round(5).isin(MAIN_TEXT_DANGO) | (runs["group"] == "cgt")
    runs["group"] = pd.Categorical(runs["group"], GROUPS, ordered=True)
    runs = runs.sort_values(["group", "created_at"]).reset_index(drop=True)
    history = pd.concat(hists, ignore_index=True).rename(
        columns={METRIC: "val_pearson", MSE: "val_mse", "_runtime": "runtime_s"}
    )
    return runs, history


def add_level_epochs(runs: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """First epoch at which each run's validation Pearson reaches LEVEL (from the history)."""
    first = {}
    for rid, hh in history.groupby("run_id"):
        hit = hh.loc[hh["val_pearson"] >= LEVEL, "epoch"]
        first[rid] = int(hit.min()) if len(hit) else -1
    runs = runs.copy()
    runs["epoch_to_level"] = runs["run_id"].map(first)
    if (runs["epoch_to_level"] < 0).any():
        raise ValueError(f"a run never reaches r >= {LEVEL}: {runs.loc[runs['epoch_to_level'] < 0, 'run_id'].tolist()}")
    return runs


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    def agg(frame: pd.DataFrame) -> pd.Series:
        x = frame["best_val_pearson"]
        sd = x.std(ddof=1) if len(x) > 1 else np.nan
        return pd.Series(
            {
                "n": len(x),
                "mean": x.mean(),
                "sd": sd,
                "sem": sd / np.sqrt(len(x)) if len(x) > 1 else np.nan,
                "min": x.min(),
                "max": x.max(),
                "mean_best_epoch": frame["best_epoch"].mean(),
                "min_best_epoch": frame["best_epoch"].min(),
                "max_best_epoch": frame["best_epoch"].max(),
                "mean_epoch_to_level": frame["epoch_to_level"].mean(),
                "min_epoch_to_level": frame["epoch_to_level"].min(),
                "max_epoch_to_level": frame["epoch_to_level"].max(),
                "mean_epoch_to_99pct": frame["epoch_to_99pct"].mean(),
                "mean_epochs_logged": frame["epochs_logged"].mean(),
            }
        )

    s = runs.groupby("group", observed=True).apply(agg, include_groups=False).reset_index()
    main = runs[runs["main_text"] & (runs["group"] != "cgt")]
    s = pd.concat([s, agg(main).to_frame().T.assign(group="dango_main_text")], ignore_index=True)
    return s


def style_axes(ax):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)
        sp.set_color("black")
    ax.tick_params(length=2, width=0.5)


def panel_curves(runs: pd.DataFrame, history: pd.DataFrame):
    """Validation Pearson against epoch, one thin line per run, colored by group."""
    w, h = PANEL_WIDTHS_MM["half"], 52.0
    fig, ax = plt.subplots(figsize=(mm_to_in(w), mm_to_in(h)))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.17)
    for g in GROUPS:
        for i, rid in enumerate(runs.loc[runs["group"] == g, "run_id"]):
            hh = history[history["run_id"] == rid].sort_values("epoch")
            ax.plot(hh["epoch"] + 1, hh["val_pearson"], color=GROUP_COLOR[g], linewidth=0.5,
                    label=GROUP_LABEL[g] if i == 0 else None)
    ax.set_xscale("log")
    ax.set_xlim(1, 1100)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Epoch (log scale)")
    ax.set_ylabel("Validation Pearson r")
    ax.set_ylim(0, 0.5)
    ax.yaxis.set_major_locator(FixedLocator([0, 0.2, 0.4]))
    ax.yaxis.set_minor_locator(FixedLocator([0.1, 0.3, 0.5]))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="both", color="#CACACA", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", handlelength=1.2)
    style_axes(ax)
    save(fig, "dango_full_dataset_curves")


def panel_best(runs: pd.DataFrame, summary: pd.DataFrame):
    """Best validation Pearson per run by group: bar = mean, whisker = SEM, marker = run
    (filled = enters the main-text figure). The y-axis is zoomed to 0.3--0.5 so the
    between-run spread of a few thousandths is visible."""
    w, h = PANEL_WIDTHS_MM["half"], 52.0
    fig, ax = plt.subplots(figsize=(mm_to_in(w), mm_to_in(h)))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.17)
    x = np.arange(len(GROUPS))
    rng = np.random.default_rng(0)
    for i, g in enumerate(GROUPS):
        row = summary[summary["group"] == g].iloc[0]
        ax.bar(i, row["mean"], 0.6, color=GROUP_COLOR[g], edgecolor="black", linewidth=0.4)
        if row["n"] > 1:
            ax.errorbar(i, row["mean"], yerr=row["sem"], fmt="none", ecolor="black",
                        elinewidth=0.6, capsize=2, capthick=0.6)
        sub = runs[runs["group"] == g]
        jitter = rng.uniform(-0.12, 0.12, len(sub))
        for j, (_, r) in enumerate(sub.iterrows()):
            ax.scatter(i + jitter[j], r["best_val_pearson"], s=7,
                       facecolor="black" if r["main_text"] else "white",
                       edgecolor="black", linewidth=0.4, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(["STRING v9.1", "STRING v11.0", "STRING v12.0", "CGT"])
    ax.set_xlabel("DANGO by STRING release, and the CGT")
    ax.set_ylabel("Best validation Pearson r")
    ax.set_ylim(0.3, 0.5)
    ax.yaxis.set_major_locator(FixedLocator([0.3, 0.35, 0.4, 0.45, 0.5]))
    ax.yaxis.set_minor_locator(FixedLocator(np.arange(0.31, 0.5, 0.01)))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="major", color="#CACACA", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.scatter([], [], s=7, facecolor="black", edgecolor="black", linewidth=0.4, label="in Fig. 2d")
    ax.scatter([], [], s=7, facecolor="white", edgecolor="black", linewidth=0.4, label="other replicate")
    ax.legend(frameon=False, loc="upper left", handlelength=1.0)
    style_axes(ax)
    save(fig, "dango_full_dataset_best")


def panel_convergence(runs: pd.DataFrame):
    """Two epochs per DANGO run, by STRING release, on a log axis: the epoch at which the
    run first reaches validation Pearson r >= LEVEL (filled) and the epoch of its maximum
    (open). Marker shape gives the GPU count of the job. Third width: the message is that
    the release moves the epoch of the maximum, not the epoch of the rise."""
    w, h = PANEL_WIDTHS_MM["third"], 52.0
    fig, ax = plt.subplots(figsize=(mm_to_in(w), mm_to_in(h)))
    fig.subplots_adjust(left=0.2, right=0.97, top=0.97, bottom=0.17)
    dango = [g for g in GROUPS if g != "cgt"]
    rng = np.random.default_rng(1)
    for i, g in enumerate(dango):
        sub = runs[runs["group"] == g]
        jitter = rng.uniform(-0.1, 0.1, len(sub))
        for j, (_, r) in enumerate(sub.iterrows()):
            marker = "o" if r["n_ranks"] == 4 else "s"
            ax.scatter(i - 0.18 + jitter[j], r["epoch_to_level"], s=9, marker=marker,
                       facecolor=GROUP_COLOR[g], edgecolor="black", linewidth=0.4, zorder=5)
            ax.scatter(i + 0.18 + jitter[j], r["best_epoch"], s=9, marker=marker,
                       facecolor="white", edgecolor=GROUP_COLOR[g], linewidth=0.6, zorder=5)
    ax.set_yscale("log")
    ax.set_ylim(5, 1500)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(np.arange(len(dango)))
    ax.set_xlim(-0.5, len(dango) - 0.5)
    ax.set_xticklabels(["v9.1", "v11.0", "v12.0"])
    ax.set_xlabel("STRING release")
    ax.set_ylabel("Epoch (log scale)")
    ax.grid(axis="y", which="major", color="#CACACA", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.scatter([], [], s=9, marker="o", facecolor="#666666", edgecolor="black", linewidth=0.4,
               label=f"first epoch with r \u2265 {LEVEL:.2f}")
    ax.scatter([], [], s=9, marker="o", facecolor="white", edgecolor="#666666", linewidth=0.6,
               label="epoch of best r")
    ax.scatter([], [], s=9, marker="o", facecolor="white", edgecolor="black", linewidth=0.4, label="4 GPUs")
    ax.scatter([], [], s=9, marker="s", facecolor="white", edgecolor="black", linewidth=0.4, label="2 GPUs")
    ax.legend(frameon=False, loc="upper left", ncol=1, handlelength=1.0, handletextpad=0.4,
              labelspacing=0.25, borderaxespad=0.3)
    style_axes(ax)
    save(fig, "dango_full_dataset_convergence")


def data_effect_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per DANGO run on either build: the 005 runs (Kuzmin 2018 only, frozen by
    dango_string_version_sweep.py) and the 006 runs of this note, with build, record count,
    STRING release and best validation Pearson (the same max-over-epochs rule in both)."""
    r005 = pd.read_csv(SWEEP_005_CSV)
    records_005 = int(pd.read_csv(SPLIT_005_CSV)["records"].item())
    a = pd.DataFrame(
        {"build": "005", "records": records_005, "string_version": r005["string_version"],
         "run_id": r005["run_id"], "best_val_pearson": r005["best_val_pearson"]}
    )
    r006 = runs[runs["group"] != "cgt"]
    b = pd.DataFrame(
        {"build": "006", "records": RECORDS_006, "string_version": r006["group"].astype(str),
         "run_id": r006["run_id"], "best_val_pearson": r006["best_val_pearson"]}
    )
    return pd.concat([a, b], ignore_index=True)


def data_effect_table(per_run: pd.DataFrame) -> pd.DataFrame:
    """Per build x STRING release: n, mean, SD, SEM, range and the run ids, so the panel and
    the prose numbers trace to the frozen run tables."""
    rows = []
    for (build, v), x in per_run.groupby(["build", "string_version"], sort=False):
        vals = x["best_val_pearson"]
        sd = vals.std(ddof=1) if len(vals) > 1 else np.nan
        rows.append(
            {
                "build": build,
                "records": int(x["records"].iloc[0]),
                "string_version": v,
                "n": len(vals),
                "mean": vals.mean(),
                "sd": sd,
                "sem": sd / np.sqrt(len(vals)) if len(vals) > 1 else np.nan,
                "min": vals.min(),
                "max": vals.max(),
                "run_ids": ";".join(x["run_id"]),
            }
        )
    order = {"9_1": 0, "11_0": 1, "12_0": 2}
    df = pd.DataFrame(rows)
    return df.sort_values(["build", "string_version"], key=lambda c: c.map(order) if c.name == "string_version" else c).reset_index(drop=True)


def panel_data_effect(per_run: pd.DataFrame, effect: pd.DataFrame):
    """Wide panel: best validation Pearson (max over epochs) on the Kuzmin 2018-only build
    against the experiment-006 build, grouped by build with one bar per STRING release;
    bar = mean over runs, whisker = SEM where n > 1, open circles = runs. The y-axis is
    zoomed to 0.30 to 0.45, as in the best-per-run panel, so the SEM whiskers are visible."""
    w, h = PANEL_WIDTHS_MM["wide"], 52.0
    fig, ax = plt.subplots(figsize=(mm_to_in(w), mm_to_in(h)))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.97, bottom=0.2)
    builds = ["005", "006"]
    versions = ["9_1", "11_0", "12_0"]
    labels = {"9_1": "STRING v9.1", "11_0": "STRING v11.0", "12_0": "STRING v12.0"}
    bw = 0.22
    rng = np.random.default_rng(0)
    for i, v in enumerate(versions):
        for j, build in enumerate(builds):
            row = effect[(effect["build"] == build) & (effect["string_version"] == v)].iloc[0]
            xi = j + (i - 1) * bw
            ax.bar(xi, row["mean"], bw, color=GROUP_COLOR[v], edgecolor="black", linewidth=0.4,
                   label=labels[v] if j == 0 else None)
            if row["n"] > 1:
                ax.errorbar(xi, row["mean"], yerr=row["sem"], fmt="none", ecolor="black",
                            elinewidth=0.6, capsize=1.5, capthick=0.6)
            vals = per_run[(per_run["build"] == build) & (per_run["string_version"] == v)]["best_val_pearson"]
            ax.scatter(xi + rng.uniform(-0.05, 0.05, len(vals)), vals, s=6, facecolor="white",
                       edgecolor="black", linewidth=0.4, zorder=5)
    ax.set_xticks(np.arange(len(builds)))
    ax.set_xticklabels([f"{BUILD_LABEL[b]}\n{int(effect[effect['build'] == b]['records'].iloc[0]):,} records" for b in builds])
    ax.set_xlim(-0.6, len(builds) - 0.4)
    ax.set_xlabel("Trigenic dataset build")
    ax.set_ylabel("Best validation Pearson r")
    ax.set_ylim(0.3, 0.45)
    ax.yaxis.set_major_locator(FixedLocator([0.3, 0.35, 0.4, 0.45]))
    ax.yaxis.set_minor_locator(FixedLocator(np.arange(0.31, 0.45, 0.01)))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="major", color="#CACACA", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", ncol=3, handlelength=1.0, columnspacing=1.0,
              handletextpad=0.4)
    style_axes(ax)
    save(fig, "dango_full_dataset_data_effect")


def save(fig, stem: str):
    os.makedirs(IMG_DIR, exist_ok=True)
    svg = osp.join(IMG_DIR, f"{stem}.svg")
    savefig_true_size_svg(fig, svg)
    fig.savefig(osp.join(IMG_DIR, f"{stem}.png"), dpi=300)
    plt.close(fig)
    print(f"  wrote {svg}")


def write_tex(runs: pd.DataFrame):
    src = "experiments/010-kuzmin-tmi/scripts/dango_full_dataset_si.py"
    lines = [
        f"%% SOURCE: {src} -- AUTO-GENERATED from wandb {DANGO_PROJECT} and {CGT_PROJECT}; do not hand-edit.",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\caption{Every DANGO run on the full trigenic dataset, and the three CGT replicates of",
        "Fig.~\\ref{fig:ggi}d. Best is the maximum over epochs of validation Pearson $r$ (an",
        "upward-biased order statistic; the epoch it occurs at follows in parentheses). At min",
        "MSE is the validation Pearson $r$ at the epoch of minimum validation MSE, the checkpoint",
        "the trainer keeps; it is within 0.0023 of Best for every DANGO run.",
        f"$r \\ge {LEVEL:.2f}$ is the first epoch at which the run reaches that",
        "validation Pearson. Epochs is the number of validation epochs logged; a 4-GPU job on a",
        "Delta A40 node stops at the 48 h queue limit, and the 2-GPU jobs on the IGB cluster ran",
        "to 1,000 epochs. Every DANGO run used hidden width 64, four heads, the linear-to-uniform",
        "loss schedule (transition at epoch 10), AdamW at learning rate $10^{-5}$, and a per-GPU",
        "batch of 64, on the experiment-006 build; the CGT replicates trained on the",
        "experiment-010 build. Runs marked $\\bullet$ are the three DANGO values averaged in",
        "Fig.~\\ref{fig:ggi}d.}",
        "\\label{tab:dango-full-runs}",
        "\\begin{tabular}{@{}l l l r r r l l r l@{}}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{STRING} & \\textbf{Run} & \\textbf{GPUs} & \\textbf{Epochs} & \\textbf{Wall (h)} & \\textbf{Best $r$ (epoch)} & \\textbf{At min MSE} & $\\boldsymbol{r \\ge " + f"{LEVEL:.2f}" + "}$ & \\\\",
        "\\midrule",
    ]
    version_label = {"9_1": "v9.1", "11_0": "v11.0", "12_0": "v12.0", "cgt": "v12.0"}
    for _, r in runs.iterrows():
        model = "CGT" if r["group"] == "cgt" else "DANGO"
        mark = "$\\bullet$" if (r["main_text"] and r["group"] != "cgt") else ""
        at_mse = f"{r['val_pearson_at_best_mse']:.3f} ({int(r['best_mse_epoch'])})" if pd.notna(r.get("val_pearson_at_best_mse")) else "--"
        lines.append(
            f"{model} & {version_label[r['group']]} & \\texttt{{{r['run_id']}}} & {int(r['n_ranks'])} & "
            f"{int(r['epochs_logged'])} & {r['wall_h']:.1f} & {r['best_val_pearson']:.4f} ({int(r['best_epoch'])}) & "
            f"{at_mse} & {int(r['epoch_to_level'])} & {mark}\\\\"
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
        runs = pd.read_csv(RUNS_CSV)
        runs["group"] = pd.Categorical(runs["group"], GROUPS, ordered=True)
        history = pd.read_csv(HIST_CSV)
    else:
        runs, history = pull()
        runs.to_csv(RUNS_CSV, index=False)
        history.to_csv(HIST_CSV, index=False)
        print(f"  wrote {RUNS_CSV} ({len(runs)} runs), {HIST_CSV} ({len(history)} epochs)")
    runs = add_level_epochs(runs, history)
    summary = summarize(runs)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(summary.to_string(index=False))
    print(runs[["group", "run_id", "cluster", "n_ranks", "epochs_logged", "wall_h", "best_val_pearson",
                "best_epoch", "epoch_to_level", "epoch_to_99pct", "main_text"]].to_string(index=False))
    per_run = data_effect_runs(runs)
    effect = data_effect_table(per_run)
    effect.to_csv(DATA_EFFECT_CSV, index=False)
    print(effect.to_string(index=False))
    panel_curves(runs, history)
    panel_best(runs, summary)
    panel_convergence(runs)
    panel_data_effect(per_run, effect)
    write_tex(runs)


if __name__ == "__main__":
    main()
