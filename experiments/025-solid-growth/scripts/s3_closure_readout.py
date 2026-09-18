# experiments/025-solid-growth/scripts/s3_closure_readout.py
# [[experiments.025-solid-growth.scripts.s3_closure_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/s3_closure_readout
"""Read out the S3 closure training cell and its S0 baselines from W&B.

THE CELL. `cgt_s3_r_kl_fit_031` trains on the S3 closure subset (triples + every single +
the pair-in-triple doubles) on the 010 random split R, evaluated on the pinned triples,
joint fitness + interaction, 130 epochs, three seeds, `per_order_metrics: true`. Its
comparison arms are the 30-epoch S0 cells on the same split: `cgt_s0_r_kl_fit_014` (joint
fitness, weight 1.0), `cgt_s0_r_kl_fit_015` (weight 0.1), `cgt_s0_r_kl_ctrl_013` (no
fitness head) and the graph-regularization lambda-0 arm.

THE SCORING RULE, and it is not a max. The score of a run is the MEAN of
`val/gene_interaction/Pearson` over a FIXED epoch window, declared before the run is read:
epochs 10 to 29 for the 30-epoch S0 arms (the window
`disjoint_embedding_readout.py` uses), and for the 130-epoch S3 cell the same 10 to 29
window plus the 100 to 129 window. A window mean is reported only when EVERY epoch of the
window is present; a partial window reports its epoch count and no score, because a mean
over whichever epochs happened to finish is not the declared statistic. The table also
carries the value at the last logged epoch, and the MAX with its epoch. The max over noisy
epochs is an upward-biased order statistic whose bias grows with the number of epochs run:
it is recorded for checkpoint selection only and is never the score of an arm.

WHERE THE lambda-0 ARM LIVES. There is no `cgt_s0_r_kl_lambda0_*` config. The
graph-regularization ladder was run under the `cgt_s0_r_kl_ctrl_013` tag with the penalty
weight swept, so that one tag holds three arms; they are separated here by the config value
`model.graph_regularization.graph_reg_lambda` (0, 1e-3, 1e-2), which is what the run script
also mirrors into the `lambda_*` tag.

RANK 0. Each job is four DDP runs and only rank 0 logs `val/`. A rank-0 run is identified
by the presence of `val/gene_interaction/Pearson` in its history. A job whose first
validation epoch has not finished carries no such key anywhere, so for those seeds the run
carrying the rank-0-only probe series (`probe/grad_norm/point`) is reported instead, as a
partial row with zero validation epochs. Partial runs are labeled partial and report the
epochs they logged; a seed that has not started yet is reported as missing.

Outputs:

- results/s3_closure_readout.csv                    one row per (config, seed, run)
- results/s3_closure_readout_summary.json           per-arm mean/sd across seeds, paired S3 vs fit_014
- notes-tex/025-s3-closure/tables/t5-training-readout.tex
- $ASSET_IMAGES_DIR/025-solid-growth/s3_closure_readout.svg|png (plus a timestamped copy)

It also names the S3 runs and puts them in the `s3_fit_031` W&B group, following
`disjoint_embedding_wandb_view.py`; runs that already carry a readable name keep it, and
the abandoned partial of seed 1 is never touched.

    python experiments/025-solid-growth/scripts/s3_closure_readout.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "025-solid-growth")
TABLE_DIR = osp.join(osp.dirname(EXPERIMENT_ROOT), "notes-tex/025-s3-closure/tables")
ENTITY = "zhao-group"
PROJECT_NAME = "torchcell_025-solid-growth_equivariant_cell_graph_transformer"
PROJECT = f"{ENTITY}/{PROJECT_NAME}"

S3_TAG = "cgt_s3_r_kl_fit_031"
S3_GROUP = "s3_fit_031"
BASELINE_TAG = "cgt_s0_r_kl_fit_014"
# The abandoned partial of seed 1 (its rank 0 is kj03xx8y) and its three non-rank-0 runs.
EXCLUDED_RUN_IDS = {"kj03xx8y", "0kaadgdu", "bekoxpor", "ztfcxu37"}

# key, tag, graph_reg_lambda selector (None = do not filter), label, budget epochs
ARMS: list[tuple[str, str, float | None, str, int]] = [
    ("s3_fit_031", S3_TAG, None, "S3 closure, joint fitness 1.0", 130),
    ("s0_fit_014", BASELINE_TAG, None, "S0, joint fitness 1.0", 30),
    ("s0_fit_015", "cgt_s0_r_kl_fit_015", None, "S0, joint fitness 0.1", 30),
    ("s0_ctrl_013", "cgt_s0_r_kl_ctrl_013", 1e-3, "S0 control, no fitness head", 30),
    ("s0_lambda0", "cgt_s0_r_kl_ctrl_013", 0.0, "S0 control, graph reg lambda 0", 30),
]

# Windows: (column suffix, lo, hi). The 100-129 window only exists for the 130-epoch cell.
WINDOWS = [("ep10_29", 10, 29), ("ep100_129", 100, 129)]
PRIMARY_WINDOW = "ep10_29"

VAL_KEYS = (
    "val/gene_interaction/Pearson",
    "val/fitness/Pearson",
    "val/point_loss",
    "val/gene_interaction/order3/Pearson",
    "val/fitness/order3/Pearson",
)
PER_ORDER_KEYS = (
    "train/fitness/order1/Pearson",
    "train/fitness/order2/Pearson",
    "train/fitness/order3/Pearson",
    "train/gene_interaction/order2/Pearson",
    "train/gene_interaction/order3/Pearson",
)
# The record counts are logged under two spellings: the new runs split them by target,
# the older ones do not. Both are read and the per-target one wins when present.
N_RECORD_KEYS = tuple(
    f"train/n_records/{t}/order{k}" for t in ("fitness", "gene_interaction") for k in (1, 2, 3)
) + tuple(f"train/n_records/order{k}" for k in (1, 2, 3))
TRAIN_KEYS = ("train/gene_interaction/Pearson", "train/loss")
RANK0_PROBE_KEY = "probe/grad_norm/point"
KEYS = VAL_KEYS + PER_ORDER_KEYS + N_RECORD_KEYS + TRAIN_KEYS + (RANK0_PROBE_KEY,)

TS = timestamp()
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6,
        "axes.labelsize": 6,
        "axes.titlesize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "svg.fonttype": "none",
        "axes.linewidth": 0.5,
    }
)
ORANGE, RED, PURPLE, YELLOW, BLUE, GRAY = PLOT_PALETTE[:6]
# Red leads: this document's primary series color, then orange, purple, yellow, blue, gray.
SERIES = [RED, ORANGE, PURPLE, YELLOW, BLUE, GRAY]


# --------------------------------------------------------------------------- history


def per_epoch(run) -> dict[str, dict[int, float]]:
    """Every logged key by epoch. `scan_history` is the unsampled history."""
    out: dict[str, dict[int, float]] = {k: {} for k in KEYS}
    epochs: set[int] = set()
    for row in run.scan_history():
        e = row.get("epoch")
        if e is None:
            continue
        epochs.add(int(e))
        for k in KEYS:
            v = row.get(k)
            if v is not None:
                out[k][int(e)] = float(v)
    out["_epochs"] = {e: 1.0 for e in epochs}
    return out


def window_mean(d: dict[int, float], lo: int, hi: int) -> tuple[float | None, int]:
    """Mean over a CLOSED epoch window, and how many of its epochs are present.

    The mean is returned only when the window is complete, so a half-finished window
    never enters the table as though it were the declared statistic.
    """
    vals = [v for e, v in d.items() if lo <= e <= hi]
    n = len(vals)
    return (statistics.fmean(vals) if n == hi - lo + 1 else None), n


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(x, nd)


def _last(d: dict[int, float]) -> tuple[int | None, float | None]:
    if not d:
        return None, None
    e = max(d)
    return e, d[e]


# ------------------------------------------------------------------------ selection


def _lambda_of(run) -> float | None:
    gr = run.config.get("model", {}).get("graph_regularization", {})
    v = gr.get("graph_reg_lambda")
    return None if v is None else float(v)


def arm_runs(api, tag: str, lam: float | None) -> list:
    runs = [r for r in api.runs(PROJECT, filters={"tags": {"$in": [tag]}}) if r.id not in EXCLUDED_RUN_IDS]
    if lam is None:
        return runs
    return [r for r in runs if _lambda_of(r) == lam]


def pick_per_seed(runs: list) -> dict[int, tuple[object, dict]]:
    """One reported run per seed: the rank-0 run, else the most-advanced partial.

    Rank 0 is the run whose history carries `val/gene_interaction/Pearson`. The run
    summary carries it too once a validation epoch has finished, so the summary is read
    first and only the seeds it cannot resolve are scanned in full.
    """
    by_seed: dict[int, list] = {}
    for r in runs:
        by_seed.setdefault(int(r.config.get("seed", 42)), []).append(r)
    picked: dict[int, tuple[object, dict]] = {}
    for seed, seed_runs in sorted(by_seed.items()):
        summary_rank0 = [r for r in seed_runs if "val/gene_interaction/Pearson" in r.summary]
        scanned = [(r, per_epoch(r)) for r in (summary_rank0 or seed_runs)]
        with_val = [(r, h) for r, h in scanned if h["val/gene_interaction/Pearson"]]
        if with_val:
            picked[seed] = max(with_val, key=lambda rh: len(rh[1]["val/gene_interaction/Pearson"]))
            continue
        # No validation epoch anywhere for this seed: report the rank-0 run of the
        # in-flight job, identified by the rank-0-only gradient probe.
        probes = [(r, h) for r, h in scanned if h[RANK0_PROBE_KEY]]
        pool = probes or scanned
        if pool:
            picked[seed] = max(pool, key=lambda rh: len(rh[1]["_epochs"]))
    return picked


# ----------------------------------------------------------------------------- rows


def make_row(key: str, tag: str, label: str, budget: int, seed: int, run, h: dict) -> dict:
    val = h["val/gene_interaction/Pearson"]
    fit = h["val/fitness/Pearson"]
    max_epoch, max_val = (max(val, key=val.get), max(val.values())) if val else (None, None)
    last_val_epoch, last_val = _last(val)
    row = {
        "arm": key,
        "config": tag,
        "arm_label": label,
        "graph_reg_lambda": _lambda_of(run),
        "budget_epochs": budget,
        "seed": seed,
        "run_id": run.id,
        "run_url": run.url,
        "state": run.state,
        "epochs_logged": len(h["_epochs"]),
        "val_epochs_logged": len(val),
        "complete": len(val) >= budget,
        "partial": len(val) < budget,
        "last_epoch": last_val_epoch,
    }
    for suffix, lo, hi in WINDOWS:
        m, n = window_mean(val, lo, hi)
        row[f"val_gi_pearson_mean_{suffix}"] = _r(m)
        row[f"val_gi_pearson_n_{suffix}"] = n
        mf, _ = window_mean(fit, lo, hi)
        row[f"val_fitness_pearson_mean_{suffix}"] = _r(mf)
    row["val_gi_pearson_last"] = _r(last_val)
    row["val_fitness_pearson_last"] = _r(_last(fit)[1])
    row["val_point_loss_last"] = _r(_last(h["val/point_loss"])[1])
    row["val_gi_order3_pearson_last"] = _r(_last(h["val/gene_interaction/order3/Pearson"])[1])
    row["val_fitness_order3_pearson_last"] = _r(_last(h["val/fitness/order3/Pearson"])[1])
    # Upward-biased order statistic over the epochs run. Checkpoint selection only.
    row["val_gi_pearson_max_biased"] = _r(max_val)
    row["val_gi_pearson_max_epoch"] = max_epoch
    row["train_gi_pearson_last"] = _r(_last(h["train/gene_interaction/Pearson"])[1])
    for k in PER_ORDER_KEYS:
        col = k.replace("train/", "train_").replace("/", "_").lower() + "_last"
        row[col] = _r(_last(h[k])[1])
    for target in ("fitness", "gene_interaction"):
        for order in (1, 2, 3):
            split = h[f"train/n_records/{target}/order{order}"]
            legacy = h[f"train/n_records/order{order}"]
            e, v = _last(split if split else legacy)
            row[f"train_n_records_{target}_order{order}"] = None if v is None else int(v)
    return row


def collect(api) -> tuple[list[dict], dict[tuple[str, int], dict]]:
    """Rows for the table, and the per-epoch history the figure draws."""
    rows: list[dict] = []
    hist: dict[tuple[str, int], dict] = {}
    for key, tag, lam, label, budget in ARMS:
        runs = arm_runs(api, tag, lam)
        if not runs:
            print(f"{key} ({tag}): no runs")
            continue
        picked = pick_per_seed(runs)
        if not picked:
            print(f"{key} ({tag}): runs exist but none carry an epoch yet")
            continue
        for seed, (run, h) in sorted(picked.items()):
            rows.append(make_row(key, tag, label, budget, seed, run, h))
            hist[(key, seed)] = h
        missing = sorted({int(r.config.get("seed", 42)) for r in runs} - set(picked))
        if missing:
            print(f"{key}: seeds with no readable run: {missing}")
    return rows, hist


# -------------------------------------------------------------------------- summary


def summarize(df: pd.DataFrame) -> dict:
    col = f"val_gi_pearson_mean_{PRIMARY_WINDOW}"
    arms: dict[str, dict] = {}
    for key, tag, lam, label, budget in ARMS:
        sub = df[df["arm"] == key]
        scored = sub[sub[col].notna()]
        vals = scored[col].tolist()
        arms[key] = {
            "config": tag,
            "label": label,
            "graph_reg_lambda": lam,
            "budget_epochs": budget,
            "n_seeds_present": int(len(sub)),
            "n_seeds_scored": int(len(vals)),
            "window": PRIMARY_WINDOW,
            "window_mean_mean": _r(statistics.fmean(vals)) if vals else None,
            "window_mean_sd": _r(statistics.stdev(vals)) if len(vals) > 1 else None,
            "last_epoch_mean": _r(
                statistics.fmean(sub["val_gi_pearson_last"].dropna().tolist())
                if sub["val_gi_pearson_last"].notna().any()
                else None
            ),
            "seeds": {
                int(r["seed"]): {
                    "run_id": r["run_id"],
                    "run_url": r["run_url"],
                    "state": r["state"],
                    "epochs_logged": int(r["epochs_logged"]),
                    "val_epochs_logged": int(r["val_epochs_logged"]),
                    "partial": bool(r["partial"]),
                    "score": r[col] if pd.notna(r[col]) else None,
                    "val_gi_pearson_last": r["val_gi_pearson_last"]
                    if pd.notna(r["val_gi_pearson_last"])
                    else None,
                }
                for _, r in sub.iterrows()
            },
        }
        if key == "s3_fit_031":
            long = df[(df["arm"] == key) & df["val_gi_pearson_mean_ep100_129"].notna()]
            arms[key]["window_ep100_129_mean"] = (
                _r(statistics.fmean(long["val_gi_pearson_mean_ep100_129"].tolist())) if len(long) else None
            )
            arms[key]["n_seeds_scored_ep100_129"] = int(len(long))
    return {
        "generated": TS,
        "project": PROJECT,
        "scoring_rule": (
            "score = mean of val/gene_interaction/Pearson over the fixed epoch window "
            f"{PRIMARY_WINDOW} (epochs 10-29), reported only when every epoch of the window "
            "is present; the S3 cell also reports the 100-129 window. The max over epochs is "
            "an upward-biased order statistic recorded for checkpoint selection, never the score."
        ),
        "excluded_run_ids": sorted(EXCLUDED_RUN_IDS),
        "arms": arms,
        "paired_s3_vs_fit_014": paired(df),
    }


def paired(df: pd.DataFrame) -> dict:
    """S3 minus fit_014 on the seeds both arms scored, in the shared 10-29 window."""
    col = f"val_gi_pearson_mean_{PRIMARY_WINDOW}"
    a = {int(r["seed"]): r[col] for _, r in df[df["arm"] == "s3_fit_031"].iterrows() if pd.notna(r[col])}
    b = {int(r["seed"]): r[col] for _, r in df[df["arm"] == "s0_fit_014"].iterrows() if pd.notna(r[col])}
    shared = sorted(set(a) & set(b))
    if not shared:
        return {
            "window": PRIMARY_WINDOW,
            "n_paired_seeds": 0,
            "status": (
                "not computable: no seed has a complete 10-29 window in both arms "
                f"(S3 scored seeds {sorted(a)}, fit_014 scored seeds {sorted(b)})"
            ),
        }
    diffs = [a[s] - b[s] for s in shared]
    return {
        "window": PRIMARY_WINDOW,
        "n_paired_seeds": len(shared),
        "seeds": shared,
        "s3": {s: a[s] for s in shared},
        "fit_014": {s: b[s] for s in shared},
        "paired_diff": [_r(d) for d in diffs],
        "paired_diff_mean": _r(statistics.fmean(diffs)),
        "paired_diff_sd": _r(statistics.stdev(diffs)) if len(diffs) > 1 else None,
    }


# ---------------------------------------------------------------------------- table


def write_table(summary: dict) -> str:
    """One row per arm. The best window mean is bolded HERE, never by hand."""
    order = [k for k, *_ in ARMS]
    scored = {k: summary["arms"][k]["window_mean_mean"] for k in order}
    best = max((k for k in order if scored[k] is not None), key=lambda k: scored[k], default=None)
    lines = [
        "%% SOURCE: experiments/025-solid-growth/scripts/s3_closure_readout.py "
        "(results/s3_closure_readout_summary.json) -- GENERATED, do not edit",
        "",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "arm & seeds scored/run & budget & window mean $\\pm$ sd & last epoch \\\\",
        "\\midrule",
    ]
    dagger = False
    for k in order:
        a = summary["arms"][k]
        n = f"{a['n_seeds_scored']}/{a['n_seeds_present']}"
        m, sd = a["window_mean_mean"], a["window_mean_sd"]
        if m is None:
            cell = "--"
        else:
            cell = f"{m:.4f}" + (f" $\\pm$ {sd:.4f}" if sd is not None else "")
            if k == best:
                cell = f"\\textbf{{{cell}}}"
        last = "--" if a["last_epoch_mean"] is None else f"{a['last_epoch_mean']:.4f}"
        # A last-epoch value read off a run that stopped short of its budget is the value
        # at the epoch it reached, not an outcome; it is marked so the table cannot be
        # read as though the arm had finished.
        if a["last_epoch_mean"] is not None and a["n_seeds_scored"] < a["n_seeds_present"]:
            last += "$^{\\dagger}$"
            dagger = True
        lines.append(f"{a['label']} & {n} & {a['budget_epochs']} & {cell} & {last} \\\\")
    lines.append("\\bottomrule")
    if dagger:
        lines.append(
            "\\multicolumn{5}{l}{\\footnotesize $\\dagger$ includes a partial run: the value "
            "is at the epoch reached, not at the budget.} \\\\"
        )
    lines += ["\\end{tabular}", ""]
    os.makedirs(TABLE_DIR, exist_ok=True)
    path = osp.join(TABLE_DIR, "t5-training-readout.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# --------------------------------------------------------------------------- figure


def _box(ax) -> None:
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)


def _tenths(ax, xmax: int) -> None:
    """Pearson axes are read on the same 0 to 1 scale across panels, tenth gridlines."""
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0, xmax)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="both", lw=0.3, color="0.85", zorder=0)
    ax.set_axisbelow(True)


def _empty(ax, msg: str) -> None:
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center", fontsize=6, color=GRAY)


def _curve(ax, d: dict[int, float], **kw) -> None:
    if not d:
        return
    xs = sorted(d)
    ax.plot(xs, [d[x] for x in xs], **kw)


def make_figure(hist: dict[tuple[str, int], dict], img_dir: str, xmax: int) -> tuple[str, str]:
    w = mm_to_in(PANEL_WIDTHS_MM["full"])
    fig, axes = plt.subplots(1, 3, figsize=(w, mm_to_in(58)))
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.17, top=0.9, wspace=0.28)

    def val_panel(ax, key: str, title: str) -> None:
        drawn = False
        for i, seed in enumerate(sorted(s for a, s in hist if a == "s3_fit_031")):
            h = hist[("s3_fit_031", seed)]
            if h[key]:
                drawn = True
            _curve(ax, h[key], color=SERIES[i % 6], lw=1.0, label=f"S3 seed {seed}")
        for i, seed in enumerate(sorted(s for a, s in hist if a == "s0_fit_014")):
            h = hist[("s0_fit_014", seed)]
            if h[key]:
                drawn = True
            _curve(ax, h[key], color=SERIES[(i + 3) % 6], lw=0.5, label=f"fit_014 seed {seed}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(title)
        _tenths(ax, xmax)
        if not drawn:
            _empty(ax, "no validation epoch\nlogged yet")
        else:
            ax.legend(frameon=False, loc="lower right", handlelength=1.4, borderpad=0.2)
        _box(ax)

    val_panel(axes[0], "val/gene_interaction/Pearson", "val interaction Pearson")
    val_panel(axes[1], "val/fitness/Pearson", "val fitness Pearson")

    ax = axes[2]
    h = hist.get(("s3_fit_031", 1))
    styles = [
        ("train/fitness/order1/Pearson", RED, "-", "fitness order 1"),
        ("train/fitness/order2/Pearson", RED, "--", "fitness order 2"),
        ("train/fitness/order3/Pearson", RED, ":", "fitness order 3"),
        ("train/gene_interaction/order2/Pearson", ORANGE, "--", "interaction order 2"),
        ("train/gene_interaction/order3/Pearson", ORANGE, ":", "interaction order 3"),
    ]
    drawn = False
    if h is not None:
        for k, color, ls, lab in styles:
            if h[k]:
                drawn = True
            _curve(ax, h[k], color=color, ls=ls, lw=0.8, label=lab)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train Pearson, S3 seed 1")
    _tenths(ax, xmax)
    if drawn:
        ax.legend(frameon=False, loc="lower right", handlelength=1.8, borderpad=0.2)
    else:
        _empty(ax, "no per-order epoch\nlogged yet")
    _box(ax)

    for ax, letter in zip(axes, "abc"):
        panel_label(ax, letter)
    os.makedirs(img_dir, exist_ok=True)
    svg = osp.join(img_dir, "s3_closure_readout.svg")
    png = osp.join(img_dir, "s3_closure_readout.png")
    savefig_true_size_svg(fig, svg)
    fig.savefig(png, dpi=300)
    savefig_true_size_svg(fig, osp.join(img_dir, f"s3_closure_readout_{TS}.svg"))
    plt.close(fig)
    return svg, png


# ------------------------------------------------------------------- run bookkeeping

AUTONAME = re.compile(r"^(run_|[a-z]+-[a-z]+-\d+$)")


def label_s3_runs(api) -> int:
    """Group the S3 runs and give them readable names, as the view script does.

    A run that already carries a readable name keeps it: the seed-1 job was relaunched
    and its two generations are distinguished by name, so overwriting would merge them
    visually. The abandoned partial is excluded upstream and is never touched.
    """
    n = 0
    runs = arm_runs(api, S3_TAG, None)
    by_seed: dict[int, list] = {}
    for r in runs:
        by_seed.setdefault(int(r.config.get("seed", 42)), []).append(r)
    for seed, seed_runs in sorted(by_seed.items()):
        ranked = sorted(
            seed_runs,
            key=lambda r: (0 if "val/gene_interaction/Pearson" in r.summary else 1, r.id),
        )
        for i, run in enumerate(ranked):
            rank0 = "val/gene_interaction/Pearson" in run.summary
            changed = False
            if run.group != S3_GROUP:
                run.group = S3_GROUP
                changed = True
            if not run.name or AUTONAME.match(run.name):
                run.name = f"{S3_GROUP}_seed{seed}" + ("_rank0" if rank0 else f"_rank{i}")
                changed = True
            for k, v in (("arm", S3_GROUP), ("seed_", seed), ("split", "R"), ("rank0", rank0)):
                if run.config.get(k) != v:
                    run.config[k] = v
                    changed = True
            if changed:
                run.update()
                n += 1
    return n


# ----------------------------------------------------------------------------- main


def main() -> None:
    api = wandb.Api()
    print(f"labeled {label_s3_runs(api)} S3 runs (group {S3_GROUP})")
    rows, hist = collect(api)
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = osp.join(RESULTS_DIR, "s3_closure_readout.csv")
    df.to_csv(csv_path, index=False)

    summary = summarize(df)
    json_path = osp.join(RESULTS_DIR, "s3_closure_readout_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    table_path = write_table(summary)

    # The epoch axis spans what has actually been logged, not the nominal ceiling,
    # so a cell that is a few epochs in is still readable.
    xmax = max((max(h["_epochs"]) for h in hist.values() if h["_epochs"]), default=30)
    svg, png = make_figure(hist, IMG_DIR, max(int(xmax), 1))

    cols = [
        "arm",
        "seed",
        "run_id",
        "state",
        "budget_epochs",
        "epochs_logged",
        "val_epochs_logged",
        "partial",
        f"val_gi_pearson_mean_{PRIMARY_WINDOW}",
        "val_gi_pearson_mean_ep100_129",
        "val_gi_pearson_last",
        "val_fitness_pearson_last",
        "val_gi_pearson_max_biased",
        "val_gi_pearson_max_epoch",
    ]
    sub = df[cols]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for _, r in sub.iterrows():
        print("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.tolist()) + " |")
    print()
    for _, r in df.iterrows():
        print(r["arm"], "seed", r["seed"], r["run_url"])
    for p in (csv_path, json_path, table_path, svg, png):
        print("wrote", p)


if __name__ == "__main__":
    main()
