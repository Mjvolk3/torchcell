# experiments/025-solid-growth/scripts/graph_reg_sweep_readout.py
# [[experiments.025-solid-growth.scripts.graph_reg_sweep_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/graph_reg_sweep_readout
"""Read the graph-regularization sweep (Delta, 025 S0, random split) out of W&B.

The sweep is ``cgt_s0_r_kl_ctrl_013`` with one key changed per arm: the soft KL prior at
lambda 0 (no penalty), 1e-5, 1e-4, 1e-3 (the control itself), 1e-2, 1e-1 and 1; the hard
attention mask on layer 1 (``cgt_s0_r_mask_028``); and the KL at 1e-3 toward degree-matched
random rewirings of the nine graphs (``cgt_s0_r_kl_rand_031``). Three seeds per arm,
30 epochs, constant rate, four A40s per run (``delta_submit_sweep.sh``).

Runs are selected by the tags the training script attaches, never by hand-typed ids, and
classified by their own config: ``attention_mask.enabled`` is the mask arm,
``random_graph.enabled`` the random-graph arm, otherwise ``graph_reg_lambda`` names the
ladder point. Only Delta rank-0 runs (the run of a job that carries the validation
history) in state ``finished`` or ``running`` are kept; a failed or crashed run is a
launch that died (OOM on the old kernel path, a rank timeout) and never a measurement.
A running run is kept and marked partial.

Per run and per epoch: validation Pearson, validation and training point loss,
training Pearson, the graph penalty as logged (lambda-weighted) and divided by lambda
(the divergence of layer-1 attention from the nine normalized adjacencies, comparable
across the ladder), edge recall at degree averaged over the nine regularized heads at
the diagnostic epochs, and the gradient-probe norms of the point loss and the penalty on
the first batch of epochs 0, 1, 2, 5, 10 and 20.

Two readings per run, reported together: the value at epoch 29 (the protocol's fixed
reading) and the max over epochs (an upward-biased order statistic). Arm rows carry the
mean and sd over seeds of each.

Writes, all under experiments/025-solid-growth/results/:
  graph_reg_sweep_runs.csv       one row per run
  graph_reg_sweep_history.csv    long table: run_id, arm, seed, epoch, key, value
  graph_reg_sweep_summary.json   arm-level means, paired differences against no penalty
                                 by seed (both readings, paired t), the run list, the pull time
and the LaTeX tables of notes-tex/025-graph-reg-sweep/tables/ (t1-arms with an arrow where
every seed of an arm is above or below no penalty, t2-runs, t3-paired).

    python experiments/025-solid-growth/scripts/graph_reg_sweep_readout.py
    python experiments/025-solid-growth/scripts/graph_reg_sweep_readout.py --offline
        (rebuild the summary and tables from the CSVs without touching W&B)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import os.path as osp
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results")
TABLES_DIR = osp.join(
    osp.dirname(EXPERIMENT_ROOT), "notes-tex", "025-graph-reg-sweep", "tables"
)
PROJECT = "zhao-group/torchcell_025-solid-growth_equivariant_cell_graph_transformer"
SELECT_TAGS = ["graph_reg_sweep", "cgt_s0_r_kl_ctrl_013", "graph_reg_round3"]
BUDGET = 30
FIXED_EPOCH = BUDGET - 1
WINDOW = (10, 29)

# Epoch-level scalars (one value per epoch, logged at epoch end).
EPOCH_KEYS = (
    "val/gene_interaction/Pearson",
    "val/gene_interaction/MSE",
    "val/point_loss",
    "val/graph_reg_loss",
    "train/gene_interaction/Pearson",
    "train/cuda_peak_allocated_gb",
)
# Step-level scalars: averaged over the steps of an epoch.
STEP_KEYS = ("train/point_loss", "train/graph_reg_loss", "train/loss")
PROBE_KEYS = (
    "probe/grad_norm/point",
    "probe/grad_norm/dist",
    "probe/grad_norm/graph_reg",
    "probe/grad_norm/total",
    "probe/grad_ratio/graph_reg_to_point",
)
GRAPHS = (
    "physical_L1_H0",
    "regulatory_L1_H1",
    "tflink_L1_H2",
    "string12_0_neighborhood_L1_H3",
    "string12_0_fusion_L1_H4",
    "string12_0_cooccurence_L1_H5",
    "string12_0_coexpression_L1_H6",
    "string12_0_experimental_L1_H7",
    "string12_0_database_L1_H8",
)
EDGE_KEYS = tuple(f"val_edge_recovery/{g}/recall_at_deg" for g in GRAPHS) + tuple(
    f"val_edge_recovery/{g}/precision_k32" for g in GRAPHS
)

# Display order of the arms: the ladder from no penalty up, then the two controls
# (round 1), then the round-1b, 2 and 3 arms of graph_reg_round2_plan.py. An arm's name
# is its mechanism (`kl_<lambda>`, `mask`, `random_<lambda>`) followed by every
# departure from the round-1 setting: `2hop`/`3hop` reach, `sym` (KL target made
# undirected) or `dir` (mask kept directed), `L<a>-<b>` for the regularized layers,
# `emb` for the composite sequence embedding, `w360` for hidden width 360, `60ep` for
# the 60-epoch budget. Round-1 arms carry none, so their names are unchanged.
ROUND1_ARMS = [
    "kl_0",
    "kl_1e-05",
    "kl_0.0001",
    "kl_0.001",
    "kl_0.01",
    "kl_0.1",
    "kl_1",
    "mask",
    "random_0.001",
]
ROUND2_ARMS = [
    # 1b: the ladder extended and the random control at the separating lambda
    "kl_10",
    "kl_100",
    "random_0.1",
    "random_1",
    # 2: reach and direction, on the same targets for both mechanisms
    "mask_dir",
    "mask_2hop_dir",
    "mask_3hop_dir",
    "kl_1_2hop",
    "kl_1_3hop",
    "kl_1_sym",
    # 3: placement
    "mask_L1-2",
    "mask_L3-4",
    "mask_L1-4",
    "kl_1_L1-2",
    "kl_1_L3-4",
    "kl_1_L1-4",
    # 3: budget
    "kl_0_60ep",
    "mask_60ep",
    "kl_1_60ep",
    # 3: representation and width
    "kl_0_emb",
    "kl_1_emb",
    "kl_0_w360",
    "kl_1_w360",
    "kl_0_emb_w360",
    "kl_1_emb_w360",
]
ARM_ORDER = ROUND1_ARMS + ROUND2_ARMS
ARM_LABEL = {
    "kl_0": "no penalty ($\\lambda = 0$)",
    "kl_1e-05": "KL $\\lambda = 10^{-5}$",
    "kl_0.0001": "KL $\\lambda = 10^{-4}$",
    "kl_0.001": "KL $\\lambda = 10^{-3}$ (control)",
    "kl_0.01": "KL $\\lambda = 10^{-2}$",
    "kl_0.1": "KL $\\lambda = 10^{-1}$",
    "kl_1": "KL $\\lambda = 1$",
    "mask": "hard mask, layer 1",
    "random_0.001": "KL $\\lambda = 10^{-3}$, random graphs",
    "kl_10": "KL $\\lambda = 10$",
    "kl_100": "KL $\\lambda = 100$",
    "random_0.1": "KL $\\lambda = 10^{-1}$, random graphs",
    "random_1": "KL $\\lambda = 1$, random graphs",
    "mask_dir": "hard mask, directed",
    "mask_2hop_dir": "hard mask, 2 hops, directed",
    "mask_3hop_dir": "hard mask, 3 hops, directed",
    "kl_1_2hop": "KL $\\lambda = 1$, 2-hop target",
    "kl_1_3hop": "KL $\\lambda = 1$, 3-hop target",
    "kl_1_sym": "KL $\\lambda = 1$, symmetric target",
    "mask_L1-2": "hard mask, layers 1--2",
    "mask_L3-4": "hard mask, layers 3--4",
    "mask_L1-4": "hard mask, layers 1--4",
    "kl_1_L1-2": "KL $\\lambda = 1$, layers 1--2",
    "kl_1_L3-4": "KL $\\lambda = 1$, layers 3--4",
    "kl_1_L1-4": "KL $\\lambda = 1$, layers 1--4",
    "kl_0_60ep": "no penalty, 60 epochs",
    "mask_60ep": "hard mask, 60 epochs",
    "kl_1_60ep": "KL $\\lambda = 1$, 60 epochs",
    "kl_0_emb": "no penalty, composite embedding",
    "kl_1_emb": "KL $\\lambda = 1$, composite embedding",
    "kl_0_w360": "no penalty, width 360",
    "kl_1_w360": "KL $\\lambda = 1$, width 360",
    "kl_0_emb_w360": "no penalty, composite, width 360",
    "kl_1_emb_w360": "KL $\\lambda = 1$, composite, width 360",
}
assert set(ARM_LABEL) == set(ARM_ORDER)


class RunRow(BaseModel):
    """One rank-0 run of the sweep."""

    run_id: str
    run_url: str
    run_name: str
    arm: str
    graph_reg_lambda: float
    mask: bool
    random_graph: bool
    seed: int
    state: str
    host: str
    slurm_job: str
    epochs_logged: int
    complete: bool
    val_pearson_fixed: float | None
    val_pearson_max: float
    val_pearson_max_epoch: int
    val_pearson_window_mean: float | None
    val_pearson_window_sd: float | None
    val_point_loss_fixed: float | None
    train_pearson_fixed: float | None
    train_point_loss_fixed: float | None
    train_graph_reg_fixed: float | None
    train_divergence_fixed: float | None
    val_divergence_fixed: float | None
    edge_recall_fixed: float | None
    edge_precision_k32_fixed: float | None
    probe_point_epoch0: float | None
    probe_graph_reg_epoch0: float | None
    probe_ratio_epoch0: float | None
    probe_ratio_epoch20: float | None
    cuda_peak_gb_max: float | None


def _layer_tag(layers: list[int]) -> str:
    """`L1-2` for a contiguous block, `L1+3` otherwise; empty for the round-1 layer 1."""
    if layers == [1]:
        return ""
    if layers == list(range(layers[0], layers[-1] + 1)):
        return f"L{layers[0]}-{layers[-1]}"
    return "L" + "+".join(str(x) for x in layers)


def classify(cfg: dict[str, Any]) -> tuple[str, float, bool, bool]:
    """Arm name from the run's own config (see ROUND2_ARMS for the naming)."""
    model = cfg.get("model", {})
    reg = model.get("graph_regularization") or {}
    lam = float(reg.get("graph_reg_lambda", 0.0))
    mask_cfg = model.get("attention_mask") or {}
    mask = bool(mask_cfg.get("enabled", False))
    rand = bool((model.get("random_graph") or {}).get("enabled", False))
    parts: list[str] = []
    if mask:
        assert lam == 0.0, "a mask arm with a KL lambda is not in the design"
        base = "mask"
        hops = int(mask_cfg.get("hops", 1))
        if hops > 1:
            parts.append(f"{hops}hop")
        if not bool(mask_cfg.get("symmetric", True)):
            parts.append("dir")
        tag = _layer_tag(sorted(int(x) for x in mask_cfg.get("layers", [1])))
        if tag:
            parts.append(tag)
    elif rand:
        base = f"random_{lam:g}"
    else:
        base = f"kl_{lam:g}"
        if lam > 0:
            hops = int(reg.get("hops", 1))
            if hops > 1:
                parts.append(f"{hops}hop")
            if bool(reg.get("symmetrize", False)):
                parts.append("sym")
            specs = {
                json.dumps(h["layer"])
                for h in (reg.get("regularized_heads") or {}).values()
            }
            assert len(specs) <= 1, f"heads regularized on different layers: {specs}"
            spec = json.loads(specs.pop()) if specs else 1
            tag = _layer_tag(sorted([spec] if isinstance(spec, int) else spec))
            if tag:
                parts.append(tag)
    if not bool((model.get("learnable_embedding") or {}).get("enabled", True)):
        parts.append("emb")
    hidden = int(model.get("hidden_channels", 180))
    if hidden != 180:
        parts.append(f"w{hidden}")
    epochs = int((cfg.get("trainer") or {}).get("max_epochs", BUDGET))
    if epochs != BUDGET:
        parts.append(f"{epochs}ep")
    return "_".join([base, *parts]), lam, mask, rand


def reference_arm(arm: str) -> str:
    """The no-penalty arm an arm pairs against: the one sharing its representation,
    width and budget, so a composite or width-360 arm reads against its own control.
    """
    shared = [p for p in arm.split("_") if p in ("emb", "w360", "60ep")]
    return "_".join(["kl_0", *shared])


def pull_history(
    run: Any,
) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, float]]]:
    """Per-epoch series and per-probe-epoch series of one run."""
    epoch_vals: dict[str, dict[int, float]] = defaultdict(dict)
    step_acc: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    probe: dict[str, dict[int, float]] = defaultdict(dict)
    for row in run.scan_history():
        e = row.get("epoch")
        pe = row.get("probe/epoch")
        if pe is not None:
            for k in PROBE_KEYS:
                if row.get(k) is not None:
                    probe[k][int(pe)] = float(row[k])
        if e is None:
            continue
        e = int(e)
        for k in EPOCH_KEYS + EDGE_KEYS:
            if row.get(k) is not None:
                epoch_vals[k][e] = float(row[k])
        for k in STEP_KEYS:
            if row.get(k) is not None:
                step_acc[k][e].append(float(row[k]))
    for k, per_epoch in step_acc.items():
        for e, vals in per_epoch.items():
            epoch_vals[k][e] = statistics.fmean(vals)
    return epoch_vals, probe


def mean_over_graphs(
    epoch_vals: dict[str, dict[int, float]], stat: str
) -> dict[int, float]:
    """Edge-recovery statistic averaged over the nine regularized heads, per epoch."""
    out: dict[int, list[float]] = defaultdict(list)
    for g in GRAPHS:
        for e, v in epoch_vals.get(f"val_edge_recovery/{g}/{stat}", {}).items():
            out[e].append(v)
    return {e: statistics.fmean(v) for e, v in out.items() if len(v) == len(GRAPHS)}


def _at(d: dict[int, float], e: int) -> float | None:
    return d.get(e)


def _div(d: dict[int, float], lam: float) -> dict[int, float]:
    """The logged penalty is lambda-weighted; dividing by lambda gives the divergence."""
    if lam <= 0:
        return {}
    return {e: v / lam for e, v in d.items()}


def build_row(
    run: Any,
    epoch_vals: dict[str, dict[int, float]],
    probe: dict[str, dict[int, float]],
) -> RunRow:
    """Summarize one run."""
    arm, lam, mask, rand = classify(run.config)
    val = epoch_vals["val/gene_interaction/Pearson"]
    best_epoch = max(val, key=lambda e: val[e])
    window = [v for e, v in val.items() if WINDOW[0] <= e <= WINDOW[1]]
    full_window = len(window) == WINDOW[1] - WINDOW[0] + 1
    train_div = _div(epoch_vals.get("train/graph_reg_loss", {}), lam)
    val_div = _div(epoch_vals.get("val/graph_reg_loss", {}), lam)
    recall = mean_over_graphs(epoch_vals, "recall_at_deg")
    prec = mean_over_graphs(epoch_vals, "precision_k32")
    ratio = probe.get("probe/grad_ratio/graph_reg_to_point", {})
    cuda = epoch_vals.get("train/cuda_peak_allocated_gb", {})
    name = run.name
    job = name.split("-")[-1].split("_")[0] if name.startswith("run_") else ""
    return RunRow(
        run_id=run.id,
        run_url=run.url,
        run_name=name,
        arm=arm,
        graph_reg_lambda=lam,
        mask=mask,
        random_graph=rand,
        seed=int(run.config["seed"]),
        state=run.state,
        host=(run.metadata or {}).get("host", ""),
        slurm_job=job,
        epochs_logged=len(val),
        complete=len(val) >= BUDGET,
        val_pearson_fixed=_at(val, FIXED_EPOCH),
        val_pearson_max=val[best_epoch],
        val_pearson_max_epoch=best_epoch,
        val_pearson_window_mean=statistics.fmean(window) if full_window else None,
        val_pearson_window_sd=statistics.stdev(window) if full_window else None,
        val_point_loss_fixed=_at(epoch_vals["val/point_loss"], FIXED_EPOCH),
        train_pearson_fixed=_at(
            epoch_vals["train/gene_interaction/Pearson"], FIXED_EPOCH
        ),
        train_point_loss_fixed=_at(epoch_vals["train/point_loss"], FIXED_EPOCH),
        train_graph_reg_fixed=_at(
            epoch_vals.get("train/graph_reg_loss", {}), FIXED_EPOCH
        ),
        train_divergence_fixed=_at(train_div, FIXED_EPOCH),
        val_divergence_fixed=_at(val_div, FIXED_EPOCH),
        edge_recall_fixed=_at(recall, FIXED_EPOCH),
        edge_precision_k32_fixed=_at(prec, FIXED_EPOCH),
        probe_point_epoch0=probe.get("probe/grad_norm/point", {}).get(0),
        probe_graph_reg_epoch0=probe.get("probe/grad_norm/graph_reg", {}).get(0),
        probe_ratio_epoch0=ratio.get(0),
        probe_ratio_epoch20=ratio.get(20),
        cuda_peak_gb_max=max(cuda.values()) if cuda else None,
    )


def history_records(
    row: RunRow,
    epoch_vals: dict[str, dict[int, float]],
    probe: dict[str, dict[int, float]],
) -> list[dict[str, Any]]:
    """Long-format rows of every per-epoch series kept for the plots."""
    series: dict[str, dict[int, float]] = {
        k: epoch_vals[k] for k in EPOCH_KEYS + STEP_KEYS if k in epoch_vals
    }
    series["train/divergence"] = _div(
        epoch_vals.get("train/graph_reg_loss", {}), row.graph_reg_lambda
    )
    series["val/divergence"] = _div(
        epoch_vals.get("val/graph_reg_loss", {}), row.graph_reg_lambda
    )
    series["val_edge_recovery/mean/recall_at_deg"] = mean_over_graphs(
        epoch_vals, "recall_at_deg"
    )
    series["val_edge_recovery/mean/precision_k32"] = mean_over_graphs(
        epoch_vals, "precision_k32"
    )
    for k in PROBE_KEYS:
        series[k] = probe.get(k, {})
    out = []
    for key, d in series.items():
        for e, v in sorted(d.items()):
            out.append(
                {
                    "run_id": row.run_id,
                    "arm": row.arm,
                    "seed": row.seed,
                    "epoch": e,
                    "key": key,
                    "value": v,
                }
            )
    return out


# W&B group per arm. The ladder was launched as overrides of ctrl_013, whose config sets
# the group, so every ladder point landed in `s0_control_30ep` and the random-graph jobs
# in per-job groups; `--regroup` moves each run (every rank) to its arm's group so the
# group page shows the arm's seeds together. The names already used by the launched
# configs are kept.
ARM_GROUP = {
    "kl_0": "s0_lambda0_30ep",
    "kl_1e-05": "s0_lambda1e-5_30ep",
    "kl_0.0001": "s0_lambda1e-4_30ep",
    "kl_0.001": "s0_control_30ep",
    "kl_0.01": "s0_lambda1e-2_30ep",
    "kl_0.1": "s0_lambda1e-1_30ep",
    "kl_1": "s0_lambda1_30ep",
    "mask": "s0_hardmask_30ep",
    "random_0.001": "s0_random_30ep",
}
# Round-1b, 2 and 3 arms are grouped by their arm name, as W&B accepts it.
ARM_GROUP.update(
    {arm: "s0_" + arm.replace(".", "p").replace("-", "to") for arm in ROUND2_ARMS}
)
assert set(ARM_GROUP) == set(ARM_ORDER)


def pull(
    regroup: bool = False,
) -> tuple[list[RunRow], list[dict[str, Any]], list[dict[str, str]]]:
    """Every selected run, with the reason for each exclusion."""
    import wandb

    api = wandb.Api(timeout=180)
    rows, hist, excluded = [], [], []
    seen: set[str] = set()
    for run in api.runs(PROJECT, filters={"tags": {"$in": SELECT_TAGS}}, per_page=200):
        if run.id in seen:
            continue
        seen.add(run.id)
        host = (run.metadata or {}).get("host", "") or ""
        if run.state not in ("finished", "running"):
            excluded.append({"run_id": run.id, "reason": f"state {run.state}"})
            continue
        if "delta" not in host:
            excluded.append(
                {
                    "run_id": run.id,
                    "reason": f"host {host or 'unknown'}, not the Delta protocol",
                }
            )
            continue
        if regroup:
            group = ARM_GROUP[classify(run.config)[0]]
            if run.group != group:
                run.group = group
                run.update()
                print(f"regrouped {run.id} -> {group}")
        epoch_vals, probe = pull_history(run)
        if not epoch_vals["val/gene_interaction/Pearson"]:
            continue  # a non-zero rank of a job: no validation history, not a run of its own
        row = build_row(run, epoch_vals, probe)
        rows.append(row)
        hist.extend(history_records(row, epoch_vals, probe))
        print(
            f"{row.arm:14s} seed {row.seed} {row.run_id} {row.state:8s} epochs {row.epochs_logged:2d} "
            f"fixed {row.val_pearson_fixed if row.val_pearson_fixed is None else round(row.val_pearson_fixed, 4)} "
            f"max {row.val_pearson_max:.4f}@{row.val_pearson_max_epoch}"
        )
    return rows, hist, excluded


def _msd(vals: list[float]) -> tuple[float | None, float | None, int]:
    vals = [
        v
        for v in vals
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    if not vals:
        return None, None, 0
    return (
        statistics.fmean(vals),
        (statistics.stdev(vals) if len(vals) > 1 else None),
        len(vals),
    )


def summarize(runs: pd.DataFrame) -> dict[str, Any]:
    """Arm-level means over seeds, complete runs only for the fixed reading."""
    arms: dict[str, Any] = {}
    for arm in ARM_ORDER:
        sub = runs[runs.arm == arm]
        if sub.empty:
            arms[arm] = {"n": 0}
            continue
        done = sub[sub.complete]
        entry: dict[str, Any] = {
            "label": ARM_LABEL[arm],
            "n": int(len(sub)),
            "n_complete": int(len(done)),
            "seeds": sorted(int(s) for s in sub.seed),
            "run_ids": list(sub.sort_values("seed").run_id),
            "run_urls": list(sub.sort_values("seed").run_url),
            "group_url": f"https://wandb.ai/{PROJECT}/groups/{ARM_GROUP[arm]}",
            "partial_epochs": {
                str(int(r["seed"])): int(r["epochs_logged"])
                for r in sub.to_dict("records")
                if not r["complete"]
            },
        }
        for col in (
            "val_pearson_fixed",
            "val_pearson_max",
            "val_pearson_window_mean",
            "val_pearson_at_min_loss",
            "val_point_loss_min",
            "val_point_loss_min_epoch",
            "val_point_loss_fixed",
            "train_pearson_fixed",
            "train_point_loss_fixed",
            "train_divergence_fixed",
            "val_divergence_fixed",
            "edge_recall_fixed",
            "edge_precision_k32_fixed",
            "probe_ratio_epoch0",
            "probe_ratio_epoch20",
        ):
            src = (
                sub
                if col
                in ("val_pearson_max", "probe_ratio_epoch0", "probe_ratio_epoch20")
                else done
            )
            m, s, n = _msd(list(src[col]))
            entry[col] = {"mean": m, "sd": s, "n": n}
        arms[arm] = entry
    return arms


REFERENCE = (
    "kl_0"  # round 1; reference_arm() picks the matching control for later rounds
)


def derived_from_history(runs: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Per-run readings that need the whole curve: the epoch of minimum validation point
    loss, the loss there, and validation Pearson at that epoch (what a loss-monitored
    checkpoint would deploy). Computed from the history CSV so ``--offline`` has them.
    """
    loss = hist[hist.key == "val/point_loss"].pivot_table(
        index="run_id", columns="epoch", values="value"
    )
    pear = hist[hist.key == "val/gene_interaction/Pearson"].pivot_table(
        index="run_id", columns="epoch", values="value"
    )
    rows = []
    for rid in runs.run_id:
        lo = loss.loc[rid].dropna()
        e_min = int(lo.idxmin())
        rows.append(
            {
                "run_id": rid,
                "val_point_loss_min": float(lo.min()),
                "val_point_loss_min_epoch": e_min,
                "val_pearson_at_min_loss": float(pear.loc[rid, e_min]),
            }
        )
    return runs.drop(
        columns=[
            c
            for c in (
                "val_point_loss_min",
                "val_point_loss_min_epoch",
                "val_pearson_at_min_loss",
            )
            if c in runs.columns
        ]
    ).merge(pd.DataFrame(rows), on="run_id")


class PairedRow(BaseModel):
    """One arm against no penalty, paired by seed."""

    arm: str
    reading: str
    seeds: list[int]
    diffs: list[float]
    mean_diff: float
    sd_diff: float | None
    t: float | None
    p_two_sided: float | None
    direction: (
        str  # "up" if every seed is above the reference, "down" if below, else "mixed"
    )


def paired(runs: pd.DataFrame) -> list[PairedRow]:
    """Per-seed differences of each arm against no penalty, both readings.

    The split is pinned and the seed fixes the initialization and the batch order for
    every arm alike, so seeds pair. The fixed reading uses complete runs only; the max
    reading uses every run (a partial run's max is a lower bound on its final max, so
    it is kept and the run is marked in the run table). t is the paired t statistic with
    n - 1 degrees of freedom; with three seeds it is reported, not leaned on.
    """
    from scipy import stats

    out = []
    for reading, col, pool in (
        ("fixed", "val_pearson_fixed", runs[runs.complete]),
        ("max", "val_pearson_max", runs),
        ("min_loss", "val_pearson_at_min_loss", runs[runs.complete]),
    ):
        for arm in ARM_ORDER:
            reference = reference_arm(arm)
            if arm == reference:
                continue
            ref = pool[pool.arm == reference].set_index("seed")[col]
            sub = pool[pool.arm == arm].set_index("seed")[col]
            seeds = sorted(set(sub.index) & set(ref.index))
            if not seeds:
                continue
            diffs = [float(sub[s_] - ref[s_]) for s_ in seeds]
            mean = statistics.fmean(diffs)
            sd = statistics.stdev(diffs) if len(diffs) > 1 else None
            t = mean / (sd / math.sqrt(len(diffs))) if sd else None
            p = (
                float(2 * stats.t.sf(abs(t), df=len(diffs) - 1))
                if t is not None
                else None
            )
            direction = (
                "up"
                if all(d > 0 for d in diffs)
                else "down"
                if all(d < 0 for d in diffs)
                else "mixed"
            )
            out.append(
                PairedRow(
                    arm=arm,
                    reading=reading,
                    seeds=[int(x) for x in seeds],
                    diffs=diffs,
                    mean_diff=mean,
                    sd_diff=sd,
                    t=t,
                    p_two_sided=p,
                    direction=direction,
                )
            )
    return out


READING_LABEL = {
    "fixed": "epoch 29",
    "max": "max over epochs",
    "min_loss": "at min val loss",
}
ARROW = {"up": " $\\uparrow$", "down": " $\\downarrow$", "mixed": ""}


def _fmt(m: float | None, s: float | None, nd: int = 3, bold: bool = False) -> str:
    if m is None:
        return "--"
    txt = f"{m:.{nd}f}" if s is None else f"{m:.{nd}f} $\\pm$ {s:.{nd}f}"
    return f"\\textbf{{{txt}}}" if bold else txt


def write_tables(
    arms: dict[str, Any], runs: pd.DataFrame, pulled_at: str, pairs: list[PairedRow]
) -> None:
    """t1-arms (one row per arm), t2-runs (one row per run), t3-paired (arm minus no penalty by seed)."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    arrow = {(p.arm, p.reading): ARROW[p.direction] for p in pairs}
    src = (
        "%% SOURCE: experiments/025-solid-growth/scripts/graph_reg_sweep_readout.py "
        f"(W&B {PROJECT}, pulled {pulled_at}) -- GENERATED, do not edit\n"
    )
    best_fixed = max(
        (a["val_pearson_fixed"]["mean"] for a in arms.values() if a.get("n_complete")),
        default=None,
    )
    best_max = max(
        (a["val_pearson_max"]["mean"] for a in arms.values() if a.get("n")),
        default=None,
    )
    best_min = max(
        (
            a["val_pearson_at_min_loss"]["mean"]
            for a in arms.values()
            if a.get("n_complete")
        ),
        default=None,
    )
    lines = [
        src,
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "arm & $n$ & Pearson, epoch 29 & Pearson, max over epochs & Pearson, at min val loss \\\\",
        "\\midrule",
    ]
    for arm in ARM_ORDER:
        a = arms[arm]
        if not a.get("n"):
            if arm in ROUND1_ARMS:
                lines.append(
                    f"{ARM_LABEL[arm]} & 0 & \\multicolumn{{3}}{{l}}{{not run}} \\\\"
                )
            continue
        n_txt = str(a["n_complete"]) + (
            f" (+{a['n'] - a['n_complete']} partial)"
            if a["n"] > a["n_complete"]
            else ""
        )
        f = a["val_pearson_fixed"]
        mx = a["val_pearson_max"]
        mn = a["val_pearson_at_min_loss"]
        lines.append(
            f"{ARM_LABEL[arm]} & {n_txt} & "
            f"{_fmt(f['mean'], f['sd'], bold=f['mean'] is not None and f['mean'] == best_fixed)}{arrow.get((arm, 'fixed'), '')} & "
            f"{_fmt(mx['mean'], mx['sd'], bold=mx['mean'] == best_max)}{arrow.get((arm, 'max'), '')} & "
            f"{_fmt(mn['mean'], mn['sd'], bold=mn['mean'] == best_min)}{arrow.get((arm, 'min_loss'), '')} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(osp.join(TABLES_DIR, "t1-arms.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    lines = [
        src,
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "arm & point loss, ep 29 & min point loss (epoch) & train Pearson, ep 29 & divergence & edge recall \\\\",
        "\\midrule",
    ]
    for arm in ARM_ORDER:
        a = arms[arm]
        if not a.get("n_complete"):
            continue
        lines.append(
            f"{ARM_LABEL[arm]} & "
            f"{_fmt(a['val_point_loss_fixed']['mean'], a['val_point_loss_fixed']['sd'])} & "
            f"{_fmt(a['val_point_loss_min']['mean'], a['val_point_loss_min']['sd'])} ({a['val_point_loss_min_epoch']['mean']:.1f}) & "
            f"{_fmt(a['train_pearson_fixed']['mean'], a['train_pearson_fixed']['sd'])} & "
            f"{_fmt(a['val_divergence_fixed']['mean'], None, nd=0)} & "
            f"{_fmt(a['edge_recall_fixed']['mean'], a['edge_recall_fixed']['sd'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(osp.join(TABLES_DIR, "t1b-diagnostics.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    lines = [
        src,
        "\\begin{tabular}{llrrrrl}",
        "\\toprule",
        "arm & seed & epochs & Pearson, ep 29 & max (epoch) & "
        "gradient ratio, ep 0 & W\\&B run \\\\",
        "\\midrule",
    ]
    for arm in ARM_ORDER:
        sub = runs[runs.arm == arm].sort_values("seed")
        for r in sub.to_dict("records"):
            fixed = (
                "--"
                if pd.isna(r["val_pearson_fixed"])
                else f"{float(r['val_pearson_fixed']):.4f}"
            )
            ratio = (
                "--"
                if pd.isna(r["probe_ratio_epoch0"])
                else f"{float(r['probe_ratio_epoch0']):.2f}"
            )
            ep = f"{int(r['epochs_logged'])}" + ("" if r["complete"] else " (running)")
            lines.append(
                f"{ARM_LABEL[arm]} & {int(r['seed'])} & {ep} & {fixed} & "
                f"{float(r['val_pearson_max']):.4f} ({int(r['val_pearson_max_epoch'])}) & "
                f"{ratio} & \\texttt{{{r['run_id']}}} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(osp.join(TABLES_DIR, "t2-runs.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    lines = [
        src,
        "\\begin{tabular}{llrlrrr}",
        "\\toprule",
        "arm & reading & $n$ & difference by seed & mean & paired $t$ & $p$ \\\\",
        "\\midrule",
    ]
    for p_ in pairs:
        diffs = ", ".join(f"{d:+.3f}" for d in p_.diffs)
        t_txt = "--" if p_.t is None else f"{p_.t:.1f}"
        p_txt = "--" if p_.p_two_sided is None else f"{p_.p_two_sided:.3f}"
        lines.append(
            f"{ARM_LABEL[p_.arm]} & {READING_LABEL[p_.reading]} & {len(p_.diffs)} & "
            f"{diffs} & {p_.mean_diff:+.4f}{ARROW[p_.direction]} & {t_txt} & {p_txt} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(osp.join(TABLES_DIR, "t3-paired.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    """Pull, summarize, write."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline",
        action="store_true",
        help="rebuild summary and tables from the CSVs",
    )
    parser.add_argument(
        "--regroup",
        action="store_true",
        help="set each run's W&B group to its arm (ARM_GROUP)",
    )
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    runs_csv = osp.join(RESULTS_DIR, "graph_reg_sweep_runs.csv")
    hist_csv = osp.join(RESULTS_DIR, "graph_reg_sweep_history.csv")
    summary_json = osp.join(RESULTS_DIR, "graph_reg_sweep_summary.json")
    if args.offline:
        runs = pd.read_csv(runs_csv)
        prev = json.load(open(summary_json))
        pulled_at, excluded = prev["pulled_at"], prev["excluded"]
    else:
        rows, hist, excluded = pull(regroup=args.regroup)
        pulled_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        runs = pd.DataFrame([r.model_dump() for r in rows])
        pd.DataFrame(hist).to_csv(hist_csv, index=False)
    runs = derived_from_history(runs, pd.read_csv(hist_csv))
    runs.to_csv(runs_csv, index=False)
    arms = summarize(runs)
    pairs = paired(runs)
    summary = {
        "project": PROJECT,
        "select_tags": SELECT_TAGS,
        "pulled_at": pulled_at,
        "budget_epochs": BUDGET,
        "fixed_epoch": FIXED_EPOCH,
        "window": list(WINDOW),
        "n_runs": int(len(runs)),
        "excluded": excluded,
        "arms": arms,
        "paired_vs_no_penalty": [p_.model_dump() for p_ in pairs],
    }
    with open(summary_json, "w") as fh:
        json.dump(summary, fh, indent=1)
    write_tables(arms, runs, pulled_at, pairs)
    print()
    for p_ in pairs:
        print(
            f"paired {p_.reading:5s} {p_.arm:14s} n={len(p_.diffs)} mean {p_.mean_diff:+.4f} {p_.direction:5s} t={p_.t if p_.t is None else round(p_.t, 1)} p={p_.p_two_sided if p_.p_two_sided is None else round(p_.p_two_sided, 4)}"
        )
    print()
    print(
        f"{'arm':14s} {'n':>6s} {'fixed (ep 29)':>22s} {'max':>22s} {'train P':>10s} {'divergence':>12s}"
    )
    for arm in ARM_ORDER:
        a = arms[arm]
        if not a.get("n"):
            if arm in ROUND1_ARMS:
                print(f"{arm:14s} {'0':>6s}  not run")
            continue
        f, mx = a["val_pearson_fixed"], a["val_pearson_max"]
        print(
            f"{arm:14s} {a['n_complete']:>3d}+{a['n'] - a['n_complete']:<2d} "
            f"{_fmt(f['mean'], f['sd']).replace('$\\pm$', '+-'):>22s} "
            f"{_fmt(mx['mean'], mx['sd']).replace('$\\pm$', '+-'):>22s} "
            f"{_fmt(a['train_pearson_fixed']['mean'], None):>10s} "
            f"{_fmt(a['val_divergence_fixed']['mean'], None, nd=0):>12s}"
        )
    print(
        f"\nexcluded {len(excluded)} runs; wrote {runs_csv}, {hist_csv}, {summary_json}, {TABLES_DIR}/t1-arms.tex, t2-runs.tex"
    )


if __name__ == "__main__":
    main()
