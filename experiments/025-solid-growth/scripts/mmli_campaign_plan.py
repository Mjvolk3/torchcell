# experiments/025-solid-growth/scripts/mmli_campaign_plan.py
# [[experiments.025-solid-growth.scripts.mmli_campaign_plan]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/mmli_campaign_plan
"""The mmli campaign as one table: every planned cell, filled where a run has finished.

THE PLAN is the `CELLS` list below: one cell per (records, split, gene representation),
with the seeds it is to be run at. The table is the plan with the measured cells filled in,
so a cell that has not run is an empty cell and a cell that is running says which epoch it
reached. Nothing in it is typed by hand.

THE SCORE of a run is the mean of `val/gene_interaction/Pearson` over epochs 10 to 29, the
window `disjoint_embedding_readout.py` and `s3_closure_readout.py` declare, reported only
when all 20 epochs are present. 50-epoch cells also report epochs 30 to 49 as a secondary
column. The max over epochs is never the score.

THE QUEUE is read from the cluster with `--refresh-queue` (one `squeue` over ssh, written
to results/mmli_queue_snapshot.json) and from that snapshot otherwise, so the table can be
rebuilt without cluster access.

    python experiments/025-solid-growth/scripts/mmli_campaign_plan.py --refresh-queue

Writes results/mmli_campaign_plan.csv, results/mmli_campaign_plan.json and
notes-tex/025-mmli-campaign/tables/t1-campaign.tex, t2-queue.tex.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import statistics
import subprocess

import pandas as pd
import wandb
from dotenv import load_dotenv
from pydantic import BaseModel

from torchcell.timestamp import timestamp

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results")
TABLE_DIR = osp.join(osp.dirname(EXPERIMENT_ROOT), "notes-tex/025-mmli-campaign/tables")
PROJECT = "zhao-group/torchcell_025-solid-growth_equivariant_cell_graph_transformer"
IGB_LOGIN = "mjvolk3@biologin.igb.illinois.edu"
QUEUE_SNAPSHOT = osp.join(RESULTS_DIR, "mmli_queue_snapshot.json")
VAL_KEY = "val/gene_interaction/Pearson"
WINDOW = (10, 29)
LATE_WINDOW = (30, 49)
# The abandoned first attempt at S3 seed 1 (rank 0 kj03xx8y) and its three other ranks.
EXCLUDED_RUN_IDS = {"kj03xx8y", "0kaadgdu", "bekoxpor", "ztfcxu37"}
# Hours per run, measured on four mmli A100s: S0 30 epochs from jobs 2408604-07
# (07:39, 07:40, 07:43, 06:54), S3 from job 2409033 (104 epochs in 68.5 h, 39.5 min each).
HOURS = {"S0": 7.5, "S3": 33.0, "S4": 33.0}


class Cell(BaseModel):
    block: str
    key: str
    config: str
    records: str
    split: str
    genes: str
    fitness: bool
    budget_epochs: int
    seeds: list[int]
    comparator: str | None = None
    graph_reg_lambda: float | None = None
    # sibling configs whose runs count toward this cell (same arm at another epoch ceiling)
    also_configs: list[str] = []
    question: str = ""


RANDOM = "A. Random split (010's pinned split, 37,673 validation triples)"
DISJOINT = "B. Query-pair-disjoint split (89 of 420 query pairs held out, 37,705 validation triples)"

CELLS: list[Cell] = [
    Cell(block=RANDOM, key="r_s0_tab_nofit", config="cgt_s0_r_kl_ctrl_013", records="S0",
         split="R", genes="table", fitness=False, budget_epochs=30, seeds=[1, 2, 3],
         graph_reg_lambda=1e-3),
    Cell(block=RANDOM, key="r_s0_tab", config="cgt_s0_r_kl_fit_014", records="S0",
         split="R", genes="table", fitness=True, budget_epochs=30, seeds=[1, 2, 3],
         comparator="r_s0_tab_nofit"),
    Cell(block=RANDOM, key="r_s3_tab", config="cgt_s3_r_kl_fit_031", records="S3",
         split="R", genes="table", fitness=True, budget_epochs=50, seeds=[1, 2, 3],
         comparator="r_s0_tab",
         question="do more fitness records improve trigenic prediction"),
    Cell(block=RANDOM, key="r_s0_emb", config="cgt_s0_r_kl_embfit_035", records="S0",
         split="R", genes="composite", fitness=True, budget_epochs=30, seeds=[1, 2, 3],
         comparator="r_s0_tab",
         question="does a transferable representation match the table where the table is at its best"),
    Cell(block=RANDOM, key="r_s3_emb", config="cgt_s3_r_kl_embfit_034", records="S3",
         split="R", genes="composite", fitness=True, budget_epochs=50, seeds=[1, 2, 3],
         comparator="r_s3_tab",
         question="does the composite keep the closure gain"),
    Cell(block=RANDOM, key="r_s4_tab", config="cgt_s4_r_kl_fit_039", records="S4",
         split="R", genes="table", fitness=True, budget_epochs=50, seeds=[1, 2, 3],
         comparator="r_s3_tab",
         question="is the gain from the triples' own pairs or from any doubles"),
    Cell(block=DISJOINT, key="q_s0_tab_nofit", config="cgt_s0_q_kl_ctrl_016", records="S0",
         split="Q", genes="table", fitness=False, budget_epochs=30, seeds=[42, 1, 2]),
    Cell(block=DISJOINT, key="q_s0_rand", config="cgt_s0_q_kl_rand_018", records="S0",
         split="Q", genes="random vector", fitness=False, budget_epochs=30, seeds=[42, 1, 2],
         comparator="q_s0_tab_nofit"),
    Cell(block=DISJOINT, key="q_s0_emb_nofit", config="cgt_s0_q_kl_emb_017", records="S0",
         split="Q", genes="composite", fitness=False, budget_epochs=30, seeds=[42, 1, 2],
         comparator="q_s0_tab_nofit"),
    Cell(block=DISJOINT, key="q_s0_calm", config="cgt_s0_q_kl_calm_020", records="S0",
         split="Q", genes="CaLM alone", fitness=False, budget_epochs=30, seeds=[42, 1, 2],
         comparator="q_s0_emb_nofit", question="which region carries the composite"),
    Cell(block=DISJOINT, key="q_s0_prot", config="cgt_s0_q_kl_prot_021", records="S0",
         split="Q", genes="ProtT5 alone", fitness=False, budget_epochs=30, seeds=[42, 1, 2],
         comparator="q_s0_emb_nofit"),
    Cell(block=DISJOINT, key="q_s0_fudt", config="cgt_s0_q_kl_fudt_032", records="S0",
         split="Q", genes="flanks alone", fitness=False, budget_epochs=30, seeds=[42, 1, 2],
         comparator="q_s0_emb_nofit", also_configs=["cgt_s0_q_kl_fudt_026"]),
    Cell(block=DISJOINT, key="q_s0_tab", config="cgt_s0_q_kl_fit_038", records="S0",
         split="Q", genes="table", fitness=True, budget_epochs=30, seeds=[1, 2, 3],
         comparator="q_s0_tab_nofit"),
    Cell(block=DISJOINT, key="q_s0_emb", config="cgt_s0_q_kl_embfit_027", records="S0",
         split="Q", genes="composite", fitness=True, budget_epochs=30, seeds=[42, 1, 2],
         comparator="q_s0_emb_nofit"),
    Cell(block=DISJOINT, key="q_s3_tab", config="cgt_s3_q_kl_fit_036", records="S3 strict",
         split="Q", genes="table", fitness=True, budget_epochs=50, seeds=[1, 2, 3],
         comparator="q_s0_tab",
         question="do more fitness records help on unseen query pairs"),
    Cell(block=DISJOINT, key="q_s3_emb", config="cgt_s3_q_kl_embfit_037", records="S3 strict",
         split="Q", genes="composite", fitness=True, budget_epochs=50, seeds=[1, 2, 3],
         comparator="q_s0_emb"),
]


# ------------------------------------------------------------------------------ queue


def refresh_queue() -> dict:
    """Pending and running mmli jobs with their Hydra config and seed, in chain order."""
    fmt = "%i|%j|%T|%E|%M"
    out = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=20", IGB_LOGIN,
         f"squeue -p mmli -u mjvolk3 -h -o '{fmt}'; echo ==; "
         "for j in $(squeue -p mmli -u mjvolk3 -h -o %i); do "
         "echo \"$j|$(scontrol show job $j | grep -oE 'Command=.*' | cut -d' ' -f2-)\"; done"],
        check=True, capture_output=True, text=True,
    ).stdout
    head, tail = out.split("==\n")
    args = dict(line.split("|", 1) for line in tail.strip().splitlines())
    jobs = {}
    for line in head.strip().splitlines():
        jid, name, state, dep, elapsed = line.split("|")
        tokens = args[jid].split()
        seed = next(int(t.split("=")[1]) for t in tokens if t.startswith("+seed="))
        after = dep.split(":")[1].split("(")[0] if dep.startswith("afterany") else None
        jobs[jid] = {"job_id": jid, "name": name, "state": state, "after": after,
                     "elapsed": elapsed, "config": tokens[0], "seed": seed}
    # chain order: the running job, then follow `after` links
    by_after = {j["after"]: j for j in jobs.values() if j["after"]}
    running = [j for j in jobs.values() if j["state"] == "RUNNING"]
    assert len(running) == 1, f"expected one running mmli job, found {len(running)}"
    order, cur = [running[0]], running[0]
    while cur["job_id"] in by_after:
        cur = by_after[cur["job_id"]]
        order.append(cur)
    assert len(order) == len(jobs), "the mmli queue is not one chain"
    snap = {"read_at": timestamp(), "chain": order}
    with open(QUEUE_SNAPSHOT, "w") as f:
        json.dump(snap, f, indent=1)
    return snap


# -------------------------------------------------------------------------------- W&B


def val_curve(run) -> dict[int, float]:
    """Validation Pearson by epoch, checked against the run's own summary.

    Read with `run.history(keys=...)`, not `scan_history`. On the 104-epoch S3 seed 1
    (yb4gjh51, 12,083 steps) `scan_history(keys=["epoch", VAL_KEY])` returns 104 rows of
    `{"_step": 0, "epoch": None}` with no metric in them, and the unkeyed scan returns 188
    rows covering 3 epochs; `history` with the same keys returns the 104 real rows. With
    `samples` above the number of validation epochs `history` does not downsample.
    Validation is logged once per epoch, so the curve must reach the epoch the summary
    reports; a shorter one is a truncated read and stops the script.
    """
    rows = run.history(keys=["epoch", VAL_KEY], samples=10_000, pandas=False)
    curve = {int(r["epoch"]): float(r[VAL_KEY]) for r in rows}
    summary_epoch = int(run.summary["epoch"])
    if not curve or max(curve) < summary_epoch - 1:
        raise RuntimeError(
            f"W&B returned a truncated history for {run.id}: {len(curve)} validation epochs, "
            f"summary epoch {summary_epoch}"
        )
    return curve


def window_mean(curve: dict[int, float], lo: int, hi: int) -> float | None:
    vals = [v for e, v in curve.items() if lo <= e <= hi]
    return statistics.fmean(vals) if len(vals) == hi - lo + 1 else None


def read_cell(api, cell: Cell) -> dict[int, dict]:
    """Per seed: the rank-0 run with the most validation epochs, scored by the window."""
    runs = [
        r for r in api.runs(PROJECT, filters={"tags": {"$in": [cell.config, *cell.also_configs]}})
        if r.id not in EXCLUDED_RUN_IDS and VAL_KEY in r.summary
    ]
    if cell.graph_reg_lambda is not None:
        runs = [
            r for r in runs
            if float(r.config["model"]["graph_regularization"]["graph_reg_lambda"])
            == cell.graph_reg_lambda
        ]
    out: dict[int, dict] = {}
    for seed in cell.seeds:
        curves = [(r, val_curve(r)) for r in runs if int(r.config.get("seed", 42)) == seed]
        if not curves:
            continue
        run, curve = max(curves, key=lambda rc: len(rc[1]))
        out[seed] = {
            "run_id": run.id, "run_url": run.url, "state": run.state,
            "val_epochs": len(curve), "last_epoch": max(curve),
            "score": window_mean(curve, *WINDOW),
            "score_late": window_mean(curve, *LATE_WINDOW),
            "max_biased": max(curve.values()), "max_epoch": max(curve, key=curve.get),
        }
    return out


# ------------------------------------------------------------------------------ table


def tex(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


def seed_entry(cell: Cell, seed: int, res: dict[int, dict], queue_pos: dict) -> str:
    r = res.get(seed)
    if r and r["score"] is not None and (cell.config, seed) not in queue_pos:
        return f"{r['score']:.3f}"
    pos = queue_pos.get((cell.config, seed))
    if pos is not None and pos[1] == "RUNNING":
        ep = r["last_epoch"] if r else 0
        return rf"\textit{{run, ep {ep}}}"
    if pos is not None:
        return rf"\textit{{q{pos[0]}}}"
    if r:
        return rf"\textit{{partial {r['val_epochs']}}}"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-queue", action="store_true")
    a = ap.parse_args()
    snap = refresh_queue() if a.refresh_queue else json.load(open(QUEUE_SNAPSHOT))
    chain = snap["chain"]
    queue_pos = {(j["config"], j["seed"]): (i, j["state"]) for i, j in enumerate(chain)}

    api = wandb.Api(timeout=120)
    results = {c.key: read_cell(api, c) for c in CELLS}
    by_key = {c.key: c for c in CELLS}

    def scores(key: str) -> dict[int, float]:
        return {
            s: r["score"] for s, r in results[key].items()
            if r["score"] is not None and (by_key[key].config, s) not in queue_pos
        }

    rows = []
    for c in CELLS:
        sc = scores(c.key)
        vals = list(sc.values())
        late = [r["score_late"] for r in results[c.key].values() if r["score_late"] is not None]
        comp = scores(c.comparator) if c.comparator else {}
        comp_mean = statistics.fmean(comp.values()) if comp else None
        rows.append({
            "block": c.block, "key": c.key, "config": c.config, "records": c.records,
            "split": c.split, "genes": c.genes, "fitness": c.fitness,
            "budget_epochs": c.budget_epochs, "seeds_planned": len(c.seeds),
            "seeds_scored": len(vals),
            "mean": statistics.fmean(vals) if vals else None,
            "sd": statistics.stdev(vals) if len(vals) > 1 else None,
            "mean_ep30_49": statistics.fmean(late) if late else None,
            "comparator": c.comparator,
            "delta_vs_comparator": (statistics.fmean(vals) - comp_mean) if vals and comp_mean is not None else None,
            "question": c.question,
            "seed_entries": [seed_entry(c, s, results[c.key], queue_pos) for s in c.seeds],
            "seed_labels": c.seeds,
            "runs": results[c.key],
        })
    with open(osp.join(RESULTS_DIR, "mmli_campaign_plan.json"), "w") as f:
        json.dump({"generated": timestamp(), "queue_read_at": snap["read_at"],
                   "scoring_rule": f"mean of {VAL_KEY} over epochs {WINDOW[0]}-{WINDOW[1]}, complete windows only",
                   "cells": rows}, f, indent=1)
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("runs", "seed_entries", "seed_labels")}
                  for r in rows]).to_csv(osp.join(RESULTS_DIR, "mmli_campaign_plan.csv"), index=False)
    write_campaign_table(rows)
    write_queue_table(chain, by_config={c.config: c for c in CELLS})
    for r in rows:
        print(f"{r['config']:26s} {r['records']:9s} {r['split']} {r['genes']:14s} "
              f"{' | '.join(e or '.' for e in r['seed_entries']):40s} mean={r['mean']}")


def write_campaign_table(rows: list[dict]) -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)
    head = ("%% SOURCE: experiments/025-solid-growth/scripts/mmli_campaign_plan.py "
            "(results/mmli_campaign_plan.json) -- GENERATED, do not edit\n")
    lines = [head, r"\begin{tabular}{llllrrrrrrr}", r"\toprule",
             r"records & genes & fit. & config & seed a & seed b & seed c & $n$ & mean & sd & $\Delta$ \\"]
    for block in dict.fromkeys(r["block"] for r in rows):
        lines += [r"\midrule", rf"\multicolumn{{11}}{{l}}{{\textbf{{{tex(block)}}}}} \\", r"\midrule"]
        brows = [r for r in rows if r["block"] == block]
        best = max((r["mean"] for r in brows if r["mean"] is not None), default=None)
        for r in brows:
            ent = (r["seed_entries"] + ["", "", ""])[:3]
            if len(r["seed_labels"]) < 3:
                ent[len(r["seed_labels"]):] = [r"n/a"] * (3 - len(r["seed_labels"]))
            mean = "" if r["mean"] is None else f"{r['mean']:.3f}"
            if r["mean"] is not None and r["mean"] == best:
                mean = rf"\textbf{{{mean}}}"
            sd = "" if r["sd"] is None else f"{r['sd']:.3f}"
            d = "" if r["delta_vs_comparator"] is None else f"{r['delta_vs_comparator']:+.3f}"
            cfg = tex(r["config"].replace("cgt_", ""))
            lines.append(
                f"{r['records']} & {r['genes']} & {'yes' if r['fitness'] else 'no'} & "
                rf"\texttt{{{cfg}}} & {ent[0]} & {ent[1]} & {ent[2]} & "
                f"{r['seeds_scored']}/{r['seeds_planned']} & {mean} & {sd} & {d} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(osp.join(TABLE_DIR, "t1-campaign.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def elapsed_hours(t: str) -> float:
    """squeue %M: [days-]hours:minutes:seconds, or minutes:seconds under an hour."""
    days, _, clock = t.rpartition("-")
    parts = [int(x) for x in clock.split(":")]
    parts = [0] * (3 - len(parts)) + parts
    return (int(days) if days else 0) * 24 + parts[0] + parts[1] / 60 + parts[2] / 3600


def write_queue_table(chain: list[dict], by_config: dict[str, Cell]) -> None:
    head = ("%% SOURCE: experiments/025-solid-growth/scripts/mmli_campaign_plan.py "
            "(results/mmli_queue_snapshot.json) -- GENERATED, do not edit\n")
    lines = [head, r"\begin{tabular}{rllllrrr}", r"\toprule",
             r"\# & job & name & config & state & seed & hours & cumulative h \\", r"\midrule"]
    cum = 0.0
    for i, j in enumerate(chain):
        c = by_config[j["config"]]
        h = HOURS[c.records.split()[0]]
        if j["state"] == "RUNNING":
            h = max(h - elapsed_hours(j["elapsed"]), 0.0)
        cum += h
        cfg = tex(j["config"].replace("cgt_", ""))
        lines.append(f"{i} & {j['job_id']} & \\texttt{{{tex(j['name'])}}} & \\texttt{{{cfg}}} & "
                     f"{j['state'].lower()} & {j['seed']} & {h:.1f} & {cum:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(osp.join(TABLE_DIR, "t2-queue.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
