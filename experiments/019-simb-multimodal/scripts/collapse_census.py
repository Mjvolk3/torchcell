# experiments/019-simb-multimodal/scripts/collapse_census.py
# [[experiments.019-simb-multimodal.scripts.collapse_census]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/collapse_census
"""Collapse census of the small-trunk expression rounds (v21 on Delta, v22 on GilaHyper).

A head LAUNCHES at the first epoch its validation prediction spread ratio
(sd of predictions over sd of targets across validation strains) reaches 0.05. A DEAD
STRETCH is a maximal run of consecutive logged epochs after launch with the ratio below
0.01, the per-gene-mean predictor. A run COLLAPSED if it has a dead stretch of at least
50 epochs (the rule of v22_readout.py); it NEVER LAUNCHED if no epoch reached 0.05.

Per dead stretch the census records its length, start epoch, whether it was still open at
the run's last logged epoch, and whether the ratio later returned to 0.05 (recovered). Per
run it records the per-epoch gradient norm at the collapse epoch against the 50 epochs
before, which is the measurement behind the one-epoch-event description. Runs stopped by
hand after collapsing keep their history, so the census is of what was logged, not of what
is still running.

    python experiments/019-simb-multimodal/scripts/collapse_census.py

Writes results/collapse_census.json and notes-tex/figure-3-gate/tables/collapse_census.tex.
"""

from __future__ import annotations

import json
import os.path as osp
import statistics as st
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import wandb

ENTITY = "zhao-group"
PROJECTS = {"v21": "torchcell_019_expr_v21", "v22": "torchcell_019_expr_v22"}
LAUNCH, DEAD, RULE = 0.05, 0.01, 50
RELAUNCH = "2026-10-06T23:38:00"  # UTC; v21_readout.py carries the same constant
BINS = [(1, 4), (5, 19), (20, 49), (50, 99), (100, 10**6)]
REPO = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
RESULTS = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")
TABLES = osp.join(REPO, "notes-tex", "figure-3-gate", "tables")


def read_run(item: tuple[str, Any]) -> dict[str, Any]:
    rnd, run = item
    head = "proteome" if "proteome" in run.tags else "expression"
    key = f"val/{head}/pred_sd_ratio"
    sd = sorted(
        (int(x["epoch"]), float(x[key]))
        for x in run.history(keys=["epoch", key], pandas=False, samples=100000)
        if x.get(key) is not None
    )
    # Per-epoch gradient norm, placed by W&B step against the epoch rows.
    gn = sorted(
        (int(x["_step"]), float(x["train/grad_norm"]))
        for x in run.history(keys=["train/grad_norm"], pandas=False, samples=100000)
        if x.get("train/grad_norm") is not None
    )
    ep_step = sorted(
        (int(x["_step"]), int(x["epoch"]))
        for x in run.history(keys=["epoch", "val/loss"], pandas=False, samples=100000)
    )
    steps = [s for s, _ in ep_step]
    import bisect

    def epoch_of(step: int) -> int:
        i = bisect.bisect_left(steps, step)
        return ep_step[min(i, len(ep_step) - 1)][1]

    grad_by_epoch: dict[int, float] = {}
    for s, g in gn:
        grad_by_epoch.setdefault(epoch_of(s), g)
    tag = next(t for t in run.tags if t.startswith(("S_", "F_")))
    if rnd == "v21":
        arm, split = tag.rsplit("_s", 1)
    else:
        arm, split = tag, next(t for t in run.tags if t.startswith("split")).removeprefix("split")
    launch = next((e for e, v in sd if v >= LAUNCH), None)
    stretches = []
    i = 0
    while launch is not None and i < len(sd):
        e, v = sd[i]
        if e > launch and v < DEAD:
            j = i
            while j < len(sd) and sd[j][1] < DEAD:
                j += 1
            later = sd[j:]
            before = [grad_by_epoch[k] for k in range(e - 50, e) if k in grad_by_epoch]
            stretches.append(
                {
                    "start": e,
                    "length": j - i,
                    "open": j >= len(sd),
                    "recovered": any(v2 >= LAUNCH for _, v2 in later),
                    "grad_norm_at_start": grad_by_epoch.get(e),
                    "grad_norm_mean_50_before": st.mean(before) if before else None,
                }
            )
            i = j
        else:
            i += 1
    return {
        "round": rnd,
        "id": run.id,
        "arm": arm,
        "split": int(split),
        # v21's first launch of 2026-10-06 was killed inside its jobs by the in-place swap
        # (18:38 CT) and relaunched; both attempts are in the project.
        "cohort": "relaunched" if rnd == "v21" and run.created_at >= RELAUNCH else "original",
        "state": run.state,
        "last_epoch": sd[-1][0] if sd else None,
        "max_spread": max((v for _, v in sd), default=None),
        "launch_epoch": launch,
        "collapsed": any(s["length"] >= RULE for s in stretches),
        "stretches": stretches,
    }


def main() -> None:
    api = wandb.Api(timeout=120)
    items = [(rnd, r) for rnd, p in PROJECTS.items() for r in api.runs(f"{ENTITY}/{p}")]
    with ThreadPoolExecutor(12) as ex:
        runs = list(ex.map(read_run, items))
    stretches = [dict(s, round=r["round"], arm=r["arm"], id=r["id"]) for r in runs for s in r["stretches"]]
    hist = []
    for lo, hi in BINS:
        s = [x for x in stretches if lo <= x["length"] <= hi]
        hist.append(
            {
                "length_from": lo,
                "length_to": hi if hi < 10**6 else None,
                "n": len(s),
                "recovered": sum(x["recovered"] for x in s),
                "open": sum(x["open"] for x in s),
            }
        )
    by_arm: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: Counter())
    for r in runs:
        k = (r["round"], r["arm"], r["cohort"])
        by_arm[k]["runs"] += 1
        by_arm[k]["collapsed"] += r["collapsed"]
        by_arm[k]["never_launched"] += r["launch_epoch"] is None and (r["last_epoch"] or 0) >= 200
    long = [s for s in stretches if s["length"] >= RULE]
    spikes = [
        s["grad_norm_at_start"] / s["grad_norm_mean_50_before"]
        for s in long
        if s["grad_norm_at_start"] and s["grad_norm_mean_50_before"]
    ]
    out = {
        "generated_by": "experiments/019-simb-multimodal/scripts/collapse_census.py",
        "rule": {"launch": LAUNCH, "dead": DEAD, "collapse_epochs": RULE},
        "dead_stretch_histogram": hist,
        "collapse_onsets": sorted(s["start"] for s in long),
        "grad_norm_spike_ratio_at_collapse": {
            "n": len(spikes),
            "median": st.median(spikes) if spikes else None,
            "min": min(spikes) if spikes else None,
            "max": max(spikes) if spikes else None,
        },
        "by_arm": [{"round": k[0], "arm": k[1], "cohort": k[2], **v} for k, v in sorted(by_arm.items())],
        "runs": runs,
    }
    with open(osp.join(RESULTS, "collapse_census.json"), "w") as f:
        json.dump(out, f, indent=1)
    for h in hist:
        print(f"dead stretch {h['length_from']}-{h['length_to'] or 'inf'}: n {h['n']} recovered {h['recovered']} open {h['open']}")
    print("onsets of stretches >= 50:", out["collapse_onsets"])
    print("grad-norm spike ratio at collapse:", out["grad_norm_spike_ratio_at_collapse"])
    for a in out["by_arm"]:
        print(f"{a['round']} {a['arm']:<16} {a['cohort']:<11} runs {a['runs']:>3} collapsed {a['collapsed']:>2} never launched {a['never_launched']}")

    lines = [
        "%% GENERATED by experiments/019-simb-multimodal/scripts/collapse_census.py. Do not edit by hand.",
        "%% SOURCE: W&B zhao-group/torchcell_019_expr_v21 and torchcell_019_expr_v22 run histories",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\footnotesize",
        "  \\caption[Collapse census]{Collapse census of the small-trunk rounds. A run launches at the first"
        " epoch its validation prediction spread ratio reaches 0.05; a dead stretch is a maximal run of"
        " logged epochs after launch with the ratio below 0.01; a run collapsed if a dead stretch lasts"
        " 50 epochs or more, and never launched if no epoch reached 0.05 by epoch 200. Left: every dead"
        " stretch across both rounds by length, how many later returned to 0.05 and how many were still"
        " open at the run's last logged epoch. Right: runs per arm and launch cohort; v21's original"
        " cohort is the launch of 2026-10-06 that the in-place swap killed inside its jobs (pack 0"
        " excepted), the relaunched cohort the attempt that ran to 1{,}200 epochs. Runs stopped by hand"
        " after collapsing keep their history. \\src{experiments/019-simb-multimodal/scripts/collapse_census.py}}",
        "  \\label{tab:collapse-census}",
        "  \\begin{tabular}{@{}lrrr@{}}",
        "    \\toprule",
        "    dead stretch, epochs & n & recovered & open \\\\",
        "    \\midrule",
    ]
    for h in hist:
        rng = f"{h['length_from']} to {h['length_to']}" if h["length_to"] else f"{h['length_from']} or more"
        lines.append(f"    {rng} & {h['n']} & {h['recovered']} & {h['open']} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "  \\hspace{8mm}", "  \\begin{tabular}{@{}lllrrr@{}}", "    \\toprule",
              "    round & arm & cohort & runs & collapsed & never launched \\\\", "    \\midrule"]
    for a in out["by_arm"]:
        lines.append(f"    {a['round']} & \\texttt{{{a['arm'].replace('_', chr(92) + '_')}}} & {a['cohort']} & {a['runs']} & {a['collapsed']} & {a['never_launched']} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    with open(osp.join(TABLES, "collapse_census.tex"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
