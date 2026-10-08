# experiments/019-simb-multimodal/scripts/v21_readout.py
# [[experiments.019-simb-multimodal.scripts.v21_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/v21_readout
"""Read the v21 small-trunk round on Delta from W&B (project torchcell_019_expr_v21).

THE ROUND (delta_expr_v21_small.slurm stage `round`, arms in gh_expr_008_arm.sh, config
cgt_expr_v21_small.yaml): nine arms on the ProtT5-only trunk (width 90, six layers, batch
32, 1,200 epochs) on the both-label store, twelve split seeds each, four runs of one arm per
A40. `S_ref` is the config default; every other arm changes one thing. `S_prot` re-points
the head at protein abundance, so it is read on `val/proteome/pearson_per_feature` and is
not paired with `S_ref`.

WHICH RUN COUNTS. The project holds two attempts of most (arm, split) pairs: the first
launch of 2026-10-06 (killed inside its job by the in-place swap at 18:38 CT, or by hand
after collapsing) and the relaunch that followed. Per (arm, split) the run with the most
logged epochs is read; ties go to the later start. The number of attempts is recorded.

THE SCORE is the registered window: the mean of the head's validation Pearson per feature
over epochs 1,000 to 1,199, defined only for runs that reached epoch 1,199. A run LAUNCHES
at the first epoch its validation prediction spread ratio reaches 0.05 and COLLAPSED if the
ratio then stayed below 0.01 for 50 or more consecutive logged epochs (collapse_census.py).
Per arm the window is summarized over every finished run and over the runs that did not
collapse; the paired contrast against `S_ref` is the per-split difference of window means,
over the splits where both finished, and again over the splits where neither collapsed.
Twelve splits resolve a paired gap of about 0.02 (sd of the per-split differences over
root twelve); a smaller mean is "not resolved", not "null".

    python experiments/019-simb-multimodal/scripts/v21_readout.py

Writes results/v21_readout.json and notes-tex/figure-3-gate/tables/v21_round.tex.
"""

from __future__ import annotations

import json
import os.path as osp
import statistics as st
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import wandb

ENTITY = "zhao-group"
PROJECT = "torchcell_019_expr_v21"
WINDOW = (1000, 1199)
LAUNCH, DEAD, RULE = 0.05, 0.01, 50
# UTC. The in-place swap relaunched the 26 untouched packs from 18:38 CT on 2026-10-06.
RELAUNCH = "2026-10-06T23:38:00"
ARMS = {
    "S_ref": "reference: single expression head, mask off",
    "S_mask": "mask schedule 0, 10, 100, 1000",
    "S_sink": "trainable null sink in the perturbation head",
    "S_prop2": "two-hop gated perturbation propagation",
    "S_nodrop": "perturbation-head dropout 0",
    "S_stack": "composite stack: FUDT, CaLM, ProtT5",
    "S_basis64": "rank-64 response basis",
    "S_hadam": "Hadamard pair term replacing the additive one",
    "S_prot": "proteome head: protein abundance",
}
REPO = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
RESULTS = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")
TABLES = osp.join(REPO, "notes-tex", "figure-3-gate", "tables")


def read_run(run: Any) -> dict[str, Any]:
    tag = next(t for t in run.tags if t.startswith("S_"))
    arm, split = tag.rsplit("_s", 1)
    head = "proteome" if arm == "S_prot" else "expression"
    val, ratio = f"val/{head}/pearson_per_feature", f"val/{head}/pred_sd_ratio"
    rows: dict[int, dict[str, float]] = {}
    for key in (val, ratio, f"traineval/{head}/pearson_per_feature", "perf/epoch_seconds"):
        for x in run.history(keys=["epoch", key], pandas=False, samples=100000):
            if x.get(key) is not None:
                rows.setdefault(int(x["epoch"]), {})[key] = float(x[key])
    epochs = sorted(e for e in rows if val in rows[e])
    last = epochs[-1] if epochs else -1
    window = None
    if last >= WINDOW[1]:
        window = st.mean(rows[e][val] for e in epochs if WINDOW[0] <= e <= WINDOW[1])
    sp = sorted((e, rows[e][ratio]) for e in rows if ratio in rows[e])
    launch = next((e for e, v in sp if v >= LAUNCH), None)
    longest, n, start, dead_from = 0, 0, None, None
    for e, v in sp:
        if launch is not None and e > launch and v < DEAD:
            if n == 0:
                start = e
            n += 1
            if n > longest:
                longest, dead_from = n, start
        else:
            n = 0
    secs = [rows[e]["perf/epoch_seconds"] for e in rows if "perf/epoch_seconds" in rows[e] and e >= 20]
    te = [rows[e][f"traineval/{head}/pearson_per_feature"] for e in sorted(rows) if f"traineval/{head}/pearson_per_feature" in rows[e]]
    return {
        "id": run.id,
        "url": f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run.id}",
        "arm": arm,
        "split": int(split),
        "head": head,
        "state": run.state,
        "created_at": run.created_at,
        "cohort": "relaunched" if run.created_at >= RELAUNCH else "original",
        "last_epoch": last,
        "finished": last >= WINDOW[1],
        "launch_epoch": launch,
        "longest_dead_stretch": longest,
        "dead_from_epoch": dead_from,
        "collapsed": longest >= RULE,
        "window_mean": window,
        "pred_sd_ratio_last": sp[-1][1] if sp else None,
        "traineval_pearson_last": te[-1] if te else None,
        "epoch_seconds": st.mean(secs) if secs else None,
    }


def summarize(vals: list[float]) -> dict[str, Any]:
    return {"n": len(vals), "mean": st.mean(vals) if vals else None, "sd": st.stdev(vals) if len(vals) > 1 else None}


def main() -> None:
    api = wandb.Api(timeout=120)
    with ThreadPoolExecutor(12) as ex:
        every = list(ex.map(read_run, list(api.runs(f"{ENTITY}/{PROJECT}"))))
    chosen: dict[tuple[str, int], dict[str, Any]] = {}
    attempts: dict[tuple[str, int], int] = {}
    for r in every:
        k = (r["arm"], r["split"])
        attempts[k] = attempts.get(k, 0) + 1
        if k not in chosen or (r["last_epoch"], r["created_at"]) > (chosen[k]["last_epoch"], chosen[k]["created_at"]):
            chosen[k] = r
    for k, r in chosen.items():
        r["attempts"] = attempts[k]
    runs = sorted(chosen.values(), key=lambda r: (list(ARMS).index(r["arm"]), r["split"]))

    print(f"{'arm':<11}{'split':>5}{'cohort':>11}{'tries':>6}{'epoch':>6}{'launch':>7}{'dead':>5}{'from':>6}{'window':>8}{'sd':>7}{'train':>7}{'s/ep':>6}  state")
    for r in runs:
        f = lambda v, p=4, w=8: f"{v:>{w}.{p}f}" if v is not None else f"{'':>{w}}"
        print(
            f"{r['arm']:<11}{r['split']:>5}{r['cohort']:>11}{r['attempts']:>6}{r['last_epoch']:>6}{str(r['launch_epoch']):>7}"
            f"{r['longest_dead_stretch']:>5}{str(r['dead_from_epoch']):>6}{f(r['window_mean'])}{f(r['pred_sd_ratio_last'], 3, 7)}"
            f"{f(r['traineval_pearson_last'], 3, 7)}{f(r['epoch_seconds'], 0, 6)}  {r['state']}"
        )

    ref = {r["split"]: r for r in runs if r["arm"] == "S_ref"}
    arms = []
    for arm, desc in ARMS.items():
        rs = [r for r in runs if r["arm"] == arm]
        fin = [r for r in rs if r["finished"]]
        healthy = [r for r in fin if not r["collapsed"]]
        a: dict[str, Any] = {
            "arm": arm,
            "change": desc,
            "head": rs[0]["head"] if rs else None,
            "runs": len(rs),
            "finished": len(fin),
            "collapsed": sum(r["collapsed"] for r in rs),
            "never_launched": sum(r["launch_epoch"] is None for r in rs),
            "window_all": summarize([r["window_mean"] for r in fin]),
            "window_healthy": summarize([r["window_mean"] for r in healthy]),
            "epoch_seconds": st.mean(r["epoch_seconds"] for r in rs if r["epoch_seconds"]) if rs else None,
        }
        if arm not in ("S_ref", "S_prot"):
            both = {r["split"]: r["window_mean"] - ref[r["split"]]["window_mean"] for r in fin if r["split"] in ref and ref[r["split"]]["finished"]}
            clean = {s: d for s, d in both.items() if not chosen[(arm, s)]["collapsed"] and not ref[s]["collapsed"]}
            for name, dd in (("paired_all", both), ("paired_healthy", clean)):
                v = list(dd.values())
                a[name] = {**summarize(v), "n_positive": sum(x > 0 for x in v), "per_split": dd}
        arms.append(a)
        wa, wh = a["window_all"], a["window_healthy"]
        line = f"{arm:<11} finished {a['finished']:>2}/{a['runs']} collapsed {a['collapsed']:>2} window all {wa['mean']:.4f}±{wa['sd']:.4f} healthy {wh['mean']:.4f}±{wh['sd']:.4f} (n={wh['n']})" if wa["mean"] is not None and wh["n"] > 1 else f"{arm:<11} finished {a['finished']}/{a['runs']} collapsed {a['collapsed']}"
        if "paired_all" in a and a["paired_all"]["n"]:
            p, q = a["paired_all"], a["paired_healthy"]
            line += f" | vs S_ref all {p['mean']:+.4f} ({p['n_positive']}/{p['n']} up)"
            if q["n"]:
                line += f" healthy {q['mean']:+.4f} ({q['n_positive']}/{q['n']} up)"
        print(line)

    base = json.load(open(osp.join(RESULTS, "baselines_both_label.json")))["rows"]
    bar = {h: max(r["val_mean"] for r in base if r["head"] == h) for h in ("expression", "proteome")}
    print("baseline bar (best validation mean per head, baselines_both_label.json):", bar)

    with open(osp.join(RESULTS, "v21_readout.json"), "w") as f:
        json.dump(
            {
                "generated_by": "experiments/019-simb-multimodal/scripts/v21_readout.py",
                "window": WINDOW,
                "rule": {"launch": LAUNCH, "dead": DEAD, "collapse_epochs": RULE, "relaunch_utc": RELAUNCH},
                "baseline_bar_val": bar,
                "arms": arms,
                "runs": runs,
            },
            f,
            indent=1,
        )

    best = max((a for a in arms if a["head"] == "expression" and a["window_healthy"]["n"]), key=lambda a: a["window_healthy"]["mean"])
    lines = [
        "%% GENERATED by experiments/019-simb-multimodal/scripts/v21_readout.py. Do not edit by hand.",
        "%% SOURCE: W&B zhao-group/torchcell_019_expr_v21 run histories (Delta jobs 22690625 to 22690651, relaunched packs 1 to 26)",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\footnotesize",
        "  \\caption[The v21 round on twelve split seeds]{The v21 small-trunk round: nine arms, twelve split"
        " seeds, 1{,}200 epochs at batch 32 on the both-label store. Window Pearson is the mean validation"
        " Pearson per feature over epochs 1{,}000 to 1{,}199 (the proteome head for \\texttt{S\\_prot}, the"
        " expression head otherwise), mean $\\pm$ sd over the finished runs and over the runs that did not"
        " collapse (a dead stretch of 50 or more epochs with the prediction spread ratio below 0.01). The"
        " paired column is the per-split difference against \\texttt{S\\_ref}, mean and number of splits"
        " above zero, over the splits where neither run collapsed. The best healthy expression mean is bold."
        f" The sequence-baseline bar on this store is {bar['expression']:.3f} (expression) and"
        f" {bar['proteome']:.3f} (proteome), the neighbor average on ProtT5 (Table~\\ref{{tab:baselines-both-label}})."
        " \\src{experiments/019-simb-multimodal/scripts/v21_readout.py}}",
        "  \\label{tab:v21-round}",
        "  \\begin{tabular}{@{}lp{36mm}rrllr@{}}",
        "    \\toprule",
        "    arm & change & finished & collapsed & window Pearson, all & window Pearson, healthy (n) & paired vs ref, healthy \\\\",
        "    \\midrule",
    ]
    for a in arms:
        wa, wh = a["window_all"], a["window_healthy"]
        cell_a = f"{wa['mean']:.3f} $\\pm$ {wa['sd']:.3f}" if wa["sd"] is not None else "--"
        cell_h = f"{wh['mean']:.3f} $\\pm$ {wh['sd']:.3f} ({wh['n']})" if wh["sd"] is not None else "--"
        if a is best:
            cell_h = f"\\textbf{{{cell_h}}}"
        p = a.get("paired_healthy")
        cell_p = f"{p['mean']:+.3f} ({p['n_positive']}/{p['n']})" if p and p["n"] else "--"
        lines.append(f"    \\texttt{{{a['arm'].replace('_', chr(92) + '_')}}} & {a['change']} & {a['finished']}/{a['runs']} & {a['collapsed']} & {cell_a} & {cell_h} & {cell_p} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    with open(osp.join(TABLES, "v21_round.tex"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
