# experiments/019-simb-multimodal/scripts/morph_v24_readout.py
# [[experiments.019-simb-multimodal.scripts.morph_v24_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/morph_v24_readout
"""Read the morphology round v24 from W&B (project torchcell_019_morph_v24).

THE ROUND (conf/cgt_morph_v24.yaml): the 116 moving CalMorph features read off the
perturbed CLS, on fig3_core. Arms: `M_cls` (genotype only, every calmorph genotype),
`M_pool` (CLS plus the mean gene pool, three seeds), `CM_expr` (the strain's expression
revealed in full as input, the strains carrying both labels) and `CM_exprperm` (another
strain's expression, the control). Twelve split seeds for the cabbi arms.

THE SCORE is the registered window, the mean of `val/morphology/pearson_per_feature` over
epochs 1,000 to 1,199, only for runs that reached 1,199; the roll-max beside it is an
upward-biased order statistic and is printed for reference only. A run COLLAPSED if, after
its spread ratio first reached 0.05, it stayed below 0.01 for 50 consecutive logged
epochs (the rule of every small-trunk readout). Two paired contrasts, by split seed:
`CM_expr - CM_exprperm` (does the strain's OWN expression help, on the same strains) and
`M_pool - M_cls` (does the gene pool add to the CLS). `CM_expr - M_cls` is NOT paired here:
the two arms are validated on different strain sets (every calmorph strain against the
strains carrying both labels), so that difference is reported as a difference of means
with that caveat.

The context columns come from committed files: the replicate ceiling per feature
(results/morphology_feature_ceiling.csv, mean sqrt reliability over the moving set) and
the sequence baselines on fig3_core (results/morphology_baselines_split_fig3_core/seed*.json,
best key selected on validation, all 278 features).

    python experiments/019-simb-multimodal/scripts/morph_v24_readout.py

Writes results/morph_v24_readout.json and notes-tex/figure-3-gate/tables/morph_v24.tex.
"""

from __future__ import annotations

import glob
import json
import math
import os.path as osp
import statistics as st
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
import wandb

ENTITY = "zhao-group"
PROJECT = "torchcell_019_morph_v24"
VAL = "val/morphology/pearson_per_feature"
RATIO = "val/morphology/pred_sd_ratio"
WINDOW = (1000, 1199)
LAUNCH, DEAD, RULE = 0.05, 0.01, 50
ARMS = {
    "M_cls": "genotype only, CLS readout, all calmorph strains",
    "M_pool": "genotype only, CLS plus gene pool, all calmorph strains",
    "CM_expr": "the strain's expression revealed, both-label strains",
    "CM_exprperm": "another strain's expression revealed, both-label strains",
}
REPO = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
RESULTS = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")
TABLES = osp.join(REPO, "notes-tex", "figure-3-gate", "tables")


def read_run(run: Any) -> dict[str, Any]:
    arm = next(t for t in run.tags if t in ARMS or t.rsplit("_s", 1)[0] in ARMS)
    arm, split = arm.rsplit("_s", 1)
    rows: dict[int, dict[str, float]] = {}
    for key in (VAL, RATIO, "val/loss", "perf/epoch_seconds", "traineval/morphology/pearson_per_feature"):
        for x in run.history(keys=["epoch", key], pandas=False, samples=100000):
            if x.get(key) is not None:
                rows.setdefault(int(x["epoch"]), {})[key] = float(x[key])
    ep = sorted(e for e in rows if VAL in rows[e])
    last = ep[-1] if ep else -1
    window = st.mean(rows[e][VAL] for e in ep if WINDOW[0] <= e <= WINDOW[1]) if last >= WINDOW[1] else None

    def trailing(e: int) -> float | None:
        v = [rows[k][VAL] for k in ep if e - 19 <= k <= e]
        return st.mean(v) if len(v) >= 10 else None

    roll = max((trailing(e) or -1.0) for e in ep) if ep else None
    sp = sorted((e, rows[e][RATIO]) for e in rows if RATIO in rows[e])
    launch = next((e for e, v in sp if v >= LAUNCH), None)
    longest = n = 0
    for e, v in sp:
        if launch is not None and e > launch and v < DEAD:
            n += 1
            longest = max(longest, n)
        else:
            n = 0
    loss = [(e, rows[e]["val/loss"]) for e in sorted(rows) if "val/loss" in rows[e]]
    secs = [rows[e]["perf/epoch_seconds"] for e in rows if "perf/epoch_seconds" in rows[e] and e >= 20]
    te = [rows[e]["traineval/morphology/pearson_per_feature"] for e in sorted(rows) if "traineval/morphology/pearson_per_feature" in rows[e]]
    return {
        "id": run.id,
        "url": f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run.id}",
        "arm": arm,
        "split": int(split),
        "state": run.state,
        "last_epoch": last,
        "finished": last >= WINDOW[1],
        "launch_epoch": launch,
        "longest_dead_stretch": longest,
        "collapsed": longest >= RULE,
        "window_mean": window,
        "roll_max": roll,
        "pred_sd_ratio_last": sp[-1][1] if sp else None,
        "loss_min": min(loss, key=lambda t: t[1])[1] if loss else None,
        "loss_min_epoch": min(loss, key=lambda t: t[1])[0] if loss else None,
        "loss_last": loss[-1][1] if loss else None,
        "traineval_pearson_last": te[-1] if te else None,
        "epoch_seconds": st.mean(secs) if secs else None,
    }


def summarize(v: list[float]) -> dict[str, Any]:
    return {"n": len(v), "mean": st.mean(v) if v else None, "sd": st.stdev(v) if len(v) > 1 else None}


def paired(a: dict[int, dict], b: dict[int, dict]) -> dict[str, Any]:
    d = {s: a[s]["window_mean"] - b[s]["window_mean"] for s in sorted(a) if s in b and a[s]["window_mean"] is not None and b[s]["window_mean"] is not None}
    v = list(d.values())
    return {**summarize(v), "n_positive": sum(x > 0 for x in v), "per_split": d}


def context() -> dict[str, Any]:
    ceil = pd.read_csv(osp.join(RESULTS, "morphology_feature_ceiling.csv")).rename(columns={"Unnamed: 0": "feature"})
    moving = ceil[ceil["reliability"] >= 0.5]
    out: dict[str, Any] = {
        "ceiling_moving_mean": float(moving["ceiling"].mean()),
        "ceiling_all_mean": float(ceil["ceiling"].mean()),
        "n_moving": int(len(moving)),
    }
    files = sorted(glob.glob(osp.join(RESULTS, "morphology_baselines_split_fig3_core", "seed*.json")))
    for B in ("B2_bilinear", "B3_neighbor_average"):
        vals = []
        for f in files:
            j = json.load(open(f))
            best = j[B]["best_embedding"]
            vals.append(j[B]["by_embedding"][best]["selected_on_val"]["val_pearson_per_feature"])
        out[f"baseline_{B}_val_all278"] = summarize(vals)
    return out


def main() -> None:
    api = wandb.Api(timeout=120)
    with ThreadPoolExecutor(12) as ex:
        every = list(ex.map(read_run, list(api.runs(f"{ENTITY}/{PROJECT}"))))
    chosen: dict[tuple[str, int], dict] = {}
    for r in every:
        k = (r["arm"], r["split"])
        if k not in chosen or r["last_epoch"] > chosen[k]["last_epoch"]:
            chosen[k] = r
    runs = sorted(chosen.values(), key=lambda r: (list(ARMS).index(r["arm"]), r["split"]))
    by: dict[str, dict[int, dict]] = {}
    for r in runs:
        by.setdefault(r["arm"], {})[r["split"]] = r

    f = lambda v, p=3, w=8: f"{v:>{w}.{p}f}" if v is not None else f"{'':>{w}}"
    print(f"{'arm':<12}{'split':>5}{'id':>10}{'epoch':>6}{'launch':>7}{'dead':>5}{'window':>8}{'rollmax':>8}{'sd':>7}{'lossmin':>9}{'@ep':>6}{'train':>7}{'s/ep':>6}")
    for r in runs:
        print(f"{r['arm']:<12}{r['split']:>5}{r['id']:>10}{r['last_epoch']:>6}{str(r['launch_epoch']):>7}{r['longest_dead_stretch']:>5}{f(r['window_mean'])}{f(r['roll_max'])}{f(r['pred_sd_ratio_last'], 3, 7)}{f(r['loss_min'], 4, 9)}{str(r['loss_min_epoch']):>6}{f(r['traineval_pearson_last'], 3, 7)}{f(r['epoch_seconds'], 1, 6)}")

    arms = []
    for arm, desc in ARMS.items():
        rs = list(by.get(arm, {}).values())
        fin = [r for r in rs if r["finished"]]
        arms.append({"arm": arm, "change": desc, "runs": len(rs), "finished": len(fin), "collapsed": sum(r["collapsed"] for r in rs),
                     "window": summarize([r["window_mean"] for r in fin]), "roll_max": summarize([r["roll_max"] for r in fin if r["roll_max"] is not None])})
    contrasts = {
        "CM_expr - CM_exprperm": paired(by.get("CM_expr", {}), by.get("CM_exprperm", {})),
        "M_pool - M_cls": paired(by.get("M_pool", {}), by.get("M_cls", {})),
    }
    ctx = context()
    print()
    for a in arms:
        w = a["window"]
        print(f"{a['arm']:<12} finished {a['finished']:>2}/{a['runs']} collapsed {a['collapsed']} window {f(w['mean'])} ± {f(w['sd'])}")
    for k, c in contrasts.items():
        print(f"{k}: mean {c['mean']:+.4f} over {c['n']} splits, {c['n_positive']} positive" if c["n"] else f"{k}: no pairs")
    print("context:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ctx.items() if not isinstance(v, dict)}, {k: round(v["mean"], 3) for k, v in ctx.items() if isinstance(v, dict) and v["mean"] is not None})

    with open(osp.join(RESULTS, "morph_v24_readout.json"), "w") as fh:
        json.dump({"generated_by": "experiments/019-simb-multimodal/scripts/morph_v24_readout.py", "window": WINDOW,
                   "rule": {"launch": LAUNCH, "dead": DEAD, "collapse_epochs": RULE}, "context": ctx, "arms": arms,
                   "contrasts": contrasts, "runs": runs}, fh, indent=1)

    lines = [
        "%% GENERATED by experiments/019-simb-multimodal/scripts/morph_v24_readout.py. Do not edit by hand.",
        "%% SOURCE: W&B zhao-group/torchcell_019_morph_v24 run histories (IGB jobs 2427784 and 2427785);",
        "%% results/morphology_feature_ceiling.csv; results/morphology_baselines_split_fig3_core/seed*.json",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\footnotesize",
        "  \\caption[The morphology round v24]{The morphology round v24: the 116 moving CalMorph features read off"
        " the perturbed CLS, on fig3\\_core, twelve split seeds (three for \\texttt{M\\_pool}). Window Pearson is"
        " the mean validation Pearson per feature over epochs 1{,}000 to 1{,}199, mean $\\pm$ sd over the finished"
        " runs; the paired column is the per-split difference against the arm named, mean and number of splits"
        " above zero. \\texttt{M\\_cls} and \\texttt{M\\_pool} are validated on every calmorph strain, the"
        " \\texttt{CM} arms on the strains carrying both labels, so only the two contrasts shown are paired."
        f" The replicate ceiling of the moving set is {ctx['ceiling_moving_mean']:.3f}; the sequence baselines on all"
        f" 278 features (four split seeds) are {ctx['baseline_B2_bilinear_val_all278']['mean']:.3f} (bilinear ridge) and"
        f" {ctx['baseline_B3_neighbor_average_val_all278']['mean']:.3f} (neighbor average)."
        " \\src{experiments/019-simb-multimodal/scripts/morph_v24_readout.py}}",
        "  \\label{tab:morph-v24}",
        "  \\begin{tabular}{@{}lp{52mm}rrlr@{}}",
        "    \\toprule",
        "    arm & change & finished & collapsed & window Pearson & paired difference \\\\",
        "    \\midrule",
    ]
    pair_of = {"CM_expr": ("CM_expr - CM_exprperm", "vs permuted"), "M_pool": ("M_pool - M_cls", "vs CLS alone")}
    best = max((a for a in arms if a["window"]["n"]), key=lambda a: a["window"]["mean"])
    for a in arms:
        w = a["window"]
        cell = f"{w['mean']:.3f} $\\pm$ {w['sd']:.3f}" if w["sd"] is not None else ("--" if w["mean"] is None else f"{w['mean']:.3f}")
        if a is best:
            cell = f"\\textbf{{{cell}}}"
        p = pair_of.get(a["arm"])
        c = contrasts[p[0]] if p else None
        pc = f"{c['mean']:+.3f} ({c['n_positive']}/{c['n']}) {p[1]}" if c and c["n"] else "--"
        lines.append(f"    \\texttt{{{a['arm'].replace('_', chr(92) + '_')}}} & {a['change']} & {a['finished']}/{a['runs']} & {a['collapsed']} & {cell} & {pc} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    with open(osp.join(TABLES, "morph_v24.tex"), "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
