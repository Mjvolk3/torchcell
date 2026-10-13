# experiments/019-simb-multimodal/scripts/morph_v25_readout.py
# [[experiments.019-simb-multimodal.scripts.morph_v25_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/morph_v25_readout
"""Readout of the inductive morphology round v25 (project torchcell_019_morph_v25).

A CELL is an arm (M_cls, M_pool, M_zs, M_xattn) plus the recipe tags the launcher wrote
(lr, mse, crps, pearson_mse, graphkl, restarts, warmup, stack); the runs of a cell differ
by split seed and initialization seed. Two statistics per run, as the gate records them:

  window      the registered window, mean of val/morphology/pearson_per_feature over
              epochs 400 to 499 (only for a run that reached epoch 499; a run short of
              it is PARTIAL and reports the mean over its last 200 epochs under
              `partial_window`, never under `window`); `window_late` is the 1,000 to
              1,199 mean of the first wave's 1,200-epoch runs;
  escape      the first epoch at which traineval/morphology/pearson_per_feature exceeds
              0.3, the plateau-escape criterion of the round (None if never), and the
              training fit at the last logged epoch.

Per cell: n runs, n finished, window mean and s.d., escape fraction, train fit. Writes
results/morph_v25_readout.json and notes-tex/figure-3-gate/tables/morph_v25.tex.

    python experiments/019-simb-multimodal/scripts/morph_v25_readout.py
"""

from __future__ import annotations

import json
import os.path as osp
import re
import statistics as st
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import wandb

ENTITY = "zhao-group"
PROJECT = "torchcell_019_morph_v25"
VAL = "val/morphology/pearson_per_feature"
TRAIN = "traineval/morphology/pearson_per_feature"
# The registered window of v25 (declared 2026-10-12 before any cell was read on it): the
# loss finishes learning by epoch 50, an escape happens by epoch 400, and an escaped
# run's validation peak sits at epochs 400 to 600 before it overfits, so the budget is
# 500 epochs and the window the last 100. The 1,200-epoch cells of the first wave also
# report their 1,000 to 1,199 mean (`window_late`) as the check that both rank alike.
WINDOW = (400, 499)
WINDOW_LATE = (1000, 1199)
# The lr 1e-4 cells were still rising at epoch 499 (+0.037 validation Pearson over the
# last 200 epochs, read 2026-10-12), so the v26b cells at that lr run 1,000 epochs and
# also report their 900 to 999 mean (`window_1000`, declared before any such run began);
# their 400 to 499 mean stays the number compared with the rest of the round.
WINDOW_1000 = (900, 999)
ESCAPE = 0.3
# The launcher's SWEEP_OV tags replace the arm script's, so a v25 run carries the arm
# (M_cls), the split (split0) and the init seed (init0) as separate tags.
ARM_TAGS = ("M_cls", "M_pool", "M_zs", "M_xattn", "MP_cls")
SPLIT_RE = re.compile(r"split(\d+)")
VARIANT_RE = re.compile(r"lr[0-9e.-]+|b\d+|mse|crps|pearson_mse|graphkl[0-9e.-]+|restarts\d+|warmup\d+|stack|align[0-9e.-]+|src[a-z_]+|nocenter")
REPO = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
RESULTS = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")
TABLES = osp.join(REPO, "notes-tex", "figure-3-gate", "tables")


def series(run: Any, key: str) -> list[tuple[int, float]]:
    return [
        (int(x["epoch"]), float(x[key]))
        for x in run.history(keys=["epoch", key], pandas=False, samples=100000)
        if x.get(key) is not None
    ]


def read_run(run: Any) -> dict[str, Any] | None:
    arm = next((t for t in run.tags if t in ARM_TAGS), None)
    sm = next((SPLIT_RE.fullmatch(t) for t in run.tags if SPLIT_RE.fullmatch(t)), None)
    if arm is None or sm is None:
        return None
    split = int(sm.group(1))
    variants = sorted(t for t in run.tags if VARIANT_RE.fullmatch(t))
    cell = "_".join([arm, *variants])
    val = series(run, VAL)
    train = series(run, TRAIN)
    last = max((e for e, _ in val), default=-1)
    finished = last >= WINDOW[1]
    in_window = [v for e, v in val if WINDOW[0] <= e <= WINDOW[1]]
    late = [v for e, v in val if WINDOW_LATE[0] <= e <= WINDOW_LATE[1]]
    tail = [v for e, v in val if e > last - 200]
    escape = next((e for e, v in train if v > ESCAPE), None)
    # Saturation: the validation score at the ladder epochs, the change over the last 200
    # epochs (mean of the last 100 minus the mean of the 100 before), and the same for the
    # training fit. A run whose last-200 change is still above the seed spread has not
    # saturated and would score higher with a longer budget.
    ladder = {str(k): st.mean([v for e, v in val if k - 10 < e <= k + 10]) for k in (200, 400, 600, 800, 1000, 1199) if any(k - 10 < e <= k + 10 for e, _ in val)}
    def change(xs: list[tuple[int, float]]) -> float | None:
        a = [v for e, v in xs if last - 200 < e <= last - 100]
        b = [v for e, v in xs if last - 100 < e <= last]
        return (st.mean(b) - st.mean(a)) if a and b else None
    return {
        "ladder": ladder,
        "val_change_last200": change(val),
        "train_change_last200": change(train),
        "id": run.id,
        "url": run.url,
        "cell": cell,
        "arm": arm,
        "variants": variants,
        "split": split,
        "init_seed": int(run.config.get("seed", 0)),
        "last_epoch": last,
        "finished": finished,
        "window": st.mean(in_window) if finished and in_window else None,
        "window_late": st.mean(late) if last >= WINDOW_LATE[1] and late else None,
        "window_1000": st.mean(w1k) if last >= WINDOW_1000[1] and (w1k := [v for e, v in val if WINDOW_1000[0] <= e <= WINDOW_1000[1]]) else None,
        "partial_window": None if finished else (st.mean(tail) if tail else None),
        "roll_max": max((v for _, v in val), default=None),
        "train_last": train[-1][1] if train else None,
        "escape_epoch": escape,
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    fin = [r for r in runs if r["finished"]]
    w = [r["window"] for r in fin]
    esc = [r["escape_epoch"] is not None for r in runs]
    return {
        "n": len(runs),
        "n_finished": len(fin),
        "window_mean": st.mean(w) if w else None,
        "window_sd": st.stdev(w) if len(w) > 1 else None,
        "escape_fraction": sum(esc) / len(esc) if esc else None,
        "n_escaped": sum(esc),
        "train_last_mean": st.mean(tl) if (tl := [r["train_last"] for r in runs if r["train_last"] is not None]) else None,
        "partial_window_mean": st.mean([r["partial_window"] for r in runs if r["partial_window"] is not None]) if any(r["partial_window"] is not None for r in runs) else None,
    }


def table(cells: dict[str, dict[str, Any]]) -> str:
    rows = sorted(cells.items(), key=lambda kv: -(kv[1]["window_mean"] if kv[1]["window_mean"] is not None else -1))
    best = max((c["window_mean"] for c in cells.values() if c["window_mean"] is not None), default=None)
    out = [
        "%% GENERATED by experiments/019-simb-multimodal/scripts/morph_v25_readout.py. Do not edit by hand.",
        "%% SOURCE: experiments/019-simb-multimodal/results/morph_v25_readout.json",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\footnotesize",
        "  \\caption[Round v25, inductive morphology]{Round v25 by cell (arm plus recipe), genotype only on "
        "fig3_core, 116 moving features. The window is the mean validation Pearson per feature over "
        "epochs 400 to 499 of finished runs (the 1{,}200-epoch runs of the first wave also carry their 1{,}000 to 1{,}199 mean in the json); the escape fraction is the share of runs whose "
        "training Pearson exceeded 0.3 at any epoch; the training fit is at the last logged epoch. A "
        "cell with unfinished runs is partial. Bold, the best finished window mean. "
        "\\src{experiments/019-simb-multimodal/scripts/morph_v25_readout.py}}",
        "  \\label{tab:morph-v25}",
        "  \\begin{tabular}{@{}lrrrrr@{}}",
        "    \\toprule",
        "    cell & runs (n) & finished (n) & window mean val Pearson per feature (s.d.) & escape fraction & train Pearson per feature, last epoch (mean) \\\\",
        "    \\midrule",
    ]
    for name, c in rows:
        w = "partial" if c["window_mean"] is None else f"{c['window_mean']:.3f}" + (f" ({c['window_sd']:.3f})" if c["window_sd"] is not None else "")
        if best is not None and c["window_mean"] == best:
            w = f"\\textbf{{{w}}}"
        esc = f"{c['n_escaped']} of {c['n']}"
        tr = "" if c["train_last_mean"] is None else f"{c['train_last_mean']:.2f}"
        out.append(f"    \\texttt{{{name.replace('_', chr(92) + '_')}}} & {c['n']} & {c['n_finished']} & {w} & {esc} & {tr} \\\\")
    out += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    return "\n".join(out)


def main() -> None:
    api = wandb.Api(timeout=120)
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", per_page=500))
    with ThreadPoolExecutor(8) as pool:
        rows = [r for r in pool.map(read_run, runs) if r is not None]
    # A run with no validation epoch is a launch that was cancelled before its first
    # epoch (the Delta packs resubmitted at the 500-epoch budget left twelve such
    # records); it is not a partial run and counts in no cell.
    rows = [r for r in rows if r["last_epoch"] >= 0]
    cells: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        cells.setdefault(r["cell"], []).append(r)
    summary = {name: summarize(rs) for name, rs in cells.items()}
    for name in sorted(summary, key=lambda n: -(summary[n]["window_mean"] or -1)):
        c = summary[name]
        fin = [r for r in cells[name] if r["finished"]]
        c["val_change_last200_mean"] = st.mean([r["val_change_last200"] for r in fin if r["val_change_last200"] is not None]) if fin else None
        c["train_change_last200_mean"] = st.mean([r["train_change_last200"] for r in fin if r["train_change_last200"] is not None]) if fin else None
        w = "partial" if c["window_mean"] is None else f"{c['window_mean']:.3f} +/- {c['window_sd'] if c['window_sd'] is not None else 0:.3f}"
        dv = c["val_change_last200_mean"]; dt = c["train_change_last200_mean"]
        print(f"{name:32s} n {c['n']:2d} fin {c['n_finished']:2d} window {w:18s} escape {c['n_escaped']}/{c['n']} train {c['train_last_mean'] if c['train_last_mean'] is None else round(c['train_last_mean'], 3)}  last200 dval {dv if dv is None else f'{dv:+.3f}'} dtrain {dt if dt is None else f'{dt:+.3f}'}")
    out = {"generated_by": "experiments/019-simb-multimodal/scripts/morph_v25_readout.py", "project": PROJECT, "window": WINDOW, "escape_threshold": ESCAPE, "cells": summary, "runs": rows}
    with open(osp.join(RESULTS, "morph_v25_readout.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(osp.join(TABLES, "morph_v25.tex"), "w") as f:
        f.write(table(summary))


if __name__ == "__main__":
    main()
