# experiments/019-simb-multimodal/scripts/v22_readout.py
# [[experiments.019-simb-multimodal.scripts.v22_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/v22_readout
"""Read the v22 fast-model round from W&B (project torchcell_019_expr_v22).

Per run: the epoch reached, the trailing 20-epoch mean of
`val/expression/pearson_per_feature` at a ladder of epochs (so arms with different
batch sizes are compared at matched epochs AND matched optimizer steps), the registered
window mean (epochs 1,000 to 1,200, only if the run reached epoch 1,199), the prediction
spread ratio, the normalized squared error and the eval-mode train Pearson at the last
logged epoch, and seconds per epoch.

Per arm against the reference: the paired difference of the window mean by split seed,
against F_ref of this round where that split seed exists, and reported separately against
v21's S_ref (project torchcell_019_expr_v21, same config and split seeds, the pre-speedup
code on Delta) when `--delta-ref` is given.

A run that has not reached epoch 1,199 has no window score and is reported as PARTIAL at
its ladder epochs only.

    python experiments/019-simb-multimodal/scripts/v22_readout.py [--delta-ref]

Writes results/v22_readout.json.
"""
from __future__ import annotations

import argparse
import json
import os.path as osp
import statistics as st

import wandb

ENTITY = "zhao-group"
PROJECT = "torchcell_019_expr_v22"
DELTA_PROJECT = "torchcell_019_expr_v21"
VAL = "val/expression/pearson_per_feature"
LADDER = [100, 200, 300, 400, 600, 799, 1000, 1199]
# Paired contrasts beyond "every arm against F_ref": (arm, reference, score). `window` is
# the registered window mean; a number is the ladder epoch (trailing 20-epoch mean), used
# for the 800-epoch regularization arms.
EXTRA_CONTRASTS = [
    ("F_b128wu_hadam", "F_b128wu", "window"),
    ("F_b128wu_wd", "F_b128wu", "799"),
    ("F_b128wu_drop", "F_b128wu", "799"),
    ("F_b128wu_wd", "F_b128wu", "400"),
    ("F_b128wu_drop", "F_b128wu", "400"),
    ("F_b128lr1", "F_ref", "1000"),
    ("F_b128wu", "F_ref", "1000"),
]
WINDOW = (1000, 1199)
KEYS = [
    "epoch",
    VAL,
    "val/expression/pred_sd_ratio",
    "val/expression/nmse",
    "traineval/expression/pearson_per_feature",
    "perf/epoch_seconds",
    "trainer/global_step",
]
RESULTS = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")


def read_run(run: wandb.apis.public.Run) -> dict:
    rows: dict[int, dict] = {}
    for key in KEYS[1:]:
        for x in run.history(keys=["epoch", key], pandas=False, samples=100000):
            if x.get(key) is not None:
                rows.setdefault(int(x["epoch"]), {})[key] = float(x[key])
    epochs = sorted(e for e in rows if VAL in rows[e])
    last = epochs[-1] if epochs else -1

    def trailing(e: int) -> float | None:
        vals = [rows[k][VAL] for k in epochs if e - 19 <= k <= e]
        return st.mean(vals) if e <= last and len(vals) >= 10 else None

    window = None
    if last >= WINDOW[1]:
        window = st.mean(rows[k][VAL] for k in epochs if WINDOW[0] <= k <= WINDOW[1])

    def latest(key: str) -> float | None:
        vals = [rows[e][key] for e in sorted(rows) if key in rows[e]]
        return vals[-1] if vals else None

    secs = [rows[e]["perf/epoch_seconds"] for e in rows if "perf/epoch_seconds" in rows[e] and e >= 20]
    # LAUNCH AND COLLAPSE from the prediction spread ratio (sd of predictions over sd of
    # targets across validation strains). A head LAUNCHES at the first epoch the ratio
    # reaches 0.05 (the round's launch gate). A run COLLAPSED if, after launching, the
    # ratio stayed below 0.01 for at least 50 consecutive logged epochs (the count is
    # stored; the threshold is applied in v22_tables.py): the per-gene-mean
    # predictor. A single-epoch dip is not a collapse.
    sp = sorted(
        (e, rows[e]["val/expression/pred_sd_ratio"])
        for e in rows
        if "val/expression/pred_sd_ratio" in rows[e]
    )
    launch_epoch = next((e for e, v in sp if v >= 0.05), None)
    longest_dead, run_len, dead_from = 0, 0, None
    start = None
    for e, v in sp:
        if launch_epoch is not None and e > launch_epoch and v < 0.01:
            if run_len == 0:
                start = e
            run_len += 1
            if run_len > longest_dead:
                longest_dead, dead_from = run_len, start
        else:
            run_len = 0
    return {
        "id": run.id,
        "state": run.state,
        "max_epochs": int(run.config["trainer"]["max_epochs"]),
        "launch_epoch": launch_epoch,
        "longest_dead_stretch": longest_dead,
        "dead_from_epoch": dead_from,
        "last_epoch": last,
        "ladder": {str(e): trailing(e) for e in LADDER},
        "window_mean": window,
        "roll_max": max((trailing(e) or -1.0) for e in epochs) if epochs else None,
        "pred_sd_ratio_last": latest("val/expression/pred_sd_ratio"),
        "nmse_last": latest("val/expression/nmse"),
        "traineval_pearson_last": latest("traineval/expression/pearson_per_feature"),
        "epoch_seconds": st.mean(secs) if secs else None,
        "global_step_last": latest("trainer/global_step"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-ref", action="store_true")
    args = parser.parse_args()
    api = wandb.Api()
    runs: dict[str, dict[int, dict]] = {}
    for run in api.runs(f"{ENTITY}/{PROJECT}"):
        arm = next(t for t in run.tags if t.startswith("F_"))
        split = int(next(t for t in run.tags if t.startswith("split")).removeprefix("split"))
        runs.setdefault(arm, {})[split] = read_run(run)
    if args.delta_ref:
        for run in api.runs(f"{ENTITY}/{DELTA_PROJECT}"):
            tags = [t for t in run.tags if t.startswith("S_ref_s")]
            if tags:
                split = int(tags[0].removeprefix("S_ref_s"))
                runs.setdefault("S_ref_delta", {})[split] = read_run(run)

    header = f"{'arm':<12}{'split':>5}{'epoch':>6}" + "".join(f"{e:>8}" for e in LADDER)
    print(header + f"{'window':>9}{'sd':>7}{'nmse':>7}{'train':>7}{'s/ep':>6}")
    for arm in sorted(runs):
        for split in sorted(runs[arm]):
            r = runs[arm][split]
            cells = "".join(
                f"{r['ladder'][str(e)]:>8.4f}" if r["ladder"][str(e)] is not None else f"{'':>8}"
                for e in LADDER
            )

            def fmt(v: float | None, w: int, p: int) -> str:
                return f"{v:>{w}.{p}f}" if v is not None else f"{'':>{w}}"

            print(
                f"{arm:<12}{split:>5}{r['last_epoch']:>6}{cells}"
                f"{fmt(r['window_mean'], 9, 4)}{fmt(r['pred_sd_ratio_last'], 7, 3)}"
                f"{fmt(r['nmse_last'], 7, 3)}{fmt(r['traineval_pearson_last'], 7, 3)}"
                f"{fmt(r['epoch_seconds'], 6, 1)}"
            )

    contrasts: dict[str, dict] = {}
    for ref in ("F_ref", "S_ref_delta"):
        if ref not in runs:
            continue
        for arm in sorted(runs):
            if arm == ref or arm == "S_ref_delta":
                continue
            diffs = {
                s: runs[arm][s]["window_mean"] - runs[ref][s]["window_mean"]
                for s in sorted(runs[arm])
                if s in runs[ref]
                and runs[arm][s]["window_mean"] is not None
                and runs[ref][s]["window_mean"] is not None
            }
            if not diffs:
                continue
            vals = list(diffs.values())
            contrasts[f"{arm} - {ref}"] = {
                "per_split": diffs,
                "mean": st.mean(vals),
                "sd": st.stdev(vals) if len(vals) > 1 else None,
                "n_positive": sum(v > 0 for v in vals),
                "n": len(vals),
            }
            print(
                f"{arm} - {ref}: mean {st.mean(vals):+.4f} over {len(vals)} split seeds, "
                f"{sum(v > 0 for v in vals)} positive, per split "
                + ", ".join(f"{s}: {v:+.4f}" for s, v in diffs.items())
            )
    for arm, ref, score in EXTRA_CONTRASTS:
        if arm not in runs or ref not in runs:
            continue

        def val(r: dict, score: str = score) -> float | None:
            return r["window_mean"] if score == "window" else r["ladder"][score]

        diffs = {
            sp: val(runs[arm][sp]) - val(runs[ref][sp])
            for sp in sorted(runs[arm])
            if sp in runs[ref] and val(runs[arm][sp]) is not None and val(runs[ref][sp]) is not None
        }
        if not diffs:
            continue
        vals = list(diffs.values())
        key = f"{arm} - {ref} @ {score}"
        contrasts[key] = {
            "per_split": diffs,
            "mean": st.mean(vals),
            "n_positive": sum(v > 0 for v in vals),
            "n": len(vals),
        }
        print(
            f"{key}: mean {st.mean(vals):+.4f} over {len(vals)} split seeds, "
            f"{sum(v > 0 for v in vals)} positive, per split "
            + ", ".join(f"{sp}: {v:+.4f}" for sp, v in diffs.items())
        )
    with open(osp.join(RESULTS, "v22_readout.json"), "w") as f:
        json.dump(
            {
                "generated_by": "experiments/019-simb-multimodal/scripts/v22_readout.py",
                "window": WINDOW,
                "ladder_statistic": "mean of val/expression/pearson_per_feature over the 20 epochs ending at the ladder epoch",
                "runs": runs,
                "contrasts": contrasts,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
