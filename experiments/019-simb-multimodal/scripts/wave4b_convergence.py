# experiments/019-simb-multimodal/scripts/wave4b_convergence.py
# [[experiments.019-simb-multimodal.scripts.wave4b_convergence]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/wave4b_convergence
"""Dump full per-epoch histories for a tagged wave and report CONVERGENCE diagnostics.

WHY THIS EXISTS SEPARATELY FROM ``score_decoder_arms.py``. That script answers "which arm
scored higher", by reducing each run to one number (max of a centered rolling mean). It
cannot answer "was the run finished when we scored it" -- and on this task that second
question dominates the first. Every architecture delta measured in this project so far is
<=0.03, taken on runs whose TRAIN pearson was still climbing at the epoch cap. If the
budget truncates arm-dependently, the ranking is partly a ranking of who was cut off later.

So this script reports, per run, whether each series had actually stopped moving:

  * ``roll_end`` vs ``roll_max`` and ``roll_argmax`` -- if argmax sits near the final
    epoch and ``roll_end ~= roll_max``, the run was still improving when it stopped.
  * OLS slope of the smoothed val metric over the last ``--tail`` epochs, with its
    standard error and a t-test against zero. This is the actual test of "is val still
    creeping up", as opposed to eyeballing a noisy curve in the W&B UI. Reported per
    1000 epochs so the numbers are readable next to a 300-epoch budget.
  * the same slope for ``train/expression/pearson_per_feature`` and the relative decline
    of ``train/expression/loss`` over the tail, plus terminal ``train/grad_norm`` against
    the clip threshold -- the train side is what tells us whether the optimizer still had
    somewhere to go.

EVERY KEY IS READ ONE AT A TIME via ``scan_history(keys=[k])``. A multi-key
``history(keys=[...])`` silently returns ZERO rows when any requested key is not co-logged
in the same step as the others, which is why the train-side series went unread in this
project for several rounds. Do not "optimize" this into one multi-key call.

Usage
-----
    python experiments/019-simb-multimodal/scripts/wave4b_convergence.py \
        --stage-tag stage-wave4b --window 5 --tail 100

Writes ``results/wave_history_<stage-tag>.csv`` (tidy per-epoch, one row per run-epoch)
and ``results/wave_convergence_<stage-tag>.csv`` (one row per run).
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import re

import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv

load_dotenv()

# Series pulled in full. Keys verified present on run 047tg3h4 -- never guess a key name
# here; a missing key yields an empty column that reads as "the mechanism did nothing".
SERIES_KEYS = [
    "val/mean/pearson_per_feature",
    "val/expression/pearson_per_feature",
    "val/expression/pearson_per_instance",
    "val/expression/loss",
    "val/expression/pred_sd_ratio",
    "train/expression/pearson_per_feature",
    "train/expression/loss",
    "train/grad_norm",
    "train/grad_norm_clip_frac",
    "gate/perturbation_transform.null_bias",
]
VAL_METRIC = "val/mean/pearson_per_feature"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", default="torchcell_019_expr_v8")
    p.add_argument("--entity", default="zhao-group")
    p.add_argument("--stage-tag", default="stage-wave4b")
    p.add_argument("--window", type=int, default=5, help="rolling-mean width in epochs")
    p.add_argument(
        "--tail",
        type=int,
        default=100,
        help="epochs at the END of the run used for the still-improving slope test",
    )
    return p.parse_args()


def arm_of(run: wandb.apis.public.Run) -> str:
    """Arm from TAGS by shape, matching score_decoder_arms.arm_label."""
    arms = {t for t in run.tags if re.match(r"^[A-Z][A-Za-z0-9]*_", t)}
    if not arms:
        raise ValueError(f"run {run.id} has no arm tag (tags={list(run.tags)})")
    if len(arms) > 1:
        raise ValueError(f"run {run.id} has ambiguous arm tags: {sorted(arms)}")
    return arms.pop()


def tail_slope(y: pd.Series, tail: int) -> tuple[float, float, float, int]:
    """OLS slope of ``y`` against epoch over its last ``tail`` finite points.

    Returns (slope_per_1000_epochs, se_per_1000, t_stat, n). The slope is rescaled
    because a per-epoch slope on this task is ~1e-5 and unreadable; per-1000 makes it
    directly comparable to the ~0.15 scale of the metric itself.
    """
    y = y.dropna()
    if len(y) < 10:
        return float("nan"), float("nan"), float("nan"), len(y)
    y = y.iloc[-tail:]
    x = np.asarray(y.index, dtype=float)
    yv = np.asarray(y.values, dtype=float)
    n = len(yv)
    xc = x - x.mean()
    sxx = float((xc**2).sum())
    slope = float((xc * (yv - yv.mean())).sum() / sxx)
    resid = yv - (yv.mean() + slope * xc)
    # df = n - 2 for a fitted intercept + slope.
    sigma2 = float((resid**2).sum() / (n - 2))
    se = float(np.sqrt(sigma2 / sxx))
    return slope * 1000.0, se * 1000.0, slope / se if se > 0 else float("nan"), n


def main() -> None:
    args = parse_args()
    api = wandb.Api(timeout=60)
    runs = [
        r
        for r in api.runs(f"{args.entity}/{args.project}", per_page=300)
        if args.stage_tag in r.tags
    ]
    if not runs:
        raise SystemExit(f"no runs tagged '{args.stage_tag}' in {args.project}")
    print(f"{len(runs)} runs tagged {args.stage_tag}")

    long_rows: list[dict[str, object]] = []
    conv_rows: list[dict[str, object]] = []

    for run in sorted(runs, key=lambda r: (arm_of(r), r.config.get("seed", -1))):
        arm, seed = arm_of(run), run.config.get("seed")
        # ONE KEY PER scan_history CALL -- see module docstring.
        series: dict[str, pd.Series] = {}
        for key in SERIES_KEYS:
            vals = [
                r[key] for r in run.scan_history(keys=[key]) if r.get(key) is not None
            ]
            series[key] = pd.Series(vals, dtype=float)

        frame = pd.DataFrame(series)
        frame.insert(0, "epoch", frame.index)
        frame.insert(0, "seed", seed)
        frame.insert(0, "arm", arm)
        frame.insert(0, "run_id", run.id)
        long_rows.extend(frame.to_dict("records"))

        val = series[VAL_METRIC]
        roll = val.rolling(args.window, center=True, min_periods=args.window).mean()
        v_slope, v_se, v_t, v_n = tail_slope(roll, args.tail)
        t_slope, t_se, t_t, _ = tail_slope(
            series["train/expression/pearson_per_feature"], args.tail
        )
        tl = series["train/expression/loss"].dropna()
        gate = series["gate/perturbation_transform.null_bias"].dropna()
        gn = series["train/grad_norm"].dropna()

        conv_rows.append(
            {
                "arm": arm,
                "seed": seed,
                "run_id": run.id,
                "state": run.state,
                "epochs": int(len(val)),
                "roll_max": float(roll.max()),
                "roll_argmax": int(roll.idxmax()),
                "roll_end": float(roll.dropna().iloc[-1]),
                "raw_peak": float(val.max()),
                # Positive, t-significant slope over the tail => still climbing at the cap.
                "val_slope_k": v_slope,
                "val_slope_se_k": v_se,
                "val_slope_t": v_t,
                "val_tail_n": v_n,
                "train_pf_end": float(
                    series["train/expression/pearson_per_feature"].dropna().iloc[-1]
                ),
                "train_pf_max": float(
                    series["train/expression/pearson_per_feature"].max()
                ),
                "train_pf_slope_k": t_slope,
                "train_pf_slope_t": t_t,
                "train_loss_end": float(tl.iloc[-1]) if len(tl) else float("nan"),
                # TREND, not an endpoint difference. The first version of this column was
                # 100*(tl[-tail] - tl[-1])/|tl[-tail]|, and it reported swings of +/-15%
                # that were pure noise: train/expression/loss oscillates roughly 0.22-0.28
                # from epoch to epoch, so ANY two-point difference on it is dominated by
                # which two epochs got sampled. Two seeds came out "loss increased 15%"
                # purely from that. Use the fitted slope and read its t against the
                # residual scatter instead.
                "train_loss_slope_k": tail_slope(tl, args.tail)[0],
                "train_loss_slope_t": tail_slope(tl, args.tail)[2],
                "train_loss_tail_sd": (
                    float(tl.iloc[-args.tail :].std()) if len(tl) else float("nan")
                ),
                "grad_norm_end": float(gn.iloc[-1]) if len(gn) else float("nan"),
                "gate_first": float(gate.iloc[0]) if len(gate) else float("nan"),
                "gate_end": float(gate.iloc[-1]) if len(gate) else float("nan"),
                "gate_absmax": float(gate.abs().max()) if len(gate) else float("nan"),
            }
        )

    conv = pd.DataFrame(conv_rows)
    hist = pd.DataFrame(long_rows)

    pd.set_option("display.width", 250)
    print(f"\nCONVERGENCE (window={args.window}, tail={args.tail} epochs)")
    print(
        conv[
            [
                "arm",
                "seed",
                "epochs",
                "roll_max",
                "roll_argmax",
                "roll_end",
                "val_slope_k",
                "val_slope_t",
                "train_pf_end",
                "train_pf_slope_k",
                "train_pf_slope_t",
                "train_loss_slope_k",
                "train_loss_slope_t",
                "train_loss_tail_sd",
                "grad_norm_end",
                "gate_end",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )

    print("\nPAIRED within-seed on roll_max (arm - R_ref)")
    ref = conv[conv["arm"] == "R_ref"].set_index("seed")["roll_max"]
    for arm, grp in conv.groupby("arm"):
        if arm == "R_ref":
            continue
        d = pd.Series(
            {int(r.seed): r.roll_max - ref[r.seed] for r in grp.itertuples()}
        )
        from scipy import stats

        half = stats.t.ppf(0.975, len(d) - 1) * d.sem()
        verdict = "CI excludes 0" if abs(d.mean()) > half else "CI CONTAINS 0"
        print(
            f"  {arm:<12} n={len(d)} mean {d.mean():+.4f} sd {d.std():.4f} "
            f"95% CI +/-{half:.4f} [{verdict}]  per-seed "
            f"{ {k: round(v, 4) for k, v in d.items()} }"
        )

    print("\nPAIRED within-seed on roll_end (arm - R_ref) -- terminal, not peak")
    ref_e = conv[conv["arm"] == "R_ref"].set_index("seed")["roll_end"]
    for arm, grp in conv.groupby("arm"):
        if arm == "R_ref":
            continue
        d = pd.Series({int(r.seed): r.roll_end - ref_e[r.seed] for r in grp.itertuples()})
        from scipy import stats

        half = stats.t.ppf(0.975, len(d) - 1) * d.sem()
        verdict = "CI excludes 0" if abs(d.mean()) > half else "CI CONTAINS 0"
        print(
            f"  {arm:<12} n={len(d)} mean {d.mean():+.4f} 95% CI +/-{half:.4f} [{verdict}]"
        )

    results_dir = osp.join(os.environ["EXPERIMENT_ROOT"], "019-simb-multimodal", "results")
    os.makedirs(results_dir, exist_ok=True)
    h_out = osp.join(results_dir, f"wave_history_{args.stage_tag}.csv")
    c_out = osp.join(results_dir, f"wave_convergence_{args.stage_tag}.csv")
    hist.to_csv(h_out, index=False)
    conv.to_csv(c_out, index=False)
    print(f"\nwrote {h_out}\nwrote {c_out}")


if __name__ == "__main__":
    main()
