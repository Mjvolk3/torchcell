# experiments/022-mulleder-metabolome/scripts/analyze_training_length.py
# [[experiments.022-mulleder-metabolome.scripts.analyze_training_length]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/022-mulleder-metabolome/scripts/analyze_training_length
"""Are the good mulleder19 runs stopping before they finish learning?

The summary metrics cannot answer this. `peak_epoch` and `n_val_epochs` say WHERE the peak
was and how long the run went, but a peak at 50% of a run is equally consistent with two
opposite stories -- the model converged and the remaining epochs were the early-stopping
patience window, or the model was still climbing and a `min_delta` threshold cut it. Only the
CURVE separates them.

The discriminator used here is the TRAIN/VAL PAIR over the final stretch:

    train rising, val flat/falling  -> converged, then over-fitting. More epochs cost nothing
                                       and gain nothing; the peak is real.
    train rising, val rising        -> UNDER-TRAINED. The stopping rule fired on noise, and
                                       `patience` / `min_delta` / `max_epochs` are the lever.
    train flat,   val flat          -> optimisation stalled. More epochs will not help either;
                                       the lever is `lr` or capacity, not run length.

Also flags runs that hit `max_epochs` while still improving -- there the cap, not the
patience, is what ended training, and the peak is a lower bound on what the config can do.

Pulls from W&B via the Export API (summary + config for every run, full history for the ones
worth reading) and writes `results/training_length_analysis.json` plus a stdout table.

    python experiments/022-mulleder-metabolome/scripts/analyze_training_length.py
    python experiments/022-mulleder-metabolome/scripts/analyze_training_length.py --arm betaxanthin
"""

import argparse
import json
import os
import os.path as osp
import statistics as st

import pandas as pd
import wandb
from dotenv import load_dotenv

load_dotenv()

ENTITY = os.getenv("WANDB_ENTITY", "zhao-group")

#: Which metric each arm is ranked on -- beta-carotene is a subjective ordinal, so Spearman.
RANKED = {
    "mulleder19": "pearson_per_feature",
    "betaxanthin": "pearson_per_feature",
    "beta_carotene": "spearman_per_feature",
}

#: Trailing window (validation epochs) over which the end-of-run trend is measured. Long
#: enough to average through epoch-to-epoch noise, short enough to describe the END of the run
#: rather than its whole shape.
TREND_WINDOW = 15


def _series(run: object, key: str) -> list[float]:
    """Full unsampled history of one metric, NaNs dropped.

    `scan_history` rather than `history`: the latter downsamples to ~500 points by default,
    which is fine at 175 epochs but would silently smooth a longer run -- and the whole
    question here is about the shape of the last few epochs.
    """
    return [
        row[key]
        for row in run.scan_history(keys=[key])  # type: ignore[attr-defined]
        if row.get(key) is not None and row[key] == row[key]
    ]


def _slope(values: list[float]) -> float:
    """Least-squares slope of ``values`` against their index, per epoch."""
    n = len(values)
    if n < 3:
        return float("nan")
    xs = list(range(n))
    mx, my = st.mean(xs), st.mean(values)
    denom = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, values, strict=True)) / denom


def _patience_probe(values: list[float], min_delta: float = 1e-4) -> dict:
    """How much early-stopping patience did this run actually NEED to reach its peak?

    Every run here stops exactly ``patience + 1`` epochs after its peak, so a large share of
    each trial's wall clock is spent proving that the peak was the peak. The question is how
    much of that is buyable back: ``max_gap`` is the longest stretch of epochs between two
    successive new bests ON THE WAY UP, so any patience at or below it would have stopped the
    run BEFORE it found its true peak.

    Reported per run rather than pooled, because the distribution is what matters -- a
    patience that is safe for the median run can still truncate the best one.
    """
    best = float("-inf")
    peak_i = values.index(max(values))
    gaps, cur = [], 0
    for i, x in enumerate(values):
        if x > best + min_delta:
            best = x
            gaps.append(cur)
            cur = 0
        else:
            cur += 1

    def truncates(patience: int) -> bool:
        b, run = float("-inf"), 0
        for i, x in enumerate(values):
            if x > b + min_delta:
                b, run = x, 0
            else:
                run += 1
                if run > patience:
                    return i < peak_i
        return False

    return {
        "max_gap_to_peak": max(gaps) if gaps else 0,
        "truncated_at_patience_20": truncates(20),
        "truncated_at_patience_25": truncates(25),
        "truncated_at_patience_40": truncates(40),
    }


def _verdict(train_slope: float, val_slope: float, val_scale: float) -> str:
    """Classify the end of a run from its trailing train/val slopes.

    Thresholded RELATIVE to the run's own metric scale: a slope of +1e-4/epoch means something
    different for a run at 0.13 than for one at 0.01, and an absolute cut would label every
    small-metric run 'flat'.
    """
    eps = max(val_scale, 1e-6) * 0.002  # 0.2% of the metric's own level, per epoch
    val_rising = val_slope > eps
    train_rising = train_slope > eps
    if val_rising and train_rising:
        return "UNDER-TRAINED (both still rising)"
    if train_rising and not val_rising:
        return "converged -> over-fitting"
    if not train_rising and not val_rising:
        return "stalled (neither rising)"
    return "val rising, train flat (unusual)"


def analyze(arm: str, project: str, top_frac: float = 0.5) -> dict:
    """Pull the sweep and classify how each worthwhile run ended."""
    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{project}"))
    metric = RANKED[arm]
    val_key = f"val/{arm}/{metric}"
    train_key = f"train/{arm}/{metric}"

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        summary_list.append(run.summary._json_dict)
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )
        name_list.append(run.name)
    runs_df = pd.DataFrame(
        {"summary": summary_list, "config": config_list, "name": name_list}
    )

    # Rank on the PEAK, not the last epoch -- these runs peak then collapse toward the
    # per-feature mean, so the summary (last-epoch) value systematically understates.
    rows = []
    for run in runs:
        if run.state not in ("finished", "running"):
            continue
        # ONE KEY PER CALL. `history(keys=[a, b])` returns only steps where EVERY requested
        # key is present, and train metrics are logged per STEP while val metrics are logged
        # per EPOCH -- so asking for both at once returns an empty intersection rather than
        # an error. That silently produced "0 runs with usable history" on a project with 25.
        val = _series(run, val_key)
        train = _series(run, train_key)
        if len(val) < TREND_WINDOW:
            continue
        peak = max(val)
        peak_i = val.index(peak)
        n_epochs = len(val)
        v_tail, t_tail = val[-TREND_WINDOW:], train[-TREND_WINDOW:] if train else []
        v_slope = _slope(v_tail)
        t_slope = _slope(t_tail) if len(t_tail) >= 3 else float("nan")
        max_epochs = run.config.get("trainer", {}).get("max_epochs")
        rows.append(
            {
                "run": run.name[-12:],
                "job": run.name.split("-")[1].split("_")[0] if "-" in run.name else "?",
                "peak": round(peak, 4),
                "final": round(val[-1], 4),
                "peak_epoch": peak_i,
                "n_epochs": n_epochs,
                "peak_frac": round(peak_i / n_epochs, 2),
                "epochs_after_peak": n_epochs - peak_i,
                "hit_cap": bool(max_epochs and n_epochs >= max_epochs),
                "val_slope_tail": v_slope,
                "train_slope_tail": t_slope,
                "verdict": _verdict(t_slope, v_slope, peak),
                "lr": run.config.get("regression_task", {})
                .get("optimizer", {})
                .get("lr"),
                "dist": run.config.get("multitask", {}).get("dist"),
                "emb": run.config.get("cell_dataset", {}).get("node_embeddings"),
                **_patience_probe(val),
            }
        )

    rows.sort(key=lambda r: -r["peak"])
    keep = [r for r in rows if r["peak"] >= top_frac * rows[0]["peak"]] if rows else []
    return {
        "arm": arm,
        "project": project,
        "n_runs": len(runs),
        "n_analyzed": len(rows),
        "runs_df_shape": list(runs_df.shape),
        "all": rows,
        "top": keep,
    }


def main() -> None:
    """Run the analysis for one arm and write the JSON + stdout table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="mulleder19", choices=sorted(RANKED))
    ap.add_argument("--project", default=None)
    args = ap.parse_args()
    project = args.project or f"torchcell_020_{args.arm}_v3"

    report = analyze(args.arm, project)
    out_dir = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    out = osp.join(out_dir, "training_length_analysis.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)

    rows = report["all"]
    print(f"\n{report['arm']}  ({report['n_analyzed']} runs with usable history "
          f"of {report['n_runs']} in {project})\n")
    hdr = (f"{'peak':>7} {'final':>7} {'pk_ep':>6} {'n_ep':>5} {'pk/n':>5} "
           f"{'after':>6} {'cap':>4} {'val_slope':>11} {'trn_slope':>11}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for r in rows[:16]:
        print(
            f"{r['peak']:7.4f} {r['final']:7.4f} {r['peak_epoch']:6d} {r['n_epochs']:5d} "
            f"{r['peak_frac']:5.2f} {r['epochs_after_peak']:6d} "
            f"{'YES' if r['hit_cap'] else '':>4} "
            f"{r['val_slope_tail']:+11.2e} {r['train_slope_tail']:+11.2e}  {r['verdict']}"
        )

    if rows:
        top = report["top"]
        print(f"\nTOP {len(top)} runs (within 2x of the best):")
        counts: dict[str, int] = {}
        for r in top:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        for v, c in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"    {c:3d}  {v}")
        print(f"\n  epochs after peak: median={st.median([r['epochs_after_peak'] for r in top])}"
              f"  (early-stopping patience is the floor)")
        print(f"  hit max_epochs   : {sum(r['hit_cap'] for r in top)}/{len(top)}")
        print(f"  longest peak_epoch: {max(r['peak_epoch'] for r in rows)} "
              f"(cap is {rows[0].get('max_epochs') or 'max_epochs'})")
        print("\n  patience probe -- would a SHORTER patience have truncated a run "
              "before its peak?")
        for p_ in (20, 25, 40):
            n = sum(r[f"truncated_at_patience_{p_}"] for r in rows)
            print(f"      patience {p_:2d}: truncates {n}/{len(rows)} runs   "
                  f"(max gap to peak, median={st.median([r['max_gap_to_peak'] for r in rows]):.0f} "
                  f"max={max(r['max_gap_to_peak'] for r in rows)})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
