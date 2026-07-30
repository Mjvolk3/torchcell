# experiments/019-simb-multimodal/scripts/score_decoder_arms.py
# [[experiments.019-simb-multimodal.scripts.score_decoder_arms]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/score_decoder_arms
"""Score the _008 decoder / pair-routing arms from W&B, robustly.

WHY NOT THE RAW PEAK. ``val/mean/pearson_per_feature`` swings by ~+/-0.05 between
consecutive epochs on this task (measured on _006 run kmd40o2h: ..., 0.1400, 0.0811,
0.1017, 0.1499, 0.1145, 0.1456, 0.0949, ...). A max over ~135 such epochs is an
upward-biased ORDER STATISTIC whose bias grows with the number of epochs run, so an arm
that merely trains longer "wins". Worse, the three _006 runs that appear to replicate at
0.1719 are the SAME seed re-executed, so they measure determinism, not variance.

This script therefore reports, per arm:

  * ``smoothed`` -- max over epochs of a centered ``--window`` epoch rolling mean of the
    val metric. Averaging first collapses the epoch-scale noise, so the max is taken over
    a far smoother trajectory and the selection bias is much smaller.
  * ``raw_peak`` -- the biased statistic, kept ONLY so the two can be compared and the
    bias is visible rather than hidden.
  * ``n_seeds`` / ``spread`` -- across-seed mean and (max - min). An arm is only called a
    win when its smoothed mean clears the baseline's by more than the baseline's own
    across-seed spread.

Usage
-----
    python experiments/019-simb-multimodal/scripts/score_decoder_arms.py \
        --project torchcell_019_expr_v8 --window 5

Writes ``experiments/019-simb-multimodal/results/decoder_arms_<project>.csv``.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import re
from collections import defaultdict

import pandas as pd
import wandb
from dotenv import load_dotenv

load_dotenv()

METRIC = "val/mean/pearson_per_feature"
# The _006/_007 reference peaks, measured from W&B history (not from the run summary,
# which stores the LAST epoch and so understates the peak badly: 0.1721 vs 0.1463).
REFERENCE_PEAKS = {
    "_006 kmd40o2h (L=6)": 0.1721,
    "_006 xenfbzn5 (L=4)": 0.1719,
    "_007 nrurev80 (L=4)": 0.1702,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", default="torchcell_019_expr_v8")
    p.add_argument("--entity", default="zhao-group")
    p.add_argument(
        "--window",
        type=int,
        default=5,
        help="rolling-mean width in epochs; the scored statistic is max of this mean",
    )
    p.add_argument(
        "--reference",
        default="A1_ref",
        help="arm used as the WITHIN-SEED paired reference; must be co-launched in-wave",
    )
    return p.parse_args()


# The source edit that split this project into two incomparable code versions. Wave-2
# configs carry six subtrees wave-1 configs lack (perturbation_head.num_layers/residual,
# attention_mask, cross_gene, observed_labels, post_perturbation_mixing, lr_scheduler),
# so a run from either side of it is a DIFFERENT EXPERIMENT. Runs are never pooled across
# it. Anything after the null-sink/pooling commit is wave 3.
WAVE_BOUNDARIES = [
    ("2026-07-28T22:26:23", "wave2"),
    ("2026-07-29T00:00:00", "wave3"),
    # wave4a: cgt_expr_010 (hard masking frozen, KL off) with the UNMATCHED null sink.
    # This boundary was MISSING at first, and its absence silently recreated the exact
    # error this table exists to prevent: the wave4a runs were created at 01:00:50, i.e.
    # after the wave3 boundary and before the wave4b one, so they landed in the `wave3`
    # bucket -- and because wave4a is where the `R_ref` arm was introduced, `R_ref` became
    # the in-wave paired reference for every wave3 arm. Each wave3 delta was then measured
    # against a DIFFERENT config, a different graph-prior mechanism, and a reference
    # CANCELLED at 77-81 epochs versus arms that ran 300. Every such number was void.
    ("2026-07-30T01:00:50", "wave4a"),
    # wave4b: the fused-SDPA gate, the masked-layer attention-dropout gap, and the
    # attention_mask.layers validation all landed here. SDPA consumes RNG differently from
    # the manual softmax path, so a wave4b run is not comparable to a wave3/wave4a run even
    # at an identical seed -- this boundary is what stops them being pooled.
    ("2026-07-30T01:38:57", "wave4b"),
    # wave5: cgt_expr_011 -- the PARTITION is pinned (split_seed 0) so `seed` varies
    # initialization only, and the eval-mode train metric (`traineval/*`) is logged. A wave5
    # run is therefore not comparable to any earlier run even at an identical seed, because
    # "seed 3" no longer means the same validation set.
    ("2026-07-30T05:07:59", "wave5"),
]


def arm_label(run: wandb.apis.public.Run) -> str:
    """Read the arm from the run's TAGS.

    WHY NOT DERIVE IT FROM THE CONFIG (what this did until 2026.07.29). The old version
    inferred the arm from ``multitask.free_gene_dim`` and ``model.perturbation_propagation``
    alone, on the reasoning that "a label can drift from what actually ran, whereas these
    keys ARE what ran". That reasoning failed in the worst possible direction: arms kept
    being added to ``gh_expr_008_arm.sh`` that changed NEITHER of those two axes -- C0,
    D1, D2, E0, F1, G1, H0, GEARS and S1/S2/S3 -- so every one of them fell through to the
    literal string ``"A0_baseline"`` and was averaged INTO the baseline by the groupby.
    The reported baseline was a blend of eleven different architectures, and the decision
    rule compared each arm against that blend plus what it called an "across-seed spread"
    but was really an across-ARM spread.

    The tag does not drift: ``gh_expr_008_arm.sh:143`` writes
    ``wandb.tags=[<config>, <ARM>, seed<N>]`` from the SAME ``case`` statement that sets
    the overrides, so the label and the behaviour are set in one place. An untagged run
    RAISES rather than silently becoming the baseline.
    """
    # SHAPE-MATCH, not a letter whitelist. The first version listed allowed initials
    # ("A","B",...,"W") and R WAS NOT AMONG THEM, so the wave-4 reference arm `R_ref` matched
    # nothing and every reference run raised "no arm tag" -- i.e. the fix for the derive-vs-
    # declare bug reintroduced the same class of bug via a different hardcoded list. Arm tags
    # are Capitalised-with-an-underscore by construction (A1_ref, D2_mask, L0_lr3e4, N_sink0,
    # R_ref, B_film, F0_zS); the declared axis tags added in wave 4 are lowercase
    # (mech-baseline, xfer-yes, stage-wave4) and the config tag is lowercase (cgt_expr_010),
    # so this pattern accepts every arm and rejects every non-arm without enumerating either.
    arms = {t for t in run.tags if re.match(r"^[A-Z][A-Za-z0-9]*_", t)}
    if not arms:
        raise ValueError(
            f"run {run.id} has no arm tag (tags={list(run.tags)}). "
            "Runs must be launched through gh_expr_008_arm.sh, which tags them."
        )
    if len(arms) > 1:
        raise ValueError(f"run {run.id} has ambiguous arm tags: {sorted(arms)}")
    return arms.pop()


def wave_of(created_at: str) -> str:
    """Bucket a run by source version, from its creation timestamp.

    Pooling across a source edit measures the edit, not the arm -- which is exactly what
    produced the phantom 0.0032 "within-config noise floor" (both of its replicates
    straddle the 22:26:23 boundary AND differ by 55-76 epochs).
    """
    wave = "wave1"
    for boundary, name in WAVE_BOUNDARIES:
        if str(created_at) >= boundary:
            wave = name
    return wave


def main() -> None:
    args = parse_args()
    api = wandb.Api(timeout=60)
    runs = list(api.runs(f"{args.entity}/{args.project}", per_page=300))
    print(f"{len(runs)} runs in {args.entity}/{args.project}")

    rows: list[dict[str, object]] = []
    for run in runs:
        if run.state == "running":
            continue
        history = run.history(keys=[METRIC], pandas=True)
        if METRIC not in history or history[METRIC].dropna().empty:
            continue
        series = history[METRIC].dropna().reset_index(drop=True)
        # A run shorter than the rolling window has NO valid smoothed value (min_periods
        # leaves every entry NaN), so `smoothed.idxmax()` raises "cannot convert float NaN
        # to integer" and takes the whole report down -- which is what happened when the six
        # crashed wave-4a runs (2-4 epochs each) entered the project. Skip them LOUDLY: the
        # scoring statistic is undefined for them, and a run that died in its first epochs
        # has no result to report either way.
        if len(series) < args.window:
            print(
                f"  skip {run.id} ({arm_label(run)}, state={run.state}): "
                f"{len(series)} epochs < window {args.window}, smoothed score undefined"
            )
            continue
        smoothed = series.rolling(args.window, center=True, min_periods=args.window).mean()
        # TRAIN-SIDE FIT. Read ONE KEY AT A TIME via scan_history: a multi-key
        # history(keys=[...]) filter that includes a key which is not co-logged with the
        # others silently returns ZERO rows, which is why nobody had read this series --
        # and it is the series that showed the model never fits its own training set
        # (train pearson 0.23 and still climbing at the cap, train loss -8% over 196
        # epochs, grad_norm 0.02-0.13 against a clip of 10.0).
        train_fit = {}
        for key, name in [
            ("train/expression/pearson_per_feature", "train_pf"),
            ("train/expression/loss", "train_loss"),
        ]:
            vals = [r[key] for r in run.scan_history(keys=[key]) if r.get(key) is not None]
            train_fit[f"{name}_last"] = float(vals[-1]) if vals else float("nan")
            train_fit[f"{name}_best"] = (
                float(max(vals) if name == "train_pf" else min(vals))
                if vals
                else float("nan")
            )

        rows.append(
            {
                "arm": arm_label(run),
                "wave": wave_of(run.created_at),
                # POOL = GPU TYPE. Wave 5 ran the SAME arms on GilaHyper RTX 6000 Ada and on
                # IGB A100s, so a groupby on (wave, arm) alone would silently average two
                # hardware pools together -- the exact defect that made _007's rankings
                # uninterpretable (A100 mean 0.0242 vs Ada 0.0182 on identical arms). Each
                # pool co-launches its own reference, so pairing must happen WITHIN a pool.
                # POOL = HOST, not gpu_type. gpu_type alone is NOT sufficient: IGB's cabbi
                # partition reports the identical device string to GilaHyper
                # ("NVIDIA RTX 6000 Ada Ge") even though SLURM advertises it as RTX6000, so
                # keying on the model silently merged GilaHyper's 2-runs-per-GPU runs with
                # cabbi's 1-run-per-GPU probes -- different packing, and it produced a
                # duplicate (arm, seed) that crashed the paired report. The host is the thing
                # that actually fixes hardware AND packing regime together.
                "pool": (run.metadata or {}).get("host", "unknown"),
                # BUDGET completes the pairing key. Within one (wave, pool) the SAME arm and
                # seed can appear from two different stages at different epoch caps -- wave 5
                # ran mask depth at 400 and the H2 replication at 300, both with their own
                # W_ref at seeds 0-3 on the same GPU type. Keying on (wave, pool, arm, seed)
                # alone made `ref_by_seed[seed]` return TWO rows, so the "delta" became a
                # Series and the report crashed. Runs at different budgets are not comparable
                # anyway (the scored statistic is a max over epochs, whose selection bias
                # grows with the number of epochs run), so the budget belongs in the key.
                "budget": int(round(len(series) / 50.0) * 50),
                "source_hash": (run.config.get("source") or {}).get("git_hash", "")[:8],
                "seed": run.config.get("seed"),
                "run_id": run.id,
                "state": run.state,
                "epochs": int(len(series)),
                "smoothed": float(smoothed.max()),
                "smoothed_argmax": int(smoothed.idxmax()),
                "raw_peak": float(series.max()),
                "last": float(series.iloc[-1]),
                **train_fit,
            }
        )

    if not rows:
        raise SystemExit(f"no finished runs with '{METRIC}' in {args.project}")

    df = pd.DataFrame(rows).sort_values(["wave", "pool", "budget", "arm", "seed"]).reset_index(drop=True)

    # GROUPED BY (wave, arm) -- NEVER by arm alone. Two runs of the same arm on opposite
    # sides of a source edit are different experiments, and averaging them reports the
    # edit as if it were seed variance.
    per_arm: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
    for (wave, pool, budget, arm), grp in df.groupby(["wave", "pool", "budget", "arm"]):
        per_arm[(str(wave), str(pool), str(budget), str(arm))] = {
            "n_seeds": len(grp),
            "smoothed_mean": grp["smoothed"].mean(),
            "smoothed_spread": grp["smoothed"].max() - grp["smoothed"].min(),
            "raw_peak_mean": grp["raw_peak"].mean(),
            "train_pf_best": grp["train_pf_best"].mean(),
        }

    summary = (
        pd.DataFrame(per_arm)
        .T.reset_index(names=["wave", "pool", "budget", "arm"])
        .sort_values(["wave", "pool", "budget", "smoothed_mean"], ascending=[True, True, True, False])
    )

    print(f"\nPER-RUN (window={args.window})")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nPER-ARM (wave, arm) -- never pooled across waves")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nREFERENCE (raw peak, IGB hardware -- compare against raw_peak_mean only)")
    for name, value in REFERENCE_PEAKS.items():
        print(f"  {name:<22} {value:.4f}")

    # PAIRED REPORT. Unpaired comparison is dead on arrival here: the across-seed sd of an
    # IDENTICAL config is 0.0652 -- 41% of the baseline's own score -- giving a 1-seed
    # unpaired MDE of 0.181, larger than the entire signal. The seed is a shared nuisance,
    # so every number reported is the WITHIN-SEED difference against the in-wave reference.
    #
    # Each arm's CI is computed from that arm's OWN per-seed deltas (df = n_seeds - 1). The
    # previously pooled paired sd of 0.0039 is NOT used: it is heteroscedastic across arms
    # (0.0055 vs 0.00084, F(2,2)=42.9, p~0.023), it was measured only on arms that are
    # exact identities at init, and it was measured pre-edit.
    for (wave, pool, budget), wave_df in df.groupby(["wave", "pool", "budget"]):
        ref = wave_df[wave_df["arm"] == args.reference]
        if ref.empty:
            print(f"\n{wave} / {pool} / {budget}ep: no in-pool reference '{args.reference}'")
            continue
        ref_by_seed = ref.set_index("seed")["smoothed"]
        print(f"\nPAIRED vs {args.reference} ({wave} / {pool} / {budget}ep)")
        for arm, grp in wave_df.groupby("arm"):
            if arm == args.reference:
                continue
            paired = [
                (int(r.seed), r.smoothed - ref_by_seed[r.seed])
                for r in grp.itertuples()
                if r.seed in ref_by_seed.index
            ]
            if not paired:
                print(f"  {arm:<22} no seed overlap with the reference")
                continue
            deltas = pd.Series([d for _, d in paired])
            line = f"  {arm:<22} n={len(deltas)}  mean {deltas.mean():+.4f}"
            if len(deltas) >= 2:
                # t(n-1) 95% CI from this arm's own deltas.
                from scipy import stats

                half = stats.t.ppf(0.975, len(deltas) - 1) * deltas.sem()
                verdict = "CI excludes 0" if abs(deltas.mean()) > half else "CI CONTAINS 0"
                line += f"  sd {deltas.std():.4f}  95% CI +/-{half:.4f}  [{verdict}]"
            else:
                line += "  (single seed -- NO significance claim)"
            print(line + f"   per-seed {[f'{d:+.4f}' for _, d in paired]}")

    results_dir = osp.join(
        os.environ["EXPERIMENT_ROOT"], "019-simb-multimodal", "results"
    )
    os.makedirs(results_dir, exist_ok=True)
    out = osp.join(results_dir, f"decoder_arms_{args.project}.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
