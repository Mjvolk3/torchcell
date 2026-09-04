# experiments/027-betaxanthin-metabolic/scripts/analyze_bxfx.py
# [[experiments.027-betaxanthin-metabolic.scripts.analyze_bxfx]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/analyze_bxfx.py

r"""The prespecified 027 analysis. Written before the grid runs; do not edit after it launches.

THE DECISION, STATED AS A RULE RATHER THAN A JUDGEMENT
-------------------------------------------------------
The metabolic module improves on the no-module CGT if and only if BOTH hold:

  (D1) the paired arm gap EXCEEDS ITS OWN PERMUTED-LABEL GAP --
       ``mean_s[anchored - pooled] - mean_s[null_anchored - null_pooled] > 0``
       with a 95% bootstrap interval excluding 0; and
  (D2) ``mean_s[test_spearman(flux_anchored)]`` exceeds the Flux Cone Learning
       RandomForest_Resampled baseline of +0.0391 on the same 639 genes, by more than one
       across-seed standard error.

(D1) is the part 026 skipped, and it is why 026 could not decide. On its val-peak statistic
the real paired gap was +0.0735 (sd 0.0303, n=10) and the PERMUTED one +0.0302 (sd 0.0722,
n=12). Against zero that is 7.7 sigma; against its own null it is +0.0433 at 1.9 sigma.
Comparing an architecture's gap to zero rather than to the same gap with the labels destroyed
credits the architecture for capacity to chase a validation set.

BALANCED PREFIX
---------------
Only seeds for which EVERY arm completed are used. A paired difference needs both members,
and an unbalanced average silently reweights the arms toward whichever seeds happened to
finish -- which on a wall-clock kill is not random.

NO MAXIMUM IS EVER REPORTED. Not over epochs (the trainer already selected on validation and
scored on a disjoint test set), not over seeds, not over arms. Every number is a mean with
its n and its across-seed SD.
"""

import argparse
import glob
import json
import os
import os.path as osp
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
EXP_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic")
RESULTS_DIR = osp.join(EXP_DIR, "results")

#: The contrasts, in the order they answer the question. Each changes exactly one thing.
CONTRASTS = [
    ("flux_anchored", "pooled", "the headline: full module vs no module"),
    ("flux_off", "pooled", "stoichiometry alone -- is the network the gain?"),
    ("flux_anchored", "flux_free", "do the TABULATED energies beat any potential?"),
    ("null_anchored", "null_pooled", "the same headline gap with labels destroyed"),
]
#: The pair whose difference (D1) tests.
HEADLINE = ("flux_anchored", "pooled")
NULL_PAIR = ("null_anchored", "null_pooled")
N_BOOT = 20000


def load_runs(results_dir: str) -> dict[tuple[str, int], dict[str, Any]]:
    """Every completed cell, keyed by ``(arm, seed)``.

    Reads the per-worker files rather than a merged one: with 72 workers there is no merge
    step, and a worker that died leaves its own file truncated instead of corrupting a shared
    one.
    """
    out: dict[tuple[str, int], dict[str, Any]] = {}
    files = sorted(glob.glob(osp.join(results_dir, "bxfx_w*.json")))
    assert files, f"no bxfx_w*.json under {results_dir}"
    for path in files:
        with open(path) as fh:
            payload = json.load(fh)
        for run in payload["runs"]:
            cell = run["cell"]
            key = (cell["arm"], int(cell["seed"]))
            # A duplicate means the shard rule failed and two workers ran the same cell --
            # the optuna SQLite race this design exists to prevent. It is an error, not a
            # thing to average over.
            assert key not in out, (
                f"cell {key} appears twice ({path}). Two workers claimed it, which means "
                "GRID_SHARD_COUNT was unset or the per-worker storage was shared."
            )
            out[key] = run
    return out


def balanced_table(
    runs: dict[tuple[str, int], dict[str, Any]], metric: str = "spearman"
) -> tuple[list[str], list[int], dict[str, np.ndarray]]:
    """``arm -> array over the seeds where EVERY arm completed``, plus the arms and seeds."""
    arms = sorted({a for a, _ in runs})
    seeds = sorted({s for _, s in runs})
    complete = [s for s in seeds if all((a, s) in runs for a in arms)]
    dropped = [s for s in seeds if s not in complete]
    if dropped:
        print(
            f"[balance] dropping {len(dropped)} incomplete seed(s): {dropped}",
            flush=True,
        )
    table = {
        a: np.array([float(runs[(a, s)]["test"]["pinned"][metric]) for s in complete])
        for a in arms
    }
    return arms, complete, table


def paired(table: dict[str, np.ndarray], a: str, b: str) -> dict[str, float]:
    d = table[a] - table[b]
    n = len(d)
    return {
        "n": n,
        "mean": float(d.mean()),
        "sd": float(d.std(ddof=1)) if n > 1 else float("nan"),
        "se": float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
    }


def bootstrap_excess(
    table: dict[str, np.ndarray], rng: np.random.Generator
) -> dict[str, float]:
    """95% percentile-bootstrap interval on (real gap - null gap).

    Resampled SEED-WISE and jointly across all four arms, so the pairing that makes the
    contrast low-variance is preserved: a bootstrap that resampled arms independently would
    destroy exactly the correlation the paired design is exploiting.
    """
    ra, rb = HEADLINE
    na, nb = NULL_PAIR
    d_real = table[ra] - table[rb]
    d_null = table[na] - table[nb]
    n = len(d_real)
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        draws[i] = d_real[idx].mean() - d_null[idx].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    # One-sided, distribution-free: resample the PERMUTED-LABEL gap alone and ask how often
    # it reaches the real gap's observed mean. This is the number to quote when the null gap
    # is skewed, which 026's was -- sd 0.0722 against the real gap's 0.0303, so a symmetric
    # interval understates how often the null can look like a result.
    null_only = np.array([d_null[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return {
        "excess": float(d_real.mean() - d_null.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "real_gap": float(d_real.mean()),
        "null_gap": float(d_null.mean()),
        "p_null_ge_real": float((null_only >= d_real.mean()).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--fcl-spearman", type=float, default=0.039064257838781265)
    parser.add_argument("--out", default="bxfx_decision.json")
    args = parser.parse_args()

    runs = load_runs(args.results_dir)
    arms, seeds, table = balanced_table(runs)
    rng = np.random.default_rng(0)

    print(
        f"\n{len(runs)} cells loaded; {len(seeds)} balanced seeds over {len(arms)} arms\n"
    )
    print(
        f"{'arm':16s} {'n':>3s} {'mean':>9s} {'sd':>8s} {'se':>8s}  test Spearman, 639 pinned genes"
    )
    per_arm = {}
    for a in arms:
        v = table[a]
        per_arm[a] = {
            "n": len(v),
            "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
            "se": float(v.std(ddof=1) / np.sqrt(len(v)))
            if len(v) > 1
            else float("nan"),
        }
        print(
            f"{a:16s} {per_arm[a]['n']:3d} {per_arm[a]['mean']:+9.4f} "
            f"{per_arm[a]['sd']:8.4f} {per_arm[a]['se']:8.4f}"
        )
    print(f"{'FCL RF (published)':16s} {'--':>3s} {args.fcl_spearman:+9.4f}")

    print(
        f"\n{'contrast':34s} {'n':>3s} {'mean':>9s} {'sd':>8s} {'se':>8s}  what it isolates"
    )
    contrasts = {}
    for a, b, why in CONTRASTS:
        p = paired(table, a, b)
        contrasts[f"{a}-{b}"] = {**p, "why": why}
        print(
            f"{a + ' - ' + b:34s} {p['n']:3d} {p['mean']:+9.4f} {p['sd']:8.4f} "
            f"{p['se']:8.4f}  {why}"
        )

    boot = bootstrap_excess(table, rng)
    d1 = boot["ci_lo"] > 0.0
    anchored = per_arm["flux_anchored"]
    d2 = anchored["mean"] - anchored["se"] > args.fcl_spearman

    print("\n---------------- THE PRESPECIFIED DECISION ----------------")
    print(
        f"D1  excess over the permuted-label gap = {boot['excess']:+.4f} "
        f"[95% CI {boot['ci_lo']:+.4f}, {boot['ci_hi']:+.4f}]  -> {'PASS' if d1 else 'FAIL'}"
    )
    print(
        f"D2  flux_anchored {anchored['mean']:+.4f} +- {anchored['se']:.4f} (se) vs "
        f"FCL RF {args.fcl_spearman:+.4f}  -> {'PASS' if d2 else 'FAIL'}"
    )
    verdict = (
        "the metabolic module improves betaxanthin prediction"
        if (d1 and d2)
        else "the metabolic module is NOT shown to improve betaxanthin prediction"
    )
    print(f"\nVERDICT: {verdict}")
    print(
        "\nA FAIL on D1 with a positive raw gap is the 026 result reproduced with the "
        "sample size to see it: the gap is architecture, not biology."
    )

    payload = {
        "n_cells": len(runs),
        "balanced_seeds": seeds,
        "arms": arms,
        "per_arm_test_spearman_pinned": per_arm,
        "contrasts": contrasts,
        "null_calibration": boot,
        "fcl_rf_test_spearman": args.fcl_spearman,
        "D1_excess_ci_excludes_zero": bool(d1),
        "D2_beats_fcl_by_one_se": bool(d2),
        "verdict": verdict,
    }
    os.makedirs(args.results_dir, exist_ok=True)
    path = osp.join(args.results_dir, args.out)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
