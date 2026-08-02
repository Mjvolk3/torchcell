# experiments/020-cachera-betaxanthin/scripts/analyze_002_noise_and_lambda.py
# [[experiments.020-cachera-betaxanthin.scripts.analyze_002_noise_and_lambda]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/020-cachera-betaxanthin/scripts/analyze_002_noise_and_lambda
"""Regenerate every number the `_003` sweep design rests on, from the `_002` Optuna studies.

The `_003` configs and sweep driver are justified by measurements, not preferences, and this
is the script those measurements come from. It reads the three `_002` SQLite studies and
emits four things:

1. REPLICATE NOISE. `_002` used a TPE sampler, which re-proposes points -- so several
   configurations were run more than once with IDENTICAL hyperparameters AND an identical
   seed (42 throughout; the sweep never overrode `cfg.seed`). The within-group standard
   deviation of those repeats is a direct estimate of the run-to-run noise every config
   comparison is made against. It is the single most consequential number for the design: it
   is what says a single-seed sweep cannot pick a winner on a scalar arm, which is why `_003`
   has a confirm stage at all.

   Reported with its dof, because these are small groups -- betaxanthin has dof 3 and
   mulleder19 only dof 1, whose sigma estimate is not usable on its own (a chi-square
   interval on 1 dof spans roughly 0.4x to 30x the point estimate). For mulleder19 the
   1/sqrt(F_eff) argument (F = 19 vs F = 1) is the better guide.

2. SELECTION INFLATION. The reported best of a sweep is a MAXIMUM over trials, so it is
   biased upward even when every configuration is equally good. Using the expected maximum of
   n standard normals, the inflation is approximately `sigma * E[max_n]`, which gives a
   bias-corrected floor for the headline number.

3. GRAPH-PRIOR CALIBRATION. `graph_reg_ratio` (the graph term divided by the data term) was
   recorded per trial, so the map from `graph_reg_lambda` to actual prior strength is
   measurable PER ARM. It is not shared: the same nominal lambda lands at very different
   ratios on different targets, which is why `_003` gives each arm its own lambda grid
   centred on its own parity point.

4. THE RUN-LENGTH CONFOUND. Ranking on `_max` takes a maximum over however many validation
   epochs a run happened to survive, and early stopping makes that length depend on the
   hyperparameters. The correlation between trial duration (a proxy for epoch count at fixed
   architecture) and the ranked objective measures how much of the ranking could be riding on
   draw count rather than model quality.

Outputs `results/analysis_002_noise_and_lambda.json` in THIS experiment directory, plus a
readable summary on stdout. Reads the sibling 021 / 022 study files as well, because the
three arms are only interpretable against each other -- the F = 1 vs F = 19 contrast in (1)
is the whole argument.

    python experiments/020-cachera-betaxanthin/scripts/analyze_002_noise_and_lambda.py
"""

import json
import math
import os
import os.path as osp
import statistics as st
from collections import Counter, defaultdict
from typing import Any

import optuna
from dotenv import load_dotenv

load_dotenv()
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_ROOT = os.environ["DATA_ROOT"]
SCRATCH_EXP = osp.join(DATA_ROOT, "experiments")
OUT_DIR = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")

#: (arm, experiment dir, study file, study name). One study per arm -- the three targets are
#: not commensurable, so they were never pooled into one study.
ARMS: list[tuple[str, str, str, str]] = [
    (
        "betaxanthin",
        "020-cachera-betaxanthin",
        "optuna_020_betaxanthin.db",
        "betaxanthin_002",
    ),
    (
        "beta_carotene",
        "021-ozaydin-beta-carotene",
        "optuna_020_beta_carotene.db",
        "beta_carotene_002",
    ),
    (
        "mulleder19",
        "022-mulleder-metabolome",
        "optuna_020_mulleder19.db",
        "mulleder19_002",
    ),
]

#: Number of features the ranked metric averages over. `pearson_per_feature` is a MEAN over
#: features, so its noise scales as 1/sqrt(F_eff) -- this column is the explanation for why
#: the two scalar arms are ~20x noisier than the vector one.
N_FEATURES = {"betaxanthin": 1, "beta_carotene": 1, "mulleder19": 19}


def expected_max_of_n(n: int) -> float:
    """Expected maximum of ``n`` iid standard normals (Blom's approximation).

    Blom: E[max] ~ Phi^-1((n - alpha) / (n - 2*alpha + 1)) with alpha = 0.375. Accurate to
    better than 1% for n >= 10, which covers every sweep size here. Used to convert the
    replicate sigma into the selection bias carried by a reported best-of-n.
    """
    if n < 2:
        return 0.0
    alpha = 0.375
    p = (n - alpha) / (n - 2 * alpha + 1)
    return st.NormalDist().inv_cdf(p)


def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation of two equal-length sequences."""
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return num / den if den else float("nan")


def replicate_sigma(trials: list[Any]) -> tuple[float | None, int, list[dict]]:
    """Pooled within-group SD over configurations that TPE happened to re-propose.

    Groups on the exact parameter dict, so a group is the same configuration at the same seed
    -- the difference between its members is pure run-to-run nondeterminism (cuDNN kernel
    selection, dataloader ordering, GPU reduction order) propagated through a chaotic training
    trajectory and then through a max over epochs.
    """
    groups: dict[tuple, list[tuple[int, float]]] = defaultdict(list)
    for t in trials:
        groups[tuple(sorted(t.params.items()))].append((t.number, t.values[0]))
    dups = [g for g in groups.values() if len(g) > 1]
    ss, dof = 0.0, 0
    detail = []
    for g in dups:
        vals = [v for _, v in g]
        mean = st.mean(vals)
        ss += sum((v - mean) ** 2 for v in vals)
        dof += len(vals) - 1
        detail.append(
            {
                "trials": [n for n, _ in g],
                "values": [round(v, 4) for v in vals],
                "range": round(max(vals) - min(vals), 4),
            }
        )
    return (math.sqrt(ss / dof) if dof else None), dof, detail


def lambda_calibration(trials: list[Any]) -> dict:
    """Map ``graph_reg_lambda`` to the MEASURED graph/data loss ratio, and locate parity.

    ratio = graph term / data term, logged per trial as ``val/graph_reg/ratio_to_data``.
    Parity (ratio = 1) is the point where the graph prior and the data loss contribute
    equally; the `_003` grids are decades around it. The relationship is slightly SUB-linear
    (ratio/lambda falls as lambda rises, because the data term moves too), so the slope is
    taken as the median of the per-point slopes rather than fitted through the origin.
    """
    by_lambda: dict[float, list[float]] = defaultdict(list)
    for t in trials:
        lam = t.params.get("graph_reg_lambda")
        ratio = t.user_attrs.get("graph_reg_ratio")
        if lam and ratio is not None:
            by_lambda[lam].append(ratio)
    points = {
        lam: {"ratio_median": round(st.median(rs), 4), "n": len(rs)}
        for lam, rs in sorted(by_lambda.items())
    }
    slopes = [st.median(rs) / lam for lam, rs in by_lambda.items() if lam]
    if not slopes:
        return {"points": points, "slope": None, "parity_lambda": None}
    slope = st.median(slopes)
    return {
        "points": points,
        "slope_ratio_per_lambda": round(slope, 1),
        "parity_lambda": float(f"{1.0 / slope:.2g}"),
    }


def marginal_effects(trials: list[Any]) -> dict:
    """Mean objective per level of each categorical axis.

    CAVEAT, and it is the reason `_003` switches sampler: `_002` used TPE, which allocates
    trials toward the region it currently believes is good. Level counts are therefore
    UNBALANCED and these means are confounded with sampling order -- a level with n = 2 was
    not given a fair hearing. Recorded with `n` so that is visible rather than implicit.
    """
    out = {}
    keys = sorted({k for t in trials for k in t.params})
    for k in keys:
        groups: dict[Any, list[float]] = defaultdict(list)
        for t in trials:
            if k in t.params:
                groups[t.params[k]].append(t.values[0])
        rows = sorted(groups.items(), key=lambda kv: -st.mean(kv[1]))
        out[k] = {
            "levels": [
                {"value": str(v), "mean": round(st.mean(vs), 4), "n": len(vs)}
                for v, vs in rows
            ],
            "spread": round(st.mean(rows[0][1]) - st.mean(rows[-1][1]), 4),
        }
    return out


def analyze(arm: str, exp_dir: str, db_file: str, study_name: str) -> dict:
    """Full `_002` readout for one arm."""
    path = osp.join(SCRATCH_EXP, exp_dir, "optuna", db_file)
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{path}")
    trials = [t for t in study.trials if t.state.name == "COMPLETE" and t.values]
    values = sorted((t.values[0] for t in trials), reverse=True)
    sigma, dof, dup_detail = replicate_sigma(trials)
    durations = [
        (t.datetime_complete - t.datetime_start).total_seconds() / 60
        for t in trials
        if t.datetime_complete
    ]
    objectives = [t.values[0] for t in trials if t.datetime_complete]
    # Within a single architecture cell, so "bigger model is slower" cannot explain the
    # correlation. hidden=90 is the modal cell on all three arms.
    cell = [
        t
        for t in trials
        if t.params.get("hidden_channels") == 90 and t.datetime_complete
    ]
    res: dict[str, Any] = {
        "study": study_name,
        "db": path,
        "n_features_in_metric": N_FEATURES[arm],
        "n_trials": len(study.trials),
        "n_complete": len(trials),
        "states": dict(Counter(t.state.name for t in study.trials)),
        "values": {
            "max": round(values[0], 4),
            "median": round(st.median(values), 4),
            "min": round(values[-1], 4),
            "sd_between_trials": round(st.pstdev(values), 4),
            "top5": [round(v, 4) for v in values[:5]],
            "top5_spread": round(values[0] - values[4], 4)
            if len(values) >= 5
            else None,
        },
        "replicate_noise": {
            "sigma": round(sigma, 4) if sigma else None,
            "dof": dof,
            "groups": dup_detail,
            "usable": dof >= 2,
        },
        "duration_min": {
            "median": round(st.median(durations), 1),
            "min": round(min(durations), 1),
            "max": round(max(durations), 1),
        },
        "run_length_confound": {
            "r_duration_objective": round(pearson(durations, objectives), 3),
            "r_within_hidden90": (
                round(
                    pearson(
                        [
                            (t.datetime_complete - t.datetime_start).total_seconds()
                            / 60
                            for t in cell
                        ],
                        [t.values[0] for t in cell],
                    ),
                    3,
                )
                if len(cell) > 8
                else None
            ),
            "n_within_cell": len(cell),
        },
        "lambda_calibration": lambda_calibration(trials),
        "marginal_effects_TPE_CONFOUNDED": marginal_effects(trials),
    }
    if sigma and len(values) >= 5:
        e_max = expected_max_of_n(len(values))
        res["selection_inflation"] = {
            "expected_max_of_n_sd_units": round(e_max, 3),
            "inflation": round(e_max * sigma, 4),
            "reported_best": round(values[0], 4),
            "bias_corrected_floor": round(values[0] - e_max * sigma, 4),
            "top5_spread_in_sigma": round((values[0] - values[4]) / sigma, 2),
            "top5_separable": (values[0] - values[4]) > 2 * sigma,
            "replicates_for_sigma_eff_0p010": math.ceil((sigma / 0.010) ** 2),
        }
    return res


def main() -> None:
    """Run the analysis for all three arms and write the JSON + stdout summary."""
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {arm: analyze(arm, d, f, s) for arm, d, f, s in ARMS}
    out = osp.join(OUT_DIR, "analysis_002_noise_and_lambda.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)

    print("=" * 78)
    print("_002 REPLICATE NOISE  (identical config AND identical seed 42)")
    print("=" * 78)
    for arm, r in report.items():
        rn = r["replicate_noise"]
        print(f"\n{arm}  (F = {r['n_features_in_metric']} in the ranked metric)")
        for g in rn["groups"]:
            print(f"    trials {g['trials']}: {g['values']}  range={g['range']}")
        note = "" if rn["usable"] else "   <-- dof too low to use on its own"
        print(f"    pooled sigma = {rn['sigma']} (dof {rn['dof']}){note}")
        si = r.get("selection_inflation")
        if si:
            print(
                f"    top5 {r['values']['top5']} spread={r['values']['top5_spread']} "
                f"= {si['top5_spread_in_sigma']} sigma -> "
                f"{'separable' if si['top5_separable'] else 'INDISTINGUISHABLE'}"
            )
            print(
                f"    best {si['reported_best']} - selection inflation "
                f"{si['inflation']} -> floor {si['bias_corrected_floor']}"
            )
            print(
                f"    replicates for sigma_eff <= 0.010: "
                f"R = {si['replicates_for_sigma_eff_0p010']}"
            )

    print("\n" + "=" * 78)
    print("GRAPH-PRIOR CALIBRATION  (ratio = graph term / data term)")
    print("=" * 78)
    for arm, r in report.items():
        lc = r["lambda_calibration"]
        pts = ", ".join(
            f"{lam:.1e}->{v['ratio_median']}(n{v['n']})"
            for lam, v in lc["points"].items()
        )
        print(f"\n{arm}: {pts}")
        print(
            f"    slope ~ {lc['slope_ratio_per_lambda']} * lambda  =>  "
            f"PARITY at lambda = {lc['parity_lambda']}"
        )

    print("\n" + "=" * 78)
    print("RUN-LENGTH CONFOUND  (r between trial duration and the ranked objective)")
    print("=" * 78)
    for arm, r in report.items():
        rl = r["run_length_confound"]
        print(
            f"  {arm:14s} r = {rl['r_duration_objective']:+.3f}   "
            f"within hidden=90: {rl['r_within_hidden90']} (n={rl['n_within_cell']})"
        )

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
