# experiments/W019-echo-crispr-array/scripts/tau_precision_model.py
# [[experiments.W019-echo-crispr-array.scripts.tau_precision_model]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/W019-echo-crispr-array/scripts/tau_precision_model
"""What precision on tau the assay actually has, and what more replication buys.

Three things the existing scripts do not produce.

1. The variance components from RUN 4, the round that measured the fitnesses tau
   consumes. `next_round_layout.py` and `assay_precision_benchmark.py` both take
   sigma_plate straight from run 3's mean across-plate SD, which
   `between_day_variance.py` later showed is dominated by one failed plate
   (`run3_P2`, 6 of 12 deletions scoring above wild type). Run 4 has no such single
   offender, so its components are the ones to project from. The across-plate SD also
   contains the colony sampling error of each plate mean, so the plate term is only
   recovered after subtracting sigma_colony^2 / n.

2. The delta-method multiplier on SE(tau). The seven terms of

       tau_abc = f_abc - f_ab f_c - f_ac f_b - f_bc f_a + 2 f_a f_b f_c

   are MULTIPLIED, so each SE enters with a coefficient set by the other fitnesses,
   d(tau)/d(f_ab) = -f_c and d(tau)/d(f_a) = 2 f_b f_c - f_bc. Every coefficient is
   +/-1 only when every fitness is 1, and that single point is where sqrt(7) comes
   from. Four run-4 gene triples have all three doubles and all three singles
   measured, so the multiplier is computed on real numbers rather than assumed.

3. The callable floor and the replication needed to move it. "Callable" here is
   |tau| > 1.96 SE(tau), the two-sided 95% convention, stated because an unstated
   significance convention is how the previous floor became unreproducible.

   Targets are counted against each floor twice, on the RAW predictions and on the
   CALIBRATED ones. The calibration line is READ FROM
   `experiments/010-kuzmin-tmi/results/prediction_calibration_stats.csv`, never
   hardcoded: an earlier version of this analysis used a label SD that turned out to
   belong to tmi_kuzmin2018 alone, which pushed predicted tau down about 3x and, with
   sqrt(7) in place of the measured multiplier, produced a spurious "0 of 39". If that
   file is absent this script FAILS; it does not fall back to an assumed slope. The
   calibrated count is reported alongside how many prediction-SDs the targets sit above
   the prediction mean, because a calibration read that far outside its fitted range does
   not license a point estimate.

Colony picking is NOT in any of this. Every strain in runs 2, 3 and 4 was plated from
one picked colony, so sigma_pick is not estimable and is set to zero, which makes every
SE below a LOWER bound on the honest error.

Outputs
  results/run4_variance_components.csv   sigma_colony / sigma_plate / sigma_day and n
  results/run4_tau_multiplier.csv        per measured triangle: fitnesses, coefficients,
                                         multiplier, and SE(tau) at the measured SEs
  results/run4_tau_precision.csv         per plate count: s, SE(tau), callable floor,
                                         and how many of the 39 in-basis targets clear it

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/W019-echo-crispr-array/scripts/tau_precision_model.py
"""

from __future__ import annotations

import itertools
import math
import os
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
EXP_DIR = osp.join(EXPERIMENT_ROOT, "W019-echo-crispr-array")
RESULTS = osp.join(EXP_DIR, "results")
REPO = osp.dirname(EXPERIMENT_ROOT)
TARGETS = osp.join(
    REPO, "experiments/010-kuzmin-tmi/results/inference_3",
    "top_k_constructible_panel12_k200.csv",
)
# Required input, no fallback. LONG format: quantity,value,source. It holds MEASURED
# quantities only; the calibration line is derived here, in the consumer, so two files
# cannot independently compute it and drift.
CALIB = osp.join(REPO, "experiments/010-kuzmin-tmi/results/prediction_calibration_stats.csv")
CALIB_KEYS = (
    "ckpt_M03_val_pearson", "label_val_sd_pop", "label_val_mean",
    "pred_sd_pop", "pred_mean",
)
# The interval of slopes the calibration identity admits from (r, RMSE, sigma_y) alone.
# The derived slope must land inside it; outside means we and the calibration script
# disagree about which checkpoint produced the predictions.
SLOPE_ADMISSIBLE = (0.805, 1.321)

# The eleven-gene panel basis the round is selected from.
PANEL = frozenset((
    "YBR203W", "YDR057W", "YER079W", "YGL087C", "YJR060W", "YKL033W-A",
    "YLL012W", "YLR104W", "YLR312C-B", "YPL046C", "YPL081W",
))
# Two-sided 95%: a tau is callable when |tau| exceeds this many SEs.
Z95 = 1.96
# This round's layout: 65 strains + WT on one 384-well plate.
WELLS_PER_STRAIN = 5
# A typical deletion fitness, for the multiplicative-null multiplier a well-behaved panel
# would give. Contrast this with the MEASURED multiplier, which run 4's inflated fitness
# scale pushes higher.
F_TYPICAL = 0.85
# Kuzmin 2018/2020 median SE(tau), back-solved from tau and p in the LMDBs. The released
# p-value column is ONE-SIDED, which an earlier back-solve missed; reading it as two-sided
# gave 0.031 and understated their error bar by 2.5x.
KUZMIN_SE_TAU = 0.0785


def tau_coefficients(
    f_a: float, f_b: float, f_c: float, f_ab: float, f_ac: float, f_bc: float
) -> list[float]:
    """d(tau)/d(each measured fitness), ordered f_abc, f_ab, f_ac, f_bc, f_a, f_b, f_c."""
    return [
        1.0,
        -f_c,
        -f_b,
        -f_a,
        2.0 * f_b * f_c - f_bc,
        2.0 * f_a * f_c - f_ac,
        2.0 * f_a * f_b - f_ab,
    ]


def read_calibration() -> dict[str, float]:
    """Measured calibration quantities, plus the line DERIVED from them.

    The file is long (quantity, value, source) and holds measurements only. The
    calibration line is a definition, not a measurement, so it is derived here:

        E[tau | tau_hat] = c + a * tau_hat,   a = r * sd(y) / sd(tau_hat),
                                              c = mean(y) - a * mean(tau_hat)

    which is the least-squares regression of the label on the prediction. a > 1 means
    the predictions are UNDER-dispersed and the calibration expands them.
    """
    if not osp.exists(CALIB):
        raise FileNotFoundError(
            f"required calibration artifact not found: {CALIB}. It carries the label SD, "
            "the measured prediction SD and the checkpoint's val Pearson. This script does "
            "NOT assume a slope: an assumed one is what produced the retracted '0 of 39'."
        )
    raw = pd.read_csv(CALIB)
    for col in ("quantity", "value", "source"):
        if col not in raw.columns:
            raise ValueError(f"{CALIB} lacks the '{col}' column")
    val = raw.set_index("quantity")["value"]
    src = raw.set_index("quantity")["source"]
    absent = [k for k in CALIB_KEYS if k not in val.index]
    if absent:
        raise ValueError(f"{CALIB} lacks required quantities {absent}")

    r = float(val["ckpt_M03_val_pearson"])
    sd_y, mean_y = float(val["label_val_sd_pop"]), float(val["label_val_mean"])
    sd_p, mean_p = float(val["pred_sd_pop"]), float(val["pred_mean"])
    slope = r * sd_y / sd_p
    if not SLOPE_ADMISSIBLE[0] <= slope <= SLOPE_ADMISSIBLE[1]:
        raise ValueError(
            f"derived calibration slope {slope:.4f} is outside the admissible interval "
            f"{SLOPE_ADMISSIBLE}. That means this script and the calibration script "
            "disagree about which checkpoint produced the predictions."
        )
    return {
        "val_pearson": r, "label_sd": sd_y, "label_mean": mean_y,
        "pred_sd": sd_p, "pred_mean": mean_p,
        "slope": slope, "intercept": mean_y - slope * mean_p,
        "source_pearson": str(src["ckpt_M03_val_pearson"]),
        "source_pred": str(src["pred_sd_pop"]),
        "source_label": str(src["label_val_sd_pop"]),
    }


def variance_components() -> dict[str, float]:
    """Split run-4 fitness variation into colony, plate and day terms.

    The plate term is split further into the offset COMMON to every strain on a plate and
    the strain-by-plate INTERACTION, because the two propagate into tau differently. WT
    normalization is supposed to remove the common factor: `next_round_layout.epsilon_se`
    states that "the common plate factor is already removed by normalising to on-plate wild
    type", leaving only interaction. Measuring the split tests that premise rather than
    assuming it.
    """
    by_plate = pd.read_csv(osp.join(RESULTS, "run4_strain_scores_by_plate.csv"))
    day = pd.read_csv(osp.join(RESULTS, "between_day_variance.csv")).iloc[0]

    mut = by_plate[by_plate.strain != "WT"]
    wide = mut.pivot_table(index="strain", columns="plate", values="fitness").dropna()
    sd = mut.pivot_table(index="strain", columns="plate", values="fitness_sd").loc[wide.index]
    n_used = mut.pivot_table(index="strain", columns="plate", values="n_used").loc[wide.index]

    sigma_colony = math.sqrt(float((sd.to_numpy(float) ** 2).mean()))
    n_bar = float(n_used.to_numpy(float).mean())
    # rms, not mean: with 3 plates a per-strain SD is a badly biased estimate of the SD,
    # while the second moment is unbiased for the variance.
    var_across = float((wide.std(axis=1, ddof=1).to_numpy(float) ** 2).mean())
    var_plate = var_across - sigma_colony**2 / n_bar

    # Common plate offset: strain-center, then average down each plate column. Each such
    # average is a mean of len(wide) residuals, so it carries var_plate/len(wide) of its
    # own sampling noise, which is subtracted off.
    centered = wide.sub(wide.mean(axis=1), axis=0)
    var_common = max(0.0, float(centered.mean(axis=0).var(ddof=1)) - var_plate / len(wide))
    return {
        "n_strains": float(len(wide)),
        "n_strain_plate_cells": float(sd.size),
        "colonies_per_strain_plate": n_bar,
        "sigma_colony": sigma_colony,
        "rms_across_plate_sd": math.sqrt(var_across),
        "sigma_plate": math.sqrt(var_plate),
        "sigma_plate_common": math.sqrt(var_common),
        "sigma_plate_interaction": math.sqrt(max(var_plate - var_common, 0.0)),
        "common_share_of_plate_var": var_common / var_plate,
        "sigma_day": float(day.sigma_day_estimate),
        "sigma_day_flagged_plate": day.flagged_plates,
        "sigma_pick": float("nan"),
    }


def measured_triangles() -> pd.DataFrame:
    """Every run-4 gene triple whose three doubles and three singles were measured."""
    boot = pd.read_csv(osp.join(RESULTS, "run4_strain_bootstrap.csv"))
    fit = dict(zip(boot.strain, boot.fitness, strict=True))
    se = dict(zip(boot.strain, boot.boot_se, strict=True))
    genes = sorted({g for s in fit if "+" in s for g in s.split("+")})
    # No triple is built yet, so the SE of f_abc is stood in for by the mean double SE.
    se_abc = float(np.mean([v for k, v in se.items() if "+" in k]))

    rows = []
    for tri in itertools.combinations(genes, 3):
        pair = {frozenset(p): "+".join(sorted(p)) for p in itertools.combinations(tri, 2)}
        if not all(name in fit for name in pair.values()):
            continue
        a, b, c = tri
        f_ab, f_ac, f_bc = ("+".join(sorted(x)) for x in ((a, b), (a, c), (b, c)))
        vals = (fit[a], fit[b], fit[c], fit[f_ab], fit[f_ac], fit[f_bc])
        coef = tau_coefficients(*vals)
        ses = [se_abc, se[f_ab], se[f_ac], se[f_bc], se[a], se[b], se[c]]
        rows.append({
            "triple": "+".join(tri),
            "f_a": fit[a], "f_b": fit[b], "f_c": fit[c],
            "f_ab": fit[f_ab], "f_ac": fit[f_ac], "f_bc": fit[f_bc],
            "coef_f_abc": coef[0], "coef_f_ab": coef[1], "coef_f_ac": coef[2],
            "coef_f_bc": coef[3], "coef_f_a": coef[4], "coef_f_b": coef[5],
            "coef_f_c": coef[6],
            "multiplier": math.sqrt(sum(x * x for x in coef)),
            # The plate-common offset hits all seven terms together, so it enters with the
            # SUM of the coefficients rather than their root sum of squares.
            "sum_coef": sum(coef),
            "se_tau_measured_ses": math.sqrt(
                sum((x * y) ** 2 for x, y in zip(coef, ses, strict=True))
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    """Write the components, the multiplier and the callable floor by plate count."""
    vc = variance_components()
    pd.DataFrame([vc]).to_csv(
        osp.join(RESULTS, "run4_variance_components.csv"), index=False
    )
    print("[1] run-4 variance components")
    for k, v in vc.items():
        print(f"    {k:>26} = {v}")
    print("    sigma_pick is NaN because every strain in runs 2-4 came from ONE picked")
    print("    colony, so the pick term is not estimable and every SE below is a LOWER bound.")

    tri = measured_triangles()
    tri.to_csv(osp.join(RESULTS, "run4_tau_multiplier.csv"), index=False)
    m_bar = float(tri.multiplier.mean())
    print(f"\n[2] SE(tau)/s over {len(tri)} measured triangles")
    print(tri[["triple", "multiplier", "se_tau_measured_ses"]].to_string(index=False))
    print(f"    mean {m_bar:.4f}, range {tri.multiplier.min():.4f}-{tri.multiplier.max():.4f}")
    m_null = math.sqrt(1 + 3 * F_TYPICAL**2 + 3 * F_TYPICAL**4)
    print(f"    sqrt(7) = {math.sqrt(7):.4f} is the value when ALL SEVEN fitnesses are 1;")
    print("    under a multiplicative null at equal single fitness f it is sqrt(1+3f^2+3f^4),")
    for f in (0.85, 0.90, 1.00):
        print(f"      f={f:.2f} -> {math.sqrt(1 + 3 * f**2 + 3 * f**4):.4f}")
    print(f"    MEASURED multiplier {m_bar:.4f} (4 run-4 triples) vs EXPECTED for a")
    print(f"    well-behaved panel at f={F_TYPICAL} {m_null:.4f}. sqrt(7) sits between them,")
    print("    so it OVERSTATES the multiplier below f=1 and UNDERSTATES it at the run-4")
    print("    fitness scale, where the singles score above the on-plate wild type.")

    cal = read_calibration()
    slope, intercept = cal["slope"], cal["intercept"]
    mean_pred, sigma_pred = cal["pred_mean"], cal["pred_sd"]
    print("\n[3] calibration, DERIVED from measured quantities (not read as a fitted line)")
    print(f"    a = r * sd(y) / sd(pred) = {cal['val_pearson']:.6f} * {cal['label_sd']:.6f}"
          f" / {cal['pred_sd']:.6f} = {slope:.6f}")
    print(f"    c = mean(y) - a * mean(pred) = {cal['label_mean']:.6f} - {slope:.6f} * "
          f"{mean_pred:.6f} = {intercept:+.6f}")
    print(f"    admissible interval {SLOPE_ADMISSIBLE}: slope is inside.")
    print(f"    a > 1, so the predictions are UNDER-dispersed and calibration EXPANDS them.")
    print(f"    sources: r <- {cal['source_pearson']}")
    print(f"             pred <- {cal['source_pred']}")
    print(f"             label <- {cal['source_label']}")

    targets = pd.read_csv(TARGETS)
    in_basis = targets[targets.apply(
        lambda r: {r.gene1, r.gene2, r.gene3} <= PANEL, axis=1)]
    raw = in_basis.prediction.to_numpy(float)
    calibrated = intercept + slope * raw
    z_extrap = (raw - mean_pred) / sigma_pred

    # Two routes to tau, computed BOTH rather than chosen between.
    #
    # STRAIN-MEAN (default): average each strain over plates, then form tau, treating the
    #   seven means as independent. `next_round_layout.epsilon_se` argues this is right
    #   after WT normalization, because "the common plate factor is already removed by
    #   normalising to on-plate wild type", leaving strain-by-plate interaction that does
    #   not cancel. Var(tau) = M^2 * [day/D + plate/P + colony/(Pc)].
    # PER-PLATE (alternative): form tau on each plate from that plate's seven fitnesses,
    #   then average. The plate-common offset is then shared by all seven terms within a
    #   plate, so it enters with the SUM of the coefficients rather than their root sum of
    #   squares. Var(tau) = M^2 * [day/D + interaction/P + colony/(Pc)] + S^2 * common/P.
    #
    # The two agree to the extent that WT normalization really did remove the common
    # factor. The gap is the measurement of that premise.
    s_sum = float(tri.sum_coef.mean())
    var_common = vc["sigma_plate_common"] ** 2
    var_inter = vc["sigma_plate_interaction"] ** 2
    rows = []
    for n_plates in (1, 3, 4, 6, 8, 12, 20, 30):
        floor_terms = (
            vc["sigma_day"] ** 2
            + vc["sigma_colony"] ** 2 / (n_plates * WELLS_PER_STRAIN)
        )
        s = math.sqrt(floor_terms + vc["sigma_plate"] ** 2 / n_plates)
        se_sm = m_bar * s
        se_pp = math.sqrt(
            m_bar**2 * (floor_terms + var_inter / n_plates)
            + s_sum**2 * var_common / n_plates
        )
        f_sm, f_pp = Z95 * se_sm, Z95 * se_pp
        rows.append({
            "n_plates": n_plates,
            "wells_per_strain": WELLS_PER_STRAIN,
            "s_strain_one_pick": s,
            "se_tau_strain_mean": se_sm,
            "se_tau_per_plate": se_pp,
            "se_tau_sqrt7": math.sqrt(7) * s,
            "se_tau_null_f085": m_null * s,
            "callable_floor_strain_mean": f_sm,
            "callable_floor_per_plate": f_pp,
            "n_raw_above_floor": int((raw > f_sm).sum()),
            "n_calibrated_above_floor": int((calibrated > f_sm).sum()),
            "n_raw_above_floor_per_plate": int((raw > f_pp).sum()),
            "n_calibrated_above_floor_per_plate": int((calibrated > f_pp).sum()),
            "n_targets": int(len(in_basis)),
            "se_tau_over_kuzmin": se_sm / KUZMIN_SE_TAU,
        })
    prec = pd.DataFrame(rows)
    prec.to_csv(osp.join(RESULTS, "run4_tau_precision.csv"), index=False)
    print(f"\n[4] callability, |tau| > {Z95} SE(tau), against {len(in_basis)} in-basis targets")
    print("    s_strain_one_pick is at ONE pick per strain, the structure actually run;")
    print("    the colony-pick term is not estimable, so every SE here is a LOWER bound.")
    print(prec.round(4).to_string(index=False))
    print(f"    raw predictions       {raw.min():.4f} to {raw.max():.4f}")
    print(f"    calibrated (slope {slope:.4f}, intercept {intercept:+.4f}) "
          f"{calibrated.min():.4f} to {calibrated.max():.4f}")
    print(f"    the targets sit {z_extrap.min():.1f} to {z_extrap.max():.1f} prediction-SDs "
          "above the prediction mean,")
    print("    so the calibrated column is an EXTRAPOLATION and is calibration-dependent.")
    gap = float((prec.se_tau_per_plate / prec.se_tau_strain_mean - 1).abs().max())
    print(f"\n    per-plate vs strain-mean: the plate-common offset is "
          f"{vc['common_share_of_plate_var']:.1%} of the plate variance, so the two routes")
    print(f"    differ by at most {gap:.1%} in SE(tau). WT normalization removed most but "
          "not all of it.")

    denom = vc["sigma_plate"] ** 2 + vc["sigma_colony"] ** 2 / WELLS_PER_STRAIN
    p_004 = denom / (0.04**2 - vc["sigma_day"] ** 2)
    s_kuz = KUZMIN_SE_TAU / m_bar
    p_kuz = denom / (s_kuz**2 - vc["sigma_day"] ** 2)
    print("\n[5] what more replication buys")
    print(f"    per-strain SE 0.04 needs P = {p_004:.1f} plates ({p_004 / 3:.1f}x the current 3),")
    print(f"    and leaves SE(tau) = {m_bar * 0.04:.4f}, "
          f"{m_bar * 0.04 / KUZMIN_SE_TAU:.2f}x Kuzmin's {KUZMIN_SE_TAU}.")
    print(f"    matching Kuzmin needs per-strain SE {s_kuz:.4f}, which is ABOVE the one-day")
    print(f"    floor sigma_day = {vc['sigma_day']:.4f}, so it is reachable within a single day")
    print(f"    at P = {p_kuz:.0f} plates ({p_kuz / 3:.0f}x the current 3).")

    print("\nwrote", osp.join(RESULTS, "run4_variance_components.csv"))
    print("wrote", osp.join(RESULTS, "run4_tau_multiplier.csv"))
    print("wrote", osp.join(RESULTS, "run4_tau_precision.csv"))


if __name__ == "__main__":
    main()
