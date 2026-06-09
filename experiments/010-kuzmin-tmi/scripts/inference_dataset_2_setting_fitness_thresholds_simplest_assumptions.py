# experiments/010-kuzmin-tmi/scripts/inference_dataset_2_setting_fitness_thresholds_simplest_assumptions.py
# [[experiments.010-kuzmin-tmi.scripts.inference_dataset_2_setting_fitness_thresholds_simplest_assumptions]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/inference_dataset_2_setting_fitness_thresholds_simplest_assumptions

"""
Simplified statistical analysis for fitness thresholds.

Reference: notes/scratch.2026.01.30.134219-fitness-hypothesis-testing.md

Assumptions:
- Worst-case SD = 0.07 (max across all datasets)
- For simplicity: SD = SE (treating each measurement as independent)
- WT = 1.00 (reference)
- Goal: Find thresholds where WT ≠ SMF ≠ DMF ≠ TMF (3 pairwise comparisons)
- Use Bonferroni correction for multiple testing
"""

import numpy as np
from scipy import stats


def print_section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def z_test_required_difference(se: float, alpha: float, two_tailed: bool = True) -> float:
    """
    Calculate the minimum difference needed for significance using z-test.

    For large samples or when we assume SD = SE, we use the normal distribution.
    """
    if two_tailed:
        z_crit = stats.norm.ppf(1 - alpha / 2)
    else:
        z_crit = stats.norm.ppf(1 - alpha)
    return z_crit * se


def compute_z_statistic(mean1: float, mean2: float, se: float) -> float:
    """Compute z-statistic for difference of means."""
    return (mean1 - mean2) / se


def compute_p_value(z: float, two_tailed: bool = True) -> float:
    """Compute p-value from z-statistic."""
    if two_tailed:
        return 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        return 1 - stats.norm.cdf(z)


def main():
    print_section("SIMPLIFIED FITNESS THRESHOLD ANALYSIS")
    print("Using worst-case assumptions for conservative estimates")

    # =========================================================================
    # SECTION 1: Assumptions
    # =========================================================================
    print_section("1. ASSUMPTIONS")

    sd = 0.07  # Worst-case SD across all datasets
    se = sd    # Simplifying assumption: SD = SE
    wt = 1.00  # Wild-type reference fitness

    print(f"\n  Standard deviation (worst-case): σ = {sd}")
    print(f"  Standard error (simplified):     SE = {se}")
    print(f"  Wild-type fitness:               WT = {wt}")
    print(f"\n  Note: Assuming SD = SE is conservative (equivalent to n=1)")

    # =========================================================================
    # SECTION 2: Multiple Testing Correction
    # =========================================================================
    print_section("2. MULTIPLE TESTING CORRECTION")

    n_tests = 3  # WT vs SMF, SMF vs DMF, DMF vs TMF
    alpha_nominal = 0.05
    alpha_bonferroni = alpha_nominal / n_tests

    alpha_nominal_01 = 0.01
    alpha_bonferroni_01 = alpha_nominal_01 / n_tests

    print(f"\n  Number of pairwise comparisons: {n_tests}")
    print(f"    1. WT  vs SMF")
    print(f"    2. SMF vs DMF")
    print(f"    3. DMF vs TMF")

    print(f"\n  Bonferroni correction:")
    print(f"    Nominal α = 0.05 → Corrected α = {alpha_bonferroni:.4f}")
    print(f"    Nominal α = 0.01 → Corrected α = {alpha_bonferroni_01:.5f}")

    # =========================================================================
    # SECTION 3: Critical Z-values
    # =========================================================================
    print_section("3. CRITICAL Z-VALUES")

    # Two-tailed tests (conservative)
    z_crit_05 = stats.norm.ppf(1 - alpha_nominal / 2)
    z_crit_05_bonf = stats.norm.ppf(1 - alpha_bonferroni / 2)
    z_crit_01 = stats.norm.ppf(1 - alpha_nominal_01 / 2)
    z_crit_01_bonf = stats.norm.ppf(1 - alpha_bonferroni_01 / 2)

    print(f"\n  Two-tailed z-critical values:")
    print(f"    α = 0.05 (no correction):   z = {z_crit_05:.3f}")
    print(f"    α = 0.05 (Bonferroni):      z = {z_crit_05_bonf:.3f}")
    print(f"    α = 0.01 (no correction):   z = {z_crit_01:.3f}")
    print(f"    α = 0.01 (Bonferroni):      z = {z_crit_01_bonf:.3f}")

    # One-tailed (if we only care about improvement direction)
    z_crit_05_one = stats.norm.ppf(1 - alpha_nominal)
    z_crit_05_bonf_one = stats.norm.ppf(1 - alpha_bonferroni)
    z_crit_01_one = stats.norm.ppf(1 - alpha_nominal_01)
    z_crit_01_bonf_one = stats.norm.ppf(1 - alpha_bonferroni_01)

    print(f"\n  One-tailed z-critical values (directional):")
    print(f"    α = 0.05 (no correction):   z = {z_crit_05_one:.3f}")
    print(f"    α = 0.05 (Bonferroni):      z = {z_crit_05_bonf_one:.3f}")
    print(f"    α = 0.01 (no correction):   z = {z_crit_01_one:.3f}")
    print(f"    α = 0.01 (Bonferroni):      z = {z_crit_01_bonf_one:.3f}")

    # =========================================================================
    # SECTION 4: Required Differences (Two-tailed)
    # =========================================================================
    print_section("4. REQUIRED FITNESS DIFFERENCES (Two-tailed)")

    # Combined SE for comparing two measurements with same SE
    se_combined = np.sqrt(se**2 + se**2)  # = se * sqrt(2)

    print(f"\n  SE for single comparison (vs WT):     {se:.4f}")
    print(f"  SE for pairwise comparison:           {se_combined:.4f}  (SE × √2)")

    # Required gaps
    gap_05 = z_crit_05 * se_combined
    gap_05_bonf = z_crit_05_bonf * se_combined
    gap_01 = z_crit_01 * se_combined
    gap_01_bonf = z_crit_01_bonf * se_combined

    print(f"\n  Minimum gap between adjacent levels:")
    print(f"    p < 0.05 (no correction):   Δ = {gap_05:.4f}")
    print(f"    p < 0.05 (Bonferroni):      Δ = {gap_05_bonf:.4f}")
    print(f"    p < 0.01 (no correction):   Δ = {gap_01:.4f}")
    print(f"    p < 0.01 (Bonferroni):      Δ = {gap_01_bonf:.4f}")

    # =========================================================================
    # SECTION 5: Required Thresholds (Two-tailed, Bonferroni)
    # =========================================================================
    print_section("5. REQUIRED THRESHOLDS (Two-tailed, Bonferroni)")

    print("\n--- At p < 0.05 (family-wise, Bonferroni corrected) ---")
    gap = gap_05_bonf
    smf_05 = wt + gap
    dmf_05 = smf_05 + gap
    tmf_05 = dmf_05 + gap

    print(f"  Required gap between levels: {gap:.4f}")
    print(f"\n  WT  = {wt:.4f}")
    print(f"  SMF > {smf_05:.4f}  (WT + {gap:.4f})")
    print(f"  DMF > {dmf_05:.4f}  (SMF + {gap:.4f})")
    print(f"  TMF > {tmf_05:.4f}  (DMF + {gap:.4f})")
    print(f"\n  Total fitness improvement: {tmf_05 - wt:.4f}")

    print("\n--- At p < 0.01 (family-wise, Bonferroni corrected) ---")
    gap = gap_01_bonf
    smf_01 = wt + gap
    dmf_01 = smf_01 + gap
    tmf_01 = dmf_01 + gap

    print(f"  Required gap between levels: {gap:.4f}")
    print(f"\n  WT  = {wt:.4f}")
    print(f"  SMF > {smf_01:.4f}  (WT + {gap:.4f})")
    print(f"  DMF > {dmf_01:.4f}  (SMF + {gap:.4f})")
    print(f"  TMF > {tmf_01:.4f}  (DMF + {gap:.4f})")
    print(f"\n  Total fitness improvement: {tmf_01 - wt:.4f}")

    # =========================================================================
    # SECTION 6: Required Thresholds (One-tailed, Bonferroni)
    # =========================================================================
    print_section("6. REQUIRED THRESHOLDS (One-tailed, Bonferroni)")
    print("(Use when we only care about fitness IMPROVEMENT)")

    # One-tailed gaps
    gap_05_one = z_crit_05_bonf_one * se_combined
    gap_01_one = z_crit_01_bonf_one * se_combined

    print("\n--- At p < 0.05 (family-wise, Bonferroni, one-tailed) ---")
    gap = gap_05_one
    smf_05_one = wt + gap
    dmf_05_one = smf_05_one + gap
    tmf_05_one = dmf_05_one + gap

    print(f"  Required gap between levels: {gap:.4f}")
    print(f"\n  WT  = {wt:.4f}")
    print(f"  SMF > {smf_05_one:.4f}  (WT + {gap:.4f})")
    print(f"  DMF > {dmf_05_one:.4f}  (SMF + {gap:.4f})")
    print(f"  TMF > {tmf_05_one:.4f}  (DMF + {gap:.4f})")
    print(f"\n  Total fitness improvement: {tmf_05_one - wt:.4f}")

    print("\n--- At p < 0.01 (family-wise, Bonferroni, one-tailed) ---")
    gap = gap_01_one
    smf_01_one = wt + gap
    dmf_01_one = smf_01_one + gap
    tmf_01_one = dmf_01_one + gap

    print(f"  Required gap between levels: {gap:.4f}")
    print(f"\n  WT  = {wt:.4f}")
    print(f"  SMF > {smf_01_one:.4f}  (WT + {gap:.4f})")
    print(f"  DMF > {dmf_01_one:.4f}  (SMF + {gap:.4f})")
    print(f"  TMF > {tmf_01_one:.4f}  (DMF + {gap:.4f})")
    print(f"\n  Total fitness improvement: {tmf_01_one - wt:.4f}")

    # =========================================================================
    # SECTION 7: Summary Table
    # =========================================================================
    print_section("7. SUMMARY TABLE")

    print("\n  Assuming SD = SE = 0.07 (worst-case, n=1 equivalent)")
    print("  SE_combined = 0.07 × √2 = 0.099")
    print("\n  " + "-" * 66)
    print(f"  {'Correction':<20} | {'α':<6} | {'Tails':<8} | {'Gap':<8} | {'SMF':<6} | {'DMF':<6} | {'TMF':<6}")
    print("  " + "-" * 66)

    scenarios = [
        ("None", 0.05, "Two", gap_05, wt + gap_05, wt + 2*gap_05, wt + 3*gap_05),
        ("Bonferroni", 0.05, "Two", gap_05_bonf, smf_05, dmf_05, tmf_05),
        ("None", 0.01, "Two", gap_01, wt + gap_01, wt + 2*gap_01, wt + 3*gap_01),
        ("Bonferroni", 0.01, "Two", gap_01_bonf, smf_01, dmf_01, tmf_01),
        ("Bonferroni", 0.05, "One", gap_05_one, smf_05_one, dmf_05_one, tmf_05_one),
        ("Bonferroni", 0.01, "One", gap_01_one, smf_01_one, dmf_01_one, tmf_01_one),
    ]

    for corr, alpha, tails, gap, smf, dmf, tmf in scenarios:
        print(f"  {corr:<20} | {alpha:<6} | {tails:<8} | {gap:<8.4f} | {smf:<6.3f} | {dmf:<6.3f} | {tmf:<6.3f}")

    print("  " + "-" * 66)

    # =========================================================================
    # SECTION 8: Practical Recommendations
    # =========================================================================
    print_section("8. PRACTICAL RECOMMENDATIONS")

    print("\n  CONSERVATIVE (Two-tailed, Bonferroni, p < 0.05):")
    print(f"    SMF > {smf_05:.2f}")
    print(f"    DMF > {dmf_05:.2f}")
    print(f"    TMF > {tmf_05:.2f}")
    print(f"    Gap needed: {gap_05_bonf:.3f}")

    print("\n  MODERATE (One-tailed, Bonferroni, p < 0.05):")
    print(f"    SMF > {smf_05_one:.2f}")
    print(f"    DMF > {dmf_05_one:.2f}")
    print(f"    TMF > {tmf_05_one:.2f}")
    print(f"    Gap needed: {gap_05_one:.3f}")

    print("\n  RELAXED (One-tailed, No correction, p < 0.05):")
    gap_relax = z_crit_05_one * se_combined
    print(f"    SMF > {wt + gap_relax:.2f}")
    print(f"    DMF > {wt + 2*gap_relax:.2f}")
    print(f"    TMF > {wt + 3*gap_relax:.2f}")
    print(f"    Gap needed: {gap_relax:.3f}")

    # =========================================================================
    # SECTION 9: Verification with Example Values
    # =========================================================================
    print_section("9. VERIFICATION WITH EXAMPLE VALUES")

    print("\n  Testing the moderate recommendation (One-tailed, Bonferroni, p < 0.05):")
    print(f"  Required gap: {gap_05_one:.4f}")

    # Test chain
    test_wt = 1.00
    test_smf = smf_05_one
    test_dmf = dmf_05_one
    test_tmf = tmf_05_one

    pairs = [
        ("WT vs SMF", test_wt, test_smf),
        ("SMF vs DMF", test_smf, test_dmf),
        ("DMF vs TMF", test_dmf, test_tmf),
    ]

    print(f"\n  {'Comparison':<12} | {'Lower':<8} | {'Upper':<8} | {'Diff':<8} | {'z':<8} | {'p (1-tail)':<12} | {'Significant?'}")
    print("  " + "-" * 85)

    for name, lower, upper in pairs:
        diff = upper - lower
        z = compute_z_statistic(upper, lower, se_combined)
        p = compute_p_value(z, two_tailed=False)
        sig = "Yes" if p < alpha_bonferroni else "No"
        print(f"  {name:<12} | {lower:<8.4f} | {upper:<8.4f} | {diff:<8.4f} | {z:<8.3f} | {p:<12.6f} | {sig}")

    print(f"\n  Bonferroni-corrected α for each test: {alpha_bonferroni:.4f}")


if __name__ == "__main__":
    main()
