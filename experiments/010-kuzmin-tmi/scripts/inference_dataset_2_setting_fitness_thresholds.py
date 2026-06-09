# experiments/010-kuzmin-tmi/scripts/inference_dataset_2_setting_fitness_thresholds.py
# [[experiments.010-kuzmin-tmi.scripts.inference_dataset_2_setting_fitness_thresholds]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/inference_dataset_2_setting_fitness_thresholds

"""
Statistical analysis for setting fitness thresholds in inference_dataset_2.

Goal: Determine thresholds for SMF and DMF such that we can claim statistically
significant iterative fitness improvement:
    WT (1.0) → Single (SMF) → Double (DMF) → Triple (TMF)

Each step must be distinguishable given measurement noise and replicates.

Key data sources:
- SmfCostanzo2016: Mean fitness stddev = 0.0633
- DmfCostanzo2016: Mean fitness stddev = 0.0424 (4 replicates)
- TmfKuzmin2018: Mean fitness stddev = 0.0692
- TmfKuzmin2020: Mean fitness stddev = 0.0529

Reference: notes/scratch.2026.01.28.142530-inference-dataset-2.fitness-noise-across-costanzo-kuzmin.md
"""

import numpy as np
from scipy import stats


def print_section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def compute_standard_error(sd: float, n: int) -> float:
    """Compute standard error of the mean."""
    return sd / np.sqrt(n)


def compute_t_statistic(observed: float, null: float, se: float) -> float:
    """Compute t-statistic for one-sample test."""
    return (observed - null) / se


def compute_t_statistic_two_sample(mean1: float, mean2: float, se1: float, se2: float) -> float:
    """Compute t-statistic for comparing two means (Welch's t-test)."""
    se_diff = np.sqrt(se1**2 + se2**2)
    return (mean1 - mean2) / se_diff


def get_critical_t(alpha: float, df: float, one_tailed: bool = True) -> float:
    """Get critical t-value for significance testing."""
    if one_tailed:
        return float(stats.t.ppf(1 - alpha, df))
    else:
        return float(stats.t.ppf(1 - alpha / 2, df))


def compute_welch_df(se1: float, se2: float, n1: int, n2: int) -> float:
    """Compute Welch-Satterthwaite degrees of freedom."""
    num = (se1**2 + se2**2) ** 2
    denom = (se1**4 / (n1 - 1)) + (se2**4 / (n2 - 1))
    return num / denom


def main():
    print_section("STATISTICAL ANALYSIS FOR FITNESS THRESHOLDS")
    print("Inference Dataset 2: Orthogonal Fitness Improvement")
    print("\nGoal: Set thresholds to ensure statistically significant")
    print("      iterative fitness improvement at each deletion step.")

    # =========================================================================
    # SECTION 1: Noise Parameters from Datasets
    # =========================================================================
    print_section("1. NOISE PARAMETERS FROM DATASETS")

    # Standard deviations from dataset measurements
    # These are the MEAN stddev across all measurements in each dataset
    sd_smf = 0.0633  # SmfCostanzo2016
    sd_dmf = 0.0424  # DmfCostanzo2016
    sd_tmf_2018 = 0.0692  # TmfKuzmin2018
    sd_tmf_2020 = 0.0529  # TmfKuzmin2020

    # Number of replicates (Costanzo2016 uses 4 replicates for DMF)
    n_replicates = 4

    print(f"\nDataset noise (standard deviation of fitness measurements):")
    print(f"  SMF (Costanzo2016):  σ = {sd_smf:.4f}")
    print(f"  DMF (Costanzo2016):  σ = {sd_dmf:.4f}  [4 replicates]")
    print(f"  TMF (Kuzmin2018):    σ = {sd_tmf_2018:.4f}")
    print(f"  TMF (Kuzmin2020):    σ = {sd_tmf_2020:.4f}")

    print(f"\nNumber of replicates assumed: n = {n_replicates}")

    # =========================================================================
    # SECTION 2: Standard Error Calculations
    # =========================================================================
    print_section("2. STANDARD ERROR CALCULATIONS")

    se_smf = compute_standard_error(sd_smf, n_replicates)
    se_dmf = compute_standard_error(sd_dmf, n_replicates)
    se_tmf = compute_standard_error(sd_tmf_2020, n_replicates)  # Use 2020 (lower noise)

    print(f"\nStandard Error = SD / √n")
    print(f"\n  SE_SMF = {sd_smf:.4f} / √{n_replicates} = {se_smf:.4f}")
    print(f"  SE_DMF = {sd_dmf:.4f} / √{n_replicates} = {se_dmf:.4f}")
    print(f"  SE_TMF = {sd_tmf_2020:.4f} / √{n_replicates} = {se_tmf:.4f}")

    # Combined SE for comparing two measurements
    se_smf_vs_wt = se_smf  # Comparing to fixed WT = 1.0 (no variance)
    se_dmf_vs_smf = np.sqrt(se_dmf**2 + se_smf**2)
    se_tmf_vs_dmf = np.sqrt(se_tmf**2 + se_dmf**2)

    print(f"\nCombined SE for pairwise comparisons:")
    print(f"  SE(SMF vs WT):  {se_smf_vs_wt:.4f}  [WT has no variance]")
    print(f"  SE(DMF vs SMF): √(SE_DMF² + SE_SMF²) = {se_dmf_vs_smf:.4f}")
    print(f"  SE(TMF vs DMF): √(SE_TMF² + SE_DMF²) = {se_tmf_vs_dmf:.4f}")

    # =========================================================================
    # SECTION 3: Critical t-values
    # =========================================================================
    print_section("3. CRITICAL t-VALUES FOR SIGNIFICANCE")

    alpha_05 = 0.05
    alpha_01 = 0.01

    df_one_sample = n_replicates - 1  # df = 3
    df_two_sample = compute_welch_df(se_smf, se_dmf, n_replicates, n_replicates)

    t_crit_05_one = get_critical_t(alpha_05, df_one_sample, one_tailed=True)
    t_crit_01_one = get_critical_t(alpha_01, df_one_sample, one_tailed=True)
    t_crit_05_two = get_critical_t(alpha_05, df_two_sample, one_tailed=True)
    t_crit_01_two = get_critical_t(alpha_01, df_two_sample, one_tailed=True)

    print(f"\nOne-sample t-test (SMF vs WT = 1.0):")
    print(f"  df = n - 1 = {df_one_sample}")
    print(f"  t_critical (α = 0.05, one-tailed) = {t_crit_05_one:.3f}")
    print(f"  t_critical (α = 0.01, one-tailed) = {t_crit_01_one:.3f}")

    print(f"\nTwo-sample t-test (DMF vs SMF, TMF vs DMF):")
    print(f"  df (Welch-Satterthwaite) ≈ {df_two_sample:.1f}")
    print(f"  t_critical (α = 0.05, one-tailed) = {t_crit_05_two:.3f}")
    print(f"  t_critical (α = 0.01, one-tailed) = {t_crit_01_two:.3f}")

    # =========================================================================
    # SECTION 4: Test Proposed Thresholds (SMF > 1.1, DMF > 1.15)
    # =========================================================================
    print_section("4. TESTING PROPOSED THRESHOLDS")

    smf_proposed = 1.10
    dmf_proposed = 1.15
    wt = 1.0

    print(f"\nProposed thresholds:")
    print(f"  SMF > {smf_proposed}")
    print(f"  DMF > {dmf_proposed}")

    # Test 1: SMF vs WT
    t_smf_vs_wt = compute_t_statistic(smf_proposed, wt, se_smf_vs_wt)
    p_smf_vs_wt = float(1 - stats.t.cdf(t_smf_vs_wt, df_one_sample))

    print(f"\n--- Test 1: Is SMF = {smf_proposed} significantly > WT = {wt}? ---")
    print(f"  Difference: {smf_proposed - wt:.2f}")
    print(f"  t-statistic: ({smf_proposed} - {wt}) / {se_smf_vs_wt:.4f} = {t_smf_vs_wt:.3f}")
    print(f"  p-value (one-tailed): {p_smf_vs_wt:.4f}")
    print(f"  t_critical (α=0.05): {t_crit_05_one:.3f}")

    if t_smf_vs_wt > t_crit_05_one:
        print(f"  SIGNIFICANT at α = 0.05 ({t_smf_vs_wt:.3f} > {t_crit_05_one:.3f})")
    else:
        print(f"  NOT significant at α = 0.05 ({t_smf_vs_wt:.3f} < {t_crit_05_one:.3f})")

    if t_smf_vs_wt > t_crit_01_one:
        print(f"  SIGNIFICANT at α = 0.01 ({t_smf_vs_wt:.3f} > {t_crit_01_one:.3f})")
    else:
        print(f"  NOT significant at α = 0.01 ({t_smf_vs_wt:.3f} < {t_crit_01_one:.3f})")

    # Test 2: DMF vs SMF
    t_dmf_vs_smf = compute_t_statistic_two_sample(dmf_proposed, smf_proposed, se_dmf, se_smf)
    p_dmf_vs_smf = float(1 - stats.t.cdf(t_dmf_vs_smf, df_two_sample))

    print(f"\n--- Test 2: Is DMF = {dmf_proposed} significantly > SMF = {smf_proposed}? ---")
    print(f"  Difference: {dmf_proposed - smf_proposed:.2f}")
    print(f"  SE_combined: {se_dmf_vs_smf:.4f}")
    print(f"  t-statistic: ({dmf_proposed} - {smf_proposed}) / {se_dmf_vs_smf:.4f} = {t_dmf_vs_smf:.3f}")
    print(f"  p-value (one-tailed): {p_dmf_vs_smf:.4f}")
    print(f"  t_critical (α=0.05): {t_crit_05_two:.3f}")

    if t_dmf_vs_smf > t_crit_05_two:
        print(f"  SIGNIFICANT at α = 0.05 ({t_dmf_vs_smf:.3f} > {t_crit_05_two:.3f})")
    else:
        print(f"  NOT significant at α = 0.05 ({t_dmf_vs_smf:.3f} < {t_crit_05_two:.3f})")

    # =========================================================================
    # SECTION 5: Compute Required Thresholds
    # =========================================================================
    print_section("5. REQUIRED THRESHOLDS FOR SIGNIFICANCE")

    print("\n--- Minimum SMF to be significantly > WT at α = 0.05 ---")
    min_smf_diff = t_crit_05_one * se_smf_vs_wt
    min_smf = wt + min_smf_diff
    print(f"  Required difference: t_crit × SE = {t_crit_05_one:.3f} × {se_smf_vs_wt:.4f} = {min_smf_diff:.4f}")
    print(f"  Minimum SMF threshold: {wt} + {min_smf_diff:.4f} = {min_smf:.3f}")

    print("\n--- Minimum DMF-SMF gap to be significantly different at α = 0.05 ---")
    min_dmf_smf_gap = t_crit_05_two * se_dmf_vs_smf
    print(f"  Required gap: t_crit × SE_combined = {t_crit_05_two:.3f} × {se_dmf_vs_smf:.4f} = {min_dmf_smf_gap:.4f}")

    print("\n--- Required DMF thresholds for various SMF values ---")
    print(f"  {'SMF':>6} | {'Min DMF':>8} | {'Gap':>6}")
    print(f"  {'-'*6} | {'-'*8} | {'-'*6}")
    for smf in [1.05, 1.08, 1.10, 1.12, 1.15]:
        min_dmf = smf + min_dmf_smf_gap
        print(f"  {smf:>6.2f} | {min_dmf:>8.3f} | {min_dmf_smf_gap:>6.3f}")

    # =========================================================================
    # SECTION 6: Recommended Thresholds
    # =========================================================================
    print_section("6. RECOMMENDED THRESHOLDS")

    # Option 1: Conservative (both steps significant at α = 0.05)
    smf_recommended = 1.10
    dmf_recommended = smf_recommended + min_dmf_smf_gap

    print("\nOPTION 1: Both steps significant at α = 0.05")
    print(f"  SMF threshold: > {smf_recommended:.2f}")
    print(f"  DMF threshold: > {dmf_recommended:.2f} (= SMF + {min_dmf_smf_gap:.3f})")
    print("  Pros: Strong statistical evidence for iterative improvement")
    print("  Cons: May exclude genes with moderate but real effects")

    # Option 2: Relaxed (lower thresholds, more genes)
    smf_relaxed = 1.08
    dmf_relaxed = smf_relaxed + min_dmf_smf_gap

    print("\nOPTION 2: Minimum significant thresholds")
    print(f"  SMF threshold: > {smf_relaxed:.2f}")
    print(f"  DMF threshold: > {dmf_relaxed:.2f} (= SMF + {min_dmf_smf_gap:.3f})")
    print("  Pros: Captures more genes while maintaining significance")
    print("  Cons: Closer to noise floor")

    # =========================================================================
    # SECTION 7: Summary Table
    # =========================================================================
    print_section("7. SUMMARY: THRESHOLD COMPARISON")

    # Original proposal
    t_orig_smf = compute_t_statistic(1.10, 1.0, se_smf_vs_wt)
    t_orig_dmf = compute_t_statistic_two_sample(1.15, 1.10, se_dmf, se_smf)
    sig_orig_smf = "p<0.05" if t_orig_smf > t_crit_05_one else "n.s."
    sig_orig_dmf = "p<0.05" if t_orig_dmf > t_crit_05_two else "n.s."

    # Recommended
    t_rec_smf = compute_t_statistic(smf_recommended, 1.0, se_smf_vs_wt)
    t_rec_dmf = compute_t_statistic_two_sample(dmf_recommended, smf_recommended, se_dmf, se_smf)
    sig_rec_smf = "p<0.05" if t_rec_smf > t_crit_05_one else "n.s."
    sig_rec_dmf = "p<0.05" if t_rec_dmf > t_crit_05_two else "n.s."

    print("\nOriginal Proposal:")
    print(f"  SMF vs WT=1.0:  SMF > 1.10,  t = {t_orig_smf:.2f},  {sig_orig_smf}")
    print(f"  DMF vs SMF:     DMF > 1.15,  t = {t_orig_dmf:.2f},  {sig_orig_dmf}")

    print("\nRecommended Thresholds:")
    print(f"  SMF vs WT=1.0:  SMF > {smf_recommended:.2f},  t = {t_rec_smf:.2f},  {sig_rec_smf}")
    print(f"  DMF vs SMF:     DMF > {dmf_recommended:.2f},  t = {t_rec_dmf:.2f},  {sig_rec_dmf}")

    # =========================================================================
    # SECTION 8: Final Recommendations
    # =========================================================================
    print_section("8. FINAL RECOMMENDATIONS")

    print(f"\nBased on the statistical analysis with n = {n_replicates} replicates:")

    print("\nORIGINAL PROPOSAL:")
    print(f"  SMF > 1.10  →  Significant vs WT (t = {t_orig_smf:.2f}, p < 0.05)")
    print(f"  DMF > 1.15  →  NOT significant vs SMF=1.10 (t = {t_orig_dmf:.2f}, gap too small)")

    print(f"\nPROBLEM: The gap between DMF (1.15) and SMF (1.10) is only 0.05, but we need")
    print(f"         at least {min_dmf_smf_gap:.3f} to achieve statistical significance.")

    print("\nRECOMMENDED THRESHOLDS FOR ITERATIVE IMPROVEMENT:")
    print(f"  SMF > {smf_recommended:.2f}")
    print(f"  DMF > {dmf_recommended:.2f}  (ensures DMF significantly > SMF at p < 0.05)")

    print("\nThis ensures that when we measure:")
    print(f"  1. Single mutant fitness > {smf_recommended:.2f}  →  Single beats WT (p < 0.05)")
    print(f"  2. Double mutant fitness > {dmf_recommended:.2f}  →  Double beats single (p < 0.05)")
    print("\nBoth improvements are statistically defensible with 4 replicates.")

    # =========================================================================
    # SECTION 9: Higher Replicate Analysis (16x24 plate = 384 wells)
    # =========================================================================
    print_section("9. HIGHER REPLICATE ANALYSIS (16x24 PLATE)")

    print("\nWith 384-well plates, we can achieve 16 or 24 replicates per strain.")
    print("This dramatically improves our statistical power.\n")

    for n_rep in [4, 8, 16, 24]:
        se_smf_n = sd_smf / np.sqrt(n_rep)
        se_dmf_n = sd_dmf / np.sqrt(n_rep)
        se_combined_n = np.sqrt(se_smf_n**2 + se_dmf_n**2)

        # Degrees of freedom
        df_one_n = n_rep - 1
        df_two_n = compute_welch_df(se_smf_n, se_dmf_n, n_rep, n_rep)

        # Critical t-values
        t_crit_one_n = get_critical_t(0.05, df_one_n, one_tailed=True)
        t_crit_two_n = get_critical_t(0.05, df_two_n, one_tailed=True)

        # Required gaps
        min_smf_gap_n = t_crit_one_n * se_smf_n
        min_dmf_gap_n = t_crit_two_n * se_combined_n

        print(f"--- n = {n_rep} replicates ---")
        print(f"  SE_SMF = {se_smf_n:.4f},  SE_DMF = {se_dmf_n:.4f},  SE_combined = {se_combined_n:.4f}")
        print(f"  t_crit (one-sample, df={df_one_n}) = {t_crit_one_n:.3f}")
        print(f"  t_crit (two-sample, df={df_two_n:.1f}) = {t_crit_two_n:.3f}")
        print(f"  Min SMF to beat WT:     SMF > {1.0 + min_smf_gap_n:.3f}")
        print(f"  Min DMF-SMF gap:        {min_dmf_gap_n:.4f}")
        print(f"  If SMF = 1.10, need:    DMF > {1.10 + min_dmf_gap_n:.3f}")
        print(f"  If SMF = 1.05, need:    DMF > {1.05 + min_dmf_gap_n:.3f}")
        print()

    # =========================================================================
    # SECTION 10: Feasibility Check with Dataset Percentiles
    # =========================================================================
    print_section("10. FEASIBILITY CHECK: DATA AVAILABILITY")

    print("\nEmpirical data from Costanzo2016 and Kuzmin2020:")
    print("  DmfCostanzo2016 total pairs: 20,705,612")
    print("  TmfKuzmin2020 total triples: 301,798")

    print("\n--- Pairs/Triples available at different DMF thresholds ---")
    # These are from the user's empirical analysis
    dmf_thresholds = [
        (1.10, "~1%", "~200,000"),
        (1.15, "~0.3%", "~60,000"),
        (1.18, "0.07%", "14,748"),
    ]

    print(f"  {'DMF Threshold':>14} | {'Percentile':>12} | {'Approx Pairs':>15}")
    print(f"  {'-'*14} | {'-'*12} | {'-'*15}")
    for thresh, pct, count in dmf_thresholds:
        print(f"  {thresh:>14.2f} | {pct:>12} | {count:>15}")

    # =========================================================================
    # SECTION 11: Recommended Strategy with 16-24 Replicates
    # =========================================================================
    print_section("11. RECOMMENDED STRATEGY WITH 16-24 REPLICATES")

    # Calculate for n=16
    n_optimal = 16
    se_smf_opt = sd_smf / np.sqrt(n_optimal)
    se_dmf_opt = sd_dmf / np.sqrt(n_optimal)
    se_combined_opt = np.sqrt(se_smf_opt**2 + se_dmf_opt**2)
    df_two_opt = compute_welch_df(se_smf_opt, se_dmf_opt, n_optimal, n_optimal)
    t_crit_two_opt = get_critical_t(0.05, df_two_opt, one_tailed=True)
    min_gap_opt = t_crit_two_opt * se_combined_opt

    print(f"\nWith n = {n_optimal} replicates:")
    print(f"  Required DMF-SMF gap for significance: {min_gap_opt:.4f}")

    smf_final = 1.10
    dmf_final = smf_final + min_gap_opt

    print(f"\nFINAL RECOMMENDED THRESHOLDS (n={n_optimal}):")
    print(f"  SMF > {smf_final:.2f}")
    print(f"  DMF > {dmf_final:.3f}")

    print(f"\nThis is MUCH more feasible than DMF > 1.18 required with n=4!")
    print(f"  With n=4:  DMF > 1.18 needed  (only 0.07% of pairs qualify)")
    print(f"  With n=16: DMF > {dmf_final:.2f} needed  (many more pairs qualify)")

    # Verify the t-statistics
    t_smf_final = (smf_final - 1.0) / se_smf_opt
    t_dmf_final = (dmf_final - smf_final) / se_combined_opt

    print(f"\nVerification (both should exceed t_critical):")
    print(f"  SMF vs WT:   t = {t_smf_final:.2f}  (t_crit ≈ 1.75)")
    print(f"  DMF vs SMF:  t = {t_dmf_final:.2f}  (t_crit ≈ {t_crit_two_opt:.2f})")

    # =========================================================================
    # SECTION 12: FULL ITERATIVE CHAIN (WT → SMF → DMF → TMF)
    # =========================================================================
    print_section("12. FULL ITERATIVE CHAIN: WT → SMF → DMF → TMF")

    print("\nAssuming similar noise on triples (using TmfKuzmin2020 σ = 0.0529)")
    print("We need THREE significant steps for true iterative improvement:\n")

    for n_rep in [8, 16, 24]:
        print(f"--- n = {n_rep} replicates ---")

        # Standard errors
        se_smf_n = sd_smf / np.sqrt(n_rep)
        se_dmf_n = sd_dmf / np.sqrt(n_rep)
        se_tmf_n = sd_tmf_2020 / np.sqrt(n_rep)

        # Combined SEs for each comparison
        se_smf_wt = se_smf_n  # WT has no variance
        se_dmf_smf = np.sqrt(se_dmf_n**2 + se_smf_n**2)
        se_tmf_dmf = np.sqrt(se_tmf_n**2 + se_dmf_n**2)

        # Degrees of freedom
        df_one = n_rep - 1
        df_dmf_smf = compute_welch_df(se_smf_n, se_dmf_n, n_rep, n_rep)
        df_tmf_dmf = compute_welch_df(se_dmf_n, se_tmf_n, n_rep, n_rep)

        # Critical t-values
        t_crit_one = get_critical_t(0.05, df_one, one_tailed=True)
        t_crit_dmf_smf = get_critical_t(0.05, df_dmf_smf, one_tailed=True)
        t_crit_tmf_dmf = get_critical_t(0.05, df_tmf_dmf, one_tailed=True)

        # Required gaps
        gap_smf_wt = t_crit_one * se_smf_wt
        gap_dmf_smf = t_crit_dmf_smf * se_dmf_smf
        gap_tmf_dmf = t_crit_tmf_dmf * se_tmf_dmf

        # Build thresholds iteratively
        smf_thresh = 1.0 + gap_smf_wt
        dmf_thresh = smf_thresh + gap_dmf_smf
        tmf_thresh = dmf_thresh + gap_tmf_dmf

        print(f"  Step 1: SMF vs WT   | gap needed: {gap_smf_wt:.4f} | SMF > {smf_thresh:.3f}")
        print(f"  Step 2: DMF vs SMF  | gap needed: {gap_dmf_smf:.4f} | DMF > {dmf_thresh:.3f}")
        print(f"  Step 3: TMF vs DMF  | gap needed: {gap_tmf_dmf:.4f} | TMF > {tmf_thresh:.3f}")
        print(f"  Total fitness gain from WT to TMF: {tmf_thresh - 1.0:.3f}")
        print()

    # =========================================================================
    # SECTION 13: PRACTICAL RECOMMENDATION
    # =========================================================================
    print_section("13. PRACTICAL RECOMMENDATION")

    print("\nFor a 384-well plate experiment with iterative fitness improvement:")

    # Use n=16 as the practical choice
    n_practical = 16
    se_smf_p = sd_smf / np.sqrt(n_practical)
    se_dmf_p = sd_dmf / np.sqrt(n_practical)
    se_tmf_p = sd_tmf_2020 / np.sqrt(n_practical)

    se_smf_wt_p = se_smf_p
    se_dmf_smf_p = np.sqrt(se_dmf_p**2 + se_smf_p**2)
    se_tmf_dmf_p = np.sqrt(se_tmf_p**2 + se_dmf_p**2)

    df_one_p = n_practical - 1
    df_dmf_smf_p = compute_welch_df(se_smf_p, se_dmf_p, n_practical, n_practical)
    df_tmf_dmf_p = compute_welch_df(se_dmf_p, se_tmf_p, n_practical, n_practical)

    t_crit_one_p = get_critical_t(0.05, df_one_p, one_tailed=True)
    t_crit_dmf_smf_p = get_critical_t(0.05, df_dmf_smf_p, one_tailed=True)
    t_crit_tmf_dmf_p = get_critical_t(0.05, df_tmf_dmf_p, one_tailed=True)

    gap_smf_wt_p = t_crit_one_p * se_smf_wt_p
    gap_dmf_smf_p = t_crit_dmf_smf_p * se_dmf_smf_p
    gap_tmf_dmf_p = t_crit_tmf_dmf_p * se_tmf_dmf_p

    # Use slightly higher SMF threshold for robustness
    smf_practical = 1.10
    dmf_practical = smf_practical + gap_dmf_smf_p
    tmf_practical = dmf_practical + gap_tmf_dmf_p

    print(f"\nWith n = {n_practical} replicates per strain:")
    print(f"\n  THRESHOLDS FOR DATASET FILTERING (existing data):")
    print(f"    SMF > {smf_practical:.2f}  (to beat WT)")
    print(f"    DMF > {dmf_practical:.2f}  (to beat SMF)")
    print(f"\n  MODEL PREDICTION (not filtering):")
    print(f"    τ_ijk = triple interaction score (model output)")
    print(f"    High τ_ijk → candidate for experimental validation")
    print(f"\n  EXPERIMENTAL VALIDATION CRITERION:")
    print(f"    Measure f_ijk for top-τ predictions")
    print(f"    Success: f_ijk > max(f_ij) + {gap_tmf_dmf_p:.3f}  (iterative improvement)")

    print(f"\n  STATISTICAL GAPS FOR SIGNIFICANCE:")
    print(f"    SMF - WT  must be > {gap_smf_wt_p:.3f}  (filtering)")
    print(f"    DMF - SMF must be > {gap_dmf_smf_p:.3f}  (filtering)")
    print(f"    TMF - DMF must be > {gap_tmf_dmf_p:.3f}  (validation)")

    print(f"\n  EXPECTED TOTAL FITNESS IMPROVEMENT (WT → TMF): {tmf_practical - 1.0:.3f}")

    print("\n  PLATE LAYOUT SUGGESTION (384 wells):")
    print(f"    - 24 strains × 16 replicates = 384 wells")
    print(f"    - Or: 16 strains × 24 replicates = 384 wells")
    print(f"    - Include WT control with same replicates for reference")


if __name__ == "__main__":
    main()
