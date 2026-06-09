# experiments/010-kuzmin-tmi/scripts/inference_dataset_3_jonckheere_terpstra_thresholds
# [[experiments.010-kuzmin-tmi.scripts.inference_dataset_3_jonckheere_terpstra_thresholds]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/inference_dataset_3_jonckheere_terpstra_thresholds
# Test file: experiments/010-kuzmin-tmi/scripts/test_inference_dataset_3_jonckheere_terpstra_thresholds.py


"""
Jonckheere-Terpstra test analysis for fitness thresholds in inference_dataset_3.

The JT test is designed for ordered alternatives:
    H0: WT = SMF = DMF = TMF (no trend)
    H1: WT ≤ SMF ≤ DMF ≤ TMF (with at least one strict inequality)

Advantages over pairwise t-tests:
1. Single test for the entire chain → No multiple testing correction needed
2. More powerful for detecting monotonic trends
3. Does not require equal gaps between levels

Reference: Jonckheere (1954), Terpstra (1952)
"""

import numpy as np
from scipy import stats
def print_section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def jonckheere_terpstra_statistic(
    groups: list[np.ndarray],
) -> tuple[float, float, float]:
    """
    Compute the Jonckheere-Terpstra test statistic.

    The JT statistic counts the number of times an observation from group i
    is less than an observation from group j (for all i < j).

    Args:
        groups: List of arrays, one per ordered group (e.g., [WT, SMF, DMF, TMF])

    Returns:
        J: The JT statistic (sum of Mann-Whitney U statistics)
        z: The standardized z-score
        p_value: One-tailed p-value for ordered alternative
    """
    k = len(groups)
    n = [len(g) for g in groups]
    N = sum(n)

    # Compute J = sum of U_ij for all i < j
    J = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            # Count how many times group_j > group_i
            for xi in groups[i]:
                for xj in groups[j]:
                    if xj > xi:
                        J += 1
                    elif xj == xi:
                        J += 0.5

    # Expected value under H0
    E_J = (N**2 - sum(ni**2 for ni in n)) / 4

    # Variance under H0 (no ties formula)
    # Var(J) = (N²(2N+3) - Σni²(2ni+3)) / 72
    var_J = (N**2 * (2 * N + 3) - sum(ni**2 * (2 * ni + 3) for ni in n)) / 72

    # Standardized statistic
    z = (J - E_J) / np.sqrt(var_J)

    # One-tailed p-value (testing for increasing trend)
    p_value = 1 - stats.norm.cdf(z)

    return J, z, p_value


def simulate_jt_power(
    means: list[float],
    sd: float,
    n_per_group: int,
    n_simulations: int = 10000,
    alpha: float = 0.05,
) -> float:
    """
    Simulate power of JT test for detecting ordered trend.

    Args:
        means: Expected means for each group [WT, SMF, DMF, TMF]
        sd: Standard deviation (same for all groups)
        n_per_group: Number of replicates per group
        n_simulations: Number of Monte Carlo simulations
        alpha: Significance level

    Returns:
        Estimated power (proportion of simulations rejecting H0)
    """
    rejections = 0
    z_crit = stats.norm.ppf(1 - alpha)

    for _ in range(n_simulations):
        # Generate data under H1
        groups = [np.random.normal(mu, sd, n_per_group) for mu in means]
        _, z, _ = jonckheere_terpstra_statistic(groups)
        if z > z_crit:
            rejections += 1

    return rejections / n_simulations


def main():
    print_section("JONCKHEERE-TERPSTRA TEST FOR ITERATIVE FITNESS IMPROVEMENT")
    print("Inference Dataset 3: Relaxed Thresholds with Trend Testing")

    # =========================================================================
    # SECTION 1: Test Overview
    # =========================================================================
    print_section("1. WHY JONCKHEERE-TERPSTRA TEST?")

    print(
        """
  PROBLEM WITH PAIRWISE TESTS:
    - 3 separate tests: WT vs SMF, SMF vs DMF, DMF vs TMF
    - Bonferroni correction: α/3 = 0.0167 per test
    - Requires large gaps (~0.18) for significance
    - Only 7 genes in Costanzo2016 have SMF > 1.12

  JONCKHEERE-TERPSTRA SOLUTION:
    - Single test for ordered alternative: WT ≤ SMF ≤ DMF ≤ TMF
    - No multiple testing correction needed
    - More powerful for detecting monotonic trends
    - Can detect smaller, consistent gaps

  THE CLAIM:
    "Fitness increases monotonically with each deletion"
    This is exactly what JT tests for!
    """
    )

    # =========================================================================
    # SECTION 2: Dataset Parameters
    # =========================================================================
    print_section("2. NOISE PARAMETERS")

    sd = 0.07  # Worst-case SD (conservative)

    print(f"\n  Using worst-case SD = {sd} across all measurements")
    print("  (This is conservative; actual SDs range from 0.04-0.07)")

    # =========================================================================
    # SECTION 3: Example JT Test
    # =========================================================================
    print_section("3. EXAMPLE: JT TEST ON SIMULATED DATA")

    # Example with modest gaps (0.05)
    np.random.seed(42)
    means_example = [1.00, 1.05, 1.10, 1.15]  # 0.05 gaps
    n_example = 16

    print(f"\n  Simulated scenario:")
    print(f"    WT  = {means_example[0]:.2f}")
    print(f"    SMF = {means_example[1]:.2f}  (gap = 0.05)")
    print(f"    DMF = {means_example[2]:.2f}  (gap = 0.05)")
    print(f"    TMF = {means_example[3]:.2f}  (gap = 0.05)")
    print(f"    SD  = {sd}")
    print(f"    n   = {n_example} replicates per group")

    # Generate example data
    groups_example = [np.random.normal(mu, sd, n_example) for mu in means_example]
    J, z, p = jonckheere_terpstra_statistic(groups_example)

    print(f"\n  JT Test Results:")
    print(f"    J statistic: {J:.1f}")
    print(f"    z-score:     {z:.3f}")
    print(f"    p-value:     {p:.6f}")
    print(f"    Significant at α=0.05? {'YES' if p < 0.05 else 'NO'}")
    print(f"    Significant at α=0.01? {'YES' if p < 0.01 else 'NO'}")

    # =========================================================================
    # SECTION 4: Power Analysis for Different Gap Sizes
    # =========================================================================
    print_section("4. POWER ANALYSIS: WHAT GAP SIZE DO WE NEED?")

    print("\n  Estimating power via Monte Carlo simulation (10,000 iterations)")
    print("  Testing various gap sizes with different replicate counts\n")

    gaps_to_test = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    replicates_to_test = [4, 8, 16]

    print(f"  {'Gap':<8} | ", end="")
    for n in replicates_to_test:
        print(f"{'n=' + str(n):<12} | ", end="")
    print()
    print("  " + "-" * 50)

    power_results = {}
    for gap in gaps_to_test:
        means = [1.00, 1.00 + gap, 1.00 + 2 * gap, 1.00 + 3 * gap]
        powers = []
        for n in replicates_to_test:
            power = simulate_jt_power(means, sd, n, n_simulations=5000)
            powers.append(power)
        power_results[gap] = powers

        print(f"  {gap:<8.2f} | ", end="")
        for power in powers:
            indicator = "✓" if power >= 0.80 else " "
            print(f"{power:>6.1%} {indicator}    | ", end="")
        print()

    print("\n  ✓ = Power ≥ 80% (conventional threshold)")

    # =========================================================================
    # SECTION 5: Threshold Recommendations
    # =========================================================================
    print_section("5. THRESHOLD RECOMMENDATIONS FOR INFERENCE DATASET 3")

    print(
        """
  GOAL: Find thresholds that give us good candidates while maintaining
        statistical power to detect the monotonic trend.

  KEY INSIGHT: With JT test, we don't need each step to be individually
               significant. We need the OVERALL TREND to be significant.
    """
    )

    # Find minimum gap for 80% power at each replicate level
    print("  Minimum gap for 80% power:")
    for i, n in enumerate(replicates_to_test):
        for gap in gaps_to_test:
            if power_results[gap][i] >= 0.80:
                print(f"    n = {n:2d}: gap ≥ {gap:.2f}")
                break
        else:
            print(
                f"    n = {n:2d}: gap > {gaps_to_test[-1]:.2f} (need more replicates)"
            )

    print("\n  RECOMMENDED THRESHOLDS (based on n=16 replicates):")
    recommended_gap = 0.04
    print(
        f"""
    Gap per step: {recommended_gap}

    NEW THRESHOLDING SCHEME FOR INFERENCE 3:
    =========================================
    The claim: "At least one gene shows iterative fitness improvement"
    We do NOT require all genes to exceed 1.0 - only the best one matters.

    FILTERING THRESHOLDS:
      max(smf) > 1.04   (at least one gene shows improvement)
      all(smf) > 0.80   (all genes are viable, relaxed baseline)
      max(dmf) > 1.08   (= max(smf) + 0.04, iterative improvement)
      all(dmf) > 0.80   (all pairs are viable, relaxed baseline)

    VALIDATION EXPECTATION:
      max(tmf) > 1.12   (= max(dmf) + 0.04, measured experimentally)

    JT TEST POWER (gap=0.04):
      n=4:  {power_results[0.04][0]:.0%}
      n=8:  {power_results[0.04][1]:.0%}
      n=16: {power_results[0.04][2]:.0%}

    COMPARISON WITH GAP=0.05:
      n=4:  {power_results[0.05][0]:.0%}
      n=8:  {power_results[0.05][1]:.0%}
      n=16: {power_results[0.05][2]:.0%}
    """
    )

    # =========================================================================
    # SECTION 6: Comparison with Pairwise Approach
    # =========================================================================
    print_section("6. COMPARISON: JT TEST vs PAIRWISE t-TESTS")

    gap = 0.05
    means = [1.00, 1.05, 1.10, 1.15]

    print(f"\n  Scenario: gaps of {gap} with n=16 replicates, SD={sd}")

    # Pairwise approach
    se = sd / np.sqrt(16)
    se_combined = np.sqrt(2) * se

    # For pairwise: need gap > t_crit * SE_combined
    # With Bonferroni: α/3 = 0.0167, z_crit ≈ 2.13
    z_crit_bonf = stats.norm.ppf(1 - 0.05 / 3)
    required_gap_pairwise = z_crit_bonf * se_combined

    print(f"\n  PAIRWISE APPROACH (Bonferroni-corrected):")
    print(f"    SE per measurement:     {se:.4f}")
    print(f"    SE combined:            {se_combined:.4f}")
    print(f"    z-critical (α/3):       {z_crit_bonf:.3f}")
    print(f"    Required gap:           {required_gap_pairwise:.4f}")
    print(
        f"    Our gap ({gap}):        {'SUFFICIENT' if gap >= required_gap_pairwise else 'INSUFFICIENT'}"
    )

    # JT approach
    jt_power = simulate_jt_power(means, sd, 16, n_simulations=5000)
    print(f"\n  JONCKHEERE-TERPSTRA APPROACH:")
    print(f"    No correction needed (single test)")
    print(f"    z-critical (α=0.05):    {stats.norm.ppf(0.95):.3f}")
    print(f"    Power with gap={gap}:   {jt_power:.1%}")
    print(
        f"    Verdict:                {'FEASIBLE' if jt_power >= 0.80 else 'MARGINAL'}"
    )

    # =========================================================================
    # SECTION 7: Data Availability Check
    # =========================================================================
    print_section("7. DATA AVAILABILITY AT RELAXED THRESHOLDS")

    print(
        """
  From Costanzo2016 single mutant fitness quantiles:
    90th percentile: 1.0305
    95th percentile: 1.0395
    99th percentile: 1.0550
    99.9th percentile: 1.0857

  With SMF > 1.05 threshold:
    ~Top 1% of genes qualify (~50-100 genes)
    Much more diversity than SMF > 1.10 (only ~7 genes)

  From Costanzo2016 double mutant fitness:
    (DMF > 1.10).sum() ≈ 200,000+ pairs available
    (DMF > 1.15).sum() ≈ 60,000 pairs available
    """
    )

    # =========================================================================
    # SECTION 8: Practical Workflow
    # =========================================================================
    print_section("8. INFERENCE DATASET 3 WORKFLOW")

    print(
        """
  STEP 1: FILTER SINGLES (SmfCostanzo2016)
    Threshold: SMF > 1.05
    Expected: ~50-100 genes (vs 7 in inference_2)

  STEP 2: FILTER DOUBLES (DmfCostanzo2016)
    Threshold: DMF > 1.10
    Expected: 200,000+ pairs

  STEP 3: GENERATE TRIPLES
    Combinatorial expansion from filtered singles/doubles
    Run through trained model for TMI prediction

  STEP 4: SELECT TOP PREDICTIONS
    Rank by predicted τ (triple interaction score)
    Select top-k for experimental validation

  STEP 5: EXPERIMENTAL VALIDATION
    Measure WT, SMF, DMF, TMF with n=16 replicates
    Apply JT test for monotonic trend

  STATISTICAL CLAIM:
    "We observe a significant monotonic increase in fitness
     from WT through single, double, to triple deletion
     (Jonckheere-Terpstra test, p < 0.05)"
    """
    )

    # =========================================================================
    # SECTION 9: Summary Table
    # =========================================================================
    print_section("9. SUMMARY: INFERENCE 2 vs INFERENCE 3")

    print(
        """
  ┌───────────────────────┬────────────────────────┬────────────────────────┐
  │                       │   INFERENCE 2          │   INFERENCE 3          │
  ├───────────────────────┼────────────────────────┼────────────────────────┤
  │ SMF max threshold     │   max(smf) > 1.10      │   max(smf) > 1.04      │
  │ SMF baseline          │   all(smf) > 1.00      │   all(smf) > 0.80      │
  │ DMF max threshold     │   max(dmf) > 1.13      │   max(dmf) > 1.08      │
  │ DMF baseline          │   all(dmf) > 1.00      │   all(dmf) > 0.80      │
  │ Gap per step          │   ~0.03                │   ~0.04                │
  │ Statistical test      │   3 pairwise t-tests   │   JT trend test        │
  │ Correction            │   Bonferroni (÷3)      │   None needed          │
  │ Power (n=16)          │   Marginal             │   ~96-100%             │
  │ Power (n=8)           │   Low                  │   ~96%                 │
  │ Power (n=4)           │   Very low             │   ~75%                 │
  │ Genes qualifying      │   ~7                   │   TBD (many more)      │
  │ Pairs qualifying      │   ~15,000              │   TBD (many more)      │
  │ Total triples         │   479K                 │   TBD                  │
  └───────────────────────┴────────────────────────┴────────────────────────┘

  KEY INSIGHT:
    Inference 2 required ALL genes/pairs to exceed 1.0 baseline.
    Inference 3 only requires the BEST gene/pair to show improvement.
    This dramatically increases the candidate pool while maintaining
    statistical power for the iterative improvement claim.
    """
    )


if __name__ == "__main__":
    main()
