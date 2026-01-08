#!/usr/bin/env python
"""
Tiered Gene Selection Analysis for Inference Optimization.

This script analyzes overlaps between gene lists and calculates tier sizes
for different selection strategies to reduce the number of triples for inference.

The goal is to find a strategy that:
1. Reduces genes from 2,273 to ~1,000 or less
2. Prioritizes genes that appear in multiple datasets (higher confidence)
3. Always includes small targeted lists (betaxanthin, sameith_doubles)

Tier Strategy:
- Tier 1 (Always Include): betaxanthin + sameith_doubles (small, targeted)
- Tier 2 (Score 3): Genes in ALL three large lists (metabolic, ohya, kemmeren)
- Tier 3 (Score 2): Genes in exactly TWO large lists
- Tier 4 (Score 1): Genes in exactly ONE large list
"""

import os
import os.path as osp
from collections import Counter
from math import comb
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT", "")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT", "")


def load_gene_list(filepath: str) -> set[str]:
    """Load genes from a text file (one gene per line)."""
    genes = set()
    with open(filepath, "r") as f:
        for line in f:
            gene = line.strip()
            # Skip empty lines and backticks (formatting artifacts)
            if gene and not gene.startswith("`"):
                genes.add(gene)
    return genes


def estimate_triples_after_dmf(n_genes: int, dmf_pass_rate: float = 0.14) -> int:
    """
    Estimate triples after DMF filtering.

    Based on observed data: 1.95B raw -> 275M filtered = ~14% pass rate.
    This is approximate since pass rate varies by gene coverage in Costanzo2016.
    """
    raw_triples = comb(n_genes, 3)
    return int(raw_triples * dmf_pass_rate)


def estimate_inference_time_hours(
    n_triples: int, seconds_per_batch: float = 6.5, batch_size: int = 4096
) -> float:
    """
    Estimate inference time in hours.

    Based on observed: ~6.5 seconds per batch of 4096 samples.
    """
    n_batches = n_triples / batch_size
    total_seconds = n_batches * seconds_per_batch
    return total_seconds / 3600


def main():
    # Paths to gene lists
    results_dir = osp.join(
        EXPERIMENT_ROOT, "006-kuzmin-tmi/results/inference_preprocessing_expansion"
    )

    # Load all gene lists
    print("=" * 80)
    print("TIERED GENE SELECTION ANALYSIS")
    print("=" * 80)
    print("\nLoading gene lists...")

    # Small lists (Tier 1 - always include)
    betaxanthin = load_gene_list(
        osp.join(results_dir, "betaxanthin_genes_list_2025-12-13-22-44-42.txt")
    )
    sameith_doubles = load_gene_list(
        osp.join(results_dir, "sameith_doubles_genes_list_2025-12-13-22-44-42.txt")
    )

    # Large lists (for overlap analysis)
    metabolic = load_gene_list(
        osp.join(results_dir, "expanded_metabolic_genes_list_2025-12-13-22-44-42.txt")
    )
    ohya = load_gene_list(
        osp.join(results_dir, "ohya_morphology_genes_list_2025-12-13-22-44-42.txt")
    )
    kemmeren = load_gene_list(
        osp.join(results_dir, "kemmeren_responsive_genes_list_2025-12-13-22-44-42.txt")
    )

    # Current full list (for comparison)
    full_list = load_gene_list(
        osp.join(results_dir, "expanded_genes_inference_1_2025-12-13-22-44-42.txt")
    )

    print("\n" + "=" * 80)
    print("GENE LIST SIZES")
    print("=" * 80)
    print("\nSmall Lists (Tier 1 - Always Include):")
    print(f"  betaxanthin:      {len(betaxanthin):>6} genes")
    print(f"  sameith_doubles:  {len(sameith_doubles):>6} genes")

    print("\nLarge Lists (For Overlap Analysis):")
    print(f"  ohya_morphology:     {len(ohya):>6} genes")
    print(f"  expanded_metabolic:  {len(metabolic):>6} genes")
    print(f"  kemmeren_responsive: {len(kemmeren):>6} genes")

    print("\nCurrent Full List (UNION):")
    print(f"  expanded_genes_inference_1: {len(full_list):>6} genes")

    # Calculate overlaps between large lists
    print("\n" + "=" * 80)
    print("PAIRWISE OVERLAPS (Large Lists)")
    print("=" * 80)

    metabolic_ohya = metabolic & ohya
    metabolic_kemmeren = metabolic & kemmeren
    ohya_kemmeren = ohya & kemmeren

    print(f"\n  metabolic AND ohya:     {len(metabolic_ohya):>6} genes")
    print(f"  metabolic AND kemmeren: {len(metabolic_kemmeren):>6} genes")
    print(f"  ohya AND kemmeren:      {len(ohya_kemmeren):>6} genes")

    # Triple intersection
    triple_intersection = metabolic & ohya & kemmeren
    print(
        f"\n  metabolic AND ohya AND kemmeren (ALL THREE): {len(triple_intersection):>6} genes"
    )

    # Score each gene by how many large lists it appears in
    print("\n" + "=" * 80)
    print("OVERLAP SCORING ANALYSIS")
    print("=" * 80)

    # Count appearances in large lists
    all_large_list_genes = metabolic | ohya | kemmeren
    gene_scores = Counter()

    for gene in all_large_list_genes:
        score = 0
        if gene in metabolic:
            score += 1
        if gene in ohya:
            score += 1
        if gene in kemmeren:
            score += 1
        gene_scores[gene] = score

    # Group by score
    score_3_genes = {g for g, s in gene_scores.items() if s == 3}
    score_2_genes = {g for g, s in gene_scores.items() if s == 2}
    score_1_genes = {g for g, s in gene_scores.items() if s == 1}

    print(f"\n  Score 3 (in ALL 3 large lists):      {len(score_3_genes):>6} genes")
    print(f"  Score 2 (in exactly 2 large lists):  {len(score_2_genes):>6} genes")
    print(f"  Score 1 (in exactly 1 large list):   {len(score_1_genes):>6} genes")
    print("  " + "-" * 45)
    print(f"  Total unique in large lists:         {len(all_large_list_genes):>6} genes")

    # Now calculate tier combinations
    print("\n" + "=" * 80)
    print("TIER COMBINATIONS (with Tier 1 always included)")
    print("=" * 80)

    tier1 = betaxanthin | sameith_doubles
    print(f"\nTier 1 (always included): {len(tier1)} genes")
    print("  (betaxanthin + sameith_doubles, deduplicated)")

    # Different selection strategies
    strategies = [
        ("Tier 1 only", tier1),
        ("Tier 1 + Score 3", tier1 | score_3_genes),
        ("Tier 1 + Score 3 + Score 2", tier1 | score_3_genes | score_2_genes),
        ("Tier 1 + Score >= 2", tier1 | score_3_genes | score_2_genes),
        ("Tier 1 + All Large Lists (Score >= 1)", tier1 | all_large_list_genes),
        ("Current (Full UNION)", full_list),
    ]

    print(
        f"\n{'Strategy':<45} {'Genes':>8} {'Raw C(n,3)':>15} {'Est. DMF':>12} {'Est. Hours':>12}"
    )
    print("-" * 95)

    for name, genes in strategies:
        n = len(genes)
        raw = comb(n, 3)
        est_dmf = estimate_triples_after_dmf(n)
        est_hours = estimate_inference_time_hours(est_dmf)

        # Format large numbers with commas
        raw_str = f"{raw:,}"
        dmf_str = f"{est_dmf:,}"
        hours_str = f"{est_hours:.1f}h"

        # Highlight strategies near 1000 genes
        marker = " ***" if 800 <= n <= 1200 else ""

        print(
            f"{name:<45} {n:>8} {raw_str:>15} {dmf_str:>12} {hours_str:>12}{marker}"
        )

    # Detailed breakdown of Score 2 genes (which pairwise overlap)
    print("\n" + "=" * 80)
    print("SCORE 2 BREAKDOWN (genes in exactly 2 lists)")
    print("=" * 80)

    # Score 2 genes come from pairwise intersections minus the triple intersection
    only_metabolic_ohya = (metabolic & ohya) - triple_intersection
    only_metabolic_kemmeren = (metabolic & kemmeren) - triple_intersection
    only_ohya_kemmeren = (ohya & kemmeren) - triple_intersection

    print(f"\n  In metabolic & ohya (not kemmeren):      {len(only_metabolic_ohya):>6} genes")
    print(f"  In metabolic & kemmeren (not ohya):      {len(only_metabolic_kemmeren):>6} genes")
    print(f"  In ohya & kemmeren (not metabolic):      {len(only_ohya_kemmeren):>6} genes")
    print("  " + "-" * 50)
    print(f"  Total Score 2:                           {len(score_2_genes):>6} genes")

    # Custom strategies targeting ~1000 genes
    print("\n" + "=" * 80)
    print("CUSTOM STRATEGIES TARGETING ~1000 GENES")
    print("=" * 80)

    # Strategy A: Tier 1 + Score 3 + largest pairwise overlap
    strategy_a = tier1 | score_3_genes | only_metabolic_ohya

    # Strategy B: Tier 1 + Score 3 + kemmeren (smallest large list)
    strategy_b = tier1 | score_3_genes | kemmeren

    # Strategy C: Tier 1 + metabolic AND ohya (largest pairwise)
    strategy_c = tier1 | metabolic_ohya

    # Strategy D: Tier 1 + all pairwise intersections (no Score 1)
    strategy_d = tier1 | metabolic_ohya | metabolic_kemmeren | ohya_kemmeren

    custom_strategies = [
        ("A: Tier1 + Score3 + (metabolic&ohya only)", strategy_a),
        ("B: Tier1 + Score3 + full kemmeren", strategy_b),
        ("C: Tier1 + (metabolic&ohya)", strategy_c),
        ("D: Tier1 + all pairwise intersections", strategy_d),
    ]

    print(
        f"\n{'Strategy':<50} {'Genes':>8} {'Est. DMF':>12} {'Est. Hours':>12}"
    )
    print("-" * 85)

    for name, genes in custom_strategies:
        n = len(genes)
        est_dmf = estimate_triples_after_dmf(n)
        est_hours = estimate_inference_time_hours(est_dmf)
        dmf_str = f"{est_dmf:,}"
        hours_str = f"{est_hours:.1f}h"
        marker = " ***" if 800 <= n <= 1200 else ""
        print(f"{name:<50} {n:>8} {dmf_str:>12} {hours_str:>12}{marker}")

    # Summary recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    # Find best strategy near 1000
    best_strategy = None
    best_diff = float("inf")
    target = 1000

    all_strategies = strategies + custom_strategies
    for name, genes in all_strategies:
        diff = abs(len(genes) - target)
        if diff < best_diff:
            best_diff = diff
            best_strategy = (name, genes)

    if best_strategy:
        name, genes = best_strategy
        n = len(genes)
        est_dmf = estimate_triples_after_dmf(n)
        est_hours = estimate_inference_time_hours(est_dmf)

        print(f"\nClosest to 1000 genes: {name}")
        print(f"  Genes: {n}")
        print(f"  Estimated triples after DMF: {est_dmf:,}")
        print(f"  Estimated inference time: {est_hours:.1f} hours")

        # Compare to current
        current_n = len(full_list)
        current_dmf = estimate_triples_after_dmf(current_n)
        reduction = (1 - est_dmf / current_dmf) * 100
        speedup = current_dmf / est_dmf

        print(f"\n  vs Current ({current_n} genes, {current_dmf:,} triples):")
        print(f"    Triple reduction: {reduction:.1f}%")
        print(f"    Speedup factor: {speedup:.1f}x")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
