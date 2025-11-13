#!/usr/bin/env python3
"""
Compare profiling results from DANGO and Lazy Hetero models.
Generates a side-by-side comparison showing where computational overhead occurs.

Usage:
    python experiments/006-kuzmin-tmi/scripts/compare_full_step_profiles.py \
        --dango_profile path/to/profile_dango_full_step_results_*.txt \
        --lazy_profile path/to/profile_gene_interaction_dango_full_step_results_*.txt \
        --output_dir experiments/006-kuzmin-tmi/profiling_results/full_step_TIMESTAMP
"""

import argparse
import os
import os.path as osp
import re
from typing import Dict, Tuple

from torchcell.timestamp import timestamp


def parse_profile_file(filepath: str) -> Dict[str, Dict[str, float]]:
    """Parse a profile .txt file and extract timing information"""

    results = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the PHASE BREAKDOWN section
    in_phase_section = False
    for i, line in enumerate(lines):
        if "PHASE BREAKDOWN" in line:
            in_phase_section = True
            # Skip the separator lines and header
            continue

        if in_phase_section and line.strip().startswith(('-', '=')):
            continue

        if in_phase_section and "TOTAL" in line:
            # Parse total line
            parts = line.split()
            if len(parts) >= 4:
                try:
                    results["TOTAL"] = {
                        "cpu": float(parts[1]),
                        "cuda": float(parts[2]),
                        "total": float(parts[3]),
                        "percent": 100.0
                    }
                except (ValueError, IndexError):
                    pass
            break

        if in_phase_section and any(phase in line for phase in ["Forward Pass", "Loss Computation", "Backward Pass", "Optimizer Step"]):
            # Parse phase line
            # Format: "1. Forward Pass                  15.234       12.456       27.690      40.5%"
            parts = line.split()
            if len(parts) >= 5:
                try:
                    # Extract phase name (combine first parts before numbers)
                    phase_name = ' '.join(parts[:-4])
                    cpu_time = float(parts[-4])
                    cuda_time = float(parts[-3])
                    total_time = float(parts[-2])
                    percent = float(parts[-1].rstrip('%'))

                    results[phase_name] = {
                        "cpu": cpu_time,
                        "cuda": cuda_time,
                        "total": total_time,
                        "percent": percent
                    }
                except (ValueError, IndexError):
                    pass

    return results


def generate_comparison(dango_results: Dict, lazy_results: Dict, output_file: str):
    """Generate a side-by-side comparison of the two profiles"""

    phase_order = [
        "1. Forward Pass",
        "2. Loss Computation",
        "3. Backward Pass",
        "4. Optimizer Step"
    ]

    # Calculate ratios
    comparisons = []
    for phase in phase_order:
        if phase in dango_results and phase in lazy_results:
            dango_time = dango_results[phase]["total"]
            lazy_time = lazy_results[phase]["total"]
            ratio = lazy_time / dango_time if dango_time > 0 else 0
            diff = lazy_time - dango_time

            comparisons.append({
                "phase": phase,
                "dango": dango_time,
                "lazy": lazy_time,
                "ratio": ratio,
                "diff": diff
            })

    # Total comparison
    if "TOTAL" in dango_results and "TOTAL" in lazy_results:
        total_comp = {
            "phase": "TOTAL",
            "dango": dango_results["TOTAL"]["total"],
            "lazy": lazy_results["TOTAL"]["total"],
            "ratio": lazy_results["TOTAL"]["total"] / dango_results["TOTAL"]["total"],
            "diff": lazy_results["TOTAL"]["total"] - dango_results["TOTAL"]["total"]
        }
    else:
        total_comp = None

    # Write comparison report
    with open(output_file, 'w') as f:
        f.write("DANGO vs LAZY HETERO: COMPLETE TRAINING STEP COMPARISON\n")
        f.write("="*100 + "\n\n")

        f.write("TIMING BREAKDOWN\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Phase':<25} {'DANGO (ms)':>12} {'Lazy Hetero (ms)':>18} {'Ratio':>10} {'Diff (ms)':>12}\n")
        f.write("-"*100 + "\n")

        for comp in comparisons:
            f.write(f"{comp['phase']:<25} {comp['dango']:>12.3f} {comp['lazy']:>18.3f} "
                   f"{comp['ratio']:>9.1f}x {comp['diff']:>12.3f}\n")

        f.write("-"*100 + "\n")

        if total_comp:
            f.write(f"{total_comp['phase']:<25} {total_comp['dango']:>12.3f} {total_comp['lazy']:>18.3f} "
                   f"{total_comp['ratio']:>9.1f}x {total_comp['diff']:>12.3f}\n")

        f.write("\n")

        # Percentage breakdown comparison
        f.write("PERCENTAGE BREAKDOWN COMPARISON\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Phase':<25} {'DANGO %':>12} {'Lazy Hetero %':>18} {'Delta':>10}\n")
        f.write("-"*100 + "\n")

        for phase in phase_order:
            if phase in dango_results and phase in lazy_results:
                dango_pct = dango_results[phase]["percent"]
                lazy_pct = lazy_results[phase]["percent"]
                delta = lazy_pct - dango_pct
                f.write(f"{phase:<25} {dango_pct:>11.1f}% {lazy_pct:>17.1f}% {delta:>9.1f}%\n")

        f.write("\n")

        # Bottleneck analysis
        f.write("BOTTLENECK IDENTIFICATION\n")
        f.write("-"*100 + "\n")

        # Find the phases with biggest slowdowns
        sorted_comps = sorted(comparisons, key=lambda x: x["ratio"], reverse=True)

        for comp in sorted_comps[:3]:  # Top 3 bottlenecks
            if comp["ratio"] > 2.0:  # Only show significant slowdowns
                f.write(f"✗ {comp['phase'].upper()}: {comp['ratio']:.1f}x slower ({comp['lazy']:.1f}ms vs {comp['dango']:.1f}ms)\n")

                if "Backward" in comp["phase"]:
                    f.write(f"  → Gradient computation through multiple graph copies\n")
                    f.write(f"  → Each convolution layer processes expanded nodes\n")
                elif "Optimizer" in comp["phase"]:
                    f.write(f"  → Gradient accumulation from expanded embeddings\n")
                    f.write(f"  → Parameter updates with larger gradient tensors\n")
                elif "Forward" in comp["phase"]:
                    f.write(f"  → Processing expanded node embeddings through convolutions\n")

                f.write("\n")

        # Overall conclusion
        f.write("OVERALL ANALYSIS\n")
        f.write("-"*100 + "\n")

        if total_comp:
            f.write(f"Total training step is {total_comp['ratio']:.1f}x slower in Lazy Hetero ({total_comp['lazy']:.1f}ms vs {total_comp['dango']:.1f}ms)\n\n")

        # Find dominant phase in Lazy Hetero
        lazy_max_phase = max(phase_order, key=lambda p: lazy_results.get(p, {}).get("percent", 0))
        lazy_max_pct = lazy_results.get(lazy_max_phase, {}).get("percent", 0)

        f.write(f"Lazy Hetero bottleneck: {lazy_max_phase} ({lazy_max_pct:.1f}% of total time)\n")

        # Identify if backward is the problem
        backward_comp = next((c for c in comparisons if "Backward" in c["phase"]), None)
        if backward_comp and backward_comp["ratio"] > 5.0:
            f.write(f"\n⚠️  CRITICAL: Backward pass is {backward_comp['ratio']:.1f}x slower!\n")
            f.write("   This confirms that gradient computation through the expanded computational graph\n")
            f.write("   is the primary bottleneck. The expand() operation creates multiple parallel\n")
            f.write("   computation paths, each requiring separate gradient computation.\n")

        f.write("\nRECOMMENDATION\n")
        f.write("-"*100 + "\n")
        f.write("Adopt DANGO's architecture:\n")
        f.write("1. Process full graph embeddings ONCE per batch\n")
        f.write("2. Use perturbation masks as indexing operations AFTER message passing\n")
        f.write("3. Avoid expand() operation that creates multiple gradient computation paths\n")
        f.write(f"\nExpected speedup: {total_comp['ratio']:.1f}x (bringing lazy hetero to DANGO performance)\n")

    print(f"\nComparison saved to: {output_file}")

    # Also print to console
    print("\n" + "="*100)
    print("COMPARISON SUMMARY")
    print("="*100)
    print(f"{'Phase':<25} {'DANGO (ms)':>12} {'Lazy Hetero (ms)':>18} {'Ratio':>10}")
    print("-"*100)
    for comp in comparisons:
        print(f"{comp['phase']:<25} {comp['dango']:>12.3f} {comp['lazy']:>18.3f} {comp['ratio']:>9.1f}x")
    if total_comp:
        print("-"*100)
        print(f"{total_comp['phase']:<25} {total_comp['dango']:>12.3f} {total_comp['lazy']:>18.3f} {total_comp['ratio']:>9.1f}x")


def main():
    parser = argparse.ArgumentParser(description="Compare DANGO and Lazy Hetero profiles")
    parser.add_argument("--dango_profile", type=str, required=True,
                       help="Path to DANGO profile results file")
    parser.add_argument("--lazy_profile", type=str, required=True,
                       help="Path to Lazy Hetero profile results file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save comparison results")
    args = parser.parse_args()

    print("Parsing DANGO profile...")
    dango_results = parse_profile_file(args.dango_profile)

    print("Parsing Lazy Hetero profile...")
    lazy_results = parse_profile_file(args.lazy_profile)

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate comparison
    timestamp_str = timestamp()
    output_file = osp.join(args.output_dir, f"comparison_full_step_{timestamp_str}.txt")

    print("\nGenerating comparison...")
    generate_comparison(dango_results, lazy_results, output_file)

    print("\nComparison complete!")


if __name__ == "__main__":
    main()
