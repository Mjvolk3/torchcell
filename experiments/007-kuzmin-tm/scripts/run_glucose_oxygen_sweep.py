#!/usr/bin/env python3
"""
Main data generation script for glucose/oxygen sensitivity analysis.
Runs FBA sweep across all media types, glucose levels, and oxygen levels.
"""

import os
import sys
import subprocess

# Add torchcell to path
sys.path.append('/home/michaelvolk/Documents/projects/torchcell')

def main():
    """Run FBA analysis for all glucose/oxygen conditions."""

    # Configuration: Define all conditions to test
    CONDITIONS = {
        'media_types': ['minimal', 'YNB', 'YPD'],
        'glucose_levels': [2, 5, 10, 20],  # mmol/gDW/h
        'oxygen_levels': [1000, 20, 10, 5],  # mmol/gDW/h (1000 = unlimited)
    }

    BASE_DIR = "experiments/007-kuzmin-tm"
    SCRIPTS_DIR = f"{BASE_DIR}/scripts"
    RESULTS_DIR = f"{BASE_DIR}/results/cobra-fba-growth"

    total_conditions = (len(CONDITIONS['media_types']) *
                       len(CONDITIONS['glucose_levels']) *
                       len(CONDITIONS['oxygen_levels']))

    print("="*70)
    print("Glucose/O2 Sensitivity Analysis - Complete Sweep")
    print("="*70)
    print(f"Testing {total_conditions} conditions:")
    print(f"  - {len(CONDITIONS['media_types'])} media types: {', '.join(CONDITIONS['media_types'])}")
    print(f"  - {len(CONDITIONS['glucose_levels'])} glucose levels: {CONDITIONS['glucose_levels']}")
    print(f"  - {len(CONDITIONS['oxygen_levels'])} O2 levels: {CONDITIONS['oxygen_levels']}")
    print()
    print("Goal: Determine if discrete fitness bands are:")
    print("  1. CONSTRAINT-DRIVEN (bands shift with glucose/O2 changes)")
    print("  2. MODEL-INTRINSIC (bands persist across all conditions)")
    print()

    # Extract perturbations if needed
    perturbations_file = f"{RESULTS_DIR}/unique_perturbations.json"
    if not os.path.exists(perturbations_file):
        print("=== Extracting Perturbations ===")
        extract_script = f"{SCRIPTS_DIR}/extract_perturbations.py"
        result = subprocess.run([sys.executable, extract_script], capture_output=False)
        if result.returncode != 0:
            print("Error: Failed to extract perturbations")
            sys.exit(1)
    else:
        print("Using existing perturbations file")
    print()

    # Run FBA for each condition
    print("=== Running FBA Sweep ===")
    fba_script = f"{SCRIPTS_DIR}/glucose_oxygen_sensitivity_all_media.py"

    condition_count = 0
    for media in CONDITIONS['media_types']:
        for glucose in CONDITIONS['glucose_levels']:
            for oxygen in CONDITIONS['oxygen_levels']:
                condition_count += 1
                print("-"*70)
                print(f"[{condition_count}/{total_conditions}] Running FBA: {media}, glucose={glucose}, O2={oxygen}")

                result = subprocess.run(
                    [sys.executable, fba_script,
                     '--media', media,
                     '--glucose', str(glucose),
                     '--oxygen', str(oxygen)],
                    capture_output=False
                )

                if result.returncode != 0:
                    print(f"Error: FBA analysis failed for {media} glucose={glucose} O2={oxygen}")
                    sys.exit(1)

                print(f"Condition complete: {media} glucose={glucose} O2={oxygen}")
                print()

    print("="*70)
    print("FBA Sweep Complete!")
    print(f"Processed {total_conditions} conditions")
    print(f"Results directory: {RESULTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
