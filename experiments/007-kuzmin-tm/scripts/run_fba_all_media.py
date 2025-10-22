#!/usr/bin/env python3
"""
Main FBA data generation script for all media conditions.
Runs targeted FBA with minimal, YNB, and YPD media.
"""

import os
import sys
import json
import subprocess

# Add torchcell to path
sys.path.append('/home/michaelvolk/Documents/projects/torchcell')

def main():
    """Run FBA analysis for all media conditions."""

    BASE_DIR = "experiments/007-kuzmin-tm"
    SCRIPTS_DIR = f"{BASE_DIR}/scripts"
    RESULTS_DIR = f"{BASE_DIR}/results/cobra-fba-growth"
    PERTURBATIONS_FILE = f"{RESULTS_DIR}/unique_perturbations.json"

    print("="*70)
    print("COBRA FBA Analysis with Multiple Media Conditions")
    print("="*70)
    print()

    # Step 1: Verify media setup
    print("=== Step 1: Verifying Media Setup ===")
    verify_script = f"{SCRIPTS_DIR}/verify_media_differences.py"
    if os.path.exists(verify_script):
        result = subprocess.run([sys.executable, verify_script], capture_output=False)
        if result.returncode != 0:
            print("Warning: Media verification encountered issues")
    print()

    # Step 2: Extract perturbations if needed
    print("=== Step 2: Preparing Perturbations Dataset ===")
    if not os.path.exists(PERTURBATIONS_FILE):
        print("Extracting perturbations from Neo4j...")
        extract_script = f"{SCRIPTS_DIR}/extract_perturbations.py"
        result = subprocess.run([sys.executable, extract_script], capture_output=False)
        if result.returncode != 0:
            print("Error: Failed to extract perturbations")
            sys.exit(1)
    else:
        print("Using existing perturbations file")
    print()

    # Step 3: Run FBA for each media condition
    print("=== Step 3: Running FBA Analysis for All Media Conditions ===")
    print("Configuration:")
    print("  - Solver timeout: 60 seconds")
    print("  - Method: YeastGEM with proper media setup")
    print()

    media_conditions = ['minimal', 'YNB', 'YPD']
    fba_script = f"{SCRIPTS_DIR}/targeted_fba_with_media_yeastgem.py"

    for media in media_conditions:
        print("-"*70)
        print(f"Running FBA with {media} media")

        result = subprocess.run(
            [sys.executable, fba_script, '--media', media],
            capture_output=False
        )

        if result.returncode != 0:
            print(f"Error: FBA analysis failed for {media} media")
            sys.exit(1)

        print(f"{media} media FBA complete")
        print()

    print("="*70)
    print("FBA Data Generation Complete!")
    print(f"Results directory: {RESULTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
