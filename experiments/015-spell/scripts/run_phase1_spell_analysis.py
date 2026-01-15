# experiments/015-spell/scripts/run_phase1_spell_analysis
# [[experiments.015-spell.scripts.run_phase1_spell_analysis]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/015-spell/scripts/run_phase1_spell_analysis

"""
Phase 1 Runner: SPELL Condition Coverage Analysis

This script executes the complete Phase 1 pipeline:
1. Load all SPELL studies (if not already loaded)
2. Extract enhanced condition metadata with structured parameters
3. Run coverage analysis and generate reports
4. Output prioritized Environment subclass recommendations

Usage:
    python experiments/015-spell/scripts/run_phase1_spell_analysis.py

    # Or with specific options:
    python experiments/015-spell/scripts/run_phase1_spell_analysis.py --max-studies 10  # For testing
    python experiments/015-spell/scripts/run_phase1_spell_analysis.py --skip-extraction  # If metadata already exists
"""

import sys
import os
import os.path as osp
import argparse
from dotenv import load_dotenv

from torchcell.datasets.scerevisiae.spell import (
    extract_and_load_all_spell_studies,
    export_condition_metadata
)
from spell_coverage_analysis import main as run_coverage_analysis

load_dotenv()

DATA_ROOT = os.getenv("DATA_ROOT", osp.expanduser(osp.join("~", "Documents", "projects", "torchcell")))


def main():
    parser = argparse.ArgumentParser(description='Run Phase 1 SPELL Coverage Analysis')
    parser.add_argument('--max-studies', type=int, default=None,
                        help='Maximum number of studies to load (for testing)')
    parser.add_argument('--skip-extraction', action='store_true',
                        help='Skip metadata extraction if CSV already exists')
    parser.add_argument('--spell-dir', type=str, default=None,
                        help='Path to SPELL data directory (default: DATA_ROOT/data/sgd/spell)')

    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 1: SPELL CONDITION COVERAGE ANALYSIS")
    print("=" * 70)
    print()

    # Determine SPELL directory
    spell_root_dir = args.spell_dir or osp.join(DATA_ROOT, "data/sgd/spell")

    if not osp.exists(spell_root_dir):
        print(f"ERROR: SPELL data directory not found: {spell_root_dir}")
        print("\nTo download SPELL data, run these commands:")
        print(f"  mkdir -p {spell_root_dir}")
        print(f"  cd {spell_root_dir}")
        print("  curl -O http://sgd-archive.yeastgenome.org/expression/microarray/all_spell_datasets.tar.gz")
        print("  tar -xzf all_spell_datasets.tar.gz")
        return 1

    # Check if enhanced metadata already exists
    enhanced_csv_path = osp.join(DATA_ROOT, "data/sgd/spell", "spell_conditions_metadata_enhanced.csv")

    if args.skip_extraction and osp.exists(enhanced_csv_path):
        print(f"✓ Skipping extraction - using existing metadata:")
        print(f"  {enhanced_csv_path}")
        print()
    else:
        # Step 1: Load SPELL data
        print("=" * 70)
        print("STEP 1: LOADING SPELL DATA")
        print("=" * 70)
        print()

        if args.max_studies:
            print(f"Loading first {args.max_studies} studies (test mode)...")
        else:
            print("Loading ALL SPELL studies (this will take several minutes)...")

        all_data = extract_and_load_all_spell_studies(
            spell_root_dir,
            max_studies=args.max_studies
        )

        # Step 2: Extract enhanced metadata
        print("\n" + "=" * 70)
        print("STEP 2: EXTRACTING ENHANCED METADATA")
        print("=" * 70)
        print()

        df_conditions = export_condition_metadata(all_data, output_path=enhanced_csv_path)

        print(f"\n✓ Enhanced metadata extraction complete!")
        print(f"  Output: {enhanced_csv_path}")
        print(f"  Total conditions: {len(df_conditions):,}")

    # Step 3: Run coverage analysis
    print("\n" + "=" * 70)
    print("STEP 3: RUNNING COVERAGE ANALYSIS")
    print("=" * 70)
    print()

    run_coverage_analysis()

    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review coverage report: data/sgd/spell/spell_coverage_report.md")
    print("  2. Examine visualizations in notes/assets/images/")
    print("  3. Proceed to Phase 2: Design Environment hierarchy based on priorities")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
