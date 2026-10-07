# torchcell/scratch/spell_demo
# [[torchcell.scratch.spell_demo]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/scratch/spell_demo
"""Load SPELL data and plot per-gene expression histograms.

Moved verbatim from torchcell/datasets/scerevisiae/spell.py on 2026-10-06 (test
campaign Phase 23) because it is a demo and not library code.

Run from the repo root:

    PYTHONPATH=$PWD python torchcell/scratch/spell_demo.py

Needs the SPELL PCL archive extracted under ``DATA_ROOT/data/sgd/spell``, where
``DATA_ROOT`` and ``ASSET_IMAGES_DIR`` are the module constants of
torchcell/datasets/scerevisiae/spell.py (not the ``.env`` variables); without the
data it prints the download commands and returns. Writes histogram images and a
condition-metadata export. No GPU.
"""

import os.path as osp

from torchcell.datasets.scerevisiae.spell import (
    ASSET_IMAGES_DIR,
    DATA_ROOT,
    check_condition_metadata_quality,
    export_condition_metadata,
    extract_and_load_all_spell_studies,
    plot_genes_across_all_studies,
)
from torchcell.timestamp import timestamp


def main() -> None:
    """Load SPELL data and plot per-gene expression histograms.

    Load SPELL expression data from multiple studies and create histograms
    for specific genes across all datasets and conditions.
    """
    spell_root_dir = osp.join(DATA_ROOT, "data/sgd/spell")

    # Check if data exists
    if not osp.exists(spell_root_dir):
        print(f"ERROR: SPELL data directory not found: {spell_root_dir}")
        print("\nTo download SPELL data, run these commands:")
        print(f"  mkdir -p {spell_root_dir}")
        print(f"  cd {spell_root_dir}")
        print(
            "  curl -O http://sgd-archive.yeastgenome.org/expression/microarray/all_spell_datasets.tar.gz"
        )
        print("  tar -xzf all_spell_datasets.tar.gz")
        return

    # Check if tar.gz exists but hasn't been extracted
    import glob

    tarfile = osp.join(spell_root_dir, "all_spell_datasets.tar.gz")
    if osp.exists(tarfile):
        zip_files = glob.glob(osp.join(spell_root_dir, "*.zip"))
        if not zip_files:
            print(f"Found {tarfile} but it hasn't been extracted yet.")
            print(f"Extracting to {spell_root_dir}...")
            import tarfile as tf

            with tf.open(tarfile, "r:gz") as tar:
                tar.extractall(spell_root_dir)
            print("✓ Extraction complete!")

    # Example 1: Load just a few studies for testing
    # studies_to_load = ['Gasch_2000_PMID_11102521', 'Gasch_2001_PMID_11598186']
    # all_data = extract_and_load_all_spell_studies(spell_root_dir, studies_to_load=studies_to_load)

    # Example 2: Load first N studies (faster for exploration)
    # all_data = extract_and_load_all_spell_studies(spell_root_dir, max_studies=10)

    # Example 3: Load ALL studies (comprehensive - will take several minutes!)
    print("=" * 70)
    print("Loading SPELL expression data from ALL studies...")
    print("This will take several minutes - loading ~600 studies...")
    print("=" * 70)
    all_data = extract_and_load_all_spell_studies(spell_root_dir)

    # Calculate total expression measurements
    total_measurements = 0
    total_conditions = 0
    total_genes_measured = set()

    for (study_name, dataset_name), (df, metadata) in all_data.items():
        n_genes = metadata["n_genes"]
        n_conds = metadata["n_conditions"]
        total_measurements += n_genes * n_conds
        total_conditions += n_conds
        total_genes_measured.update(df.index.tolist())

    print(f"\n{'=' * 70}")
    print("SPELL DATABASE SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Studies loaded:           {len(set(k[0] for k in all_data.keys())):,}")
    print(f"  Datasets loaded:          {len(all_data):,}")
    print(f"  Total conditions:         {total_conditions:,}")
    print(f"  Unique genes measured:    {len(total_genes_measured):,}")
    print("")
    print(f"  TOTAL EXPRESSION VALUES:  {total_measurements:,}")
    print("  (genes × conditions across all datasets)")
    print(f"{'=' * 70}\n")

    # Export condition metadata to CSV for analysis
    export_condition_metadata(all_data)

    # Check condition metadata quality
    check_condition_metadata_quality(all_data)

    # Plot histograms for 3 genes across ALL loaded studies and datasets
    genes_of_interest = ["YDL025C", "YJL166W", "YMR027W"]

    save_path = osp.join(ASSET_IMAGES_DIR, f"spell_genes_all_studies_{timestamp()}.png")
    plot_genes_across_all_studies(all_data, genes_of_interest, save_path=save_path)

    print(f"\n{'=' * 70}")
    print(f"Images saved to: {ASSET_IMAGES_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
