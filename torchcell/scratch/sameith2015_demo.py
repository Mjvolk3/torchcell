# torchcell/scratch/sameith2015_demo
# [[torchcell.scratch.sameith2015_demo]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/scratch/sameith2015_demo
"""Demonstrate both Sameith2015 datasets (single and double mutants).

Moved verbatim from torchcell/datasets/scerevisiae/sameith2015.py on 2026-10-06
(test campaign Phase 23) because it is a demo and not library code.

Run from the repo root:

    PYTHONPATH=$PWD python torchcell/scratch/sameith2015_demo.py

Needs ``DATA_ROOT`` in ``.env`` with the SGD genome and GO data under it; it
builds (downloading the raw files if absent) or loads the
``dm_microarray_sameith2015`` and ``sm_microarray_sameith2015`` datasets under
``DATA_ROOT/data/torchcell``. No GPU.
"""

import os
import os.path as osp
from typing import cast

from dotenv import load_dotenv

from torchcell.datasets.scerevisiae.sameith2015 import (
    DmMicroarraySameith2015Dataset,
    SmMicroarraySameith2015Dataset,
)
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome


def main() -> None:
    """Demonstrate both Sameith2015 datasets (single and double mutants)."""
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    # Initialize genome for gene name mapping
    genome = SCerevisiaeGenome(
        genome_root=osp.join(cast(str, DATA_ROOT), "data/sgd/genome"),
        go_root=osp.join(cast(str, DATA_ROOT), "data/go"),
        overwrite=False,
    )

    print("=" * 80)
    print("SAMEITH2015 MICROARRAY EXPRESSION DATASETS DEMO")
    print("=" * 80)

    # Double Mutant Dataset
    print("\n" + "=" * 80)
    print("1. DOUBLE MUTANT EXPRESSION DATASET (DmMicroarraySameith2015Dataset)")
    print("=" * 80)

    dm_dataset = DmMicroarraySameith2015Dataset(
        root=osp.join(cast(str, DATA_ROOT), "data/torchcell/dm_microarray_sameith2015"),
        genome=genome,
        io_workers=10,
        process_workers=0,  # Use sequential for demo
    )
    print("\nDataset loaded successfully")
    print(f"  Size: {len(dm_dataset)} double mutant genotypes")
    print(f"  Gene set size: {len(dm_dataset.gene_set)} unique genes")
    print(f"  First 10 genes: {list(dm_dataset.gene_set)[:10]}")

    if len(dm_dataset) > 0:
        data = dm_dataset[0]
        experiment = data["experiment"]
        reference = data["reference"]

        print("\n--- Example: First double mutant ---")
        perturbations = experiment["genotype"]["perturbations"]
        print(
            f"  Genotype: {perturbations[0]['systematic_gene_name']} × {perturbations[1]['systematic_gene_name']}"
        )
        print(f"  Strain: {reference['genome_reference']['strain']}")
        print(
            f"  Expression measurements: {len(experiment['phenotype']['expression'])} genes"
        )

        # Check if replicate statistics are available
        n_replicates = experiment["phenotype"].get("n_replicates")
        if n_replicates:
            genes_with_replicates = sum(1 for v in n_replicates.values() if v > 1)
            print(
                f"  Replicate statistics: {genes_with_replicates}/{len(n_replicates)} genes have n>1"
            )

    # Single Mutant Dataset
    print("\n" + "=" * 80)
    print("2. SINGLE MUTANT EXPRESSION DATASET (SmMicroarraySameith2015Dataset)")
    print("=" * 80)

    sm_dataset = SmMicroarraySameith2015Dataset(
        root=osp.join(cast(str, DATA_ROOT), "data/torchcell/sm_microarray_sameith2015"),
        genome=genome,
        io_workers=10,
        process_workers=0,  # Use sequential for demo
    )
    print("\nDataset loaded successfully")
    print(f"  Size: {len(sm_dataset)} single mutant genotypes")
    print(f"  Gene set size: {len(sm_dataset.gene_set)} unique genes")
    print(f"  First 10 genes: {list(sm_dataset.gene_set)[:10]}")

    if len(sm_dataset) > 0:
        data = sm_dataset[0]
        experiment = data["experiment"]
        reference = data["reference"]

        print("\n--- Example: First single mutant ---")
        perturbations = experiment["genotype"]["perturbations"]
        print(f"  Genotype: {perturbations[0]['systematic_gene_name']} deletion")
        print(f"  Strain: {reference['genome_reference']['strain']}")
        print(
            f"  Expression measurements: {len(experiment['phenotype']['expression'])} genes"
        )

        # Check if replicate statistics are available
        n_replicates = experiment["phenotype"].get("n_replicates")
        if n_replicates:
            genes_with_replicates = sum(1 for v in n_replicates.values() if v > 1)
            print(
                f"  Replicate statistics: {genes_with_replicates}/{len(n_replicates)} genes have n>1"
            )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"Double mutants: {len(dm_dataset)} genotypes (per-sample strain extraction working)"
    )
    print(f"Single mutants: {len(sm_dataset)} genotypes (BY4742 deletion library)")
    print(f"Total genotypes: {len(dm_dataset) + len(sm_dataset)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
