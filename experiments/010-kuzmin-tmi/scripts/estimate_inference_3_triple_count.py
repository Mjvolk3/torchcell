# experiments/010-kuzmin-tmi/scripts/estimate_inference_3_triple_count.py
# [[experiments.010-kuzmin-tmi.scripts.estimate_inference_3_triple_count]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/estimate_inference_3_triple_count.py

"""
Estimate the number of triples for inference_dataset_3.

This script analyzes the SMF and DMF datasets to estimate how many
triples would pass the new relaxed thresholding scheme:

INFERENCE 3 THRESHOLDS:
  max(smf) > 1.04   (at least one gene shows improvement)
  all(smf) > 0.80   (all genes are viable)
  max(dmf) > 1.08   (iterative improvement over max single)
  all(dmf) > 0.80   (all pairs are viable)

Compared to INFERENCE 2:
  max(smf) > 1.10
  all(smf) > 1.00
  max(dmf) > max(smf) + 0.03
  all(dmf) > 1.00
"""

import os
import os.path as osp
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from torchcell.datasets.scerevisiae import (
    SmfCostanzo2016Dataset,
    DmfCostanzo2016Dataset,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]


def print_section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_smf_data() -> dict[str, float]:
    """Load single mutant fitness data from Costanzo2016."""
    print("\nLoading SmfCostanzo2016Dataset...")
    dataset = SmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/smf_costanzo2016"),
        io_workers=4,
    )
    print(f"  Dataset size: {len(dataset)}")

    smf_data = {}
    for i in tqdm(range(len(dataset)), desc="  Indexing SMF"):
        item = dataset[i]
        gene = item["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
        fitness = item["experiment"]["phenotype"]["fitness"]
        # Keep max fitness if gene appears multiple times
        if gene not in smf_data or fitness > smf_data[gene]:
            smf_data[gene] = fitness

    print(f"  Unique genes: {len(smf_data)}")
    return smf_data


def load_dmf_data() -> dict[frozenset, float]:
    """Load double mutant fitness data from Costanzo2016."""
    print("\nLoading DmfCostanzo2016Dataset...")
    dataset = DmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/dmf_costanzo2016"),
        io_workers=4,
        batch_size=int(1e4),
    )
    print(f"  Dataset size: {len(dataset)}")

    dmf_data = {}
    for i in tqdm(range(len(dataset)), desc="  Indexing DMF"):
        item = dataset[i]
        genes = frozenset(
            p["systematic_gene_name"]
            for p in item["experiment"]["genotype"]["perturbations"]
        )
        fitness = item["experiment"]["phenotype"]["fitness"]
        # Keep max fitness if pair appears multiple times
        if genes not in dmf_data or fitness > dmf_data[genes]:
            dmf_data[genes] = fitness

    print(f"  Unique pairs: {len(dmf_data)}")
    return dmf_data


def analyze_thresholds(smf_data: dict[str, float], dmf_data: dict[frozenset, float]):
    """Analyze gene/pair counts at various threshold levels."""
    print_section("SMF THRESHOLD ANALYSIS")

    smf_values = list(smf_data.values())
    print(f"\n  Total genes: {len(smf_values)}")
    print(f"  Min fitness: {min(smf_values):.4f}")
    print(f"  Max fitness: {max(smf_values):.4f}")
    print(f"  Mean fitness: {sum(smf_values)/len(smf_values):.4f}")

    # Count genes at various thresholds
    print("\n  Genes exceeding threshold:")
    for thresh in [0.8, 0.9, 1.0, 1.02, 1.04, 1.05, 1.06, 1.08, 1.10]:
        count = sum(1 for v in smf_values if v > thresh)
        pct = count / len(smf_values) * 100
        print(f"    SMF > {thresh:.2f}: {count:>5} genes ({pct:.2f}%)")

    print_section("DMF THRESHOLD ANALYSIS")

    dmf_values = list(dmf_data.values())
    print(f"\n  Total pairs: {len(dmf_values):,}")
    print(f"  Min fitness: {min(dmf_values):.4f}")
    print(f"  Max fitness: {max(dmf_values):.4f}")
    print(f"  Mean fitness: {sum(dmf_values)/len(dmf_values):.4f}")

    # Count pairs at various thresholds
    print("\n  Pairs exceeding threshold:")
    for thresh in [0.8, 0.9, 1.0, 1.04, 1.08, 1.10, 1.12, 1.15]:
        count = sum(1 for v in dmf_values if v > thresh)
        pct = count / len(dmf_values) * 100
        print(f"    DMF > {thresh:.2f}: {count:>10,} pairs ({pct:.3f}%)")


def estimate_triples_inference_2(
    smf_data: dict[str, float],
    dmf_data: dict[frozenset, float],
) -> int:
    """
    Estimate triple count using INFERENCE 2 thresholds.

    Thresholds:
      max(smf) > 1.10
      all(smf) > 1.00
      max(dmf) > max(smf) + 0.03
      all(dmf) > 1.00
    """
    SMF_THRESHOLD = 1.10
    SMF_BASELINE = 1.00
    DMF_BASELINE = 1.00
    DMF_GAP = 0.03

    # Build adjacency from DMF pairs where both genes pass baseline
    adjacency = defaultdict(set)
    for pair, fitness in dmf_data.items():
        if fitness <= DMF_BASELINE:
            continue
        genes = list(pair)
        if len(genes) != 2:
            continue
        g1, g2 = genes
        if smf_data.get(g1, 0) > SMF_BASELINE and smf_data.get(g2, 0) > SMF_BASELINE:
            adjacency[g1].add(g2)
            adjacency[g2].add(g1)

    print(f"\n  Adjacency graph: {len(adjacency)} nodes")

    # Count triples
    valid_triples = 0
    genes = sorted(adjacency.keys())

    for i, g1 in enumerate(tqdm(genes, desc="  Counting triples")):
        neighbors_g1 = adjacency[g1]
        for g2 in neighbors_g1:
            if g2 <= g1:
                continue
            neighbors_g2 = adjacency[g2]
            common = neighbors_g1 & neighbors_g2
            for g3 in common:
                if g3 <= g2:
                    continue

                # Check SMF thresholds
                smfs = [smf_data.get(g, 0) for g in [g1, g2, g3]]
                if not all(s > SMF_BASELINE for s in smfs):
                    continue
                if max(smfs) <= SMF_THRESHOLD:
                    continue

                # Check DMF thresholds
                pairs = [
                    frozenset([g1, g2]),
                    frozenset([g1, g3]),
                    frozenset([g2, g3]),
                ]
                dmfs = [dmf_data.get(p, 0) for p in pairs]
                if not all(d > DMF_BASELINE for d in dmfs):
                    continue
                if max(dmfs) <= max(smfs) + DMF_GAP:
                    continue

                valid_triples += 1

    return valid_triples


def estimate_triples_inference_3(
    smf_data: dict[str, float],
    dmf_data: dict[frozenset, float],
) -> int:
    """
    Estimate triple count using INFERENCE 3 thresholds.

    Thresholds:
      max(smf) > 1.04
      all(smf) > 0.80
      max(dmf) > 1.08
      all(dmf) > 0.80
    """
    SMF_THRESHOLD = 1.04
    SMF_BASELINE = 0.80
    DMF_THRESHOLD = 1.08
    DMF_BASELINE = 0.80

    # Build adjacency from DMF pairs where both genes pass baseline
    adjacency = defaultdict(set)
    for pair, fitness in dmf_data.items():
        if fitness <= DMF_BASELINE:
            continue
        genes = list(pair)
        if len(genes) != 2:
            continue
        g1, g2 = genes
        if smf_data.get(g1, 0) > SMF_BASELINE and smf_data.get(g2, 0) > SMF_BASELINE:
            adjacency[g1].add(g2)
            adjacency[g2].add(g1)

    print(f"\n  Adjacency graph: {len(adjacency)} nodes")

    # Count triples
    valid_triples = 0
    genes = sorted(adjacency.keys())

    for i, g1 in enumerate(tqdm(genes, desc="  Counting triples")):
        neighbors_g1 = adjacency[g1]
        for g2 in neighbors_g1:
            if g2 <= g1:
                continue
            neighbors_g2 = adjacency[g2]
            common = neighbors_g1 & neighbors_g2
            for g3 in common:
                if g3 <= g2:
                    continue

                # Check SMF thresholds
                smfs = [smf_data.get(g, 0) for g in [g1, g2, g3]]
                if not all(s > SMF_BASELINE for s in smfs):
                    continue
                if max(smfs) <= SMF_THRESHOLD:
                    continue

                # Check DMF thresholds
                pairs = [
                    frozenset([g1, g2]),
                    frozenset([g1, g3]),
                    frozenset([g2, g3]),
                ]
                dmfs = [dmf_data.get(p, 0) for p in pairs]
                if not all(d > DMF_BASELINE for d in dmfs):
                    continue
                if max(dmfs) <= DMF_THRESHOLD:
                    continue

                valid_triples += 1

    return valid_triples


def main():
    print_section("INFERENCE DATASET 3: TRIPLE COUNT ESTIMATION")

    # Load data
    smf_data = load_smf_data()
    dmf_data = load_dmf_data()

    # Analyze thresholds
    analyze_thresholds(smf_data, dmf_data)

    # Estimate triple counts
    print_section("TRIPLE COUNT ESTIMATION: INFERENCE 2")
    print("\n  Thresholds:")
    print("    max(smf) > 1.10")
    print("    all(smf) > 1.00")
    print("    max(dmf) > max(smf) + 0.03")
    print("    all(dmf) > 1.00")

    count_inf2 = estimate_triples_inference_2(smf_data, dmf_data)
    print(f"\n  RESULT: {count_inf2:,} triples")

    print_section("TRIPLE COUNT ESTIMATION: INFERENCE 3")
    print("\n  Thresholds:")
    print("    max(smf) > 1.04")
    print("    all(smf) > 0.80")
    print("    max(dmf) > 1.08")
    print("    all(dmf) > 0.80")

    count_inf3 = estimate_triples_inference_3(smf_data, dmf_data)
    print(f"\n  RESULT: {count_inf3:,} triples")

    # Summary
    print_section("SUMMARY")
    print(f"""
  ┌─────────────────────┬────────────────────┬────────────────────┐
  │                     │   INFERENCE 2      │   INFERENCE 3      │
  ├─────────────────────┼────────────────────┼────────────────────┤
  │ max(smf) threshold  │   > 1.10           │   > 1.04           │
  │ all(smf) baseline   │   > 1.00           │   > 0.80           │
  │ max(dmf) threshold  │   > max(smf)+0.03  │   > 1.08           │
  │ all(dmf) baseline   │   > 1.00           │   > 0.80           │
  ├─────────────────────┼────────────────────┼────────────────────┤
  │ Triple count        │   {count_inf2:>14,} │   {count_inf3:>14,} │
  │ Ratio (inf3/inf2)   │                    │   {count_inf3/max(count_inf2,1):>14.1f}x │
  └─────────────────────┴────────────────────┴────────────────────┘
    """)


if __name__ == "__main__":
    main()
