# experiments/010-kuzmin-tmi/scripts/generate_triple_combinations_inference_2.py
# [[experiments.010-kuzmin-tmi.scripts.generate_triple_combinations_inference_2]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/generate_triple_combinations_inference_2.py
"""
Generate triple gene combinations for inference_2 with iterative fitness improvement filtering.

Key differences from inference_1:
- Singles filter: all > 1.0, max > 1.10 (SMF threshold)
- Doubles filter: all > 1.0, max > max(singles) + 0.03 (DMF must beat best SMF)
- Uses SmfCostanzo2016Dataset (lowest noise σ=0.063)
- Uses dataset objects properly via dataset.df
- Streaming Parquet output for memory efficiency

Scientific claim: f_WT < f_i < f_ij < f_ijk (iterative fitness improvement)
"""

import argparse
import os
import os.path as osp
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from torchcell.timestamp import timestamp
from torchcell.datasets.scerevisiae.costanzo2016 import (
    SmfCostanzo2016Dataset,
    DmfCostanzo2016Dataset,
)
from torchcell.datasets.scerevisiae.kuzmin2018 import DmfKuzmin2018Dataset
from torchcell.datasets.scerevisiae.kuzmin2020 import DmfKuzmin2020Dataset
from torchcell.datasets.scerevisiae.sgd import GeneEssentialitySgdDataset
from torchcell.graph import SCerevisiaeGraph
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Import for TMI filtering
from torchcell.data import Neo4jCellDataset, MeanExperimentDeduplicator, GenotypeAggregator
from torchcell.data.graph_processor import SubgraphRepresentation
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

# Load environment
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR", "notes/assets/images")

# Thresholds for iterative fitness improvement
SMF_THRESHOLD = 1.10  # At least one single must exceed this (max > 1.10)
SMF_BASELINE = 1.0    # All singles must exceed this (all > 1.0)
DMF_GAP = 0.03        # max(doubles) must beat max(singles) by this gap
DMF_BASELINE = 1.0    # All doubles must exceed this (all > 1.0)
PARQUET_BATCH_SIZE = 100_000  # Write batches of 100K triples


def load_essential_genes(data_root: str) -> set[str]:
    """
    Load essential genes from GeneEssentialitySgdDataset.
    These genes are lethal when deleted and should be excluded.

    Returns set of systematic gene names that are essential.
    """
    print("\nLoading essential genes from SGD...")

    # Initialize genome and graph for the dataset
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(data_root, "data/sgd/genome"),
        string_root=osp.join(data_root, "data/string"),
        tflink_root=osp.join(data_root, "data/tflink"),
        genome=genome,
    )

    dataset = GeneEssentialitySgdDataset(scerevisiae_graph=graph)
    print(f"  GeneEssentialitySgdDataset size: {len(dataset)}")

    essential_genes = set()
    for i in tqdm(range(len(dataset)), desc="  Extracting essential genes"):
        item = dataset[i]
        if item['experiment']['phenotype']['is_essential']:
            gene = item['experiment']['genotype']['perturbations'][0]['systematic_gene_name']
            essential_genes.add(gene)

    print(f"  Found {len(essential_genes)} essential genes to exclude")
    return essential_genes


class StreamingParquetWriter:
    """Helper class to write triples to Parquet in batches."""

    def __init__(self, filename):
        self.filename = filename
        self.batch = []
        self.schema = pa.schema([
            ('gene1', pa.string()),
            ('gene2', pa.string()),
            ('gene3', pa.string())
        ])
        self.writer = pq.ParquetWriter(
            filename,
            self.schema,
            compression='snappy',
            use_dictionary=True,
        )

    def write(self, triple_string):
        """Write a triple (as 'g1,g2,g3' string) to the batch."""
        gene1, gene2, gene3 = triple_string.strip().split(',')
        self.batch.append((gene1, gene2, gene3))

        if len(self.batch) >= PARQUET_BATCH_SIZE:
            self._flush_batch()

    def _flush_batch(self):
        """Write accumulated batch to Parquet file."""
        if not self.batch:
            return

        table = pa.table({
            'gene1': [t[0] for t in self.batch],
            'gene2': [t[1] for t in self.batch],
            'gene3': [t[2] for t in self.batch]
        })

        self.writer.write_table(table)
        self.batch = []

    def close(self):
        """Flush remaining batch and close writer."""
        self._flush_batch()
        self.writer.close()


def load_smf_from_dataset(dataset):
    """
    Load single mutant fitness from SmfCostanzo2016Dataset.
    Uses dataset.df which is the preprocessed dataframe.

    Returns dict mapping gene name to fitness value.
    """
    print(f"\nLoading SMF from {dataset.name}...")
    print(f"  Dataset size: {len(dataset)}")

    df = dataset.df
    print(f"  DataFrame shape: {df.shape}")

    gene_col = "Systematic gene name"
    fitness_col = "Single mutant fitness"

    # Filter to deletions only (KanMX or NatMX) for consistency
    deletion_mask = df["perturbation_type"].isin(["KanMX_deletion", "NatMX_deletion"])
    df_deletions = df[deletion_mask].copy()
    print(f"  Filtered to deletions: {len(df_deletions)} rows")

    # Build gene -> max fitness mapping
    gene_fitness = {}

    for _, row in tqdm(df_deletions.iterrows(), total=len(df_deletions), desc="  Building SMF map"):
        gene = row[gene_col]
        fitness = row[fitness_col]

        if pd.notna(gene) and pd.notna(fitness):
            # Track max fitness for each gene
            if gene not in gene_fitness or fitness > gene_fitness[gene]:
                gene_fitness[gene] = fitness

    print(f"  Unique genes with SMF: {len(gene_fitness)}")

    # Report statistics
    fitness_values = list(gene_fitness.values())
    print(f"  SMF range: [{min(fitness_values):.3f}, {max(fitness_values):.3f}]")
    print(f"  SMF > {SMF_THRESHOLD}: {sum(1 for f in fitness_values if f > SMF_THRESHOLD)}")

    return gene_fitness


def load_dmf_from_dataset(dataset):
    """
    Load double mutant fitness from DMF dataset.
    Uses dataset.df which is the preprocessed dataframe.

    Returns dict mapping (gene1, gene2) tuple to fitness value.
    """
    print(f"\nLoading DMF from {dataset.name}...")
    print(f"  Dataset size: {len(dataset)}")

    df = dataset.df
    print(f"  DataFrame shape: {df.shape}")

    # Determine column names based on dataset
    if hasattr(dataset, 'name'):
        if 'costanzo2016' in dataset.name.lower():
            gene1_col = "Query Systematic Name"
            gene2_col = "Array Systematic Name"
            fitness_col = "Double mutant fitness"
        elif 'kuzmin2018' in dataset.name.lower():
            gene1_col = "Query systematic name no ho"
            gene2_col = "Array systematic name"
            fitness_col = "Combined mutant fitness"
        elif 'kuzmin2020' in dataset.name.lower():
            gene1_col = "Query systematic name no ho"
            gene2_col = "Array systematic name"
            fitness_col = "fitness"
        else:
            raise ValueError(f"Unknown dataset: {dataset.name}")

    pair_fitness = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  Building DMF map"):
        gene1, gene2 = row[gene1_col], row[gene2_col]
        fitness = row[fitness_col]

        if pd.isna(gene1) or pd.isna(gene2) or pd.isna(fitness):
            continue

        # Store as sorted tuple for consistency
        pair = tuple(sorted([gene1, gene2]))

        # Track max fitness
        if pair not in pair_fitness or fitness > pair_fitness[pair]:
            pair_fitness[pair] = fitness

    print(f"  Unique pairs with DMF: {len(pair_fitness)}")

    return pair_fitness


def build_adjacency_graph(smf_fitness, dmf_fitness):
    """
    Build adjacency graph where edge (i,j) exists if:
    - f_ij > 1.0 (double not deleterious)
    - f_i > 1.0 AND f_j > 1.0 (both singles not deleterious)

    Returns:
        adjacency: dict mapping gene to set of valid neighbors
        valid_pairs: set of valid (gene1, gene2) tuples
    """
    print("\n" + "="*80)
    print(f"Building adjacency graph (DMF > {DMF_BASELINE}, both SMF > {SMF_BASELINE})")
    print("="*80)

    valid_pairs = set()
    adjacency = defaultdict(set)

    for (g1, g2), dmf in tqdm(dmf_fitness.items(), desc="Filtering pairs"):
        # Check DMF threshold
        if dmf <= DMF_BASELINE:
            continue

        # Check both singles exist and pass threshold
        smf1 = smf_fitness.get(g1)
        smf2 = smf_fitness.get(g2)

        if smf1 is None or smf2 is None:
            continue  # Missing SMF data

        if smf1 <= SMF_BASELINE or smf2 <= SMF_BASELINE:
            continue  # At least one single is deleterious

        # This pair passes the filter
        valid_pairs.add((g1, g2))
        adjacency[g1].add(g2)
        adjacency[g2].add(g1)

    print(f"\nAdjacency graph statistics:")
    print(f"  Valid pairs: {len(valid_pairs):,}")
    print(f"  Genes with valid pairs: {len(adjacency)}")

    if adjacency:
        conn_values = [len(neighbors) for neighbors in adjacency.values()]
        print(f"  Avg neighbors per gene: {np.mean(conn_values):.1f}")
        print(f"  Median neighbors: {np.median(conn_values):.1f}")
        print(f"  Min neighbors: {min(conn_values)}")
        print(f"  Max neighbors: {max(conn_values)}")

    return adjacency, valid_pairs


def generate_triples_streaming(
    adjacency, smf_fitness, dmf_fitness, is_any_perturbed_gene_index, output_files
):
    """
    Generate triples using adjacency graph with streaming output.

    Applies filtering criteria:
    - All 3 singles > 1.0, max > 1.10
    - All 3 doubles > 1.0, max > max(singles) + 0.03
    """
    print("\n" + "="*80)
    print("Generating triples with iterative fitness improvement filtering...")
    print(f"  Singles: all > {SMF_BASELINE}, max > {SMF_THRESHOLD}")
    print(f"  Doubles: all > {DMF_BASELINE}, max > max(singles) + {DMF_GAP}")
    print("="*80)

    kept_count = 0
    tmi_removed_count = 0
    total_generated = 0

    # Track rejection reasons
    failed_counts = {
        "missing_smf": 0,
        "missing_dmf": 0,
        "singles_threshold": 0,
        "doubles_threshold": 0,
    }

    genes_sorted = sorted(adjacency.keys())

    for g1 in tqdm(genes_sorted, desc="Processing genes"):
        neighbors_g1 = adjacency[g1]
        neighbors_g1_sorted = sorted(neighbors_g1)

        for g2 in neighbors_g1_sorted:
            if g2 <= g1:
                continue

            # Find mutual neighbors
            mutual_neighbors = neighbors_g1 & adjacency[g2]

            for g3 in mutual_neighbors:
                if g3 <= g2:
                    continue

                total_generated += 1

                # Get all SMF values
                smf1 = smf_fitness.get(g1)
                smf2 = smf_fitness.get(g2)
                smf3 = smf_fitness.get(g3)

                if smf1 is None or smf2 is None or smf3 is None:
                    failed_counts["missing_smf"] += 1
                    continue

                # Check singles: all > 1.0, max > 1.10
                singles = [smf1, smf2, smf3]
                if not all(s > SMF_BASELINE for s in singles):
                    failed_counts["singles_threshold"] += 1
                    continue
                if max(singles) <= SMF_THRESHOLD:
                    failed_counts["singles_threshold"] += 1
                    continue

                # Get all DMF values
                pair_12 = tuple(sorted([g1, g2]))
                pair_13 = tuple(sorted([g1, g3]))
                pair_23 = tuple(sorted([g2, g3]))

                dmf_12 = dmf_fitness.get(pair_12)
                dmf_13 = dmf_fitness.get(pair_13)
                dmf_23 = dmf_fitness.get(pair_23)

                if dmf_12 is None or dmf_13 is None or dmf_23 is None:
                    failed_counts["missing_dmf"] += 1
                    continue

                # Check doubles: all > 1.0, max > max(singles) + gap
                doubles = [dmf_12, dmf_13, dmf_23]
                if not all(d > DMF_BASELINE for d in doubles):
                    failed_counts["doubles_threshold"] += 1
                    continue

                max_single = max(singles)
                if max(doubles) <= max_single + DMF_GAP:
                    failed_counts["doubles_threshold"] += 1
                    continue

                # Triple passes all filters!
                triple = (g1, g2, g3)

                # Check TMI
                if check_triple_exists_in_tmi(triple, is_any_perturbed_gene_index):
                    output_files['tmi_removed'].write(f"{g1},{g2},{g3}\n")
                    tmi_removed_count += 1
                else:
                    output_files['kept'].write(f"{g1},{g2},{g3}\n")
                    kept_count += 1

    print(f"\nGeneration statistics:")
    print(f"  Total candidates: {total_generated:,}")
    print(f"  Passed filters: {kept_count + tmi_removed_count:,}")
    print(f"  Kept (not in TMI): {kept_count:,}")
    print(f"  Rejected (in TMI): {tmi_removed_count:,}")

    print(f"\nRejection reasons:")
    for reason, count in failed_counts.items():
        print(f"  {reason}: {count:,}")

    return kept_count, tmi_removed_count, total_generated, failed_counts


def check_triple_exists_in_tmi(triple, is_any_perturbed_gene_index):
    """Check if a triple exists in TMI datasets."""
    indices_gene1 = set(is_any_perturbed_gene_index.get(triple[0], []))
    indices_gene2 = set(is_any_perturbed_gene_index.get(triple[1], []))
    indices_gene3 = set(is_any_perturbed_gene_index.get(triple[2], []))

    common_indices = indices_gene1 & indices_gene2 & indices_gene3
    return len(common_indices) > 0


def create_visualizations(
    smf_fitness, dmf_fitness, adjacency, valid_pairs,
    total_generated, kept_count, tmi_removed_count, failed_counts, ts
):
    """Create visualizations of the filtering process."""

    # Consistent annotation box style - light gray background, black text
    ANNOTATION_BOX = dict(boxstyle='round', facecolor='#E8E8E8', alpha=0.8, edgecolor='#CCCCCC')

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))

    # 1. SMF distribution
    smf_values = list(smf_fitness.values())
    ax1.hist(smf_values, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(SMF_BASELINE, color='#D86E2F', linestyle='--', linewidth=2,
                label=f'Baseline: {SMF_BASELINE}')
    ax1.axvline(SMF_THRESHOLD, color='#B73C39', linestyle='--', linewidth=2,
                label=f'Threshold: {SMF_THRESHOLD}')
    ax1.set_xlabel('Single Mutant Fitness', fontsize=12)
    ax1.set_ylabel('Number of Genes', fontsize=12)
    ax1.set_title('SMF Distribution (Costanzo2016, σ=0.063)', fontsize=14)
    ax1.legend(loc='upper left')

    above_threshold = sum(1 for f in smf_values if f > SMF_THRESHOLD)
    # Bottom-left: SMF mass piles up near fitness=1.0 (so the upper-right collides
    # with the peak); the bottom-left corner is the clear spot here.
    ax1.text(0.03, 0.05,
             f'Total: {len(smf_values):,}\n'
             f'> {SMF_THRESHOLD}: {above_threshold:,} ({100*above_threshold/len(smf_values):.1f}%)',
             transform=ax1.transAxes, fontsize=10, color='black',
             verticalalignment='bottom', horizontalalignment='left',
             bbox=ANNOTATION_BOX)

    # 2. DMF distribution
    dmf_values = list(dmf_fitness.values())
    ax2.hist(dmf_values, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(DMF_BASELINE, color='#D86E2F', linestyle='--', linewidth=2,
                label=f'Baseline: {DMF_BASELINE}')
    ax2.set_xlabel('Double Mutant Fitness', fontsize=12)
    ax2.set_ylabel('Number of Pairs', fontsize=12)
    ax2.set_title('DMF Distribution (All Datasets)', fontsize=14)
    ax2.legend(loc='upper left')

    above_baseline = sum(1 for f in dmf_values if f > DMF_BASELINE)
    ax2.text(0.97, 0.97,
             f'Total: {len(dmf_values):,}\n'
             f'> {DMF_BASELINE}: {above_baseline:,} ({100*above_baseline/len(dmf_values):.1f}%)',
             transform=ax2.transAxes, fontsize=10, color='black',
             verticalalignment='top', horizontalalignment='right',
             bbox=ANNOTATION_BOX)

    # 3. Gene connectivity
    if adjacency:
        connectivity_values = [len(neighbors) for neighbors in adjacency.values()]
        ax3.hist(connectivity_values, bins=50, edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Number of Valid Pairs per Gene', fontsize=12)
        ax3.set_ylabel('Number of Genes', fontsize=12)
        ax3.set_title('Gene Connectivity in Adjacency Graph', fontsize=14)
        ax3.axvline(np.mean(connectivity_values), color='#B73C39', linestyle='--',
                   label=f'Mean: {np.mean(connectivity_values):.1f}')
        ax3.legend(loc='upper left')

    # 4. Rejection breakdown - use horizontal bar chart with log scale
    # Colors from torchcell.mplstyle color cycle
    categories = [
        ('Kept (final triples)', kept_count, '#6B8D3A'),           # green
        ('TMI exists (in training)', tmi_removed_count, '#775A9F'), # purple
        ('Missing SMF data', failed_counts["missing_smf"], '#7191A9'),  # blue-gray
        ('Missing DMF data', failed_counts["missing_dmf"], '#6D666F'),  # gray
        (f'Singles filter\n$\\mathit{{all>{SMF_BASELINE}}}$\n$\\mathit{{max>{SMF_THRESHOLD}}}$', failed_counts["singles_threshold"], '#D86E2F'),  # orange
        (f'Doubles filter\n$\\mathit{{all>{DMF_BASELINE}}}$\n$\\mathit{{max>max(SMF)+{DMF_GAP}}}$', failed_counts["doubles_threshold"], '#B73C39'),  # red
    ]

    # Filter out zero values and reverse for bottom-to-top display
    non_zero = [(cat, val, col) for cat, val, col in categories if val > 0]
    non_zero.reverse()

    if non_zero:
        cats, vals, cols = zip(*non_zero)
        y_pos = np.arange(len(cats))

        bars = ax4.barh(y_pos, vals, color=cols, edgecolor='black', alpha=0.9)

        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(cats, fontsize=10, color='black')
        # "Total evaluated" goes in the title (collision-free) rather than a box
        # floating over the bars.
        total_evaluated = kept_count + tmi_removed_count + sum(failed_counts.values())
        ax4.set_title(
            f'Triple Filtering Breakdown (Raw Counts)\nTotal evaluated: {total_evaluated:,}',
            fontsize=14,
        )

        # Decide scale and x-limits BEFORE labeling so positions are final.
        vmax = max(vals)
        vmin_pos = min(v for v in vals if v > 0)
        if vmax > 100 * vmin_pos:
            ax4.set_xscale('log')
            ax4.set_xlabel('Number of Triple Candidates (log scale)', fontsize=12)
            left, right = vmin_pos / 2.0, vmax * 8.0
            ax4.set_xlim(left, right)
            span = np.log10(right) - np.log10(left)
            fracs = [(np.log10(w) - np.log10(left)) / span for w in vals]
        else:
            ax4.set_xlabel('Number of Triple Candidates', fontsize=12)
            right = vmax * 1.18
            ax4.set_xlim(0, right)
            fracs = [w / right for w in vals]

        # Adaptive value labels, all in BLACK: long bars (>75% of axis) get the
        # value just INSIDE the bar end, short bars just OUTSIDE. Point offsets are
        # scale-agnostic so a label can never run off the figure.
        for bar, val, frac in zip(bars, vals, fracs):
            yc = bar.get_y() + bar.get_height() / 2
            if frac > 0.75:
                ax4.annotate(f'{val:,}', xy=(bar.get_width(), yc), xytext=(-5, 0),
                             textcoords='offset points', ha='right', va='center',
                             fontsize=10, fontweight='bold', color='black')
            else:
                ax4.annotate(f'{val:,}', xy=(bar.get_width(), yc), xytext=(5, 0),
                             textcoords='offset points', ha='left', va='center',
                             fontsize=10, fontweight='bold', color='black')

    # Add filtering criteria as figure suptitle
    fig.suptitle(
        f'Inference Dataset 2 Filtering\n'
        f'Criteria: Singles(all>{SMF_BASELINE}, max>{SMF_THRESHOLD}) · '
        f'Doubles(all>{DMF_BASELINE}, max>max(SMF)+{DMF_GAP})',
        fontsize=13, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = osp.join(ASSET_IMAGES_DIR, f"010-kuzmin-tmi/inference_2_filtering.png")
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\n  Saved visualization to {output_path}")


def parse_generation_summary(summary_file):
    """Recover the rejection/keep counts from a prior run's generation_summary.txt.

    These counts only exist after the (multi-hour) generation, so --plot-only reads
    them back rather than recomputing.
    """
    counts = {
        "total_generated": 0,
        "kept_count": 0,
        "tmi_removed_count": 0,
        "failed_counts": {},
    }
    in_rejection = False
    with open(summary_file) as f:
        for line in f:
            s = line.strip()
            if s.startswith("Candidates evaluated:"):
                counts["total_generated"] = int(s.split(":")[1])
            elif s.startswith("Kept (not in TMI):"):
                counts["kept_count"] = int(s.split(":")[1])
            elif s.startswith("Rejected (in TMI):"):
                counts["tmi_removed_count"] = int(s.split(":")[1])
            elif s.startswith("Rejection reasons:"):
                in_rejection = True
            elif in_rejection and ":" in s:
                reason, val = s.split(":")
                counts["failed_counts"][reason.strip()] = int(val)
    return counts


def replot_from_saved():
    """Redraw the filtering plot WITHOUT regenerating triples.

    Reloads SMF/DMF/adjacency (minutes — these drive the distribution histograms)
    and reads the keep/reject counts back from generation_summary.txt, then calls the
    same create_visualizations() used by the full pipeline. Skips the streaming
    generation and the Neo4j TMI load.
    """
    ts = timestamp()
    results_dir = "experiments/010-kuzmin-tmi/results/inference_dataset_2"
    summary_file = osp.join(results_dir, "generation_summary.txt")
    print(f"[--plot-only] reloading SMF/DMF/adjacency; counts from {summary_file}")

    essential_genes = load_essential_genes(DATA_ROOT)
    smf_dataset = SmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/smf_costanzo2016")
    )
    smf_fitness_raw = load_smf_from_dataset(smf_dataset)
    smf_fitness = {
        gene: fitness
        for gene, fitness in smf_fitness_raw.items()
        if gene not in essential_genes
    }

    dmf_fitness = {}
    for ds_cls, sub in [
        (DmfCostanzo2016Dataset, "dmf_costanzo2016"),
        (DmfKuzmin2018Dataset, "dmf_kuzmin2018"),
        (DmfKuzmin2020Dataset, "dmf_kuzmin2020"),
    ]:
        ds = ds_cls(root=osp.join(DATA_ROOT, f"data/torchcell/{sub}"))
        for pair, fitness in load_dmf_from_dataset(ds).items():
            if pair not in dmf_fitness or fitness > dmf_fitness[pair]:
                dmf_fitness[pair] = fitness

    adjacency, valid_pairs = build_adjacency_graph(smf_fitness, dmf_fitness)
    counts = parse_generation_summary(summary_file)

    create_visualizations(
        smf_fitness, dmf_fitness, adjacency, valid_pairs,
        counts["total_generated"], counts["kept_count"],
        counts["tmi_removed_count"], counts["failed_counts"], ts,
    )
    print("[--plot-only] done")


def main():
    ts = timestamp()
    print(f"Starting triple generation for inference_2 at {ts}")
    print(f"Thresholds: SMF > {SMF_THRESHOLD}, DMF > max(SMF) + {DMF_GAP}")

    # Setup output directories
    inference_dir = osp.join(
        DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/inference_2"
    )
    raw_dir = osp.join(inference_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    results_dir = "experiments/010-kuzmin-tmi/results/inference_dataset_2"
    os.makedirs(results_dir, exist_ok=True)

    # Load essential genes to exclude
    print("\n" + "="*80)
    print("Loading essential genes (to exclude)")
    print("="*80)

    essential_genes = load_essential_genes(DATA_ROOT)

    # Load SMF from Costanzo2016 (lowest noise σ=0.063)
    print("\n" + "="*80)
    print("Loading SMF dataset (Costanzo2016 - lowest noise σ=0.063)")
    print("="*80)

    smf_dataset = SmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/smf_costanzo2016")
    )
    smf_fitness_raw = load_smf_from_dataset(smf_dataset)

    # Filter out essential genes from SMF
    smf_fitness = {gene: fitness for gene, fitness in smf_fitness_raw.items()
                   if gene not in essential_genes}
    print(f"\nFiltered SMF: {len(smf_fitness_raw)} → {len(smf_fitness)} genes "
          f"(removed {len(smf_fitness_raw) - len(smf_fitness)} essential genes)")

    # Load DMF from all datasets
    print("\n" + "="*80)
    print("Loading DMF datasets...")
    print("="*80)

    dmf_fitness = {}

    # Costanzo2016 DMF
    dmf_costanzo = DmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/dmf_costanzo2016")
    )
    dmf_costanzo_fitness = load_dmf_from_dataset(dmf_costanzo)
    for pair, fitness in dmf_costanzo_fitness.items():
        if pair not in dmf_fitness or fitness > dmf_fitness[pair]:
            dmf_fitness[pair] = fitness

    # Kuzmin2018 DMF
    dmf_kuzmin2018 = DmfKuzmin2018Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/dmf_kuzmin2018")
    )
    dmf_kuzmin2018_fitness = load_dmf_from_dataset(dmf_kuzmin2018)
    for pair, fitness in dmf_kuzmin2018_fitness.items():
        if pair not in dmf_fitness or fitness > dmf_fitness[pair]:
            dmf_fitness[pair] = fitness

    # Kuzmin2020 DMF
    dmf_kuzmin2020 = DmfKuzmin2020Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/dmf_kuzmin2020")
    )
    dmf_kuzmin2020_fitness = load_dmf_from_dataset(dmf_kuzmin2020)
    for pair, fitness in dmf_kuzmin2020_fitness.items():
        if pair not in dmf_fitness or fitness > dmf_fitness[pair]:
            dmf_fitness[pair] = fitness

    print(f"\nTotal unique pairs with DMF: {len(dmf_fitness):,}")

    # Build adjacency graph
    adjacency, valid_pairs = build_adjacency_graph(smf_fitness, dmf_fitness)

    # Load Neo4j dataset for TMI filtering
    print("\n" + "="*80)
    print("Loading Neo4j dataset for TMI filtering...")
    print("="*80)

    with open("experiments/009-kuzmin-tmi/queries/001_small_build.cql", "r") as f:
        query = f.read()

    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()

    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/009-kuzmin-tmi/001-small-build"
    )

    neo4j_dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        uri="bolt://torchcell-database.ncsa.illinois.edu:7687",
        username="readonly",
        password="ReadOnly",
        graphs=None,
        node_embeddings=None,
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )

    is_any_perturbed_gene_index = neo4j_dataset.is_any_perturbed_gene_index

    # Setup output files
    triple_list_file = osp.join(raw_dir, "triple_combinations_list.parquet")
    tmi_removed_file = osp.join(results_dir, "tmi_removed_triples.parquet")

    print(f"\nOutput files:")
    print(f"  Kept triples: {triple_list_file}")
    print(f"  TMI removed: {tmi_removed_file}")

    # Generate triples
    writer_kept = StreamingParquetWriter(triple_list_file)
    writer_tmi = StreamingParquetWriter(tmi_removed_file)

    output_files = {
        'kept': writer_kept,
        'tmi_removed': writer_tmi
    }

    kept_count, tmi_removed_count, total_generated, failed_counts = generate_triples_streaming(
        adjacency, smf_fitness, dmf_fitness, is_any_perturbed_gene_index, output_files
    )

    writer_kept.close()
    writer_tmi.close()

    neo4j_dataset.close_lmdb()

    # Save summary
    summary_file = osp.join(results_dir, "generation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Inference Dataset 2 - Triple Generation Summary\n")
        f.write(f"Timestamp: {ts}\n\n")
        f.write(f"SMF Source: SmfCostanzo2016Dataset (σ=0.063, lowest noise)\n\n")
        f.write(f"Essential Gene Filtering:\n")
        f.write(f"  Essential genes excluded: {len(essential_genes)}\n\n")
        f.write(f"Thresholds:\n")
        f.write(f"  SMF baseline: > {SMF_BASELINE}\n")
        f.write(f"  SMF threshold: max > {SMF_THRESHOLD}\n")
        f.write(f"  DMF baseline: > {DMF_BASELINE}\n")
        f.write(f"  DMF gap: max(doubles) > max(singles) + {DMF_GAP}\n\n")
        f.write(f"Data loaded:\n")
        f.write(f"  Genes with SMF (after filtering): {len(smf_fitness)}\n")
        f.write(f"  Unique pairs with DMF: {len(dmf_fitness)}\n")
        f.write(f"  Valid pairs (adjacency): {len(valid_pairs)}\n\n")
        f.write(f"Triple generation:\n")
        f.write(f"  Candidates evaluated: {total_generated}\n")
        f.write(f"  Kept (not in TMI): {kept_count}\n")
        f.write(f"  Rejected (in TMI): {tmi_removed_count}\n\n")
        f.write(f"Rejection reasons:\n")
        for reason, count in failed_counts.items():
            f.write(f"  {reason}: {count}\n")

    print(f"\nSaved summary to {summary_file}")

    # Create visualizations
    create_visualizations(
        smf_fitness, dmf_fitness, adjacency, valid_pairs,
        total_generated, kept_count, tmi_removed_count, failed_counts, ts
    )

    # Final summary
    print("\n" + "="*80)
    print("INFERENCE DATASET 2 - TRIPLE GENERATION COMPLETE")
    print("="*80)
    print(f"SMF Source: SmfCostanzo2016Dataset (σ=0.063)")
    print(f"Essential genes excluded: {len(essential_genes)}")
    print(f"Thresholds: SMF > {SMF_THRESHOLD}, DMF > max(SMF) + {DMF_GAP}")
    print(f"\nData loaded:")
    print(f"  Genes with SMF (non-essential): {len(smf_fitness)}")
    print(f"  Pairs with DMF: {len(dmf_fitness)}")
    print(f"  Valid pairs: {len(valid_pairs)}")
    print(f"\nResults:")
    print(f"  Candidates evaluated: {total_generated:,}")
    print(f"  Final triples: {kept_count:,}")
    print(f"  Output: {triple_list_file}")

    if kept_count == 0:
        print(f"\nWARNING: No triples passed filtering!")
        print(f"Consider adjusting thresholds.")
    else:
        print(f"\nSuccessfully generated {kept_count:,} triples for inference!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate inference-2 triples; --plot-only redraws the filtering plot."
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip generation; reload SMF/DMF/adjacency + generation_summary.txt counts and redraw the filtering plot",
    )
    cli_args = parser.parse_args()
    if cli_args.plot_only:
        replot_from_saved()
    else:
        main()
