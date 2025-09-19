"""
Extract unique perturbations from the experimental dataset.
This script loads the Neo4jCellDataset and extracts all unique
gene perturbations to prepare for targeted FBA analysis.
"""

import os
import os.path as osp
import json
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv

from torchcell.data import Neo4jCellDataset
from torchcell.data import MeanExperimentDeduplicator, GenotypeAggregator
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.data.graph_processor import SubgraphRepresentation


def main():
    """Extract unique perturbations from the dataset."""
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")
    
    # Paths
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/007-kuzmin-tm/001-small-build"
    )
    results_dir = osp.join(EXPERIMENT_ROOT, "007-kuzmin-tm/results/cobra-fba-growth")
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if perturbations have already been extracted
    output_file = osp.join(results_dir, "unique_perturbations.json")
    stats_file = osp.join(results_dir, "perturbation_statistics.json")
    
    if osp.exists(output_file) and osp.exists(stats_file):
        print(f"Perturbations already extracted. Loading from {output_file}")
        with open(stats_file, "r") as f:
            stats = json.load(f)
        print(f"Dataset size: {stats['dataset_size']}")
        print(f"Unique singles: {stats['unique_singles']}")
        print(f"Unique doubles: {stats['unique_doubles']}")
        print(f"Unique triples: {stats['unique_triples']}")
        print(f"Total unique: {stats['total_unique']}")
        print(f"Reduction from exhaustive: {stats['reduction_percent']:.2f}%")
        return
    
    # Load genome
    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()
    
    print(f"Loading dataset from {dataset_root}")
    
    # Load query
    with open(
        osp.join(EXPERIMENT_ROOT, "007-kuzmin-tm/queries/001_small_build.cql"), "r"
    ) as f:
        query = f.read()
    
    # Create dataset without graphs or embeddings for efficiency
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=None,  # No graphs needed
        incidence_graphs={},  # No incidence graphs needed
        node_embeddings={},  # No embeddings needed
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Extract unique perturbations
    unique_singles = set()
    unique_doubles = set()
    unique_triples = set()
    perturbation_counts = defaultdict(int)
    
    print("Extracting unique perturbations...")
    for i in tqdm(range(len(dataset)), desc="Processing dataset"):
        data = dataset[i]
        
        # Get perturbation gene IDs
        gene_ids = data['gene'].ids_pert
        
        # Sort to ensure consistent ordering
        gene_ids = tuple(sorted(gene_ids))
        perturbation_counts[len(gene_ids)] += 1
        
        if len(gene_ids) == 1:
            unique_singles.add(gene_ids[0])
        elif len(gene_ids) == 2:
            unique_doubles.add(gene_ids)
        elif len(gene_ids) == 3:
            unique_triples.add(gene_ids)
    
    # Extract all component perturbations needed for interaction calculations
    # For doubles, we need singles
    for gene1, gene2 in unique_doubles:
        unique_singles.add(gene1)
        unique_singles.add(gene2)
    
    # For triples, we need singles and all possible doubles
    for gene1, gene2, gene3 in unique_triples:
        unique_singles.add(gene1)
        unique_singles.add(gene2)
        unique_singles.add(gene3)
        unique_doubles.add(tuple(sorted([gene1, gene2])))
        unique_doubles.add(tuple(sorted([gene1, gene3])))
        unique_doubles.add(tuple(sorted([gene2, gene3])))
    
    # Convert to lists for JSON serialization
    perturbations = {
        "singles": sorted(list(unique_singles)),
        "doubles": sorted([list(d) for d in unique_doubles]),
        "triples": sorted([list(t) for t in unique_triples]),
    }
    
    # Print statistics
    print("\n=== Perturbation Statistics ===")
    print(f"Total data points: {len(dataset)}")
    for n, count in sorted(perturbation_counts.items()):
        print(f"  {n}-gene perturbations: {count}")
    
    print(f"\nUnique perturbations:")
    print(f"  Singles: {len(perturbations['singles'])}")
    print(f"  Doubles: {len(perturbations['doubles'])}")
    print(f"  Triples: {len(perturbations['triples'])}")
    print(f"  Total: {len(perturbations['singles']) + len(perturbations['doubles']) + len(perturbations['triples'])}")
    
    # Calculate expected FBA computations
    total_fba = (
        len(perturbations['singles']) + 
        len(perturbations['doubles']) + 
        len(perturbations['triples'])
    )
    print(f"\nTotal FBA computations needed: {total_fba:,}")
    print(f"Reduction from exhaustive: {(1 - total_fba / 1564936281) * 100:.2f}%")
    
    # Save to JSON
    output_file = osp.join(results_dir, "unique_perturbations.json")
    with open(output_file, "w") as f:
        json.dump(perturbations, f, indent=2)
    
    print(f"\nPerturbations saved to: {output_file}")
    
    # Also save statistics
    stats = {
        "dataset_size": len(dataset),
        "perturbation_counts": dict(perturbation_counts),
        "unique_singles": len(perturbations['singles']),
        "unique_doubles": len(perturbations['doubles']),
        "unique_triples": len(perturbations['triples']),
        "total_unique": total_fba,
        "reduction_percent": (1 - total_fba / 1564936281) * 100
    }
    
    stats_file = osp.join(results_dir, "perturbation_statistics.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to: {stats_file}")


if __name__ == "__main__":
    main()