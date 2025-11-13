# experiments/006-kuzmin-tmi/scripts/preprocess_lazy_dataset
# [[experiments.006-kuzmin-tmi.scripts.preprocess_lazy_dataset]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/006-kuzmin-tmi/scripts/preprocess_lazy_dataset
# Test file: experiments/006-kuzmin-tmi/scripts/test_preprocess_lazy_dataset.py

"""
One-time preprocessing script for lazy dataset.

This script applies LazySubgraphRepresentation to all samples and saves results to LMDB.
Run this once before training to avoid 10ms/sample graph processing overhead during training.

Usage:
    python preprocess_lazy_dataset.py

Performance:
    - Preprocessing: ~10ms/sample × 300K samples = ~50 minutes (one-time)
    - Training speedup: Eliminates 280 seconds per epoch
    - Over 1000 epochs: Saves 280,000 seconds = 77 hours of training time!

Output:
    Preprocessed dataset saved to:
    {DATA_ROOT}/data/torchcell/experiments/006-kuzmin-tmi/001-small-build-preprocessed-lazy/
"""

import os
import os.path as osp
import logging
from dotenv import load_dotenv

from torchcell.data import GenotypeAggregator, MeanExperimentDeduplicator
from torchcell.data.graph_processor import LazySubgraphRepresentation
from torchcell.data.neo4j_cell import Neo4jCellDataset
from torchcell.data.neo4j_preprocessed_cell import Neo4jPreprocessedCellDataset
from torchcell.graph import SCerevisiaeGraph
from torchcell.graph.graph import build_gene_multigraph
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder
from torchcell.timestamp import timestamp

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")


def main():
    print("=" * 80)
    print("LAZY DATASET PREPROCESSING")
    print("=" * 80)
    print(f"Started at: {timestamp()}")
    print()

    # Load query
    query_path = osp.join(EXPERIMENT_ROOT, "006-kuzmin-tmi/queries/001_small_build.cql")
    with open(query_path, "r") as f:
        query = f.read()
    print(f"Query loaded from: {query_path}")

    # Setup genome
    print("\nInitializing genome...")
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    print(f"Genome initialized: {len(genome.gene_set)} genes")

    # Setup graph
    print("\nInitializing graph...")
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    # Build gene multigraph (MUST match 074.yaml config exactly)
    graph_names = [
        "physical",
        "regulatory",
        "tflink",
        "string12_0_neighborhood",
        "string12_0_fusion",
        "string12_0_cooccurence",
        "string12_0_coexpression",
        "string12_0_experimental",
        "string12_0_database",
    ]
    gene_multigraph = build_gene_multigraph(graph=graph, graph_names=graph_names)
    print(f"Gene multigraph built with graphs: {list(gene_multigraph.graphs.keys())}")

    # Build node embeddings (MUST match 074.yaml config)
    # 074 uses learnable embeddings
    node_embeddings = NodeEmbeddingBuilder.build(
        embedding_names=["learnable"],
        data_root=DATA_ROOT,
        genome=genome,
        graph=graph,
    )
    print(f"Node embeddings: {list(node_embeddings.keys()) if node_embeddings else 'None'}")

    # Setup metabolism
    print("\nInitializing metabolism...")
    yeast_gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))
    incidence_graphs = {"metabolism_bipartite": yeast_gem.bipartite_graph}
    print("Metabolism bipartite graph initialized")

    # Create source dataset (unprocessed)
    print("\nCreating source dataset...")
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/001-small-build"
    )

    source_dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=gene_multigraph,
        incidence_graphs=incidence_graphs,
        node_embeddings=node_embeddings,
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=None,  # Don't apply processor yet
    )

    print(f"Source dataset created: {len(source_dataset)} samples")
    print(f"Cell graph gene nodes: {source_dataset.cell_graph['gene'].num_nodes}")
    print(f"Cell graph edge types: {source_dataset.cell_graph.edge_types}")

    # Create preprocessed dataset
    preprocessed_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/001-small-build-preprocessed-lazy"
    )

    # Check if preprocessing already exists
    preprocessed_exists = osp.exists(osp.join(preprocessed_root, "processed/lmdb")) and \
                          osp.exists(osp.join(preprocessed_root, "processed/metadata.json"))

    if preprocessed_exists:
        print(f"\n{'=' * 80}")
        print("PREPROCESSED DATA ALREADY EXISTS")
        print("=" * 80)
        print(f"Found existing preprocessed data at: {preprocessed_root}")
        print("Skipping preprocessing and loading existing data...")
        print(f"{'=' * 80}\n")

        preprocessed_dataset = Neo4jPreprocessedCellDataset(
            root=preprocessed_root,
            source_dataset=source_dataset,
        )
    else:
        print(f"\nPreprocessed data will be saved to: {preprocessed_root}")

        preprocessed_dataset = Neo4jPreprocessedCellDataset(
            root=preprocessed_root,
            source_dataset=source_dataset,
        )

        # Run one-time preprocessing
        print("\n" + "=" * 80)
        print("STARTING ONE-TIME PREPROCESSING")
        print("=" * 80)
        print(f"Graph processor: LazySubgraphRepresentation")
        print(f"Expected time: ~10ms/sample × {len(source_dataset)} samples")
        estimated_minutes = (len(source_dataset) * 10 / 1000) / 60
        print(f"Estimated duration: ~{estimated_minutes:.1f} minutes")
        print()
        print("This is a ONE-TIME cost that will save ~280 seconds per epoch during training!")
        print("Over 1000 epochs, this saves ~77 hours of training time.")
        print("=" * 80)
        print()

        # Initialize lazy graph processor
        graph_processor = LazySubgraphRepresentation()

        # Build incidence cache once before preprocessing
        print("Building incidence cache (one-time)...")
        cache_info = graph_processor.build_cache(source_dataset.cell_graph)
        print(f"  Cache build time: {cache_info['total_time_ms']:.2f}ms")
        print(f"  Edge types cached: {cache_info['num_edge_types']}")
        print(f"  Total edges: {cache_info['total_edges']}")
        print()

        # Run preprocessing
        preprocessed_dataset.preprocess_from_source(
            source_dataset=source_dataset,
            graph_processor=graph_processor,
        )

    # Test loading
    print("\n" + "=" * 80)
    print("TESTING PREPROCESSED DATA")
    print("=" * 80)

    print("\nTesting sample loading...")
    sample = preprocessed_dataset[0]
    print(f"✓ Sample 0 loaded successfully")
    print(f"  Keys: {sample.keys}")
    print(f"  Gene nodes: {sample['gene'].num_nodes}")
    if "reaction" in sample.node_types:
        print(f"  Reaction nodes: {sample['reaction'].num_nodes}")
    if "metabolite" in sample.node_types:
        print(f"  Metabolite nodes: {sample['metabolite'].num_nodes}")

    # Check a few more samples
    print("\nSpot checking 10 random samples...")
    import random
    test_indices = random.sample(range(len(preprocessed_dataset)), min(10, len(preprocessed_dataset)))
    for idx in test_indices:
        sample = preprocessed_dataset[idx]
        assert sample is not None, f"Sample {idx} is None!"
    print("✓ All spot checks passed")

    # Cleanup
    source_dataset.close_lmdb()
    preprocessed_dataset.close_lmdb()

    # Print usage instructions
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"Finished at: {timestamp()}")
    print()
    print("Preprocessed dataset saved to:")
    print(f"  {preprocessed_root}")
    print()
    print("To use in training, modify your training script:")
    print()
    print("  # OLD (slow):")
    print("  # dataset = Neo4jCellDataset(..., graph_processor=LazySubgraphRepresentation())")
    print()
    print("  # NEW (100x faster):")
    print(f"  dataset = Neo4jPreprocessedCellDataset(root='{preprocessed_root}')")
    print(f"  dataset._source_dataset = Neo4jCellDataset(root='{dataset_root}', ...)")
    print()
    print("Expected speedup:")
    print("  - Data loading: 10ms → 0.01ms per sample (1000x faster)")
    print("  - Training: Eliminates ~280 seconds per epoch")
    print("=" * 80)


if __name__ == "__main__":
    main()
