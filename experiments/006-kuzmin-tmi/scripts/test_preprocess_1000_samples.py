#!/usr/bin/env python
# experiments/006-kuzmin-tmi/scripts/test_preprocess_1000_samples
# Test preprocessing with only 1000 samples to verify storage calculations

"""
Test preprocessing script that saves only 1000 samples to verify storage.

Tests both uint8 and bfloat16 storage formats to compare:
- uint8: 1 byte per boolean (original plan)
- bfloat16: 2 bytes per boolean (matches training precision)

Expected sizes for 1000 samples:
- uint8: ~2.5MB per sample = 2.5GB total
- bfloat16: ~5MB per sample = 5GB total
"""

import os
import os.path as osp
import pickle
import lmdb
import torch
from tqdm import tqdm
from dotenv import load_dotenv

from torchcell.data import GenotypeAggregator, MeanExperimentDeduplicator
from torchcell.data.graph_processor import LazySubgraphRepresentation
from torchcell.data.neo4j_cell import Neo4jCellDataset
from torchcell.graph import SCerevisiaeGraph
from torchcell.graph.graph import build_gene_multigraph
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder
from torchcell.timestamp import timestamp

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")

# Test parameters
NUM_SAMPLES = 1000
TEST_BOTH_FORMATS = True  # Test both uint8 and bfloat16


def extract_masks_uint8(processed_graph):
    """Extract masks as uint8 (1 byte per boolean)."""
    full_data = {}

    # Store sample-specific data
    gene_data = processed_graph['gene']
    full_data['gene'] = {
        'ids_pert': gene_data.ids_pert,
        'perturbation_indices': gene_data.perturbation_indices.cpu(),
        'phenotype_values': gene_data.phenotype_values.cpu(),
        'phenotype_type_indices': gene_data.phenotype_type_indices.cpu(),
        'phenotype_sample_indices': gene_data.phenotype_sample_indices.cpu(),
        'phenotype_types': gene_data.phenotype_types,
        'phenotype_stat_values': gene_data.phenotype_stat_values.cpu(),
        'phenotype_stat_type_indices': gene_data.phenotype_stat_type_indices.cpu(),
        'phenotype_stat_sample_indices': gene_data.phenotype_stat_sample_indices.cpu(),
        'phenotype_stat_types': gene_data.phenotype_stat_types,
    }

    # Store node masks as uint8
    full_data['node_masks'] = {}
    for node_type in processed_graph.node_types:
        if 'mask' in processed_graph[node_type]:
            mask = processed_graph[node_type]['mask'].cpu()
            full_data['node_masks'][node_type] = mask.to(torch.uint8)

    # Store edge masks as uint8
    full_data['edge_masks'] = {}
    for edge_type in processed_graph.edge_types:
        if 'mask' in processed_graph[edge_type]:
            mask = processed_graph[edge_type]['mask'].cpu()
            full_data['edge_masks'][edge_type] = mask.to(torch.uint8)

    return full_data


def extract_masks_bfloat16(processed_graph):
    """Extract masks as bfloat16 (2 bytes, matches training precision)."""
    full_data = {}

    # Store sample-specific data (same as uint8)
    gene_data = processed_graph['gene']
    full_data['gene'] = {
        'ids_pert': gene_data.ids_pert,
        'perturbation_indices': gene_data.perturbation_indices.cpu(),
        'phenotype_values': gene_data.phenotype_values.cpu(),
        'phenotype_type_indices': gene_data.phenotype_type_indices.cpu(),
        'phenotype_sample_indices': gene_data.phenotype_sample_indices.cpu(),
        'phenotype_types': gene_data.phenotype_types,
        'phenotype_stat_values': gene_data.phenotype_stat_values.cpu(),
        'phenotype_stat_type_indices': gene_data.phenotype_stat_type_indices.cpu(),
        'phenotype_stat_sample_indices': gene_data.phenotype_stat_sample_indices.cpu(),
        'phenotype_stat_types': gene_data.phenotype_stat_types,
    }

    # Store node masks as bfloat16
    full_data['node_masks'] = {}
    for node_type in processed_graph.node_types:
        if 'mask' in processed_graph[node_type]:
            mask = processed_graph[node_type]['mask'].cpu()
            # Convert bool -> float -> bfloat16
            full_data['node_masks'][node_type] = mask.to(torch.bfloat16)

    # Store edge masks as bfloat16
    full_data['edge_masks'] = {}
    for edge_type in processed_graph.edge_types:
        if 'mask' in processed_graph[edge_type]:
            mask = processed_graph[edge_type]['mask'].cpu()
            # Convert bool -> float -> bfloat16
            full_data['edge_masks'][edge_type] = mask.to(torch.bfloat16)

    return full_data


def main():
    print("=" * 80)
    print(f"TEST PREPROCESSING: {NUM_SAMPLES} SAMPLES")
    print("=" * 80)
    print(f"Started at: {timestamp()}")
    print()

    # Load query
    query_path = osp.join(EXPERIMENT_ROOT, "006-kuzmin-tmi/queries/001_small_build.cql")
    with open(query_path, "r") as f:
        query = f.read()

    # Setup genome
    print("Initializing genome...")
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()

    # Setup graph
    print("Initializing graph...")
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    # Build gene multigraph
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

    # Build node embeddings
    node_embeddings = NodeEmbeddingBuilder.build(
        embedding_names=["learnable"],
        data_root=DATA_ROOT,
        genome=genome,
        graph=graph,
    )

    # Setup metabolism
    print("Initializing metabolism...")
    yeast_gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))
    incidence_graphs = {"metabolism_bipartite": yeast_gem.bipartite_graph}

    # Create source dataset
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
        graph_processor=None,
    )

    print(f"Source dataset created: {len(source_dataset)} samples")
    print(f"Will process first {NUM_SAMPLES} samples only")

    # Initialize graph processor
    graph_processor = LazySubgraphRepresentation()
    print("\nBuilding incidence cache...")
    cache_info = graph_processor.build_cache(source_dataset.cell_graph)
    print(f"  Cache build time: {cache_info['total_time_ms']:.2f}ms")

    # Test UINT8 format
    if TEST_BOTH_FORMATS:
        print("\n" + "=" * 80)
        print("TESTING UINT8 FORMAT (1 byte per boolean)")
        print("=" * 80)

        test_dir_uint8 = osp.join(
            DATA_ROOT,
            f"data/torchcell/experiments/006-kuzmin-tmi/test-{NUM_SAMPLES}-samples-uint8"
        )
        os.makedirs(osp.join(test_dir_uint8, "processed"), exist_ok=True)
        lmdb_path_uint8 = osp.join(test_dir_uint8, "processed/lmdb")

        # Small map_size for test
        map_size = int(10e9)  # 10GB should be plenty for 1000 samples
        env_uint8 = lmdb.open(lmdb_path_uint8, map_size=map_size)

        total_size_uint8 = 0
        sample_sizes_uint8 = []

        with env_uint8.begin(write=True) as txn:
            for idx in tqdm(range(NUM_SAMPLES), desc="Processing (uint8)"):
                # Get raw data
                source_dataset._init_lmdb_read()
                serialized_data = source_dataset._read_from_lmdb(idx)
                data_list = source_dataset._deserialize_json(serialized_data)
                data = source_dataset._reconstruct_experiments(data_list)

                # Apply graph processor
                processed_graph = graph_processor.process(
                    source_dataset.cell_graph,
                    source_dataset.phenotype_info,
                    data
                )

                # Extract masks as uint8
                full_data = extract_masks_uint8(processed_graph)

                # Serialize and measure
                serialized = pickle.dumps(full_data)
                sample_size = len(serialized)
                total_size_uint8 += sample_size
                sample_sizes_uint8.append(sample_size)

                # Store in LMDB
                txn.put(f"{idx}".encode("utf-8"), serialized)

        env_uint8.close()
        source_dataset.close_lmdb()

        # Calculate statistics
        avg_size_uint8 = total_size_uint8 / NUM_SAMPLES
        print(f"\nUINT8 Results:")
        print(f"  Total size: {total_size_uint8 / 1e9:.3f} GB")
        print(f"  Average per sample: {avg_size_uint8 / 1e6:.3f} MB")
        print(f"  Min sample size: {min(sample_sizes_uint8) / 1e6:.3f} MB")
        print(f"  Max sample size: {max(sample_sizes_uint8) / 1e6:.3f} MB")
        print(f"  Projected for 332K samples: {(avg_size_uint8 * 332313) / 1e9:.1f} GB")

    # Test BFLOAT16 format
    print("\n" + "=" * 80)
    print("TESTING BFLOAT16 FORMAT (2 bytes, matches training precision)")
    print("=" * 80)

    test_dir_bf16 = osp.join(
        DATA_ROOT,
        f"data/torchcell/experiments/006-kuzmin-tmi/test-{NUM_SAMPLES}-samples-bfloat16"
    )
    os.makedirs(osp.join(test_dir_bf16, "processed"), exist_ok=True)
    lmdb_path_bf16 = osp.join(test_dir_bf16, "processed/lmdb")

    env_bf16 = lmdb.open(lmdb_path_bf16, map_size=map_size)

    total_size_bf16 = 0
    sample_sizes_bf16 = []

    with env_bf16.begin(write=True) as txn:
        for idx in tqdm(range(NUM_SAMPLES), desc="Processing (bfloat16)"):
            # Get raw data
            source_dataset._init_lmdb_read()
            serialized_data = source_dataset._read_from_lmdb(idx)
            data_list = source_dataset._deserialize_json(serialized_data)
            data = source_dataset._reconstruct_experiments(data_list)

            # Apply graph processor
            processed_graph = graph_processor.process(
                source_dataset.cell_graph,
                source_dataset.phenotype_info,
                data
            )

            # Extract masks as bfloat16
            full_data = extract_masks_bfloat16(processed_graph)

            # Serialize and measure
            serialized = pickle.dumps(full_data)
            sample_size = len(serialized)
            total_size_bf16 += sample_size
            sample_sizes_bf16.append(sample_size)

            # Store in LMDB
            txn.put(f"{idx}".encode("utf-8"), serialized)

    env_bf16.close()
    source_dataset.close_lmdb()

    # Calculate statistics
    avg_size_bf16 = total_size_bf16 / NUM_SAMPLES
    print(f"\nBFLOAT16 Results:")
    print(f"  Total size: {total_size_bf16 / 1e9:.3f} GB")
    print(f"  Average per sample: {avg_size_bf16 / 1e6:.3f} MB")
    print(f"  Min sample size: {min(sample_sizes_bf16) / 1e6:.3f} MB")
    print(f"  Max sample size: {max(sample_sizes_bf16) / 1e6:.3f} MB")
    print(f"  Projected for 332K samples: {(avg_size_bf16 * 332313) / 1e9:.1f} GB")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    if TEST_BOTH_FORMATS:
        print(f"Size ratio (bf16/uint8): {avg_size_bf16/avg_size_uint8:.2f}x")
        print(f"Storage difference for 332K samples: {((avg_size_bf16 - avg_size_uint8) * 332313) / 1e9:.1f} GB")

    print(f"\nAvailable storage on /scratch: 4,900 GB")
    print(f"Storage check for bf16: {'✓ FITS' if (avg_size_bf16 * 332313) / 1e9 < 4900 else '✗ TOO LARGE'}")

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print("Use BFLOAT16 if:")
    print("  1. Storage fits (likely ~1.6TB)")
    print("  2. You want to avoid conversion overhead during bf16-mixed training")
    print("  3. You want exact precision match with training")
    print("\nUse UINT8 if:")
    print("  1. Storage is tight (~820GB)")
    print("  2. Conversion overhead is acceptable")
    print("  3. You might train with different precisions")
    print("=" * 80)

    # Clean up test directories
    print(f"\nTest data saved to:")
    if TEST_BOTH_FORMATS:
        print(f"  {test_dir_uint8}")
    print(f"  {test_dir_bf16}")


if __name__ == "__main__":
    main()