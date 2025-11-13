#!/usr/bin/env python
"""
Quick test script to examine the structure of the full masks dataset.
"""

import os
import os.path as osp
import torch
from dotenv import load_dotenv

# Core imports
from torchcell.data import GenotypeAggregator, MeanExperimentDeduplicator
from torchcell.data.neo4j_cell import Neo4jCellDataset
from torchcell.data.neo4j_preprocessed_cell_full_masks import Neo4jPreprocessedCellDatasetFullMasks
from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder
from torchcell.graph import SCerevisiaeGraph
from torchcell.graph.graph import build_gene_multigraph
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")


def main():
    print("=" * 80)
    print("TESTING FULL MASKS DATASET STRUCTURE")
    print("=" * 80)

    # Setup query
    with open(
        osp.join(EXPERIMENT_ROOT, "006-kuzmin-tmi/queries/001_small_build.cql"), "r"
    ) as f:
        query = f.read()

    # Setup genome
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()

    # Setup graph
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    # Build gene multigraph (matching config 074/075)
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
    yeast_gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))
    incidence_graphs = {"metabolism_bipartite": yeast_gem.bipartite_graph}

    # Create source dataset (needed for cell_graph reference)
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

    # Load the full masks preprocessed dataset
    preprocessed_root = osp.join(
        DATA_ROOT,
        "data/torchcell/experiments/006-kuzmin-tmi/001-small-build-preprocessed-full-masks"
    )

    print(f"\nLoading full masks dataset from: {preprocessed_root}")

    dataset = Neo4jPreprocessedCellDatasetFullMasks(
        root=preprocessed_root,
        source_dataset=source_dataset,
    )

    print(f"Dataset length: {len(dataset)}")
    print("\n" + "=" * 80)
    print("EXAMINING FIRST SAMPLE")
    print("=" * 80)

    # Get the first sample
    sample = dataset[0]

    print("\nSample type:", type(sample))
    print("\nSample attributes:")
    for attr in dir(sample):
        if not attr.startswith('_'):
            try:
                value = getattr(sample, attr)
                if isinstance(value, torch.Tensor):
                    print(f"  {attr}: Tensor with shape {value.shape}, dtype {value.dtype}")
                    # Print some statistics for phenotype data
                    if 'phenotype' in attr and 'values' in attr:
                        print(f"    Range: [{value.min().item():.4f}, {value.max().item():.4f}]")
                        print(f"    Mean: {value.mean().item():.4f}")
                        print(f"    Std: {value.std().item():.4f}")
                        print(f"    First 5 values: {value[:5].tolist()}")
                elif isinstance(value, list):
                    print(f"  {attr}: List with {len(value)} items")
                    if len(value) > 0 and not callable(value[0]):
                        print(f"    First item: {value[0]}")
                elif not callable(value):
                    print(f"  {attr}: {type(value).__name__}")
            except:
                pass

    # Check for gene node data
    print("\n" + "=" * 80)
    print("PHENOTYPE DATA IN SAMPLE")
    print("=" * 80)

    # Look for 'gene' key in the HeteroData
    if 'gene' in sample:
        gene_data = sample['gene']
        print(f"Gene data type: {type(gene_data)}")
        print(f"Gene data keys: {list(gene_data.keys())}")

        if 'phenotype_values' in gene_data:
            phen_vals = gene_data['phenotype_values']
            print(f"\nPhenotype values shape: {phen_vals.shape}, dtype: {phen_vals.dtype}")
            print(f"  Range: [{phen_vals.min().item():.4f}, {phen_vals.max().item():.4f}]")
            print(f"  Mean: {phen_vals.mean().item():.4f}")
            print(f"  Std: {phen_vals.std().item():.4f}")
            print(f"  First 10 values: {phen_vals[:10].tolist()}")

        if 'phenotype_stat_values' in gene_data:
            stat_vals = gene_data['phenotype_stat_values']
            print(f"\nPhenotype stat values shape: {stat_vals.shape}, dtype: {stat_vals.dtype}")
            print(f"  Range: [{stat_vals.min().item():.4f}, {stat_vals.max().item():.4f}]")
            print(f"  Mean: {stat_vals.mean().item():.4f}")
            print(f"  Std: {stat_vals.std().item():.4f}")

        # Check for phenotype labels
        if 'phenotype_types' in gene_data:
            print(f"\nPhenotype types: {gene_data['phenotype_types']}")

        if 'phenotype_stat_types' in gene_data:
            print(f"Phenotype stat types: {gene_data['phenotype_stat_types']}")

    print("\n" + "=" * 80)
    print("CHECKING MASK TYPES")
    print("=" * 80)

    # Check node masks
    for node_type in sample.node_types:
        if hasattr(sample[node_type], 'mask'):
            mask = sample[node_type]['mask']
            print(f"{node_type} mask: shape {mask.shape}, dtype {mask.dtype}")

    # Check edge masks
    for edge_type in sample.edge_types:
        if 'mask' in sample[edge_type]:
            mask = sample[edge_type]['mask']
            print(f"{edge_type} mask: shape {mask.shape}, dtype {mask.dtype}")

    print("\n" + "=" * 80)
    print("DATASET EXAMINATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()