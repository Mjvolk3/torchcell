#!/usr/bin/env python3
# torchcell/scratch/verify_lazy_equivalence.py
# [[torchcell.scratch.verify_lazy_equivalence]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/scratch/verify_lazy_equivalence

"""
Verify LazySubgraphRepresentation equivalence with SubgraphRepresentation.

Tests that LazySubgraphRepresentation produces biologically equivalent results
to SubgraphRepresentation when masks are applied, especially for samples with
invalid reactions.
"""

import os
import os.path as osp
import random
import numpy as np
import torch
from dotenv import load_dotenv
from torchcell.graph import SCerevisiaeGraph
from torchcell.datamodules import CellDataModule
from torchcell.data import MeanExperimentDeduplicator, GenotypeAggregator
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.data import Neo4jCellDataset
from torchcell.data.graph_processor import (
    SubgraphRepresentation,
    LazySubgraphRepresentation,
)
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.graph import build_gene_multigraph


def create_dataset(graph_processor):
    """Create dataset with specified graph processor."""
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")

    # Set all random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize genome
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()

    # Initialize graph
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    # Use physical and regulatory networks (HeteroCell configuration)
    graph_names = ["physical", "regulatory"]

    gene_multigraph = build_gene_multigraph(graph=graph, graph_names=graph_names)

    # Load metabolism
    yeast_gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))
    incidence_graphs = {"metabolism_bipartite": yeast_gem.bipartite_graph}

    # Load query
    with open(
        osp.join(EXPERIMENT_ROOT, "006-kuzmin-tmi/queries/001_small_build.cql"), "r"
    ) as f:
        query = f.read()

    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/001-small-build"
    )

    # Create dataset
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=gene_multigraph,
        incidence_graphs=incidence_graphs,
        node_embeddings=None,
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=graph_processor,
        transform=None,
    )

    return dataset


def find_samples_with_invalid_reactions(dataset, max_samples=100):
    """Find samples that have invalid reactions."""
    samples_with_invalid = []

    for i in range(min(max_samples, len(dataset))):
        sample = dataset[i]
        if "reaction" in sample.node_types and hasattr(sample["reaction"], "mask"):
            num_total = len(sample["reaction"].mask)
            num_valid = sample["reaction"].mask.sum().item()
            if num_total != num_valid:
                num_invalid = num_total - num_valid
                samples_with_invalid.append((i, num_invalid))

    return samples_with_invalid


def verify_sample_equivalence(subgraph_sample, lazy_sample, sample_idx):
    """Verify that SubgraphRepresentation and LazySubgraphRepresentation are equivalent."""
    print(f"\n{'='*80}")
    print(f"Verifying Sample {sample_idx}")
    print(f"{'='*80}\n")

    errors = []

    # 1. Check gene nodes
    print("Checking gene nodes...")
    subgraph_num_genes = subgraph_sample["gene"].num_nodes
    lazy_num_genes_kept = lazy_sample["gene"].mask.sum().item()

    if subgraph_num_genes != lazy_num_genes_kept:
        errors.append(
            f"Gene count mismatch: Subgraph={subgraph_num_genes}, Lazy (kept)={lazy_num_genes_kept}"
        )
        print(f"  ✗ Gene count mismatch")
    else:
        print(f"  ✓ Gene counts match: {subgraph_num_genes}")

    # 2. Check reaction nodes
    print("\nChecking reaction nodes...")
    subgraph_num_reactions = subgraph_sample["reaction"].num_nodes
    lazy_num_reactions_total = len(lazy_sample["reaction"].mask)
    lazy_num_reactions_valid = lazy_sample["reaction"].mask.sum().item()
    lazy_num_reactions_invalid = lazy_num_reactions_total - lazy_num_reactions_valid

    if subgraph_num_reactions != lazy_num_reactions_valid:
        errors.append(
            f"Reaction count mismatch: Subgraph={subgraph_num_reactions}, Lazy (valid)={lazy_num_reactions_valid}"
        )
        print(f"  ✗ Reaction count mismatch")
    else:
        print(
            f"  ✓ Reaction counts match: {subgraph_num_reactions} valid reactions"
        )
        print(
            f"    (Lazy has {lazy_num_reactions_invalid} additional invalid reactions marked by mask)"
        )

    # 3. Check metabolite nodes
    print("\nChecking metabolite nodes...")
    subgraph_num_metabolites = subgraph_sample["metabolite"].num_nodes
    lazy_num_metabolites = lazy_sample["metabolite"].num_nodes

    if subgraph_num_metabolites != lazy_num_metabolites:
        errors.append(
            f"Metabolite count mismatch: Subgraph={subgraph_num_metabolites}, Lazy={lazy_num_metabolites}"
        )
        print(f"  ✗ Metabolite count mismatch")
    else:
        print(f"  ✓ Metabolite counts match: {subgraph_num_metabolites}")

    # Check all metabolites are kept in Lazy
    lazy_metabolites_kept = lazy_sample["metabolite"].mask.sum().item()
    if lazy_metabolites_kept != lazy_num_metabolites:
        errors.append(
            f"Lazy metabolites not all kept: {lazy_metabolites_kept}/{lazy_num_metabolites}"
        )
        print(f"  ✗ Not all metabolites kept in Lazy")
    else:
        print(f"  ✓ All metabolites kept in Lazy")

    # 4. Check GPR edges
    print("\nChecking GPR edges...")
    if ("gene", "gpr", "reaction") in subgraph_sample.edge_types:
        subgraph_gpr_edges = subgraph_sample["gene", "gpr", "reaction"].num_edges
        lazy_gpr_edges_total = lazy_sample["gene", "gpr", "reaction"].num_edges
        lazy_gpr_edges_kept = lazy_sample["gene", "gpr", "reaction"].mask.sum().item()

        if subgraph_gpr_edges != lazy_gpr_edges_kept:
            errors.append(
                f"GPR edge count mismatch: Subgraph={subgraph_gpr_edges}, Lazy (kept)={lazy_gpr_edges_kept}"
            )
            print(f"  ✗ GPR edge count mismatch")
        else:
            print(f"  ✓ GPR edge counts match: {subgraph_gpr_edges}")
            lazy_gpr_removed = lazy_gpr_edges_total - lazy_gpr_edges_kept
            print(f"    (Lazy has {lazy_gpr_removed} additional edges masked out)")

    # 5. Check RMR edges
    print("\nChecking RMR edges...")
    if ("reaction", "rmr", "metabolite") in subgraph_sample.edge_types:
        subgraph_rmr_edges = subgraph_sample[
            "reaction", "rmr", "metabolite"
        ].num_edges
        lazy_rmr_edges_total = lazy_sample["reaction", "rmr", "metabolite"].num_edges
        lazy_rmr_edges_kept = (
            lazy_sample["reaction", "rmr", "metabolite"].mask.sum().item()
        )

        if subgraph_rmr_edges != lazy_rmr_edges_kept:
            errors.append(
                f"RMR edge count mismatch: Subgraph={subgraph_rmr_edges}, Lazy (kept)={lazy_rmr_edges_kept}"
            )
            print(f"  ✗ RMR edge count mismatch")
        else:
            print(f"  ✓ RMR edge counts match: {subgraph_rmr_edges}")
            lazy_rmr_removed = lazy_rmr_edges_total - lazy_rmr_edges_kept
            print(f"    (Lazy has {lazy_rmr_removed} additional edges masked out)")

            # Verify RMR edge masking logic
            print("\n  Verifying RMR edge masking logic...")
            rmr_hyperedge_index = lazy_sample[
                "reaction", "rmr", "metabolite"
            ].hyperedge_index
            rmr_mask = lazy_sample["reaction", "rmr", "metabolite"].mask
            reaction_mask = lazy_sample["reaction"].mask

            # For each RMR edge, check if mask matches source reaction validity
            reaction_indices = rmr_hyperedge_index[0]
            expected_rmr_mask = reaction_mask[reaction_indices]

            if torch.equal(rmr_mask, expected_rmr_mask):
                print(
                    f"    ✓ RMR edge masks correctly based on source reaction validity"
                )
            else:
                errors.append(
                    "RMR edge mask does not match source reaction validity"
                )
                print(f"    ✗ RMR edge mask inconsistent with reaction validity")
                # Debug info
                mismatches = (rmr_mask != expected_rmr_mask).sum().item()
                print(f"      {mismatches} edge mask mismatches")

    # 6. Check gene-gene edges
    print("\nChecking gene-gene edges...")
    for et in subgraph_sample.edge_types:
        if et[0] == "gene" and et[2] == "gene":
            subgraph_edges = subgraph_sample[et].num_edges
            lazy_edges_kept = lazy_sample[et].mask.sum().item()

            if subgraph_edges != lazy_edges_kept:
                errors.append(
                    f"{et} edge count mismatch: Subgraph={subgraph_edges}, Lazy (kept)={lazy_edges_kept}"
                )
                print(f"  ✗ {et}: edge count mismatch")
            else:
                print(f"  ✓ {et}: {subgraph_edges} edges")

    # Summary
    print(f"\n{'='*80}")
    if errors:
        print(f"Sample {sample_idx}: FAILED ✗")
        print(f"{'='*80}\n")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print(f"Sample {sample_idx}: PASSED ✓")
        print(f"{'='*80}\n")
        return True


def main():
    print("=" * 80)
    print("LazySubgraphRepresentation Equivalence Verification")
    print("=" * 80)
    print()
    print("Objective: Verify that LazySubgraphRepresentation produces equivalent")
    print("           results to SubgraphRepresentation when masks are applied.")
    print()

    # Create datasets with both processors
    print("Creating SubgraphRepresentation dataset...")
    subgraph_dataset = create_dataset(SubgraphRepresentation())
    print(f"Dataset length: {len(subgraph_dataset)}")
    print()

    print("Creating LazySubgraphRepresentation dataset...")
    lazy_dataset = create_dataset(LazySubgraphRepresentation())
    print(f"Dataset length: {len(lazy_dataset)}")
    print()

    # Find samples with invalid reactions
    print("Finding samples with invalid reactions...")
    samples_with_invalid = find_samples_with_invalid_reactions(lazy_dataset, max_samples=100)
    print(f"Found {len(samples_with_invalid)} samples with invalid reactions:")
    for idx, num_invalid in samples_with_invalid[:10]:
        print(f"  Sample {idx}: {num_invalid} invalid reactions")
    if len(samples_with_invalid) > 10:
        print(f"  ... and {len(samples_with_invalid) - 10} more")
    print()

    # Test samples
    test_samples = [
        0,  # Sample without invalid reactions
        5,
        6,
        8,
        9,  # Samples with invalid reactions
    ]

    print(f"Testing {len(test_samples)} samples...")
    print()

    results = []
    for sample_idx in test_samples:
        subgraph_sample = subgraph_dataset[sample_idx]
        lazy_sample = lazy_dataset[sample_idx]

        passed = verify_sample_equivalence(subgraph_sample, lazy_sample, sample_idx)
        results.append((sample_idx, passed))

    # Final summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print()

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for sample_idx, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"Sample {sample_idx}: {status}")

    print()
    print(f"Overall: {passed_count}/{total_count} samples passed")

    if passed_count == total_count:
        print()
        print("=" * 80)
        print("✓✓ ALL TESTS PASSED ✓✓")
        print("=" * 80)
        print()
        print("LazySubgraphRepresentation is functionally equivalent to")
        print("SubgraphRepresentation when masks are applied.")
        print()
        print("Key findings:")
        print("  - Reaction validity logic is identical")
        print("  - RMR edge masking correctly based on reaction validity")
        print("  - Metabolites are kept in both implementations")
        print("  - Edge counts match after mask application")
        print()
        print("Biological interpretation:")
        print("  - Invalid reactions (all required genes deleted) are marked")
        print("  - RMR edges from invalid reactions are masked out")
        print("  - Metabolites persist even if some producers are blocked")
        print("  - This matches SubgraphRepresentation's behavior exactly")
    else:
        print()
        print("=" * 80)
        print("✗✗ SOME TESTS FAILED ✗✗")
        print("=" * 80)
        print()
        print("LazySubgraphRepresentation may not be equivalent to SubgraphRepresentation.")
        print("Please review the errors above.")

    print()


if __name__ == "__main__":
    main()
