#!/usr/bin/env python3
"""Generate baseline reference data BEFORE any optimizations."""

import os
import os.path as osp
import pickle
import json
from datetime import datetime
from torchcell.scratch.load_batch_005 import load_sample_data_batch


def save_baseline_reference():
    """Generate and save baseline reference data."""

    output_dir = "/scratch/projects/torchcell/data/tests/torchcell/scratch/load_batch_005"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(osp.join(output_dir, "profiling_results"), exist_ok=True)

    print("="*80)
    print("Generating Baseline Reference Data (BEFORE Optimizations)")
    print("="*80)

    print("\nLoading HeteroCell configuration...")
    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
        batch_size=2,
        num_workers=2,
        config="hetero_cell_bipartite",
        is_dense=False
    )

    single_instance = dataset[0]

    print("\n=== Single Instance Structure ===")
    print(single_instance)
    print("\n=== Batch Structure ===")
    print(batch)

    # Prepare reference data
    reference_data = {
        "single_instance": single_instance,
        "batch": batch,
        "metadata": {
            "dataset_length": len(dataset),
            "max_num_nodes": max_num_nodes,
            "input_channels": input_channels,
            "timestamp": datetime.now().isoformat(),
            "optimization_step": "baseline",

            "instance": {
                "gene_nodes": single_instance["gene"].num_nodes,
                "reaction_nodes": single_instance["reaction"].num_nodes,
                "metabolite_nodes": single_instance["metabolite"].num_nodes,
                "physical_edges": single_instance["gene", "physical", "gene"].num_edges,
                "regulatory_edges": single_instance["gene", "regulatory", "gene"].num_edges,
                "gpr_edges": single_instance["gene", "gpr", "reaction"].num_edges,
                "rmr_edges": single_instance["reaction", "rmr", "metabolite"].num_edges,
                "perturbed_genes": len(single_instance["gene"].ids_pert),
            },

            "batch": {
                "num_graphs": batch.num_graphs,
                "gene_nodes": batch["gene"].num_nodes,
                "reaction_nodes": batch["reaction"].num_nodes,
                "metabolite_nodes": batch["metabolite"].num_nodes,
                "physical_edges": batch["gene", "physical", "gene"].edge_index.size(1),
                "regulatory_edges": batch["gene", "regulatory", "gene"].edge_index.size(1),
                "gpr_edges": batch["gene", "gpr", "reaction"].hyperedge_index.size(1),
                "rmr_edges": batch["reaction", "rmr", "metabolite"].hyperedge_index.size(1),
            },
        }
    }

    # Save baseline
    output_file = osp.join(output_dir, "reference_baseline.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(reference_data, f)

    print(f"\n✓ Baseline reference saved: {output_file}")

    # Save metadata as JSON
    metadata_file = osp.join(output_dir, "metadata_baseline.json")
    with open(metadata_file, "w") as f:
        json.dump(reference_data["metadata"], f, indent=2)

    print(f"✓ Metadata saved: {metadata_file}")

    return output_file


if __name__ == "__main__":
    save_baseline_reference()
