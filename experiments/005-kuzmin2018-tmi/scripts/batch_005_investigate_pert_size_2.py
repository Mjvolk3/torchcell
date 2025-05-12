# experiments/005-kuzmin2018-tmi/scripts/batch_005_investigate_pert_size_2
# [[experiments.005-kuzmin2018-tmi.scripts.batch_005_investigate_pert_size_2]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/005-kuzmin2018-tmi/scripts/batch_005_investigate_pert_size_2
# Test file: experiments/005-kuzmin2018-tmi/scripts/test_batch_005_investigate_pert_size_2.py

import torch
from torchcell.scratch.load_batch_005 import load_sample_data_batch


def investigate_missing_perturbation():
    print("Loading DCell batch...")
    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
        batch_size=32, num_workers=4, config="dcell", is_dense=False
    )

    # Check the perturbation indices for each sample in the batch
    print("\nChecking raw perturbation indices:")
    batch_keys = [key for key in batch["gene"].keys() if "perturbation" in key]
    print(f"Available perturbation keys: {batch_keys}")

    # Get the perturbation indices
    perturbation_indices = batch["gene"].perturbation_indices
    perturbation_batch = batch["gene"].perturbation_indices_batch
    perturbation_ptr = batch["gene"].perturbation_indices_ptr

    # Print info for all samples
    for sample_idx in range(len(perturbation_ptr) - 1):
        start_idx = perturbation_ptr[sample_idx]
        end_idx = perturbation_ptr[sample_idx + 1]

        sample_perturbations = perturbation_indices[start_idx:end_idx]

        print(f"Sample {sample_idx} raw perturbation indices: {sample_perturbations}")

        # For sample 8, do additional investigation
        if sample_idx == 8:
            print("\n*** INVESTIGATING SAMPLE 8 ***")

            # Check if the gene has any GO annotations
            for gene_idx in sample_perturbations:
                # Get GO terms associated with this gene from the edge_index
                edge_index = batch["gene", "has_annotation", "gene_ontology"].edge_index
                gene_mask = edge_index[0] == gene_idx
                go_terms = edge_index[1, gene_mask]

                print(f"Gene {gene_idx}: Associated with {len(go_terms)} GO terms")

                # Check if this gene appears in the mutant state
                mutant_state = batch["gene_ontology"].mutant_state
                mutant_state_batch = batch["gene_ontology"].mutant_state_batch

                # Filter for sample 8's mutant state
                sample_mask = mutant_state_batch == sample_idx
                sample_states = mutant_state[sample_mask]

                # Check which genes are in the mutant state
                gene_present = (sample_states[:, 1] == gene_idx).any().item()

                print(f"Gene {gene_idx} present in mutant_state: {gene_present}")

                if not gene_present:
                    print(f"*** FOUND MISSING GENE: {gene_idx} ***")
                    print(f"This gene doesn't appear in the mutant_state for Sample 8!")

    return dataset, batch


if __name__ == "__main__":
    dataset, batch = investigate_missing_perturbation()
