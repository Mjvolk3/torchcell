# experiments/005-kuzmin2018-tmi/scripts/dcell_batch_005_verify_mutant_state
# [[experiments.005-kuzmin2018-tmi.scripts.dcell_batch_005_verify_mutant_state]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/005-kuzmin2018-tmi/scripts/dcell_batch_005_verify_mutant_state
# Test file: experiments/005-kuzmin2018-tmi/scripts/test_dcell_batch_005_verify_mutant_state.py

import torch
import os
import os.path as osp
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
# import hypernetx as hnx (removed visualization dependency)
import networkx as nx
from collections import defaultdict
from torchcell.scratch.load_batch_005 import load_sample_data_batch
from torchcell.timestamp import timestamp

# Load environment variables to get ASSET_IMAGES_DIR
load_dotenv()


def verify_mutant_state_differences(batch_size=32):
    # Load a batch with the dcell configuration
    print(f"Loading DCell batch with size {batch_size}...")
    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
        batch_size=batch_size, num_workers=4, config="dcell", is_dense=False
    )

    # Get mutant states from the batch
    mutant_states = batch["gene_ontology"].mutant_state

    print(f"Batch size: {batch_size}")
    print(f"Shape of mutant_state tensor: {mutant_states.shape}")

    # Get the batch indices for mutant states
    mutant_state_batch = batch["gene_ontology"].mutant_state_batch

    # Verify each sample has unique perturbations
    print("\nVerifying unique perturbations across samples:")

    # Get perturbed gene indices for each sample
    samples_perturbed_genes = {}

    # Count zero states per batch sample
    zero_states_count = {}

    # Get unique samples in the batch
    unique_samples = torch.unique(mutant_state_batch).tolist()
    print(f"Number of unique samples in batch: {len(unique_samples)}")

    for sample_idx in unique_samples:
        # Get mutant states for this sample
        sample_mask = mutant_state_batch == sample_idx
        sample_states = mutant_states[sample_mask]

        # Find perturbed genes (state value = 0)
        perturbed_mask = sample_states[:, 2] == 0
        perturbed_indices = sample_states[perturbed_mask]

        # Get the gene indices (column 1) of perturbed genes
        perturbed_genes = perturbed_indices[:, 1].unique().tolist()

        # Store results
        samples_perturbed_genes[sample_idx] = perturbed_genes
        zero_states_count[sample_idx] = perturbed_mask.sum().item()

        # Print results for this sample
        print(f"\nSample {sample_idx}:")
        print(f"  Total GO-gene mappings: {sample_states.shape[0]}")
        print(f"  Perturbed GO-gene mappings (state=0): {zero_states_count[sample_idx]}")
        print(f"  Unique perturbed gene indices: {len(perturbed_genes)}")
        if len(perturbed_genes) > 0:
            print(
                f"  Perturbed gene indices: {perturbed_genes[:5]}"
                + ("..." if len(perturbed_genes) > 5 else "")
            )

    # Check for uniqueness across samples
    unique_perturbation_sets = set()
    for sample_idx, perturbed_genes in samples_perturbed_genes.items():
        unique_perturbation_sets.add(tuple(sorted(perturbed_genes)))

    print(f"\nNumber of unique perturbation patterns: {len(unique_perturbation_sets)}")
    print(
        f"All samples have unique perturbation patterns: {len(unique_perturbation_sets) == len(unique_samples)}"
    )

    # Visualize perturbation distribution
    counts = list(zero_states_count.values())
    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=20, color='steelblue', edgecolor='black', alpha=0.8)
    plt.xlabel("Number of perturbed GO-gene mappings")
    plt.ylabel("Frequency")
    plt.title("Distribution of Perturbation Counts Across Samples")
    plt.grid(True, alpha=0.3)

    # Add mean and median lines
    mean_count = np.mean(counts)
    median_count = np.median(counts)
    plt.axvline(mean_count, color='red', linestyle='dashed', linewidth=1,
                label=f'Mean: {mean_count:.1f}')
    plt.axvline(median_count, color='green', linestyle='dashed', linewidth=1,
                label=f'Median: {median_count:.1f}')
    plt.legend()

    # Save figure properly using ASSET_IMAGES_DIR and timestamp
    title = "dcell_batch_005_perturbation_distribution"
    save_path = osp.join(os.environ["ASSET_IMAGES_DIR"], f"{title}_{timestamp()}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved perturbation distribution to {save_path}")
    plt.close()

    # Compute pairwise similarity between samples
    sample_indices = list(samples_perturbed_genes.keys())
    n_samples = len(sample_indices)
    similarity_matrix = np.zeros((n_samples, n_samples))

    for i, idx1 in enumerate(sample_indices):
        genes1 = set(samples_perturbed_genes[idx1])
        for j, idx2 in enumerate(sample_indices):
            if i <= j:
                genes2 = set(samples_perturbed_genes[idx2])

                # Calculate Jaccard similarity if both sets are non-empty
                if genes1 and genes2:
                    similarity = len(genes1.intersection(genes2)) / len(
                        genes1.union(genes2)
                    )
                # If both are empty, they're identical
                elif not genes1 and not genes2:
                    similarity = 1.0
                # If one is empty and one is not, they're completely different
                else:
                    similarity = 0.0

                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

    # Plot similarity matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Jaccard Similarity")
    plt.title("Perturbation Pattern Similarity Between Samples")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")

    # Save figure properly using ASSET_IMAGES_DIR and timestamp
    title = "dcell_batch_005_perturbation_similarity"
    save_path = osp.join(os.environ["ASSET_IMAGES_DIR"], f"{title}_{timestamp()}.png")
    plt.savefig(save_path)
    print(f"Saved perturbation similarity matrix to {save_path}")
    plt.close()

    return samples_perturbed_genes, similarity_matrix


# Visualization functions removed to simplify script


if __name__ == "__main__":
    perturbed_genes, similarity = verify_mutant_state_differences(batch_size=32)

    # Print summary statistics
    print("\nSummary Statistics:")
    num_empty = sum(1 for genes in perturbed_genes.values() if len(genes) == 0)
    print(f"Samples with no perturbations: {num_empty}")

    perturbation_counts = [len(genes) for genes in perturbed_genes.values()]
    if perturbation_counts:
        print(f"Min genes perturbed: {min(perturbation_counts)}")
        print(f"Max genes perturbed: {max(perturbation_counts)}")
        print(
            f"Mean genes perturbed: {sum(perturbation_counts)/len(perturbation_counts):.2f}"
        )

    # Print similarity stats (excluding self-similarity)
    sim_values = similarity.flatten()
    sim_values = sim_values[sim_values < 0.999]  # Exclude diagonal (self-similarity)
    if len(sim_values) > 0:
        print(f"Max similarity between different samples: {sim_values.max():.4f}")
        print(f"Mean similarity between different samples: {sim_values.mean():.4f}")

    print("\nAnalysis complete.")

