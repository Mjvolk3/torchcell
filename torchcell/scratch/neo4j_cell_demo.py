# torchcell/scratch/neo4j_cell_demo
# [[torchcell.scratch.neo4j_cell_demo]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/scratch/neo4j_cell_demo
"""Build and inspect a Neo4jCellDataset for demonstration.

Five demos: ``main`` builds the 003-fit-int small-build dataset and pulls one
training batch, ``main_incidence`` does the same with incidence (hypergraph)
graphs, ``main_transform_standardization`` tests label standardization with the
metabolic network, and ``main_transform_categorical`` and
``main_transform_categorical_dense`` test the label binning transforms (the
latter with dense conversion on a perturbation subset). The ``__main__`` block
runs ``main_transform_standardization``; the others are left as commented
alternatives, exactly as in the original.

Moved verbatim from torchcell/data/neo4j_cell.py on 2026-10-06 (test campaign
Phase 23) because it is a demo and not library code.

Run from the repo root (the query path
``experiments/003-fit-int/queries/001-small-build.cql`` is relative to it):

    PYTHONPATH=$PWD python torchcell/scratch/neo4j_cell_demo.py

Needs ``DATA_ROOT`` in ``.env`` with the SGD genome, GO, STRING, TFLink and
codon-frequency embedding data under it, and the
``data/torchcell/experiments/003-fit-int/001-small-build`` dataset built (or a
reachable Neo4j to build it from the query). The plotting demos open matplotlib
windows or write PNGs to the working directory. No GPU.
"""

import json
import os
from typing import Any, cast

from tqdm import tqdm

from torchcell.data.neo4j_cell import Neo4jCellDataset, _print_label_stats
from torchcell.graph import GeneMultiGraph
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome


def main() -> None:
    """Build and inspect a Neo4jCellDataset for demonstration."""
    # genome
    import os.path as osp

    from dotenv import load_dotenv

    from torchcell.data import GenotypeAggregator, MeanExperimentDeduplicator
    from torchcell.data.graph_processor import SubgraphRepresentation
    from torchcell.datamodels.fitness_composite_conversion import (
        CompositeFitnessConverter,
    )
    from torchcell.datamodules import CellDataModule
    from torchcell.datasets.fungal_up_down_transformer import (
        FungalUpDownTransformerDataset,
    )
    from torchcell.graph import SCerevisiaeGraph

    load_dotenv()
    DATA_ROOT = cast(str, os.getenv("DATA_ROOT"))

    with open("experiments/003-fit-int/queries/001-small-build.cql") as f:
        query = f.read()

    ### Add Embeddings
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_chrmt()
    genome.drop_empty_go()

    with open("gene_set.json", "w") as f:
        json.dump(list(genome.gene_set), f)

    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    fudt_3prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )
    fudt_5prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=cast(
            GeneMultiGraph,
            {"physical": graph.G_physical, "regulatory": graph.G_regulatory},
        ),
        node_embeddings={
            "fudt_3prime": fudt_3prime_dataset,
            "fudt_5prime": fudt_5prime_dataset,
        },
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )
    print(len(dataset))
    dataset.label_df
    # Data module testing

    # print(dataset[7])
    print(dataset[3])
    dataset.close_lmdb()
    # print(dataset[10000])

    # Assuming you have already created your dataset and CellDataModule
    cell_data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=2,
        random_seed=42,
        num_workers=2,
        pin_memory=False,
    )
    cell_data_module.setup()

    for batch in tqdm(cell_data_module.train_dataloader()):
        print(batch)
        break

    # for i in tqdm(
    #     range(len(cell_data_module.index_details.train.perturbation_count_index[1].indices))
    # ):
    #     single_pert_index = (
    #         cell_data_module.index_details.train.perturbation_count_index[1].indices[i]
    #     )
    #     if len(dataset[single_pert_index]["gene"].ids_pert) != 1:
    #         train_not_single_pert.append(single_pert_index)

    # print("len train_not_single_pert", len(train_not_single_pert))

    # # Now, instantiate the updated PerturbationSubsetDataModule
    # size = 1e4
    # seed = 42
    # perturbation_subset_data_module = PerturbationSubsetDataModule(
    #     cell_data_module=cell_data_module,
    #     size=int(size),
    #     batch_size=2,
    #     num_workers=2,
    #     pin_memory=True,
    #     prefetch=False,
    #     seed=seed,
    # )

    # # Set up the data module
    # perturbation_subset_data_module.setup()

    # # Use the data loaders
    # for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
    #     # Your training code here
    #     break


def main_incidence() -> None:
    """Build and inspect a dataset with incidence (hypergraph) graphs."""
    # genome
    import os.path as osp

    from dotenv import load_dotenv

    from torchcell.data import GenotypeAggregator, MeanExperimentDeduplicator
    from torchcell.data.graph_processor import SubgraphRepresentation
    from torchcell.datamodels.fitness_composite_conversion import (
        CompositeFitnessConverter,
    )
    from torchcell.datamodules import CellDataModule
    from torchcell.datasets import CodonFrequencyDataset
    from torchcell.datasets.fungal_up_down_transformer import (
        FungalUpDownTransformerDataset,
    )
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.metabolism.yeast_GEM import YeastGEM

    load_dotenv()
    DATA_ROOT = cast(str, os.getenv("DATA_ROOT"))

    with open("experiments/003-fit-int/queries/001-small-build.cql") as f:
        query = f.read()

    ### Add Embeddings
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_chrmt()
    genome.drop_empty_go()

    with open("gene_set.json", "w") as f:
        json.dump(list(genome.gene_set), f)

    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    codon_frequency = CodonFrequencyDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/codon_frequency_embedding"),
        genome=genome,
    )
    fudt_3prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )
    fudt_5prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )

    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=cast(
            GeneMultiGraph,
            {"physical": graph.G_physical, "regulatory": graph.G_regulatory},
        ),
        incidence_graphs={"metabolism": YeastGEM().reaction_map},
        node_embeddings={
            "codon_frequency": codon_frequency,
            "fudt_3prime": fudt_3prime_dataset,
            "fudt_5prime": fudt_5prime_dataset,
        },
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )
    print(len(dataset))
    dataset.label_df
    # Data module testing

    # print(dataset[7])
    print(dataset.cell_graph)
    print(dataset[4])

    # Get perturbed gene indices
    print("first few perturbed gene indices with modified metabolism graph:")
    [
        print(
            dataset[i]["metabolite", "reaction", "metabolite"].hyperedge_index.size()
            != dataset.cell_graph[
                "metabolite", "reaction", "metabolite"
            ].hyperedge_index.size()
        )
        for i in range(10)
    ]

    perturbed_indices = dataset[4]["gene"].cell_graph_idx_pert

    # Check which reactions contained these genes in the original graph
    reactions_with_perturbed = set()
    for rxn_idx, genes in dataset.cell_graph[
        "metabolite", "reaction", "metabolite"
    ].reaction_to_genes_indices.items():
        if any(g in perturbed_indices for g in genes):
            reactions_with_perturbed.add(rxn_idx)

    print(
        f"Number of reactions containing perturbed genes: {len(reactions_with_perturbed)}"
    )

    dataset.close_lmdb()
    # print(dataset[10000])

    # Assuming you have already created your dataset and CellDataModule
    cell_data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=2,
        random_seed=42,
        num_workers=2,
        pin_memory=False,
    )
    cell_data_module.setup()

    for batch in tqdm(cell_data_module.train_dataloader()):
        break


def main_transform_standardization() -> None:
    """Test standardization of labels using LabelNormalizationTransform with metabolic network."""
    import os.path as osp

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import torch
    from dotenv import load_dotenv
    from torch_geometric.transforms import Compose
    from tqdm import tqdm

    from torchcell.data import GenotypeAggregator, MeanExperimentDeduplicator
    from torchcell.data.graph_processor import SubgraphRepresentation
    from torchcell.data.neo4j_cell import Neo4jCellDataset
    from torchcell.datamodels.fitness_composite_conversion import (
        CompositeFitnessConverter,
    )
    from torchcell.datamodules import CellDataModule
    from torchcell.datasets import CodonFrequencyDataset
    from torchcell.datasets.fungal_up_down_transformer import (
        FungalUpDownTransformerDataset,
    )

    # Import necessary components
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.metabolism.yeast_GEM import YeastGEM
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
    from torchcell.transforms.hetero_to_dense_mask import HeteroToDenseMask
    from torchcell.transforms.regression_to_classification import (
        LabelNormalizationTransform,
    )

    # Load environment variables
    load_dotenv()
    DATA_ROOT = cast(str, os.getenv("DATA_ROOT"))

    # Load query
    with open("experiments/003-fit-int/queries/001-small-build.cql") as f:
        query = f.read()

    # Set up genome and graph
    print("Setting up genome and graph...")
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_chrmt()
    genome.drop_empty_go()

    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    # Set up node embeddings
    print("Setting up embeddings...")
    codon_frequency = CodonFrequencyDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/codon_frequency_embedding"),
        genome=genome,
    )
    fudt_3prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )
    fudt_5prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )

    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )

    # Create dataset with metabolism network
    print("Creating dataset with metabolism network...")
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=cast(
            GeneMultiGraph,
            {"physical": graph.G_physical, "regulatory": graph.G_regulatory},
        ),
        # incidence_graphs={"metabolism_hypergraph": YeastGEM().reaction_map},
        incidence_graphs={"metabolism_bipartite": YeastGEM().bipartite_graph},
        node_embeddings={
            "codon_frequency": codon_frequency,
            "fudt_3prime": fudt_3prime_dataset,
            "fudt_5prime": fudt_5prime_dataset,
        },
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )

    print(f"Dataset size: {len(dataset)}")
    dataset
    # Define the labels we want to standardize
    labels = ["fitness", "gene_interaction"]

    # Print statistics of original data using dataset.label_df
    _print_label_stats(dataset.label_df, labels)

    # Configure normalization to use standard (z-score) normalization for both labels
    norm_configs = {
        "fitness": {"strategy": "standard"},  # z-score: (x - mean) / std
        "gene_interaction": {"strategy": "standard"},  # z-score: (x - mean) / std
    }

    # Create the normalizer
    print("\nCreating normalization transform...")
    normalizer = LabelNormalizationTransform(dataset, norm_configs)

    # Print the normalization parameters
    for label, stats in normalizer.stats.items():
        print(f"\nNormalization parameters for {label}:")
        for key, value in stats.items():
            if key not in ["bin_edges", "bin_counts", "strategy"]:
                print(f"  {key}: {value:.6f}")
        print(f"  strategy: {stats['strategy']}")

    # Apply the transform to the dataset
    # HACK - start
    dense_transform = HeteroToDenseMask({"gene": len(genome.gene_set)})
    dataset.transform = Compose([normalizer, dense_transform])
    # HACK - end
    # dataset.transform = normalizer

    # Check metabolic network connectivity changes
    print("\nChecking metabolic network changes for a few examples...")
    for i in range(5):
        data = dataset[i]
        if "metabolite" in data.node_types and "reaction" in data.node_types:
            if hasattr(data["metabolite", "reaction", "metabolite"], "hyperedge_index"):
                print(
                    f"Sample {i} - hyperedge size: {data['metabolite', 'reaction', 'metabolite'].hyperedge_index.size()}"
                )

                # Check if perturbed genes affect the metabolic network
                if hasattr(data["gene"], "cell_graph_idx_pert"):
                    perturbed_indices = data["gene"].cell_graph_idx_pert

                    # Count reactions affected by gene perturbations
                    reactions_with_perturbed = set()
                    for rxn_idx, genes in dataset.cell_graph[
                        "metabolite", "reaction", "metabolite"
                    ].reaction_to_genes_indices.items():
                        if any(g in perturbed_indices for g in genes):
                            reactions_with_perturbed.add(rxn_idx)

                    print(f"  Perturbed genes: {len(perturbed_indices)}")
                    print(f"  Reactions affected: {len(reactions_with_perturbed)}")

    # Sample data points for visualization
    print("\nSampling data for visualization...")
    num_samples = 1000
    sample_indices = np.random.choice(
        len(dataset), min(num_samples, len(dataset)), replace=False
    )

    original_values: dict[str, list[float]] = {label: [] for label in labels}
    normalized_values: dict[str, list[float]] = {label: [] for label in labels}

    for idx in tqdm(sample_indices):
        data = dataset[idx]
        for label in labels:
            if label in data["gene"] and not torch.isnan(data["gene"][label]).any():
                # Get normalized value
                normalized_values[label].append(data["gene"][label].item())
                # Get original value
                original_values[label].append(data["gene"][f"{label}_original"].item())

    # Create visualization with plots
    print("\nCreating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Label Distributions: Before vs. After Standardization", fontsize=16)

    for i, label in enumerate(labels):
        # Original distribution
        sns.histplot(original_values[label], bins=50, ax=axes[0, i], kde=True)
        axes[0, i].set_title(f"Original {label}")

        # Normalized distribution
        sns.histplot(normalized_values[label], bins=50, ax=axes[1, i], kde=True)
        axes[1, i].set_title(f"Standardized {label} (z-score)")

        # Add mean and std lines to the normalized plot
        axes[1, i].axvline(x=0, color="r", linestyle="--", label="Mean (0)")
        axes[1, i].axvline(x=1, color="g", linestyle="--", label="+1 Std")
        axes[1, i].axvline(x=-1, color="g", linestyle="--", label="-1 Std")
        axes[1, i].legend()

    plt.tight_layout()
    plt.savefig("standardization_with_metabolism_comparison.png")

    # Test with datamodule
    print("\nTesting with CellDataModule...")
    cell_data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=2,
        random_seed=42,
        num_workers=2,
        pin_memory=False,
    )
    cell_data_module.setup()

    # Get a batch and check normalized values
    for batch in cell_data_module.train_dataloader():
        for label in labels:
            if label in batch["gene"]:
                print(f"\nBatch {label} statistics:")
                print(f"  Shape: {batch['gene'][label].shape}")
                print(f"  Mean: {batch['gene'][label].mean().item():.4f}")
                print(f"  Std: {batch['gene'][label].std().item():.4f}")
                print(f"  Original values present: {'_original' in batch['gene']}")
        break

    # Clean up
    dataset.close_lmdb()
    print("\nTest completed successfully.")


def main_transform_categorical() -> None:
    # Used this in hetero gnn pool when converting categorical to regression
    """Test the label binning transforms on the dataset with proper initialization."""
    import os.path as osp

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import torch
    from dotenv import load_dotenv
    from torch_geometric.transforms import Compose

    from torchcell.data import GenotypeAggregator, MeanExperimentDeduplicator
    from torchcell.data.neo4j_cell import (  # type: ignore[attr-defined]  # __main__-only; SubgraphRepresentation actually lives in graph_processor (pre-existing wrong import path)
        Neo4jCellDataset,
        SubgraphRepresentation,
    )
    from torchcell.datamodels.fitness_composite_conversion import (
        CompositeFitnessConverter,
    )
    from torchcell.datasets.fungal_up_down_transformer import (
        FungalUpDownTransformerDataset,
    )
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
    from torchcell.transforms.regression_to_classification import (
        LabelBinningTransform,
        LabelNormalizationTransform,
    )

    # Dataset setup code unchanged...
    load_dotenv()
    DATA_ROOT = cast(str, os.getenv("DATA_ROOT"))

    with open("experiments/003-fit-int/queries/001-small-build.cql") as f:
        query = f.read()

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_chrmt()
    genome.drop_empty_go()

    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    fudt_3prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )
    fudt_5prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )

    graphs = cast(
        GeneMultiGraph, {"physical": graph.G_physical, "regulatory": graph.G_regulatory}
    )
    node_embeddings = {
        "fudt_3prime": fudt_3prime_dataset,
        "fudt_5prime": fudt_5prime_dataset,
    }

    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )

    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=graphs,
        node_embeddings=node_embeddings,
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )

    # Configure transforms
    norm_configs = {
        "fitness": {"strategy": "minmax"},
        "gene_interaction": {"strategy": "minmax"},
    }

    bin_configs = {
        "fitness": {
            "num_bins": 10,
            "strategy": "equal_width",
            "store_continuous": True,
            "sigma": 0.1,
            "label_type": "soft",
        },
        "gene_interaction": {
            "num_bins": 5,
            "strategy": "equal_frequency",
            "store_continuous": True,
            "label_type": "ordinal",
        },
    }

    # Create transforms and compose them
    normalize_transform = LabelNormalizationTransform(dataset, norm_configs)
    binning_transform = LabelBinningTransform(dataset, bin_configs)
    transform = Compose([normalize_transform, binning_transform])

    # Apply transform to dataset
    dataset.transform = transform

    # Test transforms
    test_indices = [10, 100, 1000]
    print("\nTesting transforms...")
    for idx in test_indices:
        print(f"\nSample {idx}:")
        data = dataset[idx]  # This will apply the composed transform

        # Get original data (without transform)
        dataset.transform = None
        original_data = dataset[idx]

        # Restore transform
        dataset.transform = transform

        # Print results for each label
        for label in norm_configs.keys():
            print(f"\n{label}:")
            print(f"Original:     {original_data['gene'][label].item():.4f}")
            print(f"Normalized:   {data['gene'][f'{label}_continuous'].item():.4f}")
            print(f"Original (stored): {data['gene'][f'{label}_original'].item():.4f}")

            if bin_configs[label]["label_type"] == "soft":
                print(f"Soft labels shape: {data['gene'][label].shape}")
                print(f"Soft labels sum:   {data['gene'][label].sum().item():.4f}")
            elif bin_configs[label]["label_type"] == "ordinal":
                print(f"Ordinal labels shape: {data['gene'][label].shape}")
                print(f"Ordinal values:      {data['gene'][label].numpy()}")

    # Visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle("Label Distributions: Original, Normalized, and Binned", fontsize=16)

    # Sample data for visualization
    sample_size = min(1000, len(dataset))
    sample_indices = np.random.choice(len(dataset), sample_size, replace=False)

    # Get original distribution data
    dataset.transform = None
    original_data = {label: [] for label in norm_configs.keys()}
    for idx in sample_indices:
        data = dataset[idx]
        for label in norm_configs.keys():
            if not torch.isnan(data["gene"][label]).any():
                original_data[label].append(data["gene"][label].item())

    # Restore transform and get transformed data
    dataset.transform = transform
    transformed_data: dict[str, dict[str, list[Any]]] = {
        label: {"normalized": [], "binned": []} for label in norm_configs.keys()
    }

    for idx in sample_indices:
        data = dataset[idx]
        for label in norm_configs.keys():
            if not torch.isnan(data["gene"][f"{label}_continuous"]).any():
                transformed_data[label]["normalized"].append(
                    data["gene"][f"{label}_continuous"].item()
                )
                transformed_data[label]["binned"].append(data["gene"][label].numpy())

    # Plot distributions
    for i, label in enumerate(norm_configs.keys()):
        # Original distribution
        sns.histplot(original_data[label], bins=50, ax=axes[0, i], stat="density")
        axes[0, i].set_title(f"Original {label}")

        # Normalized distribution
        sns.histplot(
            transformed_data[label]["normalized"],
            bins=50,
            ax=axes[1, i],
            stat="density",
        )
        axes[1, i].set_title(f"Normalized {label}")

        # Binned distribution
        binned = np.array(transformed_data[label]["binned"])
        if bin_configs[label]["label_type"] == "soft":
            mean_soft = np.mean(binned, axis=0)
            axes[2, i].bar(range(len(mean_soft)), mean_soft)
            axes[2, i].set_title(f"Mean Soft Labels {label}")
        else:
            mean_ordinal = np.mean(binned, axis=0)
            axes[2, i].bar(range(len(mean_ordinal)), mean_ordinal)
            axes[2, i].set_title(f"Mean Ordinal Values {label}")

    plt.tight_layout()
    plt.show()

    dataset.close_lmdb()


def main_transform_categorical_dense() -> None:
    """Test label transforms and dense conversion with perturbation subset."""
    import os.path as osp

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from dotenv import load_dotenv
    from torch_geometric.transforms import Compose

    from torchcell.data import GenotypeAggregator, MeanExperimentDeduplicator
    from torchcell.data.neo4j_cell import (  # type: ignore[attr-defined]  # __main__-only; SubgraphRepresentation actually lives in graph_processor (pre-existing wrong import path)
        Neo4jCellDataset,
        SubgraphRepresentation,
    )
    from torchcell.datamodels.fitness_composite_conversion import (
        CompositeFitnessConverter,
    )
    from torchcell.datamodules import CellDataModule
    from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
    from torchcell.datasets.fungal_up_down_transformer import (
        FungalUpDownTransformerDataset,
    )
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
    from torchcell.transforms.regression_to_classification import (
        InverseCompose,
        LabelBinningTransform,
        LabelNormalizationTransform,
    )

    # Setup dataset (unchanged)
    load_dotenv()
    DATA_ROOT = cast(str, os.getenv("DATA_ROOT"))

    with open("experiments/003-fit-int/queries/001-small-build.cql") as f:
        query = f.read()

    # Dataset setup
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_chrmt()
    genome.drop_empty_go()

    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    fudt_3prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )
    fudt_5prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )

    graphs = cast(
        GeneMultiGraph, {"physical": graph.G_physical, "regulatory": graph.G_regulatory}
    )
    node_embeddings = {
        "fudt_3prime": fudt_3prime_dataset,
        "fudt_5prime": fudt_5prime_dataset,
    }

    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )

    # First create dataset without transforms
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=graphs,
        node_embeddings=node_embeddings,
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )

    # Configure transforms
    norm_configs = {
        "fitness": {"strategy": "minmax"},
        "gene_interaction": {"strategy": "minmax"},
    }

    bin_configs = {
        "fitness": {
            "num_bins": 32,
            "strategy": "equal_frequency",
            "store_continuous": True,
            # "sigma": 0.1,
            "label_type": "soft",
        },
        "gene_interaction": {
            "num_bins": 32,
            "strategy": "equal_frequency",
            "store_continuous": True,
            # "sigma": 0.1,
            "label_type": "soft",
            # "num_bins": 5,
            # "strategy": "equal_frequency",
            # "store_continuous": True,
            # "label_type": "ordinal",
        },
    }

    # Create transforms and compose them with dataset stats
    normalize_transform = LabelNormalizationTransform(dataset, norm_configs)
    binning_transform = LabelBinningTransform(dataset, bin_configs, normalize_transform)
    # TODO will need to implement inverse maybe?
    # dense_transform = HeteroToDense({"gene": len(genome.gene_set)})

    # Apply transforms to dataset
    forward_transform = Compose([normalize_transform, binning_transform])
    _inverse_transform = InverseCompose(forward_transform)

    # I want to be able to do this
    dataset.transform = forward_transform

    # Create base data module
    base_data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=32,
        random_seed=42,
        num_workers=2,
        pin_memory=True,
    )
    base_data_module.setup()

    # Create perturbation subset module
    subset_size = 10000
    subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=base_data_module,
        size=subset_size,
        batch_size=2,
        num_workers=2,
        pin_memory=True,
        prefetch=False,
        seed=42,
        dense=True,  # Important for dense format
    )
    subset_data_module.setup()

    # Test transforms on subset
    print("\nTesting transforms on subset...")
    train_loader = subset_data_module.train_dataloader()
    batch = next(iter(train_loader))

    print("\nBatch structure:")
    print(f"Batch keys: {batch.keys}")
    print(f"\nNode features shape: {batch['gene'].x.shape}")
    print("Adjacency matrix shapes:")
    # print(f"Physical: {batch['gene', 'physical_interaction', 'gene'].adj.shape}")
    # print(f"Regulatory: {batch['gene', 'regulatory_interaction', 'gene'].adj.shape}")

    # Check label shapes and values
    print("\nLabel information:")
    for label in norm_configs.keys():
        print(f"\n{label}:")
        if bin_configs[label]["label_type"] == "soft":
            print(f"Soft label shape: {batch['gene'][label].shape}")
            # Handle potentially extra dimensions in soft labels
            soft_labels = batch["gene"][label].squeeze()
            if soft_labels.dim() == 3:  # If [batch, 1, num_classes]
                soft_labels = soft_labels.squeeze(1)
            print("Soft label sums (first 5):")
            print(soft_labels.sum(dim=-1)[:5])  # Sum over classes
        else:
            print(f"Ordinal label shape: {batch['gene'][label].shape}")
            ordinal_labels = batch["gene"][label].squeeze()
            if ordinal_labels.dim() == 3:  # If [batch, 1, num_thresholds]
                ordinal_labels = ordinal_labels.squeeze(1)
            print("Ordinal values (first 5):")
            print(ordinal_labels[:5])

        # Handle continuous and original values
        cont_vals = batch["gene"][f"{label}_continuous"].squeeze()
        orig_vals = batch["gene"][f"{label}_original"].squeeze()
        print(f"Continuous values (first 5): {cont_vals[:5]}")
        print(f"Original values (first 5): {orig_vals[:5]}")

    # Create visualization of distributions in batch
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Label Distributions in Dense Batch", fontsize=16)

    for i, label in enumerate(norm_configs.keys()):
        # Original values
        orig_values = batch["gene"][f"{label}_original"].squeeze().cpu().numpy()
        valid_mask = ~np.isnan(orig_values)
        orig_values = orig_values[valid_mask]
        sns.histplot(orig_values, bins=50, ax=axes[0, i], stat="density")
        axes[0, i].set_title(f"Original {label}")

        # Transformed values
        if bin_configs[label]["label_type"] == "soft":
            soft_labels = batch["gene"][label].squeeze()
            if soft_labels.dim() == 3:
                soft_labels = soft_labels.squeeze(1)
            mean_soft = soft_labels.mean(dim=0).cpu().numpy()
            axes[1, i].bar(range(len(mean_soft)), mean_soft)
            axes[1, i].set_title(f"Mean Soft Labels {label}")
            axes[1, i].set_xlabel("Class")
            axes[1, i].set_ylabel("Mean Probability")
        else:
            ordinal_labels = batch["gene"][label].squeeze()
            if ordinal_labels.dim() == 3:
                ordinal_labels = ordinal_labels.squeeze(1)
            mean_ordinal = ordinal_labels.mean(dim=0).cpu().numpy()
            axes[1, i].bar(range(len(mean_ordinal)), mean_ordinal)
            axes[1, i].set_title(f"Mean Ordinal Values {label}")
            axes[1, i].set_xlabel("Threshold")
            axes[1, i].set_ylabel("Mean Value")

    plt.tight_layout()
    plt.show()

    dataset.close_lmdb()


if __name__ == "__main__":
    # main_transform_categorical_dense()
    # main()
    # main_incidence()
    main_transform_standardization()
