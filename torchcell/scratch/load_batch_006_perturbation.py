"""
Load batch for CellGraphTransformer using Perturbation processor.
This is for experiment 006 with the new transformer architecture.
"""

import os
import os.path as osp
from dotenv import load_dotenv
import torch
from torchcell.data import Neo4jCellDataset
from torchcell.datamodules import CellDataModule
from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
from torchcell.data import MeanExperimentDeduplicator, GenotypeAggregator
from torchcell.data.graph_processor import Perturbation
from torchcell.graph import SCerevisiaeGraph
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.graph import build_gene_multigraph
from torchcell.transforms.coo_regression_to_classification import COOLabelNormalizationTransform


def load_perturbation_batch(
    batch_size: int = 128,
    num_workers: int = 4,
    subset_size: int = 10000,
    device: torch.device = torch.device("cuda"),
):
    """
    Load a batch of data using the Perturbation processor for CellGraphTransformer.

    Returns:
        dataset: The dataset with Perturbation processor
        batch: A batch of data
        cell_graph: The cell graph with gene structure
        gene_set_size: Number of genes in the dataset
    """
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")

    print("=" * 80)
    print("Loading data with Perturbation processor...")
    print("=" * 80)

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

    # Build gene multigraph (using only physical and regulatory for simplicity)
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

    # Create dataset with Perturbation processor
    print("Creating dataset with Perturbation graph processor...")
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
        graph_processor=Perturbation(),  # Using Perturbation processor
        transform=None,
    )

    # Apply normalization transform
    norm_configs = {"gene_interaction": {"strategy": "standard"}}
    normalizer = COOLabelNormalizationTransform(dataset, norm_configs)
    dataset.transform = normalizer

    print(f"\nNormalization parameters:")
    for phenotype, params in normalizer.stats.items():
        print(f"  {phenotype}:")
        for key, value in params.items():
            if key not in ["bin_edges", "bin_counts", "strategy"]:
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value:.6f}")
                else:
                    print(f"    {key}: {value}")
        print(f"    strategy: {params['strategy']}")

    # Create base data module
    cell_data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=8,
        random_seed=42,
        num_workers=num_workers,
        pin_memory=True,
        prefetch=False,
    )
    cell_data_module.setup()

    # Create subset for faster iteration
    print(f"Setting up PerturbationSubsetDataModule...")
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=subset_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch=False,
        seed=42,
        follow_batch=["perturbation_indices"],  # Need for batch tracking
        gene_subsets={"metabolism": yeast_gem.gene_set},
    )
    perturbation_subset_data_module.setup()

    # Get dataloader and load first batch
    train_loader = perturbation_subset_data_module.train_dataloader()

    print(f"\nLoading batch of size {batch_size}...")
    batch = next(iter(train_loader))
    batch = batch.to(device)

    # Get cell graph
    cell_graph = dataset.cell_graph.to(device)

    # Get gene set size
    gene_set_size = len(dataset.gene_set)

    print(f"\nData loaded successfully!")
    print(f"  Gene set size: {gene_set_size}")
    print(f"  Batch size: {batch['gene'].perturbation_indices.shape[0]}")

    # Print batch structure
    print(f"\nBatch structure:")
    if hasattr(batch, 'node_types'):
        for node_type in batch.node_types:
            print(f"  {node_type}:")
            node_data = batch[node_type]
            for key, value in node_data.items():
                if hasattr(value, 'shape'):
                    print(f"    {key}: {value.shape}")
    else:
        # For simple batch from Perturbation processor
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    if hasattr(v, 'shape'):
                        print(f"    {k}: {v.shape}")

    return dataset, batch, cell_graph, gene_set_size


if __name__ == "__main__":
    # Test the loader
    dataset, batch, cell_graph, gene_set_size = load_perturbation_batch(
        batch_size=32,
        num_workers=4,
        subset_size=10000
    )

    print("\nCell graph structure:")
    print(f"  Nodes: {cell_graph['gene'].num_nodes}")
    print(dataset[0])
    print(batch)
    print(cell_graph)
    print(gene_set_size)
    