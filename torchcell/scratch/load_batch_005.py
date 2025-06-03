# torchcell/scratch/load_batch_005
# [[torchcell.scratch.load_batch_005]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/scratch/load_batch_005
# Test file: tests/torchcell/scratch/test_load_batch_005.py


import os
import os.path as osp
from dotenv import load_dotenv
from torchcell.graph import SCerevisiaeGraph
from torchcell.datamodules import CellDataModule
from torchcell.datamodels.fitness_composite_conversion import CompositeFitnessConverter

# from torchcell.datasets.fungal_up_down_transformer import (
#     FungalUpDownTransformerDataset,
# )
from torchcell.datasets import CodonFrequencyDataset
from torchcell.data import MeanExperimentDeduplicator
from torchcell.data import GenotypeAggregator
from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.data import Neo4jCellDataset
from torchcell.data.graph_processor import SubgraphRepresentation
from torchcell.data.graph_processor import Perturbation
from torchcell.data.graph_processor import DCellGraphProcessor
from tqdm import tqdm
from torchcell.metabolism.yeast_GEM import YeastGEM
from typing import Literal
from torchcell.transforms.hetero_to_dense_mask import HeteroToDenseMask
from torchcell.transforms.regression_to_classification import (
    LabelNormalizationTransform,
)
from torch_geometric.transforms import Compose
from torchcell.graph import build_gene_multigraph


def load_sample_data_batch(
    batch_size=2,
    num_workers=2,
    config: Literal[
        "dango_string9_1", "hetero_cell_bipartite", "dcell", "dcell_2017-07-19"
    ] = "dango_string9_1",
    is_dense: bool = False,
):
    """
    Load a sample data batch for Dango, HeteroCellBipartite, or DCell models.

    Args:
        batch_size: Batch size for dataloader
        num_workers: Number of workers for dataloader
        config: Model configuration:
                "dango_string9_1" - Dango model with STRING v9.1 networks
                "hetero_cell_bipartite" - HeteroCellBipartite model with physical/regulatory networks and metabolism
                "dcell" - DCell model with unfiltered GO graph
                "dcell_2017-07-19" - DCell model with GO graph filtered to 2017-07-19 date
        is_dense: Whether to use dense representation

    Returns:
        Tuple of (dataset, batch, input_channels, max_num_nodes)
    """
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    print(f"DATA_ROOT: {DATA_ROOT}")

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=True,
    )
    # IDEA we are trying to use all gene reprs
    # genome.drop_chrmt()
    genome.drop_empty_go()
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    # selected_node_embeddings = ["codon_frequency"]
    selected_node_embeddings = ["empty"]
    node_embeddings = {}
    # if "fudt_downstream" in selected_node_embeddings:
    #     node_embeddings["fudt_downstream"] = FungalUpDownTransformerDataset(
    #         root=osp.join(DATA_ROOT, "data/scerevisiae/fudt_embedding"),
    #         genome=genome,
    #         model_name="species_downstream",
    #     )

    # if "fudt_upstream" in selected_node_embeddings:
    #     node_embeddings["fudt_upstream"] = FungalUpDownTransformerDataset(
    #         root=osp.join(DATA_ROOT, "data/scerevisiae/fudt_embedding"),
    #         genome=genome,
    #         model_name="species_upstream",
    #     )
    if "codon_frequency" in selected_node_embeddings:
        node_embeddings["codon_frequency"] = CodonFrequencyDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/codon_frequency_embedding"),
            genome=genome,
        )

    with open("experiments/005-kuzmin2018-tmi/queries/001_small_build.cql", "r") as f:
        query = f.read()
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/005-kuzmin2018-tmi/001-small-build"
    )
    # Configuration for different models
    config_options = {
        "dango_string9_1": {
            "graph_names": [
                "string9_1_neighborhood",
                "string9_1_fusion",
                "string9_1_cooccurence",
                "string9_1_coexpression",
                "string9_1_experimental",
                "string9_1_database",
            ],
            "use_metabolism": True,
            "use_gene_ontology": False,
            "graph_processor": Perturbation(),
            "follow_batch": ["perturbation_indices"],
            "date_filter": None,
        },
        "hetero_cell_bipartite": {
            "graph_names": [
                "physical",
                "regulatory",
            ],  # HeteroCellBipartite uses physical and regulatory networks
            "use_metabolism": True,
            "use_gene_ontology": False,
            "graph_processor": SubgraphRepresentation(),
            "follow_batch": ["perturbation_indices"],
            "date_filter": None,
        },
        "dcell": {
            "graph_names": [],  # DCell doesn't use string networks
            "use_metabolism": False,
            "use_gene_ontology": True,
            "graph_processor": DCellGraphProcessor(),
            "follow_batch": ["perturbation_indices", "go_gene_strata_state"],
            "date_filter": None,
        },
        "dcell_2017-07-19": {
            "graph_names": [],  # DCell doesn't use string networks
            "use_metabolism": False,
            "use_gene_ontology": True,
            "graph_processor": DCellGraphProcessor(),
            "follow_batch": ["perturbation_indices", "go_gene_strata_state"],
            "date_filter": "2017-07-19",
        },
    }

    # Apply configuration
    selected_config = config_options[config]

    # Prepare incidence graphs
    incidence_graphs = {}

    # Add metabolism graph if requested
    if selected_config["use_metabolism"]:
        gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))
        incidence_graphs["metabolism_bipartite"] = gem.bipartite_graph

    # Add gene ontology graph if requested
    if selected_config["use_gene_ontology"]:
        # Filtering GO graph for DCell
        G_go = graph.G_go.copy()

        # Apply DCell-specific GO graph filters
        from torchcell.graph import (
            filter_by_date,
            filter_go_IGI,
            filter_redundant_terms,
            filter_by_contained_genes,
        )

        # Apply date filter if specified in the configuration
        date_filter = selected_config["date_filter"]
        if date_filter is not None:
            G_go = filter_by_date(G_go, date_filter)
            print(f"After date filter ({date_filter}): {G_go.number_of_nodes()}")

        G_go = filter_go_IGI(G_go)
        print(f"After IGI filter: {G_go.number_of_nodes()}")
        G_go = filter_redundant_terms(G_go)
        print(f"After redundant filter: {G_go.number_of_nodes()}")
        G_go = filter_by_contained_genes(G_go, n=4, gene_set=genome.gene_set)
        print(f"After containment filter: {G_go.number_of_nodes()}")

        incidence_graphs["gene_ontology"] = G_go

    # Build gene multigraph based on configuration
    gene_multigraph = build_gene_multigraph(
        graph=graph, graph_names=selected_config["graph_names"]
    )

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
        graph_processor=selected_config["graph_processor"],
    )
    # Transforms
    norm_configs = {
        # "fitness": {"strategy": "standard"},  # z-score: (x - mean) / std
        "gene_interaction": {"strategy": "standard"}  # z-score: (x - mean) / std
    }
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
    if is_dense:
        dense_transform = HeteroToDenseMask(
            {"gene": len(genome.gene_set), "reaction": 7122, "metabolite": 2806}
        )
        dataset.transform = Compose([normalizer, dense_transform])
    else:
        dataset.transform = normalizer

    seed = 42
    # Base Module

    cell_data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=[
            "phenotype_label_index",
            "perturbation_count_index",
        ],  # Add this line
        batch_size=8,
        random_seed=42,
        num_workers=4,
        pin_memory=False,
        train_shuffle=False,  # Don't shuffle the training data
    )

    cell_data_module.setup()

    # 1e4 Module
    size = 5e4
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=int(size),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch=False,
        seed=seed,
        follow_batch=selected_config["follow_batch"],
        train_shuffle=False,  # Don't shuffle the training data
    )
    perturbation_subset_data_module.setup()
    max_num_nodes = len(dataset.gene_set)
    for batch in tqdm(perturbation_subset_data_module.train_dataloader()):
        break
    input_channels = dataset.cell_graph["gene"].x.size()[-1]
    return dataset, batch, input_channels, max_num_nodes


if __name__ == "__main__":
    # Test Dango configuration
    print("\n--- Testing Dango Configuration ---")
    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch(
        batch_size=2, num_workers=2, config="dango_string9_1", is_dense=False
    )
    dataset[0]

    print("\n--- Testing HeteroCellBipartite Configuration ---")
    dataset_hetero, batch_hetero, input_channels, max_num_nodes = (
        load_sample_data_batch(
            batch_size=2, num_workers=2, config="hetero_cell_bipartite", is_dense=False
        )
    )
    dataset_hetero[0]

    print("\n--- Testing DCell Configuration (Date Filtered 2017-07-19) ---")
    dataset_filtered, batch_filtered, input_channels, max_num_nodes = (
        load_sample_data_batch(
            batch_size=2, num_workers=2, config="dcell_2017-07-19", is_dense=False
        )
    )
    dataset_filtered[0]

    print("\n--- Testing DCell Configuration (Unfiltered) ---")
    dataset_unfiltered, batch_unfiltered, input_channels, max_num_nodes = (
        load_sample_data_batch(
            batch_size=2, num_workers=2, config="dcell", is_dense=False
        )
    )
    dataset_unfiltered[0]
    print()
