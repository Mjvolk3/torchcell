from torchcell.data import Neo4jCellDataset
import os.path as osp
from dotenv import load_dotenv
from torchcell.graph import SCerevisiaeGraph
from torchcell.datamodules import CellDataModule
import os
import json
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.datasets.fungal_up_down_transformer import FungalUpDownTransformerDataset
from tqdm import tqdm
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from dotenv import load_dotenv
from torch_geometric.transforms import Compose
from tqdm import tqdm

from torchcell.data import GenotypeAggregator, MeanExperimentDeduplicator
from torchcell.data.neo4j_cell import Neo4jCellDataset
from torchcell.data.graph_processor import SubgraphRepresentation
from torchcell.datamodels.fitness_composite_conversion import CompositeFitnessConverter
from torchcell.datamodules import CellDataModule
from torchcell.datasets import CodonFrequencyDataset
from torchcell.datasets.fungal_up_down_transformer import FungalUpDownTransformerDataset

# Import necessary components
from torchcell.graph import SCerevisiaeGraph
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

# from torchcell.transforms.hetero_to_dense import HeteroToDense
from torchcell.transforms.hetero_to_dense_mask import HeteroToDenseMask
from torchcell.transforms.regression_to_classification import (
    LabelNormalizationTransform,
)


def main():
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    with open("experiments/005-kuzmin2018-tmi/queries/001_small_build.cql", "r") as f:
        query = f.read()

    ### Add Embeddings
    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    # genome.drop_chrmt()
    genome.drop_empty_go()
    
    print(f"length of gene_set: {len(genome.gene_set)}")
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
        DATA_ROOT, "data/torchcell/experiments/005-kuzmin2018-tmi/001-small-build"
    )

    codon_frequency = CodonFrequencyDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/codon_frequency_embedding"),
        genome=genome,
    )
    # Create dataset with metabolism network
    print("Creating dataset with metabolism network...")
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=None,
        incidence_graphs={"metabolism_bipartite": YeastGEM().bipartite_graph},
        node_embeddings={
            "codon_frequency": codon_frequency,
            "fudt_3prime": fudt_3prime_dataset,
            "fudt_5prime": fudt_5prime_dataset,
        },
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )

    print(len(dataset))
    # Data module testing

    data_module = CellDataModule(
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
    )
    data_module.setup()
    for batch in tqdm(data_module.all_dataloader()):
        break
        print()

    print("finished")


if __name__ == "__main__":
    main()
