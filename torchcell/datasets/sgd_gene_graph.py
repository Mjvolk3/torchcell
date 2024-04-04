# torchcell/datasets/sgd_gene_graph
# [[torchcell.datasets.sgd_gene_graph]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/sgd_gene_graph
# Test file: tests/torchcell/datasets/test_sgd_gene_graph.py

import torch
from torch_geometric.data import Data
from torchcell.datasets.embedding import BaseEmbeddingDataset
from typing import Callable
import networkx as nx


class GraphEmbeddingDataset(BaseEmbeddingDataset):
    MODEL_TO_WINDOW = {
        "normalized_chr_2_mean_pathways_16": (2, 16, True, "mean"),
        "normalized_chr_2_sum_pathways_16": (2, 16, True, "sum"),
        "normalized_chr_4_mean_pathways_32": (4, 32, True, "mean"),
        "normalized_chr_4_sum_pathways_32": (4, 32, True, "sum"),
        "chr_2_mean_pathways_16": (2, 16, False, "mean"),
        "chr_2_sum_pathways_16": (2, 16, False, "sum"),
        "chr_4_mean_pathways_32": (4, 32, False, "mean"),
        "chr_4_sum_pathways_32": (4, 32, False, "sum"),
    }

    def __init__(
        self,
        root: str,
        graph: nx.Graph,
        model_name: str | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        self.graph = graph
        super().__init__(root, model_name, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def initialize_model(self):
        pass  # No need to initialize a model for this dataset

    def process(self):
        data_list = []
        unique_chromosomes = set()

        if self.model_name not in self.MODEL_TO_WINDOW:
            raise ValueError(f"Invalid model name: {self.model_name}")

        chr_emb_size, pathways_emb_size, normalize_data, pathways_agg_method = (
            self.MODEL_TO_WINDOW[self.model_name]
        )

        # Collect feature values for each node
        feature_values = {
            "length": [],
            "molecular_weight": [],
            "pi": [],
            "median_value": [],
            "median_abs_dev_value": [],
            "start": [],
            "end": [],
        }

        for node_id, node_data in self.graph.nodes(data=True):
            for feature in feature_values.keys():
                value = node_data[feature]
                if value is not None:
                    feature_values[feature].append(value)

        # Compute median values for each feature
        feature_medians = {
            feature: torch.tensor(values).median().item()
            for feature, values in feature_values.items()
        }

        # Compute min and max values for each feature
        feature_min_max = {
            feature: (
                torch.tensor(values).min().item(),
                torch.tensor(values).max().item(),
            )
            for feature, values in feature_values.items()
        }

        for node_id, node_data in self.graph.nodes(data=True):
            # Extract node features
            length = node_data["length"] or feature_medians["length"]
            molecular_weight = (
                node_data["molecular_weight"] or feature_medians["molecular_weight"]
            )
            pi = node_data["pi"] or feature_medians["pi"]
            median_value = node_data["median_value"] or feature_medians["median_value"]
            median_abs_dev_value = (
                node_data["median_abs_dev_value"]
                or feature_medians["median_abs_dev_value"]
            )
            start = node_data["start"] or feature_medians["start"]
            end = node_data["end"] or feature_medians["end"]
            chromosome = node_data["chromosome"]
            pathways = node_data["pathways"]

            unique_chromosomes.add(chromosome)

            # Create node feature vector
            node_features = torch.tensor(
                [
                    length,
                    molecular_weight,
                    pi,
                    median_value,
                    median_abs_dev_value,
                    start,
                    end,
                ],
                dtype=torch.float,
            )

            if normalize_data:
                # Min-max scaling for each feature type
                for i, feature in enumerate(feature_values.keys()):
                    feature_min, feature_max = feature_min_max[feature]
                    node_features[i] = (node_features[i] - feature_min) / (
                        feature_max - feature_min
                    )

            # Create learnable embeddings for categorical variables
            chromosome_embedding = torch.nn.Embedding(
                len(unique_chromosomes), chr_emb_size
            )
            chromosome_index = torch.tensor(
                [list(unique_chromosomes).index(chromosome)], dtype=torch.long
            )
            chromosome_emb = chromosome_embedding(chromosome_index)

            if pathways:
                unique_pathways = list(set(pathways))
                pathways_embedding = torch.nn.Embedding(
                    len(unique_pathways), pathways_emb_size
                )
                pathways_indices = torch.tensor(
                    [unique_pathways.index(pathway) for pathway in pathways],
                    dtype=torch.long,
                )
                if pathways_agg_method == "mean":
                    pathways_emb = pathways_embedding(pathways_indices).mean(dim=0)
                elif pathways_agg_method == "sum":
                    pathways_emb = pathways_embedding(pathways_indices).sum(dim=0)
                else:
                    raise ValueError(
                        f"Invalid pathways aggregation method: {pathways_agg_method}"
                    )
            else:
                pathways_emb = torch.zeros(pathways_emb_size)

            # Concatenate node features and embeddings
            node_embedding = torch.cat(
                [node_features, chromosome_emb.squeeze(), pathways_emb], dim=0
            )

            # Create Data object
            data = Data(id=node_id, embedding=node_embedding)
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def main():
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
    import os.path as osp
    import os

    DATA_ROOT = os.getenv("DATA_ROOT")
    genome = SCerevisiaeGenome(data_root=osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()

    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )

    model_names = GraphEmbeddingDataset.MODEL_TO_WINDOW.keys()

    for model_name in model_names:
        print(f"Processing model: {model_name}")
        dataset = GraphEmbeddingDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/sgd_gene_graph"),
            graph=graph.G_gene,
            model_name=model_name,
        )
        print(f"Completed processing for model: {model_name}")
        print(dataset)
        print()

if __name__ == "__main__":
    main()

