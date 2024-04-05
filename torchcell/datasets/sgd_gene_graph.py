# torchcell/datasets/sgd_gene_graph
# [[torchcell.datasets.sgd_gene_graph]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/sgd_gene_graph
# Test file: tests/torchcell/datasets/test_sgd_gene_graph.py

import torch
from torch_geometric.data import Data
from torchcell.datasets.embedding import BaseEmbeddingDataset
from typing import Callable
import networkx as nx
import os.path as osp


class GraphEmbeddingDataset(BaseEmbeddingDataset):
    MODEL_TO_WINDOW = {"normalized_chrom_pathways": (True), "chrom_pathways": (False)}

    def __init__(
        self,
        root: str,
        graph: nx.Graph,
        model_name: str | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        categorical_features: dict | None = None,
    ):
        self.graph = graph
        self.categorical_features = categorical_features or {}
        super().__init__(root, model_name, transform, pre_transform)

        # Load the categorical_features dictionary if it exists
        if osp.exists(self.processed_paths[1]):
            self.categorical_features = torch.load(self.processed_paths[1])

        self.data, self.slices = torch.load(self.processed_paths[0])

    def initialize_model(self):
        pass  # No need to initialize a model for this dataset

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{self.model_name}.pt", "categorical_features.pt"]

    def process(self):
        data_list = []
        unique_chromosomes = set()
        unique_pathways = set()

        normalize_data = self.MODEL_TO_WINDOW[self.model_name]

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
            pathways = (
                node_data["pathways"] if node_data["pathways"] is not None else []
            )

            unique_chromosomes.add(chromosome)
            unique_pathways.update(pathways)

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

            # Get indices for categorical variables
            chromosome_index = torch.tensor(
                list(unique_chromosomes).index(chromosome), dtype=torch.long
            )

            pathways_indices = torch.tensor(
                [list(unique_pathways).index(pathway) for pathway in pathways],
                dtype=torch.long,
            )

            # Create Data object
            data = Data(id=node_id)
            data.embeddings = {self.model_name: node_features.unsqueeze(0)}
            data.chromosome_index = chromosome_index
            data.pathways_indices = pathways_indices
            data_list.append(data)

        # Update the categorical_features dictionary with the number of unique values
        if "chromosome" not in self.categorical_features:
            self.categorical_features["chromosome"] = {}
        self.categorical_features["chromosome"]["num_values"] = len(unique_chromosomes)

        if "pathways" not in self.categorical_features:
            self.categorical_features["pathways"] = {}
        self.categorical_features["pathways"]["num_values"] = len(unique_pathways)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        # Save the categorical_features dictionary
        torch.save(self.categorical_features, self.processed_paths[1])


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
            categorical_features={"chromosome": {}, "pathways": {}},
        )
        print(f"Completed processing for model: {model_name}")
        print(
            "Number of unique chromosomes:",
            dataset.categorical_features["chromosome"]["num_values"],
        )
        print(
            "Number of unique pathways:",
            dataset.categorical_features["pathways"]["num_values"],
        )
        print("Example data point:")
        print(dataset[0])
        print()


if __name__ == "__main__":
    main()
