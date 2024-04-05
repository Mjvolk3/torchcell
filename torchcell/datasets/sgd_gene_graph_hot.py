# torchcell/datasets/sgd_gene_graph_hot
# [[torchcell.datasets.sgd_gene_graph_hot]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/sgd_gene_graph_hot
# Test file: tests/torchcell/datasets/test_sgd_gene_graph_hot.py


import os
import os.path as osp
import torch
from torch_geometric.data import Data
from torchcell.datasets.embedding import BaseEmbeddingDataset
from typing import Callable
import networkx as nx


class OneHotGraphEmbeddingDataset(BaseEmbeddingDataset):
    MODEL_TO_WINDOW = {
        "normalized_chrom_pathways": (True, True),
        "normalized_chrom": (True, False),
        "chrom_pathways": (False, True),
        "chrom": (False, False),
    }

    def __init__(
        self,
        root: str,
        graph: nx.Graph,
        model_name: str,
        transform: Callable = None,
        pre_transform: Callable = None,
    ):
        self.graph = graph
        super().__init__(root, model_name, transform, pre_transform)
        self.process()

    def initialize_model(self):
        pass

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{self.model_name}.pt"]

    def one_hot(self, index: int, length: int) -> torch.Tensor:
        tensor = torch.zeros(length, dtype=torch.float)
        tensor[index] = 1.0
        return tensor

    def process(self):
        data_list = []
        unique_chromosomes = set()
        unique_pathways = set()

        normalize_data, include_pathways = self.MODEL_TO_WINDOW[self.model_name]

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

            chromosome = node_data["chromosome"]
            pathways = (
                node_data["pathways"] if node_data["pathways"] is not None else []
            )

            unique_chromosomes.add(chromosome)
            unique_pathways.update(pathways)

        # Create dictionaries to map chromosomes and pathways to their indices
        chromosome_to_index = {chrom: idx for idx, chrom in enumerate(unique_chromosomes)}
        pathway_to_index = {pathway: idx for idx, pathway in enumerate(unique_pathways)}

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

            # One-hot encode chromosome
            chromosome_one_hot = torch.zeros(len(unique_chromosomes), dtype=torch.float)
            chromosome_index = chromosome_to_index[chromosome]
            chromosome_one_hot[chromosome_index] = 1.0

            # One-hot encode pathways if include_pathways is True
            if include_pathways:
                pathways_one_hot = torch.zeros(len(unique_pathways), dtype=torch.float)
                for pathway in pathways:
                    pathway_index = pathway_to_index[pathway]
                    pathways_one_hot[pathway_index] = 1.0
            else:
                pathways_one_hot = torch.tensor([])

            # Concatenate node features, chromosome one-hot, and pathways one-hot
            concatenated_features = torch.cat(
                [node_features, chromosome_one_hot, pathways_one_hot]
            )

            # Create Data object
            data = Data(id=node_id)
            data.embeddings = {self.model_name: concatenated_features.unsqueeze(0)}
            data_list.append(data)

        print(f"Total unique chromosomes: {len(unique_chromosomes)}")
        print(f"Total unique pathways: {len(unique_pathways)}")

        # Save processed data
        torch.save(self.collate(data_list), self.processed_paths[0])


def main():
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    DATA_ROOT = os.getenv("DATA_ROOT")
    genome = SCerevisiaeGenome(data_root=osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()

    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )

    for model_name in OneHotGraphEmbeddingDataset.MODEL_TO_WINDOW.keys():
        print(f"Processing model: {model_name}")
        dataset = OneHotGraphEmbeddingDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/sgd_gene_graph_hot"),
            graph=graph.G_gene,
            model_name=model_name,
        )
        print(f"Completed processing for model: {model_name}")
        print(f"Sample data point: {dataset[0]}")
        print()
        print(f"Completed processing for model: {model_name}")

    dataset = OneHotGraphEmbeddingDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/sgd_gene_graph_hot"),
        graph=graph.G_gene,
        model_name="normalized_chrom_pathways",
    )

    count_pathway = 0
    print("len dataset:", len(dataset))

    pathway_counts = {}

    for i in range(len(dataset)):
        chromosome_sum = dataset[i].embeddings["normalized_chrom_pathways"][0, 7:24].sum()
        if chromosome_sum != 1:
            print(
                f"Data point {i}: Chromosome one-hot vector: {dataset[i].embeddings['normalized_chrom_pathways'][0, 7:24]}"
            )
            print(f"Sum of chromosome one-hot vector: {chromosome_sum}")
        
        pathways_sum = dataset[i].embeddings["normalized_chrom_pathways"][0, 24:].sum().item()
        if pathways_sum == 0:
            count_pathway += 1
        
        # Update pathway counts dictionary
        pathway_counts[pathways_sum] = pathway_counts.get(pathways_sum, 0) + 1

    print("Number of data points with no pathway annotation:", count_pathway)
    print("Pathway counts:")
    for count, freq in sorted(pathway_counts.items()):
        print(f"Number of pathways: {count}, Frequency: {freq}")


if __name__ == "__main__":
    main()
