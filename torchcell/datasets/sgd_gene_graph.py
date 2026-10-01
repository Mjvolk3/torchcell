# torchcell/datasets/sgd_gene_graph
# [[torchcell.datasets.sgd_gene_graph]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/sgd_gene_graph
# Test file: tests/torchcell/datasets/test_sgd_gene_graph.py
"""Embedding dataset built from SGD gene-graph node attributes."""

import os.path as osp
from collections.abc import Callable
from typing import Any, cast

import networkx as nx
import torch
from torch_geometric.data import Data

from torchcell.data.embedding import BaseEmbeddingDataset


class ConstantFeatureError(ValueError):
    """A feature has one value on every gene, so min-max normalization is undefined."""


def _refuse_constant_features(feature_min_max: dict[str, tuple[float, float]]) -> None:
    """Raise ``ConstantFeatureError`` naming every feature whose min equals its max."""
    constant = {
        feature: low for feature, (low, high) in feature_min_max.items() if low == high
    }
    if constant:
        raise ConstantFeatureError(
            f"features {constant} take one value on every gene; min-max "
            "normalization would divide 0 by 0, refusing to store NaN"
        )


class MissingChromosomeError(ValueError):
    """A gene carries no chromosome, so it has no chromosome index."""


class GraphEmbeddingDataset(BaseEmbeddingDataset):
    """Node-feature embeddings derived from an SGD gene graph.

    Builds per-gene feature vectors (length, molecular weight, pI, expression,
    coordinates) plus chromosome and pathway categorical indices, optionally
    min-max normalized per the selected ``model_name``.
    """

    MODEL_TO_WINDOW = {"normalized_chrom_pathways": (True), "chrom_pathways": (False)}

    def __init__(
        self,
        root: str,
        graph: nx.Graph,
        model_name: str | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        categorical_features: dict[str, Any] | None = None,
    ) -> None:
        """Store the graph and load processed tensors and categorical metadata."""
        self.graph = graph
        self.categorical_features = categorical_features or {}
        super().__init__(root, model_name, transform, pre_transform)

        # Load the categorical_features dictionary if it exists
        if osp.exists(self.processed_paths[1]):
            self.categorical_features = torch.load(
                self.processed_paths[1], weights_only=False
            )

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    def initialize_model(self) -> None:
        """Do nothing; features come from graph attributes, not a model."""
        pass  # No need to initialize a model for this dataset

    @property
    def processed_file_names(  # type: ignore[override]  # intentionally widens base return; behavior unchanged
        self,
    ) -> list[str]:
        """Return the processed tensor and categorical-feature filenames."""
        return [f"{self.model_name}.pt", "categorical_features.pt"]

    def process(self) -> None:
        """Build node feature tensors and save them with categorical metadata.

        A feature that is None is filled with that feature's median; a value of 0 is a
        value and is kept. Chromosome and pathway indices are positions in the SORTED
        vocabulary of every value in the graph, so one category maps to one index and
        the indices do not depend on ``PYTHONHASHSEED``. Min-max normalization refuses
        a constant feature (``ConstantFeatureError``) rather than storing 0 / 0 = NaN,
        and a gene whose chromosome is None raises ``MissingChromosomeError``.
        On the cached SGD ``G_gene`` (6,607 genes) no feature has a 0 and none is
        constant and none lacks a chromosome (issue #518), so only the categorical
        indices differ from the old builds, and no consumer reads them.
        """
        data_list = []

        normalize_data = self.MODEL_TO_WINDOW[cast(str, self.model_name)]

        # Collect feature values for each node
        feature_values: dict[str, list[Any]] = {
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
        if normalize_data:
            _refuse_constant_features(feature_min_max)

        no_chromosome = [
            node_id
            for node_id, node_data in self.graph.nodes(data=True)
            if node_data["chromosome"] is None
        ]
        if no_chromosome:
            raise MissingChromosomeError(
                f"genes {no_chromosome} have no chromosome; refusing to give them a "
                "chromosome index"
            )
        chromosome_vocab = sorted(
            {node_data["chromosome"] for _, node_data in self.graph.nodes(data=True)}
        )
        pathway_vocab = sorted(
            {
                pathway
                for _, node_data in self.graph.nodes(data=True)
                for pathway in (node_data["pathways"] or [])
            }
        )
        chromosome_to_index = {c: i for i, c in enumerate(chromosome_vocab)}
        pathway_to_index = {p: i for i, p in enumerate(pathway_vocab)}

        for node_id, node_data in self.graph.nodes(data=True):
            # Extract node features; only a missing (None) value takes the median
            values = [
                node_data[feature]
                if node_data[feature] is not None
                else feature_medians[feature]
                for feature in feature_values
            ]
            chromosome = node_data["chromosome"]
            pathways = (
                node_data["pathways"] if node_data["pathways"] is not None else []
            )

            # Create node feature vector
            node_features = torch.tensor(values, dtype=torch.float)

            if normalize_data:
                # Min-max scaling for each feature type
                for i, feature in enumerate(feature_values.keys()):
                    feature_min, feature_max = feature_min_max[feature]
                    node_features[i] = (node_features[i] - feature_min) / (
                        feature_max - feature_min
                    )

            # Get indices for categorical variables
            chromosome_index = torch.tensor(
                chromosome_to_index[chromosome], dtype=torch.long
            )

            pathways_indices = torch.tensor(
                [pathway_to_index[pathway] for pathway in pathways], dtype=torch.long
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
        self.categorical_features["chromosome"]["num_values"] = len(chromosome_vocab)

        if "pathways" not in self.categorical_features:
            self.categorical_features["pathways"] = {}
        self.categorical_features["pathways"]["num_values"] = len(pathway_vocab)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        # Save the categorical_features dictionary
        torch.save(self.categorical_features, self.processed_paths[1])


def main() -> None:
    """Build the gene-graph embedding dataset for each configured model."""
    import os
    import os.path as osp

    from torchcell.graph import SCerevisiaeGraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    DATA_ROOT = os.getenv("DATA_ROOT")

    genome = SCerevisiaeGenome(
        genome_root=osp.join(cast(str, DATA_ROOT), "data/sgd/genome"),
        go_root=osp.join(cast(str, DATA_ROOT), "data/go"),
        overwrite=False,
    )
    genome.drop_chrmt()
    genome.drop_empty_go()

    graph = SCerevisiaeGraph(
        sgd_root=osp.join(cast(str, DATA_ROOT), "data/sgd/genome"),
        string_root=osp.join(cast(str, DATA_ROOT), "data/string"),
        tflink_root=osp.join(cast(str, DATA_ROOT), "data/tflink"),
        genome=genome,
    )

    model_names = GraphEmbeddingDataset.MODEL_TO_WINDOW.keys()

    for model_name in model_names:
        print(f"Processing model: {model_name}")
        dataset = GraphEmbeddingDataset(
            root=osp.join(cast(str, DATA_ROOT), "data/scerevisiae/sgd_gene_graph"),
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
