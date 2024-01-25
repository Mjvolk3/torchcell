# torchcell/datasets/dcell.py
# [[torchcell.datasets.dcell]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/dcell.py
# Test file: torchcell/datasets/test_dcell.py
import json
import logging
import os
import os.path as osp
import pickle
from collections.abc import Callable

import lmdb
import networkx as nx
import numpy as np
import torch
from pydantic import validator
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import (
    add_self_loops,
    from_networkx,
)
from tqdm import tqdm

from torchcell.dataset import Dataset
from torchcell.datamodels import ModelStrictArbitrary
from torchcell.datasets.embedding import BaseEmbeddingDataset
from torchcell.datasets.scerevisiae import (
    DmfCostanzo2016Dataset,
    SmfCostanzo2016Dataset,
)
from torchcell.graph import (
    filter_by_contained_genes,
    filter_by_date,
    filter_go_IGI,
    filter_redundant_terms,
)
from torchcell.models import DCell, DCellLinear, dcell, dcell_from_networkx
from torchcell.sequence import GeneSet, Genome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

log = logging.getLogger(__name__)


class ParsedGenome(ModelStrictArbitrary):
    gene_set: GeneSet

    @validator("gene_set")
    def validate_gene_set(cls, value):
        if not isinstance(value, GeneSet):
            raise ValueError(f"gene_set must be a GeneSet, got {type(value).__name__}")
        return value


class DCellDataset(Dataset):
    """
    Represents a dataset for cellular data.
    """

    # TODO type change experiments
    def __init__(
        self,
        root: str = "data/scerevisiae/cell",
        # genome: Genome = None,
        genome: Genome = None,
        graph: nx.Graph = None,
        embeddings: BaseEmbeddingDataset | None = None,
        experiments: list[InMemoryDataset] | InMemoryDataset = None,
        zero_pert: bool = False,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
    ):
        self._gene_set = None
        self.graph = graph
        self.embeddings = embeddings
        self.experiments = experiments
        self.experiment_datasets = None

        # HACK zero out embedding of pert hack
        self.zero_pert = zero_pert

        # TODO consider moving to Dataset
        self.preprocess_dir = osp.join(root, "preprocess")

        # This is here because we can't run process without getting gene_set
        super().__init__(root, transform, pre_transform, pre_filter)
        # HACK start
        # Extract data from genome object, then remove for pickling with data loader
        self.genome = self.parse_genome(genome)
        del genome
        # HACK end
        # Create the seq graph
        graphs = []
        if self.embeddings:
            G_embedding = self.create_embedding_graph(self.genome, self.embeddings)
            graphs.append(G_embedding)
        self.cell_graph = self.to_cell_data(graphs)
        # HACK to try and get ride of no edge index issue.
        # Self loops fixes the batching problem for no edges.
        self.cell_graph.edge_index = add_self_loops(self.cell_graph.edge_index)[0]
        # LMDB env
        self.env = None

    def to_cell_data(self, graphs: list[nx.Graph]) -> Data:
        G = self.safe_compose(graphs)
        # drop nodes that don't belong to genome.gene_set
        data = from_networkx(G)
        data.ids = list(G.nodes())
        return data

    @staticmethod
    def parse_genome(genome) -> ParsedGenome:
        data = {}
        data["gene_set"] = genome.gene_set
        return ParsedGenome(**data)

    @property
    def raw_file_names(self) -> list[str]:
        # TODO consider return the processed of the experiments, etc.
        # This might cause an issue because there is expected behavior for raw,# and this is not it.
        return None  # Specify raw files if needed

    @property
    def processed_file_names(self) -> list[str]:
        return "data.lmdb"

    # HACK comment out for now
    # @property
    # def wt(self):
    #     # Need to be able to combine WTs into one WT
    #     # wts = [experiment.wt for experiment in self.experiments]
    #     # TODO aggregate WTS. For now just return the first one.
    #     wt = self.experiments.wt
    #     subset_data = self._subset_graph(wt)
    #     data = self._add_label(subset_data, wt)
    #     return data

    @property
    def wt(self):
        return None

    @staticmethod
    def safe_compose(graphs):
        if any(isinstance(G, nx.DiGraph) for G in graphs):
            # Convert all graphs to DiGraph if at least one is directed
            graphs = [
                G if isinstance(G, nx.DiGraph) else G.to_directed() for G in graphs
            ]

        # Start with an empty graph of the appropriate type
        composed_graph = (
            nx.DiGraph() if isinstance(graphs[0], nx.DiGraph) else nx.Graph()
        )

        for G in graphs:
            # Check for overlapping node data
            for node, data in G.nodes(data=True):
                if node in composed_graph:
                    for key, value in data.items():
                        if key in composed_graph.nodes[node]:
                            if isinstance(value, np.ndarray):
                                if not np.array_equal(
                                    value, composed_graph.nodes[node][key]
                                ):
                                    raise ValueError(
                                        f"Overlapping node data found for node {node}: {key}"
                                    )
                            elif composed_graph.nodes[node][key] != value:
                                raise ValueError(
                                    f"Overlapping node data found for node {node}: {key}"
                                )

            # Check for overlapping edge data
            for node1, node2, data in G.edges(data=True):
                if composed_graph.has_edge(node1, node2):
                    for key, value in data.items():
                        if key in composed_graph.edges[node1, node2]:
                            if isinstance(value, np.ndarray):
                                if not np.array_equal(
                                    value, composed_graph.edges[node1, node2][key]
                                ):
                                    raise ValueError(
                                        f"Overlapping edge data found for edge {(node1, node2)}: {key}"
                                    )
                            elif composed_graph.edges[node1, node2][key] != value:
                                raise ValueError(
                                    f"Overlapping edge data found for edge {(node1, node2)}: {key}"
                                )

            composed_graph = nx.compose(composed_graph, G)
        # After all graphs are composed unify nodes attrs into x
        # probably need to unify edge attrs too
        for node, data in composed_graph.nodes(data=True):
            attributes_list = []
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    attributes_list.append(value)
                # You can handle other types as needed

            # For simplicity, assuming all attributes are numpy arrays
            concatenated_attributes = np.concatenate(attributes_list)

            # Set the concatenated attributes to 'x' and remove other attributes
            composed_graph.nodes[node]["x"] = concatenated_attributes
            keys_to_remove = [key for key in data.keys() if key != "x"]
            for key in keys_to_remove:
                del composed_graph.nodes[node][key]

        return composed_graph

    @staticmethod
    def create_embedding_graph(genome, embeddings: BaseEmbeddingDataset) -> nx.Graph:
        """
        Create a NetworkX graph from embeddings.
        """
        # Create an empty NetworkX graph
        G = nx.Graph()

        # Extract and concatenate embeddings for all items in embeddings
        for item in embeddings:
            keys = item["embeddings"].keys()
            if item.id in genome.gene_set:
                item_embeddings = [item["embeddings"][k].squeeze(0) for k in keys]
                concatenated_embedding = torch.cat(item_embeddings)

                # Add nodes to the graph with embeddings as node attributes
                G.add_node(item.id, embedding=concatenated_embedding.numpy())

        return G

    def process(self):
        combined_data = []
        self.gene_set = self.compute_gene_set()

        # Precompute gene_set for faster lookup
        gene_set = self.gene_set

        # # Use list comprehension and any() for fast filtering
        combined_data = [
            item
            for item in tqdm(self.experiments)
            if all(i["id"] in gene_set for i in item.genotype)
        ]

        log.info("creating lmdb database")
        # Initialize LMDB environment
        env = lmdb.open(osp.join(self.processed_dir, "data.lmdb"), map_size=int(1e12))

        with env.begin(write=True) as txn:
            for idx, item in tqdm(enumerate(combined_data)):
                data = Data()
                data.genotype = item["genotype"]
                data.phenotype = item["phenotype"]

                # Serialize the data object using pickle
                serialized_data = pickle.dumps(data)

                # Save the serialized data in the LMDB environment
                txn.put(f"{idx}".encode(), serialized_data)

    @property
    def gene_set(self):
        try:
            if osp.exists(osp.join(self.preprocess_dir, "gene_set.json")):
                with open(osp.join(self.preprocess_dir, "gene_set.json")) as f:
                    self._gene_set = set(json.load(f))
            elif self._gene_set is None:
                raise ValueError(
                    "gene_set not written during process. "
                    "Please call compute_gene_set in process."
                )
            return GeneSet(self._gene_set)
        except json.JSONDecodeError:
            raise ValueError("Invalid or empty JSON file found.")

    @gene_set.setter
    def gene_set(self, value):
        if not value:
            raise ValueError("Cannot set an empty or None value for gene_set")
        if not osp.exists(self.preprocess_dir):
            os.makedirs(self.preprocess_dir)
        with open(osp.join(self.preprocess_dir, "gene_set.json"), "w") as f:
            json.dump(list(sorted(value)), f, indent=0)
        self._gene_set = value

    def compute_gene_set(self):
        if not self._gene_set:
            if isinstance(self.experiments, Dataset):
                experiment_gene_set = self.experiments.gene_set
            else:
                # TODO: handle other data types for experiments, if necessary
                raise NotImplementedError(
                    "Expected 'experiments' to be of type InMemoryDataset"
                )
            # Not sure we should take the intersection here...
            # Could use gene_set from genome instead, since this is base
            # In case of gene addition would need to update the gene_set
            # then cell_dataset should be max possible.
            cell_gene_set = set(self.genome.gene_set).intersection(experiment_gene_set)
        return cell_gene_set

    def _subset_graph(self, data: Data) -> Data:
        """
        Subset the reference graph based on the genes in data.genotype.
        """
        # Nodes to remove based on the genes in data.genotype
        nodes_to_remove = torch.tensor(
            [
                self.cell_graph.ids.index(gene["id"])
                for gene in data.genotype
                if gene["id"] in self.cell_graph.ids
            ],
            dtype=torch.long,
        )

        perturbed_nodes = nodes_to_remove.clone().detach()

        # Compute the nodes to keep
        all_nodes = torch.arange(self.cell_graph.num_nodes, dtype=torch.long)
        nodes_to_keep = torch.tensor(
            [node for node in all_nodes if node not in perturbed_nodes],
            dtype=torch.long,
        )

        # Get the induced subgraph using the nodes to keep
        subset_graph = self.cell_graph.subgraph(nodes_to_keep)
        subset_remove_graph = self.cell_graph.subgraph(nodes_to_remove)
        subset_graph.ids_pert = subset_remove_graph.ids
        # TODO remove all below.
        subset_graph.x_pert = subset_remove_graph.x
        subset_graph.x_pert_idx = perturbed_nodes
        # Dcell data
        G_dcell = dcell.delete_genes(
            go_graph=self.graph.G_go, deletion_gene_set=GeneSet(subset_graph.ids_pert)
        )
        data = dcell_from_networkx(G_dcell)
        return data

    def _add_label(self, data: Data, original_data: Data) -> Data:
        """
        Adds the dmf_fitness label to the data object if it exists in the original data's phenotype["observation"].

        Args:
            data (Data): The Data object to which the label should be added.
            original_data (Data): The original Data object from which the label should be extracted.

        Returns:
            Data: The modified Data object with the added label.
        """
        if "dmf" in original_data.phenotype["observation"]:
            # TODO change dmf in costanzo to be fitness - need to standardize
            data.fitness = original_data.phenotype["observation"]["dmf"]
        if "fitness" in original_data.phenotype["observation"]:
            # TODO change dmf in costanzo to be fitness - need to standardize
            data.fitness = original_data.phenotype["observation"]["fitness"]
        if "genetic_interaction_score" in original_data.phenotype["observation"]:
            data.genetic_interaction_score = original_data.phenotype["observation"][
                "genetic_interaction_score"
            ]
        return data

    # def get(self, idx: int) -> Data:
    #     env = lmdb.open(self.processed_paths[0], readonly=True, lock=False)
    #     with env.begin() as txn:
    #         serialized_data = txn.get(f"{idx}".encode())
    #         if serialized_data is None:
    #             return None
    #         data = pickle.loads(serialized_data)
    #         if self.transform:
    #             data = self.transform(data)

    #         # Get the subset data using the separate method
    #         subset_data = self._subset_graph(data)

    #         # Add the dmf_fitness label to the subset_data
    #         subset_data = self._add_label(subset_data, data)

    #         return subset_data

    def get(self, idx):
        """Initialize LMDB if it hasn't been initialized yet."""
        if self.env is None:
            self._init_db()

        with self.env.begin() as txn:
            serialized_data = txn.get(f"{idx}".encode())
            if serialized_data is None:
                return None
            data = pickle.loads(serialized_data)
            if self.transform:
                data = self.transform(data)

            subset_data = self._subset_graph(data)
            subset_data = self._add_label(subset_data, data)
            return subset_data

    def _init_db(self):
        """Initialize the LMDB environment."""
        self.env = lmdb.open(
            osp.join(self.processed_dir, "data.lmdb"),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def len(self) -> int:
        if self.env is None:
            self._init_db()

        with self.env.begin() as txn:
            length = txn.stat()["entries"]

        # Must be closed for dataloader num_workers > 0
        self.close_lmdb()

        return length

    def close_lmdb(self):
        if self.env is not None:
            self.env.close()
            self.env = None


def main():
    # genome
    import os.path as osp

    from dotenv import load_dotenv

    from torchcell.datasets import OneHotGeneDataset
    from torchcell.graph import SCerevisiaeGraph

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    genome = SCerevisiaeGenome(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), overwrite=True
    )
    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )
    dmf_dataset = DmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_1e5"),
        preprocess={"duplicate_resolution": "low_dmf_std"},
        # subset_n=100,
    )
    smf_dataset = SmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_smf"),
        preprocess={"duplicate_resolution": "low_std_both"},
        skip_process_file_exist_check=True,
    )
    gene_set = smf_dataset.gene_set.union(dmf_dataset.gene_set)
    #
    G = graph.G_go.copy()
    # Filtering
    G = filter_by_date(G, "2017-07-19")
    G = filter_go_IGI(G)
    G = filter_redundant_terms(G)
    G = filter_by_contained_genes(G, n=1, gene_set=gene_set)

    # replace graph
    graph.G_go = G

    # Instantiate the model
    dcell_model = DCell(go_graph=graph.G_go)

    one_hot = OneHotGeneDataset(root="data/scerevisiae/gene_one_hot", genome=genome)
    embeddings = one_hot

    # Experiments
    experiments = DmfCostanzo2016Dataset(
        preprocess={"duplicate_resolution": "low_dmf_std"},
        root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_1e3"),
        subset_n=1000,
    )

    cell_dataset = DCellDataset(
        root="data/scerevisiae/cell_1e3",
        genome=genome,
        graph=graph,
        embeddings=embeddings,
        experiments=experiments,
    )

    print(cell_dataset)
    print(cell_dataset.gene_set)
    print(cell_dataset[0])
    print(cell_dataset.wt)
    from torch_geometric.data import Batch

    batch = Batch.from_data_list([cell_dataset[0], cell_dataset[1]])
    dcell_subsystem_output = dcell_model(batch)
    dcell_linear = DCellLinear(dcell_model.subsystems, output_size=1)
    dcell_linear_output = dcell_linear(dcell_subsystem_output)
    print(dcell_linear_output)


if __name__ == "__main__":
    main()
