# torchcell/data/neo4j_cell
# [[torchcell.data.neo4j_cell]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/neo4j_cell
# Test file: tests/torchcell/data/test_neo4j_cell.py

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
from pydantic import field_validator
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import add_self_loops, from_networkx, k_hop_subgraph
from tqdm import tqdm
from enum import Enum, auto
from enum import IntEnum, auto
from torchcell.datasets.embedding import BaseEmbeddingDataset
from torch_geometric.data import Dataset
from typing import Union, Any, Dict
from torch_geometric.data import HeteroData
from torchcell.datamodels import ModelStrictArbitrary
from torchcell.datasets.embedding import BaseEmbeddingDataset
from torchcell.datasets.fungal_up_down_transformer import FungalUpDownTransformerDataset

from torchcell.sequence import GeneSet, Genome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
import attrs
import lmdb
from neo4j import GraphDatabase
import os
from tqdm import tqdm
from attrs import define, field
import os.path as osp
from torchcell.data import Neo4jQueryRaw
from torchcell.sequence import GeneSet, Genome


log = logging.getLogger(__name__)


class ParsedGenome(ModelStrictArbitrary):
    gene_set: GeneSet

    @field_validator("gene_set")
    def validate_gene_set(cls, v):
        if not isinstance(v, GeneSet):
            raise ValueError(f"gene_set must be a GeneSet, got {type(v).__name__}")
        return v


def create_embedding_graph(
    gene_set: GeneSet, embeddings: BaseEmbeddingDataset
) -> nx.Graph:
    """
    Create a NetworkX graph from embeddings.
    """
    # Create an empty NetworkX graph
    G = nx.Graph()

    # Extract and concatenate embeddings for all items in embeddings
    for item in embeddings:
        keys = item["embeddings"].keys()
        if item.id in gene_set:
            item_embeddings = [item["embeddings"][k].squeeze(0) for k in keys]
            concatenated_embedding = torch.cat(item_embeddings)

            # Add nodes to the graph with embeddings as node attributes
            G.add_node(item.id, embedding=concatenated_embedding.numpy())

    return G


def to_cell_data(graphs: Dict[str, nx.Graph]) -> HeteroData:
    hetero_data = HeteroData()

    # Get the node identifiers from the "base" graph
    base_nodes_list = sorted(list(graphs["base"].nodes()))

    # Map each node to a unique index
    node_idx_mapping = {node: idx for idx, node in enumerate(base_nodes_list)}

    # Initialize node attributes for 'gene'
    num_nodes = len(base_nodes_list)
    hetero_data["gene"].num_nodes = num_nodes
    hetero_data["gene"].node_ids = base_nodes_list

    # Initialize the 'x' attribute for 'gene' node type
    hetero_data["gene"].x = torch.zeros((num_nodes, 0), dtype=torch.float)

    # Process each graph and add edges to the HeteroData object
    for graph_type, graph in graphs.items():
        if graph.number_of_edges() > 0:
            # Convert edges to tensor
            edge_index = torch.tensor(
                [
                    (node_idx_mapping[src], node_idx_mapping[dst])
                    for src, dst in graph.edges()
                    if src in node_idx_mapping and dst in node_idx_mapping
                ],
                dtype=torch.long,
            ).t()

            # Determine edge type based on graph_type and assign edge indices
            edge_type = ("gene", f"{graph_type}_interaction", "gene")
            hetero_data[edge_type].edge_index = edge_index
            hetero_data[edge_type].num_edges = edge_index.size(1)
        else:
            # Add node embeddings to the 'x' attribute of 'gene' node type
            embeddings = np.zeros((num_nodes, 0))
            for i, node in enumerate(base_nodes_list):
                if node in graph.nodes and "embedding" in graph.nodes[node]:
                    embedding = graph.nodes[node]["embedding"]
                    if embeddings.shape[1] == 0:
                        embeddings = np.zeros((num_nodes, embedding.shape[0]))
                    embeddings[i] = embedding

            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float)
            hetero_data["gene"].x = torch.cat(
                (hetero_data["gene"].x, embeddings_tensor), dim=1
            )

    return hetero_data


def create_graph_from_gene_set(gene_set: GeneSet) -> nx.Graph:
    """
    Create a graph where nodes are gene names from the GeneSet.
    Initially, this graph will have no edges.
    """
    G = nx.Graph()
    for gene_name in gene_set:
        G.add_node(gene_name)  # Nodes are gene names
    return G


def process_graph(cell_graph: HeteroData, data: dict[str, Any]) -> HeteroData:
    processed_graph = HeteroData()  # breakpoint here

    # Nodes to remove based on the perturbations
    nodes_to_remove = {
        pert.systematic_gene_name for pert in data["experiment"].genotype.perturbations
    }

    # Assuming all nodes are of type 'gene', and copying node information to processed_graph
    processed_graph["gene"].node_ids = [
        nid for nid in cell_graph["gene"].node_ids if nid not in nodes_to_remove
    ]
    processed_graph["gene"].num_nodes = len(processed_graph["gene"].node_ids)
    # Additional information regarding perturbations
    processed_graph["gene"].ids_pert = list(nodes_to_remove)
    processed_graph["gene"].cell_graph_idx_pert = torch.tensor(
        [cell_graph["gene"].node_ids.index(nid) for nid in nodes_to_remove],
        dtype=torch.long,
    )

    # Populate x and x_pert attributes
    node_mapping = {nid: i for i, nid in enumerate(cell_graph["gene"].node_ids)}
    x = cell_graph["gene"].x
    processed_graph["gene"].x = x[
        torch.tensor([node_mapping[nid] for nid in processed_graph["gene"].node_ids])
    ]
    processed_graph["gene"].x_pert = x[processed_graph["gene"].cell_graph_idx_pert]

    # Add fitness phenotype data
    phenotype = data["experiment"].phenotype
    processed_graph["gene"].graph_level = phenotype.graph_level
    processed_graph["gene"].label = phenotype.label
    processed_graph["gene"].label_error = phenotype.label_error
    # TODO we actually want to do this renaming in the datamodel
    # We do it here to replicate behavior for downstream
    # Will break with anything other than fitness obviously
    processed_graph["gene"].label_value = phenotype.fitness
    processed_graph["gene"].label_value_std = phenotype.fitness_std

    # Mapping of node IDs to their new indices after filtering
    new_index_map = {nid: i for i, nid in enumerate(processed_graph["gene"].node_ids)}

    # Processing edges
    for edge_type in cell_graph.edge_types:
        src_type, _, dst_type = edge_type
        edge_index = cell_graph[src_type, _, dst_type].edge_index.numpy()
        filtered_edges = []

        for src, dst in edge_index.T:
            src_id = cell_graph[src_type].node_ids[src]
            dst_id = cell_graph[dst_type].node_ids[dst]

            if src_id not in nodes_to_remove and dst_id not in nodes_to_remove:
                new_src = new_index_map[src_id]
                new_dst = new_index_map[dst_id]
                filtered_edges.append([new_src, new_dst])

        if filtered_edges:
            new_edge_index = torch.tensor(filtered_edges, dtype=torch.long).t()
            processed_graph[src_type, _, dst_type].edge_index = new_edge_index
            processed_graph[src_type, _, dst_type].num_edges = new_edge_index.shape[1]
        else:
            processed_graph[src_type, _, dst_type].edge_index = torch.empty(
                (2, 0), dtype=torch.long
            )
            processed_graph[src_type, _, dst_type].num_edges = 0

    return processed_graph


def parse_genome(genome) -> ParsedGenome:
    if genome is None:
        return None
    else:
        data = {}
        data["gene_set"] = genome.gene_set
        return ParsedGenome(**data)


class Neo4jCellDataset(Dataset):
    def __init__(
        self,
        root: str,
        query: str = None,
        genome: Genome = None,
        graphs: dict[str, nx.Graph] = None,
        node_embeddings: list[BaseEmbeddingDataset] = None,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "torchcell",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
    ):
        # Here for straight pass through - Fails without...
        self.env = None
        self.root = root
        # HACK to get around sql db issue
        self.genome = parse_genome(genome)

        self.raw_db = self.load_raw(uri, username, password, root, query, self.genome)
        base_graph = self.get_init_graphs(self.raw_db, self.genome)
        self.gene_set = GeneSet(base_graph.nodes())  # breakpoint here

        super().__init__(root, transform, pre_transform, pre_filter)

        ###
        # base_graph = self.get_init_graphs(self.raw_db, self.genome)
        # self.gene_set = self.compute_gene_set(base_graph)

        # graphs
        self.graphs = graphs
        if self.graphs is not None:
            # remove edge data from graphs
            for graph in self.graphs.values():
                [graph.edges[edge].clear() for edge in graph.edges()]
            # remove node data from graphs
            for graph in self.graphs.values():
                [graph.nodes[node].clear() for node in graph.nodes()]
            self.graphs["base"] = base_graph
        else:
            self.graphs = {"base": base_graph}

        # embeddings
        self.node_embeddings = node_embeddings
        if self.node_embeddings is not None:
            for name, embedding in self.node_embeddings.items():
                self.graphs[name] = create_embedding_graph(self.gene_set, embedding)
                # Integrate node embeddings into graphs
        self.cell_graph = to_cell_data(self.graphs)

        # Clean up hanging env, for multiprocessing
        self.env = None
        self.raw_db.env = None

    def get_init_graphs(self, raw_db, genome):
        # Setting priority
        if genome is None:
            cell_graph = create_graph_from_gene_set(raw_db.gene_set)
        elif genome:
            cell_graph = create_graph_from_gene_set(genome.gene_set)
        return cell_graph

    @property
    def raw_file_names(self) -> list[str]:
        return "lmdb"

    @staticmethod
    def load_raw(uri, username, password, root_dir, query, genome):
        if genome is not None:
            gene_set = genome.gene_set
            cypher_kwargs = {"gene_set": list(gene_set)}
        else:
            cypher_kwargs = None

        # cypher_kwargs = {"gene_set": ["YAL004W", "YAL010C", "YAL011W", "YAL017W"]}
        raw_db = Neo4jQueryRaw(
            uri=uri,
            username=username,
            password=password,
            root_dir=root_dir,
            query=query,
            max_workers=10,  # IDEA simple for new, might need to parameterize
            num_workers=10,
            cypher_kwargs=cypher_kwargs,
        )
        return raw_db  # break point here

    @property
    def processed_file_names(self) -> list[str]:
        return "lmdb"

    def process(self):
        # strange that we call load_raw might want to change to load.
        if not self.raw_db:
            self.load_raw()

        log.info("Processing raw data into LMDB")
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e12))

        with env.begin(write=True) as txn:
            for idx, data in enumerate(tqdm(self.raw_db)):
                txn.put(f"{idx}".encode(), pickle.dumps(data))
        # TODO compute gene set, maybe others stuff.
        # self.gene_set = self.compute_gene_set()
        self.close_lmdb()

    @property
    def gene_set(self):
        try:
            if osp.exists(osp.join(self.processed_dir, "gene_set.json")):
                with open(osp.join(self.processed_dir, "gene_set.json")) as f:
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
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        with open(osp.join(self.processed_dir, "gene_set.json"), "w") as f:
            json.dump(list(sorted(value)), f, indent=0)
        self._gene_set = value

    def get(self, idx):
        """Initialize LMDB if it hasn't been initialized yet."""
        if self.env is None:
            self._init_lmdb_read()

        with self.env.begin() as txn:
            serialized_data = txn.get(f"{idx}".encode())
            if serialized_data is None:
                return None
            data = pickle.loads(serialized_data)
            subsetted_graph = process_graph(self.cell_graph, data)
            # if self.transform:
            #     subsetted_graph = self.transform(subsetted_graph)
            return subsetted_graph

    def _init_lmdb_read(self):
        """Initialize the LMDB environment."""
        self.env = lmdb.open(
            osp.join(self.processed_dir, "lmdb"),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def len(self) -> int:
        if self.env is None:
            self._init_lmdb_read()

        with self.env.begin(write=False) as txn:
            length = txn.stat()["entries"]
        self.close_lmdb()
        return length

    def close_lmdb(self):
        if self.env is not None:
            self.env.close()
            self.env = None


def main():
    # genome
    import os.path as osp
    from torchcell.datasets.scerevisiae import DmfCostanzo2016Dataset
    from dotenv import load_dotenv

    from torchcell.graph import SCerevisiaeGraph

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    query = """
        MATCH (e:Experiment)<-[:GenotypeMemberOf]-(g:Genotype)<-[:PerturbationMemberOf]-(p:Perturbation)
        WITH e, g, COLLECT(p) AS perturbations
        WHERE ALL(p IN perturbations WHERE p.perturbation_type = 'deletion' AND p.systematic_gene_name IN $gene_set)
        WITH DISTINCT e
        MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference),
            (e)<-[:PhenotypeMemberOf]-(phen:Phenotype),
            (e)<-[:EnvironmentMemberOf]-(env:Environment),
            (env)<-[:MediaMemberOf]-(m:Media {name: 'YEPD'}),
            (env)<-[:TemperatureMemberOf]-(t:Temperature {value: 30})
        WHERE phen.graph_level = 'global' 
            AND (phen.label = 'smf' OR phen.label = 'dmf' OR phen.label = 'tmf')
            AND phen.fitness_std < 0.001
        RETURN e, ref
    """

    ### Simplest case - works
    # dataset = Neo4jCellDataset(
    #     root=osp.join(DATA_ROOT, "data/torchcell/neo4j"), query=query
    # )
    # print(dataset[0])
    # print()

    ### Add a Genome for sequence data - works
    # genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    # genome.drop_chrmt()
    # genome.drop_empty_go()
    # dataset = Neo4jCellDataset(
    #     root=osp.join(DATA_ROOT, "data/torchcell/neo4j"), query=query, genome=genome
    # )
    # print(dataset[0])
    # print()

    ## Add Graph - works
    # genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    # genome.drop_chrmt()
    # genome.drop_empty_go()

    # graph = SCerevisiaeGraph(
    #     data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    # )
    # dataset = Neo4jCellDataset(
    #     root=osp.join(DATA_ROOT, "data/torchcell/neo4j"),
    #     query=query,
    #     genome=genome,
    #     graphs={"physical": graph.G_physical, "regulatory": graph.G_regulatory},
    # )
    # print(dataset[0])
    # print()

    ### Add Embeddings
    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()

    with open("gene_set.json", "w") as f:
        json.dump(list(genome.gene_set), f)

    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
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
    dataset = Neo4jCellDataset(
        root=osp.join(DATA_ROOT, "data/torchcell/neo4j"),
        query=query,
        genome=genome,
        graphs={"physical": graph.G_physical, "regulatory": graph.G_regulatory},
        node_embeddings={
            "fudt_3prime": fudt_3prime_dataset,
            "fudt_5prime": fudt_5prime_dataset,
        },
    )

    ## Data module testing
    from torchcell.datamodules import CellDataModule

    data_module = CellDataModule(dataset=dataset, batch_size=2, num_workers=8)
    data_module.setup()
    for i in tqdm(data_module.train_dataloader()):
        i
        pass
    print("finished")


if __name__ == "__main__":
    main()
