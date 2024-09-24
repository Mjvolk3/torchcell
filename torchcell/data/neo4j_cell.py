# torchcell/data/neo4j_cell
# [[torchcell.data.neo4j_cell]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/neo4j_cell
# Test file: tests/torchcell/data/test_neo4j_cell.py
import torch
import json
import logging
import os
import os.path as osp
from collections.abc import Callable
import lmdb
import networkx as nx
from pydantic import field_validator
from tqdm import tqdm
from torchcell.data.embedding import BaseEmbeddingDataset
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
from torchcell.datamodels import ModelStrictArbitrary
from torchcell.datamodels import Converter
from torchcell.data.deduplicate import ExperimentDeduplicator, Deduplicator
from torchcell.sequence import GeneSet, Genome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.datamodels import ExperimentType, ExperimentReferenceType
from torchcell.data.neo4j_query_raw import Neo4jQueryRaw
from typing import Type, Optional
from enum import Enum, auto
from torchcell.datamodels.schema import (
    EXPERIMENT_TYPE_MAP,
    EXPERIMENT_REFERENCE_TYPE_MAP,
)

log = logging.getLogger(__name__)


class ParsedGenome(ModelStrictArbitrary):
    gene_set: GeneSet

    @field_validator("gene_set")
    def validate_gene_set(cls, v):
        if not isinstance(v, GeneSet):
            raise ValueError(f"gene_set must be a GeneSet, got {type(v).__name__}")
        return v


# @profile
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

            G.add_node(item.id, embedding=concatenated_embedding)

    return G


# @profile
def to_cell_data(graphs: dict[str, nx.Graph]) -> HeteroData:
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
            embeddings = torch.zeros((num_nodes, 0), dtype=torch.float)
            for i, node in enumerate(base_nodes_list):
                if node in graph.nodes and "embedding" in graph.nodes[node]:
                    embedding = graph.nodes[node]["embedding"]
                    if embeddings.shape[1] == 0:
                        embeddings = torch.zeros(
                            (num_nodes, embedding.shape[0]), dtype=torch.float
                        )
                    embeddings[i] = embedding

            hetero_data["gene"].x = torch.cat(
                (hetero_data["gene"].x, embeddings), dim=1
            )

    return hetero_data


# @profile
def create_graph_from_gene_set(gene_set: GeneSet) -> nx.Graph:
    """
    Create a graph where nodes are gene names from the GeneSet.
    Initially, this graph will have no edges.
    """
    G = nx.Graph()
    for gene_name in gene_set:
        G.add_node(gene_name)  # Nodes are gene names
    return G


def process_graph(
    cell_graph: HeteroData, data: dict[str, ExperimentType | ExperimentReferenceType]
) -> HeteroData:
    if "experiment" not in data or "experiment_reference" not in data:
        raise ValueError(
            "Data must contain both 'experiment' and 'experiment_reference' keys"
        )

    if not isinstance(data["experiment"], ExperimentType) or not isinstance(
        data["experiment_reference"], ExperimentReferenceType
    ):
        raise TypeError(
            "'experiment' and 'experiment_reference' must be instances of ExperimentType and ExperimentReferenceType respectively"
        )

    processed_graph = HeteroData()

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
    processed_graph["gene"].label_name = phenotype.label_name
    processed_graph["gene"].label_statistic_name = phenotype.label_statistic_name
    processed_graph["gene"][phenotype.label_name] = getattr(
        phenotype, phenotype.label_name
    )
    if phenotype.label_statistic_name is not None:
        processed_graph["gene"][phenotype.label_statistic_name] = getattr(
            phenotype, phenotype.label_statistic_name
        )

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


class ProcessingStep(Enum):
    RAW = auto()
    CONVERSION = auto()
    DEDUPLICATION = auto()
    AGGREGATION = auto()
    PROCESSED = auto()


# TODO implement
class Aggregator:
    pass


class Neo4jCellDataset(Dataset):
    # @profile
    def __init__(
        self,
        root: str,
        query: str = None,
        genome: Genome = None,
        graphs: dict[str, nx.Graph] = None,
        node_embeddings: list[BaseEmbeddingDataset] = None,
        converter: Optional[Type[Converter]] = None,
        deduplicator: Optional[Type[Deduplicator]] = None,
        aggregator: Optional[Type[Aggregator]] = None,
        overwrite_intermediates: bool = False,
        max_size: int = None,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "torchcell",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
    ):
        self.max_size = max_size
        # Here for straight pass through - Fails without...
        self.env = None
        self.root = root
        self.overwrite_intermediates = overwrite_intermediates
        self._phenotype_label_index = None

        # HACK to get around sql db issue
        self.genome = parse_genome(genome)

        self.raw_db = self.load_raw(uri, username, password, root, query, self.genome)

        print()

        self.converter = (
            converter(root=self.root, query=self.raw_db) if converter else None
        )
        self.deduplicator = deduplicator(root=self.root) if deduplicator else None
        self.aggregator = aggregator(root=self.root) if aggregator else None

        self.processing_steps = self._determine_processing_steps()

        base_graph = self.get_init_graphs(self.raw_db, self.genome)
        self.gene_set = GeneSet(base_graph.nodes())

        super().__init__(root, transform, pre_transform, pre_filter)

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
        # TODO remove
        # node_embeddings = {}
        if node_embeddings is not None:
            for name, embedding in node_embeddings.items():
                self.graphs[name] = create_embedding_graph(self.gene_set, embedding)
                # Integrate node embeddings into graphs
        self.cell_graph = to_cell_data(self.graphs)

        # HACK removing state for mp
        del self.graphs
        del node_embeddings

        # Clean up hanging env, for multiprocessing
        self.env = None
        self.raw_db.env = None

        # compute index
        self.phenotype_label_index

    def _determine_processing_steps(self):
        steps = [ProcessingStep.RAW]
        if self.converter is not None:
            steps.append(ProcessingStep.CONVERSION)
        if self.deduplicator is not None:
            steps.append(ProcessingStep.DEDUPLICATION)
        if self.aggregator is not None:
            steps.append(ProcessingStep.AGGREGATION)
        steps.append(ProcessingStep.PROCESSED)
        return steps

    def _get_lmdb_path(self, step: ProcessingStep):
        if step == ProcessingStep.RAW:
            return os.path.join(self.root, "raw", "lmdb")
        elif step == ProcessingStep.PROCESSED:
            return os.path.join(self.processed_dir, "lmdb")
        else:
            return os.path.join(self.root, step.name.lower(), "lmdb")

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
        print("================")
        print(f"raw root_dir: {root_dir}")
        print("================")
        raw_db = Neo4jQueryRaw(
            uri=uri,
            username=username,
            password=password,
            root_dir=root_dir,
            query=query,
            io_workers=10,  # IDEA simple for new, might need to parameterize
            num_workers=10,
            cypher_kwargs=cypher_kwargs,
        )
        return raw_db  # break point here

    @property
    def processed_file_names(self) -> list[str]:
        return "lmdb"

    def process(self):
        current_step = ProcessingStep.RAW
        for next_step in self.processing_steps[1:]:
            input_path = self._get_lmdb_path(current_step)
            output_path = self._get_lmdb_path(next_step)

            if next_step == ProcessingStep.CONVERSION:
                self.converter.process(input_path, output_path)
            elif next_step == ProcessingStep.DEDUPLICATION:
                self.deduplicator.process(input_path, output_path)
            elif next_step == ProcessingStep.AGGREGATION:
                self.aggregator.process(input_path, output_path)
            elif next_step == ProcessingStep.PROCESSED:
                self._copy_lmdb(input_path, output_path)

            if self.overwrite_intermediates and next_step != ProcessingStep.PROCESSED:
                os.remove(input_path)

            current_step = next_step

        if self.max_size:
            self._apply_max_size(self._get_lmdb_path(ProcessingStep.PROCESSED))

    def _copy_lmdb(self, src_path: str, dst_path: str):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        env_src = lmdb.open(src_path, readonly=True)
        env_dst = lmdb.open(dst_path, map_size=int(1e12))

        with env_src.begin() as txn_src, env_dst.begin(write=True) as txn_dst:
            cursor = txn_src.cursor()
            for key, value in cursor:
                txn_dst.put(key, value)

        env_src.close()
        env_dst.close()

    def _apply_max_size(self, lmdb_path: str):
        env = lmdb.open(lmdb_path, map_size=int(1e12))
        with env.begin(write=True) as txn:
            cursor = txn.cursor()
            count = 0
            for key, _ in cursor:
                count += 1
                if count > self.max_size:
                    if not cursor.delete():
                        print(f"Failed to delete key: {key}")
        env.close()

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
            data = json.loads(serialized_data.decode("utf-8"))
            experiment_class = EXPERIMENT_TYPE_MAP[
                data["experiment"]["experiment_type"]
            ]
            experiment_reference_class = EXPERIMENT_REFERENCE_TYPE_MAP[
                data["experiment_reference"]["experiment_reference_type"]
            ]
            reconstructed_data = {
                "experiment": experiment_class(**data["experiment"]),
                "experiment_reference": experiment_reference_class(
                    **data["experiment_reference"]
                ),
            }
            subsetted_graph = process_graph(self.cell_graph, reconstructed_data)
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

    def compute_phenotype_label_index(self) -> dict[str, list[int]]:
        print("Computing phenotype label index...")
        phenotype_label_index = {}

        self._init_lmdb_read()  # Initialize the LMDB environment for reading

        with self.env.begin() as txn:
            cursor = txn.cursor()
            for idx, (key, value) in enumerate(cursor):
                try:
                    data = json.loads(value.decode("utf-8"))
                    experiment_class = EXPERIMENT_TYPE_MAP[
                        data["experiment"]["experiment_type"]
                    ]
                    experiment = experiment_class(**data["experiment"])
                    label_name = experiment.phenotype.label_name

                    if label_name not in phenotype_label_index:
                        phenotype_label_index[label_name] = []
                    phenotype_label_index[label_name].append(idx)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON for entry {idx}. Skipping this entry.")
                except Exception as e:
                    print(
                        f"Error processing entry {idx}: {str(e)}. Skipping this entry."
                    )

        self.close_lmdb()  # Close the LMDB environment

        return phenotype_label_index

    @property
    def phenotype_label_index(self) -> dict[str, list[bool]]:
        if osp.exists(osp.join(self.processed_dir, "phenotype_label_index.json")):
            with open(
                osp.join(self.processed_dir, "phenotype_label_index.json"), "r"
            ) as file:
                self._phenotype_label_index = json.load(file)
        else:
            self._phenotype_label_index = self.compute_phenotype_label_index()
            with open(
                osp.join(self.processed_dir, "phenotype_label_index.json"), "w"
            ) as file:
                json.dump(self._phenotype_label_index, file)
        return self._phenotype_label_index


def main():
    # genome
    import os.path as osp
    from dotenv import load_dotenv
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.datamodules import CellDataModule
    from torchcell.datamodels.gene_essentiality_to_fitness_conversion import (
        GeneEssentialityToFitnessConverter,
    )

    from torchcell.datasets.fungal_up_down_transformer import (
        FungalUpDownTransformerDataset,
    )

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    with open("experiments/003-fit-int/queries/test_query.cql", "r") as f:
        query = f.read()

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
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/test_dataset"
    )
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        genome=genome,
        graphs={"physical": graph.G_physical, "regulatory": graph.G_regulatory},
        node_embeddings={
            "fudt_3prime": fudt_3prime_dataset,
            "fudt_5prime": fudt_5prime_dataset,
        },
        converter=GeneEssentialityToFitnessConverter,
        deduplicator=ExperimentDeduplicator,
        max_size=int(1e2),
    )
    print(len(dataset))
    # Data module testing

    data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        batch_size=8,
        random_seed=42,
        num_workers=4,
        pin_memory=False,
    )
    data_module.setup()
    for batch in tqdm(data_module.all_dataloader()):
        pass
        print(batch)

    print("finished")


if __name__ == "__main__":
    main()
