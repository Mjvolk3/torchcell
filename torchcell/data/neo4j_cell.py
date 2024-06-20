# torchcell/data/neo4j_cell
# [[torchcell.data.neo4j_cell]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/neo4j_cell
# Test file: tests/torchcell/data/test_neo4j_cell.py
import torch
import json
import logging
import os
import os.path as osp
import pickle
from collections.abc import Callable
import hashlib
import lmdb
import networkx as nx
import numpy as np
import torch
from pydantic import field_validator
from tqdm import tqdm
from torchcell.datasets.embedding import BaseEmbeddingDataset
from torch_geometric.data import Dataset
from typing import Any, Dict
from torch_geometric.data import HeteroData
from torchcell.datamodels import ModelStrictArbitrary
from torchcell.datasets.fungal_up_down_transformer import FungalUpDownTransformerDataset
from torchcell.datamodels import (
    BaseEnvironment,
    Genotype,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Media,
    ReferenceGenome,
    SgaKanMxDeletionPerturbation,
    SgaNatMxDeletionPerturbation,
    SgaDampPerturbation,
    SgaSuppressorAllelePerturbation,
    SgaTsAllelePerturbation,
    Temperature,
    BaseExperiment,
    ExperimentReference,
    MeanDeletionPerturbation,
    ExperimentReference,
)
from torchcell.sequence import GeneSet, Genome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.data import Neo4jQueryRaw
from abc import ABC, abstractmethod
from memory_profiler import profile
log = logging.getLogger(__name__)


class Deduplicator(ABC):
    @abstractmethod
    def duplicate_check(self, data) -> dict[str, list[int]]: ...

    @abstractmethod
    def create_mean_entry(
        self, duplicate_experiments
    ) -> dict[str, BaseExperiment | ExperimentReference]: ...


class ParsedGenome(ModelStrictArbitrary):
    gene_set: GeneSet

    @field_validator("gene_set")
    def validate_gene_set(cls, v):
        if not isinstance(v, GeneSet):
            raise ValueError(f"gene_set must be a GeneSet, got {type(v).__name__}")
        return v

@profile
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

    return G # breakpoint here

@profile
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

@profile
def create_graph_from_gene_set(gene_set: GeneSet) -> nx.Graph:
    """
    Create a graph where nodes are gene names from the GeneSet.
    Initially, this graph will have no edges.
    """
    G = nx.Graph()
    for gene_name in gene_set:
        G.add_node(gene_name)  # Nodes are gene names
    return G

@profile
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
    @profile
    def __init__(
        self,
        root: str,
        query: str = None,
        genome: Genome = None,
        graphs: dict[str, nx.Graph] = None,
        node_embeddings: list[BaseEmbeddingDataset] = None,
        deduplicator: Deduplicator = None,
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

        self._phenotype_label_index = None
        # set deduplicator
        self.deduplicator = deduplicator
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

        # compute index
        self.phenotype_label_index

    @profile
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
        if not self.raw_db:
            self.load_raw()

        log.info("Processing raw data into LMDB")
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e12))

        if self.deduplicator is not None:
            # Deduplicate domain overlaps and compute mean entries
            duplicate_check = self.deduplicator.duplicate_check(self.raw_db)
            deduplicated_data = []
            log.info("Deduplicating domain overlaps...")
            for hash_key, indices in tqdm(duplicate_check.items()):
                if len(indices) > 1:
                    # Compute mean entry for duplicate experiments
                    duplicate_experiments = [self.raw_db[i] for i in indices]
                    mean_entry = self.deduplicator.create_mean_entry(
                        duplicate_experiments
                    )
                    deduplicated_data.append(mean_entry)
                else:
                    # Keep non-duplicate experiments as is
                    deduplicated_data.append(self.raw_db[indices[0]])
        else:
            deduplicated_data = list(self.raw_db)  # Convert to list

        # Randomly sample the data if max_size is specified
        if self.max_size is not None and self.max_size < len(deduplicated_data):
            log.info(f"Randomly sampling {self.max_size} data points...")
            indices = torch.randperm(len(deduplicated_data))[: self.max_size].tolist()
            deduplicated_data = [
                deduplicated_data[i] for i in indices
            ]  # Use list comprehension

        with env.begin(write=True) as txn:
            for idx, data in enumerate(tqdm(deduplicated_data)):
                txn.put(f"{idx}".encode(), pickle.dumps(data))

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

    @profile
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

    def compute_phenotype_label_index(self) -> dict[str, list[int]]:
        print("Computing phenotype label index...")
        phenotype_label_index = {}

        self._init_lmdb_read()  # Initialize the LMDB environment for reading

        with self.env.begin() as txn:
            cursor = txn.cursor()
            for idx, (key, value) in enumerate(cursor):
                data = pickle.loads(value)
                label = data["experiment"].phenotype.label

                if label not in phenotype_label_index:
                    phenotype_label_index[label] = []
                phenotype_label_index[label].append(idx)

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


class ExperimentDeduplicator(Deduplicator):
    def duplicate_check(self, data) -> dict[str, list[int]]:
        duplicate_check = {}
        for idx, item in enumerate(data):
            perturbations = item["experiment"].genotype.perturbations
            sorted_gene_names = sorted(
                [pert.systematic_gene_name for pert in perturbations]
            )
            hash_key = hashlib.sha256(str(sorted_gene_names).encode()).hexdigest()

            if hash_key not in duplicate_check:
                duplicate_check[hash_key] = []
            duplicate_check[hash_key].append(idx)
        return duplicate_check

    def create_mean_entry(
        self, duplicate_experiments
    ) -> dict[str, BaseExperiment | ExperimentReference]:
        # Check if all phenotypes have the same graph_level and label
        graph_levels = set(
            exp["experiment"].phenotype.graph_level for exp in duplicate_experiments
        )
        labels = set(exp["experiment"].phenotype.label for exp in duplicate_experiments)

        if len(graph_levels) > 1 or len(labels) > 1:
            raise ValueError(
                "Duplicate experiments have different phenotype graph_level or label values."
            )

        # Extract fitness values and standard deviations, excluding None values
        fitness_values = [
            exp["experiment"].phenotype.fitness
            for exp in duplicate_experiments
            if exp["experiment"].phenotype.fitness is not None
        ]
        fitness_stds = [
            exp["experiment"].phenotype.fitness_std
            for exp in duplicate_experiments
            if exp["experiment"].phenotype.fitness_std is not None
        ]

        # Calculate the mean fitness and mean standard deviation, handling empty lists
        mean_fitness = np.mean(fitness_values) if fitness_values else None
        mean_fitness_std = np.mean(fitness_stds) if fitness_stds else None

        # Create a new FitnessPhenotype with the mean values
        mean_phenotype = FitnessPhenotype(
            graph_level=duplicate_experiments[0]["experiment"].phenotype.graph_level,
            label=duplicate_experiments[0]["experiment"].phenotype.label,
            label_error=duplicate_experiments[0]["experiment"].phenotype.label_error,
            fitness=mean_fitness,
            fitness_std=mean_fitness_std,
        )

        mean_perturbations = []
        for pert in duplicate_experiments[0]["experiment"].genotype.perturbations:
            mean_pert = MeanDeletionPerturbation(
                systematic_gene_name=pert.systematic_gene_name,
                perturbed_gene_name=pert.perturbed_gene_name,
                num_duplicates=len(duplicate_experiments),
            )
            mean_perturbations.append(mean_pert)

        mean_genotype = Genotype(perturbations=mean_perturbations)

        mean_experiment = FitnessExperiment(
            genotype=mean_genotype,
            environment=duplicate_experiments[0]["experiment"].environment,
            phenotype=mean_phenotype,
        )

        # Create a new FitnessExperimentReference with the mean values
        fitness_ref_values = [
            exp["reference"].reference_phenotype.fitness
            for exp in duplicate_experiments
            if exp["reference"].reference_phenotype.fitness is not None
        ]
        fitness_ref_stds = [
            exp["reference"].reference_phenotype.fitness_std
            for exp in duplicate_experiments
            if exp["reference"].reference_phenotype.fitness_std is not None
        ]

        # Calculate the mean reference fitness and mean reference standard deviation, handling empty lists
        mean_fitness_ref = np.mean(fitness_ref_values) if fitness_ref_values else None
        mean_fitness_ref_std = np.mean(fitness_ref_stds) if fitness_ref_stds else None

        mean_reference_phenotype = FitnessPhenotype(
            graph_level=duplicate_experiments[0][
                "reference"
            ].reference_phenotype.graph_level,
            label=duplicate_experiments[0]["reference"].reference_phenotype.label,
            label_error=duplicate_experiments[0][
                "reference"
            ].reference_phenotype.label_error,
            fitness=mean_fitness_ref,
            fitness_std=mean_fitness_ref_std,
        )

        # For now we don't deal with reference harmonization - just take first reference
        mean_reference = FitnessExperimentReference(
            reference_genome=duplicate_experiments[0]["reference"].reference_genome,
            reference_environment=duplicate_experiments[0][
                "reference"
            ].reference_environment,
            reference_phenotype=mean_reference_phenotype,
        )

        return {"experiment": mean_experiment, "reference": mean_reference}

def main():
    # genome
    import os.path as osp
    from dotenv import load_dotenv
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.datamodules import CellDataModule

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    # query = """
    #     MATCH (e:Experiment)<-[:GenotypeMemberOf]-(g:Genotype)<-[:PerturbationMemberOf]-(p:Perturbation)
    #     WITH e, g, COLLECT(p) AS perturbations
    #     WHERE ALL(p IN perturbations WHERE p.perturbation_type = 'deletion')
    #     WITH DISTINCT e, perturbations
    #     MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
    #     MATCH (e)<-[:PhenotypeMemberOf]-(phen:Phenotype)
    #     WHERE phen.graph_level = 'global'
    #     AND (phen.label = 'smf' OR phen.label = 'dmf' OR phen.label = 'tmf')
    #     AND phen.fitness_std < 0.005
    #     MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
    #     MATCH (env)<-[:MediaMemberOf]-(m:Media {name: 'YEPD'})
    #     MATCH (env)<-[:TemperatureMemberOf]-(t:Temperature {value: 30})
    #     WITH e, ref, perturbations
    #     WHERE ALL(p IN perturbations WHERE p.systematic_gene_name IN $gene_set)
    #     RETURN e, ref
    # """
    with open("experiments/smf-dmf-tmf-001/query.cql", "r") as f:
        query = f.read()
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
    deduplicator = ExperimentDeduplicator()
    dataset_root = osp.join(DATA_ROOT, "data/torchcell/experiments/smf-dmf-tmf_1e02")
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        genome=genome,
        graphs={"physical": graph.G_physical, "regulatory": graph.G_regulatory},
        node_embeddings={
            "fudt_3prime": fudt_3prime_dataset,
            "fudt_5prime": fudt_5prime_dataset,
        },
        deduplicator=deduplicator,
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
        print()

    print("finished")


if __name__ == "__main__":
    main()
