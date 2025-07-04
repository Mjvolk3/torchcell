"""
InferenceDataset: A simplified dataset for inference that uses LMDB storage.

This dataset follows the Neo4jCellDataset pattern but removes the complexity of
database queries, conversion, deduplication, and aggregation. It uses LMDB for
efficient storage and retrieval of experiments.
"""

import json
import os
import os.path as osp
import pickle
from typing import Callable

import lmdb
import numpy as np
import pandas as pd
from torch_geometric.data import Dataset
from tqdm import tqdm
from sortedcontainers import SortedDict
import networkx as nx
import hypernetx as hnx

from torchcell.data.cell_data import to_cell_data
from torchcell.data.neo4j_cell import (
    create_embedding_graph,
    create_graph_from_gene_set,
    min_max_normalize_embedding,
    min_max_normalize_dataset,
)
from torchcell.datamodels import (
    Environment,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Genotype,
    Media,
    Temperature,
    ReferenceGenome,
    EXPERIMENT_TYPE_MAP,
    EXPERIMENT_REFERENCE_TYPE_MAP,
    PhenotypeType,
)
from torchcell.datamodels.schema import SgaKanMxDeletionPerturbation, Phenotype, ModelStrict
from torchcell.graph import GeneGraph, GeneMultiGraph
from torchcell.sequence import GeneSet
from typing import Optional
from pydantic import field_validator

# Import required types
from torchcell.data.embedding import BaseEmbeddingDataset
from torchcell.data.graph_processor import GraphProcessor


class InferencePhenotype(FitnessPhenotype):
    """Phenotype class for inference that doesn't require actual fitness values."""
    # Override fitness fields to be optional
    fitness: Optional[float] = None
    fitness_std: Optional[float] = None
    
    @field_validator("fitness")
    def validate_fitness(cls, v):
        # Allow None values for inference
        if v is None:
            return v
        if v <= 0:
            return 0.0
        return v


class InferenceExperiment(FitnessExperiment):
    """Experiment class for inference that uses InferencePhenotype."""
    experiment_type: str = "inference"
    phenotype: InferencePhenotype


class InferenceExperimentReference(FitnessExperimentReference):
    """Experiment reference class for inference."""
    experiment_reference_type: str = "inference"
    phenotype_reference: InferencePhenotype


class InferenceDataset(Dataset):
    """
    A simplified dataset for inference that uses LMDB to store experiments
    and applies graph structures without database dependencies.

    Follows the same LMDB storage format as Neo4jCellDataset:
    - Each entry is a JSON list containing experiment and reference data
    - Integer keys for indexing (0, 1, 2, ...)
    - Supports all standard indices (gene, phenotype, dataset_name)
    """

    def __init__(
        self,
        root: str,
        gene_set: GeneSet,
        graphs: dict[str, GeneGraph] | None = None,
        incidence_graphs: dict[str, nx.Graph | hnx.Hypergraph] | None = None,
        node_embeddings: dict[str, BaseEmbeddingDataset] | None = None,
        graph_processor: GraphProcessor | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        """
        Args:
            root: Root directory for processed data (LMDB will be created in root/processed/lmdb)
            gene_set: GeneSet for the organism
            graphs: Dict of graph names to GeneGraph objects
            incidence_graphs: Dict of incidence graph names to Graph/Hypergraph objects
            node_embeddings: Dict of embedding names to embedding datasets
            graph_processor: Processor for converting experiments to graph data
            transform: Optional transform to apply to each data object
            pre_transform: Optional pre-transform (applied once during processing)
        """
        self.gene_set = gene_set
        self.graphs = graphs or {}
        self.incidence_graphs = incidence_graphs or {}
        self.node_embeddings = node_embeddings or {}
        self.graph_processor = graph_processor
        self.env = None
        self._len = None
        self._is_any_perturbed_gene_index = None
        self._phenotype_info = None

        # Initialize parent dataset
        super().__init__(root, transform, pre_transform)

        # Initialize a GeneMultiGraph with a base graph (following Neo4jCellDataset)
        base_graph = create_graph_from_gene_set(self.gene_set)

        # Set up the GeneMultiGraph
        if graphs is None:
            # Create a new GeneMultiGraph with just the base graph
            graphs_dict = SortedDict({"base": base_graph})
            multigraph = GeneMultiGraph(graphs=graphs_dict)
        elif isinstance(graphs, GeneMultiGraph):
            # If it's already a GeneMultiGraph, ensure it has a base graph
            if "base" not in graphs.graphs:
                graphs.graphs["base"] = base_graph
            multigraph = graphs
        else:
            # Handle dict[str, GeneGraph] input
            graphs_dict = SortedDict(graphs)
            # Ensure we have a base graph
            if "base" not in graphs_dict:
                graphs_dict["base"] = base_graph
            multigraph = GeneMultiGraph(graphs=graphs_dict)

        # Add embeddings as GeneGraphs to the GeneMultiGraph
        if self.node_embeddings is not None:
            for name, embedding in self.node_embeddings.items():
                embedding_graph = create_embedding_graph(self.gene_set, embedding)
                multigraph.graphs[name] = embedding_graph

        # cell graph used in get item
        self.cell_graph = to_cell_data(
            multigraph, self.incidence_graphs, add_remaining_gene_self_loops=True
        )

    @property
    def raw_file_names(self) -> list[str]:
        """Raw file names (not used for inference dataset)."""
        return []

    @property
    def processed_file_names(self) -> list[str]:
        """Processed file names."""
        return ["lmdb"]

    @property
    def processed_dir(self) -> str:
        """Processed directory path."""
        return osp.join(self.root, "processed")

    @property
    def phenotype_info(self) -> list[PhenotypeType]:
        """Get phenotype info (computed lazily)."""
        if self._phenotype_info is None:
            self._phenotype_info = self._compute_phenotype_info()
        return self._phenotype_info

    def _init_lmdb_read(self):
        """Initialize LMDB for reading (same as Neo4jCellDataset)."""
        self.env = lmdb.open(
            osp.join(self.processed_dir, "lmdb"),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
            max_spare_txns=16,
        )

    def _init_lmdb_write(self):
        """Initialize LMDB for writing during processing."""
        os.makedirs(self.processed_dir, exist_ok=True)
        self.env = lmdb.open(
            osp.join(self.processed_dir, "lmdb"), map_size=int(1e12), readonly=False
        )

    def close_lmdb(self):
        """Close LMDB environment."""
        if self.env is not None:
            self.env.close()
            self.env = None

    def process(self):
        """Process step (creates empty LMDB if not exists)."""
        if not osp.exists(osp.join(self.processed_dir, "lmdb")):
            self._init_lmdb_write()
            self.close_lmdb()

    def len(self) -> int:
        """Return number of experiments in LMDB."""
        if self._len is None:
            self._init_lmdb_read()
            with self.env.begin() as txn:
                self._len = txn.stat()["entries"]
            self.close_lmdb()
        return self._len

    def __len__(self) -> int:
        """Return number of experiments in LMDB."""
        return self.len()

    def get(self, idx: int):
        """Get processed data for an experiment (matching Neo4jCellDataset)."""
        if self.env is None:
            self._init_lmdb_read()

        with self.env.begin() as txn:
            serialized_data = txn.get(f"{idx}".encode("utf-8"))
            if serialized_data is None:
                return None
            data_list = json.loads(serialized_data.decode("utf-8"))

            # Reconstruct experiment objects
            data = []
            for item in data_list:
                # Use local experiment type maps that include inference types
                local_experiment_map = EXPERIMENT_TYPE_MAP.copy()
                local_experiment_map["inference"] = InferenceExperiment
                
                local_reference_map = EXPERIMENT_REFERENCE_TYPE_MAP.copy()
                local_reference_map["inference"] = InferenceExperimentReference
                
                experiment_class = local_experiment_map[
                    item["experiment"]["experiment_type"]
                ]
                experiment_reference_class = local_reference_map[
                    item["experiment_reference"]["experiment_reference_type"]
                ]
                reconstructed_data = {
                    "experiment": experiment_class(**item["experiment"]),
                    "experiment_reference": experiment_reference_class(
                        **item["experiment_reference"]
                    ),
                }
                data.append(reconstructed_data)

            # Process through graph processor if available
            if self.graph_processor is not None and self.cell_graph is not None:
                processed_graph = self.graph_processor.process(
                    self.cell_graph, self.phenotype_info, data
                )

                if self.transform is not None:
                    processed_graph = self.transform(processed_graph)

                return processed_graph
            else:
                # Return raw data if no processor
                return data

    def load_experiments_to_lmdb(self, source, source_type: str = "auto"):
        """
        Load experiments from various sources into LMDB.

        Args:
            source: Path to file or list of experiments
            source_type: "pickle", "csv", "experiments", or "auto"
        """
        self._init_lmdb_write()

        # Load experiments based on source type
        if source_type == "pickle" or (
            source_type == "auto"
            and isinstance(source, str)
            and source.endswith(".pkl")
        ):
            experiments = self._load_from_pickle(source)
        elif source_type == "csv" or (
            source_type == "auto"
            and isinstance(source, str)
            and source.endswith(".csv")
        ):
            experiments = self._load_from_csv(source)
        elif isinstance(source, list):
            experiments = source
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        # Store in LMDB
        with self.env.begin(write=True) as txn:
            for idx, exp in enumerate(
                tqdm(experiments, desc="Loading experiments to LMDB")
            ):
                # Create data structure matching Neo4jCellDataset format
                data_list = [
                    {
                        "experiment": exp.model_dump(),
                        "experiment_reference": self._create_default_reference(
                            exp
                        ).model_dump(),
                    }
                ]

                # Serialize and store
                key = str(idx).encode("utf-8")
                value = json.dumps(data_list).encode("utf-8")
                txn.put(key, value)

        # Reset cached length
        self._len = None
        self.close_lmdb()
        print(f"Loaded {len(experiments)} experiments to LMDB")

    def _load_from_pickle(self, pickle_path: str) -> list:
        """Load experiments from pickle file."""
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        # Handle different pickle formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Try common keys
            for key in ["experiments", "data", "final_filtered_triples"]:
                if key in data:
                    if key == "final_filtered_triples":
                        # Convert triples to experiments
                        return [
                            self.create_experiment_from_triple(triple)
                            for triple in data[key]
                        ]
                    return data[key]

        raise ValueError(
            f"Could not extract experiments from pickle file: {pickle_path}"
        )

    def _load_from_csv(self, csv_path: str) -> list:
        """Load experiments from CSV file (expects gene1,gene2,gene3 format)."""
        # Support CSV files with or without headers
        try:
            df = pd.read_csv(csv_path)
            # Check if first row looks like gene names
            if df.iloc[0].astype(str).str.match(r"^Y[A-Z]{2}[0-9]{3}[CW]").all():
                # No header, reload
                df = pd.read_csv(csv_path, header=None)
        except:
            df = pd.read_csv(csv_path, header=None)

        experiments = []

        for _, row in df.iterrows():
            if len(row) >= 3:
                triple = (str(row[0]), str(row[1]), str(row[2]))
                exp = self.create_experiment_from_triple(triple)
                experiments.append(exp)
            else:
                print(f"Skipping row with {len(row)} columns (expected 3)")

        return experiments

    def _create_default_reference(self, experiment) -> FitnessExperimentReference:
        """Create default experiment reference."""
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        )
        environment_reference = experiment.environment.model_copy()
        # Use InferencePhenotype for reference when dealing with inference experiments
        if isinstance(experiment.phenotype, InferencePhenotype):
            phenotype_reference = InferencePhenotype(fitness=1.0, fitness_std=None)
            return InferenceExperimentReference(
                dataset_name=experiment.dataset_name,
                genome_reference=genome_reference,
                environment_reference=environment_reference,
                phenotype_reference=phenotype_reference,
            )
        else:
            phenotype_reference = FitnessPhenotype(fitness=1.0, fitness_std=None)
            return FitnessExperimentReference(
                dataset_name=experiment.dataset_name,
                genome_reference=genome_reference,
                environment_reference=environment_reference,
                phenotype_reference=phenotype_reference,
            )

    @staticmethod
    def create_experiment_from_triple(
        triple: tuple[str, str, str],
        dataset_name: str = "triple_inference",
        environment: Environment = None,
    ) -> FitnessExperiment:
        """
        Create a FitnessExperiment from a triple of gene names.

        Args:
            triple: Tuple of three systematic gene names
            dataset_name: Name for the dataset
            environment: Environment conditions (defaults to standard)

        Returns:
            FitnessExperiment object ready for inference
        """
        if environment is None:
            environment = Environment(
                media=Media(name="YPD", state="solid"),
                temperature=Temperature(value=30.0),
            )

        perturbations = [
            SgaKanMxDeletionPerturbation(
                systematic_gene_name=gene,
                perturbed_gene_name=gene,  # Use systematic name as gene name
                strain_id=f"{gene}_deletion"
            )
            for gene in triple
        ]

        genotype = Genotype(perturbations=perturbations)

        # Use InferencePhenotype for experiments where fitness is unknown
        phenotype = InferencePhenotype()

        return InferenceExperiment(
            dataset_name=dataset_name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )

    @staticmethod
    def load_triples_from_generate_script(pickle_path: str) -> list:
        """
        Load triple combinations from generate_triple_combinations.py output.

        Args:
            pickle_path: Path to pickle file from generate_triple_combinations.py

        Returns:
            List of FitnessExperiment objects
        """
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        experiments = []
        for triple in data["final_filtered_triples"]:
            exp = InferenceDataset.create_experiment_from_triple(triple)
            experiments.append(exp)

        return experiments

    def compute_is_any_perturbed_gene_index(self) -> dict[str, list[int]]:
        """Build index of genes to experiment indices."""
        print("Computing is any perturbed gene index...")
        is_any_perturbed_gene_index = {}

        self._init_lmdb_read()

        try:
            with self.env.begin() as txn:
                cursor = txn.cursor()
                entries = [(key, value) for key, value in cursor]

            for key, value in tqdm(entries, desc="Building gene index"):
                try:
                    idx = int(key.decode())
                    data_list = json.loads(value.decode())
                    for data in data_list:
                        for pert in data["experiment"]["genotype"]["perturbations"]:
                            gene = pert["systematic_gene_name"]
                            if gene not in is_any_perturbed_gene_index:
                                is_any_perturbed_gene_index[gene] = set()
                            is_any_perturbed_gene_index[gene].add(idx)
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    print(f"Error processing entry {key}: {e}")

        finally:
            self.close_lmdb()

        # Convert sets to sorted lists
        return {
            gene: sorted(list(indices))
            for gene, indices in is_any_perturbed_gene_index.items()
        }

    def _compute_phenotype_info(self) -> list[PhenotypeType]:
        """Compute phenotype info from the dataset."""
        experiment_types = set()

        # Check if LMDB exists
        if not osp.exists(osp.join(self.processed_dir, "lmdb")):
            # Default to InferencePhenotype for inference
            return [InferencePhenotype]

        self._init_lmdb_read()

        try:
            with self.env.begin() as txn:
                cursor = txn.cursor()
                # Sample first few entries to determine experiment types
                for i, (key, value) in enumerate(cursor):
                    if i > 10:  # Sample first 10 entries
                        break
                    try:
                        data_list = json.loads(value.decode())
                        for data in data_list:
                            experiment_types.add(data["experiment"]["experiment_type"])
                    except (json.JSONDecodeError, KeyError):
                        continue

            # Determine phenotype classes from experiment types
            phenotype_classes = set()
            # Use local experiment type map that includes inference
            local_experiment_map = EXPERIMENT_TYPE_MAP.copy()
            local_experiment_map["inference"] = InferenceExperiment
            
            for exp_type in experiment_types:
                experiment_class = local_experiment_map.get(exp_type)
                if experiment_class:
                    phenotype_class = experiment_class.__annotations__.get("phenotype")
                    if phenotype_class:
                        phenotype_classes.add(phenotype_class)

            return list(phenotype_classes) if phenotype_classes else [InferencePhenotype]

        finally:
            self.close_lmdb()

    def export_predictions(self, predictions: np.ndarray, output_file: str):
        """Export predictions with experiment metadata."""
        results = []

        self._init_lmdb_read()

        try:
            for idx, pred in enumerate(tqdm(predictions, desc="Exporting predictions")):
                with self.env.begin() as txn:
                    data = json.loads(txn.get(str(idx).encode()).decode())
                    exp = data[0]["experiment"]

                    results.append(
                        {
                            "index": idx,
                            "gene1": exp["genotype"]["perturbations"][0][
                                "systematic_gene_name"
                            ],
                            "gene2": (
                                exp["genotype"]["perturbations"][1][
                                    "systematic_gene_name"
                                ]
                                if len(exp["genotype"]["perturbations"]) > 1
                                else None
                            ),
                            "gene3": (
                                exp["genotype"]["perturbations"][2][
                                    "systematic_gene_name"
                                ]
                                if len(exp["genotype"]["perturbations"]) > 2
                                else None
                            ),
                            "num_perturbations": len(exp["genotype"]["perturbations"]),
                            "dataset_name": exp["dataset_name"],
                            "actual_fitness": exp["phenotype"].get("fitness"),
                            "prediction": float(pred),
                        }
                    )
        finally:
            self.close_lmdb()

        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"Exported {len(results)} predictions to {output_file}")

    def filter_by_genes(self, genes: list[str]) -> list[int]:
        """Get indices of experiments containing any of the specified genes."""
        if self._is_any_perturbed_gene_index is None:
            self._is_any_perturbed_gene_index = (
                self.compute_is_any_perturbed_gene_index()
            )

        indices = set()
        for gene in genes:
            if gene in self._is_any_perturbed_gene_index:
                indices.update(self._is_any_perturbed_gene_index[gene])

        return sorted(list(indices))

    def load_experiments_batch(
        self, sources: list[str], source_types: list[str] = None
    ):
        """Load experiments from multiple sources."""
        all_experiments = []
        for i, source in enumerate(sources):
            source_type = source_types[i] if source_types else "auto"
            if source_type == "pickle":
                experiments = self.load_triples_from_generate_script(source)
            else:
                experiments = self._load_from_source(source, source_type)
            all_experiments.extend(experiments)

        self.load_experiments_to_lmdb(all_experiments)

    def _load_from_source(self, source: str, source_type: str) -> list:
        """Load experiments from a single source."""
        if source_type == "pickle":
            return self._load_from_pickle(source)
        elif source_type == "csv":
            return self._load_from_csv(source)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
