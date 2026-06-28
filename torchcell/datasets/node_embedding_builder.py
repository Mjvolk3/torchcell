# torchcell/datasets/node_embedding_builder.py
"""Node embedding builder utility for creating embedding datasets based on configuration."""

import os.path as osp
from typing import Any, cast

from torchcell.datasets.codon_frequency import CodonFrequencyDataset
from torchcell.datasets.codon_language_model import CalmDataset
from torchcell.datasets.esm2 import Esm2Dataset
from torchcell.datasets.fungal_up_down_transformer import FungalUpDownTransformerDataset
from torchcell.datasets.nucleotide_transformer import NucleotideTransformerDataset

# Import from sibling submodules directly, not the package __init__, to avoid a
# circular import that surfaces once imports are alphabetically sorted.
from torchcell.datasets.one_hot_gene import OneHotGeneDataset
from torchcell.datasets.protT5 import ProtT5Dataset
from torchcell.datasets.random_embedding import RandomEmbeddingDataset
from torchcell.datasets.sgd_gene_graph import GraphEmbeddingDataset


class NodeEmbeddingBuilder:
    """Builder class for creating node embedding datasets based on configuration."""

    # Mapping of embedding names to their dataset classes and configurations
    EMBEDDING_CONFIGS = {
        # One-hot gene embedding
        "one_hot_gene": {
            "class": OneHotGeneDataset,
            "root_path": "data/scerevisiae/one_hot_gene_embedding",
            "requires_genome": True,
            "model_name": None,
        },
        # Codon frequency
        "codon_frequency": {
            "class": CodonFrequencyDataset,
            "root_path": "data/scerevisiae/codon_frequency_embedding",
            "requires_genome": True,
            "model_name": None,
        },
        # CALM (Codon Attention Language Model)
        "calm": {
            "class": CalmDataset,
            "root_path": "data/scerevisiae/calm_embedding",
            "requires_genome": True,
            "model_name": "calm",
        },
        # Fungal Up-Down Transformer
        "fudt_downstream": {
            "class": FungalUpDownTransformerDataset,
            "root_path": "data/scerevisiae/fudt_embedding",
            "requires_genome": True,
            "model_name": "species_downstream",
        },
        "fudt_upstream": {
            "class": FungalUpDownTransformerDataset,
            "root_path": "data/scerevisiae/fudt_embedding",
            "requires_genome": True,
            "model_name": "species_upstream",
        },
        # Nucleotide Transformer variants
        "nt_window_5979": {
            "class": NucleotideTransformerDataset,
            "root_path": "data/scerevisiae/nucleotide_transformer_embedding",
            "requires_genome": True,
            "model_name": "nt_window_5979",
        },
        "nt_window_5979_max": {
            "class": NucleotideTransformerDataset,
            "root_path": "data/scerevisiae/nucleotide_transformer_embedding",
            "requires_genome": True,
            "model_name": "nt_window_5979_max",
        },
        "nt_window_three_prime_5979": {
            "class": NucleotideTransformerDataset,
            "root_path": "data/scerevisiae/nucleotide_transformer_embedding",
            "requires_genome": True,
            "model_name": "window_three_prime_5979",
        },
        "nt_window_five_prime_5979": {
            "class": NucleotideTransformerDataset,
            "root_path": "data/scerevisiae/nucleotide_transformer_embedding",
            "requires_genome": True,
            "model_name": "nt_window_five_prime_5979",
        },
        "nt_window_three_prime_300": {
            "class": NucleotideTransformerDataset,
            "root_path": "data/scerevisiae/nucleotide_transformer_embedding",
            "requires_genome": True,
            "model_name": "nt_window_three_prime_300",
        },
        "nt_window_five_prime_1003": {
            "class": NucleotideTransformerDataset,
            "root_path": "data/scerevisiae/nucleotide_transformer_embedding",
            "requires_genome": True,
            "model_name": "nt_window_five_prime_1003",
        },
        # ProtT5 variants
        "prot_T5_all": {
            "class": ProtT5Dataset,
            "root_path": "data/scerevisiae/protT5_embedding",
            "requires_genome": True,
            "model_name": "prot_t5_xl_uniref50_all",
        },
        "prot_T5_no_dubious": {
            "class": ProtT5Dataset,
            "root_path": "data/scerevisiae/protT5_embedding",
            "requires_genome": True,
            "model_name": "prot_t5_xl_uniref50_no_dubious",
        },
        # ESM2 variants
        "esm2_t33_650M_UR50D_all": {
            "class": Esm2Dataset,
            "root_path": "data/scerevisiae/esm2_embedding",
            "requires_genome": True,
            "model_name": "esm2_t33_650M_UR50D_all",
        },
        "esm2_t33_650M_UR50D_no_dubious": {
            "class": Esm2Dataset,
            "root_path": "data/scerevisiae/esm2_embedding",
            "requires_genome": True,
            "model_name": "esm2_t33_650M_UR50D_no_dubious",
        },
        "esm2_t33_650M_UR50D_no_dubious_uncharacterized": {
            "class": Esm2Dataset,
            "root_path": "data/scerevisiae/esm2_embedding",
            "requires_genome": True,
            "model_name": "esm2_t33_650M_UR50D_no_dubious_uncharacterized",
        },
        "esm2_t33_650M_UR50D_no_uncharacterized": {
            "class": Esm2Dataset,
            "root_path": "data/scerevisiae/esm2_embedding",
            "requires_genome": True,
            "model_name": "esm2_t33_650M_UR50D_no_uncharacterized",
        },
        # Graph embeddings
        "normalized_chrom_pathways": {
            "class": GraphEmbeddingDataset,
            "root_path": "data/scerevisiae/sgd_gene_graph_hot",
            "requires_graph": True,
            "model_name": "normalized_chrom_pathways",
        },
        "chrom_pathways": {
            "class": GraphEmbeddingDataset,
            "root_path": "data/scerevisiae/sgd_gene_graph_hot",
            "requires_graph": True,
            "model_name": "chrom_pathways",
        },
        # Random embeddings
        "random_1000": {
            "class": RandomEmbeddingDataset,
            "root_path": "data/scerevisiae/random_embedding",
            "requires_genome": True,
            "model_name": "random_1000",
        },
        "random_100": {
            "class": RandomEmbeddingDataset,
            "root_path": "data/scerevisiae/random_embedding",
            "requires_genome": True,
            "model_name": "random_100",
        },
        "random_10": {
            "class": RandomEmbeddingDataset,
            "root_path": "data/scerevisiae/random_embedding",
            "requires_genome": True,
            "model_name": "random_10",
        },
        "random_1": {
            "class": RandomEmbeddingDataset,
            "root_path": "data/scerevisiae/random_embedding",
            "requires_genome": True,
            "model_name": "random_1",
        },
    }

    @classmethod
    def build(
        cls,
        embedding_names: list[str],
        data_root: str,
        genome: Any,
        graph: Any | None = None,
    ) -> dict[str, Any]:
        """Build node embeddings based on configuration.

        Args:
            embedding_names: List of embedding names to create
            data_root: Root data directory path
            genome: SCerevisiaeGenome instance
            graph: SCerevisiaeGraph instance (required for graph embeddings)

        Returns:
            Dictionary mapping embedding names to dataset instances

        Raises:
            ValueError: If an unknown embedding name is provided or required dependencies are missing
        """
        node_embeddings = {}

        for embedding_name in embedding_names:
            if embedding_name == "learnable":
                # Skip learnable embeddings as they are handled differently
                continue

            if embedding_name not in cls.EMBEDDING_CONFIGS:
                raise ValueError(
                    f"Unknown embedding name: {embedding_name}. "
                    f"Available embeddings: {list(cls.EMBEDDING_CONFIGS.keys())}"
                )

            config = cls.EMBEDDING_CONFIGS[embedding_name]
            dataset_class = cast(Any, config["class"])
            root_path = osp.join(data_root, cast(str, config["root_path"]))

            # Check dependencies
            if config.get("requires_graph") and graph is None:
                raise ValueError(
                    f"Embedding '{embedding_name}' requires a graph instance, but none was provided"
                )

            # Build kwargs for dataset initialization
            kwargs: dict[str, Any] = {"root": root_path}

            if config.get("requires_genome"):
                kwargs["genome"] = genome

            if config.get("requires_graph"):
                kwargs["graph"] = cast(Any, graph).G_gene

            if config.get("model_name"):
                kwargs["model_name"] = config["model_name"]

            # Create the dataset
            node_embeddings[embedding_name] = dataset_class(**kwargs)

        return node_embeddings

    @classmethod
    def check_learnable_embedding(cls, embedding_names: list[str]) -> bool:
        """Check if learnable embedding is requested."""
        return "learnable" in embedding_names
