import logging
import os
import os.path as osp
from dotenv import load_dotenv
from torchcell.graph import SCerevisiaeGraph
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.datasets import (
    FungalUpDownTransformerDataset,
    OneHotGeneDataset,
    ProtT5Dataset,
    GraphEmbeddingDataset,
    Esm2Dataset,
    NucleotideTransformerDataset,
    CodonFrequencyDataset,
    CalmDataset,
    RandomEmbeddingDataset,
)
from typing import Any

log = logging.getLogger(__name__)
load_dotenv()
DATA_ROOT: str = os.getenv("DATA_ROOT") or ""


def build_node_embeddings() -> dict[str, Any]:
    genome_data_root: str = osp.join(DATA_ROOT, "data/sgd/genome")
    genome = SCerevisiaeGenome(data_root=genome_data_root, overwrite=False)
    # Optionally drop chromosomes if needed:
    # genome.drop_chrmt()
    genome.drop_empty_go()
    # Build the graph for gene embeddings.
    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )
    node_embeddings: dict[str, Any] = {}

    # one hot gene
    node_embeddings["one_hot_gene"] = OneHotGeneDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/one_hot_gene_embedding"),
        genome=genome,
    )
    # codon frequency
    node_embeddings["codon_frequency"] = CodonFrequencyDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/codon_frequency_embedding"),
        genome=genome,
    )
    # codon embedding
    node_embeddings["calm"] = CalmDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/calm_embedding"),
        genome=genome,
        model_name="calm",
    )
    # FUDT datasets
    node_embeddings["fudt_downstream"] = FungalUpDownTransformerDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/fudt_embedding"),
        genome=genome,
        model_name="species_downstream",
    )
    node_embeddings["fudt_upstream"] = FungalUpDownTransformerDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/fudt_embedding"),
        genome=genome,
        model_name="species_upstream",
    )
    # Nucleotide transformer datasets
    node_embeddings["nt_window_5979"] = NucleotideTransformerDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embedding"),
        genome=genome,
        model_name="nt_window_5979",
    )
    node_embeddings["nt_window_5979_max"] = NucleotideTransformerDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embedding"),
        genome=genome,
        model_name="nt_window_5979_max",
    )
    node_embeddings["nt_window_three_prime_5979"] = NucleotideTransformerDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embedding"),
        genome=genome,
        model_name="window_three_prime_5979",
    )
    node_embeddings["nt_window_five_prime_5979"] = NucleotideTransformerDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embedding"),
        genome=genome,
        model_name="nt_window_five_prime_5979",
    )
    node_embeddings["nt_window_three_prime_300"] = NucleotideTransformerDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embedding"),
        genome=genome,
        model_name="nt_window_three_prime_300",
    )
    node_embeddings["nt_window_five_prime_1003"] = NucleotideTransformerDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embedding"),
        genome=genome,
        model_name="nt_window_five_prime_1003",
    )
    # ProtT5 datasets
    node_embeddings["prot_T5_all"] = ProtT5Dataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/protT5_embedding"),
        genome=genome,
        model_name="prot_t5_xl_uniref50_all",
    )
    node_embeddings["prot_T5_no_dubious"] = ProtT5Dataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/protT5_embedding"),
        genome=genome,
        model_name="prot_t5_xl_uniref50_no_dubious",
    )
    # ESM2 datasets with unique keys
    node_embeddings["esm2_t33_650M_UR50D_all"] = Esm2Dataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/esm2_embedding"),
        genome=genome,
        model_name="esm2_t33_650M_UR50D_all",
    )
    node_embeddings["esm2_t33_650M_UR50D_no_dubious"] = Esm2Dataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/esm2_embedding"),
        genome=genome,
        model_name="esm2_t33_650M_UR50D_no_dubious",
    )
    node_embeddings["esm2_t33_650M_UR50D_no_dubious_uncharacterized"] = Esm2Dataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/esm2_embedding"),
        genome=genome,
        model_name="esm2_t33_650M_UR50D_no_dubious_uncharacterized",
    )
    node_embeddings["esm2_t33_650M_UR50D_no_uncharacterized"] = Esm2Dataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/esm2_embedding"),
        genome=genome,
        model_name="esm2_t33_650M_UR50D_no_uncharacterized",
    )
    # SGD gene graph datasets
    node_embeddings["normalized_chrom_pathways"] = GraphEmbeddingDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/sgd_gene_graph_hot"),
        graph=graph.G_gene,
        model_name="normalized_chrom_pathways",
    )
    node_embeddings["chrom_pathways"] = GraphEmbeddingDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/sgd_gene_graph_hot"),
        graph=graph.G_gene,
        model_name="chrom_pathways",
    )
    # Random embeddings
    node_embeddings["random_1000"] = RandomEmbeddingDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/random_embedding"),
        genome=genome,
        model_name="random_1000",
    )
    node_embeddings["random_100"] = RandomEmbeddingDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/random_embedding"),
        genome=genome,
        model_name="random_100",
    )
    node_embeddings["random_10"] = RandomEmbeddingDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/random_embedding"),
        genome=genome,
        model_name="random_10",
    )
    node_embeddings["random_1"] = RandomEmbeddingDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/random_embedding"),
        genome=genome,
        model_name="random_1",
    )
    return node_embeddings


def check_node_embeddings_length(
    node_embeddings: dict[str, Any], expected_length: int = 6607
) -> None:
    """Print node embeddings whose length is not equal to expected_length."""
    for name, dataset in node_embeddings.items():
        try:
            length = len(dataset)
        except Exception as err:
            print(f"Could not get length for '{name}': {err}")
            continue
        if length != expected_length:
            print(
                f"Node embedding '{name}' has length {length} "
                f"(expected {expected_length})."
            )


def main() -> None:
    node_embeddings = build_node_embeddings()
    check_node_embeddings_length(node_embeddings)


if __name__ == "__main__":
    main()
