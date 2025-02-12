# experiments/003-fit-int/scripts/analyze_feature_distributions
# [[experiments.003-fit-int.scripts.analyze_feature_distributions]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/003-fit-int/scripts/analyze_feature_distributions
# Test file: experiments/003-fit-int/scripts/test_analyze_feature_distributions.py


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torchcell.timestamp import timestamp
import os
import os.path as osp
from dotenv import load_dotenv
import torch
from typing import Dict, Any
import pandas as pd

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
from torchcell.data.embedding import BaseEmbeddingDataset

from typing import Any
import logging

load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
MPLSTYLE_PATH = os.getenv("MPLSTYLE_PATH")
DATA_ROOT = os.getenv("DATA_ROOT")
plt.style.use(MPLSTYLE_PATH)
log = logging.getLogger(__name__)


def extract_embeddings(dataset) -> torch.Tensor:
    """Extract embeddings from a dataset."""
    # Get the first key from embeddings dictionary
    first_key = list(dataset._data.embeddings.keys())[0]
    return dataset._data.embeddings[first_key]


def plot_embedding_density(
    embedding: torch.Tensor, name: str, save_dir: str, height: int = 8
) -> None:
    """
    Create a density plot with marginal histograms (a JointGrid)
    using the first two dimensions of the embedding. The joint plot is
    annotated with min, max, mean, and std for each dimension.
    The node embedding name is included in the plot title.
    A legend is added to the joint plot, and marginal histograms are drawn
    with thin white edges to delineate the bars.
    """
    emb_np = embedding.cpu().numpy()
    if emb_np.shape[1] < 2:
        raise ValueError("Embedding must have at least 2 dimensions for density plot.")

    # Use first two dimensions.
    dim0 = emb_np[:, 0]
    dim1 = emb_np[:, 1]
    # Compute statistics.
    stats_text = (
        f"Dim0: min={dim0.min():.2f}, max={dim0.max():.2f}, "
        f"mean={dim0.mean():.2f}, std={dim0.std():.2f}\n"
        f"Dim1: min={dim1.min():.2f}, max={dim1.max():.2f}, "
        f"mean={dim1.mean():.2f}, std={dim1.std():.2f}"
    )
    # Prepare DataFrame.
    df = pd.DataFrame({"dim0": dim0, "dim1": dim1})
    # Create JointGrid.
    g = sns.JointGrid(data=df, x="dim0", y="dim1", space=0, height=height)
    # Plot joint KDE with a label.
    g.plot_joint(
        sns.kdeplot,
        fill=True,
        thresh=0,
        levels=100,
        cmap="rocket",
        clip=((dim0.min(), dim0.max()), (dim1.min(), dim1.max())),
        label="KDE Density",
    )
    # Plot marginal histograms with thin white edges to show bar separation.
    g.plot_marginals(
        sns.histplot,
        color="#03051A",
        alpha=1,
        bins=25,
        edgecolor="white",
        linewidth=0.5,
        label="Histogram",
    )

    # Annotate joint plot with statistics.
    g.ax_joint.text(
        0.05,
        0.95,
        stats_text,
        transform=g.ax_joint.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )
    g.fig.suptitle(f"{name} Embedding Density", y=1.05)
    save_path = osp.join(save_dir, f"embedding_density_{name}_{timestamp()}.png")
    g.fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(g.fig)


def plot_embedding_heatmap(
    embedding: torch.Tensor,
    name: str,
    save_dir: str,
    fig_size: tuple[int, int] = (20, 8),
) -> None:
    """
    Create a heatmap of the full embedding matrix.
    Only the first and last ticks are shown on both axes.
    The node embedding name is included in the plot title.
    """
    emb_np = embedding.cpu().numpy()
    fig, ax = plt.subplots(figsize=fig_size)
    im = ax.imshow(emb_np, aspect="auto", cmap="rocket", vmin=0, vmax=emb_np.max())
    ax.set_xticks([0, emb_np.shape[1] - 1])
    ax.set_xticklabels([0, emb_np.shape[1] - 1])
    ax.set_yticks([0, emb_np.shape[0] - 1])
    ax.set_yticklabels([0, emb_np.shape[0] - 1])
    ax.set_title(f"{name} Embedding Matrix")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Sample Index")
    plt.colorbar(im, ax=ax, label="Value")
    save_path = osp.join(save_dir, f"embedding_heatmap_{name}_{timestamp()}.png")
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def analyze_embeddings(node_embeddings: Dict[str, Any], save_dir: str) -> None:
    """Analyze all embeddings in the dictionary."""
    for name, dataset in node_embeddings.items():
        try:
            embedding = extract_embeddings(dataset)
            print(f"\nAnalyzing {name} embedding:")
            print(f"Shape: {embedding.shape}")
            plot_embedding_density(embedding, name, save_dir)
            plot_embedding_heatmap(embedding, name, save_dir)

        except Exception as e:
            print(f"Error processing {name}: {e}")


def min_max_normalize_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """Forces embedding tensor values into [0,1] range using min-max scaling per feature."""
    # Normalize each feature (column) independently
    normalized_embedding = torch.zeros_like(embedding)
    for i in range(embedding.size(1)):
        feature = embedding[:, i]
        feature_min = feature.min()
        feature_max = feature.max()

        # If feature_min == feature_max, set to 0.5 to avoid div by zero
        if feature_min == feature_max:
            normalized_embedding[:, i] = 0.5
        else:
            normalized_embedding[:, i] = (feature - feature_min) / (
                feature_max - feature_min
            )

    return normalized_embedding


def analyze_embeddings_normalized(
    node_embeddings: dict[str, Any], save_dir: str
) -> None:
    """Analyze all embeddings in the dictionary with normalization."""
    for name, dataset in node_embeddings.items():
        try:
            embedding = extract_embeddings(dataset)
            # Create normalized version
            normalized_embedding = min_max_normalize_embedding(embedding)

            print(f"\nAnalyzing normalized {name} embedding:")
            print(f"Shape: {normalized_embedding.shape}")

            # Plot normalized versions with modified names
            plot_embedding_density(
                normalized_embedding, f"{name}_[0-1]normalized", save_dir
            )
            plot_embedding_heatmap(
                normalized_embedding, f"{name}_[0-1]normalized", save_dir
            )

        except Exception as e:
            print(f"Error processing {name}: {e}")


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
        model_name="nt_window_three_prime_5979",
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


# def main():
#     node_embeddings = build_node_embeddings()

#     # Create save directory if it doesn't exist
#     save_dir = osp.join(ASSET_IMAGES_DIR, "embedding_distributions")
#     os.makedirs(save_dir, exist_ok=True)

#     # Analyze embeddings
#     analyze_embeddings(node_embeddings, save_dir)


def main():
    node_embeddings = build_node_embeddings()

    # Create save directory if it doesn't exist
    save_dir = osp.join(ASSET_IMAGES_DIR)
    os.makedirs(save_dir, exist_ok=True)

    # Analyze both original and normalized embeddings
    analyze_embeddings(node_embeddings, save_dir)
    analyze_embeddings_normalized(node_embeddings, save_dir)


if __name__ == "__main__":
    main()
