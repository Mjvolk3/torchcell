import copy
import datetime
import json
import logging
import os
import os.path as osp
import pickle
import re
import shutil
import threading
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from os import environ
from typing import List, Optional, Tuple, Union

import lmdb
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import umap
from attrs import define
from sklearn import experimental
from sklearn.manifold import TSNE
from sortedcontainers import SortedDict, SortedSet
from torch_geometric.data import Batch, Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.data.separate import separate
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph
from tqdm import tqdm

from torchcell.data import Dataset
from torchcell.datasets import CellDataset
from torchcell.datasets.embedding import BaseEmbeddingDataset
from torchcell.datasets.fungal_utr_transformer import FungalUtrTransformerDataset
from torchcell.datasets.nucleotide_transformer import NucleotideTransformerDataset
from torchcell.datasets.scerevisiae import (
    DmfCostanzo2016Dataset,
    SmfCostanzo2016Dataset,
)
from torchcell.models import FungalUpDownTransformer, NucleotideTransformer
from torchcell.models.llm import NucleotideModel
from torchcell.models.nucleotide_transformer import NucleotideTransformer
from torchcell.prof import prof, prof_input
from torchcell.sequence import Genome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

log = logging.getLogger(__name__)

plt.style.use("conf/torchcell.mplstyle")


def main():
    # genome
    import os.path as osp

    from dotenv import load_dotenv

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()

    # nucleotide transformer
    nt_dataset = NucleotideTransformerDataset(
        root="data/scerevisiae/nucleotide_transformer_embed",
        genome=genome,
        transformer_model_name="nt_window_5979",
    )

    # Experiments
    experiments = SmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_smf")
    )

    gene_embeds = {}
    for i in range(len(experiments)):
        if experiments[i].phenotype["environment"]["temperature"] == 30:
            try:
                gene = experiments[i].genotype[0]["id"]
                nt_embed = nt_dataset[gene].embeddings["nt_window_5979"]
                if gene not in gene_embeds:
                    gene_embeds[gene] = {"nt": nt_embed}
            except KeyError as e:
                log.warning(f"KeyError: {e}")

    # Old sgd form Gene_Graph
    # Load graph from gpickle
    graph_path = (
        "../Gene_Graph/data/preprocessed/gene_reprs/yeastmine/node_reprs.gpickle"
    )
    # I am not sure if these have been median filled...
    # They must be since they all have values
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    for k in gene_embeds.keys():
        if k in G.nodes:
            try:
                gene_embeds[k]["proteins.median"] = G.nodes[k]["proteins.median"]
            except KeyError as e:
                log.warning(f"KeyError: {e}")
            try:
                gene_embeds[k]["proteins.proteinHalfLife.value"] = G.nodes[k][
                    "proteins.proteinHalfLife.value"
                ]
            except KeyError as e:
                log.warning(f"KeyError: {e}")

    # Plotting here
    def plot_embedding(X, y, title, overlay_variable, save_dir):
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="plasma", s=10, alpha=0.8)
        plt.colorbar(label=overlay_variable)
        plt.title(title)

        time = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        file_name_png = "-".join(title.split(" ")).lower() + "-" + time + ".png"
        file_name_pdf = "-".join(title.split(" ")).lower() + "-" + time + ".pdf"
        file_path_png = osp.join(save_dir, file_name_png)
        file_path_pdf = osp.join(save_dir, file_name_pdf)

        plt.savefig(file_path_png)
        plt.savefig(file_path_pdf)

        plt.close()

    # UMAP function
    # Umap function
    def dimensionality_reduction(dimensionality_reduction_type: str, x: np.ndarray):
        settings_str = ""
        if dimensionality_reduction_type == "local":
            n_neighbors = 5
            min_dist = 0.001
            settings_str = "Local Mode"
        elif dimensionality_reduction_type == "global":
            n_neighbors = 200
            min_dist = 0.5
            settings_str = "Global Mode"
        elif dimensionality_reduction_type == "balanced":
            n_neighbors = 30
            min_dist = 0.1
            settings_str = "Balanced Mode"
        else:
            raise ValueError("Invalid type. Expected 'local', 'global', or 'balanced'.")

        # Create the UMAP transformer
        umap_transformer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=42,  # set the random seed
        )
        return umap_transformer.fit_transform(x), settings_str

    # preprocess gene_embeds for protein median and half life
    gene_embeds_protein_median = gene_embeds.copy()
    gene_embeds_half_life = gene_embeds.copy()

    # Keys to be removed - gene_embeds_protein_median
    keys_to_remove_median = []
    for k, v in gene_embeds_protein_median.items():
        if "proteins.median" in v and v["proteins.median"] >= 200000:
            keys_to_remove_median.append(k)

    # Remove keys - gene_embeds_protein_median
    for k in keys_to_remove_median:
        del gene_embeds_protein_median[k]

    # Keys to be removed - gene_embeds_half_life
    keys_to_remove_half_life = []
    for k, v in gene_embeds_half_life.items():
        if (
            "proteins.proteinHalfLife.value" in v
            and v["proteins.proteinHalfLife.value"] >= 150
        ):
            keys_to_remove_half_life.append(k)

    # Remove keys - gene_embeds_half_life
    for k in keys_to_remove_half_life:
        del gene_embeds_half_life[k]

    # Prepare data for t-SNE and UMAP - proteins median
    X = np.array(
        [torch.Tensor.cpu(x["nt"]).numpy() for x in gene_embeds_protein_median.values()]
    )
    proteins_median = np.array(
        [x.get("proteins.median", np.nan) for x in gene_embeds_protein_median.values()]
    )

    save_dir = "notes/assets/images"  # Update this path as needed

    # t-SNE Plotting
    perplexities = [2, 5, 30, 50, 100]
    for perplexity in perplexities:
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=5000)
        X_tsne = tsne.fit_transform(X)

        plot_title = (
            f"Protein Median Concentration t-SNE plot (perplexity={perplexity})"
        )
        plot_embedding(X_tsne, proteins_median, plot_title, "proteins median", save_dir)

    # UMAP Plotting
    for mode in ["local", "global", "balanced"]:
        # Assuming dimensionality_reduction returns the embedding and some settings string.
        embedding, settings_str = dimensionality_reduction(mode, X)

        plot_title = f"Protein Median Concentration UMAP NT Embeddings {settings_str}"
        plot_embedding(
            embedding, proteins_median, plot_title, "proteins median", save_dir
        )

    # protein half life
    X = np.array(
        [torch.Tensor.cpu(x["nt"]).numpy() for x in gene_embeds_half_life.values()]
    )
    protein_half_life = np.array(
        [
            x.get("proteins.proteinHalfLife.value", np.nan)
            for x in gene_embeds_half_life.values()
        ]
    )

    # t-SNE Plotting
    perplexities = [2, 5, 30, 50, 100]
    for perplexity in perplexities:
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=5000)
        X_tsne = tsne.fit_transform(X)

        plot_title = f"Protein Half-life t-SNE plot (perplexity={perplexity})"
        plot_embedding(
            X_tsne, protein_half_life, plot_title, "protein half-life", save_dir
        )

    # UMAP Plotting
    for mode in ["local", "global", "balanced"]:
        # Assuming dimensionality_reduction returns the embedding and some settings string.
        embedding, settings_str = dimensionality_reduction(mode, X)

        plot_title = f"Protein Half-life UMAP NT Embeddings {settings_str}"
        plot_embedding(
            embedding, protein_half_life, plot_title, "protein half-life", save_dir
        )


if __name__ == "__main__":
    main()
