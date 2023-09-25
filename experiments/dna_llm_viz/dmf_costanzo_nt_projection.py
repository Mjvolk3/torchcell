import copy
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
import umap
import matplotlib.pyplot as plt
import numpy as np
import datetime
import lmdb
import pandas as pd
import torch
from attrs import define
from sklearn import experimental
from sortedcontainers import SortedDict, SortedSet
from torch_geometric.data import Batch, Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.data.separate import separate
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph
from tqdm import tqdm

from torchcell.data import Dataset
from torchcell.datasets.fungal_utr_transformer import FungalUtrTransformerDataset
from torchcell.datasets.nucleotide_embedding import BaseEmbeddingDataset
from torchcell.datasets.nucleotide_transformer import NucleotideTransformerDataset
from torchcell.datasets.scerevisiae import (
    DMFCostanzo2016Dataset,
    SMFCostanzo2016Dataset,
)
from torchcell.models import FungalUpDownTransformer, NucleotideTransformer
from torchcell.models.llm import NucleotideModel
from torchcell.models.nucleotide_transformer import NucleotideTransformer
from torchcell.prof import prof, prof_input
from torchcell.sequence import Genome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.datasets import CellDataset
from sklearn.manifold import TSNE

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
    seq_embeddings = nt_dataset

    # Experiments
    experiments = DMFCostanzo2016Dataset(
        preprocess="low_dmf_std",
        root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_1e4"),
        # subset_n=1000000,
    )

    # Cell
    cell_dataset = CellDataset(
        root="data/scerevisiae/cell_1e4",
        genome_gene_set=genome.gene_set,
        seq_embeddings=seq_embeddings,
        experiments=experiments,
    )

    print(cell_dataset)

    # Collect Data for Umaps
    dmf = []
    x_pert_embedding_sum = []
    x_pert_embedding_mean = []

    for cell_data in cell_dataset:
        dmf.append(cell_data.dmf)
        x_pert_embedding_sum.append(cell_data.x_pert.sum(0).numpy())
        x_pert_embedding_mean.append(cell_data.x_pert.mean(0).numpy())

    # Convert to numpy arrays for easier manipulation
    dmf = np.array(dmf)
    x_pert_embedding_sum = np.vstack(x_pert_embedding_sum)
    x_pert_embedding_mean = np.vstack(x_pert_embedding_mean)

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

    # TSNE
    # Function for t-SNE based dimensionality reduction
    def dimensionality_reduction_tsne(x: np.ndarray, perplexity: int, n_iter: int):
        tsne = TSNE(
            n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42
        )  # Create t-SNE object
        return tsne.fit_transform(x)

    # t-SNE plotting with different perplexities and steps
    perplexities = [2, 5, 30, 50, 100]
    # Steps for each perplexity - n_iter
    steps = [5000, 5000, 5000, 5000, 5000]

    # for perplexity, n_iter in zip(perplexities, steps):
    #     for embedding_type in ["sum", "mean"]:
    #         if embedding_type == "sum":
    #             embedding_data = x_pert_embedding_sum
    #         elif embedding_type == "mean":
    #             embedding_data = x_pert_embedding_mean

    #         embedding = dimensionality_reduction_tsne(
    #             embedding_data, perplexity, n_iter
    #         )  # Using t-SNE here

    #         plt.figure()
    #         plt.scatter(
    #             embedding[:, 0], embedding[:, 1], c=dmf, cmap="viridis", s=5, alpha=0.8
    #         )
    #         title = f"DMF Perturbed t-SNE {embedding_type.capitalize()} NT Embeddings (Perplexity: {perplexity}, Steps: {n_iter})"
    #         plt.title(title)
    #         plt.colorbar(label="DMF")

    #         # Save both PNG and PDF formats for each plot
    #         time = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    #         file_name_png = "-".join(title.split(" ")).lower() + "-" + time + ".png"
    #         file_name_pdf = "-".join(title.split(" ")).lower() + "-" + time + ".pdf"
    #         file_path_png = osp.join("notes/assets/images", file_name_png)
    #         file_path_pdf = osp.join("notes/assets/images", file_name_pdf)

    #         plt.savefig(file_path_png)
    #         plt.savefig(file_path_pdf)

    #         plt.close()

    # Umap plotting
    # for dimensionality_reduction_type in ["local", "global", "balanced"]:
    #     for embedding_type in ["sum", "mean"]:
    for dimensionality_reduction_type in ["balanced"]:
        for embedding_type in ["sum"]:
            if embedding_type == "sum":
                embedding_data = x_pert_embedding_sum
            elif embedding_type == "mean":
                embedding_data = x_pert_embedding_mean

            embedding, settings_str = dimensionality_reduction(
                dimensionality_reduction_type, embedding_data
            )

            plt.figure()
            plt.scatter(
                embedding[:, 0], embedding[:, 1], c=dmf, cmap="viridis", s=5, alpha=0.8
            )
            title = f"DMF Perturbed UMAP {embedding_type.capitalize()} NT Embeddings {settings_str}"
            plt.title(title)
            plt.colorbar(label="DMF")

            # Save both PNG and PDF formats for each plot
            time = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
            file_name_png = "-".join(title.split(" ")).lower() + "-" + time + ".png"
            file_name_pdf = "-".join(title.split(" ")).lower() + "-" + time + ".pdf"
            file_path_png = osp.join("notes/assets/images", file_name_png)
            file_path_pdf = osp.join("notes/assets/images", file_name_pdf)

            plt.savefig(file_path_png)
            plt.savefig(file_path_pdf)

            plt.close()
    

if __name__ == "__main__":
    main()
