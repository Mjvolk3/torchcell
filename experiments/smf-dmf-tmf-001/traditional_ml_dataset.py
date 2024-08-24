# experiments/smf-dmf-tmf-001/traditional_ml_dataset
# [[experiments.smf-dmf-tmf-001.traditional_ml_dataset]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/smf-dmf-tmf-001/traditional_ml_dataset
# Test file: experiments/smf-dmf-tmf-001/test_traditional_ml_dataset.py
import warnings
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)
from tqdm import tqdm
import hashlib
import json
import logging
import os
import os.path as osp
import uuid
import hydra
import torch
import numpy as np
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torchcell.graph import SCerevisiaeGraph
import wandb
from torchcell.datamodules import CellDataModule
from torchcell.datasets import (
    FungalUpDownTransformerDataset,
    OneHotGeneDataset,
    ProtT5Dataset,
    GraphEmbeddingDataset,
    Esm2Dataset,
    NucleotideTransformerDataset,
    CodonFrequencyDataset,
    CalmDataset,
    RandomEmbeddingDataset
)
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.data import Neo4jCellDataset, ExperimentDeduplicator
from torchcell.utils import format_scientific_notation
import torch.distributed as dist
from torch_geometric.utils import unbatch
import os.path as osp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import wandb
import torchcell


log = logging.getLogger(__name__)
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")


def check_data_and_plots_exist(node_embeddings_path, split):
    data_exists = osp.exists(
        osp.join(node_embeddings_path, split, "X.npy")
    ) and osp.exists(osp.join(node_embeddings_path, split, "y.npy"))

    plot_methods = ["umap", "tsne", "pca"]
    plot_types = ["local", "global", "balanced"]

    plots_exist = all(
        osp.exists(
            osp.join(
                ASSET_IMAGES_DIR,
                f"{('-').join(node_embeddings_path.split('/')[-2:])}-{split}-{method}-{embedding_type}_embedding.png",
            )
        )
        for method in plot_methods
        for embedding_type in plot_types
        if method != "pca" or embedding_type == "global"
    )

    return data_exists and plots_exist


def create_embeddings(data, labels, type, method="umap"):
    settings = {
        "local": {"n_neighbors": 5, "min_dist": 0.001, "perplexity": 5},
        "global": {"n_neighbors": 200, "min_dist": 0.5, "perplexity": 50},
        "balanced": {"n_neighbors": 30, "min_dist": 0.1, "perplexity": 30},
    }

    if method == "umap":
        reducer = umap.UMAP(
            n_neighbors=settings[type]["n_neighbors"],
            min_dist=settings[type]["min_dist"],
            n_components=2,
        )
    elif method == "tsne":
        reducer = TSNE(
            n_components=2, perplexity=settings[type]["perplexity"], random_state=42
        )
    elif method == "pca":
        reducer = PCA(n_components=2)

    scaled_data = StandardScaler().fit_transform(data)
    embedding = reducer.fit_transform(scaled_data)

    return embedding


def plot_embedding(embedding, labels, title, image_path, dataset_size):
    style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
    plt.style.use(style_file_path)
    plt.figure(figsize=(12, 9))  # Increase figure size

    # Calculate the size of the dots based on the number of points
    max_size = 200  # Maximum size of the dots
    min_size = 20  # Minimum size of the dots
    num_points = embedding.shape[0]

    # Calculate the dot size inversely proportional to the number of points
    dot_size = max_size / (num_points ** 0.5)
    dot_size = max(min_size, dot_size)  # Ensure the dot size is not smaller than min_size

    scatter = plt.scatter(
        embedding[:, 0], embedding[:, 1], c=labels, cmap="plasma", alpha=0.65, s=dot_size
    )

    # Increase color bar label font size and tick label font size
    cbar = plt.colorbar(scatter, label="Fitness")
    cbar.ax.tick_params(labelsize=28)
    cbar.set_label(label="Fitness", size=32)

    plt.title(title, fontsize=28)  # Increase title font size
    plt.xlabel("Component 1", fontsize=32)  # Increase x-label font size
    plt.ylabel("Component 2", fontsize=32)  # Increase y-label font size
    plt.xticks(fontsize=28)  # Increase x-tick label font size
    plt.yticks(fontsize=28)  # Increase y-tick label font size
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()


def save_data_from_dataloader(dataloader, save_path, is_pert, aggregation, split):
    all_features = []
    all_labels = []
    for batch in tqdm(dataloader):
        x = batch["gene"].x_pert if is_pert else batch["gene"].x
        batch_index = batch["gene"].x_pert_batch if is_pert else batch["gene"].batch
        y = batch["gene"].label_value

        # Unbatch x based on batch indices to separate per graph
        x_unbatched = unbatch(x, batch_index)

        # Apply aggregation per graph
        if aggregation == "mean":
            x_agg = torch.stack([data.mean(0) for data in x_unbatched])
        elif aggregation == "sum":
            x_agg = torch.stack([data.sum(0) for data in x_unbatched])
        else:
            raise ValueError("Unsupported aggregation method")

        # Ensure y is correctly shaped
        y = y.view(-1) if y.dim() == 2 else y

        x_agg_np = x_agg.numpy()
        y_np = y.numpy()

        all_features.append(x_agg_np)
        all_labels.append(y_np)

    # Concatenate all features and labels
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Create a wandb table with raw data
    # TODO could log to explore projector
    # table = wandb.Table(
    #     columns=[f"feature_{i}" for i in range(all_features.shape[1])] + ["label"],
    #     data=np.concatenate((all_features, all_labels.reshape(-1, 1)), axis=1),
    # )
    # wandb.log({f"{split}_embeddings": table})

    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Save the arrays
    np.save(osp.join(save_path, "X.npy"), all_features)
    np.save(osp.join(save_path, "y.npy"), all_labels)

    return all_features, all_labels


@hydra.main(version_base=None, config_path="conf", config_name="traditional_ml_dataset")
def main(cfg: DictConfig) -> None:
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    slurm_job_id = os.environ.get("SLURM_JOB_ID", uuid.uuid4())
    sorted_cfg = json.dumps(wandb_cfg, sort_keys=True)
    hashed_cfg = hashlib.sha256(sorted_cfg.encode("utf-8")).hexdigest()
    group = f"{slurm_job_id}_{hashed_cfg}"
    wandb.init(
        mode="online",
        project=wandb_cfg["wandb"]["project"],
        config=wandb_cfg,
        group=group,
        tags=wandb_cfg["wandb"]["tags"],
    )

    # Handle sql genome access error for ddp
    if torch.cuda.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        genome_data_root = osp.join(DATA_ROOT, f"data/sgd/genome_{rank}")
    else:
        genome_data_root = osp.join(DATA_ROOT, "data/sgd/genome")
        rank = 0

    # Get reference genome
    genome = SCerevisiaeGenome(data_root=genome_data_root, overwrite=False)
    genome.drop_chrmt()
    genome.drop_empty_go()

    # Graph data
    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )
    graphs = {}
    if wandb.config.cell_dataset["graphs"] is None:
        graphs = None
    elif "physical" in wandb.config.cell_dataset["graphs"]:
        graphs = {"physical": graph.G_physical}
    elif "regulatory" in wandb.config.cell_dataset["graphs"]:
        graphs = {"regulatory": graph.G_regulatory}

    # Node embedding datasets
    # Node embedding datasets
    node_embeddings = {}
    # Node embedding datasets
    node_embeddings = {}
    # one hot gene - transductive
    if "one_hot_gene" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["one_hot_gene"] = OneHotGeneDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/one_hot_gene_embedding"),
            genome=genome,
        )
    # codon frequency
    if "codon_frequency" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["codon_frequency"] = CodonFrequencyDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/codon_frequency_embedding"),
            genome=genome,
        )
    # codon embedding
    if "calm" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["calm"] = CalmDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/calm_embedding"),
            genome=genome,
            model_name="calm",
        )
    # fudt
    if "fudt_downstream" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["fudt_downstream"] = FungalUpDownTransformerDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/fudt_embedding"),
            genome=genome,
            model_name="species_downstream",
        )

    if "fudt_upstream" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["fudt_upstream"] = FungalUpDownTransformerDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/fudt_embedding"),
            genome=genome,
            model_name="species_upstream",
        )
    # nucleotide transformer
    if "nt_window_5979" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["nt_window_5979"] = NucleotideTransformerDataset(
            root=osp.join(
                DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embedding"
            ),
            genome=genome,
            model_name="nt_window_5979",
        )
    if "nt_window_5979_max" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["nt_window_5979_max"] = NucleotideTransformerDataset(
            root=osp.join(
                DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embedding"
            ),
            genome=genome,
            model_name="nt_window_5979_max",
        )
    if "nt_window_three_prime_5979" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["nt_window_three_prime_5979"] = NucleotideTransformerDataset(
            root=osp.join(
                DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embedding"
            ),
            genome=genome,
            model_name="nt_window_three_prime_5979",
        )
    if "nt_window_five_prime_5979" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["nt_window_five_prime_5979"] = NucleotideTransformerDataset(
            root=osp.join(
                DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embedding"
            ),
            genome=genome,
            model_name="nt_window_five_prime_5979",
        )
    if "nt_window_three_prime_300" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["nt_window_three_prime_300"] = NucleotideTransformerDataset(
            root=osp.join(
                DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embedding"
            ),
            genome=genome,
            model_name="nt_window_three_prime_300",
        )
    if "nt_window_five_prime_1003" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["nt_window_five_prime_1003"] = NucleotideTransformerDataset(
            root=osp.join(
                DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embedding"
            ),
            genome=genome,
            model_name="nt_window_five_prime_1003",
        )
    # protT5
    if "prot_T5_all" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["prot_T5_all"] = ProtT5Dataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/protT5_embedding"),
            genome=genome,
            model_name="prot_t5_xl_uniref50_all",
        )
    if "prot_T5_no_dubious" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["prot_T5_no_dubious"] = ProtT5Dataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/protT5_embedding"),
            genome=genome,
            model_name="prot_t5_xl_uniref50_no_dubious",
        )
    # esm
    if "esm2_t33_650M_UR50D_all" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["esm2_t33_650M_UR50D_all"] = Esm2Dataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/esm2_embedding"),
            genome=genome,
            model_name="esm2_t33_650M_UR50D_all",
        )
    if "esm2_t33_650M_UR50D_no_dubious" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["esm2_t33_650M_UR50D_no_dubious"] = Esm2Dataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/esm2_embedding"),
            genome=genome,
            model_name="esm2_t33_650M_UR50D_no_dubious",
        )
    if (
        "esm2_t33_650M_UR50D_no_dubious_uncharacterized"
        in wandb.config.cell_dataset["node_embeddings"]
    ):
        node_embeddings["esm2_t33_650M_UR50D_no_dubious_uncharacterized"] = Esm2Dataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/esm2_embedding"),
            genome=genome,
            model_name="esm2_t33_650M_UR50D_no_dubious_uncharacterized",
        )
    if (
        "esm2_t33_650M_UR50D_no_uncharacterized"
        in wandb.config.cell_dataset["node_embeddings"]
    ):
        node_embeddings["esm2_t33_650M_UR50D_no_uncharacterized"] = Esm2Dataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/esm2_embedding"),
            genome=genome,
            model_name="esm2_t33_650M_UR50D_no_uncharacterized",
        )
    # sgd_gene_graph
    if "normalized_chrom_pathways" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["normalized_chrom_pathways"] = GraphEmbeddingDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/sgd_gene_graph_hot"),
            graph=graph.G_gene,
            model_name="normalized_chrom_pathways",
        )
    if "chrom_pathways" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["chrom_pathways"] = GraphEmbeddingDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/sgd_gene_graph_hot"),
            graph=graph.G_gene,
            model_name="chrom_pathways",
        )
    # random
    if "random_1000" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["random_1000"] = RandomEmbeddingDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/random_embedding"),
            genome=genome,
            model_name="random_1000",
        )
    if "random_100" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["random_100"] = RandomEmbeddingDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/random_embedding"),
            genome=genome,
            model_name="random_100",
        )
    if "random_10" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["random_10"] = RandomEmbeddingDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/random_embedding"),
            genome=genome,
            model_name="random_10",
        )
    if "random_1" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["random_1"] = RandomEmbeddingDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/random_embedding"),
            genome=genome,
            model_name="random_1",
        )

    # Experiments
    with open((osp.join(osp.dirname(__file__), "query.cql")), "r") as f:
        query = f.read()

    deduplicator = ExperimentDeduplicator()

    # Convert max_size to float, then format it in concise scientific notation
    max_size_str = format_scientific_notation(
        float(wandb.config.cell_dataset["max_size"])
    )
    dataset_root = osp.join(
        DATA_ROOT, f"data/torchcell/experiments/smf-dmf-tmf_{max_size_str}"
    )

    cell_dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        genome=genome,
        graphs=graphs,
        node_embeddings=node_embeddings,
        deduplicator=deduplicator,
        max_size=int(wandb.config.cell_dataset["max_size"]),
    )

    # Instantiate data module
    data_module = CellDataModule(
        dataset=cell_dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        batch_size=wandb.config.data_module["batch_size"],
        random_seed=42,
        num_workers=wandb.config.data_module["num_workers"],
        pin_memory=wandb.config.data_module["pin_memory"],
    )

    # Anytime data is accessed lmdb must be closed.
    cell_dataset.close_lmdb()

    # Assuming you have initialized your data module and setup the dataloaders
    data_module.setup()  # Prepare data splits

    base_path = osp.join(
        DATA_ROOT, "data/torchcell/experiments/smf-dmf-tmf-traditional-ml"
    )
    node_embeddings_path = osp.join(
        base_path,
        "".join(wandb.config.cell_dataset["node_embeddings"]),
        wandb.config.cell_dataset["aggregation"],
    )
    if wandb.config.cell_dataset["is_pert"]:
        node_embeddings_path += "_pert"

    node_embeddings_path = node_embeddings_path + "_" + max_size_str
    os.makedirs(node_embeddings_path, exist_ok=True)

    # Check if data and plots exist for all splits
    all_data_and_plots_exist = all(
        check_data_and_plots_exist(node_embeddings_path, split)
        for split in ["train", "val", "test", "all"]
    )

    if all_data_and_plots_exist:
        print(
            "All necessary data and plots already exist. Skipping this configuration."
        )
        return

    for split, dataloader in [
        ("train", data_module.train_dataloader()),
        ("val", data_module.val_dataloader()),
        ("test", data_module.test_dataloader()),
        ("all", data_module.all_dataloader()),
    ]:
        save_path = osp.join(node_embeddings_path, split)
        features, labels = save_data_from_dataloader(
            dataloader,
            save_path,
            is_pert=wandb.config.cell_dataset["is_pert"],
            aggregation=wandb.config.cell_dataset["aggregation"],
            split=split,
        )
        if wandb.config["is_plot_embeddings"]:
            # Generate and log embeddings
            for method in ["umap", "tsne"]:
                for embedding_type in ["local", "global", "balanced"]:
                    print("Creating embeddings...")
                    embedding = create_embeddings(
                        np.array(features), np.array(labels), embedding_type, method
                    )
                    title = f"{('-').join(node_embeddings_path.split('/')[-2:])}-{split}-{method}-{embedding_type}_embedding"
                    image_path = osp.join(ASSET_IMAGES_DIR, title) + ".png"

                    print("plotting embedding...")
                    dataset_size = len(dataloader.dataset)  # Get the dataset size
                    plot_embedding(embedding, labels, title, image_path, dataset_size)
                    wandb.log(
                        {f"{split}_{method}_{embedding_type}": wandb.Image(image_path)}
                    )

            # Generate PCA plot for each split
            embedding = create_embeddings(
                np.array(features), np.array(labels), type="global", method="pca"
            )
            title = (
                f"{('-').join(node_embeddings_path.split('/')[-2:])}-{split}-pca_embedding"
            )
            image_path = osp.join(ASSET_IMAGES_DIR, title) + ".png"
            dataset_size = len(dataloader.dataset)  # Get the dataset size
            plot_embedding(embedding, labels, title, image_path, dataset_size)
            wandb.log({f"{split}_pca": wandb.Image(image_path)})

    wandb.finish()


if __name__ == "__main__":
    main()
