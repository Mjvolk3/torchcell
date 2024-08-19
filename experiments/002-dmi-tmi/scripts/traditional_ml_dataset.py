# experiments/002-dmi-tmi/scripts/traditional_ml_dataset
# [[experiments.002-dmi-tmi.scripts.traditional_ml_dataset]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/002-dmi-tmi/scripts/traditional_ml_dataset
# Test file: experiments/002-dmi-tmi/scripts/test_traditional_ml_dataset.py


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
    RandomEmbeddingDataset,
)
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.data import Neo4jCellDataset, ExperimentDeduplicator
from torchcell.utils import format_scientific_notation
import torch.distributed as dist
from torch_geometric.utils import unbatch
import socket

log = logging.getLogger(__name__)
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")


def check_data_exists(node_embeddings_path, split):
    return osp.exists(osp.join(node_embeddings_path, split, "X.npy")) and osp.exists(
        osp.join(node_embeddings_path, split, "y.npy")
    )


def save_data_from_dataloader(dataloader, save_path, is_pert, aggregation, split, save_interval_batches=10):
    os.makedirs(save_path, exist_ok=True)
    x_file = osp.join(save_path, "X.npy")
    y_file = osp.join(save_path, "y.npy")

    total_samples = 0
    batch_count = 0

    for batch in tqdm(dataloader):
        x = batch["gene"].x_pert if is_pert else batch["gene"].x
        batch_index = batch["gene"].x_pert_batch if is_pert else batch["gene"].batch
        y = batch["gene"].label_value

        x_unbatched = unbatch(x, batch_index)

        if aggregation == "mean":
            x_agg = torch.stack([data.mean(0) for data in x_unbatched])
        elif aggregation == "sum":
            x_agg = torch.stack([data.sum(0) for data in x_unbatched])
        else:
            raise ValueError("Unsupported aggregation method")

        y = y.view(-1) if y.dim() == 2 else y

        x_agg_np = x_agg.numpy()
        y_np = y.numpy()

        if total_samples == 0:
            np.save(x_file, x_agg_np)
            np.save(y_file, y_np)
        else:
            with open(x_file, 'ab') as f:
                np.save(f, x_agg_np)
            with open(y_file, 'ab') as f:
                np.save(f, y_np)

        total_samples += x_agg_np.shape[0]
        batch_count += 1

        if batch_count >= save_interval_batches:
            print(f"Saved {total_samples} samples so far...")
            batch_count = 0

    print(f"Total samples saved: {total_samples}")

    # Reshape the saved arrays
    X = np.load(x_file, mmap_mode='r')
    y = np.load(y_file, mmap_mode='r')
    
    X_reshaped = X.reshape((total_samples, -1))
    y_reshaped = y.reshape((total_samples,))

    np.save(x_file, X_reshaped)
    np.save(y_file, y_reshaped)

    return total_samples


@hydra.main(
    version_base=None,
    config_path=osp.join(osp.dirname(__file__), "../conf"),
    config_name="traditional_ml_dataset",
)
def main(cfg: DictConfig) -> None:
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    slurm_job_id = os.environ.get("SLURM_JOB_ID", uuid.uuid4())
    hostname = socket.gethostname()
    hostname_slurm_job_id = f"{hostname}-{slurm_job_id}"
    sorted_cfg = json.dumps(wandb_cfg, sort_keys=True)
    hashed_cfg = hashlib.sha256(sorted_cfg.encode("utf-8")).hexdigest()
    group = f"{slurm_job_id}_{hashed_cfg}"
    experiment_dir = osp.join(
        DATA_ROOT, "wandb-experiments", str(hostname_slurm_job_id)
    )
    log.info(f"experiment_dir: {experiment_dir}")
    wandb.init(
        mode="online",
        project=wandb_cfg["wandb"]["project"],
        config=wandb_cfg,
        group=group,
        tags=wandb_cfg["wandb"]["tags"],
        dir=experiment_dir,
    )

    if torch.cuda.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        genome_data_root = osp.join(DATA_ROOT, f"data/sgd/genome_{rank}")
    else:
        genome_data_root = osp.join(DATA_ROOT, "data/sgd/genome")
        rank = 0

    genome = SCerevisiaeGenome(data_root=genome_data_root, overwrite=False)
    genome.drop_chrmt()
    genome.drop_empty_go()

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
        node_embeddings["nt_window_5979_max"] = NucleotideTransformerDataset(
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
            model_name="window_three_prime_5979",
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
        node_embeddings["esm2_t33_650M_UR50D_all"] = Esm2Dataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/esm2_embedding"),
            genome=genome,
            model_name="esm2_t33_650M_UR50D_no_dubious",
        )
    if (
        "esm2_t33_650M_UR50D_no_dubious_uncharacterized"
        in wandb.config.cell_dataset["node_embeddings"]
    ):
        node_embeddings["esm2_t33_650M_UR50D_all"] = Esm2Dataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/esm2_embedding"),
            genome=genome,
            model_name="esm2_t33_650M_UR50D_no_dubious_uncharacterized",
        )
    if (
        "esm2_t33_650M_UR50D_no_uncharacterized"
        in wandb.config.cell_dataset["node_embeddings"]
    ):
        node_embeddings["esm2_t33_650M_UR50D_all"] = Esm2Dataset(
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
            root=osp.join(DATA_ROOT, "data/scerevisiae/random_embedding"), genome=genome
        )
    if "random_100" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["random_100"] = RandomEmbeddingDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/random_embedding"), genome=genome
        )
    if "random_10" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["random_10"] = RandomEmbeddingDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/random_embedding"), genome=genome
        )
    if "random_1" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["random_1"] = RandomEmbeddingDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/random_embedding"), genome=genome
        )
    size_str = format_scientific_notation(float(wandb.config.cell_dataset["size"]))

    with open(
        (
            osp.join(
                ("/").join(osp.dirname(__file__).split("/")[:-1]),
                "queries",
                f"dmi-tmi_{size_str}.cql",
            )
        ),
        "r",
    ) as f:
        query = f.read()

    deduplicator = ExperimentDeduplicator()

    dataset_root = osp.join(
        DATA_ROOT, f"data/torchcell/experiments/002-dmi-tmi/{size_str}"
    )

    cell_dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        genome=genome,
        graphs=graphs,
        node_embeddings=node_embeddings,
        deduplicator=deduplicator,
    )

    data_module = CellDataModule(
        dataset=cell_dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        batch_size=wandb.config.data_module["batch_size"],
        random_seed=42,
        num_workers=wandb.config.data_module["num_workers"],
        pin_memory=wandb.config.data_module["pin_memory"],
    )

    cell_dataset.close_lmdb()

    data_module.setup()

    base_path = osp.join(
        DATA_ROOT, "data/torchcell/experiments/002-dmi-tmi/traditional-ml"
    )
    node_embeddings_path = osp.join(
        base_path,
        "".join(wandb.config.cell_dataset["node_embeddings"]),
        wandb.config.cell_dataset["aggregation"],
    )
    if wandb.config.cell_dataset["is_pert"]:
        node_embeddings_path += "_pert"

    node_embeddings_path = node_embeddings_path + "_" + size_str
    os.makedirs(node_embeddings_path, exist_ok=True)

    all_data_exists = all(
        check_data_exists(node_embeddings_path, split)
        for split in ["train", "val", "test", "all"]
    )

    if all_data_exists:
        print("All necessary data already exists. Skipping this configuration.")
        return

    for split, dataloader in [
        ("train", data_module.train_dataloader()),
        ("val", data_module.val_dataloader()),
        ("test", data_module.test_dataloader()),
        ("all", data_module.all_dataloader()),
    ]:
        save_path = osp.join(node_embeddings_path, split)
        total_samples = save_data_from_dataloader(
            dataloader,
            save_path,
            is_pert=wandb.config.cell_dataset["is_pert"],
            aggregation=wandb.config.cell_dataset["aggregation"],
            split=split,
            save_interval_batches=10  # Adjust this value as needed
        )
        print(f"Total samples saved for {split} split: {total_samples}")

    wandb.finish()


if __name__ == "__main__":
    main()
