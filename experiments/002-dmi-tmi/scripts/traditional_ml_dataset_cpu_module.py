# experiments/002-dmi-tmi/scripts/traditional_ml_dataset
# [[experiments.002-dmi-tmi.scripts.traditional_ml_dataset]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/002-dmi-tmi/scripts/traditional_ml_dataset
# Test file: experiments/002-dmi-tmi/scripts/test_traditional_ml_dataset.py
# FLAG I've commented out some of the imports since things were not working well fast on GilaHyper.

import os
import glob
import os.path as osp
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import wandb
from tqdm import tqdm
from dotenv import load_dotenv
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.data import Neo4jCellDataset, ExperimentDeduplicator
from torchcell.utils import format_scientific_notation
from torchcell.loader import CpuDataModule
from torchcell.graph import SCerevisiaeGraph
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
import multiprocessing as mp

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")


def check_data_exists(save_path):
    return osp.exists(osp.join(save_path, "X.npy")) and osp.exists(
        osp.join(save_path, "y.npy")
    )


def process_item(item, is_pert, aggregation):
    x = item["gene"]["x_pert"].numpy() if is_pert else item["gene"]["x"].numpy()
    y = item["gene"]["label_value"]

    if aggregation == "mean":
        x_agg = np.mean(x, axis=0)
    elif aggregation == "sum":
        x_agg = np.sum(x, axis=0)
    else:
        raise ValueError("Unsupported aggregation method")

    return x_agg, y


def clean_temp_files(save_path):
    for temp_file in glob.glob(osp.join(save_path, "temp_*.bin")):
        try:
            os.remove(temp_file)
            print(f"Removed temporary file: {temp_file}")
        except Exception as e:
            print(f"Error removing temporary file {temp_file}: {e}")


from multiprocessing import Pool


# def save_data_from_dataloader(
#     dataloader, save_path, is_pert, aggregation, split, num_workers
# ):
#     os.makedirs(save_path, exist_ok=True)

#     # Clean up any existing temporary files
#     clean_temp_files(save_path)

#     if check_data_exists(save_path):
#         print(f"Data for {split} split already exists in {save_path}. Skipping.")
#         return 0

#     temp_x_file = osp.join(save_path, f"temp_X_{split}.bin")
#     temp_y_file = osp.join(save_path, f"temp_y_{split}.bin")

#     total_samples = 0
#     x_shape = None

#     with Pool(processes=num_workers) as pool:  # Use all available CPUs
#         with open(temp_x_file, "wb") as fx, open(temp_y_file, "wb") as fy:
#             for batch in tqdm(dataloader, desc=f"Processing {split} split"):
#                 # Map the process_item function to each item in the batch in parallel
#                 results = pool.starmap(
#                     process_item, [(item, is_pert, aggregation) for item in batch]
#                 )

#                 for x_agg, y in results:
#                     if x_shape is None:
#                         x_shape = x_agg.shape

#                     fx.write(x_agg.tobytes())
#                     fy.write(np.array(y).tobytes())
#                     total_samples += 1

#     # Read temporary files and save as .npy
#     with open(temp_x_file, "rb") as fx, open(temp_y_file, "rb") as fy:
#         X = np.frombuffer(fx.read(), dtype=np.float32).reshape(total_samples, -1)
#         y = np.frombuffer(fy.read(), dtype=np.float32)

#     np.save(osp.join(save_path, "X.npy"), X)
#     np.save(osp.join(save_path, "y.npy"), y)

#     # Clean up temporary files
#     os.remove(temp_x_file)
#     os.remove(temp_y_file)

#     return total_samples


def save_data_from_dataloader(dataloader, save_path, is_pert, aggregation, split):
    os.makedirs(save_path, exist_ok=True)

    # Clean up any existing temporary files
    clean_temp_files(save_path)

    if check_data_exists(save_path):
        print(f"Data for {split} split already exists in {save_path}. Skipping.")
        return 0

    temp_x_file = osp.join(save_path, f"temp_X_{split}.bin")
    temp_y_file = osp.join(save_path, f"temp_y_{split}.bin")

    total_samples = 0
    x_shape = None

    with open(temp_x_file, "wb") as fx, open(temp_y_file, "wb") as fy:
        for batch in tqdm(dataloader, desc=f"Processing {split} split"):
            for item in batch:
                x_agg, y = process_item(item, is_pert, aggregation)

                if x_shape is None:
                    x_shape = x_agg.shape

                fx.write(x_agg.tobytes())
                fy.write(np.array(y).tobytes())
                total_samples += 1

    # Read temporary files and save as .npy
    with open(temp_x_file, "rb") as fx, open(temp_y_file, "rb") as fy:
        X = np.frombuffer(fx.read(), dtype=np.float32).reshape(total_samples, -1)
        y = np.frombuffer(fy.read(), dtype=np.float32)

    np.save(osp.join(save_path, "X.npy"), X)
    np.save(osp.join(save_path, "y.npy"), y)

    # Clean up temporary files
    os.remove(temp_x_file)
    os.remove(temp_y_file)

    return total_samples


@hydra.main(
    version_base=None,
    config_path=osp.join(osp.dirname(__file__), "../conf"),
    config_name="traditional_ml_dataset",
)
def main(cfg: DictConfig) -> None:
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(mode="offline", project=wandb_cfg["wandb"]["project"], config=wandb_cfg)

    genome_data_root = osp.join(DATA_ROOT, "data/sgd/genome")
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

    size_str = format_scientific_notation(float(wandb.config.cell_dataset["size"]))

    with open(
        osp.join(
            ("/").join(osp.dirname(__file__).split("/")[:-1]),
            "queries",
            f"dmi-tmi_{size_str}.cql",
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
        graphs=None,
        node_embeddings=node_embeddings,
        deduplicator=deduplicator,
    )

    data_module = CpuDataModule(
        dataset=cell_dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        batch_size=wandb.config.data_module["batch_size"],
        random_seed=42,
        num_workers=wandb.config.data_module["num_workers"],
    )

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

    # for split in ["train", "val", "test", "all"]:
    for split in ["train", "val", "test"]:
        print(f"Processing {split} split")

        if split == "train":
            loader = data_module.train_dataloader()
        elif split == "val":
            loader = data_module.val_dataloader()
        elif split == "test":
            loader = data_module.test_dataloader()
        elif split == "all":
            loader = data_module.all_dataloader()
        else:
            print(f"Unknown split: {split}")
            continue

        save_path = osp.join(node_embeddings_path, split)

        num_samples = save_data_from_dataloader(
            loader,
            save_path,
            is_pert=wandb.config.cell_dataset["is_pert"],
            aggregation=wandb.config.cell_dataset["aggregation"],
            split=split,
            # num_workers=wandb.config.data_module["num_workers"],
        )
        print(f"Saved {num_samples} samples for {split} split")

        # Close the loader to clean up resources
        loader.close()

    wandb.finish()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
