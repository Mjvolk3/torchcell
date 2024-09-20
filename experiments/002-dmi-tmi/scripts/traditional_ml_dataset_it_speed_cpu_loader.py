# FLAG I've commented out some of the imports since things were not working well fast on GilaHyper.

import os
import os.path as osp
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from tqdm import tqdm
from dotenv import load_dotenv
from torchcell.sequence.genome.scerevisiae.S288C import SCerevisiaeGenome
from torchcell.data import Neo4jCellDataset, ExperimentDeduplicator
from torchcell.utils import format_scientific_notation
from torchcell.loader import CpuDataModule
from torchcell.datasets import RandomEmbeddingDataset
import multiprocessing as mp

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")


@hydra.main(
    version_base=None,
    config_path=osp.join(osp.dirname(__file__), "../conf"),
    config_name="traditional_ml_dataset",
)
def main(cfg: DictConfig) -> None:
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(mode="disabled", project=wandb_cfg["wandb"]["project"], config=wandb_cfg)

    genome_data_root = osp.join(DATA_ROOT, "data/sgd/genome")
    genome = SCerevisiaeGenome(data_root=genome_data_root, overwrite=False)
    genome.drop_chrmt()
    genome.drop_empty_go()

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

    node_embeddings = {}
    rand_embed_str = "random_100"
    node_embeddings[rand_embed_str] = RandomEmbeddingDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/random_embedding"),
        model_name=rand_embed_str,
        genome=genome,
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
        num_workers=wandb.config.data_module["num_workers"],  # Using 10 workers as in your original script
    )

    for split in ["train", "val", "test", "all"]:
        print(f"Iterating through {split} split")

        if split == "train":
            loader = data_module.train_dataloader()
        elif split == "val":
            loader = data_module.val_dataloader()
        elif split == "test":
            loader = data_module.test_dataloader()
        else:
            loader = data_module.all_dataloader()

        total_samples = 0
        for _ in tqdm(loader, desc=f"{split} iteration", total=len(loader)):
            total_samples += 1

        print(f"Total batches in {split} split: {total_samples}")

        # Make sure to close the loader
        loader.close()

    wandb.finish()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
