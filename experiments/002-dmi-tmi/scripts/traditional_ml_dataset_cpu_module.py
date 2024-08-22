import os
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
from torchcell.datasets import RandomEmbeddingDataset

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")


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


def save_data_from_dataloader(dataloader, save_path, is_pert, aggregation, split):
    os.makedirs(save_path, exist_ok=True)

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
    node_embeddings["random_1"] = RandomEmbeddingDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/random_embedding"),
        model_name="random_1",
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
        num_workers=wandb.config.data_module["num_workers"],
    )

    base_path = osp.join(
        DATA_ROOT, "data/torchcell/experiments/002-dmi-tmi/traditional-ml"
    )
    node_embeddings_path = osp.join(
        base_path, "random_1", wandb.config.cell_dataset["aggregation"]
    )
    if wandb.config.cell_dataset["is_pert"]:
        node_embeddings_path += "_pert"

    node_embeddings_path = node_embeddings_path + "_" + size_str
    os.makedirs(node_embeddings_path, exist_ok=True)

    for split in ["train", "val", "test"]:
        print(f"Processing {split} split")

        if split == "train":
            loader = data_module.train_dataloader()
        elif split == "val":
            loader = data_module.val_dataloader()
        elif split == "test":
            loader = data_module.test_dataloader()
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
        )
        print(f"Saved {num_samples} samples for {split} split")

        # Close the loader to clean up resources
        loader.close()

    wandb.finish()


if __name__ == "__main__":
    main()
