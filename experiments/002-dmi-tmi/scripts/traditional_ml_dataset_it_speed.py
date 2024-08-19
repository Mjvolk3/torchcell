import os
import os.path as osp
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from tqdm import tqdm
from dotenv import load_dotenv
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.data import Neo4jCellDataset, ExperimentDeduplicator
from torchcell.datamodules import CellDataModule
from torchcell.utils import format_scientific_notation

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")

@hydra.main(
    version_base=None,
    config_path=osp.join(osp.dirname(__file__), "../conf"),
    config_name="traditional_ml_dataset",
)
def main(cfg: DictConfig) -> None:
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(
        mode="disabled",
        project=wandb_cfg["wandb"]["project"],
        config=wandb_cfg,
    )

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

    cell_dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        genome=genome,
        graphs=None,
        node_embeddings={},
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

    for split, dataloader in [
        ("train", data_module.train_dataloader()),
        ("val", data_module.val_dataloader()),
        ("test", data_module.test_dataloader()),
    ]:
        print(f"Iterating through {split} split")
        total_samples = 0
        for _ in tqdm(dataloader, desc=f"{split} iteration"):
            total_samples += 1
        print(f"Total batches in {split} split: {total_samples}")

    wandb.finish()

if __name__ == "__main__":
    main()