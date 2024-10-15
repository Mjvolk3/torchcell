import os
import os.path as osp
import numpy as np
import torch
import torch_scatter
from dotenv import load_dotenv
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from torchcell.graph import SCerevisiaeGraph
from torchcell.datamodules import CellDataModule
from torchcell.datamodels.fitness_composite_conversion import CompositeFitnessConverter
from torchcell.datasets.fungal_up_down_transformer import FungalUpDownTransformerDataset
from torchcell.data import MeanExperimentDeduplicator, GenotypeAggregator
from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.data import Neo4jCellDataset
from torchcell.data.neo4j_cell import PhenotypeProcessor
from torchcell.utils import format_scientific_notation

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")


def save_data_from_dataloader(dataloader, save_path, is_pert, aggregation, device):
    os.makedirs(save_path, exist_ok=True)

    X = []
    y_fitness, y_gene_interaction, pert_count = [], [], []

    for batch in tqdm(dataloader, desc="Processing batches"):
        x = batch["gene"].x_pert if is_pert else batch["gene"].x
        batch_index = batch["gene"].x_pert_batch if is_pert else batch["gene"].batch

        x = x.to(device)
        batch_index = batch_index.to(device)

        if aggregation == "mean":
            x_agg = torch_scatter.scatter_mean(x, batch_index, dim=0)
        elif aggregation == "sum":
            x_agg = torch_scatter.scatter_add(x, batch_index, dim=0)
        else:
            raise ValueError("Unsupported aggregation method")

        X.append(x_agg.cpu().numpy())
        y_fitness.append(batch["gene"].fitness.cpu().numpy())
        y_gene_interaction.append(batch["gene"].gene_interaction.cpu().numpy())

        # Handle ids_pert as a list
        sample_pert_counts = [len(ids) for ids in batch["gene"].ids_pert]
        pert_count.append(np.array(sample_pert_counts))

    X = np.concatenate(X, axis=0)
    y_fitness = np.concatenate(y_fitness, axis=0)
    y_gene_interaction = np.concatenate(y_gene_interaction, axis=0)
    pert_count = np.concatenate(pert_count, axis=0)

    np.save(os.path.join(save_path, "X.npy"), X)
    np.save(os.path.join(save_path, "y_fitness.npy"), y_fitness)
    np.save(os.path.join(save_path, "y_gene_interaction.npy"), y_gene_interaction)
    np.save(os.path.join(save_path, "pert_count.npy"), pert_count)


@hydra.main(
    version_base=None, config_path="../conf", config_name="traditional_ml_dataset"
)
def main(cfg: DictConfig) -> None:
    wandb.init(config=OmegaConf.to_container(cfg, resolve=True))

    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()

    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )

    with open("experiments/003-fit-int/queries/001-small-build.cql", "r") as f:
        query = f.read()

    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )

    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        genome=genome,
        graphs={"physical": graph.G_physical, "regulatory": graph.G_regulatory},
        node_embeddings={
            "fudt_3prime": FungalUpDownTransformerDataset(
                root="data/scerevisiae/fudt_embedding",
                genome=genome,
                model_name="species_downstream",
            ),
            "fudt_5prime": FungalUpDownTransformerDataset(
                root="data/scerevisiae/fudt_embedding",
                genome=genome,
                model_name="species_downstream",
            ),
        },
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=PhenotypeProcessor(),
    )

    base_data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=wandb.config["data_module"]["batch_size"],
        random_seed=42,
        num_workers=wandb.config["data_module"]["num_workers"],
        pin_memory=wandb.config["data_module"]["pin_memory"],
    )
    base_data_module.setup()

    if wandb.config["data_module"]["is_perturbation_subset"]:
        data_module = PerturbationSubsetDataModule(
            cell_data_module=base_data_module,
            size=int(wandb.config["data_module"]["perturbation_subset_size"]),
            batch_size=wandb.config["data_module"]["batch_size"],
            num_workers=wandb.config["data_module"]["num_workers"],
            pin_memory=wandb.config["data_module"]["pin_memory"],
            prefetch=wandb.config["data_module"]["prefetch"],
            seed=42,
        )
        data_module.setup()
    else:
        data_module = base_data_module

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size_str = format_scientific_notation(
        wandb.config["data_module"]["perturbation_subset_size"]
    )
    pert_str = "pert_" if wandb.config["cell_dataset"]["is_pert"] else ""
    save_dir = osp.join(
        DATA_ROOT,
        "data/torchcell/experiments/003-fit-int/traditional-ml",
        f"{wandb.config['cell_dataset']['node_embeddings'][0]}",
        f"{wandb.config['cell_dataset']['aggregation']}_{pert_str}{size_str}",
    )

    for split in ["train", "val", "test"]:
        if split == "train":
            dataloader = data_module.train_dataloader()
        elif split == "val":
            dataloader = data_module.val_dataloader()
        else:
            dataloader = data_module.test_dataloader()

        save_path = osp.join(save_dir, split)
        save_data_from_dataloader(
            dataloader,
            save_path,
            wandb.config["cell_dataset"]["is_pert"],
            wandb.config["cell_dataset"]["aggregation"],
            device,
        )

    # Combine train, val, and test for 'all'
    all_save_path = osp.join(save_dir, "all")
    os.makedirs(all_save_path, exist_ok=True)

    for data_type in ["X", "y_fitness", "y_gene_interaction", "pert_count"]:
        all_data = []
        for split in ["train", "val", "test"]:
            split_path = osp.join(save_dir, split, f"{data_type}.npy")
            all_data.append(np.load(split_path))
        all_data = np.concatenate(all_data, axis=0)
        np.save(osp.join(all_save_path, f"{data_type}.npy"), all_data)

    wandb.finish()


if __name__ == "__main__":
    main()
