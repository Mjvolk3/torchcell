import hashlib
import json
import logging
import os
import os.path as osp
import uuid
from torch.nn import ModuleDict
import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torchcell.graph import SCerevisiaeGraph
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from lightning.pytorch.callbacks import GradientAccumulationScheduler
import wandb
from torchcell.losses.multi_dim_nan_tolerant import CombinedCELoss
from torch_geometric.transforms import Compose
from torchcell.transforms.regression_to_classification import (
    LabelBinningTransform,
    LabelNormalizationTransform,
    InverseCompose,
)
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
from torchcell.models.hetero_gnn_pool import HeteroGnnPool
from torchcell.trainers.fit_int_hetero_gnn_pool_binary_classification import (
    ClassificationTask,
)
from torchcell.utils import format_scientific_notation
import torch.distributed as dist
import socket
from torchcell.datamodels.fitness_composite_conversion import CompositeFitnessConverter
from torchcell.data import MeanExperimentDeduplicator
from torchcell.data import GenotypeAggregator
from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
from torchcell.data import Neo4jCellDataset
from torchcell.data.neo4j_cell import PhenotypeProcessor
from lightning.pytorch.profilers import AdvancedProfiler
from typing import Any
from lightning.pytorch.profilers import AdvancedProfiler
import cProfile
from torchcell.transforms.hetero_to_dense import HeteroToDense

# from torchcell.profilers.pytorch import PyTorchProfiler


log = logging.getLogger(__name__)
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")


class CustomAdvancedProfiler(AdvancedProfiler):
    def __init__(self, dirpath: str, filename: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dirpath = dirpath
        self.filename = filename
        self.profiler = cProfile.Profile()

    def start(self, action_name: str) -> None:
        super().start(action_name)
        self.profiler.enable()

    def stop(self, action_name: str) -> None:
        self.profiler.disable()
        super().stop(action_name)

    def summary(self) -> str:
        os.makedirs(self.dirpath, exist_ok=True)
        file_path = os.path.join(self.dirpath, self.filename)
        self.profiler.dump_stats(file_path)
        return f"Profiler output saved to {file_path}"


@hydra.main(
    version_base=None,
    config_path=osp.join(osp.dirname(__file__), "../conf"),
    config_name="hetero_gnn_pool",
)
def main(cfg: DictConfig) -> None:
    print("Starting HeteroGnnPool 🎻")
    os.environ["WANDB__SERVICE_WAIT"] = "600"
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    print("wandb_cfg", wandb_cfg)
    slurm_job_id = os.environ.get("SLURM_JOB_ID", str(uuid.uuid4()))
    hostname = socket.gethostname()
    hostname_slurm_job_id = f"{hostname}-{slurm_job_id}"
    sorted_cfg = json.dumps(wandb_cfg, sort_keys=True)
    hashed_cfg = hashlib.sha256(sorted_cfg.encode("utf-8")).hexdigest()
    group = f"{hostname_slurm_job_id}_{hashed_cfg}"
    experiment_dir = osp.join(DATA_ROOT, "wandb-experiments", group)
    os.makedirs(experiment_dir, exist_ok=True)
    wandb.init(
        mode="offline",  # "online", "offline", "disabled"
        project=wandb_cfg["wandb"]["project"],
        config=wandb_cfg,
        group=group,
        tags=wandb_cfg["wandb"]["tags"],
        dir=experiment_dir,
    )

    # Initialize the WandbLogger
    wandb_logger = WandbLogger(project=wandb_cfg["wandb"]["project"], log_model=True)

    # Handle sql genome access error for ddp
    if torch.cuda.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        genome_data_root = osp.join(DATA_ROOT, f"data/sgd/genome_{rank}")
    else:
        # Fallback to default DATA_ROOT if not running in distributed mode or no GPU available
        genome_data_root = osp.join(DATA_ROOT, "data/sgd/genome")
        rank = 0  #

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
    if "physical" in wandb.config.cell_dataset["graphs"]:
        graphs["physical"] = graph.G_physical
    if "regulatory" in wandb.config.cell_dataset["graphs"]:
        graphs["regulatory"] = graph.G_regulatory

    # Node embedding datasets
    node_embeddings = {}

    # one hot gene
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

    print("=============")
    print("node.embeddings")
    print(node_embeddings)
    print("=============")

    with open("experiments/003-fit-int/queries/001-small-build.cql", "r") as f:
        query = f.read()
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )

    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=graphs,
        node_embeddings=node_embeddings,
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=PhenotypeProcessor(),
    )
    ### TODO add changes to binary classification - start
    # Configure transforms for binary classification
    norm_config = {"strategy": "minmax"}
    bin_config = {
        "num_bins": wandb.config.transforms["num_bins"],
        "strategy": wandb.config.transforms["strategy"],
        "store_continuous": wandb.config.transforms["store_continuous"],
        "label_type": wandb.config.transforms["label_type"],
    }

    # Get all label columns (excluding 'index')
    label_columns = [col for col in dataset.label_df.columns if col != "index"]

    # Create configs for all label columns
    norm_configs = {col: norm_config for col in label_columns}
    bin_configs = {col: bin_config for col in label_columns}

    print("\nProcessing labels:")
    for label in label_columns:
        print(f"- {label}")

    # Create transforms
    normalize_transform = (
        LabelNormalizationTransform(dataset, norm_configs) if norm_configs else None
    )
    binning_transform = LabelBinningTransform(dataset, bin_configs, normalize_transform)

    # Create forward transform
    if normalize_transform is not None:
        forward_transform = Compose([normalize_transform, binning_transform])
    else:
        forward_transform = binning_transform

    inverse_transform = InverseCompose(forward_transform)
    dataset.transform = forward_transform
    # Model output dimension is determined by num_bins
    target_dim = wandb.config.transforms["num_bins"]

    ### TODO add changes to binary classification - end

    # Base Module
    seed = 42
    data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=wandb.config.data_module["batch_size"],
        random_seed=seed,
        num_workers=wandb.config.data_module["num_workers"],
        pin_memory=wandb.config.data_module["pin_memory"],
    )
    data_module.setup()

    # Subset Module
    if wandb.config.data_module["is_perturbation_subset"]:
        data_module = PerturbationSubsetDataModule(
            cell_data_module=data_module,
            size=int(wandb.config.data_module["perturbation_subset_size"]),
            batch_size=wandb.config.data_module["batch_size"],
            num_workers=wandb.config.data_module["num_workers"],
            pin_memory=wandb.config.data_module["pin_memory"],
            prefetch=wandb.config.data_module["prefetch"],
            seed=seed,
        )
        data_module.setup()

    # Anytime data is accessed lmdb must be closed.
    input_dim = dataset.num_features["gene"]
    # max_num_nodes = len(dataset.gene_set)
    dataset.close_lmdb()

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(device)

    num_devices = torch.cuda.device_count()
    if wandb.config.trainer["devices"] != "auto":
        devices = wandb.config.trainer["devices"]
    elif wandb.config.trainer["devices"] == "auto" and num_devices > 0:
        devices = num_devices
    elif wandb.config.trainer["devices"] == "auto" and num_devices == 0:
        # if there are no GPUs available, use 1 CPU
        devices = 1

    # Convert graph names to edge types
    edge_types = [
        ("gene", f"{name}_interaction", "gene")
        for name in wandb.config.cell_dataset["graphs"]
    ]

    # Layer config based on conv_type
    conv_type = wandb.config.model["conv_type"]
    layer_config = {}

    if conv_type == "GCN":
        layer_config = {
            "bias": wandb.config.model["gcn_bias"],
            "add_self_loops": wandb.config.model["gcn_add_self_loops"],
            "normalize": wandb.config.model["gcn_normalize"],
            "is_skip_connection": wandb.config.model["gcn_is_skip_connection"],
        }
    elif conv_type == "GAT":
        layer_config = {
            "heads": wandb.config.model["gat_heads"],
            "concat": wandb.config.model["gat_concat"],
            "bias": wandb.config.model["gat_bias"],
            "add_self_loops": wandb.config.model["gat_add_self_loops"],
            "share_weights": wandb.config.model["gat_share_weights"],
            "is_skip_connection": wandb.config.model["gat_is_skip_connection"],
            "dropout": wandb.config.model["dropout"],
        }
    elif conv_type == "Transformer":
        layer_config = {
            "heads": wandb.config.model["transformer_heads"],
            "concat": wandb.config.model["transformer_concat"],
            "beta": wandb.config.model["transformer_beta"],
            "bias": wandb.config.model["transformer_bias"],
            "root_weight": wandb.config.model["transformer_root_weight"],
            "add_self_loops": wandb.config.model["transformer_add_self_loops"],
            "edge_dim": wandb.config.model["transformer_edge_dim"],
            "dropout": wandb.config.model["dropout"],
        }
    elif conv_type == "GIN":
        layer_config = {
            "train_eps": wandb.config.model["gin_train_eps"],
            "hidden_multiplier": wandb.config.model["gin_hidden_multiplier"],
            "add_self_loops": wandb.config.model["gin_add_self_loops"],
            "is_skip_connection": wandb.config.model["gin_is_skip_connection"],
            "num_mlp_layers": wandb.config.model["gin_num_mlp_layers"],
            "is_mlp_skip_connection": wandb.config.model["gin_is_mlp_skip_connection"],
            "dropout": wandb.config.model["dropout"],
        }

    model = HeteroGnnPool(
        in_channels=input_dim,
        hidden_channels=wandb.config.model["hidden_channels"],
        out_channels=target_dim * wandb.config.transforms["num_bins"],
        num_layers=wandb.config.model["num_layers"],
        edge_types=edge_types,
        conv_type=conv_type,
        layer_config=layer_config,
        pooling=wandb.config.model["pooling"],
        activation=wandb.config.model["activation"],
        norm=wandb.config.model["norm"],
        pred_head_dropout=wandb.config.model["dropout"],
    )

    # Log model parameters
    param_counts = model.num_parameters
    wandb.log(
        {
            "model/params_conv_layers": param_counts["conv_layers"],
            "model/params_norm_layers": param_counts["norm_layers"],
            "model/params_final_layers": param_counts["final_layers"],
            "model/params_breakdown_total": param_counts["breakdown_total"],
            "model/params_total": param_counts["total"],
        }
    )

    # wandb.watch(model, log="gradients", log_freq=1, log_graph=False)

    # loss
    loss_func = CombinedCELoss(num_classes=target_dim, weights=torch.ones(2).to(device))

    task = ClassificationTask(
        model=model,
        optimizer_config=wandb.config.regression_task["optimizer"],
        lr_scheduler_config=wandb.config.regression_task["lr_scheduler"],
        batch_size=wandb.config.data_module["batch_size"],
        clip_grad_norm=wandb.config.regression_task["clip_grad_norm"],
        clip_grad_norm_max_norm=wandb.config.regression_task["clip_grad_norm_max_norm"],
        boxplot_every_n_epochs=wandb.config.regression_task["boxplot_every_n_epochs"],
        intermediate_loss_weight=wandb.config.regression_task[
            "intermediate_loss_weight"
        ],
        loss_func=loss_func,
        grad_accumulation_schedule=wandb.config.regression_task[
            "grad_accumulation_schedule"
        ],
        device=device,
        inverse_transform=inverse_transform,
    )

    # Checkpoint Callback
    model_base_path = osp.join(DATA_ROOT, "models/checkpoints")
    os.makedirs(model_base_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=osp.join(model_base_path, group),
        save_top_k=1,
        monitor="val/loss",
        mode="min",
    )

    # In your main function:
    # profiler = CustomAdvancedProfiler(
    #     dirpath="profiles", filename="advanced_profiler_output.prof"
    # )
    print(f"devices: {devices}")
    torch.set_float32_matmul_precision("medium")

    # TODO import type issues
    # if wandb_cfg["profiler"]["is_pytorch"]:
    #     Create profile directory structure
    #     profile_dir = osp.join(DATA_ROOT, "profiles", str(hostname_slurm_job_id))
    #     print(f"Profile directory: {profile_dir}")
    #     os.makedirs(profile_dir, exist_ok=True)

    #     Determine available activities based on device
    #     activities = []
    #     if torch.cuda.is_available():
    #         activities.append(torch.profiler.ProfilerActivity.CUDA)
    #     activities.append(torch.profiler.ProfilerActivity.CPU)

    #     profiler = PyTorchProfiler(
    #         dirpath=profile_dir,
    #         filename="profiler_output",
    #         schedule=torch.profiler.schedule(
    #             wait=100,  # Wait for 5 steps
    #             warmup=1,  # Add 1 warmup step
    #             active=1,  # Profile for 3 steps
    #             repeat=100,  # Repeat every 100 steps
    #             skip_first=100,
    #         ),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
    #         activities=activities,
    #         with_stack=True,
    #         export_to_chrome=True,
    #         with_steps=True,
    #         profile_memory=True,
    #         record_shapes=True,
    #         with_flops=True,
    #         with_modules=True,
    #     )
    # else:
    #     profiler = None

    profiler = None
    trainer = L.Trainer(
        strategy=wandb.config.trainer["strategy"],
        accelerator=wandb.config.trainer["accelerator"],
        devices=devices,  # FLAG
        logger=wandb_logger,
        max_epochs=wandb.config.trainer["max_epochs"],
        callbacks=[checkpoint_callback],
        profiler=profiler,  #
        log_every_n_steps=10,
        # callbacks=[checkpoint_callback, TriggerWandbSyncLightningCallback()],
    )

    # Start the training
    trainer.fit(model=task, datamodule=data_module)


if __name__ == "__main__":
    main()
