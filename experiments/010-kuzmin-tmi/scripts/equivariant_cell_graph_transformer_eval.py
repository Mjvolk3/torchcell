# experiments/010-kuzmin-tmi/scripts/equivariant_cell_graph_transformer_eval.py
# [[experiments.010-kuzmin-tmi.scripts.equivariant_cell_graph_transformer_eval]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/equivariant_cell_graph_transformer_eval.py
#
# Full evaluation script for CellGraphTransformer on val and test sets
# - Collects ALL predictions (no subsampling)
# - Computes comprehensive metrics
# - Generates visualizations
# - Saves predictions to Parquet
# - Logs to same wandb group as training

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import logging
import os
import os.path as osp

import hydra
import lightning as L
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from scipy import stats
from torch_geometric.transforms import Compose

from torchcell.data import GenotypeAggregator, MeanExperimentDeduplicator, Neo4jCellDataset
from torchcell.data.graph_processor import Perturbation
from torchcell.datamodules import CellDataModule
from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder
from torchcell.graph import SCerevisiaeGraph
from torchcell.graph.graph import build_gene_multigraph
from torchcell.losses.logcosh import LogCoshLoss
from torchcell.losses.point_dist_graph_reg import PointDistGraphReg
from torchcell.models.equivariant_cell_graph_transformer import CellGraphTransformer
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.timestamp import timestamp
from torchcell.trainers.int_transformer_cell import RegressionTask
from torchcell.transforms.coo_regression_to_classification import (
    COOInverseCompose,
    COOLabelNormalizationTransform,
)

log = logging.getLogger(__name__)
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
WANDB_MODE = os.getenv("WANDB_MODE")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")


def compute_additional_metrics(predictions: np.ndarray, true_values: np.ndarray, threshold: float = 0.08) -> dict:
    """
    Compute additional metrics not in the trainer.

    Args:
        predictions: numpy array of predictions
        true_values: numpy array of true values
        threshold: threshold for strong genetic interaction (default 0.08 per Kuzmin paper)

    Returns:
        dict of metric name -> value
    """
    # Remove NaN values
    mask = ~np.isnan(predictions) & ~np.isnan(true_values)
    pred_np = predictions[mask]
    true_np = true_values[mask]

    metrics = {}

    # Additional point metrics
    metrics["MAE"] = float(np.mean(np.abs(pred_np - true_np)))
    metrics["Spearman"], _ = stats.spearmanr(pred_np, true_np)

    # R-squared
    ss_res = np.sum((true_np - pred_np) ** 2)
    ss_tot = np.sum((true_np - np.mean(true_np)) ** 2)
    metrics["R2"] = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    # Distribution metrics
    metrics["Wasserstein"] = float(stats.wasserstein_distance(true_np, pred_np))

    # KL-divergence (using histogram approximation)
    bins = min(100, int(np.sqrt(len(true_np))))
    true_hist, edges = np.histogram(true_np, bins=bins, density=True)
    pred_hist, _ = np.histogram(pred_np, bins=edges, density=True)
    epsilon = 1e-10
    true_hist = (true_hist + epsilon) / (true_hist.sum() + epsilon * len(true_hist))
    pred_hist = (pred_hist + epsilon) / (pred_hist.sum() + epsilon * len(pred_hist))
    m = 0.5 * (true_hist + pred_hist)
    metrics["JS_divergence"] = float(0.5 * (stats.entropy(true_hist, m) + stats.entropy(pred_hist, m)))
    metrics["KL_divergence"] = float(stats.entropy(pred_hist, true_hist))

    # Classification metrics at threshold
    # Strong genetic interaction: |score| > threshold
    true_strong = np.abs(true_np) > threshold
    pred_strong = np.abs(pred_np) > threshold

    tp = np.sum(true_strong & pred_strong)
    fp = np.sum(~true_strong & pred_strong)
    fn = np.sum(true_strong & ~pred_strong)
    tn = np.sum(~true_strong & ~pred_strong)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(pred_np) if len(pred_np) > 0 else 0.0

    metrics[f"Precision_at_{threshold}"] = float(precision)
    metrics[f"Recall_at_{threshold}"] = float(recall)
    metrics[f"F1_at_{threshold}"] = float(f1)
    metrics[f"Accuracy_at_{threshold}"] = float(accuracy)

    # Prevalence statistics
    metrics["true_strong_fraction"] = float(np.mean(true_strong))
    metrics["pred_strong_fraction"] = float(np.mean(pred_strong))
    metrics["n_samples"] = int(len(pred_np))

    return metrics


def save_predictions_parquet(
    predictions: torch.Tensor,
    true_values: torch.Tensor,
    split_name: str,
    results_dir: str,
) -> str:
    """
    Save predictions to Parquet file.

    Args:
        predictions: tensor of predictions
        true_values: tensor of true values
        split_name: "val" or "test"
        results_dir: directory to save to

    Returns:
        path to saved file
    """
    n = len(predictions)
    ts = timestamp()

    # Build table
    table_data = {
        "index": list(range(n)),
        "split": [split_name] * n,
        "prediction": predictions.numpy().flatten().astype(np.float32),
        "true_value": true_values.numpy().flatten().astype(np.float32),
    }

    schema = pa.schema([
        ("index", pa.int64()),
        ("split", pa.string()),
        ("prediction", pa.float32()),
        ("true_value", pa.float32()),
    ])

    table = pa.table(table_data, schema=schema)

    output_path = osp.join(results_dir, f"predictions_{split_name}_{ts}.parquet")
    pq.write_table(
        table,
        output_path,
        compression="snappy",
        use_dictionary=["split"],
    )

    print(f"Saved {n:,} predictions to {output_path}")
    return output_path


@hydra.main(
    version_base=None,
    config_path=osp.join(osp.dirname(__file__), "../conf"),
    config_name="equivariant_cell_graph_transformer_eval",
)
def main(cfg: DictConfig) -> None:
    print("Starting Equivariant Cell Graph Transformer Evaluation")
    ts = timestamp()

    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    print("wandb_cfg:", wandb_cfg)

    # Get training group from config
    training_group = wandb_cfg["evaluation"]["training_group"]
    threshold = wandb_cfg["evaluation"]["threshold"]

    # Create output directories
    results_dir = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi/results/eval")
    os.makedirs(results_dir, exist_ok=True)

    images_dir = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi-eval")
    os.makedirs(images_dir, exist_ok=True)

    experiment_dir = osp.join(DATA_ROOT, "wandb-experiments", f"eval_{training_group}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Initialize wandb with SAME GROUP as training
    run = wandb.init(
        mode=WANDB_MODE,
        project=wandb_cfg["wandb"]["project"],
        config=wandb_cfg,
        group=training_group,
        tags=wandb_cfg["wandb"]["tags"],
        dir=experiment_dir,
        name=f"eval_{training_group[:20]}_{ts}",
    )

    wandb_logger = WandbLogger(
        project=wandb_cfg["wandb"]["project"],
        log_model=False,
        save_dir=experiment_dir,
        name=f"eval_{training_group[:20]}_{ts}",
    )

    # Setup infrastructure (same as training script)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    genome_root = osp.join(DATA_ROOT, "data/sgd/genome")
    go_root = osp.join(DATA_ROOT, "data/go")

    genome = SCerevisiaeGenome(genome_root=genome_root, go_root=go_root, overwrite=False)
    genome.drop_empty_go()

    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    # Build gene multigraph
    graph_names = wandb.config.cell_dataset["graphs"]
    gene_multigraph = build_gene_multigraph(graph=graph, graph_names=graph_names)

    # Build node embeddings
    node_embedding_names = wandb.config.cell_dataset.get("node_embeddings", [])
    node_embeddings = NodeEmbeddingBuilder.build(
        embedding_names=node_embedding_names,
        data_root=DATA_ROOT,
        genome=genome,
        graph=graph,
    )
    print(f"Built node embeddings: {list(node_embeddings.keys()) if node_embeddings else 'None (using learnable)'}")

    # Load query and create dataset
    with open(osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi/queries/001_small_build.cql"), "r") as f:
        query = f.read()

    dataset_root = osp.join(DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build")

    # Initialize transforms
    forward_transform = None
    inverse_transform = None

    if wandb.config.transforms.get("use_transforms", False):
        print("\nInitializing transforms from configuration...")
        transform_config = wandb.config.transforms.get("forward_transform", {})

        transforms_list = []
        norm_transform = None

        # Create dataset without transforms first to get statistics
        dataset = Neo4jCellDataset(
            root=dataset_root,
            query=query,
            gene_set=genome.gene_set,
            graphs=gene_multigraph,
            node_embeddings=node_embeddings,
            converter=None,
            deduplicator=MeanExperimentDeduplicator,
            aggregator=GenotypeAggregator,
            graph_processor=Perturbation(),
            transform=None,
        )

        # Normalization transform
        if "normalization" in transform_config:
            norm_config = transform_config["normalization"]
            norm_transform = COOLabelNormalizationTransform(dataset, norm_config)
            transforms_list.append(norm_transform)
            print(f"Added normalization transform for: {list(norm_config.keys())}")

        if transforms_list:
            forward_transform = Compose(transforms_list)
            inverse_transform = COOInverseCompose(transforms_list)
            print("Transforms initialized successfully")
            dataset.transform = forward_transform
    else:
        dataset = Neo4jCellDataset(
            root=dataset_root,
            query=query,
            gene_set=genome.gene_set,
            graphs=gene_multigraph,
            node_embeddings=node_embeddings,
            converter=None,
            deduplicator=MeanExperimentDeduplicator,
            aggregator=GenotypeAggregator,
            graph_processor=Perturbation(),
            transform=None,
        )

    print(f"Dataset Length: {len(dataset)}")

    # Create data module
    seed = 42
    follow_batch = ["perturbation_indices"]

    data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=wandb.config.data_module["batch_size"],
        random_seed=seed,
        num_workers=wandb.config.data_module["num_workers"],
        pin_memory=wandb.config.data_module["pin_memory"],
        prefetch=wandb.config.data_module["prefetch"],
        prefetch_factor=wandb.config.data_module["prefetch_factor"],
        persistent_workers=wandb.config.data_module["persistent_workers"],
        follow_batch=follow_batch,
        val_batch_size=wandb.config.data_module.get("val_batch_size"),
    )
    data_module.setup()

    # Wrap with PerturbationSubsetDataModule if subset mode is enabled
    # This ensures eval uses the SAME data subset as training
    if wandb.config.data_module.get("is_perturbation_subset", False):
        print(f"\nUsing perturbation subset mode (size={wandb.config.data_module['perturbation_subset_size']})")
        data_module = PerturbationSubsetDataModule(
            cell_data_module=data_module,
            size=int(wandb.config.data_module["perturbation_subset_size"]),
            batch_size=wandb.config.data_module["batch_size"],
            num_workers=wandb.config.data_module["num_workers"],
            pin_memory=wandb.config.data_module["pin_memory"],
            prefetch=wandb.config.data_module["prefetch"],
            seed=seed,
            prefetch_factor=wandb.config.data_module["prefetch_factor"],
            persistent_workers=wandb.config.data_module["persistent_workers"],
            follow_batch=follow_batch,
            val_batch_size=wandb.config.data_module.get("val_batch_size"),
        )
        data_module.setup()

    # Log dataset split sizes
    print(f"\nDataset splits:")
    print(f"  Train: {len(data_module.train_dataloader().dataset):,}")
    print(f"  Val: {len(data_module.val_dataloader().dataset):,}")
    print(f"  Test: {len(data_module.test_dataloader().dataset):,}")

    wandb.log({
        "dataset/train_size": len(data_module.train_dataloader().dataset),
        "dataset/val_size": len(data_module.val_dataloader().dataset),
        "dataset/test_size": len(data_module.test_dataloader().dataset),
    })

    # Get graph regularization lambda
    loss_config = wandb.config.regression_task.get("loss", {})
    graph_reg_lambda = 0.0
    if isinstance(loss_config, dict):
        graph_reg_config = loss_config.get("graph_regularization", {})
        if isinstance(graph_reg_config, dict):
            lambda_val = graph_reg_config.get("lambda", 0.0)
            graph_reg_lambda = 0.0 if lambda_val is None else float(lambda_val)

    # Instantiate model
    print(f"Instantiating CellGraphTransformer ({timestamp()})")
    model = CellGraphTransformer(
        gene_num=wandb.config["model"]["gene_num"],
        hidden_channels=wandb.config["model"]["hidden_channels"],
        num_transformer_layers=wandb.config["model"]["num_transformer_layers"],
        num_attention_heads=wandb.config["model"]["num_attention_heads"],
        cell_graph=dataset.cell_graph,
        graph_regularization_config=wandb.config["model"]["graph_regularization"],
        perturbation_head_config=wandb.config["model"]["perturbation_head"],
        dropout=wandb.config["model"]["dropout"],
        graph_reg_lambda=graph_reg_lambda,
        node_embeddings=node_embeddings,
        learnable_embedding_config=wandb.config["model"].get("learnable_embedding"),
    ).to(device)

    # Create loss function
    if isinstance(loss_config, dict):
        loss_type = loss_config.get("type", "logcosh")
        if loss_type == "point_dist_graph_reg":
            loss_func = PointDistGraphReg(
                point_estimator=loss_config.get("point_estimator"),
                distribution_loss=loss_config.get("distribution_loss"),
                graph_regularization=loss_config.get("graph_regularization"),
                buffer=loss_config.get("buffer"),
                ddp=loss_config.get("ddp"),
            )
        elif loss_type == "logcosh":
            loss_func = LogCoshLoss(reduction="mean")
        elif loss_type == "mse":
            loss_func = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    else:
        loss_func = LogCoshLoss(reduction="mean")

    # Load checkpoint - prepend DATA_ROOT if path is relative (training saves to DATA_ROOT/models/checkpoints/)
    checkpoint_path = wandb.config["model"]["checkpoint_path"]

    if not os.path.isabs(checkpoint_path):
        checkpoint_path = osp.join(DATA_ROOT, checkpoint_path)

    print(f"Loading checkpoint from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    task = RegressionTask.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        model=model,
        cell_graph=dataset.cell_graph,
        loss_func=loss_func,
        device=device,
        optimizer_config=wandb_cfg["regression_task"]["optimizer"],
        lr_scheduler_config=wandb_cfg["regression_task"]["lr_scheduler"],
        batch_size=wandb_cfg["data_module"]["batch_size"],
        clip_grad_norm=wandb_cfg["regression_task"]["clip_grad_norm"],
        clip_grad_norm_max_norm=wandb_cfg["regression_task"]["clip_grad_norm_max_norm"],
        inverse_transform=inverse_transform,
        plot_every_n_epochs=wandb.config["regression_task"]["plot_every_n_epochs"],
        plot_sample_ceiling=wandb.config["regression_task"]["plot_sample_ceiling"],
        plot_edge_recovery_every_n_epochs=wandb.config["regression_task"]["plot_edge_recovery_every_n_epochs"],
        plot_transformer_diagnostics_every_n_epochs=wandb.config["regression_task"]["plot_transformer_diagnostics_every_n_epochs"],
        grad_accumulation_schedule=wandb.config["regression_task"].get("grad_accumulation_schedule"),
        execution_mode="training",  # Use training mode to get full metrics
    )
    print("Successfully loaded model weights from checkpoint")

    # Create trainer for evaluation
    torch.set_float32_matmul_precision("medium")

    trainer = L.Trainer(
        strategy="auto",
        accelerator=wandb.config.trainer["accelerator"],
        devices=1,
        num_nodes=1,
        logger=wandb_logger,
        max_epochs=1,
        callbacks=[],
        log_every_n_steps=10,
        overfit_batches=0,
        precision=wandb.config.trainer.get("precision", "32-true"),
    )

    # Run validation
    print(f"\n{'='*60}")
    print(f"Running VALIDATION evaluation ({timestamp()})")
    print(f"{'='*60}")

    val_results = trainer.validate(model=task, datamodule=data_module)
    print(f"Validation results: {val_results}")

    # Save val predictions to parquet
    if task.val_samples["predictions"]:
        val_predictions = torch.cat(task.val_samples["predictions"], dim=0).cpu()
        val_true_values = torch.cat(task.val_samples["true_values"], dim=0).cpu()

        print(f"Collected {len(val_predictions):,} validation predictions")

        # Save to parquet
        val_parquet_path = save_predictions_parquet(
            val_predictions, val_true_values, "val", results_dir
        )
        wandb.log({"val_predictions_path": val_parquet_path})

        # Compute additional metrics
        val_additional_metrics = compute_additional_metrics(
            val_predictions.numpy().flatten(),
            val_true_values.numpy().flatten(),
            threshold=threshold,
        )

        # Log additional metrics
        for key, value in val_additional_metrics.items():
            wandb.log({f"eval_val/{key}": value})
            print(f"  eval_val/{key}: {value}")

    # Run test
    print(f"\n{'='*60}")
    print(f"Running TEST evaluation ({timestamp()})")
    print(f"{'='*60}")

    test_results = trainer.test(model=task, datamodule=data_module)
    print(f"Test results: {test_results}")

    # Save test predictions to parquet
    if task.test_samples["predictions"]:
        test_predictions = torch.cat(task.test_samples["predictions"], dim=0).cpu()
        test_true_values = torch.cat(task.test_samples["true_values"], dim=0).cpu()

        print(f"Collected {len(test_predictions):,} test predictions")

        # Save to parquet
        test_parquet_path = save_predictions_parquet(
            test_predictions, test_true_values, "test", results_dir
        )
        wandb.log({"test_predictions_path": test_parquet_path})

        # Compute additional metrics
        test_additional_metrics = compute_additional_metrics(
            test_predictions.numpy().flatten(),
            test_true_values.numpy().flatten(),
            threshold=threshold,
        )

        # Log additional metrics
        for key, value in test_additional_metrics.items():
            wandb.log({f"eval_test/{key}": value})
            print(f"  eval_test/{key}: {value}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Evaluation Complete ({timestamp()})")
    print(f"{'='*60}")
    print(f"Results saved to: {results_dir}")
    print(f"Visualizations logged to wandb group: {training_group}")

    wandb.finish()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
