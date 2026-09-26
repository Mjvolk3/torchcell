# experiments/030-solid-growth-multi/scripts/equivariant_cell_graph_transformer.py
# [[experiments.030-solid-growth-multi.scripts.equivariant_cell_graph_transformer]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/equivariant_cell_graph_transformer
r"""Train the CellGraphTransformer on the 030 multi-measurement build, one row per entry.

The 025 script (experiments/025-solid-growth/scripts/equivariant_cell_graph_transformer.py)
with the four changes the no-merge build needs. 030 keeps every source measurement of a
genotype: a closure double carries its Costanzo 2016 fitness and interaction beside its
Kuzmin 2018 or 2020 query-strain fitness, a triple carries both Kuzmin screens when both
measured it. The 025 trainer masks any record whose label holds more than one value, which
here would drop most doubles; instead every stored entry is a training row (``per_entry``),
and the row's source dataset enters the readout as a one-hot over the build's source
datasets projected to a small learnable vector (``model.dataset_token``).

- ``subset.vocabulary`` names the token vocabulary (the build's 15 sorted source-dataset
  names); the model's ``vocab_size`` is derived from it, never typed into a config.
- ``subset.exclude`` names the essentiality holdout: its records leave the pool and are
  served as the second validation loader, on which the task logs the AUROC of predicted
  single-deletion fitness against essentiality (``val_ess/*``).
- ``transforms.fit_stats`` names the committed normalization constants fitted on the
  arm's training ENTRY rows (``make_normalization_stats_030.py``); the script recomputes
  the training set and refuses a file fitted on another one.
- ``smoke`` clones every entry of the named sources under a synthetic twin token,
  ``delta`` higher, and after training measures the twin against the own token
  (``arm_030.run_smoke_check``; ``smoke_report_030.py`` applies the criteria).

Validation and test on the pinned triples report one value per genotype under its own
screen's token (the policy reduction), so ``val/gene_interaction/Pearson`` reads against
the 025 S3 arms. Run through the launchers, which set the DDP rendezvous:

    PROJECT_ROOT=<worktree> sbatch -J 030-r1w1-s0 \\
        experiments/030-solid-growth-multi/scripts/igb_mmli_cgt_030.slurm cgt_030_s3_r_tok_embfit_001 +seed=0
    sbatch experiments/030-solid-growth-multi/scripts/gh_smoke_dataset_token.slurm
"""

# MUST be first import to catch SWIG warnings in worker processes
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import hashlib
import json
import logging
import os
import os.path as osp
import socket
import sys
import uuid

import hydra
import lightning as L
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch_geometric.transforms import Compose

from torchcell.graph.rewire import rewire_cell_graph
from torchcell.losses.logcosh import LogCoshLoss
from torchcell.losses.point_dist_graph_reg import PointDistGraphReg
from torchcell.models.equivariant_cell_graph_transformer import CellGraphTransformer
from torchcell.timestamp import timestamp
from torchcell.trainers.int_transformer_cell import RegressionTask
from torchcell.transforms.coo_regression_to_classification import (
    COOInverseCompose,
    COOLabelNormalizationTransform,
)
from torchcell.transforms.synthetic_token_offset import SyntheticTokenOffset

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from arm_030 import (  # noqa: E402
    EXPERIMENT,
    SmokeTrajectory,
    build_dataset,
    index_sha256,
    load_json_artifact,
    make_data_module,
    resolve_arm,
    results_dir,
    run_smoke_check,
    smoke_twins,
)

log = logging.getLogger(__name__)
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
WANDB_MODE = os.getenv("WANDB_MODE")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")


def get_slurm_nodes() -> int:
    if "SLURM_NNODES" in os.environ:
        return int(os.environ["SLURM_NNODES"])
    if "SLURM_JOB_NUM_NODES" in os.environ:
        return int(os.environ["SLURM_JOB_NUM_NODES"])
    return 1


def get_num_devices() -> int:
    if wandb.config.trainer["devices"] != "auto":
        return int(wandb.config.trainer["devices"])
    slurm_devices = os.environ.get("SLURM_GPUS_ON_NODE")
    if slurm_devices is not None:
        return int(slurm_devices)
    num_devices = torch.cuda.device_count()
    return num_devices if num_devices > 0 else 1


@hydra.main(
    version_base=None,
    config_path=osp.join(osp.dirname(__file__), "../conf"),
    config_name="cgt_030_s3_r_tok_embfit_001",
)
def main(cfg: DictConfig) -> None:
    print("Starting Equivariant Cell Graph Transformer Training (030, per entry) 🚀")
    os.environ["WANDB__SERVICE_WAIT"] = "600"
    if not (dist.is_available() and dist.is_initialized()):
        os.environ["TORCH_DISTRIBUTED_DEFAULT_TIMEOUT"] = "7200"

    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert isinstance(wandb_cfg, dict)
    print("wandb_cfg", wandb_cfg)

    slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    job_id = slurm_array_job_id or slurm_job_id or str(uuid.uuid4())
    hostname_job_id = f"{socket.gethostname()}-{job_id}"
    hashed_cfg = hashlib.sha256(
        json.dumps(wandb_cfg, sort_keys=True).encode("utf-8")
    ).hexdigest()
    group = f"{hostname_job_id}_{hashed_cfg}"
    experiment_dir = osp.join(DATA_ROOT, "wandb-experiments", group)
    os.makedirs(experiment_dir, exist_ok=True)

    seed = int(wandb_cfg.get("seed", 42))
    config_name = HydraConfig.get().job.config_name
    smoke_cfg = wandb_cfg.get("smoke") or {}
    smoke_on = bool(smoke_cfg.get("enabled", False))
    sweep_tags = [
        config_name,
        f"_{config_name.rsplit('_', 1)[-1]}",
        f"lambda_{wandb_cfg['model']['graph_regularization']['graph_reg_lambda']:g}",
        f"seed_{seed}",
        "build_" + wandb_cfg["dataset"]["root_rel"].split("/")[-1],
        "per_entry",
        "dataset_token",
    ] + (
        ["smoke", "smoke_control" if smoke_cfg.get("control") else "smoke_token"]
        if smoke_on
        else []
    )
    run = wandb.init(
        mode=WANDB_MODE,
        project=wandb_cfg["wandb"]["project"],
        config=wandb_cfg,
        group=group,
        tags=list(wandb_cfg["wandb"]["tags"]) + sweep_tags,
        dir=experiment_dir,
        name=f"run_{group}",
    )
    wandb_logger = WandbLogger(
        project=wandb_cfg["wandb"]["project"],
        log_model=True,
        save_dir=experiment_dir,
        name=f"run_{group}",
    )
    wandb.log({"config/num_workers": wandb.config.data_module["num_workers"]})

    rank = (
        dist.get_rank() if (torch.cuda.is_available() and dist.is_initialized()) else 0
    )

    # === Arm definition: pool, split, holdout, token vocabulary ===
    extra_vocab = (
        [f"Synthetic{src}" for src in smoke_cfg["sources"]] if smoke_on else []
    )
    arm = resolve_arm(wandb_cfg["subset"], extra_vocabulary=extra_vocab)
    train_records = arm.train_records()
    train_sha = index_sha256(train_records)
    print(
        f"arm: subset={arm.subset_name} pool={len(arm.pool)} holdout={len(arm.excluded)} "
        f"split={arm.split_file} { ({k: len(v) for k, v in arm.pinned.items()}) } "
        f"unpinned_to_train={arm.unpinned_to_train} train={len(train_records)} "
        f"vocabulary={len(arm.dataset_vocabulary)}"
    )
    wandb.log(
        {
            "arm/n_subset": len(arm.pool),
            "arm/n_holdout": len(arm.excluded),
            "arm/n_train": len(train_records),
            "arm/n_train_pinned": len(arm.pinned["train"]),
            "arm/n_val_pinned": len(arm.pinned["val"]),
            "arm/n_test_pinned": len(arm.pinned["test"]),
            "arm/vocab_size": len(arm.dataset_vocabulary),
        }
    )

    phenotype_labels = list(wandb.config.cell_dataset["phenotype_labels"])
    if phenotype_labels != ["fitness", "gene_interaction"]:
        raise ValueError(
            "the per-entry path trains fitness and gene_interaction jointly; "
            f"got phenotype_labels={phenotype_labels}"
        )
    with open(
        osp.join(EXPERIMENT_ROOT, EXPERIMENT, "queries/001_multi_measurement.cql")
    ) as f:
        query = f.read()
    dataset_root = osp.join(DATA_ROOT, wandb_cfg["dataset"]["root_rel"])
    print(f"dataset_root: {dataset_root}")
    dataset, genome, gene_multigraph, node_embeddings = build_dataset(
        data_root=DATA_ROOT,
        dataset_root=dataset_root,
        query=query,
        graph_names=list(wandb.config.cell_dataset["graphs"]),
        node_embedding_names=list(wandb.config.cell_dataset.get("node_embeddings", [])),
        phenotype_labels=phenotype_labels,
        dataset_vocabulary=arm.dataset_vocabulary,
        rank=rank,
    )
    print(
        f"Dataset Length: {len(dataset)}; embeddings: {list(node_embeddings) or 'learnable'}"
    )

    # === Normalization from the committed training-entry constants ===
    transform_config = wandb.config.transforms["forward_transform"]
    norm_config = transform_config["normalization"]
    stats_name = str(wandb.config.transforms["fit_stats"])
    stats_file = load_json_artifact(stats_name)
    if stats_file["train_index_sha256"] != train_sha:
        raise ValueError(
            f"{stats_name} was fitted on training set {stats_file['train_index_sha256'][:12]} "
            f"({stats_file['n_train_records']} records); this arm trains "
            f"{train_sha[:12]} ({len(train_records)} records). Rerun "
            "make_normalization_stats_030.py for this arm."
        )
    norm_transform = COOLabelNormalizationTransform(
        dataset, norm_config, fit_stats=stats_file["stats"]
    )
    transforms_list = [norm_transform]
    for label, stats in norm_transform.stats.items():
        print(f"normalization {label}: {stats}")
        wandb.log(
            {
                f"arm/norm_{label}_{key}": value
                for key, value in stats.items()
                if key in ("mean", "std")
            }
        )
    wandb.log({"arm/norm_fit_records": stats_file["n_train_records"]})
    # A COPY of the list: COOInverseCompose keeps the list it is given, and the smoke
    # transform appended below has no inverse (job 2861 failed in the sanity check).
    inverse_transform = COOInverseCompose([norm_transform])
    if smoke_on:
        # After the normalizer, so delta is a normalized offset; not part of the
        # inverse, which maps predictions back to label units.
        transforms_list.append(
            SyntheticTokenOffset(
                delta=float(smoke_cfg["delta"]),
                token_map={
                    arm.dataset_vocabulary.index(src): arm.dataset_vocabulary.index(
                        twin
                    )
                    for src, twin in smoke_twins(arm, smoke_cfg).items()
                },
                labels=list(smoke_cfg["labels"]),
                normalizer=norm_transform,
                control=bool(smoke_cfg.get("control", False)),
            )
        )
        print(f"smoke: {transforms_list[-1]}")
    dataset.transform = Compose(transforms_list)

    L.seed_everything(seed, workers=True)
    print(f"seed: {seed}")

    # Every entry row needs its genotype (phenotype_values_batch) and its token.
    follow_batch = ["perturbation_indices", "phenotype_values"]
    fitness_lambda = float(wandb_cfg["regression_task"]["fitness_lambda"])
    heads_config = wandb_cfg["model"]["heads"]
    data_module = make_data_module(
        dataset, arm, seed, dict(wandb.config.data_module), follow_batch
    )
    print(
        f"splits: train={len(data_module.index.train)} val={len(data_module.index.val)} "
        f"test={len(data_module.index.test)} val_ess={len(data_module.extra_val_datasets['val_ess'])}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    devices = get_num_devices()

    cell_graph = dataset.cell_graph
    random_graph_cfg = wandb_cfg["model"].get("random_graph") or {}
    if random_graph_cfg.get("enabled", False):
        rewire_seed = random_graph_cfg.get("seed")
        rewire_seed = seed if rewire_seed is None else int(rewire_seed)
        cell_graph, rewire_stats = rewire_cell_graph(
            dataset.cell_graph,
            seed=rewire_seed,
            swaps_per_edge=float(random_graph_cfg.get("swaps_per_edge", 5.0)),
        )
        wandb.log(
            {"random_graph/seed": rewire_seed}
            | {
                f"random_graph/{rel}/{k}": v
                for rel, st in rewire_stats.items()
                for k, v in st.items()
            }
        )

    loss_config = wandb.config.regression_task["loss"]
    graph_reg_lambda = float(
        loss_config.get("graph_regularization", {}).get("lambda", 0.0) or 0.0
    )
    token_cfg = dict(wandb_cfg["model"]["dataset_token"])
    token_cfg["vocab_size"] = len(arm.dataset_vocabulary)
    print(
        f"Instantiating CellGraphTransformer ({timestamp()}); dataset_token={token_cfg}"
    )
    model = CellGraphTransformer(
        gene_num=wandb.config["model"]["gene_num"],
        hidden_channels=wandb.config["model"]["hidden_channels"],
        num_transformer_layers=wandb.config["model"]["num_transformer_layers"],
        num_attention_heads=wandb.config["model"]["num_attention_heads"],
        cell_graph=cell_graph,
        graph_regularization_config=wandb.config["model"]["graph_regularization"],
        perturbation_head_config=wandb.config["model"]["perturbation_head"],
        dropout=wandb.config["model"]["dropout"],
        graph_reg_lambda=graph_reg_lambda,
        node_embeddings=node_embeddings,
        learnable_embedding_config=wandb.config["model"].get("learnable_embedding"),
        attention_mask_config=wandb.config["model"].get("attention_mask"),
        heads_config=heads_config,
        perturb_cls=bool(wandb_cfg["model"].get("perturb_cls", False)),
        perturbation_head_cls=str(
            wandb_cfg["model"].get("perturbation_head_cls", "wildtype")
        ),
        dataset_token=token_cfg,
    ).to(device)
    param_counts = model.num_parameters
    print("Parameter counts:", param_counts)
    wandb.log({f"model/params_{k}": v for k, v in param_counts.items()})

    loss_type = loss_config.get("type", "logcosh")
    if loss_type == "point_dist_graph_reg":
        loss_func: nn.Module = PointDistGraphReg(
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

    ess_cfg = wandb_cfg["regression_task"]["essentiality"]
    node_labels = arm.essentiality_node_labels(
        [str(g) for g in cell_graph["gene"].node_ids]
    )
    essentiality_eval = {
        "token_smf": str(ess_cfg["token_smf"]),
        "token_sgd": str(ess_cfg["token_sgd"]),
        "released": node_labels["released"],
        "matched": node_labels["matched"],
    }
    task = RegressionTask(
        model=model,
        cell_graph=cell_graph,
        optimizer_config=wandb_cfg["regression_task"]["optimizer"],
        lr_scheduler_config=wandb_cfg["regression_task"]["lr_scheduler"],
        batch_size=wandb_cfg["data_module"]["batch_size"],
        clip_grad_norm=wandb_cfg["regression_task"]["clip_grad_norm"],
        clip_grad_norm_max_norm=wandb_cfg["regression_task"]["clip_grad_norm_max_norm"],
        plot_sample_ceiling=wandb.config["regression_task"]["plot_sample_ceiling"],
        loss_func=loss_func,
        grad_accumulation_schedule=wandb.config["regression_task"][
            "grad_accumulation_schedule"
        ],
        device=device,
        inverse_transform=inverse_transform,
        plot_every_n_epochs=wandb.config["regression_task"]["plot_every_n_epochs"],
        plot_edge_recovery_every_n_epochs=wandb.config["regression_task"][
            "plot_edge_recovery_every_n_epochs"
        ],
        plot_transformer_diagnostics_every_n_epochs=wandb.config["regression_task"][
            "plot_transformer_diagnostics_every_n_epochs"
        ],
        execution_mode=wandb.config["regression_task"].get(
            "execution_mode", "training"
        ),
        fitness_lambda=fitness_lambda,
        gradient_probe_epochs=wandb_cfg["regression_task"].get("gradient_probe_epochs"),
        per_order_metrics=bool(
            wandb_cfg["regression_task"].get("per_order_metrics", False)
        ),
        per_entry=True,
        dataset_vocabulary=arm.dataset_vocabulary,
        essentiality_eval=essentiality_eval,
    )

    checkpoint_dir = osp.join(DATA_ROOT, "models/checkpoints", group)
    os.makedirs(osp.dirname(checkpoint_dir), exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_top_k=1,
            monitor="val/gene_interaction/MSE",
            mode="min",
            filename=f"{run.id}-best-mse-{{epoch:02d}}-{{val/gene_interaction/MSE:.4f}}",
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_top_k=1,
            monitor="val/gene_interaction/Pearson",
            mode="max",
            filename=f"{run.id}-best-pearson-{{epoch:02d}}-{{val/gene_interaction/Pearson:.4f}}",
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir, save_last=True, filename=f"{run.id}-last"
        ),
    ]
    if smoke_on:
        callbacks.append(
            SmokeTrajectory(
                arm, smoke_cfg, n_batches=int(smoke_cfg["n_trajectory_batches"])
            )
        )

    torch.set_float32_matmul_precision("medium")
    print(f"devices: {devices}; starting training ({timestamp()})")
    trainer = L.Trainer(
        strategy=wandb.config.trainer["strategy"],
        accelerator=wandb.config.trainer["accelerator"],
        devices=devices,
        num_nodes=get_slurm_nodes(),
        logger=wandb_logger,
        max_epochs=wandb.config.trainer["max_epochs"],
        callbacks=callbacks,
        log_every_n_steps=10,
        overfit_batches=wandb.config.trainer["overfit_batches"],
        precision=wandb.config.trainer.get("precision", "32-true"),
        limit_train_batches=wandb.config.trainer.get("limit_train_batches", 1.0),
        limit_val_batches=wandb.config.trainer.get("limit_val_batches", 1.0),
    )
    trainer.fit(model=task, datamodule=data_module)

    mse = trainer.callback_metrics["val/gene_interaction/MSE"].item()
    pearson = trainer.callback_metrics["val/gene_interaction/Pearson"].item()
    if smoke_on and trainer.is_global_zero:
        callback_metrics = {
            k: float(v)
            for k, v in trainer.callback_metrics.items()
            if k.startswith(("val/", "val_ess/")) and hasattr(v, "item")
        }
        result = run_smoke_check(
            task=task,
            loader=data_module.val_dataloader()[0],
            arm=arm,
            config_name=config_name,
            seed=seed,
            smoke_cfg=smoke_cfg,
            n_batches=int(smoke_cfg["n_val_batches"]),
            callback_metrics=callback_metrics,
            wandb_run_id=str(run.id),
        )
        out_dir = osp.join(results_dir(), "smoke")
        os.makedirs(out_dir, exist_ok=True)
        out_path = osp.join(out_dir, f"{config_name}_seed{seed}.json")
        with open(out_path, "w") as f:
            f.write(result.model_dump_json(indent=2))
        print(f"smoke result: {out_path}\n{result.model_dump_json(indent=2)}")
        wandb.log(
            {
                f"smoke/{k}": v
                for k, v in result.model_dump().items()
                if isinstance(v, (int, float))
            }
        )
    wandb.finish()
    return (mse, pearson)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
