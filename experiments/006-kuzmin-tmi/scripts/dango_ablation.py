# experiments/006-kuzmin-tmi/scripts/dango_ablation.py
# [[experiments.006-kuzmin-tmi.scripts.dango_ablation]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/006-kuzmin-tmi/scripts/dango_ablation

"""DANGO ablation on the experiment-006 trigenic build, STRING v9.1 graphs.

Trains one arm of a factorial ablation of the served DANGO configuration
(``conf/dango_kuzmin2018_tmi_string9_1*.yaml``) to test two claims:

* the reconstruction objective ("pretraining": the per-network adjacency
  reconstruction loss weighted by ``alpha``) is needed, versus training on the
  interaction loss alone (``PreThenPost(transition_epoch=0)``, alpha = 0 from the
  first step);
* the ReZero residual ``x + beta * O(attn(x))`` is needed, versus the plain
  residual ``x + O(attn(x))`` (``Dango(residual="plain")``, no gate parameter).

Arms: ``conf/dango_ablation_{baseline,no_pretrain,no_rezero,no_pretrain_no_rezero}.yaml``.
They share everything else (seed 42 split, seed 42 init and shuffle, AdamW lr 1e-5
wd 1e-6, batch 64, hidden 64, 4 heads, clip 10, 150 epochs, one GPU).

Why the LMDB is bypassed. ``scripts/dango.py`` reads the build through
``Neo4jCellDataset``, which the current pydantic schema refuses: ``ds[0]`` raises 538
validation errors (``perturbation_type: "deletion"`` where the schema now wants
``sga_kanmx_deletion``, extra keys such as ``deletion_description``, missing
``released_kind`` and ``environment.media.is_synthetic``; first breaking commit
933ee5ef3a), and its ``processed/*.json.lock`` files belong to uid 7474. DANGO needs
nothing from a record but the three perturbed genes and the interaction score, so this
script reads them from the build's side files, read-only and without file locks:
``processed/label_df.parquet`` (record id, gene_interaction),
``processed/is_any_perturbed_gene_index.json`` (gene -> record ids, exactly three genes
per record) and ``data_module_cache/index_seed_42.json`` (the train/val/test record ids
the served runs used). The model's gene index is the position in
``sorted(genome.gene_set)`` after ``drop_empty_go()`` (the sorted base graph that
``to_cell_data`` orders nodes by), the convention that
``experiments/010-kuzmin-tmi/scripts/dango_full_dataset_inference.py`` verified against
the served checkpoint 014mprap (recomputed val Pearson within 3e-5 of the logged value,
``results/dango_full_dataset_inference_check.csv``). ``load_records`` and
``build_cell_graph`` are that script's helpers.
"""

import gc
import hashlib
import json
import logging
import os
import os.path as osp
import socket
import sys
import time
import uuid
from typing import Any

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import wandb
from dotenv import load_dotenv
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from sortedcontainers import SortedDict
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import HeteroData

from torchcell.data.cell_data import to_cell_data
from torchcell.data.neo4j_cell import create_graph_from_gene_set
from torchcell.graph import GeneMultiGraph, SCerevisiaeGraph, build_gene_multigraph
from torchcell.losses.dango import SCHEDULER_MAP, DangoLoss
from torchcell.models.dango import Dango
from torchcell.sequence import GeneSet
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.trainers.int_dango import RegressionTask

# The lambda table the served runs used, imported from the sibling module as dango.py does
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from scripts.dango_lambda_determination import determine_lambda_values  # noqa: E402

log = logging.getLogger(__name__)
load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
WANDB_MODE = os.environ["WANDB_MODE"]

BUILD_DIR = osp.join(
    DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/001-small-build"
)
# The seed-42 split cache the served 006 runs trained on (CellDataModule, seed 42).
EXPECTED_SPLIT_SIZES = {"train": 265851, "val": 33231, "test": 33231}
EXPECTED_GENE_NUM = 6607


def load_records() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[str]]:
    """Perturbed gene names per record (as vocabulary positions), labels, splits, vocabulary."""
    label_df = pd.read_parquet(osp.join(BUILD_DIR, "processed", "label_df.parquet"))
    with open(
        osp.join(BUILD_DIR, "processed", "is_any_perturbed_gene_index.json")
    ) as f:
        gene_index = json.load(f)
    with open(osp.join(BUILD_DIR, "data_module_cache", "index_seed_42.json")) as f:
        split_index = json.load(f)

    record_ids = np.sort(label_df["index"].to_numpy())
    y = label_df.set_index("index").loc[record_ids, "gene_interaction"].to_numpy()
    id_to_row = {int(r): i for i, r in enumerate(record_ids)}

    gene_names = sorted(gene_index.keys())
    row_genes = np.full((len(record_ids), 3), -1, dtype=np.int64)
    fill = np.zeros(len(record_ids), dtype=np.int8)
    for j, gene in enumerate(gene_names):
        for rid in gene_index[gene]:
            row = id_to_row[int(rid)]
            row_genes[row, fill[row]] = j
            fill[row] += 1
    assert (fill == 3).all()
    splits = {
        name: np.array([id_to_row[int(r)] for r in ids], dtype=np.int64)
        for name, ids in split_index.items()
    }
    return row_genes, y, splits, gene_names


def build_cell_graph(
    genome: SCerevisiaeGenome, graph: SCerevisiaeGraph, names: list[str]
) -> HeteroData:
    """The wildtype cell graph with the run's six STRING channels, genome base graph."""
    multigraph = build_gene_multigraph(graph=graph, graph_names=names)
    graphs_dict = SortedDict(multigraph.graphs.copy())
    graphs_dict["base"] = create_graph_from_gene_set(GeneSet(genome.gene_set))
    cell_graph = to_cell_data(GeneMultiGraph(graphs=graphs_dict), incidence_graphs=None)
    for name in names:
        assert ("gene", name, "gene") in cell_graph.edge_types, (
            name,
            cell_graph.edge_types,
        )
    return cell_graph


class GenotypeSplit(Dataset):
    """One split's records: three perturbed-gene indices and the interaction score."""

    def __init__(self, triples: np.ndarray, labels: np.ndarray):
        """Hold the [n, 3] gene-index triples and the [n] raw interaction scores."""
        assert triples.ndim == 2 and triples.shape[1] == 3, triples.shape
        assert labels.shape == (triples.shape[0],), labels.shape
        self.triples = torch.from_numpy(triples.astype(np.int64))
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self) -> int:
        """Number of records."""
        return int(self.triples.shape[0])

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gene-index triple and score of record ``i``."""
        return self.triples[i], self.labels[i]


def collate_genotypes(items: list[tuple[torch.Tensor, torch.Tensor]]) -> HeteroData:
    """The batch fields ``int_dango.RegressionTask`` and ``Dango`` read, and no others.

    ``perturbation_indices`` (flattened gene indices), ``perturbation_indices_batch``
    (genotype id per gene), ``phenotype_values`` ([n] float32, raw interaction score;
    the served runs used no label transform) and ``num_graphs = n``. The served
    ``Perturbation`` processor sets no ``phenotype_values_original``, so neither does
    this; the task then reads ``phenotype_values`` for the metrics too.
    """
    triples = torch.stack([t for t, _ in items])
    n = int(triples.shape[0])
    batch = HeteroData()
    batch.num_graphs = n
    batch["gene"].perturbation_indices = triples.reshape(-1)
    batch["gene"].perturbation_indices_batch = torch.arange(n).repeat_interleave(3)
    batch["gene"].phenotype_values = torch.stack([y for _, y in items])
    return batch


class EpochTelemetry(Callback):
    """Per-epoch wall time, validation time, peak GPU memory and the ReZero gates."""

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: Any) -> None:
        """Start the epoch clock and reset the CUDA peak-memory counter."""
        self._t_epoch = time.perf_counter()
        self._val_s = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: Any) -> None:
        """Start the validation clock."""
        self._t_val = time.perf_counter()

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: Any) -> None:
        """Stop the validation clock (sanity-check passes are not counted)."""
        if not trainer.sanity_checking:
            self._val_s = time.perf_counter() - self._t_val

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: Any) -> None:
        """Log and print the epoch's timing, GPU peak and gate values."""
        metrics = {
            "perf/epoch_wall_s": time.perf_counter() - self._t_epoch,
            "perf/val_wall_s": self._val_s,
        }
        if torch.cuda.is_available():
            metrics["perf/gpu_max_allocated_gib"] = (
                torch.cuda.max_memory_allocated() / 2**30
            )
            metrics["perf/gpu_max_reserved_gib"] = (
                torch.cuda.max_memory_reserved() / 2**30
            )
        hs = pl_module.model.hyper_sagnn
        if hs.residual == "rezero":
            metrics["model/beta1"] = float(hs.beta1.detach())
            metrics["model/beta2"] = float(hs.beta2.detach())
        pl_module.log_dict(metrics, on_step=False, on_epoch=True)
        print(
            f"[epoch {trainer.current_epoch}] "
            + " ".join(f"{k} {v:.4f}" for k, v in metrics.items()),
            flush=True,
        )


@hydra.main(
    version_base=None,
    config_path=osp.join(osp.dirname(__file__), "../conf"),
    config_name="dango_ablation_baseline",
)
def main(cfg: DictConfig) -> None:
    """Train one ablation arm, then score its best-val-MSE checkpoint on test."""
    os.environ["WANDB__SERVICE_WAIT"] = "600"
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert isinstance(wandb_cfg, dict)
    print(f"arm {cfg.arm}: {json.dumps(wandb_cfg, sort_keys=True)}")

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    job_id = slurm_job_id if slurm_job_id else str(uuid.uuid4())
    hostname_job_id = f"{socket.gethostname()}-{job_id}"
    sorted_cfg = json.dumps(wandb_cfg, sort_keys=True)
    hashed_cfg = hashlib.sha256(sorted_cfg.encode("utf-8")).hexdigest()
    group = f"{hostname_job_id}_{hashed_cfg}"
    experiment_dir = osp.join(DATA_ROOT, "wandb-experiments", group)
    os.makedirs(experiment_dir, exist_ok=True)

    run = wandb.init(
        mode=WANDB_MODE,
        project=cfg.wandb.project,
        config=wandb_cfg,
        group=group,
        tags=list(cfg.wandb.tags),
        dir=experiment_dir,
        name=f"run_{group}",
    )
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        log_model=True,
        save_dir=experiment_dir,
        name=f"run_{group}",
    )

    seed = int(cfg.seed)
    L.seed_everything(seed, workers=True)

    graphs = list(cfg.cell_dataset.graphs)
    # Loads its own genome and graph and frees them on return, so it runs first.
    lambda_values = determine_lambda_values(verbose=False)
    assert set(lambda_values) == set(graphs), (sorted(lambda_values), graphs)
    print(f"lambda values: {lambda_values}")

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    cell_graph = build_cell_graph(genome, graph, graphs)
    node_ids = list(cell_graph["gene"].node_ids)
    gene_num = int(cell_graph["gene"].num_nodes)
    assert gene_num == len(node_ids) == EXPECTED_GENE_NUM, (gene_num, len(node_ids))
    edges = {n: int(cell_graph["gene", n, "gene"].num_edges) for n in graphs}
    print(f"cell graph: {gene_num} genes, edges (with self loops) {edges}")
    # The genome and graph objects are not needed past this point; drop them before
    # the DataLoader workers fork so the workers do not carry them.
    del genome, graph
    gc.collect()

    row_genes, y, splits, vocab = load_records()
    node_to_idx = {g: i for i, g in enumerate(node_ids)}
    missing = [g for g in vocab if g not in node_to_idx]
    assert not missing, (
        f"{len(missing)} build genes not in the cell graph: {missing[:5]}"
    )
    remap = np.array([node_to_idx[g] for g in vocab], dtype=np.int64)
    triples = remap[row_genes]
    split_sizes = {name: int(len(rows)) for name, rows in splits.items()}
    assert split_sizes == EXPECTED_SPLIT_SIZES, split_sizes
    datasets = {
        name: GenotypeSplit(triples[rows], y[rows]) for name, rows in splits.items()
    }
    print(
        f"records {len(y)}; "
        + ", ".join(f"{name} {len(ds)}" for name, ds in datasets.items())
    )

    dm = cfg.data_module
    loader_kwargs = dict(
        num_workers=int(dm.num_workers),
        pin_memory=bool(dm.pin_memory),
        persistent_workers=int(dm.num_workers) > 0,
        collate_fn=collate_genotypes,
    )
    train_loader = DataLoader(
        datasets["train"],
        batch_size=int(dm.batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=int(dm.eval_batch_size),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=int(dm.eval_batch_size),
        shuffle=False,
        **loader_kwargs,
    )

    device = "cuda" if cfg.trainer.accelerator == "gpu" else "cpu"
    model = Dango(
        gene_num=gene_num,
        edge_types=graphs,
        hidden_channels=int(cfg.model.hidden_channels),
        num_heads=int(cfg.model.num_heads),
        residual=cfg.model.residual,
    )
    beta_params = [n for n, _ in model.named_parameters() if "beta" in n]
    print(f"residual {model.hyper_sagnn.residual}; beta parameters {beta_params}")
    if cfg.model.residual == "plain":
        assert beta_params == [], beta_params
    else:
        assert beta_params == ["hyper_sagnn.beta1", "hyper_sagnn.beta2"], beta_params
    param_counts = model.num_parameters
    print(f"parameter counts: {param_counts}")
    wandb.log(
        {
            "model/params_pretrain_model": param_counts["pretrain_model"],
            "model/params_meta_embedding": param_counts["meta_embedding"],
            "model/params_hyper_sagnn": param_counts["hyper_sagnn"],
            "model/params_total": param_counts["total"],
        }
    )

    sched_cfg = cfg.regression_task.loss_scheduler
    scheduler = SCHEDULER_MAP[sched_cfg.type](
        transition_epoch=int(sched_cfg.transition_epoch)
    )
    print(
        f"loss scheduler {sched_cfg.type}(transition_epoch={sched_cfg.transition_epoch})"
    )
    loss_func = DangoLoss(
        edge_types=graphs,
        lambda_values=lambda_values,
        scheduler=scheduler,
        reduction="mean",
    )

    rt = wandb_cfg["regression_task"]
    task = RegressionTask(
        model=model,
        cell_graph=cell_graph,
        optimizer_config=rt["optimizer"],
        lr_scheduler_config=rt["lr_scheduler"],
        batch_size=int(dm.batch_size),
        clip_grad_norm=rt["clip_grad_norm"],
        clip_grad_norm_max_norm=rt["clip_grad_norm_max_norm"],
        plot_sample_ceiling=rt["plot_sample_ceiling"],
        plot_every_n_epochs=rt["plot_every_n_epochs"],
        loss_func=loss_func,
        grad_accumulation_schedule=None,
        device=device,
        inverse_transform=None,
        forward_transform=None,
        execution_mode="training",
    )

    checkpoint_dir = osp.join(DATA_ROOT, "models/checkpoints", group)
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Best first: trainer.test(ckpt_path="best") reads the first ModelCheckpoint.
    checkpoint_best = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=1,
        monitor="val/gene_interaction/MSE",
        mode="min",
        filename=f"{run.id}-best-{{epoch:02d}}-{{val/gene_interaction/MSE:.4f}}",
    )
    checkpoint_last = ModelCheckpoint(
        dirpath=checkpoint_dir, save_last=True, filename=f"{run.id}-last"
    )
    print(f"checkpoints: {checkpoint_dir}")

    torch.set_float32_matmul_precision("medium")
    tr = cfg.trainer
    trainer = L.Trainer(
        strategy=tr.strategy,
        accelerator=tr.accelerator,
        devices=int(tr.devices),
        num_nodes=1,
        logger=wandb_logger,
        max_epochs=int(tr.max_epochs),
        callbacks=[checkpoint_best, checkpoint_last, EpochTelemetry()],
        log_every_n_steps=10,
        limit_train_batches=tr.limit_train_batches,
        limit_val_batches=tr.limit_val_batches,
        limit_test_batches=tr.limit_test_batches,
        default_root_dir=experiment_dir,
        enable_progress_bar=False,
    )
    trainer.fit(
        model=task,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=tr.resume_ckpt_path,
    )
    final = {
        k: float(v)
        for k, v in trainer.callback_metrics.items()
        if k.startswith(("val/", "train/", "model/", "perf/"))
    }
    print(f"final-epoch metrics: {json.dumps(final, sort_keys=True)}")
    print(
        f"best val MSE {float(checkpoint_best.best_model_score):.6f} at "
        f"{checkpoint_best.best_model_path}"
    )

    # Test once, on the checkpoint selected by validation MSE. RegressionTask has no
    # on_test_epoch_end, so the test metrics are read from its collection here.
    trainer.test(model=task, dataloaders=test_loader, ckpt_path="best")
    test_metrics = {k: float(v) for k, v in task.test_metrics.compute().items()}
    test_record = {
        "arm": cfg.arm,
        "best_model_path": checkpoint_best.best_model_path,
        "best_val_mse": float(checkpoint_best.best_model_score),
        "n_test": int(task.test_metrics["MSE"].total),
    } | test_metrics
    print(f"test on best-val-MSE checkpoint: {json.dumps(test_record, sort_keys=True)}")
    run.summary.update({f"best_mse_ckpt/{k}": v for k, v in test_metrics.items()})
    with open(osp.join(checkpoint_dir, "test_metrics_best_val_mse.json"), "w") as f:
        json.dump(test_record, f, indent=2)
    wandb.finish()


if __name__ == "__main__":
    main()
