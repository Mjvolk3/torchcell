# experiments/019-simb-multimodal/scripts/train_cgt_multitask.py
# [[experiments.019-simb-multimodal.scripts.train_cgt_multitask]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/train_cgt_multitask
"""Fig-3 multitask Cell Graph Transformer training harness (SIMB 2026 WS13).

Trains the WS7 multitask CellGraphTransformer on the WS2 Fig-3 multimodal build
("one embedding, many phenotypes"). The same script trains:

* an INDIVIDUAL baseline -- a single active phenotype/head
  (``multitask.active_heads=[per_gene]`` or ``[global]`` or ``[gene_interaction]``);
* a JOINT run -- several heads at once
  (``multitask.active_heads=[gene_interaction,per_gene,global]``),

selected via the Hydra config so no code change is needed between runs.

Heads (WS7, see ``torchcell/models/equivariant_cell_graph_transformer.py``):

* ``gene_interaction`` -- the ORIGINAL scalar ``perturbation_head`` (Type II
  readout), returned as the first forward output. Fitness / gene-interaction.
* ``per_gene`` -- ``PerGeneHead`` (graph_level ``node``): expression
  (Kemmeren / Sameith microarray) and proteome, ``[B, N]``.
* ``global`` -- ``GlobalHead`` (graph_level ``global``): CalMorph morphology
  (501-D) and scalar VisualScore, ``[B, output_dim]``.
* ``per_metabolite`` -- ``PerMetaboliteHead`` (graph_level ``metabolism``),
  ``[B, M]`` (needs the metabolism incidence graph in ``cell_graph``).

The loss is ``MaskedMultitaskLoss``: each head's loss is masked to the genotypes
in the batch that actually carry that phenotype (sparse supervision), and the
graph-regularization attention term is added unchanged.

DRY-RUN (``dry_run=true``): build the model straight from ``model`` + ``multitask``
config and run ONE synthetic forward + masked-loss + backward on a tiny synthetic
``cell_graph``/``batch`` (mirrors the WS7 unit test). No genome, dataset, wandb, or
GPU required -- this is the local wiring check that runs anywhere. Use it (and
``--help`` / ``--cfg job``) to validate a config before shipping to a cluster.

Assumptions flagged for validation on the first REAL Fig-3 batch (see the WS13
note): (1) the dataset is (re)built with the ``Perturbation`` graph processor --
the transformer consumes per-genotype ``perturbation_indices`` batches, NOT the
``SubgraphRepresentation`` used by ``query_fig3.py`` for the census; (2) per-head
targets/masks are decoded from the COO ``phenotype_values`` /
``phenotype_type_indices`` / ``phenotype_sample_indices`` fields keyed by the
config ``multitask.head_phenotypes`` map. The synthetic dry-run does NOT exercise
(2); the first cluster run must confirm the COO decode against a materialized batch.
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
import uuid
from typing import Any, cast

import hydra
import lightning as L
import torch
import torch.distributed as dist
import wandb
from dotenv import load_dotenv
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import HeteroData

from torchcell.models.equivariant_cell_graph_transformer import (
    CellGraphTransformer,
    MaskedMultitaskLoss,
)
from torchcell.timestamp import timestamp

log = logging.getLogger(__name__)
load_dotenv()
WANDB_MODE = os.getenv("WANDB_MODE")


def _as_dict(node: Any) -> dict[str, Any]:
    """Resolve an OmegaConf node to a plain ``dict[str, Any]`` (typed for mypy)."""
    return cast(dict[str, Any], OmegaConf.to_container(node, resolve=True))


def get_slurm_nodes() -> int:
    if "SLURM_NNODES" in os.environ:
        return int(os.environ["SLURM_NNODES"])
    if "SLURM_JOB_NUM_NODES" in os.environ:
        return int(os.environ["SLURM_JOB_NUM_NODES"])
    return 1


def get_num_devices(cfg_devices: Any) -> int:
    if cfg_devices != "auto":
        return int(cfg_devices)
    slurm_devices = os.environ.get("SLURM_GPUS_ON_NODE")
    if slurm_devices is not None:
        return int(slurm_devices)
    num_devices = torch.cuda.device_count()
    return num_devices if num_devices > 0 else 1


def build_heads_config(cfg: DictConfig) -> dict[str, Any] | None:
    """Assemble the model ``heads_config`` from the active multitask heads.

    ``gene_interaction`` maps to the built-in ``perturbation_head`` (always present)
    and therefore contributes NO entry to ``heads_config``. Only the multitask heads
    (``global`` / ``per_gene`` / ``per_metabolite``) are declared here.
    """
    active = list(cfg.multitask.active_heads)
    head_specs = OmegaConf.to_container(cfg.multitask.heads, resolve=True)
    assert isinstance(head_specs, dict)
    heads_config: dict[str, Any] = {}
    for head in active:
        if head == "gene_interaction":
            continue
        if head not in head_specs:
            raise ValueError(
                f"active head '{head}' has no spec under multitask.heads "
                f"(available: {sorted(head_specs)})"
            )
        heads_config[head] = head_specs[head]
    return heads_config or None


class MultitaskCGTTask(L.LightningModule):
    """Lightning wrapper: multitask CGT + ``MaskedMultitaskLoss`` + optim/sched.

    Kept deliberately self-contained (not reusing the single-head ``RegressionTask``)
    so the masked-multitask supervision path is explicit. Metrics are per-head loss
    + a Pearson correlation on the ``gene_interaction`` head when it is active.
    """

    def __init__(
        self,
        model: CellGraphTransformer,
        cell_graph: HeteroData,
        active_heads: list[str],
        head_weights: dict[str, float],
        head_phenotypes: dict[str, list[str]],
        loss_fn: str,
        optimizer_config: dict[str, Any],
        lr_scheduler_config: dict[str, Any] | None,
        clip_grad_norm: bool,
        clip_grad_norm_max_norm: float,
    ) -> None:
        """Store the model, cell_graph, masked loss, and optim/sched config."""
        super().__init__()
        self.model = model
        self.cell_graph = cell_graph.clone()
        self.active_heads = active_heads
        self.head_phenotypes = head_phenotypes
        self.loss = MaskedMultitaskLoss(head_weights=head_weights, loss_fn=loss_fn)
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_norm_max_norm = clip_grad_norm_max_norm
        self.save_hyperparameters(ignore=["model", "cell_graph"])

    def forward(self, batch: HeteroData) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run the multitask CGT on a batch, moving cell_graph to its device."""
        if self.cell_graph["gene"].x.device != batch["gene"].x.device:
            self.cell_graph = self.cell_graph.to(batch["gene"].x.device)
        return cast(
            "tuple[torch.Tensor, dict[str, Any]]",
            self.model(self.cell_graph, batch),
        )

    def _batch_size(self, batch: HeteroData) -> int:
        return int(batch["gene"].perturbation_indices_batch.max().item() + 1)

    def _extract_targets_and_masks(
        self, batch: HeteroData, head_outputs: dict[str, torch.Tensor], bsz: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Decode per-head targets + supervision masks from the COO phenotype fields.

        ASSUMPTION (flagged in the module docstring + WS13 note): the batch carries
        the COO triplet ``phenotype_values`` / ``phenotype_type_indices`` /
        ``phenotype_sample_indices`` on ``batch['gene']`` plus a per-row
        ``phenotype_types`` list, produced by the graph processor. Each active head
        gathers the rows whose phenotype name is listed in
        ``multitask.head_phenotypes[head]``. Where a modality is absent for a
        genotype, the mask is False so ``MaskedMultitaskLoss`` skips it.

        This decode MUST be validated against a materialized Fig-3 batch before a
        production run; the synthetic dry-run does not touch it.
        """
        device = head_outputs[next(iter(head_outputs))].device
        gene = batch["gene"]
        values = getattr(gene, "phenotype_values", None)
        type_idx = getattr(gene, "phenotype_type_indices", None)
        sample_idx = getattr(gene, "phenotype_sample_indices", None)
        pheno_types = getattr(gene, "phenotype_types", None)
        targets: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}
        if values is None or type_idx is None or sample_idx is None:
            return targets, masks

        for head, pred in head_outputs.items():
            names = self.head_phenotypes.get(head, [])
            if not names or pheno_types is None:
                continue
            wanted = {i for i, nm in enumerate(pheno_types) if nm in names}
            if not wanted:
                continue
            row_mask = torch.zeros(bsz, dtype=torch.bool, device=device)
            target = torch.zeros_like(pred)
            sel = torch.tensor(
                [int(t.item()) in wanted for t in type_idx],
                dtype=torch.bool,
                device=values.device,
            )
            for s in sample_idx[sel].unique():
                row_mask[int(s.item())] = True
            # Scatter the observed values into the target tensor by sample row.
            for v, s in zip(values[sel], sample_idx[sel]):
                r = int(s.item())
                if target.dim() == 1:
                    target[r] = v
                else:
                    target[r] = target[r] + v
            targets[head] = target
            masks[head] = row_mask
        return targets, masks

    def _step(self, batch: HeteroData, stage: str) -> torch.Tensor:
        predictions, reps = self(batch)
        bsz = self._batch_size(batch)
        head_outputs: dict[str, torch.Tensor] = dict(reps["head_outputs"])
        if "gene_interaction" in self.active_heads:
            head_outputs["gene_interaction"] = predictions.squeeze(-1)
        targets, masks = self._extract_targets_and_masks(batch, head_outputs, bsz)
        total, per_head = self.loss(
            head_outputs, targets, masks, graph_reg_loss=reps["graph_reg_loss"]
        )
        self.log(f"{stage}/loss", total, batch_size=bsz, sync_dist=True)
        for name, val in per_head.items():
            self.log(f"{stage}/{name}/loss", val, batch_size=bsz, sync_dist=True)
        return cast(torch.Tensor, total)

    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Masked multitask training step."""
        return self._step(batch, "train")

    def validation_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Masked multitask validation step."""
        return self._step(batch, "val")

    def configure_optimizers(self) -> Any:
        """Build the optimizer and optional LR scheduler from config."""
        opt_cfg = dict(self.optimizer_config)
        opt_class = getattr(torch.optim, opt_cfg.pop("type"))
        optimizer = opt_class(self.parameters(), **opt_cfg)
        if self.lr_scheduler_config is None:
            return optimizer
        sched_cfg = dict(self.lr_scheduler_config)
        sched_type = sched_cfg.pop("type", "ReduceLROnPlateau")
        scheduler: Any
        if sched_type == "CosineAnnealingWarmupRestarts":
            from torchcell.scheduler.cosine_annealing_warmup import (
                CosineAnnealingWarmupRestarts,
            )

            scheduler = CosineAnnealingWarmupRestarts(optimizer, **sched_cfg)
        elif sched_type == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **sched_cfg
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **sched_cfg
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        """Clip gradients by norm when configured."""
        if self.clip_grad_norm:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.clip_grad_norm_max_norm,
                gradient_clip_algorithm="norm",
            )


def _synthetic_cell_graph(gene_num: int, with_metabolism: bool) -> HeteroData:
    """Tiny synthetic ``cell_graph`` for the dry-run (mirrors the WS7 test)."""
    cg = HeteroData()
    cg["gene"].num_nodes = gene_num
    cg["gene"].x = torch.zeros(gene_num, 1)
    src = torch.arange(gene_num - 1)
    dst = torch.arange(1, gene_num)
    cg["gene", "physical", "gene"].edge_index = torch.stack([src, dst])
    if with_metabolism:
        num_react = max(2, gene_num // 2)
        num_met = max(2, gene_num // 3)
        g = torch.arange(gene_num)
        r = g % num_react
        cg["gene", "gpr", "reaction"].edge_index = torch.stack([g, r])
        cg["reaction"].num_nodes = num_react
        mr = torch.arange(num_react)
        m = mr % num_met
        cg["metabolite", "reaction", "metabolite"].edge_index = torch.stack([m, mr])
        cg["metabolite"].num_nodes = num_met
    return cg


def _synthetic_batch(gene_num: int, batch_size: int) -> HeteroData:
    """Tiny synthetic perturbation batch for the dry-run."""
    batch = HeteroData()
    batch["gene"].x = torch.zeros(gene_num, 1)
    per = min(2, gene_num)
    idx = []
    bidx = []
    for b in range(batch_size):
        genes = torch.randperm(gene_num)[:per]
        idx.append(genes)
        bidx.append(torch.full((per,), b, dtype=torch.long))
    batch["gene"].perturbation_indices = torch.cat(idx)
    batch["gene"].perturbation_indices_batch = torch.cat(bidx)
    return batch


def run_dry_run(cfg: DictConfig) -> None:
    """Build the model from config and run ONE synthetic forward + masked loss.

    Exercises heads_config + MaskedMultitaskLoss wiring end-to-end with NO dataset,
    genome, wandb, or GPU. This is the local verification bar for WS13.
    """
    print(f"[dry-run] Building multitask CGT from config ({timestamp()})")
    heads_config = build_heads_config(cfg)
    print(f"[dry-run] active_heads={list(cfg.multitask.active_heads)}")
    print(f"[dry-run] heads_config={heads_config}")

    gene_num = int(cfg.model.gene_num)
    # Keep the synthetic graph tiny regardless of the real gene_num.
    gene_num = min(gene_num, 16)
    batch_size = 3
    with_metabolism = heads_config is not None and "per_metabolite" in heads_config

    cell_graph = _synthetic_cell_graph(gene_num, with_metabolism)
    batch = _synthetic_batch(gene_num, batch_size)

    model = CellGraphTransformer(
        gene_num=gene_num,
        hidden_channels=int(cfg.model.hidden_channels),
        num_transformer_layers=int(cfg.model.num_transformer_layers),
        num_attention_heads=int(cfg.model.num_attention_heads),
        cell_graph=cell_graph,
        graph_regularization_config=_as_dict(cfg.model.graph_regularization),
        perturbation_head_config=_as_dict(cfg.model.perturbation_head),
        dropout=float(cfg.model.dropout),
        graph_reg_lambda=0.0,
        learnable_embedding_config=_as_dict(cfg.model.learnable_embedding),
        heads_config=heads_config,
    )
    print("[dry-run] parameter counts:", model.num_parameters)

    predictions, reps = model(cell_graph, batch)
    print(f"[dry-run] gene_interaction prediction shape: {tuple(predictions.shape)}")
    head_outputs = dict(reps["head_outputs"])
    if "gene_interaction" in cfg.multitask.active_heads:
        head_outputs["gene_interaction"] = predictions.squeeze(-1)
    for name, out in head_outputs.items():
        print(f"[dry-run]   head '{name}' output shape: {tuple(out.shape)}")

    # One synthetic masked-loss + backward to prove the training path is connected.
    loss_fn = MaskedMultitaskLoss(loss_fn=str(cfg.multitask.loss_fn))
    targets = {k: torch.randn_like(v) for k, v in head_outputs.items()}
    masks = {
        k: torch.randint(0, 2, (batch_size,), dtype=torch.bool)
        for k in head_outputs
    }
    total, per_head = loss_fn(
        head_outputs, targets, masks, graph_reg_loss=reps["graph_reg_loss"]
    )
    total.backward()
    print(f"[dry-run] masked total loss: {total.item():.6f}")
    print(f"[dry-run] per-head losses: { {k: round(v.item(), 6) for k, v in per_head.items()} }")
    grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    print(f"[dry-run] summed grad norm (nonzero => backward connected): {grad_norm:.6f}")
    print("[dry-run] OK -- model + heads + masked loss wired correctly.")


def run_training(cfg: DictConfig) -> None:
    """Full training path: genome/graph/embeddings/dataset/datamodule + Trainer."""
    # Deferred heavy imports so --help / dry-run never pay for them.
    from torch_geometric.transforms import Compose

    from torchcell.data import (
        GenotypeAggregator,
        MeanExperimentDeduplicator,
        Neo4jCellDataset,
    )
    from torchcell.data.graph_processor import Perturbation
    from torchcell.datamodules import CellDataModule
    from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
    from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.graph.graph import build_gene_multigraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
    from torchcell.transforms.coo_regression_to_classification import (
        COOLabelNormalizationTransform,
    )

    data_root = os.environ["DATA_ROOT"]
    experiment_root = os.environ["EXPERIMENT_ROOT"]

    os.environ["WANDB__SERVICE_WAIT"] = "600"
    if not (dist.is_available() and dist.is_initialized()):
        os.environ["TORCH_DISTRIBUTED_DEFAULT_TIMEOUT"] = "7200"

    wandb_cfg = _as_dict(cfg)

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    job_id = slurm_job_id or str(uuid.uuid4())
    hostname = socket.gethostname()
    hashed_cfg = hashlib.sha256(
        json.dumps(wandb_cfg, sort_keys=True).encode("utf-8")
    ).hexdigest()
    group = f"{hostname}-{job_id}_{hashed_cfg}"
    experiment_dir = osp.join(data_root, "wandb-experiments", group)
    os.makedirs(experiment_dir, exist_ok=True)

    run = wandb.init(
        mode=cast(Any, WANDB_MODE),
        project=wandb_cfg["wandb"]["project"],
        config=wandb_cfg,
        group=group,
        tags=wandb_cfg["wandb"]["tags"],
        dir=experiment_dir,
        name=f"run_{group}",
    )
    wandb_logger = WandbLogger(
        project=wandb_cfg["wandb"]["project"],
        log_model=True,
        save_dir=experiment_dir,
        name=f"run_{group}",
    )

    if torch.cuda.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        genome_root = osp.join(data_root, f"data/sgd/genome_{rank}")
        go_root = osp.join(data_root, f"data/go/go_{rank}")
    else:
        genome_root = osp.join(data_root, "data/sgd/genome")
        go_root = osp.join(data_root, "data/go")

    genome = SCerevisiaeGenome(
        genome_root=genome_root, go_root=go_root, overwrite=False
    )
    genome.drop_empty_go()
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(data_root, "data/sgd/genome"),
        string_root=osp.join(data_root, "data/string"),
        tflink_root=osp.join(data_root, "data/tflink"),
        genome=genome,
    )

    graph_names = list(cfg.cell_dataset.graphs)
    gene_multigraph = build_gene_multigraph(graph=graph, graph_names=graph_names)
    node_embedding_names = list(cfg.cell_dataset.get("node_embeddings", []))
    node_embeddings = NodeEmbeddingBuilder.build(
        embedding_names=node_embedding_names,
        data_root=data_root,
        genome=genome,
        graph=graph,
    )

    # Fig-3 query (WS2). NOTE: built with the Perturbation processor for the
    # transformer, not the SubgraphRepresentation used by query_fig3.py's census.
    query_path = osp.join(
        experiment_root, "019-simb-multimodal/queries", cfg.cell_dataset.query_file
    )
    with open(query_path) as f:
        query = f.read()
    dataset_root = osp.join(
        data_root,
        "data/torchcell/experiments/019-simb-multimodal",
        cfg.cell_dataset.dataset_tag,
    )

    incidence_graphs = None
    if "per_metabolite" in list(cfg.multitask.active_heads):
        from torchcell.metabolism.yeast_GEM import YeastGEM

        incidence_graphs = {"metabolism_bipartite": YeastGEM().bipartite_graph}

    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=gene_multigraph,
        incidence_graphs=incidence_graphs,
        node_embeddings=node_embeddings,
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=Perturbation(),
        transform=None,
    )

    if cfg.transforms.get("use_transforms", False):
        transform_config = cfg.transforms.forward_transform
        transforms_list = []
        if "normalization" in transform_config:
            norm = COOLabelNormalizationTransform(
                dataset, _as_dict(transform_config.normalization)
            )
            transforms_list.append(norm)
        if transforms_list:
            dataset.transform = Compose(transforms_list)
    print(f"Dataset Length: {len(dataset)}")

    seed = 42
    follow_batch = ["perturbation_indices"]
    data_module: Any = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=cfg.data_module.batch_size,
        random_seed=seed,
        num_workers=cfg.data_module.num_workers,
        pin_memory=cfg.data_module.pin_memory,
        prefetch=cfg.data_module.prefetch,
        follow_batch=follow_batch,
    )
    data_module.setup()

    if cfg.data_module.is_perturbation_subset:
        data_module = PerturbationSubsetDataModule(
            cell_data_module=data_module,
            size=int(cfg.data_module.perturbation_subset_size),
            batch_size=cfg.data_module.batch_size,
            num_workers=cfg.data_module.num_workers,
            pin_memory=cfg.data_module.pin_memory,
            prefetch=cfg.data_module.prefetch,
            seed=seed,
            follow_batch=follow_batch,
        )
        data_module.setup()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    heads_config = build_heads_config(cfg)
    model = CellGraphTransformer(
        gene_num=cfg.model.gene_num,
        hidden_channels=cfg.model.hidden_channels,
        num_transformer_layers=cfg.model.num_transformer_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        cell_graph=dataset.cell_graph,
        graph_regularization_config=_as_dict(cfg.model.graph_regularization),
        perturbation_head_config=_as_dict(cfg.model.perturbation_head),
        dropout=cfg.model.dropout,
        graph_reg_lambda=float(cfg.model.graph_regularization.graph_reg_lambda),
        node_embeddings=node_embeddings,
        learnable_embedding_config=_as_dict(cfg.model.learnable_embedding),
        heads_config=heads_config,
    ).to(device)
    print("Parameter counts:", model.num_parameters)

    head_weights = _as_dict(cfg.multitask.head_weights)
    head_phenotypes = _as_dict(cfg.multitask.head_phenotypes)

    task = MultitaskCGTTask(
        model=model,
        cell_graph=dataset.cell_graph,
        active_heads=list(cfg.multitask.active_heads),
        head_weights={k: float(v) for k, v in head_weights.items()},
        head_phenotypes={k: list(v) for k, v in head_phenotypes.items()},
        loss_fn=str(cfg.multitask.loss_fn),
        optimizer_config=_as_dict(cfg.regression_task.optimizer),
        lr_scheduler_config=(
            _as_dict(cfg.regression_task.lr_scheduler)
            if cfg.regression_task.get("lr_scheduler") is not None
            else None
        ),
        clip_grad_norm=cfg.regression_task.clip_grad_norm,
        clip_grad_norm_max_norm=cfg.regression_task.clip_grad_norm_max_norm,
    )

    model_base_path = osp.join(data_root, "models/checkpoints")
    os.makedirs(model_base_path, exist_ok=True)
    checkpoint_callbacks: list[Callback] = [
        ModelCheckpoint(
            dirpath=osp.join(model_base_path, group),
            save_top_k=1,
            monitor="val/loss",
            mode="min",
            filename=f"{run.id}-best-{{epoch:02d}}-{{val/loss:.4f}}",
        ),
        ModelCheckpoint(
            dirpath=osp.join(model_base_path, group),
            save_last=True,
            filename=f"{run.id}-last",
        ),
    ]

    torch.set_float32_matmul_precision("medium")
    devices = get_num_devices(cfg.trainer.devices)
    print(f"devices: {devices} ({timestamp()})")

    trainer = L.Trainer(
        strategy=cfg.trainer.strategy,
        accelerator=cfg.trainer.accelerator,
        devices=devices,
        num_nodes=get_slurm_nodes(),
        logger=wandb_logger,
        max_epochs=cfg.trainer.max_epochs,
        callbacks=checkpoint_callbacks,
        log_every_n_steps=10,
        overfit_batches=cfg.trainer.get("overfit_batches", 0),
        precision=cfg.trainer.get("precision", "32-true"),
        fast_dev_run=cfg.trainer.get("fast_dev_run", False),
    )
    trainer.fit(model=task, datamodule=data_module)
    wandb.finish()


@hydra.main(
    version_base=None,
    config_path=osp.join(osp.dirname(__file__), "../conf"),
    config_name="train_cgt_multitask",
)
def main(cfg: DictConfig) -> None:
    print("Multitask Cell Graph Transformer (Fig-3 / WS13) 🚀")
    if cfg.get("dry_run", False):
        run_dry_run(cfg)
        return
    run_training(cfg)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
