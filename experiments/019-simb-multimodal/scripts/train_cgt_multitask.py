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

Two WS13 assumptions are now VALIDATED against a materialized Fig-3 batch (WS10a):
(1) the dataset is built with the ``Perturbation`` graph processor -- the
transformer consumes per-genotype ``perturbation_indices`` batches, NOT the
``SubgraphRepresentation`` used by ``query_fig3.py`` for the census (CONFIRMED
correct). (2) Per-head targets/masks are decoded from the COO ``phenotype_values``
/ ``phenotype_type_indices`` fields, but the placeholder assumptions in the WS13
note were WRONG and are FIXED here: the real ``phenotype_types`` strings are
``fitness`` / ``calmorph`` / ``expression_log2_ratio`` (not microarray_/rnaseq_);
the batch-row map is ``phenotype_values_batch`` (needs
``follow_batch=['phenotype_values']``), NOT ``phenotype_sample_indices`` (which
indexes experiments within a genotype and does not offset across the batch); and
``phenotype_types`` collates to a per-graph list-of-lists with graph-LOCAL type
indices. Vector heads are aligned to the measured feature subset via
``build_head_alignments`` (per_gene gathers to the 6127 measured-gene node columns;
global is the 281-D CalMorph vector). The synthetic dry-run does NOT exercise the
decode; ``_extract_targets_and_masks`` is what carries it.
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
import numpy as np
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


def _pearson(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pearson correlation between two flattened tensors (grad-free metric)."""
    p = pred.reshape(-1).float()
    t = target.reshape(-1).float()
    p = p - p.mean()
    t = t - t.mean()
    denom = (p.norm() * t.norm()).clamp_min(1e-8)
    return cast(torch.Tensor, (p * t).sum() / denom)


def _vector_phenotype_keys(dataset: Any, name: str, scan: int = 5000) -> list[str] | None:
    """Return the sorted dict keys of the first vector phenotype ``name`` found.

    The ``Perturbation`` processor flattens a dict-valued phenotype to a
    key-sorted vector and DROPS the keys, so the per-value feature identity
    (which gene an expression value belongs to; which CalMorph parameter) is not
    recoverable from a built HeteroData sample. We recover it once from the raw
    reconstructed experiment records (same key-sort the processor uses), so the
    decode can align each head to its target.
    """
    n = min(len(dataset), scan)
    if dataset.env is None:
        dataset._init_lmdb_read()
    for idx in range(n):
        raw = dataset._read_from_lmdb(idx)
        if raw is None:
            continue
        for item in dataset._deserialize_json(raw):
            value = item["experiment"]["phenotype"].get(name)
            if isinstance(value, dict):
                return sorted(value.keys())
    return None


def build_head_alignments(
    dataset: Any,
    active_heads: list[str],
    head_phenotypes: dict[str, list[str]],
    node_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Precompute per-head COO->target alignment from the real Fig-3 build.

    For each active head we resolve its phenotype name to the actual COO layout:

    * ``gene_interaction`` -- SCALAR (fitness); ``{"is_scalar": True}``.
    * ``per_gene`` -- expression, a per-gene VECTOR. The processor emits one value
      per MEASURED gene in key-sorted order; the ``per_gene`` head emits one value
      per graph NODE. We build ``col_idx`` (node positions of the measured genes
      that exist in the cell graph) so the prediction is gathered to the measured
      subset, and ``keep_mask`` (over the raw key-sorted vector) dropping measured
      genes absent from the gene set -- so gathered prediction and target align 1:1.
    * ``global`` / ``per_metabolite`` -- a fixed-length feature VECTOR (CalMorph 281,
      metabolites M); identity alignment, ``feat_dim`` = vector length.
    """
    nid_to_pos = {n: i for i, n in enumerate(node_ids)}
    align: dict[str, dict[str, Any]] = {}
    for head in active_heads:
        if head == "gene_interaction":
            align[head] = {"is_scalar": True}
            continue
        keys: list[str] | None = None
        for name in head_phenotypes.get(head, []):
            keys = _vector_phenotype_keys(dataset, name)
            if keys is not None:
                break
        if keys is None:
            # No such vector phenotype present in this build -> head unsupervised.
            align[head] = {
                "is_scalar": False,
                "keep_mask": None,
                "col_idx": None,
                "feat_dim": None,
            }
            continue
        if head == "per_gene":
            keep = torch.tensor(
                [k in nid_to_pos for k in keys], dtype=torch.bool
            )
            col = torch.tensor(
                [nid_to_pos[k] for k in keys if k in nid_to_pos], dtype=torch.long
            )
            align[head] = {
                "is_scalar": False,
                "keep_mask": keep,
                "col_idx": col,
                "feat_dim": int(keep.sum().item()),
            }
        else:
            align[head] = {
                "is_scalar": False,
                "keep_mask": None,
                "col_idx": None,
                "feat_dim": len(keys),
            }
    return align


def compute_per_feature_target_stats(
    dataset: Any,
    train_indices: list[int],
    active_heads: list[str],
    head_phenotypes: dict[str, list[str]],
    head_align: dict[str, dict[str, Any]],
    heads_to_normalize: list[str],
    eps: float,
    degenerate_robust_cv: float,
) -> dict[str, dict[str, Any]]:
    """Per-FEATURE target standardization stats, computed on the TRAIN split ONLY.

    WS10b. The morphology target (``calmorph``) is a 281-D vector whose features span
    ~8 orders of magnitude (cell-size counts ~1e4 vs 0--1 ratios), so a single pooled
    mean/std (what ``COOLabelNormalizationTransform`` would compute for the whole
    ``calmorph`` label) leaves the large-scale features dominating an un-normalized MSE
    (loss O(1e6)). Instead we z-score EACH feature independently.

    Stats are accumulated over ``train_indices`` alone (never val/test) so there is no
    leakage; the same affine is inverted at report/inference time. ``std`` is floored
    with ``eps`` so a truly constant feature maps to ~0 rather than exploding.

    For each vector head in ``heads_to_normalize`` we resolve its key-sorted feature
    vocabulary (restricted by the head's ``keep_mask`` so it matches the gathered
    prediction/target length), read every train genotype's vector, and compute per-feature
    mean/std/median/IQR. Near-constant features (robust CV = IQR/|median| below
    ``degenerate_robust_cv``) are FLAGGED (not dropped) -- the epsilon floor keeps them
    finite; the user decides whether to drop.
    """
    stats: dict[str, dict[str, Any]] = {}
    for head in active_heads:
        if head not in heads_to_normalize:
            continue
        align = head_align.get(head, {})
        if align.get("is_scalar", False) or align.get("feat_dim") is None:
            continue
        keys: list[str] | None = None
        name: str | None = None
        for cand in head_phenotypes.get(head, []):
            keys = _vector_phenotype_keys(dataset, cand)
            if keys is not None:
                name = cand
                break
        if keys is None or name is None:
            continue
        keep = align.get("keep_mask")
        keep_list = keep.tolist() if keep is not None else [True] * len(keys)
        kept_keys = [k for k, flag in zip(keys, keep_list) if flag]

        collected: list[list[float]] = []
        for idx in train_indices:
            raw = dataset._read_from_lmdb(idx)
            if raw is None:
                continue
            for item in dataset._deserialize_json(raw):
                value = item["experiment"]["phenotype"].get(name)
                if not isinstance(value, dict):
                    continue
                vec = [
                    float(value[k])
                    for k, flag in zip(keys, keep_list)
                    if flag and k in value
                ]
                if len(vec) == len(kept_keys):
                    collected.append(vec)

        arr = np.asarray(collected, dtype=float)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        median = np.nanmedian(arr, axis=0)
        q25 = np.nanpercentile(arr, 25, axis=0)
        q75 = np.nanpercentile(arr, 75, axis=0)
        iqr = q75 - q25
        with np.errstate(divide="ignore", invalid="ignore"):
            robust_cv = np.where(np.abs(median) > 0, iqr / np.abs(median), 0.0)
        degenerate_idx = [
            i for i, rc in enumerate(robust_cv) if rc < degenerate_robust_cv
        ]
        stats[head] = {
            "keys": kept_keys,
            "mean": torch.tensor(mean, dtype=torch.float32),
            "std": torch.tensor(std, dtype=torch.float32),
            "eps": float(eps),
            "n_train": int(arr.shape[0]),
            "degenerate_features": [kept_keys[i] for i in degenerate_idx],
            "robust_cv": robust_cv.tolist(),
        }
    return stats


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
        head_align: dict[str, dict[str, Any]],
        loss_fn: str,
        optimizer_config: dict[str, Any],
        lr_scheduler_config: dict[str, Any] | None,
        clip_grad_norm: bool,
        clip_grad_norm_max_norm: float,
        target_stats: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Store the model, cell_graph, masked loss, and optim/sched config."""
        super().__init__()
        self.model = model
        self.cell_graph = cell_graph.clone()
        self.active_heads = active_heads
        self.head_phenotypes = head_phenotypes
        self.head_align = head_align
        self.loss = MaskedMultitaskLoss(head_weights=head_weights, loss_fn=loss_fn)
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_norm_max_norm = clip_grad_norm_max_norm
        # WS10b per-feature target standardization (TRAIN-split stats). Registered as
        # buffers so mean/std move with .to(device) and are checkpointed for inversion.
        self.norm_heads: list[str] = list((target_stats or {}).keys())
        self.norm_eps: dict[str, float] = {}
        for head, st in (target_stats or {}).items():
            safe = head.replace("/", "_")
            self.register_buffer(f"_norm_mean_{safe}", st["mean"], persistent=True)
            self.register_buffer(f"_norm_std_{safe}", st["std"], persistent=True)
            self.norm_eps[head] = float(st["eps"])
        self.save_hyperparameters(
            ignore=["model", "cell_graph", "head_align", "target_stats"]
        )

    def _norm_mean_std(self, head: str) -> tuple[torch.Tensor, torch.Tensor]:
        safe = head.replace("/", "_")
        return (
            cast(torch.Tensor, getattr(self, f"_norm_mean_{safe}")),
            cast(torch.Tensor, getattr(self, f"_norm_std_{safe}")),
        )

    def _normalize_target(self, head: str, values: torch.Tensor) -> torch.Tensor:
        """Per-feature z-score a decoded [B, feat] target for a normalized head."""
        mean, std = self._norm_mean_std(head)
        return (values - mean) / (std + self.norm_eps[head])

    def denormalize(self, head: str, values: torch.Tensor) -> torch.Tensor:
        """Invert the per-feature z-score back to raw CalMorph scale (inference)."""
        mean, std = self._norm_mean_std(head)
        return values * (std + self.norm_eps[head]) + mean

    def forward(self, batch: HeteroData) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run the multitask CGT on a batch, moving cell_graph to its device.

        The ``Perturbation`` batch has NO ``gene.x`` (only ``perturbation_indices`` +
        the phenotype COO), so the device is taken from ``perturbation_indices``.
        """
        dev = batch["gene"].perturbation_indices.device
        if getattr(self, "_cell_graph_device", None) != dev:
            self.cell_graph = self.cell_graph.to(dev)
            self._cell_graph_device = dev
        return cast(
            "tuple[torch.Tensor, dict[str, Any]]",
            self.model(self.cell_graph, batch),
        )

    def _batch_size(self, batch: HeteroData) -> int:
        return int(batch["gene"].perturbation_indices_batch.max().item() + 1)

    def _gather_predictions(
        self, head_outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Gather VECTOR head predictions to the measured-feature subset.

        ``per_gene`` predicts one value per graph NODE ([B, N]); expression is only
        measured on a gene subset. ``col_idx`` selects the measured-gene columns so
        the gathered prediction ([B, feat_dim]) aligns 1:1 with the decoded target.
        Heads with no ``col_idx`` (scalar, global/per_metabolite identity) pass through.
        """
        out = dict(head_outputs)
        for head, pred in head_outputs.items():
            col = self.head_align.get(head, {}).get("col_idx")
            if col is not None:
                out[head] = pred.index_select(1, col.to(pred.device))
        return out

    def _extract_targets_and_masks(
        self, batch: HeteroData, head_outputs: dict[str, torch.Tensor], bsz: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Decode per-head targets + supervision masks from the COO phenotype fields.

        VALIDATED against a materialized Fig-3 batch (WS10a). The batch carries the
        COO triplet ``phenotype_values`` / ``phenotype_type_indices`` /
        ``phenotype_values_batch`` on ``batch['gene']`` (the last requires
        ``follow_batch=['phenotype_values']``), plus ``phenotype_types`` -- a
        PER-GRAPH list-of-lists after collation (one type-name list per genotype).

        Two collation facts drive this decode (both learned empirically, both broke
        the earlier assumption-based version):

        * ``phenotype_sample_indices`` is NOT the batch row -- it indexes the
          experiments WITHIN a genotype group and is not offset across the batch, so
          all graphs collide at 0,1,2. The batch row comes from
          ``phenotype_values_batch`` instead.
        * ``phenotype_types`` collates to a list-of-lists; type indices are
          LOCAL to each graph, so a value's name is ``phenotype_types[b][type_idx]``.

        For each active head, per graph ``b`` we select the values whose local name
        is in ``head_phenotypes[head]``; if any, the mask is True and the target row
        is those values (scalar mean for scalar heads; the key-sorted vector for
        vector heads, restricted by ``keep_mask`` to the measured features that the
        gathered prediction covers). Absent modalities keep mask False so
        ``MaskedMultitaskLoss`` skips them.
        """
        device = head_outputs[next(iter(head_outputs))].device
        gene = batch["gene"]
        values = getattr(gene, "phenotype_values", None)
        type_idx = getattr(gene, "phenotype_type_indices", None)
        val_batch = getattr(gene, "phenotype_values_batch", None)
        pheno_types = gene["phenotype_types"] if "phenotype_types" in gene else None
        targets: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}
        if values is None or type_idx is None or val_batch is None or pheno_types is None:
            return targets, masks

        # Normalize phenotype_types to a per-graph list-of-lists (single-graph batch
        # collates to a flat list of strings).
        if len(pheno_types) > 0 and isinstance(pheno_types[0], str):
            per_graph_types: list[list[str]] = [list(pheno_types)]
        else:
            per_graph_types = [list(t) for t in pheno_types]

        for head, pred in head_outputs.items():
            names = set(self.head_phenotypes.get(head, []))
            if not names:
                continue
            align = self.head_align.get(head, {})
            is_scalar = bool(align.get("is_scalar", False))
            keep = align.get("keep_mask")
            row_mask = torch.zeros(bsz, dtype=torch.bool, device=device)
            target = torch.zeros_like(pred)
            for b in range(bsz):
                sel_b = val_batch == b
                if not bool(sel_b.any()):
                    continue
                gtypes = per_graph_types[b]
                tb = type_idx[sel_b].tolist()
                vb = values[sel_b]
                name_sel = torch.tensor(
                    [gtypes[t] in names for t in tb], dtype=torch.bool
                )
                if not bool(name_sel.any()):
                    continue
                head_vals = vb[name_sel]
                if is_scalar:
                    target[b] = head_vals.float().mean().to(device)
                else:
                    if keep is not None:
                        head_vals = head_vals[keep]
                    target[b] = head_vals.to(device)
                row_mask[b] = True
            # WS10b: per-feature z-score (TRAIN-split stats) so a multi-scale vector
            # target (CalMorph 281-D) yields an O(1) loss. Masked rows are zeros ->
            # become -mean/std but are dropped by MaskedMultitaskLoss's row mask.
            if head in self.norm_heads:
                target = self._normalize_target(head, target)
            targets[head] = target
            masks[head] = row_mask
        return targets, masks

    def _step(self, batch: HeteroData, stage: str) -> torch.Tensor:
        predictions, reps = self(batch)
        bsz = self._batch_size(batch)
        head_outputs: dict[str, torch.Tensor] = dict(reps["head_outputs"])
        if "gene_interaction" in self.active_heads:
            head_outputs["gene_interaction"] = predictions.squeeze(-1)
        # Gather VECTOR head predictions to the measured-feature subset so pred and
        # decoded target align 1:1 (per_gene: [B, N] -> [B, n_measured_genes]).
        head_outputs = self._gather_predictions(head_outputs)
        targets, masks = self._extract_targets_and_masks(batch, head_outputs, bsz)
        total, per_head = self.loss(
            head_outputs, targets, masks, graph_reg_loss=reps["graph_reg_loss"]
        )
        self.log(f"{stage}/loss", total, batch_size=bsz, sync_dist=True)
        for name, val in per_head.items():
            self.log(f"{stage}/{name}/loss", val, batch_size=bsz, sync_dist=True)
        # Pearson metric on supervised rows (skips heads with <2 supervised rows).
        # For per-feature-normalized vector heads (WS10b) this correlates in the
        # NORMALIZED space: model prediction and target are both z-scored, so every
        # CalMorph feature contributes equally. Inverting per-feature to raw scale would
        # re-couple the 8-orders-of-magnitude spread and let a couple of huge features
        # dominate the correlation, so normalized-space Pearson is the honest metric here.
        for name, pred in head_outputs.items():
            if name not in targets:
                continue
            m = masks[name]
            if int(m.sum().item()) < 2:
                continue
            pear = _pearson(pred[m].detach(), targets[name][m].detach())
            self.log(f"{stage}/{name}/pearson", pear, batch_size=bsz, sync_dist=True)
        return cast(torch.Tensor, total)

    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Masked multitask training step."""
        return self._step(batch, "train")

    def validation_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Masked multitask validation step."""
        return self._step(batch, "val")

    def _print_epoch(self, stage: str) -> None:
        metrics = self.trainer.callback_metrics
        parts = [
            f"{k}={float(v):.5f}"
            for k, v in metrics.items()
            if k.startswith(stage) and hasattr(v, "item")
        ]
        if parts:
            print(f"[{stage} epoch {self.current_epoch}] " + "  ".join(sorted(parts)))

    def on_train_epoch_end(self) -> None:
        """Print aggregated train metrics for the epoch (smoke-run visibility)."""
        self._print_epoch("train")

    def on_validation_epoch_end(self) -> None:
        """Print aggregated val metrics for the epoch (smoke-run visibility)."""
        self._print_epoch("val")

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
    # "phenotype_values" -> phenotype_values_batch: the per-COO-value batch-row map
    # the target/mask decode relies on (phenotype_sample_indices are NOT batch rows).
    follow_batch = ["perturbation_indices", "phenotype_values"]
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
    active_heads = list(cfg.multitask.active_heads)

    # Resolve each head's COO->target alignment against the REAL build (WS10a):
    # real phenotype-type strings are fitness / calmorph / expression_log2_ratio.
    node_ids = list(dataset.cell_graph["gene"].node_ids)
    head_align = build_head_alignments(
        dataset=dataset,
        active_heads=active_heads,
        head_phenotypes={k: list(v) for k, v in head_phenotypes.items()},
        node_ids=node_ids,
    )
    print("head_align:")
    for h, a in head_align.items():
        printable = {
            k: (int(v.numel()) if hasattr(v, "numel") else v) for k, v in a.items()
        }
        print(f"  {h}: {printable}")

    # ---- WS10b: per-FEATURE target standardization stats (TRAIN split only) ----
    heads_to_normalize = list(cfg.multitask.get("normalize_vector_targets", []))
    target_stats: dict[str, dict[str, Any]] = {}
    if heads_to_normalize:
        train_indices = list(data_module.train_dataset.indices)
        target_stats = compute_per_feature_target_stats(
            dataset=dataset,
            train_indices=train_indices,
            active_heads=active_heads,
            head_phenotypes={k: list(v) for k, v in head_phenotypes.items()},
            head_align=head_align,
            heads_to_normalize=heads_to_normalize,
            eps=float(cfg.multitask.get("target_norm_eps", 1e-8)),
            degenerate_robust_cv=float(cfg.multitask.get("degenerate_robust_cv", 0.01)),
        )
        is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
        for head, st in target_stats.items():
            std = st["std"]
            print(
                f"[WS10b] per-feature norm '{head}': {len(st['keys'])} features, "
                f"n_train={st['n_train']}, std in "
                f"[{float(std.min()):.4g}, {float(std.max()):.4g}]; "
                f"{len(st['degenerate_features'])} near-constant FLAGGED (kept, "
                f"epsilon-floored): {st['degenerate_features']}"
            )
            if is_rank0:
                out = osp.join(
                    experiment_root,
                    "019-simb-multimodal/results",
                    f"calmorph_train_target_norm_{head}.json",
                )
                os.makedirs(osp.dirname(out), exist_ok=True)
                with open(out, "w") as f:
                    json.dump(
                        {
                            "head": head,
                            "n_train": st["n_train"],
                            "eps": st["eps"],
                            "degenerate_robust_cv_threshold": float(
                                cfg.multitask.get("degenerate_robust_cv", 0.01)
                            ),
                            "degenerate_features": st["degenerate_features"],
                            "keys": st["keys"],
                            "mean": st["mean"].tolist(),
                            "std": st["std"].tolist(),
                            "robust_cv": st["robust_cv"],
                        },
                        f,
                        indent=2,
                    )
                print(f"[WS10b] wrote {out}")

    task = MultitaskCGTTask(
        model=model,
        cell_graph=dataset.cell_graph,
        active_heads=active_heads,
        head_weights={k: float(v) for k, v in head_weights.items()},
        head_phenotypes={k: list(v) for k, v in head_phenotypes.items()},
        head_align=head_align,
        loss_fn=str(cfg.multitask.loss_fn),
        optimizer_config=_as_dict(cfg.regression_task.optimizer),
        lr_scheduler_config=(
            _as_dict(cfg.regression_task.lr_scheduler)
            if cfg.regression_task.get("lr_scheduler") is not None
            else None
        ),
        clip_grad_norm=cfg.regression_task.clip_grad_norm,
        clip_grad_norm_max_norm=cfg.regression_task.clip_grad_norm_max_norm,
        target_stats=target_stats,
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
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 10),
        overfit_batches=cfg.trainer.get("overfit_batches", 0),
        limit_train_batches=cfg.trainer.get("limit_train_batches", 1.0),
        limit_val_batches=cfg.trainer.get("limit_val_batches", 1.0),
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
