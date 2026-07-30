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

DECODER STUDY (``multitask.decoder`` x ``multitask.dist``). The decoder factorizes into two
orthogonal axes, both swept by ``optuna_joint_sweep.py``:

* STRUCTURAL (``multitask.decoder``, the ``global``/morphology head only) -- ``s1_pool``
  (``GlobalHead``: pool to one vector, fan out to all F features) vs ``s3_xattn``
  (``CrossAttnHead``: one learned query per feature, cross-attending the full token set).
  S1's shared pool dilutes a few-gene perturbation across ~6000 gene tokens before readout;
  S3 restores per-output support. Expression is per-token by construction (S0), so it has no
  decoder lever.
* DISTRIBUTIONAL (``multitask.dist``, every vector head) -- ``point`` (MSE),
  ``crps`` (Gaussian CRPS, a proper scoring rule) or ``quantile`` (pinball over K=19
  quantiles). This only widens the head's final projection to ``param_dim`` params per
  feature; ``output_dim`` remains the feature count F.

Metrics are namespaced by PHENOTYPE (``val/morphology/...``, ``val/expression/...``), and
the primary metric ``pearson_per_feature`` is always computed from ``DistHead.point()`` --
so Optuna ranks every loss on the same point-estimate correlation.

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
import subprocess
import uuid
from datetime import timedelta
from typing import Any, cast

import hydra
import lightning as L
import numpy as np
import torch
import torch.distributed as dist
import wandb
from dotenv import load_dotenv
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, Timer
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import HeteroData

import torchcell

# The repo whose SOURCE actually ran -- derived from the IMPORTED package, not from this
# file's path. Those differ exactly when PYTHONPATH is wrong, which is the failure this
# hash exists to catch (a worktree script silently importing the primary checkout's
# torchcell would otherwise log the worktree's commit while running main's code).
PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(torchcell.__file__)))

from torchcell.losses.distributional import (
    DEFAULT_ENERGY_RANK,
    DEFAULT_ENERGY_SAMPLES,
    coverage,
    dist_param_dim,
    make_dist_head,
    pit_ks,
)
from torchcell.models.equivariant_cell_graph_transformer import (
    CellGraphTransformer,
    MaskedMultitaskLoss,
)
from torchcell.timestamp import timestamp

log = logging.getLogger(__name__)
load_dotenv()
WANDB_MODE = os.getenv("WANDB_MODE")

# Metrics are namespaced by PHENOTYPE, not by the internal head name, so a dashboard/Optuna
# objective reads `val/morphology/pearson_per_feature` rather than `val/global/...`. The head
# name is an implementation detail (which structural module runs); the phenotype is the
# scientific object and is stable across decoder variants (s1_pool vs s3_xattn both predict
# morphology). Unknown heads fall back to their own name.
HEAD_TO_PHENOTYPE: dict[str, str] = {
    "global": "morphology",
    "per_gene": "expression",
    "gene_interaction": "fitness",
    "per_metabolite": "metabolite",
}


def phenotype_name(head: str) -> str:
    """Map an internal head name to its phenotype metric namespace."""
    return HEAD_TO_PHENOTYPE.get(head, head)


def _as_dict(node: Any) -> dict[str, Any]:
    """Resolve an OmegaConf node to a plain ``dict[str, Any]`` (typed for mypy).

    Accepts a plain dict because every caller reaches this through
    ``cfg.<section>.get("<key>", {})``, and OmegaConf's ``.get`` returns the PYTHON
    default verbatim when the key is absent -- it does not wrap it in a config node.
    ``OmegaConf.to_container({})`` then raises "Input cfg is not an OmegaConf config
    object (dict)", so any config omitting an optional key crashed. That is how
    ``head_phenotype_keys`` (optional, defined only by the metabolism configs) took down
    every config that does not set it.
    """
    if node is None:
        return {}
    if isinstance(node, dict):
        return node
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

    Two decoder-study knobs are injected into every vector head spec:

    * ``multitask.decoder`` -- the STRUCTURAL form of the ``global`` (morphology) head,
      ``s1_pool`` (pool then fan out) or ``s3_xattn`` (one learned query per feature).
      Expression is per-token by construction (S0 ``PerGeneHead``), so the decoder lever
      applies to ``global`` only.
    * ``multitask.dist`` -- the DISTRIBUTIONAL form (``point`` / ``crps`` / ``quantile``),
      applied to EVERY vector head. It only widens the head's final projection
      (``param_dim`` params per feature); ``output_dim`` stays the feature count.
    """
    active = list(cfg.multitask.active_heads)
    head_specs = OmegaConf.to_container(cfg.multitask.heads, resolve=True)
    assert isinstance(head_specs, dict)
    decoder = str(cfg.multitask.get("decoder", "s1_pool"))
    dist = str(cfg.multitask.get("dist", "point"))
    param_dim = dist_param_dim(dist)
    heads_config: dict[str, Any] = {}
    for head in active:
        if head == "gene_interaction":
            continue
        if head not in head_specs:
            raise ValueError(
                f"active head '{head}' has no spec under multitask.heads "
                f"(available: {sorted(head_specs)})"
            )
        spec = dict(head_specs[head] or {})
        spec["param_dim"] = param_dim
        if head == "global":
            spec["decoder"] = decoder
        if head == "per_gene":
            # FREE OUTPUT-GENE EMBEDDING width (0 = the original S0 head). This is the
            # capacity-only decoder arm: it unties output-gene identity from h_k without
            # touching what information reaches the decoder.
            spec["free_gene_dim"] = int(cfg.multitask.get("free_gene_dim", 0))
            # CONCAT arm: feed the head [h_pert ; h_i ; c] instead of h_pert alone, so it
            # can learn arbitrary (h_i, c_b) interactions rather than only functions of
            # their sum. Equivariant (shared MLP per token) and graph-free.
            spec["concat_context"] = bool(cfg.multitask.get("concat_context", False))
            # Rank-R multiplicative interaction between h_i and c, appended to the head
            # input. The parameter-lean alternative to concat_context.
            spec["bilinear_rank"] = int(cfg.multitask.get("bilinear_rank", 0))
            spec["pert_set_context"] = bool(cfg.multitask.get("pert_set_context", False))
            spec["film_on_pert_set"] = bool(cfg.multitask.get("film_on_pert_set", False))
            spec["response_basis_rank"] = int(
                cfg.multitask.get("response_basis_rank", 0)
            )
        heads_config[head] = spec
    return heads_config or None


def _pearson(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pearson correlation between two flattened tensors (grad-free metric).

    Retained for the SCALAR case (gene_interaction) and as the 1-feature reduction of
    :func:`per_feature_pearson`. NOT used as the vector-head metric anymore -- flattening
    a multi-feature vector correlates across features of different scales, which is a
    scale artifact rather than an honest per-phenotype correlation (Part B).
    """
    p = pred.reshape(-1).float()
    t = target.reshape(-1).float()
    p = p - p.mean()
    t = t - t.mean()
    denom = (p.norm() * t.norm()).clamp_min(1e-8)
    return cast(torch.Tensor, (p * t).sum() / denom)


def per_feature_pearson(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean over per-FEATURE Pearson correlations for a vector head (Part B).

    ``pred`` / ``target`` are ``[N, F]`` (N supervised genotypes, F features:
    morphology -> 278 CalMorph features; expression -> 6127 measured genes). We
    correlate EACH feature's column across genotypes, then average the F correlations.
    This is the honest vector metric: a flatten-then-correlate (the old ``_pearson``)
    is dominated by whichever features have the largest raw scale, so it reports a
    feature-scale artifact rather than "how well is each phenotype predicted".

    Features whose prediction OR target column is (near-)constant over the batch have an
    undefined correlation (0/0); they are DROPPED from the average rather than counted as
    zero, so a constant CalMorph feature does not deflate the reported r. A ``[N]`` /
    ``[N, 1]`` scalar input reduces to a single-feature correlation (== ``_pearson``).
    """
    if pred.ndim == 1:
        pred = pred.unsqueeze(1)
    if target.ndim == 1:
        target = target.unsqueeze(1)
    p = pred.float()
    t = target.float()
    pc = p - p.mean(dim=0, keepdim=True)
    tc = t - t.mean(dim=0, keepdim=True)
    num = (pc * tc).sum(dim=0)
    denom = pc.norm(dim=0) * tc.norm(dim=0)
    valid = denom > 1e-8
    if not bool(valid.any()):
        return torch.zeros((), device=pred.device)
    r = num[valid] / denom[valid]
    return cast(torch.Tensor, r.mean())


def _rank(x: torch.Tensor) -> torch.Tensor:
    """Column-wise average ranks of ``[N, F]`` (ties share their mean rank).

    Tie averaging is not optional here: the beta-carotene colony score is an ordinal with
    ~11 distinct values over thousands of strains, so an arbitrary tie-break would invent
    an ordering the data does not contain. ``scipy.stats.rankdata`` does this vectorized
    over the strain axis; the metric already runs at epoch end on CPU-cached tensors.
    """
    from scipy.stats import rankdata

    arr = rankdata(x.detach().cpu().numpy(), axis=0)
    return torch.as_tensor(arr, dtype=torch.float32)


def per_feature_spearman(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean over per-FEATURE Spearman rank correlations for a head.

    Pearson is the wrong primary metric for an ORDINAL target: the beta-carotene colony
    score is a subjective -5..+5 rank, so "did the model order the strains correctly" is
    the question, and the spacing between adjacent scores carries no quantitative meaning.
    Computed as :func:`per_feature_pearson` on average ranks, so it reduces to the usual
    Spearman rho for a single feature and averages per-feature rho for a vector head.
    Rank transformation is monotone, so this is also invariant to the target
    standardization -- unlike Pearson under Yeo-Johnson.
    """
    if pred.ndim == 1:
        pred = pred.unsqueeze(1)
    if target.ndim == 1:
        target = target.unsqueeze(1)
    return per_feature_pearson(_rank(pred.float()), _rank(target.float()))


def per_strain_pearson(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean over per-STRAIN (per-row) Pearson correlations for a vector head.

    ``pred`` / ``target`` are ``[N, F]`` (N supervised genotypes/strains, F features:
    expression -> the measured-gene columns). For EACH genotype (row) we correlate its
    predicted vector against its actual vector ACROSS the feature dimension, then average
    the N per-strain correlations. This is the complement of :func:`per_feature_pearson`:

    * per-FEATURE (``per_feature_pearson``) asks "across strains, is each gene's up/down
      predicted?" -- it is destroyed by regression-to-the-per-gene-mean (a model that
      always predicts each gene's train mean has zero variance across strains -> r=0).
    * per-STRAIN (this function) asks "within one strain, is the shape of its expression
      profile predicted?" -- it stays HIGH even under mean-collapse, because a strain's
      profile is dominated by the shared per-gene mean structure. The GAP between the two
      is the diagnostic: high per-strain + ~0 per-gene == mean-collapse / no real signal.

    Rows whose prediction OR target is (near-)constant across features have an undefined
    correlation (0/0) and are DROPPED rather than counted as zero. A single-feature input
    (F==1, e.g. a scalar head) has no within-row spread and yields an empty average (0).
    """
    if pred.ndim == 1:
        pred = pred.unsqueeze(1)
    if target.ndim == 1:
        target = target.unsqueeze(1)
    p = pred.float()
    t = target.float()
    pc = p - p.mean(dim=1, keepdim=True)
    tc = t - t.mean(dim=1, keepdim=True)
    num = (pc * tc).sum(dim=1)
    denom = pc.norm(dim=1) * tc.norm(dim=1)
    valid = denom > 1e-8
    if not bool(valid.any()):
        return torch.zeros((), device=pred.device)
    r = num[valid] / denom[valid]
    return cast(torch.Tensor, r.mean())


def _yeo_johnson_forward(x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Per-feature Yeo-Johnson power transform (torch, matches sklearn).

    ``x`` is ``[B, F]``; ``lam`` is ``[F]`` (broadcast over rows). Yeo-Johnson is the
    zero/negative-safe generalization of Box-Cox -- CalMorph features include zeros and
    negatives, so strict Box-Cox is undefined; this realizes Ohya SI's "Box-Cox then
    standardize" with domain safety (Part A). Definition per feature with parameter L:
    x>=0: ((x+1)^L - 1)/L (L!=0) or log(x+1) (L==0);
    x<0 : -((-x+1)^(2-L) - 1)/(2-L) (L!=2) or -log(-x+1) (L==2).
    """
    near0 = lam.abs() < 1e-6
    lam2 = 2.0 - lam
    near2 = lam2.abs() < 1e-6
    xp = torch.clamp(x, min=0.0)
    pos_ne = (torch.pow(xp + 1.0, lam) - 1.0) / torch.where(near0, torch.ones_like(lam), lam)
    pos_e = torch.log1p(xp)
    pos_val = torch.where(near0, pos_e, pos_ne)
    xn = torch.clamp(x, max=0.0)
    neg_ne = -(torch.pow(-xn + 1.0, lam2) - 1.0) / torch.where(
        near2, torch.ones_like(lam2), lam2
    )
    neg_e = -torch.log1p(-xn)
    neg_val = torch.where(near2, neg_e, neg_ne)
    return torch.where(x >= 0, pos_val, neg_val)


def _yeo_johnson_inverse(y: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Invert :func:`_yeo_johnson_forward` back to raw units (metric/inference only)."""
    near0 = lam.abs() < 1e-6
    lam2 = 2.0 - lam
    near2 = lam2.abs() < 1e-6
    yp = torch.clamp(y, min=0.0)
    base_pos = torch.clamp(yp * lam + 1.0, min=1e-8)
    pos_ne = torch.pow(base_pos, 1.0 / torch.where(near0, torch.ones_like(lam), lam)) - 1.0
    pos_e = torch.expm1(yp)
    pos_val = torch.where(near0, pos_e, pos_ne)
    yn = torch.clamp(y, max=0.0)
    base_neg = torch.clamp(-lam2 * yn + 1.0, min=1e-8)
    neg_ne = 1.0 - torch.pow(base_neg, 1.0 / torch.where(near2, torch.ones_like(lam2), lam2))
    neg_e = 1.0 - torch.expm1(-yn)
    neg_val = torch.where(near2, neg_e, neg_ne)
    return torch.where(y >= 0, pos_val, neg_val)


def _vector_phenotype_keys(
    dataset: Any, name: str, expected_keys: list[str] | None = None
) -> list[str] | None:
    """Return the sorted dict keys of a vector phenotype ``name`` in this build.

    The ``Perturbation`` processor flattens a dict-valued phenotype to a
    key-sorted vector and DROPS the keys, so the per-value feature identity
    (which gene an expression value belongs to; which CalMorph parameter; which
    amino acid) is not recoverable from a built HeteroData sample. We recover it
    once from the raw reconstructed experiment records (same key-sort the
    processor uses), so the decode can align each head to its target.

    DEFECT FIX (was ``scan=5000``): the old version scanned only the FIRST 5,000
    LMDB rows in index order, so a phenotype whose records all sort past that cap
    (e.g. every metabolome row in a union whose first ~9.2k rows are pigment
    records) was reported ABSENT -- the head then went silently unsupervised with
    ``feat_dim: None`` and zero gradient. We now iterate ``phenotype_label_index[
    name]`` -- the exact row indices carrying that label -- with NO cap, so the
    scan is both complete and cheaper than the old prefix scan.

    ``expected_keys`` pins WHICH vector phenotype is wanted when several heads
    share one ``label_name``. Betaxanthin (Cachera) and the 19-AA metabolome
    (Mulleder) are both ``MetabolitePhenotype`` and therefore both label
    ``metabolite_level``; only the key set distinguishes them. When
    ``expected_keys`` is given we return the first record whose sorted key set
    equals it exactly, and ``None`` if no such record exists (a hard, visible
    "that phenotype is not in this build" answer rather than the wrong one).
    """
    candidates = dataset.phenotype_label_index.get(name)
    if not candidates:
        return None
    if dataset.env is None:
        dataset._init_lmdb_read()
    want = sorted(expected_keys) if expected_keys is not None else None
    for idx in candidates:
        raw = dataset._read_from_lmdb(idx)
        if raw is None:
            continue
        for item in dataset._deserialize_json(raw):
            value = item["experiment"]["phenotype"].get(name)
            if not isinstance(value, dict):
                continue
            keys = sorted(value.keys())
            if want is None or keys == want:
                return keys
    return None


def build_head_alignments(
    dataset: Any,
    active_heads: list[str],
    head_phenotypes: dict[str, list[str]],
    node_ids: list[str],
    drop_features: dict[str, list[str]] | None = None,
    scalar_heads: list[str] | None = None,
    head_phenotype_keys: dict[str, list[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Precompute per-head COO->target alignment from the real build.

    For each active head we resolve its phenotype name to the actual COO layout:

    * SCALAR heads -- declared EXPLICITLY by ``scalar_heads`` (see below). One value
      per genotype: fitness (``gene_interaction``), the beta-carotene ordinal
      ``visual_score``, the 1-key betaxanthin ``metabolite_level``.
    * ``per_gene`` -- expression, a per-gene VECTOR. The processor emits one value
      per MEASURED gene in key-sorted order; the ``per_gene`` head emits one value
      per graph NODE. We build ``col_idx`` (node positions of the measured genes
      that exist in the cell graph) so the prediction is gathered to the measured
      subset, and ``keep_mask`` (over the raw key-sorted vector) dropping measured
      genes absent from the gene set -- so gathered prediction and target align 1:1.
    * every other head -- a fixed-length feature VECTOR (CalMorph 281, Mulleder 19);
      identity alignment, ``feat_dim`` = vector length.

    DEFECT FIX -- ``is_scalar`` used to be hardcoded ``True`` for the single head
    name ``gene_interaction`` and ``False`` for everything else. A scalar target
    routed into a vector head then hit ``target[b] = head_vals`` with a 1-element
    ``head_vals`` and a ``[F]`` row, which torch BROADCASTS: the one scalar is
    silently copied across all F columns and trains as if every feature had been
    measured at that value. ``scalar_heads`` now declares the scalar path per head,
    and ``raw_dim`` (below) lets the decode REJECT any row whose value count does
    not match the head, instead of broadcasting it.

    ``raw_dim`` is the number of COO values the head's phenotype contributes per
    genotype BEFORE ``keep_mask`` drops any (1 for a scalar head, ``len(keys)``
    for a vector head). It is what the decode matches against, and it is also how
    two heads sharing one ``label_name`` are told apart -- betaxanthin
    (``metabolite_level``, 1 key) vs the Mulleder metabolome (``metabolite_level``,
    19 keys). ``head_phenotype_keys`` pins each such head's exact key set so the
    resolution is by IDENTITY, not merely by count.
    """
    nid_to_pos = {n: i for i, n in enumerate(node_ids)}
    drop_features = drop_features or {}
    scalar_set = set(scalar_heads if scalar_heads is not None else ["gene_interaction"])
    head_phenotype_keys = head_phenotype_keys or {}
    align: dict[str, dict[str, Any]] = {}
    for head in active_heads:
        pinned = head_phenotype_keys.get(head)
        keys: list[str] | None = None
        for name in head_phenotypes.get(head, []):
            keys = _vector_phenotype_keys(dataset, name, expected_keys=pinned)
            if keys is not None:
                break
        if head in scalar_set:
            # A scalar head reads ONE value per genotype. Two shapes reach here: a
            # plain float phenotype (fitness, visual_score) -> no dict keys at all;
            # and a 1-key dict phenotype (betaxanthin's metabolite_level) -> keys
            # resolved above. Either way raw_dim is 1, and a pinned key set of
            # length != 1 is a config error, not something to paper over.
            if pinned is not None and len(pinned) != 1:
                raise ValueError(
                    f"scalar head '{head}' pinned to {len(pinned)} keys {pinned}; "
                    "a scalar head must name exactly one (or zero) phenotype keys."
                )
            align[head] = {
                "is_scalar": True,
                "keep_mask": None,
                "col_idx": None,
                "feat_dim": 1,
                "raw_dim": 1,
                "keys": keys,
                "dropped_features": [],
            }
            continue
        if keys is None:
            # No such vector phenotype present in this build -> head unsupervised.
            align[head] = {
                "is_scalar": False,
                "keep_mask": None,
                "col_idx": None,
                "feat_dim": None,
                "raw_dim": None,
                "keys": None,
                "dropped_features": [],
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
                "raw_dim": len(keys),
                "keys": keys,
                "dropped_features": [],
            }
        else:
            # global / per_metabolite: identity-length vector. Part A -- optionally DROP
            # degenerate features (e.g. CalMorph A113_A1B/A113_C/C123_C) from the target
            # AND the head output_dim: build a keep_mask over the key-sorted feature vector
            # so the decoded target is restricted to the kept features (the head's
            # output_dim MUST equal the kept count).
            drop_set = set(drop_features.get(head, []))
            if drop_set:
                keep = torch.tensor(
                    [k not in drop_set for k in keys], dtype=torch.bool
                )
                align[head] = {
                    "is_scalar": False,
                    "keep_mask": keep,
                    "col_idx": None,
                    "feat_dim": int(keep.sum().item()),
                    "raw_dim": len(keys),
                    "keys": keys,
                    "dropped_features": [k for k in keys if k in drop_set],
                }
            else:
                align[head] = {
                    "is_scalar": False,
                    "keep_mask": None,
                    "col_idx": None,
                    "feat_dim": len(keys),
                    "raw_dim": len(keys),
                    "keys": keys,
                    "dropped_features": [],
                }

    # Two heads may legitimately share a `label_name` (betaxanthin and the Mulleder
    # metabolome are both MetabolitePhenotype -> `metabolite_level`). The COO drops the
    # dict keys, so at decode time the ONLY thing separating their value groups within a
    # genotype is the group SIZE. Distinct sizes are therefore a hard requirement, not a
    # nicety: equal sizes would make the assignment arbitrary and silently wrong.
    by_label: dict[str, list[tuple[str, int]]] = {}
    for head, a in align.items():
        if a.get("raw_dim") is None:
            continue
        for name in head_phenotypes.get(head, []):
            by_label.setdefault(name, []).append((head, int(a["raw_dim"])))
    for name, entries in by_label.items():
        dims = [d for _, d in entries]
        if len(set(dims)) != len(dims):
            raise ValueError(
                f"active heads {[h for h, _ in entries]} all read phenotype "
                f"'{name}' with value counts {dims}; the COO drops dict keys, so heads "
                "sharing a label_name must have DISTINCT value counts to be separable."
            )
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
    vector_norm_method: str = "yeo_johnson",
    head_norm_method: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-FEATURE target normalization stats, computed on the TRAIN split ONLY.

    WS10b + Part A. The morphology target (``calmorph``) is a multi-hundred-D vector whose
    features span ~8 orders of magnitude (cell-size counts ~1e4 vs 0--1 ratios), so a single
    pooled mean/std leaves large-scale features dominating an un-normalized MSE (loss O(1e6)).
    We normalize EACH feature independently, with two selectable methods:

    * ``vector_norm_method="yeo_johnson"`` (Part A default): a per-feature Yeo-Johnson power
      transform THEN z-score, fit with ``sklearn.preprocessing.PowerTransformer(
      method="yeo-johnson", standardize=True)`` on the TRAIN targets. This realizes Ohya SI's
      published "Box-Cox then standardize", using Yeo-Johnson (the zero/negative-safe
      generalization) because CalMorph features contain zeros/negatives. The fitted params
      -- per-feature lambda + the transformed-space mean/std -- are stored and inverted at
      report/inference time for raw-unit reporting.
    * ``vector_norm_method="zscore"`` (legacy WS10b): plain per-feature z-score (mean/std),
      no power transform. Kept for ablation.

    Stats are accumulated over ``train_indices`` alone (never val/test) so there is no
    leakage. For each vector head in ``heads_to_normalize`` we resolve its key-sorted feature
    vocabulary (restricted by the head's ``keep_mask`` so it matches the gathered
    prediction/target length -- i.e. AFTER Part A drops), read every train genotype's vector,
    and compute the stats. Near-constant features (robust CV = IQR/|median| below
    ``degenerate_robust_cv``) are FLAGGED (not dropped here); the standardizer/epsilon floor
    keeps them finite.

    DEFECT FIX -- SCALAR heads used to be skipped outright (``if align["is_scalar"]:
    continue``), so a scalar target could not be standardized at all. That is wrong for the
    new production heads, whose raw units are mutually incomparable and none of which is
    O(1): betaxanthin is a population-centred CRI-SPA fluorescence (can be negative) and
    beta-carotene is an ordinal -5..+5. Scalar heads are now handled as the F=1 case --
    single-column stats, same z-score/Yeo-Johnson machinery, same inversion for raw-unit
    metric reporting. A scalar head reads either a plain float phenotype (``visual_score``)
    or a 1-key dict (``metabolite_level``); both are collected here.
    """
    assert vector_norm_method in ("yeo_johnson", "zscore"), (
        f"unsupported vector_norm_method {vector_norm_method!r}"
    )
    head_norm_method = head_norm_method or {}
    stats: dict[str, dict[str, Any]] = {}
    for head in active_heads:
        if head not in heads_to_normalize:
            continue
        # Per-head method: `standardize_per_feature_target` heads (e.g. per_gene) force
        # plain z-score; everything else uses the shared `vector_norm_method` (Yeo-Johnson
        # for morphology). Resolved here so one build can mix z-scored + power-transformed
        # heads without a second stats pass.
        method_for_head = head_norm_method.get(head, vector_norm_method)
        align = head_align.get(head, {})
        if align.get("feat_dim") is None:
            continue
        is_scalar = bool(align.get("is_scalar", False))
        # `keys` was already resolved (and, for a shared label_name, key-pinned) by
        # build_head_alignments -- reuse it instead of re-scanning the LMDB, which would
        # re-introduce the ambiguity between betaxanthin and the 19-AA metabolome.
        keys = align.get("keys")
        name: str | None = None
        for cand in head_phenotypes.get(head, []):
            if cand in dataset.phenotype_label_index:
                name = cand
                break
        if name is None:
            continue
        if is_scalar:
            kept_keys = keys if keys else [head]
            keep_list = [True] * len(kept_keys)
        else:
            if keys is None:
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
                if value is None:
                    continue
                if isinstance(value, dict):
                    if keys is None:
                        continue
                    if not set(keys) <= set(value):
                        # A different phenotype sharing this label_name (e.g. the
                        # 19-AA metabolome when standardizing betaxanthin, whose keys
                        # are disjoint from this head's).
                        continue
                    vec = [
                        float(value[k])
                        for k, flag in zip(keys, keep_list)
                        if flag and k in value
                    ]
                elif is_scalar:
                    vec = [float(value)]
                else:
                    continue
                if len(vec) == len(kept_keys):
                    collected.append(vec)

        arr = np.asarray(collected, dtype=float)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        raw_mean = np.nanmean(arr, axis=0)
        raw_std = np.nanstd(arr, axis=0)
        median = np.nanmedian(arr, axis=0)
        q25 = np.nanpercentile(arr, 25, axis=0)
        q75 = np.nanpercentile(arr, 75, axis=0)
        iqr = q75 - q25
        with np.errstate(divide="ignore", invalid="ignore"):
            robust_cv = np.where(np.abs(median) > 0, iqr / np.abs(median), 0.0)
        degenerate_idx = [
            i for i, rc in enumerate(robust_cv) if rc < degenerate_robust_cv
        ]

        # Impute any residual NaN with the per-feature median so the fit sees a dense
        # matrix (CalMorph is dense per build; this only guards rare missing values).
        col_median = np.where(np.isfinite(median), median, 0.0)
        arr_dense = np.where(np.isfinite(arr), arr, col_median[None, :])

        if method_for_head == "yeo_johnson":
            from sklearn.preprocessing import PowerTransformer

            pt = PowerTransformer(method="yeo-johnson", standardize=True)
            pt.fit(arr_dense)
            lambdas = np.asarray(pt.lambdas_, dtype=float)
            # PowerTransformer standardizes on the TRANSFORMED values via an internal
            # StandardScaler: mean/std here are in Yeo-Johnson space, not raw space.
            t_mean = np.asarray(pt._scaler.mean_, dtype=float)
            t_std = np.asarray(pt._scaler.scale_, dtype=float)
            stats[head] = {
                "method": "yeo_johnson",
                "keys": kept_keys,
                "lambdas": torch.tensor(lambdas, dtype=torch.float32),
                "mean": torch.tensor(t_mean, dtype=torch.float32),
                "std": torch.tensor(t_std, dtype=torch.float32),
                "raw_mean": raw_mean.tolist(),
                "raw_std": raw_std.tolist(),
                "eps": float(eps),
                "n_train": int(arr.shape[0]),
                "degenerate_features": [kept_keys[i] for i in degenerate_idx],
                "robust_cv": robust_cv.tolist(),
            }
        else:
            stats[head] = {
                "method": "zscore",
                "keys": kept_keys,
                "lambdas": None,
                "mean": torch.tensor(raw_mean, dtype=torch.float32),
                "std": torch.tensor(raw_std, dtype=torch.float32),
                "raw_mean": raw_mean.tolist(),
                "raw_std": raw_std.tolist(),
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
        dist: str = "point",
        energy_rank: int = DEFAULT_ENERGY_RANK,
        energy_samples: int = DEFAULT_ENERGY_SAMPLES,
    ) -> None:
        """Store the model, cell_graph, masked loss, and optim/sched config."""
        super().__init__()
        self.model = model
        self.cell_graph = cell_graph.clone()
        self.active_heads = active_heads
        self.head_phenotypes = head_phenotypes
        self.head_align = head_align
        # Distributional readout (point / crps / quantile) for every VECTOR head. The scalar
        # gene_interaction head keeps the plain point loss, so `dist` never changes the
        # fitness arm. `.point()` feeds the metric, so RANKING IS LOSS-AGNOSTIC: a CRPS run
        # and an MSE run are compared on the same per-feature Pearson of a point estimate.
        self.dist = dist
        # `num_features` is REQUIRED by dist=energy: its head owns a global low-rank factor
        # V of shape [F, k] whose Gram VV^T is the predicted gene-gene covariance, so the
        # head must know F. Sourced from head_align[h]["feat_dim"], the same per-head
        # feature count the COO->target alignment already resolved. Ignored by the
        # marginal modes (point/crps/quantile/laplace_crps/nll).
        # `energy_rank` is the JOINT ABLATION knob and only `energy` reads it: k=0 makes V
        # None, so the head is diagonal (independent genes) while the LOSS stays the energy
        # score. Holding the loss fixed and varying only k is what isolates "does modelling
        # the gene-gene covariance pay?" from "is the energy score a better objective?" --
        # comparing energy(k=32) against crps would confound the two.
        self.energy_rank = int(energy_rank)
        # `m`, the TRAINING predictive-sample count. Surfaced as a constructor arg (not left
        # to the library default) so the value that actually ran is recorded in the W&B
        # config for every run rather than being implicit.
        self.energy_samples = int(energy_samples)
        self.dist_heads = {
            h: make_dist_head(
                dist,
                num_features=int(head_align[h]["feat_dim"]),
                rank=self.energy_rank,
                num_samples=self.energy_samples,
            )
            for h in active_heads
            if h != "gene_interaction"
        }
        self.loss = MaskedMultitaskLoss(
            head_weights=head_weights,
            loss_fn=loss_fn,
            dist_heads=cast(dict[str, torch.nn.Module], self.dist_heads),
        )
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_norm_max_norm = clip_grad_norm_max_norm
        # WS10b + Part A per-feature target normalization (TRAIN-split stats). Registered
        # as buffers so params move with .to(device) and are checkpointed for inversion.
        # yeo_johnson: normalize(x) = zscore(YJ(x, lambda)); zscore: normalize(x) = zscore(x).
        self.norm_heads: list[str] = list((target_stats or {}).keys())
        self.norm_eps: dict[str, float] = {}
        self.norm_method: dict[str, str] = {}
        for head, st in (target_stats or {}).items():
            safe = head.replace("/", "_")
            self.register_buffer(f"_norm_mean_{safe}", st["mean"], persistent=True)
            self.register_buffer(f"_norm_std_{safe}", st["std"], persistent=True)
            method = st.get("method", "zscore")
            self.norm_method[head] = method
            if method == "yeo_johnson":
                self.register_buffer(
                    f"_norm_lambda_{safe}", st["lambdas"], persistent=True
                )
            self.norm_eps[head] = float(st["eps"])
        # Part B: per-FEATURE Pearson is an EPOCH-level metric in ORIGINAL (inverse-
        # transformed) units, so per-step supervised (pred, target) pairs are cached here
        # and reduced at epoch end. Keyed by stage -> head -> list of [n, feat] CPU tensors.
        self._metric_cache: dict[str, dict[str, dict[str, list[torch.Tensor]]]] = {}
        # Part C: CALIBRATION. PIT values are cached the same way (stage -> head -> list of
        # [n, feat] CPU tensors) and reduced at epoch end into coverage + KS. Kept in its own
        # cache because it is EVAL-ONLY: PIT costs a full CDF evaluation (and, for `energy`,
        # 256 predictive samples) and answers nothing about the training trajectory.
        self._calib_cache: dict[str, dict[str, list[torch.Tensor]]] = {}
        self.save_hyperparameters(
            ignore=["model", "cell_graph", "head_align", "target_stats"]
        )

    def _norm_mean_std(self, head: str) -> tuple[torch.Tensor, torch.Tensor]:
        safe = head.replace("/", "_")
        return (
            cast(torch.Tensor, getattr(self, f"_norm_mean_{safe}")),
            cast(torch.Tensor, getattr(self, f"_norm_std_{safe}")),
        )

    def _norm_lambda(self, head: str) -> torch.Tensor:
        safe = head.replace("/", "_")
        return cast(torch.Tensor, getattr(self, f"_norm_lambda_{safe}"))

    def _normalize_target(self, head: str, values: torch.Tensor) -> torch.Tensor:
        """Normalize a decoded [B, feat] target for a normalized head (train space).

        yeo_johnson: z-score of the Yeo-Johnson transform; zscore: plain per-feature
        z-score. Both use the TRAIN-split stats stored as buffers.
        """
        mean, std = self._norm_mean_std(head)
        if self.norm_method.get(head) == "yeo_johnson":
            values = _yeo_johnson_forward(values, self._norm_lambda(head))
        return (values - mean) / (std + self.norm_eps[head])

    def denormalize(self, head: str, values: torch.Tensor) -> torch.Tensor:
        """Invert normalization back to raw units (metric reporting / inference)."""
        mean, std = self._norm_mean_std(head)
        raw = values * (std + self.norm_eps[head]) + mean
        if self.norm_method.get(head) == "yeo_johnson":
            raw = _yeo_johnson_inverse(raw, self._norm_lambda(head))
        return raw

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
        is in ``head_phenotypes[head]``, then split those by
        ``phenotype_sample_indices`` (which experiment WITHIN the genotype produced
        the value) and keep the sample group(s) whose SIZE equals the head's
        ``raw_dim``. If any survive, the mask is True and the target row is those
        values (mean over groups for a scalar head; the key-sorted vector for a
        vector head, restricted by ``keep_mask`` to the measured features the
        gathered prediction covers). Absent modalities keep mask False so
        ``MaskedMultitaskLoss`` skips them.

        TWO DEFECTS ARE FIXED HERE, both of which produced a plausible-looking
        number rather than an error:

        * BROADCAST. The old code assigned ``target[b] = head_vals`` with no shape
          check. A 1-element ``head_vals`` assigned into an ``[F]`` row is a legal
          torch broadcast, so a scalar observation was silently replicated across
          every feature of a vector head and trained on as if F features had been
          measured. Every assignment is now size-checked and a mismatch RAISES.
        * LABEL COLLISION. Two heads can share a ``label_name``: betaxanthin and the
          19-AA metabolome are both ``MetabolitePhenotype`` -> ``metabolite_level``.
          Selecting purely by name concatenated 1 + 19 values into one 20-value blob.
          The COO drops the dict keys, but it does keep
          ``phenotype_sample_indices``, so the two experiments' values remain
          separable by group size -- which ``build_head_alignments`` has already
          guaranteed to be unique per label_name.
        """
        device = head_outputs[next(iter(head_outputs))].device
        gene = batch["gene"]
        values = getattr(gene, "phenotype_values", None)
        type_idx = getattr(gene, "phenotype_type_indices", None)
        val_batch = getattr(gene, "phenotype_values_batch", None)
        samp_idx = getattr(gene, "phenotype_sample_indices", None)
        pheno_types = gene["phenotype_types"] if "phenotype_types" in gene else None
        targets: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}
        if (
            values is None
            or type_idx is None
            or val_batch is None
            or samp_idx is None
            or pheno_types is None
        ):
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
            raw_dim = align.get("raw_dim")
            if raw_dim is None:
                # Head is unsupervised in this build (no such phenotype present).
                continue
            raw_dim = int(raw_dim)
            row_mask = torch.zeros(bsz, dtype=torch.bool, device=device)
            # Targets are always POINT-shaped [B, feat]: a distributional head emits
            # [B, feat, param_dim] params, but the observation it is scored against is one
            # value per feature. Size the target buffer from `.point()`, NOT from the raw
            # head output, or the decoded [feat] row cannot be assigned into it.
            dist_head = self.dist_heads.get(head)
            target = torch.zeros_like(
                dist_head.point(pred) if dist_head is not None else pred
            )
            expected = int(target.shape[1]) if target.ndim > 1 else 1
            for b in range(bsz):
                sel_b = val_batch == b
                if not bool(sel_b.any()):
                    continue
                gtypes = per_graph_types[b]
                tb = type_idx[sel_b].tolist()
                vb = values[sel_b]
                sb = samp_idx[sel_b]
                name_sel = torch.tensor(
                    [gtypes[t] in names for t in tb], dtype=torch.bool
                )
                if not bool(name_sel.any()):
                    continue
                cand_vals = vb[name_sel]
                cand_samp = sb[name_sel]
                # Keep only the per-experiment value groups of the head's own width.
                groups = [
                    cand_vals[cand_samp == s]
                    for s in cand_samp.unique(sorted=True).tolist()
                ]
                groups = [g for g in groups if int(g.numel()) == raw_dim]
                if not groups:
                    continue
                if is_scalar:
                    # Several replicate records of a scalar phenotype (e.g. two fitness
                    # measurements of one genotype) average, as before.
                    head_vals = torch.stack([g.reshape(()) for g in groups]).mean()
                    target[b] = head_vals.float().to(device)
                else:
                    if len(groups) > 1:
                        raise ValueError(
                            f"head '{head}' matched {len(groups)} value groups of width "
                            f"{raw_dim} in batch row {b}; a vector head must resolve to "
                            "exactly one measurement per genotype."
                        )
                    head_vals = groups[0]
                    if keep is not None:
                        head_vals = head_vals[keep]
                    if int(head_vals.numel()) != expected:
                        raise ValueError(
                            f"head '{head}' decoded {int(head_vals.numel())} target "
                            f"values but the head emits {expected}. Assigning these "
                            "would BROADCAST rather than align; fix the head's "
                            "output_dim / drop_features / head_phenotype_keys."
                        )
                    target[b] = head_vals.to(device)
                row_mask[b] = True
            # WS10b + Part A: per-feature normalization (TRAIN-split stats) so a multi-scale
            # vector target (CalMorph 278-D) yields an O(1) loss -- Yeo-Johnson+z-score
            # (default) or plain z-score. Masked rows are zeros -> transform to some finite
            # value but are dropped by MaskedMultitaskLoss's row mask.
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
            self.log(
                f"{stage}/{phenotype_name(name)}/loss", val, batch_size=bsz, sync_dist=True
            )
        # GRAPH-REGULARIZATION TELEMETRY. The prior was previously folded into `total` and
        # never surfaced, so "is the graph prior actually being enforced?" was
        # unanswerable from the logs -- which is how a lambda that had been silently
        # SQUARED (applied both per-graph and globally from the same config key, giving an
        # effective 1e-6) went unnoticed. Log the term itself and, more usefully, its RATIO
        # to the data loss: that ratio is the quantity to keep near a chosen centre, and
        # unlike the raw term it is comparable across graph counts and phenotypes.
        reg = reps["graph_reg_loss"]
        if reg is not None:
            data_loss = total.detach() - reg.detach()
            self.log(f"{stage}/graph_reg/loss", reg, batch_size=bsz, sync_dist=True)
            self.log(
                f"{stage}/graph_reg/ratio_to_data",
                reg.detach() / data_loss.clamp_min(1e-8),
                batch_size=bsz,
                sync_dist=True,
            )
        # Part B: cache supervised (pred, target) rows for the EPOCH-level Pearson metrics.
        # The head's params are first reduced to a POINT estimate via `DistHead.point()`
        # (identity / mu / median quantile) so the metric -- and therefore the Optuna
        # ranking -- is identical in form no matter which loss trained the run.
        #
        # TWO spaces are cached because the two metrics want different ones:
        #   * raw  -> pearson_per_feature: per-feature correlation across strains in ORIGINAL
        #     units, so each CalMorph feature / measured gene is weighted equally and the
        #     number is comparable to the abstract's r. (Per-feature correlation is invariant
        #     to a per-feature affine map, but Yeo-Johnson is nonlinear, so the space matters.)
        #   * norm -> pearson_per_instance: within-strain correlation ACROSS features, which
        #     is only meaningful on comparably-scaled features -- on raw multi-scale CalMorph
        #     values it would be dominated by the largest-magnitude features. Computing it on
        #     the z-scored/normalized features is what makes it a real shape diagnostic.
        for name, pred in head_outputs.items():
            if name not in targets:
                continue
            m = masks[name]
            if int(m.sum().item()) < 1:
                continue
            dist_head = self.dist_heads.get(name)
            point = dist_head.point(pred) if dist_head is not None else pred
            p_norm = point[m].detach()
            t_norm = targets[name][m].detach()
            if name in self.norm_heads:
                p_raw = self.denormalize(name, p_norm)
                t_raw = self.denormalize(name, t_norm)
            else:
                # Un-normalized heads (gene_interaction fitness, raw expression log2-ratios)
                # are already in raw units; the two spaces coincide.
                p_raw, t_raw = p_norm, t_norm
            cache = self._metric_cache.setdefault(stage, {}).setdefault(
                name, {"pred": [], "target": [], "pred_norm": [], "target_norm": []}
            )
            cache["pred"].append(p_raw.float().cpu())
            cache["target"].append(t_raw.float().cpu())
            cache["pred_norm"].append(p_norm.float().cpu())
            cache["target_norm"].append(t_norm.float().cpu())
            # Part C: cache PIT for the calibration metrics. Eval stages only (see
            # `_calib_cache`), and `point` has no predictive CDF so it has no PIT at all --
            # a `point` arm simply logs no calib/* keys, which is the honest answer.
            if stage != "train" and dist_head is not None and dist_head.mode != "point":
                with torch.no_grad():
                    self._calib_cache.setdefault(stage, {}).setdefault(
                        name, []
                    ).append(dist_head.pit(pred, targets[name], m).float().cpu())
        return cast(torch.Tensor, total)

    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Masked multitask training step."""
        return self._step(batch, "train")

    def validation_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Masked multitask validation step."""
        return self._step(batch, "val")

    def test_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Masked multitask test step (same masked path + epoch metric)."""
        return self._step(batch, "test")

    def _reduce_epoch_pearson(self, stage: str) -> None:
        """Compute + log the epoch Pearson metrics from the cache (Part B).

        Logged under the PHENOTYPE namespace (``val/morphology/...``), not the head name, so
        the metric key is stable across decoder variants:

        * ``pearson_per_feature`` (PRIMARY, raw units) -- across STRAINS, per feature/gene.
          The honest metric and the Optuna objective; collapses to ~0 under
          regression-to-the-per-feature-mean.
        * ``pearson_per_instance`` (DIAGNOSTIC, normalized/z-scored features) -- within each
          STRAIN, across features. Stays high under mean-collapse, so a large gap between the
          two is the mean-collapse signature.
        """
        stage_cache = self._metric_cache.get(stage, {})
        per_feature_values: list[torch.Tensor] = []
        for name, cache in stage_cache.items():
            if not cache["pred"]:
                continue
            pred = torch.cat(cache["pred"], dim=0)
            target = torch.cat(cache["target"], dim=0)
            if pred.shape[0] < 2:
                continue
            pheno = phenotype_name(name)
            # Metrics are computed from the CPU-cached rows and moved to the compute device so
            # any DDP sync_dist all-reduce runs on the NCCL (GPU) backend — a CPU tensor has no
            # NCCL backend. (Single-GPU trials make the sync a no-op, but this stays DDP-safe.)
            pear_feat = per_feature_pearson(pred, target).to(self.device)
            self.log(f"{stage}/{pheno}/pearson_per_feature", pear_feat, sync_dist=True)
            per_feature_values.append(pear_feat)
            # Rank metric, logged for every head. It is the PRIMARY metric for an ordinal
            # target (the beta-carotene colour score) and a useful monotone-invariant
            # companion elsewhere.
            spear_feat = per_feature_spearman(pred, target).to(self.device)
            self.log(f"{stage}/{pheno}/spearman_per_feature", spear_feat, sync_dist=True)
            feat_dim = pred.shape[1] if pred.ndim > 1 else 1
            # DISPERSION RATIO sd(pred)/sd(target) across STRAINS, averaged over features.
            # This is the mean-collapse diagnostic for a SCALAR head. `pearson_per_instance`
            # (the usual signature) correlates ACROSS features within a strain and is therefore
            # undefined at feat_dim == 1 -- it is skipped below, which left the betaxanthin and
            # beta-carotene arms with no collapse signal at all. The ratio has one everywhere:
            # collapse to the per-feature mean drives sd(pred) -> 0 while Pearson is scale-free
            # and cannot see it. ~1 means the head spans the target's range; << 1 means it is
            # hedging toward the mean even when the correlation still looks acceptable.
            sd_ratio = (
                pred.std(dim=0) / target.std(dim=0).clamp_min(1e-8)
            ).mean().to(self.device)
            self.log(f"{stage}/{pheno}/pred_sd_ratio", sd_ratio, sync_dist=True)
            if feat_dim > 1:
                # Per-instance runs on the NORMALIZED features (comparable scales); falls back
                # to the raw cache for heads that are not normalized (the two coincide there).
                pred_n = torch.cat(cache["pred_norm"], dim=0)
                target_n = torch.cat(cache["target_norm"], dim=0)
                pear_inst = per_strain_pearson(pred_n, target_n).to(self.device)
                self.log(
                    f"{stage}/{pheno}/pearson_per_instance", pear_inst, sync_dist=True
                )
        # Single monitorable scalar across ALL arms: the mean per-feature Pearson over the
        # supervised phenotypes this epoch. For a single-head arm (morph / expr) it equals
        # that head's value; for the JOINT arm it averages expression and morphology, which
        # is what makes one uniform EarlyStopping key possible -- Lightning can only monitor
        # one metric, and stopping the joint run on morphology alone would cut it while
        # expression was still improving (or vice versa).
        if per_feature_values:
            mean_pf = torch.stack(per_feature_values).mean()
            self.log(f"{stage}/mean/pearson_per_feature", mean_pf, sync_dist=True)
        self._metric_cache[stage] = {}

    def _reduce_epoch_calibration(self, stage: str) -> None:
        """Compute + log the epoch CALIBRATION metrics from the PIT cache (Part C).

        Pearson says whether the predicted MEAN tracks the truth; it is completely blind to
        whether the predicted UNCERTAINTY is honest. Two arms can tie on
        ``pearson_per_feature`` while one reports intervals twice as wide as it should. These
        are the metrics that separate them, and they are the reason for running a
        distributional sweep at all.

        Logged per phenotype, mirroring the Pearson namespace:

        * ``calib/coverage_50`` / ``calib/coverage_80`` (CROSS-MODE COMPARABLE) -- the
          fraction of observations inside the central 50% / 80% predictive interval.
          Calibrated is 0.50 / 0.80; BELOW is overconfident (intervals too narrow), ABOVE is
          underconfident and is the signature of NLL sigma-collapse. Both evaluate the
          predictive CDF at INTERIOR points, where no mode has a structural artifact, so
          these are the keys to compare a ``quantile`` arm against a ``crps`` arm.
        * ``calib/pit_ks`` (WITHIN-MODE ONLY) -- sup-norm distance of the whole PIT
          histogram from Uniform(0,1). Strictly more informative than the two coverages
          (it sees the entire shape, not two interior points) but NOT comparable across
          modes: the ``tau in [0.05, 0.95]`` quantile grid parks ~5% of PIT mass at each
          endpoint even when perfectly calibrated, forcing KS ~ 0.05 for reasons that have
          nothing to do with the model. Use it to track one arm across epochs.

        PIT is computed in the head's OWN output space, which is the normalized space for
        normalized heads. That costs nothing: PIT is invariant under any strictly increasing
        map applied to both prediction and observation, and both the z-score and the
        Yeo-Johnson transforms are strictly increasing.
        """
        for name, chunks in self._calib_cache.get(stage, {}).items():
            if not chunks:
                continue
            pit = torch.cat(chunks, dim=0)
            pheno = phenotype_name(name)
            for alpha in (0.5, 0.8):
                self.log(
                    f"{stage}/{pheno}/calib/coverage_{int(100 * alpha)}",
                    coverage(pit, alpha).to(self.device),
                    sync_dist=True,
                )
            self.log(
                f"{stage}/{pheno}/calib/pit_ks",
                pit_ks(pit).to(self.device),
                sync_dist=True,
            )
        self._calib_cache[stage] = {}

    def _print_epoch(self, stage: str) -> None:
        metrics = self.trainer.callback_metrics
        parts = [
            f"{k}={float(v):.5f}"
            for k, v in metrics.items()
            if k.startswith(stage) and hasattr(v, "item")
        ]
        if parts:
            print(f"[{stage} epoch {self.current_epoch}] " + "  ".join(sorted(parts)))

    def on_train_epoch_start(self) -> None:
        """Reset the train epoch metric cache (Part B)."""
        self._metric_cache["train"] = {}

    def on_validation_epoch_start(self) -> None:
        """Reset the val epoch metric + calibration caches (Parts B, C)."""
        self._metric_cache["val"] = {}
        self._calib_cache["val"] = {}

    def on_test_epoch_start(self) -> None:
        """Reset the test epoch metric + calibration caches (Parts B, C)."""
        self._metric_cache["test"] = {}
        self._calib_cache["test"] = {}

    # Scalar parameters that GATE a mechanism on or off. Every one of these is a switch the
    # optimizer may simply decline to throw, and a mechanism whose gate stayed shut has NOT
    # been tested -- it produced no signal because it was never in the forward pass.
    #
    # THIS EXISTS BECAUSE WE LOST A ROUND TO IT. The null-sink arm ran 4 seeds x ~190 epochs
    # and scored a paired delta of ~0 against its reference. That looked like a clean negative
    # result. It was not: reading `null_bias` out of the last.ckpt afterwards showed it had
    # moved from -4.0 to only -3.92/-3.95 across all four seeds, so the attention sink carried
    # ~1.9% of the mass instead of the ~1.8% it started with. The pair term the arm existed to
    # test was never actually available. Because the gate was not logged, "explored and found
    # nothing" was indistinguishable from "never engaged" until the runs were already dead.
    #
    # Matching by SUFFIX rather than a hardcoded list so a new gate is picked up automatically;
    # a mechanism added later cannot silently go unlogged.
    GATE_PARAM_SUFFIXES = ("null_bias", "gate", "beta_attn", "beta_ffn", "ablate_mask")

    def _log_gates(self) -> None:
        """Log every gate/mask scalar so gate movement is visible DURING a run, not post-mortem.

        Vector-valued gates (e.g. a learned per-dimension ablation mask) are logged as their
        mean and their max absolute deviation from the identity, which is what distinguishes
        "moved off init" from "stayed put".
        """
        for name, param in self.model.named_parameters():
            if not name.endswith(self.GATE_PARAM_SUFFIXES):
                continue
            flat = param.detach().float().flatten()
            if flat.numel() == 1:
                self.log(f"gate/{name}", flat[0], sync_dist=False)
            else:
                self.log(f"gate/{name}/mean", flat.mean(), sync_dist=False)
                # Distance from the identity element (1.0 for a multiplicative mask), i.e.
                # "how far has this actually travelled".
                self.log(
                    f"gate/{name}/max_dev", (flat - 1.0).abs().max(), sync_dist=False
                )

    def on_train_epoch_end(self) -> None:
        """Reduce + log epoch per-feature Pearson, then print aggregated train metrics."""
        self._reduce_epoch_pearson("train")
        self._log_gates()
        self._print_epoch("train")

    def on_validation_epoch_end(self) -> None:
        """Reduce + log epoch Pearson and calibration, then print aggregated val metrics."""
        self._reduce_epoch_pearson("val")
        self._reduce_epoch_calibration("val")
        self._print_epoch("val")

    def on_test_epoch_end(self) -> None:
        """Reduce + log epoch Pearson and calibration, then print aggregated test metrics."""
        self._reduce_epoch_pearson("test")
        self._reduce_epoch_calibration("test")
        self._print_epoch("test")

    def configure_optimizers(self) -> Any:
        """Build the optimizer and optional LR scheduler from config."""
        opt_cfg = dict(self.optimizer_config)
        opt_class = getattr(torch.optim, opt_cfg.pop("type"))
        optimizer = opt_class(self.parameters(), **opt_cfg)
        if self.lr_scheduler_config is None:
            return optimizer
        sched_cfg = dict(self.lr_scheduler_config)
        sched_type = sched_cfg.pop("type", "ReduceLROnPlateau")
        # `type: null` means NO SCHEDULER, not "fall through to the default". The _008
        # config declares the whole lr_scheduler block so hydra struct mode will accept
        # `regression_task.lr_scheduler.type=...` overrides for the warmup arms -- but that
        # makes the config a dict rather than None for EVERY arm, so without this guard the
        # no-warmup baseline would silently acquire a ReduceLROnPlateau it never had, and
        # A0 would stop being comparable to _006/_007.
        if sched_type is None:
            return optimizer
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
        """Log the PRE-clip gradient norm, then clip by norm when configured.

        `clip_grad_norm_max_norm` is 10.0, inherited from the _003 base config -- chosen for
        a different regime (hidden 96, L=2, lr 3e-4, MSE only). _007 sweeps lr up to 2e-3 and
        adds an `nll` arm whose gradients grow as sigma shrinks, so clipping may now be
        BINDING on some trials. That matters for attribution: if clipping saturates, the
        effective step size stops tracking the sampled lr, and the lr axis silently stops
        measuring what it claims to.

        Logged PRE-clip (`self.clip_gradients` mutates the grads in place, so the order here
        is load-bearing) and as a fraction of the threshold, so `train/grad_norm_clip_frac`
        > 1 reads directly as "this step was clipped".
        """
        total = torch.sqrt(
            sum(
                (p.grad.detach() ** 2).sum()
                for p in self.parameters()
                if p.grad is not None
            )
        )
        # `batch_size=1` is REQUIRED, not cosmetic, and is also the correct weighting.
        #   * Required: without it Lightning infers the batch size by ITERATING the batch,
        #     and this batch is a PyG HeteroData whose FeatureStore.__iter__ raises
        #     NotImplementedError -- every step would die. (Every other self.log call in
        #     this module passes batch_size=bsz, which is why none of them hit it; the
        #     batch is not in scope in this hook.)
        #   * Correct: the gradient norm is a property of the STEP, not of a sample, so the
        #     epoch mean should weight each step equally rather than by its row count.
        self.log(
            "train/grad_norm",
            total,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )
        self.log(
            "train/grad_norm_clip_frac",
            total / self.clip_grad_norm_max_norm,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )
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
    dist = str(cfg.multitask.get("dist", "point"))
    print(f"[dry-run] active_heads={list(cfg.multitask.active_heads)}")
    print(
        f"[dry-run] decoder={cfg.multitask.get('decoder', 's1_pool')} dist={dist} "
        f"param_dim={dist_param_dim(dist)}"
    )
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
    # Vector heads route through their DistHead (point/crps/quantile); gene_interaction keeps
    # the plain point loss. Targets are POINT-shaped: a DistHead consumes [B, F] targets even
    # when the head emits [B, F, P] params, so `.point()` gives the target shape.
    # `num_features` is required by dist=energy (its head owns a global [F, k] factor V).
    # run_dry_run has no head_align in scope -- it deliberately never builds the dataset --
    # so F comes from the head's own output shape, which is available because the forward
    # pass above already ran. That is the PRE-gather width for per_gene (N genes, not the
    # measured subset), which is fine here: the dry run only proves the path connects and
    # backprops, it is not a training run.
    dist_heads = {
        h: make_dist_head(
            dist,
            num_features=int(head_outputs[h].shape[1]),
            rank=int(cfg.multitask.get("energy_rank", DEFAULT_ENERGY_RANK)),
            num_samples=int(
                cfg.multitask.get("energy_samples", DEFAULT_ENERGY_SAMPLES)
            ),
        )
        for h in cfg.multitask.active_heads
        if h != "gene_interaction"
    }
    loss_fn = MaskedMultitaskLoss(
        loss_fn=str(cfg.multitask.loss_fn),
        dist_heads=cast(dict[str, torch.nn.Module], dist_heads),
    )
    targets = {}
    for k, v in head_outputs.items():
        dh = dist_heads.get(k)
        point_shape = dh.point(v).shape if dh is not None else v.shape
        targets[k] = torch.randn(point_shape)
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
    for k, v in head_outputs.items():
        dh = dist_heads.get(k)
        if dh is not None:
            print(
                f"[dry-run]   head '{k}': params {tuple(v.shape)} -> point "
                f"{tuple(dh.point(v).shape)} (mode={dh.mode})"
            )
    # Exercise the CALIBRATION path too (Part C), so a dry run proves the configured `dist`
    # can actually produce the calib/* metrics -- not just a loss. On random params these
    # numbers are meaningless; the point is that the PIT -> coverage -> KS chain runs and
    # returns finite scalars for THIS mode.
    for k, v in head_outputs.items():
        dh = dist_heads.get(k)
        if dh is None or dh.mode == "point":
            continue
        with torch.no_grad():
            pit = dh.pit(v.detach(), targets[k])
        print(
            f"[dry-run]   head '{k}' calib: coverage_50={coverage(pit, 0.5).item():.3f} "
            f"coverage_80={coverage(pit, 0.8).item():.3f} "
            f"pit_ks={pit_ks(pit).item():.3f}"
        )

    grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    print(f"[dry-run] summed grad norm (nonzero => backward connected): {grad_norm:.6f}")
    print("[dry-run] OK -- model + heads + masked loss wired correctly.")


class BestMetricTracker(Callback):
    """Track the PEAK (max) of every val metric, plus a SNAPSHOT of all metrics at the peak.

    These runs reach a per-feature-Pearson peak early, then MSE-collapse toward the per-feature
    mean — so ``trainer.callback_metrics`` (the LAST epoch) reports the post-collapse value and
    understates the achievable signal. The Optuna objective should use the peak instead; this
    callback records it so ``run_training`` can return ``{metric}_max`` alongside the last value.

    Two DIFFERENT reductions, because two kinds of metric need different ones:

    * ``{metric}_max`` — the running max, correct for any metric with a DIRECTION (Pearson,
      Spearman: higher is better).
    * ``{metric}_at_peak`` — the value at the epoch where ``anchor`` peaked. Correct for any
      metric with a TARGET rather than a direction. ``calib/coverage_80`` should be 0.8; its
      max over training is the single most over-dispersed epoch, which is the OPPOSITE of the
      best one. And a calibration number is only meaningful paired with the point estimate it
      accompanied, so it must be read at the same epoch the run is ranked on.
    * ``{metric}_smooth3_max`` — the max of a 3-epoch centred rolling mean. A max over a long
      noisy sequence is BIASED UPWARD by the number of draws taken, and run length here is set
      by early stopping, so it varies with the hyperparameters: measured on the 002 metabolism
      studies, r(duration, objective) = +0.75 on mulleder19 (+0.80 within a fixed
      architecture cell). Any axis that changes how long a run trains — `lr` above all, which
      is what this round exists to attribute — therefore gets credit partly through the number
      of draws rather than through the model. Averaging three consecutive epochs first cuts
      that inflation roughly in half. Recorded, NOT ranked: ranking stays on ``_max`` so the
      numbers remain comparable to _006/_007. If the two disagree for a config, its ``_max``
      was a noise spike.

    Also recorded, so the confound above is auditable rather than inferred:
    ``val/n_val_epochs`` (how many draws the max was taken over) and ``val/peak_epoch`` (where
    the anchor peaked). A peak_epoch near the end means the run was still improving when it
    stopped; peaks scattered early with a long tail are the signature of a noise-driven max.
    """

    def __init__(self, anchor: str = "val/mean/pearson_per_feature") -> None:
        """Start with an empty peak table.

        Args:
            anchor: The metric whose peak defines "the best epoch" for the ``_at_peak``
                snapshot. Defaults to the same uniform key EarlyStopping monitors, so the
                snapshot, the stopping rule and the Optuna ranking all agree on which epoch
                the run is being judged by.
        """
        self.anchor = anchor
        self.best_max: dict[str, float] = {}
        self.at_peak: dict[str, float] = {}
        self.peak_epoch: int = 0
        self.n_val_epochs: int = 0
        # Last two observations per metric, so a 3-epoch rolling mean needs no full history.
        self._recent: dict[str, list[float]] = {}
        self.best_smooth3: dict[str, float] = {}

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Record the running max of every finite val metric, and snapshot at the anchor peak."""
        finite = {}
        for k, v in trainer.callback_metrics.items():
            if v is None:
                continue
            fv = float(v)
            if fv != fv:  # NaN
                continue
            finite[k] = fv
            if k not in self.best_max or fv > self.best_max[k]:
                self.best_max[k] = fv
            window = self._recent.setdefault(k, [])
            window.append(fv)
            if len(window) > 3:
                window.pop(0)
            # Only score a FULL window: a 1- or 2-epoch mean at the start of training is a
            # different (noisier) estimator than the 3-epoch one, and mixing them would
            # reintroduce the very bias this metric removes.
            if len(window) == 3:
                smooth = sum(window) / 3.0
                if k not in self.best_smooth3 or smooth > self.best_smooth3[k]:
                    self.best_smooth3[k] = smooth
        anchor_value = finite.get(self.anchor)
        # `>=` rather than `>` so the very first epoch populates the snapshot; without it a
        # run that never improves after epoch 0 would report no `_at_peak` values at all.
        if anchor_value is not None and anchor_value >= self.best_max.get(
            self.anchor, float("-inf")
        ):
            self.at_peak = dict(finite)
            self.peak_epoch = int(trainer.current_epoch)
        # Counted from the callback rather than read off the trainer so sanity-check and
        # fast_dev_run passes are counted the same way the metrics were.
        self.n_val_epochs += 1


def _genotypes_with_all_head_targets(
    dataset: Any, wanted: dict[str, set[str]]
) -> set[int]:
    """Record indices carrying at least one target key for EVERY head in ``wanted``.

    Reads the LMDB once and inspects each experiment's phenotype value dict, because the
    dataset's cached indices key on phenotype LABEL and cannot see inside a
    ``metabolite_level`` payload. Cached next to the dataset so the scan is paid once per
    build rather than once per trial -- a sweep runs this hundreds of times.
    """
    cache_path = osp.join(dataset.processed_dir, "head_target_presence_index.json")
    if osp.exists(cache_path):
        with open(cache_path) as fh:
            presence: dict[str, list[int]] = json.load(fh)
    else:
        print("Computing head-target presence index...", flush=True)
        import lmdb

        found: dict[str, set[int]] = {}
        env = lmdb.open(
            osp.join(dataset.processed_dir, "lmdb"), readonly=True, lock=False, subdir=True
        )
        with env.begin() as txn:
            for raw_key, raw_value in txn.cursor():
                idx = int(raw_key.decode())
                for record in json.loads(raw_value.decode()):
                    phenotype = record["experiment"]["phenotype"]
                    label = phenotype.get("label_name")
                    value = phenotype.get(label)
                    if isinstance(value, dict):
                        for key in value:
                            found.setdefault(key, set()).add(idx)
                    elif label is not None:
                        found.setdefault(label, set()).add(idx)
        env.close()
        presence = {k: sorted(v) for k, v in found.items()}
        with open(cache_path, "w") as fh:
            json.dump(presence, fh)

    allowed: set[int] | None = None
    for head, keys in wanted.items():
        have: set[int] = set()
        for key in keys:
            have |= set(presence.get(key, []))
        assert have, (
            f"require_head_targets: head {head!r} matched no records for any of its keys "
            f"{sorted(keys)[:6]} (dataset carries: {sorted(presence)[:12]})"
        )
        allowed = have if allowed is None else (allowed & have)
    assert allowed, "require_head_targets left no genotypes -- the heads do not co-occur"
    return allowed


def _pinned_test_indices(
    cfg: DictConfig, dataset: Any, experiment_root: str
) -> set[int] | None:
    """Resolve an EXTERNAL gene-level test split into dataset record indices.

    Reproduces somebody else's split inside ours so their numbers and ours are computed on
    the same genes. Without this the comparison is not merely imprecise, it is invalid: our
    CellDataModule splits 80/10/10 at random, so about 511 of Merzbacher 2025's 639
    betaxanthin test genes land in OUR TRAIN set, and any score we quoted on them would be
    partly a training score.

    Config: ``data_module.pinned_test_split_file`` -- a path (absolute, or relative to
    ``EXPERIMENT_ROOT``) to a JSON carrying ``split.test`` as a list of SYSTEMATIC gene names.
    Absent or null means no pin, and the split is the ordinary seeded random one.

    Genes are mapped through ``dataset.is_any_deletion_gene_index`` -- the DELETION-only
    index, NOT ``is_any_perturbed_gene_index``. That distinction is load-bearing and was
    found the hard way: the pigment cassettes are emitted as perturbations on every strain,
    and three of their members are native ORFs that also exist in the deletion collection
    (``ARO4``/YBR249C and ``ARO7``/YPR060C in Cachera's betaxanthin cassette, ``BTS1``/YPL069C
    in Ozaydin's). Two of those are in Merzbacher's test list, so under the perturbed-gene
    index this pin resolved 639 genes to 4,885 of 4,930 records and left 28 training
    examples -- a silent, catastrophic split that still ran.

    A requested gene absent from the built LMDB is COUNTED AND REPORTED, never silently
    dropped -- the Cachera build is currently stale w.r.t. the shared name resolver (issue
    #195) and loses a handful, and a comparison that quietly shrank its own test set would
    be reporting on a different gene set than it claims.
    """
    pin_file = cfg.data_module.get("pinned_test_split_file")
    if not pin_file:
        return None
    path = pin_file if osp.isabs(pin_file) else osp.join(experiment_root, pin_file)
    with open(path) as fh:
        payload = json.load(fh)
    genes = payload["split"]["test"]
    gene_index = dataset.is_any_deletion_gene_index
    indices: set[int] = set()
    found = 0
    for gene in genes:
        hits = gene_index.get(gene)
        if hits:
            found += 1
            indices.update(hits)
    frac = len(indices) / len(dataset)
    print(
        f"[pinned-split] {path}\n"
        f"[pinned-split] {found}/{len(genes)} requested genes present in the build "
        f"-> {len(indices)} records pinned to TEST ({frac:.1%} of {len(dataset)}); "
        f"{len(genes) - found} absent, see issue #195",
        flush=True,
    )
    # A pinned gene list is a TEST SET, so it must leave a training set behind. Wiring this
    # through the perturbed-gene index instead of the deletion index put 99.3% of the data in
    # test and left 28 training records -- and trained anyway, reporting plausible-looking
    # numbers. Assert rather than trust: the failure is silent and the result is unusable.
    assert frac < 0.5, (
        f"pinned test set is {frac:.1%} of the dataset ({len(indices)}/{len(dataset)}). "
        f"{len(genes)} genes should select a small minority of records; this means the gene "
        "index is resolving cassette membership rather than deletions."
    )
    return indices


def _dump_test_predictions(
    task: Any, data_module: Any, dataset: Any, cfg: DictConfig, out_path: str
) -> dict[str, int]:
    """Write per-GENE test predictions for every active head to ``out_path`` (JSON).

    WHY THIS EXISTS. ``data_module.pinned_test_split_file`` reproduces somebody else's split
    inside ours, but until now nothing ever SCORED that split -- ``run_training`` called
    ``trainer.fit`` and stopped, so the pinned Merzbacher genes were held out and then never
    looked at. The head-to-head their deposit makes possible (bin our predictions with
    train-fitted thresholds; report MCC, high-producer recall, Spearman) needs the raw
    per-gene numbers, not a scalar metric, because their comparison is a CLASSIFICATION on
    bins we fit afterwards. So the predictions themselves are the artifact.

    ALIGNMENT. ``CellDataModule.test_dataloader`` is built with ``shuffle=False``, so batches
    arrive in the order of ``test_dataset.indices`` -- that is what lets a running cursor map
    row -> dataset record -> gene. The gene name comes from ``is_any_deletion_gene_index``
    (the DELETION index, NOT the perturbed-gene index: the pigment cassettes are emitted as
    perturbations on every strain and three of their members are native ORFs, so the
    perturbed index would name nearly every record).

    Predictions are reduced to a POINT estimate via ``DistHead.point()`` and DENORMALIZED, so
    the dumped numbers are in the target's raw units and comparable to the released screen
    values regardless of which scoring rule trained the run. Rows the head did not supervise
    (mask False) are written with ``target: null`` rather than dropped, so a reader can tell
    "not measured" from "measured and missed".
    """
    idx_to_genes: dict[int, list[str]] = {}
    for gene, rows in dataset.is_any_deletion_gene_index.items():
        for row in rows:
            idx_to_genes.setdefault(int(row), []).append(str(gene))
    order = [int(i) for i in data_module.test_dataset.indices]

    device = next(task.parameters()).device
    task.eval()
    records: dict[str, list[dict[str, Any]]] = {h: [] for h in task.active_heads}
    cursor = 0
    with torch.no_grad():
        for batch in data_module.test_dataloader():
            batch = batch.to(device)
            _, reps = task(batch)
            bsz = task._batch_size(batch)
            head_outputs = task._gather_predictions(dict(reps["head_outputs"]))
            targets, masks = task._extract_targets_and_masks(batch, head_outputs, bsz)
            for head in task.active_heads:
                if head not in head_outputs or head not in targets:
                    continue
                dist_head = task.dist_heads.get(head)
                pred = head_outputs[head]
                point = dist_head.point(pred) if dist_head is not None else pred
                tgt = targets[head]
                if head in task.norm_heads:
                    point = task.denormalize(head, point)
                    tgt = task.denormalize(head, tgt)
                point = point.detach().float().cpu()
                tgt = tgt.detach().float().cpu()
                mask = masks[head].detach().cpu()
                for row in range(bsz):
                    rec = order[cursor + row]
                    records[head].append(
                        {
                            "record_index": rec,
                            "genes": idx_to_genes.get(rec, []),
                            "pred": point[row].tolist(),
                            "target": tgt[row].tolist() if bool(mask[row]) else None,
                        }
                    )
            cursor += bsz
    # A short count is a silent truncation of the comparison set, so it is an error.
    assert cursor == len(order), f"test dump covered {cursor}/{len(order)} records"

    os.makedirs(osp.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(
            {
                # The run's IDENTITY, carried in the file rather than reconstructed from the
                # filename: the dump name is `<hostname>-<jobid>_<cfg-sha256>.json`, which is
                # unique but says nothing about WHICH grid cell produced it. Without these
                # three keys, ~96 dumps per job cannot be grouped by setting or averaged over
                # seeds -- i.e. the replication the grid exists for would be unusable.
                "seed": int(cfg.get("seed", 42)),
                "wandb_tags": list(cfg.wandb.get("tags", [])),
                "dist": str(cfg.multitask.get("dist", "point")),
                "n_test_records": len(order),
                "active_heads": list(task.active_heads),
                "head_keys": {
                    h: list(task.head_align.get(h, {}).get("keys", []) or [])
                    for h in task.active_heads
                },
                "predictions": records,
            },
            fh,
        )
    counts = {h: len(v) for h, v in records.items()}
    print(f"[test-dump] wrote {out_path} ({counts})", flush=True)
    return counts


def run_training(cfg: DictConfig) -> dict[str, float]:
    """Full training path: genome/graph/embeddings/dataset/datamodule + Trainer."""
    # Deferred heavy imports so --help / dry-run never pay for them.
    from torch_geometric.transforms import Compose

    from torchcell.data import (
        Aggregator,
        DeletionKeyedGenotypeAggregator,
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

    # Config-driven seed: the overnight grid replicates each config across several seeds so
    # real expression signal is separable from split/init noise on the small (~1.1k train)
    # expression set. seed_everything makes model init + the CellDataModule split reproducible.
    seed = int(cfg.get("seed", 42))
    L.seed_everything(seed, workers=True)

    os.environ["WANDB__SERVICE_WAIT"] = "600"
    if not (dist.is_available() and dist.is_initialized()):
        os.environ["TORCH_DISTRIBUTED_DEFAULT_TIMEOUT"] = "7200"

    wandb_cfg = _as_dict(cfg)
    # SOURCE VERSION -- the single highest-leverage line in the whole scoring contract.
    # A source edit on 2026-07-28 22:26:23 landed ten minutes before a wave launched, so
    # runs on two different model versions were pooled and reported as seed variance; the
    # phantom 0.0032 "noise floor" was that edit. The config hash alone cannot catch it
    # (the config was identical), and neither can the git hash alone (this worktree runs
    # dirty). Record BOTH the commit and a hash of the working-tree diff, so any two runs
    # can be tested for source identity after the fact.
    wandb_cfg["source"] = {
        "git_hash": subprocess.run(
            ["git", "-C", PROJECT_ROOT, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip(),
        "diff_sha256": hashlib.sha256(
            subprocess.run(
                ["git", "-C", PROJECT_ROOT, "diff", "HEAD"],
                capture_output=True, text=True, check=True,
            ).stdout.encode("utf-8")
        ).hexdigest(),
    }

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    job_id = slurm_job_id or str(uuid.uuid4())
    hostname = socket.gethostname()
    hashed_cfg = hashlib.sha256(
        json.dumps(wandb_cfg, sort_keys=True).encode("utf-8")
    ).hexdigest()
    group = f"{hostname}-{job_id}_{hashed_cfg}"
    experiment_dir = osp.join(data_root, "wandb-experiments", group)
    os.makedirs(experiment_dir, exist_ok=True)

    # Only global rank 0 initializes the wandb run. Under torchrun every rank runs this
    # script top-to-bottom, so an unguarded wandb.init() creates one run per GPU, all
    # writing to the same group dir → write-conflicts + "Logging error" → a rank exits 1
    # and torchrun SIGTERMs the job. WandbLogger below is rank-safe on its own (Lightning
    # @rank_zero_experiment). torchrun sets RANK before the DDP process group exists.
    is_rank_zero = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0
    run = None
    if is_rank_zero:
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
        # Default TRUE, so every existing arm keeps uploading its checkpoints. Turned OFF on
        # the Delta grid runs: those are ~96 runs per job across four jobs, and each one
        # ships two checkpoints, so leaving it on spends hours of a 2-day allocation
        # uploading artifacts nobody reads. The checkpoints still land on disk under
        # $DATA_ROOT/models/checkpoints/<group>/ either way.
        log_model=bool(wandb_cfg["wandb"].get("log_model", True)),
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

    # Genotype key. `genotype` keys on the FULL perturbation set (the default, and what
    # every Fig-3 build uses). `deletion_keyed` keys on the DELETION set alone, treating a
    # constant engineered cassette as reference-strain background -- required for the
    # pigment builds, where the betaxanthin/beta-carotene strains carry 4/3 `gene_addition`
    # perturbations that make their genotypes key-disjoint from every single-KO metabolome
    # genotype (measured co-location exactly ZERO under the full key).
    aggregator_name = str(cfg.cell_dataset.get("aggregator", "genotype"))
    aggregator_cls: type[Aggregator] = {
        "genotype": GenotypeAggregator,
        "deletion_keyed": DeletionKeyedGenotypeAggregator,
    }[aggregator_name]
    print(f"[aggregator] {aggregator_name} -> {aggregator_cls.__name__}")

    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=gene_multigraph,
        incidence_graphs=incidence_graphs,
        node_embeddings=node_embeddings,
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=aggregator_cls,
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

    # "phenotype_values" -> phenotype_values_batch: the per-COO-value batch-row map
    # the target/mask decode relies on (phenotype_sample_indices are NOT batch rows).
    follow_batch = ["perturbation_indices", "phenotype_values"]
    pinned_test = _pinned_test_indices(cfg, dataset, experiment_root)
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
        pinned_test_indices=pinned_test,
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

    # ---- Dataset-name restriction (Kemmeren-only vs Kemmeren+Sameith) ----
    # The fig3_core build fuses Kemmeren + Sameith(Sm/Dm) expression, all carrying the SAME
    # phenotype type `expression_log2_ratio`, so there is no per-head phenotype switch to
    # separate them. We restrict at the ROW level instead: `dataset.dataset_name_index`
    # maps each exact `dataset_name` -> the row indices carrying it, so intersecting each
    # already-split Subset with the allowed names keeps the split assignment but drops
    # non-selected rows. Pure Kemmeren = the exact key `MicroarrayKemmeren2014Dataset`; the
    # mean-merged `MicroarrayKemmeren2014Dataset+SmMicroarraySameith2015Dataset` twin is a
    # SEPARATE key, so it is EXCLUDED from Kemmeren-only (that cross-platform mean-merge is
    # the confound documented in the note) and INCLUDED only in the +Sameith condition.
    restrict_names = list(cfg.cell_dataset.get("restrict_dataset_names", []))
    if restrict_names:
        name_index = dataset.dataset_name_index
        missing = [nm for nm in restrict_names if nm not in name_index]
        if missing:
            raise ValueError(
                f"restrict_dataset_names {missing} not in dataset_name_index "
                f"(available: {sorted(name_index)[:12]}...)"
            )
        allowed = set()
        for nm in restrict_names:
            allowed.update(name_index[nm])
        for split_attr in ("train_dataset", "val_dataset", "test_dataset"):
            sub = getattr(data_module, split_attr)
            before = len(sub.indices)
            sub.indices = [i for i in sub.indices if i in allowed]
            print(
                f"[restrict] {split_attr}: {before} -> {len(sub.indices)} rows "
                f"(names={restrict_names})"
            )

    # ---- require_modalities: keep only genotypes carrying ALL listed phenotype types ----
    # Enables the CONTROLLED auxiliary-task experiment (does expression help morphology, and
    # vice versa): fix the instance set to those with BOTH modalities, then vary only which
    # heads are active. Unlike restrict_dataset_names (UNION over dataset names), this is the
    # INTERSECTION over phenotype-type presence, using the dataset's phenotype_label_index.
    require_modalities = list(cfg.cell_dataset.get("require_modalities", []))
    if require_modalities:
        label_index = dataset.phenotype_label_index
        missing = [m for m in require_modalities if m not in label_index]
        if missing:
            raise ValueError(
                f"require_modalities {missing} not in phenotype_label_index "
                f"(available: {sorted(label_index)[:12]}...)"
            )
        allowed = set(label_index[require_modalities[0]])
        for m in require_modalities[1:]:
            allowed &= set(label_index[m])
        for split_attr in ("train_dataset", "val_dataset", "test_dataset"):
            sub = getattr(data_module, split_attr)
            before = len(sub.indices)
            sub.indices = [i for i in sub.indices if i in allowed]
            print(
                f"[require_modalities] {split_attr}: {before} -> {len(sub.indices)} rows "
                f"(all of {require_modalities})"
            )

    # ---- require_head_targets: the same control, one level finer ----
    # `require_modalities` intersects on PHENOTYPE LABEL, which is enough to separate
    # expression from calmorph but NOT betaxanthin from the 19-AA metabolome: both are
    # `metabolite_level`, and what distinguishes them is which KEYS their value dict carries
    # (`{betaxanthin: ...}` vs `{alanine: ..., ...}`). Asking for
    # `require_modalities: [metabolite_level, metabolite_level]` is a no-op.
    #
    # This resolves each named HEAD through `multitask.head_phenotype_keys` and keeps only
    # genotypes carrying at least one key for EVERY listed head -- so the betaxanthin-only
    # control and the betaxanthin+metabolome joint arm run on an identical instance set, and
    # "the metabolome helps" cannot be a restatement of "the joint arm saw more data".
    # Measured on fig6_pigment_transfer: 4,669 betaxanthin and 4,678 metabolome genotypes
    # share 4,432, so the control costs only 237 rows (5%).
    require_head_targets = list(cfg.cell_dataset.get("require_head_targets", []))
    if require_head_targets:
        head_keys = _as_dict(cfg.multitask.get("head_phenotype_keys", {}))
        wanted = {}
        for head in require_head_targets:
            keys = list(head_keys.get(head, []))
            assert keys, (
                f"require_head_targets lists {head!r} but multitask.head_phenotype_keys has "
                f"no key set for it (have: {sorted(head_keys)})"
            )
            wanted[head] = set(keys)
        allowed = _genotypes_with_all_head_targets(dataset, wanted)
        for split_attr in ("train_dataset", "val_dataset", "test_dataset"):
            sub = getattr(data_module, split_attr)
            before = len(sub.indices)
            sub.indices = [i for i in sub.indices if i in allowed]
            print(
                f"[require_head_targets] {split_attr}: {before} -> {len(sub.indices)} rows "
                f"(all of {require_head_targets})"
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    heads_config = build_heads_config(cfg)
    # Model class. `cell_graph_transformer` is the Fig-3 model; `metabolism` is the
    # CGT-Metabolism fork, which reuses that model's encoder + PERT operator UNCHANGED
    # (parity-tested) and adds the named production/metabolome heads.
    model_class = str(cfg.model.get("model_class", "cell_graph_transformer"))
    if model_class == "metabolism":
        from torchcell.models.cell_graph_transformer_metabolism import (
            CellGraphTransformerMetabolism,
        )

        model_cls: type[CellGraphTransformer] = CellGraphTransformerMetabolism
    elif model_class == "cell_graph_transformer":
        model_cls = CellGraphTransformer
    else:
        raise ValueError(
            f"unknown model.model_class {model_class!r} "
            "(expected 'cell_graph_transformer' or 'metabolism')"
        )
    model = model_cls(
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
        perturbation_propagation_config=_as_dict(
            cfg.model.get("perturbation_propagation", None)
        ),
        post_perturbation_mixing_config=_as_dict(
            cfg.model.get("post_perturbation_mixing", None)
        ),
        attention_mask_config=_as_dict(cfg.model.get("attention_mask", None)),
        cross_gene_config=_as_dict(cfg.model.get("cross_gene", None)),
        observed_label_config=_as_dict(cfg.model.get("observed_labels", None)),
    ).to(device)
    print("Parameter counts:", model.num_parameters)

    head_weights = _as_dict(cfg.multitask.head_weights)
    head_phenotypes = _as_dict(cfg.multitask.head_phenotypes)
    active_heads = list(cfg.multitask.active_heads)

    # ---- Scale metadata -> wandb config + summary (scaling-study axis data) ----
    # Recorded to BOTH wandb.config (queryable as a hyperparameter) and wandb.summary
    # (queryable as a final scalar) so "outcome vs params x dataset size x modality x
    # dataset type" is recoverable across the whole sweep for the model-class-H scaling
    # study. Computed here because the model + the (restricted) splits both now exist.
    total_param_count = int(sum(p.numel() for p in model.parameters()))
    trainable_param_count = int(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
    standardize_heads_cfg = list(cfg.multitask.get("standardize_per_feature_target", []))
    # A head is "standardized" if it is in EITHER normalization list (z-score OR
    # Yeo-Johnson+z-score); both put the target in a comparable per-feature scale.
    normalize_heads_cfg = list(cfg.multitask.get("normalize_vector_targets", []))
    # n genotypes actually SUPERVISED for the active head(s): a split row counts only if it
    # carries an active head's phenotype label (dataset.phenotype_label_index), intersected
    # with that split's (post-restriction) indices. For the expression grid this equals the
    # restricted split sizes; the label intersection keeps it correct under any config.
    supervised_labels: set[str] = set()
    for _h in active_heads:
        supervised_labels.update(head_phenotypes.get(_h, []))
    label_index = dataset.phenotype_label_index
    supervised_idx: set[int] = set()
    for _lbl in supervised_labels:
        supervised_idx.update(label_index.get(_lbl, []))
    n_train_supervised = len(set(data_module.train_dataset.indices) & supervised_idx)
    n_val_supervised = len(set(data_module.val_dataset.indices) & supervised_idx)
    n_test_supervised = len(set(data_module.test_dataset.indices) & supervised_idx)
    # dataset_type / composition string from the row-level restriction.
    if restrict_names:
        _has_kem = any("Kemmeren" in n for n in restrict_names)
        _has_sam = any("Sameith" in n for n in restrict_names)
        if _has_kem and _has_sam:
            dataset_type = "kemmeren+sameith"
        elif _has_kem:
            dataset_type = "kemmeren_only"
        elif _has_sam:
            dataset_type = "sameith_only"
        else:
            dataset_type = "+".join(restrict_names)
    else:
        dataset_type = str(cfg.cell_dataset.get("dataset_tag", "all"))

    # How the graph prior actually entered THIS run: a soft KL on attention weights, a hard
    # structural mask, both, or neither. Derived once here so the three logged keys below
    # cannot disagree with each other.
    _n_declared_reg_heads = len(
        _as_dict(cfg.model.graph_regularization.get("regularized_heads", {}))
    )
    _kl_on = float(cfg.model.graph_regularization.graph_reg_lambda) > 0.0
    # Read through `_as_dict(cfg.model.get("attention_mask", None))`, the SAME idiom line
    # 2170 uses to build the model. Only cgt_expr_008/009/010 declare this block; the
    # metabolism, Delta and smoke configs do not, and for those "absent" genuinely means
    # "this config predates hard masking", not an error to surface.
    _mask_cfg = _as_dict(cfg.model.get("attention_mask", None))
    _mask_on = bool(_mask_cfg.get("enabled", False))
    _n_masked_heads = len(_as_dict(_mask_cfg.get("head_graphs", {}))) if _mask_on else 0
    _graph_prior = {
        (True, True): "kl+mask",
        (True, False): "kl",
        (False, True): "mask",
        (False, False): "none",
    }[(_kl_on, _mask_on)]

    scale_meta: dict[str, Any] = {
        "total_param_count": total_param_count,
        "trainable_param_count": trainable_param_count,
        "n_train_supervised": n_train_supervised,
        "n_val_supervised": n_val_supervised,
        "n_test_supervised": n_test_supervised,
        "dataset_type": dataset_type,
        "dataset_composition": (
            list(restrict_names) if restrict_names else [dataset_type]
        ),
        "active_heads": list(active_heads),
        "active_head": active_heads[0] if active_heads else None,
        # Decoder-study axes: the structural form of the morphology head and the
        # distributional readout/loss. Recorded so "outcome vs decoder x dist" is queryable
        # across the whole sweep without re-deriving it from the Optuna trial params.
        "decoder": str(cfg.multitask.get("decoder", "s1_pool")),
        "dist": str(cfg.multitask.get("dist", "point")),
        "dist_param_dim": dist_param_dim(str(cfg.multitask.get("dist", "point"))),
        # The joint ablation knob. Recorded for EVERY run, not just energy ones, so a W&B
        # filter on `dist == energy` can group by rank without a null column; for the
        # marginal modes it is simply the unused default.
        "energy_rank": int(cfg.multitask.get("energy_rank", DEFAULT_ENERGY_RANK)),
        "energy_samples": int(
            cfg.multitask.get("energy_samples", DEFAULT_ENERGY_SAMPLES)
        ),
        "hidden_channels": int(cfg.model.hidden_channels),
        "num_layers": int(cfg.model.num_transformer_layers),
        "num_heads": int(cfg.model.num_attention_heads),
        "target_standardized": bool(standardize_heads_cfg or normalize_heads_cfg),
        "standardize_heads": standardize_heads_cfg,
        "normalize_heads": normalize_heads_cfg,
        "target_norm": (
            "zscore" if standardize_heads_cfg else "yeo_johnson"
        ) if (standardize_heads_cfg or normalize_heads_cfg) else "raw",
        "graph_reg_lambda": float(cfg.model.graph_regularization.graph_reg_lambda),
        "lr": float(cfg.regression_task.optimizer.lr),
        "dropout": float(cfg.model.dropout),
        "weight_decay": float(cfg.regression_task.optimizer.weight_decay),
        "seed": seed,
        # Hardware + graph-channel breadth, to W&B (not only Optuna user_attrs) so runs
        # can be grouped and filtered on them in the UI. gpu_type matters because a study
        # pooled across IGB partitions mixes A100 (mmli) and RTX6000 (cabbi); n_graphs /
        # n_regularized_heads make the _005 (2-graph) vs _006 (9-graph) contrast queryable.
        "gpu_type": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "n_graphs": len(list(cfg.cell_dataset.graphs)),
        # GRAPH-PRIOR TELEMETRY. `n_regularized_heads` used to report the DECLARED count
        # straight from the config, so a run with graph_reg_lambda=0.0 logged
        # n_regularized_heads=9 while regularizing nothing -- the whole wave-4 series is
        # tagged that way in W&B. It now reports what actually ran: lambda=0 means zero
        # regularized heads, whatever the inherited `regularized_heads` block still declares.
        # The declared count is kept alongside it so the 9-declared/7-effective era stays
        # diagnosable, and `n_masked_heads` gives the hard-masking count that REPLACED the
        # KL, which had no telemetry at all.
        "n_declared_reg_heads": _n_declared_reg_heads,
        "n_regularized_heads": (
            _n_declared_reg_heads
            if float(cfg.model.graph_regularization.graph_reg_lambda) > 0.0
            else 0
        ),
        "n_masked_heads": _n_masked_heads,
        # One categorical so "how did the graph prior enter this run" is a single W&B
        # filter instead of a two-key inference the reader has to do by hand.
        "graph_prior": _graph_prior,
    }
    print("[scale-meta] " + json.dumps(scale_meta))
    if run is not None:
        run.config.update(scale_meta, allow_val_change=True)  # type: ignore[no-untyped-call]
        for _k, _v in scale_meta.items():
            run.summary[_k] = _v

    # Resolve each head's COO->target alignment against the REAL build (WS10a):
    # real phenotype-type strings are fitness / calmorph / expression_log2_ratio.
    # Part A: `drop_features` removes degenerate CalMorph features (e.g. A113_A1B/A113_C/
    # C123_C -> 278) from the `global` target AND the head output_dim.
    drop_features = {
        k: list(v) for k, v in _as_dict(cfg.multitask.get("drop_features", {})).items()
    }
    node_ids = list(dataset.cell_graph["gene"].node_ids)
    # `scalar_heads` declares the scalar path explicitly (was: hardcoded to
    # gene_interaction only, which let a scalar target broadcast across a vector head).
    # `head_phenotype_keys` pins the exact dict keys for heads that SHARE a label_name --
    # betaxanthin and the Mulleder metabolome are both `metabolite_level`.
    scalar_heads = list(cfg.multitask.get("scalar_heads", ["gene_interaction"]))
    head_phenotype_keys = {
        k: list(v)
        for k, v in _as_dict(cfg.multitask.get("head_phenotype_keys", {})).items()
    }
    head_align = build_head_alignments(
        dataset=dataset,
        active_heads=active_heads,
        head_phenotypes={k: list(v) for k, v in head_phenotypes.items()},
        node_ids=node_ids,
        drop_features=drop_features,
        scalar_heads=scalar_heads,
        head_phenotype_keys=head_phenotype_keys,
    )
    print("head_align:")
    for h, a in head_align.items():
        printable = {
            k: (
                int(v.numel())
                if hasattr(v, "numel")
                else (f"[{len(v)} keys]" if isinstance(v, list) and len(v) > 8 else v)
            )
            for k, v in a.items()
        }
        print(f"  {h}: {printable}")

    # Part A sanity: a head's model output_dim MUST equal its (post-drop) target feature
    # count, else MaskedMultitaskLoss gets a [B, out] vs [B, feat] shape mismatch. Scalar
    # heads are checked too (output_dim must be 1) -- they were previously skipped, which
    # is exactly how a scalar/vector mismatch reached the silent-broadcast path.
    for h, a in head_align.items():
        if a.get("feat_dim") is None:
            continue
        model_head = getattr(model, f"{h}_head", None)
        out_dim = getattr(model_head, "output_dim", None)
        if h == "per_gene":
            continue  # per_gene output is [B, N] gathered to feat_dim; out_dim is 1
        if out_dim is not None and int(out_dim) != int(a["feat_dim"]):
            raise ValueError(
                f"head '{h}' output_dim={out_dim} != target feat_dim={a['feat_dim']} "
                f"(dropped {a.get('dropped_features')}). Set heads.{h}.output_dim="
                f"{a['feat_dim']} in the config."
            )

    # ---- WS10b + Part A: per-FEATURE target normalization stats (TRAIN split only) ----
    # Two selectable levers, both fit on the TRAIN split only (no leakage) and inverted for
    # raw-unit metric reporting:
    #   normalize_vector_targets    -> `vector_norm_method` (default Yeo-Johnson) — morphology.
    #   standardize_per_feature_target -> plain per-feature z-score, FORCED. This is the
    #     anti-mean-collapse lever for expression: z-scoring each gene across train strains
    #     makes the model predict DEVIATIONS from the per-gene mean, so a constant "predict
    #     the mean" output scores ~0 (it no longer wins the raw-scale MSE by default).
    normalize_vector_targets = list(cfg.multitask.get("normalize_vector_targets", []))
    standardize_heads = list(cfg.multitask.get("standardize_per_feature_target", []))
    vector_norm_method = str(cfg.multitask.get("vector_norm_method", "yeo_johnson"))

    # ALWAYS-STANDARDIZE invariant (decoder study): every active VECTOR head must be in one
    # of the two normalization lists. `raw` was dropped as a lever because an un-standardized
    # multi-scale target makes the MSE (and the CRPS/pinball, which are in y-units) dominated
    # by the largest-magnitude features -- the loss then optimizes a handful of cell-size
    # counts while the per-feature Pearson averages all 278 equally. Comparing decoders under
    # that mismatch would confound "structural form" with "which features the loss cared
    # about", so the invariant is enforced here rather than left to config hygiene.
    #
    # OPT-IN (default off) so the pre-existing configs -- which legitimately run an
    # un-normalized head (e.g. raw expression log2-ratios) -- keep working unchanged. The
    # decoder-study base config sets `require_standardized_targets: true`.
    if bool(cfg.multitask.get("require_standardized_targets", False)):
        _normalized = set(normalize_vector_targets) | set(standardize_heads)
        _unnormalized = [
            h
            for h in active_heads
            if h != "gene_interaction" and h not in _normalized
        ]
        if _unnormalized:
            raise ValueError(
                f"active vector head(s) {_unnormalized} are not standardized. The decoder "
                "study requires every vector head in multitask.normalize_vector_targets "
                "(Yeo-Johnson) or multitask.standardize_per_feature_target (z-score). "
                "Set multitask.require_standardized_targets=false to override."
            )
    head_norm_method = {h: "zscore" for h in standardize_heads}
    heads_to_normalize = list(dict.fromkeys(normalize_vector_targets + standardize_heads))
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
            vector_norm_method=vector_norm_method,
            head_norm_method=head_norm_method,
        )
        is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
        for head, st in target_stats.items():
            std = st["std"]
            lam = st.get("lambdas")
            lam_desc = (
                f", lambda in [{float(lam.min()):.3g}, {float(lam.max()):.3g}]"
                if lam is not None
                else ""
            )
            print(
                f"[Part A] '{head}' norm={st['method']}: {len(st['keys'])} features, "
                f"n_train={st['n_train']}, std in "
                f"[{float(std.min()):.4g}, {float(std.max()):.4g}]{lam_desc}; "
                f"dropped {head_align[head].get('dropped_features')}; "
                f"{len(st['degenerate_features'])} near-constant FLAGGED (kept, "
                f"floored): {st['degenerate_features']}"
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
                            "method": st["method"],
                            "n_train": st["n_train"],
                            "eps": st["eps"],
                            "dropped_features": head_align[head].get(
                                "dropped_features", []
                            ),
                            "degenerate_robust_cv_threshold": float(
                                cfg.multitask.get("degenerate_robust_cv", 0.01)
                            ),
                            "degenerate_features": st["degenerate_features"],
                            "keys": st["keys"],
                            "lambdas": (
                                st["lambdas"].tolist()
                                if st.get("lambdas") is not None
                                else None
                            ),
                            "mean": st["mean"].tolist(),
                            "std": st["std"].tolist(),
                            "raw_mean": st["raw_mean"],
                            "raw_std": st["raw_std"],
                            "robust_cv": st["robust_cv"],
                        },
                        f,
                        indent=2,
                    )
                print(f"[Part A] wrote {out}")

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
        dist=str(cfg.multitask.get("dist", "point")),
        energy_rank=int(cfg.multitask.get("energy_rank", DEFAULT_ENERGY_RANK)),
        energy_samples=int(cfg.multitask.get("energy_samples", DEFAULT_ENERGY_SAMPLES)),
    )

    model_base_path = osp.join(data_root, "models/checkpoints")
    os.makedirs(model_base_path, exist_ok=True)
    best_tracker = BestMetricTracker()
    # Selection key for the saved "best" checkpoint. See the ModelCheckpoint comment below.
    _ckpt_cfg = _as_dict(cfg.trainer.get("checkpoint", {}))
    ckpt_monitor = str(_ckpt_cfg.get("monitor", "val/loss"))
    ckpt_mode = str(_ckpt_cfg.get("mode", "min"))
    checkpoint_callbacks: list[Callback] = [
        best_tracker,
        # WHAT "BEST" MEANS IS NOW A CONFIG DECISION, and it has to be.
        #
        # This was pinned to val/loss, which silently discards the model we actually want
        # whenever a run peaks on the METRIC and then MSE-collapses toward the per-feature
        # mean -- the documented failure mode of these runs (see BestMetricTracker). Job
        # 1341 is the worked example: betaxanthin val Pearson peaked at 0.323 on epoch 55
        # (val loss 0.885, the only point it dropped below the z-scored target's variance),
        # then decayed to EXACTLY 0.00000 by epoch ~80 while val/loss returned to 0.971.
        # Early stopping fired 31 epochs after the peak and the saved checkpoint was the
        # collapsed model -- i.e. the run found signal and then threw it away.
        #
        # Loss and metric are different statistics here, so selecting on loss is not a
        # proxy for selecting on the metric. Default stays val/loss so existing arms are
        # unchanged; metabolism arms monitor val/mean/pearson_per_feature with mode=max.
        ModelCheckpoint(
            dirpath=osp.join(model_base_path, group),
            save_top_k=1,
            monitor=ckpt_monitor,
            mode=ckpt_mode,
            filename=f"{job_id}-best-{{epoch:02d}}",
        ),
        ModelCheckpoint(
            dirpath=osp.join(model_base_path, group),
            save_last=True,
            filename=f"{job_id}-last",
        ),
    ]

    # EarlyStopping: cut the marathon. The prior 373-epoch/5.4h run overfit long after the
    # metric peaked, so stop when val/loss stops improving. Configurable via
    # trainer.early_stopping (default on, monitor val/loss, patience 20).
    es_cfg = _as_dict(cfg.trainer.get("early_stopping", {"enabled": True}))
    if es_cfg.get("enabled", True):
        es_monitor = str(es_cfg.get("monitor", "val/loss"))
        es_patience = int(es_cfg.get("patience", 20))
        checkpoint_callbacks.append(
            EarlyStopping(
                monitor=es_monitor,
                mode=str(es_cfg.get("mode", "min")),
                patience=es_patience,
                min_delta=float(es_cfg.get("min_delta", 0.0)),
                verbose=True,
            )
        )
        print(f"[early-stopping] monitor={es_monitor} patience={es_patience}")

    # WALL-CLOCK BUDGET. `trainer.max_time_s` stops training GRACEFULLY after a duration:
    # `fit` returns normally, so the metric snapshot, the test pass and the prediction dump
    # all still happen. Without it, an epoch-capped long run is killed mid-fit by slurm and
    # loses every one of those, and its optuna trial is left RUNNING rather than COMPLETE.
    #
    # This is what makes "train for the whole allocation and look at where it saturates" a
    # runnable instruction. The 019 expression measurement is the reason it is needed:
    # smoothed val Pearson peaked ~0.14 at epoch 85-136, FELL to 0.08-0.11 by epoch 200-300,
    # then rose to a project-best 0.198 at epoch 1367 -- so an epoch cap chosen in advance
    # either truncates in the dip or is a guess. A time budget is the honest cap: it spends
    # exactly the compute available and lets the curve decide where the peak is.
    max_time_s = cfg.trainer.get("max_time_s")
    if max_time_s:
        checkpoint_callbacks.append(Timer(duration=timedelta(seconds=float(max_time_s))))
        print(f"[max-time] stopping training after {float(max_time_s) / 3600:.2f} h")

    torch.set_float32_matmul_precision("medium")
    devices = get_num_devices(cfg.trainer.devices)
    print(f"devices: {devices} ({timestamp()})")

    # The masked multitask model activates only a SUBSET of heads per run (e.g. the
    # expression-only baseline uses just per_gene), so the inactive heads' parameters
    # do not contribute to the loss on that run. Vanilla DDP forbids unused parameters,
    # so map plain "ddp" -> the find-unused-parameters variant. Only rank-0 devices>1.
    strategy = cfg.trainer.strategy
    if devices == 1:
        # Single-GPU grid: no DDP process group, plain `python` launch. Force `auto` so a
        # stray `strategy: ddp` in a config never spins up a 1-rank DDP group (extra sync
        # overhead + the find-unused-parameters machinery are pointless with one device).
        strategy = "auto"
    elif strategy == "ddp" and devices > 1:
        strategy = "ddp_find_unused_parameters_true"

    trainer = L.Trainer(
        strategy=strategy,
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
    # Snapshot the final logged metrics BEFORE wandb.finish() so an external driver
    # (e.g. the Optuna sweep) can read the objective (e.g. val/global/pearson_per_gene).
    # main() ignores this return, so the plain training path is byte-for-byte unchanged.
    final_metrics = {
        k: float(v) for k, v in trainer.callback_metrics.items() if v is not None
    }
    # PEAK value of each metric over training (`{metric}_max`) — the Optuna objective uses this,
    # NOT the last epoch, because runs peak then collapse toward the per-feature mean.
    final_metrics.update({f"{k}_max": v for k, v in best_tracker.best_max.items()})
    # `_at_peak`: every val metric AS OF the best epoch. This is the correct reduction for
    # the calibration metrics, which have a target (0.5 / 0.8 / 0) rather than a direction --
    # a max over epochs would report the most over-dispersed epoch as if it were the best.
    final_metrics.update({f"{k}_at_peak": v for k, v in best_tracker.at_peak.items()})
    # `_smooth3_max`: the max of a 3-epoch rolling mean, the de-biased companion to `_max`.
    # Not the ranking metric (that stays `_max`, for comparability with _006/_007) but the
    # check on it -- a config whose `_max` far exceeds its `_smooth3_max` peaked on one lucky
    # epoch rather than on a real plateau.
    final_metrics.update(
        {f"{k}_smooth3_max": v for k, v in best_tracker.best_smooth3.items()}
    )
    # Run length and peak location: the covariates that make the max-over-epochs bias
    # auditable instead of silent. See BestMetricTracker for the measured confound.
    final_metrics["val/n_val_epochs"] = float(best_tracker.n_val_epochs)
    final_metrics["val/peak_epoch"] = float(best_tracker.peak_epoch)

    # ---- HELD-OUT TEST PASS (opt-in) ----
    # Off by default so every existing arm is byte-for-byte unchanged. Turned ON only where
    # the test split MEANS something: `data_module.pinned_test_split_file` reproduces
    # Merzbacher 2025's 639-gene betaxanthin test set inside our split, and a comparison
    # against their Fig 4b has to be computed on those genes, not on val.
    #
    # `ckpt_path="best"` loads the checkpoint selected by `trainer.checkpoint.monitor` --
    # which the metabolism arms point at the METRIC, not val/loss. Testing the LAST model
    # instead would score the mean-collapsed end state these runs decay into (job 1341:
    # peak 0.323 -> exactly 0.000 twenty-five epochs later), i.e. it would report ~0 for a
    # run that worked.
    #
    # Note the metrics are namespaced `test/...` by Lightning and are merged in AFTER the
    # fit snapshot, so no val key can be clobbered by the test run's callback_metrics.
    if bool(cfg.trainer.get("run_test", False)):
        trainer.test(model=task, datamodule=data_module, ckpt_path="best")
        final_metrics.update(
            {
                k: float(v)
                for k, v in trainer.callback_metrics.items()
                if v is not None and k.startswith("test/")
            }
        )
        if bool(cfg.trainer.get("dump_test_predictions", False)):
            _dump_test_predictions(
                task,
                data_module,
                dataset,
                cfg,
                osp.join(data_root, "test-predictions", f"{group}.json"),
            )
    if run is not None:
        wandb.finish()
    return final_metrics


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
