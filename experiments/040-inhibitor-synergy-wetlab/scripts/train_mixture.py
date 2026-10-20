# experiments/040-inhibitor-synergy-wetlab/scripts/train_mixture.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.train_mixture]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/train_mixture
"""Dose- and mixture-aware chemogenomic model, scored against the 2021 wet-lab wells.

Experiment 038's environment-encoder head (``train_factorized.EnvironmentEncoder``,
round 10: a compound token in front of the 6,607 gene tokens, one further transformer
layer, a one-layer cell graph transformer, graph prior weight 0), generalized in four
switchable pieces so each is an ablation rather than an assumption:

``compound_set``  ``tokens``: one token per inhibitor in the medium, all placed in front
                  of the gene tokens, so the cell-in-that-medium state ``H^c`` is read
                  under the whole mixture. ``mean``: the tokens are averaged into one
                  before the transformer, which is 038's single-token model.
``dose``          ``film``: an MLP of the standardized log10 molar dose produces a scale
                  and a shift on each compound token (FiLM). ``offset``: the tokens stay
                  dose-free and the dose enters as a gain on the medium OFFSET alone,
                  ``offset x (1 + g(dose))`` with ``g`` an MLP started at zero, so the
                  gene path reads the same compound token at every dose and only the
                  medium's potency moves with it. ``off``: 038's dose-free token.
``host_head``     a second readout, MLP([mean over genes of ``H^c`` ; mean of the
                  compound tokens after they have read the genome]) through a sigmoid to
                  a host fitness. It reads no genotype, which is what the bAID host is.
``sources``       the gene-level task may pool Hoepfner 2014's homozygous arm and
                  Hillenmeyer 2008's heterozygous arm behind the Vanacloig task, each
                  oriented sick-negative and standardized on its own fitted conditions,
                  each carrying a learned source token into the readout (030, 031).

THE GENE READOUT IS 038's, unchanged in form: the rows of ``H^c`` at the strain's deleted
genes summed, the strain's own post-deletion rows from the encoder's deletion operator,
the mean of ``H^c`` over genes, and the compound token after it read the genome, beside a
per-source gene bias and a compound offset. The fifth block is the source token.

EVERY RECORD IS A CELL under ``readout: cell`` (round 7): a genotype SET in a medium,
read through that one head. The host is the record with the EMPTY set, so its two
perturbed-gene blocks are zero and it carries no gene bias; it has its own source token,
scale and shift and shares every other weight with the gene-level records. The gene bias
is summed over the set, so a strain with two deletions carries both. Under
``readout: gene`` (rounds 1 to 6) the host has its own sigmoid head on the medium alone.

A STEP trains every active gene source (one gene batch each, against the fitted
conditions) and the host records together: the loss is the sum of each source's masked
MSE on its standardized values (the auxiliary sources weighted by ``aux_weight``) plus
``lambda_host`` times the host MSE. ``H^c`` for the host records reuses the wildtype gene
tokens of the Vanacloig pass, which do not depend on the strain, so the host costs no
extra encoder forward.

``host_train`` is the question the wet-lab side asks:

``anchors``        public anchors only: every unambiguous IC30 dose at host fitness 0.70
                   and the compound-free medium at 1.0. ex21 is then held out.
``anchors+ex21``   the anchors and the 54 ex21 single-agent cells, jointly, throughout.
``finetune_ex21``  stage 1 is ``anchors``; stage 2 then trains ``finetune_epochs`` epochs
                   on the ex21 cells alone at ``finetune_lr``, with ``finetune_scope``
                   choosing what is unfrozen (``host_head``, or ``host_head+env_encoder``
                   which also unfreezes the compound MLP, the FiLM MLP and the token
                   layers). The gene encoder is frozen in both.

EVALUATION, written to ``results/mixture/<sweep>/<name>_scores.csv`` in long form with
the model-free reference beside every row that has one:

(a) compound-cold Vanacloig centered Spearman on the 038 folds, so the gene-level side is
    comparable with nested ridge on FCFP4 counts (median 0.359 over the corrected store's
    32 compounds) and 038 round 2. ``require_molar_dose`` drops the four percent-dose
    compounds, so the panel is 28 and the comparison with a 32-compound median is
    approximate; every per-compound score is written out beside it.
(b) ex21: Spearman of predicted against observed fitness over all 180 wells and per
    compound, labeled as a train fit when ex21 was trained on.
(c) ex23, ONCE PER GROWTH CALL: growth / no-growth AUROC over the 63 combinations with
    the predicted fitness as the score; a ``worst_scenario_spearman`` over all 63 with a
    combination that did not grow scored at fitness zero, which is the ranking the
    detoxification-target question asks for; and fitness Spearman over the combinations
    that grew. Each call brings its own grown set, tau and fitness scale, and its own
    references: Loewe from the ex21 Hill fits (AUROC 0.958 served, 0.952 software) and
    Bliss from the observed ex23 singles (Spearman 0.573 served, 0.865 software) from
    ``results/mixture_scores.csv``, and for the worst-scenario ranking Loewe against
    ``observed_zero_mean`` (0.772 served, 0.800 software) computed from
    ``results/mixture_combinations.csv``.
(d) isoboles, SERVED CALL ONLY: the mean of observed minus predicted over the 81 interior
    cells of each grid, beside the same quantity for Bliss
    (``results/isobole_summary.csv``). The Bioscreen software generation times exist only
    for the ex23 plate set, so all 600 isobole wells and all 180 ex21 wells carry an
    empty software call; those rows therefore cannot be scored under it, and the ex21 and
    Vanacloig rows carry ``call`` = ``both``.

Run with a sweep file (a YAML list of ``MixtureConfig`` dicts):

    python train_mixture.py --sweep conf/mixture/smoke.yaml

ATTRIBUTION: ``TokenLayer``, ``StrainBatch``, ``strain_batch``, ``build_encoder``, the
OneCycle / autocast training loop and the sweep CLI are experiment 038's
``scripts/train_factorized.py`` (worktree ``exp/038-env-chemgen-vanacloig-cgt-corrected``,
commit 9427c0d4e), restated here so the 040 tree is self-contained on Delta; 038 is a
read-only reference and is not modified.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import re
import socket
import sys
import time
import warnings
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
import yaml
from dotenv import load_dotenv
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, osp.dirname(__file__))
from mixture_data import (  # noqa: E402
    ISOBOLE_RUNS,
    SOURCE_ATTRIBUTES,
    STORE,
    Fold,
    GeneSource,
    HostRecord,
    MixtureData,
    assemble,
    build_cell_graph,
    holdout_conditions,
    make_folds,
    make_gene_folds,
    score_compounds,
    score_genes,
    strain_indices,
    subsample_pool,
)
from store_manifest_040 import read_manifest  # noqa: E402

# a gene measured under no fitted compound has no mean; it is predicted NaN and left
# unscored, which is what np.nanmean warns about here (038 ``inhibitor_profiles.py``)
warnings.filterwarnings("ignore", message="Mean of empty slice")

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
WANDB_MODE = os.getenv("WANDB_MODE")
EXPERIMENT = osp.join(EXPERIMENT_ROOT, "040-inhibitor-synergy-wetlab")
RESULTS = osp.join(EXPERIMENT, "results")
WELLS_CSV = osp.join(RESULTS, "wetlab_wells.csv")
MODEL_YAML = osp.join(EXPERIMENT, "conf", "mixture", "cgt_model.yaml")
PREDICTIONS = osp.join(
    DATA_ROOT, "experiments", "040-inhibitor-synergy-wetlab", "predictions"
)
#: One checkpoint per (config, fold, seed): the weights, the configuration, the
#: standardization and the condition bookkeeping ``query_mixture.py`` needs to pose a new
#: genotype, compound set and dose to a trained model.
CHECKPOINTS = osp.join(
    DATA_ROOT, "experiments", "040-inhibitor-synergy-wetlab", "checkpoints"
)
#: Fitted auxiliary conditions whose predicted mean centers a held-out auxiliary score
#: (a full predicted matrix of Hoepfner's 550 conditions per seed is too costly).
AUX_CENTER_CONDITIONS = 32
#: The model-free bars every run's table is printed against (``results/mixture_scores.csv``
#: for ex23, ``results/isobole_summary.csv`` for the grids), served growth call.
REFERENCE_SCORES = osp.join(RESULTS, "mixture_scores.csv")
REFERENCE_ISOBOLES = osp.join(RESULTS, "isobole_summary.csv")
#: One row per (combination, call), with ``observed_zero_mean`` (no growth read as 0) and
#: each rule's prediction: the worst-scenario reference over all 63 is computed from it.
REFERENCE_COMBINATIONS = osp.join(RESULTS, "mixture_combinations.csv")
#: ``isobole_summary.csv`` holds one set of rows, on the served call, because the software
#: generation times exist only for the ex23 plate set.
ISOBOLE_REFERENCE_CALL = "served"
#: The row value of ``call`` for a score no growth call enters: the gene-level task and
#: the ex21 titrations, whose fitness column is served-only and has no software twin.
CALL_INDEPENDENT = "both"
#: Nested ridge on FCFP4 counts, compound-cold median centered Spearman over the
#: corrected store's 32 published compounds (038 round 1, slurm 3374).
RIDGE_VANACLOIG_MEDIAN = 0.359


class MixtureConfig(BaseModel):
    """One configuration, trained on every fold and seed it names."""

    model_config = ConfigDict(extra="forbid")

    name: str
    # ---- data
    sources: list[str] = ["vanacloig"]
    aux_weight: float = 1.0
    #: auxiliary conditions sampled per source per step, 0 = every condition
    cond_batch: int = 24
    #: what is held out of every auxiliary source for its own score and held-out loss:
    #: ``compound`` a fraction of its compounds, ``dose`` one dose of every multi-dose
    #: compound, ``none`` nothing (rounds 1 and 2)
    aux_holdout: Literal["none", "compound", "dose"] = "none"
    aux_holdout_frac: float = 0.1
    #: ``compound``: the 038 compound-cold folds over the Vanacloig compounds. ``gene``:
    #: gene-cold folds over the Vanacloig genes with every compound fitted; auxiliary
    #: sources keep every gene, so with them a held-out gene is cold in Vanacloig only.
    split: Literal["compound", "gene"] = "compound"
    #: how the source enters the gene readout: one learned token per source, or the sum
    #: of typed attribute tokens (ploidy, perturbation, medium, label, assay) shared
    #: across sources that agree on an attribute
    source_encoding: Literal["token", "attributes"] = "token"
    #: a per-source scale and shift on the shared head's output (one decoder, one linear
    #: head per label), beside the per-source gene bias every arm carries
    source_head: bool = False
    #: WHAT A RECORD IS. ``gene``: the public screens are read at the deleted genes
    #: through the gene head and the host through its own sigmoid head on the medium
    #: alone (rounds 1 to 6). ``cell``: every record is a cell, a genotype SET in a
    #: medium, read through ONE head; the host is the record with the EMPTY set, so its
    #: perturbed-gene blocks are zero, it carries no gene bias, it has its own source
    #: token, scale and shift, and it shares every other weight with the gene-level
    #: records. The gene-level computation is unchanged in form, so ``cell`` with one
    #: source reproduces ``gene`` on the public side up to the host's gradient.
    readout: Literal["gene", "cell"] = "gene"
    #: write a checkpoint per (fold, seed)
    checkpoint: bool = True
    require_molar_dose: bool = True
    #: the growth call the HOST HEAD TRAINS on. Only the served call covers ex21, so
    #: "software" leaves an ex21-training arm with no record and stops the run; the
    #: EVALUATION is emitted for every call the wells carry regardless of this field.
    call: Literal["served", "software"] = "served"
    embeddings: list[str] = ["fcfp4_count"]
    pca_dim: int | None = None
    # ---- the ablation ladder
    compound_set: Literal["tokens", "mean"] = "tokens"
    #: how the compound tokens of ONE medium are reduced to the medium's offset and to
    #: the token summary the head reads: ``mean`` makes a three-inhibitor medium read as
    #: the average of its inhibitors (round 7 and earlier), ``sum`` lets each inhibitor
    #: add; a single-compound medium is identical under both. ``bliss`` reads a
    #: k-compound medium as k single-compound media through the head and composes
    #: them after it, the host's fitness as the product of its single-agent fitness
    #: (Bliss independence), a source's standardized response as the sum; the head
    #: never sees a multi-compound token (needs ``readout: cell``)
    compound_pool: Literal["mean", "sum", "bliss"] = "mean"
    #: the W&B project. Rounds 1 to 8 logged to ``torchcell_040-inhibitor-synergy-wetlab``
    #: as one run per config file with fold- and seed-prefixed keys; from round 9 every
    #: (fold, seed) trajectory is its own run, grouped by arm, with the keys every run
    #: shares (``train_loss/total``, ``val_loss/total`` and their terms, by epoch), in a
    #: project of its own so the cell-readout program is not mixed with the gene-head runs
    wandb_project: str = "torchcell_040a-inhibitor-synergy-cell"
    #: ``film`` modulates every compound token by its dose before the transformer;
    #: ``offset`` leaves the tokens dose-free and scales the medium offset by
    #: ``1 + g(dose)`` (round 10: FiLM read dose but cost the public score under every
    #: readout, and under Bliss the host too); ``off`` reads no dose
    dose: Literal["film", "offset", "off"] = "film"
    #: HOW THE MEDIUM PERTURBS THE CELL (round 11). ``attend``: the compound tokens sit
    #: in front of the gene tokens through ``env_layers`` transformer layers (038).
    #: ``displace``: no medium transformer; a compound token displaces every gene
    #: embedding, ``H^c = H + sigmoid(W H) * u(e)``, a per-gene gate on a learned
    #: direction of the token, the environment's twin of the encoder's deletion
    #: operator; under ``dose: offset`` the dose gain scales the displacement as well
    #: as the offset. With one compound per medium (Bliss) a cross-attention from the
    #: genes to the medium has one key, so the gated displacement is that attention.
    env_mode: Literal["attend", "displace"] = "attend"
    #: the head reads the perturbed cell state only: the compound-token block and the
    #: medium offset are zero, so a compound's effect has to travel through ``H^c``
    cell_only: bool = False
    #: the host's functional form. ``head``: the shared head's scalar (rounds 7 to 10).
    #: ``hill``: a second linear readout of the same state gives each single-compound
    #: medium a Hill curve, a log IC50 and a slope in standardized log-dose units, and
    #: the host's fitness is that curve at the medium's dose; the k curves of a
    #: k-compound medium compose by ``compose``. Needs ``compound_pool: bliss``.
    host_form: Literal["head", "hill"] = "head"
    #: how the host's k single-compound readings compose under ``compound_pool: bliss``:
    #: ``bliss`` the product, ``loewe`` the dose-additive isobole solved on the k Hill
    #: curves (needs ``host_form: hill``), ``hsa`` the highest single agent (minimum)
    compose: Literal["bliss", "loewe", "hsa"] = "bliss"
    #: a learned deviation from the composition rule, an MLP of the product of every
    #: pair of a medium's compound tokens, zero at the start; it trains only where a
    #: multi-compound record is fitted (Hillenmeyer HET's pairs) and reaches the host
    #: zero-shot, the environment's analog of a genetic interaction term
    deviation: Literal["off", "pair"] = "off"
    host_train: Literal["anchors", "anchors+ex21", "finetune_ex21"] = "anchors"
    lambda_host: float = 1.0
    host_batch: int = 16
    finetune_epochs: int = 5
    finetune_lr: float = 1e-4
    finetune_scope: Literal["host_head", "host_head+env_encoder"] = "host_head"
    # ---- widths and optimization (038 round 10)
    dim: int = 64
    hidden: int = 256
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    steps: int = 3000
    eval_every: int = 50
    epochs: int | None = None
    eval_epochs: int = 1
    select: Literal["val", "last"] = "last"
    fit_on: Literal["train", "pool"] = "pool"
    #: the learning curve: fit only this many of the fold's fittable compounds, a fixed
    #: shuffle of the pool under ``fit_on: pool`` (038) or of the train compounds under
    #: ``fit_on: train``, where the validation compounds stay held out at every point
    n_fit_compounds: int | None = None
    gene_batch: int = 128
    seeds: list[int] = [0, 1, 2]
    fold_seed: int = 0
    n_folds: int = 5
    n_val: int = 4
    folds: list[int] | None = None
    # ---- the encoder
    env_layers: int = 1
    env_chunk: int = 0
    cgt_lambda: float = 0.0
    cgt_layers: int = 1
    cgt_heads: int = 9
    cgt_dim: int = 180
    cgt_lr: float = 5e-4


def fitted_compounds(cfg: MixtureConfig, fold: Fold) -> list[int]:
    """The Vanacloig compounds the model is fitted on (038)."""
    if cfg.split == "gene":
        # every compound; the fold's indices are genes (fitted_genes)
        return list(range(cfg_n_compounds(cfg)))
    if cfg.fit_on == "train":
        return subsample_train(fold, cfg.fold_seed, cfg.n_fit_compounds)
    return subsample_pool(fold, cfg.fold_seed, cfg.n_fit_compounds)


def subsample_train(fold: Fold, fold_seed: int, n: int | None) -> list[int]:
    """The first ``n`` of a fixed shuffle of the fold's TRAIN compounds.

    The learning curve under the held-out protocol (round 3, wave 2): the validation
    compounds stay out of the fit whatever ``n`` is, so the selection signal is the same
    held-out signal every other arm of the round selects on, and the curve's points
    differ only in how many compounds were fitted. ``subsample_pool`` is its
    ``fit_on: pool`` twin (038), which draws from train and validation together.
    """
    if n is None:
        return fold.train
    assert 2 <= n <= len(fold.train), f"{n} compounds asked of {len(fold.train)} train"
    order = np.random.default_rng([fold_seed, fold.fold, 19]).permutation(fold.train)
    return sorted(int(i) for i in order[:n])


def centering_compounds(cfg: MixtureConfig, fold: Fold) -> list[int]:
    """The compounds whose mean centers the held-out scores (038).

    Under ``fit_on: train`` the center is the fitted compounds plus the validation
    compounds, which is every non-test compound unless ``n_fit_compounds`` subsampled
    the fit; a compound the model never fitted does not enter the predicted mean.
    """
    if cfg.fit_on == "pool" or cfg.split == "gene":
        return fitted_compounds(cfg, fold)
    return sorted(fitted_compounds(cfg, fold) + fold.val)


#: Set once per run from the loaded Vanacloig source, read by ``fitted_compounds`` under
#: the gene split, where the fold indexes genes and every compound is fitted.
_N_COMPOUNDS: dict[str, int] = {}


def cfg_n_compounds(cfg: MixtureConfig) -> int:
    return _N_COMPOUNDS[cfg.name]


def fitted_genes(cfg: MixtureConfig, fold: Fold, n_genes: int) -> list[int]:
    """The Vanacloig genes the model is fitted on: all of them unless the split is gene.

    Under the gene split the fold's validation genes are fitted only under
    ``fit_on: pool``, as the validation compounds are under the compound split.
    """
    if cfg.split == "gene":
        return fold.train if cfg.fit_on == "train" else sorted(fold.train + fold.val)
    return list(range(n_genes))


def gene_split_score(
    y: NDArray[np.float64], pred: NDArray[np.float64], test_genes: list[int]
) -> float:
    """Mean over compounds of the raw Spearman over the held-out genes."""
    rows = np.array(test_genes)
    return float(
        np.nanmean(
            [fast_spearman(pred[rows, j], y[rows, j]) for j in range(y.shape[1])]
        )
    )


def fast_spearman(pred: NDArray[np.float64], obs: NDArray[np.float64]) -> float:
    ok = np.isfinite(pred) & np.isfinite(obs)
    if ok.sum() < 10 or np.std(pred[ok]) < 1e-10:
        return float("nan")
    return float(np.corrcoef(rankdata(pred[ok]), rankdata(obs[ok]))[0, 1])


def centered_val_score(
    y: NDArray[np.float64], pred: NDArray[np.float64], train: list[int], held: list[int]
) -> float:
    measured_mean = np.nanmean(y[:, train], axis=1)
    predicted_mean = pred[:, train].mean(axis=1)
    return float(
        np.nanmean(
            [
                fast_spearman(pred[:, j] - predicted_mean, y[:, j] - measured_mean)
                for j in held
            ]
        )
    )


# ---- the encoder's batch and one transformer layer (038, restated) --------- #
class StrainBatch(dict):
    """The batch the encoder reads: ``batch["gene"]`` and ``batch.num_graphs``.

    Since commit 468253ee2 the encoder sizes the perturbation transform by
    ``batch.num_graphs`` (a wildtype genotype has no row in the assignment vector, so
    ``max + 1`` dropped a trailing one); a plain dict no longer suffices.
    """

    def __init__(self, gene: SimpleNamespace, num_graphs: int) -> None:
        """``gene`` holds the perturbation indices and their genotype assignment."""
        super().__init__(gene=gene)
        self.num_graphs = num_graphs


def strain_batch(strain: torch.Tensor) -> StrainBatch:
    """``strain`` [B, order] of gene indices -> the encoder's batch for B genotypes."""
    size, order = strain.shape
    gene = SimpleNamespace(
        perturbation_indices=strain.reshape(-1),
        perturbation_indices_batch=torch.arange(
            size, device=strain.device
        ).repeat_interleave(order),
    )
    return StrainBatch(gene, size)


class TokenLayer(nn.Module):
    """One pre-norm transformer layer over [compounds ; genes], attention never stored.

    ``scaled_dot_product_attention`` runs the flash or memory-efficient kernel on a GPU,
    so the N x N attention of 6,607 gene tokens is not materialized per condition.
    """

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        """``dim`` must be divisible by ``heads``."""
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[C, T, d] -> [C, T, d]."""
        c, t, d = x.shape
        q, k, v = (
            self.qkv(self.norm1(x))
            .view(c, t, 3, self.heads, d // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        backends = (
            [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            if x.is_cuda
            else [SDPBackend.MATH]
        )
        with sdpa_kernel(backends):
            attended = nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )
        x = x + self.proj(attended.transpose(1, 2).reshape(c, t, d))
        return x + self.ffn(self.norm2(x))


def build_encoder(cfg: MixtureConfig, cell_graph):
    """The cell graph transformer of 038 ``build_encoder``, read from ``cgt_model.yaml``."""
    from torchcell.models.equivariant_cell_graph_transformer import CellGraphTransformer

    with open(MODEL_YAML) as f:
        model_cfg = yaml.safe_load(f)["model"]
    assert cfg.cgt_dim % cfg.cgt_heads == 0, "cgt_dim must be divisible by cgt_heads"
    model_cfg["learnable_embedding"]["size"] = cfg.cgt_dim
    # the prior sits on layer 1 (python index); a one-layer encoder has no layer 1, so
    # without this it would train with no prior at all. Each head's own lambda is 1 and
    # the single scale is cfg.cgt_lambda.
    reg = model_cfg["graph_regularization"]
    reg["graph_reg_layer"] = min(1, cfg.cgt_layers - 1)
    for head in reg["regularized_heads"].values():
        head["layer"] = reg["graph_reg_layer"]
        head["lambda"] = 1.0
    assert len(reg["regularized_heads"]) <= cfg.cgt_heads, "fewer heads than graphs"
    return CellGraphTransformer(
        gene_num=model_cfg["gene_num"],
        hidden_channels=cfg.cgt_dim,
        num_transformer_layers=cfg.cgt_layers,
        num_attention_heads=cfg.cgt_heads,
        cell_graph=cell_graph,
        graph_regularization_config=reg,
        perturbation_head_config=model_cfg["perturbation_head"],
        dropout=model_cfg["dropout"],
        graph_reg_lambda=cfg.cgt_lambda,
        node_embeddings=None,
        learnable_embedding_config=model_cfg["learnable_embedding"],
    )


# ---- the model ------------------------------------------------------------- #
class Medium(BaseModel):
    """One group of conditions with the same number of compounds, as tensors."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    k: int  # compounds in the medium
    rows: list[int]  # positions in the caller's condition list
    x: torch.Tensor  # [C, k, features]
    dose: torch.Tensor  # [C, k] standardized log10 molar


def group_media(
    compound_rows: list[list[int]],
    doses: list[list[float]],
    x_all: torch.Tensor,
    center: float,
    scale: float,
) -> list[Medium]:
    """Split conditions into groups of equal compound count, so no attention mask is needed."""
    by_k: dict[int, list[int]] = {}
    for i, rows in enumerate(compound_rows):
        by_k.setdefault(len(rows), []).append(i)
    device = x_all.device
    out = []
    for k, rows in sorted(by_k.items()):
        index = torch.tensor(
            [compound_rows[i] for i in rows], dtype=torch.long, device=device
        )
        dose = torch.tensor(
            [doses[i] for i in rows], dtype=torch.float32, device=device
        ).reshape(len(rows), k)
        out.append(
            Medium(
                k=k,
                rows=rows,
                x=x_all[index].reshape(len(rows), k, x_all.shape[1]),
                dose=(dose - center) / scale,
            )
        )
    return out


def loewe_fitness(
    z: torch.Tensor,
    z50: torch.Tensor,
    slope: torch.Tensor,
    log_dose_scale: torch.Tensor,
    iterations: int = 40,
) -> torch.Tensor:
    """Loewe additivity on k Hill curves: the fitness ``E`` with ``sum_i d_i / D_i(E) = 1``.

    ``z`` [M, k] are the standardized log10 molar doses of each medium's compounds,
    ``z50`` and ``slope`` [B, M, k] the curves' parameters (fitness ``sigmoid(-slope (z -
    z50))``), and ``log_dose_scale`` is ``ln10 x`` the standardization scale, so a z
    difference times it is a natural-log dose ratio. ``D_i(E)`` is the dose at which
    curve ``i`` alone reads ``E``, ``z50_i + ln((1 - E) / E) / slope_i``, and the dose
    ratio ``d_i / D_i(E)`` is ``exp(log_dose_scale (z_i - D_i(E)))``, which rises with
    ``E``; the root is bracketed by bisection without gradient and then taken one Newton
    step with gradient, which is the implicit-function derivative of ``E`` in the
    curves' parameters. A medium whose doses are all below every curve reads ``E`` at
    the upper clamp, one whose doses are all above it at the lower clamp.
    """
    z = z.unsqueeze(0).expand_as(z50).float()
    z50, slope = z50.float(), slope.float()
    scale = log_dose_scale.float()

    def residual(fitness: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logit = torch.log((1.0 - fitness) / fitness)
        ratio = torch.exp(scale * (z - z50 - logit.unsqueeze(-1) / slope))
        value = ratio.sum(-1) - 1.0
        derivative = (ratio * scale / slope).sum(-1) / (fitness * (1.0 - fitness))
        return value, derivative

    low, high = 1e-3, 1.0 - 1e-3
    with torch.no_grad():
        lo = torch.full(z50.shape[:-1], low, device=z50.device)
        hi = torch.full(z50.shape[:-1], high, device=z50.device)
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            value, _ = residual(mid)
            lo = torch.where(value < 0, mid, lo)
            hi = torch.where(value < 0, hi, mid)
        fitness = 0.5 * (lo + hi)
    value, derivative = residual(fitness)
    return (fitness - value / derivative).clamp(low, high)


class MixtureModel(nn.Module):
    """The 038 environment encoder with a compound set, a dose, and a host readout."""

    def __init__(
        self,
        cfg: MixtureConfig,
        n_nodes: int,
        n_sources: int,
        feature_dim: int,
        encoder,
        source_attributes: list[list[int]] | None = None,
        n_attribute_values: list[int] | None = None,
    ) -> None:
        """``n_nodes`` is the cell graph's gene count; the gene bias is per source.

        ``source_attributes[s][a]`` is source ``s``'s value index of attribute ``a`` and
        ``n_attribute_values[a]`` the vocabulary size of attribute ``a``; both are read
        only under ``source_encoding: attributes``.
        """
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.n_nodes = n_nodes
        d = cfg.cgt_dim
        self.gene_bias = nn.Embedding(n_sources * n_nodes, 1)
        nn.init.zeros_(self.gene_bias.weight)
        #: under ``readout: cell`` the host is one more source, the empty genotype, with
        #: its own token, scale and shift and no gene bias
        self.host_source = n_sources
        n_tokens = n_sources + 1 if cfg.readout == "cell" else n_sources
        self.source_token = nn.Embedding(n_tokens, d)
        nn.init.normal_(self.source_token.weight, std=0.02)
        if cfg.source_encoding == "attributes":
            assert cfg.readout == "gene", (
                "readout: cell gives the host its own source token; the attribute "
                "encoding has no attribute row for the empty genotype yet"
            )
            assert source_attributes is not None and n_attribute_values is not None
            self.register_buffer(
                "source_attributes",
                torch.tensor(source_attributes, dtype=torch.long),
                persistent=True,
            )
            self.attribute_tokens = nn.ModuleList(
                nn.Embedding(n, d) for n in n_attribute_values
            )
            for emb in self.attribute_tokens:
                nn.init.normal_(emb.weight, std=0.02)
        if cfg.compound_pool == "bliss":
            assert cfg.readout == "cell" and cfg.compound_set == "tokens", (
                "compound_pool: bliss composes single-compound readings of the shared "
                "head; it needs readout: cell and compound_set: tokens"
            )
        if cfg.host_form == "hill":
            assert cfg.compound_pool == "bliss", (
                "host_form: hill gives one Hill curve per single-compound medium; it "
                "needs compound_pool: bliss"
            )
        if cfg.compose == "loewe":
            assert cfg.host_form == "hill", "compose: loewe solves the Hill curves"
        if cfg.compose != "bliss" or cfg.deviation != "off":
            assert cfg.compound_pool == "bliss", (
                "compose and deviation act on the split readings of compound_pool: bliss"
            )
        if cfg.cell_only:
            assert cfg.dose != "offset" or cfg.env_mode == "displace", (
                "cell_only zeroes the medium offset, the only dose path of dose: offset "
                "under env_mode: attend; use env_mode: displace or dose: film"
            )
        if cfg.env_mode == "displace":
            # the gate on the wildtype gene embeddings and the direction each compound
            # token displaces them in; the direction starts at zero, so the model starts
            # as the compound-free cell
            self.displace_gate = nn.Linear(d, d)
            self.displace = nn.Sequential(
                nn.Linear(d, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, d)
            )
            nn.init.zeros_(self.displace[-1].weight)
            nn.init.zeros_(self.displace[-1].bias)
        if cfg.host_form == "hill":
            # log IC50 and slope of the host's Hill curve, read from the same normalized
            # state as the head; the IC50 starts at the mean fitted dose and the slope at
            # a Hill coefficient of one once ``set_dose_scale`` has been called
            self.hill = nn.Linear(5 * d, 2)
            nn.init.zeros_(self.hill.weight)
            nn.init.zeros_(self.hill.bias)
            self.register_buffer("hill_slope_init", torch.zeros(()), persistent=True)
            self.register_buffer("log_dose_scale", torch.zeros(()), persistent=True)
        if cfg.deviation == "pair":
            self.deviation = nn.Sequential(
                nn.Linear(d, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, 1)
            )
            nn.init.zeros_(self.deviation[-1].weight)
            nn.init.zeros_(self.deviation[-1].bias)
        # one decoder, one linear head per source: scale 1 and shift 0 at the start, so
        # the arm starts as the shared head
        self.source_scale = nn.Embedding(n_tokens, 1)
        self.source_shift = nn.Embedding(n_tokens, 1)
        nn.init.ones_(self.source_scale.weight)
        nn.init.zeros_(self.source_shift.weight)
        self.compound = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(feature_dim, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, d),
        )
        # FiLM on each compound token from its standardized log10 molar dose; the last
        # layer starts at zero so the model starts as the dose-free token
        self.film = nn.Sequential(
            nn.Linear(1, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, 2 * d)
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        # the gain on the medium offset under ``dose: offset``; zero at the start, so
        # the model starts as the dose-free offset
        self.dose_gain = nn.Sequential(
            nn.Linear(1, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, 1)
        )
        nn.init.zeros_(self.dose_gain[-1].weight)
        nn.init.zeros_(self.dose_gain[-1].bias)
        self.offset = nn.Linear(d, 1)
        self.layers = nn.ModuleList(
            TokenLayer(d, cfg.cgt_heads, cfg.dropout)
            for _ in range(cfg.env_layers if cfg.env_mode == "attend" else 0)
        )
        self.gene_norm = nn.LayerNorm(5 * d)
        self.gene_head = nn.Sequential(
            nn.Linear(5 * d, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, 1),
        )
        self.host_norm = nn.LayerNorm(2 * d)
        self.host_head = nn.Sequential(
            nn.Linear(2 * d, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, 1),
        )

    def env_parameters(self) -> list[nn.Parameter]:
        """The compound representation and the medium transformer, for the fine-tune."""
        modules = [self.compound, self.film, self.dose_gain, self.layers, self.offset]
        if self.cfg.env_mode == "displace":
            modules += [self.displace_gate, self.displace]
        if self.cfg.host_form == "hill":
            modules.append(self.hill)
        if self.cfg.deviation == "pair":
            modules.append(self.deviation)
        return [p for module in modules for p in module.parameters()]

    def set_dose_scale(self, dose_scale: float) -> None:
        """Under ``host_form: hill``, start every curve at a Hill coefficient of one.

        Doses reach the model standardized, ``z = (log10 molar - center) / scale``, so a
        Hill curve of coefficient ``n`` in molar units is ``sigmoid(-n ln10 scale (z -
        z50))``; the slope the head learns is ``softplus(s + hill_slope_init)`` with
        ``hill_slope_init`` the inverse softplus of ``ln10 scale``, and ``log_dose_scale``
        is what the Loewe solver needs to turn a z difference into a dose ratio.
        """
        if self.cfg.host_form != "hill":
            return
        slope = float(np.log(10.0) * dose_scale)
        self.hill_slope_init.fill_(float(np.log(np.expm1(slope))))
        self.log_dose_scale.fill_(slope)

    def cell(self, strain: torch.Tensor, cell_graph):
        """(wildtype tokens [N, d], the strains' deleted rows [B, d], graph penalty)."""
        _, out = self.encoder(cell_graph, strain_batch(strain))
        rows = torch.arange(strain.shape[0], device=strain.device).unsqueeze(1)
        h_del = out["H_genes_pert"][rows, strain].sum(dim=1)
        return out["H_genes"], h_del, out["graph_reg_loss"]

    def tokens(self, medium: Medium) -> torch.Tensor:
        """[C, k', d] compound tokens; k' is 1 when ``compound_set`` is ``mean``."""
        e = self.compound(medium.x)  # [C, k, d]
        if self.cfg.dose == "film":
            gamma, beta = self.film(medium.dose.unsqueeze(-1)).chunk(2, dim=-1)
            e = e * (1.0 + gamma) + beta
        if self.cfg.compound_set == "mean" and medium.k > 1:
            e = e.mean(dim=1, keepdim=True)
        return e

    def gain(self, medium: Medium) -> torch.Tensor | None:
        """[C, k] gain on the medium offset under ``dose: offset``, else None."""
        if self.cfg.dose != "offset" or not medium.k:
            return None
        return 1.0 + self.dose_gain(medium.dose.unsqueeze(-1)).squeeze(-1)

    def pool_compounds(self, tokens: torch.Tensor) -> torch.Tensor:
        """[C, k, d] -> [C, d]: the medium's compound tokens reduced by ``compound_pool``."""
        if self.cfg.compound_pool == "sum":
            return tokens.sum(1)
        return tokens.mean(1)

    def split_media(self, e: torch.Tensor) -> torch.Tensor:
        """Under ``compound_pool: bliss`` a k-compound medium is read as k single-compound
        media, [C, k, d] -> [C x k, 1, d], and composed after the head; otherwise ``e``.
        """
        c, k, d = e.shape
        if self.cfg.compound_pool == "bliss" and k > 1:
            return e.reshape(c * k, 1, d)
        return e

    def split_gain(self, gain: torch.Tensor | None) -> torch.Tensor | None:
        """[C, k] -> [C'] the dose gain per (split) medium, averaged over its compounds."""
        if gain is None:
            return None
        return self.split_media(gain.unsqueeze(-1)).squeeze(-1).mean(dim=1)

    def medium_state(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        strain: torch.Tensor | None,
        gain: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Run the medium's tokens with the gene tokens; reduce inside the chunk loop.

        Returns (rows of ``H^c`` at each strain's deleted genes [B, C, d] or None when no
        strain is read, the mean of ``H^c`` over genes [C, d], the mean of the compound
        tokens after they read the genome [C, d]). Under ``compound_pool: bliss`` C is
        the number of single-compound media the k-compound media split into. ``gain``
        [C, k] is the dose gain under ``dose: offset``; under ``env_mode: displace`` it
        scales the displacement of the gene embeddings.
        """
        e = self.split_media(e)
        g = self.split_gain(gain)
        n, d = h.shape
        c, k = e.shape[0], e.shape[1]
        gate = (
            torch.sigmoid(self.displace_gate(h))
            if self.cfg.env_mode == "displace" and k
            else None
        )
        env, pooled, token = [], [], []
        for start in range(0, c, self.cfg.env_chunk or c):
            part = e[start : start + (self.cfg.env_chunk or c)]
            genes = h.unsqueeze(0).expand(len(part), n, d)
            if gate is not None:
                # the medium's token displaces every gene embedding along a learned
                # direction, gated per gene; no token reads the genome
                u = self.displace(self.pool_compounds(part))
                if g is not None:
                    u = u * g[start : start + len(part), None]
                cell = genes + gate.unsqueeze(0) * u.unsqueeze(1)
                read = part
            else:
                seq = torch.cat([part, genes], dim=1) if k else genes
                for layer in self.layers:
                    seq = layer(seq)
                cell = seq[:, k:]
                read = seq[:, :k]
            if strain is not None:
                env.append(cell[:, strain].sum(2))
            pooled.append(cell.mean(1))
            token.append(
                self.pool_compounds(read)
                if k
                else torch.zeros(len(part), d, device=h.device, dtype=h.dtype)
            )
        return (
            torch.cat(env).transpose(0, 1) if strain is not None else None,
            torch.cat(pooled),
            torch.cat(token),
        )

    def gene_logits(
        self,
        env: torch.Tensor | None,
        h_del: torch.Tensor | None,
        pooled: torch.Tensor,
        token: torch.Tensor,
        strain: torch.Tensor | None,
        source_index: int,
        e: torch.Tensor,
        gain: torch.Tensor | None = None,
        dose: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """[B, C] predictions on the standardized target for the source's conditions.

        ``strain`` is ``[B, order]``: the gene bias is summed over the genotype's set, so
        a strain with two deletions carries both. ``strain`` ``None`` is the EMPTY
        genotype (the host under ``readout: cell``): the perturbed-gene blocks are zero
        and there is no gene bias, so the cell is read at its gene mean and its medium
        alone, through the same head and with its own source scale and shift. ``gain``
        is ``[C, k]`` under ``dose: offset`` and multiplies the medium offset (split
        like ``e`` under Bliss, averaged over a medium's compounds otherwise). ``dose``
        is ``[C, k]`` standardized log10 molar, read by the host under ``host_form:
        hill``.
        """
        k_set = e.shape[1]
        e = self.split_media(e)
        c, d = pooled.shape
        if env is None or h_del is None:
            assert strain is None, "an empty genotype reads no perturbed rows"
            env = torch.zeros(1, c, d, device=pooled.device, dtype=pooled.dtype)
            h_del = torch.zeros(1, d, device=pooled.device, dtype=pooled.dtype)
        b = env.shape[0]
        host = source_index == self.host_source
        source = self.source_vector(source_index, env.device)
        if self.cfg.cell_only:
            token = torch.zeros_like(token)
        state = torch.cat(
            [
                env,
                h_del[:, None].expand(b, c, d),
                pooled[None].expand(b, c, d),
                token[None].expand(b, c, d),
                source[None].expand(b, c, d),
            ],
            dim=-1,
        )
        if strain is None:
            gene = torch.zeros(b, 1, device=env.device, dtype=torch.float32)
        else:
            gene = self.gene_bias(source_index * self.n_nodes + strain).sum(dim=1)
        # the compound-free medium (the host's uninhibited anchor) has no compound
        # token, and the mean over an empty set is NaN; its offset is zero
        if e.shape[1] and not self.cfg.cell_only:
            offset = self.offset(self.pool_compounds(e)).T.expand(b, c)
            g = self.split_gain(gain)
            if g is not None:
                offset = offset * g[None, :].expand(b, c)
        else:
            offset = torch.zeros(b, c, device=env.device, dtype=torch.float32)
        normed = self.gene_norm(state.float())
        head = self.gene_head(normed).squeeze(-1)
        if self.cfg.source_head or host:
            index = torch.full((1,), source_index, dtype=torch.long, device=env.device)
            head = head * self.source_scale(index) + self.source_shift(index)
        out = offset + head
        if host and self.cfg.host_form == "hill":
            if e.shape[1]:
                # one Hill curve per single-compound medium, read at its dose
                assert dose is not None, "host_form: hill needs the medium's doses"
                z = self.split_media(dose.unsqueeze(-1)).squeeze(-1)[:, 0]
                params = self.hill(normed).float()  # [b, c, 2]
                z50 = params[..., 0]
                slope = nn.functional.softplus(params[..., 1] + self.hill_slope_init)
                out = torch.sigmoid(-slope * (z[None, :] - z50))
            else:
                # the compound-free medium: no curve, fitness one
                out = torch.ones(b, c, device=env.device, dtype=torch.float32)
        if self.cfg.compound_pool == "bliss" and k_set > 1:
            # the k single-compound readings of one medium, composed as Bliss
            # independence: on the host's fitness scale the medium's fitness is the
            # PRODUCT of its compounds' single-agent fitness (the uninhibited anchor
            # reads 1); on a source's standardized log-ratio response the compounds ADD.
            # Round 9's sum pooling read a three-inhibitor medium as fitter than its
            # singles (0.74, 0.78, 0.86 at one, two, three) because a summed token is
            # outside anything the head was trained on; here every reading is a
            # single-compound medium of the kind the head is trained on.
            out = out.view(b, c // k_set, k_set)
            if host and self.cfg.compose == "loewe":
                assert dose is not None
                z = self.split_media(dose.unsqueeze(-1)).squeeze(-1)[:, 0]
                out = loewe_fitness(
                    z.view(c // k_set, k_set),
                    z50.view(b, c // k_set, k_set),
                    slope.view(b, c // k_set, k_set),
                    self.log_dose_scale,
                )
            elif host and self.cfg.compose == "hsa":
                out = out.min(dim=-1).values
            elif host:
                out = out.clamp(min=1e-3).prod(dim=-1)
            else:
                out = out.sum(dim=-1)
            if self.cfg.deviation == "pair":
                # every pair of the medium's compound tokens, before they read the
                # genome, through one zero-started MLP; the same term on every scale
                t = e[:, 0].view(c // k_set, k_set, d)
                i, j = torch.triu_indices(k_set, k_set, offset=1, device=e.device)
                pair = self.deviation((t[:, i] * t[:, j]).float()).squeeze(-1).sum(-1)
                out = out + pair[None, :]
        return gene + out

    def host_logits(
        self,
        pooled: torch.Tensor,
        token: torch.Tensor,
        e: torch.Tensor,
        gain: torch.Tensor | None = None,
        dose: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """[C] host fitness per medium: the sigmoid head, or the shared head at the empty set."""
        if self.cfg.readout == "gene":
            return self.host_fitness(pooled, token)
        return self.gene_logits(
            None, None, pooled, token, None, self.host_source, e, gain, dose
        )[0]

    def host_parameters(self) -> list[nn.Parameter]:
        """What a host fine-tune unfreezes: the host head, or the shared head and tokens."""
        if self.cfg.readout == "gene":
            return list(self.host_head.parameters()) + list(self.host_norm.parameters())
        modules = [
            self.gene_head,
            self.gene_norm,
            self.source_token,
            self.source_scale,
            self.source_shift,
        ]
        return [p for module in modules for p in module.parameters()]

    def source_vector(self, source_index: int, device: torch.device) -> torch.Tensor:
        """[1, d] the source's token: learned per source, or the sum of its attributes."""
        index = torch.full((1,), source_index, dtype=torch.long, device=device)
        if self.cfg.source_encoding == "token":
            return self.source_token(index)
        values = self.source_attributes[source_index]
        return torch.stack(
            [emb(values[a : a + 1]) for a, emb in enumerate(self.attribute_tokens)]
        ).sum(0)

    def host_fitness(self, pooled: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        """[C] host fitness in (0, 1) from the medium alone: the empty-genotype path."""
        state = self.host_norm(torch.cat([pooled, token], dim=-1).float())
        return torch.sigmoid(self.host_head(state).squeeze(-1))


# ---- what each task trains on ---------------------------------------------- #
class Context(BaseModel):
    """The data every seed of a config shares."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: MixtureData
    cell_graph: object
    node_ids: list[str]
    #: per source name, [n_genes, order] cell-graph indices of the strain's deletions
    strain: dict[str, torch.Tensor]
    #: auxiliary conditions FITTED, after removing the Vanacloig compounds and the hold-out
    aux_conditions: dict[str, list[int]]
    #: auxiliary conditions held out for the source's own score (``aux_holdout``)
    aux_heldout: dict[str, list[int]]
    #: the fitted auxiliary conditions whose predicted mean centers that score
    aux_center: dict[str, list[int]]
    n_aux_dropped: dict[str, int]
    #: per source, its attribute value indices; and per attribute, its vocabulary
    source_attributes: list[list[int]]
    attribute_values: dict[str, list[str]]


def build_context(cfg: MixtureConfig, device: torch.device) -> Context:
    data = assemble(
        cfg.sources,
        WELLS_CSV,
        cfg.embeddings,
        cfg.pca_dim,
        cfg.require_molar_dose,
        cfg.call,
    )
    with open(MODEL_YAML) as f:
        graphs = list(yaml.safe_load(f)["cell_dataset"]["graphs"])
    cell_graph = build_cell_graph(graphs)
    node_ids = list(cell_graph["gene"].node_ids)
    strain = {s.spec.name: strain_indices(s, node_ids) for s in data.sources}
    vanacloig_rows = set(int(r) for r in data.sources[0].compound_row)
    aux, held, center, dropped = {}, {}, {}, {}
    for source in data.sources[1:]:
        keep = [
            j
            for j in range(len(source.compounds))
            if int(source.compound_row[j]) not in vanacloig_rows
            and (cfg.dose == "off" or np.isfinite(source.log10_molar[j]))
        ]
        assert keep, f"{source.spec.name} has no condition left after filtering"
        fitted, heldout = holdout_conditions(
            source, keep, cfg.aux_holdout, cfg.aux_holdout_frac, cfg.fold_seed
        )
        aux[source.spec.name] = fitted
        held[source.spec.name] = heldout
        center[source.spec.name] = sorted(
            int(j)
            for j in np.random.default_rng([cfg.fold_seed, source.index, 11]).choice(
                fitted, size=min(AUX_CENTER_CONDITIONS, len(fitted)), replace=False
            )
        )
        dropped[source.spec.name] = len(source.compounds) - len(keep)
    values = {
        a: sorted({s.spec.attributes[a] for s in data.sources})
        for a in SOURCE_ATTRIBUTES
    }
    attributes = [
        [values[a].index(s.spec.attributes[a]) for a in SOURCE_ATTRIBUTES]
        for s in data.sources
    ]
    _N_COMPOUNDS[cfg.name] = len(data.sources[0].compounds)
    return Context(
        data=data,
        cell_graph=cell_graph.to(device),
        node_ids=node_ids,
        strain=strain,
        aux_conditions=aux,
        aux_heldout=held,
        aux_center=center,
        n_aux_dropped=dropped,
        source_attributes=attributes,
        attribute_values=values,
    )


def host_training_records(
    cfg: MixtureConfig, ctx: Context, fold: Fold, stage: Literal["main", "finetune"]
) -> list[HostRecord]:
    """The host records this stage fits, with the fold's test compounds removed.

    AN ANCHOR NAMES A COMPOUND, so an anchor for a held-out compound would let the
    compound encoder see the test compound's structure and its IC30 dose. Every record
    carrying a test compound is therefore dropped, and the gene-level score stays
    compound-cold.
    """
    vanacloig = ctx.data.sources[0]
    # under the gene split the fold indexes genes and no compound is held out
    held = (
        set()
        if cfg.split == "gene"
        else {int(vanacloig.compound_row[j]) for j in fold.test}
    )
    if (
        cfg.host_train == "anchors"
        or stage == "main"
        and cfg.host_train == "finetune_ex21"
    ):
        splits = {"anchor"}
    elif cfg.host_train == "anchors+ex21":
        splits = {"anchor", "ex21"}
    else:
        splits = {"ex21"}
    return [
        record
        for record in ctx.data.host
        if record.split in splits
        and not held.intersection(record.compound_row)
        and (cfg.dose == "off" or all(np.isfinite(record.log10_molar)))
    ]


class Standardization(BaseModel):
    """How the compound features, the doses and each source's values are scaled."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    x: torch.Tensor  # [n_compounds, kept features], standardized on the fitted rows
    #: the feature scaling itself, so a compound with no row in the table (a counterfactual
    #: molecule, featurized by the same fingerprint) can be put on this model's scale
    x_keep: NDArray[np.bool_]
    x_mean: NDArray[np.float64]
    x_sd: NDArray[np.float64]
    dose_center: float
    dose_scale: float
    y_mean: dict[str, float]
    y_sd: dict[str, float]


def standardize(
    cfg: MixtureConfig,
    ctx: Context,
    fold: Fold,
    host: list[HostRecord],
    device: torch.device,
    genes_fit: list[int] | None = None,
) -> Standardization:
    """Every scale is computed on the fitted conditions only (the 031 rule).

    Under the gene split the Vanacloig scale is computed on the fitted GENES as well.
    """
    fitted = {
        ctx.data.sources[0].spec.name: fitted_compounds(cfg, fold),
        **ctx.aux_conditions,
    }
    rows, doses = set(), []
    for source in ctx.data.sources:
        for j in fitted[source.spec.name]:
            rows.add(int(source.compound_row[j]))
            if np.isfinite(source.log10_molar[j]):
                doses.append(float(source.log10_molar[j]))
    for record in host:
        doses.extend(d for d in record.log10_molar if np.isfinite(d))
    x_all = ctx.data.compounds.x_raw
    fit = x_all[sorted(rows)]
    mu, sd = fit.mean(0), fit.std(0)
    keep = sd > 1e-8
    x_std = (x_all[:, keep] - mu[keep]) / sd[keep]
    spread = float(np.std(doses))
    y_mean, y_sd = {}, {}
    for source in ctx.data.sources:
        block = source.y[:, fitted[source.spec.name]]
        if source.index == 0 and genes_fit is not None:
            block = block[np.array(genes_fit)]
        y_mean[source.spec.name] = float(np.nanmean(block))
        y_sd[source.spec.name] = float(np.nanstd(block))
    return Standardization(
        x=torch.tensor(x_std, dtype=torch.float32, device=device),
        x_keep=keep,
        x_mean=mu,
        x_sd=sd,
        dose_center=float(np.mean(doses)),
        dose_scale=spread if spread > 1e-8 else 1.0,
        y_mean=y_mean,
        y_sd=y_sd,
    )


def source_media(
    source: GeneSource, conditions: list[int], std: Standardization
) -> list[Medium]:
    """The source's conditions as media; every gene-level condition holds one compound."""
    return group_media(
        [[int(source.compound_row[j])] for j in conditions],
        [[float(source.log10_molar[j])] for j in conditions],
        std.x,
        std.dose_center,
        std.dose_scale,
    )


def host_media(records: list[HostRecord], std: Standardization) -> list[Medium]:
    return group_media(
        [list(r.compound_row) for r in records],
        [list(r.log10_molar) for r in records],
        std.x,
        std.dose_center,
        std.dose_scale,
    )


# ---- prediction ------------------------------------------------------------ #
@torch.no_grad()
def predict_gene_matrix(
    model: MixtureModel,
    ctx: Context,
    cfg: MixtureConfig,
    source: GeneSource,
    conditions: list[int],
    std: Standardization,
    device: torch.device,
    batch: int,
) -> NDArray[np.float64]:
    """[n_genes, n_conditions] predictions in the source's served units."""
    model.eval()
    strain_all = ctx.strain[source.spec.name]
    media = source_media(source, conditions, std)
    out = np.full((len(source.genes), len(conditions)), np.nan)
    for start in range(0, len(source.genes), batch):
        index = torch.arange(start, min(len(source.genes), start + batch))
        strain = strain_all[index].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            h, h_del, _ = model.cell(strain, ctx.cell_graph)
            for medium in media:
                e = model.tokens(medium)
                gain = model.gain(medium)
                env, pooled, token = model.medium_state(h, e, strain, gain)
                assert env is not None
                pred = model.gene_logits(
                    env,
                    h_del,
                    pooled,
                    token,
                    strain,
                    source.index,
                    e,
                    gain,
                    medium.dose,
                )
                columns = np.array(medium.rows)
                out[np.ix_(index.numpy(), columns)] = pred.float().cpu().numpy()
    return out * std.y_sd[source.spec.name] + std.y_mean[source.spec.name]


@torch.no_grad()
def predict_host(
    model: MixtureModel,
    ctx: Context,
    records: list[HostRecord],
    std: Standardization,
    device: torch.device,
) -> NDArray[np.float64]:
    """[len(records)] predicted host fitness; the wildtype tokens need no genotype."""
    model.eval()
    strain = ctx.strain[ctx.data.sources[0].spec.name][:1].to(device)
    out = np.full(len(records), np.nan)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        h, _, _ = model.cell(strain, ctx.cell_graph)
        for medium in host_media(records, std):
            e = model.tokens(medium)
            gain = model.gain(medium)
            _, pooled, token = model.medium_state(h, e, None, gain)
            out[np.array(medium.rows)] = (
                model.host_logits(pooled, token, e, gain, medium.dose)
                .float()
                .cpu()
                .numpy()
            )
    return out


# ---- training -------------------------------------------------------------- #
class SeedResult(BaseModel):
    """One seed's selected prediction on both tasks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vanacloig: NDArray[np.float64]  # [n_genes, n_conditions], served units
    host: NDArray[np.float64]  # [len(ctx.data.host)] predicted fitness
    #: per auxiliary source, [n_genes, n_conditions] in served units, predicted on the
    #: centering subset and the held-out conditions only, NaN elsewhere
    aux: dict[str, NDArray[np.float64]]
    history: pd.DataFrame


def predict_aux(
    model: MixtureModel,
    ctx: Context,
    cfg: MixtureConfig,
    std: Standardization,
    device: torch.device,
) -> dict[str, NDArray[np.float64]]:
    """Each auxiliary source's held-out and centering conditions, served units."""
    out = {}
    for source in ctx.data.sources[1:]:
        name = source.spec.name
        columns = sorted(set(ctx.aux_center[name]) | set(ctx.aux_heldout[name]))
        if not ctx.aux_heldout[name]:
            continue
        full = np.full((len(source.genes), len(source.compounds)), np.nan)
        full[:, columns] = predict_gene_matrix(
            model, ctx, cfg, source, columns, std, device, 256
        )
        out[name] = full
    return out


def heldout_loss(
    source: GeneSource,
    prediction: NDArray[np.float64],
    std: Standardization,
    columns: list[int],
    rows: list[int] | None = None,
) -> float:
    """Masked MSE on the standardized target over the held-out columns (and rows)."""
    y = source.y[:, columns]
    p = prediction[:, columns]
    if rows is not None:
        y, p = y[np.array(rows)], p[np.array(rows)]
    ok = np.isfinite(y) & np.isfinite(p)
    if not ok.any():
        return float("nan")
    sd = std.y_sd[source.spec.name]
    return float(np.mean(((p[ok] - y[ok]) / sd) ** 2))


def save_checkpoint(
    path: str,
    model: MixtureModel,
    cfg: MixtureConfig,
    ctx: Context,
    fold: Fold,
    seed: int,
    std: Standardization,
    selected_step: int,
) -> None:
    """Everything ``query_mixture.py`` needs to pose a new medium to this model."""
    os.makedirs(osp.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg.model_dump(),
            "fold": fold.model_dump(),
            "seed": seed,
            "selected_step": selected_step,
            "standardization": {
                "x": std.x.detach().cpu(),
                "x_keep": std.x_keep,
                "x_mean": std.x_mean,
                "x_sd": std.x_sd,
                "dose_center": std.dose_center,
                "dose_scale": std.dose_scale,
                "y_mean": std.y_mean,
                "y_sd": std.y_sd,
            },
            "compounds": {
                "inchikeys": list(ctx.data.compounds.inchikeys),
                "names": list(ctx.data.compounds.names),
            },
            "sources": [
                {
                    "name": s.spec.name,
                    "index": s.index,
                    "genes": s.genes,
                    "compounds": s.compounds,
                    "inchikeys": s.inchikeys,
                    "compound_row": s.compound_row.tolist(),
                    "log10_molar": s.log10_molar.tolist(),
                    "extra_genes": list(s.spec.extra_genes),
                    "attributes": s.spec.attributes,
                }
                for s in ctx.data.sources
            ],
            "node_ids": ctx.node_ids,
            "source_attributes": ctx.source_attributes,
            "attribute_values": ctx.attribute_values,
            "aux_conditions": ctx.aux_conditions,
            "aux_heldout": ctx.aux_heldout,
            "aux_center": ctx.aux_center,
            "model_yaml": MODEL_YAML,
        },
        path,
    )


def train_seed(
    cfg: MixtureConfig,
    ctx: Context,
    fold: Fold,
    seed: int,
    device: torch.device,
    sweep: str = "",
) -> SeedResult:
    """Fit one seed on this fold and return its selected predictions on both tasks."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    sources = ctx.data.sources
    vanacloig = sources[0]
    host = host_training_records(cfg, ctx, fold, "main")
    assert host, "the host task has no training record"
    # the host records the loss curves read but the optimizer never steps on: the ex21
    # single-agent cells outside the training list (the fold's held-out compounds under
    # anchors+ex21, all of ex21 under anchors; empty when every ex21 cell is fitted), as
    # the host's VALIDATION loss, and the 63 ex23 combinations, the private test, as its
    # TEST loss. Neither enters the selection, which reads the public validation score.
    trained_keys = {r.key for r in host}
    host_val_rows = np.array(
        [
            i
            for i, r in enumerate(ctx.data.host)
            if r.split == "ex21" and r.key not in trained_keys
        ],
        dtype=int,
    )
    host_test_rows = np.array(
        [i for i, r in enumerate(ctx.data.host) if r.run == "ex23" and r.compounds],
        dtype=int,
    )
    host_observed = np.array([r.fitness for r in ctx.data.host], dtype=float)
    genes_fit = fitted_genes(cfg, fold, len(vanacloig.genes))
    std = standardize(cfg, ctx, fold, host, device, genes_fit)
    conditions = {
        vanacloig.spec.name: fitted_compounds(cfg, fold),
        **ctx.aux_conditions,
    }
    # the genes each source's batches are drawn from: under the gene split the Vanacloig
    # test genes are never fitted; the auxiliary sources keep every gene
    gene_pool = {
        s.spec.name: torch.tensor(
            genes_fit if s.index == 0 else list(range(len(s.genes))), dtype=torch.long
        )
        for s in sources
    }
    target, mask = {}, {}
    for source in sources:
        name = source.spec.name
        block = (source.y - std.y_mean[name]) / std.y_sd[name]
        mask[name] = torch.tensor(np.isfinite(block), device=device)
        target[name] = torch.tensor(
            np.nan_to_num(block), dtype=torch.float32, device=device
        )

    model = MixtureModel(
        cfg,
        n_nodes=len(ctx.node_ids),
        n_sources=len(sources),
        feature_dim=std.x.shape[1],
        encoder=build_encoder(cfg, ctx.cell_graph),
        source_attributes=ctx.source_attributes,
        n_attribute_values=[len(ctx.attribute_values[a]) for a in SOURCE_ATTRIBUTES],
    ).to(device)
    model.set_dose_scale(std.dose_scale)
    groups = [
        {
            "params": [
                p for n, p in model.named_parameters() if not n.startswith("encoder.")
            ],
            "lr": cfg.lr,
        },
        {"params": list(model.encoder.parameters()), "lr": cfg.cgt_lr},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    steps_per_epoch = -(-len(genes_fit) // cfg.gene_batch)
    steps, eval_every = cfg.steps, cfg.eval_every
    if cfg.epochs is not None:
        steps = cfg.epochs * steps_per_epoch
        eval_every = cfg.eval_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.05
    )
    generator = torch.Generator().manual_seed(seed)
    order = {
        name: pool[torch.randperm(len(pool), generator=generator)]
        for name, pool in gene_pool.items()
    }
    cursor = dict.fromkeys(order, 0)
    rng = np.random.default_rng(seed)
    host_fitness = torch.tensor(
        [r.fitness for r in host], dtype=torch.float32, device=device
    )
    pool = centering_compounds(cfg, fold)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history: list[dict[str, float]] = []
    best = (-np.inf, -1)
    selected: SeedResult | None = None
    started = time.time()
    for step in range(steps):
        model.train()
        losses, penalties, h_wildtype = {}, [], None
        for source in sources:
            name = source.spec.name
            if cursor[name] + cfg.gene_batch > len(order[name]):
                pool_g = gene_pool[name]
                order[name] = pool_g[torch.randperm(len(pool_g), generator=generator)]
                cursor[name] = 0
            index = order[name][cursor[name] : cursor[name] + cfg.gene_batch]
            cursor[name] += cfg.gene_batch
            columns = conditions[name]
            if source.index > 0 and cfg.cond_batch and cfg.cond_batch < len(columns):
                columns = sorted(
                    rng.choice(columns, size=cfg.cond_batch, replace=False).tolist()
                )
            strain = ctx.strain[name][index].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                h, h_del, penalty = model.cell(strain, ctx.cell_graph)
                predictions = torch.zeros(
                    len(index), len(columns), device=device, dtype=torch.float32
                )
                for medium in source_media(source, columns, std):
                    e = model.tokens(medium)
                    gain = model.gain(medium)
                    env, pooled, token = model.medium_state(h, e, strain, gain)
                    assert env is not None
                    predictions = predictions.index_copy(
                        1,
                        torch.tensor(medium.rows, device=device),
                        model.gene_logits(
                            env,
                            h_del,
                            pooled,
                            token,
                            strain,
                            source.index,
                            e,
                            gain,
                            medium.dose,
                        ).float(),
                    )
            penalties.append(penalty)
            if source.index == 0:
                h_wildtype = h
            column_index = torch.tensor(columns, device=device)
            t = target[name][index.to(device)][:, column_index]
            m = mask[name][index.to(device)][:, column_index]
            losses[name] = ((predictions - t) ** 2 * m).sum() / m.sum()

        assert h_wildtype is not None
        picked = rng.choice(
            len(host), size=min(cfg.host_batch, len(host)), replace=False
        )
        records = [host[i] for i in picked]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            predicted = torch.zeros(len(records), device=device, dtype=torch.float32)
            for medium in host_media(records, std):
                e = model.tokens(medium)
                gain = model.gain(medium)
                _, pooled, token = model.medium_state(h_wildtype, e, None, gain)
                predicted = predicted.index_copy(
                    0,
                    torch.tensor(medium.rows, device=device),
                    model.host_logits(pooled, token, e, gain, medium.dose).float(),
                )
        host_loss = ((predicted - host_fitness[picked]) ** 2).mean()

        gene_loss = losses[vanacloig.spec.name] + cfg.aux_weight * sum(
            loss for name, loss in losses.items() if name != vanacloig.spec.name
        )
        total = gene_loss + cfg.lambda_host * host_loss
        if cfg.cgt_lambda:
            total = total + cfg.cgt_lambda * sum(penalties)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % eval_every == 0 or step + 1 == steps:
            matrix = predict_gene_matrix(
                model,
                ctx,
                cfg,
                vanacloig,
                list(range(len(vanacloig.compounds))),
                std,
                device,
                256,
            )
            host_pred = predict_host(model, ctx, ctx.data.host, std, device)
            aux_pred = predict_aux(model, ctx, cfg, std, device)
            if cfg.split == "gene":
                # the validation genes score the same way the test genes do; under
                # fit_on: pool they are fitted and the score is in-sample, as the
                # compound split's is
                val = gene_split_score(vanacloig.y, matrix, fold.val)
                test = gene_split_score(vanacloig.y, matrix, fold.test)
                van_heldout = heldout_loss(
                    vanacloig, matrix, std, conditions[vanacloig.spec.name], fold.test
                )
                van_val_loss = heldout_loss(
                    vanacloig, matrix, std, conditions[vanacloig.spec.name], fold.val
                )
            else:
                val = centered_val_score(
                    vanacloig.y, matrix, conditions[vanacloig.spec.name], fold.val
                )
                test = centered_val_score(vanacloig.y, matrix, pool, fold.test)
                van_heldout = heldout_loss(vanacloig, matrix, std, fold.test)
                # the validation compounds' loss in the training loss's own units;
                # in-sample under fit_on: pool, as the validation score is
                van_val_loss = heldout_loss(vanacloig, matrix, std, fold.val)
            # THE LOSS, IN FULL, FOR TRAIN AND FOR VALIDATION, each with every term it
            # is made of, so a run can be read as training from its curves alone:
            #   train_loss/total = train_loss/vanacloig
            #                    + aux_weight x sum of train_loss/<auxiliary source>
            #                    + lambda_host x train_loss/host
            #                    + cgt_lambda x train_loss/graph_penalty
            #   val_loss/total   = val_loss/vanacloig (the fold's validation compounds)
            #                    + lambda_host x val_loss/host (ex21 cells not fitted;
            #                      absent, and the term zero, when every one is fitted)
            #   test_loss/*      = the fold's test compounds, the auxiliary held-out
            #                      conditions, and the 63 ex23 combinations
            # The train terms are the step's batch (W&B smooths them); the validation
            # and test terms are full evaluations at the end of the epoch.
            host_val_loss = (
                float(
                    np.mean(
                        (host_pred[host_val_rows] - host_observed[host_val_rows]) ** 2
                    )
                )
                if len(host_val_rows)
                else float("nan")
            )
            val_total = van_val_loss + (
                cfg.lambda_host * host_val_loss if len(host_val_rows) else 0.0
            )
            record: dict[str, float] = {
                "step": step + 1,
                "epoch": (step + 1) / steps_per_epoch,
                "train_loss/total": float(total.detach()),
                "train_loss/vanacloig": float(losses[vanacloig.spec.name].detach()),
                **{
                    f"train_loss/{name}": float(loss.detach())
                    for name, loss in losses.items()
                    if name != vanacloig.spec.name
                },
                "train_loss/host": float(host_loss.detach()),
                # the graph penalty (KL of the regularized heads' attention to the nine
                # graph priors) is computed on every step and scaled by cgt_lambda
                "train_loss/graph_penalty": (
                    float(sum(penalties).detach()) if penalties else 0.0
                ),
                "val_loss/total": val_total,
                "val_loss/vanacloig": van_val_loss,
                "val_loss/host": host_val_loss,
                "test_loss/vanacloig": van_heldout,
                **{
                    f"test_loss/{name}": heldout_loss(
                        ctx.data.source(name),
                        aux_pred[name],
                        std,
                        ctx.aux_heldout[name],
                    )
                    for name in aux_pred
                },
                "test_loss/host_ex23": float(
                    np.mean(
                        (host_pred[host_test_rows] - host_observed[host_test_rows]) ** 2
                    )
                ),
                "score/val_centered_spearman": val,
                "score/test_centered_spearman": test,
                "opt/grad_norm": float(grad_norm),
                "opt/lr": float(scheduler.get_last_lr()[0]),
                "sys/seconds": time.time() - started,
                "sys/gpu_peak_gb": (
                    torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda"
                    else 0.0
                ),
            }
            history.append(record)
            if wandb.run is not None:
                wandb.log({k: v for k, v in record.items() if np.isfinite(v)})
            if cfg.select == "last" or (np.isfinite(val) and val > best[0]):
                best = (val, step + 1)
                selected = SeedResult(
                    vanacloig=matrix,
                    host=host_pred,
                    aux=aux_pred,
                    history=pd.DataFrame(history),
                )

    assert selected is not None, "no step was selected"
    if cfg.host_train == "finetune_ex21":
        selected = finetune_host(cfg, ctx, fold, model, std, selected, device)
    selected.history = pd.DataFrame(history).assign(seed=seed, selected_step=best[1])
    if cfg.checkpoint:
        # the weights saved are the FINAL ones; under select: val they differ from the
        # selected predictions, and the selected step is recorded beside them
        save_checkpoint(
            osp.join(
                CHECKPOINTS,
                sweep,
                f"{cfg.name}_fold{fold.fold}_fs{cfg.fold_seed}_seed{seed}.pt",
            ),
            model,
            cfg,
            ctx,
            fold,
            seed,
            std,
            best[1],
        )
    return selected


def finetune_host(
    cfg: MixtureConfig,
    ctx: Context,
    fold: Fold,
    model: MixtureModel,
    std: Standardization,
    selected: SeedResult,
    device: torch.device,
) -> SeedResult:
    """Stage 2: a few epochs on the ex21 single-agent cells alone.

    The gene encoder is frozen in both scopes, so the wildtype gene tokens are computed
    once in eval mode and reused. ``host_head`` unfreezes the host readout only;
    ``host_head+env_encoder`` also unfreezes the compound MLP, the FiLM MLP, the compound
    offset and the medium transformer layers, so the compound representation itself can
    move toward the titration curves.
    """
    records = host_training_records(cfg, ctx, fold, "finetune")
    assert records, "the fine-tune stage has no ex21 record"
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = model.host_parameters()
    if cfg.finetune_scope == "host_head+env_encoder":
        trainable += model.env_parameters()
    for parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.finetune_lr, weight_decay=cfg.weight_decay
    )
    model.eval()
    strain = ctx.strain[ctx.data.sources[0].spec.name][:1].to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        h, _, _ = model.cell(strain, ctx.cell_graph)
    h = h.detach()
    observed = torch.tensor(
        [r.fitness for r in records], dtype=torch.float32, device=device
    )
    steps_per_epoch = -(-len(records) // cfg.host_batch)
    rng = np.random.default_rng([cfg.fold_seed, fold.fold, 21])
    for epoch in range(cfg.finetune_epochs):
        order = rng.permutation(len(records))
        for step in range(steps_per_epoch):
            picked = order[step * cfg.host_batch : (step + 1) * cfg.host_batch]
            if not len(picked):
                continue
            batch = [records[i] for i in picked]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                predicted = torch.zeros(len(batch), device=device, dtype=torch.float32)
                for medium in host_media(batch, std):
                    e = model.tokens(medium)
                    gain = model.gain(medium)
                    _, pooled, token = model.medium_state(h, e, None, gain)
                    predicted = predicted.index_copy(
                        0,
                        torch.tensor(medium.rows, device=device),
                        model.host_logits(pooled, token, e, gain, medium.dose).float(),
                    )
            loss = ((predicted - observed[picked]) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 10.0)
            optimizer.step()
        if wandb.run is not None:
            wandb.log(
                {
                    "finetune/train_loss_host": float(loss.detach()),
                    "finetune/epoch": epoch,
                }
            )
    selected.host = predict_host(model, ctx, ctx.data.host, std, device)
    return selected


# ---- evaluation ------------------------------------------------------------ #
class References(BaseModel):
    """The model-free bars on file for ONE growth call.

    ``isobole_mean_excess`` is empty for a call the isobole grids do not carry:
    ``results/isobole_summary.csv`` holds one set of rows, computed on the served call,
    because the Bioscreen software generation times exist only for the ex23 plate set.
    """

    call: str
    loewe_growth_auroc: float
    bliss_fitness_spearman: float
    #: Spearman over ALL 63 combinations with no growth scored 0 (the worst-scenario
    #: ranking), computed here from results/mixture_combinations.csv rather than read
    #: from mixture_scores.csv, which reports its Spearman over the grown subset only
    loewe_worst_scenario_spearman: float
    bliss_worst_scenario_spearman: float
    isobole_mean_excess: dict[str, float]


def references(call: str) -> References:
    """The bars for ``call``, every one read from a committed model-free artifact."""
    scores = pd.read_csv(REFERENCE_SCORES)
    rows = scores[(scores["call"] == call) & (scores["subset"] == "all")]
    assert len(rows), f"{REFERENCE_SCORES} has no {call} / all rows"
    combinations = pd.read_csv(REFERENCE_COMBINATIONS)
    combinations = combinations[
        (combinations["call"] == call) & (combinations["n_compounds"] >= 1)
    ]
    assert len(combinations) == 63, (
        f"{REFERENCE_COMBINATIONS} has {len(combinations)} {call} combinations, not 63"
    )
    worst = {
        rule: float(
            spearmanr(combinations[rule], combinations["observed_zero_mean"])[0]
        )
        for rule in ("loewe_ex21", "bliss_ex23")
    }
    isoboles = pd.read_csv(REFERENCE_ISOBOLES)
    return References(
        call=call,
        loewe_growth_auroc=float(
            rows.loc[rows["rule"] == "loewe_ex21", "auroc"].iloc[0]
        ),
        bliss_fitness_spearman=float(
            rows.loc[rows["rule"] == "bliss_ex23", "spearman"].iloc[0]
        ),
        loewe_worst_scenario_spearman=worst["loewe_ex21"],
        bliss_worst_scenario_spearman=worst["bliss_ex23"],
        isobole_mean_excess=(
            dict(zip(isoboles["run"], isoboles["mean_excess_over_bliss"], strict=True))
            if call == ISOBOLE_REFERENCE_CALL
            else {}
        ),
    )


def gene_scores(
    cfg: MixtureConfig, ctx: Context, matrix: NDArray[np.float64], fold: Fold
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(the per-compound table, the long-form summary rows) for the gene-level task."""
    vanacloig = ctx.data.sources[0]
    if cfg.split == "gene":
        per_compound = score_genes(vanacloig, matrix, fold.test)
        subset = "test_genes"
    else:
        pool = centering_compounds(cfg, fold)
        per_compound = score_compounds(vanacloig, matrix, pool, fold.test)
        subset = "test_compounds"
    rows = []
    for target, group in per_compound.groupby("target"):
        rows.append(
            {
                "task": "vanacloig",
                "subset": subset,
                "metric": f"{target}_spearman_median",
                "value": float(group["spearman"].median()),
                "n": int(group["spearman"].notna().sum()),
                "reference_rule": (
                    "nested ridge FCFP4 counts, 32 compounds"
                    if target == "centered" and cfg.split == "compound"
                    else ""
                ),
                "reference_value": (
                    RIDGE_VANACLOIG_MEDIAN
                    if target == "centered" and cfg.split == "compound"
                    else float("nan")
                ),
                "call": CALL_INDEPENDENT,
            }
        )
    return per_compound, pd.DataFrame(rows)


def aux_scores(
    cfg: MixtureConfig, ctx: Context, aux: dict[str, NDArray[np.float64]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per held-out condition table, summary rows) for every auxiliary source.

    Scored exactly as the Vanacloig task is: centered Spearman per held-out condition,
    the prediction centered by its own mean over the centering subset of fitted
    conditions and the measurement by the measured mean over the same subset. The hold-out
    is the same across folds (``holdout_conditions``), so the fold column is a repeat
    measurement of one partition, not a cross-validation.
    """
    tables, rows = [], []
    for name, matrix in aux.items():
        source = ctx.data.source(name)
        table = score_compounds(
            source, matrix, ctx.aux_center[name], ctx.aux_heldout[name]
        )
        table["condition"] = [
            f"{source.compounds[j]}|{source.log10_molar[j]:.3f}"
            for j in ctx.aux_heldout[name]
            for _ in range(2)
        ]
        tables.append(table.assign(source=name))
        for target, group in table.groupby("target"):
            rows.append(
                {
                    "task": name,
                    "subset": f"heldout_{cfg.aux_holdout}",
                    "metric": f"{target}_spearman_median",
                    "value": float(group["spearman"].median()),
                    "n": int(group["spearman"].notna().sum()),
                    "reference_rule": "",
                    "reference_value": float("nan"),
                    "call": CALL_INDEPENDENT,
                }
            )
    if not tables:
        return pd.DataFrame(), pd.DataFrame(rows)
    return pd.concat(tables, ignore_index=True), pd.DataFrame(rows)


def host_scores(
    cfg: MixtureConfig,
    ctx: Context,
    predicted: NDArray[np.float64],
    bars: dict[str, References],
) -> pd.DataFrame:
    """Long-form wet-lab scores: ex21 curves, ex23 combinations, the isobole grids.

    A PREDICTION IS CALL-INDEPENDENT (the model is handed a medium, not a growth call),
    so ``predicted`` is scored once per call against that call's own grown set, tau and
    fitness scale, and every row carries the ``call`` it belongs to.

    The ex21 rows carry ``CALL_INDEPENDENT``: ``fitness_software`` is empty for all 180
    ex21 wells, so there is one titration readout, not two. The isobole rows are emitted
    only for the calls the grids carry, which is the served call alone, for the same
    reason (all 600 isobole wells have an empty software column). Scoring them under the
    software call would mean reading an empty growth flag as grown.
    """
    host = ctx.data.host
    trained_ex21 = cfg.host_train in ("anchors+ex21", "finetune_ex21")
    position = {record.key: i for i, record in enumerate(host)}
    rows = []

    ex21 = [(i, r) for i, r in enumerate(host) if r.run == "ex21"]
    wells_pred = np.concatenate([[predicted[i]] * r.n_wells for i, r in ex21])
    wells_obs = np.concatenate([r.well_fitness for _, r in ex21])
    suffix = "_train_fit" if trained_ex21 else ""
    rows.append(
        {
            "task": "ex21",
            "subset": "all_wells",
            "metric": f"fitness_spearman{suffix}",
            "value": fast_spearman(wells_pred, wells_obs),
            "n": int(len(wells_obs)),
            "call": CALL_INDEPENDENT,
            "reference_rule": "",
            "reference_value": float("nan"),
        }
    )
    for compound in sorted({r.compounds[0] for _, r in ex21 if r.compounds}):
        subset = [(i, r) for i, r in ex21 if r.compounds == [compound]]
        pred = np.concatenate([[predicted[i]] * r.n_wells for i, r in subset])
        obs = np.concatenate([r.well_fitness for _, r in subset])
        rows.append(
            {
                "task": "ex21",
                "subset": compound,
                "metric": f"fitness_spearman{suffix}",
                "value": fast_spearman(pred, obs),
                "n": int(len(obs)),
                "call": CALL_INDEPENDENT,
                "reference_rule": "",
                "reference_value": float("nan"),
            }
        )

    for call, bar in sorted(bars.items()):
        view = ctx.data.host_by_call[call]
        if "ex23" in ctx.data.runs_by_call[call]:
            ex23 = [
                (position[key], record)
                for key, record in view.items()
                if record.run == "ex23" and record.compounds
            ]
            assert len(ex23) == 63, f"{call}: {len(ex23)} ex23 combinations, not 63"
            grew = np.array([r.grew for _, r in ex23])
            pred = np.array([predicted[i] for i, _ in ex23])
            rows.append(
                {
                    "task": "ex23",
                    "subset": "combinations",
                    "metric": "growth_auroc",
                    "value": float(roc_auc_score(grew, pred))
                    if grew.any() and not grew.all()
                    else float("nan"),
                    "n": int(len(ex23)),
                    "call": call,
                    "reference_rule": "loewe_ex21",
                    "reference_value": bar.loewe_growth_auroc,
                }
            )
            # the ranking the detoxification-target question asks for: every combination
            # scored, a scenario that did not grow at its observed fitness of zero
            rows.append(
                {
                    "task": "ex23",
                    "subset": "combinations",
                    "metric": "worst_scenario_spearman",
                    "value": fast_spearman(
                        pred, np.array([r.fitness for _, r in ex23])
                    ),
                    "n": int(len(ex23)),
                    "call": call,
                    "reference_rule": "loewe_ex21 (all 63, no growth 0)",
                    "reference_value": bar.loewe_worst_scenario_spearman,
                }
            )
            grown = [(i, r) for i, r in ex23 if r.grew]
            rows.append(
                {
                    "task": "ex23",
                    "subset": "combinations_that_grew",
                    "metric": "fitness_spearman",
                    "value": (
                        float(
                            spearmanr(
                                [predicted[i] for i, _ in grown],
                                [r.fitness_grown for _, r in grown],
                            )[0]
                        )
                        if len(grown) >= 3
                        else float("nan")
                    ),
                    "n": int(len(grown)),
                    "call": call,
                    "reference_rule": "bliss_ex23",
                    "reference_value": bar.bliss_fitness_spearman,
                }
            )
        for run in sorted(ISOBOLE_RUNS):
            if run not in ctx.data.runs_by_call[call]:
                continue
            interior = [
                (position[key], record)
                for key, record in view.items()
                if record.run == run and len(record.compounds) == 2
            ]
            assert len(interior) == 81, f"{call} {run}: {len(interior)} interior cells"
            excess = np.array([r.fitness - predicted[i] for i, r in interior])
            rows.append(
                {
                    "task": f"isobole_{run}",
                    "subset": "interior_cells",
                    "metric": "mean_observed_minus_predicted",
                    "value": float(excess.mean()),
                    "n": int(len(interior)),
                    "call": call,
                    "reference_rule": "bliss_ex21",
                    "reference_value": bar.isobole_mean_excess[run],
                }
            )
    return pd.DataFrame(rows)


def print_table(name: str, scores: pd.DataFrame) -> None:
    """The run's summary beside the model-free bar, so the log says whether it clears it."""
    print(f"\n{name}", flush=True)
    header = (
        f"  {'task':<14}{'subset':<26}{'metric':<30}{'call':<9}"
        f"{'value':>9}{'n':>7}  reference"
    )
    print(header, flush=True)
    for _, row in scores.iterrows():
        bar = (
            ""
            if not row["reference_rule"]
            else f"{row['reference_rule']} {row['reference_value']:.3f}"
        )
        print(
            f"  {row['task']:<14}{str(row['subset'])[:25]:<26}{row['metric']:<30}"
            f"{row['call']:<9}{row['value']:>9.3f}{int(row['n']):>7}  {bar}",
            flush=True,
        )


def run(cfg: MixtureConfig, sweep: str, device: torch.device) -> pd.DataFrame:
    ctx = build_context(cfg, device)
    # one set of bars per growth call the wells carry, so a wet-lab score is read against
    # the references of its own call
    bars = {call: references(call) for call in ctx.data.runs_by_call}
    out_dir = osp.join(RESULTS, "mixture", sweep)
    os.makedirs(out_dir, exist_ok=True)
    pred_dir = osp.join(PREDICTIONS, sweep)
    os.makedirs(pred_dir, exist_ok=True)
    ctx.data.counts.assign(name=cfg.name).to_csv(
        osp.join(out_dir, f"{cfg.name}_counts.csv"), index=False
    )
    # ONE W&B RUN PER TRAJECTORY (fold, seed), grouped by arm, keys shared by every
    # run and read against the epoch; the fold ensemble's scores are written to each
    # of its seed runs under ``ensemble/`` once the fold is done
    arm = re.sub(r"_s\d+$", "", re.sub(r"_f\d+$", "", cfg.name))
    wandb_config = cfg.model_dump() | {
        "arm": arm,
        "sweep": sweep,
        "round": sweep.split("_")[0],
        "readout_": cfg.readout,
        "compound_pool_": cfg.compound_pool,
        "host_train_": cfg.host_train,
        "dose_": cfg.dose,
        "fold_seed_": cfg.fold_seed,
        "machine": socket.gethostname().split(".")[0],
        "n_aux_dropped": ctx.n_aux_dropped,
        "dropped_compounds": ctx.data.dropped_compounds,
    }
    wandb_dir = osp.join(DATA_ROOT, "wandb-experiments", "040-inhibitor-synergy-wetlab")

    def open_run(fold: int, seed: int) -> wandb.sdk.wandb_run.Run:
        run = wandb.init(
            mode=WANDB_MODE,
            project=cfg.wandb_project,
            group=arm,
            name=f"{cfg.name}_seed{seed}",
            job_type="train",
            tags=["mixture", sweep, f"readout_{cfg.readout}"],
            config=wandb_config | {"fold": fold, "seed": seed},
            dir=wandb_dir,
            reinit=True,
        )
        run.define_metric("epoch")
        run.define_metric("*", step_metric="epoch")
        run.define_metric("finetune/epoch")
        run.define_metric("finetune/*", step_metric="finetune/epoch")
        return run

    vanacloig = ctx.data.sources[0]
    if cfg.split == "gene":
        folds = make_gene_folds(len(vanacloig.genes), cfg.n_folds, cfg.fold_seed)
    else:
        folds = make_folds(
            len(vanacloig.compounds), cfg.n_folds, cfg.n_val, cfg.fold_seed
        )
    frames, compounds, histories, aux_tables = [], [], [], []
    all_runs: list[str] = []
    for fold in folds:
        if cfg.folds is not None and fold.fold not in cfg.folds:
            continue
        results, seed_runs = [], []
        for seed in cfg.seeds:
            run_wandb = open_run(fold.fold, seed)
            result = train_seed(cfg, ctx, fold, seed, device, sweep)
            results.append(result)
            histories.append(result.history.assign(fold=fold.fold))
            table, summary = gene_scores(cfg, ctx, result.vanacloig, fold)
            compounds.append(
                table.assign(fold=fold.fold, member=f"seed{seed}", name=cfg.name)
            )
            aux_table, aux_summary = aux_scores(cfg, ctx, result.aux)
            aux_tables.append(aux_table.assign(fold=fold.fold, member=f"seed{seed}"))
            seed_scores = pd.concat(
                [summary, aux_summary, host_scores(cfg, ctx, result.host, bars)]
            ).assign(fold=fold.fold, member=f"seed{seed}")
            frames.append(seed_scores)
            run_wandb.summary.update(
                {
                    f"{row['task']}/{row['subset']}/{row['metric']}/{row['call']}": row[
                        "value"
                    ]
                    for _, row in seed_scores.iterrows()
                }
                | {"selected_step": int(result.history["selected_step"].iloc[0])}
            )
            seed_runs.append(run_wandb.path)
            run_wandb.finish()
            print(
                f"{cfg.name} fold {fold.fold} seed {seed}: "
                f"{result.history['sys/seconds'].iloc[-1]:.0f} s",
                flush=True,
            )
        matrix = np.mean([r.vanacloig for r in results], axis=0)
        host = np.mean([r.host for r in results], axis=0)
        np.save(
            osp.join(pred_dir, f"{cfg.name}_fold{fold.fold}_seed{cfg.fold_seed}.npy"),
            matrix.astype(np.float32),
        )
        np.save(
            osp.join(
                pred_dir, f"{cfg.name}_fold{fold.fold}_seed{cfg.fold_seed}_host.npy"
            ),
            host.astype(np.float32),
        )
        table, summary = gene_scores(cfg, ctx, matrix, fold)
        compounds.append(table.assign(fold=fold.fold, member="ensemble", name=cfg.name))
        aux_mean = {
            name: np.mean([r.aux[name] for r in results], axis=0)
            for name in results[0].aux
        }
        aux_table, aux_summary = aux_scores(cfg, ctx, aux_mean)
        aux_tables.append(aux_table.assign(fold=fold.fold, member="ensemble"))
        ensemble_scores = pd.concat(
            [summary, aux_summary, host_scores(cfg, ctx, host, bars)]
        ).assign(fold=fold.fold, member="ensemble")
        frames.append(ensemble_scores)
        write_summary(seed_runs, "ensemble", ensemble_scores)
        all_runs.extend(seed_runs)

    scores = pd.concat(frames, ignore_index=True).assign(
        name=cfg.name, fold_seed=cfg.fold_seed, training_call=cfg.call
    )
    scores["beats_reference"] = np.where(
        scores["reference_rule"] == "",
        "",
        np.where(
            scores["metric"].str.startswith("mean_observed"),
            np.where(
                scores["value"].abs() < scores["reference_value"].abs(),
                "closer",
                "further",
            ),
            np.where(scores["value"] > scores["reference_value"], "above", "below"),
        ),
    )
    scores.to_csv(osp.join(out_dir, f"{cfg.name}_scores.csv"), index=False)
    pd.concat(compounds).to_csv(
        osp.join(out_dir, f"{cfg.name}_vanacloig_compounds.csv"), index=False
    )
    pd.concat(histories).to_csv(
        osp.join(out_dir, f"{cfg.name}_history.csv"), index=False
    )
    aux_all = pd.concat(aux_tables, ignore_index=True)
    if len(aux_all):
        aux_all.assign(name=cfg.name).to_csv(
            osp.join(out_dir, f"{cfg.name}_aux_conditions.csv"), index=False
        )
    ensemble = scores[scores["member"] == "ensemble"]
    summary = (
        ensemble.groupby(["task", "subset", "metric", "call"], as_index=False)
        .agg(
            value=("value", "mean"),
            # n is the size of ONE evaluation (the same wells are scored on every fold),
            # and n_folds says how many fold models the value is the mean of
            n=("n", "max"),
            n_folds=("value", "size"),
            reference_rule=("reference_rule", "first"),
            reference_value=("reference_value", "first"),
        )
        .sort_values(["task", "subset", "metric", "call"], ignore_index=True)
    )
    print_table(f"{cfg.name} (ensemble, mean over folds)", summary)
    write_summary(all_runs, "mean_over_folds", summary)
    return scores


def write_summary(run_paths: list[str], prefix: str, scores: pd.DataFrame) -> None:
    """Write ``scores`` (long form) to each finished run's W&B summary under ``prefix``.

    The seed runs are finished by the time their fold's ensemble and the config's mean
    over folds exist, so the values go through the API. Nothing is written when W&B
    is disabled or offline.
    """
    if not run_paths or WANDB_MODE in ("disabled", "offline"):
        return
    values = {
        f"{prefix}/{row['task']}/{row['subset']}/{row['metric']}/{row['call']}": float(
            row["value"]
        )
        for _, row in scores.iterrows()
    }
    # the public API's summary.update persists the keys it is given; run.update()
    # would push the whole summary the client fetched, which can predate the run's
    # own final upload and drop keys (seen on a seed finished seconds earlier)
    api = wandb.Api()
    for path in run_paths:
        api.run(path).summary.update(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", help="YAML list of configs")
    parser.add_argument("--only", nargs="*", default=None, help="config names to run")
    args = parser.parse_args()
    assert args.sweep, "--sweep is required"
    sweep = osp.splitext(osp.basename(args.sweep))[0]
    with open(args.sweep) as f:
        configs = [MixtureConfig(**c) for c in yaml.safe_load(f)]
    # The manifest names the store before a card is spent; the bytes are verified when
    # the store is read (mixture_data.load_cells).
    manifest = read_manifest(STORE)
    print(
        f"store {STORE}: KG {manifest.kg_release} ({manifest.kg_version}), "
        f"{manifest.n_processed_entries:,} records, data.mdb sha256 "
        f"{manifest.data_mdb_sha256[:12]}",
        flush=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for cfg in configs:
        if args.only and cfg.name not in args.only:
            continue
        done = osp.join(RESULTS, "mixture", sweep, f"{cfg.name}_scores.csv")
        if osp.exists(done):
            print(f"{cfg.name}: already scored, skipped", flush=True)
            continue
        run(cfg, sweep, device)


if __name__ == "__main__":
    main()
