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
                  and a shift on each compound token (FiLM). ``off``: 038's dose-free
                  token.
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
(c) ex23: growth / no-growth AUROC over the 63 combinations with the predicted fitness as
    the score, and fitness Spearman over the combinations that grew. References: Loewe
    from the ex21 Hill fits, AUROC 0.958, and Bliss from the observed ex23 singles,
    Spearman 0.573 (served call, ``results/mixture_scores.csv``).
(d) isoboles: the mean of observed minus predicted over the 81 interior cells of each
    grid, beside the same quantity for Bliss (``results/isobole_summary.csv``).

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
    CELL_TABLE,
    ISOBOLE_RUNS,
    Fold,
    GeneSource,
    HostRecord,
    MixtureData,
    assemble,
    build_cell_graph,
    make_folds,
    score_compounds,
    strain_indices,
    subsample_pool,
    write_vanacloig_doses,
)

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
DOSES_CSV = osp.join(RESULTS, "vanacloig_doses.csv")
MODEL_YAML = osp.join(EXPERIMENT, "conf", "mixture", "cgt_model.yaml")
PREDICTIONS = osp.join(
    DATA_ROOT, "experiments", "040-inhibitor-synergy-wetlab", "predictions"
)
#: The model-free bars every run's table is printed against (``results/mixture_scores.csv``
#: for ex23, ``results/isobole_summary.csv`` for the grids), served growth call.
REFERENCE_SCORES = osp.join(RESULTS, "mixture_scores.csv")
REFERENCE_ISOBOLES = osp.join(RESULTS, "isobole_summary.csv")
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
    require_molar_dose: bool = True
    call: Literal["served", "software"] = "served"
    embeddings: list[str] = ["fcfp4_count"]
    pca_dim: int | None = None
    # ---- the ablation ladder
    compound_set: Literal["tokens", "mean"] = "tokens"
    dose: Literal["film", "off"] = "film"
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
    if cfg.fit_on == "train":
        assert cfg.n_fit_compounds is None, "n_fit_compounds subsamples the pool"
        return fold.train
    return subsample_pool(fold, cfg.fold_seed, cfg.n_fit_compounds)


def centering_compounds(cfg: MixtureConfig, fold: Fold) -> list[int]:
    """The compounds whose mean centers the held-out scores (038)."""
    if cfg.fit_on == "pool":
        return fitted_compounds(cfg, fold)
    return sorted(fold.train + fold.val)


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


class MixtureModel(nn.Module):
    """The 038 environment encoder with a compound set, a dose, and a host readout."""

    def __init__(
        self,
        cfg: MixtureConfig,
        n_nodes: int,
        n_sources: int,
        feature_dim: int,
        encoder,
    ) -> None:
        """``n_nodes`` is the cell graph's gene count; the gene bias is per source."""
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.n_nodes = n_nodes
        d = cfg.cgt_dim
        self.gene_bias = nn.Embedding(n_sources * n_nodes, 1)
        nn.init.zeros_(self.gene_bias.weight)
        self.source_token = nn.Embedding(n_sources, d)
        nn.init.normal_(self.source_token.weight, std=0.02)
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
        self.offset = nn.Linear(d, 1)
        self.layers = nn.ModuleList(
            TokenLayer(d, cfg.cgt_heads, cfg.dropout) for _ in range(cfg.env_layers)
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
        modules = [self.compound, self.film, self.layers, self.offset]
        return [p for module in modules for p in module.parameters()]

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

    def medium_state(
        self, h: torch.Tensor, e: torch.Tensor, strain: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Run the medium's tokens with the gene tokens; reduce inside the chunk loop.

        Returns (rows of ``H^c`` at each strain's deleted genes [B, C, d] or None when no
        strain is read, the mean of ``H^c`` over genes [C, d], the mean of the compound
        tokens after they read the genome [C, d]).
        """
        n, d = h.shape
        c, k = e.shape[0], e.shape[1]
        env, pooled, token = [], [], []
        for start in range(0, c, self.cfg.env_chunk or c):
            part = e[start : start + (self.cfg.env_chunk or c)]
            genes = h.unsqueeze(0).expand(len(part), n, d)
            seq = torch.cat([part, genes], dim=1) if k else genes
            for layer in self.layers:
                seq = layer(seq)
            cell = seq[:, k:]
            if strain is not None:
                env.append(cell[:, strain].sum(2))
            pooled.append(cell.mean(1))
            token.append(
                seq[:, :k].mean(1)
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
        env: torch.Tensor,
        h_del: torch.Tensor,
        pooled: torch.Tensor,
        token: torch.Tensor,
        strain: torch.Tensor,
        source_index: int,
        e: torch.Tensor,
    ) -> torch.Tensor:
        """[B, C] predictions on the standardized target for the source's conditions."""
        b, c, d = env.shape
        source = self.source_token(
            torch.full((1,), source_index, dtype=torch.long, device=env.device)
        )
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
        gene = self.gene_bias(source_index * self.n_nodes + strain[:, 0])
        base = gene + self.offset(e.mean(1)).T
        return base + self.gene_head(self.gene_norm(state.float())).squeeze(-1)

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
    #: auxiliary conditions kept after removing the Vanacloig compounds
    aux_conditions: dict[str, list[int]]
    n_aux_dropped: dict[str, int]


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
    aux, dropped = {}, {}
    for source in data.sources[1:]:
        keep = [
            j
            for j in range(len(source.compounds))
            if int(source.compound_row[j]) not in vanacloig_rows
            and (cfg.dose == "off" or np.isfinite(source.log10_molar[j]))
        ]
        aux[source.spec.name] = keep
        dropped[source.spec.name] = len(source.compounds) - len(keep)
        assert keep, f"{source.spec.name} has no condition left after filtering"
    return Context(
        data=data,
        cell_graph=cell_graph.to(device),
        node_ids=node_ids,
        strain=strain,
        aux_conditions=aux,
        n_aux_dropped=dropped,
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
    held = {int(vanacloig.compound_row[j]) for j in fold.test}
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
) -> Standardization:
    """Every scale is computed on the fitted conditions only (the 031 rule)."""
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
        y_mean[source.spec.name] = float(np.nanmean(block))
        y_sd[source.spec.name] = float(np.nanstd(block))
    return Standardization(
        x=torch.tensor(x_std, dtype=torch.float32, device=device),
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
                env, pooled, token = model.medium_state(h, e, strain)
                assert env is not None
                pred = model.gene_logits(
                    env, h_del, pooled, token, strain, source.index, e
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
            _, pooled, token = model.medium_state(h, e, None)
            out[np.array(medium.rows)] = (
                model.host_fitness(pooled, token).float().cpu().numpy()
            )
    return out


# ---- training -------------------------------------------------------------- #
class SeedResult(BaseModel):
    """One seed's selected prediction on both tasks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vanacloig: NDArray[np.float64]  # [n_genes, n_conditions], served units
    host: NDArray[np.float64]  # [len(ctx.data.host)] predicted fitness
    history: pd.DataFrame


def train_seed(
    cfg: MixtureConfig, ctx: Context, fold: Fold, seed: int, device: torch.device
) -> SeedResult:
    """Fit one seed on this fold and return its selected predictions on both tasks."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    sources = ctx.data.sources
    vanacloig = sources[0]
    host = host_training_records(cfg, ctx, fold, "main")
    assert host, "the host task has no training record"
    std = standardize(cfg, ctx, fold, host, device)
    conditions = {
        vanacloig.spec.name: fitted_compounds(cfg, fold),
        **ctx.aux_conditions,
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
    ).to(device)
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
    steps_per_epoch = -(-len(vanacloig.genes) // cfg.gene_batch)
    steps, eval_every = cfg.steps, cfg.eval_every
    if cfg.epochs is not None:
        steps = cfg.epochs * steps_per_epoch
        eval_every = cfg.eval_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.05
    )
    generator = torch.Generator().manual_seed(seed)
    order = {
        s.spec.name: torch.randperm(len(s.genes), generator=generator) for s in sources
    }
    cursor = dict.fromkeys(order, 0)
    rng = np.random.default_rng(seed)
    host_fitness = torch.tensor(
        [r.fitness for r in host], dtype=torch.float32, device=device
    )
    pool = centering_compounds(cfg, fold)
    tag = f"fold{fold.fold}_seed{seed}"
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
            if cursor[name] + cfg.gene_batch > len(source.genes):
                order[name] = torch.randperm(len(source.genes), generator=generator)
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
                    env, pooled, token = model.medium_state(h, e, strain)
                    assert env is not None
                    predictions = predictions.index_copy(
                        1,
                        torch.tensor(medium.rows, device=device),
                        model.gene_logits(
                            env, h_del, pooled, token, strain, source.index, e
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
                _, pooled, token = model.medium_state(h_wildtype, e, None)
                predicted = predicted.index_copy(
                    0,
                    torch.tensor(medium.rows, device=device),
                    model.host_fitness(pooled, token).float(),
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
            val = centered_val_score(
                vanacloig.y, matrix, conditions[vanacloig.spec.name], fold.val
            )
            test = centered_val_score(vanacloig.y, matrix, pool, fold.test)
            record = {
                "step": step + 1,
                "epoch": (step + 1) / steps_per_epoch,
                "gene_loss": float(gene_loss.detach()),
                "host_loss": float(host_loss.detach()),
                "grad_norm": float(grad_norm),
                "lr": float(scheduler.get_last_lr()[0]),
                "val_centered_mean": val,
                "test_centered_mean": test,
                "seconds": time.time() - started,
                "gpu_peak_gb": (
                    torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda"
                    else 0.0
                ),
            }
            record |= {
                f"loss_{name}": float(loss.detach()) for name, loss in losses.items()
            }
            history.append(record)
            if wandb.run is not None:
                wandb.log(
                    {f"{tag}/{k}": v for k, v in record.items() if k != "step"}
                    | {"step": step + 1}
                )
            if cfg.select == "last" or (np.isfinite(val) and val > best[0]):
                best = (val, step + 1)
                selected = SeedResult(
                    vanacloig=matrix, host=host_pred, history=pd.DataFrame(history)
                )

    assert selected is not None, "no step was selected"
    if cfg.host_train == "finetune_ex21":
        selected = finetune_host(cfg, ctx, fold, model, std, selected, device)
    selected.history = pd.DataFrame(history).assign(seed=seed, selected_step=best[1])
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
    trainable = list(model.host_head.parameters()) + list(model.host_norm.parameters())
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
                    _, pooled, token = model.medium_state(h, e, None)
                    predicted = predicted.index_copy(
                        0,
                        torch.tensor(medium.rows, device=device),
                        model.host_fitness(pooled, token).float(),
                    )
            loss = ((predicted - observed[picked]) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 10.0)
            optimizer.step()
        if wandb.run is not None:
            wandb.log(
                {f"finetune/fold{fold.fold}_loss": float(loss.detach()), "step": epoch}
            )
    selected.host = predict_host(model, ctx, ctx.data.host, std, device)
    return selected


# ---- evaluation ------------------------------------------------------------ #
class References(BaseModel):
    """The model-free bars on file, under the run's growth call."""

    loewe_growth_auroc: float
    bliss_fitness_spearman: float
    isobole_mean_excess: dict[str, float]


def references(call: str) -> References:
    scores = pd.read_csv(REFERENCE_SCORES)
    served = scores[(scores["call"] == call) & (scores["subset"] == "all")]
    assert len(served), f"{REFERENCE_SCORES} has no {call} / all rows"
    isoboles = pd.read_csv(REFERENCE_ISOBOLES)
    return References(
        loewe_growth_auroc=float(
            served.loc[served["rule"] == "loewe_ex21", "auroc"].iloc[0]
        ),
        bliss_fitness_spearman=float(
            served.loc[served["rule"] == "bliss_ex23", "spearman"].iloc[0]
        ),
        isobole_mean_excess=dict(
            zip(isoboles["run"], isoboles["mean_excess_over_bliss"], strict=True)
        ),
    )


def gene_scores(
    cfg: MixtureConfig, ctx: Context, matrix: NDArray[np.float64], fold: Fold
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(the per-compound table, the long-form summary rows) for the gene-level task."""
    vanacloig = ctx.data.sources[0]
    pool = centering_compounds(cfg, fold)
    per_compound = score_compounds(vanacloig, matrix, pool, fold.test)
    rows = []
    for target, group in per_compound.groupby("target"):
        rows.append(
            {
                "task": "vanacloig",
                "subset": "test_compounds",
                "metric": f"{target}_spearman_median",
                "value": float(group["spearman"].median()),
                "n": int(group["spearman"].notna().sum()),
                "reference_rule": (
                    "nested ridge FCFP4 counts, 32 compounds"
                    if target == "centered"
                    else ""
                ),
                "reference_value": (
                    RIDGE_VANACLOIG_MEDIAN if target == "centered" else float("nan")
                ),
            }
        )
    return per_compound, pd.DataFrame(rows)


def host_scores(
    cfg: MixtureConfig, ctx: Context, predicted: NDArray[np.float64], bars: References
) -> pd.DataFrame:
    """Long-form wet-lab scores: ex21 curves, ex23 combinations, the isobole grids."""
    host = ctx.data.host
    trained_ex21 = cfg.host_train in ("anchors+ex21", "finetune_ex21")
    rows = []

    ex21 = [(i, r) for i, r in enumerate(host) if r.run == "ex21"]
    wells_pred = np.concatenate([[predicted[i]] * r.n_wells for i, r in ex21])
    wells_obs = np.concatenate([r.well_fitness for _, r in ex21])
    rows.append(
        {
            "task": "ex21",
            "subset": "all_wells",
            "metric": "fitness_spearman" + ("_train_fit" if trained_ex21 else ""),
            "value": fast_spearman(wells_pred, wells_obs),
            "n": int(len(wells_obs)),
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
                "metric": "fitness_spearman" + ("_train_fit" if trained_ex21 else ""),
                "value": fast_spearman(pred, obs),
                "n": int(len(obs)),
                "reference_rule": "",
                "reference_value": float("nan"),
            }
        )

    ex23 = [(i, r) for i, r in enumerate(host) if r.run == "ex23" and r.compounds]
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
            "reference_rule": "loewe_ex21",
            "reference_value": bars.loewe_growth_auroc,
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
            "reference_rule": "bliss_ex23",
            "reference_value": bars.bliss_fitness_spearman,
        }
    )

    for run in sorted(ISOBOLE_RUNS):
        interior = [
            (i, r) for i, r in enumerate(host) if r.run == run and len(r.compounds) == 2
        ]
        excess = np.array([r.fitness - predicted[i] for i, r in interior])
        rows.append(
            {
                "task": f"isobole_{run}",
                "subset": "interior_cells",
                "metric": "mean_observed_minus_predicted",
                "value": float(excess.mean()),
                "n": int(len(interior)),
                "reference_rule": "bliss_ex21",
                "reference_value": bars.isobole_mean_excess[run],
            }
        )
    return pd.DataFrame(rows)


def print_table(name: str, scores: pd.DataFrame) -> None:
    """The run's summary beside the model-free bar, so the log says whether it clears it."""
    print(f"\n{name}", flush=True)
    header = (
        f"  {'task':<14}{'subset':<26}{'metric':<36}{'value':>9}{'n':>7}  reference"
    )
    print(header, flush=True)
    for _, row in scores.iterrows():
        bar = (
            ""
            if not row["reference_rule"]
            else f"{row['reference_rule']} {row['reference_value']:.3f}"
        )
        print(
            f"  {row['task']:<14}{str(row['subset'])[:25]:<26}{row['metric']:<36}"
            f"{row['value']:>9.3f}{int(row['n']):>7}  {bar}",
            flush=True,
        )


def run(cfg: MixtureConfig, sweep: str, device: torch.device) -> pd.DataFrame:
    ctx = build_context(cfg, device)
    bars = references(cfg.call)
    out_dir = osp.join(RESULTS, "mixture", sweep)
    os.makedirs(out_dir, exist_ok=True)
    pred_dir = osp.join(PREDICTIONS, sweep)
    os.makedirs(pred_dir, exist_ok=True)
    ctx.data.counts.assign(name=cfg.name).to_csv(
        osp.join(out_dir, f"{cfg.name}_counts.csv"), index=False
    )
    run_wandb = wandb.init(
        mode=WANDB_MODE,
        project="torchcell_040-inhibitor-synergy-wetlab",
        group=f"{sweep}/{cfg.name}",
        name=cfg.name,
        tags=["mixture", sweep],
        config=cfg.model_dump()
        | {
            "n_aux_dropped": ctx.n_aux_dropped,
            "dropped_compounds": ctx.data.dropped_compounds,
        },
        dir=osp.join(DATA_ROOT, "wandb-experiments", "040-inhibitor-synergy-wetlab"),
        reinit=True,
    )
    run_wandb.define_metric("step")
    run_wandb.define_metric("*", step_metric="step")

    vanacloig = ctx.data.sources[0]
    folds = make_folds(len(vanacloig.compounds), cfg.n_folds, cfg.n_val, cfg.fold_seed)
    frames, compounds, histories = [], [], []
    for fold in folds:
        if cfg.folds is not None and fold.fold not in cfg.folds:
            continue
        results = []
        for seed in cfg.seeds:
            result = train_seed(cfg, ctx, fold, seed, device)
            results.append(result)
            histories.append(result.history.assign(fold=fold.fold))
            table, summary = gene_scores(cfg, ctx, result.vanacloig, fold)
            compounds.append(
                table.assign(fold=fold.fold, member=f"seed{seed}", name=cfg.name)
            )
            frames.append(
                pd.concat([summary, host_scores(cfg, ctx, result.host, bars)]).assign(
                    fold=fold.fold, member=f"seed{seed}"
                )
            )
            print(
                f"{cfg.name} fold {fold.fold} seed {seed}: "
                f"{result.history['seconds'].iloc[-1]:.0f} s",
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
        frames.append(
            pd.concat([summary, host_scores(cfg, ctx, host, bars)]).assign(
                fold=fold.fold, member="ensemble"
            )
        )

    scores = pd.concat(frames, ignore_index=True).assign(
        name=cfg.name, fold_seed=cfg.fold_seed, call=cfg.call
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
    ensemble = scores[scores["member"] == "ensemble"]
    summary = (
        ensemble.groupby(["task", "subset", "metric"], as_index=False)
        .agg(
            value=("value", "mean"),
            # n is the size of ONE evaluation (the same wells are scored on every fold),
            # and n_folds says how many fold models the value is the mean of
            n=("n", "max"),
            n_folds=("value", "size"),
            reference_rule=("reference_rule", "first"),
            reference_value=("reference_value", "first"),
        )
        .sort_values(["task", "subset"], ignore_index=True)
    )
    print_table(f"{cfg.name} (ensemble, mean over folds)", summary)
    run_wandb.summary.update(
        {
            f"{row['task']}/{row['subset']}/{row['metric']}": row["value"]
            for _, row in summary.iterrows()
        }
    )
    run_wandb.finish()
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", help="YAML list of configs")
    parser.add_argument("--only", nargs="*", default=None, help="config names to run")
    parser.add_argument(
        "--doses-only",
        action="store_true",
        help="write results/vanacloig_doses.csv from the dev store and stop",
    )
    args = parser.parse_args()
    assert args.sweep or args.doses_only, "--sweep is required unless --doses-only"
    if args.doses_only:
        frame = write_vanacloig_doses(DOSES_CSV)
        print(frame.to_string(), flush=True)
        print(f"wrote {DOSES_CSV}", flush=True)
        return
    sweep = osp.splitext(osp.basename(args.sweep))[0]
    with open(args.sweep) as f:
        configs = [MixtureConfig(**c) for c in yaml.safe_load(f)]
    assert osp.exists(CELL_TABLE), f"no cell table at {CELL_TABLE}"
    if not osp.exists(DOSES_CSV):
        write_vanacloig_doses(DOSES_CSV)
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
