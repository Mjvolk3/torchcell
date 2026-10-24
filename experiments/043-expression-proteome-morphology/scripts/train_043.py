# experiments/043-expression-proteome-morphology/scripts/train_043.py
# [[experiments.043-expression-proteome-morphology.scripts.train_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/train_043.py
"""Train the 043 cell model on the joint store, or on one dataset alone.

Two stages:

``--stage prepare`` (CPU, one slurm job; no vectors needed)
    Reads every record of the arm datasets from the raw 043 store through
    ``Raw043Dataset`` with the full input representation (route A by default), assigns
    each record its role under ``conf/splits/global_split_v1.json`` (fold 0 test, fold 1
    validation, Sameith 2015 Dm split B), DROPS the test records so their labels are
    never loaded, applies the target transform of the primer's table
    ``tab:output-scales`` (before the per-dataset standardization, which is fitted at
    train time on training rows), and writes a :class:`PreparedStore043` (keys, marks,
    distinct reference contexts, targets, the nine gene-gene graphs as edge lists) under
    ``$DATA_ROOT/data/torchcell/experiments/043-expression-proteome-morphology/train_043_prep/<tag>/``.

``train`` (Hydra; the default)
    Loads the prepared store, selects the arm's datasets, fits the standardization on the
    training rows and pins it by sha256, reads the four frozen vector tables for exactly
    the keys the arm's records use (a key with no vector raises), builds
    :class:`torchcell.models.cell_model_043.CellModel043`, and trains with AdamW and
    gradient clipping, logging to W&B project ``torchcell_043-expression-proteome-morphology``
    (group = arm, name = ``<arm>_s<seed>``). Validation is computed and logged; it never
    selects an epoch. The registered window (fractions of the run's epochs) is declared
    in the config before launch; checkpoints are written at its two ends and every epoch
    as ``last.pt``.

Arms (``conf/arm/*.yaml``): ``joint_9`` (Kemmeren, Messner, Zelezniak, Caudal, Muenzner
protein, Muenzner turnover, Boocock pooled (the 393 SM-medium records), N'Guessan,
Vaishnav; Sameith Dm read as split B through Kemmeren's tokens), and ``alone_<short>``
with one dataset. Loss arms ``conf/loss/{hlgauss,pinball,mse}.yaml``; weighting arms
``conf/weighting/{equal,records}.yaml``.

Store generation (``store_version_043``): the trainer reads ``data.store_version``
(``v2`` by default since 2026-10-23: ``prep_v3``, ``representation_043_v2``,
``global_split_v2``; ``v1`` gives ``prep_v2``, ``representation_043``,
``global_split_v1``, the inputs of every run before it). ``paths.split_file``,
``paths.prep_dir`` and ``paths.representation_root`` override the derived paths. The
prepare stage reads the generation from ``STORE_043_VERSION`` (the same variable
``dataset_043`` reads the raw store by).

    STORE_043_VERSION=v2 python experiments/043-expression-proteome-morphology/scripts/train_043.py --stage prepare
    python experiments/043-expression-proteome-morphology/scripts/train_043.py arm=joint_9 loss=hlgauss seed=0
    python experiments/043-expression-proteome-morphology/scripts/train_043.py data.store_version=v1
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import os.path as osp
import socket
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from torchcell.data.experiment_ontology import (
    AnnotationRecipe,
    ExperimentOntologyVocabulary,
)
from torchcell.data.representation_043 import ENCODERS, PositionGap
from torchcell.data.target_transform_043 import (
    TargetTransform043,
    fit_group_map,
    profile_keys,
)
from torchcell.losses.hl_gauss import HLGaussConfig, StandardizationStats
from torchcell.losses.readout_family_043 import (
    QuantileBins,
    ReadoutLossFamily043,
    fit_quantile_bins,
)
from torchcell.models.cell_model_043 import (
    Batch043,
    CellModel043,
    CellModel043Config,
    ContextInputs,
    MarkInputs,
    Replicon,
    build_head_mask,
    load_vector_table,
    select_marks,
)

SCRIPTS = osp.dirname(osp.abspath(__file__))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
from global_split_043 import GlobalSplit043, Role, load_split  # noqa: E402
from store_version_043 import StorePaths043, StoreVersion, store_paths  # noqa: E402
from store_version_043 import active as active_store  # noqa: E402

EXPERIMENT = "043-expression-proteome-morphology"
#: The prepared-store version is the store generation's (``store_version_043``):
#: prep_v2 (2026-10-23, store v1) adds Hu 2007, Jackson 2020 and Nadal-Ribelles 2025 to
#: prep_v1's ten datasets, the joint_9 records and their targets unchanged; prep_v3
#: (store v2) is the same build over raw_043_v2, every served record admitted. prep_v4
#: (2026-10-23) is prep_v3 plus each record's experiment ontology annotation
#: (``rec_onto`` into ``onto_profiles``, over the vocabulary ``ontology_annotation_043.py``
#: built from the whole raw store, copied into the prep dir as ``ONTOLOGY_VOCAB``).
WANDB_PROJECT = "torchcell_043-expression-proteome-morphology"
ISO_REL = "data/scerevisiae/isolate_embeddings/peter2018_1011"
#: 019's nine gene-gene relations, one per attention head (names of
#: ``torchcell.graph.graph.SCEREVISIAE_GENE_GRAPH_MAP``).
GRAPH_NAMES: tuple[str, ...] = (
    "physical",
    "regulatory",
    "tflink",
    "string12_0_neighborhood",
    "string12_0_fusion",
    "string12_0_cooccurence",
    "string12_0_coexpression",
    "string12_0_experimental",
    "string12_0_database",
)
ROLE_CODE: dict[Role, int] = {Role.TRAIN: 0, Role.VAL: 1, Role.SPLIT_B: 2}
BOOCOCK_POOLED_MEDIA = "SM (synthetic minimal"
BOOCOCK_POOLED_COUNT = 393
ONTOLOGY_VOCAB = "experiment_ontology_vocab.json"
#: Prepared-store versions that carry the ontology annotation.
ONTOLOGY_PREP_VERSIONS = frozenset({"prep_v4"})
TARGET_TRANSFORM_JSON = "target_transform.json"
QUANTILE_BINS_JSON = "quantile_bins.json"

Transform = Literal[
    "as_released",
    "log2_ratio_to_reference",
    "difference_to_reference",
    "log2_plus_one",
    "log2_positive",
    "log2_cpm_plus_one",
    "log2_cpm_ratio_to_reference",
]


class DatasetSpec(BaseModel):
    """One dataset as the trainer reads it (primer table ``tab:output-scales``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    short: str
    name: str
    level: Literal["gene", "record"]
    exp_class: str
    metric_type: str
    transform: Transform
    center_per_gene: bool
    split_b: bool = Field(
        default=False, description="evaluation only, read as split B through `read_as`"
    )
    read_as: str | None = None
    sentinel_abs: float | None = Field(
        default=None,
        gt=0,
        description="released values with |v| >= this are a sentinel, set missing "
        "(unmeasured) before the standardization and counted in the prepared meta",
    )


SPECS: dict[str, DatasetSpec] = {
    s.short: s
    for s in (
        DatasetSpec(
            short="kemmeren",
            name="MicroarrayKemmeren2014Dataset",
            level="gene",
            exp_class="MicroarrayExpressionPhenotype",
            metric_type="expression",
            transform="as_released",
            center_per_gene=False,
        ),
        DatasetSpec(
            short="messner",
            name="ProteomeMessner2023Dataset",
            level="gene",
            exp_class="ProteinAbundancePhenotype",
            metric_type="protein",
            transform="log2_ratio_to_reference",
            center_per_gene=False,
        ),
        DatasetSpec(
            short="zelezniak",
            name="ProteomeZelezniak2018Dataset",
            level="gene",
            exp_class="ProteinAbundancePhenotype",
            metric_type="protein",
            transform="difference_to_reference",
            center_per_gene=False,
        ),
        DatasetSpec(
            short="caudal",
            name="CaudalPanTranscriptome2024Dataset",
            level="gene",
            exp_class="RNASeqExpressionPhenotype",
            metric_type="expression",
            transform="log2_plus_one",
            center_per_gene=True,
        ),
        DatasetSpec(
            short="muenzner",
            name="ProteomeMuenzner2024Dataset",
            level="gene",
            exp_class="ProteinAbundancePhenotype",
            metric_type="protein",
            transform="log2_positive",
            center_per_gene=True,
        ),
        DatasetSpec(
            short="turnover",
            name="ProteinTurnoverMuenzner2024Dataset",
            level="gene",
            exp_class="ProteinTurnoverPhenotype",
            metric_type="turnover",
            transform="log2_positive",
            center_per_gene=True,
        ),
        DatasetSpec(
            short="boocock",
            name="Boocock2025Dataset",
            level="gene",
            exp_class="SingleCellExpressionPhenotype",
            metric_type="expression",
            transform="log2_cpm_plus_one",
            center_per_gene=True,
        ),
        DatasetSpec(
            short="nguessan",
            name="Nguessan2025Dataset",
            level="gene",
            exp_class="SingleCellExpressionPhenotype",
            metric_type="expression",
            transform="log2_cpm_plus_one",
            center_per_gene=True,
        ),
        DatasetSpec(
            short="vaishnav",
            name="PooledPromoterExpressionVaishnav2022Dataset",
            level="record",
            exp_class="PooledSortSeqExpressionPhenotype",
            metric_type="promoter",
            transform="as_released",
            center_per_gene=False,
        ),
        DatasetSpec(
            short="hu",
            name="MicroarrayScoreHu2007Dataset",
            level="gene",
            exp_class="MicroarrayExpressionScorePhenotype",
            metric_type="expression",
            transform="as_released",
            center_per_gene=False,
        ),
        DatasetSpec(
            short="jackson",
            name="TfDeletionPseudobulkJackson2020Dataset",
            level="gene",
            exp_class="SingleCellExpressionPhenotype",
            metric_type="expression",
            transform="log2_cpm_ratio_to_reference",
            center_per_gene=False,
        ),
        DatasetSpec(
            short="nadal",
            name="NadalRibellesPerturbSeq2025Dataset",
            level="gene",
            exp_class="PseudobulkExpressionPhenotype",
            metric_type="expression",
            transform="as_released",
            center_per_gene=False,
            sentinel_abs=20.0,
        ),
        DatasetSpec(
            short="sameith_dm",
            name="DmMicroarraySameith2015Dataset",
            level="gene",
            exp_class="MicroarrayExpressionPhenotype",
            metric_type="expression",
            transform="as_released",
            center_per_gene=False,
            split_b=True,
            read_as="kemmeren",
        ),
    )
}
JOINT_9: tuple[str, ...] = (
    "kemmeren",
    "messner",
    "zelezniak",
    "caudal",
    "muenzner",
    "turnover",
    "boocock",
    "nguessan",
    "vaishnav",
)
#: Arm ``all``: every dataset of the raw store with a working head. Sameith 2015 Sm is in
#: the store but the fold file excludes it (its 82 arrays are Kemmeren's, 82 of 82
#: identical); Ohya 2005 is the morphology probe (no record-level multi-output head).
ALL_DATASETS: tuple[str, ...] = (*JOINT_9, "hu", "jackson", "nadal")
#: The modeling-side ontology path of every experiment type the arms read (O2).
EXP_TYPE_PATHS: dict[str, tuple[str, ...]] = {
    "MicroarrayExpressionPhenotype": (
        "phenotype",
        "expression",
        "MicroarrayExpressionPhenotype",
    ),
    "RNASeqExpressionPhenotype": (
        "phenotype",
        "expression",
        "RNASeqExpressionPhenotype",
    ),
    "SingleCellExpressionPhenotype": (
        "phenotype",
        "expression",
        "SingleCellExpressionPhenotype",
    ),
    "ProteinAbundancePhenotype": ("phenotype", "protein", "ProteinAbundancePhenotype"),
    "ProteinTurnoverPhenotype": ("phenotype", "protein", "ProteinTurnoverPhenotype"),
    "PooledSortSeqExpressionPhenotype": (
        "phenotype",
        "promoter",
        "PooledSortSeqExpressionPhenotype",
    ),
    # appended (prep_v2): the earlier classes keep their indices
    "MicroarrayExpressionScorePhenotype": (
        "phenotype",
        "expression",
        "MicroarrayExpressionScorePhenotype",
    ),
    "PseudobulkExpressionPhenotype": (
        "phenotype",
        "expression",
        "PseudobulkExpressionPhenotype",
    ),
}
EXP_TYPE_CLASSES: tuple[str, ...] = tuple(EXP_TYPE_PATHS)
#: ``dataset_043.ExpType`` name of each served class (checked at prepare time).
EXP_TYPE_OF_CLASS: dict[str, str] = {
    "MicroarrayExpressionPhenotype": "MICROARRAY_EXPRESSION",
    "RNASeqExpressionPhenotype": "RNASEQ_EXPRESSION",
    "SingleCellExpressionPhenotype": "SINGLE_CELL_EXPRESSION",
    "ProteinAbundancePhenotype": "PROTEIN_ABUNDANCE",
    "ProteinTurnoverPhenotype": "PROTEIN_TURNOVER",
    "PooledSortSeqExpressionPhenotype": "POOLED_SORT_SEQ_EXPRESSION",
    "MicroarrayExpressionScorePhenotype": "MICROARRAY_EXPRESSION_SCORE",
    "PseudobulkExpressionPhenotype": "PSEUDOBULK_EXPRESSION",
}


# =========================================================================== config
class ArmConfig(BaseModel):
    """Which datasets train; ``doubles`` reads Sameith Dm as split B (needs Kemmeren)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    datasets: list[str] = Field(min_length=1)
    doubles: bool = True
    exclude_from_training: list[str] = Field(
        default_factory=list,
        description="arm datasets never trained on (no training batch holds them) but "
        "evaluated like the others: leave-one-dataset-out, read through the model's "
        "conditioning alone (needs model.conditioning.source=ontology to mean anything)",
    )

    @model_validator(mode="after")
    def _known(self) -> ArmConfig:
        """Every dataset is a trainable spec; doubles need Kemmeren."""
        for d in self.datasets:
            if d not in SPECS or SPECS[d].split_b:
                raise ValueError(f"{d!r} is not a trainable dataset: {sorted(SPECS)}")
        if len(set(self.datasets)) != len(self.datasets):
            raise ValueError("datasets repeat")
        bad = [d for d in self.exclude_from_training if d not in self.datasets]
        if bad or set(self.exclude_from_training) == set(self.datasets):
            raise ValueError(
                f"exclude_from_training {self.exclude_from_training} must be a proper "
                f"subset of the arm datasets {self.datasets}"
            )
        if self.doubles and "kemmeren" not in self.datasets:
            raise ValueError("doubles are read through Kemmeren; list kemmeren")
        return self


class LossConfig(BaseModel):
    """The objective (O4)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["hlgauss", "pinball", "mse", "mle", "binned_quantile"] = Field(
        description="hlgauss / binned_quantile (HL-Gauss over equal-mass bins of the "
        "TRAINING targets per group, K = hl_gauss.num_bins) / pinball / mse / mle "
        "(heteroscedastic Gaussian NLL, log-variance clamped to [-7, 7]); "
        "torchcell.losses.readout_family_043"
    )
    hl_gauss: HLGaussConfig = Field(default_factory=HLGaussConfig)
    num_quantiles: int = Field(default=19, gt=0)


class WeightingConfig(BaseModel):
    """Equal per dataset, or by record count (O5)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["equal", "records"]
    by: Literal["dataset", "profile"] = Field(
        default="dataset",
        description="the unit the weighting balances: dataset (legacy) or the record's "
        "experiment-ontology phenotype profile (datasets sharing a profile pool their "
        "loss first; needs a prep_v4 store)",
    )


class DataConfig(BaseModel):
    """Batching and evaluation subsets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    route: Literal["A", "B"] = "A"
    prep_version: str | None = Field(
        default=None,
        description="prepared-store version; None = the store generation's (prep_v3 "
        "for v2). prep_v4 = prep_v3 plus the ontology annotation",
    )
    store_version: StoreVersion = Field(
        default="v2",
        description="the 043 store generation (store_version_043): v2 = raw_043_v2 "
        "(every served record), prep_v3, global_split_v2; v1 = raw_043, prep_v2, "
        "global_split_v1",
    )
    sampler: Literal["records", "balanced"] = Field(
        default="records",
        description="records: uniform over training records (the per-record extreme); "
        "balanced: each batch slot draws a dataset uniformly, then a record uniformly "
        "within it. Composes with weighting (the loss-side knob).",
    )
    batch_size: int = Field(default=16, gt=0)
    label_batch_size: bool = Field(
        default=False,
        description="append _bs<batch_size x optim.accumulate_steps> (the effective batch) to "
        "the W&B group (a batch-size axis)",
    )
    num_workers: int = Field(default=4, ge=0)
    mark_cap: int | None = Field(
        default=None,
        gt=0,
        description="marks kept per record (select_marks); None = every mark (the design). "
        "A cap is a smoke-speed choice; the setup line prints the records it truncates",
    )
    val_max_records_per_dataset: int | None = Field(default=2000, gt=0)
    traineval_max_records_per_dataset: int = Field(default=256, gt=0)
    min_strains_per_gene: int = Field(default=3, ge=2)
    target_transform: Literal["zscore", "quantile_normal", "none"] = Field(
        default="zscore",
        description="second stage on the legacy z (torchcell.data.target_transform_043): "
        "zscore (legacy under target_stats_by=dataset), quantile_normal (rank to a "
        "standard normal per group, fitted on training values), none (released scale "
        "after the per-gene center). Predictions are mapped back to z before scoring",
    )
    target_stats_by: Literal["dataset", "profile"] = Field(
        default="dataset",
        description="the group the target statistics (and binned_quantile bins) are "
        "fitted over: dataset (legacy) or the experiment-ontology phenotype profile "
        "(needs a prep_v4 store)",
    )


class ConditioningConfig(BaseModel):
    """What the heads are conditioned on (``CellModel043Config.conditioning``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["name", "ontology"] = Field(
        default="name",
        description="name: type path + dataset + environment tokens and per-dataset bias "
        "tables (every run before 2026-10-23 ontology round); ontology: the experiment "
        "ontology annotation through OntologyDagEncoder (width 40 = the name token's), "
        "bias maps of it; needs a prep_v4 store",
    )
    state_dim: int = Field(default=16, gt=0)


class ModelConfig(BaseModel):
    """Widths of the model (the vocabularies come from the data)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hidden_dim: int = 144
    num_heads: int = 9
    num_layers: int = 4
    dropout: float = 0.1
    mask_layers: list[int] = Field(default_factory=lambda: [1])
    head_graphs: dict[int, str] = Field(
        default_factory=lambda: dict(enumerate(GRAPH_NAMES))
    )
    fourier_frequencies: int = 8
    operator_ffn_mult: int = 4
    type_token_dim: int = 16
    env_token_dim: int = 16
    dataset_token_dim: int = 8
    head_hidden: int = 128
    exp_type_mode: Literal["flat", "dag"] = "dag"
    conditioning: ConditioningConfig = Field(
        default_factory=lambda: ConditioningConfig()
    )
    # GilaHyper sweep (2026-10-24)
    layer_pattern: list[Literal["S", "M"]] | None = Field(
        default=None,
        description="trunk layers in order, M = multigraph-masked attention (head h sees "
        "graph head_graphs[h]), S = dense; when set it DEFINES num_layers and "
        "mask_layers (0-based), e.g. [S, M, S, S] = num_layers 4, mask_layers [1]",
    )
    trunk_ffn_mult: int = Field(
        default=4, gt=0, description="trunk FFN width / hidden_dim"
    )

    @model_validator(mode="before")
    @classmethod
    def _pattern_sets_layers(cls, data: Any) -> Any:
        """``layer_pattern`` -> ``num_layers`` and ``mask_layers``."""
        if isinstance(data, dict) and data.get("layer_pattern") is not None:
            pattern = [str(c) for c in data["layer_pattern"]]
            data = {
                **data,
                "num_layers": len(pattern),
                "mask_layers": [i for i, c in enumerate(pattern) if c == "M"],
            }
        return data

    @model_validator(mode="after")
    def _widths(self) -> ModelConfig:
        """Heads divide the width, loudly; a pattern agrees with the layers."""
        if self.hidden_dim % self.num_heads:
            raise ValueError(
                f"hidden_dim {self.hidden_dim} is not divisible by num_heads "
                f"{self.num_heads} (one head per graph channel); pick a multiple"
            )
        if self.layer_pattern is not None and (
            self.num_layers != len(self.layer_pattern)
            or self.mask_layers
            != [i for i, c in enumerate(self.layer_pattern) if c == "M"]
        ):
            raise ValueError("layer_pattern disagrees with num_layers / mask_layers")
        return self

    @property
    def pattern_label(self) -> str:
        """The trunk as S/M letters, e.g. SMSS."""
        masked = set(self.mask_layers)
        return "".join("M" if i in masked else "S" for i in range(self.num_layers))


class OptimConfig(BaseModel):
    """AdamW, linear warmup then constant (or cosine), gradient clipping by norm."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lr: float = Field(default=3.0e-4, gt=0)
    weight_decay: float = Field(default=1.0e-8, ge=0)
    warmup_steps: int = Field(default=500, ge=0)
    clip_grad_norm: float = Field(default=10.0, gt=0)
    schedule: Literal["constant", "cosine"] = Field(
        default="constant",
        description="after warmup: constant lr, or cosine from lr to min_lr_frac * lr at "
        "the run's last optimizer step",
    )
    min_lr_frac: float = Field(default=0.0, ge=0.0, le=1.0)
    accumulate_steps: int = Field(
        default=1,
        ge=1,
        description="micro-batches of data.batch_size per optimizer step (effective "
        "batch = batch_size x accumulate_steps). Steps, warmup, the lr schedule and "
        "max_steps(_per_epoch) count OPTIMIZER steps. Each micro-batch loss is weighted "
        "per dataset inside the micro-batch (weighting.kind), so with several datasets "
        "it is not exactly one step at the effective batch; one dataset with equal "
        "micro-batches is",
    )


class WindowConfig(BaseModel):
    """The registered read-out window, as fractions of the run's epochs (O10)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start_frac: float = Field(default=0.8, ge=0.0, le=1.0)
    end_frac: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> WindowConfig:
        """``start_frac < end_frac``."""
        if not self.start_frac < self.end_frac:
            raise ValueError("registered window needs start_frac < end_frac")
        return self

    def epochs(self, n_epochs: int) -> tuple[int, int]:
        """First and last epoch (1-based, inclusive) of the window for ``n_epochs``."""
        first = max(1, math.floor(self.start_frac * n_epochs) + 1)
        last = max(first, math.ceil(self.end_frac * n_epochs))
        return first, min(last, n_epochs)


class BudgetConfig(BaseModel):
    """Epochs, the wall-clock budget, and evaluation cadence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_epochs: int = Field(default=50, gt=0)
    time_budget_hours: float | None = Field(
        default=None,
        gt=0,
        description="when set, epochs = min(max_epochs, what fits by the first epoch's time)",
    )
    time_margin_minutes: float = Field(default=20.0, ge=0)
    max_steps_per_epoch: int | None = Field(default=None, gt=0, description="tests")
    max_steps: int | None = Field(
        default=None,
        gt=0,
        description=(
            "total optimizer steps for a step-matched run (an alone arm matched to a "
            "joint run): the run takes ceil(max_steps / steps_per_epoch) epochs, the "
            "last one cut so the total is exactly max_steps, and max_epochs is ignored. "
            "A time budget that would cut such a run raises (an unmatched alone run is "
            "not a transfer control)."
        ),
    )
    probe_steps: int = Field(default=30, gt=0)
    checkpoint_every_steps: int | None = Field(
        default=None,
        gt=0,
        description="also write last.pt every n optimizer steps inside an epoch (with the "
        "micro-batches done and the epoch's running sums), so a resume continues "
        "mid-epoch, skipping the micro-batches already trained (the records sampler's "
        "order is fixed by seed and epoch); the dropout RNG state is not restored. None "
        "= at epoch ends only",
    )
    eval_every_epochs: int = Field(default=1, gt=0)
    registered_window: WindowConfig = Field(default_factory=WindowConfig)


class WandbConfig(BaseModel):
    """Where the run logs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    project: str = WANDB_PROJECT
    entity: str | None = None
    mode: Literal["online", "offline", "disabled"] = "online"
    tags: list[str] = Field(default_factory=list)
    round: str | None = Field(default=None, description="round label (config only)")
    label_knobs: bool = Field(
        default=False,
        description="append the 2026-10-24 sweep knobs that differ from their defaults "
        "to the W&B group (sweep_suffix); off keeps every earlier run name",
    )


class PathsConfig(BaseModel):
    """Inputs and outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data_root: str
    split_file: str | None = None
    prep_dir: str | None = None
    representation_root: str | None = None
    alleles_root: str | None = None
    checkpoint_root: str | None = None

    def resolved_split_file(self, store: StorePaths043) -> str:
        """``conf/splits/<split_version>.json`` of the store generation."""
        return self.split_file or osp.join(
            osp.dirname(SCRIPTS), "conf", "splits", f"{store.split_version}.json"
        )

    def resolved_prep_dir(
        self, route: str, store: StorePaths043, prep_version: str | None = None
    ) -> str:
        """The prepared store directory (``prep_version`` overrides the generation's)."""
        if self.prep_dir is not None:
            return self.prep_dir
        return osp.join(
            self.data_root,
            "data/torchcell/experiments",
            EXPERIMENT,
            "train_043_prep",
            f"{prep_version or store.prep_version}_route{route}",
        )

    def resolved_representation_root(self, store: StorePaths043) -> str:
        """``representation_043`` / ``representation_043_v2`` (vocabularies)."""
        return self.representation_root or osp.join(
            self.data_root,
            "data/torchcell/experiments",
            EXPERIMENT,
            store.representation_tag,
        )

    def resolved_alleles_root(self) -> str:
        """The four ``<encoder>.h5`` tables."""
        return self.alleles_root or osp.join(self.data_root, ISO_REL, "alleles")

    def resolved_checkpoint_root(self) -> str:
        """``$DATA_ROOT/experiments/<EXPERIMENT>/checkpoints``."""
        return self.checkpoint_root or osp.join(
            self.data_root, "experiments", EXPERIMENT, "checkpoints"
        )


class TrainConfig043(BaseModel):
    """The whole run, validated from the Hydra config."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = 0
    device: Literal["auto", "cpu", "cuda", "xpu"] = "auto"
    attention_query_chunk: int | None = Field(
        default=None,
        gt=0,
        description="query block size of the model's SDPA calls (memory, not the model: "
        "the attention is the same). Set on XPU, where masked or dropout SDPA forms the "
        "score matrix (Aurora job 8959152); None = one call",
    )
    init_from: str | None = Field(
        default=None,
        description="stage 2: a stage-1 checkpoint (.pt) whose MODEL weights initialize "
        "this run (never its optimizer); the model and loss sections must match it",
    )
    init_from_label: str | None = Field(
        default=None,
        description="the stage-1 run's W&B group, read from the checkpoint by run(); "
        "names this run finetune_<dataset>_from_<label>",
    )
    resume_id: str | None = Field(
        default=None,
        description="checkpoint directory name and W&B run id: a rerun with the same id "
        "resumes from that directory's last.pt (model, optimizer, epoch, step) and the "
        "same W&B run (preemptible queues); None = one fresh directory per job",
    )
    arm: ArmConfig
    loss: LossConfig
    weighting: WeightingConfig
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    paths: PathsConfig

    @model_validator(mode="after")
    def _init_label(self) -> TrainConfig043:
        """A stage-1 label only with a stage-1 checkpoint."""
        if self.init_from_label is not None and self.init_from is None:
            raise ValueError("init_from_label without init_from")
        if (
            self.arm.exclude_from_training
            and self.model.conditioning.source != "ontology"
        ):
            raise ValueError("exclude_from_training is read through the ontology only")
        return self

    @property
    def arm_label(self) -> str:
        """``<arm>[_balanced]_<loss>_<weighting>``: the W&B group.

        A stage-2 run (``init_from``) is ``finetune_<arm without alone_>_from_<stage-1
        group>``.
        """
        if self.init_from is not None:
            if self.init_from_label is None:
                raise ValueError(
                    "init_from_label is filled from the checkpoint by run()"
                )
            return f"finetune_{self.arm.name.removeprefix('alone_')}_from_{self.init_from_label}"
        sampler = "_balanced" if self.data.sampler == "balanced" else ""
        onto = "_onto" if self.model.conditioning.source == "ontology" else ""
        lodo = "".join(f"_lodo_{d}" for d in self.arm.exclude_from_training)
        # the effective batch: batch_size x optim.accumulate_steps
        eff = self.data.batch_size * self.optim.accumulate_steps
        bs = f"_bs{eff}" if self.data.label_batch_size else ""
        return (
            f"{self.arm.name}{lodo}{onto}{sampler}_{self.loss.kind}_"
            f"{self.weighting.kind}{bs}{self.sweep_suffix}"
        )

    @property
    def sweep_suffix(self) -> str:
        """The 2026-10-24 sweep knobs that differ from their defaults (else empty);
        empty unless ``wandb.label_knobs``.
        """
        if not self.wandb.label_knobs:
            return ""
        m, d0 = self.model, ModelConfig()
        parts = []
        if m.layer_pattern is not None:
            parts.append(f"L{m.pattern_label}")
        if m.hidden_dim != d0.hidden_dim:
            parts.append(f"h{m.hidden_dim}")
        if m.trunk_ffn_mult != d0.trunk_ffn_mult:
            parts.append(f"ffn{m.trunk_ffn_mult}")
        if m.head_hidden != d0.head_hidden:
            parts.append(f"hh{m.head_hidden}")
        if self.data.target_transform != "zscore":
            parts.append(
                {"quantile_normal": "tqn", "none": "tnone"}[self.data.target_transform]
            )
        if self.data.target_stats_by == "profile":
            parts.append("statsprofile")
        if self.weighting.by == "profile":
            parts.append("wprofile")
        if self.optim.schedule == "cosine":
            parts.append("cos")
        if self.optim.lr != OptimConfig().lr:
            parts.append(f"lr{self.optim.lr:g}")
        return "".join(f"_{p}" for p in parts)

    @property
    def run_name(self) -> str:
        """``<arm label>_s<seed>``."""
        return f"{self.arm_label}_s{self.seed}"


# =========================================================================== store
class PreparedMeta(BaseModel):
    """``meta.json`` of a prepared store."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prep_version: str
    route: str
    split_version: str
    split_store_sha256: str
    datasets: list[str]
    genes: list[str]
    env_keys: list[str]
    context_ids: list[str]
    graph_names: list[str]
    counts: dict[str, dict[str, int]]
    edges_outside_index: dict[str, int] = Field(default_factory=dict)
    ontology_vocab_sha256: str | None = None
    ontology_n_nodes: int | None = None
    ontology_n_profiles: int | None = None
    created_at: str
    host: str
    slurm_job_id: str | None
    torchcell_commit: str
    wall_seconds: float


@dataclass
class PreparedStore043:
    """Every record the arms may read (test records excluded), as flat arrays.

    Records: ``rec_*`` ``[R]``. Marks in CSR form (``mark_ptr`` ``[R+1]``). Contexts
    ``ctx_*`` ``[C, G, ...]``; key arrays hold VOCABULARY ids of
    ``representation_043/vocab_<encoder>.npy`` (``-1`` = gap). ``Y`` ``[Rg, G]`` holds the
    transformed gene-level targets of the records whose ``rec_yrow >= 0``.
    """

    meta: PreparedMeta
    rec_dataset: npt.NDArray[np.int16]
    rec_raw_index: npt.NDArray[np.int64]
    rec_role: npt.NDArray[np.int8]
    rec_env: npt.NDArray[np.int32]
    rec_context: npt.NDArray[np.int32]
    rec_yrow: npt.NDArray[np.int32]
    rec_yscalar: npt.NDArray[np.float32]
    rec_genes2: npt.NDArray[np.int32]
    rec_unit: npt.NDArray[np.str_]
    mark_ptr: npt.NDArray[np.int64]
    mark_gene: npt.NDArray[np.int32]
    mark_on_graph: npt.NDArray[np.bool_]
    mark_change: npt.NDArray[np.int8]
    mark_parent: npt.NDArray[np.int8]
    mark_keys: npt.NDArray[np.int32]
    mark_gaps: npt.NDArray[np.int8]
    mark_replicon: npt.NDArray[np.int8]
    mark_relmid: npt.NDArray[np.float32]
    mark_chrom_copy: npt.NDArray[np.float32]
    mark_gene_copy: npt.NDArray[np.float32]
    mark_pgap: npt.NDArray[np.int8]
    ctx_keys: npt.NDArray[np.int32]
    ctx_gaps: npt.NDArray[np.int8]
    ctx_chrom_copy: npt.NDArray[np.float32]
    ctx_gene_copy: npt.NDArray[np.float32]
    ctx_pgap: npt.NDArray[np.int8]
    chrom_index: npt.NDArray[np.int8]
    rel_midpoint: npt.NDArray[np.float32]
    Y: npt.NDArray[np.float32]
    edges: dict[str, npt.NDArray[np.int64]] = field(default_factory=dict)
    # prep_v4: the ontology profile of every record, and the distinct profiles
    rec_onto: npt.NDArray[np.int32] = field(
        default_factory=lambda: np.zeros(0, np.int32)
    )
    onto_profiles: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((0, 0), np.float32)
    )
    ontology_vocabulary: ExperimentOntologyVocabulary | None = None

    def array_fields(self) -> dict[str, np.ndarray]:
        """Every array, by name (edges as ``edges__<name>``)."""
        out = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in ("meta", "edges", "ontology_vocabulary")
        }
        out.update({f"edges__{k}": v for k, v in self.edges.items()})
        return out

    def save(self, directory: str) -> None:
        """``store.npz`` + ``meta.json``."""
        os.makedirs(directory, exist_ok=True)
        np.savez(osp.join(directory, "store.npz"), **self.array_fields())
        with open(osp.join(directory, "meta.json"), "w") as f:
            f.write(self.meta.model_dump_json(indent=1))
        if self.ontology_vocabulary is not None:
            self.ontology_vocabulary.to_json(osp.join(directory, ONTOLOGY_VOCAB))

    @classmethod
    def load(cls, directory: str) -> PreparedStore043:
        """Read back what :meth:`save` wrote."""
        with open(osp.join(directory, "meta.json")) as f:
            meta = PreparedMeta.model_validate_json(f.read())
        z = np.load(osp.join(directory, "store.npz"))
        arrays = {k: z[k] for k in z.files}
        edges = {
            k.split("__", 1)[1]: arrays.pop(k)
            for k in list(arrays)
            if k.startswith("edges__")
        }
        vocab = None
        if meta.ontology_vocab_sha256 is not None:
            path = osp.join(directory, ONTOLOGY_VOCAB)
            with open(path, "rb") as f:
                if hashlib.sha256(f.read()).hexdigest() != meta.ontology_vocab_sha256:
                    raise ValueError(f"{path} does not match the sha256 in meta.json")
            vocab = ExperimentOntologyVocabulary.from_json(path)
        return cls(meta=meta, edges=edges, ontology_vocabulary=vocab, **arrays)


# =========================================================================== prepare
def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=SCRIPTS,
    ).stdout.strip()


def transform_target(
    transform: Transform,
    y: npt.NDArray[np.float64],
    y_ref: npt.NDArray[np.float64] | None = None,
    total_umis: float | None = None,
    total_umis_ref: float | None = None,
) -> npt.NDArray[np.float32]:
    """The pre-standardization transform of one record's gene vector (NaN = missing).

    ``log2_cpm_ratio_to_reference`` (Jackson 2020): ``log2((CPM + 1) / (CPM_ref + 1))``
    against the record's served ``experiment_reference`` of the same environment, omitted
    keys read as zero counts on both sides; a gene with zero counts on both sides carries
    no ratio and is missing.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        if transform == "as_released":
            out = y
        elif transform == "log2_ratio_to_reference":
            assert y_ref is not None
            out = np.where((y > 0) & (y_ref > 0), np.log2(y) - np.log2(y_ref), np.nan)
        elif transform == "difference_to_reference":
            assert y_ref is not None
            out = y - y_ref
        elif transform == "log2_plus_one":
            out = np.log2(y + 1.0)
        elif transform == "log2_positive":
            out = np.where(y > 0, np.log2(y), np.nan)
        elif transform == "log2_cpm_plus_one":
            if total_umis is None or not total_umis > 0:
                raise ValueError("log2_cpm_plus_one needs a positive total_umis")
            counts = np.nan_to_num(y, nan=0.0)  # omitted keys are released zeros
            out = np.log2(counts / total_umis * 1e6 + 1.0)
        elif transform == "log2_cpm_ratio_to_reference":
            if (
                y_ref is None
                or not (total_umis or 0) > 0
                or not (total_umis_ref or 0) > 0
            ):
                raise ValueError(
                    "log2_cpm_ratio_to_reference needs y_ref, total_umis and total_umis_ref"
                )
            assert total_umis is not None and total_umis_ref is not None
            c = np.nan_to_num(y, nan=0.0)
            c_ref = np.nan_to_num(y_ref, nan=0.0)
            out = np.log2(
                (c / total_umis * 1e6 + 1.0) / (c_ref / total_umis_ref * 1e6 + 1.0)
            )
            out = np.where((c > 0) | (c_ref > 0), out, np.nan)
        else:
            raise ValueError(f"unknown transform {transform!r}")
    return np.asarray(out, dtype=np.float32)


def mask_sentinels(
    y: npt.NDArray[np.float32], sentinel_abs: float | None
) -> tuple[npt.NDArray[np.float32], int]:
    """``y`` with ``|v| >= sentinel_abs`` set missing, and how many cells were set."""
    if sentinel_abs is None:
        return y, 0
    with np.errstate(invalid="ignore"):
        hit = np.abs(y) >= sentinel_abs
    out = y.copy()
    out[hit] = np.nan
    return out, int(hit.sum())


def _context_hash(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


class _PrepRows(torch.utils.data.Dataset[dict[str, Any] | None]):
    """One compact row per raw record (runs in DataLoader workers)."""

    def __init__(self, ds: Any, split: GlobalSplit043, ontology: bool = False) -> None:
        self.ds = ds
        self.split = split
        self.ontology = ontology
        self.short_of = {s.name: s.short for s in SPECS.values()}

    def __len__(self) -> int:
        return len(self.ds.items)

    def __getitem__(self, i: int) -> dict[str, Any] | None:
        import dataset_043 as d043
        import representation_043 as rep043
        from make_global_split_043 import isolate_key

        name, raw_index = self.ds.items[i]
        spec = SPECS[self.short_of[name]]
        if spec.short == "boocock":
            with self.ds._txn() as txn:
                raw = txn.get(f"data_{raw_index}".encode())
            if BOOCOCK_POOLED_MEDIA.encode() not in raw:
                return None  # de novo cells: not in the smoke (decision P1 open)
        record = self.ds.record(raw_index)
        ex = record["experiment"]
        g = ex["genotype"]
        # ---- role under the fold file
        if spec.short == "vaishnav":
            (p,) = g["perturbations"]
            key: str | list[str] = p["promoter_sequence"][
                p["variable_region_start"] - 1 : p["variable_region_end"]
            ]
        elif "perturbations" not in g:
            key = f"{g['cross']}:{g['segregant_id']}"
        elif spec.name in (
            "CaudalPanTranscriptome2024Dataset",
            "ProteomeMuenzner2024Dataset",
            "ProteinTurnoverMuenzner2024Dataset",
        ):
            key = isolate_key(record)
        else:
            genes = []
            for pp in g["perturbations"]:
                idx = self.ds._gene_idx(pp["systematic_gene_name"])
                genes.append(
                    pp["systematic_gene_name"] if idx is None else self.ds.genes[idx]
                )
            key = sorted(genes)
        role = self.split.role(spec.name, key)
        if role not in ROLE_CODE:
            return None  # test records are never loaded; probe/excluded are not arms
        sample = self.ds[i]
        if (
            d043.ExpType(sample["exp_type_id"]).name
            != EXP_TYPE_OF_CLASS[spec.exp_class]
        ):
            raise ValueError(f"{name} data_{raw_index}: experiment type mismatch")
        r = sample["representation"]
        # ---- target
        y_vec: np.ndarray | None = None
        y_scalar = float("nan")
        n_sentinel = 0
        if spec.level == "gene":
            y_ref = (
                sample["y_ref"].numpy().astype(np.float64)
                if "y_ref" in sample
                else None
            )
            cpm = spec.transform in ("log2_cpm_plus_one", "log2_cpm_ratio_to_reference")
            total = float(ex["phenotype"]["total_umis"]) if cpm else None
            total_ref = (
                float(
                    record["experiment_reference"]["phenotype_reference"]["total_umis"]
                )
                if spec.transform == "log2_cpm_ratio_to_reference"
                else None
            )
            y_vec = transform_target(
                spec.transform,
                sample["y"].numpy().astype(np.float64),
                y_ref,
                total,
                total_ref,
            )
            y_vec, n_sentinel = mask_sentinels(y_vec, spec.sentinel_abs)
        else:
            y_scalar = float(sample["y_scalar"])
        # ---- context (route A isolates carry their own; everything else is S288C)
        if r.context_id == rep043.S288C_CONTEXT:
            ctx = None
        else:
            ctx = (
                r.key_ids.astype(np.int32),
                r.key_gap.astype(np.int8),
                r.chrom_copy.astype(np.float32),
                r.gene_copy.astype(np.float32),
                r.position_gap.astype(np.int8),
            )
        # ---- marks: operator genes, then off-graph tokens
        on = np.nonzero(r.operator_mask)[0]
        n_on, n_ex = len(on), len(r.extra_tokens)
        marks = {
            "gene": np.concatenate([on, np.zeros(n_ex, dtype=np.int64)]).astype(
                np.int32
            ),
            "on_graph": np.concatenate([np.ones(n_on, bool), np.zeros(n_ex, bool)]),
            "change": np.concatenate(
                [r.change_type[on], [t.change_type for t in r.extra_tokens]]
            ).astype(np.int8),
            "parent": np.concatenate(
                [r.allele_parent[on] + 1, np.zeros(n_ex, dtype=np.int8)]
            ).astype(np.int8),
            "keys": np.concatenate(
                [
                    r.key_ids[on],
                    np.asarray([t.key_ids for t in r.extra_tokens]).reshape(-1, 4),
                ]
            ).astype(np.int32),
            "gaps": np.concatenate(
                [
                    r.key_gap[on],
                    np.asarray([t.gaps for t in r.extra_tokens]).reshape(-1, 4),
                ]
            ).astype(np.int8),
            "replicon": np.concatenate(
                [
                    r.chrom_index[on],
                    [
                        # an accessory ORF and an off-index gene (a deleted gene with
                        # no index position, prep_v3) share the row of an ORF with no
                        # reference coordinate; the reporter has its own
                        int(Replicon.REPORTER_CONSTRUCT)
                        if t.kind == "reporter"
                        else int(Replicon.ACCESSORY_ORF)
                        for t in r.extra_tokens
                    ],
                ]
            ).astype(np.int8),
            "relmid": np.concatenate([r.rel_midpoint[on], np.zeros(n_ex)]).astype(
                np.float32
            ),
            "chrom_copy": np.concatenate([r.chrom_copy[on], np.ones(n_ex)]).astype(
                np.float32
            ),
            "gene_copy": np.concatenate(
                [
                    r.gene_copy[on],
                    [
                        np.nan if t.copy_number is None else t.copy_number
                        for t in r.extra_tokens
                    ],
                ]
            ).astype(np.float32),
            "pgap": np.concatenate(
                [
                    r.position_gap[on],
                    [
                        int(PositionGap.COPY_NUMBER_UNRELEASED)
                        if t.copy_number is None
                        else 0
                        for t in r.extra_tokens
                    ],
                ]
            ).astype(np.int8),
        }
        onto = None
        if self.ontology:
            from torchcell.data.experiment_ontology import annotation_recipe
            from torchcell.datamodels.schema import (
                EXPERIMENT_REFERENCE_TYPE_MAP,
                EXPERIMENT_TYPE_MAP,
            )

            rr = record["experiment_reference"]
            recipe = annotation_recipe(
                EXPERIMENT_TYPE_MAP[ex["experiment_type"]](**ex),
                EXPERIMENT_REFERENCE_TYPE_MAP[rr["experiment_reference_type"]](**rr),
            )
            onto = json.dumps(recipe.model_dump(), sort_keys=True)
        genes2 = [-1, -1]
        if spec.short in ("kemmeren", "sameith_dm"):
            idx = [self.ds.gene_to_idx[k] for k in key if k in self.ds.gene_to_idx]
            genes2[: len(idx)] = idx[:2]
        return {
            "short": spec.short,
            "raw_index": raw_index,
            "role": ROLE_CODE[role],
            "env": sample["env_key"],
            "unit": key if isinstance(key, str) else "+".join(key),
            "y": y_vec,
            "y_scalar": y_scalar,
            "ctx": ctx,
            "marks": marks,
            "genes2": genes2,
            "n_sentinel": n_sentinel,
            "onto": onto,
        }


def _edge_lists(
    data_root: str, genes: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """The nine gene-gene graphs over the store's gene index."""
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.graph.graph import build_gene_multigraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(data_root, "data/sgd/genome"),
        string_root=osp.join(data_root, "data/string"),
        tflink_root=osp.join(data_root, "data/tflink"),
        genome=genome,
    )
    mg = build_gene_multigraph(graph=graph, graph_names=list(GRAPH_NAMES))
    gi = {g: i for i, g in enumerate(genes)}
    edges: dict[str, np.ndarray] = {}
    outside: dict[str, int] = {}
    for name in GRAPH_NAMES:
        pairs = []
        n_out = 0
        for u, v in mg[name].graph.edges():
            if u in gi and v in gi:
                pairs.append((gi[u], gi[v]))
            else:
                n_out += 1
        edges[name] = np.asarray(pairs, dtype=np.int64).T.reshape(2, -1)
        outside[name] = n_out
        print(f"graph {name}: {edges[name].shape[1]} edges, {n_out} outside the index")
    return edges, outside


def prepare(
    paths: PathsConfig,
    route: Literal["A", "B"],
    workers: int,
    generation: StorePaths043,
    prep_version: str | None = None,
    ontology_vocab: str | None = None,
) -> str:
    """Build the prepared store over every spec (module docstring).

    ``prep_version`` names the output (default: the generation's); a version in
    ``ONTOLOGY_PREP_VERSIONS`` also annotates every record against the vocabulary JSON
    ``ontology_vocab`` (written by ``ontology_annotation_043.py`` over the whole raw
    store); a record whose annotation leaves that vocabulary raises.
    """
    prep_version = prep_version or generation.prep_version
    with_onto = prep_version in ONTOLOGY_PREP_VERSIONS
    if with_onto != (ontology_vocab is not None):
        raise ValueError(
            f"{prep_version}: the ontology vocabulary is required exactly "
            f"for {sorted(ONTOLOGY_PREP_VERSIONS)}"
        )
    import dataset_043 as d043

    from torchcell.data.representation_043 import RepresentationConfig

    if active_store().version != generation.version:
        raise ValueError(
            f"prepare for store {generation.version}, but dataset_043 reads "
            f"{active_store().version} (STORE_043_VERSION)"
        )
    t0 = time.time()
    split = load_split(paths.resolved_split_file(generation))
    names = [s.name for s in SPECS.values()]
    ds = d043.Raw043Dataset(
        d043.Dataset043Config(
            data_root=paths.data_root,
            datasets=names,
            include_reference=True,
            representation=RepresentationConfig(route=route),
        )
    )
    split.check_store(osp.join(ds.config.store_root, "raw", d043.RAW_INDEX_NAME))
    G = ds.n_genes
    loader = torch.utils.data.DataLoader(
        _PrepRows(ds, split, with_onto),
        batch_size=64,
        num_workers=workers,
        collate_fn=lambda b: b,
    )
    shorts = list(SPECS)
    tables = ds.resolver.tables
    s288c = tables.index("S288C")
    ctx_rows: list[tuple[np.ndarray, ...]] = [
        (
            tables.key_ids[s288c].astype(np.int32),
            tables.key_gap[s288c].astype(np.int8),
            np.ones(G, np.float32),
            np.ones(G, np.float32),
            np.zeros(G, np.int8),
        )
    ]
    ctx_ids = ["S288C"]
    ctx_index = {_context_hash(*ctx_rows[0]): 0}
    rec: dict[str, list[Any]] = defaultdict(list)
    marks: dict[str, list[np.ndarray]] = defaultdict(list)
    ys: list[np.ndarray] = []
    env_index: dict[str, int] = {}
    onto_index: dict[str, int] = {}
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    done = 0
    for chunk in loader:
        for row in chunk:
            done += 1
            if row is None:
                continue
            if row["ctx"] is None:
                c = 0
            else:
                h = _context_hash(*row["ctx"])
                if h not in ctx_index:
                    ctx_index[h] = len(ctx_rows)
                    ctx_rows.append(row["ctx"])
                    ctx_ids.append(row["unit"])
                c = ctx_index[h]
            rec["dataset"].append(shorts.index(row["short"]))
            rec["raw_index"].append(row["raw_index"])
            rec["role"].append(row["role"])
            rec["env"].append(env_index.setdefault(row["env"], len(env_index)))
            if with_onto:
                rec["onto"].append(onto_index.setdefault(row["onto"], len(onto_index)))
            rec["context"].append(c)
            rec["yrow"].append(len(ys) if row["y"] is not None else -1)
            if row["y"] is not None:
                ys.append(row["y"])
            rec["yscalar"].append(row["y_scalar"])
            rec["genes2"].append(row["genes2"])
            rec["unit"].append(row["unit"])
            rec["n_marks"].append(len(row["marks"]["gene"]))
            for k, v in row["marks"].items():
                marks[k].append(v)
            counts[row["short"]][{0: "train", 1: "val", 2: "split_b"}[row["role"]]] += 1
            if row["n_sentinel"]:
                counts[row["short"]]["sentinel_masked_cells"] += row["n_sentinel"]
        print(
            f"{done}/{len(ds)} records read, {len(rec['raw_index'])} kept", flush=True
        )
    print(f"Boocock pooled records kept (train + val): {dict(counts['boocock'])}")
    ptr = np.zeros(len(rec["n_marks"]) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(rec["n_marks"])
    edges, outside = _edge_lists(paths.data_root, ds.genes)
    vocab = None
    onto_profiles = np.zeros((0, 0), np.float32)
    vocab_sha = None
    if with_onto:
        assert ontology_vocab is not None
        vocab = ExperimentOntologyVocabulary.from_json(ontology_vocab)
        onto_profiles = np.stack(
            [
                vocab.vector_from_recipe(
                    AnnotationRecipe.model_validate_json(k)
                ).numpy()
                for k in onto_index
            ]
        ).astype(np.float32)
        with open(ontology_vocab, "rb") as f:
            vocab_sha = hashlib.sha256(f.read()).hexdigest()
        print(f"ontology: {len(vocab)} nodes, {len(onto_index)} distinct profiles")
    prepared = PreparedStore043(
        meta=PreparedMeta(
            prep_version=prep_version,
            route=route,
            split_version=split.version,
            split_store_sha256=split.store_sha256,
            datasets=shorts,
            genes=list(ds.genes),
            env_keys=list(env_index),
            context_ids=ctx_ids,
            graph_names=list(GRAPH_NAMES),
            counts={k: dict(v) for k, v in counts.items()},
            edges_outside_index=outside,
            ontology_vocab_sha256=vocab_sha,
            ontology_n_nodes=None if vocab is None else len(vocab),
            ontology_n_profiles=len(onto_index) if with_onto else None,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            host=socket.gethostname(),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            torchcell_commit=_git_commit(),
            wall_seconds=round(time.time() - t0, 1),
        ),
        rec_dataset=np.asarray(rec["dataset"], np.int16),
        rec_raw_index=np.asarray(rec["raw_index"], np.int64),
        rec_role=np.asarray(rec["role"], np.int8),
        rec_env=np.asarray(rec["env"], np.int32),
        rec_context=np.asarray(rec["context"], np.int32),
        rec_yrow=np.asarray(rec["yrow"], np.int32),
        rec_yscalar=np.asarray(rec["yscalar"], np.float32),
        rec_genes2=np.asarray(rec["genes2"], np.int32).reshape(-1, 2),
        rec_unit=np.asarray(rec["unit"], dtype=np.str_),
        mark_ptr=ptr,
        mark_gene=np.concatenate(marks["gene"]).astype(np.int32),
        mark_on_graph=np.concatenate(marks["on_graph"]).astype(bool),
        mark_change=np.concatenate(marks["change"]).astype(np.int8),
        mark_parent=np.concatenate(marks["parent"]).astype(np.int8),
        mark_keys=np.concatenate(marks["keys"]).astype(np.int32).reshape(-1, 4),
        mark_gaps=np.concatenate(marks["gaps"]).astype(np.int8).reshape(-1, 4),
        mark_replicon=np.concatenate(marks["replicon"]).astype(np.int8),
        mark_relmid=np.concatenate(marks["relmid"]).astype(np.float32),
        mark_chrom_copy=np.concatenate(marks["chrom_copy"]).astype(np.float32),
        mark_gene_copy=np.concatenate(marks["gene_copy"]).astype(np.float32),
        mark_pgap=np.concatenate(marks["pgap"]).astype(np.int8),
        ctx_keys=np.stack([c[0] for c in ctx_rows]),
        ctx_gaps=np.stack([c[1] for c in ctx_rows]),
        ctx_chrom_copy=np.stack([c[2] for c in ctx_rows]),
        ctx_gene_copy=np.stack([c[3] for c in ctx_rows]),
        ctx_pgap=np.stack([c[4] for c in ctx_rows]),
        chrom_index=ds.resolver.pos.chrom_index.astype(np.int8),
        rel_midpoint=ds.resolver.pos.rel_midpoint.astype(np.float32),
        Y=np.stack(ys).astype(np.float32),
        edges=edges,
        rec_onto=np.asarray(rec["onto"], np.int32),
        onto_profiles=onto_profiles,
        ontology_vocabulary=vocab,
    )
    out = paths.resolved_prep_dir(route, generation, prep_version)
    prepared.save(out)
    print(prepared.meta.model_dump_json(indent=1))
    return out


# =========================================================================== training view
@dataclass
class TrainingView:
    """The arm's records, standardized targets and compact key rows (fork-shared)."""

    store: PreparedStore043
    datasets: list[str]  # arm-local order; dataset id = position
    rec: npt.NDArray[np.int64]  # store record index of every arm record
    local_dataset: npt.NDArray[np.int64]  # [R_arm]
    head_dataset: npt.NDArray[np.int64]  # [R_arm] dataset id read at the head
    level_gene: npt.NDArray[np.bool_]
    role: npt.NDArray[np.int8]
    exp_type: npt.NDArray[np.int64]
    Z: npt.NDArray[np.float32]  # [Rg_store, G] standardized (NaN = no target)
    z_scalar: npt.NDArray[np.float32]  # [R_store]
    train_mean_z: dict[str, npt.NDArray[np.float64]]
    stats: StandardizationStats
    ctx_rows: npt.NDArray[np.int32]  # [C, G, 4] compact rows
    mark_rows: npt.NDArray[np.int32]  # [M, 4] compact rows
    used_ids: list[npt.NDArray[np.int64]]  # per encoder: sorted vocab ids
    mark_cap: int | None
    train_mask: npt.NDArray[np.bool_] | None = None  # [R_arm] False = excluded dataset
    doubles: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.zeros(0, np.int64)
    )  # arm positions of split-B records
    doubles_singles: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.zeros((0, 2), np.int64)
    )  # arm positions of the Kemmeren single of each gene of each double
    # ---- 2026-10-24 sweep: target statistics group of each arm record and the training
    # targets t (None = the legacy z: zscore by dataset, bit-identical)
    group: npt.NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, np.int64))
    group_names: list[str] = field(default_factory=list)
    row_mu: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, np.float64)
    )  # [R_arm] the record's dataset mu (legacy standardization)
    row_sd: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.ones(0, np.float64)
    )
    target_map: TargetTransform043 | None = None
    T: npt.NDArray[np.float32] | None = None  # [Rg_store, G] training targets
    t_scalar: npt.NDArray[np.float32] | None = None  # [R_store]
    weight_group: npt.NDArray[np.int64] | None = None  # [n_datasets] profile of each

    @property
    def gene_targets(self) -> npt.NDArray[np.float32]:
        """The gene-level training targets (``T``, or the legacy ``Z``)."""
        return self.Z if self.T is None else self.T

    @property
    def record_targets(self) -> npt.NDArray[np.float32]:
        """The record-level training targets (``t_scalar``, or ``z_scalar``)."""
        return self.z_scalar if self.t_scalar is None else self.t_scalar

    def positions(self, role: int, dataset: str | None = None) -> npt.NDArray[np.int64]:
        """Arm positions with ``role`` (and of ``dataset`` when given)."""
        sel = self.role == role
        if dataset is not None:
            sel &= np.asarray(
                [self.datasets[d] == dataset for d in self.local_dataset], bool
            )
        return np.nonzero(sel)[0].astype(np.int64)


def record_groups(
    store: PreparedStore043,
    rec: npt.NDArray[np.int64],
    local_dataset: npt.NDArray[np.int64],
    datasets: list[str],
    by: Literal["dataset", "profile"],
) -> tuple[npt.NDArray[np.int64], list[str]]:
    """Group id of each arm record and the group names: the arm-local dataset (legacy),
    or the record's experiment-ontology phenotype profile (prep_v4; no dataset name).
    """
    if by == "dataset":
        return local_dataset.astype(np.int64).copy(), list(datasets)
    if store.ontology_vocabulary is None or not store.rec_onto.size:
        raise ValueError(
            f"grouping by profile needs a prepared store with the ontology annotation "
            f"({sorted(ONTOLOGY_PREP_VERSIONS)}); {store.meta.prep_version} has none"
        )
    keys = profile_keys(store.ontology_vocabulary, store.onto_profiles)
    rk = [keys[i] for i in store.rec_onto[rec]]
    names = sorted(set(rk))
    index = {n: i for i, n in enumerate(names)}
    return np.asarray([index[k] for k in rk], np.int64), names


def build_view(
    store: PreparedStore043,
    arm: ArmConfig,
    mark_cap: int | None,
    target_transform: Literal["zscore", "quantile_normal", "none"] = "zscore",
    target_stats_by: Literal["dataset", "profile"] = "dataset",
    weighting_by: Literal["dataset", "profile"] = "dataset",
) -> TrainingView:
    """Select the arm, fit the standardization on its training rows, compact the keys.

    ``target_transform`` / ``target_stats_by`` other than zscore / dataset add the
    second-stage training targets ``T`` (torchcell.data.target_transform_043), fitted on
    the training rows of each group; ``weighting_by = profile`` maps each arm dataset to
    its phenotype profile for the loss weighting.
    """
    shorts = store.meta.datasets
    wanted = list(arm.datasets) + (["sameith_dm"] if arm.doubles else [])
    sel = np.isin(store.rec_dataset, [shorts.index(d) for d in wanted])
    rec = np.nonzero(sel)[0].astype(np.int64)
    rec_short = [shorts[d] for d in store.rec_dataset[rec]]
    local_dataset = np.asarray(
        [arm.datasets.index(SPECS[s].read_as or s) for s in rec_short], np.int64
    )
    head_dataset = local_dataset.copy()
    level_gene = np.asarray([SPECS[s].level == "gene" for s in rec_short], bool)
    role = store.rec_role[rec]
    exp_type = np.asarray(
        [EXP_TYPE_CLASSES.index(SPECS[s].exp_class) for s in rec_short], np.int64
    )
    # ---- standardization on training rows
    fit_rows = []
    train = role == ROLE_CODE[Role.TRAIN]
    for d in arm.datasets:
        spec = SPECS[d]
        rows = rec[train & np.asarray([s == d for s in rec_short], bool)]
        if spec.level == "gene":
            y = store.Y[store.rec_yrow[rows]].astype(np.float64)
            if spec.transform == "log2_cpm_plus_one":
                detected = (y > 0).any(axis=0)
                y[:, ~detected] = np.nan  # never detected in training: not a target
        else:
            y = store.rec_yscalar[rows].astype(np.float64).reshape(-1, 1)
        fit_rows.append((d, y, spec.center_per_gene, spec.transform))
    stats = StandardizationStats.fit(
        store.meta.split_version, store.meta.split_store_sha256, fit_rows
    )
    Z = np.full_like(store.Y, np.nan)
    z_scalar = np.full(store.rec_yscalar.shape, np.nan, np.float32)
    train_mean: dict[str, np.ndarray] = {}
    for d in wanted:
        spec = SPECS[d]
        st = stats.get(spec.read_as or d)
        rows = rec[np.asarray([s == d for s in rec_short], bool)]
        if spec.level == "gene":
            yr = store.rec_yrow[rows]
            Z[yr] = st.apply(store.Y[yr])
        else:
            z_scalar[rows] = st.apply(store.rec_yscalar[rows].reshape(-1, 1))[:, 0]
    for d in arm.datasets:
        if SPECS[d].level != "gene":
            continue
        rows = rec[train & np.asarray([s == d for s in rec_short], bool)]
        with np.errstate(invalid="ignore"):
            zz = Z[store.rec_yrow[rows]].astype(np.float64)
            n = np.isfinite(zz).sum(0)
            train_mean[d] = np.where(n > 0, np.nansum(zz, 0) / np.maximum(n, 1), np.nan)
    # ---- compact keys: only the rows the arm's records use
    ctx_used = np.unique(store.rec_context[rec])
    m_sel = (
        np.concatenate(
            [np.arange(store.mark_ptr[r], store.mark_ptr[r + 1]) for r in rec]
        ).astype(np.int64)
        if len(rec)
        else np.zeros(0, np.int64)
    )
    used_ids: list[np.ndarray] = []
    ctx_rows = np.full(store.ctx_keys.shape, -1, np.int32)
    mark_rows = np.full(store.mark_keys.shape, -1, np.int32)
    for e in range(len(ENCODERS)):
        ck = store.ctx_keys[ctx_used, :, e].ravel()
        mk = store.mark_keys[m_sel, e]
        ids = np.unique(np.concatenate([ck[ck >= 0], mk[mk >= 0]])).astype(np.int64)
        used_ids.append(ids)
        for arr_in, arr_out, sub in (
            (store.ctx_keys, ctx_rows, ctx_used),
            (store.mark_keys, mark_rows, m_sel),
        ):
            v = arr_in[sub, ..., e]
            out = np.where(v >= 0, np.searchsorted(ids, v), -1).astype(np.int32)
            arr_out[sub, ..., e] = out
    view = TrainingView(
        store=store,
        datasets=list(arm.datasets),
        rec=rec,
        local_dataset=local_dataset,
        head_dataset=head_dataset,
        level_gene=level_gene,
        role=role,
        exp_type=exp_type,
        Z=Z,
        z_scalar=z_scalar,
        train_mean_z=train_mean,
        stats=stats,
        ctx_rows=ctx_rows,
        mark_rows=mark_rows,
        used_ids=used_ids,
        mark_cap=mark_cap,
        train_mask=np.asarray(
            [arm.datasets[d] not in arm.exclude_from_training for d in local_dataset],
            bool,
        ),
    )
    if arm.doubles:
        _attach_doubles(view, rec_short)
    _attach_targets(view, rec_short, train, target_transform, target_stats_by)
    if weighting_by == "profile":
        wg, _ = record_groups(store, rec, local_dataset, view.datasets, "profile")
        per_ds = []
        for i, d in enumerate(view.datasets):
            u = np.unique(wg[local_dataset == i])
            if len(u) != 1:
                raise ValueError(
                    f"weighting by profile needs one profile per dataset; {d} spans "
                    f"{len(u)}"
                )
            per_ds.append(int(u[0]))
        view.weight_group = np.asarray(per_ds, np.int64)
    return view


def _attach_targets(
    view: TrainingView,
    rec_short: list[str],
    train: npt.NDArray[np.bool_],
    kind: Literal["zscore", "quantile_normal", "none"],
    by: Literal["dataset", "profile"],
) -> None:
    """Groups, per-record dataset statistics and (unless legacy) the targets ``T``."""
    store = view.store
    view.group, view.group_names = record_groups(
        store, view.rec, view.local_dataset, view.datasets, by
    )
    st = [view.stats.get(SPECS[s].read_as or s) for s in rec_short]
    view.row_mu = np.asarray([x.mu for x in st], np.float64)
    view.row_sd = np.asarray([x.sd for x in st], np.float64)
    if kind == "zscore" and by == "dataset":
        return  # the legacy z, untouched
    gsel = view.level_gene
    yrow = store.rec_yrow[view.rec]

    def u_of(rows: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        g = rows[gsel[rows]]
        r = rows[~gsel[rows]]
        ug = view.row_mu[g, None] + view.row_sd[g, None] * view.Z[yrow[g]]
        ur = view.row_mu[r] + view.row_sd[r] * view.z_scalar[view.rec[r]]
        return np.concatenate([ug.ravel(), ur.astype(np.float64)])

    maps = []
    for gi, name in enumerate(view.group_names):
        rows = np.nonzero(train & (view.group == gi))[0]
        maps.append(fit_group_map(name, kind, u_of(rows)))
    tmap = TargetTransform043(kind=kind, by=by, groups=maps)
    T = np.full_like(view.Z, np.nan)
    t_scalar = np.full_like(view.z_scalar, np.nan)
    g = np.nonzero(gsel)[0]
    r = np.nonzero(~gsel)[0]
    T[yrow[g]] = tmap.z_to_t(
        view.Z[yrow[g]], view.group[g], view.row_mu[g], view.row_sd[g]
    )
    t_scalar[view.rec[r]] = tmap.z_to_t(
        view.z_scalar[view.rec[r]], view.group[r], view.row_mu[r], view.row_sd[r]
    )
    view.target_map, view.T, view.t_scalar = tmap, T, t_scalar


def fit_view_bins(
    view: TrainingView, train_pos: npt.NDArray[np.int64], num_bins: int
) -> list[QuantileBins]:
    """binned_quantile: equal-mass bins per target group over the TRAINING targets."""
    yrow = view.store.rec_yrow[view.rec]
    out = []
    for gi, name in enumerate(view.group_names):
        pos = train_pos[view.group[train_pos] == gi]
        g = pos[view.level_gene[pos]]
        r = pos[~view.level_gene[pos]]
        vals = np.concatenate(
            [
                view.gene_targets[yrow[g]].ravel(),
                view.record_targets[view.rec[r]].ravel(),
            ]
        )
        out.append(fit_quantile_bins(name, vals, num_bins))
    return out


def _attach_doubles(view: TrainingView, rec_short: list[str]) -> None:
    """Split-B positions and the Kemmeren single of each of their two genes."""
    store = view.store
    kem = {
        int(store.rec_genes2[view.rec[i], 0]): i
        for i, s in enumerate(rec_short)
        if s == "kemmeren"
    }
    dbl, singles = [], []
    for i, s in enumerate(rec_short):
        if s != "sameith_dm":
            continue
        a, b = (int(x) for x in store.rec_genes2[view.rec[i]])
        if a in kem and b in kem:
            dbl.append(i)
            singles.append((kem[a], kem[b]))
    view.doubles = np.asarray(dbl, np.int64)
    view.doubles_singles = np.asarray(singles, np.int64).reshape(-1, 2)


# =========================================================================== batches
@dataclass(frozen=True)
class Targets043:
    """Targets of one batch, in z units."""

    gene_rows: torch.Tensor  # [Bg] positions in the batch
    gene_z: torch.Tensor  # [Bg, G]
    record_rows: torch.Tensor  # [Br]
    record_z: torch.Tensor  # [Br]
    dataset_ids: torch.Tensor  # [B] arm-local dataset of each row (loss bookkeeping)
    groups: torch.Tensor | None = None  # [B] target-statistics group of each row


def make_batch(
    view: TrainingView, positions: Sequence[int]
) -> tuple[Batch043, Targets043]:
    """Collate arm positions into the model batch and the targets."""
    store = view.store
    pos = np.asarray(positions, np.int64)
    rec = view.rec[pos]
    ctx_u, ctx_inv = np.unique(store.rec_context[rec], return_inverse=True)
    contexts = ContextInputs(
        key_rows=torch.from_numpy(view.ctx_rows[ctx_u].astype(np.int64)),
        key_gap=torch.from_numpy(store.ctx_gaps[ctx_u].astype(np.int64)),
        chrom_copy=torch.from_numpy(store.ctx_chrom_copy[ctx_u]),
        gene_copy=torch.from_numpy(store.ctx_gene_copy[ctx_u]),
        position_gap=torch.from_numpy(store.ctx_pgap[ctx_u].astype(np.int64)),
    )
    picks = []
    for r in rec:
        lo, hi = int(store.mark_ptr[r]), int(store.mark_ptr[r + 1])
        keep = select_marks(
            store.mark_change[lo:hi], view.mark_cap, int(store.rec_raw_index[r])
        )
        picks.append(lo + keep)
    B, M = len(rec), max([len(p) for p in picks] + [0])

    def pad(
        src: np.ndarray, fill: float | int, dtype: Any, width: int = 0
    ) -> torch.Tensor:
        shape = (B, M, width) if width else (B, M)
        out = np.full(shape, fill, dtype=dtype)
        for b, p in enumerate(picks):
            out[b, : len(p)] = src[p]
        return torch.from_numpy(out)

    valid = np.zeros((B, M), bool)
    for b, p in enumerate(picks):
        valid[b, : len(p)] = True
    marks = MarkInputs(
        valid=torch.from_numpy(valid),
        on_graph=pad(store.mark_on_graph, False, bool),
        gene=pad(store.mark_gene, 0, np.int64),
        change=pad(store.mark_change, 0, np.int64),
        parent=pad(store.mark_parent, 0, np.int64),
        key_rows=pad(view.mark_rows, -1, np.int64, 4),
        key_gap=pad(store.mark_gaps, 0, np.int64, 4),
        replicon=pad(store.mark_replicon, 0, np.int64),
        rel_midpoint=pad(store.mark_relmid, 0.0, np.float32),
        chrom_copy=pad(store.mark_chrom_copy, 1.0, np.float32),
        gene_copy=pad(store.mark_gene_copy, 1.0, np.float32),
        position_gap=pad(store.mark_pgap, 0, np.int64),
    )
    batch = Batch043(
        contexts=contexts,
        record_context=torch.from_numpy(ctx_inv.astype(np.int64)),
        marks=marks,
        dataset_ids=torch.from_numpy(view.head_dataset[pos]),
        exp_type_ids=torch.from_numpy(view.exp_type[pos]),
        env_ids=torch.from_numpy(store.rec_env[rec].astype(np.int64)),
        ontology=torch.from_numpy(store.onto_profiles[store.rec_onto[rec]])
        if store.rec_onto.size
        else None,
    )
    g = np.nonzero(view.level_gene[pos])[0]
    rr = np.nonzero(~view.level_gene[pos])[0]
    G = store.Y.shape[1]
    targets = Targets043(
        gene_rows=torch.from_numpy(g.astype(np.int64)),
        gene_z=torch.from_numpy(
            view.gene_targets[store.rec_yrow[rec[g]]]
            if len(g)
            else np.zeros((0, G), np.float32)
        ),
        record_rows=torch.from_numpy(rr.astype(np.int64)),
        record_z=torch.from_numpy(view.record_targets[rec[rr]]),
        dataset_ids=torch.from_numpy(view.local_dataset[pos]),
        groups=torch.from_numpy(view.group[pos]) if view.group.size else None,
    )
    return batch, targets


class _BatchList(torch.utils.data.Dataset[tuple[Batch043, Targets043]]):
    """A list of position lists, collated in workers."""

    def __init__(
        self, view: TrainingView, batches: list[npt.NDArray[np.int64]]
    ) -> None:
        self.view = view
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, i: int) -> tuple[Batch043, Targets043]:
        return make_batch(self.view, self.batches[i].tolist())


def balanced_batches(
    view: TrainingView, positions: npt.NDArray[np.int64], batch_size: int, seed: int
) -> list[npt.NDArray[np.int64]]:
    """One epoch of dataset-balanced batches over ``positions``.

    The epoch has as many batches as the uniform sampler would
    (``ceil(len(positions) / batch_size)``); every slot of every batch draws a dataset
    uniformly among the datasets with a position, then a position uniformly within that
    dataset (with replacement), from a generator seeded by ``seed``. Small datasets are
    revisited within an epoch and large ones are subsampled.
    """
    by_dataset = [
        positions[view.local_dataset[positions] == d] for d in range(len(view.datasets))
    ]
    pools = [p for p in by_dataset if len(p)]
    if not pools:
        raise ValueError("no training positions to sample")
    rng = np.random.default_rng(seed)
    n_batches = math.ceil(len(positions) / batch_size)
    which = rng.integers(0, len(pools), size=(n_batches, batch_size))
    out = np.empty((n_batches, batch_size), np.int64)
    for k, pool in enumerate(pools):
        sel = which == k
        out[sel] = pool[rng.integers(0, len(pool), size=int(sel.sum()))]
    return list(out)


def batch_loader(
    view: TrainingView,
    positions: npt.NDArray[np.int64],
    batch_size: int,
    shuffle_seed: int | None,
    num_workers: int,
    sampler: Literal["records", "balanced"] = "records",
) -> torch.utils.data.DataLoader[tuple[Batch043, Targets043]]:
    """Batches of ``positions``: in order, shuffled (``records``) or ``balanced``."""
    if sampler == "balanced":
        if shuffle_seed is None:
            raise ValueError("the balanced sampler needs a seed")
        chunks = balanced_batches(view, positions, batch_size, shuffle_seed)
    else:
        order = positions.copy()
        if shuffle_seed is not None:
            np.random.default_rng(shuffle_seed).shuffle(order)
        chunks = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    return torch.utils.data.DataLoader(
        _BatchList(view, chunks),
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=torch.cuda.is_available(),
    )


# =========================================================================== metrics
def column_pearson(
    P: npt.NDArray[np.floating], Y: npt.NDArray[np.floating], min_n: int
) -> npt.NDArray[np.float64]:
    """Pearson per column over rows where both are finite; NaN when undefined."""
    P = np.asarray(P, np.float64)
    Y = np.asarray(Y, np.float64)
    m = np.isfinite(P) & np.isfinite(Y)
    n = m.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mp = np.where(m, P, 0).sum(0) / n
        my = np.where(m, Y, 0).sum(0) / n
        dp = np.where(m, P - mp, 0)
        dy = np.where(m, Y - my, 0)
        cov = (dp * dy).sum(0)
        vp = (dp * dp).sum(0)
        vy = (dy * dy).sum(0)
        r = cov / np.sqrt(vp * vy)
    ok = (n >= min_n) & (vp > 1e-12) & (vy > 1e-12)
    return np.where(ok, r, np.nan)


def column_spearman(
    P: npt.NDArray[np.floating], Y: npt.NDArray[np.floating], min_n: int
) -> npt.NDArray[np.float64]:
    """Spearman per column (average ranks over the jointly finite rows)."""
    P = np.asarray(P, np.float64)
    Y = np.asarray(Y, np.float64)
    m = np.isfinite(P) & np.isfinite(Y)
    Pr = pd.DataFrame(np.where(m, P, np.nan)).rank(axis=0).to_numpy()
    Yr = pd.DataFrame(np.where(m, Y, np.nan)).rank(axis=0).to_numpy()
    return column_pearson(Pr, Yr, min_n)


def _nanmedian(x: npt.NDArray[np.float64]) -> float:
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def gene_level_metrics(
    P: npt.NDArray[np.floating],
    Y: npt.NDArray[np.floating],
    train_mean: npt.NDArray[np.float64],
    min_strains: int,
) -> dict[str, float]:
    """Per-gene Pearson/Spearman medians and the centered per-strain Pearson median."""
    out = {
        "pearson_per_gene_median": _nanmedian(column_pearson(P, Y, min_strains)),
        "spearman_median": _nanmedian(column_spearman(P, Y, min_strains)),
        "pearson_per_strain_centered_median": _nanmedian(
            column_pearson((P - train_mean).T, (Y - train_mean).T, 3)
        ),
        "n_strains": float(P.shape[0]),
    }
    return out


# =========================================================================== logging
class RunLogger(Protocol):
    """Where metrics go."""

    def log(self, values: dict[str, float], step: int) -> None:
        """Record ``values`` at ``step``."""

    def finish(self) -> None:
        """Close the run."""


class MemoryLogger:
    """Keeps every logged dict (tests)."""

    def __init__(self) -> None:
        """An empty record."""
        self.rows: list[tuple[int, dict[str, float]]] = []

    def log(self, values: dict[str, float], step: int) -> None:
        """Append ``values``."""
        self.rows.append((step, dict(values)))

    def finish(self) -> None:
        """Nothing to close."""
        return None


class WandbLogger:
    """``wandb.log``."""

    def __init__(self, cfg: TrainConfig043, config_dump: dict[str, Any]) -> None:
        """``wandb.init`` with group = arm label and name = run name."""
        import wandb

        self.run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.arm_label,
            name=cfg.run_name,
            mode=cfg.wandb.mode,
            tags=list(cfg.wandb.tags),
            config=config_dump,
            id=cfg.resume_id,
            resume="allow" if cfg.resume_id is not None else None,
        )

    def log(self, values: dict[str, float], step: int) -> None:
        """``run.log``."""
        self.run.log(values, step=step)

    def finish(self) -> None:
        """``run.finish``."""
        self.run.finish()


# =========================================================================== training
VectorSource = Callable[[int, npt.NDArray[np.int64]], torch.Tensor]


def h5_vector_source(paths: PathsConfig, generation: StorePaths043) -> VectorSource:
    """Vocabulary ids of encoder ``e`` -> their vectors from ``alleles/<encoder>.h5``."""

    def read(e: int, ids: npt.NDArray[np.int64]) -> torch.Tensor:
        enc = ENCODERS[e]
        vocab = np.load(
            osp.join(paths.resolved_representation_root(generation), f"vocab_{enc}.npy")
        )
        return load_vector_table(
            osp.join(paths.resolved_alleles_root(), f"{enc}.h5"), vocab[ids]
        )

    return read


@dataclass
class RunResult:
    """What a training run returns."""

    n_epochs: int
    steps: int
    epoch_seconds: list[float]
    stats_sha256: str
    window: tuple[int, int]
    last_metrics: dict[str, float]
    checkpoint_dir: str


def _device(cfg: TrainConfig043) -> torch.device:
    if cfg.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")
    return torch.device(cfg.device)


#: Parameters indexed by the arm-local dataset id: remapped by dataset name on init_from.
DATASET_INDEXED = ("dataset_token.weight", "gene_bias", "record_bias")


def checkpoint_config(state: dict[str, Any]) -> TrainConfig043:
    """The run config stored in a checkpoint.

    Checkpoints written between a054414d7 and the ReZero removal carry
    ``model.operator_residual``; ``plain`` is today's only operator and the key is dropped,
    a ReZero checkpoint (beta gates) is refused.
    """
    raw = dict(state["config"])
    model = dict(raw.get("model", {}))
    residual = model.pop("operator_residual", "plain")
    if residual != "plain":
        raise ValueError(
            f"checkpoint operator residual {residual!r}: only plain exists"
        )
    raw["model"] = model
    return TrainConfig043.model_validate(raw)


def stage1_label(path: str) -> str:
    """The W&B group of the run that wrote the checkpoint at ``path``."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint_config(state).arm_label


def init_from_checkpoint(
    model: CellModel043, cfg: TrainConfig043, view: TrainingView
) -> dict[str, float]:
    """Load the stage-1 MODEL weights of ``cfg.init_from`` (the optimizer is not loaded).

    The model and loss sections must equal the checkpoint's; dataset-indexed rows
    (``DATASET_INDEXED``) are taken by dataset name, so a one-dataset arm reads its own
    row of a twelve-dataset stage 1; every arm dataset must be a stage-1 dataset, and its
    standardization must equal the stage-1 one (same training rows). Loads strictly.
    """
    assert cfg.init_from is not None
    state = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
    src = checkpoint_config(state)
    if src.model != cfg.model:
        raise ValueError(f"model section differs from the checkpoint's: {src.model}")
    if src.loss != cfg.loss:
        raise ValueError(f"loss differs from the checkpoint's: {src.loss}")
    missing = [d for d in view.datasets if d not in src.arm.datasets]
    if missing:
        raise KeyError(f"{missing} not trained in stage 1 ({src.arm.datasets})")
    with open(osp.join(osp.dirname(cfg.init_from), "standardization.json")) as f:
        src_stats = StandardizationStats.model_validate_json(f.read())
    for d in view.datasets:
        if src_stats.get(d) != view.stats.get(d):
            raise ValueError(f"{d}: standardization differs from stage 1")
    _check_stage1_targets(osp.dirname(cfg.init_from), view, cfg)
    rows = torch.as_tensor([src.arm.datasets.index(d) for d in view.datasets])
    sd = dict(state["model"])
    for k in DATASET_INDEXED:
        if k in sd:  # absent under ontology conditioning
            sd[k] = sd[k][rows]
    model.load_state_dict(sd, strict=True)
    return {
        "setup/init_from_epoch": float(state["epoch"]),
        "setup/init_from_step": float(state["step"]),
        "setup/init_from_lr": float(src.optim.lr),
    }


def _check_stage1_targets(
    src_dir: str, view: TrainingView, cfg: TrainConfig043
) -> None:
    """The fine-tune's target maps and quantile bins equal stage 1's for its groups."""
    path = osp.join(src_dir, TARGET_TRANSFORM_JSON)
    if osp.exists(path) != (view.target_map is not None):
        raise ValueError("target_transform differs from stage 1 (one is the legacy z)")
    if view.target_map is not None:
        with open(path) as f:
            src_map = TargetTransform043.model_validate_json(f.read())
        have = src_map.by_name()
        for g in view.target_map.groups:
            if have.get(g.group) != g:
                raise ValueError(f"{g.group}: target map differs from stage 1")
    if cfg.loss.kind == "binned_quantile":
        with open(osp.join(src_dir, QUANTILE_BINS_JSON)) as f:
            src_bins = {
                b["group"]: QuantileBins.model_validate(b) for b in json.load(f)
            }
        train_pos = view.positions(ROLE_CODE[Role.TRAIN])
        for b in fit_view_bins(view, train_pos, cfg.loss.hl_gauss.num_bins):
            if src_bins.get(b.group) != b:
                raise ValueError(f"{b.group}: quantile bins differ from stage 1")


def mark_cap_report(view: TrainingView) -> tuple[str, dict[str, float]]:
    """Marks per record over the arm: the maximum, and what ``view.mark_cap`` drops."""
    store = view.store
    n = (store.mark_ptr[view.rec + 1] - store.mark_ptr[view.rec]).astype(np.int64)
    top = int(np.argmax(n)) if len(n) else 0
    mx = int(n.max(initial=0))
    cap = view.mark_cap
    over = n > cap if cap is not None else np.zeros(len(n), bool)
    dropped = int((n[over] - cap).sum()) if cap is not None else 0
    by_ds = defaultdict(int)
    for i in np.nonzero(over)[0]:
        by_ds[view.datasets[view.local_dataset[i]]] += 1
    line = (
        f"[setup] marks per record: max {mx} ({view.datasets[view.local_dataset[top]] if len(n) else '-'}); "
        f"cap {cap if cap is not None else 'none'}: {int(over.sum())} of {len(n)} records "
        f"truncated {dict(by_ds)}, {dropped} marks dropped"
    )
    return line, {
        "setup/marks_max": float(mx),
        "setup/mark_cap": float(cap or 0),
        "setup/records_truncated": float(over.sum()),
        "setup/marks_dropped": float(dropped),
    }


def dataset_record_counts(
    view: TrainingView,
    train_pos: npt.NDArray[np.int64],
    val_pos: npt.NDArray[np.int64],
    te_pos: npt.NDArray[np.int64],
) -> list[str]:
    """``<dataset> <train>/<val>/<traineval>`` per arm dataset, plus split B and the
    sentinel cells the prepare step set missing (``sentinel_masked_cells``).
    """
    out = []
    for i, d in enumerate(view.datasets):
        n = [
            int((view.local_dataset[p] == i).sum())
            for p in (train_pos, val_pos, te_pos)
        ]
        item = f"{d} {n[0]}/{n[1]}/{n[2]}"
        masked = view.store.meta.counts.get(d, {}).get("sentinel_masked_cells")
        if masked:
            item += (
                f" ({masked} sentinel cells |v| >= {SPECS[d].sentinel_abs:g} masked)"
            )
        out.append(item)
    if len(view.doubles):
        out.append(f"sameith_dm split B {len(view.doubles)}")
    return out


def build_model(
    cfg: TrainConfig043, view: TrainingView, tables: list[torch.Tensor]
) -> CellModel043:
    """The model for the arm's vocabularies."""
    store = view.store
    mc = CellModel043Config(
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        mask_layers=tuple(cfg.model.mask_layers),
        head_graphs=dict(cfg.model.head_graphs),
        fourier_frequencies=cfg.model.fourier_frequencies,
        operator_ffn_mult=cfg.model.operator_ffn_mult,
        trunk_ffn_mult=cfg.model.trunk_ffn_mult,
        n_parents=int(store.mark_parent.max(initial=0)) + 1
        if store.mark_parent.size
        else 1,
        n_datasets=len(view.datasets),
        n_envs=len(store.meta.env_keys),
        exp_type_classes=EXP_TYPE_CLASSES,
        exp_type_mode=cfg.model.exp_type_mode,
        exp_type_paths=EXP_TYPE_PATHS if cfg.model.exp_type_mode == "dag" else None,
        type_token_dim=cfg.model.type_token_dim,
        env_token_dim=cfg.model.env_token_dim,
        dataset_token_dim=cfg.model.dataset_token_dim,
        conditioning=cfg.model.conditioning.source,
        ontology_state_dim=cfg.model.conditioning.state_dim,
        head_hidden=cfg.model.head_hidden,
        loss=cfg.loss.kind,
        hl_gauss=cfg.loss.hl_gauss,
        num_quantiles=cfg.loss.num_quantiles,
        attention_query_chunk=cfg.attention_query_chunk,
    )
    G = len(store.meta.genes)
    head_mask = (
        build_head_mask(
            {k: torch.from_numpy(v) for k, v in store.edges.items()},
            mc.head_graphs,
            mc.num_heads,
            G,
        )
        if mc.mask_layers
        else None
    )
    return CellModel043(
        mc,
        tables,
        chrom_index=torch.from_numpy(store.chrom_index.astype(np.int64)),
        rel_midpoint=torch.from_numpy(store.rel_midpoint),
        head_mask=head_mask,
        encoder_dims=[int(t.shape[1]) for t in tables],
        ontology_vocabulary=_ontology_vocabulary(cfg, store),
    )


def _ontology_vocabulary(
    cfg: TrainConfig043, store: PreparedStore043
) -> ExperimentOntologyVocabulary | None:
    """The prep_v4 vocabulary under ontology conditioning (raises without one)."""
    if cfg.model.conditioning.source != "ontology":
        return None
    if store.ontology_vocabulary is None or not store.rec_onto.size:
        raise ValueError(
            f"ontology conditioning needs a prepared store with the annotation "
            f"({sorted(ONTOLOGY_PREP_VERSIONS)}); {store.meta.prep_version} has none"
        )
    return store.ontology_vocabulary


def batch_loss(
    out_gene: torch.Tensor,
    out_record: torch.Tensor,
    targets: Targets043,
    objective: ReadoutLossFamily043,
    n_datasets: int,
    weighting: Literal["equal", "records"],
    dataset_group: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[int, float]]:
    """Weighted per-dataset mean loss of a batch, and each dataset's mean (for logs)."""
    sums, counts, nrec = loss_sums(out_gene, out_record, targets, objective, n_datasets)
    total, means = combine_loss(sums, counts, nrec, weighting, dataset_group)
    if total is None:
        total = (out_gene.sum() + out_record.sum()) * 0.0
    present = counts > 0
    detached = means.detach()
    logs = {
        int(i): float(detached[i]) for i in torch.nonzero(present).flatten().tolist()
    }
    return total, logs


def loss_sums(
    out_gene: torch.Tensor,
    out_record: torch.Tensor,
    targets: Targets043,
    objective: ReadoutLossFamily043,
    n_datasets: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per dataset: summed elementwise loss, measured entries, records (one batch)."""
    device = out_gene.device
    sums = torch.zeros(n_datasets, device=device)
    counts = torch.zeros(n_datasets, device=device)
    nrec = torch.zeros(n_datasets, device=device)
    ds = targets.dataset_ids.to(device)
    grp = None if targets.groups is None else targets.groups.to(device)
    if targets.gene_rows.numel():
        elem, measured = objective.elementwise(
            out_gene,
            targets.gene_z.to(device),
            None if grp is None else grp[targets.gene_rows.to(device)],
        )
        d = ds[targets.gene_rows.to(device)]
        sums = sums.index_add(0, d, elem.sum(1))
        counts = counts.index_add(0, d, measured.sum(1).to(sums.dtype))
        nrec = nrec.index_add(0, d, torch.ones_like(d, dtype=sums.dtype))
    if targets.record_rows.numel():
        elem, measured = objective.elementwise(
            out_record,
            targets.record_z.to(device),
            None if grp is None else grp[targets.record_rows.to(device)],
        )
        d = ds[targets.record_rows.to(device)]
        sums = sums.index_add(0, d, elem)
        counts = counts.index_add(0, d, measured.to(sums.dtype))
        nrec = nrec.index_add(0, d, torch.ones_like(d, dtype=sums.dtype))
    return sums, counts, nrec


def combine_loss(
    sums: torch.Tensor,
    counts: torch.Tensor,
    nrec: torch.Tensor,
    weighting: Literal["equal", "records"],
    dataset_group: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """The objective from per-dataset sums: equal = mean of the dataset means, records =
    their mean weighted by records; None when no dataset has a measured entry.

    With ``dataset_group`` (the phenotype profile of each dataset, ``weighting.by =
    profile``) the sums pool per profile first and the weighting balances profiles;
    the returned means stay per dataset (the logged ``<dataset>/loss``).
    """
    present = counts > 0
    means = torch.where(present, sums / counts.clamp_min(1.0), torch.zeros_like(sums))
    if not bool(present.any()):
        return None, means
    if dataset_group is not None:
        dg = dataset_group.to(sums.device)
        n_g = int(dg.max()) + 1
        gs = sums.new_zeros(n_g).index_add(0, dg, sums)
        gc = counts.new_zeros(n_g).index_add(0, dg, counts)
        gn = nrec.new_zeros(n_g).index_add(0, dg, nrec)
        total, _ = combine_loss(gs, gc, gn, weighting)
        return total, means
    if weighting == "equal":
        return means[present].mean(), means
    w = nrec * present
    return (means * w).sum() / w.sum(), means


@torch.no_grad()
def predict(
    model: CellModel043,
    objective: ReadoutLossFamily043,
    view: TrainingView,
    positions: npt.NDArray[np.int64],
    batch_size: int,
    device: torch.device,
    num_workers: int,
    weighting: Literal["equal", "records"] | None = None,
) -> dict[str, Any]:
    """Point predictions (z) for ``positions``, with the HL-Gauss readouts.

    With ``weighting``, also the training objective over the whole pass (``loss``: the
    per-dataset sums of every batch combined as ``batch_loss`` combines one batch) and
    each dataset's mean loss (``dataset_loss``).
    """
    model.eval()
    G = view.store.Y.shape[1]
    n_ds = len(view.datasets)
    acc_sums = torch.zeros(n_ds, dtype=torch.float64)
    acc_counts = torch.zeros(n_ds, dtype=torch.float64)
    acc_nrec = torch.zeros(n_ds, dtype=torch.float64)
    gene_pred = np.full((len(positions), G), np.nan, np.float32)
    rec_pred = np.full(len(positions), np.nan, np.float32)
    acc_hits = acc_n = 0.0
    ent_sum = ent_n = 0.0
    op_sums: dict[str, float] = defaultdict(float)
    op_n = 0
    cursor = 0
    for batch, tg in batch_loader(view, positions, batch_size, None, num_workers):
        b = batch.to(device)
        with _autocast(device):
            out = model(b, tg.gene_rows.to(device), tg.record_rows.to(device))
        B = len(tg.dataset_ids)
        for k, v in model.operator.last_norms.items():
            op_sums[k] += float(v) * B
        op_n += B
        if weighting is not None:
            s_, c_, n_ = loss_sums(
                out.gene.float(), out.record.float(), tg, objective, n_ds
            )
            acc_sums += s_.double().cpu()
            acc_counts += c_.double().cpu()
            acc_nrec += n_.double().cpu()
        grp = None if tg.groups is None else tg.groups.to(device)
        if tg.gene_rows.numel():
            gene_pred[cursor + tg.gene_rows.numpy()] = (
                objective.point(
                    out.gene, None if grp is None else grp[tg.gene_rows.to(device)]
                )
                .cpu()
                .numpy()
            )
        if tg.record_rows.numel():
            rec_pred[cursor + tg.record_rows.numpy()] = (
                objective.point(
                    out.record, None if grp is None else grp[tg.record_rows.to(device)]
                )
                .cpu()
                .numpy()
            )
        if objective.kind == "hlgauss" and tg.gene_rows.numel():
            z = tg.gene_z.to(device)
            hits, meas = objective.hl.accuracy_within_one(out.gene, z)
            acc_hits += float(hits.sum())
            acc_n += float(meas.sum())
            ent = objective.hl.entropy(out.gene)
            ent_sum += float((ent * meas).sum())
            ent_n += float(meas.sum())
        cursor += B
    model.train()
    if view.target_map is not None:
        # predictions are in the training targets t; score them as legacy z
        gp, mu, sd = (
            view.group[positions],
            view.row_mu[positions],
            view.row_sd[positions],
        )
        gene_pred = view.target_map.t_to_z(gene_pred, gp, mu, sd)
        rec_pred = view.target_map.t_to_z(rec_pred, gp, mu, sd)
    extra = {}
    dataset_loss: dict[int, float] = {}
    loss: float | None = None
    if weighting is not None:
        total, means = combine_loss(
            acc_sums,
            acc_counts,
            acc_nrec,
            weighting,
            None if view.weight_group is None else torch.from_numpy(view.weight_group),
        )
        loss = None if total is None else float(total)
        dataset_loss = {
            int(i): float(means[i])
            for i in torch.nonzero(acc_counts > 0).flatten().tolist()
        }
    if acc_n:
        extra["acc_within_one_bin"] = acc_hits / acc_n
        extra["entropy_mean"] = ent_sum / ent_n
    return {
        "gene": gene_pred,
        "record": rec_pred,
        "extra": extra,
        "operator": {k: v / op_n for k, v in op_sums.items()} if op_n else {},
        "loss": loss,
        "dataset_loss": dataset_loss,
    }


def _autocast(device: torch.device) -> Any:
    if device.type in ("cuda", "xpu"):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def eval_positions(
    view: TrainingView, role: int, cap: int | None, seed: int
) -> npt.NDArray[np.int64]:
    """Per dataset, the ``role`` positions, a fixed seeded subset of ``cap`` when larger."""
    out = []
    for d in view.datasets:
        p = view.positions(role, d)
        if cap is not None and len(p) > cap:
            p = np.sort(np.random.default_rng(seed).choice(p, cap, replace=False))
        out.append(p)
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def evaluate(
    prefix: str,
    model: CellModel043,
    objective: ReadoutLossFamily043,
    view: TrainingView,
    positions: npt.NDArray[np.int64],
    cfg: TrainConfig043,
    device: torch.device,
) -> dict[str, float]:
    """``<prefix>/<dataset>/...`` and ``<prefix>/<type>/...`` (type = mean over datasets)."""
    pred = predict(
        model,
        objective,
        view,
        positions,
        cfg.data.batch_size,
        device,
        cfg.data.num_workers,
        cfg.weighting.kind,
    )
    store = view.store
    logs: dict[str, float] = {}
    # the training objective on this pass (same loss, same weighting): <prefix>/loss and
    # <prefix>/<dataset>/loss, so train and validation loss share an axis
    if pred["loss"] is not None:
        logs[f"{prefix}/loss"] = pred["loss"]
    for i, v in pred["dataset_loss"].items():
        logs[f"{prefix}/{view.datasets[i]}/loss"] = v
    by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for d in view.datasets:
        spec = SPECS[d]
        sel = view.local_dataset[positions] == view.datasets.index(d)
        if not sel.any():
            continue
        rec = view.rec[positions[sel]]
        if spec.level == "gene":
            Pz = pred["gene"][sel]
            Yz = view.Z[store.rec_yrow[rec]]
            m = gene_level_metrics(
                Pz, Yz, view.train_mean_z[d], cfg.data.min_strains_per_gene
            )
        else:
            P = pred["record"][sel].astype(np.float64)
            Y = view.z_scalar[rec].astype(np.float64)
            m = {
                "pearson": float(column_pearson(P[:, None], Y[:, None], 3)[0]),
                "spearman": float(column_spearman(P[:, None], Y[:, None], 3)[0]),
                "n_strains": float(sel.sum()),
            }
        for k, v in m.items():
            logs[f"{prefix}/{d}/{k}"] = v
            if k != "n_strains":
                by_type[spec.metric_type][k].append(v)
    for t, metrics in by_type.items():
        for k, vals in metrics.items():
            arr = np.asarray(vals, np.float64)
            arr = arr[np.isfinite(arr)]
            logs[f"{prefix}/{t}/{k}"] = float(arr.mean()) if arr.size else float("nan")
    for k, v in pred["extra"].items():
        logs[f"{prefix}/{k}"] = v
    if prefix == "val":
        # is the operator live: its branches against the context they update, over the
        # validation pass (MarkOperator043.last_norms; record-weighted mean of batches)
        names = {
            "attn": "relative_norm",
            "ffn": "ffn_relative_norm",
            "update": "update_relative_norm",
        }
        for k, v in pred["operator"].items():
            logs[f"operator/{names[k]}"] = v
    return logs


def evaluate_doubles(
    model: CellModel043,
    objective: ReadoutLossFamily043,
    view: TrainingView,
    cfg: TrainConfig043,
    device: torch.device,
) -> dict[str, float]:
    """Split B: the Sameith doubles against the additive null, in released units.

    Centered on Kemmeren's training-mean profile, per double, over all genes. Logged as
    ``val/doubles/...`` by request; these records are split B (a test set) and must not
    steer any choice.
    """
    if len(view.doubles) == 0:
        return {}
    singles = view.doubles_singles
    pos = np.concatenate([view.doubles, singles[:, 0], singles[:, 1]])
    pred = predict(
        model, objective, view, pos, cfg.data.batch_size, device, cfg.data.num_workers
    )
    n = len(view.doubles)
    st = view.stats.get("kemmeren")
    store = view.store
    released = st.invert(pred["gene"])
    measured = st.invert(view.Z[store.rec_yrow[view.rec[pos]]])
    mean_profile = st.invert(view.train_mean_z["kemmeren"])
    p_xy, p_x, p_y = released[:n], released[n : 2 * n], released[2 * n :]
    m_xy, m_x, m_y = measured[:n], measured[n : 2 * n], measured[2 * n :]

    def centered(a: np.ndarray) -> float:
        return _nanmedian(
            column_pearson((a - mean_profile).T, (m_xy - mean_profile).T, 3)
        )

    return {
        "val/doubles/pearson_per_strain_centered_median": centered(p_xy),
        "val/doubles/additive_model_pearson_per_strain_centered_median": centered(
            p_x + p_y
        ),
        "val/doubles/additive_measured_pearson_per_strain_centered_median": centered(
            m_x + m_y
        ),
        "val/doubles/n_doubles": float(n),
    }


def _lr_at(step: int, cfg: OptimConfig, total_steps: int | None = None) -> float:
    """Linear warmup, then constant or cosine to ``min_lr_frac * lr`` at ``total_steps``."""
    if cfg.warmup_steps and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if cfg.schedule == "constant":
        return cfg.lr
    if total_steps is None:
        raise ValueError("a cosine schedule needs the run's total optimizer steps")
    span = max(total_steps - cfg.warmup_steps, 1)
    prog = min(max(step - cfg.warmup_steps, 0) / span, 1.0)
    floor = cfg.min_lr_frac
    return cfg.lr * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * prog)))


#: Sweep keys added 2026-10-24, with their defaults; a checkpoint written before them
#: holds none, and a resume of it compares equal when every one is at its default.
SWEEP_DEFAULTS: dict[tuple[str, str], Any] = {
    ("model", "layer_pattern"): None,
    ("model", "trunk_ffn_mult"): 4,
    ("data", "target_transform"): "zscore",
    ("data", "target_stats_by"): "dataset",
    ("weighting", "by"): "dataset",
    ("optim", "schedule"): "constant",
    ("optim", "min_lr_frac"): 0.0,
    ("optim", "accumulate_steps"): 1,
    ("budget", "checkpoint_every_steps"): None,
}


def config_matches(stored: dict[str, Any], cfg: TrainConfig043) -> bool:
    """``stored`` (a checkpoint's config dump) equals ``cfg``, reading a sweep key the
    checkpoint predates as its default.
    """
    cur = cfg.model_dump(mode="json")
    for (sec, key), default in SWEEP_DEFAULTS.items():
        if key not in stored.get(sec, {}) and cur[sec][key] == default:
            cur[sec] = {k: v for k, v in cur[sec].items() if k != key}
    return bool(stored == cur)


def train(
    cfg: TrainConfig043,
    store: PreparedStore043,
    vector_source: VectorSource,
    logger: RunLogger,
) -> RunResult:
    """The training loop (module docstring)."""
    t_start = time.time()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = _device(cfg)
    view = build_view(
        store,
        cfg.arm,
        cfg.data.mark_cap,
        cfg.data.target_transform,
        cfg.data.target_stats_by,
        cfg.weighting.by,
    )
    tables = [vector_source(e, view.used_ids[e]) for e in range(len(ENCODERS))]
    model = build_model(cfg, view, tables)
    init_logs: dict[str, float] = {}
    if cfg.init_from is not None:
        init_logs = init_from_checkpoint(model, cfg, view)
        print(
            f"[setup] init_from {cfg.init_from} (stage-1 epoch "
            f"{init_logs['setup/init_from_epoch']:.0f}, step "
            f"{init_logs['setup/init_from_step']:.0f}, stage-1 lr "
            f"{init_logs['setup/init_from_lr']:g}); model weights only, fresh optimizer at "
            f"lr {cfg.optim.lr:g}",
            flush=True,
        )
    model = model.to(device)
    bins: list[QuantileBins] | None = None
    stats_sha = view.stats.sha256()
    table_gb = model.composite.table_bytes() / 1e9
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_pos = view.positions(ROLE_CODE[Role.TRAIN])
    if cfg.arm.exclude_from_training:
        assert view.train_mask is not None
        train_pos = train_pos[view.train_mask[train_pos]]
        print(
            f"[setup] leave-dataset-out: {cfg.arm.exclude_from_training} never trained "
            f"on, evaluated through the ontology conditioning only",
            flush=True,
        )
    val_pos = eval_positions(
        view, ROLE_CODE[Role.VAL], cfg.data.val_max_records_per_dataset, cfg.seed
    )
    te_pos = eval_positions(
        view,
        ROLE_CODE[Role.TRAIN],
        cfg.data.traineval_max_records_per_dataset,
        cfg.seed,
    )
    if cfg.loss.kind == "binned_quantile":
        bins = fit_view_bins(view, train_pos, cfg.loss.hl_gauss.num_bins)
    objective = ReadoutLossFamily043(model.config, bins).to(device)
    # one directory per job, so a rerun of a run name never overwrites an earlier run's
    # checkpoints (the 2026-10-22 runs sit directly in <run_name>/)
    ckpt_dir = osp.join(
        cfg.paths.resolved_checkpoint_root(),
        cfg.run_name,
        cfg.resume_id
        or os.environ.get("SLURM_JOB_ID")
        or os.environ.get("PBS_JOBID")
        or f"local-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}",
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(osp.join(ckpt_dir, "standardization.json"), "w") as f:
        f.write(view.stats.canonical_json())
    if view.target_map is not None:
        with open(osp.join(ckpt_dir, TARGET_TRANSFORM_JSON), "w") as f:
            f.write(view.target_map.canonical_json())
    if bins is not None:
        with open(osp.join(ckpt_dir, QUANTILE_BINS_JSON), "w") as f:
            f.write(
                json.dumps([b.model_dump(mode="json") for b in bins], sort_keys=True)
            )
    print(
        f"[setup] trunk {cfg.model.pattern_label} (S dense, M multigraph-masked; "
        f"{cfg.model.num_heads} heads x {cfg.model.hidden_dim // cfg.model.num_heads} "
        f"dims, trunk FFN {cfg.model.trunk_ffn_mult * cfg.model.hidden_dim}, operator FFN "
        f"{cfg.model.operator_ffn_mult * cfg.model.hidden_dim}, head_hidden "
        f"{cfg.model.head_hidden}); loss {cfg.loss.kind}; targets "
        f"{cfg.data.target_transform} by {cfg.data.target_stats_by}"
        + (
            f" (groups {view.group_names}, sha256 {view.target_map.sha256()})"
            if view.target_map is not None
            else " (legacy z)"
        )
        + f"; weighting {cfg.weighting.kind} by {cfg.weighting.by}; lr "
        f"{cfg.optim.lr:g} warmup {cfg.optim.warmup_steps} {cfg.optim.schedule}",
        flush=True,
    )
    print(
        f"[setup] device {device}; {len(train_pos)} train / {len(val_pos)} val / "
        f"{len(te_pos)} traineval records; {len(view.doubles)} doubles; "
        f"{n_params / 1e6:.2f} M trainable parameters; frozen tables {table_gb:.2f} GB "
        f"({[len(u) for u in view.used_ids]} rows); standardization sha256 {stats_sha}",
        flush=True,
    )
    print(
        "[setup] composite gene token: encoders "
        + ", ".join(
            f"{e} {int(t.shape[1])}-d x {int(t.shape[0])} rows"
            for e, t in zip(ENCODERS, tables, strict=True)
        )
        + f"; conditioning {cfg.model.conditioning.source}"
        + (
            f" ({len(store.ontology_vocabulary)} ontology nodes, "
            f"{int(np.unique(store.rec_onto[view.rec]).size)} profiles in the arm)"
            if cfg.model.conditioning.source == "ontology"
            and store.ontology_vocabulary is not None
            else ""
        ),
        flush=True,
    )
    cap_line, cap_logs = mark_cap_report(view)
    print(cap_line, flush=True)
    print(
        "[setup] records per dataset (train / val / traineval): "
        + "; ".join(dataset_record_counts(view, train_pos, val_pos, te_pos)),
        flush=True,
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    acc = cfg.optim.accumulate_steps
    micro_per_epoch = math.ceil(len(train_pos) / cfg.data.batch_size)
    steps_per_epoch = math.ceil(micro_per_epoch / acc)
    if cfg.budget.max_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, cfg.budget.max_steps_per_epoch)
    max_steps = cfg.budget.max_steps
    n_epochs = (
        cfg.budget.max_epochs
        if max_steps is None
        else math.ceil(max_steps / steps_per_epoch)
    )
    window = cfg.budget.registered_window.epochs(n_epochs)
    total_steps = max_steps if max_steps is not None else steps_per_epoch * n_epochs
    logger.log(
        {
            "setup/max_steps": float(max_steps or 0),
            "setup/n_train": float(len(train_pos)),
            "setup/n_val": float(len(val_pos)),
            "setup/steps_per_epoch": float(steps_per_epoch),
            "setup/table_gb": table_gb,
            "setup/n_params_m": n_params / 1e6,
            "setup/seconds": time.time() - t_start,
            **cap_logs,
            **init_logs,
        },
        step=0,
    )
    step = 0
    epoch_seconds: list[float] = []
    last: dict[str, float] = {}
    epoch = 0
    resume_path = osp.join(ckpt_dir, "last.pt")
    if cfg.resume_id is not None and osp.exists(resume_path):
        st = torch.load(resume_path, map_location=device, weights_only=False)
        if not config_matches(st["config"], cfg):
            raise ValueError(f"{resume_path} was written under another config")
        model.load_state_dict(st["model"], strict=True)
        opt.load_state_dict(st["optimizer"])
        epoch, step = int(st["epoch"]), int(st["step"])
        skip_micro = int(st.get("micro_done", 0))
        partial = st.get("partial")
        print(
            f"[resume] {resume_path}: epoch {epoch}, step {step}"
            + (
                f", {skip_micro} micro-batches of epoch {epoch + 1} done"
                if skip_micro
                else ""
            ),
            flush=True,
        )
    else:
        skip_micro, partial = 0, None
    while epoch < n_epochs:
        epoch += 1
        t_epoch = time.time()
        ds_sum: dict[int, float] = defaultdict(float)
        ds_n: dict[int, int] = defaultdict(int)
        loss_sum = 0.0
        n_steps = 0
        if partial is not None:
            # mid-epoch resume: the epoch's running sums as the checkpoint left them
            loss_sum, n_steps = float(partial["loss_sum"]), int(partial["n_steps"])
            ds_sum.update({int(k): float(v) for k, v in partial["ds_sum"].items()})
            ds_n.update({int(k): int(v) for k, v in partial["ds_n"].items()})
            partial = None
        loader = batch_loader(
            view,
            train_pos,
            cfg.data.batch_size,
            cfg.seed * 100003 + epoch,
            cfg.data.num_workers,
            cfg.data.sampler,
        )
        t_probe = None
        step_loss = 0.0
        for i, (batch, tg) in enumerate(loader):
            k, j = divmod(
                i, acc
            )  # optimizer step within the epoch, micro-batch within it
            if k >= steps_per_epoch or (max_steps is not None and step >= max_steps):
                break
            if i < skip_micro:
                continue
            group = min(acc, micro_per_epoch - k * acc)
            if not epoch_seconds and i == 5:
                t_probe = time.time()
            if j == 0:
                for g in opt.param_groups:
                    g["lr"] = _lr_at(step, cfg.optim, total_steps)
                opt.zero_grad(set_to_none=True)
                step_loss = 0.0
            b = batch.to(device)
            with _autocast(device):
                out = model(b, tg.gene_rows.to(device), tg.record_rows.to(device))
            loss, per_ds = batch_loss(
                out.gene,
                out.record,
                tg,
                objective,
                len(view.datasets),
                cfg.weighting.kind,
                None
                if view.weight_group is None
                else torch.from_numpy(view.weight_group),
            )
            (loss / group if group > 1 else loss).backward()
            # train/loss and train/<dataset>/loss: means over micro-batches
            loss_sum += float(loss.detach())
            step_loss += float(loss.detach()) / group
            n_steps += 1
            for d, v in per_ds.items():
                ds_sum[d] += v
                ds_n[d] += 1
            if j < group - 1:
                continue
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.optim.clip_grad_norm
                )
            )
            if not math.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm at step {step}")
            opt.step()
            step += 1
            every = cfg.budget.checkpoint_every_steps
            if every is not None and step % every == 0 and k + 1 < steps_per_epoch:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "epoch": epoch - 1,  # completed epochs
                        "step": step,
                        "micro_done": i + 1,
                        "partial": {
                            "loss_sum": loss_sum,
                            "n_steps": n_steps,
                            "ds_sum": dict(ds_sum),
                            "ds_n": dict(ds_n),
                        },
                        "config": cfg.model_dump(mode="json"),
                        "standardization_sha256": stats_sha,
                    },
                    osp.join(ckpt_dir, "last.pt"),
                )
            if step % 50 == 0:
                logger.log(
                    {
                        "train/loss_step": step_loss,
                        "train/grad_norm": grad_norm,
                        "train/grad_norm_clip_frac": grad_norm
                        / cfg.optim.clip_grad_norm,
                        "train/lr": opt.param_groups[0]["lr"],
                    },
                    step=step,
                )
            if (
                not epoch_seconds
                and t_probe is not None
                and i >= 5 + cfg.budget.probe_steps
            ):
                # per micro-batch (= per optimizer step when accumulate_steps is 1); the
                # first optimizer step at or after the probe window reports it, once
                s_per_step = (time.time() - t_probe) / (i - 5)
                t_probe = None
                proj = s_per_step * steps_per_epoch * acc
                unit = "step" if acc == 1 else f"micro-step (accumulate {acc})"
                print(
                    f"[projection] {s_per_step:.3f} s/{unit} over {cfg.budget.probe_steps} steps; "
                    f"{steps_per_epoch} steps/epoch -> {proj / 60:.1f} min/epoch of training "
                    f"(evaluation not included); {n_epochs} epochs -> "
                    f"{proj * n_epochs / 3600:.2f} h",
                    flush=True,
                )
                logger.log(
                    {
                        "setup/probe_seconds_per_step": s_per_step,
                        "setup/projected_train_minutes_per_epoch": proj / 60,
                    },
                    step=step,
                )
        skip_micro = 0  # only the resumed epoch skips
        logs: dict[str, float] = {
            "epoch": float(epoch),
            "train/loss": loss_sum / max(n_steps, 1),
        }
        by_type: dict[str, list[float]] = defaultdict(list)
        for d, total in ds_sum.items():
            name = view.datasets[d]
            logs[f"train/{name}/loss"] = total / ds_n[d]
            by_type[SPECS[name].metric_type].append(total / ds_n[d])
        for t, vals in by_type.items():
            logs[f"train/{t}/loss"] = float(np.mean(vals))
        if epoch % cfg.budget.eval_every_epochs == 0 or epoch == n_epochs:
            logs.update(evaluate("val", model, objective, view, val_pos, cfg, device))
            logs.update(
                evaluate("traineval", model, objective, view, te_pos, cfg, device)
            )
            logs.update(evaluate_doubles(model, objective, view, cfg, device))
        dt = time.time() - t_epoch
        epoch_seconds.append(dt)
        logs["time/epoch_seconds"] = dt
        if device.type in ("cuda", "xpu"):
            # peak device memory since the run started (training and evaluation)
            logs["time/peak_gpu_gb"] = (
                torch.get_device_module(device).max_memory_allocated(device) / 1e9
            )
        logs["time/epochs_per_hour"] = 3600.0 / dt
        if len(epoch_seconds) == 1 and cfg.budget.time_budget_hours is not None:
            remaining = (
                cfg.budget.time_budget_hours * 3600
                - (time.time() - t_start)
                - cfg.budget.time_margin_minutes * 60
            )
            fit = 1 + max(0, int(remaining // dt))
            if max_steps is not None and fit < n_epochs:
                raise RuntimeError(
                    f"the {cfg.budget.time_budget_hours} h budget fits {fit} epochs but "
                    f"max_steps={max_steps} needs {n_epochs}; a cut run is not step-matched"
                )
            cap = cfg.budget.max_epochs if max_steps is None else n_epochs
            n_epochs = max(1, min(cap, fit))
            window = cfg.budget.registered_window.epochs(n_epochs)
            print(
                f"[budget] first epoch {dt / 60:.1f} min (training + evaluation); "
                f"{cfg.budget.time_budget_hours} h budget -> {n_epochs} epochs; "
                f"registered window epochs {window[0]}..{window[1]}",
                flush=True,
            )
        logs["setup/n_epochs"] = float(n_epochs)
        logs["setup/window_first_epoch"] = float(window[0])
        logs["setup/window_last_epoch"] = float(window[1])
        logger.log(logs, step=step)
        print(
            f"[epoch {epoch}/{n_epochs}] {dt:.0f} s; "
            + "; ".join(
                f"{k} {v:.4f}"
                for k, v in sorted(logs.items())
                if k.startswith(
                    (
                        "train/loss",
                        "val/loss",
                        "val/expression/pearson_per_gene",
                        "val/protein/pearson_per_gene",
                        "val/promoter/pearson",
                        "val/doubles/pearson",
                        "operator/relative_norm",
                        "time/peak_gpu_gb",
                    )
                )
            ),
            flush=True,
        )
        state = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": cfg.model_dump(mode="json"),
            "standardization_sha256": stats_sha,
        }
        torch.save(state, osp.join(ckpt_dir, "last.pt"))
        if epoch in window:
            torch.save(state, osp.join(ckpt_dir, f"epoch_{epoch:04d}.pt"))
        last = logs
    return RunResult(
        n_epochs=n_epochs,
        steps=step,
        epoch_seconds=epoch_seconds,
        stats_sha256=stats_sha,
        window=window,
        last_metrics=last,
        checkpoint_dir=ckpt_dir,
    )


# =========================================================================== entry points
def config_dump(cfg: TrainConfig043, store: PreparedStore043) -> dict[str, Any]:
    """The W&B config: the run config plus the split pins."""
    d = cfg.model_dump(mode="json")
    d["split"] = {
        "version": store.meta.split_version,
        "store_sha256": store.meta.split_store_sha256,
    }
    d["prep"] = {
        "version": store.meta.prep_version,
        "torchcell_commit": store.meta.torchcell_commit,
    }
    d["registered_window"] = cfg.budget.registered_window.model_dump()
    d["round"] = cfg.wandb.round
    d["prep"]["ontology_n_nodes"] = store.meta.ontology_n_nodes
    d["prep"]["ontology_vocab_sha256"] = store.meta.ontology_vocab_sha256
    d["host"] = socket.gethostname()
    d["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    d["torchcell_commit"] = _git_commit()
    return d


def run(cfg: TrainConfig043) -> RunResult:
    """Load the prepared store and train with W&B logging."""
    if cfg.init_from is not None and cfg.init_from_label is None:
        cfg = cfg.model_copy(update={"init_from_label": stage1_label(cfg.init_from)})
    generation = store_paths(cfg.data.store_version)
    store = PreparedStore043.load(
        cfg.paths.resolved_prep_dir(cfg.data.route, generation, cfg.data.prep_version)
    )
    split = load_split(cfg.paths.resolved_split_file(generation))
    if (split.version, split.store_sha256) != (
        store.meta.split_version,
        store.meta.split_store_sha256,
    ):
        raise ValueError("the prepared store was built under another split file")
    dump = config_dump(cfg, store)
    logger = WandbLogger(cfg, dump)
    try:
        result = train(cfg, store, h5_vector_source(cfg.paths, generation), logger)
        logger.log(
            {
                "split.version_logged": 1.0,
                "standardization.sha256_prefix": float(
                    int(result.stats_sha256[:8], 16)
                ),
            },
            step=result.steps,
        )
    finally:
        logger.finish()
    return result


def _hydra_main() -> None:
    import hydra
    from omegaconf import DictConfig, OmegaConf

    @hydra.main(
        version_base=None, config_path="../conf", config_name="train_043_default"
    )
    def main(dcfg: DictConfig) -> None:
        raw = OmegaConf.to_container(dcfg, resolve=True)
        assert isinstance(raw, dict)
        raw.pop("hydra", None)
        cfg = TrainConfig043.model_validate(raw)
        print(cfg.model_dump_json(indent=1), flush=True)
        result = run(cfg)
        print(
            f"[done] {result.n_epochs} epochs, {result.steps} steps, checkpoints {result.checkpoint_dir}"
        )

    main()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--stage":
        import argparse

        from dotenv import load_dotenv

        load_dotenv()
        ap = argparse.ArgumentParser()
        ap.add_argument("--stage", choices=["prepare"], required=True)
        ap.add_argument("--route", choices=["A", "B"], default="A")
        ap.add_argument("--workers", type=int, default=32)
        ap.add_argument("--prep-version", default=None, help="e.g. prep_v4")
        ap.add_argument(
            "--ontology-vocab",
            default=None,
            help="vocabulary JSON of ontology_annotation_043.py (prep_v4)",
        )
        args = ap.parse_args()
        paths = PathsConfig(data_root=os.environ["DATA_ROOT"])
        print(
            prepare(
                paths,
                args.route,
                args.workers,
                active_store(),
                args.prep_version,
                args.ontology_vocab,
            )
        )
    else:
        _hydra_main()
