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

    python experiments/043-expression-proteome-morphology/scripts/train_043.py --stage prepare
    python experiments/043-expression-proteome-morphology/scripts/train_043.py arm=joint_9 loss=hlgauss seed=0
"""

from __future__ import annotations

import hashlib
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

from torchcell.data.representation_043 import ENCODERS, PositionGap
from torchcell.losses.hl_gauss import HLGaussConfig, StandardizationStats
from torchcell.models.cell_model_043 import (
    Batch043,
    CellModel043,
    CellModel043Config,
    ContextInputs,
    MarkInputs,
    ReadoutLoss043,
    Replicon,
    build_head_mask,
    load_vector_table,
    select_marks,
)

SCRIPTS = osp.dirname(osp.abspath(__file__))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
from global_split_043 import GlobalSplit043, Role, load_split  # noqa: E402

EXPERIMENT = "043-expression-proteome-morphology"
PREP_VERSION = "prep_v1"
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

Transform = Literal[
    "as_released",
    "log2_ratio_to_reference",
    "difference_to_reference",
    "log2_plus_one",
    "log2_positive",
    "log2_cpm_plus_one",
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
}


# =========================================================================== config
class ArmConfig(BaseModel):
    """Which datasets train; ``doubles`` reads Sameith Dm as split B (needs Kemmeren)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    datasets: list[str] = Field(min_length=1)
    doubles: bool = True

    @model_validator(mode="after")
    def _known(self) -> ArmConfig:
        """Every dataset is a trainable spec; doubles need Kemmeren."""
        for d in self.datasets:
            if d not in SPECS or SPECS[d].split_b:
                raise ValueError(f"{d!r} is not a trainable dataset: {sorted(SPECS)}")
        if len(set(self.datasets)) != len(self.datasets):
            raise ValueError("datasets repeat")
        if self.doubles and "kemmeren" not in self.datasets:
            raise ValueError("doubles are read through Kemmeren; list kemmeren")
        return self


class LossConfig(BaseModel):
    """The objective (O4)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["hlgauss", "pinball", "mse"]
    hl_gauss: HLGaussConfig = Field(default_factory=HLGaussConfig)
    num_quantiles: int = Field(default=19, gt=0)


class WeightingConfig(BaseModel):
    """Equal per dataset, or by record count (O5)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["equal", "records"]


class DataConfig(BaseModel):
    """Batching and evaluation subsets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    route: Literal["A", "B"] = "A"
    sampler: Literal["records", "balanced"] = Field(
        default="records",
        description="records: uniform over training records (the per-record extreme); "
        "balanced: each batch slot draws a dataset uniformly, then a record uniformly "
        "within it. Composes with weighting (the loss-side knob).",
    )
    batch_size: int = Field(default=16, gt=0)
    num_workers: int = Field(default=4, ge=0)
    mark_cap: int | None = Field(
        default=1024,
        gt=0,
        description="SMOKE ONLY: marks kept per record (select_marks); the design has no cap",
    )
    val_max_records_per_dataset: int | None = Field(default=2000, gt=0)
    traineval_max_records_per_dataset: int = Field(default=256, gt=0)
    min_strains_per_gene: int = Field(default=3, ge=2)


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
    operator_residual: Literal["plain", "rezero"] = Field(
        default="plain",
        description="plain: the trunk's post-norm residual (default since 2026-10-23); "
        "rezero: zero-initialized beta gates, kept only as a control",
    )
    type_token_dim: int = 16
    env_token_dim: int = 16
    dataset_token_dim: int = 8
    head_hidden: int = 128
    exp_type_mode: Literal["flat", "dag"] = "dag"


class OptimConfig(BaseModel):
    """AdamW, linear warmup then constant, gradient clipping by norm."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lr: float = Field(default=3.0e-4, gt=0)
    weight_decay: float = Field(default=1.0e-8, ge=0)
    warmup_steps: int = Field(default=500, ge=0)
    clip_grad_norm: float = Field(default=10.0, gt=0)


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
    eval_every_epochs: int = Field(default=1, gt=0)
    registered_window: WindowConfig = Field(default_factory=WindowConfig)


class WandbConfig(BaseModel):
    """Where the run logs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    project: str = WANDB_PROJECT
    entity: str | None = None
    mode: Literal["online", "offline", "disabled"] = "online"
    tags: list[str] = Field(default_factory=list)


class PathsConfig(BaseModel):
    """Inputs and outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data_root: str
    split_file: str
    prep_dir: str | None = None
    representation_root: str | None = None
    alleles_root: str | None = None
    checkpoint_root: str | None = None

    def resolved_prep_dir(self, route: str) -> str:
        """The prepared store directory."""
        if self.prep_dir is not None:
            return self.prep_dir
        return osp.join(
            self.data_root,
            "data/torchcell/experiments",
            EXPERIMENT,
            "train_043_prep",
            f"{PREP_VERSION}_route{route}",
        )

    def resolved_representation_root(self) -> str:
        """``representation_043`` (vocabularies)."""
        return self.representation_root or osp.join(
            self.data_root,
            "data/torchcell/experiments",
            EXPERIMENT,
            "representation_043",
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
    device: Literal["auto", "cpu", "cuda"] = "auto"
    arm: ArmConfig
    loss: LossConfig
    weighting: WeightingConfig
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    paths: PathsConfig

    @property
    def arm_label(self) -> str:
        """``<arm>[_balanced]_<loss>_<weighting>[_rezero]``: the W&B group.

        ``_rezero`` marks the ReZero operator control; the plain residual is unmarked. The
        overnight runs of 2026-10-22 were ReZero under the unmarked name (it was the only
        residual then).
        """
        sampler = "_balanced" if self.data.sampler == "balanced" else ""
        rezero = "_rezero" if self.model.operator_residual == "rezero" else ""
        return (
            f"{self.arm.name}{sampler}_{self.loss.kind}_{self.weighting.kind}{rezero}"
        )

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

    def array_fields(self) -> dict[str, np.ndarray]:
        """Every array, by name (edges as ``edges__<name>``)."""
        out = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in ("meta", "edges")
        }
        out.update({f"edges__{k}": v for k, v in self.edges.items()})
        return out

    def save(self, directory: str) -> None:
        """``store.npz`` + ``meta.json``."""
        os.makedirs(directory, exist_ok=True)
        np.savez(osp.join(directory, "store.npz"), **self.array_fields())
        with open(osp.join(directory, "meta.json"), "w") as f:
            f.write(self.meta.model_dump_json(indent=1))

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
        return cls(meta=meta, edges=edges, **arrays)


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
) -> npt.NDArray[np.float32]:
    """The pre-standardization transform of one record's gene vector (NaN = missing)."""
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
        else:
            raise ValueError(f"unknown transform {transform!r}")
    return np.asarray(out, dtype=np.float32)


def _context_hash(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


class _PrepRows(torch.utils.data.Dataset[dict[str, Any] | None]):
    """One compact row per raw record (runs in DataLoader workers)."""

    def __init__(self, ds: Any, split: GlobalSplit043) -> None:
        self.ds = ds
        self.split = split
        self.short_of = {s.name: s.short for s in SPECS.values()}

    def __len__(self) -> int:
        return len(self.ds.items)

    def __getitem__(self, i: int) -> dict[str, Any] | None:
        import dataset_043 as d043
        import representation_043 as rep043
        from make_global_split_043 import isolate_key

        from torchcell.models.genotype_tokens_043 import CHANGE_TYPE_INDEX, ChangeType

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
        if spec.level == "gene":
            y_ref = (
                sample["y_ref"].numpy().astype(np.float64)
                if "y_ref" in sample
                else None
            )
            total = (
                float(ex["phenotype"]["total_umis"])
                if spec.transform == "log2_cpm_plus_one"
                else None
            )
            y_vec = transform_target(
                spec.transform, sample["y"].numpy().astype(np.float64), y_ref, total
            )
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
        acc = CHANGE_TYPE_INDEX[ChangeType.present]
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
                        int(Replicon.ACCESSORY_ORF)
                        if t.change_type == acc
                        else int(Replicon.REPORTER_CONSTRUCT)
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


def prepare(paths: PathsConfig, route: Literal["A", "B"], workers: int) -> str:
    """Build the prepared store over every spec (module docstring)."""
    import dataset_043 as d043

    from torchcell.data.representation_043 import RepresentationConfig

    t0 = time.time()
    split = load_split(paths.split_file)
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
        _PrepRows(ds, split), batch_size=64, num_workers=workers, collate_fn=lambda b: b
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
        print(
            f"{done}/{len(ds)} records read, {len(rec['raw_index'])} kept", flush=True
        )
    print(f"Boocock pooled records kept (train + val): {dict(counts['boocock'])}")
    ptr = np.zeros(len(rec["n_marks"]) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(rec["n_marks"])
    edges, outside = _edge_lists(paths.data_root, ds.genes)
    store = PreparedStore043(
        meta=PreparedMeta(
            prep_version=PREP_VERSION,
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
    )
    out = paths.resolved_prep_dir(route)
    store.save(out)
    print(store.meta.model_dump_json(indent=1))
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
    doubles: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.zeros(0, np.int64)
    )  # arm positions of split-B records
    doubles_singles: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.zeros((0, 2), np.int64)
    )  # arm positions of the Kemmeren single of each gene of each double

    def positions(self, role: int, dataset: str | None = None) -> npt.NDArray[np.int64]:
        """Arm positions with ``role`` (and of ``dataset`` when given)."""
        sel = self.role == role
        if dataset is not None:
            sel &= np.asarray(
                [self.datasets[d] == dataset for d in self.local_dataset], bool
            )
        return np.nonzero(sel)[0].astype(np.int64)


def build_view(
    store: PreparedStore043, arm: ArmConfig, mark_cap: int | None
) -> TrainingView:
    """Select the arm, fit the standardization on its training rows, compact the keys."""
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
    )
    if arm.doubles:
        _attach_doubles(view, rec_short)
    return view


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
    )
    g = np.nonzero(view.level_gene[pos])[0]
    rr = np.nonzero(~view.level_gene[pos])[0]
    G = store.Y.shape[1]
    targets = Targets043(
        gene_rows=torch.from_numpy(g.astype(np.int64)),
        gene_z=torch.from_numpy(
            view.Z[store.rec_yrow[rec[g]]] if len(g) else np.zeros((0, G), np.float32)
        ),
        record_rows=torch.from_numpy(rr.astype(np.int64)),
        record_z=torch.from_numpy(view.z_scalar[rec[rr]]),
        dataset_ids=torch.from_numpy(view.local_dataset[pos]),
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
        )

    def log(self, values: dict[str, float], step: int) -> None:
        """``run.log``."""
        self.run.log(values, step=step)

    def finish(self) -> None:
        """``run.finish``."""
        self.run.finish()


# =========================================================================== training
VectorSource = Callable[[int, npt.NDArray[np.int64]], torch.Tensor]


def h5_vector_source(paths: PathsConfig) -> VectorSource:
    """Vocabulary ids of encoder ``e`` -> their vectors from ``alleles/<encoder>.h5``."""

    def read(e: int, ids: npt.NDArray[np.int64]) -> torch.Tensor:
        enc = ENCODERS[e]
        vocab = np.load(
            osp.join(paths.resolved_representation_root(), f"vocab_{enc}.npy")
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
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


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
        operator_residual=cfg.model.operator_residual,
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
        head_hidden=cfg.model.head_hidden,
        loss=cfg.loss.kind,
        hl_gauss=cfg.loss.hl_gauss,
        num_quantiles=cfg.loss.num_quantiles,
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
    )


def batch_loss(
    out_gene: torch.Tensor,
    out_record: torch.Tensor,
    targets: Targets043,
    objective: ReadoutLoss043,
    n_datasets: int,
    weighting: Literal["equal", "records"],
) -> tuple[torch.Tensor, dict[int, float]]:
    """Weighted per-dataset mean loss of a batch, and each dataset's mean (for logs)."""
    sums, counts, nrec = loss_sums(out_gene, out_record, targets, objective, n_datasets)
    total, means = combine_loss(sums, counts, nrec, weighting)
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
    objective: ReadoutLoss043,
    n_datasets: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per dataset: summed elementwise loss, measured entries, records (one batch)."""
    device = out_gene.device
    sums = torch.zeros(n_datasets, device=device)
    counts = torch.zeros(n_datasets, device=device)
    nrec = torch.zeros(n_datasets, device=device)
    ds = targets.dataset_ids.to(device)
    if targets.gene_rows.numel():
        elem, measured = objective.elementwise(out_gene, targets.gene_z.to(device))
        d = ds[targets.gene_rows.to(device)]
        sums = sums.index_add(0, d, elem.sum(1))
        counts = counts.index_add(0, d, measured.sum(1).to(sums.dtype))
        nrec = nrec.index_add(0, d, torch.ones_like(d, dtype=sums.dtype))
    if targets.record_rows.numel():
        elem, measured = objective.elementwise(out_record, targets.record_z.to(device))
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
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """The objective from per-dataset sums: equal = mean of the dataset means, records =
    their mean weighted by records; None when no dataset has a measured entry.
    """
    present = counts > 0
    means = torch.where(present, sums / counts.clamp_min(1.0), torch.zeros_like(sums))
    if not bool(present.any()):
        return None, means
    if weighting == "equal":
        return means[present].mean(), means
    w = nrec * present
    return (means * w).sum() / w.sum(), means


@torch.no_grad()
def predict(
    model: CellModel043,
    objective: ReadoutLoss043,
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
        if tg.gene_rows.numel():
            gene_pred[cursor + tg.gene_rows.numpy()] = (
                objective.point(out.gene).cpu().numpy()
            )
        if tg.record_rows.numel():
            rec_pred[cursor + tg.record_rows.numpy()] = (
                objective.point(out.record).cpu().numpy()
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
    extra = {}
    dataset_loss: dict[int, float] = {}
    loss: float | None = None
    if weighting is not None:
        total, means = combine_loss(acc_sums, acc_counts, acc_nrec, weighting)
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
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
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
    objective: ReadoutLoss043,
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
    objective: ReadoutLoss043,
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


def _lr_at(step: int, cfg: OptimConfig) -> float:
    if cfg.warmup_steps and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    return cfg.lr


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
    view = build_view(store, cfg.arm, cfg.data.mark_cap)
    tables = [vector_source(e, view.used_ids[e]) for e in range(len(ENCODERS))]
    model = build_model(cfg, view, tables).to(device)
    objective = ReadoutLoss043(model.config).to(device)
    stats_sha = view.stats.sha256()
    table_gb = model.composite.table_bytes() / 1e9
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_pos = view.positions(ROLE_CODE[Role.TRAIN])
    val_pos = eval_positions(
        view, ROLE_CODE[Role.VAL], cfg.data.val_max_records_per_dataset, cfg.seed
    )
    te_pos = eval_positions(
        view,
        ROLE_CODE[Role.TRAIN],
        cfg.data.traineval_max_records_per_dataset,
        cfg.seed,
    )
    ckpt_dir = osp.join(cfg.paths.resolved_checkpoint_root(), cfg.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(osp.join(ckpt_dir, "standardization.json"), "w") as f:
        f.write(view.stats.canonical_json())
    print(
        f"[setup] device {device}; {len(train_pos)} train / {len(val_pos)} val / "
        f"{len(te_pos)} traineval records; {len(view.doubles)} doubles; "
        f"{n_params / 1e6:.2f} M trainable parameters; frozen tables {table_gb:.2f} GB "
        f"({[len(u) for u in view.used_ids]} rows); standardization sha256 {stats_sha}",
        flush=True,
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    steps_per_epoch = math.ceil(len(train_pos) / cfg.data.batch_size)
    if cfg.budget.max_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, cfg.budget.max_steps_per_epoch)
    max_steps = cfg.budget.max_steps
    n_epochs = (
        cfg.budget.max_epochs
        if max_steps is None
        else math.ceil(max_steps / steps_per_epoch)
    )
    window = cfg.budget.registered_window.epochs(n_epochs)
    logger.log(
        {
            "setup/max_steps": float(max_steps or 0),
            "setup/n_train": float(len(train_pos)),
            "setup/n_val": float(len(val_pos)),
            "setup/steps_per_epoch": float(steps_per_epoch),
            "setup/table_gb": table_gb,
            "setup/n_params_m": n_params / 1e6,
            "setup/seconds": time.time() - t_start,
        },
        step=0,
    )
    step = 0
    epoch_seconds: list[float] = []
    last: dict[str, float] = {}
    epoch = 0
    while epoch < n_epochs:
        epoch += 1
        t_epoch = time.time()
        ds_sum: dict[int, float] = defaultdict(float)
        ds_n: dict[int, int] = defaultdict(int)
        loss_sum = 0.0
        n_steps = 0
        loader = batch_loader(
            view,
            train_pos,
            cfg.data.batch_size,
            cfg.seed * 100003 + epoch,
            cfg.data.num_workers,
            cfg.data.sampler,
        )
        t_probe = None
        for i, (batch, tg) in enumerate(loader):
            if i >= steps_per_epoch or (max_steps is not None and step >= max_steps):
                break
            if epoch == 1 and i == 5:
                t_probe = time.time()
            for g in opt.param_groups:
                g["lr"] = _lr_at(step, cfg.optim)
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
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.optim.clip_grad_norm
                )
            )
            if not math.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm at step {step}")
            opt.step()
            step += 1
            loss_sum += float(loss.detach())
            n_steps += 1
            for d, v in per_ds.items():
                ds_sum[d] += v
                ds_n[d] += 1
            if step % 50 == 0:
                logger.log(
                    {
                        "train/loss_step": float(loss.detach()),
                        "train/grad_norm": grad_norm,
                        "train/grad_norm_clip_frac": grad_norm
                        / cfg.optim.clip_grad_norm,
                        "train/lr": opt.param_groups[0]["lr"],
                    },
                    step=step,
                )
            if epoch == 1 and t_probe is not None and i == 5 + cfg.budget.probe_steps:
                s_per_step = (time.time() - t_probe) / cfg.budget.probe_steps
                proj = s_per_step * steps_per_epoch
                print(
                    f"[projection] {s_per_step:.3f} s/step over {cfg.budget.probe_steps} steps; "
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
        logs: dict[str, float] = {
            "epoch": float(epoch),
            "train/loss": loss_sum / max(n_steps, 1),
        }
        if cfg.model.operator_residual == "rezero":
            logs["train/beta_attn"] = float(model.operator.beta_attn.detach())
            logs["train/beta_ffn"] = float(model.operator.beta_ffn.detach())
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
        logs["time/epochs_per_hour"] = 3600.0 / dt
        if epoch == 1 and cfg.budget.time_budget_hours is not None:
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
    d["host"] = socket.gethostname()
    d["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    d["torchcell_commit"] = _git_commit()
    return d


def run(cfg: TrainConfig043) -> RunResult:
    """Load the prepared store and train with W&B logging."""
    store = PreparedStore043.load(cfg.paths.resolved_prep_dir(cfg.data.route))
    split = load_split(cfg.paths.split_file)
    if (split.version, split.store_sha256) != (
        store.meta.split_version,
        store.meta.split_store_sha256,
    ):
        raise ValueError("the prepared store was built under another split file")
    dump = config_dump(cfg, store)
    logger = WandbLogger(cfg, dump)
    try:
        result = train(cfg, store, h5_vector_source(cfg.paths), logger)
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
        args = ap.parse_args()
        data_root = os.environ["DATA_ROOT"]
        paths = PathsConfig(
            data_root=data_root,
            split_file=osp.join(
                osp.dirname(SCRIPTS), "conf/splits/global_split_v1.json"
            ),
        )
        print(prepare(paths, args.route, args.workers))
    else:
        _hydra_main()
