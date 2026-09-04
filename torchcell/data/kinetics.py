# torchcell/data/kinetics.py
# [[torchcell.data.kinetics]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/kinetics.py
# Test file: tests/torchcell/data/test_kinetics.py

r"""Base dataset for predicted enzyme kinetic parameters, one module per predictor.

WHY THIS MIRRORS THE NODE-EMBEDDING DATASETS
---------------------------------------------
Which kinetic predictor feeds the flux layer is a design choice to experiment over, not a
pipeline detail to settle once. The sequence-embedding datasets already solve that exact
shape -- ``BaseEmbeddingDataset`` plus one module per model plus a name-to-class registry
-- so the kinetic predictors copy it rather than introduce a second convention. A
parameter, like an embedding, is materialized once, cached, and assigned at construction;
neither is ever computed inside a forward pass.

THE ONE STRUCTURAL DIFFERENCE, AND IT DRIVES EVERYTHING
--------------------------------------------------------
**A node embedding is keyed by GENE. A kinetic parameter is keyed by a PAIR.** A turnover
number is a property of one enzyme acting on one substrate, so the key is
``(uniprot, substrate_met_id, predictor, parameter)``. Two consequences follow from the
key rather than being separate decisions:

* Collapsing to one row per gene would silently pick a substrate, and collapsing to one row
  per catalytic unit would silently pick a subunit. So the store holds pairs, and
  **aggregation lives in the resolver, not here**: the dataset records what was predicted,
  and the caller decides what the layer sees.
* ``parameter`` is part of the key rather than two parallel stores, because UniKP and
  EITLEM emit both :math:`k_{cat}` and :math:`K_M` from one forward pass.

WHAT IS STORED, AND WHY IT IS NOT JUST A NUMBER
------------------------------------------------
Every row carries the mirror ``sha256`` of the weights that produced it. A predicted
:math:`k_{cat}` is a model output, not a measurement, and the difference has to survive
into the table: a value with no provenance is indistinguishable from a measured one two
steps later, and the flux layer's coverage gate has to be able to tell an experimental
value, a predicted value, and a filled default apart.
"""

from __future__ import annotations

import os
import os.path as osp
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class KineticParameter(StrEnum):
    """The parameters a predictor can emit."""

    k_cat = "k_cat"
    K_M = "K_M"


class ValueProvenance(StrEnum):
    """Where a kinetic value came from. The gate reads this, so it stays explicit.

    ``predicted`` is not a lesser ``experimental``; they are different quantities and a
    coverage number that adds them without saying so is misleading.
    """

    experimental = "experimental"
    predicted = "predicted"
    default = "default"


class KineticPrediction(BaseModel):
    """One predicted parameter for one (enzyme, substrate) pair.

    ``value`` is in the parameter's canonical unit: 1/s for :math:`k_{cat}`, mM for
    :math:`K_M`. Predictors that emit :math:`\\log_{10}` are converted at the edge of
    their own module, so nothing downstream has to know which convention a model used.
    """

    model_config = ConfigDict(extra="forbid")

    unit_id: int = Field(description="Catalytic-unit index in the GEM's GPR expansion.")
    reaction_id: str
    gene_id: str = Field(description="Systematic ORF name, the permanent cache key.")
    uniprot: str
    substrate_met_id: str = Field(description="GEM metabolite id, e.g. s_0025.")
    substrate_name: str
    parameter: KineticParameter
    value: float
    unit: str
    predictor: str = Field(description="Mirror name, e.g. 'dlkcat'.")
    weights_sha256: str = Field(
        description="sha256 of the weight file that produced this value."
    )
    provenance: ValueProvenance = ValueProvenance.predicted


class KineticsDatasetSummary(BaseModel):
    """What a build covered, written beside it so coverage is never recomputed by hand."""

    model_config = ConfigDict(extra="forbid")

    predictor: str
    parameter: KineticParameter
    n_rows: int
    n_units: int
    n_units_total: int
    n_genes: int
    n_substrates: int
    coverage_frac: float
    value_median: float
    value_p05: float
    value_p95: float
    n_failed: int
    failure_reasons: dict[str, int] = Field(default_factory=dict)
    weights_sha256: str
    built_at: str


class BaseKineticsDataset(ABC):
    """Abstract per-predictor kinetic parameter store, cached on the ORF permanently.

    A concrete subclass implements :meth:`initialize_model` and :meth:`predict_batch`, and
    declares what it ``EMITS``. Everything else -- caching, the pair key, the summary, the
    provenance stamp -- is handled here so that adding a predictor is one small module.

    The cache is a parquet file per (predictor, parameter). A systematic gene name plus a
    substrate id plus a predictor version is a stable key forever, so a rebuild recomputes
    nothing that is already present unless ``overwrite`` is set.
    """

    EMITS: tuple[KineticParameter, ...] = ()
    MIRROR_NAME: str = ""

    def __init__(
        self,
        root: str,
        inputs: pd.DataFrame,
        data_root: str,
        device: str = "cpu",
        transform: Callable[..., Any] | None = None,
    ) -> None:
        """Bind a predictor to its input pairs without loading the model.

        The model is deliberately NOT constructed here. Building it costs seconds to
        minutes and allocates GPU memory, and a caller that only wants to read a cached
        table should pay neither.
        """
        self.root = root
        self.inputs = inputs
        self.data_root = data_root
        self.device = device
        self.transform = transform
        self._model: Any | None = None
        os.makedirs(self.processed_dir, exist_ok=True)

    @property
    def processed_dir(self) -> str:
        """Where this predictor's tables live."""
        return osp.join(self.root, self.MIRROR_NAME, "processed")

    def processed_path(self, parameter: KineticParameter) -> str:
        """The parquet file holding one parameter's predictions."""
        return osp.join(self.processed_dir, f"{parameter.value}.parquet")

    def summary_path(self, parameter: KineticParameter) -> str:
        """The summary written beside a build."""
        return osp.join(self.processed_dir, f"{parameter.value}_summary.json")

    @property
    def model(self) -> Any:
        """The predictor, constructed on first use."""
        if self._model is None:
            self._model = self.initialize_model()
        return self._model

    @abstractmethod
    def initialize_model(self) -> Any:
        """Construct the predictor from its mirror, and verify the mirror first."""

    @abstractmethod
    def predict_batch(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Predict for a block of input pairs.

        Returns a frame indexed like ``rows`` with one column per emitted parameter, plus
        a ``failure`` column naming why a row has no value. A row that fails must produce
        a reason rather than a silent NaN: 'rdkit could not parse this SMILES' and 'this
        metabolite is a protein' are different facts, and the coverage gate needs both.
        """

    def load(self, parameter: KineticParameter) -> pd.DataFrame:
        """Read a cached table."""
        return pd.read_parquet(self.processed_path(parameter))

    def exists(self, parameter: KineticParameter) -> bool:
        """Whether a build is already cached."""
        return osp.exists(self.processed_path(parameter))
