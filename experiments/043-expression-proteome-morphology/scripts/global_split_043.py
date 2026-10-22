# experiments/043-expression-proteome-morphology/scripts/global_split_043.py
# [[experiments.043-expression-proteome-morphology.scripts.global_split_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/global_split_043
"""The 043 fold file record, ``GlobalSplit043``, and the role of a record under it.

One fold id per key in four key spaces (primer, "The evaluation protocol", "The fold
file"):

* ``gene_folds``: systematic ORF of every gene deleted in Kemmeren 2014, Hu 2007, Jackson
  2020, Nadal-Ribelles 2025, Messner 2023, Zelezniak 2018 or Ohya 2005 -> fold 0..9. The
  ``pinned_training_genes`` (the 82 Sameith 2015 singles, and the HO control of Jackson)
  only take folds 2..9.
* ``isolate_folds``: Caudal 2024 / Muenzner 2024 strain name -> fold 0..4.
* ``segregant_folds``: ``f"{dataset}:{record_id}"`` with ``record_id`` the store's
  ``f"{cross}:{segregant_id}"`` -> fold 0..4, drawn within each cross and panel.
* ``promoter_folds``: the distinct Vaishnav 2022 80-mer (variable region) -> fold 0..4.

Fold ``test_fold`` (0) is test, ``val_fold`` (1) validation, every other fold training.
A deletion record trains only if none of its genes is in fold 0 or 1; it is test when any
gene is in fold 0, otherwise validation when any gene is in fold 1. Sameith 2015 Dm is
always split B (test), Ohya 2005 always the probe, Sameith 2015 Sm always excluded.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

SPLIT_VERSION = "global_split_v1"
N_GENE_FOLDS = 10
N_OTHER_FOLDS = 5

DELETION_DATASETS: list[str] = [
    "MicroarrayKemmeren2014Dataset",
    "MicroarrayScoreHu2007Dataset",
    "TfDeletionPseudobulkJackson2020Dataset",
    "NadalRibellesPerturbSeq2025Dataset",
    "ProteomeMessner2023Dataset",
    "ProteomeZelezniak2018Dataset",
    "ScmdOhya2005Dataset",
]
ISOLATE_DATASETS: list[str] = [
    "CaudalPanTranscriptome2024Dataset",
    "ProteomeMuenzner2024Dataset",
    "ProteinTurnoverMuenzner2024Dataset",
]
SEGREGANT_DATASETS: list[str] = ["Boocock2025Dataset", "Nguessan2025Dataset"]
PROMOTER_DATASETS: list[str] = ["PooledPromoterExpressionVaishnav2022Dataset"]
SPLIT_B_DATASET = "DmMicroarraySameith2015Dataset"
PROBE_DATASET = "ScmdOhya2005Dataset"
EXCLUDED_DATASET = "SmMicroarraySameith2015Dataset"


class Role(StrEnum):
    """What a record is for under the fold file."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    SPLIT_B = "split_b_test"
    PROBE = "probe"
    EXCLUDED = "excluded"


class SplitCounts(BaseModel):
    """Records per role for one dataset (and distinct units per role)."""

    train: int = 0
    val: int = 0
    test: int = 0
    split_b_test: int = 0
    probe: int = 0
    excluded: int = 0
    units_train: int = 0
    units_val: int = 0
    units_test: int = 0
    unit: str = Field(description="what one held-out unit is (gene, isolate, ...)")


class GlobalSplit043(BaseModel):
    """The 043 fold file (``conf/splits/global_split_v1.json``)."""

    version: str = SPLIT_VERSION
    seed: int = 0
    store_sha256: str = Field(
        description="sha256 of raw/raw_dataset_name_index.json of the raw store"
    )
    store_index_path: str
    gene_set_sha256: str
    test_fold: int = 0
    val_fold: int = 1
    n_gene_folds: int = N_GENE_FOLDS
    n_isolate_folds: int = N_OTHER_FOLDS
    n_segregant_folds: int = N_OTHER_FOLDS
    n_promoter_folds: int = N_OTHER_FOLDS
    pinned_training_genes: list[str] = Field(
        description="genes kept out of the test and validation folds (Sameith singles, HO)"
    )
    pinned_reason: dict[str, str]
    gene_folds: dict[str, int]
    gene_strata: dict[str, str] = Field(
        description="gene -> its dataset-membership pattern (the stratum of the draw)"
    )
    isolate_folds: dict[str, int]
    isolate_strata: dict[str, str]
    segregant_folds: dict[str, int]
    promoter_folds: dict[str, int]
    counts: dict[str, SplitCounts]
    generated_by: str
    slurm_job_id: str | None = None

    @model_validator(mode="after")
    def _check(self) -> GlobalSplit043:
        for name, folds, n in (
            ("gene_folds", self.gene_folds, self.n_gene_folds),
            ("isolate_folds", self.isolate_folds, self.n_isolate_folds),
            ("segregant_folds", self.segregant_folds, self.n_segregant_folds),
            ("promoter_folds", self.promoter_folds, self.n_promoter_folds),
        ):
            bad = [k for k, f in folds.items() if not 0 <= f < n]
            if bad:
                raise ValueError(
                    f"{name}: {len(bad)} keys outside 0..{n - 1}: {bad[:3]}"
                )
        for g in self.pinned_training_genes:
            if g in self.gene_folds and self.gene_folds[g] in (
                self.test_fold,
                self.val_fold,
            ):
                raise ValueError(f"pinned gene {g} is in fold {self.gene_folds[g]}")
        return self

    # ------------------------------------------------------------------ roles

    def _fold_role(self, fold: int) -> Role:
        if fold == self.test_fold:
            return Role.TEST
        if fold == self.val_fold:
            return Role.VAL
        return Role.TRAIN

    def deletion_role(self, genes: list[str]) -> Role:
        """Test if any gene is in the test fold, else val if any in val, else train."""
        folds = [self.gene_folds[g] for g in genes]
        if self.test_fold in folds:
            return Role.TEST
        if self.val_fold in folds:
            return Role.VAL
        return Role.TRAIN

    def role(self, dataset: str, key: str | list[str]) -> Role:
        """Role of one record: ``key`` is its deleted genes (deletion datasets), its
        strain name (isolates), ``f"{cross}:{segregant_id}"`` (segregants) or its 80-mer
        (promoters).
        """
        if dataset == SPLIT_B_DATASET:
            return Role.SPLIT_B
        if dataset == EXCLUDED_DATASET:
            return Role.EXCLUDED
        if dataset == PROBE_DATASET:
            return Role.PROBE
        if dataset in DELETION_DATASETS:
            assert isinstance(key, list)
            return self.deletion_role(key)
        assert isinstance(key, str)
        if dataset in ISOLATE_DATASETS:
            return self._fold_role(self.isolate_folds[key])
        if dataset in SEGREGANT_DATASETS:
            return self._fold_role(self.segregant_folds[f"{dataset}:{key}"])
        if dataset in PROMOTER_DATASETS:
            return self._fold_role(self.promoter_folds[key])
        raise KeyError(f"{dataset} has no role under {self.version}")

    def check_store(self, index_path: str) -> None:
        """Refuse a store whose raw index differs from the one the file was drawn on."""
        with open(index_path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        if sha != self.store_sha256:
            raise ValueError(
                f"store index sha256 {sha} != fold file {self.store_sha256} ({self.version})"
            )


def load_split(path: str) -> GlobalSplit043:
    """Read and validate a fold file."""
    with open(path) as f:
        return GlobalSplit043.model_validate(json.load(f))
