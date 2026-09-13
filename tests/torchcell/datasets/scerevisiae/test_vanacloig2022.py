# tests/torchcell/datasets/scerevisiae/test_vanacloig2022.py
"""Vanacloig-Pedros 2022 loader: retention rules, sourced encoding, batch-matched control.

Everything but the mirror audit runs on a synthetic count matrix and a fake genome, so
no build tree and no network are needed. The audit test binds each module-level
``SourcedValue`` to its verbatim quote in the sha256-pinned mirror and skips when the
mirror is not mounted.
"""

from __future__ import annotations

import gzip
import math
import os
import os.path as osp
from typing import Any

import pandas as pd
import pytest

from torchcell.datamodels.media import MEDIA_LIBRARY
from torchcell.datamodels.schema import (
    BarcodedKanMxDeletionPerturbation,
    DoseBasis,
    EnvironmentPhysicalPerturbation,
    PhysicalFactor,
    SmallMoleculePerturbation,
    UncertaintyType,
)
from torchcell.datasets.scerevisiae import vanacloig2022 as v
from torchcell.verification.sourced import SourcedValue, audit_sourced_value

# The synthetic library: two current genes, one retired ORF, one legacy spelling of a
# gene the library ALSO carries under its current name.
_GENES = {"YAL001C", "YAL002W", "YBR001C"}
_STANDARD = {"TFC3": ["YAL001C"], "VPS8": ["YAL002W"], "SPO23": ["YBR001C"]}
_RENAMED = {"YAL003W": "YAL002W"}  # legacy spelling of a gene already in the pool


class _Resolution:
    def __init__(self, status: str, systematic: str | None) -> None:
        self.status = status
        self.systematic_name = systematic

    @property
    def is_current_gene(self) -> bool:
        return self.status in ("current", "renamed")


class _FakeGenome:
    """The slice of ``SCerevisiaeGenome`` the loader's gene-name policy reads."""

    gene_set = _GENES
    feature_index = {"standard_to_ids": _STANDARD}

    def resolve_gene_name(self, name: str) -> _Resolution:
        upper = name.upper()
        if upper in _GENES:
            return _Resolution("current", upper)
        if upper in _RENAMED:
            return _Resolution("renamed", _RENAMED[upper])
        for standard, ids in _STANDARD.items():
            if upper == standard:
                return _Resolution("renamed", ids[0])
        return _Resolution("retired", upper)


def _matrix() -> pd.DataFrame:
    """Four library rows x (2 controls per batch + 2 compounds x 3 reps)."""
    rows = [
        ("YAL001C_AAAACCCCGGGGTTTTACGT", "TFC3"),
        ("YAL002W_CCCCGGGGTTTTAAAACGTA", "VPS8"),
        ("YAL003W_GGGGTTTTAAAACCCCGTAC", "YAL003W"),  # legacy duplicate -> dropped
        ("YPL999C_TTTTAAAACCCCGGGGTACG", "NOPE"),  # retired -> dropped
    ]
    data: dict[str, Any] = {
        "gene": [r[0] for r in rows],
        "std_name": [r[1] for r in rows],
        "Control1_CG003": [100, 200, 300, 400],
        "Control2_CG003": [120, 220, 320, 420],
        "Control1_CG004": [400, 300, 200, 100],
        "Control2_CG004": [420, 320, 220, 120],
        # Benomyl: one replicate per batch (paired), row 1 is an all-zero cell
        "Benomyl_CG003_rep1": [110, 0, 310, 410],
        "Benomyl_CG004_rep2": [410, 0, 210, 110],
        "Benomyl_CG003_rep3": [130, 0, 330, 430],
        # DMSO: the vehicle control column, dropped whole
        "DMSO_CG003_rep1": [90, 190, 290, 390],
        "DMSO_CG004_rep2": [390, 290, 190, 90],
        "DMSO_CG003_rep3": [95, 195, 295, 395],
        # MBO: no structure identifier resolves, dropped whole
        "MBO_CG003_rep1": [70, 170, 270, 370],
        "MBO_CG004_rep2": [370, 270, 170, 70],
        "MBO_CG003_rep3": [75, 175, 275, 375],
    }
    return pd.DataFrame(data)


@pytest.fixture()
def built(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Build the synthetic dataset end to end into a temporary root."""
    monkeypatch.setattr(v, "default_genome", lambda: _FakeGenome())
    monkeypatch.setattr(v.EnvChemgenVanacloig2022Dataset, "download", lambda self: None)
    monkeypatch.setattr(
        v.EnvChemgenVanacloig2022Dataset, "_load_matrix", lambda self: _matrix()
    )
    root = str(tmp_path / "env_chemgen_vanacloig2022")
    os.makedirs(osp.join(root, "raw"), exist_ok=True)
    with gzip.open(osp.join(root, "raw", v.DATA_FILENAME), "wt") as handle:
        handle.write("placeholder\n")  # presence only; _load_matrix is patched
    return v.EnvChemgenVanacloig2022Dataset(root=root)


def test_drop_rules_account_for_every_source_record(built: Any) -> None:
    log = v.DropLog.model_validate_json(
        open(osp.join(built.root, "preprocess", "dropped_records.json")).read()
    )
    rules = {rule.rule: rule for rule in log.rules}
    assert rules["vehicle_control_served_as_a_treatment"].items == ["DMSO"]
    assert rules["compound_without_a_structure_identifier"].items == ["MBO"]
    assert rules["orf_is_not_a_current_genome_gene"].items == ["YPL999C"]
    assert rules["orf_is_a_legacy_spelling_of_another_library_orf"].items == ["YAL003W"]
    # one compound (Benomyl) x two kept rows, minus the all-zero cell
    assert rules["all_three_replicate_counts_are_zero"].n_records == 1
    assert log.source_records == 4 * 3
    assert log.kept_records == 1
    assert log.dropped_records == sum(rule.n_records for rule in log.rules)
    assert len(built) == log.kept_records


def test_record_carries_the_barcode_the_canonical_name_and_the_shared_medium(
    built: Any,
) -> None:
    record = built[0]
    experiment = record["experiment"]
    deletion = experiment["genotype"]["perturbations"][0]
    assert deletion["perturbation_type"] == "barcoded_kanmx_deletion"
    assert deletion["systematic_gene_name"] == "YAL001C"
    assert (
        deletion["perturbed_gene_name"] == "TFC3"
    )  # the genome's spelling, not std_name
    assert deletion["barcode"] == "AAAACCCCGGGGTTTTACGT"
    assert deletion["collection"] == v.LIBRARY_COLLECTION.value
    # the three constant background deletions ride on every genotype
    assert {
        p["systematic_gene_name"] for p in experiment["genotype"]["perturbations"]
    } == {"YAL001C", "YGL013C", "YBL005W", "YDR011W"}
    assert experiment["environment"]["media"] == MEDIA_LIBRARY["SYNBASE"].model_dump()
    assert experiment["environment"]["aerobicity"] == "anaerobic"
    assert experiment["environment"]["duration_hours"] == 48.0
    assert experiment["environment"]["duration_generations"] == 6.5


def test_environment_carries_the_compound_and_a_typed_ph(built: Any) -> None:
    perturbations = built[0]["experiment"]["environment"]["perturbations"]
    compound = next(
        p for p in perturbations if p["perturbation_type"] == "small_molecule"
    )
    assert compound["compound"]["name"] == "benomyl"
    assert compound["compound"]["inchikey"] == "RIOXQFHNBCKOKP-UHFFFAOYSA-N"
    assert compound["concentration"]["value"] == 10.0
    assert compound["concentration"]["unit"] == "ug/mL"
    assert compound["concentration"]["basis"] == "fixed"
    # the vehicle is the field that is actually None, so it is the one that is gapped;
    # the dose is not gapped, because an IC30 / fixed basis is always known
    assert compound["solvent"] is None
    assert [g["field"] for g in compound["provenance_gaps"]] == ["solvent"]
    assert compound["provenance_gaps"][0]["reason"] == "deferred_pending_source_review"
    assert compound["provenance_gaps"][0]["resolve_with"]["page"] == "Table S1"
    ph = next(
        p for p in perturbations if p["perturbation_type"] == "environment_physical"
    )
    assert ph["factor"] == "pH"
    assert ph["magnitude"]["value"] == 5.0 and ph["magnitude"]["unit"] == "pH"
    assert ph["agent"]["name"] == "hydrochloric acid"  # the acid the paper names
    assert ph["provenance_gaps"] == []


def test_reference_environment_holds_no_inhibitor(built: Any) -> None:
    reference = built[0]["reference"]
    types = {
        p["perturbation_type"]
        for p in reference["environment_reference"]["perturbations"]
    }
    assert types == {"environment_physical"}  # pH only: the control is inhibitor-free
    assert reference["phenotype_reference"]["environment_response"] == 0.0
    assert reference["genome_reference"]["strain"] == "S288C"


def test_response_is_the_batch_matched_log2_ratio_with_a_nonzero_sample_sd(
    built: Any,
) -> None:
    phenotype = built[0]["experiment"]["phenotype"]
    assert phenotype["measurement_type"] == "log2_ratio"
    assert phenotype["assay_type"] == "pooled_competitive_growth_barcode"
    assert phenotype["n_samples"] == 3
    assert phenotype["sample_unit"] == "biological_replicate"
    assert (
        phenotype["environment_response_uncertainty_type"] == UncertaintyType.sample_sd
    )
    assert phenotype["environment_response_uncertainty"] > 0.0
    assert phenotype["environment_response_se"] == pytest.approx(
        phenotype["environment_response_uncertainty"] / math.sqrt(3)
    )
    assert "SAME CG batch" in phenotype["units"]


def test_unpaired_compound_keeps_the_pooled_control_in_its_units() -> None:
    dataset = v.EnvChemgenVanacloig2022Dataset.__new__(v.EnvChemgenVanacloig2022Dataset)
    dataset.name = "EnvChemgenVanacloig2022Dataset"
    assert "pooled rather than batch-matched" in dataset._units("MMS")
    assert "SAME CG batch" in dataset._units("Furfural")
    # MMS's dose was published, not set to an IC30, but its unit is not stated
    mms = dataset._concentration("MMS")
    assert mms.value is None and mms.basis is DoseBasis.fixed
    assert dataset._concentration("Furfural").basis is DoseBasis.IC30


def test_resolve_library_rows_separates_retired_from_legacy_duplicates() -> None:
    rows = v.resolve_library_rows(
        pd.Series(["YAL001C", "YAL002W", "YAL003W", "YPL999C"]),
        pd.Series(["AAAA", "TTTT", "CCCC", "GGGG"]),
        _FakeGenome(),  # type: ignore[arg-type]  # the slice of the genome API used
    )
    assert rows.keep_mask == [True, True, False, False]
    assert rows.systematic == ["YAL001C", "YAL002W"]
    assert rows.common == ["TFC3", "VPS8"]
    assert rows.dropped_retired == ["YPL999C"]
    assert rows.dropped_legacy_duplicate == ["YAL003W"]


def test_every_sourced_value_is_backed_by_a_verbatim_quote_in_the_mirror() -> None:
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        pytest.skip("DATA_ROOT not set")
    library = osp.join(data_root, "torchcell-library")
    if not osp.isdir(osp.join(library, v.CITATION_KEY)):
        pytest.skip("paper mirror not mounted")
    values = [
        getattr(v, name)
        for name in dir(v)
        if isinstance(getattr(v, name), SourcedValue)
    ]
    assert len(values) >= 12
    for value in values:
        assert audit_sourced_value(value, library).passed


def test_perturbation_leaves_are_the_typed_ones() -> None:
    dataset = v.EnvChemgenVanacloig2022Dataset.__new__(v.EnvChemgenVanacloig2022Dataset)
    dataset.name = "EnvChemgenVanacloig2022Dataset"
    genotype = dataset._genotype("YAL001C", "TFC3", "ACGT")
    assert isinstance(genotype.perturbations[0], BarcodedKanMxDeletionPerturbation)
    environment = dataset._environment("Furfural")
    assert isinstance(environment.perturbations[0], SmallMoleculePerturbation)
    assert isinstance(environment.perturbations[1], EnvironmentPhysicalPerturbation)
    assert environment.perturbations[1].factor is PhysicalFactor.ph
