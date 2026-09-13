# tests/torchcell/datasets/scerevisiae/test_mormino2022.py
"""Mormino 2022 loader: the Table 1 audit, the SC medium, the ATc inducer, the CBL comparator.

The values are a literal, so the load-bearing test is the audit: every stored Table 1 row
must appear VERBATIM in the sha256-pinned OCR of the article table. That test skips when
the mirror is not mounted; everything else is pure.
"""

from __future__ import annotations

import os
import os.path as osp

import pytest

from torchcell.datamodels.media import MEDIA_LIBRARY, SC
from torchcell.datamodels.schema import (
    AssayType,
    ConcentrationUnit,
    EnvironmentPhysicalPerturbation,
    GeneAdditionPerturbation,
    PhysicalFactor,
    ResponseCategory,
    SmallMoleculePerturbation,
)
from torchcell.datasets.scerevisiae import mormino2022 as m
from torchcell.verification.sourced import SourcedValue, audit_sourced_value


def test_table_1_has_the_twelve_isolated_strains() -> None:
    assert len(m.TABLE_1) == 12
    assert {row[1] for row in m.TABLE_1} == {"+", "="}
    assert sum(1 for row in m.TABLE_1 if row[1] == "+") == 6


def test_the_rfp_call_maps_onto_the_shared_response_category_axis() -> None:
    assert m.RFP_CATEGORY["+"] == (ResponseCategory.enhanced, "+")
    assert m.RFP_CATEGORY["="] == (ResponseCategory.no_change, "=")
    assert m.REFERENCE_CATEGORY is ResponseCategory.no_change


def test_the_comparator_is_the_cbl_pool_with_the_released_threshold() -> None:
    assert m.COMPARATOR.value == "CBL"
    assert "compared to the CBL" in m.COMPARATOR.quote
    assert "CBL" in m.UNITS and "30% higher" in m.UNITS
    assert "CC23" not in m.UNITS


def test_the_medium_is_the_shared_sc_object_not_sd() -> None:
    assert MEDIA_LIBRARY["SC"] is SC
    assert m.MEDIUM.value == "SC"
    assert "synthetic complete medium (SC)" in m.MEDIUM.quote


def test_the_environment_carries_acid_inducer_and_a_typed_ph() -> None:
    dataset = m.CrispriMormino2022Dataset.__new__(m.CrispriMormino2022Dataset)
    environment = m.CrispriMormino2022Dataset._environment(dataset)
    assert environment.media is SC
    small = [
        p for p in environment.perturbations if isinstance(p, SmallMoleculePerturbation)
    ]
    physical = [
        p
        for p in environment.perturbations
        if isinstance(p, EnvironmentPhysicalPerturbation)
    ]
    assert len(small) == 2 and len(physical) == 1
    by_name = {p.compound.name: p for p in small}
    acid = by_name["acetic acid"]
    assert acid.concentration.value == 50.0
    assert acid.concentration.unit is ConcentrationUnit.millimolar
    assert acid.compound.inchikey == "QTBSBXVTEAMEQO-UHFFFAOYSA-N"
    atc = by_name["anhydrotetracycline"]
    assert atc.concentration.value == 2.0
    assert atc.concentration.unit is ConcentrationUnit.ug_per_ml
    assert atc.compound.inchikey is not None
    assert atc.solvent is not None and atc.solvent.name == "DMSO"
    assert atc.solvent.compound is not None
    assert atc.solvent.compound.inchikey is not None
    (ph,) = physical
    assert ph.factor is PhysicalFactor.ph
    assert ph.magnitude is not None
    assert ph.magnitude.value == 3.5
    assert ph.magnitude.unit is ConcentrationUnit.ph


def test_the_biosensor_cassette_is_a_constant_background_in_the_genotype() -> None:
    cassette = m.biosensor_cassette()
    assert len(cassette) == 2
    assert all(isinstance(p, GeneAdditionPerturbation) for p in cassette)
    assert {p.systematic_gene_name for p in cassette} == m.BACKGROUND_GENES
    for perturbation in cassette:
        assert perturbation.is_heterologous
        assert perturbation.localization == "chromosomal_integration"
        assert perturbation.integration_locus == "HO"
        assert perturbation.construct_name == "pMM4_14L"
    assert "integrated into the HO locus" in m.BIOSENSOR_CASSETTE.quote


def test_the_replicate_count_and_assay_are_sourced() -> None:
    assert m.N_REPLICATES.value == 2
    assert "two biological replicates" in m.N_REPLICATES.quote
    assert m.TEMPERATURE_C.value == 30.0


def test_reference_carries_the_cbl_baseline_call() -> None:
    dataset = m.CrispriMormino2022Dataset.__new__(m.CrispriMormino2022Dataset)
    dataset.name = "crispri_mormino2022"
    environment = m.CrispriMormino2022Dataset._environment(dataset)
    reference = m.CrispriMormino2022Dataset._reference(dataset, environment)
    phenotype = reference.phenotype_reference
    assert phenotype.category is ResponseCategory.no_change
    assert phenotype.category_label == "="
    assert phenotype.assay_type is AssayType.biosensor_readout
    assert phenotype.n_samples == 2


def test_the_audit_rejects_a_row_the_source_table_does_not_carry() -> None:
    fragment = m.table_1_row_fragment(m.TABLE_1[0])
    with pytest.raises(RuntimeError, match="not present verbatim"):
        m.audit_table_1("nothing like the table")
    assert fragment.startswith("<td>#3</td>")


def _mirror_paper_md() -> str | None:
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        return None
    path = osp.join(data_root, "torchcell-library", m.CITATION_KEY, "paper.md")
    return path if osp.exists(path) else None


def test_every_stored_table_1_row_is_verbatim_in_the_pinned_ocr() -> None:
    path = _mirror_paper_md()
    if path is None:
        pytest.skip("torchcell-library mirror not mounted")
    m.audit_table_1(open(path, encoding="utf-8").read())


def _library_root() -> str | None:
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        return None
    root = osp.join(data_root, "torchcell-library")
    return root if osp.isdir(root) else None


def test_every_sourced_value_is_backed_by_its_verbatim_quote() -> None:
    root = _library_root()
    if root is None:
        pytest.skip("torchcell-library mirror not mounted")
    values = [value for value in vars(m).values() if isinstance(value, SourcedValue)]
    assert values, "the loader must carry module-level SourcedValues"
    for value in values:
        result = audit_sourced_value(value, root)
        assert result.passed, f"{value.value!r}: {result.message}"
