# tests/torchcell/datasets/scerevisiae/test_mota2024.py
"""Mota 2024 loader: pH as a typed edit, the ordinal grade, and the merge/drop rules.

The three spreadsheets are written synthetically with openpyxl, so the parser, the dedup
rule and the retention rule all run for real without the raw mirror.
"""

from __future__ import annotations

import json
import os.path as osp
from pathlib import Path
from typing import cast

import openpyxl
import pandas as pd
import pytest

from torchcell.datamodels.media import YPD_AGAR
from torchcell.datamodels.schema import (
    AssayType,
    EnvironmentPhysicalPerturbation,
    MeasurementType,
    PhysicalFactor,
    ResponseCategory,
    SmallMoleculePerturbation,
)
from torchcell.datasets.scerevisiae import mota2024 as m
from torchcell.sequence.genome.scerevisiae.s288c import (
    GeneNameResolution,
    GeneNameStatus,
    SCerevisiaeGenome,
)

# EFG1 and YGR272C are the SI's two names for one gene; RLM2 is retired.
_RENAME = {"EFG1": "YGR271C-A", "YGR272C": "YGR271C-A", "RNR4": "YGR180C"}
_RETIRED = {"RLM2"}


class _StubGenome:
    """Resolver + attribute table, the two things the loader asks a genome for."""

    gene_attribute_table = pd.DataFrame(
        {"ID": ["YGR271C-A", "YGR180C", "YAL001C"], "gene": ["EFG1", "RNR4", "TFC3"]}
    )

    def resolve_gene_name(self, name: str) -> GeneNameResolution:
        upper = name.upper()
        if upper in _RENAME:
            return GeneNameResolution(
                input_name=name,
                status=GeneNameStatus.RENAMED,
                systematic_name=_RENAME[upper],
            )
        if upper in _RETIRED:
            return GeneNameResolution(
                input_name=name, status=GeneNameStatus.RETIRED, systematic_name=upper
            )
        return GeneNameResolution(
            input_name=name, status=GeneNameStatus.CURRENT, systematic_name=upper
        )


# (token, score) rows per acid; the header row the parser keys on is written first.
_ROWS: dict[str, list[tuple[str, str]]] = {
    "acetic": [("YAL001C", "+"), ("EFG1", "++"), ("YGR272C", "++"), ("RLM2", "+")],
    "butyric": [("YAL001C", "++"), ("RNR4", "+"), ("RNR4\xa0", "+")],
    "octanoic": [("EFG1", "+"), ("YGR272C", "++")],
}


def _write_sheets(raw: Path) -> None:
    for spec in m._ACID_SPECS:
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append(["Gene/ORF name", "Encoded Protein Function", "Growth inhibition"])
        for token, score in _ROWS[spec["acid"]]:
            sheet.append([token, "synthetic", score])
        workbook.save(raw / spec["filename"])


@pytest.fixture
def built(tmp_path: Path) -> m.EnvChemgenMota2024Dataset:
    """A tiny end-to-end build over the synthetic spreadsheets."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_sheets(raw)
    return m.EnvChemgenMota2024Dataset(
        root=str(tmp_path), genome=cast(SCerevisiaeGenome, _StubGenome())
    )


def test_ph_is_a_typed_perturbation_and_the_medium_is_the_shared_object() -> None:
    spec = next(s for s in m._ACID_SPECS if s["acid"] == "acetic")
    dataset = m.EnvChemgenMota2024Dataset.__new__(m.EnvChemgenMota2024Dataset)
    environment = dataset._environment(spec)
    assert environment.media is YPD_AGAR
    assert environment.media.base_medium == "YPD"
    assert "pH" not in environment.media.name
    acid_entry, ph_entry = environment.perturbations
    acid = cast(SmallMoleculePerturbation, acid_entry)
    ph = cast(EnvironmentPhysicalPerturbation, ph_entry)
    assert acid.compound.chebi_id == "CHEBI:15366"
    assert acid.concentration.value == 75.0
    assert ph.factor is PhysicalFactor.ph
    assert ph.magnitude is not None
    assert ph.magnitude.unit is not None
    assert (ph.magnitude.value, ph.magnitude.unit.value) == (4.5, "pH")
    assert ph.agent is not None
    assert ph.agent.name == "hydrochloric acid"
    assert ph.agent.inchikey == "VEXZGXHMUGYJMC-UHFFFAOYSA-N"
    assert environment.duration_hours == 48.0


def test_duration_rule_is_the_scoring_anchor_not_the_photograph_range() -> None:
    assert m._DURATION.value == 48.0
    assert "$4 8 \\ \\mathrm { h }$ of incubation" in m._DURATION.quote
    assert "36-48 h" in (m._DURATION.note or "")


def test_the_ordinal_grade_is_stored_with_its_typed_call(
    built: m.EnvChemgenMota2024Dataset,
) -> None:
    grades = {}
    for i in range(len(built)):
        phenotype = built[i]["experiment"]["phenotype"]
        assert phenotype["measurement_type"] is MeasurementType.ordinal
        assert phenotype["assay_type"] is AssayType.spot_dilution
        grades[phenotype["category_label"]] = (
            phenotype["environment_response"],
            phenotype["category"],
        )
    assert grades == {
        "+": (1.0, ResponseCategory.reduced),
        "++": (2.0, ResponseCategory.severely_reduced),
    }
    reference = built[0]["reference"]["phenotype_reference"]
    assert reference["environment_response"] == 0.0
    assert reference["category"] is ResponseCategory.no_change
    assert reference["category_label"] == "0"


def test_unreported_replicate_design_is_a_typed_gap_not_a_silent_none(
    built: m.EnvChemgenMota2024Dataset,
) -> None:
    phenotype = built[0]["experiment"]["phenotype"]
    assert phenotype["n_samples"] is None and phenotype["sample_unit"] is None
    gapped = {gap["field"] for gap in phenotype["provenance_gaps"]}
    assert gapped == {"n_samples", "sample_unit"}
    reasons = {str(gap["reason"]) for gap in phenotype["provenance_gaps"]}
    assert reasons == {"not_reported_by_primary"}


def test_two_source_names_for_one_gene_merge_under_its_canonical_name(
    built: m.EnvChemgenMota2024Dataset,
) -> None:
    log = json.loads(open(osp.join(built.root, m._DROPPED_FILENAME)).read())
    merged = {(e["acid"], e["systematic_name"]): e for e in log["merged"]}
    acetic = merged[("acetic", "YGR271C-A")]
    assert acetic["source_tokens"] == ["EFG1", "YGR272C"]
    assert acetic["kept_score"] == "++"
    assert acetic["stored_gene_name"] == "EFG1"  # the genome's canonical common name
    # octanoic: EFG1 is + and YGR272C is ++, so the MORE SEVERE grade wins
    assert merged[("octanoic", "YGR271C-A")]["kept_score"] == "++"
    # butyric: the RNR4 source duplicate is one token twice, so the token survives
    butyric = merged[("butyric", "YGR180C")]
    assert butyric["source_tokens"] == ["RNR4"] and butyric["kept_score"] == "+"
    assert "MORE SEVERE" in log["dedup_rule"]


def test_retired_token_is_dropped_and_counted(
    built: m.EnvChemgenMota2024Dataset,
) -> None:
    log = json.loads(open(osp.join(built.root, m._DROPPED_FILENAME)).read())
    assert log["n_raw_rows"] == 9
    assert log["n_dropped_records"] == 1
    assert log["n_merged_records"] == 3
    assert len(built) == 5 == log["n_kept_records"]
    (dropped,) = log["dropped"]
    assert dropped["token"] == "RLM2" and dropped["status"] == "retired"
    assert dropped["acids"] == ["acetic"]


def test_renamed_systematic_looking_token_is_rekeyed(
    built: m.EnvChemgenMota2024Dataset,
) -> None:
    stored = {
        p["systematic_gene_name"]
        for i in range(len(built))
        for p in built[i]["experiment"]["genotype"]["perturbations"]
    }
    assert "YGR271C-A" in stored and "YGR272C" not in stored
