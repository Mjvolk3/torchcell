# tests/torchcell/datasets/scerevisiae/test_auesukaree2009.py
"""Auesukaree 2009 loader: shared YPD, the unperturbed reference, the two adjudications.

All synthetic: the PDF table parse is monkeypatched and the genome is a stub resolver, so
nothing here needs the raw mirror or a built LMDB.
"""

from __future__ import annotations

import json
import os.path as osp
from pathlib import Path
from typing import cast

import pytest

from torchcell.datamodels.media import YPD_AGAR
from torchcell.datamodels.schema import (
    AssayType,
    MeasurementType,
    ResponseCategory,
    SmallMoleculePerturbation,
)
from torchcell.datasets.scerevisiae import auesukaree2009 as a
from torchcell.sequence.genome.scerevisiae.s288c import (
    GeneNameResolution,
    GeneNameStatus,
    SCerevisiaeGenome,
)

_AMBIGUOUS = {"PPA1": ["YBR011C", "YHR026W"], "FEN1": ["YCR034W", "YKL113C"]}
_RENAME = {"VMA2": "YBR127C", "VMA16": "YHR026W", "ELO2": "YCR034W"}
_RETIRED = {"NOSUCHGENE"}


class _StubGenome:
    """A hand-written resolver covering the three statuses the loader distinguishes."""

    def resolve_gene_name(self, name: str) -> GeneNameResolution:
        upper = name.upper()
        if upper in _AMBIGUOUS:
            return GeneNameResolution(
                input_name=name,
                status=GeneNameStatus.AMBIGUOUS,
                systematic_name=None,
                candidates=_AMBIGUOUS[upper],
            )
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


_TABLES = {
    "ethanol": ["VMA2", "YAL001C"],
    "methanol": ["FEN1"],
    "1-propanol": ["PPA1"],
    "heat": ["YAL002W"],
    "NaCl": ["YAL003W"],
    "H2O2": ["NOSUCHGENE"],
}


def _dataset() -> a.EnvChemgenAuesukaree2009Dataset:
    """An uninitialized instance: the methods under test read no build state."""
    return a.EnvChemgenAuesukaree2009Dataset.__new__(a.EnvChemgenAuesukaree2009Dataset)


@pytest.fixture
def built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> a.EnvChemgenAuesukaree2009Dataset:
    """A tiny end-to-end build over synthetic stress tables."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / a._PDF_FILENAME).write_bytes(b"")  # presence only: download() is skipped
    monkeypatch.setattr(a, "_EXPECTED_LISTED", {k: len(v) for k, v in _TABLES.items()})
    monkeypatch.setattr(
        a.EnvChemgenAuesukaree2009Dataset, "_parse_tables", lambda self: _TABLES
    )
    return a.EnvChemgenAuesukaree2009Dataset(
        root=str(tmp_path), genome=cast(SCerevisiaeGenome, _StubGenome())
    )


def test_media_is_the_shared_library_object_not_free_text() -> None:
    spec = next(s for s in a._STRESS_SPECS if s["stress"] == "ethanol")
    environment = _dataset()._environment(spec)
    # the PLATE member of the YPD family, not the bare join anchor: a spot assay is on
    # solid medium, so this dataset shares one media node with Mota 2024's plates
    assert environment.media is YPD_AGAR
    assert environment.media.state == "solid"
    assert environment.media.base_medium == "YPD"
    assert [c.compound.name for c in environment.media.components] == [
        "yeast extract",
        "peptone",
        "D-glucose",
        "agar",
    ]
    assert environment.duration_hours == 72.0
    assert environment.temperature is not None
    assert environment.temperature.value == 30.0


def test_heat_is_a_temperature_edit_with_no_perturbation_object() -> None:
    spec = next(s for s in a._STRESS_SPECS if s["stress"] == "heat")
    environment = _dataset()._environment(spec)
    assert environment.perturbations == []
    assert environment.temperature is not None
    assert environment.temperature.value == 37.0


def test_every_stress_compound_is_identified_at_its_sourced_dose() -> None:
    doses = {}
    for spec in a._STRESS_SPECS:
        if spec["kind"] != "small_molecule":
            continue
        environment = _dataset()._environment(spec)
        (entry,) = environment.perturbations
        perturbation = cast(SmallMoleculePerturbation, entry)
        assert perturbation.compound.inchikey is not None
        assert perturbation.concentration.unit is not None
        doses[perturbation.compound.name] = (
            perturbation.concentration.value,
            perturbation.concentration.unit.value,
        )
    assert doses == {
        "ethanol": (10.0, "percent_v/v"),
        "methanol": (16.0, "percent_v/v"),
        "1-propanol": (7.0, "percent_v/v"),
        "sodium chloride": (1.0, "M"),
        "hydrogen peroxide": (5.0, "mM"),
    }


def test_ambiguous_tokens_are_adjudicated_by_evidence_never_first_matched() -> None:
    dataset = a.EnvChemgenAuesukaree2009Dataset.__new__(
        a.EnvChemgenAuesukaree2009Dataset
    )
    dataset.genome = cast(SCerevisiaeGenome, _StubGenome())
    # YBR011C is the alphabetically first candidate and the old code's silent answer
    assert a._AMBIGUOUS_ADJUDICATIONS["PPA1"].candidates[0] == "YBR011C"
    assert dataset._resolve_token("PPA1") == ("YHR026W", "VMA16")
    assert dataset._resolve_token("FEN1") == ("YCR034W", "ELO2")
    assert "ESSENTIAL" in a._AMBIGUOUS_ADJUDICATIONS["PPA1"].evidence
    assert "RAD27 IS" in a._AMBIGUOUS_ADJUDICATIONS["FEN1"].evidence


def test_an_unlisted_ambiguous_token_raises_rather_than_guessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = a.EnvChemgenAuesukaree2009Dataset.__new__(
        a.EnvChemgenAuesukaree2009Dataset
    )
    dataset.genome = cast(SCerevisiaeGenome, _StubGenome())
    monkeypatch.setitem(_AMBIGUOUS, "SOMEGENE", ["YAL001C", "YAL002W"])
    with pytest.raises(RuntimeError, match="AMBIGUOUS"):
        dataset._resolve_token("SOMEGENE")


def test_category_is_typed_and_the_source_word_is_kept(
    built: a.EnvChemgenAuesukaree2009Dataset,
) -> None:
    phenotype = built[0]["experiment"]["phenotype"]
    assert phenotype["measurement_type"] is MeasurementType.categorical
    assert phenotype["assay_type"] is AssayType.spot_dilution
    assert phenotype["category"] is ResponseCategory.sensitive
    assert phenotype["category_label"] == "sensitive"
    reference = built[0]["reference"]["phenotype_reference"]
    assert reference["category"] is ResponseCategory.no_change
    assert reference["category_label"] == "tolerant"


def test_reference_environment_is_the_unperturbed_non_stress_plate(
    built: a.EnvChemgenAuesukaree2009Dataset,
) -> None:
    for i in range(len(built)):
        environment = built[i]["reference"]["environment_reference"]
        assert environment["perturbations"] == []
        assert environment["temperature"]["value"] == 30.0
        assert environment["media"] == YPD_AGAR.model_dump()
    assert "vs the SAME strain on the matched non-stress" in a.MEASUREMENT_UNITS


def test_retired_token_is_dropped_and_logged(
    built: a.EnvChemgenAuesukaree2009Dataset,
) -> None:
    assert len(built) == 6  # 7 listed tokens, one of them retired
    log = json.loads(open(osp.join(built.root, a._DROPPED_FILENAME)).read())
    assert log["n_listed_tokens"] == 7
    assert log["n_kept_records"] == 6
    assert log["dropped_tokens"] == {"NOSUCHGENE": 1}
    assert {entry["token"] for entry in log["adjudicated"]} == {"PPA1", "FEN1"}
