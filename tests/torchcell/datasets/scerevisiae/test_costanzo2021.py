# tests/torchcell/datasets/scerevisiae/test_costanzo2021.py
"""Costanzo 2021 loader: shared media, the ORF retention rule, and dose provenance.

Every test is synthetic: the fitness sheet is monkeypatched and the genome is a stub
resolver, so nothing here needs the raw mirror or a built LMDB.
"""

from __future__ import annotations

import json
import os.path as osp
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from torchcell.datamodels.compound_identity import (
    CompoundResolutionStatus,
    resolve_compound_identity,
)
from torchcell.datamodels.media import SGA_DM_SELECTION, SGA_DM_SELECTION_GALACTOSE
from torchcell.datamodels.schema import (
    AssayType,
    MeasurementType,
    SmallMoleculePerturbation,
)
from torchcell.datasets.scerevisiae import costanzo2021 as c
from torchcell.sequence.genome.scerevisiae.s288c import (
    GeneNameResolution,
    GeneNameStatus,
    SCerevisiaeGenome,
)

_RENAME = {"YAR044W": "YAR042W"}
_NON_GENE = {"YER108C": ("YER109C", "blocked_reading_frame")}
_RETIRED = {"YAR037W"}


class _StubGenome:
    """The two methods the loader asks a genome for, with a hand-written answer table."""

    def resolve_gene_name(self, name: str) -> GeneNameResolution:
        upper = name.upper()
        if upper in _RENAME:
            return GeneNameResolution(
                input_name=name,
                status=GeneNameStatus.RENAMED,
                systematic_name=_RENAME[upper],
            )
        if upper in _NON_GENE:
            systematic, feature = _NON_GENE[upper]
            return GeneNameResolution(
                input_name=name,
                status=GeneNameStatus.NON_GENE_FEATURE,
                systematic_name=systematic,
                feature_type=feature,
            )
        if upper in _RETIRED:
            return GeneNameResolution(
                input_name=name, status=GeneNameStatus.RETIRED, systematic_name=upper
            )
        return GeneNameResolution(
            input_name=name, status=GeneNameStatus.CURRENT, systematic_name=upper
        )


def _frame() -> pd.DataFrame:
    """Four strains: one CURRENT deletion, one ts allele, one RENAMED, one dropped each."""
    rows = [
        ("YAL001C", "TFC3", None, "dma1", 0.1, 0.2),
        ("YFL039C", "ACT1", "act1-101", "tsa1", -0.5, None),
        ("YAR044W", "YAR044W", None, "dma2", 0.3, 0.4),
        ("YER108C", None, None, "dma3", 0.9, 0.9),
        ("YAR037W", None, None, "dma4", 0.8, None),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "Systematic Name",
            "Gene Name",
            "Allele (Essential genes only)",
            "Strain ID",
            "Benomyl",
            "Galactose",
        ],
    )


def _dataset() -> c.EnvChemgenCostanzo2021Dataset:
    """An uninitialized instance: ``_environment`` reads no build state."""
    return c.EnvChemgenCostanzo2021Dataset.__new__(c.EnvChemgenCostanzo2021Dataset)


@pytest.fixture
def built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> c.EnvChemgenCostanzo2021Dataset:
    """A tiny end-to-end build over the synthetic sheet, with only 2 of the 14 columns."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / c._S1_FILENAME).write_bytes(b"")  # presence only: download() is skipped
    monkeypatch.setattr(
        c,
        "_CONDITIONS",
        [s for s in c._CONDITIONS if s["col"] in {"Benomyl", "Galactose"}],
    )
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: _frame())
    return c.EnvChemgenCostanzo2021Dataset(
        root=str(tmp_path), genome=cast(SCerevisiaeGenome, _StubGenome())
    )


def test_tunicamycin_is_identified_as_a_mixture_so_its_records_stay() -> None:
    """The review expected 4,390 dropped records; the curated table now identifies it."""
    resolution = resolve_compound_identity(name="tunicamycin")
    assert resolution.status is CompoundResolutionStatus.RESOLVED_MIXTURE
    assert resolution.inchikey is None
    assert resolution.chebi_id == "CHEBI:29699"
    assert resolution.identified  # the retention rule accepts ChEBI or CID
    assert {spec["name"] for spec in c._CONDITIONS} >= {"tunicamycin"}


def test_every_condition_compound_carries_a_structure_identifier() -> None:
    for spec in c._CONDITIONS:
        compound = resolve_compound_identity(name=spec["name"])
        assert compound.identified, spec["name"]


def test_small_molecule_condition_uses_the_shared_sga_medium() -> None:
    spec = next(s for s in c._CONDITIONS if s["col"] == "Benomyl")
    environment = _dataset()._environment(spec)
    assert environment.media == SGA_DM_SELECTION
    assert environment.temperature is not None
    assert environment.temperature.value == 26.0
    (entry,) = environment.perturbations
    perturbation = cast(SmallMoleculePerturbation, entry)
    assert perturbation.compound.inchikey == "RIOXQFHNBCKOKP-UHFFFAOYSA-N"
    assert perturbation.concentration.value == 30.0
    assert [gap.field for gap in environment.provenance_gaps] == ["duration_hours"]


def test_galactose_is_a_derived_medium_not_an_added_compound() -> None:
    spec = next(s for s in c._CONDITIONS if s["col"] == "Galactose")
    environment = _dataset()._environment(spec)
    assert environment.media == SGA_DM_SELECTION_GALACTOSE
    assert environment.media.base_medium == "SD_MSG"
    assert environment.perturbations == []


def test_dose_provenance_lives_on_the_condition_not_in_the_phenotype_units() -> None:
    bortezomib = next(s for s in c._CONDITIONS if s["name"] == "bortezomib")
    assert bortezomib["dose"].quote == "1300 mM"
    assert "implausible" in (bortezomib["dose"].note or "")
    assert bortezomib["dose"].provenance.sha256 == c._S1_SHA256
    galactose = next(s for s in c._CONDITIONS if s["name"] == "galactose")
    assert galactose["dose"].quote == "0.02"
    assert "2% w/v" in (galactose["dose"].note or "")


def test_build_keeps_renamed_orfs_and_drops_non_genes_and_retired(
    built: c.EnvChemgenCostanzo2021Dataset,
) -> None:
    # YAL001C 2 cells + YFL039C 1 (the galactose cell is empty) + YAR044W 2 = 5
    assert len(built) == 5
    log = json.loads(open(osp.join(built.root, c._DROPPED_FILENAME)).read())
    assert log["n_kept_strains"] == 3 and log["n_dropped_strains"] == 2
    assert log["dropped_by_status"] == {"non_gene_feature": 1, "retired": 1}
    assert log["n_dropped_records"] == 3
    assert "CURRENT or RENAMED" in log["rule"]
    stored = {entry["source_name"]: entry["resolved_to"] for entry in log["dropped"]}
    assert stored == {"YER108C": "YER109C", "YAR037W": "YAR037W"}


def test_renamed_orf_is_stored_under_its_current_systematic_name(
    built: c.EnvChemgenCostanzo2021Dataset,
) -> None:
    stored = {
        (p["systematic_gene_name"], p["perturbed_gene_name"])
        for i in range(len(built))
        for p in built[i]["experiment"]["genotype"]["perturbations"]
    }
    # the RENAMED strain is keyed to its CURRENT systematic name, while the sheet's own
    # name survives as the strain's perturbed_gene_name (what keeps a merged pair apart)
    assert ("YAR042W", "YAR044W") in stored
    assert not any(systematic == "YAR044W" for systematic, _ in stored)


def test_units_is_one_shared_definition_and_the_axes_are_typed(
    built: c.EnvChemgenCostanzo2021Dataset,
) -> None:
    units = {built[i]["experiment"]["phenotype"]["units"] for i in range(len(built))}
    assert units == {c.MEASUREMENT_UNITS}
    phenotype = built[0]["experiment"]["phenotype"]
    assert phenotype["measurement_type"] is MeasurementType.differential_fitness
    assert phenotype["assay_type"] is AssayType.colony_size_array
    assert phenotype["n_samples"] == 3


def test_reference_is_the_standard_sga_condition_at_zero(
    built: c.EnvChemgenCostanzo2021Dataset,
) -> None:
    reference = built[0]["reference"]
    assert reference["genome_reference"]["strain"] == "S288C"
    assert reference["environment_reference"]["media"] == SGA_DM_SELECTION.model_dump()
    assert reference["environment_reference"]["perturbations"] == []
    assert reference["phenotype_reference"]["environment_response"] == 0.0
