# tests/torchcell/datasets/scerevisiae/test_costanzo2021.py
# [[tests.torchcell.datasets.scerevisiae.test_costanzo2021]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_costanzo2021.py
"""Costanzo 2021 loader: shared media, the ORF retention rule, and dose provenance.

Every test is synthetic: the fitness sheet is monkeypatched and the genome is a stub
resolver, so nothing here needs the raw mirror or a built LMDB.

2026.09.30 (Phase 13): a real xlsx in the released layout (a decoy first sheet, then
``Diff. Mutant fitness_Conditions`` with the four identity columns and two condition
columns, one header written ``" benomyl "`` to exercise the strip + lower-case lookup),
with ``_CONDITIONS`` cut to Benomyl and Galactose and a stub resolver (``_StubGenome``
plus ``AMB1`` -> AMBIGUOUS and every unknown name -> RETIRED, as the real resolver does).
Seven rows, in order, with (Benomyl, Galactose):

1. YAL001C TFC3 dma1 (0.1, 0.2) -> records 0, 1 (KanMX deletion named TFC3)
2. YFL039C ACT1 act1-101 tsa1 (-0.5, blank) -> record 2 (ts allele)
3. YFL039C ACT1 act1-102 tsa2 (-0.25, 0.0) -> records 3, 4 (same ORF, a second
   allele: the duplicate identifier; 0.0 is a value, not a blank)
4. YAR044W (no gene name) dma2 (0.3, blank) -> record 5, stored as YAR042W with
   ``perturbed_gene_name`` "YAR044W" (the source ORF, line 731)
5. YER108C (0.9, 0.9) -> dropped, non_gene_feature, 2 cells
6. YAR037W (0.8, blank) -> dropped, retired, 1 cell
7. AMB1 (0.7, 0.7) -> dropped, ambiguous, resolved_to None, candidates
   [YAL001C, YFL039C], 2 cells

So 6 records from 4 of 7 strains; 3 strains and 2 + 1 + 2 = 5 cells dropped, by status
{non_gene_feature: 1, retired: 1, ambiguous: 1}; the ledger is sorted by source name,
"AMB1" < "YAR037W" < "YER108C". Every record shares the one reference (differential 0 on
SGA_DM_SELECTION at 26 C), so the reference index is one entry over members 0..5. Records are compared whole against a
hand-built ``EnvironmentResponseExperiment``: Benomyl 30 g/L (the "30 mg/mL" cell) on
SGA_DM_SELECTION, 26.0 C (a derivation, recorded in ``_TEMPERATURE.note``), aerobic, a
``duration_hours`` gap; ``n_samples`` 3 screens, no uncertainty (none released).

The sha256 contract (issue #524, fixed): ``download`` hashes the mirror file before
copying, so a refusal leaves nothing in ``raw/``; ``process`` verifies the file in
``raw/`` against the pin before reading a row; ``deposit_raw_mirror`` checks the source
before creating any mirror directory.

The strain-row contract (issue #524, fixed 2026.10.01): a blank Systematic Name raises
``BlankSystematicNameError`` and a strain on two rows (the sheet-row-2 strain again, padded)
raises ``RepeatedStrainRowError``, both before any store is opened; an AMBIGUOUS drop
records the resolver's candidate list. The released sheet has neither a blank name nor a
repeated strain (0 of 4,429 rows), so no stored record changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.datamodels.compound_identity import (
    CompoundResolutionStatus,
    resolve_compound_identity,
    resolved_compound,
)
from torchcell.datamodels.media import SGA_DM_SELECTION, SGA_DM_SELECTION_GALACTOSE
from torchcell.datamodels.schema import (
    AssayType,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Genotype,
    MeasurementType,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SgaKanMxDeletionPerturbation,
    SgaTsAllelePerturbation,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.scerevisiae import costanzo2021 as c
from torchcell.sequence.genome.scerevisiae.s288c import (
    GeneNameResolution,
    GeneNameStatus,
    SCerevisiaeGenome,
)
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

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


# --------------------------------------------------------------------------- #
# Phase 13: the real-xlsx build, the ledger, the refusals, the raw mirror
# --------------------------------------------------------------------------- #
_ID_COLUMNS = [
    "Systematic Name",
    "Gene Name",
    "Allele (Essential genes only)",
    "Strain ID",
]
_SHEET_ROWS: list[tuple[Any, ...]] = [
    ("YAL001C", "TFC3", None, "dma1", 0.1, 0.2),
    ("YFL039C", "ACT1", "act1-101", "tsa1", -0.5, None),
    ("YFL039C", "ACT1", "act1-102", "tsa2", -0.25, 0.0),
    ("YAR044W", None, None, "dma2", 0.3, None),
    ("YER108C", None, None, "dma3", 0.9, 0.9),
    ("YAR037W", None, None, "dma4", 0.8, None),
    ("AMB1", None, None, "dma8", 0.7, 0.7),
]


class _FullStub(_StubGenome):
    """``_StubGenome`` plus the two statuses it lacks: AMBIGUOUS, and RETIRED for any
    name it does not know (what the real resolver returns for an off-annotation name).
    """

    known = {"YAL001C", "YFL039C"} | set(_RENAME) | set(_NON_GENE) | _RETIRED

    def resolve_gene_name(self, name: str) -> GeneNameResolution:
        if name == "AMB1":
            return GeneNameResolution(
                input_name=name,
                status=GeneNameStatus.AMBIGUOUS,
                systematic_name=None,
                candidates=["YAL001C", "YFL039C"],
            )
        if name.upper() not in self.known:
            return GeneNameResolution(
                input_name=name, status=GeneNameStatus.RETIRED, systematic_name=name
            )
        return super().resolve_gene_name(name)


def _write_xlsx(
    path: Path,
    columns: list[str] | None = None,
    rows: list[tuple[Any, ...]] | None = None,
) -> None:
    """The released layout: a decoy first sheet, then the differential-fitness sheet."""
    header = columns or [*_ID_COLUMNS, " benomyl ", "Galactose"]
    frame = pd.DataFrame(rows or _SHEET_ROWS, columns=header)
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"Condition": ["Benomyl"]}).to_excel(
            writer, sheet_name=c._CONDITIONS_SHEET, index=False
        )
        frame.to_excel(writer, sheet_name=c._FITNESS_SHEET, index=False)


def _two_conditions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        c,
        "_CONDITIONS",
        [s for s in c._CONDITIONS if s["col"] in {"Benomyl", "Galactose"}],
    )


@pytest.fixture
def sheet_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> c.EnvChemgenCostanzo2021Dataset:
    """The seven-row xlsx of the module docstring, read by the real ``pd.read_excel``."""
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    _two_conditions(monkeypatch)
    (tmp_path / "raw").mkdir()
    _write_xlsx(tmp_path / "raw" / c._S1_FILENAME)
    return c.EnvChemgenCostanzo2021Dataset(
        root=str(tmp_path), genome=cast(SCerevisiaeGenome, _FullStub())
    )


def _gap() -> list[ProvenanceGap]:
    return [
        ProvenanceGap(
            field="duration_hours",
            reason=ProvenanceGapReason.deferred_pending_source_review,
        )
    ]


def _benomyl_environment() -> Environment:
    return Environment(
        media=SGA_DM_SELECTION,
        temperature=Temperature(value=26.0),
        perturbations=[
            SmallMoleculePerturbation(
                compound=resolved_compound("benomyl"),
                concentration=Concentration(value=30.0, unit=ConcentrationUnit.g_per_l),
            )
        ],
        aerobicity="aerobic",
        provenance_gaps=_gap(),
    )


def _phenotype(value: float) -> EnvironmentResponsePhenotype:
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.differential_fitness,
        assay_type=AssayType.colony_size_array,
        environment_response=value,
        n_samples=3,
        sample_unit=SampleUnit.screen,
        units=c.MEASUREMENT_UNITS,
    )


def _reference() -> dict[str, Any]:
    return EnvironmentResponseExperimentReference(
        dataset_name="EnvChemgenCostanzo2021Dataset",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        ),
        environment_reference=Environment(
            media=SGA_DM_SELECTION,
            temperature=Temperature(value=26.0),
            perturbations=[],
            aerobicity="aerobic",
            provenance_gaps=_gap(),
        ),
        phenotype_reference=_phenotype(0.0),
    ).model_dump()


def test_sheet_build_writes_the_records_in_row_then_condition_order(
    sheet_built: c.EnvChemgenCostanzo2021Dataset,
) -> None:
    """Rows 1, 2, 3, 4 kept; within a row, Benomyl before Galactose; blanks skipped."""
    got = [
        (
            p["systematic_gene_name"],
            p["perturbed_gene_name"],
            p["strain_id"],
            r["experiment"]["environment"]["media"]["name"],
            r["experiment"]["phenotype"]["environment_response"],
        )
        for r in (sheet_built[i] for i in range(len(sheet_built)))
        for p in r["experiment"]["genotype"]["perturbations"]
    ]
    benomyl = SGA_DM_SELECTION.name
    galactose = SGA_DM_SELECTION_GALACTOSE.name
    assert got == [
        ("YAL001C", "TFC3", "dma1", benomyl, 0.1),
        ("YAL001C", "TFC3", "dma1", galactose, 0.2),
        ("YFL039C", "act1-101", "tsa1", benomyl, -0.5),
        ("YFL039C", "act1-102", "tsa2", benomyl, -0.25),
        ("YFL039C", "act1-102", "tsa2", galactose, 0.0),
        ("YAR042W", "YAR044W", "dma2", benomyl, 0.3),
    ]


def test_full_deletion_record_equals_the_hand_built_experiment(
    sheet_built: c.EnvChemgenCostanzo2021Dataset,
) -> None:
    """Record 0 (YAL001C, Benomyl 0.1): experiment, reference and publication whole."""
    experiment = EnvironmentResponseExperiment(
        dataset_name="EnvChemgenCostanzo2021Dataset",
        genotype=Genotype(
            perturbations=[
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name="YAL001C",
                    perturbed_gene_name="TFC3",
                    strain_id="dma1",
                )
            ]
        ),
        environment=_benomyl_environment(),
        phenotype=_phenotype(0.1),
    )
    record = sheet_built[0]
    assert record["experiment"] == experiment.model_dump()
    assert record["reference"] == _reference()
    assert (
        record["publication"]
        == Publication(
            doi="10.1126/science.abf8424",
            doi_url="https://doi.org/10.1126/science.abf8424",
        ).model_dump()
    )


def test_ts_allele_edge_record_and_the_medium_condition_record(
    sheet_built: c.EnvChemgenCostanzo2021Dataset,
) -> None:
    """Record 4: the second allele of an ORF on galactose, where 0.0 is a value."""
    experiment = EnvironmentResponseExperiment(
        dataset_name="EnvChemgenCostanzo2021Dataset",
        genotype=Genotype(
            perturbations=[
                SgaTsAllelePerturbation(
                    systematic_gene_name="YFL039C",
                    perturbed_gene_name="act1-102",
                    strain_id="tsa2",
                )
            ]
        ),
        environment=Environment(
            media=SGA_DM_SELECTION_GALACTOSE,
            temperature=Temperature(value=26.0),
            perturbations=[],
            aerobicity="aerobic",
            provenance_gaps=_gap(),
        ),
        phenotype=_phenotype(0.0),
    )
    assert sheet_built[4]["experiment"] == experiment.model_dump()
    phenotype = sheet_built[4]["experiment"]["phenotype"]
    assert phenotype["environment_response_uncertainty"] is None
    assert phenotype["environment_response_uncertainty_type"] is None


def test_drop_log_is_written_exactly(
    sheet_built: c.EnvChemgenCostanzo2021Dataset,
) -> None:
    """The ledger in full: an AMBIGUOUS drop carries the resolver's two candidates, the
    other drops an empty candidate list (issue #524).
    """
    log = json.loads((Path(sheet_built.root) / c._DROPPED_FILENAME).read_text())
    assert log == {
        "dataset": "EnvChemgenCostanzo2021Dataset",
        "rule": c.DROP_RULE,
        "n_source_strains": 7,
        "n_kept_strains": 4,
        "n_kept_records": 6,
        "n_dropped_strains": 3,
        "n_dropped_records": 5,
        "dropped_by_status": {"non_gene_feature": 1, "retired": 1, "ambiguous": 1},
        "dropped": [
            {
                "source_name": "AMB1",
                "status": "ambiguous",
                "resolved_to": None,
                "feature_type": None,
                "n_records": 2,
                "candidates": ["YAL001C", "YFL039C"],
            },
            {
                "source_name": "YAR037W",
                "status": "retired",
                "resolved_to": "YAR037W",
                "feature_type": None,
                "n_records": 1,
                "candidates": [],
            },
            {
                "source_name": "YER108C",
                "status": "non_gene_feature",
                "resolved_to": "YER109C",
                "feature_type": "blocked_reading_frame",
                "n_records": 2,
                "candidates": [],
            },
        ],
    }


def test_one_reference_covers_every_record(
    sheet_built: c.EnvChemgenCostanzo2021Dataset,
) -> None:
    """All six records share the one reference, so the index is one entry."""
    index = sheet_built.experiment_reference_index
    assert index is not None
    (entry,) = index
    assert entry.member_indices == [0, 1, 2, 3, 4, 5]
    assert entry.reference.model_dump() == _reference()


def _refused_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: tuple[Any, ...]
) -> pytest.ExceptionInfo[ValueError]:
    """Build over the seven rows plus ``extra`` (sheet row 9); return the refusal."""
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    _two_conditions(monkeypatch)
    (tmp_path / "raw").mkdir()
    _write_xlsx(tmp_path / "raw" / c._S1_FILENAME, rows=[*_SHEET_ROWS, extra])
    with pytest.raises(ValueError) as err:
        c.EnvChemgenCostanzo2021Dataset(
            root=str(tmp_path), genome=cast(SCerevisiaeGenome, _FullStub())
        )
    assert not (tmp_path / "processed" / "lmdb").exists()
    assert not (tmp_path / c._DROPPED_FILENAME).exists()
    return err


def test_a_blank_systematic_name_is_refused_before_any_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row with no Systematic Name names no strain: refused with its sheet row (9),
    not resolved as the string ``"nan"`` and dropped as RETIRED.
    """
    err = _refused_build(tmp_path, monkeypatch, (None, None, None, "dma9", 0.5, None))
    assert type(err.value) is c.BlankSystematicNameError
    assert str(err.value) == (
        "Data File S1 sheet rows [9] have a blank 'Systematic Name'; a row without an "
        "identifier names no strain, refusing to resolve it"
    )


def test_a_whitespace_systematic_name_is_refused_as_blank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A name of only spaces is blank after stripping: refused with its sheet row (9),
    not resolved as an empty string.
    """
    err = _refused_build(tmp_path, monkeypatch, ("  ", None, None, "dma9", 0.5, None))
    assert type(err.value) is c.BlankSystematicNameError
    assert str(err.value) == (
        "Data File S1 sheet rows [9] have a blank 'Systematic Name'; a row without an "
        "identifier names no strain, refusing to resolve it"
    )


def test_a_repeated_strain_row_is_refused_naming_both_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Row 9 repeats the row-2 strain with padding (`` YAL001C ``, `` dma1 ``): refused,
    naming the stripped strain and both sheet rows, instead of storing its two cells a
    second time with no ledger entry.
    """
    err = _refused_build(
        tmp_path, monkeypatch, (" YAL001C ", "TFC3", None, " dma1 ", 0.1, 0.2)
    )
    assert type(err.value) is c.RepeatedStrainRowError
    assert str(err.value) == (
        "Data File S1 repeats strain YAL001C (dma1) at sheet rows 2 and 9; one strain "
        "is one row, refusing to store its condition cells twice"
    )


def test_the_same_orf_under_another_strain_id_is_not_a_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows 3 and 4 share YFL039C as two alleles (tsa1, tsa2): two strains, both built."""
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    _two_conditions(monkeypatch)
    (tmp_path / "raw").mkdir()
    _write_xlsx(tmp_path / "raw" / c._S1_FILENAME, rows=_SHEET_ROWS[1:3])
    dataset = c.EnvChemgenCostanzo2021Dataset(
        root=str(tmp_path), genome=cast(SCerevisiaeGenome, _FullStub())
    )
    assert [
        dataset[i]["experiment"]["genotype"]["perturbations"][0]["strain_id"]
        for i in range(len(dataset))
    ] == ["tsa1", "tsa2", "tsa2"]


def test_side_files_gene_set_and_reference_index(
    sheet_built: c.EnvChemgenCostanzo2021Dataset,
) -> None:
    """The gene set holds the CURRENT names only (YAR044W is stored as YAR042W)."""
    preprocess = Path(sheet_built.preprocess_dir)
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YAR042W",
        "YFL039C",
    ]
    stored = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in stored] == [[0, 1, 2, 3, 4, 5]]


def test_every_condition_dose_maps_to_its_typed_concentration() -> None:
    """The 13 small-molecule doses as Concentrations (mg/mL recorded as g/L); galactose
    is the derived medium and carries no perturbation.
    """
    table: dict[str, tuple[float, ConcentrationUnit]] = {}
    for spec in c._CONDITIONS:
        environment = _dataset()._environment(spec)
        if spec["kind"] == "medium":
            assert (spec["name"], environment.perturbations) == ("galactose", [])
            continue
        (entry,) = environment.perturbations
        dose = cast(SmallMoleculePerturbation, entry).concentration
        assert dose.value is not None and dose.unit is not None
        table[spec["name"]] = (dose.value, dose.unit)
    g_l, mm, nm = (
        ConcentrationUnit.g_per_l,
        ConcentrationUnit.millimolar,
        ConcentrationUnit.nanomolar,
    )
    assert table == {
        "actinomycin D": (20.0, mm),
        "benomyl": (30.0, g_l),
        "bortezomib": (1300.0, mm),
        "caspofungin": (0.1, g_l),
        "concanamycin A": (100.0, nm),
        "cycloheximide": (0.1, g_l),
        "fluconazole": (16.0, g_l),
        "geldanamycin": (10.0, mm),
        "methyl methanesulfonate": (0.01, ConcentrationUnit.percent_v_v),
        "monensin": (50.0, g_l),
        "rapamycin": (100.0, nm),
        "sorbitol": (1.0, ConcentrationUnit.molar),
        "tunicamycin": (1.0, g_l),
    }


def test_temperature_is_a_recorded_derivation_and_n_samples_counts_screens() -> None:
    assert c._TEMPERATURE.value == 26.0
    assert (c._TEMPERATURE.note or "").startswith("DERIVATION, not a quote")
    assert c._N_SAMPLES.value == 3
    assert c._N_SAMPLES.quote == c._TEMPERATURE.quote
    assert c._VARIANCE_NOTE.value is None


def test_process_refuses_without_a_genome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / c._S1_FILENAME).write_bytes(b"")
    with pytest.raises(RuntimeError) as err:
        c.EnvChemgenCostanzo2021Dataset(root=str(tmp_path), genome=None)
    assert str(err.value) == (
        "EnvChemgenCostanzo2021Dataset requires a genome for R64 ORF resolution; "
        "inject SCerevisiaeGenome(...)"
    )
    assert not (tmp_path / "processed" / "lmdb").exists()


def test_a_sheet_missing_a_condition_column_refuses_with_its_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The header lookup is strict: a condition absent from the sheet is a KeyError."""
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    _two_conditions(monkeypatch)
    (tmp_path / "raw").mkdir()
    _write_xlsx(
        tmp_path / "raw" / c._S1_FILENAME, columns=[*_ID_COLUMNS, "Benomyl", "Glucose"]
    )
    with pytest.raises(KeyError, match="^'galactose'$"):
        c.EnvChemgenCostanzo2021Dataset(
            root=str(tmp_path), genome=cast(SCerevisiaeGenome, _FullStub())
        )


def test_a_workbook_without_the_fitness_sheet_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    (tmp_path / "raw").mkdir()
    pd.DataFrame({"x": [1]}).to_excel(
        tmp_path / "raw" / c._S1_FILENAME, sheet_name="Sheet1", index=False
    )
    with pytest.raises(
        ValueError,
        match="^Worksheet named 'Diff. Mutant fitness_Conditions' not found$",
    ):
        c.EnvChemgenCostanzo2021Dataset(
            root=str(tmp_path), genome=cast(SCerevisiaeGenome, _FullStub())
        )


def test_download_refuses_when_the_mirror_file_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "dr"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    dataset = _dataset()
    dataset.root = str(tmp_path / "build")
    mirror = osp.join(
        str(data_root),
        "torchcell-raw",
        "costanzoEnvironmentalRobustnessGlobal2021",
        "data",
        c._S1_FILENAME,
    )
    with pytest.raises(RuntimeError) as err:
        dataset.download()
    assert str(err.value) == (
        f"raw-mirror file not found: {mirror}. Costanzo 2021's Science SI is not "
        "scriptable (403); deposit Data File S1 from "
        "https://www.science.org/doi/10.1126/science.abf8424 into the raw mirror with "
        "deposit_raw_mirror(), then rebuild (sha256 verified)."
    )
    assert os.listdir(tmp_path / "build" / "raw") == []


def _mirror_file(data_root: Path) -> Path:
    path = Path(c.raw_mirror_dir(str(data_root))) / c._S1_RAW_RELPATH
    path.parent.mkdir(parents=True)
    _write_xlsx(path)
    return path


def test_a_failed_sha256_check_leaves_nothing_in_raw_and_every_retry_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #524): ``download`` hashes the mirror file BEFORE copying, so a
    mirror file off the pin raises ``RawSha256MismatchError`` naming the mirror path and
    both digests, ``raw/`` stays empty (no copy, no ``.partial``), and the next
    construction runs ``download`` again and refuses again instead of building.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    _two_conditions(monkeypatch)
    data_root = tmp_path / "dr"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    digest = hashlib.sha256(_mirror_file(data_root).read_bytes()).hexdigest()
    root = str(tmp_path / "build")
    genome = cast(SCerevisiaeGenome, _FullStub())

    mirror = Path(c.raw_mirror_dir(str(data_root))) / c._S1_RAW_RELPATH
    expected = (
        f"sha256 mismatch for {mirror}: expected "
        f"f6c313de416ce8cc6ae87e2020b4389bd4adeb07cdb6a438aecaf1e45e6228ad, "
        f"observed {digest}"
    )
    for _ in range(2):
        with pytest.raises(RawSha256MismatchError) as err:
            c.EnvChemgenCostanzo2021Dataset(root=root, genome=genome)
        assert str(err.value) == expected
        assert os.listdir(Path(root) / "raw") == []
        assert not (Path(root) / "processed" / "lmdb").exists()


def test_download_copies_from_the_mirror_and_verifies_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the pin set to the synthetic file's digest the build runs through download."""
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    _two_conditions(monkeypatch)
    data_root = tmp_path / "dr"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    source = _mirror_file(data_root)
    monkeypatch.setattr(
        c, "_S1_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    root = tmp_path / "build"
    built = c.EnvChemgenCostanzo2021Dataset(
        root=str(root), genome=cast(SCerevisiaeGenome, _FullStub())
    )
    assert (root / "raw" / c._S1_FILENAME).read_bytes() == source.read_bytes()
    assert len(built) == 6


class _FrozenDatetime:
    @staticmethod
    def now(tz: Any = None) -> datetime:
        return datetime(2026, 9, 30, 12, 0, tzinfo=UTC)


def test_deposit_raw_mirror_writes_the_file_and_an_exact_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first deposit copies the bytes and writes the manual-browser manifest; a
    second deposit of the same bytes is a no-op on the file.
    """
    source = tmp_path / "S1.xlsx"
    source.write_bytes(b"synthetic data file S1")
    pin = hashlib.sha256(b"synthetic data file S1").hexdigest()
    monkeypatch.setattr(c, "_S1_SHA256", pin)
    monkeypatch.setattr(c, "datetime", _FrozenDatetime)
    data_root = tmp_path / "dr"

    root = c.deposit_raw_mirror(
        source_xlsx=str(source), retrieved_at="2026-09-12", data_root=str(data_root)
    )

    assert root == str(
        data_root / "torchcell-raw" / "costanzoEnvironmentalRobustnessGlobal2021"
    )
    dest = Path(root) / "data" / c._S1_FILENAME
    assert dest.read_bytes() == b"synthetic data file S1"
    url = "https://www.science.org/doi/10.1126/science.abf8424"
    assert json.loads((Path(root) / "manifest.json").read_text()) == {
        "version": 1,
        "citation_key": "costanzoEnvironmentalRobustnessGlobal2021",
        "doi": "10.1126/science.abf8424",
        "title": "Environmental robustness of the global yeast genetic interaction "
        "network",
        "library_id": "6582362",
        "zotero_item_key": "CJ5NIJI9",
        "collections": [],
        "files": [
            {
                "path": f"data/{c._S1_FILENAME}",
                "role": "raw_data",
                "bytes": 22,
                "sha256": pin,
                "source": url,
                "zotero_md5": None,
                "retrieval": {
                    "method": "manual_browser",
                    "source_url": url,
                    "retriever": "manual",
                    "params": {"retrieval_command": c.MANUAL_RECIPE},
                    "sha256": pin,
                    "retrieved_at": "2026-09-12",
                    "last_check": None,
                },
                "processing": None,
            }
        ],
        "si_data_sources": [url],
        "si_expected": [
            "Data file S1 (Costanzo et al_Data File S1_Conditions_Strains_Fitness.xlsx)",
            "Supplementary Materials PDF (names the reference condition's solvent; NOT "
            "mirrored, and the reason SmallMoleculePerturbation.solvent is None here)",
        ],
        "provenance_complete": True,
        "created_at": "2026-09-30T12:00:00+00:00",
    }
    mtime = dest.stat().st_mtime_ns
    c.deposit_raw_mirror(
        source_xlsx=str(source), retrieved_at="2026-09-12", data_root=str(data_root)
    )
    assert dest.stat().st_mtime_ns == mtime


def test_manual_recipe_names_the_url_the_file_and_the_real_pin() -> None:
    assert c.MANUAL_RECIPE == (
        "manual browser download -- science.org returns HTTP 403 behind a Cloudflare "
        "challenge to any client (verified 2026-09-12), so: open "
        "https://www.science.org/doi/10.1126/science.abf8424 in a signed-in browser, "
        "follow 'Supplementary Materials', download 'Data file S1' "
        "(Costanzo et al_Data File S1_Conditions_Strains_Fitness.xlsx), and verify "
        "sha256 f6c313de416ce8cc6ae87e2020b4389bd4adeb07cdb6a438aecaf1e45e6228ad"
    )


def test_deposit_refuses_a_source_off_the_pin_before_creating_the_mirror_dir(
    tmp_path: Path,
) -> None:
    """Contract (issue #524): ``deposit_raw_mirror`` checks the source digest before it
    creates anything, so a refused deposit leaves no mirror directory behind.
    """
    source = tmp_path / "S1.xlsx"
    source.write_bytes(b"not the released file")
    digest = hashlib.sha256(b"not the released file").hexdigest()
    data_root = tmp_path / "dr"
    with pytest.raises(RawSha256MismatchError) as err:
        c.deposit_raw_mirror(
            source_xlsx=str(source), retrieved_at="x", data_root=str(data_root)
        )
    assert str(err.value) == (
        f"sha256 mismatch for {source}: expected "
        "f6c313de416ce8cc6ae87e2020b4389bd4adeb07cdb6a438aecaf1e45e6228ad, "
        f"observed {digest}"
    )
    assert not data_root.exists()


def test_deposit_refuses_to_overwrite_a_differing_mirror_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "S1.xlsx"
    source.write_bytes(b"synthetic data file S1")
    monkeypatch.setattr(
        c, "_S1_SHA256", hashlib.sha256(b"synthetic data file S1").hexdigest()
    )
    data_root = tmp_path / "dr"
    dest = Path(c.raw_mirror_dir(str(data_root))) / c._S1_RAW_RELPATH
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"someone else's bytes")
    with pytest.raises(RuntimeError) as err:
        c.deposit_raw_mirror(
            source_xlsx=str(source), retrieved_at="x", data_root=str(data_root)
        )
    assert str(err.value) == f"{dest} exists with a different sha256; refusing"
    assert dest.read_bytes() == b"someone else's bytes"


def test_raw_mirror_dir_uses_the_argument_then_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = "costanzoEnvironmentalRobustnessGlobal2021"
    assert c.raw_mirror_dir("/a") == f"/a/torchcell-raw/{key}"
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert c.raw_mirror_dir() == f"{tmp_path}/torchcell-raw/{key}"


def test_schema_classes_raw_file_and_the_inline_stubs() -> None:
    """The class wiring; ``preprocess_raw`` is an identity and ``create_experiment``
    refuses because both steps live inside ``process``.
    """
    dataset = _dataset()
    assert dataset.experiment_class is EnvironmentResponseExperiment
    assert dataset.reference_class is EnvironmentResponseExperimentReference
    assert dataset.raw_file_names == [
        "Costanzo et al_Data File S1_Conditions_Strains_Fitness.xlsx"
    ]
    frame = pd.DataFrame({"a": [1]})
    assert dataset.preprocess_raw(frame) is frame
    with pytest.raises(NotImplementedError):
        dataset.create_experiment()


def test_a_raw_file_off_the_pin_is_refused_at_build_time(
    off_pin_raw: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #524): with Data File S1 already in ``raw/`` PyG skips
    ``download``, so ``process`` verifies it first. A file off the pin raises
    ``RawSha256MismatchError`` naming the file and both digests before any row is read;
    no store is written and the file is left as found.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    staged = off_pin_raw(c, [c._S1_FILENAME])
    raw = staged.root / "raw" / c._S1_FILENAME
    with pytest.raises(RawSha256MismatchError) as err:
        c.EnvChemgenCostanzo2021Dataset(
            root=str(staged.root), genome=cast(SCerevisiaeGenome, _FullStub())
        )
    assert str(err.value) == (
        f"sha256 mismatch for {raw}: expected "
        "f6c313de416ce8cc6ae87e2020b4389bd4adeb07cdb6a438aecaf1e45e6228ad, "
        f"observed {staged.observed}"
    )
    assert os.listdir(staged.root / "processed") == []
    assert not (staged.root / "preprocess").exists()
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == staged.observed
