# tests/torchcell/datasets/ecoli/test_girgis2009.py
# [[tests.torchcell.datasets.ecoli.test_girgis2009]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_girgis2009.py
"""The Girgis 2009 antibiotic transposon loader (``torchcell.datasets.ecoli.girgis2009``).

Synthetic tests (run everywhere) build all four released sheets in ``tmp_path``. The
hermetic build uses the real ``EcoliK12MG1655Genome`` over the synthetic MG1655 assembly
of ``tests/torchcell/sequence/genome/_bacterial_fixtures.py``, with ``b0006`` given the
extra ``/gene_synonym`` ``b0010`` so the merged-locus case exists, served through a
stubbed ``resolve`` with the network refused. ``verify_raw_files`` is replaced with a
presence check (synthetic bytes cannot carry the real pins; the pins are asserted by the
refusal test and the data-gated tests).

The six synthetic loci, each carrying the same value in every hybridization of both
reference sets, so the combination rule's answer is that value:

    UNIQID  annotation            S3/S4    Dataset S5   Dataset S1
    b0001   thrL, current         +3.0     3.0          3.0 (fus is ND everywhere)
    b0002   thrA, current         -1.8     -1.8         -1.8 on the four 3-repetition
                                                        drugs, 0 on the other 13
    b0004   yaaP, pseudogene      +1.0/-1.0   0.0       0
    b0005   proB, current         +0.0005  0.0          0    (the rounded-to-zero path)
    b0010   /gene_synonym of b0006  +5.0    5.0         5.0  -> dropped, issue #753
    b0099   on no locus           -5.0     -5.0         -5.0 -> dropped, retired

Five of the six distinct names resolve (0.833 < 0.98), so the build fixture lowers
``MIN_RESOLVED_FRACTION`` and a separate test shows the default stops the build. 67
records: four kept loci x 17 drugs minus the one ``ND`` cell.

Data-gated tests (``--data``) read the real raw mirror and the built dev-tree LMDB under
``$DATA_ROOT`` (they never build it): the manifest pins, the provenance audit of every
sourced value and every Table 1 row, the measured counts and sign split, hand-checked
cells read off ``si24.xls``, and the recorded ledgers.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd
import pytest

import torchcell.datasets.ecoli.girgis2009 as m
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    MG1655_GAF,
    MG1655_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.data import ManifestPinMismatchError, RawSha256MismatchError
from torchcell.datamodels.media import M9, MEDIA_LIBRARY
from torchcell.datamodels.schema import (
    AssayType,
    AssemblyReferenceGenome,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    ConcentrationUnit,
    MeasurementType,
    MediaComponentRole,
    SampleUnit,
    SmallMoleculePerturbation,
    TransposonInsertionPerturbation,
)
from torchcell.datasets.bacteria_common import LocusTagResolutionError
from torchcell.datasets.dataset_registry import dataset_registry
from torchcell.literature.manifest import RetrievalMethod, RetrievalRecord
from torchcell.sequence.genome.ecoli.k12 import MG1655_ASSEMBLY, EcoliK12MG1655Genome
from torchcell.verification.environment_response import _condition_signature
from torchcell.verification.sourced import ProvenanceGapReason, audit_sourced_value

REFERENCE = AssemblyReferenceGenome(
    species="Escherichia coli",
    strain=m.LIBRARY_PARENT,
    assembly_set="ecoli_K12_MG1655_ASM584v2",
    assembly_accession="GCA_000005845.2",
    background=m.library_background(),
)
#: The four drugs whose repetition count is 3, hence whose Dataset S1 cut is 1.5.
THREE_REPETITION = ("lom", "slf", "dox", "amp")
#: ``(UNIQID, description, per-hybridization value in S3, in S4, Dataset S5 cell)``.
SYNTHETIC_ROWS: tuple[tuple[str, str, float, float, Any], ...] = (
    ("b0001", "thrL thr operon leader peptide", 3.0, 3.0, 3.0),
    ("b0002", "thrA aspartokinase I", -1.8, -1.8, -1.8),
    ("b0004", "yaaP pseudogene", 1.0, -1.0, 0.0),
    ("b0005", "proB glutamate 5-kinase", 0.0005, 0.0005, 0.0),
    ("b0010", "newG merged into b0006", 5.0, 5.0, 5.0),
    ("b0099", "ghostG on no locus", -5.0, -5.0, -5.0),
)
KEPT_LOCI = ("b0001", "b0002", "b0004", "b0005")
#: The one drug whose b0001 cell is released as ND.
ND_DRUG = "fus"
#: The one hybridization released as LQ (b0002, amk, Dataset S3, first array).
LQ_CELL = ("b0002", "amk")
SYNTHETIC_RECORDS = len(KEPT_LOCI) * len(m.DRUGS) - 1


def _significant(code: str, value: float) -> Any:
    """The Dataset S1 cell of a Dataset S5 value: itself, or 0 below the drug's cut."""
    spec = m.DRUGS_BY_CODE[code]
    return value if abs(value) >= spec.threshold else 0.0


def _write_sheet(
    path: Path,
    spec: m.SheetSpec,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    title: str | None = None,
    extra_sheet: bool = False,
) -> None:
    """Write one sheet with its title cell, filler rows, header and data."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = spec.sheet_name
    sheet.append([spec.title if title is None else title])
    for _ in range(spec.header_row - 1):
        sheet.append(["a note"])
    sheet.append(list(columns))
    for row in rows:
        sheet.append(list(row))
    if extra_sheet:
        workbook.create_sheet("second")
    workbook.save(path)


def _reference_columns() -> list[str]:
    """The ``<CODE>_R<n>`` columns of Datasets S3 and S4, drug-major."""
    return [
        f"{spec.code.upper()}_R{n}"
        for spec in m.DRUGS
        for n in range(1, spec.hybridizations + 1)
    ]


def _reference_rows(which: str) -> list[list[Any]]:
    """Dataset S3 (``"s3"``) or S4 (``"s4"``) rows for the synthetic loci."""
    rows: list[list[Any]] = []
    for b_number, description, s3, s4, released in SYNTHETIC_ROWS:
        value = s3 if which == "s3" else s4
        row: list[Any] = [b_number, description, 0.0, 0.5]
        for spec in m.DRUGS:
            for n in range(1, spec.hybridizations + 1):
                if b_number == "b0001" and spec.code == ND_DRUG:
                    row.append(m.LOW_QUALITY_MARKER)
                elif which == "s3" and (b_number, spec.code) == LQ_CELL and n == 1:
                    row.append(m.LOW_QUALITY_MARKER)
                else:
                    row.append(value)
        del description, released
        rows.append(row)
    return rows


def _combined_rows() -> list[list[Any]]:
    """Dataset S5 rows for the synthetic loci."""
    rows: list[list[Any]] = []
    for b_number, description, _s3, _s4, released in SYNTHETIC_ROWS:
        row: list[Any] = [b_number, description]
        for spec in m.DRUGS:
            if b_number == "b0001" and spec.code == ND_DRUG:
                row.append(m.NO_DATA_MARKER)
            else:
                row.append(released)
        rows.append(row)
    return rows


def _significant_rows() -> list[list[Any]]:
    """Dataset S1 rows: Dataset S5 with every insignificant cell zeroed."""
    rows: list[list[Any]] = []
    for b_number, description, _s3, _s4, released in SYNTHETIC_ROWS:
        row: list[Any] = [b_number, description]
        for spec in m.DRUGS:
            if b_number == "b0001" and spec.code == ND_DRUG:
                row.append(m.NO_DATA_MARKER)
            else:
                row.append(_significant(spec.code, released))
        rows.append(row)
    return rows


def _write_raw(raw: Path) -> None:
    """All four released sheets, under the names the loader links from the mirror."""
    raw.mkdir(parents=True, exist_ok=True)
    reference_columns = [
        "bnum",
        "Description",
        "Average",
        "Stdev",
        *_reference_columns(),
    ]
    combined_columns = ["UNIQID", "NAME", *(spec.code for spec in m.DRUGS)]
    _write_sheet(
        raw / m.DATASET_S1,
        m.SHEETS[m.DATASET_S1],
        combined_columns,
        _significant_rows(),
    )
    _write_sheet(
        raw / m.DATASET_S3,
        m.SHEETS[m.DATASET_S3],
        reference_columns,
        _reference_rows("s3"),
    )
    _write_sheet(
        raw / m.DATASET_S4,
        m.SHEETS[m.DATASET_S4],
        reference_columns,
        _reference_rows("s4"),
    )
    _write_sheet(
        raw / m.DATASET_S5, m.SHEETS[m.DATASET_S5], combined_columns, _combined_rows()
    )


def _frames(raw: Path) -> dict[str, pd.DataFrame]:
    return {name: m.read_sheet(raw / name, spec) for name, spec in m.SHEETS.items()}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def mg1655(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EcoliK12MG1655Genome:
    """The real MG1655 class over the synthetic assembly, b0010 a synonym of b0006."""
    loci = [
        locus.model_copy(update={"synonyms": (*locus.synonyms, "b0010")})
        if locus.tag == "b0006"
        else locus
        for locus in MG1655_LOCI
    ]
    files = write_assembly(tmp_path / "tier", MG1655_ASSEMBLY, loci, MG1655_GAF)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, files)
    root = tmp_path / "mg1655"
    root.mkdir()
    return EcoliK12MG1655Genome(genome_root=str(root), overwrite=False)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """A directory holding all four synthetic sheets."""
    raw = tmp_path / "sheets"
    _write_raw(raw)
    return raw


@pytest.fixture
def presence_only_pins(monkeypatch: pytest.MonkeyPatch) -> list[Mapping[str, str]]:
    """Replace the build-time byte check with a presence check that records the pins."""
    calls: list[Mapping[str, str]] = []

    def record(raw: str, pins: Mapping[str, str]) -> None:
        missing = [f for f in pins if not osp.exists(osp.join(raw, f))]
        if missing:
            raise FileNotFoundError(f"pinned raw files absent: {missing}")
        calls.append(dict(pins))

    monkeypatch.setattr(m, "verify_raw_files", record)
    return calls


@pytest.fixture
def synthetic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mg1655: EcoliK12MG1655Genome,
    presence_only_pins: list[Mapping[str, str]],
) -> Path:
    """A dataset root whose ``raw/`` holds the four synthetic sheets."""
    root = tmp_path / m.DATASET_ROOT_REL
    _write_raw(root / "raw")
    monkeypatch.setattr(m, "reference_genome", lambda *a, **k: REFERENCE)
    return root


@pytest.fixture
def built(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch, mg1655: EcoliK12MG1655Genome
) -> m.EnvChemgenGirgis2009Dataset:
    monkeypatch.setattr(m, "MIN_RESOLVED_FRACTION", 0.8)
    return m.EnvChemgenGirgis2009Dataset(root=str(synthetic), ecoli_genome=mg1655)


# --------------------------------------------------------------------------- #
# The condition table
# --------------------------------------------------------------------------- #
def test_the_seventeen_drugs_are_the_dataset_s5_columns() -> None:
    assert len(m.DRUGS) == 17
    assert m.DRUG_CODES == (
        "amk",
        "gen",
        "str",
        "tob",
        "lom",
        "nal",
        "slf",
        "dox",
        "tet",
        "blm",
        "fox",
        "ery",
        "nit",
        "amp",
        "pip",
        "trm",
        "fus",
    )
    assert len(m.DRUGS_BY_CODE) == 17


def test_repetitions_drop_the_averaged_technical_array() -> None:
    """The four drugs Text S1 names have three columns but two repetitions."""
    assert m.AVERAGED_CODES == ("PIP", "FOX", "TET", "TRM")
    for code in ("pip", "fox", "tet", "trm"):
        spec = m.DRUGS_BY_CODE[code]
        assert (spec.hybridizations, spec.repetitions) == (3, 2)
    assert {spec.code for spec in m.DRUGS if spec.repetitions == 3} == set(
        THREE_REPETITION
    )


def test_each_drug_threshold_follows_its_repetition_count() -> None:
    assert m.SIGNIFICANCE_THRESHOLDS == {2: 2.15, 3: 1.5}
    for spec in m.DRUGS:
        assert spec.threshold == (1.5 if spec.code in THREE_REPETITION else 2.15)


def test_screen_id_joins_the_code_and_table_1_day() -> None:
    assert m.DRUGS_BY_CODE["amk"].screen_id == "amk day 4"
    assert m.DRUGS_BY_CODE["dox"].screen_id == "dox day 2/3"
    assert len({spec.screen_id for spec in m.DRUGS}) == 17


def test_table_1_sample_column_disagrees_for_exactly_streptomycin_and_sulfa() -> None:
    """The published # Samples cell is right for 15 drugs and wrong for two."""
    disagree = {
        spec.code: (spec.table1_samples, spec.repetitions)
        for spec in m.DRUGS
        if spec.table1_samples != spec.repetitions
    }
    assert disagree == {"str": (3, 2), "slf": (2, 3)}


def test_every_drug_dose_quote_names_its_drug_and_code() -> None:
    for spec in m.DRUGS:
        value = spec.dose_sourced()
        assert value.value == spec.dose_ug_per_ml
        assert f"<td>{spec.name}</td>" in value.quote
        assert f"<td>{spec.code.upper()}</td>" in value.quote
        assert value.provenance.source_uri == m.PAPER_MD


# --------------------------------------------------------------------------- #
# Cells and the combination rule
# --------------------------------------------------------------------------- #
def test_cell_reads_numbers_and_the_two_markers() -> None:
    assert m._cell(1.5) == 1.5
    assert m._cell(2) == 2.0
    assert m._cell(m.NO_DATA_MARKER) is None
    assert m._cell(f" {m.LOW_QUALITY_MARKER} ") is None
    assert m._cell(None) is None
    assert m._cell(float("nan")) is None


@pytest.mark.parametrize("value", ["beneficial", True, float("inf")])
def test_cell_refuses_anything_else(value: Any) -> None:
    with pytest.raises(m.SheetFormatError):
        m._cell(value)


def test_combined_z_takes_the_score_closest_to_zero_when_signs_agree() -> None:
    assert m.combined_z([3.0, 1.2, 2.0, 5.0]) == 1.2
    assert m.combined_z([-3.0, -1.2, -2.0]) == -1.2
    assert m.combined_z([1.2]) == 1.2


def test_combined_z_is_zero_when_signs_disagree_or_a_score_is_zero() -> None:
    assert m.combined_z([3.0, -1.2]) == 0.0
    assert m.combined_z([3.0, 0.0]) == 0.0
    assert m.combined_z([0.0]) == 0.0


def test_drug_z_scores_keeps_every_hybridization_of_a_plain_drug() -> None:
    spec = m.DRUGS_BY_CODE["amk"]
    assert m.drug_z_scores([[1.0, 2.0], [3.0, 4.0]], spec) == [1.0, 2.0, 3.0, 4.0]


def test_drug_z_scores_averages_the_second_and_third_array_of_an_averaged_drug() -> (
    None
):
    spec = m.DRUGS_BY_CODE["trm"]
    assert m.drug_z_scores([[1.0, 2.0, 4.0], [5.0, 6.0, 8.0]], spec) == [
        1.0,
        3.0,
        5.0,
        7.0,
    ]


def test_drug_z_scores_drops_a_low_quality_array_and_its_whole_reference_set() -> None:
    spec = m.DRUGS_BY_CODE["amk"]
    assert m.drug_z_scores(
        [[m.LOW_QUALITY_MARKER, 2.0], [m.LOW_QUALITY_MARKER, m.LOW_QUALITY_MARKER]],
        spec,
    ) == [2.0]


def test_drug_z_scores_of_an_averaged_drug_with_one_usable_array() -> None:
    """A single surviving value is not averaged with itself."""
    spec = m.DRUGS_BY_CODE["fox"]
    assert m.drug_z_scores(
        [[2.0, m.LOW_QUALITY_MARKER, m.LOW_QUALITY_MARKER]], spec
    ) == [2.0]


def test_hybridization_columns_are_ordered_by_their_number() -> None:
    frame = pd.DataFrame(
        columns=["bnum", "FOX_R3", "FOX_R1", "FOX_R2", "FUS_R1", "FUS_R2"]
    )
    assert m.hybridization_columns(frame, "fox") == ["FOX_R1", "FOX_R2", "FOX_R3"]
    assert m.hybridization_columns(frame, "amk") == []


# --------------------------------------------------------------------------- #
# Reading the sheets
# --------------------------------------------------------------------------- #
def test_read_sheet_reads_every_row_of_dataset_s5(raw_dir: Path) -> None:
    frame = m.read_sheet(raw_dir / m.DATASET_S5, m.SHEETS[m.DATASET_S5])
    assert frame["UNIQID"].tolist() == [row[0] for row in SYNTHETIC_ROWS]
    assert frame["amk"].tolist() == [row[4] for row in SYNTHETIC_ROWS]
    assert frame[ND_DRUG].tolist()[0] == m.NO_DATA_MARKER


def test_read_sheet_refuses_a_changed_title(tmp_path: Path) -> None:
    spec = m.SHEETS[m.DATASET_S5]
    path = tmp_path / m.DATASET_S5
    _write_sheet(
        path,
        spec,
        ["UNIQID", "NAME", "amk"],
        [["b0001", "thrL", 1.0]],
        title="Dataset S5: something else",
    )
    with pytest.raises(m.SheetFormatError, match="title"):
        m.read_sheet(path, spec)


def test_read_sheet_refuses_a_second_sheet(tmp_path: Path) -> None:
    spec = m.SHEETS[m.DATASET_S5]
    path = tmp_path / m.DATASET_S5
    _write_sheet(
        path,
        spec,
        ["UNIQID", "NAME", "amk"],
        [["b0001", "thrL", 1.0]],
        extra_sheet=True,
    )
    with pytest.raises(m.SheetFormatError, match="sheets"):
        m.read_sheet(path, spec)


def test_read_sheet_refuses_a_missing_id_column(tmp_path: Path) -> None:
    spec = m.SHEETS[m.DATASET_S5]
    path = tmp_path / m.DATASET_S5
    _write_sheet(path, spec, ["GENE", "NAME", "amk"], [["b0001", "thrL", 1.0]])
    with pytest.raises(m.SheetFormatError, match="UNIQID"):
        m.read_sheet(path, spec)


def test_read_sheet_refuses_an_id_that_is_not_a_b_number(tmp_path: Path) -> None:
    spec = m.SHEETS[m.DATASET_S5]
    path = tmp_path / m.DATASET_S5
    _write_sheet(path, spec, ["UNIQID", "NAME", "amk"], [["thrL", "thrL", 1.0]])
    with pytest.raises(m.SheetFormatError, match="not b-numbers"):
        m.read_sheet(path, spec)


def test_check_sheets_align_returns_the_shared_ids(raw_dir: Path) -> None:
    assert m.check_sheets_align(_frames(raw_dir)) == [row[0] for row in SYNTHETIC_ROWS]


def test_check_sheets_align_refuses_a_reordered_sheet(raw_dir: Path) -> None:
    frames = _frames(raw_dir)
    frames[m.DATASET_S3] = frames[m.DATASET_S3].iloc[::-1].reset_index(drop=True)
    with pytest.raises(m.SheetFormatError, match="does not match"):
        m.check_sheets_align(frames)


def test_check_sheets_align_refuses_a_changed_description(raw_dir: Path) -> None:
    frames = _frames(raw_dir)
    frames[m.DATASET_S4] = frames[m.DATASET_S4].assign(
        Description=["other"] * len(SYNTHETIC_ROWS)
    )
    with pytest.raises(m.SheetFormatError, match="descriptions differ"):
        m.check_sheets_align(frames)


def test_check_sheets_align_refuses_a_repeated_id(raw_dir: Path) -> None:
    frames = _frames(raw_dir)
    doubled = frames[m.DATASET_S5].astype(object)
    doubled.loc[1, "UNIQID"] = "b0001"
    frames[m.DATASET_S5] = doubled
    with pytest.raises(m.SheetFormatError, match="repeated ids"):
        m.check_sheets_align(frames)


def test_check_sheets_align_refuses_a_missing_drug_column(raw_dir: Path) -> None:
    frames = _frames(raw_dir)
    frames[m.DATASET_S5] = frames[m.DATASET_S5].drop(columns=["fus"])
    with pytest.raises(m.SheetFormatError, match="no column for"):
        m.check_sheets_align(frames)


# --------------------------------------------------------------------------- #
# The two cross-checks on the release
# --------------------------------------------------------------------------- #
def test_check_combination_rule_reproduces_every_released_cell(raw_dir: Path) -> None:
    frames = _frames(raw_dir)
    checks = {
        check.drug: check
        for check in m.check_combination_rule(
            frames[m.DATASET_S3], frames[m.DATASET_S4], frames[m.DATASET_S5]
        )
    }
    assert not [c for c in checks.values() if c.disagreements]
    # b0005's 0.0005 is released as 0 in every drug: the rounded-to-zero path.
    assert all(check.n_rounded_to_zero == 1 for check in checks.values())
    assert checks["amk"].n_exact == 5 and checks["amk"].n_no_data == 0
    assert checks[ND_DRUG].n_exact == 4 and checks[ND_DRUG].n_no_data == 1
    assert checks["trm"].repetitions == 2 and checks["amp"].repetitions == 3


def test_check_combination_rule_flags_a_cell_the_rule_misses(raw_dir: Path) -> None:
    frames = _frames(raw_dir)
    moved = frames[m.DATASET_S5].astype(object)
    moved.loc[0, "amk"] = 9.0
    checks = {
        check.drug: check
        for check in m.check_combination_rule(
            frames[m.DATASET_S3], frames[m.DATASET_S4], moved
        )
    }
    assert len(checks["amk"].disagreements) == 1
    assert "the rule gives 3.0" in checks["amk"].disagreements[0]
    assert not checks["gen"].disagreements


def test_check_combination_rule_flags_a_value_with_no_usable_z_score(
    raw_dir: Path,
) -> None:
    frames = _frames(raw_dir)
    blanked = frames[m.DATASET_S3].astype(object)
    other = frames[m.DATASET_S4].astype(object)
    for frame in (blanked, other):
        for column in ("AMK_R1", "AMK_R2"):
            frame.loc[0, column] = m.LOW_QUALITY_MARKER
    checks = {
        check.drug: check
        for check in m.check_combination_rule(blanked, other, frames[m.DATASET_S5])
    }
    assert checks["amk"].disagreements == [
        "b0001: Dataset S5 gives 3.0 with no usable z-score"
    ]


def test_check_combination_rule_refuses_a_wrong_hybridization_count(
    raw_dir: Path,
) -> None:
    frames = _frames(raw_dir)
    short = frames[m.DATASET_S3].drop(columns=["AMK_R2"])
    with pytest.raises(m.SheetFormatError, match="hybridization columns"):
        m.check_combination_rule(short, frames[m.DATASET_S4], frames[m.DATASET_S5])


def test_check_thresholds_proves_each_drug_repetition_count(raw_dir: Path) -> None:
    frames = _frames(raw_dir)
    checks = {
        check.drug: check
        for check in m.check_thresholds(frames[m.DATASET_S1], frames[m.DATASET_S5])
    }
    for code in THREE_REPETITION:
        assert checks[code].threshold == 1.5
        assert checks[code].n_below_two_repetition_cut == 1
        assert checks[code].min_abs == 1.8
    assert checks["amk"].threshold == 2.15
    assert checks["amk"].n_below_two_repetition_cut == 0
    assert checks["amk"].min_abs == 3.0
    assert checks["amk"].n_significant == 3


def test_check_thresholds_refuses_a_significant_value_below_the_cut(
    raw_dir: Path,
) -> None:
    frames = _frames(raw_dir)
    lowered = frames[m.DATASET_S1].astype(object)
    lowered.loc[1, "amk"] = -1.8
    raised = frames[m.DATASET_S5].astype(object)
    with pytest.raises(m.ThresholdError, match="below the 2.15 cut"):
        m.check_thresholds(lowered, raised)


def test_check_thresholds_refuses_a_three_repetition_drug_with_no_low_hit(
    raw_dir: Path,
) -> None:
    frames = _frames(raw_dir)
    zeroed = frames[m.DATASET_S1].astype(object)
    zeroed.loc[1, "amp"] = 0.0
    with pytest.raises(m.ThresholdError, match="no significant"):
        m.check_thresholds(zeroed, frames[m.DATASET_S5])


def test_check_thresholds_refuses_dataset_s1_disagreeing_with_s5(raw_dir: Path) -> None:
    frames = _frames(raw_dir)
    moved = frames[m.DATASET_S1].astype(object)
    moved.loc[0, "amk"] = 4.0
    with pytest.raises(m.ThresholdError, match="Dataset S1 gives 4.0"):
        m.check_thresholds(moved, frames[m.DATASET_S5])


def test_check_thresholds_refuses_a_no_data_disagreement(raw_dir: Path) -> None:
    frames = _frames(raw_dir)
    moved = frames[m.DATASET_S1].astype(object)
    moved.loc[0, ND_DRUG] = 3.0
    with pytest.raises(m.ThresholdError, match="disagree on no-data"):
        m.check_thresholds(moved, frames[m.DATASET_S5])


# --------------------------------------------------------------------------- #
# Identifiers
# --------------------------------------------------------------------------- #
def test_resolve_b_numbers_keeps_only_locus_tags_of_the_pinned_annotation(
    mg1655: EcoliK12MG1655Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "MIN_RESOLVED_FRACTION", 0.8)
    kept, ledger = m.resolve_b_numbers(
        mg1655, [row[0] for row in SYNTHETIC_ROWS], label="synthetic"
    )
    assert sorted(kept) == list(KEPT_LOCI)
    assert kept["b0001"] == "thrL"
    assert kept["b0004"] == "yaaP"
    assert ledger.n_locus_tags == 4
    assert sorted(ledger.not_a_locus_tag) == ["b0010", "b0099"]
    assert ledger.not_a_locus_tag["b0010"].startswith("renamed: gene synonym of")
    assert ledger.not_a_locus_tag["b0099"].startswith("retired:")
    assert ledger.reconciliation.gene_namespace == "ecoli_k12_mg1655_bnumber"
    assert ledger.min_resolved_fraction == 0.8


def test_resolve_b_numbers_stops_below_the_default_threshold(
    mg1655: EcoliK12MG1655Genome,
) -> None:
    with pytest.raises(LocusTagResolutionError, match="0.98"):
        m.resolve_b_numbers(
            mg1655, [row[0] for row in SYNTHETIC_ROWS], label="synthetic"
        )


def test_canonical_symbol_falls_back_to_the_tag_when_the_symbol_moves(
    mg1655: EcoliK12MG1655Genome,
) -> None:
    assert m.canonical_symbol(mg1655, "b0002") == "thrA"
    assert m.canonical_symbol(mg1655, "b0007") == "insZ"


# --------------------------------------------------------------------------- #
# The retention ledger
# --------------------------------------------------------------------------- #
def test_build_drop_log_accounts_for_every_cell(raw_dir: Path) -> None:
    frames = _frames(raw_dir)
    kept = {tag: tag for tag in KEPT_LOCI}
    log = m.build_drop_log(
        "girgis", frames[m.DATASET_S5], [row[0] for row in SYNTHETIC_ROWS], kept
    )
    assert (log.source_loci, log.kept_loci) == (6, 4)
    assert log.source_records == 6 * 17
    assert log.kept_records == SYNTHETIC_RECORDS
    assert log.dropped_records == 6 * 17 - SYNTHETIC_RECORDS
    rules = {rule.rule: rule for rule in log.rules}
    assert rules["b_number_is_not_a_locus_tag_of_the_pinned_annotation"].n_records == 34
    assert rules["b_number_is_not_a_locus_tag_of_the_pinned_annotation"].items == [
        "b0010",
        "b0099",
    ]
    assert rules["no_combined_score_released"].items == [f"b0001 {ND_DRUG}"]


def test_build_drop_log_sign_split_is_per_drug(raw_dir: Path) -> None:
    frames = _frames(raw_dir)
    kept = {tag: tag for tag in KEPT_LOCI}
    log = m.build_drop_log(
        "girgis", frames[m.DATASET_S5], [row[0] for row in SYNTHETIC_ROWS], kept
    )
    by_drug = {drug.drug: drug for drug in log.drugs}
    assert (
        by_drug["amk"].n_positive,
        by_drug["amk"].n_negative,
        by_drug["amk"].n_zero,
    ) == (1, 1, 2)
    assert by_drug[ND_DRUG].no_data == 1
    assert by_drug[ND_DRUG].n_positive == 0
    assert by_drug["amk"].dropped_identifier == 2
    assert sum(drug.n_positive for drug in log.drugs) == 16
    assert sum(drug.n_negative for drug in log.drugs) == 17
    assert sum(drug.n_zero for drug in log.drugs) == 34


def test_stored_cells_skips_dropped_genes_and_no_data(raw_dir: Path) -> None:
    frames = _frames(raw_dir)
    kept = {tag: tag for tag in KEPT_LOCI}
    cells = list(
        m.stored_cells(frames[m.DATASET_S5], [row[0] for row in SYNTHETIC_ROWS], kept)
    )
    assert len(cells) == SYNTHETIC_RECORDS
    assert {tag for tag, _symbol, _spec, _value in cells} == set(KEPT_LOCI)
    assert (ND_DRUG, "b0001") not in {(spec.code, tag) for tag, _s, spec, _v in cells}
    assert cells[0] == ("b0001", "b0001", m.DRUGS_BY_CODE["amk"], 3.0)


# --------------------------------------------------------------------------- #
# Medium, environment, genotype, phenotype
# --------------------------------------------------------------------------- #
def test_the_medium_derives_from_the_m9_library_key() -> None:
    medium = m.GIRGIS_MEDIUM
    assert medium.base_medium == "M9"
    assert medium.base_medium in MEDIA_LIBRARY
    assert medium.state == "liquid"
    assert medium.is_synthetic is False
    assert len(medium.components) == len(M9.components) + 5


def test_the_m9_salts_assert_no_amount_but_keep_their_roles() -> None:
    salts = m.GIRGIS_MEDIUM.components[: len(M9.components)]
    assert [c.compound.name for c in salts] == [c.compound.name for c in M9.components]
    assert [c.role for c in salts] == [c.role for c in M9.components]
    assert all(c.concentration is None for c in salts)
    assert all("prints no amount" in (c.note or "") for c in salts)


def test_the_five_supplements_carry_the_paper_doses() -> None:
    doses = {
        c.compound.name: (c.concentration.value, c.concentration.unit, c.role)
        for c in m.GIRGIS_MEDIUM.components
        if c.concentration is not None
    }
    assert doses["D-glucose"] == (
        0.4,
        ConcentrationUnit.percent_w_v,
        MediaComponentRole.carbon_source,
    )
    assert doses["casamino acids"] == (
        0.1,
        ConcentrationUnit.percent_w_v,
        MediaComponentRole.complex_ingredient,
    )
    assert doses["magnesium sulfate"] == (
        1.0,
        ConcentrationUnit.millimolar,
        MediaComponentRole.bulk_salt,
    )
    assert doses["calcium chloride"] == (
        0.1,
        ConcentrationUnit.millimolar,
        MediaComponentRole.bulk_salt,
    )
    assert doses["thiamine"] == (
        1.5,
        ConcentrationUnit.micromolar,
        MediaComponentRole.vitamin,
    )


def test_casamino_acids_is_an_undefined_mixture() -> None:
    (component,) = [
        c for c in m.GIRGIS_MEDIUM.components if c.compound.name == "casamino acids"
    ]
    assert component.definition.value == "intrinsically_undefined"
    assert "casamino acids" in m.GIRGIS_MEDIUM.open_gaps
    assert m.GIRGIS_MEDIUM.is_fully_characterized is False


def test_each_environment_carries_one_antibiotic_at_its_dose() -> None:
    env = m.environment(m.DRUGS_BY_CODE["fus"])
    (perturbation,) = env.perturbations
    assert isinstance(perturbation, SmallMoleculePerturbation)
    assert perturbation.compound.name == "fusidic acid"
    assert perturbation.concentration.value == 180.0
    assert perturbation.concentration.unit == ConcentrationUnit.ug_per_ml
    assert env.temperature is not None and env.temperature.value == 37.0
    assert env.aerobicity == "aerobic"
    assert env.media.name == m.GIRGIS_MEDIUM.name


def test_the_duration_is_a_typed_gap_on_every_environment() -> None:
    for spec in m.DRUGS:
        (gap,) = m.environment(spec).provenance_gaps
        assert gap.field == "duration_hours"
        assert gap.reason is ProvenanceGapReason.not_reported_by_primary


def test_the_seventeen_conditions_have_distinct_signatures() -> None:
    signatures = {
        _condition_signature(
            {
                "environment": m.environment(spec).model_dump(),
                "phenotype": {"screen_id": spec.screen_id},
            }
        )
        for spec in m.DRUGS
    }
    assert len(signatures) == 17


def test_nine_antibiotics_are_name_only_and_carry_an_inchikey_gap() -> None:
    """The curated compound table is not edited here, so the absence is typed."""
    gapped = {
        spec.compound_label
        for spec in m.DRUGS
        if m.antibiotic(spec).compound.inchikey is None
    }
    assert gapped == {
        "amikacin",
        "ampicillin",
        "cefoxitin",
        "doxycycline hyclate",
        "fusidic acid",
        "gentamycin",
        "nitrofurantoin",
        "piperacillin",
        "streptomycin",
    }
    compound = m.antibiotic(m.DRUGS_BY_CODE["amk"]).compound
    assert [gap.field for gap in compound.provenance_gaps] == ["inchikey"]


def test_the_genotype_is_one_gene_level_transposon_insertion() -> None:
    (perturbation,) = m.insertion_genotype("b0002", "thrA").perturbations
    assert isinstance(perturbation, TransposonInsertionPerturbation)
    assert perturbation.systematic_gene_name == "b0002"
    assert perturbation.perturbed_gene_name == "thrA"
    assert perturbation.gene_namespace == "ecoli_k12_mg1655_bnumber"
    assert perturbation.identifier_mapping is None
    assert perturbation.barcode is None
    assert perturbation.insertion_position is None
    assert perturbation.insertion_strand is None
    assert perturbation.transposon is None


def test_the_four_absent_perturbation_fields_are_typed_gaps() -> None:
    gaps = {gap.field: gap for gap in m.PERTURBATION_FIELD_GAPS}
    assert sorted(gaps) == [
        "barcode",
        "insertion_position",
        "insertion_strand",
        "transposon",
    ]
    assert (
        gaps["transposon"].reason is ProvenanceGapReason.deferred_pending_source_review
    )
    assert gaps["transposon"].resolve_with is not None
    assert (
        gaps["transposon"].resolve_with.citation_key
        == "girgisComprehensiveGeneticCharacterization2007"
    )
    assert gaps["barcode"].reason is ProvenanceGapReason.not_reported_by_primary


def test_the_phenotype_is_a_signed_z_score_with_the_drug_replicate_count() -> None:
    spec = m.DRUGS_BY_CODE["amp"]
    leaf = m.phenotype(-1.25, spec)
    assert leaf.measurement_type is MeasurementType.z_score
    assert leaf.assay_type is AssayType.other
    assert leaf.environment_response == -1.25
    assert leaf.n_samples == 3
    assert leaf.sample_unit is SampleUnit.biological_replicate
    assert leaf.screen_id == "amp day 2"
    assert leaf.environment_response_se is None
    assert [gap.field for gap in leaf.provenance_gaps] == [
        "environment_response_uncertainty",
        "environment_response_se",
    ]


def test_the_reference_phenotype_scores_zero_in_the_same_condition() -> None:
    spec = m.DRUGS_BY_CODE["amk"]
    leaf = m.reference_phenotype(spec)
    assert leaf.environment_response == 0.0
    assert leaf.screen_id == spec.screen_id
    assert leaf.n_samples == 2
    assert leaf.units == m.UNITS_REFERENCE


def test_the_library_background_states_the_genotype_without_typing_an_allele() -> None:
    background = m.library_background()
    assert background.name == "MG1655 delta-lacZ"
    assert background.reference_strain == "MG1655"
    assert background.assembly_set == "ecoli_K12_MG1655_ASM584v2"
    assert background.parents == ["MG1655"]
    assert background.alleles == []
    assert background.genotype_statement == "MG1655 ∆lacZ"
    assert background.provenance is not None and len(background.provenance) == 2


def test_build_experiment_and_reference_validate_as_the_assembly_pinned_pair() -> None:
    spec = m.DRUGS_BY_CODE["tob"]
    env = m.environment(spec)
    experiment = m.build_experiment("girgis", "b0005", "proB", 2.5, spec, env)
    reference = m.build_reference("girgis", REFERENCE, spec, env)
    BacterialEnvironmentResponseExperiment.model_validate(experiment.model_dump())
    BacterialEnvironmentResponseExperimentReference.model_validate(
        reference.model_dump()
    )
    assert experiment.phenotype.environment_response == 2.5
    assert reference.phenotype_reference.environment_response == 0.0
    assert reference.genome_reference.strain == m.LIBRARY_PARENT


# --------------------------------------------------------------------------- #
# The hermetic build
# --------------------------------------------------------------------------- #
def test_the_build_stores_one_record_per_kept_cell(
    built: m.EnvChemgenGirgis2009Dataset,
) -> None:
    assert len(built) == SYNTHETIC_RECORDS


def test_the_build_checks_every_pinned_raw_file(
    built: m.EnvChemgenGirgis2009Dataset, presence_only_pins: list[Mapping[str, str]]
) -> None:
    assert presence_only_pins == [m.DATA_SHA256]
    assert sorted(m.DATA_SHA256) == [
        m.DATASET_S1,
        m.DATASET_S3,
        m.DATASET_S4,
        m.DATASET_S5,
    ]


def test_the_built_records_carry_the_released_values(
    built: m.EnvChemgenGirgis2009Dataset,
) -> None:
    records = [built[i] for i in range(len(built))]
    first = records[0]["experiment"]
    assert first["genotype"]["perturbations"][0]["systematic_gene_name"] == "b0001"
    assert first["genotype"]["perturbations"][0]["perturbation_type"] == (
        "transposon_insertion"
    )
    assert first["phenotype"]["environment_response"] == 3.0
    assert first["phenotype"]["screen_id"] == "amk day 4"
    assert first["phenotype"]["measurement_type"] == "z_score"
    values = [r["experiment"]["phenotype"]["environment_response"] for r in records]
    assert values.count(0.0) == 34
    assert values.count(3.0) == 16
    assert values.count(-1.8) == 17
    genes = {
        r["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
        for r in records
    }
    assert genes == set(KEPT_LOCI)


def test_the_build_writes_every_ledger(built: m.EnvChemgenGirgis2009Dataset) -> None:
    out = Path(built.preprocess_dir)
    log = json.loads((out / "dropped_records.json").read_text())
    assert log["kept_records"] == SYNTHETIC_RECORDS
    identifiers = json.loads((out / "identifier_reconciliation.json").read_text())
    assert sorted(identifiers["not_a_locus_tag"]) == ["b0010", "b0099"]
    replicates = json.loads((out / "replicate_structure.json").read_text())
    assert replicates["averaged_codes"] == ["PIP", "FOX", "TET", "TRM"]
    assert sorted(
        line.split(":")[0] for line in replicates["table1_disagreements"]
    ) == ["slf", "str"]
    combination = json.loads((out / "combination_rule.json").read_text())
    assert len(combination) == 17
    assert not [c for c in combination if c["disagreements"]]
    gaps = json.loads((out / "perturbation_field_gaps.json").read_text())
    assert [gap["field"] for gap in gaps] == [
        "barcode",
        "insertion_position",
        "insertion_strand",
        "transposon",
    ]


def test_the_build_emits_one_reference_per_condition(
    built: m.EnvChemgenGirgis2009Dataset,
) -> None:
    references = [built[i]["reference"] for i in range(len(built))]
    screens = {
        reference["phenotype_reference"]["screen_id"] for reference in references
    }
    assert screens == {spec.screen_id for spec in m.DRUGS}
    assert len(screens) == 17
    assert {
        reference["phenotype_reference"]["environment_response"]
        for reference in references
    } == {0.0}
    assert {reference["genome_reference"]["strain"] for reference in references} == {
        m.LIBRARY_PARENT
    }


def test_the_build_refuses_a_genome_of_another_strain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mg1655: EcoliK12MG1655Genome,
    presence_only_pins: list[Mapping[str, str]],
) -> None:
    monkeypatch.setattr(m, "MIN_RESOLVED_FRACTION", 0.8)
    monkeypatch.setattr(m, "reference_genome", lambda *a, **k: REFERENCE)
    root = tmp_path / "other"
    _write_raw(root / "raw")

    class Other:
        ASSEMBLY_SET = "ecoli_K12_BW25113_ASM75055v1"

    with pytest.raises(ValueError, match="needs the ecoli_K12_MG1655_ASM584v2 genome"):
        m.EnvChemgenGirgis2009Dataset(
            root=str(root),
            ecoli_genome=Other(),  # type: ignore[arg-type]
        )


def test_the_build_refuses_a_cell_the_combination_rule_misses(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch, mg1655: EcoliK12MG1655Genome
) -> None:
    monkeypatch.setattr(m, "MIN_RESOLVED_FRACTION", 0.8)
    rows = _combined_rows()
    rows[2][2] = 9.0
    _write_sheet(
        synthetic / "raw" / m.DATASET_S5,
        m.SHEETS[m.DATASET_S5],
        ["UNIQID", "NAME", *(spec.code for spec in m.DRUGS)],
        rows,
    )
    with pytest.raises(m.CombinationRuleError, match="not reproduced"):
        m.EnvChemgenGirgis2009Dataset(root=str(synthetic), ecoli_genome=mg1655)


def test_the_dataset_is_registered_under_its_class_name() -> None:
    assert dataset_registry["EnvChemgenGirgis2009Dataset"] is (
        m.EnvChemgenGirgis2009Dataset
    )
    assert m.EnvChemgenGirgis2009Dataset.REFERENCE_STRAIN == m.REFERENCE_STRAIN_NAME


def test_the_dataset_declares_its_schema_classes() -> None:
    dataset = m.EnvChemgenGirgis2009Dataset.__new__(m.EnvChemgenGirgis2009Dataset)
    assert dataset.experiment_class is BacterialEnvironmentResponseExperiment
    assert dataset.reference_class is (BacterialEnvironmentResponseExperimentReference)
    assert dataset.raw_file_names == [
        m.DATASET_S1,
        m.DATASET_S3,
        m.DATASET_S4,
        m.DATASET_S5,
    ]
    assert dataset.preprocess_raw(pd.DataFrame({"a": [1]})).equals(
        pd.DataFrame({"a": [1]})
    )
    with pytest.raises(NotImplementedError):
        dataset.create_experiment()


# --------------------------------------------------------------------------- #
# The raw mirror
# --------------------------------------------------------------------------- #
def test_every_raw_file_records_a_re_runnable_pmc_retrieval() -> None:
    assert len(m.RAW_FILES) == 4
    for raw in m.RAW_FILES:
        assert raw.retrieval.method is RetrievalMethod.pmc_cloud
        assert raw.retrieval.retriever == (
            "torchcell.literature.retrieve.pmc_cloud_object"
        )
        assert raw.retrieval.params["key"].startswith(f"{m.PMCID}.1/pone.0005629.s0")
        assert raw.retrieval.sha256 == raw.sha256
        assert raw.mirror_relpath == f"data/{raw.name}"


def test_deposit_raw_mirror_writes_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = {}
    for raw in m.RAW_FILES:
        path = tmp_path / "src" / raw.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw.name.encode())
        sources[raw.name] = path
    monkeypatch.setattr(m, "_sha256", lambda path: m.DATA_SHA256[Path(path).name])
    root = m.deposit_raw_mirror(sources=sources, data_root=str(tmp_path / "root"))
    manifest = m.load_manifest(str(tmp_path / "root"))
    assert manifest.citation_key == m.CITATION_KEY
    assert manifest.doi == m.PAPER_DOI
    assert [record.path for record in manifest.files] == [
        raw.mirror_relpath for raw in m.RAW_FILES
    ]
    assert m.manifest_sha256(manifest, "data/si24.xls") == m.DATA_SHA256[m.DATASET_S5]
    assert list(manifest.si_expected) == list(m.NOT_MIRRORED)
    assert (root / "data" / m.DATASET_S5).exists()


def test_deposit_raw_mirror_refuses_a_missing_source(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match=m.DATASET_S1):
        m.deposit_raw_mirror(sources={}, data_root=str(tmp_path))


def test_deposit_raw_mirror_refuses_a_hash_mismatch(tmp_path: Path) -> None:
    sources: dict[str, str | Path] = {}
    for raw in m.RAW_FILES:
        path = tmp_path / "src" / raw.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not the released bytes")
        sources[raw.name] = path
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        m.deposit_raw_mirror(sources=sources, data_root=str(tmp_path / "root"))


def test_manifest_sha256_refuses_an_unknown_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from torchcell.literature.manifest import Manifest

    manifest = Manifest(citation_key=m.CITATION_KEY, doi=m.PAPER_DOI, title="t")
    with pytest.raises(KeyError, match="not in the raw-mirror manifest"):
        m.manifest_sha256(manifest, "data/si24.xls")
    del tmp_path, monkeypatch


def test_retrieve_raw_files_runs_the_recorded_retriever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served: list[str] = []

    def fake(record: RetrievalRecord) -> bytes:
        served.append(record.params["key"])
        return b"bytes"

    monkeypatch.setattr(m, "run_retriever", fake)
    monkeypatch.setattr(m, "write_verified", lambda data, path, sha, url: None)
    out = m.retrieve_raw_files(tmp_path, names=[m.DATASET_S5])
    assert list(out) == [m.DATASET_S5]
    assert served == [f"{m.PMCID}.1/pone.0005629.s024.xls"]


def test_the_mirror_directories_hang_off_data_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATA_ROOT", "/tmp/root")
    assert m.raw_mirror_dir() == Path(f"/tmp/root/torchcell-raw/{m.CITATION_KEY}")
    assert m.library_dir() == Path(f"/tmp/root/torchcell-library/{m.CITATION_KEY}")


# --------------------------------------------------------------------------- #
# Sourced values
# --------------------------------------------------------------------------- #
def test_every_sourced_value_is_auditable() -> None:
    assert m.SOURCED_VALUES
    for key, value in m.SOURCED_VALUES.items():
        assert value.provenance.citation_key == m.CITATION_KEY, key
        assert value.provenance.source_uri in (m.PAPER_MD, m.TEXT_S1_MD), key
        assert value.provenance.sha256 in (m.PAPER_MD_SHA256, m.TEXT_S1_MD_SHA256), key
        assert value.quote.strip(), key


def test_the_derived_constants_come_from_the_sourced_values() -> None:
    assert m.TEMPERATURE_C == m.SOURCED_VALUES["temperature_c"].value
    assert m.AEROBICITY == m.SOURCED_VALUES["aerobicity"].value
    assert m.LIBRARY_PARENT == m.SOURCED_VALUES["library_parent"].value
    assert m.SIGNIFICANCE_THRESHOLDS == (
        m.SOURCED_VALUES["significance_thresholds"].value
    )
    assert m.AVERAGED_CODES == (m.SOURCED_VALUES["technical_replicates_averaged"].value)


def test_the_publication_is_the_paper() -> None:
    assert m.PUBLICATION.pubmed_id == "19462005"
    assert m.PUBLICATION.doi == "10.1371/journal.pone.0005629"
    assert m.PMCID == "PMC2680486"


# --------------------------------------------------------------------------- #
# Data-gated: the real mirror and the built dev store
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """The real ``DATA_ROOT`` the ``--data`` bucket reads (exported by the caller)."""
    return os.environ["DATA_ROOT"]


def _mirror() -> Path:
    root = Path(_data_root()) / m.RAW_DIR_REL
    if not (root / "manifest.json").exists():
        pytest.skip("the Girgis 2009 raw mirror is not deposited")
    return root


def _built_root() -> str:
    root = osp.join(_data_root(), m.DATASET_ROOT_REL)
    if not osp.isdir(osp.join(root, "processed", "lmdb")):
        pytest.skip("the Girgis 2009 dev-tree LMDB is not built")
    return root


@pytest.mark.data
def test_the_real_mirror_matches_every_pin() -> None:
    root = _mirror()
    manifest = m.load_manifest(str(Path(_data_root())))
    for raw in m.RAW_FILES:
        assert m.manifest_sha256(manifest, raw.mirror_relpath) == raw.sha256
        assert m._sha256(root / raw.mirror_relpath) == raw.sha256


@pytest.mark.data
def test_every_sourced_value_is_backed_by_its_verbatim_quote() -> None:
    library = Path(_data_root()) / "torchcell-library"
    if not (library / m.CITATION_KEY / m.PAPER_MD).exists():
        pytest.skip("the Girgis 2009 literature mirror is not available")
    for key, value in m.SOURCED_VALUES.items():
        result = audit_sourced_value(value, library)
        assert result.passed, f"{key}: {result.message}"
    for spec in m.DRUGS:
        result = audit_sourced_value(spec.dose_sourced(), library)
        assert result.passed, f"{spec.code}: {result.message}"


@pytest.mark.data
def test_the_released_dataset_s5_has_the_measured_shape() -> None:
    root = _mirror()
    frame = m.read_sheet(root / f"data/{m.DATASET_S5}", m.SHEETS[m.DATASET_S5])
    assert frame.shape == (3976, 19)
    assert frame["UNIQID"].tolist()[:3] == ["b0002", "b0003", "b0004"]
    no_data = sum(
        1
        for spec in m.DRUGS
        for cell in frame[spec.code].tolist()
        if m._cell(cell) is None
    )
    assert no_data == 1305


@pytest.mark.data
def test_hand_checked_cells_of_the_released_sheet() -> None:
    root = _mirror()
    frame = m.read_sheet(root / f"data/{m.DATASET_S5}", m.SHEETS[m.DATASET_S5])
    row = frame.set_index("UNIQID")
    assert row.at["b0002", "NAME"] == (
        "thrA aspartate kinase / homoserine dehydrogenase"
    )
    assert m._cell(row.at["b0002", "tob"]) == pytest.approx(1.075013, abs=1e-6)
    assert m._cell(row.at["b0003", "str"]) == pytest.approx(-1.441714, abs=1e-6)
    assert m._cell(row.at["b0002", "gen"]) == 0.0


@pytest.mark.data
def test_the_built_store_holds_the_measured_record_count_and_sign_split() -> None:
    log = json.loads(
        Path(_built_root(), "preprocess", "dropped_records.json").read_text()
    )
    assert log["source_loci"] == 3976
    assert log["kept_loci"] == 3821
    assert log["source_records"] == 67592
    assert log["kept_records"] == m.EXPECTED_RECORDS == 63766
    rules = {rule["rule"]: rule["n_records"] for rule in log["rules"]}
    assert rules == {
        "b_number_is_not_a_locus_tag_of_the_pinned_annotation": 2635,
        "no_combined_score_released": 1191,
    }
    assert sum(drug["n_positive"] for drug in log["drugs"]) == 9600
    assert sum(drug["n_negative"] for drug in log["drugs"]) == 11523
    assert sum(drug["n_zero"] for drug in log["drugs"]) == 42643


@pytest.mark.data
def test_the_built_identifier_ledger_records_the_measured_routes() -> None:
    ledger = json.loads(
        Path(_built_root(), "preprocess", "identifier_reconciliation.json").read_text()
    )
    assert ledger["n_locus_tags"] == 3821
    assert len(ledger["not_a_locus_tag"]) == 155
    histogram = ledger["reconciliation"]["status_histogram"]
    assert histogram["current"] == 3758
    assert histogram["renamed"] == 39
    assert histogram["non_gene_feature"] == 134
    assert histogram["retired"] == 45
    assert histogram["ambiguous"] == 0
    assert ledger["reconciliation"]["layer_histogram"]["locus tag"] == 3821
    assert ledger["reconciliation"]["layer_histogram"]["gene synonym"] == 110
    assert ledger["reconciliation"]["outside_namespace"] == []


@pytest.mark.data
def test_the_built_combination_check_reproduces_the_release() -> None:
    checks = json.loads(
        Path(_built_root(), "preprocess", "combination_rule.json").read_text()
    )
    assert len(checks) == 17
    assert not [check for check in checks if check["disagreements"]]
    assert sum(check["n_exact"] for check in checks) == 66240
    assert sum(check["n_rounded_to_zero"] for check in checks) == 47
    assert sum(check["n_no_data"] for check in checks) == 1305


@pytest.mark.data
def test_the_built_replicate_structure_names_both_table_1_disagreements() -> None:
    replicates = json.loads(
        Path(_built_root(), "preprocess", "replicate_structure.json").read_text()
    )
    per_drug = {drug["drug"]: drug for drug in replicates["per_drug"]}
    assert per_drug["str"]["repetitions"] == 2
    assert per_drug["str"]["min_abs"] == pytest.approx(2.1505, abs=1e-3)
    assert per_drug["slf"]["repetitions"] == 3
    assert per_drug["slf"]["n_below_two_repetition_cut"] == 8
    assert len(replicates["table1_disagreements"]) == 2


@pytest.mark.data
def test_the_built_store_passes_l0_to_l4() -> None:
    report = json.loads(
        Path(_built_root(), "preprocess", "verification_report.json").read_text()
    )
    failed = [r["name"] for r in report["results"] if not r["passed"]]
    assert not failed, failed


@pytest.mark.data
def test_a_tampered_raw_file_is_refused(tmp_path: Path) -> None:
    root = _mirror()
    raw = tmp_path / "raw"
    raw.mkdir()
    for spec in m.RAW_FILES:
        (raw / spec.name).write_bytes((root / spec.mirror_relpath).read_bytes())
    (raw / m.DATASET_S5).write_bytes(b"tampered")
    from torchcell.data import verify_raw_files

    with pytest.raises(RawSha256MismatchError):
        verify_raw_files(str(raw), m.DATA_SHA256)


@pytest.mark.data
def test_a_manifest_pin_that_drifts_is_refused() -> None:
    from torchcell.data import check_manifest_pin

    with pytest.raises(ManifestPinMismatchError):
        check_manifest_pin(
            f"data/{m.DATASET_S5}", "0" * 64, m.DATA_SHA256[m.DATASET_S5]
        )


# --------------------------------------------------------------------------- #
# download() and the CLI
# --------------------------------------------------------------------------- #
@pytest.fixture
def tmp_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A ``DATA_ROOT`` whose raw mirror holds the four synthetic sheets, pinned to
    their real sha256 so ``download`` and the manifest checks run unmodified.
    """
    data_root = tmp_path / "root"
    sources: dict[str, str | Path] = {}
    staged = tmp_path / "staged"
    _write_raw(staged)
    for raw in m.RAW_FILES:
        sources[raw.name] = staged / raw.name
    monkeypatch.setattr(m, "_sha256", lambda path: m.DATA_SHA256[Path(path).name])
    m.deposit_raw_mirror(sources=sources, data_root=str(data_root))
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    return data_root


class _Downloader(m.EnvChemgenGirgis2009Dataset):
    """``download()`` alone, with a raw directory the test chooses.

    ``raw_dir`` is a read-only property on the base class, so the only way to exercise
    the mirror link step without a full build is to override it.
    """

    def __init__(self, raw: Path) -> None:
        self._raw = raw

    @property
    def raw_dir(self) -> str:
        return str(self._raw)


def test_download_links_every_mirror_file_into_raw(
    tmp_mirror: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        m,
        "link_verified",
        lambda src, dest, sha: Path(dest).write_bytes(Path(src).read_bytes()),
    )
    dataset = _Downloader(tmp_path / "linked")
    dataset.download()
    assert sorted(p.name for p in Path(dataset.raw_dir).iterdir()) == sorted(
        m.DATA_SHA256
    )


def test_download_refuses_a_file_missing_from_the_mirror(
    tmp_mirror: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "link_verified", lambda src, dest, sha: None)
    (tmp_mirror / m.RAW_DIR_REL / "data" / m.DATASET_S5).unlink()
    dataset = _Downloader(tmp_path / "linked")
    with pytest.raises(RuntimeError, match="required raw artifact missing"):
        dataset.download()


def test_download_refuses_a_manifest_pin_that_drifts(
    tmp_mirror: Path, tmp_path: Path
) -> None:
    path = tmp_mirror / m.RAW_DIR_REL / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    dataset = _Downloader(tmp_path / "linked")
    with pytest.raises(ManifestPinMismatchError):
        dataset.download()


def test_cli_deposits_from_the_library_or_a_rerun_retrieval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root = tmp_path / "root"
    si = data_root / m.LIBRARY_DIR_REL / "si"
    si.mkdir(parents=True)
    _write_raw(si)
    monkeypatch.setattr(m, "_sha256", lambda path: m.DATA_SHA256[Path(path).name])
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    assert m.main(["deposit"]) == 0
    assert capsys.readouterr().out.strip() == str(data_root / m.RAW_DIR_REL)

    retrieved: list[str] = []

    def fake_retrieve(dest: str) -> dict[str, Path]:
        retrieved.append(dest)
        return {raw.name: si / raw.name for raw in m.RAW_FILES}

    monkeypatch.setattr(m, "retrieve_raw_files", fake_retrieve)
    assert m.main(["deposit", "--retrieve-into", str(tmp_path / "fetched")]) == 0
    assert retrieved == [str(tmp_path / "fetched")]


def test_cli_build_and_verify(
    built: m.EnvChemgenGirgis2009Dataset,
    mg1655: EcoliK12MG1655Genome,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``build`` re-loads the store the ``built`` fixture already wrote; ``verify``
    runs the environment-response gate over it with the synthetic genome.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert m.main(["build"]) == 0
    assert capsys.readouterr().out.strip().endswith(f"len = {SYNTHETIC_RECORDS}")

    monkeypatch.setattr(m, "SOURCED_VALUES", {})
    monkeypatch.setattr(m, "DRUGS", ())
    monkeypatch.setattr(m, "EXPECTED_RECORDS", SYNTHETIC_RECORDS)
    monkeypatch.setattr(m, "bacterial_genome", lambda *a, **k: mg1655)
    assert m.main(["verify"]) == 0
    out = capsys.readouterr().out
    assert "ecoli_env_chemgen_girgis2009: PASS" in out


def test_verify_build_fails_on_a_wrong_expected_count(
    built: m.EnvChemgenGirgis2009Dataset,
    mg1655: EcoliK12MG1655Genome,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m, "SOURCED_VALUES", {})
    monkeypatch.setattr(m, "DRUGS", ())
    report = m.verify_build(
        built.root,
        genome=mg1655,
        data_root=str(Path(built.root).parent),
        expected_count=1,
    )
    assert not report.passed
    assert [r.name for r in report.results if not r.passed] == ["count"]
