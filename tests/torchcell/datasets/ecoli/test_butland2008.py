# tests/torchcell/datasets/ecoli/test_butland2008.py
# [[tests.torchcell.datasets.ecoli.test_butland2008]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_butland2008.py
"""The Butland 2008 unfiltered eSGA matrix loader (``torchcell.datasets.ecoli.butland2008``).

Synthetic tests (run everywhere) build Supplementary Tables 1-4 in ``tmp_path`` over the
synthetic MG1655 assembly of ``tests/torchcell/sequence/genome/_bacterial_fixtures.py``,
served through a stubbed ``resolve`` with the network refused, with ``b0010`` added as a
``/gene_synonym`` of ``b0003`` so the annotation-remap case exists. ``verify_raw_files``
is replaced with a presence check (synthetic bytes cannot carry the real pins; the pins
are asserted by the refusal test and the data-gated tests), the release-count constants
are monkeypatched to the synthetic release's own numbers, and ``read_served_babu`` is
stubbed so the partition runs without opening the real Babu store.

The synthetic release is a 2 x 9 matrix, and every cell exercises one outcome:

    recipient row                       query b0005   query b0007
    b0001 Isolate 1  non-essential      kept -5.0     kept +4.0
    b0001 Isolate 2  non-essential      kept +2.5     dropped: all-zero colonies, S 0.0
    b0003 Isolate 1  non-essential      kept -3.0     kept +1.5
    b0004 Isolate 1  non-essential      kept -1.0     kept  0.0 (a released zero)
    b0010 Isolate 1  non-essential      dropped: b0010 is a synonym of b0003 (#753)
    b0099 Isolate 1  non-essential      dropped: on no locus of the annotation
    b0007 Isolate 1  non-essential      kept -2.0     dropped: self pair
    b0006 SPA-tag    SPA-tag essential  kept -7.0     dropped: served
    b0002 Isolate 1  non-essential      dropped: served  kept +3.5

Ten records survive eight drops; the ``b0006`` row is the hypomorph case, stored on
``BacterialMarkedAllelePerturbation`` since issue #792. Data-gated tests (``--data``)
read the real raw mirror
and the built dev-tree LMDB under ``$DATA_ROOT`` (they never build it): the manifest
pins, the provenance audit of every sourced value including the seven quoted from
workbook cells, the measured release counts, hand-checked cells read off ``si5.xls``,
and the recorded ledgers.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd
import pytest

import torchcell.datasets.ecoli.butland2008 as m
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    MG1655_GAF,
    MG1655_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.data import ManifestPinMismatchError, check_manifest_pin
from torchcell.datamodels.media import MEDIA_LIBRARY
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    BacterialDeletionPerturbation,
    BacterialGeneInteractionExperiment,
    BacterialGeneInteractionExperimentReference,
    BacterialMarkedAllelePerturbation,
    MediaComponentRole,
    SampleUnit,
)
from torchcell.datasets.dataset_registry import dataset_registry
from torchcell.literature.manifest import RetrievalMethod
from torchcell.sequence.genome.ecoli.k12 import MG1655_ASSEMBLY, EcoliK12MG1655Genome
from torchcell.verification.sourced import ProvenanceGapReason, audit_sourced_value

REFERENCE = AssemblyReferenceGenome(
    species="Escherichia coli",
    strain="Hfr Cavalli x Keio (K-12 BW25113) eSGA conjugant",
    assembly_set="ecoli_K12_MG1655_ASM584v2",
    assembly_accession="GCA_000005845.2",
    background=m.chassis_background(),
)

#: ``(gene, b-number)`` of the synthetic Supplementary Table 2 query roster.
SYNTHETIC_QUERIES: tuple[tuple[str, str], ...] = (("proB", "b0005"), ("insZ", "b0007"))
#: ``(label, gene, b-number, strain version)`` of the synthetic recipient array, in the
#: order the matrix carries them.
SYNTHETIC_ARRAY: tuple[tuple[str, str, str, str], ...] = (
    (m.LABEL_NON_ESSENTIAL, "thrL", "b0001", "Isolate 1"),
    (m.LABEL_NON_ESSENTIAL, "thrL", "b0001", "Isolate 2"),
    (m.LABEL_NON_ESSENTIAL, "thrW", "b0003", "Isolate 1"),
    (m.LABEL_NON_ESSENTIAL, "yaaP", "b0004", "Isolate 1"),
    (m.LABEL_NON_ESSENTIAL, "newG", "b0010", "Isolate 1"),
    (m.LABEL_NON_ESSENTIAL, "ghostG", "b0099", "Isolate 1"),
    (m.LABEL_NON_ESSENTIAL, "insZ", "b0007", "Isolate 1"),
    (m.LABEL_SPA_TAG, "proC", "b0006", m.LABEL_SPA_TAG),
    (m.LABEL_NON_ESSENTIAL, "yaaX", "b0002", "Isolate 1"),
)
#: The S score of each cell, row-major over ``SYNTHETIC_ARRAY`` x ``(b0005, b0007)``.
SYNTHETIC_SCORES: tuple[tuple[float, float], ...] = (
    (-5.0, 4.0),
    (2.5, 0.0),
    (-3.0, 1.5),
    (-1.0, 0.0),
    (3.3, -3.4),
    (3.5, -3.6),
    (-2.0, 1.1),
    (-7.0, 1.2),
    (-8.0, 3.5),
)
#: The raw-colony cell of each (row, query) pair, in the SAME order as the scores. The
#: one all-zero cell is the contradiction case; the 8-colony cell exercises a replicate
#: count that is not the documented four.
_FOUR = "1:(100,200), 2:(300,400)"
_EIGHT = "1:(10,20,30,40), 2:(50,60,70,80)"
_NONE = "1:(0,0), 2:(0,0)"
SYNTHETIC_RAW: tuple[tuple[str, str], ...] = (
    (_FOUR, _EIGHT),
    (_FOUR, _NONE),
    (_FOUR, _FOUR),
    (_FOUR, _FOUR),
    (_FOUR, _FOUR),
    (_FOUR, _FOUR),
    (_FOUR, _FOUR),
    (_FOUR, _FOUR),
    (_FOUR, _FOUR),
)
#: ``(query tag, recipient tag, strain version, score)`` of the nine kept cells.
KEPT_CELLS: tuple[tuple[str, str, str, float], ...] = (
    ("b0005", "b0001", "Isolate 1", -5.0),
    ("b0007", "b0001", "Isolate 1", 4.0),
    ("b0005", "b0001", "Isolate 2", 2.5),
    ("b0005", "b0003", "Isolate 1", -3.0),
    ("b0007", "b0003", "Isolate 1", 1.5),
    ("b0005", "b0004", "Isolate 1", -1.0),
    ("b0007", "b0004", "Isolate 1", 0.0),
    ("b0005", "b0007", "Isolate 1", -2.0),
    ("b0005", "b0006", m.LABEL_SPA_TAG, -7.0),
    ("b0007", "b0002", "Isolate 1", 3.5),
)
#: The one kept cell whose recipient row is SPA-tag essential, so its recipient leaf is
#: a marked allele rather than a deletion and its "version" is the label.
KEPT_HYPOMORPH_CELL: tuple[str, str, str, float] = (
    "b0005",
    "b0006",
    m.LABEL_SPA_TAG,
    -7.0,
)
#: The stubbed served Babu store, one pair per group of the reverse proof: a storable
#: cell on a Keio-isolate row, a storable cell on a SPA-tag essential row (the group PR
#: #837 gave Babu and this loader now stores itself), a pair this release does not name
#: at all, and a "This Study" pair the reverse check must ignore. The first two are both
#: removed by the served rule.
SERVED_STUB: dict[tuple[str, str], str] = {
    ("b0005", "b0002"): m.BABU_SCREEN_TAG,
    ("b0007", "b0006"): m.BABU_SCREEN_TAG,
    ("b0005", "b4486"): m.BABU_SCREEN_TAG,
    ("b0003", "b0001"): "This Study",
}
#: ``(query gene, query tag, recipient gene, recipient tag, version, essentiality,
#: association, S, |Z|, log2(Q/R))`` of the synthetic Supplementary Table 3.
SYNTHETIC_HIGH_CONFIDENCE: tuple[tuple[Any, ...], ...] = (
    (
        "proB",
        "b0005",
        "thrL",
        "b0001",
        "Isolate 1",
        m.S3_NON_ESSENTIAL,
        "STRING",
        -5.0,
        -6.1,
        -1.2,
    ),
    (
        "insZ",
        "b0007",
        "thrW",
        "b0003",
        "Isolate 1",
        m.S3_NON_ESSENTIAL,
        "STRING",
        1.5,
        4.2,
        0.4,
    ),
    (
        "proB",
        "b0005",
        "proC",
        "b0006",
        m.LABEL_SPA_TAG,
        m.LABEL_SPA_TAG,
        "STRING",
        -7.0,
        -9.0,
        -2.0,
    ),
)


# --------------------------------------------------------------------------- #
# Writing the synthetic release
# --------------------------------------------------------------------------- #
def _matrix_sheet(
    sheet: Any,
    *,
    title: tuple[str, str],
    banner: str,
    queries: Sequence[tuple[str, str]],
    cells: Sequence[Sequence[Any]],
) -> None:
    """One Table 4 sheet in the layout the loader reads."""
    pad = [None] * m.FIRST_VALUE_COL
    sheet.append([title[0]])
    sheet.append([title[1]])
    sheet.append([banner])
    sheet.append([*pad, *(tag for _, tag in queries)])
    sheet.append([*pad, *(gene for gene, _ in queries)])
    sheet.append(["Essentiala/Non-essentialb", "Gene name", "b-numberc", None])
    for (label, gene, tag, version), row in zip(SYNTHETIC_ARRAY, cells, strict=True):
        sheet.append([label, gene, tag, version, *row])


def write_table_s4(
    path: Path, *, scores: Sequence[Sequence[Any]] | None = None
) -> None:
    """The synthetic Supplementary Table 4: four sheets, two of them read.

    The raw sheet's query columns are written in the OPPOSITE order to the S sheet's, the
    way the release does, so the build's reordering is exercised. Exactly one S cell
    carries the footnote letter, which is what the loader asserts.
    """
    workbook = openpyxl.Workbook()
    # the released title runs over A1 and A2, and A1 carries a trailing space
    head, marker, tail = str(m.SOURCED_VALUES["unfiltered_matrix"].quote).partition(
        " of each mutant gene pair"
    )
    title = (f"{head} ", f"{marker.lstrip()}{tail}")
    banner = str(m.SOURCED_VALUES["spa_tag_recipients"].quote)
    chosen = SYNTHETIC_SCORES if scores is None else scores
    marked = [[f"{chosen[0][0]}f", chosen[0][1]], *[list(row) for row in chosen[1:]]]

    s_sheet = workbook.active
    assert s_sheet is not None
    s_sheet.title = m.SHEET_S
    _matrix_sheet(
        s_sheet, title=title, banner=banner, queries=SYNTHETIC_QUERIES, cells=marked
    )
    for line in (
        str(m.SOURCED_VALUES["spa_tag_definition"].quote),
        *str(m.SOURCED_VALUES["score_definition"].quote).split("\n"),
    ):
        s_sheet.append([line])

    raw_sheet = workbook.create_sheet(m.SHEET_RAW)
    _matrix_sheet(
        raw_sheet,
        title=title,
        banner=banner,
        queries=tuple(reversed(SYNTHETIC_QUERIES)),
        cells=[list(reversed(row)) for row in SYNTHETIC_RAW],
    )
    for line in str(m.SOURCED_VALUES["replicate_design"].quote).split("\n"):
        raw_sheet.append([line])

    workbook.create_sheet(m.SHEET_NORMALIZED)
    workbook.create_sheet(m.SHEET_Z)
    workbook._sheets = [  # noqa: SLF001  (the loader asserts the released sheet order)
        workbook[name] for name in m.MATRIX_SHEETS
    ]
    workbook.save(path)


def write_table_s1(path: Path) -> None:
    """The synthetic Supplementary Table 1 recipient roster, plus its footnote d."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = m.ROSTER_SHEET
    for _ in range(m.BODY_ROW):
        sheet.append(["a note"])
    for row in SYNTHETIC_ARRAY:
        sheet.append(list(row))
    sheet.append([str(m.SOURCED_VALUES["isolate_is_a_strain"].quote)])
    workbook.save(path)


def write_table_s2(path: Path) -> None:
    """The synthetic Supplementary Table 2 query roster."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = m.QUERY_SHEET
    for _ in range(3):
        sheet.append(["a note"])
    for gene, tag in SYNTHETIC_QUERIES:
        sheet.append([gene, tag, "-", 1, 2, 3])
    workbook.save(path)


def write_table_s3(path: Path, *, rows: Sequence[Sequence[Any]] | None = None) -> None:
    """The synthetic Supplementary Table 3, plus its own Collins 2006 footnote."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = m.HIGH_CONFIDENCE_SHEET
    sheet.append(["Supplementary Table 3."])
    sheet.append(["a note"])
    sheet.append(list(m._S3_COLUMNS))  # noqa: SLF001  (the release's own column order)
    for row in SYNTHETIC_HIGH_CONFIDENCE if rows is None else rows:
        sheet.append(list(row))
    for line in str(m.SOURCED_VALUES["score_definition_table_s3"].quote).split("\n"):
        sheet.append([line])
    workbook.save(path)


def write_raw(raw: Path, *, scores: Sequence[Sequence[Any]] | None = None) -> None:
    """All four consumed workbooks, under the names the loader links from the mirror."""
    raw.mkdir(parents=True, exist_ok=True)
    write_table_s1(raw / m.TABLE_S1)
    write_table_s2(raw / m.TABLE_S2)
    write_table_s3(raw / m.TABLE_S3)
    write_table_s4(raw / m.TABLE_S4, scores=scores)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def mg1655(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EcoliK12MG1655Genome:
    """The real MG1655 class over the synthetic assembly, b0010 a synonym of b0003."""
    loci = [
        locus.model_copy(update={"synonyms": (*locus.synonyms, "b0010")})
        if locus.tag == "b0003"
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
def synthetic_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every asserted release and partition count at the synthetic release."""
    monkeypatch.setattr(m, "N_SCREENS", len(SYNTHETIC_QUERIES))
    monkeypatch.setattr(
        m,
        "N_NON_ESSENTIAL_STRAINS",
        sum(1 for row in SYNTHETIC_ARRAY if row[0] == m.LABEL_NON_ESSENTIAL),
    )
    monkeypatch.setattr(
        m,
        "N_SPA_TAG_STRAINS",
        sum(1 for row in SYNTHETIC_ARRAY if row[0] == m.LABEL_SPA_TAG),
    )
    monkeypatch.setattr(m, "N_HIGH_CONFIDENCE_PAIRS", len(SYNTHETIC_HIGH_CONFIDENCE))
    monkeypatch.setattr(m, "SERVED_BUTLAND_RECORDS", 3)
    monkeypatch.setattr(m, "SERVED_BUTLAND_PAIRS", 3)
    monkeypatch.setattr(m, "SERVED_OVERLAP_PAIRS", 2)
    monkeypatch.setattr(m, "SERVED_OVERLAP_CELLS", 2)
    monkeypatch.setattr(m, "SERVED_OVERLAP_PAIRS_ON_A_KEIO_RECIPIENT", 1)
    monkeypatch.setattr(m, "SERVED_PAIRS_ON_A_SPA_TAG_RECIPIENT", 1)
    monkeypatch.setattr(m, "SERVED_PAIRS_NOT_IN_THIS_RELEASE", ("b0005 -> b4486",))


@pytest.fixture
def served_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the served Babu store so the partition runs without opening an LMDB."""
    monkeypatch.setattr(
        m, "read_served_babu", lambda root: (dict(SERVED_STUB), len(SERVED_STUB))
    )


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """A directory holding all four synthetic workbooks."""
    raw = tmp_path / "sheets"
    write_raw(raw)
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
    synthetic_counts: None,
    served_stub: None,
    presence_only_pins: list[Mapping[str, str]],
) -> Path:
    """A dataset root whose ``raw/`` holds the four synthetic workbooks."""
    root = tmp_path / m.DATASET_ROOT_REL
    write_raw(root / "raw")
    monkeypatch.setattr(m, "reference_genome", lambda *a, **k: REFERENCE)
    return root


@pytest.fixture
def built(
    synthetic: Path, mg1655: EcoliK12MG1655Genome
) -> Iterator[m.GeneInteractionButland2008Dataset]:
    dataset = m.GeneInteractionButland2008Dataset(
        root=str(synthetic), ecoli_genome=mg1655, served_root="stubbed"
    )
    yield dataset
    dataset.close_lmdb()


# --------------------------------------------------------------------------- #
# The pinned release
# --------------------------------------------------------------------------- #
def test_the_four_consumed_files_are_pinned_with_their_own_retrieval() -> None:
    assert [raw.name for raw in m.RAW_FILES] == [
        "si2.xls",
        "si3.xls",
        "si4.xls",
        "si5.xls",
    ]
    for raw in m.RAW_FILES:
        assert raw.retrieval.method is RetrievalMethod.springer_esm
        assert raw.retrieval.sha256 == raw.sha256 == m.DATA_SHA256[raw.name]
        assert raw.retrieval.source_url is not None
        assert raw.retrieval.source_url.startswith(
            "https://static-content.springer.com/esm/"
        )
        assert raw.mirror_relpath == f"data/{raw.name}"
    assert m.DATA_SHA256[m.TABLE_S4] == (
        "74a6ea3a0373fa6e1776b0becb4212f29b9876647b96327db5222dcd68a8822b"
    )


def test_the_two_colony_size_sheets_are_recorded_as_not_loaded() -> None:
    joined = " ".join(m.NOT_LOADED)
    assert "no colony-size phenotype class" in joined
    assert "'Z scores'" in joined
    assert "issue #793" in joined
    assert "PROPER SUBSET" in joined


def test_every_sourced_value_names_a_pinned_artifact() -> None:
    for name, value in m.SOURCED_VALUES.items():
        assert value.provenance.citation_key == m.CITATION_KEY, name
        assert value.quote.strip(), name
        assert value.provenance.page, name
        assert value.provenance.sha256 in {
            m.PAPER_MD_SHA256,
            m.METHODS_MD_SHA256,
            *m.DATA_SHA256.values(),
        }, name
    assert set(m.TEXT_QUOTED) | {e.name for e in m.WORKBOOK_QUOTES} == set(
        m.SOURCED_VALUES
    )


def test_the_score_definition_defers_to_collins_2006_in_both_tables() -> None:
    for name in ("score_definition", "score_definition_table_s3"):
        quote = str(m.SOURCED_VALUES[name].quote)
        assert "implemented for yeast SGA (Collins et al., 2006)" in quote
        assert "Negative S-scores correspond to aggravating interactions" in quote
    note = " ".join(str(m.SOURCED_VALUES["score_definition"].note).split())
    assert "Collins et al. 2006 (Genome Biol 7:R63)" in note
    assert "NOT in the literature mirror" in note
    assert "no other source is substituted for it" in note


def test_the_replicate_design_is_four_colonies_two_per_isolate() -> None:
    value = m.SOURCED_VALUES["replicate_design"].value
    assert value == {"colonies": 4, "colonies_per_isolate": 2, "replicate_screens": 2}
    assert "pinned twice leaving four replicate recipient colonies" in str(
        m.SOURCED_VALUES["replicate_design"].quote
    )
    assert "issue #793" in str(m.SOURCED_VALUES["replicate_design"].note)


def test_the_two_kanamycin_doses_disagree_and_neither_is_asserted() -> None:
    assert m.SOURCED_VALUES["kanamycin_dose_50"].value == 50.0
    assert m.SOURCED_VALUES["kanamycin_dose_25"].value == 25.0
    kanamycin = next(
        component
        for component in m.SELECTION_MEDIUM.components
        if component.compound.name == "kanamycin"
    )
    assert kanamycin.concentration is None
    assert "50 ug/ml" in str(kanamycin.note) and "25 ug/ml" in str(kanamycin.note)


# --------------------------------------------------------------------------- #
# Reading the released sheets
# --------------------------------------------------------------------------- #
def test_the_s_sheet_is_read_with_its_title_banner_and_query_header(
    raw_dir: Path, synthetic_counts: None
) -> None:
    matrix = m.read_s_scores(raw_dir / m.TABLE_S4)
    assert matrix.sheet == m.SHEET_S
    assert matrix.query_tags == tuple(tag for _, tag in SYNTHETIC_QUERIES)
    assert matrix.recipient_tags == tuple(row[2] for row in SYNTHETIC_ARRAY)
    assert matrix.versions == tuple(row[3] for row in SYNTHETIC_ARRAY)
    block = m.score_block(matrix)
    assert block[0][0] == -5.0
    assert block.shape == (len(SYNTHETIC_ARRAY), len(SYNTHETIC_QUERIES))


def test_the_s_sheet_refuses_a_recipient_census_the_release_does_not_state(
    raw_dir: Path, monkeypatch: pytest.MonkeyPatch, synthetic_counts: None
) -> None:
    monkeypatch.setattr(m, "N_SPA_TAG_STRAINS", 7)
    with pytest.raises(m.ReleaseContentError, match="recipient rows"):
        m.read_s_scores(raw_dir / m.TABLE_S4)


def test_the_s_sheet_refuses_a_query_count_the_paper_does_not_state(
    raw_dir: Path, monkeypatch: pytest.MonkeyPatch, synthetic_counts: None
) -> None:
    monkeypatch.setattr(m, "N_SCREENS", 39)
    with pytest.raises(m.ReleaseContentError, match="query columns"):
        m.read_s_scores(raw_dir / m.TABLE_S4)


def test_the_s_sheet_refuses_a_shifted_title(
    raw_dir: Path, synthetic_counts: None
) -> None:
    frame = pd.read_excel(raw_dir / m.TABLE_S4, sheet_name=m.SHEET_S, header=None)
    frame.iat[0, 0] = "Supplementary Table 4. Something else"
    with pd.ExcelWriter(raw_dir / "shifted.xlsx") as writer:
        for sheet in m.MATRIX_SHEETS:
            body = (
                frame
                if sheet == m.SHEET_S
                else pd.read_excel(raw_dir / m.TABLE_S4, sheet_name=sheet, header=None)
            )
            body.to_excel(writer, sheet_name=sheet, header=False, index=False)
    with pytest.raises(m.SheetFormatError, match="title"):
        m.read_s_scores(raw_dir / "shifted.xlsx")


def test_the_score_block_refuses_more_than_one_footnote_marked_cell(
    tmp_path: Path, synthetic_counts: None
) -> None:
    scores: list[list[Any]] = [list(row) for row in SYNTHETIC_SCORES]
    scores[1][0] = "2.5f"
    write_table_s4(tmp_path / m.TABLE_S4, scores=scores)
    with pytest.raises(m.SheetFormatError, match="non-numeric cells"):
        m.score_block(m.read_s_scores(tmp_path / m.TABLE_S4))


def test_the_raw_sheet_yields_each_cells_colony_and_screen_counts(
    raw_dir: Path,
) -> None:
    matrix = m.read_matrix_sheet(raw_dir / m.TABLE_S4, m.SHEET_RAW)
    colonies, zeros, screens = m.colony_counts(matrix)
    assert colonies.shape == (len(SYNTHETIC_ARRAY), len(SYNTHETIC_QUERIES))
    assert set(colonies.ravel().tolist()) == {4, 8}
    assert set(screens.ravel().tolist()) == {2}
    # the raw sheet's columns are reversed, so the 8-colony cell is in column 0 here
    assert colonies[0][0] == 8
    assert zeros[1][0] == 4 and colonies[1][0] == 4


def test_the_raw_sheet_refuses_a_cell_that_is_not_colony_groups(raw_dir: Path) -> None:
    matrix = m.read_matrix_sheet(raw_dir / m.TABLE_S4, m.SHEET_RAW)
    broken = matrix.model_copy(update={"values": matrix.values.copy()})
    broken.values[0][0] = "no colonies here"
    with pytest.raises(m.SheetFormatError, match="colony groups|not '<n>"):
        m.colony_counts(broken)


def test_the_query_roster_is_the_39_screens(
    raw_dir: Path, synthetic_counts: None
) -> None:
    assert m.read_query_roster(raw_dir / m.TABLE_S2) == {
        tag: gene for gene, tag in SYNTHETIC_QUERIES
    }


def test_the_query_roster_refuses_a_count_the_paper_does_not_state(
    raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "N_SCREENS", 39)
    with pytest.raises(m.ReleaseContentError, match="query strains"):
        m.read_query_roster(raw_dir / m.TABLE_S2)


def test_the_array_roster_keys_on_b_number_and_strain_version(
    raw_dir: Path, synthetic_counts: None
) -> None:
    roster = m.read_array_roster(raw_dir / m.TABLE_S1)
    assert roster[("b0001", "Isolate 2")] == m.LABEL_NON_ESSENTIAL
    assert roster[("b0006", m.LABEL_SPA_TAG)] == m.LABEL_SPA_TAG
    assert len(roster) == len(SYNTHETIC_ARRAY)


def test_the_array_roster_refuses_a_census_the_release_does_not_state(
    raw_dir: Path, monkeypatch: pytest.MonkeyPatch, synthetic_counts: None
) -> None:
    monkeypatch.setattr(m, "N_NON_ESSENTIAL_STRAINS", 7924)
    with pytest.raises(m.ReleaseContentError, match="roster rows"):
        m.read_array_roster(raw_dir / m.TABLE_S1)


def test_the_high_confidence_table_is_read_and_counted(
    raw_dir: Path, synthetic_counts: None
) -> None:
    table = m.read_high_confidence(raw_dir / m.TABLE_S3)
    assert len(table) == len(SYNTHETIC_HIGH_CONFIDENCE)
    assert list(table.columns) == list(m._S3_COLUMNS)  # noqa: SLF001
    assert table["s_score"].tolist() == [-5.0, 1.5, -7.0]


def test_the_high_confidence_table_refuses_a_count_the_paper_does_not_state(
    raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "N_HIGH_CONFIDENCE_PAIRS", 1288)
    with pytest.raises(m.ReleaseContentError, match="ordered pairs"):
        m.read_high_confidence(raw_dir / m.TABLE_S3)


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #
def _leaves(
    genotype: Any,
) -> list[BacterialDeletionPerturbation | BacterialMarkedAllelePerturbation]:
    """The genotype's two bacterial gene-perturbation leaves, typed."""
    leaves = [
        leaf
        for leaf in genotype.perturbations
        if isinstance(
            leaf, BacterialDeletionPerturbation | BacterialMarkedAllelePerturbation
        )
    ]
    assert len(leaves) == 2
    return leaves


def _recipient(genotype: Any) -> str:
    """The released row token of the kan-marked recipient leaf.

    A Keio isolate carries it as the construction ``batch``; a SPA-tagged hypomorph
    carries no construction and the row's own label as its ``collection``.
    """
    leaf = next(leaf for leaf in _leaves(genotype) if leaf.cassette == "kan")
    if isinstance(leaf, BacterialMarkedAllelePerturbation):
        assert leaf.construction is None
        return str(leaf.collection)
    assert leaf.construction is not None
    assert leaf.construction.batch is not None
    return leaf.construction.batch


def test_the_genotype_is_two_leaves_distinguished_by_cassette_and_isolate() -> None:
    genotype = m.pair_genotype("b0005", "proB", "b0001", "thrL", "Isolate 2")
    leaves = _leaves(genotype)
    by_cassette = {str(leaf.cassette): leaf for leaf in leaves}
    assert sorted(by_cassette) == ["cat", "kan"]
    query, recipient = by_cassette["cat"], by_cassette["kan"]
    assert query.systematic_gene_name == "b0005"
    assert recipient.systematic_gene_name == "b0001"
    assert query.collection == m.QUERY_COLLECTION
    assert recipient.collection == m.RECIPIENT_COLLECTION
    assert query.construction is None
    assert recipient.construction is not None
    assert recipient.construction.batch == "Isolate 2"
    assert query.gene_namespace == recipient.gene_namespace == m.MG1655_NAMESPACE


def test_a_spa_tag_row_is_a_marked_allele_with_no_isolate() -> None:
    """#792: the hypomorph recipient is the marked-allele leaf, stated field by field.

    Every field is footnote a's own sentence; ``insertion_site`` is None because no
    artifact of THIS release names where the cassette went, and ``construction`` is None
    because the row's "Strain Versions" cell repeats the label instead of an isolate id.
    """
    genotype = m.pair_genotype(
        "b0005", "proB", "b0006", "proC", m.LABEL_SPA_TAG, recipient_is_hypomorph=True
    )
    query, recipient = genotype.perturbations
    assert isinstance(query, BacterialDeletionPerturbation)
    assert isinstance(recipient, BacterialMarkedAllelePerturbation)
    assert recipient.systematic_gene_name == "b0006"
    assert recipient.perturbed_gene_name == "proC"
    assert recipient.cassette == "kan"
    assert recipient.tag == "SPA"
    assert recipient.terminus == "C"
    assert recipient.allele_effect == "hypomorphic"
    assert recipient.collection == m.LABEL_SPA_TAG == "SPA-tag essential"
    assert recipient.insertion_site is None
    assert recipient.construction is None
    assert recipient.gene_namespace == m.MG1655_NAMESPACE
    assert recipient.perturbation_type == m.MARKED_ALLELE_TYPE


def test_a_spa_tag_row_that_carries_an_isolate_token_is_refused() -> None:
    """A versioned SPA row would mean the release had constructed those strains twice."""
    with pytest.raises(m.ReleaseContentError, match="names no isolate"):
        m.pair_genotype(
            "b0005", "proB", "b0006", "proC", "Isolate 1", recipient_is_hypomorph=True
        )


def test_the_hypomorph_leaf_matches_the_served_babu_leaf_but_for_one_field() -> None:
    """One physical strain set, two releases: only ``insertion_site`` differs.

    Babu 2014 states "a Kan-R marker was integrated into the 3'-UTR" and stores it; this
    release never names the site, so the field stays None here rather than borrowing a
    later paper's sentence. Every other field, the collection string included, agrees.
    """
    from torchcell.datasets.ecoli import babu2014 as babu

    ours = m.recipient_hypomorph_perturbation("b0006", "proC")
    theirs = babu.recipient_hypomorph_perturbation("b0006", "proC")
    assert theirs.insertion_site == "3'-UTR"
    assert ours.insertion_site is None
    assert ours.collection == theirs.collection == "SPA-tag essential"
    assert ours.model_dump(exclude={"insertion_site"}) == theirs.model_dump(
        exclude={"insertion_site"}
    )


def test_the_two_isolates_of_one_pair_are_two_distinct_genotypes() -> None:
    first = m.pair_genotype("b0005", "proB", "b0001", "thrL", "Isolate 1")
    second = m.pair_genotype("b0005", "proB", "b0001", "thrL", "Isolate 2")
    assert first != second
    batches = [_recipient(genotype) for genotype in (first, second)]
    assert batches == ["Isolate 1", "Isolate 2"]


def test_the_phenotype_is_a_signed_score_with_a_gapped_p_value() -> None:
    negative = m.phenotype(-5.0, 4)
    assert negative.gene_interaction == -5.0
    assert negative.gene_interaction_p_value is None
    assert negative.screen_id is None
    gap = negative.provenance_gaps[0]
    assert gap.field == "gene_interaction_p_value"
    assert gap.reason is ProvenanceGapReason.not_reported_by_primary
    assert m.phenotype(0.0, 4).gene_interaction == 0.0
    assert m.reference_phenotype().gene_interaction == 0.0


def test_the_phenotype_carries_the_cells_own_measured_colony_count() -> None:
    """#793: ``n_samples`` is counted off the raw sheet per cell, not the documented 4.

    The documented design is four colonies and it is the MODE rather than the rule, so a
    constant 4 would overstate the precision of every record released with more. The
    builder therefore takes the count and never defaults it.
    """
    assert m.phenotype(-5.0, 4).n_samples == 4
    assert m.phenotype(-5.0, 182).n_samples == 182
    assert m.phenotype(-5.0, 4).sample_unit is SampleUnit.colony
    design = m.SOURCED_VALUES["replicate_design"].value
    assert int(design["colonies"]) == 4
    assert int(design["colonies_per_isolate"]) * int(design["replicate_screens"]) == 4
    # the reference is 0 by construction, not a measured cell
    assert m.reference_phenotype().n_samples is None
    assert m.reference_phenotype().sample_unit is None


def test_the_chassis_background_pins_mg1655_and_names_both_parents() -> None:
    background = m.chassis_background()
    assert background.reference_strain == "MG1655"
    assert background.parents == ["Hfr Cavalli", "Keio collection (K-12 BW25113)"]
    assert background.alleles == []
    assert background.genotype_statement is None
    assert background.provenance_gaps[0].field == "genotype_statement"


def test_the_medium_is_lb_with_both_drugs_and_no_asserted_amount() -> None:
    medium = m.SELECTION_MEDIUM
    assert medium.state == "solid"
    assert medium.base_medium == "LB"
    assert medium.base_medium in MEDIA_LIBRARY
    assert all(component.concentration is None for component in medium.components)
    roles = {component.compound.name: component.role for component in medium.components}
    assert roles["kanamycin"] is MediaComponentRole.selection_agent
    assert roles["chloramphenicol"] is MediaComponentRole.selection_agent
    assert roles["agar"] is MediaComponentRole.gelling_agent


def test_the_environment_is_one_condition() -> None:
    environment = m.SCREEN_ENVIRONMENT
    assert environment.temperature is not None
    assert environment.temperature.value == 32.0
    assert environment.duration_hours == 24.0
    assert environment.aerobicity == "aerobic"
    assert environment.perturbations == []
    assert m.PUBLICATION.pubmed_id is None
    assert m.PUBLICATION.doi == m.PAPER_DOI


# --------------------------------------------------------------------------- #
# The build
# --------------------------------------------------------------------------- #
def test_the_dataset_is_registered_under_its_class_name() -> None:
    assert (
        dataset_registry["GeneInteractionButland2008Dataset"]
        is m.GeneInteractionButland2008Dataset
    )


def test_the_build_keeps_only_the_ten_typable_cells(
    built: m.GeneInteractionButland2008Dataset,
    presence_only_pins: list[Mapping[str, str]],
) -> None:
    assert len(built) == len(KEPT_CELLS)
    assert presence_only_pins[0] == m.DATA_SHA256
    stored: set[tuple[str, str, str, float]] = set()
    types: set[str] = set()
    for index in range(len(built)):
        experiment = built[index]["experiment"]
        by_cassette = {
            p["cassette"]: p for p in experiment["genotype"]["perturbations"]
        }
        recipient = by_cassette["kan"]
        construction = recipient["construction"]
        stored.add(
            (
                by_cassette["cat"]["systematic_gene_name"],
                recipient["systematic_gene_name"],
                construction["batch"]
                if construction is not None
                else recipient["collection"],
                float(experiment["phenotype"]["gene_interaction"]),
            )
        )
        types.add(str(recipient["perturbation_type"]))
    assert stored == set(KEPT_CELLS)
    assert KEPT_HYPOMORPH_CELL in stored
    assert types == {m.DELETION_TYPE, m.MARKED_ALLELE_TYPE}


def test_the_build_ledgers_account_for_every_released_cell(
    built: m.GeneInteractionButland2008Dataset,
) -> None:
    drops = json.loads(Path(built.preprocess_dir, "dropped_records.json").read_text())
    assert drops["source_records"] == len(SYNTHETIC_ARRAY) * len(SYNTHETIC_QUERIES)
    assert drops["kept_records"] == len(KEPT_CELLS)
    assert drops["dropped_records"] == drops["source_records"] - len(KEPT_CELLS)
    by_rule = {rule["rule"]: rule["n_records"] for rule in drops["rules"]}
    assert by_rule == {
        m.RULE_NOT_A_TAG: 2,
        m.RULE_REMAPPED: 2,
        m.RULE_SELF_PAIR: 1,
        m.RULE_CONTRADICTION: 1,
        m.RULE_SERVED: 2,
    }
    assert sum(by_rule.values()) == drops["dropped_records"]
    assert drops["kept_strain_versions"] == {
        "Isolate 1": 8,
        "Isolate 2": 1,
        m.LABEL_SPA_TAG: 1,
    }
    assert drops["kept_hypomorph_records"] == 1
    assert (drops["n_aggravating"], drops["n_alleviating"], drops["n_zero"]) == (
        5,
        4,
        1,
    )
    items = {rule["rule"]: rule["items"] for rule in drops["rules"]}
    assert items[m.RULE_SELF_PAIR] == ["b0007 -> b0007 (Isolate 1)"]
    assert items[m.RULE_SERVED] == [
        "b0005 -> b0002 (Isolate 1)",
        "b0007 -> b0006 (SPA-tag essential)",
    ]
    assert items[m.RULE_CONTRADICTION] == ["b0007 -> b0001 (Isolate 2)"]


def test_the_build_measures_the_replicate_design_rather_than_asserting_it(
    built: m.GeneInteractionButland2008Dataset,
) -> None:
    structure = json.loads(
        Path(built.preprocess_dir, "replicate_structure.json").read_text()
    )
    assert structure["sample_unit"] == "colony"
    assert structure["documented_n_samples"] == 4
    assert structure["measured_n_samples_counts"] == {"4": 9, "8": 1}
    assert structure["measured_replicate_screen_counts"] == {"2": 10}
    assert structure["modal_n_samples"] == 4
    assert structure["records_at_modal_n_samples"] == 9
    assert structure["sha256"] == m.DATA_SHA256[m.TABLE_S4]
    assert "#793" in structure["note"]


def test_every_built_record_stores_its_own_measured_colony_count(
    built: m.GeneInteractionButland2008Dataset,
) -> None:
    """The ledger's measured distribution and the records' ``n_samples`` agree exactly.

    The synthetic release holds 9 cells at 4 colonies and 1 at 8, so the two views must
    be the same histogram; a builder that defaulted to 4 would read 10 at 4 and 0 at 8.
    """
    structure = json.loads(
        Path(built.preprocess_dir, "replicate_structure.json").read_text()
    )
    counts: dict[int, int] = {}
    for index in range(len(built)):
        phenotype = built[index]["experiment"]["phenotype"]
        assert phenotype["sample_unit"] == SampleUnit.colony
        n = int(phenotype["n_samples"])
        counts[n] = counts.get(n, 0) + 1
    built.close_lmdb()
    assert {str(k): v for k, v in sorted(counts.items())} == structure[
        "measured_n_samples_counts"
    ]


def test_the_build_proves_table_3_is_a_subset_of_the_matrix(
    built: m.GeneInteractionButland2008Dataset,
) -> None:
    ledger = json.loads(
        Path(built.preprocess_dir, "high_confidence_pairs.json").read_text()
    )
    assert ledger["rows"] == ledger["located_in_matrix"] == ledger["identical_score"]
    assert ledger["rows"] == len(SYNTHETIC_HIGH_CONFIDENCE)
    assert ledger["non_essential_pairs"] == 2
    assert ledger["spa_tag_pairs"] == 1
    assert ledger["stored_rows"] == 3
    assert ledger["not_stored_rows"] == 0
    assert ledger["pairs"] == [
        "b0005 -> b0001 (Isolate 1)",
        "b0005 -> b0006 (SPA-tag essential)",
        "b0007 -> b0003 (Isolate 1)",
    ]


def test_the_build_proves_the_partition_against_the_served_store(
    built: m.GeneInteractionButland2008Dataset,
) -> None:
    partition = json.loads(
        Path(built.preprocess_dir, "served_partition.json").read_text()
    )
    assert partition["shared_pairs"] == 0
    assert partition["overlap_pairs_dropped"] == 2
    assert partition["overlap_cells_dropped"] == 2
    assert partition["served_butland_pairs"] == 3
    assert partition["overlap_pairs_on_a_keio_recipient"] == 1
    assert partition["served_pairs_on_a_spa_tag_recipient"] == 1
    assert partition["served_pairs_not_in_this_release"] == ["b0005 -> b4486"]
    assert partition["stored_records"] == len(KEPT_CELLS)
    assert partition["served_fraction_of_this_release"] == pytest.approx(3 / 18)


def test_the_build_writes_the_identifier_and_not_loaded_ledgers(
    built: m.GeneInteractionButland2008Dataset,
) -> None:
    reconciliation = json.loads(
        Path(built.preprocess_dir, "identifier_reconciliation.json").read_text()
    )
    assert reconciliation["unique_names"] == 9
    assert reconciliation["retired_kept"] == ["b0099"]
    assert reconciliation["kept_on_collision"] == ["b0003", "b0010"]
    assert json.loads(
        Path(built.preprocess_dir, "files_not_loaded.json").read_text()
    ) == list(m.NOT_LOADED)


def test_the_build_refuses_a_table_3_row_the_matrix_contradicts(
    synthetic: Path, mg1655: EcoliK12MG1655Genome
) -> None:
    rows = [list(row) for row in SYNTHETIC_HIGH_CONFIDENCE]
    rows[0][7] = -5.5
    write_table_s3(synthetic / "raw" / m.TABLE_S3, rows=rows)
    with pytest.raises(m.ReleaseContentError, match="the matrix cell reads"):
        m.GeneInteractionButland2008Dataset(
            root=str(synthetic), ecoli_genome=mg1655, served_root="stubbed"
        )


def test_the_build_refuses_a_query_the_annotation_does_not_carry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mg1655: EcoliK12MG1655Genome,
    synthetic_counts: None,
    served_stub: None,
    presence_only_pins: list[Mapping[str, str]],
) -> None:
    monkeypatch.setattr(m, "reference_genome", lambda *a, **k: REFERENCE)
    monkeypatch.setattr(
        m, "SYNTHETIC_QUERIES", None, raising=False
    )  # the module has no such name; the queries come from the sheets
    queries = (("proB", "b0005"), ("ghostQ", "b0099"))
    root = tmp_path / m.DATASET_ROOT_REL
    write_raw(root / "raw")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = m.QUERY_SHEET
    for _ in range(3):
        sheet.append(["a note"])
    for gene, tag in queries:
        sheet.append([gene, tag, "-", 1, 2, 3])
    workbook.save(root / "raw" / m.TABLE_S2)
    with pytest.raises(m.ReleaseContentError, match="query columns are not"):
        m.GeneInteractionButland2008Dataset(
            root=str(root), ecoli_genome=mg1655, served_root="stubbed"
        )


#: The three groups of the stubbed release, as the partition's reverse proof sees them:
#: every pair the synthetic matrix prints a cell for, the ones on its SPA-tag row, and
#: the one served pair rule 6 removes.
STUB_RELEASED_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (query, recipient)
    for _, _, recipient, _ in SYNTHETIC_ARRAY
    for _, query in SYNTHETIC_QUERIES
)
STUB_SPA_TAG_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (query, recipient)
    for label, _, recipient, _ in SYNTHETIC_ARRAY
    if label == m.LABEL_SPA_TAG
    for _, query in SYNTHETIC_QUERIES
)
STUB_OVERLAP_PAIRS: tuple[tuple[str, str], ...] = (
    ("b0005", "b0002"),
    ("b0007", "b0006"),
)


def _partition(
    *,
    stored_pairs: Sequence[tuple[str, str]] = (),
    released_pairs: Sequence[tuple[str, str]] = STUB_RELEASED_PAIRS,
    spa_tag_pairs: Sequence[tuple[str, str]] = STUB_SPA_TAG_PAIRS,
    overlap_pairs: Sequence[tuple[str, str]] = STUB_OVERLAP_PAIRS,
    stored_records: int = 0,
    overlap_cells: int = 2,
) -> m.ServedPartition:
    """Run the partition over the stubbed served store and the synthetic release."""
    return m.assert_served_partition(
        "stubbed",
        dict(SERVED_STUB),
        len(SERVED_STUB),
        stored_pairs,
        released_pairs,
        spa_tag_pairs,
        overlap_pairs,
        stored_records=stored_records,
        released_cells=18,
        overlap_cells=overlap_cells,
    )


@pytest.mark.usefixtures("synthetic_counts")
def test_the_partition_accounts_for_every_served_pair_of_this_screen() -> None:
    """The three groups of the reverse proof sum to the served record count."""
    partition = _partition()
    assert partition.served_butland_records == partition.served_butland_pairs == 3
    assert partition.overlap_pairs_dropped == 2
    assert partition.overlap_pairs_on_a_keio_recipient == 1
    assert partition.served_pairs_on_a_spa_tag_recipient == 1
    assert partition.served_pairs_not_in_this_release == ["b0005 -> b4486"]
    groups = (
        partition.overlap_pairs_on_a_keio_recipient
        + partition.served_pairs_on_a_spa_tag_recipient
        + len(partition.served_pairs_not_in_this_release)
    )
    assert groups == partition.served_butland_records
    assert (
        partition.overlap_pairs_dropped
        == partition.overlap_pairs_on_a_keio_recipient
        + partition.served_pairs_on_a_spa_tag_recipient
    )


def test_the_partition_refuses_a_stored_pair_the_served_store_holds() -> None:
    with pytest.raises(RuntimeError, match="already served by Babu 2014"):
        _partition(stored_pairs=[("b0005", "b0002")], stored_records=1)


def test_the_partition_refuses_a_served_butland_count_it_was_not_measured_against(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m, "SERVED_BUTLAND_RECORDS", 727)
    with pytest.raises(RuntimeError, match="records under screen_id"):
        _partition()


@pytest.mark.usefixtures("synthetic_counts")
def test_the_partition_refuses_a_served_pair_count_that_disagrees_with_the_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m, "SERVED_BUTLAND_PAIRS", 727)
    with pytest.raises(RuntimeError, match="oriented pairs under screen_id"):
        _partition()


@pytest.mark.usefixtures("synthetic_counts")
def test_the_partition_refuses_an_overlap_pair_the_screen_does_not_tag() -> None:
    """Rule 6 may only drop pairs the served store tags with THIS screen."""
    with pytest.raises(RuntimeError, match="does not tag them"):
        _partition(overlap_pairs=[("b0003", "b0001")])


@pytest.mark.usefixtures("synthetic_counts")
def test_the_partition_refuses_an_overlap_pair_count_it_was_not_measured_against(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m, "SERVED_OVERLAP_PAIRS", 1123)
    with pytest.raises(RuntimeError, match="storable cells of this release"):
        _partition()


@pytest.mark.usefixtures("synthetic_counts")
def test_the_partition_refuses_a_spa_tag_group_it_was_not_measured_against(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m, "SERVED_PAIRS_ON_A_SPA_TAG_RECIPIENT", 398)
    with pytest.raises(RuntimeError, match="SPA-tag essential recipient row"):
        _partition()


@pytest.mark.usefixtures("synthetic_counts")
def test_the_partition_refuses_a_keio_overlap_group_it_was_not_measured_against(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m, "SERVED_OVERLAP_PAIRS_ON_A_KEIO_RECIPIENT", 725)
    with pytest.raises(RuntimeError, match="Keio-isolate"):
        _partition()


@pytest.mark.usefixtures("synthetic_counts")
def test_the_partition_refuses_a_spa_tag_pair_the_served_rule_did_not_remove(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A SPA-tag pair this loader stored although the served store holds it raises.

    The SPA-tag group is a SUBSET of the served rule's pairs now that those rows are
    storable, so the containment is asserted and not inferred from the counts: a pair in
    the group but not in the overlap would be a record stored beside Babu's own. The
    overlap count is pinned to the smaller overlap first, so it is the containment that
    fires rather than the count.
    """
    monkeypatch.setattr(m, "SERVED_OVERLAP_PAIRS", 1)
    with pytest.raises(RuntimeError, match="were NOT removed as served"):
        _partition(overlap_pairs=[("b0005", "b0002")], overlap_cells=1)


@pytest.mark.usefixtures("synthetic_counts")
def test_the_partition_refuses_a_served_pair_no_group_accounts_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A served cell of this release the served rule did not remove raises.

    This is the drift the 727 -> 1,125 growth would have produced had Babu's new records
    landed on storable rows: the count pins would still pass and the cells would be
    stored twice, so the residue is checked pair by pair rather than by arithmetic.
    """
    monkeypatch.setattr(m, "SERVED_PAIRS_ON_A_SPA_TAG_RECIPIENT", 0)
    monkeypatch.setattr(m, "SERVED_OVERLAP_PAIRS", 1)
    monkeypatch.setattr(m, "SERVED_OVERLAP_PAIRS_ON_A_KEIO_RECIPIENT", 1)
    with pytest.raises(RuntimeError, match="the served rule does not remove"):
        _partition(
            spa_tag_pairs=(), overlap_pairs=[("b0005", "b0002")], overlap_cells=1
        )


def test_reading_the_served_store_refuses_two_records_of_one_oriented_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The partition counts served records against served pairs, so they must agree."""

    def two_of_one_pair(root: str) -> Iterator[dict[str, Any]]:
        leaves = [
            {"cassette": "cat", "systematic_gene_name": "b0005"},
            {"cassette": "kan", "systematic_gene_name": "b0002"},
        ]
        for score in (-1.0, -2.0):
            yield {
                "experiment": {
                    "genotype": {"perturbations": leaves},
                    "phenotype": {
                        "screen_id": m.BABU_SCREEN_TAG,
                        "gene_interaction": score,
                    },
                }
            }

    monkeypatch.setattr(
        "torchcell.verification.runners.stream_records", two_of_one_pair
    )
    with pytest.raises(RuntimeError, match="two records for the oriented pair"):
        m.read_served_babu("stubbed")


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def test_the_l0_to_l4_gate_passes_on_the_synthetic_build(
    built: m.GeneInteractionButland2008Dataset, synthetic: Path
) -> None:
    records = [built[index] for index in range(len(built))]
    released = m.released_scores(osp.join(synthetic, "raw", m.TABLE_S4))
    universe = {locus.tag for locus in MG1655_LOCI}
    report = m.verify_records(
        records,
        released=released,
        served=dict(SERVED_STUB),
        universe=universe,
        expected_count=len(KEPT_CELLS),
    )
    failures = [result.name for result in report.results if not result.passed]
    assert failures == [], report.summary()
    by_name = {result.name: result for result in report.results}
    assert by_name["recipient_leaf_states_its_array_row"].details["versions"] == {
        "Isolate 1": 8,
        "Isolate 2": 1,
        m.LABEL_SPA_TAG: 1,
    }
    assert (
        by_name["signed_unclamped_interaction_score_with_zero_reference"].details[
            "n_zero"
        ]
        == 1
    )
    assert (
        by_name["partitioned_from_the_served_babu2014_store"].details["n_shared"] == 0
    )


def test_released_scores_is_keyed_on_the_isolate(synthetic: Path) -> None:
    released = m.released_scores(osp.join(synthetic, "raw", m.TABLE_S4))
    assert released[("b0005", "b0001", "Isolate 1")] == -5.0
    assert released[("b0005", "b0001", "Isolate 2")] == 2.5
    assert released[("b0005", "b0006", m.LABEL_SPA_TAG)] == -7.0
    assert len(released) == len(SYNTHETIC_ARRAY) * len(SYNTHETIC_QUERIES)


# --------------------------------------------------------------------------- #
# Data-gated: the real mirror and the built dev store
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    root = os.environ.get("DATA_ROOT")
    if root is None or not osp.isdir(osp.join(root, "torchcell-raw")):
        pytest.skip("no DATA_ROOT with a raw mirror")
    return root


@pytest.mark.data
def test_the_raw_mirror_holds_exactly_the_consumed_files() -> None:
    manifest = m.load_manifest(_data_root())
    assert manifest.citation_key == m.CITATION_KEY
    assert manifest.doi == m.PAPER_DOI
    held = {record.path: record.sha256 for record in manifest.files}
    for raw in m.RAW_FILES:
        assert held[raw.mirror_relpath] == raw.sha256
        path = m.raw_mirror_dir(_data_root()) / raw.mirror_relpath
        assert path.stat().st_size == raw.bytes
    assert m.MIRROR_EXPECTATION in manifest.si_expected


@pytest.mark.data
def test_a_manifest_pin_mismatch_is_refused() -> None:
    manifest = m.load_manifest(_data_root())
    recorded = m.manifest_sha256(manifest, f"data/{m.TABLE_S4}")
    check_manifest_pin(f"data/{m.TABLE_S4}", recorded, m.DATA_SHA256[m.TABLE_S4])
    with pytest.raises(ManifestPinMismatchError):
        check_manifest_pin(f"data/{m.TABLE_S4}", recorded, "0" * 64)


@pytest.mark.data
def test_every_text_quoted_value_audits_against_the_literature_mirror() -> None:
    library = Path(_data_root()) / "torchcell-library"
    if not library.is_dir():
        pytest.skip("literature mirror is not mounted")
    for name in m.TEXT_QUOTED:
        result = audit_sourced_value(m.SOURCED_VALUES[name], library)
        assert result.passed, f"{name}: {result.message}"


@pytest.mark.data
def test_every_workbook_quoted_value_audits_against_its_pinned_cells() -> None:
    for entry in m.WORKBOOK_QUOTES:
        result = m.audit_workbook_quote(entry, _data_root())
        assert result.passed, f"{entry.name}: {result.message}"


@pytest.mark.data
def test_the_real_matrix_is_the_unfiltered_314847_cell_release() -> None:
    path = m.raw_mirror_dir(_data_root()) / f"data/{m.TABLE_S4}"
    scores = m.read_s_scores(path)
    assert len(scores.query_tags) == 39
    assert len(scores.recipient_tags) == 8073
    assert len(scores.recipient_tags) * len(scores.query_tags) == 314847
    assert len(set(scores.recipient_tags)) == 4117
    block = m.score_block(scores)
    assert block.shape == (8073, 39)
    assert "without any filtering parameters" in scores.title
    # the first data cell of the sheet, read by hand off si5.xls
    assert scores.recipient_tags[0] == "b0001"
    assert scores.query_tags[0] == "b0119"
    assert block[0][0] == 0.0456


@pytest.mark.data
def test_the_real_high_confidence_table_is_a_subset_of_the_real_matrix() -> None:
    mirror = m.raw_mirror_dir(_data_root())
    table = m.read_high_confidence(mirror / f"data/{m.TABLE_S3}")
    assert len(table) == 1379
    scores = m.read_s_scores(mirror / f"data/{m.TABLE_S4}")
    ledger = m.high_confidence_ledger(table, scores, [])
    assert ledger.located_in_matrix == ledger.identical_score == 1379
    assert ledger.ordered_pairs == 1288
    assert ledger.non_essential_pairs == 799
    assert ledger.spa_tag_pairs == 489


@pytest.mark.data
def test_the_dev_store_ledgers_state_the_measured_build() -> None:
    preprocess = Path(_data_root(), m.DATASET_ROOT_REL, "preprocess")
    if not preprocess.is_dir():
        pytest.skip("the dev store is not built")
    drops = json.loads((preprocess / "dropped_records.json").read_text())
    assert drops["source_records"] == 314847
    assert drops["kept_records"] == m.EXPECTED_RECORDS == 301803
    assert {rule["rule"]: rule["n_records"] for rule in drops["rules"]} == {
        m.RULE_NOT_A_TAG: 6318,
        m.RULE_REMAPPED: 4407,
        m.RULE_SELF_PAIR: 78,
        m.RULE_CONTRADICTION: 395,
        m.RULE_SERVED: 1846,
    }
    assert (drops["n_aggravating"], drops["n_alleviating"], drops["n_zero"]) == (
        146121,
        147747,
        7935,
    )
    assert drops["kept_queries"] == 39
    assert drops["kept_recipients"] == 3978
    assert drops["kept_hypomorph_records"] == 5413
    assert drops["kept_strain_versions"] == {
        "Isolate 1": 148408,
        "Isolate 2": 147982,
        m.LABEL_SPA_TAG: 5413,
    }


@pytest.mark.data
def test_the_dev_store_partition_is_the_measured_0_36_percent() -> None:
    """The partition as rebuilt once BOTH loaders store the SPA-tagged hypomorphs.

    Babu's own store grew from 38,579 to 41,988 records with PR #837, and 398 of the
    admitted records carry this screen, so the served-by-Babu side of the partition is
    1,125 rather than the 727 the first build measured. Since this loader took the same
    leaf (issue #792), those 398 are storable cells here too, so they reach the served
    rule rather than a SPA-tag rule: the overlap is 1,123 pairs over 1,846 cells, split
    725 Keio and 398 SPA-tag, and 2 pairs this release does not name.
    """
    preprocess = Path(_data_root(), m.DATASET_ROOT_REL, "preprocess")
    if not preprocess.is_dir():
        pytest.skip("the dev store is not built")
    partition = json.loads((preprocess / "served_partition.json").read_text())
    assert partition["served_records"] == 41988
    assert partition["served_butland_records"] == m.SERVED_BUTLAND_RECORDS == 1125
    assert partition["served_butland_pairs"] == m.SERVED_BUTLAND_PAIRS == 1125
    assert partition["served_fraction_of_this_release"] == pytest.approx(
        1125 / 314847, rel=1e-9
    )
    assert round(partition["served_fraction_of_this_release"] * 100, 2) == 0.36
    assert partition["shared_pairs"] == 0
    assert partition["overlap_pairs_dropped"] == m.SERVED_OVERLAP_PAIRS == 1123
    assert partition["overlap_cells_dropped"] == m.SERVED_OVERLAP_CELLS == 1846
    assert (
        partition["overlap_pairs_on_a_keio_recipient"]
        == m.SERVED_OVERLAP_PAIRS_ON_A_KEIO_RECIPIENT
        == 725
    )
    assert (
        partition["served_pairs_on_a_spa_tag_recipient"]
        == m.SERVED_PAIRS_ON_A_SPA_TAG_RECIPIENT
        == 398
    )
    assert partition["served_pairs_not_in_this_release"] == list(
        m.SERVED_PAIRS_NOT_IN_THIS_RELEASE
    )
    assert 725 + 398 + 2 == m.SERVED_BUTLAND_RECORDS
    assert 725 + 398 == m.SERVED_OVERLAP_PAIRS


@pytest.mark.data
def test_the_dev_store_records_validate_as_the_schema_pair() -> None:
    root = Path(_data_root(), m.DATASET_ROOT_REL)
    if not (root / "processed").is_dir():
        pytest.skip("the dev store is not built")
    from torchcell.verification.runners import stream_records

    cassettes: set[tuple[str, ...]] = set()
    rows: set[str] = set()
    references: set[float] = set()
    seen = 0
    for record in stream_records(str(root)):
        experiment = BacterialGeneInteractionExperiment.model_validate(
            record["experiment"]
        )
        reference = BacterialGeneInteractionExperimentReference.model_validate(
            record["reference"]
        )
        leaves = _leaves(experiment.genotype)
        cassettes.add(tuple(sorted(str(leaf.cassette) for leaf in leaves)))
        rows.add(_recipient(experiment.genotype))
        references.add(reference.phenotype_reference.gene_interaction)
        assert experiment.phenotype.screen_id is None
        assert experiment.environment.media.base_medium == "LB"
        seen += 1
        if seen == 50:
            break
    assert seen == 50
    assert cassettes == {("cat", "kan")}
    assert rows <= {"Isolate 1", "Isolate 2", m.LABEL_SPA_TAG}
    assert references == {0.0}


@pytest.mark.data
def test_the_dev_store_holds_the_spa_tag_half_on_the_marked_allele_leaf() -> None:
    """#792: the 5,413 hypomorph records, and no record stored twice against Babu.

    The hypomorph half is read off the built store rather than counted from the ledger,
    and every one of its recipient leaves carries the four fields footnote a states with
    no insertion site and no construction.
    """
    root = Path(_data_root(), m.DATASET_ROOT_REL)
    if not (root / "processed").is_dir():
        pytest.skip("the dev store is not built")
    from torchcell.verification.runners import stream_records

    hypomorphs = 0
    pairs: set[tuple[str, str]] = set()
    for record in stream_records(str(root)):
        leaves = {
            str(leaf["cassette"]): leaf
            for leaf in record["experiment"]["genotype"]["perturbations"]
        }
        pairs.add(
            (
                str(leaves["cat"]["systematic_gene_name"]),
                str(leaves["kan"]["systematic_gene_name"]),
            )
        )
        recipient = leaves["kan"]
        if recipient["perturbation_type"] != m.MARKED_ALLELE_TYPE:
            continue
        hypomorphs += 1
        assert recipient["cassette"] == "kan"
        assert recipient["tag"] == "SPA"
        assert recipient["terminus"] == "C"
        assert recipient["allele_effect"] == "hypomorphic"
        assert recipient["collection"] == m.LABEL_SPA_TAG
        assert recipient["insertion_site"] is None
        assert recipient["construction"] is None
    assert hypomorphs == 5413
    served, _ = m.read_served_babu(osp.join(_data_root(), m.BABU_ROOT_REL))
    assert pairs & set(served) == set()


# --------------------------------------------------------------------------- #
# Auditing a quote that lives in a workbook cell
# --------------------------------------------------------------------------- #
@pytest.fixture
def workbook_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A raw mirror holding the four synthetic workbooks, pinned to their own bytes."""
    root = tmp_path / "root"
    mirror = root / m.RAW_DIR_REL / "data"
    write_raw(mirror)
    pins = {raw.name: m._sha256(mirror / raw.name) for raw in m.RAW_FILES}  # noqa: SLF001
    monkeypatch.setattr(
        m,
        "RAW_FILES",
        tuple(
            raw.model_copy(
                update={
                    "sha256": pins[raw.name],
                    "bytes": (mirror / raw.name).stat().st_size,
                    "retrieval": raw.retrieval.model_copy(
                        update={"sha256": pins[raw.name]}
                    ),
                }
            )
            for raw in m.RAW_FILES
        ),
    )
    monkeypatch.setattr(m, "DATA_SHA256", pins)
    return root


@pytest.mark.parametrize("entry", m.WORKBOOK_QUOTES, ids=lambda e: e.name)
def test_a_workbook_quote_audits_against_the_synthetic_cells(
    entry: Any, workbook_mirror: Path
) -> None:
    result = m.audit_workbook_quote(entry, str(workbook_mirror))
    assert result.passed, result.message
    assert result.name == "provenance_audit"
    assert result.details["sha256_ok"] is True
    assert result.details["quote_present"] is True
    assert result.details["where"] == entry.where


def test_a_workbook_quote_fails_on_a_sha256_drift(
    workbook_mirror: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(m.DATA_SHA256, m.TABLE_S4, "0" * 64)
    result = m.audit_workbook_quote(m.WORKBOOK_QUOTES[0], str(workbook_mirror))
    assert not result.passed
    assert "sha256 drift" in result.message
    assert result.details["sha256_ok"] is False
    assert result.details["quote_present"] is False


def test_a_workbook_quote_fails_when_the_cells_no_longer_carry_it(
    workbook_mirror: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = next(e for e in m.WORKBOOK_QUOTES if e.name == "score_definition")
    monkeypatch.setitem(
        m.SOURCED_VALUES,
        "score_definition",
        m.SOURCED_VALUES["score_definition"].model_copy(
            update={"quote": "a sentence this workbook does not print"}
        ),
    )
    result = m.audit_workbook_quote(entry, str(workbook_mirror))
    assert not result.passed
    assert "quote no longer found" in result.message
    assert result.details["sha256_ok"] is True


def test_a_workbook_quote_raises_on_an_absent_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="source artifact not found"):
        m.audit_workbook_quote(m.WORKBOOK_QUOTES[0], str(tmp_path))


# --------------------------------------------------------------------------- #
# Linking the mirror into raw/
# --------------------------------------------------------------------------- #
def test_download_links_every_pinned_workbook_from_the_mirror(
    workbook_mirror: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(workbook_mirror))
    mirror = workbook_mirror / m.RAW_DIR_REL
    m.deposit_raw_mirror(
        sources=dict(sources_map(mirror / "data")), data_root=str(workbook_mirror)
    )
    dataset = m.GeneInteractionButland2008Dataset.__new__(
        m.GeneInteractionButland2008Dataset
    )
    raw = tmp_path / "linked"
    monkeypatch.setattr(
        type(dataset), "raw_dir", property(lambda self: str(raw)), raising=False
    )
    dataset.download()
    assert sorted(path.name for path in raw.iterdir()) == [
        raw_file.name for raw_file in m.RAW_FILES
    ]


def test_download_refuses_a_mirror_missing_a_pinned_workbook(
    workbook_mirror: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(workbook_mirror))
    mirror = workbook_mirror / m.RAW_DIR_REL
    m.deposit_raw_mirror(
        sources=dict(sources_map(mirror / "data")), data_root=str(workbook_mirror)
    )
    (mirror / "data" / m.TABLE_S3).unlink()
    dataset = m.GeneInteractionButland2008Dataset.__new__(
        m.GeneInteractionButland2008Dataset
    )
    monkeypatch.setattr(
        type(dataset),
        "raw_dir",
        property(lambda self: str(tmp_path / "linked")),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="required raw artifact missing"):
        dataset.download()
    assert not (tmp_path / "linked").exists()


# --------------------------------------------------------------------------- #
# Depositing the raw mirror
# --------------------------------------------------------------------------- #
def test_deposit_extends_a_manifest_another_run_wrote(workbook_mirror: Path) -> None:
    mirror = workbook_mirror / m.RAW_DIR_REL
    (mirror / "manifest.json").write_text(
        json.dumps(
            {
                "citation_key": m.CITATION_KEY,
                "doi": m.PAPER_DOI,
                "title": "eSGA: E. coli synthetic genetic array analysis",
                "files": [],
                "si_data_sources": [],
                "si_expected": ["another run's sentence"],
                "provenance_complete": True,
            }
        )
    )
    m.deposit_raw_mirror(
        sources=dict(sources_map(mirror / "data")), data_root=str(workbook_mirror)
    )
    manifest = json.loads((mirror / "manifest.json").read_text())
    assert [record["path"] for record in manifest["files"]] == [
        f"data/{raw.name}" for raw in m.RAW_FILES
    ]
    assert "another run's sentence" in manifest["si_expected"]
    assert m.MIRROR_EXPECTATION in manifest["si_expected"]
    assert manifest["provenance_complete"] is True


def test_deposit_refuses_to_overwrite_a_mirror_file_of_another_hash(
    workbook_mirror: Path, tmp_path: Path
) -> None:
    mirror = workbook_mirror / m.RAW_DIR_REL / "data"
    other = tmp_path / "other"
    other.mkdir()
    for raw in m.RAW_FILES:
        (other / raw.name).write_bytes((mirror / raw.name).read_bytes())
    (mirror / m.TABLE_S1).write_bytes(b"a different release")
    with pytest.raises(RuntimeError, match="exists with a different sha256"):
        m.deposit_raw_mirror(
            sources=dict(sources_map(other)), data_root=str(workbook_mirror)
        )


def sources_map(directory: Path) -> dict[str, Path]:
    """``{name: path}`` for every consumed workbook in ``directory``."""
    return {raw.name: directory / raw.name for raw in m.RAW_FILES}


def test_deposit_refuses_a_missing_source(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="no source given"):
        m.deposit_raw_mirror(sources={}, data_root=str(tmp_path))


def test_deposit_refuses_a_hash_mismatch(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    for raw in m.RAW_FILES:
        (sources / raw.name).write_bytes(b"not the released bytes")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        m.deposit_raw_mirror(
            sources=dict(sources_map(sources)), data_root=str(tmp_path)
        )
