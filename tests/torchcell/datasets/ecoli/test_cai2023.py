# tests/torchcell/datasets/ecoli/test_cai2023.py
# [[tests.torchcell.datasets.ecoli.test_cai2023]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_cai2023.py
"""The MCF2Chem 2023 provenance record: the decision, and the measurement behind it.

Synthetic tests run everywhere. They write WordprocessingML files from scratch -- the
three supplementary files in the shapes the real release has, plus a counterfactual
file that DOES carry a production table -- so the inventory rule is shown to separate
the two rather than merely to return ``False`` on the real bytes.

The ``@pytest.mark.data`` tests read the real mirror: they audit every quote against
the pinned ``sha256``, inventory the real supplements, and assert the one statement the
docx layer cannot audit after the fact (Table S2's coverage rates).
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pytest

import torchcell.datasets.ecoli.cai2023 as cai
from torchcell.verification.sourced import (
    ProvenanceGapReason,
    SourcedValue,
    audit_sourced_value,
)

DATA_ROOT = os.environ.get("DATA_ROOT")

#: The shapes :func:`cai.release_inventory` measures on the real release, which is what
#: the synthetic files reproduce: one table per file, 269 x 2, 5 x 4 and 7 x 2.
REAL_SHAPES = {"si1": [(269, 2)], "si2": [(5, 4)], "si3": [(7, 2)]}

TABLE_S2_ROWS = (
    ("Year", "2016", "2017", "2018"),
    ("Number of articles collected by MCF2Chem", "40", "42", "36"),
    ("Number of articles published in Metabolic Engineering", "61", "65", "60"),
    ("MCF2Chem coverage rate", "66%", "65%", "60%"),
    ("MCF2Chem average coverage rate", "63%"),
)


# --------------------------------------------------------------------------- #
# Synthetic WordprocessingML
# --------------------------------------------------------------------------- #
def _paragraph(text: str) -> str:
    return f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"


def _row(cells: tuple[str, ...]) -> str:
    body = "".join(f"<w:tc>{_paragraph(cell)}</w:tc>" for cell in cells)
    return f"<w:tr>{body}</w:tr>"


def _table(rows: tuple[tuple[str, ...], ...]) -> str:
    return "<w:tbl>" + "".join(_row(row) for row in rows) + "</w:tbl>"


def write_docx(path: Path, body: str) -> None:
    """A minimal .docx carrying ``body`` as its WordprocessingML body."""
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/'
        f'2006/main"><w:body>{body}</w:body></w:document>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)


def _si1_rows(n_reviews: int = 268) -> tuple[tuple[str, ...], ...]:
    header = ("Review_title", "Review_doi")
    body = tuple(
        (f"20{17 + i % 6}-A review of something", f"10.1000/review.{i}")
        for i in range(n_reviews)
    )
    return (header, *body)


def _si3_rows() -> tuple[tuple[str, ...], ...]:
    routes = ("S2C", "S2C2C", "S2S2C", "C2S", "C2S2S", "C2C2S")
    return (("Route", "Score"), *((route, "1") for route in routes))


def write_release(base: Path) -> Path:
    """The three supplementary files, in the real release's shapes."""
    library = base / cai.LIBRARY_DIR_REL / cai.CITATION_KEY
    (library / "si").mkdir(parents=True)
    write_docx(library / cai.SI_FILES["si1"].relpath, _table(_si1_rows()))
    write_docx(
        library / cai.SI_FILES["si2"].relpath,
        "".join(_paragraph(f"Fig. S{i} A figure.") for i in range(1, 10))
        + _table(TABLE_S2_ROWS),
    )
    write_docx(library / cai.SI_FILES["si3"].relpath, _table(_si3_rows()))
    return library


# --------------------------------------------------------------------------- #
# The decision the module records
# --------------------------------------------------------------------------- #
def test_the_module_registers_no_dataset() -> None:
    """The row is NOT loaded, so nothing of it may reach the registry or the graph."""
    from torchcell.datasets.dataset_registry import dataset_registry

    assert not hasattr(cai, "experiment_class")
    assert not any(cls.__module__ == cai.__name__ for cls in dataset_registry.values())


def test_the_unreleased_records_are_a_terminal_typed_gap() -> None:
    """Not recoverable work: there is no artifact to fetch, so not a deferral."""
    gap = cai.PER_RECORD_VALUES_GAP
    assert gap.field == "production_records"
    assert gap.reason is ProvenanceGapReason.not_carried_by_curation
    # The worklist field stays empty: a deferral would promise a fetchable artifact.
    assert gap.resolve_with is None
    assert gap.looked_in is not None
    assert gap.looked_in.citation_key == cai.CITATION_KEY


def test_the_extraction_window_closes_before_the_paper_appeared() -> None:
    """The window is the reviews' publication dates, not the primary work's."""
    assert cai.EXTRACTION_FROM_REVIEWS.value == ("2017-08-01", "2022-07-31")
    assert cai.ORIGINAL_ARTICLE_SPAN.value == (1946, 2022)
    assert cai.REVIEW_WINDOW_END < "2023-01-01"


def test_the_per_record_citation_is_a_review_reference_column() -> None:
    """A per-record citation exists; it is not evidence for the number beside it."""
    assert cai.PER_RECORD_REFERENCE_RULE.value == "review_reference_column"
    assert "reference columns in review tables" in cai.PER_RECORD_REFERENCE_RULE.quote
    assert cai.RECORD_COUNTS.value == {
        "records": 8888,
        "reviews": 268,
        "original_articles": 4765,
        "patents": 92,
    }
    assert cai.PATENT_RECORDS.value == 92
    assert cai.BACTERIAL_RECORDS.value["records"] == 5276


def test_every_schema_blocker_names_a_required_field_and_carries_a_quote() -> None:
    """A blocker without the release's own statement of the limit is an opinion."""
    fields = [blocker.field for blocker in cai.SCHEMA_BLOCKERS]
    assert fields == [
        "ProductTiterExperimentReference.phenotype_reference",
        "ProductTiterPhenotype.product",
        "ProductTiterPhenotype.titer",
        "ProductTiterPhenotype.titer_unit",
        "ProductTiterExperiment.genotype",
    ]
    for blocker in cai.SCHEMA_BLOCKERS:
        assert isinstance(blocker.evidence, SourcedValue)
        assert blocker.evidence.provenance.citation_key == cai.CITATION_KEY
        assert blocker.evidence.provenance.sha256 == cai.PAPER_MD_SHA256
        assert blocker.evidence.provenance.page
        assert blocker.evidence.quote.strip()


def test_the_blocked_titer_fields_are_required_on_the_live_schema() -> None:
    """The blockers are read off the schema, so a schema change breaks this test."""
    from torchcell.datamodels.schema import (
        ProductTiterExperiment,
        ProductTiterExperimentReference,
        ProductTiterPhenotype,
    )

    assert ProductTiterExperimentReference.model_fields[
        "phenotype_reference"
    ].is_required()
    for name in ("product", "titer", "titer_unit"):
        assert ProductTiterPhenotype.model_fields[name].is_required()
    assert ProductTiterExperiment.model_fields["genotype"].is_required()


def test_the_recorded_probe_found_no_data_route() -> None:
    """The one named channel answered a holding page and a dead API."""
    probed = {row.path: row.status for row in cai.ACCESSION_PROBE_2026_10_08}
    assert probed["/"] == 200
    assert probed["/openapi.json"] == 502
    assert all(
        status == 404
        for path, status in probed.items()
        if path not in ("/", "/openapi.json")
    )
    assert cai.PROBE_PATHS[0] == ""
    assert len(cai.PROBE_PATHS) == len(cai.ACCESSION_PROBE_2026_10_08)


# --------------------------------------------------------------------------- #
# Synthetic: the inventory rule
# --------------------------------------------------------------------------- #
def test_the_docx_reader_walks_paragraphs_and_table_cells(tmp_path: Path) -> None:
    path = tmp_path / "si.docx"
    write_docx(path, _paragraph("Table S2 coverage") + _table((("a", "b"), ("1", "2"))))
    assert cai.docx_text(path) == "Table S2 coverage\na\nb\n1\n2"
    assert cai.docx_tables(path) == [[["a", "b"], ["1", "2"]]]


def test_a_docx_without_a_body_is_refused(tmp_path: Path) -> None:
    """No fallback: a file that is not WordprocessingML raises rather than reads empty."""
    path = tmp_path / "empty.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/'
            'wordprocessingml/2006/main"/>',
        )
    with pytest.raises(ValueError, match="no WordprocessingML body"):
        cai.docx_body(path)


def test_the_release_holds_no_production_table(tmp_path: Path) -> None:
    library = write_release(tmp_path)
    inventory = cai.release_inventory(library)
    assert inventory.citation_key == cai.CITATION_KEY
    assert inventory.accession_url == cai.ACCESSION_URL
    assert inventory.table_shapes == REAL_SHAPES
    assert inventory.n_tables == 3
    assert inventory.n_production_tables == 0
    assert inventory.per_record_artifact_released is False
    assert [artifact.relpath for artifact in inventory.si_files] == [
        "si/si1.docx",
        "si/si2.docx",
        "si/si3.docx",
    ]


@pytest.mark.parametrize(
    "header",
    [
        ("Strain", "Compound", "Titer (g/L)"),
        ("species", "product", "productivity"),
        ("id", "content"),
    ],
)
def test_a_file_that_did_carry_records_would_be_found(
    tmp_path: Path, header: tuple[str, ...]
) -> None:
    """The rule is generous on purpose: one production header is enough to flip it."""
    library = write_release(tmp_path)
    write_docx(
        library / cai.SI_FILES["si3"].relpath,
        _table((header, tuple("1" for _ in header))),
    )
    inventory = cai.release_inventory(library)
    assert inventory.n_production_tables == 1
    assert inventory.per_record_artifact_released is True


def test_the_inventory_refuses_a_partial_set_of_supplements() -> None:
    with pytest.raises(ValueError, match="expected tables for"):
        cai.inventory_from_tables({"si1": [[["Review_title", "Review_doi"]]]})


def test_table_s2_coverage_is_read_and_asserted(tmp_path: Path) -> None:
    library = write_release(tmp_path)
    assert cai.table_s2_coverage(library) == cai.TABLE_S2_COVERAGE
    assert cai.TABLE_S2_COVERAGE["average"] == "63%"


def test_a_moved_table_s2_rate_raises(tmp_path: Path) -> None:
    """A docx quote cannot be audited after the fact, so a drift must raise on read."""
    library = write_release(tmp_path)
    moved = (
        *TABLE_S2_ROWS[:3],
        ("MCF2Chem coverage rate", "99%", "65%", "60%"),
        TABLE_S2_ROWS[4],
    )
    write_docx(library / cai.SI_FILES["si2"].relpath, _table(moved))
    with pytest.raises(ValueError, match="Table S2 coverage moved"):
        cai.table_s2_coverage(library)


def test_two_tables_in_additional_file_two_is_refused(tmp_path: Path) -> None:
    library = write_release(tmp_path)
    write_docx(
        library / cai.SI_FILES["si2"].relpath,
        _table(TABLE_S2_ROWS) + _table((("Year", "2016"),)),
    )
    with pytest.raises(ValueError, match="holds 2 tables, expected 1"):
        cai.table_s2_coverage(library)


def test_library_dir_is_the_keys_directory_under_the_given_root(tmp_path: Path) -> None:
    assert cai.library_dir(str(tmp_path)) == (
        tmp_path / cai.LIBRARY_DIR_REL / cai.CITATION_KEY
    )


def test_library_dir_falls_back_to_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without an argument the root is DATA_ROOT, and a missing one raises."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert cai.library_dir() == tmp_path / cai.LIBRARY_DIR_REL / cai.CITATION_KEY
    monkeypatch.delenv("DATA_ROOT")
    monkeypatch.setattr(cai, "load_dotenv", lambda *args, **kwargs: False)
    with pytest.raises(KeyError, match="DATA_ROOT"):
        cai.library_dir()


# --------------------------------------------------------------------------- #
# The real mirror
# --------------------------------------------------------------------------- #
@pytest.mark.data
@pytest.mark.skipif(DATA_ROOT is None, reason="DATA_ROOT is not set")
def test_every_quote_audits_against_the_pinned_paper() -> None:
    """A sourced value whose quote moved is a value we can no longer defend."""
    assert DATA_ROOT is not None
    library_root = Path(DATA_ROOT) / cai.LIBRARY_DIR_REL
    if not (library_root / cai.CITATION_KEY / cai.PAPER_MD).exists():
        pytest.skip("the OCR mirror for this key is not present")
    sourced = [
        value for value in vars(cai).values() if isinstance(value, SourcedValue)
    ] + [blocker.evidence for blocker in cai.SCHEMA_BLOCKERS]
    assert len(sourced) == 15
    for value in sourced:
        result = audit_sourced_value(value, library_root)
        assert result.passed, f"{value.quote[:60]!r}: {result.message}"


@pytest.mark.data
@pytest.mark.skipif(DATA_ROOT is None, reason="DATA_ROOT is not set")
def test_the_real_supplements_match_the_pinned_digests_and_shapes() -> None:
    """The inventory the module docstring states, measured on the real bytes."""
    from torchcell.data import verify_sha256

    assert DATA_ROOT is not None
    library = cai.library_dir(DATA_ROOT)
    if not (library / cai.SI_FILES["si1"].relpath).exists():
        pytest.skip("the SI capture for this key is not present")
    verify_sha256(library / cai.PAPER_MD, cai.PAPER_MD_SHA256)
    for artifact in cai.SI_FILES.values():
        path = library / artifact.relpath
        verify_sha256(path, artifact.sha256)
        assert path.stat().st_size == artifact.n_bytes
    inventory = cai.release_inventory(library)
    assert inventory.table_shapes == REAL_SHAPES
    assert inventory.n_production_tables == 0
    assert inventory.per_record_artifact_released is False
    assert cai.table_s2_coverage(library) == cai.TABLE_S2_COVERAGE


@pytest.mark.data
@pytest.mark.skipif(DATA_ROOT is None, reason="DATA_ROOT is not set")
def test_table_s1_holds_two_hundred_and_sixty_eight_reviews() -> None:
    """The source corpus is reviews, and the count matches the paper's own figure."""
    assert DATA_ROOT is not None
    library = cai.library_dir(DATA_ROOT)
    if not (library / cai.SI_FILES["si1"].relpath).exists():
        pytest.skip("the SI capture for this key is not present")
    tables = cai.docx_tables(library / cai.SI_FILES["si1"].relpath)
    assert len(tables) == 1
    rows = tables[0]
    assert rows[0] == ["Review_title", "Review_doi"]
    assert len(rows) - 1 == cai.RECORD_COUNTS.value["reviews"] == 268
