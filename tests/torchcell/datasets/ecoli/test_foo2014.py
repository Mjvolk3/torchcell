# tests/torchcell/datasets/ecoli/test_foo2014.py
# [[tests.torchcell.datasets.ecoli.test_foo2014]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_foo2014.py
"""The Foo 2014 E. coli isopentenol titer loader.

Synthetic tests run everywhere: they exercise the standard-library ``.docx`` table
reader over a Table S3 stand-in, the Table 1 parse out of the module's own verbatim
rows, the titer and environment builders with the typed gaps this paper forces, every
cross-source refusal, and a full end-to-end build over a synthetic MG1655 assembly, a
synthetic Table S3 and a synthetic OCR mirror -- no network and no ``$DATA_ROOT``.

The synthetic fixtures reproduce the released strings and numbers exactly, because the
checks they feed are joins onto OTHER statements of the same thing (Table 1's own
``Improvement in titer (%)`` column, the Methods' plasmid pair): a fixture with invented
values would exercise the plumbing while retiring the check.

The ``@pytest.mark.data`` tests read the real ``$DATA_ROOT``: the raw mirror's recorded
digest against the module pin, every sourced quote re-read out of the pinned ``paper.md``
bytes, the eight overexpressed symbols typed against the deposited MG1655 annotation,
and L0 to L4 over the built LMDB. They are skipped unless the mirrors and the built
store are present.

Derived expectations for the pinned bytes: nine records (Table 1's eight tolerance
strains plus the PS+RFP control of its footnote b), 28 article quotes, eight locus tags
(seven matched on a gene symbol and ``gidB`` on a synonym of ``rsmG``), and a reference
titer of 834 ug/mL.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd
import pytest

import tests.torchcell.sequence.genome._bacterial_fixtures as fixtures
import torchcell.datasets.ecoli.foo2014 as foo
from tests.torchcell.sequence.genome._bacterial_fixtures import SyntheticLocus
from torchcell.datamodels.media import MM9_FOO2014
from torchcell.datamodels.schema import (
    Compound,
    ConcentrationUnit,
    EndpointRule,
    Genotype,
    ProductTiterExperiment,
    ProductTiterExperimentReference,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.literature.manifest import ROLE_SI_DATA, RetrievalMethod
from torchcell.sequence.genome.ecoli.k12 import MG1655_ASSEMBLY, EcoliK12MG1655Genome
from torchcell.verification.sourced import ProvenanceGapReason

DATA_ROOT = os.environ.get("DATA_ROOT", "")
_MIRROR = Path(DATA_ROOT, "torchcell-raw", foo.CITATION_KEY) if DATA_ROOT else None
_LIBRARY = Path(DATA_ROOT, "torchcell-library", foo.CITATION_KEY) if DATA_ROOT else None
_STORE = Path(DATA_ROOT, foo.DEV_TREE_RELPATH) if DATA_ROOT else None

requires_mirror = pytest.mark.skipif(
    _MIRROR is None or not (_MIRROR / foo.SI_TABLE3_REL).exists(),
    reason="the Foo 2014 raw mirror is not deposited under $DATA_ROOT",
)
requires_library = pytest.mark.skipif(
    _LIBRARY is None or not (_LIBRARY / foo.PAPER_MD).exists(),
    reason="the Foo 2014 OCR mirror is not present under $DATA_ROOT",
)
requires_store = pytest.mark.skipif(
    _STORE is None or not (_STORE / "processed" / "lmdb").exists(),
    reason="the Foo 2014 dev store is not built under $DATA_ROOT",
)


# --------------------------------------------------------------------------- #
# A synthetic Table S3, written the way the loader reads one
# --------------------------------------------------------------------------- #
_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

#: Table S3's ``Production strains`` block as released, in the order it lists them.
PRODUCTION_STRAIN_ROWS: tuple[tuple[str, str], ...] = (
    ("PS+SoxS", "soxS"),
    ("PS+GidB", "gidB"),
    ("PS+NrdH", "nrdH"),
    ("PS+YqhD", "yqhD"),
    ("PS+IbpA", "ibpA"),
    ("PS+MetR", "metR"),
    ("PS+Fpr", "fpr"),
    ("PS+MdlB", "mdlB"),
    ("PS+RFP", "rfp"),
)
CHASSIS = "pJBEI-6830 + pJBEI-6833"
#: Table S3's ``Production plasmids`` block as released.
PRODUCTION_PLASMID_ROWS: tuple[tuple[str, str], ...] = (
    ("pJBEI-6830", "pBbA5c-MevTsa-PMK-MK"),
    ("pJBEI-6833", "pTrc99A-NudB-PMD"),
)


def _paragraph(text: str) -> str:
    return f'<w:p><w:r><w:t xml:space="preserve">{text}</w:t></w:r></w:p>'


def _row(cells: tuple[str, ...]) -> str:
    return (
        "<w:tr>" + "".join(f"<w:tc>{_paragraph(c)}</w:tc>" for c in cells) + "</w:tr>"
    )


def _docx(path: Path, rows: tuple[tuple[str, ...], ...], *, tables: int = 1) -> Path:
    """Write a minimal ``.docx`` whose body holds ``tables`` copies of ``rows``."""
    table = "<w:tbl>" + "".join(_row(r) for r in rows) + "</w:tbl>"
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_W}"><w:body>{table * tables}</w:body></w:document>'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/'
            'package/2006/content-types"/>',
        )
        archive.writestr("word/document.xml", document)
    return path


def table_s3_rows(
    strains: tuple[tuple[str, str], ...] = PRODUCTION_STRAIN_ROWS,
    plasmids: tuple[tuple[str, str], ...] = PRODUCTION_PLASMID_ROWS,
    *,
    chassis_row: bool = True,
) -> tuple[tuple[str, ...], ...]:
    """The released Table S3 rows the loader reads, in their released order."""
    rows: list[tuple[str, ...]] = [("Plasmids", "Description", "Reference")]
    rows.append(("Production plasmids", "", ""))
    rows.extend((name, composition, "(45)") for name, composition in plasmids)
    rows.append(("", "", ""))
    rows.append(("Strains", "Description", "Reference"))
    rows.append(("Production strains", "", ""))
    rows.extend(
        (strain, f"{CHASSIS} + pBbS5k-{gene}", "This study") for strain, gene in strains
    )
    if chassis_row:
        rows.append((foo.CHASSIS_STRAIN, CHASSIS, "(45)"))
    return tuple(rows)


def write_table_s3(path: Path, **kwargs: Any) -> Path:
    """A synthetic Table S3 document carrying the two blocks the loader reads."""
    return _docx(path, table_s3_rows(**kwargs))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# The .docx reader
# --------------------------------------------------------------------------- #
def test_the_reader_returns_every_row_of_the_single_table(tmp_path: Path) -> None:
    """A ``.docx`` body is one ``w:tbl``; each row is its cells, whitespace-normalized."""
    rows = foo.read_table_s3(write_table_s3(tmp_path / "si9.docx"))
    assert rows[0] == ["Plasmids", "Description", "Reference"]
    assert rows[-1] == [foo.CHASSIS_STRAIN, CHASSIS, "(45)"]
    assert len(rows) == len(table_s3_rows())


def test_the_reader_normalizes_whitespace_inside_a_cell(tmp_path: Path) -> None:
    """A cell split across runs and lines is one space-joined string."""
    path = _docx(tmp_path / "si9.docx", ((" PS+SoxS\n ", "  a   b ", ""),))
    assert foo.read_table_s3(path) == [["PS+SoxS", "a b", ""]]


def test_the_reader_refuses_a_document_that_is_not_one_table(tmp_path: Path) -> None:
    """Two tables is a replaced or re-numbered supplement, not a table to pick from."""
    path = _docx(tmp_path / "si9.docx", table_s3_rows(), tables=2)
    with pytest.raises(RuntimeError, match="holds 2 tables"):
        foo.read_table_s3(path)


def test_the_reader_refuses_a_document_with_no_body(tmp_path: Path) -> None:
    """A document whose root carries no ``w:body`` is not a Word document we read."""
    path = tmp_path / "si9.docx"
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml", f'<w:document xmlns:w="{_W}"></w:document>'
        )
    with pytest.raises(RuntimeError, match="has no w:body"):
        foo.read_table_s3(path)


# --------------------------------------------------------------------------- #
# The two blocks
# --------------------------------------------------------------------------- #
def test_the_nine_production_strains_are_read_in_their_released_order(
    tmp_path: Path,
) -> None:
    """One shared chassis plus exactly one pBbS5k gene is the whole genotype axis."""
    strains = foo.read_production_strains(write_table_s3(tmp_path / "si9.docx"))
    assert [(s.strain, s.gene) for s in strains] == list(PRODUCTION_STRAIN_ROWS)
    assert {s.plasmid for s in strains} == {
        f"pBbS5k-{gene}" for _, gene in PRODUCTION_STRAIN_ROWS
    }
    assert {s.composition for s in strains} == {
        f"{CHASSIS} + pBbS5k-{gene}" for _, gene in PRODUCTION_STRAIN_ROWS
    }


def test_a_production_strain_that_is_not_chassis_plus_one_gene_is_refused(
    tmp_path: Path,
) -> None:
    """A row that reads otherwise is a changed table, not a row to skip."""
    rows = list(table_s3_rows())
    index = rows.index(("PS+SoxS", f"{CHASSIS} + pBbS5k-soxS", "This study"))
    rows[index] = ("PS+SoxS", f"{CHASSIS} + pBbA5k-soxS", "This study")
    path = _docx(tmp_path / "si9.docx", tuple(rows))
    with pytest.raises(
        RuntimeError, match="not 'pJBEI-6830 \\+ pJBEI-6833 \\+ pBbS5k-"
    ):
        foo.read_production_strains(path)


def test_two_production_strain_headings_are_refused(tmp_path: Path) -> None:
    """Exactly one block names the strains this dataset is about."""
    rows = table_s3_rows() + (("Production strains", "", ""),)
    path = _docx(tmp_path / "si9.docx", rows)
    with pytest.raises(RuntimeError, match="holds 2 'Production strains' headings"):
        foo.read_production_strains(path)


def test_the_production_plasmids_block_is_the_quoted_composition(
    tmp_path: Path,
) -> None:
    """The one Table S3 quote no article quote covers, read back from the bytes."""
    plasmids = foo.read_production_plasmids(write_table_s3(tmp_path / "si9.docx"))
    assert plasmids == dict(PRODUCTION_PLASMID_ROWS)
    assert tuple(plasmids.values()) == tuple(foo.CHASSIS_COMPOSITION.value)


def test_a_production_plasmid_block_naming_other_plasmids_is_refused(
    tmp_path: Path,
) -> None:
    """A changed block is a changed chassis, and the Methods name the pair."""
    path = write_table_s3(
        tmp_path / "si9.docx", plasmids=(("pJBEI-6830", "pBbA5c-MevTsa-PMK-MK"),)
    )
    with pytest.raises(RuntimeError, match="names \\['pJBEI-6830'\\]"):
        foo.read_production_plasmids(path)


def test_two_production_plasmid_headings_are_refused(tmp_path: Path) -> None:
    """One block, as for the strains."""
    rows = (("Production plasmids", "", ""),) + table_s3_rows()
    with pytest.raises(RuntimeError, match="holds 2 'Production plasmids' headings"):
        foo.read_production_plasmids(_docx(tmp_path / "si9.docx", rows))


def test_the_chassis_row_is_the_pair_the_methods_state(tmp_path: Path) -> None:
    """``PS`` carries no pBbS5k plasmid: it is what every record is an edit of."""
    row = foo.chassis_row(write_table_s3(tmp_path / "si9.docx"))
    assert row[:2] == [foo.CHASSIS_STRAIN, str(foo.CHASSIS_PLASMIDS.value)]


def test_a_table_without_the_chassis_row_is_refused(tmp_path: Path) -> None:
    """Without ``PS`` there is no stated chassis to write the records against."""
    path = write_table_s3(tmp_path / "si9.docx", chassis_row=False)
    with pytest.raises(RuntimeError, match="has no 'PS' row"):
        foo.chassis_row(path)


# --------------------------------------------------------------------------- #
# Table 1, parsed out of its own verbatim rows
# --------------------------------------------------------------------------- #
def test_table_1_parses_to_the_eight_released_titers() -> None:
    """The stored titer comes from the bytes the quote is checked against."""
    assert [(row.gene, row.titer_mg_per_l) for row in foo.TABLE1_ROWS] == [
        ("soxS", 838.0),
        ("nrdH", 860.0),
        ("mdlB", 931.0),
        ("ibpA", 967.0),
        ("gidB", 965.0),
        ("fpr", 989.0),
        ("yqhD", 994.0),
        ("metR", 1290.0),
    ]
    assert [row.uncertainty_mg_per_l for row in foo.TABLE1_ROWS] == [
        29.0,
        3.0,
        16.0,
        23.0,
        1.0,
        10.0,
        13.0,
        20.0,
    ]
    assert foo.TOLERANCE_GENES == tuple(row.gene for row in foo.TABLE1_ROWS)
    assert set(foo.STRAIN_TO_GENE.values()) == set(foo.TOLERANCE_GENES) | {
        foo.CONTROL_GENE
    }


def test_a_table_1_row_with_the_wrong_cell_count_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Six cells is the released row shape; anything else is a changed table."""
    monkeypatch.setattr(foo, "_Q_TABLE1_ROWS", ("<tr><td>soxS</td><td>838</td></tr>",))
    with pytest.raises(RuntimeError, match="has 2 cells, not 6"):
        foo.parse_table1()


def test_every_stored_titer_reproduces_the_released_improvement_column() -> None:
    """The L4 join in pure form: a DIFFERENT column of the same released table."""
    control = float(foo.CONTROL_TITER_MG_PER_L.value)
    assert [
        foo._improvement_percent(row.titer_mg_per_l, control) for row in foo.TABLE1_ROWS
    ] == [row.improvement_percent for row in foo.TABLE1_ROWS]


def test_the_released_reference_titer_and_its_replicate_design_are_sourced() -> None:
    """834 mg/L over three replicates, each quoting the pinned article bytes."""
    assert float(foo.CONTROL_TITER_MG_PER_L.value) == 834.0
    assert float(foo.CONTROL_UNCERTAINTY_MG_PER_L.value) == 5.0
    assert int(foo.N_REPLICATES.value) == 3
    assert "Averages from triplicates" in str(foo.N_REPLICATES.quote)
    assert "834" in str(foo.CONTROL_TITER_MG_PER_L.quote)
    for sourced in (
        foo.CONTROL_TITER_MG_PER_L,
        foo.N_REPLICATES,
        foo.UNCERTAINTY_TYPE_NOT_STATED,
        foo.UNCERTAINTY_BACK_SOLVE,
    ):
        assert sourced.provenance.sha256 == foo.PAPER_MD_SHA256
        assert sourced.provenance.source_uri == foo.PAPER_MD
    assert foo.CHASSIS_COMPOSITION.provenance.sha256 == foo.SI_TABLE3_SHA256
    assert foo.UNCERTAINTY_TYPE_NOT_STATED.value is None
    assert foo.UNCERTAINTY_BACK_SOLVE.value == "inconclusive"


# --------------------------------------------------------------------------- #
# The quote audit over a synthetic OCR mirror
# --------------------------------------------------------------------------- #
def _synthetic_paper_md() -> str:
    """An OCR stand-in carrying every quote the loader reads out of ``paper.md``."""
    return "\n\n".join(foo.PAPER_QUOTES)


def _write_library(root: Path, text: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = root / "torchcell-library" / foo.CITATION_KEY / foo.PAPER_MD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(foo, "PAPER_MD_SHA256", _sha256(path))
    return path


def test_the_quote_audit_hashes_the_ocr_before_it_trusts_a_quote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Table 1 is the one consumed column with no released data file behind it."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    _write_library(tmp_path, _synthetic_paper_md(), monkeypatch)
    assert foo.verify_paper_quotes() == len(foo.PAPER_QUOTES) == 28


def test_a_drifted_ocr_hash_stops_the_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed mirror is upstream drift, and every quote must be re-read."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    _write_library(tmp_path, _synthetic_paper_md(), monkeypatch)
    monkeypatch.setattr(foo, "PAPER_MD_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="the OCR mirror moved"):
        foo.verify_paper_quotes()


def test_a_quote_the_ocr_no_longer_carries_stops_the_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A titer whose quote vanished is not a value this loader will store."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    text = _synthetic_paper_md().replace(foo._Q_CONTROL_TITER, "")
    _write_library(tmp_path, text, monkeypatch)
    with pytest.raises(RuntimeError, match="quotes are not verbatim"):
        foo.verify_paper_quotes()


# --------------------------------------------------------------------------- #
# The record parts
# --------------------------------------------------------------------------- #
def test_the_product_is_the_canonical_isoprenol_entity() -> None:
    """Resolved through the paper's own IUPAC name, so these titers join that axis."""
    compound = foo.isopentenol()
    assert compound.inchikey == "CPJRRXSHAYUTGL-UHFFFAOYSA-N"
    assert compound.pubchem_cid == 12988
    assert foo.PRODUCT_IDENTITY.value == foo.PRODUCT_SYNONYM


def test_a_product_that_stops_resolving_stops_the_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unresolved compound would split the isoprenol axis in two."""
    monkeypatch.setattr(
        foo, "resolved_compound", lambda name: Compound(name=name), raising=True
    )
    with pytest.raises(RuntimeError, match="no longer resolves to a compound"):
        foo.isopentenol()


def test_the_titer_phenotype_stores_the_released_mg_per_l_as_ug_per_ml() -> None:
    """1 mg/L is exactly 1 ug/mL, so no arithmetic touches a source value."""
    phenotype = foo.titer_phenotype(834.0)
    assert phenotype.titer == 834.0
    assert phenotype.titer_unit == ConcentrationUnit.ug_per_ml
    assert phenotype.n_samples == 3
    assert phenotype.sample_unit == SampleUnit.biological_replicate
    assert phenotype.quantification_method == "gas chromatography"


def test_every_uncertainty_is_a_typed_gap_and_never_a_guess() -> None:
    """The plus-minus numbers are released; what they ARE is stated nowhere."""
    phenotype = foo.titer_phenotype(1290.0)
    assert phenotype.titer_uncertainty is None
    assert phenotype.titer_uncertainty_type is None
    gaps = {gap.field: gap for gap in phenotype.provenance_gaps}
    assert set(gaps) == {
        "titer_uncertainty",
        "titer_uncertainty_type",
        "product_yield",
        "product_yield_unit",
        "productivity",
        "productivity_unit",
    }
    assert "KIND is not" in str(gaps["titer_uncertainty"].note)
    assert all(
        gap.reason == ProvenanceGapReason.not_reported_by_primary
        for gap in phenotype.provenance_gaps
    )


def test_the_environment_is_the_one_production_condition() -> None:
    """MM9 at 30 C, read at 48 h, with the two additions the source scopes here."""
    environment = foo.environment()
    assert environment.media == MM9_FOO2014
    assert environment.temperature == Temperature(value=30.0)
    assert environment.duration_hours == 48.0
    assert environment.aerobicity == "aerobic"
    additions = [
        perturbation
        for perturbation in environment.perturbations
        if isinstance(perturbation, SmallMoleculePerturbation)
    ]
    assert len(additions) == len(environment.perturbations) == 2
    assert [
        (
            addition.compound.name,
            addition.concentration.value,
            addition.concentration.unit,
        )
        for addition in additions
    ] == [
        ("IPTG", 500.0, ConcentrationUnit.micromolar),
        ("kanamycin", 15.0, ConcentrationUnit.ug_per_ml),
    ]


def test_the_culture_format_types_the_two_fields_the_source_never_states() -> None:
    """A 5 mL shaken culture with no vessel and no speed released."""
    culture = foo._culture_format()
    assert culture.working_volume_ul == 5000.0
    assert culture.inoculum_od600 == 0.08
    assert culture.endpoint == EndpointRule.fixed_duration
    assert culture.vessel is None
    assert culture.shaking_rpm is None
    assert {gap.field for gap in culture.provenance_gaps} == {"vessel", "shaking_rpm"}


def test_a_tolerance_gene_is_an_extra_copy_of_a_native_gene() -> None:
    """The case the leaf names explicitly: a real locus tag, not a heterologous symbol."""
    perturbation = foo.overexpression_perturbation("metR", "b3828", "pBbS5k-metR")
    assert perturbation.systematic_gene_name == "b3828"
    assert perturbation.perturbed_gene_name == "metR"
    assert perturbation.gene_namespace == foo.NAMESPACE
    assert perturbation.source_organism == foo.HOST_SPECIES
    assert perturbation.is_heterologous is False
    assert perturbation.pathway_name == foo.PATHWAY_TOLERANCE
    assert perturbation.localization == "episomal_plasmid"
    assert perturbation.construct_name == "pBbS5k-metR"
    assert perturbation.promoter_name == "lacUV5"
    assert perturbation.copy_number == 1.0


def test_the_rfp_control_carries_an_unreported_source_organism() -> None:
    """The reporter's origin is never stated, so it is the explicit sentinel."""
    perturbation = foo.overexpression_perturbation("rfp", "rfp", "pBbS5k-rfp")
    assert perturbation.source_organism == foo.SOURCE_ORGANISM_UNREPORTED
    assert perturbation.is_heterologous is True
    assert perturbation.pathway_name == foo.PATHWAY_REPORTER


def test_the_genotype_is_one_perturbation_on_the_shared_chassis() -> None:
    """What distinguishes a record is the single pBbS5k-borne gene."""
    genotype = foo.genotype("fpr", "b3924", "pBbS5k-fpr")
    assert len(genotype.perturbations) == 1
    assert genotype.perturbations[0].systematic_gene_name == "b3924"


def test_the_chassis_background_carries_the_plasmids_and_no_invented_alleles() -> None:
    """DH1's K-12 marker genotype is never written, so ``alleles`` is empty."""
    background = foo.chassis_background(CHASSIS)
    assert background.name == foo.CHASSIS_STRAIN
    assert background.reference_strain == foo.REFERENCE_STRAIN
    assert background.genotype_statement == CHASSIS
    assert background.alleles == []
    assert background.parents == ["E. coli DH1 (ATCC 33849)"]
    assert "pBbA5c-MevTsa-PMK-MK" in str(background.construction)
    assert "pTrc99A-NudB-PMD" in str(background.construction)
    assert background.provenance is not None
    assert len(background.provenance) == 3


def test_the_publication_is_this_paper_by_doi() -> None:
    """The mirror's manifest records no PubMed id, so the DOI is the key."""
    publication = foo.publication()
    assert publication.doi == foo.DOI
    assert publication.doi_url == f"https://doi.org/{foo.DOI}"


def test_the_build_accounting_refuses_a_ledger_that_does_not_add_up() -> None:
    """Kept plus dropped is the source rows, or the build is not accounted for."""
    accounting = foo.BuildAccounting(
        dataset="probe",
        source_rows=9,
        kept_records=8,
        dropped_records=0,
        paper_quotes_checked=28,
        notes=[],
    )
    with pytest.raises(RuntimeError, match="is not the 9 source rows"):
        accounting.check()
    accounting.kept_records = 9
    accounting.check()


def test_the_declines_are_stated_in_the_build_accounting_strings() -> None:
    """Each decline names what was read and why it is not a record."""
    assert "GSE53138" in foo.GEO_NOT_A_RECORD
    assert "No raw reads are read" in foo.GEO_NOT_A_RECORD
    assert "Table S2" in foo.TOLERANCE_SCREEN_NOT_A_RECORD
    assert "source_organism" in foo.PRODUCTION_PATHWAY_NOT_TYPED
    assert "chloramphenicol" in foo.ANTIBIOTICS_NOT_TYPED


# --------------------------------------------------------------------------- #
# The raw mirror
# --------------------------------------------------------------------------- #
def test_the_mirror_paths_hang_off_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both mirrors live under ``DATA_ROOT``, read from the environment."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert foo.raw_mirror_dir() == tmp_path / "torchcell-raw" / foo.CITATION_KEY
    assert foo.library_mirror_dir() == tmp_path / "torchcell-library" / foo.CITATION_KEY
    assert foo.pmc_cloud_key() == f"{foo.PMC_PREFIX}/{foo.SI_TABLE3_SOURCE_FILENAME}"


def test_the_deposit_writes_exactly_the_file_the_loader_consumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One deposited file, its retrieval record, and the declines beside it."""
    source = write_table_s3(tmp_path / "source" / "si9.docx")
    monkeypatch.setattr(foo, "SI_TABLE3_SHA256", _sha256(source))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "root"))
    root = foo.deposit_raw_mirror(table_s3_path=source)
    assert sorted(p.name for p in root.rglob("*") if p.is_file()) == [
        "manifest.json",
        "si9.docx",
    ]
    manifest = foo.load_manifest()
    assert [record.path for record in manifest.files] == [foo.SI_TABLE3_REL]
    record = manifest.files[0]
    assert record.role == ROLE_SI_DATA
    assert record.sha256 == _sha256(source)
    assert record.original_filename == foo.SI_TABLE3_SOURCE_FILENAME
    assert record.retrieval is not None
    assert record.retrieval.method == RetrievalMethod.pmc_cloud
    assert foo.SI_TABLE3_SOURCE_FILENAME in str(record.retrieval.source_url)
    assert record.retrieval.retrieved_at == foo.SI_RETRIEVED_AT
    assert manifest.provenance_complete is True
    assert any("NEVER released" in entry for entry in manifest.si_expected)


def test_the_deposit_is_idempotent_by_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A matching file is left alone, so a re-run is not a re-copy."""
    source = write_table_s3(tmp_path / "source" / "si9.docx")
    monkeypatch.setattr(foo, "SI_TABLE3_SHA256", _sha256(source))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "root"))
    root = foo.deposit_raw_mirror(table_s3_path=source)
    deposited = root / foo.SI_TABLE3_REL
    stamp = deposited.stat().st_mtime_ns
    foo.deposit_raw_mirror(table_s3_path=source)
    assert deposited.stat().st_mtime_ns == stamp


def test_a_deposited_file_with_a_different_digest_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upstream drift creates a new provenance record; it never overwrites one."""
    source = write_table_s3(tmp_path / "source" / "si9.docx")
    monkeypatch.setattr(foo, "SI_TABLE3_SHA256", _sha256(source))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "root"))
    root = foo.deposit_raw_mirror(table_s3_path=source)
    (root / foo.SI_TABLE3_REL).write_bytes(b"not the deposited document")
    with pytest.raises(RuntimeError, match="exists with a different sha256"):
        foo.deposit_raw_mirror(table_s3_path=source)


def test_a_source_that_does_not_match_its_pin_is_refused_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The source is verified first, so a refusal leaves no partial deposit."""
    source = write_table_s3(tmp_path / "source" / "si9.docx")
    monkeypatch.setattr(foo, "SI_TABLE3_SHA256", "0" * 64)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "root"))
    with pytest.raises(Exception):
        foo.deposit_raw_mirror(table_s3_path=source)
    assert not (tmp_path / "root" / foo.RAW_DIR_REL).exists()


# --------------------------------------------------------------------------- #
# The cross-source assertion over the two places in the mirror
# --------------------------------------------------------------------------- #
def _assert_agrees(
    strains: tuple[tuple[str, str], ...] = PRODUCTION_STRAIN_ROWS,
    composition: str = CHASSIS,
    plasmids: dict[str, str] | None = None,
) -> None:
    foo.IsopentenolTiterFoo2014Dataset._assert_table_s3_agrees_with_table1(
        [
            foo.ProductionStrain(
                strain=strain,
                composition=f"{CHASSIS} + pBbS5k-{gene}",
                plasmid=f"pBbS5k-{gene}",
                gene=gene,
            )
            for strain, gene in strains
        ],
        composition,
        dict(PRODUCTION_PLASMID_ROWS) if plasmids is None else plasmids,
    )


def test_the_two_places_in_the_mirror_name_one_panel() -> None:
    """Nine strains, Table 1's eight genes plus the control, and the stated chassis."""
    _assert_agrees()
    assert sorted(foo.STRAIN_TO_GENE) == sorted(
        strain for strain, _ in PRODUCTION_STRAIN_ROWS
    )


def test_a_missing_production_strain_is_refused() -> None:
    """Table S3 must list exactly the nine strains the gene map names."""
    with pytest.raises(RuntimeError, match="lists production strains"):
        _assert_agrees(strains=PRODUCTION_STRAIN_ROWS[:-1])


def test_a_strain_paired_with_another_genes_plasmid_is_refused() -> None:
    """The pairing is what makes a titer a titer OF a genotype."""
    swapped = (("PS+SoxS", "metR"), ("PS+MetR", "soxS")) + tuple(
        row for row in PRODUCTION_STRAIN_ROWS if row[0] not in ("PS+SoxS", "PS+MetR")
    )
    with pytest.raises(RuntimeError, match="which is not the 'soxS' plasmid"):
        _assert_agrees(strains=swapped)


def test_a_gene_set_that_is_not_table_1s_eight_plus_the_control_is_refused() -> None:
    """The released panel and the article's table are one experiment or neither."""
    renamed = PRODUCTION_STRAIN_ROWS[:-1] + (("PS+RFP", "gfp"),)
    with pytest.raises(RuntimeError, match="pBbS5k genes are"):
        _assert_agrees(strains=renamed)


def test_a_chassis_row_that_is_not_the_stated_pair_is_refused() -> None:
    """The Methods and Table S3 state the same two production plasmids."""
    with pytest.raises(RuntimeError, match="PS row reads"):
        _assert_agrees(composition="pJBEI-6830")


def test_a_composition_that_is_not_the_quoted_one_is_refused() -> None:
    """The quote on CHASSIS_COMPOSITION is checked against the deposited bytes."""
    plasmids = dict(PRODUCTION_PLASMID_ROWS) | {"pJBEI-6833": "pTrc99A-NudB"}
    with pytest.raises(RuntimeError, match="composes the production plasmids as"):
        _assert_agrees(plasmids=plasmids)


# --------------------------------------------------------------------------- #
# Hermetic end to end: a synthetic MG1655 assembly, a synthetic Table S3 and a
# synthetic OCR mirror
# --------------------------------------------------------------------------- #
#: The MG1655 locus of each overexpressed gene, as the deposited annotation types it:
#: seven matched on the gene symbol, ``gidB`` on a synonym of its current name ``rsmG``.
LOCUS_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("b0449", "mdlB", ()),
    ("b2673", "nrdH", ()),
    ("b3011", "yqhD", ()),
    ("b3687", "ibpA", ()),
    ("b3740", "rsmG", ("gidB",)),
    ("b3828", "metR", ()),
    ("b3924", "fpr", ()),
    ("b4062", "soxS", ()),
)
ASSEMBLY_REPORT = """# Assembly name:  ASM584v2
# Organism name:  Escherichia coli str. K-12 substr. MG1655 (E. coli)
# Infraspecific name:  strain=K-12 substr. MG1655
# Taxid:          511145
# GenBank assembly accession: GCA_000005845.2
# RefSeq assembly accession: GCF_000005845.2
# RefSeq assembly and GenBank assemblies identical: yes
#
## Assembly-Units:
"""
ASSEMBLY_REPORT_MEMBER = "GCA_000005845.2_ASM584v2_assembly_report.txt"


def _synthetic_loci() -> list[SyntheticLocus]:
    """One locus per :data:`LOCUS_SPECS` entry, laid end to end."""
    loci: list[SyntheticLocus] = []
    cursor = 1
    for index, (tag, symbol, synonyms) in enumerate(LOCUS_SPECS):
        start, end = cursor, cursor + 11
        cursor = end + 3
        loci.append(
            SyntheticLocus(
                tag=tag,
                parts=((start, end),),
                strand="+" if index % 2 == 0 else "-",
                symbol=symbol,
                synonyms=synonyms,
                product=f"synthetic product {tag}",
                protein_id=f"AAC{index:05d}.1",
                protein="MKV",
            )
        )
    return loci


@pytest.fixture
def synthetic_mg1655(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The real MG1655 class over a synthetic assembly, with the network refused."""
    monkeypatch.setattr(fixtures, "SEQUENCE", fixtures.SEQUENCE * 3)
    files = fixtures.write_assembly(
        tmp_path / "tier",
        MG1655_ASSEMBLY,
        _synthetic_loci(),
        [fixtures.gaf_row("metR", "metR|b3828", "GO:0000001")],
    )
    fixtures.forbid_network(monkeypatch)
    fixtures.serve_tier(monkeypatch, files)
    report = tmp_path / "tier" / ASSEMBLY_REPORT_MEMBER
    report.write_text(ASSEMBLY_REPORT)

    def serve_report(assembly_set: str, filename: str, **_: Any) -> str:
        if filename != ASSEMBLY_REPORT_MEMBER:
            raise FileNotFoundError(f"{assembly_set}/{filename} is not in the fixture")
        return str(report)

    import torchcell.datasets.bacteria_common as bacteria_common

    monkeypatch.setattr(bacteria_common, "resolve", serve_report)
    root = tmp_path / "mg1655"
    root.mkdir()
    return EcoliK12MG1655Genome(genome_root=str(root), overwrite=False)


def test_the_assembly_pin_is_the_genbank_mg1655_assembly(synthetic_mg1655: Any) -> None:
    """Every record is an edit of PS on the pinned assembly."""
    reference = foo.chassis_reference(CHASSIS)
    assert reference.species == "Escherichia coli"
    assert reference.strain == foo.CHASSIS_STRAIN
    assert reference.assembly_set == "ecoli_K12_MG1655_ASM584v2"
    assert reference.assembly_accession == "GCA_000005845.2"
    dumped = ProductTiterExperimentReference(
        dataset_name="probe",
        genome_reference=reference,
        environment_reference=foo.environment(),
        phenotype_reference=foo.titer_phenotype(834.0),
    ).model_dump()
    assert dumped["genome_reference"]["background"]["name"] == foo.CHASSIS_STRAIN
    assert dumped["phenotype_reference"]["titer"] == 834.0


@pytest.fixture
def built_store(
    synthetic_mg1655: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> str:
    """Deposit a synthetic mirror and build the loader end to end under ``tmp_path``."""
    data_root = tmp_path / "root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    source = write_table_s3(tmp_path / "source" / "si9.docx")
    monkeypatch.setattr(foo, "SI_TABLE3_SHA256", _sha256(source))
    foo.deposit_raw_mirror(table_s3_path=source)
    _write_library(data_root, _synthetic_paper_md(), monkeypatch)
    root = str(data_root / foo.DEV_TREE_RELPATH)
    dataset = foo.IsopentenolTiterFoo2014Dataset(
        root=root, ecoli_genome=synthetic_mg1655
    )
    assert len(dataset) == foo.EXPECTED_RECORDS
    dataset.close_lmdb()
    return root


def test_the_built_store_holds_one_record_per_released_titer(built_store: str) -> None:
    """Nine records: Table 1's eight and the control of its footnote b."""
    from torchcell.verification.runners import load_records

    records = load_records(built_store)
    assert len(records) == 9
    ledger = pd.read_csv(osp.join(built_store, "preprocess", "titer_rows.csv"))
    assert dict(ledger["source"].value_counts()) == {
        "Table 1": 8,
        "Table 1 footnote b": 1,
    }
    assert set(ledger["uncertainty_type"]) == {"not_released"}
    assert sorted(ledger["locus_tag"]) == sorted(
        [tag for tag, _, _ in LOCUS_SPECS] + ["rfp"]
    )
    for record in records:
        experiment = ProductTiterExperiment.model_validate(record["experiment"])
        assert experiment.phenotype.titer_unit == ConcentrationUnit.ug_per_ml
        genotype = experiment.genotype
        assert isinstance(genotype, Genotype)
        assert len(genotype.perturbations) == 1
        reference = ProductTiterExperimentReference.model_validate(record["reference"])
        assert reference.phenotype_reference.titer == 834.0


def test_the_build_accounting_declines_nothing_numeric(built_store: str) -> None:
    """Every released isopentenol titer in the mirror is a record."""
    accounting = json.loads(
        Path(osp.join(built_store, "preprocess", "build_accounting.json")).read_text()
    )
    assert accounting["dataset"] == "IsopentenolTiterFoo2014Dataset"
    assert accounting["source_rows"] == 9
    assert accounting["kept_records"] == 9
    assert accounting["dropped_records"] == 0
    assert accounting["paper_quotes_checked"] == 28
    assert any("GSE53138" in note for note in accounting["notes"])


def test_the_built_store_passes_l0_to_l4(built_store: str) -> None:
    """The module's own runner, over the store it just built."""
    report = foo.verify_build(built_store, os.environ["DATA_ROOT"])
    assert report.passed, report.summary()
    assert {result.level.name for result in report.results} == {
        "L0",
        "L1",
        "L2",
        "L3",
        "L4",
    }
    names = [result.name for result in report.results]
    assert "titer_against_released_improvement_percent" in names
    assert "construct_against_deposited_table_s3" in names
    written = json.loads(
        Path(
            osp.join(built_store, "preprocess", "verification_report.json")
        ).read_text()
    )
    assert written["dataset_name"] == "IsopentenolTiterFoo2014Dataset"


def test_the_module_entry_point_builds_and_verifies(
    built_store: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``python -m torchcell.datasets.ecoli.foo2014`` over the store it just built."""
    monkeypatch.setattr(foo, "load_manifest", foo.load_manifest)
    foo.main()
    out = capsys.readouterr().out
    assert f"len = {foo.EXPECTED_RECORDS}" in out
    assert "PASS" in out


def test_the_injected_genome_wins_and_the_tier_is_the_fallback(
    synthetic_mg1655: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The MG1655 genome is injected by the build, or opened from the genomes tier."""
    opened: list[tuple[str, str]] = []

    def record(host: str, strain: str) -> Any:
        opened.append((host, strain))
        return synthetic_mg1655

    monkeypatch.setattr(foo, "bacterial_genome", record)
    shell = foo.IsopentenolTiterFoo2014Dataset.__new__(
        foo.IsopentenolTiterFoo2014Dataset
    )
    shell.ecoli_genome = None
    assert shell._genome() is synthetic_mg1655
    assert opened == [("ecoli", foo.REFERENCE_STRAIN)]
    assert shell._genome() is synthetic_mg1655
    assert len(opened) == 1


def test_the_loader_declares_its_schema_classes_and_its_one_raw_file() -> None:
    """The class-level contract the registry sweep reads."""
    shell = foo.IsopentenolTiterFoo2014Dataset.__new__(
        foo.IsopentenolTiterFoo2014Dataset
    )
    assert shell.experiment_class is ProductTiterExperiment
    assert shell.reference_class is ProductTiterExperimentReference
    assert shell.raw_file_names == [foo.SI_TABLE3_FILENAME]
    assert shell.preprocess_raw("df") == "df"
    with pytest.raises(NotImplementedError):
        shell.create_experiment()


# --------------------------------------------------------------------------- #
# The pinned mirrors and the built dev store
# --------------------------------------------------------------------------- #
@pytest.mark.data
@requires_mirror
def test_the_raw_mirror_records_the_digest_the_module_pins() -> None:
    """The deposited bytes, their recorded retrieval and the module constant agree."""
    assert _MIRROR is not None
    manifest = foo.load_manifest(DATA_ROOT)
    record = next(r for r in manifest.files if r.path == foo.SI_TABLE3_REL)
    assert record.sha256 == foo.SI_TABLE3_SHA256
    assert _sha256(_MIRROR / foo.SI_TABLE3_REL) == foo.SI_TABLE3_SHA256
    assert record.retrieval is not None
    assert record.retrieval.method == RetrievalMethod.pmc_cloud


@pytest.mark.data
@requires_library
def test_every_sourced_quote_is_verbatim_in_the_pinned_ocr() -> None:
    """The audit against the real mirror, which is what the build runs."""
    assert _LIBRARY is not None
    assert _sha256(_LIBRARY / foo.PAPER_MD) == foo.PAPER_MD_SHA256
    assert foo.verify_paper_quotes(DATA_ROOT) == 28


@pytest.mark.data
@requires_mirror
def test_the_deposited_table_s3_reads_as_the_nine_released_strains() -> None:
    """The real deposited document, through the same reader the build uses."""
    assert _MIRROR is not None
    path = _MIRROR / foo.SI_TABLE3_REL
    assert [s.strain for s in foo.read_production_strains(path)] == [
        strain for strain, _ in PRODUCTION_STRAIN_ROWS
    ]
    assert foo.read_production_plasmids(path) == dict(PRODUCTION_PLASMID_ROWS)
    assert foo.chassis_row(path)[1] == str(foo.CHASSIS_PLASMIDS.value)


@pytest.mark.data
@requires_store
def test_the_built_dev_store_holds_nine_records_and_passes_l0_to_l4() -> None:
    """The real store, verified by the module's own runner."""
    from torchcell.verification.runners import load_records

    assert _STORE is not None
    records = load_records(str(_STORE))
    assert len(records) == 9
    titers = sorted(record["experiment"]["phenotype"]["titer"] for record in records)
    assert titers == [834.0, 838.0, 860.0, 931.0, 965.0, 967.0, 989.0, 994.0, 1290.0]
    report = foo.verify_build(str(_STORE), DATA_ROOT)
    assert report.passed, report.summary()
