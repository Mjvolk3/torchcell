# tests/torchcell/datasets/pputida/test_yunus2026.py
# [[tests.torchcell.datasets.pputida.test_yunus2026]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/pputida/test_yunus2026.py
"""The Yunus 2026 P. putida CRISPRi knockdown loaders.

Synthetic tests run everywhere, with no network and no ``$DATA_ROOT``: a minimal
``.docx`` is written in-process (``read_docx_tables`` reads only
``word/document.xml``, so an OPC skeleton is not needed), the real KT2440 genome class
is built over a synthetic assembly, and both loaders are built end to end under
``tmp_path``. They cover the docx reader and its structure assertions, the Table S3 and
Table S8-S12 parsers, the construct round-trip, the Table S7 guide library and its
reverse-complement check, the guide-assignment rule and all four of its outcomes, the
relative-expression phenotypes, the deferred chassis background, the production
environment, the retention arithmetic, and the reference-denominator L4 rule.

The ``@pytest.mark.data`` tests read the real ``$DATA_ROOT``: they pin the raw mirror's
digests against the module constants and assert the counts measured on the pinned
``mmc1.docx`` (sha256 ``daa2c91d...``): 14 tables, 125 Table S3 rows with 23 ``n.d.``,
102 records, 204 Table S7 oligo pairs all reverse-complement-consistent, 25 array
constructs over 153 released cells, and 102 of 102 screened targets resolving to KT2440
locus tags. They are skipped unless the mirror and the KT2440 tier cache are present.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import os.path as osp
import zipfile
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import torchcell.datasets.pputida.yunus2026 as y26
from torchcell.datamodels.media import M9
from torchcell.datamodels.schema import (
    BACTERIAL_ASSEMBLY_SETS,
    ConcentrationUnit,
    EnvironmentPhysicalPerturbation,
    PhysicalFactor,
    ProteinAbundancePhenotype,
    SmallMoleculePerturbation,
)
from torchcell.literature.manifest import ROLE_RAW_DATA, Manifest, RetrievalMethod
from torchcell.verification.report import Level

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


# --------------------------------------------------------------------------- #
# A minimal .docx the module's own reader consumes
# --------------------------------------------------------------------------- #
def _xml_escape(text: str) -> str:
    """Escape the three characters that matter inside an XML text node."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _cell_xml(text: str) -> str:
    """One ``w:tc`` holding a single paragraph of ``text``."""
    return f"<w:tc><w:p><w:r><w:t>{_xml_escape(text)}</w:t></w:r></w:p></w:tc>"


def _table_xml(rows: list[list[str]]) -> str:
    """One ``w:tbl`` from a list of rows of cell strings."""
    body = "".join(
        "<w:tr>" + "".join(_cell_xml(cell) for cell in row) + "</w:tr>" for row in rows
    )
    return f"<w:tbl>{body}</w:tbl>"


def write_docx(path: Path, tables: list[list[list[str]]]) -> Path:
    """Write a zip whose ``word/document.xml`` holds ``tables`` in document order."""
    document = (
        f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{W}"><w:body>'
        + "".join(_table_xml(rows) for rows in tables)
        + "</w:body></w:document>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)
    return path


def _sha256_bytes(path: Path) -> str:
    """sha256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# The synthetic supplementary file
# --------------------------------------------------------------------------- #
#: The screened loci of the synthetic Table S3, with the symbol the assembly gives them.
SCREEN_LOCI: tuple[tuple[str, str | None], ...] = (
    ("PP_0100", None),
    ("PP_0101", None),
    ("PP_0102", None),
    ("PP_0103", None),
    ("PP_0104", None),
    ("PP_0105", None),
    ("PP_0106", None),
    ("PP_0107", None),
    ("PP_0108", None),
    ("PP_0109", "accA"),
    ("PP_0110", None),
    ("PP_0111", None),
)
#: Loci only the array panels touch.
ARRAY_LOCI: tuple[tuple[str, str | None], ...] = (
    ("PP_0200", None),
    ("PP_0201", None),
    ("PP_0202", None),
)
ALL_LOCI = SCREEN_LOCI + ARRAY_LOCI

#: Table S3 rows the synthetic file carries: (strain, target, value-or-"n.d.").
#: Covers an exact-variant hit, a tag with no oligo, a tag with two distinct spacers,
#: a symbol-labeled oligo, a zero value and three ``n.d.`` rows.
SCREEN_ROWS: tuple[tuple[str, str, str], ...] = (
    ("IY001", "PP_0100", "0.25"),
    ("IY002", "PP_0101", "0.0"),
    ("IY003", "PP_0102", "1.5"),
    ("IY004", "PP_0103", "n.d."),
    ("IY005", "PP_0104", "0.4"),
    ("IY006", "PP_0105", "n.d."),
    ("IY007", "PP_0106_NT2", "0.1"),
    ("IY008", "PP_0106_NT3", "0.2"),
    ("IY009", "PP_0107", "0.3"),
    ("IY010", "PP_0108", "n.d."),
    ("IY011", "PP_0109", "0.5"),
    ("IY012", "PP_0110", "0.6"),
    ("IY013", "PP_0111", "0.7"),
)
SCREEN_ND = sum(1 for _, _, value in SCREEN_ROWS if value == y26.NOT_DETECTED)

#: ``(label, spacer)`` of the synthetic Table S7 sgRNA oligos.
OLIGO_SPECS: tuple[tuple[str, str], ...] = (
    ("PP0100_sgRNA", "ACGTACGTACGTACGTACGTAC"),
    ("PP0101_NT1_sgRNA", "TTGGCCAATTGGCCAATTGGCC"),
    ("PP0102_sgRNA_NT1", "GGCCTTAAGGCCTTAAGGCCTT"),
    ("PP0103_sgRNA", "CCGGAATTCCGGAATTCCGGAA"),
    ("PP0104_sgRNA", "AATTCCGGAATTCCGGAATTCC"),
    ("PP0105_sgRNA", "TTAACCGGTTAACCGGTTAACC"),
    ("PP0106_NT2_sgRNA", "GGTTCCAAGGTTCCAAGGTTCC"),
    ("PP0106_NT3_sgRNA", "CCAAGGTTCCAAGGTTCCAAGG"),
    # two distinct spacers for one locus, with no variant to separate them
    ("PP0107_NT1_sgRNA", "ACACACACACACACACACACAC"),
    ("PP0107_NT2_sgRNA", "GTGTGTGTGTGTGTGTGTGTGT"),
    # a symbol-labeled oligo the annotation resolves
    ("accA_NT1_sgRNA", "ATATATATATATATATATATAT"),
    # a 21 nt spacer, against the Methods' stated 22
    ("PP0110_sgRNA", "ACGTACGTACGTACGTACGTA"),
    # a label that names no gene of the assembly, documented in NON_GENE_OLIGO_LABELS
    ("RFP_NT1_sgRNA", "TGCATGCATGCATGCATGCATG"),
    ("PP0200_sgRNA", "CATCATCATCATCATCATCATC"),
    ("PP0201_sgRNA", "GATGATGATGATGATGATGATG"),
    ("PP0202_sgRNA", "TACTACTACTACTACTACTACT"),
)
#: ``PP_0111`` deliberately has no oligo, and ``PP_0108`` / ``PP_0109`` reach their
#: spacer only through the symbol layer or not at all.
ARRAY_PANEL_ROWS: dict[str, tuple[tuple[str, float], ...]] = {
    "PP_0200": (("PP_0200", 0.2), ("PP_0200_0201", 0.3), ("PP_0200_0201_0202", 0.4)),
    "PP_0201": (("PP_0200_0201", 0.1), ("PP_0200_0201_0202", 0.15)),
    "PP_0202": (("PP_0200_0201_0202", 0.05),),
    "PP_0100": (("PP_0200_0201_0202", 0.9),),
    "PP_0101": (("PP_0200", 0.8),),
}

ASSEMBLY_REPORT = """# Assembly name:  ASM756v2
# Organism name:  Pseudomonas putida KT2440 (g-proteobacteria)
# Infraspecific name:  strain=KT2440
# Taxid:          160488
# GenBank assembly accession: GCA_000007565.2
# RefSeq assembly accession: GCF_000007565.2
# RefSeq assembly and GenBank assemblies identical: yes
#
## Assembly-Units:
AE015451.2\tassembled-molecule\tna\tChromosome\tAE015451.2\t=\tNC_002947.4
"""
ASSEMBLY_REPORT_MEMBER = "GCA_000007565.2_ASM756v2_assembly_report.txt"


def _oligo_rows() -> list[list[str]]:
    """The synthetic Table S7: a forward/reverse pair per spec, plus one primer."""
    rows = [list(y26.TABLE_S7_HEADER)]
    for index, (label, spacer) in enumerate(OLIGO_SPECS):
        forward = y26.OLIGO_FORWARD_PREFIX + spacer + y26.OLIGO_FORWARD_SUFFIX
        reverse = (
            y26.OLIGO_REVERSE_PREFIX
            + y26.reverse_complement(spacer)
            + y26.OLIGO_REVERSE_SUFFIX
        )
        rows.append([f"IY{index:04d}_{label}_F", forward])
        rows.append([f"IY{index:04d}_{label}_R", reverse])
    rows.append(["IY77_dCas9_F", "cgaatcttggagctcccgctg"])
    return rows


def _target_list_rows(header: tuple[str, ...], tags: list[str]) -> list[list[str]]:
    """A synthetic Table S1 or S2 over ``tags``."""
    rows = [list(header)]
    for number, tag in enumerate(tags, start=1):
        rows.append(
            [str(number), tag, "Enz", "a synthetic enzyme", "K00001", "TCA", ""]
        )
    return rows


def _array_rows(protein: str) -> list[list[str]]:
    """One synthetic Fig. 3J-N panel over three replicates."""
    rows = [["", "Replicate", f"Relative expression level of {protein}"]]
    for construct, base in ARRAY_PANEL_ROWS[protein]:
        for offset, replicate in enumerate(y26.ARRAY_REPLICATES):
            rows.append([construct, replicate, f"{base + offset * 0.01:.4f}"])
    return rows


def synthetic_tables() -> list[list[list[str]]]:
    """The 14 tables of the synthetic supplementary file, in document order."""
    screen = [list(y26.TABLE_S3_HEADER)] + [list(row) for row in SCREEN_ROWS]
    tables: list[list[list[str]]] = [
        _target_list_rows(y26.TABLE_S1_HEADER, [tag for tag, _ in SCREEN_LOCI[:6]]),
        _target_list_rows(y26.TABLE_S2_HEADER, [tag for tag, _ in SCREEN_LOCI[6:]]),
        screen,
        [["Protein", "Fold Change"], ["Rbsb", "0.003"]],
        [["Protein", "Fold Change"], ["Pp_2686", "459.0"]],
        [["Plasmid No.", "Plasmid description"], ["pIY989", "pRSF1010-Gm-dCas9"]],
        _oligo_rows(),
    ]
    tables.extend(_array_rows(protein) for _, protein in ARRAY_PANELS_SYNTHETIC)
    tables.append([["Step"], ["hybridize"]])
    tables.append([["Step"], ["digest"]])
    return tables


#: The synthetic panels, in the order the module expects S8..S12.
ARRAY_PANELS_SYNTHETIC: tuple[tuple[str, str], ...] = (
    ("S8", "PP_0200"),
    ("S9", "PP_0201"),
    ("S10", "PP_0202"),
    ("S11", "PP_0100"),
    ("S12", "PP_0101"),
)


def _synthetic_loci() -> list[Any]:
    """One ``SyntheticLocus`` per :data:`ALL_LOCI` entry, laid end to end."""
    from tests.torchcell.sequence.genome._bacterial_fixtures import SyntheticLocus

    loci = []
    cursor = 1
    for index, (tag, symbol) in enumerate(ALL_LOCI):
        start, end = cursor, cursor + 11
        cursor = end + 3
        loci.append(
            SyntheticLocus(
                tag=tag,
                parts=((start, end),),
                strand="+" if index % 2 == 0 else "-",
                symbol=symbol,
                product=f"synthetic product {tag}",
                protein_id=f"AAN{index:05d}.1",
                protein="MKV",
            )
        )
    return loci


def _write_feature_table(path: Path, loci: list[Any]) -> None:
    """A minimal NCBI feature table: one ``gene`` row and one ``CDS`` row per locus.

    ``torchcell.verification.runners._bacterial_gene_set`` builds the L4 locus-tag
    universe from this member and cross-checks it against the protein FASTA, so the
    synthetic assembly needs both for the containment rule to run at all.
    """
    import gzip

    columns = (
        "feature",
        "class",
        "assembly",
        "assembly_unit",
        "seq_type",
        "chromosome",
        "genomic_accession",
        "start",
        "end",
        "strand",
        "product_accession",
        "non-redundant_refseq",
        "related_accession",
        "name",
        "symbol",
        "GeneID",
        "locus_tag",
    )
    lines = ["# " + "\t".join(columns)]
    for locus in loci:
        for feature, accession in (("gene", ""), ("CDS", locus.protein_id or "")):
            row = [""] * len(columns)
            row[columns.index("feature")] = feature
            row[columns.index("product_accession")] = accession
            row[columns.index("locus_tag")] = locus.tag
            lines.append("\t".join(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        handle.write("\n".join(lines) + "\n")


@pytest.fixture
def synthetic_kt2440(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The real KT2440 class over a synthetic assembly, with the network refused."""
    import tests.torchcell.sequence.genome._bacterial_fixtures as fixtures
    from torchcell.sequence.genome.pputida.kt2440 import (
        KT2440_ASSEMBLY,
        PPutidaKT2440Genome,
    )

    monkeypatch.setattr(fixtures, "SEQUENCE", fixtures.SEQUENCE * 6)
    files = fixtures.write_assembly(
        tmp_path / "tier",
        KT2440_ASSEMBLY,
        _synthetic_loci(),
        [fixtures.gaf_row("accA", "accA|PP_0109", "GO:0000001")],
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
    import torchcell.verification.runners as runners

    feature_table = (
        tmp_path / "tier" / (f"{KT2440_ASSEMBLY.genbank_assembly}_feature_table.txt.gz")
    )
    _write_feature_table(feature_table, _synthetic_loci())
    files[(KT2440_ASSEMBLY.assembly_set, feature_table.name)] = feature_table

    def serve_tier_member(assembly_set: str, filename: str, **_: Any) -> str:
        """The L4 gene universe is read through ``runners.resolve``, not the genome's."""
        if filename == ASSEMBLY_REPORT_MEMBER:
            return str(report)
        key = (assembly_set, filename)
        if key not in files:
            raise FileNotFoundError(f"{assembly_set}/{filename} is not in the fixture")
        return str(files[key])

    monkeypatch.setattr(bacteria_common, "resolve", serve_report)
    monkeypatch.setattr(runners, "resolve", serve_tier_member)
    root = tmp_path / "kt2440"
    root.mkdir()
    return PPutidaKT2440Genome(genome_root=str(root), overwrite=False)


@pytest.fixture
def synthetic_docx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The synthetic supplementary file, with its own digests pinned over the real ones."""
    path = write_docx(tmp_path / "si" / y26.SI_DOCX, synthetic_tables())
    monkeypatch.setattr(y26, "SI_DOCX_SHA256", _sha256_bytes(path))
    monkeypatch.setattr(y26, "DATA_SHA256", {y26.SI_DOCX: _sha256_bytes(path)})
    monkeypatch.setattr(y26, "ARRAY_PANELS", ARRAY_PANELS_SYNTHETIC)
    monkeypatch.setattr(y26, "TABLE_S3_ROWS", len(SCREEN_ROWS))
    monkeypatch.setattr(y26, "SCREEN_SAMPLES", len(SCREEN_ROWS))
    monkeypatch.setattr(y26, "NOT_DETECTED_COUNT", SCREEN_ND)
    tables = y26.read_docx_tables(path)
    monkeypatch.setattr(
        y26,
        "TABLE_DIGESTS",
        {
            name: y26.table_digest(tables[y26.TABLE_INDEX[name]])
            for name in y26.TABLE_DIGESTS
        },
    )
    return path


# --------------------------------------------------------------------------- #
# Synthetic: the docx reader and the structure assertions
# --------------------------------------------------------------------------- #
def test_read_docx_tables_returns_every_table_in_document_order(
    synthetic_docx: Path,
) -> None:
    """The reader yields ``[table][row][cell]`` of stripped text, in document order."""
    tables = y26.read_docx_tables(synthetic_docx)
    assert len(tables) == y26.SI_TABLE_COUNT
    assert tuple(tables[y26.TABLE_INDEX["S3"]][0]) == y26.TABLE_S3_HEADER
    assert tables[y26.TABLE_INDEX["S3"]][1] == ["IY001", "PP_0100", "0.25"]


def test_read_docx_tables_joins_a_multi_paragraph_cell_with_one_space(
    tmp_path: Path,
) -> None:
    """A cell holding two paragraphs becomes one space-joined string."""
    document = (
        f'<w:document xmlns:w="{W}"><w:body><w:tbl><w:tr><w:tc>'
        "<w:p><w:r><w:t>first</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>second</w:t></w:r></w:p>"
        "</w:tc></w:tr></w:tbl></w:body></w:document>"
    )
    path = tmp_path / "two.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)
    assert y26.read_docx_tables(path) == [[["first second"]]]


def test_read_docx_tables_refuses_a_document_with_no_body(tmp_path: Path) -> None:
    """A document.xml with no ``w:body`` is not the file this loader was written on."""
    path = tmp_path / "nobody.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", f'<w:document xmlns:w="{W}"/>')
    with pytest.raises(y26.TableExtractionError, match="no w:body"):
        y26.read_docx_tables(path)


def test_supplementary_tables_refuses_a_changed_table_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-released file with another table count stops the build."""
    path = write_docx(tmp_path / "short.docx", synthetic_tables()[:5])
    monkeypatch.setattr(y26, "ARRAY_PANELS", ARRAY_PANELS_SYNTHETIC)
    with pytest.raises(y26.TableExtractionError, match="holds 5 tables"):
        y26.supplementary_tables(path)


def test_supplementary_tables_refuses_a_changed_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A renamed Table S3 column stops the build rather than shifting a column."""
    tables = synthetic_tables()
    tables[y26.TABLE_INDEX["S3"]][0] = ["Strain", "Target", "Ratio"]
    path = write_docx(tmp_path / "renamed.docx", tables)
    monkeypatch.setattr(y26, "ARRAY_PANELS", ARRAY_PANELS_SYNTHETIC)
    with pytest.raises(y26.TableExtractionError, match="Table S3 header"):
        y26.supplementary_tables(path)


def test_supplementary_tables_refuses_a_changed_array_panel_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A panel measuring another protein is a different experiment, not a relabel."""
    tables = synthetic_tables()
    tables[y26.TABLE_INDEX["S8"]][0] = [
        "",
        "Replicate",
        "Relative expression level of PP_9999",
    ]
    path = write_docx(tmp_path / "panel.docx", tables)
    monkeypatch.setattr(y26, "ARRAY_PANELS", ARRAY_PANELS_SYNTHETIC)
    with pytest.raises(y26.TableExtractionError, match="Table S8 header"):
        y26.supplementary_tables(path)


def test_supplementary_tables_refuses_a_drifted_parsed_digest(
    synthetic_docx: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parsed-table digest is the extraction check, and it is not advisory."""
    monkeypatch.setitem(y26.TABLE_DIGESTS, "S3", "0" * 64)
    with pytest.raises(y26.TableExtractionError, match="Table S3 parsed sha256"):
        y26.supplementary_tables(synthetic_docx)


def test_table_digest_is_order_and_content_sensitive() -> None:
    """Two different parses hash differently; the same parse hashes the same."""
    rows = [["a", "b"], ["c", "d"]]
    assert y26.table_digest(rows) == y26.table_digest([["a", "b"], ["c", "d"]])
    assert y26.table_digest(rows) != y26.table_digest([["c", "d"], ["a", "b"]])


# --------------------------------------------------------------------------- #
# Synthetic: the Table S3 parser
# --------------------------------------------------------------------------- #
def test_parse_table_s3_types_every_row_and_separates_the_not_detected_ones(
    synthetic_docx: Path,
) -> None:
    """An ``n.d.`` row carries ``None``, which is what ``not_detected`` reports."""
    rows = y26.parse_table_s3(y26.supplementary_tables(synthetic_docx)["S3"])
    assert len(rows) == len(SCREEN_ROWS)
    assert sum(1 for row in rows if row.not_detected) == SCREEN_ND
    first = rows[0]
    assert (first.strain, first.locus_tag, first.variant) == ("IY001", "PP_0100", None)
    assert first.relative_expression == pytest.approx(0.25)
    assert rows[1].relative_expression == 0.0
    assert rows[3].relative_expression is None


def test_parse_table_s3_reads_a_variant_label_as_part_of_the_target(
    synthetic_docx: Path,
) -> None:
    """``PP_0106_NT2`` is locus ``PP_0106`` with guide variant ``NT2``."""
    rows = y26.parse_table_s3(y26.supplementary_tables(synthetic_docx)["S3"])
    variants = {row.strain: (row.locus_tag, row.variant) for row in rows}
    assert variants["IY007"] == ("PP_0106", "NT2")
    assert variants["IY008"] == ("PP_0106", "NT3")


def test_parse_table_s3_refuses_a_target_that_is_not_a_locus_tag() -> None:
    """A target outside ``PP_<4 digits>[_NT<n>]`` is not a name this loader can place."""
    rows = [list(y26.TABLE_S3_HEADER), ["IY001", "sucB", "0.2"]]
    with pytest.raises(y26.TableExtractionError, match="is not PP_"):
        y26.parse_table_s3(rows)


def test_parse_table_s3_refuses_a_row_count_the_results_do_not_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One row per sample is the reading n_replicates = 1 rests on, so it is checked."""
    monkeypatch.setattr(y26, "TABLE_S3_ROWS", 99)
    rows = [list(y26.TABLE_S3_HEADER), ["IY001", "PP_0100", "0.2"]]
    with pytest.raises(y26.TableExtractionError, match="holds 1 rows"):
        y26.parse_table_s3(rows)


def test_parse_table_s3_refuses_a_repeated_strain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two rows for one strain would make the per-strain record ambiguous."""
    monkeypatch.setattr(y26, "TABLE_S3_ROWS", 2)
    monkeypatch.setattr(y26, "NOT_DETECTED_COUNT", 0)
    rows = [
        list(y26.TABLE_S3_HEADER),
        ["IY001", "PP_0100", "0.2"],
        ["IY001", "PP_0101", "0.3"],
    ]
    with pytest.raises(y26.TableExtractionError, match="repeats a strain"):
        y26.parse_table_s3(rows)


def test_parse_table_s3_refuses_a_not_detected_count_the_results_do_not_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 23 'n.d.' rows are a sourced number, so a different count stops the build."""
    monkeypatch.setattr(y26, "TABLE_S3_ROWS", 1)
    monkeypatch.setattr(y26, "NOT_DETECTED_COUNT", 5)
    rows = [list(y26.TABLE_S3_HEADER), ["IY001", "PP_0100", "n.d."]]
    with pytest.raises(y26.TableExtractionError, match="holds 1 'n.d.' rows"):
        y26.parse_table_s3(rows)


# --------------------------------------------------------------------------- #
# Synthetic: the array construct parser
# --------------------------------------------------------------------------- #
def test_parse_construct_expands_every_abbreviated_target() -> None:
    """The released names abbreviate after the first target; every tag comes back."""
    assert y26.parse_construct("PP_4188") == ("PP_4188",)
    assert y26.parse_construct("PP_4188_0528") == ("PP_4188", "PP_0528")
    assert y26.parse_construct("PP_4188_0812_4160_0168_0528") == (
        "PP_4188",
        "PP_0812",
        "PP_4160",
        "PP_0168",
        "PP_0528",
    )


def test_parse_construct_refuses_a_name_that_does_not_round_trip() -> None:
    """A name whose digit groups do not re-serialize would silently lose a guide."""
    for name in ("PP_4188_NT1", "sucB", "PP_418", "PP4188"):
        with pytest.raises(y26.TableExtractionError, match="array construct"):
            y26.parse_construct(name)


def test_parse_array_panel_types_every_replicate_cell(synthetic_docx: Path) -> None:
    """Each cell knows its construct, that construct's guides, and its replicate."""
    tables = y26.supplementary_tables(synthetic_docx)
    cells = y26.parse_array_panel(tables["S8"], "PP_0200")
    assert len(cells) == len(ARRAY_PANEL_ROWS["PP_0200"]) * len(y26.ARRAY_REPLICATES)
    triple = [c for c in cells if c.construct_name == "PP_0200_0201_0202"]
    assert {c.replicate for c in triple} == set(y26.ARRAY_REPLICATES)
    assert triple[0].guide_targets == ("PP_0200", "PP_0201", "PP_0202")
    assert triple[0].protein == "PP_0200"


def test_parse_array_panel_refuses_an_unknown_replicate_label() -> None:
    """Only R1-R3 are released; another label means the panel design changed."""
    rows = [
        ["", "Replicate", "Relative expression level of PP_0200"],
        ["PP_0200", "R4", "0.2"],
    ]
    with pytest.raises(y26.TableExtractionError, match="replicate label"):
        y26.parse_array_panel(rows, "PP_0200")


def test_parse_array_panel_refuses_a_repeated_replicate() -> None:
    """A repeated replicate would shrink the SE, so it is refused."""
    rows = [
        ["", "Replicate", "Relative expression level of PP_0200"],
        ["PP_0200", "R1", "0.2"],
        ["PP_0200", "R1", "0.3"],
    ]
    with pytest.raises(y26.TableExtractionError, match="repeats a"):
        y26.parse_array_panel(rows, "PP_0200")


def test_parse_target_list_labels_the_selection_method(synthetic_docx: Path) -> None:
    """Table S1 is the intuition list and Table S2 the FluxRETAP list."""
    tables = y26.supplementary_tables(synthetic_docx)
    s1 = y26.parse_target_list(tables["S1"], table="S1")
    s2 = y26.parse_target_list(tables["S2"], table="S2")
    assert {row.selection for row in s1} == {"intuition"}
    assert {row.selection for row in s2} == {"fluxretap"}
    assert s1[0].locus_tag == "PP_0100"


def test_parse_target_list_refuses_a_target_that_is_not_a_locus_tag() -> None:
    """The target-list column is locus tags; anything else is a parse error."""
    rows = [list(y26.TABLE_S1_HEADER), ["1", "sucB", "", "", "", "", ""]]
    with pytest.raises(y26.TableExtractionError, match="is not a PP_ locus tag"):
        y26.parse_target_list(rows, table="S1")


# --------------------------------------------------------------------------- #
# Synthetic: the guide library
# --------------------------------------------------------------------------- #
def test_reverse_complement_round_trips() -> None:
    """The helper is an involution on unambiguous ACGT."""
    assert y26.reverse_complement("ACGTTTGA") == "TCAAACGT"
    assert y26.reverse_complement(y26.reverse_complement("ACGGTA")) == "ACGGTA"


def test_build_guide_library_reads_every_pair_and_keys_it_on_tag_and_variant(
    synthetic_docx: Path, synthetic_kt2440: Any
) -> None:
    """Each pair contributes one spacer under ``(locus tag, variant)``."""
    tables = y26.supplementary_tables(synthetic_docx)
    library = y26.build_guide_library(tables["S7"], synthetic_kt2440)
    assert library.n_pairs == len(OLIGO_SPECS)
    assert library.by_key[y26.GuideLibrary.key("PP_0106", "NT2")].spacer == (
        "GGTTCCAAGGTTCCAAGGTTCC"
    )
    assert library.by_key[y26.GuideLibrary.key("PP_0100", None)].spacer == (
        "ACGTACGTACGTACGTACGTAC"
    )


def test_build_guide_library_resolves_a_symbol_labeled_oligo(
    synthetic_docx: Path, synthetic_kt2440: Any
) -> None:
    """``accA_NT1_sgRNA`` lands on the locus the annotation gives that symbol."""
    tables = y26.supplementary_tables(synthetic_docx)
    library = y26.build_guide_library(tables["S7"], synthetic_kt2440)
    assert y26.GuideLibrary.key("PP_0109", "NT1") in library.by_key


def test_build_guide_library_records_an_off_length_spacer_without_correcting_it(
    synthetic_docx: Path, synthetic_kt2440: Any
) -> None:
    """A 21 nt spacer is stored verbatim and reported, never padded."""
    tables = y26.supplementary_tables(synthetic_docx)
    library = y26.build_guide_library(tables["S7"], synthetic_kt2440)
    assert library.n_off_length == ["PP0110_sgRNA (21 nt)"]
    assert len(library.by_key[y26.GuideLibrary.key("PP_0110", None)].spacer) == 21


def test_build_guide_library_keeps_a_documented_non_gene_label_unmapped(
    synthetic_docx: Path, synthetic_kt2440: Any
) -> None:
    """``RFP`` names no gene, so it is reported rather than remapped."""
    tables = y26.supplementary_tables(synthetic_docx)
    library = y26.build_guide_library(tables["S7"], synthetic_kt2440)
    assert library.unmapped_labels == {"RFP": ["RFP_NT1_sgRNA"]}


def test_build_guide_library_refuses_an_undocumented_unmapped_label(
    synthetic_docx: Path, synthetic_kt2440: Any
) -> None:
    """A new label that names no locus stops the build until it is documented."""
    tables = y26.supplementary_tables(synthetic_docx)
    rows = [
        *tables["S7"],
        [
            "IY900_mystery_F",
            y26.OLIGO_FORWARD_PREFIX + "A" * 22 + y26.OLIGO_FORWARD_SUFFIX,
        ],
        [
            "IY900_mystery_R",
            y26.OLIGO_REVERSE_PREFIX + "T" * 22 + y26.OLIGO_REVERSE_SUFFIX,
        ],
    ]
    with pytest.raises(y26.TableExtractionError, match="not documented in"):
        y26.build_guide_library(rows, synthetic_kt2440)


def test_build_guide_library_refuses_a_pair_without_the_basic_flanks(
    synthetic_kt2440: Any,
) -> None:
    """A spacer is read by stripping known flanks; an unknown flank is not guessed at."""
    rows = [
        list(y26.TABLE_S7_HEADER),
        ["IY001_PP0100_sgRNA_F", "GGGG" + "A" * 22 + "CCCC"],
        ["IY001_PP0100_sgRNA_R", "TTTT" + "T" * 22 + "AAAA"],
    ]
    with pytest.raises(y26.TableExtractionError, match="BASIC flanks"):
        y26.build_guide_library(rows, synthetic_kt2440)


def test_build_guide_library_refuses_a_pair_that_is_not_reverse_complementary(
    synthetic_kt2440: Any,
) -> None:
    """The paired oligos check each other; a mis-transcribed spacer is caught."""
    rows = [
        list(y26.TABLE_S7_HEADER),
        [
            "IY001_PP0100_sgRNA_F",
            y26.OLIGO_FORWARD_PREFIX + "A" * 22 + y26.OLIGO_FORWARD_SUFFIX,
        ],
        [
            "IY001_PP0100_sgRNA_R",
            y26.OLIGO_REVERSE_PREFIX + "G" * 22 + y26.OLIGO_REVERSE_SUFFIX,
        ],
    ]
    with pytest.raises(y26.TableExtractionError, match="reverse complement"):
        y26.build_guide_library(rows, synthetic_kt2440)


def test_build_guide_library_ignores_an_oligo_with_no_partner(
    synthetic_kt2440: Any,
) -> None:
    """A lone forward oligo (a sequencing primer) is not a guide."""
    rows = [list(y26.TABLE_S7_HEADER), ["IY77_dCas9_F", "cgaatcttggagctcccgctg"]]
    library = y26.build_guide_library(rows, synthetic_kt2440)
    assert library.n_pairs == 0
    assert library.by_key == {}


def test_oligo_label_parts_strips_sgrna_and_the_variant_in_either_order() -> None:
    """Both released orders of the ``sgRNA`` and ``NT<n>`` tokens parse the same."""
    assert y26._oligo_label_parts("PP4549_NT1_sgRNA") == ("PP4549", "NT1")
    assert y26._oligo_label_parts("PP0103_sgRNA_NT1") == ("PP0103", "NT1")
    assert y26._oligo_label_parts("PP4188_sgRNA") == ("PP4188", None)


# --------------------------------------------------------------------------- #
# Synthetic: the guide-assignment rule, all four outcomes
# --------------------------------------------------------------------------- #
@pytest.fixture
def library(synthetic_docx: Path, synthetic_kt2440: Any) -> y26.GuideLibrary:
    """The guide library of the synthetic Table S7."""
    return y26.build_guide_library(
        y26.supplementary_tables(synthetic_docx)["S7"], synthetic_kt2440
    )


def test_assign_prefers_the_exact_tag_and_variant(library: y26.GuideLibrary) -> None:
    """A screened target with a variant label gets that variant's own spacer."""
    assignment = library.assign("PP_0106", "NT3")
    assert assignment.spacer == "CCAAGGTTCCAAGGTTCCAAGG"
    assert assignment.oligo_label == "PP0106_NT3_sgRNA"
    assert assignment.reason == "exact (locus tag, variant) oligo"


def test_assign_falls_through_to_a_tags_only_spacer(library: y26.GuideLibrary) -> None:
    """A target with no variant label resolves when the tag has exactly one spacer."""
    assignment = library.assign("PP_0101", None)
    assert assignment.spacer == "TTGGCCAATTGGCCAATTGGCC"
    assert assignment.reason == "the tag's only oligo spacer"


def test_assign_refuses_to_choose_between_two_distinct_spacers(
    library: y26.GuideLibrary,
) -> None:
    """Two variant oligos and no variant label means the source does not say which."""
    assignment = library.assign("PP_0107", None)
    assert assignment.spacer is None
    assert "distinct spacers" in assignment.reason


def test_assign_reports_a_locus_with_no_oligo(library: y26.GuideLibrary) -> None:
    """A locus Table S7 never names gets a typed reason, not a borrowed spacer."""
    assignment = library.assign("PP_0111", None)
    assert assignment.spacer is None
    assert assignment.reason == "no Table S7 oligo names this locus"


# --------------------------------------------------------------------------- #
# Synthetic: the phenotypes
# --------------------------------------------------------------------------- #
def test_relative_expression_phenotype_names_the_scale_and_gaps_the_missing_se() -> (
    None
):
    """One protein, the released ratio, n = 1, and a typed absence for the SE."""
    pheno = y26.relative_expression_phenotype({"PP_0100": 0.25}, n_replicates=1)
    assert pheno.protein_abundance == {"PP_0100": 0.25}
    assert pheno.n_replicates == {"PP_0100": 1}
    assert pheno.measurement_type == y26.MEASUREMENT_TYPE
    assert pheno.protein_abundance_se is None
    assert pheno.gapped_fields() == {"protein_abundance_se"}


def test_relative_expression_phenotype_keeps_a_released_zero() -> None:
    """A released 0 means no detectable target protein; it is kept, not imputed."""
    pheno = y26.relative_expression_phenotype({"PP_0101": 0.0}, n_replicates=1)
    assert pheno.protein_abundance["PP_0101"] == 0.0


def test_relative_expression_phenotype_refuses_an_empty_map() -> None:
    """A record with no measured protein is not a measurement."""
    with pytest.raises(RuntimeError, match="at least one protein"):
        y26.relative_expression_phenotype({}, n_replicates=1)


def test_relative_expression_phenotype_refuses_a_non_finite_value() -> None:
    """An inf or NaN ratio would poison every downstream statistic."""
    with pytest.raises(RuntimeError, match="non-finite"):
        y26.relative_expression_phenotype({"PP_0100": math.inf}, n_replicates=1)


def test_array_phenotype_carries_the_mean_with_the_standard_error() -> None:
    """The SE the Fig. 3 caption's sample SD implies: SD / sqrt(n)."""
    pheno = y26.array_phenotype({"PP_0200": 0.21}, {"PP_0200": 0.01}, {"PP_0200": 3})
    assert pheno.protein_abundance_se == {"PP_0200": 0.01}
    assert pheno.n_replicates == {"PP_0200": 3}
    assert pheno.gapped_fields() == set()


def test_array_phenotype_refuses_disagreeing_key_sets() -> None:
    """A mean with no SE, or an SE with no mean, is a programming error."""
    with pytest.raises(RuntimeError, match="keys disagree"):
        y26.array_phenotype({"PP_0200": 0.2}, {}, {"PP_0200": 3})


def test_reference_phenotype_is_the_ratios_denominator() -> None:
    """The control strain's value on this scale is 1.0, for every measured protein."""
    pheno = y26.reference_phenotype(
        ["PP_0200", "PP_0201"], n_replicates=3, with_se=True
    )
    assert pheno.protein_abundance == {"PP_0200": 1.0, "PP_0201": 1.0}
    assert pheno.protein_abundance_se == {"PP_0200": 0.0, "PP_0201": 0.0}
    assert y26.REFERENCE_RELATIVE_EXPRESSION == 1.0


def test_reference_phenotype_without_se_carries_the_same_typed_gap() -> None:
    """The Table S3 family's reference gaps its SE exactly as its records do."""
    pheno = y26.reference_phenotype(["PP_0100"], n_replicates=1, with_se=False)
    assert pheno.protein_abundance == {"PP_0100": 1.0}
    assert pheno.gapped_fields() == {"protein_abundance_se"}


def test_reference_phenotype_refuses_an_empty_protein_set() -> None:
    """A reference must cover the record's own measured proteins."""
    with pytest.raises(RuntimeError, match="measured proteins"):
        y26.reference_phenotype([], n_replicates=1, with_se=False)


def test_protein_abundance_phenotype_still_requires_matched_replicate_keys() -> None:
    """The schema invariant this module relies on: one replicate count per protein."""
    with pytest.raises(ValidationError):
        ProteinAbundancePhenotype(
            protein_abundance={"PP_0100": 0.2},
            n_replicates={"PP_0101": 1},
            measurement_type=y26.MEASUREMENT_TYPE,
        )


# --------------------------------------------------------------------------- #
# Synthetic: the genotype and the environment
# --------------------------------------------------------------------------- #
def test_chassis_background_defers_the_genotype_it_cannot_source() -> None:
    """``IY1452`` is named, its alleles are not released, and both are typed."""
    background = y26.chassis_background()
    assert background.name == y26.CHASSIS_STRAIN == "IY1452"
    assert background.reference_strain == "KT2440"
    assert background.assembly_set == BACTERIAL_ASSEMBLY_SETS["KT2440"]
    assert background.parents == ["KT2440"]
    assert background.alleles == []
    assert background.gapped_fields() == {"genotype_statement", "construction"}


def test_chassis_background_gap_points_at_the_paper_that_would_close_it() -> None:
    """A deferral is a worklist item, so it names where we looked and what resolves it."""
    gap = next(
        g
        for g in y26.chassis_background().provenance_gaps
        if g.field == "genotype_statement"
    )
    assert gap.reason.value == "deferred_pending_source_review"
    assert gap.looked_in is not None
    assert gap.looked_in.citation_key == y26.CITATION_KEY
    assert gap.resolve_with is not None
    assert y26.CHASSIS_SOURCE_DOI in gap.resolve_with.source_uri


def test_production_environment_records_the_medium_the_methods_name() -> None:
    """The library's M9 salts, 30 C, 48 h, aerobic."""
    env = y26.production_environment()
    assert env.media.name == M9.name
    assert env.temperature is not None
    assert env.temperature.value == pytest.approx(30.0)
    assert env.duration_hours == pytest.approx(48.0)
    assert env.aerobicity == "aerobic"


def test_production_environment_carries_the_glucose_as_the_carbon_source_factor() -> (
    None
):
    """2 % glucose is the carbon source; the library has no M9-plus-glucose entry yet."""
    env = y26.production_environment()
    carbon = [
        p
        for p in env.perturbations
        if isinstance(p, EnvironmentPhysicalPerturbation)
        and p.factor is PhysicalFactor.carbon_source
    ]
    assert len(carbon) == 1
    magnitude = carbon[0].magnitude
    agent = carbon[0].agent
    assert magnitude is not None
    assert magnitude.value == pytest.approx(2.0)
    assert magnitude.unit is ConcentrationUnit.percent_w_v
    assert agent is not None
    assert agent.name == "D-glucose"


def test_production_environment_doses_the_inducer_and_both_antibiotics() -> None:
    """mg/L is stored as the numerically identical ug/mL, since the enum has no mg/L."""
    env = y26.production_environment()
    doses = {
        p.compound.name: p.concentration
        for p in env.perturbations
        if isinstance(p, SmallMoleculePerturbation)
    }
    assert doses["L-arabinose"].value == pytest.approx(0.2)
    assert doses["L-arabinose"].unit is ConcentrationUnit.percent_w_v
    assert doses["kanamycin"].value == pytest.approx(50.0)
    assert doses["kanamycin"].unit is ConcentrationUnit.ug_per_ml
    assert doses["gentamicin"].value == pytest.approx(10.0)
    assert "mg/L" not in {unit.value for unit in ConcentrationUnit}


def test_crispri_perturbation_carries_the_effector_and_the_sourced_spacer() -> None:
    """One guide per gene, the paper's dCas9, and the spacer when the source states it."""
    assignment = y26.GuideAssignment(
        locus_tag="PP_0100",
        variant=None,
        spacer="ACGTACGTACGTACGTACGTAC",
        oligo_label="PP0100_sgRNA",
        reason="the tag's only oligo spacer",
    )
    pert = y26.crispri_perturbation("PP_0100", "ppsA", assignment)
    assert pert.perturbation_type == "bacterial_crispr_interference"
    assert pert.gene_namespace == y26.KT2440_NAMESPACE
    assert pert.crispr is not None
    assert pert.crispr.effector == "dCas9"
    assert pert.crispr.guide_sequence == "ACGTACGTACGTACGTACGTAC"
    assert pert.crispr.n_guides == 1


def test_crispri_perturbation_leaves_the_spacer_none_when_it_is_not_sourced() -> None:
    """An unresolved guide scaffolds-and-defers rather than borrowing a sequence."""
    assignment = y26.GuideAssignment(
        locus_tag="PP_0111",
        variant=None,
        spacer=None,
        oligo_label=None,
        reason="no Table S7 oligo names this locus",
    )
    pert = y26.crispri_perturbation("PP_0111", "PP_0111", assignment)
    assert pert.crispr is not None
    assert pert.crispr.guide_sequence is None


def test_publication_carries_the_doi_this_paper_has() -> None:
    """No PubMed id is in the mirrored metadata, so the DOI is the identifier."""
    pub = y26.publication()
    assert pub.doi == y26.PAPER_DOI
    assert pub.doi_url is not None and y26.PAPER_DOI in pub.doi_url


def test_check_isoprenol_identity_passes_while_the_compound_has_no_row() -> None:
    """Today the resolver returns the name with an inchikey gap, so the guard is quiet."""
    from torchcell.datamodels.compound_identity import resolved_compound

    compound = resolved_compound("isoprenol")
    assert compound.name == "isoprenol"
    assert compound.inchikey is None
    assert compound.gapped_fields() == {"inchikey"}
    assert y26.ISOPRENOL_INCHIKEY == "CPJRRXSHAYUTGL-UHFFFAOYSA-N"
    y26.check_isoprenol_identity()


def test_check_isoprenol_identity_stops_on_a_disagreeing_curated_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the table gains a different key, this module must be revisited, not ignored."""
    from torchcell.datamodels.schema import Compound

    monkeypatch.setattr(
        y26,
        "resolved_compound",
        lambda name: Compound(name=name, inchikey="AAAAAAAAAAAAAA-BBBBBBBBBB-C"),
    )
    with pytest.raises(RuntimeError, match="not the pinned"):
        y26.check_isoprenol_identity()


# --------------------------------------------------------------------------- #
# Synthetic: the retention ledger
# --------------------------------------------------------------------------- #
def test_drop_log_check_passes_on_balanced_arithmetic() -> None:
    """Kept + dropped equals the candidates, and the rules account for the drops."""
    log = y26.DropLog(
        dataset="d",
        source_rows=10,
        candidate_records=10,
        kept_records=8,
        dropped_records=2,
        rules=[y26.DropRule(rule="r", description="d", n_records=2, items=["a", "b"])],
    )
    log.check()
    assert log.kept_records + log.dropped_records == log.candidate_records
    assert sum(rule.n_records for rule in log.rules) == log.dropped_records
    assert log.rules[0].items == ["a", "b"]


def test_drop_log_check_refuses_unbalanced_totals() -> None:
    """A record that is neither kept nor dropped has vanished."""
    log = y26.DropLog(
        dataset="d",
        source_rows=10,
        candidate_records=10,
        kept_records=8,
        dropped_records=1,
        rules=[y26.DropRule(rule="r", description="d", n_records=1)],
    )
    with pytest.raises(RuntimeError, match="!= 10 candidates"):
        log.check()


def test_drop_log_check_refuses_unaccounted_drops() -> None:
    """A drop with no rule is a silent exclusion."""
    log = y26.DropLog(
        dataset="d",
        source_rows=10,
        candidate_records=10,
        kept_records=8,
        dropped_records=2,
        rules=[y26.DropRule(rule="r", description="d", n_records=1)],
    )
    with pytest.raises(RuntimeError, match="rules total 1"):
        log.check()


# --------------------------------------------------------------------------- #
# Synthetic: the raw mirror
# --------------------------------------------------------------------------- #
def test_si_retrieval_is_the_scriptable_elsevier_cdn_get() -> None:
    """A plain GET of ars.els-cdn.com, with the PII and filename as its params."""
    record = y26.si_retrieval()
    assert record.method is RetrievalMethod.direct_url
    assert record.retriever == "torchcell.literature.retrieve.elsevier_mmc"
    assert record.params == {"pii": y26.PAPER_PII, "filename": "mmc1.docx"}
    assert record.sha256 == y26.SI_DOCX_SHA256


def test_deposit_raw_mirror_writes_the_file_and_a_complete_manifest(
    synthetic_docx: Path, tmp_path: Path
) -> None:
    """The deposit is one artifact with its retrieval, its extraction recipe and a hash."""
    data_root = tmp_path / "root"
    root = y26.deposit_raw_mirror(source=synthetic_docx, data_root=str(data_root))
    assert (root / y26.SI_MIRROR_RELPATH).exists()
    manifest = Manifest.model_validate_json((root / "manifest.json").read_text())
    assert manifest.citation_key == y26.CITATION_KEY
    assert manifest.doi == y26.PAPER_DOI
    assert len(manifest.files) == 1
    record = manifest.files[0]
    assert record.role == ROLE_RAW_DATA
    assert record.sha256 == y26.SI_DOCX_SHA256
    assert record.retrieval is not None
    assert record.processing is not None
    assert record.processing.processor.endswith("read_docx_tables")
    assert y26.SI_SOURCE_URL in manifest.si_data_sources
    assert any("Benchling" in line for line in manifest.si_expected)


def test_deposit_raw_mirror_is_idempotent(synthetic_docx: Path, tmp_path: Path) -> None:
    """A second deposit of the same bytes leaves the mirror file alone."""
    data_root = tmp_path / "root"
    root = y26.deposit_raw_mirror(source=synthetic_docx, data_root=str(data_root))
    before = (root / y26.SI_MIRROR_RELPATH).stat().st_mtime_ns
    y26.deposit_raw_mirror(source=synthetic_docx, data_root=str(data_root))
    assert (root / y26.SI_MIRROR_RELPATH).stat().st_mtime_ns == before


def test_deposit_raw_mirror_refuses_a_source_with_another_hash(
    synthetic_docx: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upstream drift is detected, never followed."""
    monkeypatch.setattr(y26, "SI_DOCX_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        y26.deposit_raw_mirror(source=synthetic_docx, data_root=str(tmp_path / "r"))


def test_deposit_raw_mirror_refuses_to_overwrite_a_differing_mirror_file(
    synthetic_docx: Path, tmp_path: Path
) -> None:
    """A mirror file whose bytes differ is a provenance break, not a cache miss."""
    data_root = tmp_path / "root"
    root = y26.deposit_raw_mirror(source=synthetic_docx, data_root=str(data_root))
    (root / y26.SI_MIRROR_RELPATH).write_bytes(b"other")
    with pytest.raises(RuntimeError, match="different sha256; refusing"):
        y26.deposit_raw_mirror(source=synthetic_docx, data_root=str(data_root))


def test_manifest_sha256_raises_on_a_path_the_manifest_does_not_carry(
    synthetic_docx: Path, tmp_path: Path
) -> None:
    """A pin can only be checked against a recorded file."""
    y26.deposit_raw_mirror(source=synthetic_docx, data_root=str(tmp_path / "root"))
    manifest = y26.load_manifest(str(tmp_path / "root"))
    assert y26.manifest_sha256(manifest, y26.SI_MIRROR_RELPATH) == y26.SI_DOCX_SHA256
    with pytest.raises(KeyError, match="not in the raw-mirror manifest"):
        y26.manifest_sha256(manifest, "data/absent.docx")


def test_retrieve_raw_files_refuses_bytes_that_do_not_match_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed upstream becomes a NEW provenance record, never a silent overwrite."""
    import torchcell.literature.provenance as provenance

    monkeypatch.setattr(provenance, "run_retriever", lambda record: b"drifted")
    with pytest.raises(RuntimeError, match="upstream changed"):
        y26.retrieve_raw_files(tmp_path / "dl")


def test_retrieve_raw_files_writes_the_verified_bytes(
    synthetic_docx: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recorded retrieval re-runs and the bytes are kept only when they match."""
    import torchcell.literature.provenance as provenance

    payload = synthetic_docx.read_bytes()
    monkeypatch.setattr(provenance, "run_retriever", lambda record: payload)
    out = y26.retrieve_raw_files(tmp_path / "dl")
    assert out[y26.SI_DOCX].read_bytes() == payload


def test_raw_mirror_dir_reads_data_root_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mirror path is derived, never configured per call site."""
    monkeypatch.setenv("DATA_ROOT", "/from_env")
    assert y26.raw_mirror_dir() == Path("/from_env") / y26.RAW_DIR_REL
    assert y26.library_dir() == Path("/from_env") / y26.LIBRARY_DIR_REL


# --------------------------------------------------------------------------- #
# Synthetic: both loaders built end to end under tmp_path
# --------------------------------------------------------------------------- #
@pytest.fixture
def synthetic_mirror(
    synthetic_docx: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """A raw mirror of the synthetic docx, with ``DATA_ROOT`` pointed at it."""
    data_root = tmp_path / "data_root"
    y26.deposit_raw_mirror(source=synthetic_docx, data_root=str(data_root))
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    return data_root


def _dump(obj: Any) -> dict[str, Any]:
    """A record part as a plain dict, whether the dataset returned a model or a dict."""
    return obj if isinstance(obj, dict) else obj.model_dump()


def _build(cls: type, root: Path, genome: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Build one loader with the synthetic genome injected."""
    monkeypatch.setattr(y26, "bacterial_genome", lambda *a, **k: genome)
    return cls(root=str(root), pputida_genome=genome)


def test_knockdown_dataset_builds_one_record_per_released_ratio(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 'n.d.' rows are dropped by a named rule; every other row is a record."""
    dataset = _build(
        y26.CrispriKnockdownYunus2026Dataset,
        tmp_path / "knockdown",
        synthetic_kt2440,
        monkeypatch,
    )
    assert len(dataset) == len(SCREEN_ROWS) - SCREEN_ND
    log = json.loads(Path(dataset.preprocess_dir, "dropped_records.json").read_text())
    assert log["kept_records"] == len(dataset)
    assert log["dropped_records"] == SCREEN_ND
    assert log["rules"][0]["rule"] == "control_strain_expression_not_detected"
    assert len(log["rules"][0]["items"]) == SCREEN_ND


def test_knockdown_record_stores_the_ratio_against_a_denominator_reference(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The experiment over its reference reproduces the released number exactly."""
    dataset = _build(
        y26.CrispriKnockdownYunus2026Dataset,
        tmp_path / "knockdown",
        synthetic_kt2440,
        monkeypatch,
    )
    item = dataset[0]
    experiment = _dump(item["experiment"])
    reference = _dump(item["reference"])
    abundance = experiment["phenotype"]["protein_abundance"]
    tag = next(iter(abundance))
    value = abundance[tag]
    denominator = reference["phenotype_reference"]["protein_abundance"][tag]
    assert denominator == 1.0
    assert value / denominator == value
    assert experiment["phenotype"]["measurement_type"] == y26.MEASUREMENT_TYPE
    assert reference["genome_reference"]["strain"] == y26.CHASSIS_STRAIN
    assert reference["genome_reference"]["assembly_accession"].startswith("GCA_")


def test_knockdown_record_genotype_is_one_crispri_perturbation_on_its_target(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The measured protein IS the knocked-down gene for this family."""
    dataset = _build(
        y26.CrispriKnockdownYunus2026Dataset,
        tmp_path / "knockdown",
        synthetic_kt2440,
        monkeypatch,
    )
    for index in range(len(dataset)):
        experiment = _dump(dataset[index]["experiment"])
        perturbations = experiment["genotype"]["perturbations"]
        assert len(perturbations) == 1
        assert {p["systematic_gene_name"] for p in perturbations} == set(
            experiment["phenotype"]["protein_abundance"]
        )


def test_knockdown_build_writes_the_guide_assignment_ledger(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every record's guide outcome is auditable, including the three that have none."""
    import pandas as pd

    dataset = _build(
        y26.CrispriKnockdownYunus2026Dataset,
        tmp_path / "knockdown",
        synthetic_kt2440,
        monkeypatch,
    )
    table = pd.read_csv(osp.join(dataset.preprocess_dir, "table_s3.csv"))
    assert len(table) == len(dataset)
    without = table[table.guide_spacer.isna()]
    assert set(without.locus_tag) == {"PP_0107", "PP_0111"}
    assert (
        osp.exists(osp.join(dataset.preprocess_dir, "guide_assignment.csv"))
        and osp.exists(osp.join(dataset.preprocess_dir, "guide_library.json"))
        and osp.exists(osp.join(dataset.preprocess_dir, "target_lists.csv"))
    )


def test_array_dataset_builds_one_record_per_construct(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every panel's cells for one construct land in that construct's single record."""
    dataset = _build(
        y26.CrispriArrayYunus2026Dataset,
        tmp_path / "array",
        synthetic_kt2440,
        monkeypatch,
    )
    constructs = {
        construct for rows in ARRAY_PANEL_ROWS.values() for construct, _ in rows
    }
    assert len(dataset) == len(constructs)
    log = json.loads(Path(dataset.preprocess_dir, "dropped_records.json").read_text())
    assert log["dropped_records"] == 0


def test_array_record_carries_every_protein_its_panels_measured(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A construct measured in three panels holds three proteins, with SEs over n = 3."""
    dataset = _build(
        y26.CrispriArrayYunus2026Dataset,
        tmp_path / "array",
        synthetic_kt2440,
        monkeypatch,
    )
    by_guides = {}
    for index in range(len(dataset)):
        experiment = _dump(dataset[index]["experiment"])
        key = tuple(
            sorted(
                p["systematic_gene_name"]
                for p in experiment["genotype"]["perturbations"]
            )
        )
        by_guides[key] = experiment["phenotype"]
    triple = by_guides[("PP_0200", "PP_0201", "PP_0202")]
    assert set(triple["protein_abundance"]) == {
        "PP_0200",
        "PP_0201",
        "PP_0202",
        "PP_0100",
    }
    assert set(triple["n_replicates"].values()) == {3}
    assert triple["protein_abundance_se"] is not None
    assert triple["protein_abundance"]["PP_0200"] == pytest.approx(0.41)
    assert triple["protein_abundance_se"]["PP_0200"] == pytest.approx(
        0.01 / math.sqrt(3)
    )


def test_array_dataset_measures_a_protein_in_a_construct_with_no_guide_for_it(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A panel's own control: the measured protein need not be a guide target."""
    dataset = _build(
        y26.CrispriArrayYunus2026Dataset,
        tmp_path / "array",
        synthetic_kt2440,
        monkeypatch,
    )
    for index in range(len(dataset)):
        experiment = _dump(dataset[index]["experiment"])
        guides = {
            p["systematic_gene_name"] for p in experiment["genotype"]["perturbations"]
        }
        measured = set(experiment["phenotype"]["protein_abundance"])
        if measured - guides:
            return
    pytest.fail("no construct measures a protein it carries no guide for")


def test_array_dataset_refuses_a_panel_with_the_wrong_replicate_count(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Fig. 3 caption states three biological replicates, and that is enforced."""
    monkeypatch.setattr(y26, "ARRAY_N_REPLICATES", 4)
    with pytest.raises(y26.TableExtractionError, match="replicate counts"):
        _build(
            y26.CrispriArrayYunus2026Dataset,
            tmp_path / "array",
            synthetic_kt2440,
            monkeypatch,
        )


def test_loader_refuses_a_genome_of_another_assembly_set(
    synthetic_mirror: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record's identifiers mean one assembly; another genome is refused outright."""

    class WrongGenome:
        ASSEMBLY_SET = "ecoli_K12_MG1655_ASM584v2"

    injected: dict[str, Any] = {"pputida_genome": WrongGenome()}
    with pytest.raises(ValueError, match="needs the pputida_KT2440_ASM756v2 genome"):
        y26.CrispriKnockdownYunus2026Dataset(root=str(tmp_path / "wrong"), **injected)


def test_download_refuses_a_mirror_whose_file_is_absent(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A manifest without its bytes is a broken mirror, not a cache to repopulate."""
    (synthetic_mirror / y26.RAW_DIR_REL / y26.SI_MIRROR_RELPATH).unlink()
    with pytest.raises(RuntimeError, match="missing from mirror"):
        _build(
            y26.CrispriKnockdownYunus2026Dataset,
            tmp_path / "knockdown",
            synthetic_kt2440,
            monkeypatch,
        )


def test_verification_passes_on_a_synthetic_build(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L0-L3 plus the reference-denominator L4 rule pass on the synthetic LMDB."""
    from torchcell.verification.protein import verify_protein_dataset
    from torchcell.verification.runners import load_records

    root = tmp_path / "knockdown"
    dataset = _build(
        y26.CrispriKnockdownYunus2026Dataset, root, synthetic_kt2440, monkeypatch
    )
    records = load_records(str(root))
    report = verify_protein_dataset(
        records,
        dataset_name=dataset.name,
        provenance=y26.verifier_provenance("crispri_knockdown_yunus2026"),
        expected_count=len(dataset),
        allow_duplicate_orfs=True,
    )
    report.add(y26._l4_reference_is_the_ratio_denominator(records))
    assert report.passed, report.summary()
    denominator = next(
        r for r in report.results if r.name == "reference_is_the_ratio_denominator"
    )
    assert denominator.level is Level.L4


def test_l4_reference_rule_fails_a_rescaled_reference() -> None:
    """A reference off 1.0 would silently rescale every record, so the rule catches it."""
    records = [
        {"reference": {"phenotype_reference": {"protein_abundance": {"PP_0100": 2.0}}}}
    ]
    result = y26._l4_reference_is_the_ratio_denominator(records)
    assert not result.passed
    assert "PP_0100=2.0" in result.details["examples"][0]


def test_verifier_provenance_names_the_table_each_family_consumes() -> None:
    """The verifier's own record says which table and which derivation it checked."""
    knockdown = y26.verifier_provenance("crispri_knockdown_yunus2026")
    array = y26.verifier_provenance("crispri_array_yunus2026")
    assert knockdown.page is not None and "Table S3" in knockdown.page
    assert array.page is not None and "S8-S12" in array.page
    assert knockdown.sha256 == y26.SI_DOCX_SHA256


def test_print_table_digests_prints_one_line_per_pinned_table(
    synthetic_docx: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The pinning helper prints exactly the dict body it is pasted into."""
    digests = y26.print_table_digests(synthetic_docx)
    assert set(digests) == set(y26.TABLE_DIGESTS)
    assert len(capsys.readouterr().out.strip().splitlines()) == len(digests)


def test_not_loaded_names_every_unconsumed_release() -> None:
    """The reasons are in the module, not only in a note."""
    joined = " ".join(y26.NOT_LOADED)
    for needle in ("Benchling", "PXD062697", "Supplementary Tables S4 and S5", "mmc2"):
        assert needle in joined


def test_sourced_values_record_both_spellings_of_the_best_target() -> None:
    """The Abstract's PP_4118 and the Results' PP_4188 are both kept, verbatim."""
    assert y26.SOURCED_VALUES["abstract_best_target"].value == "PP_4118"
    assert y26.SOURCED_VALUES["results_best_target"].value == "PP_4188"
    note = y26.SOURCED_VALUES["abstract_best_target"].note
    assert note is not None and "PP_4188" in note


def test_every_sourced_value_is_pinned_to_the_paper_mirror() -> None:
    """One citation key, one sha256, one quote each: nothing unanchored."""
    for name, value in y26.SOURCED_VALUES.items():
        assert value.provenance.citation_key == y26.CITATION_KEY, name
        assert value.provenance.sha256 == y26.PAPER_MD_SHA256, name
        assert value.quote.strip(), name


# --------------------------------------------------------------------------- #
# Data-gated: the real mirror and the real annotation
# --------------------------------------------------------------------------- #
def _data_root_or_skip() -> str:
    """``$DATA_ROOT``, or skip when it is not mounted."""
    from dotenv import load_dotenv

    load_dotenv()
    root = os.environ.get("DATA_ROOT")
    if not root or not osp.isdir(root):
        pytest.skip("DATA_ROOT is not mounted")
    return root


@pytest.mark.data
def test_real_mirror_matches_the_pinned_digest() -> None:
    """The deposited docx is the bytes the module was written against."""
    root = _data_root_or_skip()
    path = y26.raw_mirror_dir(root) / y26.SI_MIRROR_RELPATH
    if not path.exists():
        pytest.skip("the Yunus 2026 raw mirror is not deposited")
    assert _sha256_bytes(path) == y26.SI_DOCX_SHA256
    manifest = y26.load_manifest(root)
    assert y26.manifest_sha256(manifest, y26.SI_MIRROR_RELPATH) == y26.SI_DOCX_SHA256


@pytest.mark.data
def test_real_tables_hold_the_counts_this_loader_was_measured_on() -> None:
    """125 Table S3 rows with 23 'n.d.', 204 guide pairs, and 25 array constructs."""
    root = _data_root_or_skip()
    path = y26.raw_mirror_dir(root) / y26.SI_MIRROR_RELPATH
    if not path.exists():
        pytest.skip("the Yunus 2026 raw mirror is not deposited")
    tables = y26.supplementary_tables(path)
    rows = y26.parse_table_s3(tables["S3"])
    assert len(rows) == 125
    assert sum(1 for row in rows if row.not_detected) == 23
    cells = [
        cell
        for table, protein in y26.ARRAY_PANELS
        for cell in y26.parse_array_panel(tables[table], protein)
    ]
    assert len(cells) == 153
    assert len({cell.construct_name for cell in cells}) == 25


@pytest.mark.data
def test_real_guide_library_is_reverse_complement_consistent_throughout() -> None:
    """204 of 204 released pairs check each other on the real annotation."""
    root = _data_root_or_skip()
    path = y26.raw_mirror_dir(root) / y26.SI_MIRROR_RELPATH
    if not path.exists():
        pytest.skip("the Yunus 2026 raw mirror is not deposited")
    from torchcell.datasets.bacteria_common import bacterial_genome

    try:
        genome = bacterial_genome("pputida", "KT2440", root)
    except Exception:  # noqa: BLE001 - the tier cache may be absent on this machine
        pytest.skip("the KT2440 genomes tier is not available")
    library = y26.build_guide_library(y26.supplementary_tables(path)["S7"], genome)
    assert library.n_pairs == 204
    assert library.n_off_length == ["PP4650_sgRNA_NT1 (21 nt)"]
    assert set(library.unmapped_labels) == {"RFP", "BFP", "nontarget", "glgC"}


@pytest.mark.data
def test_real_screened_targets_all_resolve_to_kt2440_loci() -> None:
    """Every Table S3 target is a locus tag of the pinned assembly."""
    root = _data_root_or_skip()
    path = y26.raw_mirror_dir(root) / y26.SI_MIRROR_RELPATH
    if not path.exists():
        pytest.skip("the Yunus 2026 raw mirror is not deposited")
    from torchcell.datasets.bacteria_common import bacterial_genome

    try:
        genome = bacterial_genome("pputida", "KT2440", root)
    except Exception:  # noqa: BLE001 - the tier cache may be absent on this machine
        pytest.skip("the KT2440 genomes tier is not available")
    import pandas as pd

    from torchcell.datasets.bacteria_common import reconcile_locus_tags

    rows = y26.parse_table_s3(y26.supplementary_tables(path)["S3"])
    tags = sorted({row.locus_tag for row in rows})
    _, report = reconcile_locus_tags(
        genome, pd.Series(tags, dtype=object), label="yunus2026-data"
    )
    assert list(report.outside_namespace) == []
    assert len(tags) == 123


@pytest.mark.data
def test_real_builds_pass_l0_to_l4() -> None:
    """Both built dev-tree LMDBs pass the protein family gate and both L4 rules."""
    root = _data_root_or_skip()
    for name, spec in y26.DATASETS.items():
        if not osp.exists(osp.join(root, str(spec["root"]), "processed", "lmdb")):
            pytest.skip(f"{name} is not built in the dev tree")
        report = y26.run_verification(name, root)
        assert report.passed, report.summary()


# --------------------------------------------------------------------------- #
# Synthetic: the module CLI
# --------------------------------------------------------------------------- #
def test_main_digests_prints_the_pinning_block(
    synthetic_docx: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``digests --path`` prints one line per pinned table and exits 0."""
    monkeypatch.setattr(y26, "load_dotenv", lambda *a, **k: None, raising=False)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "root"))
    assert y26.main(["digests", "--path", str(synthetic_docx)]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == len(y26.TABLE_DIGESTS)


def test_main_deposit_uses_the_literature_mirrors_captured_file(
    synthetic_docx: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``--retrieve-into`` the deposit reads the literature mirror's si/ copy."""
    data_root = tmp_path / "root"
    captured = data_root / y26.LIBRARY_DIR_REL / "si" / y26.SI_DOCX
    captured.parent.mkdir(parents=True)
    captured.write_bytes(synthetic_docx.read_bytes())
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    assert y26.main(["deposit"]) == 0
    assert (data_root / y26.RAW_DIR_REL / y26.SI_MIRROR_RELPATH).exists()


def test_main_deposit_can_re_run_the_recorded_retrieval(
    synthetic_docx: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--retrieve-into`` re-runs the retriever and deposits the verified bytes."""
    import torchcell.literature.provenance as provenance

    payload = synthetic_docx.read_bytes()
    monkeypatch.setattr(provenance, "run_retriever", lambda record: payload)
    data_root = tmp_path / "root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    assert y26.main(["deposit", "--retrieve-into", str(tmp_path / "dl")]) == 0
    mirrored = data_root / y26.RAW_DIR_REL / y26.SI_MIRROR_RELPATH
    assert mirrored.read_bytes() == payload


def test_main_build_then_verify_runs_both_families(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``build`` then ``verify`` over both datasets, with the synthetic genome injected."""
    monkeypatch.setattr(y26, "bacterial_genome", lambda *a, **k: synthetic_kt2440)
    roots = {
        "crispri_knockdown_yunus2026": "data/torchcell/knockdown",
        "crispri_array_yunus2026": "data/torchcell/array",
    }
    for name, rel in roots.items():
        monkeypatch.setitem(y26.DATASETS[name], "root", rel)
    assert y26.main(["build"]) == 0
    out = capsys.readouterr().out
    assert "CrispriKnockdownYunus2026Dataset: len = " in out
    assert "CrispriArrayYunus2026Dataset: len = " in out
    assert y26.main(["verify"]) == 0
    assert "PASS" in capsys.readouterr().out


def test_main_verify_can_select_one_family(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--dataset`` narrows both subcommands to one family."""
    monkeypatch.setattr(y26, "bacterial_genome", lambda *a, **k: synthetic_kt2440)
    monkeypatch.setitem(
        y26.DATASETS["crispri_knockdown_yunus2026"], "root", "data/torchcell/only"
    )
    args = ["--dataset", "crispri_knockdown_yunus2026"]
    assert y26.main(["build", *args]) == 0
    capsys.readouterr()
    assert y26.main(["verify", *args]) == 0
    out = capsys.readouterr().out
    assert "CrispriKnockdownYunus2026Dataset" in out
    assert "CrispriArrayYunus2026Dataset" not in out


def test_run_verification_refuses_more_than_one_genome_reference(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every record of one paper is written against one assembly-pinned reference."""
    import torchcell.verification.runners as runners

    monkeypatch.setattr(y26, "bacterial_genome", lambda *a, **k: synthetic_kt2440)
    monkeypatch.setitem(
        y26.DATASETS["crispri_knockdown_yunus2026"], "root", "data/torchcell/two_refs"
    )
    assert y26.main(["build", "--dataset", "crispri_knockdown_yunus2026"]) == 0
    real = runners.load_records

    def two_references(abs_root: str) -> list[dict[str, Any]]:
        records = real(abs_root)
        drifted = json.loads(json.dumps(records[0]))
        drifted["reference"]["genome_reference"]["strain"] = "KT2440"
        return [*records, drifted]

    monkeypatch.setattr(runners, "load_records", two_references)
    with pytest.raises(ValueError, match="distinct genome references"):
        y26.run_verification("crispri_knockdown_yunus2026", str(synthetic_mirror))


def test_run_verification_skips_the_audits_when_the_library_is_not_mounted(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Without the literature mirror the record gate still runs and says the audits did not."""
    monkeypatch.setattr(y26, "bacterial_genome", lambda *a, **k: synthetic_kt2440)
    monkeypatch.setitem(
        y26.DATASETS["crispri_knockdown_yunus2026"], "root", "data/torchcell/no_library"
    )
    assert y26.main(["build", "--dataset", "crispri_knockdown_yunus2026"]) == 0
    with caplog.at_level("WARNING"):
        report = y26.run_verification(
            "crispri_knockdown_yunus2026", str(synthetic_mirror)
        )
    assert report.passed
    assert not any(r.name == "provenance_audit" for r in report.results)
    assert "provenance audits" in caplog.text
