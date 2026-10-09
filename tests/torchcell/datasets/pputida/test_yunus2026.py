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

import pandas as pd
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


#: The synthetic Tables S4 and S5, as ``(protein key, fold change, p-value)``. Keys that
#: resolve to a locus of the synthetic assembly, plus one that resolves to nothing, which
#: is the shape the pinned docx has (305 of 338 resolve). The knocked-down gene
#: ``PP_0100`` is deliberately ABSENT from both, as ``PP_4188`` is in the real tables.
DIFFERENTIAL_DOWN: tuple[tuple[str, float, float], ...] = (
    ("PP_0101", 0.05, 0.0001),
    ("PP_0102", 0.1, 0.001),
    ("PP_0103", 0.25, 0.01),
    ("PP_0106", 0.3, 0.015),
    ("PP_0107", 0.45, 0.04),
    ("Notagene", 0.4, 0.02),
)
DIFFERENTIAL_UP: tuple[tuple[str, float, float], ...] = (
    ("PP_0104", 8.0, 0.0005),
    ("PP_0105", 2.5, 0.03),
    ("PP_0108", 4.0, 0.002),
    ("PP_0109", 3.0, 0.004),
    ("PP_0110", 2.25, 0.045),
)


def _differential_rows(
    specs: tuple[tuple[str, float, float], ...], *, descending: bool
) -> list[list[str]]:
    """A synthetic Table S4 or S5 whose derived columns satisfy every build oracle."""
    rows = [list(y26.TABLE_S4_S5_HEADER)]
    ordered = sorted(specs, key=lambda spec: spec[1], reverse=descending)
    for rank, (key, fold_change, p_value) in enumerate(ordered, start=1):
        rows.append(
            [
                f"Q{rank:05d}",
                f"{key.upper()}_PSEPK",
                key,
                f"synthetic {key}",
                repr(fold_change),
                repr(math.log2(fold_change)),
                repr(p_value),
                repr(-math.log10(p_value)),
                str(rank),
            ]
        )
    return rows


def synthetic_tables() -> list[list[list[str]]]:
    """The 14 tables of the synthetic supplementary file, in document order."""
    screen = [list(y26.TABLE_S3_HEADER)] + [list(row) for row in SCREEN_ROWS]
    tables: list[list[list[str]]] = [
        _target_list_rows(y26.TABLE_S1_HEADER, [tag for tag, _ in SCREEN_LOCI[:6]]),
        _target_list_rows(y26.TABLE_S2_HEADER, [tag for tag, _ in SCREEN_LOCI[6:]]),
        screen,
        _differential_rows(DIFFERENTIAL_DOWN, descending=False),
        _differential_rows(DIFFERENTIAL_UP, descending=True),
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
    monkeypatch.setattr(
        y26,
        "DIFFERENTIAL_PANELS",
        (
            ("S4", "downregulated", len(DIFFERENTIAL_DOWN)),
            ("S5", "upregulated", len(DIFFERENTIAL_UP)),
        ),
    )
    monkeypatch.setattr(y26, "DIFFERENTIAL_STRAIN_TARGET", SCREEN_LOCI[0][0])
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


def test_check_isoprenol_identity_passes_on_the_committed_table_row() -> None:
    """The table's isoprenol row carries the pinned key, so the guard is quiet."""
    from torchcell.datamodels.compound_identity import resolved_compound

    compound = resolved_compound("isoprenol")
    assert compound.name == "isoprenol"
    assert compound.inchikey == y26.ISOPRENOL_INCHIKEY == "CPJRRXSHAYUTGL-UHFFFAOYSA-N"
    assert compound.gapped_fields() == set()
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
    synthetic_docx: Path, synthetic_benchling: tuple[str, str], tmp_path: Path
) -> None:
    """The deposit is one artifact with its retrieval, its extraction recipe and a hash."""
    data_root = tmp_path / "root"
    _write_benchling_deposit(data_root / y26.RAW_DIR_REL, synthetic_benchling)
    root = y26.deposit_raw_mirror(source=synthetic_docx, data_root=str(data_root))
    assert (root / y26.SI_MIRROR_RELPATH).exists()
    manifest = Manifest.model_validate_json((root / "manifest.json").read_text())
    assert manifest.citation_key == y26.CITATION_KEY
    assert manifest.doi == y26.PAPER_DOI
    assert len(manifest.files) == 3
    record = manifest.files[0]
    assert record.role == ROLE_RAW_DATA
    assert record.sha256 == y26.SI_DOCX_SHA256
    assert record.retrieval is not None
    assert record.processing is not None
    assert record.processing.processor.endswith("read_docx_tables")
    assert y26.SI_SOURCE_URL in manifest.si_data_sources
    assert any("Benchling" in line for line in manifest.si_expected)


def test_deposit_raw_mirror_is_idempotent(
    synthetic_docx: Path, synthetic_benchling: tuple[str, str], tmp_path: Path
) -> None:
    """A second deposit of the same bytes leaves the mirror file alone."""
    data_root = tmp_path / "root"
    _write_benchling_deposit(data_root / y26.RAW_DIR_REL, synthetic_benchling)
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
    synthetic_docx: Path, synthetic_benchling: tuple[str, str], tmp_path: Path
) -> None:
    """A mirror file whose bytes differ is a provenance break, not a cache miss."""
    data_root = tmp_path / "root"
    _write_benchling_deposit(data_root / y26.RAW_DIR_REL, synthetic_benchling)
    root = y26.deposit_raw_mirror(source=synthetic_docx, data_root=str(data_root))
    (root / y26.SI_MIRROR_RELPATH).write_bytes(b"other")
    with pytest.raises(RuntimeError, match="different sha256; refusing"):
        y26.deposit_raw_mirror(source=synthetic_docx, data_root=str(data_root))


def test_manifest_sha256_raises_on_a_path_the_manifest_does_not_carry(
    synthetic_docx: Path, synthetic_benchling: tuple[str, str], tmp_path: Path
) -> None:
    """A pin can only be checked against a recorded file."""
    _write_benchling_deposit(tmp_path / "root" / y26.RAW_DIR_REL, synthetic_benchling)
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
    synthetic_docx: Path,
    synthetic_benchling: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """A raw mirror of the synthetic docx and both synthetic deposits.

    The two Benchling tables are a MANUAL deposit, so they are written into the mirror
    before ``deposit_raw_mirror`` runs: that function verifies them in place and never
    copies them, exactly as the real deposit works.
    """
    data_root = tmp_path / "data_root"
    _write_benchling_deposit(data_root / y26.RAW_DIR_REL, synthetic_benchling)
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
def test_real_differential_tables_hold_the_counts_and_the_oracles() -> None:
    """Tables S4 and S5 on the real bytes: 145 + 193 rows, 305 of 338 keys resolving."""
    root = _data_root_or_skip()
    path = y26.raw_mirror_dir(root) / y26.SI_MIRROR_RELPATH
    if not path.exists():
        pytest.skip("the raw mirror is not deposited")
    tables = y26.supplementary_tables(path)
    rows = y26.read_differential(tables)
    assert len(rows) == 338
    assert sum(1 for row in rows if row.direction == "downregulated") == 145
    assert sum(1 for row in rows if row.direction == "upregulated") == 193
    assert len({row.protein for row in rows}) == 338
    assert rows[0].protein == "Rbsb"
    assert rows[0].fold_change == pytest.approx(0.003286423)
    assert rows[145].protein == "Pp_2686"
    assert rows[145].fold_change == pytest.approx(459.0034538)
    assert max(abs(math.log2(r.fold_change) - r.log2_fold_change) for r in rows) < 1e-6
    printed = [row[6].strip() for name in ("S4", "S5") for row in tables[name][1:]]
    assert len(printed) == 338
    assert sum(1 for cell in printed if "E" in cell.upper()) == 23
    worst = max(
        abs(10.0**-row.neg_log10_p_value - row.p_value) / row.p_value for row in rows
    )
    assert worst == pytest.approx(3.2488e-3, rel=1e-3)
    assert worst < y26._P_VALUE_TOL


@pytest.mark.data
def test_the_knocked_down_gene_is_absent_from_both_real_tables() -> None:
    """The audit's claim, re-measured, plus the reason it holds: Kgdb is PP_4188."""
    root = _data_root_or_skip()
    path = y26.raw_mirror_dir(root) / y26.SI_MIRROR_RELPATH
    if not path.exists():
        pytest.skip("the raw mirror is not deposited")
    tables = y26.supplementary_tables(path)
    rows = y26.read_differential(tables)
    assert y26.DIFFERENTIAL_STRAIN_TARGET == "PP_4188"
    assert not any(row.protein == "PP_4188" for row in rows)
    assert not any("4188" in row.protein for row in rows)
    kgdb = next(row for row in rows if row.protein == "Kgdb")
    assert kgdb.accession == "Q88FB0"
    assert kgdb.description == (
        "Dihydrolipoyllysine-residue succinyltransferase component of 2-oxoglutarate "
        "dehydrogenase complex"
    )
    assert kgdb.fold_change == pytest.approx(0.238537433)
    kgda = next(row for row in rows if row.protein == "Kgda")
    assert kgda.accession == "Q88FA9"
    assert kgda.description == "2-oxoglutarate dehydrogenase, E1 component"

    from torchcell.datasets.bacteria_common import (
        bacterial_genome,
        reconcile_locus_tags,
    )

    genome = bacterial_genome("pputida", "KT2440", root)
    exact, _ = genome.feature_index["symbol"]
    assert exact.get("sucB") == ["PP_4188"]
    assert exact.get("sucA") == ["PP_4189"]
    assert exact.get("kgdB") is None
    keys = sorted({row.protein for row in rows})
    stored, report = reconcile_locus_tags(
        genome, pd.Series(keys, dtype=object), label="yunus_differential"
    )
    assert report.unique_names == 338
    assert report.resolved == 305
    assert report.resolved_fraction == pytest.approx(305 / 338)
    assert report.resolved_fraction >= (
        y26.CrispriDifferentialProteomeYunus2026Dataset.MIN_RESOLVED_FRACTION
    )
    outside = set(report.outside_namespace)
    assert len(outside) == 33
    assert {"Kgda", "Kgdb"} <= outside
    resolved = dict(zip(keys, stored.tolist(), strict=True))
    assert not [key for key, tag in resolved.items() if tag == "PP_4188"]


@pytest.mark.data
def test_every_docx_sourced_quote_is_verbatim_in_the_pinned_bytes() -> None:
    """The three values quoted from the docx, audited against the parsed docx itself."""
    root = _data_root_or_skip()
    path = y26.raw_mirror_dir(root) / y26.SI_MIRROR_RELPATH
    if not path.exists():
        pytest.skip("the raw mirror is not deposited")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == y26.SI_DOCX_SHA256
    with zipfile.ZipFile(path) as archive:
        document = archive.read("word/document.xml")
    import xml.etree.ElementTree as ElementTree

    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    body = ElementTree.fromstring(document).find(f"{namespace}body")
    assert body is not None
    text = {
        "".join(node.text or "" for node in child.iter(f"{namespace}t")).strip()
        for child in body
    }
    for name, value in y26.SI_SOURCED_VALUES.items():
        assert value.quote in text, name


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
# Synthetic: the PP_4188 differential (Tables S4 and S5)
# --------------------------------------------------------------------------- #
@pytest.fixture
def built_differential(
    synthetic_mirror: Path, synthetic_kt2440: Any, tmp_path: Path
) -> Any:
    """The differential family built over the synthetic mirror and annotation."""
    return y26.CrispriDifferentialProteomeYunus2026Dataset(
        root=str(tmp_path / "build" / "differential"), pputida_genome=synthetic_kt2440
    )


def test_the_differential_parser_types_every_released_column(
    synthetic_docx: Path,
) -> None:
    """Nine columns, both halves, and the keys of the two tables are disjoint."""
    tables = y26.supplementary_tables(synthetic_docx)
    rows = y26.read_differential(tables)
    assert len(rows) == len(DIFFERENTIAL_DOWN) + len(DIFFERENTIAL_UP)
    down = [row for row in rows if row.direction == "downregulated"]
    up = [row for row in rows if row.direction == "upregulated"]
    assert {row.protein for row in down} == {key for key, _, _ in DIFFERENTIAL_DOWN}
    assert {row.protein for row in up} == {key for key, _, _ in DIFFERENTIAL_UP}
    assert all(row.fold_change < 1.0 for row in down)
    assert all(row.fold_change > 1.0 for row in up)
    assert [row.rank for row in down] == list(range(1, len(down) + 1))
    assert [row.fold_change for row in down] == sorted(row.fold_change for row in down)
    assert [row.fold_change for row in up] == sorted(
        (row.fold_change for row in up), reverse=True
    )


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        (5, "0.0", "are not one quantity"),
        (6, "0.9", "beyond the printed precision"),
        (8, "99", "is not 1.."),
    ],
)
def test_a_differential_column_that_lost_its_relation_is_refused(
    synthetic_docx: Path, column: int, value: str, match: str
) -> None:
    """Each derived column is an oracle on the released bytes, not decoration."""
    rows = [list(row) for row in y26.supplementary_tables(synthetic_docx)["S4"]]
    rows[1][column] = value
    with pytest.raises(y26.TableExtractionError, match=match):
        y26.parse_differential(rows, table="S4", direction="downregulated")


def test_a_differential_row_on_the_wrong_side_of_one_is_refused(
    synthetic_docx: Path,
) -> None:
    """A fold change above 1 cannot be in the downregulated table."""
    rows = [list(row) for row in y26.supplementary_tables(synthetic_docx)["S4"]]
    rows[1][4] = "4.0"
    rows[1][5] = repr(math.log2(4.0))
    with pytest.raises(y26.TableExtractionError, match="in the downregulated table"):
        y26.parse_differential(rows, table="S4", direction="downregulated")


def test_a_panel_row_count_that_moved_is_refused(synthetic_docx: Path) -> None:
    """The pinned per-panel row count is asserted before a record is written."""
    tables = y26.supplementary_tables(synthetic_docx)
    with pytest.raises(y26.TableExtractionError, match="holds 6 rows, pinned 99"):
        y26.read_differential(
            tables, (("S4", "downregulated", 99), ("S5", "upregulated", 5))
        )


def test_the_differential_record_is_one_profile_on_its_own_scale(
    built_differential: Any,
) -> None:
    """One record, the resolvable keys only, on a measurement_type of its own."""
    assert len(built_differential) == 1
    record = built_differential[0]
    phenotype = record["experiment"]["phenotype"]
    assert phenotype["measurement_type"] == y26.DIFFERENTIAL_MEASUREMENT_TYPE
    assert phenotype["measurement_type"] != y26.MEASUREMENT_TYPE
    expected = {
        key
        for key, _, _ in (*DIFFERENTIAL_DOWN, *DIFFERENTIAL_UP)
        if key.startswith("PP_")
    }
    assert set(phenotype["protein_abundance"]) == expected
    assert phenotype["protein_abundance"]["PP_0101"] == pytest.approx(0.05)
    assert phenotype["protein_abundance"]["PP_0104"] == pytest.approx(8.0)
    assert set(phenotype["n_replicates"].values()) == {y26.DIFFERENTIAL_N_REPLICATES}
    assert phenotype["protein_abundance_se"] is None
    assert [gap["field"] for gap in phenotype["provenance_gaps"]] == [
        "protein_abundance_se"
    ]
    reference = record["reference"]["phenotype_reference"]
    assert set(reference["protein_abundance"].values()) == {
        y26.REFERENCE_RELATIVE_EXPRESSION
    }
    assert reference["measurement_type"] == y26.DIFFERENTIAL_MEASUREMENT_TYPE


def test_the_differential_genotype_is_the_knocked_down_gene_with_its_spacer(
    built_differential: Any,
) -> None:
    """One CRISPRi perturbation, and the knocked-down gene is NOT a measured key."""
    record = built_differential[0]
    perturbations = record["experiment"]["genotype"]["perturbations"]
    assert len(perturbations) == 1
    assert perturbations[0]["perturbation_type"] == "bacterial_crispr_interference"
    assert perturbations[0]["systematic_gene_name"] == SCREEN_LOCI[0][0]
    assert perturbations[0]["crispr"]["guide_sequence"] == OLIGO_SPECS[0][1]
    assert (
        SCREEN_LOCI[0][0] not in record["experiment"]["phenotype"]["protein_abundance"]
    )


def test_a_differential_table_that_measures_the_knocked_down_gene_is_refused(
    synthetic_docx: Path,
) -> None:
    """A strain's own target cannot also be one of its measured fold changes."""
    tables = y26.supplementary_tables(synthetic_docx)
    rows = y26.read_differential(tables)
    screen = y26.parse_table_s3(tables["S3"])
    cls = y26.CrispriDifferentialProteomeYunus2026Dataset
    cls._assert_the_knocked_down_gene_is_not_a_measured_key(rows, screen)
    clashing = [
        *rows,
        rows[0].model_copy(update={"protein": y26.DIFFERENTIAL_STRAIN_TARGET}),
    ]
    with pytest.raises(y26.TableExtractionError, match="is this record's genotype"):
        cls._assert_the_knocked_down_gene_is_not_a_measured_key(clashing, screen)
    with pytest.raises(y26.TableExtractionError, match="no longer screens"):
        cls._assert_the_knocked_down_gene_is_not_a_measured_key(rows, [])


def test_the_differential_drop_log_accounts_for_the_unresolvable_keys(
    built_differential: Any,
) -> None:
    """Every released row is either in the profile or in the ledger with stored=False."""
    preprocess = Path(built_differential.preprocess_dir)
    drops = json.loads((preprocess / "dropped_records.json").read_text())
    assert drops["source_rows"] == len(DIFFERENTIAL_DOWN) + len(DIFFERENTIAL_UP)
    assert drops["candidate_records"] == 1
    assert drops["kept_records"] == 1
    assert drops["dropped_records"] == 0
    ledger = pd.read_csv(preprocess / "differential.csv")
    assert len(ledger) == drops["source_rows"]
    assert set(ledger.loc[~ledger["stored"], "protein"]) == {"Notagene"}
    assert set(ledger.columns) >= {"p_value", "neg_log10_p_value", "rank"}
    for note in y26.DIFFERENTIAL_NOT_STORED:
        assert note in drops["notes"]


def test_the_two_columns_blocked_on_gap_r_are_read_and_never_stored(
    built_differential: Any,
) -> None:
    """The p-value and the rank reach the ledger and no record field."""
    record = built_differential[0]
    phenotype = record["experiment"]["phenotype"]
    assert set(phenotype) & {"p_value", "rank", "protein_abundance_p_value"} == set()
    ledger = pd.read_csv(Path(built_differential.preprocess_dir) / "differential.csv")
    assert ledger["p_value"].notna().all()
    assert "p-value" in y26.DIFFERENTIAL_NOT_STORED[0].lower()
    assert "Rank" in y26.DIFFERENTIAL_NOT_STORED[1]


def test_the_docx_sourced_values_are_kept_out_of_the_text_audit_loop() -> None:
    """``audit_sourced_value`` reads its artifact as text, so a docx quote cannot be
    found there; the three docx-quoted values live in their own dict.
    """
    assert set(y26.SI_SOURCED_VALUES) == {
        "differential_replicates",
        "differential_down_caption",
        "differential_up_caption",
    }
    assert set(y26.SI_SOURCED_VALUES) & set(y26.SOURCED_VALUES) == set()
    for value in y26.SI_SOURCED_VALUES.values():
        assert value.provenance.sha256 == y26.SI_DOCX_SHA256
        assert value.provenance.source_uri == y26.SI_MIRROR_RELPATH
    for value in y26.SOURCED_VALUES.values():
        assert value.provenance.source_uri == y26.PAPER_MD


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
    synthetic_docx: Path,
    synthetic_benchling: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``--retrieve-into`` the deposit reads the literature mirror's si/ copy."""
    data_root = tmp_path / "root"
    _write_benchling_deposit(data_root / y26.RAW_DIR_REL, synthetic_benchling)
    captured = data_root / y26.LIBRARY_DIR_REL / "si" / y26.SI_DOCX
    captured.parent.mkdir(parents=True)
    captured.write_bytes(synthetic_docx.read_bytes())
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    assert y26.main(["deposit"]) == 0
    assert (data_root / y26.RAW_DIR_REL / y26.SI_MIRROR_RELPATH).exists()


def test_main_deposit_can_re_run_the_recorded_retrieval(
    synthetic_docx: Path,
    synthetic_benchling: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--retrieve-into`` re-runs the retriever and deposits the verified bytes."""
    import torchcell.literature.provenance as provenance

    payload = synthetic_docx.read_bytes()
    monkeypatch.setattr(provenance, "run_retriever", lambda record: payload)
    data_root = tmp_path / "root"
    _write_benchling_deposit(data_root / y26.RAW_DIR_REL, synthetic_benchling)
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
        "crispri_differential_proteome_yunus2026": "data/torchcell/differential",
    }
    for name, rel in roots.items():
        monkeypatch.setitem(y26.DATASETS[name], "root", rel)
    assert y26.main(["build"]) == 0
    out = capsys.readouterr().out
    assert "CrispriKnockdownYunus2026Dataset: len = " in out
    assert "CrispriArrayYunus2026Dataset: len = " in out
    assert "CrispriDifferentialProteomeYunus2026Dataset: len = 1" in out
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


# --------------------------------------------------------------------------- #
# The two hand-deposited Benchling tables (issues #699 and #788 item 2)
# --------------------------------------------------------------------------- #
#: The synthetic deposit's protein columns: ten reach one locus, one reaches two, one
#: reaches none, so both accession drop rules and the 0.81 floor are exercised at once.
PANEL_ACCESSIONS: tuple[str, ...] = tuple(f"UP_A{index:02d}" for index in range(12))
PANEL_SINGLE: dict[str, str] = {
    f"UP_A{index:02d}": tag for index, (tag, _) in enumerate(SCREEN_LOCI[:10])
}
PANEL_MULTI: dict[str, tuple[str, ...]] = {
    "UP_A10": (SCREEN_LOCI[10][0], SCREEN_LOCI[11][0])
}
PANEL_UNMAPPED = "UP_A11"

#: The synthetic deposit's row labels. Twelve match a Table S3 target exactly, one
#: carries the undefined marker over a label whose bare twin is also a row, one matches
#: only after the ``_NT<digit>`` strip, and one is the control.
PANEL_ROWS: tuple[tuple[str, float], ...] = (
    ("PP_0100", 957.5),
    ("PP_0101", 1400.0),
    ("PP_0102", 120.0),
    ("PP_0103", 130.0),
    ("PP_0104", 140.0),
    ("PP_0105", 150.0),
    ("PP_0106_NT2", 160.0),
    ("PP_0106_NT3", 170.0),
    ("PP_0107", 180.0),
    ("PP_0108", 190.0),
    ("PP_0109", 200.0),
    ("PP_0110", 210.0),
    ("PP_0100 (S)", 220.0),
    ("PP_0107_NT1", 230.0),
    ("Control", 845.73),
)
PANEL_MARKED = tuple(label for label, _ in PANEL_ROWS if label.endswith(" (S)"))
PANEL_RECORDS = len(PANEL_ROWS) - 1 - len(PANEL_MARKED)
#: The synthetic correlation table's rows.
CORRELATION_ROWS: tuple[tuple[str, float, float], ...] = (
    ("UP_A00", -0.671135904, 1.30e-18),
    ("UP_A01", 0.459300000, 3.00e-08),
    ("UP_A11", 0.012000000, 0.9988959),
)


def _panel_tsv(
    rows: tuple[tuple[str, float], ...] = PANEL_ROWS,
    accessions: tuple[str, ...] = PANEL_ACCESSIONS,
    header: tuple[str, ...] = ("strain", "isoprenol_production"),
) -> str:
    """The synthetic ``strain, isoprenol_production, <accessions>`` TSV as text.

    One cell per row is a released ``0``, so the zero-is-a-measurement rule is covered,
    and every other cell is a distinct positive number.
    """
    lines = ["\t".join([*header, *accessions])]
    for row_index, (label, titer) in enumerate(rows):
        cells = [
            "0" if column == 0 else f"{row_index * 100 + column * 7 + 1}.5"
            for column in range(len(accessions))
        ]
        lines.append("\t".join([label, repr(titer), *cells]))
    return "\n".join(lines) + "\n"


def _correlation_tsv(
    rows: tuple[tuple[str, float, float], ...] = CORRELATION_ROWS,
    header: tuple[str, ...] = (
        "Protein",
        "Correlation_with_isoprenol_production",
        "p_value",
    ),
) -> str:
    """The synthetic Pearson-output TSV as text."""
    lines = ["\t".join(header)]
    for protein, correlation, p_value in rows:
        lines.append("\t".join([protein, repr(correlation), repr(p_value)]))
    return "\n".join(lines) + "\n"


def _write_benchling_deposit(root: Path, texts: tuple[str, str]) -> None:
    """Write both deposited tables under ``root`` at their mirror-relative paths."""
    panel, correlation = texts
    for relpath, text in (
        (y26.BENCHLING_TITER_REL, panel),
        (y26.BENCHLING_CORRELATION_REL, correlation),
    ):
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


@pytest.fixture
def synthetic_crosswalk() -> Any:
    """A ``UniProtLocusCrosswalk`` over :data:`PANEL_ACCESSIONS`.

    The crosswalk is replaced rather than read: the synthetic assembly fixture carries
    one GAF row, so a real crosswalk over it would resolve almost nothing. Everything
    the loader does WITH the crosswalk, including the split and both drop rules, is the
    code under test.
    """
    from torchcell.datasets.bacteria_common import UniProtLocusCrosswalk

    return UniProtLocusCrosswalk(
        assembly_set=y26.KT2440_ASSEMBLY_SET,
        member="109.P_putida_KT2440.goa",
        sha256="0" * 64,
        rows=len(PANEL_ACCESSIONS),
        single=dict(PANEL_SINGLE),
        multi=dict(PANEL_MULTI),
    )


@pytest.fixture
def synthetic_benchling(
    monkeypatch: pytest.MonkeyPatch, synthetic_crosswalk: Any
) -> tuple[str, str]:
    """Both synthetic deposits' bytes, with every pin they are checked against.

    The pinned digests, row counts and cross-source oracle labels are all module
    constants read at call time, so re-pointing them here re-points the deposit records,
    the shape assertion and the Results-text join at the synthetic bytes.
    """
    panel = _panel_tsv()
    correlation = _correlation_tsv()
    monkeypatch.setattr(
        y26, "BENCHLING_TITER_SHA256", hashlib.sha256(panel.encode()).hexdigest()
    )
    monkeypatch.setattr(
        y26,
        "BENCHLING_CORRELATION_SHA256",
        hashlib.sha256(correlation.encode()).hexdigest(),
    )
    monkeypatch.setattr(y26, "BENCHLING_TITER_ROWS", len(PANEL_ROWS))
    monkeypatch.setattr(y26, "BENCHLING_ACCESSIONS", len(PANEL_ACCESSIONS))
    monkeypatch.setattr(y26, "BENCHLING_CORRELATION_ROWS", len(CORRELATION_ROWS))
    # PP_0100's titer is 957.5 against the real 958 mg/L the Results print, so the
    # asserted agreement holds; PP_0101's 1400.0 is 69 mg/L off the printed 1469 and is
    # the recorded disagreement.
    monkeypatch.setattr(y26, "TITER_ORACLE_AGREES", "PP_0100")
    monkeypatch.setattr(y26, "TITER_ORACLE_DISAGREES", "PP_0101")
    monkeypatch.setattr(
        y26, "uniprot_locus_crosswalk", lambda *a, **k: synthetic_crosswalk
    )
    return panel, correlation


def test_benchling_deposits_name_both_files_with_their_pins() -> None:
    """The deposit list reads its digests at call time, so a test can re-point them."""
    deposits = y26.benchling_deposits()
    assert [relpath for relpath, _, _, _ in deposits] == [
        "data/benchling/strain_isoprenol_production_protein_abundance.tsv",
        "data/benchling/protein_correlation_with_isoprenol_production.tsv",
    ]
    assert [role for _, role, _, _ in deposits] == [ROLE_RAW_DATA, "si_data"]
    assert [digest for _, _, digest, _ in deposits] == [
        y26.BENCHLING_TITER_SHA256,
        y26.BENCHLING_CORRELATION_SHA256,
    ]


def test_benchling_artifact_records_carry_the_paste_caveat_and_the_recipe(
    synthetic_benchling: tuple[str, str], tmp_path: Path
) -> None:
    """Every record is manual_browser, and its params carry the deposit's own words."""
    root = tmp_path / "mirror"
    _write_benchling_deposit(root, synthetic_benchling)
    records = y26.benchling_artifact_records(root)
    assert len(records) == 2
    for record in records:
        retrieval = record.retrieval
        assert retrieval is not None
        assert retrieval.method is RetrievalMethod.manual_browser
        assert retrieval.sha256 == record.sha256
        params = retrieval.params
        assert params["retrieval_command"] == y26.BENCHLING_MANUAL_RECIPE
        assert params["paste_caveat"] == y26.BENCHLING_PASTE_CAVEAT
        assert params["page_mapping"] == y26.BENCHLING_PAGE_MAPPING
        assert params["retrieved_by"] == y26.BENCHLING_RETRIEVED_BY
        assert params["deposit_record"] == "data/benchling/DEPOSIT.md"
        assert params["checksums"] == "data/benchling/SHA256SUMS.txt"
    assert records[0].source == y26.BENCHLING_INPUT_URL
    assert records[1].source == y26.BENCHLING_ANALYSIS_URL


def test_benchling_artifact_records_name_the_recipe_when_a_file_is_absent(
    synthetic_benchling: tuple[str, str], tmp_path: Path
) -> None:
    """An absent manual deposit cannot be refetched, so the refusal states the recipe."""
    root = tmp_path / "mirror"
    _write_benchling_deposit(root, synthetic_benchling)
    (root / y26.BENCHLING_CORRELATION_REL).unlink()
    with pytest.raises(RuntimeError, match="MANUAL RECIPE"):
        y26.benchling_artifact_records(root)


def test_benchling_artifact_records_refuse_drifted_bytes(
    synthetic_benchling: tuple[str, str], tmp_path: Path
) -> None:
    """Altered deposited bytes are a NEW provenance record, never a silent update."""
    root = tmp_path / "mirror"
    _write_benchling_deposit(root, synthetic_benchling)
    path = root / y26.BENCHLING_TITER_REL
    path.write_text(path.read_text() + "PP_0111\t1.0" + "\t1.0" * 12 + "\n")
    with pytest.raises(RuntimeError, match="NEW provenance record"):
        y26.benchling_artifact_records(root)


def test_deposit_raw_mirror_records_both_manual_deposits(
    synthetic_mirror: Path,
) -> None:
    """The mirror manifest carries the docx and both deposits, and nothing else."""
    manifest = y26.load_manifest(str(synthetic_mirror))
    assert [record.path for record in manifest.files] == [
        y26.SI_MIRROR_RELPATH,
        y26.BENCHLING_TITER_REL,
        y26.BENCHLING_CORRELATION_REL,
    ]
    for record in manifest.files:
        assert record.retrieval is not None
        assert record.retrieval.sha256 == record.sha256
    assert any(
        "RECORDED ONLY" in entry and "p_value" in entry
        for entry in manifest.si_expected
    )
    assert any(
        "IS loaded" in entry and y26.BENCHLING_TITER_REL in entry
        for entry in manifest.si_expected
    )
    assert not any("the family is not built" in entry for entry in manifest.si_expected)


# --------------------------------------------------------------------------- #
# The deposited readers and every refusal branch
# --------------------------------------------------------------------------- #
def test_read_benchling_strain_table_types_every_row(tmp_path: Path) -> None:
    """Labels, the titer and all twelve abundances, with the control separated out."""
    path = tmp_path / "panel.tsv"
    path.write_text(_panel_tsv())
    table = y26.read_benchling_strain_table(path)
    assert table.accessions == PANEL_ACCESSIONS
    assert len(table.rows) == len(PANEL_ROWS)
    assert len(table.strains) == len(PANEL_ROWS) - 1
    assert table.control.label == "Control"
    assert table.control.isoprenol_production == 845.73
    assert table.zero_cells == len(PANEL_ROWS)
    assert [row.label for row in table.rows if row.marked] == list(PANEL_MARKED)
    assert table.titer_by_label["PP_0100"] == 957.5


def test_read_benchling_strain_table_refuses_a_changed_header(tmp_path: Path) -> None:
    """The first two columns are the file's contract with Supplementary Note 1's script."""
    path = tmp_path / "panel.tsv"
    path.write_text(_panel_tsv(header=("strain", "isoprenol_titer")))
    with pytest.raises(y26.TableExtractionError, match="isoprenol_production"):
        y26.read_benchling_strain_table(path)


def test_read_benchling_strain_table_refuses_a_repeated_protein_column(
    tmp_path: Path,
) -> None:
    """A repeated accession would key one gene from two columns."""
    path = tmp_path / "panel.tsv"
    path.write_text(_panel_tsv(accessions=PANEL_ACCESSIONS[:-1] + ("UP_A00",)))
    with pytest.raises(y26.TableExtractionError, match="repeats the protein"):
        y26.read_benchling_strain_table(path)


def test_read_benchling_strain_table_refuses_a_repeated_strain_row(
    tmp_path: Path,
) -> None:
    """One row is one strain, so a repeat would store two titers for one identity."""
    path = tmp_path / "panel.tsv"
    path.write_text(_panel_tsv(rows=PANEL_ROWS + (("PP_0100", 1.0),)))
    with pytest.raises(y26.TableExtractionError, match="repeats the strain row"):
        y26.read_benchling_strain_table(path)


def test_read_benchling_strain_table_refuses_a_table_with_no_control_row(
    tmp_path: Path,
) -> None:
    """The control row is the reference of both families and nothing else supplies one."""
    path = tmp_path / "panel.tsv"
    path.write_text(_panel_tsv(rows=PANEL_ROWS[:-1]))
    with pytest.raises(y26.TableExtractionError, match="0 rows labeled 'Control'"):
        y26.read_benchling_strain_table(path)


def test_read_benchling_strain_table_refuses_a_blank_cell(tmp_path: Path) -> None:
    """A blank is an absence this loader has no sourced rule for, so it stops."""
    path = tmp_path / "panel.tsv"
    text = _panel_tsv()
    lines = text.split("\n")
    cells = lines[1].split("\t")
    cells[3] = ""
    lines[1] = "\t".join(cells)
    path.write_text("\n".join(lines))
    with pytest.raises(y26.TableExtractionError, match="is blank"):
        y26.read_benchling_strain_table(path)


def test_read_benchling_strain_table_refuses_a_short_row(tmp_path: Path) -> None:
    """A row with fewer cells than the header is a changed release."""
    path = tmp_path / "panel.tsv"
    lines = _panel_tsv().split("\n")
    lines[1] = "\t".join(lines[1].split("\t")[:-1])
    path.write_text("\n".join(lines))
    with pytest.raises(y26.TableExtractionError, match="cells, header has"):
        y26.read_benchling_strain_table(path)


def test_read_benchling_strain_table_refuses_a_negative_titer(tmp_path: Path) -> None:
    """A titer is a measured amount."""
    path = tmp_path / "panel.tsv"
    path.write_text(_panel_tsv(rows=(("PP_0100", -1.0), ("Control", 1.0))))
    with pytest.raises(y26.TableExtractionError, match="isoprenol_production"):
        y26.read_benchling_strain_table(path)


def test_read_benchling_correlation_table_types_every_row(tmp_path: Path) -> None:
    """The recorded Pearson output parses to its three typed columns."""
    path = tmp_path / "corr.tsv"
    path.write_text(_correlation_tsv())
    rows = y26.read_benchling_correlation_table(path)
    assert [row.protein for row in rows] == [p for p, _, _ in CORRELATION_ROWS]
    assert rows[0].correlation == -0.671135904
    assert rows[2].p_value == 0.9988959


def test_read_benchling_correlation_table_refuses_a_changed_header(
    tmp_path: Path,
) -> None:
    """The three columns are what Supplementary Note 1's script writes."""
    path = tmp_path / "corr.tsv"
    path.write_text(_correlation_tsv(header=("Protein", "r", "p_value")))
    with pytest.raises(y26.TableExtractionError, match="header is"):
        y26.read_benchling_correlation_table(path)


def test_read_benchling_correlation_table_refuses_an_out_of_range_correlation(
    tmp_path: Path,
) -> None:
    """A Pearson r outside [-1, 1] is not a correlation."""
    path = tmp_path / "corr.tsv"
    path.write_text(_correlation_tsv(rows=(("UP_A00", 1.5, 0.01),)))
    with pytest.raises(y26.TableExtractionError, match=r"outside \[-1, 1\]"):
        y26.read_benchling_correlation_table(path)


def test_read_benchling_correlation_table_refuses_an_out_of_range_p_value(
    tmp_path: Path,
) -> None:
    """A p-value outside [0, 1] is not a p-value."""
    path = tmp_path / "corr.tsv"
    path.write_text(_correlation_tsv(rows=(("UP_A00", 0.5, 1.5),)))
    with pytest.raises(y26.TableExtractionError, match=r"outside \[0, 1\]"):
        y26.read_benchling_correlation_table(path)


def test_read_benchling_correlation_table_refuses_a_repeated_protein(
    tmp_path: Path,
) -> None:
    """One row is one protein's statistic."""
    path = tmp_path / "corr.tsv"
    path.write_text(_correlation_tsv(rows=CORRELATION_ROWS + CORRELATION_ROWS[:1]))
    with pytest.raises(y26.TableExtractionError, match="repeats the protein row"):
        y26.read_benchling_correlation_table(path)


# --------------------------------------------------------------------------- #
# Reconciling the deposited row labels
# --------------------------------------------------------------------------- #
def test_reconcile_benchling_labels_splits_by_the_three_documented_routes() -> None:
    """Exact, marker-stripped, variant-stripped, and what no route reaches."""
    reconciliation = y26.reconcile_benchling_labels(
        ["PP_0100", "PP_0100 (S)", "PP_0107_NT1", "Control"], {"PP_0100", "PP_0107"}
    )
    assert reconciliation.exact == ("PP_0100",)
    assert reconciliation.after_marker_strip == ("PP_0100 (S)",)
    assert reconciliation.after_variant_strip == ("PP_0107_NT1",)
    assert reconciliation.unmatched == ("Control",)
    assert reconciliation.rows == 4
    assert reconciliation.known_targets == 2


def test_reconcile_benchling_labels_prefers_the_released_label_over_a_strip() -> None:
    """A label the lists already name is matched as released, not re-parsed."""
    reconciliation = y26.reconcile_benchling_labels(
        ["PP_0106_NT2"], {"PP_0106_NT2", "PP_0106"}
    )
    assert reconciliation.exact == ("PP_0106_NT2",)
    assert reconciliation.after_variant_strip == ()


def test_benchling_target_reads_the_variant_as_part_of_the_key() -> None:
    """``NT<n>`` is this paper's guide-variant number, measured on its own Table S7."""
    assert y26.benchling_target("PP_0106_NT2") == ("PP_0106", "NT2")
    assert y26.benchling_target("PP_0100") == ("PP_0100", None)


def test_benchling_target_refuses_a_label_it_cannot_key() -> None:
    """A label this loader cannot parse would store a titer against a guessed gene."""
    with pytest.raises(y26.TableExtractionError, match="optional"):
        y26.benchling_target("Control")


# --------------------------------------------------------------------------- #
# The build-time cross-source proofs
# --------------------------------------------------------------------------- #
def test_the_results_text_join_asserts_one_titer_and_records_the_other(
    synthetic_benchling: tuple[str, str], tmp_path: Path
) -> None:
    """One agreement is asserted; the other difference is recorded, not repaired."""
    path = tmp_path / "panel.tsv"
    path.write_text(synthetic_benchling[0])
    proofs = y26.assert_benchling_titers_match_the_results_text(
        y26.read_benchling_strain_table(path)
    )
    assert "inside the 1.0 mg/L the paper prints to" in proofs[0]
    assert proofs[1].startswith("MEASURED DISAGREEMENT, kept:")
    assert "69.0 mg/L" in proofs[1]


def test_the_results_text_join_refuses_a_deposit_that_moved_too_far(
    synthetic_benchling: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deposit whose asserted titer drifts past the printed precision is not this campaign."""
    path = tmp_path / "panel.tsv"
    path.write_text(_panel_tsv(rows=(("PP_0100", 900.0), ("Control", 1.0))))
    with pytest.raises(RuntimeError, match="no longer the same campaign"):
        y26.assert_benchling_titers_match_the_results_text(
            y26.read_benchling_strain_table(path)
        )


def test_the_results_text_join_refuses_a_deposit_missing_an_oracle_row(
    synthetic_benchling: tuple[str, str], tmp_path: Path
) -> None:
    """Without the oracle row the only cross-source check there is cannot run."""
    path = tmp_path / "panel.tsv"
    path.write_text(_panel_tsv(rows=(("PP_0102", 1.0), ("Control", 1.0))))
    with pytest.raises(RuntimeError, match="cannot run"):
        y26.assert_benchling_titers_match_the_results_text(
            y26.read_benchling_strain_table(path)
        )


def test_the_deposit_shape_proof_names_the_shared_protein_set(
    synthetic_benchling: tuple[str, str], tmp_path: Path
) -> None:
    """The two deposits cover different protein sets, which the proof states."""
    panel_path = tmp_path / "panel.tsv"
    panel_path.write_text(synthetic_benchling[0])
    corr_path = tmp_path / "corr.tsv"
    corr_path.write_text(synthetic_benchling[1])
    proofs = y26.assert_benchling_deposit_shape(
        y26.read_benchling_strain_table(panel_path),
        y26.read_benchling_correlation_table(corr_path),
    )
    assert "no blank cell" in proofs[0]
    assert (
        f"shares {len(CORRELATION_ROWS)} of the panel's {len(PANEL_ACCESSIONS)}"
        in (proofs[1])
    )


def test_the_deposit_shape_proof_refuses_a_changed_row_count(
    synthetic_benchling: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pinned shape is asserted, so a re-paste that gained a row stops the build."""
    panel_path = tmp_path / "panel.tsv"
    panel_path.write_text(synthetic_benchling[0])
    corr_path = tmp_path / "corr.tsv"
    corr_path.write_text(synthetic_benchling[1])
    monkeypatch.setattr(y26, "BENCHLING_TITER_ROWS", len(PANEL_ROWS) + 1)
    with pytest.raises(y26.TableExtractionError, match="rows, pinned"):
        y26.assert_benchling_deposit_shape(
            y26.read_benchling_strain_table(panel_path),
            y26.read_benchling_correlation_table(corr_path),
        )


# --------------------------------------------------------------------------- #
# The deposited phenotypes
# --------------------------------------------------------------------------- #
def test_isoprenol_titer_phenotype_stores_the_number_under_the_identical_unit() -> None:
    """mg/L is stored as ug/mL verbatim, with the uncertainty a typed gap."""
    phenotype = y26.isoprenol_titer_phenotype(957.246595)
    assert phenotype.titer == 957.246595
    assert phenotype.titer_unit is ConcentrationUnit.ug_per_ml
    assert phenotype.titer_uncertainty is None
    assert phenotype.titer_se is None
    assert phenotype.n_samples == 3
    assert phenotype.sample_unit is not None
    assert phenotype.sample_unit.value == "biological_replicate"
    assert {gap.field for gap in phenotype.provenance_gaps} == {
        "titer_uncertainty",
        "titer_uncertainty_type",
        "titer_se",
        "product_yield",
        "product_yield_unit",
        "productivity",
        "productivity_unit",
    }


def test_isoprenol_titer_phenotype_takes_the_conservative_end_of_the_range() -> None:
    """The caption releases 3 to 6; a back-solve is precluded, so 3 is stored."""
    assert y26.TITER_REPLICATE_RANGE == (3, 6)
    assert y26.TITER_N_REPLICATES == 3
    assert y26.isoprenol_titer_phenotype(1.0).n_samples == y26.TITER_REPLICATE_RANGE[0]


def test_isoprenol_titer_phenotype_refuses_a_negative_number() -> None:
    """A titer is a measured amount."""
    with pytest.raises(RuntimeError, match="not a measured amount"):
        y26.isoprenol_titer_phenotype(-1.0)


def test_panel_proteome_phenotype_keeps_a_released_zero_and_names_its_scale() -> None:
    """A released 0 is a present measurement, and the scale is its own string."""
    phenotype = y26.panel_proteome_phenotype({"PP_0100": 0.0, "PP_0101": 12.5})
    assert phenotype.protein_abundance == {"PP_0100": 0.0, "PP_0101": 12.5}
    assert phenotype.measurement_type == "dia_nn_top3_signal_benchling_displayed"
    assert phenotype.n_replicates == {"PP_0100": 1, "PP_0101": 1}
    assert phenotype.protein_abundance_se is None
    note = phenotype.provenance_gaps[0].note
    assert note is not None and y26.BENCHLING_PASTE_CAVEAT in note


def test_panel_proteome_measurement_type_is_not_any_other_proteome_scale() -> None:
    """Heterogeneous proteomics is never pooled, so the strings must all differ."""
    from torchcell.datasets.pputida import carruthers2025 as c25

    others = {
        y26.MEASUREMENT_TYPE,
        y26.DIFFERENTIAL_MEASUREMENT_TYPE,
        c25.PROTEOME_MEASUREMENT_TYPE,
        c25.CAMPAIGN_MEASUREMENT_TYPE,
    }
    assert y26.PANEL_PROTEOME_MEASUREMENT_TYPE not in others


def test_panel_proteome_phenotype_refuses_an_empty_profile() -> None:
    """A record with no protein is not a proteome."""
    with pytest.raises(RuntimeError, match="at least one protein"):
        y26.panel_proteome_phenotype({})


def test_panel_proteome_phenotype_refuses_a_negative_signal() -> None:
    """An absolute Top3 signal cannot be negative."""
    with pytest.raises(RuntimeError, match="not a measured signal"):
        y26.panel_proteome_phenotype({"PP_0100": -1.0})


def test_titer_environment_keeps_the_vessel_a_typed_gap() -> None:
    """The Methods state the volume and never the container, so only the volume is set."""
    environment = y26.titer_environment()
    assert environment.culture_format is not None
    assert environment.culture_format.working_volume_ul == 5000.0
    assert environment.culture_format.shaking_rpm == 180.0
    assert environment.culture_format.inoculum_od600 == 0.2
    assert environment.culture_format.vessel is None
    assert [gap.field for gap in environment.culture_format.provenance_gaps] == [
        "vessel"
    ]
    assert environment.media == y26.production_environment().media
    assert environment.duration_hours == 48.0


def test_titer_environment_survives_the_dump_the_base_environment_loses() -> None:
    """A CultureEnvironment in the narrowed slot keeps what Environment drops."""
    assert "culture_format" in y26.titer_environment().model_dump()
    assert "culture_format" not in y26.production_environment().model_dump()


# --------------------------------------------------------------------------- #
# Building both deposited families on synthetic bytes
# --------------------------------------------------------------------------- #
def test_titer_dataset_builds_one_record_per_deposited_strain_row(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control row is the reference and the marked rows are dropped by their rule."""
    dataset = _build(
        y26.IsoprenolTiterYunus2026Dataset,
        tmp_path / "titer",
        synthetic_kt2440,
        monkeypatch,
    )
    assert len(dataset) == PANEL_RECORDS
    log = json.loads(Path(dataset.preprocess_dir, "dropped_records.json").read_text())
    assert log["source_rows"] == len(PANEL_ROWS)
    assert log["candidate_records"] == len(PANEL_ROWS) - 1
    assert log["kept_records"] == PANEL_RECORDS
    assert log["dropped_records"] == len(PANEL_MARKED)
    assert [rule["rule"] for rule in log["rules"]] == [
        "row_label_carries_an_undefined_marker"
    ]
    assert log["rules"][0]["items"] == list(PANEL_MARKED)
    assert log["rules"][0]["scope"] == "record"


def test_the_marker_drop_rule_names_the_search_that_found_no_definition(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A marker nothing defines is dropped WITH the evidence that nothing defines it."""
    dataset = _build(
        y26.IsoprenolTiterYunus2026Dataset,
        tmp_path / "titer",
        synthetic_kt2440,
        monkeypatch,
    )
    log = json.loads(Path(dataset.preprocess_dir, "dropped_records.json").read_text())
    description = log["rules"][0]["description"]
    assert y26.BENCHLING_MARKER_SEARCH in description
    assert "The unmarked twin is kept" in description


def test_titer_record_stores_the_deposited_number_against_the_control_row(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Experiment titer is the row's number; the reference is the control's, not 1.0."""
    dataset = _build(
        y26.IsoprenolTiterYunus2026Dataset,
        tmp_path / "titer",
        synthetic_kt2440,
        monkeypatch,
    )
    titers = sorted(
        _dump(dataset[index]["experiment"])["phenotype"]["titer"]
        for index in range(len(dataset))
    )
    assert titers == sorted(
        titer
        for label, titer in PANEL_ROWS
        if label != "Control" and not label.endswith(" (S)")
    )
    reference = _dump(dataset[0]["reference"])
    assert reference["phenotype_reference"]["titer"] == 845.73
    assert reference["environment_reference"]["culture_format"]["shaking_rpm"] == 180.0


def test_titer_record_genotype_is_one_crispri_knockdown_of_its_label(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One CRISPRi leaf per record, on the locus the row label names."""
    dataset = _build(
        y26.IsoprenolTiterYunus2026Dataset,
        tmp_path / "titer",
        synthetic_kt2440,
        monkeypatch,
    )
    rows = pd.read_csv(Path(dataset.preprocess_dir, "benchling_titers.csv"))
    assert set(rows["locus_tag"]) == {
        y26.benchling_target(label)[0]
        for label, _ in PANEL_ROWS
        if label != "Control" and not label.endswith(" (S)")
    }
    for index in range(len(dataset)):
        genotype = _dump(dataset[index]["experiment"])["genotype"]
        assert len(genotype["perturbations"]) == 1
        assert (
            genotype["perturbations"][0]["perturbation_type"]
            == "bacterial_crispr_interference"
        )


def test_titer_build_writes_the_label_reconciliation_and_the_proofs(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every deposited label is in the ledger with the route that reached it."""
    dataset = _build(
        y26.IsoprenolTiterYunus2026Dataset,
        tmp_path / "titer",
        synthetic_kt2440,
        monkeypatch,
    )
    rows = pd.read_csv(Path(dataset.preprocess_dir, "label_reconciliation.csv"))
    assert len(rows) == len(PANEL_ROWS)
    counts = rows["route"].value_counts().to_dict()
    assert counts["exact"] == 12
    assert counts["after_marker_strip"] == 1
    assert counts["after_variant_strip"] == 1
    assert counts["unmatched_reference_row"] == 1
    proofs = json.loads(
        Path(dataset.preprocess_dir, "benchling_proofs.json").read_text()
    )
    assert any(p.startswith("MEASURED DISAGREEMENT, kept:") for p in proofs)


def test_the_build_refuses_a_deposited_label_no_route_reaches(
    synthetic_benchling: tuple[str, str],
    synthetic_docx: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A label outside the target lists would store a titer against a guessed gene."""
    panel = _panel_tsv(rows=PANEL_ROWS + (("PP_0999", 1.0),))
    monkeypatch.setattr(
        y26, "BENCHLING_TITER_SHA256", hashlib.sha256(panel.encode()).hexdigest()
    )
    monkeypatch.setattr(y26, "BENCHLING_TITER_ROWS", len(PANEL_ROWS) + 1)
    data_root = tmp_path / "data_root"
    _write_benchling_deposit(
        data_root / y26.RAW_DIR_REL, (panel, synthetic_benchling[1])
    )
    mirror = y26.deposit_raw_mirror(source=synthetic_docx, data_root=str(data_root))
    assert (mirror / "manifest.json").is_file()
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    with pytest.raises(RuntimeError, match="reach no target of Tables S1, S2 or S3"):
        _build(
            y26.IsoprenolTiterYunus2026Dataset,
            tmp_path / "titer",
            synthetic_kt2440,
            monkeypatch,
        )


def test_panel_proteome_dataset_keys_every_record_by_locus_tag(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ten of twelve accessions resolve, so every record carries those ten loci."""
    dataset = _build(
        y26.CrispriPanelProteomeYunus2026Dataset,
        tmp_path / "panel",
        synthetic_kt2440,
        monkeypatch,
    )
    assert len(dataset) == PANEL_RECORDS
    phenotype = _dump(dataset[0]["experiment"])["phenotype"]
    assert set(phenotype["protein_abundance"]) == set(PANEL_SINGLE.values())
    assert phenotype["measurement_type"] == y26.PANEL_PROTEOME_MEASUREMENT_TYPE
    reference = _dump(dataset[0]["reference"])["phenotype_reference"]
    assert set(reference["protein_abundance"]) == set(PANEL_SINGLE.values())
    assert set(reference["protein_abundance"].values()) != {1.0}


def test_panel_proteome_build_lists_both_accession_drop_rules(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unmapped and multi-locus accessions are two named rules of accession scope."""
    dataset = _build(
        y26.CrispriPanelProteomeYunus2026Dataset,
        tmp_path / "panel",
        synthetic_kt2440,
        monkeypatch,
    )
    log = json.loads(Path(dataset.preprocess_dir, "dropped_records.json").read_text())
    by_rule = {rule["rule"]: rule for rule in log["rules"]}
    assert by_rule["no_locus_tag_in_the_goa_proteome_file"]["items"] == [PANEL_UNMAPPED]
    assert by_rule["no_locus_tag_in_the_goa_proteome_file"]["scope"] == (
        "protein_accession"
    )
    assert by_rule["no_locus_tag_in_the_goa_proteome_file"]["n_records"] == 0
    assert by_rule["accession_names_several_loci"]["items"] == list(PANEL_MULTI)
    assert by_rule["accession_names_several_loci"]["scope"] == "protein_accession"
    dropped = pd.read_csv(Path(dataset.preprocess_dir, "dropped_accessions.csv"))
    assert set(dropped["accession"]) == {PANEL_UNMAPPED, *PANEL_MULTI}


def test_panel_proteome_build_keeps_every_released_zero(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One cell per deposited row is a released 0 and every kept record carries it."""
    dataset = _build(
        y26.CrispriPanelProteomeYunus2026Dataset,
        tmp_path / "panel",
        synthetic_kt2440,
        monkeypatch,
    )
    rows = pd.read_csv(Path(dataset.preprocess_dir, "benchling_panel.csv"))
    assert set(rows["n_zero"]) == {1}
    assert set(rows["n_proteins"]) == {len(PANEL_SINGLE)}


def test_panel_proteome_build_refuses_a_resolution_below_its_floor(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    synthetic_crosswalk: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crosswalk that stopped reaching the loci is a changed annotation, not a build."""
    from torchcell.datasets.bacteria_common import LocusTagResolutionError

    thin = synthetic_crosswalk.model_copy(
        update={"single": dict(list(PANEL_SINGLE.items())[:4])}
    )
    monkeypatch.setattr(y26, "uniprot_locus_crosswalk", lambda *a, **k: thin)
    with pytest.raises(LocusTagResolutionError, match="below 0.81"):
        _build(
            y26.CrispriPanelProteomeYunus2026Dataset,
            tmp_path / "panel",
            synthetic_kt2440,
            monkeypatch,
        )


def test_panel_proteome_build_refuses_two_accessions_reaching_one_locus(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    synthetic_crosswalk: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A collision would store two proteins' abundance under one gene."""
    collided = dict(PANEL_SINGLE)
    collided["UP_A09"] = collided["UP_A00"]
    monkeypatch.setattr(
        y26,
        "uniprot_locus_crosswalk",
        lambda *a, **k: synthetic_crosswalk.model_copy(update={"single": collided}),
    )
    with pytest.raises(RuntimeError, match="name one locus from several"):
        _build(
            y26.CrispriPanelProteomeYunus2026Dataset,
            tmp_path / "panel",
            synthetic_kt2440,
            monkeypatch,
        )


def test_the_deposited_loaders_refuse_a_mirror_whose_deposit_is_absent(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``download`` names the missing mirror artifact rather than building without it."""
    (synthetic_mirror / y26.RAW_DIR_REL / y26.BENCHLING_TITER_REL).unlink()
    with pytest.raises(RuntimeError, match="missing from mirror"):
        _build(
            y26.IsoprenolTiterYunus2026Dataset,
            tmp_path / "titer",
            synthetic_kt2440,
            monkeypatch,
        )


# --------------------------------------------------------------------------- #
# The L0-L4 battery of both deposited families on the synthetic build
# --------------------------------------------------------------------------- #
def test_titer_verification_passes_on_a_synthetic_build(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared titer gate plus both of this family's own rules pass on the LMDB."""
    root = tmp_path / "titer"
    _build(y26.IsoprenolTiterYunus2026Dataset, root, synthetic_kt2440, monkeypatch)
    report = y26.verify_build(str(root), str(synthetic_mirror), family="titer")
    assert report.passed, report.summary()
    names = {result.name for result in report.results}
    assert "titer_reference_is_one_released_control" in names
    assert "titers_are_the_deposited_column" in names
    oracle = next(
        r for r in report.results if r.name == "titers_are_the_deposited_column"
    )
    assert oracle.level is Level.L4
    assert oracle.details["control_titer"] == 845.73
    assert oracle.details["n_stored"] == PANEL_RECORDS
    assert (
        json.loads(Path(root, "preprocess", "verification_report.json").read_text())[
            "dataset_name"
        ]
        == "IsoprenolTiterYunus2026Dataset"
    )


def test_panel_proteome_verification_passes_on_a_synthetic_build(
    synthetic_mirror: Path,
    synthetic_kt2440: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared protein gate plus all three of this family's own rules pass."""
    root = tmp_path / "panel"
    _build(
        y26.CrispriPanelProteomeYunus2026Dataset, root, synthetic_kt2440, monkeypatch
    )
    monkeypatch.setattr(y26, "bacterial_genome", lambda *a, **k: synthetic_kt2440)
    report = y26.verify_build(str(root), str(synthetic_mirror), family="panel_proteome")
    assert report.passed, report.summary()
    oracle = next(
        r
        for r in report.results
        if r.name == "panel_profiles_are_the_deposited_columns"
    )
    assert oracle.level is Level.L4
    assert oracle.details["n_resolved_accessions"] == len(PANEL_SINGLE)
    assert oracle.details["n_zeros_kept"] == PANEL_RECORDS
    shared = next(r for r in report.results if r.name == "panel_key_set_is_shared")
    assert shared.level is Level.L1


def test_the_titer_l4_oracle_catches_a_rescaled_titer(synthetic_mirror: Path) -> None:
    """A stored titer that is not the deposited number would silently rescale the label."""
    result = y26._l4_titers_are_the_deposited_column(
        [
            {
                "experiment": {"phenotype": {"titer": 1.0}},
                "reference": {"phenotype_reference": {"titer": 2.0}},
            }
        ],
        str(synthetic_mirror),
    )
    assert not result.passed


def test_the_titer_l4_oracle_refuses_a_drifted_deposit(synthetic_mirror: Path) -> None:
    """The oracle re-verifies the deposit's sha256 before reading it."""
    path = synthetic_mirror / y26.RAW_DIR_REL / y26.BENCHLING_TITER_REL
    path.write_text(path.read_text().replace("845.73", "900.0"))
    with pytest.raises(RuntimeError, match="cannot read a deposit that drifted"):
        y26._l4_titers_are_the_deposited_column([], str(synthetic_mirror))


def test_the_titer_l3_rule_catches_a_second_reference_titer() -> None:
    """Two reference titers would mean the build invented one."""
    records = [
        {"reference": {"phenotype_reference": {"titer": 845.73}}},
        {"reference": {"phenotype_reference": {"titer": 1.0}}},
    ]
    assert not y26._l3_titer_reference_is_one_released_control(records).passed


def test_the_panel_l3_rule_catches_a_denominator_reference() -> None:
    """An all-1.0 reference is the ratio families' denominator, not a measured control."""
    records = [
        {"reference": {"phenotype_reference": {"protein_abundance": {"PP_0100": 1.0}}}}
    ]
    assert not y26._l3_panel_reference_is_a_measured_control(records).passed


def test_the_panel_l1_rule_catches_a_record_with_its_own_key_set() -> None:
    """Every deposited row carries every column, so one key set is the whole family."""
    records = [
        {"experiment": {"phenotype": {"protein_abundance": {"PP_0100": 1.0}}}},
        {
            "experiment": {
                "phenotype": {"protein_abundance": {"PP_0100": 1.0, "PP_0101": 2.0}}
            }
        },
    ]
    assert not y26._l1_panel_key_set_is_shared(records).passed


def test_bioproduction_provenance_names_the_deposit_each_family_consumes() -> None:
    """The verifier's provenance points at the deposited bytes and their caveat."""
    for name in y26.BIOPRODUCTION_DATASETS:
        provenance = y26.bioproduction_provenance(name)
        assert provenance.source_uri == y26.BENCHLING_TITER_REL
        assert provenance.sha256 == y26.BENCHLING_TITER_SHA256
        method = provenance.method
        assert method is not None and y26.BENCHLING_PASTE_CAVEAT in method


def test_all_datasets_holds_every_family_this_module_serves() -> None:
    """The CLI builds and verifies all five, through two verification entry points."""
    assert set(y26.ALL_DATASETS) == set(y26.DATASETS) | set(y26.BIOPRODUCTION_DATASETS)
    assert set(y26.BIOPRODUCTION_DATASETS) == {
        "isoprenol_titer_yunus2026",
        "crispri_panel_proteome_yunus2026",
    }
    assert {spec["family"] for spec in y26.BIOPRODUCTION_DATASETS.values()} == {
        "titer",
        "panel_proteome",
    }


# --------------------------------------------------------------------------- #
# The real manual deposit, pinned (data-gated)
# --------------------------------------------------------------------------- #
def _deposit_or_skip() -> Path:
    """The real raw-mirror directory, or skip when the deposit is not present."""
    root = y26.raw_mirror_dir(_data_root_or_skip())
    if not (root / y26.BENCHLING_TITER_REL).exists():
        pytest.skip("the Yunus 2026 Benchling deposit is not present")
    return root


@pytest.mark.data
def test_real_deposit_matches_its_pinned_digests_and_sizes() -> None:
    """Both deposited tables are the bytes this module was written against."""
    root = _deposit_or_skip()
    manifest = y26.load_manifest(str(root.parent.parent))
    for relpath, _, expected, _ in y26.benchling_deposits():
        path = root / relpath
        assert _sha256_bytes(path) == expected, relpath
        assert y26.manifest_sha256(manifest, relpath) == expected, relpath
    assert (root / y26.BENCHLING_TITER_REL).stat().st_size == y26.BENCHLING_TITER_BYTES
    assert (
        root / y26.BENCHLING_CORRELATION_REL
    ).stat().st_size == y26.BENCHLING_CORRELATION_BYTES


@pytest.mark.data
def test_real_deposit_holds_the_shape_the_pins_describe() -> None:
    """132 rows x 255 columns, 253 accessions, 2,659 correlation rows, zero blanks."""
    root = _deposit_or_skip()
    panel = y26.read_benchling_strain_table(root / y26.BENCHLING_TITER_REL)
    assert len(panel.rows) == y26.BENCHLING_TITER_ROWS == 132
    assert len(panel.accessions) == y26.BENCHLING_ACCESSIONS == 253
    assert len(panel.accessions) + 2 == y26.BENCHLING_TITER_COLUMNS == 255
    assert panel.zero_cells == 5978
    correlation = y26.read_benchling_correlation_table(
        root / y26.BENCHLING_CORRELATION_REL
    )
    assert len(correlation) == y26.BENCHLING_CORRELATION_ROWS == 2659
    assert max(row.p_value for row in correlation) > 0.05


@pytest.mark.data
def test_real_deposit_titers_span_the_measured_range() -> None:
    """The deposited column's extremes and the control row, as measured 2026-10-09."""
    panel = y26.read_benchling_strain_table(
        _deposit_or_skip() / y26.BENCHLING_TITER_REL
    )
    titers = panel.titer_by_label
    assert panel.control.isoprenol_production == 845.73
    assert min(titers.values()) == titers["PP_5203"] == 1.58969662
    assert max(titers.values()) == titers["PP_4188"] == 1494.98874
    assert titers["PP_0168"] == 957.246595


@pytest.mark.data
def test_real_deposit_labels_reconcile_123_6_2_and_the_control() -> None:
    """The reconciliation measured 2026-10-09 against Tables S1, S2 and S3."""
    root = _deposit_or_skip()
    tables = y26.supplementary_tables(root / y26.SI_MIRROR_RELPATH)
    screen = y26.parse_table_s3(tables["S3"])
    known = {row.target for row in screen} | {row.locus_tag for row in screen}
    known |= {
        row.locus_tag
        for table in ("S1", "S2")
        for row in y26.parse_target_list(tables[table], table=table)
    }
    panel = y26.read_benchling_strain_table(root / y26.BENCHLING_TITER_REL)
    reconciliation = y26.reconcile_benchling_labels(
        [row.label for row in panel.rows], known
    )
    assert len(reconciliation.exact) == y26.BENCHLING_EXACT_LABELS == 123
    assert len(reconciliation.after_marker_strip) == y26.BENCHLING_MARKER_LABELS == 6
    assert len(reconciliation.after_variant_strip) == y26.BENCHLING_VARIANT_LABELS == 2
    assert reconciliation.unmatched == (y26.BENCHLING_CONTROL_LABEL,)
    assert sorted(reconciliation.after_marker_strip) == sorted(
        y26.BENCHLING_MARKED_LABELS
    )
    assert sorted(reconciliation.after_variant_strip) == ["PP_1607_NT1", "PP_1607_NT3"]


@pytest.mark.data
def test_real_marked_labels_all_have_an_unmarked_twin_in_the_table() -> None:
    """Dropping the marked rows loses no target, which is why the rule is safe."""
    panel = y26.read_benchling_strain_table(
        _deposit_or_skip() / y26.BENCHLING_TITER_REL
    )
    labels = {row.label for row in panel.rows}
    for marked in y26.BENCHLING_MARKED_LABELS:
        assert marked in labels
        assert marked[: -len(y26.BENCHLING_SOLID_MARKER)] in labels


@pytest.mark.data
def test_real_accessions_resolve_207_of_253_through_the_goa_crosswalk() -> None:
    """The measured resolution split, and that the pinned floor sits just below it."""
    from torchcell.datasets.bacteria_common import (
        bacterial_genome,
        resolve_uniprot_accessions,
        uniprot_locus_crosswalk,
    )

    root = _deposit_or_skip()
    data_root = _data_root_or_skip()
    panel = y26.read_benchling_strain_table(root / y26.BENCHLING_TITER_REL)
    genome = bacterial_genome("pputida", y26.KT2440_STRAIN, data_root)
    resolution = resolve_uniprot_accessions(
        uniprot_locus_crosswalk(genome, data_root), panel.accessions, label="pin"
    )
    assert len(resolution.resolved) == y26.PANEL_RESOLVED_ACCESSIONS == 207
    assert len(resolution.multi_locus) == 2
    assert sorted(resolution.multi_locus) == ["Q877U6", "Q877V8"]
    assert len(resolution.unmapped) == 44
    assert resolution.collisions == {}
    assert y26.PANEL_MIN_RESOLVED_FRACTION < resolution.resolved_fraction
    assert resolution.resolved_fraction == pytest.approx(0.8182, abs=1e-4)


@pytest.mark.data
def test_real_results_text_join_agrees_on_pp_0168_and_differs_on_pp_4188() -> None:
    """One agreement inside the printed precision, one recorded 25.98874 mg/L gap."""
    panel = y26.read_benchling_strain_table(
        _deposit_or_skip() / y26.BENCHLING_TITER_REL
    )
    proofs = y26.assert_benchling_titers_match_the_results_text(panel)
    assert "PP_0168" in proofs[0] and "0.7534" in proofs[0]
    assert proofs[1].startswith("MEASURED DISAGREEMENT, kept:")
    assert "PP_4188" in proofs[1] and "25.98874" in proofs[1]
    assert "1.7691%" in proofs[1]


@pytest.mark.data
def test_real_deposited_stores_hold_125_records_each() -> None:
    """Both built dev stores, measured: 132 rows less the control less the 6 marked."""
    from torchcell.verification.runners import load_records

    data_root = _data_root_or_skip()
    for name, spec in y26.BIOPRODUCTION_DATASETS.items():
        root = osp.join(data_root, str(spec["root"]))
        if not osp.isdir(osp.join(root, "processed", "lmdb")):
            pytest.skip(f"{name} is not built under $DATA_ROOT")
        records = load_records(root)
        assert len(records) == y26.BENCHLING_RECORDS == 125, name


def test_read_benchling_strain_table_refuses_an_empty_file(tmp_path: Path) -> None:
    """An empty paste is not a deposit."""
    path = tmp_path / "panel.tsv"
    path.write_text("")
    with pytest.raises(y26.TableExtractionError, match="is empty"):
        y26.read_benchling_strain_table(path)


def test_read_benchling_strain_table_refuses_a_table_with_no_protein_column(
    tmp_path: Path,
) -> None:
    """The protein columns are what the panel family stores."""
    path = tmp_path / "panel.tsv"
    path.write_text("strain\tisoprenol_production\nControl\t1.0\n")
    with pytest.raises(y26.TableExtractionError, match="no protein column"):
        y26.read_benchling_strain_table(path)


def test_read_benchling_strain_table_refuses_a_non_finite_abundance(
    tmp_path: Path,
) -> None:
    """A rendered 'nan' is not a measurement this loader has a rule for."""
    path = tmp_path / "panel.tsv"
    path.write_text("strain\tisoprenol_production\tUP_A00\nControl\t1.0\tnan\n")
    with pytest.raises(y26.TableExtractionError, match="UP_A00 is 'nan'"):
        y26.read_benchling_strain_table(path)


def test_read_benchling_correlation_table_refuses_a_short_row(tmp_path: Path) -> None:
    """A row that is not three cells is a changed paste."""
    path = tmp_path / "corr.tsv"
    lines = _correlation_tsv().split("\n")
    lines[1] = "\t".join(lines[1].split("\t")[:-1])
    path.write_text("\n".join(lines))
    with pytest.raises(y26.TableExtractionError, match="has 2 cells"):
        y26.read_benchling_correlation_table(path)


def test_the_deposit_shape_proof_refuses_a_changed_column_count(
    synthetic_benchling: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pinned accession count is asserted, so a re-paste that lost one stops."""
    panel_path = tmp_path / "panel.tsv"
    panel_path.write_text(synthetic_benchling[0])
    corr_path = tmp_path / "corr.tsv"
    corr_path.write_text(synthetic_benchling[1])
    monkeypatch.setattr(y26, "BENCHLING_ACCESSIONS", len(PANEL_ACCESSIONS) - 1)
    with pytest.raises(y26.TableExtractionError, match="protein columns, pinned"):
        y26.assert_benchling_deposit_shape(
            y26.read_benchling_strain_table(panel_path),
            y26.read_benchling_correlation_table(corr_path),
        )


def test_the_deposit_shape_proof_refuses_a_changed_correlation_row_count(
    synthetic_benchling: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recorded table is pinned too, even though no record stores it."""
    panel_path = tmp_path / "panel.tsv"
    panel_path.write_text(synthetic_benchling[0])
    corr_path = tmp_path / "corr.tsv"
    corr_path.write_text(synthetic_benchling[1])
    monkeypatch.setattr(y26, "BENCHLING_CORRELATION_ROWS", len(CORRELATION_ROWS) + 1)
    with pytest.raises(y26.TableExtractionError, match="correlation table has"):
        y26.assert_benchling_deposit_shape(
            y26.read_benchling_strain_table(panel_path),
            y26.read_benchling_correlation_table(corr_path),
        )
