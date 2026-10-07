# tests/torchcell/sequence/genome/test_bacterial.py
# [[tests.torchcell.sequence.genome.test_bacterial]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sequence/genome/test_bacterial.py
"""The NCBI parsers and records of the bacterial layer, on synthetic files.

The genome classes themselves are exercised in ``ecoli/test_k12.py`` and
``pputida/test_kt2440.py``; this module pins what each reader extracts and what it
refuses. Fixtures are written by ``_bacterial_fixtures.write_assembly`` (Biopython writes
the GenBank file), so nothing reads the tier.
"""

import gzip
from pathlib import Path

import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from pydantic import ValidationError

from tests.torchcell.sequence.genome._bacterial_fixtures import (
    BW25113_LOCI,
    MG1655_GAF,
    MG1655_LOCI,
    SEQUENCE,
    gaf_row,
    write_assembly,
)
from torchcell.sequence.genome.bacterial import (
    GenBankLocus,
    GenomeAnnotationMismatchError,
    GoRoute,
    GoSourceSpec,
    read_fasta,
    read_gaf_synonym_go,
    read_genbank,
    read_protein_fasta,
    read_refseq_gff,
    refseq_inline_go,
)
from torchcell.sequence.genome.ecoli.k12 import BW25113_ASSEMBLY, MG1655_ASSEMBLY


def _revcomp(start: int, end: int) -> str:
    return str(Seq(SEQUENCE[start - 1 : end]).reverse_complement())


@pytest.fixture
def mg1655_files(tmp_path: Path) -> dict[tuple[str, str], Path]:
    """The synthetic MG1655 set."""
    return write_assembly(tmp_path, MG1655_ASSEMBLY, MG1655_LOCI, MG1655_GAF)


def _member(files: dict[tuple[str, str], Path], member: str) -> str:
    return next(str(p) for (_, m), p in files.items() if m == member)


def test_read_genbank_records_every_locus_field(
    mg1655_files: dict[tuple[str, str], Path],
) -> None:
    """Symbols, ``;``-split synonyms, xrefs, strand, product, protein and isoform ids,
    pseudo flags and joined segments come from the flat file as written.
    """
    member = MG1655_ASSEMBLY.genbank_member
    annotation, _ = read_genbank(_member(mg1655_files, member), member)
    assert annotation.member == member
    assert [(r.accession, r.length, r.topology) for r in annotation.replicons] == [
        ("U00096.3", 100, "circular")
    ]
    assert list(annotation.loci) == [f"b000{i}" for i in range(1, 8)]
    assert annotation.gene_tags == ["b0001", "b0002", "b0003", "b0005", "b0006"]
    assert annotation.pseudogene_tags == ["b0004", "b0007"]
    assert annotation.loci["b0002"] == GenBankLocus(
        locus_tag="b0002",
        symbol="thrA",
        synonyms=("ECK0002", "Hs", "thrA1"),
        old_locus_tags=(),
        db_xrefs=("NCBI_GP:AAC73113.1",),
        pseudo=False,
        replicon="U00096.3",
        start=12,
        end=26,
        strand="-",
        segments=1,
        product_feature_type="CDS",
        product="aspartokinase I",
        protein_id="AAC73113.1",
        isoform_protein_ids=("QNV50512.1",),
    )
    assert annotation.loci["b0001"].db_xrefs == ("ECOCYC:EG11277", "NCBI_GP:AAC73112.1")
    trna = annotation.loci["b0003"]
    assert (trna.product_feature_type, trna.product, trna.protein_id) == (
        "tRNA",
        "tRNA-Thr",
        None,
    )
    pseudo = annotation.loci["b0004"]
    assert (pseudo.pseudo, pseudo.product_feature_type, pseudo.protein_id) == (
        True,
        "CDS",
        None,
    )
    joined = annotation.loci["b0007"]
    assert (joined.pseudo, joined.segments, joined.start, joined.end) == (
        True,
        2,
        84,
        97,
    )
    assert joined.product_feature_type is None


def test_read_genbank_extracts_the_coding_cds_only(
    mg1655_files: dict[tuple[str, str], Path],
) -> None:
    """The CDS of every coding locus, minus strand reverse-complemented, keyed by locus
    tag with the protein id as description; RNA and pseudo loci have none.
    """
    member = MG1655_ASSEMBLY.genbank_member
    _, cds = read_genbank(_member(mg1655_files, member), member)
    assert sorted(cds) == ["b0001", "b0002", "b0005", "b0006"]
    assert str(cds["b0001"].seq) == SEQUENCE[0:9]
    assert str(cds["b0002"].seq) == _revcomp(12, 26)
    assert (cds["b0002"].id, cds["b0002"].description) == ("b0002", "AAC73113.1")


def _write_genbank(path: Path, features: list[SeqFeature]) -> str:
    record = SeqRecord(
        Seq(SEQUENCE),
        id="U00096.3",
        name="U00096",
        annotations={"molecule_type": "DNA", "topology": "circular"},
    )
    record.features.extend(features)
    with gzip.open(path, "wt") as handle:
        SeqIO.write(record, handle, "genbank")
    return str(path)


def _feature(ftype: str, start: int, end: int, tag: str) -> SeqFeature:
    return SeqFeature(
        SimpleLocation(start - 1, end, 1), type=ftype, qualifiers={"locus_tag": [tag]}
    )


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (
            [_feature("gene", 1, 9, "b0001"), _feature("CDS", 12, 20, "b0002")],
            "product features name locus tags with no gene feature: ['b0002']",
        ),
        (
            [_feature("gene", 1, 9, "b0001"), _feature("gene", 12, 20, "b0001")],
            "locus tag b0001 is repeated",
        ),
        (
            [
                _feature("gene", 1, 9, "b0001"),
                _feature("CDS", 1, 9, "b0001"),
                _feature("ncRNA", 1, 9, "b0001"),
            ],
            "2 product features span the gene (['CDS', 'ncRNA']); the product is not "
            "unique",
        ),
    ],
)
def test_read_genbank_refuses_an_inconsistent_file(
    tmp_path: Path, features: list[SeqFeature], message: str
) -> None:
    """An orphan product, a repeated tag and a non-unique product are named."""
    path = _write_genbank(tmp_path / "bad.gbff.gz", features)
    with pytest.raises(ValueError) as err:
        read_genbank(path, "bad.gbff.gz")
    assert message in str(err.value)


def test_read_protein_fasta_rekeys_to_locus_tags(
    mg1655_files: dict[tuple[str, str], Path],
) -> None:
    """Coding loci get their protein under the locus tag; isoforms are left out."""
    member = MG1655_ASSEMBLY.genbank_member
    annotation, _ = read_genbank(_member(mg1655_files, member), member)
    proteins = read_protein_fasta(
        _member(mg1655_files, MG1655_ASSEMBLY.protein_fasta_member), annotation
    )
    assert {tag: str(r.seq) for tag, r in proteins.items()} == {
        "b0001": "MK",
        "b0002": "MRVLK",
        "b0005": "MSDS",
        "b0006": "MEKK",
    }
    assert proteins["b0002"].description == "AAC73113.1 aspartokinase I"


def test_read_protein_fasta_refuses_a_missing_protein(
    tmp_path: Path, mg1655_files: dict[tuple[str, str], Path]
) -> None:
    """A protein id the flat file names but the FASTA lacks is named."""
    member = MG1655_ASSEMBLY.genbank_member
    annotation, _ = read_genbank(_member(mg1655_files, member), member)
    short = tmp_path / "short.faa"
    short.write_text(">AAC73112.1 x\nMK\n")
    with pytest.raises(GenomeAnnotationMismatchError) as err:
        read_protein_fasta(str(short), annotation)
    assert "short.faa lacks 3 protein ids" in str(err.value)


def test_read_fasta_reads_plain_and_gzip_alike(tmp_path: Path) -> None:
    """The ``.gz`` suffix selects gzip; the records are the same."""
    plain = tmp_path / "x.fna"
    plain.write_text(">r1 one\nACGT\n>r2 two\nGG\n")
    packed = tmp_path / "x.fna.gz"
    with gzip.open(packed, "wt") as handle:
        handle.write(plain.read_text())
    for path in (plain, packed):
        records = read_fasta(str(path))
        assert {k: str(v.seq) for k, v in records.items()} == {"r1": "ACGT", "r2": "GG"}


def test_read_gaf_synonym_go_reads_column_11_tokens(
    mg1655_files: dict[tuple[str, str], Path],
) -> None:
    """NOT rows are excluded, ``/`` joins split, ``b0005.1`` is not b0005, and a row
    with no b-number is counted.
    """
    gaf = read_gaf_synonym_go(
        _member(mg1655_files, "ECOLI-uniprot.gaf.gz"), "ECOLI-uniprot.gaf.gz", r"b\d{4}"
    )
    assert (gaf.rows, gaf.not_rows, gaf.rows_without_identifier) == (7, 1, 1)
    assert gaf.annotations == {
        "b0001": ("GO:0000001",),
        "b0002": ("GO:0000003",),
        "b0004": ("GO:0000002",),
        "b0005": ("GO:0000001", "GO:0000002"),
        "b0099": ("GO:0000002",),
    }


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("\t".join(["UniProtKB", "P1", "x"]), "a GAF 2.x row has 17 columns, got 3"),
        (gaf_row("thrL", "b0001", "GO:1"), "malformed GO id 'GO:1'"),
    ],
)
def test_read_gaf_synonym_go_refuses_malformed_rows(
    tmp_path: Path, row: str, message: str
) -> None:
    """A short row and a malformed GO id are refused by name."""
    path = tmp_path / "bad.gaf"
    path.write_text("!gaf-version: 2.2\n" + row + "\n")
    with pytest.raises(ValueError) as err:
        read_gaf_synonym_go(str(path), "bad.gaf", r"b\d{4}")
    assert message in str(err.value)


def test_read_refseq_gff_and_inline_go_crosswalk(tmp_path: Path) -> None:
    """RefSeq tags map to their ``old_locus_tag``; Ontology_term rows reach GenBank tags
    through it, and the RefSeq-only gene's row is counted as unreachable.
    """
    files = write_assembly(tmp_path, BW25113_ASSEMBLY, BW25113_LOCI)
    member = BW25113_ASSEMBLY.refseq_gff_member
    refseq = read_refseq_gff(_member(files, member), member)
    assert refseq.old_locus_tags == {
        "BW25113_RS00005": ("BW25113_0001",),
        "BW25113_RS00010": ("BW25113_0002",),
        "BW25113_RS00015": ("BW25113_4412",),
        "BW25113_RS00020": ("BW25113_0004",),
        "BW25113_RS00025": ("BW25113_0005",),
        "BW25113_0008": (),
        "X_RS99999": (),
    }
    assert refseq.ontology_term_rows == (
        ("BW25113_RS00005", ("GO:0000001",)),
        ("BW25113_RS00010", ("GO:0000003", "GO:0000002")),
        ("BW25113_RS00020", ("GO:0000001",)),
        ("X_RS99999", ("GO:0000004",)),
    )
    annotations, without = refseq_inline_go(refseq)
    assert without == 1
    assert annotations == {
        "BW25113_0001": ("GO:0000001",),
        "BW25113_0002": ("GO:0000002", "GO:0000003"),
        "BW25113_0004": ("GO:0000001",),
    }


def test_read_refseq_gff_refuses_disagreeing_rows(tmp_path: Path) -> None:
    """Two rows of one RefSeq gene with different ``old_locus_tag`` are refused."""
    path = tmp_path / "bad.gff"
    row = "NZ_X\tRefSeq\tpseudogene\t1\t9\t.\t+\t.\tID=gene-R1;locus_tag=R1;old_locus_tag={}"
    path.write_text(row.format("A_0001") + "\n" + row.format("A_0002") + "\n")
    with pytest.raises(ValueError) as err:
        read_refseq_gff(str(path), "bad.gff")
    assert "rows of R1 disagree on old_locus_tag" in str(err.value)


def test_refseq_inline_go_refuses_a_row_without_its_gene(tmp_path: Path) -> None:
    """An Ontology_term row whose RefSeq gene has no gene row is refused."""
    path = tmp_path / "orphan.gff"
    path.write_text(
        "NZ_X\tRefSeq\tCDS\t1\t9\t.\t+\t0\tID=cds-1;locus_tag=R9;Ontology_term=GO:0000001\n"
    )
    refseq = read_refseq_gff(str(path), "orphan.gff")
    with pytest.raises(ValueError) as err:
        refseq_inline_go(refseq)
    assert "names R9, which has no gene or pseudogene row" in str(err.value)


def test_go_source_spec_ties_the_pattern_to_the_route() -> None:
    """A GAF route needs an identifier pattern; the RefSeq route takes none."""
    cases: tuple[tuple[GoRoute, str | None], ...] = (
        ("gaf_synonym_column", None),
        ("refseq_gff_ontology_term", r"b\d{4}"),
    )
    for route, pattern in cases:
        with pytest.raises(ValidationError) as err:
            GoSourceSpec(
                route=route, assembly_set="s", member="m", identifier_pattern=pattern
            )
        assert "identifier_pattern is required for the gaf_synonym_column" in str(
            err.value
        )
    spec = MG1655_ASSEMBLY.go_source
    assert spec.identifier == (
        "GAF column 11 (DB Object Synonym), values split on '|' and '/', tokens "
        r"fully matching b\d{4}"
    )


def test_bacterial_assembly_names_its_members() -> None:
    """Member names derive from the GenBank and RefSeq assembly names."""
    assert (
        MG1655_ASSEMBLY.genbank_member,
        MG1655_ASSEMBLY.gff_member,
        MG1655_ASSEMBLY.dna_fasta_member,
        MG1655_ASSEMBLY.protein_fasta_member,
        MG1655_ASSEMBLY.refseq_gff_member,
    ) == (
        "GCA_000005845.2_ASM584v2_genomic.gbff.gz",
        "GCA_000005845.2_ASM584v2_genomic.gff.gz",
        "GCA_000005845.2_ASM584v2_genomic.fna.gz",
        "GCA_000005845.2_ASM584v2_protein.faa.gz",
        "GCF_000005845.2_ASM584v2_genomic.gff.gz",
    )
