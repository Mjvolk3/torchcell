# tests/torchcell/sequence/genome/ecoli/test_w3110.py
# [[tests.torchcell.sequence.genome.ecoli.test_w3110]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sequence/genome/ecoli/test_w3110.py
"""E. coli K-12 W3110, read RefSeq first (#910).

The synthetic tests run everywhere: the real class reads a synthetic RefSeq-primary set
(a GCF flat file with ``/translation`` values, a GCF GFF3 with ``Ontology_term`` rows, a
GCA FASTA under its GenBank accession) through a stubbed ``resolve``. The tier tests
(``--data``) build the genome from the deposited set into a temporary cache root.
"""

import gzip
import os
import os.path as osp
from pathlib import Path
from typing import Any

import pytest
from Bio import SeqIO

import torchcell.verification.runners as runners
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    GO_OBO,
    SEQUENCE,
    SyntheticLocus,
    _genbank,
    forbid_network,
    serve_tier,
)
from torchcell.sequence.genome.bacterial import (
    GO_BASIC_OBO,
    GenomeAnnotationMismatchError,
    RefSeqGffAnnotation,
    refseq_primary_go,
)
from torchcell.sequence.genome.base import GeneNameStatus
from torchcell.sequence.genome.ecoli import (
    W3110_ASSEMBLY,
    EcoliK12W3110Gene,
    EcoliK12W3110Genome,
)
from torchcell.sequence.genome.registry import ECOLI_K12_W3110, GO_RELEASE_20260805

GCA = "GCA_000010245.1_ASM1024v1"
GCF = "GCF_000010245.2_ASM1024v1"

LOCI = [
    SyntheticLocus(
        tag="Y75_RS00005",
        parts=((1, 9),),
        strand="+",
        symbol="thrL",
        product="thr operon leader peptide",
        protein_id="WP_1.1",
        protein="MKR",
    ),
    SyntheticLocus(
        tag="Y75_RS00010",
        parts=((12, 26),),
        strand="-",
        symbol="thrA",
        product="aspartate kinase",
        protein_id="WP_2.1",
        protein="MRVLK",
    ),
    SyntheticLocus(
        tag="Y75_RS00015", parts=((30, 41),), strand="+", symbol="caiB", pseudo=True
    ),
]
#: (locus tag, old locus tag, GO ids) of the RefSeq GFF rows.
GFF_ROWS = (
    ("Y75_RS00005", "Y75_p0001", ("GO:0000001",)),
    ("Y75_RS00010", "Y75_p0002", ("GO:0000003",)),
    ("Y75_RS00015", None, ()),
)


def _gff(seqid: str = "NC_007779.1") -> str:
    rows = ["##gff-version 3"]
    for locus, (tag, old, go) in zip(LOCI, GFF_ROWS, strict=True):
        kind = "pseudogene" if locus.pseudo else "gene"
        attrs = f"ID=gene-{tag};locus_tag={tag};gene={locus.symbol}"
        if old:
            attrs += f";old_locus_tag={old}"
        start, end = locus.parts[0]
        rows.append(
            "\t".join([seqid, "RefSeq", kind, str(start), str(end), ".", locus.strand])
            + f"\t.\t{attrs}"
        )
        if go:
            rows.append(
                "\t".join(
                    [seqid, "Protein Homology", "CDS", str(start), str(end), "."]
                    + [locus.strand, "0"]
                )
                + f"\tID=cds-{tag};Parent=gene-{tag};locus_tag={tag};"
                f"Ontology_term={','.join(go)}"
            )
    return "\n".join(rows) + "\n"


def write_set(
    root: Path,
    *,
    fasta_id: str = "AP009048.1",
    sequence: str = SEQUENCE,
    translations: bool = True,
) -> dict[tuple[str, str], Path]:
    """The members the W3110 class asks for, written under ``root``."""
    files: dict[tuple[str, str], Path] = {}

    def put(member: str, text: str, set_id: str = ECOLI_K12_W3110) -> None:
        path = root / set_id / member
        path.parent.mkdir(parents=True, exist_ok=True)
        if member.endswith(".gz"):
            with gzip.open(path, "wt") as handle:
                handle.write(text)
        else:
            path.write_text(text)
        files[(set_id, member)] = path

    record = _genbank(W3110_ASSEMBLY, LOCI)
    for feature in record.features:
        protein = next(
            (
                locus.protein
                for locus in LOCI
                if feature.type == "CDS"
                and feature.qualifiers.get("locus_tag") == [locus.tag]
            ),
            None,
        )
        if protein and translations:
            feature.qualifiers["translation"] = [protein]
    flat = root / ECOLI_K12_W3110 / W3110_ASSEMBLY.annotation_member
    flat.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(flat, "wt") as handle:
        SeqIO.write(record, handle, "genbank")
    files[(ECOLI_K12_W3110, W3110_ASSEMBLY.annotation_member)] = flat
    put(W3110_ASSEMBLY.gff_member, _gff())
    put(W3110_ASSEMBLY.dna_fasta_member, f">{fasta_id} synthetic\n{sequence}\n")
    put(GO_BASIC_OBO, GO_OBO, GO_RELEASE_20260805)
    return files


@pytest.fixture
def tier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    files = write_set(tmp_path / "tier")
    forbid_network(monkeypatch)
    return serve_tier(monkeypatch, files)


@pytest.fixture
def w3110(tmp_path: Path, tier: list[tuple[str, str]]) -> EcoliK12W3110Genome:
    return EcoliK12W3110Genome(genome_root=str(tmp_path / "w3110"), overwrite=True)


def test_the_assembly_reads_refseq_first() -> None:
    assert W3110_ASSEMBLY.annotation_source == "refseq"
    assert W3110_ASSEMBLY.annotation_assembly == GCF
    assert W3110_ASSEMBLY.annotation_member == f"{GCF}_genomic.gbff.gz"
    assert W3110_ASSEMBLY.gff_member == f"{GCF}_genomic.gff.gz"
    assert W3110_ASSEMBLY.refseq_gff_member == W3110_ASSEMBLY.gff_member
    assert W3110_ASSEMBLY.protein_fasta_member == W3110_ASSEMBLY.annotation_member
    assert W3110_ASSEMBLY.dna_fasta_member == f"{GCA}_genomic.fna.gz"
    assert W3110_ASSEMBLY.genbank_member == f"{GCA}_genomic.gbff.gz"
    assert W3110_ASSEMBLY.assembly_report_member == f"{GCF}_assembly_report.txt"
    assert (W3110_ASSEMBLY.replicon, W3110_ASSEMBLY.sequence_replicon) == (
        "NC_007779.1",
        "AP009048.1",
    )
    assert W3110_ASSEMBLY.go_source.identifier == (
        "RefSeq GFF Ontology_term row's own locus_tag"
    )
    assert EcoliK12W3110Genome.gene_class() is EcoliK12W3110Gene
    assert EcoliK12W3110Gene.seqid_to_chromosome("NC_007779.1") == 1


def test_construction_reads_the_gcf_files_and_the_gca_fasta(
    tmp_path: Path, tier: list[tuple[str, str]]
) -> None:
    genome = EcoliK12W3110Genome(genome_root=str(tmp_path / "w3110"), overwrite=True)
    assert tier == [
        (ECOLI_K12_W3110, f"{GCA}_genomic.fna.gz"),
        (ECOLI_K12_W3110, f"{GCF}_genomic.gff.gz"),
        (ECOLI_K12_W3110, f"{GCF}_genomic.gbff.gz"),
        (ECOLI_K12_W3110, f"{GCF}_genomic.gbff.gz"),
        (GO_RELEASE_20260805, "go-basic.obo"),
        (ECOLI_K12_W3110, f"{GCF}_genomic.gff.gz"),
        (ECOLI_K12_W3110, f"{GCF}_genomic.gff.gz"),
    ]
    assert genome.strain == "K-12 W3110"
    assert list(genome.gene_set) == ["Y75_RS00005", "Y75_RS00010"]
    assert genome.genbank.pseudogene_tags == ["Y75_RS00015"]
    assert list(genome.fasta_dna) == ["NC_007779.1"]
    assert genome.fasta_dna["NC_007779.1"].description.startswith("AP009048.1 of ")
    assert {t: str(r.seq) for t, r in genome.fasta_protein.items()} == {
        "Y75_RS00005": "MKR",
        "Y75_RS00010": "MRVLK",
    }
    assert genome.chr_to_nc == {1: "NC_007779.1"}


def test_go_is_the_refseq_rows_own_locus_tag(w3110: EcoliK12W3110Genome) -> None:
    assert {t: list(g) for t, g in w3110.go_annotations.items()} == {
        "Y75_RS00005": ["GO:0000001"],
        "Y75_RS00010": ["GO:0000003"],
    }
    source = w3110.go_source
    assert (source.route, source.rows, source.rows_without_identifier) == (
        "refseq_gff_ontology_term_primary",
        2,
        0,
    )
    gene = w3110["Y75_RS00010"]
    assert gene is not None
    assert (gene.chromosome, gene.start, gene.end, gene.strand) == (1, 12, 26, "-")
    assert gene.symbol == "thrA"


def test_names_resolve_through_symbol_and_old_locus_tag(
    w3110: EcoliK12W3110Genome,
) -> None:
    by_symbol = w3110.resolve_gene_name("thrA")
    assert (by_symbol.status, by_symbol.systematic_name) == (
        GeneNameStatus.RENAMED,
        "Y75_RS00010",
    )
    assert w3110.refseq.old_locus_tags["Y75_RS00005"] == ("Y75_p0001",)
    pseudo = w3110.resolve_gene_name("Y75_RS00015")
    assert pseudo.status is GeneNameStatus.NON_GENE_FEATURE


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fasta_id": "XX000001.1"}, "not the one record AP009048.1"),
        ({"sequence": "A" * 100}, "are not byte-identical"),
        (
            {"translations": False},
            "the CDS WP_1.1 of Y75_RS00005 carries no /translation",
        ),
    ],
    ids=["wrong-record", "other-sequence", "no-translation"],
)
def test_a_set_whose_members_disagree_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    files = write_set(tmp_path / "tier", **kwargs)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, files)
    with pytest.raises(GenomeAnnotationMismatchError, match=message):
        EcoliK12W3110Genome(genome_root=str(tmp_path / "w3110"), overwrite=True)


def test_refseq_primary_go_refuses_a_row_with_no_gene() -> None:
    annotation = RefSeqGffAnnotation(
        member="m.gff.gz",
        old_locus_tags={"A_RS1": ()},
        ontology_term_rows=(("A_RS1", ("GO:0000001",)), ("A_RS9", ("GO:0000003",))),
    )
    with pytest.raises(ValueError, match="names A_RS9, which has no gene"):
        refseq_primary_go(annotation)
    ok = annotation.model_copy(
        update={"ontology_term_rows": annotation.ontology_term_rows[:1]}
    )
    assert refseq_primary_go(ok) == ({"A_RS1": ("GO:0000001",)}, 0)


def test_the_runner_universe_is_the_refseq_loci_and_refuses_a_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = write_set(tmp_path / "tier")

    def serve(assembly_set: str, filename: str, **_: Any) -> str:
        return str(files[(assembly_set, filename)])

    monkeypatch.setattr(runners, "resolve", serve)
    universe = runners._bacterial_gene_set(W3110_ASSEMBLY, str(tmp_path))
    assert universe == {"Y75_RS00005", "Y75_RS00010", "Y75_RS00015"}
    gff = files[(ECOLI_K12_W3110, W3110_ASSEMBLY.gff_member)]
    with gzip.open(gff, "wt") as handle:
        handle.write(_gff().replace("Y75_RS00015", "Y75_RS00099"))
    with pytest.raises(ValueError, match="disagree: only in the GFF"):
        runners._bacterial_gene_set(W3110_ASSEMBLY, str(tmp_path))


# --------------------------------------------------------------------------------------
# The deposited tier (--data)
# --------------------------------------------------------------------------------------
DATA_ROOT = os.environ.get("DATA_ROOT", "")
requires_tier = pytest.mark.skipif(
    not osp.isfile(
        osp.join(DATA_ROOT, "torchcell-genomes", ECOLI_K12_W3110, "manifest.json")
    ),
    reason=f"requires {ECOLI_K12_W3110} under $DATA_ROOT/torchcell-genomes",
)


@pytest.mark.data
@requires_tier
def test_the_deposited_set_reads_refseq_first(tmp_path: Path) -> None:
    """4,531 loci (4,346 genes, 185 pseudogenes), the FASTA re-keyed, GO on 2,201."""
    genome = EcoliK12W3110Genome(genome_root=str(tmp_path / "w3110"), overwrite=True)
    assert len(genome.genbank.loci) == 4_531
    assert len(genome.gene_set) == 4_346
    assert len(genome.genbank.pseudogene_tags) == 185
    assert genome.chr_to_len == {1: 4_646_332}
    assert genome.go_source.genes_annotated == 2_201
    assert genome.go_source.rows == 2_298
    assert genome.resolve_gene_name("thrL").systematic_name == "Y75_RS00005"
    assert genome.resolve_gene_name("Y75_p0001").systematic_name == "Y75_RS00005"
    assert runners._bacterial_gene_set(W3110_ASSEMBLY, DATA_ROOT) == set(
        genome.genbank.loci
    )


@pytest.mark.data
@requires_tier
def test_the_w3110_reference_pins_the_refseq_accession() -> None:
    from torchcell.datasets.bacteria_common import assembly_reference

    reference = assembly_reference("W3110", data_root=DATA_ROOT)
    assert (reference.assembly_set, reference.assembly_accession) == (
        ECOLI_K12_W3110,
        "GCF_000010245.2",
    )
