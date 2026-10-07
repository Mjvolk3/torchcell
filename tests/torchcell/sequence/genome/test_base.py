# tests/torchcell/sequence/genome/test_base.py
# [[tests.torchcell.sequence.genome.test_base]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sequence/genome/test_base.py
"""AnnotatedGenome through a minimal non-yeast subclass: the hooks, not SGD, decide.

``ToyGenome`` is a host with one linear replicon (``contig1``, 60 nt, chromosome key
1), its own locus feature types (``gene`` and ``pseudogene``), and GO carried in a GFF
attribute named ``go_terms``. Every SGD convention is deliberately absent: no roman
numerals, no ``chrmt``, no ``Ontology_term`` GO, no ``orf_classification``, no
``go_root`` field, no download. ``resolve`` is stubbed at its import site
(``torchcell.sequence.genome.base.resolve``) to serve files the fixture writes, and the
OBO hook returns a fixture file.

Fixture (GFF coordinates are 1-based inclusive; Python slices are ``[start - 1:end]``):

* ``T0001`` gene, ``+``, 1..9, standard name ``abcA``, aliases ``ABC1, SHARED``,
  ``go_terms=GO:0000003,GO:0000001,EC:1.1.1.1``, with a CDS child.
* ``T0002`` gene, ``-``, 11..25, alias ``SHARED``, ``go_terms=GO:0000002``.
* ``T0003`` gene, ``+``, 30..41, no ``go_terms`` but ``Ontology_term=GO:0000099`` (the
  SGD attribute, which this host does not read GO from).
* ``T0004`` pseudogene, ``+``, 45..50, standard name ``psiB``.
* ``abcA`` region, 52..55 (a non-locus feature whose id is ``T0001``'s standard name).

``go.obo`` holds ``GO:0000001`` (live) and ``GO:0000002`` (obsolete); ``GO:0000003`` and
``GO:0000099`` are absent.
"""

import ast
import pickle
import re
from pathlib import Path
from typing import Any, ClassVar

import pytest
from attrs import define
from sortedcontainers import SortedDict, SortedSet

import torchcell.sequence.genome.base as base
from torchcell.literature.manifest import sha256_file
from torchcell.sequence import DnaSelectionResult
from torchcell.sequence.genome.base import (
    AnnotatedGene,
    AnnotatedGenome,
    GeneNameResolution,
    GeneNameStatus,
    GenomeDatabaseSource,
    GenomeReleaseFiles,
    GenomeRootNotFoundError,
    database_content_digest,
    read_genome_database_record,
)

CONTIG = "ATGAAACCCTTACCCGGGTTTCATGGATCCATGCATGCAAGCTTGGTAACCTGCAGGTCA"
TOY_SET = "toy_assembly_v1"
FILES = GenomeReleaseFiles(
    dna_fasta="toy_genomic.fna",
    gff="toy_genomic.gff",
    protein_fasta="toy_protein.faa",
    cds_fasta="toy_cds.fna",
)
GFF_ROWS = [
    (
        "gene",
        1,
        9,
        "+",
        "ID=T0001;Name=T0001;gene=abcA;Alias=ABC1,SHARED;"
        "go_terms=GO:0000003,GO:0000001,EC:1.1.1.1",
    ),
    ("CDS", 1, 9, "+", "ID=T0001_cds;Parent=T0001"),
    ("gene", 11, 25, "-", "ID=T0002;Name=T0002;Alias=SHARED;go_terms=GO:0000002"),
    ("gene", 30, 41, "+", "ID=T0003;Name=T0003;Ontology_term=GO:0000099"),
    ("pseudogene", 45, 50, "+", "ID=T0004;Name=T0004;gene=psiB"),
    ("region", 52, 55, "+", "ID=abcA"),
]
GO_OBO = """format-version: 1.2

[Term]
id: GO:0000001
name: mitochondrion inheritance
namespace: biological_process

[Term]
id: GO:0000002
name: retired process
namespace: biological_process
is_obsolete: true
"""


@define(repr=False)
class ToyGene(AnnotatedGene):
    """A gene of the toy host: one replicon, GO in ``go_terms``."""

    GO_ATTRIBUTE: ClassVar[str] = "go_terms"

    @classmethod
    def seqid_to_chromosome(cls, seqid: str) -> int:
        """The one replicon is chromosome 1."""
        if seqid != "contig1":
            raise ValueError(f"toy host has one replicon, not {seqid!r}")
        return 1


@define(eq=False)
class ToyGenome(AnnotatedGenome[ToyGene]):
    """The toy host: every hook set, no SGD convention, no ``go_root``; its init
    fields are the base's own (``genome_root``, ``overwrite``).
    """

    ASSEMBLY_SET: ClassVar[str] = TOY_SET
    GENOME_VERSION: ClassVar[str] = "v1"
    LOCUS_FEATURE_TYPES: ClassVar[frozenset[str]] = frozenset({"gene", "pseudogene"})
    ANNOTATION_NAME: ClassVar[str] = "ToyAnno"
    ANNOTATION_RELEASE: ClassVar[str] = "ToyAnno-v1"
    #: Set by the fixture: the OBO file the GO hook returns.
    OBO_PATH: ClassVar[str] = ""

    @classmethod
    def gene_class(cls) -> type[ToyGene]:
        """Toy genes."""
        return ToyGene

    @classmethod
    def release_files(cls) -> GenomeReleaseFiles:
        """The toy members."""
        return FILES

    @classmethod
    def fasta_chromosome(cls, record: Any) -> int:
        """The one replicon is chromosome 1, keyed by the record id."""
        if record.id != "contig1":
            raise ValueError(f"toy host has one replicon, not {record.id!r}")
        return 1

    def _prepare_go_obo(self) -> str:
        """The fixture OBO; never a download."""
        return self.OBO_PATH


@pytest.fixture
def resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Write the toy release and serve it through a recording ``resolve`` stub."""
    release = tmp_path / "release"
    release.mkdir()
    gff = ["##gff-version 3"] + [
        "\t".join(["contig1", "Toy", ftype, str(s), str(e), ".", strand, ".", attrs])
        for ftype, s, e, strand, attrs in GFF_ROWS
    ]
    texts = {
        FILES.dna_fasta: f">contig1 toy linear replicon\n{CONTIG}\n",
        FILES.gff: "\n".join(gff) + "\n",
        FILES.protein_fasta: ">T0001\nMKP*\n",
        FILES.cds_fasta: ">T0001\nATGAAACCC\n",
    }
    for name, text in texts.items():
        (release / name).write_text(text)
    calls: list[tuple[str, str]] = []

    def serve(assembly_set: str, filename: str) -> str:
        calls.append((assembly_set, filename))
        return str(release / filename)

    monkeypatch.setattr(base, "resolve", serve)
    obo = tmp_path / "go.obo"
    obo.write_text(GO_OBO)
    monkeypatch.setattr(ToyGenome, "OBO_PATH", str(obo))
    return calls


@pytest.fixture
def genome(tmp_path: Path, resolved: list[tuple[str, str]]) -> ToyGenome:
    """The toy genome, its ``data.db`` built by the constructor (root absent)."""
    return ToyGenome(genome_root=str(tmp_path / "genome"), overwrite=True)


def test_construction_resolves_every_member_and_records_the_subclass_source(
    tmp_path: Path, resolved: list[tuple[str, str]]
) -> None:
    """The four members come from the tier under ``ASSEMBLY_SET``, in order, and the
    ``data.db`` record names the toy set and GFF; a second construction trusts it.
    """
    root = tmp_path / "genome"
    ToyGenome(genome_root=str(root), overwrite=True)
    assert resolved == [
        (TOY_SET, "toy_genomic.fna"),
        (TOY_SET, "toy_genomic.gff"),
        (TOY_SET, "toy_protein.faa"),
        (TOY_SET, "toy_cds.fna"),
    ]
    record = read_genome_database_record(str(root / "data.db"))
    assert record is not None
    assert record.source == GenomeDatabaseSource(
        assembly_set=TOY_SET,
        gff_filename="toy_genomic.gff",
        gff_sha256=sha256_file(tmp_path / "release" / "toy_genomic.gff"),
        keep_order=True,
        merge_strategy="merge",
        sort_attribute_values=True,
    )
    assert record.featuretype_counts == {
        "CDS": 1,
        "gene": 3,
        "pseudogene": 1,
        "region": 1,
    }
    assert ToyGenome.database_untrusted_reason(str(root)) is None
    assert ToyGenome.database_untrusted_reason(str(tmp_path / "empty")) == (
        "it does not exist"
    )


def test_one_linear_replicon_is_chromosome_one(genome: ToyGenome) -> None:
    """``fasta_chromosome`` keys the single record as 1; its length is the contig's."""
    assert genome.chr_to_nc == {1: "contig1"}
    assert genome.nc_to_chr == {"contig1": 1}
    assert genome.chr_to_len == {1: 60}


def test_gene_set_is_the_gene_features_only(genome: ToyGenome) -> None:
    """The pseudogene, the CDS and the region are features but not genes."""
    assert list(genome.gene_set) == ["T0001", "T0002", "T0003"]
    assert len(genome) == 3
    assert sorted(genome.feature_types) == ["CDS", "gene", "pseudogene", "region"]


def test_genes_resolve_on_both_strands_of_the_replicon(genome: ToyGenome) -> None:
    """``+`` T0001 is ``CONTIG[0:9]``; ``-`` T0002 is revcomp(``CONTIG[10:25]``)."""
    plus = genome["T0001"]
    assert type(plus) is ToyGene
    assert (plus.chromosome, plus.start, plus.end, plus.strand) == (1, 1, 9, "+")
    assert plus.seq == CONTIG[0:9] == "ATGAAACCC"
    assert plus.protein is not None and str(plus.protein.seq) == "MKP*"
    assert str(plus.cds.seq) == "ATGAAACCC"
    assert (plus.alias, plus.name) == (["ABC1", "SHARED"], ["T0001"])
    minus = genome["T0002"]
    assert minus is not None
    assert (minus.chromosome, minus.start, minus.end, minus.strand) == (1, 11, 25, "-")
    assert CONTIG[10:25] == "TACCCGGGTTTCATG"
    assert minus.seq == "CATGAAACCCGGGTA"
    assert (minus.protein, minus.cds) == (None, None)


def test_go_comes_from_the_go_attribute_hook(genome: ToyGenome) -> None:
    """``go_terms`` supplies GO (non-GO values dropped); T0003's ``Ontology_term`` is
    not read, so its ``go`` is None.
    """
    t1, t2, t3 = genome["T0001"], genome["T0002"], genome["T0003"]
    assert t1 is not None and t2 is not None and t3 is not None
    assert t1.go == SortedSet(["GO:0000001", "GO:0000003"])
    assert t2.go == SortedSet(["GO:0000002"])
    assert t3.go is None
    assert genome.go == SortedSet(["GO:0000001", "GO:0000002", "GO:0000003"])
    assert genome.go_genes == SortedDict(
        {
            "GO:0000001": SortedSet(["T0001"]),
            "GO:0000002": SortedSet(["T0002"]),
            "GO:0000003": SortedSet(["T0001"]),
        }
    )


def test_resolver_uses_the_locus_types_and_annotation_labels(genome: ToyGenome) -> None:
    """Each layer, with the notes naming ``ToyAnno``; the ``abcA`` region does not
    shadow the gene whose standard name is ``abcA``.
    """
    results = {
        name: genome.resolve_gene_name(name)
        for name in ["t0001", "T0004", "abcA", "ABC1", "SHARED", "psiB", "T9999"]
    }
    assert results == {
        "t0001": GeneNameResolution(
            input_name="t0001", status=GeneNameStatus.CURRENT, systematic_name="T0001"
        ),
        "T0004": GeneNameResolution(
            input_name="T0004",
            status=GeneNameStatus.NON_GENE_FEATURE,
            systematic_name="T0004",
            feature_type="pseudogene",
            note="valid ToyAnno pseudogene, not a gene feature",
        ),
        "abcA": GeneNameResolution(
            input_name="abcA",
            status=GeneNameStatus.RENAMED,
            systematic_name="T0001",
            note="standard name of current gene T0001",
        ),
        "ABC1": GeneNameResolution(
            input_name="ABC1",
            status=GeneNameStatus.RENAMED,
            systematic_name="T0001",
            note="alias of current gene T0001",
        ),
        "SHARED": GeneNameResolution(
            input_name="SHARED",
            status=GeneNameStatus.AMBIGUOUS,
            systematic_name=None,
            candidates=["T0001", "T0002"],
            note="alias of multiple current genes",
        ),
        "psiB": GeneNameResolution(
            input_name="psiB",
            status=GeneNameStatus.NON_GENE_FEATURE,
            systematic_name="T0004",
            feature_type="pseudogene",
            note="standard name of pseudogene T0004 (not a gene feature)",
        ),
        "T9999": GeneNameResolution(
            input_name="T9999",
            status=GeneNameStatus.RETIRED,
            systematic_name="T9999",
            note="not found in ToyAnno-v1; retained as a legacy systematic name",
        ),
    }


def test_refusal_names_the_subclass_construction(
    tmp_path: Path, resolved: list[tuple[str, str]]
) -> None:
    """A missing root is refused with the toy class and its own init fields (no
    ``go_root``), so the named rebuild call is one this subclass accepts.
    """
    root = str(tmp_path / "absent")
    with pytest.raises(GenomeRootNotFoundError) as refused:
        ToyGenome(genome_root=root)
    assert str(refused.value) == (
        f"genome_root {root!r} does not exist. Pass the existing genome cache "
        f"directory, or build a new one deliberately: "
        f"ToyGenome(genome_root={root!r}, overwrite=True)"
    )


def test_pickle_reopens_through_the_generic_restore(genome: ToyGenome) -> None:
    """``__reduce_ex__`` names the base restore with the subclass's init fields and
    ``overwrite=False``; the round trip yields the same genes, the DAG cleared.
    """
    genome.go_dag  # noqa: B018
    reduced = genome.__reduce_ex__(2)
    assert reduced[0] is base._restore_annotated_genome
    assert reduced[1] == (
        ToyGenome,
        {"genome_root": genome.genome_root, "overwrite": False},
        None,
    )
    restored = pickle.loads(pickle.dumps(genome))
    assert type(restored) is ToyGenome
    assert (restored.genome_root, restored._go_dag) == (genome.genome_root, None)
    assert restored.overwrite is True  # the original value, restored from state
    assert list(restored.gene_set) == ["T0001", "T0002", "T0003"]


def test_remove_deprecated_go_terms_rewrites_only_the_go_attribute(
    genome: ToyGenome,
) -> None:
    """Against the fixture DAG: T0001 keeps GO:0000001 (live) and its non-GO value,
    T0002 loses its only (obsolete) term, T0003's ``Ontology_term`` is untouched; the
    shared ``data.db`` is not written.
    """
    shared = f"{genome.genome_root}/data.db"
    before = database_content_digest(shared)
    genome.remove_deprecated_go_terms()
    assert sorted(genome.db["T0001"].attributes["go_terms"]) == [
        "EC:1.1.1.1",
        "GO:0000001",
    ]
    assert "go_terms" not in genome.db["T0002"].attributes
    assert genome.db["T0003"].attributes["Ontology_term"] == ["GO:0000099"]
    t1, t2 = genome["T0001"], genome["T0002"]
    assert t1 is not None and t2 is not None
    assert (t1.go, t2.go) == (SortedSet(["GO:0000001"]), None)
    assert database_content_digest(shared) == before


def test_drop_empty_go_drops_genes_without_go(genome: ToyGenome) -> None:
    """T0003 carries no ``go_terms``: it leaves the gene set and resolves as RETIRED."""
    genome.drop_empty_go()
    assert list(genome.gene_set) == ["T0001", "T0002"]
    assert genome.resolve_gene_name("T0003").status is GeneNameStatus.RETIRED


def test_get_seq_reads_the_replicon_by_its_chromosome_key(genome: ToyGenome) -> None:
    """``-`` on chromosome 1 [0, 6) is revcomp(ATGAAA) = TTTCAT; the FASTA key is
    refused by name.
    """
    vars(genome)["id"] = "toy"
    assert genome.get_seq(1, 0, 6, "-") == DnaSelectionResult(
        id="toy", chromosome=1, strand="-", start=0, end=6, seq="TTTCAT"
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Chromosome must be one of the chromosome numbers [1], got 'contig1'"
        ),
    ):
        genome.get_seq("contig1", 0, 6, "+")


def test_the_hooks_are_the_abstract_surface() -> None:
    """A subclass must supply exactly these; everything else is shared."""
    assert AnnotatedGenome.__abstractmethods__ == frozenset(
        {"gene_class", "release_files", "fasta_chromosome", "_prepare_go_obo"}
    )
    assert AnnotatedGene.__abstractmethods__ == frozenset({"seqid_to_chromosome"})


def test_base_imports_nothing_organism_specific() -> None:
    """No import in ``base`` reaches an organism subpackage or a download helper."""
    tree = ast.parse(Path(base.__file__).read_text())
    imported = sorted(
        {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
    )
    assert [m for m in imported if m.startswith("torchcell")] == [
        "torchcell.literature.manifest",
        "torchcell.sequence",
        "torchcell.sequence.db_connection",
        "torchcell.sequence.genome.registry",
    ]
    assert "torch_geometric.data" not in imported
