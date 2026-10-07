# tests/torchcell/sequence/genome/ecoli/test_rel606.py
# [[tests.torchcell.sequence.genome.ecoli.test_rel606]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sequence/genome/ecoli/test_rel606.py
"""E. coli B REL606 genome.

The synthetic tests run everywhere: the real class reads a synthetic assembly set
written by ``_bacterial_fixtures`` through a stubbed ``resolve``, with every network
entry point raising. The tier tests (``@pytest.mark.data``, skipped when the REL606 set
is absent from ``$DATA_ROOT/torchcell-genomes``) build the genome from the deposited set
into a temporary cache root and pin the counts measured on 2026-10-07.
"""

import os
import os.path as osp
from collections.abc import Iterator
from pathlib import Path

import pytest
from Bio.Seq import Seq
from sortedcontainers import SortedSet

from tests.torchcell.sequence.genome._bacterial_fixtures import (
    REL606_LOCI,
    SEQUENCE,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.sequence.genome.base import GeneNameStatus, read_genome_database_record
from torchcell.sequence.genome.ecoli import (
    REL606_ASSEMBLY,
    EcoliBREL606Gene,
    EcoliBREL606Genome,
)
from torchcell.sequence.genome.registry import ECOLI_B_REL606, GO_RELEASE_20260805

GCA = "GCA_000017985.1_ASM1798v1"
GCF = "GCF_000017985.1_ASM1798v1"

# --------------------------------------------------------------------------------------
# Synthetic assembly set (runs everywhere)


@pytest.fixture
def tier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """The synthetic REL606 set served through ``resolve``; the network refuses."""
    files = write_assembly(tmp_path / "tier", REL606_ASSEMBLY, REL606_LOCI)
    forbid_network(monkeypatch)
    return serve_tier(monkeypatch, files)


@pytest.fixture
def rel606(tmp_path: Path, tier: list[tuple[str, str]]) -> EcoliBREL606Genome:
    """The synthetic REL606 genome, its ``data.db`` built by the constructor."""
    return EcoliBREL606Genome(genome_root=str(tmp_path / "rel606"), overwrite=True)


def test_rel606_reads_only_its_tier_members_and_records_its_set(
    tmp_path: Path, tier: list[tuple[str, str]]
) -> None:
    """The GenBank-route members, the GO release, then the RefSeq GFF twice (the
    crosswalk, then the GO route that reads it); ``data.db`` records the GCA GFF.
    """
    root = tmp_path / "rel606"
    genome = EcoliBREL606Genome(genome_root=str(root), overwrite=True)
    assert tier == [
        (ECOLI_B_REL606, f"{GCA}_genomic.fna.gz"),
        (ECOLI_B_REL606, f"{GCA}_genomic.gff.gz"),
        (ECOLI_B_REL606, f"{GCA}_protein.faa.gz"),
        (ECOLI_B_REL606, f"{GCA}_genomic.gbff.gz"),
        (GO_RELEASE_20260805, "go-basic.obo"),
        (ECOLI_B_REL606, f"{GCF}_genomic.gff.gz"),
        (ECOLI_B_REL606, f"{GCF}_genomic.gff.gz"),
    ]
    record = read_genome_database_record(str(root / "data.db"))
    assert record is not None
    assert record.source.model_dump(include={"assembly_set", "gff_filename"}) == {
        "assembly_set": ECOLI_B_REL606,
        "gff_filename": f"{GCA}_genomic.gff.gz",
    }
    assert EcoliBREL606Genome.database_untrusted_reason(str(root)) is None
    assert (genome.strain, genome.GENOME_VERSION, genome.ANNOTATION_RELEASE) == (
        "B REL606",
        "ASM1798v1",
        GCA,
    )


def test_rel606_assembly_names_its_set_replicon_pattern_and_cache_root() -> None:
    """CP000819.1, the five-digit ``ECB_`` pattern, the RefSeq GFF GO route and the
    ``data/ecoli/rel606/genome`` default root.
    """
    assert REL606_ASSEMBLY.model_dump() == {
        "organism": "Escherichia coli",
        "strain": "B REL606",
        "assembly_set": ECOLI_B_REL606,
        "genbank_assembly": GCA,
        "refseq_assembly": GCF,
        "replicon": "CP000819.1",
        "locus_tag_pattern": r"ECB_[rt]?\d{5}",
        "go_source": {
            "route": "refseq_gff_ontology_term",
            "assembly_set": ECOLI_B_REL606,
            "member": f"{GCF}_genomic.gff.gz",
            "identifier_pattern": None,
        },
        "default_genome_root": "data/ecoli/rel606/genome",
    }
    assert EcoliBREL606Genome.gene_class() is EcoliBREL606Gene
    assert EcoliBREL606Gene.seqid_to_chromosome("CP000819.1") == 1
    with pytest.raises(ValueError, match="seqid 'NC_012967.1' is not one of"):
        EcoliBREL606Gene.seqid_to_chromosome("NC_012967.1")


def test_rel606_gene_set_keeps_trna_and_rrna_tags(rel606: EcoliBREL606Genome) -> None:
    """Numbered, tRNA and rRNA tags are genes; the pseudogene is a locus only."""
    assert list(rel606.gene_set) == [
        "ECB_00001",
        "ECB_00002",
        "ECB_00003",
        "ECB_r00001",
        "ECB_t00001",
        "ECB_t00002",
    ]
    assert rel606.genbank.pseudogene_tags == ["ECB_00042"]
    assert rel606.chr_to_nc == {1: "CP000819.1"}
    assert rel606.chr_to_len == {1: 100}


def test_rel606_gene_carries_coordinates_sequences_names_and_go(
    rel606: EcoliBREL606Genome,
) -> None:
    """A minus-strand gene: reverse-complemented sequence, its re-keyed protein, the
    GenBank symbol (no synonyms in this annotation) and the RefSeq inline GO.
    """
    gene = rel606["ECB_00002"]
    assert gene is not None
    expected = str(Seq(SEQUENCE[11:26]).reverse_complement())
    assert (gene.chromosome, gene.start, gene.end, gene.strand) == (1, 12, 26, "-")
    assert gene.seq == expected
    assert gene.protein is not None
    assert str(gene.protein.seq) == "MRVLK"
    assert (gene.symbol, gene.synonyms, gene.alias, gene.protein_id) == (
        "thrA",
        [],
        None,
        "ACT37695.1",
    )
    assert gene.go == SortedSet(["GO:0000002", "GO:0000003"])
    rrna = rel606["ECB_r00001"]
    assert rrna is not None
    assert (rrna.product, rrna.go, rrna.protein) == ("16S ribosomal RNA", None, None)


def test_rel606_go_is_the_refseq_inline_route(rel606: EcoliBREL606Genome) -> None:
    """Four ``Ontology_term`` rows: three reach GenBank loci through ``old_locus_tag``
    (two genes, one pseudogene), the RefSeq-only gene's row reaches none.
    """
    assert rel606.go_source.model_dump(exclude={"sha256"}) == {
        "route": "refseq_gff_ontology_term",
        "assembly_set": ECOLI_B_REL606,
        "member": f"{GCF}_genomic.gff.gz",
        "identifier": (
            "RefSeq GFF Ontology_term row's locus_tag, through its gene row's "
            "old_locus_tag"
        ),
        "rows": 4,
        "not_rows_excluded": 0,
        "rows_without_identifier": 1,
        "identifiers": 3,
        "identifiers_not_in_annotation": (),
        "pseudogenes_annotated": 1,
        "genes_annotated": 2,
        "terms": 3,
    }
    rel606.remove_deprecated_go_terms()
    assert dict(rel606.go_annotations) == {
        "ECB_00001": SortedSet(["GO:0000001"]),
        "ECB_00002": SortedSet(["GO:0000003"]),
        "ECB_00042": SortedSet(["GO:0000001"]),
    }


def test_rel606_resolve_gene_name_round_trips(rel606: EcoliBREL606Genome) -> None:
    """Locus tags (numbered, tRNA, rRNA), a symbol, a RefSeq tag, a shared tRNA symbol,
    the pseudogene, a retired tag, and the K-12 namespaces, which are not REL606's.
    """
    expected = {
        "ECB_00002": (GeneNameStatus.CURRENT, "ECB_00002", [], None),
        "ECB_t00001": (GeneNameStatus.CURRENT, "ECB_t00001", [], None),
        "ECB_r00001": (GeneNameStatus.CURRENT, "ECB_r00001", [], None),
        "thrA": (
            GeneNameStatus.RENAMED,
            "ECB_00002",
            [],
            "gene symbol of current gene ECB_00002",
        ),
        "ECB_RS00035": (
            GeneNameStatus.RENAMED,
            "ECB_00003",
            [],
            "RefSeq locus tag of current gene ECB_00003",
        ),
        "metZ": (
            GeneNameStatus.AMBIGUOUS,
            None,
            ["ECB_t00001", "ECB_t00002"],
            "gene symbol of multiple current genes",
        ),
        "caiB": (
            GeneNameStatus.NON_GENE_FEATURE,
            "ECB_00042",
            [],
            "gene symbol of pseudogene ECB_00042 (not a gene feature)",
        ),
        "ECB_99999": (
            GeneNameStatus.RETIRED,
            "ECB_99999",
            [],
            f"not found in {GCA}; retained as given",
        ),
        "b0002": (
            GeneNameStatus.RETIRED,
            "b0002",
            [],
            f"not found in {GCA}; retained as given",
        ),
    }
    for name, (status, systematic, candidates, note) in expected.items():
        r = rel606.resolve_gene_name(name)
        assert (r.status, r.systematic_name, r.candidates, r.note) == (
            status,
            systematic,
            candidates,
            note,
        ), name


# --------------------------------------------------------------------------------------
# The deposited tier (data-gated)

DATA_ROOT = os.environ.get("DATA_ROOT", "")
TIER_PRESENT = all(
    osp.isfile(osp.join(DATA_ROOT, "torchcell-genomes", s, "manifest.json"))
    for s in (ECOLI_B_REL606, GO_RELEASE_20260805)
)


@pytest.fixture(scope="module")
def tier_rel606(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[EcoliBREL606Genome]:
    """REL606 from the tier into a temporary cache root, with the network refusing."""
    with pytest.MonkeyPatch.context() as mp:
        forbid_network(mp)
        genome = EcoliBREL606Genome(
            genome_root=str(tmp_path_factory.mktemp("rel606")), overwrite=True
        )
    yield genome


class TestTierREL606:
    """GCA_000017985.1 as deposited."""

    pytestmark = [
        pytest.mark.data,
        pytest.mark.skipif(
            not TIER_PRESENT,
            reason="requires the REL606 assembly set under $DATA_ROOT/torchcell-genomes",
        ),
    ]

    def test_gene_features_from_the_genbank_route(
        self, tier_rel606: EcoliBREL606Genome
    ) -> None:
        """4,383 gene features (4,276 numbered, 85 tRNA, 22 rRNA): 4,316 genes and 67
        pseudogenes, none joined; 4,209 CDS with their proteins; no ``gene_synonym``
        and no ``old_locus_tag`` in the GenBank file.
        """
        loci = tier_rel606.genbank.loci
        assert len(loci) == 4383
        assert len(tier_rel606.gene_set) == 4316
        assert len(tier_rel606.genbank.pseudogene_tags) == 67
        assert [sum(t[4] == k for t in loci) for k in "tr"] == [85, 22]
        assert sum(locus.segments > 1 for locus in loci.values()) == 0
        assert (len(tier_rel606.fasta_cds), len(tier_rel606.fasta_protein)) == (
            4209,
            4209,
        )
        assert sum(len(locus.synonyms) for locus in loci.values()) == 0
        assert sum(len(locus.old_locus_tags) for locus in loci.values()) == 0

    def test_go_coverage_of_the_refseq_inline_route(
        self, tier_rel606: EcoliBREL606Genome
    ) -> None:
        """2,325 ``Ontology_term`` rows, 37 on RefSeq-only genes; 2,253 GenBank loci
        reached, 2,232 of them genes, with 1,649 terms.
        """
        assert tier_rel606.go_source.model_dump(exclude={"identifier"}) == {
            "route": "refseq_gff_ontology_term",
            "assembly_set": ECOLI_B_REL606,
            "member": f"{GCF}_genomic.gff.gz",
            "sha256": "27c302a37ac517de79999cc8438c744367e5c12ad4ac60b34f55bfd753214f25",
            "rows": 2325,
            "not_rows_excluded": 0,
            "rows_without_identifier": 37,
            "identifiers": 2253,
            "identifiers_not_in_annotation": (),
            "pseudogenes_annotated": 21,
            "genes_annotated": 2232,
            "terms": 1649,
        }
        assert len(tier_rel606.refseq.old_locus_tags) == 4507
        assert sum(
            1 for olds in tier_rel606.refseq.old_locus_tags.values() if olds
        ) == (4253)

    def test_obsolete_terms_leave_with_remove_deprecated_go_terms(
        self, tier_rel606: EcoliBREL606Genome
    ) -> None:
        """50 RefSeq terms are obsolete in go-basic 2026-07-26; removing them leaves
        2,230 genes with 1,599 terms.
        """
        tier_rel606.remove_deprecated_go_terms()
        genes = set(tier_rel606.gene_set)
        kept = {t: v for t, v in tier_rel606.go_annotations.items() if t in genes}
        assert (len(kept), len({x for v in kept.values() for x in v})) == (2230, 1599)

    def test_resolve_gene_name_round_trips(
        self, tier_rel606: EcoliBREL606Genome
    ) -> None:
        """ECB_00002 current; thrA and its RefSeq tag renamed to it; tRNA and rRNA tags
        current; a fabricated tag and the K-12 namespaces retired.
        """
        resolve = tier_rel606.resolve_gene_name
        for name in ("ECB_00002", "ECB_t00001", "ECB_r00001"):
            r = resolve(name)
            assert (r.status, r.systematic_name) == (GeneNameStatus.CURRENT, name)
        for name in ("thrA", "ECB_RS00010"):
            r = resolve(name)
            assert (r.status, r.systematic_name) == (
                GeneNameStatus.RENAMED,
                "ECB_00002",
            ), name
        for name in ("ECB_99999", "b0002", "BW25113_0002", "ECK0002"):
            r = resolve(name)
            assert (r.status, r.systematic_name) == (GeneNameStatus.RETIRED, name)

    def test_cds_translates_to_the_rekeyed_protein(
        self, tier_rel606: EcoliBREL606Genome
    ) -> None:
        """Every CDS translates (table 11, start read as M) to its protein except the
        three selenoproteins, as in MG1655.
        """
        mismatches = [
            tag
            for tag, cds in tier_rel606.fasta_cds.items()
            if "M" + str(cds.seq.translate(table=11)).rstrip("*")[1:]
            != str(tier_rel606.fasta_protein[tag].seq)
        ]
        assert mismatches == ["ECB_01432", "ECB_03779", "ECB_03951"]
        assert [tier_rel606.genbank.loci[t].symbol for t in mismatches] == [
            "fdnG",
            "fdoG",
            "fdhF",
        ]
