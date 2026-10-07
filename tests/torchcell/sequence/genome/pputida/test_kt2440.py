# tests/torchcell/sequence/genome/pputida/test_kt2440.py
# [[tests.torchcell.sequence.genome.pputida.test_kt2440]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sequence/genome/pputida/test_kt2440.py
"""P. putida KT2440 genome.

The synthetic tests run everywhere: the real class reads a synthetic assembly set
through a stubbed ``resolve`` with every network entry point raising. The tier tests
(``@pytest.mark.data``, skipped when the KT2440 set is absent from
``$DATA_ROOT/torchcell-genomes``) build the genome from the deposited set into a
temporary cache root and pin the measured counts.
"""

import os
import os.path as osp
from collections.abc import Iterator
from pathlib import Path

import pytest
from Bio.Seq import Seq
from sortedcontainers import SortedSet

from tests.torchcell.sequence.genome._bacterial_fixtures import (
    KT2440_GAF,
    KT2440_LOCI,
    SEQUENCE,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.sequence.genome.base import GeneNameStatus
from torchcell.sequence.genome.pputida.kt2440 import (
    KT2440_ASSEMBLY,
    PPutidaKT2440Genome,
)
from torchcell.sequence.genome.registry import GO_RELEASE_20260805, PPUTIDA_KT2440

# --------------------------------------------------------------------------------------
# Synthetic assembly set (runs everywhere)


@pytest.fixture
def tier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """The synthetic KT2440 set served through ``resolve``; the network refuses."""
    files = write_assembly(tmp_path / "tier", KT2440_ASSEMBLY, KT2440_LOCI, KT2440_GAF)
    forbid_network(monkeypatch)
    return serve_tier(monkeypatch, files)


@pytest.fixture
def kt2440(tmp_path: Path, tier: list[tuple[str, str]]) -> PPutidaKT2440Genome:
    """The synthetic KT2440 genome, its ``data.db`` built by the constructor."""
    return PPutidaKT2440Genome(genome_root=str(tmp_path / "kt2440"), overwrite=True)


def test_kt2440_reads_only_its_tier_members(
    tmp_path: Path, tier: list[tuple[str, str]]
) -> None:
    """The GenBank-route members, the GO release and the GOA file, in order."""
    genome = PPutidaKT2440Genome(genome_root=str(tmp_path / "kt2440"), overwrite=True)
    gca = "GCA_000007565.2_ASM756v2"
    assert tier == [
        (PPUTIDA_KT2440, f"{gca}_genomic.fna.gz"),
        (PPUTIDA_KT2440, f"{gca}_genomic.gff.gz"),
        (PPUTIDA_KT2440, f"{gca}_protein.faa.gz"),
        (PPUTIDA_KT2440, f"{gca}_genomic.gbff.gz"),
        (GO_RELEASE_20260805, "go-basic.obo"),
        (PPUTIDA_KT2440, "GCF_000007565.2_ASM756v2_genomic.gff.gz"),
        (PPUTIDA_KT2440, "109.P_putida_KT2440.goa"),
    ]
    assert genome.database_untrusted_reason(str(tmp_path / "kt2440")) is None


def test_kt2440_gene_set_keeps_named_rna_tags(kt2440: PPutidaKT2440Genome) -> None:
    """Numbered and named (rRNA, tRNA) tags are genes; the pseudogene is a locus."""
    assert list(kt2440.gene_set) == [
        "PP_0001",
        "PP_0002",
        "PP_0005",
        "PP_0006",
        "PP_16SA",
        "PP_t01",
    ]
    assert kt2440.genbank.pseudogene_tags == ["PP_0007"]
    assert kt2440.chr_to_nc == {1: "AE015451.2"}


def test_kt2440_gene_and_go(kt2440: PPutidaKT2440Genome) -> None:
    """A minus-strand gene without a symbol, and the GOA terms on column 11 tags."""
    gene = kt2440["PP_0002"]
    assert gene is not None
    assert (gene.chromosome, gene.start, gene.end, gene.strand) == (1, 12, 26, "-")
    assert gene.seq == str(Seq(SEQUENCE[11:26]).reverse_complement())
    assert gene.protein is not None
    assert (gene.symbol, gene.alias, str(gene.protein.seq)) == (None, None, "MAKVF")
    assert gene.go == SortedSet(["GO:0000003"])
    rrna = kt2440["PP_16SA"]
    assert rrna is not None
    assert (rrna.product, rrna.go) == ("16S ribosomal RNA", None)
    assert kt2440.go_source.model_dump(
        include={"rows", "identifiers", "genes_annotated", "terms"}
    ) == {"rows": 3, "identifiers": 3, "genes_annotated": 3, "terms": 2}


def test_kt2440_resolve_gene_name_round_trips(kt2440: PPutidaKT2440Genome) -> None:
    """Tag, named RNA tag, RefSeq tag, a shared symbol, a pseudogene and a fabricated
    tag; E. coli names are another namespace.
    """
    expected = {
        "PP_0001": (GeneNameStatus.CURRENT, "PP_0001", []),
        "PP_16SA": (GeneNameStatus.CURRENT, "PP_16SA", []),
        "parB": (GeneNameStatus.RENAMED, "PP_0001", []),
        "PP_RS00010": (GeneNameStatus.RENAMED, "PP_0002", []),
        "asd": (GeneNameStatus.AMBIGUOUS, None, ["PP_0005", "PP_0006"]),
        "PP_0007": (GeneNameStatus.NON_GENE_FEATURE, "PP_0007", []),
        "PP_9999": (GeneNameStatus.RETIRED, "PP_9999", []),
        "b0002": (GeneNameStatus.RETIRED, "b0002", []),
    }
    for name, (status, systematic, candidates) in expected.items():
        r = kt2440.resolve_gene_name(name)
        assert (r.status, r.systematic_name, r.candidates) == (
            status,
            systematic,
            candidates,
        ), name


# --------------------------------------------------------------------------------------
# The deposited tier (data-gated)

DATA_ROOT = os.environ.get("DATA_ROOT", "")
TIER_PRESENT = all(
    osp.isfile(osp.join(DATA_ROOT, "torchcell-genomes", s, "manifest.json"))
    for s in (PPUTIDA_KT2440, GO_RELEASE_20260805)
)


@pytest.fixture(scope="module")
def tier_kt2440(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[PPutidaKT2440Genome]:
    """KT2440 from the tier into a temporary cache root, with the network refusing."""
    with pytest.MonkeyPatch.context() as mp:
        forbid_network(mp)
        genome = PPutidaKT2440Genome(
            genome_root=str(tmp_path_factory.mktemp("kt2440")), overwrite=True
        )
    yield genome


class TestTierKT2440:
    """GCA_000007565.2 as deposited."""

    pytestmark = [
        pytest.mark.data,
        pytest.mark.skipif(
            not TIER_PRESENT,
            reason="requires the KT2440 assembly set under $DATA_ROOT/torchcell-genomes",
        ),
    ]

    def test_gene_features_from_the_genbank_route(
        self, tier_kt2440: PPutidaKT2440Genome
    ) -> None:
        """5,786 gene features: 5,729 genes (165 of them named RNA tags) and 57
        pseudogenes; 5,564 CDS with their proteins.
        """
        loci = tier_kt2440.genbank.loci
        assert len(loci) == 5786
        assert len(tier_kt2440.gene_set) == 5729
        assert len(tier_kt2440.genbank.pseudogene_tags) == 57
        assert sum(not tag[3:].isdigit() for tag in loci) == 165
        assert (len(tier_kt2440.fasta_cds), len(tier_kt2440.fasta_protein)) == (
            5564,
            5564,
        )

    def test_go_coverage_of_the_goa_proteome_file(
        self, tier_kt2440: PPutidaKT2440Genome
    ) -> None:
        """Every one of the 25,276 rows carries a ``PP_`` tag: 3,912 tags, all genes,
        2,619 terms.
        """
        assert tier_kt2440.go_source.model_dump(exclude={"identifier"}) == {
            "route": "gaf_synonym_column",
            "assembly_set": PPUTIDA_KT2440,
            "member": "109.P_putida_KT2440.goa",
            "sha256": "575731316d9fcb98580dd7e2209a0239c909ada389e42c4052e5a8f7a1069a81",
            "rows": 25276,
            "not_rows_excluded": 0,
            "rows_without_identifier": 0,
            "identifiers": 3912,
            "identifiers_not_in_annotation": (),
            "pseudogenes_annotated": 0,
            "genes_annotated": 3912,
            "terms": 2619,
        }

    def test_resolve_gene_name_round_trips(
        self, tier_kt2440: PPutidaKT2440Genome
    ) -> None:
        """PP_0002 current; parA and PP_RS00010 renamed to it; named RNA tags current;
        a fabricated tag and E. coli names retired.
        """
        resolve = tier_kt2440.resolve_gene_name
        for name in ("PP_0002", "PP_16SA", "PP_23SB", "PP_t01", "PP_tm01", "PP_mr01"):
            assert resolve(name).status == GeneNameStatus.CURRENT, name
        for name in ("parA", "PP_RS00010"):
            r = resolve(name)
            assert (r.status, r.systematic_name) == (
                GeneNameStatus.RENAMED,
                "PP_0002",
            ), name
        for name in ("PP_9999", "b0002", "ECK0002"):
            assert resolve(name).status == GeneNameStatus.RETIRED, name

    def test_cds_translates_to_the_rekeyed_protein(
        self, tier_kt2440: PPutidaKT2440Genome
    ) -> None:
        """Every CDS translates to its protein except the selenoprotein fdoG."""
        mismatches = [
            tag
            for tag, cds in tier_kt2440.fasta_cds.items()
            if "M" + str(cds.seq.translate(table=11)).rstrip("*")[1:]
            != str(tier_kt2440.fasta_protein[tag].seq)
        ]
        assert mismatches == ["PP_0489"]
