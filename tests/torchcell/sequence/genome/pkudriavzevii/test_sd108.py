# tests/torchcell/sequence/genome/pkudriavzevii/test_sd108.py
# [[tests.torchcell.sequence.genome.pkudriavzevii.test_sd108]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sequence/genome/pkudriavzevii/test_sd108.py
"""P. kudriavzevii SD108 v1 genome.

The hermetic tests pin what decides whether a gene lands on the right replicon and in
the right identifier space: the id pattern, the replicon table, the member names, and
the readers' refusals. The tier tests (``@pytest.mark.data``, skipped when the SD108 set
is absent from ``$DATA_ROOT/torchcell-genomes``) build the genome from the deposited set
into a temporary cache root and pin the measured counts.
"""

from __future__ import annotations

import os
import os.path as osp
import pickle
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from torchcell.sequence.genome.bacterial import GenomeAnnotationMismatchError
from torchcell.sequence.genome.pkudriavzevii.sd108 import (
    GENE_XREF,
    GENOME_FAI,
    GFF3,
    MITOCHONDRION,
    RDNA_UNIT,
    SD108_GENE_ID_PATTERN,
    SD108_REPLICONS,
    PKudriavzeviiSD108Gene,
    PKudriavzeviiSD108Genome,
    read_fai,
    read_gene_xref,
    rekey_by_gene,
)
from torchcell.sequence.genome.registry import (
    GO_RELEASE_20260805,
    PKUDRIAVZEVII_SD108,
    resolve,
)

# --------------------------------------------------------------------------------------
# Hermetic


@pytest.mark.parametrize(
    "gene_id",
    [
        "SD108_X1G0001",
        "SD108_X2G1933",
        "SD108_X5G5144",
        "SD108_X1T0001",
        "SD108_X3R0005",
        "SD108_XMG5147",
        "SD108_XMT0175",
        "SD108_XMR0009",
        "SD108_XUR0011",
        "SD108_X1G0123A",
    ],
)
def test_the_id_pattern_admits_every_gene_form(gene_id: str) -> None:
    """Chromosomes 1-5, mito and rDNA; protein-coding, tRNA and rRNA; the letter rule."""
    assert re.fullmatch(SD108_GENE_ID_PATTERN, gene_id)


@pytest.mark.parametrize(
    "not_a_gene_id",
    [
        "SD108_X2G1933.t1",
        "SD108_X2G1933.t1.cds",
        "SD108_X2G1933_H2",
        "SD108_X6G0001",
        "SD108_X2G193",
        "SD108_02G05860",
        "URA3",
        "YEL021W",
        "JL09_g1026",
    ],
)
def test_the_id_pattern_refuses_transcripts_alleles_and_other_namespaces(
    not_a_gene_id: str,
) -> None:
    """A transcript, a haplotype-2 allele, an old-scheme id and a foreign name are not
    gene ids of the primary assembly.
    """
    assert not re.fullmatch(SD108_GENE_ID_PATTERN, not_a_gene_id)


def test_the_replicon_table_keys_five_chromosomes_the_mito_and_the_rdna_unit() -> None:
    """Chromosomes take their number, the mitochondrion 0 and the rDNA unit 6."""
    assert dict(SD108_REPLICONS) == {
        "SD108_chr1": 1,
        "SD108_chr2": 2,
        "SD108_chr3": 3,
        "SD108_chr4": 4,
        "SD108_chr5": 5,
        "SD108_mito": MITOCHONDRION,
        "SD108_rDNA_unit": RDNA_UNIT,
    }
    assert (MITOCHONDRION, RDNA_UNIT) == (0, 6)
    assert PKudriavzeviiSD108Gene.seqid_to_chromosome("SD108_chr2") == 2
    with pytest.raises(ValueError, match="is not one of the replicons"):
        PKudriavzeviiSD108Gene.seqid_to_chromosome("SD108_chr1_H2")


def test_the_genome_class_declares_its_set_release_and_files() -> None:
    """What the class reads from which set, and its default cache root."""
    assert PKudriavzeviiSD108Genome.ASSEMBLY_SET == PKUDRIAVZEVII_SD108
    assert PKUDRIAVZEVII_SD108 == "pkudriavzevii_SD108_v1"
    assert PKudriavzeviiSD108Genome.GENOME_VERSION == "v1"
    files = PKudriavzeviiSD108Genome.release_files()
    assert files.model_dump() == {
        "dna_fasta": "SD108_genome_v1.fasta",
        "gff": "SD108_genome_v1.gff3",
        "protein_fasta": "proteins.faa",
        "cds_fasta": "cds.fna",
    }
    assert PKudriavzeviiSD108Genome.LOCUS_FEATURE_TYPES == frozenset({"gene"})
    assert "no GO release is keyed" in PKudriavzeviiSD108Genome.GO_ABSENCE
    defaults = {
        a.name: a.default for a in PKudriavzeviiSD108Genome.__attrs_attrs__ if a.init
    }
    assert defaults == {
        "genome_root": "data/pkudriavzevii/sd108/genome",
        "overwrite": False,
    }


def test_read_fai(tmp_path: Path) -> None:
    """Name to length; an empty index is refused."""
    path = tmp_path / "g.fasta.fai"
    path.write_text("SD108_chr1\t100\t12\t80\t81\nSD108_mito\t50\t200\t80\t81\n")
    assert read_fai(str(path)) == {"SD108_chr1": 100, "SD108_mito": 50}
    path.write_text("")
    with pytest.raises(ValueError, match="lists no sequence"):
        read_fai(str(path))


XREF_HEADER = (
    "gene_id\tsc_gene\tsc_systematic\tsc_name_source\tsc_name_check\tsc_identity\t"
    "sc_coverage_sd108\tsc_coverage_scer\tsc_evalue\tsc_best_hit\n"
)


def test_read_gene_xref_keeps_every_id_and_crosswalks_only_named_genes(
    tmp_path: Path,
) -> None:
    """A named gene carries its evidence; an unnamed gene has no crosswalk entry."""
    path = tmp_path / GENE_XREF
    path.write_text(
        XREF_HEADER
        + "SD108_X2G1933\tURA3\tYEL021W\tCBS573_annotation_note\tconfirmed\t72.2\t0.99"
        "\t0.97\t1.4e-137\t\n" + "SD108_X1T0001\t\t\t\t\t\t\t\t\t\n"
    )
    ids, crosswalk = read_gene_xref(str(path))
    assert ids == ["SD108_X2G1933", "SD108_X1T0001"]
    assert set(crosswalk) == {"SD108_X2G1933"}
    assert crosswalk["SD108_X2G1933"].model_dump() == {
        "sc_gene": "URA3",
        "sc_systematic": "YEL021W",
        "sc_name_source": "CBS573_annotation_note",
        "sc_name_check": "confirmed",
        "sc_identity": 72.2,
        "sc_coverage_sd108": 0.99,
        "sc_coverage_scer": 0.97,
        "sc_evalue": 1.4e-137,
        "sc_best_hit": None,
    }


def test_read_gene_xref_refuses_a_name_without_its_evidence(tmp_path: Path) -> None:
    """A name with no sc_name_check would be carried without what qualifies it."""
    path = tmp_path / GENE_XREF
    path.write_text(XREF_HEADER + "SD108_X2G1933\tURA3\tYEL021W\tx\t\t\t\t\t\t\n")
    with pytest.raises(GenomeAnnotationMismatchError, match="no sc_name_check"):
        read_gene_xref(str(path))


def _records(*ids: str) -> dict[str, SeqRecord]:
    return {i: SeqRecord(Seq("M"), id=i, description=i) for i in ids}


def test_rekey_by_gene_maps_transcripts_to_genes() -> None:
    """``<gene>.t1`` becomes the gene id, on the key and on the record."""
    out = rekey_by_gene(
        _records("SD108_X2G1933.t1", "SD108_XMG5147.t1"),
        {"SD108_X2G1933", "SD108_XMG5147"},
        "proteins.faa",
    )
    assert sorted(out) == ["SD108_X2G1933", "SD108_XMG5147"]
    assert out["SD108_X2G1933"].id == "SD108_X2G1933"


def test_rekey_by_gene_refuses_an_unmapped_record_and_a_gene_without_one() -> None:
    """A record of no coding gene, and a coding gene with no record, are each refused."""
    with pytest.raises(GenomeAnnotationMismatchError, match="map to no protein-coding"):
        rekey_by_gene(
            _records("SD108_X2G1933.t1", "SD108_X1T0001.t1"),
            {"SD108_X2G1933"},
            "proteins.faa",
        )
    with pytest.raises(GenomeAnnotationMismatchError, match="map to no protein-coding"):
        rekey_by_gene(_records("SD108_X2G1933"), {"SD108_X2G1933"}, "proteins.faa")
    with pytest.raises(GenomeAnnotationMismatchError, match="have no record"):
        rekey_by_gene(
            _records("SD108_X2G1933.t1"),
            {"SD108_X2G1933", "SD108_X2G1934"},
            "proteins.faa",
        )


# --------------------------------------------------------------------------------------
# The deposited tier (data-gated)

DATA_ROOT = os.environ.get("DATA_ROOT", "")
TIER_PRESENT = all(
    osp.isfile(osp.join(DATA_ROOT, "torchcell-genomes", s, "manifest.json"))
    for s in (PKUDRIAVZEVII_SD108, GO_RELEASE_20260805)
)


@pytest.fixture(scope="module")
def sd108(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[PKudriavzeviiSD108Genome]:
    """SD108 from the tier into a temporary cache root."""
    yield PKudriavzeviiSD108Genome(
        genome_root=str(tmp_path_factory.mktemp("sd108")), overwrite=True
    )


def _gff_rows(path: str) -> list[list[str]]:
    with open(path) as handle:
        return [line.rstrip("\n").split("\t") for line in handle if line[0] != "#"]


class TestTierSD108:
    """``pkudriavzevii_SD108_v1`` as deposited from commit 8ca6a106."""

    pytestmark = [
        pytest.mark.data,
        pytest.mark.skipif(
            not TIER_PRESENT,
            reason="requires the SD108 assembly set under $DATA_ROOT/torchcell-genomes",
        ),
    ]

    def test_replicons_and_lengths_are_the_fai(
        self, sd108: PKudriavzeviiSD108Genome
    ) -> None:
        """Seven sequences, each the length the deposited ``.fai`` lists."""
        fai = read_fai(resolve(PKUDRIAVZEVII_SD108, GENOME_FAI))
        assert set(fai) == set(SD108_REPLICONS)
        assert sd108.chr_to_len == {
            SD108_REPLICONS[n]: n_len for n, n_len in fai.items()
        }
        assert fai == {
            "SD108_chr1": 2_844_978,
            "SD108_chr2": 2_727_922,
            "SD108_chr3": 2_555_198,
            "SD108_chr4": 1_375_460,
            "SD108_chr5": 1_291_925,
            "SD108_mito": 50_964,
            "SD108_rDNA_unit": 22_471,
        }

    def test_gene_counts_by_biotype_and_proteins(
        self, sd108: PKudriavzeviiSD108Genome
    ) -> None:
        """5,376 genes: 5,159 protein-coding (5,144 nuclear + 15 mitochondrial), 199
        tRNA, 18 rRNA; one protein and one CDS per protein-coding gene.
        """
        assert len(sd108.gene_set) == 5376
        biotypes = Counter(
            f.attributes["gene_biotype"][0] for f in sd108.db.features_of_type("gene")
        )
        assert biotypes == {"protein_coding": 5159, "tRNA": 199, "rRNA": 18}
        coding = {
            f.id
            for f in sd108.db.features_of_type("gene")
            if f.attributes["gene_biotype"][0] == "protein_coding"
        }
        assert sum(g.startswith("SD108_XM") for g in coding) == 15
        assert set(sd108.fasta_protein) == coding == set(sd108.fasta_cds)

    def test_the_id_pattern_matches_every_gene_id_and_nothing_else(self) -> None:
        """Every gene row's ID matches; no mRNA, exon or CDS ID does."""
        path = resolve(PKUDRIAVZEVII_SD108, GFF3)
        assert osp.basename(osp.dirname(path)) == PKUDRIAVZEVII_SD108
        rows = _gff_rows(path)
        by_type: dict[str, list[str]] = {}
        for row in rows:
            (row_id,) = (kv[3:] for kv in row[8].split(";") if kv.startswith("ID="))
            by_type.setdefault(row[2], []).append(row_id)
        assert set(by_type) == {"gene", "mRNA", "tRNA", "rRNA", "exon", "CDS"}
        assert len(by_type["gene"]) == 5376
        assert all(re.fullmatch(SD108_GENE_ID_PATTERN, g) for g in by_type["gene"])
        assert not any(
            re.fullmatch(SD108_GENE_ID_PATTERN, i)
            for t, row_ids in by_type.items()
            if t != "gene"
            for i in row_ids
        )
        # v1 issues no letter-suffixed id
        assert not any(g[-1].isalpha() for g in by_type["gene"])

    def test_ura3(self, sd108: PKudriavzeviiSD108Genome) -> None:
        """SD108_X2G1933 is URA3 by crosswalk: chr2:1,175,606-1,176,394, minus strand,
        262 aa, its coding span equal to its ``cds.fna`` record.
        """
        gene = sd108["SD108_X2G1933"]
        assert gene is not None
        assert (gene.chromosome, gene.start, gene.end, gene.strand) == (
            2,
            1_175_606,
            1_176_394,
            "-",
        )
        assert (gene.locus_start, gene.locus_end) == (1_175_606, 1_176_394)
        assert gene.protein is not None
        assert len(gene.protein.seq) == 262
        assert gene.seq == str(gene.cds.seq)
        assert gene.seq.startswith("ATG")
        assert gene.biotype == "protein_coding"
        assert gene.sc_gene == "URA3"
        assert gene.sc_systematic == "YEL021W"
        assert gene.sc_crosswalk is not None
        assert gene.sc_crosswalk.sc_name_check == "confirmed"
        assert gene.go is None

    def test_a_gene_with_utrs_reads_its_coding_span(
        self, sd108: PKudriavzeviiSD108Genome
    ) -> None:
        """SD108_X1G0026: gene row 53,808-58,292, two CDS rows 54,110-54,136 and
        54,218-58,192; the gene's span is the CDS span.
        """
        gene = sd108["SD108_X1G0026"]
        assert gene is not None
        assert (gene.start, gene.end) == (54_110, 58_192)
        assert (gene.locus_start, gene.locus_end) == (53_808, 58_292)

    def test_comma_names_are_rejoined(self, sd108: PKudriavzeviiSD108Genome) -> None:
        """The three names with a literal comma read whole, as gene_xref.tsv has them."""
        names = {
            g: c.sc_gene for g, c in sd108.sc_crosswalk.items() if "," in c.sc_gene
        }
        assert sorted(names.values()) == ["ADE5,7", "ARG5,6", "DUR1,2"]
        for gene_id, name in names.items():
            gene = sd108[gene_id]
            assert gene is not None and gene.sc_gene == name

    def test_crosswalk_is_an_attribute_never_a_key(
        self, sd108: PKudriavzeviiSD108Genome
    ) -> None:
        """3,471 named genes; a S. cerevisiae name or a transcript id is no key."""
        assert len(sd108.sc_crosswalk) == 3471
        assert Counter(c.sc_name_check for c in sd108.sc_crosswalk.values()) == {
            "confirmed": 3193,
            "best_hit_weak": 233,
            "paralog": 45,
        }
        assert sd108["URA3"] is None
        assert sd108["YEL021W"] is None
        assert sd108["SD108_X2G1933.t1"] is None
        assert sd108["SD108_X9G0001"] is None

    def test_go_is_stated_absent(self, sd108: PKudriavzeviiSD108Genome) -> None:
        """No GO anywhere, and the record names the set and the reason."""
        assert sd108.go_annotations == {}
        assert len(sd108.go) == 0
        assert sd108.go_source.model_dump() == {
            "assembly_set": PKUDRIAVZEVII_SD108,
            "members_inspected": (GFF3, GENE_XREF),
            "reason": PKudriavzeviiSD108Genome.GO_ABSENCE,
        }

    def test_rna_and_mitochondrial_genes(self, sd108: PKudriavzeviiSD108Genome) -> None:
        """A tRNA has no protein; mitochondrial COX1 sits on key 0 with its protein."""
        trna = sd108["SD108_X1T0001"]
        assert trna is not None
        assert (trna.biotype, trna.gene_name, trna.protein) == (
            "tRNA",
            "tRNA-Gly(GCC)",
            None,
        )
        cox1 = sd108["SD108_XMG5147"]
        assert cox1 is not None
        assert (cox1.chromosome, cox1.gene_name, cox1.sc_gene) == (
            MITOCHONDRION,
            "COX1",
            None,
        )
        assert cox1.protein is not None
        rdna = [g for g in sd108.gene_set if g.startswith("SD108_XU")]
        assert len(rdna) == 8
        first = sd108[rdna[0]]
        assert first is not None and first.chromosome == RDNA_UNIT

    def test_pickle_round_trip(self, sd108: PKudriavzeviiSD108Genome) -> None:
        """An unpickled genome reads the same genes and crosswalk."""
        copy = pickle.loads(pickle.dumps(sd108))
        assert copy.gene_set == sd108.gene_set
        assert copy.sc_crosswalk == sd108.sc_crosswalk
        gene = copy["SD108_X2G1933"]
        assert gene is not None and (gene.start, gene.sc_gene) == (1_175_606, "URA3")
