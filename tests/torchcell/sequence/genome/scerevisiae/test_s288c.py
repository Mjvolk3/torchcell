"""Tests for the S. cerevisiae S288C genome wrapper."""

import os
import os.path as osp

import pytest
from dotenv import load_dotenv

from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

load_dotenv()

DATA_ROOT = os.getenv("DATA_ROOT")

# Skip the whole module where DATA_ROOT is unset (CI): these tests build a real
# SCerevisiaeGenome from the SGD genome data under DATA_ROOT. Without a module-level
# guard, the genome fixture's assert would surface as an ERROR (not a skip) for every
# test. Mirrors the guard idiom in tests/torchcell/data/test_cell_data.py.
if DATA_ROOT is None:
    pytest.skip("requires DATA_ROOT data (absent in CI)", allow_module_level=True)


@pytest.fixture
def genome():
    # This fixture builds a real SCerevisiaeGenome from the SGD genome data under
    # DATA_ROOT, so it only runs where that dataset is present. Narrow the env
    # lookup to str so the osp.join calls below type-check under strict mypy.
    assert DATA_ROOT is not None, "DATA_ROOT must be set to run the S288C genome tests"
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
    )
    return genome


def test_gene_YAL037W(genome):
    # well behaved, CDS is gene
    gene = genome["YAL037W"]
    assert gene.id == "YAL037W"
    assert gene.chromosome == 1
    assert gene.strand == "+"
    assert gene.start == 74020
    assert gene.end == 74823
    assert (
        gene.seq
        == "ATGGATATGGAAATCGAAGATTCAAGCCCCATAGATGACCTGAAGTTACAAAAACTGGATACCAATGTTTATTTTGGACCCTGTGAGATATTGACACAACCTATTCTTTTGCAATATGAAAATATTAAGTTCATCATTGGTGTCAATCTAAGTACTGAAAAGATAGCGTCGTTTTATACCCAGTATTTCAGGAACTCTAATTCGGTAGTCGTGAATCTTTGCTCACCAACTACAGCAGCAGTAGCAACAAAGAAGGCCGCAATTGATTTGTATATACGAAACAATACAATACTACTACAGAAATTCGTTGGACAGTACTTGCAGATGGGCAAAAAGATAAAAACATCTTTAACACAGGCACAAACCGATACAATCCAATCACTGCCCCAGTTTTGTAATTCGAATGTCCTCAGTGGTGAGCCCTTGGTACAGTACCAGGCATTCAACGATCTGTTGGCACTCTTTAAGTCATTTAGTCATTTTGGAAATATCTTGGTTATATCATCACATTCCTATGATTGCGCACTTCTCAAATTTCTTATTTCCAGGGTGATGACCTACTATCCACTAGTGACCATCCAGGATTCTTTGCAATATATGAAAGCAACCCTGAACATATCCATCAGTACATCCGATGAGTTCGATATTCTGAATGATAAAGAACTGTGGGAGTTTGGCCAAACCCAGGAAATTCTAAAACGTAGGCAGACGAGCTCAGTCAAGAGGAGATGTGTCAATTTACCAGAAAACTCTACGATCGATAACAGAATGCTTATGGGTACCACAAAGCGAGGTCGCTTTTGA"
    )


def test_gene_YDL061C(genome):
    # Has five_prime_UTR_intron
    gene = genome["YDL061C"]
    assert gene.id == "YDL061C"
    assert gene.chromosome == 4
    assert gene.strand == "-"
    assert gene.start == 340628
    assert gene.end == 340798
    assert (
        gene.seq
        == "ATGGCTCACGAAAACGTTTGGTTCTCCCACCCAAGAAGATTCGGTAAAGGTTCCCGTCAATGTCGTGTCTGCTCCTCCCACACTGGTTTGGTCAGAAAGTACGACTTAAACATCTGTCGTCAATGTTTCAGAGAAAAGGCTAACGACATTGGTTTCCACAAGTACAGATAA"
    )


def test_gene_YIL111W(genome):
    # Internal intron, 2 CDS, 1bp CDS 5'
    gene = genome["YIL111W"]
    assert gene.id == "YIL111W"
    assert gene.chromosome == 9
    assert gene.strand == "+"
    assert gene.start == 155222
    assert gene.end == 155765
    assert (
        gene.seq
        == "AGCATGTATAACAAACACTGATTTTTGTTTTGAGTTTTAAAAGATATCCATTTACTAACATTCGAGGTGTACAAGCACAAGTTTTGCAGTGTTGCGTACTTCTCTTACTAAAGGGGCACGGCTAACTGGGACAAGATTTGTTCAAACAAAGGCCCTTTCGAAGGCAACATTGACAGATCTGCCCGAAAGATGGGAAAATATGCCAAACTTAGAACAGAAAGAGATTGCAGATAATTTGACAGAACGTCAAAAGCTTCCATGGAAAACTCTCAATAACGAGGAAATCAAAGCAGCTTGGTACATATCCTACGGCGAGTGGGGACCTAGAAGACCTGTACACGGAAAAGGCGATGTTGCATTTATAACTAAAGGAGTATTTTTAGGGTTAGGAATCTCATTTGGGCTCTTTGGTTTAGTGAGACTATTAGCCAATCCTGAAACTCCAAAGACTATGAACAGGGAATGGCAGTTGAAATCAGACGAGTATCTGAAGTCAAAAAATGCCAATCCTTGGGGAGGTTATTCTCAAGTTCAATCTAAATAA"
    )


def test_YAL037W_window_five_prime(genome):
    # + strand
    # No start codon
    dna_result = genome["YAL037W"].window_five_prime(9, include_start_codon=False)
    assert dna_result.id == "YAL037W"
    assert dna_result.chromosome == 1
    assert dna_result.strand == "+"
    assert dna_result.start_window == 74011
    assert dna_result.end_window == 74020
    assert dna_result.seq == "ACACTGCTA"

    # With start codon
    dna_result = genome["YAL037W"].window_five_prime(9, include_start_codon=True)
    assert dna_result.id == "YAL037W"
    assert dna_result.chromosome == 1
    assert dna_result.strand == "+"
    assert dna_result.start_window == 74013
    assert dna_result.end_window == 74022
    assert dna_result.seq == "ACTGCTATG"


def test_YAL037W_window_three_prime(genome):
    # + strand
    # No stop codon
    dna_result = genome["YAL037W"].window_three_prime(9, include_stop_codon=False)
    assert dna_result.id == "YAL037W"
    assert dna_result.chromosome == 1
    assert dna_result.strand == "+"
    assert dna_result.start_window == 74823
    assert dna_result.end_window == 74832
    assert dna_result.seq == "AGAGCCCTC"

    # With stop codon
    dna_result = genome["YAL037W"].window_three_prime(9, include_stop_codon=True)
    assert dna_result.id == "YAL037W"
    assert dna_result.chromosome == 1
    assert dna_result.strand == "+"
    assert dna_result.start_window == 74820
    assert dna_result.end_window == 74829
    assert dna_result.seq == "TGAAGAGCC"


def test_YDL061C_window_five_prime(genome):
    # - strand
    # No start codon
    window_result = genome["YDL061C"].window_five_prime(10, include_start_codon=False)
    assert window_result.id == "YDL061C"
    assert window_result.chromosome == 4
    assert window_result.strand == "-"
    assert window_result.start_window == 340798
    assert window_result.end_window == 340808
    assert window_result.seq == "TATATACAAA"

    # With start codon
    window_result = genome["YDL061C"].window_five_prime(10, include_start_codon=True)
    assert window_result.id == "YDL061C"
    assert window_result.chromosome == 4
    assert window_result.strand == "-"
    assert window_result.start_window == 340795
    assert window_result.end_window == 340805
    assert window_result.seq == "ATACAAAATG"


def test_YDL061C_window_three_prime(genome):
    # - strand
    # No stop codon
    window_result = genome["YDL061C"].window_three_prime(10, include_stop_codon=False)
    assert window_result.id == "YDL061C"
    assert window_result.chromosome == 4
    assert window_result.strand == "-"
    assert window_result.start_window == 340617
    assert window_result.end_window == 340627
    assert window_result.seq == "GTCAAGAGCG"

    # With stop codon
    window_result = genome["YDL061C"].window_three_prime(10, include_stop_codon=True)
    assert window_result.id == "YDL061C"
    assert window_result.chromosome == 4
    assert window_result.strand == "-"
    assert window_result.start_window == 340620
    assert window_result.end_window == 340630
    assert window_result.seq == "TAAGTCAAGA"


def test_YAL037W_window(genome):
    # + strand
    window_result = genome["YAL037W"].window(len(genome["YAL037W"]) + 10)
    assert window_result.id == "YAL037W"
    assert window_result.chromosome == 1
    assert window_result.strand == "+"
    assert window_result.start_window == 74014
    assert window_result.end_window == 74828
    assert (
        window_result.seq
        == "CTGCTATGGATATGGAAATCGAAGATTCAAGCCCCATAGATGACCTGAAGTTACAAAAACTGGATACCAATGTTTATTTTGGACCCTGTGAGATATTGACACAACCTATTCTTTTGCAATATGAAAATATTAAGTTCATCATTGGTGTCAATCTAAGTACTGAAAAGATAGCGTCGTTTTATACCCAGTATTTCAGGAACTCTAATTCGGTAGTCGTGAATCTTTGCTCACCAACTACAGCAGCAGTAGCAACAAAGAAGGCCGCAATTGATTTGTATATACGAAACAATACAATACTACTACAGAAATTCGTTGGACAGTACTTGCAGATGGGCAAAAAGATAAAAACATCTTTAACACAGGCACAAACCGATACAATCCAATCACTGCCCCAGTTTTGTAATTCGAATGTCCTCAGTGGTGAGCCCTTGGTACAGTACCAGGCATTCAACGATCTGTTGGCACTCTTTAAGTCATTTAGTCATTTTGGAAATATCTTGGTTATATCATCACATTCCTATGATTGCGCACTTCTCAAATTTCTTATTTCCAGGGTGATGACCTACTATCCACTAGTGACCATCCAGGATTCTTTGCAATATATGAAAGCAACCCTGAACATATCCATCAGTACATCCGATGAGTTCGATATTCTGAATGATAAAGAACTGTGGGAGTTTGGCCAAACCCAGGAAATTCTAAAACGTAGGCAGACGAGCTCAGTCAAGAGGAGATGTGTCAATTTACCAGAAAACTCTACGATCGATAACAGAATGCTTATGGGTACCACAAAGCGAGGTCGCTTTTGAAGAGC"
    )


def test_YDL061C_window(genome):
    # - strand
    window_result = genome["YDL061C"].window(len(genome["YDL061C"]) + 10)
    assert window_result.id == "YDL061C"
    assert window_result.chromosome == 4
    assert window_result.strand == "-"
    assert window_result.start_window == 340622
    assert window_result.end_window == 340803
    assert (
        window_result.seq
        == "ACAAAATGGCTCACGAAAACGTTTGGTTCTCCCACCCAAGAAGATTCGGTAAAGGTTCCCGTCAATGTCGTGTCTGCTCCTCCCACACTGGTTTGGTCAGAAAGTACGACTTAAACATCTGTCGTCAATGTTTCAGAGAAAAGGCTAACGACATTGGTTTCCACAAGTACAGATAAGTCAA"
    )


if __name__ == "__main__":
    print()
    # genome['YAL037W'].seq


# --- Gene-name resolver (resolve_gene_name) -------------------------------------------- #
from torchcell.sequence.genome.scerevisiae.s288c import GeneNameStatus  # noqa: E402


def test_resolve_current_gene(genome):
    r = genome.resolve_gene_name("YAL002W")
    assert r.status is GeneNameStatus.CURRENT
    assert r.systematic_name == "YAL002W"
    assert r.is_current_gene


def test_resolve_common_name_standard_owner(genome):
    # AAP1 is an alias of both YHR047C (its standard-name owner) and Q0080 (secondary
    # alias); the standard-name layer must disambiguate to YHR047C, not flag AMBIGUOUS.
    r = genome.resolve_gene_name("AAP1")
    assert r.status is GeneNameStatus.RENAMED
    assert r.systematic_name == "YHR047C"


def test_resolve_common_name_not_shadowed_by_region(genome):
    # A non-locus "region" feature is literally id'd ADE1; it must NOT intercept the gene.
    r = genome.resolve_gene_name("ADE1")
    assert r.status is GeneNameStatus.RENAMED
    assert r.systematic_name == "YAR015W"


def test_resolve_non_gene_feature_pseudogene(genome):
    # FLO8 is a blocked_reading_frame pseudogene (YER109C) -- a valid, retained locus.
    r = genome.resolve_gene_name("FLO8")
    assert r.status is GeneNameStatus.NON_GENE_FEATURE
    assert r.systematic_name == "YER109C"
    assert r.feature_type == "blocked_reading_frame"


def test_resolve_legacy_systematic_rename(genome):
    # 2005 ORF YGR272C is an SGD alias of the current gene YGR271C-A.
    r = genome.resolve_gene_name("YGR272C")
    assert r.status is GeneNameStatus.RENAMED
    assert r.systematic_name == "YGR271C-A"


def test_resolve_retired(genome):
    r = genome.resolve_gene_name("YAR037W")
    assert r.status is GeneNameStatus.RETIRED
    assert r.systematic_name == "YAR037W"  # retained verbatim as a legacy name


def test_resolve_ambiguous_common_name(genome):
    # FEN1 names two distinct genes (YCR034W, YKL113C); never silently pick one.
    r = genome.resolve_gene_name("FEN1")
    assert r.status is GeneNameStatus.AMBIGUOUS
    assert r.systematic_name is None
    assert set(r.candidates) == {"YCR034W", "YKL113C"}


def test_resolve_is_case_insensitive(genome):
    assert genome.resolve_gene_name("yal002w").systematic_name == "YAL002W"
