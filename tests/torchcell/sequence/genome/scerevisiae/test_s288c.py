import os
import os.path as osp
import random

import gffutils
import pytest
from dotenv import load_dotenv

from torchcell.sequence import DnaSelectionResult
from torchcell.sequence.genome.scerevisiae.S288C import SCerevisiaeGenome

load_dotenv()

DATA_ROOT = os.getenv("DATA_ROOT")


@pytest.fixture
def genome():
    genome = SCerevisiaeGenome(data_root=osp.join(DATA_ROOT, "data/sgd/genome"))
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
