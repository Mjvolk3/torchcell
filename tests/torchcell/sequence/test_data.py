# tests/torchcell/sequence/test_data.py
# [[tests.torchcell.sequence.test_data]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sequence/test_data.py
"""Tests for torchcell.sequence.data.

2026.09.30 (Phase 16). Every fixture is a hand-built coordinate tuple, a short CDS string
or a two-method ``Genome`` subclass; nothing reads a genome. Expected values, derived from
the source:

- ``calculate_window_bounds(40, 61, strand, 30, 100)``: the gene is 21 bp, the flank is
  ``(30 - 21) // 2 = 4``, so the raw window is ``(36, 65)``, 29 bp. The one-base
  correction adds upstream: ``start - 1`` on ``+`` gives ``(35, 65)``, ``end + 1`` on
  ``-`` gives ``(36, 66)``. A strand that is neither is refused by name with
  ``ValueError`` before any arithmetic (issue #538).
- ``calculate_window_undersized`` with a strand other than ``+``/``-`` raises
  ``ValueError`` naming the strand (issue #538).
- ``compute_codon_frequency("ATGGCGGCGCTGAAA")``: five codons ATG, GCG, GCG, CTG, AAA, so
  GCG = 2/5 = 0.4, ATG = CTG = AAA = 1/5 = 0.2, the other 60 codons 0.0. The repr sorts by
  frequency with a stable sort over the SortedDict's alphabetical key order, so the
  0.2 tie resolves to AAA, ATG (CTG drops out of the top three).
- ``GeneSet.__repr__`` lists every member when the size is at most 3 and the first
  three plus ``...`` above that (issue #538 fixed the size-3 gap).
- ``get_chr_from_description`` raises ``ValueError`` quoting a description with neither
  a chromosome nor a mitochondrion location (issue #538).
"""

import logging
import re
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from pydantic import ValidationError
from sortedcontainers import SortedDict, SortedSet

from torchcell.sequence.data import (
    CodonFrequency,
    DnaSelectionResult,
    DnaWindowResult,
    Gene,
    GeneSet,
    Genome,
    ParsedGenome,
    calculate_window_bounds,
    calculate_window_bounds_symmetric,
    calculate_window_undersized,
    calculate_window_undersized_symmetric,
    compute_codon_frequency,
    get_chr_from_description,
    mismatch_positions,
    roman_to_int,
)
from torchcell.sequence.db_connection import DatabaseConnectionManager

log = logging.getLogger()


# Test DnaSelectionResult
def test_valid_dna_selection_result():
    result = DnaSelectionResult(
        id="gene_name", seq="ATGC", chromosome=1, start=0, end=4, strand="+"
    )
    assert len(result) == 4


def test_invalid_chromosome():
    with pytest.raises(ValueError):
        DnaSelectionResult(
            id="gene_name", seq="ATGC", chromosome=-1, start=0, end=4, strand="+"
        )


def test_invalid_start():
    with pytest.raises(ValueError):
        DnaSelectionResult(
            id="gene_name", seq="ATGC", chromosome=1, start=-1, end=4, strand="+"
        )


def test_invalid_start_end():
    with pytest.raises(ValueError):
        DnaSelectionResult(
            id="gene_name", seq="ATGC", chromosome=1, start=4, end=0, strand="+"
        )


def test_invalid_strand():
    with pytest.raises(ValueError):
        DnaSelectionResult(
            id="gene_name", seq="ATGC", chromosome=1, start=0, end=4, strand="x"
        )


@pytest.fixture
def seq1():
    return DnaSelectionResult(
        id="1", chromosome=1, strand="+", start=1, end=5, seq="ATCG"
    )


@pytest.fixture
def seq2():
    return DnaSelectionResult(
        id="2", chromosome=1, strand="+", start=1, end=4, seq="ATC"
    )


def test_ge(seq1, seq2):
    assert seq1 >= seq2

    with pytest.raises(TypeError):
        assert seq1 >= "ATCG"


def test_le(seq1, seq2):
    assert seq2 <= seq1

    with pytest.raises(TypeError):
        assert seq2 <= "ATCG"


# Test DnaWindowResult
def test_repr():
    dna_window = DnaWindowResult(
        id="1",
        chromosome=1,
        strand="+",
        start=1,
        end=5,
        start_window=1,
        end_window=5,
        seq="ATCG",
    )
    expected_repr = "DnaWindowResult(id='1', chromosome=1, strand='+', start_window=1, end_window=5, seq='ATCG')"
    assert repr(dna_window) == expected_repr


# Test the start_window and end_window validators
def test_window_validators():
    # This should raise ValueError because start_window is negative
    with pytest.raises(ValueError):
        DnaWindowResult(
            id="1",
            chromosome=1,
            strand="+",
            start=1,
            end=5,
            start_window=-1,
            end_window=5,
            seq="ATCG",
        )

    # This should raise ValueError because end_window is negative
    with pytest.raises(ValueError):
        DnaWindowResult(
            id="1",
            chromosome=1,
            strand="+",
            start=1,
            end=5,
            start_window=1,
            end_window=-1,
            seq="ATCG",
        )

    # This should not raise
    try:
        DnaWindowResult(
            id="1",
            chromosome=1,
            strand="+",
            start=1,
            end=5,
            start_window=0,
            end_window=0,
            seq="ATCG",
        )
    except ValueError:
        pytest.fail("Unexpected ValueError")


# Test utility functions
def test_get_chr_from_desccription():
    # Test with chromosome I
    description1 = "ref|NC_001133| [org=Saccharomyces cerevisiae] [strain=S288C] [moltype=genomic] [chromosome=I]"
    assert get_chr_from_description(description1) == 1

    # Test with chromosome II
    description2 = "ref|NC_001134| [org=Saccharomyces cerevisiae] [strain=S288C] [moltype=genomic] [chromosome=II]"
    assert get_chr_from_description(description2) == 2

    # Test with mitochondrion
    description3 = "ref|NC_001224| [org=Saccharomyces cerevisiae] [strain=S288C] [moltype=genomic] [location=mitochondrion] [top=circular]"
    assert get_chr_from_description(description3) == 0


def test_mismatch_positions() -> None:
    seq1 = "ATGC"
    seq2 = "ATCC"
    assert mismatch_positions(seq1, seq2) == [2]


def test_mismatch_positions_different_lengths():
    seq1 = "ATC"
    seq2 = "ATCG"
    with pytest.raises(ValueError):
        mismatch_positions(seq1, seq2)


def test_roman_to_int() -> None:
    assert roman_to_int("IV") == 4
    assert roman_to_int("XII") == 12
    assert roman_to_int("X") == 10
    assert roman_to_int("LXIII") == 63


# Test window functions


def test_calculate_window_undersized_strand():
    # For a positive strand, the start window should be equal to start,
    # and the end window should be start + window_size
    actual_start_window, actual_end_window = calculate_window_undersized(
        start=10, end=50, strand="+", window_size=20
    )

    assert (actual_start_window, actual_end_window) == (10, 30)

    # For a negative strand, the start window should be end - window_size,
    # and the end window should be equal to end
    actual_start_window, actual_end_window = calculate_window_undersized(
        start=10, end=50, strand="-", window_size=20
    )

    assert (actual_start_window, actual_end_window) == (30, 50)


def test_calculate_window_bounds_errors() -> None:
    # "+" strand
    with pytest.raises(ValueError, match="End position is out of bounds of chromosome"):
        calculate_window_bounds(
            start=10, end=110, strand="+", window_size=30, chromosome_length=100
        )

    with pytest.raises(
        ValueError, match="Start position must be less than end position"
    ):
        calculate_window_bounds(
            start=30, end=30, strand="+", window_size=30, chromosome_length=100
        )

    with pytest.raises(
        ValueError, match="Window size should never be greater than chromosome length"
    ):
        calculate_window_bounds(
            start=10, end=20, strand="+", window_size=200, chromosome_length=100
        )
    # "-" strand
    with pytest.raises(ValueError, match="End position is out of bounds of chromosome"):
        calculate_window_bounds(
            start=10, end=110, strand="-", window_size=30, chromosome_length=100
        )

    with pytest.raises(
        ValueError, match="Start position must be less than end position"
    ):
        calculate_window_bounds(
            start=30, end=30, strand="-", window_size=30, chromosome_length=100
        )

    with pytest.raises(
        ValueError, match="Window size should never be greater than chromosome length"
    ):
        calculate_window_bounds(
            start=10, end=20, strand="-", window_size=200, chromosome_length=100
        )


def test_calculate_window_bounds() -> None:
    # "+" strand
    assert calculate_window_bounds(
        start=0, end=20, strand="+", window_size=40, chromosome_length=100
    ) == (0, 40)
    # Window start would be negative, so it gets adjusted
    assert calculate_window_bounds(
        start=5, end=25, strand="+", window_size=50, chromosome_length=100
    ) == (0, 50)
    # Window end would exceed chromosome_length, so it gets adjusted
    assert calculate_window_bounds(
        start=75, end=95, strand="+", window_size=50, chromosome_length=100
    ) == (50, 100)
    # Window size is same as chromosome_length
    assert calculate_window_bounds(
        start=0, end=20, strand="+", window_size=100, chromosome_length=100
    ) == (0, 100)
    # Window size with odd selection, gets max sized window, +1bp 5utr
    assert calculate_window_bounds(
        start=40, end=61, strand="+", window_size=30, chromosome_length=100
    ) == (35, 65)
    # "-" strand
    assert calculate_window_bounds(
        start=0, end=20, strand="-", window_size=40, chromosome_length=100
    ) == (0, 40)
    # Window start would be negative, so it gets adjusted
    assert calculate_window_bounds(
        start=5, end=25, strand="-", window_size=50, chromosome_length=100
    ) == (0, 50)
    # Window end would exceed chromosome_length, so it gets adjusted
    assert calculate_window_bounds(
        start=75, end=95, strand="-", window_size=50, chromosome_length=100
    ) == (50, 100)
    # Window size is same as chromosome_length
    assert calculate_window_bounds(
        start=0, end=20, strand="-", window_size=100, chromosome_length=100
    ) == (0, 100)
    # Window size with odd selection, gets max sized window, +1bp upstream
    assert calculate_window_bounds(
        start=40, end=61, strand="-", window_size=30, chromosome_length=100
    ) == (36, 66)
    # Expected: The function should call calculate_window_undersized
    # and the window_size should be smaller than the sequence length
    calculate_window_bounds(
        start=10, end=20, strand="+", window_size=5, chromosome_length=100
    )
    # Test where end_window is at chromosome_length
    # Expected: The function should adjust the start_window -= 1 to meet the window_size
    assert calculate_window_bounds(
        start=48, end=50, strand="+", window_size=3, chromosome_length=50
    ) == (47, 50)

    # Test where start_window is at 0 and the difference between end_window
    # and start_window is 1
    # Expected: The function should adjust the end_window += 1
    assert calculate_window_bounds(
        start=0, end=1, strand="+", window_size=2, chromosome_length=50
    ) == (0, 2)


def test_calculate_window_undersized_symmetric():
    # Test with an even window size,
    # where the middle is exactly in between start and end.
    assert calculate_window_undersized_symmetric(start=10, end=20, window_size=4) == (
        13,
        17,
    )

    # Test with an odd window size,
    # where the middle is exactly in between start and end.
    # Since the window size is 5, and start_window is calculated
    # as middle - flank_size, the result will be (12, 17).
    assert calculate_window_undersized_symmetric(start=10, end=20, window_size=5) == (
        13,
        17,
    )

    # Test error raising for equal start and end
    with pytest.raises(ValueError, match="Start and end positions are the same"):
        calculate_window_undersized_symmetric(start=15, end=15, window_size=4)

    # Test error raising for window size less than 2.
    with pytest.raises(ValueError, match="Window size must be at least 2"):
        calculate_window_undersized_symmetric(start=10, end=20, window_size=1)


def test_calculate_window_bounds_symmetric():
    # Existing tests
    assert calculate_window_bounds_symmetric(
        start=5, end=15, window_size=30, chromosome_length=100
    ) == (0, 20)
    assert calculate_window_bounds_symmetric(
        start=80, end=95, window_size=30, chromosome_length=100
    ) == (75, 100)
    assert calculate_window_bounds_symmetric(
        start=45, end=55, window_size=30, chromosome_length=100
    ) == (35, 65)

    # Test for end being greater than chromosome_length
    with pytest.raises(ValueError, match="End position is out of bounds of chromosome"):
        calculate_window_bounds_symmetric(
            start=10, end=105, window_size=10, chromosome_length=100
        )

    # Test for start being greater than or equal to end
    with pytest.raises(
        ValueError, match="Start position must be less than end position"
    ):
        calculate_window_bounds_symmetric(
            start=50, end=50, window_size=10, chromosome_length=100
        )

    # Test for window_size being greater than chromosome_length
    with pytest.raises(
        ValueError, match="Window size should never be greater than chromosome length"
    ):
        calculate_window_bounds_symmetric(
            start=10, end=20, window_size=101, chromosome_length=100
        )
    # This test should trigger the condition where window_size < end - start
    actual_start_window, actual_end_window = calculate_window_bounds_symmetric(
        start=10, end=20, window_size=5, chromosome_length=100
    )

    assert (actual_start_window, actual_end_window) == (13, 17)


@pytest.fixture
def sample_geneset():
    return GeneSet(["YAL001C", "YAL002W", "YAL003W", "YAL004W"])


def test_initialization_with_strings(sample_geneset):
    assert len(sample_geneset) == 4
    assert "YAL001C" in sample_geneset
    assert "YAL002W" in sample_geneset
    assert "YAL003W" in sample_geneset
    assert "YAL004W" in sample_geneset


def test_initialization_with_non_string():
    with pytest.raises(ValueError, match="All items in gene_set must be str"):
        # Deliberately pass non-str items to exercise the validation; cast keeps
        # the wrong runtime values while satisfying the Iterable[str] parameter.
        GeneSet(cast(list[str], [1, 2, 3]))


def test_repr_method(sample_geneset):
    expected_repr = "GeneSet(size=4, items=['YAL001C', 'YAL002W', 'YAL003W']...)"
    assert repr(sample_geneset) == expected_repr


def test_empty_initialization():
    genes = GeneSet()
    assert len(genes) == 0
    assert repr(genes) == "GeneSet(size=0, items=[])"


# Additional test to check the scenario with more items in the GeneSet
def test_repr_method_with_more_items():
    genes = GeneSet(["YAL001C", "YAL002W", "YAL003W", "YAL004W", "YAL005C"])
    expected_repr = "GeneSet(size=5, items=['YAL001C', 'YAL002W', 'YAL003W']...)"
    assert repr(genes) == expected_repr


def test_compute_codon_frequency():
    cds_str = "ATGGCGGCGCTGAAA"
    codon_frequency = compute_codon_frequency(cds_str)

    # Asserting that the returned object is a SortedDict
    assert isinstance(codon_frequency, SortedDict), (
        "The returned object is not a SortedDict"
    )

    # Asserting that the sum of the frequencies is 1.0
    assert sum(codon_frequency.values()) == pytest.approx(1.0), (
        "The sum of the frequencies is not 1.0"
    )


def test_compute_codon_frequency_exact_values() -> None:
    """ATG GCG GCG CTG AAA: GCG 2/5, the three singletons 1/5, the other 60 codons 0."""
    freq = compute_codon_frequency("ATGGCGGCGCTGAAA")
    assert len(freq) == 64
    nonzero = {codon: value for codon, value in freq.items() if value != 0.0}
    assert nonzero == {"AAA": 0.2, "ATG": 0.2, "CTG": 0.2, "GCG": 0.4}
    assert list(freq.keys())[:3] == ["AAA", "AAC", "AAG"]


def test_compute_codon_frequency_refuses_bad_cds() -> None:
    """A length not divisible by 3 and a non-ACGT base are both refused, same message."""
    message = "Invalid CDS string; length must be a multiple of 3 and only contain A, T, G, C."
    with pytest.raises(ValueError) as short:
        compute_codon_frequency("ATGC")
    assert str(short.value) == message
    with pytest.raises(ValueError) as ambiguous:
        compute_codon_frequency("ATGNNN")
    assert str(ambiguous.value) == message


def test_compute_codon_frequency_empty_cds_is_refused() -> None:
    """An empty CDS is refused at validation by name, never divided by zero codons.

    ``0 % 3 == 0`` and the empty set is a subset, so the length/alphabet check alone let
    it through to ``count / 0`` (issue #538). The ValueError keeps the codon-frequency
    dataset's ``except ValueError: continue`` skip working for empty CDSs.
    """
    with pytest.raises(ValueError) as empty:
        compute_codon_frequency("")
    assert str(empty.value) == (
        "Empty CDS string; a codon frequency needs at least one codon."
    )


def test_codon_frequency_repr_top_three_and_tie_order() -> None:
    """GCG 0.4 first, then the 0.2 tie in alphabetical order: AAA, ATG (CTG is cut)."""
    freq = compute_codon_frequency("ATGGCGGCGCTGAAA")
    assert repr(freq) == (
        "CodonFrequency(size=64, most_frequent_codons="
        "[('GCG', 0.4), ('AAA', 0.2), ('ATG', 0.2)]...)"
    )


def test_codon_frequency_repr_rounds_to_four_decimals() -> None:
    """Three distinct codons: each 1/3 renders as 0.3333, ordered alphabetically."""
    freq = compute_codon_frequency("TTTGGGCCC")
    assert repr(freq) == (
        "CodonFrequency(size=64, most_frequent_codons="
        "[('CCC', 0.3333), ('GGG', 0.3333), ('TTT', 0.3333)]...)"
    )


def test_codon_frequency_repr_flags_wrong_size_and_bad_sum() -> None:
    """One codon is not 64; 64 codons summing to 0.5 fall outside [0.9999, 1.0001]."""
    assert repr(CodonFrequency({"AAA": 1.0})) == (
        "Invalid CodonFrequency: Expected 64 codons"
    )
    half = {codon: 0.0 for codon in compute_codon_frequency("AAA")}
    half["AAA"] = 0.5
    assert repr(CodonFrequency(half)) == (
        "Invalid CodonFrequency: Frequencies do not sum to 1 (sum=0.5)"
    )


def test_geneset_repr_size_three_and_four() -> None:
    """Size 3 lists all three members with no ellipsis; size 4 shows three plus ``...``.

    Size 3 used to fall between the ``> 3`` and ``< 3`` branches and return None
    (issue #538).
    """
    assert repr(GeneSet(["YAL003W", "YAL001C", "YAL002W"])) == (
        "GeneSet(size=3, items=['YAL001C', 'YAL002W', 'YAL003W'])"
    )
    assert repr(GeneSet(["YAL004W", "YAL003W", "YAL001C", "YAL002W"])) == (
        "GeneSet(size=4, items=['YAL001C', 'YAL002W', 'YAL003W']...)"
    )


def test_geneset_sorts_and_deduplicates() -> None:
    """Members come back sorted and a repeated id counts once."""
    genes = GeneSet(["YBR002W", "YAL001C", "YBR002W"])
    assert list(genes) == ["YAL001C", "YBR002W"]
    assert repr(genes) == "GeneSet(size=2, items=['YAL001C', 'YBR002W'])"


def test_get_chr_from_description_non_mito_location_and_no_tag() -> None:
    """A description with no chromosome tag and no mitochondrion location is refused.

    The ``ValueError`` quotes the description, honoring the ``-> int`` annotation instead
    of returning None (issue #538). A chromosome tag after a non-mitochondrial location
    still resolves.
    """
    with pytest.raises(ValueError) as plastid:
        get_chr_from_description("ref|X| [location=plastid] [top=circular]")
    assert str(plastid.value) == (
        "Description has no [chromosome=...] or [location=mitochondrion] tag: "
        "'ref|X| [location=plastid] [top=circular]'"
    )
    with pytest.raises(ValueError) as tagless:
        get_chr_from_description("no tags here")
    assert str(tagless.value) == (
        "Description has no [chromosome=...] or [location=mitochondrion] tag: "
        "'no tags here'"
    )
    assert get_chr_from_description("[location=plastid] [chromosome=XVI]") == 16
    assert get_chr_from_description("[location=mitochondrion] [top=circular]") == 0


def test_roman_to_int_subtractive_pairs_and_refusal() -> None:
    """XIV: 10, +1, then V > I so +5 - 2*1, total 14; MCMXC = 1990; a non-numeral is a KeyError."""
    assert roman_to_int("XIV") == 14
    assert roman_to_int("MCMXC") == 1990
    assert roman_to_int("XVI") == 16
    with pytest.raises(KeyError, match="Z"):
        roman_to_int("XZ")


def test_calculate_window_undersized_unknown_strand_refused() -> None:
    """A strand other than ``+``/``-`` is refused with a ValueError naming it.

    It used to bind no window variable and raise UnboundLocalError (issue #538).
    """
    with pytest.raises(ValueError) as unknown:
        calculate_window_undersized(10, 50, ".", 20)
    assert str(unknown.value) == "Strand must be '+' or '-', got '.'"


def test_calculate_window_bounds_undersized_delegates_by_strand() -> None:
    """A 5 bp window on a 10 bp gene keeps the 5' end: (10, 15) on +, (15, 20) on -."""
    assert calculate_window_bounds(10, 20, "+", 5, 100) == (10, 15)
    assert calculate_window_bounds(10, 20, "-", 5, 100) == (15, 20)


def test_calculate_window_bounds_odd_flank_is_exact_and_unknown_strand_refused() -> (
    None
):
    """Every returned window is exactly the requested length; an unknown strand is refused.

    21 bp gene (40, 61), 30 bp window: flank (30 - 21) // 2 = 4 gives (36, 65), 29 bp.
    The odd base goes upstream: ``+`` moves the start to 36 - 1 = 35, giving (35, 65),
    65 - 35 = 30; ``-`` moves the end to 65 + 1 = 66, giving (36, 66), 66 - 36 = 30.
    A ``.`` strand has no upstream, so it is refused by name before the arithmetic
    instead of returning the 29 bp (36, 65) (issue #538).
    """
    assert calculate_window_bounds(40, 61, "+", 30, 100) == (35, 65)
    assert calculate_window_bounds(40, 61, "-", 30, 100) == (36, 66)
    with pytest.raises(ValueError) as unknown:
        calculate_window_bounds(40, 61, ".", 30, 100)
    assert str(unknown.value) == "Strand must be '+' or '-', got '.'"


def test_calculate_window_bounds_symmetric_clips_to_shorter_flank() -> None:
    """Gene (3, 13) in a 100 bp chromosome with a 30 bp window: flank 10 is capped by 0.

    The 5' side has only 3 bp, so both flanks shrink to 3: (0, 16), 16 bp, symmetric.
    At the far end, gene (90, 98) with flank 11 is capped by the 2 bp to 100: (88, 100).
    """
    assert calculate_window_bounds_symmetric(3, 13, 30, 100) == (0, 16)
    assert calculate_window_bounds_symmetric(90, 98, 30, 100) == (88, 100)


def test_dna_selection_missing_start_is_named_validation_error() -> None:
    """A payload without ``start`` is a pydantic missing-field error at ``start``.

    The start/end check now runs after field validation (issue #538), so it never
    compares ``None > end`` and raises a bare TypeError.
    """
    with pytest.raises(ValidationError) as missing:
        DnaSelectionResult.model_validate(
            {"id": "x", "chromosome": 1, "strand": "+", "end": 3, "seq": "A"}
        )
    errors = missing.value.errors()
    assert [(e["loc"], e["type"]) for e in errors] == [(("start",), "missing")]


def test_dna_selection_refusal_messages() -> None:
    """Each validator names its failure: start > end, a bad strand, a negative coordinate."""
    with pytest.raises(ValidationError, match="Start must be less than end"):
        DnaSelectionResult(id="x", chromosome=1, strand="+", start=5, end=4, seq="")
    with pytest.raises(ValidationError, match="Strand must be either '\\+' or '-'"):
        DnaSelectionResult(id="x", chromosome=1, strand="*", start=0, end=4, seq="")
    with pytest.raises(ValidationError, match="-2 must be positive"):
        DnaSelectionResult(id="x", chromosome=-2, strand="+", start=0, end=4, seq="")


def test_parsed_genome_accepts_geneset_and_refuses_sortedset() -> None:
    """Pydantic's instance check refuses a plain SortedSet before the field validator runs."""
    parsed = ParsedGenome(gene_set=GeneSet(["YB", "YA"]))
    assert list(parsed.gene_set) == ["YA", "YB"]
    with pytest.raises(ValidationError, match="Input should be an instance of GeneSet"):
        ParsedGenome.model_validate({"gene_set": SortedSet(["YA"])})


def test_gene_and_genome_refuse_instantiation_without_abstract_methods() -> None:
    """The ABCs name every abstract method a subclass must implement."""
    abstract_gene: Any = Gene
    abstract_genome: Any = Genome
    with pytest.raises(TypeError) as gene_err:
        abstract_gene()
    assert str(gene_err.value) == (
        "Can't instantiate abstract class Gene without an implementation for abstract "
        "methods 'window', 'window_five_prime', 'window_three_prime'"
    )
    with pytest.raises(TypeError) as genome_err:
        abstract_genome()
    assert str(genome_err.value) == (
        "Can't instantiate abstract class Genome without an implementation for abstract "
        "methods '__getitem__', 'compute_gene_set', 'feature_types', "
        "'gene_attribute_table', 'get_seq'"
    )


class _FiveBaseGene(Gene):
    """A concrete gene whose windows are unused; only ``seq`` and ``__len__`` matter."""

    def __init__(self, seq: str) -> None:
        self.seq = seq

    def window(self, window_size: int, is_max_size: bool = True) -> DnaWindowResult:
        raise NotImplementedError

    def window_five_prime(
        self, window_size: int, allow_undersize: bool = False
    ) -> DnaWindowResult:
        raise NotImplementedError

    def window_three_prime(
        self, window_size: int, allow_undersize: bool = False
    ) -> DnaWindowResult:
        raise NotImplementedError


class _CountingGenome(Genome):
    """A genome that records how often ``compute_gene_set`` runs."""

    def __init__(self, data_root: str | None = None) -> None:
        super().__init__(data_root)
        self.compute_calls = 0

    def compute_gene_set(self) -> GeneSet:
        self.compute_calls += 1
        return GeneSet(["YAL002W", "YAL001C"])

    def get_seq(
        self, chr: int | str, start: int, end: int, strand: str
    ) -> DnaSelectionResult:
        raise NotImplementedError

    @property
    def gene_attribute_table(self) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def feature_types(self) -> list[str]:
        raise NotImplementedError

    def __getitem__(self, item: str) -> Gene | None:
        return None


class _RecordingDb:
    """Stands in for ``FeatureDB``: records each constructor path."""

    opened: list[str] = []

    def __init__(self, path: str) -> None:
        _RecordingDb.opened.append(path)
        self.path = path


def test_gene_len_is_sequence_length() -> None:
    """``Gene.__len__`` is ``len(self.seq)``."""
    assert len(_FiveBaseGene("ATGCA")) == 5


def test_genome_init_state_and_lazy_gene_set() -> None:
    """The base init stores ``data_root`` and leaves every cache None; the gene set is

    computed once on first access, cached, and ``len(genome)`` is its size (2).
    """
    genome = _CountingGenome(data_root="/nowhere")
    assert genome.data_root == "/nowhere"
    assert (
        genome.fasta_sequences,
        genome.chr_to_nc,
        genome.nc_to_chr,
        genome.chr_to_len,
        genome._fasta_path,
        genome._gff_path,
    ) == (None, None, None, None, None, None)
    assert genome.db is None
    assert genome.compute_calls == 0
    assert list(genome.gene_set) == ["YAL001C", "YAL002W"]
    assert len(genome) == 2
    assert genome.compute_calls == 1


def test_genome_gene_set_setter_bypasses_compute() -> None:
    """Assigning a gene set replaces the cache; ``compute_gene_set`` never runs."""
    genome = _CountingGenome()
    genome.gene_set = GeneSet(["YBR001C"])
    assert list(genome.gene_set) == ["YBR001C"]
    assert len(genome) == 1
    assert genome.compute_calls == 0


def test_genome_db_opens_once_through_the_connection_manager(tmp_path: Path) -> None:
    """``db`` delegates to the manager, which opens the file once per thread."""
    db_file = tmp_path / "genes.db"
    db_file.write_text("")
    _RecordingDb.opened = []
    genome = _CountingGenome()
    genome._db_connection_manager = DatabaseConnectionManager(
        str(db_file), cast(Any, _RecordingDb)
    )
    first = genome.db
    second = genome.db
    assert _RecordingDb.opened == [str(db_file)]
    assert isinstance(first, _RecordingDb) and first.path == str(db_file)
    assert second == first


def test_genome_db_missing_file_refused(tmp_path: Path) -> None:
    """A manager pointed at a missing file raises the manager's exact FileNotFoundError."""
    genome = _CountingGenome()
    missing = tmp_path / "absent.db"
    genome._db_connection_manager = DatabaseConnectionManager(str(missing))
    with pytest.raises(FileNotFoundError) as err:
        _ = genome.db
    assert str(err.value) == f"Database not found at {missing}"


if __name__ == "__main__":
    cds_str = "ATGGCGGCGCTGAAA"
    codon_frequency_vector = compute_codon_frequency(cds_str)
    print(codon_frequency_vector)


# Phase 24: ParsedGenome's own gene_set check, reached only by a direct call


def test_parsed_genome_validator_names_the_rejected_type() -> None:
    """Finding: the pydantic ``is_instance_of`` check refuses a non-GeneSet before the
    field validator runs, so its message is reachable only by calling it directly.
    ``check_seq_len``'s negative-length branch is unreachable (``len`` is never < 0).
    """
    with pytest.raises(
        ValidationError, match=re.escape("Input should be an instance of GeneSet")
    ):
        ParsedGenome(gene_set=["YAL001C"])  # type: ignore[arg-type]
    validate = cast(Any, ParsedGenome).validate_gene_set
    with pytest.raises(
        ValueError, match=re.escape("gene_set must be a GeneSet, got list")
    ):
        validate(["YAL001C"])
    genes = GeneSet(["YAL001C"])
    assert validate(genes) is genes
