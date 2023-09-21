import logging
from unittest.mock import patch

import pytest

from torchcell.models.constants import DNA_LLM_MAX_TOKEN_SIZE
from torchcell.sequence.data import (
    DnaSelectionResult,
    DnaWindowResult,
    GeneSet,
    calculate_window_bounds,
    calculate_window_bounds_symmetric,
    calculate_window_undersized,
    calculate_window_undersized_symmetric,
    get_chr_from_description,
    mismatch_positions,
    roman_to_int,
)

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


def test_invalid_seq_len():
    with pytest.raises(ValueError):
        seq = "A" * (DNA_LLM_MAX_TOKEN_SIZE + 1)
        DnaSelectionResult(
            id="gene_name", seq=seq, chromosome=1, start=0, end=len(seq), strand="+"
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


def test_calculate_window_undersized_positive_strand():
    start = 10
    end = 50
    strand = "+"
    window_size = 20

    # For a positive strand, the start window should be equal to start, and the end window should be start + window_size
    expected_start_window = start
    expected_end_window = start + window_size

    actual_start_window, actual_end_window = calculate_window_undersized(
        start, end, strand, window_size
    )

    assert (
        actual_start_window == expected_start_window
    ), f"Expected {expected_start_window}, but got {actual_start_window}"
    assert (
        actual_end_window == expected_end_window
    ), f"Expected {expected_end_window}, but got {actual_end_window}"


def test_calculate_window_undersized_negative_strand():
    start = 10
    end = 50
    strand = "-"
    window_size = 20

    # For a negative strand, the start window should be end - window_size, and the end window should be equal to end
    expected_start_window = end - window_size
    expected_end_window = end

    actual_start_window, actual_end_window = calculate_window_undersized(
        start, end, strand, window_size
    )

    assert (
        actual_start_window == expected_start_window
    ), f"Expected {expected_start_window}, but got {actual_start_window}"
    assert (
        actual_end_window == expected_end_window
    ), f"Expected {expected_end_window}, but got {actual_end_window}"


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

    assert actual_start_window, actual_end_window == (13, 17)


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
        GeneSet([1, 2, 3])


def test_repr_method(sample_geneset):
    expected_repr = "GeneSet(size=4, items=['YAL001C', 'YAL002W', 'YAL003W']...)"
    assert repr(sample_geneset) == expected_repr


def test_empty_initialization():
    genes = GeneSet()
    assert len(genes) == 0
    assert repr(genes) == "GeneSet(size=0, items=[]...)"


# Additional test to check the scenario with more items in the GeneSet
def test_repr_method_with_more_items():
    genes = GeneSet(["YAL001C", "YAL002W", "YAL003W", "YAL004W", "YAL005C"])
    expected_repr = "GeneSet(size=5, items=['YAL001C', 'YAL002W', 'YAL003W']...)"
    assert repr(genes) == expected_repr
