import logging

import pytest

from torchcell.models.constants import DNA_LLM_MAX_TOKEN_SIZE
from torchcell.sequence.data import (
    DnaSelectionResult,
    DnaWindowResult,
    calculate_window_bounds,
    calculate_window_bounds_symmetric,
    calculate_window_undersized,
    calculate_window_undersized_symmetric,
    mismatch_positions,
    roman_to_int,
)

log = logging.getLogger()


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


def test_mismatch_positions() -> None:
    seq1 = "ATGC"
    seq2 = "ATCC"
    assert mismatch_positions(seq1, seq2) == [2]


def test_roman_to_int() -> None:
    assert roman_to_int("IV") == 4
    assert roman_to_int("XII") == 12
    assert roman_to_int("X") == 10
    assert roman_to_int("LXIII") == 63


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
    # Window size with odd selection, gets max sized window, +1bp 5utr
    assert calculate_window_bounds(
        start=40, end=61, strand="-", window_size=30, chromosome_length=100
    ) == (36, 66)


def test_calculate_window_bounds_symmetric() -> None:
    # Additional tests
    # Test with symmetry and adjustment at the beginning
    assert calculate_window_bounds_symmetric(
        start=5, end=15, window_size=30, chromosome_length=100
    ) == (0, 20)
    # Test with symmetry and adjustment at the end
    assert calculate_window_bounds_symmetric(
        start=80, end=95, window_size=30, chromosome_length=100
    ) == (75, 100)
    # Test with perfect symmetry
    assert calculate_window_bounds_symmetric(
        start=45, end=55, window_size=30, chromosome_length=100
    ) == (35, 65)


if __name__ == "__main__":
    pass
