# torchcell/sequence/data.py
# [[torchcell.sequence.data]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sequence/data.py
# Test file: /torchcell/sequence/test_data.py

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import product
from turtle import st
from typing import Set

import gffutils
import matplotlib.pyplot as plt
import pandas as pd
from attrs import define, field
from Bio import SeqIO
from Bio.Seq import Seq
from gffutils import Feature, FeatureDB
from gffutils.biopython_integration import to_seqfeature
from matplotlib import pyplot as plt
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, root_validator
from sortedcontainers import SortedDict, SortedSet
from sympy import sequence

from torchcell.datamodels import ModelStrict, ModelStrictArbitrary

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class DnaSelectionResult(ModelStrict):
    id: str
    chromosome: int
    strand: str
    start: int
    end: int
    seq: str

    def __len__(self) -> int:
        return len(self.seq)

    def __ge__(self, other: str) -> bool:
        if isinstance(other, DnaSelectionResult):
            return len(self.seq) >= len(other.seq)
        return NotImplemented

    def __le__(self, other: str) -> bool:
        if isinstance(other, DnaSelectionResult):
            return len(self.seq) <= len(other.seq)
        return NotImplemented

    @root_validator(pre=True)
    def end_leq_start(cls, values):
        start, end = values.get("start"), values.get("end")
        if start > end:
            raise ValueError("Start must be less than end")
        return values

    @field_validator("strand")
    def check_strand(cls, v):
        if v not in ["+", "-"]:
            raise ValueError("Strand must be either '+' or '-'")
        return v

    @field_validator("chromosome", "start", "end")
    def check_positive(cls, v):
        if v < 0:
            raise ValueError(f"{v} must be positive")
        return v

    # TODO consider adding chromosome length
    @field_validator("seq")
    def check_seq_len(cls, v):
        sequence_length = len(v)
        if sequence_length < 0:
            raise ValueError(f"Sequence length ({sequence_length}) not geq 0")
        return v


class DnaWindowResult(DnaSelectionResult):
    start_window: int
    end_window: int

    def __repr__(self) -> str:
        # Use f-string to create a formatted string
        return f"DnaWindowResult(id={self.id!r}, chromosome={self.chromosome!r}, strand={self.strand!r}, start_window={self.start_window!r}, end_window={self.end_window!r}, seq={self.seq!r})"

    @field_validator("start_window", "end_window")
    def check_window(cls, v):
        if v < 0:
            raise ValueError(f"{v} must be positive")
        return v


class GeneSet(SortedSet):
    def __init__(self, iterable=None, key=None):
        super().__init__(iterable, key)
        for item in self:
            if not isinstance(item, str):
                raise ValueError(
                    f"All items in gene_set must be str, got {type(item).__name__}"
                )

    def __repr__(self):
        n = len(self)
        limited_items = (self)[:3]
        if len(self) > 3:
            return f"GeneSet(size={n}, items={limited_items}...)"
        elif len(self) < 3:
            return f"GeneSet(size={n}, items={limited_items})"


###########
# Abstract Base Class for structure


##########
# Class holding gene
class Gene(ABC):
    # model_config = ConfigDict(frozen=True, extra="forbid")
    seq: str

    @abstractmethod
    def window(self, window_size: int, is_max_size: bool = True) -> DnaWindowResult:
        pass

    @abstractmethod
    def window_five_prime(
        self, window_size: int, allow_undersize: bool = False
    ) -> DnaWindowResult:
        pass

    @abstractmethod
    def window_three_prime(
        self, window_size: int, allow_undersize: bool = False
    ) -> DnaWindowResult:
        pass

    def __len__(self) -> int:
        return len(self.seq)

    # name: str
    # seq: str
    # chromosome: int
    # start: int
    # end: int
    # strand: str
    # five_utr: str
    # three_utr: str


class Genome(ABC):
    # Used elsewhere [[torchcell/sgd/validation/valid_models.py]]
    # CHECK IF THIS IS NEEDED.. I think this is a pydantic thing
    # model_config = ConfigDict(frozen=True, extra="forbid")
    # TODO not sure if we need to specify all vars in the __init__
    # TODO do we need to set data_root like this?
    def __init__(self, data_root: str = None):
        self.data_root: str = data_root
        self.db: FeatureDB = None
        self.fasta_sequences: dict = None
        self.chr_to_nc: dict = None
        self.nc_to_chr: dict = None
        self.chr_to_len: dict = None
        self._gene_set: set[str] = None
        self._fasta_path: str = None
        self._gff_path: str = None

    @property
    def gene_set(self) -> GeneSet:
        if self._gene_set is None:
            self._gene_set = self.compute_gene_set()
        return self._gene_set

    @gene_set.setter
    def gene_set(self, value: set[str]):
        self._gene_set = value

    @abstractmethod
    def compute_gene_set(self) -> set[str]:
        pass  # Abstract methods don't have a body

    @abstractmethod
    def get_seq(
        self, chr: int | str, start: int, end: int, strand: str
    ) -> DnaSelectionResult:
        pass

    @property
    @abstractmethod
    def gene_attribute_table(self) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def feature_types(self) -> list[str]:
        pass

    @abstractmethod
    def __getitem__(self, item: str) -> Gene | None:
        pass

    # Not sure if it makes more sense to have the number of genes be the length or the sum bp over all chromosomes.
    def __len__(self) -> int:
        return len(self.gene_set)


############
# Helper functions
def mismatch_positions(seq1: str, seq2: str) -> list[int]:
    """
    Computes the positions at which two sequences differ.

    This function takes two sequences, seq1 and seq2, represented as strings
    and returns a list of positions at which the two sequences have different
    characters. The sequences must be of the same length, else a ValueError is raised.

    Args:
        seq1 (str): The first sequence to compare.
        seq2 (str): The second sequence to compare.

    Returns:
        list[str]: A list containing the positions at
        which the two sequences differ.
        An empty list is returned if the sequences are identical.

    Raises:
        ValueError: If the lengths of seq1 and seq2 are not equal.

    Example:
        >>> mismatch_positions("ATGC", "ATCC")
        [2]
    """

    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be the same length")
    mismatches = [i for i, (n1, n2) in enumerate(zip(seq1, seq2)) if n1 != n2]
    return mismatches


def get_chr_from_description(description: str) -> int:
    """
    Extracts the chromosome number from a given description string.

    Processes a description string containing either a chromosome
    in Roman numeral (e.g., "[chromosome=IX]").
    Or a location (e.g., "[location=mitochondrion]").
    If a chromosome is found, it converts the Roman numeral to an integer.
    If the location is a mitochondrion, it returns 0.
    Handles descriptions containing chromosome information in a predefined format.

    Args:
        description (str): Description string containing chromosome number

    Returns:
        int: Chromosome number.  Returns 0 if the location is mitochondrion.

    Raises:
        ValueError: If the Roman numeral conversion fails due to invalid format.

    Example:
        >>> get_chr_from_description("[chromosome=IX] some other info")
        9
        >>> get_chr_from_description("[location=mitochondrion] some other info")
        0
    """
    # CHECK - format might be specific yeast s288c genome
    desc_split = description.split()
    for part in desc_split:
        if part.startswith("[chromosome="):
            roman_num = part[
                len("[chromosome=") : -1
            ]  # Exclude the last char, which is "]"
            return roman_to_int(roman_num)
        elif part.startswith("[location="):
            # we assign mitochondiral DNA to chromosome 0 as convenience
            if part[len("[location=") : -1] == "mitochondrion":
                return 0


def roman_to_int(s: str) -> int:
    """
    Converts a Roman numeral string to an integer.

    This function interprets the given string `s` as a Roman numeral and
    returns its value as an integer. It handles the standard Roman numeral
    symbols (I, V, X, L, C, D, M) and uses the subtractive notation rule,
    where placing a smaller numeral to the left of a larger numeral
    represents subtraction (e.g., IV for 4).

    Args:
        s (str): The Roman numeral string to convert, consisting of the characters
                 I, V, X, L, C, D, M. It is assumed to be a valid Roman numeral.

    Returns:
        int: The integer value of the Roman numeral.

    Raises:
        KeyError: If input string contains characters without valid Roman numeral.

    Example:
        >>> roman_to_int("IV")
        4
        >>> roman_to_int("IX")
        9
        >>> roman_to_int("XIII")
        13

    """
    roman_to_int_mapping = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000,
    }
    result = 0
    for i in range(len(s)):
        if i > 0 and roman_to_int_mapping[s[i]] > roman_to_int_mapping[s[i - 1]]:
            result += roman_to_int_mapping[s[i]] - 2 * roman_to_int_mapping[s[i - 1]]
        else:
            result += roman_to_int_mapping[s[i]]
    return result


# Selection Window functions


def calculate_window_undersized(
    start: int, end: int, strand: str, window_size: int
) -> tuple[int, int]:
    """
    Calculate the start and end points of a genomic window, respecting the given strand.

    For "+" strand, the window is created from the `start` point, and for "-" strand,
    the window is created from the `end` point. This method ensures that the resulting
    window is of the specified `window_size`, handling undersized windows in the
    process. This method is particularly useful when selecting from the start of a gene.

    Args:
        start (int): The start point of the gene on the genome.
        end (int): The end point of the gene on the genome.
        strand (str): The strand of the gene, either "+" or "-".
        window_size (int): The desired size of the window.

    Returns:
        tuple[int, int]: The calculated start and end points of the window.

    Raises:
        AssertionError: If the resulting window size doesn't match the
            specified `window_size`.

    Example:
        >>> calculate_window_undersized(10, 50, "+", 20)
        (10, 30)

        >>> calculate_window_undersized(10, 50, "-", 20)
        (30, 50)

    """
    # select from start of gene, since this is such a strong signal for function
    if strand == "+":
        start_window = start
        end_window = start + window_size
    elif strand == "-":
        start_window = end - window_size
        end_window = end
    assert (
        end_window - start_window
    ) == window_size, (
        f"Window sizing is incorrect. Window is larger than {window_size}bp"
    )
    return start_window, end_window


def calculate_window_bounds(
    start: int, end: int, strand: str, window_size: int, chromosome_length: int
) -> tuple[int, int]:
    """
    Calculate the window bounds for genomic sequences.

    This function calculates window bounds for genomic sequences while considering
    strand direction. It ensures the window does not exceed the chromosome length.

    Args:
        start (int): The start point of the sequence on the genome.
        end (int): The end point of the sequence on the genome.
        strand (str): The strand of the gene, either "+" or "-".
        window_size (int): Desired window size.
        chromosome_length (int): Length of the chromosome.

    Returns:
        tuple[int, int]: The calculated start and end points of the window.

    Raises:
        ValueError: If the end position is out of bounds of the chromosome,
            if the start position is greater than or equal to the end position,
            or if the window size is greater than the chromosome length.

    Examples:
        >>> calculate_window_bounds(0, 20, "+", 40, 100)
        (0, 40)
        >>> calculate_window_bounds(5, 25, "+", 50, 100)
        (0, 50)
        >>> calculate_window_bounds(75, 95, "+", 50, 100)
        (50, 100)
        >>> calculate_window_bounds(0, 20, "-", 40, 100)
        (0, 40)
    """
    if end > chromosome_length:
        raise ValueError("End position is out of bounds of chromosome")
    if start >= end:
        raise ValueError("Start position must be less than end position")
    if window_size > chromosome_length:
        raise ValueError("Window size should never be greater than chromosome length")

    seq_length = end - start
    if window_size < seq_length:
        # log info that the window size is smaller than the sequence
        log.info(
            f"Window size {window_size} is smaller than sequence length {end - start}."
        )
        start_window, end_window = calculate_window_undersized(
            start, end, strand, window_size
        )
        return start_window, end_window

    flank_seq_length = (window_size - seq_length) // 2
    start_window = start - flank_seq_length
    end_window = end + flank_seq_length

    # Case where start of window is out of bounds
    if start_window < 0:
        start_window = 0  # Set start of window to beginning of sequence
        end_window = window_size

    # Case where end of window is out of bounds
    if end_window > chromosome_length:
        end_window = chromosome_length  # Set end of window to end of sequence
        start_window = chromosome_length - window_size

    # Edge case if the adjusted window does not match the window size
    # and is up against chromosome ends
    if (
        abs((end_window - start_window) - window_size) == 1
        and end_window == chromosome_length
    ):
        start_window -= 1
    elif abs((end_window - start_window) - window_size) == 1 and start_window == 0:
        end_window += 1
    # Select more of the upstream
    elif abs((end_window - start_window) - window_size) == 1 and strand == "+":
        start_window -= 1
    elif abs((end_window - start_window) - window_size) == 1 and strand == "-":
        end_window += 1
    assert start_window <= start, "Start window must be leq start."
    assert end_window >= end, "End window must be geq end."
    return start_window, end_window


def calculate_window_undersized_symmetric(
    start: int, end: int, window_size: int
) -> tuple[int, int]:
    """
    Calculate symmetric window bounds for genomic sequences.

    This function calculates symmetric window bounds for genomic sequences and
    ensures that the window size is valid and the start and end are not equal.

    Args:
        start (int): The start point of the sequence on the genome.
        end (int): The end point of the sequence on the genome.
        window_size (int): Desired window size.

    Returns:
        tuple[int, int]: The calculated start and end points of the window.

    Raises:
        ValueError: If the start and end positions are the same, or if the
            window size is less than 2.

    Examples:
        >>> calculate_window_undersized_symmetric(10, 20, 4)
        (13, 17)
        >>> calculate_window_undersized_symmetric(10, 20, 5)
        (13, 17)

    Note:
        For odd window sizes, the result will be adjusted to keep the window
        symmetric around the middle of the start and end points.
    """
    if start == end:
        raise ValueError("Start and end positions are the same")
    if window_size < 2:
        raise ValueError("Window size must be at least 2")
    # find the middle
    middle = (start + end) // 2
    # take half above and half below
    flank_size = window_size // 2
    start_window = middle - flank_size
    end_window = middle + flank_size
    # Minus one for odd.
    assert (
        end_window - start_window
    ) >= window_size - 1, (
        f"Window sizing is incorrect. Window is larger than {window_size}bp"
    )
    return start_window, end_window


def calculate_window_bounds_symmetric(
    start: int, end: int, window_size: int, chromosome_length: int
) -> tuple[int, int]:
    r"""
    Calculate symmetric window bounds considering chromosome limits.

    This function calculates symmetric window bounds for genomic sequences, taking
    the chromosome length into account to avoid exceeding it, and also ensuring
    that the window size, start and end positions are valid within these limits.

    Args:
        start (int): The start point of the sequence on the genome.
        end (int): The end point of the sequence on the genome.
        window_size (int): The desired window size.
        chromosome_length (int): The total length of the chromosome.

    Returns:
        tuple[int, int]: The calculated start and end points of the symmetric window.

    Raises:
        ValueError: If the end position is out of chromosome bounds, start position
            is not less than end position, or window size is greater than chromosome
            length.

    Examples:
        >>> calculate_window_bounds_symmetric(5, 15, 30, 100)
        (0, 20)
        >>> calculate_window_bounds_symmetric(80, 95, 30, 100)
        (75, 100)
        >>> calculate_window_bounds_symmetric(45, 55, 30, 100)
        (35, 65)

    Note:
        The function also asserts that the calculated window is within the desired
        limits, symmetric around the middle of the sequence, and the start and end
        of the window are appropriately bounded within the sequence limits.
    """

    if end > chromosome_length:
        raise ValueError("End position is out of bounds of chromosome")
    if start >= end:
        raise ValueError("Start position must be less than end position")
    if window_size > chromosome_length:
        raise ValueError("Window size should never be greater than chromosome length")

    if window_size < end - start:
        start_window, end_window = calculate_window_undersized_symmetric(
            start, end, window_size
        )
        return start_window, end_window

    seq_length = end - start
    # Find limiting window size
    flank_size = (window_size - seq_length) // 2
    start_flank_pos = start - flank_size
    end_flank_pos = end + flank_size
    if start_flank_pos >= 0:
        start_window = start_flank_pos
    else:
        start_window = 0
    if end_flank_pos <= chromosome_length:
        end_window = end_flank_pos
    else:
        end_window = chromosome_length

    # Based on limiting window size take symmetric window
    sym_flank_len = min((start - start_window), (end_window - end))
    start_window = start - sym_flank_len
    end_window = end + sym_flank_len

    assert (
        end_window - start_window
    ) <= window_size, (
        f"Window sizing is incorrect. Window is larger than {window_size}bp"
    )
    assert (
        end_window - start_window
    ) >= end - start, (
        f"Window sizing is incorrect. Window is smaller than sequence {end - start}bp"
    )
    assert (start - start_window) == (end_window - end), "sequences are not symmetric"
    assert start_window <= start, "Start window must be leq start."
    assert end_window >= end, "End window must be geq end."

    return start_window, end_window


class CodonFrequency(SortedDict):
    def __repr__(self):
        if len(self) != 64:
            return "Invalid CodonFrequency: Expected 64 codons"

        sum_freq = sum(self.values())
        if not 0.9999 <= sum_freq <= 1.0001:  # Allowing a small deviation
            return (
                f"Invalid CodonFrequency: Frequencies do not sum to 1 "
                f"(sum={sum_freq})"
            )

        # Sort by frequency in descending order and get the 3 most frequent codons
        most_frequent_codons = sorted(self.items(), key=lambda x: x[1], reverse=True)[
            :3
        ]

        # Round the frequencies to the 4th decimal
        rounded_codons = [
            (codon, round(freq, 4)) for codon, freq in most_frequent_codons
        ]

        return (
            f"CodonFrequency(size={len(self)}, "
            f"most_frequent_codons={rounded_codons}...)"
        )


def compute_codon_frequency(cds_str: str) -> CodonFrequency:
    nucleotides = ["A", "T", "G", "C"]
    all_codons = ["".join(codon) for codon in product(nucleotides, repeat=3)]
    if len(cds_str) % 3 != 0 or not set(cds_str).issubset(set(nucleotides)):
        raise ValueError(
            "Invalid CDS string; length must be a multiple of 3 and only contain A, T, G, C."
        )

    codon_counts = defaultdict(int, {codon: 0 for codon in all_codons})

    for i in range(0, len(cds_str), 3):
        codon = cds_str[i : i + 3]
        codon_counts[codon] += 1

    total_codons = len(cds_str) // 3

    # Convert counts to frequency and create a CodonFrequency object
    codon_frequency = CodonFrequency(
        {codon: count / total_codons for codon, count in codon_counts.items()}
    )

    return codon_frequency


class ParsedGenome(ModelStrictArbitrary):
    gene_set: GeneSet

    @field_validator("gene_set")
    def validate_gene_set(cls, value):
        if not isinstance(value, GeneSet):
            raise ValueError(f"gene_set must be a GeneSet, got {type(value).__name__}")
        return value
