# torchcell/sequence/data.py
# [[torchcell.sequence.data]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sequence/data.py
# Test file: /torchcell/sequence/test_data.py
"""Core sequence data models, genome/gene abstractions, and codon helpers."""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable
from itertools import product
from typing import Any

import pandas as pd
from gffutils import FeatureDB
from pydantic import field_validator, model_validator
from sortedcontainers import SortedDict, SortedSet

from torchcell.datamodels import ModelStrict, ModelStrictArbitrary
from torchcell.sequence.db_connection import DatabaseConnectionManager

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class DnaSelectionResult(ModelStrict):
    """A validated DNA region selection with coordinates, strand, and sequence."""

    id: str
    chromosome: int
    strand: str
    start: int
    end: int
    seq: str

    def __len__(self) -> int:
        """Return the length of the selected sequence."""
        return len(self.seq)

    def __ge__(self, other: str) -> bool:
        """Compare by sequence length (greater-than-or-equal)."""
        if isinstance(other, DnaSelectionResult):
            return len(self.seq) >= len(other.seq)
        return NotImplemented

    def __le__(self, other: str) -> bool:
        """Compare by sequence length (less-than-or-equal)."""
        if isinstance(other, DnaSelectionResult):
            return len(self.seq) <= len(other.seq)
        return NotImplemented

    @model_validator(mode="before")
    @classmethod
    def end_leq_start(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate that start is not greater than end."""
        start: Any = values.get("start")
        end: Any = values.get("end")
        if start > end:
            raise ValueError("Start must be less than end")
        return values

    @field_validator("strand")
    def check_strand(cls, v: str) -> str:
        """Validate that the strand is '+' or '-'."""
        if v not in ["+", "-"]:
            raise ValueError("Strand must be either '+' or '-'")
        return v

    @field_validator("chromosome", "start", "end")
    def check_positive(cls, v: int) -> int:
        """Validate that the coordinate value is non-negative."""
        if v < 0:
            raise ValueError(f"{v} must be positive")
        return v

    # TODO consider adding chromosome length
    @field_validator("seq")
    def check_seq_len(cls, v: str) -> str:
        """Validate that the sequence length is non-negative."""
        sequence_length = len(v)
        if sequence_length < 0:
            raise ValueError(f"Sequence length ({sequence_length}) not geq 0")
        return v


class DnaWindowResult(DnaSelectionResult):
    """A DNA selection extended with the surrounding window coordinates."""

    start_window: int
    end_window: int

    def __repr__(self) -> str:
        """Return a detailed representation including window coordinates."""
        # Use f-string to create a formatted string
        return f"DnaWindowResult(id={self.id!r}, chromosome={self.chromosome!r}, strand={self.strand!r}, start_window={self.start_window!r}, end_window={self.end_window!r}, seq={self.seq!r})"

    @field_validator("start_window", "end_window")
    def check_window(cls, v: int) -> int:
        """Validate that the window coordinate is non-negative."""
        if v < 0:
            raise ValueError(f"{v} must be positive")
        return v


class GeneSet(SortedSet):  # type: ignore[misc]  # SortedSet is untyped (no stubs) under strict
    """A sorted set that requires all members to be strings."""

    def __init__(
        self,
        iterable: Iterable[str] | None = None,
        key: Callable[[str], Any] | None = None,
    ) -> None:
        """Build the sorted set and validate every item is a string.

        Args:
            iterable: Optional initial gene identifiers.
            key: Optional sort key passed to ``SortedSet``.
        """
        super().__init__(iterable, key)
        for item in self:
            if not isinstance(item, str):
                raise ValueError(
                    f"All items in gene_set must be str, got {type(item).__name__}"
                )

    def __repr__(self) -> str:  # type: ignore[return]  # pre-existing gap: size==3 returns None
        """Return a summary representation showing size and first items."""
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
    """Abstract gene exposing windowed-sequence extraction methods."""

    # model_config = ConfigDict(frozen=True, extra="forbid")
    seq: str

    @abstractmethod
    def window(self, window_size: int, is_max_size: bool = True) -> DnaWindowResult:
        """Return a sequence window centered on the gene."""
        pass

    @abstractmethod
    def window_five_prime(
        self, window_size: int, allow_undersize: bool = False
    ) -> DnaWindowResult:
        """Return a sequence window anchored at the 5' end."""
        pass

    @abstractmethod
    def window_three_prime(
        self, window_size: int, allow_undersize: bool = False
    ) -> DnaWindowResult:
        """Return a sequence window anchored at the 3' end."""
        pass

    def __len__(self) -> int:
        """Return the length of the gene sequence."""
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
    """Abstract genome providing lazy gene-set and sequence access."""

    # Used elsewhere [[torchcell/sgd/validation/valid_models.py]]
    # CHECK IF THIS IS NEEDED.. I think this is a pydantic thing
    # model_config = ConfigDict(frozen=True, extra="forbid")
    # TODO not sure if we need to specify all vars in the __init__
    # TODO do we need to set data_root like this?
    def __init__(self, data_root: str | None = None):
        """Initialize empty caches and store the data root.

        Args:
            data_root: Optional path to the genome data directory.
        """
        self.data_root: str | None = data_root
        self._db_connection_manager: DatabaseConnectionManager[FeatureDB] | None = None
        # FASTA records are dynamic BioPython SeqRecord objects keyed by id.
        self.fasta_sequences: dict[str, Any] | None = None
        self.chr_to_nc: dict[int, str] | None = None
        self.nc_to_chr: dict[str, int] | None = None
        self.chr_to_len: dict[int, int] | None = None
        self._gene_set: GeneSet | None = None
        self._fasta_path: str | None = None
        self._gff_path: str | None = None

    @property
    def db(self) -> FeatureDB | None:
        """Get database connection - lazy loaded per process/thread.

        Subclasses should set self._db_connection_manager to enable database access.
        """
        if self._db_connection_manager is None:
            return None
        return self._db_connection_manager.get_connection()

    @property
    def gene_set(self) -> GeneSet:
        """Return the gene set, computing and caching it on first access."""
        if self._gene_set is None:
            self._gene_set = self.compute_gene_set()
        return self._gene_set

    @gene_set.setter
    def gene_set(self, value: GeneSet) -> None:
        """Override the cached gene set."""
        self._gene_set = value

    @abstractmethod
    def compute_gene_set(self) -> GeneSet:
        """Compute and return the set of gene identifiers."""
        pass  # Abstract methods don't have a body

    @abstractmethod
    def get_seq(
        self, chr: int | str, start: int, end: int, strand: str
    ) -> DnaSelectionResult:
        """Return the DNA selection for the given coordinates and strand."""
        pass

    @property
    @abstractmethod
    def gene_attribute_table(self) -> pd.DataFrame:
        """Return a table of per-gene attributes."""
        pass

    @property
    @abstractmethod
    def feature_types(self) -> list[str]:
        """Return the list of supported feature types."""
        pass

    @abstractmethod
    def __getitem__(self, item: str) -> Gene | None:
        """Return the gene for the given identifier, or None if absent."""
        pass

    # Not sure if it makes more sense to have the number of genes be the length or the sum bp over all chromosomes.
    def __len__(self) -> int:
        """Return the number of genes in the genome."""
        return len(self.gene_set)


############
# Helper functions
def mismatch_positions(seq1: str, seq2: str) -> list[int]:
    """Computes the positions at which two sequences differ.

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


def get_chr_from_description(description: str) -> int:  # type: ignore[return]  # pre-existing: None for unmatched descriptions
    """Extracts the chromosome number from a given description string.

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
    # CHECK - format might be specific yeast S288C genome
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
    """Converts a Roman numeral string to an integer.

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
    """Calculate the start and end points of a genomic window, respecting the given strand.

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
    assert (end_window - start_window) == window_size, (
        f"Window sizing is incorrect. Window is larger than {window_size}bp"
    )
    return start_window, end_window


def calculate_window_bounds(
    start: int, end: int, strand: str, window_size: int, chromosome_length: int
) -> tuple[int, int]:
    """Calculate the window bounds for genomic sequences.

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
    """Calculate symmetric window bounds for genomic sequences.

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
    assert (end_window - start_window) >= window_size - 1, (
        f"Window sizing is incorrect. Window is larger than {window_size}bp"
    )
    return start_window, end_window


def calculate_window_bounds_symmetric(
    start: int, end: int, window_size: int, chromosome_length: int
) -> tuple[int, int]:
    r"""Calculate symmetric window bounds considering chromosome limits.

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

    assert (end_window - start_window) <= window_size, (
        f"Window sizing is incorrect. Window is larger than {window_size}bp"
    )
    assert (end_window - start_window) >= end - start, (
        f"Window sizing is incorrect. Window is smaller than sequence {end - start}bp"
    )
    assert (start - start_window) == (end_window - end), "sequences are not symmetric"
    assert start_window <= start, "Start window must be leq start."
    assert end_window >= end, "End window must be geq end."

    return start_window, end_window


class CodonFrequency(SortedDict):  # type: ignore[misc]  # SortedDict is untyped (no stubs) under strict
    """Sorted mapping of all 64 codons to their relative frequencies."""

    def __repr__(self) -> str:
        """Return a representation validating 64 codons summing to one."""
        if len(self) != 64:
            return "Invalid CodonFrequency: Expected 64 codons"

        sum_freq = sum(self.values())
        if not 0.9999 <= sum_freq <= 1.0001:  # Allowing a small deviation
            return (
                f"Invalid CodonFrequency: Frequencies do not sum to 1 (sum={sum_freq})"
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
    """Compute relative codon frequencies for a coding sequence.

    Args:
        cds_str: Coding sequence whose length must be a multiple of three.

    Returns:
        A CodonFrequency mapping every codon to its relative frequency.
    """
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
    """Validated container holding a genome's GeneSet."""

    gene_set: GeneSet

    @field_validator("gene_set")
    def validate_gene_set(cls, value: GeneSet) -> GeneSet:
        """Validate that the provided value is a GeneSet."""
        if not isinstance(value, GeneSet):
            raise ValueError(f"gene_set must be a GeneSet, got {type(value).__name__}")
        return value
