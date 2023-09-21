# src/torchcell/sequence/data.py
# [[src.torchcell.sequence.data]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/sequence/data.py
# Test file: /src/torchcell/sequence/test_data.py

import logging
from abc import ABC, abstractmethod
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
from pydantic import BaseModel, ConfigDict, Field, ValidationError, validator
from sortedcontainers import SortedSet
from sympy import sequence

from torchcell.datamodels import ModelStrict
from torchcell.models.constants import DNA_LLM_MAX_TOKEN_SIZE

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


###########
# Classes holding data
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

    @validator("end", pre=True, always=True)
    def end_leq_start(cls, v, values):
        if "start" in values and v <= values["start"]:
            raise ValueError("Start must be less than end")
        return v

    @validator("strand", pre=True, always=True)
    def check_strand(cls, v):
        if v not in ["+", "-"]:
            raise ValueError("Strand must be either '+' or '-'")
        return v

    @validator("chromosome", "start", "end", pre=True, always=True)
    def check_positive(cls, v):
        if v < 0:
            raise ValueError(f"{v} must be positive")
        return v

    @validator("seq", pre=True, always=True)
    def check_seq_len(cls, v):
        sequence_length = len(v)
        if sequence_length < 0 or sequence_length > DNA_LLM_MAX_TOKEN_SIZE:
            raise ValueError(
                f"Sequence length ({sequence_length}) not geq 0 and leq {DNA_LLM_MAX_TOKEN_SIZE}"
            )
        return v


class DnaWindowResult(DnaSelectionResult):
    start_window: int
    end_window: int

    def __repr__(self) -> str:
        # Use f-string to create a formatted string
        return f"DnaWindowResult(id={self.id!r}, chromosome={self.chromosome!r}, strand={self.strand!r}, start_window={self.start_window!r}, end_window={self.end_window!r}, seq={self.seq!r})"

    @validator("start_window", "end_window", pre=True, always=True)
    def check_window(cls, v):
        if v < 0:
            raise ValueError(f"{v} must be positive")
        return v


###########
# Abstract Base Class for structure


##########
# Class holding gene
class Gene(ABC):
    model_config = ConfigDict(frozen=True, extra="forbid")

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

    # name: str
    # seq: str
    # chromosome: int
    # start: int
    # end: int
    # strand: str
    # five_utr: str
    # three_utr: str


class Genome(ABC):
    # Used elsewhere [[src/torchcell/sgd/validation/valid_models.py]]
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
    def gene_set(self) -> set[str]:
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

    @abstractmethod
    def select_feature_window(
        self, feature: str, window_size: int, is_max_size: bool = True
    ) -> DnaWindowResult:
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
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be the same length")
    mismatches = [i for i, (n1, n2) in enumerate(zip(seq1, seq2)) if n1 != n2]
    return mismatches


# CHECK - format might be specific yeast
def get_chr_from_description(description: str) -> int:
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


#######


# selection_feature_window functions
def calculate_window_undersized(
    start: int, end: int, strand: str, window_size: int
) -> tuple[int, int]:
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
    if end > chromosome_length:
        raise ValueError("End position is out of bounds of chromosome")
    if start >= end:
        raise ValueError("Start position must be less than end position")
    if window_size > chromosome_length:
        raise ValueError("Window size should never be greater than chromosome length")

    if window_size < end - start:
        # log info that the window size is smaller than the sequence
        log.info(
            f"Window size {window_size} is smaller than sequence length {end - start}."
        )
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


#
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
        return f"GeneSet(size={n}, items={limited_items}...)"


if __name__ == "__main__":
    DnaSelectionResult(
        id="gene_name", seq="ATGC", chromosome=1, start=4, end=0, strand="+"
    )
    pass
