# src/torchcell/sequence/data.py
from abc import ABC, abstractmethod
from turtle import st

import gffutils
import matplotlib.pyplot as plt
import pandas as pd
from attrs import define, field
from Bio import SeqIO
from Bio.Seq import Seq
from gffutils import Feature, FeatureDB
from gffutils.biopython_integration import to_seqfeature
from matplotlib import pyplot as plt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from sympy import sequence
import logging
from torchcell.data_models import BaseModelStrict
from torchcell.models.constants import DNA_LLM_MAX_TOKEN_SIZE

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


###########
# Abstract Base Class for structure
# CHECK not finished
class Genome(ABC):
    # Used elsewhere [[src/torchcell/sgd/validation/valid_models.py]]
    model_config = ConfigDict(frozen=True, extra="forbid")


##########
# Class holding gene
@define
class Gene:
    name: str
    seq: str
    chromosome: int
    start: int
    end: int
    strand: str
    five_utr: str
    three_utr: str

    def __attrs_post__init():
        pass

    # @property()
    # def


###########
# Classes holding data
class DnaSelectionResult(BaseModel):
    seq: str
    chromosome: int
    start: int
    end: int
    strand: str

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

    @model_validator(mode="after")  # type : ignore
    @classmethod
    def check_seq_len(cls, model: "DnaSelectionResult") -> "DnaSelectionResult":
        sequence_length = len(model.seq)
        if sequence_length < 0 or sequence_length > DNA_LLM_MAX_TOKEN_SIZE:
            raise ValueError(
                f"Sequence length ({sequence_length}) not geq 0 and leq {DNA_LLM_MAX_TOKEN_SIZE}"
            )
        return model

    @model_validator(mode="after")
    @classmethod
    def start_geq_end(cls, model: "DnaSelectionResult") -> "DnaSelectionResult":
        if model.start >= model.end:
            raise ValueError("Start must be less than end")
        return model

    @field_validator("strand")
    @classmethod
    def check_strand(cls, v: str) -> str:
        if v not in ["+", "-"]:
            raise ValueError("Strand must be either '+' or '-'")
        return v

    @field_validator("chromosome")
    @classmethod
    def check_chromosome(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Chromosome must be positive")
        return v

    @field_validator("start")
    @classmethod
    def check_start(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Start must be positive")
        return v

    @field_validator("end")
    @classmethod
    def check_end(cls, v: int) -> int:
        if v < 0:
            raise ValueError("End must be positive")
        return v


class DnaWindowResult(DnaSelectionResult):
    size: int
    start_window: int
    end_window: int

    @model_validator(mode="after")
    @classmethod
    def check_window(cls, model: "DnaWindowResult") -> "DnaWindowResult":
        if model.start_window < 0:
            raise ValueError("Start window must be positive")
        if model.end_window < 0:
            raise ValueError("End window must be positive")
        return model


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
    start: int,
    end: int,
    strand: str,
    window_size: int,
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
    start: int,
    end: int,
    strand: str,
    window_size: int,
    chromosome_length: int,
) -> tuple[int, int]:
    if end >= chromosome_length:
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
        start_window, end_window = calculate_window_undersized(
            start, end, strand, window_size
        )
        return start_window, end_window

    seq_length = end - start
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

    # Edge case for if the adjusted window does not match the window size
    # Against chromosome ends
    if abs(
        (end_window - start_window) - window_size == 1
        and end_window == chromosome_length
    ):
        start_window -= 1
    elif abs((end_window - start_window) - window_size) == 1 and start_window == 0:
        end_window += 1
    # Select more of the 5' UTR
    elif abs((end_window - start_window) - window_size) == 1 and strand == "+":
        start_window -= 1
    elif abs((end_window - start_window) - window_size) == 1 and strand == "-":
        end_window += 1
    assert start_window <= start, "Start window must be leq start."
    assert end_window >= end, "End window must be geq end."
    return start_window, end_window


def calculate_window_undersized_symmetric(
    start: int,
    end: int,
    window_size: int,
) -> tuple[int, int]:
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
    start: int,
    end: int,
    window_size: int,
    chromosome_length: int,
) -> tuple[int, int]:
    if end >= chromosome_length:
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
            start, end, window_size, chromosome_length
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


if __name__ == "__main__":
    pass
