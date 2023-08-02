# src/torchcell/sequence/sequence.py
from abc import ABC, abstractmethod

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

from torchcell.data_models import BaseModelStrict
from torchcell.models.constants import DNA_LLM_MAX_TOKEN_SIZE


###########
# Abstract Base Class for structure
class AbcGenome(ABC):
    # Used elsewhere [[src/torchcell/sgd/validation/valid_models.py]]
    model_config = ConfigDict(frozen=True, extra="forbid")


###########
# Classes holding data
class DnaSelectionResult(BaseModel):
    seq: str
    chromosome: int
    start: int
    end: int
    strand: str

    def __len__(self):
        return len(self.seq)

    def __ge__(self, other):
        if isinstance(other, DnaSelectionResult):
            return len(self.seq) >= len(other.seq)
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, DnaSelectionResult):
            return len(self.seq) <= len(other.seq)
        return NotImplemented

    @model_validator(mode="after")
    @classmethod
    def check_seq_len(cls, model: "DnaSelectionResult") -> "DnaSelectionResult":
        sequence_length = len(model.seq)
        if sequence_length <= 0 or sequence_length >= DNA_LLM_MAX_TOKEN_SIZE:
            raise ValueError(
                "Sequence length must be positive and less than or equal to 60kb"
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
    def check_strand(cls, v):
        if v not in ["+", "-"]:
            raise ValueError("Strand must be either '+' or '-'")
        return v

    @field_validator("chromosome")
    @classmethod
    def check_chromosome(cls, v):
        if v < 0:
            raise ValueError("Chromosome must be positive")
        return v

    @field_validator("start")
    @classmethod
    def check_start(cls, v):
        if v < 0:
            raise ValueError("Start must be positive")
        return v

    @field_validator("end")
    @classmethod
    def check_end(cls, v):
        if v < 0:
            raise ValueError("End must be positive")
        return v


############
# Helper functions
def mismatch_positions(seq1, seq2):
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be the same length")
    mismatches = [i for i, (n1, n2) in enumerate(zip(seq1, seq2)) if n1 != n2]
    return mismatches


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
            part = part[len("[location=") : -1] == "mitochondrion"
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


if __name__ == "__main__":
    pass
