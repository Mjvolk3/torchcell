# torchcell/data/data.py
# [[torchcell.data.data]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/data.py
# Test file: tests/torchcell/data/test_data.py

import hashlib
import json
from typing import List

from pydantic import field_validator

from torchcell.datamodels import (
    ExperimentReference,
    FitnessExperimentReference,
    ModelStrict,
)

from typing import Union

# List of classes that can be part of ExperimentReference
ExperimentReferenceType = ExperimentReference | FitnessExperimentReference


class ExperimentReferenceIndex(ModelStrict):
    reference: ExperimentReferenceType
    index: List[bool]

    # Use for parallel computation of a Dataset ExperimentReferenceIndices
    def combine(self, other: "ExperimentReferenceIndex"):
        if self.reference != other.reference:
            raise ValueError(
                "Cannot combine ExperimentReferenceIndex objects with different references"
            )
        combined_index = [a or b for a, b in zip(self.index, other.index)]
        return ExperimentReferenceIndex(reference=self.reference, index=combined_index)

    def __repr__(self):
        if len(self.index) > 5:
            return f"ExperimentReferenceIndex(reference={self.reference}, index={self.index[:5]}...)"
        else:
            return f"ExperimentReferenceIndex(reference={self.reference}, index={self.index})"


class ReferenceIndex(ModelStrict):
    data: List[ExperimentReferenceIndex]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    @field_validator("data")
    def validate_data(cls, v):
        summed_indices = sum(
            [
                boolean_value
                for exp_ref_index in v
                for boolean_value in exp_ref_index.index
            ]
        )

        if summed_indices != len(v[0].index):
            raise ValueError("Sum of indices must equal the number of experiments")
        return v


def compute_sha256_hash(content: str) -> str:
    """
    Compute the sha256 hash of a string.
    """
    return hashlib.sha256(content.encode()).hexdigest()
