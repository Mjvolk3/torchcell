# torchcell/data/data.py
# [[torchcell.data.data]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/data.py
# Test file: tests/torchcell/data/test_data.py
"""Core data containers mapping experiments to their shared references."""

import hashlib

from pydantic import field_validator

from torchcell.datamodels import ExperimentReferenceType, ModelStrict


class ExperimentReferenceIndex(ModelStrict):
    """A reference paired with a boolean mask of experiments that use it."""

    reference: ExperimentReferenceType
    index: list[bool]

    # Use for parallel computation of a Dataset ExperimentReferenceIndices
    def combine(self, other: "ExperimentReferenceIndex"):
        """Return a new index OR-ing two masks that share the same reference."""
        if self.reference != other.reference:
            raise ValueError(
                "Cannot combine ExperimentReferenceIndex objects with different references"
            )
        combined_index = [a or b for a, b in zip(self.index, other.index)]
        return ExperimentReferenceIndex(reference=self.reference, index=combined_index)

    def __repr__(self):
        """Return a string showing the reference and a truncated mask."""
        if len(self.index) > 5:
            return f"ExperimentReferenceIndex(reference={self.reference}, index={self.index[:5]}...)"
        else:
            return f"ExperimentReferenceIndex(reference={self.reference}, index={self.index})"


class ReferenceIndex(ModelStrict):
    """A collection of reference-indices partitioning a dataset's experiments."""

    data: list[ExperimentReferenceIndex]

    def __getitem__(self, index):
        """Return the reference-index entry at ``index``."""
        return self.data[index]

    def __len__(self):
        """Return the number of reference-index entries."""
        return len(self.data)

    def __iter__(self):
        """Iterate over the reference-index entries."""
        return iter(self.data)

    @field_validator("data")
    def validate_data(cls, v):
        """Check that the masks partition every experiment exactly once."""
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
    """Compute the sha256 hash of a string."""
    return hashlib.sha256(content.encode()).hexdigest()
