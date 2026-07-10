# torchcell/data/data.py
# [[torchcell.data.data]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/data.py
# Test file: tests/torchcell/data/test_data.py
"""Core data containers mapping experiments to their shared references."""

import hashlib
from collections.abc import Iterator
from typing import Any

from pydantic import field_validator

from torchcell.datamodels import ExperimentReferenceType, ModelStrict


class ExperimentReferenceIndex(ModelStrict):
    """A reference paired with the sorted record indices of the experiments using it.

    SPARSE by construction (WS15 streaming redesign): every experiment belongs to exactly
    ONE reference, so ``member_indices`` across a dataset's references sum to N -- NOT the
    former dense ``index: list[bool]`` mask, which stored a length-N boolean per reference
    and was therefore O(references x N). That blowup (~60 GB at Hoepfner's ~30M records x
    ~1.8k references) made genome-scale datasets infeasible to build or store; the sparse
    form is O(N) total for any dataset size, small or large, with ONE representation and
    ONE (always-streaming) build path.

    ``mask(n)`` reconstructs the dense length-``n`` boolean mask ON DEMAND for consumers
    that want it (small datasets / vectorized code). Genome-scale consumers iterate
    ``member_indices`` directly and never materialize the dense mask.
    """

    reference: ExperimentReferenceType
    member_indices: list[int]

    @field_validator("member_indices")
    def _validate_member_indices(cls, v: list[int]) -> list[int]:
        """Member indices are unique, non-negative, and stored sorted (canonical form)."""
        if any(i < 0 for i in v):
            raise ValueError("member_indices must be non-negative record positions")
        if len(v) != len(set(v)):
            raise ValueError("member_indices must be unique")
        return sorted(v)

    @classmethod
    def from_stored(cls, item: dict[str, Any]) -> "ExperimentReferenceIndex":
        """Build from a stored JSON entry, accepting BOTH the new and legacy formats.

        New entries carry ``member_indices``; legacy entries (written before the WS15
        sparse redesign) carry a dense ``index: list[bool]`` mask. Converting on read
        keeps every already-built dataset's ``experiment_reference_index.json`` loadable
        without a forced rebuild; the next rebuild writes the new sparse format.
        """
        if "member_indices" in item:
            return cls.model_validate(item)
        if "index" in item:
            mask = item["index"]
            member_indices = [i for i, present in enumerate(mask) if present]
            return cls(reference=item["reference"], member_indices=member_indices)
        raise ValueError(
            "stored ExperimentReferenceIndex entry has neither 'member_indices' nor 'index'"
        )

    def mask(self, n: int) -> list[bool]:
        """Reconstruct the dense length-``n`` boolean membership mask (opt-in, small sets)."""
        dense = [False] * n
        for i in self.member_indices:
            dense[i] = True
        return dense

    # Use for parallel computation of a Dataset ExperimentReferenceIndices
    def combine(self, other: "ExperimentReferenceIndex") -> "ExperimentReferenceIndex":
        """Return a new index UNION-ing two member-index sets that share a reference."""
        if self.reference != other.reference:
            raise ValueError(
                "Cannot combine ExperimentReferenceIndex objects with different references"
            )
        combined = sorted(set(self.member_indices) | set(other.member_indices))
        return ExperimentReferenceIndex(
            reference=self.reference, member_indices=combined
        )

    def __repr__(self) -> str:
        """Return a string showing the reference and a truncated member-index list."""
        head = self.member_indices[:5]
        ellipsis = "..." if len(self.member_indices) > 5 else ""
        return (
            f"ExperimentReferenceIndex(reference={self.reference}, "
            f"member_indices={head}{ellipsis})"
        )


class ReferenceIndex(ModelStrict):
    """A collection of reference-indices partitioning a dataset's experiments."""

    data: list[ExperimentReferenceIndex]

    def __getitem__(self, index: int) -> ExperimentReferenceIndex:
        """Return the reference-index entry at ``index``."""
        return self.data[index]

    def __len__(self) -> int:
        """Return the number of reference-index entries."""
        return len(self.data)

    def __iter__(self) -> Iterator[ExperimentReferenceIndex]:  # type: ignore[override]  # intentionally iterates entries, not pydantic BaseModel's (field, value) tuples
        """Iterate over the reference-index entries."""
        return iter(self.data)

    @field_validator("data")
    def validate_data(
        cls, v: list[ExperimentReferenceIndex]
    ) -> list[ExperimentReferenceIndex]:
        """Check the member-index sets partition ``range(N)`` exactly (cover once, no gap)."""
        all_indices = [i for eri in v for i in eri.member_indices]
        if sorted(all_indices) != list(range(len(all_indices))):
            raise ValueError(
                "member_indices across references must partition range(N) exactly "
                "(every experiment covered by exactly one reference, no gaps/overlaps)"
            )
        return v


def compute_sha256_hash(content: str) -> str:
    """Compute the sha256 hash of a string."""
    return hashlib.sha256(content.encode()).hexdigest()
