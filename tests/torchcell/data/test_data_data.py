# tests/torchcell/data/test_data_data.py
# [[tests.torchcell.data.test_data_data]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_data_data.py
"""Tests for the reference-index containers and the hash helper (``data/data.py``).

Named ``test_data_data.py`` because ``tests/torchcell/sequence/test_data.py`` already owns
the basename ``test_data`` under pytest's prepend import mode (the invariants of the sparse
index itself are in ``test_experiment_reference_index.py``).

2026.09.30 (Phase 17). The fixture is one minimal ``FitnessExperimentReference``
distinguished by ``dataset_name``; the reference is an opaque payload to every function
here. Expected values:

- ``from_stored`` keeps the concrete reference class through the union for both the
  sparse and the legacy dense format, refuses an entry with neither key with the exact
  message, and refuses an entry carrying both (``ModelStrict`` forbids the extra
  ``index`` key) instead of letting one format win.
- ``repr`` shows at most the first five member indices, with ``...`` only past five.
- ``mask(n)`` with a member index at or beyond ``n`` raises ``IndexError``.
- ``ReferenceIndex`` indexes and iterates its entries (not pydantic's field tuples),
  refuses a gap in the partition, and (since issue #541) refuses two entries with equal
  references by ``DuplicateReferenceError``.
- ``compute_sha256_hash`` is SHA-256 of the UTF-8 bytes: the FIPS 180-2 test vectors for
  ``""`` and ``"abc"``, and ``"é"`` hashed as its two UTF-8 bytes ``c3 a9``.
"""

from __future__ import annotations

import hashlib
import math

import pytest
from pydantic import ValidationError

from torchcell.data.data import (
    DuplicateReferenceError,
    ExperimentReferenceIndex,
    ReferenceIndex,
    compute_sha256_hash,
    reference_key,
)
from torchcell.datamodels.schema import (
    Environment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Media,
    ReferenceGenome,
    Temperature,
)


def _reference(dataset_name: str) -> FitnessExperimentReference:
    return FitnessExperimentReference(
        dataset_name=dataset_name,
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="S288C"
        ),
        environment_reference=Environment(
            media=Media(name="YEPD", state="solid", is_synthetic=False),
            temperature=Temperature(value=30.0),
        ),
        phenotype_reference=FitnessPhenotype(
            graph_level="global",
            label_name="fitness",
            label_statistic_name="fitness_std",
            fitness=1.0,
            fitness_std=0.0,
        ),
    )


def test_from_stored_keeps_the_concrete_reference_class_in_both_formats() -> None:
    """A dumped fitness reference comes back as ``FitnessExperimentReference``, not a base."""
    dump = _reference("d").model_dump()
    sparse = ExperimentReferenceIndex.from_stored(
        {"reference": dump, "member_indices": [3, 1]}
    )
    legacy = ExperimentReferenceIndex.from_stored(
        {"reference": dump, "index": [False, False]}
    )
    assert type(sparse.reference).__name__ == "FitnessExperimentReference"
    assert sparse.member_indices == [1, 3]
    assert type(legacy.reference).__name__ == "FitnessExperimentReference"
    assert legacy.member_indices == []
    assert sparse.reference == _reference("d")


def test_from_stored_refuses_an_entry_with_neither_format() -> None:
    with pytest.raises(ValueError) as excinfo:
        ExperimentReferenceIndex.from_stored(
            {"reference": _reference("d").model_dump()}
        )
    assert str(excinfo.value) == (
        "stored ExperimentReferenceIndex entry has neither 'member_indices' nor 'index'"
    )


def test_from_stored_refuses_an_entry_carrying_both_formats() -> None:
    """Both keys: the sparse path validates and the strict model rejects ``index``."""
    with pytest.raises(ValidationError) as excinfo:
        ExperimentReferenceIndex.from_stored(
            {
                "reference": _reference("d").model_dump(),
                "member_indices": [0],
                "index": [True],
            }
        )
    assert [(e["loc"], e["type"]) for e in excinfo.value.errors()] == [
        (("index",), "extra_forbidden")
    ]


def test_repr_truncates_member_indices_after_five() -> None:
    reference = _reference("d")
    five = ExperimentReferenceIndex(reference=reference, member_indices=[4, 3, 2, 1, 0])
    six = ExperimentReferenceIndex(reference=reference, member_indices=list(range(6)))
    assert repr(five) == (
        f"ExperimentReferenceIndex(reference={reference}, member_indices=[0, 1, 2, 3, 4])"
    )
    assert repr(six) == (
        f"ExperimentReferenceIndex(reference={reference}, "
        "member_indices=[0, 1, 2, 3, 4]...)"
    )


def test_mask_refuses_a_length_shorter_than_the_largest_member() -> None:
    """``mask(3)`` cannot place member 5, and says so with ``IndexError``."""
    eri = ExperimentReferenceIndex(reference=_reference("d"), member_indices=[0, 5])
    assert eri.mask(6) == [True, False, False, False, False, True]
    with pytest.raises(IndexError):
        eri.mask(3)


def test_combine_merges_disjoint_members_and_refuses_another_reference() -> None:
    """References are compared by value: two separately built equal references combine
    into the sorted union of disjoint member sets ([2] and [0, 3] give [0, 2, 3], which
    neither input equals), and a different reference is refused with the exact message
    (data.py lines 73 to 76).
    """
    a = ExperimentReferenceIndex(reference=_reference("d"), member_indices=[2])
    b = ExperimentReferenceIndex(reference=_reference("d"), member_indices=[0, 3])
    assert a.combine(b).member_indices == [0, 2, 3]
    other = ExperimentReferenceIndex(reference=_reference("e"), member_indices=[1])
    with pytest.raises(
        ValueError,
        match=r"^Cannot combine ExperimentReferenceIndex objects with different references$",
    ):
        a.combine(other)


def test_reference_index_indexes_and_iterates_entries() -> None:
    """``ri[i]`` and ``iter(ri)`` yield entries, in stored order, not field tuples."""
    first = ExperimentReferenceIndex(reference=_reference("a"), member_indices=[2, 3])
    second = ExperimentReferenceIndex(reference=_reference("b"), member_indices=[0, 1])
    ri = ReferenceIndex(data=[first, second])
    assert [entry.member_indices for entry in ri] == [[2, 3], [0, 1]]
    assert ri[1].reference.dataset_name == "b"
    assert ri[-1].member_indices == [0, 1]
    assert len(ReferenceIndex(data=[])) == 0


def test_reference_index_refuses_a_gap_with_the_exact_message() -> None:
    """Members {0, 1, 3} leave 2 uncovered: sorted([0, 1, 3]) != [0, 1, 2]."""
    with pytest.raises(ValidationError) as excinfo:
        ReferenceIndex(
            data=[
                ExperimentReferenceIndex(
                    reference=_reference("a"), member_indices=[0, 1]
                ),
                ExperimentReferenceIndex(reference=_reference("b"), member_indices=[3]),
            ]
        )
    assert [e["msg"] for e in excinfo.value.errors()] == [
        "Value error, member_indices across references must partition range(N) "
        "exactly (every experiment covered by exactly one reference, no gaps/overlaps)"
    ]


def test_reference_index_refuses_one_reference_split_over_two_entries() -> None:
    """Two entries carrying EQUAL references are refused with ``DuplicateReferenceError``.

    Contract (issue #541): each reference appears in exactly one entry. The two entries
    below tile ``range(3)`` (so the partition check alone would pass) and their
    references are separately built but equal by value, at positions 0 and 2; entry 1
    is a different reference and is not named. The refusal is a ``ValueError`` subclass,
    so pydantic wraps it and keeps the exception object in the error context.
    Evidence the refusal breaks no build: 38 built dev stores under
    ``$DATA_ROOT/data/torchcell/`` hold no entries with equal references.
    """
    with pytest.raises(ValidationError) as excinfo:
        ReferenceIndex(
            data=[
                ExperimentReferenceIndex(reference=_reference("d"), member_indices=[0]),
                ExperimentReferenceIndex(reference=_reference("e"), member_indices=[1]),
                ExperimentReferenceIndex(reference=_reference("d"), member_indices=[2]),
            ]
        )
    errors = excinfo.value.errors()
    assert [e["msg"] for e in errors] == [
        "Value error, entries 0 and 2 carry equal references; each reference must "
        "appear in exactly one entry (merge them with ExperimentReferenceIndex.combine)"
    ]
    assert type(errors[0]["ctx"]["error"]) is DuplicateReferenceError


def _nan_reference() -> FitnessExperimentReference:
    """``_reference("d")`` with a NaN reference fitness, built fresh on every call.

    ``float("nan")`` makes a new NaN object each call, as two separately loaded
    references would hold; ``math.nan`` is one shared object, which ``==`` would match
    by identity.
    """
    reference = _reference("d")
    phenotype = reference.phenotype_reference.model_copy(
        update={"fitness": float("nan")}
    )
    return reference.model_copy(update={"phenotype_reference": phenotype})


def test_nan_bearing_references_are_refused_together_and_combine_merges_them() -> None:
    """Two separately built NaN-bearing references are unequal under ``==`` but share
    ``reference_key``; the container refuses them and the remedy its message names,
    ``combine``, merges them (issue #541 review).
    """
    a = ExperimentReferenceIndex(reference=_nan_reference(), member_indices=[0])
    b = ExperimentReferenceIndex(reference=_nan_reference(), member_indices=[1])
    assert a.reference != b.reference
    assert reference_key(a.reference) == reference_key(b.reference)
    with pytest.raises(ValidationError) as excinfo:
        ReferenceIndex(data=[a, b])
    assert type(excinfo.value.errors()[0]["ctx"]["error"]) is DuplicateReferenceError
    merged = a.combine(b)
    assert merged.member_indices == [0, 1]
    stored = merged.reference.model_dump()["phenotype_reference"]["fitness"]
    assert math.isnan(stored)
    assert len(ReferenceIndex(data=[merged])) == 1


def test_compute_sha256_hash_hashes_the_utf8_bytes() -> None:
    assert compute_sha256_hash("") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert compute_sha256_hash("abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert compute_sha256_hash("é") == hashlib.sha256(b"\xc3\xa9").hexdigest()
