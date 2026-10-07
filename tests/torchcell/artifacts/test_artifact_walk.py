# tests/torchcell/artifacts/test_artifact_walk.py
# [[tests.torchcell.artifacts.test_artifact_walk]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/artifacts/test_artifact_walk.py
"""``iter_refs`` / ``distinct_refs`` over minimal pydantic trees.

Refs A, B, C pin three different byte strings; A2 is A's file with another member, so it
shares A's ``(tier, key, path, sha256)``. The fixture ``Outer`` holds A in a direct field,
B inside a nested model inside a list, C as a dict value, A2 in a tuple, A again in an
extra field and B in a set; walking it yields six refs in field order and three
distinct keys, the first ref met per key kept (A, not A2).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from tests.torchcell.artifacts._fakes import ref_for
from torchcell.artifacts import ArtifactRef, distinct_refs, iter_refs, ref_key
from torchcell.datamodels import schema as s

A = ref_for("objects", "set-a", "a.npy", b"alpha")
A2 = ref_for("objects", "set-a", "a.npy", b"alpha", member="row-7")
B = ref_for("genomes", "peter2018", "genes.tar.gz", b"beta", member="YAL001C.fasta#t")
C = ref_for("raw", "caudal2024", "table.tsv", b"gamma")


class Inner(BaseModel):
    """A nested model holding one ref and a scalar."""

    label: str
    ref: ArtifactRef | None


class Outer(BaseModel):
    """Every container shape the walk descends, plus extras."""

    model_config = ConfigDict(extra="allow")

    direct: ArtifactRef
    nested: list[Inner]
    by_name: dict[str, ArtifactRef]
    pair: tuple[int, ArtifactRef]
    scalar: float
    unique: set[ArtifactRef]


def _outer() -> Outer:
    return Outer.model_validate(
        {
            "direct": A,
            "nested": [Inner(label="x", ref=None), Inner(label="y", ref=B)],
            "by_name": {"c": C},
            "pair": (1, A2),
            "scalar": 0.5,
            "unique": {B},
            "late": A,
        }
    )


def test_iter_refs_yields_every_ref_in_field_order_then_extras() -> None:
    """Direct A, nested B, dict C, tuple A2, set B, extra A: six, with repeats."""
    assert list(iter_refs(_outer())) == [A, B, C, A2, B, A]


def test_distinct_refs_keys_by_file_identity_and_keeps_the_first() -> None:
    """Three keys; A2 collapses onto A (same file), and A is the one kept."""
    found = distinct_refs([_outer()])
    assert list(found) == [ref_key(A), ref_key(B), ref_key(C)]
    assert found[ref_key(A)] is A
    assert ref_key(A) == ("objects", "set-a", "a.npy", A.sha256)


def test_distinct_refs_spans_models_in_order() -> None:
    """Over [Inner(B), Inner(C)] the keys are B then C."""
    found = distinct_refs([Inner(label="b", ref=B), Inner(label="c", ref=C)])
    assert list(found.values()) == [B, C]


def test_a_ref_itself_is_its_own_only_ref() -> None:
    """A root that is a ref yields itself and is not descended into."""
    assert list(iter_refs(A)) == [A]


def test_a_schema_record_with_no_refs_yields_nothing() -> None:
    """A real fitness experiment (deletion, YPD, fitness 0.9) holds no ``ArtifactRef``."""
    experiment = s.FitnessExperiment(
        dataset_name="toy",
        genotype=s.Genotype(
            perturbations=[
                s.KanMxDeletionPerturbation(
                    systematic_gene_name="YAL001C", perturbed_gene_name="YAL001C"
                )
            ]
        ),
        environment=s.Environment(
            media=s.Media(name="YPD", state="solid", is_synthetic=False)
        ),
        phenotype=s.FitnessPhenotype(fitness=0.9),
    )
    assert list(iter_refs(experiment)) == []
    assert distinct_refs([experiment]) == {}


def test_models_without_extra_allow_have_no_extras_to_walk() -> None:
    """``model_extra`` is None unless ``extra='allow'``; the walk skips it."""
    holder: Any = A
    assert holder.model_extra is None
    assert list(iter_refs(Inner(label="z", ref=None))) == []
