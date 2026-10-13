# tests/torchcell/datamodels/test_liskov_fields.py
# [[tests.torchcell.datamodels.test_liskov_fields]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datamodels/test_liskov_fields.py
"""Liskov substitutability at the FIELD level, for every model in the schema.

``test_schema_invariants`` and ``test_ontology_invariants`` check the nominal half of
S4: each union member ``issubclass`` its root. That is satisfied by any subclass, even
one that redeclares a parent field with an incompatible type. The behavioral half is
what makes a leaf usable wherever its parent is: every field a child redeclares must
hold a SUBTYPE of what the parent holds, so a reader written against the parent
reads something it understands. The models are frozen (``ModelStrict``), so covariant
narrowing is sound and is the only permitted change:

* a parent field typed as a model may be narrowed to a subclass of that model
  (``Experiment.phenotype: Phenotype`` -> ``FitnessExperiment.phenotype: FitnessPhenotype``);
* ``X | None`` may drop the ``None`` (``Phenotype.label_statistic_name: str | None`` ->
  ``str``);
* ``str`` may be narrowed to a ``Literal``, and a ``Literal`` to a subset of its values;
* a union may shrink to a subset of its members; ``list[A]`` may become ``list[B]`` for
  ``B`` a subtype of ``A``.

Everything else is a violation: widening (``Genotype`` -> ``Genotype | list[Genotype]``,
removed 2026-10-07), a Literal replaced by a different Literal (the abstract
``DeletionPerturbation`` used to pin ``"deletion"`` while its leaves pinned their own
tags; its annotation is now ``str``), or a sibling type that is no subclass at all.

``KNOWN_VIOLATIONS`` is the grandfather list. It may only shrink: an entry must still be
a violation (so a fixed one has to be deleted here), and no violation outside it may
appear. The seven entries today, each with its planned fix:

* ``SegregantGrowthExperiment.genotype`` is typed ``SegregantGenotype``, a sibling of
  ``Genotype`` rather than a subclass (a segregant is a haplotype mosaic, not a list of
  perturbations). Fix: a shared abstract ``GenotypeBase`` root for both, with
  ``Experiment.genotype: GenotypeBase``; the graph schema already says ``segregant
  genotype is_a genotype``.
* ``BarcodedKanMxDeletionPerturbation``, ``SgaKanMxDeletionPerturbation``,
  ``SgaNatMxDeletionPerturbation``, ``BacterialCrisprInterferencePerturbation`` and
  ``HeterologousPathwayPerturbation`` each pin a tag different from the one their parent
  pins, and the parent is itself a concrete union leaf (``KanMxDeletionPerturbation``,
  ``NatMxDeletionPerturbation``, ``CrisprInterferencePerturbation``,
  ``GeneAdditionPerturbation``). A leaf cannot be the parent of another leaf without
  breaking the tag field. Fix: an abstract base per family (tag annotated ``str``,
  holding the shared fields) with the current leaves as siblings under it, the shape
  ``DeletionPerturbation`` now has.
* ``PromoterReplacementPerturbation.crispr`` widens the abstract
  ``ExpressionModulationPerturbation.crispr: CrisprConstruct`` to ``CrisprConstruct |
  None``, because a promoter replacement has no guide construct. Fix: the construct
  belongs to the CRISPR leaves, not to the expression-modulation axis base; move it
  down, or annotate the base ``CrisprConstruct | None``.

Every entry changes the contract fingerprint of a class inside served dataset closures,
so each fix lands with a full knowledge-graph rebuild (``torchcell.provenance.schema_deps``
hashes a class's own fields and bases; the records themselves serialize unchanged).
"""

from __future__ import annotations

import types
import typing
from typing import Any, Literal

from pydantic import BaseModel

import torchcell.datamodels.schema as schema

KNOWN_VIOLATIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("SegregantGrowthExperiment", "genotype"),
        ("BarcodedKanMxDeletionPerturbation", "perturbation_type"),
        ("SgaKanMxDeletionPerturbation", "perturbation_type"),
        ("SgaNatMxDeletionPerturbation", "perturbation_type"),
        ("BacterialCrisprInterferencePerturbation", "perturbation_type"),
        ("HeterologousPathwayPerturbation", "perturbation_type"),
        ("PromoterReplacementPerturbation", "crispr"),
    }
)


def _models() -> list[type[BaseModel]]:
    out = [
        obj
        for obj in vars(schema).values()
        if isinstance(obj, type)
        and issubclass(obj, BaseModel)
        and obj.__module__ == schema.__name__
    ]
    return sorted(out, key=lambda c: c.__name__)


def _model_parents(cls: type[BaseModel]) -> list[type[BaseModel]]:
    return [
        b
        for b in cls.__bases__
        if isinstance(b, type) and issubclass(b, BaseModel) and b is not BaseModel
    ]


def _union_args(ann: Any) -> tuple[Any, ...] | None:
    origin = typing.get_origin(ann)
    if origin is typing.Union or origin is types.UnionType:
        return typing.get_args(ann)
    return None


def is_subtype(child: Any, parent: Any) -> bool:
    """``child`` may be read wherever ``parent`` is expected (covariant reads)."""
    if child == parent or parent is Any:
        return True
    if child is Any:
        return False
    child_union, parent_union = _union_args(child), _union_args(parent)
    if child_union is not None:
        return all(is_subtype(c, parent) for c in child_union)
    if parent_union is not None:
        return any(is_subtype(child, p) for p in parent_union)
    child_origin, parent_origin = typing.get_origin(child), typing.get_origin(parent)
    if child_origin is Literal:
        values = typing.get_args(child)
        if parent_origin is Literal:
            return set(values) <= set(typing.get_args(parent))
        return all(is_subtype(type(v), parent) for v in values)
    if parent_origin is Literal:
        return False
    if child_origin is not None or parent_origin is not None:
        if child_origin is not parent_origin:
            return False
        c_args, p_args = typing.get_args(child), typing.get_args(parent)
        return len(c_args) == len(p_args) and all(
            is_subtype(c, p) for c, p in zip(c_args, p_args, strict=True)
        )
    if child is type(None) or parent is type(None):
        return child is parent
    return (
        isinstance(child, type)
        and isinstance(parent, type)
        and issubclass(child, parent)
    )


def _violations() -> dict[tuple[str, str], str]:
    found: dict[tuple[str, str], str] = {}
    for cls in _models():
        own = cls.__annotations__ if "__annotations__" in vars(cls) else {}
        for parent in _model_parents(cls):
            for name in own:
                if name not in parent.model_fields:
                    continue
                child_t = cls.model_fields[name].annotation
                parent_t = parent.model_fields[name].annotation
                if not is_subtype(child_t, parent_t):
                    found[(cls.__name__, name)] = (
                        f"{parent.__name__}.{name}: {parent_t!r} -> "
                        f"{cls.__name__}.{name}: {child_t!r}"
                    )
    return found


def test_is_subtype_relation_on_the_forms_the_schema_uses() -> None:
    """The relation the sweep relies on, pinned on exact cases: narrowing passes, the
    three historical violations fail.
    """
    assert is_subtype(schema.FitnessPhenotype, schema.Phenotype)
    assert not is_subtype(schema.Phenotype, schema.FitnessPhenotype)
    assert is_subtype(str, str | None)
    assert not is_subtype(str | None, str)
    assert is_subtype(Literal["kanmx_deletion"], str)
    assert is_subtype(Literal["a"], Literal["a", "b"])
    assert not is_subtype(Literal["kanmx_deletion"], Literal["deletion"])
    assert not is_subtype(str, Literal["deletion"])
    assert not is_subtype(schema.Genotype | list[schema.Genotype], schema.Genotype)
    assert is_subtype(list[schema.FitnessPhenotype], list[schema.Phenotype])
    assert not is_subtype(list[schema.Phenotype], list[schema.FitnessPhenotype])
    assert not is_subtype(schema.SegregantGenotype, schema.Genotype)
    assert is_subtype(schema.Genotype, schema.Genotype | schema.SegregantGenotype)


def test_every_redeclared_field_is_a_subtype_of_the_parents() -> None:
    """No field override outside ``KNOWN_VIOLATIONS`` widens or sidesteps its parent."""
    found = _violations()
    new = {k: v for k, v in found.items() if k not in KNOWN_VIOLATIONS}
    assert not new, "field overrides that are not subtypes:\n" + "\n".join(
        sorted(new.values())
    )


def test_known_violations_are_still_violations() -> None:
    """A grandfathered entry that no longer violates must be removed from the list, so
    the list only shrinks and never shelters a fixed case.
    """
    found = _violations()
    stale = KNOWN_VIOLATIONS - set(found)
    assert not stale, f"fixed; delete from KNOWN_VIOLATIONS: {sorted(stale)}"
    assert found[("SegregantGrowthExperiment", "genotype")].endswith(
        "SegregantGrowthExperiment.genotype: <class "
        "'torchcell.datamodels.schema.SegregantGenotype'>"
    )
    assert found[("SgaKanMxDeletionPerturbation", "perturbation_type")] == (
        "KanMxDeletionPerturbation.perturbation_type: typing.Literal['kanmx_deletion']"
        " -> SgaKanMxDeletionPerturbation.perturbation_type: "
        "typing.Literal['sga_kanmx_deletion']"
    )


def test_the_sweep_covers_the_experiment_and_phenotype_families() -> None:
    """The sweep saw the overrides that matter: every concrete experiment narrows
    ``phenotype``, and no experiment redeclares ``genotype`` except the segregant one.
    """
    experiments = [
        c
        for c in _models()
        if issubclass(c, schema.Experiment) and c is not schema.Experiment
    ]
    # The union carries the root itself (grandfathered, see S7 in the enforcement note).
    assert set(experiments) == set(typing.get_args(schema.ExperimentType)) - {
        schema.Experiment
    }
    for cls in experiments:
        assert cls.model_fields["phenotype"].annotation is not schema.Phenotype
        assert is_subtype(cls.model_fields["phenotype"].annotation, schema.Phenotype)
        own = cls.__annotations__
        if cls is schema.SegregantGrowthExperiment:
            assert "genotype" in own
        else:
            assert "genotype" not in own
            assert cls.model_fields["genotype"].annotation is schema.Genotype
