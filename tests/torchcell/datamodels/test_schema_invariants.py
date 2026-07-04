"""Schema-wide invariants (WS1 schema hardening).

Covers the structural guarantees the schema must uphold regardless of any single
model: Liskov substitutability across the ``*Type`` unions, ``*_TYPE_MAP``
registry completeness/consistency, the DRY ``validate_label_fields`` invariant
(label_name / label_statistic_name name real fields on each concrete phenotype),
and serialization round-trip.
"""

import typing

import pytest

from torchcell.datamodels.schema import (
    EXPERIMENT_REFERENCE_TYPE_MAP,
    EXPERIMENT_TYPE_MAP,
    Experiment,
    ExperimentReference,
    ExperimentReferenceType,
    ExperimentType,
    FitnessPhenotype,
    GeneEssentialityPhenotype,
    GeneInteractionPhenotype,
    Phenotype,
    PhenotypeType,
    SyntheticLethalityPhenotype,
    SyntheticRescuePhenotype,
)


def _members(union: object) -> tuple[type, ...]:
    """Concrete classes in a ``X | Y | ...`` union annotation."""
    return typing.get_args(union)


# --------------------------------------------------------------------------- #
# Liskov: every union member is a subclass of its declared base.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("member", _members(PhenotypeType))
def test_phenotype_union_members_subclass_base(member: type) -> None:
    assert issubclass(member, Phenotype)


@pytest.mark.parametrize("member", _members(ExperimentType))
def test_experiment_union_members_subclass_base(member: type) -> None:
    assert issubclass(member, Experiment)


@pytest.mark.parametrize("member", _members(ExperimentReferenceType))
def test_reference_union_members_subclass_base(member: type) -> None:
    assert issubclass(member, ExperimentReference)


# --------------------------------------------------------------------------- #
# Registry completeness + consistency.
# --------------------------------------------------------------------------- #
def test_experiment_maps_share_keys() -> None:
    """Experiment and reference maps must describe the same phenotype kinds."""
    assert set(EXPERIMENT_TYPE_MAP) == set(EXPERIMENT_REFERENCE_TYPE_MAP)


def test_experiment_map_covers_every_concrete_experiment() -> None:
    """Every concrete Experiment in the union (minus the base) is registered."""
    concrete = {m for m in _members(ExperimentType) if m is not Experiment}
    assert concrete == set(EXPERIMENT_TYPE_MAP.values())


def test_reference_map_covers_every_concrete_reference() -> None:
    concrete = {
        m for m in _members(ExperimentReferenceType) if m is not ExperimentReference
    }
    assert concrete == set(EXPERIMENT_REFERENCE_TYPE_MAP.values())


def test_map_values_are_correct_subclasses() -> None:
    for cls in EXPERIMENT_TYPE_MAP.values():
        assert issubclass(cls, Experiment)
    for cls in EXPERIMENT_REFERENCE_TYPE_MAP.values():
        assert issubclass(cls, ExperimentReference)


# --------------------------------------------------------------------------- #
# The DRY validate_label_fields invariant (now inherited from base Phenotype):
# each concrete phenotype's label_name / label_statistic_name defaults must name
# fields declared on that concrete class.
# --------------------------------------------------------------------------- #
_CONCRETE_PHENOTYPES = [m for m in _members(PhenotypeType) if m is not Phenotype]


@pytest.mark.parametrize("cls", _CONCRETE_PHENOTYPES)
def test_label_defaults_are_declared_fields(cls: type[Phenotype]) -> None:
    own_fields = cls.__annotations__
    label_name = cls.model_fields["label_name"].default
    assert label_name in own_fields, f"{cls.__name__}.label_name '{label_name}'"
    stat = cls.model_fields["label_statistic_name"].default
    if stat is not None:
        assert stat in own_fields, f"{cls.__name__}.label_statistic_name '{stat}'"


def test_inherited_validator_accepts_valid_phenotype() -> None:
    """A valid phenotype builds — proving the base validator does not over-reject."""
    ph = FitnessPhenotype(fitness=0.9)
    assert ph.label_name == "fitness"


def test_inherited_validator_rejects_unknown_label_name() -> None:
    """A label_name that names no field raises — proving the base validator fires
    through inheritance (there is no per-subclass copy anymore).
    """
    with pytest.raises(ValueError):
        FitnessPhenotype(fitness=0.9, label_name="not_a_real_field")


# --------------------------------------------------------------------------- #
# Serialization round-trip for constructible scalar phenotypes.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "obj",
    [
        FitnessPhenotype(fitness=0.87),
        GeneInteractionPhenotype(gene_interaction=0.1),
        GeneEssentialityPhenotype(),
        SyntheticLethalityPhenotype(),
        SyntheticRescuePhenotype(),
    ],
)
def test_phenotype_round_trip(obj: Phenotype) -> None:
    assert obj == type(obj)(**obj.model_dump())
