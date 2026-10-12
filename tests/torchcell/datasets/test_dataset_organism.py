# tests/torchcell/datasets/test_dataset_organism.py
# [[tests.torchcell.datasets.test_dataset_organism]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_dataset_organism.py
"""Every dataset class declares its host organism, and the declaration matches its package.

The knowledge-graph build selects datasets by ``ExperimentDataset.organism`` (the
``--organisms`` switch, ``KG_ORGANISMS`` in the live rebuild), so a missing or wrong
declaration would build a graph other than the one asked for. The declaration is read
off the CLASS, and this module holds it to the package the loader lives in:
``torchcell.datasets.<organism>``, with the in-house ``private_torchcell`` package being
yeast. A mismatch is a test failure, never a run-time correction.
"""

from __future__ import annotations

from typing import get_args

import pytest

import torchcell.datasets.ecoli  # noqa: F401  # populates the registry
import torchcell.datasets.pputida  # noqa: F401  # populates the registry
import torchcell.datasets.private_torchcell  # noqa: F401  # populates the registry
import torchcell.datasets.scerevisiae  # noqa: F401  # populates the registry
from torchcell.data.experiment_dataset import (
    ALL_ORGANISMS,
    ExperimentDataset,
    Organism,
    OrganismUndeclared,
    dataset_organism,
    format_organisms,
    ordered_organisms,
    parse_organisms,
)
from torchcell.datasets.dataset_registry import dataset_registry
from torchcell.knowledge_graphs.dataset_adapter_map import build_adapter_map
from torchcell.knowledge_graphs.kg_manifest import OrganismName

#: The package under ``torchcell.datasets`` -> the organism its loaders must declare.
PACKAGE_ORGANISM = {
    "scerevisiae": Organism.scerevisiae,
    "ecoli": Organism.ecoli,
    "pputida": Organism.pputida,
    "private_torchcell": Organism.scerevisiae,
}

MAPPED = sorted(build_adapter_map(include_private=True), key=lambda c: c.__name__)
REGISTERED = sorted(
    (
        cls
        for cls in dataset_registry.values()
        if isinstance(cls, type) and issubclass(cls, ExperimentDataset)
    ),
    key=lambda c: c.__name__,
)


def _package(cls: type) -> str:
    parts = cls.__module__.split(".")
    assert parts[:2] == ["torchcell", "datasets"], cls.__module__
    return parts[2]


@pytest.mark.parametrize("cls", MAPPED, ids=lambda c: c.__name__)
def test_every_mapped_class_declares_the_organism_of_its_package(cls: type) -> None:
    """Declared on the class itself (not inherited), and equal to its package's."""
    assert "organism" in vars(cls), f"{cls.__name__} declares no organism of its own"
    assert dataset_organism(cls) is PACKAGE_ORGANISM[_package(cls)]


@pytest.mark.parametrize("cls", REGISTERED, ids=lambda c: c.__name__)
def test_every_registered_experiment_dataset_declares_one(cls: type) -> None:
    """A loader registered outside the maps (YeastPhenome) is held to the same rule."""
    assert dataset_organism(cls) is PACKAGE_ORGANISM[_package(cls)]


def test_the_registry_and_the_maps_hold_the_same_dataset_classes_but_one() -> None:
    """The two sweeps above cover each other: the registry adds only YeastPhenome."""
    assert sorted(c.__name__ for c in set(REGISTERED) - set(MAPPED)) == [
        "YeastPhenomeDataset"
    ]
    assert set(MAPPED) <= set(REGISTERED)


def test_the_base_class_annotates_organism_without_a_value() -> None:
    """No default to inherit: a loader that forgot it is refused, not filed somewhere."""
    assert "organism" in ExperimentDataset.__annotations__
    assert "organism" not in vars(ExperimentDataset)
    with pytest.raises(
        OrganismUndeclared, match="ExperimentDataset declares no organism"
    ):
        dataset_organism(ExperimentDataset)


def test_dataset_organism_refuses_a_class_without_an_organism_value() -> None:
    class _NoOrganism:
        pass

    class _StringOrganism:
        organism = "scerevisiae"

    for cls in (_NoOrganism, _StringOrganism):
        with pytest.raises(
            OrganismUndeclared,
            match=rf"^{cls.__name__} declares no organism; set `organism = "
            r"Organism.<scerevisiae\|ecoli\|pputida>` on the class$",
        ):
            dataset_organism(cls)


def test_the_organism_maps_partition_the_full_map() -> None:
    """Each organism's map is disjoint from the others and together they are the map."""
    for include_private in (False, True):
        full = build_adapter_map(include_private)
        parts = [
            set(build_adapter_map(include_private, frozenset({organism})))
            for organism in Organism
        ]
        assert sum(len(part) for part in parts) == len(full)
        assert set().union(*parts) == set(full)


def test_the_enum_order_and_the_manifest_literal_agree() -> None:
    """Yeast first, then the bacteria; the manifest's string Literal has the same members."""
    assert [o.value for o in Organism] == ["scerevisiae", "ecoli", "pputida"]
    assert list(get_args(OrganismName)) == [o.value for o in Organism]
    assert frozenset(Organism) == ALL_ORGANISMS


def test_parse_and_format_round_trip_in_enum_order() -> None:
    assert parse_organisms("scerevisiae") == frozenset({Organism.scerevisiae})
    assert parse_organisms(" pputida , ecoli ") == frozenset(
        {Organism.ecoli, Organism.pputida}
    )
    assert format_organisms(parse_organisms("pputida,scerevisiae")) == (
        "scerevisiae,pputida"
    )
    assert format_organisms(ALL_ORGANISMS) == "scerevisiae,ecoli,pputida"
    assert ordered_organisms([Organism.pputida, Organism.ecoli, Organism.pputida]) == [
        Organism.ecoli,
        Organism.pputida,
    ]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", r"^empty organism name in ''$"),
        ("scerevisiae,", r"^empty organism name in 'scerevisiae,'$"),
        (
            "yeast,ecoli,human",
            r"^unknown organism\(s\) human, yeast in 'yeast,ecoli,human'; known: "
            r"scerevisiae,ecoli,pputida$",
        ),
        ("ecoli,ecoli", r"^organism\(s\) ecoli given twice in 'ecoli,ecoli'$"),
    ],
)
def test_parse_organisms_refuses_by_name(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_organisms(text)
