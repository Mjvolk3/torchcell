# tests/torchcell/knowledge_graphs/test_dataset_organism_gate.py
# [[tests.torchcell.knowledge_graphs.test_dataset_organism_gate]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/knowledge_graphs/test_dataset_organism_gate.py
"""The build-time organism switch: a build serves exactly the organisms it was asked for.

The twin of ``test_dataset_visibility.py``. ``--organisms`` is taken off ``argv`` by
``take_organisms_flag`` (hydra never sees it), ``build_adapter_map`` keeps the mapped
classes whose declared ``organism`` is in the set, and ``refuse_organism_mismatch``
names a configured class outside the set instead of dropping it.
"""

from __future__ import annotations

import pytest

from torchcell.data.experiment_dataset import (
    ALL_ORGANISMS,
    Organism,
    OrganismUndeclared,
    Visibility,
    dataset_organism,
)
from torchcell.knowledge_graphs.dataset_adapter_map import (
    ORGANISMS_FLAG,
    PRIVATE_DATASET_ADAPTER_MAP,
    OrganismRefused,
    build_adapter_map,
    dataset_adapter_map,
    refuse_organism_mismatch,
    select_organisms,
    take_organisms_flag,
)

YEAST = frozenset({Organism.scerevisiae})
BACTERIA = frozenset({Organism.ecoli, Organism.pputida})


class _YeastLoader:
    organism = Organism.scerevisiae


class _EcoliLoader:
    organism = Organism.ecoli


class _PutidaLoader:
    organism = Organism.pputida


class _UndeclaredLoader:
    """A mapped class that forgot ``organism``."""


class _Adapter:
    pass


def test_take_organisms_flag_removes_it_and_returns_the_set() -> None:
    """Both spellings are taken off argv; everything else stays in order."""
    argv = ["prog", "--config-name", "kg_uncapped", "--organisms", "ecoli,scerevisiae"]
    assert take_organisms_flag(argv) == frozenset(
        {Organism.scerevisiae, Organism.ecoli}
    )
    assert argv == ["prog", "--config-name", "kg_uncapped"]
    argv = ["prog", "--organisms=pputida", "--include-private"]
    assert take_organisms_flag(argv) == frozenset({Organism.pputida})
    assert argv == ["prog", "--include-private"]


def test_without_the_flag_a_build_takes_every_organism() -> None:
    """Absent is every organism, the build as it was before the switch existed."""
    argv = ["prog", "--config-name", "kg_uncapped"]
    assert take_organisms_flag(argv) is ALL_ORGANISMS
    assert argv == ["prog", "--config-name", "kg_uncapped"]
    assert ORGANISMS_FLAG == "--organisms"


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["prog", "--organisms"], r"^--organisms needs a value, e.g. scerevisiae$"),
        (
            ["prog", "--organisms", "ecoli", "--organisms=pputida"],
            r"^--organisms given 2 times \(ecoli, pputida\); pass one comma-separated "
            r"list$",
        ),
        (
            ["prog", "--organisms", "--include-private"],
            r"^unknown organism\(s\) --include-private in '--include-private'",
        ),
    ],
)
def test_a_malformed_flag_stops_the_build(argv: list[str], message: str) -> None:
    """A missing value, a repeat, or an unknown name each build another graph."""
    with pytest.raises(ValueError, match=message):
        take_organisms_flag(argv)


def test_select_organisms_keeps_the_declared_organisms_in_map_order() -> None:
    adapter_map: dict[type, type] = {
        _EcoliLoader: _Adapter,
        _YeastLoader: _Adapter,
        _PutidaLoader: _Adapter,
    }
    assert list(select_organisms(adapter_map, YEAST)) == [_YeastLoader]
    assert list(select_organisms(adapter_map, BACTERIA)) == [
        _EcoliLoader,
        _PutidaLoader,
    ]
    assert list(select_organisms(adapter_map, ALL_ORGANISMS)) == list(adapter_map)


def test_an_undeclared_mapped_class_breaks_every_selection() -> None:
    """Even the full set reads each class, so a missing declaration is never kept by
    default: the build stops on it, naming it.
    """
    for organisms in (YEAST, ALL_ORGANISMS):
        with pytest.raises(OrganismUndeclared, match="^_UndeclaredLoader declares"):
            select_organisms(
                {_YeastLoader: _Adapter, _UndeclaredLoader: _Adapter}, organisms
            )


def test_build_adapter_map_narrows_the_public_and_the_private_map() -> None:
    """The private map joins only under include_private, and the organism filter
    applies to both maps alike.
    """
    yeast_public = build_adapter_map(organisms=YEAST)
    assert set(yeast_public) == {
        cls
        for cls in dataset_adapter_map
        if dataset_organism(cls) is Organism.scerevisiae
    }
    assert build_adapter_map() == dict(dataset_adapter_map)
    PRIVATE_DATASET_ADAPTER_MAP[_EcoliLoader] = _Adapter
    try:
        assert _EcoliLoader not in build_adapter_map(True, YEAST)
        assert build_adapter_map(True, frozenset({Organism.ecoli}))[_EcoliLoader] is (
            _Adapter
        )
        assert _EcoliLoader not in build_adapter_map(False, frozenset({Organism.ecoli}))
    finally:
        del PRIVATE_DATASET_ADAPTER_MAP[_EcoliLoader]


def test_the_private_bioscreen_dataset_is_yeast() -> None:
    """The in-house loaders are yeast, so a yeast-only in-house build still has them."""
    from torchcell.datasets.private_torchcell.volk2021_inhibitor_bioscreen import (
        InhibitorBioscreenVolk2021Dataset,
    )

    assert InhibitorBioscreenVolk2021Dataset.visibility is Visibility.private
    assert InhibitorBioscreenVolk2021Dataset in build_adapter_map(True, YEAST)
    assert InhibitorBioscreenVolk2021Dataset not in build_adapter_map(True, BACTERIA)


def test_refuse_organism_mismatch_names_every_class_outside_the_set() -> None:
    """Sorted, each with its organism, plus the set and the flag that would admit it."""
    refuse_organism_mismatch([_YeastLoader, _EcoliLoader, _PutidaLoader])
    refuse_organism_mismatch([_YeastLoader], YEAST)
    with pytest.raises(OrganismRefused) as excinfo:
        refuse_organism_mismatch([_YeastLoader, _PutidaLoader, _EcoliLoader], YEAST)
    assert str(excinfo.value) == (
        "these datasets are outside the build's organism set (scerevisiae): "
        "_EcoliLoader (ecoli), _PutidaLoader (pputida). Add their organism to "
        "--organisms (KG_ORGANISMS in the slurm builds), or leave them out of the "
        "config."
    )


def test_every_build_entry_point_reads_the_flag_through_the_same_function() -> None:
    from torchcell.knowledge_graphs import create_kg

    assert create_kg.take_organisms_flag is take_organisms_flag
