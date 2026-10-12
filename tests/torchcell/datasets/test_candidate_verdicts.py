# tests/torchcell/datasets/test_candidate_verdicts.py
# [[tests.torchcell.datasets.test_candidate_verdicts]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_candidate_verdicts.py
"""Every registered dataset class carries a passing candidate verdict, or is grandfathered.

The registry is filled by importing every module under ``torchcell/datasets`` (the
registration mechanism itself, not the adapter map, which excludes record-only modules).
``GRANDFATHERED`` is the set registered when the gate landed and may only shrink; a class
outside it needs a module-level ``CITATION_KEY`` and an ``admissible`` or
``admissible_with_gaps`` verdict in ``database/candidates/`` (plan decision 8 of
[[plan.dataset-admission-pipeline.2026.10.10]]).
"""

import importlib
import pkgutil

import torchcell.datasets as datasets_package
from torchcell.candidates.store import GRANDFATHERED, enforcement_violations, load_store
from torchcell.datasets.dataset_registry import dataset_registry

for _module in pkgutil.walk_packages(datasets_package.__path__, "torchcell.datasets."):
    importlib.import_module(_module.name)

REGISTERED = dict(sorted(dataset_registry.items()))


def test_grandfathered_is_sorted_unique_and_still_registered() -> None:
    """The tuple is sorted, has no duplicate, and names no class that left the registry."""
    assert list(GRANDFATHERED) == sorted(set(GRANDFATHERED))
    assert sorted(set(GRANDFATHERED) - set(REGISTERED)) == []


def test_every_class_outside_grandfathered_has_a_passing_verdict() -> None:
    """No registered class escapes: grandfathered, or keyed with an admissible verdict."""
    assert enforcement_violations(REGISTERED, GRANDFATHERED, load_store()) == []


def test_the_registry_walk_found_every_grandfathered_class() -> None:
    """The walk imports all 134 classes registered at main ecbd943ac (a drop is a regression)."""
    assert len(GRANDFATHERED) == 134
    assert len(REGISTERED) >= len(GRANDFATHERED)
