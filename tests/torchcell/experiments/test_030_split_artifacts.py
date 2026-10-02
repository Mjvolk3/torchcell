# tests/torchcell/experiments/test_030_split_artifacts.py
"""Pure functions of the 030 split and holdout scripts, on hand-built inputs.

The scripts live under ``experiments/`` (not a package), so they are loaded by file
path. Nothing here touches the build, the closure cache, or the released CSV.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

SCRIPTS = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "030-solid-growth-multi"
    / "scripts"
)


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def transfer() -> ModuleType:
    return load("transfer_010_tmi_splits_030")


@pytest.fixture(scope="module")
def holdout() -> ModuleType:
    return load("build_essentiality_holdout_030")


@pytest.fixture(scope="module")
def subset() -> ModuleType:
    return load("subset_definitions_030")


def _record(*genes: str) -> list[dict[str, Any]]:
    perts = [
        {"systematic_gene_name": g, "perturbation_type": "deletion"} for g in genes
    ]
    return [{"experiment": {"genotype": {"perturbations": perts}}}]


def test_record_identity_sorts_and_dedups(transfer: ModuleType) -> None:
    assert transfer.record_identity(_record("YB", "YA", "YB")) == ("YA", "YB")
    assert transfer.identity_of_gene_string("YB|YA") == ("YA", "YB")
    assert transfer.record_identity(_record("YC", "YA", "YB")) == (
        transfer.identity_of_gene_string("YA|YB|YC")
    )


def test_join_by_identity_one_to_one(transfer: ModuleType) -> None:
    source = {("A", "B", "C"): "train", ("A", "B", "D"): "val", ("B", "C", "D"): "test"}
    target = {10: ("A", "B", "D"), 11: ("B", "C", "D"), 12: ("A", "B", "C")}
    pinned, diag = transfer.join_by_identity(source, target)
    assert pinned == {"train": [12], "val": [10], "test": [11]}
    assert diag == {"unmatched": 0, "multi_matched": 0, "unclaimed": 0}


def test_join_by_identity_reports_defects(transfer: ModuleType) -> None:
    source = {("A", "B", "C"): "train", ("X", "Y", "Z"): "val"}
    target = {1: ("A", "B", "C"), 2: ("A", "B", "C"), 3: ("Q", "R", "S")}
    pinned, diag = transfer.join_by_identity(source, target)
    assert pinned == {"train": [], "val": [], "test": []}
    assert diag == {"unmatched": 1, "multi_matched": 1, "unclaimed": 3}


def test_closure_pairs(subset: ModuleType) -> None:
    pairs = subset.closure_pairs(["A|B|C", "A|B|D"])
    assert pairs == {
        frozenset(p)
        for p in (("A", "B"), ("A", "C"), ("B", "C"), ("A", "D"), ("B", "D"))
    }


def test_released_label_is_inverted(holdout: ModuleType) -> None:
    assert holdout.released_label("0.0") == 1
    assert holdout.released_label("1.0") == 0
    assert holdout.released_label("1") == 0


def test_auroc_extremes(holdout: ModuleType) -> None:
    assert holdout.auroc([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert holdout.auroc([1.0, 2.0], [3.0, 4.0]) == 0.0
    assert holdout.auroc([1.0, 1.0], [1.0, 1.0]) == 0.5


def test_greedy_match_prefers_exact_and_uses_each_control_once(
    holdout: ModuleType,
) -> None:
    covariate = {
        "case1": (5.0, 2.0),
        "case2": (5.0, 2.0),
        "c_exact": (5.0, 2.0),
        "c_near": (6.0, 2.0),
        "c_far": (50.0, 20.0),
    }
    pairs = holdout.greedy_match(
        ["case1", "case2"],
        ["c_far", "c_near", "c_exact"],
        covariate,
        np.random.default_rng(0),
    )
    assert [p[1] for p in pairs] == ["c_exact", "c_near"]
    assert [p[2] for p in pairs] == [0.0, 1.0]


def test_greedy_match_needs_enough_controls(holdout: ModuleType) -> None:
    with pytest.raises(AssertionError):
        holdout.greedy_match(
            ["a", "b"],
            ["c"],
            {"a": (0.0,), "b": (0.0,), "c": (0.0,)},
            np.random.default_rng(0),
        )


def test_gene_coverage_counts_membership(holdout: ModuleType) -> None:
    import pandas as pd

    counts = holdout.gene_coverage(pd.Series(["A|B", "A|C", "B|C|D"]))
    assert counts == {"A": 2, "B": 2, "C": 2, "D": 1}
