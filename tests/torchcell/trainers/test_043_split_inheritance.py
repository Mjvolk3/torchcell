# tests/torchcell/trainers/test_043_split_inheritance.py
"""Split inheritance (owner, 2026-10-23): a new split version keeps every fold of its
predecessor and draws only the keys the predecessor did not have.

``deal_new`` is tested on hand-built strata, and the committed
``conf/splits/global_split_v2.json`` is checked against ``global_split_v1.json``: every v1
key of every key space keeps its v1 fold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXP = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "043-expression-proteome-morphology"
)
if str(EXP / "scripts") not in sys.path:
    sys.path.insert(0, str(EXP / "scripts"))

from global_split_043 import (  # type: ignore[import-not-found]  # noqa: E402
    GlobalSplit043,
)
from make_global_split_043 import (  # type: ignore[import-not-found]  # noqa: E402
    deal,
    deal_new,
)


def test_deal_new_keeps_previous_folds_and_draws_only_new_keys() -> None:
    strata = {"a": ["g1", "g2", "g3", "g9"], "b": ["g4", "g5", "g8"]}
    previous = {"g1": 3, "g2": 0, "g3": 7, "g4": 1, "g5": 5}
    out = deal_new(strata, list(range(10)), np.random.default_rng(0), previous)
    assert set(out) == {"g1", "g2", "g3", "g4", "g5", "g8", "g9"}
    assert all(out[k] == f for k, f in previous.items())
    # the new keys are dealt exactly as ``deal`` deals the new-keys-only strata
    expected = deal(
        {"a": ["g9"], "b": ["g8"]}, list(range(10)), np.random.default_rng(0)
    )
    assert {k: out[k] for k in ("g8", "g9")} == expected


def test_deal_new_respects_the_allowed_folds_for_pinned_keys() -> None:
    strata = {"p": [f"n{i}" for i in range(20)]}
    out = deal_new(strata, list(range(2, 10)), np.random.default_rng(0), {})
    assert set(out.values()) <= set(range(2, 10))


def test_committed_v2_inherits_every_v1_fold() -> None:
    splits = EXP / "conf" / "splits"
    v1 = GlobalSplit043.model_validate_json(
        (splits / "global_split_v1.json").read_text()
    )
    v2 = GlobalSplit043.model_validate_json(
        (splits / "global_split_v2.json").read_text()
    )
    assert (v2.mode, v2.inherits_from, v2.seed) == ("inherit", "global_split_v1", 0)
    for space in ("gene_folds", "isolate_folds", "segregant_folds", "promoter_folds"):
        before, now = getattr(v1, space), getattr(v2, space)
        assert set(before) <= set(now), space
        assert all(now[k] == f for k, f in before.items()), space
    assert v2.pinned_training_genes == v1.pinned_training_genes
    assert all(v2.gene_folds[g] >= 2 for g in v2.pinned_training_genes)
