# tests/torchcell/datamodels/test_interned_constant.py
# [[tests.torchcell.datamodels.test_interned_constant]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datamodels/test_interned_constant.py
"""The interned-constant contract, tested on the module's pure functions.

``split_experiment_dump`` replaces ``environment`` with a pointer
``{"$ref": sha256(json.dumps(value)), "kind": "environment"}`` when its JSON is at
least 512 bytes, and ``genotype`` likewise at 8192 bytes; every other field stays
inline and the input dict is not mutated. ``resolve_pointers`` splices the constants
back, so ``resolve(split(dump)) == dump``; a pointer to an absent id raises
``KeyError``. ``verified_constant`` parses a payload only when it hashes to the id it
was fetched by, and raises ``ValueError`` otherwise.

Hand-derived values: the payloads are built so their sizes are exact.
``json.dumps({"name": "x" * k})`` is ``12 + k`` bytes (``{"name": ""}`` is 12), so
``k = 500`` sits exactly on the environment floor and ``k = 499`` one byte below it.
``json.dumps({"blocks": "y" * k})`` is ``14 + k`` bytes, so ``k = 8178`` sits on the
genotype floor and ``k = 8177`` one byte below. The two pinned digests are the
standard sha256 of the empty string and of ``{}``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest

from torchcell.datamodels.interned_constant import (
    EXPERIMENT_POINTER_MIN_BYTES,
    INTERNED_CONSTANT_LABEL,
    INTERNED_CONSTANT_NEO4J_LABEL,
    POINTER_KEY,
    collect_pointers,
    constant_id,
    resolve_pointers,
    split_experiment_dump,
    verified_constant,
)

_SHA_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
_SHA_BRACES = "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _environment(k: int) -> dict[str, Any]:
    return {"name": "x" * k}


def _genotype(k: int) -> dict[str, Any]:
    return {"blocks": "y" * k}


def _dump(env_k: int, geno_k: int) -> dict[str, Any]:
    return {
        "experiment_type": "fitness",
        "environment": _environment(env_k),
        "genotype": _genotype(geno_k),
        "phenotype": {"fitness": 0.5, "std": [0.1, 0.2]},
    }


def test_labels_key_and_thresholds() -> None:
    assert INTERNED_CONSTANT_LABEL == "interned constant"
    assert INTERNED_CONSTANT_NEO4J_LABEL == "InternedConstant"
    assert POINTER_KEY == "$ref"
    assert EXPERIMENT_POINTER_MIN_BYTES == {"environment": 512, "genotype": 8192}


def test_constant_id_is_sha256_of_utf8_payload() -> None:
    assert constant_id("") == _SHA_EMPTY
    assert constant_id("{}") == _SHA_BRACES
    assert constant_id('{"name": "é"}') == _sha('{"name": "é"}')


def test_payload_sizes_are_as_derived() -> None:
    assert len(json.dumps(_environment(500))) == 512
    assert len(json.dumps(_environment(499))) == 511
    assert len(json.dumps(_genotype(8178))) == 8192
    assert len(json.dumps(_genotype(8177))) == 8191


def test_environment_at_floor_becomes_pointer_and_round_trips() -> None:
    dump = _dump(env_k=500, geno_k=8177)
    original = copy.deepcopy(dump)
    out, constants = split_experiment_dump(dump)

    env_payload = json.dumps(_environment(500))
    env_id = _sha(env_payload)
    assert constants == [(env_id, "environment", env_payload)]
    assert out == {
        "experiment_type": "fitness",
        "environment": {"$ref": env_id, "kind": "environment"},
        "genotype": _genotype(8177),
        "phenotype": {"fitness": 0.5, "std": [0.1, 0.2]},
    }
    assert dump == original

    refs: set[str] = set()
    collect_pointers(out, refs)
    assert refs == {env_id}
    resolved = resolve_pointers(out, {env_id: json.loads(env_payload)})
    assert resolved == original
    assert json.dumps(resolved) == json.dumps(original)


def test_below_both_floors_nothing_is_interned() -> None:
    dump = _dump(env_k=499, geno_k=8177)
    out, constants = split_experiment_dump(dump)
    assert constants == []
    assert out == dump
    assert out is not dump

    refs: set[str] = set()
    collect_pointers(out, refs)
    assert refs == set()
    assert resolve_pointers(out, {}) == dump


def test_genotype_at_floor_becomes_pointer_and_round_trips() -> None:
    dump = _dump(env_k=500, geno_k=8178)
    original = copy.deepcopy(dump)
    out, constants = split_experiment_dump(dump)

    env_payload = json.dumps(_environment(500))
    geno_payload = json.dumps(_genotype(8178))
    env_id, geno_id = _sha(env_payload), _sha(geno_payload)
    assert constants == [
        (env_id, "environment", env_payload),
        (geno_id, "genotype", geno_payload),
    ]
    assert out["environment"] == {"$ref": env_id, "kind": "environment"}
    assert out["genotype"] == {"$ref": geno_id, "kind": "genotype"}

    refs: set[str] = set()
    collect_pointers(out, refs)
    assert refs == {env_id, geno_id}
    store = {ref: verified_constant(ref, payload) for ref, _, payload in constants}
    assert resolve_pointers(out, store) == original


def test_collect_and_resolve_walk_nested_lists_and_dicts() -> None:
    obj: dict[str, Any] = {
        "a": [{"$ref": "r1", "kind": "environment"}, 3, "s"],
        "b": {"c": {"$ref": "r2", "kind": "genotype"}, "d": None},
    }
    refs: set[str] = {"r0"}
    collect_pointers(obj, refs)
    assert refs == {"r0", "r1", "r2"}
    assert resolve_pointers(obj, {"r1": {"e": 1}, "r2": [7]}) == {
        "a": [{"e": 1}, 3, "s"],
        "b": {"c": [7], "d": None},
    }


def test_resolve_missing_id_raises_key_error() -> None:
    with pytest.raises(KeyError, match="absent"):
        resolve_pointers({"x": {"$ref": "absent", "kind": "environment"}}, {})


def test_verified_constant_accepts_matching_and_rejects_mismatch() -> None:
    assert verified_constant(_SHA_BRACES, "{}") == {}
    with pytest.raises(
        ValueError,
        match=f"interned constant {_SHA_EMPTY} holds a payload hashing to "
        f"{_SHA_BRACES}; the store is corrupt",
    ):
        verified_constant(_SHA_EMPTY, "{}")
