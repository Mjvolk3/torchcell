"""Tests for content-addressed interning of constant sub-objects (ExperimentDataset).

Verifies the interning contract end-to-end at the LMDB level: a heavy constant
Environment is stored ONCE in the sibling ``interned`` env and resolved back to
the full object on read; a legacy record with no ``$ref`` is an exact no-op.
"""

import json
import pickle
from pathlib import Path

import lmdb

from torchcell.data import compute_sha256_hash
from torchcell.data.experiment_dataset import (
    INTERN_MIN_BYTES,
    canonical_json,
    resolve_interned,
)
from torchcell.datamodels.media import SGA_DM_SELECTION
from torchcell.datamodels.schema import Environment, Temperature


def _sga_environment() -> Environment:
    return Environment(media=SGA_DM_SELECTION, temperature=Temperature(value=26))


def test_canonical_json_deterministic_and_mode_json() -> None:
    """canonical_json is stable and equals a mode=json + sorted dump."""
    env = _sga_environment()
    assert canonical_json(env) == canonical_json(_sga_environment())
    assert canonical_json(env) == json.dumps(
        env.model_dump(mode="json"), sort_keys=True
    )


def test_environment_exceeds_intern_threshold() -> None:
    """The component-based SGA Environment is heavy enough to be interned."""
    assert len(canonical_json(_sga_environment())) >= INTERN_MIN_BYTES


def test_resolve_interned_legacy_noop() -> None:
    """A record with no $ref anywhere resolves byte-for-byte unchanged."""
    legacy = {
        "experiment": {
            "genotype": {"perturbations": []},
            "phenotype": {"fitness": 1.0},
        },
        "reference": {"dataset_name": "x"},
        "publication": {"pubmed_id": "1"},
    }
    assert resolve_interned(legacy, {}) == legacy


def test_intern_roundtrip_and_dedup(tmp_path: Path) -> None:
    """Two records sharing an Environment -> ONE interned row + full reconstruction."""
    env_obj = _sga_environment()
    env_dict = env_obj.model_dump()
    digest = compute_sha256_hash(canonical_json(env_obj))

    # Records env and interned env are SEPARATE sibling envs (production design):
    # the records env stays a pristine 0..N keyspace; the interned env holds the
    # deduped constant sub-objects. A named sub-db would register its name as a key
    # in the records env, poisoning raw cursors (the `\x00` UnpicklingError).
    env = lmdb.open(str(tmp_path / "lmdb"), map_size=int(1e9))
    interned_env = lmdb.open(str(tmp_path / "interned"), map_size=int(1e9))
    with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
        for i in range(2):
            if itxn.get(digest.encode()) is None:
                itxn.put(digest.encode(), pickle.dumps(env_dict))
            record = {
                "experiment": {"environment": {"$ref": digest, "name": "SGA"}},
                "reference": {},
                "publication": {},
            }
            txn.put(f"{i}".encode(), pickle.dumps(record))

    # Dedup: the shared Environment is stored exactly once.
    with interned_env.begin() as itxn:
        assert itxn.stat()["entries"] == 1

    # Load interned into RAM, resolve every record -> full Environment restored.
    interned: dict[str, object] = {}
    with interned_env.begin() as itxn:
        for key, value in itxn.cursor():
            interned[key.decode()] = pickle.loads(value)
    for i in range(2):
        with env.begin() as txn:
            record = pickle.loads(txn.get(f"{i}".encode()))
        resolved = resolve_interned(record, interned)
        assert resolved["experiment"]["environment"] == env_dict
        # media name survived + is_synthetic present after reconstruction
        assert "SGA" in resolved["experiment"]["environment"]["media"]["name"]
        assert resolved["experiment"]["environment"]["media"]["is_synthetic"] is True
    env.close()
    interned_env.close()
