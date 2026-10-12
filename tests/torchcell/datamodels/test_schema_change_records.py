# tests/torchcell/datamodels/test_schema_change_records.py
# [[tests.torchcell.datamodels.test_schema_change_records]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datamodels/test_schema_change_records.py
"""Every schema change since the grandfathered baseline is carried by a record.

The CI half of [[torchcell.datamodels.schema-change-policy]]; the commit half is the
``schema-change-records`` pre-commit hook. ``database/schema_changes/grandfathered.json``
pins every symbol of the fingerprinted surface (``schema.py`` + ``pydant.py``) at main
``eb07bd213`` (2026-10-12, 284 symbols: 229 classes and 55 module-level bindings), the
commit the policy landed on: nothing changed before it needs a record, everything changed
after it does. A record whose ``after`` fingerprint no longer matches the schema is stale
and fails here, as does a moved symbol no record names, a record without its PR, and
fingerprints computed under another interpreter (they hash ``ast.unparse`` text, which
differs across 3.13 patch releases).

``TORCHCELL_SCHEMA_RECORD_ACK=1`` skips the drift check for a local run in an emergency.
CI never sets it, so it cannot let a change land without its record.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

import pytest

from torchcell.knowledge_graphs.kg_manifest import surface_in_worktree
from torchcell.provenance.schema_change import (
    ACK_ENV,
    BASELINE_FILENAME,
    enforcement_problems,
    load_baseline,
    load_records,
    records_dir,
)

REPO = Path(__file__).resolve().parents[3]
BASELINE_COMMIT = "eb07bd213ca1b8f4138cc4327664558109024dd3"


def test_the_baseline_is_the_commit_the_policy_landed_on() -> None:
    """Re-pinning the baseline is a visible edit of this constant, never a quiet one."""
    baseline = load_baseline(REPO)
    assert (baseline.base_commit, baseline.python_version) == (
        BASELINE_COMMIT,
        "3.13.0",
    )
    assert len(baseline.fingerprints) == 284


def test_the_interpreter_matches_the_baseline() -> None:
    """Fingerprints computed under another Python patch release would all differ."""
    assert platform.python_version() == load_baseline(REPO).python_version


def test_every_file_in_the_store_is_a_loadable_record() -> None:
    """load_records refuses a stray file, a misnamed record and a repeated slug."""
    on_disk = sorted(
        path.name
        for path in records_dir(REPO).glob("*.json")
        if path.name != BASELINE_FILENAME
    )
    assert sorted(load_records(REPO)) == on_disk


def test_records_carry_every_change_since_the_baseline() -> None:
    """No unrecorded drift, no stale record, every record names its PR."""
    if os.environ.get(ACK_ENV):
        pytest.skip(f"{ACK_ENV} set: schema-change drift check skipped (emergency)")
    assert (
        enforcement_problems(
            load_baseline(REPO), load_records(REPO), surface_in_worktree(REPO)
        )
        == []
    )
