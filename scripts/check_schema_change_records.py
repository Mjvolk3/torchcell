# scripts/check_schema_change_records.py
# [[scripts.check_schema_change_records]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/check_schema_change_records.py
"""Schema-change tripwire: a commit that moves a schema fingerprint ships its record.

The staged schema surface (``torchcell/datamodels/schema.py`` + ``pydant.py``, read with
``git show :<path>``) is diffed against the merge base with ``--base`` (default
``origin/main``) by ``torchcell.provenance.schema_impact``. Every symbol whose contract
fingerprint moved (added, removed or modified) must be named by exactly one record staged
under ``database/schema_changes/`` since that merge base, with the same before and after
fingerprints, status and reasons, and every staged record must validate (a removal, a
rename, a loosened required field, a changed field meaning or a removed enum member
without the owner's sign-off is refused by the model). The one exemption is a NEW symbol
that no served dataset's closure holds (served = the newest release snapshot under
``database/releases/``); it is printed so the PR body lists it.

The diff is index-vs-merge-base, the shape of ``scripts/check_candidate_verdicts.py``:
the pre-commit hook sees the staged change, and on a clean tree it sees the branch's
commits. Whether the records on disk still explain the whole schema since the
grandfathered baseline is the enforcement test's job
(``tests/torchcell/datamodels/test_schema_change_records.py``).

Usage::

    python scripts/check_schema_change_records.py
    python scripts/check_schema_change_records.py --base main

Exit 1 names every unrecorded or stale change. ``TORCHCELL_SCHEMA_RECORD_ACK=1`` passes
the commit (CI still requires the record before landing). The logic lives in
``torchcell/provenance/schema_change.py``; policy in
[[torchcell.datamodels.schema-change-policy]].
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# A script's sys.path[0] is scripts/, so `import torchcell` would resolve to the editable
# install (the primary checkout) and check a worktree with main's code.
sys.path.insert(0, str(REPO))

from torchcell.provenance.schema_change import main  # noqa: E402


def run(argv: list[str] | None = None) -> int:
    """``schema-change tripwire`` with this script's arguments (``--base``, ``--repo``)."""
    return main(["tripwire", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    sys.exit(run())
