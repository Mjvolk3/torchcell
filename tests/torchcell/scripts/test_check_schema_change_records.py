# tests/torchcell/scripts/test_check_schema_change_records.py
# [[tests.torchcell.scripts.test_check_schema_change_records]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/scripts/test_check_schema_change_records.py
"""``scripts/check_schema_change_records.py`` against a temporary git repository.

The script is the pre-commit entry of ``schema-change tripwire``; the logic is tested in
``tests/torchcell/provenance/test_schema_change.py``. Here: the script puts its own repo
root first on ``sys.path`` (so a worktree's hook runs the worktree's code, not the
editable install's), forwards ``--base``/``--repo``, and exits 1 on an unrecorded change
of a served class and 0 once nothing changed. Git is hermetic.
"""

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.torchcell.provenance._schema_change_repo import (
    edit_schema,
    hermetic_git,
    make_repo,
)

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import check_schema_change_records as cscr  # type: ignore[import-not-found]  # noqa: E402


@pytest.fixture(autouse=True)
def _hermetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    hermetic_git(monkeypatch, tmp_path)
    monkeypatch.delenv("TORCHCELL_SCHEMA_RECORD_ACK", raising=False)
    yield


def test_the_script_runs_this_checkouts_code() -> None:
    """The repo root the script inserts is the checkout the script lives in."""
    module_file = sys.modules[cscr.main.__module__].__file__
    assert module_file is not None
    assert (cscr.REPO, Path(module_file).is_relative_to(cscr.REPO)) == (
        SCRIPTS.parent,
        True,
    )


def test_the_script_forwards_base_and_repo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Clean branch: exit 0; an unrecorded served change: exit 1 naming it."""
    repo = make_repo(tmp_path)
    assert cscr.run(["--base", "main", "--repo", str(repo)]) == 0
    assert capsys.readouterr().out == (
        "schema-change-records: every schema change vs main is recorded\n"
    )
    edit_schema(
        repo,
        "    fitness: float\n",
        "    fitness: float\n    note: str | None = None\n",
    )
    assert cscr.run(["--base", "main", "--repo", str(repo)]) == 1
    assert (
        capsys.readouterr()
        .out.splitlines()[0]
        .startswith(
            "schema-change-records: FitnessPhenotype (modified, stale; served via "
            "['ServedDataset']) changed since the merge base"
        )
    )
