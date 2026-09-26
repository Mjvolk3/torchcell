# tests/torchcell/scripts/test_deprecate_sh.py
# [[tests.torchcell.scripts.test_deprecate_sh]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/scripts/test_deprecate_sh.py
"""``scripts/deprecate.sh`` moving a path into a graveyard under ``tmp_path``.

``DEPRECATED_DIR`` names the graveyard and ``DATA_ROOT`` a sibling directory, so the
refusal branch (a graveyard nested inside the data root exits 2 and moves nothing) is
exercised alongside the normal move: the target lands under
``<graveyard>/<timestamp>__<name>/<name>`` next to a ``DEPRECATION.txt`` manifest whose
fields are asserted line by line.
"""

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "deprecate.sh"


def _run(
    tmp_path: Path, graveyard: Path, *args: str
) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "DEPRECATED_DIR": str(graveyard),
        "DATA_ROOT": str(tmp_path / "data"),
    }
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        env=env,
    )


def test_deprecate_moves_the_target_and_writes_a_manifest(tmp_path: Path) -> None:
    """The file leaves its place, the manifest records its absolute path, host, git head, reason."""
    (tmp_path / "work").mkdir()
    target = tmp_path / "work" / "thing.txt"
    target.write_text("payload\n")
    graveyard = tmp_path / "graveyard"
    result = _run(tmp_path, graveyard, "work/thing.txt", "because")
    assert result.returncode == 0, result.stderr
    assert not target.exists()
    (dest,) = list(graveyard.iterdir())
    assert dest.name.endswith("__thing.txt")
    assert (dest / "thing.txt").read_text() == "payload\n"
    manifest = dict(
        line.split(": ", 1)
        for line in (dest / "DEPRECATION.txt").read_text().splitlines()
    )
    assert manifest["original_path"] == str(target)
    assert manifest["reason"] == "because"
    assert manifest["git_head"] == "not-a-git-repo"
    assert manifest["deprecated_at"] == dest.name.split("__", 1)[0]
    assert set(manifest) == {
        "original_path",
        "deprecated_at",
        "host",
        "user",
        "git_head",
        "size",
        "reason",
    }
    lines = result.stdout.splitlines()
    assert lines[0].startswith(f"deprecated -> {dest / 'thing.txt'}  (")
    assert lines[1] == f"  manifest: {dest / 'DEPRECATION.txt'}"
    assert (
        lines[2]
        == f"  graveyard: {graveyard}  (purge by hand; nothing is auto-deleted)"
    )


def test_deprecate_refuses_a_graveyard_inside_data_root(tmp_path: Path) -> None:
    """Exit 2, nothing moved, nothing created."""
    (tmp_path / "work").mkdir()
    target = tmp_path / "work" / "thing.txt"
    target.write_text("payload\n")
    graveyard = tmp_path / "data" / "deprecated"
    result = _run(tmp_path, graveyard, "work/thing.txt")
    assert result.returncode == 2
    assert "refusing graveyard inside DATA_ROOT" in result.stderr
    assert target.exists() and not graveyard.exists()


def test_deprecate_rejects_a_missing_target_and_no_argument(tmp_path: Path) -> None:
    """A missing path exits 1 naming it; no argument exits 1 with the usage."""
    graveyard = tmp_path / "graveyard"
    missing = _run(tmp_path, graveyard, "work/nope.txt")
    assert missing.returncode == 1
    assert missing.stderr.strip() == "deprecate: no such path: work/nope.txt"
    none = _run(tmp_path, graveyard)
    assert none.returncode == 1
    assert "usage: deprecate.sh <path> [reason]" in none.stderr
    assert not graveyard.exists()
