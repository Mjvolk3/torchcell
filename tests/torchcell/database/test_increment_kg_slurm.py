# tests/torchcell/database/test_increment_kg_slurm.py
# [[tests.torchcell.database.test_increment_kg_slurm]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/database/test_increment_kg_slurm.py
"""``gilahyper_increment_kg-slurm_docker.slurm``: the ``WIDEN_ORGANISMS`` switch.

Everything else in the script runs docker and neo4j-admin against the served store, so
the hermetic part is the dispatch that decides whether the admission check may widen the
store's organism set, and that its arguments reach the ``admit`` call. The block is
extracted by its own first and last line, as in ``test_live_rebuild_slurm.py``.
"""

import subprocess
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "database"
    / "slurm"
    / "scripts"
    / "gilahyper_increment_kg-slurm_docker.slurm"
)
TEXT = SCRIPT.read_text(encoding="utf-8")
FIRST = 'WIDEN_ORGANISMS="${WIDEN_ORGANISMS:-0}"'


def _dispatch() -> str:
    lines = TEXT.splitlines()
    start = lines.index(FIRST)
    end = lines.index("esac", start)
    return "\n".join(lines[start : end + 1])


def _run(value: str | None) -> subprocess.CompletedProcess[str]:
    env = {"PATH": "/usr/bin:/bin"}
    if value is not None:
        env["WIDEN_ORGANISMS"] = value
    return subprocess.run(
        ["bash", "-c", f'set -eu\n{_dispatch()}\necho "args=[${{WIDEN_ARGS[*]:-}}]"'],
        capture_output=True,
        text=True,
        env=env,
    )


def test_an_increment_never_widens_the_organism_set_by_default() -> None:
    """Unset means 0 means no flag: a dataset of another organism blocks by name."""
    for value in (None, "0"):
        result = _run(value)
        assert (result.returncode, result.stdout.strip()) == (0, "args=[]")


def test_widen_organisms_one_passes_the_flag() -> None:
    result = _run("1")
    assert (result.returncode, result.stdout.strip()) == (0, "args=[--widen-organisms]")


def test_an_unparseable_widen_value_stops_the_job() -> None:
    result = _run("yes")
    assert result.returncode == 1
    assert result.stdout.strip() == "WIDEN_ORGANISMS must be 0 or 1, got 'yes'"


def test_the_admission_call_carries_the_widen_arguments() -> None:
    assert (
        '--neo4j-uri "$NEO4J_BOLT_URI" --database "$NEO4J_DATABASE" '
        '"${PRIVATE_ARGS[@]}" "${WIDEN_ARGS[@]}"' in TEXT
    )
