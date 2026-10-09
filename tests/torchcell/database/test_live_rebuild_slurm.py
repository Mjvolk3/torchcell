# tests/torchcell/database/test_live_rebuild_slurm.py
# [[tests.torchcell.database.test_live_rebuild_slurm]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/database/test_live_rebuild_slurm.py
"""``gilahyper_live_rebuild-slurm_docker.slurm``: the private-data switch of a FULL build.

The rest of the script runs docker, slurm and neo4j-admin against the served store, so
the hermetic part is the ``INCLUDE_PRIVATE`` dispatch: which flag it hands the generator,
that the preflight and the dev fence count the private map under the same switch, and
that an unparseable value stops the job instead of silently building the public graph.
The block is extracted by its own first and last line rather than by line number, so it
stays pinned when the script grows above it.
"""

import subprocess
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "database"
    / "slurm"
    / "scripts"
    / "gilahyper_live_rebuild-slurm_docker.slurm"
)
TEXT = SCRIPT.read_text(encoding="utf-8")
FIRST = 'INCLUDE_PRIVATE="${INCLUDE_PRIVATE:-1}"'


def _dispatch() -> str:
    """The ``INCLUDE_PRIVATE`` assignment plus its ``case`` block, verbatim."""
    lines = TEXT.splitlines()
    start = lines.index(FIRST)
    end = lines.index("esac", start)
    return "\n".join(lines[start : end + 1])


def _run(value: str | None) -> subprocess.CompletedProcess[str]:
    """Run the extracted dispatch and echo the flag it chose."""
    env = {"PATH": "/usr/bin:/bin"}
    if value is not None:
        env["INCLUDE_PRIVATE"] = value
    return subprocess.run(
        ["bash", "-c", f'set -eu\n{_dispatch()}\necho "flag=[$PRIVATE_FLAG]"'],
        capture_output=True,
        text=True,
        env=env,
    )


def test_a_full_rebuild_includes_the_private_datasets_by_default() -> None:
    """Unset means 1 means ``--include-private``: KG 4.0 serves the in-house data."""
    result = _run(None)
    assert result.returncode == 0
    assert result.stdout.strip() == "flag=[--include-private]"
    assert _run("1").stdout.strip() == "flag=[--include-private]"


def test_include_private_zero_builds_the_public_datasets_alone() -> None:
    """The opt-out passes no flag, so the generator refuses a private dataset by name."""
    result = _run("0")
    assert result.returncode == 0
    assert result.stdout.strip() == "flag=[]"


def test_an_unparseable_value_stops_the_job() -> None:
    """A value that is neither 0 nor 1 is an error, not a default to the public build."""
    result = _run("yes")
    assert result.returncode == 1
    # the message is echoed on stdout, which slurm folds into the same job log
    assert result.stdout.strip() == "INCLUDE_PRIVATE must be 0 or 1, got 'yes'"
    assert result.stderr == ""
    assert "flag=" not in result.stdout


def test_the_preflight_the_fence_and_the_generator_read_one_switch() -> None:
    """All three places take the same variable, so none can disagree with the others.

    The freshness preflight and the dev fence pass ``$INCLUDE_PRIVATE`` into the call
    that enumerates the mapped datasets (a public-only preflight would not notice a
    stale private store), and the generator line carries ``$PRIVATE_FLAG``.
    """
    assert TEXT.count('include_private=sys.argv[2] == "1"') == 2
    assert TEXT.count('"$DEV_DATA_ROOT" "$INCLUDE_PRIVATE"') == 3
    assert "python -m $KG_MODULE --config-name $KG_CONFIG $PRIVATE_FLAG" in TEXT


def test_the_preflight_is_the_same_function_list_stale_prints_from() -> None:
    """The preflight calls ``mapped_store_status`` rather than re-stating the check.

    Issue #833: the inline block it replaced compared fingerprints and stopped there, so
    a store pickled under a class the build commit's schema does not define read fresh
    at preflight and failed hours later inside an adapter. The one function it now calls
    ends with a bounded read of each store's first record, and because it is literally
    the function behind ``build_dataset_lmdb --list-stale``, the list the owner rebuilds
    from and the list this job refuses on cannot disagree.
    """
    assert (
        "from torchcell.database.build_dataset_lmdb import mapped_store_status" in TEXT
    )
    assert "bad = [s.describe() for s in statuses if s.needs_rebuild]" in TEXT
    # the superseded re-statement of the check is gone from the script
    assert "check_manifest(BuildManifest.model_validate_json" not in TEXT
