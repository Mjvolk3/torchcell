# tests/torchcell/database/test_live_rebuild_slurm.py
# [[tests.torchcell.database.test_live_rebuild_slurm]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/database/test_live_rebuild_slurm.py
"""``gilahyper_live_rebuild-slurm_docker.slurm``: the private-data and organism switches.

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
    # four readers of the switch since 2026-10-15: the freshness walk, the lane walk,
    # the content-id walk (all three in the same preflight block, same argv) and the
    # dev fence
    assert TEXT.count('include_private=sys.argv[2] == "1"') == 4
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


# --------------------------------------------------------------------------- #
# 2026.10.12: the organism switch, KG_ORGANISMS.
# --------------------------------------------------------------------------- #
ORGANISMS_FIRST = 'KG_ORGANISMS="${KG_ORGANISMS:-scerevisiae,ecoli,pputida}"'
ORGANISMS_LAST = 'KG_ORGANISMS="$ORGANISMS_CANONICAL"'
RESUME_FIRST = "    PREV_ORGANISMS=$(grep -oE '; organisms [a-z,]+' \"$PREV_OUT\" | head -1 | cut -d' ' -f3 || true)"


def _organisms_dispatch() -> str:
    """The ``KG_ORGANISMS`` default, validation and canonical ordering, verbatim."""
    lines = TEXT.splitlines()
    start = lines.index(ORGANISMS_FIRST)
    end = lines.index(ORGANISMS_LAST, start)
    return "\n".join(lines[start : end + 1])


def _run_organisms(value: str | None) -> subprocess.CompletedProcess[str]:
    env = {"PATH": "/usr/bin:/bin"}
    if value is not None:
        env["KG_ORGANISMS"] = value
    return subprocess.run(
        [
            "bash",
            "-c",
            f'set -euo pipefail\n{_organisms_dispatch()}\necho "organisms=[$KG_ORGANISMS]"',
        ],
        capture_output=True,
        text=True,
        env=env,
    )


def test_a_full_rebuild_serves_every_organism_by_default() -> None:
    """Unset (or empty) is all three, in Organism order."""
    for value in (None, ""):
        result = _run_organisms(value)
        assert result.returncode == 0
        assert result.stdout.strip() == "organisms=[scerevisiae,ecoli,pputida]"


def test_kg_organisms_is_put_in_enum_order_so_every_log_spells_it_one_way() -> None:
    """The set, not the typing order, reaches the preflight, the generator and the
    manifest; a repeated name collapses into the set.
    """
    expected = {
        "scerevisiae": "scerevisiae",
        "pputida,scerevisiae": "scerevisiae,pputida",
        "pputida,ecoli": "ecoli,pputida",
        "ecoli,ecoli": "ecoli",
    }
    for value, canonical in expected.items():
        result = _run_organisms(value)
        assert (result.returncode, result.stdout.strip()) == (
            0,
            f"organisms=[{canonical}]",
        ), value


def test_an_unknown_or_malformed_organism_set_stops_the_job() -> None:
    """Never a narrower or a default build: the job ends with the value it refused."""
    for value in ("yeast", "scerevisiae,", "scerevisiae ecoli", ",ecoli", "Ecoli"):
        result = _run_organisms(value)
        assert result.returncode == 1, value
        assert result.stdout.strip() == (
            "KG_ORGANISMS must be a comma-separated subset of "
            f"scerevisiae,ecoli,pputida, got '{value}'"
        )


def test_the_script_spells_the_organism_enum_in_its_order() -> None:
    """The bash validation and ordering name exactly the Organism members, in order,
    so a new organism added to the enum fails here until the script learns it.
    """
    from torchcell.data.experiment_dataset import Organism

    members = [o.value for o in Organism]
    dispatch = _organisms_dispatch()
    assert f"for ORGANISM in {' '.join(members)}; do" in dispatch
    alternation = "|".join(members)
    assert f"^({alternation})(,({alternation}))*$" in dispatch
    assert ORGANISMS_FIRST.endswith(f'{",".join(members)}}}"')


def test_the_preflight_the_fence_the_generator_and_the_manifest_read_kg_organisms() -> (
    None
):
    """One organism set narrows every walk of the map and is recorded.

    The freshness preflight and the dev fence both parse ``sys.argv[3]`` into the map
    they walk (so a yeast-only build neither blocks on a stale bacterial store nor
    counts a bacterial dataset), the generator gets ``--organisms``, and the manifest
    bootstrap records the set. The lane walk and the content-id walk of 2026-10-15 are
    the third and fourth readers of ``sys.argv[3]``, in the preflight block beside the
    freshness walk.
    """
    assert TEXT.count("organisms=parse_organisms(sys.argv[3])") == 4
    assert TEXT.count('"$DEV_DATA_ROOT" "$INCLUDE_PRIVATE" "$KG_ORGANISMS"') == 3
    assert (
        "python -m $KG_MODULE --config-name $KG_CONFIG $PRIVATE_FLAG "
        "--organisms $KG_ORGANISMS" in TEXT
    )
    assert '    --organisms "$KG_ORGANISMS"\n' in TEXT
    assert 'include private $INCLUDE_PRIVATE; organisms $KG_ORGANISMS"' in TEXT


def _resume_check() -> str:
    """The resume's organism comparison: the grep and its guard, verbatim."""
    lines = TEXT.splitlines()
    start = lines.index(RESUME_FIRST)
    return "\n".join(lines[start : start + 3])


def _run_resume(tmp_path: Path, log: str, kg_organisms: str) -> str:
    prev = tmp_path / "prev.out"
    prev.write_text(log, encoding="utf-8")
    result = subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{_resume_check()}\necho resumed"],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "PREV_OUT": str(prev),
            "KG_ORGANISMS": kg_organisms,
            "RESUME_JOB": "4242",
        },
    )
    return f"{result.returncode}:{result.stdout.strip()}"


def test_a_resume_must_name_the_organism_set_the_resumed_job_generated(
    tmp_path: Path,
) -> None:
    """The CSVs being resumed were generated for one set; the manifest must record it."""
    log = (
        "config kg_uncapped; commit abc; dev tree /d; serve /s; next /n; "
        "include private 1; organisms scerevisiae\n"
    )
    assert _run_resume(tmp_path, log, "scerevisiae") == "0:resumed"
    assert _run_resume(tmp_path, log, "scerevisiae,ecoli,pputida") == (
        "1:job 4242 generated organisms 'scerevisiae'; resubmit the resume with "
        "KG_ORGANISMS=scerevisiae, not scerevisiae,ecoli,pputida"
    )
    # a job from before the switch logged no set: the resume says so instead of dying
    # silently on the empty grep under pipefail
    pre_switch = "config kg_uncapped; commit abc; include private 1\n"
    assert _run_resume(tmp_path, pre_switch, "scerevisiae") == (
        "1:job 4242 generated organisms '(not logged)'; resubmit the resume with "
        "KG_ORGANISMS=<that set>, not scerevisiae"
    )


# --------------------------------------------------------------------------- #
# 2026.10.15: the ontology unification guards in the preflight and before the import.
# --------------------------------------------------------------------------- #


def test_the_preflight_checks_every_stores_leaves_against_its_confs_lanes() -> None:
    """The lane walk runs in the same preflight block as the freshness walk, on the same
    three arguments, and refuses the job on a store whose manifest lacks the field or
    carries a leaf no enabled lane emits.
    """
    assert (
        "from torchcell.database.build_dataset_lmdb import mapped_store_lanes" in TEXT
    )
    assert (
        'lanes = mapped_store_lanes(sys.argv[1], include_private=sys.argv[2] == "1", '
        "organisms=parse_organisms(sys.argv[3]))"
    ) in TEXT
    assert "unserved = [s.describe() for s in lanes if s.blocks]" in TEXT
    freshness = TEXT.index("statuses = mapped_store_status(")
    lanes = TEXT.index("lanes = mapped_store_lanes(")
    count = TEXT.index("print(len(statuses))")
    assert freshness < lanes < count


def test_the_preflight_refuses_a_store_whose_content_id_collisions_are_undeclared() -> (
    None
):
    """The collision walk runs in the same preflight block, after the lane walk and
    before the count, on the same three arguments. A record that shares an Experiment
    content id with another record of its store is absent from the graph with no error,
    so the job stops unless the dataset declares the number (lane V, 2026-10-15).
    """
    assert (
        "from torchcell.database.build_dataset_lmdb import mapped_store_collisions"
        in TEXT
    )
    assert (
        "collisions = mapped_store_collisions(sys.argv[1], include_private=sys.argv[2] "
        '== "1", organisms=parse_organisms(sys.argv[3]))'
    ) in TEXT
    assert "undeclared = [s.describe() for s in collisions if s.blocks]" in TEXT
    lanes = TEXT.index("lanes = mapped_store_lanes(")
    collisions = TEXT.index("collisions = mapped_store_collisions(")
    count = TEXT.index("print(len(statuses))")
    assert lanes < collisions < count


def test_the_csv_endpoint_gate_is_a_hard_stop_before_the_import() -> None:
    """The gate runs inside the import branch, after the dev fence and before the build
    container starts, and a failure exits the job; skip_bad_relationships stays as it is.
    """
    gate = TEXT.index("python -m torchcell.knowledge_graphs.csv_endpoint_gate")
    fence = TEXT.index("dev fence intact")
    container = TEXT.index("---------------- 2 IMPORT into")
    import_call = TEXT.index('bash "$OUT_CONT/neo4j-admin-import-call.sh"')
    assert fence < gate < container < import_call
    block = TEXT[gate : TEXT.index("\nfi\n", gate) + 4]
    assert '"$OUT_HOST" --report "$ENDPOINT_REPORT"' in block
    assert "CSV endpoint gate FAILED" in block and "exit 1" in block
    assert "skip_bad_relationships: true" in TEXT


# ------------------------------------------------------- GENERATION_MODE (2026-10-15)

MODE_FIRST = 'RESUME_FROM="${RESUME_FROM:-import}"'
MODE_LAST = 'ARRAY_FOLLOWUP="${ARRAY_FOLLOWUP:-0}"'


def _mode_dispatch() -> str:
    """The RESUME_FROM check and the GENERATION_MODE assignment plus its case, verbatim."""
    lines = TEXT.splitlines()
    start = lines.index(MODE_FIRST)
    end = lines.index(MODE_LAST, start)
    return "\n".join(lines[start:end])


def _run_mode(mode: str | None, resume_from: str | None = None) -> tuple[int, str]:
    env = {"PATH": "/usr/bin:/bin"}
    if mode is not None:
        env["GENERATION_MODE"] = mode
    if resume_from is not None:
        env["RESUME_FROM"] = resume_from
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'set -eu\n{_mode_dispatch()}\necho "mode=[$GENERATION_MODE] resume=[$RESUME_FROM]"',
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    return result.returncode, result.stdout.strip()


def test_generation_is_serial_unless_the_owner_flips_it() -> None:
    """The default stays serial; array is opt-in; RESUME_FROM=array implies array."""
    assert _run_mode(None) == (0, "mode=[serial] resume=[import]")
    assert _run_mode("array") == (0, "mode=[array] resume=[import]")
    assert _run_mode(None, "array") == (0, "mode=[array] resume=[array]")
    assert _run_mode("serial", "generation") == (0, "mode=[serial] resume=[generation]")
    assert _run_mode("parallel") == (
        1,
        "GENERATION_MODE must be serial or array, got 'parallel'",
    )
    assert _run_mode(None, "merge") == (
        1,
        "RESUME_FROM must be import, generation or array, got 'merge'",
    )


def test_the_array_mode_chains_a_follow_up_that_never_re_runs_a_task() -> None:
    """The generating job submits the arrays through the conductor and resubmits itself
    with --dependency=afterany and ARRAY_FOLLOWUP=1; a follow-up with missing markers
    prints the recovery command and exits 1, a by-hand resume re-runs them.
    """
    assert "bash database/scripts/kg_generation_array_conductor.sh" not in TEXT
    assert (
        'CONDUCTOR="$TORCHCELL_SRC/database/scripts/kg_generation_array_conductor.sh"'
        in TEXT
    )
    assert '--dependency="afterany:${ARRAY_JOBS//,/:}"' in TEXT
    assert "RESUME_FROM=array,ARRAY_FOLLOWUP=1" in TEXT
    assert 'if [ "$ARRAY_FOLLOWUP" = 1 ]; then' in TEXT
    assert "this follow-up never re-runs a task" in TEXT
    assert 'INDICES="$MISSING_IDS"' in TEXT
    assert (
        "python -m torchcell.knowledge_graphs.generation_array merge --out-dir" in TEXT
    )
    # the array path reaches the import branch's endpoint gate with OUT_HOST set to the
    # <ts> directory, whose adapters/*/ the gate walks as one output
    merge = TEXT.index("python -m torchcell.knowledge_graphs.generation_array merge")
    gate = TEXT.index("python -m torchcell.knowledge_graphs.csv_endpoint_gate")
    assert merge < gate
    assert "kg_csv_endpoint_check" not in TEXT
    assert '${MERGE_REPORT:+--merge-report "$MERGE_REPORT"}' in TEXT
    # the serial path is intact: its generation container, its generate-log checks
    assert 'docker run --rm --entrypoint bash --name "$GEN_CONTAINER"' in TEXT
    assert (
        "python -m $KG_MODULE --config-name $KG_CONFIG $PRIVATE_FLAG --organisms $KG_ORGANISMS"
        in TEXT
    )
    assert "grep -oE 'across [0-9]+ adapters'" in TEXT
