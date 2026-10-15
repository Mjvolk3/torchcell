# tests/torchcell/knowledge_graphs/test_generation_array.py
# [[tests.torchcell.knowledge_graphs.test_generation_array]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/knowledge_graphs/test_generation_array.py
"""``generation_array``: the tier table, the index-to-adapter order, the DONE
protocol, the sbatch plan, the merge and the duplicate check, on synthetic
directories shaped exactly as ``FastCsvSink`` plus ``BioCypher.write_import_call``
leave them (one ``<Label>-part000.csv`` per label, a ``<Label>-header.csv``, a
``neo4j-admin-import-call.sh``, and the marker last).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from torchcell.data.experiment_dataset import ALL_ORGANISMS, Organism
from torchcell.knowledge_graphs import generation_array as ga
from torchcell.knowledge_graphs.generation_array import (
    DONE_FILENAME,
    IMPORT_CALL_FILENAME,
    MERGED_DIRNAME,
    AdapterSlot,
    DoneMarker,
    DuplicateConflict,
    GenerationTier,
    TierTable,
    adapter_order,
    call_prefix_and_suffix,
    check_label_duplicates,
    compress_ids,
    discover_label_files,
    load_merge_report,
    load_tier_table,
    merge_outputs,
    missing_slots,
    read_done,
    read_order,
    recovery_commands,
    scan_part,
    slot_by_index,
    tier_submissions,
    write_order,
)

# ------------------------------------------------------------------------ tier table


def test_the_tier_table_carries_the_overestimated_sizes_of_the_decision() -> None:
    """Standard 16/64/8 h, large 32/160/16 h, hoepfner 64/320/36 h; standard is the default."""
    table = load_tier_table()
    assert table.default_tier == "standard"
    sizes = {
        name: (t.cpus, t.mem_gb, t.hours, t.concurrency)
        for name, t in table.tiers.items()
    }
    assert sizes == {
        "standard": (16, 64, 8, 4),
        "large": (32, 160, 16, 2),
        "hoepfner": (64, 320, 36, 1),
    }
    assert table.tier_for("EnvChemgenHoepfner2014Adapter").name == "hoepfner"
    assert table.tier_for("DmfCostanzo2016Adapter").name == "large"
    assert table.tier_for("SmfCostanzo2016Adapter").name == "standard"
    assert table.tiers["hoepfner"].time_limit == "36:00:00"


def test_every_adapter_the_tier_table_names_is_a_mapped_adapter_class() -> None:
    """A renamed adapter would otherwise silently fall back to the standard tier."""
    from torchcell.knowledge_graphs.dataset_adapter_map import build_adapter_map

    mapped = {a.__name__ for a in build_adapter_map(include_private=True).values()}
    unknown = sorted(set(load_tier_table().adapters) - mapped)
    assert unknown == []


def test_a_tier_table_must_resolve_its_default_and_every_adapter_tier() -> None:
    tier = GenerationTier(name="standard", cpus=1, mem_gb=1, hours=1, concurrency=1)
    with pytest.raises(ValueError, match="default_tier 'big' is not a tier"):
        TierTable(default_tier="big", tiers={"standard": tier})
    with pytest.raises(ValueError, match="tiers that do not exist"):
        TierTable(
            default_tier="standard", tiers={"standard": tier}, adapters={"X": "big"}
        )
    with pytest.raises(ValueError, match="carries name"):
        TierTable(default_tier="s", tiers={"s": tier})


# ----------------------------------------------------------------------------- order


def test_the_order_is_the_build_adapter_map_in_its_own_order() -> None:
    """Index i is position i of build_adapter_map; private only with the flag."""
    from torchcell.knowledge_graphs.dataset_adapter_map import (
        PRIVATE_DATASET_ADAPTER_MAP,
        build_adapter_map,
    )

    public = adapter_order(include_private=False)
    expected = list(build_adapter_map(include_private=False).items())
    assert [(s.dataset_class, s.adapter_class) for s in public] == [
        (d.__name__, a.__name__) for d, a in expected
    ]
    assert [s.index for s in public] == list(range(len(expected)))
    private_names = {d.__name__ for d in PRIVATE_DATASET_ADAPTER_MAP}
    assert private_names.isdisjoint(s.dataset_class for s in public)
    full = adapter_order(include_private=True)
    assert len(full) == len(public) + len(PRIVATE_DATASET_ADAPTER_MAP)
    assert [s.dataset_class for s in full[: len(public)]] == [
        s.dataset_class for s in public
    ]
    assert private_names <= {s.dataset_class for s in full}
    yeast = adapter_order(
        include_private=True, organisms=frozenset({Organism.scerevisiae})
    )
    assert 0 < len(yeast) < len(full)
    assert ALL_ORGANISMS >= frozenset({Organism.scerevisiae})


def test_slot_by_index_refuses_an_index_outside_the_order() -> None:
    order = [
        AdapterSlot(index=i, dataset_class=f"D{i}", adapter_class=f"A{i}")
        for i in range(3)
    ]
    assert slot_by_index(order, 2).dirname == "002_A2"
    with pytest.raises(IndexError, match="outside the build order of 3"):
        slot_by_index(order, 3)
    with pytest.raises(IndexError):
        slot_by_index(order, -1)


def test_the_order_round_trips_through_its_json_file(tmp_path: Path) -> None:
    order = [
        AdapterSlot(index=i, dataset_class=f"D{i}", adapter_class=f"A{i}")
        for i in range(4)
    ]
    write_order(order, tmp_path / "adapter_order.json")
    assert read_order(tmp_path / "adapter_order.json") == order


def test_compress_ids_is_the_slurm_array_spelling() -> None:
    assert compress_ids([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"
    assert compress_ids([3]) == "3"
    assert compress_ids([9, 8, 8, 7]) == "7-9"
    assert ga._expand_ids("0-2,5,7-8") == [0, 1, 2, 5, 7, 8]
    with pytest.raises(ValueError):
        compress_ids([])


# ------------------------------------------------------------------- submissions


def _order(n: int = 5) -> list[AdapterSlot]:
    names = [
        "SmallA",
        "DmfCostanzo2016Adapter",
        "SmallB",
        "EnvChemgenHoepfner2014Adapter",
        "SmallC",
    ]
    return [
        AdapterSlot(index=i, dataset_class=f"D{i}", adapter_class=names[i])
        for i in range(n)
    ]


def test_one_sbatch_per_tier_with_the_tier_sizes_and_the_environment() -> None:
    """Multi-member values ride the environment with --export=ALL, not --export's list."""
    subs = tier_submissions(
        _order(),
        load_tier_table(),
        script="array.slurm",
        out_name="2026-10-15_00-00-00",
        environment={
            "KG_ORGANISMS": "scerevisiae,ecoli,pputida",
            "INCLUDE_PRIVATE": "1",
        },
    )
    assert [(s.tier, s.indices) for s in subs] == [
        ("standard", [0, 2, 4]),
        ("large", [1]),
        ("hoepfner", [3]),
    ]
    standard = subs[0]
    assert standard.argv == [
        "sbatch",
        "--job-name=gen_adapter_standard",
        "--array=0,2,4%4",
        "--cpus-per-task=16",
        "--mem=64G",
        "--time=08:00:00",
        "--export=ALL",
        "array.slurm",
    ]
    assert standard.environment == {
        "KG_ORGANISMS": "scerevisiae,ecoli,pputida",
        "INCLUDE_PRIVATE": "1",
        "OUT_NAME": "2026-10-15_00-00-00",
        "GEN_TIER": "standard",
        "GEN_CPUS": "16",
        "GEN_MEM_GB": "64",
    }
    assert subs[2].argv[2:6] == [
        "--array=3%1",
        "--cpus-per-task=64",
        "--mem=320G",
        "--time=36:00:00",
    ]
    line = subs[2].shell()
    assert line.startswith(
        "GEN_CPUS=64 GEN_MEM_GB=320 GEN_TIER=hoepfner INCLUDE_PRIVATE=1 KG_ORGANISMS=scerevisiae,ecoli,pputida OUT_NAME="
    )
    assert line.endswith("--export=ALL array.slurm")
    assert "," not in line.split("--export=")[1].split()[0]


def test_a_dependency_is_an_explicit_sbatch_argument() -> None:
    """SBATCH_DEPENDENCY in the environment was not honored (jobs 4920/4921)."""
    subs = tier_submissions(
        _order(),
        load_tier_table(),
        script="s",
        out_name="o",
        environment={},
        indices=[3],
        dependency="afterany:4919",
    )
    assert subs[0].argv[-3:] == ["--dependency=afterany:4919", "--export=ALL", "s"]
    assert "--dependency" not in " ".join(
        tier_submissions(
            _order(), load_tier_table(), script="s", out_name="o", environment={}
        )[0].argv
    )


def test_a_submission_restricted_to_some_indices_covers_only_those_tiers() -> None:
    subs = tier_submissions(
        _order(),
        load_tier_table(),
        script="s",
        out_name="o",
        environment={},
        indices=[0, 4],
    )
    assert [(s.tier, s.indices, s.argv[2]) for s in subs] == [
        ("standard", [0, 4], "--array=0,4%4")
    ]
    with pytest.raises(IndexError, match=r"outside the build order of 5: \[7\]"):
        tier_submissions(
            _order(),
            load_tier_table(),
            script="s",
            out_name="o",
            environment={},
            indices=[7],
        )


def test_submit_dry_run_prints_and_never_calls_sbatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("sbatch was called in a dry run")

    monkeypatch.setattr(subprocess, "run", refuse)
    subs = tier_submissions(
        _order(), load_tier_table(), script="s", out_name="o", environment={}
    )
    assert ga.submit(subs, dry_run=True) == []
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 3 and all(line.startswith("DRY-RUN ") for line in out)


def test_the_plan_command_is_a_dry_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        ga, "adapter_order", lambda include_private=False, organisms=None: _order()
    )

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("sbatch was called by plan")

    monkeypatch.setattr(subprocess, "run", refuse)
    assert (
        ga.main(["plan", "--out-name", "o", "--script", "s", "--include-private"]) == 0
    )
    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("DRY-RUN")
    ]
    assert len(lines) == 3
    assert "INCLUDE_PRIVATE=1" in lines[0] and "--array=0,2,4%4" in lines[0]


def test_submit_reads_the_job_ids_sbatch_prints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs["env"]))
        return subprocess.CompletedProcess(
            argv, 0, stdout=f"Submitted batch job {4000 + len(calls)}\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    subs = tier_submissions(
        _order(), load_tier_table(), script="s", out_name="o", environment={"X": "1"}
    )
    assert ga.submit(subs, dry_run=False) == [4001, 4002, 4003]
    assert [c[1]["GEN_TIER"] for c in calls] == ["standard", "large", "hoepfner"]
    assert all(c[1]["X"] == "1" for c in calls)


# ------------------------------------------------------------------------------ DONE


def _marker(slot: AdapterSlot, **overrides: Any) -> str:
    fields: dict[str, Any] = {
        "index": slot.index,
        "adapter_class": slot.adapter_class,
        "dataset_class": slot.dataset_class,
        "dataset_len": 3,
        "n_nodes": 4,
        "n_edges": 5,
        "generation_wall_s": 1.5,
        "peak_memory_gb": 0.5,
        "torchcell_version": "1.7.5",
        "build_commit": "abc",
        "written_at": "2026-10-15T00:00:00+00:00",
    }
    fields.update(overrides)
    return DoneMarker(**fields).model_dump_json()


def test_missing_slots_are_those_without_a_marker_and_the_recovery_is_per_tier(
    tmp_path: Path,
) -> None:
    order = _order()
    for slot in order:
        (tmp_path / "adapters" / slot.dirname).mkdir(parents=True)
    for slot in (order[0], order[2], order[3]):
        (tmp_path / "adapters" / slot.dirname / DONE_FILENAME).write_text(_marker(slot))
    missing = missing_slots(tmp_path, order)
    assert [s.index for s in missing] == [1, 4]
    lines = recovery_commands(
        missing,
        load_tier_table(),
        script="array.slurm",
        out_name="o",
        environment={"K": "v"},
    )
    assert len(lines) == 2
    assert "--array=4%4" in lines[0] and "GEN_TIER=standard" in lines[0]
    assert (
        "--array=1%2" in lines[1]
        and "GEN_TIER=large" in lines[1]
        and "GEN_MEM_GB=160" in lines[1]
    )
    assert read_done(tmp_path, order[0]) is not None
    assert read_done(tmp_path, order[1]) is None


def test_a_marker_naming_another_slot_is_an_error_not_a_done(tmp_path: Path) -> None:
    order = _order(2)
    d = tmp_path / "adapters" / order[1].dirname
    d.mkdir(parents=True)
    (d / DONE_FILENAME).write_text(_marker(order[0]))
    with pytest.raises(ValueError, match="names index 0 SmallA, expected 1"):
        read_done(tmp_path, order[1])


# ----------------------------------------------------------------------------- merge

CONTAINER_ROOT = "/var/lib/neo4j/biocypher-out/2026-10-15_00-00-00"
PREFIX = (
    'bin/neo4j-admin database import full --delimiter="\\t" --array-delimiter="|" '
    '--quote="\'" --overwrite-destination=true --skip-bad-relationships=true '
    "--skip-duplicate-nodes=true "
)


def _write_dir(
    out: Path,
    slot: AdapterSlot,
    nodes: dict[str, list[str]],
    edges: dict[str, list[str]],
    *,
    done: bool = True,
    node_columns: dict[str, str] | None = None,
    skip_duplicate_nodes: bool = True,
) -> Path:
    """One adapter directory as the sink + BioCypher leave it; rows are given whole."""
    d = out / "adapters" / slot.dirname
    d.mkdir(parents=True)
    cdir = f"{CONTAINER_ROOT}/adapters/{slot.dirname}"
    args = ""
    for label, rows in nodes.items():
        header = (node_columns or {}).get(label, ":ID\tname\t:LABEL")
        (d / f"{label}-header.csv").write_text(header)
        (d / f"{label}-part000.csv").write_text("".join(r + "\n" for r in rows))
        args += f'--nodes="{cdir}/{label}-header.csv,{cdir}/{label}-part.*" '
    for label, rows in edges.items():
        (d / f"{label}-header.csv").write_text(":START_ID\tid\t:END_ID\t:TYPE")
        (d / f"{label}-part000.csv").write_text("".join(r + "\n" for r in rows))
        args += f'--relationships="{cdir}/{label}-header.csv,{cdir}/{label}-part.*" '
    prefix = (
        PREFIX
        if skip_duplicate_nodes
        else PREFIX.replace("--skip-duplicate-nodes=true ", "")
    )
    (d / IMPORT_CALL_FILENAME).write_text(
        "#!/bin/bash\nversion=$(bin/neo4j-admin --version | cut -d '.' -f 1)\n"
        f"if [[ $version -ge 5 ]]; then\n\t{prefix}{args}torchcell \nelse\n\tx\nfi"
    )
    if done:
        (d / DONE_FILENAME).write_text(_marker(slot))
    return d


def _two_dirs(tmp_path: Path, *, done_b: bool = True) -> list[AdapterSlot]:
    """A and B share genome g1 (identical rows) and the edge g1->e1 (identical rows)."""
    order = [
        AdapterSlot(index=0, dataset_class="DA", adapter_class="AAdapter"),
        AdapterSlot(index=1, dataset_class="DB", adapter_class="BAdapter"),
    ]
    _write_dir(
        tmp_path,
        order[0],
        nodes={
            "Genome": ["g1\t'S288C'\t'Genome'"],
            "Experiment": ["x1\t'one'\t'Experiment'", "x2\t'two'\t'Experiment'"],
            "OnlyA": ["a1\t'a'\t'OnlyA'"],
        },
        edges={
            "GenomeMemberOf": [
                "g1\t\te1\t'GenomeMemberOf'",
                "g1\t\te2\t'GenomeMemberOf'",
            ],
            "ExperimentMemberOf": [
                "x1\t\tDA\t'ExperimentMemberOf'",
                "x2\t\tDA\t'ExperimentMemberOf'",
            ],
        },
    )
    _write_dir(
        tmp_path,
        order[1],
        nodes={
            "Genome": ["g1\t'S288C'\t'Genome'", "g2\t'BY4741'\t'Genome'"],
            "Experiment": ["x3\t'three'\t'Experiment'"],
        },
        edges={
            "GenomeMemberOf": [
                "g1\t\te1\t'GenomeMemberOf'",
                "g2\t\te3\t'GenomeMemberOf'",
            ],
            "ExperimentMemberOf": ["x3\t\tDB\t'ExperimentMemberOf'"],
        },
        done=done_b,
    )
    write_order(order, tmp_path / "adapter_order.json")
    return order


def test_the_merge_writes_one_call_per_label_header_and_comma_joined_parts(
    tmp_path: Path,
) -> None:
    """Nodes first, relationships after, the first directory's header, every part."""
    order = _two_dirs(tmp_path)
    report = merge_outputs(tmp_path, order, workers=1)
    script = (tmp_path / IMPORT_CALL_FILENAME).read_text()
    v5 = next(
        line for line in script.splitlines() if "database import full" in line
    ).lstrip("\t")
    assert v5.startswith(PREFIX)
    assert v5.endswith(" torchcell ")
    a = f"{CONTAINER_ROOT}/adapters/000_AAdapter"
    b = f"{CONTAINER_ROOT}/adapters/001_BAdapter"
    assert (
        f'--nodes="{a}/Experiment-header.csv,{a}/Experiment-part000.csv,{b}/Experiment-part000.csv" '
        in v5
    )
    assert (
        f'--nodes="{a}/Genome-header.csv,{a}/Genome-part000.csv,{b}/Genome-part000.csv" '
        in v5
    )
    assert f'--nodes="{a}/OnlyA-header.csv,{a}/OnlyA-part000.csv" ' in v5
    # the duplicated edge row of B was dropped into a rewritten part under merged/
    merged_b = (
        f"{CONTAINER_ROOT}/{MERGED_DIRNAME}/001_BAdapter/GenomeMemberOf-part000.csv"
    )
    assert (
        f'--relationships="{a}/GenomeMemberOf-header.csv,{a}/GenomeMemberOf-part000.csv,{merged_b}" '
        in v5
    )
    assert (
        f'--relationships="{a}/ExperimentMemberOf-header.csv,{a}/ExperimentMemberOf-part000.csv,{b}/ExperimentMemberOf-part000.csv" '
        in v5
    )
    assert v5.index("--nodes=") < v5.index("--relationships=")
    assert v5.rindex("--nodes=") < v5.index("--relationships=")
    rewritten = (
        tmp_path / MERGED_DIRNAME / "001_BAdapter" / "GenomeMemberOf-part000.csv"
    )
    assert rewritten.read_text() == "g2\t\te3\t'GenomeMemberOf'\n"
    assert report.node_duplicates == {"Experiment": 0, "Genome": 1, "OnlyA": 0}
    assert report.edge_duplicates == {"ExperimentMemberOf": 0, "GenomeMemberOf": 1}
    assert report.rewritten == {
        str(tmp_path / "adapters" / "001_BAdapter" / "GenomeMemberOf-part000.csv"): str(
            rewritten
        )
    }
    assert report.labels["Genome"].duplicate_keys == 1
    assert report.labels["Genome"].rows == 3
    assert load_merge_report(tmp_path / "merge_report.json") == report
    # the original part file of B is untouched
    assert (
        tmp_path / "adapters" / "001_BAdapter" / "GenomeMemberOf-part000.csv"
    ).read_text().count("\n") == 2


def test_the_merge_refuses_while_a_directory_lacks_its_marker(tmp_path: Path) -> None:
    order = _two_dirs(tmp_path, done_b=False)
    with pytest.raises(
        FileNotFoundError,
        match=r"1 of 2 adapter directories carry no DONE: indices 1 \(BAdapter\)",
    ):
        merge_outputs(tmp_path, order)
    assert not (tmp_path / IMPORT_CALL_FILENAME).exists()


def test_a_duplicate_node_id_with_a_different_row_refuses_and_names_it(
    tmp_path: Path,
) -> None:
    order = [
        AdapterSlot(index=0, dataset_class="DA", adapter_class="AAdapter"),
        AdapterSlot(index=1, dataset_class="DB", adapter_class="BAdapter"),
    ]
    _write_dir(
        tmp_path, order[0], nodes={"Genome": ["g1\t'S288C'\t'Genome'"]}, edges={}
    )
    _write_dir(
        tmp_path, order[1], nodes={"Genome": ["g1\t'S288C v2'\t'Genome'"]}, edges={}
    )
    with pytest.raises(DuplicateConflict) as info:
        merge_outputs(tmp_path, order)
    message = str(info.value)
    assert message.startswith(
        "Genome: 1 duplicate ids map to rows that are not byte-identical"
    )
    assert (
        "g1 in [" in message
        and "000_AAdapter/Genome-part000.csv" in message
        and "001_BAdapter/Genome-part000.csv" in message
    )


def test_a_duplicate_edge_key_with_a_different_row_refuses(tmp_path: Path) -> None:
    order = [
        AdapterSlot(index=0, dataset_class="DA", adapter_class="AAdapter"),
        AdapterSlot(index=1, dataset_class="DB", adapter_class="BAdapter"),
    ]
    _write_dir(
        tmp_path,
        order[0],
        nodes={"N": ["n1\t'a'\t'N'"]},
        edges={"MemberOf": ["g1\tr1\te1\t'MemberOf'"]},
    )
    _write_dir(
        tmp_path,
        order[1],
        nodes={"N": ["n2\t'b'\t'N'"]},
        edges={"MemberOf": ["g1\tr2\te1\t'MemberOf'"]},
    )
    with pytest.raises(
        DuplicateConflict,
        match="MemberOf: 1 duplicate edge keys map to rows that are not byte-identical across directories; first: g1 -> e1 in",
    ):
        merge_outputs(tmp_path, order)


def test_a_key_repeated_inside_one_directory_is_what_the_adapter_wrote(
    tmp_path: Path,
) -> None:
    """Only a key seen in two or more directories is a duplicate; rows within one stay."""
    order = [AdapterSlot(index=0, dataset_class="DA", adapter_class="AAdapter")]
    _write_dir(
        tmp_path,
        order[0],
        nodes={"N": ["n1\t'a'\t'N'"]},
        edges={"MemberOf": ["g1\t\te1\t'MemberOf'", "g1\t\te1\t'MemberOf'"]},
    )
    report = merge_outputs(tmp_path, order)
    assert report.labels["MemberOf"].duplicate_rows == 0
    assert report.rewritten == {}


def test_headers_of_one_label_must_be_byte_identical_across_directories(
    tmp_path: Path,
) -> None:
    order = [
        AdapterSlot(index=0, dataset_class="DA", adapter_class="AAdapter"),
        AdapterSlot(index=1, dataset_class="DB", adapter_class="BAdapter"),
    ]
    _write_dir(tmp_path, order[0], nodes={"N": ["n1\t'a'\t'N'"]}, edges={})
    _write_dir(
        tmp_path,
        order[1],
        nodes={"N": ["n2\t'b'\t1\t'N'"]},
        edges={},
        node_columns={"N": ":ID\tname\tn:long\t:LABEL"},
    )
    with pytest.raises(ValueError, match="N: header in .*001_BAdapter differs from"):
        discover_label_files(tmp_path, order)


def test_the_merged_call_needs_skip_duplicate_nodes_from_the_biocypher_config(
    tmp_path: Path,
) -> None:
    order = [AdapterSlot(index=0, dataset_class="DA", adapter_class="AAdapter")]
    _write_dir(
        tmp_path,
        order[0],
        nodes={"N": ["n1\t'a'\t'N'"]},
        edges={},
        skip_duplicate_nodes=False,
    )
    with pytest.raises(ValueError, match="did not set skip_duplicate_nodes: true"):
        merge_outputs(tmp_path, order)
    prefix, suffix, cdir = call_prefix_and_suffix(
        (tmp_path / "adapters" / "000_AAdapter" / IMPORT_CALL_FILENAME)
        .read_text()
        .replace("--skip-bad", "--skip-duplicate-nodes=true --skip-bad")
    )
    assert prefix.endswith("--skip-duplicate-nodes=true --skip-bad-relationships=true ")
    assert (suffix, cdir) == ("torchcell ", f"{CONTAINER_ROOT}/adapters/000_AAdapter")


def test_scan_part_hashes_the_key_and_the_whole_row_and_parses_a_quoted_tab(
    tmp_path: Path,
) -> None:
    """Two rows with the same id and different text hash to the same key and different rows;
    a row whose quoted value holds a tab is parsed with the csv module, not split.
    """
    part = tmp_path / "N-part000.csv"
    part.write_text("n1\t'a'\t'N'\nn1\t'b'\t'N'\nn2\t'with\ttab'\t'N'\n")
    rows = scan_part(str(part), 3, 0, None)
    assert len(rows) == 3
    assert (rows[0]["k0"], rows[0]["k1"]) == (rows[1]["k0"], rows[1]["k1"])
    assert rows[0]["row"] != rows[1]["row"]
    assert (rows[2]["k0"], rows[2]["k1"]) != (rows[0]["k0"], rows[0]["k1"])
    edge = tmp_path / "E-part000.csv"
    edge.write_text("s1\t\tt1\t'E'\ns1\t\tt2\t'E'\n")
    e = scan_part(str(edge), 4, 0, 2)
    assert (e[0]["k0"], e[0]["k1"]) != (e[1]["k0"], e[1]["k1"])


def test_check_label_duplicates_counts_rows_beyond_the_first_directory(
    tmp_path: Path,
) -> None:
    order = [
        AdapterSlot(index=0, dataset_class="DA", adapter_class="AAdapter"),
        AdapterSlot(index=1, dataset_class="DB", adapter_class="BAdapter"),
        AdapterSlot(index=2, dataset_class="DC", adapter_class="CAdapter"),
    ]
    for slot in order:
        _write_dir(
            tmp_path,
            slot,
            nodes={"G": ["g1\t'x'\t'G'", f"g{slot.index + 10}\t'y'\t'G'"]},
            edges={},
        )
    files = discover_label_files(tmp_path, order)["G"]
    outcome, conflicts = check_label_duplicates(files, workers=2)
    assert (
        outcome.rows,
        outcome.duplicate_keys,
        outcome.duplicate_rows,
        outcome.conflicting_keys,
    ) == (6, 1, 2, 0)
    assert conflicts == [] and outcome.drop_rows == {}


def test_the_missing_command_exits_one_and_prints_the_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    order = _two_dirs(tmp_path, done_b=False)
    monkeypatch.setattr(
        ga, "adapter_order", lambda include_private=False, organisms=None: order
    )
    code = ga.main(
        ["missing", "--out-dir", str(tmp_path), "--out-name", "o", "--script", "s"]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "1 of 2 adapter directories carry no DONE" in out
    assert "  001 BAdapter" in out
    assert "sbatch --job-name=gen_adapter_standard --array=1%4" in out
    (tmp_path / "adapters" / order[1].dirname / DONE_FILENAME).write_text(
        _marker(order[1])
    )
    assert (
        ga.main(
            ["missing", "--out-dir", str(tmp_path), "--out-name", "o", "--script", "s"]
        )
        == 0
    )


def test_the_merge_command_checks_the_expected_adapter_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _two_dirs(tmp_path)
    with pytest.raises(SystemExit, match="the order has 2 adapters, expected 3"):
        ga.main(["merge", "--out-dir", str(tmp_path), "--expect", "3"])
    assert (
        ga.main(
            ["merge", "--out-dir", str(tmp_path), "--expect", "2", "--workers", "2"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Genome: 3 rows, 1 duplicate rows over 1 keys (node)" in out
    assert f"MERGED_CALL={tmp_path / IMPORT_CALL_FILENAME}" in out
    assert json.loads((tmp_path / "merge_report.json").read_text())["n_adapters"] == 2
