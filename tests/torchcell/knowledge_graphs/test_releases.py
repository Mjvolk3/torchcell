# tests/torchcell/knowledge_graphs/test_releases.py
# [[tests.torchcell.knowledge_graphs.test_releases]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/knowledge_graphs/test_releases.py
"""Release identity, version bumps, content hashes, diffs, the node round trip, and the
serving, git, and CLI paths against a scripted driver and a fake ``git`` on PATH.

The scripted DBMS (``_ScriptedDriver``) answers the exact Cypher strings the module sends:
``SHOW DATABASES`` lists ``system``, ``torchcell`` (default, aliases ``latest`` and
``pinned``, 7 nodes, 2 datasets, carrying the release node ``RELEASE``), ``neo4j``
(online, its ``Dataset`` count raises ``java.io.IOException``), and ``old`` (offline).
A session on ``latest`` or ``pinned`` answers as ``torchcell`` does, the way the real
DBMS resolves an alias. ``content_hashes_from_store`` sees experiment ids
``<dataset>-e2``, ``<dataset>-e1`` for each dataset, so the expected digest is the sha256
of ``DsA-e1`` and ``DsA-e2`` each followed by a newline, computed in the test with hashlib.

The fake ``git`` (a bash script under tmp_path prepended to PATH, argv appended to a log)
knows commit ``abc`` (on main, 1200 commits deep, dated 2026.09.17, 3 behind main) and
main (``deadbeef``, 2026.09.27, subject ``subject line``); every other sha exits 1.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

import torchcell.knowledge_graphs.releases as releases
from torchcell.knowledge_graphs.kg_manifest import (
    KgBuildManifest,
    KgDatasetEntry,
    KgEvent,
    load_manifest,
    save_manifest,
)
from torchcell.knowledge_graphs.releases import (
    DatasetDrift,
    KgRelease,
    ReleaseCompatibility,
    ReleaseDataset,
    ServedDatabase,
    _fault_text,
    behind_main,
    commit_date,
    commit_index,
    compatibility,
    content_hashes_from_csv,
    content_hashes_from_store,
    content_sha256,
    datasets,
    diff,
    format_table,
    list_databases,
    list_databases_bounded,
    main,
    main_build,
    next_version,
    package_label,
    read_release,
    release_from_manifest,
    release_id,
    resolve_database,
    stamp_manifest,
    status_rows,
    write_release,
)


def _manifest(
    datasets: dict[str, int], commit: str = "7715ee35d95c"
) -> KgBuildManifest:
    return KgBuildManifest(
        database="torchcell",
        store_host="gilahyper",
        neo4j_version="5.26.28",
        biocypher_version="0.15.2",
        torchcell_commit=commit,
        graph_schema={},
        cell_adapter_methods={},
        cell_adapter_table={},
        adapter_files={},
        datasets={
            name: KgDatasetEntry(
                dataset_class=name,
                loader_relpath=f"torchcell/datasets/{name}.py",
                adapter_files=[],
                closure={"Experiment": "aa", "Genotype": "bb"},
                n_experiments=n,
                biocypher_out="2026-09-16_00-44-53",
                import_mode="full",
                admitted_at="2026-09-17T20:36:32-05:00",
                torchcell_commit=commit,
            )
            for name, n in datasets.items()
        },
        events=[
            KgEvent(
                kind="bootstrap",
                at="2026-09-18T01:36:59+00:00",
                torchcell_commit=commit,
                datasets=sorted(datasets),
                biocypher_out="2026-09-16_00-44-53",
            )
        ],
        created_at="2026-09-18T01:36:59+00:00",
        updated_at="2026-09-18T01:36:59+00:00",
    )


def test_release_id_is_build_date_and_commit_prefix() -> None:
    assert release_id("2026-09-17T20:36:32-05:00", "7715ee35d95c535620b9") == (
        "2026.09.17-7715ee35"
    )


def test_next_version_bumps_major_on_full_and_minor_on_increment() -> None:
    assert next_version(None, "full") == "1.0"
    assert next_version("1.0", "incremental") == "1.1"
    assert next_version("1.3", "full") == "2.0"


def test_content_sha256_is_order_independent_and_matches_definition() -> None:
    ids = ["c", "a", "b"]
    expected = hashlib.sha256(b"a\nb\nc\n").hexdigest()
    assert content_sha256(ids) == expected
    assert content_sha256(reversed(ids)) == expected
    assert content_sha256(["a", "b"]) != expected


def test_content_hashes_from_csv_groups_experiment_ids_by_dataset(
    tmp_path: Path,
) -> None:
    (tmp_path / "ExperimentMemberOf-header.csv").write_text(
        ":START_ID\t:END_ID\t:TYPE\n", encoding="utf-8"
    )
    (tmp_path / "ExperimentMemberOf-part000.csv").write_text(
        "'e1'\t'DsA'\t'ExperimentMemberOf'\n'e2'\t'DsA'\t'ExperimentMemberOf'\n"
        "'e3'\t'DsB'\t'ExperimentMemberOf'\n",
        encoding="utf-8",
    )
    hashes = content_hashes_from_csv(tmp_path)
    assert hashes == {
        "DsA": content_sha256(["e1", "e2"]),
        "DsB": content_sha256(["e3"]),
    }


def test_stamp_manifest_then_release_round_trips_through_node_properties() -> None:
    manifest = _manifest({"DsA": 10, "DsB": 20})
    hashes = {"DsA": "a" * 64, "DsB": "b" * 64}
    stamp_manifest(
        manifest,
        kind="full",
        built_at="2026-09-17T20:36:32-05:00",
        content_hashes=hashes,
        previous_version=None,
        torchcell_version="1.2.0",
        torchcell_tag=None,
    )
    assert manifest.version == "1.0"
    assert manifest.release == "2026.09.17-7715ee35"
    assert manifest.datasets["DsA"].content_sha256 == "a" * 64
    assert (manifest.torchcell_version, manifest.torchcell_tag) == ("1.2.0", None)

    release = release_from_manifest(
        manifest,
        built_at="2026-09-17T20:36:32-05:00",
        content_hashes=hashes,
        n_nodes=99_723_455,
    )
    assert release.n_datasets == 2
    assert release.closures["DsB"] == {"Experiment": "aa", "Genotype": "bb"}
    assert (release.torchcell_version, release.torchcell_tag) == ("1.2.0", None)
    props = release.to_properties()
    assert isinstance(props["datasets_json"], str)
    assert (props["torchcell_version"], props["torchcell_tag"]) == ("1.2.0", None)
    assert KgRelease.from_properties(props) == release


def test_incremental_stamp_keeps_hashes_of_untouched_datasets() -> None:
    manifest = _manifest({"DsA": 10, "DsB": 20})
    stamp_manifest(
        manifest,
        kind="full",
        built_at="2026-09-17T20:36:32-05:00",
        content_hashes={"DsA": "a" * 64, "DsB": "b" * 64},
        previous_version=None,
        torchcell_version="1.2.0",
        torchcell_tag=None,
    )
    manifest.datasets["DsC"] = manifest.datasets["DsA"].model_copy(
        update={"dataset_class": "DsC", "content_sha256": None}
    )
    manifest.events.append(
        KgEvent(
            kind="incremental_admission",
            at="2026-09-30T00:00:00+00:00",
            torchcell_commit="abcdef0123456789",
            datasets=["DsC"],
            biocypher_out="2026-09-30_00-00-00",
        )
    )
    stamp_manifest(
        manifest,
        kind="incremental",
        built_at="2026-09-30T01:00:00+00:00",
        content_hashes={"DsC": "c" * 64},
        previous_version=manifest.version,
        torchcell_version="1.2.1",
        torchcell_tag="v1.2.1",
    )
    assert manifest.version == "1.1"
    assert manifest.release == "2026.09.30-abcdef01"
    assert (manifest.torchcell_version, manifest.torchcell_tag) == ("1.2.1", "v1.2.1")
    assert manifest.datasets["DsA"].content_sha256 == "a" * 64
    assert manifest.datasets["DsC"].content_sha256 == "c" * 64
    manifest.datasets["DsD"] = manifest.datasets["DsC"].model_copy(
        update={"dataset_class": "DsD", "content_sha256": None}
    )
    with pytest.raises(ValueError, match="DsD"):
        stamp_manifest(
            manifest,
            kind="incremental",
            built_at="2026-10-01T00:00:00+00:00",
            content_hashes={},
            previous_version=manifest.version,
            torchcell_version="1.2.1",
            torchcell_tag="v1.2.1",
        )


def test_release_from_manifest_refuses_a_missing_hash() -> None:
    manifest = _manifest({"DsA": 10})
    stamp_manifest(
        manifest,
        kind="full",
        built_at="2026-09-17T20:36:32-05:00",
        content_hashes={"DsA": "a" * 64},
        previous_version="1.2",
        torchcell_version="1.2.0",
        torchcell_tag=None,
    )
    assert manifest.version == "2.0"
    with pytest.raises(ValueError, match="no content hash"):
        release_from_manifest(manifest, built_at="x", content_hashes={}, n_nodes=None)


def test_release_from_manifest_and_stamp_refuse_an_unstamped_or_commitless_manifest() -> (
    None
):
    """The checks run in order: hashes, then the full-build commit, then version/release."""
    manifest = _manifest({"DsA": 10})
    hashes = {"DsA": "a" * 64}
    with pytest.raises(
        ValueError, match="the manifest carries no version/release; stamp it first"
    ):
        release_from_manifest(
            manifest, built_at="x", content_hashes=hashes, n_nodes=None
        )
    manifest.torchcell_commit = None
    with pytest.raises(ValueError, match="the manifest records no full-build commit"):
        release_from_manifest(
            manifest, built_at="x", content_hashes=hashes, n_nodes=None
        )
    with pytest.raises(ValueError, match="the manifest records no full-build commit"):
        stamp_manifest(
            manifest,
            kind="full",
            built_at="x",
            content_hashes=hashes,
            previous_version=None,
            torchcell_version="1.2.0",
            torchcell_tag=None,
        )


def _release(tag: str, hashes: dict[str, str]) -> KgRelease:
    return KgRelease(
        release=tag,
        version="1.0",
        torchcell_commit="7715ee35",
        built_at="2026-09-17",
        biocypher_out="2026-09-16_00-44-53",
        datasets={
            name: ReleaseDataset(dataset_class=name, n_experiments=1, content_sha256=h)
            for name, h in hashes.items()
        },
    )


def test_diff_names_unchanged_changed_added_removed() -> None:
    a = _release("2026.09.17-7715ee35", {"DsA": "1", "DsB": "2", "DsC": "3"})
    b = _release("2026.09.30-abcdef01", {"DsA": "1", "DsB": "9", "DsD": "4"})
    result = diff(a, b)
    assert result.unchanged == ["DsA"]
    assert result.changed == ["DsB"]
    assert result.added == ["DsD"]
    assert result.removed == ["DsC"]


def test_status_rows_report_faults_and_missing_release_nodes() -> None:
    healthy = ServedDatabase(
        name="torchcell",
        aliases=["latest", "pinned"],
        default=True,
        status="online",
        release=_release("2026.09.17-7715ee35", {"DsA": "1"}),
        n_nodes=99_723_455,
        n_datasets=51,
    )
    faulting = ServedDatabase(
        name="torchcell",
        aliases=[],
        default=True,
        status="online",
        n_datasets=None,
        fault="java.io.IOException: Input/output error",
    )
    bare = ServedDatabase(
        name="neo4j", aliases=[], default=False, status="online", n_datasets=35
    )
    rows = status_rows("gilahyper", [healthy], None) + status_rows(
        "radiant", [faulting, bare], None
    )
    assert rows[0][:5] == [
        "gilahyper",
        "torchcell [default]",
        "1.0",
        "2026.09.17-7715ee35",
        "-",
    ]
    assert rows[0][8] == "99,723,455" and rows[0][9] == "latest,pinned"
    assert rows[1][10].startswith("faulting (java.io.IOException")
    assert rows[2][10] == "online (no release node)"
    text = format_table(rows)
    assert text.splitlines()[0].startswith("HOST")
    assert json.dumps(rows)  # plain strings only


def test_pkg_column_is_the_tag_or_the_version_marked_untagged() -> None:
    """A release from a tagged checkout shows the tag; an untagged build shows the
    version and says so; a node written before the spine shows ``-``.
    """
    tagged = _release("2026.09.17-7715ee35", {"DsA": "1"}).model_copy(
        update={"torchcell_version": "1.2.0", "torchcell_tag": "v1.2.0"}
    )
    untagged = tagged.model_copy(update={"torchcell_tag": None})
    assert package_label(tagged) == "v1.2.0"
    assert package_label(untagged) == "1.2.0 (untagged)"
    assert package_label(_release("x", {})) == "-"
    assert package_label(None) == "-"
    served = [
        ServedDatabase(
            name="torchcell",
            aliases=[],
            default=False,
            status="online",
            release=release,
            n_datasets=1,
        )
        for release in (tagged, untagged)
    ]
    assert [row[4] for row in status_rows("gh", served, None)] == [
        "v1.2.0",
        "1.2.0 (untagged)",
    ]
    assert format_table(status_rows("gh", served, None)).splitlines()[0] == (
        "HOST  DATABASE   VERSION  RELEASE              PKG               COMMIT#  "
        "DATE  DATASETS  NODES  ALIASES  STATUS"
    )


def test_properties_round_trip_the_package_version_and_tag() -> None:
    release = RELEASE.model_copy(
        update={"torchcell_version": "1.2.0", "torchcell_tag": "v1.2.0"}
    )
    props = release.to_properties()
    assert (props["torchcell_version"], props["torchcell_tag"]) == ("1.2.0", "v1.2.0")
    assert KgRelease.from_properties(props) == release
    assert KgRelease.from_properties(RELEASE.to_properties()) == RELEASE


# --------------------------------------------------------------------------- the node


RELEASE = KgRelease(
    release="2026.09.17-7715ee35",
    version="1.0",
    torchcell_commit="7715ee35d95c",
    built_at="2026-09-17T20:36:32-05:00",
    biocypher_out="2026-09-16_00-44-53",
    datasets={
        "DsB": ReleaseDataset(
            dataset_class="DsB", n_experiments=2, content_sha256="b" * 64
        ),
        "DsA": ReleaseDataset(
            dataset_class="DsA", n_experiments=1, content_sha256="a" * 64
        ),
    },
    closures={"DsA": {"Experiment": "aa"}},
)


def test_to_properties_flattens_the_maps_to_compact_sorted_json() -> None:
    """Datasets are sorted by name and serialized without spaces; None fields stay None."""
    assert RELEASE.to_properties() == {
        "release": "2026.09.17-7715ee35",
        "version": "1.0",
        "torchcell_commit": "7715ee35d95c",
        "torchcell_version": None,
        "torchcell_tag": None,
        "built_at": "2026-09-17T20:36:32-05:00",
        "biocypher_out": "2026-09-16_00-44-53",
        "neo4j_version": None,
        "n_nodes": None,
        "n_datasets": 2,
        "datasets_json": (
            '{"DsA":{"dataset_class":"DsA","n_experiments":1,"content_sha256":"'
            + "a" * 64
            + '"},"DsB":{"dataset_class":"DsB","n_experiments":2,"content_sha256":"'
            + "b" * 64
            + '"}}'
        ),
        "closures_json": '{"DsA":{"Experiment":"aa"}}',
    }


def test_from_properties_defaults_the_optional_fields_of_a_pre_closure_node() -> None:
    """A node written before closures were recorded has no closures_json: empty closures."""
    release = KgRelease.from_properties(
        {
            "release": "x",
            "version": "1.0",
            "torchcell_commit": "c",
            "built_at": "b",
            "biocypher_out": "o",
            "datasets_json": "{}",
        }
    )
    assert release == KgRelease(
        release="x",
        version="1.0",
        torchcell_commit="c",
        built_at="b",
        biocypher_out="o",
        datasets={},
    )
    assert release.n_datasets == 0


def test_fault_text_keeps_the_first_line_capped_at_120_chars_or_names_the_type() -> (
    None
):
    """Whitespace is stripped, only the first line survives, 120 chars at most."""
    assert _fault_text(ValueError("  first line\nsecond  ")) == "first line"
    assert _fault_text(ValueError("")) == "ValueError"
    assert _fault_text(ValueError("x" * 200)) == "x" * 120


# --------------------------------------------------------------------------- scripted DBMS


class _Result:
    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows

    def values(self) -> list[Any]:
        return self.rows

    def single(self) -> Any:
        return self.rows[0]

    def consume(self) -> None:
        return None

    def __iter__(self) -> Any:
        return iter(self.rows)


class _Session:
    def __init__(
        self, driver: _ScriptedDriver, database: str | None = None, **_: Any
    ) -> None:
        self.driver = driver
        self.database = database

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def run(self, query: str, **params: Any) -> _Result:
        self.driver.calls.append((self.database, query, params))
        return self.driver.answer(self.database, query, params)


class _ScriptedDriver:
    """Answers the module's Cypher for the DBMS the module docstring describes."""

    show_databases = [
        ["system", [], False, "online", "system"],
        ["torchcell", ["latest", "pinned"], True, "online", "standard"],
        ["neo4j", [], False, "online", "standard"],
        ["old", [], False, "offline", "standard"],
    ]

    def __init__(self, uri: str, auth: tuple[str, str], **kwargs: Any) -> None:
        self.uri = uri
        self.auth = auth
        self.kwargs = kwargs
        self.calls: list[tuple[str | None, str, dict[str, Any]]] = []
        self.closed = 0
        # databases whose KgRelease node exists; a test may add "neo4j" or duplicate rows
        self.release_rows: dict[str, list[list[Any]]] = {
            "torchcell": [[RELEASE.to_properties()]]
        }

    def session(self, **kwargs: Any) -> _Session:
        return _Session(self, **kwargs)

    def close(self) -> None:
        self.closed += 1

    def answer(
        self, database: str | None, query: str, params: dict[str, Any]
    ) -> _Result:
        if database in ("latest", "pinned"):
            database = "torchcell"
        if query.startswith("SHOW DATABASES"):
            return _Result(self.show_databases)
        if query == "MATCH (n) RETURN count(n)":
            return _Result([[7]])
        if query == "MATCH (d:Dataset) RETURN count(d)":
            if database == "neo4j":
                raise OSError("java.io.IOException: Input/output error\nat Foo.java")
            return _Result([[2]])
        if query == "MATCH (d:Dataset) RETURN d.id LIMIT 1":
            return _Result([])
        if query == "MATCH (r:KgRelease) RETURN r":
            return _Result(self.release_rows.get(database or "", []))
        if query == "MATCH (r:KgRelease) RETURN r.release, r.version":
            rows = self.release_rows.get(database or "", [])
            return _Result([[r[0]["release"], r[0]["version"]] for r in rows[:1]])
        if query == "MATCH (d:Dataset) RETURN d.id ORDER BY d.id":
            return _Result([["DsA"], ["DsB"]])
        if query.startswith("MATCH (d:Dataset {id: $id})"):
            return _Result([{"id": f"{params['id']}-e2"}, {"id": f"{params['id']}-e1"}])
        if query.startswith("MERGE (r:KgRelease {singleton: true})"):
            return _Result([])
        raise AssertionError(f"unscripted query: {query}")


@pytest.fixture
def drivers(monkeypatch: pytest.MonkeyPatch) -> list[_ScriptedDriver]:
    """Install the scripted driver as ``neo4j.GraphDatabase.driver``; returns every instance."""
    made: list[_ScriptedDriver] = []

    def driver(uri: str, auth: tuple[str, str], **kwargs: Any) -> _ScriptedDriver:
        made.append(_ScriptedDriver(uri, auth, **kwargs))
        return made[-1]

    monkeypatch.setattr("neo4j.GraphDatabase.driver", driver)
    return made


BOLT = "bolt://x:7687"


def test_list_databases_probes_online_stores_and_reports_the_faulting_one(
    drivers: list[_ScriptedDriver],
) -> None:
    """torchcell: counts and release; neo4j: the IOException's first line as fault and no
    dataset count; old: offline, never probed; system skipped; driver closed once.
    """
    served = list_databases(BOLT, "u", "p")
    assert served == [
        ServedDatabase(
            name="torchcell",
            aliases=["latest", "pinned"],
            default=True,
            status="online",
            release=RELEASE,
            n_nodes=7,
            n_datasets=2,
        ),
        ServedDatabase(
            name="neo4j",
            aliases=[],
            default=False,
            status="online",
            n_nodes=7,
            fault="java.io.IOException: Input/output error",
        ),
        ServedDatabase(name="old", aliases=[], default=False, status="offline"),
    ]
    driver = drivers[0]
    assert (driver.uri, driver.auth, driver.kwargs) == (
        BOLT,
        ("u", "p"),
        {"connection_timeout": 10},
    )
    assert driver.closed == 1
    assert [(db, q) for db, q, _ in driver.calls] == [
        (
            "system",
            "SHOW DATABASES YIELD name, aliases, default, currentStatus, type "
            "RETURN name, aliases, default, currentStatus, type",
        ),
        ("torchcell", "MATCH (n) RETURN count(n)"),
        ("torchcell", "MATCH (d:Dataset) RETURN count(d)"),
        ("torchcell", "MATCH (d:Dataset) RETURN d.id LIMIT 1"),
        ("torchcell", "MATCH (r:KgRelease) RETURN r"),
        ("neo4j", "MATCH (n) RETURN count(n)"),
        ("neo4j", "MATCH (d:Dataset) RETURN count(d)"),
    ]
    unprobed = list_databases(BOLT, "u", "p", probe=False)
    assert [(db.name, db.n_nodes, db.release) for db in unprobed] == [
        ("torchcell", None, None),
        ("neo4j", None, None),
        ("old", None, None),
    ]


def test_list_databases_reports_a_dbms_whose_system_database_faults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When SHOW DATABASES itself raises, the answer is one ``(dbms)`` row with the fault."""

    class Broken(_ScriptedDriver):
        def answer(
            self, database: str | None, query: str, params: dict[str, Any]
        ) -> _Result:
            raise ConnectionError("Connection refused\nretrying")

    made: list[Broken] = []

    def driver(uri: str, auth: tuple[str, str], **kwargs: Any) -> Broken:
        made.append(Broken(uri, auth, **kwargs))
        return made[-1]

    monkeypatch.setattr("neo4j.GraphDatabase.driver", driver)
    assert list_databases(BOLT, "u", "p") == [
        ServedDatabase(
            name="(dbms)",
            aliases=[],
            default=False,
            status="?",
            fault="Connection refused",
        )
    ]
    assert made[0].closed == 1


def test_resolve_database_passes_aliases_and_names_through_and_looks_up_releases(
    drivers: list[_ScriptedDriver],
) -> None:
    """Alias and physical names return unchanged; a release id or version resolves to the
    database carrying the node; an unknown version raises with the inventory.
    """
    assert resolve_database("latest", BOLT, "u", "p") == "latest"
    assert resolve_database("neo4j", BOLT, "u", "p") == "neo4j"
    assert resolve_database("2026.09.17-7715ee35", BOLT, "u", "p") == "torchcell"
    assert resolve_database("1.0", BOLT, "u", "p") == "torchcell"
    # the lookup skipped the offline database: only the two online stores were asked
    assert [db for db, q, _ in drivers[-1].calls] == ["torchcell", "neo4j"]
    with pytest.raises(
        LookupError,
        match=(
            r"no database at bolt://x:7687 serves version '9\.9'; "
            r"databases \['neo4j', 'old', 'torchcell'\], aliases \['latest', 'pinned'\]"
        ),
    ):
        resolve_database("9.9", BOLT, "u", "p")


def test_resolve_database_refuses_a_version_served_by_two_databases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two online stores carrying the same release node is ambiguous."""

    class Twice(_ScriptedDriver):
        def __init__(self, uri: str, auth: tuple[str, str], **kwargs: Any) -> None:
            super().__init__(uri, auth, **kwargs)
            self.release_rows["neo4j"] = self.release_rows["torchcell"]

    monkeypatch.setattr("neo4j.GraphDatabase.driver", Twice)
    with pytest.raises(
        LookupError,
        match=r"version '1\.0' is served by several databases: \['torchcell', 'neo4j'\]",
    ):
        resolve_database("1.0", BOLT, "u", "p")


def test_read_release_returns_the_node_none_or_refuses_two_nodes(
    monkeypatch: pytest.MonkeyPatch, drivers: list[_ScriptedDriver]
) -> None:
    """Torchcell carries RELEASE; neo4j has no node; a duplicated node is an error."""
    assert read_release(BOLT, "u", "p", "torchcell") == RELEASE
    assert read_release(BOLT, "u", "p", "neo4j") is None
    assert [d.closed for d in drivers] == [1, 1]

    class Doubled(_ScriptedDriver):
        def __init__(self, uri: str, auth: tuple[str, str], **kwargs: Any) -> None:
            super().__init__(uri, auth, **kwargs)
            self.release_rows["torchcell"] = self.release_rows["torchcell"] * 2

    monkeypatch.setattr("neo4j.GraphDatabase.driver", Doubled)
    with pytest.raises(ValueError, match="torchcell carries 2 KgRelease nodes"):
        read_release(BOLT, "u", "p", "torchcell")


def test_write_release_merges_the_singleton_node_with_the_flat_properties(
    drivers: list[_ScriptedDriver],
) -> None:
    """One MERGE on the named database with ``to_properties()`` as ``$props``."""
    write_release(BOLT, "u", "p", "torchcell", RELEASE)
    assert drivers[0].calls == [
        (
            "torchcell",
            "MERGE (r:KgRelease {singleton: true}) SET r += $props",
            {"props": RELEASE.to_properties()},
        )
    ]
    assert drivers[0].closed == 1


def test_datasets_lists_the_release_datasets_sorted_or_refuses_a_bare_store(
    drivers: list[_ScriptedDriver],
) -> None:
    """``latest`` resolves through the alias to RELEASE's datasets; neo4j has no node."""
    assert datasets("latest", BOLT, "u", "p") == [
        ReleaseDataset(dataset_class="DsA", n_experiments=1, content_sha256="a" * 64),
        ReleaseDataset(dataset_class="DsB", n_experiments=2, content_sha256="b" * 64),
    ]
    with pytest.raises(LookupError, match="neo4j carries no KgRelease node"):
        datasets("neo4j", BOLT, "u", "p")
    # the CLI's _release_for raises the same way for a store without a node
    with pytest.raises(LookupError, match="neo4j carries no KgRelease node"):
        main(["--uri", BOLT, "diff", "neo4j", "latest"])


def test_content_hashes_from_store_hashes_each_dataset_ids_in_sorted_order(
    drivers: list[_ScriptedDriver],
) -> None:
    """Ids arrive e2 then e1; the digest is over the sorted, newline-terminated ids."""
    assert content_hashes_from_store(BOLT, "u", "p", "torchcell") == {
        "DsA": hashlib.sha256(b"DsA-e1\nDsA-e2\n").hexdigest(),
        "DsB": hashlib.sha256(b"DsB-e1\nDsB-e2\n").hexdigest(),
    }
    assert drivers[0].calls[0] == (
        "torchcell",
        "MATCH (d:Dataset) RETURN d.id ORDER BY d.id",
        {},
    )
    assert drivers[0].calls[1][2] == {"id": "DsA"}


def test_list_databases_bounded_turns_a_refused_connection_or_a_hang_into_one_row(
    monkeypatch: pytest.MonkeyPatch, drivers: list[_ScriptedDriver]
) -> None:
    """The bounded call passes a healthy answer through and maps the two failure modes."""
    assert [db.name for db in list_databases_bounded(BOLT, "u", "p", timeout_s=5)] == [
        "torchcell",
        "neo4j",
        "old",
    ]

    def refused(*_: Any, **__: Any) -> list[ServedDatabase]:
        raise ConnectionRefusedError("[Errno 111] Connection refused")

    monkeypatch.setattr(releases, "list_databases", refused)
    assert list_databases_bounded(BOLT, "u", "p", timeout_s=5) == [
        ServedDatabase(
            name="(dbms)",
            aliases=[],
            default=False,
            status="?",
            fault="[Errno 111] Connection refused",
        )
    ]

    def hangs(*_: Any, **__: Any) -> list[ServedDatabase]:
        time.sleep(0.5)
        return []

    monkeypatch.setattr(releases, "list_databases", hangs)
    assert list_databases_bounded(BOLT, "u", "p", timeout_s=0.05) == [
        ServedDatabase(
            name="(dbms)",
            aliases=[],
            default=False,
            status="?",
            fault="no answer within 0.05s",
        )
    ]


# --------------------------------------------------------------------------- compatibility


class _Surface:
    fingerprints = {"Experiment": "aa", "Genotype": "gg"}


def test_compatibility_names_compatible_drifted_and_unchecked_datasets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """DsA's closure matches the surface; DsB has no recorded closure; DsC drifts on both."""
    monkeypatch.setattr(releases, "surface_in_worktree", lambda root: _Surface())
    release = RELEASE.model_copy(
        update={
            "datasets": {
                **RELEASE.datasets,
                "DsC": ReleaseDataset(
                    dataset_class="DsC", n_experiments=3, content_sha256="c" * 64
                ),
            },
            "closures": {
                "DsA": {"Experiment": "aa"},
                "DsC": {"Experiment": "zz", "Genotype": "yy"},
            },
        }
    )
    report = compatibility(release, tmp_path)
    assert report == ReleaseCompatibility(
        release="2026.09.17-7715ee35",
        torchcell_commit="7715ee35d95c",
        compatible=["DsA"],
        drifted=[
            DatasetDrift(
                dataset_class="DsC", changed_symbols=["Experiment", "Genotype"]
            )
        ],
        unchecked=["DsB"],
    )
    assert report.ok is False
    assert compatibility(RELEASE, tmp_path).ok is True


# --------------------------------------------------------------------------- git columns


_FAKE_GIT = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FAKE_GIT_LOG"
case "$*" in
  *'cat-file -e abc^{commit}') exit 0;;
  *'cat-file -e same^{commit}') exit 0;;
  *'cat-file -e ahead^{commit}') exit 0;;
  *'cat-file -e '*) exit 1;;
  *'merge-base --is-ancestor abc main') exit 0;;
  *'rev-list --count abc..main') echo 3;;
  *'rev-list --count main..abc') echo 0;;
  *'rev-list --count same..main') echo 0;;
  *'rev-list --count main..same') echo 0;;
  *'rev-list --count ahead..main') echo 0;;
  *'rev-list --count main..ahead') echo 2;;
  *'rev-list --count abc') echo 1200;;
  *'--format=%cd --date=format:%Y.%m.%d abc') echo 2026.09.17;;
  *'rev-parse --short=8 main') echo deadbeef;;
  *'--format=%cd --date=format:%Y.%m.%d main') echo 2026.09.27;;
  *'log -1 --format=%s main') echo 'subject line';;
  *) exit 1;;
esac
"""


@pytest.fixture
def fake_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A ``git`` on PATH that answers for commit ``abc`` and main; returns its argv log."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    script = bindir / "git"
    script.write_text(_FAKE_GIT, encoding="utf-8")
    script.chmod(0o755)
    log = tmp_path / "git.log"
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_GIT_LOG", str(log))
    return log


def test_git_columns_come_from_the_git_cli_on_path(
    tmp_path: Path, fake_git: Path
) -> None:
    """Known commit: index 1200, date 2026.09.17, 3 behind; unknown: the labeled fallbacks."""
    repo = tmp_path / "repo"
    assert commit_index(repo, "abc") == "1200"
    assert commit_index(repo, "zzz") == "(unknown)"
    assert commit_date(repo, "abc") == "2026.09.17"
    assert commit_date(repo, "zzz") == "(unknown)"
    assert main_build(repo) == ("2026.09.27-deadbeef", "subject line")
    assert behind_main(repo, "abc") == "3 behind main"
    assert behind_main(repo, "zzz") == "(commit zzz not in local history)"
    assert behind_main(repo, "same") == "up to date with main"
    assert behind_main(repo, "ahead") == "0 behind, 2 ahead of main"
    assert fake_git.read_text(encoding="utf-8").splitlines() == [
        f"-C {repo} cat-file -e abc^{{commit}}",
        f"-C {repo} merge-base --is-ancestor abc main",
        f"-C {repo} rev-list --count abc",
        f"-C {repo} cat-file -e zzz^{{commit}}",
        f"-C {repo} show -s --format=%cd --date=format:%Y.%m.%d abc",
        f"-C {repo} show -s --format=%cd --date=format:%Y.%m.%d zzz",
        f"-C {repo} rev-parse --short=8 main",
        f"-C {repo} show -s --format=%cd --date=format:%Y.%m.%d main",
        f"-C {repo} log -1 --format=%s main",
        f"-C {repo} cat-file -e abc^{{commit}}",
        f"-C {repo} rev-list --count abc..main",
        f"-C {repo} rev-list --count main..abc",
        f"-C {repo} cat-file -e zzz^{{commit}}",
        f"-C {repo} cat-file -e same^{{commit}}",
        f"-C {repo} rev-list --count same..main",
        f"-C {repo} rev-list --count main..same",
        f"-C {repo} cat-file -e ahead^{{commit}}",
        f"-C {repo} rev-list --count ahead..main",
        f"-C {repo} rev-list --count main..ahead",
    ]


def test_commit_index_labels_a_commit_off_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A commit git knows but that is not an ancestor of main is ``(off-main)``."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    script = bindir / "git"
    script.write_text(
        '#!/usr/bin/env bash\ncase "$*" in\n  *"cat-file -e"*) exit 0;;\n  *) exit 1;;\nesac\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    assert commit_index(tmp_path, "abc") == "(off-main)"


def test_status_rows_fill_the_git_columns_when_a_repo_root_is_given(
    tmp_path: Path, fake_git: Path
) -> None:
    """COMMIT# and DATE come from the release commit; ``abc`` is what the fake git knows."""
    served = ServedDatabase(
        name="torchcell",
        aliases=["latest"],
        default=False,
        status="online",
        release=RELEASE.model_copy(update={"torchcell_commit": "abc"}),
        n_nodes=7,
        n_datasets=2,
    )
    assert status_rows("gh", [served], tmp_path) == [
        [
            "gh",
            "torchcell",
            "1.0",
            "2026.09.17-7715ee35",
            "-",
            "1200",
            "2026.09.17",
            "2",
            "7",
            "latest",
            "online",
        ]
    ]


# --------------------------------------------------------------------------- CLI

STATUS_TABLE = [
    "HOST  DATABASE             VERSION  RELEASE              PKG  COMMIT#  DATE  DATASETS  NODES  ALIASES        STATUS",
    "gh    torchcell [default]  1.0      2026.09.17-7715ee35  -    -        -     2         7      latest,pinned  online",
    "gh    neo4j                -        -                    -    -        -     -         7      -              faulting (java.io.IOException: Input/output error)",
    "gh    old                  -        -                    -    -        -     -         -      -              offline",
]


def test_cli_status_prints_the_aligned_table_with_or_without_the_header(
    drivers: list[_ScriptedDriver], capsys: pytest.CaptureFixture[str]
) -> None:
    """``status --label gh`` renders STATUS_TABLE; ``--no-header`` drops its first line."""
    common = [
        "--uri",
        BOLT,
        "--user",
        "u",
        "--password",
        "p",
        "status",
        "--label",
        "gh",
    ]
    assert main(common) == 0
    assert capsys.readouterr().out.splitlines() == STATUS_TABLE
    assert main([*common, "--no-header"]) == 0
    assert capsys.readouterr().out.splitlines() == STATUS_TABLE[1:]
    assert (drivers[0].uri, drivers[0].auth) == (BOLT, ("u", "p"))


def test_cli_status_json_lists_every_host_from_the_host_specs(
    drivers: list[_ScriptedDriver], capsys: pytest.CaptureFixture[str]
) -> None:
    """``--host LABEL=URI|USER|PASSWORD`` overrides the connection; missing parts inherit."""
    argv = [
        "--user",
        "u0",
        "--password",
        "p0",
        "status",
        "--host",
        "gh=bolt://x:7687|u|p",
        "--host",
        "rad=bolt://y:7687",
        "--json",
    ]
    assert main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert list(report) == ["gh", "rad"]
    assert [db["name"] for db in report["gh"]] == ["torchcell", "neo4j", "old"]
    assert report["gh"][0]["release"] == RELEASE.model_dump()
    assert report["rad"][1]["fault"] == "java.io.IOException: Input/output error"
    assert [(d.uri, d.auth) for d in drivers] == [
        ("bolt://x:7687", ("u", "p")),
        ("bolt://y:7687", ("u0", "p0")),
    ]


def test_cli_datasets_prints_tab_rows_or_json(
    drivers: list[_ScriptedDriver], capsys: pytest.CaptureFixture[str]
) -> None:
    """One ``class<TAB>n<TAB>sha`` line per dataset; ``--json`` dumps the model list."""
    assert main(["--uri", BOLT, "datasets"]) == 0
    assert capsys.readouterr().out == f"DsA\t1\t{'a' * 64}\nDsB\t2\t{'b' * 64}\n"
    assert main(["--uri", BOLT, "datasets", "--version", "1.0", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == [
        {"dataset_class": "DsA", "n_experiments": 1, "content_sha256": "a" * 64},
        {"dataset_class": "DsB", "n_experiments": 2, "content_sha256": "b" * 64},
    ]


def test_cli_diff_prints_the_four_buckets_or_json(
    drivers: list[_ScriptedDriver], capsys: pytest.CaptureFixture[str]
) -> None:
    """Diffing a release against itself: both datasets unchanged, the other buckets ``-``."""
    assert main(["--uri", BOLT, "diff", "latest", "1.0"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "2026.09.17-7715ee35 -> 2026.09.17-7715ee35",
        "  unchanged (2): DsA, DsB",
        "  changed (0): -",
        "  added (0): -",
        "  removed (0): -",
    ]
    assert main(["--uri", BOLT, "diff", "latest", "1.0", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "from_release": "2026.09.17-7715ee35",
        "to_release": "2026.09.17-7715ee35",
        "unchanged": ["DsA", "DsB"],
        "changed": [],
        "added": [],
        "removed": [],
    }


def test_cli_hashes_requires_exactly_one_source_and_writes_the_json(
    tmp_path: Path, drivers: list[_ScriptedDriver], capsys: pytest.CaptureFixture[str]
) -> None:
    """Neither or both of ``--csv-dir``/``--database`` is a usage error (exit 2); the store
    path writes ``Dataset.id -> sha256`` with indent 1.
    """
    out = tmp_path / "hashes.json"
    with pytest.raises(SystemExit) as excinfo:
        main(["hashes", "--output", str(out)])
    assert excinfo.value.code == 2
    assert "give exactly one of --csv-dir or --database" in capsys.readouterr().err
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "hashes",
                "--csv-dir",
                str(tmp_path),
                "--database",
                "torchcell",
                "--output",
                str(out),
            ]
        )
    assert excinfo.value.code == 2
    capsys.readouterr()
    assert (
        main(["--uri", BOLT, "hashes", "--database", "torchcell", "--output", str(out)])
        == 0
    )
    assert capsys.readouterr().out == f"2 content hashes -> {out}\n"
    expected = {
        "DsA": hashlib.sha256(b"DsA-e1\nDsA-e2\n").hexdigest(),
        "DsB": hashlib.sha256(b"DsB-e1\nDsB-e2\n").hexdigest(),
    }
    assert out.read_text(encoding="utf-8") == json.dumps(expected, indent=1)


def test_cli_hashes_from_csv_dir_needs_no_connection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CSV path never opens a driver (the conftest guard would fail it if it did)."""
    (tmp_path / "ExperimentMemberOf-header.csv").write_text(
        ":START_ID\t:END_ID\t:TYPE\n", encoding="utf-8"
    )
    (tmp_path / "ExperimentMemberOf-part000.csv").write_text(
        "'e1'\t'DsA'\t'ExperimentMemberOf'\n", encoding="utf-8"
    )
    out = tmp_path / "h.json"
    assert main(["hashes", "--csv-dir", str(tmp_path), "--output", str(out)]) == 0
    assert capsys.readouterr().out == f"1 content hashes -> {out}\n"
    assert json.loads(out.read_text(encoding="utf-8")) == {
        "DsA": hashlib.sha256(b"e1\n").hexdigest()
    }


def test_cli_stamp_versions_the_manifest_file_in_place(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``stamp --kind full`` writes version 1.0, the release id, the hashes, and the
    package version and tag of the checkout (``checkout_package_version`` on ``--repo``,
    or the imported checkout) back.
    """
    asked: list[Path] = []

    def checkout(root: Path) -> tuple[str, str | None]:
        asked.append(root)
        return "1.2.0", "v1.2.0"

    monkeypatch.setattr(releases, "checkout_package_version", checkout)
    manifest_path = tmp_path / "kg_manifest.json"
    save_manifest(_manifest({"DsA": 10, "DsB": 20}), manifest_path)
    hashes_path = tmp_path / "hashes.json"
    hashes_path.write_text(
        json.dumps({"DsA": "a" * 64, "DsB": "b" * 64}), encoding="utf-8"
    )
    assert (
        main(
            [
                "stamp",
                "--manifest",
                str(manifest_path),
                "--kind",
                "full",
                "--built-at",
                "2026-09-17T20:36:32-05:00",
                "--hashes",
                str(hashes_path),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == (
        f"{manifest_path}: version 1.0, release 2026.09.17-7715ee35, "
        "torchcell 1.2.0 (v1.2.0)\n"
    )
    stamped = load_manifest(manifest_path)
    assert (stamped.version, stamped.release) == ("1.0", "2026.09.17-7715ee35")
    assert (stamped.torchcell_version, stamped.torchcell_tag) == ("1.2.0", "v1.2.0")
    assert {n: e.content_sha256 for n, e in stamped.datasets.items()} == {
        "DsA": "a" * 64,
        "DsB": "b" * 64,
    }
    assert asked == [releases.package_checkout()]
    assert (
        main(
            [
                "--repo",
                str(tmp_path),
                "stamp",
                "--manifest",
                str(manifest_path),
                "--kind",
                "incremental",
                "--built-at",
                "2026-09-30T00:00:00+00:00",
                "--hashes",
                str(hashes_path),
                "--previous-version",
                "1.0",
            ]
        )
        == 0
    )
    assert asked[-1] == tmp_path.resolve()
    capsys.readouterr()


def test_cli_snapshot_writes_the_committed_files_and_bootstraps_the_version(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stamped manifest becomes ``database/releases/<release>.json`` and its closures
    under ``--repo-root``; ``--torchcell-version`` fills a pre-spine manifest and is
    refused for one that records its version.
    """
    from torchcell.knowledge_graphs.release_snapshot import load_closures, load_snapshot

    manifest = _manifest({"DsA": 10, "DsB": 20})
    stamp_manifest(
        manifest,
        kind="full",
        built_at="2026-09-17T20:36:32-05:00",
        content_hashes={"DsA": "a" * 64, "DsB": "b" * 64},
        previous_version=None,
        torchcell_version="1.2.0",
        torchcell_tag=None,
    )
    manifest_path = tmp_path / "kg_manifest.json"
    save_manifest(manifest, manifest_path)
    argv = [
        "snapshot",
        "--manifest",
        str(manifest_path),
        "--n-nodes",
        "99",
        "--built-at",
        "2026-09-17T21:00:00-05:00",
        "--repo-root",
        str(tmp_path),
    ]
    assert main(argv) == 0
    composite = hashlib.sha256(("a" * 64 + "\n" + "b" * 64 + "\n").encode()).hexdigest()
    snapshot_path = tmp_path / "database" / "releases" / "2026.09.17-7715ee35.json"
    assert capsys.readouterr().out == (
        f"2026.09.17-7715ee35: torchcell 1.2.0 (untagged), composite {composite} -> "
        f"{snapshot_path}, {tmp_path / 'database' / 'releases' / '2026.09.17-7715ee35.closures.json'}\n"
    )
    snapshot = load_snapshot(snapshot_path)
    assert (snapshot.n_nodes, snapshot.built_at, snapshot.torchcell_version) == (
        99,
        "2026-09-17T21:00:00-05:00",
        "1.2.0",
    )
    assert load_closures(tmp_path, "2026.09.17-7715ee35") == {
        "DsA": {"Experiment": "aa", "Genotype": "bb"},
        "DsB": {"Experiment": "aa", "Genotype": "bb"},
    }
    with pytest.raises(ValueError, match="already records torchcell_version 1.2.0"):
        main([*argv, "--torchcell-version", "9.9.9"])
    manifest.torchcell_version = None
    save_manifest(manifest, manifest_path)
    assert (
        main([*argv, "--torchcell-version", "1.2.0", "--torchcell-tag", "v1.2.0"]) == 0
    )
    capsys.readouterr()
    snapshot = load_snapshot(snapshot_path)
    assert (snapshot.torchcell_version, snapshot.torchcell_tag) == ("1.2.0", "v1.2.0")
    assert snapshot.events[-1].note == (
        "bootstrapped: torchcell_version 1.2.0 and torchcell_tag v1.2.0 were supplied "
        "to `releases snapshot --torchcell-version` because the manifest predates the "
        "versioning spine"
    )


def test_cli_write_node_merges_the_release_built_from_a_stamped_manifest(
    tmp_path: Path, drivers: list[_ScriptedDriver], capsys: pytest.CaptureFixture[str]
) -> None:
    """A stamped manifest becomes one MERGE with its release properties; an unstamped one
    exits with the stamp-first message.
    """
    manifest = _manifest({"DsA": 10, "DsB": 20})
    stamp_manifest(
        manifest,
        kind="full",
        built_at="2026-09-17T20:36:32-05:00",
        content_hashes={"DsA": "a" * 64, "DsB": "b" * 64},
        previous_version=None,
        torchcell_version="1.2.0",
        torchcell_tag=None,
    )
    manifest_path = tmp_path / "kg_manifest.json"
    save_manifest(manifest, manifest_path)
    argv = [
        "--uri",
        BOLT,
        "write-node",
        "--manifest",
        str(manifest_path),
        "--database",
        "torchcell",
        "--built-at",
        "2026-09-17T21:00:00-05:00",
        "--n-nodes",
        "99",
    ]
    assert main(argv) == 0
    assert (
        capsys.readouterr().out
        == "torchcell: KgRelease 2026.09.17-7715ee35 (version 1.0)\n"
    )
    expected = release_from_manifest(
        manifest,
        built_at="2026-09-17T21:00:00-05:00",
        content_hashes={"DsA": "a" * 64, "DsB": "b" * 64},
        n_nodes=99,
    )
    assert drivers[0].calls == [
        (
            "torchcell",
            "MERGE (r:KgRelease {singleton: true}) SET r += $props",
            {"props": expected.to_properties()},
        )
    ]
    assert expected.n_nodes == 99 and expected.neo4j_version == "5.26.28"
    save_manifest(_manifest({"DsA": 10}), manifest_path)
    with pytest.raises(
        SystemExit, match="manifest has datasets without content_sha256; stamp it first"
    ):
        main(argv)


def test_cli_compat_requires_a_repo_and_exits_one_on_drift(
    tmp_path: Path,
    drivers: list[_ScriptedDriver],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No ``--repo`` is a usage error; RELEASE (DsA compatible, DsB unchecked) exits 0; a
    drifted closure prints the symbols and exits 1.
    """
    with pytest.raises(SystemExit) as excinfo:
        main(["--uri", BOLT, "compat"])
    assert excinfo.value.code == 2
    assert "--repo is required for compat" in capsys.readouterr().err
    monkeypatch.setattr(releases, "surface_in_worktree", lambda root: _Surface())
    assert main(["--uri", BOLT, "--repo", str(tmp_path), "compat"]) == 0
    assert capsys.readouterr().out == (
        f"release 2026.09.17-7715ee35 (7715ee35) vs {tmp_path}: "
        "1 compatible, 0 drifted, 1 unchecked\n"
    )
    drifted = RELEASE.model_copy(
        update={
            "closures": {
                **RELEASE.closures,
                "DsB": {"Experiment": "zz", "Genotype": "gg"},
            }
        }
    )

    class Drifted(_ScriptedDriver):
        def __init__(self, uri: str, auth: tuple[str, str], **kwargs: Any) -> None:
            super().__init__(uri, auth, **kwargs)
            self.release_rows["torchcell"] = [[drifted.to_properties()]]

    monkeypatch.setattr("neo4j.GraphDatabase.driver", Drifted)
    assert (
        main(["--uri", BOLT, "--repo", str(tmp_path), "compat", "--version", "1.0"])
        == 1
    )
    assert capsys.readouterr().out.splitlines() == [
        f"release 2026.09.17-7715ee35 (7715ee35) vs {tmp_path}: "
        "1 compatible, 1 drifted, 0 unchecked",
        "  DsB: Experiment",
    ]
