# tests/torchcell/knowledge_graphs/test_release_snapshot.py
# [[tests.torchcell.knowledge_graphs.test_release_snapshot]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/knowledge_graphs/test_release_snapshot.py
"""Release snapshots from a hand-built manifest: the exact composite hash, sorted
datasets, byte-stable files, the round trip, the bootstrap override, and the refusals.

The manifest serves ``DsB`` (hash ``b`` * 64, 20 records) and ``DsA`` (hash ``a`` * 64,
10 records), listed in that order, stamped as version 1.0 / release
``2026.09.17-7715ee35``. The composite is sha256 over the sorted hashes newline-joined
with a trailing newline, sha256 of ``"a" * 64``, newline, ``"b" * 64``, newline, =
``913f9338fb6c17253f3a14816fc08d52522454bfc19a8c99538a197cfb23fb41``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from torchcell.knowledge_graphs.kg_manifest import (
    GraphSchemaEntry,
    KgBuildManifest,
    KgDatasetEntry,
    KgEvent,
)
from torchcell.knowledge_graphs.release_snapshot import (
    KgReleaseSnapshot,
    SnapshotDataset,
    SnapshotEvent,
    bootstrap_package_version,
    composite_sha256,
    load_closures,
    load_snapshot,
    load_snapshots,
    snapshot_from_manifest,
    snapshot_paths,
    write_snapshot,
)

COMPOSITE = "913f9338fb6c17253f3a14816fc08d52522454bfc19a8c99538a197cfb23fb41"


def _manifest(stamped: bool = True) -> KgBuildManifest:
    def entry(name: str, n: int, digest: str) -> KgDatasetEntry:
        return KgDatasetEntry(
            dataset_class=name,
            loader_relpath=f"torchcell/datasets/{name}.py",
            adapter_files=[],
            closure={"Experiment": f"e-{name}", "Genotype": "gg"},
            n_experiments=n,
            biocypher_out="2026-09-16_00-44-53",
            import_mode="full",
            admitted_at="2026-09-17T20:36:32-05:00",
            torchcell_commit="7715ee35d95c",
            content_sha256=digest,
        )

    return KgBuildManifest(
        database="torchcell",
        store_host="gilahyper",
        neo4j_version="5.26.28",
        biocypher_version="0.15.2",
        torchcell_commit="7715ee35d95c",
        version="1.0" if stamped else None,
        release="2026.09.17-7715ee35" if stamped else None,
        torchcell_version="1.2.0" if stamped else None,
        torchcell_tag="v1.2.0" if stamped else None,
        graph_schema={
            "experiment": GraphSchemaEntry(kind="node", properties=["serialized_data"]),
            "phenotype member of": GraphSchemaEntry(
                kind="edge", source=["fitness phenotype"], target=["experiment"]
            ),
        },
        cell_adapter_methods={},
        cell_adapter_table={},
        adapter_files={},
        datasets={"DsB": entry("DsB", 20, "b" * 64), "DsA": entry("DsA", 10, "a" * 64)},
        events=[
            KgEvent(
                kind="bootstrap",
                at="2026-09-18T01:36:59+00:00",
                torchcell_commit="7715ee35d95c",
                datasets=["DsA", "DsB"],
                biocypher_out="2026-09-16_00-44-53",
                note="reconstructed",
            )
        ],
        created_at="2026-09-18T01:36:59+00:00",
        updated_at="2026-09-18T01:36:59+00:00",
    )


def test_composite_is_sha256_over_sorted_hashes_newline_joined() -> None:
    datasets = {
        "DsB": SnapshotDataset(
            dataset_class="DsB",
            n_experiments=1,
            content_sha256="b" * 64,
            import_mode="full",
            admitted_at="t",
        ),
        "DsA": SnapshotDataset(
            dataset_class="DsA",
            n_experiments=1,
            content_sha256="a" * 64,
            import_mode="full",
            admitted_at="t",
        ),
    }
    assert composite_sha256(datasets) == COMPOSITE
    assert (
        hashlib.sha256(("a" * 64 + "\n" + "b" * 64 + "\n").encode()).hexdigest()
        == COMPOSITE
    )


def test_snapshot_from_manifest_sorts_datasets_and_defaults_built_at_to_the_last_event() -> (
    None
):
    snapshot = snapshot_from_manifest(_manifest(), n_nodes=99)
    assert snapshot == KgReleaseSnapshot(
        release="2026.09.17-7715ee35",
        version="1.0",
        torchcell_commit="7715ee35d95c",
        torchcell_version="1.2.0",
        torchcell_tag="v1.2.0",
        built_at="2026-09-18T01:36:59+00:00",
        neo4j_version="5.26.28",
        biocypher_version="0.15.2",
        store_host="gilahyper",
        n_nodes=99,
        datasets={
            "DsA": SnapshotDataset(
                dataset_class="DsA",
                n_experiments=10,
                content_sha256="a" * 64,
                import_mode="full",
                admitted_at="2026-09-17T20:36:32-05:00",
            ),
            "DsB": SnapshotDataset(
                dataset_class="DsB",
                n_experiments=20,
                content_sha256="b" * 64,
                import_mode="full",
                admitted_at="2026-09-17T20:36:32-05:00",
            ),
        },
        graph_schema={
            "experiment": GraphSchemaEntry(kind="node", properties=["serialized_data"]),
            "phenotype member of": GraphSchemaEntry(
                kind="edge", source=["fitness phenotype"], target=["experiment"]
            ),
        },
        events=[
            SnapshotEvent(
                kind="bootstrap",
                at="2026-09-18T01:36:59+00:00",
                torchcell_commit="7715ee35d95c",
                datasets=["DsA", "DsB"],
                note="reconstructed",
            )
        ],
        composite_sha256=COMPOSITE,
    )
    assert list(snapshot.datasets) == ["DsA", "DsB"]
    stamped = snapshot_from_manifest(_manifest(), built_at="2026-09-17T21:00:00-05:00")
    assert (stamped.built_at, stamped.n_nodes) == ("2026-09-17T21:00:00-05:00", None)


def test_snapshot_from_manifest_refuses_unstamped_and_hashless_manifests() -> None:
    with pytest.raises(ValueError, match="carries no version/release; stamp it first"):
        snapshot_from_manifest(_manifest(stamped=False))
    manifest = _manifest()
    manifest.datasets["DsA"].content_sha256 = None
    with pytest.raises(ValueError, match="no content hash for served dataset DsA"):
        snapshot_from_manifest(manifest)
    manifest = _manifest()
    manifest.datasets["DsB"].n_experiments = None
    with pytest.raises(ValueError, match="no experiment count for served dataset DsB"):
        snapshot_from_manifest(manifest)
    manifest = _manifest()
    manifest.torchcell_commit = None
    with pytest.raises(ValueError, match="records no full-build commit"):
        snapshot_from_manifest(manifest)
    manifest = _manifest()
    manifest.events = []
    with pytest.raises(ValueError, match="records no events"):
        snapshot_from_manifest(manifest)


def test_write_snapshot_is_byte_stable_and_round_trips(tmp_path: Path) -> None:
    """Sorted keys, indent 2, trailing newline; a second write is byte-identical; the
    closures file holds the manifest's closure maps; loading gives the snapshot back.
    """
    manifest = _manifest()
    snapshot = snapshot_from_manifest(manifest, n_nodes=99)
    closures = {name: dict(e.closure) for name, e in manifest.datasets.items()}
    paths = write_snapshot(snapshot, closures, tmp_path)
    assert paths == snapshot_paths(tmp_path, "2026.09.17-7715ee35")
    assert paths == (
        tmp_path / "database" / "releases" / "2026.09.17-7715ee35.json",
        tmp_path / "database" / "releases" / "2026.09.17-7715ee35.closures.json",
    )
    first = [p.read_bytes() for p in paths]
    write_snapshot(snapshot, closures, tmp_path)
    assert [p.read_bytes() for p in paths] == first
    text = paths[0].read_text(encoding="utf-8")
    assert text.endswith("}\n") and not text.endswith("}\n\n")
    assert text == json.dumps(snapshot.model_dump(), indent=2, sort_keys=True) + "\n"
    assert text.startswith('{\n  "biocypher_version": "0.15.2",\n  "built_at":')
    assert paths[1].read_text(encoding="utf-8") == (
        "{\n"
        '  "DsA": {\n    "Experiment": "e-DsA",\n    "Genotype": "gg"\n  },\n'
        '  "DsB": {\n    "Experiment": "e-DsB",\n    "Genotype": "gg"\n  }\n'
        "}\n"
    )
    assert load_snapshot(paths[0]) == snapshot
    assert load_closures(tmp_path, "2026.09.17-7715ee35") == closures


def test_write_snapshot_refuses_closures_for_other_datasets(tmp_path: Path) -> None:
    snapshot = snapshot_from_manifest(_manifest())
    with pytest.raises(ValueError, match=r"different datasets.*\['DsC'\]"):
        write_snapshot(snapshot, {"DsA": {}, "DsB": {}, "DsC": {}}, tmp_path)
    assert not (tmp_path / "database").exists()


def test_load_snapshots_orders_by_built_at_and_skips_closure_files(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    closures = {name: dict(e.closure) for name, e in manifest.datasets.items()}
    later = snapshot_from_manifest(manifest, built_at="2026-09-30T00:00:00+00:00")
    later = later.model_copy(
        update={"release": "2026.09.30-abcdef01", "version": "1.1"}
    )
    earlier = snapshot_from_manifest(manifest, built_at="2026-09-17T20:36:32-05:00")
    write_snapshot(later, closures, tmp_path)
    write_snapshot(earlier, closures, tmp_path)
    assert [s.release for s in load_snapshots(tmp_path)] == [
        "2026.09.17-7715ee35",
        "2026.09.30-abcdef01",
    ]
    assert load_snapshots(tmp_path / "empty") == []


def test_bootstrap_package_version_fills_the_fields_and_notes_it_once() -> None:
    """A manifest from before the spine (version None) takes the supplied values and the
    last event's note says so; a manifest that already records one is refused.
    """
    manifest = _manifest()
    manifest.torchcell_version = None
    manifest.torchcell_tag = None
    bare = snapshot_from_manifest(manifest)
    filled = bootstrap_package_version(
        bare, torchcell_version="1.2.0", torchcell_tag=None, note="from git show"
    )
    assert (filled.torchcell_version, filled.torchcell_tag) == ("1.2.0", None)
    assert filled.events[-1].note == (
        "reconstructed; bootstrapped: torchcell_version 1.2.0 and torchcell_tag none "
        "were supplied to `releases snapshot --torchcell-version` because the manifest "
        "predates the versioning spine; from git show"
    )
    assert bare.events[-1].note == "reconstructed"
    assert filled.composite_sha256 == COMPOSITE
    with pytest.raises(ValueError, match="already records torchcell_version 1.2.0"):
        bootstrap_package_version(
            filled, torchcell_version="9.9.9", torchcell_tag="v9.9.9", note=None
        )
    tagged = bootstrap_package_version(
        bare.model_copy(
            update={"events": [bare.events[0].model_copy(update={"note": None})]}
        ),
        torchcell_version="1.2.1",
        torchcell_tag="v1.2.1",
        note=None,
    )
    assert tagged.events[-1].note == (
        "bootstrapped: torchcell_version 1.2.1 and torchcell_tag v1.2.1 were supplied "
        "to `releases snapshot --torchcell-version` because the manifest predates the "
        "versioning spine"
    )
