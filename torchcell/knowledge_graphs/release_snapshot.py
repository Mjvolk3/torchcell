# torchcell/knowledge_graphs/release_snapshot.py
# [[torchcell.knowledge_graphs.release_snapshot]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/release_snapshot
# Test file: tests/torchcell/knowledge_graphs/test_release_snapshot.py
"""Committed snapshots of served knowledge-graph releases.

The build manifest (``kg_manifest.json``) lives beside the store on GilaHyper and
describes one physical database; nothing in git says which package version can read
which release. A **release snapshot** is the compact, committed answer: every ``stamp``
writes ``database/releases/<release>.json`` (identity, package version and tag, per
dataset the record count and ``content_sha256``, the graph-schema classes, the event
log, and one composite hash) plus ``<release>.closures.json`` (the per-dataset schema
closure fingerprints the compatibility check compares), so CI and a fresh clone can run
``scripts/kg_compat_page.py`` without the machine-local manifest
([[plan.data-release-program.2026.09.29]], Decisions 1 and 2).

``composite_sha256`` is the sha256 over the sorted per-dataset ``content_sha256``
values, newline-joined with a trailing newline (the same rule ``releases.content_sha256``
applies to experiment ids): two releases with equal composites serve byte-identical
experiment records for the same dataset set. Output is byte-stable: keys sorted,
``indent=2``, one trailing newline, so a second write of the same input is identical.

    python -m torchcell.knowledge_graphs.releases snapshot --manifest kg_manifest.json
        --n-nodes 99724909 --built-at 2026-09-21T20:25:23+00:00 --repo-root .
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from torchcell.knowledge_graphs.kg_manifest import GraphSchemaEntry, KgBuildManifest
from torchcell.knowledge_graphs.releases import content_sha256

__all__ = [
    "RELEASES_RELPATH",
    "SnapshotDataset",
    "SnapshotEvent",
    "KgReleaseSnapshot",
    "composite_sha256",
    "snapshot_from_manifest",
    "bootstrap_package_version",
    "snapshot_paths",
    "write_snapshot",
    "load_snapshot",
    "load_snapshots",
    "load_closures",
]

RELEASES_RELPATH = "database/releases"


class SnapshotDataset(BaseModel):
    """One served dataset as the release snapshot records it."""

    dataset_class: str
    n_experiments: int
    content_sha256: str
    import_mode: Literal["full", "incremental"]
    admitted_at: str


class SnapshotEvent(BaseModel):
    """One manifest event, reduced to what the compatibility page needs."""

    kind: Literal[
        "bootstrap", "full_build", "incremental_admission", "superset_admission"
    ]
    at: str
    torchcell_commit: str | None
    datasets: list[str]
    note: str | None = None


class KgReleaseSnapshot(BaseModel):
    """The committed description of one served release."""

    release: str
    version: str
    torchcell_commit: str
    torchcell_version: str | None
    torchcell_tag: str | None
    built_at: str
    neo4j_version: str
    biocypher_version: str
    store_host: str
    n_nodes: int | None
    datasets: dict[str, SnapshotDataset]
    graph_schema: dict[str, GraphSchemaEntry]
    events: list[SnapshotEvent]
    composite_sha256: str


def composite_sha256(datasets: Mapping[str, SnapshotDataset]) -> str:
    """sha256 over the sorted ``content_sha256`` values, newline-joined, trailing newline."""
    return content_sha256(entry.content_sha256 for entry in datasets.values())


def snapshot_from_manifest(
    manifest: KgBuildManifest, n_nodes: int | None = None, built_at: str | None = None
) -> KgReleaseSnapshot:
    """The snapshot of a stamped manifest.

    ``built_at`` defaults to the time of the manifest's last event, the one that
    produced the release; the slurm scripts pass the stamp's own timestamp. A manifest
    that is not stamped, or a dataset without a content hash or record count, is an
    error: the snapshot never fills a value the manifest does not carry.
    """
    if manifest.version is None or manifest.release is None:
        raise ValueError("the manifest carries no version/release; stamp it first")
    if manifest.torchcell_commit is None:
        raise ValueError("the manifest records no full-build commit")
    if not manifest.events:
        raise ValueError("the manifest records no events")
    datasets: dict[str, SnapshotDataset] = {}
    for name, entry in sorted(manifest.datasets.items()):
        if entry.content_sha256 is None:
            raise ValueError(f"no content hash for served dataset {name}; stamp first")
        if entry.n_experiments is None:
            raise ValueError(f"no experiment count for served dataset {name}")
        datasets[name] = SnapshotDataset(
            dataset_class=entry.dataset_class,
            n_experiments=entry.n_experiments,
            content_sha256=entry.content_sha256,
            import_mode=entry.import_mode,
            admitted_at=entry.admitted_at,
        )
    if built_at is None:
        built_at = manifest.events[-1].at
    return KgReleaseSnapshot(
        release=manifest.release,
        version=manifest.version,
        torchcell_commit=manifest.torchcell_commit,
        torchcell_version=manifest.torchcell_version,
        torchcell_tag=manifest.torchcell_tag,
        built_at=built_at,
        neo4j_version=manifest.neo4j_version,
        biocypher_version=manifest.biocypher_version,
        store_host=manifest.store_host,
        n_nodes=n_nodes,
        datasets=datasets,
        graph_schema={
            name: entry.model_copy()
            for name, entry in sorted(manifest.graph_schema.items())
        },
        events=[
            SnapshotEvent(
                kind=event.kind,
                at=event.at,
                torchcell_commit=event.torchcell_commit,
                datasets=list(event.datasets),
                note=event.note,
            )
            for event in manifest.events
        ],
        composite_sha256=composite_sha256(datasets),
    )


def bootstrap_package_version(
    snapshot: KgReleaseSnapshot,
    *,
    torchcell_version: str,
    torchcell_tag: str | None,
    note: str | None,
) -> KgReleaseSnapshot:
    """Fill the package version of a snapshot whose manifest predates the spine.

    Refused when the manifest already recorded one: the recorded value is the truth
    and an override would rewrite it. The last event's note says the values were
    supplied at snapshot time, plus ``note`` (how they were derived).
    """
    if snapshot.torchcell_version is not None:
        raise ValueError(
            f"the manifest already records torchcell_version "
            f"{snapshot.torchcell_version}; --torchcell-version is for bootstrapping"
        )
    text = (
        f"bootstrapped: torchcell_version {torchcell_version} and torchcell_tag "
        f"{torchcell_tag or 'none'} were supplied to `releases snapshot "
        "--torchcell-version` because the manifest predates the versioning spine"
    )
    if note:
        text = f"{text}; {note}"
    last = snapshot.events[-1]
    joined = f"{last.note}; {text}" if last.note else text
    events = [*snapshot.events[:-1], last.model_copy(update={"note": joined})]
    return snapshot.model_copy(
        update={
            "torchcell_version": torchcell_version,
            "torchcell_tag": torchcell_tag,
            "events": events,
        }
    )


def snapshot_paths(repo_root: Path, release: str) -> tuple[Path, Path]:
    """``(<release>.json, <release>.closures.json)`` under ``database/releases/``."""
    directory = repo_root / RELEASES_RELPATH
    return directory / f"{release}.json", directory / f"{release}.closures.json"


def _dump(data: object) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def write_snapshot(
    snapshot: KgReleaseSnapshot,
    closures: Mapping[str, Mapping[str, str]],
    repo_root: Path,
) -> tuple[Path, Path]:
    """Write the snapshot and its closures companion; returns the two paths.

    ``closures`` is the manifest's per-dataset closure map and must name exactly the
    snapshot's datasets.
    """
    if set(closures) != set(snapshot.datasets):
        raise ValueError(
            "closures name different datasets than the snapshot: "
            f"{sorted(set(closures) ^ set(snapshot.datasets))}"
        )
    snapshot_path, closures_path = snapshot_paths(repo_root, snapshot.release)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(_dump(snapshot.model_dump(mode="json")), encoding="utf-8")
    closures_path.write_text(
        _dump({name: dict(closure) for name, closure in closures.items()}),
        encoding="utf-8",
    )
    return snapshot_path, closures_path


def load_snapshot(path: Path) -> KgReleaseSnapshot:
    """Read one snapshot file."""
    return KgReleaseSnapshot.model_validate_json(path.read_text(encoding="utf-8"))


def load_snapshots(repo_root: Path) -> list[KgReleaseSnapshot]:
    """Every committed snapshot, oldest build first (ties broken by release id)."""
    directory = repo_root / RELEASES_RELPATH
    paths = [
        path
        for path in sorted(directory.glob("*.json"))
        if not path.name.endswith(".closures.json")
    ]
    snapshots = [load_snapshot(path) for path in paths]
    return sorted(snapshots, key=lambda s: (s.built_at, s.release))


def load_closures(repo_root: Path, release: str) -> dict[str, dict[str, str]]:
    """The per-dataset closure fingerprints committed beside ``release``'s snapshot."""
    _, closures_path = snapshot_paths(repo_root, release)
    data = json.loads(closures_path.read_text(encoding="utf-8"))
    return {name: dict(closure) for name, closure in data.items()}
