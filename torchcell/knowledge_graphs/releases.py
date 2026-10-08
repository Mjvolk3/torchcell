# torchcell/knowledge_graphs/releases.py
# [[torchcell.knowledge_graphs.releases]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/releases
"""Knowledge-graph releases: what version a served store is, which datasets it holds,
and whether a dataset's bytes changed between two versions.

A **release** is one built store. Its id is ``<build date>-<torchcell commit>`` in the
shape the iBioFoundry deployments use (``2026.09.17-7715ee35``): the immutable identity a
dump, a backup, and a served database all carry. Its **version** is ``<major>.<minor>``:
a full rebuild bumps the major, an incremental admission bumps the minor. The store
describes itself with one ``KgRelease`` node written before it is served, so a dump loaded
anywhere answers "what are you" with a one-line query, and the manifest file beside the
store carries the same fields.

**Byte identity.** Every ``Experiment`` node id is the sha256 of the serialized record
(``CellAdapter`` builds ids that way), so the sorted list of a dataset's experiment ids is
a fingerprint of its content. ``content_sha256`` is the sha256 of those ids, sorted and
newline-joined. Two releases with the same hash for a dataset hold the same serialized
experiment records; a changed value, medium, or schema closure changes it. The hash is
computed from the CSVs at build time and from the store for a cross-check, and
:func:`diff` turns two releases into unchanged / changed / added / removed datasets.

**Choosing a version from code.** Clients name a version, not a database:
``latest`` (the default) and ``pinned`` are aliases in the served DBMS, retargeted when a
release is published; an explicit release id or version resolves to the physical
database that carries it. ``TORCHCELL_KG_VERSION`` sets the default for a whole run.

**Source code moving with the store.** Records are serialized under the schema contract
of the commit that built them. :func:`compatibility` compares the release's per-dataset
closure fingerprints against the local schema surface and names the datasets whose
records the local code would serialize differently, the same drift the admission gate
checks before an increment.

**The package version a release names.** ``stamp`` records ``torchcell_version`` (the
stamping checkout's ``torchcell.__version__``) and ``torchcell_tag`` (``git describe
--tags --exact-match HEAD``, None between package releases) in the manifest, and the
node and the committed snapshot (``snapshot``, ``torchcell.knowledge_graphs
.release_snapshot``) carry them, so ``scripts/kg_compat_page.py`` can say which package
tags read which release without the machine-local manifest.

**The files the graph points at.** A release also records its artifact pointer set
(``artifact_refs``, dataset -> file-level ``ArtifactRef``; recorded by ``kg_manifest
artifact-refs`` before the stamp, None on a release that predates the recording).
``artifacts`` reads it from a store's node and asks tc-data whether every pointed file is
listed with the sha256 the pointer pins, printing one ``STATE<TAB>CODE<TAB>DETAIL`` line
for ``scripts/ops.sh``.

    python -m torchcell.knowledge_graphs.releases status --label gilahyper
    python -m torchcell.knowledge_graphs.releases datasets --version latest
    python -m torchcell.knowledge_graphs.releases diff 2026.09.17-7715ee35 latest
    python -m torchcell.knowledge_graphs.releases snapshot --manifest kg_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from torchcell.knowledge_graphs.kg_manifest import (
    ArtifactPointer,
    KgBuildManifest,
    checkout_package_version,
    manifest_artifact_refs,
    surface_in_worktree,
)
from torchcell.provenance.schema_deps import SchemaSurface

RELEASE_LABEL = "KgRelease"
LATEST_ALIAS = "latest"
PINNED_ALIAS = "pinned"
DEFAULT_VERSION = LATEST_ALIAS
SYSTEM_DATABASES = ("system",)
_COMMIT_CHARS = 8
DEFAULT_TC_DATA_API_KEY_ENV = "TC_DATA_API_KEY"
#: How many failing pointers the one-line ``artifacts`` verdict names.
_FAILING_SHOWN = 3


# --------------------------------------------------------------------------- models


class ReleaseDataset(BaseModel):
    """One dataset as one release serves it."""

    dataset_class: str
    n_experiments: int
    content_sha256: str


class KgRelease(BaseModel):
    """The self-description a served store carries as its ``KgRelease`` node."""

    release: str  # <YYYY.MM.DD>-<commit[:8]>; the build date, the generation commit
    version: str  # <major>.<minor>
    torchcell_commit: str
    # ``torchcell.__version__`` of the checkout that stamped the release and the git tag
    # on its HEAD (None when the build ran from an untagged commit); both recorded at
    # the stamp, never inferred later. None on a node written before the versioning
    # spine ([[plan.data-release-program.2026.09.29]]).
    torchcell_version: str | None = None
    torchcell_tag: str | None = None
    built_at: str
    biocypher_out: str
    neo4j_version: str | None = None
    n_nodes: int | None = None
    datasets: dict[str, ReleaseDataset]
    # dataset -> schema symbol -> contract fingerprint at build; the compatibility check
    closures: dict[str, dict[str, str]] = Field(default_factory=dict)
    # dataset -> the file-level artifact pointers its records carry; None on a node
    # written before pointer recording, or from a manifest with an unrecorded entry
    artifact_refs: dict[str, list[ArtifactPointer]] | None = None

    @property
    def n_datasets(self) -> int:
        """Number of datasets in the release."""
        return len(self.datasets)

    def to_properties(self) -> dict[str, Any]:
        """Flat property map for the graph node (maps go in as JSON strings)."""
        return {
            "release": self.release,
            "version": self.version,
            "torchcell_commit": self.torchcell_commit,
            "torchcell_version": self.torchcell_version,
            "torchcell_tag": self.torchcell_tag,
            "built_at": self.built_at,
            "biocypher_out": self.biocypher_out,
            "neo4j_version": self.neo4j_version,
            "n_nodes": self.n_nodes,
            "n_datasets": self.n_datasets,
            "datasets_json": json.dumps(
                {k: v.model_dump() for k, v in sorted(self.datasets.items())},
                separators=(",", ":"),
            ),
            "closures_json": json.dumps(self.closures, separators=(",", ":")),
            "artifact_refs_json": (
                None
                if self.artifact_refs is None
                else json.dumps(
                    {
                        name: [ref.model_dump(exclude_none=True) for ref in refs]
                        for name, refs in sorted(self.artifact_refs.items())
                    },
                    separators=(",", ":"),
                )
            ),
        }

    @classmethod
    def from_properties(cls, props: dict[str, Any]) -> KgRelease:
        """Inverse of :meth:`to_properties`."""
        datasets = {
            k: ReleaseDataset.model_validate(v)
            for k, v in json.loads(props["datasets_json"]).items()
        }
        refs_json = props.get("artifact_refs_json")
        artifact_refs = (
            None
            if refs_json is None
            else {
                name: [ArtifactPointer.model_validate(ref) for ref in refs]
                for name, refs in json.loads(refs_json).items()
            }
        )
        return cls(
            release=props["release"],
            version=props["version"],
            torchcell_commit=props["torchcell_commit"],
            torchcell_version=props.get("torchcell_version"),
            torchcell_tag=props.get("torchcell_tag"),
            built_at=props["built_at"],
            biocypher_out=props["biocypher_out"],
            neo4j_version=props.get("neo4j_version"),
            n_nodes=props.get("n_nodes"),
            datasets=datasets,
            closures=json.loads(props.get("closures_json") or "{}"),
            artifact_refs=artifact_refs,
        )


class ServedDatabase(BaseModel):
    """One database of a DBMS as ``SHOW DATABASES`` and its release node describe it."""

    name: str
    aliases: list[str]
    default: bool
    status: str  # currentStatus
    release: KgRelease | None = None
    n_nodes: int | None = None
    n_datasets: int | None = None
    fault: str | None = None  # the error a store read raised, if any


class ReleaseDiff(BaseModel):
    """Datasets by what happened to their bytes between two releases."""

    from_release: str
    to_release: str
    unchanged: list[str]  # same content_sha256: byte-identical serialized records
    changed: list[str]
    added: list[str]
    removed: list[str]


class DatasetDrift(BaseModel):
    """A served dataset the local code would serialize under a different contract."""

    dataset_class: str
    changed_symbols: list[str]


class ReleaseCompatibility(BaseModel):
    """Whether the local schema surface matches the contracts a release was built under."""

    release: str
    torchcell_commit: str
    compatible: list[str]
    drifted: list[DatasetDrift]
    unchecked: list[str]  # datasets whose closure the release did not record

    @property
    def ok(self) -> bool:
        """True when no served dataset drifted from the local schema surface."""
        return not self.drifted

    @property
    def paired(self) -> bool:
        """True when EVERY served dataset is verified: none drifted, none unchecked.

        ``ok`` tolerates datasets whose closure the release did not record; the
        pairing gate does not, because an unverified dataset is one the client may
        read under the wrong contract.
        """
        return not self.drifted and not self.unchecked


class IncompatibleReleaseError(RuntimeError):
    """A served release and the installed package are not a pair.

    Raised by :func:`require_paired` when the installed schema surface does not
    reproduce every served dataset's contract fingerprints, or when the store carries
    no release node at all. A package version and a knowledge-graph release either
    read every served record under the contract it was written with, or the client
    refuses; there is no partial pair.
    """


# --------------------------------------------------------------------------- identity


def release_id(built_at: str | datetime, commit: str) -> str:
    """``<YYYY.MM.DD>-<commit[:8]>``: the build date and the generation commit."""
    when = datetime.fromisoformat(built_at) if isinstance(built_at, str) else built_at
    return f"{when:%Y.%m.%d}-{commit[:_COMMIT_CHARS]}"


def next_version(current: str | None, kind: Literal["full", "incremental"]) -> str:
    """``major.minor`` after a full build (major + 1, minor 0) or an admission (minor + 1)."""
    if current is None:
        return "1.0"
    major, minor = (int(part) for part in current.split("."))
    if kind == "full":
        return f"{major + 1}.0"
    return f"{major}.{minor + 1}"


def content_sha256(ids: Iterable[str]) -> str:
    """sha256 of the sorted, newline-joined ids (trailing newline)."""
    digest = hashlib.sha256()
    for item in sorted(ids):
        digest.update(item.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def content_hashes_from_csv(out_dir: Path) -> dict[str, str]:
    """``Dataset.id -> content_sha256`` from a BioCypher output directory.

    Reads the ``ExperimentMemberOf`` edge files: ``:START_ID`` is the experiment id,
    ``:END_ID`` the dataset id (the dataset class name). Every id is held in memory,
    grouped by dataset, so this belongs on the build node.
    """
    from torchcell.knowledge_graphs.incremental_import import discover_csv_groups

    csv.field_size_limit(sys.maxsize)
    group = next(
        g for g in discover_csv_groups(out_dir) if g.label == "ExperimentMemberOf"
    )
    start = group.columns.index(":START_ID")
    end = group.columns.index(":END_ID")
    ids: dict[str, list[str]] = defaultdict(list)
    for part in group.part_paths:
        with open(part, encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle, delimiter="\t", quotechar="'", strict=True):
                ids[row[end].strip("'")].append(row[start].strip("'"))
    return {name: content_sha256(members) for name, members in sorted(ids.items())}


def content_hashes_from_store(
    uri: str, user: str, password: str, database: str
) -> dict[str, str]:
    """``Dataset.id -> content_sha256`` streamed from a running store, one dataset at a time."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))
    hashes: dict[str, str] = {}
    with driver.session(database=database, fetch_size=10000) as session:
        names = [
            row[0]
            for row in session.run(
                "MATCH (d:Dataset) RETURN d.id ORDER BY d.id"
            ).values()
        ]
        for name in names:
            result = session.run(
                "MATCH (d:Dataset {id: $id})<-[:ExperimentMemberOf]-(e:Experiment) "
                "RETURN e.id AS id",
                id=name,
            )
            hashes[name] = content_sha256(record["id"] for record in result)
    driver.close()
    return hashes


# --------------------------------------------------------------------------- the node


def read_release(uri: str, user: str, password: str, database: str) -> KgRelease | None:
    """The store's release node, or None when the store predates release nodes."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session(database=database) as session:
        rows = session.run(f"MATCH (r:{RELEASE_LABEL}) RETURN r").values()
    driver.close()
    if not rows:
        return None
    if len(rows) > 1:
        raise ValueError(f"{database} carries {len(rows)} {RELEASE_LABEL} nodes")
    return KgRelease.from_properties(dict(rows[0][0]))


def write_release(
    uri: str, user: str, password: str, database: str, release: KgRelease
) -> None:
    """Write (or replace) the store's single release node; the database must be writable."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session(database=database) as session:
        session.run(
            f"MERGE (r:{RELEASE_LABEL} {{singleton: true}}) SET r += $props",
            props=release.to_properties(),
        )
    driver.close()


def release_from_manifest(
    manifest: KgBuildManifest,
    *,
    built_at: str,
    content_hashes: dict[str, str],
    n_nodes: int | None,
) -> KgRelease:
    """The release record a manifest describes, with the measured content hashes."""
    missing = sorted(set(manifest.datasets) - set(content_hashes))
    if missing:
        raise ValueError(f"no content hash for served datasets {missing}")
    if manifest.torchcell_commit is None:
        raise ValueError("the manifest records no full-build commit")
    if manifest.version is None or manifest.release is None:
        raise ValueError("the manifest carries no version/release; stamp it first")
    datasets = {
        name: ReleaseDataset(
            dataset_class=name,
            n_experiments=int(entry.n_experiments or 0),
            content_sha256=content_hashes[name],
        )
        for name, entry in sorted(manifest.datasets.items())
    }
    return KgRelease(
        release=manifest.release,
        version=manifest.version,
        torchcell_commit=manifest.torchcell_commit,
        torchcell_version=manifest.torchcell_version,
        torchcell_tag=manifest.torchcell_tag,
        built_at=built_at,
        biocypher_out=manifest.events[-1].biocypher_out or "",
        neo4j_version=manifest.neo4j_version,
        n_nodes=n_nodes,
        datasets=datasets,
        closures={
            name: dict(entry.closure) for name, entry in manifest.datasets.items()
        },
        artifact_refs=manifest_artifact_refs(manifest),
    )


def stamp_manifest(
    manifest: KgBuildManifest,
    *,
    kind: Literal["full", "incremental"],
    built_at: str,
    content_hashes: dict[str, str],
    previous_version: str | None,
    torchcell_version: str,
    torchcell_tag: str | None,
) -> KgBuildManifest:
    """Give a manifest its version, release id, package version, and content hashes.

    A full build passes a hash for every dataset. An incremental admission passes the
    hashes of the datasets it imported; the rest keep the hash they already carry,
    since incremental import never touches existing nodes. Any dataset left without a
    hash is an error. ``torchcell_version`` and ``torchcell_tag`` are what the stamping
    checkout reports (``checkout_package_version``); the release names them from now on.
    """
    if manifest.torchcell_commit is None:
        raise ValueError("the manifest records no full-build commit")
    manifest.version = next_version(previous_version, kind)
    commit = manifest.events[-1].torchcell_commit or manifest.torchcell_commit
    manifest.release = release_id(built_at, commit)
    manifest.torchcell_version = torchcell_version
    manifest.torchcell_tag = torchcell_tag
    for name, entry in manifest.datasets.items():
        if name in content_hashes:
            entry.content_sha256 = content_hashes[name]
    missing = sorted(n for n, e in manifest.datasets.items() if not e.content_sha256)
    if missing:
        raise ValueError(f"no content hash for served datasets {missing}")
    return manifest


# --------------------------------------------------------------------------- serving


def _fault_text(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return text[:120]


def list_databases(
    uri: str, user: str, password: str, *, probe: bool = True
) -> list[ServedDatabase]:
    """Every user database of the DBMS at ``uri`` with its aliases, release, and health.

    ``probe`` reads one ``Dataset`` node from each online database: the count store
    answers even when the files underneath fault, so a real property read is what tells
    a healthy store from one whose cold pages raise ``IOException``.
    """
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=10)
    try:
        with driver.session(database="system") as session:
            rows = session.run(
                "SHOW DATABASES YIELD name, aliases, default, currentStatus, type "
                "RETURN name, aliases, default, currentStatus, type"
            ).values()
    except Exception as exc:  # a DBMS whose system database itself faults
        driver.close()
        return [
            ServedDatabase(
                name="(dbms)",
                aliases=[],
                default=False,
                status="?",
                fault=_fault_text(exc),
            )
        ]
    served: list[ServedDatabase] = []
    for name, aliases, default, status, kind in rows:
        if kind == "system" or name in SYSTEM_DATABASES:
            continue
        entry = ServedDatabase(
            name=name, aliases=list(aliases), default=bool(default), status=status
        )
        if probe and status == "online":
            try:
                with driver.session(database=name) as session:
                    entry.n_nodes = int(
                        session.run("MATCH (n) RETURN count(n)").single()[0]
                    )
                    entry.n_datasets = int(
                        session.run("MATCH (d:Dataset) RETURN count(d)").single()[0]
                    )
                    session.run("MATCH (d:Dataset) RETURN d.id LIMIT 1").consume()
                    node = session.run(f"MATCH (r:{RELEASE_LABEL}) RETURN r").values()
                if node:
                    entry.release = KgRelease.from_properties(dict(node[0][0]))
            except Exception as exc:  # the probe's purpose is to report the fault
                entry.fault = _fault_text(exc)
        served.append(entry)
    driver.close()
    return served


def list_databases_bounded(
    uri: str, user: str, password: str, *, timeout_s: float
) -> list[ServedDatabase]:
    """:func:`list_databases` with a wall-clock bound; a hung host yields one row."""
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FutureTimeout

    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(list_databases, uri, user, password)
    try:
        return future.result(timeout=timeout_s)
    except FutureTimeout:
        return [
            ServedDatabase(
                name="(dbms)",
                aliases=[],
                default=False,
                status="?",
                fault=f"no answer within {timeout_s:g}s",
            )
        ]
    except Exception as exc:  # connection refused, auth, DNS: one row, not a crash
        return [
            ServedDatabase(
                name="(dbms)",
                aliases=[],
                default=False,
                status="?",
                fault=_fault_text(exc),
            )
        ]
    finally:
        pool.shutdown(wait=False)


def resolve_database(version: str, uri: str, user: str, password: str) -> str:
    """The database name to open for ``version``.

    ``latest`` and ``pinned`` are aliases and pass through once the DBMS has them; a
    physical database name passes through; a release id or a ``major.minor`` version
    resolves through the release nodes. Anything else raises ``LookupError``.
    """
    served = list_databases(uri, user, password, probe=False)
    names = {db.name for db in served}
    aliases = {alias for db in served for alias in db.aliases}
    if version in aliases or version in names:
        return version
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))
    matches: list[str] = []
    for db in served:
        if db.status != "online":
            continue
        with driver.session(database=db.name) as session:
            rows = session.run(
                f"MATCH (r:{RELEASE_LABEL}) RETURN r.release, r.version"
            ).values()
        if rows and version in (rows[0][0], rows[0][1]):
            matches.append(db.name)
    driver.close()
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise LookupError(
            f"no database at {uri} serves version {version!r}; "
            f"databases {sorted(names)}, aliases {sorted(aliases)}"
        )
    raise LookupError(f"version {version!r} is served by several databases: {matches}")


def datasets(version: str, uri: str, user: str, password: str) -> list[ReleaseDataset]:
    """The datasets a version serves, with record counts and content hashes."""
    database = resolve_database(version, uri, user, password)
    release = read_release(uri, user, password, database)
    if release is None:
        raise LookupError(f"{database} carries no {RELEASE_LABEL} node")
    return [release.datasets[name] for name in sorted(release.datasets)]


def diff(a: KgRelease, b: KgRelease) -> ReleaseDiff:
    """Datasets unchanged (byte-identical), changed, added, and removed from ``a`` to ``b``."""
    shared = sorted(set(a.datasets) & set(b.datasets))
    return ReleaseDiff(
        from_release=a.release,
        to_release=b.release,
        unchanged=[
            n
            for n in shared
            if a.datasets[n].content_sha256 == b.datasets[n].content_sha256
        ],
        changed=[
            n
            for n in shared
            if a.datasets[n].content_sha256 != b.datasets[n].content_sha256
        ],
        added=sorted(set(b.datasets) - set(a.datasets)),
        removed=sorted(set(a.datasets) - set(b.datasets)),
    )


def compatibility(release: KgRelease, repo_root: Path) -> ReleaseCompatibility:
    """Which served datasets the checkout at ``repo_root`` would serialize differently."""
    return compatibility_with_surface(release, surface_in_worktree(repo_root))


def compatibility_with_surface(
    release: KgRelease, surface: SchemaSurface
) -> ReleaseCompatibility:
    """:func:`compatibility` against an already-loaded schema surface."""
    return closure_compatibility(
        release.release,
        release.torchcell_commit,
        release.datasets,
        release.closures,
        surface,
    )


def closure_compatibility(
    release: str,
    torchcell_commit: str,
    dataset_names: Iterable[str],
    closures: Mapping[str, Mapping[str, str]],
    surface: SchemaSurface,
) -> ReleaseCompatibility:
    """The compatibility verdict from a release's closures and any schema surface.

    The surface may come from the working tree, from ``git show`` at a package tag
    (``scripts/kg_compat_page.py`` reads the closures from the committed snapshot), or
    from any two files parsed by ``torchcell.provenance.schema_deps.load_surface``.
    """
    compatible: list[str] = []
    drifted: list[DatasetDrift] = []
    unchecked: list[str] = []
    for name in sorted(dataset_names):
        closure = closures.get(name)
        if not closure:
            unchecked.append(name)
            continue
        changed = sorted(
            symbol
            for symbol, fingerprint in closure.items()
            if surface.fingerprints.get(symbol) != fingerprint
        )
        if changed:
            drifted.append(DatasetDrift(dataset_class=name, changed_symbols=changed))
        else:
            compatible.append(name)
    return ReleaseCompatibility(
        release=release,
        torchcell_commit=torchcell_commit,
        compatible=compatible,
        drifted=drifted,
        unchecked=unchecked,
    )


def require_paired(
    release: KgRelease | None,
    surface: SchemaSurface,
    *,
    installed_version: str,
    database: str,
) -> ReleaseCompatibility:
    """The compatibility report, or :class:`IncompatibleReleaseError`.

    The gate every client passes at connect: ``release`` is the store's node (None when
    the store has none), ``surface`` the installed package's schema surface
    (``schema_deps.load_default_surface``). The report comes back only when every
    served dataset's closure fingerprints are reproduced; a drifted or unverified
    dataset, or a store without a release node, raises with the paired package named
    so the remedy is one line: install that package, or point
    ``TORCHCELL_KG_VERSION`` at a release built under the installed one.
    """
    if release is None:
        raise IncompatibleReleaseError(
            f"database {database!r} carries no {RELEASE_LABEL} node, so no package "
            "version is paired with it; a store is served only after `releases "
            "write-node` (the live rebuild does this) or a `kg_release.sh deploy`"
        )
    report = compatibility_with_surface(release, surface)
    if report.paired:
        return report
    paired = package_label(release)
    lines = [
        f"knowledge-graph release {release.release} (KG {release.version}, database "
        f"{database!r}) is paired with torchcell {paired}; the installed torchcell "
        f"{installed_version} is not a pair:"
    ]
    for drift in report.drifted:
        lines.append(
            f"  {drift.dataset_class}: serialized under a different contract for "
            + ", ".join(drift.changed_symbols)
        )
    for name in report.unchecked:
        lines.append(f"  {name}: the release recorded no closure to verify against")
    if release.torchcell_tag is not None:
        remedy = f"Install the paired package (pip install torchcell=={paired[1:]})"
    else:
        remedy = (
            "The release names no package tag; pick one that reads it on the "
            "compatibility page (docs/source/database/compatibility.md)"
        )
    lines.append(
        f"{len(report.drifted)} drifted and {len(report.unchecked)} unverified of "
        f"{release.n_datasets} served datasets. {remedy} or set "
        "TORCHCELL_KG_VERSION to a release built under the installed schema."
    )
    raise IncompatibleReleaseError("\n".join(lines))


# --------------------------------------------------------------------------- reporting


def package_checkout() -> Path:
    """The checkout this ``torchcell`` was imported from (the parent of the package)."""
    import torchcell

    return Path(torchcell.__file__).resolve().parents[1]


def _git(repo_root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def commit_index(repo_root: Path, sha: str) -> str:
    """Position of ``sha`` on linear main, ``(off-main)``, or ``(unknown)``."""
    if _git(repo_root, "cat-file", "-e", f"{sha}^{{commit}}") is None:
        return "(unknown)"
    if _git(repo_root, "merge-base", "--is-ancestor", sha, "main") is None:
        return "(off-main)"
    return _git(repo_root, "rev-list", "--count", sha) or "(unknown)"


def commit_date(repo_root: Path, sha: str) -> str:
    """Committer date of ``sha`` as ``YYYY.MM.DD``."""
    return (
        _git(repo_root, "show", "-s", "--format=%cd", "--date=format:%Y.%m.%d", sha)
        or "(unknown)"
    )


def main_build(repo_root: Path) -> tuple[str, str]:
    """``(<date>-<sha>, subject)`` of local main, the reference a served release lags."""
    sha = (
        _git(repo_root, "rev-parse", f"--short={_COMMIT_CHARS}", "main") or "(no main)"
    )
    date = (
        _git(repo_root, "show", "-s", "--format=%cd", "--date=format:%Y.%m.%d", "main")
        or "????.??.??"
    )
    subject = _git(repo_root, "log", "-1", "--format=%s", "main") or ""
    return f"{date}-{sha}", subject


def behind_main(repo_root: Path, sha: str) -> str:
    """How far a served commit lags local main."""
    if _git(repo_root, "cat-file", "-e", f"{sha}^{{commit}}") is None:
        return f"(commit {sha} not in local history)"
    behind = _git(repo_root, "rev-list", "--count", f"{sha}..main") or "?"
    ahead = _git(repo_root, "rev-list", "--count", f"main..{sha}") or "?"
    if behind == "0" and ahead == "0":
        return "up to date with main"
    if ahead == "0":
        return f"{behind} behind main"
    return f"{behind} behind, {ahead} ahead of main"


def package_label(release: KgRelease | None) -> str:
    """The PKG cell: the tag when the build ran from one, else the version marked untagged."""
    if release is None or release.torchcell_version is None:
        return "-"
    if release.torchcell_tag is None:
        return f"{release.torchcell_version} (untagged)"
    return release.torchcell_tag


def status_rows(
    label: str, served: list[ServedDatabase], repo_root: Path | None
) -> list[list[str]]:
    """Table rows for one host: HOST DATABASE VERSION RELEASE PKG COMMIT# DATE DATASETS NODES ALIASES STATUS."""
    rows: list[list[str]] = []
    for db in served:
        release = db.release
        commit = release.torchcell_commit if release else None
        status = db.status
        if db.fault:
            status = f"faulting ({db.fault})"
        elif db.status == "online" and release is None and db.n_datasets:
            status = "online (no release node)"
        rows.append(
            [
                label,
                db.name + (" [default]" if db.default else ""),
                release.version if release else "-",
                release.release if release else "-",
                package_label(release),
                commit_index(repo_root, commit) if (repo_root and commit) else "-",
                commit_date(repo_root, commit) if (repo_root and commit) else "-",
                str(db.n_datasets) if db.n_datasets is not None else "-",
                f"{db.n_nodes:,}" if db.n_nodes is not None else "-",
                ",".join(db.aliases) or "-",
                status,
            ]
        )
    return rows


def format_table(rows: list[list[str]]) -> str:
    """Fixed-width table with the header row."""
    header = [
        "HOST", "DATABASE", "VERSION", "RELEASE", "PKG", "COMMIT#", "DATE",
        "DATASETS", "NODES", "ALIASES", "STATUS",
    ]  # fmt: skip
    table = [header, *rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(header))]
    return "\n".join(
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        for row in table
    )


# --------------------------------------------------------------------------- artifacts


class PointerCheck(BaseModel):
    """One distinct pointed file and what tc-data's manifest says about it."""

    ref: str  # tc://<tier>/<key>/<path>
    sha256: str  # what the pointer pins
    state: Literal["listed", "sha256_mismatch", "missing"]
    listed_sha256: str | None = None  # what the manifest lists, when it lists the path
    datasets: list[str]  # the served datasets whose records point at the file


class ArtifactProbe(BaseModel):
    """The verdict of ``artifacts``: one ops line plus the full classification."""

    state: Literal["ok", "warn", "fail"]
    code: str
    detail: str
    release: str | None = None
    checks: list[PointerCheck] = Field(default_factory=list)

    def line(self) -> str:
        """``STATE<TAB>CODE<TAB>DETAIL``, the form ``scripts/ops.sh`` renders."""
        return f"{self.state}\t{self.code}\t{self.detail}"


def read_release_bounded(
    uri: str, user: str, password: str, database: str, *, timeout_s: float
) -> KgRelease | None:
    """:func:`read_release` with a wall-clock bound; ``TimeoutError`` past it.

    The read runs on a daemon thread: a driver call that never returns (a host that
    accepts the TCP connection and then hangs) must not hold the interpreter open at
    exit, or the verdict printed after the bound would never reach the panel, whose
    outer ``timeout`` would fire first.
    """
    import threading

    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = read_release(uri, user, password, database)
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the caller's thread
            box["error"] = exc

    thread = threading.Thread(target=run, name="read-release", daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise TimeoutError(f"read_release did not answer within {timeout_s} s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def classify_pointers(
    artifact_refs: Mapping[str, list[ArtifactPointer]], source: Any
) -> list[PointerCheck]:
    """Each distinct pointed file, classified against ``source``'s manifests.

    Pointers are deduplicated across datasets by ``(tier, key, path, sha256)`` and grouped by ``(tier, key)``,
    so each key's manifest is fetched once. A file is ``listed`` when the manifest lists
    its path with the pinned sha256, ``sha256_mismatch`` when it lists the path with
    another, and ``missing`` when the path is absent or the source has no such key (a
    ``RemoteMissError``, which includes a tier tc-data does not serve). Any other error
    of the source propagates.
    """
    from torchcell.artifacts.resolve import RemoteMissError
    from torchcell.artifacts.tiers import find_record

    pointed: dict[tuple[str, str, str, str], tuple[ArtifactPointer, list[str]]] = {}
    for name in sorted(artifact_refs):
        for ref in artifact_refs[name]:
            entry = pointed.setdefault(ref.sort_key(), (ref, []))
            if name not in entry[1]:
                entry[1].append(name)
    by_key: dict[tuple[str, str], list[tuple[ArtifactPointer, list[str]]]] = (
        defaultdict(list)
    )
    for file_key in sorted(pointed):
        ref, names = pointed[file_key]
        by_key[(ref.tier, ref.key)].append((ref, names))
    checks: list[PointerCheck] = []
    for (tier, tier_key), members in by_key.items():
        try:
            records = source.manifest(tier, tier_key).files
        except RemoteMissError:
            records = []
        for ref, names in members:
            record = find_record(records, ref.path)
            if record is None:
                state: Literal["listed", "sha256_mismatch", "missing"] = "missing"
            elif record.sha256 == ref.sha256:
                state = "listed"
            else:
                state = "sha256_mismatch"
            checks.append(
                PointerCheck(
                    ref=ref.uri,
                    sha256=ref.sha256,
                    state=state,
                    listed_sha256=None if record is None else record.sha256,
                    datasets=names,
                )
            )
    return checks


def artifact_probe(
    release: KgRelease | None,
    *,
    tc_data_url: str,
    api_key: str | None,
    api_key_env: str = DEFAULT_TC_DATA_API_KEY_ENV,
    http: Any = None,
    timeout_s: float = 5.0,
) -> ArtifactProbe:
    """Whether tc-data lists every file ``release`` points at, with its sha256.

    The checks run in the order that needs the least: a store without a release node or
    a release that predates pointer recording is a warning (nothing to check), a release
    pointing at no file is ok without asking tc-data, and only then are the key and the
    endpoint required. ``http`` injects the client (tests).
    """
    import httpx

    from torchcell.artifacts.resolve import RemoteEndpointError, TcDataSource

    if release is None:
        return ArtifactProbe(state="warn", code="n/a", detail="no release node")
    tag = release.release
    if release.artifact_refs is None:
        return ArtifactProbe(
            state="warn",
            code="n/a",
            detail="release predates pointer recording",
            release=tag,
        )
    if not any(release.artifact_refs.values()):
        return ArtifactProbe(
            state="ok",
            code="0",
            detail="the release points at no artifact file",
            release=tag,
        )
    if not api_key:
        return ArtifactProbe(
            state="fail", code="n/a", detail=f"{api_key_env} unset", release=tag
        )
    source = TcDataSource(tc_data_url, api_key, http=http, timeout=timeout_s)
    try:
        checks = classify_pointers(release.artifact_refs, source)
    except (RemoteEndpointError, httpx.HTTPError) as exc:
        return ArtifactProbe(
            state="fail",
            code="n/a",
            detail=f"tc-data unreachable: {_fault_text(exc)}",
            release=tag,
        )
    n_total = len(checks)
    listed = sum(check.state == "listed" for check in checks)
    if listed == n_total:
        return ArtifactProbe(
            state="ok",
            code=f"{n_total}/{n_total}",
            detail="every pointer listed by tc-data with its sha256",
            release=tag,
            checks=checks,
        )
    failing = [check for check in checks if check.state != "listed"]
    missing = sum(check.state == "missing" for check in failing)
    shown = ", ".join(check.ref for check in failing[:_FAILING_SHOWN])
    more = ", ..." if len(failing) > _FAILING_SHOWN else ""
    return ArtifactProbe(
        state="fail",
        code=f"{listed}/{n_total}",
        detail=(
            f"{missing} missing, {len(failing) - missing} sha256 mismatch: "
            f"{shown}{more}"
        ),
        release=tag,
        checks=checks,
    )


# --------------------------------------------------------------------------- CLI


def _host_spec(spec: str, user: str, password: str) -> tuple[str, str, str, str]:
    """``LABEL=URI[|USER|PASSWORD]`` as ``(label, uri, user, password)``; missing parts inherit."""
    label, _, rest = spec.partition("=")
    parts = rest.split("|")
    return (
        label,
        parts[0],
        parts[1] if len(parts) > 1 else user,
        parts[2] if len(parts) > 2 else password,
    )


def _connection(args: argparse.Namespace) -> tuple[str, str, str]:
    from torchcell.database.connection import neo4j_connection_settings

    settings = neo4j_connection_settings()
    return (
        args.uri or settings.uri,
        args.user or settings.username,
        args.password or settings.password,
    )


def _release_for(version: str, uri: str, user: str, password: str) -> KgRelease:
    database = resolve_database(version, uri, user, password)
    release = read_release(uri, user, password, database)
    if release is None:
        raise LookupError(f"{database} carries no {RELEASE_LABEL} node")
    return release


def main(argv: list[str] | None = None) -> int:
    """Command-line entry: status, datasets, diff, compat, hashes, write-node, stamp."""
    parser = argparse.ArgumentParser(
        prog="python -m torchcell.knowledge_graphs.releases",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("--uri", default=None, help="bolt URI (default NEO4J_URI)")
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument(
        "--repo", default=None, help="torchcell checkout for git columns"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="one table row per served database")
    p_status.add_argument("--label", default="gilahyper", help="the HOST column")
    p_status.add_argument(
        "--host",
        action="append",
        default=None,
        help="LABEL=URI[|USER|PASSWORD]; repeat for several hosts in one aligned "
        "table (overrides --uri/--label; a host that does not answer within "
        "--host-timeout seconds gets one unreachable row)",
    )
    p_status.add_argument("--host-timeout", type=float, default=20.0)
    p_status.add_argument("--no-header", action="store_true")
    p_status.add_argument("--json", action="store_true")

    p_ds = sub.add_parser("datasets", help="datasets a version serves")
    p_ds.add_argument("--version", default=DEFAULT_VERSION)
    p_ds.add_argument("--json", action="store_true")

    p_diff = sub.add_parser("diff", help="byte-identity diff between two versions")
    p_diff.add_argument("a")
    p_diff.add_argument("b")
    p_diff.add_argument("--json", action="store_true")

    p_compat = sub.add_parser("compat", help="local schema surface vs a served release")
    p_compat.add_argument("--version", default=DEFAULT_VERSION)

    p_hash = sub.add_parser("hashes", help="content hashes from CSVs or the store")
    p_hash.add_argument("--csv-dir", default=None, help="BioCypher output directory")
    p_hash.add_argument("--database", default=None, help="stream from this database")
    p_hash.add_argument(
        "--output", required=True, help="write Dataset.id -> sha256 JSON"
    )

    p_stamp = sub.add_parser(
        "stamp", help="give a manifest its version, release id, and content hashes"
    )
    p_stamp.add_argument("--manifest", required=True)
    p_stamp.add_argument("--kind", choices=["full", "incremental"], required=True)
    p_stamp.add_argument("--built-at", required=True)
    p_stamp.add_argument("--hashes", required=True, help="JSON from `hashes`")
    p_stamp.add_argument(
        "--previous-version", default=None, help="version of the store this replaces"
    )

    p_snap = sub.add_parser(
        "snapshot",
        help="write database/releases/<release>.json (+ .closures.json) from a "
        "stamped manifest, for the committed compatibility page",
    )
    p_snap.add_argument("--manifest", required=True)
    p_snap.add_argument("--n-nodes", type=int, default=None)
    p_snap.add_argument(
        "--built-at",
        default=None,
        help="the stamp's timestamp (default: the manifest's last event time)",
    )
    p_snap.add_argument(
        "--repo-root",
        default=None,
        help="checkout whose database/releases/ receives the files (default: the "
        "checkout this torchcell was imported from)",
    )
    p_snap.add_argument(
        "--torchcell-version",
        default=None,
        help="BOOTSTRAP ONLY: the package version for a manifest stamped before "
        "the versioning spine (refused when the manifest already records one)",
    )
    p_snap.add_argument(
        "--torchcell-tag",
        default=None,
        help="BOOTSTRAP ONLY: the tag on the release commit, with --torchcell-version",
    )
    p_snap.add_argument(
        "--note",
        default=None,
        help="BOOTSTRAP ONLY: how the two values were derived; appended to the "
        "snapshot's last event note",
    )

    p_retag = sub.add_parser(
        "retag",
        help="pair a committed snapshot (and the manifest) with a package tag cut "
        "after the build, once the surface at that tag reproduces every closure",
    )
    p_retag.add_argument("--release", required=True, help="the release id")
    p_retag.add_argument("--tag", required=True, help="the package tag, vX.Y.Z")
    p_retag.add_argument(
        "--repo-root",
        default=None,
        help="checkout holding database/releases/ and the tag (default: the checkout "
        "this torchcell was imported from)",
    )
    p_retag.add_argument(
        "--manifest",
        default=None,
        help="also record the pairing in this kg_manifest.json, so `write-node` "
        "carries it into the store",
    )

    p_node = sub.add_parser(
        "write-node", help="write the KgRelease node from a manifest"
    )
    p_node.add_argument("--manifest", required=True)
    p_node.add_argument("--database", required=True)
    p_node.add_argument("--built-at", required=True)
    p_node.add_argument("--n-nodes", type=int, default=None)

    p_art = sub.add_parser(
        "artifacts",
        help="whether tc-data lists every file a served release points at, with its "
        "sha256; one STATE<TAB>CODE<TAB>DETAIL line (exit 0 always)",
    )
    p_art.add_argument("--host", required=True, help="LABEL=URI[|USER|PASSWORD]")
    p_art.add_argument("--database", required=True)
    p_art.add_argument("--tc-data-url", required=True)
    p_art.add_argument("--host-timeout", type=float, default=20.0)
    p_art.add_argument(
        "--tc-data-timeout", type=float, default=5.0, help="seconds per tc-data request"
    )
    p_art.add_argument(
        "--api-key-env",
        default=DEFAULT_TC_DATA_API_KEY_ENV,
        help="the environment variable holding the tc-data key (read after the repo "
        ".env is loaded; the key itself is never printed)",
    )
    p_art.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    repo_root = Path(args.repo).resolve() if args.repo else None

    if args.command == "hashes":
        if (args.csv_dir is None) == (args.database is None):
            parser.error("give exactly one of --csv-dir or --database")
        if args.csv_dir:
            hashes = content_hashes_from_csv(Path(args.csv_dir))
        else:
            uri, user, password = _connection(args)
            hashes = content_hashes_from_store(uri, user, password, args.database)
        Path(args.output).write_text(json.dumps(hashes, indent=1), encoding="utf-8")
        print(f"{len(hashes)} content hashes -> {args.output}")
        return 0

    if args.command == "stamp":
        from torchcell.knowledge_graphs.kg_manifest import load_manifest, save_manifest

        manifest = load_manifest(Path(args.manifest))
        hashes = json.loads(Path(args.hashes).read_text(encoding="utf-8"))
        torchcell_version, torchcell_tag = checkout_package_version(
            repo_root or package_checkout()
        )
        stamp_manifest(
            manifest,
            kind=args.kind,
            built_at=args.built_at,
            content_hashes=hashes,
            previous_version=args.previous_version,
            torchcell_version=torchcell_version,
            torchcell_tag=torchcell_tag,
        )
        save_manifest(manifest, Path(args.manifest))
        print(
            f"{args.manifest}: version {manifest.version}, release {manifest.release}, "
            f"torchcell {torchcell_version} ({torchcell_tag or 'untagged'})"
        )
        return 0

    if args.command == "snapshot":
        from torchcell.knowledge_graphs.kg_manifest import load_manifest
        from torchcell.knowledge_graphs.release_snapshot import (
            bootstrap_package_version,
            snapshot_from_manifest,
            write_snapshot,
        )

        manifest = load_manifest(Path(args.manifest))
        snapshot = snapshot_from_manifest(
            manifest, n_nodes=args.n_nodes, built_at=args.built_at
        )
        if args.torchcell_version is not None:
            snapshot = bootstrap_package_version(
                snapshot,
                torchcell_version=args.torchcell_version,
                torchcell_tag=args.torchcell_tag,
                note=args.note,
            )
        target = (
            Path(args.repo_root).resolve() if args.repo_root else package_checkout()
        )
        paths = write_snapshot(
            snapshot,
            {name: dict(entry.closure) for name, entry in manifest.datasets.items()},
            target,
        )
        print(
            f"{snapshot.release}: torchcell {snapshot.torchcell_version} "
            f"({snapshot.torchcell_tag or 'untagged'}), composite "
            f"{snapshot.composite_sha256} -> {paths[0]}, {paths[1]}"
        )
        return 0

    if args.command == "retag":
        from torchcell.knowledge_graphs.kg_manifest import (
            load_manifest,
            save_manifest,
            surface_at_ref,
        )
        from torchcell.knowledge_graphs.release_snapshot import (
            load_closures,
            load_snapshot,
            pair_package_tag,
            snapshot_paths,
            write_snapshot,
        )

        root = Path(args.repo_root).resolve() if args.repo_root else package_checkout()
        snapshot_path, _ = snapshot_paths(root, args.release)
        snapshot = load_snapshot(snapshot_path)
        closures = load_closures(root, args.release)
        paired = pair_package_tag(
            snapshot, closures, args.tag, surface_at_ref(root, args.tag)
        )
        write_snapshot(paired, closures, root)
        if args.manifest:
            manifest = load_manifest(Path(args.manifest))
            if manifest.release != paired.release:
                raise SystemExit(
                    f"{args.manifest} describes release {manifest.release}, "
                    f"not {paired.release}"
                )
            manifest.torchcell_version = paired.torchcell_version
            manifest.torchcell_tag = paired.torchcell_tag
            save_manifest(manifest, Path(args.manifest))
        print(
            f"{paired.release}: paired with {args.tag} (torchcell "
            f"{paired.torchcell_version}) -> {snapshot_path}"
            + (f", {args.manifest}" if args.manifest else "")
        )
        return 0

    uri, user, password = _connection(args)

    if args.command == "write-node":
        from torchcell.knowledge_graphs.kg_manifest import load_manifest

        manifest = load_manifest(Path(args.manifest))
        hashes = {
            name: entry.content_sha256 or ""
            for name, entry in manifest.datasets.items()
        }
        if any(not h for h in hashes.values()):
            raise SystemExit(
                "manifest has datasets without content_sha256; stamp it first"
            )
        release = release_from_manifest(
            manifest,
            built_at=args.built_at,
            content_hashes=hashes,
            n_nodes=args.n_nodes,
        )
        write_release(uri, user, password, args.database, release)
        print(
            f"{args.database}: {RELEASE_LABEL} {release.release} (version {release.version})"
        )
        return 0

    if args.command == "artifacts":
        load_dotenv()
        _, host_uri, host_user, host_password = _host_spec(args.host, user, password)
        try:
            served = read_release_bounded(
                host_uri,
                host_user,
                host_password,
                args.database,
                timeout_s=args.host_timeout,
            )
        except TimeoutError:
            probe = ArtifactProbe(
                state="fail",
                code="n/a",
                detail=f"host unreachable within {args.host_timeout:g} s",
            )
        except Exception as exc:  # a reporter: refusal, auth or a store fault is a line
            probe = ArtifactProbe(
                state="fail",
                code="n/a",
                detail=f"release read failed: {_fault_text(exc)}",
            )
        else:
            probe = artifact_probe(
                served,
                tc_data_url=args.tc_data_url,
                api_key=os.environ.get(args.api_key_env),
                api_key_env=args.api_key_env,
                timeout_s=args.tc_data_timeout,
            )
        # flushed: ops.sh bounds this process, and a hung driver thread can hold exit
        print(
            probe.model_dump_json(indent=1) if args.json else probe.line(), flush=True
        )
        return 0

    if args.command == "status":
        hosts: list[tuple[str, str, str, str]] = [(args.label, uri, user, password)]
        if args.host:
            hosts = [_host_spec(spec, user, password) for spec in args.host]
        by_host = {
            label: list_databases_bounded(u, usr, pw, timeout_s=args.host_timeout)
            for label, u, usr, pw in hosts
        }
        if args.json:
            print(
                json.dumps(
                    {k: [db.model_dump() for db in v] for k, v in by_host.items()},
                    indent=1,
                )
            )
            return 0
        rows = [
            row
            for label, served in by_host.items()
            for row in status_rows(label, served, repo_root)
        ]
        text = format_table(rows)
        print("\n".join(text.splitlines()[1:]) if args.no_header else text)
        return 0

    if args.command == "datasets":
        listed = datasets(args.version, uri, user, password)
        if args.json:
            print(json.dumps([d.model_dump() for d in listed], indent=1))
            return 0
        for d in listed:
            print(f"{d.dataset_class}\t{d.n_experiments}\t{d.content_sha256}")
        return 0

    if args.command == "diff":
        result = diff(
            _release_for(args.a, uri, user, password),
            _release_for(args.b, uri, user, password),
        )
        if args.json:
            print(result.model_dump_json(indent=1))
            return 0
        print(f"{result.from_release} -> {result.to_release}")
        for kind in ("unchanged", "changed", "added", "removed"):
            names = getattr(result, kind)
            print(f"  {kind} ({len(names)}): {', '.join(names) or '-'}")
        return 0

    if args.command == "compat":
        if repo_root is None:
            parser.error("--repo is required for compat")
        report = compatibility(
            _release_for(args.version, uri, user, password), repo_root
        )
        print(
            f"release {report.release} ({report.torchcell_commit[:8]}) vs {repo_root}: "
            f"{len(report.compatible)} compatible, {len(report.drifted)} drifted, "
            f"{len(report.unchecked)} unchecked"
        )
        for drift in report.drifted:
            print(f"  {drift.dataset_class}: {', '.join(drift.changed_symbols)}")
        return 0 if report.ok else 1

    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
