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

    python -m torchcell.knowledge_graphs.releases status --label gilahyper
    python -m torchcell.knowledge_graphs.releases datasets --version latest
    python -m torchcell.knowledge_graphs.releases diff 2026.09.17-7715ee35 latest
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from torchcell.knowledge_graphs.kg_manifest import KgBuildManifest, surface_in_worktree

RELEASE_LABEL = "KgRelease"
LATEST_ALIAS = "latest"
PINNED_ALIAS = "pinned"
DEFAULT_VERSION = LATEST_ALIAS
SYSTEM_DATABASES = ("system",)
_COMMIT_CHARS = 8


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
    built_at: str
    biocypher_out: str
    neo4j_version: str | None = None
    n_nodes: int | None = None
    datasets: dict[str, ReleaseDataset]
    # dataset -> schema symbol -> contract fingerprint at build; the compatibility check
    closures: dict[str, dict[str, str]] = Field(default_factory=dict)

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
        }

    @classmethod
    def from_properties(cls, props: dict[str, Any]) -> KgRelease:
        """Inverse of :meth:`to_properties`."""
        datasets = {
            k: ReleaseDataset.model_validate(v)
            for k, v in json.loads(props["datasets_json"]).items()
        }
        return cls(
            release=props["release"],
            version=props["version"],
            torchcell_commit=props["torchcell_commit"],
            built_at=props["built_at"],
            biocypher_out=props["biocypher_out"],
            neo4j_version=props.get("neo4j_version"),
            n_nodes=props.get("n_nodes"),
            datasets=datasets,
            closures=json.loads(props.get("closures_json") or "{}"),
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
        built_at=built_at,
        biocypher_out=manifest.events[-1].biocypher_out or "",
        neo4j_version=manifest.neo4j_version,
        n_nodes=n_nodes,
        datasets=datasets,
        closures={
            name: dict(entry.closure) for name, entry in manifest.datasets.items()
        },
    )


def stamp_manifest(
    manifest: KgBuildManifest,
    *,
    kind: Literal["full", "incremental"],
    built_at: str,
    content_hashes: dict[str, str],
    previous_version: str | None,
) -> KgBuildManifest:
    """Give a manifest its version, release id, and per-dataset content hashes.

    A full build passes a hash for every dataset. An incremental admission passes the
    hashes of the datasets it imported; the rest keep the hash they already carry,
    since incremental import never touches existing nodes. Any dataset left without a
    hash is an error.
    """
    if manifest.torchcell_commit is None:
        raise ValueError("the manifest records no full-build commit")
    manifest.version = next_version(previous_version, kind)
    commit = manifest.events[-1].torchcell_commit or manifest.torchcell_commit
    manifest.release = release_id(built_at, commit)
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
    surface = surface_in_worktree(repo_root)
    compatible: list[str] = []
    drifted: list[DatasetDrift] = []
    unchecked: list[str] = []
    for name in sorted(release.datasets):
        closure = release.closures.get(name)
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
        release=release.release,
        torchcell_commit=release.torchcell_commit,
        compatible=compatible,
        drifted=drifted,
        unchecked=unchecked,
    )


# --------------------------------------------------------------------------- reporting


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


def status_rows(
    label: str, served: list[ServedDatabase], repo_root: Path | None
) -> list[list[str]]:
    """Table rows for one host: HOST DATABASE VERSION RELEASE COMMIT# DATE DATASETS NODES ALIASES STATUS."""
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
        "HOST", "DATABASE", "VERSION", "RELEASE", "COMMIT#", "DATE",
        "DATASETS", "NODES", "ALIASES", "STATUS",
    ]  # fmt: skip
    table = [header, *rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(header))]
    return "\n".join(
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        for row in table
    )


# --------------------------------------------------------------------------- CLI


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

    p_node = sub.add_parser(
        "write-node", help="write the KgRelease node from a manifest"
    )
    p_node.add_argument("--manifest", required=True)
    p_node.add_argument("--database", required=True)
    p_node.add_argument("--built-at", required=True)
    p_node.add_argument("--n-nodes", type=int, default=None)

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
        stamp_manifest(
            manifest,
            kind=args.kind,
            built_at=args.built_at,
            content_hashes=hashes,
            previous_version=args.previous_version,
        )
        save_manifest(manifest, Path(args.manifest))
        print(
            f"{args.manifest}: version {manifest.version}, release {manifest.release}"
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

    if args.command == "status":
        hosts: list[tuple[str, str, str, str]] = [(args.label, uri, user, password)]
        if args.host:
            hosts = []
            for spec in args.host:
                label, _, rest = spec.partition("=")
                parts = rest.split("|")
                hosts.append(
                    (
                        label,
                        parts[0],
                        parts[1] if len(parts) > 1 else user,
                        parts[2] if len(parts) > 2 else password,
                    )
                )
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
