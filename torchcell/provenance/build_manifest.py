# torchcell/provenance/build_manifest.py
# [[torchcell.provenance.build_manifest]]
"""Per-dataset build manifests: the authoritative "which built LMDB is now stale" record.

When a dataset's ``process()`` finishes, :func:`write_build_manifest` drops a
``build_manifest.json`` into that dataset's ``preprocess/`` directory (beside
``verification_report.json``) recording the contract fingerprint of every schema symbol the
loader transitively depends on. A later :func:`check_all` recomputes those fingerprints from
the *local* schema and flags any dataset whose stored fingerprints no longer match.

Multi-machine design:
  - Fingerprints are content-addressed (see ``schema_deps``), so a manifest is meaningful on
    any machine; staleness is judged against each machine's LOCAL schema, no network needed.
  - Manifests live under ``DATA_ROOT`` (per-machine, next to the LMDB), NOT in git -- so each
    machine tracks its own built artifacts with zero cross-machine git contention.
  - ``torchcell_commit`` is recorded for provenance only and is NEVER used for the staleness
    decision (a schema commit that doesn't touch a loader's closure leaves it fresh). It is
    ``None`` when the built code is not a git checkout (e.g. an installed wheel).

Fingerprints alone cannot see every stale store (issue #833). A record pickled under a
class the local schema no longer DEFINES is in no closure, so its manifest compares equal
and the store reads fresh while nothing can deserialize it: 5 bacterial dev stores were in
that state on 2026-10-09 and ``--list-stale`` named 1. :func:`check_store` therefore ends
with a bounded read of the store's FIRST record (:func:`read_first_record`) and reports
``unreadable`` with the exception when it fails. It is the ONE freshness function: the
``build_manifest`` CLI (:func:`check_all`), ``build_dataset_lmdb --list-stale`` and the
live rebuild's preflight all call it, so the three cannot drift apart.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from torchcell.provenance.schema_deps import (
    SchemaSurface,
    load_default_surface,
    loader_closure,
)

__all__ = [
    "BuildManifest",
    "SymbolDrift",
    "StaleResult",
    "DatasetCheck",
    "StoreFreshness",
    "StoreState",
    "compute_manifest",
    "write_build_manifest",
    "check_manifest",
    "check_store",
    "read_first_record",
    "check_all",
    "MANIFEST_FILENAME",
]

MANIFEST_FILENAME = "build_manifest.json"
MANIFEST_SCHEMA_VERSION = 1

StoreState = Literal["fresh", "stale", "no_lmdb", "no_manifest", "unreadable"]
"""What one built store is, against the local schema. Only ``fresh`` needs no rebuild."""


class BuildManifest(BaseModel):
    """Records the schema contract a built LMDB was produced against."""

    manifest_schema_version: int = MANIFEST_SCHEMA_VERSION
    dataset_name: str  # slug, e.g. "env_chemgen_hoepfner2014"
    loader_class: str  # e.g. "EnvChemgenHoepfner2014Dataset"
    loader_module: str  # e.g. "torchcell.datasets.scerevisiae.hoepfner2014"
    surface_modules: list[
        str
    ]  # basenames of the schema-surface files, e.g. ["schema.py", ...]
    closure: dict[
        str, str
    ]  # symbol -> contract fingerprint (THE staleness key), sorted
    built_at: str  # ISO-8601 UTC
    hostname: str
    torchcell_commit: str | None = None  # provenance only; NOT used for staleness
    torchcell_dirty: bool | None = None


class SymbolDrift(BaseModel):
    """One schema symbol whose contract fingerprint no longer matches the manifest."""

    symbol: str
    stored_fingerprint: str
    current_fingerprint: str | None  # None == symbol removed from the schema entirely


class StaleResult(BaseModel):
    """Result of checking one manifest against the current schema surface."""

    dataset_name: str
    preprocess_dir: str
    is_stale: bool
    drift: list[SymbolDrift]


class StoreFreshness(BaseModel):
    """One built store's state: the answer :func:`check_store` gives about one directory."""

    model_config = ConfigDict(frozen=True)

    root: str
    state: StoreState
    drift: list[str] = Field(default_factory=list)
    reason: str | None = None  # set only for ``unreadable``: "<ExcClass>: <message>"

    @property
    def needs_rebuild(self) -> bool:
        """Everything but ``fresh`` is rebuilt before a full knowledge-graph build."""
        return self.state != "fresh"

    def describe(self) -> str:
        """One line naming what is wrong, for a preflight's refusal list."""
        if self.state == "fresh":
            return "fresh"
        if self.state == "stale":
            return f"stale on {self.drift}"
        if self.state == "no_lmdb":
            return f"no LMDB at {self.root}"
        if self.state == "no_manifest":
            return "no build manifest"
        return f"unreadable: {self.reason}"


class DatasetCheck(BaseModel):
    """Per-dataset outcome of a fleet-wide staleness scan."""

    dataset_name: str
    status: StoreState
    drift: list[SymbolDrift] = Field(default_factory=list)
    reason: str | None = None


def _git_info(repo_dir: Path) -> tuple[str | None, bool | None]:
    """Best-effort (commit, dirty) for provenance; ``(None, None)`` if not a git checkout.

    Uses explicit return-code checks (never ``check=True``), so a non-checkout is modeled as
    absent metadata rather than a masked error -- consistent with the no-fallback rule since
    the commit is provenance-only and never gates staleness.
    """
    head = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if head.returncode != 0:
        return None, None
    status = subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return head.stdout.strip(), dirty


def compute_manifest(
    *,
    dataset_name: str,
    loader_module: str,
    loader_class: str,
    loader_path: Path,
    surface: SchemaSurface,
    built_at: str,
    hostname: str,
    torchcell_commit: str | None,
    torchcell_dirty: bool | None,
) -> BuildManifest:
    """Build a :class:`BuildManifest` from a loader's transitive schema closure."""
    closure = loader_closure(loader_path, surface)
    surface_modules = sorted(
        {osp.basename(module) for module in surface.module_of.values()}
    )
    return BuildManifest(
        dataset_name=dataset_name,
        loader_class=loader_class,
        loader_module=loader_module,
        surface_modules=surface_modules,
        closure={name: surface.fingerprints[name] for name in sorted(closure)},
        built_at=built_at,
        hostname=hostname,
        torchcell_commit=torchcell_commit,
        torchcell_dirty=torchcell_dirty,
    )


def write_build_manifest(dataset: Any) -> Path:
    """Compute and write ``build_manifest.json`` for a freshly built dataset instance.

    Called from the ``post_process`` build hook (``torchcell/data/experiment_dataset.py``),
    so every dataset build records its schema contract automatically. ``dataset`` is typed
    ``Any`` to avoid importing ``ExperimentDataset`` here (which would create an import cycle,
    since that module imports this one).
    """
    cls = type(dataset)
    module = sys.modules[cls.__module__]
    module_file = module.__file__
    assert module_file is not None, f"loader module {cls.__module__} has no __file__"
    loader_path = Path(module_file)
    surface = load_default_surface()
    commit, dirty = _git_info(loader_path.parent)
    manifest = compute_manifest(
        dataset_name=osp.basename(osp.normpath(dataset.root)),
        loader_module=cls.__module__,
        loader_class=cls.__name__,
        loader_path=loader_path,
        surface=surface,
        built_at=datetime.now(UTC).isoformat(),
        hostname=socket.gethostname(),
        torchcell_commit=commit,
        torchcell_dirty=dirty,
    )
    out = Path(dataset.preprocess_dir) / MANIFEST_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return out


def check_manifest(
    manifest: BuildManifest, surface: SchemaSurface, preprocess_dir: str
) -> StaleResult:
    """Compare a manifest's stored fingerprints against the current local schema surface."""
    drift = [
        SymbolDrift(
            symbol=symbol,
            stored_fingerprint=stored,
            current_fingerprint=surface.fingerprints.get(symbol),
        )
        for symbol, stored in manifest.closure.items()
        if surface.fingerprints.get(symbol) != stored
    ]
    return StaleResult(
        dataset_name=manifest.dataset_name,
        preprocess_dir=preprocess_dir,
        is_stale=bool(drift),
        drift=drift,
    )


def read_first_record(root: str | Path) -> dict[str, Any]:
    """Deserialize the FIRST record of ``<root>/processed/lmdb`` and type it; return it.

    The bounded read behind the ``unreadable`` state. It is the store's own read path:
    the interned sibling env, ``pickle.loads``, ``resolve_interned`` (what
    ``ExperimentDataset.get_single_item`` does), then the three stored dicts validated
    into the classes their own ``experiment_type`` /
    ``experiment_reference_type`` name. Both ways a store goes unreadable raise here:
    a record (or an interned sub-object) pickled under a class the local schema no
    longer defines fails in ``pickle.loads``, and a record carrying a field the class
    no longer accepts fails pydantic with ``extra_forbidden``.

    ONE record, because unpickling is per-record and the classes a store uses are fixed
    at build time: reading record 0 exercises every class in its records' closure, and
    the interned table it loads first carries the reference and publication objects of
    the whole store. Raises for the caller to report; never returns a verdict.
    """
    import pickle

    import lmdb

    from torchcell.data.experiment_dataset import resolve_interned
    from torchcell.datamodels.schema import (
        EXPERIMENT_REFERENCE_TYPE_MAP,
        EXPERIMENT_TYPE_MAP,
        Publication,
    )

    processed = Path(root) / "processed"
    interned: dict[str, Any] = {}
    if (processed / "interned").is_dir():
        ienv = lmdb.open(str(processed / "interned"), readonly=True, lock=False)
        try:
            with ienv.begin() as itxn:
                for key, value in itxn.cursor():
                    interned[key.decode()] = pickle.loads(value)  # noqa: S301
        finally:
            ienv.close()
    env = lmdb.open(str(processed / "lmdb"), readonly=True, lock=False, readahead=False)
    try:
        with env.begin() as txn:
            cursor = txn.cursor()
            if not cursor.first():
                raise ValueError(f"{processed / 'lmdb'} holds no records")
            raw = cursor.value()
    finally:
        env.close()
    record: dict[str, Any] = resolve_interned(pickle.loads(raw), interned)  # noqa: S301
    experiment = record["experiment"]
    EXPERIMENT_TYPE_MAP[experiment["experiment_type"]](**experiment)
    reference = record["reference"]
    EXPERIMENT_REFERENCE_TYPE_MAP[reference["experiment_reference_type"]](**reference)
    Publication(**record["publication"])
    return record


def check_store(root: str | Path, surface: SchemaSurface) -> StoreFreshness:
    """The ONE freshness verdict on one built dataset directory.

    In order: the LMDB must exist, its build manifest must exist, every fingerprint in
    the manifest's closure must equal the current surface's, and the store's first
    record must deserialize. The read runs LAST and only on a store whose fingerprints
    all match, since a store already reported ``stale`` is rebuilt either way and the
    read costs an LMDB open.

    ``except Exception`` around the read is the probe's product, not a swallowed error:
    the state and the exception text are reported as the verdict, and nothing continues
    as though the store were readable.
    """
    lmdb_dir = Path(root) / "processed" / "lmdb"
    preprocess = Path(root) / "preprocess"
    if not lmdb_dir.is_dir():
        return StoreFreshness(root=str(root), state="no_lmdb")
    manifest_path = preprocess / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return StoreFreshness(root=str(root), state="no_manifest")
    manifest = BuildManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    result = check_manifest(manifest, surface, str(preprocess))
    if result.is_stale:
        return StoreFreshness(
            root=str(root),
            state="stale",
            drift=sorted({d.symbol for d in result.drift}),
        )
    try:
        read_first_record(root)
    except Exception as error:  # the probe's verdict; reported, never ignored
        return StoreFreshness(
            root=str(root),
            state="unreadable",
            reason=f"{type(error).__name__}: {error}",
        )
    return StoreFreshness(root=str(root), state="fresh")


def _built_dataset_dirs(data_root: Path) -> list[Path]:
    """Directories under DATA_ROOT that hold a built LMDB (``<slug>/processed/lmdb``)."""
    return sorted(
        {
            lmdb.parent.parent
            for lmdb in data_root.glob("data/torchcell/*/processed/lmdb")
        }
    )


def check_all(data_root: Path, surface: SchemaSurface) -> list[DatasetCheck]:
    """Check every built dataset under DATA_ROOT for schema staleness.

    Per directory this is :func:`check_store`, so the scan sees an unreadable store the
    same way ``build_dataset_lmdb --list-stale`` and the live rebuild's preflight do.
    The reported name is the manifest's own ``dataset_name`` where there is a manifest
    to read it from, and the directory name otherwise.
    """
    results: list[DatasetCheck] = []
    for slug_dir in _built_dataset_dirs(data_root):
        freshness = check_store(slug_dir, surface)
        manifest_path = slug_dir / "preprocess" / MANIFEST_FILENAME
        name = slug_dir.name
        drift: list[SymbolDrift] = []
        if manifest_path.is_file():
            manifest = BuildManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            name = manifest.dataset_name
            drift = check_manifest(
                manifest, surface, str(slug_dir / "preprocess")
            ).drift
        results.append(
            DatasetCheck(
                dataset_name=name,
                status=freshness.state,
                drift=drift,
                reason=freshness.reason,
            )
        )
    return results


def _format_report(checks: list[DatasetCheck]) -> str:
    stale = [c for c in checks if c.status == "stale"]
    unreadable = [c for c in checks if c.status == "unreadable"]
    unmanifested = [c for c in checks if c.status in ("no_manifest", "no_lmdb")]
    fresh = [c for c in checks if c.status == "fresh"]
    lines: list[str] = [
        f"Built datasets: {len(checks)}  "
        f"(fresh {len(fresh)}, stale {len(stale)}, unreadable {len(unreadable)}, "
        f"unmanifested {len(unmanifested)})",
        "",
    ]
    for check in stale:
        symbols = ", ".join(sorted(d.symbol for d in check.drift))
        lines.append(f"  [STALE] {check.dataset_name}  -> rebuild; changed: {symbols}")
    for check in unreadable:
        lines.append(
            f"  [UNREADABLE] {check.dataset_name}  -> rebuild; unreadable: "
            f"{check.reason}"
        )
    for check in unmanifested:
        lines.append(
            f"  [no manifest] {check.dataset_name}  -> written on next rebuild"
        )
    if not stale and not unreadable and not unmanifested:
        lines.append("  All built datasets are fresh against the local schema.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m torchcell.provenance.build_manifest",
        description="Check built datasets under DATA_ROOT for schema-contract staleness.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="root holding data/torchcell/<slug>/ (default: $DATA_ROOT from the environment)",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = Path(args.data_root or os.environ["DATA_ROOT"])
    checks = check_all(data_root, load_default_surface())
    print(_format_report(checks))
    return 1 if any(c.status in ("stale", "unreadable") for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
