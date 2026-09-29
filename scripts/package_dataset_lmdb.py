# scripts/package_dataset_lmdb.py
# [[scripts.package_dataset_lmdb]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/package_dataset_lmdb.py
"""Package one built dataset directory into the ``tc-data`` artifact store.

Tars ``processed/`` (the records LMDB, the sibling ``interned`` LMDB, PyG's
``pre_filter.pt`` and ``pre_transform.pt``) and ``preprocess/`` (``build_manifest.json``,
``gene_set.json``, ``experiment_reference_index.json``, ``data.csv``) of a dataset
built under ``$DATA_ROOT/data/torchcell/<slug>/``, never ``raw/``: a consumer reads the
LMDB, and the raw files are served separately from the raw mirror. The archive is
deterministic (members in sorted order, mtime fixed to the manifest's ``built_at``,
uid/gid 0, modes normalized, LMDB ``lock.mdb`` excluded because it is a runtime lock
that changes with every reader), so packaging the same build twice yields the same
bytes, the same sha256 and the same file name. Compression is xz through the stdlib
``tarfile`` (``zstandard`` is not in the environment).

Refusals, none of them overridable: no ``preprocess/build_manifest.json`` (an
unmanifested build has no schema contract to publish), no ``processed/lmdb``, a
manifest whose ``dataset_name`` differs from the directory name, and a manifest that is
STALE against the local schema surface (the artifact would claim the packager's
``torchcell.__version__`` for an LMDB the local schema would serialize differently;
rebuild it first).

Usage (from the repo root, in the torchcell environment)::

    python scripts/package_dataset_lmdb.py
        --dataset-dir $DATA_ROOT/data/torchcell/smf_costanzo2016
        --store $TC_DATA_ROOT [--kg-release R --kg-version V --status supported]

Writes ``<store>/<slug>/<slug>-<version>-<sha8>.tar.xz``, upserts the row in
``<store>/index.json`` and rewrites ``<store>/SHA256SUMS``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, get_args

import lmdb

from torchcell import __version__
from torchcell.datasets.artifact import (
    INDEX_FILENAME,
    SHA256SUMS_FILENAME,
    ArtifactIndex,
    ArtifactStatus,
    DatasetArtifact,
    archive_name,
    content_sha256,
)
from torchcell.provenance.build_manifest import (
    MANIFEST_FILENAME,
    BuildManifest,
    check_manifest,
)
from torchcell.provenance.schema_deps import load_default_surface

PACKAGED_DIRS = ("preprocess", "processed")
EXCLUDED_BASENAMES = frozenset({"lock.mdb"})
XZ_PRESET: Literal[6] = 6
CHUNK = 1 << 20


class PackagingRefused(RuntimeError):
    """The dataset directory cannot be packaged as it stands (see the message)."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_manifest(dataset_dir: Path) -> BuildManifest:
    """``preprocess/build_manifest.json``, refusing when absent or misnamed."""
    manifest_path = dataset_dir / "preprocess" / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise PackagingRefused(
            f"no {MANIFEST_FILENAME} under {dataset_dir / 'preprocess'}: an unmanifested "
            "build has no schema contract to publish; rebuild it with "
            "torchcell.database.build_dataset_lmdb"
        )
    manifest = BuildManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.dataset_name != dataset_dir.name:
        raise PackagingRefused(
            f"manifest dataset_name {manifest.dataset_name!r} != directory name "
            f"{dataset_dir.name!r}"
        )
    if not (dataset_dir / "processed" / "lmdb").is_dir():
        raise PackagingRefused(f"no processed/lmdb under {dataset_dir}")
    return manifest


def refuse_if_stale(manifest: BuildManifest, dataset_dir: Path) -> None:
    """Refuse a manifest whose closure fingerprints drifted from the local schema."""
    result = check_manifest(
        manifest, load_default_surface(), str(dataset_dir / "preprocess")
    )
    if result.is_stale:
        symbols = ", ".join(sorted(d.symbol for d in result.drift))
        raise PackagingRefused(
            f"{manifest.dataset_name} is STALE against the local schema (changed: "
            f"{symbols}); rebuild before packaging"
        )


def _resolve_interned(obj: Any, interned: dict[str, Any]) -> Any:
    """``torchcell.data.experiment_dataset.resolve_interned``, without importing torch."""
    if isinstance(obj, dict):
        ref = obj.get("$ref")
        if ref is not None:
            return interned[ref]
        return {k: _resolve_interned(v, interned) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_interned(v, interned) for v in obj]
    return obj


def experiment_ids(processed_dir: Path) -> list[str]:
    """The KG node id of every record: ``sha256(json.dumps(experiment.model_dump()))``.

    Records store ``experiment.model_dump()`` verbatim (constant sub-objects interned
    by ``$ref`` and spliced back here), so hashing the resolved dict reproduces
    ``CellAdapter._experiment_node`` without re-validating each record.
    """
    interned: dict[str, Any] = {}
    interned_dir = processed_dir / "interned"
    if interned_dir.is_dir():
        ienv = lmdb.open(str(interned_dir), readonly=True, lock=False, readahead=False)
        with ienv.begin() as itxn:
            for key, value in itxn.cursor():
                interned[key.decode()] = pickle.loads(value)
        ienv.close()
    env = lmdb.open(
        str(processed_dir / "lmdb"), readonly=True, lock=False, readahead=False
    )
    ids: list[str] = []
    with env.begin() as txn:
        for _key, value in txn.cursor():
            record = _resolve_interned(pickle.loads(value), interned)
            payload = json.dumps(record["experiment"]).encode("utf-8")
            ids.append(hashlib.sha256(payload).hexdigest())
    env.close()
    return ids


def archive_members(dataset_dir: Path) -> list[Path]:
    """Files under ``preprocess/`` and ``processed/`` in sorted order, minus lock files."""
    members: list[Path] = []
    for sub in PACKAGED_DIRS:
        for path in sorted((dataset_dir / sub).rglob("*")):
            if path.is_file() and path.name not in EXCLUDED_BASENAMES:
                members.append(path)
    return members


def write_archive(dataset_dir: Path, out: Path, mtime: int) -> None:
    """Deterministic ``tar.xz`` of :func:`archive_members` relative to ``dataset_dir``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:xz", preset=XZ_PRESET, format=tarfile.GNU_FORMAT) as tar:
        for path in archive_members(dataset_dir):
            info = tar.gettarinfo(str(path), arcname=str(path.relative_to(dataset_dir)))
            info.mtime = mtime
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with path.open("rb") as handle:
                tar.addfile(info, handle)


def sha256_of_file(path: Path) -> str:
    """Streaming sha256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def package_dataset(
    dataset_dir: Path,
    store: Path,
    *,
    kg_release: str | None = None,
    kg_version: str | None = None,
    status: ArtifactStatus = "supported",
    compute_content_hash: bool = True,
    packaged_at: str | None = None,
) -> DatasetArtifact:
    """Package ``dataset_dir`` into ``store`` and record it in the store's index."""
    dataset_dir = dataset_dir.resolve()
    manifest = load_manifest(dataset_dir)
    refuse_if_stale(manifest, dataset_dir)
    processed_dir = dataset_dir / "processed"

    env = lmdb.open(
        str(processed_dir / "lmdb"), readonly=True, lock=False, readahead=False
    )
    with env.begin() as txn:
        n_experiments = int(txn.stat()["entries"])
    env.close()
    content_hash = (
        content_sha256(experiment_ids(processed_dir)) if compute_content_hash else None
    )

    mtime = int(datetime.fromisoformat(manifest.built_at).timestamp())
    tmp = store / manifest.dataset_name / f".{manifest.dataset_name}.partial.tar.xz"
    write_archive(dataset_dir, tmp, mtime)
    archive_sha256 = sha256_of_file(tmp)
    name = archive_name(manifest.dataset_name, __version__, archive_sha256)
    final = tmp.with_name(name)
    os.replace(tmp, final)

    artifact = DatasetArtifact(
        slug=manifest.dataset_name,
        dataset_class=manifest.loader_class,
        torchcell_version=__version__,
        torchcell_commit=manifest.torchcell_commit,
        kg_release=kg_release,
        kg_version=kg_version,
        content_sha256=content_hash,
        n_experiments=n_experiments,
        archive=name,
        archive_sha256=archive_sha256,
        archive_bytes=final.stat().st_size,
        built_at=manifest.built_at,
        packaged_at=packaged_at or _utc_now(),
        status=status,
        build_manifest=manifest,
    )
    index_path = store / INDEX_FILENAME
    index = (
        ArtifactIndex.load(index_path)
        if index_path.is_file()
        else ArtifactIndex(generated_at=artifact.packaged_at)
    )
    index = index.upsert(artifact, generated_at=artifact.packaged_at)
    index.save(index_path)
    (store / SHA256SUMS_FILENAME).write_text(index.sha256sums(), encoding="utf-8")
    return artifact


def main(argv: list[str] | None = None) -> int:
    """CLI entry: package one dataset directory; exit 1 on a refusal."""
    parser = argparse.ArgumentParser(
        prog="python scripts/package_dataset_lmdb.py", description=__doc__
    )
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--store", required=True, type=Path, help="TC_DATA_ROOT")
    parser.add_argument("--kg-release", default=None)
    parser.add_argument("--kg-version", default=None)
    parser.add_argument(
        "--status", default="supported", choices=list(get_args(ArtifactStatus))
    )
    parser.add_argument(
        "--no-content-hash",
        action="store_true",
        help="skip the per-record experiment-id pass (content_sha256 becomes null)",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    try:
        artifact = package_dataset(
            args.dataset_dir,
            args.store,
            kg_release=args.kg_release,
            kg_version=args.kg_version,
            status=args.status,
            compute_content_hash=not args.no_content_hash,
        )
    except PackagingRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started
    print(
        f"packaged {artifact.slug} ({artifact.n_experiments} records) -> "
        f"{args.store / artifact.rel_path}\n"
        f"  sha256 {artifact.archive_sha256}\n"
        f"  bytes  {artifact.archive_bytes}\n"
        f"  content_sha256 {artifact.content_sha256}\n"
        f"  {elapsed:.1f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
