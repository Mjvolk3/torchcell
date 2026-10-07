# torchcell/artifacts/deposit.py
# [[torchcell.artifacts.deposit]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/artifacts/deposit.py
# Test file: tests/torchcell/artifacts/test_artifact_deposit.py

"""Deposit a directory of files into the ``objects`` (or ``raw``) tier with a manifest.

``deposit`` hashes every file under ``directory``, copies the files into
``<tier root>/<key>/`` (in place when ``directory`` already is that key directory), and
writes ``manifest.json`` in the literature ``Manifest`` shape. An existing manifest is
updated, never rebuilt: records for files the deposit does not touch are kept verbatim,
an unchanged file keeps its record, and a file whose sha256 changed is refused unless
``allow_change=True``, in which case its new record's ``source`` names the superseded
sha256. Depositing the same bytes twice therefore writes the same manifest bytes.

The genomes tier has its own deposit (``registry.deposit_assembly_set``) and the library
tier its own capture pipeline, so both are refused here.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from torchcell.artifacts.tiers import key_dir, load_manifest
from torchcell.literature.manifest import (
    MANIFEST_FILENAME,
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    ProcessingRecord,
    sha256_file,
    write_manifest,
)

#: Default role of a file in the objects tier when ``roles`` does not name it.
ROLE_OBJECT = "object"
#: The tiers ``deposit`` writes, and the default role of a file in each.
DepositTier = Literal["objects", "raw"]
DEPOSIT_TIERS: dict[str, str] = {"objects": ROLE_OBJECT, "raw": ROLE_RAW_DATA}


class DepositChangeError(RuntimeError):
    """A file the manifest already lists now has different bytes."""


def _scan(directory: Path) -> dict[str, Path]:
    """Every file under ``directory`` except ``manifest.json``, by relative path."""
    return {
        path.relative_to(directory).as_posix(): path
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and path.relative_to(directory).as_posix() != MANIFEST_FILENAME
    }


def deposit(
    directory: str | Path,
    *,
    tier: DepositTier = "objects",
    key: str,
    processing: ProcessingRecord | None = None,
    roles: Mapping[str, str] | None = None,
    data_root: str | Path | None = None,
    allow_change: bool = False,
    created_at: str | None = None,
) -> Manifest:
    """Hash, place and manifest every file of ``directory`` under ``<tier>/<key>/``.

    Args:
        directory: The files to deposit (relative paths are kept).
        tier: ``objects`` (default) or ``raw``.
        key: The key directory (a citation key or a named derived set).
        processing: Attached to every new or changed record (the producing script and
            its input sha256s).
        roles: Role per relative path; a path it does not name gets the tier default
            (``object`` or ``raw_data``). Naming a path not in ``directory`` raises.
        data_root: Defaults to ``DATA_ROOT`` after ``load_dotenv()``.
        allow_change: Permit replacing a listed file whose sha256 changed.
        created_at: For a NEW manifest only; defaults to now (UTC). An existing
            manifest keeps its own.

    Returns the manifest written. Raises ``DepositChangeError`` for a changed file
    without ``allow_change`` (nothing is copied or written in that case).
    """
    if tier not in DEPOSIT_TIERS:
        raise ValueError(
            f"deposit writes the tiers {sorted(DEPOSIT_TIERS)}, not {tier!r}; the "
            "genomes tier uses registry.deposit_assembly_set and the library tier "
            "its capture pipeline"
        )
    source_dir = Path(directory)
    files = _scan(source_dir)
    if not files:
        raise ValueError(f"{source_dir} holds no files to deposit")
    role_map = dict(roles or {})
    unknown = sorted(set(role_map) - set(files))
    if unknown:
        raise ValueError(f"roles name paths not in {source_dir}: {unknown}")
    dest_dir = key_dir(tier, key, data_root)
    in_place = dest_dir.resolve() == source_dir.resolve()
    existing = (
        load_manifest(tier, key, data_root)
        if (dest_dir / MANIFEST_FILENAME).is_file()
        else None
    )
    records = {rec.path: rec for rec in (existing.files if existing else [])}
    hashes = {rel: sha256_file(path) for rel, path in files.items()}
    changed = {
        rel: records[rel].sha256
        for rel, sha in hashes.items()
        if rel in records and records[rel].sha256 != sha
    }
    if changed and not allow_change:
        raise DepositChangeError(
            f"{tier}/{key}: listed files changed sha256 "
            + ", ".join(
                f"{rel} ({old} -> {hashes[rel]})" for rel, old in changed.items()
            )
            + "; pass allow_change=True to replace them"
        )
    for rel, path in files.items():
        target = dest_dir / rel
        if not in_place and (
            rel not in records or rel in changed or not target.is_file()
        ):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        if rel in records and rel not in changed:
            continue
        records[rel] = ArtifactRecord(
            path=rel,
            role=role_map.get(rel, DEPOSIT_TIERS[tier]),
            bytes=path.stat().st_size,
            sha256=hashes[rel],
            source=f"supersedes sha256:{changed[rel]}" if rel in changed else None,
            processing=processing,
        )
    ordered = [records[rel] for rel in sorted(records)]
    complete = all(
        rec.processing is not None or rec.retrieval is not None for rec in ordered
    )
    if existing is None:
        manifest = Manifest(
            citation_key=key,
            files=ordered,
            provenance_complete=complete,
            created_at=created_at or datetime.now(UTC).isoformat(),
        )
    else:
        manifest = existing.model_copy(
            update={"files": ordered, "provenance_complete": complete}
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(dest_dir, manifest)
    return manifest
