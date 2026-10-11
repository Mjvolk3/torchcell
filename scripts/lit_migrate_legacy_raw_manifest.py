#!/usr/bin/env python
# scripts/lit_migrate_legacy_raw_manifest.py
# [[scripts.lit_migrate_legacy_raw_manifest]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/lit_migrate_legacy_raw_manifest.py

r"""Regenerate a hand-written raw-mirror manifest into the ``Manifest`` schema (#923).

``torchcell-raw/zhangCombiningMechanisticMachine2020/manifest.json`` was written by hand
on 2026-09-14 in a shape ``RetrievalRecord`` rejects: each ``retrieval`` carries free
``retrieval_command`` and ``pinned_commit`` fields and lacks ``retriever`` and ``sha256``.
This script reads that legacy record through :class:`LegacyManifest` (``extra="forbid"``,
so an unforeseen field refuses instead of being dropped) and rebuilds the manifest by the
subsystem's own path, :func:`torchcell.literature.manifest.build_manifest`, which
re-derives every file's ``sha256``, ``bytes`` and ``role`` from the bytes on disk.

Kept from the legacy record, because it is still true: ``doi``, ``title``, each file's
``source`` and ``source_url``, ``retrieved_at``, and the ``direct_url`` method. The
pinned commit is already in each ``source_url`` (checked, not assumed) and is also kept
as the repository's ``tree/<commit>`` URL in ``si_data_sources``. The retrieval becomes
the versioned retriever ``torchcell.literature.retrieve.direct_url`` with
``params={"url": source_url}``; ``--check-source`` re-runs it and records the result as
``last_check``, refusing a mismatch. Dropped by the schema: the ``gh api`` command
string, per-file ``description`` and ``data_availability_note`` (the legacy file is
moved to the graveyard, not deleted, and the note quotes what it said).

A measured ``sha256`` or ``bytes`` that differs from the legacy record refuses: the
legacy hash is the pin the bytes were deposited under.

Usage, from the repo root::

    python scripts/lit_migrate_legacy_raw_manifest.py \
        --key-dir $DATA_ROOT/torchcell-raw/zhangCombiningMechanisticMachine2020 \
        --legacy <graveyard copy of the hand-written manifest.json> --check-source
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from torchcell.literature.manifest import (
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
    build_manifest,
    write_manifest,
)
from torchcell.literature.provenance import check_source

DIRECT_URL_RETRIEVER = "torchcell.literature.retrieve.direct_url"
_GITHUB_RAW = re.compile(
    r"https://github\.com/(?P<repo>[^/]+/[^/]+)/raw/(?P<commit>[0-9a-f]{40})/.+"
)


class LegacyRetrieval(BaseModel):
    """The hand-written retrieval shape of 2026-09-14."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["direct_url"]
    source_url: str
    retrieval_command: str
    pinned_commit: str
    retrieved_at: str


class LegacyFile(BaseModel):
    """One hand-written file entry (with its free ``description``)."""

    model_config = ConfigDict(extra="forbid")

    path: str
    role: str
    description: str
    bytes: int
    sha256: str
    source: str
    retrieval: LegacyRetrieval
    processing: None


class LegacyManifest(BaseModel):
    """The hand-written manifest, ``data_availability_note`` included."""

    model_config = ConfigDict(extra="forbid")

    version: int
    citation_key: str
    doi: str
    title: str
    data_availability_note: str
    files: list[LegacyFile]


def migrate(
    key_dir: Path,
    legacy: LegacyManifest,
    *,
    check_now: str | None = None,
    registry: dict[str, Callable[..., bytes]] | None = None,
) -> Manifest:
    """The schema-valid manifest for ``key_dir``, re-derived from its bytes.

    ``check_now`` (an ISO timestamp) re-runs each retriever and records ``last_check``;
    ``registry`` overrides the retriever registry (tests).
    """
    by_path = {f.path: f for f in legacy.files}
    repos: set[str] = set()
    for f in legacy.files:
        match = _GITHUB_RAW.fullmatch(f.retrieval.source_url)
        if match is None or match["commit"] != f.retrieval.pinned_commit:
            raise ValueError(
                f"{f.path}: source_url {f.retrieval.source_url} does not pin "
                f"commit {f.retrieval.pinned_commit}"
            )
        repos.add(f"https://github.com/{match['repo']}/tree/{match['commit']}")
    rebuilt = build_manifest(
        key_dir,
        citation_key=legacy.citation_key,
        doi=legacy.doi,
        title=legacy.title,
        sources={f.path: f.source for f in legacy.files},
        si_data_sources=sorted(repos),
    )
    on_disk = [f.path for f in rebuilt.files]
    if sorted(on_disk) != sorted(by_path):
        raise ValueError(f"files on disk {on_disk} != legacy files {sorted(by_path)}")
    for record in rebuilt.files:
        old = by_path[record.path]
        if (record.sha256, record.bytes) != (old.sha256, old.bytes):
            raise ValueError(
                f"{record.path}: measured {record.sha256} ({record.bytes} B) != "
                f"legacy pin {old.sha256} ({old.bytes} B)"
            )
        retrieval = RetrievalRecord(
            method=RetrievalMethod.direct_url,
            source_url=old.retrieval.source_url,
            retriever=DIRECT_URL_RETRIEVER,
            params={"url": old.retrieval.source_url},
            sha256=record.sha256,
            retrieved_at=old.retrieval.retrieved_at,
        )
        if check_now is not None:
            check = check_source(retrieval, now=check_now, registry=registry)
            if not check.matches:
                raise ValueError(
                    f"{record.path}: re-running {DIRECT_URL_RETRIEVER} produced "
                    f"{check.produced_sha256}, not {record.sha256}"
                )
            retrieval.last_check = check
        record.retrieval = retrieval
    return rebuilt


def main(argv: list[str] | None = None) -> int:
    """Validate the legacy record, then write the regenerated manifest."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--key-dir", required=True, type=Path)
    parser.add_argument("--legacy", required=True, type=Path)
    parser.add_argument("--check-source", action="store_true")
    args = parser.parse_args(argv)
    legacy = LegacyManifest.model_validate_json(args.legacy.read_text())
    if (args.key_dir / "manifest.json").exists():
        parser.error(f"{args.key_dir}/manifest.json exists; move the legacy file first")
    now = datetime.now(UTC).isoformat() if args.check_source else None
    write_manifest(args.key_dir, migrate(args.key_dir, legacy, check_now=now))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
