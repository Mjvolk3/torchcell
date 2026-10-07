# torchcell/benchmark/storage.py
# [[torchcell.benchmark.storage]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/benchmark/storage.py
# Test file: tests/torchcell/benchmark/test_storage.py

"""The archive kept for every scored submission.

After grading, the uploaded predictions, the metadata and the result are written as one
deflate-compressed zip at ``<root>/<dataset_slug>/<YYYY>/<MM>/<submission_id>.zip`` and
its sha256 is recorded on the submission row, so any score on the board can be
recomputed from the bytes that produced it. The zip is byte-deterministic for given
inputs (members sorted by name, member timestamps set to the submission time), and it
is written to a temporary name and renamed, so a crash never leaves a partial archive
under the final name. A rejected upload is not archived; only its sha256 and its
rejection reasons are stored, so junk uploads cannot fill the disk.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import zipfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

_SAFE_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_SAFE_MEMBER = re.compile(r"^[A-Za-z0-9_.\-]+$")


class ArchiveRecord(BaseModel):
    """Where an archive was written, relative to the submissions root, and its hash."""

    model_config = ConfigDict(frozen=True)

    relative_path: str
    sha256: str
    n_bytes: int


def build_archive(files: Mapping[str, bytes], timestamp: datetime) -> bytes:
    """The bytes of a deterministic zip holding ``files`` (name -> content)."""
    date_time = (
        timestamp.year,
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        timestamp.minute,
        timestamp.second,
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            if not _SAFE_MEMBER.fullmatch(name):
                raise ValueError(f"unsafe archive member name: {name!r}")
            info = zipfile.ZipInfo(name, date_time=date_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o640 << 16
            archive.writestr(info, files[name])
    return buffer.getvalue()


def archive_submission(
    root: Path,
    dataset_slug: str,
    submission_id: str,
    submitted_at: datetime,
    files: Mapping[str, bytes],
) -> ArchiveRecord:
    """Write the submission's zip under ``root`` and return its path and sha256."""
    if not _SAFE_NAME.fullmatch(dataset_slug) or not _SAFE_NAME.fullmatch(
        submission_id
    ):
        raise ValueError("dataset slug and submission id must be lowercase slugs")
    relative = (
        Path(dataset_slug)
        / f"{submitted_at.year:04d}"
        / f"{submitted_at.month:02d}"
        / f"{submission_id}.zip"
    )
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    data = build_archive(files, submitted_at)
    temporary = target.with_suffix(".zip.part")
    temporary.write_bytes(data)
    os.replace(temporary, target)
    return ArchiveRecord(
        relative_path=relative.as_posix(),
        sha256=hashlib.sha256(data).hexdigest(),
        n_bytes=len(data),
    )
