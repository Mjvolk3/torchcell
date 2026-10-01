# tests/torchcell/benchmark/test_storage.py
# [[tests.torchcell.benchmark.test_storage]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/benchmark/test_storage.py
"""``torchcell.benchmark.storage``: the per-submission zip.

The archive lands at ``<slug>/<YYYY>/<MM>/<id>.zip``, its recorded sha256 is the hash of
the bytes on disk, two builds of the same inputs are byte-identical, members are stored
sorted and deflated with the submission time, and no ``.part`` file is left behind.
Names that could leave the root (``..``, a slash, upper case) are refused.
"""

import hashlib
import io
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from torchcell.benchmark.storage import archive_submission, build_archive

WHEN = datetime(2026, 10, 1, 14, 30, 8, tzinfo=UTC)
FILES = {
    "result.json": b'{"status": "provisional"}',
    "predictions.csv": b"record_id,split,target,prediction\n"
    + b"v1,val,fitness,1.0\n" * 50,
    "metadata.json": b'{"method_name": "m"}',
}


def test_archive_path_hash_and_contents(tmp_path: Path) -> None:
    record = archive_submission(tmp_path, "toy-fitness", "abc123", WHEN, FILES)
    assert record.relative_path == "toy-fitness/2026/10/abc123.zip"
    data = (tmp_path / record.relative_path).read_bytes()
    assert record.sha256 == hashlib.sha256(data).hexdigest()
    assert record.n_bytes == len(data)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert archive.namelist() == ["metadata.json", "predictions.csv", "result.json"]
        assert archive.read("predictions.csv") == FILES["predictions.csv"]
        info = archive.getinfo("predictions.csv")
        assert info.compress_type == zipfile.ZIP_DEFLATED
        assert info.compress_size < info.file_size
        assert info.date_time == (2026, 10, 1, 14, 30, 8)
    assert [p.name for p in (tmp_path / "toy-fitness/2026/10").iterdir()] == [
        "abc123.zip"
    ]


def test_archive_bytes_are_deterministic() -> None:
    reordered = dict(reversed(list(FILES.items())))
    assert build_archive(FILES, WHEN) == build_archive(reordered, WHEN)


@pytest.mark.parametrize(
    ("slug", "submission_id"),
    [
        ("../etc", "abc"),
        ("toy/fitness", "abc"),
        ("toy-fitness", "../abc"),
        ("Toy", "abc"),
    ],
)
def test_unsafe_path_parts_are_refused(
    tmp_path: Path, slug: str, submission_id: str
) -> None:
    with pytest.raises(ValueError, match="must be lowercase slugs"):
        archive_submission(tmp_path, slug, submission_id, WHEN, FILES)
    assert list(tmp_path.iterdir()) == []


def test_unsafe_member_names_are_refused() -> None:
    with pytest.raises(ValueError, match="unsafe archive member name: '../x.csv'"):
        build_archive({"../x.csv": b""}, WHEN)
