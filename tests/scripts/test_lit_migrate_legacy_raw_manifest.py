# tests/scripts/test_lit_migrate_legacy_raw_manifest.py
"""``scripts/lit_migrate_legacy_raw_manifest.py`` on a hermetic legacy deposit (#923)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from torchcell.literature.manifest import Manifest

REPO = Path(__file__).resolve().parents[2]
COMMIT = "f95f7d103c4a9990446ad6ff7b1a883a0b11db07"
URL = f"https://github.com/owner/repo/raw/{COMMIT}/a.csv"


def _load() -> ModuleType:
    path = REPO / "scripts" / "lit_migrate_legacy_raw_manifest.py"
    spec = importlib.util.spec_from_file_location("lit_migrate_legacy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["lit_migrate_legacy"] = module
    spec.loader.exec_module(module)
    return module


mig = _load()


def _legacy(tmp_path: Path, payload: bytes = b"a,b\n") -> tuple[Path, Path]:
    key_dir = tmp_path / "torchcell-raw" / "zzKey2020"
    (key_dir / "data").mkdir(parents=True)
    (key_dir / "data" / "a.csv").write_bytes(payload)
    record = {
        "version": 1,
        "citation_key": "zzKey2020",
        "doi": "10.1000/zz",
        "title": "T",
        "data_availability_note": "GitHub copy",
        "files": [
            {
                "path": "data/a.csv",
                "role": "raw_data",
                "description": "a table",
                "bytes": 4,
                "sha256": hashlib.sha256(b"a,b\n").hexdigest(),
                "source": URL,
                "retrieval": {
                    "method": "direct_url",
                    "source_url": URL,
                    "retrieval_command": "gh api x | base64 -d > a.csv",
                    "pinned_commit": COMMIT,
                    "retrieved_at": "2026-09-14",
                },
                "processing": None,
            }
        ],
    }
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(record))
    return key_dir, legacy


def test_migrate_rederives_files_and_keeps_true_retrieval(tmp_path: Path) -> None:
    """sha256/bytes/role from disk; retrieval re-expressed as the direct_url retriever."""
    key_dir, legacy_path = _legacy(tmp_path)
    assert mig.main(["--key-dir", str(key_dir), "--legacy", str(legacy_path)]) == 0
    manifest = Manifest.model_validate_json((key_dir / "manifest.json").read_text())
    assert (manifest.citation_key, manifest.doi, manifest.title) == (
        "zzKey2020",
        "10.1000/zz",
        "T",
    )
    assert manifest.si_data_sources == [f"https://github.com/owner/repo/tree/{COMMIT}"]
    assert manifest.provenance_complete is True
    (record,) = manifest.files
    digest = hashlib.sha256(b"a,b\n").hexdigest()
    assert (record.path, record.role, record.bytes, record.sha256, record.source) == (
        "data/a.csv",
        "raw_data",
        4,
        digest,
        URL,
    )
    assert record.retrieval is not None
    assert record.retrieval.model_dump() == {
        "method": "direct_url",
        "source_url": URL,
        "retriever": "torchcell.literature.retrieve.direct_url",
        "params": {"url": URL},
        "sha256": digest,
        "retrieved_at": "2026-09-14",
        "last_check": None,
    }


def test_migrate_records_a_matching_source_check(tmp_path: Path) -> None:
    key_dir, legacy_path = _legacy(tmp_path)
    legacy = mig.LegacyManifest.model_validate_json(legacy_path.read_text())
    calls: list[str] = []

    def fake(url: str) -> bytes:
        calls.append(url)
        return b"a,b\n"

    manifest = mig.migrate(
        key_dir,
        legacy,
        check_now="2026-10-11T00:00:00",
        registry={mig.DIRECT_URL_RETRIEVER: fake},
    )
    assert calls == [URL]
    check = manifest.files[0].retrieval.last_check
    assert check is not None
    assert (check.checked_at, check.matches) == ("2026-10-11T00:00:00", True)


def test_migrate_refuses_a_drifted_source(tmp_path: Path) -> None:
    key_dir, legacy_path = _legacy(tmp_path)
    legacy = mig.LegacyManifest.model_validate_json(legacy_path.read_text())
    with pytest.raises(ValueError, match="re-running .* produced"):
        mig.migrate(
            key_dir,
            legacy,
            check_now="t",
            registry={mig.DIRECT_URL_RETRIEVER: lambda url: b"changed"},
        )


def test_migrate_refuses_bytes_that_differ_from_the_legacy_pin(tmp_path: Path) -> None:
    key_dir, legacy_path = _legacy(tmp_path, payload=b"other\n")
    legacy = mig.LegacyManifest.model_validate_json(legacy_path.read_text())
    with pytest.raises(ValueError, match="!= legacy pin"):
        mig.migrate(key_dir, legacy)


def test_migrate_refuses_an_unlisted_file_and_an_unpinned_url(tmp_path: Path) -> None:
    key_dir, legacy_path = _legacy(tmp_path)
    legacy = mig.LegacyManifest.model_validate_json(legacy_path.read_text())
    (key_dir / "data" / "extra.csv").write_bytes(b"x")
    with pytest.raises(ValueError, match="files on disk"):
        mig.migrate(key_dir, legacy)
    legacy.files[0].retrieval.pinned_commit = "0" * 40
    with pytest.raises(ValueError, match="does not pin commit"):
        mig.migrate(key_dir, legacy)


def test_main_refuses_to_overwrite_a_manifest(tmp_path: Path) -> None:
    key_dir, legacy_path = _legacy(tmp_path)
    (key_dir / "manifest.json").write_text("{}")
    with pytest.raises(SystemExit):
        mig.main(["--key-dir", str(key_dir), "--legacy", str(legacy_path)])
    assert (key_dir / "manifest.json").read_text() == "{}"


def test_legacy_model_refuses_an_unforeseen_field(tmp_path: Path) -> None:
    _, legacy_path = _legacy(tmp_path)
    record = json.loads(legacy_path.read_text())
    record["files"][0]["retrieval"]["extra"] = "x"
    with pytest.raises(ValueError, match="extra"):
        mig.LegacyManifest.model_validate(record)
