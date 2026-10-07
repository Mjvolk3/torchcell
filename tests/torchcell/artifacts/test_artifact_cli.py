# tests/torchcell/artifacts/test_artifact_cli.py
# [[tests.torchcell.artifacts.test_artifact_cli]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/artifacts/test_artifact_cli.py
"""``python -m torchcell.artifacts``: resolve, check and deposit against tmp tiers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.torchcell.artifacts._fakes import sha, write_tier
from torchcell.artifacts.__main__ import main
from torchcell.artifacts.resolve import ArtifactUnresolvableError, ResolvedArtifact
from torchcell.literature.manifest import Manifest, ProcessingRecord

DATA = b"gene\tvalue\nYAL001C\t1\n"
URI = "tc://raw/k/data/f.tsv"


@pytest.fixture(autouse=True)
def _no_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI builds its remote from the environment; make that empty."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("TC_DATA_URL", raising=False)


def test_resolve_prints_the_resolved_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_tier(tmp_path, "raw", "k", {"data/f.tsv": DATA})
    argv = ["--data-root", str(tmp_path), "resolve", URI, "--sha256", sha(DATA)]
    assert main(argv) == 0
    printed = ResolvedArtifact.model_validate_json(capsys.readouterr().out)
    assert printed == ResolvedArtifact(
        path=tmp_path / "torchcell-raw" / "k" / "data" / "f.tsv",
        source="local",
        verified=True,
    )
    assert main([*argv, "--no-materialize"]) == 0
    assert (
        ResolvedArtifact.model_validate_json(capsys.readouterr().out).verified is False
    )


def test_resolve_of_an_unresolvable_ref_raises(tmp_path: Path) -> None:
    with pytest.raises(ArtifactUnresolvableError, match="tc://raw/k/data/f.tsv"):
        main(["--data-root", str(tmp_path), "resolve", URI, "--sha256", sha(DATA)])


def test_check_exits_zero_or_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--data-root", str(tmp_path), "check", URI, "--sha256", sha(DATA)]
    assert main(argv) == 1
    assert capsys.readouterr().out == "unresolvable\n"
    write_tier(tmp_path, "raw", "k", {"data/f.tsv": DATA})
    assert main(argv) == 0
    assert capsys.readouterr().out == "resolves\n"


def test_deposit_writes_the_manifest_with_a_processing_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "f.tsv").write_bytes(DATA)
    record = ProcessingRecord(
        processor="scripts.make.main", tool="python", version="3.13"
    )
    record_file = tmp_path / "processing.json"
    record_file.write_text(record.model_dump_json())
    root = tmp_path / "root"
    argv = ["--data-root", str(root), "deposit", str(staging), "--key", "set1"]
    assert main([*argv, "--processing", str(record_file)]) == 0
    dest = root / "torchcell-objects" / "set1" / "manifest.json"
    assert capsys.readouterr().out == f"1 files -> {dest}\n"
    manifest = Manifest.model_validate_json(dest.read_text())
    assert [(r.path, r.processing) for r in manifest.files] == [("f.tsv", record)]
    (staging / "f.tsv").write_bytes(DATA + b"x")
    assert main([*argv, "--allow-change"]) == 0
    capsys.readouterr()
    changed = Manifest.model_validate_json(dest.read_text()).files[0]
    assert changed.source == f"supersedes sha256:{sha(DATA)}"
    assert changed.processing is None
