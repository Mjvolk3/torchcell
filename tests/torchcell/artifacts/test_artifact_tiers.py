# tests/torchcell/artifacts/test_artifact_tiers.py
# [[tests.torchcell.artifacts.test_artifact_tiers]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/artifacts/test_artifact_tiers.py
"""Tier roots under ``DATA_ROOT`` and manifest loading, against ``tmp_path`` tiers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.torchcell.artifacts._fakes import (
    literature_manifest,
    ref_for,
    sha,
    write_genomes,
    write_tier,
)
from torchcell.artifacts.tiers import (
    TierManifestKeyError,
    key_dir,
    load_manifest,
    load_tier_manifest,
    load_tier_records,
    manifest_path,
    manifest_record,
    resolve_data_root,
)
from torchcell.literature.manifest import Manifest, write_manifest
from torchcell.sequence.genome.registry import GenomeIntegrityError, GenomeManifest

DATA = b"gene\tfitness\nYAL001C\t0.9\n"


def test_resolve_data_root_reads_the_environment_after_load_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: calls.append(1))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert resolve_data_root(None) == tmp_path
    assert calls == [1]
    assert resolve_data_root("/x/y") == Path("/x/y")
    assert calls == [1]


def test_resolve_data_root_without_data_root_has_no_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("DATA_ROOT", raising=False)
    with pytest.raises(KeyError) as err:
        resolve_data_root(None)
    assert err.value.args == ("DATA_ROOT",)


def test_key_dirs_of_the_four_tiers(tmp_path: Path) -> None:
    assert key_dir("raw", "k", tmp_path) == tmp_path / "torchcell-raw" / "k"
    assert key_dir("library", "k", tmp_path) == tmp_path / "torchcell-library" / "k"
    assert key_dir("objects", "k", tmp_path) == tmp_path / "torchcell-objects" / "k"
    assert key_dir("genomes", "s", tmp_path) == tmp_path / "torchcell-genomes" / "s"
    assert (
        manifest_path("objects", "k", tmp_path)
        == tmp_path / "torchcell-objects" / "k" / "manifest.json"
    )


def test_literature_tier_manifest_loads_its_records(tmp_path: Path) -> None:
    written = write_tier(tmp_path, "raw", "kemmeren2014", {"data/expr.tsv": DATA})
    loaded = load_tier_manifest("raw", "kemmeren2014", tmp_path)
    assert isinstance(loaded, Manifest)
    assert loaded == written
    records = load_tier_records("raw", "kemmeren2014", tmp_path)
    assert [(r.path, r.sha256, r.bytes) for r in records] == [
        ("data/expr.tsv", sha(DATA), len(DATA))
    ]
    assert load_manifest("raw", "kemmeren2014", tmp_path) == written


def test_absent_manifest_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no objects manifest for key 'k'"):
        load_tier_manifest("objects", "k", tmp_path)


def test_a_manifest_naming_another_key_raises(tmp_path: Path) -> None:
    directory = tmp_path / "torchcell-objects" / "mine"
    directory.mkdir(parents=True)
    write_manifest(directory, literature_manifest("theirs", {"a.npy": DATA}))
    with pytest.raises(
        TierManifestKeyError, match="names citation_key 'theirs', not 'mine'"
    ):
        load_tier_manifest("objects", "mine", tmp_path)


def test_genomes_tier_reads_the_registry_manifest(tmp_path: Path) -> None:
    write_genomes(tmp_path, "set_v1", {"genes.tar.gz": DATA})
    loaded = load_tier_manifest("genomes", "set_v1", tmp_path)
    assert isinstance(loaded, GenomeManifest)
    assert [(r.path, r.sha256) for r in loaded.files] == [("genes.tar.gz", sha(DATA))]


def test_genomes_tier_without_the_set_raises_the_registry_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="seed it with: rsync -a"):
        load_tier_manifest("genomes", "absent_set", tmp_path)


def test_genomes_manifest_naming_another_set_raises(tmp_path: Path) -> None:
    write_genomes(tmp_path, "set_v1", {"genes.tar.gz": DATA})
    moved = tmp_path / "torchcell-genomes" / "set_v2"
    (tmp_path / "torchcell-genomes" / "set_v1").rename(moved)
    with pytest.raises(GenomeIntegrityError, match="names assembly set 'set_v1'"):
        load_tier_manifest("genomes", "set_v2", tmp_path)


def test_manifest_record_locates_or_returns_none(tmp_path: Path) -> None:
    write_tier(tmp_path, "library", "smith2020", {"paper.md": DATA})
    hit = manifest_record(ref_for("library", "smith2020", "paper.md", DATA), tmp_path)
    assert hit is not None
    assert (hit.path, hit.sha256, hit.role) == ("paper.md", sha(DATA), "paper_ocr")
    unlisted = ref_for("library", "smith2020", "paper.pdf", DATA)
    assert manifest_record(unlisted, tmp_path) is None
    no_manifest = ref_for("library", "absent2020", "paper.md", DATA)
    assert manifest_record(no_manifest, tmp_path) is None


def test_manifest_record_does_not_judge_the_sha256(tmp_path: Path) -> None:
    """Locating is separate from verifying: a record with another sha256 is returned."""
    write_tier(tmp_path, "raw", "k", {"f.tsv": DATA})
    other = ref_for("raw", "k", "f.tsv", b"different bytes")
    record = manifest_record(other, tmp_path)
    assert record is not None
    assert record.sha256 == sha(DATA)
