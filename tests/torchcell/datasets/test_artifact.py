# tests/torchcell/datasets/test_artifact.py
# [[tests.torchcell.datasets.test_artifact]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_artifact.py
"""``torchcell.datasets.artifact``: artifact records, the sorted byte-stable index, selection."""

import hashlib
from pathlib import Path

import pytest

from torchcell.datasets.artifact import (
    ArtifactIndex,
    DatasetArtifact,
    archive_name,
    content_sha256,
    major_minor,
    parse_version,
)
from torchcell.knowledge_graphs.releases import content_sha256 as kg_content_sha256
from torchcell.provenance.build_manifest import BuildManifest

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _manifest() -> BuildManifest:
    return BuildManifest(
        dataset_name="smf_fake",
        loader_class="SmfFakeDataset",
        loader_module="tests.fake",
        surface_modules=["schema.py"],
        closure={"Environment": "0" * 64},
        built_at="2026-09-14T02:25:00+00:00",
        hostname="gilahyper",
        torchcell_commit="901d5ec31",
        torchcell_dirty=False,
    )


def _artifact(
    slug: str = "smf_fake",
    version: str = "1.2.1",
    sha: str = SHA_A,
    status: str = "supported",
    packaged_at: str = "2026-09-29T10:00:00+00:00",
    manifest: BuildManifest | None = None,
) -> DatasetArtifact:
    return DatasetArtifact(
        slug=slug,
        dataset_class="SmfFakeDataset",
        torchcell_version=version,
        torchcell_commit="901d5ec31",
        kg_release="2026.09.21-ab6d8c5d",
        kg_version="1.2",
        content_sha256=SHA_C,
        n_experiments=3,
        archive=archive_name(slug, version, sha),
        archive_sha256=sha,
        archive_bytes=1234,
        built_at="2026-09-14T02:25:00+00:00",
        packaged_at=packaged_at,
        status=status,  # type: ignore[arg-type]
        build_manifest=manifest,
    )


def test_parse_version_and_major_minor() -> None:
    assert parse_version("1.12.3") == (1, 12, 3)
    assert parse_version("1.2.1.dev4") == (1, 2, 1)
    assert major_minor("1.12.3") == "1.12"


def test_parse_version_rejects_a_two_part_version() -> None:
    with pytest.raises(ValueError, match="not a major.minor.patch version: '1.2'"):
        parse_version("1.2")


def test_content_sha256_matches_the_release_digest_and_is_order_free() -> None:
    ids = ["id-b", "id-a", "id-c"]
    expected = hashlib.sha256(b"id-a\nid-b\nid-c\n").hexdigest()
    assert content_sha256(ids) == expected
    assert content_sha256(reversed(ids)) == expected
    assert kg_content_sha256(ids) == expected


def test_archive_name_embeds_slug_version_and_sha_prefix() -> None:
    assert archive_name("smf_fake", "1.2.1", SHA_A) == "smf_fake-1.2.1-aaaaaaaa.tar.xz"


def test_artifact_round_trip_and_derived_fields() -> None:
    artifact = _artifact(manifest=_manifest())
    restored = DatasetArtifact.model_validate_json(artifact.model_dump_json())
    assert restored == artifact
    assert restored.build_manifest == _manifest()
    assert artifact.rel_path == "smf_fake/smf_fake-1.2.1-aaaaaaaa.tar.xz"
    assert artifact.sort_key == ("smf_fake", (1, 2, 1), SHA_A)


def test_artifact_status_is_a_closed_vocabulary() -> None:
    with pytest.raises(ValueError, match="status"):
        _artifact(status="retired")


def test_index_save_sorts_rows_and_reload_is_byte_stable(tmp_path: Path) -> None:
    rows = [
        _artifact(slug="zzz_last", version="1.2.0", sha=SHA_B),
        _artifact(slug="smf_fake", version="1.10.0", sha=SHA_C),
        _artifact(slug="smf_fake", version="1.2.1", sha=SHA_A),
    ]
    index = ArtifactIndex(generated_at="2026-09-29T10:00:00+00:00", artifacts=rows)
    path = index.save(tmp_path / "index.json")
    first = path.read_bytes()
    loaded = ArtifactIndex.load(path)
    assert [a.sort_key for a in loaded.artifacts] == [
        ("smf_fake", (1, 2, 1), SHA_A),
        ("smf_fake", (1, 10, 0), SHA_C),
        ("zzz_last", (1, 2, 0), SHA_B),
    ]
    assert loaded.schema_version == 1
    assert loaded.save(path).read_bytes() == first
    assert first.endswith(b"}\n")


def test_upsert_replaces_the_same_archive_and_appends_a_new_one() -> None:
    index = ArtifactIndex(generated_at="t0", artifacts=[_artifact(sha=SHA_A)])
    replaced = index.upsert(
        _artifact(sha=SHA_A, packaged_at="2026-09-30T00:00:00+00:00"), "t1"
    )
    assert replaced.generated_at == "t1"
    assert [a.packaged_at for a in replaced.artifacts] == ["2026-09-30T00:00:00+00:00"]
    appended = replaced.upsert(_artifact(sha=SHA_B), "t2")
    assert [a.archive_sha256 for a in appended.artifacts] == [SHA_A, SHA_B]
    assert index.artifacts[0].packaged_at == "2026-09-29T10:00:00+00:00"


def test_for_slug_returns_only_that_slug_in_order() -> None:
    index = ArtifactIndex(
        generated_at="t0",
        artifacts=[
            _artifact(slug="other", sha=SHA_B),
            _artifact(version="1.3.0", sha=SHA_C),
            _artifact(version="1.2.1", sha=SHA_A),
        ],
    )
    assert [a.torchcell_version for a in index.for_slug("smf_fake")] == [
        "1.2.1",
        "1.3.0",
    ]
    assert index.for_slug("missing") == []


def test_select_takes_the_newest_supported_row_on_the_installed_line() -> None:
    index = ArtifactIndex(
        generated_at="t0",
        artifacts=[
            _artifact(version="1.2.0", sha=SHA_A, packaged_at="2026-09-01T00:00:00Z"),
            _artifact(version="1.2.1", sha=SHA_B, packaged_at="2026-09-02T00:00:00Z"),
            _artifact(version="1.2.1", sha=SHA_C, packaged_at="2026-09-03T00:00:00Z"),
            _artifact(version="1.3.0", sha="d" * 64),
            _artifact(version="1.2.5", sha="e" * 64, status="deprecated"),
        ],
    )
    chosen = index.select("smf_fake", "1.2.9")
    assert chosen is not None
    assert (chosen.torchcell_version, chosen.archive_sha256) == ("1.2.1", SHA_C)
    on_1_3 = index.select("smf_fake", "1.3.0")
    assert on_1_3 is not None
    assert on_1_3.archive_sha256 == "d" * 64


def test_select_returns_none_off_line_or_deprecated_only() -> None:
    deprecated = ArtifactIndex(
        generated_at="t0", artifacts=[_artifact(status="deprecated")]
    )
    assert deprecated.select("smf_fake", "1.2.1") is None
    supported = ArtifactIndex(generated_at="t0", artifacts=[_artifact()])
    assert supported.select("smf_fake", "2.0.0") is None
    assert supported.select("other", "1.2.1") is None


def test_sha256sums_lines_are_sha256sum_compatible() -> None:
    index = ArtifactIndex(
        generated_at="t0",
        artifacts=[_artifact(slug="zzz", sha=SHA_B), _artifact(sha=SHA_A)],
    )
    assert index.sha256sums() == (
        f"{SHA_A}  smf_fake/smf_fake-1.2.1-aaaaaaaa.tar.xz\n"
        f"{SHA_B}  zzz/zzz-1.2.1-bbbbbbbb.tar.xz\n"
    )
