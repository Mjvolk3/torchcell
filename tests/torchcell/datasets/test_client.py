# tests/torchcell/datasets/test_client.py
# [[tests.torchcell.datasets.test_client]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_client.py
"""``torchcell.datasets.client`` against the tc-data app through the ASGI test client."""

import hashlib
import io
import tarfile
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from torchcell.datasets.artifact import ArtifactIndex, DatasetArtifact, archive_name
from torchcell.datasets.client import (
    ArtifactIntegrityError,
    DatasetClient,
    unpack_artifact,
)
from torchcell.datasets.server import DataKeys, DataServerConfig, create_app

KEY = "client-key"
PAYLOAD = bytes(range(256)) * 20
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()
ARCHIVE = archive_name("smf_fake", "1.2.1", PAYLOAD_SHA)


def _artifact(sha: str = PAYLOAD_SHA, version: str = "1.2.1") -> DatasetArtifact:
    return DatasetArtifact(
        slug="smf_fake",
        dataset_class="SmfFakeDataset",
        torchcell_version=version,
        n_experiments=3,
        archive=ARCHIVE,
        archive_sha256=sha,
        archive_bytes=len(PAYLOAD),
        built_at="2026-09-14T02:25:00+00:00",
        packaged_at="2026-09-29T10:00:00+00:00",
    )


def _store(root: Path, artifact: DatasetArtifact) -> Path:
    store = root / "store"
    (store / "smf_fake").mkdir(parents=True)
    (store / "smf_fake" / ARCHIVE).write_bytes(PAYLOAD)
    ArtifactIndex(generated_at="t0", artifacts=[artifact]).save(store / "index.json")
    return store


def _client(root: Path, artifact: DatasetArtifact) -> DatasetClient:
    raw = root / "raw"
    raw.mkdir(exist_ok=True)
    config = DataServerConfig(
        store_root=_store(root, artifact),
        raw_root=raw,
        keys=DataKeys.from_pairs(f"t:{KEY}"),
    )
    return DatasetClient("http://testserver/", KEY, http=TestClient(create_app(config)))


def test_index_and_artifacts_round_trip_through_http(tmp_path: Path) -> None:
    client = _client(tmp_path, _artifact())
    assert client.url == "http://testserver"
    assert client.index() == ArtifactIndex.load(tmp_path / "store" / "index.json")
    assert client.artifacts("smf_fake") == [_artifact()]


def test_select_applies_the_major_minor_rule(tmp_path: Path) -> None:
    client = _client(tmp_path, _artifact())
    assert client.select("smf_fake", "1.2.9") == _artifact()
    assert client.select("smf_fake", "1.3.0") is None
    assert client.select("other", "1.2.1") is None


def test_download_streams_verifies_and_renames_into_place(tmp_path: Path) -> None:
    client = _client(tmp_path, _artifact())
    dest = tmp_path / "out" / ARCHIVE
    assert client.download(_artifact(), dest) == dest
    assert dest.read_bytes() == PAYLOAD
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_resumes_a_partial_file_with_a_range_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, _artifact())
    dest = tmp_path / "out" / ARCHIVE
    dest.parent.mkdir()
    dest.with_name(dest.name + ".part").write_bytes(PAYLOAD[:1000])
    seen: list[dict[str, str]] = []
    original = client._http.stream

    def spy(method: str, url: str, **kwargs: Any) -> Any:
        seen.append(dict(kwargs["headers"]))
        return original(method, url, **kwargs)

    monkeypatch.setattr(client._http, "stream", spy)
    client.download(_artifact(), dest)
    assert [h["Range"] for h in seen] == ["bytes=1000-"]
    assert dest.read_bytes() == PAYLOAD


def test_download_rejects_a_server_hash_that_disagrees_with_the_row(
    tmp_path: Path,
) -> None:
    served = _artifact()
    client = _client(tmp_path, served)
    stale_row = _artifact(sha="0" * 64)
    with pytest.raises(ArtifactIntegrityError, match="server X-Artifact-SHA256"):
        client.download(stale_row, tmp_path / "out" / ARCHIVE)
    assert not (tmp_path / "out" / (ARCHIVE + ".part")).exists()


def test_download_removes_a_corrupt_partial_and_raises(tmp_path: Path) -> None:
    client = _client(tmp_path, _artifact())
    dest = tmp_path / "out" / ARCHIVE
    dest.parent.mkdir()
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(b"\xff" * 1000)
    with pytest.raises(ArtifactIntegrityError, match="partial file removed"):
        client.download(_artifact(), dest)
    assert not part.exists()
    assert not dest.exists()


def test_download_without_verify_keeps_the_bytes_as_served(tmp_path: Path) -> None:
    client = _client(tmp_path, _artifact())
    dest = tmp_path / "out" / ARCHIVE
    dest.parent.mkdir()
    dest.with_name(dest.name + ".part").write_bytes(b"\xff" * 1000)
    client.download(_artifact(), dest, verify=False)
    assert dest.read_bytes() == b"\xff" * 1000 + PAYLOAD[1000:]


def test_download_raises_on_an_http_error(tmp_path: Path) -> None:
    client = _client(tmp_path, _artifact())
    with pytest.raises(httpx.HTTPStatusError, match="expected HTTP 200, got 404"):
        client.download(
            _artifact().model_copy(update={"archive": "missing.tar.xz"}),
            tmp_path / "out" / "missing.tar.xz",
        )


def test_from_env_reads_url_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TC_DATA_URL", "http://radiant:8724/")
    monkeypatch.setenv("TC_DATA_API_KEY", "k")
    client = DatasetClient.from_env(http=httpx.Client())
    assert client.url == "http://radiant:8724"
    assert client._headers == {"X-API-Key": "k"}
    monkeypatch.delenv("TC_DATA_API_KEY")
    with pytest.raises(KeyError, match="TC_DATA_API_KEY"):
        DatasetClient.from_env()


def _tar_xz(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:xz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_unpack_artifact_extracts_members_into_the_root(tmp_path: Path) -> None:
    archive = tmp_path / "a.tar.xz"
    archive.write_bytes(
        _tar_xz(
            {
                "preprocess/build_manifest.json": b"{}",
                "processed/lmdb/data.mdb": b"\x00" * 16,
            }
        )
    )
    root = tmp_path / "ds"
    names = unpack_artifact(archive, root)
    assert names == ["preprocess/build_manifest.json", "processed/lmdb/data.mdb"]
    assert (root / "preprocess" / "build_manifest.json").read_bytes() == b"{}"
    assert (root / "processed" / "lmdb" / "data.mdb").read_bytes() == b"\x00" * 16


def test_unpack_artifact_refuses_a_member_outside_the_root(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar.xz"
    archive.write_bytes(_tar_xz({"../escape.txt": b"x"}))
    with pytest.raises(tarfile.OutsideDestinationError):
        unpack_artifact(archive, tmp_path / "ds")
    assert not (tmp_path / "escape.txt").exists()
