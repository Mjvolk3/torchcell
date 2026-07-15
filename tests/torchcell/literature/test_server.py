"""Tests for torchcell.literature.server (keyed read-only endpoint)."""

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from torchcell.literature.backfill import backfill_mirror
from torchcell.literature.server import (
    LiteratureKeys,
    LiteratureServerConfig,
    create_app,
)

GOOD_KEY = "mac-secret-key-123"
BAD_KEY = "not-a-real-key"
HEADERS = {"X-API-Key": GOOD_KEY}


def _build_mirror(root: Path) -> None:
    paper = root / "fakePaperKey2020"
    (paper / "images").mkdir(parents=True)
    (paper / "paper.pdf").write_bytes(b"%PDF-1.5 fake pdf bytes")
    (paper / "paper.md").write_text("# Fake paper\n\nBody mentions glycolysis flux.")
    (paper / "images" / "fig1.jpg").write_bytes(b"\xff\xd8\xff fake jpeg")
    backfill_mirror(root, use_zotero=False)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    _build_mirror(tmp_path)
    keys = LiteratureKeys.from_pairs(f"mac:{GOOD_KEY},collab:another-key")
    config = LiteratureServerConfig(mirror_root=tmp_path, keys=keys, port=8899)
    return TestClient(create_app(config))


def test_health_needs_no_auth(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["n_keys"] == 1


def test_missing_key_rejected(client: TestClient) -> None:
    assert client.get("/keys").status_code == 401


def test_bad_key_rejected(client: TestClient) -> None:
    assert client.get("/keys", headers={"X-API-Key": BAD_KEY}).status_code == 401


def test_list_keys_reflects_mirror(client: TestClient) -> None:
    resp = client.get("/keys", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json()["citation_keys"] == ["fakePaperKey2020"]


def test_manifest_returned_as_model(client: TestClient) -> None:
    resp = client.get("/keys/fakePaperKey2020/manifest", headers=HEADERS)
    assert resp.status_code == 200
    manifest = resp.json()
    assert manifest["citation_key"] == "fakePaperKey2020"
    assert manifest["provenance_complete"] is False


def test_manifest_404_for_unknown_key(client: TestClient) -> None:
    resp = client.get("/keys/doesNotExist/manifest", headers=HEADERS)
    assert resp.status_code == 404


def test_artifact_download_matches_sha256(client: TestClient) -> None:
    # Get the expected sha256 from the manifest.
    files = client.get("/keys/fakePaperKey2020/files", headers=HEADERS).json()
    paper_md = next(f for f in files if f["path"] == "paper.md")

    resp = client.get("/keys/fakePaperKey2020/artifact/paper.md", headers=HEADERS)
    assert resp.status_code == 200
    downloaded_sha = hashlib.sha256(resp.content).hexdigest()
    assert downloaded_sha == paper_md["sha256"]
    assert resp.headers["X-Artifact-SHA256"] == paper_md["sha256"]


def test_path_traversal_blocked(client: TestClient) -> None:
    # Try to escape the citation-key dir up to the mirror root and beyond.
    resp = client.get(
        "/keys/fakePaperKey2020/artifact/../../etc/passwd", headers=HEADERS
    )
    assert resp.status_code in (400, 404)
    # Ensure no file leaked.
    assert b"root:" not in resp.content


def test_search_finds_paper_md_body(client: TestClient) -> None:
    resp = client.get("/search", params={"q": "glycolysis"}, headers=HEADERS)
    assert resp.status_code == 200
    hits = resp.json()["hits"]
    assert len(hits) == 1
    assert hits[0]["citation_key"] == "fakePaperKey2020"
    assert "paper.md" in hits[0]["where"]


def test_search_requires_auth(client: TestClient) -> None:
    assert client.get("/search", params={"q": "x"}).status_code == 401
