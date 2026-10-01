# tests/torchcell/literature/test_server.py
# [[tests.torchcell.literature.test_server]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_server.py
"""``torchcell.literature.server`` (``tc-lit``) through ``TestClient`` on a hand-built mirror.

2026.09.30 - Phase 13 fixture and expected values. ``build_mirror`` writes, under
``tmp_path``, five entries at the mirror root:

* ``alphaKey2020/`` with ``paper.pdf`` (``PDF_BYTES``), ``paper.md`` (``MD_BYTES``),
  ``LICENSE`` (``PAYLOAD``, 5120 bytes, ``bytes(range(256)) * 20`` so every offset is
  distinguishable) and ``si/table.csv``; its hand-written ``manifest.json`` lists the
  first three with their true sha256 and deliberately leaves ``si/table.csv`` out;
* ``bareKey2021/`` with no manifest (the dynamic listing path): ``paper.md``,
  ``si/S1.pdf``, ``data/x.tsv`` and a nested ``data/manifest.json``;
* ``corruptKey2022/`` whose ``manifest.json`` is ``{not json`` beside a ``paper.md``;
* ``_bib/``, the bibliography store: ``manifest.json`` lists ``paper`` (``paper.bib``,
  present, ``BIB_BYTES``) and ``ghost`` (``ghost.bib``, absent on disk);
* ``_sync_reports/`` holding ``report.json`` (``{}``), and a plain file ``README.txt``.

So ``/keys`` lists exactly the three non-underscore directories, ``/health`` reports
``n_keys`` 3 and ``n_bibs`` 2, and every sha256 below is ``hashlib.sha256`` of the bytes
the fixture wrote. The 404-versus-500 contract from CLAUDE.md is pinned on
``corruptKey2022``: an absent file is a clean 404 (the file check runs before the
manifest is read), while a present file, ``/files`` and ``/manifest`` all reach the
corrupt manifest and surface as 500 with the pydantic ``ValidationError`` behind it.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

import torchcell.literature.server as server
from torchcell.api_keys import hash_key
from torchcell.literature.bib_store import BibRecord, BibScope, BibStoreManifest
from torchcell.literature.manifest import ArtifactRecord, Manifest
from torchcell.literature.server import (
    LiteratureKeys,
    LiteratureServerConfig,
    create_app,
)

GOOD_KEY = "mac-secret-key-123"
COLLAB_KEY = "another-key"
BAD_KEY = "not-a-real-key"
HEADERS = {"X-API-Key": GOOD_KEY}
PDF_BYTES = b"%PDF-1.5 fake pdf bytes"
MD_BYTES = b"# Fake paper\n\nBody mentions Glycolysis flux."
PAYLOAD = bytes(range(256)) * 20
BIB_BYTES = b"@article{a2020,\n  title = {A},\n}\n"


def sha(data: bytes) -> str:
    """sha256 hex of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _record(path: str, role: str, data: bytes) -> ArtifactRecord:
    return ArtifactRecord(path=path, role=role, bytes=len(data), sha256=sha(data))


def build_mirror(root: Path, *, with_bib: bool = True) -> Path:
    """The five-entry mirror described in the module docstring; returns ``root``."""
    alpha = root / "alphaKey2020"
    (alpha / "si").mkdir(parents=True)
    (alpha / "paper.pdf").write_bytes(PDF_BYTES)
    (alpha / "paper.md").write_bytes(MD_BYTES)
    (alpha / "LICENSE").write_bytes(PAYLOAD)
    (alpha / "si" / "table.csv").write_bytes(b"a,b\n1,2\n")
    manifest = Manifest(
        citation_key="alphaKey2020",
        doi="10.0/alpha",
        files=[
            _record("paper.pdf", "paper_pdf", PDF_BYTES),
            _record("paper.md", "paper_ocr", MD_BYTES),
            _record("LICENSE", "other", PAYLOAD),
        ],
        created_at="2026-09-30T00:00:00+00:00",
    )
    (alpha / "manifest.json").write_text(manifest.model_dump_json(indent=2))

    bare = root / "bareKey2021"
    (bare / "si").mkdir(parents=True)
    (bare / "data").mkdir()
    (bare / "paper.md").write_bytes(b"bare body")
    (bare / "si" / "S1.pdf").write_bytes(b"%PDF si")
    (bare / "data" / "x.tsv").write_bytes(b"g\tv\n")
    (bare / "data" / "manifest.json").write_bytes(b"{}")

    corrupt = root / "corruptKey2022"
    corrupt.mkdir()
    (corrupt / "manifest.json").write_text("{not json")
    (corrupt / "paper.md").write_bytes(b"corrupt body")

    (root / "_sync_reports").mkdir()
    (root / "_sync_reports" / "report.json").write_bytes(b"{}")
    (root / "README.txt").write_text("not a key")
    if with_bib:
        store = root / "_bib"
        store.mkdir()
        (store / "paper.bib").write_bytes(BIB_BYTES)
        scope = BibScope(group_library_id="123", group_collection="W46ATS7B")
        bib_manifest = BibStoreManifest(
            generated_at="2026-09-30T00:00:00+00:00",
            bibs=[
                BibRecord(
                    name="paper",
                    path="paper.bib",
                    bytes=len(BIB_BYTES),
                    sha256=sha(BIB_BYTES),
                    n_entries=1,
                    scope=scope,
                    origin="paper/nature-biotech/zotero_export_bib.py",
                    generated_at="2026-09-30T00:00:00+00:00",
                ),
                BibRecord(
                    name="ghost",
                    path="ghost.bib",
                    bytes=0,
                    sha256="0" * 64,
                    n_entries=0,
                    scope=scope,
                    origin="notes-tex/ghost/Makefile",
                    generated_at="2026-09-30T00:00:00+00:00",
                ),
            ],
        )
        (store / "manifest.json").write_text(bib_manifest.model_dump_json(indent=2))
    return root


def _config(root: Path) -> LiteratureServerConfig:
    keys = LiteratureKeys.from_pairs(f"mac:{GOOD_KEY},collab:{COLLAB_KEY}")
    return LiteratureServerConfig(mirror_root=root, keys=keys, port=8899)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(
        create_app(_config(build_mirror(tmp_path))), raise_server_exceptions=False
    )


# --- health, the key gate, the route table ------------------------------------------ #
def test_health_needs_no_auth_and_counts_keys_and_bibs(
    client: TestClient, tmp_path: Path
) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "n_keys": 3,
        "n_bibs": 2,
        "mirror_root": str(tmp_path),
    }


def test_health_without_a_bib_store_reports_zero_bibs(tmp_path: Path) -> None:
    app = create_app(_config(build_mirror(tmp_path, with_bib=False)))
    assert TestClient(app).get("/health").json()["n_bibs"] == 0


@pytest.mark.parametrize(
    "path",
    [
        "/keys",
        "/keys/alphaKey2020/manifest",
        "/keys/alphaKey2020/files",
        "/keys/alphaKey2020/artifact/paper.pdf",
        "/search?q=x",
        "/bib",
        "/bib/paper",
    ],
)
def test_every_keyed_route_refuses_a_missing_empty_or_wrong_key(
    client: TestClient, path: str
) -> None:
    """401 with the exact detail for no header, an empty header, and a wrong key."""
    for headers in ({}, {"X-API-Key": ""}, {"X-API-Key": BAD_KEY}):
        resp = client.get(path, headers=headers)
        assert resp.status_code == 401
        assert resp.json() == {"detail": "invalid or missing API key"}


def test_both_named_keys_are_accepted(client: TestClient) -> None:
    """Two named keys (``mac`` and ``collab``) each unlock the same listing."""
    mac = client.get("/keys", headers=HEADERS)
    collab = client.get("/keys", headers={"X-API-Key": COLLAB_KEY})
    assert (mac.status_code, collab.status_code) == (200, 200)
    assert mac.json() == collab.json()


def test_openapi_lists_exactly_the_eight_routes(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "torchcell literature endpoint"
    assert schema["info"]["version"] == "1.1.0"
    assert set(schema["paths"]) == {
        "/health",
        "/keys",
        "/keys/{citation_key}/manifest",
        "/keys/{citation_key}/files",
        "/keys/{citation_key}/artifact/{rel_path}",
        "/search",
        "/bib",
        "/bib/{name}",
    }


# --- /keys and /keys/{ck}/... ------------------------------------------------------- #
def test_list_keys_excludes_service_dirs_and_files_and_is_dynamic(
    client: TestClient, tmp_path: Path
) -> None:
    """``_bib``, ``_sync_reports`` and ``README.txt`` are not keys; a new dir shows up."""
    resp = client.get("/keys", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json() == {
        "citation_keys": ["alphaKey2020", "bareKey2021", "corruptKey2022"],
        "count": 3,
    }
    (tmp_path / "newKey2026").mkdir()
    assert client.get("/keys", headers=HEADERS).json() == {
        "citation_keys": [
            "alphaKey2020",
            "bareKey2021",
            "corruptKey2022",
            "newKey2026",
        ],
        "count": 4,
    }


def test_files_come_from_the_manifest_verbatim(client: TestClient) -> None:
    """Three manifest records in manifest order; the unlisted ``si/table.csv`` is absent."""
    resp = client.get("/keys/alphaKey2020/files", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json() == [
        {
            "path": "paper.pdf",
            "role": "paper_pdf",
            "bytes": len(PDF_BYTES),
            "sha256": sha(PDF_BYTES),
        },
        {
            "path": "paper.md",
            "role": "paper_ocr",
            "bytes": len(MD_BYTES),
            "sha256": sha(MD_BYTES),
        },
        {"path": "LICENSE", "role": "other", "bytes": 5120, "sha256": sha(PAYLOAD)},
    ]


def test_files_without_a_manifest_are_listed_live_with_null_sha(
    client: TestClient,
) -> None:
    """Sorted ``rglob`` order, roles from ``_role_for``, and every file named
    ``manifest.json`` skipped, the nested ``data/manifest.json`` included.
    """
    resp = client.get("/keys/bareKey2021/files", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json() == [
        {"path": "data/x.tsv", "role": "raw_data", "bytes": 4, "sha256": None},
        {"path": "paper.md", "role": "paper_ocr", "bytes": 9, "sha256": None},
        {"path": "si/S1.pdf", "role": "si_pdf", "bytes": 7, "sha256": None},
    ]


def test_manifest_route_returns_the_model_or_404_when_not_backfilled(
    client: TestClient,
) -> None:
    resp = client.get("/keys/alphaKey2020/manifest", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert (body["citation_key"], body["doi"], body["created_at"]) == (
        "alphaKey2020",
        "10.0/alpha",
        "2026-09-30T00:00:00+00:00",
    )
    assert [f["path"] for f in body["files"]] == ["paper.pdf", "paper.md", "LICENSE"]
    bare = client.get("/keys/bareKey2021/manifest", headers=HEADERS)
    assert bare.status_code == 404
    assert bare.json() == {"detail": "no manifest (not backfilled)"}


@pytest.mark.parametrize(
    "path",
    [
        "/keys/doesNotExist/files",
        "/keys/doesNotExist/manifest",
        "/keys/doesNotExist/artifact/paper.md",
        "/keys/%2e%2e/files",
    ],
)
def test_unknown_or_escaping_citation_key_is_404(client: TestClient, path: str) -> None:
    """An absent key and ``..`` (resolving to the mirror's parent) share one 404."""
    resp = client.get(path, headers=HEADERS)
    assert resp.status_code == 404
    assert resp.json() == {"detail": "unknown citation key"}


def test_service_directories_answer_as_unknown_citation_keys(
    client: TestClient,
) -> None:
    """An underscore directory is a service store, never a citation key, on every
    ``/keys`` route: ``_sync_reports`` and ``_bib`` answer the same clean 404 as an
    absent key, while ``/bib`` keeps serving the bibliography store on its own route.
    """
    for path in (
        "/keys/_sync_reports/files",
        "/keys/_sync_reports/manifest",
        "/keys/_sync_reports/artifact/report.json",
        "/keys/_bib/files",
        "/keys/_bib/manifest",
        "/keys/_bib/artifact/paper.bib",
    ):
        resp = client.get(path, headers=HEADERS)
        assert (path, resp.status_code, resp.json()) == (
            path,
            404,
            {"detail": "unknown citation key"},
        )
    bib = client.get("/bib/paper", headers=HEADERS)
    assert (bib.status_code, bib.content) == (200, BIB_BYTES)


# --- /keys/{ck}/artifact/{path} ----------------------------------------------------- #
def test_artifact_streams_exact_bytes_with_the_manifest_sha(client: TestClient) -> None:
    for rel, data in (("paper.pdf", PDF_BYTES), ("paper.md", MD_BYTES)):
        resp = client.get(f"/keys/alphaKey2020/artifact/{rel}", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.content == data
        assert resp.headers["X-Artifact-SHA256"] == sha(data)
    pdf = client.get("/keys/alphaKey2020/artifact/paper.pdf", headers=HEADERS)
    assert pdf.headers["content-type"] == "application/pdf"


def test_artifact_media_type_falls_back_to_octet_stream(client: TestClient) -> None:
    """``LICENSE`` has no extension, so ``mimetypes`` guesses None."""
    resp = client.get("/keys/alphaKey2020/artifact/LICENSE", headers=HEADERS)
    assert resp.headers["content-type"] == "application/octet-stream"
    assert resp.headers["content-length"] == "5120"


def test_artifact_without_a_manifest_record_carries_no_sha_header(
    client: TestClient,
) -> None:
    """``si/table.csv`` (alpha, unlisted) and ``paper.md`` (bare, no manifest) stream
    with no ``X-Artifact-SHA256``; the header is never computed live.
    """
    unlisted = client.get("/keys/alphaKey2020/artifact/si/table.csv", headers=HEADERS)
    bare = client.get("/keys/bareKey2021/artifact/paper.md", headers=HEADERS)
    assert (unlisted.status_code, unlisted.content) == (200, b"a,b\n1,2\n")
    assert (bare.status_code, bare.content) == (200, b"bare body")
    assert "x-artifact-sha256" not in unlisted.headers
    assert "x-artifact-sha256" not in bare.headers


def test_sha_header_is_the_manifest_claim_not_a_rehash(
    client: TestClient, tmp_path: Path
) -> None:
    """After the bytes change on disk the header still carries the manifest's digest,
    so a client comparing the header with its own hash detects the drift.
    """
    (tmp_path / "alphaKey2020" / "paper.md").write_bytes(b"tampered")
    resp = client.get("/keys/alphaKey2020/artifact/paper.md", headers=HEADERS)
    assert resp.content == b"tampered"
    assert resp.headers["X-Artifact-SHA256"] == sha(MD_BYTES)
    assert sha(resp.content) != resp.headers["X-Artifact-SHA256"]


def test_range_request_returns_206_with_the_exact_slice(client: TestClient) -> None:
    url = "/keys/alphaKey2020/artifact/LICENSE"
    mid = client.get(url, headers={**HEADERS, "Range": "bytes=100-199"})
    assert mid.status_code == 206
    assert mid.content == PAYLOAD[100:200]
    assert mid.headers["content-range"] == "bytes 100-199/5120"
    assert mid.headers["X-Artifact-SHA256"] == sha(PAYLOAD)
    tail = client.get(url, headers={**HEADERS, "Range": "bytes=5000-"})
    assert tail.status_code == 206
    assert tail.content == PAYLOAD[5000:]
    assert tail.headers["content-range"] == "bytes 5000-5119/5120"


@pytest.mark.parametrize(
    "rel",
    ["%2e%2e/bareKey2021/paper.md", "..%2FbareKey2021%2Fpaper.md", "escape/secret"],
)
def test_path_traversal_is_rejected_with_400(
    client: TestClient, tmp_path: Path, rel: str
) -> None:
    """Two encodings of ``../bareKey2021/paper.md`` and a symlink out of the key dir
    all resolve outside ``alphaKey2020`` and are refused before any read.
    """
    outside = tmp_path / "_outside"
    outside.mkdir()
    (outside / "secret").write_bytes(b"do not serve")
    (tmp_path / "alphaKey2020" / "escape").symlink_to(outside)
    resp = client.get(f"/keys/alphaKey2020/artifact/{rel}", headers=HEADERS)
    assert resp.status_code == 400
    assert resp.json() == {"detail": "path traversal rejected"}


@pytest.mark.parametrize("rel", ["nope.md", "si"])
def test_absent_file_or_a_directory_is_404(client: TestClient, rel: str) -> None:
    resp = client.get(f"/keys/alphaKey2020/artifact/{rel}", headers=HEADERS)
    assert resp.status_code == 404
    assert resp.json() == {"detail": "artifact not found"}


# --- the corrupt-manifest contract --------------------------------------------------- #
def test_corrupt_manifest_is_500_while_an_absent_file_is_a_clean_404(
    client: TestClient,
) -> None:
    """The CLAUDE.md contract: a genuinely absent file is 404 even under a corrupt
    manifest, because ``_resolve_artifact`` runs before ``_load_manifest``; anything that
    reads the manifest is a 500.
    """
    absent = client.get("/keys/corruptKey2022/artifact/nope.md", headers=HEADERS)
    assert absent.status_code == 404
    assert absent.json() == {"detail": "artifact not found"}
    for path in (
        "/keys/corruptKey2022/artifact/paper.md",
        "/keys/corruptKey2022/files",
        "/keys/corruptKey2022/manifest",
    ):
        resp = client.get(path, headers=HEADERS)
        assert resp.status_code == 500
        assert resp.text == "Internal Server Error"


def test_corrupt_manifest_raises_the_pydantic_validation_error(tmp_path: Path) -> None:
    """Behind the 500 is ``Manifest.model_validate_json`` failing on ``{not json``."""
    raising = TestClient(create_app(_config(build_mirror(tmp_path))))
    with pytest.raises(ValidationError, match="Invalid JSON"):
        raising.get("/keys/corruptKey2022/files", headers=HEADERS)


# --- /search ---------------------------------------------------------------------- #
def test_search_matches_key_names_and_paper_md_case_insensitively(
    client: TestClient,
) -> None:
    """``glycolysis`` is in alpha's ``paper.md`` (as ``Glycolysis``); ``KEY20`` matches
    all three key names; ``body`` is in the three ``paper.md`` files.
    """
    body = client.get("/search", params={"q": "glycolysis"}, headers=HEADERS).json()
    assert body == {
        "query": "glycolysis",
        "hits": [{"citation_key": "alphaKey2020", "where": ["paper.md"]}],
        "truncated": False,
    }
    names = client.get("/search", params={"q": "KEY20"}, headers=HEADERS).json()
    assert [h["citation_key"] for h in names["hits"]] == [
        "alphaKey2020",
        "bareKey2021",
        "corruptKey2022",
    ]
    assert all(h["where"] == ["citation_key"] for h in names["hits"])
    both = client.get("/search", params={"q": "alpha"}, headers=HEADERS).json()
    assert both["hits"] == [{"citation_key": "alphaKey2020", "where": ["citation_key"]}]
    corrupt = client.get("/search", params={"q": "corrupt"}, headers=HEADERS).json()
    assert corrupt["hits"] == [
        {"citation_key": "corruptKey2022", "where": ["citation_key", "paper.md"]}
    ]


def test_search_requires_q(client: TestClient) -> None:
    resp = client.get("/search", headers=HEADERS)
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["query", "q"]


def test_search_cap_truncates(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the cap at 2, ``KEY20`` stops after alpha and bare and flags truncation."""
    monkeypatch.setattr(server, "SEARCH_RESULT_CAP", 2)
    body = client.get("/search", params={"q": "key20"}, headers=HEADERS).json()
    assert [h["citation_key"] for h in body["hits"]] == ["alphaKey2020", "bareKey2021"]
    assert body["truncated"] is True


def test_search_is_not_truncated_when_hits_exactly_fill_the_cap(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``truncated`` means a hit was dropped: with the cap at 3 all three keys match
    ``key20``, none is dropped, so the result is complete and ``truncated`` is False.
    """
    monkeypatch.setattr(server, "SEARCH_RESULT_CAP", 3)
    body = client.get("/search", params={"q": "key20"}, headers=HEADERS).json()
    assert [h["citation_key"] for h in body["hits"]] == [
        "alphaKey2020",
        "bareKey2021",
        "corruptKey2022",
    ]
    assert body["truncated"] is False


# --- /bib --------------------------------------------------------------------------- #
def test_bib_lists_the_store_manifest_verbatim(
    client: TestClient, tmp_path: Path
) -> None:
    resp = client.get("/bib", headers=HEADERS)
    assert resp.status_code == 200
    on_disk = json.loads((tmp_path / "_bib" / "manifest.json").read_text())
    assert resp.json() == on_disk
    assert [b["name"] for b in resp.json()["bibs"]] == ["paper", "ghost"]


def test_bib_streams_bytes_with_sha_type_and_filename(client: TestClient) -> None:
    resp = client.get("/bib/paper", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.content == BIB_BYTES
    assert resp.headers["X-Artifact-SHA256"] == sha(BIB_BYTES)
    assert resp.headers["content-type"] == "application/x-bibtex"
    assert resp.headers["content-disposition"] == 'attachment; filename="paper.bib"'


@pytest.mark.parametrize(
    ("name", "status", "detail"),
    [
        ("nope", 404, "unknown bibliography"),
        ("ghost", 500, "bibliography listed in the store manifest but absent on disk"),
        (".hidden", 400, "illegal bibliography name: '.hidden'"),
        ("-dash", 400, "illegal bibliography name: '-dash'"),
    ],
)
def test_bib_name_errors(
    client: TestClient, name: str, status: int, detail: str
) -> None:
    """Unknown name 404, listed-but-absent 500, and a name failing
    ``^[A-Za-z0-9][A-Za-z0-9_.-]*$`` 400 with ``validate_bib_name``'s message.
    """
    resp = client.get(f"/bib/{name}", headers=HEADERS)
    assert resp.status_code == status
    assert resp.json() == {"detail": detail}


def test_bib_routes_404_without_a_store(tmp_path: Path) -> None:
    client = TestClient(create_app(_config(build_mirror(tmp_path, with_bib=False))))
    hint = "no bibliography store; run scripts/lit_bib_store.py on the host"
    for path in ("/bib", "/bib/paper"):
        resp = client.get(path, headers=HEADERS)
        assert resp.status_code == 404
        assert resp.json() == {"detail": hint}


# --- keys and configuration from the environment ---------------------------------- #
def _clear_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("TC_LIT_KEYS_FILE", "TC_LIT_API_KEYS", "TC_LIT_HOST", "TC_LIT_PORT"):
        monkeypatch.delenv(var, raising=False)


def test_keys_from_the_file_are_preferred_over_inline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The keys file holds ``{name: sha256hex}``; when both variables are set the file
    wins and the inline key no longer verifies.
    """
    _clear_key_env(monkeypatch)
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps({"filekey": hash_key("from-file")}))
    monkeypatch.setenv("TC_LIT_KEYS_FILE", str(keys_file))
    monkeypatch.setenv("TC_LIT_API_KEYS", "inline:from-inline")
    keys = LiteratureKeys.from_env()
    assert keys.hashes == {"filekey": hash_key("from-file")}
    assert keys.verify("from-file") == "filekey"
    assert keys.verify("from-inline") is None


def test_keys_from_the_inline_form_and_the_unauthenticated_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_key_env(monkeypatch)
    monkeypatch.setenv("TC_LIT_API_KEYS", " a:one , b:two ,")
    keys = LiteratureKeys.from_env()
    assert keys.hashes == {"a": hash_key("one"), "b": hash_key("two")}
    assert (keys.verify("one"), keys.verify("two")) == ("a", "b")
    monkeypatch.delenv("TC_LIT_API_KEYS")
    with pytest.raises(
        KeyError, match="Set TC_LIT_KEYS_FILE or TC_LIT_API_KEYS to run the server."
    ):
        LiteratureKeys.from_env()
    with pytest.raises(ValueError, match="bad TC_LIT_API_KEYS pair: 'mac'"):
        LiteratureKeys.from_pairs("mac")


def test_config_from_env_reads_data_root_host_and_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_key_env(monkeypatch)
    (tmp_path / "torchcell-library").mkdir()
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("TC_LIT_API_KEYS", f"mac:{GOOD_KEY}")
    defaults = LiteratureServerConfig.from_env()
    assert (defaults.mirror_root, defaults.host, defaults.port) == (
        tmp_path / "torchcell-library",
        "0.0.0.0",
        8723,
    )
    monkeypatch.setenv("TC_LIT_HOST", "127.0.0.1")
    monkeypatch.setenv("TC_LIT_PORT", "9001")
    custom = LiteratureServerConfig.from_env()
    assert (custom.host, custom.port) == ("127.0.0.1", 9001)
    assert custom.keys.verify(GOOD_KEY) == "mac"


def test_config_from_env_refuses_a_missing_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_key_env(monkeypatch)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("TC_LIT_API_KEYS", f"mac:{GOOD_KEY}")
    missing = tmp_path / "torchcell-library"
    with pytest.raises(
        FileNotFoundError, match=re.escape(f"mirror root does not exist: {missing}")
    ):
        LiteratureServerConfig.from_env()


def test_create_app_from_env_binds_the_env_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``load_dotenv`` is replaced so the repo's ``.env`` cannot inject real keys."""
    _clear_key_env(monkeypatch)
    loads: list[bool] = []
    monkeypatch.setattr(server, "load_dotenv", lambda: loads.append(True))
    build_mirror(tmp_path / "torchcell-library")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("TC_LIT_API_KEYS", f"mac:{GOOD_KEY}")
    app = server.create_app_from_env()
    assert loads == [True]
    assert app.state.config.mirror_root == tmp_path / "torchcell-library"
    keys = TestClient(app).get("/keys", headers=HEADERS).json()
    assert keys["count"] == 3


# --- the CLI ------------------------------------------------------------------------ #
def test_gen_key_prints_a_key_and_the_matching_keys_file_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Five lines: the banner, the key (43 url-safe characters from
    ``token_urlsafe(32)``), a blank line, the keys-file instruction, and the JSON entry
    whose value is the sha256 of the printed key.
    """
    monkeypatch.setattr(server, "load_dotenv", lambda: None)
    monkeypatch.setattr(sys, "argv", ["server", "--gen-key", "collab"])
    server.main()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 5
    assert (
        lines[0] == "API key for 'collab' (give this to the client, it is NOT stored):"
    )
    key_match = re.fullmatch(r"  ([A-Za-z0-9_-]{43})", lines[1])
    assert key_match is not None
    assert lines[2] == ""
    assert lines[3] == "Add this to your TC_LIT_KEYS_FILE (JSON of {name: sha256hex}):"
    entry = json.loads(lines[4].strip())
    assert entry == {"collab": hash_key(key_match.group(1))}
    assert LiteratureKeys(hashes=entry).verify(key_match.group(1)) == "collab"


def test_main_runs_uvicorn_with_config_or_override_host_and_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No overrides: the config's host and port. ``--host``/``--port`` win, and
    ``--port 0`` is honored (an ephemeral port, the usual meaning of 0 to a socket
    bind): the parsed ``args.port`` is 0 and ``0`` is what reaches ``uvicorn.run``.
    Each run logs the mirror and the bound address it hands to ``uvicorn.run``.
    """
    caplog.set_level("INFO", logger="torchcell.literature.server")
    _clear_key_env(monkeypatch)
    build_mirror(tmp_path / "torchcell-library")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("TC_LIT_API_KEYS", f"mac:{GOOD_KEY}")
    monkeypatch.setenv("TC_LIT_PORT", "9100")
    monkeypatch.setattr(server, "load_dotenv", lambda: None)
    calls: list[tuple[Any, str, int]] = []

    def fake_run(app: FastAPI, host: str, port: int) -> None:
        calls.append((app, host, port))

    monkeypatch.setattr(uvicorn, "run", fake_run)
    parsed: list[argparse.Namespace] = []
    real_parse_args = argparse.ArgumentParser.parse_args

    def recording_parse_args(
        self: argparse.ArgumentParser, *args: Any, **kwargs: Any
    ) -> argparse.Namespace:
        namespace: argparse.Namespace = real_parse_args(self, *args, **kwargs)
        parsed.append(namespace)
        return namespace

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", recording_parse_args)
    for argv in (
        ["server"],
        ["server", "--host", "127.0.0.1", "--port", "9200"],
        ["server", "--port", "0"],
    ):
        monkeypatch.setattr(sys, "argv", argv)
        server.main()
    assert [ns.port for ns in parsed] == [None, 9200, 0]
    assert [(host, port) for _, host, port in calls] == [
        ("0.0.0.0", 9100),
        ("127.0.0.1", 9200),
        ("0.0.0.0", 0),
    ]
    app = calls[0][0]
    assert isinstance(app, FastAPI)
    assert app.state.config.mirror_root == tmp_path / "torchcell-library"
    mirror = tmp_path / "torchcell-library"
    assert [r.getMessage() for r in caplog.records] == [
        f"literature endpoint: serving {mirror} on 0.0.0.0:9100",
        f"literature endpoint: serving {mirror} on 127.0.0.1:9200",
        f"literature endpoint: serving {mirror} on 0.0.0.0:0",
    ]
