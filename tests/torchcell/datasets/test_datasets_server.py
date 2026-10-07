# tests/torchcell/datasets/test_datasets_server.py
# [[tests.torchcell.datasets.test_datasets_server]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_datasets_server.py
"""``torchcell.datasets.server`` (tc-data) on a hand-built artifact store and raw mirror.

The file is ``test_datasets_server.py`` (not ``test_server.py``) because
``tests/torchcell/literature/test_server.py`` already takes that basename under pytest's
prepend import mode; the pairing is registered in ``[tool.torchcell.test_exceptions.pairs]``.

2026.09.30 (Phase 15): the fixture store holds one indexed 5,120-byte archive
(``bytes(range(256)) * 20``, so every offset is distinguishable) plus one unlisted file,
and the raw mirror one manifested key with a 25-byte CSV, one bare key and one ``_``
service directory. Ranges follow RFC 9110 on both routes: ``bytes=5-10`` of the CSV is
the six bytes ``fitnes``, ``bytes=-120`` of the archive is offsets 5000 to 5119, and a
start past the end is 416 with ``bytes */5120``. The ``X-Artifact-SHA256`` header is
read from the index row or the manifest entry, never computed from the bytes (a
recorded ``"f" * 64`` is served as is). The error contract: missing bytes behind a
listed entry is 404 ``file not found``, a manifest entry that escapes its key directory
is 400, an unknown or escaping citation key is 404, and a manifest or index that does not
parse is 500 (a key with no manifest stays a clean 404). ``--gen-key`` prints a
``TC_DATA_KEYS_FILE`` line and starts nothing; ``main`` hands ``uvicorn.run`` the config
host and port unless overridden, and ``--port 0`` is passed through as 0.

2026.09.30 (issue #564): an underscore-prefixed name is a service directory, never a
citation key. ``/raw`` does not list it and every ``/raw/{key}/...`` route answers it 404
``unknown citation key``, the same as an absent key, even when the directory holds a
valid manifest and the file it lists (the literature server's ``_key_dir`` rule).

2026.10.07 (artifact tier, phase 2): the genomes tier (``GenomeManifest`` per assembly
set) and the objects tier (literature ``Manifest`` per object key) are served beside
``/raw`` through the same manifest-gated file path. The fixture genomes tier holds one
manifested set with a 5,120-byte container and an unlisted file, one set with no
manifest, and one ``_`` service directory; the objects tier one manifested key with a
nested ``emb/vectors.npy`` of 64 bytes. Unlike ``/raw``, the two new listings name only
keys that carry a ``manifest.json``. A root absent on disk lists ``[]`` and answers
every other route of its tier 404 ``unknown assembly set`` / ``unknown object key``;
``from_env`` does not require either root to exist.
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

from torchcell.datasets import server
from torchcell.datasets.artifact import ArtifactIndex, DatasetArtifact, archive_name
from torchcell.datasets.server import APP_TITLE, DataKeys, DataServerConfig, create_app
from torchcell.literature.manifest import ArtifactRecord, Manifest
from torchcell.sequence.genome.registry import GenomeManifest

GOOD_KEY = "data-secret-key-123"
HEADERS = {"X-API-Key": GOOD_KEY}
PAYLOAD = bytes(range(256)) * 20  # 5120 bytes, every offset distinguishable
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()
RAW_CSV = b"gene,fitness\nYAL001C,0.9\n"
RAW_CSV_SHA = hashlib.sha256(RAW_CSV).hexdigest()
ARCHIVE = archive_name("smf_fake", "1.2.1", PAYLOAD_SHA)
GENOME_SET = "fakeSet2018"
GENOME_FILE = "allReferenceGenesWithSNPsAndIndels.tar.gz"
OBJECT_KEY = "fakeObjects2024"
OBJECT_FILE = "emb/vectors.npy"
OBJECT_BYTES = bytes(range(64))
OBJECT_SHA = hashlib.sha256(OBJECT_BYTES).hexdigest()


def _artifact() -> DatasetArtifact:
    return DatasetArtifact(
        slug="smf_fake",
        dataset_class="SmfFakeDataset",
        torchcell_version="1.2.1",
        torchcell_commit="901d5ec31",
        content_sha256="c" * 64,
        n_experiments=3,
        archive=ARCHIVE,
        archive_sha256=PAYLOAD_SHA,
        archive_bytes=len(PAYLOAD),
        built_at="2026-09-14T02:25:00+00:00",
        packaged_at="2026-09-29T10:00:00+00:00",
    )


def build_store(root: Path) -> Path:
    """An artifact store with one indexed archive and one unlisted file."""
    store = root / "store"
    (store / "smf_fake").mkdir(parents=True)
    (store / "smf_fake" / ARCHIVE).write_bytes(PAYLOAD)
    (store / "smf_fake" / "unlisted-1.2.1-00000000.tar.xz").write_bytes(b"orphan")
    ArtifactIndex(
        generated_at="2026-09-29T10:00:00+00:00", artifacts=[_artifact()]
    ).save(store / "index.json")
    return store


def build_raw(root: Path) -> Path:
    """A raw mirror: one manifested key, one bare key, one service directory."""
    raw = root / "raw"
    key = raw / "fakeKey2020"
    (key / "data").mkdir(parents=True)
    (key / "data" / "table.csv").write_bytes(RAW_CSV)
    (key / "data" / "unlisted.csv").write_bytes(b"not in manifest")
    manifest = Manifest(
        citation_key="fakeKey2020",
        files=[
            ArtifactRecord(
                path="data/table.csv",
                role="raw_data",
                bytes=len(RAW_CSV),
                sha256=RAW_CSV_SHA,
            )
        ],
    )
    (key / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    (raw / "bareKey2021").mkdir()
    (raw / "_sync_reports").mkdir()
    return raw


def _genome_manifest(files: list[ArtifactRecord]) -> GenomeManifest:
    return GenomeManifest(
        assembly_set=GENOME_SET,
        organism="Saccharomyces cerevisiae",
        strain_or_population="1,011 isolates",
        source="Peter et al. 2018",
        release="2018",
        citation_key="peterGenomeEvolution10112018",
        files=files,
        provenance_complete=True,
        created_at="2026-10-07T00:00:00+00:00",
    )


def build_genomes(root: Path) -> Path:
    """A genomes tier: one manifested set, one set without a manifest, one service dir."""
    genomes = root / "genomes"
    key = genomes / GENOME_SET
    key.mkdir(parents=True)
    (key / GENOME_FILE).write_bytes(PAYLOAD)
    (key / "unlisted.fasta").write_bytes(b">x\nACGT\n")
    manifest = _genome_manifest(
        [
            ArtifactRecord(
                path=GENOME_FILE,
                role="container",
                bytes=len(PAYLOAD),
                sha256=PAYLOAD_SHA,
            )
        ]
    )
    (key / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    (genomes / "bareSet2020").mkdir()
    (genomes / "_staging").mkdir()
    return genomes


def build_objects(root: Path) -> Path:
    """An objects tier: one manifested key with a nested file, one bare key."""
    objects = root / "objects"
    key = objects / OBJECT_KEY
    (key / "emb").mkdir(parents=True)
    (key / OBJECT_FILE).write_bytes(OBJECT_BYTES)
    manifest = Manifest(
        citation_key=OBJECT_KEY,
        files=[
            ArtifactRecord(
                path=OBJECT_FILE,
                role="embedding",
                bytes=len(OBJECT_BYTES),
                sha256=OBJECT_SHA,
            )
        ],
    )
    (key / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    (objects / "bareObjects2025").mkdir()
    return objects


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    config = DataServerConfig(
        store_root=build_store(tmp_path),
        raw_root=build_raw(tmp_path),
        genomes_root=build_genomes(tmp_path),
        objects_root=build_objects(tmp_path),
        keys=DataKeys.from_pairs(f"mac:{GOOD_KEY}"),
    )
    return TestClient(create_app(config))


def test_health_needs_no_auth_and_counts_store_and_raw(
    client: TestClient, tmp_path: Path
) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "n_artifacts": 1,
        "n_raw_keys": 2,
        "n_genome_sets": 1,
        "n_object_keys": 1,
        "store_root": str(tmp_path / "store"),
        "raw_root": str(tmp_path / "raw"),
        "genomes_root": str(tmp_path / "genomes"),
        "objects_root": str(tmp_path / "objects"),
    }


@pytest.mark.parametrize(
    "path",
    [
        "/datasets",
        "/datasets/smf_fake",
        f"/datasets/smf_fake/{ARCHIVE}",
        "/raw",
        "/genomes",
        f"/genomes/{GENOME_SET}/manifest",
        f"/genomes/{GENOME_SET}/artifact/{GENOME_FILE}",
        "/objects",
        f"/objects/{OBJECT_KEY}/files",
        f"/objects/{OBJECT_KEY}/artifact/{OBJECT_FILE}",
    ],
)
def test_missing_or_bad_key_is_401(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get(path).json() == {"detail": "invalid or missing API key"}


def test_index_route_returns_the_store_index_verbatim(
    client: TestClient, tmp_path: Path
) -> None:
    resp = client.get("/datasets", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json() == json.loads((tmp_path / "store" / "index.json").read_text())


def test_slug_route_lists_that_slug_or_404(client: TestClient) -> None:
    resp = client.get("/datasets/smf_fake", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json() == [_artifact().model_dump(mode="json")]
    missing = client.get("/datasets/nope", headers=HEADERS)
    assert missing.status_code == 404
    assert missing.json() == {"detail": "no artifacts for this slug"}


def test_archive_streams_bytes_with_sha_header(client: TestClient) -> None:
    resp = client.get(f"/datasets/smf_fake/{ARCHIVE}", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.content == PAYLOAD
    assert resp.headers["X-Artifact-SHA256"] == PAYLOAD_SHA
    assert resp.headers["accept-ranges"] == "bytes"
    assert resp.headers["content-length"] == str(len(PAYLOAD))
    assert resp.headers["content-type"] == "application/x-xz"


def test_range_request_returns_206_with_the_exact_slice(client: TestClient) -> None:
    resp = client.get(
        f"/datasets/smf_fake/{ARCHIVE}", headers={**HEADERS, "Range": "bytes=100-199"}
    )
    assert resp.status_code == 206
    assert resp.content == PAYLOAD[100:200]
    assert resp.headers["content-range"] == f"bytes 100-199/{len(PAYLOAD)}"
    assert resp.headers["X-Artifact-SHA256"] == PAYLOAD_SHA
    tail = client.get(
        f"/datasets/smf_fake/{ARCHIVE}", headers={**HEADERS, "Range": "bytes=5000-"}
    )
    assert tail.status_code == 206
    assert tail.content == PAYLOAD[5000:]
    assert tail.headers["content-range"] == f"bytes 5000-5119/{len(PAYLOAD)}"


def test_unlisted_archive_and_traversal_are_404(client: TestClient) -> None:
    orphan = client.get(
        "/datasets/smf_fake/unlisted-1.2.1-00000000.tar.xz", headers=HEADERS
    )
    assert orphan.status_code == 404
    assert orphan.json() == {"detail": "artifact not in the index"}
    traversal = client.get("/datasets/smf_fake/%2e%2e%2findex.json", headers=HEADERS)
    assert traversal.status_code == 404
    assert b"schema_version" not in traversal.content


def test_index_missing_is_404_with_the_packager_hint(tmp_path: Path) -> None:
    store = tmp_path / "empty-store"
    store.mkdir()
    config = _config(store, build_raw(tmp_path), tmp_path)
    empty = TestClient(create_app(config))
    assert empty.get("/health").json()["n_artifacts"] == 0
    resp = empty.get("/datasets", headers=HEADERS)
    assert resp.status_code == 404
    assert "scripts/package_dataset_lmdb.py" in resp.json()["detail"]


def test_raw_keys_exclude_service_dirs(client: TestClient) -> None:
    resp = client.get("/raw", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json() == {"citation_keys": ["bareKey2021", "fakeKey2020"], "count": 2}


def test_underscore_directory_is_an_unknown_key_even_with_a_manifest(
    tmp_path: Path,
) -> None:
    raw = build_raw(tmp_path)
    hidden = raw / "_sync_reports"
    (hidden / "data").mkdir()
    (hidden / "data" / "table.csv").write_bytes(RAW_CSV)
    (hidden / "manifest.json").write_text(
        (raw / "fakeKey2020" / "manifest.json").read_text()
    )
    http = TestClient(create_app(_config(build_store(tmp_path), raw, tmp_path)))

    assert http.get("/raw", headers=HEADERS).json() == {
        "citation_keys": ["bareKey2021", "fakeKey2020"],
        "count": 2,
    }
    for route in ("manifest", "files", "artifact/data/table.csv"):
        resp = http.get(f"/raw/_sync_reports/{route}", headers=HEADERS)
        assert (resp.status_code, resp.json()) == (
            404,
            {"detail": "unknown citation key"},
        )
    # The same manifest under a citation-key name is served.
    listed = http.get("/raw/fakeKey2020/artifact/data/table.csv", headers=HEADERS)
    assert (listed.status_code, listed.content) == (200, RAW_CSV)


def test_raw_files_and_manifest_come_from_the_manifest(client: TestClient) -> None:
    files = client.get("/raw/fakeKey2020/files", headers=HEADERS)
    assert files.status_code == 200
    assert files.json() == [
        {
            "path": "data/table.csv",
            "role": "raw_data",
            "bytes": len(RAW_CSV),
            "sha256": RAW_CSV_SHA,
        }
    ]
    manifest = client.get("/raw/fakeKey2020/manifest", headers=HEADERS)
    assert manifest.status_code == 200
    assert manifest.json()["citation_key"] == "fakeKey2020"
    bare = client.get("/raw/bareKey2021/files", headers=HEADERS)
    assert bare.status_code == 404
    assert bare.json() == {"detail": "no manifest.json for this citation key"}
    assert client.get("/raw/nope/files", headers=HEADERS).status_code == 404


def test_raw_artifact_streams_listed_files_only(client: TestClient) -> None:
    resp = client.get("/raw/fakeKey2020/artifact/data/table.csv", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.content == RAW_CSV
    assert resp.headers["X-Artifact-SHA256"] == RAW_CSV_SHA
    unlisted = client.get(
        "/raw/fakeKey2020/artifact/data/unlisted.csv", headers=HEADERS
    )
    assert unlisted.status_code == 404
    assert unlisted.json() == {"detail": "file not in the manifest"}
    traversal = client.get(
        "/raw/fakeKey2020/artifact/../../etc/passwd", headers=HEADERS
    )
    assert traversal.status_code == 404
    assert b"root:" not in traversal.content


def test_openapi_lists_every_route_and_docs_is_served(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == APP_TITLE
    assert set(schema["paths"]) == {
        "/health",
        "/datasets",
        "/datasets/{slug}",
        "/datasets/{slug}/{archive}",
        "/raw",
        "/raw/{citation_key}/manifest",
        "/raw/{citation_key}/files",
        "/raw/{citation_key}/artifact/{rel_path}",
        "/genomes",
        "/genomes/{assembly_set}/manifest",
        "/genomes/{assembly_set}/files",
        "/genomes/{assembly_set}/artifact/{rel_path}",
        "/objects",
        "/objects/{object_key}/manifest",
        "/objects/{object_key}/files",
        "/objects/{object_key}/artifact/{rel_path}",
    }
    docs = client.get("/docs")
    assert docs.status_code == 200
    assert "swagger-ui" in docs.text


def test_config_from_env_reads_tc_data_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = build_store(tmp_path)
    raw = build_raw(tmp_path)
    genomes = build_genomes(tmp_path)
    objects = build_objects(tmp_path)
    monkeypatch.setenv("TC_DATA_ROOT", str(store))
    monkeypatch.setenv("TC_DATA_RAW_ROOT", str(raw))
    monkeypatch.setenv("TC_DATA_GENOMES_ROOT", str(genomes))
    monkeypatch.setenv("TC_DATA_OBJECTS_ROOT", str(objects))
    monkeypatch.setenv("TC_DATA_API_KEYS", f"mac:{GOOD_KEY}")
    monkeypatch.setenv("TC_DATA_PORT", "9001")
    monkeypatch.delenv("TC_DATA_KEYS_FILE", raising=False)
    config = DataServerConfig.from_env()
    assert (
        config.store_root,
        config.raw_root,
        config.genomes_root,
        config.objects_root,
        config.port,
    ) == (store, raw, genomes, objects, 9001)
    assert config.keys.verify(GOOD_KEY) == "mac"


def test_config_from_env_defaults_raw_root_under_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = build_store(tmp_path)
    (tmp_path / "torchcell-raw").mkdir()
    monkeypatch.setenv("TC_DATA_ROOT", str(store))
    for var in ("TC_DATA_RAW_ROOT", "TC_DATA_GENOMES_ROOT", "TC_DATA_OBJECTS_ROOT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("TC_DATA_API_KEYS", f"mac:{GOOD_KEY}")
    monkeypatch.delenv("TC_DATA_KEYS_FILE", raising=False)
    config = DataServerConfig.from_env()
    assert config.raw_root == tmp_path / "torchcell-raw"
    # The genomes and objects roots default under DATA_ROOT and need not exist.
    assert (config.genomes_root, config.objects_root) == (
        tmp_path / "torchcell-genomes",
        tmp_path / "torchcell-objects",
    )
    assert not config.genomes_root.exists() and not config.objects_root.exists()
    monkeypatch.setenv("TC_DATA_ROOT", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="artifact store does not exist"):
        DataServerConfig.from_env()


def test_data_keys_error_names_the_tc_data_variable() -> None:
    with pytest.raises(ValueError, match="bad TC_DATA_API_KEYS pair: 'mac'"):
        DataKeys.from_pairs("mac")


# --- 2026.09.30 (Phase 15): the remaining refusals, the header source, the CLI ------ #
def _config(store: Path, raw: Path, tmp_path: Path) -> DataServerConfig:
    """Store and raw mirror as given; genomes and objects roots that do not exist."""
    return DataServerConfig(
        store_root=store,
        raw_root=raw,
        genomes_root=tmp_path / "no-genomes",
        objects_root=tmp_path / "no-objects",
        keys=DataKeys.from_pairs(f"mac:{GOOD_KEY}"),
    )


def _clear_data_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "TC_DATA_KEYS_FILE",
        "TC_DATA_API_KEYS",
        "TC_DATA_HOST",
        "TC_DATA_PORT",
        "TC_DATA_RAW_ROOT",
        "TC_DATA_GENOMES_ROOT",
        "TC_DATA_OBJECTS_ROOT",
    ):
        monkeypatch.delenv(var, raising=False)


def _write_manifest(key_dir: Path, records: list[ArtifactRecord]) -> None:
    manifest = Manifest(citation_key=key_dir.name, files=records)
    (key_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))


def test_raw_range_request_is_a_206_slice_carrying_the_whole_file_hash(
    client: TestClient,
) -> None:
    """``bytes=5-10`` of the 25-byte CSV is ``b"fitnes"`` (offsets 5 through 10), with
    ``Content-Range: bytes 5-10/25``; the SHA header is still the manifest hash of the
    WHOLE file, never a hash of the slice.
    """
    resp = client.get(
        "/raw/fakeKey2020/artifact/data/table.csv",
        headers={**HEADERS, "Range": "bytes=5-10"},
    )
    assert resp.status_code == 206
    assert resp.content == RAW_CSV[5:11] == b"fitnes"
    assert resp.headers["content-range"] == f"bytes 5-10/{len(RAW_CSV)}"
    assert resp.headers["X-Artifact-SHA256"] == RAW_CSV_SHA


def test_suffix_and_unsatisfiable_ranges_on_an_archive(client: TestClient) -> None:
    """``bytes=-120`` is the last 120 bytes (offsets 5000 to 5119); a range starting past
    the 5,120-byte end is 416 with ``Content-Range: bytes */5120``.
    """
    url = f"/datasets/smf_fake/{ARCHIVE}"
    tail = client.get(url, headers={**HEADERS, "Range": "bytes=-120"})
    assert tail.status_code == 206
    assert tail.content == PAYLOAD[-120:]
    assert tail.headers["content-range"] == f"bytes 5000-5119/{len(PAYLOAD)}"
    past = client.get(url, headers={**HEADERS, "Range": "bytes=6000-"})
    assert past.status_code == 416
    assert past.headers["content-range"] == f"bytes */{len(PAYLOAD)}"


def test_sha_headers_come_from_the_index_and_manifest_not_from_the_bytes(
    tmp_path: Path,
) -> None:
    """The server never re-hashes: an index row and a manifest entry that record a hash
    the bytes do not have are served with that recorded hash, byte-for-byte unchanged.
    Verification is the client's job (``DatasetClient.download``).
    """
    store = build_store(tmp_path)
    ArtifactIndex(
        generated_at="2026-09-29T10:00:00+00:00",
        artifacts=[_artifact().model_copy(update={"archive_sha256": "f" * 64})],
    ).save(store / "index.json")
    raw = build_raw(tmp_path)
    _write_manifest(
        raw / "fakeKey2020",
        [
            ArtifactRecord(
                path="data/table.csv", role="raw_data", bytes=1, sha256="e" * 64
            )
        ],
    )
    http = TestClient(create_app(_config(store, raw, tmp_path)))
    archive = http.get(f"/datasets/smf_fake/{ARCHIVE}", headers=HEADERS)
    assert (archive.content, archive.headers["X-Artifact-SHA256"]) == (
        PAYLOAD,
        "f" * 64,
    )
    table = http.get("/raw/fakeKey2020/artifact/data/table.csv", headers=HEADERS)
    assert (table.content, table.headers["X-Artifact-SHA256"]) == (RAW_CSV, "e" * 64)


def test_indexed_or_manifested_file_missing_from_disk_is_404_file_not_found(
    tmp_path: Path,
) -> None:
    """The index or manifest lists it but the bytes are gone: 404 ``file not found``."""
    store = build_store(tmp_path)
    (store / "smf_fake" / ARCHIVE).unlink()
    raw = build_raw(tmp_path)
    (raw / "fakeKey2020" / "data" / "table.csv").unlink()
    http = TestClient(create_app(_config(store, raw, tmp_path)))
    for path in (
        f"/datasets/smf_fake/{ARCHIVE}",
        "/raw/fakeKey2020/artifact/data/table.csv",
    ):
        resp = http.get(path, headers=HEADERS)
        assert (resp.status_code, resp.json()) == (404, {"detail": "file not found"})


def test_manifest_entry_escaping_its_key_directory_is_400(tmp_path: Path) -> None:
    """A manifest that lists ``../otherKey2019/secret.csv`` names a real file outside the
    key's directory; the containment check answers 400 ``path traversal rejected`` and
    the file's bytes are not sent.
    """
    raw = build_raw(tmp_path)
    (raw / "otherKey2019").mkdir()
    (raw / "otherKey2019" / "secret.csv").write_bytes(b"secret")
    _write_manifest(
        raw / "fakeKey2020",
        [
            ArtifactRecord(
                path="../otherKey2019/secret.csv",
                role="raw_data",
                bytes=6,
                sha256=hashlib.sha256(b"secret").hexdigest(),
            )
        ],
    )
    http = TestClient(create_app(_config(build_store(tmp_path), raw, tmp_path)))
    resp = http.get(
        "/raw/fakeKey2020/artifact/%2e%2e/otherKey2019/secret.csv", headers=HEADERS
    )
    assert (resp.status_code, resp.json()) == (
        400,
        {"detail": "path traversal rejected"},
    )


def test_citation_key_resolving_outside_the_mirror_is_404(client: TestClient) -> None:
    """``..`` as the citation key resolves to the mirror's parent: 404 ``unknown citation
    key``, the same answer as a key that does not exist.
    """
    for key in ("%2e%2e", "noSuchKey1999"):
        resp = client.get(f"/raw/{key}/manifest", headers=HEADERS)
        assert (resp.status_code, resp.json()) == (
            404,
            {"detail": "unknown citation key"},
        )


def test_corrupt_manifest_is_500_while_a_missing_one_is_404(tmp_path: Path) -> None:
    """A ``manifest.json`` that does not parse is a server error (500) on every route that
    reads it, never a silent empty listing; a key with no manifest at all is the clean 404.
    A corrupt ``index.json`` is likewise 500 on ``/datasets`` and on ``/health``.
    """
    store = build_store(tmp_path)
    raw = build_raw(tmp_path)
    (raw / "fakeKey2020" / "manifest.json").write_text("{not json")
    http = TestClient(
        create_app(_config(store, raw, tmp_path)), raise_server_exceptions=False
    )
    for route in ("manifest", "files", "artifact/data/table.csv"):
        assert http.get(f"/raw/fakeKey2020/{route}", headers=HEADERS).status_code == 500
    bare = http.get("/raw/bareKey2021/manifest", headers=HEADERS)
    assert (bare.status_code, bare.json()) == (
        404,
        {"detail": "no manifest.json for this citation key"},
    )
    (store / "index.json").write_text("{not json")
    assert http.get("/datasets", headers=HEADERS).status_code == 500
    assert http.get("/health").status_code == 500


def test_config_from_env_refuses_a_missing_raw_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``TC_DATA_RAW_ROOT`` naming a directory that does not exist raises, naming it."""
    _clear_data_env(monkeypatch)
    missing = tmp_path / "no-raw"
    monkeypatch.setenv("TC_DATA_ROOT", str(build_store(tmp_path)))
    monkeypatch.setenv("TC_DATA_RAW_ROOT", str(missing))
    monkeypatch.setenv("TC_DATA_API_KEYS", f"mac:{GOOD_KEY}")
    with pytest.raises(FileNotFoundError) as excinfo:
        DataServerConfig.from_env()
    assert str(excinfo.value) == f"raw mirror does not exist: {missing}"


def test_create_app_from_env_binds_the_env_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``load_dotenv`` is replaced so the repo's ``.env`` cannot inject real keys; the app
    serves the store and mirror named by the variables.
    """
    _clear_data_env(monkeypatch)
    loads: list[bool] = []
    monkeypatch.setattr(server, "load_dotenv", lambda: loads.append(True))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("TC_DATA_ROOT", str(build_store(tmp_path)))
    monkeypatch.setenv("TC_DATA_RAW_ROOT", str(build_raw(tmp_path)))
    monkeypatch.setenv("TC_DATA_API_KEYS", f"mac:{GOOD_KEY}")
    app = server.create_app_from_env()
    assert loads == [True]
    http = TestClient(app)
    assert http.get("/raw", headers=HEADERS).json() == {
        "citation_keys": ["bareKey2021", "fakeKey2020"],
        "count": 2,
    }
    assert http.get("/health").json()["n_artifacts"] == 1


def test_gen_key_prints_a_key_and_the_tc_data_keys_file_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Five lines: the banner, the 43-character ``token_urlsafe(32)`` key, a blank line,
    the ``TC_DATA_KEYS_FILE`` instruction, and the JSON entry whose value is the sha256 of
    the printed key, which ``DataKeys`` then verifies. No server starts.
    """
    monkeypatch.setattr(server, "load_dotenv", lambda: None)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("server started"))
    monkeypatch.setattr(sys, "argv", ["tc-data", "--gen-key", "collab"])
    server.main()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 5
    assert (
        lines[0] == "API key for 'collab' (give this to the client, it is NOT stored):"
    )
    key_match = re.fullmatch(r"  ([A-Za-z0-9_-]{43})", lines[1])
    assert key_match is not None
    assert lines[2] == ""
    assert lines[3] == "Add this to your TC_DATA_KEYS_FILE (JSON of {name: sha256hex}):"
    entry = json.loads(lines[4].strip())
    assert entry == {"collab": hashlib.sha256(key_match.group(1).encode()).hexdigest()}
    assert DataKeys(hashes=entry).verify(key_match.group(1)) == "collab"


def test_main_runs_uvicorn_with_config_or_override_host_and_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No overrides: the config's host (default 0.0.0.0) and ``TC_DATA_PORT``.
    ``--host``/``--port`` win, and ``--port 0`` is honored (an ephemeral port, the
    usual meaning of 0 to a socket bind): the parsed ``args.port`` is 0 and ``0`` is
    what reaches ``uvicorn.run``. Each run logs both roots and the bound address.
    """
    caplog.set_level("INFO", logger="torchcell.datasets.server")
    _clear_data_env(monkeypatch)
    store = build_store(tmp_path)
    raw = build_raw(tmp_path)
    monkeypatch.setenv("TC_DATA_ROOT", str(store))
    monkeypatch.setenv("TC_DATA_RAW_ROOT", str(raw))
    monkeypatch.setenv("TC_DATA_API_KEYS", f"mac:{GOOD_KEY}")
    monkeypatch.setenv("TC_DATA_PORT", "9100")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
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
        ["tc-data"],
        ["tc-data", "--host", "127.0.0.1", "--port", "9200"],
        ["tc-data", "--port", "0"],
    ):
        monkeypatch.setattr(sys, "argv", argv)
        server.main()
    assert [ns.port for ns in parsed] == [None, 9200, 0]
    assert [(host, port) for _, host, port in calls] == [
        ("0.0.0.0", 9100),
        ("127.0.0.1", 9200),
        ("0.0.0.0", 0),
    ]
    assert calls[0][0].state.config.store_root == store
    tiers = (
        f"genomes {tmp_path / 'torchcell-genomes'}, "
        f"objects {tmp_path / 'torchcell-objects'}"
    )
    assert [r.getMessage() for r in caplog.records] == [
        f"dataset endpoint: store {store}, raw {raw}, {tiers} on 0.0.0.0:9100",
        f"dataset endpoint: store {store}, raw {raw}, {tiers} on 127.0.0.1:9200",
        f"dataset endpoint: store {store}, raw {raw}, {tiers} on 0.0.0.0:0",
    ]


# --- 2026.10.07 (artifact tier, phase 2): the genomes and objects tiers ------------- #
def test_genome_and_object_listings_name_only_manifested_keys(
    client: TestClient,
) -> None:
    """``bareSet2020`` / ``bareObjects2025`` have no manifest and ``_staging`` is a
    service directory: neither listing names them (``/raw`` still lists bare keys).
    """
    genomes = client.get("/genomes", headers=HEADERS)
    assert (genomes.status_code, genomes.json()) == (
        200,
        {"assembly_sets": [GENOME_SET], "count": 1},
    )
    objects = client.get("/objects", headers=HEADERS)
    assert (objects.status_code, objects.json()) == (
        200,
        {"object_keys": [OBJECT_KEY], "count": 1},
    )


def test_absent_tier_roots_list_nothing_and_404_everything_else(tmp_path: Path) -> None:
    """Roots that do not exist on disk: ``[]`` listings, zero health counts, and 404
    ``unknown <noun>`` on every per-key route; nothing is created on disk.
    """
    config = _config(build_store(tmp_path), build_raw(tmp_path), tmp_path)
    http = TestClient(create_app(config))
    assert http.get("/genomes", headers=HEADERS).json() == {
        "assembly_sets": [],
        "count": 0,
    }
    assert http.get("/objects", headers=HEADERS).json() == {
        "object_keys": [],
        "count": 0,
    }
    health = http.get("/health").json()
    assert (health["n_genome_sets"], health["n_object_keys"]) == (0, 0)
    for prefix, noun, key, rel in (
        ("genomes", "assembly set", GENOME_SET, GENOME_FILE),
        ("objects", "object key", OBJECT_KEY, OBJECT_FILE),
    ):
        for route in ("manifest", "files", f"artifact/{rel}"):
            resp = http.get(f"/{prefix}/{key}/{route}", headers=HEADERS)
            assert (resp.status_code, resp.json()) == (
                404,
                {"detail": f"unknown {noun}"},
            )
    assert not (tmp_path / "no-genomes").exists()
    assert not (tmp_path / "no-objects").exists()


def test_genome_manifest_round_trips_as_a_genome_manifest(
    client: TestClient, tmp_path: Path
) -> None:
    resp = client.get(f"/genomes/{GENOME_SET}/manifest", headers=HEADERS)
    assert resp.status_code == 200
    on_disk = GenomeManifest.model_validate_json(
        (tmp_path / "genomes" / GENOME_SET / "manifest.json").read_text()
    )
    assert GenomeManifest.model_validate(resp.json()) == on_disk
    assert resp.json()["organism"] == "Saccharomyces cerevisiae"


def test_object_manifest_round_trips_as_a_literature_manifest(
    client: TestClient, tmp_path: Path
) -> None:
    resp = client.get(f"/objects/{OBJECT_KEY}/manifest", headers=HEADERS)
    assert resp.status_code == 200
    on_disk = Manifest.model_validate_json(
        (tmp_path / "objects" / OBJECT_KEY / "manifest.json").read_text()
    )
    assert Manifest.model_validate(resp.json()) == on_disk


def test_tier_files_routes_list_the_manifest_rows(client: TestClient) -> None:
    genome_files = client.get(f"/genomes/{GENOME_SET}/files", headers=HEADERS)
    assert genome_files.json() == [
        {
            "path": GENOME_FILE,
            "role": "container",
            "bytes": len(PAYLOAD),
            "sha256": PAYLOAD_SHA,
        }
    ]
    object_files = client.get(f"/objects/{OBJECT_KEY}/files", headers=HEADERS)
    assert object_files.json() == [
        {
            "path": OBJECT_FILE,
            "role": "embedding",
            "bytes": len(OBJECT_BYTES),
            "sha256": OBJECT_SHA,
        }
    ]
    for path, noun in (
        ("/genomes/bareSet2020/files", "assembly set"),
        ("/objects/bareObjects2025/files", "object key"),
    ):
        resp = client.get(path, headers=HEADERS)
        assert (resp.status_code, resp.json()) == (
            404,
            {"detail": f"no manifest.json for this {noun}"},
        )


@pytest.mark.parametrize(
    ("url", "body", "sha"),
    [
        ("/raw/fakeKey2020/artifact/data/table.csv", RAW_CSV, RAW_CSV_SHA),
        (f"/genomes/{GENOME_SET}/artifact/{GENOME_FILE}", PAYLOAD, PAYLOAD_SHA),
        (f"/objects/{OBJECT_KEY}/artifact/{OBJECT_FILE}", OBJECT_BYTES, OBJECT_SHA),
    ],
)
def test_every_tier_streams_a_listed_file_the_same_way(
    client: TestClient, url: str, body: bytes, sha: str
) -> None:
    """Raw, genomes and objects go through one serving path: same media type, same
    header source, same ``Accept-Ranges``, and a ``bytes=5-10`` range is a 206 slice
    carrying the whole-file hash.
    """
    resp = client.get(url, headers=HEADERS)
    assert resp.status_code == 200
    assert resp.content == body
    assert resp.headers["X-Artifact-SHA256"] == sha
    assert resp.headers["content-type"] == "application/octet-stream"
    assert resp.headers["accept-ranges"] == "bytes"
    assert resp.headers["content-length"] == str(len(body))
    part = client.get(url, headers={**HEADERS, "Range": "bytes=5-10"})
    assert part.status_code == 206
    assert part.content == body[5:11]
    assert part.headers["content-range"] == f"bytes 5-10/{len(body)}"
    assert part.headers["X-Artifact-SHA256"] == sha


def test_tier_unlisted_file_service_dir_and_traversal_are_refused(
    client: TestClient, tmp_path: Path
) -> None:
    """An unlisted file is 404 ``file not in the manifest`` even though it is on disk;
    ``_staging`` answers like an absent set even with a valid manifest in it; a ``..``
    path never reaches bytes outside the set.
    """
    unlisted = client.get(
        f"/genomes/{GENOME_SET}/artifact/unlisted.fasta", headers=HEADERS
    )
    assert (unlisted.status_code, unlisted.json()) == (
        404,
        {"detail": "file not in the manifest"},
    )
    staging = tmp_path / "genomes" / "_staging"
    (staging / GENOME_FILE).write_bytes(PAYLOAD)
    (staging / "manifest.json").write_text(
        (tmp_path / "genomes" / GENOME_SET / "manifest.json").read_text()
    )
    hidden = client.get(f"/genomes/_staging/artifact/{GENOME_FILE}", headers=HEADERS)
    assert (hidden.status_code, hidden.json()) == (
        404,
        {"detail": "unknown assembly set"},
    )
    traversal = client.get(
        f"/objects/{OBJECT_KEY}/artifact/../../raw/fakeKey2020/data/table.csv",
        headers=HEADERS,
    )
    assert traversal.status_code == 404
    assert RAW_CSV not in traversal.content


def test_genome_manifest_naming_another_set_is_a_500(tmp_path: Path) -> None:
    """``registry.load_genome_manifest``'s rule on the server: a set directory whose
    manifest names a different ``assembly_set`` is an integrity error, never served.
    """
    genomes = build_genomes(tmp_path)
    renamed = genomes / "renamedSet2019"
    (genomes / GENOME_SET).rename(renamed)
    config = _config(build_store(tmp_path), build_raw(tmp_path), tmp_path)
    config = config.model_copy(update={"genomes_root": genomes})
    http = TestClient(create_app(config), raise_server_exceptions=False)
    for route in ("manifest", "files", f"artifact/{GENOME_FILE}"):
        resp = http.get(f"/genomes/renamedSet2019/{route}", headers=HEADERS)
        assert resp.status_code == 500
