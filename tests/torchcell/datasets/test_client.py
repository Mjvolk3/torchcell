# tests/torchcell/datasets/test_client.py
# [[tests.torchcell.datasets.test_client]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/test_client.py
"""``torchcell.datasets.client`` against the tc-data app through the ASGI test client.

2026.10.07 (artifact tier, phase 2): the genomes and objects methods run against a
genomes tier holding one ``GenomeManifest`` set with a 5,120-byte container and an
objects tier holding one literature-``Manifest`` key with a nested 64-byte file. A
tampering HTTP client (wrapping the ASGI client) rewrites the hash header or the body to
pin that the client checks both; a file changed on disk after its manifest was written
is the same tamper seen from the server side. The raw mirror gets the same three
methods (``raw_manifest``, ``raw_files``, ``download_raw_file``) on the same path,
pinned here on a 25-byte CSV under one manifested citation key.

2026.10.12 (public read-only): the client sends no key. ``DatasetClient()`` defaults to
the public ``DEFAULT_URL``, ``from_env`` prefers ``TC_DATA_URL`` when set, and the only
header a request carries is ``Range`` on a resume.

2026.10.19 (library tier): ``library_keys``, ``library_manifest``, ``library_files`` and
``download_library_file`` run against a library tier holding one key with an SI GenBank
file whose name carries parentheses, beside a paper PDF and an OCR image the server
withholds. The client never sees the withheld records, so asking for the paper raises
``KeyError`` before any byte is requested.
"""

import hashlib
import io
import tarfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from torchcell.datasets.artifact import (
    ArtifactIndex,
    DatasetArtifact,
    ManifestFileListing,
    archive_name,
)
from torchcell.datasets.client import (
    ArtifactIntegrityError,
    DatasetClient,
    EndpointError,
    unpack_artifact,
)
from torchcell.datasets.server import DataServerConfig, create_app
from torchcell.endpoint_http import DEFAULT_URL
from torchcell.literature.manifest import ArtifactRecord, Manifest
from torchcell.sequence.genome.registry import GenomeManifest

PAYLOAD = bytes(range(256)) * 20
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()
ARCHIVE = archive_name("smf_fake", "1.2.1", PAYLOAD_SHA)
GENOME_SET = "fakeSet2018"
GENOME_FILE = "allReferenceGenesWithSNPsAndIndels.tar.gz"
OBJECT_KEY = "fakeObjects2024"
OBJECT_FILE = "emb/vectors.npy"
OBJECT_BYTES = bytes(range(64))
OBJECT_SHA = hashlib.sha256(OBJECT_BYTES).hexdigest()
RAW_KEY = "fakeKey2020"
RAW_FILE = "data/table.csv"
RAW_CSV = b"gene,fitness\nYAL001C,0.9\n"
RAW_CSV_SHA = hashlib.sha256(RAW_CSV).hexdigest()
LIBRARY_KEY = "cacheraCRISPAHighthroughputMethod2023"
LIBRARY_FILE = "si/plasmids/pBTX001(ante113).gb"
LIBRARY_BYTES = b"LOCUS       pBTX001  5120 bp  DNA  circular\nORIGIN\n//\n"
LIBRARY_SHA = hashlib.sha256(LIBRARY_BYTES).hexdigest()
LIBRARY_PAPER = b"%PDF-1.7 the paper itself"
LIBRARY_OCR = b"\x89PNG ocr page image"


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


def _genomes(root: Path) -> Path:
    genomes = root / "genomes"
    (genomes / GENOME_SET).mkdir(parents=True)
    (genomes / GENOME_SET / GENOME_FILE).write_bytes(PAYLOAD)
    manifest = GenomeManifest(
        assembly_set=GENOME_SET,
        organism="Saccharomyces cerevisiae",
        strain_or_population="1,011 isolates",
        source="Peter et al. 2018",
        release="2018",
        files=[
            ArtifactRecord(
                path=GENOME_FILE,
                role="container",
                bytes=len(PAYLOAD),
                sha256=PAYLOAD_SHA,
            )
        ],
        provenance_complete=True,
        created_at="2026-10-07T00:00:00+00:00",
    )
    (genomes / GENOME_SET / "manifest.json").write_text(manifest.model_dump_json())
    return genomes


def _objects(root: Path) -> Path:
    objects = root / "objects"
    (objects / OBJECT_KEY / "emb").mkdir(parents=True)
    (objects / OBJECT_KEY / OBJECT_FILE).write_bytes(OBJECT_BYTES)
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
    (objects / OBJECT_KEY / "manifest.json").write_text(manifest.model_dump_json())
    return objects


def _raw(root: Path) -> Path:
    raw = root / "raw"
    (raw / RAW_KEY / "data").mkdir(parents=True)
    (raw / RAW_KEY / RAW_FILE).write_bytes(RAW_CSV)
    manifest = Manifest(
        citation_key=RAW_KEY,
        files=[
            ArtifactRecord(
                path=RAW_FILE, role="raw_data", bytes=len(RAW_CSV), sha256=RAW_CSV_SHA
            )
        ],
    )
    (raw / RAW_KEY / "manifest.json").write_text(manifest.model_dump_json())
    return raw


def _library(root: Path) -> Path:
    library = root / "library"
    key = library / LIBRARY_KEY
    (key / "si" / "plasmids").mkdir(parents=True)
    (key / "images").mkdir()
    (key / LIBRARY_FILE).write_bytes(LIBRARY_BYTES)
    (key / "paper.pdf").write_bytes(LIBRARY_PAPER)
    (key / "images" / "p1.png").write_bytes(LIBRARY_OCR)
    manifest = Manifest(
        citation_key=LIBRARY_KEY,
        files=[
            ArtifactRecord(
                path="paper.pdf",
                role="paper_pdf",
                bytes=len(LIBRARY_PAPER),
                sha256=hashlib.sha256(LIBRARY_PAPER).hexdigest(),
            ),
            ArtifactRecord(
                path=LIBRARY_FILE,
                role="si_data",
                bytes=len(LIBRARY_BYTES),
                sha256=LIBRARY_SHA,
            ),
            ArtifactRecord(
                path="images/p1.png",
                role="ocr_image",
                bytes=len(LIBRARY_OCR),
                sha256=hashlib.sha256(LIBRARY_OCR).hexdigest(),
            ),
        ],
    )
    (key / "manifest.json").write_text(manifest.model_dump_json())
    return library


def _client(root: Path, artifact: DatasetArtifact) -> DatasetClient:
    raw = _raw(root)
    config = DataServerConfig(
        store_root=_store(root, artifact),
        raw_root=raw,
        genomes_root=_genomes(root),
        objects_root=_objects(root),
        library_root=_library(root),
    )
    return DatasetClient("http://testserver/", http=TestClient(create_app(config)))


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
    assert seen == [{"Range": "bytes=1000-"}]
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
    with pytest.raises(EndpointError, match="expected HTTP 200, got 404"):
        client.download(
            _artifact().model_copy(update={"archive": "missing.tar.xz"}),
            tmp_path / "out" / "missing.tar.xz",
        )


def test_from_env_reads_url_with_the_public_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset ``TC_DATA_URL`` is the public endpoint; set, it wins (trailing slash
    stripped). ``DatasetClient()`` with no arguments is the public endpoint too.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    assert DatasetClient.from_env(http=httpx.Client()).url == DEFAULT_URL
    assert DatasetClient(http=httpx.Client()).url == DEFAULT_URL
    monkeypatch.setenv("TC_DATA_URL", "http://radiant:8724/")
    assert DatasetClient.from_env(http=httpx.Client()).url == "http://radiant:8724"


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


# --- 2026.10.07 (artifact tier, phase 2): genomes and objects ---------------------- #
class _TamperedStream:
    """A streamed response whose header or body was rewritten in transit."""

    def __init__(self, inner: Any, header: str | None, body: bytes | None) -> None:
        self._inner = inner
        self.status_code = inner.status_code
        self.request = inner.request
        self.headers = httpx.Headers(inner.headers)
        if header is not None:
            self.headers["X-Artifact-SHA256"] = header
        self._body = body

    def iter_bytes(self, chunk_size: int) -> Iterator[bytes]:
        if self._body is None:
            yield from self._inner.iter_bytes(chunk_size)
        else:
            yield self._body


class TamperingHttp:
    """An ``HttpClient`` over the ASGI test client that rewrites streamed responses."""

    def __init__(
        self, inner: TestClient, header: str | None = None, body: bytes | None = None
    ) -> None:
        """Wrap ``inner``; a non-None ``header`` or ``body`` replaces the served one."""
        self._inner = inner
        self._header = header
        self._body = body

    def get(self, url: str) -> Any:
        """Buffered GETs (listings, manifests) pass through untouched."""
        return self._inner.get(url)

    @contextmanager
    def stream(
        self, method: str, url: str, *, headers: Mapping[str, str] | None = None
    ) -> Iterator[_TamperedStream]:
        """The ASGI stream with the header and/or body rewritten."""
        with self._inner.stream(method, url, headers=headers) as response:
            yield _TamperedStream(response, self._header, self._body)


def _tampering_client(
    root: Path, header: str | None, body: bytes | None
) -> DatasetClient:
    plain = _client(root, _artifact())
    assert isinstance(plain._http, TestClient)
    return DatasetClient(
        "http://testserver", http=TamperingHttp(plain._http, header, body)
    )


def test_tier_listings_and_manifests_round_trip(tmp_path: Path) -> None:
    client = _client(tmp_path, _artifact())
    assert client.genomes() == [GENOME_SET]
    assert client.objects() == [OBJECT_KEY]
    assert client.genome_manifest(GENOME_SET) == GenomeManifest.model_validate_json(
        (tmp_path / "genomes" / GENOME_SET / "manifest.json").read_text()
    )
    assert client.object_manifest(OBJECT_KEY) == Manifest.model_validate_json(
        (tmp_path / "objects" / OBJECT_KEY / "manifest.json").read_text()
    )
    assert client.genome_files(GENOME_SET) == [
        ManifestFileListing(
            path=GENOME_FILE, role="container", bytes=len(PAYLOAD), sha256=PAYLOAD_SHA
        )
    ]
    assert client.object_files(OBJECT_KEY) == [
        ManifestFileListing(
            path=OBJECT_FILE,
            role="embedding",
            bytes=len(OBJECT_BYTES),
            sha256=OBJECT_SHA,
        )
    ]


def test_tier_downloads_verify_and_rename_into_place(tmp_path: Path) -> None:
    client = _client(tmp_path, _artifact())
    genome_dest = tmp_path / "out" / GENOME_FILE
    assert client.download_genome_file(GENOME_SET, GENOME_FILE, genome_dest) == (
        genome_dest
    )
    assert genome_dest.read_bytes() == PAYLOAD
    object_dest = tmp_path / "out" / "vectors.npy"
    assert client.download_object_file(OBJECT_KEY, OBJECT_FILE, object_dest) == (
        object_dest
    )
    assert object_dest.read_bytes() == OBJECT_BYTES
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        GENOME_FILE,
        "vectors.npy",
    ]


def test_tier_download_resumes_a_partial_file_with_a_range_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, _artifact())
    dest = tmp_path / "out" / GENOME_FILE
    dest.parent.mkdir()
    dest.with_name(dest.name + ".part").write_bytes(PAYLOAD[:1000])
    seen: list[dict[str, str]] = []
    original = client._http.stream

    def spy(method: str, url: str, **kwargs: Any) -> Any:
        seen.append(dict(kwargs["headers"]))
        return original(method, url, **kwargs)

    monkeypatch.setattr(client._http, "stream", spy)
    client.download_genome_file(GENOME_SET, GENOME_FILE, dest)
    assert seen == [{"Range": "bytes=1000-"}]
    assert dest.read_bytes() == PAYLOAD


def test_tier_download_refuses_a_header_that_disagrees_with_the_manifest(
    tmp_path: Path,
) -> None:
    client = _tampering_client(tmp_path, header="0" * 64, body=None)
    dest = tmp_path / "out" / "vectors.npy"
    with pytest.raises(
        ArtifactIntegrityError,
        match=f"objects/{OBJECT_KEY}/{OBJECT_FILE}: server X-Artifact-SHA256 0{{64}} "
        f"!= manifest {OBJECT_SHA}",
    ):
        client.download_object_file(OBJECT_KEY, OBJECT_FILE, dest)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_tier_download_refuses_a_tampered_body_and_removes_the_partial(
    tmp_path: Path,
) -> None:
    client = _tampering_client(tmp_path, header=None, body=b"\x00" * len(PAYLOAD))
    dest = tmp_path / "out" / GENOME_FILE
    with pytest.raises(ArtifactIntegrityError, match="partial file removed"):
        client.download_genome_file(GENOME_SET, GENOME_FILE, dest)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_file_changed_on_disk_after_its_manifest_is_refused(tmp_path: Path) -> None:
    """The server sends the manifest hash for bytes that no longer have it; the client
    hashes what arrived and refuses.
    """
    client = _client(tmp_path, _artifact())
    (tmp_path / "objects" / OBJECT_KEY / OBJECT_FILE).write_bytes(b"\xff" * 64)
    dest = tmp_path / "out" / "vectors.npy"
    with pytest.raises(
        ArtifactIntegrityError,
        match=f"sha256 {hashlib.sha256(b'\xff' * 64).hexdigest()} != manifest "
        f"{OBJECT_SHA}; partial file removed",
    ):
        client.download_object_file(OBJECT_KEY, OBJECT_FILE, dest)
    assert not dest.exists()


def test_tier_download_of_an_unlisted_path_raises_before_any_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, _artifact())
    monkeypatch.setattr(client._http, "stream", lambda *a, **k: pytest.fail("streamed"))
    with pytest.raises(KeyError, match="genomes/fakeSet2018/unlisted.fasta"):
        client.download_genome_file(GENOME_SET, "unlisted.fasta", tmp_path / "x")


def test_tier_methods_raise_on_an_unknown_key(tmp_path: Path) -> None:
    client = _client(tmp_path, _artifact())
    with pytest.raises(EndpointError, match="HTTP 404"):
        client.genome_manifest("noSuchSet")
    with pytest.raises(EndpointError, match="HTTP 404"):
        client.object_files("noSuchKey")


# --- 2026.10.07: the raw-tier counterpart on the same streaming-and-verify path ----- #
def test_raw_manifest_and_files_round_trip(tmp_path: Path) -> None:
    client = _client(tmp_path, _artifact())
    assert client.raw_manifest(RAW_KEY) == Manifest.model_validate_json(
        (tmp_path / "raw" / RAW_KEY / "manifest.json").read_text()
    )
    assert client.raw_files(RAW_KEY) == [
        ManifestFileListing(
            path=RAW_FILE, role="raw_data", bytes=len(RAW_CSV), sha256=RAW_CSV_SHA
        )
    ]


def test_raw_download_verifies_and_resumes_with_a_range_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, _artifact())
    whole = tmp_path / "out" / "table.csv"
    assert client.download_raw_file(RAW_KEY, RAW_FILE, whole) == whole
    assert whole.read_bytes() == RAW_CSV
    resumed = tmp_path / "out" / "resumed.csv"
    resumed.with_name(resumed.name + ".part").write_bytes(RAW_CSV[:5])
    seen: list[dict[str, str]] = []
    original = client._http.stream

    def spy(method: str, url: str, **kwargs: Any) -> Any:
        seen.append(dict(kwargs["headers"]))
        return original(method, url, **kwargs)

    monkeypatch.setattr(client._http, "stream", spy)
    client.download_raw_file(RAW_KEY, RAW_FILE, resumed)
    assert seen == [{"Range": "bytes=5-"}]
    assert resumed.read_bytes() == RAW_CSV


def test_raw_download_refuses_a_header_that_disagrees_with_the_manifest(
    tmp_path: Path,
) -> None:
    client = _tampering_client(tmp_path, header="0" * 64, body=None)
    dest = tmp_path / "out" / "table.csv"
    with pytest.raises(
        ArtifactIntegrityError,
        match=f"raw/{RAW_KEY}/{RAW_FILE}: server X-Artifact-SHA256 0{{64}} "
        f"!= manifest {RAW_CSV_SHA}",
    ):
        client.download_raw_file(RAW_KEY, RAW_FILE, dest)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_raw_download_refuses_a_tampered_body_and_removes_the_partial(
    tmp_path: Path,
) -> None:
    tampered = b"x" * len(RAW_CSV)
    client = _tampering_client(tmp_path, header=None, body=tampered)
    dest = tmp_path / "out" / "table.csv"
    with pytest.raises(
        ArtifactIntegrityError,
        match=f"sha256 {hashlib.sha256(tampered).hexdigest()} != manifest "
        f"{RAW_CSV_SHA}; partial file removed",
    ):
        client.download_raw_file(RAW_KEY, RAW_FILE, dest)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_raw_download_of_an_unlisted_path_raises_before_any_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, _artifact())
    monkeypatch.setattr(client._http, "stream", lambda *a, **k: pytest.fail("streamed"))
    with pytest.raises(KeyError, match=f"raw/{RAW_KEY}/data/unlisted.csv"):
        client.download_raw_file(RAW_KEY, "data/unlisted.csv", tmp_path / "x")


# --- 2026.10.19: the library tier, supplementary material only ---------------------- #
def test_library_listing_manifest_and_files_hold_only_the_si_file(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path, _artifact())
    assert client.library_keys() == [LIBRARY_KEY]
    on_disk = Manifest.model_validate_json(
        (tmp_path / "library" / LIBRARY_KEY / "manifest.json").read_text()
    )
    public = client.library_manifest(LIBRARY_KEY)
    assert public == on_disk.model_copy(update={"files": [on_disk.files[1]]})
    assert client.library_files(LIBRARY_KEY) == [
        ManifestFileListing(
            path=LIBRARY_FILE,
            role="si_data",
            bytes=len(LIBRARY_BYTES),
            sha256=LIBRARY_SHA,
        )
    ]


def test_library_download_verifies_and_resumes_with_a_range_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parenthesized path is percent-encoded on the way out and the bytes that come
    back are the GenBank file, hash-checked against the ``/files`` row.
    """
    client = _client(tmp_path, _artifact())
    whole = tmp_path / "out" / "pBTX001(ante113).gb"
    assert client.download_library_file(LIBRARY_KEY, LIBRARY_FILE, whole) == whole
    assert whole.read_bytes() == LIBRARY_BYTES
    resumed = tmp_path / "out" / "resumed.gb"
    resumed.with_name(resumed.name + ".part").write_bytes(LIBRARY_BYTES[:7])
    seen: list[tuple[str, dict[str, str]]] = []
    original = client._http.stream

    def spy(method: str, url: str, **kwargs: Any) -> Any:
        seen.append((url, dict(kwargs["headers"])))
        return original(method, url, **kwargs)

    monkeypatch.setattr(client._http, "stream", spy)
    client.download_library_file(LIBRARY_KEY, LIBRARY_FILE, resumed)
    assert seen == [
        (
            f"http://testserver/library/{LIBRARY_KEY}/artifact/"
            "si/plasmids/pBTX001%28ante113%29.gb",
            {"Range": "bytes=7-"},
        )
    ]
    assert resumed.read_bytes() == LIBRARY_BYTES


def test_library_download_refuses_a_tampered_body_and_removes_the_partial(
    tmp_path: Path,
) -> None:
    tampered = b"x" * len(LIBRARY_BYTES)
    client = _tampering_client(tmp_path, header=None, body=tampered)
    dest = tmp_path / "out" / "plasmid.gb"
    with pytest.raises(
        ArtifactIntegrityError,
        match=f"sha256 {hashlib.sha256(tampered).hexdigest()} != manifest "
        f"{LIBRARY_SHA}; partial file removed",
    ):
        client.download_library_file(LIBRARY_KEY, LIBRARY_FILE, dest)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


@pytest.mark.parametrize("rel", ["paper.pdf", "images/p1.png", "si/absent.gb"])
def test_library_download_of_a_path_the_public_manifest_lacks_raises_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rel: str
) -> None:
    """The paper and OCR image are in the on-disk manifest but not the public one, so
    the client refuses them exactly like a path no manifest lists, before streaming.
    """
    client = _client(tmp_path, _artifact())
    monkeypatch.setattr(client._http, "stream", lambda *a, **k: pytest.fail("streamed"))
    with pytest.raises(KeyError, match=f"library/{LIBRARY_KEY}/{rel}"):
        client.download_library_file(LIBRARY_KEY, rel, tmp_path / "x")
    assert not (tmp_path / "x").exists()


def test_library_methods_raise_on_an_unknown_key(tmp_path: Path) -> None:
    client = _client(tmp_path, _artifact())
    with pytest.raises(EndpointError, match="HTTP 404"):
        client.library_manifest("noSuchKey1999")
    with pytest.raises(EndpointError, match="HTTP 404"):
        client.library_files("noSuchKey1999")
