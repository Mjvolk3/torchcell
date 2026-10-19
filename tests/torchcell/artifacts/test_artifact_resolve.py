# tests/torchcell/artifacts/test_artifact_resolve.py
# [[tests.torchcell.artifacts.test_artifact_resolve]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/artifacts/test_artifact_resolve.py
"""The resolver: local tier, cache, remote source, in that order, and nothing else.

Hermetic: tiers are built under ``tmp_path``; the remote is ``FakeRemote`` (bytes from a
dict) or ``TcDataSource`` over the real tc-data app through the ASGI test client. Every
integrity refusal asserts the exact message, and every unresolvable ref asserts the full
ordered list of sources tried.

2026.10.12 (public read-only tc-data): no key anywhere. Without a ``client`` and without
``TC_DATA_URL`` the default remote is tc-data on ``DEFAULT_URL``; tests of that path stub
the source (``offline_tc_data`` or a recording ``manifest``) so no request is made.

2026.10.19 (library tier): ``TcDataSource`` serves the library tier. A library ref to a
supplementary file (``si/plasmids/pBTX001(ante113).gb``) resolves end to end through
tc-data; the paper beside it is withheld by the server, so its ref is a miss.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from tests.torchcell.artifacts._fakes import (
    FakeRemote,
    literature_manifest,
    offline_tc_data,
    ref_for,
    sha,
    write_genomes,
    write_tier,
)
from torchcell.artifacts import (
    ArtifactIntegrityError,
    ArtifactRef,
    ArtifactUnresolvableError,
    ResolvedArtifact,
    TcDataSource,
    check,
    materialize,
    resolve,
)
from torchcell.artifacts.resolve import RemoteEndpointError, RemoteMissError, cache_path
from torchcell.datasets.artifact import ArtifactIndex
from torchcell.datasets.server import DataServerConfig, create_app
from torchcell.endpoint_http import DEFAULT_URL
from torchcell.literature.manifest import Manifest
from torchcell.sequence.genome.registry import GenomeManifest

DATA = b">YAL001C\nATGAAA\n"
OTHER = b">YAL001C\nATGTTT\n"
LIBRARY_KEY = "cacheraCRISPAHighthroughputMethod2023"
LIBRARY_FILE = "si/plasmids/pBTX001(ante113).gb"


def _no_env_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ``.env``, no ``TC_DATA_URL``: the default remote is the public tc-data,
    stubbed offline so it misses without a request.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    offline_tc_data(monkeypatch)


def _remote_with(tier: str, key: str, files: dict[str, bytes]) -> FakeRemote:
    return FakeRemote(
        {(tier, key): literature_manifest(key, files)},
        {(tier, key, rel): data for rel, data in files.items()},
    )


# -- source 1: the local tier ---------------------------------------------------------


def test_local_hit_is_verified(tmp_path: Path) -> None:
    write_tier(tmp_path, "objects", "esm2", {"emb/iso.npy": DATA})
    ref = ref_for("objects", "esm2", "emb/iso.npy", DATA)
    remote = FakeRemote({}, {})
    got = resolve(ref, data_root=tmp_path, client=remote)
    assert got == ResolvedArtifact(
        path=tmp_path / "torchcell-objects" / "esm2" / "emb" / "iso.npy",
        source="local",
        verified=True,
    )
    assert remote.manifest_calls == []
    assert materialize(ref, data_root=tmp_path, client=remote) == got.path


def test_local_check_without_materialize_reads_only_the_manifest(
    tmp_path: Path,
) -> None:
    write_tier(tmp_path, "raw", "k", {"f.tsv": DATA})
    ref = ref_for("raw", "k", "f.tsv", DATA)
    # Corrupt the bytes: materialize=False does not hash, so it still resolves.
    (tmp_path / "torchcell-raw" / "k" / "f.tsv").write_bytes(OTHER)
    got = resolve(ref, materialize=False, data_root=tmp_path, client=FakeRemote({}, {}))
    assert got == ResolvedArtifact(
        path=tmp_path / "torchcell-raw" / "k" / "f.tsv", source="local", verified=False
    )
    assert check(ref, data_root=tmp_path, client=FakeRemote({}, {})) is True


def test_local_manifest_listing_a_different_sha256_raises(tmp_path: Path) -> None:
    write_tier(tmp_path, "raw", "k", {"f.tsv": OTHER})
    ref = ref_for("raw", "k", "f.tsv", DATA)
    remote = _remote_with("raw", "k", {"f.tsv": DATA})
    where = f"local raw manifest {tmp_path}/torchcell-raw/k/manifest.json"
    with pytest.raises(ArtifactIntegrityError) as err:
        resolve(ref, data_root=tmp_path, client=remote)
    assert str(err.value) == (
        f"tc://raw/k/f.tsv: {where} lists sha256 {sha(OTHER)}, the ref pins {sha(DATA)}"
    )
    # An integrity disagreement is never routed around to the remote.
    assert remote.manifest_calls == []
    with pytest.raises(ArtifactIntegrityError):
        check(ref, data_root=tmp_path, client=remote)


def test_local_manifest_listing_a_different_size_raises(tmp_path: Path) -> None:
    write_tier(tmp_path, "raw", "k", {"f.tsv": DATA})
    ref = ArtifactRef(
        tier="raw", key="k", path="f.tsv", sha256=sha(DATA), bytes=len(DATA) + 1
    )
    with pytest.raises(
        ArtifactIntegrityError, match=r"lists 16 bytes, the ref pins 17"
    ):
        resolve(ref, data_root=tmp_path, client=FakeRemote({}, {}))


def test_local_bytes_not_matching_the_manifest_raise(tmp_path: Path) -> None:
    write_tier(tmp_path, "raw", "k", {"f.tsv": DATA})
    on_disk = tmp_path / "torchcell-raw" / "k" / "f.tsv"
    on_disk.write_bytes(OTHER)
    ref = ref_for("raw", "k", "f.tsv", DATA)
    with pytest.raises(ArtifactIntegrityError) as err:
        resolve(ref, data_root=tmp_path, client=FakeRemote({}, {}))
    assert str(err.value) == (
        f"tc://raw/k/f.tsv: local file {on_disk} has sha256 {sha(OTHER)}, "
        f"the ref pins {sha(DATA)}"
    )


def test_local_bytes_with_the_right_hash_but_a_wrong_pinned_size_raise(
    tmp_path: Path,
) -> None:
    """The manifest agrees with the bytes; only the ref's own size pin is wrong."""
    directory = tmp_path / "torchcell-raw" / "k"
    write_tier(tmp_path, "raw", "k", {"f.tsv": DATA})
    manifest_file = directory / "manifest.json"
    manifest_file.write_text(
        manifest_file.read_text().replace('"bytes": 16', '"bytes": 9')
    )
    ref = ArtifactRef(tier="raw", key="k", path="f.tsv", sha256=sha(DATA), bytes=9)
    with pytest.raises(ArtifactIntegrityError, match=r"has 16 bytes, the ref pins 9"):
        resolve(ref, data_root=tmp_path, client=FakeRemote({}, {}))


def test_the_genomes_tier_resolves_through_the_registry_manifest(
    tmp_path: Path,
) -> None:
    write_genomes(tmp_path, "peter2018_1011_assemblies", {"genes.tar.gz": DATA})
    ref = ref_for(
        "genomes",
        "peter2018_1011_assemblies",
        "genes.tar.gz",
        DATA,
        member="YAL001C.fasta#AAA_1",
    )
    got = resolve(ref, data_root=tmp_path, client=FakeRemote({}, {}))
    assert got == ResolvedArtifact(
        path=tmp_path
        / "torchcell-genomes"
        / "peter2018_1011_assemblies"
        / "genes.tar.gz",
        source="local",
        verified=True,
    )


# -- sources 2 and 3: the cache and the remote ----------------------------------------


def test_local_miss_then_remote_hit_lands_verified_in_the_cache(tmp_path: Path) -> None:
    ref = ref_for("objects", "esm2", "emb/iso.npy", DATA)
    remote = _remote_with("objects", "esm2", {"emb/iso.npy": DATA})
    got = resolve(ref, data_root=tmp_path, client=remote)
    cached = tmp_path / "artifact-cache" / sha(DATA) / "iso.npy"
    assert got == ResolvedArtifact(path=cached, source="remote", verified=True)
    assert cached.read_bytes() == DATA
    assert sorted(p.name for p in cached.parent.iterdir()) == ["iso.npy"]
    assert cache_path(ref, tmp_path) == cached
    assert [d[:3] for d in remote.downloads] == [("objects", "esm2", "emb/iso.npy")]
    assert remote.downloads[0][3] == cached.with_name("iso.npy.part")


def test_a_cached_file_is_reverified_and_served_without_the_remote(
    tmp_path: Path,
) -> None:
    ref = ref_for("objects", "esm2", "emb/iso.npy", DATA)
    resolve(
        ref,
        data_root=tmp_path,
        client=_remote_with("objects", "esm2", {"emb/iso.npy": DATA}),
    )
    remote = FakeRemote({}, {})
    got = resolve(ref, data_root=tmp_path, client=remote)
    assert got == ResolvedArtifact(
        path=cache_path(ref, tmp_path), source="remote", verified=True
    )
    assert remote.manifest_calls == []


def test_a_corrupted_cache_file_raises(tmp_path: Path) -> None:
    ref = ref_for("objects", "esm2", "iso.npy", DATA)
    cached = cache_path(ref, tmp_path)
    cached.parent.mkdir(parents=True)
    cached.write_bytes(OTHER)
    with pytest.raises(ArtifactIntegrityError) as err:
        resolve(ref, data_root=tmp_path, client=FakeRemote({}, {}))
    assert str(err.value) == (
        f"tc://objects/esm2/iso.npy: cached file {cached} has sha256 {sha(OTHER)}, "
        f"the ref pins {sha(DATA)}"
    )


def test_remote_bytes_that_hash_wrong_raise_and_leave_no_cached_file(
    tmp_path: Path,
) -> None:
    ref = ref_for("objects", "esm2", "iso.npy", DATA)
    remote = FakeRemote(
        {("objects", "esm2"): literature_manifest("esm2", {"iso.npy": DATA})},
        {("objects", "esm2", "iso.npy"): OTHER},
    )
    with pytest.raises(ArtifactIntegrityError) as err:
        resolve(ref, data_root=tmp_path, client=remote)
    assert str(err.value) == (
        f"tc://objects/esm2/iso.npy: fake-remote served 16 bytes with sha256 "
        f"{sha(OTHER)}, the ref pins sha256 {sha(DATA)} and 16 bytes; the download "
        "was discarded"
    )
    assert not (tmp_path / "artifact-cache" / sha(DATA)).exists()
    assert sorted((tmp_path / "artifact-cache").iterdir()) == []


def test_a_discarded_download_keeps_an_unrelated_file_in_the_sha_directory(
    tmp_path: Path,
) -> None:
    """Only the ``.part`` is removed; the directory goes only when it is empty."""
    ref = ArtifactRef(tier="objects", key="esm2", path="iso.npy", sha256=sha(DATA))
    sibling = cache_path(ref, tmp_path).with_name("other.npy")
    sibling.parent.mkdir(parents=True)
    sibling.write_bytes(DATA)
    remote = FakeRemote(
        {("objects", "esm2"): literature_manifest("esm2", {"iso.npy": DATA})},
        {("objects", "esm2", "iso.npy"): OTHER},
    )
    with pytest.raises(ArtifactIntegrityError, match=r"the ref pins sha256 \w+; the"):
        resolve(ref, data_root=tmp_path, client=remote)
    assert sorted(p.name for p in sibling.parent.iterdir()) == ["other.npy"]


def test_remote_manifest_listing_a_different_sha256_raises(tmp_path: Path) -> None:
    ref = ref_for("objects", "esm2", "iso.npy", DATA)
    remote = _remote_with("objects", "esm2", {"iso.npy": OTHER})
    with pytest.raises(ArtifactIntegrityError) as err:
        resolve(ref, materialize=False, data_root=tmp_path, client=remote)
    assert str(err.value) == (
        f"tc://objects/esm2/iso.npy: fake-remote manifest lists sha256 {sha(OTHER)}, "
        f"the ref pins {sha(DATA)}"
    )
    assert remote.downloads == []


def test_materialize_false_never_downloads(tmp_path: Path) -> None:
    ref = ref_for("objects", "esm2", "iso.npy", DATA)
    remote = _remote_with("objects", "esm2", {"iso.npy": DATA})
    got = resolve(ref, materialize=False, data_root=tmp_path, client=remote)
    assert got == ResolvedArtifact(
        path=cache_path(ref, tmp_path), source="remote", verified=False
    )
    assert remote.manifest_calls == [("objects", "esm2")]
    assert remote.downloads == []
    assert not (tmp_path / "artifact-cache").exists()
    assert check(ref, data_root=tmp_path, client=remote) is True
    assert remote.downloads == []


def test_unresolvable_names_every_source_tried_in_order(tmp_path: Path) -> None:
    ref = ref_for("objects", "esm2", "iso.npy", DATA)
    remote = FakeRemote({}, {})
    with pytest.raises(ArtifactUnresolvableError) as err:
        resolve(ref, data_root=tmp_path, client=remote)
    assert str(err.value) == (
        f"tc://objects/esm2/iso.npy (sha256 {sha(DATA)}) did not resolve; sources "
        f"tried: (1) local objects manifest {tmp_path}/torchcell-objects/esm2/"
        "manifest.json: absent; "
        f"(2) artifact cache: {tmp_path}/artifact-cache/{sha(DATA)}/iso.npy absent; "
        "(3) fake-remote manifest: no key objects/esm2"
    )
    assert check(ref, data_root=tmp_path, client=remote) is False


def test_unresolvable_reasons_for_unlisted_and_missing_files(tmp_path: Path) -> None:
    """Local manifest lists another file; remote manifest omits the path."""
    write_tier(tmp_path, "raw", "k", {"other.tsv": DATA})
    ref = ref_for("raw", "k", "f.tsv", DATA)
    remote = _remote_with("raw", "k", {"other.tsv": DATA})
    with pytest.raises(ArtifactUnresolvableError) as err:
        resolve(ref, materialize=False, data_root=tmp_path, client=remote)
    assert str(err.value) == (
        f"tc://raw/k/f.tsv (sha256 {sha(DATA)}) did not resolve; sources tried: "
        f"(1) local raw manifest {tmp_path}/torchcell-raw/k/manifest.json: does not "
        "list 'f.tsv'; (2) fake-remote manifest: does not list 'f.tsv'"
    )


def test_a_listed_file_missing_from_disk_goes_on_to_the_remote(tmp_path: Path) -> None:
    write_tier(tmp_path, "raw", "k", {"f.tsv": DATA})
    on_disk = tmp_path / "torchcell-raw" / "k" / "f.tsv"
    on_disk.unlink()
    ref = ref_for("raw", "k", "f.tsv", DATA)
    remote = FakeRemote({("raw", "k"): literature_manifest("k", {"f.tsv": DATA})}, {})
    with pytest.raises(ArtifactUnresolvableError) as err:
        resolve(ref, data_root=tmp_path, client=remote)
    assert str(err.value).endswith(
        f"(1) local raw manifest {tmp_path}/torchcell-raw/k/manifest.json: lists "
        f"'f.tsv' but {on_disk} is not on disk; (2) artifact cache: "
        f"{tmp_path}/artifact-cache/{sha(DATA)}/f.tsv absent; (3) fake-remote "
        "download: no file raw/k/f.tsv"
    )
    # The sha directory made for the download is removed again.
    assert sorted((tmp_path / "artifact-cache").iterdir()) == []


def test_without_a_client_or_tc_data_url_the_default_source_is_the_public_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """There is always a remote: unset ``TC_DATA_URL`` builds tc-data on ``DEFAULT_URL``
    (its manifest call is stubbed, so nothing touches the network).
    """
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    built: list[TcDataSource] = []

    def fake_manifest(self: TcDataSource, tier: str, key: str) -> GenomeManifest:
        built.append(self)
        raise RemoteMissError("stubbed")

    monkeypatch.setattr(TcDataSource, "manifest", fake_manifest)
    ref = ref_for("raw", "k", "f.tsv", DATA)
    with pytest.raises(ArtifactUnresolvableError) as err:
        resolve(ref, materialize=False, data_root=tmp_path)
    assert str(err.value).endswith(f"(2) tc-data {DEFAULT_URL} manifest: stubbed")
    assert len(built) == 1
    assert built[0].url == DEFAULT_URL


def test_without_a_client_tc_data_is_built_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("TC_DATA_URL", "http://tc-data.invalid:8724/")
    built: list[TcDataSource] = []

    def fake_manifest(self: TcDataSource, tier: str, key: str) -> GenomeManifest:
        built.append(self)
        raise RemoteMissError("stubbed")

    monkeypatch.setattr(TcDataSource, "manifest", fake_manifest)
    ref = ref_for("raw", "k", "f.tsv", DATA)
    with pytest.raises(
        ArtifactUnresolvableError,
        match=r"\(2\) tc-data http://tc-data.invalid:8724 manifest: stubbed$",
    ):
        resolve(ref, materialize=False, data_root=tmp_path)
    assert [repr(s) for s in built] == ["tc-data http://tc-data.invalid:8724"]


def test_resolve_reads_data_root_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_env_remote(monkeypatch)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    write_tier(tmp_path, "raw", "k", {"f.tsv": DATA})
    got = resolve(ref_for("raw", "k", "f.tsv", DATA))
    assert got.path == tmp_path / "torchcell-raw" / "k" / "f.tsv"


# -- TcDataSource ---------------------------------------------------------------------


def _tc_data(tmp_path: Path, raise_server_exceptions: bool = True) -> TcDataSource:
    """tc-data over the real app: one raw key and one library key, via the ASGI client.

    ``raise_server_exceptions=False`` lets a server fault arrive as its HTTP 500.
    """
    raw_root = tmp_path / "served" / "torchcell-raw"
    write_tier(tmp_path / "served", "raw", "kemmeren2014", {"data/expr.tsv": DATA})
    write_tier(
        tmp_path / "served",
        "library",
        LIBRARY_KEY,
        {LIBRARY_FILE: DATA, "paper.pdf": b"%PDF-1.7 paper"},
    )
    store = tmp_path / "store"
    store.mkdir()
    ArtifactIndex(generated_at="t0", artifacts=[]).save(store / "index.json")
    config = DataServerConfig(
        store_root=store,
        raw_root=raw_root,
        genomes_root=tmp_path / "served-genomes",
        objects_root=tmp_path / "served-objects",
        library_root=tmp_path / "served" / "torchcell-library",
    )
    http = TestClient(
        create_app(config), raise_server_exceptions=raise_server_exceptions
    )
    return TcDataSource("http://testserver/", http=http)


def test_tc_data_source_resolves_the_raw_tier_end_to_end(tmp_path: Path) -> None:
    source = _tc_data(tmp_path)
    assert repr(source) == "tc-data http://testserver"
    client_root = tmp_path / "client"
    ref = ref_for("raw", "kemmeren2014", "data/expr.tsv", DATA)
    assert check(ref, data_root=client_root, client=source) is True
    got = resolve(ref, data_root=client_root, client=source)
    assert got == ResolvedArtifact(
        path=client_root / "artifact-cache" / sha(DATA) / "expr.tsv",
        source="remote",
        verified=True,
    )
    assert got.path.read_bytes() == DATA


def test_tc_data_source_turns_404s_into_misses(tmp_path: Path) -> None:
    source = _tc_data(tmp_path)
    with pytest.raises(RemoteMissError, match=r"/raw/absent2020/manifest: HTTP 404"):
        source.manifest("raw", "absent2020")
    with pytest.raises(RemoteMissError, match=r"/artifact/data/none.tsv: HTTP 404"):
        source.download("raw", "kemmeren2014", "data/none.tsv", tmp_path / "x")
    with pytest.raises(
        RemoteMissError, match=r"/library/absent2020/manifest: HTTP 404"
    ):
        source.manifest("library", "absent2020")


def test_tc_data_source_serves_the_library_tier_without_the_paper(
    tmp_path: Path,
) -> None:
    """The supplementary file resolves through tc-data; the paper is listed in the
    on-disk manifest yet absent from the public one, so its ref has no remote source.
    """
    assert TcDataSource.SERVED_TIERS == ("raw", "genomes", "objects", "library")
    source = _tc_data(tmp_path)
    on_disk = tmp_path / "served" / "torchcell-library" / LIBRARY_KEY / "manifest.json"
    listed = Manifest.model_validate_json(on_disk.read_text()).files
    roles = {f.path: f.role for f in listed}
    assert roles == {LIBRARY_FILE: "si_data", "paper.pdf": "paper_pdf"}
    public = source.manifest("library", LIBRARY_KEY)
    assert [f.path for f in public.files] == [LIBRARY_FILE]
    client_root = tmp_path / "client"
    ref = ref_for("library", LIBRARY_KEY, LIBRARY_FILE, DATA)
    assert check(ref, data_root=client_root, client=source) is True
    got = resolve(ref, data_root=client_root, client=source)
    assert got == ResolvedArtifact(
        path=client_root / "artifact-cache" / sha(DATA) / "pBTX001(ante113).gb",
        source="remote",
        verified=True,
    )
    assert got.path.read_bytes() == DATA
    paper = ref_for("library", LIBRARY_KEY, "paper.pdf", b"%PDF-1.7 paper")
    assert check(paper, data_root=client_root, client=source) is False
    with pytest.raises(RemoteMissError, match=r"/artifact/paper.pdf: HTTP 404"):
        source.download("library", LIBRARY_KEY, "paper.pdf", tmp_path / "x")


def test_tc_data_source_raises_on_a_server_error(tmp_path: Path) -> None:
    """A corrupt ``manifest.json`` is a 500 from tc-data on both routes, which is an
    endpoint error, not a miss.
    """
    source = _tc_data(tmp_path, raise_server_exceptions=False)
    manifest = tmp_path / "served" / "torchcell-raw" / "kemmeren2014" / "manifest.json"
    manifest.write_text("{not json", encoding="utf-8")
    with pytest.raises(RemoteEndpointError, match=r"/manifest: HTTP 500$"):
        source.manifest("raw", "kemmeren2014")
    with pytest.raises(RemoteEndpointError, match="expected HTTP 200, got 500"):
        source.download("raw", "kemmeren2014", "data/expr.tsv", tmp_path / "x")


class _Response:
    def __init__(self, status: int, content: bytes) -> None:
        self.status_code = status
        self.content = content
        self.request = httpx.Request("GET", "http://fake")

    def raise_for_status(self) -> None:
        assert self.status_code == 200

    def iter_bytes(self, chunk: int) -> Iterator[bytes]:
        yield self.content


class _UrlRecorder:
    """An ``HttpClient`` serving fixed bodies by URL; records each URL requested."""

    def __init__(self, bodies: dict[str, bytes]) -> None:
        self.bodies = bodies
        self.urls: list[str] = []

    def get(self, url: str) -> Any:
        self.urls.append(url)
        return _Response(200, self.bodies[url])

    @contextmanager
    def stream(
        self, method: str, url: str, *, headers: Mapping[str, str] | None = None
    ) -> Iterator[_Response]:
        self.urls.append(url)
        yield _Response(200, self.bodies[url])


def test_tc_data_source_uses_the_genomes_and_objects_url_shapes(tmp_path: Path) -> None:
    genomes = GenomeManifest(
        assembly_set="set_v1",
        organism="Saccharomyces cerevisiae",
        strain_or_population="1,011 isolates",
        source="test",
        release="v1",
        files=literature_manifest("x", {"genes.tar.gz": DATA}).files,
        provenance_complete=False,
        created_at="t0",
    )
    objects = literature_manifest("esm2", {"emb/iso.npy": DATA})
    http = _UrlRecorder(
        {
            "http://tc/genomes/set_v1/manifest": genomes.model_dump_json().encode(),
            "http://tc/genomes/set_v1/artifact/genes.tar.gz": DATA,
            "http://tc/objects/esm2/manifest": objects.model_dump_json().encode(),
            "http://tc/objects/esm2/artifact/emb/iso.npy": DATA,
        }
    )
    source = TcDataSource("http://tc", http=http)
    got = resolve(
        ref_for("genomes", "set_v1", "genes.tar.gz", DATA),
        data_root=tmp_path,
        client=source,
    )
    assert got.source == "remote"
    assert source.manifest("genomes", "set_v1") == genomes
    assert source.manifest("objects", "esm2") == objects
    source.download("objects", "esm2", "emb/iso.npy", tmp_path / "iso.npy")
    assert (tmp_path / "iso.npy").read_bytes() == DATA
    assert http.urls == [
        "http://tc/genomes/set_v1/manifest",
        "http://tc/genomes/set_v1/artifact/genes.tar.gz",
        "http://tc/genomes/set_v1/manifest",
        "http://tc/objects/esm2/manifest",
        "http://tc/objects/esm2/artifact/emb/iso.npy",
    ]


def test_tc_data_source_from_env_defaults_to_the_public_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    assert repr(TcDataSource.from_env()) == f"tc-data {DEFAULT_URL}"
    assert TcDataSource().url == DEFAULT_URL
    monkeypatch.setenv("TC_DATA_URL", "http://a/")
    assert repr(TcDataSource.from_env()) == "tc-data http://a"
