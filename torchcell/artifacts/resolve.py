# torchcell/artifacts/resolve.py
# [[torchcell.artifacts.resolve]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/artifacts/resolve.py
# Test file: tests/torchcell/artifacts/test_artifact_resolve.py

"""The one resolver from an ``ArtifactRef`` to verified bytes on disk.

Sources, in order, and no others:

1. The local tier root, read through that tier's manifest. The manifest must list
   ``ref.path`` with ``ref.sha256``; with ``materialize`` the bytes on disk must hash to
   it too.
2. With ``materialize``, the artifact cache ``$DATA_ROOT/artifact-cache/<sha256>/
   <basename>``; a cached file is re-hashed before it is returned.
3. A remote source (tc-data over HTTP by default, ``TC_DATA_URL`` + ``TC_DATA_API_KEY``):
   its manifest must list the path with ``ref.sha256``; with ``materialize`` the file is
   downloaded to ``<cache file>.part``, hashed, and only then renamed into the cache.

A sha256 disagreement anywhere raises ``ArtifactIntegrityError`` at once (it is never
treated as a miss to route around). A ref that no source holds raises
``ArtifactUnresolvableError`` naming every source tried and why each failed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Protocol, Self

import httpx

from torchcell.api_keys import API_KEY_HEADER
from torchcell.artifacts.ref import ArtifactRef, Tier
from torchcell.artifacts.tiers import (
    TierManifest,
    find_record,
    key_dir,
    manifest_path,
    manifest_record,
    resolve_data_root,
)
from torchcell.datamodels.pydant import ModelStrict
from torchcell.datasets.client import (
    API_KEY_VAR,
    CHUNK,
    DEFAULT_TIMEOUT,
    URL_VAR,
    HttpClient,
)
from torchcell.literature.manifest import ArtifactRecord, Manifest, sha256_file
from torchcell.sequence.genome.registry import GenomeManifest

CACHE_DIR = "artifact-cache"
PART_SUFFIX = ".part"


class ArtifactUnresolvableError(LookupError):
    """No source holds the ref; the message names every source tried and why."""


class ArtifactIntegrityError(RuntimeError):
    """A manifest or the bytes disagree with the sha256 (or size) the ref pins."""


class RemoteMissError(LookupError):
    """A remote source does not hold the key or the path (an HTTP 404, say)."""


class ResolvedArtifact(ModelStrict):
    """Where a ref resolved: the file path, which source held it, whether it was hashed.

    ``verified`` is True only when the bytes at ``path`` were hashed to ``ref.sha256``
    in this call. A ``materialize=False`` resolve verifies the manifest entry, not the
    bytes, and for a remote hit ``path`` is the cache file the bytes would land in.
    """

    path: Path
    source: Literal["local", "remote"]
    verified: bool


class RemoteSource(Protocol):
    """A remote holder of tier files: a manifest per key and the files it lists."""

    def manifest(self, tier: Tier, key: str) -> TierManifest:
        """The key's manifest; ``RemoteMissError`` when the source has no such key."""
        ...

    def download(self, tier: Tier, key: str, path: str, dest: Path) -> None:
        """Write the bytes of ``path`` to ``dest``; ``RemoteMissError`` when absent."""
        ...


class TcDataSource:
    """``RemoteSource`` over the tc-data endpoint for the raw, genomes and objects tiers.

    URL shapes: ``/{tier}/{key}/manifest`` and ``/{tier}/{key}/artifact/{path}``. The
    raw tier is served today; ``/genomes`` and ``/objects`` arrive with phase 2 of the
    artifact tier. The library tier is served by tc-lit, not tc-data, so asking this
    source for it is a miss.
    """

    SERVED_TIERS: tuple[str, ...] = ("raw", "genomes", "objects")

    def __init__(
        self,
        url: str,
        api_key: str,
        http: HttpClient | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """``url`` is the endpoint base; ``http`` injects a client (tests)."""
        self.url = url.rstrip("/")
        self._headers = {API_KEY_HEADER: api_key}
        self._http: HttpClient = (
            http if http is not None else httpx.Client(timeout=timeout)
        )

    @classmethod
    def from_env(cls, http: HttpClient | None = None) -> Self:
        """Build from ``TC_DATA_URL`` and ``TC_DATA_API_KEY`` (``KeyError`` if unset)."""
        return cls(os.environ[URL_VAR], os.environ[API_KEY_VAR], http=http)

    def __repr__(self) -> str:
        """``tc-data <url>``, the form error messages name the source by."""
        return f"tc-data {self.url}"

    def _served(self, tier: Tier) -> None:
        if tier not in self.SERVED_TIERS:
            raise RemoteMissError(
                f"tc-data does not serve the {tier!r} tier (serves {self.SERVED_TIERS})"
            )

    def manifest(self, tier: Tier, key: str) -> TierManifest:
        """``GET /{tier}/{key}/manifest``; a 404 is a ``RemoteMissError``."""
        self._served(tier)
        url = f"{self.url}/{tier}/{key}/manifest"
        response = self._http.get(url, headers=self._headers)
        if response.status_code == 404:
            raise RemoteMissError(f"{url}: HTTP 404")
        response.raise_for_status()
        if tier == "genomes":
            return GenomeManifest.model_validate_json(response.content)
        return Manifest.model_validate_json(response.content)

    def download(self, tier: Tier, key: str, path: str, dest: Path) -> None:
        """Stream ``GET /{tier}/{key}/artifact/{path}`` into ``dest``; 404 is a miss."""
        self._served(tier)
        url = f"{self.url}/{tier}/{key}/artifact/{path}"
        with self._http.stream("GET", url, headers=self._headers) as response:
            if response.status_code == 404:
                raise RemoteMissError(f"{url}: HTTP 404")
            if response.status_code != 200:
                raise httpx.HTTPStatusError(
                    f"{url}: expected HTTP 200, got {response.status_code}",
                    request=response.request,
                    response=response,
                )
            with dest.open("wb") as handle:
                for chunk in response.iter_bytes(CHUNK):
                    handle.write(chunk)


def cache_path(ref: ArtifactRef, data_root: str | Path | None = None) -> Path:
    """``$DATA_ROOT/artifact-cache/<sha256>/<basename of ref.path>``."""
    root = resolve_data_root(data_root)
    return root / CACHE_DIR / ref.sha256 / Path(ref.path).name


def _check_record(ref: ArtifactRef, record: ArtifactRecord, where: str) -> None:
    """Raise when a manifest record pins other bytes than the ref does."""
    if record.sha256 != ref.sha256:
        raise ArtifactIntegrityError(
            f"{ref}: {where} lists sha256 {record.sha256}, the ref pins {ref.sha256}"
        )
    if ref.bytes is not None and record.bytes != ref.bytes:
        raise ArtifactIntegrityError(
            f"{ref}: {where} lists {record.bytes} bytes, the ref pins {ref.bytes}"
        )


def _check_bytes(ref: ArtifactRef, path: Path, where: str) -> None:
    """Raise when the file at ``path`` does not hash (or size) to what the ref pins."""
    got = sha256_file(path)
    if got != ref.sha256:
        raise ArtifactIntegrityError(
            f"{ref}: {where} {path} has sha256 {got}, the ref pins {ref.sha256}"
        )
    size = path.stat().st_size
    if ref.bytes is not None and size != ref.bytes:
        raise ArtifactIntegrityError(
            f"{ref}: {where} {path} has {size} bytes, the ref pins {ref.bytes}"
        )


def _resolve_local(
    ref: ArtifactRef, root: Path, materialize: bool, tried: list[str]
) -> ResolvedArtifact | None:
    """Source 1, the local tier; None (with the reason appended) on a miss."""
    where = f"local {ref.tier} manifest {manifest_path(ref.tier, ref.key, root)}"
    if not manifest_path(ref.tier, ref.key, root).is_file():
        tried.append(f"{where}: absent")
        return None
    record = manifest_record(ref, root)
    if record is None:
        tried.append(f"{where}: does not list {ref.path!r}")
        return None
    _check_record(ref, record, where)
    path = key_dir(ref.tier, ref.key, root) / record.path
    if not path.is_file():
        tried.append(f"{where}: lists {ref.path!r} but {path} is not on disk")
        return None
    if materialize:
        _check_bytes(ref, path, "local file")
    return ResolvedArtifact(path=path, source="local", verified=materialize)


def _default_source(tried: list[str]) -> RemoteSource | None:
    """tc-data from the environment; None (reason appended) when it is not configured."""
    from dotenv import load_dotenv

    load_dotenv()
    if URL_VAR not in os.environ:
        tried.append(
            f"tc-data: {URL_VAR} is not set, so no remote source is configured"
        )
        return None
    return TcDataSource.from_env()


def _discard(part: Path) -> None:
    """Remove a failed download's ``.part`` and its sha256 directory once empty."""
    part.unlink(missing_ok=True)
    if not any(part.parent.iterdir()):
        part.parent.rmdir()


def _resolve_remote(
    ref: ArtifactRef,
    root: Path,
    materialize: bool,
    source: RemoteSource,
    tried: list[str],
) -> ResolvedArtifact | None:
    """Source 3, the remote manifest then (with ``materialize``) a verified download."""
    try:
        manifest = source.manifest(ref.tier, ref.key)
    except RemoteMissError as miss:
        tried.append(f"{source!r} manifest: {miss}")
        return None
    record = find_record(manifest.files, ref.path)
    if record is None:
        tried.append(f"{source!r} manifest: does not list {ref.path!r}")
        return None
    _check_record(ref, record, f"{source!r} manifest")
    cached = cache_path(ref, root)
    if not materialize:
        return ResolvedArtifact(path=cached, source="remote", verified=False)
    cached.parent.mkdir(parents=True, exist_ok=True)
    part = cached.with_name(cached.name + PART_SUFFIX)
    try:
        source.download(ref.tier, ref.key, ref.path, part)
    except RemoteMissError as miss:
        _discard(part)
        tried.append(f"{source!r} download: {miss}")
        return None
    got = sha256_file(part)
    size = part.stat().st_size
    if got != ref.sha256 or (ref.bytes is not None and size != ref.bytes):
        _discard(part)
        raise ArtifactIntegrityError(
            f"{ref}: {source!r} served {size} bytes with sha256 {got}, the ref pins "
            f"sha256 {ref.sha256}"
            + ("" if ref.bytes is None else f" and {ref.bytes} bytes")
            + "; the download was discarded"
        )
    os.replace(part, cached)
    return ResolvedArtifact(path=cached, source="remote", verified=True)


def resolve(
    ref: ArtifactRef,
    *,
    materialize: bool = True,
    data_root: str | Path | None = None,
    client: RemoteSource | None = None,
) -> ResolvedArtifact:
    """Resolve ``ref`` through the local tier, the cache, then the remote source.

    With ``materialize=False`` only manifests are consulted (local, then remote): the
    answer is whether the ref resolves, nothing is downloaded and nothing is hashed.
    ``client`` overrides the remote source; when None, tc-data is built from
    ``TC_DATA_URL`` / ``TC_DATA_API_KEY`` only if the local tier misses.

    Raises ``ArtifactIntegrityError`` on any sha256 or size disagreement and
    ``ArtifactUnresolvableError`` when no source holds the ref.
    """
    root = resolve_data_root(data_root)
    tried: list[str] = []
    local = _resolve_local(ref, root, materialize, tried)
    if local is not None:
        return local
    if materialize:
        cached = cache_path(ref, root)
        if cached.is_file():
            _check_bytes(ref, cached, "cached file")
            return ResolvedArtifact(path=cached, source="remote", verified=True)
        tried.append(f"artifact cache: {cached} absent")
    source = client if client is not None else _default_source(tried)
    if source is not None:
        remote = _resolve_remote(ref, root, materialize, source, tried)
        if remote is not None:
            return remote
    raise ArtifactUnresolvableError(
        f"{ref} (sha256 {ref.sha256}) did not resolve; sources tried: "
        + "; ".join(f"({i}) {reason}" for i, reason in enumerate(tried, start=1))
    )


def check(
    ref: ArtifactRef,
    *,
    data_root: str | Path | None = None,
    client: RemoteSource | None = None,
) -> bool:
    """Whether ``ref`` resolves (``resolve(..., materialize=False)``), downloading nothing.

    False only for ``ArtifactUnresolvableError``; an integrity disagreement raises.
    """
    try:
        resolve(ref, materialize=False, data_root=data_root, client=client)
    except ArtifactUnresolvableError:
        return False
    return True


def materialize(
    ref: ArtifactRef,
    *,
    data_root: str | Path | None = None,
    client: RemoteSource | None = None,
) -> Path:
    """The verified local path of ``ref``'s file, fetching it into the cache if needed."""
    return resolve(ref, materialize=True, data_root=data_root, client=client).path
