# torchcell/datasets/client.py
# [[torchcell.datasets.client]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/client.py
# Test file: tests/torchcell/datasets/test_client.py

"""Client for the ``tc-data`` endpoint: pick, download, verify and unpack an artifact.

``DatasetClient`` talks to ``torchcell.datasets.server`` over httpx with the
``X-API-Key`` header. ``select`` applies the compatibility rule (the newest
``supported`` row of a slug whose ``torchcell_version`` shares the installed
``major.minor``); ``download`` streams the archive to ``<dest>.part``, resumes with a
``Range`` request when a partial file exists, verifies the sha256 against the index row
and the ``X-Artifact-SHA256`` header, and only then renames it into place; a mismatch
removes the bad file and raises. ``unpack_artifact`` extracts ``processed/`` and
``preprocess/`` into a dataset root with the ``data`` tar filter (no traversal).

The raw, genomes and objects tiers are file trees gated by a per-key ``manifest.json``:
``genomes`` / ``objects`` list the keys, ``*_manifest`` and ``*_files`` return the
manifest and its file rows, and ``download_raw_file`` / ``download_genome_file`` /
``download_object_file`` stream one listed file through the same resumable path as ``download``. The expected
sha256 and size come from the key's ``/files`` row; the server's ``X-Artifact-SHA256``
must equal that row and the completed bytes must hash to it, or the partial file is
removed and :class:`ArtifactIntegrityError` is raised.

Environment: ``TC_DATA_URL`` and ``TC_DATA_API_KEY`` (``DatasetClient.from_env``).
"""

from __future__ import annotations

import hashlib
import os
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, Self
from urllib.parse import quote

import httpx

from torchcell import __version__
from torchcell.api_keys import API_KEY_HEADER
from torchcell.datasets.artifact import (
    ArtifactIndex,
    DatasetArtifact,
    ManifestFileListing,
)
from torchcell.literature.manifest import Manifest
from torchcell.sequence.genome.registry import GenomeManifest

URL_VAR = "TC_DATA_URL"
API_KEY_VAR = "TC_DATA_API_KEY"
PART_SUFFIX = ".part"
CHUNK = 1 << 20
DEFAULT_TIMEOUT = 120.0


#: The manifest-gated tiers the client downloads single files from.
FileTierName = Literal["raw", "genomes", "objects"]


class ArtifactIntegrityError(RuntimeError):
    """Downloaded bytes, or the server's hash header, disagree with the recorded sha256."""


class HttpClient(Protocol):
    """The two ``httpx.Client`` calls the client makes; a test client satisfies it too."""

    def get(self, url: str, *, headers: Mapping[str, str]) -> Any:
        """A buffered GET returning a response with ``raise_for_status`` and ``content``."""
        ...

    def stream(self, method: str, url: str, *, headers: Mapping[str, str]) -> Any:
        """A context manager yielding a response with ``iter_bytes`` and ``headers``."""
        ...


class DatasetClient:
    """Keyed HTTP client for one ``tc-data`` endpoint."""

    def __init__(
        self,
        url: str,
        api_key: str,
        http: HttpClient | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """``url`` is the endpoint base (no trailing slash needed); ``http`` injects a client."""
        self.url = url.rstrip("/")
        self._headers = {API_KEY_HEADER: api_key}
        self._http: HttpClient = (
            http if http is not None else httpx.Client(timeout=timeout)
        )

    @classmethod
    def from_env(cls, http: HttpClient | None = None) -> Self:
        """Build from ``TC_DATA_URL`` and ``TC_DATA_API_KEY`` (``KeyError`` if unset)."""
        return cls(os.environ[URL_VAR], os.environ[API_KEY_VAR], http=http)

    def _get(self, path: str) -> Any:
        response = self._http.get(f"{self.url}{path}", headers=self._headers)
        response.raise_for_status()
        return response

    def index(self) -> ArtifactIndex:
        """``GET /datasets``: the whole artifact index."""
        return ArtifactIndex.model_validate_json(self._get("/datasets").content)

    def artifacts(self, slug: str) -> list[DatasetArtifact]:
        """``GET /datasets/{slug}``: every packaged version of one slug."""
        payload = self._get(f"/datasets/{slug}").json()
        return [DatasetArtifact.model_validate(row) for row in payload]

    def select(
        self, slug: str, torchcell_version: str = __version__
    ) -> DatasetArtifact | None:
        """The newest ``supported`` artifact of ``slug`` on ``torchcell_version``'s line.

        None when the index has nothing compatible; callers decide whether to stop.
        """
        return self.index().select(slug, torchcell_version)

    def download(
        self, artifact: DatasetArtifact, dest: Path, verify: bool = True
    ) -> Path:
        """Stream ``artifact`` to the file ``dest``, resuming a ``<dest>.part`` if present.

        With ``verify`` the completed bytes must hash to ``artifact.archive_sha256``
        (and the server's ``X-Artifact-SHA256`` must agree); a mismatch deletes the
        partial file and raises :class:`ArtifactIntegrityError`.
        """
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + PART_SUFFIX)
        offset = part.stat().st_size if part.is_file() else 0
        if offset < artifact.archive_bytes:
            self._stream_into(
                f"/datasets/{artifact.slug}/{artifact.archive}",
                part,
                offset,
                expected_sha256=artifact.archive_sha256,
                label=artifact.rel_path,
                recorded_in="index",
            )
        if verify:
            _verify_part(part, artifact.archive_sha256, artifact.rel_path, "index")
        os.replace(part, dest)
        return dest

    def raw_manifest(self, citation_key: str) -> Manifest:
        """``GET /raw/{citation_key}/manifest``: the key's literature ``Manifest``."""
        content = self._get(f"/raw/{_segment(citation_key)}/manifest").content
        return Manifest.model_validate_json(content)

    def raw_files(self, citation_key: str) -> list[ManifestFileListing]:
        """``GET /raw/{citation_key}/files``: path, role, bytes, sha256 per file."""
        return self._tier_files("raw", citation_key)

    def download_raw_file(self, citation_key: str, rel_path: str, dest: Path) -> Path:
        """Stream one raw-mirror file to ``dest``, verified; see :meth:`_download_listed`."""
        return self._download_listed("raw", citation_key, rel_path, dest)

    def genomes(self) -> list[str]:
        """``GET /genomes``: assembly sets in the genomes tier that carry a manifest."""
        names: list[str] = self._get("/genomes").json()["assembly_sets"]
        return names

    def genome_manifest(self, assembly_set: str) -> GenomeManifest:
        """``GET /genomes/{assembly_set}/manifest``: the set's ``GenomeManifest``."""
        content = self._get(f"/genomes/{_segment(assembly_set)}/manifest").content
        return GenomeManifest.model_validate_json(content)

    def genome_files(self, assembly_set: str) -> list[ManifestFileListing]:
        """``GET /genomes/{assembly_set}/files``: path, role, bytes, sha256 per file."""
        return self._tier_files("genomes", assembly_set)

    def download_genome_file(
        self, assembly_set: str, rel_path: str, dest: Path
    ) -> Path:
        """Stream one genomes-tier file to ``dest``, verified; see :meth:`_download_listed`."""
        return self._download_listed("genomes", assembly_set, rel_path, dest)

    def objects(self) -> list[str]:
        """``GET /objects``: object keys in the objects tier that carry a manifest."""
        names: list[str] = self._get("/objects").json()["object_keys"]
        return names

    def object_manifest(self, object_key: str) -> Manifest:
        """``GET /objects/{object_key}/manifest``: the key's literature ``Manifest``."""
        content = self._get(f"/objects/{_segment(object_key)}/manifest").content
        return Manifest.model_validate_json(content)

    def object_files(self, object_key: str) -> list[ManifestFileListing]:
        """``GET /objects/{object_key}/files``: path, role, bytes, sha256 per file."""
        return self._tier_files("objects", object_key)

    def download_object_file(self, object_key: str, rel_path: str, dest: Path) -> Path:
        """Stream one objects-tier file to ``dest``, verified; see :meth:`_download_listed`."""
        return self._download_listed("objects", object_key, rel_path, dest)

    def _tier_files(self, tier: FileTierName, key: str) -> list[ManifestFileListing]:
        payload = self._get(f"/{tier}/{_segment(key)}/files").json()
        return [ManifestFileListing.model_validate(row) for row in payload]

    def _download_listed(
        self, tier: FileTierName, key: str, rel_path: str, dest: Path
    ) -> Path:
        """Stream a manifest-listed file to ``dest``, resuming a ``<dest>.part``.

        The key's ``/files`` row gives the expected size and sha256; a path the
        manifest does not list raises ``KeyError`` before any byte is requested. The
        server's ``X-Artifact-SHA256`` must equal the row and the completed bytes must
        hash to it; on a byte mismatch the partial file is removed. Either failure
        raises :class:`ArtifactIntegrityError`.
        """
        record = next(
            (f for f in self._tier_files(tier, key) if f.path == rel_path), None
        )
        label = f"{tier}/{key}/{rel_path}"
        if record is None:
            raise KeyError(f"{label} is not in the manifest")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + PART_SUFFIX)
        offset = part.stat().st_size if part.is_file() else 0
        if offset < record.bytes:
            self._stream_into(
                f"/{tier}/{_segment(key)}/artifact/{quote(rel_path, safe='/')}",
                part,
                offset,
                expected_sha256=record.sha256,
                label=label,
                recorded_in="manifest",
            )
        _verify_part(part, record.sha256, label, "manifest")
        os.replace(part, dest)
        return dest

    def _stream_into(
        self,
        path: str,
        part: Path,
        offset: int,
        *,
        expected_sha256: str,
        label: str,
        recorded_in: str,
    ) -> None:
        """GET ``path`` into ``part`` from ``offset`` (a ``Range`` request when > 0).

        The status must be 206 for a resume and 200 otherwise, and the server's
        ``X-Artifact-SHA256`` must equal ``expected_sha256`` before a byte is written.
        """
        headers = dict(self._headers)
        if offset:
            headers["Range"] = f"bytes={offset}-"
        expected_status = 206 if offset else 200
        url = f"{self.url}{path}"
        with self._http.stream("GET", url, headers=headers) as response:
            if response.status_code != expected_status:
                raise httpx.HTTPStatusError(
                    f"{url}: expected HTTP {expected_status}, got "
                    f"{response.status_code}",
                    request=response.request,
                    response=response,
                )
            served = response.headers.get("X-Artifact-SHA256")
            if served != expected_sha256:
                raise ArtifactIntegrityError(
                    f"{label}: server X-Artifact-SHA256 {served} != {recorded_in} "
                    f"{expected_sha256}"
                )
            with part.open("ab" if offset else "wb") as handle:
                for chunk in response.iter_bytes(CHUNK):
                    handle.write(chunk)


def _segment(key: str) -> str:
    """One URL path segment: a key never carries a ``/``, so it is fully quoted."""
    return quote(key, safe="")


def _verify_part(
    part: Path, expected_sha256: str, label: str, recorded_in: str
) -> None:
    """Hash the completed ``part``; on a mismatch remove it and raise."""
    digest = _sha256_of_file(part)
    if digest != expected_sha256:
        part.unlink()
        raise ArtifactIntegrityError(
            f"{label}: sha256 {digest} != {recorded_in} {expected_sha256}; "
            "partial file removed"
        )


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_artifact(archive: Path, dataset_root: Path) -> list[str]:
    """Extract an archive's ``processed/`` and ``preprocess/`` into ``dataset_root``.

    Returns the member names extracted. Uses the ``data`` tar filter, which rejects
    absolute paths, ``..`` components and links pointing outside the root.
    """
    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:xz") as tar:
        names = tar.getnames()
        tar.extractall(dataset_root, filter="data")
    return names
