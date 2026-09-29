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

Environment: ``TC_DATA_URL`` and ``TC_DATA_API_KEY`` (``DatasetClient.from_env``).
"""

from __future__ import annotations

import hashlib
import os
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, Self

import httpx

from torchcell import __version__
from torchcell.api_keys import API_KEY_HEADER
from torchcell.datasets.artifact import ArtifactIndex, DatasetArtifact

URL_VAR = "TC_DATA_URL"
API_KEY_VAR = "TC_DATA_API_KEY"
PART_SUFFIX = ".part"
CHUNK = 1 << 20
DEFAULT_TIMEOUT = 120.0


class ArtifactIntegrityError(RuntimeError):
    """A downloaded archive's sha256 does not match the index row."""


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
            self._fetch_range(artifact, part, offset)
        if verify:
            digest = _sha256_of_file(part)
            if digest != artifact.archive_sha256:
                part.unlink()
                raise ArtifactIntegrityError(
                    f"{artifact.rel_path}: sha256 {digest} != index "
                    f"{artifact.archive_sha256}; partial file removed"
                )
        os.replace(part, dest)
        return dest

    def _fetch_range(self, artifact: DatasetArtifact, part: Path, offset: int) -> None:
        headers = dict(self._headers)
        if offset:
            headers["Range"] = f"bytes={offset}-"
        expected_status = 206 if offset else 200
        url = f"{self.url}/datasets/{artifact.slug}/{artifact.archive}"
        with self._http.stream("GET", url, headers=headers) as response:
            if response.status_code != expected_status:
                raise httpx.HTTPStatusError(
                    f"{url}: expected HTTP {expected_status}, got "
                    f"{response.status_code}",
                    request=response.request,
                    response=response,
                )
            served = response.headers.get("X-Artifact-SHA256")
            if served != artifact.archive_sha256:
                raise ArtifactIntegrityError(
                    f"{artifact.rel_path}: server X-Artifact-SHA256 {served} != index "
                    f"{artifact.archive_sha256}"
                )
            with part.open("ab" if offset else "wb") as handle:
                for chunk in response.iter_bytes(CHUNK):
                    handle.write(chunk)


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
