# torchcell/datasets/server.py
# [[torchcell.datasets.server]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/server.py
# Test file: tests/torchcell/datasets/test_datasets_server.py

"""``tc-data``: keyed read-only HTTP endpoint for packaged datasets and the raw mirror.

Serves two directory trees over the network, with the same key scheme as the
literature endpoint (``torchcell.literature.server``): named API keys stored as sha256
hashes, presented in the ``X-API-Key`` header, compared constant-time.

- The artifact store ``$TC_DATA_ROOT``: ``index.json`` (an ``ArtifactIndex``) and one
  ``<slug>/<archive>`` per packaged dataset, written by
  ``scripts/package_dataset_lmdb.py``. The index is authoritative: an archive on disk
  that the index does not list is not served, and every artifact response carries
  ``X-Artifact-SHA256`` from its index row. Archives are large, so
  ``Accept-Ranges: bytes`` is advertised and a ``Range`` header is honored (206 with
  ``Content-Range``), which is what lets the client resume a partial download.
- The raw mirror ``$TC_DATA_RAW_ROOT`` (default ``$DATA_ROOT/torchcell-raw``): one
  directory per citation key with a ``manifest.json`` in the literature ``Manifest``
  shape (path, role, bytes, sha256 per file). Files are served only when the manifest
  lists them, and ``X-Artifact-SHA256`` carries the manifest hash.

Swagger UI is on at ``/docs`` (``/openapi.json`` for the schema); ``/health`` needs no
key. Configuration is environment-driven: ``TC_DATA_ROOT``, ``TC_DATA_RAW_ROOT``,
``TC_DATA_HOST``, ``TC_DATA_PORT`` (8724), and keys from ``TC_DATA_KEYS_FILE`` (JSON
``{name: sha256hex}``, preferred) or ``TC_DATA_API_KEYS`` (``name:key,...``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Self

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from torchcell.api_keys import API_KEY_HEADER, ApiKeys, print_minted_key
from torchcell.datasets.artifact import INDEX_FILENAME, ArtifactIndex, DatasetArtifact
from torchcell.literature.manifest import MANIFEST_FILENAME, Manifest

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

APP_TITLE = "torchcell dataset endpoint"
APP_VERSION = "1.0.0"
DEFAULT_PORT = 8724
RAW_SUBDIR = "torchcell-raw"
KEYS_FILE_VAR = "TC_DATA_KEYS_FILE"
INLINE_KEYS_VAR = "TC_DATA_API_KEYS"
ARCHIVE_MEDIA_TYPE = "application/x-xz"


class DataKeys(ApiKeys):
    """The shared :class:`ApiKeys` model bound to the ``TC_DATA_*`` variables."""

    @classmethod
    def from_pairs(cls, spec: str, env_name: str = INLINE_KEYS_VAR) -> Self:
        """Parse ``name1:key1,name2:key2`` plaintext pairs, hashing each key."""
        return super().from_pairs(spec, env_name=env_name)

    @classmethod
    def from_env(cls) -> Self:
        """Load keys from ``TC_DATA_KEYS_FILE`` (preferred) or ``TC_DATA_API_KEYS``."""
        return cls.from_env_names(KEYS_FILE_VAR, INLINE_KEYS_VAR)


class DataServerConfig(BaseModel):
    """Runtime configuration for the dataset endpoint."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    store_root: Path = Field(description="The artifact store (index.json + archives).")
    raw_root: Path = Field(description="The raw mirror (one dir per citation key).")
    keys: DataKeys
    host: str = "0.0.0.0"
    port: int = DEFAULT_PORT

    @classmethod
    def from_env(cls) -> Self:
        """Build config from ``TC_DATA_*`` (raw root defaults to ``$DATA_ROOT/torchcell-raw``)."""
        store_root = Path(os.environ["TC_DATA_ROOT"])
        if not store_root.is_dir():
            raise FileNotFoundError(f"artifact store does not exist: {store_root}")
        raw_env = os.environ.get("TC_DATA_RAW_ROOT")
        raw_root = (
            Path(raw_env) if raw_env else Path(os.environ["DATA_ROOT"]) / RAW_SUBDIR
        )
        if not raw_root.is_dir():
            raise FileNotFoundError(f"raw mirror does not exist: {raw_root}")
        return cls(
            store_root=store_root,
            raw_root=raw_root,
            keys=DataKeys.from_env(),
            host=os.environ.get("TC_DATA_HOST", "0.0.0.0"),
            port=int(os.environ.get("TC_DATA_PORT", str(DEFAULT_PORT))),
        )


class RawFileListing(BaseModel):
    """One raw-mirror file as its citation key's manifest records it."""

    path: str
    role: str
    bytes: int
    sha256: str


class RawKeyListing(BaseModel):
    """Citation keys present in the raw mirror."""

    citation_keys: list[str]
    count: int


class Health(BaseModel):
    """Liveness summary (no auth)."""

    status: str
    n_artifacts: int
    n_raw_keys: int
    store_root: str
    raw_root: str


_api_key_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def require_key(
    request: Request, api_key: str | None = Depends(_api_key_scheme)
) -> str:
    """Auth dependency: resolve + validate the ``X-API-Key`` header, return its name."""
    config: DataServerConfig = request.app.state.config
    name = config.keys.verify(api_key) if api_key else None
    if name is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing API key",
        )
    return name


def _load_index(config: DataServerConfig) -> ArtifactIndex:
    """The store's ``index.json``, read on every request so a new packaging shows up."""
    path = config.store_root / INDEX_FILENAME
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="no index.json in the artifact store; run "
            "scripts/package_dataset_lmdb.py on the host",
        )
    return ArtifactIndex.load(path)


def _artifact_row(index: ArtifactIndex, slug: str, archive: str) -> DatasetArtifact:
    for row in index.artifacts:
        if row.slug == slug and row.archive == archive:
            return row
    raise HTTPException(status_code=404, detail="artifact not in the index")


def _contained_file(base: Path, rel_path: str, root: Path) -> Path:
    """Resolve ``base / rel_path`` and require it to be a file inside ``root``."""
    target = (base / rel_path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise HTTPException(status_code=400, detail="path traversal rejected")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return target


def _raw_key_dir(config: DataServerConfig, citation_key: str) -> Path:
    """Resolve + containment-check a citation-key directory under the raw mirror.

    An underscore-prefixed name is a service directory, never a citation key
    (``_list_raw_keys`` hides it), so it answers exactly like an absent key. The
    literature server's ``_key_dir`` applies the same rule.
    """
    base = (config.raw_root / citation_key).resolve()
    if (
        citation_key.startswith("_")
        or not base.is_relative_to(config.raw_root.resolve())
        or not base.is_dir()
    ):
        raise HTTPException(status_code=404, detail="unknown citation key")
    return base


def _raw_manifest(base: Path) -> Manifest:
    """A key's ``manifest.json``; a key without one is not served (404)."""
    path = base / MANIFEST_FILENAME
    if not path.is_file():
        raise HTTPException(
            status_code=404, detail="no manifest.json for this citation key"
        )
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))


def _list_raw_keys(config: DataServerConfig) -> list[str]:
    return sorted(
        p.name
        for p in config.raw_root.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )


def create_app(config: DataServerConfig) -> FastAPI:
    """Build the FastAPI app bound to ``config`` (stored on ``app.state``)."""
    app = FastAPI(
        title=APP_TITLE,
        summary="Keyed read-only access to packaged torchcell datasets and the raw mirror.",
        description=(
            "Every route except `/health` needs an `X-API-Key` header. Dataset "
            "archives are `tar.xz` files holding a built dataset's `processed/` "
            "(LMDB) and `preprocess/` (build manifest, gene set, reference index) "
            "directories; verify a download against `X-Artifact-SHA256` or the "
            "index row's `archive_sha256`. The raw mirror serves the source files "
            "each dataset loader consumed, hash-pinned by its `manifest.json`."
        ),
        version=APP_VERSION,
    )
    app.state.config = config

    @app.get("/health", response_model=Health, tags=["service"])
    def health() -> Health:
        """Liveness: artifact count, raw-key count, and the served roots. No key needed."""
        index_path = config.store_root / INDEX_FILENAME
        n_artifacts = (
            len(ArtifactIndex.load(index_path).artifacts) if index_path.is_file() else 0
        )
        return Health(
            status="ok",
            n_artifacts=n_artifacts,
            n_raw_keys=len(_list_raw_keys(config)),
            store_root=str(config.store_root),
            raw_root=str(config.raw_root),
        )

    @app.get("/datasets", response_model=ArtifactIndex, tags=["datasets"])
    def list_datasets(_: str = Depends(require_key)) -> ArtifactIndex:
        """The whole artifact index (`index.json`): one row per packaged archive.

        Each row names the dataset slug, the loader class, the `torchcell_version`
        and git commit it was built with, the KG release it was admitted to, the
        `content_sha256` of its experiment ids, the archive name, sha256 and size,
        and `status` (`supported` or `deprecated`). Pick the newest `supported` row
        whose `torchcell_version` shares your installed `major.minor`.
        """
        return _load_index(config)

    @app.get(
        "/datasets/{slug}", response_model=list[DatasetArtifact], tags=["datasets"]
    )
    def list_dataset_artifacts(
        slug: str, _: str = Depends(require_key)
    ) -> list[DatasetArtifact]:
        """Every packaged version of one dataset slug (404 if none is indexed)."""
        rows = _load_index(config).for_slug(slug)
        if not rows:
            raise HTTPException(status_code=404, detail="no artifacts for this slug")
        return rows

    @app.get("/datasets/{slug}/{archive}", tags=["datasets"])
    def get_dataset_archive(
        slug: str, archive: str, _: str = Depends(require_key)
    ) -> FileResponse:
        """Stream one archive named by an index row.

        Responds with `X-Artifact-SHA256` (the row's `archive_sha256`) and
        `Accept-Ranges: bytes`; a `Range: bytes=<start>-` request returns 206 with
        `Content-Range`, so an interrupted download resumes from its partial file.
        An archive on disk that the index does not list is 404.
        """
        row = _artifact_row(_load_index(config), slug, archive)
        target = _contained_file(config.store_root / slug, archive, config.store_root)
        return FileResponse(
            target,
            media_type=ARCHIVE_MEDIA_TYPE,
            filename=archive,
            headers={"X-Artifact-SHA256": row.archive_sha256},
        )

    @app.get("/raw", response_model=RawKeyListing, tags=["raw"])
    def list_raw_keys(_: str = Depends(require_key)) -> RawKeyListing:
        """Citation keys present in the raw mirror (dynamic; a new key needs no restart)."""
        keys = _list_raw_keys(config)
        return RawKeyListing(citation_keys=keys, count=len(keys))

    @app.get("/raw/{citation_key}/manifest", response_model=Manifest, tags=["raw"])
    def get_raw_manifest(citation_key: str, _: str = Depends(require_key)) -> Manifest:
        """A key's `manifest.json` verbatim: per-file role, bytes, sha256, retrieval."""
        return _raw_manifest(_raw_key_dir(config, citation_key))

    @app.get(
        "/raw/{citation_key}/files", response_model=list[RawFileListing], tags=["raw"]
    )
    def list_raw_files(
        citation_key: str, _: str = Depends(require_key)
    ) -> list[RawFileListing]:
        """The files a key's manifest lists, with role, bytes and sha256."""
        manifest = _raw_manifest(_raw_key_dir(config, citation_key))
        return [
            RawFileListing(path=f.path, role=f.role, bytes=f.bytes, sha256=f.sha256)
            for f in manifest.files
        ]

    @app.get("/raw/{citation_key}/artifact/{rel_path:path}", tags=["raw"])
    def get_raw_artifact(
        citation_key: str, rel_path: str, _: str = Depends(require_key)
    ) -> FileResponse:
        """Stream one raw file; `X-Artifact-SHA256` carries the manifest hash.

        Only files the manifest lists are served; `Range` requests are honored.
        """
        base = _raw_key_dir(config, citation_key)
        manifest = _raw_manifest(base)
        record = next((f for f in manifest.files if f.path == rel_path), None)
        if record is None:
            raise HTTPException(status_code=404, detail="file not in the manifest")
        target = _contained_file(base, rel_path, base)
        return FileResponse(
            target,
            media_type="application/octet-stream",
            filename=Path(rel_path).name,
            headers={"X-Artifact-SHA256": record.sha256},
        )

    return app


def create_app_from_env() -> FastAPI:
    """Factory for ``uvicorn --factory torchcell.datasets.server:create_app_from_env``."""
    load_dotenv()
    return create_app(DataServerConfig.from_env())


def main() -> None:
    """CLI: run the server, or ``--gen-key NAME`` to mint a key."""
    import argparse

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen-key", metavar="NAME", help="Mint a key and exit.")
    parser.add_argument("--host", default=None, help="Override TC_DATA_HOST.")
    parser.add_argument("--port", type=int, default=None, help="Override TC_DATA_PORT.")
    args = parser.parse_args()

    if args.gen_key:
        print_minted_key(args.gen_key, KEYS_FILE_VAR)
        return

    config = DataServerConfig.from_env()
    host = args.host or config.host
    port = config.port if args.port is None else args.port
    log.info(
        "dataset endpoint: store %s, raw %s on %s:%d",
        config.store_root,
        config.raw_root,
        host,
        port,
    )
    uvicorn.run(create_app(config), host=host, port=port)


if __name__ == "__main__":
    main()
