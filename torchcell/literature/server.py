# torchcell/literature/server.py
# [[torchcell.literature.server]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/server.py
# Test file: tests/torchcell/literature/test_server.py

"""Keyed read-only HTTP endpoint for the OCR literature mirror.

Serves ``$DATA_ROOT/torchcell-library/<citation_key>/`` over the local network so
the M1 Mac (and, if granted a key, a collaborator) can list + pull artifacts on
demand, sha256-verified against each key's ``manifest.json``. A client fetch is a
scriptable retrieval whose bytes are hash-checked exactly like ``direct_url``.

This is a private service hosted on GilaHyper (where the mirror physically lives). It
is distinct from the ``radiant_endpoint`` ``RetrievalMethod`` slot (issue #20), which
is reserved for the Radiant VM serving library-rebuild artifacts alongside the graph
database -- a separate concern that this endpoint does not fill.

The server is read-only, dynamic (a newly added citation-key directory shows up with
no restart), and authenticated by named API keys (multiple keys so a collaborator can
be granted -- and later revoked -- a distinct key). Keys are stored as sha256 hashes,
never plaintext; comparison is constant-time; key values are never logged.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import mimetypes
import os
import secrets
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from torchcell.literature.backfill import LIBRARY_SUBDIR
from torchcell.literature.manifest import MANIFEST_FILENAME, Manifest, _role_for

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"
SEARCH_RESULT_CAP = 200


def _hash_key(key: str) -> str:
    """sha256 hex digest of an API key (what we store + compare, never the key)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class LiteratureKeys(BaseModel):
    """Named API keys, stored as sha256 hashes for constant-time verification."""

    model_config = ConfigDict(frozen=True)

    hashes: dict[str, str] = Field(
        description="Map of key name -> sha256 hex of the key value."
    )

    @classmethod
    def from_file(cls, path: str | Path) -> LiteratureKeys:
        """Load ``{name: sha256hex}`` from a JSON keys file."""
        data = json.loads(Path(path).read_text())
        return cls(hashes={str(k): str(v) for k, v in data.items()})

    @classmethod
    def from_pairs(cls, spec: str) -> LiteratureKeys:
        """Parse ``name1:key1,name2:key2`` plaintext pairs, hashing each key.

        Convenient for quick-start/tests; the keys file (hashes at rest) is
        preferred for anything real since env values are visible via ``ps``.
        """
        hashes: dict[str, str] = {}
        for pair in spec.split(","):
            pair = pair.strip()
            if not pair:
                continue
            name, _, key = pair.partition(":")
            if not name or not key:
                raise ValueError(f"bad TC_LIT_API_KEYS pair: {pair!r}")
            hashes[name.strip()] = _hash_key(key.strip())
        return cls(hashes=hashes)

    @classmethod
    def from_env(cls) -> LiteratureKeys:
        """Load keys from ``TC_LIT_KEYS_FILE`` (preferred) or ``TC_LIT_API_KEYS``.

        Raises ``KeyError`` if neither is set -- the server never runs unauthenticated.
        """
        keys_file = os.environ.get("TC_LIT_KEYS_FILE")
        if keys_file:
            return cls.from_file(keys_file)
        inline = os.environ.get("TC_LIT_API_KEYS")
        if inline:
            return cls.from_pairs(inline)
        raise KeyError("Set TC_LIT_KEYS_FILE or TC_LIT_API_KEYS to run the server.")

    def verify(self, presented: str) -> str | None:
        """Return the name of the key matching ``presented``, else None.

        Constant-time over the stored hashes; the presented value is never logged.
        """
        candidate = _hash_key(presented)
        for name, stored in self.hashes.items():
            if hmac.compare_digest(candidate, stored):
                return name
        return None


class LiteratureServerConfig(BaseModel):
    """Runtime configuration for the literature endpoint."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    mirror_root: Path = Field(description="The torchcell-library directory to serve.")
    keys: LiteratureKeys
    host: str = "0.0.0.0"
    port: int = 8723

    @classmethod
    def from_env(cls) -> LiteratureServerConfig:
        """Build config from env: ``DATA_ROOT`` + ``TC_LIT_*``."""
        mirror_root = Path(os.environ["DATA_ROOT"]) / LIBRARY_SUBDIR
        if not mirror_root.is_dir():
            raise FileNotFoundError(f"mirror root does not exist: {mirror_root}")
        return cls(
            mirror_root=mirror_root,
            keys=LiteratureKeys.from_env(),
            host=os.environ.get("TC_LIT_HOST", "0.0.0.0"),
            port=int(os.environ.get("TC_LIT_PORT", "8723")),
        )


class FileListing(BaseModel):
    """A single artifact file as listed by the endpoint."""

    path: str
    role: str
    bytes: int
    sha256: str | None = Field(
        default=None, description="From the manifest; null if not yet backfilled."
    )


class KeyListing(BaseModel):
    """Citation keys currently present in the mirror."""

    citation_keys: list[str]
    count: int


class SearchHit(BaseModel):
    """A citation key matching a search query."""

    citation_key: str
    where: list[str] = Field(description="'citation_key' and/or 'paper.md'.")


class SearchResult(BaseModel):
    """Search results (filename + paper.md substring; NOT semantic)."""

    query: str
    hits: list[SearchHit]
    truncated: bool = False


_api_key_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def require_key(
    request: Request, api_key: str | None = Depends(_api_key_scheme)
) -> str:
    """Auth dependency: resolve + validate the ``X-API-Key`` header, return its name."""
    config: LiteratureServerConfig = request.app.state.config
    name = config.keys.verify(api_key) if api_key else None
    if name is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing API key",
        )
    return name


def _key_dir(config: LiteratureServerConfig, citation_key: str) -> Path:
    """Resolve + containment-check a citation-key directory under the mirror root."""
    base = (config.mirror_root / citation_key).resolve()
    root = config.mirror_root.resolve()
    if not base.is_relative_to(root) or not base.is_dir():
        raise HTTPException(status_code=404, detail="unknown citation key")
    return base


def _resolve_artifact(base: Path, rel_path: str) -> Path:
    """Resolve a file within ``base``, blocking path traversal."""
    target = (base / rel_path).resolve()
    if not target.is_relative_to(base):
        raise HTTPException(status_code=400, detail="path traversal rejected")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return target


def _load_manifest(base: Path) -> Manifest | None:
    """Load a directory's manifest, or None if it has not been backfilled."""
    manifest_path = base / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    return Manifest.model_validate_json(manifest_path.read_text())


def _list_citation_keys(config: LiteratureServerConfig) -> list[str]:
    """Directories directly under the mirror root (dynamic; reflects new papers)."""
    return sorted(p.name for p in config.mirror_root.iterdir() if p.is_dir())


def create_app(config: LiteratureServerConfig) -> FastAPI:
    """Build the FastAPI app bound to ``config`` (config stored on ``app.state``)."""
    app = FastAPI(
        title="torchcell literature endpoint",
        summary="Keyed read-only access to the OCR literature mirror.",
        version="1.0.0",
    )
    app.state.config = config

    @app.get("/health")
    def health() -> dict[str, object]:
        """Liveness + mirror summary (no auth)."""
        return {
            "status": "ok",
            "n_keys": len(_list_citation_keys(config)),
            "mirror_root": str(config.mirror_root),
        }

    @app.get("/keys")
    def list_keys(_: str = Depends(require_key)) -> KeyListing:
        """List citation keys present in the mirror."""
        keys = _list_citation_keys(config)
        return KeyListing(citation_keys=keys, count=len(keys))

    @app.get("/keys/{citation_key}/manifest")
    def get_manifest(citation_key: str, _: str = Depends(require_key)) -> Manifest:
        """Return a key's ``manifest.json`` (404 if not yet backfilled)."""
        base = _key_dir(config, citation_key)
        manifest = _load_manifest(base)
        if manifest is None:
            raise HTTPException(status_code=404, detail="no manifest (not backfilled)")
        return manifest

    @app.get("/keys/{citation_key}/files")
    def list_files(
        citation_key: str, _: str = Depends(require_key)
    ) -> list[FileListing]:
        """List a key's files with role + sha256 (from manifest, or live if absent)."""
        base = _key_dir(config, citation_key)
        manifest = _load_manifest(base)
        if manifest is not None:
            return [
                FileListing(path=f.path, role=f.role, bytes=f.bytes, sha256=f.sha256)
                for f in manifest.files
            ]
        # Dynamic fallback for a freshly added key with no manifest yet.
        listings: list[FileListing] = []
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.name == MANIFEST_FILENAME:
                continue
            rel = str(path.relative_to(base))
            listings.append(
                FileListing(
                    path=rel,
                    role=_role_for(rel),
                    bytes=path.stat().st_size,
                    sha256=None,
                )
            )
        return listings

    @app.get("/keys/{citation_key}/artifact/{rel_path:path}")
    def get_artifact(
        citation_key: str, rel_path: str, _: str = Depends(require_key)
    ) -> FileResponse:
        """Stream one artifact; ``X-Artifact-SHA256`` carries the manifest hash."""
        base = _key_dir(config, citation_key)
        target = _resolve_artifact(base, rel_path)
        manifest = _load_manifest(base)
        headers: dict[str, str] = {}
        if manifest is not None:
            for f in manifest.files:
                if f.path == rel_path:
                    headers["X-Artifact-SHA256"] = f.sha256
                    break
        media_type, _enc = mimetypes.guess_type(target.name)
        return FileResponse(
            target, media_type=media_type or "application/octet-stream", headers=headers
        )

    @app.get("/search")
    def search(q: str, _: str = Depends(require_key)) -> SearchResult:
        """Substring search over citation keys + ``paper.md`` (NOT semantic)."""
        needle = q.lower()
        hits: list[SearchHit] = []
        truncated = False
        for citation_key in _list_citation_keys(config):
            where: list[str] = []
            if needle in citation_key.lower():
                where.append("citation_key")
            paper_md = config.mirror_root / citation_key / "paper.md"
            if (
                paper_md.is_file()
                and needle in paper_md.read_text(errors="ignore").lower()
            ):
                where.append("paper.md")
            if where:
                hits.append(SearchHit(citation_key=citation_key, where=where))
            if len(hits) >= SEARCH_RESULT_CAP:
                truncated = True
                log.info("search: capped at %d hits for %r", SEARCH_RESULT_CAP, q)
                break
        return SearchResult(query=q, hits=hits, truncated=truncated)

    return app


def create_app_from_env() -> FastAPI:
    """Factory for ``uvicorn --factory torchcell.literature.server:create_app_from_env``."""
    load_dotenv()
    return create_app(LiteratureServerConfig.from_env())


def _gen_key(name: str) -> None:
    """Print a fresh random API key + the JSON keys-file line to store its hash."""
    key = secrets.token_urlsafe(32)
    entry = {name: _hash_key(key)}
    print(f"API key for '{name}' (give this to the client, it is NOT stored):\n  {key}")
    print("\nAdd this to your TC_LIT_KEYS_FILE (JSON of {name: sha256hex}):")
    print(f"  {json.dumps(entry)}")


def main() -> None:
    """CLI: run the server, or ``--gen-key NAME`` to mint a key."""
    import argparse

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen-key", metavar="NAME", help="Mint a key and exit.")
    parser.add_argument("--host", default=None, help="Override TC_LIT_HOST.")
    parser.add_argument("--port", type=int, default=None, help="Override TC_LIT_PORT.")
    args = parser.parse_args()

    if args.gen_key:
        _gen_key(args.gen_key)
        return

    config = LiteratureServerConfig.from_env()
    host = args.host or config.host
    port = args.port or config.port
    log.info("literature endpoint: serving %s on %s:%d", config.mirror_root, host, port)
    uvicorn.run(create_app(config), host=host, port=port)


if __name__ == "__main__":
    main()
