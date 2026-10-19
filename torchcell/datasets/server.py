# torchcell/datasets/server.py
# [[torchcell.datasets.server]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/server.py
# Test file: tests/torchcell/datasets/test_datasets_server.py

"""``tc-data``: public read-only HTTP endpoint for packaged datasets and the file tiers.

Serves five directory trees over the network with no key on any route: everything here
is published data (packaged datasets, SI and raw data files, reference genomes, derived
objects, a paper's supplementary material). The one thing withheld is paper content: in
the raw, objects and library tiers a manifest record whose ``role`` starts with
``paper_`` or ``ocr_`` (``WITHHELD_ROLE_PREFIXES``) is dropped from ``/manifest`` and
``/files`` and its bytes answer 404. Papers, their OCR and layout files are served by the
keyed literature endpoint (``torchcell.literature.server``), not here.

- The artifact store ``$TC_DATA_ROOT``: ``index.json`` (an ``ArtifactIndex``) and one
  ``<slug>/<archive>`` per packaged dataset, written by
  ``scripts/package_dataset_lmdb.py``. The index is authoritative: an archive on disk
  that the index does not list is not served, and every artifact response carries
  ``X-Artifact-SHA256`` from its index row. Archives are large, so
  ``Accept-Ranges: bytes`` is advertised and a ``Range`` header is honored (206 with
  ``Content-Range``), which is what lets the client resume a partial download.
- Four manifest-gated file tiers, one directory per key, each key holding a
  ``manifest.json`` that lists every file with path, role, bytes and sha256:

  - the raw mirror ``$TC_DATA_RAW_ROOT`` (default ``$DATA_ROOT/torchcell-raw``), keyed
    by citation key, literature ``Manifest`` shape;
  - the genomes tier ``$TC_DATA_GENOMES_ROOT`` (default ``$DATA_ROOT/torchcell-genomes``),
    keyed by assembly set, ``GenomeManifest`` shape
    (``torchcell.sequence.genome.registry``);
  - the objects tier ``$TC_DATA_OBJECTS_ROOT`` (default ``$DATA_ROOT/torchcell-objects``),
    keyed by a citation key or a named derived set, literature ``Manifest`` shape;
  - the library tier ``$TC_DATA_LIBRARY_ROOT`` (default ``$DATA_ROOT/torchcell-library``),
    keyed by citation key, literature ``Manifest`` shape. Only a paper's supplementary
    material is served from it (the ``si_*`` and data roles); the paper PDF, its OCR
    markdown, OCR images and layout files are never listed or served.

  The four tiers share one serving path (:func:`_serve_listed_file`): a file is served
  only when its key's manifest lists it, ``X-Artifact-SHA256`` carries the manifest hash
  (never a hash computed from the bytes), and ranges are honored as for archives. The
  raw mirror must exist at startup; a genomes, objects or library root that is absent on
  disk lists no keys and answers every other route of its tier 404.

Swagger UI is on at ``/docs`` (``/openapi.json`` for the schema). Configuration is
environment-driven: ``TC_DATA_ROOT``, ``TC_DATA_RAW_ROOT``, ``TC_DATA_GENOMES_ROOT``,
``TC_DATA_OBJECTS_ROOT``, ``TC_DATA_LIBRARY_ROOT``, ``TC_DATA_HOST`` and ``TC_DATA_PORT``
(8724).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from torchcell.datasets.artifact import (
    INDEX_FILENAME,
    ArtifactIndex,
    DatasetArtifact,
    ManifestFileListing,
)
from torchcell.literature.manifest import MANIFEST_FILENAME, ArtifactRecord, Manifest
from torchcell.sequence.genome.registry import (
    GENOMES_DIR,
    GenomeIntegrityError,
    GenomeManifest,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

APP_TITLE = "torchcell dataset endpoint"
APP_VERSION = "2.1.0"
DEFAULT_PORT = 8724
RAW_SUBDIR = "torchcell-raw"
GENOMES_SUBDIR = GENOMES_DIR
OBJECTS_SUBDIR = "torchcell-objects"
#: Spelled out, not imported from ``torchcell.artifacts.tiers.LIBRARY_DIR``: importing the
#: artifacts package pulls in httpx and the schema, which the slim tc-data image lacks
#: (docker/Dockerfile.tc-data). A test pins the two equal.
LIBRARY_SUBDIR = "torchcell-library"
ARCHIVE_MEDIA_TYPE = "application/x-xz"
FILE_MEDIA_TYPE = "application/octet-stream"
#: Manifest role prefixes never listed or served: the paper itself (paper_pdf,
#: paper_ocr, ...) and its OCR derivatives (ocr_image, ocr_layout, ocr_provenance).
WITHHELD_ROLE_PREFIXES: tuple[str, ...] = ("paper_", "ocr_")


def _tier_root_from_env(var: str, subdir: str) -> Path:
    """``$<var>`` when set, else ``$DATA_ROOT/<subdir>`` (``KeyError`` if neither is)."""
    value = os.environ.get(var)
    return Path(value) if value else Path(os.environ["DATA_ROOT"]) / subdir


class DataServerConfig(BaseModel):
    """Runtime configuration for the dataset endpoint."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    store_root: Path = Field(description="The artifact store (index.json + archives).")
    raw_root: Path = Field(description="The raw mirror (one dir per citation key).")
    genomes_root: Path = Field(
        description="The genomes tier (one dir per assembly set); may be absent."
    )
    objects_root: Path = Field(
        description="The objects tier (one dir per object key); may be absent."
    )
    library_root: Path = Field(
        description="The library tier (one dir per citation key); may be absent."
    )
    host: str = "0.0.0.0"
    port: int = DEFAULT_PORT

    @classmethod
    def from_env(cls) -> Self:
        """Build config from ``TC_DATA_*``; the tier roots default under ``$DATA_ROOT``.

        The store and the raw mirror must exist. The genomes, objects and library roots
        are not checked: a root that is absent serves an empty listing, so a host without
        one of those tiers runs unchanged.
        """
        store_root = Path(os.environ["TC_DATA_ROOT"])
        if not store_root.is_dir():
            raise FileNotFoundError(f"artifact store does not exist: {store_root}")
        raw_root = _tier_root_from_env("TC_DATA_RAW_ROOT", RAW_SUBDIR)
        if not raw_root.is_dir():
            raise FileNotFoundError(f"raw mirror does not exist: {raw_root}")
        return cls(
            store_root=store_root,
            raw_root=raw_root,
            genomes_root=_tier_root_from_env("TC_DATA_GENOMES_ROOT", GENOMES_SUBDIR),
            objects_root=_tier_root_from_env("TC_DATA_OBJECTS_ROOT", OBJECTS_SUBDIR),
            library_root=_tier_root_from_env("TC_DATA_LIBRARY_ROOT", LIBRARY_SUBDIR),
            host=os.environ.get("TC_DATA_HOST", "0.0.0.0"),
            port=int(os.environ.get("TC_DATA_PORT", str(DEFAULT_PORT))),
        )


class RawKeyListing(BaseModel):
    """Citation keys present in the raw mirror."""

    citation_keys: list[str]
    count: int


class GenomeSetListing(BaseModel):
    """Assembly sets in the genomes tier that carry a ``manifest.json``."""

    assembly_sets: list[str]
    count: int


class ObjectKeyListing(BaseModel):
    """Object keys in the objects tier that carry a ``manifest.json``."""

    object_keys: list[str]
    count: int


class LibraryKeyListing(BaseModel):
    """Citation keys in the library tier that carry a ``manifest.json``."""

    citation_keys: list[str]
    count: int


class Health(BaseModel):
    """Liveness summary (no auth)."""

    status: str
    n_artifacts: int
    n_raw_keys: int
    n_genome_sets: int
    n_object_keys: int
    n_library_keys: int
    store_root: str
    raw_root: str
    genomes_root: str
    objects_root: str
    library_root: str


class FileTier(BaseModel):
    """One manifest-gated tier: where it lives and what its keys are called.

    ``list_bare_keys`` keeps the raw mirror's listing contract (every key directory,
    with or without a manifest); the genomes, objects and library tiers list only keys
    that carry a ``manifest.json``, the only keys they can serve.
    """

    model_config = ConfigDict(frozen=True)

    root: Path
    key_noun: str = Field(description='"citation key", "assembly set", "object key".')
    list_bare_keys: bool


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


def _tier_key_dir(tier: FileTier, key: str) -> Path:
    """Resolve + containment-check one key directory under a tier root.

    An underscore-prefixed name is a service directory, never a key
    (:func:`_list_tier_keys` hides it), so it answers exactly like an absent key. The
    literature server's ``_key_dir`` applies the same rule. A tier root absent on disk
    makes every key absent.
    """
    base = (tier.root / key).resolve()
    if (
        key.startswith("_")
        or not base.is_relative_to(tier.root.resolve())
        or not base.is_dir()
    ):
        raise HTTPException(status_code=404, detail=f"unknown {tier.key_noun}")
    return base


def _tier_manifest_text(tier: FileTier, base: Path) -> str:
    """A key's ``manifest.json`` text; a key without one is not served (404)."""
    path = base / MANIFEST_FILENAME
    if not path.is_file():
        raise HTTPException(
            status_code=404, detail=f"no manifest.json for this {tier.key_noun}"
        )
    return path.read_text(encoding="utf-8")


def _list_tier_keys(tier: FileTier) -> list[str]:
    """Key directories under a tier root, sorted; an absent root lists none."""
    if not tier.root.is_dir():
        return []
    return sorted(
        p.name
        for p in tier.root.iterdir()
        if p.is_dir()
        and not p.name.startswith("_")
        and (tier.list_bare_keys or (p / MANIFEST_FILENAME).is_file())
    )


def _is_withheld(record: ArtifactRecord) -> bool:
    """Whether a manifest record is paper content, which tc-data never lists or serves."""
    return record.role.startswith(WITHHELD_ROLE_PREFIXES)


def _public_files(files: Sequence[ArtifactRecord]) -> list[ArtifactRecord]:
    """The manifest records tc-data publishes: every record that is not paper content."""
    return [f for f in files if not _is_withheld(f)]


def _public_manifest(manifest: Manifest) -> Manifest:
    """A literature-shaped manifest with its paper records removed."""
    return manifest.model_copy(update={"files": _public_files(manifest.files)})


def _serve_public_file(
    base: Path, files: Sequence[ArtifactRecord], rel_path: str
) -> FileResponse:
    """Serve a raw, objects or library file; a paper record is 404 (not served).

    The withheld check comes first so the detail names the reason; every other path
    goes through :func:`_serve_listed_file` over the public records.
    """
    if any(f.path == rel_path and _is_withheld(f) for f in files):
        raise HTTPException(status_code=404, detail="paper content is not served")
    return _serve_listed_file(base, _public_files(files), rel_path)


def _file_listing(files: Sequence[ArtifactRecord]) -> list[ManifestFileListing]:
    """The ``/files`` rows of a manifest: path, role, bytes, sha256 per file."""
    return [
        ManifestFileListing(path=f.path, role=f.role, bytes=f.bytes, sha256=f.sha256)
        for f in files
    ]


def _serve_listed_file(
    base: Path, files: Sequence[ArtifactRecord], rel_path: str
) -> FileResponse:
    """Stream ``base / rel_path`` when the manifest lists it; the one tier file path.

    A path the manifest does not list is 404 ``file not in the manifest``; a listed
    entry escaping ``base`` is 400; listed bytes missing from disk are 404
    ``file not found``. ``X-Artifact-SHA256`` is the manifest's recorded hash, and
    ``FileResponse`` advertises ``Accept-Ranges: bytes`` and answers ``Range`` with 206.
    """
    record = next((f for f in files if f.path == rel_path), None)
    if record is None:
        raise HTTPException(status_code=404, detail="file not in the manifest")
    target = _contained_file(base, rel_path, base)
    return FileResponse(
        target,
        media_type=FILE_MEDIA_TYPE,
        filename=Path(rel_path).name,
        headers={"X-Artifact-SHA256": record.sha256},
    )


def raw_tier(config: DataServerConfig) -> FileTier:
    """The raw mirror as a :class:`FileTier` (lists keys with or without a manifest)."""
    return FileTier(root=config.raw_root, key_noun="citation key", list_bare_keys=True)


def genomes_tier(config: DataServerConfig) -> FileTier:
    """The genomes tier as a :class:`FileTier` (lists manifested assembly sets)."""
    return FileTier(
        root=config.genomes_root, key_noun="assembly set", list_bare_keys=False
    )


def objects_tier(config: DataServerConfig) -> FileTier:
    """The objects tier as a :class:`FileTier` (lists manifested object keys)."""
    return FileTier(
        root=config.objects_root, key_noun="object key", list_bare_keys=False
    )


def library_tier(config: DataServerConfig) -> FileTier:
    """The library tier as a :class:`FileTier` (lists manifested citation keys)."""
    return FileTier(
        root=config.library_root, key_noun="citation key", list_bare_keys=False
    )


def _literature_manifest(tier: FileTier, base: Path) -> Manifest:
    """A raw, objects or library key's literature-shaped ``Manifest``; corrupt JSON is a 500."""
    return Manifest.model_validate_json(_tier_manifest_text(tier, base))


def _genome_manifest(tier: FileTier, base: Path, assembly_set: str) -> GenomeManifest:
    """A set's ``GenomeManifest``; one naming another set is an integrity error (500).

    The same check ``registry.load_genome_manifest`` applies on local reads.
    """
    manifest = GenomeManifest.model_validate_json(_tier_manifest_text(tier, base))
    if manifest.assembly_set != assembly_set:
        raise GenomeIntegrityError(
            f"{base / MANIFEST_FILENAME} names assembly set "
            f"{manifest.assembly_set!r}, not {assembly_set!r}"
        )
    return manifest


def create_app(config: DataServerConfig) -> FastAPI:
    """Build the FastAPI app bound to ``config`` (stored on ``app.state``)."""
    app = FastAPI(
        title=APP_TITLE,
        summary=(
            "Public read-only access to packaged torchcell datasets, the raw mirror, "
            "the genomes tier, the objects tier and the library tier's supplementary "
            "material; no key; paper content withheld."
        ),
        description=(
            "No route needs a key. Dataset archives are `tar.xz` files holding a built dataset's `processed/` "
            "(LMDB) and `preprocess/` (build manifest, gene set, reference index) "
            "directories; verify a download against `X-Artifact-SHA256` or the "
            "index row's `archive_sha256`. The raw mirror serves the source files "
            "each dataset loader consumed, the genomes tier serves reference and "
            "isolate assembly sets, and the objects tier serves derived bytes that "
            "graph records point at (embeddings, matrices, indexed FASTA), and the "
            "library tier serves a paper's supplementary material (SI files and "
            "data); each is hash-pinned by its key's `manifest.json`. Paper content "
            "(manifest roles beginning `paper_` or `ocr_`: the paper, its OCR and "
            "layout files) is withheld from the raw, objects and library tiers: it is "
            "not listed and its bytes answer 404."
        ),
        version=APP_VERSION,
    )
    app.state.config = config
    raw = raw_tier(config)
    genomes = genomes_tier(config)
    objects = objects_tier(config)
    library = library_tier(config)

    @app.get("/health", response_model=Health, tags=["service"])
    def health() -> Health:
        """Liveness: artifact and per-tier key counts, and the served roots."""
        index_path = config.store_root / INDEX_FILENAME
        n_artifacts = (
            len(ArtifactIndex.load(index_path).artifacts) if index_path.is_file() else 0
        )
        return Health(
            status="ok",
            n_artifacts=n_artifacts,
            n_raw_keys=len(_list_tier_keys(raw)),
            n_genome_sets=len(_list_tier_keys(genomes)),
            n_object_keys=len(_list_tier_keys(objects)),
            n_library_keys=len(_list_tier_keys(library)),
            store_root=str(config.store_root),
            raw_root=str(config.raw_root),
            genomes_root=str(config.genomes_root),
            objects_root=str(config.objects_root),
            library_root=str(config.library_root),
        )

    @app.get("/datasets", response_model=ArtifactIndex, tags=["datasets"])
    def list_datasets() -> ArtifactIndex:
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
    def list_dataset_artifacts(slug: str) -> list[DatasetArtifact]:
        """Every packaged version of one dataset slug (404 if none is indexed)."""
        rows = _load_index(config).for_slug(slug)
        if not rows:
            raise HTTPException(status_code=404, detail="no artifacts for this slug")
        return rows

    @app.get("/datasets/{slug}/{archive}", tags=["datasets"])
    def get_dataset_archive(slug: str, archive: str) -> FileResponse:
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
    def list_raw_keys() -> RawKeyListing:
        """Citation keys present in the raw mirror (dynamic; a new key needs no restart)."""
        keys = _list_tier_keys(raw)
        return RawKeyListing(citation_keys=keys, count=len(keys))

    @app.get("/raw/{citation_key}/manifest", response_model=Manifest, tags=["raw"])
    def get_raw_manifest(citation_key: str) -> Manifest:
        """A key's `manifest.json` without paper records: role, bytes, sha256, retrieval."""
        return _public_manifest(
            _literature_manifest(raw, _tier_key_dir(raw, citation_key))
        )

    @app.get(
        "/raw/{citation_key}/files",
        response_model=list[ManifestFileListing],
        tags=["raw"],
    )
    def list_raw_files(citation_key: str) -> list[ManifestFileListing]:
        """The public files a key's manifest lists, with role, bytes and sha256."""
        return _file_listing(
            _public_files(
                _literature_manifest(raw, _tier_key_dir(raw, citation_key)).files
            )
        )

    @app.get("/raw/{citation_key}/artifact/{rel_path:path}", tags=["raw"])
    def get_raw_artifact(citation_key: str, rel_path: str) -> FileResponse:
        """Stream one raw file; `X-Artifact-SHA256` carries the manifest hash.

        Only files the manifest lists are served, never paper content (404 `paper
        content is not served`); `Range` requests are honored.
        """
        base = _tier_key_dir(raw, citation_key)
        return _serve_public_file(base, _literature_manifest(raw, base).files, rel_path)

    @app.get("/genomes", response_model=GenomeSetListing, tags=["genomes"])
    def list_genome_sets() -> GenomeSetListing:
        """Assembly sets with a `manifest.json` (empty when the genomes root is absent)."""
        sets = _list_tier_keys(genomes)
        return GenomeSetListing(assembly_sets=sets, count=len(sets))

    @app.get(
        "/genomes/{assembly_set}/manifest",
        response_model=GenomeManifest,
        tags=["genomes"],
    )
    def get_genome_manifest(assembly_set: str) -> GenomeManifest:
        """A set's `GenomeManifest`: organism, release, and per-file role, bytes, sha256."""
        return _genome_manifest(
            genomes, _tier_key_dir(genomes, assembly_set), assembly_set
        )

    @app.get(
        "/genomes/{assembly_set}/files",
        response_model=list[ManifestFileListing],
        tags=["genomes"],
    )
    def list_genome_files(assembly_set: str) -> list[ManifestFileListing]:
        """The files a set's manifest lists, with role, bytes and sha256."""
        base = _tier_key_dir(genomes, assembly_set)
        return _file_listing(_genome_manifest(genomes, base, assembly_set).files)

    @app.get("/genomes/{assembly_set}/artifact/{rel_path:path}", tags=["genomes"])
    def get_genome_artifact(assembly_set: str, rel_path: str) -> FileResponse:
        """Stream one genomes-tier file; `X-Artifact-SHA256` carries the manifest hash.

        Only files the manifest lists are served; `Range` requests are honored.
        """
        base = _tier_key_dir(genomes, assembly_set)
        manifest = _genome_manifest(genomes, base, assembly_set)
        return _serve_listed_file(base, manifest.files, rel_path)

    @app.get("/objects", response_model=ObjectKeyListing, tags=["objects"])
    def list_object_keys() -> ObjectKeyListing:
        """Object keys with a `manifest.json` (empty when the objects root is absent)."""
        keys = _list_tier_keys(objects)
        return ObjectKeyListing(object_keys=keys, count=len(keys))

    @app.get(
        "/objects/{object_key}/manifest", response_model=Manifest, tags=["objects"]
    )
    def get_object_manifest(object_key: str) -> Manifest:
        """A key's `manifest.json` without paper records (literature `Manifest` shape)."""
        return _public_manifest(
            _literature_manifest(objects, _tier_key_dir(objects, object_key))
        )

    @app.get(
        "/objects/{object_key}/files",
        response_model=list[ManifestFileListing],
        tags=["objects"],
    )
    def list_object_files(object_key: str) -> list[ManifestFileListing]:
        """The public files a key's manifest lists, with role, bytes and sha256."""
        return _file_listing(
            _public_files(
                _literature_manifest(objects, _tier_key_dir(objects, object_key)).files
            )
        )

    @app.get("/objects/{object_key}/artifact/{rel_path:path}", tags=["objects"])
    def get_object_artifact(object_key: str, rel_path: str) -> FileResponse:
        """Stream one objects-tier file; `X-Artifact-SHA256` carries the manifest hash.

        Only files the manifest lists are served, never paper content (404 `paper
        content is not served`); `Range` requests are honored.
        """
        base = _tier_key_dir(objects, object_key)
        return _serve_public_file(
            base, _literature_manifest(objects, base).files, rel_path
        )

    @app.get("/library", response_model=LibraryKeyListing, tags=["library"])
    def list_library_keys() -> LibraryKeyListing:
        """Citation keys with a `manifest.json` (empty when the library root is absent)."""
        keys = _list_tier_keys(library)
        return LibraryKeyListing(citation_keys=keys, count=len(keys))

    @app.get(
        "/library/{citation_key}/manifest", response_model=Manifest, tags=["library"]
    )
    def get_library_manifest(citation_key: str) -> Manifest:
        """A key's `manifest.json` without paper, OCR or layout records."""
        return _public_manifest(
            _literature_manifest(library, _tier_key_dir(library, citation_key))
        )

    @app.get(
        "/library/{citation_key}/files",
        response_model=list[ManifestFileListing],
        tags=["library"],
    )
    def list_library_files(citation_key: str) -> list[ManifestFileListing]:
        """The supplementary files a key's manifest lists, with role, bytes and sha256."""
        return _file_listing(
            _public_files(
                _literature_manifest(
                    library, _tier_key_dir(library, citation_key)
                ).files
            )
        )

    @app.get("/library/{citation_key}/artifact/{rel_path:path}", tags=["library"])
    def get_library_artifact(citation_key: str, rel_path: str) -> FileResponse:
        """Stream one supplementary file; `X-Artifact-SHA256` carries the manifest hash.

        Only files the manifest lists are served, never the paper, its OCR or layout
        files (404 `paper content is not served`); `Range` requests are honored.
        """
        base = _tier_key_dir(library, citation_key)
        return _serve_public_file(
            base, _literature_manifest(library, base).files, rel_path
        )

    return app


def create_app_from_env() -> FastAPI:
    """Factory for ``uvicorn --factory torchcell.datasets.server:create_app_from_env``."""
    load_dotenv()
    return create_app(DataServerConfig.from_env())


def main() -> None:
    """CLI: run the server (``--host`` / ``--port`` override the environment)."""
    import argparse

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=None, help="Override TC_DATA_HOST.")
    parser.add_argument("--port", type=int, default=None, help="Override TC_DATA_PORT.")
    args = parser.parse_args()

    config = DataServerConfig.from_env()
    host = args.host or config.host
    port = config.port if args.port is None else args.port
    log.info(
        "dataset endpoint: store %s, raw %s, genomes %s, objects %s, "
        "library %s (supplementary material only) on %s:%d",
        config.store_root,
        config.raw_root,
        config.genomes_root,
        config.objects_root,
        config.library_root,
        host,
        port,
    )
    uvicorn.run(create_app(config), host=host, port=port)


if __name__ == "__main__":
    main()
