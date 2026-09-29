# torchcell/datasets/artifact.py
# [[torchcell.datasets.artifact]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/artifact.py
# Test file: tests/torchcell/datasets/test_artifact.py

"""Records for packaged dataset archives: the unit the ``tc-data`` endpoint serves.

A built dataset directory (``processed/`` with the records LMDB and the sibling
``interned`` LMDB, plus ``preprocess/`` with ``build_manifest.json``, ``gene_set.json``
and the reference index) is packaged once by ``scripts/package_dataset_lmdb.py`` into
one archive under an artifact store ``$TC_DATA_ROOT/<slug>/<archive>``. The store's
``index.json`` is an :class:`ArtifactIndex`; each row is a :class:`DatasetArtifact`
naming the loader, the package version and git commit the LMDB was built with, the
knowledge-graph release it was admitted to, the content hash of its experiment ids, and
the archive's sha256 and size. The archive file name embeds the slug, the package
version and the first eight hex digits of the archive sha256, so two packagings of
different bytes never share a name.

The client (``torchcell.datasets.client``) picks the newest ``supported`` row whose
``torchcell_version`` shares the installed ``major.minor``; a ``deprecated`` row stays
in the index so a pinned checkout can still name what it used, but is never selected.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from torchcell.provenance.build_manifest import BuildManifest

INDEX_FILENAME = "index.json"
SHA256SUMS_FILENAME = "SHA256SUMS"
INDEX_SCHEMA_VERSION = 1
# tarfile + lzma (stdlib); ``zstandard`` is not in the torchcell environment.
ARCHIVE_SUFFIX = ".tar.xz"
ArtifactStatus = Literal["supported", "deprecated"]

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)")


def parse_version(version: str) -> tuple[int, int, int]:
    """``"1.2.1"`` -> ``(1, 2, 1)``; a trailing local or dev suffix is ignored."""
    match = _VERSION_RE.match(version)
    if match is None:
        raise ValueError(f"not a major.minor.patch version: {version!r}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def major_minor(version: str) -> str:
    """``"1.2.1"`` -> ``"1.2"``, the compatibility line an artifact is selected on."""
    major, minor, _ = parse_version(version)
    return f"{major}.{minor}"


def content_sha256(ids: Iterable[str]) -> str:
    """sha256 of the sorted, newline-joined experiment ids (trailing newline).

    The same digest as ``torchcell.knowledge_graphs.releases.content_sha256`` over the
    same ids, so an artifact's content hash is comparable with the release manifest's.
    """
    digest = hashlib.sha256()
    for item in sorted(ids):
        digest.update(item.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def archive_name(slug: str, torchcell_version: str, archive_sha256: str) -> str:
    """``<slug>-<torchcell_version>-<archive_sha256[:8]><ARCHIVE_SUFFIX>``."""
    return f"{slug}-{torchcell_version}-{archive_sha256[:8]}{ARCHIVE_SUFFIX}"


class DatasetArtifact(BaseModel):
    """One packaged dataset archive and the provenance of the LMDB inside it."""

    model_config = ConfigDict(frozen=True)

    slug: str = Field(description="Dataset directory name, e.g. smf_costanzo2016.")
    dataset_class: str = Field(description="Loader class name that built the LMDB.")
    torchcell_version: str = Field(description="torchcell.__version__ at build time.")
    torchcell_commit: str | None = Field(
        default=None, description="Git commit of the build checkout; None for a wheel."
    )
    kg_release: str | None = Field(
        default=None, description="KG release id this build was admitted to, if any."
    )
    kg_version: str | None = Field(
        default=None, description="KG version (e.g. 1.2) of that release, if any."
    )
    content_sha256: str | None = Field(
        default=None,
        description="content_sha256 over the sorted experiment ids; None if not computed.",
    )
    n_experiments: int = Field(description="Records in the LMDB.")
    archive: str = Field(description="Archive file name under <store>/<slug>/.")
    archive_sha256: str
    archive_bytes: int
    built_at: str = Field(description="ISO-8601 UTC from build_manifest.json.")
    packaged_at: str = Field(description="ISO-8601 UTC when the archive was written.")
    status: ArtifactStatus = "supported"
    build_manifest: BuildManifest | None = Field(
        default=None, description="preprocess/build_manifest.json, verbatim."
    )

    @property
    def rel_path(self) -> str:
        """Path of the archive relative to the store root."""
        return f"{self.slug}/{self.archive}"

    @property
    def sort_key(self) -> tuple[str, tuple[int, int, int], str]:
        """Deterministic index order: slug, version, archive sha256."""
        return (self.slug, parse_version(self.torchcell_version), self.archive_sha256)


class ArtifactIndex(BaseModel):
    """``<store>/index.json``: every packaged artifact, sorted and byte-stable."""

    schema_version: int = INDEX_SCHEMA_VERSION
    generated_at: str = Field(description="ISO-8601 UTC of the last save.")
    artifacts: list[DatasetArtifact] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Read an index file."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def sorted(self) -> Self:
        """A copy with the artifacts in :attr:`DatasetArtifact.sort_key` order."""
        return self.model_copy(
            update={"artifacts": sorted(self.artifacts, key=lambda a: a.sort_key)}
        )

    def to_json(self) -> str:
        """The byte-stable serialization: sorted rows, two-space indent, one newline."""
        return self.sorted().model_dump_json(indent=2) + "\n"

    def save(self, path: str | Path) -> Path:
        """Write :meth:`to_json` to ``path``; saving a loaded index reproduces its bytes."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(), encoding="utf-8")
        return out

    def upsert(self, artifact: DatasetArtifact, generated_at: str) -> Self:
        """A new index with ``artifact`` replacing any row of the same slug and archive."""
        kept = [
            a
            for a in self.artifacts
            if not (a.slug == artifact.slug and a.archive == artifact.archive)
        ]
        return self.model_copy(
            update={"artifacts": [*kept, artifact], "generated_at": generated_at}
        ).sorted()

    def for_slug(self, slug: str) -> list[DatasetArtifact]:
        """Every row for ``slug``, in index order."""
        return [a for a in self.sorted().artifacts if a.slug == slug]

    def select(self, slug: str, torchcell_version: str) -> DatasetArtifact | None:
        """The newest ``supported`` row for ``slug`` on ``torchcell_version``'s major.minor.

        Newest is the highest package version, then the latest ``packaged_at``. None
        when no row matches; the caller decides whether that is an error.
        """
        line = major_minor(torchcell_version)
        candidates = [
            a
            for a in self.artifacts
            if a.slug == slug
            and a.status == "supported"
            and major_minor(a.torchcell_version) == line
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda a: (parse_version(a.torchcell_version), a.packaged_at),
        )

    def sha256sums(self) -> str:
        """``sha256sum -c``-compatible lines, one per artifact, relative to the store."""
        return "".join(
            f"{a.archive_sha256}  {a.rel_path}\n" for a in self.sorted().artifacts
        )
