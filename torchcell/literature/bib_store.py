# torchcell/literature/bib_store.py
# [[torchcell.literature.bib_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/bib_store.py
# Test file: tests/torchcell/literature/test_bib_store.py

r"""Named bibliographies materialized into the mirror, for tc-lit to serve.

Both LaTeX bibliography flows (``notes-tex/<slug>/references.bib`` via
``make bib``, and ``paper/nature-biotech/references.bib`` via
``zotero_export_bib.py``) read the Better BibTeX endpoint on ``localhost:23119``,
so they only run on a machine with Zotero desktop open. GilaHyper has no Zotero
desktop, and a second machine regenerating a bib by hand is how two copies of the
same bibliography drift apart.

This module makes the bibliography an ARTIFACT like every other file in the mirror:
a host-side job pulls each named scope over the Zotero Web API (headless, the same
pull :mod:`torchcell.literature.bib` does for the Dendron ``bib.bib``), writes
``<name>.bib`` into ``<DATA_ROOT>/torchcell-library/_bib/`` beside a
``manifest.json`` that pins each file's sha256, and ``tc-lit`` streams the files
read-through with the hash in ``X-Artifact-SHA256``. A client on any machine then
pulls one pinned, verifiable bibliography instead of re-exporting its own.

**Names come from the repo, not from a config.** :func:`discover_bib_specs` reads
the scope of each bibliography from where it is already declared:

- ``paper`` -- the group ``paper`` collection ONLY, the manuscript's publication
  guarantee (``paper/nature-biotech/zotero_export_bib.py``).
- one per ``notes-tex/<slug>/`` -- the ``ZOTERO_COLLECTION`` and
  ``ZOTERO_PERSONAL_COLLECTION`` lines of that document's Makefile, named for the
  slug so ``make bib-pull`` can ask for ``$(DOC)``.
- ``library`` -- the group library unioned with the personal ``torchcell`` tree,
  the Dendron scope.

**The bytes are content-stable.** The header names the scope but carries no
timestamp, so an unchanged Zotero collection re-exports to an identical file and
an identical sha256; ``generated_at`` lives in the manifest. A pull whose hash
matches the one already on disk is a no-op for the client.

The store directory is underscore-prefixed, like ``_sync_reports``, which is the
convention for a service directory in the mirror root that is NOT a citation key.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from torchcell.literature.bib import (
    fetch_bibtex_entries,
    fetch_paired_collection_entries,
    fetch_union_bibtex_entries,
    write_bib_entries,
)
from torchcell.literature.manifest import sha256_file
from torchcell.literature.zotero import ZoteroLibrary

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BIB_STORE_SUBDIR = "_bib"
BIB_STORE_MANIFEST = "manifest.json"
BIB_STORE_VERSION = 1

# The manuscript's collection: the group library's `paper` collection, addressed by
# key so a rename cannot move it. Same value as DEFAULT_COLLECTION in
# paper/nature-biotech/zotero_export_bib.py, which is a script, not a module.
PAPER_COLLECTION_KEY = "W46ATS7B"
PAPER_BIB_NAME = "paper"
LIBRARY_BIB_NAME = "library"
DEFAULT_USER_ROOT_COLLECTION = "torchcell"

# A served bibliography name: a path segment with no separators, so the name can
# never address a file outside the store.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")

# `ZOTERO_COLLECTION := FE8DQKUH` in a notes-tex document Makefile.
_MAKEFILE_VAR_RE = re.compile(
    r"^\s*(ZOTERO_COLLECTION|ZOTERO_PERSONAL_COLLECTION)\s*[:?]?=\s*(\S*)\s*$"
)


class BibScope(BaseModel):
    """Which Zotero collections a bibliography is the export of.

    Exactly one of three shapes: a single group collection (the manuscript); a
    group collection paired with one personal collection (a notes-tex document);
    or the group library unioned with a personal collection tree (the Dendron
    scope). Collections are addressed by key where the source declares a key.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    group_library_id: str
    group_collection: str | None = Field(
        default=None,
        description="One group collection (key or name); None = the whole group.",
    )
    user_library_id: str | None = None
    user_collection: str | None = Field(
        default=None, description="One personal collection paired with the group one."
    )
    user_root_collection: str | None = Field(
        default=None, description="A personal collection tree unioned in recursively."
    )


class BibSpec(BaseModel):
    """A named bibliography and where its scope was declared."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    scope: BibScope
    origin: str = Field(description="Repo file the scope was read from.")


class BibRecord(BaseModel):
    """One exported bibliography as pinned in the store manifest."""

    model_config = ConfigDict(extra="forbid")

    name: str
    path: str = Field(description="Relative to the store directory: '<name>.bib'.")
    bytes: int
    sha256: str
    n_entries: int
    scope: BibScope
    origin: str
    generated_at: str = Field(description="ISO timestamp of the export (UTC).")


class BibStoreManifest(BaseModel):
    """Integrity record for the whole store, written as ``_bib/manifest.json``."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=BIB_STORE_VERSION)
    bibs: list[BibRecord] = Field(default_factory=list)
    generated_at: str

    def get(self, name: str) -> BibRecord | None:
        """The record for ``name``, else None."""
        for record in self.bibs:
            if record.name == name:
                return record
        return None


def validate_bib_name(name: str) -> str:
    """Return ``name`` if it is a legal store name, else raise ``ValueError``."""
    if not _NAME_RE.match(name):
        raise ValueError(f"illegal bibliography name: {name!r}")
    return name


def bib_store_dir(mirror_root: str | Path) -> Path:
    """``<mirror_root>/_bib``."""
    return Path(mirror_root) / BIB_STORE_SUBDIR


def load_bib_store(mirror_root: str | Path) -> BibStoreManifest:
    """Read the store manifest; ``FileNotFoundError`` if no export has run."""
    path = bib_store_dir(mirror_root) / BIB_STORE_MANIFEST
    return BibStoreManifest.model_validate_json(path.read_text())


def parse_makefile_collections(makefile: Path) -> tuple[str, str]:
    """``(ZOTERO_COLLECTION, ZOTERO_PERSONAL_COLLECTION)`` from a notes-tex Makefile.

    Either may be empty: a document that cites nothing declares neither.
    """
    values = {"ZOTERO_COLLECTION": "", "ZOTERO_PERSONAL_COLLECTION": ""}
    for line in makefile.read_text().splitlines():
        match = _MAKEFILE_VAR_RE.match(line)
        if match:
            values[match.group(1)] = match.group(2)
    return values["ZOTERO_COLLECTION"], values["ZOTERO_PERSONAL_COLLECTION"]


def discover_bib_specs(
    project_root: str | Path,
    *,
    group_library_id: str,
    user_library_id: str,
    user_root_collection: str = DEFAULT_USER_ROOT_COLLECTION,
) -> list[BibSpec]:
    """Every bibliography the repo declares, read from where it is declared.

    Args:
        project_root: The torchcell checkout.
        group_library_id: The torchcell group library.
        user_library_id: The personal library that the notes-tex documents and
            the Dendron scope union in.
        user_root_collection: The personal collection tree for ``library``.
    """
    root = Path(project_root)
    specs = [
        BibSpec(
            name=PAPER_BIB_NAME,
            scope=BibScope(
                group_library_id=group_library_id, group_collection=PAPER_COLLECTION_KEY
            ),
            origin="paper/nature-biotech/zotero_export_bib.py",
        )
    ]
    for makefile in sorted((root / "notes-tex").glob("*/Makefile")):
        group_collection, user_collection = parse_makefile_collections(makefile)
        if not group_collection:
            continue
        slug = validate_bib_name(makefile.parent.name)
        specs.append(
            BibSpec(
                name=slug,
                scope=BibScope(
                    group_library_id=group_library_id,
                    group_collection=group_collection,
                    user_library_id=user_library_id if user_collection else None,
                    user_collection=user_collection or None,
                ),
                origin=str(makefile.relative_to(root)),
            )
        )
    specs.append(
        BibSpec(
            name=LIBRARY_BIB_NAME,
            scope=BibScope(
                group_library_id=group_library_id,
                user_library_id=user_library_id,
                user_root_collection=user_root_collection,
            ),
            origin="scripts/lit_bib.py",
        )
    )
    return specs


def fetch_scope_entries(
    scope: BibScope, group: ZoteroLibrary, user: ZoteroLibrary
) -> list[dict[str, Any]]:
    """Pull the entries a scope denotes, dispatching on its shape.

    A group collection paired with a personal collection is the notes-tex union
    (personal wins on a shared key, as in :func:`fetch_paired_collection_entries`);
    a personal root collection is the Dendron union; a lone group collection is
    the manuscript export.
    """
    if scope.user_collection is not None:
        if scope.group_collection is None:
            raise ValueError("a personal collection needs a group collection to pair")
        return fetch_paired_collection_entries(
            group,
            user,
            group_collection=scope.group_collection,
            user_collection=scope.user_collection,
        )
    if scope.user_root_collection is not None:
        return fetch_union_bibtex_entries(
            group, user, user_root_collection=scope.user_root_collection
        )
    if scope.group_collection is None:
        return fetch_bibtex_entries(group)
    if re.fullmatch(r"[A-Z0-9]{8}", scope.group_collection):
        return fetch_bibtex_entries(group, collection_key=scope.group_collection)
    return fetch_bibtex_entries(group, collection=scope.group_collection)


def _header(spec: BibSpec, n_entries: int) -> str:
    """Generated-file banner. No timestamp, so unchanged content hashes the same."""
    scope = spec.scope
    parts = [f"group {scope.group_library_id}/{scope.group_collection or '*'}"]
    if scope.user_collection:
        parts.append(f"personal {scope.user_library_id}/{scope.user_collection}")
    if scope.user_root_collection:
        parts.append(
            f"personal {scope.user_library_id}/{scope.user_root_collection}/** (tree)"
        )
    return (
        "% GENERATED by torchcell.literature.bib_store -- do not hand-edit.\n"
        f"% name: {spec.name}  entries: {n_entries}\n"
        f"% scope: {' + '.join(parts)}\n"
        f"% declared in: {spec.origin}\n"
        "% served by tc-lit at /bib/" + spec.name + "; verify X-Artifact-SHA256.\n\n"
    )


def write_bib(
    store_dir: Path, spec: BibSpec, entries: list[dict[str, Any]], *, suffix: str = ""
) -> Path:
    """Write ``<store_dir>/<name>.bib<suffix>`` with the banner; return its path."""
    if not entries:
        raise RuntimeError(
            f"refusing to write {spec.name}.bib: Zotero returned 0 entries for "
            f"{spec.scope.model_dump(exclude_none=True)}"
        )
    path = store_dir / f"{spec.name}.bib{suffix}"
    write_bib_entries(path, entries)
    path.write_text(
        _header(spec, len(entries)) + path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return path


def export_bib_store(
    mirror_root: str | Path,
    specs: list[BibSpec],
    group: ZoteroLibrary,
    user: ZoteroLibrary,
    *,
    generated_at: str | None = None,
) -> BibStoreManifest:
    """Pull every spec, write the files, and write the manifest.

    A store manifest already on disk is replaced wholesale by the specs given, so
    a spec removed from the repo stops being served on the next export. Every
    file is pulled and written under a ``.part`` suffix first; the served files
    and the manifest are swapped in only once every spec succeeded, so a pull that
    fails part-way leaves the previous store intact and consistent, and the
    manifest never advertises a hash the file beside it does not have.
    """
    store_dir = bib_store_dir(mirror_root)
    store_dir.mkdir(parents=True, exist_ok=True)
    stamp = generated_at or datetime.now(UTC).isoformat()
    staged: list[tuple[BibSpec, Path, int]] = []
    for spec in specs:
        validate_bib_name(spec.name)
        entries = fetch_scope_entries(spec.scope, group, user)
        path = write_bib(store_dir, spec, entries, suffix=".part")
        staged.append((spec, path, len(entries)))
        log.info("bib_store: %s -> %d entries", spec.name, len(entries))

    records: list[BibRecord] = []
    for spec, part, n_entries in staged:
        final = part.with_suffix("")  # strip .part -> <name>.bib
        part.replace(final)
        records.append(
            BibRecord(
                name=spec.name,
                path=final.name,
                bytes=final.stat().st_size,
                sha256=sha256_file(final),
                n_entries=n_entries,
                scope=spec.scope,
                origin=spec.origin,
                generated_at=stamp,
            )
        )
    manifest = BibStoreManifest(bibs=records, generated_at=stamp)
    (store_dir / BIB_STORE_MANIFEST).write_text(manifest.model_dump_json(indent=2))
    log.info("bib_store: wrote %d bibliographies -> %s", len(records), store_dir)
    return manifest
