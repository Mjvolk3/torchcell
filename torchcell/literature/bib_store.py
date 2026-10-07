# torchcell/literature/bib_store.py
# [[torchcell.literature.bib_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/bib_store.py
# Test file: tests/torchcell/literature/test_bib_store.py

r"""Named bibliographies materialized into the mirror, for tc-lit to serve.

Both LaTeX bibliography flows (``notes-tex/<group>/<slug>/references.bib`` via
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
- one per ``notes-tex/<group>/<slug>/`` -- the ``ZOTERO_COLLECTION`` and
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

**Collections are addressed by KEY.** Every declaration the repo makes is a Zotero
collection key (``PAPER_COLLECTION_KEY``, and the Makefile values ``build_bib.py`` sends
to Better BibTeX), so :class:`BibScope` holds keys and refuses anything else by value.
A key is never inferred from the shape of a name.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

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

# A served file that no longer belongs to a declared spec is moved under
# `_bib/_retired/<generated_at>/`, never deleted.
BIB_STORE_RETIRED_SUBDIR = "_retired"

# The exporter's stamp, ``datetime.now(UTC).isoformat()``; it names a directory.
_GENERATED_AT_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{6})?\+00:00")

# A Zotero collection key: 8 upper-case letters or digits. Used only to VALIDATE a
# value declared as a key, never to decide whether a value is a key.
_COLLECTION_KEY_RE = re.compile(r"[A-Z0-9]{8}")

# `ZOTERO_COLLECTION := FE8DQKUH  # optional comment` in a notes-tex document Makefile.
# The value (group 2) runs to the end of the line; a `#` comment is cut off after.
_MAKEFILE_VAR_RE = re.compile(
    r"^\s*(ZOTERO_COLLECTION|ZOTERO_PERSONAL_COLLECTION)\s*[:?]?=(.*)$"
)


class BibScope(BaseModel):
    """Which Zotero collections a bibliography is the export of.

    Exactly one of three shapes: a single group collection (the manuscript); a
    group collection paired with one personal collection (a notes-tex document);
    or the group library unioned with a personal collection tree (the Dendron
    scope). ``group_collection`` and ``user_collection`` are collection KEYS, as
    every repo declaration states them; ``user_root_collection`` is a NAME.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    group_library_id: str
    group_collection: str | None = Field(
        default=None,
        description="One group collection, by KEY; None = the whole group.",
    )
    user_library_id: str | None = None
    user_collection: str | None = Field(
        default=None,
        description="One personal collection, by KEY, paired with the group one.",
    )
    user_root_collection: str | None = Field(
        default=None,
        description="A personal collection tree, by NAME, unioned in recursively.",
    )

    @field_validator("group_collection", "user_collection")
    @classmethod
    def _require_collection_key(cls, value: str | None) -> str | None:
        """Refuse a value that is not a collection key, naming it."""
        if value is not None and not _COLLECTION_KEY_RE.fullmatch(value):
            raise ValueError(
                f"not a Zotero collection key (8 upper-case letters or digits): "
                f"{value!r}; bibliography scopes address collections by key"
            )
        return value


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

    Either may be empty: a document that cites nothing declares neither. A trailing
    ``# comment`` is cut off the value, as make does. A value that is not one
    collection key (a name, two words) is refused with ``ValueError`` naming the
    Makefile and the variable, so a document is never silently left without its
    bibliography and the refusal points at the file to fix.
    """
    values = {"ZOTERO_COLLECTION": "", "ZOTERO_PERSONAL_COLLECTION": ""}
    for line in makefile.read_text().splitlines():
        match = _MAKEFILE_VAR_RE.match(line)
        if match:
            value = match.group(2).split("#", 1)[0].strip()
            if value and not _COLLECTION_KEY_RE.fullmatch(value):
                raise ValueError(
                    f"{makefile}: {match.group(1)} must be one Zotero collection key "
                    f"(8 upper-case letters or digits), got {value!r}"
                )
            values[match.group(1)] = value
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
    # Documents live at notes-tex/<group>/<slug>/Makefile. The bibliography keeps
    # the slug as its name, so `/bib/<slug>` and `make bib-pull` (which asks for
    # $(DOC), the directory name) are unchanged by which group a document sits in.
    for makefile in sorted((root / "notes-tex").glob("*/*/Makefile")):
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
    the manuscript export. Collections are sent as the keys the scope holds.
    """
    if scope.user_collection is not None:
        if scope.group_collection is None:
            raise ValueError("a personal collection needs a group collection to pair")
        return fetch_paired_collection_entries(
            group,
            user,
            group_collection=scope.group_collection,
            user_collection=scope.user_collection,
            as_keys=True,
        )
    if scope.user_root_collection is not None:
        return fetch_union_bibtex_entries(
            group, user, user_root_collection=scope.user_root_collection
        )
    if scope.group_collection is None:
        return fetch_bibtex_entries(group)
    return fetch_bibtex_entries(group, collection_key=scope.group_collection)


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


def validate_generated_at(stamp: str) -> str:
    """Return ``stamp`` if it is a UTC ISO timestamp as the exporter writes it.

    ``datetime.now(UTC).isoformat()`` gives ``YYYY-MM-DDTHH:MM:SS[.ffffff]+00:00``;
    the stamp names a ``_retired/`` directory, so anything else (a path separator,
    ``..``, or a well-shaped but impossible date such as month 13) is refused by
    value before the store is touched.
    """
    if not (_GENERATED_AT_RE.fullmatch(stamp) and _parses_as_datetime(stamp)):
        raise ValueError(
            "generated_at must be a UTC ISO timestamp "
            f"(YYYY-MM-DDTHH:MM:SS[.ffffff]+00:00), got {stamp!r}"
        )
    return stamp


def _parses_as_datetime(stamp: str) -> bool:
    """True when ``datetime.fromisoformat`` accepts ``stamp`` (a real date and time)."""
    try:
        datetime.fromisoformat(stamp)
    except ValueError:
        return False
    return True


def _carried_records(
    store_dir: Path, declared: list[BibSpec], exported: set[str]
) -> dict[str, BibRecord]:
    """Previous-manifest records of declared specs this run does not re-export.

    A full export (every declared spec re-exported) carries nothing and never reads
    the previous manifest, so a full run repairs a store whose manifest is damaged.
    On a subset run, a previous manifest that does not validate is refused naming
    it; a record is carried only when its file is on disk with the manifest's
    sha256, otherwise the export is refused by name, because carrying it would
    advertise a file the store does not have. No previous manifest carries nothing.
    """
    manifest_path = store_dir / BIB_STORE_MANIFEST
    if {spec.name for spec in declared} <= exported or not manifest_path.is_file():
        return {}
    try:
        previous = BibStoreManifest.model_validate_json(manifest_path.read_text())
    except ValidationError as err:
        raise ValueError(
            f"cannot carry bibliographies forward: the previous manifest "
            f"{manifest_path} does not validate; run a full export"
        ) from err
    carried: dict[str, BibRecord] = {}
    for spec in declared:
        record = previous.get(spec.name)
        if spec.name in exported or record is None:
            continue
        target = store_dir / record.path
        if not target.is_file():
            raise ValueError(
                f"cannot carry {spec.name} forward: the previous manifest lists "
                f"{record.path} but it is absent on disk; run a full export"
            )
        if sha256_file(target) != record.sha256:
            raise ValueError(
                f"cannot carry {spec.name} forward: {record.path} no longer has the "
                "sha256 the previous manifest pins; run a full export"
            )
        carried[spec.name] = record
    return carried


def export_bib_store(
    mirror_root: str | Path,
    specs: list[BibSpec],
    group: ZoteroLibrary,
    user: ZoteroLibrary,
    *,
    declared: list[BibSpec],
    generated_at: str | None = None,
) -> BibStoreManifest:
    """Pull ``specs``, write the files, and write the manifest.

    ``declared`` is EVERY spec the repo declares (:func:`discover_bib_specs`);
    ``specs`` is the subset to re-export now (all of them on the nightly run, one
    or more on a ``--name`` run) and must be drawn from it. The new manifest lists,
    in declared order, a fresh record for each exported spec and the previous
    manifest's record, unchanged, for each declared spec not re-exported (refused by
    name if that record's file is missing or its sha256 drifted). A spec removed
    from the repo is therefore unlisted on any run, and a subset run leaves the
    other served bibliographies exactly as they were.

    Every file is pulled and written under a ``.part`` suffix first; the served
    files and the manifest are swapped in only once every spec succeeded, so a
    pull that fails part-way leaves the previous store intact and consistent. A
    failed export removes every ``.part`` file it staged before re-raising.

    After a successful export, every ``<name>.bib`` whose name no declared spec
    carries, and every leftover ``*.bib.part``, is moved to
    ``_retired/<generated_at>/`` with a warning naming it; nothing is deleted.
    """
    stamp = validate_generated_at(generated_at or datetime.now(UTC).isoformat())
    declared_names = [validate_bib_name(spec.name) for spec in declared]
    undeclared = sorted({spec.name for spec in specs} - set(declared_names))
    if undeclared:
        raise ValueError(f"exported specs not declared by the repo: {undeclared}")
    store_dir = bib_store_dir(mirror_root)
    store_dir.mkdir(parents=True, exist_ok=True)
    exported = {spec.name for spec in specs}
    carried = _carried_records(store_dir, declared, exported)
    staged: list[tuple[BibSpec, Path, int]] = []
    attempted: list[Path] = []
    try:
        for spec in specs:
            validate_bib_name(spec.name)
            attempted.append(store_dir / f"{spec.name}.bib.part")
            entries = fetch_scope_entries(spec.scope, group, user)
            path = write_bib(store_dir, spec, entries, suffix=".part")
            staged.append((spec, path, len(entries)))
            log.info("bib_store: %s -> %d entries", spec.name, len(entries))
    except BaseException:
        for part in attempted:
            part.unlink(missing_ok=True)
        raise

    fresh: dict[str, BibRecord] = {}
    for spec, part, n_entries in staged:
        final = part.with_suffix("")  # strip .part -> <name>.bib
        part.replace(final)
        fresh[spec.name] = BibRecord(
            name=spec.name,
            path=final.name,
            bytes=final.stat().st_size,
            sha256=sha256_file(final),
            n_entries=n_entries,
            scope=spec.scope,
            origin=spec.origin,
            generated_at=stamp,
        )
    records = [
        fresh.get(name) or carried[name]
        for name in declared_names
        if name in fresh or name in carried
    ]
    manifest = BibStoreManifest(bibs=records, generated_at=stamp)
    _write_manifest(store_dir, manifest)
    log.info(
        "bib_store: wrote %d bibliographies (%d exported, %d carried) -> %s",
        len(records),
        len(fresh),
        len(carried),
        store_dir,
    )
    _retire_undeclared(store_dir, {f"{name}.bib" for name in declared_names}, stamp)
    return manifest


def _write_manifest(store_dir: Path, manifest: BibStoreManifest) -> None:
    """Write ``manifest.json`` atomically: a temporary file beside it, then
    ``os.replace``, so a failed write never leaves a truncated manifest.
    """
    final = store_dir / BIB_STORE_MANIFEST
    temp = store_dir / f"{BIB_STORE_MANIFEST}.tmp"
    temp.write_text(manifest.model_dump_json(indent=2))
    os.replace(temp, final)


def _retired_target(retired_dir: Path, name: str) -> Path:
    """``<retired_dir>/<name>``, or ``<name>.<n>`` with the smallest free ``n >= 1``
    when an earlier run with the same stamp already retired a file of that name, so
    a retired file is never overwritten.
    """
    target = retired_dir / name
    n = 0
    while target.exists():
        n += 1
        target = retired_dir / f"{name}.{n}"
    return target


def _retire_undeclared(store_dir: Path, declared: set[str], stamp: str) -> list[Path]:
    """Move undeclared ``*.bib`` files and leftover ``*.bib.part`` files aside.

    A ``<name>.bib`` stays when ``<name>`` is a spec the repo declares (served or
    not); every other one, and every ``*.bib.part`` (staging a successful export
    has already renamed), moves to ``<store_dir>/_retired/<stamp>/`` with a warning
    naming it; a name already retired under the same stamp gets a numeric suffix
    rather than being overwritten. Returns the new paths.
    """
    retired_dir = store_dir / BIB_STORE_RETIRED_SUBDIR / stamp
    moved: list[Path] = []
    for path in sorted([*store_dir.glob("*.bib"), *store_dir.glob("*.bib.part")]):
        if path.name in declared:
            continue
        retired_dir.mkdir(parents=True, exist_ok=True)
        target = path.replace(_retired_target(retired_dir, path.name))
        reason = (
            "is a leftover staging file"
            if path.name.endswith(".part")
            else "is not declared by any spec in the repo"
        )
        log.warning(
            "bib_store: %s %s; moved %s -> %s",
            path.name.split(".bib", 1)[0],
            reason,
            path,
            target,
        )
        moved.append(target)
    return moved
