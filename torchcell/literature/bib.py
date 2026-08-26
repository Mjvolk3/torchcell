# torchcell/literature/bib.py
# [[torchcell.literature.bib]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/bib.py
# Test file: tests/torchcell/literature/test_bib.py

r"""Generate the Dendron BibTeX file from Zotero -- never hand-edit ``bib.bib``.

The notes bibliography (``notes/assets/bib/bib.bib``, plus its publish twin) is the
citation source for every Dendron note rendered through pandoc. This module keeps it
in sync with the TorchCell Zotero group by pulling entries over the **Zotero Web API
via pyzotero** (``format="bibtex"``), so it works headless -- no Zotero desktop, no
Better BibTeX, no localhost endpoint. That distinguishes it from
``paper/nature-biotech/zotero_export_bib.py``, which regenerates the *manuscript's*
``references.bib`` through the BBT local export and therefore only runs on a machine
with Zotero open.

**The entry key is the mirror key.** The Web API's BibTeX export emits Zotero 7's
native ``citationKey``, which is exactly what :func:`_resolve_citation_key` uses to
name artifact directories under ``<DATA_ROOT>/torchcell-library/<citation_key>/``.
So a ``\cite{brettnerUltraHighthroughputMassively2024}`` in a note and the OCR
markdown it is citing are addressed by the same string, with no mapping table.

**Sync is add-only, never truncating.** ``bib.bib`` predates the Zotero group and
holds hundreds of entries the group does not (older exports from a personal
library). A wholesale overwrite would silently break every note citing one of them,
so :func:`sync_bib_file` appends Zotero-only keys and leaves every existing entry
byte-identical; nothing is ever dropped. Pass ``update_existing=True`` to also
refresh shared keys field-wise -- see :func:`merge_bib_entries` for why that is opt
-in. :func:`plan_bib_sync` is the read-only preview.
"""

import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import bibtexparser
from bibtexparser.bibdatabase import BibDatabase
from bibtexparser.bwriter import BibTexWriter
from pydantic import BaseModel, Field

from torchcell.literature.zotero import (
    ZoteroLibrary,
    _resolve_citation_key,
    with_zotero_retry,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# The notes bibliography and its publish-pipeline twin. Both are generated; the
# publish copy exists because the pandoc publish scripts resolve paths relative to
# `notes/assets/publish/`.
BIB_RELPATHS: tuple[str, ...] = (
    "notes/assets/bib/bib.bib",
    "notes/assets/publish/bib/bib.bib",
)

# bibtexparser's structural key for the citation key of an entry.
_ID = "ID"

# Item types that are not citable works. Zotero's BibTeX export emits an entry for
# every item it is handed -- including PDF **attachments**, which come through as
# `@misc{noauthor_notitle_nodate...}` stubs with no author, title, or date. On the
# TorchCell group that is 89 of 185 exported entries, so the export is filtered
# against the citable item set rather than trusted wholesale.
NON_CITABLE_ITEM_TYPES = frozenset({"attachment", "note", "annotation"})


class BibEntryChange(BaseModel):
    """What happened to one citation key on a sync pass."""

    citation_key: str
    mode: str = Field(description="added | updated | unchanged")


class BibFileSyncReport(BaseModel):
    """Outcome of syncing one ``.bib`` file against a Zotero pull."""

    path: str
    n_before: int
    n_after: int
    n_zotero: int
    changes: list[BibEntryChange] = Field(default_factory=list)
    preserved: list[str] = Field(
        default_factory=list,
        description="Keys kept verbatim because Zotero does not hold them.",
    )
    written: bool = False

    def by_mode(self, mode: str) -> list[BibEntryChange]:
        """Changes with a given mode."""
        return [c for c in self.changes if c.mode == mode]

    def summary(self) -> str:
        """One-line tally for logs."""
        counts = {m: len(self.by_mode(m)) for m in ("added", "updated", "unchanged")}
        tally = " ".join(f"{m}={counts[m]}" for m in sorted(counts))
        return (
            f"{self.path}: {self.n_before} -> {self.n_after} entries | {tally} "
            f"preserved={len(self.preserved)} (zotero={self.n_zotero})"
        )


# Characters permitted in a citation key. A key outside this set (e.g. a literal
# `$` from a maths-laden title) makes the whole file unparseable to pandoc -- it
# aborts on the entry, so ONE bad key silently costs every citation in the file.
_INVALID_KEY_CHARS = re.compile(r"[^A-Za-z0-9_:.\-]")


def sanitize_citation_key(key: str) -> str:
    """Strip characters that would make a citation key unparseable to pandoc."""
    return _INVALID_KEY_CHARS.sub("", key)


def _is_citable_entry(entry: dict[str, Any]) -> bool:
    """True for a real work, False for an attachment stub.

    Zotero exports an entry for every item it is handed, including PDF
    **attachments**, which arrive as `@misc{noauthor_notitle_nodate...}` carrying no
    author, title, or year. Judging by content rather than by matching that magic
    key means the rule survives Zotero changing the placeholder, and it does not
    discard a real paper whose exported key happens to differ from its stored one.
    """
    return bool(
        (entry.get("title") or "").strip() or (entry.get("author") or "").strip()
    )


def citable_citation_keys(
    lib: ZoteroLibrary,
    collection: str | None = None,
    *,
    collection_key: str | None = None,
) -> set[str]:
    """Citation keys of the real works in a library/collection (no attachments).

    Resolved with :func:`_resolve_citation_key`, the same function that names
    artifact directories in the mirror -- so the bibliography and the OCR mirror are
    keyed identically by construction.
    """
    if collection_key is not None:
        items: list[dict[str, Any]] = with_zotero_retry(
            lambda: lib.zot.everything(lib.zot.collection_items(collection_key))
        )
    elif collection is None:
        items = with_zotero_retry(lambda: lib.zot.everything(lib.zot.items()))
    else:
        coll_key = lib.collection_key(collection)
        items = with_zotero_retry(
            lambda: lib.zot.everything(lib.zot.collection_items(coll_key))
        )
    return {
        _resolve_citation_key(item)
        for item in items
        if item["data"].get("itemType") not in NON_CITABLE_ITEM_TYPES
    }


def fetch_paired_collection_entries(
    group: ZoteroLibrary,
    user: ZoteroLibrary,
    *,
    group_collection: str,
    user_collection: str,
) -> list[dict[str, Any]]:
    """Union ONE group collection with ONE personal collection.

    This is the per-document bibliography: a ``notes-tex/<slug>/references.bib``
    cites the reading for that slug, which lives in a same-named collection in each
    library (e.g. group ``microbe-perturb-seq`` and personal
    ``torchcell/torchcell-topics/microbe-perturb-seq``). The whole-tree union
    (:func:`fetch_union_bibtex_entries`) is far too wide for that -- it would drag
    every torchcell paper into a single document's bibliography.

    Personal wins on a shared key, matching the whole-tree union.

    Args:
        group: The torchcell group library.
        user: The personal library.
        group_collection: Collection name or key in the group.
        user_collection: Collection name or key in the personal library.

    Returns:
        Deduplicated entries from the two collections.
    """
    by_key: dict[str, dict[str, Any]] = {}
    for lib, name, label in (
        (group, group_collection, "group"),
        (user, user_collection, "personal"),
    ):
        entries = fetch_bibtex_entries(lib, **_collection_selector(lib, name))
        for entry in entries:
            if key := entry.get(_ID):
                by_key[key] = entry  # personal is applied second, so it wins
        log.info("bib: %s/%s -> %d entries", label, name, len(entries))
    log.info("bib: paired-collection union -> %d entries", len(by_key))
    return list(by_key.values())


def _collection_selector(lib: ZoteroLibrary, name_or_key: str) -> dict[str, str]:
    """Address a collection by key when it looks like one, else by name.

    Zotero collection keys are 8 uppercase alphanumerics. Accepting either means a
    caller can pin the stable key (safe across renames) or use the readable name.
    """
    if re.fullmatch(r"[A-Z0-9]{8}", name_or_key):
        return {"collection_key": name_or_key}
    return {"collection": name_or_key}


def fetch_union_bibtex_entries(
    group: ZoteroLibrary,
    user: ZoteroLibrary | None = None,
    *,
    user_root_collection: str | None = None,
) -> list[dict[str, Any]]:
    """BibTeX for the group UNION the personal ``torchcell/*`` tree.

    The notes bibliography cites more widely than the manuscript does, and new
    reading lands in the personal library first -- a group-only pull therefore
    cannot see it. (The manuscript's ``references.bib`` is deliberately group-only,
    so every published citation is recoverable from the shared library.)

    Personal takes precedence on a shared citation key: it is where the entry is
    curated first, and it is the library the mirror is keyed from, so preferring it
    keeps ``@key`` and the artifact directory in agreement.

    Args:
        group: The torchcell group library.
        user: The personal library; ``None`` pulls group only.
        user_root_collection: Personal collection tree to scope to (recursively).
            Required when ``user`` is given -- the personal library holds thousands
            of items that are not torchcell work.

    Returns:
        Deduplicated bibtexparser entries, personal winning on a key collision.
    """
    entries = fetch_bibtex_entries(group)
    if user is None:
        return entries
    if user_root_collection is None:
        raise ValueError(
            "user_root_collection is required when a user library is given"
        )

    by_key = {e[_ID]: e for e in entries if e.get(_ID)}
    n_group = len(by_key)
    for sub in _collection_tree(user, user_root_collection):
        for entry in fetch_bibtex_entries(user, collection_key=sub):
            key = entry.get(_ID)
            if key:
                by_key[key] = entry  # personal wins
    log.info(
        "bib: union -> %d entries (%d group, %d added or overridden from personal)",
        len(by_key),
        n_group,
        len(by_key) - n_group,
    )
    return list(by_key.values())


def _collection_tree(lib: ZoteroLibrary, root_collection: str) -> list[str]:
    """Keys of a collection and every collection nested beneath it.

    ``collection_items`` returns only direct members, so a nested collection such as
    ``torchcell/torchcell-topics/microbe-perturb-seq`` is invisible without walking
    the tree -- which is exactly where new reading is filed.
    """
    from collections import defaultdict

    root_key = lib.collection_key(root_collection)
    children: dict[str, list[str]] = defaultdict(list)
    for c in lib.zot.everything(lib.zot.collections()):
        children[c["data"].get("parentCollection") or ""].append(c["key"])

    def walk(key: str) -> list[str]:
        out = [key]
        for kid in children.get(key, []):
            out += walk(kid)
        return out

    tree = walk(root_key)
    log.info("bib: personal tree '%s' -> %d collections", root_collection, len(tree))
    return tree


def fetch_bibtex_entries(
    lib: ZoteroLibrary,
    collection: str | None = None,
    *,
    collection_key: str | None = None,
) -> list[dict[str, Any]]:
    """Pull BibTeX entries for the citable works of a library or collection.

    Args:
        lib: Connected Zotero library.
        collection: Restrict to one collection by name; ``None`` pulls the whole
            library (every citable work, across all collections).
        collection_key: Restrict to one collection by *key*, skipping the
            name lookup. Used when walking a collection tree, where the children
            are already known by key and may share names across libraries.

    Returns:
        bibtexparser entry dicts keyed by Zotero's native ``citationKey``, filtered
        to the citable item set -- attachment stubs
        (``@misc{noauthor_notitle_nodate...}``) are dropped, never written to a
        bibliography.
    """
    if collection_key is not None:
        db: BibDatabase = with_zotero_retry(
            lambda: lib.zot.everything(
                lib.zot.collection_items(collection_key, format="bibtex")
            )
        )
    elif collection is None:
        db = with_zotero_retry(
            lambda: lib.zot.everything(lib.zot.items(format="bibtex"))
        )
    else:
        coll_key = lib.collection_key(collection)
        db = with_zotero_retry(
            lambda: lib.zot.everything(
                lib.zot.collection_items(coll_key, format="bibtex")
            )
        )
    # An EMPTY collection comes back as a plain (empty) list rather than a
    # BibDatabase -- pyzotero only parses BibTeX when there is a body to parse.
    # Several collections in the personal tree hold only sub-collections, so this
    # is the normal case, not an error.
    raw = db.entries if hasattr(db, "entries") else list(db)
    entries = [e for e in raw if _is_citable_entry(e)]

    # Zotero's BibTeX export does NOT always reproduce an item's stored
    # `citationKey`: it may regenerate the key from the title, or drop a
    # disambiguating suffix. Measured: identical on all 102 group items, but
    # 4 of 53 disagree in personal `microbe-perturb-seq`. Filtering on key
    # membership therefore DROPS those papers silently, so the filter above is
    # content-based instead. The disagreement is still worth surfacing, because
    # a note citing the stored key will not resolve against the exported one.
    stored = citable_citation_keys(
        lib, collection=collection, collection_key=collection_key
    )
    exported = {e.get(_ID) for e in entries}
    if drifted := stored - exported:
        log.warning(
            "bib: %d stored citationKey(s) differ from the BibTeX export and will "
            "not resolve if cited by the stored key: %s",
            len(drifted),
            ", ".join(sorted(drifted)),
        )
    # Sanitize INCOMING keys too, not just ones read off disk: a citation key is
    # free text in Zotero, and a single `$` makes the whole .bib unparseable to
    # pandoc -- which silently kills every citation in the file, not just this one.
    for entry in entries:
        key = entry.get(_ID)
        if key and (clean := sanitize_citation_key(key)) != key:
            log.warning(
                "bib: Zotero citation key %r is not a valid BibTeX key; writing %r "
                "(fix it in Zotero so the two agree)",
                key,
                clean,
            )
            entry[_ID] = clean
    log.info(
        "bib: pulled %d citable entries from Zotero (%s; %d non-citable dropped)",
        len(entries),
        collection or collection_key or "whole library",
        len(raw) - len(entries),
    )
    return entries


def read_bib_entries(path: str | Path) -> list[dict[str, Any]]:
    """Parse a ``.bib`` file into bibtexparser entry dicts (empty if absent).

    Citation keys are sanitized on read, so a legacy key carrying an invalid
    character cannot propagate into the generated file. Renames are logged at
    WARNING: a key that some note actually cites would need that note updated.
    """
    path = Path(path)
    if not path.exists():
        return []
    parser = bibtexparser.bparser.BibTexParser(common_strings=True)
    parser.ignore_nonstandard_types = False
    db = bibtexparser.loads(path.read_text(encoding="utf-8", errors="replace"), parser)
    entries: list[dict[str, Any]] = list(db.entries)
    for entry in entries:
        key = entry.get(_ID)
        if key and (clean := sanitize_citation_key(key)) != key:
            log.warning("bib: sanitized invalid citation key %r -> %r", key, clean)
            entry[_ID] = clean
    return entries


def write_bib_entries(path: str | Path, entries: Sequence[dict[str, Any]]) -> None:
    """Write entries to a ``.bib`` file, sorted by citation key for a stable diff."""
    db = BibDatabase()
    db.entries = sorted(entries, key=lambda e: e.get(_ID, "").lower())
    writer = BibTexWriter()
    writer.indent = "  "
    writer.order_entries_by = None  # already sorted; keep our ordering
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(writer.write(db), encoding="utf-8")


def _dedupe_by_key(entries: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index entries by citation key, last occurrence winning.

    Both the existing ``bib.bib`` and the Zotero group carry a small number of
    duplicate keys (dup-twin items). Indexing collapses them deterministically
    rather than emitting a bibliography pandoc would warn on.
    """
    return {e[_ID]: e for e in entries if e.get(_ID)}


def merge_bib_entries(
    existing: Sequence[dict[str, Any]],
    incoming: Sequence[dict[str, Any]],
    *,
    update_existing: bool = False,
) -> tuple[list[dict[str, Any]], list[BibEntryChange], list[str]]:
    """Merge a Zotero pull into existing entries.

    Args:
        existing: Entries already in the ``.bib`` file.
        incoming: Entries pulled from Zotero.
        update_existing: How to treat a key both sides hold.

            ``False`` (default) -- **add-only**: the existing entry is left
            untouched and only Zotero-only keys are added. Measured against the
            live bibliography this renders byte-identically to the current file,
            so pulling new papers can never perturb an existing citation.

            ``True`` -- **field-level update**: Zotero's value wins per field, but
            fields only the local entry has are kept. That preserves the biblatex
            ``shortjournal`` entries the legacy export carries, which the Nature CSL
            uses for abbreviated journal names ("Nat Methods"); a wholesale
            overwrite drops them and silently expands every such name.

    Returns:
        ``(merged_entries, changes, preserved_keys)`` where ``preserved_keys`` are
        the existing keys Zotero does not hold -- kept verbatim so a note citing a
        pre-Zotero reference never breaks.
    """
    have = _dedupe_by_key(existing)
    new = _dedupe_by_key(incoming)

    changes: list[BibEntryChange] = []
    merged: dict[str, dict[str, Any]] = dict(have)
    for key, entry in new.items():
        if key not in have:
            merged[key] = entry
            changes.append(BibEntryChange(citation_key=key, mode="added"))
            continue
        if not update_existing:
            changes.append(BibEntryChange(citation_key=key, mode="unchanged"))
            continue
        candidate = {**have[key], **entry}
        merged[key] = candidate
        mode = "unchanged" if candidate == have[key] else "updated"
        changes.append(BibEntryChange(citation_key=key, mode=mode))

    preserved = sorted(set(have) - set(new))
    return list(merged.values()), changes, preserved


def plan_bib_sync(
    path: str | Path,
    incoming: Sequence[dict[str, Any]],
    *,
    update_existing: bool = False,
) -> BibFileSyncReport:
    """Classify what a sync would do to one ``.bib`` file, writing nothing."""
    existing = read_bib_entries(path)
    merged, changes, preserved = merge_bib_entries(
        existing, incoming, update_existing=update_existing
    )
    return BibFileSyncReport(
        path=str(path),
        n_before=len(_dedupe_by_key(existing)),
        n_after=len(merged),
        n_zotero=len(incoming),
        changes=changes,
        preserved=preserved,
        written=False,
    )


def sync_bib_file(
    path: str | Path,
    incoming: Sequence[dict[str, Any]],
    *,
    dry_run: bool = False,
    update_existing: bool = False,
) -> BibFileSyncReport:
    """Merge a Zotero pull into one ``.bib`` file and write it.

    Args:
        path: Target ``.bib`` file.
        incoming: Entries from :func:`fetch_bibtex_entries`.
        dry_run: Classify only; write nothing.
        update_existing: Refresh shared keys field-wise from Zotero; see
            :func:`merge_bib_entries`.

    Returns:
        The per-file report.

    Raises:
        RuntimeError: If the merge would shrink the file. The merge is additive by
            construction, so a shrink means a parse failure upstream -- refuse
            rather than write a truncated bibliography over a good one.
    """
    report = plan_bib_sync(path, incoming, update_existing=update_existing)
    if report.n_after < report.n_before:
        raise RuntimeError(
            f"refusing to write {path}: merge would drop "
            f"{report.n_before - report.n_after} entries "
            f"({report.n_before} -> {report.n_after}); the merge is additive, so "
            "this indicates a parse failure on the existing file."
        )
    if dry_run:
        return report

    existing = read_bib_entries(path)
    merged, _, _ = merge_bib_entries(
        existing, incoming, update_existing=update_existing
    )
    write_bib_entries(path, merged)
    report.written = True
    log.info("bib: wrote %d entries -> %s", report.n_after, path)
    return report
