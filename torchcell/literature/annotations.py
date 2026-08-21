# torchcell/literature/annotations.py
# [[torchcell.literature.annotations]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/annotations.py
# Test file: tests/torchcell/literature/test_annotations.py

"""Capture Zotero highlights, comments and notes into the literature mirror.

The OCR mirror holds what the *paper* says; this holds what **we** said about it.
Reading notes live in Zotero as `annotation` and `note` items, which the capture
pipeline never touched -- so the comments written while reading were unreachable
from the machine the writing happens on. This module pulls them per paper into
``annotations.json`` (structured) and ``annotations.md`` (readable), alongside
``paper.md``.

**Scope is the same rule the capture uses:** the personal ``torchcell/*`` collection
tree (recursively) UNION the torchcell group. Annotations elsewhere in the personal
library are out of scope and never fetched.

**Provenance is per annotation, because the same paper is annotated in both places.**
A paper can exist in the personal library *and* the group -- Zotero cannot merge
across libraries -- and notes may be written on either copy. Every record therefore
carries the libraries it was found in. Byte-identical content found in both is
emitted **once** with both sources listed, rather than duplicated; content that
differs is kept as separate records. So "where did this come from" is always
answerable, and copying a paper into the group does not double its notes.

Zotero's own vocabulary is kept: a *highlight* is the paper's words (
``annotationText``), a *comment* is ours (``annotationComment``), and a *note* is a
free-standing item. Every comment observed so far is anchored to a highlight, and
they are stored together for that reason -- a comment without its anchor is unusable.
"""

import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from torchcell.literature.zotero import ZoteroLibrary, _resolve_citation_key

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

ANNOTATIONS_JSON = "annotations.json"
ANNOTATIONS_MD = "annotations.md"

# Zotero's default highlight palette. Kept as data, not rendered to emoji: the
# markdown states the colour name so it is greppable, and the JSON keeps the hex.
COLOR_NAMES: dict[str, str] = {
    "#ffd400": "yellow",
    "#ff6666": "red",
    "#5fb236": "green",
    "#2ea8e5": "blue",
    "#a28ae5": "purple",
    "#e56eee": "magenta",
    "#f19837": "orange",
    "#aaaaaa": "gray",
    "#d7d7ff": "lavender",
}


class Annotation(BaseModel):
    """One highlight/comment, with the libraries it was found in."""

    kind: str = Field(description="highlight | note")
    text: str = Field(default="", description="The paper's words (annotationText).")
    comment: str = Field(default="", description="Our words (annotationComment).")
    page: str | None = Field(default=None, description="annotationPageLabel.")
    color: str | None = None
    color_name: str | None = None
    sources: list[str] = Field(
        default_factory=list,
        description="Libraries this exact content was found in: personal | group.",
    )
    item_keys: dict[str, str] = Field(
        default_factory=dict, description="library -> Zotero item key, for zotero://."
    )

    @property
    def has_comment(self) -> bool:
        """True when we wrote something, as opposed to only highlighting."""
        return bool(self.comment.strip())


class PaperAnnotations(BaseModel):
    """Every annotation and note captured for one paper."""

    citation_key: str
    annotations: list[Annotation] = Field(default_factory=list)

    @property
    def comments(self) -> list[Annotation]:
        """Annotations carrying our own words."""
        return [a for a in self.annotations if a.has_comment]

    @property
    def highlights(self) -> list[Annotation]:
        """Highlights with no comment attached."""
        return [
            a for a in self.annotations if a.kind == "highlight" and not a.has_comment
        ]

    @property
    def notes(self) -> list[Annotation]:
        """Free-standing note items."""
        return [a for a in self.annotations if a.kind == "note"]

    def summary(self) -> str:
        """One-line tally for logs."""
        srcs = sorted({s for a in self.annotations for s in a.sources})
        return (
            f"{self.citation_key}: {len(self.comments)} comments, "
            f"{len(self.highlights)} highlights, {len(self.notes)} notes "
            f"[{'+'.join(srcs) or 'none'}]"
        )


def _content_hash(kind: str, text: str, comment: str, page: str | None) -> str:
    """Stable identity for an annotation's content, ignoring which library holds it.

    This is what lets a highlight copied into the group collapse with its personal
    original instead of appearing twice.
    """
    payload = "\x1f".join([kind, text.strip(), comment.strip(), (page or "").strip()])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _strip_html(html: str) -> str:
    """Flatten a Zotero note's HTML to plain text (notes are stored as HTML)."""
    import re

    text = re.sub(r"<br\s*/?>|</p>|</div>|</li>", "\n", html)
    text = re.sub(r"<li>", "- ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
    )
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def collect_library_annotations(
    lib: ZoteroLibrary, source: str, parent_keys: set[str] | None = None
) -> dict[str, list[Annotation]]:
    """Annotations + notes for one library, keyed by citation key.

    Args:
        lib: Connected Zotero library.
        source: Provenance label recorded on every record (``personal``/``group``).
        parent_keys: Restrict to these top-level item keys; ``None`` = whole library.
            The personal library passes the ``torchcell/*`` tree here so the other
            collections are never captured.

    Returns:
        ``citation_key -> [Annotation]`` for papers that have any.
    """
    items: list[dict[str, Any]] = lib.zot.everything(lib.zot.items())
    tops = {
        i["key"]: i
        for i in items
        if i["data"].get("itemType") not in ("attachment", "note", "annotation")
    }
    att_parent = {
        i["key"]: i["data"].get("parentItem")
        for i in items
        if i["data"].get("itemType") == "attachment"
    }

    out: dict[str, list[Annotation]] = defaultdict(list)
    for i in items:
        d = i["data"]
        itype = d.get("itemType")
        if itype == "annotation":
            top = att_parent.get(d.get("parentItem"))
            if top is None or top not in tops:
                continue
            if parent_keys is not None and top not in parent_keys:
                continue
            color = d.get("annotationColor")
            ann = Annotation(
                kind="highlight",
                text=(d.get("annotationText") or "").strip(),
                comment=(d.get("annotationComment") or "").strip(),
                page=d.get("annotationPageLabel") or None,
                color=color,
                color_name=COLOR_NAMES.get((color or "").lower()),
                sources=[source],
                item_keys={source: i["key"]},
            )
        elif itype == "note":
            top = d.get("parentItem")
            if top is None or top not in tops:
                continue
            if parent_keys is not None and top not in parent_keys:
                continue
            ann = Annotation(
                kind="note",
                comment=_strip_html(d.get("note") or ""),
                sources=[source],
                item_keys={source: i["key"]},
            )
        else:
            continue
        if not (ann.text or ann.comment):
            continue  # an empty highlight carries no information
        out[_resolve_citation_key(tops[top])].append(ann)
    log.info(
        "annotations: %s -> %d papers, %d records",
        source,
        len(out),
        sum(len(v) for v in out.values()),
    )
    return dict(out)


def merge_annotations(
    *sources: dict[str, list[Annotation]],
) -> dict[str, PaperAnnotations]:
    """Merge per-library maps, collapsing identical content and unioning provenance.

    Identical content in two libraries yields ONE record whose ``sources`` lists
    both; differing content stays separate. Records are ordered comments first,
    then highlights, then notes -- the order they are most useful to read in.
    """
    merged: dict[str, dict[str, Annotation]] = defaultdict(dict)
    for src in sources:
        for ck, anns in src.items():
            for a in anns:
                h = _content_hash(a.kind, a.text, a.comment, a.page)
                if (prev := merged[ck].get(h)) is not None:
                    for s in a.sources:
                        if s not in prev.sources:
                            prev.sources.append(s)
                    prev.item_keys.update(a.item_keys)
                else:
                    merged[ck][h] = a.model_copy(deep=True)

    out: dict[str, PaperAnnotations] = {}
    for ck, by_hash in merged.items():
        anns = list(by_hash.values())
        anns.sort(
            key=lambda a: (
                0
                if a.has_comment and a.kind == "highlight"
                else (2 if a.kind == "note" else 1),
                a.page or "",
            )
        )
        out[ck] = PaperAnnotations(citation_key=ck, annotations=anns)
    return out


def render_markdown(pa: PaperAnnotations) -> str:
    """Human-readable annotations for a paper, comments first.

    Every entry names the library it came from, so a note written on the group copy
    is never confused with one written on the personal copy.
    """
    L = [f"# Annotations — {pa.citation_key}", ""]
    srcs = sorted({s for a in pa.annotations for s in a.sources})
    L.append(
        f"{len(pa.comments)} comments · {len(pa.highlights)} highlights · "
        f"{len(pa.notes)} notes · sources: {', '.join(srcs) or 'none'}"
    )
    L.append("")
    if pa.comments:
        L += ["## Comments", ""]
        for a in pa.comments:
            where = "+".join(a.sources)
            page = f"p{a.page}" if a.page else "—"
            color = f" · {a.color_name}" if a.color_name else ""
            L.append(f"**[{where}]** {page}{color}")
            if a.text:
                L.append(f"> {a.text}")
            L.append("")
            L.append(a.comment)
            L.append("")
    if pa.notes:
        L += ["## Notes", ""]
        for a in pa.notes:
            L.append(f"**[{'+'.join(a.sources)}]**")
            L.append("")
            L.append(a.comment)
            L.append("")
    if pa.highlights:
        L += ["## Highlights (no comment)", ""]
        for a in pa.highlights:
            where = "+".join(a.sources)
            page = f"p{a.page}" if a.page else "—"
            color = f" · {a.color_name}" if a.color_name else ""
            L.append(f"- **[{where}]** {page}{color} — {a.text}")
        L.append("")
    return "\n".join(L)


def write_annotations(artifact_dir: str | Path, pa: PaperAnnotations) -> list[Path]:
    """Write ``annotations.json`` + ``annotations.md`` into a paper's artifact dir."""
    d = Path(artifact_dir)
    d.mkdir(parents=True, exist_ok=True)
    jp, mp = d / ANNOTATIONS_JSON, d / ANNOTATIONS_MD
    jp.write_text(json.dumps(pa.model_dump(), indent=1), encoding="utf-8")
    mp.write_text(render_markdown(pa), encoding="utf-8")
    return [jp, mp]


def personal_tree_item_keys(lib: ZoteroLibrary, root_collection: str) -> set[str]:
    """Top-level item keys under a collection and every collection beneath it.

    ``collection_items`` returns only a collection's direct members, so a nested
    collection (``torchcell/torchcell-topics/microbe-perturb-seq``) is invisible
    without walking the tree -- this walks it.
    """
    root_key = lib.collection_key(root_collection)
    colls: list[dict[str, Any]] = lib.zot.everything(lib.zot.collections())
    children: dict[str, list[str]] = defaultdict(list)
    for c in colls:
        children[c["data"].get("parentCollection") or ""].append(c["key"])

    def walk(key: str) -> list[str]:
        out = [key]
        for kid in children.get(key, []):
            out += walk(kid)
        return out

    keys: set[str] = set()
    for ckey in walk(root_key):
        for item in lib.zot.everything(lib.zot.collection_items_top(ckey)):
            if item["data"].get("itemType") not in ("attachment", "note", "annotation"):
                keys.add(item["key"])
    log.info(
        "annotations: personal tree '%s' -> %d collections, %d papers",
        root_collection,
        len(walk(root_key)),
        len(keys),
    )
    return keys


def capture_annotations(
    mirror_root: str | Path,
    merged: dict[str, PaperAnnotations],
    *,
    only_mirrored: bool = True,
) -> list[str]:
    """Write annotation artifacts for every paper we have annotations for.

    Args:
        mirror_root: The ``torchcell-library`` directory.
        merged: Output of :func:`merge_annotations`.
        only_mirrored: Skip papers with no artifact directory yet, rather than
            creating a directory holding annotations but no paper.

    Returns:
        Citation keys written.
    """
    root = Path(mirror_root)
    written: list[str] = []
    for ck, pa in sorted(merged.items()):
        d = root / ck
        if only_mirrored and not (d / "manifest.json").exists():
            continue
        write_annotations(d, pa)
        written.append(ck)
    log.info("annotations: wrote artifacts for %d papers", len(written))
    return written


def unmirrored_with_annotations(
    mirror_root: str | Path, merged: dict[str, PaperAnnotations]
) -> list[str]:
    """Papers that have annotations but no mirrored artifact directory yet."""
    root = Path(mirror_root)
    return sorted(ck for ck in merged if not (root / ck / "manifest.json").exists())


def collect_all(
    personal: ZoteroLibrary, group: ZoteroLibrary, *, root_collection: str
) -> dict[str, PaperAnnotations]:
    """Collect and merge annotations across the personal tree and the whole group."""
    tree = personal_tree_item_keys(personal, root_collection)
    p = collect_library_annotations(personal, "personal", parent_keys=tree)
    g = collect_library_annotations(group, "group")
    return merge_annotations(p, g)
