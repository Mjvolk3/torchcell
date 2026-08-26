#!/usr/bin/env python
# notes-tex/common/zotero_publish.py
# [[notes-tex.common.zotero_publish]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/common/zotero_publish.py
"""Publish a built LaTeX PDF into the personal Zotero library for review.

Why Zotero and not a shared folder: the point is *annotation*. Zotero's reader
keeps highlights and comments attached to a specific PDF attachment, so a comment
always refers to the exact rendering it was made on. That is the property we want
from a review copy -- a note saying "this number is wrong" is meaningless unless
you can tell which build it was made against.

Layout, which is DERIVED from the repo path rather than invented::

    repo    <dir>/<pdf>.pdf
    zotero  torchcell / <dir parent>          <- one flat index per document kind
            torchcell / <dir leaf>            <- that document and its versions

    notes-tex/024-perturb-seq-costing/main.pdf
      -> torchcell / notes-tex / 024-perturb-seq-costing
    paper/nature-biotech/editing.pdf
      -> torchcell / paper / nature-biotech

so there is never a question of which Zotero collection a document belongs in.
The topic collections (``torchcell / torchcell-topics / *``) hold OTHER people's
papers; these trees hold ours. Keeping the two apart is what stops a draft from
being mistaken for literature when the bibliography is rebuilt.

That separation has already been load-bearing once. Two Zotero collections were
named ``microbe-perturb-seq``: this script's publication collection, and the
topic collection that feeds ``make bib``. Renaming the wrong one would have
broken the bibliography. Derive the path from the repo, never from a name that
happens to match.

The document is filed in BOTH collections, deliberately. Zotero does not show a
sub-collection's items in its parent unless the reader has turned on View ->
Show Items from Subcollections, so filing only into the leaf makes the parent
look empty -- which is exactly what happened the first time this ran. An item may
belong to any number of collections and its file is stored once, so the cost is
nil and the parent becomes a usable index of every document of that kind.

Versioning. Each publish creates one child attachment named::

    <doc>_<YYYY-MM-DD-HH-MM-SS>_<sha256[:8]>.pdf

Both halves earn their place. The timestamp is what a human sorts by; the hash is
what actually identifies the bytes, and it is the same identifier the repo's
provenance rule uses everywhere else -- the stored artifact plus its sha256 is
canonical. Given both, a comment on "the 2026-08-21 build" can be traced to an
exact file, and re-running with unchanged content is detected as a duplicate
rather than piling up near-identical PDFs.

Dedupe by hash only works because the build is reproducible, and it was not until
this script needed it: hyperref writes a wall-clock ``/CreationDate`` into a
compressed object stream, so two builds of identical sources used to differ. The
fix is ``SOURCE_DATE_EPOCH`` in ``Makefile.common``, pinned to the newest source
mtime. Measured, not assumed: two forced rebuilds now produce the same sha256.

So a re-run of ``make`` with nothing edited is recognised as the same version and
skipped, while any real edit produces a new hash. Do not "simplify" that Makefile
line away -- versioning here rests on it.

Provenance is written into the parent item's ``extra`` field: the git commit and
branch the build came from. A reviewer who finds a problem can go from the Zotero
item to the exact source state without asking anyone.

Credentials via repo-root ``.env`` (``ZOTERO_USER_ID``, ``ZOTERO_API_KEY``). The
key needs user-library write and file access; ``--dry-run`` needs neither beyond
read.

Usage::

    # a notes-tex document: a bare name still means notes-tex/<name>
    python notes-tex/common/zotero_publish.py 024-perturb-seq-costing --dry-run
    python notes-tex/common/zotero_publish.py 024-perturb-seq-costing
    python notes-tex/common/zotero_publish.py 024-perturb-seq-costing --list

    # the manuscript: a repo-relative directory, a named PDF, and the tex that
    # actually declares the title
    python notes-tex/common/zotero_publish.py paper/nature-biotech \\
        --pdf editing --tex sections/frontmatter.tex

``--tex`` is explicit rather than discovered. Searching the directory for the one
file containing ``\\title`` finds two in paper/nature-biotech, because the stock
``sn-article.tex`` sample carries a placeholder title, and a rule that silently
picks one of two is worse than a flag that has to be written down.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import os.path as osp
import re
import shutil
import subprocess
import sys
import tempfile

from dotenv import load_dotenv
from pydantic import BaseModel
from pyzotero import zotero
from pyzotero.zotero import Zupload

# The document trees live beside the topic tree, under the same `torchcell` root.
ROOT_COLLECTION = "torchcell"

# A bare name with no slash is a notes-tex document. That is where every document
# lived when this script was written, and keeping the short form working means the
# published Doc Key markers of the existing collections stay correct: the key IS
# the repo-relative directory, so `024-perturb-seq-costing` and
# `notes-tex/024-perturb-seq-costing` resolve to the same key and the same parent
# item. Without that, generalizing this script would have orphaned 19 versions.
DEFAULT_PARENT_DIR = "notes-tex"

# Marker written into the parent item's `extra`, and the key this script matches
# on when deciding whether a parent already exists. Matching on the title would
# break the moment someone edits the title in Zotero.
DOC_KEY_PREFIX = "Doc Key:"


class BuiltDoc(BaseModel):
    """A built PDF and everything needed to identify it later."""

    doc_dir: str  # repo-relative, e.g. "notes-tex/024-..." or "paper/nature-biotech"
    pdf_stem: str  # "main", "main-clean", "editing", "submission", ...
    pdf_path: str
    title: str
    subtitle: str | None
    authors: list[tuple[str, str]]  # (first, last), in document order
    sha256: str
    n_bytes: int
    git_commit: str
    git_branch: str
    built_at: str  # YYYY-MM-DD-HH-MM-SS, local

    @property
    def doc(self) -> str:
        """Leaf directory name, which is also the Zotero collection name."""
        return self.doc_dir.rsplit("/", 1)[-1]

    @property
    def collection_path(self) -> list[str]:
        """Collection names from the root down, derived from the repo path."""
        return self.doc_dir.split("/")

    @property
    def doc_key(self) -> str:
        return self.doc_dir

    @property
    def filename(self) -> str:
        # The PDF stem is part of the name whenever it is not the plain `main`
        # build. A draft and a share build, or an editing and a submission build,
        # are different bytes and different sha256, so without it the two sort
        # together and a reviewer cannot tell which one they are commenting on.
        # `main` contributes nothing, which keeps every already-published
        # notes-tex filename byte-identical to what it was.
        stem = "" if self.pdf_stem == "main" else f"-{self.pdf_stem}"
        return f"{self.doc}{stem}_{self.built_at}_{self.sha256[:8]}.pdf"

    @property
    def full_title(self) -> str:
        return f"{self.title}: {self.subtitle}" if self.subtitle else self.title

    def extra_field(self) -> str:
        return "\n".join(
            [
                f"{DOC_KEY_PREFIX} {self.doc_key}",
                f"Git Commit: {self.git_commit}",
                f"Git Branch: {self.git_branch}",
            ]
        )


def _git(repo: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _braced(src: str, open_idx: int) -> str:
    """Contents of the balanced {...} group starting at ``open_idx``.

    A regex cannot do this: the manuscript's title carries a nested macro
    (``\\cb{14/15 words}``), so a non-greedy match stops at the inner brace and a
    greedy one runs to the end of the file.
    """
    depth, out = 0, []
    for i in range(open_idx, len(src)):
        c = src[i]
        if c == "{":
            depth += 1
            if depth == 1:
                continue
        elif c == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(c)
    sys.exit(f"unbalanced {{}} after position {open_idx}")


def _parse_title(tex_path: str) -> tuple[str, str | None, list[tuple[str, str]]]:
    """Pull title / subtitle / author out of the tex that declares them.

    Read from the source rather than passed on the command line: a title typed
    into a flag drifts from the one on the page, and then the Zotero item names a
    document that no longer exists under that name.

    Two dialects are supported, dispatched on what the file actually contains
    rather than tried in turn:

    * ``notes-tex`` documents, which use plain ``\\title{Title\\\\ {\\large Sub}}``
      and ``\\author{Name}``.
    * the ``sn-jnl`` manuscript, which uses ``\\title[short]{long}`` and one
      ``\\author*[affils]{\\fnm{First} \\sur{Last}}`` per author.
    """
    src = open(tex_path, encoding="utf-8").read()

    m = re.search(r"\\title\s*(\[[^\]]*\])?\s*\{", src)
    if not m:
        sys.exit(f"no \\title found in {tex_path}")
    raw = _braced(src, m.end() - 1)

    # Editing-only annotation macros never belong in a bibliographic title.
    raw = re.sub(r"\\cb\{[^}]*\}", "", raw)
    raw = re.sub(r"\\vspace\{[^}]*\}", "", raw)
    raw = re.sub(r"\{\\large\s*", "", raw)
    raw = raw.replace("\\\\", "\x00").replace("}", "").replace("{", "")
    # Collapse whitespace PER LINE. Doing it across the whole string eats the
    # title/subtitle break and silently yields one run-on title.
    parts = [" ".join(p.split()) for p in raw.split("\x00")]
    lines = [p for p in parts if p]
    title = lines[0]
    subtitle = lines[1] if len(lines) > 1 else None

    # Returned as (first, last) pairs rather than one joined string. Joining and
    # re-splitting on whitespace is what the single-author version did, and on a
    # three-author manuscript it yields the first author's given name beside the
    # last author's surname.
    if "\\fnm{" in src:
        authors = [
            (f.strip(), l.strip())
            for f, l in re.findall(r"\\fnm\{([^}]*)\}\s*\\sur\{([^}]*)\}", src)
        ]
    else:
        a = re.search(r"\\author\{([^}]*)\}", src)
        parts = a.group(1).split() if a else []
        authors = [(" ".join(parts[:-1]), parts[-1])] if parts else []
    return title, subtitle, authors


def load_built_doc(repo: str, rel_dir: str, pdf_stem: str, tex_rel: str) -> BuiltDoc:
    """Load a built PDF given a repo-relative directory and a PDF stem.

    ``main.pdf`` carries the status chips and provenance flags and is the right
    thing to review in-group; ``main-clean.pdf`` is what leaves the group. For the
    manuscript the same distinction is ``editing`` against ``submission``.
    """
    doc_dir = osp.join(repo, rel_dir)
    if not osp.isdir(doc_dir):
        sys.exit(f"{doc_dir} is not a directory.")
    pdf = osp.join(doc_dir, f"{pdf_stem}.pdf")
    tex = osp.join(doc_dir, tex_rel)
    if not osp.exists(pdf):
        sys.exit(f"{pdf} does not exist -- build it first.")
    if not osp.exists(tex):
        sys.exit(f"{tex} does not exist -- pass --tex with the file declaring \\title.")

    raw = open(pdf, "rb").read()
    title, subtitle, authors = _parse_title(tex)
    # --untracked-files=no on purpose. `scratch.*` notes and their rendered
    # PDFs are untracked BY RULE and never committable, so counting untracked
    # files here marked every build from the primary checkout "-dirty" and the
    # flag stopped meaning anything. What the provenance stamp needs to say is
    # "were there uncommitted TRACKED changes in the build inputs", which is
    # this.
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=no") != ""
    commit = _git(repo, "rev-parse", "HEAD")[:12] + ("-dirty" if dirty else "")

    return BuiltDoc(
        doc_dir=rel_dir,
        pdf_stem=pdf_stem,
        pdf_path=pdf,
        title=title,
        subtitle=subtitle,
        authors=authors,
        sha256=hashlib.sha256(raw).hexdigest(),
        n_bytes=len(raw),
        git_commit=commit,
        git_branch=_git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        built_at=datetime.datetime.fromtimestamp(osp.getmtime(pdf)).strftime(
            "%Y-%m-%d-%H-%M-%S"
        ),
    )


def refresh_provenance(zot: zotero.Zotero, parent: dict, built: BuiltDoc) -> None:
    """Point the parent item at the build the top attachment came from.

    Both the upload path and the identical-bytes path call this: the stamp is a
    property of the SOURCE, not of the file, so it can go stale even when nothing
    was uploaded.
    """
    item = parent if parent.get("data") else zot.item(parent["key"])
    was = item["data"].get("extra", "")
    item["data"]["extra"] = built.extra_field()
    item["data"]["date"] = built.built_at[:10]
    if item["data"]["extra"] == was:
        print(f"  parent provenance already current -> {built.git_branch} @ {built.git_commit}")
        return
    zot.update_item(item)
    print(f"  parent provenance updated -> {built.git_branch} @ {built.git_commit}")


def find_collection(zot: zotero.Zotero, name: str, parent: str | None) -> str | None:
    """Find a LIVE collection by name under a given parent (None = top level).

    Trashed collections are skipped, and that is not a detail. `zot.collections()`
    returns collections that are in the Zotero trash, carrying `deleted: True` in
    their data, while the Zotero client hides them. Without this filter the script
    happily files a document into a deleted collection: the API reports success,
    every lookup afterwards resolves, and the document is invisible in the client
    with no error anywhere to explain why.

    That is exactly what happened on 2026.08.25. `torchcell/paper` had been trashed
    at some earlier point, this function found it, `ensure_collection` reused it
    rather than creating a live one, and the manuscript was published into a
    subtree nobody could see. It read as a sync failure for some time, which is the
    expensive part: the symptom points at the client, and the cause is here.
    """
    for c in zot.everything(zot.collections()):
        d = c["data"]
        if d.get("deleted"):
            continue
        if d["name"] == name and (d.get("parentCollection") or None) == parent:
            return c["key"]
    return None


def ensure_collection(
    zot: zotero.Zotero, name: str, parent: str | None, dry: bool
) -> str | None:
    key = find_collection(zot, name, parent)
    if key:
        return key
    if dry:
        print(f"  [dry-run] would create collection {name!r}")
        return None
    payload: dict = {"name": name}
    if parent:
        payload["parentCollection"] = parent
    resp = zot.create_collections([payload])
    key = resp["successful"]["0"]["key"]
    print(f"  created collection {name!r} ({key})")
    return key


def find_parent_item(zot: zotero.Zotero, coll: str, doc_key: str) -> dict | None:
    for it in zot.everything(zot.collection_items_top(coll)):
        if f"{DOC_KEY_PREFIX} {doc_key}" in it["data"].get("extra", ""):
            return it
    return None


def existing_hashes(zot: zotero.Zotero, parent_key: str) -> dict[str, str]:
    """sha8 -> attachment filename, read back off the child attachments.

    Taken from the FILENAME rather than by downloading each attachment and
    hashing it. The filename is written by this script and is the only thing that
    survives into the Zotero UI, so if the two ever disagree the filename is what
    a human would go by.
    """
    out = {}
    for ch in zot.children(parent_key):
        d = ch["data"]
        fn = d.get("filename") or d.get("title") or ""
        m = re.search(r"_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_([0-9a-f]{8})\.pdf$", fn)
        if m:
            out[m.group(2)] = fn
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("doc", help="repo-relative document directory, e.g. "
                                "paper/nature-biotech; a bare name means "
                                "notes-tex/<name>")
    ap.add_argument("--pdf", default=None, metavar="STEM",
                    help="PDF stem to publish (default: main). e.g. editing")
    ap.add_argument("--tex", default=None, metavar="PATH",
                    help="document-relative tex declaring \\title (default: "
                         "<STEM>.tex). e.g. sections/frontmatter.tex")
    ap.add_argument("--dry-run", action="store_true", help="preview, upload nothing")
    ap.add_argument("--clean", action="store_true",
                    help="shorthand for --pdf main-clean, the share view")
    ap.add_argument("--list", action="store_true", help="list published versions and exit")
    args = ap.parse_args()

    common_dir = osp.dirname(osp.abspath(__file__))
    repo = _git(common_dir, "rev-parse", "--show-toplevel")
    load_dotenv(osp.join(repo, ".env"))
    load_dotenv()

    user_id, api_key = os.getenv("ZOTERO_USER_ID"), os.getenv("ZOTERO_API_KEY")
    if not (user_id and api_key):
        sys.exit("Set ZOTERO_USER_ID and ZOTERO_API_KEY in repo-root .env.")
    zot = zotero.Zotero(user_id, "user", api_key)

    if args.clean and args.pdf:
        sys.exit("--clean and --pdf say the same thing; pass only one.")
    rel_dir = args.doc.strip("/")
    if "/" not in rel_dir:
        rel_dir = f"{DEFAULT_PARENT_DIR}/{rel_dir}"
    pdf_stem = args.pdf or ("main-clean" if args.clean else "main")
    tex_rel = args.tex or f"{pdf_stem}.tex"

    built = load_built_doc(repo, rel_dir, pdf_stem, tex_rel)
    print(f"{built.full_title}")
    print(f"  {built.pdf_path}")
    print(f"  {built.n_bytes:,} bytes  sha256 {built.sha256}")
    print(f"  built {built.built_at}  from {built.git_branch} @ {built.git_commit}")
    print(f"  -> {built.filename}\n")

    root = find_collection(zot, ROOT_COLLECTION, None)
    if not root:
        sys.exit(f"no top-level {ROOT_COLLECTION!r} collection in the personal library.")
    # Walk the repo path down, creating what is missing. `docs` ends up as the
    # immediate parent so the item can be filed into the index as well as the leaf.
    docs, coll = root, root
    for name in built.collection_path:
        docs = coll
        coll = ensure_collection(zot, name, coll, args.dry_run)
        if not coll:
            break
    if coll:
        print(f"  collection {ROOT_COLLECTION}/{'/'.join(built.collection_path)} = {coll}")

    parent = find_parent_item(zot, coll, built.doc_key) if coll else None

    if args.list:
        if not parent:
            print("\nnothing published yet.")
            return
        print(f"\npublished versions of {built.doc}:")
        for sha8, fn in sorted(existing_hashes(zot, parent["key"]).items(), key=lambda x: x[1]):
            here = "  <- current build" if sha8 == built.sha256[:8] else ""
            print(f"  {fn}{here}")
        return

    # --- parent item ---------------------------------------------------------
    # itemType `report`: it is our own document with an author and a date, not a
    # journal article and not a bare file. A standalone attachment would work but
    # gives the collection a list of filenames instead of a titled document with
    # its versions underneath.
    if parent:
        print(f"  parent item exists: {parent['key']}")
        # Backfill for items created before the top-level filing existed, and a
        # cheap self-heal if someone drags the item out of one collection.
        item = parent if parent.get("data") else zot.item(parent["key"])
        cols = set(item["data"].get("collections") or [])
        if not {coll, docs} <= cols and not args.dry_run:
            item["data"]["collections"] = sorted(cols | {coll, docs})
            zot.update_item(item)
            print(f"  filed into the {built.collection_path[-2]} index as well")
    elif args.dry_run:
        print("  [dry-run] would create parent report item")
    else:
        tmpl = zot.item_template("report")
        tmpl["title"] = built.full_title
        tmpl["creators"] = [
            {"creatorType": "author", "firstName": first, "lastName": last}
            for first, last in built.authors
        ]
        tmpl["reportType"] = "Internal document"
        tmpl["institution"] = "University of Illinois Urbana-Champaign"
        tmpl["date"] = built.built_at[:10]
        tmpl["extra"] = built.extra_field()
        # Both: the per-document collection AND the flat index above it.
        tmpl["collections"] = [coll, docs]
        resp = zot.create_items([tmpl])
        parent = {"key": resp["successful"]["0"]["key"]}
        print(f"  created parent item {parent['key']}")

    if args.dry_run:
        print("\n[dry-run] nothing uploaded.")
        return

    # --- version attachment --------------------------------------------------
    have = existing_hashes(zot, parent["key"])
    if built.sha256[:8] in have:
        print(f"\nalready published as {have[built.sha256[:8]]} -- identical bytes.")
        # Same bytes, but not necessarily the same provenance. Publishing from a
        # dirty tree and then committing leaves the PDF identical while the commit
        # it can be traced to changes, and returning here used to strand the parent
        # on the older stamp for good. Refresh it, then stop.
        refresh_provenance(zot, parent, built)
        return

    # Upload from a temp copy so the versioned name is what lands in Zotero
    # storage rather than a generic "main.pdf".
    #
    # NOT pyzotero's attachment_simple: it puts the path it is handed into the
    # item's `filename` field, and the API rejects that outright --
    # "Stored-file filename '...' cannot contain a directory path". Zupload's
    # basedir is the supported way to say "this directory, that bare filename".
    with tempfile.TemporaryDirectory() as td:
        shutil.copy2(built.pdf_path, osp.join(td, built.filename))
        tmpl = {
            "itemType": "attachment",
            "linkMode": "imported_file",
            "title": built.filename,
            "filename": built.filename,
            "contentType": "application/pdf",
            "charset": "",
            "note": "",
            "tags": [],
            "relations": {},
        }
        resp = Zupload(zot, [tmpl], parent["key"], basedir=td).upload()

    if not (resp.get("success") or resp.get("unchanged")):
        sys.exit(f"upload failed: {resp}")
    print(f"\nuploaded {built.filename}")

    refresh_provenance(zot, parent, built)
    print(f"  {len(have) + 1} version(s) now in the collection")


if __name__ == "__main__":
    main()
