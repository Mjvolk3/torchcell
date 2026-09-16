#!/usr/bin/env python
# notes-tex/common/bib_from_store.py
# [[notes-tex.common.bib_from_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/common/bib_from_store.py
"""Build a document's ``references.bib`` from a served bibliography, by cited key.

The third route to a ``references.bib``, for a document that does not yet have a
Zotero collection of its own. The other two are unchanged and are still preferred
when they apply:

* ``build_bib.py`` exports the union of a group and a personal Zotero collection
  named for the document. It needs Zotero and Better BibTeX running locally, and
  it needs the collection to exist.
* ``lit_bib_pull.py`` downloads a bibliography the tc-lit store already serves by
  that document's name.

A document with no collection can cite nothing under either route, which is how
008-xue-ffa-epistasis ended up naming a dozen prior works in prose with no
``\\cite`` anywhere. This script closes that gap WITHOUT writing to Zotero: it
reads the nightly whole-library export that ``scripts/lit_bib_store.py`` already
writes (``$DATA_ROOT/torchcell-library/_bib/library.bib``), verifies it against
that store's manifest sha256, scans the document's ``.tex`` for cited keys, and
writes only those entries.

So the CURATION still happens in Zotero and only in Zotero. A key this script
cannot find is an error, never a stub: the work is not in the library, and adding
it there is a separate, human decision (CLAUDE.md, "NEVER put a paper into
Zotero"). Re-running after that decision picks it up.

Usage::

    python ../common/bib_from_store.py --out references.bib
    python ../common/bib_from_store.py --check      # report only, write nothing

Exit 1 when a cited key is missing from the store, or when the store's bytes do
not match its manifest hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import re
import sys

from dotenv import load_dotenv

CITE_RE = re.compile(r"\\(?:cite|citep|citet|mirror)\*?(?:\[[^\]]*\])*\{([^}]*)\}")
ENTRY_RE = re.compile(r"^@(\w+)\{([^,]+),", re.M)

# Fields carried by the Zotero export that are private bookkeeping, and that the
# bibliography style either prints as noise or would print if the style changed. The
# emoji in `keywords` are reading-status tags; `abstract` is several hundred words per
# entry; `copyright` and `file` are local to one machine.
DROP_FIELDS = {"abstract", "keywords", "file", "copyright", "annotation", "annote"}
# Better BibTeX writes the citation key and its citation counts into `note`, and
# sn-nature.bst prints `note` verbatim, which is why a reference came out reading
# "... Science 353, aaf1420 (2016). URL ... CostanzoGlobalGeneticInteraction2016 844
# citations (Semantic Scholar/DOI) [2022-11-26] 683 citations (Crossref) [2022-04-15]."
# A note line is dropped when it is one of those; anything else in `note` is kept, since
# for an internal document it carries the doc key and the commit it was built from.
NOTE_NOISE = re.compile(r"^\s*(\d+\s+citations\s*\(|Number:\s|\S+\d{4}\s*$)")


def store_path(name: str) -> tuple[str, str]:
    """Path to the served bibliography ``name`` and to its store manifest."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    bib_dir = osp.join(data_root, "torchcell-library", "_bib")
    return osp.join(bib_dir, f"{name}.bib"), osp.join(bib_dir, "manifest.json")


def read_verified(name: str) -> tuple[str, str]:
    """The served bibliography's text and sha256, verified against the manifest.

    The manifest is the trust anchor, exactly as in ``lit_bib_pull.py``: a file whose
    bytes do not hash to the recorded value is a corrupt or half-written export, and
    building a bibliography from it would silently change what the document may cite.
    """
    bib_file, manifest_file = store_path(name)
    with open(bib_file, "rb") as fh:
        raw = fh.read()
    digest = hashlib.sha256(raw).hexdigest()
    with open(manifest_file, encoding="utf-8") as fh:
        manifest = json.load(fh)
    record = next((b for b in manifest["bibs"] if b["name"] == name), None)
    if record is None:
        raise SystemExit(f"{manifest_file}: no bibliography named {name!r}")
    if record["sha256"] != digest:
        raise SystemExit(
            f"{bib_file}: sha256 {digest[:12]} does not match the manifest's "
            f"{record['sha256'][:12]}; the export is stale or truncated"
        )
    return raw.decode("utf-8"), digest


def cited_keys(root: str = ".") -> set[str]:
    """Every key cited by any ``.tex`` under ``root``."""
    keys: set[str] = set()
    for dirpath, _dirs, files in os.walk(root):
        if any(part in dirpath for part in (".git", "figures")):
            continue
        for fn in files:
            if not fn.endswith(".tex"):
                continue
            with open(osp.join(dirpath, fn), encoding="utf-8", errors="replace") as fh:
                for match in CITE_RE.finditer(fh.read()):
                    keys.update(k.strip() for k in match.group(1).split(",") if k.strip())
    return keys


def split_fields(entry: str) -> tuple[str, list[tuple[str, str]]]:
    """An entry split into its ``@type{key,`` header and its ``(name, value)`` fields.

    Written by hand rather than with a bibtex parser because the entries go back out
    verbatim: a round trip through a parser would renormalize braces, capitalization
    protection and unicode, and the exported file is the thing whose sha256 is recorded.
    """
    head, _, body = entry.partition(",")
    fields: list[tuple[str, str]] = []
    depth, start, name = 0, 0, None
    for i, ch in enumerate(body):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "=" and depth == 0 and name is None:
            name = body[start:i].strip().lower()
        elif ch == "," and depth == 0 and name is not None:
            fields.append((name, body[start:i].split("=", 1)[1].strip()))
            start, name = i + 1, None
    tail = body[start:].rstrip().rstrip("}").rstrip()
    if name is not None and tail:
        fields.append((name, tail.split("=", 1)[1].strip()))
    return head + ",", fields


def clean_entry(entry: str) -> str:
    """Drop the private-bookkeeping fields and the citation-count lines from ``note``."""
    head, fields = split_fields(entry)
    kept = []
    for name, value in fields:
        if name in DROP_FIELDS:
            continue
        if name == "note":
            lines = [ln for ln in value.strip("{}").splitlines()
                     if ln.strip() and not NOTE_NOISE.match(ln)]
            if not lines:
                continue
            value = "{" + "\n".join(lines) + "}"
        kept.append((name, value))
    body = "".join(f"\n  {name} = {value}," for name, value in kept)
    return head + body.rstrip(",") + "\n}\n"


def split_entries(text: str) -> dict[str, str]:
    """The bibliography split into ``{key: entry text}``."""
    starts = [(m.start(), m.group(2).strip()) for m in ENTRY_RE.finditer(text)]
    entries = {}
    for i, (pos, key) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        entries[key] = text[pos:end].rstrip() + "\n"
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="library",
                    help="served bibliography to read (default: the whole library)")
    ap.add_argument("--out", default="references.bib")
    ap.add_argument("--root", default=".", help="directory to scan for \\cite keys")
    ap.add_argument("--check", action="store_true",
                    help="report only, write nothing; exit 1 if a key is missing")
    args = ap.parse_args()

    text, digest = read_verified(args.name)
    entries = split_entries(text)
    keys = cited_keys(args.root)
    missing = sorted(k for k in keys if k not in entries)
    kept = sorted(k for k in keys if k in entries)

    print(f"{args.name}.bib  {len(entries)} entries  sha256={digest[:12]}")
    print(f"cited {len(keys)}, resolved {len(kept)}, missing {len(missing)}")
    for key in missing:
        print(f"  MISSING {key} -- not in the Zotero library; adding it is a curation "
              f"decision, not this script's")
    if missing:
        return 1
    if args.check:
        return 0

    header = (
        f"%% GENERATED by notes-tex/common/bib_from_store.py -- do not edit by hand.\n"
        f"%% Source: $DATA_ROOT/torchcell-library/_bib/{args.name}.bib\n"
        f"%%   sha256 {digest}\n"
        f"%% Entries: the {len(kept)} keys cited by this document, of "
        f"{len(entries)} served.\n"
        f"%% Fields dropped per entry: {', '.join(sorted(DROP_FIELDS))}, and the citation-\n"
        f"%% count lines Better BibTeX writes into `note` (sn-nature.bst prints `note`).\n"
        f"%% Regenerate: make bib-store\n\n"
    )
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(header)
        for key in kept:
            fh.write(clean_entry(entries[key]))
            fh.write("\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
