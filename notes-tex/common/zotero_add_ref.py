#!/usr/bin/env python
# notes-tex/common/zotero_add_ref.py
# [[notes-tex.common.zotero_add_ref]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/common/zotero_add_ref.py
"""Add a journal article to a notes-tex document's PERSONAL Zotero collection, by DOI.

``build_bib.py`` regenerates ``references.bib`` from Zotero, so hand-editing that
file is pointless -- the next ``make bib`` overwrites it. The library is the
source of truth, which means "cite a new paper" has to mean "put it in the
library first". That is the rule; this is the tool that makes obeying it cheap.

Metadata comes from CrossRef rather than being typed, because a hand-entered
reference is a silent-corruption risk of exactly the kind this repo's provenance
rules exist to prevent: a wrong volume or a wrong year is invisible in the built
PDF. The DOI is the input, CrossRef is the authority, and the DOI is recorded on
the item so the mapping is auditable afterwards.

Deliberately targets the PERSONAL collection only. Promotion to the shared group
library is a curation decision a person makes, not something a script should do
on the way past -- see notes-tex/README.md on why the two tiers differ.

Better BibTeX generates the citekey once the item lands, so the key this prints
is a PREDICTION of BBT's author-title-year format; run ``make bib`` and use the
key that actually appears in references.bib.

Usage:
    python zotero_add_ref.py --collection AC8MFJXK 10.1186/s13059-016-0904-5 ...
    python zotero_add_ref.py --collection AC8MFJXK --dry-run <doi>
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import re
import sys
import urllib.request

from dotenv import load_dotenv

CROSSREF = "https://api.crossref.org/works/"


def crossref(doi: str) -> dict:
    """Authoritative metadata for one DOI. Raises rather than guessing."""
    req = urllib.request.Request(
        CROSSREF + doi,
        headers={"User-Agent": "torchcell-notes-tex (mailto:michaeljvolk7@gmail.com)"},
    )
    with urllib.request.urlopen(req, timeout=30) as fh:
        import json

        return json.load(fh)["message"]


def to_zotero(m: dict) -> dict:
    """CrossRef record -> Zotero journalArticle. Only fields CrossRef actually gives."""
    creators = [
        {
            "creatorType": "author",
            "firstName": a.get("given", ""),
            "lastName": a.get("family", "") or a.get("name", ""),
        }
        for a in m.get("author", [])
    ]
    issued = m.get("published-print") or m.get("published-online") or m.get("issued", {})
    year = str(issued.get("date-parts", [["" ]])[0][0] or "")
    return {
        "itemType": "journalArticle",
        "title": (m.get("title") or [""])[0],
        "creators": creators,
        "publicationTitle": (m.get("container-title") or [""])[0],
        "volume": m.get("volume", "") or "",
        "issue": m.get("issue", "") or "",
        "pages": m.get("page", "") or "",
        "date": year,
        "DOI": m.get("DOI", ""),
        "ISSN": ", ".join(m.get("ISSN", []) or []),
        "url": m.get("URL", ""),
        # The marker the repo uses for anything that did not arrive by the normal
        # curated route, so it can be found and reviewed later.
        "extra": "added-by: notes-tex/common/zotero_add_ref.py (CrossRef)",
    }


def predict_citekey(item: dict) -> str:
    """Better BibTeX's authorTitleYear shape, for reporting only."""
    last = (item["creators"][0]["lastName"] if item["creators"] else "anon").lower()
    words = [w for w in re.findall(r"[A-Za-z]+", item["title"]) if len(w) > 3][:3]
    return re.sub(r"[^a-z]", "", last) + "".join(w.capitalize() for w in words) + item["date"]


def main() -> int:
    load_dotenv(osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), ".env"))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dois", nargs="+",
                    help="DOI, or DOI=citekey to PIN the Better BibTeX key")
    ap.add_argument("--collection", required=True, help="personal collection key")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--emit-bibtex", action="store_true",
        help="print BibTeX from CrossRef instead of writing to Zotero. For the "
             "case where Better BibTeX's export endpoint is unreachable and "
             "references.bib has to be patched by hand until `make bib` works.",
    )
    a = ap.parse_args()

    if a.emit_bibtex:
        for spec in a.dois:
            doi, _, pin = spec.partition("=")
            req = urllib.request.Request(
                "https://doi.org/" + doi,
                headers={"Accept": "application/x-bibtex",
                         "User-Agent": "torchcell-notes-tex"},
            )
            with urllib.request.urlopen(req, timeout=30) as fh:
                bib = fh.read().decode("utf-8").strip()
            if pin:
                bib = re.sub(r"^@(\w+)\{[^,]*,", rf"@\1{{{pin},", bib, count=1)
            # Mark it, so a later reader can tell a hand-patched entry from a
            # Zotero-exported one without diffing against the library. Insert
            # before the FINAL brace only -- rstrip("}") would also eat the
            # closing brace of the last field, e.g. pages={32-41}.
            cut = bib.rindex("}")
            bib = bib[:cut].rstrip().rstrip(",") + ",\n\tkeywords = {zotero-pending}\n}"
            first = re.search(r"author=\{([^,}]+)", bib)
            print(f"%% first author per CrossRef: {first.group(1) if first else '?'}")
            print(bib + "\n")
        return 0

    user_id, api_key = os.getenv("ZOTERO_USER_ID"), os.getenv("ZOTERO_API_KEY")
    if not a.dry_run and not (user_id and api_key):
        sys.exit("Set ZOTERO_USER_ID and ZOTERO_API_KEY in repo-root .env.")

    items = []
    for spec in a.dois:
        doi, _, pin = spec.partition("=")
        it = to_zotero(crossref(doi))
        if not it["title"]:
            sys.exit(f"{doi}: CrossRef returned no title; refusing to add a blank item")
        # A PINNED key is what makes a \citep stable. Without it BBT regenerates
        # the key from metadata, and a key that changes silently breaks a
        # citation that used to resolve.
        if pin:
            it["extra"] = f"Citation Key: {pin}\n{it['extra']}"
        items.append(it)
        print(f"{doi}\n  {it['title'][:88]}\n  {it['publicationTitle']} "
              f"{it['volume']}({it['issue']}) {it['pages']} {it['date']}\n"
              f"  citekey: {pin or predict_citekey(it) + ' (predicted, NOT pinned)'}")

    if a.dry_run:
        print("\n[dry-run] nothing written.")
        return 0

    from pyzotero import zotero

    zot = zotero.Zotero(user_id, "user", api_key)
    resp = zot.create_items(items)
    created = list(resp.get("successful", {}).values())
    for it in created:
        zot.addto_collection(a.collection, it)
    print(f"\nadded {len(created)} item(s) to collection {a.collection}")
    for it in created:
        print(f"  {it['key']}  {it['data']['title'][:70]}")
    if resp.get("failed"):
        print(f"FAILED: {resp['failed']}")
        return 1
    print("\nnow run `make bib` in the document directory to regenerate references.bib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
