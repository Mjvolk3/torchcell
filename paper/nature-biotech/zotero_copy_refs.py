"""
paper/nature-biotech/zotero_copy_refs.py

Copy pinned-citekey references from the personal ("main") Zotero library into the
shared **torchcell** group library's target collection, so the group library stays
the single polished source that maintains references.bib.

Same pyzotero pattern as Swanki/Zendron. Credentials via env (load_dotenv):
  ZOTERO_API_KEY     - key with WRITE access to the group (zotero.org/settings/keys)
  ZOTERO_LIBRARY_ID  - your *user* library id
Copy those from Swanki/.env into torchcell/.env (or export them) before running.

Matching is by BetterBibTeX pinned citekey -- the Zotero 7 native `citationKey`
field, or `Citation Key: <key>` in `extra` (the same lookup Swanki uses). Because
the keys are pinned they carry over unchanged, so they keep matching the \\cite
commands in the manuscript.

Run it yourself (not CI). Preview first:
    python paper/nature-biotech/zotero_copy_refs.py --dry-run
    python paper/nature-biotech/zotero_copy_refs.py                   # the 15 annual-review refs
    python paper/nature-biotech/zotero_copy_refs.py KEY1 KEY2 ...     # specific citekeys
    python paper/nature-biotech/zotero_copy_refs.py --group torchcell --collection "torchcell paper"

Idempotent: items whose citekey is already in the target collection are skipped.
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from pyzotero import zotero

# The 15 refs pulled from the 2025 annual review into references.bib.
DEFAULT_KEYS = [
    "stephanopoulos1998me", "bordbar2014cbm", "presnell2019ml", "sapoval2022dl",
    "lam2014alcohol", "costanzo2016gin", "opgenorth2019dbtl", "eslami2022ai",
    "culley2020yeast", "zhang2020tryptophan", "stewart2022torchgeo",
    "ohya2005morphology", "kalfon2025scprint", "cui2024scgpt", "kuzmin2018trigenic",
]

# Item-`data` fields that must not carry into a freshly created item.
_DROP_FIELDS = {"key", "version", "dateAdded", "dateModified", "collections", "relations"}


def citekey_of(item: dict) -> str | None:
    """BetterBibTeX pinned citekey of an item (native field or `extra` line)."""
    data = item.get("data", {})
    if data.get("citationKey"):
        return data["citationKey"]
    for line in data.get("extra", "").splitlines():
        if line.startswith("Citation Key:"):
            return line.split(":", 1)[1].strip()
    return None


def find_in_user_library(zot: zotero.Zotero, citekey: str) -> dict | None:
    """Top-level user-library item whose pinned citekey is exactly `citekey`."""
    # qmode=everything searches `extra`, where BBT writes `Citation Key: <key>`.
    for item in zot.items(q=citekey, qmode="everything", limit=50):
        if item["data"].get("itemType") in ("attachment", "note"):
            continue
        if citekey_of(item) == citekey:
            return item
    return None


def resolve_group(zot_user: zotero.Zotero, name_substr: str) -> tuple[str, str]:
    """(group_id, name) for the single group whose name contains `name_substr`."""
    groups = zot_user.groups()
    matches = [g for g in groups if name_substr.lower() in g["data"]["name"].lower()]
    have = ", ".join(g["data"]["name"] for g in groups) or "(none)"
    if not matches:
        sys.exit(f"No group matching {name_substr!r}. You can access: {have}")
    if len(matches) > 1:
        hit = ", ".join(g["data"]["name"] for g in matches)
        sys.exit(f"Ambiguous group {name_substr!r} -> {hit}; pass --group exactly.")
    return str(matches[0]["id"]), matches[0]["data"]["name"]


def resolve_collection(zot_group: zotero.Zotero, name: str) -> str:
    """Collection key in the group whose name equals `name` (case-insensitive)."""
    cols = zot_group.everything(zot_group.collections())
    for c in cols:
        if c["data"]["name"].lower() == name.lower():
            return c["key"]
    have = ", ".join(c["data"]["name"] for c in cols) or "(none)"
    sys.exit(f"No collection {name!r} in the group. Collections: {have}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Copy pinned citekeys: main -> group collection.")
    ap.add_argument("citekeys", nargs="*", help="citekeys to copy (default: 15 annual-review refs)")
    ap.add_argument("--group", default="torchcell", help="substring of the target group name")
    ap.add_argument("--group-id", default=None, help="target group id (skips name discovery)")
    ap.add_argument("--collection", default="paper", help="exact target collection name")
    ap.add_argument("--dry-run", action="store_true", help="preview without writing")
    args = ap.parse_args()

    load_dotenv()
    api_key = os.getenv("ZOTERO_API_KEY")
    user_id = os.getenv("ZOTERO_USER_ID")               # personal ("main") library
    group_id = args.group_id or os.getenv("ZOTERO_LIBRARY_ID")  # target group
    if not (api_key and user_id):
        sys.exit("Set ZOTERO_API_KEY and ZOTERO_USER_ID (personal library id) in torchcell/.env.")

    keys = args.citekeys or DEFAULT_KEYS
    zot_user = zotero.Zotero(user_id, "user", api_key)
    if "user" not in zot_user.key_info().get("access", {}):
        sys.exit(
            "API key lacks personal-library read access. Enable 'Allow library access' for the "
            "personal library at https://www.zotero.org/settings/keys, then re-run."
        )
    if not group_id:
        group_id, group_name = resolve_group(zot_user, args.group)
    else:
        group_name = f"id={group_id}"
    zot_group = zotero.Zotero(group_id, "group", api_key)
    coll_key = resolve_collection(zot_group, args.collection)
    print(f"target: group {group_name!r} ({group_id}) / collection {args.collection!r} ({coll_key})")

    present = {
        citekey_of(it)
        for it in zot_group.everything(zot_group.collection_items_top(coll_key))
    }
    present.discard(None)

    to_create, missing, skipped = [], [], []
    for ck in keys:
        if ck in present:
            skipped.append(ck)
            continue
        item = find_in_user_library(zot_user, ck)
        if item is None:
            missing.append(ck)
            continue
        data = {k: v for k, v in item["data"].items() if k not in _DROP_FIELDS}
        data["collections"] = [coll_key]
        to_create.append(data)

    print(f"copy {len(to_create)} | skip-present {len(skipped)} | missing {len(missing)}")
    if skipped:
        print("  already present:", ", ".join(skipped))
    if missing:
        print("  MISSING in main library (check the pinned key):", ", ".join(missing))
    if args.dry_run:
        print("  [dry-run] would copy:", ", ".join(citekey_of({"data": d}) or "?" for d in to_create))
        return

    created = 0
    for i in range(0, len(to_create), 50):
        resp = zot_group.create_items(to_create[i : i + 50])
        created += len(resp.get("successful", {}))
        if resp.get("failed"):
            print("  FAILED:", resp["failed"])
    print(f"created {created} item(s) in {args.collection!r}.")


if __name__ == "__main__":
    main()
