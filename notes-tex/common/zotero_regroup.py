#!/usr/bin/env python
# notes-tex/common/zotero_regroup.py
# [[notes-tex.common.zotero_regroup]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes-tex/common/zotero_regroup.py
"""Move flat ``torchcell/notes-tex/<slug>`` collections under their group.

``zotero_publish.py`` derives a document's Zotero collection from its repo path.
When the repo gained the group layer (``notes-tex/<group>/<slug>/``) every
collection that had been published flat was left at ``torchcell/notes-tex/<slug>``
with a ``Doc Key: notes-tex/<slug>`` marker on its parent item. Publishing the
moved document would then look for ``notes-tex/<group>/<slug>``, find nothing,
and start a second history beside the first. This script is the one-shot move
that prevents that, and it is idempotent, so it can be re-run as more documents
are grouped.

For every live collection directly under ``torchcell/notes-tex`` whose name is a
known slug it

* re-parents the collection under ``torchcell/notes-tex/<group>`` (created if
  missing),
* rewrites ``Doc Key: notes-tex/<slug>`` on each top-level item to the grouped
  path (a Doc Key that is not the flat path, such as a rendered Dendron note's
  ``notes/<fname>.md``, is left as it is), and
* files each item into the group collection as well, so the group reads as an
  index the way the ``notes-tex`` index already does.

Where the grouping comes from: the repo tree, ``notes-tex/<group>/<slug>/``, for
every document that has landed, plus ``--assign SLUG=GROUP`` for a document that
exists only in an unlanded worktree. A flat collection that neither names is
reported and left alone; the report prints the ``--assign`` to add.

Nothing is deleted, no attachment is touched, and ``--dry-run`` prints the plan
and writes nothing.

Usage::

    python notes-tex/common/zotero_regroup.py --dry-run
    python notes-tex/common/zotero_regroup.py \\
        --assign 019-simb-multimodal=multimodal \\
        --assign 031-betaxanthin-module=metabolism
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

from dotenv import load_dotenv
from pydantic import BaseModel
from pyzotero import zotero
from zotero_publish import (
    DEFAULT_PARENT_DIR,
    DOC_KEY_PREFIX,
    ROOT_COLLECTION,
    _git,
    find_collection,
    notes_tex_groups,
)


class ItemMove(BaseModel):
    """One parent item: what its Doc Key says now and what it will say after."""

    key: str
    title: str
    old_doc_key: str | None
    new_doc_key: str | None  # None when the key is not the flat path and stays


class CollectionMove(BaseModel):
    """One flat collection and where it goes."""

    slug: str
    group: str
    collection_key: str
    group_key: str | None  # None when the group collection does not exist yet
    items: list[ItemMove]


class Plan(BaseModel):
    """Everything a run would do, decided from reads alone so it can be printed first."""

    moves: list[CollectionMove]
    unassigned: list[str]  # flat collections no mapping names
    already_grouped: list[str]  # "<group>/<slug>" seen under a group already


def _doc_key(extra: str) -> str | None:
    for line in extra.splitlines():
        if line.startswith(DOC_KEY_PREFIX):
            return line[len(DOC_KEY_PREFIX):].strip()
    return None


def _rewrite_doc_key(extra: str, new_key: str) -> str:
    return "\n".join(
        f"{DOC_KEY_PREFIX} {new_key}" if line.startswith(DOC_KEY_PREFIX) else line
        for line in extra.splitlines()
    )


def plan(zot: zotero.Zotero, mapping: dict[str, str]) -> Plan:
    """Read the tree and decide every move. Reads only."""
    root = find_collection(zot, ROOT_COLLECTION, None)
    if not root:
        sys.exit(f"no top-level {ROOT_COLLECTION!r} collection in the personal library.")
    index = find_collection(zot, DEFAULT_PARENT_DIR, root)
    if not index:
        sys.exit(f"no {ROOT_COLLECTION}/{DEFAULT_PARENT_DIR} collection; nothing to regroup.")

    live = [c for c in zot.everything(zot.collections()) if not c["data"].get("deleted")]
    flat = {
        c["data"]["name"]: c["key"]
        for c in live
        if (c["data"].get("parentCollection") or None) == index
    }
    # A group is a flat collection the mapping names, or one that already has
    # child collections (a document collection never has children), so a group
    # that no document in this checkout belongs to is still read as a group and
    # not reported as an unassigned document.
    parents = {c["data"].get("parentCollection") for c in live}
    group_keys = {
        name: key for name, key in flat.items()
        if name in set(mapping.values()) or key in parents
    }
    groups_present = set(group_keys)

    moves: list[CollectionMove] = []
    unassigned: list[str] = []
    for name in sorted(flat):
        if name in groups_present:
            continue
        if name not in mapping:
            unassigned.append(name)
            continue
        group = mapping[name]
        items: list[ItemMove] = []
        for it in zot.everything(zot.collection_items_top(flat[name])):
            d = it["data"]
            old = _doc_key(d.get("extra") or "")
            new = f"{DEFAULT_PARENT_DIR}/{group}/{name}" if old == f"{DEFAULT_PARENT_DIR}/{name}" else None
            items.append(ItemMove(key=it["key"], title=d.get("title", ""),
                                  old_doc_key=old, new_doc_key=new))
        moves.append(CollectionMove(slug=name, group=group, collection_key=flat[name],
                                    group_key=group_keys.get(group), items=items))

    already: list[str] = []
    for group, gkey in group_keys.items():
        for c in live:
            if (c["data"].get("parentCollection") or None) == gkey:
                already.append(f"{group}/{c['data']['name']}")
    return Plan(moves=moves, unassigned=unassigned, already_grouped=sorted(already))


def apply(zot: zotero.Zotero, p: Plan, index_key: str, dry: bool) -> None:
    """Carry out the plan, creating group collections first so re-parenting can refer to them."""
    created: dict[str, str] = {}
    for mv in p.moves:
        if mv.group_key or mv.group in created:
            continue
        if dry:
            print(f"  [dry-run] would create group collection {mv.group!r}")
            created[mv.group] = f"<new {mv.group}>"
            continue
        resp = zot.create_collections([{"name": mv.group, "parentCollection": index_key}])
        created[mv.group] = resp["successful"]["0"]["key"]
        print(f"  created group collection {mv.group!r} ({created[mv.group]})")

    for mv in p.moves:
        gkey = mv.group_key or created[mv.group]
        print(f"  {mv.slug}  ->  {DEFAULT_PARENT_DIR}/{mv.group}/  ({mv.collection_key} under {gkey})")
        if not dry:
            coll = zot.collection(mv.collection_key)
            coll["data"]["parentCollection"] = gkey
            zot.update_collection(coll)
        for im in mv.items:
            # An item with no Doc Key was not published by zotero_publish.py: it
            # is someone else's paper dragged into the collection by hand. It moves
            # with its collection and is otherwise left exactly as it is.
            if im.old_doc_key is None:
                print(f"      item {im.key}  {im.title[:60]!r}  no Doc Key; not ours, left alone")
                continue
            change = f"Doc Key {im.old_doc_key} -> {im.new_doc_key}" if im.new_doc_key else f"Doc Key {im.old_doc_key} kept"
            print(f"      item {im.key}  {im.title[:60]!r}  {change}; filed into {mv.group}")
            if dry:
                continue
            item = zot.item(im.key)
            if im.new_doc_key:
                item["data"]["extra"] = _rewrite_doc_key(item["data"].get("extra") or "", im.new_doc_key)
            cols = set(item["data"].get("collections") or [])
            item["data"]["collections"] = sorted(cols | {gkey, mv.collection_key, index_key})
            zot.update_item(item)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--assign", action="append", default=[], metavar="SLUG=GROUP",
                    help="group for a document that is not in this checkout's tree")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    args = ap.parse_args()

    common_dir = osp.dirname(osp.abspath(__file__))
    repo = _git(common_dir, "rev-parse", "--show-toplevel")
    load_dotenv(osp.join(repo, ".env"))
    load_dotenv()
    user_id, api_key = os.getenv("ZOTERO_USER_ID"), os.getenv("ZOTERO_API_KEY")
    if not (user_id and api_key):
        sys.exit("Set ZOTERO_USER_ID and ZOTERO_API_KEY in repo-root .env.")
    zot = zotero.Zotero(user_id, "user", api_key)

    mapping = notes_tex_groups(repo)
    for spec in args.assign:
        if "=" not in spec:
            sys.exit(f"--assign takes SLUG=GROUP, got {spec!r}")
        slug, group = spec.split("=", 1)
        if slug in mapping and mapping[slug] != group:
            sys.exit(f"{slug} is under {mapping[slug]}/ in the tree; --assign says {group}.")
        mapping[slug] = group

    p = plan(zot, mapping)
    index_key = find_collection(zot, DEFAULT_PARENT_DIR, find_collection(zot, ROOT_COLLECTION, None))
    assert index_key is not None  # plan() exited otherwise

    print(f"{len(p.moves)} collections to move, {len(p.already_grouped)} already grouped, "
          f"{len(p.unassigned)} flat and unassigned\n")
    apply(zot, p, index_key, args.dry_run)
    if p.unassigned:
        print("\nleft flat under notes-tex (no group named for them):")
        for name in p.unassigned:
            print(f"  --assign {name}=<group>")
    if args.dry_run:
        print("\n[dry-run] nothing changed.")


if __name__ == "__main__":
    main()
