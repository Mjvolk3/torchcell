# tests/torchcell/literature/_fake_zotero.py
"""A read-only pyzotero stand-in for the literature tests (not a test module).

It implements only the READ calls the library code makes (``collections``,
``everything``, ``follow``, ``items``, ``top``, ``collection_items``,
``collection_items_top``, ``children``, ``file``) and records each one. ``top()``
answers as the live ``/items/top`` endpoint does: every item whose ``data`` has no
``parentItem`` (a standalone attachment or note is top-level; a child is not).

Paging follows pyzotero 1.13: every list-returning read answers with at most
``page_size`` (default 100, pyzotero's ``limit``) rows and records a ``next`` link in
``links`` when more remain; ``follow()`` returns the next page; ``everything(query)``
extends the first page with ``follow()`` while ``links["next"]`` is set. A caller that
skips ``everything`` therefore sees only the first 100 rows, exactly as against the live
API. A BibTeX read (a ``BibDatabase``) is returned whole with no ``next`` link. It deliberately has NO write methods
(``create_collections``, ``create_items``, ``upload_attachments``, ...): a code path that
tries to write fails with ``AttributeError`` instead of pretending to succeed, so no test
can exercise a Zotero write, real or imitated.
"""

from __future__ import annotations

from typing import Any

import pytest
from bibtexparser.bibdatabase import BibDatabase
from pydantic import SecretStr

from torchcell.literature.zotero import ZoteroConfig, ZoteroLibrary


class FakeZot:
    """Records read calls and answers them from in-memory fixtures."""

    def __init__(
        self,
        library_id: str = "1",
        library_type: str = "group",
        api_key: str = "k",
        *,
        collections: list[dict[str, Any]] | None = None,
        items: list[dict[str, Any]] | None = None,
        collection_members: dict[str, list[dict[str, Any]]] | None = None,
        children: dict[str, list[dict[str, Any]]] | None = None,
        files: dict[str, bytes] | None = None,
        bibtex: dict[str | None, Any] | None = None,
        page_size: int = 100,
    ) -> None:
        self.init_args = (library_id, library_type, api_key)
        self._collections = collections or []
        self._items = items or []
        self._members = collection_members or {}
        self._children = children or {}
        self._files = files or {}
        self._bibtex = bibtex or {}
        self.calls: list[tuple[Any, ...]] = []
        self.page_size = page_size
        self.links: dict[str, str] = {}
        self._rest: list[dict[str, Any]] = []

    def _page(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """First page of ``rows``; the remainder waits behind a ``next`` link."""
        self._rest = rows[self.page_size :]
        self.links = {"next": "page"} if self._rest else {}
        return rows[: self.page_size]

    def _bib(self, key: str | None) -> Any:
        """A BibTeX read: returned whole (pyzotero parses it into one object)."""
        self.links = {}
        return self._bibtex[key]

    def follow(self) -> list[dict[str, Any]]:
        self.calls.append(("follow",))
        return self._page(self._rest)

    def everything(self, query: Any) -> Any:
        self.calls.append(("everything",))
        if not isinstance(query, list):
            return query
        rows = list(query)
        while self.links.get("next"):
            rows.extend(self.follow())
        return rows

    def collections(self) -> list[dict[str, Any]]:
        self.calls.append(("collections",))
        return self._page(self._collections)

    def items(self, format: str | None = None) -> Any:
        self.calls.append(("items", format))
        if format == "bibtex":
            return self._bib(None)
        return self._page(self._items)

    def top(self) -> list[dict[str, Any]]:
        self.calls.append(("top",))
        return self._page([i for i in self._items if "parentItem" not in i["data"]])

    def collection_items(self, key: str, format: str | None = None) -> Any:
        self.calls.append(("collection_items", key, format))
        if format == "bibtex":
            return self._bib(key)
        return self._page(self._members.get(key, []))

    def collection_items_top(self, key: str) -> list[dict[str, Any]]:
        self.calls.append(("collection_items_top", key))
        return self._page(self._members.get(key, []))

    def children(self, item_key: str) -> list[dict[str, Any]]:
        self.calls.append(("children", item_key))
        return self._page(self._children.get(item_key, []))

    def file(self, key: str) -> bytes:
        self.calls.append(("file", key))
        return self._files[key]


def bib_db(entries: list[dict[str, str]]) -> BibDatabase:
    """A BibDatabase holding ``entries``, as pyzotero returns for ``format="bibtex"``."""
    db = BibDatabase()
    db.entries = entries
    return db


def make_library(monkeypatch: pytest.MonkeyPatch, zot: FakeZot) -> ZoteroLibrary:
    """A ZoteroLibrary whose pyzotero client (stubbed at its import site) is ``zot``."""
    monkeypatch.setattr("torchcell.literature.zotero.zotero.Zotero", lambda *args: zot)
    return ZoteroLibrary(ZoteroConfig(library_id="1", api_key=SecretStr("k")))


def collection(key: str, name: str, parent: str | None = None) -> dict[str, Any]:
    """A Zotero collection record."""
    data: dict[str, Any] = {"name": name}
    if parent is not None:
        data["parentCollection"] = parent
    return {"key": key, "data": data}
