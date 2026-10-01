# tests/torchcell/literature/test_zotero.py
# [[tests.torchcell.literature.test_zotero]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_zotero.py
"""ZoteroLibrary read paths over a recording, read-only pyzotero stand-in.

``pyzotero.zotero.Zotero`` is replaced at its import site
(``torchcell.literature.zotero.zotero``) by :class:`FakeZot` from ``_fake_zotero.py``,
which has no write methods, so nothing here can create, modify or upload a Zotero item.
The collection-creating branch of ``collection_key(create_if_missing=True)`` is a write
and is deliberately not exercised; ``create_if_missing=True`` is called only on a name
that exists, where the fake's missing ``create_collections`` would raise if a create
were attempted.

Paging: ``FakeZot`` answers at most 100 rows per request and pages through
``everything``/``follow`` like pyzotero, so a lookup that reads only the first page
misses a collection at position 120 of a 150-collection library (issue #563).

Retry arithmetic: delay = ``base_delay * 2 ** attempt + uniform(0, base_delay)``; with
``uniform`` stubbed to 0.25 and ``base_delay`` 2.0 the two delays before the third
attempt are 2.25 and 4.25.

The generated citation key for authors Ada Lovelace + Bob Smith, date 2020-05-01, title
"On Analytical Engines" is ``lovelaceAnalyticalEngines2020`` (first author's last name,
the title's significant words, the year).
"""

import logging
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

import torchcell.literature.zotero as zotero_module
from tests.torchcell.literature._fake_zotero import FakeZot, collection, make_library
from torchcell.literature.zotero import (
    CollectionNode,
    ZoteroConfig,
    ZoteroLibrary,
    _is_main_article,
    _is_retryable,
    _is_si_attachment,
    _resolve_citation_key,
    make_zotero_client,
    with_zotero_retry,
)

REQUEST = httpx.Request("GET", "https://api.zotero.invalid/items")


def _status_error(code: int) -> httpx.HTTPStatusError:
    return httpx.HTTPStatusError(
        f"{code}", request=REQUEST, response=httpx.Response(code, request=REQUEST)
    )


def _pdf(key: str, title: str = "", filename: str = "") -> dict[str, Any]:
    return {
        "key": key,
        "data": {
            "contentType": "application/pdf",
            "title": title,
            "filename": filename,
        },
    }


def test_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Library id + key from the environment; type "user" honored, anything else group."""
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "6582362")
    monkeypatch.setenv("ZOTERO_API_KEY", "secret")
    monkeypatch.delenv("ZOTERO_LIBRARY_TYPE", raising=False)
    config = ZoteroConfig.from_env()
    assert (config.library_id, config.library_type) == ("6582362", "group")
    assert config.api_key.get_secret_value() == "secret"
    assert "secret" not in repr(config)
    monkeypatch.setenv("ZOTERO_LIBRARY_TYPE", "user")
    assert ZoteroConfig.from_env().library_type == "user"
    monkeypatch.setenv("ZOTERO_LIBRARY_TYPE", "organization")
    assert ZoteroConfig.from_env().library_type == "group"


def test_config_from_env_requires_both_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing id or a missing key raises KeyError naming it; no unauthenticated
    fallback.
    """
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
    monkeypatch.setenv("ZOTERO_API_KEY", "secret")
    with pytest.raises(KeyError, match="ZOTERO_LIBRARY_ID"):
        ZoteroConfig.from_env()
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "6582362")
    monkeypatch.delenv("ZOTERO_API_KEY")
    with pytest.raises(KeyError, match="ZOTERO_API_KEY"):
        ZoteroConfig.from_env()


def test_make_client_and_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The client receives (id, type, unwrapped key); ``from_env`` wires the same path."""
    made: list[tuple[str, ...]] = []

    def factory(*args: str) -> FakeZot:
        made.append(args)
        return FakeZot(*args)

    monkeypatch.setattr("torchcell.literature.zotero.zotero.Zotero", factory)
    config = ZoteroConfig(library_id="42", library_type="user", api_key=SecretStr("s"))
    client = make_zotero_client(config)
    assert isinstance(client, FakeZot) and client.init_args == ("42", "user", "s")
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "7")
    monkeypatch.setenv("ZOTERO_API_KEY", "t")
    monkeypatch.delenv("ZOTERO_LIBRARY_TYPE", raising=False)
    lib = ZoteroLibrary.from_env()
    assert made == [("42", "user", "s"), ("7", "group", "t")]
    assert lib.config.library_id == "7"


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (httpx.ReadTimeout("slow", request=REQUEST), True),
        (httpx.ConnectError("down", request=REQUEST), True),
        (_status_error(503), True),
        (_status_error(500), True),
        (_status_error(404), False),
        (_status_error(429), False),
        (ValueError("x"), False),
    ],
)
def test_is_retryable(exc: Exception, expected: bool) -> None:
    """Timeouts, transport errors and 500/502/503/504 retry; 4xx and others do not."""
    assert _is_retryable(exc) is expected


def test_retry_succeeds_after_transient_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two 503s then a value: delays 2*1 + 0.25 and 2*2 + 0.25."""
    monkeypatch.setattr("torchcell.literature.zotero.random.uniform", lambda a, b: 0.25)
    outcomes: list[Exception | str] = [_status_error(503), _status_error(503), "ok"]
    delays: list[float] = []

    def call() -> str:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert with_zotero_retry(call, sleep=delays.append) == "ok"
    assert delays == [2.25, 4.25]


def test_retry_exhausts_and_reraises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every attempt times out: ``max_tries`` calls, ``max_tries - 1`` sleeps, reraise."""
    monkeypatch.setattr("torchcell.literature.zotero.random.uniform", lambda a, b: 0.0)
    calls: list[int] = []
    delays: list[float] = []

    def call() -> str:
        calls.append(1)
        raise httpx.ReadTimeout("slow", request=REQUEST)

    with pytest.raises(httpx.ReadTimeout, match="slow"):
        with_zotero_retry(call, max_tries=3, base_delay=1.0, sleep=delays.append)
    assert (len(calls), delays) == (3, [1.0, 2.0])


def test_retry_does_not_retry_a_404() -> None:
    """A terminal 4xx raises on the first attempt without sleeping."""
    delays: list[float] = []

    def call() -> str:
        raise _status_error(404)

    with pytest.raises(httpx.HTTPStatusError, match="404"):
        with_zotero_retry(call, sleep=delays.append)
    assert delays == []


@pytest.mark.parametrize(
    ("title", "filename", "is_si", "is_main"),
    [
        ("Full Text PDF", "paper.pdf", False, True),
        ("Supplementary Information", "", True, False),
        ("", "41586_2020_MOESM1_ESM.pdf", True, False),
        ("SI-Appendix", "", True, False),
        ("", "data_si1.pdf", True, False),
        ("Network analysis", "analysis.pdf", False, True),
        ("Full text with supplement", "", True, True),
    ],
)
def test_si_and_main_article_classification(
    title: str, filename: str, is_si: bool, is_main: bool
) -> None:
    """Word indicators match anywhere; bare "si" only when delimited; "full text" wins."""
    att = _pdf("K", title, filename)
    assert (_is_si_attachment(att), _is_main_article(att)) == (is_si, is_main)


def test_resolve_citation_key_order() -> None:
    """Native field, then ``Citation Key:`` in extra, then generated from metadata."""
    native = {"data": {"citationKey": "nativeKey2021", "extra": "Citation Key: other"}}
    extra = {"data": {"extra": "tex.x: 1\nCitation Key: extraKey2022\n"}}
    generated = {
        "data": {
            "creators": [
                {"creatorType": "author", "firstName": "Ada", "lastName": "Lovelace"},
                {"creatorType": "editor", "firstName": "Ed", "lastName": "Itor"},
                {"creatorType": "author", "firstName": "Bob", "lastName": "Smith"},
            ],
            "date": "2020-05-01",
            "title": "On Analytical Engines",
        }
    }
    assert [_resolve_citation_key(i) for i in (native, extra, generated)] == [
        "nativeKey2021",
        "extraKey2022",
        "lovelaceAnalyticalEngines2020",
    ]


def test_collection_key_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Case-insensitive name match; a miss lists the sorted available names."""
    zot = FakeZot(collections=[collection("K1", "torchcell"), collection("K2", "Beta")])
    lib = make_library(monkeypatch, zot)
    assert lib.collection_key("TorchCell") == "K1"
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Zotero collection 'missing' not found. Available: ['Beta', 'torchcell']."
        ),
    ):
        lib.collection_key("missing")


def _big_library(target_index: int = 120, n: int = 150) -> list[dict[str, Any]]:
    """``n`` top-level collections ``C000``.. named ``coll-000``.., with the one at
    ``target_index`` renamed ``torchcell`` (key ``TCROOT01``).
    """
    rows = [collection(f"C{i:03d}", f"coll-{i:03d}") for i in range(n)]
    rows[target_index] = collection("TCROOT01", "torchcell")
    return rows


def test_list_collections_reads_every_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """All 150 collections come back in API order: page one plus one followed page."""
    rows = _big_library()
    zot = FakeZot(collections=rows)
    lib = make_library(monkeypatch, zot)
    assert [c["key"] for c in lib.list_collections()] == [c["key"] for c in rows]
    assert zot.calls == [("collections",), ("everything",), ("follow",)]


def test_collection_key_finds_name_past_first_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A collection at position 120 (page two) resolves; this failed before #563."""
    lib = make_library(monkeypatch, FakeZot(collections=_big_library()))
    assert lib.collection_key("torchcell") == "TCROOT01"


def test_collection_key_create_if_missing_finds_page_two_and_creates_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``create_if_missing=True`` on a name past page one returns its key, no create.

    ``FakeZot`` has no ``create_collections``, so a create attempt would raise
    ``AttributeError``; the recorded calls are reads only.
    """
    zot = FakeZot(collections=_big_library())
    lib = make_library(monkeypatch, zot)
    assert lib.collection_key("torchcell", create_if_missing=True) == "TCROOT01"
    assert zot.calls == [("collections",), ("everything",), ("follow",)]


def test_collection_key_refuses_ambiguous_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two collections sharing a name: a refusal naming each key and its parent path."""
    zot = FakeZot(
        collections=[
            collection("R", "torchcell"),
            collection("T", "torchcell-topics", parent="R"),
            collection("N", "notes-tex", parent="R"),
            collection("M1", "microbe-perturb-seq", parent="T"),
            collection("M2", "microbe-perturb-seq", parent="N"),
            collection("M3", "Microbe-Perturb-Seq"),
        ]
    )
    lib = make_library(monkeypatch, zot)
    with pytest.raises(ValueError) as excinfo:
        lib.collection_key("microbe-perturb-seq")
    assert str(excinfo.value) == (
        "Zotero collection 'microbe-perturb-seq' is ambiguous: 3 collections share "
        "the name: M1 under torchcell/torchcell-topics; M2 under torchcell/notes-tex; "
        "M3 at the top level. Address it by collection key."
    )


def test_collection_tree_root_and_children_past_first_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Root at position 120 and its child at 149 are both found from one paged listing."""
    rows = _big_library()
    rows[149] = collection("TCKID001", "torchcell-topics", parent="TCROOT01")
    zot = FakeZot(collections=rows)
    lib = make_library(monkeypatch, zot)
    assert lib.collection_tree("torchcell") == [
        CollectionNode(key="TCROOT01", name="torchcell", path="torchcell"),
        CollectionNode(
            key="TCKID001", name="torchcell-topics", path="torchcell/torchcell-topics"
        ),
    ]
    assert zot.calls == [("collections",), ("everything",), ("follow",)]


def test_collection_tree_depth_first_with_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Root first, children sorted by key, each with its slash-joined path."""
    zot = FakeZot(
        collections=[
            collection("R", "torchcell"),
            collection("B", "topics", parent="R"),
            collection("A", "paper", parent="R"),
            collection("C", "deep-set", parent="B"),
            collection("X", "unrelated"),
        ]
    )
    lib = make_library(monkeypatch, zot)
    assert lib.collection_tree("torchcell") == [
        CollectionNode(key="R", name="torchcell", path="torchcell"),
        CollectionNode(key="A", name="paper", path="torchcell/paper"),
        CollectionNode(key="B", name="topics", path="torchcell/topics"),
        CollectionNode(key="C", name="deep-set", path="torchcell/topics/deep-set"),
    ]


def test_find_item_by_doi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exact DOI match after strip + lowercase; blank DOI short-circuits with no call."""
    paper = {"key": "P1", "data": {"DOI": " 10.1000/ABC "}}
    zot = FakeZot(items=[{"key": "N", "data": {}}, paper])
    lib = make_library(monkeypatch, zot)
    assert lib.find_item_by_doi("   ") is None
    assert zot.calls == []
    assert lib.find_item_by_doi("10.1000/abc") is paper
    assert lib.doi_in_library("10.1000/ABC") is True
    assert lib.doi_in_library("10.1000/zzz") is False


def test_pdf_attachments_main_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-PDF children dropped; the main article sorts ahead of SI (stable order)."""
    si1 = _pdf("S1", "Supplementary Data 1")
    main = _pdf("M", "Full Text PDF")
    si2 = _pdf("S2", "", "moesm2.pdf")
    html = {"key": "H", "data": {"contentType": "text/html", "title": "Snapshot"}}
    zot = FakeZot(children={"ITEM": [si1, html, main, si2]})
    lib = make_library(monkeypatch, zot)
    assert [a["key"] for a in lib.pdf_attachments("ITEM")] == ["M", "S1", "S2"]


def test_download_artifact_writes_paper_and_si(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """paper.pdf from the main attachment, si/si1.pdf and si/si2.pdf from the rest."""
    zot = FakeZot(
        children={
            "ITEM": [
                _pdf("S1", "Supplement 1"),
                _pdf("M", "Full Text PDF"),
                _pdf("S2", "SI-2"),
            ]
        },
        files={"M": b"main", "S1": b"one", "S2": b"two"},
    )
    lib = make_library(monkeypatch, zot)
    item = {"key": "ITEM", "data": {"citationKey": "keyA2024"}}
    out = lib.download_artifact(item, data_root=tmp_path, library_dirname="lib")
    assert out == tmp_path / "lib" / "keyA2024"
    files = {
        p.relative_to(out).as_posix(): p.read_bytes()
        for p in sorted(out.rglob("*"))
        if p.is_file()
    }
    assert files == {"paper.pdf": b"main", "si/si1.pdf": b"one", "si/si2.pdf": b"two"}
    assert [c for c in zot.calls if c[0] == "file"] == [
        ("file", "M"),
        ("file", "S1"),
        ("file", "S2"),
    ]


def test_download_artifact_defaults_to_data_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no data_root the mirror is ``$DATA_ROOT/torchcell-library/<key>``."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    lib = make_library(monkeypatch, FakeZot())
    out = lib.download_artifact({"key": "ITEM", "data": {"citationKey": "k2020"}})
    assert out == tmp_path / "torchcell-library" / "k2020"
    assert list(out.iterdir()) == []


def test_smoke_test_logs_collections(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``_smoke_test`` lists collections sorted case-insensitively (dotenv stubbed)."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "9")
    monkeypatch.setenv("ZOTERO_API_KEY", "t")
    monkeypatch.delenv("ZOTERO_LIBRARY_TYPE", raising=False)
    zot = FakeZot(collections=[collection("K2", "beta"), collection("K1", "Alpha")])
    monkeypatch.setattr("torchcell.literature.zotero.zotero.Zotero", lambda *args: zot)
    with caplog.at_level(logging.INFO, logger=zotero_module.log.name):
        zotero_module._smoke_test()
    assert [r.getMessage() for r in caplog.records] == [
        "Connected to Zotero group library 9",
        "Found 2 collection(s):",
        "  - Alpha (key=K1)",
        "  - beta (key=K2)",
    ]
