# tests/torchcell/literature/test_bib_store.py
# [[tests.torchcell.literature.test_bib_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_bib_store.py
"""Tests for the served bibliography store (export side) and its /bib endpoint.

Zotero is never contacted: the pull is monkeypatched to return canned entries, so
what is exercised is spec discovery from the repo's Makefiles, the atomic write +
manifest, the content-stable hash, and the endpoint's hash header.

2026.09.30 (Phase 14). Added, with the three ``torchcell.literature.bib`` fetchers
replaced by recorders (no Zotero, no network):

- ``fetch_scope_entries`` dispatch on each scope shape (paired, tree, whole group,
  a group collection key), which library each fetcher receives, the paired shape
  winning over a tree, and the refusal of a personal collection with no group one;
- the exact bytes of a store written from three specs (manuscript, notes-tex pair,
  Dendron tree): banner lines, BibTeX body sorted case-insensitively by key, the
  directory listing, and every manifest row (bytes, sha256, entries, scope, origin,
  stamp) plus ``manifest.json`` as written;
- refusals: the exact 0-entry message, an illegal spec name raised before any pull,
  ``load_bib_store`` on an empty mirror;
- ``discover_bib_specs`` on Makefiles using ``=``, ``?=`` and ``:=``, an empty personal
  collection, a repeated assignment (the last wins), a dot directory (refused by name).

2026.10.01 (issue #529) retired four Findings; now asserted: scope collections are
KEYS by declaration (a value that is not a key is refused by value, and an 8-character
upper-case value is sent as the key it is declared to be, never classified by shape);
a failed export leaves no ``.part`` file; a spec dropped from the repo has its ``.bib``
moved to ``_bib/_retired/<generated_at>/`` with a warning naming it; an inline
``# comment`` after a Makefile value is cut off, and a two-word value or a name is
refused naming the Makefile. After review of PR #589: a ``--name`` subset run carries
every other declared bibliography's record forward unchanged (refusing a record whose
file is missing or edited), and ``generated_at`` must be the exporter's own timestamp
format, since it names a directory.
"""

import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import torchcell.literature.bib_store as store_mod
from torchcell.literature.bib_store import (
    BIB_STORE_MANIFEST,
    BibScope,
    BibSpec,
    BibStoreManifest,
    bib_store_dir,
    discover_bib_specs,
    export_bib_store,
    load_bib_store,
    parse_makefile_collections,
    validate_bib_name,
    validate_generated_at,
)
from torchcell.literature.server import (
    LiteratureKeys,
    LiteratureServerConfig,
    create_app,
)
from torchcell.literature.zotero import ZoteroLibrary

KEY = "test-key"
HEADERS = {"X-API-Key": KEY}
# The pull is monkeypatched, so no library is ever touched.
NO_LIB = cast(ZoteroLibrary, None)
Canned = dict[str, list[dict[str, Any]]]
# Stamps in the exporter's own format, datetime.now(UTC).isoformat().
T0 = "2026-01-01T00:00:00+00:00"
T1 = "2026-01-02T03:04:05.123456+00:00"


def _entry(key: str, title: str) -> dict[str, str]:
    return {
        "ID": key,
        "ENTRYTYPE": "article",
        "author": "Zotero, Zed",
        "title": title,
        "year": "2020",
    }


def _spec(name: str, **scope: str) -> BibSpec:
    return BibSpec(
        name=name, scope=BibScope(group_library_id="6582362", **scope), origin="test"
    )


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """A repo skeleton with two notes-tex documents, one of which cites nothing."""
    cited = tmp_path / "notes-tex" / "eqtl-data-model"
    cited.mkdir(parents=True)
    (cited / "Makefile").write_text(
        "ZOTERO_COLLECTION          := VNDH4NMX\n"
        "ZOTERO_PERSONAL_COLLECTION := 4VNJWJAW\n"
        "include ../common/Makefile.common\n"
    )
    silent = tmp_path / "notes-tex" / "w019-strain-build-list"
    silent.mkdir(parents=True)
    (silent / "Makefile").write_text(
        "# No ZOTERO_COLLECTION: the document cites nothing\n"
        "include ../common/Makefile.common\n"
    )
    return tmp_path


@pytest.fixture
def canned_pull(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict[str, Any]]]:
    """Replace the Zotero pull with a per-name canned response."""
    canned: dict[str, list[dict[str, Any]]] = {}

    def fake_fetch(scope: BibScope, group: Any, user: Any) -> list[dict[str, Any]]:
        return canned[scope.group_collection or "*"]

    monkeypatch.setattr(store_mod, "fetch_scope_entries", fake_fetch)
    return canned


def test_parse_makefile_collections(fake_repo: Path) -> None:
    makefile = fake_repo / "notes-tex" / "eqtl-data-model" / "Makefile"
    assert parse_makefile_collections(makefile) == ("VNDH4NMX", "4VNJWJAW")
    silent = fake_repo / "notes-tex" / "w019-strain-build-list" / "Makefile"
    assert parse_makefile_collections(silent) == ("", "")


def test_discover_specs_reads_repo_declarations(fake_repo: Path) -> None:
    specs = discover_bib_specs(
        fake_repo, group_library_id="6582362", user_library_id="1234"
    )
    by_name = {s.name: s for s in specs}
    # paper first, library last, one per citing notes-tex document in between.
    assert [s.name for s in specs] == ["paper", "eqtl-data-model", "library"]
    assert by_name["paper"].scope.group_collection == "W46ATS7B"
    assert by_name["paper"].scope.user_collection is None
    doc = by_name["eqtl-data-model"].scope
    assert (doc.group_collection, doc.user_collection) == ("VNDH4NMX", "4VNJWJAW")
    assert doc.user_library_id == "1234"
    assert by_name["eqtl-data-model"].origin == "notes-tex/eqtl-data-model/Makefile"
    assert by_name["library"].scope.user_root_collection == "torchcell"


def test_name_validation_rejects_path_separators() -> None:
    assert validate_bib_name("024-perturb-seq-costing") == "024-perturb-seq-costing"
    for bad in ("../x", "a/b", ".hidden", ""):
        with pytest.raises(ValueError):
            validate_bib_name(bad)


def test_export_writes_files_and_manifest(tmp_path: Path, canned_pull: Canned) -> None:
    canned_pull["W46ATS7B"] = [_entry("b2020", "Second"), _entry("a2020", "First")]
    spec = _spec("paper", group_collection="W46ATS7B")

    manifest = export_bib_store(
        tmp_path, [spec], NO_LIB, NO_LIB, declared=[spec], generated_at=T0
    )

    store = bib_store_dir(tmp_path)
    text = (store / "paper.bib").read_text()
    assert text.startswith("% GENERATED by torchcell.literature.bib_store")
    assert "% scope: group 6582362/W46ATS7B" in text
    assert text.index("a2020") < text.index("b2020")  # sorted by key
    assert not list(store.glob("*.part"))
    record = manifest.get("paper")
    assert record is not None
    assert record.n_entries == 2
    assert record.sha256 == hashlib.sha256(text.encode()).hexdigest()
    assert load_bib_store(tmp_path) == manifest
    assert (store / BIB_STORE_MANIFEST).is_file()


def test_export_is_content_stable_across_runs(
    tmp_path: Path, canned_pull: Canned
) -> None:
    """Same entries on a later date -> same bytes and sha256; only the stamp moves."""
    canned_pull["W46ATS7B"] = [_entry("a2020", "First")]
    spec = _spec("paper", group_collection="W46ATS7B")
    first = export_bib_store(
        tmp_path, [spec], NO_LIB, NO_LIB, declared=[spec], generated_at=T0
    )
    second = export_bib_store(
        tmp_path, [spec], NO_LIB, NO_LIB, declared=[spec], generated_at=T1
    )
    a, b = first.get("paper"), second.get("paper")
    assert a is not None and b is not None
    assert a.sha256 == b.sha256
    assert second.generated_at == T1


def test_export_refuses_empty_pull_and_keeps_previous_store(
    tmp_path: Path, canned_pull: Canned
) -> None:
    """A 0-entry pull raises before the served files or manifest change."""
    canned_pull["W46ATS7B"] = [_entry("a2020", "First")]
    spec = _spec("paper", group_collection="W46ATS7B")
    before = export_bib_store(
        tmp_path, [spec], NO_LIB, NO_LIB, declared=[spec], generated_at=T0
    )

    canned_pull["W46ATS7B"] = []
    with pytest.raises(RuntimeError, match="0 entries"):
        export_bib_store(
            tmp_path, [spec], NO_LIB, NO_LIB, declared=[spec], generated_at=T1
        )

    assert load_bib_store(tmp_path) == before
    assert not list(bib_store_dir(tmp_path).glob("*.part"))


def test_partial_failure_leaves_previous_store_intact(
    tmp_path: Path, canned_pull: Canned
) -> None:
    """If the second spec fails, the first spec's new file is NOT swapped in."""
    canned_pull["W46ATS7B"] = [_entry("a2020", "First")]
    canned_pull["VNDH4NMX"] = [_entry("q2020", "Doc")]
    paper = _spec("paper", group_collection="W46ATS7B")
    doc = _spec("eqtl-data-model", group_collection="VNDH4NMX")
    before = export_bib_store(
        tmp_path, [paper, doc], NO_LIB, NO_LIB, declared=[paper, doc], generated_at=T0
    )

    canned_pull["W46ATS7B"] = [_entry("a2020", "First"), _entry("z2021", "Added")]
    canned_pull["VNDH4NMX"] = []
    with pytest.raises(RuntimeError):
        export_bib_store(
            tmp_path,
            [paper, doc],
            NO_LIB,
            NO_LIB,
            declared=[paper, doc],
            generated_at=T1,
        )

    assert load_bib_store(tmp_path) == before
    paper_text = (bib_store_dir(tmp_path) / "paper.bib").read_text()
    assert "z2021" not in paper_text
    assert sorted(p.name for p in bib_store_dir(tmp_path).iterdir()) == [
        "eqtl-data-model.bib",
        "manifest.json",
        "paper.bib",
    ]


# --- endpoint -----------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path, canned_pull: Canned) -> TestClient:
    (tmp_path / "fakePaperKey2020").mkdir()
    (tmp_path / "_sync_reports").mkdir()
    canned_pull["W46ATS7B"] = [_entry("a2020", "First")]
    paper = [_spec("paper", group_collection="W46ATS7B")]
    export_bib_store(tmp_path, paper, NO_LIB, NO_LIB, declared=paper)
    config = LiteratureServerConfig(
        mirror_root=tmp_path, keys=LiteratureKeys.from_pairs(f"t:{KEY}"), port=8899
    )
    return TestClient(create_app(config))


def test_service_dirs_are_not_citation_keys(client: TestClient) -> None:
    assert client.get("/keys", headers=HEADERS).json()["citation_keys"] == [
        "fakePaperKey2020"
    ]
    health = client.get("/health").json()
    assert health["n_keys"] == 1
    assert health["n_bibs"] == 1


def test_bib_listing_is_the_store_manifest(client: TestClient) -> None:
    assert client.get("/bib").status_code == 401
    resp = client.get("/bib", headers=HEADERS)
    assert resp.status_code == 200
    manifest = BibStoreManifest.model_validate(resp.json())
    assert [b.name for b in manifest.bibs] == ["paper"]


def test_bib_download_matches_manifest_sha256(client: TestClient) -> None:
    record = BibStoreManifest.model_validate(
        client.get("/bib", headers=HEADERS).json()
    ).get("paper")
    assert record is not None
    resp = client.get("/bib/paper", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.headers["X-Artifact-SHA256"] == record.sha256
    assert hashlib.sha256(resp.content).hexdigest() == record.sha256
    assert b"@article{a2020" in resp.content


def test_bib_unknown_and_illegal_names(client: TestClient) -> None:
    assert client.get("/bib/nope", headers=HEADERS).status_code == 404
    assert client.get("/bib/..%2Fmanifest.json", headers=HEADERS).status_code in (
        400,
        404,
    )


def test_bib_404_when_no_store(tmp_path: Path) -> None:
    config = LiteratureServerConfig(
        mirror_root=tmp_path, keys=LiteratureKeys.from_pairs(f"t:{KEY}"), port=8899
    )
    c = TestClient(create_app(config))
    assert c.get("/bib", headers=HEADERS).status_code == 404
    assert c.get("/health").json()["n_bibs"] == 0


# --- scope dispatch, exact store bytes, refusals (Phase 14) --------------------- #


class _Lib:
    """A stand-in library: only its label is read, to see which one a fetcher got."""

    def __init__(self, label: str) -> None:
        self.label = label


GROUP = cast(ZoteroLibrary, _Lib("group"))
USER = cast(ZoteroLibrary, _Lib("user"))


@pytest.fixture
def fetch_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Replace the three bib fetchers with recorders returning one entry each."""
    calls: list[tuple[Any, ...]] = []

    def single(lib: Any, collection: Any = None, *, collection_key: Any = None) -> Any:
        calls.append(("single", lib.label, collection, collection_key))
        return [_entry("s2020", "Single")]

    def paired(
        group: Any,
        user: Any,
        *,
        group_collection: str,
        user_collection: str,
        as_keys: bool = False,
    ) -> Any:
        calls.append(
            (
                "paired",
                group.label,
                user.label,
                group_collection,
                user_collection,
                as_keys,
            )
        )
        return [_entry("p2020", "Paired")]

    def union(group: Any, user: Any, *, user_root_collection: str) -> Any:
        calls.append(("union", group.label, user.label, user_root_collection))
        return [_entry("u2020", "Union")]

    monkeypatch.setattr(store_mod, "fetch_bibtex_entries", single)
    monkeypatch.setattr(store_mod, "fetch_paired_collection_entries", paired)
    monkeypatch.setattr(store_mod, "fetch_union_bibtex_entries", union)
    return calls


def test_fetch_scope_entries_dispatches_on_the_scope_shape(
    fetch_calls: list[tuple[Any, ...]],
) -> None:
    """Paired (group + personal collection) -> the paired pull with the group library
    first and ``as_keys=True``; a personal tree -> the union pull; no collection -> the
    whole group; a group collection -> ``collection_key``. A scope carrying both a
    personal collection and a tree takes the paired pull (the first branch).
    """
    scopes = [
        BibScope(
            group_library_id="6582362",
            group_collection="VNDH4NMX",
            user_library_id="1234",
            user_collection="4VNJWJAW",
        ),
        BibScope(
            group_library_id="6582362",
            user_library_id="1234",
            user_root_collection="torchcell",
        ),
        BibScope(group_library_id="6582362"),
        BibScope(group_library_id="6582362", group_collection="W46ATS7B"),
        BibScope(
            group_library_id="6582362",
            group_collection="VNDH4NMX",
            user_library_id="1234",
            user_collection="4VNJWJAW",
            user_root_collection="torchcell",
        ),
    ]
    results = [store_mod.fetch_scope_entries(s, GROUP, USER) for s in scopes]
    assert [r[0]["ID"] for r in results] == [
        "p2020",
        "u2020",
        "s2020",
        "s2020",
        "p2020",
    ]
    assert fetch_calls == [
        ("paired", "group", "user", "VNDH4NMX", "4VNJWJAW", True),
        ("union", "group", "user", "torchcell"),
        ("single", "group", None, None),
        ("single", "group", None, "W46ATS7B"),
        ("paired", "group", "user", "VNDH4NMX", "4VNJWJAW", True),
    ]


def test_scope_collections_are_keys_by_declaration(
    fetch_calls: list[tuple[Any, ...]],
) -> None:
    """``group_collection`` and ``user_collection`` are declared KEYS, so nothing is
    classified by shape: ``RNASEQ01`` is sent as the key it is declared to be, and a
    value that cannot be a key (a name, a 7-character or lower-case string) is refused
    when the scope is built, with the exact message naming the value and the field.
    """
    scope = BibScope(group_library_id="6582362", group_collection="RNASEQ01")
    assert store_mod.fetch_scope_entries(scope, GROUP, USER) == [
        _entry("s2020", "Single")
    ]
    assert fetch_calls == [("single", "group", None, "RNASEQ01")]
    refused = [
        ("group_collection", "microbe-perturb-seq"),
        ("group_collection", "ABCDEFG"),
        ("user_collection", "w46ats7b"),
    ]
    for field, value in refused:
        with pytest.raises(ValidationError) as excinfo:
            BibScope(
                **{
                    "group_library_id": "6582362",
                    "group_collection": "VNDH4NMX",
                    field: value,
                }
            )
        [error] = excinfo.value.errors()
        assert (error["loc"], error["msg"]) == (
            (field,),
            "Value error, not a Zotero collection key (8 upper-case letters or "
            f"digits): {value!r}; bibliography scopes address collections by key",
        )


def test_personal_collection_without_a_group_collection_is_refused(
    fetch_calls: list[tuple[Any, ...]],
) -> None:
    scope = BibScope(
        group_library_id="6582362", user_library_id="1234", user_collection="4VNJWJAW"
    )
    with pytest.raises(
        ValueError, match="^a personal collection needs a group collection to pair$"
    ):
        store_mod.fetch_scope_entries(scope, GROUP, USER)
    assert fetch_calls == []


_BODY_P = "@article{p2020,\n  author = {Zotero, Zed},\n  title = {Paired},\n  year = {2020}\n}\n"
_BODY_S = "@article{s2020,\n  author = {Zotero, Zed},\n  title = {Single},\n  year = {2020}\n}\n"
_BODY_U = "@article{u2020,\n  author = {Zotero, Zed},\n  title = {Union},\n  year = {2020}\n}\n"


def _three_specs() -> list[BibSpec]:
    return [
        BibSpec(
            name="paper",
            scope=BibScope(group_library_id="6582362", group_collection="W46ATS7B"),
            origin="paper/nature-biotech/zotero_export_bib.py",
        ),
        BibSpec(
            name="eqtl-data-model",
            scope=BibScope(
                group_library_id="6582362",
                group_collection="VNDH4NMX",
                user_library_id="1234",
                user_collection="4VNJWJAW",
            ),
            origin="notes-tex/eqtl-data-model/Makefile",
        ),
        BibSpec(
            name="library",
            scope=BibScope(
                group_library_id="6582362",
                user_library_id="1234",
                user_root_collection="torchcell",
            ),
            origin="scripts/lit_bib.py",
        ),
    ]


def test_export_writes_exact_files_and_manifest_rows(
    tmp_path: Path, fetch_calls: list[tuple[Any, ...]]
) -> None:
    """Each file is the banner (name and entry count, the scope parts joined by
    `` + ``, the declaring file, the served path) then the BibTeX body. The paper scope
    has no personal part; the pair adds ``personal 1234/4VNJWJAW``; the tree adds
    ``personal 1234/torchcell/** (tree)`` and its group part is ``6582362/*``. Each
    manifest row's ``bytes`` and ``sha256`` are of these exact texts, and
    ``manifest.json`` is the model's two-space JSON.
    """
    expected = {
        "paper": (
            "% GENERATED by torchcell.literature.bib_store -- do not hand-edit.\n"
            "% name: paper  entries: 1\n"
            "% scope: group 6582362/W46ATS7B\n"
            "% declared in: paper/nature-biotech/zotero_export_bib.py\n"
            "% served by tc-lit at /bib/paper; verify X-Artifact-SHA256.\n\n" + _BODY_S
        ),
        "eqtl-data-model": (
            "% GENERATED by torchcell.literature.bib_store -- do not hand-edit.\n"
            "% name: eqtl-data-model  entries: 1\n"
            "% scope: group 6582362/VNDH4NMX + personal 1234/4VNJWJAW\n"
            "% declared in: notes-tex/eqtl-data-model/Makefile\n"
            "% served by tc-lit at /bib/eqtl-data-model; verify X-Artifact-SHA256.\n\n"
            + _BODY_P
        ),
        "library": (
            "% GENERATED by torchcell.literature.bib_store -- do not hand-edit.\n"
            "% name: library  entries: 1\n"
            "% scope: group 6582362/* + personal 1234/torchcell/** (tree)\n"
            "% declared in: scripts/lit_bib.py\n"
            "% served by tc-lit at /bib/library; verify X-Artifact-SHA256.\n\n"
            + _BODY_U
        ),
    }
    specs = _three_specs()
    manifest = export_bib_store(
        tmp_path, specs, GROUP, USER, declared=specs, generated_at=T0
    )
    store = bib_store_dir(tmp_path)
    assert sorted(p.name for p in store.iterdir()) == [
        "eqtl-data-model.bib",
        "library.bib",
        "manifest.json",
        "paper.bib",
    ]
    for name, text in expected.items():
        assert (store / f"{name}.bib").read_text() == text
    assert manifest.model_dump() == {
        "version": 1,
        "generated_at": T0,
        "bibs": [
            {
                "name": spec.name,
                "path": f"{spec.name}.bib",
                "bytes": len(expected[spec.name].encode()),
                "sha256": hashlib.sha256(expected[spec.name].encode()).hexdigest(),
                "n_entries": 1,
                "scope": spec.scope.model_dump(),
                "origin": spec.origin,
                "generated_at": T0,
            }
            for spec in specs
        ],
    }
    assert json.loads((store / BIB_STORE_MANIFEST).read_text()) == json.loads(
        manifest.model_dump_json()
    )
    assert (store / BIB_STORE_MANIFEST).read_text() == manifest.model_dump_json(
        indent=2
    )


def test_body_is_sorted_case_insensitively_by_key(
    tmp_path: Path, canned_pull: Canned
) -> None:
    """``b2020`` then ``A2021`` from the pull are written ``A2021`` first (keys are
    compared lower-cased), and the banner counts both.
    """
    canned_pull["W46ATS7B"] = [
        _entry("b2020", "Second"),
        {"ID": "A2021", "ENTRYTYPE": "book", "title": "First", "year": "2021"},
    ]
    spec = _spec("paper", group_collection="W46ATS7B")
    export_bib_store(tmp_path, [spec], NO_LIB, NO_LIB, declared=[spec], generated_at=T0)
    text = (bib_store_dir(tmp_path) / "paper.bib").read_text()
    assert text.split("\n\n", 1)[1] == (
        "@book{A2021,\n  title = {First},\n  year = {2021}\n}\n\n"
        "@article{b2020,\n  author = {Zotero, Zed},\n  title = {Second},\n"
        "  year = {2020}\n}\n"
    )
    assert "% name: paper  entries: 2\n" in text


def test_empty_pull_message_and_no_part_file_left_by_a_failed_export(
    tmp_path: Path, canned_pull: Canned
) -> None:
    """The refusal names the file and the scope without its None fields, and the
    export removes what it staged before re-raising: ``paper.bib.part``, staged for
    the spec before the failing one, is gone and ``_bib/`` is empty.
    """
    canned_pull["W46ATS7B"] = [_entry("a2020", "First")]
    canned_pull["VNDH4NMX"] = []
    specs = [
        _spec("paper", group_collection="W46ATS7B"),
        _spec("eqtl-data-model", group_collection="VNDH4NMX"),
    ]
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "refusing to write eqtl-data-model.bib: Zotero returned 0 entries for "
            "{'group_library_id': '6582362', 'group_collection': 'VNDH4NMX'}"
        ),
    ):
        export_bib_store(
            tmp_path, specs, NO_LIB, NO_LIB, declared=specs, generated_at=T0
        )
    assert sorted(p.name for p in bib_store_dir(tmp_path).iterdir()) == []


def test_illegal_spec_name_is_refused_before_any_pull(
    tmp_path: Path, fetch_calls: list[tuple[Any, ...]]
) -> None:
    """``BibSpec`` accepts any name; ``export_bib_store`` validates each name before
    touching the store, so ``../escape`` raises with the exact message, no fetcher
    runs, and the store directory is never created.
    """
    spec = BibSpec(
        name="../escape",
        scope=BibScope(group_library_id="6582362", group_collection="W46ATS7B"),
        origin="test",
    )
    with pytest.raises(
        ValueError, match=re.escape("illegal bibliography name: '../escape'")
    ):
        export_bib_store(
            tmp_path, [spec], GROUP, USER, declared=[spec], generated_at=T0
        )
    assert fetch_calls == []
    assert not bib_store_dir(tmp_path).exists()


def test_load_bib_store_without_an_export_raises_file_not_found(tmp_path: Path) -> None:
    manifest = tmp_path / "_bib" / "manifest.json"
    with pytest.raises(
        FileNotFoundError,
        match=re.escape(f"[Errno 2] No such file or directory: '{manifest}'"),
    ):
        load_bib_store(tmp_path)


def _client(mirror_root: Path) -> TestClient:
    config = LiteratureServerConfig(
        mirror_root=mirror_root, keys=LiteratureKeys.from_pairs(f"t:{KEY}"), port=8899
    )
    return TestClient(create_app(config))


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def _three_served(tmp_path: Path, canned_pull: Canned) -> list[BibSpec]:
    """A store serving ``paper``, ``eqtl-data-model`` and ``pilot`` from one full run."""
    canned_pull["W46ATS7B"] = [_entry("a2020", "First")]
    canned_pull["VNDH4NMX"] = [_entry("q2020", "Doc")]
    canned_pull["FE8DQKUH"] = [_entry("p2021", "Pilot")]
    specs = [
        _spec("paper", group_collection="W46ATS7B"),
        _spec("eqtl-data-model", group_collection="VNDH4NMX"),
        _spec("pilot", group_collection="FE8DQKUH"),
    ]
    export_bib_store(tmp_path, specs, NO_LIB, NO_LIB, declared=specs, generated_at=T0)
    return specs


def test_spec_removed_from_the_repo_is_unserved_and_moved_aside(
    tmp_path: Path, canned_pull: Canned, caplog: pytest.LogCaptureFixture
) -> None:
    """Replaces the 2026.09.30 test that pinned a 404 after a run naming fewer specs:
    that 404 is now only for a spec the REPO no longer declares. ``eqtl-data-model`` is
    dropped from the declaration and a full run exports the other two: the manifest
    lists ``paper`` and ``pilot``, ``/bib/eqtl-data-model`` answers 404 ``unknown
    bibliography``, and its file (plus a stray ``old-doc.bib.part``) is moved, with its
    bytes, to ``_bib/_retired/<T1>/`` with one exact warning each; nothing is deleted.
    """
    paper, doc, pilot = _three_served(tmp_path, canned_pull)
    store = bib_store_dir(tmp_path)
    dropped_text = (store / "eqtl-data-model.bib").read_text()
    (store / "old-doc.bib.part").write_text("stale")
    caplog.set_level(logging.WARNING, logger=store_mod.__name__)
    kept = [paper, pilot]
    export_bib_store(tmp_path, kept, NO_LIB, NO_LIB, declared=kept, generated_at=T1)
    assert [b.name for b in load_bib_store(tmp_path).bibs] == ["paper", "pilot"]
    assert sorted(p.name for p in store.iterdir()) == [
        "_retired",
        "manifest.json",
        "paper.bib",
        "pilot.bib",
    ]
    retired = store / "_retired" / T1
    assert sorted(p.name for p in retired.iterdir()) == [
        "eqtl-data-model.bib",
        "old-doc.bib.part",
    ]
    assert (retired / "eqtl-data-model.bib").read_text() == dropped_text
    assert (retired / "old-doc.bib.part").read_text() == "stale"
    assert _warnings(caplog) == [
        f"bib_store: eqtl-data-model is not declared by any spec in the repo; moved "
        f"{store / 'eqtl-data-model.bib'} -> {retired / 'eqtl-data-model.bib'}",
        f"bib_store: old-doc is a leftover staging file; moved "
        f"{store / 'old-doc.bib.part'} -> {retired / 'old-doc.bib.part'}",
    ]
    response = _client(tmp_path).get("/bib/eqtl-data-model", headers=HEADERS)
    assert (response.status_code, response.json()) == (
        404,
        {"detail": "unknown bibliography"},
    )


def test_subset_run_carries_the_other_declared_bibliographies_forward(
    tmp_path: Path, canned_pull: Canned, caplog: pytest.LogCaptureFixture
) -> None:
    """Three served bibliographies, then a ``--name paper`` run (``paper`` re-exported
    with a new entry, all three still declared): the manifest lists all three in
    declared order; ``paper`` has the new stamp and hash, the other two records are
    byte-identical to the previous manifest's and their files are untouched; GET
    serves each with its pinned hash; nothing is retired and nothing is warned.
    """
    specs = _three_served(tmp_path, canned_pull)
    store = bib_store_dir(tmp_path)
    before = load_bib_store(tmp_path)
    files_before = {
        name: (store / f"{name}.bib").read_bytes()
        for name in ("eqtl-data-model", "pilot")
    }
    canned_pull["W46ATS7B"] = [_entry("a2020", "First"), _entry("z2021", "Added")]
    caplog.set_level(logging.WARNING, logger=store_mod.__name__)
    after = export_bib_store(
        tmp_path, specs[:1], NO_LIB, NO_LIB, declared=specs, generated_at=T1
    )
    assert load_bib_store(tmp_path) == after
    assert [b.name for b in after.bibs] == ["paper", "eqtl-data-model", "pilot"]
    assert after.bibs[1:] == before.bibs[1:]
    assert [b.model_dump_json() for b in after.bibs[1:]] == [
        b.model_dump_json() for b in before.bibs[1:]
    ]
    paper = after.bibs[0]
    assert (paper.generated_at, paper.n_entries) == (T1, 2)
    assert paper.sha256 != before.bibs[0].sha256
    for name, data in files_before.items():
        assert (store / f"{name}.bib").read_bytes() == data
    assert sorted(p.name for p in store.iterdir()) == [
        "eqtl-data-model.bib",
        "manifest.json",
        "paper.bib",
        "pilot.bib",
    ]
    assert _warnings(caplog) == []
    client = _client(tmp_path)
    for record in after.bibs:
        response = client.get(f"/bib/{record.name}", headers=HEADERS)
        assert response.status_code == 200
        assert response.headers["X-Artifact-SHA256"] == record.sha256
        assert hashlib.sha256(response.content).hexdigest() == record.sha256


def test_subset_run_still_retires_a_spec_removed_from_the_repo(
    tmp_path: Path, canned_pull: Canned, caplog: pytest.LogCaptureFixture
) -> None:
    """``pilot`` is dropped from the declaration and a ``--name paper`` run follows:
    ``eqtl-data-model`` is carried forward, ``pilot`` is unlisted and its file moves
    to ``_retired/<T1>/`` with the exact warning.
    """
    paper, doc, _pilot = _three_served(tmp_path, canned_pull)
    store = bib_store_dir(tmp_path)
    caplog.set_level(logging.WARNING, logger=store_mod.__name__)
    export_bib_store(
        tmp_path, [paper], NO_LIB, NO_LIB, declared=[paper, doc], generated_at=T1
    )
    assert [b.name for b in load_bib_store(tmp_path).bibs] == [
        "paper",
        "eqtl-data-model",
    ]
    retired = store / "_retired" / T1
    assert sorted(p.name for p in retired.iterdir()) == ["pilot.bib"]
    assert _warnings(caplog) == [
        f"bib_store: pilot is not declared by any spec in the repo; moved "
        f"{store / 'pilot.bib'} -> {retired / 'pilot.bib'}"
    ]


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        ("missing", "the previous manifest lists pilot.bib but it is absent on disk"),
        ("edited", "pilot.bib no longer has the sha256 the previous manifest pins"),
    ],
)
def test_subset_run_refuses_to_carry_a_broken_record_by_name(
    tmp_path: Path, canned_pull: Canned, damage: str, reason: str
) -> None:
    """A previous record whose file is gone, or whose bytes no longer hash to the
    pinned sha256, is never carried forward silently: the subset run raises naming
    the bibliography before any pull, and the manifest and ``paper.bib`` are as the
    full run left them.
    """
    specs = _three_served(tmp_path, canned_pull)
    store = bib_store_dir(tmp_path)
    before = load_bib_store(tmp_path)
    paper_before = (store / "paper.bib").read_bytes()
    if damage == "missing":
        (store / "pilot.bib").replace(tmp_path / "pilot.bib.moved")
    else:
        (store / "pilot.bib").write_text("edited by hand")
    canned_pull["W46ATS7B"] = []  # a pull would raise a different error
    with pytest.raises(ValueError) as excinfo:
        export_bib_store(
            tmp_path, specs[:1], NO_LIB, NO_LIB, declared=specs, generated_at=T1
        )
    assert str(excinfo.value) == (
        f"cannot carry pilot forward: {reason}; run a full export"
    )
    assert load_bib_store(tmp_path) == before
    assert (store / "paper.bib").read_bytes() == paper_before


def test_exported_spec_must_be_declared(tmp_path: Path, canned_pull: Canned) -> None:
    """Exporting a spec the declaration does not carry is refused before the store
    directory is created.
    """
    paper = _spec("paper", group_collection="W46ATS7B")
    extra = _spec("extra", group_collection="VNDH4NMX")
    with pytest.raises(ValueError) as excinfo:
        export_bib_store(
            tmp_path, [paper, extra], NO_LIB, NO_LIB, declared=[paper], generated_at=T0
        )
    assert str(excinfo.value) == "exported specs not declared by the repo: ['extra']"
    assert not bib_store_dir(tmp_path).exists()


@pytest.mark.parametrize(
    "stamp", ["../../escaped", "T1", "2026-01-02T03:04:05", "2026-01-02 03:04:05+00:00"]
)
def test_generated_at_outside_the_exporter_format_is_refused(
    tmp_path: Path, canned_pull: Canned, stamp: str
) -> None:
    """``generated_at`` names the ``_retired/`` directory, so only the exporter's own
    format (``YYYY-MM-DDTHH:MM:SS[.ffffff]+00:00``) is accepted. A path-like or other
    stamp is refused by value before anything is pulled, written or moved: the
    served store and an undeclared ``stray.bib`` stay exactly where they were and no
    file appears outside ``_bib/``.
    """
    specs = _three_served(tmp_path, canned_pull)
    store = bib_store_dir(tmp_path)
    (store / "stray.bib").write_text("stray")
    listing = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    before = load_bib_store(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        export_bib_store(
            tmp_path, specs, NO_LIB, NO_LIB, declared=specs, generated_at=stamp
        )
    assert str(excinfo.value) == (
        "generated_at must be a UTC ISO timestamp "
        f"(YYYY-MM-DDTHH:MM:SS[.ffffff]+00:00), got {stamp!r}"
    )
    assert (
        sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
        == listing
    )
    assert load_bib_store(tmp_path) == before


def test_validate_generated_at_accepts_the_exporter_default() -> None:
    """The stamp ``export_bib_store`` makes when none is given passes its own check."""
    stamp = datetime.now(UTC).isoformat()
    assert validate_generated_at(stamp) == stamp


def _makefile(root: Path, slug: str, text: str) -> None:
    directory = root / "notes-tex" / slug
    directory.mkdir(parents=True)
    (directory / "Makefile").write_text(text)


def test_discover_reads_every_assignment_form(tmp_path: Path) -> None:
    """``a-plain`` uses ``=`` with no personal collection (so no personal library on its
    scope); ``b-cond`` uses ``?=`` and sets ``ZOTERO_COLLECTION`` twice (the last wins);
    ``c-colon`` uses ``:=``. Documents are visited in sorted directory order between
    ``paper`` and ``library``, and ``user_root_collection`` passes through.
    """
    _makefile(tmp_path, "a-plain", "ZOTERO_COLLECTION = AAAA1111\n")
    _makefile(
        tmp_path,
        "b-cond",
        "ZOTERO_COLLECTION ?= OLD00000\n"
        "ZOTERO_PERSONAL_COLLECTION ?= PPPP2222\n"
        "ZOTERO_COLLECTION ?= BBBB2222\n",
    )
    _makefile(tmp_path, "c-colon", "  ZOTERO_COLLECTION:=CCCC3333  \n")
    specs = discover_bib_specs(
        tmp_path,
        group_library_id="6582362",
        user_library_id="1234",
        user_root_collection="lab",
    )
    assert [s.model_dump() for s in specs] == [
        {
            "name": "paper",
            "scope": BibScope(
                group_library_id="6582362", group_collection="W46ATS7B"
            ).model_dump(),
            "origin": "paper/nature-biotech/zotero_export_bib.py",
        },
        {
            "name": "a-plain",
            "scope": BibScope(
                group_library_id="6582362", group_collection="AAAA1111"
            ).model_dump(),
            "origin": "notes-tex/a-plain/Makefile",
        },
        {
            "name": "b-cond",
            "scope": BibScope(
                group_library_id="6582362",
                group_collection="BBBB2222",
                user_library_id="1234",
                user_collection="PPPP2222",
            ).model_dump(),
            "origin": "notes-tex/b-cond/Makefile",
        },
        {
            "name": "c-colon",
            "scope": BibScope(
                group_library_id="6582362", group_collection="CCCC3333"
            ).model_dump(),
            "origin": "notes-tex/c-colon/Makefile",
        },
        {
            "name": "library",
            "scope": BibScope(
                group_library_id="6582362",
                user_library_id="1234",
                user_root_collection="lab",
            ).model_dump(),
            "origin": "scripts/lit_bib.py",
        },
    ]


def test_inline_comment_after_the_value_is_cut_off(tmp_path: Path) -> None:
    """``ZOTERO_COLLECTION := VNDH4NMX  # eQTL`` reads as ``VNDH4NMX`` (make drops the
    comment), and so does the personal line with ``#`` right after the value; the
    document is discovered with both keys.
    """
    _makefile(
        tmp_path,
        "eqtl",
        "ZOTERO_COLLECTION := VNDH4NMX  # eQTL\n"
        "ZOTERO_PERSONAL_COLLECTION := 4VNJWJAW# personal\n",
    )
    makefile = tmp_path / "notes-tex" / "eqtl" / "Makefile"
    assert parse_makefile_collections(makefile) == ("VNDH4NMX", "4VNJWJAW")
    specs = discover_bib_specs(
        tmp_path, group_library_id="6582362", user_library_id="1"
    )
    assert [s.name for s in specs] == ["paper", "eqtl", "library"]
    assert specs[1].scope == BibScope(
        group_library_id="6582362",
        group_collection="VNDH4NMX",
        user_library_id="1",
        user_collection="4VNJWJAW",
    )


@pytest.mark.parametrize(
    ("line", "variable", "value"),
    [
        (
            "ZOTERO_COLLECTION := VNDH4NMX extra  # two\n",
            "ZOTERO_COLLECTION",
            "VNDH4NMX extra",
        ),
        (
            "ZOTERO_COLLECTION := VNDH4NMX\nZOTERO_PERSONAL_COLLECTION := microbe-perturb-seq\n",
            "ZOTERO_PERSONAL_COLLECTION",
            "microbe-perturb-seq",
        ),
    ],
)
def test_a_makefile_value_that_is_not_one_key_is_refused_naming_the_makefile(
    tmp_path: Path, line: str, variable: str, value: str
) -> None:
    """A two-word value, or a collection NAME where a key is declared, is refused with
    the Makefile path, the variable and the value, rather than read as empty, as its
    first word, or as a scope error that does not say which file to fix.
    """
    _makefile(tmp_path, "eqtl", line)
    makefile = tmp_path / "notes-tex" / "eqtl" / "Makefile"
    with pytest.raises(ValueError) as excinfo:
        discover_bib_specs(tmp_path, group_library_id="6582362", user_library_id="1")
    assert str(excinfo.value) == (
        f"{makefile}: {variable} must be one Zotero collection key "
        f"(8 upper-case letters or digits), got {value!r}"
    )


def test_a_citing_dot_directory_is_refused_by_name(tmp_path: Path) -> None:
    """``notes-tex/.draft/Makefile`` declares a collection; ``glob("*/Makefile")``
    visits dot directories, and the slug fails the name rule with the exact message.
    """
    _makefile(tmp_path, ".draft", "ZOTERO_COLLECTION := AAAA1111\n")
    with pytest.raises(
        ValueError, match=re.escape("illegal bibliography name: '.draft'")
    ):
        discover_bib_specs(tmp_path, group_library_id="6582362", user_library_id="1")


# --- delta review of PR #589: damaged manifest, atomic write, stamp, retire -- #


def test_full_export_over_a_truncated_manifest_succeeds(
    tmp_path: Path, canned_pull: Canned
) -> None:
    """A full export re-exports every declared spec, carries nothing, and never reads
    the previous manifest: over a manifest truncated mid-JSON it succeeds and writes
    a manifest that validates and lists all three bibliographies, leaving no
    ``manifest.json.tmp`` behind.
    """
    specs = _three_served(tmp_path, canned_pull)
    store = bib_store_dir(tmp_path)
    manifest_path = store / BIB_STORE_MANIFEST
    manifest_path.write_text(manifest_path.read_text()[:40])
    after = export_bib_store(
        tmp_path, specs, NO_LIB, NO_LIB, declared=specs, generated_at=T1
    )
    assert load_bib_store(tmp_path) == after
    assert [b.name for b in after.bibs] == ["paper", "eqtl-data-model", "pilot"]
    assert [b.generated_at for b in after.bibs] == [T1, T1, T1]
    assert sorted(p.name for p in store.iterdir()) == [
        "eqtl-data-model.bib",
        "manifest.json",
        "paper.bib",
        "pilot.bib",
    ]


def test_subset_export_over_a_truncated_manifest_is_refused(
    tmp_path: Path, canned_pull: Canned
) -> None:
    """A subset run must read the previous manifest to carry records forward; over a
    truncated one it refuses with the exact message naming the manifest and the
    remedy, before any pull, and the damaged bytes stay as they were.
    """
    specs = _three_served(tmp_path, canned_pull)
    manifest_path = bib_store_dir(tmp_path) / BIB_STORE_MANIFEST
    damaged = manifest_path.read_text()[:40]
    manifest_path.write_text(damaged)
    canned_pull["W46ATS7B"] = []  # a pull would raise a different error
    with pytest.raises(ValueError) as excinfo:
        export_bib_store(
            tmp_path, specs[:1], NO_LIB, NO_LIB, declared=specs, generated_at=T1
        )
    assert str(excinfo.value) == (
        f"cannot carry bibliographies forward: the previous manifest "
        f"{manifest_path} does not validate; run a full export"
    )
    assert manifest_path.read_text() == damaged


def test_failed_manifest_write_leaves_the_previous_manifest_byte_identical(
    tmp_path: Path, canned_pull: Canned, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest is written to ``manifest.json.tmp`` and swapped in with
    ``os.replace``; an ``OSError`` injected into that temporary write propagates and
    the served ``manifest.json`` keeps its previous bytes.
    """
    specs = _three_served(tmp_path, canned_pull)
    manifest_path = bib_store_dir(tmp_path) / BIB_STORE_MANIFEST
    before = manifest_path.read_bytes()
    real_write_text = Path.write_text

    def failing_write_text(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        if self.name == "manifest.json.tmp":
            raise OSError("disk full")
        return real_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", failing_write_text)
    with pytest.raises(OSError, match="^disk full$"):
        export_bib_store(
            tmp_path, specs, NO_LIB, NO_LIB, declared=specs, generated_at=T1
        )
    assert manifest_path.read_bytes() == before


def test_an_impossible_date_in_the_right_shape_is_refused(
    tmp_path: Path, canned_pull: Canned
) -> None:
    """``2026-13-99T99:99:99+00:00`` has the exporter's shape but is not a date;
    ``datetime.fromisoformat`` rejects it, so it is refused with the exact message.
    """
    specs = _three_served(tmp_path, canned_pull)
    stamp = "2026-13-99T99:99:99+00:00"
    with pytest.raises(ValueError) as excinfo:
        export_bib_store(
            tmp_path, specs, NO_LIB, NO_LIB, declared=specs, generated_at=stamp
        )
    assert str(excinfo.value) == (
        "generated_at must be a UTC ISO timestamp "
        f"(YYYY-MM-DDTHH:MM:SS[.ffffff]+00:00), got {stamp!r}"
    )


def test_retiring_twice_under_one_stamp_never_overwrites(
    tmp_path: Path, canned_pull: Canned
) -> None:
    """Two runs with the same stamp each retire a ``stray.bib`` with different
    bytes: the first keeps the plain name, the second gets ``stray.bib.1``, and both
    contents survive.
    """
    specs = _three_served(tmp_path, canned_pull)
    store = bib_store_dir(tmp_path)
    retired = store / "_retired" / T1
    for content in ("first", "second"):
        (store / "stray.bib").write_text(content)
        export_bib_store(
            tmp_path, specs, NO_LIB, NO_LIB, declared=specs, generated_at=T1
        )
    assert sorted(p.name for p in retired.iterdir()) == ["stray.bib", "stray.bib.1"]
    assert (retired / "stray.bib").read_text() == "first"
    assert (retired / "stray.bib.1").read_text() == "second"
