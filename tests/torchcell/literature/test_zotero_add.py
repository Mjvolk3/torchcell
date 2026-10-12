# tests/torchcell/literature/test_zotero_add.py
# [[tests.torchcell.literature.test_zotero_add]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_zotero_add.py
"""The database-collection filing path, over a scripted server and a recording Zotero.

Nothing leaves the process. HTTP goes to an ``httpx.MockTransport`` that answers only
the URLs a test scripts and records every request; an unscripted URL raises
``KeyError`` out of the handler, so a route the test did not expect fails the test.
Zotero is :class:`WritableZot`, the read-only ``FakeZot`` plus write methods that only
record their payloads, and uploads go through :func:`recording_upload`, which adds the
attachment (with the MD5 of the bytes it was handed) to the fake's children. The
refusal tests assert that no write method was called.

Expected values are spelled out: sha256 and MD5 are computed with ``hashlib`` on the
scripted bytes (the independent oracle), URLs are written out literally from the
publishers' patterns, and the Crossref record is the fixture :data:`WORK`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr, ValidationError

import torchcell.literature.zotero_add as za
from tests.torchcell.literature._fake_zotero import FakeZot, collection
from torchcell.literature.manifest import (
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
    write_manifest,
)
from torchcell.literature.zotero import ZoteroConfig, ZoteroLibrary

DOI = "10.1234/example.5678"
PDF = b"%PDF-1.7\n1 0 obj\n<< >>\nendobj\ntrailer\n%%EOF\n"
OTHER_PDF = b"%PDF-1.4\nsupplement\n%%EOF\n"
HTML = b"<!DOCTYPE html><html><body>Please wait</body></html>"
TODAY = "2026-10-12"
CROSSREF_URL = "https://api.crossref.org/works/10.1234/example.5678"
IDCONV_URL = (
    "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
    "?ids=10.1234%2Fexample.5678&format=json&tool=torchcell"
)
LIST_URL = "https://pmc-oa-opendata.s3.amazonaws.com/?list-type=2&prefix=PMC42."
PMC_PDF_URL = "https://pmc-oa-opendata.s3.amazonaws.com/PMC42.2/PMC42.2.pdf"
ELSEVIER_PDF_URL = (
    "https://www.sciencedirect.com/science/article/pii/S0092867410013747/pdfft"
    "?isDTMRedir=true&download=true"
)
CROSSREF_LINK = "https://publisher.example/article/example.5678.pdf"

WORK: dict[str, Any] = {
    "DOI": DOI,
    "member": "78",
    "title": ["Phenotypic\n   landscape of a <i>bacterial</i> cell"],
    "author": [
        {"given": "Robert", "family": "Nichols", "sequence": "first"},
        {"given": "Carol", "family": "Gross", "sequence": "additional"},
        {"name": "The Consortium", "sequence": "additional"},
    ],
    "container-title": ["Cell"],
    "volume": "144",
    "issue": "1",
    "page": "143-156",
    "ISSN": ["0092-8674", "1097-4172"],
    "issued": {"date-parts": [[2011, 1, 7]]},
    "alternative-id": ["S0092867410013747"],
    "link": [
        {"URL": CROSSREF_LINK, "content-type": "application/pdf"},
        {"URL": "https://publisher.example/x.xml", "content-type": "text/xml"},
    ],
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _md5(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def _listing(keys: list[str]) -> bytes:
    contents = "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys)
    return (
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"{contents}<IsTruncated>false</IsTruncated></ListBucketResult>"
    ).encode()


class Server:
    """URL -> (status, body, content-type); records every requested URL."""

    def __init__(self, routes: dict[str, tuple[int, bytes, str]]) -> None:
        """Serve ``routes``; nothing else."""
        self.routes = routes
        self.seen: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Answer a scripted URL; an unscripted one raises ``KeyError``."""
        url = str(request.url)
        self.seen.append(url)
        status, body, kind = self.routes[url]
        return httpx.Response(status, content=body, headers={"content-type": kind})

    def client(self) -> httpx.Client:
        """A real client on this server's transport."""
        return httpx.Client(
            transport=httpx.MockTransport(self.handler), follow_redirects=True
        )


def _crossref(work: dict[str, Any] = WORK) -> tuple[int, bytes, str]:
    return 200, json.dumps({"message": work}).encode(), "application/json"


def _blocked_publisher() -> dict[str, tuple[int, bytes, str]]:
    return {
        CROSSREF_URL: _crossref(),
        CROSSREF_LINK: (403, HTML, "text/html"),
        ELSEVIER_PDF_URL: (403, HTML, "text/html"),
    }


def _pmc_with_pdf() -> dict[str, tuple[int, bytes, str]]:
    return {
        IDCONV_URL: (
            200,
            json.dumps({"records": [{"pmcid": "PMC42"}]}).encode(),
            "application/json",
        ),
        LIST_URL: (
            200,
            _listing(
                ["PMC42.1/PMC42.1.xml", "PMC42.2/PMC42.2.xml", "PMC42.2/PMC42.2.pdf"]
            ),
            "application/xml",
        ),
        PMC_PDF_URL: (200, PDF, "application/pdf"),
    }


class WritableZot(FakeZot):
    """``FakeZot`` plus write methods that only record what they are sent."""

    def __init__(self, **kwargs: Any) -> None:
        """``FakeZot`` fixtures plus empty write logs."""
        super().__init__(**kwargs)
        self.created: list[list[dict[str, Any]]] = []
        self.deleted: list[str] = []
        self.uploads: list[tuple[str, dict[str, Any], bytes]] = []

    def item_template(self, itemtype: str) -> dict[str, Any]:
        """A minimal template; ``add_paper`` fills it from Crossref."""
        return {"itemType": itemtype, "title": "", "creators": [], "extra": ""}

    def create_items(self, payload: list[dict[str, Any]]) -> dict[str, Any]:
        """Record the payload and answer the API's success shape as NEWITEM1."""
        self.created.append(payload)
        self._items.append({"key": "NEWITEM1", "data": dict(payload[0])})
        return {"successful": {"0": {"key": "NEWITEM1"}}, "failed": {}}

    def item(self, key: str) -> dict[str, Any]:
        """The stored item with this key."""
        return next(i for i in self._items if i["key"] == key)

    def delete_item(self, item: dict[str, Any]) -> None:
        """Record the deleted key."""
        self.deleted.append(item["key"])

    def writes(self) -> tuple[int, int, int]:
        """(items created, items deleted, files uploaded)."""
        return len(self.created), len(self.deleted), len(self.uploads)


def recording_upload(
    zot: Any, templates: list[dict[str, Any]], parent: str, basedir: str
) -> dict[str, list[Any]]:
    """Add the attachment to the fake's children, keyed ATT<n>, with its file MD5."""
    template = dict(templates[0])
    content = (Path(basedir) / template["filename"]).read_bytes()
    key = f"ATT{len(zot.uploads) + 1}"
    zot.uploads.append((parent, template, content))
    zot._children.setdefault(parent, []).append(
        {"key": key, "data": {**template, "md5": _md5(content)}}
    )
    return {"success": [{**template, "key": key}], "failure": [], "unchanged": []}


def failing_upload(
    zot: Any, templates: list[dict[str, Any]], parent: str, basedir: str
) -> dict[str, list[Any]]:
    """The upload answers with the attachment in ``failure``."""
    zot.uploads.append((parent, dict(templates[0]), b""))
    return {"success": [], "failure": [dict(templates[0])], "unchanged": []}


COLLECTIONS = [
    collection("DBROOT", "database"),
    collection("ECOLI", "Escherichia-coli", "DBROOT"),
    collection("PPUT", "Pseudomonas-putida", "DBROOT"),
    collection("PAPER", "paper"),
]


def _lib(
    monkeypatch: pytest.MonkeyPatch,
    zot: FakeZot,
    library_id: str = za.GROUP_LIBRARY_ID,
    library_type: str = "group",
) -> ZoteroLibrary:
    monkeypatch.setattr("torchcell.literature.zotero.zotero.Zotero", lambda *a: zot)
    return ZoteroLibrary(
        ZoteroConfig(
            library_id=library_id,
            library_type="user" if library_type == "user" else "group",
            api_key=SecretStr("k"),
        )
    )


def _item(key: str, collections: list[str], doi: str = DOI) -> dict[str, Any]:
    return {"key": key, "data": {"DOI": doi, "collections": collections}}


def _request(tmp_path: Path, **kwargs: Any) -> za.AddPaperRequest:
    fields: dict[str, Any] = {
        "doi": DOI,
        "collection": "database/Escherichia-coli",
        "first_author": "Nichols",
        "raw_root": tmp_path / "raw",
    }
    fields.update(kwargs)
    (tmp_path / "raw").mkdir(exist_ok=True)
    return za.AddPaperRequest(**fields)


def _raw_key(
    raw_root: Path, key: str, doi: str, content: bytes, recorded_sha: str | None = None
) -> Path:
    """A raw-mirror key recording ``paper/PMC7.1.pdf`` as its paper_pdf."""
    key_dir = raw_root / key
    (key_dir / "paper").mkdir(parents=True)
    (key_dir / "paper" / "PMC7.1.pdf").write_bytes(content)
    sha = recorded_sha or _sha(content)
    record = ArtifactRecord(
        path="paper/PMC7.1.pdf",
        role="paper_pdf",
        bytes=len(content),
        sha256=sha,
        retrieval=RetrievalRecord(
            method=RetrievalMethod.pmc_cloud,
            source_url="https://pmc-oa-opendata.s3.amazonaws.com/PMC7.1/PMC7.1.pdf",
            retriever=za.R_PMC_CLOUD,
            params={"key": "PMC7.1/PMC7.1.pdf"},
            sha256=sha,
            retrieved_at="2026-10-08",
        ),
    )
    write_manifest(key_dir, Manifest(citation_key=key, doi=doi, files=[record]))
    return key_dir


# -- checks ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (PDF, True),
        (b"\n" * 1000 + PDF, True),
        (b" " * 1024 + PDF, False),
        (HTML, False),
        (b"%PDF-1.7\n1 0 obj truncated", False),
    ],
)
def test_is_pdf_bytes(content: bytes, expected: bool) -> None:
    """Header within the first KiB and a trailer near the end, or not a PDF."""
    assert za.is_pdf_bytes(content) is expected


def test_check_collection_path_accepts_a_database_child() -> None:
    assert za.check_collection_path("database/Escherichia-coli") == (
        "database/Escherichia-coli"
    )


@pytest.mark.parametrize(
    "path",
    [
        "paper",
        "database",
        "database/",
        "database/Escherichia-coli/sub",
        "torchcell/database/Escherichia-coli",
        "Database/Escherichia-coli",
        "microbe-perturb-seq",
    ],
)
def test_check_collection_path_refuses_everything_else(path: str) -> None:
    with pytest.raises(za.RefusedError, match="not a database/<Organism> collection"):
        za.check_collection_path(path)


@pytest.mark.parametrize(
    ("library_id", "library_type"), [("6582362", "user"), ("9999999", "group")]
)
def test_check_group_library_refuses_other_libraries(
    monkeypatch: pytest.MonkeyPatch, library_id: str, library_type: str
) -> None:
    lib = _lib(monkeypatch, FakeZot(), library_id, library_type)
    with pytest.raises(za.RefusedError, match="not the TorchCell group library"):
        za.check_group_library(lib)


def test_resolve_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    """A database child resolves by full path; a missing one lists what exists."""
    lib = _lib(monkeypatch, FakeZot(collections=COLLECTIONS))
    assert za.resolve_collection(lib, "database/Pseudomonas-putida") == (
        "PPUT",
        "DBROOT",
    )
    with pytest.raises(za.RefusedError) as exc:
        za.resolve_collection(lib, "database/Bacillus-subtilis")
    assert "['database/Escherichia-coli', 'database/Pseudomonas-putida']" in str(
        exc.value
    )


def test_paper_identity_and_zotero_fields() -> None:
    """Whitespace collapsed, markup kept in the stored title, every author carried."""
    identity = za.paper_identity(WORK)
    assert identity == za.PaperIdentity(
        doi=DOI,
        title="Phenotypic landscape of a <i>bacterial</i> cell",
        first_author="Nichols",
        date="2011-01-07",
        container="Cell",
        member="78",
    )
    assert za.journal_article_fields(WORK, identity) == {
        "title": "Phenotypic landscape of a <i>bacterial</i> cell",
        "creators": [
            {"creatorType": "author", "firstName": "Robert", "lastName": "Nichols"},
            {"creatorType": "author", "firstName": "Carol", "lastName": "Gross"},
            {"creatorType": "author", "name": "The Consortium"},
        ],
        "publicationTitle": "Cell",
        "volume": "144",
        "issue": "1",
        "pages": "143-156",
        "date": "2011-01-07",
        "DOI": DOI,
        "ISSN": "0092-8674, 1097-4172",
        "url": f"https://doi.org/{DOI}",
        "extra": "added-by: scripts/zotero_add_paper.py (CrossRef)",
    }


def test_paper_identity_takes_the_sequence_first_author() -> None:
    work = {
        **WORK,
        "author": [
            {"family": "Second", "sequence": "additional"},
            {"family": "First", "sequence": "first"},
        ],
        "issued": {"date-parts": [[2019]]},
    }
    identity = za.paper_identity(work)
    assert (identity.first_author, identity.date) == ("First", "2019")


@pytest.mark.parametrize(
    ("work", "message"),
    [
        ({**WORK, "author": []}, "lists no author"),
        ({**WORK, "title": []}, "gives no title"),
        ({**WORK, "issued": {"date-parts": [[None]]}}, "no issued date"),
    ],
)
def test_paper_identity_refuses_incomplete_records(
    work: dict[str, Any], message: str
) -> None:
    with pytest.raises(za.RefusedError, match=message):
        za.paper_identity(work)


def test_verify_identity() -> None:
    """Accent- and case-insensitive author, title substring on the markup-free title."""
    identity = za.paper_identity({**WORK, "author": [{"family": "Novaková"}]})
    za.verify_identity(identity, DOI.upper(), "NOVAKOVA", "bacterial cell")
    with pytest.raises(za.RefusedError, match="resolved"):
        za.verify_identity(identity, "10.1234/other", "Novaková", None)
    with pytest.raises(za.RefusedError, match="first author is 'Novaková'"):
        za.verify_identity(identity, DOI, "Nichols", None)
    with pytest.raises(za.RefusedError, match="does not contain 'archaeal'"):
        za.verify_identity(identity, DOI, "Novakova", "archaeal")


@pytest.mark.parametrize(
    ("member", "doi", "expected_tail"),
    [
        (
            "297",
            "10.1038/s41564-017-0028-z",
            ["https://link.springer.com/content/pdf/10.1038/s41564-017-0028-z.pdf"],
        ),
        (
            "235",
            "10.1128/mbio.02577-18",
            ["https://journals.asm.org/doi/pdf/10.1128/mbio.02577-18"],
        ),
        (
            "221",
            "10.1126/science.abk2066",
            ["https://www.science.org/doi/pdf/10.1126/science.abk2066"],
        ),
        (
            "311",
            "10.1111/mmi.1",
            ["https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/mmi.1"],
        ),
        (
            "340",
            "10.1371/journal.pgen.1004120",
            [
                "https://journals.plos.org/plosgenetics/article/file"
                "?id=10.1371/journal.pgen.1004120&type=printable"
            ],
        ),
        ("340", "10.1371/journal.xyz.1", []),
        ("286", "10.1093/nar/gku1137", []),
    ],
)
def test_publisher_pdf_urls_by_member(
    member: str, doi: str, expected_tail: list[str]
) -> None:
    """Crossref's PDF link first (deduplicated), then the member's own PDF URL."""
    work = {
        "DOI": doi,
        "member": member,
        "link": [
            {"URL": CROSSREF_LINK, "content-type": "application/pdf"},
            {"URL": CROSSREF_LINK, "content-type": "application/pdf"},
        ],
    }
    assert za.publisher_pdf_urls(work) == [CROSSREF_LINK, *expected_tail]


def test_publisher_pdf_urls_elsevier_needs_a_pii() -> None:
    assert za.publisher_pdf_urls(WORK) == [CROSSREF_LINK, ELSEVIER_PDF_URL]
    no_pii = {"DOI": DOI, "member": "78", "alternative-id": ["x"]}
    assert za.publisher_pdf_urls(no_pii) == []


@pytest.mark.parametrize(
    ("status", "body", "outcome"),
    [
        (403, HTML, za.AttemptOutcome.BLOCKED),
        (429, HTML, za.AttemptOutcome.BLOCKED),
        (404, HTML, za.AttemptOutcome.NOT_FOUND),
        (500, HTML, za.AttemptOutcome.HTTP_ERROR),
        (200, HTML, za.AttemptOutcome.NOT_PDF),
        (200, PDF, za.AttemptOutcome.PDF),
    ],
)
def test_fetch_pdf_outcomes(
    status: int, body: bytes, outcome: za.AttemptOutcome
) -> None:
    """Only verified PDF bytes come back; an HTML 200 is an interstitial."""
    server = Server({CROSSREF_LINK: (status, body, "text/html")})
    attempt, content = za.fetch_pdf(server.client(), CROSSREF_LINK, za.Route.PUBLISHER)
    assert attempt.outcome == outcome
    assert content == (PDF if outcome == za.AttemptOutcome.PDF else None)
    assert attempt.detail.startswith(f"HTTP {status} from {CROSSREF_LINK}")


def test_fetch_pdf_transport_error_is_an_outcome() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    client = httpx.Client(transport=httpx.MockTransport(refuse))
    attempt, content = za.fetch_pdf(client, CROSSREF_LINK, za.Route.PUBLISHER)
    assert (attempt.outcome, content) == (za.AttemptOutcome.HTTP_ERROR, None)
    assert attempt.detail == "ConnectError: no route to host"


def test_publisher_route_without_urls_records_why() -> None:
    attempts, file = za.publisher_route(
        Server({}).client(), {"DOI": DOI, "member": "286"}, TODAY
    )
    assert file is None
    assert [(a.outcome, a.detail) for a in attempts] == [
        (za.AttemptOutcome.NOT_FOUND, "no PDF URL known for Crossref member 286")
    ]


def test_publisher_route_records_a_direct_url_retrieval() -> None:
    server = Server({CROSSREF_LINK: (200, PDF, "application/pdf")})
    attempts, file = za.publisher_route(server.client(), WORK, TODAY)
    assert file is not None
    assert [a.outcome for a in attempts] == [za.AttemptOutcome.PDF]
    assert file.retrieval == RetrievalRecord(
        method=RetrievalMethod.direct_url,
        source_url=CROSSREF_LINK,
        retriever="torchcell.literature.retrieve.direct_url",
        params={"url": CROSSREF_LINK},
        sha256=_sha(PDF),
        retrieved_at=TODAY,
    )
    assert (file.filename, file.title) == ("10.1234_example.5678.pdf", "Full Text PDF")


# -- raw mirror ------------------------------------------------------------------------


def test_raw_mirror_route_reuses_the_recorded_retrieval(tmp_path: Path) -> None:
    key_dir = _raw_key(tmp_path, "nicholsKey2011", DOI.upper(), PDF)
    attempt, file = za.raw_mirror_route(tmp_path, DOI)
    assert file is not None
    assert attempt.outcome == za.AttemptOutcome.PDF
    assert attempt.detail == (
        "nicholsKey2011/paper/PMC7.1.pdf, recorded pmc_cloud retrieval of 2026-10-08"
    )
    assert (file.route, file.origin, file.sha256, file.md5) == (
        za.Route.RAW_MIRROR,
        str(key_dir / "paper" / "PMC7.1.pdf"),
        _sha(PDF),
        _md5(PDF),
    )
    assert file.retrieval.params == {"key": "PMC7.1/PMC7.1.pdf"}
    assert file.retrieval.retrieved_at == "2026-10-08"


def test_raw_mirror_route_refuses_hash_drift(tmp_path: Path) -> None:
    _raw_key(tmp_path, "k2011", DOI, PDF, recorded_sha="0" * 64)
    with pytest.raises(za.RefusedError, match="hashes to"):
        za.raw_mirror_route(tmp_path, DOI)


def test_raw_mirror_route_refuses_a_doi_two_keys_claim(tmp_path: Path) -> None:
    _raw_key(tmp_path, "a2011", DOI, PDF)
    _raw_key(tmp_path, "b2011", DOI, PDF)
    with pytest.raises(za.RefusedError, match=r"several raw-mirror keys: \['a2011'"):
        za.raw_mirror_route(tmp_path, DOI)


def test_raw_mirror_route_refuses_a_recorded_pdf_that_is_not_one(
    tmp_path: Path,
) -> None:
    _raw_key(tmp_path, "k2011", DOI, HTML)
    with pytest.raises(za.RefusedError, match="is not a PDF"):
        za.raw_mirror_route(tmp_path, DOI)


def test_raw_mirror_route_without_a_paper_pdf(tmp_path: Path) -> None:
    (tmp_path / "k2011").mkdir()
    write_manifest(tmp_path / "k2011", Manifest(citation_key="k2011", doi=DOI))
    attempt, file = za.raw_mirror_route(tmp_path, DOI)
    assert (attempt.outcome, attempt.detail, file) == (
        za.AttemptOutcome.NOT_FOUND,
        "k2011 records no paper_pdf file",
        None,
    )


def test_raw_mirror_route_without_the_doi_names_failing_manifests(
    tmp_path: Path,
) -> None:
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "manifest.json").write_text('{"files": 3}')
    attempt, file = za.raw_mirror_route(tmp_path, DOI)
    assert file is None
    assert attempt.detail == (
        f"no manifest under {tmp_path} records this DOI; "
        "1 manifests failed validation: ['broken']"
    )


def test_raw_mirror_route_refuses_a_record_without_retrieval(tmp_path: Path) -> None:
    record = ArtifactRecord(
        path="paper/PMC7.1.pdf", role="paper_pdf", bytes=len(PDF), sha256=_sha(PDF)
    )
    (tmp_path / "k2011").mkdir()
    write_manifest(
        tmp_path / "k2011", Manifest(citation_key="k2011", doi=DOI, files=[record])
    )
    with pytest.raises(za.RefusedError, match="has no retrieval record"):
        za.raw_mirror_route(tmp_path, DOI)


# -- PMC cloud ---------------------------------------------------------------------------


def test_pmc_cloud_route_takes_the_newest_version_pdf() -> None:
    attempts, file = za.pmc_cloud_route(Server(_pmc_with_pdf()).client(), DOI, TODAY)
    assert file is not None
    assert [a.outcome for a in attempts] == [za.AttemptOutcome.PDF]
    assert file.retrieval == RetrievalRecord(
        method=RetrievalMethod.pmc_cloud,
        source_url=PMC_PDF_URL,
        retriever="torchcell.literature.retrieve.pmc_cloud_object",
        params={"key": "PMC42.2/PMC42.2.pdf"},
        sha256=_sha(PDF),
        retrieved_at=TODAY,
    )


def test_pmc_cloud_route_author_manuscript_has_no_pdf() -> None:
    routes = _pmc_with_pdf()
    routes[LIST_URL] = (
        200,
        _listing(["PMC42.2/PMC42.2.txt", "PMC42.2/PMC42.2.xml"]),
        "",
    )
    attempts, file = za.pmc_cloud_route(Server(routes).client(), DOI, TODAY)
    assert file is None
    assert attempts[0].detail == (
        "PMC42.2 holds no article PDF (author manuscript or non-OA); "
        "objects: ['PMC42.2.txt', 'PMC42.2.xml']"
    )


@pytest.mark.parametrize(
    ("idconv", "listing", "outcome", "detail"),
    [
        (
            (200, b'{"records": [{"doi": "x"}]}', "application/json"),
            None,
            za.AttemptOutcome.NOT_FOUND,
            "the NCBI ID converter has no PMCID",
        ),
        (
            (200, b'{"records": [{"pmcid": "PMC42"}]}', "application/json"),
            (200, _listing([]), "application/xml"),
            za.AttemptOutcome.NOT_FOUND,
            "PMC42 is not in the PMC Article Datasets bucket",
        ),
        ((403, HTML, "text/html"), None, za.AttemptOutcome.BLOCKED, "HTTP 403 from"),
        (
            (502, HTML, "text/html"),
            None,
            za.AttemptOutcome.HTTP_ERROR,
            "HTTPStatusError",
        ),
    ],
)
def test_pmc_cloud_route_misses(
    idconv: tuple[int, bytes, str],
    listing: tuple[int, bytes, str] | None,
    outcome: za.AttemptOutcome,
    detail: str,
) -> None:
    routes = {IDCONV_URL: idconv}
    if listing is not None:
        routes[LIST_URL] = listing
    attempts, file = za.pmc_cloud_route(Server(routes).client(), DOI, TODAY)
    assert file is None
    assert attempts[0].outcome == outcome
    assert attempts[0].detail.startswith(detail)


# -- request and local files -------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pdf": Path("x.pdf")},
        {"pdf_source_url": "https://example.org"},
        {"si": (Path("s.pdf"),)},
        {"si_source_url": "https://example.org"},
    ],
)
def test_request_pairs_local_files_with_their_source(
    tmp_path: Path, kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError, match="go together"):
        _request(tmp_path, **kwargs)


def test_local_pdf_is_a_manual_browser_retrieval(tmp_path: Path) -> None:
    path = tmp_path / "Supp Table 1.pdf"
    path.write_bytes(OTHER_PDF)
    file = za.local_pdf(path, "https://publisher.example/suppl", "si", DOI)
    assert (file.filename, file.title, file.route) == (
        "Supp Table 1.pdf",
        "Supplementary material: Supp Table 1.pdf",
        za.Route.LOCAL_FILE,
    )
    assert file.retrieval.method == RetrievalMethod.manual_browser
    assert file.retrieval.retriever == "manual"
    assert file.retrieval.sha256 == _sha(OTHER_PDF)
    assert file.retrieval.params["local_path"] == str(path)
    html_file = tmp_path / "page.pdf"
    html_file.write_bytes(HTML)
    with pytest.raises(za.RefusedError, match="is not a PDF"):
        za.local_pdf(html_file, "https://publisher.example", "paper", DOI)


# -- add_paper end to end ----------------------------------------------------------------


def test_add_paper_creates_the_item_with_its_pdf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Publisher blocked, no raw copy, PMC has the PDF: one item, one attachment."""
    zot = WritableZot(collections=COLLECTIONS)
    server = Server({**_blocked_publisher(), **_pmc_with_pdf()})
    report = za.add_paper(
        _request(tmp_path),
        _lib(monkeypatch, zot),
        server.client(),
        upload=recording_upload,
        today=TODAY,
    )
    assert report.outcome == za.Outcome.FILED
    assert (report.item_key, report.created_item) == ("NEWITEM1", True)
    assert [(a.route, a.outcome) for a in report.attempts] == [
        (za.Route.PUBLISHER, za.AttemptOutcome.BLOCKED),
        (za.Route.PUBLISHER, za.AttemptOutcome.BLOCKED),
        (za.Route.RAW_MIRROR, za.AttemptOutcome.NOT_FOUND),
        (za.Route.PMC_CLOUD, za.AttemptOutcome.PDF),
    ]
    created = zot.created[0][0]
    assert (created["collections"], created["DOI"], created["extra"]) == (
        ["ECOLI"],
        DOI,
        "added-by: scripts/zotero_add_paper.py (CrossRef)",
    )
    parent, template, content = zot.uploads[0]
    assert (parent, content, template["title"]) == ("NEWITEM1", PDF, "Full Text PDF")
    assert _sha(PDF) in template["note"]
    assert "pmc_cloud_object" in template["note"]
    assert report.files == [
        za.AttachedFile(
            attachment_key="ATT1",
            role="paper",
            filename="10.1234_example.5678.pdf",
            route=za.Route.PMC_CLOUD,
            method=RetrievalMethod.pmc_cloud,
            source_url=PMC_PDF_URL,
            origin=PMC_PDF_URL,
            sha256=_sha(PDF),
            md5=_md5(PDF),
            n_bytes=len(PDF),
        )
    ]
    assert zot.deleted == []


def test_add_paper_prefers_the_raw_mirror_to_pmc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A recorded raw-mirror copy is used, and PMC is never asked."""
    zot = WritableZot(collections=COLLECTIONS)
    server = Server(_blocked_publisher())
    request = _request(tmp_path)
    _raw_key(request.raw_root, "k2011", DOI, PDF)
    report = za.add_paper(
        request, _lib(monkeypatch, zot), server.client(), upload=recording_upload
    )
    assert report.files[0].route == za.Route.RAW_MIRROR
    assert report.files[0].method == RetrievalMethod.pmc_cloud
    assert IDCONV_URL not in server.seen


def test_add_paper_refuses_a_doi_the_library_holds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    zot = WritableZot(collections=COLLECTIONS, items=[_item("HAVEIT", ["DBROOT"])])
    server = Server({CROSSREF_URL: _crossref()})
    with pytest.raises(za.RefusedError, match="already item HAVEIT"):
        za.add_paper(_request(tmp_path), _lib(monkeypatch, zot), server.client())
    assert zot.writes() == (0, 0, 0)


@pytest.mark.parametrize(
    ("items", "attach_to", "message"),
    [
        ([], "NOPE", "no item in the group library has DOI"),
        ([_item("REAL", ["ECOLI"])], "OTHER", "belongs to item REAL"),
        (
            [_item("REAL", ["PAPER", "PPUT"])],
            "REAL",
            "is in neither database/Escherichia-coli",
        ),
    ],
)
def test_add_paper_attach_to_refusals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    items: list[dict[str, Any]],
    attach_to: str,
    message: str,
) -> None:
    zot = WritableZot(collections=COLLECTIONS, items=items)
    server = Server({CROSSREF_URL: _crossref()})
    with pytest.raises(za.RefusedError, match=message):
        za.add_paper(
            _request(tmp_path, attach_to=attach_to),
            _lib(monkeypatch, zot),
            server.client(),
        )
    assert zot.writes() == (0, 0, 0)


def _main_pdf(key: str, md5: str) -> dict[str, Any]:
    return {
        "key": key,
        "data": {
            "contentType": "application/pdf",
            "title": "Full Text PDF",
            "md5": md5,
        },
    }


def test_add_paper_refuses_a_second_article_pdf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    zot = WritableZot(
        collections=COLLECTIONS,
        items=[_item("REAL", ["ECOLI"])],
        children={"REAL": [_main_pdf("PDF1", _md5(PDF))]},
    )
    server = Server({CROSSREF_URL: _crossref()})
    with pytest.raises(za.RefusedError, match="already has an article PDF"):
        za.add_paper(
            _request(tmp_path, attach_to="REAL"),
            _lib(monkeypatch, zot),
            server.client(),
        )
    assert zot.writes() == (0, 0, 0)


def test_add_paper_attaches_si_to_an_item_with_its_pdf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No article route runs; the SI is uploaded under the existing item."""
    si = tmp_path / "mmc1.pdf"
    si.write_bytes(OTHER_PDF)
    zot = WritableZot(
        collections=COLLECTIONS,
        items=[_item("REAL", ["ECOLI"])],
        children={"REAL": [_main_pdf("PDF1", _md5(PDF))]},
    )
    server = Server({CROSSREF_URL: _crossref()})
    report = za.add_paper(
        _request(
            tmp_path,
            attach_to="REAL",
            si=(si,),
            si_source_url="https://publisher.example/suppl",
        ),
        _lib(monkeypatch, zot),
        server.client(),
        upload=recording_upload,
    )
    assert server.seen == [CROSSREF_URL]
    assert (report.outcome, report.created_item, report.attempts) == (
        za.Outcome.FILED,
        False,
        [],
    )
    assert [(f.attachment_key, f.role, f.filename) for f in report.files] == [
        ("ATT1", "si", "mmc1.pdf")
    ]
    assert zot.uploads[0][1]["title"] == "Supplementary material: mmc1.pdf"
    assert report.notes[0].startswith("SI PDFs attached: once lit_sync mirrors them")


def test_add_paper_refuses_an_si_file_already_attached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    si = tmp_path / "mmc1.pdf"
    si.write_bytes(PDF)
    zot = WritableZot(
        collections=COLLECTIONS,
        items=[_item("REAL", ["ECOLI"])],
        children={"REAL": [_main_pdf("PDF1", _md5(PDF))]},
    )
    with pytest.raises(za.RefusedError, match="already attached to REAL as PDF1"):
        za.add_paper(
            _request(
                tmp_path, attach_to="REAL", si=(si,), si_source_url="https://x.example"
            ),
            _lib(monkeypatch, zot),
            Server({CROSSREF_URL: _crossref()}).client(),
        )
    assert zot.writes() == (0, 0, 0)


def test_add_paper_without_any_pdf_is_manual_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    routes = {**_blocked_publisher(), **_pmc_with_pdf()}
    routes[LIST_URL] = (200, _listing(["PMC42.2/PMC42.2.xml"]), "application/xml")
    zot = WritableZot(collections=COLLECTIONS, items=[_item("REAL", ["ECOLI"])])
    report = za.add_paper(
        _request(tmp_path, attach_to="REAL"),
        _lib(monkeypatch, zot),
        Server(routes).client(),
    )
    assert report.outcome == za.Outcome.MANUAL
    assert report.manual_url == f"https://doi.org/{DOI}"
    assert zot.writes() == (0, 0, 0)


def test_add_paper_uses_the_local_pdf_last(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    routes = {**_blocked_publisher(), **_pmc_with_pdf()}
    routes[LIST_URL] = (200, _listing(["PMC42.2/PMC42.2.xml"]), "application/xml")
    local = tmp_path / "download.pdf"
    local.write_bytes(PDF)
    zot = WritableZot(collections=COLLECTIONS, items=[_item("REAL", ["ECOLI"])])
    report = za.add_paper(
        _request(
            tmp_path,
            attach_to="REAL",
            pdf=local,
            pdf_source_url="https://doi.org/10.1234/example.5678",
        ),
        _lib(monkeypatch, zot),
        Server(routes).client(),
        upload=recording_upload,
    )
    assert report.outcome == za.Outcome.FILED
    assert report.attempts[-1].route == za.Route.LOCAL_FILE
    assert (report.files[0].method, report.files[0].source_url) == (
        RetrievalMethod.manual_browser,
        "https://doi.org/10.1234/example.5678",
    )
    assert zot.created == []


def test_add_paper_dry_run_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    zot = WritableZot(collections=COLLECTIONS)
    report = za.add_paper(
        _request(tmp_path, dry_run=True),
        _lib(monkeypatch, zot),
        Server({**_blocked_publisher(), **_pmc_with_pdf()}).client(),
    )
    assert report.outcome == za.Outcome.WOULD_FILE
    assert [(f.attachment_key, f.sha256) for f in report.files] == [(None, _sha(PDF))]
    assert zot.writes() == (0, 0, 0)


def test_add_paper_rolls_back_a_created_item_whose_upload_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The item this run created is deleted, so no metadata-only item remains."""
    zot = WritableZot(collections=COLLECTIONS)
    with pytest.raises(RuntimeError, match="upload of 10.1234_example.5678.pdf failed"):
        za.add_paper(
            _request(tmp_path),
            _lib(monkeypatch, zot),
            Server({**_blocked_publisher(), **_pmc_with_pdf()}).client(),
            upload=failing_upload,
        )
    assert zot.deleted == ["NEWITEM1"]


def test_add_paper_checks_the_stored_md5(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An upload whose stored MD5 differs from the bytes sent fails loudly."""

    def corrupting_upload(
        zot: Any, templates: list[dict[str, Any]], parent: str, basedir: str
    ) -> dict[str, list[Any]]:
        result = recording_upload(zot, templates, parent, basedir)
        zot._children[parent][-1]["data"]["md5"] = "0" * 32
        return result

    zot = WritableZot(collections=COLLECTIONS)
    with pytest.raises(RuntimeError, match="stores md5 0{32}"):
        za.add_paper(
            _request(tmp_path),
            _lib(monkeypatch, zot),
            Server({**_blocked_publisher(), **_pmc_with_pdf()}).client(),
            upload=corrupting_upload,
        )


def test_add_paper_refuses_a_doi_crossref_does_not_know(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    zot = WritableZot(collections=COLLECTIONS)
    server = Server({CROSSREF_URL: (404, b"Resource not found.", "text/plain")})
    with pytest.raises(za.RefusedError, match="Crossref does not know"):
        za.add_paper(_request(tmp_path), _lib(monkeypatch, zot), server.client())


# -- report, upload wrapper, command line --------------------------------------------------


def test_format_report_lists_attempts_and_files() -> None:
    report = za.AddPaperReport(
        doi=DOI,
        title="T",
        first_author="Nichols",
        collection="database/Escherichia-coli",
        collection_key="ECOLI",
        item_key="NEWITEM1",
        created_item=True,
        dry_run=False,
        outcome=za.Outcome.FILED,
        attempts=[
            za.RouteAttempt(
                route=za.Route.PUBLISHER,
                url=CROSSREF_LINK,
                outcome=za.AttemptOutcome.BLOCKED,
                detail="HTTP 403",
            )
        ],
        notes=["n1"],
    )
    assert za.format_report(report).splitlines() == [
        f"{DOI}  Nichols  T",
        "collection database/Escherichia-coli = ECOLI",
        f"  publisher  blocked    {CROSSREF_LINK}  (HTTP 403)",
        "outcome filed: item NEWITEM1 (created)",
        "  note: n1",
    ]


def test_zupload_passes_the_bare_filename_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[Any, ...]] = []

    class FakeZupload:
        def __init__(
            self, zot: Any, payload: list[dict[str, Any]], parent: str, basedir: str
        ) -> None:
            seen.append((zot, payload, parent, basedir))

        def upload(self) -> dict[str, list[Any]]:
            return {"success": [{"key": "A"}], "failure": [], "unchanged": []}

    monkeypatch.setattr(za, "Zupload", FakeZupload)
    result = za.zupload("ZOT", [{"filename": "a.pdf"}], "PARENT", "/tmp/dir")
    assert result["success"] == [{"key": "A"}]
    assert seen == [("ZOT", [{"filename": "a.pdf"}], "PARENT", "/tmp/dir")]


def test_main_refuses_a_collection_outside_database(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = za.main(["--doi", DOI, "--collection", "paper", "--first-author", "Nichols"])
    assert code == za.EXIT_REFUSED
    assert capsys.readouterr().err.startswith(
        "REFUSED: collection 'paper' is not a database/<Organism> collection"
    )


def test_main_requires_a_source_url_with_pdf() -> None:
    with pytest.raises(SystemExit) as exc:
        za.main(
            [
                "--doi",
                DOI,
                "--collection",
                "database/Escherichia-coli",
                "--first-author",
                "Nichols",
                "--pdf",
                "x.pdf",
            ]
        )
    assert exc.value.code == 2


def test_main_files_and_writes_the_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    zot = WritableZot(collections=COLLECTIONS)
    lib = _lib(monkeypatch, zot)
    server = Server({**_blocked_publisher(), **_pmc_with_pdf()})
    monkeypatch.setattr(ZoteroLibrary, "from_env", classmethod(lambda cls: lib))
    monkeypatch.setattr(za, "make_client", server.client)
    (tmp_path / "raw").mkdir()
    report_path = tmp_path / "report.json"
    code = za.main(
        [
            "--doi",
            DOI,
            "--collection",
            "database/Escherichia-coli",
            "--first-author",
            "Nichols",
            "--raw-root",
            str(tmp_path / "raw"),
            "--dry-run",
            "--report",
            str(report_path),
        ]
    )
    assert code == 0
    assert json.loads(report_path.read_text())["outcome"] == "would_file"
    assert "outcome would_file: item -" in capsys.readouterr().out


def test_add_paper_attaches_to_an_owner_item_in_the_database_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An item the owner filed in the ``database`` root gets its PDF; nothing moves."""
    zot = WritableZot(collections=COLLECTIONS, items=[_item("OWNER", ["DBROOT"])])
    report = za.add_paper(
        _request(tmp_path, attach_to="OWNER"),
        _lib(monkeypatch, zot),
        Server({**_blocked_publisher(), **_pmc_with_pdf()}).client(),
        upload=recording_upload,
    )
    assert (report.outcome, report.item_key, report.created_item) == (
        za.Outcome.FILED,
        "OWNER",
        False,
    )
    assert [(f.attachment_key, f.role) for f in report.files] == [("ATT1", "paper")]
    assert report.notes == [
        "item OWNER sits in the database root, not in database/Escherichia-coli; "
        "its collection membership is left as it is"
    ]
    assert zot.created == []
    assert zot.item("OWNER")["data"]["collections"] == ["DBROOT"]
