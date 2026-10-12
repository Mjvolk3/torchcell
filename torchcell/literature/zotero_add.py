# torchcell/literature/zotero_add.py
# [[torchcell.literature.zotero_add]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/zotero_add.py
# Test file: tests/torchcell/literature/test_zotero_add.py

"""File a dataset's primary paper, with its PDF, into a group-library database collection.

The sanctioned agent write path into Zotero (owner policy of 2026-10-12, recorded in
``CLAUDE.md``). In the TorchCell GROUP library (``6582362``) the ``database/<Organism>``
collections hold the papers the knowledge graph is built from, and their purpose is to
collect each dataset's primary paper from the publisher. Exactly three writes are
allowed:

1. create an item for a dataset's primary paper WITH its PDF attached, in the named
   ``database/<Organism>`` collection;
2. attach supplementary PDFs to that item (``--si``);
3. attach the article PDF to an existing item that has none (``--attach-to``), when
   the item sits in the named collection or in the ``database`` root, where the owner
   files dataset papers too. Its collection membership is left as it is.

Everything else is refused: a metadata-only item (an item is created only after its PDF
has been retrieved and verified, and is deleted again if the upload of that PDF fails),
a new item outside ``database/<Organism>``, an attachment on an item outside the
``database`` tree, a second item for a DOI the library already holds, a second article
PDF on an item that has one, a file that is not a PDF, and any library other than the
group. Citation keys, collection membership, merges and ``references.bib`` are never
touched.

The article PDF is taken from the first route that yields verified PDF bytes:

``publisher``
    The version of record: every ``application/pdf`` link Crossref lists for the DOI,
    then the publisher's own PDF URL by Crossref member (Elsevier and Cell Press
    ``sciencedirect.com/.../pdfft``, Springer Nature ``link.springer.com/content/pdf``,
    ASM / ACS / AAAS ``/doi/pdf``, Wiley ``pdfdirect``, PLOS ``type=printable``).
    Measured 2026-10-12 from GilaHyper: Elsevier, Cell Press, OUP, ASM and AAAS answer
    HTTP 403 and ``nature.com`` redirects a non-subscriber to the HTML article (HTTP
    200), so the route is tried first and usually blocked.
``raw_mirror``
    A PDF already mirrored under ``$DATA_ROOT/torchcell-raw/<key>/`` with role
    ``paper_pdf`` in a manifest whose ``doi`` matches. The recorded retrieval and sha256
    are reused (the file is re-hashed and must still match); nothing is downloaded again,
    so the attachment is byte-identical to the copy the loaders were built against.
``pmc_cloud``
    The article PDF of the newest version in the PMC Article Datasets bucket
    (``<PMCID>.<v>/<PMCID>.<v>.pdf``), the successor of the retired PMC OA API. Author
    manuscripts in the bucket carry text and XML only, so the route finds no PDF for
    them.
``local_file``
    ``--pdf``: a file the owner downloaded in a browser, recorded as a
    ``manual_browser`` retrieval from ``--pdf-source-url``.

Every response is checked to be a PDF (``%PDF-`` header in the first kilobyte and a
``%%EOF`` trailer near the end) so a publisher's HTML interstitial is never attached.
Each attachment carries its retrieval record (a
:class:`~torchcell.literature.manifest.RetrievalRecord`: method, source URL, retriever
and params, sha256, date) as its Zotero note, and after the upload the MD5 Zotero
stores is compared with the bytes sent.

Attaching SI PDFs has a cost to know about: once ``lit_sync`` mirrors them as
``si/si<N>.pdf``, :mod:`torchcell.literature.capture_si` treats the key's SI as held
from another path and no longer fetches the publisher's supplementary files (data
tables included) for it.

``scripts/zotero_add_paper.py`` is the command-line wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import os
import re
import sys
import tempfile
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pyzotero.zotero import Zupload

from torchcell.literature.capture_si import (
    ATYPON_HOSTS,
    MEMBER_ELSEVIER,
    MEMBER_PLOS,
    MEMBER_SPRINGER,
    MEMBER_WILEY,
    PLOS_JOURNALS,
    BlockedError,
    _is_blocked,
    _version_number,
    bucket_keys,
    crossref_work,
    elsevier_pii,
    make_client,
    pmcid_for_doi,
)
from torchcell.literature.manifest import (
    ROLE_PAPER_PDF,
    RetrievalMethod,
    RetrievalRecord,
    scan_mirror,
)
from torchcell.literature.retrieve import pmc_cloud_url, user_agent_for
from torchcell.literature.sync import DATABASE_COLLECTION
from torchcell.literature.zotero import (
    ZoteroLibrary,
    _is_main_article,
    with_zotero_retry,
)

#: The TorchCell group library: the only library this module writes to.
GROUP_LIBRARY_ID = "6582362"
TOOL = "scripts/zotero_add_paper.py"
#: The repo's marker for an item that did not arrive by hand, so it can be reviewed.
EXTRA_MARKER = f"added-by: {TOOL} (CrossRef)"
#: Title of the article attachment; ``_is_main_article`` reads "full text" as main.
MAIN_TITLE = "Full Text PDF"
#: Prefix of an SI attachment title; ``_is_si_attachment`` reads "supplement" as SI.
SI_TITLE_PREFIX = "Supplementary material"
R_DIRECT = "torchcell.literature.retrieve.direct_url"
R_PMC_CLOUD = "torchcell.literature.retrieve.pmc_cloud_object"
R_MANUAL = "manual"
PDF_MAGIC = b"%PDF-"
PDF_EOF = b"%%EOF"
#: The PDF header may follow up to 1 KiB of leading bytes (ISO 32000-1, 7.5.2).
PDF_HEADER_WINDOW = 1024
#: Window at the end of the file searched for the ``%%EOF`` trailer.
PDF_TRAILER_WINDOW = 2048
EXIT_REFUSED = 2
EXIT_MANUAL = 3


class RefusedError(RuntimeError):
    """A policy refusal. Raised before any write, so nothing was written."""


class Route(StrEnum):
    """Where the article PDF was taken from, in the order the routes are tried."""

    PUBLISHER = "publisher"
    RAW_MIRROR = "raw_mirror"
    PMC_CLOUD = "pmc_cloud"
    LOCAL_FILE = "local_file"


class AttemptOutcome(StrEnum):
    """What one URL or mirror lookup gave."""

    PDF = "pdf"
    BLOCKED = "blocked"
    NOT_PDF = "not_pdf"
    NOT_FOUND = "not_found"
    HTTP_ERROR = "http_error"


class Outcome(StrEnum):
    """Result of one run."""

    FILED = "filed"
    WOULD_FILE = "would_file"
    MANUAL = "manual"


class PaperIdentity(BaseModel):
    """Who and what a DOI names, per Crossref."""

    model_config = ConfigDict(frozen=True)

    doi: str
    title: str = Field(description="Crossref title, whitespace collapsed.")
    first_author: str = Field(description="Family name of the first author.")
    date: str = Field(description="``issued`` date, YYYY[-MM[-DD]].")
    container: str
    member: str = Field(description="Crossref member id of the publisher.")


class RouteAttempt(BaseModel):
    """One URL fetched or one mirror lookup, and what it gave."""

    route: Route
    url: str | None
    outcome: AttemptOutcome
    detail: str


class PdfFile(BaseModel):
    """A verified PDF ready to attach, with the retrieval that produced it."""

    role: Literal["paper", "si"]
    route: Route
    filename: str
    title: str
    origin: str = Field(description="URL fetched or local path read.")
    sha256: str
    md5: str = Field(description="What Zotero stores for the file; checked on upload.")
    n_bytes: int
    retrieval: RetrievalRecord
    content: bytes = Field(exclude=True, repr=False)


class AttachedFile(BaseModel):
    """One attachment written (or, in a dry run, planned)."""

    attachment_key: str | None
    role: Literal["paper", "si"]
    filename: str
    route: Route
    method: RetrievalMethod
    source_url: str | None
    origin: str
    sha256: str
    md5: str
    n_bytes: int


class AddPaperRequest(BaseModel):
    """One filing, as given on the command line."""

    model_config = ConfigDict(frozen=True)

    doi: str
    collection: str
    first_author: str = Field(
        description="Expected first-author family name; Crossref must agree."
    )
    title_contains: str | None = None
    pdf: Path | None = None
    pdf_source_url: str | None = None
    si: tuple[Path, ...] = ()
    si_source_url: str | None = None
    attach_to: str | None = None
    dry_run: bool = False
    raw_root: Path

    @model_validator(mode="after")
    def _paired_sources(self) -> AddPaperRequest:
        """A local file is a manual retrieval, so it needs the URL it came from."""
        if (self.pdf is None) != (self.pdf_source_url is None):
            raise ValueError("--pdf and --pdf-source-url go together")
        if bool(self.si) != (self.si_source_url is not None):
            raise ValueError("--si and --si-source-url go together")
        return self


class AddPaperReport(BaseModel):
    """Everything one run found and did."""

    doi: str
    title: str
    first_author: str
    collection: str
    collection_key: str
    item_key: str | None
    created_item: bool
    dry_run: bool
    outcome: Outcome
    attempts: list[RouteAttempt] = Field(default_factory=list)
    files: list[AttachedFile] = Field(default_factory=list)
    manual_url: str | None = None
    notes: list[str] = Field(default_factory=list)


UploadFn = Callable[[Any, list[dict[str, Any]], str, str], dict[str, list[Any]]]


def zupload(
    zot: Any, templates: list[dict[str, Any]], parent_key: str, basedir: str
) -> dict[str, list[Any]]:
    """Create attachment items under ``parent_key`` and upload their files.

    Not pyzotero's ``attachment_simple``: it puts the path it is handed into the item's
    ``filename`` and the API rejects that ("Stored-file filename ... cannot contain a
    directory path"). ``Zupload`` with ``basedir`` sends the bare name, as
    ``notes-tex/common/zotero_publish.py`` does.
    """
    result: dict[str, list[Any]] = Zupload(
        zot, templates, parent_key, basedir=basedir
    ).upload()
    return result


# -- checks ----------------------------------------------------------------------


def is_pdf_bytes(content: bytes) -> bool:
    """``%PDF-`` in the first KiB and ``%%EOF`` in the last 2 KiB.

    An HTML interstitial (a challenge page, a login redirect, the article page a
    ``.pdf`` URL redirects to) fails the header test; a truncated download fails the
    trailer test.
    """
    return (
        PDF_MAGIC in content[:PDF_HEADER_WINDOW]
        and PDF_EOF in content[-PDF_TRAILER_WINDOW:]
    )


def check_collection_path(path: str) -> str:
    """Accept exactly ``database/<Name>``; refuse every other collection."""
    parts = path.split("/")
    if len(parts) != 2 or parts[0] != DATABASE_COLLECTION or not parts[1]:
        raise RefusedError(
            f"collection {path!r} is not a database/<Organism> collection; agents "
            "write only to those (CLAUDE.md, Provenance & Reproducibility)"
        )
    return path


def check_group_library(lib: ZoteroLibrary) -> None:
    """Refuse any library but the TorchCell group."""
    config = lib.config
    if config.library_type != "group" or config.library_id != GROUP_LIBRARY_ID:
        raise RefusedError(
            f"library {config.library_type}/{config.library_id} is not the TorchCell "
            f"group library {GROUP_LIBRARY_ID}"
        )


def resolve_collection(lib: ZoteroLibrary, path: str) -> tuple[str, str]:
    """Keys of collection ``path`` (``database/<Name>``) and of the ``database`` root."""
    nodes = lib.collection_tree(DATABASE_COLLECTION)
    matches = [n.key for n in nodes if n.path == path]
    if len(matches) != 1:
        available = sorted(n.path for n in nodes if n.path != DATABASE_COLLECTION)
        raise RefusedError(
            f"collection {path!r}: {len(matches)} matches in the group library; "
            f"the database collections are {available}"
        )
    return matches[0], nodes[0].key


def _fold(text: str) -> str:
    """Case- and accent-insensitive form for name and title comparison."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def _collapse(text: str) -> str:
    """Runs of whitespace (Crossref titles carry newlines) to one space."""
    return re.sub(r"\s+", " ", text).strip()


def _plain(text: str) -> str:
    """Markup-free text (Crossref titles carry ``<i>`` and the like)."""
    return _collapse(html.unescape(re.sub(r"<[^>]+>", "", text)))


def _first_author(work: dict[str, Any]) -> dict[str, Any]:
    """The author Crossref marks ``sequence: first``, else the first listed."""
    authors: list[dict[str, Any]] = work.get("author") or []
    if not authors:
        raise RefusedError(f"Crossref lists no author for {work.get('DOI')}")
    return next((a for a in authors if a.get("sequence") == "first"), authors[0])


def _issued(work: dict[str, Any]) -> str:
    """Crossref ``issued`` as ``YYYY``, ``YYYY-MM`` or ``YYYY-MM-DD``."""
    parts = work["issued"]["date-parts"][0]
    if not parts or parts[0] is None:
        raise RefusedError(f"Crossref gives no issued date for {work.get('DOI')}")
    return "-".join([str(parts[0]), *(f"{int(p):02d}" for p in parts[1:])])


def paper_identity(work: dict[str, Any]) -> PaperIdentity:
    """The identity fields of a Crossref ``message``."""
    titles: list[str] = work.get("title") or []
    if not titles:
        raise RefusedError(f"Crossref gives no title for {work.get('DOI')}")
    first = _first_author(work)
    family = first.get("family") or first.get("name")
    if not family:
        raise RefusedError(f"Crossref's first author of {work.get('DOI')} has no name")
    containers: list[str] = work.get("container-title") or []
    return PaperIdentity(
        doi=str(work["DOI"]),
        title=_collapse(titles[0]),
        first_author=str(family),
        date=_issued(work),
        container=containers[0] if containers else "",
        member=str(work["member"]),
    )


def verify_identity(
    identity: PaperIdentity, doi: str, first_author: str, title_contains: str | None
) -> None:
    """Refuse unless Crossref's record is the paper the caller meant."""
    if identity.doi.lower() != doi.strip().lower():
        raise RefusedError(f"Crossref resolved {doi} to {identity.doi}")
    if _fold(identity.first_author) != _fold(first_author):
        raise RefusedError(
            f"{doi}: Crossref's first author is {identity.first_author!r}, "
            f"not {first_author!r}"
        )
    if title_contains is not None and _fold(title_contains) not in _fold(
        _plain(identity.title)
    ):
        raise RefusedError(
            f"{doi}: Crossref's title {identity.title!r} does not contain "
            f"{title_contains!r}"
        )


def journal_article_fields(
    work: dict[str, Any], identity: PaperIdentity
) -> dict[str, Any]:
    """Zotero ``journalArticle`` fields from Crossref; only fields Crossref gives."""
    creators: list[dict[str, str]] = []
    for author in work.get("author") or []:
        if author.get("family"):
            creators.append(
                {
                    "creatorType": "author",
                    "firstName": str(author.get("given", "")),
                    "lastName": str(author["family"]),
                }
            )
        else:
            creators.append({"creatorType": "author", "name": str(author["name"])})
    return {
        "title": identity.title,
        "creators": creators,
        "publicationTitle": identity.container,
        "volume": str(work.get("volume", "")),
        "issue": str(work.get("issue", "")),
        "pages": str(work.get("page", "")),
        "date": identity.date,
        "DOI": identity.doi,
        "ISSN": ", ".join(work.get("ISSN") or []),
        "url": f"https://doi.org/{identity.doi}",
        "extra": EXTRA_MARKER,
    }


# -- routes ----------------------------------------------------------------------


def _hashes(content: bytes) -> tuple[str, str]:
    """(sha256, md5) hex digests; Zotero records the MD5 of a stored file."""
    return (
        hashlib.sha256(content).hexdigest(),
        hashlib.md5(content, usedforsecurity=False).hexdigest(),
    )


def _doi_filename(doi: str) -> str:
    """A stored-file name for an article PDF: the DOI with ``/`` as ``_``."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", doi) + ".pdf"


def _paper_file(
    content: bytes, route: Route, origin: str, retrieval: RetrievalRecord, doi: str
) -> PdfFile:
    sha256, md5 = _hashes(content)
    return PdfFile(
        role="paper",
        route=route,
        filename=_doi_filename(doi),
        title=MAIN_TITLE,
        origin=origin,
        sha256=sha256,
        md5=md5,
        n_bytes=len(content),
        retrieval=retrieval,
        content=content,
    )


def publisher_pdf_urls(work: dict[str, Any]) -> list[str]:
    """Version-of-record PDF URLs: Crossref's PDF links, then the member's own URL."""
    doi = str(work["DOI"])
    urls = [
        str(link["URL"])
        for link in work.get("link") or []
        if link.get("content-type") == "application/pdf"
    ]
    member = str(work["member"])
    if member == MEMBER_ELSEVIER:
        pii = elsevier_pii(work)
        if pii is not None:
            urls.append(
                f"https://www.sciencedirect.com/science/article/pii/{pii}/pdfft"
                "?isDTMRedir=true&download=true"
            )
    elif member == MEMBER_SPRINGER:
        urls.append(f"https://link.springer.com/content/pdf/{doi}.pdf")
    elif member in ATYPON_HOSTS:
        urls.append(f"{ATYPON_HOSTS[member][1]}/doi/pdf/{doi}")
    elif member == MEMBER_WILEY:
        urls.append(f"https://onlinelibrary.wiley.com/doi/pdfdirect/{doi}")
    elif member == MEMBER_PLOS:
        match = re.fullmatch(r"10\.1371/journal\.([a-z]+)\.\d+", doi, re.IGNORECASE)
        if match is not None and match.group(1).lower() in PLOS_JOURNALS:
            journal = PLOS_JOURNALS[match.group(1).lower()]
            urls.append(
                f"https://journals.plos.org/{journal}/article/file"
                f"?id={doi}&type=printable"
            )
    return list(dict.fromkeys(urls))


def fetch_pdf(
    client: httpx.Client, url: str, route: Route
) -> tuple[RouteAttempt, bytes | None]:
    """GET ``url`` with the User-Agent its recorded retriever would send.

    A transport error or an unexpected status is an attempt outcome, not a crash: the
    next route is tried. Only verified PDF bytes are returned.
    """
    try:
        resp = client.get(url, headers={"User-Agent": user_agent_for(url)})
    except httpx.HTTPError as exc:
        return RouteAttempt(
            route=route,
            url=url,
            outcome=AttemptOutcome.HTTP_ERROR,
            detail=f"{type(exc).__name__}: {exc}",
        ), None
    status = f"HTTP {resp.status_code} from {resp.url}"
    if _is_blocked(resp):
        return RouteAttempt(
            route=route, url=url, outcome=AttemptOutcome.BLOCKED, detail=status
        ), None
    if resp.status_code == 404:
        return RouteAttempt(
            route=route, url=url, outcome=AttemptOutcome.NOT_FOUND, detail=status
        ), None
    if not resp.is_success:
        return RouteAttempt(
            route=route, url=url, outcome=AttemptOutcome.HTTP_ERROR, detail=status
        ), None
    if not is_pdf_bytes(resp.content):
        kind = resp.headers.get("content-type", "no content-type")
        return RouteAttempt(
            route=route,
            url=url,
            outcome=AttemptOutcome.NOT_PDF,
            detail=f"{status}: {len(resp.content)} bytes of {kind}, not a PDF",
        ), None
    return RouteAttempt(
        route=route, url=url, outcome=AttemptOutcome.PDF, detail=status
    ), resp.content


def publisher_route(
    client: httpx.Client, work: dict[str, Any], today: str
) -> tuple[list[RouteAttempt], PdfFile | None]:
    """The version of record from the publisher, if any URL serves it."""
    attempts: list[RouteAttempt] = []
    for url in publisher_pdf_urls(work):
        attempt, content = fetch_pdf(client, url, Route.PUBLISHER)
        attempts.append(attempt)
        if content is not None:
            retrieval = RetrievalRecord(
                method=RetrievalMethod.direct_url,
                source_url=url,
                retriever=R_DIRECT,
                params={"url": url},
                sha256=_hashes(content)[0],
                retrieved_at=today,
            )
            file = _paper_file(
                content, Route.PUBLISHER, url, retrieval, str(work["DOI"])
            )
            return attempts, file
    if not attempts:
        attempts.append(
            RouteAttempt(
                route=Route.PUBLISHER,
                url=None,
                outcome=AttemptOutcome.NOT_FOUND,
                detail=f"no PDF URL known for Crossref member {work['member']}",
            )
        )
    return attempts, None


def raw_mirror_route(raw_root: Path, doi: str) -> tuple[RouteAttempt, PdfFile | None]:
    """The ``paper_pdf`` a raw-mirror manifest records for this DOI, re-verified.

    Refuses (never skips) a recorded file whose bytes no longer hash to the record, a
    record without its retrieval, and a DOI that two raw keys claim.
    """
    scan = scan_mirror(raw_root)
    keys = sorted(
        key
        for key, manifest in scan.manifests.items()
        if manifest.doi is not None and manifest.doi.strip().lower() == doi.lower()
    )
    if len(keys) > 1:
        raise RefusedError(f"{doi} is recorded by several raw-mirror keys: {keys}")
    if not keys:
        failing = [f.citation_key for f in scan.failures]
        detail = f"no manifest under {raw_root} records this DOI"
        if failing:
            detail += f"; {len(failing)} manifests failed validation: {failing}"
        return RouteAttempt(
            route=Route.RAW_MIRROR,
            url=None,
            outcome=AttemptOutcome.NOT_FOUND,
            detail=detail,
        ), None
    key = keys[0]
    pdfs = [f for f in scan.manifests[key].files if f.role == ROLE_PAPER_PDF]
    if not pdfs:
        return RouteAttempt(
            route=Route.RAW_MIRROR,
            url=None,
            outcome=AttemptOutcome.NOT_FOUND,
            detail=f"{key} records no {ROLE_PAPER_PDF} file",
        ), None
    if len(pdfs) > 1:
        raise RefusedError(
            f"{key} records {len(pdfs)} {ROLE_PAPER_PDF} files: "
            f"{[f.path for f in pdfs]}"
        )
    record = pdfs[0]
    path = raw_root / key / record.path
    if record.retrieval is None:
        raise RefusedError(f"{path} has no retrieval record to attach it under")
    content = path.read_bytes()
    sha256, _ = _hashes(content)
    if sha256 != record.sha256 or record.retrieval.sha256 != record.sha256:
        raise RefusedError(
            f"{path} hashes to {sha256}; its manifest records {record.sha256} "
            f"(retrieval record {record.retrieval.sha256})"
        )
    if not is_pdf_bytes(content):
        raise RefusedError(f"{path} is recorded as {ROLE_PAPER_PDF} but is not a PDF")
    attempt = RouteAttempt(
        route=Route.RAW_MIRROR,
        url=record.retrieval.source_url,
        outcome=AttemptOutcome.PDF,
        detail=f"{key}/{record.path}, recorded {record.retrieval.method} "
        f"retrieval of {record.retrieval.retrieved_at}",
    )
    return attempt, _paper_file(
        content, Route.RAW_MIRROR, str(path), record.retrieval, doi
    )


def pmc_cloud_route(
    client: httpx.Client, doi: str, today: str
) -> tuple[list[RouteAttempt], PdfFile | None]:
    """The article PDF of the newest PMC version in the PMC Article Datasets bucket."""

    def miss(outcome: AttemptOutcome, detail: str) -> list[RouteAttempt]:
        return [
            RouteAttempt(
                route=Route.PMC_CLOUD, url=None, outcome=outcome, detail=detail
            )
        ]

    try:
        pmcid = pmcid_for_doi(client, doi)
        keys = bucket_keys(client, f"{pmcid}.") if pmcid is not None else []
    except BlockedError as exc:
        return miss(AttemptOutcome.BLOCKED, str(exc)), None
    except httpx.HTTPError as exc:
        return miss(AttemptOutcome.HTTP_ERROR, f"{type(exc).__name__}: {exc}"), None
    if pmcid is None:
        return miss(
            AttemptOutcome.NOT_FOUND, "the NCBI ID converter has no PMCID"
        ), None
    if not keys:
        return miss(
            AttemptOutcome.NOT_FOUND,
            f"{pmcid} is not in the PMC Article Datasets bucket",
        ), None
    version = max({k.split("/", 1)[0] for k in keys}, key=_version_number)
    pdf_key = f"{version}/{version}.pdf"
    if pdf_key not in keys:
        held = sorted(k.split("/", 1)[1] for k in keys if k.startswith(f"{version}/"))
        return miss(
            AttemptOutcome.NOT_FOUND,
            f"{version} holds no article PDF (author manuscript or non-OA); "
            f"objects: {held}",
        ), None
    url = pmc_cloud_url(pdf_key)
    attempt, content = fetch_pdf(client, url, Route.PMC_CLOUD)
    if content is None:
        return [attempt], None
    retrieval = RetrievalRecord(
        method=RetrievalMethod.pmc_cloud,
        source_url=url,
        retriever=R_PMC_CLOUD,
        params={"key": pdf_key},
        sha256=_hashes(content)[0],
        retrieved_at=today,
    )
    return [attempt], _paper_file(content, Route.PMC_CLOUD, url, retrieval, doi)


def local_pdf(
    path: Path, source_url: str, role: Literal["paper", "si"], doi: str
) -> PdfFile:
    """A browser-downloaded file as a ``manual_browser`` retrieval; must be a PDF."""
    content = path.read_bytes()
    if not is_pdf_bytes(content):
        raise RefusedError(f"{path} is not a PDF")
    sha256, md5 = _hashes(content)
    retrieved_at = datetime.fromtimestamp(path.stat().st_mtime, UTC).date().isoformat()
    retrieval = RetrievalRecord(
        method=RetrievalMethod.manual_browser,
        source_url=source_url,
        retriever=R_MANUAL,
        params={
            "retrieval_command": f"manual browser download from {source_url}; the "
            f"downloaded file was handed to {TOOL} as {path.name}",
            "local_path": str(path),
        },
        sha256=sha256,
        retrieved_at=retrieved_at,
    )
    is_paper = role == "paper"
    return PdfFile(
        role=role,
        route=Route.LOCAL_FILE,
        filename=_doi_filename(doi) if is_paper else path.name,
        title=MAIN_TITLE if is_paper else f"{SI_TITLE_PREFIX}: {path.name}",
        origin=str(path),
        sha256=sha256,
        md5=md5,
        n_bytes=len(content),
        retrieval=retrieval,
        content=content,
    )


# -- writes ----------------------------------------------------------------------


def provenance_note(file: PdfFile, today: str) -> str:
    """The attachment's Zotero note: who added it and its retrieval record."""
    record = html.escape(file.retrieval.model_dump_json(indent=1))
    return (
        f"<p>Added by {TOOL} (torchcell) on {today}. Route: {file.route}; "
        f"origin: {html.escape(file.origin)}; sha256 {file.sha256}.</p>"
        f"<pre>{record}</pre>"
    )


def attachment_template(file: PdfFile, today: str) -> dict[str, Any]:
    """A stored-file attachment item for ``file``."""
    return {
        "itemType": "attachment",
        "linkMode": "imported_file",
        "title": file.title,
        "filename": file.filename,
        "contentType": "application/pdf",
        "charset": "",
        "note": provenance_note(file, today),
        "tags": [],
        "relations": {},
    }


def upload_file(
    zot: Any, parent_key: str, file: PdfFile, today: str, upload: UploadFn
) -> str:
    """Upload one file as a child of ``parent_key``; return the attachment key."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / file.filename).write_bytes(file.content)
        resp = upload(zot, [attachment_template(file, today)], parent_key, tmp)
    done = resp["success"] + resp["unchanged"]
    if resp["failure"] or len(done) != 1:
        raise RuntimeError(f"upload of {file.filename} failed: {resp}")
    return str(done[0]["key"])


def create_item(
    zot: Any, work: dict[str, Any], identity: PaperIdentity, collection_key: str
) -> str:
    """Create the ``journalArticle`` in ``collection_key``; return its key.

    The POST is not retried: a write retried after a timeout could create a duplicate.
    """
    template: dict[str, Any] = with_zotero_retry(
        lambda: zot.item_template("journalArticle")
    )
    template.update(journal_article_fields(work, identity))
    template["collections"] = [collection_key]
    resp = zot.create_items([template])
    if resp.get("failed") or "0" not in resp.get("successful", {}):
        raise RuntimeError(f"creating the item for {identity.doi} failed: {resp}")
    return str(resp["successful"]["0"]["key"])


def verify_stored(lib: ZoteroLibrary, item_key: str, files: list[AttachedFile]) -> None:
    """Every attachment written is a PDF child of the item holding the MD5 we sent."""
    stored = {a["key"]: a["data"].get("md5") for a in lib.pdf_attachments(item_key)}
    for attached in files:
        held = stored.get(str(attached.attachment_key))
        if held != attached.md5:
            raise RuntimeError(
                f"attachment {attached.attachment_key} ({attached.filename}) of "
                f"{item_key} stores md5 {held}; the bytes sent have md5 {attached.md5}"
            )


# -- the filing ------------------------------------------------------------------


def _attached(file: PdfFile, key: str | None) -> AttachedFile:
    return AttachedFile(
        attachment_key=key,
        role=file.role,
        filename=file.filename,
        route=file.route,
        method=file.retrieval.method,
        source_url=file.retrieval.source_url,
        origin=file.origin,
        sha256=file.sha256,
        md5=file.md5,
        n_bytes=file.n_bytes,
    )


def _existing_target(
    lib: ZoteroLibrary, request: AddPaperRequest, collection_key: str, root_key: str
) -> dict[str, Any] | None:
    """The item the filing writes to, or None when a new item is to be created.

    One item per paper: a DOI the group library already holds is refused unless
    ``--attach-to`` names that very item. The item must sit in the named collection
    or in the ``database`` root (where the owner files dataset papers too), so an
    attachment never lands on an item outside the database tree.
    """
    existing = lib.find_item_by_doi(request.doi)
    if request.attach_to is None:
        if existing is not None:
            raise RefusedError(
                f"{request.doi} is already item {existing['key']} (collections "
                f"{existing['data'].get('collections')}); pass --attach-to "
                f"{existing['key']} to attach a PDF to it, never a second item"
            )
        return None
    if existing is None:
        raise RefusedError(
            f"--attach-to {request.attach_to}: no item in the group library has DOI "
            f"{request.doi}"
        )
    if existing["key"] != request.attach_to:
        raise RefusedError(
            f"--attach-to {request.attach_to}: DOI {request.doi} belongs to item "
            f"{existing['key']}"
        )
    if not {collection_key, root_key} & set(existing["data"].get("collections") or []):
        raise RefusedError(
            f"item {existing['key']} is in neither {request.collection} "
            f"({collection_key}) nor the {DATABASE_COLLECTION} root ({root_key}); its "
            f"collections are {existing['data'].get('collections')}"
        )
    return existing


def add_paper(
    request: AddPaperRequest,
    lib: ZoteroLibrary,
    client: httpx.Client,
    *,
    upload: UploadFn = zupload,
    today: str | None = None,
) -> AddPaperReport:
    """File one paper (see the module docstring for policy and routes).

    Every check and every retrieval runs before the first write, so a refusal
    (:class:`RefusedError`) or a ``manual`` outcome leaves the library untouched.
    """
    stamp = today or datetime.now(UTC).date().isoformat()
    check_collection_path(request.collection)
    check_group_library(lib)
    work = crossref_work(client, request.doi)
    if work is None:
        raise RefusedError(f"Crossref does not know {request.doi}")
    identity = paper_identity(work)
    verify_identity(identity, request.doi, request.first_author, request.title_contains)
    collection_key, root_key = resolve_collection(lib, request.collection)
    target = _existing_target(lib, request, collection_key, root_key)

    report = AddPaperReport(
        doi=identity.doi,
        title=identity.title,
        first_author=identity.first_author,
        collection=request.collection,
        collection_key=collection_key,
        item_key=None if target is None else str(target["key"]),
        created_item=False,
        dry_run=request.dry_run,
        outcome=Outcome.MANUAL,
    )
    if target is not None and collection_key not in target["data"]["collections"]:
        report.notes.append(
            f"item {report.item_key} sits in the {DATABASE_COLLECTION} root, not in "
            f"{request.collection}; its collection membership is left as it is"
        )
    existing_pdfs = [] if target is None else lib.pdf_attachments(str(target["key"]))
    has_main = any(_is_main_article(a) for a in existing_pdfs)
    if has_main and not request.si:
        raise RefusedError(
            f"item {report.item_key} already has an article PDF; nothing to attach"
        )

    paper: PdfFile | None = None
    if not has_main:
        attempts, paper = publisher_route(client, work, stamp)
        report.attempts.extend(attempts)
        if paper is None:
            attempt, paper = raw_mirror_route(request.raw_root, identity.doi)
            report.attempts.append(attempt)
        if paper is None:
            attempts, paper = pmc_cloud_route(client, identity.doi, stamp)
            report.attempts.extend(attempts)
        if paper is None and request.pdf is not None and request.pdf_source_url:
            paper = local_pdf(
                request.pdf, request.pdf_source_url, "paper", identity.doi
            )
            report.attempts.append(
                RouteAttempt(
                    route=Route.LOCAL_FILE,
                    url=request.pdf_source_url,
                    outcome=AttemptOutcome.PDF,
                    detail=str(request.pdf),
                )
            )
        if paper is None:
            report.manual_url = f"https://doi.org/{identity.doi}"
            report.notes.append(
                "no route yielded the article PDF; nothing was written. Download it "
                f"in a browser from {report.manual_url} and re-run with --pdf"
            )
            if request.si:
                report.notes.append(
                    "SI not attached: the item would have no article PDF"
                )
            return report

    si_files: list[PdfFile] = []
    if request.si and request.si_source_url:
        si_files = [
            local_pdf(path, request.si_source_url, "si", identity.doi)
            for path in request.si
        ]
        report.notes.append(
            "SI PDFs attached: once lit_sync mirrors them, capture_si treats this "
            "key's SI as held from another path and skips the publisher's SI files"
        )
    files = ([paper] if paper is not None else []) + si_files
    held_md5 = {a["data"].get("md5"): a["key"] for a in existing_pdfs}
    for file in files:
        if file.md5 in held_md5:
            raise RefusedError(
                f"{file.filename} is already attached to {report.item_key} as "
                f"{held_md5[file.md5]} (same md5)"
            )

    if request.dry_run:
        report.outcome = Outcome.WOULD_FILE
        report.files = [_attached(f, None) for f in files]
        return report

    zot = lib.zot
    if target is None:
        item_key = create_item(zot, work, identity, collection_key)
        report.item_key = item_key
        report.created_item = True
    else:
        item_key = str(target["key"])
    if paper is not None:
        try:
            report.files.append(
                _attached(paper, upload_file(zot, item_key, paper, stamp, upload))
            )
        except Exception:
            # Roll back: an item without its PDF is the metadata-only item the policy
            # forbids. Only the item this run created is ever deleted.
            if report.created_item:
                zot.delete_item(zot.item(item_key))
            raise
    for file in si_files:
        report.files.append(
            _attached(file, upload_file(zot, item_key, file, stamp, upload))
        )
    verify_stored(lib, item_key, report.files)
    report.outcome = Outcome.FILED
    return report


def format_report(report: AddPaperReport) -> str:
    """Human-readable summary of a run."""
    lines = [
        f"{report.doi}  {report.first_author}  {report.title}",
        f"collection {report.collection} = {report.collection_key}",
    ]
    lines += [
        f"  {a.route:<10} {a.outcome:<10} {a.url or '-'}  ({a.detail})"
        for a in report.attempts
    ]
    item = report.item_key or "-"
    created = " (created)" if report.created_item else ""
    lines.append(f"outcome {report.outcome}: item {item}{created}")
    lines += [
        f"  {f.role:<5} {f.attachment_key or '(dry run)'}  {f.filename}  "
        f"sha256 {f.sha256}  {f.n_bytes} bytes  route {f.route}  method {f.method}"
        for f in report.files
    ]
    lines += [f"  note: {n}" for n in report.notes]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Command line; the caller loads ``.env``. Exit 2 refused, 3 manual, 0 filed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--doi", required=True)
    parser.add_argument(
        "--collection",
        required=True,
        help="database/<Organism>, e.g. database/Escherichia-coli",
    )
    parser.add_argument(
        "--first-author",
        required=True,
        help="Expected first-author family name; Crossref must agree.",
    )
    parser.add_argument("--title-contains", default=None)
    parser.add_argument(
        "--pdf", type=Path, default=None, help="Browser-downloaded PDF."
    )
    parser.add_argument("--pdf-source-url", default=None)
    parser.add_argument(
        "--si", type=Path, action="append", default=[], help="SI PDF; repeatable."
    )
    parser.add_argument(
        "--si-source-url", default=None, help="Page the --si files came from."
    )
    parser.add_argument("--attach-to", default=None, metavar="ITEM_KEY")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="torchcell-raw dir (default: $DATA_ROOT/torchcell-raw).",
    )
    parser.add_argument("--report", type=Path, default=None, help="JSON report path.")
    args = parser.parse_args(argv)
    if (args.pdf is None) != (args.pdf_source_url is None):
        parser.error("--pdf and --pdf-source-url go together")
    if bool(args.si) != (args.si_source_url is not None):
        parser.error("--si and --si-source-url go together")
    try:
        request = AddPaperRequest(
            doi=args.doi,
            collection=check_collection_path(args.collection),
            first_author=args.first_author,
            title_contains=args.title_contains,
            pdf=args.pdf,
            pdf_source_url=args.pdf_source_url,
            si=tuple(args.si),
            si_source_url=args.si_source_url,
            attach_to=args.attach_to,
            dry_run=args.dry_run,
            raw_root=args.raw_root or Path(os.environ["DATA_ROOT"]) / "torchcell-raw",
        )
        lib = ZoteroLibrary.from_env()
        with make_client() as client:
            report = add_paper(request, lib, client)
    except RefusedError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    print(format_report(report))
    if args.report is not None:
        args.report.write_text(report.model_dump_json(indent=2) + "\n")
    return EXIT_MANUAL if report.outcome == Outcome.MANUAL else 0
