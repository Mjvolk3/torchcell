# torchcell/literature/capture_si.py
# [[torchcell.literature.capture_si]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/capture_si.py
# Test file: tests/torchcell/literature/test_capture_si.py

r"""Capture publisher supplementary files into already-mirrored literature keys.

The nightly sync (:mod:`torchcell.literature.sync`) captures what Zotero holds, which
for most papers is the article PDF alone. The supplementary files that hold a paper's
data tables live with the publisher. This module resolves them from the key's DOI,
downloads each one into ``<key>/si/si<N>.<ext>`` (``N`` continues after any SI file
the key already has; the publisher's name is kept as ``original_filename``) and
appends an :class:`~torchcell.literature.manifest.ArtifactRecord` per file to the key's
``manifest.json``, carrying the sha256 and a
:class:`~torchcell.literature.manifest.RetrievalRecord` (retriever dotted path,
params, source URL, ``retrieved_at``) that re-runs the retrieval.

Routes, tried in order; the first one that yields files is used:

1. ``pmc_cloud``: the DOI's PMCID (NCBI ID converter), then the PMC Article Datasets
   bucket (``pmc-oa-opendata`` on AWS, the successor of the retired OA package
   service). The newest article version's JATS XML lists the supplementary files
   (``<supplementary-material>`` media); those present in the bucket are taken. Author
   manuscripts list their supplements but the bucket holds none of them, so they fall
   through to the publisher, and so does an article whose JATS lists no supplementary
   material at all: PMC4690272 lists none while its Springer page lists four files.
2. The publisher, chosen by Crossref member id:

   - Springer Nature, BMC, EMBO (member 297): ESM links on
     ``link.springer.com/article/<doi>``, fetched from ``static-content.springer.com``.
   - PLOS (340): the article JATS XML from ``journals.plos.org``; each file by its
     object DOI.
   - Elsevier and Cell Press (78): the PII from Crossref, then a HEAD probe of
     ``ars.els-cdn.com/.../1-s2.0-<PII>-mmc<N>.<ext>`` over :data:`ELSEVIER_EXTENSIONS`
     until two consecutive numbers have no file. ScienceDirect itself answers 403, so
     the probe is the only scriptable listing; a number with no probed extension
     between two found ones is reported as unretrieved.
   - eLife (4374): ``additionalFiles`` and figure ``sourceData`` from the eLife API.
   - ASM (235), ACS (316), AAAS (221): the Atypon ``/doi/suppl/<doi>`` page; Wiley
     (311): ``downloadSupplement`` links on the article page. All four answered HTTP
     403 to scripted requests when measured (2026-10-07); a block is reported as
     ``manual`` with the page URL, and the parsers run if the page ever loads.
   - OUP (286), Cold Spring Harbor / bioRxiv (246) and any other publisher: no
     scripted route; ``manual`` with the URL to open.

Outcomes (:class:`SiOutcome`): ``captured``, ``would_capture`` (dry run), ``partial``
(the route listed files it could not deliver, named in ``unretrieved``), ``present``
(every resolved file is already recorded, or the key holds SI from another path such
as a Zotero attachment), ``none_listed`` (an enumerating route lists no supplementary
file), ``manual`` (blocked or no route; ``manual_url`` is what a person opens),
``no_mirror_dir`` (the key has no ``manifest.json`` yet) and ``failed``.

Idempotent: a file whose retriever and params are already recorded in the manifest is
never downloaded again, and the manifest is rewritten after every file so an
interrupted run leaves no unrecorded file behind. ``scripts/lit_capture_si.py`` is the
command-line wrapper.

A supplement shipped inside a recorded archive is promoted to a file of its own with
:func:`store_zip_member` (``--zip-member si/si1.zip:Table_S1.pdf``): the member becomes
the next ``si/si<N>`` file with a ``zip_member`` retrieval over the archive's recorded
URL and sha256, so a loader can quote its OCR text.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import logging
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field

from torchcell.literature.backfill import library_root
from torchcell.literature.manifest import (
    MANIFEST_FILENAME,
    ROLE_SI_DATA,
    ROLE_SI_PDF,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
    _role_for,
    build_manifest,
    sha256_file,
    write_manifest,
)
from torchcell.literature.ocr import ocr_pdf
from torchcell.literature.retrieve import (
    PMC_CLOUD_BUCKET,
    RETRIEVERS,
    elsevier_mmc_url,
    plos_supplementary_url,
    pmc_cloud_url,
)
from torchcell.literature.sync import _collection_items
from torchcell.literature.zotero import ZoteroLibrary, _resolve_citation_key

log = logging.getLogger(__name__)

PMC_IDCONV = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
#: Discovery identifies itself as a script. Measured 2026-10-07: link.springer.com
#: answers a browser-like User-Agent from this client with a JavaScript challenge
#: page (HTTP 200) and serves the article to this one.
SI_UA = "torchcell-literature (+https://github.com/Mjvolk3/torchcell)"
CROSSREF_WORKS = "https://api.crossref.org/works/"
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
#: HTTP statuses a publisher uses to refuse a scripted client.
BLOCKED_STATUS = frozenset({401, 403, 429})
REPORTS_SUBDIR = "_sync_reports"

R_PMC_CLOUD = "torchcell.literature.retrieve.pmc_cloud_object"
R_SPRINGER = "torchcell.literature.retrieve.springer_esm"
R_PLOS = "torchcell.literature.retrieve.plos_supplementary"
R_ELSEVIER = "torchcell.literature.retrieve.elsevier_mmc"
R_DIRECT = "torchcell.literature.retrieve.direct_url"
R_ZIP_MEMBER = "torchcell.literature.retrieve.zip_member"

MEMBER_SPRINGER = "297"
MEMBER_PLOS = "340"
MEMBER_ELSEVIER = "78"
MEMBER_ELIFE = "4374"
MEMBER_WILEY = "311"
MEMBER_CSHL = "246"
#: Crossref member -> (publisher label, Atypon host serving ``/doi/suppl/<doi>``).
ATYPON_HOSTS: dict[str, tuple[str, str]] = {
    "235": ("ASM", "https://journals.asm.org"),
    "316": ("ACS", "https://pubs.acs.org"),
    "221": ("AAAS", "https://www.science.org"),
}
#: PLOS DOI journal code (``10.1371/journal.<code>.<n>``) -> journals.plos.org slug.
PLOS_JOURNALS: dict[str, str] = {
    "pone": "plosone",
    "pbio": "plosbiology",
    "pgen": "plosgenetics",
    "pcbi": "ploscompbiol",
    "ppat": "plospathogens",
    "pmed": "plosmedicine",
    "pntd": "plosntds",
}
#: Extensions probed per Elsevier ``mmc<N>``, most frequent first (stop at first hit).
ELSEVIER_EXTENSIONS: tuple[str, ...] = tuple(
    "pdf xlsx docx xls doc csv txt zip pptx ppt mp4 mov avi wmv mpg tif tiff jpg png "
    "gif eps svg gz tar rar 7z fasta fa gb gbk json xml html htm tsv dat py r m mat "
    "sbml mp3 wav".split()
)
#: Consecutive ``mmc<N>`` numbers with no file that end the Elsevier probe.
ELSEVIER_STOP_AFTER_MISSES = 2
_ESM_LINK = re.compile(
    r"https?://(?:media\.springernature\.com/[^\"\s]*?/springer-static"
    r"|static-content\.springer\.com)/esm/(art%3A[^\"/\s]+)/MediaObjects/"
    r"([^\"?#\s]+)"
)
_PII = re.compile(r"S[0-9]{4}[0-9X]{4}[0-9]{2}[0-9]{5}[0-9X]")
_SI_INDEX = re.compile(r"si(\d+)(?:[._].*)?")


class SiOutcome(StrEnum):
    """Per-key outcome of a supplementary-file capture pass."""

    CAPTURED = "captured"
    WOULD_CAPTURE = "would_capture"
    PARTIAL = "partial"
    PRESENT = "present"
    NONE_LISTED = "none_listed"
    MANUAL = "manual"
    NO_MIRROR_DIR = "no_mirror_dir"
    FAILED = "failed"


class RouteStatus(StrEnum):
    """What one resolution route found for a DOI."""

    FILES = "files"  # one or more retrievable supplementary files
    NONE_LISTED = "none_listed"  # the route lists the article and no SI file
    NOT_FOUND = "not_found"  # the route has no retrievable copy of the article's SI
    MANUAL = "manual"  # blocked, or no scripted route: a person opens manual_url


class SiCandidate(BaseModel):
    """One supplementary file a route resolved, with the recipe that fetches it."""

    model_config = ConfigDict(extra="forbid")

    original_filename: str
    source_url: str
    method: RetrievalMethod
    retriever: str = Field(description="Registry key into retrieve.RETRIEVERS.")
    params: dict[str, Any]


class RouteResult(BaseModel):
    """Outcome of one route for one DOI."""

    route: str
    status: RouteStatus
    candidates: list[SiCandidate] = Field(default_factory=list)
    unretrieved: list[str] = Field(
        default_factory=list,
        description="Files the route lists but cannot deliver here.",
    )
    manual_url: str | None = None
    detail: str | None = None


class Resolution(BaseModel):
    """Every route tried for a DOI, in order, and the one whose files are used."""

    doi: str
    pmcid: str | None = None
    publisher: str | None = None
    routes: list[RouteResult] = Field(default_factory=list)
    chosen: RouteResult | None = None


class SiFile(BaseModel):
    """A supplementary file captured (or, in a dry run, planned) for a key."""

    path: str
    original_filename: str
    source_url: str
    retriever: str
    bytes: int | None = None
    sha256: str | None = None


class KeySiResult(BaseModel):
    """Per-key outcome written to the JSON report."""

    citation_key: str | None
    doi: str | None
    outcome: SiOutcome
    route: str | None = None
    publisher: str | None = None
    pmcid: str | None = None
    routes_tried: list[str] = Field(
        default_factory=list, description="``<route>:<status>`` in the order tried."
    )
    files: list[SiFile] = Field(default_factory=list)
    already_recorded: int = 0
    unretrieved: list[str] = Field(default_factory=list)
    manual_url: str | None = None
    detail: str | None = None
    ocr: list[str] = Field(
        default_factory=list, description="Markdown written by --ocr, per SI PDF."
    )
    error: str | None = None


class SiCaptureReport(BaseModel):
    """One capture pass over a set of keys."""

    generated_at: str
    mirror_root: str
    dry_run: bool
    results: list[KeySiResult]

    def tally(self) -> dict[str, int]:
        """``outcome -> count`` over the results, sorted by outcome."""
        counts: dict[str, int] = {}
        for result in self.results:
            counts[result.outcome] = counts.get(result.outcome, 0) + 1
        return dict(sorted(counts.items()))

    def summary(self) -> str:
        """One-line ``outcome=count`` tally for logs."""
        tally = " ".join(f"{k}={v}" for k, v in self.tally().items())
        return f"{len(self.results)} keys | {tally}"


class KeyRequest(BaseModel):
    """A key to process: a citation key, a DOI, or both."""

    citation_key: str | None
    doi: str | None = None


class BlockedError(RuntimeError):
    """A host refused a scripted request (401/403/429, or a challenge page)."""

    def __init__(self, url: str, status: int) -> None:
        """Record the refused URL and the status it answered with."""
        super().__init__(f"HTTP {status} from {url}")
        self.url = url
        self.status = status


def _is_blocked(resp: httpx.Response) -> bool:
    """A refusal of the client rather than an answer about the resource."""
    if resp.status_code in BLOCKED_STATUS:
        return True
    return resp.status_code == 503 and "cf-mitigated" in resp.headers


def _fetch(client: httpx.Client, url: str) -> httpx.Response:
    """GET ``url``; raise :class:`BlockedError` on a block, ``HTTPStatusError`` else."""
    resp = client.get(url)
    if _is_blocked(resp):
        raise BlockedError(url, resp.status_code)
    resp.raise_for_status()
    return resp


def make_client() -> httpx.Client:
    """The discovery client: redirects followed, cookies kept for the run."""
    return httpx.Client(
        follow_redirects=True, timeout=120.0, headers={"User-Agent": SI_UA}
    )


# -- PMC Article Datasets -----------------------------------------------------


def pmcid_for_doi(client: httpx.Client, doi: str) -> str | None:
    """The DOI's PMCID from the NCBI ID converter, or None when PMC has no copy."""
    url = f"{PMC_IDCONV}?ids={quote(doi, safe='')}&format=json&tool=torchcell"
    records = _fetch(client, url).json()["records"]
    pmcid = records[0].get("pmcid") if records else None
    return str(pmcid) if pmcid else None


def bucket_keys(client: httpx.Client, prefix: str) -> list[str]:
    """Every key under ``prefix`` in the PMC bucket, following continuation tokens."""
    keys: list[str] = []
    token: str | None = None
    while True:
        url = f"{PMC_CLOUD_BUCKET}/?list-type=2&prefix={quote(prefix, safe='')}"
        if token is not None:
            url += f"&continuation-token={quote(token, safe='')}"
        root = ET.fromstring(_fetch(client, url).content)
        keys.extend(
            el.text for el in root.iterfind("s3:Contents/s3:Key", _S3_NS) if el.text
        )
        if root.findtext("s3:IsTruncated", namespaces=_S3_NS) != "true":
            return keys
        token = root.findtext("s3:NextContinuationToken", namespaces=_S3_NS)


def _version_number(version: str) -> int:
    """``PMC123.4`` -> 4."""
    return int(version.rsplit(".", 1)[1])


def jats_supplementary_hrefs(xml_bytes: bytes) -> list[str]:
    """File names a JATS article lists as supplementary material, in document order.

    Taken from each ``<supplementary-material>`` (its own ``xlink:href`` and those of
    its ``<media>`` / ``<inline-supplementary-material>`` descendants) and from
    ``<media>`` inside a ``<sec sec-type="supplementary-material">``. Other links in
    the block (``<ext-link>`` to a license or a GEO accession) are not files.
    """
    root = ET.fromstring(xml_bytes)
    hrefs: list[str] = []

    def add(href: str | None) -> None:
        if href and href not in hrefs:
            hrefs.append(href)

    for block in root.iter("supplementary-material"):
        add(block.get(XLINK_HREF))
        for el in block.iter():
            if el is not block and el.tag in ("media", "inline-supplementary-material"):
                add(el.get(XLINK_HREF))
    for sec in root.iter("sec"):
        if sec.get("sec-type") == "supplementary-material":
            for el in sec.iter("media"):
                add(el.get(XLINK_HREF))
    return hrefs


def route_pmc_cloud(client: httpx.Client, pmcid: str | None) -> RouteResult:
    """Supplementary files of the newest PMC article version present in the bucket."""
    route = "pmc_cloud"
    if pmcid is None:
        return RouteResult(
            route=route, status=RouteStatus.NOT_FOUND, detail="DOI has no PMCID"
        )
    keys = bucket_keys(client, f"{pmcid}.")
    if not keys:
        return RouteResult(
            route=route,
            status=RouteStatus.NOT_FOUND,
            detail=f"{pmcid} is not in the PMC Article Datasets bucket",
        )
    version = max({k.split("/", 1)[0] for k in keys}, key=_version_number)
    hosted = {k.split("/", 1)[1] for k in keys if k.startswith(f"{version}/")}
    xml_name = f"{version}.xml"
    if xml_name not in hosted:
        raise ValueError(f"{version} has no {xml_name} in the PMC bucket")
    listed = jats_supplementary_hrefs(
        _fetch(client, pmc_cloud_url(f"{version}/{xml_name}")).content
    )
    if not listed:
        return RouteResult(
            route=route,
            status=RouteStatus.NONE_LISTED,
            detail=f"{version} JATS lists no supplementary material",
        )
    candidates = [
        SiCandidate(
            original_filename=href,
            source_url=pmc_cloud_url(f"{version}/{href}"),
            method=RetrievalMethod.pmc_cloud,
            retriever=R_PMC_CLOUD,
            params={"key": f"{version}/{href}"},
        )
        for href in listed
        if href in hosted
    ]
    unretrieved = [href for href in listed if href not in hosted]
    if not candidates:
        return RouteResult(
            route=route,
            status=RouteStatus.NOT_FOUND,
            unretrieved=unretrieved,
            detail=f"{version} lists {len(listed)} supplementary files and the bucket "
            "holds none of them (author manuscript or non-OA)",
        )
    return RouteResult(
        route=route,
        status=RouteStatus.FILES,
        candidates=candidates,
        unretrieved=unretrieved,
        detail=f"{version}: {len(candidates)} of {len(listed)} listed files hosted",
    )


# -- publishers -----------------------------------------------------------------


def crossref_work(client: httpx.Client, doi: str) -> dict[str, Any] | None:
    """Crossref's ``message`` for a DOI, or None when Crossref does not know it."""
    resp = client.get(CROSSREF_WORKS + quote(doi, safe="/"))
    if resp.status_code == 404:
        return None
    if _is_blocked(resp):
        raise BlockedError(str(resp.request.url), resp.status_code)
    resp.raise_for_status()
    message: dict[str, Any] = resp.json()["message"]
    return message


def _manual(route: str, url: str, detail: str) -> RouteResult:
    return RouteResult(
        route=route, status=RouteStatus.MANUAL, manual_url=url, detail=detail
    )


def _require_doi_in_page(text: str, doi: str, url: str) -> None:
    """Treat a page that never names the DOI as a block (a challenge page).

    Bot challenges answer HTTP 200 with a stub page, so the status alone cannot tell
    them from the article; an article page always carries its DOI.
    """
    if doi.lower() not in text.lower():
        raise BlockedError(url, 200)


def route_springer(client: httpx.Client, doi: str) -> RouteResult:
    """ESM links on the link.springer.com article page (Nature, BMC, EMBO too)."""
    page = f"https://link.springer.com/article/{doi}"
    try:
        text = _fetch(client, page).text
        _require_doi_in_page(text, doi, page)
    except BlockedError as exc:
        return _manual("springer", page, str(exc))
    seen: list[tuple[str, str]] = []
    for art, name in _ESM_LINK.findall(text):
        if (art, name) not in seen:
            seen.append((art, name))
    if not seen:
        return RouteResult(
            route="springer",
            status=RouteStatus.NONE_LISTED,
            detail="article page lists no ESM file",
        )
    candidates = []
    for art, name in seen:
        url = f"https://static-content.springer.com/esm/{art}/MediaObjects/{name}"
        candidates.append(
            SiCandidate(
                original_filename=unquote(name),
                source_url=url,
                method=RetrievalMethod.springer_esm,
                retriever=R_SPRINGER,
                params={"url": url},
            )
        )
    return RouteResult(
        route="springer", status=RouteStatus.FILES, candidates=candidates
    )


def plos_filename(client: httpx.Client, url: str) -> str:
    """The stored file name of a PLOS supplementary object, from its redirect.

    The article-file URL answers 302 to a signed storage URL whose path ends in the
    real file name (``pgen.1004120.s030.xlsx``). The XML ``mimetype`` cannot name it:
    PLOS writes values such as ``image`` and ``application/x-excel`` there.
    """
    resp = client.head(url, follow_redirects=False)
    if _is_blocked(resp):
        raise BlockedError(url, resp.status_code)
    if not resp.is_redirect:
        raise ValueError(f"{url} answered {resp.status_code}, expected a redirect")
    return unquote(urlsplit(resp.headers["location"]).path.rsplit("/", 1)[1])


def route_plos(client: httpx.Client, doi: str) -> RouteResult:
    """Supplementary objects listed in the PLOS article XML."""
    match = re.fullmatch(r"10\.1371/journal\.([a-z]+)\.\d+", doi, flags=re.IGNORECASE)
    if match is None or match.group(1).lower() not in PLOS_JOURNALS:
        return _manual("plos", f"https://doi.org/{doi}", "unrecognized PLOS journal")
    journal = PLOS_JOURNALS[match.group(1).lower()]
    xml_url = (
        f"https://journals.plos.org/{journal}/article/file?id={doi}&type=manuscript"
    )
    try:
        root = ET.fromstring(_fetch(client, xml_url).content)
    except BlockedError as exc:
        return _manual("plos", f"https://doi.org/{doi}", str(exc))
    candidates = []
    try:
        for block in root.iter("supplementary-material"):
            href = block.get(XLINK_HREF) or ""
            object_doi = href.removeprefix("info:doi/")
            if not object_doi.startswith("10.1371/"):
                raise ValueError(f"PLOS supplementary href {href!r} is not a DOI")
            url = plos_supplementary_url(journal, object_doi)
            candidates.append(
                SiCandidate(
                    original_filename=plos_filename(client, url),
                    source_url=url,
                    method=RetrievalMethod.direct_url,
                    retriever=R_PLOS,
                    params={"journal": journal, "object_doi": object_doi},
                )
            )
    except BlockedError as exc:
        return _manual("plos", f"https://doi.org/{doi}", str(exc))
    if not candidates:
        return RouteResult(
            route="plos",
            status=RouteStatus.NONE_LISTED,
            detail="article XML lists no supplementary material",
        )
    return RouteResult(route="plos", status=RouteStatus.FILES, candidates=candidates)


def elsevier_pii(work: dict[str, Any]) -> str | None:
    """The unpunctuated PII among Crossref's ``alternative-id`` values."""
    for alt in work.get("alternative-id", []):
        compact = re.sub(r"[^0-9A-Z]", "", str(alt).upper())
        if _PII.fullmatch(compact):
            return compact
    return None


def _head_exists(client: httpx.Client, url: str) -> bool:
    """True on 200, False on 404; raise on a block or any other status."""
    resp = client.head(url)
    if _is_blocked(resp):
        raise BlockedError(url, resp.status_code)
    if resp.status_code == 404:
        return False
    resp.raise_for_status()
    return True


def route_elsevier(client: httpx.Client, doi: str, work: dict[str, Any]) -> RouteResult:
    """Probe the Elsevier CDN for ``mmc1``, ``mmc2``, ... (see module docstring)."""
    pii = elsevier_pii(work)
    if pii is None:
        return _manual("elsevier", f"https://doi.org/{doi}", "no PII in Crossref")
    article = f"https://www.sciencedirect.com/science/article/pii/{pii}"
    candidates: list[SiCandidate] = []
    pending_misses: list[str] = []
    holes: list[str] = []
    number = 0
    try:
        while len(pending_misses) < ELSEVIER_STOP_AFTER_MISSES:
            number += 1
            hit = next(
                (
                    f"mmc{number}.{ext}"
                    for ext in ELSEVIER_EXTENSIONS
                    if _head_exists(client, elsevier_mmc_url(pii, f"mmc{number}.{ext}"))
                ),
                None,
            )
            if hit is None:
                pending_misses.append(f"mmc{number}")
                continue
            holes.extend(pending_misses)
            pending_misses = []
            candidates.append(
                SiCandidate(
                    original_filename=hit,
                    source_url=elsevier_mmc_url(pii, hit),
                    method=RetrievalMethod.direct_url,
                    retriever=R_ELSEVIER,
                    params={"pii": pii, "filename": hit},
                )
            )
    except BlockedError as exc:
        return _manual("elsevier", article, str(exc))
    if not candidates:
        return _manual(
            "elsevier",
            article,
            f"no mmc1 under any of {len(ELSEVIER_EXTENSIONS)} probed extensions; "
            "the probe cannot tell 'no SI' from an unprobed extension",
        )
    unretrieved = [f"{hole} (no file under a probed extension)" for hole in holes]
    return RouteResult(
        route="elsevier",
        status=RouteStatus.FILES,
        candidates=candidates,
        unretrieved=unretrieved,
        manual_url=article if unretrieved else None,
        detail=f"PII {pii}: probe found {len(candidates)} mmc files",
    )


def _elife_files(node: Any, found: list[dict[str, Any]]) -> None:
    """Collect every ``additionalFiles`` / ``sourceData`` entry, depth first."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("additionalFiles", "sourceData") and isinstance(value, list):
                found.extend(v for v in value if isinstance(v, dict) and "uri" in v)
            _elife_files(value, found)
    elif isinstance(node, list):
        for value in node:
            _elife_files(value, found)


def route_elife(client: httpx.Client, doi: str) -> RouteResult:
    """``additionalFiles`` and figure ``sourceData`` from the eLife article API."""
    match = re.fullmatch(r"10\.7554/elife\.(\d+)(?:\.\d+)?", doi, flags=re.IGNORECASE)
    if match is None:
        return _manual("elife", f"https://doi.org/{doi}", "unrecognized eLife DOI")
    api = f"https://api.elifesciences.org/articles/{match.group(1)}"
    try:
        article = _fetch(client, api).json()
    except BlockedError as exc:
        return _manual("elife", f"https://doi.org/{doi}", str(exc))
    entries: list[dict[str, Any]] = []
    _elife_files(article, entries)
    candidates: list[SiCandidate] = []
    for entry in entries:
        uri = str(entry["uri"])
        if any(c.source_url == uri for c in candidates):
            continue
        candidates.append(
            SiCandidate(
                original_filename=str(entry.get("filename") or uri.rsplit("/", 1)[1]),
                source_url=uri,
                method=RetrievalMethod.direct_url,
                retriever=R_DIRECT,
                params={"url": uri},
            )
        )
    if not candidates:
        return RouteResult(
            route="elife",
            status=RouteStatus.NONE_LISTED,
            detail="article API lists no additional file or source data",
        )
    return RouteResult(route="elife", status=RouteStatus.FILES, candidates=candidates)


def _direct_candidates(urls: list[str]) -> list[SiCandidate]:
    """One ``direct_url`` candidate per URL, named by its last path segment."""
    return [
        SiCandidate(
            original_filename=unquote(urlsplit(url).path.rsplit("/", 1)[1]),
            source_url=url,
            method=RetrievalMethod.direct_url,
            retriever=R_DIRECT,
            params={"url": url},
        )
        for url in urls
    ]


def route_atypon(client: httpx.Client, doi: str, member: str) -> RouteResult:
    """``suppl_file`` links on an Atypon ``/doi/suppl/<doi>`` page (ASM, ACS, AAAS)."""
    label, host = ATYPON_HOSTS[member]
    route = label.lower()
    page = f"{host}/doi/suppl/{doi}"
    try:
        text = _fetch(client, page).text
        _require_doi_in_page(text, doi, page)
    except BlockedError as exc:
        return _manual(route, page, str(exc))
    pattern = re.compile(
        r'href="(/doi/suppl/' + re.escape(doi) + r'/suppl_file/[^"]+)"',
        flags=re.IGNORECASE,
    )
    urls: list[str] = []
    for path in pattern.findall(text):
        url = host + html.unescape(path)
        if url not in urls:
            urls.append(url)
    if not urls:
        return RouteResult(
            route=route,
            status=RouteStatus.NONE_LISTED,
            detail="supplementary page lists no file",
        )
    return RouteResult(
        route=route, status=RouteStatus.FILES, candidates=_direct_candidates(urls)
    )


def route_wiley(client: httpx.Client, doi: str) -> RouteResult:
    """``downloadSupplement`` links on the Wiley Online Library article page."""
    host = "https://onlinelibrary.wiley.com"
    page = f"{host}/doi/full/{doi}"
    try:
        text = _fetch(client, page).text
        _require_doi_in_page(text, doi, page)
    except BlockedError as exc:
        return _manual("wiley", page, str(exc))
    candidates: list[SiCandidate] = []
    for raw in re.findall(r'href="(/action/downloadSupplement\?[^"]+)"', text):
        path = html.unescape(raw)
        url = host + path
        if any(c.source_url == url for c in candidates):
            continue
        name = parse_qs(urlsplit(path).query)["file"][0]
        candidates.append(
            SiCandidate(
                original_filename=name,
                source_url=url,
                method=RetrievalMethod.direct_url,
                retriever=R_DIRECT,
                params={"url": url},
            )
        )
    if not candidates:
        return RouteResult(
            route="wiley",
            status=RouteStatus.NONE_LISTED,
            detail="article page lists no supporting information",
        )
    return RouteResult(route="wiley", status=RouteStatus.FILES, candidates=candidates)


def route_publisher(
    client: httpx.Client, doi: str, work: dict[str, Any] | None
) -> RouteResult:
    """Dispatch on the Crossref member id; publishers with no route are manual."""
    if work is None:
        return _manual("crossref", f"https://doi.org/{doi}", "DOI not in Crossref")
    member = str(work.get("member", ""))
    if member == MEMBER_SPRINGER:
        return route_springer(client, doi)
    if member == MEMBER_PLOS:
        return route_plos(client, doi)
    if member == MEMBER_ELSEVIER:
        return route_elsevier(client, doi, work)
    if member == MEMBER_ELIFE:
        return route_elife(client, doi)
    if member in ATYPON_HOSTS:
        return route_atypon(client, doi, member)
    if member == MEMBER_WILEY:
        return route_wiley(client, doi)
    publisher = str(work.get("publisher") or f"member {member}")
    if member == MEMBER_CSHL and work.get("type") == "posted-content":
        return _manual(
            "biorxiv",
            f"https://www.biorxiv.org/content/{doi}.supplementary-material",
            "bioRxiv pages refuse scripted clients (HTTP 429/403); no scripted route",
        )
    return _manual(
        "publisher", f"https://doi.org/{doi}", f"no scripted route for {publisher}"
    )


def resolve_si(client: httpx.Client, doi: str) -> Resolution:
    """Try PMC, then the publisher; pick the route whose files are used.

    A complete PMC answer (every listed file hosted) ends the search. Otherwise the
    publisher route runs; its files win, and a partial PMC answer is used only when
    the publisher yields none.
    """
    pmcid = pmcid_for_doi(client, doi)
    pmc = route_pmc_cloud(client, pmcid)
    resolution = Resolution(doi=doi, pmcid=pmcid, routes=[pmc])
    if pmc.status == RouteStatus.FILES and not pmc.unretrieved:
        resolution.chosen = pmc
        return resolution
    work = crossref_work(client, doi)
    resolution.publisher = str(work.get("publisher")) if work else None
    publisher = route_publisher(client, doi, work)
    resolution.routes.append(publisher)
    if publisher.status == RouteStatus.FILES:
        resolution.chosen = publisher
    elif pmc.status == RouteStatus.FILES:
        resolution.chosen = pmc
    return resolution


# -- writing into a key -----------------------------------------------------------


def _extension(name: str) -> str:
    """Lower-case extension, keeping ``.tar.gz``-style compound suffixes."""
    lower = name.lower()
    for compound in (".tar.gz", ".tar.bz2", ".tar.xz"):
        if lower.endswith(compound):
            return compound
    return Path(lower).suffix


def next_si_index(key_dir: Path, manifest: Manifest) -> int:
    """One past the largest ``si/si<N>`` index on disk or in the manifest."""
    used = [0]
    for record in manifest.files:
        if record.path.startswith("si/") and record.path.count("/") == 1:
            match = _SI_INDEX.fullmatch(record.path[3:])
            if match:
                used.append(int(match.group(1)))
    si_dir = key_dir / "si"
    if si_dir.is_dir():
        for path in si_dir.iterdir():
            match = _SI_INDEX.fullmatch(path.name)
            if match:
                used.append(int(match.group(1)))
    return max(used) + 1


def _is_recorded(manifest: Manifest, candidate: SiCandidate) -> bool:
    """A record already holds the file this candidate's recipe retrieves."""
    return any(
        r.retrieval is not None
        and r.retrieval.retriever == candidate.retriever
        and r.retrieval.params == candidate.params
        for r in manifest.files
    )


def foreign_si(manifest: Manifest) -> list[str]:
    """SI files the key holds that this module did not capture.

    ``capture_si`` sets ``original_filename`` on every file it writes, so an SI PDF or
    data file without it came from another path (a Zotero attachment, Dryad, a hand
    retrieval).
    """
    return [
        r.path
        for r in manifest.files
        if r.role in (ROLE_SI_PDF, ROLE_SI_DATA) and r.original_filename is None
    ]


def _plan(candidates: list[SiCandidate], start: int) -> list[tuple[SiCandidate, str]]:
    """Assign ``si/si<N><ext>`` paths from ``start`` in candidate order."""
    return [
        (cand, f"si/si{start + i}{_extension(cand.original_filename)}")
        for i, cand in enumerate(candidates)
    ]


def _store(
    key_dir: Path,
    manifest: Manifest,
    cand: SiCandidate,
    rel: str,
    data: bytes,
    retrieved_at: str,
) -> SiFile:
    """Write one retrieved file at ``rel``, record it and rewrite the manifest.

    The bytes land under a hidden ``.part`` name and are renamed into place, so a
    killed run never leaves a half-written ``si<N>`` file; the manifest is rewritten
    right after, so no written file is ever unrecorded for longer than one write.
    """
    dest = key_dir / rel
    if dest.exists():
        raise FileExistsError(f"{dest} exists and is not in the manifest")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(f".{dest.name}.part")
    part.write_bytes(data)
    part.rename(dest)
    digest = sha256_file(dest)
    manifest.files.append(
        ArtifactRecord(
            path=rel,
            role=_role_for(rel),
            bytes=len(data),
            sha256=digest,
            source=cand.source_url,
            original_filename=cand.original_filename,
            retrieval=RetrievalRecord(
                method=cand.method,
                source_url=cand.source_url,
                retriever=cand.retriever,
                params=cand.params,
                sha256=digest,
                retrieved_at=retrieved_at,
            ),
        )
    )
    if cand.source_url not in manifest.si_data_sources:
        manifest.si_data_sources.append(cand.source_url)
    write_manifest(key_dir, manifest)
    return SiFile(
        path=rel,
        original_filename=cand.original_filename,
        source_url=cand.source_url,
        retriever=cand.retriever,
        bytes=len(data),
        sha256=digest,
    )


def _record_ocr_outputs(key_dir: Path, manifest: Manifest, stems: list[str]) -> None:
    """Append records for the OCR outputs of ``si/<stem>.pdf`` not yet recorded.

    Roles, the ``mineru-ocr`` source and the processing record come from
    :func:`build_manifest`'s scan, so they match what a full rebuild would write.
    """
    recorded = {r.path for r in manifest.files}
    prefixes = tuple(f"si/{stem}" for stem in stems) + tuple(
        f"si/images/{stem}/" for stem in stems
    )
    scan = build_manifest(key_dir, citation_key=manifest.citation_key)
    for record in scan.files:
        if record.path in recorded:
            continue
        if record.path.startswith(prefixes):
            manifest.files.append(record)


def store_zip_member(
    key_dir: Path, manifest: Manifest, zip_rel: str, member: str, *, now: str
) -> SiFile:
    """Store one member of a recorded SI zip as the key's next ``si/si<N>.<ext>``.

    A publisher often ships every supplement in one archive (``si/si1.zip``), while a
    loader's quote audit needs the one document it quotes as a file of its own, with
    an OCR text beside it. The member is read out of the key's own stored zip, whose
    bytes are first checked against the sha256 the manifest records for it, so a
    corrupted or swapped archive raises instead of yielding a different member.

    The new record's retrieval re-runs from source: retriever
    ``torchcell.literature.retrieve.zip_member`` with the zip's recorded
    ``source_url`` and sha256 as ``url`` and ``container_sha256``, and the zip's own
    retrieval ``method`` (the method names the host the bytes come from; the zip is
    the container, not a different host). ``original_filename`` is the member name.

    Idempotent: when a record with the same retriever and params exists, it is
    returned and nothing is written.

    Args:
        key_dir: The mirror key directory.
        manifest: The key's manifest, rewritten in place on a store.
        zip_rel: Key-relative path of the recorded zip (``si/si1.zip``).
        member: Member name inside the zip (``Table_S1.pdf``).
        now: ISO timestamp recorded as ``retrieved_at``.

    Raises:
        KeyError: ``zip_rel`` is not recorded in the manifest, or ``member`` is not in
            the zip.
        ValueError: The zip record has no retrieval, or the stored zip's sha256 does
            not match its record.
    """
    container = next((r for r in manifest.files if r.path == zip_rel), None)
    if container is None:
        raise KeyError(f"{zip_rel} is not recorded in {key_dir / MANIFEST_FILENAME}")
    if container.retrieval is None or container.retrieval.source_url is None:
        raise ValueError(f"{zip_rel} has no recorded retrieval source_url")
    zip_bytes = (key_dir / zip_rel).read_bytes()
    got = hashlib.sha256(zip_bytes).hexdigest()
    if got != container.sha256:
        raise ValueError(
            f"{key_dir / zip_rel} sha256 {got} does not match its record "
            f"{container.sha256}"
        )
    url = container.retrieval.source_url
    cand = SiCandidate(
        original_filename=member,
        source_url=url,
        method=container.retrieval.method,
        retriever=R_ZIP_MEMBER,
        params={"url": url, "member": member, "container_sha256": container.sha256},
    )
    existing = next(
        (
            r
            for r in manifest.files
            if r.retrieval is not None
            and r.retrieval.retriever == cand.retriever
            and r.retrieval.params == cand.params
        ),
        None,
    )
    if existing is not None:
        return SiFile(
            path=existing.path,
            original_filename=member,
            source_url=url,
            retriever=cand.retriever,
            bytes=existing.bytes,
            sha256=existing.sha256,
        )
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        data = archive.read(member)
    rel = f"si/si{next_si_index(key_dir, manifest)}{_extension(member)}"
    return _store(key_dir, manifest, cand, rel, data, now)


def ocr_stored_pdfs(
    key_dir: Path, manifest: Manifest, rels: Sequence[str]
) -> list[str]:
    """OCR stored ``si/*.pdf`` files with :func:`ocr_pdf` and record the outputs.

    The device follows :func:`ocr_pdf` (``$MINERU_DEVICE_MODE``). Returns the
    key-relative markdown paths, in ``rels`` order.
    """
    pdfs = [key_dir / rel for rel in rels]
    written = [str(ocr_pdf(pdf).relative_to(key_dir)) for pdf in pdfs]
    _record_ocr_outputs(key_dir, manifest, [p.stem for p in pdfs])
    write_manifest(key_dir, manifest)
    return written


def capture_key(
    client: httpx.Client,
    root: Path,
    request: KeyRequest,
    *,
    dry_run: bool = False,
    do_ocr: bool = False,
    now: str | None = None,
) -> KeySiResult:
    """Resolve and capture the supplementary files of one mirrored key.

    Args:
        client: Discovery client (see :func:`make_client`).
        root: The ``torchcell-library`` directory.
        request: Citation key (and DOI, when the caller knows it).
        dry_run: Resolve and plan; download and write nothing.
        do_ocr: OCR each newly captured SI PDF with :func:`ocr_pdf` and record it.
        now: ISO timestamp for ``retrieved_at``; defaults to now (UTC).
    """
    key = request.citation_key
    if key is None or not (root / key / MANIFEST_FILENAME).is_file():
        detail = (
            "no mirror directory with this DOI"
            if key is None
            else f"{key}/{MANIFEST_FILENAME} absent (paper not captured yet)"
        )
        return KeySiResult(
            citation_key=key,
            doi=request.doi,
            outcome=SiOutcome.NO_MIRROR_DIR,
            detail=detail,
        )
    key_dir = root / key
    manifest = Manifest.model_validate_json((key_dir / MANIFEST_FILENAME).read_text())
    doi = request.doi or manifest.doi
    result = KeySiResult(citation_key=key, doi=doi, outcome=SiOutcome.FAILED)
    if doi is None:
        result.error = "no DOI in the request or the manifest"
        return result
    other = foreign_si(manifest)
    if other:
        result.outcome = SiOutcome.PRESENT
        result.detail = f"holds {len(other)} SI files from another path: {other[:3]}"
        return result
    try:
        resolution = resolve_si(client, doi)
    except Exception as exc:  # noqa: BLE001 -- report the key, keep the batch going
        log.exception("capture_si: resolving %s (%s) failed", key, doi)
        result.error = f"{type(exc).__name__}: {exc}"
        return result
    result.pmcid = resolution.pmcid
    result.publisher = resolution.publisher
    result.routes_tried = [f"{r.route}:{r.status}" for r in resolution.routes]
    chosen = resolution.chosen
    if chosen is None:
        last = resolution.routes[-1]
        pmc = resolution.routes[0]
        if last.status == RouteStatus.NONE_LISTED and not pmc.unretrieved:
            result.outcome = SiOutcome.NONE_LISTED
            result.route = last.route
            result.detail = last.detail
            return result
        result.outcome = SiOutcome.MANUAL
        result.route = last.route
        result.manual_url = last.manual_url or f"https://doi.org/{doi}"
        details = [r.detail for r in resolution.routes if r.detail]
        if last.status == RouteStatus.NONE_LISTED:
            details.append("PMC lists files the publisher page does not")
        result.detail = "; ".join(details)
        result.unretrieved = pmc.unretrieved
        return result

    result.route = chosen.route
    result.detail = chosen.detail
    result.unretrieved = list(chosen.unretrieved)
    result.manual_url = chosen.manual_url
    if result.unretrieved and result.manual_url is None:
        result.manual_url = next(
            (r.manual_url for r in resolution.routes if r.manual_url),
            f"https://doi.org/{doi}",
        )
    todo = [c for c in chosen.candidates if not _is_recorded(manifest, c)]
    result.already_recorded = len(chosen.candidates) - len(todo)
    if not todo:
        result.outcome = SiOutcome.PARTIAL if result.unretrieved else SiOutcome.PRESENT
        return result
    if dry_run:
        result.outcome = SiOutcome.WOULD_CAPTURE
        result.files = [
            SiFile(
                path=rel,
                original_filename=cand.original_filename,
                source_url=cand.source_url,
                retriever=cand.retriever,
            )
            for cand, rel in _plan(todo, next_si_index(key_dir, manifest))
        ]
        return result

    stamp = now or datetime.now(UTC).isoformat()
    index = next_si_index(key_dir, manifest)
    captured_pdfs: list[Path] = []
    try:
        for cand in todo:
            try:
                data = RETRIEVERS[cand.retriever](**cand.params)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in BLOCKED_STATUS:
                    raise
                result.unretrieved.append(
                    f"{cand.original_filename} ({cand.source_url})"
                )
                result.manual_url = result.manual_url or cand.source_url
                continue
            rel = f"si/si{index}{_extension(cand.original_filename)}"
            index += 1
            result.files.append(_store(key_dir, manifest, cand, rel, data, stamp))
            if rel.endswith(".pdf"):
                captured_pdfs.append(key_dir / rel)
        if do_ocr and captured_pdfs:
            result.ocr.extend(
                ocr_stored_pdfs(
                    key_dir,
                    manifest,
                    [str(p.relative_to(key_dir)) for p in captured_pdfs],
                )
            )
    except Exception as exc:  # noqa: BLE001 -- report the key, keep the batch going
        log.exception("capture_si: capturing %s (%s) failed", key, doi)
        result.error = f"{type(exc).__name__}: {exc}"
        return result
    if not result.files:
        result.outcome = SiOutcome.MANUAL
    elif result.unretrieved:
        result.outcome = SiOutcome.PARTIAL
    else:
        result.outcome = SiOutcome.CAPTURED
    return result


def capture_keys(
    requests: Sequence[KeyRequest],
    root: Path,
    *,
    dry_run: bool = False,
    do_ocr: bool = False,
    client: httpx.Client | None = None,
) -> SiCaptureReport:
    """:func:`capture_key` over every request, one discovery client for the run."""
    own = client is None
    http = make_client() if client is None else client
    try:
        results = []
        for request in requests:
            result = capture_key(http, root, request, dry_run=dry_run, do_ocr=do_ocr)
            log.info(
                "capture_si: %-48s %-13s %s",
                result.citation_key or result.doi,
                result.outcome,
                result.route or "",
            )
            results.append(result)
    finally:
        if own:
            http.close()
    return SiCaptureReport(
        generated_at=datetime.now(UTC).isoformat(),
        mirror_root=str(root),
        dry_run=dry_run,
        results=results,
    )


# -- request sources --------------------------------------------------------------


def requests_for_dois(root: Path, dois: Sequence[str]) -> list[KeyRequest]:
    """Map each DOI to the mirror key whose manifest records it (case-insensitive)."""
    if not dois:
        return []
    by_doi: dict[str, str] = {}
    for manifest_path in sorted(root.glob(f"*/{MANIFEST_FILENAME}")):
        if manifest_path.parent.name.startswith("_"):
            continue  # _bib and the other underscore stores are not paper keys
        manifest = Manifest.model_validate_json(manifest_path.read_text())
        if manifest.doi:
            by_doi.setdefault(manifest.doi.strip().lower(), manifest_path.parent.name)
    return [
        KeyRequest(citation_key=by_doi.get(doi.strip().lower()), doi=doi.strip())
        for doi in dois
    ]


def requests_for_collection(collection: str) -> list[KeyRequest]:
    """Every paper of a group-library Zotero collection (read-only)."""
    lib = ZoteroLibrary.from_env()
    items = _collection_items(
        lib, collection, collection_key=lib.collection_key(collection)
    )
    return [
        KeyRequest(
            citation_key=_resolve_citation_key(item),
            doi=(item["data"].get("DOI") or "").strip() or None,
        )
        for item in items
    ]


def dedupe(requests: Sequence[KeyRequest]) -> list[KeyRequest]:
    """Drop repeats of a key (a paper in two collections), keeping the first."""
    seen: set[str] = set()
    out: list[KeyRequest] = []
    for request in requests:
        ident = request.citation_key or f"doi:{request.doi}"
        if ident not in seen:
            seen.add(ident)
            out.append(request)
    return out


def write_report(report: SiCaptureReport, report_dir: Path) -> Path:
    """Write ``si_[dryrun_]<UTC stamp>.json`` under ``report_dir``."""
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    prefix = "si_dryrun_" if report.dry_run else "si_"
    path = report_dir / f"{prefix}{stamp}.json"
    path.write_text(report.model_dump_json(indent=2))
    return path


def main(argv: list[str] | None = None) -> int:
    """Command line; the caller loads ``.env`` first. Exit 1 when a key failed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("keys", nargs="*", help="Citation keys (mirror directories).")
    parser.add_argument(
        "--collection",
        action="append",
        default=[],
        metavar="NAME",
        help="Group-library Zotero collection whose papers to process; repeatable.",
    )
    parser.add_argument(
        "--doi", action="append", default=[], help="A DOI to process; repeatable."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Resolve and list; download nothing."
    )
    parser.add_argument(
        "--ocr", action="store_true", help="OCR captured SI PDFs with MinerU (GPU)."
    )
    parser.add_argument(
        "--mirror-root",
        type=Path,
        default=None,
        help="torchcell-library dir (default: $DATA_ROOT/torchcell-library).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help=f"Where the JSON report goes (default: <mirror-root>/{REPORTS_SUBDIR}).",
    )
    parser.add_argument(
        "--zip-member",
        action="append",
        default=[],
        metavar="ZIP:MEMBER",
        help="Store MEMBER of the key's recorded zip (e.g. si/si1.zip:Table_S1.pdf) "
        "as the next si/si<N> file; repeatable; needs exactly one citation key and "
        "no publisher resolution runs. With --ocr, stored PDFs are OCR'd (device "
        "from $MINERU_DEVICE_MODE).",
    )
    args = parser.parse_args(argv)
    if not (args.keys or args.collection or args.doi):
        parser.error("give citation keys, --collection or --doi")
    root = args.mirror_root or library_root(os.environ["DATA_ROOT"])
    if args.zip_member:
        if len(args.keys) != 1 or args.collection or args.doi or args.dry_run:
            parser.error("--zip-member takes exactly one citation key and no other")
        specs = [spec.partition(":") for spec in args.zip_member]
        for spec, (zip_rel, sep, member) in zip(args.zip_member, specs, strict=True):
            if not (zip_rel and sep and member):
                parser.error(f"--zip-member {spec!r} is not ZIP:MEMBER")
        key_dir = root / args.keys[0]
        manifest = Manifest.model_validate_json(
            (key_dir / MANIFEST_FILENAME).read_text()
        )
        stamp = datetime.now(UTC).isoformat()
        stored = [
            store_zip_member(key_dir, manifest, zip_rel, member, now=stamp)
            for zip_rel, _, member in specs
        ]
        for si in stored:
            print(f"stored {si.path} <- {si.original_filename} sha256={si.sha256}")
        if args.ocr:
            pdfs = [si.path for si in stored if si.path.endswith(".pdf")]
            for md in ocr_stored_pdfs(key_dir, manifest, pdfs):
                print(f"ocr {md} sha256={sha256_file(key_dir / md)}")
        return 0
    requests = [KeyRequest(citation_key=k) for k in args.keys]
    for collection in args.collection:
        requests.extend(requests_for_collection(collection))
    requests.extend(requests_for_dois(root, args.doi))
    report = capture_keys(dedupe(requests), root, dry_run=args.dry_run, do_ocr=args.ocr)
    for result in report.results:
        line = f"{result.outcome:<13} {result.citation_key or '-'} ({result.doi})"
        if result.route:
            line += f" route={result.route}"
        if result.files:
            line += f" files={len(result.files)}"
        if result.manual_url:
            line += f" manual={result.manual_url}"
        if result.error:
            line += f" error={result.error}"
        print(line)
    path = write_report(report, args.report_dir or root / REPORTS_SUBDIR)
    print(report.summary())
    print(f"report -> {path}")
    return 1 if any(r.outcome == SiOutcome.FAILED for r in report.results) else 0
