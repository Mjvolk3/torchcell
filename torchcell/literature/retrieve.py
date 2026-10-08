# torchcell/literature/retrieve.py
# [[torchcell.literature.retrieve]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/retrieve.py
# Test file: tests/torchcell/literature/test_provenance.py
"""Versioned artifact retrievers.

Retrieval provenance references these functions by dotted path (not a free-form
shell command), so *how* an artifact was fetched is versioned source code we can
test and re-run. A ``RetrievalRecord`` stores the retriever's dotted path + its
params; ``run_retriever`` resolves and calls it.

Retrieval-method reality (verified 2026.07): Springer ESM
(``static-content.springer.com``) and the PMC OA API are scriptable; PMC file
downloads (JS proof-of-work) and ``nature.com`` (auth redirect) are not -- those
route through Zotero or, in future, the Radiant VM endpoint (issue #20).

Measured 2026-10-07: the PMC OA web service (``oa.fcgi``) answers HTTP 404 for every
id tried, so :func:`pmc_oa_api` no longer retrieves anything. Its successor is the PMC
Article Datasets bucket on AWS (``pmc-oa-opendata``), read with
:func:`pmc_cloud_object`. The supplementary-file retrievers used by
``torchcell.literature.capture_si`` (:func:`pmc_cloud_object`,
:func:`plos_supplementary`, :func:`elsevier_mmc`, plus :func:`springer_esm` and
:func:`direct_url`) are each a plain GET of one URL built from their params.
"""

from __future__ import annotations

from collections.abc import Callable
from urllib.parse import quote, urlsplit

import httpx

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120 Safari/537.36 torchcell-literature"
)

_PLAIN_UA = "torchcell-literature (+https://github.com/Mjvolk3/torchcell)"
"""The non-browser User-Agent, for hosts that refuse a browser string.

Zenodo answers :data:`_UA` with HTTP **403** and a non-browser agent with 200/206
(measured 2026-10-07 with curl: ``torchcell-literature``, ``python-httpx/0.27.0`` and
``Mozilla/5.0 torchcell-literature`` all 206, the Chrome 120 string 403), so a recorded
``zip_member`` retrieval of the PRECISE-1K archive did not re-run. The browser string is
kept as the default because the publisher CDNs the other retrievers use were chosen
against it; only the hosts in :data:`PLAIN_UA_HOSTS` get this one.
"""

PLAIN_UA_HOSTS: frozenset[str] = frozenset({"zenodo.org"})
"""Hosts served :data:`_PLAIN_UA` instead of the browser string.

Matched on the URL's host, exactly or as a subdomain of a listed host: ``zenodo.org``
covers ``sandbox.zenodo.org`` (the same software), and the match is on a dotted boundary,
so a host that merely ends in the same letters (``notzenodo.org``) is not covered. A host
is added here only after its refusal is measured, with the measurement recorded.
"""

#: The PMC Article Datasets bucket: ``<PMCID>.<version>/<file>`` per article version.
PMC_CLOUD_BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"
#: Elsevier's asset CDN; supplementary files are ``1-s2.0-<PII>-mmc<N>.<ext>``.
ELSEVIER_ARS = "https://ars.els-cdn.com/content/image"


def user_agent_for(url: str) -> str:
    """The User-Agent this URL's host is served.

    :data:`_PLAIN_UA` for a host in :data:`PLAIN_UA_HOSTS` (or a subdomain of one),
    :data:`_UA` otherwise. Case-insensitive, since a host name is.
    """
    host = (urlsplit(url).hostname or "").lower()
    return (
        _PLAIN_UA
        if any(host == plain or host.endswith(f".{plain}") for plain in PLAIN_UA_HOSTS)
        else _UA
    )


def _get(url: str, *, timeout: float = 120.0) -> bytes:
    """GET a URL following redirects; raise on non-2xx (no silent partials)."""
    with httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": user_agent_for(url)},
    ) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.content


def springer_esm(url: str) -> bytes:
    """Retrieve a Springer ESM (supplementary) file.

    ``static-content.springer.com/esm/...`` is a directly scriptable CDN (unlike
    nature.com's auth gate). Example ``url``:
    ``https://static-content.springer.com/esm/art%3A10.1038%2Fnmeth.1534/MediaObjects/41592_2010_BFnmeth1534_MOESM167_ESM.pdf``
    """
    return _get(url)


def direct_url(url: str) -> bytes:
    """Retrieve any directly-downloadable URL (Dryad, GEO, lab servers, Zenodo).

    A Zenodo URL is sent the non-browser User-Agent (:data:`PLAIN_UA_HOSTS`), which is
    what it answers; the browser string gets 403 there.
    """
    return _get(url)


def zip_member(url: str, member: str, container_sha256: str | None) -> bytes:
    """Retrieve one member file out of a zip archive served at ``url``.

    The whole container is downloaded and, when ``container_sha256`` is given, its
    sha256 is asserted BEFORE any member is read, so a rebuild that meets a re-packed
    archive whose contents changed fails loudly instead of silently yielding a
    different member. ``None`` is for hosts that re-zip per request with a fresh
    container hash but stable members (Europe PMC's supplementaryFiles endpoint,
    measured 2026-09-12: two retrievals, two container hashes, one member hash); the
    member's own sha256, pinned by the caller, is then the only anchor. The member
    bytes are returned as-is.
    """
    import hashlib
    import io
    import zipfile

    container = _get(url, timeout=1800.0)
    if container_sha256 is not None:
        got = hashlib.sha256(container).hexdigest()
        if got != container_sha256:
            raise ValueError(
                f"zip container sha256 mismatch for {url}: got {got}, "
                f"expected {container_sha256}"
            )
    with zipfile.ZipFile(io.BytesIO(container)) as archive:
        return archive.read(member)


def pmc_oa_api(pmcid: str) -> bytes:
    """Retrieve a PMC open-access package tarball via the OA API.

    Raises ``ValueError`` if the id is not in the redistributable OA subset (author
    manuscripts often are not), so the caller falls back to Zotero rather than
    silently getting an interstitial.
    """
    import xml.etree.ElementTree as ET

    api = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
    root = ET.fromstring(_get(api).decode("utf-8", errors="replace"))
    error = root.find(".//error")
    if error is not None:
        raise ValueError(f"PMC {pmcid} not open-access: {error.get('code')}")
    link = root.find(".//link[@format='tgz']")
    if link is None or not link.get("href"):
        raise ValueError(f"PMC {pmcid}: no tgz package link in OA response")
    href = link.get("href", "").replace("ftp://", "https://")
    return _get(href)


def pmc_cloud_url(key: str) -> str:
    """HTTPS URL of one object in the PMC Article Datasets bucket."""
    return f"{PMC_CLOUD_BUCKET}/{quote(key, safe='/')}"


def pmc_cloud_object(key: str) -> bytes:
    """Retrieve one object of the PMC Article Datasets bucket (``pmc-oa-opendata``).

    ``key`` is the bucket key, ``<PMCID>.<version>/<file>``, e.g.
    ``PMC11176082.1/41588_2024_1769_MOESM3_ESM.xlsx``. The bucket holds the
    open-access and author-manuscript subsets of PMC, one prefix per article version,
    and is the scriptable successor of the retired OA package service.
    """
    return _get(pmc_cloud_url(key))


def plos_supplementary_url(journal: str, object_doi: str) -> str:
    """PLOS article-file URL of one supplementary object (``<doi>.s001`` ...)."""
    return (
        f"https://journals.plos.org/{journal}/article/file"
        f"?id={object_doi}&type=supplementary"
    )


def plos_supplementary(journal: str, object_doi: str) -> bytes:
    """Retrieve one PLOS supplementary file by its object DOI.

    ``journal`` is the site slug (``plosgenetics``, ``plosone`` ...) and
    ``object_doi`` the file's own DOI, e.g. ``10.1371/journal.pgen.1004120.s001``. The
    URL redirects to a signed storage URL that changes per request; the article-file
    URL is the stable one and is what the record keeps.
    """
    return _get(plos_supplementary_url(journal, object_doi))


def elsevier_mmc_url(pii: str, filename: str) -> str:
    """Elsevier CDN URL of a supplementary file (``filename`` like ``mmc2.xlsx``)."""
    return f"{ELSEVIER_ARS}/1-s2.0-{pii}-{filename}"


def elsevier_mmc(pii: str, filename: str) -> bytes:
    """Retrieve one Elsevier / Cell Press supplementary file from ``ars.els-cdn.com``.

    ``pii`` is the article's PII without punctuation (Crossref ``alternative-id``,
    e.g. ``S2405471220303665``) and ``filename`` the multimedia component name
    (``mmc1.pdf``). The CDN serves these directly although ScienceDirect and cell.com
    article pages answer scripted clients with HTTP 403.
    """
    return _get(elsevier_mmc_url(pii, filename))


class ArchiveHashMismatchError(ValueError):
    """A local-archive file's bytes no longer hash to the sha256 its record pins."""


def local_archive(path: str, sha256: str) -> bytes:
    """Read one file out of a local archive, refusing bytes that do not hash to ``sha256``.

    For in-house material with no public URL (the thesis archive on ``/bulk``): the
    archive keeps its own manifest of where every file originally lived, and the
    record pins the sha256 that manifest lists. A rebuild re-reads the archive copy; a
    file edited, replaced or truncated since raises :class:`ArchiveHashMismatchError`
    instead of being followed, and a missing file raises ``FileNotFoundError``.
    """
    import hashlib
    from pathlib import Path

    data = Path(path).read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        raise ArchiveHashMismatchError(
            f"local archive sha256 mismatch for {path}: got {got}, expected {sha256}"
        )
    return data


# Registry: dotted path -> retriever. RetrievalRecord.retriever names a key here.
# The ``radiant_endpoint`` RetrievalMethod slot (issue #20) is intentionally left
# without a retriever here: it is reserved for the Radiant VM serving library-rebuild
# artifacts, a separate concern from the private literature endpoint in
# ``torchcell.literature.server`` (which runs on GilaHyper). Its retriever is added
# when that rebuild path is actually built.
RETRIEVERS: dict[str, Callable[..., bytes]] = {
    "torchcell.literature.retrieve.springer_esm": springer_esm,
    "torchcell.literature.retrieve.direct_url": direct_url,
    "torchcell.literature.retrieve.zip_member": zip_member,
    "torchcell.literature.retrieve.pmc_oa_api": pmc_oa_api,
    "torchcell.literature.retrieve.pmc_cloud_object": pmc_cloud_object,
    "torchcell.literature.retrieve.plos_supplementary": plos_supplementary,
    "torchcell.literature.retrieve.elsevier_mmc": elsevier_mmc,
    "torchcell.literature.retrieve.local_archive": local_archive,
}
