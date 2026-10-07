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
from urllib.parse import quote

import httpx

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120 Safari/537.36 torchcell-literature"
)

#: The PMC Article Datasets bucket: ``<PMCID>.<version>/<file>`` per article version.
PMC_CLOUD_BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"
#: Elsevier's asset CDN; supplementary files are ``1-s2.0-<PII>-mmc<N>.<ext>``.
ELSEVIER_ARS = "https://ars.els-cdn.com/content/image"


def _get(url: str, *, timeout: float = 120.0) -> bytes:
    """GET a URL following redirects; raise on non-2xx (no silent partials)."""
    with httpx.Client(
        follow_redirects=True, timeout=timeout, headers={"User-Agent": _UA}
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
    """Retrieve any directly-downloadable URL (Dryad, GEO, lab servers, Zenodo)."""
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
}
