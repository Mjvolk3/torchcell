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
"""

from __future__ import annotations

from collections.abc import Callable

import httpx

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120 Safari/537.36 torchcell-literature"
)


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


# Registry: dotted path -> retriever. RetrievalRecord.retriever names a key here.
# The ``radiant_endpoint`` RetrievalMethod slot (issue #20) is intentionally left
# without a retriever here: it is reserved for the Radiant VM serving library-rebuild
# artifacts, a separate concern from the private literature endpoint in
# ``torchcell.literature.server`` (which runs on GilaHyper). Its retriever is added
# when that rebuild path is actually built.
RETRIEVERS: dict[str, Callable[..., bytes]] = {
    "torchcell.literature.retrieve.springer_esm": springer_esm,
    "torchcell.literature.retrieve.direct_url": direct_url,
    "torchcell.literature.retrieve.pmc_oa_api": pmc_oa_api,
}
