# torchcell/literature/zotero.py
# [[torchcell.literature.zotero]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/zotero.py
# Test file: tests/torchcell/literature/test_zotero.py

"""Zotero library access: collections, items, attachments, with retries."""

import logging
import os
import random
import re
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypeVar

import httpx
from pydantic import BaseModel, Field, SecretStr
from pyzotero import zotero

from torchcell.literature.citation_keys import generate_citation_key

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

T = TypeVar("T")

# Retry policy for the flaky Zotero web API. 5xx and httpx transport/timeout
# errors are transient and retried; 4xx (notably 404 "missing item") is terminal.
RETRYABLE_STATUS = frozenset({500, 502, 503, 504})
MAX_TRIES = 3
BASE_DELAY = 2.0

# Supplementary-information indicators in an attachment title or filename. Whole
# words ("supplement", "supporting", publisher SI tokens "moesm"/"esm") match as
# substrings, but bare "si" must match only as a delimited token or prefix
# (``SI-...``, ``_si1``, `` si ``) -- never inside a word like "analysis".
_SI_WORD_INDICATORS = ("supplement", "supporting", "moesm", "esm")
_SI_TOKEN_RE = re.compile(r"(^|[^a-z])si([^a-z]|$)")


class ZoteroConfig(BaseModel):
    """Credentials for a single Zotero library.

    For the TorchCell group library only two values are needed: the group id and
    an API key. ``library_type`` defaults to ``"group"`` -- the only mode we use.
    The key is wrapped in ``SecretStr`` so it never leaks into logs or reprs.
    """

    model_config = {"frozen": True}

    library_id: str = Field(
        description="Zotero library id, e.g. '6582362' for the TorchCell group."
    )
    library_type: Literal["group", "user"] = Field(default="group")
    api_key: SecretStr = Field(description="Zotero API key.")

    @classmethod
    def from_env(cls) -> "ZoteroConfig":
        """Build credentials from ``ZOTERO_LIBRARY_ID`` + ``ZOTERO_API_KEY``.

        ``ZOTERO_LIBRARY_TYPE`` is honored if set but defaults to ``"group"``.
        Raises ``KeyError`` if either required variable is missing -- we do not
        silently degrade to an unauthenticated client.
        """
        library_id = os.environ["ZOTERO_LIBRARY_ID"]
        api_key = os.environ["ZOTERO_API_KEY"]
        raw_type = os.environ.get("ZOTERO_LIBRARY_TYPE", "group")
        library_type: Literal["group", "user"] = (
            "user" if raw_type == "user" else "group"
        )
        return cls(
            library_id=library_id, library_type=library_type, api_key=SecretStr(api_key)
        )


def make_zotero_client(config: ZoteroConfig) -> zotero.Zotero:
    """Construct a pyzotero client from a :class:`ZoteroConfig`."""
    return zotero.Zotero(
        config.library_id, config.library_type, config.api_key.get_secret_value()
    )


def _is_retryable(exc: Exception) -> bool:
    """Whether a Zotero call failure is transient and worth retrying."""
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    return False


def with_zotero_retry[T](
    call: Callable[[], T],
    *,
    max_tries: int = MAX_TRIES,
    base_delay: float = BASE_DELAY,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run a zero-arg Zotero call with bounded exponential-backoff retry.

    Wrap a single read (``lambda: zot.items(...)``, ``lambda: zot.file(k)``).
    Transient failures (httpx timeout/transport, 5xx) retry up to ``max_tries``
    with exponential backoff plus jitter; a 404 / other 4xx raises immediately.
    """
    for attempt in range(max_tries):
        try:
            return call()
        except Exception as exc:
            if not _is_retryable(exc) or attempt == max_tries - 1:
                raise
            delay = base_delay * (2**attempt) + random.uniform(0, base_delay)
            log.warning(
                "Zotero call failed (%s); retry %d/%d in %.1fs",
                exc.__class__.__name__,
                attempt + 1,
                max_tries - 1,
                delay,
            )
            sleep(delay)
    raise AssertionError("with_zotero_retry exhausted without raising")


def _is_si_attachment(attachment: dict[str, Any]) -> bool:
    """True if a PDF attachment looks like supplementary information.

    Matches whole-word indicators anywhere in the title/filename, plus a bare
    ``si`` token only when delimited (``SI-...``, ``_si1``) so it never trips on
    words like "analysis".
    """
    data = attachment["data"]
    text = f"{data.get('title') or ''} {data.get('filename') or ''}".lower()
    if any(w in text for w in _SI_WORD_INDICATORS):
        return True
    return bool(_SI_TOKEN_RE.search(text))


def _is_main_article(attachment: dict[str, Any]) -> bool:
    """True if a PDF attachment looks like the main article, not SI.

    Zotero auto-imports often title the main PDF "Full Text PDF"; otherwise the
    main article is simply the attachment that does not look like SI.
    """
    title = (attachment["data"].get("title") or "").lower()
    if "full text" in title:
        return True
    return not _is_si_attachment(attachment)


def _resolve_citation_key(item: dict[str, Any]) -> str:
    """The citation key for an item, preferring what Zotero already stores.

    Order: Zotero 7 native ``citationKey`` field -> Better BibTeX
    ``Citation Key: ...`` line in ``extra`` -> deterministic generation from
    metadata. Using the stored key keeps the on-disk artifact directory byte
    -identical to what the user sees in Zotero.
    """
    data = item["data"]
    native = data.get("citationKey")
    if native:
        return str(native)
    match = re.search(r"Citation Key:\s*(\S+)", data.get("extra") or "")
    if match:
        return match.group(1)
    creators = [
        f"{c.get('firstName', '')} {c.get('lastName', '')}".strip()
        for c in data.get("creators", [])
        if c.get("creatorType") == "author"
    ]
    return generate_citation_key(creators, data.get("date", ""), data.get("title", ""))


class ZoteroLibrary:
    """Synchronous handle to one Zotero library.

    A thin, retry-hardened wrapper over a pyzotero client exposing the
    operations the literature-capture pipeline needs: list/resolve collections,
    look up an item by DOI, enumerate a paper's PDF attachments (main article
    first, SI after), and download them into the on-disk artifact layout

        ``<data_root>/torchcell-library/<citation_key>/``
            paper.pdf
            si/si1.pdf  si/si2.pdf ...

    This mirror is a backup to the existing URL-based dataset downloads, keyed by
    DOI so a dataset writer can fall back to it when an upstream source is down.
    """

    def __init__(self, config: ZoteroConfig) -> None:
        """Wrap a Zotero client built from ``config``."""
        self.config = config
        self.zot = make_zotero_client(config)

    @classmethod
    def from_env(cls) -> "ZoteroLibrary":
        """Build a library from ``ZoteroConfig.from_env()``."""
        return cls(ZoteroConfig.from_env())

    # -- collections ---------------------------------------------------------

    def list_collections(self) -> list[dict[str, Any]]:
        """All collections in the library (name + key dicts)."""
        collections: list[dict[str, Any]] = with_zotero_retry(
            lambda: self.zot.collections()
        )
        return collections

    def collection_key(self, name: str, create_if_missing: bool = False) -> str:
        """Resolve a collection name to its key, optionally creating it.

        Raises ``ValueError`` listing available collections when the name is not
        found and ``create_if_missing`` is False.
        """
        name_lower = name.lower()
        collections = self.list_collections()
        for coll in collections:
            if coll["data"]["name"].lower() == name_lower:
                return str(coll["key"])
        if create_if_missing:
            resp = with_zotero_retry(
                lambda: self.zot.create_collections([{"name": name}])
            )
            return str(resp["successful"]["0"]["data"]["key"])
        available = sorted(c["data"]["name"] for c in collections)
        raise ValueError(
            f"Zotero collection '{name}' not found. Available: {available}."
        )

    # -- items by DOI --------------------------------------------------------

    def find_item_by_doi(self, doi: str) -> dict[str, Any] | None:
        """Return the library item whose ``data.DOI`` matches exactly, else None.

        Zotero's ``q`` quick search indexes title/creator/year (and, with
        ``qmode=everything``, attachment full text) but NOT the DOI metadata
        field, so a ``q=<doi>`` search returns nothing here. The reliable lookup
        is a full ``everything(items())`` scan matched against each item's actual
        ``data["DOI"]`` -- cheap for a curated library (a few paginated GETs).
        """
        target = doi.strip().lower()
        if not target:
            return None
        items: list[dict[str, Any]] = with_zotero_retry(
            lambda: self.zot.everything(self.zot.items())
        )
        for item in items:
            candidate = item.get("data", {}).get("DOI", "")
            if candidate and candidate.strip().lower() == target:
                return item
        return None

    def doi_in_library(self, doi: str) -> bool:
        """True if an item with this exact DOI already exists in the library."""
        return self.find_item_by_doi(doi) is not None

    # -- attachments ---------------------------------------------------------

    def pdf_attachments(self, item_key: str) -> list[dict[str, Any]]:
        """PDF attachments of an item, main article first then SI.

        Auto-paginates: heavily-annotated items can accrue more than the default
        100-item page, which would otherwise hide the source PDF.
        """
        children: list[dict[str, Any]] = with_zotero_retry(
            lambda: self.zot.everything(self.zot.children(item_key))
        )
        pdfs = [
            c for c in children if c["data"].get("contentType") == "application/pdf"
        ]
        pdfs.sort(key=lambda a: 0 if _is_main_article(a) else 1)
        return pdfs

    def download_artifact(
        self,
        item: dict[str, Any],
        data_root: str | Path | None = None,
        library_dirname: str = "torchcell-library",
    ) -> Path:
        """Download an item's PDFs into its citation-keyed artifact directory.

        Writes the main article to ``paper.pdf`` and each SI PDF to
        ``si/si1.pdf``, ``si/si2.pdf``, ... under
        ``<data_root>/<library_dirname>/<citation_key>/``. The citation key is
        taken from Zotero (native field or ``extra``) so the directory matches
        the library; see :func:`_resolve_citation_key`.

        Args:
            item: A Zotero item dict (as returned by :meth:`find_item_by_doi`).
            data_root: Root for the mirror; defaults to ``$DATA_ROOT``.
            library_dirname: Name of the library subdirectory under the root.

        Returns:
            The artifact directory path.
        """
        root = Path(data_root if data_root is not None else os.environ["DATA_ROOT"])
        citation_key = _resolve_citation_key(item)
        artifact_dir = root / library_dirname / citation_key
        artifact_dir.mkdir(parents=True, exist_ok=True)

        attachments = self.pdf_attachments(item["key"])
        si_dir = artifact_dir / "si"
        si_index = 0
        for i, att in enumerate(attachments):
            content: bytes = with_zotero_retry(partial(self.zot.file, att["key"]))
            if i == 0:
                dest = artifact_dir / "paper.pdf"
            else:
                si_index += 1
                si_dir.mkdir(parents=True, exist_ok=True)
                dest = si_dir / f"si{si_index}.pdf"
            dest.write_bytes(content)
            log.info("Zotero: wrote %s (%d bytes)", dest, len(content))

        return artifact_dir


def _smoke_test() -> None:
    """Connect to the configured library and print a connectivity summary.

    Run with ``python -m torchcell.literature.zotero`` after setting
    ``ZOTERO_LIBRARY_ID`` and ``ZOTERO_API_KEY`` (loaded from .env).
    """
    from dotenv import load_dotenv

    load_dotenv()
    lib = ZoteroLibrary.from_env()
    log.info(
        "Connected to Zotero %s library %s",
        lib.config.library_type,
        lib.config.library_id,
    )
    collections = lib.list_collections()
    log.info("Found %d collection(s):", len(collections))
    for coll in sorted(collections, key=lambda c: c["data"]["name"].lower()):
        log.info("  - %s (key=%s)", coll["data"]["name"], coll["key"])


if __name__ == "__main__":
    _smoke_test()
