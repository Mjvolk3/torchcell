# torchcell/provenance/pinned_fetch
# [[torchcell.provenance.pinned_fetch]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/provenance/pinned_fetch
# Test file: tests/torchcell/provenance/test_pinned_fetch.py

"""Fetch one sha256-pinned file: the recorded source first, our mirror as the fallback.

The reproducibility policy for every reference file outside the graph (genome releases
now; network releases such as STRING next): the bytes are pinned by sha256 and size, the
source URL is retrieval metadata, and a copy is kept in the data lake. A fetch therefore

1. downloads from the recorded ``source_url`` when there is one, and accepts the bytes
   only when their size and sha256 equal the pin;
2. on any failure of step 1 (the request fails, the status is not 200, or the bytes
   differ from the pin) logs the reason as a warning and takes the file from the mirror,
   which is held to the same pin.

The source is tried first on purpose: that is how a rotted URL or a changed upstream
file is detected rather than silently papered over. :class:`FetchOutcome` records which
origin served the file and why the source was not used, so a caller can write the result
beside the files.

The mirror is a callable so this module knows nothing about a tier: it is given the
destination and must leave the verified file there, or raise.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict

log = logging.getLogger(__name__)

CHUNK = 1 << 20
#: Seconds to wait on the source (connect, and between bytes) before the mirror is used.
SOURCE_TIMEOUT = 60.0
SOURCE_PART_SUFFIX = ".source.part"

Origin = Literal["source", "mirror"]


class FetchOutcome(BaseModel):
    """Where one pinned file came from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str
    bytes: int
    origin: Origin
    source_url: str | None
    #: Why the source did not serve the file; ``None`` when it did.
    source_failure: str | None


def sha256_of(path: Path) -> str:
    """sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _from_source(
    url: str, dest: Path, sha256: str, nbytes: int, http: Any
) -> str | None:
    """Download ``url`` to ``dest`` if its bytes match the pin; return the failure or None."""
    part = dest.with_name(dest.name + SOURCE_PART_SUFFIX)
    try:
        with http.stream("GET", url) as response:
            if response.status_code != 200:
                return f"HTTP {response.status_code}"
            with part.open("wb") as handle:
                for chunk in response.iter_bytes(CHUNK):
                    handle.write(chunk)
    except httpx.HTTPError as error:
        part.unlink(missing_ok=True)
        return f"{type(error).__name__}: {error}"
    size = part.stat().st_size
    got = sha256_of(part)
    if size != nbytes or got != sha256:
        part.unlink()
        return (
            f"source bytes differ from the pin: {size} bytes sha256 {got}, pinned "
            f"{nbytes} bytes sha256 {sha256}"
        )
    os.replace(part, dest)
    return None


def fetch_pinned(
    dest: Path,
    *,
    sha256: str,
    nbytes: int,
    source_url: str | None,
    mirror: Callable[[Path], Any],
    label: str,
    http: Any | None = None,
) -> FetchOutcome:
    """Leave the pinned file at ``dest``; the source first, the mirror on any failure.

    ``mirror(dest)`` must write the file to ``dest``; its bytes are re-hashed here, so a
    mirror cannot satisfy the pin by assertion. ``http`` injects the client used for the
    source (an ``httpx.Client`` that follows redirects by default). ``label`` names the
    file in the log.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    failure: str | None
    if source_url is None:
        failure = "no source URL is recorded for this file"
    elif not source_url.startswith(("http://", "https://")):
        failure = f"source URL scheme is not http(s): {source_url}"
    else:
        client = http or httpx.Client(timeout=SOURCE_TIMEOUT, follow_redirects=True)
        failure = _from_source(source_url, dest, sha256, nbytes, client)
        if failure is None:
            return FetchOutcome(
                path=str(dest),
                sha256=sha256,
                bytes=nbytes,
                origin="source",
                source_url=source_url,
                source_failure=None,
            )
        log.warning(
            "%s: source %s not used (%s); taking the mirror copy",
            label,
            source_url,
            failure,
        )
    mirror(dest)
    size = dest.stat().st_size
    got = sha256_of(dest)
    if size != nbytes or got != sha256:
        dest.unlink()
        raise RuntimeError(
            f"{label}: mirror bytes differ from the pin ({size} bytes sha256 {got}, "
            f"pinned {nbytes} bytes sha256 {sha256}); file removed"
        )
    return FetchOutcome(
        path=str(dest),
        sha256=sha256,
        bytes=nbytes,
        origin="mirror",
        source_url=source_url,
        source_failure=failure,
    )
