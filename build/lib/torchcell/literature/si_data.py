# torchcell/literature/si_data.py
# [[torchcell.literature.si_data]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/si_data.py
# Test file: tests/torchcell/literature/test_si_data.py

"""Fetch supplementary-data files from Dryad and Zotero attachments."""

import logging
from pathlib import Path
from urllib.parse import quote, unquote

import httpx

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DRYAD_HOST = "https://datadryad.org"


def dryad_files(dryad_doi: str) -> list[tuple[str, str]]:
    """Resolve a DRYAD dataset DOI to ``[(filename, download_url), ...]``.

    Walks dataset -> latest version -> files via the DRYAD v2 API. The download
    hrefs are API-relative; they are returned as absolute URLs.

    Args:
        dryad_doi: e.g. ``"10.5061/dryad.tt367"`` (no ``doi:`` prefix).
    """
    enc = quote(f"doi:{dryad_doi}", safe="")
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        ds = client.get(f"{DRYAD_HOST}/api/v2/datasets/{enc}")
        ds.raise_for_status()
        version_href = ds.json()["_links"]["stash:version"]["href"]
        resp = client.get(f"{DRYAD_HOST}{version_href}/files")
        resp.raise_for_status()
        files = resp.json()["_embedded"]["stash:files"]
    return [
        (f["path"], DRYAD_HOST + f["_links"]["stash:download"]["href"]) for f in files
    ]


def download_file(url: str, dest: Path, *, timeout: float = 600.0) -> Path:
    """Stream a URL to ``dest`` (following redirects). Raises on HTTP error."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in resp.iter_bytes(1 << 20):
                    fh.write(chunk)
    return dest


def fetch_si_data(
    artifact_dir: str | Path,
    *,
    dryad_doi: str | None = None,
    extra_urls: list[str] | None = None,
    overwrite: bool = False,
) -> list[tuple[Path, str]]:
    """Download SI data files into ``<artifact_dir>/si/si_data/``.

    Combines a DRYAD dataset (all files) with any explicit ``extra_urls``. These
    external repositories (DRYAD/GEO/Zenodo/GitHub) are the durable, freely
    available home of the supplementary *data* -- the SI PDF lists what should
    exist; this fetches the actual files so reproduction never needs the
    publisher.

    Returns:
        List of ``(local_path, source_url)`` for each downloaded file.
    """
    artifact_dir = Path(artifact_dir)
    dest_dir = artifact_dir / "si" / "si_data"

    pairs: list[tuple[str, str]] = []
    if dryad_doi:
        pairs.extend(dryad_files(dryad_doi))
    for url in extra_urls or []:
        # Decode percent-encoding (e.g. %20) for a clean local filename.
        name = unquote(url.rsplit("/", 1)[-1]) or "file"
        pairs.append((name, url))

    results: list[tuple[Path, str]] = []
    for filename, url in pairs:
        dest = dest_dir / filename
        if not overwrite and dest.exists() and dest.stat().st_size > 0:
            log.info(
                "SI data: keeping existing %s (%d bytes)", dest, dest.stat().st_size
            )
            results.append((dest, url))
            continue
        download_file(url, dest)
        log.info("SI data: wrote %s (%d bytes) from %s", dest, dest.stat().st_size, url)
        results.append((dest, url))
    return results
