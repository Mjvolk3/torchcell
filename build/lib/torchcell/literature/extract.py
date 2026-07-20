# torchcell/literature/extract.py
# [[torchcell.literature.extract]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/extract.py
# Test file: tests/torchcell/literature/test_extract.py

"""Born-digital detection and text-layer extraction for paper artifacts.

The literature-capture pipeline mirrors a paper's PDFs and OCRs them, but OCR is
only the *fallback*. A born-digital PDF (one whose pages are painted vector text,
not scanned images) already carries the data as a recoverable text layer that is
exact, deterministic, and instant -- strictly better than OCR for tables. This
module decides which regime a PDF is in and, for the born-digital case, pulls the
text layer with column geometry preserved.

Strategy hierarchy (see :func:`pdf_kind`):

    born_digital -> trust the text layer (``pdftotext -layout``)
    scanned      -> the text layer is absent or untrustworthy; use high-DPI VLM
                    OCR via :mod:`torchcell.literature.ocr`

Built on the poppler CLIs (``pdffonts``/``pdfimages``/``pdftotext``/``pdfinfo``),
which are a ubiquitous system dependency -- no Python PDF library is required.
"""

import re
import subprocess
from pathlib import Path
from typing import Literal

PdfKind = Literal["born_digital", "scanned"]

# A born-digital page is essentially all vector text; a scan is one page-sized
# raster per page. We call a PDF scanned when a near-page-sized image appears on
# most pages (covers both image-only scans and Acrobat-OCR'd scans, whose text
# layer is OCR output and must NOT be trusted). Born-digital figures are smaller
# than a full page, so this does not misfire on text papers that contain plots.
_MIN_TEXT_WORDS = 20  # fewer words than this => no usable text layer
_PAGE_IMAGE_AREA_FRAC = 0.5  # image >= this fraction of page area == "page-sized"


def _run(cmd: list[str]) -> str:
    """Run a poppler CLI and return stdout (text). Raises on non-zero exit."""
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def pdf_fonts(pdf_path: str | Path) -> list[str]:
    """Names of fonts embedded/used in the PDF (empty for a pure image scan)."""
    out = _run(["pdffonts", str(pdf_path)])
    rows = out.splitlines()[2:]  # skip the two header lines
    return [r.split()[0] for r in rows if r.strip()]


def pdf_page_count(pdf_path: str | Path) -> int:
    """Number of pages, via ``pdfinfo``."""
    out = _run(["pdfinfo", str(pdf_path)])
    match = re.search(r"^Pages:\s*(\d+)", out, re.MULTILINE)
    if match is None:
        raise ValueError(f"pdfinfo reported no page count for {pdf_path}")
    return int(match.group(1))


def pdf_text(pdf_path: str | Path, *, layout: bool = True) -> str:
    """Extract the text layer. ``layout=True`` preserves column geometry.

    ``pdftotext -layout`` keeps a table's columns aligned by whitespace, which is
    what makes header-anchored parsing of born-digital tables reliable.
    """
    cmd = ["pdftotext"]
    if layout:
        cmd.append("-layout")
    cmd += [str(pdf_path), "-"]
    return _run(cmd)


def _page_sized_image_count(pdf_path: str | Path) -> int:
    """Number of pages bearing a near-page-sized raster image.

    Parses ``pdfimages -list`` (width/height in px, x/y ppi per image) and counts
    distinct pages where an image covers >= ``_PAGE_IMAGE_AREA_FRAC`` of the page.
    A page at 72 pt/in and the image's own ppi gives the page's pixel area for the
    comparison.
    """
    out = _run(["pdfimages", "-list", str(pdf_path)])
    rows = out.splitlines()[2:]
    # Page geometry in points (1/72 in) -> compare against image px at its ppi.
    info = _run(["pdfinfo", str(pdf_path)])
    size = re.search(r"Page size:\s*([\d.]+)\s*x\s*([\d.]+)\s*pts", info)
    if size is None:
        return 0
    page_w_pt, page_h_pt = float(size.group(1)), float(size.group(2))

    pages_with_big_image: set[int] = set()
    for r in rows:
        cols = r.split()
        if len(cols) < 15:
            continue
        page = int(cols[0])
        img_w, img_h = int(cols[3]), int(cols[4])
        x_ppi, y_ppi = float(cols[12]), float(cols[13])
        if x_ppi <= 0 or y_ppi <= 0:
            continue
        page_w_px = page_w_pt / 72.0 * x_ppi
        page_h_px = page_h_pt / 72.0 * y_ppi
        page_area = page_w_px * page_h_px
        if page_area <= 0:
            continue
        if (img_w * img_h) / page_area >= _PAGE_IMAGE_AREA_FRAC:
            pages_with_big_image.add(page)
    return len(pages_with_big_image)


def pdf_kind(pdf_path: str | Path) -> PdfKind:
    """Classify a PDF as ``born_digital`` (trust text layer) or ``scanned``.

    Born-digital requires: real embedded fonts, a substantive text layer, and the
    absence of a page-sized image on most pages (which would mark a scan, possibly
    OCR'd). Otherwise the text layer is missing or untrustworthy and OCR is needed.
    """
    if not pdf_fonts(pdf_path):
        return "scanned"
    if len(pdf_text(pdf_path, layout=False).split()) < _MIN_TEXT_WORDS:
        return "scanned"
    pages = pdf_page_count(pdf_path)
    if _page_sized_image_count(pdf_path) >= max(1, pages):
        return "scanned"
    return "born_digital"


def iter_layout_rows(text: str) -> list[list[str]]:
    """Split ``pdftotext -layout`` output into per-line whitespace-token lists.

    Blank lines are dropped. Each returned row is ``line.split()`` -- callers
    pick out the fields they need (e.g. the first integer = a row number, the
    token after it = an id). Underscore-joined cells (common in scientific SI
    tables) stay single tokens, so a multi-word trailing column is recoverable as
    ``" ".join(tokens[k:])``.
    """
    return [line.split() for line in text.splitlines() if line.strip()]
