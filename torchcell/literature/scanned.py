# torchcell/literature/scanned.py
# [[torchcell.literature.scanned]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/scanned.py
# Test file: tests/torchcell/literature/test_scanned.py

"""Self-verifying multi-pass OCR for scanned (non-born-digital) tables.

A born-digital PDF carries an exact text layer; a scan does not, so OCR must
carry completeness on its own. A single MinerU/VLM pass drops rows from dense
tables, and -- because layout detection runs on the page image at the chosen DPI
-- *different DPIs drop different rows*. So the robust strategy is:

    1. OCR the PDF at several DPIs (a sweep), accumulating the output text.
    2. After each pass, parse keys from the UNION of all text so far. A row found
       by ANY pass is kept -> recall climbs monotonically with each pass.
    3. Run a shape-of-data oracle (expected count / known schema). Stop as soon as
       it clears; otherwise escalate to the next, higher-resolution pass.

Effective resolution saturates near ~360 DPI (Qwen's pixel budget downsamples
larger pages back to the cap), so a sweep of ~{300, 350} plus an optional budget
-raised/tiled escalation covers the useful range; cranking DPI to 600 is wasted.

The oracle proves completeness of row *coverage*, not correctness of cell values;
for value precision on noisy scans, prefer cross-pass agreement (a separate step).
"""

import logging
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel

from torchcell.literature.ocr import ocr_pdf

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ShapeReport(BaseModel):
    """Verdict of the shape-of-data oracle for an accumulated extraction."""

    found: int
    expected: int | None
    missing_keys: list[str]
    complete: bool


def shape_check(
    keys: Iterable[str],
    *,
    expected_keys: Iterable[str] | None = None,
    expected_n: int | None = None,
) -> ShapeReport:
    """Judge whether an extraction is complete by the shape of the data.

    Two modes, strongest first:

    - ``expected_keys`` (known-schema): the exact id set the table should yield
      (e.g. a target dataset's parameter ids). Reports precisely which are
      missing -- complete iff none are.
    - ``expected_n`` (count-only): the row count the caption advertises (e.g.
      "501 parameters"). Complete once that many distinct keys are found.
    """
    found = set(keys)
    if expected_keys is not None:
        expected_set = set(expected_keys)
        missing = sorted(expected_set - found)
        return ShapeReport(
            found=len(found),
            expected=len(expected_set),
            missing_keys=missing,
            complete=not missing,
        )
    if expected_n is not None:
        return ShapeReport(
            found=len(found),
            expected=expected_n,
            missing_keys=[],
            complete=len(found) >= expected_n,
        )
    return ShapeReport(found=len(found), expected=None, missing_keys=[], complete=False)


def extract_scanned(
    pdf_path: str | Path,
    parse_keys: Callable[[str], set[str]],
    *,
    dpis: Sequence[int] = (300, 350),
    backend: str = "vlm-auto-engine",
    expected_keys: Iterable[str] | None = None,
    expected_n: int | None = None,
    **ocr_kwargs: object,
) -> tuple[set[str], list[tuple[int, ShapeReport]]]:
    """OCR a scanned table across a DPI sweep until the oracle says it is complete.

    Each pass OCRs at the next DPI and appends its markdown to the accumulated
    text; ``parse_keys`` is applied to the union so coverage only grows. The sweep
    stops early the moment :func:`shape_check` reports complete.

    Args:
        pdf_path: The scanned PDF (route born-digital PDFs through the text layer).
        parse_keys: Maps OCR text -> the set of keys (e.g. ids) it contains.
        dpis: Ascending DPIs to try; later passes add resolution and vary the crop.
        backend: MinerU backend (the VLM handles messy tables far better here).
        expected_keys: Oracle set of keys the table should contain.
        expected_n: Oracle count of keys the table should contain.
        ocr_kwargs: Forwarded to :func:`ocr_pdf` (e.g. ``device_mode``).

    Returns:
        ``(found_keys, reports)`` where ``reports`` is one ``(dpi, ShapeReport)``
        per pass actually run.
    """
    pdf_path = Path(pdf_path)
    accumulated = ""
    found: set[str] = set()
    reports: list[tuple[int, ShapeReport]] = []
    for dpi in dpis:
        md = ocr_pdf(pdf_path, backend=backend, dpi=dpi, **ocr_kwargs)  # type: ignore[arg-type]
        accumulated += "\n" + md.read_text()
        found = parse_keys(accumulated)
        report = shape_check(found, expected_keys=expected_keys, expected_n=expected_n)
        reports.append((dpi, report))
        log.info(
            "scanned pass dpi=%d: found %d/%s (%s)",
            dpi,
            report.found,
            report.expected,
            "complete" if report.complete else "incomplete",
        )
        if report.complete:
            break
    return found, reports
