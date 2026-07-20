# torchcell/literature/calmorph.py
# [[torchcell.literature.calmorph]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/literature/calmorph.py
# Test file: tests/torchcell/literature/test_calmorph.py

"""Extraction recipe: rebuild the CalMorph 501-parameter schema from the paper.

This is the extraction-backed replacement for ``scripts/generate_calmorph_labels``,
which depended on a *manually prepared* Excel file (``SI_1_parameters.xlsx``) that
was, in turn, hand-cleaned from a poor Mathpix OCR of the Ohya 2005 SI. The same
table is "SI - table1 parameter description" in the Zotero ``database`` collection
for DOI ``10.1073/pnas.0509436102`` -- a born-digital PDF whose text layer carries
all 501 ``ID -> Description`` rows exactly. This recipe reads that text layer and
reconstructs the dictionary, with the manual ``calmorph_labels`` module available
as ground truth for verification (see the test file).

Each row of the born-digital table is, after ``pdftotext -layout`` + whitespace
tokenization::

    [Nuclear_stage?]   No.   ID   Description   [Definition...]

The Nuclear_stage cell is sparse (only on the first row of each stage block) and
Definition may contain spaces, but ``No.`` is the first integer, ``ID`` and
``Description`` are the two single tokens after it, so the row parses
unambiguously. A row is accepted only when the token after ``No.`` matches a
CalMorph id, which discards the title/header/footer lines.
"""

import re
from pathlib import Path

from torchcell.literature.extract import pdf_kind, pdf_text

# A data row, anchored on the running ``No.`` (the first standalone integer) and
# the id directly after it. CalMorph ids: a channel letter (C cell / A actin /
# D DNA / T total), an optional ``CV`` (coefficient-of-variation variant), a
# parameter number with an optional ``-n`` subindex, and an OPTIONAL nuclear-stage
# suffix (_A G1 / _A1B single-nucleus / _B / _C anaphase). The suffix is optional
# because the ``Total_stage`` block uses bare ids (C119, A105, D199). The id must
# start the regex on a token boundary so the ``No.`` integer is never mistaken for
# part of the id. E.g. C11-1_A, A101_A1B, CCV11-1_A, DCV136_A1B, C119.
_ROW_RE = re.compile(
    r"(?<!\S)(\d+)\s+([ACDT](?:CV)?\d+(?:-\d)?(?:_(?:A1B|A|B|C))?)\s+(.+)$"
)
# ``pdftotext -layout`` separates columns by runs of 2+ spaces; a single space is
# inside a cell (one 2005 source typo writes a description as "..._in_whole cell").
_COL_GAP_RE = re.compile(r"\s{2,}")


def parse_calmorph_table(layout_text: str) -> dict[str, str]:
    """Parse ``ID -> Description`` from the table1 ``pdftotext -layout`` text.

    Each data line is ``[Nuclear_stage?]  No.  ID  Description  [Definition...]``.
    We anchor on ``No.`` + ``ID``, then split the remainder on column gaps (2+
    spaces) and take the first field as the description -- so a stray single space
    inside a description stays put and a multi-token Definition is ignored. The
    description's internal spaces are normalized to underscores, matching both the
    source's dominant convention and the manual ``calmorph_labels`` build.

    Args:
        layout_text: Output of ``pdf_text(table1_pdf, layout=True)``.

    Returns:
        Mapping of every CalMorph parameter id to its description.
    """
    params: dict[str, str] = {}
    for line in layout_text.splitlines():
        match = _ROW_RE.search(line)
        if match is None:
            continue
        param_id = match.group(2)
        description = _COL_GAP_RE.split(match.group(3).strip())[0]
        params[param_id] = "_".join(description.split())
    return params


def extract_calmorph_parameters(table1_pdf: str | Path) -> dict[str, str]:
    """Reconstruct the CalMorph parameter schema from the SI table1 PDF.

    Requires a born-digital PDF -- the recipe trusts the text layer and does not
    fall back to OCR here (scanned input should be routed through the VLM path
    upstream). Raises ``ValueError`` if the PDF is not born-digital or yields no
    parameters.

    Args:
        table1_pdf: Path to "SI - table1 parameter description.pdf".

    Returns:
        Mapping of all CalMorph parameter ids to descriptions.
    """
    table1_pdf = Path(table1_pdf)
    kind = pdf_kind(table1_pdf)
    if kind != "born_digital":
        raise ValueError(
            f"{table1_pdf.name} is {kind!r}; this recipe needs a born-digital PDF "
            "with a trustworthy text layer (route scans through OCR)."
        )
    params = parse_calmorph_table(pdf_text(table1_pdf, layout=True))
    if not params:
        raise ValueError(f"No CalMorph parameters parsed from {table1_pdf.name}")
    return params
