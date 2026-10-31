# paper/nature-biotech/scripts/compose_cgt_architecture_si.py
# [[paper.nature-biotech.scripts.compose_cgt_architecture_si]]
# https://github.com/Mjvolk3/torchcell/tree/main/paper/nature-biotech/scripts/compose_cgt_architecture_si
"""Compose the Cell Graph Transformer architecture diagrams into the SI figure sheets.

The panels are the mermaid renders of three notes, each a function view of one model (every
box names its inputs, its map, its outputs and its parameter count), rendered by
``bash notes/assets/publish/scripts/mermaid_pdf.sh <note>``:

* ``notes/experiments.025-solid-growth.architecture-mermaid.md`` -- the trigenic model
  (``-> experiments.025-solid-growth.architecture-mermaid.pdf``);
* ``notes/experiments.030-solid-growth-multi.architecture-mermaid.md`` -- the multi-measurement
  model with the experiment ontology as its prior;
* ``notes/experiments.028-gene-essentiality.architecture-mermaid.md`` -- the essentiality CGT
  (``-1.pdf``) and the GCN (``-2.pdf``), the pair of the thesis sheet
  ``fig-ch03-essentiality-architecture.pdf``.

Each panel is scaled so that its dominant text size prints at ``--text-pt`` (6 pt, Nature's
default figure text); a multi-panel sheet puts its panels side by side under one shared scale
with an 8 pt bold lowercase letter at each panel's outer top-left, shrinking the shared scale
only when the row would exceed the sheet's width. The script refuses a sheet whose text would
print below 5 pt or whose height exceeds its ``max_height_mm``, rather than shrinking silently.
Sheets are written at true size into ``paper/nature-biotech/figures/``.

Run from the repo root (pymupdf lives in the thesis env)::

    ~/miniconda3/envs/thesis/bin/python paper/nature-biotech/scripts/compose_cgt_architecture_si.py
"""

from __future__ import annotations

import argparse
import os.path as osp
from dataclasses import dataclass

import pymupdf

HERE = osp.dirname(osp.abspath(__file__))
ROOT = osp.abspath(osp.join(HERE, "..", "..", ".."))
PDF_OUT = osp.join(ROOT, "notes", "assets", "pdf-output")
FIGURES = osp.join(ROOT, "paper", "nature-biotech", "figures")
MM = 72 / 25.4


@dataclass(frozen=True)
class Sheet:
    """One output sheet: its panels (render stems) and its size limits."""

    name: str
    stems: tuple[str, ...]
    max_width_mm: float
    max_height_mm: float
    lettered: bool


SHEETS = (
    Sheet(
        "SI-cgt-trigenic-architecture",
        ("experiments.025-solid-growth.architecture-mermaid",),
        180.0,
        170.0,
        False,
    ),
    Sheet(
        "SI-cgt-ontology-architecture",
        ("experiments.030-solid-growth-multi.architecture-mermaid",),
        180.0,
        170.0,
        False,
    ),
    # The thesis sheet (165 mm, two panels under one scale): the same panels at the same
    # width so the manuscript and the dissertation show one figure.
    Sheet(
        "SI-cgt-essentiality-architecture",
        (
            "experiments.028-gene-essentiality.architecture-mermaid-1",
            "experiments.028-gene-essentiality.architecture-mermaid-2",
        ),
        165.0,
        170.0,
        True,
    ),
)


def dominant_text_pt(page: pymupdf.Page) -> float:
    """The font size carrying the most characters on the page."""
    sizes: dict[float, int] = {}
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                key = round(span["size"], 2)
                sizes[key] = sizes.get(key, 0) + len(span["text"])
    return max(sizes, key=lambda k: sizes[k])


def compose(sheet: Sheet, text_pt: float, gap_mm: float, letter_pt: float) -> str:
    """Write one sheet and return its path."""
    docs = [pymupdf.open(osp.join(PDF_OUT, f"{stem}.pdf")) for stem in sheet.stems]
    pages = [d[0] for d in docs]
    doms = [dominant_text_pt(p) for p in pages]
    scale = [text_pt / d for d in doms]

    gap = gap_mm * MM
    letter_w = (4.0 * MM) if sheet.lettered else 0.0
    widths = [p.rect.width * s for p, s in zip(pages, scale, strict=True)]
    heights = [p.rect.height * s for p, s in zip(pages, scale, strict=True)]
    total_w = sum(widths) + letter_w * len(pages) + gap * (len(pages) - 1)
    if total_w > sheet.max_width_mm * MM:
        # Every panel shares one scale so its text prints at one size; shrink to fit.
        f = sheet.max_width_mm * MM / total_w
        scale = [s * f for s in scale]
        widths = [w * f for w in widths]
        heights = [h * f for h in heights]
        total_w = sheet.max_width_mm * MM
    printed_pt = [s * d for s, d in zip(scale, doms, strict=True)]
    assert min(printed_pt) >= 5.0 - 1e-6, (
        f"{sheet.name}: text would print at {min(printed_pt):.2f} pt (floor 5 pt)"
    )
    top = 1.5 * MM if sheet.lettered else 0.0
    total_h = max(heights) + top
    assert total_h <= sheet.max_height_mm * MM, (
        f"{sheet.name}: {total_h / MM:.1f} mm tall exceeds {sheet.max_height_mm} mm"
    )

    out = pymupdf.open()
    page = out.new_page(width=total_w, height=total_h)
    x = 0.0
    for k, (src, w, h) in enumerate(zip(pages, widths, heights, strict=True)):
        if sheet.lettered:
            page.insert_text(
                (x, top + letter_pt * 0.75), "ab"[k], fontsize=letter_pt, fontname="hebo"
            )
            x += letter_w
        page.show_pdf_page(pymupdf.Rect(x, top, x + w, top + h), src.parent, 0)
        x += w + gap
    dst = osp.join(FIGURES, f"{sheet.name}.pdf")
    out.save(dst)
    print(
        f"{sheet.name}: {total_w / MM:.1f} x {total_h / MM:.1f} mm, text "
        f"{', '.join(f'{p:.2f}' for p in printed_pt)} pt, panels "
        f"{[f'{w / MM:.0f}x{h / MM:.0f} mm' for w, h in zip(widths, heights, strict=True)]}"
        f" -> {dst}"
    )
    return dst


def main() -> None:
    """Compose every sheet, or the one named by ``--only``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-pt", type=float, default=6.0)
    parser.add_argument("--gap-mm", type=float, default=4.0)
    parser.add_argument("--letter-pt", type=float, default=8.0)
    parser.add_argument(
        "--only", default=None, help="compose one sheet by name instead of all"
    )
    args = parser.parse_args()
    for sheet in SHEETS:
        if args.only and sheet.name != args.only:
            continue
        compose(sheet, args.text_pt, args.gap_mm, args.letter_pt)


if __name__ == "__main__":
    main()
