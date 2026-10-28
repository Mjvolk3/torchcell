# experiments/010-kuzmin-tmi/scripts/dango_full_dataset_compose_figure.py
# [[experiments.010-kuzmin-tmi.scripts.dango_full_dataset_compose_figure]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/dango_full_dataset_compose_figure
"""Compose the DANGO full-dataset SI figure as a draw.io file from the true-size panel SVGs.

A 2 x 3 grid of six third-width panels (57.8 mm each, 52 mm tall). Panels a--d come from
``dango_full_dataset_si.py`` (this folder): (a) validation curves, (b) best validation
Pearson per run, (c) the convergence epochs, (d) the data-effect panel (Kuzmin 2018-only
build against the experiment-006 build, by STRING release). Panels e and f come from
``dango_full_dataset_panels_ef.py``, which reads the predictions and channel weights
``dango_full_dataset_inference.py`` scored from the DANGO checkpoints: (e) predicted against
measured tau and (f) the per-gene meta-embedding channel weights (STRING v9.1 against
v12.0). Row 1 is a, b, c; row 2 is d, e, f. The script refuses any panel that is not third
width. Each panel SVG declares its size in draw.io units (100 per inch, via
``savefig_true_size_svg``), so the image cells are placed at exact physical size and the
figure is WYSIWYG when ``make -C paper/nature-biotech fig`` exports it to
``figures/FigS-dango-full-dataset.pdf``.

Panel letters are 8 pt bold lowercase (draw.io fontSize 11.1). Rerun after regenerating any
panel; the draw.io file is overwritten, never edited by hand.

Layout convention shared by every composed SI figure (the "white cross"): COL_GAP = 12
units (3 mm) between columns, ROW_GAP = 22 units (5.5 mm) between rows, a TOP_STRIP of 16
units above every row, and each panel letter in that strip at the panel's top-left
(x = panel_x, y = row_top), so no letter sits over an axis label or a neighbor's title.

Run from the repo root:
    python experiments/010-kuzmin-tmi/scripts/dango_full_dataset_compose_figure.py
"""

import base64
import os
import os.path as osp
import re
from xml.sax.saxutils import quoteattr

from dotenv import load_dotenv

load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
DRAWIO_DIR = "notes/assets/drawio"
FIG_NAME = "FigS-dango-full-dataset"

FULL_WIDTH = 709  # 180 mm in draw.io units (cap)
MAX_HEIGHT = 669  # 170 mm
COL_GAP = 12  # 3 mm between columns
ROW_GAP = 22  # 5.5 mm between rows; the next row's TOP_STRIP is the lower part of it
TOP_STRIP = 16  # the letter strip above every row
LETTER_W, LETTER_H = 18, 14
LETTER_STYLE = (
    "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=top;"
    "whiteSpace=wrap;rounded=0;fontFamily=Arial;fontSize=11.1;fontStyle=1;"
)
IMAGE_STYLE = (
    "shape=image;verticalLabelPosition=bottom;labelBackgroundColor=default;"
    "verticalAlign=top;aspect=fixed;imageAspect=0;image=data:image/svg+xml,{b64};"
)
P010 = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")
THIRD_WIDTH_MM = 57.8  # torchcell.utils.PANEL_WIDTHS_MM["third"]
UNITS_PER_MM = 100 / 25.4  # draw.io canvas units (100 per inch)


def svg_size(path: str) -> tuple[float, float]:
    head = open(path, encoding="utf-8").read(2000)
    w = float(re.search(r'width="([0-9.]+)"', head).group(1))
    h = float(re.search(r'height="([0-9.]+)"', head).group(1))
    return w, h


def image_cell(cid: int, path: str, x: float, y: float) -> str:
    w, h = svg_size(path)
    b64 = base64.b64encode(open(path, "rb").read()).decode("ascii")
    style = IMAGE_STYLE.format(b64=b64)
    return (
        f'<mxCell id="{cid}" value="" style={quoteattr(style)} vertex="1" parent="1">'
        f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.4f}" height="{h:.4f}" as="geometry"/></mxCell>'
    )


def letter_cell(cid: int, letter: str, x: float, row_top: float) -> str:
    """Panel letter in the TOP_STRIP above the panel, flush with the panel's left edge."""
    return (
        f'<mxCell id="{cid}" value="{letter}" style={quoteattr(LETTER_STYLE)} vertex="1" parent="1">'
        f'<mxGeometry x="{x:.1f}" y="{row_top:.1f}" width="{LETTER_W}" height="{LETTER_H}" as="geometry"/></mxCell>'
    )


def main():
    rows = [
        [("a", "dango_full_dataset_curves.svg"),
         ("b", "dango_full_dataset_best.svg"),
         ("c", "dango_full_dataset_convergence.svg")],
        [("d", "dango_full_dataset_data_effect.svg"),
         ("e", "dango_full_dataset_pred_vs_measured.svg"),
         ("f", "dango_full_dataset_channel_weights.svg")],
    ]
    third = THIRD_WIDTH_MM * UNITS_PER_MM
    for row in rows:
        for _, name in row:
            w = svg_size(osp.join(P010, name))[0]
            if abs(w - third) > 0.5:
                raise SystemExit(f"{name} is {w:.2f} units wide; every panel must be third width ({third:.2f})")

    cells, cid = [], 2
    row_top = 0.0
    extent_w = extent_h = 0.0
    for row in rows:
        y = row_top + TOP_STRIP
        row_h = max(svg_size(osp.join(P010, name))[1] for _, name in row)
        for col, (letter, name) in enumerate(row):
            x = col * (third + COL_GAP)
            cells.append(image_cell(cid, osp.join(P010, name), x, y))
            cells.append(letter_cell(cid + 1, letter, x, row_top))
            cid += 2
            extent_w = max(extent_w, x + svg_size(osp.join(P010, name))[0])
        extent_h = y + row_h
        row_top = y + row_h + (ROW_GAP - TOP_STRIP)

    xml = (
        '<mxfile host="dango_full_dataset_compose_figure.py">'
        f'<diagram id="{FIG_NAME}" name="Page-1">'
        '<mxGraphModel dx="0" dy="0" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" '
        'arrows="1" fold="1" page="1" pageScale="1" pageWidth="850" pageHeight="1100" math="0" shadow="0">'
        '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>"
    )
    out = osp.join(DRAWIO_DIR, f"{FIG_NAME}.drawio")
    with open(out, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"wrote {out}: {extent_w:.0f} x {extent_h:.0f} units = {extent_w / 3.937:.1f} x {extent_h / 3.937:.1f} mm")
    if extent_w > FULL_WIDTH + 8 or extent_h > MAX_HEIGHT + 8:
        raise SystemExit(f"{FIG_NAME} exceeds the Nature print box (709 x 669 units)")


if __name__ == "__main__":
    main()
