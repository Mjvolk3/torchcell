# experiments/010-kuzmin-tmi/scripts/dango_full_dataset_compose_figure.py
# [[experiments.010-kuzmin-tmi.scripts.dango_full_dataset_compose_figure]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/dango_full_dataset_compose_figure
"""Compose the DANGO full-dataset SI figure as a draw.io file from the true-size panel SVGs.

Panels a--d come from ``dango_full_dataset_si.py`` (this folder): (a) validation curves and
(b) best validation Pearson per run at half width; (c) the convergence epochs and (d) the
data-effect panel (Kuzmin 2018-only build against the experiment-006 build, by STRING
release), both at third width. Panels e--g come from ``dango_full_dataset_panels_efg.py``,
which reads the predictions and channel weights ``dango_full_dataset_inference.py`` scored
from the DANGO checkpoints: (e) predicted against measured tau completes row 2 at third
width; (f) absolute error against |tau| (DANGO beside the CGT, each on its own validation
split) and (g) the per-gene meta-embedding channel weights (STRING v9.1 against v12.0)
share row 3 at half width. Each panel SVG declares its size in draw.io units (100 per
inch, via ``savefig_true_size_svg``), so the image cells are placed at exact physical size
and the figure is WYSIWYG when ``make -C paper/nature-biotech fig`` exports it to
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
    curves = osp.join(P010, "dango_full_dataset_curves.svg")
    best = osp.join(P010, "dango_full_dataset_best.svg")
    conv = osp.join(P010, "dango_full_dataset_convergence.svg")
    effect = osp.join(P010, "dango_full_dataset_data_effect.svg")
    pred = osp.join(P010, "dango_full_dataset_pred_vs_measured.svg")
    error = osp.join(P010, "dango_full_dataset_error_vs_tau.svg")
    weights = osp.join(P010, "dango_full_dataset_channel_weights.svg")

    half = svg_size(curves)[0]
    third = svg_size(conv)[0]
    for path in (effect, pred):
        if abs(svg_size(path)[0] - third) > 0.5:
            raise SystemExit(f"{path} must be third width to tile row 2 with c, d and e")
    for path in (best, error, weights):
        if abs(svg_size(path)[0] - half) > 0.5:
            raise SystemExit(f"{path} must be half width to tile its row")
    col2 = half + COL_GAP
    # Row 1: (a) curves and (b) best per run, two half-width panels.
    row1_top = 0
    y1 = row1_top + TOP_STRIP
    row1_h = max(svg_size(curves)[1], svg_size(best)[1])
    # Row 2: (c) convergence, (d) the data effect, (e) predicted vs measured, three thirds.
    row2_top = y1 + row1_h + (ROW_GAP - TOP_STRIP)
    y2 = row2_top + TOP_STRIP
    row2_h = max(svg_size(conv)[1], svg_size(effect)[1], svg_size(pred)[1])
    x_effect = third + COL_GAP
    x_e = 2 * (third + COL_GAP)
    # Row 3: (f) error against |tau| and (g) channel weights, two half-width panels.
    row3_top = y2 + row2_h + (ROW_GAP - TOP_STRIP)
    y3 = row3_top + TOP_STRIP
    row3_h = max(svg_size(error)[1], svg_size(weights)[1])

    cells, cid = [], 2

    def add(cell_xml: str, letter: str, x: float, row_top: float):
        nonlocal cid
        cells.append(cell_xml)
        cells.append(letter_cell(cid + 1, letter, x, row_top))
        cid += 2

    add(image_cell(cid, curves, 0, y1), "a", 0, row1_top)
    add(image_cell(cid, best, col2, y1), "b", col2, row1_top)
    add(image_cell(cid, conv, 0, y2), "c", 0, row2_top)
    add(image_cell(cid, effect, x_effect, y2), "d", x_effect, row2_top)
    add(image_cell(cid, pred, x_e, y2), "e", x_e, row2_top)
    add(image_cell(cid, error, 0, y3), "f", 0, row3_top)
    add(image_cell(cid, weights, col2, y3), "g", col2, row3_top)

    extent_w = max(col2 + svg_size(best)[0], x_e + third, col2 + svg_size(weights)[0])
    extent_h = y3 + row3_h
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
