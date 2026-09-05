# experiments/006-kuzmin-tmi/scripts/dcell_training_compose_figure.py
# [[experiments.006-kuzmin-tmi.scripts.dcell_training_compose_figure]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/006-kuzmin-tmi/scripts/dcell_training_compose_figure
"""Compose the DCell-training SI figure as a draw.io file from the true-size panel SVGs.

Panels come from ``dcell_training_wandb.py`` (this folder). Each panel SVG declares its
size in draw.io units (100 per inch, via ``savefig_true_size_svg``), so the image cells
are placed at exact physical size and the figure is WYSIWYG when exported to
``paper/nature-biotech/figures/FigS-dcell-training.pdf``.

Layout (180 mm wide, three rows of two half-width panels): (a) validation Pearson vs
epoch, (b) losses; (c) cost comparison, (d) speed-up stages as a table with bars; (e) the
CPU per-phase profile of one training step from ``dcell_training_cpu_profile.py``, (f) best
validation Pearson on the Kuzmin 2018-only build against the experiment-006 build.

Layout convention shared by every composed SI figure (the "white cross"): COL_GAP = 12
units (3 mm) between columns, ROW_GAP = 22 units (5.5 mm) between rows, a TOP_STRIP of 16
units above every row, and each panel letter in that strip at the panel's top-left
(x = panel_x, y = row_top), so no letter sits over an axis label or a neighbor's title.

Panel letters are 8 pt bold lowercase (draw.io fontSize 11.1). Rerun after regenerating
any panel; the draw.io file is overwritten, never edited by hand. Export with

    "/Applications/draw.io.app/Contents/MacOS/draw.io" -x -f pdf --crop \
        -o paper/nature-biotech/figures/FigS-dcell-training.pdf \
        notes/assets/drawio/FigS-dcell-training.drawio

Run from the repo root:
    python experiments/006-kuzmin-tmi/scripts/dcell_training_compose_figure.py
"""

import base64
import os
import os.path as osp
import re
from xml.sax.saxutils import quoteattr

from dotenv import load_dotenv

load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
assert ASSET_IMAGES_DIR is not None, "ASSET_IMAGES_DIR must be set in the environment"

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(SCRIPT_DIR)))
DRAWIO_DIR = osp.join(REPO_ROOT, "notes", "assets", "drawio")
PANELS = osp.join(ASSET_IMAGES_DIR, "006-kuzmin-tmi")

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
        ("ab", ["dcell_training_val_pearson", "dcell_training_loss"]),
        ("cd", ["dcell_training_cost", "dcell_training_stages"]),
        ("ef", ["dcell_training_cpu_profile", "dcell_training_data_effect"]),
    ]
    cells, cid = [], 2
    row_top, extent_w, extent_h = 0, 0.0, 0.0
    for letters, names in rows:
        y = row_top + TOP_STRIP
        x, row_h = 0.0, 0.0
        for letter, name in zip(letters, names):
            path = osp.join(PANELS, f"{name}.svg")
            w, h = svg_size(path)
            cells.append(image_cell(cid, path, x, y))
            cells.append(letter_cell(cid + 1, letter, x, row_top))
            cid += 2
            x += w + COL_GAP
            row_h = max(row_h, h)
        extent_w = max(extent_w, x - COL_GAP)
        extent_h = y + row_h
        row_top = extent_h + (ROW_GAP - TOP_STRIP)
    xml = (
        '<mxfile host="dcell_training_compose_figure.py">'
        '<diagram id="FigS-dcell-training" name="Page-1">'
        '<mxGraphModel dx="0" dy="0" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" '
        'arrows="1" fold="1" page="1" pageScale="1" pageWidth="850" pageHeight="1100" math="0" shadow="0">'
        '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>"
    )
    out = osp.join(DRAWIO_DIR, "FigS-dcell-training.drawio")
    with open(out, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"wrote {out}: {extent_w:.0f} x {extent_h:.0f} units = {extent_w / 3.937:.1f} x {extent_h / 3.937:.1f} mm")
    if extent_w > FULL_WIDTH + 8 or extent_h > MAX_HEIGHT + 8:
        raise SystemExit("FigS-dcell-training exceeds the Nature print box (709 x 669 units)")


if __name__ == "__main__":
    main()
