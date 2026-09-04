# experiments/006-kuzmin-tmi/scripts/dcell_training_compose_figure.py
# [[experiments.006-kuzmin-tmi.scripts.dcell_training_compose_figure]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/006-kuzmin-tmi/scripts/dcell_training_compose_figure
"""Compose the DCell-training SI figure as a draw.io file from the true-size panel SVGs.

Panels come from ``dcell_training_wandb.py`` (this folder). Each panel SVG declares its
size in draw.io units (100 per inch, via ``savefig_true_size_svg``), so the image cells
are placed at exact physical size and the figure is WYSIWYG when exported to
``paper/nature-biotech/figures/FigS-dcell-training.pdf``.

Layout (180 mm wide): (a) validation Pearson vs epoch, (b) losses; (c) cost comparison,
(d) speed-up stages; (e) a lettered placeholder for the per-operation profiler breakdown
of a DCell training step, which needs a GPU run on the cluster.

Panel letters are 8 pt bold lowercase (draw.io fontSize 11.1); the placeholder label is
fontSize 8.3 (6 pt). Rerun after regenerating any panel; the draw.io file is overwritten,
never edited by hand. Export with

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
from xml.sax.saxutils import escape, quoteattr

from dotenv import load_dotenv

load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
assert ASSET_IMAGES_DIR is not None, "ASSET_IMAGES_DIR must be set in the environment"

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(SCRIPT_DIR)))
DRAWIO_DIR = osp.join(REPO_ROOT, "notes", "assets", "drawio")
PANELS = osp.join(ASSET_IMAGES_DIR, "006-kuzmin-tmi")

FULL_WIDTH = 709  # 180 mm in draw.io units
MAX_HEIGHT = 669  # 170 mm
GAP = 6
LETTER_W, LETTER_H = 18, 14
LETTER_STYLE = (
    "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=top;"
    "whiteSpace=wrap;rounded=0;fontFamily=Arial;fontSize=11.1;fontStyle=1;"
)
IMAGE_STYLE = (
    "shape=image;verticalLabelPosition=bottom;labelBackgroundColor=default;"
    "verticalAlign=top;aspect=fixed;imageAspect=0;image=data:image/svg+xml,{b64};"
)
PLACEHOLDER_STYLE = (
    "rounded=0;whiteSpace=wrap;html=1;strokeColor=#666666;fillColor=none;dashed=1;"
    "fontFamily=Arial;fontSize=8.3;fontColor=#666666;align=center;verticalAlign=middle;"
)
PLACEHOLDER_TEXT = (
    "[placeholder: per-operation profiler breakdown of one DCell training step "
    "(per-term subsystem forward, gene-state gather, backward, optimizer, DataLoader wait); "
    "requires a torch.profiler run of dcell.py on a cluster GPU node]"
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


def letter_cell(cid: int, letter: str, x: float, y: float) -> str:
    return (
        f'<mxCell id="{cid}" value="{letter}" style={quoteattr(LETTER_STYLE)} vertex="1" parent="1">'
        f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{LETTER_W}" height="{LETTER_H}" as="geometry"/></mxCell>'
    )


def placeholder_cell(cid: int, text: str, x: float, y: float, w: float, h: float) -> str:
    return (
        f'<mxCell id="{cid}" value={quoteattr(escape(text))} style={quoteattr(PLACEHOLDER_STYLE)} vertex="1" parent="1">'
        f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/></mxCell>'
    )


def main():
    a = osp.join(PANELS, "dcell_training_val_pearson.svg")
    b = osp.join(PANELS, "dcell_training_loss.svg")
    c = osp.join(PANELS, "dcell_training_cost.svg")
    d = osp.join(PANELS, "dcell_training_stages.svg")
    half_w = svg_size(a)[0]
    col2 = FULL_WIDTH - half_w  # right column flush with the 180 mm edge
    row1_h = max(svg_size(a)[1], svg_size(b)[1])
    y2 = row1_h + GAP
    row2_h = max(svg_size(c)[1], svg_size(d)[1])
    y3 = y2 + row2_h + GAP
    ph_h = 70

    cells, cid = [], 2
    for letter, path, x, y in [("a", a, 0, 0), ("b", b, col2, 0), ("c", c, 0, y2), ("d", d, col2, y2)]:
        cells.append(image_cell(cid, path, x, y))
        cells.append(letter_cell(cid + 1, letter, x - 2, y - 4))
        cid += 2
    cells.append(placeholder_cell(cid, PLACEHOLDER_TEXT, LETTER_W, y3, FULL_WIDTH - LETTER_W, ph_h))
    cells.append(letter_cell(cid + 1, "e", -2, y3 - 4))

    extent_h = y3 + ph_h
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
    print(f"wrote {out}: {FULL_WIDTH} x {extent_h:.0f} units = {FULL_WIDTH / 3.937:.1f} x {extent_h / 3.937:.1f} mm")
    if extent_h > MAX_HEIGHT + 8:
        raise SystemExit("FigS-dcell-training exceeds the Nature print box (709 x 669 units)")


if __name__ == "__main__":
    main()
