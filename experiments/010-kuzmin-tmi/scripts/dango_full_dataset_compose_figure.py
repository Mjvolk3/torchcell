# experiments/010-kuzmin-tmi/scripts/dango_full_dataset_compose_figure.py
# [[experiments.010-kuzmin-tmi.scripts.dango_full_dataset_compose_figure]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/dango_full_dataset_compose_figure
"""Compose the DANGO full-dataset SI figure as a draw.io file from the true-size panel SVGs.

Panels a--c come from ``dango_full_dataset_si.py`` (this folder). Panels d--f need a trained
DANGO checkpoint and inference on the validation split, which cannot run on this machine;
each is reserved as an empty lettered box whose label states exactly what fills it. Each
panel SVG declares its size in draw.io units (100 per inch, via ``savefig_true_size_svg``),
so the image cells are placed at exact physical size and the figure is WYSIWYG when
``make -C paper/nature-biotech fig`` exports it to ``figures/FigS-dango-full-dataset.pdf``.

Panel letters are 8 pt bold lowercase (draw.io fontSize 11.1); placeholder labels are 6 pt
(fontSize 8.3). Rerun after regenerating any panel; the draw.io file is overwritten, never
edited by hand.

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
from xml.sax.saxutils import escape, quoteattr

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
PLACEHOLDER_STYLE = (
    "rounded=0;whiteSpace=wrap;html=1;dashed=1;strokeColor=#666666;fillColor=none;"
    "fontFamily=Arial;fontSize=8.3;fontColor=#666666;align=center;verticalAlign=middle;"
    "spacing=6;"
)

P010 = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")

# What each reserved panel needs, stated so it can be filled on the cluster later.
PLACEHOLDERS = {
    "d": (
        "[placeholder: predicted vs measured trigenic interaction on the validation split, "
        "DANGO STRING v9.1 run 014mprap. Needs the run's best checkpoint "
        "(DATA_ROOT/models/checkpoints/compute-3-3-1941704_ff85…/014mprap-best-*.ckpt), "
        "the 006 dataset LMDB, and experiments/006-kuzmin-tmi/scripts/dango.py with "
        "regression_task.execution_mode=inference]"
    ),
    "e": (
        "[placeholder: absolute error vs |τ| on the validation split, DANGO (014mprap) beside "
        "CGT (c7671wgj best-pearson-epoch=24 checkpoint). Needs both checkpoints, the 006 and "
        "010 dataset LMDBs, and the prediction dumps from dango.py inference and "
        "equivariant_cell_graph_transformer_eval.py]"
    ),
    "f": (
        "[placeholder: per-STRING-channel weight of DANGO's meta-embedding attention on the "
        "validation split, STRING v9.1 vs v12.0 checkpoints (014mprap, 9jpfy547). Needs both "
        "checkpoints and a forward hook on torchcell.models.dango.Dango meta-embedding]"
    ),
}


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


def placeholder_cell(cid: int, text: str, x: float, y: float, w: float, h: float) -> str:
    return (
        f'<mxCell id="{cid}" value={quoteattr(escape(text))} style={quoteattr(PLACEHOLDER_STYLE)} vertex="1" parent="1">'
        f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/></mxCell>'
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

    half = svg_size(curves)[0]
    col2 = half + COL_GAP
    row1_top = 0
    y1 = row1_top + TOP_STRIP
    row1_h = max(svg_size(curves)[1], svg_size(best)[1])
    row2_top = y1 + row1_h + (ROW_GAP - TOP_STRIP)
    y2 = row2_top + TOP_STRIP
    row2_h = svg_size(conv)[1]
    row3_top = y2 + row2_h + (ROW_GAP - TOP_STRIP)
    y3 = row3_top + TOP_STRIP
    row3_h = row2_h

    cells, cid = [], 2

    def add(cell_xml: str, letter: str, x: float, row_top: float):
        nonlocal cid
        cells.append(cell_xml)
        cells.append(letter_cell(cid + 1, letter, x, row_top))
        cid += 2

    add(image_cell(cid, curves, 0, y1), "a", 0, row1_top)
    add(image_cell(cid, best, col2, y1), "b", col2, row1_top)
    add(image_cell(cid, conv, 0, y2), "c", 0, row2_top)
    add(placeholder_cell(cid, PLACEHOLDERS["d"], col2, y2, half, row2_h), "d", col2, row2_top)
    add(placeholder_cell(cid, PLACEHOLDERS["e"], 0, y3, half, row3_h), "e", 0, row3_top)
    add(placeholder_cell(cid, PLACEHOLDERS["f"], col2, y3, half, row3_h), "f", col2, row3_top)

    extent_w = col2 + max(svg_size(best)[0], half)
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
