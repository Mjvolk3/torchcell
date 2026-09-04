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


def letter_cell(cid: int, letter: str, x: float, y: float) -> str:
    return (
        f'<mxCell id="{cid}" value="{letter}" style={quoteattr(LETTER_STYLE)} vertex="1" parent="1">'
        f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{LETTER_W}" height="{LETTER_H}" as="geometry"/></mxCell>'
    )


def main():
    curves = osp.join(P010, "dango_full_dataset_curves.svg")
    best = osp.join(P010, "dango_full_dataset_best.svg")
    conv = osp.join(P010, "dango_full_dataset_convergence.svg")

    half = svg_size(curves)[0]
    col2 = FULL_WIDTH - half  # right column flush with the 180 mm edge
    row1_h = max(svg_size(curves)[1], svg_size(best)[1])
    y2 = row1_h + GAP
    row2_h = svg_size(conv)[1]
    y3 = y2 + row2_h + GAP
    row3_h = row2_h

    cells, cid = [], 2

    def add(cell_xml: str, letter: str, x: float, y: float):
        nonlocal cid
        cells.append(cell_xml)
        cells.append(letter_cell(cid + 1, letter, x - 2, y - 4))
        cid += 2

    add(image_cell(cid, curves, 0, 0), "a", 0, 0)
    add(image_cell(cid, best, col2, 0), "b", col2, 0)
    add(image_cell(cid, conv, 0, y2), "c", 0, y2)
    add(placeholder_cell(cid, PLACEHOLDERS["d"], col2 + 14, y2 + 10, half - 14, row2_h - 10), "d", col2, y2)
    add(placeholder_cell(cid, PLACEHOLDERS["e"], 14, y3 + 10, half - 14, row3_h - 10), "e", 0, y3)
    add(placeholder_cell(cid, PLACEHOLDERS["f"], col2 + 14, y3 + 10, half - 14, row3_h - 10), "f", col2, y3)

    extent_w = FULL_WIDTH
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
    if extent_w > FULL_WIDTH + 2 or extent_h > MAX_HEIGHT + 8:
        raise SystemExit(f"{FIG_NAME} exceeds the Nature print box (709 x 669 units)")


if __name__ == "__main__":
    main()
