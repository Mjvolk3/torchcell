# experiments/010-kuzmin-tmi/scripts/compose_graph_si_figures.py
# [[experiments.010-kuzmin-tmi.scripts.compose_graph_si_figures]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/compose_graph_si_figures
"""Compose the two graph SI figures as draw.io files from the true-size panel SVGs.

Panels come from ``graph_statistics.py`` (this folder) and
``experiments/005-kuzmin2018-tmi/scripts/dango_string_version_sweep.py``. Each panel SVG
declares its size in draw.io units (100 per inch, via ``savefig_true_size_svg``), so the
image cells below are placed at exact physical size and the figure is WYSIWYG when
``make -C paper/nature-biotech fig`` exports it to ``figures/<name>.pdf``.

Figures written to notes/assets/drawio/:
  FigS-graph-attention-priors.drawio   six panels, 180 mm x <=170 mm
  FigS-dango-string-versions.drawio    two panels, 180 mm wide, one row

Panel letters are 8 pt bold lowercase (draw.io fontSize 11.1, the ladder value). Rerun after
regenerating any panel; the draw.io file is overwritten, never edited by hand.

Run from the repo root:
    python experiments/010-kuzmin-tmi/scripts/compose_graph_si_figures.py
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

FULL_WIDTH = 709  # 180 mm in draw.io units
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

P010 = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")
P005 = osp.join(ASSET_IMAGES_DIR, "005-kuzmin2018-tmi")


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


def write_figure(name: str, layout: list[tuple[str, str, float, float]]):
    """layout: (letter, svg path, x, y) per panel; the letter sits at the panel's top-left."""
    cells, cid = [], 2
    for letter, path, x, y in layout:
        cells.append(image_cell(cid, path, x, y))
        cells.append(letter_cell(cid + 1, letter, x - 2, y - 4))
        cid += 2
    extent_w = max(x + svg_size(p)[0] for _, p, x, _ in layout)
    extent_h = max(y + svg_size(p)[1] for _, p, _, y in layout)
    xml = (
        '<mxfile host="compose_graph_si_figures.py">'
        f'<diagram id="{name}" name="Page-1">'
        '<mxGraphModel dx="0" dy="0" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" '
        'arrows="1" fold="1" page="1" pageScale="1" pageWidth="850" pageHeight="1100" math="0" shadow="0">'
        '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>"
    )
    out = osp.join(DRAWIO_DIR, f"{name}.drawio")
    with open(out, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"wrote {out}: {extent_w:.0f} x {extent_h:.0f} units = {extent_w / 3.937:.1f} x {extent_h / 3.937:.1f} mm")
    if extent_w > FULL_WIDTH + 2 or extent_h > 669 + 8:
        raise SystemExit(f"{name} exceeds the Nature print box (709 x 669 units)")


def main():
    sizes = osp.join(P010, "graphs_sizes.svg")
    degree = osp.join(P010, "graphs_degree_ccdf.svg")
    jaccard = osp.join(P010, "graphs_jaccard.svg")
    contain = osp.join(P010, "graphs_containment.svg")
    mult = osp.join(P010, "graphs_edge_multiplicity.svg")
    strver = osp.join(P010, "graphs_string_versions.svg")
    dango = osp.join(P005, "dango_string_version_sweep.svg")

    half = svg_size(sizes)[0]
    col2 = FULL_WIDTH - half  # right column flush with the 180 mm edge
    row1_h = max(svg_size(sizes)[1], svg_size(degree)[1])
    y2 = row1_h + GAP
    row2_h = max(svg_size(jaccard)[1], svg_size(contain)[1])
    y3 = y2 + row2_h + GAP
    write_figure(
        "FigS-graph-attention-priors",
        [
            ("a", sizes, 0, 0),
            ("b", degree, col2, 0),
            ("c", jaccard, 0, y2),
            ("d", contain, col2, y2),
            ("e", mult, 0, y3),
        ],
    )
    write_figure(
        "FigS-dango-string-versions",
        [("a", strver, 0, 0), ("b", dango, col2, 0)],
    )


if __name__ == "__main__":
    main()
