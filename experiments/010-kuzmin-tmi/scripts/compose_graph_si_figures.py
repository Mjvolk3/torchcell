# experiments/010-kuzmin-tmi/scripts/compose_graph_si_figures.py
# [[experiments.010-kuzmin-tmi.scripts.compose_graph_si_figures]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/compose_graph_si_figures
"""Compose the two gene--gene graph SI figures as draw.io files from the true-size panel SVGs.

Panels come from ``graph_statistics.py`` (this folder). Each panel SVG declares its size in
draw.io units (100 per inch, via ``savefig_true_size_svg``), so the image cells below are
placed at exact physical size and the figure is WYSIWYG when ``make -C paper/nature-biotech
fig`` exports it to ``figures/<name>.pdf``.

Figures written to notes/assets/drawio/, grouped by theme:
  FigS-graph-attention-priors.drawio     each graph on its own. Row 1: (a) sizes and the union
                                         row, (b) degree CCDF (half + half); row 2: (c)
                                         structure, (d) hubs (half + half); row 3: (e) other
                                         components of the cell representation (full)
  FigS-graph-attention-priors-2.drawio   how the graphs relate to each other. Row 1: (a)
                                         Jaccard, (b) containment, (c) shared pairs (three
                                         thirds); row 2: (d) edge multiplicity (third), (e) the
                                         two transcription-factor graphs (wide)
The STRING-release panel (graphs_string_releases.svg) is embedded by the DANGO reproduction
figure (experiments/005-kuzmin2018-tmi/scripts/compose_dango_si_figures.py), not here.

Layout rules (the "white cross"): every column is separated by COL_GAP and every row by
ROW_GAP of clear white, and each panel letter sits in the gutter above-left of its panel, at
(panel_x, panel_y - LETTER_STRIP), never over a neighbor's labels or over the panel's own
y-axis label. The first row gets a TOP_STRIP of the same depth. Rows are placed one after
another at equal gaps, so panels of one row must share a height (set in graph_statistics.py)
for their axes to align; heatmaps carry their column labels below the matrix so their top
edges align too. Panel letters are 8 pt bold lowercase (draw.io fontSize 11.1, the ladder
value). Rerun after regenerating any panel; the draw.io file is overwritten, never edited by
hand.

Paths are resolved from this file, so the script runs from any working directory:
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
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
DRAWIO_DIR = osp.join(REPO_ROOT, "notes/assets/drawio")

FULL_WIDTH = 709  # 180 mm in draw.io units (100 per inch)
MAX_HEIGHT = 669  # 170 mm
HEIGHT_GRACE = 8  # units of export rounding the size gate tolerates
COL_GAP = 12  # 3 mm of white between columns
ROW_GAP = 22  # 5.5 mm of white between rows; the next row's letters sit in it
TOP_STRIP = 16  # 4 mm above the first row for its letters
LETTER_STRIP = 16  # a letter's top edge sits this far above its panel's top edge
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


def letter_cell(cid: int, letter: str, x: float, y: float) -> str:
    return (
        f'<mxCell id="{cid}" value="{letter}" style={quoteattr(LETTER_STYLE)} vertex="1" parent="1">'
        f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{LETTER_W}" height="{LETTER_H}" as="geometry"/></mxCell>'
    )


def write_figure(name: str, layout: list[tuple[str, str, float, float]]):
    """layout: (letter, svg path, x, y) per panel; the letter sits LETTER_STRIP above the
    panel's top-left corner, inside the white gutter."""
    cells, cid = [], 2
    for letter, path, x, y in layout:
        cells.append(image_cell(cid, path, x, y))
        cells.append(letter_cell(cid + 1, letter, x, y - LETTER_STRIP))
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
    if extent_w > FULL_WIDTH + 2 or extent_h > MAX_HEIGHT + HEIGHT_GRACE:
        raise SystemExit(f"{name} exceeds the Nature print box ({FULL_WIDTH} x {MAX_HEIGHT} units)")


def rows(panel_rows: list[list[tuple[str, str]]]) -> list[tuple[str, str, float, float]]:
    """Place rows of (letter, svg) pairs left to right with COL_GAP between panels and
    ROW_GAP between rows, the first row TOP_STRIP below the top edge. Panels of one row are
    expected to share a height; the row advances by the tallest."""
    layout, y = [], float(TOP_STRIP)
    for r in panel_rows:
        x = 0.0
        for letter, p in r:
            layout.append((letter, p, x, y))
            x += svg_size(p)[0] + COL_GAP
        y += max(svg_size(p)[1] for _, p in r) + ROW_GAP
    return layout


def main():
    p = lambda n: osp.join(P010, f"{n}.svg")  # noqa: E731
    write_figure(
        "FigS-graph-attention-priors",
        rows(
            [
                [("a", p("graphs_sizes")), ("b", p("graphs_degree_ccdf"))],
                [("c", p("graphs_structure")), ("d", p("graphs_hubs"))],
                [("e", p("graphs_components"))],
            ]
        ),
    )
    write_figure(
        "FigS-graph-attention-priors-2",
        rows(
            [
                [("a", p("graphs_jaccard")), ("b", p("graphs_containment")), ("c", p("graphs_shared_pairs"))],
                [("d", p("graphs_edge_multiplicity")), ("e", p("graphs_tf_overlap"))],
            ]
        ),
    )


if __name__ == "__main__":
    main()
