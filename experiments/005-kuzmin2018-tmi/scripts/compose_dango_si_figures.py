# experiments/005-kuzmin2018-tmi/scripts/compose_dango_si_figures.py
# [[experiments.005-kuzmin2018-tmi.scripts.compose_dango_si_figures]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/005-kuzmin2018-tmi/scripts/compose_dango_si_figures
"""Compose the DANGO reproduction SI figure as a draw.io file.

Panels, in reading order:

  a  STRING release drift per channel (``graphs_string_releases.svg`` from
     ``experiments/010-kuzmin-tmi/scripts/graph_statistics.py``): pairs in each release,
     retained / added / dropped. It motivates DANGO's rule of setting the zero weight
     ``lambda_k`` from the change between releases.
  b  DANGO in the manuscript's notation, authored here as draw.io cells at half width: one
     box per stage with a plain-text heading (Arial, ladder size 8.3) and one line of real
     LaTeX typeset by MathJax (``math="1"`` on the model, ``$$...$$`` labels, fontSize 7,
     which MathJax renders at about 6 pt; see the style guide). Explanations live in the
     caption, not in the boxes.
  c  ``dango_decreased_zeros.svg`` from ``dango_construction_si.py`` (decreased zeros and
     lambda per channel).
  d  ``dango_string_version_sweep.svg`` and
  e  ``dango_string_version_curves.svg`` from ``dango_string_version_sweep.py``.

Each SVG declares its size in draw.io units (100 per inch), so the image cells are placed
at exact physical size and the figure is WYSIWYG when ``make -C paper/nature-biotech fig``
exports it. Figure written to notes/assets/drawio/FigS-dango-reproduction.drawio (<= 709 x
669 units = 180 x 170 mm). Rerun after regenerating any panel; the file is overwritten,
never edited by hand.

Layout convention shared by every composed SI figure (the "white cross"): COL_GAP = 12
units (3 mm) between columns, ROW_GAP = 22 units (5.5 mm) between rows, a TOP_STRIP of 16
units above every row, and each panel letter in that strip at the panel's top-left
(x = panel_x, y = row_top), so no letter sits over an axis label or a neighbor's title.

Export (also done by ``make -C paper/nature-biotech fig``):
    /Applications/draw.io.app/Contents/MacOS/draw.io -x -f pdf --crop \\
        -o paper/nature-biotech/figures/FigS-dango-reproduction.pdf \\
        notes/assets/drawio/FigS-dango-reproduction.drawio

Run from the repo root:
    python experiments/005-kuzmin2018-tmi/scripts/compose_dango_si_figures.py
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
OUT = osp.join(DRAWIO_DIR, "FigS-dango-reproduction.drawio")

FULL_WIDTH = 705  # two 88 mm columns + COL_GAP; draw.io adds a 1-unit border per side on export
MAX_HEIGHT = 669
COL_GAP = 12  # 3 mm between columns
ROW_GAP = 22  # 5.5 mm between rows; the next row's TOP_STRIP is the lower part of it
TOP_STRIP = 16  # the letter strip above every row
LETTER_W, LETTER_H = 18, 14
BODY = 8.3  # ladder value: prints at 5.98 pt
LETTER = 11.1  # ladder value: prints at 7.99 pt, panel letters only
MATH = 7  # MathJax renders ~1.19x the cell size: 7 units -> ~6 pt on the page (measured)

# Palette (slots 1-6): stroke / fill.
ORANGE = ("#D79B00", "#FFE6CC")
RED = ("#B85450", "#F8CECC")
PURPLE = ("#9673A6", "#E1D5E7")
YELLOW = ("#D6B656", "#FFF2CC")
BLUE = ("#6C8EBF", "#DAE8FC")
GRAY = ("#666666", "#F5F5F5")

LETTER_STYLE = (
    "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=top;"
    f"whiteSpace=wrap;rounded=0;fontFamily=Arial;fontSize={LETTER};fontStyle=1;"
)
IMAGE_STYLE = (
    "shape=image;verticalLabelPosition=bottom;labelBackgroundColor=default;"
    "verticalAlign=top;aspect=fixed;imageAspect=0;image=data:image/svg+xml,{b64};"
)
P005 = osp.join(ASSET_IMAGES_DIR, "005-kuzmin2018-tmi")
P010 = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")


class Cells:
    def __init__(self):
        self.cells: list[str] = []
        self.n = 2

    def _id(self) -> int:
        self.n += 1
        return self.n

    def image(self, path: str, x: float, y: float) -> tuple[float, float]:
        w, h = svg_size(path)
        b64 = base64.b64encode(open(path, "rb").read()).decode("ascii")
        style = IMAGE_STYLE.format(b64=b64)
        self.cells.append(
            f'<mxCell id="{self._id()}" value="" style={quoteattr(style)} vertex="1" parent="1">'
            f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.4f}" height="{h:.4f}" as="geometry"/></mxCell>'
        )
        return w, h

    def letter(self, letter: str, x: float, row_top: float):
        """Panel letter in the TOP_STRIP above the panel, flush with the panel's left edge."""
        self.cells.append(
            f'<mxCell id="{self._id()}" value="{letter}" style={quoteattr(LETTER_STYLE)} vertex="1" parent="1">'
            f'<mxGeometry x="{x:.1f}" y="{row_top:.1f}" width="{LETTER_W}" height="{LETTER_H}" as="geometry"/></mxCell>'
        )

    def box(self, x, y, w, h, heading: str, color=GRAY):
        """A stage box: bold plain-text heading at the top-left; the math line is added separately."""
        stroke, fill = color
        style = (
            f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
            f"fontFamily=Arial;fontSize={BODY};fontStyle=1;align=left;verticalAlign=top;"
            f"spacingLeft=3;spacingRight=3;spacingTop=0;strokeWidth=0.75;"
        )
        self.cells.append(
            f'<mxCell id="{self._id()}" value={quoteattr(heading)} style={quoteattr(style)} vertex="1" parent="1">'
            f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/></mxCell>'
        )

    def math(self, latex: str, x, y, w, h, align="left"):
        """A LaTeX label typeset by MathJax (requires math="1" on the model)."""
        style = (
            f"text;html=1;strokeColor=none;fillColor=none;align={align};verticalAlign=middle;"
            f"whiteSpace=wrap;rounded=0;fontFamily=Arial;fontSize={MATH};spacing=0;spacingLeft=4;"
        )
        self.cells.append(
            f'<mxCell id="{self._id()}" value={quoteattr("$$" + latex + "$$")} style={quoteattr(style)} vertex="1" parent="1">'
            f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/></mxCell>'
        )

    def arrow(self, x1, y1, x2, y2, color="#666666"):
        style = f"endArrow=classic;html=1;strokeWidth=0.75;strokeColor={color};endSize=3;"
        self.cells.append(
            f'<mxCell id="{self._id()}" style={quoteattr(style)} edge="1" parent="1">'
            f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x1:.1f}" y="{y1:.1f}" as="sourcePoint"/>'
            f'<mxPoint x="{x2:.1f}" y="{y2:.1f}" as="targetPoint"/></mxGeometry></mxCell>'
        )


def svg_size(path: str) -> tuple[float, float]:
    head = open(path, encoding="utf-8").read(2000)
    w = float(re.search(r'width="([0-9.]+)"', head).group(1))
    h = float(re.search(r'height="([0-9.]+)"', head).group(1))
    return w, h


def schematic(c: Cells, x0: float, y0: float, w: float, h: float):
    """DANGO as a vertical pipeline of stage boxes at half width: heading + one math line each.

    Six rows of equal height fill exactly (w x h); rows B and E split into two boxes so the
    branch (encoder -> reconstruction head; readout -> loss) reads left to right.
    """
    n_rows, gap = 6, 6.0
    bh = (h - (n_rows - 1) * gap) / n_rows  # box height (29.1 at the 204.7-unit panel height)
    head_h = 13.0  # heading line; the math line is centered in the rest of the box
    math_y, math_h = head_h, bh - head_h

    def stage(x, y, bw, heading, latex, color):
        c.box(x, y, bw, bh, heading, color)
        c.math(r"\textstyle " + latex, x, y + math_y, bw, math_h)

    rows = [y0 + i * (bh + gap) for i in range(n_rows)]
    mid = x0 + w / 2
    # Row A: the six channels as adjacencies.
    stage(x0, rows[0], w, "Six STRING channels as message-passing edges",
          r"A^{(k)}\in\{0,1\}^{N\times N},\quad k=1,\dots,6,\quad N=6607", ORANGE)
    # Row B: per-channel encoder (left) and reconstruction head (right).
    wl = 196.0
    wr = w - wl - gap
    stage(x0, rows[1], wl, "Channel encoder, two GraphSAGE layers",
          r"H^{(k)}=F^{(k)}(\mathbf{E},A^{(k)}),\quad \mathbf{E}\in\mathbb{R}^{N\times d}", ORANGE)
    stage(x0 + wl + gap, rows[1], wr, "Reconstruction head",
          r"\hat A^{(k)}=H^{(k)}W_k,\quad \mathcal{L}_{\mathrm{rec}}(\lambda_k)", RED)
    # Row C: meta-embedding.
    stage(x0, rows[2], w, "Meta-embedding merges the six channels",
          r"h_i=\sum\nolimits_k a_{ik}\,h_i^{(k)},\quad a_{i:}=\operatorname{softmax}_k\,\mathrm{MLP}(h_i^{(k)}),\quad H=(h_1,\dots,h_N)", PURPLE)
    # Row D: perturbation as row selection.
    stage(x0, rows[3], w, "Perturbation selects rows; no perturbation operator",
          r"p=\{g_i,g_j,g_k\}\ \mapsto\ (h_i,h_j,h_k),\quad H\ \text{unchanged}", YELLOW)
    # Row E: Hyper-SAGNN readout (left) and interaction loss (right).
    wl2 = 176.0
    wr2 = w - wl2 - gap
    stage(x0, rows[4], wl2, "Hyper-SAGNN readout",
          r"\hat y_{\mathrm{int}}=|p|^{-1}\sum\nolimits_{t\in p}\mathrm{FC}\big((d_t-s_t)^{2}\big)", BLUE)
    stage(x0 + wl2 + gap, rows[4], wr2, "Interaction loss",
          r"\mathcal{L}_{\mathrm{int}}=\operatorname{mean}\log\cosh(\hat y_{\mathrm{int}}-y_{\mathrm{int}})", RED)
    # Row F: the scheduled objective.
    stage(x0, rows[5], w, "Objective; the pretraining weight follows one of three schedules over epochs 1 to 10",
          r"\mathcal{L}=\alpha_e\,\mathcal{L}_{\mathrm{rec}}+(1-\alpha_e)\,\mathcal{L}_{\mathrm{int}}", GRAY)
    # Flow arrows: down the left column through the gaps, and across the two split rows.
    xa = x0 + wl / 2
    c.arrow(xa, rows[0] + bh, xa, rows[1])
    c.arrow(x0 + wl, rows[1] + bh / 2, x0 + wl + gap, rows[1] + bh / 2)
    c.arrow(xa, rows[1] + bh, xa, rows[2])
    c.arrow(mid, rows[2] + bh, mid, rows[3])
    c.arrow(mid, rows[3] + bh, mid, rows[4])
    c.arrow(x0 + wl2, rows[4] + bh / 2, x0 + wl2 + gap, rows[4] + bh / 2)
    c.arrow(mid, rows[4] + bh, mid, rows[5])


def write_figure(cells: Cells, extent_w: float, extent_h: float):
    xml = (
        '<mxfile host="compose_dango_si_figures.py">'
        '<diagram id="FigS-dango-reproduction" name="Page-1">'
        '<mxGraphModel dx="0" dy="0" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" '
        'arrows="1" fold="1" page="1" pageScale="1" pageWidth="850" pageHeight="1100" math="1" shadow="0">'
        '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells.cells) + "</root></mxGraphModel></diagram></mxfile>"
    )
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"wrote {OUT}: {extent_w:.0f} x {extent_h:.0f} units = {extent_w / 3.937:.1f} x {extent_h / 3.937:.1f} mm")
    if extent_w > 709 or extent_h > MAX_HEIGHT:
        raise SystemExit("FigS-dango-reproduction exceeds the Nature print box (709 x 669 units)")


def main():
    releases = osp.join(P010, "graphs_string_releases.svg")
    zeros = osp.join(P005, "dango_decreased_zeros.svg")
    sweep = osp.join(P005, "dango_string_version_sweep.svg")
    curves = osp.join(P005, "dango_string_version_curves.svg")
    c = Cells()
    half = svg_size(releases)[0]
    col2 = half + COL_GAP
    # Row 1: (a) STRING release drift, (b) the schematic at the same size as panel a.
    row_top = 0
    y1 = row_top + TOP_STRIP
    wa, ha = c.image(releases, 0, y1)
    c.letter("a", 0, row_top)
    schematic(c, col2, y1, wa, ha)
    c.letter("b", col2, row_top)
    # Row 2: (c) decreased zeros, (d) best validation Pearson by release and schedule.
    row_top = y1 + ha + (ROW_GAP - TOP_STRIP)
    y2 = row_top + TOP_STRIP
    _, hc = c.image(zeros, 0, y2)
    _, hd = c.image(sweep, col2, y2)
    c.letter("c", 0, row_top)
    c.letter("d", col2, row_top)
    # Row 3: (e) the training curves, full width.
    row_top = y2 + max(hc, hd) + (ROW_GAP - TOP_STRIP)
    y3 = row_top + TOP_STRIP
    we, he = c.image(curves, 0, y3)
    c.letter("e", 0, row_top)
    extent_w = max(FULL_WIDTH, col2 + wa, we)
    extent_h = y3 + he
    write_figure(c, extent_w, extent_h)


if __name__ == "__main__":
    main()
