# experiments/006-kuzmin-tmi/scripts/dcell_model_compose_figure.py
# [[experiments.006-kuzmin-tmi.scripts.dcell_model_compose_figure]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/006-kuzmin-tmi/scripts/dcell_model_compose_figure
"""Compose FigS-dcell-model.drawio: the filtered GO DAG with the model's equations, plus the DAG panels.

Panel (a) is the true-size DAG rendering from ``dcell_model_go_stats.py`` (this folder;
``dcell_model_go_dag.svg``, the whole filtered ontology with one triple deletion
propagating to the root) with a column of draw.io cells to its right that state the model
in the manuscript's notation: gene-state input, the subsystem of Eq. dcell-subsystem, the
root readout, the auxiliary heads, and the loss. The equations are real LaTeX: the
diagram sets ``math="1"`` and draw.io typesets ``$$...$$`` labels with MathJax, which the
headless PDF export honors (verified with draw.io 31.3.1; the glyphs come out as vector
paths). MathJax renders about 1.19x the cell's ``fontSize``, so math cells are typed at
``fontSize=7`` to print at ~6 pt, the size of the matplotlib panels (measured: at 7 the
math cap height matches Arial at 8.3 within 5%). Plain text stays on the ladder
(8.3 body / 9.7 headers / 11.1 panel letters). Panels (b)-(d) are the other true-size SVGs
from the same script, placed at exact physical size (100 draw.io units per inch).

Layout convention shared by every composed SI figure (the "white cross"): COL_GAP = 12
units (3 mm) between columns, ROW_GAP = 22 units (5.5 mm) between rows, a TOP_STRIP of 16
units above every row, and each panel letter in that strip at the panel's top-left
(x = panel_x, y = row_top), so no letter sits over an axis label or a neighbor's title
and clear white gutters cross the figure both ways. The figure stays <= 709 x 669 units.

Export (also done by ``make -C paper/nature-biotech fig``):
    /Applications/draw.io.app/Contents/MacOS/draw.io -x -f pdf --crop \\
        -o paper/nature-biotech/figures/FigS-dcell-model.pdf notes/assets/drawio/FigS-dcell-model.drawio

Run from the repo root:
    python experiments/006-kuzmin-tmi/scripts/dcell_model_compose_figure.py
"""

import base64
import os
import os.path as osp
import re
from xml.sax.saxutils import quoteattr

from dotenv import load_dotenv

load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "006-kuzmin-tmi")
DRAWIO_DIR = "notes/assets/drawio"
NAME = "FigS-dcell-model"

FULL_WIDTH = 707  # 179.5 mm; draw.io adds a 1-unit border per side on export (709 = 180 mm)
MAX_HEIGHT = 669  # 170 mm
COL_GAP = 12  # 3 mm between columns
ROW_GAP = 22  # 5.5 mm between rows; the next row's TOP_STRIP is the lower part of it
TOP_STRIP = 16  # the letter strip above every row
LETTER_W, LETTER_H = 18, 14

# Palette (PLOT_PALETTE / PLOT_PALETTE_FILL slots 1-6).
ORANGE = ("#D79B00", "#FFE6CC")
RED = ("#B85450", "#F8CECC")
PURPLE = ("#9673A6", "#E1D5E7")
YELLOW = ("#D6B656", "#FFF2CC")
BLUE = ("#6C8EBF", "#DAE8FC")
GRAY = ("#666666", "#F5F5F5")

FS = 8.3  # 6 pt
FS_HEAD = 9.7  # 7 pt
FS_LETTER = 11.1  # 8 pt, panel letters only
FS_MATH = 7  # MathJax renders ~1.19x the cell size: 7 units -> ~6 pt on the page (measured)

LETTER_STYLE = (
    "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=top;"
    f"whiteSpace=wrap;rounded=0;fontFamily=Arial;fontSize={FS_LETTER};fontStyle=1;"
)
IMAGE_STYLE = (
    "shape=image;verticalLabelPosition=bottom;labelBackgroundColor=default;"
    "verticalAlign=top;aspect=fixed;imageAspect=0;image=data:image/svg+xml,{b64};"
)


class Canvas:
    def __init__(self):
        self.cells: list[str] = []
        self.n = 2

    def _id(self) -> str:
        self.n += 1
        return f"c{self.n}"

    def box(self, value, x, y, w, h, color=GRAY, fs=FS, bold=False, rounded=0, align="center", valign="middle", dashed=False, fill=True):
        style = (
            f"rounded={rounded};whiteSpace=wrap;html=1;fontFamily=Arial;fontSize={fs};align={align};verticalAlign={valign};"
            f"strokeColor={color[0]};fillColor={color[1] if fill else 'none'};strokeWidth=0.75;"
            + ("fontStyle=1;" if bold else "")
            + ("dashed=1;" if dashed else "")
            + ("spacingLeft=3;" if align == "left" else "")
            + ("spacingTop=1;" if valign == "top" else "")
        )
        self.cells.append(
            f'<mxCell id="{self._id()}" value={quoteattr(value)} style={quoteattr(style)} vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )

    def text(self, value, x, y, w, h, fs=FS, bold=False, align="left", valign="top", color=None):
        style = (
            f"text;html=1;strokeColor=none;fillColor=none;align={align};verticalAlign={valign};"
            f"whiteSpace=wrap;rounded=0;fontFamily=Arial;fontSize={fs};"
            + ("fontStyle=1;" if bold else "")
            + (f"fontColor={color};" if color else "")
        )
        self.cells.append(
            f'<mxCell id="{self._id()}" value={quoteattr(value)} style={quoteattr(style)} vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )

    def math(self, latex, x, y, w, h, align="left"):
        """A LaTeX label typeset by MathJax (requires math="1" on the model)."""
        self.text(f"$${latex}$$", x, y, w, h, fs=FS_MATH, align=align, valign="middle")

    def arrow(self, x1, y1, x2, y2, color="#666666", dashed=False, width=0.75, head="classic"):
        style = (
            f"endArrow={head};html=1;strokeWidth={width};strokeColor={color};endSize=3;"
            + ("dashed=1;dashPattern=3 2;" if dashed else "")
        )
        self.cells.append(
            f'<mxCell id="{self._id()}" style={quoteattr(style)} edge="1" parent="1">'
            f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x1}" y="{y1}" as="sourcePoint"/>'
            f'<mxPoint x="{x2}" y="{y2}" as="targetPoint"/></mxGeometry></mxCell>'
        )

    def image(self, path, x, y):
        w, h = svg_size(path)
        b64 = base64.b64encode(open(path, "rb").read()).decode("ascii")
        self.cells.append(
            f'<mxCell id="{self._id()}" value="" style={quoteattr(IMAGE_STYLE.format(b64=b64))} vertex="1" parent="1">'
            f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.4f}" height="{h:.4f}" as="geometry"/></mxCell>'
        )
        return w, h

    def letter(self, letter, x, row_top):
        """Panel letter in the TOP_STRIP above the panel, flush with the panel's left edge."""
        self.cells.append(
            f'<mxCell id="{self._id()}" value="{letter}" style={quoteattr(LETTER_STYLE)} vertex="1" parent="1">'
            f'<mxGeometry x="{x:.1f}" y="{row_top:.1f}" width="{LETTER_W}" height="{LETTER_H}" as="geometry"/></mxCell>'
        )


def svg_size(path: str) -> tuple[float, float]:
    head = open(path, encoding="utf-8").read(2000)
    w = float(re.search(r'width="([0-9.]+)"', head).group(1))
    h = float(re.search(r'height="([0-9.]+)"', head).group(1))
    return w, h


def equations(c: Canvas, x0: float, y0: float, w: float) -> float:
    """The model in the shared notation, as a column of boxes to the right of the DAG. Returns the bottom."""
    pad = 4
    iw = w - 2 * pad
    y = y0

    # -- gene-state input: the strain row, as a small visual with minimal words.
    h = 74
    c.box("Perturbation enters as data", x0, y, w, h, color=GRAY, align="left", valign="top", bold=True)
    chips = [1, 1, 0, 1, 1, 0, 1, 0, 1, 1]
    cw, cg = 13, 3
    cx0 = x0 + pad + (iw - (len(chips) * (cw + cg) - cg)) / 2
    for i, s in enumerate(chips):
        c.box(str(s), cx0 + i * (cw + cg), y + 18, cw, 11, color=(RED if s == 0 else GRAY), fs=FS)
    c.math(r"s_i = 0 \text{ if } g_i \in p, \text{ else } 1", x0 + pad, y + 32, iw, 14)
    c.text("(term, gene) rows copied per strain; rows of deleted genes zeroed", x0 + pad, y + 47, iw, 24)
    y += h + 6

    # -- subsystem
    h = 62
    c.box("Subsystem t, one per GO term", x0, y, w, h, color=YELLOW, align="left", valign="top", bold=True)
    c.math(r"I_t = \big[\, \Vert_{c \in \mathrm{ch}(t)} O_c \ \big\Vert\ s_{\mathrm{genes}(t)} \big]", x0 + pad, y + 13, iw, 16)
    c.math(r"O_t = \tanh\big(\mathrm{BN}(W_t I_t + b_t)\big) \in [-1,1]^{L_t}", x0 + pad, y + 29, iw, 16)
    c.math(r"L_t = \max\big(20,\, \lceil 0.3\,\lvert \mathrm{genes}(t) \rvert \rceil\big)", x0 + pad, y + 45, iw, 16)
    y += h + 6

    # -- readout
    h = 30
    c.box("Root readout: trigenic interaction", x0, y, w, h, color=ORANGE, align="left", valign="top", bold=True)
    c.math(r"\hat y = w_r^{\top} O_{\mathrm{ROOT}} + b_r", x0 + pad, y + 13, iw, 16)
    y += h + 6

    # -- auxiliary heads
    h = 30
    c.box("Auxiliary head on every t &#8800; r", x0, y, w, h, color=PURPLE, align="left", valign="top", bold=True)
    c.math(r"\hat y_t = w_t^{\top} O_t + b_t", x0 + pad, y + 13, iw, 16)
    y += h + 6

    # -- loss
    h = 52
    c.box("Loss", x0, y, w, h, color=GRAY, align="left", valign="top", bold=True)
    c.math(r"\mathcal{L} = \mathrm{MSE}(\hat y, y) + \alpha \operatorname*{mean}_{t \neq r} \mathrm{MSE}(\hat y_t, y)", x0 + pad, y + 15, iw, 22)
    c.math(r"\alpha = 0.3", x0 + pad, y + 37, iw, 14)
    return y + h


def main():
    c = Canvas()
    dag = osp.join(IMG_DIR, "dcell_model_go_dag.svg")
    # Row 1: panel (a) = the DAG rendering with the equations column to its right.
    row_top = 0
    y1 = row_top + TOP_STRIP
    c.letter("a", 0, row_top)
    w_dag, h_dag = c.image(dag, 0, y1)
    x_eq = w_dag + COL_GAP
    bottom_eq = equations(c, x_eq, y1, FULL_WIDTH - x_eq)
    bottom1 = max(y1 + h_dag, bottom_eq)

    # Row 2: panels (b)-(d), three third-width panels across the page.
    row_top = bottom1 + (ROW_GAP - TOP_STRIP)
    y2 = row_top + TOP_STRIP
    panels = ["dcell_model_terms_per_stratum", "dcell_model_genes_per_term", "dcell_model_terms_per_gene"]
    x = 0.0
    h2 = 0.0
    for letter, name in zip("bcd", panels):
        path = osp.join(IMG_DIR, f"{name}.svg")
        w, h = c.image(path, x, y2)
        c.letter(letter, x, row_top)
        x += w + COL_GAP
        h2 = max(h2, h)
    extent_w = x - COL_GAP
    extent_h = y2 + h2
    xml = (
        f'<mxfile host="{osp.basename(__file__)}">'
        f'<diagram id="{NAME}" name="Page-1">'
        '<mxGraphModel dx="0" dy="0" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" '
        'arrows="1" fold="1" page="1" pageScale="1" pageWidth="850" pageHeight="1100" math="1" shadow="0">'
        '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(c.cells) + "</root></mxGraphModel></diagram></mxfile>"
    )
    out = osp.join(DRAWIO_DIR, f"{NAME}.drawio")
    with open(out, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"wrote {out}: {max(extent_w, FULL_WIDTH):.0f} x {extent_h:.0f} units = "
          f"{max(extent_w, FULL_WIDTH) / 3.937:.1f} x {extent_h / 3.937:.1f} mm")
    if extent_w > FULL_WIDTH + 2 or extent_h > MAX_HEIGHT:
        raise SystemExit(f"{NAME} exceeds the Nature print box (709 x 669 units)")


if __name__ == "__main__":
    main()
