# experiments/007-kuzmin-tm/scripts/fba_baseline_compose_figure.py
# [[experiments.007-kuzmin-tm.scripts.fba_baseline_compose_figure]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/007-kuzmin-tm/scripts/fba_baseline_compose_figure
"""Compose FigS-yeast9-fba.drawio: the FBA baseline pipeline schematic plus its data panels.

Panel (a) is a draw.io schematic of the pipeline behind the "Yeast9 FBA" value of Fig. 2d as
six stage cards, each carrying a graphic rather than a paragraph: the model's gene coverage
of the genome and of the screen (a true-size SVG from ``fba_baseline_si.py``), the
gene-reaction rules drawn as OR and AND gates with one deleted gene, the deletion sets as a
bar chart with the fully covered part shaded (SVG from the same script), FBA as its linear
program in an algorithm block, the fitness proxy, and the trigenic interaction of Fig. 2a.
The counts and correlations are in the caption. The equations are real LaTeX typeset by
MathJax (``math="1"`` on the model, ``$$...$$`` labels; math cells at ``fontSize=7`` print
at ~6 pt). Panels (b)-(f) are the true-size SVGs
from the same script, placed at exact physical size (100 draw.io units per inch). Panel (g)
is the same comparison as (b) for the rerun on the screen's own medium
(``fba_screen_medium_tau.svg`` from ``fba_screen_medium_si.py``, this folder).

Layout convention shared by every composed SI figure (the "white cross"): COL_GAP = 12
units (3 mm) between columns, ROW_GAP = 22 units (5.5 mm) between rows, a TOP_STRIP of 16
units above every row, and each panel letter in that strip at the panel's top-left
(x = panel_x, y = row_top). Nothing but a letter may enter the strip. The figure stays
<= 709 x 669 units.

Export (also done by ``make -C paper/nature-biotech fig``):
    /Applications/draw.io.app/Contents/MacOS/draw.io -x -f pdf --crop \\
        -o paper/nature-biotech/figures/FigS-yeast9-fba.pdf notes/assets/drawio/FigS-yeast9-fba.drawio

Run from the repo root:
    python experiments/007-kuzmin-tm/scripts/fba_baseline_compose_figure.py
"""

import base64
import json
import os
import os.path as osp
import re
from xml.sax.saxutils import quoteattr

from dotenv import load_dotenv

load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
assert ASSET_IMAGES_DIR is not None, "ASSET_IMAGES_DIR must be set in the environment"

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
EXP_DIR = osp.dirname(SCRIPT_DIR)
REPO_ROOT = osp.dirname(osp.dirname(EXP_DIR))
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "007-kuzmin-tm")
STATS = osp.join(EXP_DIR, "results", "fba_baseline_si", "stats.json")
DRAWIO_DIR = osp.join(REPO_ROOT, "notes", "assets", "drawio")
NAME = "FigS-yeast9-fba"

FULL_WIDTH = 707  # 179.5 mm; draw.io adds a 1-unit border per side on export (709 = 180 mm)
MAX_HEIGHT = 669  # 170 mm
COL_GAP = 12  # 3 mm between columns
ROW_GAP = 22  # 5.5 mm between rows; the next row's TOP_STRIP is the lower part of it
TOP_STRIP = 16  # the letter strip above every row
LETTER_W, LETTER_H = 18, 14

# Palette (PLOT_PALETTE / PLOT_PALETTE_FILL slots 1-6), (stroke, fill).
ORANGE = ("#D79B00", "#FFE6CC")
RED = ("#B85450", "#F8CECC")
PURPLE = ("#9673A6", "#E1D5E7")
YELLOW = ("#D6B656", "#FFF2CC")
BLUE = ("#6C8EBF", "#DAE8FC")
GRAY = ("#666666", "#F5F5F5")

# Box color by role, the scheme of the DANGO and DCell schematics: data the pipeline reads
# is orange, the perturbation's entry yellow, the readout blue, the derived quantity purple,
# the score compared against the label red.
ROLE_COLOR = {"input": ORANGE, "rules": ORANGE, "perturbation": YELLOW, "readout": BLUE, "proxy": PURPLE, "score": RED}

FS = 8.3  # 6 pt
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
        if y < TOP_STRIP:
            raise SystemExit(f"box {value!r} at y={y} intrudes into the letter strip")
        style = (
            f"rounded={rounded};whiteSpace=wrap;html=1;fontFamily=Arial;fontSize={fs};align={align};verticalAlign={valign};"
            f"strokeColor={color[0]};fillColor={color[1] if fill else 'none'};strokeWidth=0.75;"
            + ("fontStyle=1;" if bold else "")
            + ("dashed=1;" if dashed else "")
            + ("spacingLeft=3;spacingRight=3;" if align == "left" else "")
            + ("spacingTop=1;" if valign == "top" else "")
        )
        self.cells.append(
            f'<mxCell id="{self._id()}" value={quoteattr(value)} style={quoteattr(style)} vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )

    def ellipse(self, value, x, y, w, h, color=GRAY, fs=FS, dashed=False, fill=True):
        style = (
            f"ellipse;whiteSpace=wrap;html=1;fontFamily=Arial;fontSize={fs};align=center;verticalAlign=middle;"
            f"strokeColor={color[0]};fillColor={color[1] if fill else 'none'};strokeWidth=0.75;"
            + ("dashed=1;" if dashed else "")
        )
        self.cells.append(
            f'<mxCell id="{self._id()}" value={quoteattr(value)} style={quoteattr(style)} vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )

    def line(self, x1, y1, x2, y2, color="#666666", width=0.75, dashed=False):
        style = f"endArrow=none;html=1;strokeWidth={width};strokeColor={color};" + ("dashed=1;" if dashed else "")
        self.cells.append(
            f'<mxCell id="{self._id()}" style={quoteattr(style)} edge="1" parent="1">'
            f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x1}" y="{y1}" as="sourcePoint"/>'
            f'<mxPoint x="{x2}" y="{y2}" as="targetPoint"/></mxGeometry></mxCell>'
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

    def arrow(self, x1, y1, x2, y2, color="#666666", width=0.75, dashed=False):
        style = f"endArrow=classic;html=1;strokeWidth={width};strokeColor={color};endSize=3;" + ("dashed=1;" if dashed else "")
        self.cells.append(
            f'<mxCell id="{self._id()}" style={quoteattr(style)} edge="1" parent="1">'
            f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x1}" y="{y1}" as="sourcePoint"/>'
            f'<mxPoint x="{x2}" y="{y2}" as="targetPoint"/></mxGeometry></mxCell>'
        )

    def image(self, path, x, y):
        if y < TOP_STRIP:
            raise SystemExit(f"image {osp.basename(path)} at y={y} intrudes into the letter strip")
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


ARROW_GAP = 14  # between the boxes of the pipeline row
PIPE_H = 168  # height of the pipeline boxes (42.7 mm)
BOX_W = [116, 108, 116, 100, 76, 121]  # sums with the five gaps to FULL_WIDTH
HEAD_H = 15  # the bold heading band inside each box


def rule_sketch(c: Canvas, x0: float, y0: float, w: float):
    """Two gene-reaction rules drawn as gates: isozymes joined by OR survive one deletion,
    a complex joined by AND is blocked by one. Genes are circles (a deleted gene dashed and
    white), the gate a small box, the reaction an arrow (dashed when blocked).
    """
    stroke = ROLE_COLOR["rules"][0]
    r = 13  # gene circle diameter
    gx = x0 + 6  # gene column
    gate_x = x0 + 40
    rxn_x = x0 + 74
    for row, (kind, gate, blocked, top) in enumerate([("isozymes", "OR", False, y0), ("complex", "AND", True, y0 + 44)]):
        ya, yb = top + 2, top + 20  # two gene rows
        c.ellipse(f"g<sub>{2 * row + 1}</sub>", gx, ya, r, r, color=ROLE_COLOR["rules"], fs=7)
        c.ellipse(f"g<sub>{2 * row + 2}</sub>", gx, yb, r, r, color=ROLE_COLOR["rules"], fs=7, dashed=True, fill=False)
        c.text("&Delta;", gx + r + 1, yb - 1, 10, 12, fs=7)
        ym = (ya + yb + r) / 2
        c.line(gx + r, ya + r / 2, gate_x, ym - 4, color=stroke)
        c.line(gx + r, yb + r / 2, gate_x, ym + 4, color=stroke, dashed=True)
        c.box(gate, gate_x, ym - 7, 22, 14, color=ROLE_COLOR["rules"], fs=7, bold=True)
        c.arrow(gate_x + 22, ym, rxn_x - 2, ym, color=stroke, dashed=blocked)
        c.text("v<sub>r</sub> = 0" if blocked else "v<sub>r</sub>", rxn_x, ym - 7, w - (rxn_x - x0) - 2, 14, fs=7, valign="middle")
        c.text(f"{kind}: {gate}", gx, top + 34, 70, 10, fs=7)


def algorithm_block(c: Canvas, x0: float, y0: float, w: float, h: float, lines: list[str], note: str):
    """An algorithm-style block: a rule under the heading, math lines, a note under a rule."""
    c.line(x0, y0, x0 + w, y0, color="#666666", width=0.5)
    y = y0 + 3
    for latex in lines:
        c.math(latex, x0, y, w, 14)
        y += 15
    c.line(x0, y + 2, x0 + w, y + 2, color="#666666", width=0.5)
    c.text(note, x0, y + 5, w, h - (y + 5 - y0), fs=FS)


def pipeline(c: Canvas, st: dict, y0: float) -> float:
    """Panel a as six stage cards: a graphic per stage, the numbers in the caption."""
    m = st["model"]
    assert sum(BOX_W) + 5 * ARROW_GAP == FULL_WIDTH, sum(BOX_W) + 5 * ARROW_GAP
    x, pad, lefts = 0.0, 4, []
    for w in BOX_W:
        lefts.append(x)
        x += w + ARROW_GAP
    h = PIPE_H
    heads = ["Yeast9 metabolism", "Gene-reaction rules", "Deletion sets", "FBA growth", "Fitness proxy", "Interaction (Fig. 2a)"]
    roles = ["input", "rules", "perturbation", "readout", "proxy", "score"]
    for x0, w, head, role in zip(lefts, BOX_W, heads, roles):
        c.box(head, x0, y0, w, h, color=ROLE_COLOR[role], align="left", valign="top", bold=True)

    # 1. Yeast9: how much of the genome, and of the screened genes, the model carries.
    x0, w = lefts[0], BOX_W[0]
    gw, gh = c.image(osp.join(IMG_DIR, "fba_schematic_genes.svg"), x0 + pad, y0 + HEAD_H)
    c.text(
        f"Default medium as distributed: {m['n_medium_exchanges_open']} of {m['n_exchange_reactions']} exchanges open; "
        f"glucose {m['glucose_uptake_bound']:g} mmol gDW<sup>-1</sup> h<sup>-1</sup>, NH<sub>4</sub><sup>+</sup>, "
        "no amino acid or vitamin.",
        x0 + pad, y0 + HEAD_H + gh + 2, w - 2 * pad, h - HEAD_H - gh - 4,
    )

    # 2. Gene-reaction rules as gates.
    x0, w = lefts[1], BOX_W[1]
    rule_sketch(c, x0 + pad, y0 + HEAD_H + 2, w - 2 * pad)
    c.math(r"s_g = 0 \text{ if } g \in p,\ \text{else } 1", x0 + pad, y0 + h - 40, w - 2 * pad, 16)
    c.math(r"v_r = 0 \text{ if } \mathrm{GPR}_r(\mathbf{s}) = \text{false}", x0 + pad, y0 + h - 24, w - 2 * pad, 16)

    # 3. Deletion sets: the screen's gene sets, and how many the model covers in full.
    x0, w = lefts[2], BOX_W[2]
    dw, dh = c.image(osp.join(IMG_DIR, "fba_schematic_deletions.svg"), x0 + pad, y0 + HEAD_H)
    c.text("The Kuzmin 2018 and 2020 triples and every single and double inside them; each set is one deletion strain.",
           x0 + pad, y0 + HEAD_H + dh + 2, w - 2 * pad, h - HEAD_H - dh - 4)

    # 4. FBA as the linear program it is.
    x0, w = lefts[3], BOX_W[3]
    algorithm_block(
        c, x0 + pad, y0 + HEAD_H + 2, w - 2 * pad, h - HEAD_H - 4,
        [r"\mu = \max\ v_{\mathrm{growth}}", r"\text{s.t. } S v = 0", r"lb \le v \le ub", r"v_r = 0 \text{ for blocked } r"],
        "One linear program per deletion set (GLPK, 60 s limit); the wild type is the same program with no gene deleted.",
    )

    # 5. Fitness proxy.
    x0, w = lefts[4], BOX_W[4]
    c.math(r"f = \mu / \mu_{\mathrm{WT}}", x0 + pad, y0 + HEAD_H + 4, w - 2 * pad, 16)
    c.text("Growth relative to the wild type; a solve with no optimum counts as f = 0.",
           x0 + pad, y0 + HEAD_H + 26, w - 2 * pad, h - HEAD_H - 28)

    # 6. Interaction, as in Fig. 2a.
    x0, w = lefts[5], BOX_W[5]
    c.math(r"\varepsilon_{ij} = f_{ij} - f_i f_j", x0 + pad, y0 + HEAD_H + 4, w - 2 * pad, 16)
    c.math(r"\tau_{ijk} = f_{ijk} - f_i f_j f_k", x0 + pad, y0 + HEAD_H + 20, w - 2 * pad, 16)
    c.math(r"\quad - \varepsilon_{ij} f_k - \varepsilon_{ik} f_j - \varepsilon_{jk} f_i", x0 + pad, y0 + HEAD_H + 34, w - 2 * pad, 16)
    c.text("From the predicted fitness of the triple, its three doubles and three singles; "
           "compared with the measured &tau; of the same gene set (b, g).",
           x0 + pad, y0 + HEAD_H + 56, w - 2 * pad, h - HEAD_H - 58)

    ym = y0 + h / 2
    for i in range(5):
        c.arrow(lefts[i] + BOX_W[i] + 1, ym, lefts[i + 1] - 1, ym)
    return y0 + h


def main():
    st = json.load(open(STATS))
    c = Canvas()

    # Row 1: panel a, the pipeline schematic across the full width.
    row_top = 0
    y1 = row_top + TOP_STRIP
    c.letter("a", 0, row_top)
    bottom1 = pipeline(c, st, y1)

    # Row 2: panels b-d, three third-width panels.
    row_top = bottom1 + (ROW_GAP - TOP_STRIP)
    y2 = row_top + TOP_STRIP
    x, h2 = 0.0, 0.0
    for letter, name in zip("bcd", ["fba_baseline_tau", "fba_baseline_fitness", "fba_baseline_growth_bands"]):
        w, h = c.image(osp.join(IMG_DIR, f"{name}.svg"), x, y2)
        c.letter(letter, x, row_top)
        x += w + COL_GAP
        h2 = max(h2, h)
    extent_w = x - COL_GAP
    bottom2 = y2 + h2

    # Row 3: panels e (evaluable set), f (measured landscape), and the lettered placeholder g.
    row_top = bottom2 + (ROW_GAP - TOP_STRIP)
    y3 = row_top + TOP_STRIP
    x, h3 = 0.0, 0.0
    for letter, name in zip("ef", ["fba_baseline_evaluable", "fba_baseline_landscape"]):
        w, h = c.image(osp.join(IMG_DIR, f"{name}.svg"), x, y3)
        c.letter(letter, x, row_top)
        x += w + COL_GAP
        h3 = max(h3, h)
    w, h = c.image(osp.join(IMG_DIR, "fba_screen_medium_tau.svg"), x, y3)
    c.letter("g", x, row_top)
    x += w
    h3 = max(h3, h)
    extent_w = max(extent_w, x)
    extent_h = y3 + h3

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
