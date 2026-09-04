# experiments/005-kuzmin2018-tmi/scripts/compose_dango_si_figures.py
# [[experiments.005-kuzmin2018-tmi.scripts.compose_dango_si_figures]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/005-kuzmin2018-tmi/scripts/compose_dango_si_figures
"""Compose the DANGO reproduction SI figure as a draw.io file.

Panel (a) is a schematic of DANGO in the manuscript's notation, authored here as draw.io
cells (palette fills, Arial, ladder font sizes 8.3 body / 11.1 panel letters). Panels (b)
to (d) are the true-size SVGs written by ``dango_construction_si.py`` (b, decreased zeros
and lambda) and ``dango_string_version_sweep.py`` (c, best validation Pearson by release
and schedule; d, train and validation Pearson per epoch). Each SVG declares its size in
draw.io units (100 per inch), so the image cells are placed at exact physical size and the
figure is WYSIWYG when ``make -C paper/nature-biotech fig`` exports it.

Figure written to notes/assets/drawio/FigS-dango-reproduction.drawio (<= 709 x 669 units =
180 x 170 mm). Rerun after regenerating any panel; the file is overwritten, never edited by
hand.

Run from the repo root:
    python experiments/005-kuzmin2018-tmi/scripts/compose_dango_si_figures.py
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
OUT = osp.join(DRAWIO_DIR, "FigS-dango-reproduction.drawio")

FULL_WIDTH = 706  # 179.4 mm; draw.io adds a 1-unit border per side on export
MAX_HEIGHT = 669
GAP = 8
LETTER_W, LETTER_H = 18, 14
BODY = 8.3  # ladder value: prints at 5.98 pt
LETTER = 11.1  # ladder value: prints at 7.99 pt, panel letters only

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


class Cells:
    def __init__(self):
        self.cells: list[str] = []
        self.n = 2

    def _id(self) -> int:
        self.n += 1
        return self.n

    def image(self, path: str, x: float, y: float):
        w, h = svg_size(path)
        b64 = base64.b64encode(open(path, "rb").read()).decode("ascii")
        style = IMAGE_STYLE.format(b64=b64)
        self.cells.append(
            f'<mxCell id="{self._id()}" value="" style={quoteattr(style)} vertex="1" parent="1">'
            f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.4f}" height="{h:.4f}" as="geometry"/></mxCell>'
        )

    def letter(self, letter: str, x: float, y: float):
        self.cells.append(
            f'<mxCell id="{self._id()}" value="{letter}" style={quoteattr(LETTER_STYLE)} vertex="1" parent="1">'
            f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{LETTER_W}" height="{LETTER_H}" as="geometry"/></mxCell>'
        )

    def box(self, x, y, w, h, html: str, color=GRAY, align="center", dashed=False, bold_first=False):
        stroke, fill = color
        style = (
            f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
            f"fontFamily=Arial;fontSize={BODY};align={align};verticalAlign=middle;"
            f"spacingLeft=3;spacingRight=3;spacingTop=1;spacingBottom=1;strokeWidth=0.75;"
            + ("dashed=1;" if dashed else "")
        )
        self.cells.append(
            f'<mxCell id="{self._id()}" value={quoteattr(html)} style={quoteattr(style)} vertex="1" parent="1">'
            f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/></mxCell>'
        )

    def text(self, x, y, w, h, html: str, align="left", valign="top"):
        style = (
            f"text;html=1;strokeColor=none;fillColor=none;align={align};verticalAlign={valign};"
            f"whiteSpace=wrap;rounded=0;fontFamily=Arial;fontSize={BODY};spacing=0;"
        )
        self.cells.append(
            f'<mxCell id="{self._id()}" value={quoteattr(html)} style={quoteattr(style)} vertex="1" parent="1">'
            f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/></mxCell>'
        )

    def arrow(self, x1, y1, x2, y2, dashed=False, color="#666666", points=None):
        style = (
            f"endArrow=classic;html=1;strokeWidth=1;strokeColor={color};endSize=3;"
            + ("dashed=1;" if dashed else "")
        )
        pts = "".join(f'<mxPoint x="{px:.1f}" y="{py:.1f}"/>' for px, py in (points or []))
        arr = f'<Array as="points">{pts}</Array>' if points else ""
        self.cells.append(
            f'<mxCell id="{self._id()}" style={quoteattr(style)} edge="1" parent="1">'
            f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x1:.1f}" y="{y1:.1f}" as="sourcePoint"/>'
            f'<mxPoint x="{x2:.1f}" y="{y2:.1f}" as="targetPoint"/>{arr}</mxGeometry></mxCell>'
        )


def svg_size(path: str) -> tuple[float, float]:
    head = open(path, encoding="utf-8").read(2000)
    w = float(re.search(r'width="([0-9.]+)"', head).group(1))
    h = float(re.search(r'height="([0-9.]+)"', head).group(1))
    return w, h


# HTML fragments for the labels (draw.io renders the value as HTML; entities are written
# literally here and escaped once by quoteattr).
def i(s: str) -> str:
    return f"<i>{s}</i>"


def sub(s: str) -> str:
    return f"<sub>{s}</sub>"


def sup(s: str) -> str:
    return f"<sup>{s}</sup>"


LAM = "&#955;"
ALPHA = "&#945;"
SIGMA = "&#963;"
SUM = "&#931;"
ARROW = "&#8594;"
TIMES = "&#215;"
MINUS = "&#8722;"
IN = "&#8712;"
YHAT = "&#375;"
AHAT = "&#194;"
PSI = "&#968;"
RING = "&#8728;"
LE = "&#8804;"


def schematic(c: Cells, x0: float, y0: float) -> float:
    """DANGO in the paper's notation, laid out on a column grid. Returns the height used."""
    # Column grid (x, width) and the row of the main flow.
    cols = {
        "graphs": (x0 + 0, 112),
        "gnn": (x0 + 132, 132),
        "meta": (x0 + 284, 118),
        "pert": (x0 + 422, 116),
        "read": (x0 + 558, 148),
    }
    ytop = y0 + 4
    channels = ["neighborhood", "fusion", "co-occurrence", "co-expression", "experimental", "database"]
    # --- column 1: the six STRING channels as A^(k)
    gx, gw = cols["graphs"]
    c.text(gx, ytop, gw, 24, f"Six STRING channels<br>{i('A')}{sup('(' + i('k') + ')')} {IN} {{0,1}}{sup(i('N') + TIMES + i('N'))}, {i('k')} = 1,&#8230;,6", align="center")
    ch_y = ytop + 28
    for j, name in enumerate(channels):
        c.box(gx + 6, ch_y + j * 17, gw - 12, 14, name, ORANGE)
    # --- column 2: shared lookup, per-channel GNN, reconstruction head
    nx_, nw = cols["gnn"]
    c.box(nx_, ytop, nw, 30,
          f"Shared lookup {i('E')} {IN} R{sup(i('N') + TIMES + i('d'))}<br>{i('N')} = 6,607, {i('d')} = 64",
          GRAY)
    c.box(nx_, ytop + 40, nw, 44,
          f"Channel encoder {i('F')}{sup('(' + i('k') + ')')}: two GraphSAGE layers with {i('A')}{sup('(' + i('k') + ')')} as message-passing edges<br>{i('H')}{sup('(' + i('k') + ')')} = {i('F')}{sup('(' + i('k') + ')')}({i('E')}, {i('A')}{sup('(' + i('k') + ')')})",
          ORANGE)
    c.box(nx_, ytop + 96, nw, 44,
          f"Reconstruction head<br>{AHAT}{sup('(' + i('k') + ')')} = {i('H')}{sup('(' + i('k') + ')')}{i('W')}{sub(i('k'))}<br>{i('L')}{sub('rec')}: weighted MSE, zeros weighted by {LAM}{sub(i('k'))}",
          RED)
    # --- column 3: meta-embedding -> H
    mx, mw = cols["meta"]
    c.box(mx, ytop + 40, mw, 44,
          f"Meta-embedding over channels<br>{i('h')}{sub(i('i'))} = {SUM}{sub(i('k'))} {i('a')}{sub(i('ik'))} {i('h')}{sub(i('i'))}{sup('(' + i('k') + ')')}<br>{i('a')}{sub(i('i') + ':')} = softmax{sub(i('k'))} MLP({i('h')}{sub(i('i'))}{sup('(' + i('k') + ')')})",
          PURPLE)
    c.box(mx, ytop + 96, mw, 30,
          f"Cell representation<br>{i('H')} = ({i('h')}{sub('1')}, &#8230;, {i('h')}{sub(i('N'))})",
          GRAY)
    # --- column 4: perturbation as row selection (no operator)
    px, pw = cols["pert"]
    c.box(px, ytop + 40, pw, 44,
          f"Perturbation {i('p')} = {{{i('g')}{sub(i('i'))}, {i('g')}{sub(i('j'))}, {i('g')}{sub(i('k'))}}}<br>selects rows {i('h')}{sub(i('j'))}, {i('g')}{sub(i('j'))} {IN} {i('p')}<br>(no operator {i('T')}{sub(PSI)}; {i('H')} unchanged)",
          YELLOW)
    # --- column 5: Hyper-SAGNN readout and loss
    rx, rw = cols["read"]
    c.box(rx, ytop + 30, rw, 64,
          f"Hyper-SAGNN readout {i('R')}{sub('int')}<br>static {i('s')}{sub(i('t'))} = {SIGMA}(FC {i('h')}{sub(i('t'))}); dynamic {i('d')}{sub(i('t'))} from two self-attention layers over {i('p')} (ReZero)<br>{YHAT}{sub('int')} = (1/|{i('p')}|) {SUM}{sub(i('t'))} FC(({i('d')}{sub(i('t'))} {MINUS} {i('s')}{sub(i('t'))}){sup('2')}), squared elementwise",
          BLUE)
    c.box(rx, ytop + 104, rw, 30,
          f"{i('L')}{sub('int')} = mean log cosh({YHAT}{sub('int')} {MINUS} {i('y')}{sub('int')})",
          RED)
    # --- arrows along the flow
    c.arrow(gx + gw - 6, ch_y + 50, nx_, ytop + 62)                       # channels -> GNN
    c.arrow(nx_ + nw / 2, ytop + 30, nx_ + nw / 2, ytop + 40)             # lookup -> GNN
    c.arrow(nx_ + nw / 2, ytop + 84, nx_ + nw / 2, ytop + 96)             # GNN -> recon head
    c.arrow(nx_ + nw, ytop + 62, mx, ytop + 62)                           # GNN -> meta
    c.arrow(mx + mw / 2, ytop + 84, mx + mw / 2, ytop + 96)               # meta -> H
    c.arrow(mx + mw, ytop + 62, px, ytop + 62)                            # meta -> pert
    c.arrow(px + pw, ytop + 62, rx, ytop + 62)                            # pert -> readout
    c.arrow(rx + rw / 2, ytop + 94, rx + rw / 2, ytop + 104)              # readout -> loss
    # --- objective strip
    sy = ytop + 150
    c.box(x0, sy, FULL_WIDTH, 30,
          f"Objective at epoch {i('e')}: {i('L')} = {ALPHA}{sub(i('e'))} {i('L')}{sub('rec')} + (1 {MINUS} {ALPHA}{sub(i('e'))}) {i('L')}{sub('int')}; "
          f"{LAM}{sub(i('k'))} = 0.1 if the channel's decreased zeros exceed 1%, else 1 (panel b). "
          f"Schedules with transition epoch 10: pretrain then main, {ALPHA}{sub(i('e'))} = 1 for {i('e')} &lt; 10, then 0; "
          f"linear to uniform, {ALPHA}{sub(i('e'))} 1 {ARROW} 0.5 by epoch 10, then 0.5; linear to flipped, {ALPHA}{sub(i('e'))} 1 {ARROW} 0 by epoch 10, then 0.",
          GRAY, align="left")
    return sy + 30 - y0


def write_figure(cells: Cells, extent_w: float, extent_h: float):
    xml = (
        '<mxfile host="compose_dango_si_figures.py">'
        '<diagram id="FigS-dango-reproduction" name="Page-1">'
        '<mxGraphModel dx="0" dy="0" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" '
        'arrows="1" fold="1" page="1" pageScale="1" pageWidth="850" pageHeight="1100" math="0" shadow="0">'
        '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells.cells) + "</root></mxGraphModel></diagram></mxfile>"
    )
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"wrote {OUT}: {extent_w:.0f} x {extent_h:.0f} units = {extent_w / 3.937:.1f} x {extent_h / 3.937:.1f} mm")
    if extent_w > FULL_WIDTH + 2 or extent_h > MAX_HEIGHT:
        raise SystemExit("FigS-dango-reproduction exceeds the Nature print box (709 x 669 units)")


def main():
    zeros = osp.join(P005, "dango_decreased_zeros.svg")
    sweep = osp.join(P005, "dango_string_version_sweep.svg")
    curves = osp.join(P005, "dango_string_version_curves.svg")
    c = Cells()
    # (a) schematic
    c.letter("a", -2, -4)
    ha = schematic(c, 0, 12)
    # (b) decreased zeros (left) and (c) sweep (right), flush with the 180 mm edge; (d) the
    # training curves, full width. Images first, letters after: later cells draw on top, and
    # each panel SVG carries an opaque background that would hide a letter placed under it.
    yb = 12 + ha + GAP
    col2 = FULL_WIDTH - svg_size(sweep)[0]
    row2_h = max(svg_size(zeros)[1], svg_size(sweep)[1])
    yd = yb + row2_h + GAP
    c.image(zeros, 0, yb)
    c.image(sweep, col2, yb)
    c.image(curves, 0, yd)
    c.letter("b", -2, yb - 4)
    c.letter("c", col2 - 2, yb - 4)
    c.letter("d", -2, yd - 4)
    extent_h = yd + svg_size(curves)[1]
    write_figure(c, FULL_WIDTH, extent_h)


if __name__ == "__main__":
    main()
