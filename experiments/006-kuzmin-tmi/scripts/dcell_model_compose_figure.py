# experiments/006-kuzmin-tmi/scripts/dcell_model_compose_figure.py
# [[experiments.006-kuzmin-tmi.scripts.dcell_model_compose_figure]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/006-kuzmin-tmi/scripts/dcell_model_compose_figure
"""Compose FigS-dcell-model.drawio: the DCell-in-TorchCell schematic plus the GO-DAG panels.

Panel (a) is a hand-authored draw.io schematic written here as mxGraph XML (palette
colors, Arial, font ladder 8.3 / 9.7 / 11.1). Panels (b)-(d) are the true-size SVGs from
``dcell_model_go_stats.py`` (this folder), placed at exact physical size (100 draw.io units
per inch, via ``savefig_true_size_svg``). The figure is 180 mm wide and well under 170 mm
tall. Rerun after regenerating any panel; the .drawio is overwritten, never hand-edited.

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

FULL_WIDTH = 702  # 178.2 mm; draw.io adds a 1-unit border per side on export
GAP_Y = 8

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

    def box(self, value, x, y, w, h, color=GRAY, fs=FS, bold=False, rounded=0, align="center", dashed=False, fill=True):
        style = (
            f"rounded={rounded};whiteSpace=wrap;html=1;fontFamily=Arial;fontSize={fs};align={align};"
            f"strokeColor={color[0]};fillColor={color[1] if fill else 'none'};strokeWidth=0.75;"
            + ("fontStyle=1;" if bold else "")
            + ("dashed=1;" if dashed else "")
            + ("spacingLeft=3;" if align == "left" else "")
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

    def letter(self, letter, x, y):
        self.cells.append(
            f'<mxCell id="{self._id()}" value="{letter}" style={quoteattr(LETTER_STYLE)} vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="18" height="14" as="geometry"/></mxCell>'
        )


def svg_size(path: str) -> tuple[float, float]:
    head = open(path, encoding="utf-8").read(2000)
    w = float(re.search(r'width="([0-9.]+)"', head).group(1))
    h = float(re.search(r'height="([0-9.]+)"', head).group(1))
    return w, h


def sub(s: str) -> str:
    return f"<sub>{s}</sub>"


def schematic(c: Canvas, y0: float) -> float:
    """Draw panel (a) starting at y0; return its bottom edge."""
    # Column grid (x): A = perturbation text 16..168, B = GO DAG + gene input layer 180..450,
    # C = subsystem + readout 462..702.  Rows in B: root r0, namespace roots r1, terms r2,
    # leaves r3, gene states r4 (the input layer sits under the leaves, as in DCell Fig. 1).
    hdr = y0 + 2
    c.text("Perturbation enters as data", 16, hdr, 152, 12, fs=FS_HEAD, bold=True)
    c.text("GO DAG in place of a gene&#8211;gene A<sup>(k)</sup>", 180, hdr, 270, 12, fs=FS_HEAD, bold=True)
    c.text("Readout and losses", 462, hdr, 240, 12, fs=FS_HEAD, bold=True)

    # ---- Column A: what the data-side perturbation does.
    ax, aw = 16, 152
    c.text("strain = wildtype with p = {g" + sub("2") + ", g" + sub("5") + "} deleted", ax, y0 + 20, aw, 12)
    c.text("gene state s" + sub("i") + " = 0 if g" + sub("i") + " &#8712; p, else 1", ax, y0 + 33, aw, 12)
    c.box("0", ax, y0 + 49, 14, 11, color=RED)
    c.text("deleted", ax + 16, y0 + 48, 40, 12)
    c.box("1", ax + 60, y0 + 49, 14, 11, color=GRAY)
    c.text("present", ax + 76, y0 + 48, 40, 12)
    c.text("The reference's (term, gene) annotation table (59,986 rows) is copied per strain and the rows of deleted genes are zeroed. The strain is a rebuilt data object; no operator acts on a shared encoding H.",
           ax, y0 + 66, aw, 70)
    c.text("Each subsystem reads the states of its own annotated genes and the outputs of its child subsystems; every gene is in at least two subsystems.",
           ax, y0 + 128, aw, 58)

    # ---- Column B: the GO DAG with the gene input layer beneath it.
    bx = 180
    tw, th = 48, 13
    r0, r1, r2, r3, r4 = y0 + 22, y0 + 52, y0 + 82, y0 + 112, y0 + 146
    c.box("GO:ROOT", bx + 106, r0, tw, th, color=ORANGE, bold=True)
    ns = [("BP", bx + 40), ("MF", bx + 106), ("CC", bx + 172)]
    for name, x in ns:
        c.box(name, x, r1, tw, th, color=YELLOW)
        c.arrow(x + tw / 2, r1, bx + 106 + tw / 2, r0 + th)
    mids = [("t<sub>1</sub>", bx + 8), ("t<sub>2</sub>", bx + 72), ("t<sub>3</sub>", bx + 140), ("t<sub>4</sub>", bx + 204)]
    for name, x in mids:
        c.box(name, x, r2, tw, th, color=YELLOW)
    # child -> parent edges (is_child_of); a term may have several parents.
    c.arrow(mids[0][1] + tw / 2, r2, ns[0][1] + tw / 2, r1 + th)
    c.arrow(mids[1][1] + tw / 2, r2, ns[0][1] + tw / 2, r1 + th)
    c.arrow(mids[1][1] + tw / 2, r2, ns[1][1] + tw / 2, r1 + th)
    c.arrow(mids[2][1] + tw / 2, r2, ns[1][1] + tw / 2, r1 + th)
    c.arrow(mids[3][1] + tw / 2, r2, ns[2][1] + tw / 2, r1 + th)
    leaves = [("t<sub>5</sub>", bx + 8), ("t<sub>6</sub>", bx + 72), ("t<sub>7</sub>", bx + 140)]
    for name, x in leaves:
        c.box(name, x, r3, tw, th, color=YELLOW)
    c.arrow(leaves[0][1] + tw / 2, r3, mids[0][1] + tw / 2, r2 + th)
    c.arrow(leaves[1][1] + tw / 2, r3, mids[1][1] + tw / 2, r2 + th)
    c.arrow(leaves[2][1] + tw / 2, r3, mids[2][1] + tw / 2, r2 + th)
    c.arrow(leaves[2][1] + tw / 2, r3, mids[3][1] + tw / 2, r2 + th)
    # gene input layer: one box per gene, deleted genes zeroed (red).
    genes = [("g<sub>1</sub>", 1), ("g<sub>2</sub>", 0), ("g<sub>3</sub>", 1), ("g<sub>4</sub>", 1), ("g<sub>5</sub>", 0), ("g<sub>6</sub>", 1)]
    gw, gs = 36, 6
    gx0 = bx + 8
    gene_x = {}
    for i, (g, s) in enumerate(genes):
        x = gx0 + i * (gw + gs)
        gene_x[i] = x + gw / 2
        c.box(f"{g} = {s}", x, r4, gw, th, color=(RED if s == 0 else GRAY))
    # dashed: annotated gene states feed the terms the genes are annotated to.
    targets = {0: leaves[0], 1: leaves[0], 2: leaves[1], 3: leaves[1], 4: leaves[2], 5: mids[3]}
    for i, (name, x) in targets.items():
        ty = r3 + th if (name, x) in leaves else r2 + th
        offset = -8 if i % 2 == 0 else 8
        c.arrow(gene_x[i], r4, x + tw / 2 + offset, ty, color=(RED[0] if genes[i][1] == 0 else GRAY[0]), dashed=True)
    c.text("edges: is_child_of (child &#8594; parent); dashed: gene states s" + sub("genes(t)") + " entering the subsystems that annotate them; subsystems run by stratum, leaves first (12 &#8594; 0)",
           bx, r4 + th + 4, 270, 36)

    # ---- Column C: subsystem definition, readout, loss.
    cx, cw = 462, 240
    c.box("subsystem t (one per GO term)<br>I<sub>t</sub> = [ O<sub>c</sub> for children c of t &#8214; s<sub>genes(t)</sub> ]<br>"
          "O<sub>t</sub> = tanh(BN(W<sub>t</sub> I<sub>t</sub> + b<sub>t</sub>)) &#8712; [&#8722;1, 1]<sup>L<sub>t</sub></sup><br>"
          "L<sub>t</sub> = max(20, &#8968;0.3 &#183; |genes(t)|&#8969;)",
          cx, y0 + 20, cw, 50, color=YELLOW, align="left")
    c.box("root readout&nbsp; &#375; = w<sub>r</sub><sup>&#8868;</sup> O<sub>ROOT</sub> + b<sub>r</sub>&nbsp; (trigenic interaction y<sub>int</sub>)",
          cx, y0 + 78, cw, 16, color=ORANGE, align="left")
    c.box("auxiliary heads on every t &#8800; r:&nbsp; &#375;<sub>t</sub> = w<sub>t</sub><sup>&#8868;</sup> O<sub>t</sub> + b<sub>t</sub>",
          cx, y0 + 100, cw, 16, color=PURPLE, align="left")
    c.box("loss = MSE(&#375;, y) + &#945; &#183; mean<sub>t &#8800; r</sub> MSE(&#375;<sub>t</sub>, y),&nbsp; &#945; = 0.3<br>"
          "weight decay in AdamW replaces &#955;&#8214;W&#8214;<sub>2</sub>; init U(&#8722;0.001, 0.001)",
          cx, y0 + 122, cw, 30, color=GRAY, align="left")
    c.arrow(cx + cw / 2, y0 + 70, cx + cw / 2, y0 + 78, color=GRAY[0])
    c.arrow(cx + cw / 2, y0 + 94, cx + cw / 2, y0 + 100, color=GRAY[0])
    c.arrow(cx + cw / 2, y0 + 116, cx + cw / 2, y0 + 122, color=GRAY[0])
    c.text("built on the filtered DAG: 2,655 subsystems, 3,208 edges, 13 strata, 20.6 M parameters",
           cx, y0 + 158, cw, 24)
    return y0 + 204


def main():
    c = Canvas()
    c.letter("a", 0, 0)
    bottom = schematic(c, 4)
    y2 = bottom + GAP_Y
    panels = ["dcell_model_terms_per_stratum", "dcell_model_genes_per_term", "dcell_model_terms_per_gene"]
    paths = [osp.join(IMG_DIR, f"{p}.svg") for p in panels]
    widths = [svg_size(p)[0] for p in paths]
    gap_x = (FULL_WIDTH - sum(widths)) / (len(paths) - 1)
    x = 0.0
    h2 = 0.0
    for letter, path in zip("bcd", paths):
        w, h = c.image(path, x, y2)
        c.letter(letter, x - 2, y2 - 4)
        x += w + gap_x
        h2 = max(h2, h)
    extent_h = y2 + h2
    xml = (
        f'<mxfile host="{osp.basename(__file__)}">'
        f'<diagram id="{NAME}" name="Page-1">'
        '<mxGraphModel dx="0" dy="0" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" '
        'arrows="1" fold="1" page="1" pageScale="1" pageWidth="850" pageHeight="1100" math="0" shadow="0">'
        '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(c.cells) + "</root></mxGraphModel></diagram></mxfile>"
    )
    out = osp.join(DRAWIO_DIR, f"{NAME}.drawio")
    with open(out, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"wrote {out}: {FULL_WIDTH} x {extent_h:.0f} units = {FULL_WIDTH / 3.937:.1f} x {extent_h / 3.937:.1f} mm")
    if extent_h > 669:
        raise SystemExit(f"{NAME} exceeds the Nature print height (669 units)")


if __name__ == "__main__":
    main()
