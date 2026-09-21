"""Generate the two-families explainer, panel a of the 028 summary figure.

    python notes/assets/drawio/028-families.gen.py notes/assets/drawio

writes 028-families-explainer.drawio: one gene deletion scored by two model families
whose standardized logits are averaged. 705 units wide (179 mm at draw.io's 100 units per
inch) by 186 tall (47 mm); fonts on the ladder only (8.3 body, 9.7 box titles, 11.1 bold
for the panel letter). The colors are the summary figure's, one meaning each: orange the
transformer reading the metabolic solve, red message passing over the gene graphs, purple
the equal-weight average of the two, gray a comparator. The genome-scale model keeps the
readout diagrams' style, the light purple fill with a DASHED primary-purple border (a
fixed table); the average is the terminal node and takes the dark-tier purple line, solid
and heavier, so the two purples are not read as one thing. The offline metabolic columns
are white with a dashed gray border: inputs, nothing learned. Yellow is left out because
the summary figure spends it on models that read no solve and no graph.

Every number is in notes-tex/028-gene-essentiality: 0.955, 0.934, 0.917, 0.742 and 0.640
in sections/8-summary.tex (first description item), 1,121 and 2,362 in
sections/1-endpoint.tex, 4,131 in sections/8-summary.tex, the two convolution layers
over the union of nine graphs in sections/2-comparators.tex, and the per-model
standardization in experiments/028-gene-essentiality/scripts/family_ensembles.py
(`family_score`).
"""

from __future__ import annotations

import os.path as osp
import sys
from xml.sax.saxutils import escape

OUT_DIR = sys.argv[1]

W = 705
FS = "8.3"
FS_T = "9.7"
FS_P = "11.1"

WHITE = ("#FFFFFF", "#666666")
ORANGE = ("#FFE6CC", "#D79B00")
RED = ("#F8CECC", "#B85450")
PURPLE = ("#E1D5E7", "#9673A6")
PURPLE_DARK = ("#E1D5E7", "#6E5479")
GRAY = ("#F5F5F5", "#666666")


class Canvas:
    def __init__(self) -> None:
        self.cells: list[str] = []
        self._id = 10
        self.height = 0.0

    def nid(self) -> str:
        self._id += 1
        return str(self._id)

    def box(
        self,
        title: str,
        body: str,
        x: float,
        y: float,
        w: float,
        h: float,
        color: tuple[str, str],
        dashed: bool = False,
        stroke_width: float = 0.8,
    ) -> str:
        fill, stroke = color
        i = self.nid()
        label = f'<span style="font-size: {FS_T}px"><b>{title}</b></span>'
        if body:
            label += "<br>" + body
        style = (
            f"rounded=1;arcSize=10;whiteSpace=wrap;html=1;fontFamily=Arial;fontSize={FS};"
            f"fillColor={fill};strokeColor={stroke};strokeWidth={stroke_width};"
            "align=center;verticalAlign=middle;spacing=2;spacingLeft=3;spacingRight=3;"
        )
        if dashed:
            style += "dashed=1;dashPattern=4 3;"
        self.cells.append(
            f'<mxCell id="{i}" value="{escape(label, {chr(34): "&quot;"})}" style="{style}" '
            f'vertex="1" parent="1"><mxGeometry x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" '
            f'height="{h:.2f}" as="geometry"/></mxCell>'
        )
        self.height = max(self.height, y + h)
        return i

    def text(
        self, value: str, x: float, y: float, w: float, h: float, fs: str = FS,
        align: str = "left", bold: bool = False, color: str = "#000000",
    ) -> None:
        i = self.nid()
        style = (
            f"text;html=1;align={align};verticalAlign=middle;fontFamily=Arial;fontSize={fs};"
            f"fontColor={color};strokeColor=none;fillColor=none;whiteSpace=wrap;spacing=0;"
            + ("fontStyle=1;" if bold else "")
        )
        self.cells.append(
            f'<mxCell id="{i}" value="{escape(value, {chr(34): "&quot;"})}" style="{style}" '
            f'vertex="1" parent="1"><mxGeometry x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" '
            f'height="{h:.2f}" as="geometry"/></mxCell>'
        )
        self.height = max(self.height, y + h)

    def arrow(
        self,
        src: str,
        dst: str,
        exit_: tuple[float, float],
        entry: tuple[float, float],
    ) -> None:
        i = self.nid()
        style = (
            "endArrow=blockThin;endFill=1;endSize=4;html=1;strokeColor=#000000;"
            "strokeWidth=0.8;rounded=0;"
            f"exitX={exit_[0]};exitY={exit_[1]};exitDx=0;exitDy=0;"
            f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
        )
        self.cells.append(
            f'<mxCell id="{i}" value="" style="{style}" edge="1" parent="1" '
            f'source="{src}" target="{dst}"><mxGeometry relative="1" as="geometry"/></mxCell>'
        )

    def write(self, path: str, name: str) -> None:
        h = int(self.height + 0.5)
        xml = (
            f'<mxfile><diagram id="{name}" name="Page-1"><mxGraphModel dx="0" dy="0" grid="1" '
            'gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" '
            f'pageScale="1" pageWidth="{W}" pageHeight="{h}" math="0" shadow="0">'
            '<root><mxCell id="0"/><mxCell id="1" parent="0"/>'
            + "".join(self.cells)
            + "</root></mxGraphModel></diagram></mxfile>"
        )
        with open(path, "w") as fh:
            fh.write(xml)
        print(
            f"{osp.basename(path)}: canvas {W} x {h} units = "
            f"{W * 0.254:.1f} x {h * 0.254:.1f} mm"
        )


def build() -> Canvas:
    c = Canvas()
    gap = 14
    h1, h2, h3 = 58, 50, 50
    y1 = 0
    y2 = y1 + h1 + gap
    y3 = y2 + h2 + gap
    bottom = y3 + h3

    # Panel letter at the outer top-left; the content starts to its right.
    c.text("a", 0, 0, 12, 14, fs=FS_P, bold=True)

    x_in, w_in = 16, 98
    x_a, w_a = x_in + w_in + gap, 112
    x_b, w_b = x_a + w_a + gap, 160
    x_c, w_c = x_b + w_b + gap, 150
    x_d = x_c + w_c + gap
    w_d = W - x_d

    inp = c.box(
        "one gene deletion",
        "label: essential or not<br>1,121 labeled<br>yeast-GEM genes,<br>195 of them the<br>released held-out set",
        x_in, y1, w_in, bottom - y1, WHITE,
    )

    # Lane 1: what is computed offline from the genome-scale model.
    gem = c.box(
        "yeast-GEM",
        "fixed stoichiometry, bounds and gene rules; 4,131 reactions",
        x_a, y1, w_a, h1, PURPLE, dashed=True,
    )
    x_cols = x_b
    w_cols = x_c + 116 - x_b
    cols = c.box(
        "metabolic columns, computed offline",
        "FBA growth ratios on four media<br>log(1 + |v|) of one parsimonious FBA solve "
        "per genotype and medium<br>six gene-protein-reaction rule counts",
        x_cols, y1, w_cols, h1, WHITE, dashed=True,
    )
    x_cmp = x_cols + w_cols + gap
    c.box(
        "comparators",
        "on the same 195 genes<br>FBA growth ratio alone 0.640<br>Flux Cone Learning 0.742",
        x_cmp, y1, W - x_cmp, h1, GRAY,
    )

    # Lane 2: family 1, the transformer reading the solve.
    c.text(
        "family 1: the cell graph transformer reading the metabolic solve",
        x_a, y2 - gap, 300, gap, color="#9D7100",
    )
    tok = c.box("gene tokens", "pathway multi-hot and ProtT5", x_a, y2, w_a, h2, ORANGE)
    cgt = c.box(
        "cell graph transformer",
        "attention regularized toward nine gene graphs; the deleted gene's token is pooled",
        x_b, y2, w_b, h2, ORANGE,
    )
    f1 = c.box(
        "logit + affine",
        "a zero-initialized affine adds the metabolic columns<br>five-fold ensemble 0.934",
        x_c, y2, w_c, h2, ORANGE,
    )

    # Lane 3: family 2, message passing on the gene graphs.
    c.text(
        "family 2: message passing on the gene graphs",
        x_a, y3 - gap, 300, gap, color="#863D3A",
    )
    go = c.box(
        "node features", "propagated GO multi-hot, 2,362 columns", x_a, y3, w_a, h3, RED
    )
    gcn = c.box(
        "graph convolution",
        "two layers over the union of<br>nine gene-gene graphs",
        x_b, y3, w_b, h3, RED,
    )
    f2 = c.box(
        "one logit per gene", "five-fold ensemble 0.917", x_c, y3, w_c, h3, RED
    )

    avg = c.box(
        "average of the two families",
        "each model's logits standardized, then equal weights, nothing fitted"
        "<br><b>held-out AUROC 0.955</b>",
        x_d, y2, w_d, bottom - y2, PURPLE_DARK, stroke_width=1.6,
    )

    span = bottom - y1
    c.arrow(inp, gem, ((0 + 1), (y1 + h1 / 2 - y1) / span), (0, 0.5))
    c.arrow(inp, tok, (1, (y2 + h2 / 2 - y1) / span), (0, 0.5))
    c.arrow(inp, go, (1, (y3 + h3 / 2 - y1) / span), (0, 0.5))
    c.arrow(gem, cols, (1, 0.5), (0, 0.5))
    # The columns box overhangs the logit box; drop straight down into it.
    x_drop = x_c + 30
    c.arrow(cols, f1, ((x_drop - x_cols) / w_cols, 1), ((x_drop - x_c) / w_c, 0))
    c.arrow(tok, cgt, (1, 0.5), (0, 0.5))
    c.arrow(cgt, f1, (1, 0.5), (0, 0.5))
    c.arrow(go, gcn, (1, 0.5), (0, 0.5))
    c.arrow(gcn, f2, (1, 0.5), (0, 0.5))
    avg_span = bottom - y2
    c.arrow(f1, avg, (1, 0.5), (0, (h2 / 2) / avg_span))
    c.arrow(f2, avg, (1, 0.5), (0, (y3 + h3 / 2 - y2) / avg_span))
    return c


canvas = build()
canvas.write(osp.join(OUT_DIR, "028-families-explainer.drawio"), "families-explainer")
