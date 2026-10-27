# notes/assets/drawio/FigS-queried-dataset.gen.py
# [[paper.nature-biotech.si-database-figures]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes/assets/drawio/FigS-queried-dataset.gen.py
# Copied 2026-10-26 from the uncommitted worktree fig/si-database-panels (same path there)
# for manuscript SI figures; that copy is the origin of this file.
"""Generate FigS-queried-dataset.drawio: one row, three panels, on what a query returns.

Panel a: the Cypher query that selects experiment records from the knowledge graph, and the
``Neo4jCellDataset`` call that turns them into a dataset (real keyword names,
torchcell/data/neo4j_cell.py). Panel b: the three steps that collapse records to one
instance per genotype: convert (a non-fitness label to fitness), deduplicate (equal
genotypes averaged), aggregate (different labels of one genotype joined). Panel c: the
instance the model receives, drawn and printed (the print is the real one in
notes/torchcell.data.neo4j_cell.md, abridged to the fields the drawing shows).

Class names are real: CompositeFitnessConverter (torchcell/datamodels/
fitness_composite_conversion.py), MeanExperimentDeduplicator (torchcell/data/
mean_experiment_deduplicate.py), GenotypeAggregator (torchcell/data/genotype_aggregate.py),
SubgraphRepresentation (torchcell/data/graph_processor.py). The query is the shape of
experiments/003-fit-int/queries/fit-int.cql. Label values in b are symbols, not numbers.

Layout: full width (696 units = 177 mm), COL_GAP 12, a 16-unit letter strip. Fonts on the
ladder: 8.3 Arial body, 8.3 Courier New code, 11.1 bold panel letters. Shapes use palette
slots 1-6; code uses the GitHub light scheme the thesis substrate figure uses.

Run from the repo root:
    python notes/assets/drawio/FigS-queried-dataset.gen.py
"""

from __future__ import annotations

import re
from xml.sax.saxutils import quoteattr

OUT = "notes/assets/drawio/FigS-queried-dataset.drawio"

ORANGE = ("#D79B00", "#FFE6CC")
RED = ("#B85450", "#F8CECC")
# 2026-10-23 (author): the metabolite purple is the palette's dark purple, plum (slot 9), not
# the primary lilac, and the record boxes of b and c are sand (slot 16, the beige) rather than
# yellow. PLOT_PALETTE / PLOT_PALETTE_FILL in torchcell/paper/plot.py.
PURPLE = ("#6E5479", "#CAB8D4")  # plum, slot 9 (was lilac, "#9673A6", "#E1D5E7")
YELLOW = ("#D2AE7D", "#F5EEDD")  # the beige background of Fig1-torchcell-data (sand stroke, slot 16); was wheat "#D6B656"/"#FFF2CC", then sand fill "#EBD6BC", too dark (author, 2026-10-23)
BLUE = ("#6C8EBF", "#DAE8FC")
GRAY = ("#666666", "#F5F5F5")
WHITE = ("#666666", "#FFFFFF")

FS = 8.3  # 8.5 in the source worktree; 8.5 is off the CLAUDE.md ladder (7 / 8.3 / 9.7 / 11.1)
FS_CODE = 8.3
FS_LETTER = 11.1
CODE_PITCH = 11
FULL_W = 696
COL_GAP = 12
STRIP = 16

A_X, A_W = 0, 226
B_X, B_W = A_X + A_W + COL_GAP, 190
C_X, C_W = B_X + B_W + COL_GAP, FULL_W - (B_X + B_W + COL_GAP)
ROW_Y = STRIP

# Code colors: the GitHub light scheme (comments, keywords, classes, strings, numbers,
# node labels, relationship types, keyword arguments / fields).
C_TXT = "#24292E"
C_CMT = "#6A737D"
C_KW = "#D73A49"
C_CLS = "#6F42C1"
C_STR = "#032F62"
C_NUM = "#005CC5"
C_LBL = "#005CC5"
C_REL = "#E36209"
C_ARG = "#E36209"

EDGE_W = 1.25  # graph edges in panel c


class Canvas:
    def __init__(self) -> None:
        self.cells: list[str] = []
        self.n = 1

    def _id(self) -> str:
        self.n += 1
        return f"c{self.n}"

    def box(self, text, x, y, w, h, color=GRAY, fs=FS, bold=False, dashed=False, stroke_w=0.75, align="center", fill=True, rounded=0, valign="middle", arc=4):
        style = (
            f"rounded={rounded};arcSize={arc};whiteSpace=wrap;html=1;fontFamily=Arial;fontSize={fs};"
            f"fillColor={color[1] if fill else 'none'};strokeColor={color[0]};strokeWidth={stroke_w};"
            f"align={align};verticalAlign={valign};spacing=2;spacingLeft=3;spacingRight=3;"
        )
        if bold:
            style += "fontStyle=1;"
        if dashed:
            style += "dashed=1;"
        self.cells.append(
            f'<mxCell id="{self._id()}" value={quoteattr(text)} style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )

    def text(self, text, x, y, w, h, fs=FS, bold=False, align="left", valign="top", color="#000000", font="Arial"):
        style = (
            f"text;html=1;whiteSpace=wrap;fontFamily={font};fontSize={fs};align={align};verticalAlign={valign};"
            f"fontColor={color};spacing=0;spacingLeft=0;spacingRight=0;spacingTop=0;"
        )
        if bold:
            style += "fontStyle=1;"
        self.cells.append(
            f'<mxCell id="{self._id()}" value={quoteattr(text)} style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )

    def code(self, lines, x, y, w, pitch=CODE_PITCH):
        """Courier lines, one text cell each (no wrapping). ``lines`` are HTML-escaped
        fragments; spaces are kept with &nbsp;. Returns the bottom y."""
        for i, line in enumerate(lines):
            # Non-breaking spaces only in the text, never inside a tag: a space turned into
            # &nbsp; inside <font color="..."> breaks the tag and the sanitizer drops it.
            html = "".join(p if p.startswith("<") else p.replace(" ", "&nbsp;") for p in re.split(r"(<[^>]+>)", line))
            style = (
                f"text;html=1;whiteSpace=nowrap;fontFamily=Courier New;fontSize={FS_CODE};align=left;"
                f"verticalAlign=middle;fontColor={C_TXT};spacing=0;spacingLeft=0;spacingRight=0;spacingTop=0;"
            )
            self.cells.append(
                f'<mxCell id="{self._id()}" value={quoteattr(html)} style="{style}" vertex="1" parent="1">'
                f'<mxGeometry x="{x}" y="{y + i * pitch}" width="{w}" height="{pitch}" as="geometry"/></mxCell>'
            )
        return y + len(lines) * pitch

    def code_block(self, lines, x, y, w, pad=6):
        """A code box with its lines inside; the corner arc is small so no line meets it."""
        h = len(lines) * CODE_PITCH + 2 * pad
        self.box("", x, y, w, h, GRAY, rounded=1, arc=4)
        self.code(lines, x + pad, y + pad, w - 2 * pad)
        return y + h

    def bullets(self, items, x, y, w, pitch=12, color="#000000"):
        for i, item in enumerate(items):
            self.text("• " + item, x, y + i * pitch, w, pitch, valign="middle", color=color)
        return y + len(items) * pitch

    def dot(self, x, y, d, color, stroke="#666666", stroke_w=0.75, label="", shape="ellipse"):
        style = (
            f"{shape};whiteSpace=wrap;html=1;aspect=fixed;fontFamily=Arial;fontSize={FS};"
            f"fillColor={color};strokeColor={stroke};strokeWidth={stroke_w};spacing=0;"
        )
        if shape == "rect":
            style = style.replace("rect;", "rounded=0;")
        self.cells.append(
            f'<mxCell id="{self._id()}" value={quoteattr(label)} style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x - d / 2:.2f}" y="{y - d / 2:.2f}" width="{d}" height="{d}" as="geometry"/></mxCell>'
        )

    def line(self, x1, y1, x2, y2, color="#666666", width=0.75, dashed=False, head="none", label=""):
        style = f"endArrow={head};html=1;strokeWidth={width};strokeColor={color};endSize=3;edgeStyle=none;fontFamily=Arial;fontSize={FS};"
        if dashed:
            style += "dashed=1;dashPattern=3 2;"
        self.cells.append(
            f'<mxCell id="{self._id()}" value={quoteattr(label)} style="{style}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
            f'<mxPoint x="{x1:.2f}" y="{y1:.2f}" as="sourcePoint"/><mxPoint x="{x2:.2f}" y="{y2:.2f}" as="targetPoint"/>'
            f"</mxGeometry></mxCell>"
        )

    def edge(self, p, q, r, **kw):
        """A line between two circle boundaries (radius ``r`` at both ends)."""
        dx, dy = q[0] - p[0], q[1] - p[1]
        n = (dx * dx + dy * dy) ** 0.5
        ux, uy = dx / n, dy / n
        self.line(p[0] + ux * r, p[1] + uy * r, q[0] - ux * r, q[1] - uy * r, **kw)

    def cross(self, x, y, s, color, width):
        """An x centered on (x, y) with half-size ``s``."""
        self.line(x - s, y - s, x + s, y + s, color=color, width=width)
        self.line(x - s, y + s, x + s, y - s, color=color, width=width)

    def swatch_legend(self, entries, x, y, w, pitch=12):
        """Rows of (kind, color, label): a colored box, a dot, or a ring beside its label."""
        for kind, col, label in entries:
            if kind == "box":
                self.box("", x + 2, y + 1.5, 14, 9, col, stroke_w=1)
            elif kind == "ring":
                self.dot(x + 9, y + 6, 9, WHITE[1], stroke=col, stroke_w=1.6)
            else:
                self.dot(x + 9, y + 6, 9, col[1], stroke=col[0])
            self.text(label, x + 20, y, w - 20, pitch, valign="middle")
            y += pitch
        return y

    def letter(self, ch, x, y):
        self.text(ch, x - 2, y - 1, 16, 14, fs=FS_LETTER, bold=True)

    def write(self, path: str, w: int, h: int) -> None:
        body = "\n".join(self.cells)
        xml = (
            '<mxfile host="app.diagrams.net" agent="claude-code">\n'
            '  <diagram name="FigS-queried-dataset" id="FigS-queried-dataset">\n'
            f'    <mxGraphModel dx="1400" dy="1000" grid="0" page="1" pageWidth="{w}" pageHeight="{h}" math="0" shadow="0">\n'
            '      <root>\n        <mxCell id="0"/>\n        <mxCell id="1" parent="0"/>\n'
            f"{body}\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n"
        )
        with open(path, "w") as fh:
            fh.write(xml)


# Token colorers for the code lines (HTML fragments).
def col(s: str, color: str, bold: bool = False) -> str:
    inner = f"<b>{s}</b>" if bold else s
    return f'<font color="{color}">{inner}</font>'


def cmt(s):
    return col(s, C_CMT)


def kw(s):
    return col(s, C_KW)


def cls(s):
    return col(s, C_CLS, bold=True)


def st(s):
    return col(s, C_STR)


def num(s):
    return col(s, C_NUM)


def lbl(s):
    return col(s, C_LBL)


def rel(s):
    return col(s, C_REL)


def arg(s):
    return col(s, C_ARG)


PRINT_TYPES = {"gene", "reaction", "metabolite"}


def hl_print(line: str) -> str:
    """Color a printed HeteroData line: numbers, node types, and ``field=`` names."""
    out = []
    tokens = re.findall(r"\d+|[A-Za-z_]+|.", line)
    for i, t in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if t.isdigit():
            out.append(num(t))
        elif t == "HeteroData":
            out.append(cls(t))
        elif t in PRINT_TYPES and nxt != "=":
            out.append(col(t, C_LBL))
        elif re.fullmatch(r"[A-Za-z_]+", t) and nxt == "=":
            out.append(arg(t))
        else:
            out.append(t)
    return "".join(out)


def panel_a(c: Canvas, bottom: float) -> float:
    """The query and the dataset call. Returns the bottom y of its content."""
    x0, y0 = A_X, ROW_Y
    c.letter("a", x0, STRIP - 15)
    c.text("The query and the dataset call", x0, y0, A_W, 12)
    y = y0 + 16
    cypher = [
        cmt("// experiments of one dataset"),
        kw("MATCH") + " (d:" + lbl("Dataset") + ")",
        "  &lt;-[:" + rel("ExperimentMemberOf") + "]-(e:" + lbl("Experiment") + ")",
        kw("WHERE") + " d.id = " + st("'TmiKuzmin2018Dataset'"),
        kw("MATCH") + " (e)&lt;-[:" + rel("GenotypeMemberOf") + "]-(g)",
        kw("MATCH") + " (g)&lt;-[:" + rel("PerturbationMemberOf") + "]-(p)",
        kw("MATCH") + " (e)&lt;-[:" + rel("EnvironmentMemberOf") + "]-(env)",
        kw("MATCH") + " (env)&lt;-[:" + rel("MediaMemberOf") + "]-(m)",
        kw("MATCH") + " (env)&lt;-[:" + rel("TemperatureMemberOf") + "]-(t)",
        kw("MATCH") + " (e)&lt;-[:" + rel("PhenotypeMemberOf") + "]-(phen)",
        kw("MATCH") + " (e)&lt;-[:" + rel("ExperimentReferenceOf") + "]-(ref)",
        kw("WHERE") + " m.name = " + st("'YEPD'") + " " + kw("AND") + " t.value = " + num("30"),
        "  " + kw("AND") + " p.perturbation_type = " + st("'deletion'"),
        kw("RETURN") + " e, ref",
    ]
    y = c.code_block(cypher, x0 + 2, y, A_W - 4) + 6
    call = [
        cmt("# records become one dataset"),
        "dataset = " + cls("Neo4jCellDataset") + "(",
        "  root, " + arg("query") + "=cypher,",
        "  " + arg("gene_set") + "=genome.gene_set,",
        "  " + arg("graphs") + "=graphs,  " + cmt("# the multigraph"),
        "  " + arg("node_embeddings") + "=embeddings,  " + cmt("# nt, prot"),
        "  " + arg("converter") + "=" + cls("CompositeFitnessConverter") + ",",
        "  " + arg("deduplicator") + "=" + cls("MeanExperimentDeduplicator") + ",",
        "  " + arg("aggregator") + "=" + cls("GenotypeAggregator") + ",",
        "  " + arg("graph_processor") + "=" + cls("SubgraphRepresentation") + "(),",
        ")",
        "dataset[" + num("0") + "]  " + cmt("# panel c"),
    ]
    y = c.code_block(call, x0 + 2, y, A_W - 4) + 6
    return y


def panel_b(c: Canvas, bottom: float) -> float:
    """Three steps collapse the returned records to one instance per genotype."""
    x0, y0 = B_X, ROW_Y
    c.letter("b", x0, STRIP - 15)
    c.text("Records collapse to one per genotype", x0, y0, B_W, 12)
    in_w, out_w, bh = 88, 88, 26  # 24 in the source worktree; 26 keeps subscripted labels 0.4 mm off the box line
    ax = x0 + in_w + 2
    ox = x0 + B_W - out_w
    legend = [("box", BLUE, "instance"), ("box", YELLOW, "record")]
    legend_top = bottom - 6 - 12
    steps = [
        ("convert", "CompositeFitnessConverter",
         [("GeneEssentiality", "is_essential = True", YELLOW)], ("Fitness", "fitness = 0", YELLOW)),
        ("deduplicate", "MeanExperimentDeduplicator",
         [("Fitness", "fitness = f<sub>1</sub>", YELLOW), ("Fitness", "fitness = f<sub>2</sub>", YELLOW)],
         ("Fitness", "fitness = mean(f<sub>1</sub>, f<sub>2</sub>)", YELLOW)),
        ("aggregate", "GenotypeAggregator",
         [("Fitness", "fitness = f", YELLOW), ("GeneInteraction", "gene_interaction = g", YELLOW)],
         ("one instance", "fitness = f<br>gene_interaction = g", BLUE)),
    ]
    # Each row: the verb, the class under it, then the boxes; rows spread evenly between
    # the title and the legend.
    head_h = 24  # 23 in the source worktree; 24 clears the class name 0.4 mm above its box
    top = y0 + 18
    row_h = [head_h + bh, head_h + 2 * bh + 4 + 10, head_h + 2 * bh + 4 + 10]
    free = (legend_top - 8 - top) - sum(row_h)
    pitch_gap = free / 2
    y = top
    for (verb, klass, inputs, out), rh in zip(steps, row_h):
        c.text(verb, x0, y, B_W, 12, bold=True, valign="middle")
        c.text(klass, x0, y + 11, B_W, 11, fs=FS_CODE, valign="middle", font="Courier New", color=C_CLS, bold=True)
        iy = y + head_h
        for name, val, colr in inputs:
            c.box(f"<b>{name}</b><br>{val}", x0, iy, in_w, bh, colr)
            iy += bh + 4
        mid = y + head_h + (rh - head_h) / 2
        oh = bh + 10 if "<br>" in out[1] else bh
        c.box(f"<b>{out[0]}</b><br>{out[1]}", ox, mid - oh / 2, out_w, oh, out[2])
        c.line(ax, mid, ox - 2, mid, head="classic")
        if len(inputs) == 2:
            c.text("same genotype", x0, iy - 2, in_w, 10, align="center", valign="middle", color="#666666")
        y += rh + pitch_gap
    c.swatch_legend(legend[:1], x0, legend_top, 70)
    c.swatch_legend(legend[1:], x0 + 70, legend_top, 70)
    return bottom


def panel_c(c: Canvas, bottom: float) -> float:
    """The instance the model receives: drawn, then printed."""
    x0, y0 = C_X, ROW_Y
    c.letter("c", x0, STRIP - 15)
    c.text("The instance the model receives", x0, y0, C_W, 12)
    d = 8
    r = d / 2
    # The slack against ``bottom`` goes above the drawing, between it and the legend, and
    # between the legend and the print, in equal parts.
    natural = y0 + 22 + 58 + 12 + 18 + 14 * CODE_PITCH + 12 + 6
    slack = max(0.0, bottom - natural)
    gy = y0 + 22 + slack / 3
    genes = [(x0 + 14, gy + 10), (x0 + 38, gy + 4), (x0 + 62, gy + 12), (x0 + 24, gy + 34), (x0 + 50, gy + 36), (x0 + 76, gy + 32)]
    phys = [(0, 1), (1, 2), (0, 3), (3, 4), (4, 5), (2, 5)]
    regu = [(1, 4), (2, 4)]
    masked = {3, 5}
    for i, j in phys:
        c.edge(genes[i], genes[j], r, width=EDGE_W)
    for i, j in regu:
        c.edge(genes[i], genes[j], r, width=EDGE_W, dashed=True, head="classic", color=ORANGE[0])
    for k, (gx, gyy) in enumerate(genes):
        if k in masked:
            c.dot(gx, gyy, d, RED[1], stroke=RED[0])
            c.cross(gx, gyy, 2.4, RED[0], 1.2)
        else:
            c.dot(gx, gyy, d, ORANGE[1], stroke=ORANGE[0])
    c.text("gene", x0 + 4, gy + 44, 60, 11, valign="middle")
    # Metabolism: metabolite -- reaction -- metabolite hyperedges.
    mx = x0 + 106
    mets = [(mx, gy + 6), (mx, gy + 36), (mx + 50, gy + 6), (mx + 50, gy + 36)]
    rxn = (mx + 25, gy + 21)
    for m in mets:
        c.edge(m, rxn, r, width=EDGE_W, color=PURPLE[0])
    for m in mets:
        c.dot(*m, d, PURPLE[1], stroke=PURPLE[0])
    c.dot(*rxn, d, GRAY[1], stroke=GRAY[0], shape="rect")
    c.text("metabolite", mx - 12, gy + 44, 60, 11, valign="middle")
    c.text("reaction", mx + 6, gy - 8, 60, 11, valign="middle")
    # Label box (yellow: purple is the metabolite color in this panel).
    c.box("<b>labels</b><br>fitness,<br>gene_interaction", x0 + C_W - 76, gy + 4, 76, 36, YELLOW)
    # Legend, two rows under the drawing; the slack against ``bottom`` is split between
    # the drawing-legend gap and the legend-print gap.
    ly = gy + 58 + slack / 3
    c.dot(x0 + 6, ly + 6, d, ORANGE[1], stroke=ORANGE[0])
    c.text("gene", x0 + 14, ly, 30, 12, valign="middle")
    c.dot(x0 + 50, ly + 6, d, RED[1], stroke=RED[0])
    c.cross(x0 + 50, ly + 6, 2.4, RED[0], 1.2)
    c.text("masked gene", x0 + 58, ly, 60, 12, valign="middle")
    c.dot(x0 + 128, ly + 6, d, PURPLE[1], stroke=PURPLE[0])
    c.text("metabolite", x0 + 136, ly, 50, 12, valign="middle")
    c.dot(x0 + 192, ly + 6, d, GRAY[1], stroke=GRAY[0], shape="rect")
    c.text("reaction", x0 + 200, ly, 50, 12, valign="middle")
    ly += 12
    c.line(x0 + 2, ly + 6, x0 + 16, ly + 6, width=EDGE_W)
    c.text("physical", x0 + 19, ly, 40, 12, valign="middle")
    c.line(x0 + 62, ly + 6, x0 + 76, ly + 6, width=EDGE_W, dashed=True, head="classic", color=ORANGE[0])
    c.text("regulatory", x0 + 79, ly, 46, 12, valign="middle")
    c.line(x0 + 128, ly + 6, x0 + 142, ly + 6, width=EDGE_W, color=PURPLE[0])
    c.text("stoichiometry", x0 + 145, ly, 60, 12, valign="middle")
    # The print, abridged to the fields drawn; the real print is in the module note.
    y = ly + 18 + slack / 3
    printed = [
        "HeteroData(",
        "  gene={ num_nodes=6577, x=[6577, 1600],",
        "         ids_pert=[2], x_pert=[2, 1600],",
        "         fitness=[1], gene_interaction=[1],",
        "         pert_mask=[6579] },",
        "  reaction={ num_nodes=7121 },",
        "  metabolite={ num_nodes=2806 },",
        "  (gene, physical_interaction, gene)=",
        "         { edge_index=[2, 143784] },",
        "  (gene, regulatory_interaction, gene)=",
        "         { edge_index=[2, 16064] },",
        "  (metabolite, reaction, metabolite)=",
        "         { hyperedge_index=[2, 26320],",
        "           stoichiometry=[26320] })",
    ]
    y = c.code_block([hl_print(p) for p in printed], x0 + 2, y, C_W - 4) + 6
    return y


def main() -> None:
    probe = Canvas()
    bottom = max(panel_a(probe, 0), panel_c(probe, 0))
    c = Canvas()
    panel_a(c, bottom)
    panel_b(c, bottom)
    panel_c(c, bottom)
    h = int(bottom) + 2
    c.write(OUT, FULL_W, h)
    print(f"wrote {OUT}: {FULL_W} x {h} units ({FULL_W / 3.9392:.1f} x {h / 3.9392:.1f} mm)")


if __name__ == "__main__":
    main()
