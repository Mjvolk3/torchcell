# notes/assets/drawio/thesis-fig-ch02-kv-vs-graph.gen.py
# [[paper.nature-biotech.si-database-figures]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes/assets/drawio/thesis-fig-ch02-kv-vs-graph.gen.py
"""Generate thesis-fig-ch02-kv-vs-graph.drawio: why the database is a graph.

Thesis Fig. 2.11 (``fig:ch02-kv-vs-graph``), placed after "What a query returns" in the
subsection "The queried dataset". Redrawn to the thesis standard from
``notes/assets/drawio/kv-iteration-vs-graph-db.drawio.png`` (an editable draw.io PNG, added
2024-01-11 in 7af5afbb14, task note 2024.01.08 "Graphic for comparing key value querying vs
graph db querying"), which is also slide 13 ("Database Comparison -- Neo4j with Biocypher") and
slide 14 ("Querying is Scalable for Graph Database but not for Key Value Store") of two 2024
group-update decks. The original stays as it is for the notes that embed it.

a. Twelve records of three datasets D1 to D3 kept as key-value pairs, each value one record with
   its medium above the rest of the record. A query that names a medium has to read every value
   to test it: 12 read, 6 returned.
b. The same records as a graph. Each record links to its dataset node and to its medium node, one
   node per medium (TorchCell hashes the medium's identity into the node id,
   ``CellAdapter._media_node_id``, so every record grown on YEPD meets one YEPD node). The query
   starts at the YEPD node and follows its 6 edges: 6 followed, 6 returned. One drawn line stands
   for the two edges the graph holds, medium to environment to experiment.

Which records are YEPD follows the original's blue/red pattern: records 1-4, 6 and 7. The
original's metadata symbol M and instance symbol I are replaced by a named medium and a record
number, because the chapter already uses calligraphic I for the corpus of contingent
observations.

Layout: 650 units = 165 mm (the thesis text width), placed at true size with no rescale. Fonts
on the ladder: 8.3 Arial body (5.98 pt), 11.1 bold panel letters. Colors: the pale fills of the
locked palette with their strokes. The medium the query names is steel blue (slot 5), the other
medium light purple (slot 3; author, 2026-10-29, it was white), records are the beige record of
FigS-queried-dataset (sand stroke, slot 16), a record the query never reads is gray (slot 6), an
edge followed black, one not followed the charcoal fill (slot 12). Two stronger recolors on
2026-10-28 (red/sand, then solid blue/yellow primaries) were tried and the author chose the
pale original with the purple SC. Every text is a one-line cell, held at least 0.4 mm from every
line (checked by ``common/drawio_text_clearance.py --check`` in the thesis).

Run from the repo root:
    python notes/assets/drawio/thesis-fig-ch02-kv-vs-graph.gen.py
"""

from __future__ import annotations

from xml.sax.saxutils import quoteattr

OUT = "notes/assets/drawio/thesis-fig-ch02-kv-vs-graph.drawio"

NAMED = ("#6C8EBF", "#DAE8FC")  # medium named in the query (palette slot 5)
OTHER = ("#9673A6", "#E1D5E7")  # another medium (slot 3, light purple)
WHITE = ("#666666", "#FFFFFF")  # the dataset nodes
READ = ("#D2AE7D", "#F5EEDD")  # a record the query reads (beige of FigS-queried-dataset)
UNREAD = ("#666666", "#F5F5F5")  # a record the query never reads (slot 6)
FOLLOWED = "#000000"  # an edge the query follows
FAINT = "#CACACA"  # an edge the query does not follow (slot 12 fill)
BLACK = "#000000"

FS = 8.3
FS_LETTER = 11.1
FULL_W = 650

# Panel a
A_X = 0
LABEL_W = 14  # "D1" column
REC_X0 = 30  # first record box
REC_W, TOP_H, BOT_H = 34, 13, 14
REC_PITCH = 50
ROW_Y0 = 26
ROW_P = 52
FOOT_Y = 170

# Panel b
B_X = 328
NODE_D = 20
D_Y, R_Y, M_Y = 40, 98, 150
M_W, M_H = 44, 20
R_PITCH, GROUP_GAP = 25, 12

DATASETS = [4, 5, 3]  # records per dataset
YEPD = {1, 2, 3, 4, 6, 7}  # records grown on the medium the query names


class Canvas:
    def __init__(self) -> None:
        self.cells: list[str] = []
        self.n = 1

    def _id(self) -> str:
        self.n += 1
        return f"c{self.n}"

    def _cell(self, value: str, style: str, x: float, y: float, w: float, h: float) -> None:
        self.cells.append(
            f'<mxCell id="{self._id()}" value={quoteattr(value)} style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" as="geometry"/></mxCell>'
        )

    def box(self, text, x, y, w, h, color, stroke_w=0.75, shape="rounded=0", font_color="#000000"):
        style = (
            f"{shape};whiteSpace=nowrap;html=1;fontFamily=Arial;fontSize={FS};fontColor={font_color};"
            f"fillColor={color[1]};strokeColor={color[0]};strokeWidth={stroke_w};"
            "align=center;verticalAlign=middle;spacing=0;"
        )
        self._cell(text, style, x, y, w, h)

    def text(self, text, x, y, w, h, fs=FS, bold=False, align="left", color="#000000"):
        style = (
            f"text;html=1;whiteSpace=nowrap;fontFamily=Arial;fontSize={fs};align={align};verticalAlign=middle;"
            f"fontColor={color};spacing=0;spacingLeft=0;spacingRight=0;spacingTop=0;spacingBottom=0;"
        )
        if bold:
            style += "fontStyle=1;"
        self._cell(text, style, x, y, w, h)

    def node(self, text, cx, cy, w, h, color, stroke_w=0.75, font_color="#000000"):
        self.box(text, cx - w / 2, cy - h / 2, w, h, color, stroke_w=stroke_w, shape="ellipse", font_color=font_color)

    def path(self, pts, color=BLACK, width=0.75, head="none"):
        """A polyline through ``pts`` (first and last are its ends), sharp corners."""
        style = (
            f"endArrow={head};endFill=1;endSize=3;html=1;strokeWidth={width};strokeColor={color};"
            "edgeStyle=none;rounded=0;"
        )
        (x1, y1), (x2, y2) = pts[0], pts[-1]
        mids = "".join(f'<mxPoint x="{x:.2f}" y="{y:.2f}"/>' for x, y in pts[1:-1])
        arr = f'<Array as="points">{mids}</Array>' if mids else ""
        self.cells.append(
            f'<mxCell id="{self._id()}" value="" style="{style}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
            f'<mxPoint x="{x1:.2f}" y="{y1:.2f}" as="sourcePoint"/><mxPoint x="{x2:.2f}" y="{y2:.2f}" as="targetPoint"/>'
            f"{arr}</mxGeometry></mxCell>"
        )

    def letter(self, ch, x, y):
        self.text(ch, x, y, 10, 14, fs=FS_LETTER, bold=True)

    def write(self, path: str, w: int, h: int) -> None:
        body = "\n".join(self.cells)
        xml = (
            '<mxfile host="app.diagrams.net" agent="claude-code">\n'
            '  <diagram name="thesis-fig-ch02-kv-vs-graph" id="thesis-fig-ch02-kv-vs-graph">\n'
            f'    <mxGraphModel dx="1400" dy="1000" grid="0" page="1" pageWidth="{w}" pageHeight="{h}" math="0" shadow="0">\n'
            '      <root>\n        <mxCell id="0"/>\n        <mxCell id="1" parent="0"/>\n'
            f"{body}\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n"
        )
        with open(path, "w") as fh:
            fh.write(xml)


def ellipse_point(cx, cy, a, b, tx, ty):
    """Where the ray from the ellipse's center (cx, cy) toward (tx, ty) leaves the ellipse."""
    dx, dy = tx - cx, ty - cy
    t = 1.0 / ((dx / a) ** 2 + (dy / b) ** 2) ** 0.5
    return cx + dx * t, cy + dy * t


def panel_a(c: Canvas) -> None:
    c.letter("a", A_X, 0)
    c.text("Records as key&#8211;value pairs, read in sequence", A_X + 12, 0, 280, 14)
    k = 1
    ends = []  # (right edge x, center y) of each row's last record
    starts = []  # (left edge x, center y) of each row's first record
    for r, n in enumerate(DATASETS):
        y = ROW_Y0 + r * ROW_P
        cy = y + (TOP_H + BOT_H) / 2
        c.text(f"D{r + 1}", A_X, cy - 7, LABEL_W, 14)
        for i in range(n):
            x = A_X + REC_X0 + i * REC_PITCH
            if k in YEPD:
                c.box("YEPD", x, y, REC_W, TOP_H, NAMED)
            else:
                c.box("SC", x, y, REC_W, TOP_H, OTHER)
            c.box(str(k), x, y + TOP_H, REC_W, BOT_H, READ)
            if i > 0:
                c.path([(x - REC_PITCH + REC_W + 3, cy), (x - 3, cy)], head="classic")
            k += 1
        starts.append((A_X + REC_X0, cy))
        ends.append((A_X + REC_X0 + (n - 1) * REC_PITCH + REC_W, cy))
    # The read continues from the last record of one dataset to the first of the next.
    for r in range(len(DATASETS) - 1):
        (xe, ye), (xs, ys) = ends[r], starts[r + 1]
        y_mid = ROW_Y0 + r * ROW_P + TOP_H + BOT_H + (ROW_P - TOP_H - BOT_H) / 2
        x_left = A_X + LABEL_W + 7
        c.path([(xe + 3, ye), (xe + 9, ye), (xe + 9, y_mid), (x_left, y_mid), (x_left, ys), (xs - 3, ys)], head="classic")
    c.text("12 records read, 6 returned", A_X + REC_X0, FOOT_Y, 240, 12)


def panel_b(c: Canvas) -> None:
    c.letter("b", B_X - 10, 0)
    c.text("Records as a graph, walked from the medium", B_X + 2, 0, 300, 14)
    # Record centers: dataset groups of 4, 5 and 3 on one row.
    rx = {}
    x = B_X + 10
    k = 1
    groups = []
    for n in DATASETS:
        g = []
        for _ in range(n):
            rx[k] = x
            g.append(k)
            k += 1
            x += R_PITCH
        groups.append(g)
        x += GROUP_GAP
    r = NODE_D / 2
    # Dataset nodes over the middle of their records.
    dx = {}
    for j, g in enumerate(groups):
        dx[j] = sum(rx[i] for i in g) / len(g)
    # Medium nodes under the mean of the records they hold.
    yepd_x = sum(rx[i] for i in YEPD) / len(YEPD)
    sc = [i for i in rx if i not in YEPD]
    sc_x = sum(rx[i] for i in sc) / len(sc)

    def edge(p, a1, b1, q, a2, b2, color, width):
        s = ellipse_point(p[0], p[1], a1, b1, q[0], q[1])
        t = ellipse_point(q[0], q[1], a2, b2, p[0], p[1])
        c.path([s, t], color=color, width=width)

    # Edges first, so nodes sit on top. Not followed: faint; followed: black.
    for j, g in enumerate(groups):
        for i in g:
            edge((dx[j], D_Y), r, r, (rx[i], R_Y), r, r, FAINT, 0.75)
    for i in sc:
        edge((sc_x, M_Y), M_W / 2, M_H / 2, (rx[i], R_Y), r, r, FAINT, 0.75)
    for i in sorted(YEPD):
        edge((yepd_x, M_Y), M_W / 2, M_H / 2, (rx[i], R_Y), r, r, FOLLOWED, 1.0)
    for j in dx:
        c.node(f"D{j + 1}", dx[j], D_Y, NODE_D, NODE_D, WHITE)
    for i in rx:
        c.node(str(i), rx[i], R_Y, NODE_D, NODE_D, READ if i in YEPD else UNREAD)
    c.node("YEPD", yepd_x, M_Y, M_W, M_H, NAMED, stroke_w=1.0)
    c.node("SC", sc_x, M_Y, M_W, M_H, OTHER)
    c.text("6 edges followed, 6 records returned", B_X + 2, FOOT_Y, 240, 12)


def key_row(c: Canvas, y: float) -> None:
    """One row naming every color and line style."""
    x = A_X
    entries = [
        ("box", NAMED, "medium named in the query"),
        ("box", OTHER, "another medium"),
        ("box", READ, "record read"),
        ("box", UNREAD, "record not read"),
        ("line", (FOLLOWED, 1.0), "edge followed"),
        ("line", (FAINT, 0.75), "edge not followed"),
    ]
    widths = [128, 82, 68, 84, 76, 90]
    for (kind, col, label), w in zip(entries, widths):
        if kind == "box":
            c.box("", x + 1, y + 2, 14, 9, col)
        else:
            c.path([(x + 1, y + 6.5), (x + 15, y + 6.5)], color=col[0], width=col[1])
        c.text(label, x + 19, y, w - 19, 13)
        x += w + 8


def main() -> None:
    c = Canvas()
    panel_a(c)
    panel_b(c)
    key_y = FOOT_Y + 20
    key_row(c, key_y)
    h = int(key_y + 13)
    c.write(OUT, FULL_W, h)
    print(f"wrote {OUT}: {FULL_W} x {h} units ({FULL_W / 3.937:.1f} x {h / 3.937:.1f} mm)")


if __name__ == "__main__":
    main()
