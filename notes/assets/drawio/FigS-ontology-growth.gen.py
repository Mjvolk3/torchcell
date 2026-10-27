# notes/assets/drawio/FigS-ontology-growth.gen.py
# [[paper.nature-biotech.si-database-figures]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes/assets/drawio/FigS-ontology-growth.gen.py
# Copied 2026-10-26 from the uncommitted worktree fig/si-database-panels (same path there)
# for manuscript SI figures; that copy is the origin of this file.
"""Generate FigS-ontology-growth.drawio: one row, three panels, on how the ontology grows
with its datasets. The example is a fitness record (FitnessExperiment) and what happens
when a liquid-growth dataset is added beside a solid-growth one.

Panel a: one record and the classes it touches (its closure, the thing the store manifest
fingerprints). Panel b: a second dataset arrives and needs three things, each with a
different consequence: a vocabulary member that already exists (nothing grows), a new leaf
(the ontology grows, only the new dataset's closure holds it, incremental import), a
change to a shared class (every dataset whose closure holds it rebuilds). Panel c: the
ontology after each admitted dataset, each leaf keeping the color of the dataset that
brought it.

Class names are real (torchcell/datamodels/schema.py): FitnessExperiment, Genotype,
Environment, FitnessPhenotype, DeletionPerturbation, Media and its ``state`` vocabulary
(solid | liquid | gas), AssayType.liquid_od_growth (added 2026-07-20 with the environment
schema). The datasets are schematic (D1, D2, ...): the figure is about the mechanic, not a
measurement, so it carries no numbers.

Layout: full width (702 units = 178 mm), COL_GAP 12 between panels, a 16-unit letter
strip above the row (the white-cross convention of the composed SI figures). Fonts on the
ladder: 8.5 body (6.1 pt), 11.1 bold panel letters. Palette slots 1-6 only.

Run from the repo root (refuses without --force: the .drawio is hand-maintained since
2026-10-20, see main()):
    python notes/assets/drawio/FigS-ontology-growth.gen.py --force
"""

from __future__ import annotations

from xml.sax.saxutils import quoteattr

OUT = "notes/assets/drawio/FigS-ontology-growth.drawio"

ORANGE = ("#D79B00", "#FFE6CC")
RED = ("#B85450", "#F8CECC")
PURPLE = ("#9673A6", "#E1D5E7")
YELLOW = ("#D6B656", "#FFF2CC")
BLUE = ("#6C8EBF", "#DAE8FC")
GRAY = ("#666666", "#F5F5F5")
WHITE = ("#666666", "#FFFFFF")

FS = 8.5
FS_LETTER = 11.1
FULL_W = 696
COL_GAP = 12
STRIP = 16

# Column grid of the row: three panels.
A_X, A_W = 0, 178
B_X, B_W = A_X + A_W + COL_GAP, 282
C_X, C_W = B_X + B_W + COL_GAP, FULL_W - (B_X + B_W + COL_GAP)
ROW_Y = STRIP


class Canvas:
    def __init__(self) -> None:
        self.cells: list[str] = []
        self.n = 1

    def _id(self) -> str:
        self.n += 1
        return f"c{self.n}"

    def box(self, text, x, y, w, h, color=GRAY, fs=FS, bold=False, dashed=False, stroke_w=0.75, align="center", fill=True, rounded=0, dotted=False, font_color="#000000"):
        style = (
            f"rounded={rounded};whiteSpace=wrap;html=1;fontFamily=Arial;fontSize={fs};fontColor={font_color};"
            f"fillColor={color[1] if fill else 'none'};strokeColor={color[0]};strokeWidth={stroke_w};"
            f"align={align};verticalAlign=middle;spacing=2;spacingLeft=3;spacingRight=3;"
        )
        if bold:
            style += "fontStyle=1;"
        if dashed:
            style += "dashed=1;"
        if dotted:
            style += "dashed=1;dashPattern=1 2;"
        self.cells.append(
            f'<mxCell id="{self._id()}" value={quoteattr(text)} style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )

    def text(self, text, x, y, w, h, fs=FS, bold=False, align="left", valign="top", color="#000000"):
        style = (
            f"text;html=1;whiteSpace=wrap;fontFamily=Arial;fontSize={fs};align={align};verticalAlign={valign};"
            f"fontColor={color};spacing=0;spacingLeft=0;spacingRight=0;spacingTop=0;"
        )
        if bold:
            style += "fontStyle=1;"
        self.cells.append(
            f'<mxCell id="{self._id()}" value={quoteattr(text)} style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )

    def bullets(self, items, x, y, w, pitch=12, color="#000000"):
        """One text cell per bullet so each line keeps its own baseline. Returns the bottom y."""
        for i, item in enumerate(items):
            self.text("\u2022 " + item, x, y + i * pitch, w, pitch, valign="middle", color=color)
        return y + len(items) * pitch

    def dot(self, x, y, d, color, stroke="#666666", stroke_w=0.75, label=""):
        style = (
            f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fontFamily=Arial;fontSize={FS};"
            f"fillColor={color};strokeColor={stroke};strokeWidth={stroke_w};spacing=0;"
        )
        self.cells.append(
            f'<mxCell id="{self._id()}" value={quoteattr(label)} style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x - d / 2}" y="{y - d / 2}" width="{d}" height="{d}" as="geometry"/></mxCell>'
        )

    def line(self, x1, y1, x2, y2, color="#666666", width=0.75, dashed=False, head="none", fill_head=True):
        style = f"endArrow={head};endFill={1 if fill_head else 0};html=1;strokeWidth={width};strokeColor={color};endSize=3;edgeStyle=none;"
        if dashed:
            style += "dashed=1;dashPattern=3 2;"
        self.cells.append(
            f'<mxCell id="{self._id()}" style="{style}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
            f'<mxPoint x="{x1}" y="{y1}" as="sourcePoint"/><mxPoint x="{x2}" y="{y2}" as="targetPoint"/>'
            f"</mxGeometry></mxCell>"
        )

    def letter(self, ch, x, y):
        self.text(ch, x - 2, y - 1, 16, 14, fs=FS_LETTER, bold=True)

    def write(self, path: str, w: int, h: int) -> None:
        body = "\n".join(self.cells)
        xml = (
            '<mxfile host="app.diagrams.net" agent="claude-code">\n'
            '  <diagram name="FigS-ontology-growth" id="FigS-ontology-growth">\n'
            f'    <mxGraphModel dx="1400" dy="1000" grid="0" page="1" pageWidth="{w}" pageHeight="{h}" math="0" shadow="0">\n'
            '      <root>\n        <mxCell id="0"/>\n        <mxCell id="1" parent="0"/>\n'
            f"{body}\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n"
        )
        with open(path, "w") as fh:
            fh.write(xml)


# Dataset colors: the color a dataset carries wherever it appears.
D1 = ORANGE[0]
D2 = PURPLE[0]
D3 = YELLOW[0]
D4 = BLUE[0]


def panel_a(c: Canvas, bottom: float) -> int:
    """One record and its closure, as three chains under the record class. Returns the bottom y."""
    x0, y0 = A_X, ROW_Y
    c.letter("a", x0, STRIP - 15)
    c.text("One fitness record and its closure", x0, y0, A_W, 12, bold=True)
    bh = 22
    c.box("FitnessExperiment", x0 + 2, y0 + 18, 96, bh, YELLOW, stroke_w=1.5)
    spine_x = x0 + 12  # the vertical line the three chains hang from
    # (name, width, color, dotted border). Inheritance (Phenotype over FitnessPhenotype,
    # GenePerturbation over DeletionPerturbation) is left out for simplicity; the chains
    # are the fields the record holds.
    chains = [
        [("Genotype", 56, ORANGE, False), ("DeletionPerturbation", 88, ORANGE, False)],
        [("Environment", 56, RED, False), ("Media", 32, RED, False), ("state = solid", 54, GRAY, True)],
        [("FitnessPhenotype", 78, PURPLE, False), ("fitness, se", 52, PURPLE, False)],
    ]
    items = ["D1: colony-size fitness on agar", "thick border: the record's closure",
             "dashed: a field holding a class"]
    bpitch = 12
    bullets_top = bottom - bpitch * len(items)
    rec_bottom = y0 + 18 + bh
    y = rec_bottom + 12
    # The three chains spread evenly from under the record down to the bullets.
    pitch = (bullets_top - 10 - bh - y) / 2
    pitches = [pitch, pitch]
    c.line(spine_x, rec_bottom, spine_x, y + sum(pitches) + bh / 2, dashed=True)
    for k, chain in enumerate(chains):
        x = x0 + 18
        c.line(spine_x, y + bh / 2, x, y + bh / 2, dashed=True)
        for i, (name, w, col, dotted) in enumerate(chain):
            c.box(name, x, y, w, bh, col, stroke_w=1.5 if not dotted else 1.25, dotted=dotted)
            if dotted:
                c.text("see b, row 1", x, y + bh + 1, w, 10, align="center", valign="middle", color="#666666")
            if i + 1 < len(chain):
                c.line(x + w, y + bh / 2, x + w + 6, y + bh / 2, dashed=True)
            x += w + 6
        c.dot(x0 + A_W + 4, y + bh / 2, 9, D1, stroke=D1)  # one column at the right margin
        if k < 2:
            y += pitches[k]
    c.bullets(items, x0 + 18, bullets_top, A_W - 18, pitch=bpitch)
    return bottom


def panel_b(c: Canvas) -> int:
    """A second dataset arrives: three outcomes. Returns the panel's bottom y."""
    x0, y0 = B_X, ROW_Y
    c.letter("b", x0, STRIP - 15)
    c.text("A liquid-growth dataset D2 arrives", x0, y0, B_W, 12, bold=True)
    col_w = [72, 96, 102]
    col_x = [x0, x0 + col_w[0] + 6, x0 + col_w[0] + col_w[1] + 12]
    yh = y0 + 16
    for x, w, head in zip(col_x, col_w, ("D2 update", "ontology update", "served datasets")):
        c.text(head, x, yh, w, 12, bold=True)
    rows = [
        ("Media.state = liquid", "already in the vocabulary: the ontology does not expand", GRAY, 0.75, False,
         ["incremental import", "D1 untouched"], (False, False)),
        ("an OD growth assay", "new leaf in AssayType: liquid_od_growth", ORANGE, 1.5, False,
         ["only D2 holds the leaf", "incremental import", "D1 untouched"], (False, False)),
        ("a new required field on Media", "a shared class changes", BLUE, 1.5, False,
         ["every closure with Media", "D1 and D2 rebuilt in full"], (True, True)),
    ]
    rh = 52
    y = yh + 16
    for need, onto, col, sw, dashed, served, rebuilt in rows:
        c.box(need, col_x[0], y, col_w[0], rh - 8, WHITE, stroke_w=0.75)
        c.line(col_x[0] + col_w[0], y + (rh - 8) / 2, col_x[1], y + (rh - 8) / 2, head="classic")
        c.box(onto, col_x[1], y, col_w[1], rh - 8, col, stroke_w=sw, dashed=dashed)
        c.line(col_x[1] + col_w[1], y + (rh - 8) / 2, col_x[2], y + (rh - 8) / 2, head="classic")
        # The served datasets as dots, thick red border when rebuilt.
        dx = col_x[2] + 7
        for i, (dcol, name) in enumerate(((D1, "D1"), (D2, "D2"))):
            stroke = RED[0] if rebuilt[i] else dcol
            c.dot(dx + 11 * i, y + 7, 9, dcol, stroke=stroke, stroke_w=2 if rebuilt[i] else 0.75)
        c.bullets(served, col_x[2], y + 13, col_w[2], pitch=11)
        y += rh
    # Legend: two columns of swatches, labels aligned beside them.
    ly0 = y + 4
    entries = [
        ("box", ORANGE, 1.5, "new leaf: the ontology expands"),
        ("box", BLUE, 1.5, "shared class changed"),
        ("ring", RED[0], 2, "dataset rebuilt in full"),
        ("dot", "#666666", 0.75, "dataset untouched"),
    ]
    lcol_w = 150
    for i, (kind, col, sw, label) in enumerate(entries):
        lx = x0 + (i // 2) * lcol_w
        ly = ly0 + (i % 2) * 12
        if kind == "box":
            c.box("", lx + 2, ly + 1.5, 14, 9, col, stroke_w=sw)
        elif kind == "ring":
            c.dot(lx + 9, ly + 6, 9, GRAY[1], stroke=col, stroke_w=sw)
        else:
            c.dot(lx + 9, ly + 6, 9, col, stroke=col, stroke_w=sw)
        c.text(label, lx + 20, ly, lcol_w - 20, 12, valign="middle")
    return ly0 + 24 + 2


def mini_tree(c: Canvas, x0: float, y0: float, leaves: list[list[tuple[str, bool]]], changed: int | None = None, height: float = 36) -> None:
    """A small tree: root, three branch nodes, and per branch its leaves as (color, rebuilt).
    Edges run between node boundaries. ``changed`` colors that branch node blue. ``height``
    is the root-to-leaf distance; the branch row sits halfway."""
    d = 7
    r = d / 2
    by_ = y0 + height / 2

    def edge(p: tuple[float, float], q: tuple[float, float]) -> None:
        dx, dy = q[0] - p[0], q[1] - p[1]
        n = (dx * dx + dy * dy) ** 0.5
        ux, uy = dx / n, dy / n
        c.line(p[0] + ux * r, p[1] + uy * r, q[0] - ux * r, q[1] - uy * r, width=0.6)

    root = (x0 + 24, y0)
    branches = [(x0 + 4, by_), (x0 + 24, by_), (x0 + 44, by_)]
    for bxy in branches:
        edge(root, bxy)
    slots = []
    for k, members in enumerate(leaves):
        for j, (col, hit) in enumerate(members):
            slots.append((branches[k][0] + (j - (len(members) - 1) / 2) * 10, y0 + height, k, col, hit))
    for lx, ly, k, _, _ in slots:
        edge(branches[k], (lx, ly))
    c.dot(root[0], root[1], d, GRAY[1])
    for k, (bx, by) in enumerate(branches):
        if changed == k:
            c.dot(bx, by, d, BLUE[0], stroke=BLUE[0])
        else:
            c.dot(bx, by, d, GRAY[1])
    for lx, ly, _, col, hit in slots:
        c.dot(lx, ly, d, col, stroke=RED[0] if hit else col, stroke_w=1.6 if hit else 0.75)


def panel_c(c: Canvas, bottom: float) -> int:
    """The ontology after each admitted dataset, tied to the rows of b. Returns the bottom y."""
    x0, y0 = C_X, ROW_Y
    c.letter("c", x0, STRIP - 15)
    c.text("The ontology grows with its datasets", x0, y0, C_W, 12, bold=True)
    tree_w = 50
    gap = (C_W - 3 * tree_w) / 2
    legend_top = bottom - 12 * 6
    note_pitch = 11
    notes_top = legend_top - 6 - 2 * note_pitch
    yt = y0 + 34  # root row, under the tree titles
    tree_h = 48  # a compact tree; the notes follow it, the legend keeps the row's bottom edge
    notes_top = yt + tree_h + 8
    d1 = (D1, False)
    stages = [
        ("D1", [[d1], [d1], [d1]], None, ("", "update: D1")),
        ("D2: b, row 2", [[d1], [d1, (D2, False)], [d1]], None, ("parent untouched", "update: D2")),
        ("D3: b, row 3", [[d1], [(D1, True), (D2, True), (D3, False)], [d1]], 1, ("parent changed", "update: D1, D2, D3")),
    ]
    for i, (title, leaves, changed, notes) in enumerate(stages):
        tx = x0 + i * (tree_w + gap)
        c.text(title, tx - 11, y0 + 16, tree_w + 22, 12, align="center")
        mini_tree(c, tx + 1, yt, leaves, changed=changed, height=tree_h)
        # Two note lines under the tree: the parent's state, and the datasets to update.
        for j, line in enumerate(notes):
            if line:
                c.text(line, tx - 11, notes_top + j * note_pitch, tree_w + 22, note_pitch, align="center", valign="middle")
    # Legend: every color in the panel, swatches on their label lines, as in b, bottom-aligned.
    ly = legend_top
    entries = [
        ("dot", D1, "D1 leaf"),
        ("dot", D2, "D2 leaf"),
        ("dot", D3, "D3 leaf"),
        ("dot", GRAY[1], "shared class"),
        ("dot", BLUE[0], "changed shared class"),
        ("ring", RED[0], "leaf rebuilt in full"),
    ]
    for kind, col, label in entries:
        if kind == "ring":
            c.dot(x0 + 6, ly + 6, 9, GRAY[1], stroke=col, stroke_w=2)
        else:
            c.dot(x0 + 6, ly + 6, 9, col, stroke=col if col != GRAY[1] else "#666666")
        c.text(label, x0 + 16, ly, C_W - 16, 12, valign="middle")
        ly += 12
    return bottom


def main() -> None:
    # Since 2026-10-20 the .drawio is maintained by hand in the draw.io GUI (the owner's
    # layout edits live only there). Running this script overwrites those edits, so it
    # refuses unless --force is given; it is kept as the record of how the figure was
    # first laid out.
    import sys

    if "--force" not in sys.argv:
        sys.exit(f"{OUT} is hand-maintained; rerun with --force to overwrite it from this script")
    c = Canvas()
    bottom = panel_b(c)  # the tallest panel sets the row's bottom edge; a and c align to it
    bottoms = [panel_a(c, bottom), bottom, panel_c(c, bottom)]
    h = int(max(bottoms)) + 2
    c.write(OUT, FULL_W, h)
    print(f"wrote {OUT}: {FULL_W} x {h} units ({FULL_W / 3.9392:.1f} x {h / 3.9392:.1f} mm)")


if __name__ == "__main__":
    main()
