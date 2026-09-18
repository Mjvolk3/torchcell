"""Regenerate the three metabolic-module diagrams of notes-tex/028-gene-essentiality.

    python notes/assets/drawio/028-module.gen.py notes/assets/drawio

writes 028-module-projection.drawio (the alternating layer, rounds 2 to 13),
028-module-lp.drawio (the layer that solves, rounds 15 to 15d) and
028-module-elastic.drawio (the elastic program with the pool by construction, round 18),
each 717 units wide (182 mm), fonts on the ladder only (8.3 body, 9.7 titles, 11.1 panel
letters), one color per role across all three: orange the backbone's tokens and context,
red a learned head of the layer, purple the constraint set (availability, ceilings, box,
pool, balance), yellow the flux block, blue the readout, gray a measured note.
"""

from __future__ import annotations

import os.path as osp
import sys
from xml.sax.saxutils import escape

OUT_DIR = sys.argv[1]

W = 717
FS = "8.3"
FS_T = "9.7"

ORANGE = ("#FFE6CC", "#D79B00")
RED = ("#F8CECC", "#B85450")
PURPLE = ("#E1D5E7", "#9673A6")
YELLOW = ("#FFF2CC", "#D6B656")
BLUE = ("#DAE8FC", "#6C8EBF")
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
        stroke_width: float = 0.8,
        align: str = "center",
    ) -> str:
        fill, stroke = color
        i = self.nid()
        label = ""
        if title:
            label += f'<span style="font-size: {FS_T}px"><b>{title}</b></span>'
        if body:
            label += ("<br>" if title else "") + body
        style = (
            f"rounded=1;arcSize=8;whiteSpace=wrap;html=1;fontFamily=Arial;fontSize={FS};"
            f"fillColor={fill};strokeColor={stroke};strokeWidth={stroke_width};"
            f"align={align};verticalAlign=middle;spacing=3;spacingLeft=4;spacingRight=4;"
        )
        self.cells.append(
            f'<mxCell id="{i}" value="{escape(label, {chr(34): "&quot;"})}" style="{style}" '
            f'vertex="1" parent="1"><mxGeometry x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" '
            f'height="{h:.2f}" as="geometry"/></mxCell>'
        )
        self.height = max(self.height, y + h)
        return i

    def text(
        self, value: str, x: float, y: float, w: float, h: float, fs: str = FS, align: str = "center", extra: str = ""
    ) -> str:
        i = self.nid()
        style = (
            f"text;html=1;align={align};verticalAlign=middle;fontFamily=Arial;fontSize={fs};"
            f"strokeColor=none;fillColor=none;whiteSpace=wrap;{extra}"
        )
        self.cells.append(
            f'<mxCell id="{i}" value="{escape(value, {chr(34): "&quot;"})}" style="{style}" '
            f'vertex="1" parent="1"><mxGeometry x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" '
            f'height="{h:.2f}" as="geometry"/></mxCell>'
        )
        self.height = max(self.height, y + h)
        return i

    def arrow(
        self,
        src: str,
        dst: str,
        label: str = "",
        dashed: bool = False,
        exit_: tuple[float, float] | None = None,
        entry: tuple[float, float] | None = None,
        points: list[tuple[float, float]] | None = None,
        color: str = "#000000",
        orthogonal: bool = False,
    ) -> None:
        i = self.nid()
        style = (
            f"endArrow=blockThin;endFill=1;endSize=5;html=1;strokeColor={color};strokeWidth=0.8;"
            f"fontFamily=Arial;fontSize={FS};labelBackgroundColor=#FFFFFF;"
        )
        if dashed:
            style += "dashed=1;dashPattern=4 3;"
        if orthogonal:
            style += "edgeStyle=orthogonalEdgeStyle;"
        if exit_ is not None:
            style += f"exitX={exit_[0]};exitY={exit_[1]};exitDx=0;exitDy=0;"
        if entry is not None:
            style += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
        geo = '<mxGeometry relative="1" as="geometry">'
        if points:
            geo += "<Array as=\"points\">" + "".join(
                f'<mxPoint x="{px:.2f}" y="{py:.2f}"/>' for px, py in points
            ) + "</Array>"
        geo += "</mxGeometry>"
        self.cells.append(
            f'<mxCell id="{i}" value="{escape(label, {chr(34): "&quot;"})}" style="{style}" '
            f'edge="1" parent="1" source="{src}" target="{dst}">{geo}</mxCell>'
        )

    def write(self, path: str, name: str) -> None:
        h = int(self.height + 1)
        xml = (
            f'<mxfile><diagram id="{name}" name="Page-1"><mxGraphModel dx="0" dy="0" grid="1" '
            'gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" '
            f'pageScale="1" pageWidth="{W}" pageHeight="{h}" math="1" shadow="0">'
            '<root><mxCell id="0"/><mxCell id="1" parent="0"/>'
            + "".join(self.cells)
            + "</root></mxGraphModel></diagram></mxfile>"
        )
        with open(path, "w") as fh:
            fh.write(xml)
        print(f"{osp.basename(path)}: canvas {W} x {h} units = {W * 0.254:.0f} x {h * 0.254:.0f} mm")


# Symbols as HTML so every label is one font at one size.
GAMMA = "<i>&gamma;</i>"
SIG = "<i>&sigma;</i>"
H_G = "<i>h</i><sub><i>g</i></sub>"
E_G = "<i>E</i><sub><i>g</i></sub>"
E_U = "<i>E</i><sub><i>u</i></sub>"
E0 = "<i>E</i><sup>0</sup><sub><i>g</i></sub>"
KCAT = "<i>k</i><sub>cat,<i>u</i></sub>"
C_J = "<i>c</i><sub><i>j</i></sub>"
LB = "<i>lb</i><sub><i>j</i></sub>"
UB = "<i>ub</i><sub><i>j</i></sub>"
V = "<i>v</i>"
VG = "<i>v</i><sub>growth</sub>"
Z = "<i>z</i>"
ZS = "<i>z</i><sup>*</sup>"
VS = "<i>v</i><sup>*</sup>"
LEQ = "&le;"
SUM = "&Sigma;"
DG = "&Delta;<i>G</i><sub><i>j</i></sub>"


def chain(c: Canvas, y: float, h: float, specs: list[tuple[str, str, tuple[str, str], float]], x0: float = 0, gap: float = 13, widths: list[float] | None = None) -> list[str]:
    """A left-to-right row of boxes joined by arrows; returns their ids."""
    ids: list[str] = []
    x = x0
    for k, (title, body, color, sw) in enumerate(specs):
        w = widths[k] if widths else (W - x0 - gap * (len(specs) - 1)) / len(specs)
        ids.append(c.box(title, body, x, y, w, h, color, stroke_width=sw))
        x += w + gap
    for a, b in zip(ids, ids[1:]):
        c.arrow(a, b)
    return ids


# ---------------------------------------------------------------- 1. projection
def projection() -> Canvas:
    c = Canvas()
    row_h = 78
    y1 = 0
    top = chain(
        c,
        y1,
        row_h,
        [
            ("gene tokens", f"{H_G} from the backbone on the perturbed graph, the deleted gene's token removed", ORANGE, 0.8),
            ("availability gate", f"{GAMMA}<sub><i>g</i></sub> = {SIG}(<i>w</i>&middot;{H_G}); a deleted gene is set to 0 hard", RED, 0.8),
            ("GPR fold", f"reaction availability {C_J}: softmin over a complex's subunits, sum over isozymes", PURPLE, 0.8),
            ("the box", f"[{LB}{C_J}, {UB}{C_J}] tightened by the enzyme ceiling {KCAT}{E_U} (nominal or predicted abundance)", PURPLE, 0.8),
            ("alternating projections", f"<i>K</i> steps between the balance projection onto {{<i>S</i>{V} = 0}} and the box clip onto [<i>lb</i>, <i>ub</i>]; the last one holds exactly", YELLOW, 0.8),
            ("growth column", f"{VG}, one coordinate of the flux {V}", YELLOW, 0.8),
            ("readout", f"forced: <i>a</i> {VG} + <i>b</i>; gated: head({Z}) + <i>g</i> (<i>a</i> {VG} + <i>b</i>), <i>g</i><sub>0</sub> = 0", BLUE, 0.8),
        ],
        widths=[76, 84, 84, 106, 120, 64, 105],
    )
    # the proposal path feeding the alternation from below
    y2 = y1 + row_h + 30
    ctx = c.box("whole-cell context", f"{Z}, the pooled perturbed tokens", 190, y2, 110, 40, ORANGE)
    mlp = c.box("latent MLP", f"{Z} &rarr; latent coordinates", 318, y2, 92, 40, RED)
    v0 = c.box("proposal", f"{V}<sub>0</sub> = <i>N</i> {Z}, <i>N</i> a basis of the null space of <i>S</i>: <i>S</i>{V}<sub>0</sub> = 0 exactly", 428, y2, 150, 40, YELLOW)
    c.arrow(ctx, mlp)
    c.arrow(mlp, v0)
    c.arrow(v0, top[4], exit_=(0.5, 0), entry=(0.5, 1), orthogonal=True)
    # the constraint set and the measured note
    y3 = y2 + 40 + 16
    cs = (
        f"<i>S</i>{V} = 0: exact when the alternation ends on the balance, else a residual<br>"
        f"<i>lb</i> {LEQ} {V} {LEQ} <i>ub</i>: exact when it ends on the box, else a penalty<br>"
        f"second law {V}<sub><i>j</i></sub> {DG} {LEQ} 0: a hinge, or with tdg a directional box<br>"
        f"protein pool {SUM}<sub><i>g</i></sub> MW<sub><i>g</i></sub> {E_G} {LEQ} <i>p</i><sub>avail</sub>: a hinge<br>"
        f"Gibbs dissipation: a logged diagnostic; the anchored registry entry carried it as a hinge at nonzero constraint scale"
    )
    c.box("the constraint set", cs, 0, y3, 440, 84, PURPLE, align="left")
    c.box(
        "measured (rounds 2 to 10)",
        "never reaches the polytope: 58 percent of reactions out of the box at the balance end, balance residual 0.85 to 2.0 at the box end; the growth column is a free signed coordinate",
        456,
        y3,
        261,
        84,
        GRAY,
        align="left",
    )
    return c


# ---------------------------------------------------------------- 2. lp
def solve_rows(c: Canvas, elastic: bool) -> None:
    row_h = 80
    sw_new = 1.8 if elastic else 0.8
    gap = 11 if elastic else 13
    if elastic:
        widths = [64, 94, 74, 82, 92, 122, 48, 64]
    else:
        widths = [78, 106, 86, 104, 100, 112, 53]
    specs: list[tuple[str, str, tuple[str, str], float]] = [
        ("gene tokens", f"{H_G} from the backbone, perturbed graph", ORANGE, 0.8),
        ("abundance head", f"{E_G} = {GAMMA}<sub><i>g</i></sub> {E0} exp(<i>r</i> tanh(<i>w</i>&middot;{H_G})); {GAMMA} hard (1 present, 0 deleted), no learned gate", RED, 0.8),
    ]
    if elastic:
        specs.append(("pool, exact", f"{E_G} &larr; {E_G} min(1, <i>p</i><sub>avail</sub> / {SUM}<sub><i>g</i></sub> MW<sub><i>g</i></sub> {E_G}): the GECKO pool holds", PURPLE, sw_new))
    specs += [
        ("unit abundance", f"{E_U}: power mean over a complex's subunits, sum over isozymes", PURPLE, 0.8),
        ("box", f"|{V}<sub><i>j</i></sub>| {LEQ} {KCAT}{E_U} &times; scale, inside [{LB}, {UB}] scaled by availability", PURPLE, 0.8),
    ]
    if elastic:
        specs.append((
            "elastic growth LP",
            f"max {VG} &minus; <i>M</i> {SUM}<i>s</i> s.t. <i>S</i>{V} = 0, {V}<sub><i>i</i></sub> &ge; <i>lb</i><sub><i>i</i></sub> &minus; <i>s</i><sub><i>i</i></sub> on forced bounds (0 {LEQ} <i>s</i> {LEQ} <i>lb</i>), the box otherwise; <i>M</i> = 1; HiGHS in micromole units",
            YELLOW,
            sw_new,
        ))
    else:
        specs.append((
            "growth LP",
            f"max {{{VG} : <i>S</i>{V} = 0, <i>lb</i> {LEQ} {V} {LEQ} <i>ub</i>}}, HiGHS per genotype in a forked pool",
            YELLOW,
            0.8,
        ))
    specs += [
        ("optimum", f"{VS}, {ZS}", YELLOW, 0.8),
        ("readout", f"<i>a</i> {ZS} + <i>b</i>, or gated: head({Z}) + <i>g</i> (<i>a</i> {ZS} + <i>b</i>)", BLUE, 0.8),
    ]
    top = chain(c, 0, row_h, specs, widths=widths, gap=gap)
    head_ix = 1
    lp_ix = 5 if elastic else 4
    # the backward path, dashed, below the row
    yb = row_h + 26
    if elastic:
        back_label = (
            f"backward: &part;{ZS}/&part;<i>ub</i><sub><i>j</i></sub> is the dual of the ceiling (envelope theorem), nonzero only where it binds, "
            f"plus the dual of the elastic row on the ceilings that block maintenance; through {E_G} to the backbone"
        )
    else:
        back_label = (
            f"backward: &part;{ZS}/&part;<i>ub</i><sub><i>j</i></sub> is the dual of the ceiling (envelope theorem), "
            f"nonzero only where the bound binds; through {E_G} to the backbone"
        )
    x_lp = sum(widths[:lp_ix]) + gap * lp_ix + widths[lp_ix] / 2
    x_head = sum(widths[:head_ix]) + gap * head_ix + widths[head_ix] / 2
    c.arrow(
        top[lp_ix],
        top[head_ix],
        dashed=True,
        exit_=(0.5, 1),
        entry=(0.5, 1),
        points=[(x_lp, yb), (x_head, yb)],
    )
    c.text(back_label, x_head + 10, yb + 4, x_lp - x_head - 20, 24)
    # what holds exactly, and the measured note
    y3 = yb + 40
    if elastic:
        exact = (
            f"<i>S</i>{V} = 0 and every ceiling hold exactly at {VS}; the pool holds exactly on {E_G}<br>"
            f"feasible for every genotype by construction: the zero flux with full slack satisfies every row and bound<br>"
            f"a genotype that cannot meet its maintenance pays <i>M</i> per unit unmet, so {ZS} goes negative<br>"
            f"arm flux_projection_anchored_nodiss: no dissipation hinge; Gibbs dissipation is logged only"
        )
        note = (
            "calibration through the layer at initialization: at ceiling scale 1 the maintenance shortfall is 0.54 of 0.70 mmol/gDW/h with zero growth; "
            "from scale 10 no slack; a gene-owned bound carries a dual on 97 percent of genotypes at scales 30 to 3,000"
        )
        note_title = "measured (round 18 calibration)"
    else:
        exact = (
            f"<i>S</i>{V} = 0 and <i>lb</i> {LEQ} {V} {LEQ} <i>ub</i> hold exactly at {VS}, the ceilings with them<br>"
            f"{VS} is piecewise constant in the bounds and carries no gradient; only {ZS} does<br>"
            f"the forced flux of the maintenance reaction (lb 0.7 mmol/gDW/h) must be carried inside the ceilings"
        )
        note = (
            "an infeasible program returns the zero flux and no gradient, so the state is absorbing (rounds 15b to 15d); "
            "duals on exchange bounds belong to no gene, so with no ceiling binding the backbone receives nothing (round 15b)"
        )
        note_title = "measured (rounds 15 to 15d)"
    c.box("by construction", exact, 0, y3, 440, 74, PURPLE, align="left")
    c.box(note_title, note, 456, y3, 261, 74, GRAY, align="left")


def lp() -> Canvas:
    c = Canvas()
    solve_rows(c, elastic=False)
    return c


def elastic() -> Canvas:
    c = Canvas()
    solve_rows(c, elastic=True)
    return c


for name, fn in (("projection", projection), ("lp", lp), ("elastic", elastic)):
    canvas = fn()
    canvas.write(osp.join(OUT_DIR, f"028-module-{name}.drawio"), f"module-{name}")
