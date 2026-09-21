"""Regenerate the five metabolic-module diagrams of notes-tex/028-gene-essentiality.

    python notes/assets/drawio/028-module.gen.py notes/assets/drawio

writes 028-module-projection.drawio (the alternating layer, rounds 2 to 13),
028-module-lp.drawio (the layer that solves, rounds 15 to 15d),
028-module-elastic.drawio (the elastic program with the pool by construction, round 18),
028-module-gated-flux.drawio (the gated elastic cell with a head over the solved
flux vector, round 21) and 028-module-gem-box.drawio (the solve on the GEM's own box,
bounds scaled by the availability gate and no capacity ceiling, rounds 15b, 24 and 27),
each 717 units wide (182 mm), fonts on the ladder only (8.3 body, 9.7 titles), every
formula as MathJax LaTeX (`$$...$$`, math="1" on the model), and the document's readout
color key (figures/readout_key): white the input to the layer (tokens, context), orange a
learned part (the availability gate, the abundance head, the latent MLP, the pool
projection), yellow the flux layer and its capacity terms (the GPR fold, the box, the
unit abundance, the alternations or the solve, the optimum), purple the fixed
stoichiometry and bounds of the GEM (the constraint-set block), red the prediction and
the loss (the readout), gray a measured note.
"""

from __future__ import annotations

import os.path as osp
import sys
from xml.sax.saxutils import escape

OUT_DIR = sys.argv[1]

W = 717
FS = "8.3"
FS_T = "9.7"

WHITE = ("#FFFFFF", "#666666")
ORANGE = ("#FFE6CC", "#D79B00")
RED = ("#F8CECC", "#B85450")
PURPLE = ("#E1D5E7", "#9673A6")
YELLOW = ("#FFF2CC", "#D6B656")
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
        self,
        value: str,
        x: float,
        y: float,
        w: float,
        h: float,
        fs: str = FS,
        align: str = "center",
        extra: str = "",
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
            geo += (
                '<Array as="points">'
                + "".join(f'<mxPoint x="{px:.2f}" y="{py:.2f}"/>' for px, py in points)
                + "</Array>"
            )
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
        print(
            f"{osp.basename(path)}: canvas {W} x {h} units = {W * 0.254:.0f} x {h * 0.254:.0f} mm"
        )


def m(tex: str) -> str:
    """One inline MathJax span (\\(...\\), inline mode, not the display-mode $$...$$ that would break the line); the export typesets it at the label font size."""
    return f"\\({tex}\\)"


# The document's notation, as LaTeX.
H_G = m(r"h_g")
GAMMA_G = m(r"\gamma_g")
E_G = m(r"E_g")
E_U = m(r"E_u")
C_J = m(r"c_j")
V = m(r"v")
VG = m(r"v_{\mathrm{growth}}")
Z = m(r"z")
ZS = m(r"z^*")
VS = m(r"v^*")


def chain(
    c: Canvas,
    y: float,
    h: float,
    specs: list[tuple[str, str, tuple[str, str], float]],
    x0: float = 0,
    gap: float = 13,
    widths: list[float] | None = None,
) -> list[str]:
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
    row_h = 84
    y1 = 0
    top = chain(
        c,
        y1,
        row_h,
        [
            ("gene tokens", f"{H_G} from the backbone on the perturbed graph, the deleted gene's token removed", WHITE, 0.8),
            ("availability gate", m(r"\gamma_g=\sigma(w^{\top}h_g)") + "; a deleted gene is set to 0 hard", ORANGE, 0.8),
            ("GPR fold", f"reaction availability {C_J}: softmin over a complex's subunits, sum over isozymes", YELLOW, 0.8),
            ("the box", m(r"[\ell_j c_j,\ u_j c_j]") + " tightened by the enzyme ceiling " + m(r"k_{\mathrm{cat},u}E_u") + " (nominal or predicted abundance)", YELLOW, 0.8),
            ("alternating projections", m(r"K") + " steps between the balance projection onto " + m(r"\{Sv=0\}") + " and the box clip onto " + m(r"[\ell,u]") + "; the last one holds exactly", YELLOW, 0.8),
            ("growth column", f"{VG}, one coordinate of the flux {V}", YELLOW, 0.8),
            ("readout", "forced: " + m(r"\ell=a\,v_{\mathrm{growth}}+b") + "<br>gated: " + m(r"\ell=\mathrm{head}(z)") + "<br>" + m(r"+\,g\,(a\,v_{\mathrm{growth}}+b),\ g_0=0"), RED, 0.8),
        ],
        widths=[76, 84, 84, 106, 110, 64, 115],
    )
    y2 = y1 + row_h + 30
    ctx = c.box("whole-cell context", f"{Z}, the pooled perturbed tokens", 190, y2, 110, 40, WHITE)
    mlp = c.box("latent MLP", f"{Z} " + m(r"\rightarrow") + " latent coordinates", 318, y2, 92, 40, ORANGE)
    v0 = c.box("proposal", m(r"v_0=Nz") + ", " + m(r"N") + " a basis of the null space of " + m(r"S") + ": " + m(r"Sv_0=0") + " exactly", 428, y2, 150, 40, YELLOW)
    c.arrow(ctx, mlp)
    c.arrow(mlp, v0)
    c.arrow(v0, top[4], exit_=(0.5, 0), entry=(0.5, 1), orthogonal=True)
    y3 = y2 + 40 + 16
    cs = (
        m(r"Sv=0") + ": exact when the alternation ends on the balance, else a residual<br>"
        + m(r"\ell\le v\le u") + ": exact when it ends on the box, else a penalty<br>"
        + "second law " + m(r"v_j\,\Delta_r G_j\le 0") + ": a hinge, or with tdg a directional box<br>"
        + "protein pool " + m(r"\sum_g \mathrm{MW}_g E_g\le p_{\mathrm{avail}}") + ": a hinge<br>"
        + "Gibbs dissipation: a logged diagnostic; the anchored registry entry carried it as a hinge at nonzero constraint scale"
    )
    c.box("the constraint set", cs, 0, y3, 440, 88, PURPLE, align="left")
    c.box(
        "measured (rounds 2 to 10)",
        "never reaches the polytope: 58 percent of reactions out of the box at the balance end, balance residual 0.85 to 2.0 at the box end; the growth column is a free signed coordinate",
        456,
        y3,
        261,
        88,
        GRAY,
        align="left",
    )
    return c


# ---------------------------------------------------------------- 2. lp / 3. elastic
def solve_rows(c: Canvas, elastic: bool, flux_head: bool = False) -> None:
    row_h = 88
    sw_new = 1.8 if elastic else 0.8
    gap = 11 if elastic else 13
    if flux_head:
        widths = [50, 96, 92, 62, 78, 112, 40, 110]
    elif elastic:
        widths = [56, 100, 104, 66, 84, 120, 40, 70]
    else:
        widths = [66, 112, 80, 100, 130, 44, 107]
    if flux_head:
        # Round 21 changes the readout only; the elastic blocks keep the plain border.
        sw_new = 0.8
    specs: list[tuple[str, str, tuple[str, str], float]] = [
        ("gene tokens", f"{H_G} from the backbone, perturbed graph", WHITE, 0.8),
        ("abundance head", m(r"E_g=\gamma_g E^0_g\,\cdot") + "<br>" + m(r"\exp(r\tanh(w^{\top}h_g))") + "<br>" + GAMMA_G + " hard (1 present, 0 deleted), no learned gate", ORANGE, 0.8),
    ]
    if elastic:
        specs.append(("pool, exact", m(r"E\leftarrow \rho E") + "<br>" + m(r"\rho=\min(1,\ p_{\mathrm{avail}}/P)") + "<br>" + m(r"P=\textstyle\sum_g \mathrm{MW}_g E_g") + "<br>the GECKO pool holds", ORANGE, sw_new))
    specs += [
        ("unit abundance", f"{E_U}: power mean over a complex's subunits, sum over isozymes", YELLOW, 0.8),
        ("box", m(r"|v_j|\le k_{\mathrm{cat},u}E_u") + " times the scale, inside " + m(r"[\ell_j,u_j]") + " scaled by availability", YELLOW, 0.8),
    ]
    if elastic:
        specs.append((
            "elastic growth LP",
            m(r"\max\ v_{\mathrm{growth}}-M\textstyle\sum_i s_i") + "<br>s.t. " + m(r"Sv=0") + ", " + m(r"v_i\ge \ell_i-s_i") + ",<br>" + m(r"0\le s_i\le \ell_i") + " on forced bounds, the box otherwise; " + m(r"M=1") + "; HiGHS in micromole units",
            YELLOW,
            sw_new,
        ))
    else:
        specs.append((
            "growth LP",
            m(r"\max\ v_{\mathrm{growth}}") + " s.t. " + m(r"Sv=0,\ \ell\le v\le u") + "<br>HiGHS per genotype in a forked pool",
            YELLOW,
            0.8,
        ))
    specs.append(("optimum", f"{VS}, {ZS}", YELLOW, 0.8))
    if flux_head:
        specs.append((
            "readout",
            m(r"\ell=\mathrm{head}(z)+g\,(a\,z^*+b)") + "<br>" + m(r"+\,w^{\top}\log(1+|v^*_{\mathcal{R}}|),\ w_0=0") + "<br>" + m(r"\mathcal{R}") + ": all 4,131 reactions, or the 274 exchanges",
            RED,
            1.8,
        ))
    else:
        specs.append(("readout", "forced: " + m(r"\ell=a\,z^*+b") + "<br>gated: " + m(r"\ell=\mathrm{head}(z)") + "<br>" + m(r"+\,g\,(a\,z^*+b)"), RED, 0.8))
    top = chain(c, 0, row_h, specs, widths=widths, gap=gap)
    head_ix = 1
    lp_ix = 5 if elastic else 4
    yb = row_h + 26
    if elastic:
        back_label = (
            "backward: " + m(r"\partial z^*/\partial u_j") + " is the dual of the ceiling (envelope theorem), nonzero only where it binds, "
            + "plus the dual of the elastic row on the ceilings that block maintenance; through " + E_G + " to the backbone"
        )
    else:
        back_label = (
            "backward: " + m(r"\partial z^*/\partial u_j") + " is the dual of the ceiling (envelope theorem), "
            + "nonzero only where the bound binds; through " + E_G + " to the backbone"
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
    y3 = yb + 40
    if flux_head:
        exact = (
            VS + " enters the head detached: the argmax is piecewise constant in the bounds, so the head reads it as a fixed feature of the current solve and no gradient flows through it<br>"
            + "the growth column keeps its gated affine and " + m(r"w_0=0") + ", so the cell is the round-18 gated cell at initialization<br>"
            + "the head's weight norm is logged each epoch (ess_flux_head_w_norm); " + m(r"Sv=0") + ", the ceilings and the pool hold exactly at " + VS + " as in round 18"
        )
        note = (
            "round 11 measured the deletion's pFBA flux vector at 0.77 as an encoding against its growth ratio at 0.65, and the dual-overlap probe measured that the ceilings bind on the same six reactions for every genotype; "
            "this reads what the solve returns instead of what it can move"
        )
        note_title = "measured (rounds 11 and 18)"
    elif elastic:
        exact = (
            m(r"Sv=0") + " and every ceiling hold exactly at " + VS + "; the pool holds exactly on " + E_G + "<br>"
            + "feasible for every genotype by construction: the zero flux with full slack satisfies every row and bound<br>"
            + "a genotype that cannot meet its maintenance pays " + m(r"M") + " per unit unmet, so " + ZS + " goes negative<br>"
            + "arm flux_projection_anchored_nodiss: no dissipation hinge; Gibbs dissipation is logged only"
        )
        note = (
            "calibration through the layer at initialization: at ceiling scale 1 the maintenance shortfall is 0.54 of 0.70 mmol/gDW/h with zero growth; "
            "from scale 10 no slack; a gene-owned bound carries a dual on 97 percent of genotypes at scales 30 to 3,000"
        )
        note_title = "measured (round 18 calibration)"
    else:
        exact = (
            m(r"Sv=0") + " and " + m(r"\ell\le v\le u") + " hold exactly at " + VS + ", the ceilings with them<br>"
            + VS + " is piecewise constant in the bounds and carries no gradient; only " + ZS + " does<br>"
            + "the forced flux of the maintenance reaction (" + m(r"\ell=0.7") + " mmol/gDW/h) must be carried inside the ceilings"
        )
        note = (
            "an infeasible program returns the zero flux and no gradient, so the state is absorbing (rounds 15b to 15d); "
            "duals on exchange bounds belong to no gene, so with no ceiling binding the backbone receives nothing (round 15b)"
        )
        note_title = "measured (rounds 15 to 15d)"
    c.box("by construction", exact, 0, y3, 440, 78, PURPLE, align="left")
    c.box(note_title, note, 456, y3, 261, 78, GRAY, align="left")


def lp() -> Canvas:
    c = Canvas()
    solve_rows(c, elastic=False)
    return c


def elastic() -> Canvas:
    c = Canvas()
    solve_rows(c, elastic=True)
    return c


def gated_flux() -> Canvas:
    c = Canvas()
    solve_rows(c, elastic=True, flux_head=True)
    return c


# ---------------------------------------------------------------- 5. GEM box
def gem_box() -> Canvas:
    """The layer that solves on the GEM's own box: rounds 15b, 24 and 27."""
    c = Canvas()
    row_h = 88
    gap = 13
    widths = [62, 96, 84, 96, 130, 44, 127]
    specs: list[tuple[str, str, tuple[str, str], float]] = [
        ("gene tokens", f"{H_G} from the backbone, perturbed graph", WHITE, 0.8),
        ("availability gate", m(r"\gamma_g=\sigma(w^{\top}h_g)") + "; a deleted gene is set to 0 hard; no abundance head", ORANGE, 1.8),
        ("GPR fold", f"reaction availability {C_J}: softmin over a complex's subunits, sum over isozymes", YELLOW, 0.8),
        ("box", m(r"[\ell_j c_j,\ u_j c_j]") + ", the GEM's own bounds scaled by availability; no " + m(r"k_{\mathrm{cat}}") + " ceiling, no second law", YELLOW, 1.8),
        ("growth LP", m(r"\max\ v_{\mathrm{growth}}") + " s.t. " + m(r"Sv=0,\ \ell c\le v\le uc") + "<br>HiGHS per genotype; from round 27 a solved program is stored and reused when its bounds carry no gradient", YELLOW, 0.8),
        ("optimum", f"{VS}, {ZS}", YELLOW, 0.8),
        ("readout", "round 15b, gated: " + m(r"\ell=\mathrm{head}(z)+g\,(a\,z^*+b)") + "<br>rounds 24 and 27: " + m(r"+\,w^{\top}\log(1+|v^*|),\ w_0=0"), RED, 0.8),
    ]
    top = chain(c, 0, row_h, specs, widths=widths, gap=gap)
    gate_ix, lp_ix = 1, 4
    yb = row_h + 26
    x_lp = sum(widths[:lp_ix]) + gap * lp_ix + widths[lp_ix] / 2
    x_gate = sum(widths[:gate_ix]) + gap * gate_ix + widths[gate_ix] / 2
    c.arrow(
        top[lp_ix],
        top[gate_ix],
        dashed=True,
        exit_=(0.5, 1),
        entry=(0.5, 1),
        points=[(x_lp, yb), (x_gate, yb)],
    )
    c.text(
        "backward: " + m(r"\partial z^*/\partial u_j") + " is the dual of the availability-scaled bound (envelope theorem), nonzero only where a gene-owned bound binds; through " + GAMMA_G + " to the backbone",
        x_gate + 10,
        yb + 4,
        x_lp - x_gate - 20,
        24,
    )
    y3 = yb + 40
    exact = (
        m(r"Sv=0") + " and " + m(r"\ell c\le v\le uc") + " hold exactly at " + VS + "; the box is the GEM's, so it contains growth for every viable deletion<br>"
        + VS + " enters the flux head detached, as a fixed feature of the current solve; only " + ZS + " carries a gradient<br>"
        + "arm flux_off under the lp parameterization: no abundance head, no capacity ceiling, no thermodynamic term"
    )
    note = (
        "round 15b: feasible from the first forward pass, the growth column ranks the held-out genes at 0.734 before training, and the forced cell receives no gradient when only exchange bounds bind; "
        "round 24: the flux head over this solve reads 0.852 on ProtT5; round 27: 0.882 on the pathway input, pooled 0.917"
    )
    c.box("by construction", exact, 0, y3, 440, 78, PURPLE, align="left")
    c.box("measured (rounds 15b, 24 and 27)", note, 456, y3, 261, 78, GRAY, align="left")
    return c


for name, fn in (
    ("projection", projection),
    ("lp", lp),
    ("elastic", elastic),
    ("gated-flux", gated_flux),
    ("gem-box", gem_box),
):
    canvas = fn()
    canvas.write(osp.join(OUT_DIR, f"028-module-{name}.drawio"), f"module-{name}")
