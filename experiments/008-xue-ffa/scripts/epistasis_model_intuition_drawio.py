# experiments/008-xue-ffa/scripts/epistasis_model_intuition_drawio.py
# [[experiments.008-xue-ffa.scripts.epistasis_model_intuition_drawio]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/epistasis_model_intuition_drawio
#
# The epistasis-model figure as a native draw.io page: two schematic panels that say what
# the figure is for, four embedded data panels (written by
# epistasis_model_intuition_panels.py), a schematic of where the two families of null come
# from, and a table of what each of the four models assumes and estimates. Everything
# except the panels and the equations is an mxCell, so the wording and the arrangement can
# be worked on in draw.io.
#
# EVERY EXPRESSION IS AN IMAGE, typeset as math by the panels script and placed here at its
# recorded size. draw.io's HTML <sub> is not typesetting: the variables come out upright
# where they should be italic, the subscripts sit at the wrong size, and the PDF export set
# the subscript runs in a SERIF fallback, which put a fourth typeface into the figure.
#
# WHY TWO SCHEMATIC PANELS ON TOP (author review, 2026.09.18). The data panels answer how
# the four nulls differ; they do not say why a reader of a metabolic engineering paper
# should care which null is chosen. Panel a is the motivation: a yeast in nature carries
# robustness and cross-coupling that a factory does not need and cannot switch off, and
# most of that coupling is not on the curated map at all (Wu et al. 2026 predict a
# reaction space fourteen times the model's, with weaker predicted affinities). Panel b is
# the classic picture: an interaction is the departure of a measured combination from an
# expectation, and whether such departures can be predicted decides whether a design
# climbs or tinkers. The genome-wide counts on growth (Costanzo 2016, Kuzmin 2018) say
# what kind of landscape that is: mostly negative, and a hundred times denser at three
# genes than at two.
#
# WHY A SCHEMATIC OF THE NULLS AT ALL. The four models are four lines of algebra in the
# Methods, and a reader who takes the algebra at face value will read "multiplicative" as
# the default and the other three as robustness checks. The schematic says what each null
# is a statement ABOUT: a multiplicative null says two deletions each keep a FRACTION of
# what reaches them, which is how independent steps of one flux compose and is why the
# growth screens use it; an additive null says each deletion removes an ABSOLUTE amount
# from a shared pool, which is closer to how a titer is built. The two differ by exactly
# (1 - f_i)(1 - f_j): the part of the first deletion's loss that the second deletion would
# have taken again. That is the whole of the disagreement, and it is the reason the choice
# of null is a modeling claim about the system rather than a taste.
#
# UNITS and TYPE come from drawio_doc: 3.9392 canvas units per mm, Arial, fontSize 8.3 for
# figure text (5.98 pt) and 11.1 for panel letters (7.99 pt).

import argparse
import base64
import json
import os
import os.path as osp
import re
import sys

from dotenv import load_dotenv

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from drawio_doc import (  # noqa: E402
    FONT,
    PT,
    U,
    Doc,
    export,
    letter_style,
    line_style,
    text_style,
)

load_dotenv()
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "008-xue-ffa/results")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
DRAWIO_DIR = osp.join(osp.dirname(ASSET_IMAGES_DIR), "drawio")

# Model colors, the same four the data panels use (multiplicative blue, additive brick,
# GLM lilac, log-OLS orange; see epistasis_model_intuition_panels.py for why).
C_MULT = "#6C8EBF"
C_ADD = "#B85450"
C_GLM = "#9673A6"
C_OLS = "#D79B00"
C_YELLOW = "#D6B656"
C_RULE = "#666666"
FILL = {C_MULT: "#DAE8FC", C_ADD: "#F8CECC", C_GLM: "#E1D5E7", C_OLS: "#FFE6CC",
        C_YELLOW: "#FFF2CC", C_RULE: "#F5F5F5"}

W_MM = 179.0
PANEL_STEM = ["mi_panel_expectations", "mi_panel_null_fit", "mi_panel_mean_variance"]

# Row geometry in millimetres from the top of the page. Five rows, each the full width:
# the two schematic panels, the three measured panels, the level sets of all four models
# side by side, the schematic of the two nulls, and the table. The heights of the
# measured panels and the level sets were cut (52 to 38 mm, 36 to 31 mm) to pay for the
# new top row inside the 170 mm cap; the table's rows were cut from 8.0 to 6.4 mm.
Y_ROW0 = 5.0          # top of the two schematic panels
H_ROW0 = 31.0
Y_ROW1 = 40.5         # top of the three measured panels (38 mm tall)
Y_ROW2 = 83.0         # top of the level-set row (31 mm tall)
Y_ROW3 = 118.5        # top of the schematic of the nulls
Y_ROW4 = 138.5        # top of the table
LETTER_DY = 4.6       # a letter sits this far above its block

# The "expects" column carries a short expression and the "residual" column a two-word
# phrase that wrapped at 21 mm (author review, 2026.09.18): 27 and 21 became 18 and 30.
TABLE_COLS = [
    ("model", 20.0),
    ("expects, for a double", 18.0),
    ("residual measured on", 30.0),
    ("fit to, and what that assumes", 42.0),
    ("the reading that makes it the natural null", 65.0),
]
# (color, name, equation key, residual scale, what it is fit to, the reading)
TABLE_ROWS = [
    (C_MULT, "multiplicative", "mult",
     "linear titer",
     "strain means; one standard error per strain, propagated by the delta method",
     "each deletion keeps a FRACTION of what reaches it, as independent steps of one flux "
     "do; the null of the growth screens"),
    (C_ADD, "additive", "add",
     "linear titer",
     "strain means; one standard error per strain, propagated by the delta method",
     "each deletion removes a fixed AMOUNT from a shared pool, closer to how a titer in "
     "mg/L is built up"),
    (C_GLM, "GLM log-link", "glm",
     "log titer",
     "replicate titers; Gamma family, spread proportional to the mean (panel e)",
     "the readout is positive and noisier where it is larger, and every strain's replicates "
     "are used rather than its mean"),
    (C_OLS, "log-OLS", "glm",
     "log titer",
     "replicate log ratios to the base strain; constant spread on the log scale",
     "the question is whether a deletion's FOLD effect carries into a new background, which "
     "is a log-scale question"),
]

# The genome-wide counts on growth that panel b quotes. Each is a verbatim figure from the
# mirrored paper (torchcell-library/<key>/paper.md), and the caption cites both.
GROWTH_COUNTS = {
    # costanzoGlobalGeneticInteraction2016, abstract: "identifying about 550,000 negative
    # and about 350,000 positive genetic interactions"
    "digenic_negative": "550,000",
    "digenic_positive": "350,000",
    # kuzminSystematicAnalysisComplex2018, Results: "We predict that ~10^8 trigenic
    # combinations exhibit a negative genetic interaction ... on the order of 100 times as
    # many trigenic interactions as observed for the global digenic network"; "trigenic
    # interactions tend to be ~25% weaker than digenic interactions"
    "trigenic_fold": "100",
    "trigenic_weaker_pct": "25",
}


def svg_size_units(path):
    """A true-size panel SVG's width and height in canvas units."""
    head = open(path, encoding="utf-8").read(2000)
    m = re.search(r'width="([\d.]+)(px)?" height="([\d.]+)(px)?"', head)
    if m is None:
        raise ValueError(f"{path}: no width/height on the <svg> element")
    return float(m.group(1)), float(m.group(3))


def data_uri(path):
    payload = base64.b64encode(open(path, "rb").read()).decode("ascii")
    return f"data:image/svg+xml,{payload}"


def box_style(color, dashed=False, fill=None):
    return (f"rounded=1;arcSize=18;whiteSpace=wrap;html=1;"
            f"fillColor={fill or FILL.get(color, '#FFFFFF')};"
            f"strokeColor={color};strokeWidth={0.5 * PT:.2f};fontFamily=Arial;"
            f"fontSize={FONT};fontColor=#000000;verticalAlign=middle;"
            + ("dashed=1;dashPattern=3 2;" if dashed else ""))


def cell_style(fill="#FFFFFF", align="left", bold=False):
    return (f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={C_RULE};"
            f"strokeWidth={0.25 * PT:.2f};fontFamily=Arial;fontSize={FONT};"
            f"fontColor=#000000;align={align};verticalAlign=middle;spacingLeft=2;"
            f"spacingRight=2;" + ("fontStyle=1;" if bold else ""))


def shape_style(shape, color, fill=None, dashed=False, width_pt=0.5):
    return (f"{shape};whiteSpace=wrap;html=1;fillColor={fill or FILL.get(color, '#FFFFFF')};"
            f"strokeColor={color};strokeWidth={width_pt * PT:.2f};"
            + ("dashed=1;dashPattern=2 1.5;" if dashed else ""))


def mm(v):
    return v * U


EQ_SIZES = json.load(open(osp.join(RESULTS_DIR, "epistasis_model_equation_sizes.json")))


def equation(doc, cid, name, cx_mm, cy_mm, scale=1.0):
    """Place expression `name`, centered on (cx_mm, cy_mm), at its recorded size.

    The size comes from the panels script, which measured the rendered ink, so a changed
    expression changes its box here without anything being re-measured by hand.
    """
    w, h = (v * scale for v in EQ_SIZES[name])
    payload = base64.b64encode(
        open(osp.join(IMAGES_DIR, f"mi_eq_{name}.svg"), "rb").read()).decode("ascii")
    doc.vertex(cid, "", "shape=image;imageAspect=0;aspect=fixed;html=1;"
               f"image=data:image/svg+xml,{payload};",
               mm(cx_mm - w / 2), mm(cy_mm - h / 2), mm(w), mm(h))
    return w, h


def text(doc, cid, value, x, y, w, h, align="left", extra=""):
    doc.vertex(cid, value, text_style(align, extra), mm(x), mm(y), mm(w), mm(h))


def motivation(doc, x0, y0, w, h):
    """Panel a: a yeast in nature against a yeast in a bioreactor.

    Left, a cell drawn as a web: many nodes, many cross-links, most of them off the
    curated map. Right, a vessel with one arrow out of it, the single titer a campaign
    measures. Between them, what a campaign does. The two glyphs are native shapes so
    that an illustration (BioRender or otherwise) can replace either without touching
    the text around it: each occupies a fixed box named in its cell id.
    """
    third = (w - 2 * 4.0) / 3.0
    glyph_h = 15.0
    gy = y0 + 3.5
    top = "verticalAlign=top;"
    # --- the cell in nature: an ellipse holding a dense, cross-linked web
    cx, cy = x0 + third / 2, gy + glyph_h / 2
    doc.vertex("a-cell", "", shape_style("ellipse", C_YELLOW),
               mm(cx - 13.0), mm(cy - glyph_h / 2), mm(26.0), mm(glyph_h))
    nodes = [(-8.5, -3.0), (-4.0, -5.2), (1.0, -5.0), (6.0, -3.5), (8.8, 0.5),
             (5.5, 4.2), (0.5, 5.0), (-4.5, 4.4), (-8.5, 1.5), (-1.5, 0.0), (3.0, 0.8)]
    links = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 0),
             (0, 9), (9, 2), (9, 6), (9, 4), (1, 9), (10, 3), (10, 5), (10, 9), (10, 7),
             (8, 10), (1, 7), (2, 5)]
    for k, (i, j) in enumerate(links):
        (xa, ya), (xb, yb) = nodes[i], nodes[j]
        doc.edge(f"a-link{k}", line_style(C_RULE, 0.35),
                 points=[(mm(cx + xa), mm(cy + ya)), (mm(cx + xb), mm(cy + yb))])
    for k, (dx, dy) in enumerate(nodes):
        doc.vertex(f"a-node{k}", "", shape_style("ellipse", C_RULE, fill="#FFFFFF"),
                   mm(cx + dx - 0.8), mm(cy + dy - 0.8), mm(1.6), mm(1.6))
    text(doc, "a-cell-t", "yeast in nature", x0, y0 - 0.5, third, 4, "center", "fontStyle=1;")
    text(doc, "a-cell-n",
         "robust and cross-coupled; fourteen times the curated model's reaction space, "
         "at weaker affinities",
         x0, gy + glyph_h + 1.0, third, h - glyph_h - 4.5, "center", top)

    # --- what a campaign does: title, what is done, the arrow, what is left
    mx = x0 + third + 4.0
    text(doc, "a-mid-t", "engineer", mx, y0 - 0.5, third, 4, "center", "fontStyle=1;")
    text(doc, "a-mid-n",
         "delete or overexpress a few genes; keep the strain that makes more of one product",
         mx, y0 + 3.5, third, 10, "center", top)
    doc.edge("a-arrow", line_style(C_RULE, 0.6, arrow=True),
             points=[(mm(mx + 3.0), mm(y0 + 15.5)), (mm(mx + third - 3.0), mm(y0 + 15.5))])
    text(doc, "a-mid-n2",
         "the coupling that made the cell robust is still there, and is not measured",
         mx, y0 + 17.5, third, 10, "center", top)

    # --- the factory: a vessel, an impeller, one arrow out
    rx = mx + third + 4.0
    vx, vw = rx + 1.0, 14.0
    doc.vertex("a-vessel", "", shape_style("rounded=1;arcSize=25", C_MULT),
               mm(vx), mm(gy + 1.0), mm(vw), mm(glyph_h - 1.0))
    doc.edge("a-shaft", line_style(C_MULT, 0.6),
             points=[(mm(vx + vw / 2), mm(gy - 0.5)), (mm(vx + vw / 2), mm(gy + glyph_h - 4.5))])
    doc.vertex("a-impeller", "", shape_style("rounded=0", C_MULT, fill=C_MULT),
               mm(vx + vw / 2 - 3.0), mm(gy + glyph_h - 5.0), mm(6.0), mm(1.2))
    doc.edge("a-product", line_style(C_RULE, 0.6, arrow=True),
             points=[(mm(vx + vw), mm(cy)), (mm(vx + vw + 7.0), mm(cy))])
    text(doc, "a-product-t", "one titer", vx + vw - 1.0, cy - 5.0, 10.0, 4, "center")
    text(doc, "a-fac-t", "yeast in a bioreactor", rx, y0 - 0.5, third, 4, "center",
         "fontStyle=1;")
    text(doc, "a-fac-n",
         "one product, one number per strain; what pays off in nature is not what a "
         "factory needs",
         rx, gy + glyph_h + 1.0, third, h - glyph_h - 4.5, "center", top)


def classic(doc, x0, y0, w, h):
    """Panel b: an interaction is the departure of a measured combination from an
    expectation.

    Left, the bars: base strain, the two singles, the expected double as an outline and
    the measured double filled, with the gap between them named above. A negative
    interaction is drawn because it is the common case on growth. Right, what is known
    about how common such departures are on yeast growth, and what predicting them would
    buy.
    """
    # --- bars, in titer relative to the base strain, generic values
    bars = [("base", 1.00, C_RULE, False), ("i", 0.70, C_MULT, False),
            ("j", 0.60, C_MULT, False), ("expected", 0.42, C_MULT, True),
            ("measured", 0.22, C_ADD, False)]
    bar_w, gap, axis_h = 7.0, 2.4, 12.5
    bx = x0 + 2.0
    base_y = y0 + 6.0 + axis_h
    xs = []
    for k, (name, v, color, dashed) in enumerate(bars):
        x = bx + k * (bar_w + gap)
        xs.append(x)
        fill = "#FFFFFF" if dashed else FILL[color]
        doc.vertex(f"b-bar{k}", "", shape_style("rounded=0", color, fill=fill, dashed=dashed),
                   mm(x), mm(base_y - v * axis_h), mm(bar_w), mm(v * axis_h))
    # axis line and the base level
    doc.edge("b-axis", line_style("#000000", 0.5),
             points=[(mm(bx - 1.0), mm(base_y)), (mm(xs[-1] + bar_w + 1.0), mm(base_y))])
    doc.edge("b-base", line_style(C_RULE, 0.4, dashed=True),
             points=[(mm(bx - 1.0), mm(base_y - axis_h)),
                     (mm(xs[-1] + bar_w + 1.0), mm(base_y - axis_h))])
    # labels under the bars: words for the base and the double, math for the singles
    lw = bar_w + gap
    text(doc, "b-l0", "base", xs[0] - gap / 2, base_y + 0.3, lw, 4, "center")
    equation(doc, "b-l1", "f_i", xs[1] + bar_w / 2, base_y + 2.2)
    equation(doc, "b-l2", "f_j", xs[2] + bar_w / 2, base_y + 2.2)
    text(doc, "b-l3", "expected", xs[3] - gap / 2, base_y + 0.3, lw, 4, "center")
    text(doc, "b-l4", "measured", xs[4] - gap / 2, base_y + 0.3, lw, 4, "center")
    equation(doc, "b-l3e", "mult", xs[3] + bar_w / 2, base_y + 5.2)
    equation(doc, "b-l4e", "f_ij", xs[4] + bar_w / 2, base_y + 5.2)
    # the interaction: a double-headed bracket from the expected top to the measured top,
    # and its definition above the bars, where nothing else is
    ax_x = xs[4] + bar_w + 1.4
    doc.edge("b-gap", line_style("#000000", 0.5, arrow=True)
             + "startArrow=classic;startFill=1;startSize=2.5;",
             points=[(mm(ax_x), mm(base_y - 0.42 * axis_h)), (mm(ax_x), mm(base_y - 0.22 * axis_h))])
    doc.edge("b-tick1", line_style(C_RULE, 0.3, dashed=True),
             points=[(mm(xs[3] + bar_w), mm(base_y - 0.42 * axis_h)), (mm(ax_x + 0.8), mm(base_y - 0.42 * axis_h))])
    doc.edge("b-lead", line_style(C_RULE, 0.3),
             points=[(mm(ax_x), mm(base_y - 0.42 * axis_h - 0.5)), (mm(ax_x), mm(y0 + 3.4))])
    equation(doc, "b-eps", "eps_def", ax_x - 5.0, y0 + 1.6)
    text(doc, "b-eps-t", "the interaction", ax_x - 12.0, y0 - 0.5 - 2.6, 14, 4, "center")

    # --- the right column: what is known on growth, and what predicting buys
    tx = ax_x + 2.5
    tw = x0 + w - tx
    top = "verticalAlign=top;"
    text(doc, "b-t1",
         f"On yeast growth, genome-wide: ~{GROWTH_COUNTS['digenic_negative']} negative to "
         f"~{GROWTH_COUNTS['digenic_positive']} positive digenic interactions, and ~"
         f"{GROWTH_COUNTS['trigenic_fold']} times as many trigenic, ~"
         f"{GROWTH_COUNTS['trigenic_weaker_pct']}% weaker and scored negative.",
         tx, y0 - 0.5, tw, 16, "left", top)
    text(doc, "b-t2",
         "Predicted, an interaction is a step a design can climb. Unpredicted, a campaign "
         "is left to trial and error, and the combinations that raise a product are the "
         "rarer sign.",
         tx, y0 + 16.0, tw, 15, "left", top)


def schematic(doc, x0, y0, w):
    """Where the two families of null come from, as one short row.

    Left, a flux through two steps, each keeping a fraction. Right, one pool that each
    deletion takes an absolute amount out of. One line under each, and under both the
    identity that is the whole of their disagreement. The titles the halves carried and
    the second sentence under each were cut (author review, 2026.09.18): the boxes say
    what the halves are, and the row is 15.5 mm rather than 22.
    """
    half = (w - 8.0) / 2.0
    x1 = x0 + half + 8.0

    # --- fractions compose
    bw, gap = 17.0, 3.0
    xs = [x0 + i * (bw + gap) for i in range(4)]
    box_y, box_h = y0, 8.0
    for i, x in enumerate(xs):
        color = C_MULT if 0 < i < 3 else C_RULE
        doc.vertex(f"d-mult{i}", "", box_style(color), mm(x), mm(box_y), mm(bw), mm(box_h))
    for i in range(3):
        doc.edge(f"d-multa{i}", line_style(C_RULE, 0.5, arrow=True),
                 source=f"d-mult{i}", target=f"d-mult{i + 1}")
    # Box 0 is a number and boxes 1 and 2 are a word over an expression, so the word is a
    # cell of its own above the image rather than part of a label draw.io would not typeset.
    doc.vertex("d-mult0-t", "base 1.00", text_style("center"),
               mm(xs[0]), mm(box_y + box_h / 2) - 7, mm(bw), 14)
    for i, name in ((1, "f_i"), (2, "f_j")):
        doc.vertex(f"d-mult{i}-t", "keeps", text_style("center"),
                   mm(xs[i]), mm(box_y + 2.3) - 7, mm(bw), 14)
        equation(doc, f"d-mult{i}-eq", name, xs[i] + bw / 2, box_y + 5.7)
    equation(doc, "d-mult3-eq", "mult", xs[3] + bw / 2, box_y + box_h / 2)
    doc.vertex("d-mult-note",
               "FRACTIONS in series: the second step acts only on what the first let through",
               text_style("left"), mm(x0), mm(y0 + 8.6), mm(half), 12)

    # --- amounts add
    pool_y, pool_h = y0, 8.0
    loss_w = 21.0
    keep_w = half - 2 * loss_w
    doc.vertex("d-pool-keep", "", cell_style("#FFFFFF", "center"),
               mm(x1), mm(pool_y), mm(keep_w), mm(pool_h))
    doc.vertex("d-pool-i", "", cell_style(FILL[C_ADD], "center"),
               mm(x1 + keep_w), mm(pool_y), mm(loss_w), mm(pool_h))
    doc.vertex("d-pool-j", "", cell_style(FILL[C_ADD], "center"),
               mm(x1 + keep_w + loss_w), mm(pool_y), mm(loss_w), mm(pool_h))
    equation(doc, "d-pool-keep-eq", "add", x1 + keep_w / 2, pool_y + pool_h / 2)
    equation(doc, "d-pool-i-eq", "loss_i", x1 + keep_w + loss_w / 2, pool_y + pool_h / 2)
    equation(doc, "d-pool-j-eq", "loss_j", x1 + keep_w + 1.5 * loss_w, pool_y + pool_h / 2)
    doc.vertex("d-add-note",
               "AMOUNTS from one pool: the two losses overlap, and the overlap is counted twice",
               text_style("left"), mm(x1), mm(y0 + 8.6), mm(half), 12)

    # --- the identity, as three cells so the expression is typeset rather than spelled
    eq_w = EQ_SIZES["gap"][0]
    lead_w = 70.0
    y_gap = y0 + 13.5
    doc.vertex("d-gap-a", "the two nulls differ by exactly", text_style("right"),
               mm(x0), mm(y_gap) - 7, mm(lead_w), 14)
    equation(doc, "d-gap-eq", "gap", x0 + lead_w + 1.0 + eq_w / 2, y_gap)
    doc.vertex("d-gap-b", ", the loss the second deletion would take again",
               text_style("left"), mm(x0 + lead_w + 2.0 + eq_w), mm(y_gap) - 7,
               mm(x0 + w - (x0 + lead_w + 2.0 + eq_w)), 14)


def table(doc, y0):
    """The four models on one grid, one row each.

    Rows hold two lines of 5.98 pt type, 4.2 mm of ink, in 6.4 mm; the 8 mm they had
    carried a band of air the page could no longer afford once the top row was added.
    """
    head_h, row_h = 5.0, 6.4
    x = 2.0
    for j, (title, w) in enumerate(TABLE_COLS):
        doc.vertex(f"e-h{j}", title, cell_style("#F5F5F5", "left", bold=True),
                   mm(x), mm(y0), mm(w), mm(head_h))
        x += w
    for i, (color, name, eq, *cols) in enumerate(TABLE_ROWS):
        x = 2.0
        y = y0 + head_h + i * row_h
        values = [name, ""] + list(cols)
        for j, ((_, w), value) in enumerate(zip(TABLE_COLS, values)):
            fill = FILL[color] if j == 0 else "#FFFFFF"
            doc.vertex(f"e-r{i}c{j}", value, cell_style(fill, "left", bold=(j == 0)),
                       mm(x), mm(y), mm(w), mm(row_h))
            if j == 1:
                equation(doc, f"e-r{i}-eq", eq, x + w / 2, y + row_h / 2)
            x += w
    return y0 + head_h + len(TABLE_ROWS) * row_h


def build(out_path):
    doc = Doc(osp.splitext(osp.basename(out_path))[0])

    doc.layer("printbox", "print box", visible=False)
    doc.vertex("printbox-rect", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=none;"
               "strokeColor=#B85450;strokeWidth=1;dashed=1;", 0, 0, mm(179.4), mm(170),
               parent="printbox")

    # --- row 0: the two schematic panels, half the width each
    # Letters at 2 and 90.5 mm, content indented 4 mm past each, and both panels end
    # inside the 177 mm the other rows use.
    half = (W_MM - 6.0) / 2.0
    doc.vertex("letter-a", "a", letter_style(), mm(2), mm(Y_ROW0 - LETTER_DY), 24, 18)
    motivation(doc, 6.0, Y_ROW0, half - 4.0, H_ROW0)
    xb = 2.0 + half + 2.0
    doc.vertex("letter-b", "b", letter_style(), mm(xb), mm(Y_ROW0 - LETTER_DY), 24, 18)
    classic(doc, xb + 4.0, Y_ROW0, 177.0 - (xb + 4.0), H_ROW0)

    # --- row 1: the three embedded data panels
    x = 2.0
    for letter, stem in zip("cde", PANEL_STEM):
        svg = osp.join(IMAGES_DIR, stem + ".svg")
        w_u, h_u = svg_size_units(svg)
        doc.vertex(f"letter-{letter}", letter, letter_style(),
                   mm(x), mm(Y_ROW1 - LETTER_DY), 24, 18)
        doc.vertex(f"panel-{letter}", "",
                   "shape=image;imageAspect=0;aspect=fixed;html=1;"
                   f"image={data_uri(svg)};",
                   mm(x), mm(Y_ROW1), w_u, h_u)
        x += w_u / U + 1.0

    # --- row 2: the four models' level sets, side by side and to one scale
    svg = osp.join(IMAGES_DIR, "mi_panel_level_sets.svg")
    w_u, h_u = svg_size_units(svg)
    doc.vertex("letter-f", "f", letter_style(), mm(2), mm(Y_ROW2 - LETTER_DY), 24, 18)
    doc.vertex("panel-f", "", "shape=image;imageAspect=0;aspect=fixed;html=1;"
               f"image={data_uri(svg)};", mm(2), mm(Y_ROW2), w_u, h_u)

    # --- row 3: where the two families of null come from
    doc.vertex("letter-g", "g", letter_style(), mm(2), mm(Y_ROW3 - LETTER_DY), 24, 18)
    schematic(doc, 6.0, Y_ROW3, 171.0)

    doc.vertex("letter-h", "h", letter_style(), mm(2), mm(Y_ROW4 - LETTER_DY), 24, 18)
    bottom = table(doc, Y_ROW4)
    if bottom > 170.0:
        raise ValueError(f"content reaches {bottom:.1f} mm, over the 170 mm cap")

    doc.write(out_path)
    print(f"wrote {out_path}  (content to {bottom:.1f} mm)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=osp.join(
        DRAWIO_DIR, "ffa-epistasis-fig3-model-intuition.drawio"))
    ap.add_argument("--drawio", default=None)
    args = ap.parse_args()
    os.makedirs(osp.dirname(args.out), exist_ok=True)
    build(args.out)
    if args.drawio:
        stem = osp.splitext(osp.basename(args.out))[0]
        export(args.drawio, args.out, osp.join(IMAGES_DIR, stem + ".svg"),
               osp.join(IMAGES_DIR, stem + ".png"))


if __name__ == "__main__":
    main()
