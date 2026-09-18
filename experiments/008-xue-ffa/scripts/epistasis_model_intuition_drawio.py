# experiments/008-xue-ffa/scripts/epistasis_model_intuition_drawio.py
# [[experiments.008-xue-ffa.scripts.epistasis_model_intuition_drawio]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/epistasis_model_intuition_drawio
#
# The epistasis-model figure as a native draw.io page: four embedded data panels (written
# by epistasis_model_intuition_panels.py), a schematic of where the two families of null
# come from, and a table of what each of the four models assumes and estimates. Everything
# except the panels and the equations is an mxCell, so the wording and the arrangement can
# be worked on in draw.io.
#
# EVERY EXPRESSION IS AN IMAGE, typeset as math by the panels script and placed here at its
# recorded size. draw.io's HTML <sub> is not typesetting: the variables come out upright
# where they should be italic, the subscripts sit at the wrong size, and the PDF export set
# the subscript runs in a SERIF fallback, which put a fourth typeface into the figure.
#
# WHY A SCHEMATIC AT ALL. The four models are four lines of algebra in the Methods, and a
# reader who takes the algebra at face value will read "multiplicative" as the default and
# the other three as robustness checks. The schematic says what each null is a statement
# ABOUT: a multiplicative null says two deletions each keep a FRACTION of what reaches
# them, which is how independent steps of one flux compose and is why the growth screens
# use it; an additive null says each deletion removes an ABSOLUTE amount from a shared
# pool, which is closer to how a titer is built. The two differ by exactly the quantity
# drawn in panel d, (1 - f_i)(1 - f_j): the part of the first deletion's loss that the
# second deletion would have taken again. That is the whole of the disagreement, and it is
# the reason the choice of null is a modeling claim about the system rather than a taste.
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

# Model colors, the same four the data panels use.
C_MULT = "#D79B00"
C_ADD = "#B85450"
C_GLM = "#9673A6"
C_OLS = "#6C8EBF"
FILL = {C_MULT: "#FFE6CC", C_ADD: "#F8CECC", C_GLM: "#E1D5E7", C_OLS: "#DAE8FC"}
C_RULE = "#666666"

W_MM = 179.0
PANEL_STEM = ["mi_panel_expectations", "mi_panel_null_fit", "mi_panel_mean_variance"]

# Row geometry in millimetres from the top of the page. Four rows, each the full width:
# the three measured panels, then the level sets of all four models side by side, then
# the schematic, then the table. The schematic shared a row with a single surface panel
# before, which made it half as wide and twice as tall as it needed to be and put two
# unlike things on one line (author review, 2026.09.18).
Y_ROW1 = 5.0          # top of the three measured panels
H_ROW1 = 52.0
Y_ROW2 = 63.0         # top of the level-set row
Y_ROW3 = 104.0        # top of the schematic
H_ROW3 = 22.0
Y_ROW4 = 131.0        # top of the table
LETTER_DY = 4.6       # a letter sits this far above its block

TABLE_COLS = [
    ("model", 20.0),
    ("expects, for a double", 27.0),
    ("residual measured on", 21.0),
    ("fit to, and what that assumes", 44.0),
    ("the reading that makes it the natural null", 63.0),
]
# (color, name, equation key, residual scale, what it is fit to, the reading)
TABLE_ROWS = [
    (C_MULT, "multiplicative", "mult",
     "linear titer",
     "strain means; one standard error per strain, propagated by the delta method",
     "each deletion keeps a fixed FRACTION of what reaches it, so independent steps of one "
     "flux compose; the null of the growth screens"),
    (C_ADD, "additive", "add",
     "linear titer",
     "strain means; one standard error per strain, propagated by the delta method",
     "each deletion removes a fixed AMOUNT from a shared pool, closer to how a titer in "
     "mg/L is built up"),
    (C_GLM, "GLM log-link", "glm",
     "log titer",
     "replicate titers; Gamma family, spread proportional to the mean (panel c)",
     "the readout is positive and noisier where it is larger, and every strain's replicates "
     "are used rather than its mean"),
    (C_OLS, "log-OLS", "glm",
     "log titer",
     "replicate log ratios to the base strain; constant spread on the log scale",
     "the question is whether a deletion's FOLD effect carries into a new background, which "
     "is a log-scale question"),
]


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


def box_style(color, dashed=False):
    return (f"rounded=1;arcSize=18;whiteSpace=wrap;html=1;fillColor={FILL.get(color, '#FFFFFF')};"
            f"strokeColor={color};strokeWidth={0.5 * PT:.2f};fontFamily=Arial;"
            f"fontSize={FONT};fontColor=#000000;verticalAlign=middle;"
            + ("dashed=1;dashPattern=3 2;" if dashed else ""))


def cell_style(fill="#FFFFFF", align="left", bold=False):
    return (f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={C_RULE};"
            f"strokeWidth={0.25 * PT:.2f};fontFamily=Arial;fontSize={FONT};"
            f"fontColor=#000000;align={align};verticalAlign=middle;spacingLeft=2;"
            f"spacingRight=2;" + ("fontStyle=1;" if bold else ""))


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


def schematic(doc, x0, y0, w):
    """Where the two families of null come from, as one row of its own.

    Left, a flux through two steps, each keeping a fraction. Right, one pool that each
    deletion takes an absolute amount out of. Under both, the identity that is the whole
    of their disagreement, which the level sets above show as two shapes.

    The two halves sit side by side rather than stacked. Stacked in half the page width
    they ran to 52 mm of height for 22 mm of content, and the reader met the second story
    only after scrolling past the first, though the two are a pair.
    """
    half = (w - 8.0) / 2.0
    x1 = x0 + half + 8.0

    # --- fractions compose
    doc.vertex("d-mult-title", "effects are FRACTIONS: two steps in series",
               text_style("left", "fontStyle=1;"), mm(x0), mm(y0), mm(half), 12)
    bw, gap = 17.0, 3.0
    xs = [x0 + i * (bw + gap) for i in range(4)]
    box_y, box_h = y0 + 4.0, 8.0
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
               "a fraction of a fraction: the second step acts only on what the first "
               "let through",
               text_style("left"), mm(x0), mm(y0 + 13.0), mm(half), 16)

    # --- amounts add
    doc.vertex("d-add-title", "effects are AMOUNTS: two draws on one pool",
               text_style("left", "fontStyle=1;"), mm(x1), mm(y0), mm(half), 12)
    pool_y, pool_h = y0 + 4.0, 8.0
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
               "each loss is taken from the whole pool, so the part the first deletion "
               "removed is counted a second time",
               text_style("left"), mm(x1), mm(y0 + 13.0), mm(half), 16)

    # --- the identity, as three cells so the expression is typeset rather than spelled
    eq_w = EQ_SIZES["gap"][0]
    lead_w = 70.0
    y_gap = y0 + 19.0
    doc.vertex("d-gap-a", "the two nulls differ by exactly", text_style("right"),
               mm(x0), mm(y_gap) - 7, mm(lead_w), 14)
    equation(doc, "d-gap-eq", "gap", x0 + lead_w + 1.0 + eq_w / 2, y_gap)
    doc.vertex("d-gap-b", ", the loss the second deletion would take again",
               text_style("left"), mm(x0 + lead_w + 2.0 + eq_w), mm(y_gap) - 7,
               mm(x0 + w - (x0 + lead_w + 2.0 + eq_w)), 14)


def table(doc, y0):
    """The four models on one grid, one row each.

    Rows are sized to the two-line cells they actually hold; at 13 mm they carried a band
    of empty space under every row and pushed the figure past the page, and at 10 mm they
    still carried one (author review, 2026.09.18). Two lines of 5.98 pt type occupy
    4.2 mm, so 8 mm is the text plus a millimetre of air above and below it.
    """
    head_h, row_h = 5.5, 8.0
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

    # --- row 1: the three embedded data panels
    x = 2.0
    for letter, stem in zip("abc", PANEL_STEM):
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
    doc.vertex("letter-d", "d", letter_style(), mm(2), mm(Y_ROW2 - LETTER_DY), 24, 18)
    doc.vertex("panel-d", "", "shape=image;imageAspect=0;aspect=fixed;html=1;"
               f"image={data_uri(svg)};", mm(2), mm(Y_ROW2), w_u, h_u)

    # --- row 3: where the two families of null come from
    doc.vertex("letter-e", "e", letter_style(), mm(2), mm(Y_ROW3 - LETTER_DY), 24, 18)
    schematic(doc, 6.0, Y_ROW3, 171.0)

    doc.vertex("letter-f", "f", letter_style(), mm(2), mm(Y_ROW4 - LETTER_DY), 24, 18)
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
