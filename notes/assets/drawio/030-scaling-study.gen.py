"""Generate the four planned figures of notes-tex/modeling/030-scaling-study (no data).

Writes, next to itself:
  030-scaling-grid.drawio          the grid: N ladder x D ladder, seeds, the 025 pair, budgets
  030-scaling-readout.drawio       the readout pipeline from W&B keys to fits and the gap
  030-scaling-results-plan.drawio  the three-panel results figure as a sketch
  030-scaling-reading.drawio       how each fit is read: the questions and their outcomes

The grid figure reads its numbers (N per size, records and rows per subset, budgets)
from experiments/030-solid-growth-multi/results/scaling_grid_table_030.json, written by
scaling_grid_table_030.py, so the figure and the tables agree. Everything else is a
sketch: curves are drawn by hand to show what each panel will hold, not measured.

Export with the recipe in [[drawio-headless-export-gilahyper]] (input file FIRST):
  xvfb-run -a <drawio> 030-scaling-grid.drawio --no-sandbox --disable-gpu \
      -x -f png -s 4 -o 030-scaling-grid.drawio.png
Sizes are on the font ladder (8.3 / 9.7 / 11.1 bold panel letters);
paper/nature-biotech/scripts/drawio_font_band.py --check audits them.
"""

import json
import os.path as osp
from xml.sax.saxutils import escape

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.dirname(osp.dirname(osp.dirname(HERE)))
TABLE_JSON = osp.join(
    REPO, "experiments/030-solid-growth-multi/results/scaling_grid_table_030.json"
)

OR, ORF = "#D79B00", "#FFE6CC"
RD, RDF = "#B85450", "#F8CECC"
PU, PUF = "#9673A6", "#E1D5E7"
YE, YEF = "#D6B656", "#FFF2CC"
BL, BLF = "#6C8EBF", "#DAE8FC"
GR, GRF = "#666666", "#F5F5F5"
BK = "#000000"

W = 702  # full-width target in draw.io units (100 per inch); the cap is 706.7
FS = 8.3  # 5.98 pt, the default rung of the ladder
FT = 9.7  # 6.98 pt, titles
FL = 11.1  # 7.99 pt, panel letters


def label_html(s: str) -> str:
    """Escape a label for the XML attribute; a newline becomes an HTML break."""
    return escape(s).replace("\n", "&lt;br&gt;")


class Canvas:
    """A list of mxCells with the helpers of the FigS-graph-regularization-sweep
    generator, so every figure here shares one vocabulary of shapes."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.cells: list[str] = []
        self._n = 0

    def nid(self) -> str:
        self._n += 1
        return f"c{self._n}"

    def text(
        self, x, y, w, h, s, size=FS, bold=False, align="left", valign="top", rot=0,
        color=BK, italic=False,
    ) -> None:
        st = (
            f"text;html=1;strokeColor=none;fillColor=none;align={align};"
            f"verticalAlign={valign};whiteSpace=wrap;rounded=0;fontFamily=Arial;"
            f"fontSize={size};fontColor={color};"
        )
        if bold:
            st += "fontStyle=1;"
        if italic:
            st += "fontStyle=2;"
        if rot:
            st += f"rotation={rot};"
        self.cells.append(
            f'<mxCell id="{self.nid()}" value="{label_html(s)}" style="{st}" vertex="1" '
            f'parent="1"><mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
            f'height="{h:.1f}" as="geometry"/></mxCell>'
        )

    def rect(
        self, x, y, w, h, fill="#FFFFFF", stroke=BK, sw=0.75, label="", dashed=False,
        size=FS, bold=False, align="center", rounded=False,
    ) -> None:
        st = (
            f"rounded={1 if rounded else 0};whiteSpace=wrap;html=1;fillColor={fill};"
            f"strokeColor={stroke};strokeWidth={sw};fontFamily=Arial;fontSize={size};"
            f"align={align};"
        )
        if dashed:
            st += "dashed=1;"
        if bold:
            st += "fontStyle=1;"
        if align == "left":
            st += "spacingLeft=4;"
        self.cells.append(
            f'<mxCell id="{self.nid()}" value="{label_html(label)}" style="{st}" vertex="1" '
            f'parent="1"><mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
            f'height="{h:.1f}" as="geometry"/></mxCell>'
        )

    def line(self, pts, color=BK, sw=1.0, dashed=False, curved=False, arrow=False) -> None:
        st = (
            f"endArrow={'classic' if arrow else 'none'};html=1;strokeWidth={sw};"
            f"strokeColor={color};curved={1 if curved else 0};endSize=4;"
        )
        if dashed:
            st += "dashed=1;"
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        mid = "".join(f'<mxPoint x="{px:.1f}" y="{py:.1f}"/>' for px, py in pts[1:-1])
        self.cells.append(
            f'<mxCell id="{self.nid()}" style="{st}" edge="1" parent="1">'
            f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x0:.1f}" y="{y0:.1f}" '
            f'as="sourcePoint"/><mxPoint x="{x1:.1f}" y="{y1:.1f}" as="targetPoint"/>'
            f'<Array as="points">{mid}</Array></mxGeometry></mxCell>'
        )

    def dot(self, x, y, color, r=2.4, fill=None, square=False, sw=0.75) -> None:
        shape = "" if square else "ellipse;aspect=fixed;"
        st = (
            f"{shape}whiteSpace=wrap;html=1;fillColor={fill or color};strokeColor={color};"
            f"strokeWidth={sw};"
        )
        self.cells.append(
            f'<mxCell id="{self.nid()}" value="" style="{st}" vertex="1" parent="1">'
            f'<mxGeometry x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r:.1f}" '
            f'height="{2 * r:.1f}" as="geometry"/></mxCell>'
        )

    def letter(self, x, y, s) -> None:
        self.text(x, y, 14, 14, s, size=FL, bold=True)

    def write(self, page_h: int) -> str:
        body = "\n".join(self.cells)
        xml = (
            '<mxfile host="app.diagrams.net" agent="claude-code">\n'
            f'  <diagram name="{self.name}" id="{self.name}">\n'
            f'    <mxGraphModel dx="1400" dy="1000" grid="0" page="1" pageWidth="{W}" '
            f'pageHeight="{page_h}" math="0" shadow="0">\n      <root>\n'
            '        <mxCell id="0"/><mxCell id="1" parent="0"/>\n'
            f"{body}\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n"
        )
        path = osp.join(HERE, f"{self.name}.drawio")
        with open(path, "w") as f:
            f.write(xml)
        return path


def panel(c: Canvas, px, py, pw, ph, title, xlabel, ylabel, note="", note_lines=0):
    """Panel frame, title, axes box and labels; returns (ax, ay, aw, ah)."""
    c.rect(px, py, pw, ph, fill="#FFFFFF", stroke=GR, sw=0.5)
    c.text(px + 2, py + 1, pw - 4, 24, title, bold=True)
    note_h = 11 * note_lines + (6 if note_lines else 0)
    ax, ay, aw, ah = px + 30, py + 28, pw - 38, ph - 30 - 14 - 13 - 8 - note_h
    c.rect(ax, ay, aw, ah, fill="#FFFFFF", stroke=BK, sw=0.75)
    if note:
        c.text(px + 4, py + ph - note_h - 2, pw - 8, note_h, note)
    c.text(px + 12 - ah / 2, ay + ah / 2 - 8, ah, 16, ylabel, align="center",
           valign="middle", rot=-90)
    c.text(ax, ay + ah + 14, aw, 13, xlabel, align="center")
    return ax, ay, aw, ah


def xticks(c: Canvas, ax, ay, aw, ah, labels):
    n = len(labels)
    xs = []
    for i, lab in enumerate(labels):
        xx = ax + aw * (i + 0.5) / n
        c.line([(xx, ay + ah), (xx, ay + ah + 3)], BK, sw=0.75)
        c.text(xx - 24, ay + ah + 2, 48, 11, lab, align="center")
        xs.append(xx)
    return xs


def fmt_k(n: float) -> str:
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


# ------------------------------------------------------------------ figure 1: the grid
def fig_grid(t: dict) -> str:
    c = Canvas("030-scaling-grid")
    sizes = t["sizes"]
    subsets = t["subsets"]  # 1/64, 1/16, 1/4, full
    LBL_W = 96
    COL_W, GAP = 118, 8
    X0 = LBL_W + 10
    cols = [X0 + i * (COL_W + GAP) for i in range(4)]
    ROW_H = 24
    y = 0
    c.letter(0, y, "a")
    c.text(16, y - 1, 560, 14, "The 030 grid: every (size, subset) at seed 1, replicates where they are cheap, "
           "the 025 pair at the current size", bold=True, size=FT)
    y += 18
    # header row: subset, records, rows
    c.text(0, y, LBL_W, 36, "columns: nested S3Q subsets\nrows: size and N", bold=True)
    for i, s in enumerate(subsets):
        name = "full" if s["denominator"] == 1 else f"1/{s['denominator']}"
        c.rect(cols[i], y, COL_W, 36, fill=GRF, stroke=GR, sw=0.5,
               label=f"{name}: {s['pairs_kept']} pairs\n{fmt_k(s['train_records'])} records, "
                     f"{fmt_k(s['gi_entry_rows'])} gi rows")
    y += 40
    for r in sizes:
        label = f"{r['width']} x {r['depth']}:  N {fmt_k(r['params_non_embedding'])}"
        c.rect(0, y, LBL_W, ROW_H, fill=ORF, stroke=OR, sw=0.5, label=label, align="left")
        for i, s in enumerate(subsets):
            d = s["denominator"]
            c.rect(cols[i], y, COL_W, ROW_H, fill="#FFFFFF", stroke=GR, sw=0.5)
            cx, cy = cols[i] + 12, y + ROW_H / 2
            c.dot(cx, cy, OR)
            seeds = 1
            if d in (64, 16) and (r["width"], r["depth"]) in ((45, 2), (180, 8)):
                seeds = 3
                c.dot(cx + 8, cy, OR)
                c.dot(cx + 16, cy, OR)
            txt = f"{seeds} seed" + ("s" if seeds > 1 else "")
            if d == 4 and (r["width"], r["depth"]) == (45, 2):
                c.dot(cx + 10, cy, RD, r=3.4, fill="#FFFFFF", sw=1.0)
                txt = "1 seed + lr 1e-4, 1e-3"
            if d == 1 and (r["width"], r["depth"]) == (180, 8):
                txt = "1 seed, 10 epochs: headline"
            c.text(cols[i] + (30 if seeds > 1 or d == 4 and r["width"] == 45 else 22),
                   y + 1, COL_W - 32, ROW_H - 2, txt, valign="middle")
        y += ROW_H + 2
    # the 025 pair
    y += 6
    c.rect(0, y, LBL_W, ROW_H, fill=PUF, stroke=PU, sw=0.5,
           label="025 pair, 180 x 8\nmerged label", align="left")
    for i, s in enumerate(subsets):
        d = s["denominator"]
        c.rect(cols[i], y, COL_W, ROW_H, fill="#FFFFFF", stroke=GR, sw=0.5)
        cx, cy = cols[i] + 12, y + ROW_H / 2
        c.dot(cx, cy, PU, square=True)
        seeds = 1
        if d in (64, 16):
            seeds = 3
            c.dot(cx + 8, cy, PU, square=True)
            c.dot(cx + 16, cy, PU, square=True)
        c.text(cols[i] + (30 if seeds > 1 else 22), y + 1, COL_W - 32, ROW_H - 2,
               f"{seeds} seed" + ("s" if seeds > 1 else ""), valign="middle")
    y += ROW_H + 8
    # budget strip
    rows = [
        ("epochs (one 48 h job)", [str(s["epochs"]) for s in subsets]),
        ("records seen", [f"{s['records_seen'] / 1e6:.2f}M" for s in subsets]),
        ("hours at 68 records/s (assumed)", [f"{s['hours_at_pace']:.0f}" for s in subsets]),
    ]
    for lab, vals in rows:
        c.text(0, y, LBL_W + 6, 13, lab, bold=True)
        for i, v in enumerate(vals):
            c.text(cols[i], y, COL_W, 13, v, align="center")
        y += 13
    # matched budget line
    y += 4
    c.line([(cols[0], y), (cols[3] + COL_W, y)], RD, sw=1.0, dashed=True)
    c.text(cols[0], y + 2, cols[3] + COL_W - cols[0], 13,
           "matched read at B = the smallest total any point reaches, "
           f"{subsets[0]['records_seen'] / 1e6:.2f}M records (the 1/64 total); every point is read there",
           color=RD, align="center")
    y += 20
    # calibration + legend
    c.rect(0, y, 300, 34, fill=YEF, stroke=YE, sw=0.5, align="left",
           label="calibration first: 400 batches at 180 x 8 and at 45 x 2 on the full pool, 3 h each. "
                 "Equal paces mean the loader binds and no size finishes sooner than 180 x 8.")
    lx = 316
    c.dot(lx + 4, y + 7, OR)
    c.text(lx + 10, y, 170, 13, "030 per-entry arm, dataset token (seed 1)")
    c.dot(lx + 4, y + 20, PU, square=True)
    c.text(lx + 10, y + 13, 170, 13, "025 merged-label twin, same subsets")
    c.dot(lx + 200, y + 7, RD, r=3.4, fill="#FFFFFF", sw=1.0)
    c.text(lx + 206, y, 180, 13, "learning-rate check at the smallest size")
    c.text(lx + 196, y + 13, 190, 26, "every job: batch 256, lr 2.5e-4 constant, one A40, "
           "full-pool normalization constants")
    y += 40
    return c.write(y)


# -------------------------------------------------------------- figure 2: the readout
def fig_readout() -> str:
    c = Canvas("030-scaling-readout")
    c.letter(0, 0, "b")
    c.text(16, -1, 600, 14, "From W&B rows to fitted exponents: the readout, with the three joints "
           "repaired before launch", bold=True, size=FT)
    y0 = 20
    BW, BH, G = 128, 100, 14
    xs = [i * (BW + G) for i in range(5)]
    boxes = [
        ("W&B runs, one sweep tag each",
         "val/gene_interaction/MSE\nval/token/[screen]/.../MSE\ncompute/records_seen, flops,\n"
         "gpu_hours, params\nepoch on every row", BLF, BL),
        ("per-epoch table",
         "one point = (size, subset,\nseed, lr); chain links\nstitched by resumed_from;\n"
         "label units after the\ninverse transform", GRF, GR),
        ("two reads per point",
         "selected: the epoch of\nlowest pooled val MSE\nmatched: interpolated at\n"
         "B = min records seen\nover all points", YEF, YE),
        ("fits, per read",
         "L = A x^-k and L = E + A x^-k\nx: D records, D rows, N\ntargets: pooled, 2018, 2020\n"
         "largest x held out\n300 bootstraps over runs", ORF, OR),
        ("outputs",
         "[sweep]_points.csv\n[sweep]_fits.json: CI90, E,\nheld-out prediction\n"
         "[sweep]_gap.csv\nfigure a, b, c", GRF, GR),
    ]
    for (title, body, fill, stroke), x in zip(boxes, xs):
        c.rect(x, y0, BW, 16, fill=fill, stroke=stroke, sw=0.6, label=title, bold=True)
        c.rect(x, y0 + 16, BW, BH - 16, fill="#FFFFFF", stroke=stroke, sw=0.6, label=body,
               align="left")
    for i in range(4):
        c.line([(xs[i] + BW, y0 + BH / 2), (xs[i + 1], y0 + BH / 2)], GR, sw=1.2, arrow=True)
    # side inputs
    y1 = y0 + BH + 22
    c.rect(xs[3] - 60, y1, 250, 44, fill=RDF, stroke=RD, sw=0.6, align="left",
           label="replicate floor per screen, E = (1 - r) var(tau),\n"
                 "from triple_noise_ceiling_030.json;\ndrawn beside every fitted E")
    c.line([(xs[3] + 60, y1), (xs[3] + 60, y0 + BH)], RD, sw=1.0, arrow=True)
    c.rect(0, y1, 300, 44, fill=PUF, stroke=PU, sw=0.6, align="left",
           label="025 paired sweep: the same reads at the same B;\nthe gap is 030 minus 025 pooled MSE per subset,\n"
                 "seed means and sds (pooled only: 025 logs no screen key)")
    c.line([(xs[1] + BW / 2, y1), (xs[1] + BW / 2, y0 + BH)], PU, sw=1.0, arrow=True)
    # repaired joints
    y2 = y1 + 54
    c.text(0, y2, 200, 12, "repaired 2026.10.12, each verified before the fix:", bold=True, color=RD)
    fixes = [
        (xs[0], "the override transforms.fit_stats_strict=false named an undefined key and "
                "Hydra refused it; the config now defines the key"),
        (xs[1], "compute rows carried no epoch, so a history query for epoch plus a compute "
                "key returned zero rows; the callback logs epoch"),
        (xs[2], "no per-token MSE was logged, so the per-screen keys were None and the "
                "first float(None) crashed the reads; MSE logged per token"),
        (xs[3], "the matched budget was the full-data total, which no subset reached; "
                "it is now the smallest total any point reached"),
    ]
    for x, s in fixes:
        c.rect(x, y2 + 18, BW + 10, 70, fill="#FFFFFF", stroke=RD, sw=0.6, dashed=True,
               label=s, align="left")
    return c.write(y2 + 92)


# -------------------------------------------------- figure 3: the planned results figure
def fig_results_plan() -> str:
    c = Canvas("030-scaling-results-plan")
    PW = (W - 2 * 12) / 3
    PH = 212
    X = [0, PW + 12, 2 * (PW + 12)]
    TOP = 16
    c.text(X[0], 0, 400, 14, "sketch, no data: what each panel of the results figure will hold",
           italic=True, color=GR)
    # a: L(D) at 180x8
    ax, ay, aw, ah = panel(c, X[0], TOP + 16, PW, PH, "Loss against unique data at 180 x 8",
                           "D, training records (log)", "val MSE, label units (log)",
                           note="030 orange, 025 purple; filled = matched read, hollow = "
                                "selected read; dashed = replicate floors per screen; "
                                "ring = the held-out largest point, predicted from the rest",
                           note_lines=3)
    xs = xticks(c, ax, ay, aw, ah, ["1/64", "1/16", "1/4", "full"])
    y030 = [ay + 28, ay + 48, ay + 64, ay + 74]
    y025 = [ay + 14, ay + 34, ay + 50, ay + 60]
    c.line(list(zip(xs, y030)), OR, sw=1.5, curved=True)
    c.line(list(zip(xs, y025)), PU, sw=1.5, curved=True, dashed=True)
    for xx, yy in zip(xs, y030):
        c.dot(xx, yy, OR)
        c.dot(xx + 5, yy + 6, OR, fill="#FFFFFF")
    for xx, yy in zip(xs, y025):
        c.dot(xx, yy, PU, square=True)
    c.dot(xs[3], y030[3], BK, r=5.5, fill="none", sw=0.75)
    c.line([(ax + 2, ay + ah - 4), (ax + aw - 2, ay + ah - 4)], GR, sw=0.75, dashed=True)
    c.line([(ax + 2, ay + ah - 26), (ax + aw - 2, ay + ah - 26)], GR, sw=0.75, dashed=True)
    c.text(ax + 4, ay + ah - 43, 90, 11, "E, 2020 same-strain", color=GR)
    c.text(ax + 4, ay + ah - 21, 90, 11, "E, 2018 bounds", color=GR)
    c.text(ax + aw - 132, ay + 2, 130, 24, "slope = beta on L - E; flattens\nif the full point is unconverged",
           align="right")
    # b: L(N) at full D
    ax, ay, aw, ah = panel(c, X[1], TOP + 16, PW, PH, "Loss against size at the full pool",
                           "N, non-embedding parameters (log)", "val MSE, label units (log)",
                           note="five sizes, 10 epochs each: a fixed-steps slice, speed per epoch "
                                "as much as capacity until the small sizes run to convergence",
                           note_lines=3)
    xs = xticks(c, ax, ay, aw, ah, ["45x2", "63x3", "90x4", "126x6", "180x8"])
    yN = [ay + 22, ay + 40, ay + 56, ay + 68, ay + 76]
    c.line(list(zip(xs, yN)), OR, sw=1.5, curved=True)
    for xx, yy in zip(xs, yN):
        c.dot(xx, yy, OR)
    c.dot(xs[4], yN[4], BK, r=5.5, fill="none", sw=0.75)
    c.line([(ax + 2, ay + ah - 24), (ax + aw - 2, ay + ah - 24)], GR, sw=0.75, dashed=True)
    c.text(ax + 4, ay + ah - 21, 120, 11, "plateau E + B D^-beta", color=GR)
    c.text(ax + aw - 110, ay + 2, 108, 24, "slope = alpha; CI90 from\nthe bootstrap over runs",
           align="right")
    # c: the gap
    ax, ay, aw, ah = panel(c, X[2], TOP + 16, PW, PH, "What keeping every measurement buys",
                           "D, training records (log)", "030 minus 025 pooled MSE",
                           note="seed mean and sd at 1/64 and 1/16 (3 seeds each side); "
                                "the full point is two unconverged 10-epoch runs, drawn hollow",
                           note_lines=3)
    xs = xticks(c, ax, ay, aw, ah, ["1/64", "1/16", "1/4", "full"])
    zero = ay + ah * 0.45
    c.line([(ax + 2, zero), (ax + aw - 2, zero)], GR, sw=0.75, dashed=True)
    c.text(ax + aw - 40, zero - 16, 38, 11, "0", color=GR, align="right")
    yg = [zero + 26, zero + 22, zero + 16, zero + 12]
    for i, (xx, yy) in enumerate(zip(xs, yg)):
        if i < 2:
            c.line([(xx, yy - 10), (xx, yy + 10)], OR, sw=1.0)
            c.dot(xx, yy, OR)
        elif i == 2:
            c.dot(xx, yy, OR)
        else:
            c.dot(xx, yy, OR, fill="#FFFFFF")
    c.text(ax + 4, ay + 4, 150, 24, "below 0: the per-entry rows and the\ntoken lower the loss "
           "at that D")
    for x, s in zip(X, "abc"):
        c.letter(x, TOP, s)
    return c.write(TOP + 16 + PH + 4)


# -------------------------------------------------------- figure 4: how the fits are read
def fig_reading() -> str:
    c = Canvas("030-scaling-reading")
    c.letter(0, 0, "d")
    c.text(16, -1, 600, 14, "Reading the three fits: the question each one answers and what "
           "each outcome says to do next", bold=True, size=FT)
    CW = (W - 2 * 14) / 3
    X = [0, CW + 14, 2 * (CW + 14)]
    y0 = 20
    QH, OH = 40, 58
    cols = [
        ("Is the loss still falling with D at the full pool, on the matched read?",
         [("yes", "a new screen beats a larger model: a doubling of distinct records buys "
                  "2^-beta of the reducible loss, and beta with its interval is the number "
                  "to carry", ORF, OR),
          ("no", "data saturates at this size: the curve has met the plateau E + A N^-alpha, "
                 "and the next unit of effort goes to N or to the floor", PUF, PU)]),
        ("Is the loss still falling with N at the full pool?",
         [("yes", "capacity binds: the next size pays, and alpha says how much per doubling; "
                  "check that lr was right for the top of the ladder first", PUF, PU),
          ("no", "the sizes sit on the plateau E + B D^-beta: more data before more "
                 "parameters, and the N exponent is descriptive, not a forecast", ORF, OR)]),
        ("Where does the fitted E sit against the replicate floor of each screen?",
         [("below", "the fit is wrong: no predictor beats the replicate noise, so E and the "
                    "exponent traded off on too few points", RDF, RD),
          ("near", "the model family is near the noise limit on that screen; the remaining "
                   "loss is measurement, not modeling", GRF, GR),
          ("far above", "the model family is the limit, not the data: a different inductive "
                        "bias before a bigger grid", YEF, YE)]),
    ]
    maxh = 0
    for (q, outs), x in zip(cols, X):
        c.rect(x, y0, CW, QH, fill=YEF, stroke=YE, sw=0.6, label=q, bold=True)
        yy = y0 + QH + 12
        for tag, s, fill, stroke in outs:
            c.line([(x + CW / 2, yy - 12), (x + CW / 2, yy)], GR, sw=1.0, arrow=True)
            c.rect(x, yy, 46, OH, fill=fill, stroke=stroke, sw=0.6, label=tag, bold=True)
            c.rect(x + 46, yy, CW - 46, OH, fill="#FFFFFF", stroke=stroke, sw=0.6, label=s,
                   align="left")
            yy += OH + 12
        maxh = max(maxh, yy)
    y1 = maxh + 2
    c.rect(0, y1, W, 30, fill=GRF, stroke=GR, sw=0.6, align="left",
           label="gate on every fit: the largest point is held out and predicted from the rest. "
                 "Predicted within the bootstrap interval, the exponent extrapolates one step; "
                 "missed, it describes the points it was fit on and nothing beyond them. The "
                 "floored form needs four distinct x in sample for this test.")
    return c.write(y1 + 36)


def main() -> None:
    with open(TABLE_JSON) as f:
        t = json.load(f)
    for path in (fig_grid(t), fig_readout(), fig_results_plan(), fig_reading()):
        print("wrote", path)


if __name__ == "__main__":
    main()
