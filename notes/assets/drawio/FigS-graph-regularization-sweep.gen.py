"""Generate the mock-up FigS-graph-regularization-sweep.drawio (no data; sketched panels).

Run from anywhere; writes the .drawio next to itself. Export with the recipe in
[[drawio-headless-export-gilahyper]] (input file FIRST, then the Electron flags):
  xvfb-run -a squashfs-root/drawio FigS-graph-regularization-sweep.drawio \
      --no-sandbox --disable-gpu -x -f png -e -s 4 -o FigS-graph-regularization-sweep.drawio.png
"""
import os.path as osp
from xml.sax.saxutils import escape

OUT = osp.join(osp.dirname(osp.abspath(__file__)), "FigS-graph-regularization-sweep.drawio")

OR, ORF = "#D79B00", "#FFE6CC"
RD, RDF = "#B85450", "#F8CECC"
PU, PUF = "#9673A6", "#E1D5E7"
BL, BLF = "#6C8EBF", "#DAE8FC"
GR, GRF = "#666666", "#F5F5F5"

# grid, in draw.io units (100 per inch). Full-width target 702.
W = 702
COL_GAP, ROW_GAP, TOP_STRIP = 12, 24, 16
PW = (W - 2 * COL_GAP) / 3  # 226
PH = 166
PH1 = 240  # row 1 carries a definition note under each panel
X = [0, PW + COL_GAP, 2 * (PW + COL_GAP)]
ROW0 = 0
ROW1 = ROW0 + TOP_STRIP + PH + ROW_GAP
TAB0 = ROW1 + TOP_STRIP + PH1 + ROW_GAP
FS = 8.5  # 6.1 pt

cells = []
_n = [0]


def nid():
    _n[0] += 1
    return f"c{_n[0]}"


def text(x, y, w, h, s, size=FS, bold=False, align="left", valign="top", rot=0, color="#000000", italic=False):
    st = (f"text;html=1;strokeColor=none;fillColor=none;align={align};verticalAlign={valign};"
          f"whiteSpace=wrap;rounded=0;fontFamily=Arial;fontSize={size};fontColor={color};")
    if bold:
        st += "fontStyle=1;"
    if italic:
        st += "fontStyle=2;"
    if rot:
        st += f"rotation={rot};"
    cells.append(f'<mxCell id="{nid()}" value="{escape(s)}" style="{st}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/></mxCell>')


def rect(x, y, w, h, fill="#FFFFFF", stroke="#000000", sw=0.75, label="", dashed=False, size=FS):
    st = (f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth={sw};"
          f"fontFamily=Arial;fontSize={size};")
    if dashed:
        st += "dashed=1;"
    cells.append(f'<mxCell id="{nid()}" value="{escape(label)}" style="{st}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/></mxCell>')


def curve(pts, color, sw=1.5, dashed=False, curved=True):
    st = f"endArrow=none;html=1;strokeWidth={sw};strokeColor={color};curved={1 if curved else 0};"
    if dashed:
        st += "dashed=1;"
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    mid = "".join(f'<mxPoint x="{px:.1f}" y="{py:.1f}"/>' for px, py in pts[1:-1])
    cells.append(f'<mxCell id="{nid()}" style="{st}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                 f'<mxPoint x="{x0:.1f}" y="{y0:.1f}" as="sourcePoint"/><mxPoint x="{x1:.1f}" y="{y1:.1f}" as="targetPoint"/>'
                 f'<Array as="points">{mid}</Array></mxGeometry></mxCell>')


def dot(x, y, color, r=2.2, fill=None):
    st = f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor={fill or color};strokeColor={color};strokeWidth=0.75;"
    cells.append(f'<mxCell id="{nid()}" value="" style="{st}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r:.1f}" height="{2 * r:.1f}" as="geometry"/></mxCell>')


def letter(x, y, s):
    text(x, y, 14, 14, s, size=11.1, bold=True)


def panel(px, py, title, xlabel, ylabel, xlab_h=13, note="", note_lines=0, ph=None):
    """Panel frame, title, axes box, axis labels, optional definition note under the
    axes (11 units per line). Returns (ax, ay, aw, ah)."""
    ph = ph or PH
    rect(px, py, PW, ph, fill="#FFFFFF", stroke=GR, sw=0.5)
    text(px + 2, py + 1, PW - 4, 22, title, bold=True)
    note_h = 11 * note_lines + (6 if note_lines else 0)
    ax, ay, aw, ah = px + 28, py + 26, PW - 36, ph - 30 - 14 - xlab_h - 8 - note_h
    rect(ax, ay, aw, ah, fill="#FFFFFF", stroke="#000000", sw=0.75)
    if note:
        text(px + 4, py + ph - note_h - 2, PW - 8, note_h, note, size=8.5)
    # y label: a rotated cell is placed by its center, so center a cell of width ah on x = px + 12.
    text(px + 12 - ah / 2, ay + ah / 2 - 8, ah, 16, ylabel, align="center", valign="middle", rot=-90)
    text(ax, ay + ah + 14, aw, xlab_h, xlabel, align="center")
    return ax, ay, aw, ah


def xticks(ax, ay, aw, ah, labels):
    n = len(labels)
    for i, lab in enumerate(labels):
        xx = ax + aw * (i + 0.5) / n
        curve([(xx, ay + ah), (xx, ay + ah + 3)], "#000000", sw=0.75, curved=False)
        text(xx - 22, ay + ah + 2, 44, 11, lab, align="center")


LAMBDA_TICKS = ["0", "10⁻⁵", "10⁻³", "10⁻¹", "mask"]

# ---------------------------------------------------------------- row 0
# a: the Note's first signature, support contraction
letter(X[0], ROW0, "a")
ax, ay, aw, ah = panel(X[0], ROW0 + TOP_STRIP, "Attention contracts onto the graph as λ grows",
                       "λ (graph prior weight)", "Divergence to Ã")
xticks(ax, ay, aw, ah, LAMBDA_TICKS)
curve([(ax + 10, ay + 12), (ax + aw * 0.3, ay + 40), (ax + aw * 0.55, ay + ah - 30), (ax + aw - 8, ay + ah - 5)], OR)
curve([(ax + 10, ay + 36), (ax + aw * 0.35, ay + 58), (ax + aw * 0.6, ay + ah - 22), (ax + aw - 8, ay + ah - 5)], BL, dashed=True)
text(ax + aw * 0.40, ay + 4, 108, 24, "D_KL(Ã || α), orange\noff-graph attention mass, blue", size=FS)
text(ax + aw - 86, ay + ah - 46, 84, 22, "0 at the mask:\nthe Prop. limit", align="right")

# b: the sweep with the two extremes
letter(X[1], ROW0, "b")
ax, ay, aw, ah = panel(X[1], ROW0 + TOP_STRIP, "Accuracy peaks between the two extremes",
                       "λ (graph prior weight)", "Held-out Pearson")
xticks(ax, ay, aw, ah, LAMBDA_TICKS)
n = 5
xs = [ax + aw * (i + 0.5) / n for i in range(n)]
ys = [ay + ah - 16, ay + 44, ay + 22, ay + 36, ay + ah - 14]
curve(list(zip(xs, ys)), OR)
for xx, yy in zip(xs, ys):
    for k in (-1, 0, 1):
        dot(xx + 4 * k, yy + (3 if k else -3), OR, r=2.0, fill="#FFFFFF")
rect(xs[0] - 24, ay + 4, 48, 14, fill=GRF, stroke=GR, sw=0.5, label="no penalty")
rect(xs[4] - 24, ay + 4, 48, 14, fill=RDF, stroke=RD, sw=0.5, label="hard mask")
text(ax + 2, ay + ah - 62, 74, 34, "measured, 010:\n0.03 to 0.10 (n = 3)")
text(ax + aw - 78, ay + ah - 62, 76, 34, "measured, 025:\n0.31 at epoch 1,\nthen 0 (n = 1)", align="right")
text(ax + aw * 0.5 - 50, ay + ah - 30, 100, 24, "measured, 010:\n0.456 to 0.464 (n = 3)", align="center")

# c: the mechanism, gradient budget at key epochs
letter(X[2], ROW0, "c")
ax, ay, aw, ah = panel(X[2], ROW0 + TOP_STRIP, "Where the gradient comes from, by epoch",
                       "Epoch (probe batch at 0, 1, 2, 5, 10, 20)", "Gradient norm, log scale",
                       note="hypothesis: the penalty carries the gradient before the point loss can", note_lines=2)
xticks(ax, ay, aw, ah, ["0", "1", "2", "5", "10", "20"])
n = 6
xs = [ax + aw * (i + 0.5) / n for i in range(n)]
f = lambda q: ay + ah * q
curve(list(zip(xs, [f(0.42), f(0.48), f(0.58), f(0.70), f(0.80), f(0.86)])), OR)
curve(list(zip(xs, [f(0.86), f(0.80), f(0.70), f(0.60), f(0.56), f(0.54)])), OR, dashed=True)
curve(list(zip(xs, [f(0.86), f(0.85), f(0.84), f(0.84), f(0.84), f(0.84)])), GR, dashed=True)
text(ax + 4, ay + 2, aw - 8, 12, "∇ graph penalty, soft KL (solid)", color=OR)
text(ax + 4, ay + 13, aw - 8, 12, "∇ point loss, soft KL (dashed)", color=OR)
text(ax + 4, ay + 24, aw - 8, 12, "∇ point loss, λ = 0 or hard mask", color=GR)

# ---------------------------------------------------------------- row 1
# d: discovery, defined on graphs already in hand
letter(X[0], ROW1, "d")
ax, ay, aw, ah = panel(X[0], ROW1 + TOP_STRIP, "Discovery: off-graph attention recovers unseen edges",
                       "Top-k off-graph attention entries", "Recall of unseen edges", ph=PH1,
                       note="unseen edges: edges of an independent graph absent from the head's prior, "
                            "e.g. the regulatory head scored on TFLink-only edges, the physical head on "
                            "STRING-experimental-only edges. Null: degree-matched random edges. "
                            "The hard mask has no off-graph mass, so no curve.", note_lines=6)
xticks(ax, ay, aw, ah, ["10²", "10³", "10⁴", "10⁵"])
curve([(ax + 6, ay + ah - 14), (ax + aw * 0.35, ay + ah - 40), (ax + aw * 0.7, ay + 34), (ax + aw - 6, ay + 14)], OR)
curve([(ax + 6, ay + ah - 12), (ax + aw * 0.5, ay + ah - 26), (ax + aw - 6, ay + ah - 40)], BL, dashed=True)
curve([(ax + 6, ay + ah - 8), (ax + 18, ay + ah - 8)], RD, sw=2.5, curved=False)
text(ax + 4, ay + 2, 150, 12, "soft prior at best λ", color=OR)
text(ax + 4, ay + 13, 150, 12, "random edges, degree-matched", color=BL)
text(ax + 4, ay + 24, 150, 12, "hard mask", color=RD)

# e: overruling, defined with measured digenic interactions
letter(X[1], ROW1, "e")
ax, ay, aw, ah = panel(X[1], ROW1 + TOP_STRIP, "Overruling: which catalogued edges the head keeps",
                       "Attention on the edge relative to its prior 1/dᵢ", "Share with |ε| > 0.08", ph=PH1,
                       note="for every prior edge (i, j) of a graph: the trained head's α_ij divided by the "
                            "uniform prior 1/dᵢ (down-weighted at left, kept at right), binned, against the "
                            "Costanzo digenic ε of the same pair. Dashed: share over all edges of the graph. "
                            "Prediction: the head down-weights edges with no measured interaction.", note_lines=6)
xticks(ax, ay, aw, ah, ["< 0.1", "0.1 to 0.5", "0.5 to 2", "> 2"])
n = 4
xs = [ax + aw * (i + 0.5) / n for i in range(n)]
hs = [ah * 0.18, ah * 0.30, ah * 0.48, ah * 0.66]
bw = aw * 0.16
for xx, hh in zip(xs, hs):
    rect(xx - bw / 2, ay + ah - 6 - hh, bw, hh, fill=ORF, stroke=OR)
curve([(ax + 4, ay + ah - 6 - ah * 0.34), (ax + aw - 4, ay + ah - 6 - ah * 0.34)], GR, sw=0.75, dashed=True, curved=False)
text(ax + 4, ay + 3, 100, 12, "all edges (dashed)", color=GR)

# f: the control that separates what from how
letter(X[2], ROW1, "f")
ax, ay, aw, ah = panel(X[2], ROW1 + TOP_STRIP, "Is it the biology or any target? Random-graph control",
                       "", "Held-out Pearson at best λ", ph=PH1,
                       note="purple ≈ orange: the penalty conditions training (how). purple ≈ gray: the edges "
                            "carry information (what). Random graphs: each of the nine graphs rewired with its "
                            "degree sequence kept (configuration model), one rewiring per seed.", note_lines=6)
xs = [ax + aw * 0.2, ax + aw * 0.5, ax + aw * 0.8]
hs = [ah * 0.62, ah * 0.40, ah * 0.08]
cols = [(OR, ORF), (PU, PUF), (GR, GRF)]
labs = ["biological", "random,\ndegree-matched", "none (λ = 0)"]
for xx, hh, (s_, f_), lab in zip(xs, hs, cols, labs):
    rect(xx - bw / 2, ay + ah - 8 - hh, bw, hh, fill=f_, stroke=s_, dashed=(s_ == PU))
    text(xx - 34, ay + ah + 2, 68, 24, lab, align="center")
text(xs[1] - 34, ay + ah - 8 - hs[1] - 18, 68, 12, "? not run", align="center", color=PU, italic=True)

# ---------------------------------------------------------------- design strip
text(X[0], TAB0, 460, 14, "Experiment design (Delta, gpuA40x4, 4-GPU DDP, one node per run)", bold=True)
text(W - 220, TAB0, 220, 14, "MOCK-UP: sketched curves, not data", bold=True, align="right", color=RD)
rows = [
    ["Arm", "λ", "Graph target", "Seeds", "Runs", "Wall clock", "Panels"],
    ["no penalty", "0", "none", "3", "3", "12 h", "b, c, f"],
    ["λ sweep", "10⁻⁵, 10⁻⁴, 10⁻³, 10⁻², 10⁻¹", "9 biological graphs, layer 1", "3 each", "15", "24 h", "a, b, c, d, e"],
    ["hard mask", "∞ (mask at layer 1)", "9 biological graphs", "3", "3", "12 h", "b, c, d"],
    ["random-graph control", "best λ from the sweep", "degree-matched random", "3", "3", "24 h", "f"],
    ["total", "", "", "", "24", "", "2,016 GPU-h"],
]
cw = [108, 160, 160, 48, 40, 60, 90]
ty = TAB0 + 16
rh = 15
for r, row in enumerate(rows):
    xx = X[0]
    for c, val in enumerate(row):
        rect(xx, ty + r * rh, cw[c], rh, fill="#F5F5F5" if r == 0 else "#FFFFFF", stroke=GR, sw=0.5, label=val)
        xx += cw[c]
text(X[0], ty + len(rows) * rh + 4, W, 36,
     "Fixed across arms: experiment-025 S0 (the 376,732 records of 010), the pinned 010 random split, 010's "
     "schedule (batch 256, bf16, OneCycle to 5e-4), one regularized layer (layer 1, all nine heads), each run read "
     "at its best-validation checkpoint. GPU-hours charge per GPU: 4 x wall clock per run. Anchors already "
     "measured: 010 λ = 0 and λ = 10⁻³ (x367 coefficient), 025 soft KL and hard mask. Panels d and e can be computed "
     "today on the three 010 checkpoints, before any run.")

xml = f'''<mxfile host="app.diagrams.net" agent="claude-code">
  <diagram name="FigS-graph-regularization-sweep" id="FigS-graph-regularization-sweep">
    <mxGraphModel dx="1400" dy="1000" grid="0" page="1" pageWidth="{W}" pageHeight="{int(ty + len(rows) * rh + 50)}" math="0" shadow="0">
      <root>
        <mxCell id="0"/><mxCell id="1" parent="0"/>
        {chr(10).join(cells)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''
open(OUT, "w").write(xml)
print("wrote", OUT, "cells", len(cells), "height ~", ty + len(rows) * rh + 44)
