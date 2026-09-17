"""Regenerate notes/assets/drawio/028-wl-test.drawio at print size (717 units = 182 mm).

Two six-node graphs G and G' (topologies and node placements taken from the MATH-490
figure), 1-WL color refinement at k = 0 and k = 1 with the hash tables, the count
vectors and the WL subtree kernel values. Fonts on the ladder only: 8.3 default, 9.7 for
row titles and formulas, 11.1 bold for panel letters.
"""

from __future__ import annotations

import sys
from xml.sax.saxutils import escape

OUT = sys.argv[1]

W = 717
LETTER_W = 14
GAP = 11
COL_W = (W - LETTER_W - GAP) // 2  # 346
LX = LETTER_W
RX = LETTER_W + COL_W + GAP

FS = "8.3"
FS_T = "9.7"
FS_L = "11.1"

COLOR = {
    2: ("#FFE6CC", "#D79B00"),
    3: ("#F8CECC", "#B85450"),
    4: ("#E1D5E7", "#9673A6"),
    5: ("#DAE8FC", "#6C8EBF"),
    6: ("#FFF2CC", "#D6B656"),
    7: ("#F5F5F5", "#666666"),
    8: ("#B9CDEA", "#4F688B"),
    9: ("#CAB8D4", "#6E5479"),
    10: ("#F3D08F", "#9D7100"),
    11: ("#CACACA", "#4A4A4A"),
    12: ("#F3E0A9", "#9C853F"),
    13: ("#F9D0BE", "#C48161"),
    14: ("#E5A9A7", "#863D3A"),
    15: ("#EFD2CF", "#9A6B66"),
}

# Normalized node positions (x, y in [0, 1]) and edges, per graph.
G_NODES = {
    "a": (0.00, 0.14),
    "b": (0.24, 1.00),
    "c": (1.00, 0.86),
    "d": (1.00, 0.22),
    "e": (0.59, 0.51),
    "f": (0.58, 0.00),
}
G_EDGES = [("b", "c"), ("b", "e"), ("c", "e"), ("a", "e"), ("d", "c"), ("d", "f")]
GP_NODES = {
    "a": (0.00, 0.23),
    "b": (0.07, 1.00),
    "c": (1.00, 0.85),
    "d": (1.00, 0.15),
    "e": (0.50, 0.53),
    "f": (0.50, 0.00),
}
GP_EDGES = [("b", "c"), ("b", "e"), ("c", "e"), ("a", "e"), ("d", "c"), ("f", "e")]

# Labels per round. k0 multiset = own color and neighbor colors at k = 0 (all 1).
G_K0_SET = {"a": "1,1", "b": "1,11", "c": "1,111", "d": "1,11", "e": "1,111", "f": "1,1"}
G_K0_HASH = {"a": 2, "b": 3, "c": 4, "d": 3, "e": 4, "f": 2}
G_K1_SET = {"a": "2,4", "b": "3,44", "c": "4,334", "d": "3,24", "e": "4,234", "f": "2,3"}
G_K1_HASH = {"a": 7, "b": 10, "c": 14, "d": 9, "e": 12, "f": 6}
GP_K0_SET = {"a": "1,1", "b": "1,11", "c": "1,111", "d": "1,1", "e": "1,1111", "f": "1,1"}
GP_K0_HASH = {"a": 2, "b": 3, "c": 4, "d": 2, "e": 5, "f": 2}
GP_K1_SET = {"a": "2,5", "b": "3,45", "c": "4,235", "d": "2,4", "e": "5,2234", "f": "2,5"}
GP_K1_HASH = {"a": 8, "b": 11, "c": 13, "d": 7, "e": 15, "f": 8}

K0_TABLE = [("{1,{1}}", 2), ("{1,{1,1}}", 3), ("{1,{1,1,1}}", 4), ("{1,{1,1,1,1}}", 5)]
K1_TABLE = [
    ("{2,{3}}", 6),
    ("{2,{4}}", 7),
    ("{2,{5}}", 8),
    ("{3,{2,4}}", 9),
    ("{3,{4,4}}", 10),
    ("{3,{4,5}}", 11),
    ("{4,{2,3,4}}", 12),
    ("{4,{2,3,5}}", 13),
    ("{4,{3,3,4}}", 14),
    ("{5,{2,2,3,4}}", 15),
]


def counts(hashes: dict[str, int], table: list[tuple[str, int]]) -> list[int]:
    return [sum(1 for h in hashes.values() if h == code) for _, code in table]


G_K0_COUNT = counts(G_K0_HASH, K0_TABLE)
GP_K0_COUNT = counts(GP_K0_HASH, K0_TABLE)
G_K1_COUNT = counts(G_K1_HASH, K1_TABLE)
GP_K1_COUNT = counts(GP_K1_HASH, K1_TABLE)
PHI_G = [6] + G_K0_COUNT + G_K1_COUNT
PHI_GP = [6] + GP_K0_COUNT + GP_K1_COUNT


def dot(u: list[int], v: list[int]) -> int:
    return sum(a * b for a, b in zip(u, v))


cells: list[str] = []
_id = [10]


def nid() -> str:
    _id[0] += 1
    return str(_id[0])


def vertex(value: str, style: str, x: float, y: float, w: float, h: float) -> str:
    i = nid()
    cells.append(
        f'<mxCell id="{i}" value="{escape(value, {chr(34): "&quot;"})}" style="{style}" '
        f'vertex="1" parent="1"><mxGeometry x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" '
        f'height="{h:.2f}" as="geometry"/></mxCell>'
    )
    return i


def edge(src: str, dst: str) -> None:
    i = nid()
    cells.append(
        f'<mxCell id="{i}" value="" style="endArrow=none;html=1;strokeColor=#000000;'
        f'strokeWidth=0.8;" edge="1" parent="1" source="{src}" target="{dst}">'
        f'<mxGeometry relative="1" as="geometry"/></mxCell>'
    )


BOX = "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#666666;strokeWidth=0.8;fontFamily=Arial;fontSize=8.3;"
TXT = (
    "text;html=1;align={align};verticalAlign=middle;fontFamily=Arial;fontSize={fs};"
    "strokeColor=none;fillColor=none;whiteSpace=wrap;{extra}"
)
CELL = (
    "rounded=0;whiteSpace=wrap;html=1;fontFamily=Arial;fontSize=8.3;fillColor={fill};"
    "strokeColor={stroke};strokeWidth=0.6;align=center;"
)
NODE = (
    "ellipse;whiteSpace=wrap;html=1;fontFamily=Arial;fontSize=8.3;fillColor={fill};"
    "strokeColor={stroke};strokeWidth=0.8;"
)


def text(value: str, x: float, y: float, w: float, h: float, fs: str = FS, align: str = "center", extra: str = "") -> str:
    return vertex(value, TXT.format(align=align, fs=fs, extra=extra), x, y, w, h)


def letter(ch: str, y: float) -> None:
    text(ch, 0, y - 3, LETTER_W, 14, fs=FS_L, align="left", extra="fontStyle=1;")


def node_size(label: str) -> tuple[float, float]:
    n = len(label)
    if n <= 1:
        return 20, 20
    return {3: 26, 4: 31, 5: 37, 6: 43}.get(n, 26 + 5.5 * (n - 3)), 18


def graph(
    nodes: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
    labels: dict[str, str],
    colors: dict[str, int] | None,
    x: float,
    y: float,
    gw: float,
    gh: float,
) -> None:
    ids: dict[str, str] = {}
    for k, (nx, ny) in nodes.items():
        lab = labels[k]
        w, h = node_size(lab)
        cx, cy = x + w / 2 + nx * (gw - w), y + h / 2 + ny * (gh - h)
        if colors is not None:
            fill, stroke = COLOR[colors[k]]
        else:
            fill, stroke = "#FFFFFF", "#000000"
        ids[k] = vertex(lab, NODE.format(fill=fill, stroke=stroke), cx - w / 2, cy - h / 2, w, h)
    for a, b in edges:
        edge(ids[a], ids[b])


def table(rows: list[tuple[str, int, int]], x: float, y: float, widths: tuple[float, float, float] = (60, 40, 30), rh: float = 12) -> float:
    """Header + rows; returns bottom y."""
    hdr = ["<i>&beta;</i>", "HASH(<i>&beta;</i>)", "Count"]
    cx = x
    for text_, w in zip(hdr, widths):
        vertex(text_, CELL.format(fill="#FFFFFF", stroke="#000000"), cx, y, w, rh)
        cx += w
    yy = y + rh
    for beta, code, count in rows:
        cx = x
        fill, stroke = COLOR.get(code, ("#FFFFFF", "#000000"))
        vals = [(beta, "#FFFFFF", "#000000"), (str(code), fill, stroke), (str(count), "#FFFFFF", "#000000")]
        for (v, f, s), w in zip(vals, widths):
            vertex(v, CELL.format(fill=f, stroke=s), cx, yy, w, rh)
            cx += w
        yy += rh
    return yy


def rowbox(x: float, y: float, w: float, h: float) -> None:
    vertex("", BOX, x, y, w, h)


# ---------------- row a: the pair and the refinement rule ----------------
Y_A, H_A = 0, 96
letter("a", Y_A)
for bx, nodes, edges, name in ((LX, G_NODES, G_EDGES, "<i>G</i>"), (RX, GP_NODES, GP_EDGES, "<i>G&prime;</i>")):
    rowbox(bx, Y_A, COL_W, H_A)
    text(name, bx + 4, Y_A + 3, 20, 12, fs=FS_T, align="left")
    graph(nodes, edges, {k: "1" for k in nodes}, None, bx + 26, Y_A + 12, 96, 72)
    text(
        "$$\\beta=\\left\\{c^{(k)}(v),\\left\\{c^{(k)}(u)\\right\\}_{u\\in N(v)}\\right\\}$$",
        bx + 138, Y_A + 6, 200, 22, fs=FS_T,
    )
    text("initial color <i>c</i><sup>(0)</sup>(<i>v</i>) = 1 for every node", bx + 138, Y_A + 30, 200, 12)
    table([("{1}", 1, 6)], bx + 172, Y_A + 50, widths=(50, 40, 32))

# ---------------- row b: k = 0 ----------------
Y_B, H_B = Y_A + H_A + 8, 118
letter("b", Y_B)
for bx, nodes, edges, sets, hashes, cnt in (
    (LX, G_NODES, G_EDGES, G_K0_SET, G_K0_HASH, G_K0_COUNT),
    (RX, GP_NODES, GP_EDGES, GP_K0_SET, GP_K0_HASH, GP_K0_COUNT),
):
    rowbox(bx, Y_B, COL_W, H_B)
    text("<i>k</i> = 0", bx, Y_B + 3, COL_W, 12, fs=FS_T)
    graph(nodes, edges, sets, None, bx + 6, Y_B + 20, 96, 88)
    text("&rarr;", bx + 104, Y_B + 56, 12, 12, fs=FS_T)
    graph(nodes, edges, {k: str(v) for k, v in hashes.items()}, hashes, bx + 122, Y_B + 22, 82, 84)
    table([(b, c, n) for (b, c), n in zip(K0_TABLE, cnt)], bx + 214, Y_B + 28, widths=(60, 40, 30))

# ---------------- row c: k = 1 ----------------
Y_C, H_C = Y_B + H_B + 8, 156
letter("c", Y_C)
for bx, nodes, edges, sets, hashes, cnt in (
    (LX, G_NODES, G_EDGES, G_K1_SET, G_K1_HASH, G_K1_COUNT),
    (RX, GP_NODES, GP_EDGES, GP_K1_SET, GP_K1_HASH, GP_K1_COUNT),
):
    rowbox(bx, Y_C, COL_W, H_C)
    text("<i>k</i> = 1", bx, Y_C + 3, COL_W, 12, fs=FS_T)
    graph(nodes, edges, sets, None, bx + 6, Y_C + 30, 100, 100)
    text("&rarr;", bx + 106, Y_C + 74, 12, 12, fs=FS_T)
    graph(nodes, edges, {k: str(v) for k, v in hashes.items()}, hashes, bx + 122, Y_C + 36, 82, 88)
    table([(b, c, n) for (b, c), n in zip(K1_TABLE, cnt)], bx + 214, Y_C + 18, widths=(60, 40, 30))

# ---------------- rows d, e, f ----------------
def formula_row(ch: str, y: float, title: str, left: str, right: str | None) -> float:
    h = 40
    letter(ch, y)
    if right is None:
        rowbox(LX, y, W - LX, h)
        text(title, LX, y + 2, W - LX, 12, fs=FS_T)
        text(left, LX, y + 15, W - LX, 22, fs=FS_T)
    else:
        for bx, f in ((LX, left), (RX, right)):
            rowbox(bx, y, COL_W, h)
            text(title, bx, y + 2, COL_W, 12, fs=FS_T)
            text(f, bx, y + 15, COL_W, 22, fs=FS_T)
    return y + h


def vec(v: list[int]) -> str:
    return "[" + ",".join(str(i) for i in v) + "]"


Y_D = Y_C + H_C + 8
Y_E = formula_row("d", Y_D, "Count vector over the colors of rounds 0 and 1",
                  f"$$\\phi(G)={vec(PHI_G)}$$", f"$$\\phi(G')={vec(PHI_GP)}$$") + 8
Y_F = formula_row("e", Y_E, "WL subtree kernel of each graph with itself",
                  f"$$\\phi(G)^{{\\top}}\\phi(G)={dot(PHI_G, PHI_G)}$$",
                  f"$$\\phi(G')^{{\\top}}\\phi(G')={dot(PHI_GP, PHI_GP)}$$") + 8
Y_END = formula_row("f", Y_F, "WL test: the count vectors differ after one round, so the graphs are not isomorphic",
                    f"$$\\phi(G)^{{\\top}}\\phi(G')={dot(PHI_G, PHI_GP)}$$", None)

xml = (
    '<mxfile><diagram id="wl" name="Page-1"><mxGraphModel dx="0" dy="0" grid="1" '
    'gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" '
    f'pageScale="1" pageWidth="{W}" pageHeight="{int(Y_END)}" math="1" shadow="0">'
    '<root><mxCell id="0"/><mxCell id="1" parent="0"/>'
    + "".join(cells)
    + "</root></mxGraphModel></diagram></mxfile>"
)
with open(OUT, "w") as fh:
    fh.write(xml)
print(f"canvas {W} x {Y_END:.0f} units = {W*0.254:.0f} x {Y_END*0.254:.0f} mm; phi(G)={PHI_G} phi(G')={PHI_GP}; "
      f"kernels {dot(PHI_G, PHI_G)} {dot(PHI_GP, PHI_GP)} {dot(PHI_G, PHI_GP)}")
