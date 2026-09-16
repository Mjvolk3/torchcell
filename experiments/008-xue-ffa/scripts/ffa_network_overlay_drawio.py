# experiments/008-xue-ffa/scripts/ffa_network_overlay_drawio.py
# [[experiments.008-xue-ffa.scripts.ffa_network_overlay_drawio]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/ffa_network_overlay_drawio
#
# The network overlay as a NATIVE draw.io figure: every factor, gene, reaction, metabolite
# and edge is its own mxCell, so the figure can be rearranged, relabeled and annotated in
# draw.io the way the manuscript's Fig. 1 is, with edges that follow the shapes they join.
# The embedded-SVG form that build_perspective_drawio.py writes for the other figures is a
# picture inside a box; this one is the diagram itself.
#
# Geometry comes from ffa_network_overlay_panel.layout(), so the two renderings place every
# node at the same millimetre. Units: draw.io's canvas is 100 per inch, 3.9392 per mm; the
# Nature full-page box is 179.4 x 170 mm, and the content here is 178 x 118 mm so the
# 1-unit export border per side stays inside the cap. y is flipped (draw.io grows downward).
#
# Style follows the draw.io house rules ([[paper.nature-biotech.style-guide]]): Arial,
# fontSize 8.3 (5.98 pt) for labels and 11.1 for a panel letter, palette stroke/fill pairs
# (purple = deleted factor, yellow = pathway gene, amber = measured species, gray =
# reaction and intermediate), blue and brick reserved for the positive and negative
# interaction edges. Line widths are in canvas units, 1 unit = 0.72 pt, so a 0.4 pt line
# is strokeWidth 0.56.
#
# EDGES ARE ROUTED, NOT ATTACHED. A pair in both a negative and a positive interaction
# gets its two edges side by side (ffa_network_overlay_panel.interaction_segments), which
# an attached draw.io connector cannot do: it always runs center to center. So every
# interaction edge is drawn through explicit points from node center to node center, and
# the node shapes, added after the edges, cover the ends. Moving a factor box in draw.io
# therefore does NOT move its edges; regenerate instead.
#
# THREE RINGS, ONE ORIENTATION. The labeled ring in the middle carries both signs; above
# and below it the same ten factors are drawn again at the same angles with one sign each
# and no labels, so the shape a sign makes can be read on its own. An earlier review view
# drew each triple as a translucent filled triangle instead; 75 overlapping negatives came
# out as one lens-shaped blob with no structure in it, and it was dropped.
#
# A hidden layer named "print box" carries the 179.4 x 170 mm frame. Toggle it on in
# draw.io to see the cap while arranging; hidden layers are not exported.

import argparse
import os
import os.path as osp
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

from dotenv import load_dotenv

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import ffa_network_overlay_panel as panel  # noqa: E402

load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
DRAWIO_DIR = osp.join(osp.dirname(ASSET_IMAGES_DIR), "drawio")

U = 706.6915 / 179.4  # canvas units per mm
PT = 1 / 0.72  # canvas units per point
FONT = "8.3"
LETTER_FONT = "11.1"

STROKE = {"purple": "#9673A6", "yellow": "#D6B656", "amber": "#D79B00", "gray": "#666666"}
FILL = {"purple": "#E1D5E7", "yellow": "#FFF2CC", "amber": "#FFE6CC", "gray": "#F5F5F5"}
C_POS, C_NEG = panel.C_POS, panel.C_NEG
C_REG = "#666666"
C_PATH = "#BBBBBB"


def ux(x_mm):
    return x_mm * U


def uy(y_mm):
    return (panel.H_MM - y_mm) * U


class Doc:
    """An mxGraph document: one diagram, cells appended in draw order."""

    def __init__(self, name):
        """Start a diagram named ``name`` on a Nature full-page canvas."""
        self.mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "agent": "torchcell"})
        diagram = ET.SubElement(self.mxfile, "diagram", {"name": name, "id": name})
        model = ET.SubElement(diagram, "mxGraphModel", {
            "dx": "1400", "dy": "1000", "grid": "0", "gridSize": "10", "guides": "1",
            "tooltips": "1", "connect": "1", "arrows": "1", "fold": "1", "page": "1",
            "pageScale": "1", "pageWidth": f"{179.4 * U:.0f}", "pageHeight": f"{170 * U:.0f}",
            "math": "0", "shadow": "0",
        })
        self.root = ET.SubElement(model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})
        self.n = 0

    def layer(self, cid, name, visible=True):
        """Add a layer; a hidden one is visible in the GUI toggle and absent from exports."""
        attrs = {"id": cid, "value": name, "style": "", "parent": "0"}
        if not visible:
            attrs["visible"] = "0"
        ET.SubElement(self.root, "mxCell", attrs)

    def vertex(self, cid, value, style, x, y, w, h, parent="1"):
        """Add a shape at (x, y) with size (w, h), all in canvas units."""
        c = ET.SubElement(self.root, "mxCell", {
            "id": cid, "value": value, "style": style, "vertex": "1", "parent": parent})
        ET.SubElement(c, "mxGeometry", {
            "x": f"{x:.2f}", "y": f"{y:.2f}", "width": f"{w:.2f}", "height": f"{h:.2f}",
            "as": "geometry"})

    def edge(self, cid, style, source=None, target=None, points=None, parent="1"):
        """Add a connector, attached to cell ids or routed through explicit points."""
        attrs = {"id": cid, "value": "", "style": style, "edge": "1", "parent": parent}
        if source is not None:
            attrs["source"] = source
        if target is not None:
            attrs["target"] = target
        c = ET.SubElement(self.root, "mxCell", attrs)
        g = ET.SubElement(c, "mxGeometry", {"relative": "1", "as": "geometry"})
        if points:
            ET.SubElement(g, "mxPoint", {"x": f"{points[0][0]:.2f}",
                                         "y": f"{points[0][1]:.2f}", "as": "sourcePoint"})
            ET.SubElement(g, "mxPoint", {"x": f"{points[-1][0]:.2f}",
                                         "y": f"{points[-1][1]:.2f}", "as": "targetPoint"})
            if len(points) > 2:
                arr = ET.SubElement(g, "Array", {"as": "points"})
                for x, y in points[1:-1]:
                    ET.SubElement(arr, "mxPoint", {"x": f"{x:.2f}", "y": f"{y:.2f}"})

    def write(self, path):
        """Write the indented XML."""
        ET.indent(self.mxfile)
        ET.ElementTree(self.mxfile).write(path, encoding="utf-8", xml_declaration=True)


def node_style(color, shape, extra=""):
    return (f"{shape}whiteSpace=wrap;html=1;fillColor={FILL[color]};"
            f"strokeColor={STROKE[color]};strokeWidth={0.5 * PT:.2f};fontFamily=Arial;"
            f"fontSize={FONT};fontColor=#000000;{extra}")


def text_style(align="center"):
    return (f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align={align};"
            f"verticalAlign=middle;fontFamily=Arial;fontSize={FONT};fontColor=#000000;")


def line_style(color, width_pt, dashed=False, arrow=False):
    s = (f"edgeStyle=none;html=1;strokeColor={color};strokeWidth={width_pt * PT:.2f};"
         f"endArrow={'classic' if arrow else 'none'};startArrow=none;"
         f"endFill=1;endSize=2.5;rounded=0;")
    if dashed:
        s += "dashed=1;dashPattern=1 2;"
    return s


def build(model, readout, graph, letter, out_path):
    L = panel.layout(model, readout, graph)
    pos, G = L["pos"], L["G"]
    doc = Doc(osp.splitext(osp.basename(out_path))[0])

    # --- hidden print box, so the cap is visible in the GUI and absent from the export
    doc.layer("printbox", "print box", visible=False)
    doc.vertex("printbox-rect", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=none;"
               "strokeColor=#B85450;strokeWidth=1;dashed=1;fontFamily=Arial;fontSize=8.3;",
               0, 0, 179.4 * U, 170 * U, parent="printbox")
    doc.vertex("printbox-label", "Nature full page 179.4 x 170 mm (hidden layer)",
               text_style("left") + "fontColor=#B85450;", 4, 170 * U + 2, 300, 12,
               parent="printbox")

    # --- edges first so every shape sits above them
    for i, (u, v, d) in enumerate(G.edges(data=True)):
        if u in pos and v in pos and d.get("edge_type") in ("catalyzes", "consumed_by",
                                                            "produces"):
            doc.edge(f"path{i}", line_style(C_PATH, 0.3), source=f"n-{u}", target=f"n-{v}")

    sys_to_tf = {v: k for k, v in L["tf_sys"].items()}
    for i, (tf, gene) in enumerate(L["reg"]):
        doc.edge(f"reg{i}", line_style(C_REG, 0.5, dashed=True, arrow=True),
                 source=f"n-{sys_to_tf[tf]}", target=f"n-{gene}")

    width = panel.edge_width_pt

    for i, (sign, k, (xa, ya), (xb, yb)) in enumerate(
            panel.interaction_segments(pos, L["neg_mult"], L["pos_mult"])):
        doc.edge(f"{'pos' if sign > 0 else 'neg'}{i}",
                 line_style(C_POS if sign > 0 else C_NEG, width(k)),
                 points=[(ux(xa), uy(ya)), (ux(xb), uy(yb))])

    # the two sign-split rings, above and below the labeled one
    for sign, mult, color in ((+1, L["pos_mult"], C_POS), (-1, L["neg_mult"], C_NEG)):
        tag = "ringpos" if sign > 0 else "ringneg"
        rpos = L["ring_pos"][sign]
        for i, ((a, b), k) in enumerate(sorted(mult.items())):
            doc.edge(f"{tag}e{i}", line_style(color, panel.ring_edge_width_pt(k)),
                     points=[(ux(rpos[a][0]), uy(rpos[a][1])),
                             (ux(rpos[b][0]), uy(rpos[b][1]))])
        dia = 2 * panel.RING_NODE_R * U
        for tf in panel.TF_GENES:
            x, y = rpos[tf]
            doc.vertex(f"{tag}n-{tf}", "",
                       node_style("purple", "ellipse;", "strokeWidth=0.42;"),
                       ux(x) - dia / 2, uy(y) - dia / 2, dia, dia)

    # --- nodes
    side = panel.RXN_SIDE * U
    for r in L["rxns"]:
        x, y = pos[r]
        doc.vertex(f"n-{r}", "", node_style("gray", "rounded=0;", "strokeWidth=0.4;"),
                   ux(x) - side / 2, uy(y) - side / 2, side, side)
    dia = 2 * panel.INT_R * U
    for m in L["intermediates"]:
        x, y = pos[m]
        doc.vertex(f"n-{m}", "", node_style("gray", "ellipse;", "strokeWidth=0.4;"),
                   ux(x) - dia / 2, uy(y) - dia / 2, dia, dia)
    dia = 2 * panel.SPECIES_R * U
    for m in L["species_sorted"]:
        x, y = pos[m]
        doc.vertex(f"n-{m}", "", node_style("amber", "ellipse;"),
                   ux(x) - dia / 2, uy(y) - dia / 2, dia, dia)
    gw, gh = panel.GENE_BOX[0] * U, panel.GENE_BOX[1] * U
    for gene in L["core_sorted"]:
        x, y = pos[gene]
        doc.vertex(f"n-{gene}", L["gene_std"][gene],
                   node_style("yellow", "rounded=1;arcSize=40;"),
                   ux(x) - gw / 2, uy(y) - gh / 2, gw, gh)
    tw, th = panel.TF_BOX[0] * U, panel.TF_BOX[1] * U
    for tf in panel.TF_GENES:
        x, y = pos[tf]
        doc.vertex(f"n-{tf}", tf, node_style("purple", "rounded=1;arcSize=40;"),
                   ux(x) - tw / 2, uy(y) - th / 2, tw, th)

    # --- species brackets and labels
    species, name_of, label_of = L["species_sorted"], L["name_of"], L["label_of"]
    i = 0
    k = 0
    while i < len(species):
        name = name_of[species[i]]
        j = i
        while j + 1 < len(species) and name_of[species[j + 1]] == name:
            j += 1
        y_hi, y_lo = pos[species[i]][1], pos[species[j]][1]
        xb = panel.X_BRACKET
        pts = [(ux(xb), uy(y_hi + 0.9)), (ux(xb + 0.8), uy(y_hi + 0.9)),
               (ux(xb + 0.8), uy(y_lo - 0.9)), (ux(xb), uy(y_lo - 0.9))]
        doc.edge(f"bracket{k}", line_style("#000000", 0.4), points=pts)
        doc.vertex(f"specieslabel{k}", label_of[name], text_style("left"),
                   ux(panel.X_LABEL + 0.8), uy((y_hi + y_lo) / 2) - 7, 22 * U, 14)
        k += 1
        i = j + 1

    # --- column headers
    for k, (x, text) in enumerate(((panel.X_TF, "deleted transcription factors"),
                                   (panel.X_GENE, "pathway genes"),
                                   (panel.X_RXN, "reactions"),
                                   (panel.X_INT, "intermediates"),
                                   (panel.X_SPECIES + 8, "measured species"))):
        doc.vertex(f"header{k}", text, text_style("center"),
                   ux(x) - 20 * U, uy(panel.Y_HEADER) - 7, 40 * U, 14)

    # --- legend: framed, bottom-left, 1 mm up from the canvas edge, which is what leaves
    # the lower sign-split ring clear.
    kmax = max(list(L["neg_mult"].values()) + list(L["pos_mult"].values()))
    rows = [
        (C_NEG, width(1), False, False,
         f"negative trigenic interaction (n = {L['n_neg']})"),
        (C_POS, width(1), False, False,
         f"positive trigenic interaction (n = {L['n_pos']})"),
        # The width row shows both ends of the scale: a 1-interaction line on the
        # left of the swatch and a kmax-interaction line on the right.
        (C_NEG, (width(1), width(kmax)), False, False,
         f"width: interactions per pair, 1 (left) to {kmax} (right)"),
        (C_REG, 0.5, True, True, "factor regulates gene (SGD or TFLink)"),
        (C_PATH, 0.5, False, False, "catalyzes, consumes or produces"),
    ]
    row_h = 3.2 * U
    lx, ly = ux(2.0), uy(1.0 + 3.2 * len(rows) + 2.0)
    lw = 62 * U
    lh = row_h * len(rows) + 2 * U
    doc.vertex("legend-frame", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;"
               f"strokeColor=#000000;strokeWidth={0.5 * PT:.2f};", lx, ly, lw, lh)
    for k, (color, wpt, dashed, arrow, label) in enumerate(rows):
        yc = ly + U + row_h * (k + 0.5)
        x_l, x_r = lx + 1.5 * U, lx + 7.5 * U
        if isinstance(wpt, tuple):
            x_mid = (x_l + x_r) / 2
            doc.edge(f"legend-line{k}a", line_style(color, wpt[0]),
                     points=[(x_l, yc), (x_mid - 0.5 * U, yc)])
            doc.edge(f"legend-line{k}b", line_style(color, wpt[1]),
                     points=[(x_mid + 0.5 * U, yc), (x_r, yc)])
        else:
            doc.edge(f"legend-line{k}", line_style(color, wpt, dashed=dashed, arrow=arrow),
                     points=[(x_l, yc), (x_r, yc)])
        doc.vertex(f"legend-text{k}", label, text_style("left"),
                   lx + 8.5 * U, yc - 7, lw - 9 * U, 14)

    # --- panel letter, only if asked for: a single-panel figure carries none
    if letter:
        doc.vertex("letter", letter,
                   f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=left;"
                   f"verticalAlign=top;fontFamily=Arial;fontSize={LETTER_FONT};fontStyle=1;",
                   0, 0, 24, 18)

    doc.write(out_path)
    print(f"wrote {out_path}")


def export(drawio_bin, src, out_stem):
    """Export the diagram to a true-size SVG and a 3x PNG under the images directory.

    The SVG is measured against the Nature cap and the run fails if it is over. On Linux
    the input path goes FIRST: drawio-desktop 24.7 rejects an input placed after the
    Electron flags.
    """
    os.makedirs(IMAGES_DIR, exist_ok=True)
    svg = osp.join(IMAGES_DIR, out_stem + ".svg")
    png = osp.join(IMAGES_DIR, out_stem + ".png")
    for fmt, out, extra in (("svg", svg, []), ("png", png, ["-s", "3"])):
        if sys.platform == "darwin":
            cmd = [drawio_bin, "-x", "-f", fmt, *extra, "-o", out, src]
        else:
            cmd = ["xvfb-run", "-a", drawio_bin, src, "--no-sandbox", "--disable-gpu",
                   "-x", "-f", fmt, *extra, "-o", out]
        # xvfb-run exits 1 after the export because its own cleanup kill finds no process,
        # so the exit code says nothing; the output file is the success condition.
        subprocess.run(cmd, capture_output=True)
        if not osp.exists(out) or osp.getsize(out) == 0:
            raise RuntimeError(f"draw.io export wrote nothing: {' '.join(cmd)}")
    head = open(svg, encoding="utf-8").read(2000)
    m = re.search(r'width="([\d.]+)px" height="([\d.]+)px"', head)
    if m is None:
        raise ValueError(f"{svg}: no px width/height on the <svg> element")
    w_mm, h_mm = float(m.group(1)) / U, float(m.group(2)) / U
    print(f"exported {w_mm:.1f} x {h_mm:.1f} mm -> {svg}\n         {png}")
    if w_mm > 179.4 or h_mm > 170:
        raise ValueError(f"export is {w_mm:.1f} x {h_mm:.1f} mm, over the 179.4 x 170 cap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="multiplicative",
                    choices=["multiplicative", "additive", "log_ols", "glm_log_link"])
    ap.add_argument("--readout", default=panel.TOTAL)
    ap.add_argument("--graph", default=None)
    ap.add_argument("--letter", default="", help="panel letter, e.g. a; none by default")
    ap.add_argument("--out", default=osp.join(DRAWIO_DIR, "ffa-epistasis-fig4-network-overlay.drawio"))
    ap.add_argument("--drawio", default=None,
                    help="path to the draw.io binary; when given, also export a true-size "
                         "SVG and a PNG under notes/assets/images/008-xue-ffa/ and check "
                         "the size against the Nature cap")
    args = ap.parse_args()
    os.makedirs(osp.dirname(args.out), exist_ok=True)
    build(args.model, args.readout, args.graph, args.letter, args.out)
    if args.drawio:
        export(args.drawio, args.out, osp.splitext(osp.basename(args.out))[0])


if __name__ == "__main__":
    main()
