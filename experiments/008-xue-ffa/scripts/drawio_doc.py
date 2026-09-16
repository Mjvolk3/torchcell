# experiments/008-xue-ffa/scripts/drawio_doc.py
# [[experiments.008-xue-ffa.scripts.drawio_doc]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/drawio_doc
#
# The mxGraph document writer the perspective's native draw.io figures share. Two figures
# emit draw.io XML directly (the network overlay and the epistasis-model explainer), and a
# second copy of this class would drift: the unit constant, the page size and the geometry
# element are the parts a reader has to trust are identical across figures.
#
# UNITS. draw.io's canvas is 100 units per inch. Nature's full-page box is 179.4 mm wide,
# which the app writes as 706.6915 units, so U = 706.6915 / 179.4 units per millimetre.
# A point is 1 / 0.72 units, which is why a font typed as 8.3 prints at 5.98 pt and a
# 0.4 pt line is strokeWidth 0.56.

import xml.etree.ElementTree as ET

U = 706.6915 / 179.4  # canvas units per mm
PT = 1 / 0.72  # canvas units per point
FONT = "8.3"  # 5.98 pt, the default for figure text
LETTER_FONT = "11.1"  # 7.99 pt, panel letters only
PAGE_W_MM = 179.4
PAGE_H_MM = 170.0


class Doc:
    """An mxGraph document: one diagram, cells appended in draw order."""

    def __init__(self, name):
        """Start a diagram named ``name`` on a Nature full-page canvas."""
        self.mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "agent": "torchcell"})
        diagram = ET.SubElement(self.mxfile, "diagram", {"name": name, "id": name})
        model = ET.SubElement(diagram, "mxGraphModel", {
            "dx": "1400", "dy": "1000", "grid": "0", "gridSize": "10", "guides": "1",
            "tooltips": "1", "connect": "1", "arrows": "1", "fold": "1", "page": "1",
            "pageScale": "1", "pageWidth": f"{PAGE_W_MM * U:.0f}",
            "pageHeight": f"{PAGE_H_MM * U:.0f}", "math": "0", "shadow": "0",
        })
        self.root = ET.SubElement(model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})

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


def text_style(align="center", extra=""):
    return (f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align={align};"
            f"verticalAlign=middle;fontFamily=Arial;fontSize={FONT};fontColor=#000000;"
            f"{extra}")


def letter_style():
    return (f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;align=left;"
            f"verticalAlign=top;fontFamily=Arial;fontSize={LETTER_FONT};fontStyle=1;")


def line_style(color, width_pt, dashed=False, arrow=False):
    s = (f"edgeStyle=none;html=1;strokeColor={color};strokeWidth={width_pt * PT:.2f};"
         f"endArrow={'classic' if arrow else 'none'};startArrow=none;"
         f"endFill=1;endSize=2.5;rounded=0;")
    if dashed:
        s += "dashed=1;dashPattern=1 2;"
    return s
