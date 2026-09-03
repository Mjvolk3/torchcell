# experiments/008-xue-ffa/scripts/build_perspective_drawio.py
# [[experiments.008-xue-ffa.scripts.build_perspective_drawio]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/build_perspective_drawio
#
# Assemble the perspective's draw.io figures from the individual true-size panel SVGs, so
# the panels can be rearranged, relettered, and annotated by hand without leaving draw.io.
#
# WHY A GENERATOR rather than four hand-built .drawio files. Every panel is regenerated
# whenever perspective_figure_panels.py or notable_interaction_selection.py runs, and a
# hand-placed panel would then silently show the previous render. Running this script after
# a panel rebuild re-embeds the current SVGs at the current sizes. The layout below is a
# starting arrangement, not a final one: moving a panel in draw.io is exactly the
# workshopping this is for, and re-running the script discards those moves.
#
# UNITS. draw.io's canvas is 100 units per inch, so 1 mm = 3.937 units and Nature's
# 180 mm full-page box is ~709 units. torchcell.utils.savefig_true_size_svg already writes
# each panel's width/height in those units, which is why a panel's own SVG header can be
# read straight into an mxCell geometry with no conversion. The network overlay is the one
# exception: it is written in millimetres and is converted here.
#
# TYPE. Panel letters are 8 pt bold lowercase, the only figure text Nature allows above
# 7 pt. In draw.io the font-size field is in canvas units, not points, so 8 pt is typed as
# 8 / 0.72 = 11.1. Audit any hand edits with
# paper/nature-biotech/scripts/drawio_font_band.py --check.

import base64
import os
import os.path as osp
import re
import xml.etree.ElementTree as ET

from dotenv import load_dotenv

load_dotenv()

ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
DRAWIO_DIR = osp.join(osp.dirname(ASSET_IMAGES_DIR), "drawio")

MM_TO_UNITS = 3.937008
FULL_WIDTH_UNITS = 179.0 * MM_TO_UNITS  # 704.7, Nature's full-page box
MAX_HEIGHT_UNITS = 170.0 * MM_TO_UNITS  # 669.3

# 8 pt bold lowercase panel letter, typed in draw.io's canvas units.
LETTER_FONT_UNITS = 11.1
LETTER_BOX_W = 24.0
LETTER_BOX_H = 18.0
# Vertical space a letter row occupies above its panel, and the gaps between panels.
LETTER_ROW_H = 16.0
COL_GAP = 11.0
ROW_GAP = 16.0

PANEL_STYLE = (
    "shape=image;imageAspect=0;aspect=fixed;html=1;verticalLabelPosition=bottom;"
    "verticalAlign=top;labelBackgroundColor=none;"
)
LETTER_STYLE = (
    f"text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;"
    f"fontSize={LETTER_FONT_UNITS};fontStyle=1;fontFamily=Arial;whiteSpace=wrap;"
)


def svg_size_units(path):
    """The panel's own drawn size, in draw.io canvas units.

    savefig_true_size_svg writes bare numbers already in 100-units-per-inch; the network
    overlay's rescaler writes millimetres. Both forms appear in this figure set, so the
    unit is read rather than assumed.
    """
    with open(path, "r", encoding="utf-8") as fh:
        head = fh.read(2000)
    m = re.search(r'<svg[^>]*\swidth="([\d.]+)(mm)?"[^>]*\sheight="([\d.]+)(mm)?"', head)
    if not m:
        raise ValueError(f"{path}: no width/height on the <svg> element")
    w, w_mm, h, h_mm = float(m.group(1)), m.group(2), float(m.group(3)), m.group(4)
    if w_mm:
        w *= MM_TO_UNITS
    if h_mm:
        h *= MM_TO_UNITS
    return w, h


def data_uri(path):
    """draw.io's own embedded-image form: `data:image/svg+xml,` then base64, no `;base64`."""
    with open(path, "rb") as fh:
        return "data:image/svg+xml," + base64.b64encode(fh.read()).decode("ascii")


def build(figure_name, title, rows, out_path):
    """One draw.io page: rows of panels, each with a lowercase letter above its top-left.

    rows is a list of lists of (letter, svg_path). Panels in a row are laid out left to
    right at their true size with a fixed gap; rows stack top to bottom. Each row is left
    aligned rather than centered, because a Nature multi-panel figure aligns its panel
    letters on a common left edge and centering would break that on any ragged row.

    A letter of None places the panel with no letter row above it. A single-panel figure
    has nothing to letter, and reserving the row anyway pushed a full-height figure past
    the print box.
    """
    mxfile = ET.Element("mxfile", {"host": "torchcell", "type": "device"})
    diagram = ET.SubElement(mxfile, "diagram", {"name": figure_name, "id": figure_name})
    model = ET.SubElement(diagram, "mxGraphModel", {
        "dx": "1200", "dy": "800", "grid": "1", "gridSize": "10", "guides": "1",
        "tooltips": "1", "connect": "1", "arrows": "1", "fold": "1", "page": "1",
        "pageScale": "1",
        "pageWidth": f"{FULL_WIDTH_UNITS:.0f}", "pageHeight": f"{MAX_HEIGHT_UNITS:.0f}",
        "math": "0", "shadow": "0",
    })
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    def cell(cid, style, value, x, y, w, h):
        c = ET.SubElement(root, "mxCell", {
            "id": cid, "value": value, "style": style, "vertex": "1", "parent": "1"})
        ET.SubElement(c, "mxGeometry", {
            "x": f"{x:.2f}", "y": f"{y:.2f}", "w": "0", "h": "0",
            "width": f"{w:.2f}", "height": f"{h:.2f}", "as": "geometry"})

    y = 0.0
    n = 0
    widest = 0.0
    for row in rows:
        sizes = [svg_size_units(osp.join(IMAGES_DIR, p)) for _, p in row]
        row_h = max(h for _, h in sizes)
        letter_h = LETTER_ROW_H if any(letter for letter, _ in row) else 0.0
        x = 0.0
        for (letter, rel), (w, h) in zip(row, sizes):
            n += 1
            if letter:
                cell(f"letter{n}", LETTER_STYLE, letter, x, y, LETTER_BOX_W, LETTER_BOX_H)
            cell(f"panel{n}", PANEL_STYLE + f"image={data_uri(osp.join(IMAGES_DIR, rel))};",
                 "", x, y + letter_h, w, h)
            x += w + COL_GAP
        widest = max(widest, x - COL_GAP)
        y += letter_h + row_h + ROW_GAP
    total_h = y - ROW_GAP

    ET.ElementTree(mxfile).write(out_path, encoding="utf-8", xml_declaration=True)
    over = " OVER the 170 mm box" if total_h > MAX_HEIGHT_UNITS + 0.5 else ""
    print(f"wrote {osp.basename(out_path)}")
    print(f"       {widest / MM_TO_UNITS:6.1f} x {total_h / MM_TO_UNITS:6.1f} mm"
          f"   ({n} panels){over}")
    print(f"       {title}")
    return total_h


# Each figure is one argument. The panels within a figure are the evidence for it, in the
# order a reader needs them, and the letters follow that order rather than the layout.
FIGURES = [
    (
        "ffa-epistasis-fig1-interaction-landscape",
        "Interactions are the rule, not the exception, and the third deletion is where "
        "they bite.",
        [
            [("a", "panel_interaction_distribution.svg"),
             ("b", "panel_volcano_total_titer.svg"),
             ("c", "panel_consensus_agreement.svg")],
        ],
    ),
    (
        "ffa-epistasis-fig2-stepwise-inaccessibility",
        "Improving combinations exist; almost none is reachable one deletion at a time.",
        [
            [("a", "panel_improving_by_order.svg"),
             ("b", "panel_greedy_walk.svg"),
             ("c", "panel_path_accessibility.svg")],
            [("d", "ffa_epistatic_path_panels_top.svg")],
        ],
    ),
    (
        "ffa-epistasis-figS1-supporting",
        "Supporting panels: how deep the valleys run, and which interactions are largest.",
        [
            [("a", "panel_path_valley.svg"),
             ("b", "panel_top_interactions.svg")],
            [("c", "ffa_epistatic_path_panels_divergent.svg")],
        ],
    ),
    (
        "ffa-epistasis-fig3-model-and-scale",
        "What counts as an interaction depends on the null and on the scale.",
        [
            [("a", "panel_model_agreement.svg"),
             ("b", "panel_graph_enrichment.svg")],
            [("c", "panel_scale_scatter.svg"),
             ("d", "panel_scale_slope.svg"),
             ("e", "panel_readout_sign.svg")],
        ],
    ),
    (
        "ffa-epistasis-fig4-network-overlay",
        "The interactions drawn on the pathway they act through.",
        [
            [(None, "ffa_multigraph_overlays/multiplicative/all_ffa/"
                   "multiplicative_ffa_multigraph_Genetic_Interactions_connected"
                   "_unenriched_Total_Titer_fdr_within.svg")],
        ],
    ),
]


def main():
    os.makedirs(DRAWIO_DIR, exist_ok=True)
    for name, title, rows in FIGURES:
        build(name, title, rows, osp.join(DRAWIO_DIR, f"{name}.drawio"))


if __name__ == "__main__":
    main()
