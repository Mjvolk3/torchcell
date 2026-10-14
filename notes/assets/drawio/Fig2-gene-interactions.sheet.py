# notes/assets/drawio/Fig2-gene-interactions.sheet.py
# [[notes.assets.drawio.Fig2-gene-interactions.sheet]]
# https://github.com/Mjvolk3/torchcell/tree/main/notes/assets/drawio/Fig2-gene-interactions.sheet
"""Compose the Figure 2 sheet in draw.io: the matplotlib sheet with slot a empty as a true-size
image, the hand-drawn explainer laid over the slot as vector cells, and a dashed guide for
the slot's box, so the explainer can be edited at true size and exported as vector.

Reads fig2_gene_interactions_slot_a.svg and _slot_a.json (the slot's box in mm) written by
experiments/025-solid-growth/scripts/fig2_gene_interactions.py, takes the explainer cells
from Fig2-gene-interactions.drawio.svg with the same selection as the explainer export
(first slot, frame and letter excluded), moves them so their top-left sits on the slot's
top-left, unscaled, and writes Fig2-gene-interactions-sheet.drawio (plain XML, open it in
draw.io). With DRAWIO set it also exports the composition to PDF and PNG beside the sheet.
Type sizes follow the ladder (8.3 for figure text, 7 the floor), never a resized raster.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import subprocess
import sys
import xml.etree.ElementTree as ET

from dotenv import load_dotenv

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "publish", "scripts"))
from drawio_true_size_import import image_cell, load_drawio  # noqa: E402

load_dotenv()
HERE = osp.dirname(osp.abspath(__file__))
FIG = osp.join(HERE, "Fig2-gene-interactions.drawio.svg")
OUT = osp.join(HERE, "Fig2-gene-interactions-sheet.drawio")
IMG_DIR = osp.join(os.environ["ASSET_IMAGES_DIR"], "025-solid-growth")
SHEET_SVG = osp.join(IMG_DIR, "fig2_gene_interactions_slot_a.svg")
SLOT_JSON = osp.join(IMG_DIR, "fig2_gene_interactions_slot_a.json")
SLOT = (232.0, 240.0)
EXCLUDE = {"194", "48"}
UNITS_PER_MM = 100 / 25.4


def explainer_cells() -> tuple[list[ET.Element], dict[str, ET.Element]]:
    _, mx, _ = load_drawio(FIG)
    root = ET.fromstring(mx)
    cells = root.find(".//mxGraphModel/root")
    assert cells is not None
    by_id = {c.get("id"): c for c in cells}

    def top(c: ET.Element) -> ET.Element:
        while c.get("parent") not in ("0", "1", None):
            c = by_id[c.get("parent")]
        return c

    def inside(c: ET.Element) -> bool:
        g = top(c).find("mxGeometry")
        return g is not None and float(g.get("x") or 0) < SLOT[0] and float(g.get("y") or 0) < SLOT[1]

    keep = []
    for c in list(cells):
        if c.get("id") in ("0", "1"):
            continue
        if c.get("edge") == "1":
            ends = [by_id[e] for e in (c.get("source"), c.get("target")) if e in by_id]
            if ends:
                ok = all(inside(e) for e in ends)
            else:
                pts = [p for p in c.iter("mxPoint") if p.get("x") is not None]
                ok = bool(pts) and all(float(p.get("x") or 0) < SLOT[0] and float(p.get("y") or 0) < SLOT[1] for p in pts)
        else:
            ok = inside(c)
        if ok and top(c).get("id") not in EXCLUDE:
            keep.append(c)
    return keep, by_id


def main() -> None:
    slot = json.load(open(SLOT_JSON))
    sx, sy = slot["x_mm"] * UNITS_PER_MM, slot["y_mm"] * UNITS_PER_MM
    sw, sh = slot["width_mm"] * UNITS_PER_MM, slot["height_mm"] * UNITS_PER_MM
    keep, _ = explainer_cells()
    # The explainer's own top-left, over its top-level vertices and free edge points.
    xs, ys = [], []
    for c in keep:
        if c.get("parent") != "1":
            continue
        g = c.find("mxGeometry")
        if c.get("edge") == "1":
            for pt in c.iter("mxPoint"):
                if pt.get("x") is not None:
                    xs.append(float(pt.get("x"))), ys.append(float(pt.get("y")))
        elif g is not None:
            xs.append(float(g.get("x") or 0)), ys.append(float(g.get("y") or 0))
    dx, dy = sx - min(xs), sy - min(ys)
    for c in keep:
        if c.get("parent") != "1":
            continue
        g = c.find("mxGeometry")
        if c.get("edge") == "1":
            for pt in c.iter("mxPoint"):
                if pt.get("x") is not None:
                    pt.set("x", f"{float(pt.get('x')) + dx:.2f}")
                    pt.set("y", f"{float(pt.get('y')) + dy:.2f}")
        elif g is not None:
            g.set("x", f"{float(g.get('x') or 0) + dx:.2f}")
            g.set("y", f"{float(g.get('y') or 0) + dy:.2f}")
    sheet_cell, sheet_w = image_cell("sheet", SHEET_SVG, 0.0)
    guide = (
        f'<mxCell id="slot_a_guide" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#B85450;dashed=1;strokeWidth=0.5;" vertex="1" parent="1">'
        f'<mxGeometry x="{sx:.2f}" y="{sy:.2f}" width="{sw:.2f}" height="{sh:.2f}" as="geometry" /></mxCell>'
    )
    page_w, page_h = slot["page_width_mm"] * UNITS_PER_MM, slot["page_height_mm"] * UNITS_PER_MM
    body = "".join(ET.tostring(c, encoding="unicode") for c in keep)
    mxfile = (
        '<mxfile host="torchcell" type="device"><diagram id="fig2sheet" name="Fig2 sheet">'
        f'<mxGraphModel dx="0" dy="0" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{page_w:.2f}" pageHeight="{page_h:.2f}" math="1" shadow="0">'
        '<root><mxCell id="0" /><mxCell id="1" parent="0" />'
        f"{sheet_cell}{guide}{body}"
        "</root></mxGraphModel></diagram></mxfile>"
    )
    open(OUT, "w", encoding="utf-8").write(mxfile)
    print(f"{len(keep)} explainer cells moved by ({dx:.1f}, {dy:.1f}) units onto the slot {slot['width_mm']:.1f} x {slot['height_mm']:.1f} mm; wrote {OUT}")
    drawio = os.environ.get("DRAWIO")
    if drawio:
        for fmt in ("pdf", "png"):
            out = osp.join(IMG_DIR, f"fig2_gene_interactions_composed.{fmt}")
            cmd = ["xvfb-run", "-a", drawio, "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--export", "--format", fmt, "--crop", "--output", out, OUT]
            if fmt == "png":
                cmd[-3:-3] = ["--scale", "4"]
            subprocess.run(cmd, check=True, env={**os.environ, "ELECTRON_DISABLE_SECURITY_WARNINGS": "true"}, capture_output=True)
            print("exported", out)


if __name__ == "__main__":
    main()
