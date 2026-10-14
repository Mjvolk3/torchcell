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
draw.io). With DRAWIO set it exports the composition as the sheet's own PDF and PNG,
fig2_gene_interactions.pdf and .png, the one deliverable; the sheet script writes only SVGs.
Type sizes follow the ladder (8.3 for figure text, 7 the floor), never a resized raster.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import re
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


def scale_cells(keep: list[ET.Element], origin: tuple[float, float], s: float, target: tuple[float, float], fonts: str) -> None:
    """Scale the explainer by s about its own top-left, move it to target, and set its type.

    Geometry scales uniformly, children included (their geometry is relative to the parent,
    so it scales the same way). Type does not scale with geometry: with fonts="keep" every
    label keeps the size it was typed at; with fonts="ladder" each label is retyped to the
    nearest ladder step at or above its current size (7, 8.3, 9.7 draw.io units, which
    print at 5.0, 6.0 and 7.0 pt), the repo's rule for figure text.
    """
    ladder = [7.0, 8.3, 9.7]

    def retype(px: str) -> str:
        v = float(px)
        if fonts == "keep":
            return px
        if fonts == "uniform":
            # The whole explainer, type included, scales as one, so the composition is the
            # sheet's own look with the type kept as vector.
            return f"{v * s:.2f}"
        return f"{next((step for step in ladder if step >= v - 1e-6), ladder[-1]):g}"

    for c in keep:
        top = c.get("parent") == "1"
        g = c.find("mxGeometry")
        if g is not None:
            for k in ("x", "y", "width", "height"):
                if g.get(k) is not None:
                    v = float(g.get(k)) * s
                    if top and k in ("x", "y"):
                        v += target[0] - origin[0] * s if k == "x" else target[1] - origin[1] * s
                    g.set(k, f"{v:.2f}")
            for pt in g.iter("mxPoint"):
                for k in ("x", "y"):
                    if pt.get(k) is not None:
                        v = float(pt.get(k)) * s
                        if top:
                            v += target[0] - origin[0] * s if k == "x" else target[1] - origin[1] * s
                        pt.set(k, f"{v:.2f}")
        if c.get("value"):
            c.set("value", re.sub(r"font-size:\s*([\d.]+)px", lambda m: f"font-size: {retype(m.group(1))}px", c.get("value")))
        st = c.get("style") or ""
        if "fontSize=" in st:
            c.set("style", re.sub(r"fontSize=([\d.]+)", lambda m: f"fontSize={retype(m.group(1))}", st))


TAGS = {"161": "left", "162": "left", "163": "right", "200": "right"}  # growth, digenic, trigenic, general
TAG_SIZE = 7.0  # the ladder's floor, 5.04 pt


EQUATIONS = {"83": "left", "76": "left", "80": "left", "154": "center"}  # growth, digenic, trigenic, general
EQUATION_SIZE = 6.5  # 4.7 pt; 7 would run the trigenic line into its tag


def retype_cells(keep: list[ET.Element], which: dict[str, str], size: float) -> None:
    """Retype the named labels to one draw.io size and grow their boxes by the same factor,
    anchored on the left edge, the right edge or the center (review 2026-10-14: the tags
    to the ladder's floor, the equations as large as the slot's rows allow)."""
    for c in keep:
        anchor = which.get(c.get("id") or "")
        if anchor is None:
            continue
        m = re.search(r"font-size:\s*([\d.]+)px", c.get("value") or "")
        assert m is not None, f"cell {c.get('id')} has no inline size"
        f = size / float(m.group(1))
        c.set("value", re.sub(r"font-size:\s*([\d.]+)px", f"font-size: {size:g}px", c.get("value")))
        g = c.find("mxGeometry")
        assert g is not None
        x, y, w, h = (float(g.get(k) or 0) for k in ("x", "y", "width", "height"))
        nw, nh = w * f, h * f
        nx = {"left": x, "right": x + w - nw, "center": x - (nw - w) / 2}[anchor]
        g.set("x", f"{nx:.2f}")
        g.set("y", f"{y - (nh - h) / 2:.2f}")
        g.set("width", f"{nw:.2f}")
        g.set("height", f"{nh:.2f}")


def enlarge_tags(keep: list[ET.Element]) -> None:
    retype_cells(keep, TAGS, TAG_SIZE)
    retype_cells(keep, EQUATIONS, EQUATION_SIZE)


NUDGES = {"154": (-12.0, 0.0), "163": (10.0, 0.0), "200": (10.0, 0.0)}  # the beige box left 3 mm; the trigenic and general tags right 2.5 mm, clear of the larger equations


def nudge(keep: list[ET.Element]) -> None:
    for c in keep:
        d = NUDGES.get(c.get("id") or "")
        if d is None:
            continue
        g = c.find("mxGeometry")
        assert g is not None
        g.set("x", f"{float(g.get('x') or 0) + d[0]:.2f}")
        g.set("y", f"{float(g.get('y') or 0) + d[1]:.2f}")


def main() -> None:
    fonts = os.environ.get("FIG2_FONTS", "uniform")
    slot = json.load(open(SLOT_JSON))
    sx, sy = slot["x_mm"] * UNITS_PER_MM, slot["y_mm"] * UNITS_PER_MM
    sw, sh = slot["width_mm"] * UNITS_PER_MM, slot["height_mm"] * UNITS_PER_MM
    keep, _ = explainer_cells()
    xs, ys, x2, y2 = [], [], [], []
    for c in keep:
        if c.get("parent") != "1":
            continue
        g = c.find("mxGeometry")
        if c.get("edge") == "1":
            for pt in c.iter("mxPoint"):
                if pt.get("x") is not None:
                    xs.append(float(pt.get("x"))), ys.append(float(pt.get("y"))), x2.append(float(pt.get("x"))), y2.append(float(pt.get("y")))
        elif g is not None:
            x, y, w, h = (float(g.get(k) or 0) for k in ("x", "y", "width", "height"))
            xs.append(x), ys.append(y), x2.append(x + w), y2.append(y + h)
    ew, eh = max(x2) - min(xs), max(y2) - min(ys)
    s = min(sw / ew, sh / eh)
    scale_cells(keep, (min(xs), min(ys)), s, (sx, sy), fonts)
    enlarge_tags(keep)
    nudge(keep)
    sheet_cell, _ = image_cell("sheet", SHEET_SVG, 0.0)
    # The slot's outline lives on a hidden layer: visible while editing, absent from exports.
    guide = (
        '<mxCell id="guides" value="guides" style="locked=1;" parent="0" visible="0" />'
        f'<mxCell id="slot_a_guide" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#B85450;dashed=1;strokeWidth=0.5;" vertex="1" parent="guides">'
        f'<mxGeometry x="{sx:.2f}" y="{sy:.2f}" width="{sw:.2f}" height="{sh:.2f}" as="geometry" /></mxCell>'
    )
    page_w, page_h = slot["page_width_mm"] * UNITS_PER_MM, slot["page_height_mm"] * UNITS_PER_MM
    body = "".join(ET.tostring(c, encoding="unicode") for c in keep)
    mxfile = (
        '<mxfile host="torchcell" type="device"><diagram id="fig2sheet" name="Fig2 sheet">'
        f'<mxGraphModel dx="0" dy="0" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{page_w:.2f}" pageHeight="{page_h:.2f}" math="1" shadow="0">'
        '<root><mxCell id="0" /><mxCell id="1" parent="0" />'
        f"{sheet_cell}{body}{guide}"
        "</root></mxGraphModel></diagram></mxfile>"
    )
    open(OUT, "w", encoding="utf-8").write(mxfile)
    sizes = sorted({float(m) for c in keep for m in re.findall(r"font-size:\s*([\d.]+)px", c.get("value") or "")})
    print(f"{len(keep)} explainer cells, {ew * 0.254:.1f} x {eh * 0.254:.1f} mm, scaled by {s:.3f} into the slot {slot['width_mm']:.1f} x {slot['height_mm']:.1f} mm, type {fonts}: draw.io sizes {sizes} = {[round(v * 0.72, 1) for v in sizes]} pt; wrote {OUT}")
    drawio = os.environ.get("DRAWIO")
    if drawio:
        for fmt in ("pdf", "png"):
            out = osp.join(IMG_DIR, f"fig2_gene_interactions.{fmt}")
            cmd = ["xvfb-run", "-a", drawio, "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--export", "--format", fmt, "--crop", "--output", out, OUT]
            if fmt == "png":
                cmd[-3:-3] = ["--scale", "4"]
            subprocess.run(cmd, check=True, env={**os.environ, "ELECTRON_DISABLE_SECURITY_WARNINGS": "true"}, capture_output=True)
            print("exported", out)


if __name__ == "__main__":
    main()
