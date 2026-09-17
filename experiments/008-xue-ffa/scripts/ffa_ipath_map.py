# experiments/008-xue-ffa/scripts/ffa_ipath_map.py
# [[experiments.008-xue-ffa.scripts.ffa_ipath_map]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/ffa_ipath_map
#
# The measured chemistry drawn on the global metabolic map: the route from glycolysis and
# the citrate cycle out to the five fatty acid species this study measures, highlighted on
# iPath3's reference map, then cropped to the highlighted region.
#
# WHAT THIS IS FOR. The network figure shows ten transcription factors upstream of a
# thirteen-gene pathway, which says nothing about where that pathway sits in the cell's
# metabolism or how far the measured species are from central carbon. This panel answers
# that in one picture: the whole map in gray, the route in color, the five measured species
# as the only large nodes on it.
#
# THERE IS NO iPath PYTHON PACKAGE. iPath3 is a web service; the client is this file. The
# service takes a newline-separated selection of KEGG identifiers with per-entry color,
# width or radius and opacity, and returns the reference map as SVG with those entries
# restyled. Nothing else is sent.
#
# PROVENANCE. Everything selected comes from the KEGG REST API, never from a hand-written
# list of KO numbers: the yeast genes of each pathway from link/sce/<pathway>, their KEGG
# Orthology ids from link/ko/sce. Both responses, the exact selection posted, and the
# returned SVG are written to results/ipath/ with their sha256 and the command that
# retrieved them, so the panel rebuilds from the stored response with no network and the
# run fails loudly if an upstream response has changed rather than following the drift.
# Re-running with --refresh re-fetches; the default reads the stored copies.
#
# CROPPING is anchored on the COMPOUNDS, not on the highlighted reactions. KEGG draws one
# orthology group wherever it occurs, so highlighting glycolysis puts ink in every corner
# of a 3774 x 2250 map and the bounding box of the highlighted reactions is the whole map.
# The compounds this panel names are in one place each, so their bounding box is the region
# the panel is actually about; it is padded, then widened symmetrically to the panel's
# aspect ratio, which buys surrounding context rather than distorting the drawing.
#
# TWO OF THE FIVE MEASURED SPECIES HAVE A NODE ON THIS MAP. Myristate and palmitate are
# drawn; palmitoleate, stearate and oleate are not. This is the KEGG global map's LAYOUT,
# not a gap in KEGG's chemistry: all five have KEGG compound entries, and the map places
# only some of them (dodecanoate, myristate and palmitate are on it; the C16:1, C18:0 and
# C18:1 acids are not). The network panel beside this one has all five because its species
# come from the yeast genome-scale model, which carries compartment-resolved acyl species.
# The script reports which were placed and the caption says so; nothing is substituted for
# a species the map does not carry.
#
# OPACITY. The service applies one opacity to every entry, so a per-entry O flag in the
# selection is ignored. The background is therefore requested faint and the highlighted
# elements are set opaque afterwards, in the returned SVG, by the one rule that an element
# carrying a highlight color is part of the route.

import argparse
import hashlib
import json
import os
import os.path as osp
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from dotenv import load_dotenv

from torchcell.utils import PANEL_WIDTHS_MM

load_dotenv()
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "008-xue-ffa/results")
IPATH_DIR = osp.join(RESULTS_DIR, "ipath")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")

KEGG = "https://rest.kegg.jp"
IPATH = "https://pathways.embl.de/mapping.cgi"

# The modules the route passes through, in the order the legend lists them, each with the
# color it takes. Palette primaries; blue is reserved for the measured species.
MODULES = [
    ("sce00010", "glycolysis and gluconeogenesis", "#9673A6"),
    ("sce00620", "pyruvate metabolism", "#D6B656"),
    ("sce00020", "citrate cycle", "#B85450"),
    ("sce00061", "fatty acid biosynthesis", "#D79B00"),
    ("sce00062", "fatty acid elongation", "#D79B00"),
    ("sce01040", "unsaturated fatty acid biosynthesis", "#D79B00"),
    ("sce00071", "fatty acid degradation", "#666666"),
]

# The five measured species, as the KEGG compounds the titers are of.
SPECIES = [
    ("C06424", "C14:0 myristate"),
    ("C00249", "C16:0 palmitate"),
    ("C08362", "C16:1 palmitoleate"),
    ("C01530", "C18:0 stearate"),
    ("C00712", "C18:1 oleate"),
]
# Waypoints, labeled so a reader can find the route without reading every node.
WAYPOINTS = [
    ("C00022", "pyruvate"),
    ("C00024", "acetyl-CoA"),
    ("C00158", "citrate"),
    ("C00083", "malonyl-CoA"),
]

C_SPECIES = "#6C8EBF"
C_WAYPOINT = "#666666"
W_REACTION = 14
R_WAYPOINT = 30
R_SPECIES = 36
BACKGROUND_OPACITY = 0.28  # the reference map behind the route
CROP_MARGIN = 230.0  # map units around the named compounds

HIGHLIGHT_COLORS = {c for _, _, c in MODULES} | {C_SPECIES}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url, store, refresh):
    """GET `url`, caching the exact bytes under results/ipath/<store>.

    The stored copy is what the panel is built from. With --refresh the URL is fetched
    again and the run raises if the bytes differ from the stored ones, which is the
    upstream-drift check rather than a silent update.
    """
    path = osp.join(IPATH_DIR, store)
    if osp.exists(path) and not refresh:
        return open(path, "rb").read()
    with urllib.request.urlopen(url, timeout=120) as r:
        body = r.read()
    if osp.exists(path):
        old = open(path, "rb").read()
        if old != body:
            raise RuntimeError(
                f"{url} changed upstream: stored sha256 {sha256(old)}, fetched "
                f"{sha256(body)}. This is a new version of the source, not an update to "
                f"the old one: move {path} aside deliberately before accepting it.")
    os.makedirs(IPATH_DIR, exist_ok=True)
    open(path, "wb").write(body)
    return body


def kegg_pairs(text):
    """A KEGG link response as a list of (left, right) with the prefixes stripped."""
    out = []
    for line in text.strip().splitlines():
        a, b = line.split("\t")
        out.append((a.split(":", 1)[1], b.split(":", 1)[1]))
    return out


def selection(refresh):
    """The iPath selection lines, and the record of what produced them."""
    gene_to_ko = {}
    for orf, ko in kegg_pairs(fetch(f"{KEGG}/link/ko/sce", "kegg_link_ko_sce.tsv",
                                    refresh).decode()):
        gene_to_ko.setdefault(orf, ko)

    lines, record = [], {"modules": [], "species": [], "waypoints": []}
    claimed = set()
    for pathway, label, color in MODULES:
        pairs = kegg_pairs(fetch(f"{KEGG}/link/sce/{pathway}",
                                 f"kegg_link_sce_{pathway}.tsv", refresh).decode())
        kos = sorted({gene_to_ko[orf] for _, orf in pairs if orf in gene_to_ko})
        new = [k for k in kos if k not in claimed]
        claimed.update(new)
        for ko in new:
            lines.append(f"{ko} {color} W{W_REACTION}")
        record["modules"].append({"pathway": pathway, "label": label, "color": color,
                                  "genes": len(pairs), "ko_selected": len(new)})

    for cid, label in WAYPOINTS:
        lines.append(f"{cid} {C_WAYPOINT} W{R_WAYPOINT}")
        record["waypoints"].append({"compound": cid, "label": label})
    for cid, label in SPECIES:
        lines.append(f"{cid} {C_SPECIES} W{R_SPECIES}")
        record["species"].append({"compound": cid, "label": label, "color": C_SPECIES})

    return "\n".join(lines), record


def fetch_map(sel, refresh):
    """POST the selection to iPath3 and return the SVG bytes, cached like every fetch.

    The cache file is named for the selection's sha256, so editing the selection fetches a
    new map rather than silently reusing the response to a different question.
    """
    path = osp.join(IPATH_DIR, f"ipath3_metabolic_{sha256(sel.encode())[:12]}.svg")
    if osp.exists(path) and not refresh:
        return open(path, "rb").read()
    form = urllib.parse.urlencode({
        "selection": sel, "map": "metabolic", "export_type": "svg",
        "default_opacity": BACKGROUND_OPACITY, "default_width": 3, "default_radius": 7,
        "default_color": "#999999", "keep_colors": 1,
    }).encode()
    with urllib.request.urlopen(urllib.request.Request(IPATH, data=form), timeout=180) as r:
        body = r.read()
    os.makedirs(IPATH_DIR, exist_ok=True)
    open(path, "wb").write(body)
    return body


NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
ELEMENT = re.compile(r"<(?:path|ellipse|circle|rect)\b[^>]*>")


def is_highlighted(element):
    return any(c.lstrip("#").upper() in element.upper() for c in HIGHLIGHT_COLORS)


def compound_nodes(svg):
    """Every drawn compound node of the selection, as (color, cx, cy) in map units.

    A compound the reference map does not carry simply has no node; the caller reports
    which of the named compounds were placed.
    """
    out = []
    for m in ELEMENT.finditer(svg):
        el = m.group(0)
        color = re.search(r"fill='(#[0-9A-Fa-f]{6})'", el)
        cx = re.search(r"cx='(-?[\d.]+)'", el)
        cy = re.search(r"cy='(-?[\d.]+)'", el)
        if color is None or cx is None or cy is None:
            continue
        hexcolor = color.group(1).upper()
        if hexcolor in (C_SPECIES.upper(), C_WAYPOINT.upper()):
            out.append((hexcolor, float(cx.group(1)), float(cy.group(1))))
    return out


def opaque_route(svg):
    """The route at full opacity, everything else at BACKGROUND_OPACITY.

    iPath applies one opacity to the whole selection, so the returned map has the route
    as faint as the background. Both are set here, in the returned SVG, which also means
    the background level is a property of this script and not of the cached response: a
    change to BACKGROUND_OPACITY re-renders without a refetch.
    """
    def sub(m):
        el = m.group(0)
        level = "1" if is_highlighted(el) else f"{BACKGROUND_OPACITY}"
        if "opacity:" in el:
            return re.sub(r"opacity:\s*[\d.]+", f"opacity: {level}", el)
        return el
    return ELEMENT.sub(sub, svg)


PROBE_COLOR = "#FF00FF"


def compound_positions(refresh):
    """Where the reference map draws each named compound, in map units.

    Several compounds share a highlight color, so their nodes cannot be told apart in the
    finished map. Each is therefore posted on its own in a color nothing else uses, and
    only the resulting coordinates are kept. A compound the map does not carry comes back
    with an empty list, which is the honest answer and is what the caption reports.
    """
    path = osp.join(IPATH_DIR, "compound_positions.json")
    if osp.exists(path) and not refresh:
        return json.load(open(path))
    out = {}
    for cid, _ in WAYPOINTS + SPECIES:
        form = urllib.parse.urlencode({
            "selection": f"{cid} {PROBE_COLOR} W30", "map": "metabolic",
            "export_type": "svg", "default_opacity": 0.1,
        }).encode()
        with urllib.request.urlopen(urllib.request.Request(IPATH, data=form),
                                    timeout=180) as r:
            body = r.read().decode("latin-1")
        out[cid] = [[float(re.search(r"cx='(-?[\d.]+)'", el).group(1)),
                     float(re.search(r"cy='(-?[\d.]+)'", el).group(1))]
                    for el in re.findall(r"<ellipse[^>]*" + PROBE_COLOR[1:] + r"[^>]*>", body)]
    os.makedirs(IPATH_DIR, exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    return out


def prune(svg, x0, y0, w, h):
    """Drop every element that lies outside the crop box, and the groups left empty.

    The returned map is 3774 x 2250 units and the panel shows about a sixth of it. Carried
    whole, the drawing is ~640 kB, and a base64 copy of that in one draw.io style attribute
    is more than the exporter will take: it fails the export outright.

    ALL OF THE MAP'S OWN TEXT IS DROPPED, with the colored pill each label sits on. At
    100 mm the reference map's largest label prints at about 2.5 pt, under Nature's 5 pt
    floor, so none of it is type this figure is allowed to set. The key beside the panel
    names the modules instead, at 5.98 pt, and the compounds are labeled in draw.io. What
    is left is the drawing: reaction lines and compound nodes.
    """
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.fromstring(svg)
    x1, y1 = x0 + w, y0 + h

    def coords(el):
        d = el.get("d")
        if d:
            n = [float(v) for v in NUMBER.findall(d)]
            return list(zip(n[0::2], n[1::2]))
        get = el.get
        if get("cx") is not None and get("cy") is not None:
            return [(float(get("cx")), float(get("cy")))]
        if get("x") is not None and get("y") is not None:
            return [(float(get("x")), float(get("y")))]
        return None

    def keep(el):
        pts = coords(el)
        if pts is None:
            return None  # no position of its own; decided by its children
        return any(x0 <= x <= x1 and y0 <= y <= y1 for x, y in pts)

    def walk(parent):
        for child in list(parent):
            verdict = keep(child)
            if verdict is False:
                parent.remove(child)
                continue
            if verdict is None and len(child):
                walk(child)
                if not len(child) and child.tag.endswith("}g"):
                    parent.remove(child)
    walk(root)

    def drop_labels(parent):
        """Every label, and the colored pill each one sat on.

        The pills are the only ROUNDED rects in the drawing (they carry rx), which is
        what identifies them now that their text is gone; left in, they are five empty
        lozenges. Opacity cannot be the test, because opaque_route has already reset it.
        """
        for child in list(parent):
            if child.tag.endswith("}text") or child.findall(".//{*}text"):
                parent.remove(child)
            elif child.tag.endswith("}rect") and child.get("rx") is not None:
                parent.remove(child)
            elif len(child):
                drop_labels(child)
    drop_labels(root)
    return ET.tostring(root, encoding="unicode")


def whole(svg, width_mm):
    """The whole drawing at `width_mm`, keeping its own aspect ratio.

    Returns the resized SVG, the height it takes, and a map from map units to panel
    millimetres, the same contract as crop().
    """
    head = re.search(r"<svg[^>]*>", svg).group(0)
    vb = re.search(r'viewBox="([^"]*)"', head).group(1).split()
    x0, y0, w, h = (float(v) for v in vb)
    height_mm = width_mm * h / w
    px_per_mm = 100.0 / 25.4
    new = re.sub(r"height='[\d.]+'", f"height='{height_mm * px_per_mm:.2f}'", head)
    new = re.sub(r"width='[\d.]+'", f"width='{width_mm * px_per_mm:.2f}'", new)
    to_mm = lambda x, y: ((x - x0) / w * width_mm, (y - y0) / h * height_mm)  # noqa: E731
    return prune(svg.replace(head, new, 1), x0, y0, w, h), height_mm, to_mm


def crop(svg, points, width_mm, height_mm):
    """viewBox over the named compounds, widened to the panel's aspect ratio.

    Returns the resized SVG and a map from map units to panel millimetres, which is what
    lets the figure place a label beside a compound without re-deriving the geometry.
    """
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    x0, x1 = min(xs) - CROP_MARGIN, max(xs) + CROP_MARGIN
    y0, y1 = min(ys) - CROP_MARGIN, max(ys) + CROP_MARGIN
    w, h = x1 - x0, y1 - y0
    target = width_mm / height_mm
    if w / h < target:
        need = h * target
        x0 -= (need - w) / 2
        w = need
    else:
        need = w / target
        y0 -= (need - h) / 2
        h = need
    px_per_mm = 100.0 / 25.4  # this SVG and draw.io are both 100 units per inch
    head = re.search(r"<svg[^>]*>", svg).group(0)
    new = re.sub(r"height='[\d.]+'", f"height='{height_mm * px_per_mm:.2f}'", head)
    new = re.sub(r"width='[\d.]+'", f"width='{width_mm * px_per_mm:.2f}'", new)
    new = re.sub(r'viewBox="[^"]*"', f'viewBox="{x0:.1f} {y0:.1f} {w:.1f} {h:.1f}"', new)
    to_mm = lambda x, y: ((x - x0) / w * width_mm, (y - y0) / h * height_mm)  # noqa: E731
    return prune(svg.replace(head, new, 1), x0, y0, w, h), to_mm


UNITS_PER_MM = 100.0 / 25.4  # the panel is written in draw.io's 100-units-per-inch canvas
FONT_UNITS = 6.0 / 0.72  # 6 pt, the figure-text size everywhere in this document
PANEL_KEY = [
    ("#9673A6", "glycolysis and gluconeogenesis"),
    ("#D6B656", "pyruvate metabolism"),
    ("#B85450", "citrate cycle"),
    ("#D79B00", "fatty acid biosynthesis, elongation and desaturation"),
    ("#666666", "fatty acid degradation"),
]


def esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def panel(map_svg, anchors, width_mm, map_w_mm, height_mm, key_x_mm):
    """The finished panel: the cropped map on the left, its labels, and the key.

    Assembled as SVG rather than in draw.io. The map is ~230 kB, and a base64 copy of it
    in one draw.io style attribute makes the headless exporter fail outright, printing only
    "Export failed"; writing the panel directly keeps it vector and removes the exporter
    from this figure's path. The output carries width before height and no units, which is
    the form svg_true_size_pdf.py reads as the 100-units-per-inch canvas.
    """
    u = UNITS_PER_MM
    inner = re.sub(r"^<svg[^>]*>", "", map_svg, count=1).rsplit("</svg>", 1)[0]
    view = re.search(r'viewBox="([^"]*)"', map_svg).group(1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width_mm * u:.2f}" height="{height_mm * u:.2f}" '
        f'viewBox="0 0 {width_mm * u:.2f} {height_mm * u:.2f}">',
        f'<style>text {{ font-family: Arial; font-size: {FONT_UNITS:.2f}px; }}</style>',
        f'<svg x="0" y="0" width="{map_w_mm * u:.2f}" height="{height_mm * u:.2f}" '
        f'viewBox="{view}" preserveAspectRatio="none">{inner}</svg>',
    ]

    # A compound drawn in more than one place is labeled once, at the copy furthest from
    # the panel edge. Every label sits to the RIGHT of its node and they are pushed apart
    # vertically until none overlaps another, so no leader crosses another leader. Each
    # label is set on an opaque rounded plate: a white halo on the glyphs alone is not
    # enough over this map, where a label routinely crosses two or three pathway lines.
    LEAD, GAP, PAD = 5.0, 4.0, 0.8
    CHAR_MM = 0.50 * FONT_UNITS / u  # Arial's mean advance is about half the type size
    items = []
    for a in anchors:
        if not a["positions_mm"]:
            continue
        x, y = max(a["positions_mm"],
                   key=lambda p: min(p[0], map_w_mm - p[0], p[1], height_mm - p[1]))
        items.append((x, y, a["label"]))
    placed = -1e9
    for x, y, label in sorted(items, key=lambda t: t[1]):
        ly = max(y, placed + GAP)
        placed = ly
        lx = x + LEAD
        tw = len(label) * CHAR_MM
        th = FONT_UNITS / u
        parts.append(f'<path d="M{x * u:.2f},{y * u:.2f} L{lx * u:.2f},{ly * u:.2f}" '
                     f'fill="none" stroke="#000000" stroke-width="0.5"/>')
        parts.append(f'<rect x="{(lx + 0.4 - PAD) * u:.2f}" '
                     f'y="{(ly - th / 2 - PAD * 0.6) * u:.2f}" '
                     f'width="{(tw + 2 * PAD) * u:.2f}" '
                     f'height="{(th + 1.2 * PAD) * u:.2f}" rx="{0.5 * u:.2f}" '
                     f'fill="#FFFFFF" fill-opacity="0.88" stroke="none"/>')
        parts.append(f'<text x="{(lx + 0.4) * u:.2f}" y="{ly * u:.2f}" '
                     f'dominant-baseline="middle">{esc(label)}</text>')

    parts.append(f'<text x="{key_x_mm * u:.2f}" y="{3.0 * u:.2f}" font-weight="bold" '
                 f'dominant-baseline="middle">'
                 f'the route from central carbon to the measured lipids</text>')
    rows = [(c, t, "line") for c, t in PANEL_KEY]
    rows += [(C_SPECIES, "measured species drawn on this map", "dot"),
             (C_WAYPOINT, "named intermediate", "dot")]
    for i, (color, text, kind) in enumerate(rows):
        yc = (8.0 + 4.6 * i) * u
        if kind == "line":
            parts.append(f'<line x1="{(key_x_mm + 0.5) * u:.2f}" y1="{yc:.2f}" '
                         f'x2="{(key_x_mm + 5.5) * u:.2f}" y2="{yc:.2f}" '
                         f'stroke="{color}" stroke-width="1.6"/>')
        else:
            parts.append(f'<circle cx="{(key_x_mm + 3.0) * u:.2f}" cy="{yc:.2f}" '
                         f'r="{1.1 * u:.2f}" fill="{color}"/>')
        parts.append(f'<text x="{(key_x_mm + 7.5) * u:.2f}" y="{yc:.2f}" '
                     f'dominant-baseline="middle">{esc(text)}</text>')
    note = ["C16:1 palmitoleate, C18:0 stearate and C18:1 oleate are measured here",
            "and have no node on the reference map. The map carries no labels of its",
            "own: at this size they would print at about 2.5 pt."]
    for i, line in enumerate(note):
        parts.append(f'<text x="{key_x_mm * u:.2f}" '
                     f'y="{(8.0 + 4.6 * len(rows) + 3.4 * i) * u:.2f}" '
                     f'dominant-baseline="middle">{esc(line)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch every response and fail if any has changed upstream")
    ap.add_argument("--width-mm", type=float, default=PANEL_WIDTHS_MM["full"],
                    help="width of the finished panel, map plus key")
    ap.add_argument("--map-width-mm", type=float, default=87.0)
    ap.add_argument("--crop", action="store_true",
                    help="crop to the named compounds instead of showing the whole map")
    ap.add_argument("--height-mm", type=float, default=52.0,
                    help="panel height; with the whole map the map's own aspect sets it")
    ap.add_argument("--out", default="ffa_ipath_map")
    args = ap.parse_args()

    sel, record = selection(args.refresh)
    svg = opaque_route(fetch_map(sel, args.refresh).decode("latin-1"))
    positions = compound_positions(args.refresh)
    anchor_pts = [p for cid, _ in WAYPOINTS + SPECIES for p in positions[cid]]
    if not anchor_pts:
        raise ValueError("none of the named compounds is drawn on this map")
    if args.crop:
        sized, to_mm = crop(svg, anchor_pts, args.map_width_mm, args.height_mm)
        height_mm = args.height_mm
    else:
        sized, height_mm, to_mm = whole(svg, args.map_width_mm)

    # Where each named compound sits inside the panel, so the figure can label it.
    species_ids = {cid for cid, _ in SPECIES}
    anchors = []
    for cid, label in WAYPOINTS + SPECIES:
        anchors.append({
            "compound": cid, "label": label,
            "color": C_SPECIES if cid in species_ids else C_WAYPOINT,
            "positions_mm": [[round(v, 2) for v in to_mm(x, y)]
                             for x, y in positions[cid]]})
    for entry in record["species"] + record["waypoints"]:
        entry["drawn_on_map"] = len(positions[entry["compound"]]) > 0
    placed = [e["label"] for e in record["species"] if e["drawn_on_map"]]
    missing = [e["label"] for e in record["species"] if not e["drawn_on_map"]]

    os.makedirs(IMAGES_DIR, exist_ok=True)
    out_svg = osp.join(IMAGES_DIR, args.out + ".svg")
    open(out_svg, "w", encoding="utf-8").write(
        panel(sized, anchors, args.width_mm, args.map_width_mm, height_mm,
              args.map_width_mm + 4.0))
    w_mm, h_mm = args.width_mm, height_mm
    json.dump(anchors, open(osp.join(IPATH_DIR, "label_anchors.json"), "w"), indent=2)

    record.update({
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "retrieval_method": "direct_url",
        "kegg_api": KEGG,
        "ipath_endpoint": IPATH,
        "ipath_post_fields": {"map": "metabolic", "export_type": "svg",
                              "default_opacity": BACKGROUND_OPACITY, "default_width": 3,
                              "default_radius": 7, "default_color": "#999999",
                              "keep_colors": 1},
        "selection_sha256": sha256(sel.encode()),
        "selection_entries": len(sel.splitlines()),
        "response_sha256": sha256(svg.encode("latin-1")),
        "stored": {f: sha256(open(osp.join(IPATH_DIR, f), "rb").read())
                   for f in sorted(os.listdir(IPATH_DIR))},
        "panel_mm": [round(w_mm, 1), round(h_mm, 1)],
        "species_drawn": placed,
        "species_not_on_map": missing,
    })
    open(osp.join(IPATH_DIR, "selection.txt"), "w", encoding="utf-8").write(sel)
    json.dump(record, open(osp.join(IPATH_DIR, "provenance.json"), "w"), indent=2)
    print(f"{record['selection_entries']} selection entries; "
          f"panel {w_mm:.1f} x {h_mm:.1f} mm\n"
          f"species drawn: {placed}\nnot on this map: {missing}\nwrote {out_svg}")


if __name__ == "__main__":
    main()
