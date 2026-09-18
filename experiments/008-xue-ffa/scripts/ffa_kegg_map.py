# experiments/008-xue-ffa/scripts/ffa_kegg_map.py
# [[experiments.008-xue-ffa.scripts.ffa_kegg_map]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/ffa_kegg_map
#
# The measured chemistry drawn on KEGG's global map of yeast metabolism, redrawn here as
# vector from KEGG's own coordinates, with the yeast genome-scale model (Yeast9) overlaid:
# what Yeast9 contains in gray, what KEGG draws for yeast but Yeast9 lacks fainter, the
# route from glycolysis and the citrate cycle to the measured fatty acids in color.
#
# WHY KEGG'S COORDINATES AND NOT A LAYOUT OF YEAST9. A coherent metabolic map, with the
# citrate cycle drawn as a circle and glycolysis as a spine, is a hand-drawn artifact;
# no layout algorithm produces one from a stoichiometric model, and Yeast9 ships no map.
# KEGG publishes its yeast global map (sce01100) as KGML with coordinates for every
# reaction line and compound circle, which is the drawing behind the reference map the
# earlier iPath3 panel borrowed, but current: it carries palmitoleate, stearate,
# palmitoyl-CoA, and the elongases and desaturase (ELO1/2/3, OLE1) that iPath3's older
# drawing lacked. Redrawn from the KGML, the map takes this document's palette, line
# weights and type, and every Yeast9 reaction and metabolite that KEGG places is marked.
#
# PROVENANCE. The KGML comes from the KEGG REST API (get/sce01100/kgml), the pathway gene
# lists from link/sce/<pathway>, both cached with their sha256 under results/kegg_map/ and
# refetched only with --refresh, which raises on upstream drift. Yeast9's gene, reaction
# and metabolite identifiers are read from the yeast-GEM 9.0.2 SBML file under DATA_ROOT
# and cached with that file's sha256. Nothing on the map is hand-placed: coordinates are
# KEGG's, membership is Yeast9's, the route is KEGG's gene lists for the named pathways.
#
# WHAT THE MAP STILL LACKS. Oleate (C00712) has no node on sce01100, nor do the C18
# acyl-CoAs, so oleate is attached along the yeast GEM's shortest path from a drawn
# compound, exactly as the iPath3 panel attached three species. And no map places
# Yeast9's transport, exchange and per-chain lipid reactions; the network panel beside
# this one carries that compartment and acyl-chain detail.
#
# THE CHECK THE FIGURE IS MOTIVATION FOR. The ten deleted genes are regulators, and the
# KGML says directly which yeast genes the map draws: none of the ten is among them,
# while all thirteen pathway genes are. The key and caption take these counts from the
# recorded data, not from a written sentence.

import argparse
import hashlib
import json
import os
import os.path as osp
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import map_labels as ml
from dotenv import load_dotenv
from ffa_ipath_map import (
    CURRENCY,
    MODULES,
    SPECIES,
    WAYPOINTS,
    gem_paths,
    gene_run,
    kegg_gene,
    kegg_pairs,
)
from ffa_network_overlay_panel import GENE_ORDER, TF_GENES

from torchcell.utils import PANEL_WIDTHS_MM

load_dotenv()
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
DATA_ROOT = os.getenv("DATA_ROOT")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "008-xue-ffa/results")
KEGG_DIR = osp.join(RESULTS_DIR, "kegg_map")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
YEAST9_XML = osp.join(DATA_ROOT, "data/torchcell/yeast-GEM/yeast-GEM-9.0.2/model/yeast-GEM.xml")

KEGG = "https://rest.kegg.jp"
MAP = "sce01100"

C_SPECIES = "#6C8EBF"
C_WAYPOINT = "#666666"
# Three tiers of the drawing. What Yeast9 contains is the mid gray; what KEGG draws for
# yeast (or for any organism) but Yeast9 lacks is the faint tier; the route is in color.
TIER = {
    "yeast9": {"color": "#666666", "opacity": 0.75, "width": 0.15, "r": 0.17},
    "other": {"color": "#BBBBBB", "opacity": 0.45, "width": 0.09, "r": 0.11},
}
W_ROUTE = 0.30  # mm, the colored route
# KEGG places the measured acids a few units apart at the end of the fatty acid comb, so
# the species nodes are kept small enough not to merge there.
R_SPECIES, R_WAYPOINT = 0.62, 0.52  # mm
_ = CURRENCY  # the GEM path search's currency list lives with gem_paths


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url, store, refresh):
    """GET `url`, caching the exact bytes under results/kegg_map/<store>.

    With --refresh the URL is fetched again and the run raises if the bytes differ from
    the stored ones: an upstream change is a new version of the source, not an update.
    """
    path = osp.join(KEGG_DIR, store)
    if osp.exists(path) and not refresh:
        return open(path, "rb").read()
    with urllib.request.urlopen(url, timeout=120) as r:
        body = r.read()
    if osp.exists(path):
        old = open(path, "rb").read()
        if old != body:
            raise RuntimeError(
                f"{url} changed upstream: stored sha256 {sha256(old)}, fetched "
                f"{sha256(body)}. Move {path} aside deliberately before accepting it.")
    os.makedirs(KEGG_DIR, exist_ok=True)
    open(path, "wb").write(body)
    return body


def yeast9_ids():
    """Gene, KEGG reaction and KEGG compound identifiers of Yeast9, cached by file hash.

    Reading the SBML takes about a minute; the cache is keyed on the model file's
    sha256, so a different model version is read afresh rather than reused.
    """
    path = osp.join(KEGG_DIR, "yeast9_ids.json")
    digest = sha256(open(YEAST9_XML, "rb").read())
    if osp.exists(path):
        cached = json.load(open(path))
        if cached["model_sha256"] == digest:
            return cached
    import cobra
    m = cobra.io.read_sbml_model(YEAST9_XML)

    def ids(ann, key):
        v = ann.get(key, [])
        return [v] if isinstance(v, str) else list(v)

    core = [r for r in m.reactions if len({x.compartment for x in r.metabolites}) == 1
            and not r.subsystem.startswith(("Exchange", "SLIME", "Transport"))]
    out = {
        "model_path": YEAST9_XML, "model_sha256": digest, "model_id": m.id,
        "n_reactions": len(m.reactions), "n_metabolites": len(m.metabolites),
        "n_genes": len(m.genes), "n_core_reactions": len(core),
        "genes": sorted(g.id for g in m.genes),
        "kegg_reactions": sorted({k for r in m.reactions for k in ids(r.annotation, "kegg.reaction")}),
        "kegg_compounds": sorted({k for x in m.metabolites for k in ids(x.annotation, "kegg.compound")}),
    }
    os.makedirs(KEGG_DIR, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    return out


def kgml(refresh):
    """The yeast global map: its lines, compounds and pathway labels, in KEGG units."""
    root = ET.fromstring(fetch(f"{KEGG}/get/{MAP}/kgml", f"{MAP}.kgml", refresh))
    lines, compounds = [], []
    for e in root.findall("entry"):
        g = e.find("graphics")
        if g is None:
            continue
        if g.get("type") == "line":
            c = [float(v) for v in g.get("coords").split(",")]
            lines.append({
                "id": e.get("id"), "type": e.get("type"),
                "orfs": [n.split(":")[1] for n in e.get("name").split() if n.startswith("sce:")],
                "reactions": [r.split(":")[1] for r in (e.get("reaction") or "").split()],
                "verts": list(zip(c[0::2], c[1::2])),
            })
        elif g.get("type") == "circle":
            compounds.append({
                "id": e.get("id"), "cid": e.get("name").split(":")[1],
                "x": float(g.get("x")), "y": float(g.get("y")),
            })
    return root.attrib, lines, compounds


def module_orfs(refresh):
    """ORF -> (module color, module label) from KEGG's gene lists, first module wins."""
    out = {}
    for pathway, label, color in MODULES:
        pairs = kegg_pairs(fetch(f"{KEGG}/link/sce/{pathway}", f"kegg_link_sce_{pathway}.tsv",
                                 refresh).decode())
        for _, orf in pairs:
            out.setdefault(orf, (color, label, pathway))
    return out


def gene_check(lines, refresh):
    """For the ten regulators and the thirteen pathway genes: how many lines draw them."""
    drawn = {}
    for ln in lines:
        for orf in ln["orfs"]:
            drawn[orf] = drawn.get(orf, 0) + 1
    rows = []
    for role, names in (("regulator", TF_GENES), ("pathway", GENE_ORDER)):
        for name in names:
            orf = kegg_gene(name, refresh)
            rows.append({"gene": name, "orf": orf, "role": role, "map_lines": drawn.get(orf, 0)})
    return rows


KEY_ROW, KEY_NOTE_ROW = 3.4, 2.9
PANEL_KEY = [
    ("#9673A6", "glycolysis and gluconeogenesis"),
    ("#D6B656", "pyruvate metabolism"),
    ("#B85450", "citrate cycle"),
    ("#D79B00", "fatty acid biosynthesis, elongation and desaturation"),
    ("#666666", "fatty acid degradation"),
]
KEY_EXTRA = 6  # species (2 kinds), intermediate, two tiers, the link
KEY_NOTE_LINES = 3


def key_height_mm():
    return 8.0 + KEY_ROW * (len(PANEL_KEY) + KEY_EXTRA) + KEY_NOTE_ROW * KEY_NOTE_LINES


def panel(attrs, lines, compounds, y9, mods, gene_rows, attachments, width_mm, map_w_mm,
          key_x_mm, orf_to_name):
    """The finished panel as SVG: the redrawn map with its labels, and the key beside it.

    Written directly as SVG in the 100-units-per-inch canvas (width before height, no
    units), the form svg_true_size_pdf.py reads.
    """
    u = ml.UNITS_PER_MM
    xs = [x for ln in lines for x, _ in ln["verts"]] + [c["x"] for c in compounds]
    ys = [y for ln in lines for _, y in ln["verts"]] + [c["y"] for c in compounds]
    m = 25.0  # KEGG units of margin around the drawing
    x0, y0 = min(xs) - m, min(ys) - m
    scale = map_w_mm / (max(xs) + m - x0)
    map_h_mm = (max(ys) + m - y0) * scale
    to_mm = lambda x, y: ((x - x0) * scale, (y - y0) * scale)  # noqa: E731
    panel_h_mm = max(map_h_mm, key_height_mm())

    y9_genes, y9_rxns, y9_cpds = set(y9["genes"]), set(y9["kegg_reactions"]), set(y9["kegg_compounds"])
    pts = ml.Ink()
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width_mm * u:.2f}" height="{panel_h_mm * u:.2f}" '
        f'viewBox="0 0 {width_mm * u:.2f} {panel_h_mm * u:.2f}">',
        f'<style>text {{ font-family: Arial; font-size: {ml.FONT_UNITS:.2f}px; }}</style>',
    ]

    # Reaction lines in three tiers, faint first so the route is drawn on top.
    tiers = {"other": [], "yeast9": [], "route": []}
    counts = {"lines": len(lines), "lines_yeast9": 0, "lines_route": 0, "lines_yeast_gene": 0}
    for ln in lines:
        verts = [to_mm(x, y) for x, y in ln["verts"]]
        in_y9 = any(o in y9_genes for o in ln["orfs"]) or any(r in y9_rxns for r in ln["reactions"])
        route = next(((mods[o][0], mods[o][2]) for o in ln["orfs"] if o in mods), None)
        counts["lines_yeast_gene"] += bool(ln["orfs"])
        counts["lines_yeast9"] += in_y9
        counts["lines_route"] += route is not None
        d = "M" + " L".join(f"{x * u:.2f},{y * u:.2f}" for x, y in verts)
        if route:
            tiers["route"].append((d, route[0]))
            pts.add_polyline(verts, 1.0)
        elif in_y9:
            tiers["yeast9"].append(d)
            pts.add_polyline(verts, ml.Ink.BACKGROUND_WEIGHT * 2)
        else:
            tiers["other"].append(d)
            pts.add_polyline(verts, ml.Ink.BACKGROUND_WEIGHT)
    for tier in ("other", "yeast9"):
        t = TIER[tier]
        parts.append(f'<g fill="none" stroke="{t["color"]}" stroke-opacity="{t["opacity"]}" '
                     f'stroke-width="{t["width"] * u:.2f}" stroke-linecap="round" '
                     f'stroke-linejoin="round">')
        parts += [f'<path d="{d}"/>' for d in tiers[tier]]
        parts.append("</g>")
    parts.append(f'<g fill="none" stroke-width="{W_ROUTE * u:.2f}" stroke-linecap="round" '
                 f'stroke-linejoin="round">')
    parts += [f'<path d="{d}" stroke="{c}"/>' for d, c in tiers["route"]]
    parts.append("</g>")

    # Compound circles, the named ones large. A compound drawn in more than one place
    # keeps its large node only at the labeled copy.
    named = {cid: (label, name) for cid, label, name in WAYPOINTS + SPECIES}
    copies = {}
    for c in compounds:
        if c["cid"] in named:
            copies.setdefault(named[c["cid"]][1], []).append(to_mm(c["x"], c["y"]))
    node_at = {name: max(ps, key=lambda p: min(p[0], map_w_mm - p[0], p[1], map_h_mm - p[1]))
               for name, ps in copies.items()}
    labels = {name: label for _, label, name in WAYPOINTS + SPECIES}
    species_names = {name for _, _, name in SPECIES}
    counts["compounds"] = len(compounds)
    counts["compounds_yeast9"] = sum(c["cid"] in y9_cpds for c in compounds)

    # The attached species: rings in the window whose rings and links cover the least ink.
    node_r = R_SPECIES
    col_w = ml.column_width(attachments, labels)
    layout, col_h = ml.column_layout(attachments, labels, col_w, node_r)
    bounds = (0.3, 0.3, map_w_mm - 0.3, map_h_mm - 0.3)
    inset = (bounds[0] + 2.0, bounds[1] + 2.0, bounds[2] - 2.0, bounds[3] - 2.0)
    wx, wy, side, chosen = ml.attachment_window(pts, attachments, copies, list(node_at.values()),
                                                col_w, col_h, layout, inset)
    rings, _ = layout(wx, wy, side)
    print(f"attachment window at ({wx:.1f}, {wy:.1f}) mm, labels {side} of the rings")
    for a in attachments:
        node_at[a["species"]] = rings[a["species"]]
        copies[a["species"]] = [rings[a["species"]]]
    resolved = []
    for a, (c, src) in zip(attachments, chosen):
        node_at[c["anchor"]] = src  # the label goes on the copy the link starts from
        resolved.append({"species": a["species"], "anchor": c["anchor"], "path": c["path"],
                         "genes": c["genes"], "candidates": [x["anchor"] for x in a["candidates"]]})

    for c in compounds:
        x, y = to_mm(c["x"], c["y"])
        name = named.get(c["cid"], (None, None))[1]
        if name is not None and node_at.get(name) == (x, y):
            color = C_SPECIES if name in species_names else C_WAYPOINT
            r = R_SPECIES if name in species_names else R_WAYPOINT
            parts.append(f'<circle cx="{x * u:.2f}" cy="{y * u:.2f}" r="{r * u:.2f}" fill="{color}"/>')
            pts.add(x, y, 2.0)
            continue
        t = TIER["yeast9" if c["cid"] in y9_cpds else "other"]
        parts.append(f'<circle cx="{x * u:.2f}" cy="{y * u:.2f}" r="{t["r"] * u:.2f}" '
                     f'fill="{t["color"]}" fill-opacity="{t["opacity"]}"/>')
        pts.add(x, y, ml.Ink.BACKGROUND_WEIGHT)

    placer = ml.Placer(pts, node_at, labels, bounds)
    for a, (c, src) in zip(attachments, chosen):
        dst = node_at[a["species"]]
        parts.append(f'<path d="M{src[0] * u:.2f},{src[1] * u:.2f} L{dst[0] * u:.2f},'
                     f'{dst[1] * u:.2f}" fill="none" stroke="#000000" stroke-width="0.6" '
                     f'stroke-dasharray="2.2,1.6"/>')
        placer.add_line(src, dst)
        pts.add(*dst, 2.0)

    attached = {a["species"] for a in attachments}
    # Labels: the attached species first (beside their rings), then the drawn species,
    # then the intermediates, each group in its best order.
    if attachments:
        placer.group([a["species"] for a in attachments],
                     directions=ml.RIGHT if side == "right" else ml.LEFT,
                     sub={a["species"]: gene_run(a["genes"]) for a in resolved}, attached=attached)
    rest = [n for n in node_at if n not in attached]
    placer.group([n for n in rest if n in species_names])
    placer.group([n for n in rest if n not in species_names])
    parts += placer.svg
    # Rings go over the leaders, which start at the ring's center.
    for name in attached:
        x, y = node_at[name]
        parts.append(f'<circle cx="{x * u:.2f}" cy="{y * u:.2f}" r="{node_r * u:.2f}" '
                     f'fill="#FFFFFF" stroke="{C_SPECIES}" stroke-width="1.6"/>')

    # The key.
    parts.append(f'<text x="{key_x_mm * u:.2f}" y="{3.0 * u:.2f}" font-weight="bold" '
                 f'dominant-baseline="middle">the route from central carbon to the measured '
                 f'lipids, on KEGG\'s yeast map</text>')
    rows = [(c, t, "route") for c, t in PANEL_KEY]
    rows += [(C_SPECIES, "measured species, a node of the map", "dot"),
             (C_SPECIES, "measured species attached from the yeast GEM", "ring"),
             (C_WAYPOINT, "named intermediate", "dot"),
             ("yeast9", "reaction or compound in Yeast9", "tier"),
             ("other", "drawn by KEGG, not in Yeast9", "tier"),
             ("#000000", "the GEM's shortest path from a drawn compound; its genes under the name", "dash")]
    if len(rows) != len(PANEL_KEY) + KEY_EXTRA:
        raise ValueError(f"key has {len(rows)} rows, expected {len(PANEL_KEY) + KEY_EXTRA}")
    for i, (color, text, kind) in enumerate(rows):
        yc = (8.0 + KEY_ROW * i) * u
        x1, x2 = (key_x_mm + 0.5) * u, (key_x_mm + 5.5) * u
        if kind == "route":
            parts.append(f'<line x1="{x1:.2f}" y1="{yc:.2f}" x2="{x2:.2f}" y2="{yc:.2f}" '
                         f'stroke="{color}" stroke-width="{W_ROUTE * u:.2f}"/>')
        elif kind == "tier":
            t = TIER[color]
            parts.append(f'<line x1="{x1:.2f}" y1="{yc:.2f}" x2="{x2:.2f}" y2="{yc:.2f}" '
                         f'stroke="{t["color"]}" stroke-opacity="{t["opacity"]}" '
                         f'stroke-width="{t["width"] * u:.2f}"/>')
            parts.append(f'<circle cx="{(key_x_mm + 3.0) * u:.2f}" cy="{yc:.2f}" '
                         f'r="{t["r"] * u:.2f}" fill="{t["color"]}" fill-opacity="{t["opacity"]}"/>')
        elif kind == "dash":
            parts.append(f'<line x1="{x1:.2f}" y1="{yc:.2f}" x2="{x2:.2f}" y2="{yc:.2f}" '
                         f'stroke="{color}" stroke-width="0.6" stroke-dasharray="2.2,1.6"/>')
        elif kind == "dot":
            parts.append(f'<circle cx="{(key_x_mm + 3.0) * u:.2f}" cy="{yc:.2f}" '
                         f'r="{1.0 * u:.2f}" fill="{color}"/>')
        else:
            parts.append(f'<circle cx="{(key_x_mm + 3.0) * u:.2f}" cy="{yc:.2f}" '
                         f'r="{0.9 * u:.2f}" fill="#FFFFFF" stroke="{color}" stroke-width="1.6"/>')
        parts.append(f'<text x="{(key_x_mm + 7.5) * u:.2f}" y="{yc:.2f}" '
                     f'dominant-baseline="middle">{ml.esc(text)}</text>')

    drawn_orfs = {o for ln in lines for o in ln["orfs"]}
    counts["yeast_genes_drawn"] = len(drawn_orfs)
    counts["yeast_genes_drawn_in_yeast9"] = len(drawn_orfs & y9_genes)
    n_reg = sum(r["role"] == "regulator" for r in gene_rows)
    reg_on = sum(r["role"] == "regulator" and r["map_lines"] > 0 for r in gene_rows)
    n_pw = sum(r["role"] == "pathway" for r in gene_rows)
    pw_on = sum(r["role"] == "pathway" and r["map_lines"] > 0 for r in gene_rows)
    if resolved:
        att = " and ".join(f"{labels[a['species']].split(' ', 1)[1]} from {labels[a['anchor']].split(' ', 1)[-1]}"
                           for a in resolved)
        line2 = (f"{att[0].upper() + att[1:]} is attached along the GEM's shortest path: "
                 f"the map has no node for it.")
    else:
        line2 = "All five measured species are nodes of the map."
    note = [
        f"The map draws {counts['yeast_genes_drawn']} yeast genes, {counts['yeast_genes_drawn_in_yeast9']} "
        f"in Yeast9; {counts['lines_yeast9']} of its {counts['lines']} reaction lines are in Yeast9.",
        line2,
        f"Of the {n_reg} deleted regulators, {reg_on} draw a reaction here; of the {n_pw} pathway genes "
        f"of b, {pw_on} do.",
    ]
    if len(note) != KEY_NOTE_LINES:
        raise ValueError(f"note has {len(note)} lines, KEY_NOTE_LINES says {KEY_NOTE_LINES}")
    for i, text in enumerate(note):
        parts.append(f'<text x="{key_x_mm * u:.2f}" '
                     f'y="{(8.0 + KEY_ROW * len(rows) + KEY_NOTE_ROW * i) * u:.2f}" '
                     f'dominant-baseline="middle">{ml.esc(text)}</text>')
    parts.append("</svg>")
    return "\n".join(parts), placer.records, resolved, counts, (map_w_mm, map_h_mm, panel_h_mm)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch every KEGG response and fail if any has changed upstream")
    ap.add_argument("--width-mm", type=float, default=PANEL_WIDTHS_MM["full"])
    # KEGG's yeast map is 1.55 times wider than tall; 83 mm keeps the panel at 54 mm so
    # the figure, with the network panel under it, stays inside the 170 mm page cap.
    ap.add_argument("--map-width-mm", type=float, default=83.0)
    ap.add_argument("--out", default="ffa_kegg_map")
    args = ap.parse_args()

    attrs, lines, compounds = kgml(args.refresh)
    y9 = yeast9_ids()
    mods = module_orfs(args.refresh)
    gene_rows = gene_check(lines, args.refresh)
    orf_to_name = {r["orf"]: r["gene"] for r in gene_rows if r["role"] == "pathway"}

    cids = {c["cid"] for c in compounds}
    drawn = [name for cid, _, name in WAYPOINTS + SPECIES if cid in cids]
    undrawn = [name for cid, _, name in SPECIES if cid not in cids]
    attachments = gem_paths(drawn, undrawn, orf_to_name)

    svg, records, resolved, counts, dims = panel(
        attrs, lines, compounds, y9, mods, gene_rows, attachments, args.width_mm,
        args.map_width_mm, args.map_width_mm + 4.0, orf_to_name)

    os.makedirs(IMAGES_DIR, exist_ok=True)
    out_svg = osp.join(IMAGES_DIR, args.out + ".svg")
    open(out_svg, "w", encoding="utf-8").write(svg)
    json.dump(records, open(osp.join(KEGG_DIR, "label_anchors.json"), "w"), indent=2)
    record = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "retrieval_method": "direct_url",
        "kegg_api": KEGG, "map": MAP, "map_title": attrs.get("title"),
        "kgml_url": f"{KEGG}/get/{MAP}/kgml",
        "kgml_sha256": sha256(open(osp.join(KEGG_DIR, f"{MAP}.kgml"), "rb").read()),
        "yeast9": {k: y9[k] for k in ("model_path", "model_sha256", "model_id", "n_reactions",
                                      "n_metabolites", "n_genes", "n_core_reactions")},
        "modules": [{"pathway": p, "label": lab, "color": c} for p, lab, c in MODULES],
        "counts": counts,
        "species_drawn": drawn, "species_attached": resolved,
        "genes": gene_rows,
        "panel_mm": [round(args.width_mm, 1), round(dims[2], 1)],
        "map_mm": [round(dims[0], 1), round(dims[1], 1)],
    }
    record["stored"] = {f: sha256(open(osp.join(KEGG_DIR, f), "rb").read())
                        for f in sorted(os.listdir(KEGG_DIR))}
    json.dump(record, open(osp.join(KEGG_DIR, "provenance.json"), "w"), indent=2)

    print(f"panel {args.width_mm:.1f} x {dims[2]:.1f} mm, map {dims[0]:.1f} x {dims[1]:.1f} mm")
    print(f"drawn by the map: {drawn}")
    for a in resolved:
        print(f"attached: {a['species']} <- {a['anchor']} via {' > '.join(a['path'][1:-1]) or '-'} "
              f"[{gene_run(a['genes'])}]")
    print("counts:", counts)
    for r in gene_rows:
        print(f"  {r['role']:9s} {r['gene']:5s} {r['orf']:8s} lines: {r['map_lines']}")
    print(f"wrote {out_svg}")


if __name__ == "__main__":
    main()
