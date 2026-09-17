# experiments/008-xue-ffa/scripts/ffa_ipath_map.py
# [[experiments.008-xue-ffa.scripts.ffa_ipath_map]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/ffa_ipath_map
#
# The measured chemistry drawn on the global metabolic map: the route from glycolysis and
# the citrate cycle out to the five fatty acid species this study measures, highlighted on
# iPath3's reference map, the whole map shown with a key beside it.
#
# WHAT THIS IS FOR. The network figure shows ten transcription factors upstream of a
# thirteen-gene pathway, which says nothing about where that pathway sits in the cell's
# metabolism or how far the measured species are from central carbon. This panel answers
# that in one picture: the whole map in gray, the route in color, the five measured species
# as the only large nodes on it. It also carries the check the figure is motivation for: the
# ten deleted genes are regulators, and none of them draws a reaction on this map, while
# every one of the thirteen pathway genes does.
#
# THERE IS NO iPath PYTHON PACKAGE. iPath3 is a web service; the client is this file. The
# service takes a newline-separated selection of KEGG identifiers with per-entry color,
# width or radius and opacity, and returns the reference map as SVG with those entries
# restyled. Nothing else is sent.
#
# PROVENANCE. Everything selected comes from the KEGG REST API, never from a hand-written
# list of KO numbers: the yeast genes of each pathway from link/sce/<pathway>, their KEGG
# Orthology ids from link/ko/sce, the systematic name of each named gene from
# find/sce/<name>. Every response, the exact selection posted, and the returned SVG are
# written to results/ipath/ with their sha256 and the command that retrieved them, so the
# panel rebuilds from the stored responses with no network and the run fails loudly if an
# upstream response has changed rather than following the drift. Re-running with --refresh
# re-fetches; the default reads the stored copies.
#
# TWO OF THE FIVE MEASURED SPECIES HAVE A NODE ON THIS MAP, AND THE OTHER THREE ARE
# ATTACHED FROM THE YEAST GEM. Myristate and palmitate are drawn by the reference map;
# palmitoleate, stearate and oleate are not, neither as free acids nor as their acyl-CoA
# thioesters (probed one identifier at a time, below). KEGG's own global map lists
# palmitoleate and stearate, so this is the layout of iPath3's drawing, not a gap in KEGG.
# The network panel beside this one has all five because its species come from the yeast
# genome-scale model. So that the two panels show the same five species, the three the map
# lacks are attached here along the SHORTEST PATH OF THAT SAME MODEL from a compound the
# map does draw, with the genes of the path's reactions named on the link. The path is
# computed from the pathway subgraph the network panel reads, not written by hand, and an
# attached species is drawn as an open ring so it cannot be mistaken for a node of the map.
#
# OPACITY. The service applies one opacity to every entry, so a per-entry O flag in the
# selection is ignored. The background is therefore requested faint and the highlighted
# elements are set opaque afterwards, in the returned SVG, by the one rule that an element
# carrying a highlight color is part of the route.
#
# LABELS ARE PLACED WHERE THE MAP HAS THE LEAST INK. Every element of the drawing is
# sampled into points, weighted by its opacity, and a label tries eight directions at two
# distances from its node; the placement whose plate and leader cover the least ink wins,
# and no plate may overlap another. The attached species go into the emptiest window of the
# map near the compounds they attach to, found the same way.

import argparse
import hashlib
import itertools
import json
import math
import os
import os.path as osp
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import networkx as nx
from dotenv import load_dotenv

from ffa_network_overlay_panel import GENE_ORDER, TF_GENES
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

# The five measured species: the KEGG compound the titer is of, the label, and the name
# the yeast GEM gives the same compound (which is how an attachment path is found).
SPECIES = [
    ("C06424", "C14:0 myristate", "myristate"),
    ("C00249", "C16:0 palmitate", "palmitate"),
    ("C08362", "C16:1 palmitoleate", "palmitoleate"),
    ("C01530", "C18:0 stearate", "stearate"),
    ("C00712", "C18:1 oleate", "oleate"),
]
# The acyl-CoA thioester of each species, probed so the caption can say the map carries
# neither form rather than only that the free acid is absent.
SPECIES_COA = [
    ("C02593", "myristoyl-CoA"),
    ("C00154", "palmitoyl-CoA"),
    ("C21072", "palmitoleoyl-CoA"),
    ("C00412", "stearoyl-CoA"),
    ("C00510", "oleoyl-CoA"),
]
# Waypoints, labeled so a reader can find the route without reading every node, with the
# GEM name of each. Palmitoyl-CoA is on the map and is where the GEM's desaturation
# starts, which is what makes it the anchor of the palmitoleate attachment.
WAYPOINTS = [
    ("C00022", "pyruvate", "pyruvate"),
    ("C00024", "acetyl-CoA", "acetyl-CoA"),
    ("C00158", "citrate", "citrate"),
    ("C00083", "malonyl-CoA", "malonyl-CoA"),
    ("C00154", "palmitoyl-CoA", "palmitoyl-CoA"),
]
# Metabolites that take part in nearly every reaction and would make every path one step.
CURRENCY = {"ATP", "ADP", "AMP", "coenzyme A", "H+", "H2O", "NAD", "NADH", "NADP(+)",
            "NADPH", "oxygen", "hydrogen peroxide", "carbon dioxide", "bicarbonate",
            "phosphate", "diphosphate"}

C_SPECIES = "#6C8EBF"
C_WAYPOINT = "#666666"
W_REACTION = 14
R_WAYPOINT = 30
R_SPECIES = 36
BACKGROUND_OPACITY = 0.28  # the reference map behind the route

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


def kegg_gene(name, refresh):
    """The systematic name KEGG gives the yeast gene called `name`.

    find/sce/<name> matches the string anywhere in an entry, so the response is filtered
    to the entries whose own symbol list carries the name. KEGG lists the standard name
    first, so the entry whose FIRST symbol is `name` wins: TFC7 is the standard name of
    YOR110W and an alias of YNL039W (BDP1), and it is YOR110W that is meant. Failing a
    standard-name match, a unique alias match is accepted.
    """
    text = fetch(f"{KEGG}/find/sce/{name}", f"kegg_find_sce_{name}.tsv", refresh).decode()
    standard, alias = [], []
    for line in text.strip().splitlines():
        entry, desc = line.split("\t")
        symbols = [s.strip() for s in desc.split(";", 1)[0].split(",")]
        if symbols[0] == name:
            standard.append(entry.split(":", 1)[1])
        elif name in symbols:
            alias.append(entry.split(":", 1)[1])
    hits = standard if standard else alias
    if len(hits) != 1:
        raise ValueError(f"KEGG find/sce/{name}: expected one entry naming {name}, "
                         f"got standard {standard}, alias {alias}")
    return hits[0]


def gene_to_ko(refresh):
    out = {}
    for orf, ko in kegg_pairs(fetch(f"{KEGG}/link/ko/sce", "kegg_link_ko_sce.tsv",
                                    refresh).decode()):
        out.setdefault(orf, ko)
    return out


def selection(refresh):
    """The iPath selection lines, and the record of what produced them."""
    orf_to_ko = gene_to_ko(refresh)
    lines, record = [], {"modules": [], "species": [], "waypoints": []}
    claimed = set()
    for pathway, label, color in MODULES:
        pairs = kegg_pairs(fetch(f"{KEGG}/link/sce/{pathway}",
                                 f"kegg_link_sce_{pathway}.tsv", refresh).decode())
        kos = sorted({orf_to_ko[orf] for _, orf in pairs if orf in orf_to_ko})
        new = [k for k in kos if k not in claimed]
        claimed.update(new)
        for ko in new:
            lines.append(f"{ko} {color} W{W_REACTION}")
        record["modules"].append({"pathway": pathway, "label": label, "color": color,
                                  "genes": len(pairs), "ko_selected": len(new)})

    for cid, label, _ in WAYPOINTS:
        lines.append(f"{cid} {C_WAYPOINT} W{R_WAYPOINT}")
        record["waypoints"].append({"compound": cid, "label": label})
    for cid, label, _ in SPECIES:
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
PROBES_FILE = "ipath_probes.json"


def probes(identifiers, refresh):
    """What the reference map draws for each identifier, posted on its own.

    Several entries share a highlight color, so their elements cannot be told apart in the
    finished map. Each is therefore posted alone in a color nothing else uses, and only
    what came back in that color is kept: the node centers in map units, and the count of
    reaction lines. An identifier the map does not carry comes back with nothing, which is
    the honest answer and is what the caption reports. Results are cached per identifier.
    """
    path = osp.join(IPATH_DIR, PROBES_FILE)
    store = json.load(open(path)) if osp.exists(path) else {}
    for ident in identifiers:
        if ident in store and not refresh:
            continue
        form = urllib.parse.urlencode({
            "selection": f"{ident} {PROBE_COLOR} W30", "map": "metabolic",
            "export_type": "svg", "default_opacity": 0.1,
        }).encode()
        with urllib.request.urlopen(urllib.request.Request(IPATH, data=form),
                                    timeout=180) as r:
            body = r.read().decode("latin-1")
        color = PROBE_COLOR[1:]
        nodes = [[float(re.search(r"cx='(-?[\d.]+)'", el).group(1)),
                  float(re.search(r"cy='(-?[\d.]+)'", el).group(1))]
                 for el in re.findall(r"<ellipse[^>]*" + color + r"[^>]*>", body)]
        lines = len(re.findall(r"<path[^>]*" + color + r"[^>]*>", body))
        store[ident] = {"nodes": nodes, "reaction_lines": lines}
    os.makedirs(IPATH_DIR, exist_ok=True)
    json.dump(store, open(path, "w"), indent=2)
    return {i: store[i] for i in identifiers}


def genes_on_map(refresh):
    """For the ten deleted regulators and the thirteen pathway genes: what the map draws.

    KEGG gives each gene its orthology group; the map is drawn by orthology group; so a
    gene is on the map exactly when its group draws at least one element there.
    """
    orf_to_ko = gene_to_ko(refresh)
    rows = []
    for role, names in (("regulator", TF_GENES), ("pathway", GENE_ORDER)):
        for name in names:
            orf = kegg_gene(name, refresh)
            if orf not in orf_to_ko:
                raise ValueError(f"{name} ({orf}) has no KEGG orthology group")
            rows.append({"gene": name, "orf": orf, "ko": orf_to_ko[orf], "role": role})
    hit = probes([r["ko"] for r in rows], refresh)
    for r in rows:
        p = hit[r["ko"]]
        r["map_elements"] = len(p["nodes"]) + p["reaction_lines"]
    return rows


def gem_paths(drawn, targets, orf_to_name):
    """How the yeast GEM joins each undrawn species to something the map does draw.

    The pathway subgraph is collapsed to metabolites by name (compartments merged,
    currency metabolites dropped), with every substrate of a reaction joined to every
    product of it by an edge carrying that reaction's genes. Substrates of one reaction
    are not joined to each other: malonyl-CoA and stearoyl-CoA are both substrates of the
    ELO2/ELO3 elongation, and joining them would put ELO2/3 on the FAS1/FAS2 step that
    makes stearoyl-CoA. Each target then takes the shortest path from any drawn compound,
    or from a species already attached, so the attachments form one tree rather than
    three parallel links. Returned per target: every anchor at the shortest length and
    the genes of each path in order of first appearance.
    """
    G = nx.read_graphml(osp.join(RESULTS_DIR, "ffa_bipartite_network.graphml"))
    H = nx.Graph()
    for r in [n for n in G if G.nodes[n].get("node_type") == "reaction"]:
        genes = {n for n in G.predecessors(r) if G.nodes[n].get("node_type") == "gene"}
        subs = {G.nodes[n]["name"] for n in G.predecessors(r)
                if G.nodes[n].get("node_type") == "metabolite"
                and G.nodes[n]["name"] not in CURRENCY}
        prods = {G.nodes[n]["name"] for n in G.successors(r)
                 if G.nodes[n].get("node_type") == "metabolite"
                 and G.nodes[n]["name"] not in CURRENCY}
        for a, b in itertools.product(sorted(subs), sorted(prods)):
            if a == b:
                continue
            H.add_edge(a, b)
            H.edges[a, b].setdefault("genes", set()).update(genes)

    out = []
    for t in targets:
        # Every anchor at the minimum path length is a candidate; the panel takes the one
        # nearest the ring it draws, so equal-length paths resolve by geometry rather than
        # by list order. Species already attached are candidates too, which is what lets
        # oleate hang from stearate instead of running its own link to the map. Drawn
        # compounds the pathway subgraph does not reach (citrate, pyruvate) cannot anchor
        # anything and are skipped.
        sources = [a["species"] for a in out] + [n for n in drawn if n in H]
        paths = {s: nx.shortest_path(H, s, t) for s in sources}
        shortest = min(len(p) for p in paths.values())
        candidates = []
        for s, path in paths.items():
            if len(path) != shortest:
                continue
            genes = []
            for a, b in zip(path, path[1:]):
                for orf in sorted(H.edges[a, b]["genes"]):
                    if orf_to_name[orf] not in genes:
                        genes.append(orf_to_name[orf])
            candidates.append({"anchor": s, "path": path, "genes": genes})
        out.append({"species": t, "candidates": candidates})
    return out


def gene_run(names):
    """FAA1, FAA2, FAA4 -> FAA1/2/4, keeping first-appearance order of the families."""
    fam = {}
    for n in names:
        m = re.match(r"^([A-Z]+)(\d+)$", n)
        fam.setdefault(m.group(1), []).append(m.group(2))
    return ", ".join(f"{k}{'/'.join(sorted(v, key=int))}" for k, v in fam.items())


def prune(svg, x0, y0, w, h):
    """Drop every element that lies outside the box, and the groups left empty.

    ALL OF THE MAP'S OWN TEXT IS DROPPED, with the colored pill each label sits on. At
    100 mm the reference map's largest label prints at about 2.5 pt, under Nature's 5 pt
    floor, so none of it is type this figure is allowed to set. The key beside the panel
    names the modules instead, at 5.98 pt, and the compounds are labeled here. What is
    left is the drawing: reaction lines and compound nodes.
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


def demote_copies(map_svg, points_mm, to_mm):
    """Return the map with the nodes at `points_mm` restyled as background nodes.

    The nodes are found by their map-unit centers, which the probes recorded and to_mm
    converts one way; the match is on the rounded centers so float formatting cannot
    miss. Each is given the background's fill, radius and opacity.
    """
    targets = set()
    for m in re.finditer(r'<ellipse[^>]*cx="(-?[\d.]+)"[^>]*cy="(-?[\d.]+)"[^>]*>', map_svg):
        p = to_mm(float(m.group(1)), float(m.group(2)))
        if any(math.hypot(p[0] - q[0], p[1] - q[1]) < 0.05 for q in points_mm):
            targets.add(m.group(0))
    if len(targets) != len(points_mm):
        raise ValueError(f"expected {len(points_mm)} nodes to demote, matched {len(targets)}")
    out = map_svg
    for el in targets:
        new = re.sub(r'fill="#[0-9A-Fa-f]{6}"', 'fill="#999999"', el)
        new = re.sub(r'rx="[\d.]+"', 'rx="7"', new)
        new = re.sub(r'ry="[\d.]+"', 'ry="7"', new)
        new = re.sub(r"opacity:\s*[\d.]+", f"opacity: {BACKGROUND_OPACITY}", new)
        out = out.replace(el, new)
    return out


def whole(svg, width_mm):
    """The whole drawing at `width_mm`, keeping its own aspect ratio.

    Returns the resized SVG, the height it takes, and a map from map units to panel
    millimetres.
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


def segment_points(a, b, step_mm=0.4):
    (ax, ay), (bx, by) = a, b
    k = max(1, int(math.hypot(bx - ax, by - ay) / step_mm))
    return [(ax + (bx - ax) * i / k, ay + (by - ay) * i / k) for i in range(k + 1)]


class Ink:
    """How much of the drawing lies under a rectangle or along a line, in panel mm.

    Every drawn element is sampled into points, the route weighted 1 and the faint
    background BACKGROUND_WEIGHT, and the points are binned on a CELL mm grid so a query
    sums a few cells rather than scanning thirty thousand points. The background weighs
    much less than its opacity would say: a label over faint gray lines costs the reader
    nothing, a label over the route hides the content of the panel. A path contributes
    points along each of its segments every STEP mm, so a long straight line counts
    along its whole length and not only at its ends.
    """
    CELL, STEP = 0.5, 0.4
    BACKGROUND_WEIGHT = 0.08

    def __init__(self, map_svg, to_mm):
        self.cells = {}
        for m in ELEMENT.finditer(map_svg):
            el = m.group(0)
            w = 1.0 if is_highlighted(el) else self.BACKGROUND_WEIGHT
            d = re.search(r'\bd="([^"]*)"', el)
            if d:
                n = [float(v) for v in NUMBER.findall(d.group(1))]
                verts = [to_mm(x, y) for x, y in zip(n[0::2], n[1::2])]
                for a, b in zip(verts, verts[1:]):
                    for x, y in segment_points(a, b, self.STEP):
                        self.add(x, y, w)
                continue
            cx = re.search(r'cx="(-?[\d.]+)"', el)
            cy = re.search(r'cy="(-?[\d.]+)"', el)
            if cx and cy:
                self.add(*to_mm(float(cx.group(1)), float(cy.group(1))), w)

    def add(self, x, y, w):
        key = (int(x // self.CELL), int(y // self.CELL))
        self.cells[key] = self.cells.get(key, 0.0) + w

    def add_line(self, a, b, w=1.0):
        for x, y in segment_points(a, b, self.STEP):
            self.add(x, y, w)

    def rect(self, x, y, w, h, pad=0.3):
        i0, i1 = int((x - pad) // self.CELL), int((x + w + pad) // self.CELL)
        j0, j1 = int((y - pad) // self.CELL), int((y + h + pad) // self.CELL)
        return sum(self.cells.get((i, j), 0.0)
                   for i in range(i0, i1 + 1) for j in range(j0, j1 + 1))

    def line(self, a, b, halo=0.35):
        seen, total = set(), 0.0
        r = int(math.ceil(halo / self.CELL))
        for x, y in segment_points(a, b, self.STEP):
            ci, cj = int(x // self.CELL), int(y // self.CELL)
            for i in range(ci - r, ci + r + 1):
                for j in range(cj - r, cj + r + 1):
                    if (i, j) not in seen:
                        seen.add((i, j))
                        total += self.cells.get((i, j), 0.0)
        return total


def overlaps(r, others, gap=0.4):
    x, y, w, h = r
    return any(x < ox + ow + gap and ox < x + w + gap and y < oy + oh + gap and oy < y + h + gap
               for ox, oy, ow, oh in others)


def inside(p, r, gap=0.3):
    x, y, w, h = r
    return x - gap <= p[0] <= x + w + gap and y - gap <= p[1] <= y + h + gap


def crosses(segment, rects):
    """Whether a leader or link passes through any of the plates."""
    return any(inside(p, r) for p in segment_points(*segment) for r in rects)


def covers(rect, segments):
    """Whether a plate would sit on any leader or link already drawn."""
    return any(inside(p, rect) for s in segments for p in segment_points(*s))


def intersects(s1, s2):
    """Whether two segments cross (proper intersection, shared endpoints excluded)."""
    (p1, p2), (p3, p4) = s1, s2

    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return (v > 1e-9) - (v < -1e-9)
    if p1 in (p3, p4) or p2 in (p3, p4):
        return False
    return (orient(p1, p2, p3) * orient(p1, p2, p4) < 0
            and orient(p3, p4, p1) * orient(p3, p4, p2) < 0)


UNITS_PER_MM = 100.0 / 25.4  # the panel is written in draw.io's 100-units-per-inch canvas
FONT_UNITS = 6.0 / 0.72  # 6 pt, the figure-text size everywhere in this document
FONT_SMALL_UNITS = 7.0  # 5.04 pt, the floor, for the gene names along an attachment link
CHAR_MM = 0.50 * FONT_UNITS / UNITS_PER_MM  # Arial's mean advance is about half the size
CHAR_SMALL_MM = 0.50 * FONT_SMALL_UNITS / UNITS_PER_MM
LEAD, PAD = 4.5, 0.8
SUB_H = FONT_SMALL_UNITS / UNITS_PER_MM + 0.3  # the second line under an attached species
# Sixteen directions a label may take from its node, as (dx, dy, bias): the bias is a
# small preference for the right-hand side, so that with equal ink the labels read the
# same way. Exact 1 and -1 on the axes let the ring column pick "right" and "left".
DIRECTIONS = [(round(math.cos(k * math.pi / 8), 6), round(math.sin(k * math.pi / 8), 6),
               0.25 * (1 - math.cos(k * math.pi / 8))) for k in range(16)]
LEADS = (1.0, 1.8, 2.8, 4.0)  # multiples of LEAD a leader may run
PANEL_KEY = [
    ("#9673A6", "glycolysis and gluconeogenesis"),
    ("#D6B656", "pyruvate metabolism"),
    ("#B85450", "citrate cycle"),
    ("#D79B00", "fatty acid biosynthesis, elongation and desaturation"),
    ("#666666", "fatty acid degradation"),
]


def esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def place_label(pts, placed, lines, nodes, x, y, label, bounds, char_mm=CHAR_MM,
                font_units=FONT_UNITS, lead=LEAD, directions=DIRECTIONS, leader=True,
                why=None, sub_h=0.0):
    """The least-inked placement of `label` beside the point (x, y).

    Sixteen directions at four distances. The plate sits on the far side of the leader's end
    from the node, so the leader never crosses its own label. A candidate is rejected if
    its plate leaves the map, overlaps a placed plate, covers a labeled node, or sits on
    a leader or link already drawn, or if its leader passes through a placed plate or
    crosses a leader or link already drawn. A rejected candidate is tallied in `why` by
    reason. Returns the score, the
    leader end, the plate rectangle and the text origin, or None when nothing fits; the
    caller records the plate and leader in `placed` and `lines` once it commits to one.
    """
    tw, th = len(label) * char_mm, font_units / UNITS_PER_MM
    x0, y0, x1, y1 = bounds
    best = None
    for dx, dy, bias in directions:
        for k in LEADS:
            dist = lead * k
            lx, ly = x + dx * dist, y + dy * dist
            if dx > 0.01:
                px = lx + 0.4
            elif dx < -0.01:
                px = lx - 0.4 - tw
            else:
                px = lx - tw / 2
            py = ly - th / 2 + (0.0 if abs(dx) > 0.01 else dy * th * 0.7)
            rect = (px - PAD, py - PAD * 0.6, tw + 2 * PAD, th + 1.2 * PAD + sub_h)
            reasons = {
                "off map": rect[0] < x0 or rect[1] < y0 or rect[0] + rect[2] > x1 or rect[1] + rect[3] > y1,
                "on a plate": overlaps(rect, placed),
                "on a node": any(inside(p, rect, gap=0.8) for p in nodes if p != (x, y)),
                "on a line": covers(rect, lines),
                "leader through a plate": leader and crosses(((x, y), (lx, ly)), placed),
                "leader across a line": leader and any(intersects(((x, y), (lx, ly)), s)
                                                       for s in lines),
            }
            if any(reasons.values()):
                if why is not None:
                    for k, v in reasons.items():
                        why[k] = why.get(k, 0) + int(v)
                continue
            score = pts.rect(*rect) + pts.line((x, y), (lx, ly)) + bias * 3 + dist * 0.2
            if best is None or score < best[0]:
                best = (score, (lx, ly), rect, px, py + th / 2)
    return best


def nearest(points, to):
    return min(points, key=lambda p: math.hypot(p[0] - to[0], p[1] - to[1]))


def attachment_window(pts, attachments, copies, labeled, w, h, layout, bounds, step=1.0,
                      pull=1.0):
    """Where the attached species go: the window whose rings and links cover the least ink.

    Each candidate window is scored by the ink under it plus the ink along the link
    from every ring to the nearest copy of its nearest candidate anchor, so a window
    that is empty but reached only across the map's densest region loses to a slightly
    busier one beside its anchors. `pull` is a small cost per millimetre of link length
    that breaks ties toward the shorter link. Both sides of the column are tried, labels
    right of the rings and labels left of them, and a candidate whose links would pass
    through its own ring labels is rejected. Returns the window origin, the side, and
    per attachment the anchor chosen.
    """
    x0, y0, x1, y1 = bounds
    best = None
    for side in ("right", "left"):
        x = x0
        while x + w <= x1:
            y = y0
            while y + h <= y1:
                # A labeled compound needs room for its own label, so no window comes
                # within 3 mm of one.
                if any(inside(p, (x, y, w, h), gap=3.0) for p in labeled):
                    y += step
                    continue
                rings, plates = layout(x, y, side)
                score, chosen, ok = pts.rect(x, y, w, h, pad=0.0), [], True
                for a in attachments:
                    dst = rings[a["species"]]
                    options = []
                    for c in a["candidates"]:
                        src = nearest(copies.get(c["anchor"]) or [rings[c["anchor"]]], dst)
                        if crosses((src, dst), plates):
                            continue
                        options.append((pts.line(src, dst) + pull * math.hypot(src[0] - dst[0], src[1] - dst[1]), c, src))
                    if not options:
                        ok = False
                        break
                    cost, c, src = min(options, key=lambda t: t[0])
                    score += cost
                    chosen.append((c, src))
                if ok and (best is None or score < best[0]):
                    best = (score, x, y, side, chosen)
                y += step
            x += step
    if best is None:
        raise ValueError("no window places the attached species without a link through a label")
    return best[1], best[2], best[3], best[4]


KEY_ROW = 3.6  # mm between key rows
KEY_NOTE_ROW = 3.0
KEY_ROWS = len(PANEL_KEY) + 4  # the modules, two species kinds, intermediates, the link
KEY_NOTE_LINES = 4


def key_height_mm():
    return 8.0 + KEY_ROW * KEY_ROWS + KEY_NOTE_ROW * KEY_NOTE_LINES


def panel(map_svg, to_mm, positions, attachments, gene_rows, width_mm, map_w_mm,
          map_h_mm, panel_h_mm, key_x_mm):
    """The finished panel: the whole map on the left, its labels, and the key.

    Assembled as SVG rather than in draw.io. The map is several hundred kB, and a base64
    copy of it in one draw.io style attribute makes the headless exporter fail outright,
    printing only "Export failed"; writing the panel directly keeps it vector and removes
    the exporter from this figure's path. The output carries width before height and no
    units, which is the form svg_true_size_pdf.py reads as the 100-units-per-inch canvas.
    """
    u = UNITS_PER_MM
    view = re.search(r'viewBox="([^"]*)"', map_svg).group(1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width_mm * u:.2f}" height="{panel_h_mm * u:.2f}" '
        f'viewBox="0 0 {width_mm * u:.2f} {panel_h_mm * u:.2f}">',
        f'<style>text {{ font-family: Arial; font-size: {FONT_UNITS:.2f}px; }}</style>',
        None,  # the map, inserted once the copies to demote are known
    ]
    pts = Ink(map_svg, to_mm)
    bounds = (0.3, 0.3, map_w_mm - 0.3, map_h_mm - 0.3)

    # Every drawn copy of each labeled compound, and the one that carries the label: the
    # copy furthest from the panel edge, unless a link picks another copy below.
    copies = {name: [to_mm(x, y) for x, y in positions[cid]["nodes"]]
              for cid, _, name in WAYPOINTS + SPECIES if positions[cid]["nodes"]}
    node_at = {name: max(c, key=lambda p: min(p[0], map_w_mm - p[0], p[1], map_h_mm - p[1]))
               for name, c in copies.items()}

    # The attached species: one column of open rings in the window whose rings and links
    # cover the least ink, each joined to its anchor by a dashed link that names the
    # path's genes.
    labels = {name: label for _, label, name in WAYPOINTS + SPECIES}
    # Each ring's plate carries the name and, under it, the genes of its path, 5.4 mm
    # tall in all; rings 6.8 mm apart keep the plates clear of each other.
    DY, NODE_R = 6.8, R_SPECIES / 3774 * map_w_mm
    th = FONT_UNITS / u
    plate_h = th + 1.2 * PAD + SUB_H
    col_w = 1.5 + LEAD + max(len(labels[a["species"]]) for a in attachments) * CHAR_MM + 2.5
    col_h = DY * (len(attachments) - 1) + plate_h + 1.0
    inset = (bounds[0] + 2.0, bounds[1] + 2.0, bounds[2] - 2.0, bounds[3] - 2.0)

    def layout(x, y, side):
        """Ring centers and label plates of the column at (x, y), labels on `side`."""
        rings, plates = {}, []
        for i, a in enumerate(attachments):
            cy = y + 1.5 + i * DY
            tw = len(labels[a["species"]]) * CHAR_MM
            if side == "right":
                cx = x + 1.5 + NODE_R
                px = cx + LEAD + 0.4
            else:
                cx = x + col_w - 1.5 - NODE_R
                px = cx - LEAD - 0.4 - tw
            rings[a["species"]] = (cx, cy)
            plates.append((px - PAD, cy - th / 2 - PAD * 0.6, tw + 2 * PAD, plate_h))
        return rings, plates

    wx, wy, side, chosen = attachment_window(pts, attachments, copies, list(node_at.values()),
                                             col_w, col_h, layout, inset)
    rings, _ = layout(wx, wy, side)
    print(f"attachment window at ({wx:.1f}, {wy:.1f}) mm, {col_w:.1f} x {col_h:.1f} mm, "
          f"labels {side} of the rings, ink {pts.rect(wx, wy, col_w, col_h, pad=0.0):.0f}")
    placed, lines = [], []
    for a in attachments:
        node_at[a["species"]] = rings[a["species"]]
        copies[a["species"]] = [node_at[a["species"]]]
    links, resolved = [], []
    for a, (c, src) in zip(attachments, chosen):
        dst = node_at[a["species"]]
        node_at[c["anchor"]] = src  # the label goes on the copy the link starts from
        parts.append(f'<path d="M{src[0] * u:.2f},{src[1] * u:.2f} L{dst[0] * u:.2f},'
                     f'{dst[1] * u:.2f}" fill="none" stroke="#000000" stroke-width="0.6" '
                     f'stroke-dasharray="2.2,1.6"/>')
        links.append((src, dst, gene_run(c["genes"])))
        lines.append((src, dst))
        resolved.append({"species": a["species"], "anchor": c["anchor"], "path": c["path"],
                         "genes": c["genes"], "candidates": [x["anchor"] for x in a["candidates"]]})
        pts.add_line(src, dst)
    for a in attachments:
        pts.add(*node_at[a["species"]], 2.0)

    # A compound the map draws in more than one place keeps its large node only at the
    # labeled copy; the other copies are returned to the map's own faint node style, so
    # no large unlabeled dot is left for the reader to wonder about.
    demote = [p for name, c in copies.items() for p in c if p != node_at[name]]
    demoted = demote_copies(map_svg, demote, to_mm)
    inner = re.sub(r"^<svg[^>]*>", "", demoted, count=1).rsplit("</svg>", 1)[0]
    parts[2] = (f'<svg x="0" y="0" width="{map_w_mm * u:.2f}" height="{map_h_mm * u:.2f}" '
                f'viewBox="{view}" preserveAspectRatio="none">{inner}</svg>')

    nodes = list(node_at.values())

    def label_group(names, directions=DIRECTIONS, sub=None):
        """Place the labels of `names` together, in the best order.

        Greedy placement in one fixed order can strand the last label of a crowded
        cluster, so every order is tried and the order that fits all of them with the
        least total ink is kept. Plates and leaders are committed only for that order.
        `sub` maps a name to a second, smaller line set under its label on the same
        plate: the genes of the path that attaches a species.
        """
        sub = sub or {}
        best = None
        for order in itertools.permutations(names):
            plates, segs, trial, total = list(placed), list(lines), [], 0.0
            for name in order:
                x, y = node_at[name]
                fit = place_label(pts, plates, segs, nodes, x, y, labels[name], bounds,
                                  directions=directions, sub_h=SUB_H if name in sub else 0.0)
                if fit is None:
                    break
                plates.append(fit[2])
                segs.append(((x, y), fit[1]))
                total += fit[0]
                trial.append((name, fit))
            else:
                if best is None or total < best[0]:
                    best = (total, trial, plates, segs)
        if best is None:
            why = {}
            for name in names:
                place_label(pts, placed, lines, nodes, *node_at[name], labels[name], bounds,
                            directions=directions, why=why)
            raise ValueError(f"no order places the labels {names}: candidates rejected {why}")
        _, trial, placed[:], lines[:] = best
        out = []
        for name, (_, (lx, ly), rect, tx, ty) in trial:
            x, y = node_at[name]
            parts.append(f'<path d="M{x * u:.2f},{y * u:.2f} L{lx * u:.2f},{ly * u:.2f}" '
                         f'fill="none" stroke="#000000" stroke-width="0.5"/>')
            parts.append(f'<rect x="{rect[0] * u:.2f}" y="{rect[1] * u:.2f}" '
                         f'width="{rect[2] * u:.2f}" height="{rect[3] * u:.2f}" '
                         f'rx="{0.5 * u:.2f}" fill="#FFFFFF" fill-opacity="0.9" stroke="none"/>')
            parts.append(f'<text x="{tx * u:.2f}" y="{ty * u:.2f}" '
                         f'dominant-baseline="middle">{esc(labels[name])}</text>')
            if name in sub:
                parts.append(f'<text x="{tx * u:.2f}" y="{(ty + SUB_H) * u:.2f}" '
                             f'font-size="{FONT_SMALL_UNITS:.2f}px" '
                             f'dominant-baseline="middle">{esc(sub[name])}</text>')
            out.append({"name": name, "label": labels[name], "node_mm": [round(x, 2), round(y, 2)],
                        "attached": name in attached, "label_mm": [round(tx, 2), round(ty, 2)],
                        "genes": sub.get(name)})
        return out

    # The attached species are labeled first and always to the right of their ring, so
    # the column reads as one list. The genes of the path that attaches each species go
    # on a second, smaller line under its name: a label floating beside a link is
    # ambiguous where two links run nearly parallel, a line under the name is not.
    attached = {a["species"] for a in attachments}
    ring_side = [d for d in DIRECTIONS if d[:2] == ((1.0, 0.0) if side == "right" else (-1.0, 0.0))]
    if len(ring_side) != 1:
        raise ValueError(f"expected one {side} direction, found {ring_side}")
    anchors_out = label_group([a["species"] for a in attachments], directions=ring_side,
                              sub={a["species"]: gene_run(a["genes"]) for a in resolved})

    for name in attached:
        x, y = node_at[name]
        parts.append(f'<circle cx="{x * u:.2f}" cy="{y * u:.2f}" r="{NODE_R * u:.2f}" '
                     f'fill="#FFFFFF" stroke="{C_SPECIES}" stroke-width="1.6"/>')

    # Every other compound label: the drawn species first, then the intermediates.
    species_names = {name for _, _, name in SPECIES}
    rest = [n for n in node_at if n not in attached]
    anchors_out += label_group([n for n in rest if n in species_names])
    anchors_out += label_group([n for n in rest if n not in species_names])

    # The key.
    parts.append(f'<text x="{key_x_mm * u:.2f}" y="{3.0 * u:.2f}" font-weight="bold" '
                 f'dominant-baseline="middle">'
                 f'the route from central carbon to the measured lipids</text>')
    rows = [(c, t, "line") for c, t in PANEL_KEY]
    rows += [(C_SPECIES, "measured species, a node of the reference map", "dot"),
             (C_SPECIES, "measured species attached from the yeast GEM", "ring"),
             (C_WAYPOINT, "named intermediate", "dot"),
             ("#000000", "the GEM's shortest path from a drawn compound; its genes under the name", "dash")]
    if len(rows) != KEY_ROWS:
        raise ValueError(f"key has {len(rows)} rows, KEY_ROWS says {KEY_ROWS}")
    ROW = KEY_ROW
    for i, (color, text, kind) in enumerate(rows):
        yc = (8.0 + ROW * i) * u
        if kind in ("line", "dash"):
            dash = ' stroke-dasharray="2.2,1.6"' if kind == "dash" else ""
            width = 0.6 if kind == "dash" else 1.6
            parts.append(f'<line x1="{(key_x_mm + 0.5) * u:.2f}" y1="{yc:.2f}" '
                         f'x2="{(key_x_mm + 5.5) * u:.2f}" y2="{yc:.2f}" '
                         f'stroke="{color}" stroke-width="{width}"{dash}/>')
        elif kind == "dot":
            parts.append(f'<circle cx="{(key_x_mm + 3.0) * u:.2f}" cy="{yc:.2f}" '
                         f'r="{1.1 * u:.2f}" fill="{color}"/>')
        else:
            parts.append(f'<circle cx="{(key_x_mm + 3.0) * u:.2f}" cy="{yc:.2f}" '
                         f'r="{1.0 * u:.2f}" fill="#FFFFFF" stroke="{color}" stroke-width="1.6"/>')
        parts.append(f'<text x="{(key_x_mm + 7.5) * u:.2f}" y="{yc:.2f}" '
                     f'dominant-baseline="middle">{esc(text)}</text>')
    n_reg = sum(1 for r in gene_rows if r["role"] == "regulator")
    reg_on = sum(1 for r in gene_rows if r["role"] == "regulator" and r["map_elements"])
    n_pw = sum(1 for r in gene_rows if r["role"] == "pathway")
    pw_on = sum(1 for r in gene_rows if r["role"] == "pathway" and r["map_elements"])
    pw_off = gene_run([r["gene"] for r in gene_rows if r["role"] == "pathway" and not r["map_elements"]])
    note = [
        "Palmitoleate, stearate and oleate have no node on the reference map, as acids",
        "or acyl-CoAs; each is attached along the yeast GEM's shortest path from a drawn",
        f"compound. Of the {n_reg} deleted regulators, {reg_on} draw a reaction on this map; of the",
        f"{n_pw} pathway genes of b, {pw_on} do and {pw_off} do not.",
    ]
    if len(note) != KEY_NOTE_LINES:
        raise ValueError(f"note has {len(note)} lines, KEY_NOTE_LINES says {KEY_NOTE_LINES}")
    for i, line in enumerate(note):
        parts.append(f'<text x="{key_x_mm * u:.2f}" '
                     f'y="{(8.0 + ROW * len(rows) + KEY_NOTE_ROW * i) * u:.2f}" '
                     f'dominant-baseline="middle">{esc(line)}</text>')
    parts.append("</svg>")
    return "\n".join(parts), anchors_out, resolved


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch every response and fail if any has changed upstream")
    ap.add_argument("--width-mm", type=float, default=PANEL_WIDTHS_MM["full"],
                    help="width of the finished panel, map plus key")
    # The map's own aspect ratio makes an 89 mm map 53 mm tall, which with the network
    # panel under it keeps the figure inside the 170 mm page cap.
    ap.add_argument("--map-width-mm", type=float, default=89.0)
    ap.add_argument("--out", default="ffa_ipath_map")
    args = ap.parse_args()

    sel, record = selection(args.refresh)
    svg = opaque_route(fetch_map(sel, args.refresh).decode("latin-1"))
    compounds = [cid for cid, _, _ in WAYPOINTS + SPECIES] + [cid for cid, _ in SPECIES_COA]
    positions = probes(compounds, args.refresh)
    gene_rows = genes_on_map(args.refresh)
    orf_to_name = {r["orf"]: r["gene"] for r in gene_rows if r["role"] == "pathway"}

    drawn = [name for cid, _, name in WAYPOINTS + SPECIES if positions[cid]["nodes"]]
    undrawn = [name for cid, _, name in SPECIES if not positions[cid]["nodes"]]
    attachments = gem_paths(drawn, undrawn, orf_to_name)

    sized, map_h_mm, to_mm = whole(svg, args.map_width_mm)
    key_x = args.map_width_mm + 4.0
    # The panel is as tall as the taller of the map and the key beside it.
    height_mm = max(map_h_mm, key_height_mm())
    out, anchors, attachments = panel(sized, to_mm, positions, attachments, gene_rows,
                                      args.width_mm, args.map_width_mm, map_h_mm, height_mm,
                                      key_x)

    os.makedirs(IMAGES_DIR, exist_ok=True)
    out_svg = osp.join(IMAGES_DIR, args.out + ".svg")
    open(out_svg, "w", encoding="utf-8").write(out)
    json.dump(anchors, open(osp.join(IPATH_DIR, "label_anchors.json"), "w"), indent=2)

    for entry in record["species"]:
        entry["drawn_on_map"] = len(positions[entry["compound"]]["nodes"]) > 0
    for entry in record["waypoints"]:
        entry["drawn_on_map"] = len(positions[entry["compound"]]["nodes"]) > 0
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
        "species_acyl_coa_on_map": {label: len(positions[cid]["nodes"]) > 0
                                    for cid, label in SPECIES_COA},
        "attachments": attachments,
        "genes": gene_rows,
        "panel_mm": [round(args.width_mm, 1), round(height_mm, 1)],
        "map_mm": [round(args.map_width_mm, 1), round(map_h_mm, 1)],
    })
    record["stored"] = {f: sha256(open(osp.join(IPATH_DIR, f), "rb").read())
                        for f in sorted(os.listdir(IPATH_DIR))}
    open(osp.join(IPATH_DIR, "selection.txt"), "w", encoding="utf-8").write(sel)
    json.dump(record, open(osp.join(IPATH_DIR, "provenance.json"), "w"), indent=2)

    print(f"{record['selection_entries']} selection entries; "
          f"panel {args.width_mm:.1f} x {height_mm:.1f} mm, map {args.map_width_mm:.1f} x "
          f"{map_h_mm:.1f} mm")
    print(f"drawn by the map: {drawn}")
    for a in attachments:
        print(f"attached: {a['species']} <- {a['anchor']} via {' > '.join(a['path'][1:-1]) or '-'}"
              f" [{gene_run(a['genes'])}]")
    for r in gene_rows:
        print(f"  {r['role']:9s} {r['gene']:5s} {r['orf']:8s} {r['ko']}  map elements: {r['map_elements']}")
    print(f"wrote {out_svg}")


if __name__ == "__main__":
    main()
