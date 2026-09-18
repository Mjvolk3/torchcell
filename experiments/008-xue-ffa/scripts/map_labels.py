# experiments/008-xue-ffa/scripts/map_labels.py
# [[experiments.008-xue-ffa.scripts.map_labels]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/map_labels
#
# Label placement over a dense drawing, shared by the reference-map panels
# (ffa_ipath_map.py, ffa_kegg_map.py). Everything here is in panel millimetres.
#
# THE RULES, EACH ADDED AFTER A RENDER SHOWED THE FAILURE IT PREVENTS. A label tries
# sixteen directions at four leader lengths from its node and takes the candidate whose
# plate and leader cover the least ink. A candidate is rejected outright if its plate
# leaves the map, overlaps a placed plate, covers a labeled node, or sits on a leader or
# link already drawn, or if its leader passes through a placed plate or crosses another
# leader or link. A crowded group of labels is placed in the best of all orders rather than
# greedily, because a fixed order strands the last label. The ink grid weights the route
# far above the faint background: a label over gray lines costs the reader nothing, a
# label over the route hides the panel's content.

import itertools
import math

UNITS_PER_MM = 100.0 / 25.4  # the panels are written in draw.io's 100-units-per-inch canvas
FONT_UNITS = 6.0 / 0.72  # 6 pt, the figure-text size everywhere in this document
FONT_SMALL_UNITS = 7.0  # 5.04 pt, the floor, for the gene names under an attached species
CHAR_MM = 0.50 * FONT_UNITS / UNITS_PER_MM  # Arial's mean advance is about half the size
CHAR_SMALL_MM = 0.50 * FONT_SMALL_UNITS / UNITS_PER_MM
LEAD, PAD = 4.5, 0.8
SUB_H = FONT_SMALL_UNITS / UNITS_PER_MM + 0.3  # the second line under an attached species
# Sixteen directions a label may take from its node, as (dx, dy, bias): the bias is a
# small preference for the right-hand side, so that with equal ink the labels read the
# same way. Exact 1 and -1 on the axes let the ring column pick "right" and "left".
DIRECTIONS = [(round(math.cos(k * math.pi / 8), 6), round(math.sin(k * math.pi / 8), 6),
               0.25 * (1 - math.cos(k * math.pi / 8))) for k in range(16)]
RIGHT = [d for d in DIRECTIONS if d[:2] == (1.0, 0.0)]
LEFT = [d for d in DIRECTIONS if d[:2] == (-1.0, 0.0)]
LEADS = (1.0, 1.8, 2.8, 4.0)  # multiples of LEAD a leader may run


def esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def segment_points(a, b, step_mm=0.4):
    (ax, ay), (bx, by) = a, b
    k = max(1, int(math.hypot(bx - ax, by - ay) / step_mm))
    return [(ax + (bx - ax) * i / k, ay + (by - ay) * i / k) for i in range(k + 1)]


class Ink:
    """How much of the drawing lies under a rectangle or along a line, in panel mm.

    Every drawn element is sampled into points, the route weighted 1 and the faint
    background BACKGROUND_WEIGHT, and the points are binned on a CELL mm grid so a query
    sums a few cells rather than scanning thirty thousand points. A polyline contributes
    points along each of its segments every STEP mm, so a long straight line counts
    along its whole length and not only at its ends.
    """
    CELL, STEP = 0.5, 0.4
    BACKGROUND_WEIGHT = 0.08

    def __init__(self):
        self.cells = {}

    def add(self, x, y, w):
        key = (int(x // self.CELL), int(y // self.CELL))
        self.cells[key] = self.cells.get(key, 0.0) + w

    def add_line(self, a, b, w=1.0):
        for x, y in segment_points(a, b, self.STEP):
            self.add(x, y, w)

    def add_polyline(self, verts, w=1.0):
        for a, b in zip(verts, verts[1:]):
            self.add_line(a, b, w)

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


def nearest(points, to):
    return min(points, key=lambda p: math.hypot(p[0] - to[0], p[1] - to[1]))


def place_label(pts, placed, lines, nodes, x, y, label, bounds, char_mm=CHAR_MM,
                font_units=FONT_UNITS, lead=LEAD, directions=DIRECTIONS, leader=True,
                why=None, sub_h=0.0):
    """The least-inked placement of `label` beside the point (x, y).

    Sixteen directions at four distances. The plate sits on the far side of the leader's
    end from the node, so the leader never crosses its own label. A rejected candidate is
    tallied in `why` by reason. Returns (score, leader end, plate rect, text x, text y),
    or None when nothing fits; the caller records the plate and leader in `placed` and
    `lines` once it commits to one.
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
                    for k2, v in reasons.items():
                        why[k2] = why.get(k2, 0) + int(v)
                continue
            score = pts.rect(*rect) + pts.line((x, y), (lx, ly)) + bias * 3 + dist * 0.2
            if best is None or score < best[0]:
                best = (score, (lx, ly), rect, px, py + th / 2)
    return best


def column_layout(attachments, labels, col_w, node_r, dy=6.8):
    """The attached-species column: rings and their label plates as a function of origin.

    Each ring's plate carries the name and, under it, the genes of its path; rings `dy`
    apart keep the plates clear of each other. Returns (layout, col_h) where
    layout(x, y, side) gives ring centers by species and the plate rectangles.
    """
    th = FONT_UNITS / UNITS_PER_MM
    plate_h = th + 1.2 * PAD + SUB_H
    col_h = dy * (len(attachments) - 1) + plate_h + 1.0

    def layout(x, y, side):
        rings, plates = {}, []
        for i, a in enumerate(attachments):
            cy = y + 1.5 + i * dy
            tw = len(labels[a["species"]]) * CHAR_MM
            if side == "right":
                cx = x + 1.5 + node_r
                px = cx + LEAD + 0.4
            else:
                cx = x + col_w - 1.5 - node_r
                px = cx - LEAD - 0.4 - tw
            rings[a["species"]] = (cx, cy)
            plates.append((px - PAD, cy - th / 2 - PAD * 0.6, tw + 2 * PAD, plate_h))
        return rings, plates
    return layout, col_h


def column_width(attachments, labels):
    return 1.5 + LEAD + max(len(labels[a["species"]]) for a in attachments) * CHAR_MM + 2.5


def attachment_window(pts, attachments, copies, labeled, w, h, layout, bounds, step=1.0,
                      pull=1.0):
    """Where the attached species go: the window whose rings and links cover the least ink.

    Each candidate window is scored by the ink under it plus the ink along the link
    from every ring to the nearest copy of its nearest candidate anchor, so a window
    that is empty but reached only across the map's densest region loses to a slightly
    busier one beside its anchors. `pull` is a small cost per millimetre of link length
    that breaks ties toward the shorter link. Both sides of the column are tried, labels
    right of the rings and labels left of them, and a candidate whose links would pass
    through its own ring labels is rejected, as is any window within 3 mm of a labeled
    node. Returns the window origin, the side, and per attachment the anchor chosen.
    """
    x0, y0, x1, y1 = bounds
    best = None
    for side in ("right", "left"):
        x = x0
        while x + w <= x1:
            y = y0
            while y + h <= y1:
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


class Placer:
    """Places groups of labels over one drawing, keeping what has been placed.

    `pts` is the Ink grid, `node_at` the labeled position of every named node, `labels`
    the text of each, `bounds` the map rectangle. Plates and leaders committed so far are
    in `placed` and `lines`; every placed label is appended to `out` as SVG fragments.
    """

    def __init__(self, pts, node_at, labels, bounds):
        self.pts, self.node_at, self.labels, self.bounds = pts, node_at, labels, bounds
        self.placed, self.lines, self.nodes = [], [], list(node_at.values())
        self.svg, self.records = [], []

    def add_line(self, src, dst):
        self.lines.append((src, dst))
        self.pts.add_line(src, dst)

    def group(self, names, directions=DIRECTIONS, sub=None, attached=frozenset()):
        """Place the labels of `names` together, in the best order.

        Greedy placement in one fixed order can strand the last label of a crowded
        cluster, so every order is tried and the order that fits all of them with the
        least total ink is kept. `sub` maps a name to a second, smaller line set under
        its label on the same plate: the genes of the path that attaches a species.
        """
        u = UNITS_PER_MM
        sub = sub or {}
        best = None
        for order in itertools.permutations(names):
            plates, segs, trial, total = list(self.placed), list(self.lines), [], 0.0
            for name in order:
                x, y = self.node_at[name]
                fit = place_label(self.pts, plates, segs, self.nodes, x, y, self.labels[name],
                                  self.bounds, directions=directions,
                                  sub_h=SUB_H if name in sub else 0.0)
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
                place_label(self.pts, self.placed, self.lines, self.nodes, *self.node_at[name],
                            self.labels[name], self.bounds, directions=directions, why=why)
            raise ValueError(f"no order places the labels {names}: candidates rejected {why}")
        _, trial, self.placed[:], self.lines[:] = best
        for name, (_, (lx, ly), rect, tx, ty) in trial:
            x, y = self.node_at[name]
            self.svg.append(f'<path d="M{x * u:.2f},{y * u:.2f} L{lx * u:.2f},{ly * u:.2f}" '
                            f'fill="none" stroke="#000000" stroke-width="0.5"/>')
            self.svg.append(f'<rect x="{rect[0] * u:.2f}" y="{rect[1] * u:.2f}" '
                            f'width="{rect[2] * u:.2f}" height="{rect[3] * u:.2f}" '
                            f'rx="{0.5 * u:.2f}" fill="#FFFFFF" fill-opacity="0.9" stroke="none"/>')
            self.svg.append(f'<text x="{tx * u:.2f}" y="{ty * u:.2f}" '
                            f'dominant-baseline="middle">{esc(self.labels[name])}</text>')
            if name in sub:
                self.svg.append(f'<text x="{tx * u:.2f}" y="{(ty + SUB_H) * u:.2f}" '
                                f'font-size="{FONT_SMALL_UNITS:.2f}px" '
                                f'dominant-baseline="middle">{esc(sub[name])}</text>')
            self.records.append({"name": name, "label": self.labels[name],
                                 "node_mm": [round(x, 2), round(y, 2)],
                                 "attached": name in attached,
                                 "label_mm": [round(tx, 2), round(ty, 2)],
                                 "genes": sub.get(name)})
        return self.records
