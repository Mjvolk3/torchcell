"""Lay out and render :mod:`torchcell.paper.ontology_graph` as a zoomable SVG map.

The layout is deliberately domain-aware rather than a generic force/graphviz pass.
The schema has a real shape -- an hourglass whose waist is
``Experiment = (Genotype x Environment) -> Phenotype`` -- and a generic layout hides
that shape under a hairball. So each domain gets a titled lane, each inheritance
tree inside a lane is drawn as a horizontal tree (parent left, children stacked
right, parent centred on its children), and composition edges are drawn as light
curves anchored at the exact field row that holds the reference.

Everything is emitted as one self-contained SVG with no external references, so the
same file drops into draw.io, into LaTeX (via rsvg/Inkscape), and into the HTML
explorer.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from pydantic import BaseModel, Field

from torchcell.paper.ontology_graph import (
    LANE_HEADINGS,
    LANE_ORDER,
    LANE_PALETTE_INDEX,
    LANE_SUBTITLES,
    LANE_TITLES,
    OntologyGraph,
)
from torchcell.utils import PLOT_PALETTE, PLOT_PALETTE_FILL

# --- Card + lane geometry, in SVG user units. One unit is arbitrary; the physical
# size comes from the width/height attributes, and the viewBox makes the whole thing
# scale losslessly. Row height and font size are chosen so that a card reads
# comfortably at roughly 4x zoom in a PDF viewer.
CARD_W = 258.0
HEADER_H = 21.0
ROW_H = 12.6
CARD_PAD_Y = 5.0
GAP_X = 46.0  # horizontal gap between tree levels
GAP_Y = 9.0  # vertical gap between sibling cards
TREE_GAP_Y = 34.0  # vertical gap between separate trees in one lane
LANE_PAD = 26.0
LANE_TITLE_H = 62.0
LANE_GAP_X = 54.0
LANE_GAP_Y = 46.0

FONT_CLASS = 8.6
FONT_FIELD = 7.0
FONT_LANE = 34.0
FONT_LANE_SUB = 13.0

# Backbone composition edges: the ones that *are* the data model. Drawn bold; every
# other composition edge is drawn as a hairline so the backbone stays legible.
BACKBONE_EDGES = frozenset(
    {
        ("Experiment", "Genotype"),
        ("Experiment", "Environment"),
        ("Experiment", "Phenotype"),
        ("ExperimentReference", "ReferenceGenome"),
        ("ExperimentReference", "Environment"),
        ("ExperimentReference", "Phenotype"),
        ("Genotype", "GenePerturbation"),
        ("Environment", "Media"),
    }
)

# Which column and stacking slot each lane occupies. Columns read left to right as
# the hourglass: inputs, waist, output, supporting vocabulary.
LANE_GRID: dict[str, tuple[int, int]] = {
    "genotype": (0, 0),
    "experiment": (1, 0),
    "environment": (1, 1),
    "phenotype": (2, 0),
    "provenance": (3, 0),
    "enum": (3, 1),
}

# Target height for a lane before its trees wrap into another column. Lanes made of
# many shallow roots (environment, the enums) would otherwise render as a single
# absurdly tall thin strip; wrapping them keeps every lane roughly rectangular so the
# lanes tile without leaving large voids.
LANE_TARGET_H: dict[str, float] = {
    "genotype": 3000.0,
    "environment": 900.0,
    "experiment": 2000.0,
    "phenotype": 1900.0,
    "provenance": 420.0,
    "enum": 900.0,
}

# Compact cards are ~6x shorter, so no lane needs to wrap; a wrap there just leaves a
# one-item second column and a wide empty frame.
LANE_TARGET_H_COMPACT: dict[str, float] = dict.fromkeys(LANE_TARGET_H, 1.0e6)

# Compact mode rebalances the grid: with short lanes, environment fits beside
# phenotype instead of hanging below experiment and stretching the whole canvas.
LANE_GRID_COMPACT: dict[str, tuple[int, int]] = {
    "genotype": (0, 0),
    "experiment": (1, 0),
    "phenotype": (2, 0),
    "environment": (2, 1),
    "provenance": (3, 0),
    "enum": (3, 1),
}


def _text_w(text: str, font_size: float, factor: float = 0.53) -> float:
    return len(text) * font_size * factor


def _truncate(text: str, max_w: float, font_size: float) -> str:
    if _text_w(text, font_size) <= max_w:
        return text
    keep = max(1, int(max_w / (font_size * 0.53)) - 1)
    return text[:keep] + "…"


class Card(BaseModel):
    """A placed class box."""

    name: str
    lane: str
    x: float = 0.0
    y: float = 0.0
    w: float = CARD_W
    h: float = 0.0
    row_anchors: dict[str, float] = Field(
        default_factory=dict, description="field name -> y offset of its row centre"
    )

    @property
    def cx(self) -> float:
        """Card centre x."""
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        """Card centre y."""
        return self.y + self.h / 2


class Lane(BaseModel):
    """A titled domain block containing one or more laid-out trees."""

    key: str
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    card_names: list[str] = Field(default_factory=list)


class Layout(BaseModel):
    """A fully placed diagram, ready to serialise."""

    cards: dict[str, Card] = Field(default_factory=dict)
    lanes: dict[str, Lane] = Field(default_factory=dict)
    width: float = 0.0
    height: float = 0.0


def _card_height(graph: OntologyGraph, name: str, compact: bool = False) -> float:
    """Card height. In compact mode the body is dropped and only the header shows.

    Compact mode is what makes a Nature-sized panel possible: with 95 classes the
    full field list is ~3700 units tall, which at 179 mm wide would be 212 mm tall --
    over the 170 mm ceiling. Collapsing to headers keeps the same lanes, the same
    topology and the same colors, just without the field rows.
    """
    if compact:
        return HEADER_H + 3.0
    return HEADER_H + graph.classes[name].row_count * ROW_H + CARD_PAD_Y * 2


def _lane_roots(graph: OntologyGraph, lane: str) -> list[str]:
    """Trees to draw in a lane: classes with no parent, or whose parent is elsewhere."""
    members = set(graph.lane_members(lane))
    roots = [
        n
        for n in sorted(members)
        if graph.classes[n].parent is None or graph.classes[n].parent not in members
    ]
    # Put the conceptually primary root first so the lane reads top-down sensibly.
    priority = {"Genotype": 0, "Environment": 0, "Experiment": 0, "Phenotype": 0}
    return sorted(roots, key=lambda n: (priority.get(n, 1), n))


def _place_tree(
    graph: OntologyGraph,
    name: str,
    x: float,
    y: float,
    out: dict[str, Card],
    lane: str,
    compact: bool = False,
) -> tuple[float, float]:
    """Place ``name`` and its subtree; return (block height, rightmost x reached).

    Parent sits at ``x`` vertically centred on the stacked block of its children,
    which start at ``x + CARD_W + GAP_X``. This is the classic horizontal tree: it
    keeps sibling classes aligned in a readable column and makes depth in the
    ontology map directly onto horizontal distance.
    """
    own_h = _card_height(graph, name, compact)
    children = graph.children_of(name)
    if not children:
        out[name] = Card(name=name, lane=lane, x=x, y=y, h=own_h)
        return own_h, x + CARD_W

    child_x = x + CARD_W + GAP_X
    cursor = y
    far_right = child_x + CARD_W
    gap_y = GAP_Y if not compact else GAP_Y * 0.7
    for child in children:
        block_h, reach = _place_tree(graph, child, child_x, cursor, out, lane, compact)
        far_right = max(far_right, reach)
        cursor += block_h + gap_y
    block_h = cursor - y - gap_y

    out[name] = Card(
        name=name, lane=lane, x=x, y=y + max(0.0, (block_h - own_h) / 2), h=own_h
    )
    return max(block_h, own_h), far_right


def _flow_lane(
    graph: OntologyGraph, lane: str, compact: bool
) -> tuple[dict[str, Card], float, float]:
    """Lay a lane's trees out, wrapping into a new column past the target height."""
    target = (LANE_TARGET_H_COMPACT if compact else LANE_TARGET_H)[lane]
    cards: dict[str, Card] = {}
    col_x = 0.0
    cursor_y = 0.0
    col_right = 0.0
    max_x = 0.0
    max_y = 0.0
    tree_gap = TREE_GAP_Y if not compact else TREE_GAP_Y * 0.6

    for root in _lane_roots(graph, lane):
        probe: dict[str, Card] = {}
        block_h, _ = _place_tree(graph, root, 0.0, 0.0, probe, lane, compact)
        if cursor_y > 0 and cursor_y + block_h > target:
            col_x = col_right + LANE_GAP_X
            cursor_y = 0.0
        _, reach = _place_tree(graph, root, col_x, cursor_y, cards, lane, compact)
        col_right = max(col_right, reach)
        max_x = max(max_x, reach)
        cursor_y += block_h + tree_gap
        max_y = max(max_y, cursor_y - tree_gap)

    return cards, max_x, max_y


def build_layout(graph: OntologyGraph, compact: bool = False) -> Layout:
    """Place every class into its lane, then pack lanes into the grid."""
    layout = Layout()

    # 1. Lay each lane out at the origin, measuring its natural extent.
    lane_local: dict[str, dict[str, Card]] = {}
    for lane in LANE_ORDER:
        cards, max_x, max_y = _flow_lane(graph, lane, compact)
        lane_local[lane] = cards
        title_h = LANE_TITLE_H if not compact else LANE_TITLE_H * 0.72
        layout.lanes[lane] = Lane(
            key=lane,
            w=max_x + LANE_PAD * 2,
            h=max_y + LANE_PAD * 2 + title_h,
            card_names=sorted(cards),
        )

    # 2. Pack lanes into columns. A lane keeps its OWN width -- forcing it to the
    #    column width drew a huge empty box around the narrow lanes -- but the next
    #    column starts past the widest lane in this one, so columns stay aligned.
    grid = LANE_GRID_COMPACT if compact else LANE_GRID
    col_w: dict[int, float] = {}
    for lane, (col, _) in grid.items():
        col_w[col] = max(col_w.get(col, 0.0), layout.lanes[lane].w)

    col_x: dict[int, float] = {}
    running = 0.0
    for col in sorted(col_w):
        col_x[col] = running
        running += col_w[col] + LANE_GAP_X
    total_w = running - LANE_GAP_X

    col_cursor: dict[int, float] = dict.fromkeys(col_w, 0.0)
    for lane in sorted(grid, key=lambda k: (grid[k][0], grid[k][1])):
        col, _ = grid[lane]
        entry = layout.lanes[lane]
        entry.x = col_x[col]
        entry.y = col_cursor[col]
        col_cursor[col] += entry.h + LANE_GAP_Y

    total_h = max(col_cursor.values()) - LANE_GAP_Y

    # 3. Translate each lane's cards into absolute space and record row anchors.
    title_h = LANE_TITLE_H if not compact else LANE_TITLE_H * 0.72
    for lane, cards in lane_local.items():
        entry = layout.lanes[lane]
        dx = entry.x + LANE_PAD
        dy = entry.y + title_h + LANE_PAD
        for card in cards.values():
            card.x += dx
            card.y += dy
            cls = graph.classes[card.name]
            rows = (
                cls.enum_members
                if cls.kind == "enum"
                else [f.name for f in cls.own_fields]
            )
            for i, row_name in enumerate(rows):
                card.row_anchors[row_name] = (
                    card.y + HEADER_H + CARD_PAD_Y + i * ROW_H + ROW_H / 2
                )
            layout.cards[card.name] = card

    layout.width = total_w
    layout.height = total_h
    return layout


def _lane_colors(lane: str) -> tuple[str, str]:
    idx = LANE_PALETTE_INDEX[lane]
    return PLOT_PALETTE[idx], PLOT_PALETTE_FILL[idx]


def _card_svg(graph: OntologyGraph, card: Card, compact: bool = False) -> str:
    cls = graph.classes[card.name]
    if compact:
        stroke, fill = _lane_colors(card.lane)
        dash = ' stroke-dasharray="3 2"' if cls.is_abstract else ""
        title = _truncate(cls.name, card.w - 12, FONT_CLASS)
        return (
            f'<g class="node" id="node-{card.name}" data-name="{card.name}" '
            f'data-lane="{card.lane}">'
            f'<rect class="card" x="{card.x:.1f}" y="{card.y:.1f}" '
            f'width="{card.w:.1f}" height="{card.h:.1f}" rx="3" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1"{dash}/>'
            f'<text x="{card.x + 6:.1f}" y="{card.y + card.h / 2 + 3.1:.1f}" '
            f'font-size="{FONT_CLASS}" font-weight="700" fill="#2B2B2B">'
            f"{escape(title)}</text></g>"
        )
    stroke, fill = _lane_colors(card.lane)
    parts: list[str] = []
    parts.append(
        f'<g class="node" id="node-{card.name}" data-name="{card.name}" '
        f'data-lane="{card.lane}">'
    )
    dash = ' stroke-dasharray="4 2.5"' if cls.is_abstract else ""
    parts.append(
        f'<rect class="card" x="{card.x:.1f}" y="{card.y:.1f}" '
        f'width="{card.w:.1f}" height="{card.h:.1f}" rx="4" '
        f'fill="#FFFFFF" stroke="{stroke}" stroke-width="1.1"{dash}/>'
    )
    parts.append(
        f'<path class="card-head" d="M{card.x:.1f} {card.y + 4:.1f} '
        f"a4 4 0 0 1 4 -4 h{card.w - 8:.1f} a4 4 0 0 1 4 4 "
        f'v{HEADER_H - 4:.1f} h{-card.w:.1f} Z" fill="{fill}" stroke="none"/>'
    )
    parts.append(
        f'<line x1="{card.x:.1f}" y1="{card.y + HEADER_H:.1f}" '
        f'x2="{card.x + card.w:.1f}" y2="{card.y + HEADER_H:.1f}" '
        f'stroke="{stroke}" stroke-width="1.1"/>'
    )

    title = _truncate(cls.name, card.w - 16, FONT_CLASS)
    parts.append(
        f'<text class="cls-name" x="{card.x + 8:.1f}" '
        f'y="{card.y + HEADER_H - 6.5:.1f}" font-size="{FONT_CLASS}" '
        f'font-weight="700" fill="#2B2B2B">{escape(title)}</text>'
    )
    if cls.kind == "enum":
        parts.append(
            f'<text class="cls-tag" x="{card.x + card.w - 8:.1f}" '
            f'y="{card.y + HEADER_H - 6.5:.1f}" font-size="{FONT_FIELD - 0.4}" '
            f'text-anchor="end" fill="#666666">enum</text>'
        )
    elif cls.inherited_field_count:
        parts.append(
            f'<text class="cls-tag" x="{card.x + card.w - 8:.1f}" '
            f'y="{card.y + HEADER_H - 6.5:.1f}" font-size="{FONT_FIELD - 0.4}" '
            f'text-anchor="end" fill="#8A8A8A">'
            f"+{cls.inherited_field_count} inherited</text>"
        )

    parts.append('<g class="lod-detail">')
    if cls.kind == "enum":
        for i, member in enumerate(cls.enum_members):
            ty = card.y + HEADER_H + CARD_PAD_Y + i * ROW_H + ROW_H - 3.4
            parts.append(
                f'<text x="{card.x + 8:.1f}" y="{ty:.1f}" font-size="{FONT_FIELD}" '
                f'fill="#4A4A4A">{escape(_truncate(member, card.w - 16, FONT_FIELD))}'
                f"</text>"
            )
    else:
        for i, field in enumerate(cls.own_fields):
            ty = card.y + HEADER_H + CARD_PAD_Y + i * ROW_H + ROW_H - 3.4
            label = field.name + ("" if field.required else "?")
            name_w = min(_text_w(label, FONT_FIELD) + 6, card.w * 0.52)
            type_w = card.w - 16 - name_w
            parts.append(
                f'<text x="{card.x + 8:.1f}" y="{ty:.1f}" font-size="{FONT_FIELD}" '
                f'fill="#2B2B2B">'
                f"{escape(_truncate(label, name_w, FONT_FIELD))}</text>"
            )
            parts.append(
                f'<text x="{card.x + card.w - 8:.1f}" y="{ty:.1f}" '
                f'font-size="{FONT_FIELD}" text-anchor="end" fill="#8A8A8A">'
                f"{escape(_truncate(field.type_label, type_w, FONT_FIELD))}</text>"
            )
    parts.append("</g>")
    parts.append("</g>")
    return "".join(parts)


def _inheritance_svg(graph: OntologyGraph, layout: Layout) -> str:
    """UML generalisation: elbow from the child's left edge back to the parent."""
    parts: list[str] = []
    for name, cls in graph.classes.items():
        if not cls.parent or cls.parent not in layout.cards or name not in layout.cards:
            continue
        child = layout.cards[name]
        parent = layout.cards[cls.parent]
        stroke, _ = _lane_colors(cls.lane)
        x1, y1 = parent.x + parent.w, parent.cy
        x2, y2 = child.x, child.cy
        mid = x1 + GAP_X / 2
        parts.append(
            f'<path class="inherit" d="M{x2:.1f} {y2:.1f} H{mid:.1f} '
            f'V{y1:.1f} H{x1 + 7:.1f}" fill="none" stroke="{stroke}" '
            f'stroke-width="0.9" stroke-opacity="0.85"/>'
        )
        parts.append(
            f'<path class="inherit-head" d="M{x1:.1f} {y1:.1f} '
            f'l7 -3.4 v6.8 Z" fill="#FFFFFF" stroke="{stroke}" stroke-width="0.9"/>'
        )
    return "".join(parts)


def _composition_svg(graph: OntologyGraph, layout: Layout) -> str:
    """Curved has-a edges, anchored on the field row that holds the reference."""
    parts: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for edge in graph.composition_edges:
        key = (edge.source, edge.target, edge.field_name)
        if key in seen:
            continue
        seen.add(key)
        if edge.source not in layout.cards or edge.target not in layout.cards:
            continue
        src, dst = layout.cards[edge.source], layout.cards[edge.target]
        backbone = (edge.source, edge.target) in BACKBONE_EDGES
        stroke, _ = _lane_colors(graph.classes[edge.target].lane)

        y1 = src.row_anchors.get(edge.field_name, src.cy)
        going_right = dst.x >= src.x + src.w
        x1 = src.x + src.w if going_right else src.x
        x2 = dst.x if going_right else dst.x + dst.w
        y2 = dst.cy
        bow = max(70.0, abs(x2 - x1) * 0.42)
        c1 = x1 + bow if going_right else x1 - bow
        c2 = x2 - bow if going_right else x2 + bow

        width = "1.9" if backbone else "0.6"
        opacity = "0.9" if backbone else "0.3"
        cls_attr = "compose backbone" if backbone else "compose"
        parts.append(
            f'<path class="{cls_attr}" data-src="{edge.source}" '
            f'data-dst="{edge.target}" '
            f'd="M{x1:.1f} {y1:.1f} C{c1:.1f} {y1:.1f} {c2:.1f} {y2:.1f} '
            f'{x2:.1f} {y2:.1f}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width}" stroke-opacity="{opacity}" '
            f'stroke-dasharray="{"none" if backbone else "3 2.5"}"/>'
        )
        if backbone:
            direction = -1 if going_right else 1
            parts.append(
                f'<path class="compose-head" d="M{x2:.1f} {y2:.1f} '
                f'l{direction * 7.5:.1f} -3.6 v7.2 Z" fill="{stroke}"/>'
            )
    return "".join(parts)


def _lane_frames_svg(
    graph: OntologyGraph, layout: Layout, compact: bool = False
) -> str:
    parts: list[str] = []
    for key in LANE_ORDER:
        lane = layout.lanes[key]
        stroke, fill = _lane_colors(key)
        head, _, sub = LANE_TITLES[key].partition(" — ")
        head = head.upper()

        # Shrink the banner to whatever the lane is actually wide, so a narrow lane
        # cannot spill its title past its own frame.
        count_label = f"{len(lane.card_names)} classes"
        avail = lane.w - LANE_PAD * 2 - _text_w(count_label, FONT_LANE_SUB) - 16
        size = FONT_LANE if not compact else FONT_LANE * 0.72
        while size > 11 and _text_w(head, size, 0.62) > avail:
            size -= 1.0
        sub_size = FONT_LANE_SUB if not compact else FONT_LANE_SUB * 0.8
        base_y = lane.y + (38 if not compact else 28)

        parts.append(
            f'<g class="lane" id="lane-{key}">'
            f'<rect x="{lane.x:.1f}" y="{lane.y:.1f}" width="{lane.w:.1f}" '
            f'height="{lane.h:.1f}" rx="10" fill="{fill}" fill-opacity="0.26" '
            f'stroke="{stroke}" stroke-width="1.4"/>'
            f'<text x="{lane.x + LANE_PAD:.1f}" y="{base_y:.1f}" '
            f'font-size="{size:.1f}" font-weight="700" fill="{stroke}">'
            f"{escape(head)}</text>"
            f'<text x="{lane.x + LANE_PAD:.1f}" y="{base_y + 16:.1f}" '
            f'font-size="{sub_size:.1f}" fill="#666666">'
            f"{escape(_truncate(sub, avail, sub_size))}</text>"
            f'<text x="{lane.x + lane.w - LANE_PAD:.1f}" y="{base_y:.1f}" '
            f'font-size="{sub_size:.1f}" text-anchor="end" fill="{stroke}" '
            f'fill-opacity="0.75">{count_label}</text>'
            f"</g>"
        )
    return "".join(parts)


def _legend_svg(
    layout: Layout,
    x: float,
    y: float,
    compact: bool = False,
    explore_url: str | None = None,
) -> str:
    rows = [
        ("solid box", "concrete class you can instantiate"),
        ("dashed box", "abstract base class"),
        ("solid line + hollow arrow", "inheritance (is-a), arrow points to the parent"),
        ("bold curve", "the data-model backbone (genotype × environment → phenotype)"),
    ]
    if not compact:
        rows += [
            (
                "faint dashed curve",
                "composition (has-a), anchored on the field that holds it",
            ),
            ("field?", "optional field; unmarked fields are required"),
            ("+n inherited", "fields inherited from ancestors, shown on the card"),
        ]
    else:
        rows += [
            ("faint dashed curve", "composition (has-a)"),
            ("box colour", "domain lane; every class keeps its lane's colour"),
        ]

    scale = 1.0 if not compact else 1.35  # legend text must survive the 179 mm render
    fs = 7.4 * scale
    lh = 15.0 * scale
    box_w = CARD_W * (1.55 if not compact else 2.3)
    box_h = 28 * scale + len(rows) * lh + (lh if explore_url else 0)

    parts = [
        f'<g class="legend"><rect x="{x:.1f}" y="{y:.1f}" width="{box_w:.1f}" '
        f'height="{box_h:.1f}" rx="6" fill="#FFFFFF" '
        f'stroke="#666666" stroke-width="1"/>'
        f'<text x="{x + 10:.1f}" y="{y + 18 * scale:.1f}" '
        f'font-size="{10 * scale:.1f}" font-weight="700" '
        f'fill="#2B2B2B">How to read this map</text>'
    ]
    for i, (mark, meaning) in enumerate(rows):
        ty = y + 34 * scale + i * lh
        parts.append(
            f'<text x="{x + 10:.1f}" y="{ty:.1f}" font-size="{fs:.1f}" '
            f'font-weight="700" fill="#4A4A4A">{escape(mark)}</text>'
            f'<text x="{x + 10 + box_w * 0.42:.1f}" y="{ty:.1f}" '
            f'font-size="{fs:.1f}" fill="#666666">{escape(meaning)}</text>'
        )
    if explore_url:
        ty = y + 34 * scale + len(rows) * lh
        parts.append(
            f'<text x="{x + 10:.1f}" y="{ty:.1f}" font-size="{fs:.1f}" '
            f'font-weight="700" fill="#4A4A4A">explore it</text>'
            f'<text x="{x + 10 + box_w * 0.42:.1f}" y="{ty:.1f}" '
            f'font-size="{fs:.1f}" fill="#6C8EBF">{escape(explore_url)}</text>'
        )
    parts.append("</g>")
    return "".join(parts)


# Child rows are indented by shifting the text's x, not by prefixing spaces: leading
# whitespace in an SVG text node is collapsed by renderers (rsvg drops it outright,
# and a non-breaking space is honoured inconsistently), which silently flattens the
# parent/child structure. An x offset is unambiguous everywhere.
INDENT_W = 7.0


def _wrap_csv(names: list[str], max_w: float, font_size: float) -> list[str]:
    """Comma-join ``names`` and wrap so no line exceeds ``max_w``."""
    lines: list[str] = []
    current = ""
    for i, name in enumerate(names):
        piece = name + ("," if i < len(names) - 1 else "")
        trial = f"{current} {piece}" if current else piece
        if current and _text_w(trial, font_size) > max_w:
            lines.append(current)
            current = piece
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def _suffix_family(parent: str, children: list[str]) -> list[str] | None:
    """Children with the parent's name stripped, when every child carries it as a suffix.

    ``Phenotype`` -> ``FitnessPhenotype, CalMorphPhenotype, ...`` is the dominant shape
    in this schema, and stripping the shared suffix is what lets all thirteen fit on a
    printed panel without dropping any of them.
    """
    if not children or not all(c.endswith(parent) and c != parent for c in children):
        return None
    return [c[: -len(parent)] for c in children]


def _lane_body_lines(
    graph: OntologyGraph,
    lane: str,
    max_w: float,
    font_size: float = 6.0,
    max_lines: int = 11,
) -> list[tuple[float, str]]:
    """Derive a lane block's body as ``(x offset, text)`` rows.

    Nothing here is typed by hand: the roots, the subtype counts, and the names all
    come off :class:`OntologyGraph`, so the panel cannot name a class the schema does
    not have, and cannot omit one it does without saying how many it dropped.
    """

    def in_lane(names: list[str]) -> list[str]:
        return [n for n in names if graph.classes[n].lane == lane]

    roots = graph.lane_roots(lane)
    branching = [r for r in roots if in_lane(graph.children_of(r))]
    flat = [r for r in roots if not in_lane(graph.children_of(r))]

    inner_w = max_w - INDENT_W
    lines: list[tuple[float, str]] = []
    seen_families: dict[tuple[str, ...], str] = {}
    for root in branching:
        children = in_lane(graph.children_of(root))
        n_desc = len(in_lane(graph.descendants_of(root)))
        lines.append((0.0, f"{root}  ·  {n_desc} subtypes"))
        family = _suffix_family(root, children)
        if family is None:
            for child in children:
                extra = len(in_lane(graph.descendants_of(child)))
                tail = f"  ·  {extra}" if extra else ""
                lines.append((INDENT_W, f"{child}{tail}"))
            continue
        key = tuple(family)
        if key in seen_families:
            same = f"same {len(family)} families as {seen_families[key]}"
            lines.append((INDENT_W, same))
            continue
        seen_families[key] = root
        lines.extend((INDENT_W, text) for text in _wrap_csv(family, inner_w, font_size))
    if flat:
        lines.extend((0.0, text) for text in _wrap_csv(flat, max_w, font_size))

    if len(lines) > max_lines:
        dropped = len(lines) - max_lines + 1
        lines = lines[: max_lines - 1]
        lines.append(
            (0.0, f"…  +{dropped} more lines — full list in the interactive map")
        )
    return lines


def render_schematic_svg(
    graph: OntologyGraph,
    physical_width_mm: float = 179.0,
    explore_url: str | None = None,
) -> str:
    """Render the legible, Nature-compliant structural panel.

    The full map cannot be a printed figure: every class name across 179 mm forces the
    labels to about 1.2 pt, five times under Nature's 6 pt floor. So this panel drops
    to the level that *is* legible -- the six domains, the hourglass between them, and
    each domain's roots and subtype counts read off the graph -- and prints the
    explorer URL for the rest.

    The whole thing is laid out in POINTS, with the viewBox width set to the physical
    width in points. A font-size of 6 in user units is therefore exactly 6 pt in the
    rendered figure, which makes the type-size guarantee structural rather than a
    thing to re-check by hand.
    """
    vb_w = physical_width_mm * 72.0 / 25.4
    margin = 10.0
    gap = 9.0
    title_band = 26.0

    # Three columns: inputs | waist | output, with the two supporting lanes beneath.
    col_w = (vb_w - margin * 2 - gap * 2) / 3
    body_w = col_w - 10.0

    # (lane, heading, gloss, body lines). Heading and gloss are the only editorial
    # strings; every body line is derived from the graph, so a renamed or newly added
    # class cannot leave a stale label behind.
    blocks: list[tuple[str, str, str, list[tuple[float, str]]]] = [
        (
            lane,
            LANE_HEADINGS[lane],
            LANE_SUBTITLES[lane],
            _lane_body_lines(graph, lane, body_w),
        )
        for lane in LANE_ORDER
    ]

    line_h = 7.6
    head_h = 20.0

    def block_h(lines: list[tuple[float, str]]) -> float:
        return head_h + len(lines) * line_h + 7.0

    placed: dict[str, tuple[float, float, float, float]] = {}
    by_key = {b[0]: b for b in blocks}

    x0, x1, x2 = margin, margin + col_w + gap, margin + (col_w + gap) * 2
    y = margin + title_band

    h_gen = block_h(by_key["genotype"][3])
    h_env = block_h(by_key["environment"][3])
    h_exp = block_h(by_key["experiment"][3])
    h_phe = block_h(by_key["phenotype"][3])
    placed["genotype"] = (x0, y, col_w, h_gen)
    placed["environment"] = (x0, y + h_gen + gap, col_w, h_env)
    left_h = h_gen + gap + h_env
    placed["experiment"] = (x1, y + (left_h - h_exp) / 2, col_w, h_exp)
    placed["phenotype"] = (x2, y + (left_h - h_phe) / 2, col_w, h_phe)

    y2 = y + left_h + gap * 1.6
    h_prov = block_h(by_key["provenance"][3])
    h_enum = block_h(by_key["enum"][3])
    placed["provenance"] = (x0, y2, col_w, h_prov)
    placed["enum"] = (x1, y2, col_w * 2 + gap, h_enum)

    parts: list[str] = []
    arrows: list[str] = []

    def arrow(xa: float, ya: float, xb: float, yb: float, color: str) -> str:
        """Route a backbone arrow through the gutter between two columns.

        The columns are adjacent -- the gutter is ``gap`` units wide while the blocks
        it connects sit tens of units apart vertically. A bezier whose control points
        sit at the horizontal midpoint therefore degenerates into a near-vertical
        squiggle crammed into that gutter, so the connector is drawn as an orthogonal
        elbow instead: out horizontally, down (or up) the middle of the gutter, then
        horizontally into the target. Arrival is horizontal by construction, which is
        what makes the fixed right-pointing head correct.
        """
        tip = xb - 4.4
        if abs(yb - ya) < 0.5:
            spine = f"M{xa:.1f} {ya:.1f} H{tip:.1f}"
        else:
            mid = (xa + xb) / 2
            r = min(3.0, abs(yb - ya) / 2, (xb - xa) / 2)
            sweep_in, sweep_out = (1, 0) if yb > ya else (0, 1)
            step = r if yb > ya else -r
            spine = (
                f"M{xa:.1f} {ya:.1f} H{mid - r:.1f} "
                f"A{r:.1f} {r:.1f} 0 0 {sweep_in} {mid:.1f} {ya + step:.1f} "
                f"V{yb - step:.1f} "
                f"A{r:.1f} {r:.1f} 0 0 {sweep_out} {mid + r:.1f} {yb:.1f} "
                f"H{tip:.1f}"
            )
        return (
            f'<path d="{spine}" fill="none" stroke="{color}" stroke-width="1.4" '
            f'stroke-linecap="butt" stroke-linejoin="round"/>'
            f'<path d="M{xb:.1f} {yb:.1f} l-4.4 -2.2 v4.4 Z" fill="{color}"/>'
        )

    gx, gy, gw, gh = placed["genotype"]
    ex, ey, ew, eh = placed["environment"]
    xx, xy, xw, xh = placed["experiment"]
    px, py, pw, ph = placed["phenotype"]
    arrows.append(arrow(gx + gw, gy + gh / 2, xx, xy + xh * 0.36, PLOT_PALETTE[0]))
    arrows.append(arrow(ex + ew, ey + eh / 2, xx, xy + xh * 0.64, PLOT_PALETTE[1]))
    arrows.append(arrow(xx + xw, xy + xh / 2, px, py + ph / 2, PLOT_PALETTE[2]))

    for key, head, sub, lines in blocks:
        bx, by, bw, bh = placed[key]
        stroke, fill = _lane_colors(key)
        parts.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
            f'rx="3" fill="#FFFFFF" stroke="{stroke}" stroke-width="1"/>'
            f'<path d="M{bx:.1f} {by + 3:.1f} a3 3 0 0 1 3 -3 h{bw - 6:.1f} '
            f'a3 3 0 0 1 3 3 v{head_h - 3:.1f} h{-bw:.1f} Z" fill="{fill}"/>'
            f'<text x="{bx + 5:.1f}" y="{by + 9.4:.1f}" font-size="7" '
            f'font-weight="700" fill="{stroke}">{escape(head)}</text>'
            f'<text x="{bx + bw - 5:.1f}" y="{by + 9.4:.1f}" font-size="6" '
            f'text-anchor="end" fill="{stroke}">'
            f"{len(graph.lane_members(key))} classes</text>"
            f'<text x="{bx + 5:.1f}" y="{by + 17.2:.1f}" font-size="6" '
            f'fill="#666666">{escape(_truncate(sub, bw - 10, 6))}</text>'
        )
        for i, (dx, line) in enumerate(lines):
            parts.append(
                f'<text x="{bx + 5 + dx:.1f}" y="{by + head_h + 6 + i * line_h:.1f}" '
                f'font-size="6" fill="#2B2B2B">'
                f"{escape(_truncate(line, bw - 10 - dx, 6))}</text>"
            )

    # Arrows last: they run in the gutters, so they must not be painted over by the
    # opaque block rectangles (which is what reduced them to steep slivers before).
    parts.extend(arrows)

    n_models = sum(1 for c in graph.classes.values() if c.kind == "model")
    n_fields = sum(len(c.own_fields) for c in graph.classes.values())
    foot_y = y2 + max(h_prov, h_enum) + 13.0
    parts.append(
        f'<text x="{margin:.1f}" y="{foot_y:.1f}" font-size="6" fill="#666666">'
        f"{n_models} pydantic classes &#183; {n_fields} declared fields &#183; "
        f"generated from torchcell.datamodels.schema</text>"
    )
    if explore_url:
        parts.append(
            f'<text x="{vb_w - margin:.1f}" y="{foot_y:.1f}" font-size="6" '
            f'text-anchor="end" fill="{PLOT_PALETTE[4]}">'
            f"full interactive map: {escape(explore_url)}</text>"
        )

    vb_h = foot_y + 8.0
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{physical_width_mm:.2f}mm" '
        f'height="{vb_h * 25.4 / 72.0:.2f}mm" '
        f'viewBox="0 0 {vb_w:.1f} {vb_h:.1f}" '
        f'font-family="Arial, Helvetica, sans-serif">'
        f"<style>text{{font-family:Arial,Helvetica,sans-serif}}</style>"
        f'<rect width="100%" height="100%" fill="#FFFFFF"/>'
        f'<text x="{margin:.1f}" y="{margin + 8:.1f}" font-size="9" '
        f'font-weight="700" fill="#2B2B2B">The torchcell experiment ontology</text>'
        f'<text x="{margin:.1f}" y="{margin + 17:.1f}" font-size="6.5" '
        f'fill="#666666">Every record is a typed genotype '
        f"× environment → phenotype experiment, with provenance "
        f"attached to each sourced value.</text>"
        f"{''.join(parts)}</svg>"
    )


def render_svg(
    graph: OntologyGraph,
    layout: Layout,
    physical_width_mm: float,
    title: str,
    subtitle: str,
    compact: bool = False,
    explore_url: str | None = None,
) -> str:
    """Serialise a placed layout to one self-contained SVG document."""
    margin = 40.0
    header_h = 96.0 if not compact else 72.0
    # Derived, not guessed: a hardcoded legend band clipped the box off the canvas.
    legend_scale = 1.0 if not compact else 1.35
    legend_rows = 6 if compact else 7
    legend_h = (
        26.0
        + 28 * legend_scale
        + (legend_rows + (1 if explore_url else 0)) * 15.0 * legend_scale
        + 16.0
    )
    vb_w = layout.width + margin * 2
    vb_h = layout.height + margin * 2 + header_h + legend_h
    scale = physical_width_mm / vb_w

    body = "".join(
        [
            _lane_frames_svg(graph, layout, compact),
            _composition_svg(graph, layout),
            _inheritance_svg(graph, layout),
            "".join(
                _card_svg(graph, layout.cards[n], compact) for n in sorted(layout.cards)
            ),
        ]
    )

    n_models = sum(1 for c in graph.classes.values() if c.kind == "model")
    n_enums = sum(1 for c in graph.classes.values() if c.kind == "enum")
    n_fields = sum(len(c.own_fields) for c in graph.classes.values())

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{physical_width_mm:.2f}mm" height="{vb_h * scale:.2f}mm" '
        f'viewBox="0 0 {vb_w:.1f} {vb_h:.1f}" '
        f'font-family="Arial, Helvetica, sans-serif">'
        f"<style>text{{font-family:Arial,Helvetica,sans-serif}}</style>"
        f'<rect width="100%" height="100%" fill="#FFFFFF"/>'
        f'<text x="{margin:.1f}" y="46" font-size="30" font-weight="700" '
        f'fill="#2B2B2B">{escape(title)}</text>'
        f'<text x="{margin:.1f}" y="70" font-size="13" fill="#666666">'
        f"{escape(subtitle)}</text>"
        f'<text x="{margin:.1f}" y="88" font-size="11" fill="#8A8A8A">'
        f"{n_models} classes &#183; {n_enums} controlled vocabularies &#183; "
        f"{n_fields} declared fields &#183; generated from "
        f"torchcell.datamodels.schema</text>"
        f'<g id="diagram" transform="translate({margin:.1f},{margin + header_h:.1f})">'
        f"{body}</g>"
        f"{_legend_svg(layout, margin, margin + header_h + layout.height + 26, compact, explore_url)}"
        f"</svg>"
    )
