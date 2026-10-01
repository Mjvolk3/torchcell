# tests/torchcell/paper/test_ontology_svg.py
# [[tests.torchcell.paper.test_ontology_svg]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/paper/test_ontology_svg.py
"""The ontology map and schematic SVGs, pinned on a hand-built eight-class schema.

2026.09.30, Phase 17. Fixture ``_graph()``: seven models and one enum.

* genotype lane: ``Genotype`` (1 field), abstract ``GenePerturbation`` (1 field) and its
  child ``DeletionPerturbation`` (1 optional field, 1 inherited).
* experiment lane: ``Experiment`` (2 fields). environment lane: empty.
* phenotype lane: ``Phenotype`` (1 field, parent ``ProvenanceGapMixin`` in the
  provenance lane) and its child ``FitnessPhenotype`` (2 fields, 1 inherited).
* provenance lane: ``ProvenanceGapMixin`` (no fields). enum lane: ``Strand`` with
  members forward, reverse.
* composition edges: Experiment.genotype -> Genotype, Experiment.phenotype ->
  Phenotype, Genotype.perturbations -> GenePerturbation (three backbone edges),
  FitnessPhenotype.strand -> Strand (no ``strand`` row), a duplicate of the first edge,
  and an edge to a class that is not in the graph.

Layout arithmetic (CARD_W 258, HEADER_H 21, ROW_H 12.6, CARD_PAD_Y 5, GAP_X 46, GAP_Y 9,
TREE_GAP_Y 34, LANE_PAD 26, LANE_TITLE_H 62, LANE_GAP_X 54, LANE_GAP_Y 46). A card is
21 + 12.6 * rows + 10 tall: 43.6 for one row, 56.2 for two, 31.0 for none; 24.0 in
compact mode. A lane is (reach + 52) wide and (tree extent + 52 + 62) tall:

* genotype: roots Genotype (priority 0), then GenePerturbation at y 43.6 + 34 = 77.6
  with its child at x 258 + 46 = 304; extent 562 x 121.2, lane 614 x 235.2.
* experiment 310 x 170.2; environment (empty) 52 x 114; phenotype: Phenotype (43.6) is
  centered on its 56.2 child, y = (56.2 - 43.6) / 2 = 6.3, lane 614 x 170.2;
  provenance 310 x 145; enum 310 x 170.2.
* grid columns 0..3 have widths 614, 310, 614, 310 and start at 0, 668, 1032, 1700;
  the width is 1700 + 310 = 2010. Column 1 stacks experiment (y 0) over environment
  (y 170.2 + 46 = 216.2); column 3 stacks provenance (y 0) over enum (y 191). The height
  is the tallest column, 191 + 170.2 = 361.2.
* cards shift by (lane.x + 26, lane.y + 62 + 26): FitnessPhenotype at (1362, 88),
  Phenotype at (1058, 94.3), Strand at (1726, 279). A row anchor is
  card.y + 21 + 5 + 12.6 i + 6.3, so Strand's rows sit at 311.3 and 323.9.

The rendered document is 2010 + 80 wide and 361.2 + 80 + 96 + legend band tall, where
the band is 26 + 28 + 15 * (7 rows + 1 url row) + 16 = 190: viewBox 2090 x 727.2 and
179 mm x 62.28 mm. Colors are the six locked primaries of ``PLOT_PALETTE`` and their
fills, written as literal hex so a palette edit shows up here. The schematic is laid
out in points (179 mm = 507.4 pt) and every body line comes off the graph.

The real schema is rendered read-only (nothing is written) and checked by structural
identities: one node per class, one inheritance elbow per in-graph parent link, one
composition curve per distinct in-graph edge, the header counts, and the schematic's
6 pt type floor.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest

from torchcell.paper import ontology_svg as svg
from torchcell.paper.ontology_graph import (
    LANE_ORDER,
    CompositionEdge,
    OntologyClass,
    OntologyField,
    OntologyGraph,
    build_ontology_graph,
)

NS = "{http://www.w3.org/2000/svg}"
ELLIPSIS = "…"
MIDDOT = "·"

LANE_STROKE = {
    "genotype": "#D79B00",
    "environment": "#B85450",
    "phenotype": "#9673A6",
    "experiment": "#D6B656",
    "provenance": "#6C8EBF",
    "enum": "#666666",
}
LANE_FILL = {
    "genotype": "#FFE6CC",
    "environment": "#F8CECC",
    "phenotype": "#E1D5E7",
    "experiment": "#FFF2CC",
    "provenance": "#DAE8FC",
    "enum": "#F5F5F5",
}


def _field(name: str, type_label: str = "str", required: bool = True) -> OntologyField:
    return OntologyField(
        name=name, type_label=type_label, required=required, inherited=False
    )


def _model(
    name: str,
    lane: str,
    fields: list[OntologyField] | None = None,
    parent: str | None = None,
    inherited: int = 0,
    abstract: bool = False,
) -> OntologyClass:
    return OntologyClass(
        name=name,
        kind="model",
        lane=lane,
        module="m",
        parent=parent,
        own_fields=fields or [],
        inherited_field_count=inherited,
        is_abstract=abstract,
    )


def _graph() -> OntologyGraph:
    classes = [
        _model("Genotype", "genotype", [_field("perturbations", "list[GP]")]),
        _model(
            "GenePerturbation",
            "genotype",
            [_field("systematic_gene_name")],
            abstract=True,
        ),
        _model(
            "DeletionPerturbation",
            "genotype",
            [_field("strain_id", "str | None", required=False)],
            parent="GenePerturbation",
            inherited=1,
        ),
        _model(
            "Experiment",
            "experiment",
            [_field("genotype", "Genotype"), _field("phenotype", "Phenotype")],
        ),
        _model(
            "Phenotype",
            "phenotype",
            [_field("graph_level")],
            parent="ProvenanceGapMixin",
        ),
        _model(
            "FitnessPhenotype",
            "phenotype",
            [_field("fitness", "float"), _field("fitness_std", "float | None", False)],
            parent="Phenotype",
            inherited=1,
        ),
        _model("ProvenanceGapMixin", "provenance"),
        OntologyClass(
            name="Strand",
            kind="enum",
            lane="enum",
            module="m",
            enum_members=["forward", "reverse"],
        ),
    ]
    edges = [
        ("Experiment", "Genotype", "genotype"),
        ("Experiment", "Phenotype", "phenotype"),
        ("Genotype", "GenePerturbation", "perturbations"),
        ("FitnessPhenotype", "Strand", "strand"),
        ("Experiment", "Genotype", "genotype"),
        ("Experiment", "Missing", "x"),
    ]
    return OntologyGraph(
        classes={c.name: c for c in classes},
        composition_edges=[
            CompositionEdge(source=s, target=t, field_name=f) for s, t, f in edges
        ],
    )


def _elements(document: str, tag: str, cls: str | None = None) -> list[ET.Element]:
    root = ET.fromstring(document)
    return [
        e
        for e in root.iter(f"{NS}{tag}")
        if cls is None or (e.get("class") or "").split() == cls.split()
    ]


# ---------------------------------------------------------------- text and card sizes


def test_text_width_and_truncation_closed_forms() -> None:
    """Width = len * size * 0.53. "Genotype" at 8.6 is 8 * 4.558 = 36.464, so it fits
    37 and is cut at 20 to int(20 / 4.558) - 1 = 3 characters plus an ellipsis; any
    width keeps at least one character.
    """
    assert svg._text_w("Genotype", 8.6) == pytest.approx(36.464)
    assert svg._truncate("Genotype", 37.0, 8.6) == "Genotype"
    assert svg._truncate("Genotype", 20.0, 8.6) == "Gen" + ELLIPSIS
    assert svg._truncate("Genotype", -50.0, 8.6) == "G" + ELLIPSIS


def test_card_height_counts_fields_or_members_and_compact_is_header_only() -> None:
    """21 + 12.6 rows + 10: one field 43.6, two fields or two enum members 56.2, no
    field 31.0; compact is 21 + 3 = 24 whatever the rows.
    """
    graph = _graph()
    heights = {
        name: svg._card_height(graph, name)
        for name in ["Genotype", "Experiment", "Strand", "ProvenanceGapMixin"]
    }
    assert heights == pytest.approx(
        {
            "Genotype": 43.6,
            "Experiment": 56.2,
            "Strand": 56.2,
            "ProvenanceGapMixin": 31.0,
        }
    )
    assert svg._card_height(graph, "Experiment", compact=True) == 24.0


def test_lane_roots_put_the_primary_root_first_and_count_outside_parents() -> None:
    """Genotype sorts before GenePerturbation by the priority table, although
    alphabetical order says otherwise; Phenotype is a root of its lane because its
    parent lives in the provenance lane.
    """
    graph = _graph()
    assert svg._lane_roots(graph, "genotype") == ["Genotype", "GenePerturbation"]
    assert svg._lane_roots(graph, "phenotype") == ["Phenotype"]
    assert svg._lane_roots(graph, "environment") == []


# ---------------------------------------------------------------- layout


def test_layout_packs_the_six_lanes_into_the_four_column_grid() -> None:
    """Lane rectangles (x, y, w, h) and the canvas size from the docstring arithmetic."""
    layout = svg.build_layout(_graph())
    lanes = {k: (v.x, v.y, v.w, v.h) for k, v in layout.lanes.items()}
    assert lanes == {
        "genotype": pytest.approx((0.0, 0.0, 614.0, 235.2)),
        "environment": pytest.approx((668.0, 216.2, 52.0, 114.0)),
        "experiment": pytest.approx((668.0, 0.0, 310.0, 170.2)),
        "phenotype": pytest.approx((1032.0, 0.0, 614.0, 170.2)),
        "provenance": pytest.approx((1700.0, 0.0, 310.0, 145.0)),
        "enum": pytest.approx((1700.0, 191.0, 310.0, 170.2)),
    }
    assert (layout.width, layout.height) == pytest.approx((2010.0, 361.2))
    assert layout.lanes["genotype"].card_names == [
        "DeletionPerturbation",
        "GenePerturbation",
        "Genotype",
    ]


def test_layout_centers_parents_on_children_and_anchors_each_row() -> None:
    """Card origins and row anchors in absolute space: a child one level right of its
    parent (x + 304), a parent centered on a taller child (Phenotype 6.3 below
    FitnessPhenotype), and anchors at card.y + 26 + 12.6 i + 6.3 (enum members too).
    """
    layout = svg.build_layout(_graph())
    cards = {k: (v.x, v.y, v.h) for k, v in layout.cards.items()}
    assert cards == {
        "Genotype": pytest.approx((26.0, 88.0, 43.6)),
        "GenePerturbation": pytest.approx((26.0, 165.6, 43.6)),
        "DeletionPerturbation": pytest.approx((330.0, 165.6, 43.6)),
        "Experiment": pytest.approx((694.0, 88.0, 56.2)),
        "Phenotype": pytest.approx((1058.0, 94.3, 43.6)),
        "FitnessPhenotype": pytest.approx((1362.0, 88.0, 56.2)),
        "ProvenanceGapMixin": pytest.approx((1726.0, 88.0, 31.0)),
        "Strand": pytest.approx((1726.0, 279.0, 56.2)),
    }
    assert layout.cards["Experiment"].row_anchors == pytest.approx(
        {"genotype": 120.3, "phenotype": 132.9}
    )
    assert layout.cards["Strand"].row_anchors == pytest.approx(
        {"forward": 311.3, "reverse": 323.9}
    )
    assert layout.cards["ProvenanceGapMixin"].row_anchors == {}


def test_compact_layout_shrinks_the_gaps_and_moves_environment_beside_phenotype() -> (
    None
):
    """Compact: cards 24 tall, tree gap 34 * 0.6 = 20.4, title band 62 * 0.72 = 44.64.
    The genotype extent is 24 + 20.4 + 24 = 68.4, so its lane is 68.4 + 52 + 44.64 =
    165.04 tall. Environment moves to column 2 below phenotype (y 120.64 + 46 =
    166.64) and the empty lane is 52 + 44.64 = 96.64 tall.
    """
    layout = svg.build_layout(_graph(), compact=True)
    assert layout.lanes["genotype"].h == pytest.approx(165.04)
    env = layout.lanes["environment"]
    assert (env.x, env.y, env.h) == pytest.approx((1032.0, 166.64, 96.64))
    assert layout.lanes["phenotype"].h == pytest.approx(120.64)
    assert layout.cards["GenePerturbation"].y == pytest.approx(44.64 + 26 + 44.4)


def test_flow_lane_wraps_a_tree_that_would_pass_the_target_height() -> None:
    """Three 10-field provenance roots (157 tall each) against the 420 target: A at y 0,
    B at 157 + 34 = 191 (191 + 157 = 348 fits), C would end at 382 + 157 = 539, so it
    starts a new column at 258 + 54 = 312, y 0. Extent: 312 + 258 = 570 wide, 348 tall.
    """
    ten = [_field(f"f{i}") for i in range(10)]
    graph = OntologyGraph(
        classes={n: _model(n, "provenance", ten) for n in ["A", "B", "C"]}
    )
    cards, max_x, max_y = svg._flow_lane(graph, "provenance", compact=False)
    assert {n: (c.x, c.y) for n, c in cards.items()} == pytest.approx(
        {"A": (0.0, 0.0), "B": (0.0, 191.0), "C": (312.0, 0.0)}
    )
    assert (max_x, max_y) == pytest.approx((570.0, 348.0))


# ---------------------------------------------------------------- SVG pieces


def test_a_fieldless_card_renders_to_the_exact_markup() -> None:
    """ProvenanceGapMixin at (1726, 88), 258 x 31, provenance colors: white body with the
    lane stroke, a rounded header path filled with the lane fill, the header rule at
    y 88 + 21, the name at (x + 8, y + 21 - 6.5), no tag (no inherited fields) and an
    empty detail group.
    """
    graph = _graph()
    layout = svg.build_layout(graph)
    assert svg._card_svg(graph, layout.cards["ProvenanceGapMixin"]) == (
        '<g class="node" id="node-ProvenanceGapMixin" data-name="ProvenanceGapMixin" '
        'data-lane="provenance">'
        '<rect class="card" x="1726.0" y="88.0" width="258.0" height="31.0" rx="4" '
        'fill="#FFFFFF" stroke="#6C8EBF" stroke-width="1.1"/>'
        '<path class="card-head" d="M1726.0 92.0 a4 4 0 0 1 4 -4 h250.0 a4 4 0 0 1 4 4 '
        'v17.0 h-258.0 Z" fill="#DAE8FC" stroke="none"/>'
        '<line x1="1726.0" y1="109.0" x2="1984.0" y2="109.0" stroke="#6C8EBF" '
        'stroke-width="1.1"/>'
        '<text class="cls-name" x="1734.0" y="102.5" font-size="8.6" font-weight="700" '
        'fill="#2B2B2B">ProvenanceGapMixin</text>'
        '<g class="lod-detail"></g></g>'
    )


def test_card_rows_tags_dashes_and_type_truncation() -> None:
    """Abstract GenePerturbation has the "4 2.5" dash, concrete DeletionPerturbation has
    none and carries "+1 inherited"; its optional field reads "strain_id?". Strand is
    tagged "enum" and lists its members at card.y + 26 + 12.6 i + 9.2 (314.2, 326.8).
    A 60-character type label in a 258 card: the name column is
    10 * 3.71 + 6 = 43.1, leaving 198.9 for the type, which keeps
    int(198.9 / 3.71) - 1 = 52 characters and an ellipsis.
    """
    graph = _graph()
    layout = svg.build_layout(graph)
    abstract = svg._card_svg(graph, layout.cards["GenePerturbation"])
    concrete = svg._card_svg(graph, layout.cards["DeletionPerturbation"])
    assert 'stroke-width="1.1" stroke-dasharray="4 2.5"/>' in abstract
    assert "stroke-dasharray" not in concrete
    assert re.findall(r">([^<>]+)</text>", concrete) == [
        "DeletionPerturbation",
        "+1 inherited",
        "strain_id?",
        "str | None",
    ]
    strand = svg._card_svg(graph, layout.cards["Strand"])
    assert re.findall(r'y="([\d.]+)"[^>]*>([^<>]+)</text>', strand) == [
        ("293.5", "Strand"),
        ("293.5", "enum"),
        ("314.2", "forward"),
        ("326.8", "reverse"),
    ]
    long_type = "x" * 60
    graph.classes["DeletionPerturbation"].own_fields[0].type_label = long_type
    cut = svg._card_svg(graph, layout.cards["DeletionPerturbation"])
    assert re.findall(r">([^<>]+)</text>", cut)[-1] == "x" * 52 + ELLIPSIS


def test_a_compact_card_is_one_filled_box_with_its_name_centered() -> None:
    """Compact Strand at (1726, 237.3): lane fill instead of white, rx 3, stroke width
    1, name baseline at y + 24 / 2 + 3.1 = 252.4, x + 6 = 1732.
    """
    graph = _graph()
    layout = svg.build_layout(graph, compact=True)
    assert svg._card_svg(graph, layout.cards["Strand"], compact=True) == (
        '<g class="node" id="node-Strand" data-name="Strand" data-lane="enum">'
        '<rect class="card" x="1726.0" y="237.3" width="258.0" height="24.0" rx="3" '
        'fill="#F5F5F5" stroke="#666666" stroke-width="1"/>'
        '<text x="1732.0" y="252.4" font-size="8.6" font-weight="700" '
        'fill="#2B2B2B">Strand</text></g>'
    )
    layout.cards["GenePerturbation"].h = 24.0
    abstract = svg._card_svg(graph, layout.cards["GenePerturbation"], compact=True)
    assert 'stroke-width="1" stroke-dasharray="3 2"/>' in abstract


def test_composition_curves_style_backbone_direction_dedup_and_the_row_anchor() -> None:
    """Six edges give four curves: the duplicate and the edge to an absent class are
    dropped. Experiment -> Genotype runs LEFT: from Experiment's left edge (694) at its
    ``genotype`` row (120.3) to Genotype's right edge (26 + 258 = 284) at its center
    (88 + 21.8 = 109.8), bow max(70, 410 * 0.42) = 172.2, bold backbone style in the
    TARGET lane's color with a head pointing left. FitnessPhenotype -> Strand runs
    right from 1362 + 258 = 1620 to 1726; ``strand`` is not a row, so it leaves from
    the card center 116.1; bow max(70, 106 * 0.42) = 70; hairline, dashed, no head.
    """
    graph = _graph()
    out = svg._composition_svg(graph, svg.build_layout(graph))
    assert out.count('<path class="compose') == 4 + 3
    assert out.count('class="compose-head"') == 3
    assert (
        '<path class="compose backbone" data-src="Experiment" data-dst="Genotype" '
        'd="M694.0 120.3 C521.8 120.3 456.2 109.8 284.0 109.8" fill="none" '
        'stroke="#D79B00" stroke-width="1.9" stroke-opacity="0.9" '
        'stroke-dasharray="none"/>'
        '<path class="compose-head" d="M284.0 109.8 l7.5 -3.6 v7.2 Z" fill="#D79B00"/>'
    ) in out
    assert out.endswith(
        '<path class="compose" data-src="FitnessPhenotype" data-dst="Strand" '
        'd="M1620.0 116.1 C1690.0 116.1 1656.0 307.1 1726.0 307.1" fill="none" '
        'stroke="#666666" stroke-width="0.6" stroke-opacity="0.3" '
        'stroke-dasharray="3 2.5"/>'
    )


def test_inheritance_elbow_within_a_lane_is_exact() -> None:
    """GenePerturbation (284 right edge, center 165.6 + 21.8 = 187.4) to its child at
    x 330: out of the child's left edge, across to the gutter middle 284 + 23 = 307, to
    the parent's center height, and into x1 + 7 = 291 where the hollow head starts.
    """
    graph = _graph()
    out = svg._inheritance_svg(graph, svg.build_layout(graph))
    assert (
        '<path class="inherit" d="M330.0 187.4 H307.0 V187.4 H291.0" fill="none" '
        'stroke="#D79B00" stroke-width="0.9" stroke-opacity="0.85"/>'
        '<path class="inherit-head" d="M284.0 187.4 l7 -3.4 v6.8 Z" fill="#FFFFFF" '
        'stroke="#D79B00" stroke-width="0.9"/>'
    ) in out
    assert out.count('class="inherit"') == 3


def test_a_parent_right_of_its_child_is_reached_from_the_child_right_edge() -> None:
    """A parent wholly right of its child is entered on its LEFT edge, head pointing right.

    Contract (issue #541): ProvenanceGapMixin (x 1726, cy 103.5) sits right of Phenotype
    (x 1058, w 258, cy 116.1), as in the real schema. The elbow leaves Phenotype's right
    edge 1316, turns at 1726 - LANE_PAD / 2 = 1713 (inside the provenance lane's left
    pad, which holds no card), and ends at the head base 1726 - 7 = 1719; the head tip
    is the parent's left edge with ``l-7``. No coordinate of the path lies inside either
    card's x span except the endpoints on their edges. The left-parent elbow of
    FitnessPhenotype is unchanged.
    """
    graph = _graph()
    out = svg._inheritance_svg(graph, svg.build_layout(graph))
    assert (
        '<path class="inherit" d="M1316.0 116.1 H1713.0 V103.5 H1719.0" fill="none" '
        'stroke="#9673A6" stroke-width="0.9" stroke-opacity="0.85"/>'
        '<path class="inherit-head" d="M1726.0 103.5 l-7 -3.4 v6.8 Z" fill="#FFFFFF" '
        'stroke="#9673A6" stroke-width="0.9"/>'
    ) in out
    assert (
        '<path class="inherit" d="M1362.0 116.1 H1339.0 V116.1 H1323.0" fill="none" '
        'stroke="#9673A6" stroke-width="0.9" stroke-opacity="0.85"/>'
        '<path class="inherit-head" d="M1316.0 116.1 l7 -3.4 v6.8 Z" fill="#FFFFFF" '
        'stroke="#9673A6" stroke-width="0.9"/>'
    ) in out


def test_lane_frames_colors_titles_and_the_banner_shrink() -> None:
    """Each frame uses its lane's locked stroke and fill. The banner shrinks by 1 until
    len * size * 0.62 fits lane.w - 52 - width("N classes") - 16: GENOTYPE (614 wide)
    keeps 34.0; EXPERIMENT in 310 has 179.99 available, 10 * 0.62 * s <= 179.99 gives
    29.0; CONTROLLED VOCABULARIES (23 characters) gives 12.0 and, with no dash
    separator in its title, an empty subtitle; the empty 52-wide environment frame
    bottoms out at the 11.0 floor and its subtitle is cut to one letter.
    """
    graph = _graph()
    out = svg._lane_frames_svg(graph, svg.build_layout(graph))
    frames = re.findall(r'<g class="lane" id="lane-(\w+)">(.*?)</g>', out)
    assert [k for k, _ in frames] == list(LANE_ORDER)
    parsed: dict[str, tuple[tuple[str, ...], list[tuple[str, str]]]] = {}
    for key, body in frames:
        rect = re.search(r'fill="(#\w+)" fill-opacity="0.26" stroke="(#\w+)"', body)
        assert rect is not None
        found = re.findall(r'font-size="([\d.]+)"[^>]*>([^<]*)</text>', body)
        parsed[key] = (rect.groups(), found)
    for key, ((fill, stroke), _) in parsed.items():
        assert (fill, stroke) == (LANE_FILL[key], LANE_STROKE[key]), key
    texts = {k: v[1] for k, v in parsed.items()}
    assert texts["genotype"] == [
        ("34.0", "GENOTYPE"),
        ("13.0", "what was changed in the cell"),
        ("13.0", "3 classes"),
    ]
    assert texts["experiment"][0] == ("29.0", "EXPERIMENT")
    assert texts["enum"] == [
        ("12.0", "CONTROLLED VOCABULARIES"),
        ("13.0", ""),
        ("13.0", "1 classes"),
    ]
    assert texts["environment"] == [
        ("11.0", "ENVIRONMENT"),
        ("13.0", "w" + ELLIPSIS),
        ("13.0", "0 classes"),
    ]


def test_legend_rows_and_box_height_by_mode() -> None:
    """Full mode: 7 rows, box 258 * 1.55 = 399.9 wide and 28 + 7 * 15 = 133 tall. Compact
    (scale 1.35): 6 rows, 258 * 2.3 = 593.4 wide, 37.8 + 6 * 20.25 = 159.3 tall. A URL
    adds one "explore it" row (+15 or +20.25) whose text is escaped.
    """
    layout = svg.Layout()

    def box(document: str) -> tuple[str | None, str | None, int]:
        rect = re.search(
            r'<rect x="0.0" y="0.0" width="([\d.]+)" height="([\d.]+)"', document
        )
        assert rect is not None
        return (
            rect.group(1),
            rect.group(2),
            document.count('font-weight="700" fill="#4A4A4A"'),
        )

    assert box(svg._legend_svg(layout, 0.0, 0.0)) == ("399.9", "133.0", 7)
    assert box(svg._legend_svg(layout, 0.0, 0.0, compact=True)) == ("593.4", "159.3", 6)
    with_url = svg._legend_svg(layout, 0.0, 0.0, explore_url="https://x/?a=1&b=2")
    assert box(with_url) == ("399.9", "148.0", 8)
    assert ">explore it</text>" in with_url
    assert 'fill="#6C8EBF">https://x/?a=1&amp;b=2</text>' in with_url


# ---------------------------------------------------------------- full documents


@pytest.mark.parametrize(
    ("compact", "url", "vb_h", "height_mm"),
    [
        (False, "https://x/?a=1&b=2", 727.2, "62.28"),
        (False, None, 712.2, "61.00"),
        (True, None, 640.58, "54.86"),
    ],
)
def test_render_svg_counts_header_and_a_legend_that_ends_inside_the_canvas(
    compact: bool, url: str | None, vb_h: float, height_mm: str
) -> None:
    """The document parses as XML (every label escaped), holds one node per class, six
    lane frames, three inheritance elbows, four composition curves (three backbone), and
    the header "7 classes, 1 controlled vocabularies, 8 declared fields".

    Height: layout height + 80 + header (96, compact 72) + legend band
    26 + 28 s + 15 s (rows + url) + 16. Full with URL: 361.2 + 176 + 190 = 727.2; full
    without: 712.2; compact (layout height 287.28, s = 1.35, 6 rows) is
    287.28 + 152 + 26 + 37.8 + 121.5 + 16 = 640.58. The legend box ends at exactly
    vb_h - 40 - 16 in all three, which is the derivation that stopped it clipping.
    """
    graph = _graph()
    layout = svg.build_layout(graph, compact=compact)
    document = svg.render_svg(
        graph, layout, 179.0, "A & B", "sub", compact=compact, explore_url=url
    )
    root = ET.fromstring(document)
    view = [float(v) for v in (root.get("viewBox") or "").split()]
    assert view == pytest.approx([0.0, 0.0, 2090.0, vb_h], abs=0.05)
    assert (root.get("width"), root.get("height")) == ("179.00mm", f"{height_mm}mm")
    assert len(_elements(document, "g", "node")) == 8
    assert len(_elements(document, "g", "lane")) == 6
    assert len(_elements(document, "path", "inherit")) == 3
    assert len(_elements(document, "path", "compose")) == 1
    assert len(_elements(document, "path", "compose backbone")) == 3
    texts = [t.text for t in root.iter(f"{NS}text")]
    assert texts[:3] == [
        "A & B",
        "sub",
        f"7 classes {MIDDOT} 1 controlled vocabularies {MIDDOT} 8 declared fields "
        f"{MIDDOT} generated from torchcell.datamodels.schema",
    ]
    (legend,) = _elements(document, "g", "legend")
    rect = legend.find(f"{NS}rect")
    assert rect is not None
    bottom = float(rect.get("y") or "nan") + float(rect.get("height") or "nan")
    assert bottom == pytest.approx(view[3] - 56.0)
    (diagram,) = [g for g in root.iter(f"{NS}g") if g.get("id") == "diagram"]
    assert diagram.get("transform") == (
        "translate(40.0,136.0)" if not compact else "translate(40.0,112.0)"
    )


def test_wrap_csv_breaks_before_the_overflowing_name_and_suffix_family() -> None:
    """At 6 pt a character is 3.18 wide: "A1, B2," is 7 * 3.18 = 22.26 and fits 25, adding
    " C3" (10 characters, 31.8) does not, so C3 starts line 2. A name wider than the line
    stays alone rather than being split. The suffix family strips a shared parent
    suffix and refuses an empty list, a child equal to the parent, or one child
    without the suffix.
    """
    assert svg._wrap_csv(["A1", "B2", "C3"], 25.0, 6.0) == ["A1, B2,", "C3"]
    assert svg._wrap_csv(["Longname", "B"], 5.0, 6.0) == ["Longname,", "B"]
    assert svg._suffix_family("Phenotype", ["FitnessPhenotype", "GrowthPhenotype"]) == [
        "Fitness",
        "Growth",
    ]
    assert svg._suffix_family("Phenotype", []) is None
    assert svg._suffix_family("Phenotype", ["Phenotype"]) is None
    assert svg._suffix_family("Phenotype", ["FitnessPhenotype", "Media"]) is None


def test_lane_body_lines_for_roots_children_tails_and_flat_roots() -> None:
    """Genotype lane: branching root GenePerturbation (1 subtype; its child does not end
    in "GenePerturbation", so it is listed by name, indented 7, no tail), then the flat
    root Genotype. A child with its own in-lane subtree gets a middle-dot tail with its
    descendant count: Base -> {Leaf, Mid}, Mid -> Deep reads "Base, 3 subtypes",
    "Leaf", "Mid, 1" (each comma here is the two-space middle-dot separator).
    """
    graph = _graph()
    assert svg._lane_body_lines(graph, "genotype", 140.0) == [
        (0.0, f"GenePerturbation  {MIDDOT}  1 subtypes"),
        (7.0, "DeletionPerturbation"),
        (0.0, "Genotype"),
    ]
    assert svg._lane_body_lines(graph, "phenotype", 140.0) == [
        (0.0, f"Phenotype  {MIDDOT}  1 subtypes"),
        (7.0, "Fitness"),
    ]
    tree = OntologyGraph(
        classes={
            "Base": _model("Base", "provenance"),
            "Leaf": _model("Leaf", "provenance", parent="Base"),
            "Mid": _model("Mid", "provenance", parent="Base"),
            "Deep": _model("Deep", "provenance", parent="Mid"),
        }
    )
    assert svg._lane_body_lines(tree, "provenance", 140.0) == [
        (0.0, f"Base  {MIDDOT}  3 subtypes"),
        (7.0, "Leaf"),
        (7.0, f"Mid  {MIDDOT}  1"),
    ]


def test_lane_body_lines_reuse_a_repeated_family_and_cap_the_line_count() -> None:
    """Experiment and ExperimentReference both strip to [Fitness, Growth], so the second
    prints "same 2 families as Experiment". With ``max_lines`` 3 the 4 lines become the
    first 2 plus a note that 4 - 3 + 1 = 2 lines were dropped.
    """
    names = [
        ("Experiment", None),
        ("FitnessExperiment", "Experiment"),
        ("GrowthExperiment", "Experiment"),
        ("ExperimentReference", None),
        ("FitnessExperimentReference", "ExperimentReference"),
        ("GrowthExperimentReference", "ExperimentReference"),
    ]
    graph = OntologyGraph(
        classes={n: _model(n, "experiment", parent=p) for n, p in names}
    )
    full = svg._lane_body_lines(graph, "experiment", 140.0)
    assert full == [
        (0.0, f"Experiment  {MIDDOT}  2 subtypes"),
        (7.0, "Fitness, Growth"),
        (0.0, f"ExperimentReference  {MIDDOT}  2 subtypes"),
        (7.0, "same 2 families as Experiment"),
    ]
    capped = svg._lane_body_lines(graph, "experiment", 140.0, max_lines=3)
    assert capped == [
        *full[:2],
        (0.0, f"{ELLIPSIS}  +2 more lines \u2014 full list in the interactive map"),
    ]


def test_schematic_blocks_arrows_footer_and_the_elbow_route() -> None:
    """In points: 179 mm = 507.4; columns (507.4 - 20 - 18) / 3 = 156.47 wide at x 10,
    175.47, 340.93. Block height 20 + 7.6 lines + 7: genotype (3 lines) 49.8,
    environment (0) 27, experiment and provenance and enum (1) 34.6, phenotype (2) 42.2.
    Genotype at y 36, environment at 36 + 49.8 + 9 = 94.8, left stack 85.8 tall;
    experiment centered at 36 + (85.8 - 34.6) / 2 = 61.6, phenotype at 57.8; the
    second row at 36 + 85.8 + 14.4 = 136.2 with enum spanning two columns (321.9).
    Footer at 136.2 + 34.6 + 13 = 183.8, canvas 191.8 pt = 67.66 mm.

    The experiment -> phenotype arrow is straight: both centers sit at 78.9, so it is
    "M331.9 78.9 H336.5" (tip 4.4 short of 340.9). The genotype -> experiment arrow is
    an elbow from (166.47, 60.9) to the head base at 175.47 - 4.4 = 171.07, y 74.1.

    Contract (issue #541): the vertical leg sits midway between the source and the head
    base, mid = (166.47 + 171.07) / 2 = 168.77, with radius min(3, 13.2 / 2, 4.6 / 2)
    = 2.3, so the first arc starts at mid - r = 166.47 and the last arc ends at
    mid + r = 171.07, exactly the tip: neither horizontal leg runs backward (before the
    fix the last leg ran 2.9 units backward under the head, from 174.0 to 171.1).
    """
    document = svg.render_schematic_svg(_graph(), explore_url="https://x")
    root = ET.fromstring(document)
    assert root.get("viewBox") == "0 0 507.4 191.8"
    assert root.get("height") == "67.66mm"
    rects = [
        tuple(r.get(k) for k in ("x", "y", "width", "height"))
        for r in root.iter(f"{NS}rect")
        if r.get("rx") == "3"
    ]
    assert rects == [
        ("10.0", "36.0", "156.5", "49.8"),
        ("10.0", "94.8", "156.5", "27.0"),
        ("175.5", "61.6", "156.5", "34.6"),
        ("340.9", "57.8", "156.5", "42.2"),
        ("10.0", "136.2", "156.5", "34.6"),
        ("175.5", "136.2", "321.9", "34.6"),
    ]
    spines = [p.get("d") for p in root.iter(f"{NS}path") if p.get("fill") == "none"]
    assert spines == [
        "M166.5 60.9 H166.5 A2.3 2.3 0 0 1 168.8 63.2 V71.8 "
        "A2.3 2.3 0 0 0 171.1 74.1 H171.1",
        "M166.5 108.3 H166.5 A2.3 2.3 0 0 0 168.8 106.0 V86.0 "
        "A2.3 2.3 0 0 1 171.1 83.7 H171.1",
        "M331.9 78.9 H336.5",
    ]
    texts = [t.text for t in root.iter(f"{NS}text")]
    assert texts[-2:] == [
        f"7 pydantic classes {MIDDOT} 8 declared fields {MIDDOT} generated from "
        "torchcell.datamodels.schema",
        "full interactive map: https://x",
    ]
    assert "Fitness" in texts and "ProvenanceGapMixin" in texts


# ---------------------------------------------------------------- the shipped schema


@pytest.fixture(scope="module")
def real_graph() -> OntologyGraph:
    return build_ontology_graph()


@pytest.mark.parametrize("compact", [False, True])
def test_real_schema_map_has_one_mark_per_class_parent_link_and_edge(
    real_graph: OntologyGraph, compact: bool
) -> None:
    """The docs explorer's map (179 mm, full) and the overview (compact), rendered in
    memory. Counts are derived from the graph, not typed: nodes = classes; lanes = 6;
    inheritance elbows = classes whose parent is a class; composition curves = distinct
    (source, target, field) edges with both ends drawn, of which the backbone ones are
    those in BACKBONE_EDGES; the header repeats the model, enum and field counts.
    """
    graph = real_graph
    document = svg.render_svg(
        graph,
        svg.build_layout(graph, compact=compact),
        179.0,
        "t",
        "s",
        compact=compact,
        explore_url="https://example.org/ontology/",
    )
    classes = graph.classes
    edges = {
        (e.source, e.target, e.field_name)
        for e in graph.composition_edges
        if e.source in classes and e.target in classes
    }
    backbone = {e for e in edges if (e[0], e[1]) in svg.BACKBONE_EDGES}
    parents = [c for c in classes.values() if c.parent in classes]
    assert len(_elements(document, "g", "node")) == len(classes)
    assert len(_elements(document, "g", "lane")) == 6
    assert len(_elements(document, "path", "inherit")) == len(parents)
    assert len(_elements(document, "path", "compose backbone")) == len(backbone)
    assert len(_elements(document, "path", "compose")) == len(edges) - len(backbone)
    n_models = sum(c.kind == "model" for c in classes.values())
    n_enums = sum(c.kind == "enum" for c in classes.values())
    n_fields = sum(len(c.own_fields) for c in classes.values())
    counts = f"{n_models} classes {MIDDOT} {n_enums} controlled vocabularies"
    root = ET.fromstring(document)
    header = [t.text for t in root.iter(f"{NS}text")][2]
    assert header is not None and header.startswith(
        f"{counts} {MIDDOT} {n_fields} declared fields"
    )
    assert {b for b in svg.BACKBONE_EDGES if b not in {e[:2] for e in edges}} == {
        ("Genotype", "GenePerturbation")
    }


def test_real_schema_genotype_backbone_edge_is_never_drawn_bold(
    real_graph: OntologyGraph,
) -> None:
    """Finding: ``Genotype.perturbations`` references the concrete perturbation classes,
    not ``GenePerturbation``, so the BACKBONE_EDGES entry ("Genotype",
    "GenePerturbation") (ontology_svg.py:63) matches no edge: the map draws one faint
    hairline per concrete class instead of the bold genotype-to-perturbation spine the
    legend promises. Pinned until the backbone test accepts a subclass target.
    Left open by the issue #541 fix: matching every descendant of GenePerturbation was
    rendered and drew twenty bold curves that bury the genotype lane, so which edge
    is the genotype spine is a figure-design decision for the author.
    """
    concrete: list[str] = [
        e.target
        for e in real_graph.composition_edges
        if (e.source, e.field_name) == ("Genotype", "perturbations")
    ]
    assert "GenePerturbation" not in concrete
    assert len(concrete) == len(set(concrete)) > 1
    assert set(concrete) <= set(real_graph.descendants_of("GenePerturbation"))
    document = svg._composition_svg(real_graph, svg.build_layout(real_graph))
    from_genotype = re.findall(
        r'<path class="(compose[^"]*)" data-src="Genotype" data-dst="([^"]+)"', document
    )
    assert sorted(dst for _, dst in from_genotype) == sorted(concrete)
    assert {cls for cls, _ in from_genotype} == {"compose"}


def test_real_schema_schematic_uses_exactly_the_four_point_sizes(
    real_graph: OntologyGraph,
) -> None:
    """The schematic's docstring promises Nature-legible type because the viewBox is in
    points: the panel emits exactly four sizes, 6 (body lines, counts, glosses), 6.5, 7
    and the 9 pt title (lines 806 to 849), nothing below the 6 pt floor and nothing
    else. The footer counts match the graph.
    """
    document = svg.render_schematic_svg(real_graph)
    sizes = {float(s) for s in re.findall(r'font-size="([\d.]+)"', document)}
    assert sizes == {6.0, 6.5, 7.0, 9.0}
    n_models = sum(c.kind == "model" for c in real_graph.classes.values())
    assert f"{n_models} pydantic classes &#183; " in document
