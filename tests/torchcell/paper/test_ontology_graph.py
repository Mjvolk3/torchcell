"""The ontology figure must stay derived from the schema, not typed alongside it.

An earlier revision of the schematic panel carried hand-written exemplar lines while
claiming to read them off the graph. It drifted silently: the panel named
``SourcedValue`` (a class in a module the introspection never reads) and omitted
``ProvenanceGapMixin`` (one it does), all while printing a computed class count beside
the stale list. These tests pin the two properties that prevent a repeat -- every name
on the panel resolves to a real class, and an unassignable class fails loudly.
"""

from __future__ import annotations

import re

import pytest

from torchcell.paper.ontology_graph import (
    LANE_HEADINGS,
    LANE_ORDER,
    LANE_ROOTS,
    LANE_SUBTITLES,
    _lane_for,
    build_ontology_graph,
)
from torchcell.paper.ontology_svg import _lane_body_lines

# CamelCase runs are how a class name looks on the panel; the derived lines otherwise
# contain only counts and lowercase connective words.
CLASS_TOKEN = re.compile(r"\b[A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*\b")

# Words that legitimately appear capitalised in a derived line without naming a class.
NON_CLASS_TOKENS = frozenset({"Experiment", "ExperimentReference"})


@pytest.fixture(scope="module")
def graph():
    return build_ontology_graph()


def test_every_lane_has_a_heading_and_gloss():
    assert set(LANE_HEADINGS) == set(LANE_ORDER)
    assert set(LANE_SUBTITLES) == set(LANE_ORDER)


def test_panel_names_only_classes_that_exist(graph):
    """No name printed on the schematic may be absent from the live schema."""
    known = set(graph.classes)
    for lane in LANE_ORDER:
        for _, text in _lane_body_lines(graph, lane, max_w=140.0):
            for token in CLASS_TOKEN.findall(text):
                if token in NON_CLASS_TOKENS:
                    continue
                # Family lines strip a shared suffix, so re-attach the lane's roots
                # before deciding a token is unknown.
                candidates = {token} | {
                    token + root for root in LANE_ROOTS.get(lane, ())
                }
                candidates |= {token + suffix for suffix in ("Phenotype", "Experiment")}
                assert candidates & known, (
                    f"{lane} panel prints {token!r}, which is not a class in the "
                    f"schema -- the line was hand-authored or has gone stale"
                )


def test_panel_counts_match_the_graph(graph):
    """A "N subtypes" count on the panel must equal the real lane-scoped subtree."""
    for lane in LANE_ORDER:
        for _, text in _lane_body_lines(graph, lane, max_w=140.0):
            match = re.match(r"^(\w+)\s+·\s+(\d+) subtypes$", text)
            if match is None:
                continue
            name, claimed = match.group(1), int(match.group(2))
            actual = len(
                [d for d in graph.descendants_of(name) if graph.classes[d].lane == lane]
            )
            assert claimed == actual, (
                f"panel claims {name} has {claimed} subtypes in {lane}, graph has "
                f"{actual}"
            )


def test_unassignable_class_raises_instead_of_defaulting():
    """A class with no lane must fail the build, not land in a catch-all lane."""
    with pytest.raises(KeyError, match="has no lane"):
        _lane_for("SomeBrandNewTopLevelModel", [])


def test_every_class_lands_in_a_declared_lane(graph):
    assert {c.lane for c in graph.classes.values()} <= set(LANE_ORDER)


def test_compound_identity_is_introspected(graph):
    """The module was added to the graph; regressing it would silently shrink the map."""
    assert "CompoundIdentityRecord" in graph.classes
    assert graph.classes["CompoundIdentityRecord"].lane == "environment"
    assert graph.classes["CompoundResolutionStatus"].lane == "enum"
