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

from torchcell.datamodels import compound_identity as compound_identity_module
from torchcell.datamodels import media as media_module
from torchcell.datamodels import pydant as pydant_module
from torchcell.datamodels import schema as schema_module
from torchcell.paper.ontology_graph import (
    BEHAVIOR_MIXINS,
    LANE_HEADINGS,
    LANE_ORDER,
    LANE_ROOTS,
    LANE_SUBTITLES,
    OntologyGraph,
    _lane_for,
    build_ontology_graph,
)
from torchcell.paper.ontology_svg import _lane_body_lines, build_layout

# CamelCase runs are how a class name looks on the panel; the derived lines otherwise
# contain only counts and lowercase connective words.
CLASS_TOKEN = re.compile(r"\b[A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*\b")

# Words that legitimately appear capitalised in a derived line without naming a class.
NON_CLASS_TOKENS = frozenset({"Experiment", "ExperimentReference"})


@pytest.fixture(scope="module")
def graph():
    return build_ontology_graph()


def _model_bases(name: str, graph: OntologyGraph) -> list[str]:
    """``name``'s direct bases that the graph draws as models, from the LIVE class.

    The graph's own ``parent`` is one answer to "which base is this class's kind";
    this reads the bases again from the introspected modules so a test can check that
    answer instead of restating it.
    """
    for module in (
        schema_module,
        media_module,
        pydant_module,
        compound_identity_module,
    ):
        obj = getattr(module, name, None)
        if isinstance(obj, type):
            return [
                base.__name__
                for base in obj.__bases__
                if base.__name__ in graph.classes
                and graph.classes[base.__name__].kind == "model"
            ]
    raise AssertionError(f"{name} is in the graph but in none of its modules")


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


def test_layout_never_moves_a_card_out_of_its_lane(graph):
    """A laid-out card must stay in the lane the graph assigned it.

    Inheritance crosses lane boundaries: Phenotype, Compound and Environment all
    derive from ProvenanceGapMixin, which lives in the provenance lane. _place_tree
    once recursed through the unfiltered child list and stamped every descendant with
    the lane it was called with, so the provenance pass overwrote all fourteen
    phenotype cards and the phenotype frame rendered empty. The drift check cannot
    catch that, because a consistently wrong render matches a consistently wrong
    commit, so the invariant is pinned here instead.
    """
    for compact in (True, False):
        layout = build_layout(graph, compact=compact)
        misplaced = {
            name: (graph.classes[name].lane, card.lane)
            for name, card in layout.cards.items()
            if card.lane != graph.classes[name].lane
        }
        assert not misplaced, f"compact={compact}: cards moved lane: {misplaced}"


def test_every_lane_frame_holds_its_full_membership(graph):
    """Each lane's laid-out cards are exactly its graph members, no more, no fewer."""
    for compact in (True, False):
        layout = build_layout(graph, compact=compact)
        for lane in LANE_ORDER:
            assert sorted(layout.lanes[lane].card_names) == graph.lane_members(lane), (
                f"compact={compact}: lane {lane} membership diverged from the graph"
            )


def test_behavior_mixin_listed_first_is_not_drawn_as_the_parent(graph):
    """``HashableProvenanceGapMixin`` precedes a leaf's domain base in its bases (the
    MRO keeps the mixin's ``__hash__``), so taking the first base filed the three #507
    perturbation leaves under the mixin, in the provenance lane. The drawn parent is
    the first base that is not a ``BEHAVIOR_MIXINS`` entry; ``ConstructedOrf``, whose
    only schema base is the mixin, keeps it.
    """
    parents = {
        name: (graph.classes[name].parent, graph.classes[name].lane)
        for name in (
            "BarcodedKanMxDeletionPerturbation",
            "HeterozygousDeletionPerturbation",
            "ConditionalAllelePerturbation",
            "ConstructedOrf",
            "HashableProvenanceGapMixin",
        )
    }
    assert parents == {
        "BarcodedKanMxDeletionPerturbation": ("KanMxDeletionPerturbation", "genotype"),
        "HeterozygousDeletionPerturbation": ("PresenceAbsencePerturbation", "genotype"),
        "ConditionalAllelePerturbation": ("SequencePerturbation", "genotype"),
        "ConstructedOrf": ("HashableProvenanceGapMixin", "genotype"),
        "HashableProvenanceGapMixin": ("ProvenanceGapMixin", "provenance"),
    }
    # The children are DERIVED, not listed. A hand-kept list went stale the moment
    # three lanes of the ontology unification each added a mixin-only class
    # (ReferenceGenome took the mixin for interning, ConstructPlacement,
    # ConstructVariantCall and HostPlacement are composed value objects), while the
    # property worth pinning never moves: the mixin is drawn as a parent for exactly
    # those classes whose only schema-model base IS a behavior mixin. Re-derived here
    # from the live bases, so a leaf that gains a domain base and still lands under
    # the mixin fails, which is the regression this test is named for.
    mixin_only = sorted(
        name
        for name, entry in graph.classes.items()
        if entry.kind == "model"
        and (bases := _model_bases(name, graph))
        and set(bases) <= set(BEHAVIOR_MIXINS)
    )
    assert graph.children_of("HashableProvenanceGapMixin") == mixin_only
    assert "ConstructedOrf" in mixin_only and "NaturalIsolateDescriptor" in mixin_only
    assert not {
        "BarcodedKanMxDeletionPerturbation",
        "HeterozygousDeletionPerturbation",
        "ConditionalAllelePerturbation",
    } & set(mixin_only)
    assert graph.classes["NaturalIsolateDescriptor"].lane == "genotype"
    assert {
        "BarcodedKanMxDeletionPerturbation",
        "HeterozygousDeletionPerturbation",
        "ConditionalAllelePerturbation",
    } <= set(graph.descendants_of("GenePerturbation"))
