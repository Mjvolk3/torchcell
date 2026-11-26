# tests/torchcell/metabolism/test_bipartite.py
# [[tests.torchcell.metabolism.test_bipartite]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/metabolism/test_bipartite.py
"""``cobra_bipartite_graph`` over a three-reaction model, and against ``YeastGEM``.

The toy, the same shape ``test_yeast_GEM_synthetic`` uses so the two builders can be
compared on identical input:

* ``R1: A_c <=> B_c`` reversible, GPR ``(g1 and g2) or g3``
* ``R2: 2 B_c --> C_c`` irreversible, GPR ``g4``
* ``EX_A: A_c <=>`` reversible, no GPR

That gives 7 reaction nodes: R1 two OR terms times two directions, R2 one forward,
EX_A a forward and a reverse ``noGene`` pair. Every edge points reaction to
metabolite, ``stoichiometry`` is the absolute coefficient, and ``edge_type`` flips
between the forward and reverse node of the same reaction.

:func:`test_matches_yeast_gem_builder_on_the_same_model` is the one that matters: it
asserts the extraction is byte-for-byte the same structure ``YeastGEM.bipartite_graph``
produces, since the whole point of the module is that the yeast path can migrate onto
it without changing what ``to_cell_data`` sees.
"""

from pathlib import Path

import cobra
import pytest

from torchcell.metabolism.bipartite import (
    GROWTH_SUBSYSTEM,
    MemberlessReactionError,
    cobra_bipartite_graph,
    parse_gene_combinations,
    refuse_memberless_reactions,
)


def _toy_model() -> cobra.Model:
    model = cobra.Model("toy")
    a = cobra.Metabolite(
        "A_c", name="alpha", formula="C6H12O6", compartment="c", charge=0
    )
    b = cobra.Metabolite(
        "B_c", name="beta", formula="C3H4O3", compartment="c", charge=-1
    )
    c = cobra.Metabolite(
        "C_c", name="gamma", formula="C6H8O6", compartment="c", charge=-2
    )
    r1 = cobra.Reaction("R1", lower_bound=-1000.0, upper_bound=1000.0)
    r1.add_metabolites({a: -1.0, b: 1.0})
    r2 = cobra.Reaction("R2", lower_bound=0.0, upper_bound=1000.0)
    r2.add_metabolites({b: -2.0, c: 1.0})
    ex = cobra.Reaction("EX_A", lower_bound=-10.0, upper_bound=1000.0)
    ex.add_metabolites({a: -1.0})
    model.add_reactions([r1, r2, ex])
    r1.gene_reaction_rule = "(g1 and g2) or g3"
    r2.gene_reaction_rule = "g4"
    r1.subsystem = "glycolysis"
    r2.subsystem = "pentose phosphate"
    return model


def test_parse_gene_combinations_splits_or_then_and() -> None:
    """A DNF rule becomes one AND-set per OR term; an empty rule is one empty set."""
    assert parse_gene_combinations("(g1 and g2) or g3") == [{"g1", "g2"}, {"g3"}]
    assert parse_gene_combinations("g4") == [{"g4"}]
    assert parse_gene_combinations("") == [set()]


def test_reaction_nodes_are_one_per_or_term_and_direction() -> None:
    """R1 yields 4 nodes, R2 one, EX_A a noGene pair."""
    graph = cobra_bipartite_graph(_toy_model())
    reactions = sorted(
        node for node, data in graph.nodes(data=True) if data["node_type"] == "reaction"
    )
    assert reactions == [
        "EX_A_noGene_fwd",
        "EX_A_noGene_rev",
        "R1_comb0_fwd",
        "R1_comb0_rev",
        "R1_comb1_fwd",
        "R1_comb1_rev",
        "R2_comb0_fwd",
    ]
    metabolites = sorted(
        node
        for node, data in graph.nodes(data=True)
        if data["node_type"] == "metabolite"
    )
    assert metabolites == ["A_c", "B_c", "C_c"]


def test_reaction_node_carries_the_and_set_and_swaps_roles_on_reverse() -> None:
    """``genes`` is the OR term's AND-set; reverse swaps reactants and products."""
    graph = cobra_bipartite_graph(_toy_model())
    forward = graph.nodes["R1_comb0_fwd"]
    reverse = graph.nodes["R1_comb0_rev"]
    assert forward["genes"] == {"g1", "g2"}
    assert graph.nodes["R1_comb1_fwd"]["genes"] == {"g3"}
    assert forward["reactants"] == ["A_c"]
    assert forward["products"] == ["B_c"]
    assert reverse["reactants"] == ["B_c"]
    assert reverse["products"] == ["A_c"]
    assert forward["reversibility"] is True
    assert graph.nodes["R2_comb0_fwd"]["genes"] == {"g4"}
    assert graph.nodes["EX_A_noGene_fwd"]["genes"] == set()


def test_edge_type_flips_with_direction_and_stoichiometry_is_absolute() -> None:
    """R2's B_c edge carries 2.0, and A_c is a reactant forward, a product reverse."""
    graph = cobra_bipartite_graph(_toy_model())
    assert graph.edges["R2_comb0_fwd", "B_c"]["stoichiometry"] == 2.0
    assert graph.edges["R2_comb0_fwd", "B_c"]["edge_type"] == "reactant"
    assert graph.edges["R2_comb0_fwd", "C_c"]["edge_type"] == "product"
    assert graph.edges["R1_comb0_fwd", "A_c"]["edge_type"] == "reactant"
    assert graph.edges["R1_comb0_rev", "A_c"]["edge_type"] == "product"
    assert graph.edges["R1_comb0_rev", "A_c"]["stoichiometry"] == 1.0
    assert graph.edges["R1_comb0_rev", "A_c"]["direction"] == "reverse"
    assert graph.edges["R2_comb0_fwd", "B_c"]["reaction_id"] == "R2"


def test_metabolite_nodes_carry_the_attributes_to_cell_data_reads() -> None:
    """``composition`` is the formula, and charge and compartment come through."""
    graph = cobra_bipartite_graph(_toy_model())
    node = graph.nodes["B_c"]
    assert node["name"] == "beta"
    assert node["composition"] == "C3H4O3"
    assert node["compartment"] == "c"
    assert node["charge"] == -1
    assert node["miriam"] == ""


def test_growth_reaction_ids_relabel_subsystem_and_keep_the_source() -> None:
    """A named reaction gets ``Growth``; its own subsystem survives separately."""
    graph = cobra_bipartite_graph(_toy_model(), growth_reaction_ids=("R2",))
    assert graph.nodes["R2_comb0_fwd"]["subsystem"] == GROWTH_SUBSYSTEM
    assert graph.nodes["R2_comb0_fwd"]["subsystem_source"] == "pentose phosphate"
    assert graph.nodes["R1_comb0_fwd"]["subsystem"] == "glycolysis"
    assert graph.nodes["R1_comb0_fwd"]["subsystem_source"] == "glycolysis"


def test_induced_gene_set_drops_only_fully_outside_terms() -> None:
    """``{g1, g2}`` survives on one gene; ``{g3}`` and ``{g4}`` go."""
    graph = cobra_bipartite_graph(_toy_model(), induced_gene_set={"g1"})
    reactions = sorted(
        node for node, data in graph.nodes(data=True) if data["node_type"] == "reaction"
    )
    assert reactions == [
        "EX_A_noGene_fwd",
        "EX_A_noGene_rev",
        "R1_comb0_fwd",
        "R1_comb0_rev",
    ]


def test_memberless_reaction_is_refused_by_name() -> None:
    """A reaction with no metabolites raises and the message names it."""
    model = _toy_model()
    model.add_reactions([cobra.Reaction("EMPTY", lower_bound=0.0, upper_bound=1.0)])
    with pytest.raises(MemberlessReactionError, match="EMPTY"):
        refuse_memberless_reactions(model)
    with pytest.raises(MemberlessReactionError, match="EMPTY"):
        cobra_bipartite_graph(model)


def test_matches_yeast_gem_builder_on_the_same_model(tmp_path: Path) -> None:
    """The extraction reproduces ``YeastGEM.bipartite_graph`` exactly.

    ``subsystem_source`` is the one key the extraction adds, and with no growth
    reaction named it always equals ``subsystem``, so it is excluded from the
    comparison and asserted separately.
    """
    from torchcell.metabolism.yeast_GEM import YeastGEM

    model = _toy_model()
    checkout = tmp_path / "yeast-GEM-9.0.2" / "model"
    checkout.mkdir(parents=True)
    cobra.io.write_sbml_model(model, str(checkout / "yeast-GEM.xml"))
    reference = YeastGEM(root=str(tmp_path)).bipartite_graph
    candidate = cobra_bipartite_graph(YeastGEM(root=str(tmp_path)).model)

    assert set(reference.nodes) == set(candidate.nodes)
    assert set(reference.edges) == set(candidate.edges)
    for node in reference.nodes:
        expected = dict(reference.nodes[node])
        observed = dict(candidate.nodes[node])
        source = observed.pop("subsystem_source", None)
        assert observed == expected, node
        if observed["node_type"] == "reaction":
            assert source == observed["subsystem"], node
    for edge in reference.edges:
        assert dict(candidate.edges[edge]) == dict(reference.edges[edge]), edge
