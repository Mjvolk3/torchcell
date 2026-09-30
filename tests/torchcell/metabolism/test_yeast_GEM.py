# tests/torchcell/metabolism/test_yeast_GEM.py
# [[tests.torchcell.metabolism.test_yeast_GEM]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/metabolism/test_yeast_GEM.py
"""Tests for the YeastGEM metabolic model wrapper.

``YeastGEM()`` downloads the yeast-GEM release from GitHub when the checkout under
``$DATA_ROOT/data/torchcell/yeast-GEM`` is absent, so the fixture reads the checkout
when it exists, downloads into a tmp root only under ``--network``, and skips otherwise
(a plain ``pytest`` never reaches the network and never writes under ``DATA_ROOT``).

2026.09.30 - hand-built model (Phase 15). The tests after the real-model block run on a
five-reaction cobra model written to SBML under ``tmp_path`` (``_toy_model`` below; the
three-reaction model of ``test_yeast_GEM_synthetic.py`` covers the OR and reversible GPR
cases):

* ``RBIG: A_c + 2 B_c --> C_c + 3 D_e`` (irreversible), GPR ``g1 and g2 and g3``: one
  hyperedge ``RBIG_comb0_fwd`` with gene set {g1, g2, g3} and the stoichiometric row
  (-1, -2, 1, 3) in species order; bipartite edges carry |coef| (1, 2, 1, 3);
* ``EX_A_irr: A_c -->`` (irreversible, no GPR): a forward ``noGene`` edge only;
* ``T_A: A_c <=> A_e`` (reversible, no GPR): ``noGene`` forward and reverse, the
  reverse row negated to (1, -1);
* ``R_OTHER: B_c --> C_c`` (irreversible, no GPR);
* ``EMPTY: -->`` (no species): a bipartite reaction node with no edge (isolated), and
  no hyperedge at all (hypernetx drops an edge with no members).

So the hypergraph has 5 edges on 5 nodes and the bipartite graph 11 edges on 11 nodes
(6 reaction units, 5 metabolites). No solver runs and no network is reached.
"""

import os
import os.path as osp
import random
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cobra
import hypernetx as hnx
import matplotlib.pyplot as plt
import networkx as nx
import pytest
import requests
from networkx.drawing.layout import kamada_kawai_layout, spring_layout

import torchcell.metabolism.yeast_GEM as yeast_gem_module
from torchcell.metabolism.yeast_GEM import (
    YeastGEM,
    analyze_reactions_without_genes,
    plot_full_network,
    plot_random_network,
    sanity_check_metabolic_networks,
)
from torchcell.sequence import GeneSet

_CHECKOUT = osp.join(
    os.environ.get("DATA_ROOT", "/nonexistent"), "data/torchcell/yeast-GEM"
)


@pytest.fixture(scope="module")
def yeast_gem(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> YeastGEM:
    """The real yeast-GEM model: from the checkout, or downloaded under --network."""
    if osp.isdir(_CHECKOUT):
        return YeastGEM(root=_CHECKOUT)
    if request.config.getoption("--network"):
        return YeastGEM(root=str(tmp_path_factory.mktemp("yeast_gem")))
    pytest.skip(
        "yeast-GEM checkout absent under $DATA_ROOT; pass --network to download"
    )


def test_reaction_map_exists(yeast_gem):
    """Test that reaction_map exists and returns a hypergraph."""
    reaction_map = yeast_gem.reaction_map
    assert isinstance(reaction_map, hnx.Hypergraph)
    assert len(reaction_map.edges) > 0


def test_reaction_map_gene_rule_parsing(yeast_gem):
    """Test that gene rules are correctly parsed to AND combinations or empty sets."""
    reaction_map = yeast_gem.reaction_map
    model = yeast_gem.model

    # Map to store reactions by ID for easier lookup
    reactions_by_id = {}
    for reaction in model.reactions:
        reactions_by_id[reaction.id] = reaction

    # Check each edge's gene set
    for edge_id, edge in reaction_map.edges.elements.items():
        props = reaction_map.edges[edge_id].properties
        reaction_id = props["reaction_id"]
        gene_set = props["genes"]

        # Get the original reaction
        reaction = reactions_by_id[reaction_id]

        # Check different types of gene rules
        if not reaction.gene_reaction_rule or reaction.gene_reaction_rule == "":
            # Reactions with no gene rules should have empty gene sets
            assert len(gene_set) == 0, (
                f"Edge {edge_id} for reaction without gene rule should have empty gene set"
            )
        else:
            # For edges representing a single AND combination (no ORs)
            # Each gene in the set should appear in the original rule
            for gene in gene_set:
                assert gene in reaction.gene_reaction_rule, (
                    f"Gene {gene} not found in rule: {reaction.gene_reaction_rule}"
                )

            # Make sure no OR terms appear within a single edge's genes
            # An edge should only represent one combination from the OR terms
            genes_str = " and ".join(sorted(gene_set))
            assert " or " not in genes_str, (
                f"Edge {edge_id} contains OR terms within genes: {gene_set}"
            )


def test_no_gene_reactions_included(yeast_gem):
    """Test that reactions without gene associations are properly included."""
    reaction_map = yeast_gem.reaction_map

    # Find reactions without gene rules in the model
    no_gene_reactions = [
        r.id for r in yeast_gem.model.reactions if not r.gene_reaction_rule
    ]
    assert len(no_gene_reactions) > 0, "Test needs no-gene reactions to be valid"

    # Check that all no-gene reactions appear in the map
    found_reactions = set()
    for edge_id in reaction_map.edges:
        props = reaction_map.edges[edge_id].properties
        reaction_id = props["reaction_id"]

        if reaction_id in no_gene_reactions:
            found_reactions.add(reaction_id)
            # Verify empty gene set
            assert len(props["genes"]) == 0, (
                f"Edge {edge_id} should have empty gene set"
            )

    # All no-gene reactions should be in the map
    missing_reactions = set(no_gene_reactions) - found_reactions
    assert not missing_reactions, (
        f"Reactions without genes missing from map: {missing_reactions}"
    )


def test_reaction_directions(yeast_gem):
    """Test that forward and reverse directions are properly represented."""
    reaction_map = yeast_gem.reaction_map

    # Group edges by reaction_id and direction
    reactions_directions: dict[str, set[str]] = {}
    for edge_id in reaction_map.edges:
        props = reaction_map.edges[edge_id].properties
        reaction_id = props["reaction_id"]
        direction = props["direction"]

        if reaction_id not in reactions_directions:
            reactions_directions[reaction_id] = set()
        reactions_directions[reaction_id].add(direction)

    # Check each reaction in the model
    for reaction in yeast_gem.model.reactions:
        # All reactions should have forward edges
        assert reaction.id in reactions_directions, (
            f"Reaction {reaction.id} missing from map"
        )
        assert "forward" in reactions_directions[reaction.id], (
            f"Reaction {reaction.id} missing forward direction"
        )

        # Reversible reactions should also have reverse edges
        if reaction.reversibility:
            assert "reverse" in reactions_directions[reaction.id], (
                f"Reversible reaction {reaction.id} missing reverse direction"
            )
        else:
            assert "reverse" not in reactions_directions[reaction.id], (
                f"Non-reversible reaction {reaction.id} should not have reverse direction"
            )


def test_gene_combination_consistency(yeast_gem):
    """Test that gene combinations are consistent across forward and reverse edges."""
    reaction_map = yeast_gem.reaction_map

    # Group edges by reaction_id, direction, and gene combination
    edge_groups: dict[tuple[str, frozenset[str]], set[str]] = {}
    for edge_id in reaction_map.edges:
        props = reaction_map.edges[edge_id].properties
        reaction_id = props["reaction_id"]
        direction = props["direction"]
        genes = frozenset(props["genes"])  # Make hashable

        key = (reaction_id, genes)
        if key not in edge_groups:
            edge_groups[key] = set()
        edge_groups[key].add(direction)

    # For reversible reactions, each gene combination should have both directions
    for (reaction_id, genes), directions in edge_groups.items():
        reaction = yeast_gem.model.reactions.get_by_id(reaction_id)
        if reaction.reversibility:
            assert len(directions) == 2, (
                f"Reversible reaction {reaction_id} with genes {genes} missing a direction"
            )
            assert "forward" in directions, (
                f"Reaction {reaction_id} missing forward direction"
            )
            assert "reverse" in directions, (
                f"Reaction {reaction_id} missing reverse direction"
            )


def test_or_relationships_create_multiple_edges(yeast_gem):
    """Test that OR relationships in gene rules create multiple edges."""
    reaction_map = yeast_gem.reaction_map

    # Find reactions with OR in gene rule
    or_reactions = [
        r for r in yeast_gem.model.reactions if " or " in r.gene_reaction_rule
    ]
    assert len(or_reactions) > 0, (
        "Test needs reactions with OR relationships to be valid"
    )

    for reaction in or_reactions:
        # Count edges for this reaction
        reaction_edges = [
            e
            for e in reaction_map.edges
            if reaction_map.edges[e].properties["reaction_id"] == reaction.id
            and reaction_map.edges[e].properties["direction"] == "forward"
        ]

        # Parse gene combinations
        gene_combinations = yeast_gem._parse_gene_combinations(
            reaction.gene_reaction_rule
        )

        # Should have one edge per gene combination
        assert len(reaction_edges) == len(gene_combinations), (
            f"Reaction {reaction.id} has {len(gene_combinations)} gene combinations but {len(reaction_edges)} edges"
        )


def test_bipartite_graph_structure(yeast_gem):
    """Test the structure of the unified bipartite graph representation."""
    B = yeast_gem.bipartite_graph

    # Verify it's a directed graph
    assert isinstance(B, nx.DiGraph), "Bipartite graph should be directed"

    # Check that nodes have correct types
    reaction_nodes = [n for n, d in B.nodes(data=True) if d["node_type"] == "reaction"]
    metabolite_nodes = [
        n for n, d in B.nodes(data=True) if d["node_type"] == "metabolite"
    ]

    assert len(reaction_nodes) > 0, "No reaction nodes found"
    assert len(metabolite_nodes) > 0, "No metabolite nodes found"

    # Verify edges are always from reaction to metabolite
    for u, v in B.edges():
        u_type = B.nodes[u]["node_type"]
        v_type = B.nodes[v]["node_type"]

        assert u_type == "reaction", (
            f"Source node should be reaction, got {u_type} for edge {u}->{v}"
        )
        assert v_type == "metabolite", (
            f"Target node should be metabolite, got {v_type} for edge {u}->{v}"
        )

        # Verify edge has the correct edge_type (reactant or product)
        assert B.edges[u, v]["edge_type"] in ["reactant", "product"], (
            f"Edge {u}->{v} has invalid edge_type: {B.edges[u, v]['edge_type']}"
        )


def test_bipartite_graph_gene_associations(yeast_gem):
    """Test that the bipartite graph correctly handles gene associations."""
    B = yeast_gem.bipartite_graph

    # Get reactions with and without gene rules
    reactions_with_genes = [
        r.id for r in yeast_gem.model.reactions if r.gene_reaction_rule
    ]
    reactions_without_genes = [
        r.id for r in yeast_gem.model.reactions if not r.gene_reaction_rule
    ]

    assert len(reactions_with_genes) > 0, "Need reactions with genes for testing"
    assert len(reactions_without_genes) > 0, "Need reactions without genes for testing"

    # Test sample of reactions with genes
    for reaction_id in reactions_with_genes[:5]:  # Test first 5
        reaction = yeast_gem.model.reactions.get_by_id(reaction_id)

        # Find reaction nodes for this reaction
        r_nodes = [
            n
            for n, d in B.nodes(data=True)
            if d.get("node_type") == "reaction" and d.get("reaction_id") == reaction_id
        ]

        assert len(r_nodes) > 0, f"No nodes found for reaction {reaction_id}"

        # Verify gene information stored correctly
        for r_node in r_nodes:
            genes = B.nodes[r_node]["genes"]
            assert isinstance(genes, set), f"Genes should be a set, got {type(genes)}"
            assert len(genes) > 0, (
                f"Reaction with gene rule has empty gene set: {r_node}"
            )

            # Check if each gene is in the original rule
            for gene in genes:
                assert gene in reaction.gene_reaction_rule, (
                    f"Gene {gene} not in rule: {reaction.gene_reaction_rule}"
                )

    # Test sample of reactions without genes
    for reaction_id in reactions_without_genes[:5]:  # Test first 5
        # Find reaction nodes for this reaction
        r_nodes = [
            n
            for n, d in B.nodes(data=True)
            if d.get("node_type") == "reaction" and d.get("reaction_id") == reaction_id
        ]

        assert len(r_nodes) > 0, (
            f"No nodes found for reaction {reaction_id} without genes"
        )

        # Verify empty gene set and proper node naming
        for r_node in r_nodes:
            assert "_noGene" in r_node, (
                f"Reaction node without genes should contain '_noGene': {r_node}"
            )
            genes = B.nodes[r_node]["genes"]
            assert genes == set(), (
                f"Reaction without gene rule should have empty gene set: {r_node}"
            )


def test_bipartite_graph_directionality(yeast_gem):
    """Test that the bipartite graph correctly handles reaction directionality."""
    B = yeast_gem.bipartite_graph

    # Find some reversible and irreversible reactions
    reversible = [r.id for r in yeast_gem.model.reactions if r.reversibility][:5]
    irreversible = [r.id for r in yeast_gem.model.reactions if not r.reversibility][:5]

    # Test reversible reactions
    for reaction_id in reversible:
        # Get forward and reverse nodes
        fwd_nodes = [
            n
            for n, d in B.nodes(data=True)
            if d.get("reaction_id") == reaction_id and d.get("direction") == "forward"
        ]
        rev_nodes = [
            n
            for n, d in B.nodes(data=True)
            if d.get("reaction_id") == reaction_id and d.get("direction") == "reverse"
        ]

        assert len(fwd_nodes) > 0, (
            f"No forward nodes for reversible reaction {reaction_id}"
        )
        assert len(rev_nodes) > 0, (
            f"No reverse nodes for reversible reaction {reaction_id}"
        )

        # Check that reactants and products are swapped in reverse direction
        for fwd_node in fwd_nodes:
            fwd_reactants = B.nodes[fwd_node]["reactants"]
            fwd_products = B.nodes[fwd_node]["products"]

            # Find matching reverse node with same gene combination
            for rev_node in rev_nodes:
                if B.nodes[rev_node]["genes"] == B.nodes[fwd_node]["genes"]:
                    rev_reactants = B.nodes[rev_node]["reactants"]
                    rev_products = B.nodes[rev_node]["products"]

                    # Verify reactants and products are swapped
                    assert set(fwd_reactants) == set(rev_products), (
                        "Forward reactants should equal reverse products"
                    )
                    assert set(fwd_products) == set(rev_reactants), (
                        "Forward products should equal reverse reactants"
                    )
                    break

    # Test irreversible reactions
    for reaction_id in irreversible:
        # Get forward and reverse nodes
        fwd_nodes = [
            n
            for n, d in B.nodes(data=True)
            if d.get("reaction_id") == reaction_id and d.get("direction") == "forward"
        ]
        rev_nodes = [
            n
            for n, d in B.nodes(data=True)
            if d.get("reaction_id") == reaction_id and d.get("direction") == "reverse"
        ]

        assert len(fwd_nodes) > 0, (
            f"No forward nodes for irreversible reaction {reaction_id}"
        )
        assert len(rev_nodes) == 0, (
            f"Should be no reverse nodes for irreversible reaction {reaction_id}"
        )


def test_bipartite_graph_edge_properties(yeast_gem):
    """Test that unified edge representation has correct properties."""
    B = yeast_gem.bipartite_graph

    # Sample a few reactions
    for reaction in list(yeast_gem.model.reactions)[:5]:
        # Get nodes for this reaction
        r_nodes = [
            n for n, d in B.nodes(data=True) if d.get("reaction_id") == reaction.id
        ]

        for r_node in r_nodes:
            direction = B.nodes[r_node]["direction"]

            # Verify all outgoing edges are to metabolites
            for r, m in B.out_edges(r_node):
                # Target should be a metabolite
                assert B.nodes[m]["node_type"] == "metabolite", (
                    f"Edge target should be metabolite: {r}->{m}"
                )

                # Get edge type (reactant or product)
                edge_type = B.edges[r, m]["edge_type"]
                assert edge_type in ["reactant", "product"], (
                    f"Invalid edge_type: {edge_type}"
                )

                # Verify stoichiometry and edge type based on reaction direction
                metabolite_id = m
                orig_metabolites = reaction.metabolites

                for orig_m, coef in orig_metabolites.items():
                    if orig_m.id == metabolite_id:
                        if direction == "forward":
                            if edge_type == "reactant":
                                # Forward direction, reactant edge
                                assert coef < 0, (
                                    f"Reactant should have negative coefficient in forward direction: {coef}"
                                )
                                assert B.edges[r, m]["stoichiometry"] == abs(coef), (
                                    "Stoichiometry mismatch"
                                )
                            else:  # product
                                # Forward direction, product edge
                                assert coef > 0, (
                                    f"Product should have positive coefficient in forward direction: {coef}"
                                )
                                assert B.edges[r, m]["stoichiometry"] == coef, (
                                    "Stoichiometry mismatch"
                                )
                        else:  # Reverse direction
                            if edge_type == "reactant":
                                # In reverse direction, original products become reactants
                                assert coef > 0, (
                                    f"Reactant in reverse direction should be product in forward: {coef}"
                                )
                                assert B.edges[r, m]["stoichiometry"] == coef, (
                                    "Stoichiometry mismatch"
                                )
                            else:  # product
                                # In reverse direction, original reactants become products
                                assert coef < 0, (
                                    f"Product in reverse direction should be reactant in forward: {coef}"
                                )
                                assert B.edges[r, m]["stoichiometry"] == abs(coef), (
                                    "Stoichiometry mismatch"
                                )
                        break


# -- Phase 15: a hand-built five-reaction model -----------------------------------


def _toy_model() -> cobra.Model:
    model = cobra.Model("toy5")
    a = cobra.Metabolite("A_c", name="alpha", formula="C2", compartment="c", charge=0)
    a_e = cobra.Metabolite("A_e", name="alpha", formula="C2", compartment="e", charge=0)
    b = cobra.Metabolite("B_c", name="beta", formula="C3", compartment="c", charge=-1)
    c = cobra.Metabolite("C_c", name="gamma", formula="C4", compartment="c", charge=1)
    d = cobra.Metabolite("D_e", name="delta", formula="C5", compartment="e", charge=0)
    big = cobra.Reaction("RBIG", lower_bound=0.0, upper_bound=1000.0)
    big.add_metabolites({a: -1.0, b: -2.0, c: 1.0, d: 3.0})
    ex = cobra.Reaction("EX_A_irr", lower_bound=0.0, upper_bound=1000.0)
    ex.add_metabolites({a: -1.0})
    transport = cobra.Reaction("T_A", lower_bound=-1000.0, upper_bound=1000.0)
    transport.add_metabolites({a: -1.0, a_e: 1.0})
    other = cobra.Reaction("R_OTHER", lower_bound=0.0, upper_bound=1000.0)
    other.add_metabolites({b: -1.0, c: 1.0})
    empty = cobra.Reaction("EMPTY", lower_bound=0.0, upper_bound=1000.0)
    model.add_reactions([big, ex, transport, other, empty])
    big.gene_reaction_rule = "g1 and g2 and g3"
    return model


def _write(root: Path, model: cobra.Model) -> None:
    model_dir = root / "yeast-GEM-9.0.2" / "model"
    model_dir.mkdir(parents=True)
    cobra.io.write_sbml_model(model, str(model_dir / "yeast-GEM.xml"))


@pytest.fixture
def toy_root(tmp_path: Path) -> Path:
    root = tmp_path / "gem"
    _write(root, _toy_model())
    return root


@pytest.fixture
def toy(toy_root: Path) -> YeastGEM:
    return YeastGEM(root=str(toy_root))


def _edges(gem: YeastGEM) -> dict[str, tuple[list[str], dict[str, Any]]]:
    """Hyperedge members and module-written properties (hypernetx ``_`` keys dropped)."""
    hg = gem.reaction_map
    return {
        eid: (
            list(hg.edges.elements[eid]),
            {
                k: v
                for k, v in dict(hg.edges[eid].properties).items()
                if not k.startswith("_")
            },
        )
        for eid in hg.edges
    }


def test_reaction_map_rows_for_a_complex_and_irreversible_gene_free_reactions(
    toy: YeastGEM,
) -> None:
    """Exact hyperedges: a three-subunit AND is one gene set; the row is the signed S column.

    Finding: yeast_GEM.py:138 gives ``EMPTY`` (no species) the member list ``[]`` and
    hypernetx drops a memberless edge, so the reaction is absent from ``reaction_map``
    while ``bipartite_graph`` keeps it (see the next test). Pinned until the two views
    agree on metabolite-free reactions.
    """
    assert _edges(toy) == {
        "RBIG_comb0_fwd": (
            ["A_c", "B_c", "C_c", "D_e"],
            {
                "genes": {"g1", "g2", "g3"},
                "reaction_id": "RBIG",
                "direction": "forward",
                "reactants": ["A_c", "B_c"],
                "products": ["C_c", "D_e"],
                "equation": "A_c + 2.0 B_c --> C_c + 3.0 D_e",
                "reversibility": False,
                "stoichiometry": [-1.0, -2.0, 1.0, 3.0],
                "stoich_coefficient-A_c": -1.0,
                "stoich_coefficient-B_c": -2.0,
                "stoich_coefficient-C_c": 1.0,
                "stoich_coefficient-D_e": 3.0,
                "weight": 1,
            },
        ),
        "EX_A_irr_noGene_fwd": (
            ["A_c"],
            {
                "genes": set(),
                "reaction_id": "EX_A_irr",
                "direction": "forward",
                "reactants": ["A_c"],
                "products": [],
                "equation": "A_c --> ",
                "reversibility": False,
                "stoichiometry": [-1.0],
                "stoich_coefficient-A_c": -1.0,
                "weight": 1,
            },
        ),
        "T_A_noGene_fwd": (
            ["A_c", "A_e"],
            {
                "genes": set(),
                "reaction_id": "T_A",
                "direction": "forward",
                "reactants": ["A_c"],
                "products": ["A_e"],
                "equation": "A_c <=> A_e",
                "reversibility": True,
                "stoichiometry": [-1.0, 1.0],
                "stoich_coefficient-A_c": -1.0,
                "stoich_coefficient-A_e": 1.0,
                "weight": 1,
            },
        ),
        "T_A_noGene_rev": (
            ["A_c", "A_e"],
            {
                "genes": set(),
                "reaction_id": "T_A",
                "direction": "reverse",
                "reactants": ["A_e"],
                "products": ["A_c"],
                "equation": "A_c <=> A_e",
                "reversibility": True,
                "stoichiometry": [1.0, -1.0],
                "stoich_coefficient-A_c": 1.0,
                "stoich_coefficient-A_e": -1.0,
                "weight": 1,
            },
        ),
        "R_OTHER_noGene_fwd": (
            ["B_c", "C_c"],
            {
                "genes": set(),
                "reaction_id": "R_OTHER",
                "direction": "forward",
                "reactants": ["B_c"],
                "products": ["C_c"],
                "equation": "B_c --> C_c",
                "reversibility": False,
                "stoichiometry": [-1.0, 1.0],
                "stoich_coefficient-B_c": -1.0,
                "stoich_coefficient-C_c": 1.0,
                "weight": 1,
            },
        ),
    }
    assert sorted(toy.reaction_map.nodes) == ["A_c", "A_e", "B_c", "C_c", "D_e"]


def test_bipartite_graph_edges_and_the_isolated_empty_reaction(toy: YeastGEM) -> None:
    """Exact edge list; the metabolite-free ``EMPTY`` node is kept and isolated.

    Edges carry |coef| (RBIG's B_c is 2.0, D_e 3.0); the reversible transport flips
    reactant and product on its reverse node; D_e and A_e keep compartment ``e``.
    """
    graph = toy.bipartite_graph
    assert list(graph.edges(data="stoichiometry")) == [
        ("RBIG_comb0_fwd", "A_c", 1.0),
        ("RBIG_comb0_fwd", "B_c", 2.0),
        ("RBIG_comb0_fwd", "C_c", 1.0),
        ("RBIG_comb0_fwd", "D_e", 3.0),
        ("EX_A_irr_noGene_fwd", "A_c", 1.0),
        ("T_A_noGene_fwd", "A_c", 1.0),
        ("T_A_noGene_fwd", "A_e", 1.0),
        ("T_A_noGene_rev", "A_c", 1.0),
        ("T_A_noGene_rev", "A_e", 1.0),
        ("R_OTHER_noGene_fwd", "B_c", 1.0),
        ("R_OTHER_noGene_fwd", "C_c", 1.0),
    ]
    assert [t for _, _, t in graph.edges(data="edge_type")] == [
        "reactant",
        "reactant",
        "product",
        "product",
        "reactant",
        "reactant",
        "product",
        "product",
        "reactant",
        "reactant",
        "product",
    ]
    assert list(nx.isolates(graph)) == ["EMPTY_noGene_fwd"]
    assert graph.nodes["EMPTY_noGene_fwd"] == {
        "node_type": "reaction",
        "reaction_id": "EMPTY",
        "direction": "forward",
        "genes": set(),
        "equation": " --> ",
        "reversibility": False,
        "reactants": [],
        "products": [],
        "subsystem": "",
    }
    assert {
        n: d["compartment"]
        for n, d in graph.nodes(data=True)
        if d["node_type"] == "metabolite"
    } == {"A_c": "c", "B_c": "c", "C_c": "c", "D_e": "e", "A_e": "e"}


def test_empty_induced_gene_set_keeps_only_gene_free_reactions(toy_root: Path) -> None:
    """An induced set that holds none of RBIG's genes drops RBIG from both views.

    Gene-free reactions are never filtered: an unannotated reaction is not one the
    induced genome lacks.
    """
    gem = YeastGEM(root=str(toy_root), induced_gene_set=GeneSet(["g9"]))
    assert sorted(_edges(gem)) == [
        "EX_A_irr_noGene_fwd",
        "R_OTHER_noGene_fwd",
        "T_A_noGene_fwd",
        "T_A_noGene_rev",
    ]
    assert sorted(
        n
        for n, d in gem.bipartite_graph.nodes(data=True)
        if d["node_type"] == "reaction"
    ) == [
        "EMPTY_noGene_fwd",
        "EX_A_irr_noGene_fwd",
        "R_OTHER_noGene_fwd",
        "T_A_noGene_fwd",
        "T_A_noGene_rev",
    ]


def test_gene_set_and_bipartite_are_cached_while_reaction_map_is_rebuilt(
    toy: YeastGEM,
) -> None:
    """After an in-memory GPR edit, only ``reaction_map`` reflects it.

    ``gene_set`` and ``bipartite_graph`` are memoized on first read (lines 450 and 255);
    ``reaction_map`` is recomputed on every access (line 106). So after R_OTHER gains
    ``g9`` the hypergraph has ``R_OTHER_comb0_fwd`` while the other two views still
    show the old model.
    """
    assert toy.gene_set == GeneSet(["g1", "g2", "g3"])
    assert "R_OTHER_noGene_fwd" in toy.bipartite_graph
    toy.model.reactions.get_by_id("R_OTHER").gene_reaction_rule = "g9"
    assert sorted(g.id for g in toy.model.genes) == ["g1", "g2", "g3", "g9"]
    assert toy.gene_set == GeneSet(["g1", "g2", "g3"])
    assert "R_OTHER_noGene_fwd" in toy.bipartite_graph
    assert "R_OTHER_comb0_fwd" not in toy.bipartite_graph
    assert _edges(toy)["R_OTHER_comb0_fwd"][1]["genes"] == {"g9"}
    assert "R_OTHER_noGene_fwd" not in _edges(toy)


def test_download_refuses_a_failed_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-2xx response raises from ``raise_for_status`` before any file is written."""

    def raise_404() -> None:
        raise requests.HTTPError("404 Client Error")

    def fake_get(url: str) -> SimpleNamespace:
        return SimpleNamespace(content=b"", raise_for_status=raise_404)

    monkeypatch.setattr("torchcell.metabolism.yeast_GEM.requests.get", fake_get)
    root = tmp_path / "root"
    with pytest.raises(requests.HTTPError, match="^404 Client Error$"):
        YeastGEM(root=str(root), version="0.0.1")
    assert os.listdir(root) == []


def test_analyze_reactions_without_genes_classifies_and_counts(
    toy: YeastGEM, capsys: pytest.CaptureFixture[str]
) -> None:
    """Four gene-free reactions: one exchange, one transport, two other.

    Exchange = one species (EX_A_irr); transport = two species whose ids differ only in
    the last character (A_c, A_e); the rest (R_OTHER, EMPTY) are other. 4 / 5 = 80.0%,
    1 / 4 = 25.0%, 2 / 4 = 50.0%. Species occurrences over gene-free reactions: c is
    A_c (EX_A_irr), A_c (T_A), B_c and C_c (R_OTHER) = 4, e is A_e = 1.
    """
    assert analyze_reactions_without_genes(toy) == {
        "total_reactions": 5,
        "no_gene_reactions": ["EX_A_irr", "T_A", "R_OTHER", "EMPTY"],
        "exchange_reactions": ["EX_A_irr"],
        "transport_reactions": ["T_A"],
        "other_reactions": ["R_OTHER", "EMPTY"],
        "methods_consistent": True,
    }
    out = capsys.readouterr().out
    assert out.startswith(
        "\n===== Analysis of Reactions Without Gene Associations =====\n"
        "Total reactions in model: 5\n"
        "Reactions without gene rules: 4 (80.0%)\n"
        "✓ All detection methods give consistent results\n"
        "\nReaction classification:\n"
        "  - Exchange reactions: 1 (25.0%)\n"
        "  - Transport reactions: 1 (25.0%)\n"
        "  - Other reactions: 2 (50.0%)\n"
        "\nReactions by number of metabolites:\n"
        "  - 0 metabolites: 1 reactions\n"
        "  - 1 metabolites: 1 reactions\n"
        "  - 2 metabolites: 2 reactions\n"
        "\nCompartments involved:\n"
        "  - c: 4 occurrences\n"
        "  - e: 1 occurrences\n"
    )
    assert out.endswith(
        "\nExample other reactions (showing 2 of 2):\n"
        "  1. R_OTHER: B_c --> C_c\n"
        "  2. EMPTY:  --> \n"
    )


def test_transport_rule_reads_consecutive_ids_as_one_species(tmp_path: Path) -> None:
    """Finding: yeast_GEM.py:1173 calls two species a compartment pair when their ids
    differ only in the last character. yeast-GEM ids are ``s_NNNN`` with the
    compartment elsewhere, so ``s_0001 --> s_0002`` in one compartment is classified as
    transport. Pinned until the rule compares compartments rather than id suffixes.
    """
    model = cobra.Model("ids")
    s1 = cobra.Metabolite("s_0001", compartment="c")
    s2 = cobra.Metabolite("s_0002", compartment="c")
    rxn = cobra.Reaction("r_0001", lower_bound=0.0, upper_bound=1000.0)
    rxn.add_metabolites({s1: -1.0, s2: 1.0})
    model.add_reactions([rxn])
    _write(tmp_path, model)
    result = analyze_reactions_without_genes(YeastGEM(root=str(tmp_path)))
    assert result["transport_reactions"] == ["r_0001"]
    assert result["other_reactions"] == []


def test_sanity_check_truncates_and_reports_isolated_nodes(
    toy: YeastGEM, capsys: pytest.CaptureFixture[str]
) -> None:
    """RBIG has four species, so the sample shows three and names the one left out.

    The metabolite-free EMPTY node is the one isolated node. Totals: 5 reactions,
    5 metabolites, 3 genes, 5 hyperedges on 5 nodes (EMPTY has none), 11 bipartite
    edges on 11 nodes; the partition counts are 0 (the ``bipartite`` key Finding in
    test_yeast_GEM_synthetic.py).
    """
    random.seed(0)
    sanity_check_metabolic_networks(toy, num_reactions=5)
    out = capsys.readouterr().out
    assert re.findall(r"^REACTION: (\S+)$", out, flags=re.M) == [
        "R_OTHER",
        "EMPTY",
        "RBIG",
        "EX_A_irr",
        "T_A",
    ]
    assert (
        "    3. RBIG_comb0_fwd -- C_c\n"
        "       Edge type: product\n"
        "       Stoichiometry: 1.0\n"
        "    ... and 1 more metabolites\n"
    ) in out
    assert out[out.index("===== Overall Statistics =====") :] == (
        "===== Overall Statistics =====\n"
        "Total reactions in model: 5\n"
        "Total metabolites in model: 5\n"
        "Total genes in model: 3\n"
        "Total edges in hypergraph: 5\n"
        "Total nodes in hypergraph: 5\n"
        "Total edges in bipartite graph: 11\n"
        "Total nodes in bipartite graph: 11\n"
        "Reaction nodes: 0\n"
        "Metabolite nodes: 0\n"
        "✓ All edges connect reactions to metabolites (bipartite property verified)\n"
        "WARNING: 1 isolated nodes found!\n"
        "  1. EMPTY_noGene_fwd (Type: reaction)\n"
    )


def test_main_test_bipartite_attributes_seeds_42(
    toy: YeastGEM, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The entry point seeds 42 and samples 3 of 5: RBIG, EMPTY, T_A, in that order.

    RBIG lists three of its four species then "... and 1 more"; EMPTY has no
    neighbors, so its node prints no "Connected Metabolites" block.
    """
    monkeypatch.setattr(yeast_gem_module, "YeastGEM", lambda: toy)
    yeast_gem_module.main_test_bipartite_attributes()
    out = capsys.readouterr().out
    assert out.startswith(
        "Initializing YeastGEM...\n\n===== Bipartite Graph Attribute Test =====\n"
    )
    assert re.findall(r"^REACTION: (\S+) - $", out, flags=re.M) == [
        "RBIG",
        "EMPTY",
        "T_A",
    ]
    assert (
        "      Stoichiometry: 1.0\n\n    ... and 1 more metabolites\n"
        "\n==================================================\n"
        "REACTION: EMPTY - \n"
        "==================================================\n"
        "\nReaction Node: EMPTY_noGene_fwd\n"
        "  Subsystem: \n"
        "  Direction: forward\n"
        "  Reversibility: False\n"
        "\n==================================================\n"
        "REACTION: T_A - \n"
    ) in out


def test_main_plots_the_full_network_and_seven_random_samples(  # test-quality: allow main returns None; asserts the recorded plot calls it makes
    toy: YeastGEM, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``main`` writes the full network and random samples of 5 to 4881 hyperedges.

    Every output lands under ``ASSET_IMAGES_DIR`` with the spring layout in the name.
    """
    calls: list[tuple[str, Any]] = []
    monkeypatch.setenv("ASSET_IMAGES_DIR", str(tmp_path))
    monkeypatch.setattr(yeast_gem_module, "YeastGEM", lambda: toy)
    monkeypatch.setattr(
        yeast_gem_module,
        "plot_full_network",
        lambda gem, path: calls.append(("full", path)),
    )
    monkeypatch.setattr(
        yeast_gem_module,
        "plot_random_network",
        lambda gem, **kw: calls.append(("random", kw)),
    )
    yeast_gem_module.main()
    sizes = [5, 10, 20, 50, 100, 1000, 4881]
    assert calls == [("full", str(tmp_path / "yeast_metabolic_network.png"))] + [
        (
            "random",
            {
                "n_edges": n,
                "layout": "spring",
                "output_path": str(
                    tmp_path / f"yeast_metabolic_random_nc_spring_{n}.png"
                ),
            },
        )
        for n in sizes
    ]


def test_main_bipartite_writes_the_full_bipartite_network(  # test-quality: allow main_bipartite returns None; asserts the recorded plot call
    toy: YeastGEM, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``main_bipartite`` plots the whole graph to ``full_bipartite_network.png``."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setenv("ASSET_IMAGES_DIR", str(tmp_path))
    monkeypatch.setattr(yeast_gem_module, "YeastGEM", lambda: toy)
    monkeypatch.setattr(
        yeast_gem_module, "plot_bipartite_network", lambda gem, **kw: calls.append(kw)
    )
    yeast_gem_module.main_bipartite()
    assert calls == [{"output_path": str(tmp_path / "full_bipartite_network.png")}]


def test_main_with_gene_set_passes_a_nonexistent_keyword(
    toy_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finding: yeast_GEM.py:916 calls ``YeastGEM(gene_set=...)``; the field is
    ``induced_gene_set``, so the second construction raises ``TypeError`` after the
    unfiltered count (5 hyperedges) is printed and the genome's mitochondrial
    chromosome is dropped. Pinned until the call uses ``induced_gene_set``.
    """
    genome_calls: list[str] = []

    class FakeGenome:
        def __init__(self, path: str) -> None:
            genome_calls.append(path)
            self.gene_set = GeneSet(["g1"])

        def drop_chrmt(self) -> None:
            genome_calls.append("drop_chrmt")

    def make_gem(**kwargs: Any) -> YeastGEM:
        return YeastGEM(root=str(toy_root), **kwargs)

    monkeypatch.setattr(yeast_gem_module, "YeastGEM", make_gem)
    monkeypatch.setattr(
        "torchcell.sequence.genome.scerevisiae.s288c.SCerevisiaeGenome", FakeGenome
    )
    with pytest.raises(TypeError, match="unexpected keyword argument 'gene_set'"):
        yeast_gem_module.main_with_gene_set()
    assert capsys.readouterr().out == ("H num edges without gene_set edge drop: 5\n")
    assert genome_calls == [
        osp.join(os.environ["DATA_ROOT"], "data/sgd/genome"),
        "drop_chrmt",
    ]


def _record_draw(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, dict[str, Any]]]:
    """Replace the module's ``hnx.draw`` with a recorder that draws nothing."""
    calls: list[tuple[Any, dict[str, Any]]] = []

    def recording_draw(hypergraph: Any, **kwargs: Any) -> None:
        calls.append((hypergraph, kwargs))

    monkeypatch.setattr("torchcell.metabolism.yeast_GEM.hnx.draw", recording_draw)
    return calls


def test_plot_full_network_draw_arguments(
    toy: YeastGEM,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With the draw call stubbed, the rest of the function runs to the PNG.

    One size per hypergraph node (5 x 20), the seeded spring layout settings, and a
    face color that reads the edge direction: forward light blue, reverse light green.
    Two of the five hyperedges belong to the reversible T_A.
    """
    calls = _record_draw(monkeypatch)
    out_path = tmp_path / "full.png"
    plot_full_network(toy, str(out_path))
    assert len(calls) == 1
    kwargs = calls[0][1]
    assert kwargs["nodes_kwargs"]["sizes"] == [20, 20, 20, 20, 20]
    assert kwargs["layout_kwargs"] == {
        "seed": 42,
        "k": 50,
        "iterations": 200,
        "weight": None,
        "scale": 20,
    }
    face = kwargs["edges_kwargs"]["facecolors"]
    assert face("T_A_noGene_fwd") == "lightblue"
    assert face("T_A_noGene_rev") == "lightgreen"
    assert capsys.readouterr().out == (
        "\nNetwork Statistics:\n"
        "Number of metabolites (nodes): 5\n"
        "Number of reactions (edges): 5\n"
        "Number of reversible reactions: 2\n"
        f"\nNetwork visualization saved to {out_path}\n"
    )
    assert out_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize(
    ("layout", "expected_layout", "expected_kwargs"),
    [
        ("spring", spring_layout, {"k": 10, "iterations": 1000}),
        ("kamada_kawai", kamada_kawai_layout, {}),
    ],
)
def test_plot_random_network_layouts(  # test-quality: allow the plot returns None; asserts the recorded hnx.draw call
    toy: YeastGEM,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
    expected_layout: Any,
    expected_kwargs: dict[str, Any],
) -> None:
    """Each layout name maps to its networkx function and its fixed keyword arguments."""
    calls = _record_draw(monkeypatch)
    random.seed(1)
    plot_random_network(
        toy, n_edges=2, output_path=str(tmp_path / "r.png"), layout=layout
    )
    assert calls[0][1]["layout"] == expected_layout
    assert calls[0][1]["layout_kwargs"] == expected_kwargs
    assert len(calls[0][0].edges) == 2


def test_plot_random_network_unknown_layout(
    toy: YeastGEM, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: yeast_GEM.py:669-685 has no ``else`` for an unknown layout name, so
    ``layout_kwargs`` is unbound at the draw call and the error is an
    ``UnboundLocalError`` rather than a named refusal. Pinned until the function
    raises ``ValueError`` for a name it does not know.
    """
    calls = _record_draw(monkeypatch)
    with pytest.raises(UnboundLocalError, match="layout_kwargs"):
        plot_random_network(
            toy, n_edges=1, output_path=str(tmp_path / "r.png"), layout="circular"
        )
    plt.close("all")
    assert calls == []
    assert not (tmp_path / "r.png").exists()
