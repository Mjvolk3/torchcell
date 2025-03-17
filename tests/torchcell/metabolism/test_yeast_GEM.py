# Test file
import os
import os.path as osp

import hypernetx as hnx
import networkx as nx
import pytest

from torchcell.metabolism.yeast_GEM import YeastGEM


@pytest.fixture
def yeast_gem():
    """Create a YeastGEM instance for testing."""
    return YeastGEM()


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
            assert (
                len(gene_set) == 0
            ), f"Edge {edge_id} for reaction without gene rule should have empty gene set"
        else:
            # For edges representing a single AND combination (no ORs)
            # Each gene in the set should appear in the original rule
            for gene in gene_set:
                assert (
                    gene in reaction.gene_reaction_rule
                ), f"Gene {gene} not found in rule: {reaction.gene_reaction_rule}"

            # Make sure no OR terms appear within a single edge's genes
            # An edge should only represent one combination from the OR terms
            genes_str = " and ".join(sorted(gene_set))
            assert (
                " or " not in genes_str
            ), f"Edge {edge_id} contains OR terms within genes: {gene_set}"


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
            assert (
                len(props["genes"]) == 0
            ), f"Edge {edge_id} should have empty gene set"

    # All no-gene reactions should be in the map
    missing_reactions = set(no_gene_reactions) - found_reactions
    assert (
        not missing_reactions
    ), f"Reactions without genes missing from map: {missing_reactions}"


def test_reaction_directions(yeast_gem):
    """Test that forward and reverse directions are properly represented."""
    reaction_map = yeast_gem.reaction_map

    # Group edges by reaction_id and direction
    reactions_directions = {}
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
        assert (
            reaction.id in reactions_directions
        ), f"Reaction {reaction.id} missing from map"
        assert (
            "forward" in reactions_directions[reaction.id]
        ), f"Reaction {reaction.id} missing forward direction"

        # Reversible reactions should also have reverse edges
        if reaction.reversibility:
            assert (
                "reverse" in reactions_directions[reaction.id]
            ), f"Reversible reaction {reaction.id} missing reverse direction"
        else:
            assert (
                "reverse" not in reactions_directions[reaction.id]
            ), f"Non-reversible reaction {reaction.id} should not have reverse direction"


def test_gene_combination_consistency(yeast_gem):
    """Test that gene combinations are consistent across forward and reverse edges."""
    reaction_map = yeast_gem.reaction_map

    # Group edges by reaction_id, direction, and gene combination
    edge_groups = {}
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
            assert (
                len(directions) == 2
            ), f"Reversible reaction {reaction_id} with genes {genes} missing a direction"
            assert (
                "forward" in directions
            ), f"Reaction {reaction_id} missing forward direction"
            assert (
                "reverse" in directions
            ), f"Reaction {reaction_id} missing reverse direction"


def test_or_relationships_create_multiple_edges(yeast_gem):
    """Test that OR relationships in gene rules create multiple edges."""
    reaction_map = yeast_gem.reaction_map

    # Find reactions with OR in gene rule
    or_reactions = [
        r for r in yeast_gem.model.reactions if " or " in r.gene_reaction_rule
    ]
    assert (
        len(or_reactions) > 0
    ), "Test needs reactions with OR relationships to be valid"

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
        assert len(reaction_edges) == len(
            gene_combinations
        ), f"Reaction {reaction.id} has {len(gene_combinations)} gene combinations but {len(reaction_edges)} edges"


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

        assert (
            u_type == "reaction"
        ), f"Source node should be reaction, got {u_type} for edge {u}->{v}"
        assert (
            v_type == "metabolite"
        ), f"Target node should be metabolite, got {v_type} for edge {u}->{v}"

        # Verify edge has the correct edge_type (reactant or product)
        assert B.edges[u, v]["edge_type"] in [
            "reactant",
            "product",
        ], f"Edge {u}->{v} has invalid edge_type: {B.edges[u, v]['edge_type']}"


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
            assert (
                len(genes) > 0
            ), f"Reaction with gene rule has empty gene set: {r_node}"

            # Check if each gene is in the original rule
            for gene in genes:
                assert (
                    gene in reaction.gene_reaction_rule
                ), f"Gene {gene} not in rule: {reaction.gene_reaction_rule}"

    # Test sample of reactions without genes
    for reaction_id in reactions_without_genes[:5]:  # Test first 5
        # Find reaction nodes for this reaction
        r_nodes = [
            n
            for n, d in B.nodes(data=True)
            if d.get("node_type") == "reaction" and d.get("reaction_id") == reaction_id
        ]

        assert (
            len(r_nodes) > 0
        ), f"No nodes found for reaction {reaction_id} without genes"

        # Verify empty gene set and proper node naming
        for r_node in r_nodes:
            assert (
                "_noGene" in r_node
            ), f"Reaction node without genes should contain '_noGene': {r_node}"
            genes = B.nodes[r_node]["genes"]
            assert (
                genes == set()
            ), f"Reaction without gene rule should have empty gene set: {r_node}"


def test_bipartite_graph_directionality(yeast_gem):
    """Test that the bipartite graph correctly handles reaction directionality."""
    B = yeast_gem.bipartite_graph

    # Find some reversible and irreversible reactions
    reversible = [r.id for r in yeast_gem.model.reactions if r.reversibility][:5]
    irreversible = [r.id for r in yeast_gem.model.reactions if not r.reversibility][:5]

    # Test reversible reactions
    for reaction_id in reversible:
        reaction = yeast_gem.model.reactions.get_by_id(reaction_id)

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

        assert (
            len(fwd_nodes) > 0
        ), f"No forward nodes for reversible reaction {reaction_id}"
        assert (
            len(rev_nodes) > 0
        ), f"No reverse nodes for reversible reaction {reaction_id}"

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
                    assert set(fwd_reactants) == set(
                        rev_products
                    ), "Forward reactants should equal reverse products"
                    assert set(fwd_products) == set(
                        rev_reactants
                    ), "Forward products should equal reverse reactants"
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

        assert (
            len(fwd_nodes) > 0
        ), f"No forward nodes for irreversible reaction {reaction_id}"
        assert (
            len(rev_nodes) == 0
        ), f"Should be no reverse nodes for irreversible reaction {reaction_id}"


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
                assert (
                    B.nodes[m]["node_type"] == "metabolite"
                ), f"Edge target should be metabolite: {r}->{m}"

                # Get edge type (reactant or product)
                edge_type = B.edges[r, m]["edge_type"]
                assert edge_type in [
                    "reactant",
                    "product",
                ], f"Invalid edge_type: {edge_type}"

                # Verify stoichiometry and edge type based on reaction direction
                metabolite_id = m
                orig_metabolites = reaction.metabolites

                for orig_m, coef in orig_metabolites.items():
                    if orig_m.id == metabolite_id:
                        if direction == "forward":
                            if edge_type == "reactant":
                                # Forward direction, reactant edge
                                assert (
                                    coef < 0
                                ), f"Reactant should have negative coefficient in forward direction: {coef}"
                                assert B.edges[r, m]["stoichiometry"] == abs(
                                    coef
                                ), "Stoichiometry mismatch"
                            else:  # product
                                # Forward direction, product edge
                                assert (
                                    coef > 0
                                ), f"Product should have positive coefficient in forward direction: {coef}"
                                assert (
                                    B.edges[r, m]["stoichiometry"] == coef
                                ), "Stoichiometry mismatch"
                        else:  # Reverse direction
                            if edge_type == "reactant":
                                # In reverse direction, original products become reactants
                                assert (
                                    coef > 0
                                ), f"Reactant in reverse direction should be product in forward: {coef}"
                                assert (
                                    B.edges[r, m]["stoichiometry"] == coef
                                ), "Stoichiometry mismatch"
                            else:  # product
                                # In reverse direction, original reactants become products
                                assert (
                                    coef < 0
                                ), f"Product in reverse direction should be reactant in forward: {coef}"
                                assert B.edges[r, m]["stoichiometry"] == abs(
                                    coef
                                ), "Stoichiometry mismatch"
                        break
