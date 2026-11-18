# torchcell/metabolism/bipartite
# [[torchcell.metabolism.bipartite]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/metabolism/bipartite
# Test file: tests/torchcell/metabolism/test_bipartite.py
"""The reaction-metabolite bipartite graph of any cobra model.

``YeastGEM.bipartite_graph`` builds the structure ``to_cell_data`` turns into the
``("reaction", "rmr", "metabolite")`` and ``("gene", "gpr", "reaction")`` stores,
and nothing in that construction is specific to yeast: it reads
``reaction.gene_reaction_rule``, ``reaction.metabolites`` and the metabolite
annotations, all of which any cobra model carries. This module is that builder
with the organism removed, so a second genome-scale model does not need a second
copy of it.

The GPR is encoded structurally rather than as a string. A rule
``(a and b) or c`` is a disjunction of conjunctions, so it becomes one reaction
node per OR term, and a reversible reaction becomes two nodes per term:

    {reaction_id}_comb{idx}_fwd / _rev     with a GPR
    {reaction_id}_noGene_fwd / _rev        without one

Each node carries ``genes`` as the AND-set, which is what makes a complex (min
over subunits) distinguishable from an isozyme (capacities add) downstream. Edges
run reaction to metabolite with ``edge_type`` of ``reactant`` or ``product`` and
an absolute ``stoichiometry``; the sign is restored by ``to_cell_data``, which
negates reactants. A reverse node swaps the two roles.

``growth_reaction_ids`` exists because ``to_cell_data`` marks the growth
reactions by the literal test ``subsystem == "Growth"``, which is a yeast-GEM
convention rather than a property of SBML. A BiGG model calls its biomass
subsystem something else, so a caller names the growth reactions and this builder
labels them, keeping the model's own string under ``subsystem_source`` so nothing
is lost.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:  # pragma: no cover
    import cobra

#: The subsystem string ``torchcell.data.cell_data`` tests for when it builds the
#: ``reaction.w_growth`` tensor.
GROWTH_SUBSYSTEM: str = "Growth"


class MemberlessReactionError(ValueError):
    """A reaction with no metabolites, which the graph views cannot represent.

    Raised rather than skipped: hypernetx drops a hyperedge with no members, so
    such a reaction would vanish from a hypergraph view while a bipartite view
    kept it as an isolated node, and the two views would disagree on the
    reaction set.
    """


def parse_gene_combinations(rule: str) -> list[set[str]]:
    """Split a GPR rule into its OR terms, each an AND-set of gene ids.

    An empty or absent rule returns ``[set()]``, which is the signal for a
    reaction with no gene association rather than an empty list.
    """
    if not rule or rule == "":
        return [set()]
    terms = re.split(" or ", rule.replace("(", "").replace(")", ""))
    combinations = []
    for term in terms:
        genes = {gene.strip() for gene in term.split(" and ")}
        genes.discard("")
        combinations.append(genes)
    return combinations


def refuse_memberless_reactions(model: cobra.Model) -> None:
    """Raise :class:`MemberlessReactionError` naming every reaction with no species."""
    memberless = [r.id for r in model.reactions if not r.metabolites]
    if memberless:
        raise MemberlessReactionError(
            f"Reactions with no metabolites cannot be represented in a bipartite "
            f"graph: {memberless}"
        )


def _add_metabolite_node(graph: nx.DiGraph, metabolite: cobra.Metabolite) -> None:
    """Add a metabolite node with the attributes ``to_cell_data`` reads."""
    if graph.has_node(metabolite.id):
        return
    graph.add_node(
        metabolite.id,
        node_type="metabolite",
        name=metabolite.name,
        composition=metabolite.formula,
        compartment=metabolite.compartment,
        charge=metabolite.charge,
        miriam=metabolite.annotation.get("miriam", ""),
        inchi=metabolite.annotation.get("inchi", ""),
        replacement_id=metabolite.annotation.get("replacement_id", ""),
    )


def _add_metabolite_edges(
    graph: nx.DiGraph, reaction: cobra.Reaction, reaction_node_id: str, direction: str
) -> None:
    """Connect one reaction node to its metabolites, with roles set by direction."""
    for metabolite in reaction.metabolites:
        _add_metabolite_node(graph, metabolite)

    for metabolite, coefficient in reaction.metabolites.items():
        consumed = coefficient < 0 if direction == "forward" else coefficient > 0
        graph.add_edge(
            reaction_node_id,
            metabolite.id,
            edge_type="reactant" if consumed else "product",
            direction=direction,
            stoichiometry=abs(coefficient),
            reaction_id=reaction.id,
        )


def cobra_bipartite_graph(
    model: cobra.Model,
    induced_gene_set: set[str] | None = None,
    growth_reaction_ids: tuple[str, ...] = (),
) -> nx.DiGraph:
    """Build the reaction-metabolite bipartite graph of a cobra model.

    Args:
        model: Any cobra model.
        induced_gene_set: When given, an OR term whose genes lie entirely outside
            this set is dropped, which is how a model is restricted to the genes
            an organism's annotation actually carries. A term with at least one
            gene inside the set is kept whole.
        growth_reaction_ids: Reaction ids whose nodes are labeled with the
            ``Growth`` subsystem so the growth tensor downstream is populated.
            The model's own subsystem is preserved as ``subsystem_source``.

    Returns:
        A directed bipartite graph whose nodes carry ``node_type`` of
        ``reaction`` or ``metabolite``.

    Raises:
        MemberlessReactionError: If any reaction has no metabolites.
    """
    refuse_memberless_reactions(model)
    growth = set(growth_reaction_ids)
    graph = nx.DiGraph()

    for reaction in model.reactions:
        gene_combinations = parse_gene_combinations(reaction.gene_reaction_rule)

        reactants = []
        products = []
        for metabolite, coefficient in reaction.metabolites.items():
            if coefficient < 0:
                reactants.append(metabolite.id)
            else:
                products.append(metabolite.id)

        subsystem_source = reaction.subsystem
        subsystem = GROWTH_SUBSYSTEM if reaction.id in growth else reaction.subsystem

        if gene_combinations and gene_combinations != [set()]:
            node_stems = []
            for idx, genes in enumerate(gene_combinations):
                if induced_gene_set is not None and genes - induced_gene_set == genes:
                    continue
                node_stems.append((f"{reaction.id}_comb{idx}", genes))
        else:
            node_stems = [(f"{reaction.id}_noGene", set())]

        for stem, genes in node_stems:
            graph.add_node(
                f"{stem}_fwd",
                node_type="reaction",
                reaction_id=reaction.id,
                direction="forward",
                genes=genes,
                equation=reaction.reaction,
                reversibility=reaction.reversibility,
                reactants=reactants,
                products=products,
                subsystem=subsystem,
                subsystem_source=subsystem_source,
            )
            _add_metabolite_edges(graph, reaction, f"{stem}_fwd", "forward")

            if reaction.reversibility:
                graph.add_node(
                    f"{stem}_rev",
                    node_type="reaction",
                    reaction_id=reaction.id,
                    direction="reverse",
                    genes=genes,
                    equation=reaction.reaction,
                    reversibility=reaction.reversibility,
                    reactants=products,
                    products=reactants,
                    subsystem=subsystem,
                    subsystem_source=subsystem_source,
                )
                _add_metabolite_edges(graph, reaction, f"{stem}_rev", "reverse")

    return graph
