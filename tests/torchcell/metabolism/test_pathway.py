# tests/torchcell/metabolism/test_pathway.py

"""Tests for adding a heterologous pathway to a genome-scale model.

Property tests, not regression tests. The properties are the claims the module makes: a
derived formula balances its reaction by construction, a spontaneous reaction cannot carry
a gene, applying a pathway does not disturb the host, and a heterologous gene stays
available when no gene token exists for it.

Most run on a small synthetic model so they need no data root. The tests that need the real
betaxanthin chemistry are marked and skipped when the yeast-GEM checkout is absent.
"""

import os
import os.path as osp

import cobra
import pytest

from torchcell.metabolism.pathway import (
    EvidenceTier,
    HeterologousPathway,
    MetaboliteSpec,
    ReactionSpec,
    apply_pathway,
)

_GEM_PATH = osp.join(
    os.environ.get("DATA_ROOT", ""),
    "data/torchcell/yeast-GEM/yeast-GEM-9.0.2/model/yeast-GEM.xml",
)
_needs_gem = pytest.mark.skipif(
    not osp.exists(_GEM_PATH), reason="yeast-GEM checkout absent"
)


def _toy_model() -> cobra.Model:
    """A two-reaction host: import A, convert A to B, export B."""
    m = cobra.Model("toy")
    a = cobra.Metabolite("a_c", formula="C3H7NO2", charge=0, compartment="c")
    b = cobra.Metabolite("b_c", formula="C3H6O3", charge=0, compartment="c")
    water = cobra.Metabolite("h2o_c", formula="H2O", charge=0, compartment="c")
    m.add_metabolites([a, b, water])
    ex = cobra.Reaction("EX_a", lower_bound=-10.0, upper_bound=1000.0)
    m.add_reactions([ex])
    ex.add_metabolites({a: -1})
    return m


def test_spontaneous_reaction_cannot_carry_a_gene() -> None:
    """The distinction is load-bearing: no deletion can remove an uncatalyzed step."""
    with pytest.raises(ValueError, match="spontaneous"):
        ReactionSpec(
            id="r_x",
            name="x",
            stoichiometry={"a_c": -1, "b_c": 1},
            spontaneous=True,
            gene_reaction_rule="SOME_GENE",
        )


def test_derived_formula_balances_its_reaction() -> None:
    """A product whose formula is None is solved for, not guessed."""
    m = _toy_model()
    pw = HeterologousPathway(
        name="toy_pathway",
        base_strain="toy",
        metabolites=[
            MetaboliteSpec(id="p_c", name="product", evidence=EvidenceTier.DERIVED)
        ],
        reactions=[
            ReactionSpec(
                id="r_condense",
                name="condensation",
                stoichiometry={"a_c": -1, "b_c": -1, "p_c": 1, "h2o_c": 1},
                spontaneous=True,
            )
        ],
    )
    out = apply_pathway(m, pw)
    product = out.metabolites.get_by_id("p_c")
    # C3H7NO2 + C3H6O3 - H2O = C6H11NO4
    assert product.formula == "C6H11NO4"
    assert out.reactions.get_by_id("r_condense").check_mass_balance() == {}


def test_unbalanced_reaction_is_rejected() -> None:
    """A heterologous reaction that unbalances the network must fail loudly.

    Silently unbalanced stoichiometry turns every downstream mass-balance residual into a
    constant offset that cannot be attributed to anything.
    """
    m = _toy_model()
    pw = HeterologousPathway(
        name="bad",
        base_strain="toy",
        reactions=[
            ReactionSpec(
                id="r_bad",
                name="bad",
                stoichiometry={"a_c": -1, "b_c": 1},
                spontaneous=True,
            )
        ],
    )
    with pytest.raises(ValueError, match="unbalanced"):
        apply_pathway(m, pw)


def test_apply_pathway_does_not_mutate_the_base_model() -> None:
    """The base model is shared across arms, so a mutated copy is an invisible difference."""
    m = _toy_model()
    before = len(m.reactions)
    pw = HeterologousPathway(
        name="toy",
        base_strain="toy",
        metabolites=[MetaboliteSpec(id="p_c", name="p", formula="C3H6O3", charge=0)],
        reactions=[ReactionSpec(id="DM_p", name="demand", stoichiometry={"p_c": -1})],
    )
    out = apply_pathway(m, pw)
    assert len(m.reactions) == before
    assert len(out.reactions) == before + 1


def test_duplicate_ids_are_rejected() -> None:
    m = _toy_model()
    pw = HeterologousPathway(
        name="clash",
        base_strain="toy",
        reactions=[ReactionSpec(id="EX_a", name="clash", stoichiometry={"a_c": -1})],
    )
    with pytest.raises(ValueError, match="already in model"):
        apply_pathway(m, pw)


@_needs_gem
def test_betaxanthin_pathway_balances_and_leaves_growth_untouched() -> None:
    """Adding the cassette must not change what the host can do on its own."""
    from torchcell.metabolism.betaxanthin import build_betaxanthin_pathway

    base = cobra.io.read_sbml_model(_GEM_PATH)
    growth_before = base.slim_optimize()
    out = apply_pathway(base, build_betaxanthin_pathway(base))
    assert out.slim_optimize() == pytest.approx(growth_before, abs=1e-6)


@_needs_gem
def test_derived_betaxanthin_masses_match_the_source() -> None:
    """The condensation stoichiometry is checked against DeLoache's reported masses.

    Tyrosine-betaxanthin and betanidin are the two the paper quantifies by mass, so a wrong
    Schiff-base stoichiometry would show up here rather than as a silent modeling error.
    """
    from torchcell.metabolism.betaxanthin import build_betaxanthin_pathway

    base = cobra.io.read_sbml_model(_GEM_PATH)
    out = apply_pathway(base, build_betaxanthin_pathway(base))
    assert out.metabolites.get_by_id("s_btx_tyrosine_c").formula == "C18H18N2O7"
    assert out.metabolites.get_by_id("s_betanidin_c").formula == "C18H16N2O8"


@_needs_gem
def test_every_condensation_partner_gets_a_reaction_and_a_demand() -> None:
    """Betaxanthin is a family, so the readout is a sum over partners, not one flux."""
    from torchcell.metabolism.betaxanthin import (
        betaxanthin_demand_ids,
        build_betaxanthin_pathway,
    )

    base = cobra.io.read_sbml_model(_GEM_PATH)
    pw = build_betaxanthin_pathway(base)
    demands = betaxanthin_demand_ids(pw)
    condensations = [r for r in pw.reactions if r.id.startswith("r_btx_")]
    assert len(demands) == len(condensations) > 1
    assert all(r.spontaneous for r in condensations)


@_needs_gem
def test_cassette_genes_are_constitutive_in_the_flux_layer() -> None:
    """A heterologous gene has no gene token, and zero availability would switch it off.

    Without the declaration the availability chain reads absence from the token universe as
    availability zero, which silently disables the pathway that was just added.
    """
    import torch

    from torchcell.metabolism.betaxanthin import build_betaxanthin_pathway
    from torchcell.metabolism.constraints import build_gem_tensors
    from torchcell.metabolism.flux_layer import FluxLayer, FluxLayerConfig

    base = cobra.io.read_sbml_model(_GEM_PATH)
    pw = build_betaxanthin_pathway(base)
    out = apply_pathway(base, pw)
    gem = build_gem_tensors(out)
    host_gene_ids = [g.id for g in base.genes]

    empty = torch.zeros(0, dtype=torch.long)
    layer = FluxLayer(
        gem, host_gene_ids, FluxLayerConfig(), constitutive_genes=pw.constitutive_genes
    )
    h = torch.randn(1, len(host_gene_ids), layer.availability.in_features)
    gamma = layer.gene_availability(h, empty, empty)
    c_j = layer.reaction_availability(gamma)

    gene_ids = gem.catalytic_units.gene_ids
    for gene in pw.constitutive_genes:
        assert gamma[0, gene_ids.index(gene)].item() == pytest.approx(1.0)
    for rxn in ("r_CYP76AD1_th", "r_DOD"):
        assert c_j[0, gem.rxn_ids.index(rxn)].item() > 0.0

    undeclared = FluxLayer(gem, host_gene_ids, FluxLayerConfig())
    undeclared.availability = layer.availability
    gamma_off = undeclared.gene_availability(h, empty, empty)
    assert gamma_off[0, gene_ids.index("DOD")].item() == pytest.approx(0.0)
