# tests/torchcell/metabolism/test_thermo_pipeline.py

"""Tests for the organism-agnostic thermodynamic pipeline.

The whole point of the module is that nothing in it knows about yeast, so the tests are
built on a synthetic five-metabolite GEM whose every number can be worked out on paper.
That model carries, deliberately, one of each thing the pipeline has to distinguish: a
metabolite with a structure, one with only an accession, one whose formula is a
placeholder, one that is a modeling device, and one a plugged-in structure writer can
compose from its name.

The second level runs against yeast-GEM 9.0.2 when it is on disk, and checks only the
things that must hold for ANY model: identities parse, the compartment table is honored,
and a reaction is priced if and only if every participant is.
"""

from __future__ import annotations

import os
import os.path as osp

import cobra
import pytest

from torchcell.metabolism.thermo_pipeline import (
    KJ_PER_KCAL,
    CompartmentConditions,
    MetaboliteIdentity,
    MetabolitePrice,
    MetaboliteResolver,
    TableResolver,
    ThermoPipelineConfig,
    ThermoRoute,
    UnpricedReason,
    identities_from_model,
    resolve_metabolites,
    run_thermo_pipeline,
    summarize,
)


def _synthetic_model() -> cobra.Model:
    """A five-metabolite, three-reaction GEM covering every resolution outcome.

    ``a_c``   carries a SMILES, so a structure route can reach it.
    ``b_c``   carries only a KEGG id, so only an accession route can.
    ``r_c``   has a placeholder formula, so nothing can.
    ``pool_c`` is a biomass pool: no formula at all.
    ``lip_c`` has a definite formula, no structure and no accession, and a name a writer
    can compose from. It is the tier-b case in miniature.
    """
    model = cobra.Model("synthetic")
    a = cobra.Metabolite("a_c", formula="C6H12O6", name="alpha", compartment="c")
    a.annotation = {"smiles": "OCC1OC(O)C(O)C(O)C1O"}
    b = cobra.Metabolite("b_c", formula="C3H4O3", name="beta", compartment="c")
    b.annotation = {"kegg.compound": ["C00022"]}
    r = cobra.Metabolite("r_c", formula="C10H18RO", name="acyl-R", compartment="c")
    pool = cobra.Metabolite("pool_c", formula="", name="biomass pool", compartment="c")
    lip = cobra.Metabolite(
        "lip_c", formula="C10H20O2", name="written lipid (1-10:0)", compartment="v"
    )

    r1 = cobra.Reaction("R1")
    r1.add_metabolites({a: -1.0, b: 2.0})
    r2 = cobra.Reaction("R2")
    r2.add_metabolites({b: -1.0, r: 1.0})
    r3 = cobra.Reaction("R3")
    r3.add_metabolites({a: -1.0, pool: 1.0, lip: 1.0})
    model.add_reactions([r1, r2, r3])
    return model


class _FixedResolver:
    """A resolver that prices exactly the ids it was given and refuses the rest."""

    def __init__(
        self,
        name: str,
        values: dict[str, float],
        route: ThermoRoute,
        reason: UnpricedReason = UnpricedReason.NOT_ATTEMPTED,
    ) -> None:
        self.name = name
        self.values = values
        self.route = route
        self.reason = reason
        self.seen: list[str] = []

    def resolve(
        self, metabolites: list[MetaboliteIdentity], conditions: CompartmentConditions
    ) -> list[MetabolitePrice]:
        self.seen = [m.met_id for m in metabolites]
        out = []
        for met in metabolites:
            if met.met_id in self.values:
                p_h, ionic, p_mg = conditions.for_compartment(met.compartment)
                out.append(
                    MetabolitePrice(
                        met_id=met.met_id,
                        dgf_prime_kj_per_mol=self.values[met.met_id],
                        route=self.route,
                        p_h=p_h,
                        ionic_strength_m=ionic,
                        p_mg=p_mg,
                    )
                )
            else:
                out.append(MetabolitePrice(met_id=met.met_id, reason=self.reason))
        return out


def test_identities_read_structures_and_accessions() -> None:
    identities = {m.met_id: m for m in identities_from_model(_synthetic_model())}
    assert identities["a_c"].smiles == "OCC1OC(O)C(O)C(O)C1O"
    assert identities["a_c"].has_structure
    # cobra stores a single annotation value as a bare string and several as a list; both
    # have to land as a list, or the second shape is dropped silently.
    assert identities["b_c"].accessions["kegg.compound"] == ["C00022"]
    assert not identities["b_c"].has_structure
    assert identities["r_c"].has_placeholder_formula
    assert not identities["a_c"].has_placeholder_formula


def test_stage_order_is_the_precedence_and_later_stages_never_overwrite() -> None:
    identities = identities_from_model(_synthetic_model())
    first = _FixedResolver("first", {"a_c": -900.0}, ThermoRoute.CACHE_STRUCTURE)
    second = _FixedResolver(
        "second", {"a_c": -1.0, "b_c": -470.0}, ThermoRoute.EXTERNAL_PKA
    )
    prices = {
        p.met_id: p
        for p in resolve_metabolites(
            identities, [first, second], CompartmentConditions()
        )
    }
    assert prices["a_c"].dgf_prime_kj_per_mol == -900.0
    assert prices["a_c"].route is ThermoRoute.CACHE_STRUCTURE
    assert prices["b_c"].route is ThermoRoute.EXTERNAL_PKA
    # a_c is not even offered to the second stage
    assert "a_c" not in second.seen


def test_unpriced_reasons_are_specific_rather_than_blank() -> None:
    model = _synthetic_model()
    identities = identities_from_model(model)
    prices = {
        p.met_id: p
        for p in resolve_metabolites(
            identities,
            [_FixedResolver("only_a", {"a_c": -900.0}, ThermoRoute.CACHE_STRUCTURE)],
            CompartmentConditions(),
            pseudo_metabolite_ids=["pool_c"],
        )
    }
    assert prices["r_c"].reason is UnpricedReason.PLACEHOLDER_FORMULA
    assert prices["pool_c"].reason is UnpricedReason.PSEUDO_METABOLITE
    assert prices["b_c"].reason is UnpricedReason.NO_STRUCTURE_AVAILABLE
    assert prices["lip_c"].reason is UnpricedReason.NO_STRUCTURE_AVAILABLE


def test_a_reaction_is_priced_only_when_every_participant_is() -> None:
    model = _synthetic_model()
    config = ThermoPipelineConfig(organism="synthetic")
    stage = _FixedResolver(
        "known", {"a_c": -900.0, "b_c": -470.0}, ThermoRoute.CACHE_STRUCTURE
    )
    result = run_thermo_pipeline(model, config, [stage])
    reactions = {r.reaction_id: r for r in result.reactions}

    # R1: a -> 2b, both priced. 2*(-470) - (-900) = -40.
    assert reactions["R1"].delta_r_g0_kj_per_mol == pytest.approx(-40.0)
    assert reactions["R1"].route is ThermoRoute.CACHE_STRUCTURE
    # R2 touches the placeholder, R3 the pool and the lipid.
    assert reactions["R2"].delta_r_g0_kj_per_mol is None
    assert reactions["R2"].reason is UnpricedReason.BLOCKED_BY_METABOLITE
    assert reactions["R2"].blocking_met_ids == ["r_c"]
    assert UnpricedReason.PLACEHOLDER_FORMULA in reactions["R2"].blocking_reasons
    assert sorted(reactions["R3"].blocking_met_ids) == ["lip_c", "pool_c"]
    assert result.reaction_counts.n_priced == 1
    assert result.reaction_counts.n_total == 3


def test_two_routes_in_one_reaction_are_tagged_mixed() -> None:
    model = _synthetic_model()
    stages: list[MetaboliteResolver] = [
        _FixedResolver("cache", {"a_c": -900.0}, ThermoRoute.CACHE_STRUCTURE),
        _FixedResolver("external", {"b_c": -470.0}, ThermoRoute.EXTERNAL_PKA),
    ]
    result = run_thermo_pipeline(model, ThermoPipelineConfig(), stages)
    reactions = {r.reaction_id: r for r in result.reactions}
    assert reactions["R1"].route is ThermoRoute.MIXED


def test_reaction_level_fill_is_its_own_route_and_records_what_it_papered_over() -> (
    None
):
    model = _synthetic_model()
    config = ThermoPipelineConfig(reaction_table={"R2": 12.0})
    stage = _FixedResolver(
        "known", {"a_c": -900.0, "b_c": -470.0}, ThermoRoute.CACHE_STRUCTURE
    )
    result = run_thermo_pipeline(model, config, [stage])
    reactions = {r.reaction_id: r for r in result.reactions}
    assert reactions["R2"].delta_r_g0_kj_per_mol == pytest.approx(12.0)
    assert reactions["R2"].route is ThermoRoute.SHIPPED_REACTION
    # The fill does not erase the fact that a participant is unpriced.
    assert reactions["R2"].blocking_met_ids == ["r_c"]


def test_a_mixed_unit_reaction_table_converts_only_the_rows_that_need_it() -> None:
    model = _synthetic_model()
    config = ThermoPipelineConfig(
        reaction_table={"R2": 10.0, "R3": 10.0},
        reaction_table_units="kcal/mol",
        reaction_table_already_kj=["R3"],
    )
    result = run_thermo_pipeline(model, config, [])
    reactions = {r.reaction_id: r for r in result.reactions}
    assert reactions["R2"].delta_r_g0_kj_per_mol == pytest.approx(10.0 * KJ_PER_KCAL)
    assert reactions["R3"].delta_r_g0_kj_per_mol == pytest.approx(10.0)


def test_table_resolver_converts_kcal_on_read() -> None:
    identities = identities_from_model(_synthetic_model())
    resolver = TableResolver(values={"a_c": -100.0}, units="kcal/mol")
    prices = {
        p.met_id: p for p in resolver.resolve(identities, CompartmentConditions())
    }
    assert prices["a_c"].dgf_prime_kj_per_mol == pytest.approx(-100.0 * KJ_PER_KCAL)
    assert prices["a_c"].route is ThermoRoute.SHIPPED_TABLE


def test_a_structure_writer_is_a_stage_and_closes_the_tier_b_case() -> None:
    """The pluggable half: composing a structure from a name is not special-cased."""

    class NameWriterResolver:
        """A minimal writer stage: recognize a name pattern, price what it composes."""

        name = "name_structure_writer"

        def resolve(
            self,
            metabolites: list[MetaboliteIdentity],
            conditions: CompartmentConditions,
        ) -> list[MetabolitePrice]:
            out = []
            for met in metabolites:
                if "(1-10:0)" in met.name and met.formula == "C10H20O2":
                    out.append(
                        MetabolitePrice(
                            met_id=met.met_id,
                            dgf_prime_kj_per_mol=-350.0,
                            route=ThermoRoute.STRUCTURE_WRITER,
                            provenance={"smiles": "CCCCCCCCCC(=O)O"},
                        )
                    )
                else:
                    out.append(
                        MetabolitePrice(
                            met_id=met.met_id,
                            reason=UnpricedReason.NO_STRUCTURE_AVAILABLE,
                        )
                    )
            return out

    model = _synthetic_model()
    stages: list[MetaboliteResolver] = [
        _FixedResolver("cache", {"a_c": -900.0}, ThermoRoute.CACHE_STRUCTURE),
        NameWriterResolver(),
    ]
    result = run_thermo_pipeline(
        model, ThermoPipelineConfig(pseudo_metabolite_ids=["pool_c"]), stages
    )
    prices = {p.met_id: p for p in result.metabolites}
    assert prices["lip_c"].route is ThermoRoute.STRUCTURE_WRITER
    assert prices["lip_c"].dgf_prime_kj_per_mol == pytest.approx(-350.0)
    # The pseudo-metabolite still blocks R3, and says why.
    reactions = {r.reaction_id: r for r in result.reactions}
    assert reactions["R3"].blocking_met_ids == ["pool_c"]
    assert reactions["R3"].blocking_reasons == [UnpricedReason.PSEUDO_METABOLITE]


def test_compartment_conditions_report_what_they_did_not_name() -> None:
    conditions = CompartmentConditions(p_h={"c": 7.2}, source="test")
    model = _synthetic_model()
    result = run_thermo_pipeline(model, ThermoPipelineConfig(conditions=conditions), [])
    # 'v' is never named, so it is priced at the default AND said so.
    assert result.compartments_at_default == ["v"]
    assert conditions.for_compartment("c")[0] == 7.2
    assert conditions.for_compartment("v")[0] == conditions.default_p_h


def test_sigma_propagates_in_quadrature_with_the_stoichiometry() -> None:
    model = _synthetic_model()
    stage = _FixedResolver(
        "known", {"a_c": -900.0, "b_c": -470.0}, ThermoRoute.CACHE_STRUCTURE
    )
    result = run_thermo_pipeline(
        model, ThermoPipelineConfig(), [stage], formation_sigma={"a_c": 3.0, "b_c": 4.0}
    )
    reactions = {r.reaction_id: r for r in result.reactions}
    # sqrt((-1*3)^2 + (2*4)^2) = sqrt(9 + 64)
    assert reactions["R1"].sigma_kj_per_mol == pytest.approx((9.0 + 64.0) ** 0.5)


def test_flux_carrying_fraction_is_over_the_reactions_the_caller_names() -> None:
    model = _synthetic_model()
    stage = _FixedResolver(
        "known", {"a_c": -900.0, "b_c": -470.0}, ThermoRoute.CACHE_STRUCTURE
    )
    result = run_thermo_pipeline(
        model, ThermoPipelineConfig(flux_carrying_reaction_ids=["R1", "R2"]), [stage]
    )
    assert result.flux_carrying["n_flux_carrying"] == 2.0
    assert result.flux_carrying["n_priced"] == 1.0
    assert result.flux_carrying["fraction"] == pytest.approx(0.5)


def test_a_stage_that_answers_for_a_metabolite_it_was_not_handed_is_an_error() -> None:
    class Rogue:
        name = "rogue"

        def resolve(self, metabolites, conditions):
            return [MetabolitePrice(met_id="not_in_the_model", reason=None)]

    with pytest.raises(ValueError, match="was not handed"):
        resolve_metabolites(
            identities_from_model(_synthetic_model()),
            [Rogue()],
            CompartmentConditions(),
        )


def test_blocking_metabolites_rank_the_work_queue() -> None:
    model = _synthetic_model()
    result = run_thermo_pipeline(
        model, ThermoPipelineConfig(pseudo_metabolite_ids=["pool_c"]), []
    )
    ranked = {
        row["met_id"]: row["n_reactions_blocked"] for row in result.blocking_metabolites
    }
    # a_c is in R1 and R3, so with nothing priced it blocks two reactions.
    assert ranked["a_c"] == 2
    assert summarize(result).startswith("organism:")


YEAST_GEM = osp.join(
    os.environ.get("DATA_ROOT", "/scratch/projects/torchcell-scratch"),
    "data",
    "torchcell",
    "yeast-GEM",
    "yeast-GEM-9.0.2",
    "model",
    "yeast-GEM.xml",
)


@pytest.mark.skipif(not osp.isfile(YEAST_GEM), reason="yeast-GEM 9.0.2 is not on disk")
def test_yeast_gem_is_just_another_model() -> None:
    """The organism-agnostic claim, against the test case it was written from."""
    model = cobra.io.read_sbml_model(YEAST_GEM)
    identities = identities_from_model(model)
    assert len(identities) == len(model.metabolites)
    assert any(m.has_placeholder_formula for m in identities)

    conditions = CompartmentConditions(p_h={"c": 7.2, "e": 5.0}, source="test")
    result = run_thermo_pipeline(
        model, ThermoPipelineConfig(organism="S. cerevisiae", conditions=conditions), []
    )
    # With no stages the only reactions that carry a value are the ones whose every
    # participant CANCELS, whose chemical energy is zero whatever the table says, and
    # every other reaction says why it has none.
    priced = [r for r in result.reactions if r.delta_r_g0_kj_per_mol is not None]
    assert all(r.route is ThermoRoute.CANCELLED_PARTICIPANTS for r in priced)
    assert all(r.delta_r_g0_kj_per_mol == 0.0 for r in priced)
    assert all(
        r.reason is not None
        for r in result.reactions
        if r.delta_r_g0_kj_per_mol is None
    )
    # Every compartment other than the two named is reported as running on the default.
    assert "v" in result.compartments_at_default
    assert "c" not in result.compartments_at_default


def test_a_later_stage_with_nothing_to_say_does_not_erase_a_stated_reason() -> None:
    """NOT_ATTEMPTED means "no comment", not "no reason".

    A diagnosis-only stage placed last returns NOT_ATTEMPTED for every metabolite it does
    not cover. Letting that overwrite an earlier stage's specific reason wiped the
    group-decomposition and pKa reasons off the real yeast-GEM run and pushed those species
    into the generic fallback, which silently changed the coverage ceiling.
    """

    class Specific:
        name = "specific"

        def resolve(self, metabolites, conditions):
            return [
                MetabolitePrice(
                    met_id=m.met_id, reason=UnpricedReason.GROUP_DECOMPOSITION_FAILED
                )
                for m in metabolites
            ]

    class NoComment:
        name = "no_comment"

        def resolve(self, metabolites, conditions):
            return [
                MetabolitePrice(met_id=m.met_id, reason=UnpricedReason.NOT_ATTEMPTED)
                for m in metabolites
            ]

    identities = identities_from_model(_synthetic_model())
    prices = {
        p.met_id: p
        for p in resolve_metabolites(
            identities, [Specific(), NoComment()], CompartmentConditions()
        )
    }
    # b_c has a definite formula and no structure, so the generic fallback would have
    # called it NO_STRUCTURE_AVAILABLE; the specific stage said otherwise and wins.
    assert prices["b_c"].reason is UnpricedReason.GROUP_DECOMPOSITION_FAILED
