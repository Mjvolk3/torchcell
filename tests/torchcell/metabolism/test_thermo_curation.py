# tests/torchcell/metabolism/test_thermo_curation.py

"""Tests for curated thermodynamic records and the stage that prices from them.

Built on the same synthetic GEM the pipeline tests use, extended with the three cases
curation exists for: a species with no structure that a curator supplies one for, a species
whose GEM formula is wrong, and a redox carrier that has a midpoint potential rather than a
formation energy. The structure route is a stub, because what is under test is the
admission policy and the routing, not component contribution.
"""

from __future__ import annotations

import json
import os.path as osp
from pathlib import Path
from typing import Any

import cobra
import pytest

from torchcell.metabolism.thermo_curation import (
    ADMITTED,
    Admission,
    CarrierPotentialValue,
    CuratedRecordSet,
    CuratedResolver,
    CuratedThermoRecord,
    CurationKind,
    PricedStructure,
    PricedStructureTable,
    StructureValue,
    apply_formula_corrections,
    carrier_direction,
    load_curated_records,
    pricing_requests,
)
from torchcell.metabolism.thermo_pipeline import (
    CompartmentConditions,
    MetaboliteIdentity,
    MetabolitePrice,
    ThermoPipelineConfig,
    ThermoRoute,
    UnpricedReason,
    cancelling_participants,
    chemical_identity,
    element_counts,
    identities_from_model,
    is_non_chemical_reaction,
    legendre_transform_kj,
    reaction_balance,
    run_thermo_pipeline,
)

CONDITIONS = CompartmentConditions(
    p_h={"c": 7.2, "e": 5.0}, temperature_k=303.15, source="test"
)


def _model() -> cobra.Model:
    """A GEM with a transport that cancels, an exchange, and a redox carrier couple."""
    model = cobra.Model("synthetic_curation")
    a = cobra.Metabolite("a_c", formula="C6H12O6", name="alpha", compartment="c")
    a.annotation = {"smiles": "OCC1OC(O)C(O)C(O)C1O"}
    b = cobra.Metabolite("b_c", formula="C3H4O3", name="beta", compartment="c")
    poly_c = cobra.Metabolite(
        "poly_c", formula="C6H10O5", name="glycan", compartment="c"
    )
    poly_e = cobra.Metabolite(
        "poly_e", formula="C6H10O5", name="glycan", compartment="e"
    )
    ox = cobra.Metabolite(
        "ox_c", formula="C10H12FeN2", name="ferricarrier", compartment="c"
    )
    red = cobra.Metabolite(
        "red_c", formula="C10H13FeN2", name="ferrocarrier", compartment="c"
    )
    bad = cobra.Metabolite("bad_c", formula="C43H44N9O", name="wrong", compartment="c")

    transport = cobra.Reaction("T1")
    transport.add_metabolites({poly_c: -1.0, poly_e: 1.0})
    consume = cobra.Reaction("R1")
    consume.add_metabolites({a: -1.0, b: 2.0})
    redox = cobra.Reaction("R2")
    redox.add_metabolites({a: -1.0, ox: -2.0, b: 1.0, red: 2.0})
    with_poly = cobra.Reaction("R3")
    with_poly.add_metabolites({poly_c: -1.0, a: 1.0})
    wrong = cobra.Reaction("R4")
    wrong.add_metabolites({bad: -1.0, b: 1.0})
    model.add_reactions([transport, consume, redox, with_poly, wrong])
    exchange = cobra.Reaction("EX_poly_e")
    exchange.name = "glycan exchange"
    exchange.add_metabolites({poly_e: -1.0})
    pseudo = cobra.Reaction("BIO")
    pseudo.name = "biomass pseudoreaction"
    pseudo.add_metabolites({a: -1.0, b: -1.0})
    model.add_reactions([exchange, pseudo])
    return model


def _record(**kwargs: Any) -> dict[str, Any]:
    base = {
        "slug": "x",
        "met_ids": ["a_c"],
        "kind": "structure",
        "admission": "tabulated",
        "value": {"smiles": "CCO", "formula": "C2H6O", "charge": 0},
    }
    base.update(kwargs)
    return base


class _StubPricer:
    """A structure pricer that prices whatever it is told to and refuses the rest."""

    name = "stub"

    def __init__(self, values: dict[str, float], outcome: str = "priced") -> None:
        self.values = values
        self.outcome = outcome
        self.calls: list[tuple[str, tuple[float, ...]]] = []

    def price(
        self,
        smiles: str,
        pkas: list[float],
        p_h: float,
        ionic_strength_m: float,
        temperature_k: float,
        p_mg: float,
    ) -> PricedStructure:
        self.calls.append((smiles, tuple(pkas)))
        if smiles not in self.values:
            return PricedStructure(
                smiles=smiles,
                p_h=p_h,
                ionic_strength_m=ionic_strength_m,
                temperature_k=temperature_k,
                p_mg=p_mg,
                pkas=list(pkas),
                outcome="failed_group_decomposition",
            )
        return PricedStructure(
            smiles=smiles,
            p_h=p_h,
            ionic_strength_m=ionic_strength_m,
            temperature_k=temperature_k,
            p_mg=p_mg,
            pkas=list(pkas),
            dgf_prime_kj_per_mol=self.values[smiles],
            sigma_kj_per_mol=2.0,
            group_decomposition_ok=True,
            outcome=self.outcome,
        )


class _FixedPrices:
    """A stage that prices exactly the ids it is given, for the reaction-level tests."""

    name = "fixed"

    def __init__(self, values: dict[str, float]) -> None:
        self.values = values

    def resolve(
        self, metabolites: list[MetaboliteIdentity], conditions: CompartmentConditions
    ) -> list[MetabolitePrice]:
        return [
            MetabolitePrice(
                met_id=m.met_id,
                dgf_prime_kj_per_mol=self.values.get(m.met_id),
                route=(
                    ThermoRoute.CACHE_ACCESSION
                    if m.met_id in self.values
                    else ThermoRoute.NONE
                ),
                reason=None
                if m.met_id in self.values
                else UnpricedReason.NOT_ATTEMPTED,
            )
            for m in metabolites
        ]


def _set(*records: dict[str, Any]) -> CuratedRecordSet:
    return CuratedRecordSet(
        directory="memory",
        records=[CuratedThermoRecord.model_validate(r) for r in records],
    )


# ------------------------------------------------------------------ the record itself


def test_the_value_is_parsed_with_the_model_its_kind_names() -> None:
    structure = CuratedThermoRecord.model_validate(_record())
    assert isinstance(structure.value, StructureValue)
    assert structure.value.smiles == "CCO"
    carrier = CuratedThermoRecord.model_validate(
        _record(
            kind="carrier_potential",
            admission="measured",
            value={"e_prime_mv": 250.0, "n_electrons": 1},
        )
    )
    assert isinstance(carrier.value, CarrierPotentialValue)
    assert carrier.value.e_prime_mv == 250.0
    assert carrier.value.n_electrons == 1


def test_a_value_of_the_wrong_shape_for_its_kind_raises() -> None:
    with pytest.raises(Exception, match="e_prime_mv"):
        CuratedThermoRecord.model_validate(
            _record(
                kind="carrier_potential", admission="measured", value={"smiles": "CCO"}
            )
        )


def test_extra_keys_are_kept_inside_value_and_refused_at_the_top() -> None:
    kept = CuratedThermoRecord.model_validate(
        _record(value={"smiles": "CCO", "formula": "C2H6O", "why_this_one": "because"})
    )
    assert isinstance(kept.value, StructureValue)
    assert kept.value.model_extra is not None
    assert kept.value.model_extra["why_this_one"] == "because"
    with pytest.raises(Exception, match="extra_forbidden|Extra inputs"):
        CuratedThermoRecord.model_validate(_record(unexpected_top_level_key=1))


def test_not_found_and_admission_have_to_agree() -> None:
    with pytest.raises(Exception, match="admission=none needs kind=not_found"):
        CuratedThermoRecord.model_validate(_record(admission="none"))
    with pytest.raises(Exception, match="cannot carry admission"):
        CuratedThermoRecord.model_validate(
            _record(kind="not_found", admission="tabulated", value=None)
        )
    fine = CuratedThermoRecord.model_validate(
        _record(kind="not_found", admission="none", value=None)
    )
    assert not fine.admitted


def test_only_measured_and_tabulated_are_admitted() -> None:
    assert ADMITTED == {Admission.MEASURED, Admission.TABULATED}
    for admission, expected in (
        ("measured", True),
        ("tabulated", True),
        ("computed", False),
        ("provisional", False),
    ):
        record = CuratedThermoRecord.model_validate(_record(admission=admission))
        assert record.admitted is expected


def test_the_loader_validates_every_file_and_names_the_one_that_failed(
    tmp_path: Path,
) -> None:
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_record(slug="good")))
    assert len(load_curated_records(str(tmp_path)).records) == 1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"slug": "bad", "met_ids": ["a_c"]}))
    with pytest.raises(ValueError, match="bad.json is not a valid curation record"):
        load_curated_records(str(tmp_path))


def test_the_file_name_has_to_be_the_slug(tmp_path: Path) -> None:
    (tmp_path / "heme.json").write_text(json.dumps(_record(slug="heme_a")))
    with pytest.raises(ValueError, match="holds slug 'heme_a', not 'heme'"):
        load_curated_records(str(tmp_path))


# ------------------------------------------------------------------ the resolver


def test_an_admitted_structure_is_priced_through_the_structure_route() -> None:
    records = _set(_record(slug="alpha", met_ids=["a_c"], value={"smiles": "CCO"}))
    pricer = _StubPricer({"CCO": -180.0})
    prices = {
        p.met_id: p
        for p in CuratedResolver(records, pricer).resolve(
            identities_from_model(_model()), CONDITIONS
        )
    }
    assert prices["a_c"].dgf_prime_kj_per_mol == pytest.approx(-180.0)
    assert prices["a_c"].route is ThermoRoute.CURATED_STRUCTURE
    assert prices["a_c"].p_h == 7.2
    assert prices["b_c"].reason is UnpricedReason.NOT_ATTEMPTED


def test_a_computed_record_is_reported_and_never_priced() -> None:
    records = _set(_record(slug="alpha", admission="computed", value={"smiles": "CCO"}))
    prices = {
        p.met_id: p
        for p in CuratedResolver(records, _StubPricer({"CCO": -180.0})).resolve(
            identities_from_model(_model()), CONDITIONS
        )
    }
    assert prices["a_c"].dgf_prime_kj_per_mol is None
    assert prices["a_c"].reason is UnpricedReason.RECORDED_NOT_ADMITTED


def test_a_structure_of_the_wrong_composition_is_refused() -> None:
    records = _set(_record(slug="alpha", value={"smiles": "CCO", "formula": "C2H6O"}))
    prices = {
        p.met_id: p
        for p in CuratedResolver(records, _StubPricer({"CCO": -180.0})).resolve(
            identities_from_model(_model()), CONDITIONS
        )
    }
    # a_c is C6H12O6 in the model and the record says C2H6O.
    assert prices["a_c"].dgf_prime_kj_per_mol is None
    assert prices["a_c"].reason is UnpricedReason.NO_STRUCTURE_AVAILABLE
    assert "different molecule" in prices["a_c"].detail


def test_a_structure_the_decomposer_refuses_says_so() -> None:
    records = _set(_record(slug="alpha", value={"smiles": "CCO"}))
    prices = {
        p.met_id: p
        for p in CuratedResolver(records, _StubPricer({})).resolve(
            identities_from_model(_model()), CONDITIONS
        )
    }
    assert prices["a_c"].reason is UnpricedReason.GROUP_DECOMPOSITION_FAILED


def test_an_attachment_point_is_never_sent_to_the_structure_route() -> None:
    records = _set(
        _record(
            slug="glycan",
            met_ids=["poly_c"],
            kind="polymer_convention",
            value={"repeat_unit_smiles": "*OC1OC(*)C(O)C(O)C1O"},
        )
    )
    pricer = _StubPricer({})
    prices = {
        p.met_id: p
        for p in CuratedResolver(records, pricer).resolve(
            identities_from_model(_model()), CONDITIONS
        )
    }
    assert prices["poly_c"].reason is UnpricedReason.PLACEHOLDER_STRUCTURE
    assert pricer.calls == []


def test_a_curated_value_is_used_when_its_conditions_match_and_refused_when_they_do_not() -> (
    None
):
    def _records(p_h: float) -> CuratedRecordSet:
        return _set(
            _record(
                slug="glycan",
                met_ids=["poly_c"],
                kind="polymer_convention",
                value={
                    "repeat_unit_smiles": "*OC1OC(*)C(O)C(O)C1O",
                    "preferred_dimer_difference": {
                        "dgf_prime_kj_per_mol": {"poly_c": -248.7},
                        "sigma_kj_per_mol": 1.44,
                        "conditions": {
                            "p_h": p_h,
                            "ionic_strength_m": 0.25,
                            "temperature_k": 303.15,
                        },
                    },
                },
            )
        )

    identities = identities_from_model(_model())
    priced = {
        p.met_id: p
        for p in CuratedResolver(_records(7.2), _StubPricer({})).resolve(
            identities, CONDITIONS
        )
    }
    assert priced["poly_c"].dgf_prime_kj_per_mol == pytest.approx(-248.7)
    assert priced["poly_c"].sigma_kj_per_mol == pytest.approx(1.44)
    refused = {
        p.met_id: p
        for p in CuratedResolver(_records(6.0), _StubPricer({})).resolve(
            identities, CONDITIONS
        )
    }
    assert refused["poly_c"].dgf_prime_kj_per_mol is None
    assert refused["poly_c"].reason is UnpricedReason.RECORDED_NOT_ADMITTED


def test_a_formation_energy_is_transformed_to_the_compartment() -> None:
    records = _set(
        _record(
            slug="alpha",
            kind="formation_energy",
            admission="measured",
            value={
                "dgf_kj_per_mol": -900.0,
                "standard_or_transformed": "standard",
                "T_K": 303.15,
                "charge": -1,
                "num_protons": 2,
            },
        )
    )
    price = {
        p.met_id: p
        for p in CuratedResolver(records).resolve(
            identities_from_model(_model()), CONDITIONS
        )
    }["a_c"]
    expected = -900.0 + legendre_transform_kj(
        charge=-1, num_protons=2, p_h=7.2, ionic_strength_m=0.25, temperature_k=303.15
    )
    assert price.dgf_prime_kj_per_mol == pytest.approx(expected)
    assert price.route is ThermoRoute.CURATED_FORMATION


def test_a_transformed_formation_energy_round_trips_through_its_own_conditions() -> (
    None
):
    """Reporting a value at the pipeline's own conditions must return it unchanged."""
    transformed = -900.0 + legendre_transform_kj(
        charge=-1, num_protons=2, p_h=7.2, ionic_strength_m=0.25, temperature_k=303.15
    )
    records = _set(
        _record(
            slug="alpha",
            kind="formation_energy",
            admission="measured",
            value={
                "dgf_kj_per_mol": transformed,
                "standard_or_transformed": "transformed",
                "T_K": 303.15,
                "pH": 7.2,
                "ionic_strength_M": 0.25,
                "charge": -1,
                "num_protons": 2,
            },
        )
    )
    price = {
        p.met_id: p
        for p in CuratedResolver(records).resolve(
            identities_from_model(_model()), CONDITIONS
        )
    }["a_c"]
    assert price.dgf_prime_kj_per_mol == pytest.approx(transformed)
    assert price.provenance["standard_kj_per_mol"] == pytest.approx(-900.0)


def test_an_admitted_structure_without_a_pricer_raises_rather_than_being_dropped() -> (
    None
):
    records = _set(_record(slug="alpha", value={"smiles": "CCO"}))
    with pytest.raises(ValueError, match="need the structure route"):
        CuratedResolver(records)


def test_the_pricing_requests_are_one_per_metabolite_and_deduplicated() -> None:
    records = _set(
        _record(slug="glycan", met_ids=["poly_c", "poly_e"], value={"smiles": "CCO"})
    )
    requests = pricing_requests(records, identities_from_model(_model()), CONDITIONS)
    # poly_c and poly_e sit at different pH, so the same structure is two calls.
    assert len(requests) == 2
    assert {round(r.p_h, 1) for r in requests} == {7.2, 5.0}
    same = _set(_record(slug="glycan", met_ids=["poly_c"], value={"smiles": "CCO"}))
    assert len(pricing_requests(same, identities_from_model(_model()), CONDITIONS)) == 1


def test_the_priced_structure_table_raises_on_a_pairing_it_does_not_hold(
    tmp_path: Path,
) -> None:
    path = tmp_path / "priced.json"
    row = PricedStructure(
        smiles="CCO",
        p_h=7.2,
        ionic_strength_m=0.25,
        temperature_k=303.15,
        p_mg=3.0,
        dgf_prime_kj_per_mol=-180.0,
        outcome="priced",
    )
    path.write_text(json.dumps({"records": [row.model_dump()]}))
    table = PricedStructureTable(str(path))
    assert table.price("CCO", [], 7.2, 0.25, 303.15, 3.0).dgf_prime_kj_per_mol == -180.0
    with pytest.raises(KeyError, match="holds no pricing"):
        table.price("CCO", [], 5.0, 0.25, 303.15, 3.0)


# ------------------------------------------------------------------ formula corrections


def test_a_formula_correction_rewrites_the_identity_and_is_recorded() -> None:
    records = _set(
        _record(
            slug="fixed",
            met_ids=["bad_c"],
            kind="formula_correction",
            admission="tabulated",
            value={
                "proposed_formula": "C55H100N3O39P3",
                "proposed_charge": -2,
                "derivation": "walked the pathway",
            },
        )
    )
    identities, overrides = apply_formula_corrections(
        identities_from_model(_model()), records
    )
    fixed = {m.met_id: m for m in identities}["bad_c"]
    assert fixed.formula == "C55H100N3O39P3"
    assert fixed.charge == -2
    assert overrides[0].gem_formula == "C43H44N9O"
    assert overrides[0].slug == "fixed"


def test_a_provisional_formula_correction_changes_nothing() -> None:
    records = _set(
        _record(
            slug="fixed",
            met_ids=["bad_c"],
            kind="formula_correction",
            admission="provisional",
            value={"proposed_formula": "C55H100N3O39P3"},
        )
    )
    identities, overrides = apply_formula_corrections(
        identities_from_model(_model()), records
    )
    assert overrides == []
    assert {m.met_id: m for m in identities}["bad_c"].formula == "C43H44N9O"


# ------------------------------------------------------------------ carriers


def _carrier_records(**value: Any) -> CuratedRecordSet:
    payload: dict[str, Any] = {
        "e_prime_mv": 272.0,
        "n_electrons": 1,
        "partner_couple_e_prime_mv": 45.0,
    }
    payload.update(value)
    return _set(
        _record(
            slug="carrier",
            met_ids=["ox_c", "red_c"],
            kind="carrier_potential",
            admission="measured",
            value=payload,
        )
    )


def test_the_direction_of_a_couple_comes_from_the_record_then_charge_then_hydrogen() -> (
    None
):
    value = CarrierPotentialValue(e_prime_mv=1.0, n_electrons=1)
    low = MetaboliteIdentity(met_id="lo", formula="Fe", charge=2)
    high = MetaboliteIdentity(met_id="hi", formula="Fe", charge=3)
    by_charge = carrier_direction(value, low, high)
    assert by_charge is not None
    assert by_charge.reduced_met_id == "lo"
    assert by_charge.decided_by == "charge"

    # Equal charge, one more hydrogen ATOM: that hydrogen is a proton plus an electron.
    fewer = MetaboliteIdentity(met_id="ferri", formula="C42H44FeN8O8S2R4", charge=0)
    more = MetaboliteIdentity(met_id="ferro", formula="C42H45FeN8O8S2R4", charge=0)
    by_hydrogen = carrier_direction(value, fewer, more)
    assert by_hydrogen is not None
    assert by_hydrogen.reduced_met_id == "ferro"
    assert by_hydrogen.decided_by == "hydrogen_at_equal_charge"

    stated = CarrierPotentialValue(
        e_prime_mv=1.0,
        n_electrons=1,
        reduced_met_ids=["ferri"],
        oxidized_met_ids=["ferro"],
    )
    by_record = carrier_direction(stated, fewer, more)
    assert by_record is not None
    assert by_record.reduced_met_id == "ferri"
    assert by_record.decided_by == "record"

    blind = MetaboliteIdentity(met_id="a", formula="", charge=0)
    also_blind = MetaboliteIdentity(met_id="b", formula="", charge=0)
    assert carrier_direction(value, blind, also_blind) is None


def test_an_undetermined_direction_prices_nothing_and_says_so() -> None:
    records = _set(
        _record(
            slug="carrier",
            met_ids=["poly_c", "poly_e"],
            kind="carrier_potential",
            admission="measured",
            value={"e_prime_mv": 5.0, "n_electrons": 1},
        )
    )
    prices = {
        p.met_id: p
        for p in CuratedResolver(records).resolve(
            identities_from_model(_model()), CONDITIONS
        )
    }
    # Two copies of one glycan in different compartments: nothing says which is reduced,
    # and they are not even in the same compartment.
    assert prices["poly_c"].dgf_prime_kj_per_mol is None
    assert prices["poly_c"].reason is UnpricedReason.NOT_ATTEMPTED


def test_a_carrier_couple_gets_a_pseudo_pair_whose_difference_is_the_potential() -> (
    None
):
    prices = {
        p.met_id: p
        for p in CuratedResolver(_carrier_records()).resolve(
            identities_from_model(_model()), CONDITIONS
        )
    }
    # ferrocarrier has one more hydrogen at equal charge, so it is the reduced form.
    assert prices["ox_c"].dgf_prime_kj_per_mol == 0.0
    assert prices["red_c"].dgf_prime_kj_per_mol == pytest.approx(
        -1 * 96.48533212 * 0.272, abs=1e-6
    )
    assert prices["ox_c"].route is ThermoRoute.CARRIER_POTENTIAL
    assert prices["red_c"].provenance["form"] == "reduced"
    assert (
        prices["red_c"].provenance["direction_decided_by"] == "hydrogen_at_equal_charge"
    )


def test_a_carrier_needs_no_partner_couple_to_price_a_reaction() -> None:
    """The gauge cancels, so the ordinary stoichiometric sum is the reaction energy."""
    records = _carrier_records(partner_couple_e_prime_mv=None)
    stage = CuratedResolver(records)
    known = _FixedPrices({"a_c": -900.0, "b_c": -470.0})
    result = run_thermo_pipeline(
        _model(), ThermoPipelineConfig(conditions=CONDITIONS), [stage, known]
    )
    reactions = {r.reaction_id: r for r in result.reactions}
    # R2: a + 2 ox -> b + 2 red. The carrier contributes 2 * (dGf'(red) - dGf'(ox)).
    carrier = -1 * 96.48533212 * 0.272
    assert reactions["R2"].delta_r_g0_kj_per_mol == pytest.approx(
        (-470.0 + 2 * carrier) - (-900.0 + 0.0), abs=1e-6
    )
    assert reactions["R2"].route is ThermoRoute.MIXED


def test_the_potential_moves_with_ph_only_when_the_record_gives_a_slope() -> None:
    flat = {
        p.met_id: p
        for p in CuratedResolver(_carrier_records(pH=7.4)).resolve(
            identities_from_model(_model()), CONDITIONS
        )
    }["red_c"]
    assert flat.provenance["e_prime_mv_used"] == 272.0
    sloped = {
        p.met_id: p
        for p in CuratedResolver(
            _carrier_records(pH=7.4, e_prime_slope_mv_per_ph=-10.0)
        ).resolve(identities_from_model(_model()), CONDITIONS)
    }["red_c"]
    # The cytosol is at pH 7.2, so a -10 mV/pH slope raises E' by 2 mV.
    assert sloped.provenance["e_prime_mv_used"] == pytest.approx(274.0)


def test_a_reaction_that_does_not_conserve_the_carrier_is_refused() -> None:
    """The pseudo pair's gauge is arbitrary and only cancels where it is conserved."""
    model = _model()
    synthase = cobra.Reaction("SYN")
    synthase.name = "carrier synthase"
    synthase.add_metabolites(
        {
            model.metabolites.get_by_id("a_c"): -1.0,
            model.metabolites.get_by_id("red_c"): 1.0,
        }
    )
    model.add_reactions([synthase])
    result = run_thermo_pipeline(
        model,
        ThermoPipelineConfig(conditions=CONDITIONS),
        [
            CuratedResolver(_carrier_records()),
            _FixedPrices({"a_c": -900.0, "b_c": -470.0}),
        ],
    )
    reactions = {r.reaction_id: r for r in result.reactions}
    assert reactions["SYN"].delta_r_g0_kj_per_mol is None
    assert reactions["SYN"].reason is UnpricedReason.CARRIER_NOT_CONSERVED
    assert reactions["SYN"].blocking_met_ids == ["red_c"]
    # The conserved reaction in the same model is still priced.
    assert reactions["R2"].delta_r_g0_kj_per_mol is not None


def test_a_provisional_carrier_prices_nothing() -> None:
    records = _set(
        _record(
            slug="carrier",
            met_ids=["ox_c", "red_c"],
            kind="carrier_potential",
            admission="provisional",
            value={"e_prime_mv": 5.5, "n_electrons": 1},
        )
    )
    prices = {
        p.met_id: p
        for p in CuratedResolver(records).resolve(
            identities_from_model(_model()), CONDITIONS
        )
    }
    assert prices["ox_c"].dgf_prime_kj_per_mol is None
    assert prices["ox_c"].reason is UnpricedReason.RECORDED_NOT_ADMITTED


# ------------------------------------------------------------------ the reaction rules


def test_identical_species_on_both_sides_cancel_and_unblock_the_reaction() -> None:
    identities = {
        m.met_id: chemical_identity(m) for m in identities_from_model(_model())
    }
    assert identities["poly_c"] == identities["poly_e"]
    cancelled = cancelling_participants(
        {"poly_c": -1.0, "poly_e": 1.0}, identities, {"poly_c", "poly_e"}
    )
    assert cancelled == ["poly_c", "poly_e"]
    # A group with no unpriced member is left alone, so no reaction that already had a
    # value gets a different one.
    assert (
        cancelling_participants({"poly_c": -1.0, "poly_e": 1.0}, identities, set())
        == []
    )
    # A group that does not sum to zero is not a cancellation.
    assert (
        cancelling_participants(
            {"poly_c": -2.0, "poly_e": 1.0}, identities, {"poly_c", "poly_e"}
        )
        == []
    )


def test_chemical_identity_separates_two_charges_of_one_formula() -> None:
    ferri = MetaboliteIdentity(
        met_id="a", name="Ferricytochrome b5", formula="Fe", charge=3
    )
    ferro = MetaboliteIdentity(
        met_id="b", name="Ferrocytochrome b5", formula="Fe", charge=2
    )
    same = MetaboliteIdentity(
        met_id="c", name="ferricytochrome  b5", formula="Fe", charge=3
    )
    assert chemical_identity(ferri) != chemical_identity(ferro)
    assert chemical_identity(ferri) == chemical_identity(same)


def test_a_cancelling_transport_is_priced_and_flagged() -> None:
    result = run_thermo_pipeline(
        _model(), ThermoPipelineConfig(conditions=CONDITIONS), []
    )
    reactions = {r.reaction_id: r for r in result.reactions}
    # Nothing is priced, so T1 is priced only because everything in it cancels.
    assert reactions["T1"].delta_r_g0_kj_per_mol == pytest.approx(0.0)
    assert reactions["T1"].route is ThermoRoute.CANCELLED_PARTICIPANTS
    assert reactions["T1"].cancelled_met_ids == ["poly_c", "poly_e"]
    assert reactions["R3"].delta_r_g0_kj_per_mol is None


def test_cancellation_can_be_turned_off() -> None:
    config = ThermoPipelineConfig(
        conditions=CONDITIONS, cancel_identical_participants=False
    )
    result = run_thermo_pipeline(_model(), config, [])
    reactions = {r.reaction_id: r for r in result.reactions}
    assert reactions["T1"].delta_r_g0_kj_per_mol is None


def test_exchanges_and_pseudoreactions_leave_the_chemical_denominator() -> None:
    model = _model()
    config = ThermoPipelineConfig(conditions=CONDITIONS)
    assert is_non_chemical_reaction(model.reactions.get_by_id("EX_poly_e"), config)
    assert is_non_chemical_reaction(model.reactions.get_by_id("BIO"), config)
    assert not is_non_chemical_reaction(model.reactions.get_by_id("R1"), config)
    result = run_thermo_pipeline(model, config, [])
    assert result.n_non_chemical_reactions == 2
    assert result.reaction_counts.n_total == 7
    assert result.reaction_counts_chemical.n_total == 5
    reactions = {r.reaction_id: r for r in result.reactions}
    assert reactions["EX_poly_e"].reason is UnpricedReason.NON_CHEMICAL
    assert reactions["R1"].reason is UnpricedReason.BLOCKED_BY_METABOLITE


def test_the_non_chemical_rule_is_a_config_field() -> None:
    model = _model()
    off = ThermoPipelineConfig(
        conditions=CONDITIONS,
        non_chemical_boundary=False,
        non_chemical_name_tokens=[],
        non_chemical_sbo_terms=[],
    )
    assert not is_non_chemical_reaction(model.reactions.get_by_id("EX_poly_e"), off)
    assert run_thermo_pipeline(model, off, []).n_non_chemical_reactions == 0


def test_element_counts_refuses_a_placeholder_formula() -> None:
    assert element_counts("C6H12O6") == {"C": 6.0, "H": 12.0, "O": 6.0}
    with pytest.raises(ValueError, match="placeholder formula"):
        element_counts("C10H18RO")
    with pytest.raises(ValueError, match="empty formula"):
        element_counts("")


def test_an_imbalanced_reaction_is_reported_and_still_priced() -> None:
    model = _model()
    balance = reaction_balance(
        model.reactions.get_by_id("R4"),
        {m.met_id: m for m in identities_from_model(model)},
    )
    assert balance is not None
    assert balance.reaction_id == "R4"
    assert set(balance.offending_met_ids) == {"bad_c", "b_c"}
    # The balanced transport is not in the list.
    assert (
        reaction_balance(
            model.reactions.get_by_id("T1"),
            {m.met_id: m for m in identities_from_model(model)},
        )
        is None
    )
    result = run_thermo_pipeline(model, ThermoPipelineConfig(conditions=CONDITIONS), [])
    assert "R4" in {b.reaction_id for b in result.imbalanced_reactions}


# ------------------------------------------------------------------ the real records


CURATION_DIR = osp.join(
    osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))),
    "experiments",
    "026-metabolism-flux",
    "results",
    "curation",
)


@pytest.mark.skipif(
    not osp.isdir(CURATION_DIR)
    or not osp.isfile(osp.join(CURATION_DIR, "heme_a.json")),
    reason="the curation directory is not populated",
)
def test_every_shipped_curation_record_validates() -> None:
    """The loader raises on a malformed record, so this passing IS the validation."""
    records = load_curated_records(CURATION_DIR)
    assert records.records
    for record in records.records:
        assert record.met_ids
        assert record.kind in set(CurationKind)
        if record.admitted:
            assert record.sources, f"{record.slug} is admitted with no source"


# ------------------------------------------------------------------ price filters


def test_a_price_filter_may_withdraw_a_value_a_stage_produced() -> None:
    """A composition check finds a value that EXISTS and is wrong; a stage cannot undo it."""

    class Guard:
        name = "guard"

        def __call__(self, prices: list[MetabolitePrice]) -> list[MetabolitePrice]:
            return [
                (
                    MetabolitePrice(
                        met_id=p.met_id,
                        reason=UnpricedReason.NO_STRUCTURE_AVAILABLE,
                        reason_is_final=True,
                        detail="the structure behind this price is another molecule",
                    )
                    if p.met_id == "a_c"
                    else p
                )
                for p in prices
            ]

    model = _model()
    result = run_thermo_pipeline(
        model,
        ThermoPipelineConfig(conditions=CONDITIONS),
        [_FixedPrices({"a_c": -900.0, "b_c": -470.0})],
        price_filters=[Guard()],
    )
    prices = {p.met_id: p for p in result.metabolites}
    assert prices["a_c"].dgf_prime_kj_per_mol is None
    assert prices["b_c"].dgf_prime_kj_per_mol == -470.0
    # R1 loses its value with a_c, and says which metabolite took it away.
    reactions = {r.reaction_id: r for r in result.reactions}
    assert reactions["R1"].delta_r_g0_kj_per_mol is None
    assert reactions["R1"].blocking_met_ids == ["a_c"]


def test_a_price_filter_that_changes_the_metabolite_set_raises() -> None:
    class Rogue:
        name = "rogue"

        def __call__(self, prices: list[MetabolitePrice]) -> list[MetabolitePrice]:
            return prices[:-1]

    with pytest.raises(ValueError, match="changed which metabolites"):
        run_thermo_pipeline(
            _model(),
            ThermoPipelineConfig(conditions=CONDITIONS),
            [],
            price_filters=[Rogue()],
        )


def test_a_structure_value_may_carry_its_own_pka_ladder() -> None:
    records = _set(
        _record(slug="glycan", value={"smiles": "CCO", "pkas": [5.25, 3.19]})
    )
    pricer = _StubPricer({"CCO": -180.0})
    CuratedResolver(records, pricer).resolve(
        identities_from_model(_model()), CONDITIONS
    )
    assert pricer.calls[0][1] == (5.25, 3.19)
