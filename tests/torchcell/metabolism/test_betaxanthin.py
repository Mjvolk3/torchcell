# tests/torchcell/metabolism/test_betaxanthin.py
# [[tests.torchcell.metabolism.test_betaxanthin]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/metabolism/test_betaxanthin.py
"""The betaxanthin cassette against a ten-species stand-in for yeast-GEM.

Fixture: a cobra model holding only the cytosolic species the cassette touches, under
their yeast-GEM 9.0.2 ids and charged formulas: tyrosine ``s_1051`` C9H11NO3 (0), O2
``s_1275``, NADPH ``s_1212`` C21H26N7O17P3 (-4), NADP+ ``s_1207`` C21H25N7O17P3 (-3),
water ``s_0803``, proton ``s_0794`` H (+1), and three more partners: alanine ``s_0955``
C3H7NO2 (0), glutamate ``s_0991`` C5H8NO4 (-1), lysine ``s_1025`` C6H15N2O2 (+1). So 4
of the 20 default partners are present (tyrosine is both substrate and partner).

Derivations (every condensation is betalamic acid C9H9NO5 + partner - H2O):
alanine -> C12H14N2O6 (0); glutamate -> C14H15N2O8 (-1); lysine -> C15H22N3O6 (+1);
tyrosine -> C18H18N2O7 (0); betanidin = betalamic + cyclo-DOPA C9H9NO4 - H2O =
C18H16N2O8 (0). The hydroxylase balances: C 30 = 30, H 38 = 38, O 22 = 22, charge
-3 = -3. Evidence census with the oxidase branch: convention 3 (th, DOD, DO), sourced
2 + 4 (cyclization, betanidin, one condensation per partner), derived 1 + 4 (betanidin
demand, one demand per partner).
"""

from __future__ import annotations

import logging

import cobra
import pytest

from torchcell.metabolism.betaxanthin import (
    DEFAULT_PARTNERS,
    betaxanthin_demand_ids,
    build_betaxanthin_pathway,
)
from torchcell.metabolism.pathway import EvidenceTier, apply_pathway

_SPECIES = {
    "s_1051": ("C9H11NO3", 0),
    "s_1275": ("O2", 0),
    "s_1212": ("C21H26N7O17P3", -4),
    "s_1207": ("C21H25N7O17P3", -3),
    "s_0803": ("H2O", 0),
    "s_0794": ("H", 1),
    "s_0955": ("C3H7NO2", 0),
    "s_0991": ("C5H8NO4", -1),
    "s_1025": ("C6H15N2O2", 1),
}


def _gem() -> cobra.Model:
    model = cobra.Model("mini_gem")
    model.add_metabolites(
        [
            cobra.Metabolite(mid, formula=f, charge=c, compartment="c")
            for mid, (f, c) in _SPECIES.items()
        ]
    )
    return model


def test_partners_filtered_to_the_model_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="torchcell.metabolism.betaxanthin")
    pw = build_betaxanthin_pathway(_gem())
    missing = sorted(set(DEFAULT_PARTNERS) - {"s_0955", "s_0991", "s_1025", "s_1051"})
    assert [r.getMessage() for r in caplog.records] == [
        f"betaxanthin: 16 partner(s) absent from model: {missing}"
    ]
    assert missing[0] == "s_0271" and len(missing) == 16
    assert [r.id for r in pw.reactions] == [
        "r_CYP76AD1_th",
        "r_DOD",
        "r_CYP76AD1_do",
        "r_cyclodopa_spont",
        "r_betanidin_spont",
        "DM_betanidin",
        "r_btx_alanine_spont",
        "DM_btx_alanine",
        "r_btx_glutamate_spont",
        "DM_btx_glutamate",
        "r_btx_lysine_spont",
        "DM_btx_lysine",
        "r_btx_tyrosine_spont",
        "DM_btx_tyrosine",
    ]
    assert [m.id for m in pw.metabolites] == [
        "s_ldopa_c",
        "s_betalamic_c",
        "s_dopaquinone_c",
        "s_cyclodopa_c",
        "s_betanidin_c",
        "s_btx_alanine_c",
        "s_btx_glutamate_c",
        "s_btx_lysine_c",
        "s_btx_tyrosine_c",
    ]


def test_stoichiometry_genes_and_spontaneity() -> None:
    pw = build_betaxanthin_pathway(_gem())
    by_id = {r.id: r for r in pw.reactions}
    assert by_id["r_CYP76AD1_th"].stoichiometry == {
        "s_1051": -1,
        "s_1275": -1,
        "s_1212": -1,
        "s_0794": -1,
        "s_ldopa_c": 1,
        "s_0803": 1,
        "s_1207": 1,
    }
    assert by_id["r_DOD"].stoichiometry == {
        "s_ldopa_c": -1,
        "s_1275": -1,
        "s_betalamic_c": 1,
        "s_0803": 1,
    }
    assert by_id["r_CYP76AD1_do"].stoichiometry == {
        "s_ldopa_c": -1,
        "s_1275": -0.5,
        "s_dopaquinone_c": 1,
        "s_0803": 1,
    }
    assert by_id["r_btx_glutamate_spont"].stoichiometry == {
        "s_betalamic_c": -1,
        "s_0991": -1,
        "s_btx_glutamate_c": 1,
        "s_0803": 1,
    }
    assert by_id["DM_btx_glutamate"].stoichiometry == {"s_btx_glutamate_c": -1}
    assert {
        r.id: r.gene_reaction_rule for r in pw.reactions if r.gene_reaction_rule
    } == {"r_CYP76AD1_th": "CYP76AD1", "r_DOD": "DOD", "r_CYP76AD1_do": "CYP76AD1"}
    assert sorted(r.id for r in pw.reactions if r.spontaneous) == [
        "r_betanidin_spont",
        "r_btx_alanine_spont",
        "r_btx_glutamate_spont",
        "r_btx_lysine_spont",
        "r_btx_tyrosine_spont",
        "r_cyclodopa_spont",
    ]
    assert pw.constitutive_genes == ["CYP76AD1", "DOD"]
    assert pw.base_strain == "BY4741"
    assert pw.source_keys == [
        "deloacheEnzymecoupledBiosensorEnables2015",
        "cacheraCRISPAHighthroughputMethod2023",
    ]
    assert pw.evidence_census() == {"convention": 3, "sourced": 6, "derived": 5}


def test_demand_ids_are_the_betaxanthin_family_only() -> None:
    """Betanidin is violet, so its demand is excluded from the yellow sum."""
    pw = build_betaxanthin_pathway(_gem())
    assert betaxanthin_demand_ids(pw) == [
        "DM_btx_alanine",
        "DM_btx_glutamate",
        "DM_btx_lysine",
        "DM_btx_tyrosine",
    ]


def test_product_ids_include_host_cofactors_and_betanidin() -> None:
    """Finding: ``HeterologousPathway.product_ids`` documents that for betaxanthin it is
    "one species per condensation partner", but on the real cassette it also returns
    water (``s_0803``), NADP+ (``s_1207``) and betanidin, because those are produced and
    never consumed by an internal reaction. No caller reads it yet. Pinned until the
    host species (and the side product) are excluded or the docstring says otherwise
    (pathway.py:137-148).
    """
    pw = build_betaxanthin_pathway(_gem())
    assert pw.product_ids() == [
        "s_0803",
        "s_1207",
        "s_betanidin_c",
        "s_btx_alanine_c",
        "s_btx_glutamate_c",
        "s_btx_lysine_c",
        "s_btx_tyrosine_c",
    ]


def test_without_the_oxidase_branch() -> None:
    pw = build_betaxanthin_pathway(_gem(), include_oxidase_branch=False)
    assert [m.id for m in pw.metabolites][:2] == ["s_ldopa_c", "s_betalamic_c"]
    assert "s_dopaquinone_c" not in [m.id for m in pw.metabolites]
    assert [r.id for r in pw.reactions][:3] == [
        "r_CYP76AD1_th",
        "r_DOD",
        "r_btx_alanine_spont",
    ]
    assert pw.evidence_census() == {"convention": 2, "sourced": 4, "derived": 4}


def test_explicit_partners_override_the_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="torchcell.metabolism.betaxanthin")
    pw = build_betaxanthin_pathway(_gem(), partners={"s_1025": "lysine"})
    assert caplog.records == []
    assert betaxanthin_demand_ids(pw) == ["DM_btx_lysine"]
    assert [m.evidence for m in pw.metabolites if m.id.startswith("s_btx_")] == [
        EvidenceTier.DERIVED
    ]


def test_applied_cassette_derives_charged_formulas_and_balances() -> None:
    """Every derived formula and charge matches the hand sums in the module docstring,
    and ``apply_pathway``'s own balance check passes on all 13 internal reactions.
    """
    out = apply_pathway(_gem(), build_betaxanthin_pathway(_gem()))
    got = {
        mid: (
            out.metabolites.get_by_id(mid).formula,
            out.metabolites.get_by_id(mid).charge,
        )
        for mid in [
            "s_btx_alanine_c",
            "s_btx_glutamate_c",
            "s_btx_lysine_c",
            "s_btx_tyrosine_c",
            "s_betanidin_c",
        ]
    }
    assert got == {
        "s_btx_alanine_c": ("C12H14N2O6", 0),
        "s_btx_glutamate_c": ("C14H15N2O8", -1),
        "s_btx_lysine_c": ("C15H22N3O6", 1),
        "s_btx_tyrosine_c": ("C18H18N2O7", 0),
        "s_betanidin_c": ("C18H16N2O8", 0),
    }
    th = out.reactions.get_by_id("r_CYP76AD1_th")
    assert th.check_mass_balance() == {}
    assert sorted(g.id for g in out.genes) == ["CYP76AD1", "DOD"]
    assert out.metabolites.get_by_id("s_betalamic_c").annotation == {
        "metanetx.chemical": "MNXM732452",
        "chebi": "CHEBI:27483",
    }
