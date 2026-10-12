# tests/torchcell/metabolism/test_transport.py

"""Tests for the Gibbs energy of transport.

Two levels. The first is a hand-built two-compartment GEM carrying one symport and one
antiport, where every quantity the module reports can be worked out on paper: which
species move, how many protons and how much charge cross, and what the two energy terms
come to. That level owns the SIGNS, which are the part of Thermo-Flux Eq. (7) a paper
cannot settle and a wrong choice of which silently inverts every proton pump in a model.

The second runs the module against yeast-GEM 9.0.2 and checks it against the reaction
column that model ships, on the reactions where both carry a nonzero term. That test is
skipped when the GEM is not on disk, and its tolerance is the MEASURED agreement rather
than a target: the two do not agree exactly and are not expected to, since the shipped
column also carries the protonation change of every transported reactant and this module
counts only explicit protons and charge.
"""

from __future__ import annotations

import math
import os
import os.path as osp
from typing import Any

import cobra
import pytest
import torch

from torchcell.metabolism.transport import (
    FARADAY_KJ_PER_MOL_V,
    GAS_CONSTANT_KJ,
    CompartmentElectrochemistry,
    TransportTable,
    build_transport_table,
    compare_to_shipped,
    pipeline_conditions,
    reaction_transport_term,
)

GEM_XML = osp.join(
    os.environ.get("DATA_ROOT", "/nonexistent"),
    "data/torchcell/yeast-GEM/yeast-GEM-9.0.2/model/yeast-GEM.xml",
)
GEM_RXN_DELTA_G = osp.join(
    os.environ.get("DATA_ROOT", "/nonexistent"),
    "data/torchcell/yeast-GEM/yeast-GEM-9.0.2/data/databases/model_rxnDeltaG.csv",
)

#: The shipped column is yETFL's, in kcal/mol despite the yeast9 paper's kJ/mol label.
#: Measured in experiments/026-metabolism-flux/results/thermo_yeast9_provenance.json:
#: regressing our kJ/mol formation energies on yETFL's over 1,927 shared metabolites gives
#: a slope of 4.187 against the thermochemical 4.184.
KJ_PER_KCAL = 4.184


def _toy_conditions() -> CompartmentElectrochemistry:
    """Two aqueous compartments, a 2.2-unit pH gradient and a 60 mV membrane potential.

    The numbers are the pipeline's own cytosol and extracellular values, so the toy and
    the real model exercise the same arithmetic.
    """
    return CompartmentElectrochemistry(
        p_h={"c": 7.2, "e": 5.0},
        ionic_strength_m={"c": 0.25, "e": 0.25},
        potential_mv={"e": 60.0},
        reference_compartment="c",
        temperature_k=303.15,
        source="test fixture",
    )


def _toy_model() -> cobra.Model:
    """A GEM with one proton symport and one charge-moving antiport.

    ``symport``  : H+[e] + S[e] -> H+[c] + S[c], with S carrying charge -1. One proton
    and one negative charge cross together, so the electrical work cancels exactly and
    only the pH term survives.

    ``antiport`` : A[c] + B[e] -> A[e] + B[c], with A carrying +1 and B neutral. No proton
    crosses, so the pH term is zero and only the electrical work survives, and it has the
    sign of moving a cation from the cytosol toward the more positive medium.

    ``internal`` : S[c] -> A[c], a same-compartment reaction that must produce nothing.
    """
    model = cobra.Model("toy")
    mets = {}
    for name, compartment, charge, formula in [
        ("H+", "c", 1, "H"),
        ("H+", "e", 1, "H"),
        ("S", "c", -1, "C6H11O9P"),
        ("S", "e", -1, "C6H11O9P"),
        ("A", "c", 1, "C5H14NO"),
        ("A", "e", 1, "C5H14NO"),
        ("B", "c", 0, "C6H12O6"),
        ("B", "e", 0, "C6H12O6"),
    ]:
        met = cobra.Metabolite(
            f"{name}_{compartment}", formula=formula, name=name, compartment=compartment
        )
        met.charge = charge
        mets[(name, compartment)] = met
    model.add_metabolites(list(mets.values()))

    symport = cobra.Reaction("symport")
    symport.add_metabolites(
        {
            mets[("H+", "e")]: -1.0,
            mets[("S", "e")]: -1.0,
            mets[("H+", "c")]: 1.0,
            mets[("S", "c")]: 1.0,
        }
    )
    antiport = cobra.Reaction("antiport")
    antiport.add_metabolites(
        {
            mets[("A", "c")]: -1.0,
            mets[("B", "e")]: -1.0,
            mets[("A", "e")]: 1.0,
            mets[("B", "c")]: 1.0,
        }
    )
    internal = cobra.Reaction("internal")
    internal.add_metabolites({mets[("S", "c")]: -1.0, mets[("A", "c")]: 1.0})
    model.add_reactions([symport, antiport, internal])
    return model


def test_symport_moves_one_proton_and_no_net_charge():
    """N_H, q, the membrane crossed, and both energy terms, all by hand.

    RT ln(10) at 303.15 K is 5.8037 kJ/mol, so carrying one proton from pH 5.0 into pH 7.2
    releases -5.8037 x 2.2 = -12.768 kJ/mol. The proton's +1 and the substrate's -1 cross
    together, so q is zero and the 60 mV term contributes nothing.
    """
    term = reaction_transport_term(
        _toy_model().reactions.get_by_id("symport"), _toy_conditions()
    )
    assert term.crosses_membrane
    assert term.n_proton == pytest.approx(1.0)
    assert term.charge_moved == pytest.approx(0.0)
    assert term.compartment_pairs == [("e", "c")]
    rt_ln10 = GAS_CONSTANT_KJ * 303.15 * math.log(10.0)
    assert term.delta_g_ph_kj_per_mol == pytest.approx(-rt_ln10 * 2.2, abs=1e-9)
    assert term.delta_g_ph_kj_per_mol == pytest.approx(-12.7681, abs=1e-3)
    assert term.delta_g_electrical_kj_per_mol == pytest.approx(0.0, abs=1e-12)
    assert term.delta_g_transport_kj_per_mol == pytest.approx(-12.7681, abs=1e-3)


def test_antiport_moves_charge_and_no_proton():
    """A cation leaving the cytosol for the more positive medium costs +5.789 kJ/mol.

    The cytosol sits 60 mV below the medium, so Phi_in - Phi_out is +0.060 V and the work
    is +1 x 96.485 x 0.060. The neutral counter-substrate contributes nothing, and no
    proton crosses, so the pH term is exactly zero even across a 2.2-unit gradient.
    """
    term = reaction_transport_term(
        _toy_model().reactions.get_by_id("antiport"), _toy_conditions()
    )
    assert term.n_proton == pytest.approx(0.0)
    assert term.delta_g_ph_kj_per_mol == pytest.approx(0.0, abs=1e-12)
    assert term.delta_g_electrical_kj_per_mol == pytest.approx(
        FARADAY_KJ_PER_MOL_V * 0.060, abs=1e-9
    )
    assert term.delta_g_electrical_kj_per_mol == pytest.approx(5.7891, abs=1e-3)
    assert sorted(term.compartment_pairs) == [("c", "e"), ("e", "c")]


def test_single_compartment_reaction_has_no_transport_term():
    """A reaction inside one compartment must produce nothing, not a small number."""
    term = reaction_transport_term(
        _toy_model().reactions.get_by_id("internal"), _toy_conditions()
    )
    assert not term.crosses_membrane
    assert term.species == []
    assert term.delta_g_transport_kj_per_mol == 0.0


def test_reversing_a_reaction_flips_the_sign():
    """Direction is read off the stoichiometry, so writing the reaction backwards negates
    the term. A module that returned a magnitude would pass every other test here.
    """
    model = _toy_model()
    forward = reaction_transport_term(
        model.reactions.get_by_id("symport"), _toy_conditions()
    )
    reverse = cobra.Reaction("reverse")
    reverse.add_metabolites(
        {m: -c for m, c in model.reactions.get_by_id("symport").metabolites.items()}
    )
    model.add_reactions([reverse])
    backward = reaction_transport_term(reverse, _toy_conditions())
    assert backward.delta_g_transport_kj_per_mol == pytest.approx(
        -forward.delta_g_transport_kj_per_mol, abs=1e-9
    )
    # N_H counts moles moved and stays positive; the DIRECTION is in the pair, and that
    # is what carries the sign into the pH term.
    assert backward.n_proton == pytest.approx(1.0)
    assert backward.compartment_pairs == [("c", "e")]


def test_same_compartment_copies_cancel_before_pairing():
    """A chemical consumed and produced in the cytosol as well as exported must not be
    routed across the membrane by the pairing. Cancelling the matching compartment first
    is what prevents an invented transport event.
    """
    model = _toy_model()
    mets = {(m.name, m.compartment): m for m in model.metabolites}
    tricky = cobra.Reaction("tricky")
    # Two S consumed in the cytosol, one returned to the cytosol, one sent outside: the
    # honest reading is that exactly one S crosses.
    tricky.add_metabolites(
        {
            mets[("S", "c")]: -2.0,
            mets[("A", "c")]: -1.0,
            mets[("S", "e")]: 1.0,
            mets[("B", "c")]: 1.0,
        }
    )
    model.add_reactions([tricky])
    term = reaction_transport_term(tricky, _toy_conditions())
    assert [
        (s.name, s.compartment_out, s.compartment_in, s.moles) for s in term.species
    ] == [("S", "c", "e", 1.0)]


def test_table_round_trips_and_aligns_to_reaction_order():
    """The flux layer reads this table off disk, so serialization has to be lossless and
    the alignment has to be by reaction id rather than by position.
    """
    table = build_transport_table(_toy_model(), _toy_conditions())
    assert table.n_reactions == 3
    assert table.n_crossing == 2
    assert table.n_nonzero == 2
    values, mask = table.as_tensor(["internal", "absent", "symport"])
    assert mask.tolist() == [True, False, True]
    assert float(values[0]) == 0.0
    assert float(values[2]) == pytest.approx(-12.7681, abs=1e-3)
    restored = TransportTable.model_validate_json(table.model_dump_json())
    assert restored.terms[0].delta_g_transport_kj_per_mol == pytest.approx(
        table.terms[0].delta_g_transport_kj_per_mol
    )


def test_untabulated_compartment_falls_back_to_the_reference_pH():
    """A compartment with no entry takes the reference compartment's pH and zero
    potential, which is stated behavior rather than an error, because yeast-GEM carries
    membrane pseudo-compartments that no measurement covers.
    """
    conditions = _toy_conditions()
    assert conditions.p_h_of("vm") == conditions.p_h["c"]
    assert conditions.potential("vm") == 0.0


def _read_shipped(path: str) -> dict[str, float]:
    """The GEM's reaction column, with both of its missing-value conventions rejected."""
    values: dict[str, float] = {}
    with open(path) as handle:
        next(handle)
        for line in handle:
            key, _, raw = line.strip().partition(",")
            try:
                value = float(raw)
            except ValueError:
                continue
            if abs(value) >= 9.0e6 or math.isnan(value):
                continue
            values[key] = value
    return values


@pytest.mark.skipif(
    not osp.exists(GEM_XML) or not osp.exists(GEM_RXN_DELTA_G),
    reason="yeast-GEM 9.0.2 is not in DATA_ROOT",
)
def test_computed_terms_track_the_shipped_column_on_yeast_gem():
    """The measured agreement, asserted as a floor rather than as a target.

    On yeast-GEM 9.0.2 the module finds 1,468 reactions crossing a compartment boundary
    and gives 326 of them a nonzero term. Against the shipped reaction column converted to
    kJ/mol, on the 287 reactions nonzero in both, the agreement is: sign 0.847, Pearson
    0.804, median absolute difference 7.04 kJ/mol. The thresholds below sit just outside
    those, so a change in the physics fails the test while ordinary drift does not.

    The 613 reactions with a nonzero shipped value and a zero computed term are the
    expected shortfall, not a bug: they move an uncharged, non-proton reactant whose
    average protonation differs between the two compartments, which is the part of N_H
    this module does not yet compute.
    """
    model = cobra.io.read_sbml_model(GEM_XML)
    table = build_transport_table(model, pipeline_conditions())
    assert table.n_crossing > 1000
    stats = compare_to_shipped(table, _read_shipped(GEM_RXN_DELTA_G), scale=KJ_PER_KCAL)
    assert stats["n_nonzero_in_both"] > 250
    assert stats["sign_agreement"] > 0.80
    assert stats["pearson"] > 0.75
    assert stats["median_abs_difference_kj"] < 10.0


@pytest.mark.skipif(
    not osp.exists(GEM_XML), reason="yeast-GEM 9.0.2 is not in DATA_ROOT"
)
def test_atp_synthase_pumps_two_protons_into_the_matrix():
    """yeast-GEM writes mitochondrial ATP synthase as 3 H+[c] consumed against 2 H+[m]
    produced: one proton is chemistry and two translocate. Getting the pairing wrong here
    would report three, or zero.
    """
    model = cobra.io.read_sbml_model(GEM_XML)
    term = reaction_transport_term(
        model.reactions.get_by_id("r_0226"), pipeline_conditions()
    )
    assert term.n_proton == pytest.approx(2.0)
    assert term.compartment_pairs == [("c", "m")]
    # Two protons carried up a 0.6-unit pH gradient and into a 160 mV negative matrix.
    rt_ln10 = GAS_CONSTANT_KJ * 303.15 * math.log(10.0)
    assert term.delta_g_ph_kj_per_mol == pytest.approx(-rt_ln10 * 2 * 0.6, abs=1e-6)
    assert term.delta_g_electrical_kj_per_mol == pytest.approx(
        2 * FARADAY_KJ_PER_MOL_V * (-0.160), abs=1e-6
    )
    assert term.delta_g_transport_kj_per_mol < 0.0


def test_flux_layer_transport_source_switch():
    """The three routes are distinguishable, and 'computed' refuses to guess a path."""
    from tests.torchcell.metabolism.test_flux_layer import _toy_gem
    from torchcell.metabolism.flux_layer import FluxLayer, FluxLayerConfig

    gem = _toy_gem()
    common: dict[str, Any] = {"hidden_dim": 8, "reaction_embed_dim": 4}

    off = FluxLayer(gem, ["g0", "g1", "g2"], config=FluxLayerConfig(**common))
    assert torch.all(off.delta_r_g_transport == 0.0)
    assert off.n_transport_terms_sourced == 0

    with pytest.raises(ValueError, match="transport_terms_path"):
        FluxLayer(
            gem,
            ["g0", "g1", "g2"],
            config=FluxLayerConfig(transport_delta_g_source="computed", **common),
        )

    with pytest.raises(ValueError, match="not one of"):
        FluxLayer(
            gem,
            ["g0", "g1", "g2"],
            config=FluxLayerConfig(transport_delta_g_source="shipped_2", **common),
        )


def test_flux_layer_reads_a_computed_table(tmp_path):
    """A computed table lands on the right reactions and nowhere else."""
    from tests.torchcell.metabolism.test_flux_layer import _toy_gem
    from torchcell.metabolism.flux_layer import FluxLayer, FluxLayerConfig

    gem = _toy_gem()
    table = build_transport_table(_toy_model(), _toy_conditions())
    # Rename one term onto a reaction the toy GEM actually has, so the alignment is
    # exercised rather than trivially empty.
    table.terms[0].reaction_id = "r2"
    path = str(tmp_path / "transport_terms.json")
    table.write(path)

    layer = FluxLayer(
        gem,
        ["g0", "g1", "g2"],
        config=FluxLayerConfig(
            hidden_dim=8,
            reaction_embed_dim=4,
            transport_delta_g_source="computed",
            transport_terms_path=path,
        ),
    )
    assert float(layer.delta_r_g_transport[2]) == pytest.approx(-12.7681, abs=1e-3)
    assert float(layer.delta_r_g_transport[0]) == 0.0
    assert layer.n_transport_terms_sourced == 1
