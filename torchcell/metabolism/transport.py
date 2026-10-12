# torchcell/metabolism/transport.py
# [[torchcell.metabolism.transport]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/metabolism/transport.py

r"""Gibbs energy of transport, computed from the GEM rather than recovered from a table.

WHY THIS EXISTS
---------------
Summing formation energies over a transport reaction cancels exactly: the same chemical
appears on both sides, so :math:`\sum_i S_{ij}\Delta_f G'^\circ_i = 0` whenever every
compartment is priced at the same pH. A transport reaction's entire standard energy is
therefore its **transport term**, and a pipeline that only sums a metabolite column
asserts that moving a proton against a two-unit pH gradient is free.

Thermo-Flux Eq. (7) (Smith et al. 2026) gives the term as

.. math::
    \Delta_r G^t_j = -N_H R T \ln(10)\,\Delta\mathrm{pH} - F q\,\Delta\Phi

where :math:`N_H` is the number of protons moved across the membrane and :math:`q` the
charge moved. The paper does not fix which side each difference is taken from, so this
module fixes it explicitly and derives the signs rather than copying them:

* Species move from the compartment they appear in as a REACTANT (``out``) to the one they
  appear in as a PRODUCT (``in``), as the reaction is written.
* :math:`\Delta\mathrm{pH} = \mathrm{pH}_{\text{in}} - \mathrm{pH}_{\text{out}}`. The
  chemical potential of a proton in compartment X is :math:`\mu^\circ - RT\ln(10)\,
  \mathrm{pH}_X`, so moving :math:`N_H` of them costs
  :math:`-N_H RT\ln(10)(\mathrm{pH}_{\text{in}}-\mathrm{pH}_{\text{out}})`.
* :math:`\Delta\Phi = \Phi_{\text{out}} - \Phi_{\text{in}}`, which makes the electrical
  work :math:`+qF(\Phi_{\text{in}}-\Phi_{\text{out}})`: moving positive charge toward a
  more positive compartment costs energy. This is the sign matTFA implements in
  ``calcDGtpt_RHS``, where the tabulated potential of a compartment pair is
  :math:`\Phi_{\text{in}}-\Phi_{\text{out}}`.

Both terms are per unit reaction extent, in kJ/mol.

WHAT IS AND IS NOT MODELED
--------------------------
:math:`N_H` counts the **explicit H+ species** yeast-GEM carries per compartment, one per
compartment, and nothing else. Thermo-Flux's :math:`N_H` additionally absorbs the protons
released or bound when a reactant's average protonation differs between two compartments
at different pH, which needs the eQuilibrator species distribution for every transported
compound. That part is not computed here, and it is the term to add next; a reaction whose
transported species is uncharged and carries no explicit proton therefore gets exactly
zero from this module even where a small protonation contribution is real.
"""

from __future__ import annotations

import json
import math
import os.path as osp
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import torch
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover - typing only
    import cobra

#: Gas constant, kJ/(mol K).
GAS_CONSTANT_KJ = 8.314462618e-3

#: Faraday constant, kJ/(mol V). The membrane potentials below are in mV.
FARADAY_KJ_PER_MOL_V = 96.485

LN10 = math.log(10.0)

#: Amounts smaller than this are numerical dust from a stoichiometric coefficient, not a
#: transported species.
MOVED_TOLERANCE = 1e-9


class CompartmentElectrochemistry(BaseModel):
    """pH, ionic strength and electrical potential per compartment, with its source.

    ``potential_mv`` is signed and measured against ``reference_compartment``, which is
    the convention that makes a compartment pair's difference unambiguous. A per-pair
    magnitude table cannot say which side is negative, and getting that backwards flips
    the sign of every proton-pumping reaction in the model.
    """

    model_config = ConfigDict(extra="forbid")

    p_h: dict[str, float]
    ionic_strength_m: dict[str, float]
    potential_mv: dict[str, float] = Field(
        description=(
            "Electrical potential of each compartment MINUS that of the reference "
            "compartment, in mV. Cytosol 0 by construction."
        )
    )
    reference_compartment: str = "c"
    temperature_k: float = 303.15
    source: str

    def potential(self, compartment: str) -> float:
        """Potential of one compartment relative to the reference, mV; 0 if untabulated."""
        return self.potential_mv.get(compartment, 0.0)

    def p_h_of(self, compartment: str) -> float:
        """PH of one compartment; the reference compartment's pH if untabulated."""
        return self.p_h.get(compartment, self.p_h[self.reference_compartment])


def pipeline_conditions() -> CompartmentElectrochemistry:
    """The table the 026 pipeline runs on, assembled from where each number lives.

    pH is ``COMPARTMENT_PH`` from ``build_equilibrator_thermo.py``, which is textbook
    organellar pH and labeled ``assumed`` there rather than sourced. Ionic strength and
    the two membrane potentials are Thermo-Flux's iMM904 parameter set (Smith et al. 2026,
    Table 1), which is also what ``FluxLayerConfig``'s ``CompartmentParameters`` carries.
    The SIGNS are not in that table, which gives magnitudes: the cytosol is negative
    relative to the medium and the matrix is negative relative to the cytosol, so the
    medium sits at +60 mV and the matrix at -160 mV on a cytosol-zero scale.
    """
    p_h = {
        "c": 7.2,
        "e": 5.0,
        "m": 7.8,
        "n": 7.2,
        "p": 7.0,
        "er": 7.1,
        "g": 6.6,
        "v": 6.2,
        "ce": 5.0,
        "lp": 7.2,
        "erm": 7.1,
        "vm": 6.2,
        "gm": 6.6,
        "mm": 7.8,
    }
    return CompartmentElectrochemistry(
        p_h=p_h,
        ionic_strength_m={k: 0.25 for k in p_h},
        # The membrane pseudo-compartments take the potential of the aqueous phase they
        # face, so a reaction routed through one is not silently made electroneutral.
        potential_mv={"e": 60.0, "ce": 60.0, "m": -160.0, "mm": -160.0},
        reference_compartment="c",
        temperature_k=303.15,
        source=(
            "pH: COMPARTMENT_PH in experiments/026-metabolism-flux/scripts/"
            "build_equilibrator_thermo.py, textbook organellar values, assumed. "
            "Ionic strength and membrane potential magnitudes: Smith et al. 2026 "
            "Thermo-Flux, Table 1, iMM904 parameter set. Signs assigned here."
        ),
    )


def yetfl_conditions(path: str) -> CompartmentElectrochemistry:
    r"""The table yETFL used, read from its own ``yeast_compartment_data.json``.

    This is what yeast-GEM 9's shipped deltaG columns rest on, so it is the comparison
    partner whenever the question is why our numbers differ from the shipped ones. The
    potential is stored there per compartment PAIR; the cytosol-relative form is read off
    the cytosol's row, whose entry for compartment X is already
    :math:`\Phi_X - \Phi_{\mathrm{c}}`.
    """
    with open(path) as handle:
        raw = json.load(handle)
    cytosol = raw["c"]["membranePot"]
    return CompartmentElectrochemistry(
        p_h={k: float(v["pH"]) for k, v in raw.items()},
        ionic_strength_m={k: float(v["ionicStr"]) for k, v in raw.items()},
        potential_mv={k: float(v) for k, v in cytosol.items() if float(v) != 0.0},
        reference_compartment="c",
        temperature_k=298.15,
        source=(
            "yETFL (Oftadeh et al. 2021), info_yeast/thermo/yeast_compartment_data.json, "
            "Zenodo 10.5281/zenodo.4778047. Applied by matTFA's prepModelforTFA."
        ),
    )


class TransportedSpecies(BaseModel):
    """One chemical moving across one membrane in one reaction."""

    model_config = ConfigDict(extra="forbid")

    name: str
    met_id_out: str
    met_id_in: str
    compartment_out: str
    compartment_in: str
    moles: float = Field(description="Moles moved per unit reaction extent, positive.")
    charge: float
    is_proton: bool


class TransportTerm(BaseModel):
    r"""The transport free energy of one reaction, with everything it was built from."""

    model_config = ConfigDict(extra="forbid")

    reaction_id: str
    crosses_membrane: bool
    n_proton: float = Field(
        description="N_H, net explicit protons moved out -> in per unit extent."
    )
    charge_moved: float = Field(description="q, net charge moved out -> in.")
    compartment_pairs: list[tuple[str, str]] = Field(
        description="(out, in) for every membrane this reaction crosses."
    )
    delta_g_ph_kj_per_mol: float = Field(
        description="-N_H R T ln(10) (pH_in - pH_out), summed over moved protons."
    )
    delta_g_electrical_kj_per_mol: float = Field(
        description="+q F (Phi_in - Phi_out), summed over moved species."
    )
    delta_g_transport_kj_per_mol: float
    species: list[TransportedSpecies] = Field(default_factory=list)


class TransportTable(BaseModel):
    """Every reaction's transport term, plus the conditions that produced it."""

    model_config = ConfigDict(extra="forbid")

    terms: list[TransportTerm]
    conditions: CompartmentElectrochemistry
    model_id: str = ""
    n_reactions: int = 0
    n_crossing: int = 0
    n_nonzero: int = 0
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def as_tensor(self, rxn_ids: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Align the terms onto a reaction-index order.

        Returns ``(delta_g, mask)``, both ``[len(rxn_ids)]``. ``mask`` is True where this
        table has a term for that reaction, whether or not the term is zero, so a caller
        can tell "no value" from "computed zero".
        """
        by_id = {t.reaction_id: t.delta_g_transport_kj_per_mol for t in self.terms}
        values = torch.zeros(len(rxn_ids))
        mask = torch.zeros(len(rxn_ids), dtype=torch.bool)
        for index, rxn_id in enumerate(rxn_ids):
            if rxn_id in by_id:
                values[index] = by_id[rxn_id]
                mask[index] = True
        return values, mask

    def write(self, path: str) -> None:
        """Serialize to JSON; the flux layer reads this back."""
        with open(path, "w") as handle:
            handle.write(self.model_dump_json(indent=2))

    @classmethod
    def read(cls, path: str) -> TransportTable:
        """Load a table written by :meth:`write`."""
        with open(path) as handle:
            return cls.model_validate_json(handle.read())


def _pair_across_compartments(
    reactant_side: list[tuple[str, str, float]],
    product_side: list[tuple[str, str, float]],
) -> list[tuple[str, str, str, str, float]]:
    """Match one chemical's reactant copies to its product copies, largest first.

    Each entry is ``(met_id, compartment, amount)`` with ``amount > 0``. Copies in the
    SAME compartment on both sides are chemistry rather than transport, so they cancel
    first: pairing by size alone would otherwise route a cytosolic reactant to a
    mitochondrial product and invent a transport event that the reaction does not contain.

    Returns ``(met_id_out, met_id_in, compartment_out, compartment_in, moles)`` per moved
    portion.
    """
    left_id = {comp: mid for mid, comp, _ in reactant_side}
    right_id = {comp: mid for mid, comp, _ in product_side}
    left = {comp: amount for _, comp, amount in reactant_side}
    right = {comp: amount for _, comp, amount in product_side}
    for comp in set(left) & set(right):
        canceled = min(left[comp], right[comp])
        left[comp] -= canceled
        right[comp] -= canceled
    remaining_out = sorted(
        ((c, left_id[c], v) for c, v in left.items() if v > MOVED_TOLERANCE),
        key=lambda item: -item[2],
    )
    remaining_in = sorted(
        ((c, right_id[c], v) for c, v in right.items() if v > MOVED_TOLERANCE),
        key=lambda item: -item[2],
    )
    moved: list[tuple[str, str, str, str, float]] = []
    out_index = in_index = 0
    out_left = list(remaining_out)
    in_left = list(remaining_in)
    while out_index < len(out_left) and in_index < len(in_left):
        comp_out, mid_out, amount_out = out_left[out_index]
        comp_in, mid_in, amount_in = in_left[in_index]
        portion = min(amount_out, amount_in)
        moved.append((mid_out, mid_in, comp_out, comp_in, portion))
        out_left[out_index] = (comp_out, mid_out, amount_out - portion)
        in_left[in_index] = (comp_in, mid_in, amount_in - portion)
        if out_left[out_index][2] <= MOVED_TOLERANCE:
            out_index += 1
        if in_left[in_index][2] <= MOVED_TOLERANCE:
            in_index += 1
    return moved


def reaction_transport_term(
    reaction: cobra.Reaction, conditions: CompartmentElectrochemistry
) -> TransportTerm:
    r"""Compute :math:`\Delta_r G^t` for one reaction from its stoichiometry.

    A chemical is identified by its metabolite NAME, because yeast-GEM gives the same
    chemical a different ``s_XXXX`` id in every compartment; matching on the id would find
    no transported species at all.
    """
    by_chemical: dict[str, dict[str, list[tuple[str, str, float]]]] = defaultdict(
        lambda: {"reactant": [], "product": []}
    )
    charges: dict[str, float] = {}
    protons: set[str] = set()
    for metabolite, coefficient in reaction.metabolites.items():
        if abs(coefficient) <= MOVED_TOLERANCE:
            continue
        side = "reactant" if coefficient < 0 else "product"
        by_chemical[metabolite.name][side].append(
            (metabolite.id, metabolite.compartment, abs(coefficient))
        )
        charges[metabolite.name] = float(metabolite.charge or 0.0)
        if metabolite.formula == "H":
            protons.add(metabolite.name)

    rt_ln10 = GAS_CONSTANT_KJ * conditions.temperature_k * LN10
    species: list[TransportedSpecies] = []
    n_proton = 0.0
    charge_moved = 0.0
    delta_ph = 0.0
    delta_electrical = 0.0
    pairs: list[tuple[str, str]] = []
    for name, sides in by_chemical.items():
        for mid_out, mid_in, comp_out, comp_in, moles in _pair_across_compartments(
            sides["reactant"], sides["product"]
        ):
            if comp_out == comp_in:
                continue
            charge = charges[name]
            is_proton = name in protons
            species.append(
                TransportedSpecies(
                    name=name,
                    met_id_out=mid_out,
                    met_id_in=mid_in,
                    compartment_out=comp_out,
                    compartment_in=comp_in,
                    moles=moles,
                    charge=charge,
                    is_proton=is_proton,
                )
            )
            if (comp_out, comp_in) not in pairs:
                pairs.append((comp_out, comp_in))
            charge_moved += moles * charge
            potential_difference = (
                conditions.potential(comp_in) - conditions.potential(comp_out)
            ) / 1000.0
            delta_electrical += (
                moles * charge * FARADAY_KJ_PER_MOL_V * potential_difference
            )
            if is_proton:
                n_proton += moles
                delta_ph += (
                    -moles
                    * rt_ln10
                    * (conditions.p_h_of(comp_in) - conditions.p_h_of(comp_out))
                )

    return TransportTerm(
        reaction_id=reaction.id,
        crosses_membrane=len({m.compartment for m in reaction.metabolites}) > 1,
        n_proton=n_proton,
        charge_moved=charge_moved,
        compartment_pairs=pairs,
        delta_g_ph_kj_per_mol=delta_ph,
        delta_g_electrical_kj_per_mol=delta_electrical,
        delta_g_transport_kj_per_mol=delta_ph + delta_electrical,
        species=species,
    )


def build_transport_table(
    model: cobra.Model, conditions: CompartmentElectrochemistry | None = None
) -> TransportTable:
    """Compute the transport term for every reaction in a GEM."""
    conditions = conditions or pipeline_conditions()
    terms = [reaction_transport_term(r, conditions) for r in model.reactions]
    return TransportTable(
        terms=terms,
        conditions=conditions,
        model_id=model.id or "",
        n_reactions=len(terms),
        n_crossing=sum(1 for t in terms if t.crosses_membrane),
        n_nonzero=sum(
            1 for t in terms if abs(t.delta_g_transport_kj_per_mol) > MOVED_TOLERANCE
        ),
    )


def compare_to_shipped(
    table: TransportTable,
    shipped: dict[str, float],
    scale: float = 1.0,
    nonzero_tolerance: float = 1e-6,
) -> dict[str, float]:
    """How the computed terms line up with a GEM's own curated reaction column.

    ``scale`` converts the shipped column into kJ/mol. yeast-GEM 9's is in kcal/mol
    despite the paper's label, so the caller passes 4.184 to compare like with like;
    passing 1.0 reproduces the comparison as the document previously made it.

    Only reactions where BOTH are nonzero enter the agreement statistics: a pair of zeros
    agrees for a reason that has nothing to do with the physics, and pooling those in
    reports an agreement the comparison never tested. Only reactions that CROSS a
    compartment boundary are considered at all, since a single-compartment reaction has no
    transport term to compare and its shipped value is ordinary reaction chemistry.
    """
    computed = {t.reaction_id: t.delta_g_transport_kj_per_mol for t in table.terms}
    crossing = {t.reaction_id for t in table.terms if t.crosses_membrane}
    both_nonzero, ours_only, theirs_only = [], [], []
    for rxn_id, shipped_value in shipped.items():
        if rxn_id not in crossing:
            continue
        ours = computed[rxn_id]
        theirs = shipped_value * scale
        if abs(ours) > nonzero_tolerance and abs(theirs) > nonzero_tolerance:
            both_nonzero.append((rxn_id, ours, theirs))
        elif abs(ours) > nonzero_tolerance:
            ours_only.append(rxn_id)
        elif abs(theirs) > nonzero_tolerance:
            theirs_only.append(rxn_id)

    if not both_nonzero:
        raise ValueError("no reaction has a nonzero term in both tables")
    ours_values = torch.tensor([o for _, o, _ in both_nonzero])
    theirs_values = torch.tensor([t for _, _, t in both_nonzero])
    difference = (ours_values - theirs_values).abs()
    same_sign = torch.sign(ours_values) == torch.sign(theirs_values)
    return {
        "scale_applied_to_shipped": scale,
        "n_crossing_reactions_with_a_shipped_value": len(
            [r for r in shipped if r in crossing]
        ),
        "n_crossing_a_membrane": len(crossing),
        "n_nonzero_in_both": len(both_nonzero),
        "n_nonzero_only_computed": len(ours_only),
        "n_nonzero_only_shipped": len(theirs_only),
        "sign_agreement": float(same_sign.float().mean()),
        "median_abs_difference_kj": float(difference.median()),
        "mean_abs_difference_kj": float(difference.mean()),
        "p90_abs_difference_kj": float(difference.quantile(0.9)),
        "max_abs_difference_kj": float(difference.max()),
        "pearson": float(
            torch.corrcoef(torch.stack([ours_values, theirs_values]))[0, 1]
        ),
        "median_abs_computed_kj": float(ours_values.abs().median()),
        "median_abs_shipped_kj": float(theirs_values.abs().median()),
    }


def write_transport_table(
    model: cobra.Model,
    out_dir: str,
    conditions: CompartmentElectrochemistry | None = None,
    filename: str = "transport_terms.json",
) -> str:
    """Build and write the table; returns the path the flux layer should be pointed at."""
    table = build_transport_table(model, conditions)
    path = osp.join(out_dir, filename)
    table.write(path)
    return path
