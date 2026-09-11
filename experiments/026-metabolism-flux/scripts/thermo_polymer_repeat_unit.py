# experiments/026-metabolism-flux/scripts/thermo_polymer_repeat_unit.py
# [[experiments.026-metabolism-flux.scripts.thermo_polymer_repeat_unit]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/thermo_polymer_repeat_unit.py

r"""Price a polymer's REPEAT UNIT, but only after checking the GEM means the repeat unit.

yeast-GEM carries chitin, (1->3)-beta-D-glucan, mannan, pectin, starch and dextrin as
species whose formula is a repeat unit rather than a molecule: chitin is ``C8H13NO5``, one
N-acetylglucosamine residue, not a chain. A structure cannot be written for a chain of
unspecified length, which is why these sat in ``no_structure_available``. A residue is a
different matter, and this script prices it.

**The precondition is checked, not assumed, and the script refuses without it.** A repeat-unit
surrogate is only meaningful if the model's STOICHIOMETRY is written per repeat unit. If the
coefficients were per chain, a residue energy would be wrong by a factor of the chain length.
The check is the model's own mass balance: every reaction carrying the species must balance
with the species counted as one residue. Measured on yeast-GEM 9.0.2, they do, including
``r_0271`` (chitin deacetylase), ``r_0005`` (1,3-beta-glucan synthase) and ``r_4331``
(glycanohydrolase); the only imbalances are the exchange reactions, which are unbalanced by
construction, and the carbohydrate pseudo-reaction. A species failing the check is reported
and left unpriced.

**Which monomer a residue belongs to is read off the model, not off the name.** The GEM's own
hydrolysis reaction says it: ``chitin + H2O <=> N-acetyl-D-glucosamine`` identifies the
monomer, and the formula arithmetic confirms it, since a residue is the monomer minus one
water. Where no hydrolysis reaction exists the fallback is stated per compound rather than
silent.

**The correction is MEASURED on this table's own energy zero, not assumed.** Setting the
residue to monomer minus water would say the glycosidic bond is worth nothing. It is not:
this table already prices eight glycoside hydrolyses over six distinct disaccharides, and
they give a mean hydrolysis energy with a real spread. Writing the disaccharide as two
residues plus one water,

.. math::

    G(D) = 2R + G(\mathrm{H_2O}), \quad
    \Delta_r G_\mathrm{hyd} = 2G(\mathrm{mono}) - G(D) - G(\mathrm{H_2O})

so

.. math::

    R = G(\mathrm{mono}) - G(\mathrm{H_2O}) - \tfrac{1}{2}\Delta_r G_\mathrm{hyd}

and the sigma the surrogate carries is the monomer's sigma and half the spread of the
calibration set, in quadrature. A species of ``n`` residues takes ``n`` times both, treating
the residues as fully correlated, which is the conservative choice.

These values are a SURROGATE and are labeled one: the route is
``ThermoRoute.REPEAT_UNIT_SURROGATE``, so a reader excludes every one of them with a single
filter on the route.

Writes ``results/thermo_polymer_repeat_unit.json``.

Run under the eQuilibrator environment, which carries cobra::

    /scratch/projects/torchcell-scratch/envs/equilibrator/bin/python \
        experiments/026-metabolism-flux/scripts/thermo_polymer_repeat_unit.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import re
from datetime import datetime, timezone

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
GEM_DIR = osp.join(DATA_ROOT, "data", "torchcell", "yeast-GEM", "yeast-GEM-9.0.2")
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
PIPELINE_JSON = osp.join(RESULTS, "thermo_pipeline_yeast9.json")
OUT_PATH = osp.join(RESULTS, "thermo_polymer_repeat_unit.json")

#: Reactions that are unbalanced by construction and so cannot test the stoichiometry
#: convention: an exchange crosses the system boundary and a biomass pseudo-reaction has
#: fractional coefficients in grams rather than moles.
def _is_boundary(reaction) -> bool:
    """True for an exchange, sink or pseudo-reaction, which cannot test mass balance."""
    return (
        len(reaction.metabolites) == 1
        or "pseudoreaction" in (reaction.name or "").lower()
        or "exchange" in (reaction.name or "").lower()
    )


_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def composition(formula: str) -> dict[str, int]:
    """Elemental composition of a formula string."""
    out: dict[str, int] = {}
    for element, count in _TOKEN.findall(formula or ""):
        if element:
            out[element] = out.get(element, 0) + (int(count) if count else 1)
    return out


def _minus_water(comp: dict[str, int]) -> dict[str, int]:
    out = dict(comp)
    out["H"] = out.get("H", 0) - 2
    out["O"] = out.get("O", 0) - 1
    return {k: v for k, v in out.items() if v}


class Calibration(BaseModel):
    """The glycosidic hydrolysis energy, measured on this table's own priced values."""

    n_reactions: int
    n_distinct_substrates: int
    reactions: list[dict[str, object]]
    mean_kj_per_mol: float
    sd_kj_per_mol: float
    min_kj_per_mol: float
    max_kj_per_mol: float
    per_residue_correction_kj_per_mol: float
    per_residue_sigma_kj_per_mol: float
    rule: str = (
        "a disaccharide is two residues plus one water, so R = G(monomer) - G(H2O) - "
        "dG_hyd/2; the sigma the surrogate carries is half the SPREAD of the calibration "
        "set, in quadrature with the monomer's own sigma"
    )


class PolymerRecord(BaseModel):
    """One repeat-unit species: the checks, the monomer, and the surrogate."""

    met_id: str
    met_ids: list[str] = Field(default_factory=list)
    name: str
    formula: str
    charge: int
    n_residues: int | None = None
    stoichiometry_is_per_repeat_unit: bool = False
    balance_checked_reactions: list[str] = Field(default_factory=list)
    unbalanced_reactions: list[str] = Field(default_factory=list)
    monomer_met_id: str | None = None
    monomer_name: str | None = None
    monomer_source: str = ""
    monomer_dgf_prime_kj_per_mol: float | None = None
    water_dgf_prime_kj_per_mol: float | None = None
    dgf_prime_kj_per_mol: float | None = None
    sigma_kj_per_mol: float | None = None
    outcome: str = ""
    detail: str = ""


class PolymerReport(BaseModel):
    """What the repeat-unit surrogate priced, and on what evidence."""

    built_at: str
    route: str = "repeat_unit_surrogate"
    precondition: str = (
        "a repeat-unit surrogate is only meaningful where the GEM's stoichiometry is per "
        "repeat unit; every species here is mass-balance checked against its own "
        "non-boundary reactions and a species that fails is reported unpriced"
    )
    calibration: Calibration
    n_candidates: int
    n_priced: int
    n_metabolite_rows_priced: int
    outcomes: dict[str, int] = Field(default_factory=dict)
    records: list[PolymerRecord] = Field(default_factory=list)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


def _calibrate(model, priced: dict[str, float]) -> Calibration:
    """Every fully priced glycoside hydrolysis to free sugars in this model."""
    water_ids = {m.id for m in model.metabolites if m.formula == "H2O"}
    rows: list[dict[str, object]] = []
    for reaction in model.reactions:
        mets = reaction.metabolites
        substrates = [m for m, c in mets.items() if c < 0]
        waters = [m for m in substrates if m.id in water_ids]
        if len(waters) != 1 or len(substrates) != 2:
            continue
        if any(m.id not in priced for m in mets):
            continue
        if reaction.check_mass_balance():
            continue
        other = next(m for m in substrates if m.id not in water_ids)
        substrate = composition(other.formula)
        if set(substrate) != {"C", "H", "O"} or substrate.get("C", 0) < 12:
            continue
        # Every product has to be a free sugar CnH2nOn, which is what makes the substrate a
        # glycoside rather than an ester or an amide.
        products = [(m, c) for m, c in mets.items() if c > 0]
        if not products:
            continue
        if not all(
            set(composition(m.formula)) == {"C", "H", "O"}
            and composition(m.formula).get("H") == 2 * composition(m.formula).get("C", 0)
            and composition(m.formula).get("O") == composition(m.formula).get("C", 0)
            for m, _ in products
        ):
            continue
        if sum(c * composition(m.formula).get("C", 0) for m, c in products) != substrate["C"]:
            continue
        drg = sum(c * priced[m.id] for m, c in mets.items())
        rows.append(
            {
                "reaction_id": reaction.id,
                "substrate": other.name,
                "substrate_formula": other.formula,
                "products": [m.name for m, _ in products],
                "delta_r_g_prime_kj_per_mol": float(drg),
            }
        )
    values = np.array([float(r["delta_r_g_prime_kj_per_mol"]) for r in rows])
    by_substrate: dict[str, list[float]] = {}
    for row in rows:
        by_substrate.setdefault(str(row["substrate"]), []).append(
            float(row["delta_r_g_prime_kj_per_mol"])
        )
    per_substrate = np.array([float(np.mean(v)) for v in by_substrate.values()])
    if per_substrate.size < 2:
        raise RuntimeError(
            "fewer than two distinct calibration substrates; the correction would carry no "
            "measured spread and the surrogate would be a guess"
        )
    return Calibration(
        n_reactions=len(rows),
        n_distinct_substrates=len(by_substrate),
        reactions=rows,
        mean_kj_per_mol=float(per_substrate.mean()),
        sd_kj_per_mol=float(per_substrate.std(ddof=1)),
        min_kj_per_mol=float(values.min()),
        max_kj_per_mol=float(values.max()),
        per_residue_correction_kj_per_mol=float(-per_substrate.mean() / 2.0),
        per_residue_sigma_kj_per_mol=float(per_substrate.std(ddof=1) / 2.0),
    )


def main() -> None:
    import cobra

    model = cobra.io.read_sbml_model(osp.join(GEM_DIR, "model", "yeast-GEM.xml"))
    with open(PIPELINE_JSON) as handle:
        pipeline = json.load(handle)
    priced = {
        row["met_id"]: row["dgf_prime_kj_per_mol"]
        for row in pipeline["metabolites"]
        if row["dgf_prime_kj_per_mol"] is not None
    }
    sigma_of = {
        row["met_id"]: row["sigma_kj_per_mol"]
        for row in pipeline["metabolites"]
        if row.get("sigma_kj_per_mol") is not None
    }
    # The candidate list comes from the eQuilibrator RECOMPUTATION, a fixed upstream
    # artifact this script never writes, and NOT from the pipeline's current unpriced list.
    # This file is itself a pipeline stage, so asking the pipeline what is unpriced would
    # drop every species it had priced on the previous run and the run after would unprice
    # them again. A candidate another stage already prices never reaches this one, because
    # the stage sits late in the precedence order, so the wider list costs nothing.
    import pandas as pd

    recomputation = pd.read_parquet(
        osp.join(
            DATA_ROOT, "data", "torchcell", "thermo_equilibrator",
            "metabolite_dgf_prime.parquet",
        )
    )
    unpriced = {
        str(row.met_id): "unpriced_by_the_recomputation"
        for row in recomputation.itertuples()
        if not np.isfinite(row.dgf_prime_kj_per_mol)
    }
    calibration = _calibrate(model, priced)
    print(
        f"calibration: {calibration.n_reactions} reactions over "
        f"{calibration.n_distinct_substrates} substrates, mean "
        f"{calibration.mean_kj_per_mol:.3f} sd {calibration.sd_kj_per_mol:.3f} kJ/mol "
        f"-> per-residue correction {calibration.per_residue_correction_kj_per_mol:+.3f} "
        f"+/- {calibration.per_residue_sigma_kj_per_mol:.3f}"
    )

    water = next(m for m in model.metabolites if m.formula == "H2O" and m.id in priced)
    g_water = priced[water.id]
    water_ids = {m.id for m in model.metabolites if m.formula == "H2O"}

    # A candidate is an unpriced species whose composition is a whole number of residues,
    # i.e. n * (some monomer - H2O). The monomer is identified from the model's own
    # hydrolysis reaction where one exists.
    records: list[PolymerRecord] = []
    by_name: dict[str, PolymerRecord] = {}
    for met_id in sorted(unpriced):
        met = model.metabolites.get_by_id(met_id)
        comp = composition(met.formula or "")
        if set(comp) - {"C", "H", "O", "N"} or "C" not in comp:
            continue
        record = PolymerRecord(
            met_id=met_id,
            met_ids=[met_id],
            name=met.name,
            formula=met.formula or "",
            charge=int(met.charge or 0),
        )
        # 1. the stoichiometry check
        checked, bad = [], []
        for reaction in met.reactions:
            if _is_boundary(reaction):
                continue
            checked.append(reaction.id)
            if reaction.check_mass_balance():
                bad.append(reaction.id)
        record.balance_checked_reactions = checked
        record.unbalanced_reactions = bad
        record.stoichiometry_is_per_repeat_unit = bool(checked) and not bad
        if not checked:
            continue

        # 2. the monomer, from a hydrolysis reaction of this species
        monomer = None
        for reaction in met.reactions:
            mets = reaction.metabolites
            if mets.get(met, 0) >= 0:
                continue
            waters = [m for m in mets if m.id in water_ids and mets[m] < 0]
            substrates = [m for m, c in mets.items() if c < 0]
            products = [m for m, c in mets.items() if c > 0]
            if not waters or len(substrates) != 2 or len(products) != 1:
                continue
            candidate = products[0]
            if candidate.id not in priced:
                continue
            n_waters = int(-mets[waters[0]])
            n_products = int(mets[candidate])
            if composition(candidate.formula) != _minus_water(
                composition(candidate.formula)
            ) and _minus_water(composition(candidate.formula)) and n_products:
                pass
            # the residue arithmetic: n residues + n waters -> n monomers
            residue = _minus_water(composition(candidate.formula))
            scaled = {k: v * n_products for k, v in residue.items()}
            if scaled == comp and n_waters == n_products:
                monomer = candidate
                record.n_residues = n_products
                record.monomer_source = f"hydrolysis {reaction.id}"
                break
            # a disaccharide-shaped species: n residues plus one water
            plus_water = dict(scaled)
            plus_water["H"] = plus_water.get("H", 0) + 2
            plus_water["O"] = plus_water.get("O", 0) + 1
            if plus_water == comp and n_waters == n_products - 1:
                monomer = candidate
                record.n_residues = n_products
                record.monomer_source = (
                    f"hydrolysis {reaction.id}; the species is n residues plus one water"
                )
                break
        if monomer is None:
            record.outcome = "no_hydrolysis_reaction_identifies_a_monomer"
            record.detail = (
                "the model carries no reaction of the form species + n H2O -> n monomer "
                "with the monomer priced, so which residue this is cannot be read off the "
                "stoichiometry and no surrogate is written"
            )
            records.append(record)
            continue
        if not record.stoichiometry_is_per_repeat_unit:
            record.outcome = "stoichiometry_not_per_repeat_unit"
            record.detail = (
                "mass balance fails on "
                + ", ".join(bad)
                + "; a residue energy would be wrong by the chain length"
            )
            records.append(record)
            continue

        n = record.n_residues or 1
        residue_energy = (
            priced[monomer.id] - g_water + calibration.per_residue_correction_kj_per_mol
        )
        record.monomer_met_id = monomer.id
        record.monomer_name = monomer.name
        record.monomer_dgf_prime_kj_per_mol = priced[monomer.id]
        record.water_dgf_prime_kj_per_mol = g_water
        extra_water = "plus one water" in record.monomer_source
        record.dgf_prime_kj_per_mol = n * residue_energy + (g_water if extra_water else 0.0)
        monomer_sigma = sigma_of.get(monomer.id, 0.0)
        record.sigma_kj_per_mol = float(
            n
            * np.sqrt(
                monomer_sigma**2
                + sigma_of.get(water.id, 0.0) ** 2
                + calibration.per_residue_sigma_kj_per_mol**2
            )
        )
        record.outcome = "repeat_unit_surrogate"
        record.detail = (
            f"{n} x [G({monomer.name}) - G(H2O) "
            f"{calibration.per_residue_correction_kj_per_mol:+.3f}]"
            + (" + G(H2O)" if extra_water else "")
        )
        records.append(record)
        by_name[met.name] = record
        print(
            f"  {met_id} {met.name[:30]:30s} n={n} -> {record.dgf_prime_kj_per_mol:9.2f}"
            f" +/- {record.sigma_kj_per_mol:.2f} via {monomer.name}"
        )

    # Every other compartment copy of a priced chemical is the same species.
    for met in model.metabolites:
        if met.id in unpriced and met.name in by_name:
            record = by_name[met.name]
            if met.id not in record.met_ids:
                record.met_ids.append(met.id)

    priced_records = [r for r in records if r.outcome == "repeat_unit_surrogate"]
    outcomes: dict[str, int] = {}
    for record in records:
        outcomes[record.outcome] = outcomes.get(record.outcome, 0) + 1
    report = PolymerReport(
        built_at=_timestamp(),
        calibration=calibration,
        n_candidates=len(records),
        n_priced=len(priced_records),
        n_metabolite_rows_priced=sum(len(r.met_ids) for r in priced_records),
        outcomes=dict(sorted(outcomes.items(), key=lambda kv: -kv[1])),
        records=records,
    )
    with open(OUT_PATH, "w") as handle:
        handle.write(report.model_dump_json(indent=1))
    print(f"wrote {OUT_PATH}")
    print(
        f"  {report.n_priced} species over {report.n_metabolite_rows_priced} metabolite "
        f"rows: {report.outcomes}"
    )


if __name__ == "__main__":
    main()
