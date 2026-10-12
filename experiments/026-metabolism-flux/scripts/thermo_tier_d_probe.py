# experiments/026-metabolism-flux/scripts/thermo_tier_d_probe.py
# [[experiments.026-metabolism-flux.scripts.thermo_tier_d_probe]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/thermo_tier_d_probe.py

r"""Probe every metabolite the group decomposer refuses, and price only where it is defensible.

Tier d is the class where a STRUCTURE exists and eQuilibrator's group decomposition refuses
it, so a pKa predictor and a structure writer both change nothing. The staged pipeline still
leaves 30 such species unpriced, and they block 29 reactions. Three questions decide whether
that is closable, and this script answers each per compound rather than for the class.

**1. Does the reactant-contribution path reach it anyway?** Component contribution is a sum
of a reactant-contribution term, which needs the compound to appear in the training
reactions, and a group-contribution term, which needs the decomposition. A compound with no
group vector can still carry a reactant-contribution estimate. ``standard_dg_formation`` is
called on the cache compound and its answer is recorded, together with the sigma
eQuilibrator returns, since a reactant-contribution-only estimate on a compound seen in one
reaction is not the same object as a fully decomposed one.

**2. Is it a monatomic ion?** Potassium, sodium, chloride, the divalent metals: these have
no group decomposition because they are not organic molecules, and no amount of structure
work will give them one. Their standard formation energies ARE tabulated, in the
thermochemical literature rather than in a group-contribution model, so the remedy is a
curation request against a named table and not a computation.

**3. Would a composition surrogate be defensible?** ``improve_parameter_coverage.py``
already measured, per composition, how far the cache compounds sharing that heavy-atom
composition disagree. The spread is reported here per compound and the admission bound is
stated, so "we could have borrowed a number" is a measurement rather than an opinion.

Nothing is priced on the strength of a guess: a compound is priced only when
``standard_dg_formation`` returns a value, and every other row carries the reason it does
not, plus the number of reactions it blocks.

Writes ``results/thermo_tier_d_probe.json``.

Run under the eQuilibrator environment::

    /scratch/projects/torchcell-scratch/envs/equilibrator/bin/python \
        experiments/026-metabolism-flux/scripts/thermo_tier_d_probe.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
THERMO_DIR = osp.join(DATA_ROOT, "data", "torchcell", "thermo_equilibrator")
PIPELINE_JSON = osp.join(RESULTS, "thermo_pipeline_yeast9.json")
AUDIT_CSV = osp.join(RESULTS, "unresolved_metabolites.csv")
OUT_PATH = osp.join(RESULTS, "thermo_tier_d_probe.json")

P_H = 7.2
IONIC_STRENGTH_M = 0.25
TEMPERATURE_K = 303.15
P_MG = 3.0

#: Spread within a composition below which borrowing an energy from a same-composition
#: compound is admitted. RT ln(10) at 303 K is 5.8 kJ/mol, so 5 is under one decade in the
#: equilibrium constant.
TIGHT_SPREAD_KJ = 5.0

#: The reason enum the staged pipeline assigns to this class.
TARGET_REASON = "group_decomposition_failed"


class TierDProbe(BaseModel):
    """One refused structure, and whether any route reaches it."""

    met_id: str
    name: str
    compartment: str
    formula: str
    charge: int
    n_reactions_blocked: int
    is_monatomic: bool
    cache_compound_id: int | None = None
    cache_has_structure: bool = False
    cache_has_group_vector: bool = False
    reactant_contribution_dgf_prime_kj_per_mol: float | None = None
    reactant_contribution_sigma_kj_per_mol: float | None = None
    surrogate_dgf_prime_kj_per_mol: float | None = None
    surrogate_spread_kj_per_mol: float | None = None
    surrogate_n: int | None = None
    surrogate_admissible: bool = False
    verdict: str = ""
    remedy: str = ""


class TierDReport(BaseModel):
    """What the probe found, and what it changes."""

    built_at: str
    conditions: dict[str, float]
    surrogate_admission_bound_kj_per_mol: float = TIGHT_SPREAD_KJ
    n_probed: int
    n_priced_by_reactant_contribution: int
    n_monatomic: int
    n_surrogate_admissible: int
    n_closed_permanently: int
    reactions_blocked_before: int
    reactions_that_would_close: int
    verdicts: dict[str, int]
    records: list[TierDProbe] = Field(default_factory=list)
    note: str


def main() -> None:
    from equilibrator_api import Q_, ComponentContribution
    from equilibrator_cache.models import Compound

    with open(PIPELINE_JSON) as handle:
        pipeline = json.load(handle)
    targets = {
        row["met_id"]
        for row in pipeline["metabolites"]
        if row.get("reason") == TARGET_REASON
    }
    blocked: dict[str, list[str]] = {}
    for reaction in pipeline["reactions"]:
        if reaction["delta_r_g0_kj_per_mol"] is not None:
            continue
        for met_id in reaction["blocking_met_ids"]:
            blocked.setdefault(met_id, []).append(reaction["reaction_id"])

    audit = pd.read_csv(AUDIT_CSV).set_index("met_id")
    surrogate = pd.read_parquet(
        osp.join(THERMO_DIR, "metabolite_dgf_surrogate.parquet")
    ).set_index("met_id")

    cc = ComponentContribution()
    cc.p_h = Q_(P_H)
    cc.ionic_strength = Q_(f"{IONIC_STRENGTH_M}M")
    cc.temperature = Q_(f"{TEMPERATURE_K}K")
    cc.p_mg = Q_(P_MG)
    session = cc.ccache.session

    records: list[TierDProbe] = []
    for met_id in sorted(targets):
        row = audit.loc[met_id] if met_id in audit.index else None
        formula = str(row["formula"]) if row is not None else ""
        name = str(row["name"]) if row is not None else ""
        # A monatomic species is one whose formula is a single element symbol with no
        # count: K, Na, Cl, Zn. Written as a test on the string rather than a hand list, so
        # another organism's model gets the same treatment.
        is_monatomic = bool(formula) and formula.isalpha() and len(formula) <= 2

        probe = TierDProbe(
            met_id=met_id,
            name=name,
            compartment=str(row["compartment"]) if row is not None else "",
            formula=formula,
            charge=int(row["charge"]) if row is not None and pd.notna(row["charge"]) else 0,
            n_reactions_blocked=len(blocked.get(met_id, [])),
            is_monatomic=is_monatomic,
        )
        compound_id = (
            int(row["baseline_compound_id"])
            if row is not None and pd.notna(row.get("baseline_compound_id"))
            else None
        )
        if compound_id is not None:
            compound = (
                session.query(Compound).filter(Compound.id == compound_id).one_or_none()
            )
            if compound is not None:
                probe.cache_compound_id = compound.id
                probe.cache_has_structure = bool(compound.smiles or compound.inchi_key)
                probe.cache_has_group_vector = compound.group_vector is not None
                mu, sigma, _ = cc.standard_dg_formation(compound)
                if mu is not None:
                    correction = compound.transform(
                        Q_(P_H),
                        Q_(f"{IONIC_STRENGTH_M}M"),
                        Q_(f"{TEMPERATURE_K}K"),
                        Q_(P_MG),
                    )
                    probe.reactant_contribution_dgf_prime_kj_per_mol = float(mu) + float(
                        correction.m_as("kJ/mol")
                    )
                    if sigma is not None:
                        import numpy as np

                        probe.reactant_contribution_sigma_kj_per_mol = float(
                            np.linalg.norm(np.asarray(sigma))
                        )

        if met_id in surrogate.index:
            entry = surrogate.loc[met_id]
            probe.surrogate_dgf_prime_kj_per_mol = float(
                entry["dgf_prime_surrogate_kj_per_mol"]
            )
            probe.surrogate_spread_kj_per_mol = float(entry["surrogate_spread_kj_per_mol"])
            probe.surrogate_n = int(entry["n_surrogates"])
            probe.surrogate_admissible = (
                probe.surrogate_spread_kj_per_mol <= TIGHT_SPREAD_KJ
            )

        if probe.reactant_contribution_dgf_prime_kj_per_mol is not None:
            probe.verdict = "priceable_by_reactant_contribution"
            probe.remedy = "already reachable; the staged pipeline should read this value"
        elif probe.surrogate_admissible:
            probe.verdict = "surrogate_admissible"
            probe.remedy = (
                "same-composition compounds agree within the bound, so a surrogate with "
                "that spread as its sigma is defensible"
            )
        elif probe.is_monatomic:
            probe.verdict = "monatomic_ion_needs_a_literature_value"
            probe.remedy = (
                "no group decomposition exists for a bare ion and none ever will; the "
                "standard formation energy is tabulated in the thermochemical literature, "
                "so this is a curation request against a named table"
            )
        else:
            probe.verdict = "no_defensible_route"
            probe.remedy = (
                "the decomposer refuses the structure, the reactant-contribution path does "
                "not reach it, and same-composition compounds disagree by more than the "
                "bound; nothing here is defensible without new data"
            )
        records.append(probe)
        print(
            f"  {met_id} {name[:38]:38s} {probe.verdict}"
            f" blocked {probe.n_reactions_blocked}",
            flush=True,
        )

    would_close = {
        reaction
        for probe in records
        if probe.verdict in ("priceable_by_reactant_contribution", "surrogate_admissible")
        for reaction in blocked.get(probe.met_id, [])
    }
    verdicts: dict[str, int] = {}
    for probe in records:
        verdicts[probe.verdict] = verdicts.get(probe.verdict, 0) + 1

    report = TierDReport(
        built_at=datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        conditions={
            "p_h": P_H,
            "ionic_strength_m": IONIC_STRENGTH_M,
            "temperature_k": TEMPERATURE_K,
            "p_mg": P_MG,
        },
        n_probed=len(records),
        n_priced_by_reactant_contribution=sum(
            1
            for p in records
            if p.verdict == "priceable_by_reactant_contribution"
        ),
        n_monatomic=sum(1 for p in records if p.is_monatomic),
        n_surrogate_admissible=sum(1 for p in records if p.surrogate_admissible),
        n_closed_permanently=sum(
            1
            for p in records
            if p.verdict
            in ("monatomic_ion_needs_a_literature_value", "no_defensible_route")
        ),
        reactions_blocked_before=len(
            {r for p in records for r in blocked.get(p.met_id, [])}
        ),
        reactions_that_would_close=len(would_close),
        verdicts=dict(sorted(verdicts.items(), key=lambda kv: -kv[1])),
        records=records,
        note=(
            "A compound is priced only where standard_dg_formation returns a value. A "
            "surrogate is reported as ADMISSIBLE rather than applied here: applying it is "
            "the staged pipeline's composition_surrogate stage, which uses the same bound."
        ),
    )
    with open(OUT_PATH, "w") as handle:
        handle.write(report.model_dump_json(indent=1))
    print(f"wrote {OUT_PATH}")
    print(f"  verdicts: {report.verdicts}")
    print(
        f"  {report.reactions_that_would_close} of "
        f"{report.reactions_blocked_before} blocked reactions would close"
    )


if __name__ == "__main__":
    main()
