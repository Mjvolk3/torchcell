# experiments/026-metabolism-flux/scripts/thermo_pipeline_yeast9.py
# [[experiments.026-metabolism-flux.scripts.thermo_pipeline_yeast9]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/thermo_pipeline_yeast9.py

r"""Run the organism-agnostic thermodynamic pipeline on yeast-GEM 9.0.2, end to end.

:mod:`torchcell.metabolism.thermo_pipeline` is the portable half: it takes a cobra model, a
compartment table and an ORDERED list of resolver stages, and returns a reaction
:math:`\Delta_r G'^\circ` table with a route per value and an
:class:`~torchcell.metabolism.thermo_pipeline.UnpricedReason` per gap. Nothing in it names
an organism, a compartment letter or a file.

This script is the yeast-specific half, and it is deliberately thin: it builds the stages
out of the artifacts the 026 experiment already produced and hands them over. Porting to
another organism means writing another file like this one, not touching the module.

THE STAGES, IN PRECEDENCE ORDER
--------------------------------
1. ``equilibrator_recomputation`` -- ``metabolite_dgf_prime.parquet``, one self-consistent
   component-contribution method at each compartment's own pH, and the only stage that
   carries a formation sigma.
2. ``structure_writer_lipids`` -- ``lipid_priced.json``, the lipids whose structure was
   composed from their name. Two routes inside it, kept apart: a written structure that the
   compound cache already held (``cache_lookup``, which brings ChemAxon pKas AND magnesium
   constants) and one created through the external-pKa bridge.
3. ``external_pka_targets`` -- ``external_pka_applied.json``, the tier-a compounds and the
   re-priced lipids from the selected pKa method under the pinned protonation convention.
   Absent until ``external_pka_apply_selection.py price`` has run; the stage is skipped with
   a printed line rather than silently.
4. ``composition_surrogate`` -- ``metabolite_dgf_surrogate.parquet``, an energy borrowed
   from cache compounds of the same heavy-atom composition. Admitted ONLY where those
   compounds agree to within ``--surrogate-spread``; the spread is carried as the sigma.
5. ``shipped_metabolite_column`` -- yeast-GEM's own table, in kcal/mol under a kJ/mol
   header, converted on read.

Plus a reaction-level last resort: the shipped REACTION column, with the 33 dGPredictor rows
identified by ``thermo_dgpredictor_rows.py`` exempted from the kcal conversion.

Outputs ``results/thermo_pipeline_yeast9.json`` (the full per-entity result),
``results/reaction_thermo_coverage_final.json`` (the coverage ceiling by route and reason)
and ``results/thermo_exclusions.json`` (one record per metabolite that cannot be priced,
with the reactions it blocks).

Run from the repository root under the torchcell env::

    ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/thermo_pipeline_yeast9.py
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from datetime import UTC, datetime
from typing import cast

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

from torchcell.metabolism.thermo_pipeline import (
    CompartmentConditions,
    IonFormationResolver,
    MetaboliteIdentity,
    MetabolitePrice,
    TableResolver,
    ThermoPipelineConfig,
    ThermoPipelineResult,
    ThermoRoute,
    UnpricedReason,
    identities_from_model,
    run_thermo_pipeline,
    sha256_of,
    summarize,
    write_result,
)

load_dotenv()

DATA_ROOT = cast(str, os.environ["DATA_ROOT"])
EXPERIMENT_ROOT = cast(str, os.environ["EXPERIMENT_ROOT"])
GEM_DIR = osp.join(DATA_ROOT, "data", "torchcell", "yeast-GEM", "yeast-GEM-9.0.2")
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
THERMO_DIR = osp.join(DATA_ROOT, "data", "torchcell", "thermo_equilibrator")

#: yeast-GEM's "no value" marker in the shipped deltaG CSVs.
SHIPPED_SENTINEL = 1.0e7

#: The compartment table the 026 build prices at, from
#: ``build_equilibrator_thermo.COMPARTMENT_PH``. Textbook organellar values; the note
#: ``[[experiments.026-metabolism-flux.enzyme-constrained-thermodynamic-flux-layer]]``
#: records that no primary yeast measurement of any of them is in the library.
COMPARTMENT_PH: dict[str, float] = {
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


class ParquetResolver:
    """A stage over a parquet with a ``met_id`` column and one value column."""

    def __init__(
        self,
        name: str,
        path: str,
        value_column: str,
        route: ThermoRoute,
        sigma_column: str | None = None,
        reason: UnpricedReason = UnpricedReason.NOT_ATTEMPTED,
        row_filter=None,
        tier_column: str | None = None,
    ) -> None:
        """Read the parquet once and index it by metabolite id."""
        self.name = name
        self.path = path
        self.route = route
        self.reason = reason
        frame = pd.read_parquet(path)
        if row_filter is not None:
            frame = frame[row_filter(frame)]
        frame = frame[np.isfinite(frame[value_column])]
        self.values = dict(zip(frame["met_id"].astype(str), frame[value_column]))
        self.sigma = (
            dict(zip(frame["met_id"].astype(str), frame[sigma_column]))
            if sigma_column and sigma_column in frame
            else {}
        )
        self.tier = (
            dict(zip(frame["met_id"].astype(str), frame[tier_column].astype(str)))
            if tier_column and tier_column in frame
            else {}
        )
        self.sha256 = sha256_of(path)

    def resolve(
        self,
        metabolites: list[MetaboliteIdentity],
        conditions: CompartmentConditions,
    ) -> list[MetabolitePrice]:
        """Price every metabolite this parquet has a finite row for."""
        out: list[MetabolitePrice] = []
        for met in metabolites:
            value = self.values.get(met.met_id)
            if value is None:
                out.append(MetabolitePrice(met_id=met.met_id, reason=self.reason))
                continue
            sigma = self.sigma.get(met.met_id)
            p_h, ionic, p_mg = conditions.for_compartment(met.compartment)
            out.append(
                MetabolitePrice(
                    met_id=met.met_id,
                    dgf_prime_kj_per_mol=float(value),
                    sigma_kj_per_mol=(
                        float(sigma) if sigma is not None and np.isfinite(sigma) else None
                    ),
                    route=self.route,
                    p_h=p_h,
                    ionic_strength_m=ionic,
                    p_mg=p_mg,
                    temperature_k=conditions.temperature_k,
                    detail=self.tier.get(met.met_id, ""),
                    provenance={"source_path": self.path, "sha256": self.sha256},
                )
            )
        return out


class JsonRecordResolver:
    """A stage over a JSON file of records that each price one or more metabolite ids.

    Used for the two structure-composition routes, whose output is a record per CHEMICAL
    (one written structure) that prices every compartment copy of it.
    """

    def __init__(
        self,
        name: str,
        path: str,
        records_key: str = "records",
        met_ids_key: str = "met_ids",
        value_key: str = "dgf_prime_kj_per_mol",
        sigma_key: str = "sigma_kj_per_mol",
        route_key: str = "route",
        route_map: dict[str, ThermoRoute] | None = None,
        default_route: ThermoRoute = ThermoRoute.STRUCTURE_WRITER,
        provenance_keys: tuple[str, ...] = (),
    ) -> None:
        """Read the JSON once and index its records by metabolite id."""
        self.name = name
        self.path = path
        self.sha256 = sha256_of(path)
        with open(path) as handle:
            payload = json.load(handle)
        self.rows: dict[str, dict] = {}
        for record in payload[records_key]:
            value = record.get(value_key)
            if value is None or not np.isfinite(value):
                continue
            route = (route_map or {}).get(str(record.get(route_key)), default_route)
            for met_id in record.get(met_ids_key, []):
                self.rows[str(met_id)] = {
                    "value": float(value),
                    "sigma": record.get(sigma_key),
                    "route": route,
                    "provenance": {
                        "source_path": path,
                        "sha256": self.sha256,
                        **{k: record.get(k) for k in provenance_keys},
                    },
                }

    def resolve(
        self,
        metabolites: list[MetaboliteIdentity],
        conditions: CompartmentConditions,
    ) -> list[MetabolitePrice]:
        """Price every metabolite one of this file's records names."""
        out: list[MetabolitePrice] = []
        for met in metabolites:
            row = self.rows.get(met.met_id)
            if row is None:
                out.append(
                    MetabolitePrice(
                        met_id=met.met_id, reason=UnpricedReason.NOT_ATTEMPTED
                    )
                )
                continue
            sigma = row["sigma"]
            p_h, ionic, p_mg = conditions.for_compartment(met.compartment)
            out.append(
                MetabolitePrice(
                    met_id=met.met_id,
                    dgf_prime_kj_per_mol=row["value"],
                    sigma_kj_per_mol=(
                        float(sigma)
                        if sigma is not None and np.isfinite(sigma)
                        else None
                    ),
                    route=row["route"],
                    p_h=p_h,
                    ionic_strength_m=ionic,
                    p_mg=p_mg,
                    temperature_k=conditions.temperature_k,
                    provenance=row["provenance"],
                )
            )
        return out


class StructureAttemptReasonResolver:
    """A diagnosis-only stage over ``thermo_structure_attempt.json``.

    It prices nothing. What it adds is a reason drawn from the STRUCTURE rather than from
    the namespace: a structure carrying an attachment point makes the species a residue in
    a macromolecule, and a structure the decomposer refuses is a different problem from one
    that does not exist.
    """

    name = "structure_attempt_diagnosis"

    REASON_BY_KEY = {
        "placeholder_structure": UnpricedReason.PLACEHOLDER_STRUCTURE,
        "group_decomposition_failed": UnpricedReason.GROUP_DECOMPOSITION_FAILED,
        "no_structure_available": UnpricedReason.NO_STRUCTURE_AVAILABLE,
    }

    def __init__(self, path: str) -> None:
        """Read the attempt report and index its stated reason by metabolite id."""
        self.path = path
        with open(path) as handle:
            payload = json.load(handle)
        self.reason: dict[str, tuple[UnpricedReason, str]] = {}
        for record in payload["records"]:
            reason = self.REASON_BY_KEY.get(str(record.get("reason")))
            if reason is None:
                continue
            for met_id in record.get("met_ids", []) or [record["met_id"]]:
                self.reason[str(met_id)] = (reason, str(record.get("outcome", "")))

    def resolve(
        self,
        metabolites: list[MetaboliteIdentity],
        conditions: CompartmentConditions,
    ) -> list[MetabolitePrice]:
        """State a reason for every species the attempt looked at; price nothing."""
        out: list[MetabolitePrice] = []
        for met in metabolites:
            entry = self.reason.get(met.met_id)
            out.append(
                MetabolitePrice(
                    met_id=met.met_id,
                    reason=entry[0] if entry else UnpricedReason.NOT_ATTEMPTED,
                    detail=(
                        f"thermo_structure_attempt outcome={entry[1]}" if entry else ""
                    ),
                    provenance={"source_path": self.path} if entry else {},
                )
            )
        return out


class AuditReasonResolver:
    """A stage that prices NOTHING and only states why a metabolite cannot be priced.

    ``audit_unresolved_metabolites.py`` already worked out, per metabolite, which of four
    things is missing, and that distinction is what makes a coverage ceiling actionable:
    a structure that the group decomposer refuses needs a decomposition, and no pKa
    predictor or structure writer will ever help it. Without this stage every such species
    falls back on the pipeline's own default, ``no_structure_available``, which is the wrong
    remedy for it.

    Placed LAST, so it can only speak about metabolites nothing priced.
    """

    name = "audit_diagnosis"

    #: The audit's blocker vocabulary mapped onto the pipeline's reason enum.
    BLOCKER_TO_REASON = {
        "structure_known_but_no_group_decomposition": (
            UnpricedReason.GROUP_DECOMPOSITION_FAILED
        ),
        "placeholder_formula_no_definite_structure": UnpricedReason.PLACEHOLDER_FORMULA,
        "no_usable_structure_available": UnpricedReason.NO_STRUCTURE_AVAILABLE,
        # Tier a: a structure resolves, the cache has no compound for it, and the external
        # bridge could not create one. What is missing is the pKa route, not the structure.
        "structure_absent_from_cache": UnpricedReason.NO_PKA_PREDICTION,
        "only_composition_mismatched_candidates": UnpricedReason.NO_STRUCTURE_AVAILABLE,
    }

    def __init__(self, path: str) -> None:
        """Read the audit table and index its blocker column by metabolite id."""
        self.path = path
        frame = pd.read_csv(path)
        self.blocker = {
            str(met_id): str(blocker)
            for met_id, blocker in zip(frame["met_id"], frame["final_blocker"])
            if isinstance(blocker, str)
        }

    def resolve(
        self,
        metabolites: list[MetaboliteIdentity],
        conditions: CompartmentConditions,
    ) -> list[MetabolitePrice]:
        """State a reason for every metabolite; price nothing."""
        out: list[MetabolitePrice] = []
        for met in metabolites:
            blocker = self.blocker.get(met.met_id)
            reason = self.BLOCKER_TO_REASON.get(blocker or "")
            out.append(
                MetabolitePrice(
                    met_id=met.met_id,
                    reason=reason or UnpricedReason.NOT_ATTEMPTED,
                    detail=f"audit_unresolved_metabolites final_blocker={blocker}",
                    provenance={"source_path": self.path},
                )
            )
        return out


def _shipped_series(name: str) -> dict[str, float]:
    """One shipped table as ``bare id -> raw value``, sentinel and NaN dropped."""
    frame = pd.read_csv(osp.join(GEM_DIR, "data", "databases", name))
    values = pd.to_numeric(frame["Var2"], errors="coerce")
    out: dict[str, float] = {}
    for key, value in zip(frame["Var1"].astype(str), values):
        if not np.isfinite(value) or value >= SHIPPED_SENTINEL:
            continue
        out[key] = float(value)
    return out


class ExclusionRecord(BaseModel):
    """One metabolite that cannot be priced, with what it costs and what would fix it."""

    met_id: str
    name: str
    compartment: str
    formula: str
    charge: int
    reason: UnpricedReason
    n_reactions_blocked: int
    reactions_blocked: list[str]
    #: Which of the three states this is in: nothing is possible, a structure is needed,
    #: a decomposition is needed.
    remedy: str


#: Reasons whose species are not molecules at all. A reaction blocked ONLY by these can
#: never be priced by anyone: a SLIME backbone's formula is a difference formula and an R,
#: X or * formula names a family. Every other reason names work that is possible.
PERMANENTLY_CLOSED_REASONS = frozenset(
    {
        UnpricedReason.PSEUDO_METABOLITE,
        UnpricedReason.PLACEHOLDER_FORMULA,
        # A structure carrying an attachment point is the same kind of thing as a
        # placeholder formula, one level down: the species is a residue inside a
        # macromolecule and has no formation energy of its own. Detected from the
        # structure by `thermo_structure_attempt.py`, not from the name.
        UnpricedReason.PLACEHOLDER_STRUCTURE,
    }
)


class CeilingRung(BaseModel):
    """One rung of the coverage ladder: what is priced, and what it would take to get here."""

    name: str
    n_reactions_priced: int
    fraction_reactions_priced: float
    n_flux_carrying_priced: int | None = None
    fraction_flux_carrying_priced: float | None = None
    #: Which remaining reasons this rung assumes resolved, and how many reactions each
    #: contributes. Empty for the rung that reports the state as built.
    reasons_resolved: dict[str, int] = {}
    what_it_would_take: str
    evidence: str = ""


class CeilingLadder(BaseModel):
    """Now, reachable and hard ceiling, derived from the result rather than asserted.

    A rung is computed by asking which reactions would close if a given set of blocking
    reasons were resolved, using the SAME per-reaction blocker records the coverage report
    carries. Nothing here is a hand-entered number.
    """

    built_at: str
    n_reactions: int
    n_flux_carrying: int
    rungs: list[CeilingRung]
    n_permanently_closed_reactions: int
    permanently_closed_reasons: dict[str, int]
    note: str


class CoverageCeiling(BaseModel):
    """The reaction coverage ceiling, by route and by remaining reason."""

    built_at: str
    gem: str
    gem_sha256: str
    stages: list[str]
    n_metabolites: int
    n_metabolites_priced: int
    metabolite_routes: dict[str, int]
    metabolite_reasons: dict[str, int]
    n_reactions: int
    n_reactions_priced: int
    fraction_reactions_priced: float
    reaction_routes: dict[str, int]
    reaction_reasons: dict[str, int]
    #: reason -> how many reactions are blocked ONLY by metabolites carrying that reason.
    reactions_by_sole_blocking_reason: dict[str, int]
    flux_carrying: dict[str, float]
    n_reactions_with_sigma: int
    top_blocking_metabolites: list[dict[str, object]]
    remedy_summary: dict[str, dict[str, int]]
    betaxanthin_route: dict[str, object]
    ceiling_ladder: CeilingLadder
    notes: str


#: What each remaining reason would take, stated once so the ceiling is readable.
REMEDY = {
    UnpricedReason.PLACEHOLDER_FORMULA: (
        "nothing is possible: an R, X or * in the formula means the species is a family of "
        "molecules and no single formation energy exists for it"
    ),
    UnpricedReason.PSEUDO_METABOLITE: (
        "nothing is possible: a modeling device (a pool, a SLIME backbone, a difference "
        "formula), not a molecule"
    ),
    UnpricedReason.NO_STRUCTURE_AVAILABLE: (
        "a STRUCTURE is needed: a definite formula with no structure in any namespace and "
        "no writer that composes one from the name"
    ),
    UnpricedReason.GROUP_DECOMPOSITION_FAILED: (
        "a DECOMPOSITION is needed: the structure exists and component contribution refuses "
        "it, so a pKa predictor changes nothing; monatomic ions are in this class and their "
        "energies are tabulated in the thermochemical literature rather than predicted"
    ),
    UnpricedReason.NO_PKA_PREDICTION: (
        "a pKa LADDER is needed: the structure decomposes and the pKa stage returned nothing"
    ),
    UnpricedReason.SURROGATE_SPREAD_TOO_WIDE: (
        "nothing defensible: same-composition compounds disagree by more than the admitted "
        "bound, so borrowing would be a guess with a number attached"
    ),
    UnpricedReason.PLACEHOLDER_STRUCTURE: (
        "nothing is possible: the only structure that resolves carries an attachment "
        "point, so the species is a residue inside a macromolecule rather than a molecule"
    ),
    UnpricedReason.NOT_ATTEMPTED: "no configured stage claimed it",
    UnpricedReason.BLOCKED_BY_METABOLITE: "see the blocking metabolites",
    UnpricedReason.NO_PARTICIPANTS: "an empty stoichiometry; not a reaction",
}


def build_stages(
    args: argparse.Namespace, ion_gauge_shift_kj_per_mol: float = 0.0
) -> list:
    """The yeast-GEM stage list, in precedence order, skipping artifacts not yet built.

    ``ion_gauge_shift_kj_per_mol`` is forwarded to the ion stage and exists only for
    ``thermo_ion_gauge_check.py``, which reruns the whole pipeline under a different
    gauge to measure which reaction energies depend on the ion table's absolute
    values. A production run leaves it at zero.
    """
    stages: list = []

    recomputation = osp.join(THERMO_DIR, "metabolite_dgf_prime.parquet")
    stages.append(
        ParquetResolver(
            name="equilibrator_recomputation",
            path=recomputation,
            value_column="dgf_prime_kj_per_mol",
            sigma_column="sigma_kj_per_mol",
            tier_column="resolution_tier",
            route=ThermoRoute.CACHE_ACCESSION,
        )
    )

    # The accession sweep is the same ROUTE as the recomputation, a cache lookup, with a
    # better stopping rule: try every accession and accept the first that carries a finite
    # formation energy rather than the first that resolves to a record. It runs second
    # because it only speaks about metabolites the recomputation left unpriced.
    sweep = osp.join(RESULTS, "thermo_accession_sweep.json")
    if osp.isfile(sweep):
        stages.append(
            JsonRecordResolver(
                name="accession_sweep",
                path=sweep,
                default_route=ThermoRoute.CACHE_ACCESSION,
                provenance_keys=(
                    "name",
                    "accepted_namespace",
                    "accepted_compound_id",
                    "previous_reason",
                    "outcome",
                ),
            )
        )
    else:
        print(f"skip accession_sweep: no {sweep}")

    # The re-priced targets come FIRST, ahead of the lipid file they correct. 48 of the
    # lipid records were priced on the neutral parent, a convention that moves dGf'0 by a
    # median 1,585 kJ/mol against the pH 7 microspecies the cache references, so leaving
    # the lipid stage ahead of this one would keep the wrong value for every compound the
    # re-pricing exists to fix.
    applied = osp.join(RESULTS, "external_pka_applied.json")
    if osp.isfile(applied):
        stages.append(
            JsonRecordResolver(
                name="external_pka_targets",
                path=applied,
                default_route=ThermoRoute.EXTERNAL_PKA,
                provenance_keys=(
                    "name",
                    "method",
                    "method_version",
                    "weights_sha256",
                    "smiles",
                    "ph7_smiles",
                    "pkas",
                    "conditions",
                ),
            )
        )
    else:
        print(f"skip external_pka_targets: no {applied}")

    lipids = osp.join(RESULTS, "lipid_priced.json")
    if osp.isfile(lipids):
        stages.append(
            JsonRecordResolver(
                name="structure_writer_lipids",
                path=lipids,
                route_map={
                    "cache_lookup": ThermoRoute.CACHE_STRUCTURE,
                    "external_pka": ThermoRoute.EXTERNAL_PKA,
                },
                default_route=ThermoRoute.STRUCTURE_WRITER,
                provenance_keys=(
                    "name",
                    "chemical_class",
                    "smiles",
                    "inchi_key",
                    "route",
                    "cache_compound_id",
                    "pkas",
                ),
            )
        )
    else:
        print(f"skip structure_writer_lipids: no {lipids}")

    # The bare ions, from a pinned thermochemical table. A group-contribution model cannot
    # reach them even in principle, so this is the one route that is a measurement from the
    # literature rather than a computation, and it is tagged as its own.
    ion_table = osp.join(
        DATA_ROOT, "data", "torchcell", "ion_formation_energies",
        "ion_formation_energies.json",
    )
    if osp.isfile(ion_table):
        stages.append(
            IonFormationResolver(
                ion_table, gauge_shift_kj_per_mol=ion_gauge_shift_kj_per_mol
            )
        )
    else:
        print(
            f"skip ion_formation_table: no {ion_table}; build it with "
            "experiments/026-metabolism-flux/scripts/fetch_ion_formation_energies.py"
        )

    surrogate = osp.join(THERMO_DIR, "metabolite_dgf_surrogate.parquet")
    if osp.isfile(surrogate) and not args.no_surrogate:
        stages.append(
            ParquetResolver(
                name="composition_surrogate",
                path=surrogate,
                value_column="dgf_prime_surrogate_kj_per_mol",
                sigma_column="surrogate_sd_kj_per_mol",
                route=ThermoRoute.COMPOSITION_SURROGATE,
                row_filter=lambda f: f["surrogate_spread_kj_per_mol"]
                <= args.surrogate_spread,
                reason=UnpricedReason.SURROGATE_SPREAD_TOO_WIDE,
            )
        )

    # The repeat-unit surrogate sits LAST among the value-producing stages, after the
    # shipped column, because it is the weakest evidence in the list: a residue energy with
    # a measured bond correction and an inflated sigma. Anything else that can price a
    # species gets it first.
    polymer = osp.join(RESULTS, "thermo_polymer_repeat_unit.json")
    if osp.isfile(polymer) and not args.no_repeat_unit:
        stages.append(
            JsonRecordResolver(
                name="repeat_unit_surrogate",
                path=polymer,
                route_key="outcome",
                route_map={"repeat_unit_surrogate": ThermoRoute.REPEAT_UNIT_SURROGATE},
                default_route=ThermoRoute.REPEAT_UNIT_SURROGATE,
                provenance_keys=(
                    "name",
                    "n_residues",
                    "monomer_name",
                    "monomer_source",
                    "detail",
                    "outcome",
                ),
            )
        )

    shipped = _shipped_series("model_metDeltaG.csv")
    # The shipped metabolite table is keyed by BARE id while the model compartmentizes it;
    # yeast-GEM's ids happen to be bare in both, but the mapping is made explicit so a model
    # that tags them does not silently miss every row.
    stages.append(
        TableResolver(
            name="shipped_metabolite_column",
            values=shipped,
            units="kcal/mol",
            source_path=osp.join(GEM_DIR, "data", "databases", "model_metDeltaG.csv"),
            sha256=sha256_of(osp.join(GEM_DIR, "data", "databases", "model_metDeltaG.csv")),
        )
    )

    audit = osp.join(RESULTS, "unresolved_metabolites.csv")
    if osp.isfile(audit):
        stages.append(AuditReasonResolver(audit))
    else:
        print(f"skip audit_diagnosis: no {audit}")

    # Diagnosis only: the structure attempt prices nothing and states, per species, which
    # of three things stopped it. It runs LAST because a later stage's stated reason wins,
    # and this one looked at the STRUCTURE where the audit only looked at the namespace.
    attempt = osp.join(RESULTS, "thermo_structure_attempt.json")
    if osp.isfile(attempt):
        stages.append(StructureAttemptReasonResolver(attempt))
    else:
        print(f"skip structure_attempt_diagnosis: no {attempt}")
    return stages


def _reaction_table() -> tuple[dict[str, float], list[str], str]:
    """The shipped REACTION column and the rows that must not be converted from kcal."""
    path = osp.join(RESULTS, "dgpredictor_reaction_rows.json")
    if not osp.isfile(path):
        raise FileNotFoundError(
            f"{path} is missing; run thermo_dgpredictor_rows.py first, or the 33 kJ/mol "
            "rows of the shipped reaction column are converted a second time"
        )
    with open(path) as handle:
        record = json.load(handle)
    return (
        _shipped_series("model_rxnDeltaG.csv"),
        list(record["dgpredictor_reaction_ids"]),
        path,
    )


#: FVA width below which a reaction counts as blocked. Solver noise puts a width of order
#: 1e-13 on reactions that are exactly blocked, so a bare ``> 0`` calls 2,877 reactions
#: flux carrying against this threshold's 2,845. The earlier coverage ladders used 1e-9;
#: the same value is used here so the before and after are the same population.
FVA_WIDTH_FLOOR = 1e-9


def _flux_carrying_ids() -> list[str]:
    """Reactions with a nonzero FVA width, from the wild-type FVA already computed."""
    path = osp.join(RESULTS, "fva_wildtype.csv")
    if not osp.isfile(path):
        return []
    frame = pd.read_csv(path, index_col=0)
    return [str(i) for i in frame.index[frame["width"] > FVA_WIDTH_FLOOR]]


def _pseudo_metabolite_ids(model) -> list[str]:
    """Metabolites that are modeling devices rather than molecules.

    Two families, both identified by a PROPERTY rather than by a hand list: a species with
    no formula at all, and yeast-GEM's SLIME ``... backbone`` species, whose formula is a
    DIFFERENCE formula (``triglyceride backbone`` is ``C3H2``) and so describes a bookkeeping
    quantity. The name test is yeast-GEM's convention and lives here, in the organism-
    specific driver, rather than in the module.
    """
    out = []
    for met in model.metabolites:
        if not (met.formula or "").strip():
            out.append(met.id)
        elif (met.name or "").strip().endswith("backbone"):
            out.append(met.id)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--surrogate-spread",
        type=float,
        default=5.0,
        help=(
            "admit a composition surrogate only when the compounds it is drawn from agree "
            "within this many kJ/mol; 5 is under RT ln(10) at 303 K"
        ),
    )
    parser.add_argument("--no-surrogate", action="store_true")
    parser.add_argument(
        "--no-repeat-unit",
        action="store_true",
        help="drop the repeat-unit surrogate stage, for a table with no surrogates in it",
    )
    args = parser.parse_args()

    import cobra

    gem_path = osp.join(GEM_DIR, "model", "yeast-GEM.xml")
    model = cobra.io.read_sbml_model(gem_path)
    _, _, rows_path = _reaction_table()
    config = build_config(model)

    stages = build_stages(args)
    result = run_thermo_pipeline(model, config, stages, formation_sigma())
    print(summarize(result))

    os.makedirs(RESULTS, exist_ok=True)
    full_path = osp.join(RESULTS, "thermo_pipeline_yeast9.json")
    write_result(result, full_path)
    print(f"wrote {full_path}")

    _write_ceiling(result, model, rows_path)
    _write_exclusions(result, model)


def build_config(model) -> ThermoPipelineConfig:
    """The yeast-GEM conditions, reaction-level fill and pseudo-metabolite list."""
    gem_path = osp.join(GEM_DIR, "model", "yeast-GEM.xml")
    reaction_values, already_kj, _ = _reaction_table()
    conditions = CompartmentConditions(
        p_h=dict(COMPARTMENT_PH),
        default_p_h=7.0,
        default_ionic_strength_m=0.25,
        default_p_mg=3.0,
        temperature_k=303.15,
        source=(
            "pH: build_equilibrator_thermo.COMPARTMENT_PH, textbook organellar values, "
            "assumed rather than measured; ionic strength and pMg: eQuilibrator defaults; "
            "T: 303.15 K, the temperature the yeast phenotype data was collected at"
        ),
    )
    config = ThermoPipelineConfig(
        organism="Saccharomyces cerevisiae (yeast-GEM 9.0.2)",
        gem_path=gem_path,
        gem_sha256=sha256_of(gem_path),
        conditions=conditions,
        flux_carrying_reaction_ids=_flux_carrying_ids(),
        reaction_table=reaction_values,
        reaction_table_units="kcal/mol",
        reaction_table_already_kj=already_kj,
        reaction_table_source=osp.join(
            GEM_DIR, "data", "databases", "model_rxnDeltaG.csv"
        ),
        reaction_table_sha256=sha256_of(
            osp.join(GEM_DIR, "data", "databases", "model_rxnDeltaG.csv")
        ),
        pseudo_metabolite_ids=_pseudo_metabolite_ids(model),
    )
    return config


def formation_sigma() -> dict[str, float]:
    """Per-metabolite formation sigma from the recomputation, where it has one."""
    frame = pd.read_parquet(
        osp.join(THERMO_DIR, "metabolite_dgf_prime.parquet"),
        columns=["met_id", "sigma_kj_per_mol"],
    )
    return {
        str(k): float(v)
        for k, v in zip(frame["met_id"], frame["sigma_kj_per_mol"])
        if np.isfinite(v)
    }


def _build_ceiling_ladder(result: ThermoPipelineResult) -> CeilingLadder:
    """Now, reachable and hard ceiling, all derived from the per-reaction blocker records.

    The rungs are cumulative and each names the reasons it assumes resolved. A reaction
    counts toward a rung when EVERY reason blocking it is in that rung's resolved set, which
    is the same all-participants predicate the pricing itself uses.
    """
    reason_by_met = {p.met_id: p.reason for p in result.metabolites}
    flux_ids = set(result.config.flux_carrying_reaction_ids)

    unpriced = [r for r in result.reactions if r.delta_r_g0_kj_per_mol is None]
    blocking_reasons: dict[str, set[UnpricedReason]] = {}
    for reaction in unpriced:
        reasons = {
            reason_by_met[m]
            for m in reaction.blocking_met_ids
            if reason_by_met.get(m) is not None
        }
        blocking_reasons[reaction.reaction_id] = reasons  # type: ignore[assignment]

    # A reaction is closed for good as soon as ANY ONE of its blockers can never be
    # priced, not only when all of them are. Requiring all of them would leave a reaction
    # blocked by a SLIME backbone AND a missing structure in neither bucket, which is how
    # two of yeast-GEM's reactions went unaccounted for on the first pass.
    permanently_closed = [
        reaction_id
        for reaction_id, reasons in blocking_reasons.items()
        if reasons & PERMANENTLY_CLOSED_REASONS
    ]
    closed_counts: dict[str, int] = {}
    for reaction_id in permanently_closed:
        for reason in blocking_reasons[reaction_id] & PERMANENTLY_CLOSED_REASONS:
            closed_counts[str(reason)] = closed_counts.get(str(reason), 0) + 1

    n_total = result.reaction_counts.n_total
    n_flux = len(flux_ids)
    base_priced = result.reaction_counts.n_priced
    base_flux = sum(
        1
        for r in result.reactions
        if r.reaction_id in flux_ids and r.delta_r_g0_kj_per_mol is not None
    )

    def _rung(
        name: str, resolved: set[UnpricedReason], what: str, evidence: str = ""
    ) -> CeilingRung:
        gained = [
            reaction_id
            for reaction_id, reasons in blocking_reasons.items()
            if reasons and reasons <= resolved
        ]
        per_reason: dict[str, int] = {}
        for reaction_id in gained:
            for reason in blocking_reasons[reaction_id]:
                per_reason[str(reason)] = per_reason.get(str(reason), 0) + 1
        gained_flux = sum(1 for reaction_id in gained if reaction_id in flux_ids)
        return CeilingRung(
            name=name,
            n_reactions_priced=base_priced + len(gained),
            fraction_reactions_priced=(base_priced + len(gained)) / n_total,
            n_flux_carrying_priced=base_flux + gained_flux if n_flux else None,
            fraction_flux_carrying_priced=(
                (base_flux + gained_flux) / n_flux if n_flux else None
            ),
            reasons_resolved=dict(sorted(per_reason.items(), key=lambda kv: -kv[1])),
            what_it_would_take=what,
            evidence=evidence,
        )

    structure_side = {
        UnpricedReason.NO_STRUCTURE_AVAILABLE,
        UnpricedReason.NO_PKA_PREDICTION,
    }
    everything_possible = {
        reason
        for reasons in blocking_reasons.values()
        for reason in reasons
    } - PERMANENTLY_CLOSED_REASONS

    rungs = [
        CeilingRung(
            name="now",
            n_reactions_priced=base_priced,
            fraction_reactions_priced=base_priced / n_total,
            n_flux_carrying_priced=base_flux if n_flux else None,
            fraction_flux_carrying_priced=base_flux / n_flux if n_flux else None,
            reasons_resolved={},
            what_it_would_take="nothing; this is the table as built",
        ),
        _rung(
            "structures_and_pka_ladders_resolved",
            structure_side,
            "a structure from a namespace or a curator for every species that has none, "
            "and a pKa ladder inside (0, 14) for every species whose ladder is empty",
            "no measurement here says these are obtainable; the rung is what WOULD close",
        ),
        _rung(
            "everything_chemically_possible_resolved",
            everything_possible,
            "the rung above plus a decomposition or a tabulated value for every structure "
            "the group decomposer refuses",
            "thermo_tier_d_probe.json measured that none of the remaining refused "
            "structures is reachable by the reactant-contribution path or by a surrogate "
            "within the admitted spread, so this rung needs new data rather than new code",
        ),
    ]
    return CeilingLadder(
        built_at=datetime.now(UTC).isoformat(),
        n_reactions=n_total,
        n_flux_carrying=n_flux,
        rungs=rungs,
        n_permanently_closed_reactions=len(permanently_closed),
        permanently_closed_reasons=dict(
            sorted(closed_counts.items(), key=lambda kv: -kv[1])
        ),
        note=(
            "A reaction is permanently closed when every metabolite blocking it is not a "
            "molecule: a SLIME backbone whose formula is a difference formula, or a "
            "placeholder formula that names a family. No pKa predictor, structure writer, "
            "decomposer or literature table changes those, so the last rung is the ceiling "
            "for any method at all rather than for this one. The rungs and the permanently "
            "closed set partition the unpriced reactions exactly: every unpriced reaction "
            "is either closed for good or named by the last rung."
        ),
    )


def _write_ceiling(result: ThermoPipelineResult, model, rows_path: str) -> None:
    reason_by_met = {p.met_id: p.reason for p in result.metabolites}
    sole: dict[str, int] = {}
    for reaction in result.reactions:
        if reaction.delta_r_g0_kj_per_mol is not None:
            continue
        reasons = {
            str(reason_by_met.get(m))
            for m in reaction.blocking_met_ids
            if reason_by_met.get(m) is not None
        }
        key = reasons.pop() if len(reasons) == 1 else "mixed_reasons"
        sole[key] = sole.get(key, 0) + 1

    remedy_summary: dict[str, dict[str, int]] = {}
    for price in result.metabolites:
        if price.priced or price.reason is None:
            continue
        remedy_summary.setdefault(REMEDY[price.reason], {"n_metabolites": 0})
        remedy_summary[REMEDY[price.reason]]["n_metabolites"] += 1

    ceiling = CoverageCeiling(
        built_at=datetime.now(UTC).isoformat(),
        gem=result.config.gem_path,
        gem_sha256=result.config.gem_sha256,
        stages=result.config.stage_names,
        n_metabolites=result.metabolite_counts.n_total,
        n_metabolites_priced=result.metabolite_counts.n_priced,
        metabolite_routes=result.metabolite_counts.by_route,
        metabolite_reasons=result.metabolite_counts.by_reason,
        n_reactions=result.reaction_counts.n_total,
        n_reactions_priced=result.reaction_counts.n_priced,
        fraction_reactions_priced=result.reaction_counts.fraction,
        reaction_routes=result.reaction_counts.by_route,
        reaction_reasons=result.reaction_counts.by_reason,
        reactions_by_sole_blocking_reason=dict(
            sorted(sole.items(), key=lambda kv: -kv[1])
        ),
        flux_carrying=result.flux_carrying,
        n_reactions_with_sigma=sum(
            1 for r in result.reactions if r.sigma_kj_per_mol is not None
        ),
        top_blocking_metabolites=result.blocking_metabolites[:25],
        remedy_summary=remedy_summary,
        betaxanthin_route=_betaxanthin_block(result, model),
        ceiling_ladder=_build_ceiling_ladder(result),
        notes=(
            "A reaction is priced only when EVERY participant carries a formation energy, "
            "or when the shipped reaction column fills it directly. The shipped reaction "
            "column mixes units and the 33 dGPredictor rows named in "
            f"{osp.basename(rows_path)} are exempted from the kcal conversion."
        ),
    )
    path = osp.join(RESULTS, "reaction_thermo_coverage_final.json")
    with open(path, "w") as handle:
        handle.write(ceiling.model_dump_json(indent=1))
    print(f"wrote {path}")
    print(
        f"  reactions {ceiling.n_reactions_priced}/{ceiling.n_reactions}"
        f" = {ceiling.fraction_reactions_priced:.3%}"
    )
    print(f"  sole blocking reason: {ceiling.reactions_by_sole_blocking_reason}")


def _betaxanthin_block(result: ThermoPipelineResult, model) -> dict[str, object]:
    """The betaxanthin route's own coverage, extending ``pathway_thermo_betaxanthin.json``.

    The heterologous route is not in yeast-GEM, so what is reported here is the coverage of
    the YEAST reactions it draws on: every reaction touching one of the pathway's yeast
    precursors. Those are the reactions whose driving force the flux layer would have to get
    right for a betaxanthin prediction to mean anything.
    """
    from torchcell.metabolism.betaxanthin import (
        DEFAULT_PARTNERS,
        NADP,
        NADPH,
        OXYGEN,
        PROTON,
        TYROSINE,
        WATER,
    )

    path = osp.join(RESULTS, "pathway_thermo_betaxanthin.json")
    if not osp.isfile(path):
        return {"note": f"no {path}"}
    with open(path) as handle:
        pathway = json.load(handle)

    # The route's own five intermediates are heterologous and are NOT in yeast-GEM, so
    # their energies come from the cache directly and are reported from the pathway file.
    # What the GEM contributes is the anchors the heterologous reactions draw on: the
    # tyrosine precursor, the redox pair, oxygen, water, the proton, and the amine-donor
    # pool the betaxanthin condensation uses. Those ids are the ones
    # torchcell.metabolism.betaxanthin itself uses to write the reactions, so the anchor
    # list is not restated here.
    anchors = {
        TYROSINE: "tyrosine precursor",
        OXYGEN: "oxygen",
        NADPH: "NADPH",
        NADP: "NADP+",
        WATER: "water",
        PROTON: "proton",
        **{met_id: f"amine donor: {name}" for met_id, name in DEFAULT_PARTNERS.items()},
    }
    prices = {p.met_id: p for p in result.metabolites}
    anchor_rows = [
        {
            "met_id": met_id,
            "role": role,
            "priced": bool(met_id in prices and prices[met_id].priced),
            "route": str(prices[met_id].route) if met_id in prices else "absent",
            "dgf_prime_kj_per_mol": (
                prices[met_id].dgf_prime_kj_per_mol if met_id in prices else None
            ),
        }
        for met_id, role in anchors.items()
    ]
    present = {row["met_id"] for row in anchor_rows if row["route"] != "absent"}
    touching = [
        r.id for r in model.reactions if {m.id for m in r.metabolites} & present
    ]
    by_id = {r.reaction_id: r for r in result.reactions}
    priced = [r for r in touching if by_id[r].delta_r_g0_kj_per_mol is not None]
    reasons: dict[str, int] = {}
    for reaction_id in touching:
        row = by_id[reaction_id]
        if row.delta_r_g0_kj_per_mol is None and row.reason is not None:
            reasons[str(row.reason)] = reasons.get(str(row.reason), 0) + 1
    return {
        "pathway_file": path,
        "n_pathway_compounds": pathway.get("n_compounds"),
        "n_pathway_compounds_resolved": pathway.get("n_resolved"),
        "heterologous_compounds_all_priced": (
            pathway.get("n_resolved") == pathway.get("n_compounds")
        ),
        "anchor_metabolites": anchor_rows,
        "n_anchors": len(anchor_rows),
        "n_anchors_priced": sum(1 for row in anchor_rows if row["priced"]),
        "n_yeast_reactions_touching_an_anchor": len(touching),
        "n_priced": len(priced),
        "fraction_priced": (len(priced) / len(touching)) if touching else 0.0,
        "unpriced_reasons": reasons,
        "note": (
            "the five heterologous intermediates are not in yeast-GEM and are priced "
            "directly from the compound cache (pathway_thermo_betaxanthin.json); what is "
            "counted here is the yeast side the heterologous reactions draw on, whose "
            "driving forces the flux layer has to get right for a betaxanthin prediction "
            "to mean anything"
        ),
    }


def _write_exclusions(result: ThermoPipelineResult, model) -> None:
    """One record per unpriced metabolite: the reason, the formula, the reactions blocked."""
    blocked: dict[str, list[str]] = {}
    for reaction in result.reactions:
        if reaction.delta_r_g0_kj_per_mol is not None:
            continue
        for met_id in reaction.blocking_met_ids:
            blocked.setdefault(met_id, []).append(reaction.reaction_id)
    identities = {m.met_id: m for m in identities_from_model(model)}

    records: list[ExclusionRecord] = []
    for price in result.metabolites:
        if price.priced or price.reason is None:
            continue
        met = identities[price.met_id]
        reactions = sorted(blocked.get(price.met_id, []))
        records.append(
            ExclusionRecord(
                met_id=price.met_id,
                name=met.name,
                compartment=met.compartment,
                formula=met.formula,
                charge=met.charge,
                reason=price.reason,
                n_reactions_blocked=len(reactions),
                reactions_blocked=reactions,
                remedy=REMEDY[price.reason],
            )
        )
    records.sort(key=lambda r: (-r.n_reactions_blocked, r.met_id))

    payload = {
        "built_at": datetime.now(UTC).isoformat(),
        "n_excluded_metabolites": len(records),
        "by_reason": {
            reason: sum(1 for r in records if str(r.reason) == reason)
            for reason in sorted({str(r.reason) for r in records})
        },
        "reactions_blocked_by_reason": {
            reason: len(
                {
                    reaction
                    for r in records
                    if str(r.reason) == reason
                    for reaction in r.reactions_blocked
                }
            )
            for reason in sorted({str(r.reason) for r in records})
        },
        "records": [r.model_dump() for r in records],
    }
    path = osp.join(RESULTS, "thermo_exclusions.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f"wrote {path}: {len(records)} excluded metabolites")
    print(f"  by reason: {payload['by_reason']}")
    print(f"  reactions each reason touches: {payload['reactions_blocked_by_reason']}")


if __name__ == "__main__":
    main()
