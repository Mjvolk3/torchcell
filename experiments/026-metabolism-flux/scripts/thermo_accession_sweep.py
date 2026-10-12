# experiments/026-metabolism-flux/scripts/thermo_accession_sweep.py
# [[experiments.026-metabolism-flux.scripts.thermo_accession_sweep]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/thermo_accession_sweep.py

r"""Try EVERY accession a metabolite carries, not the first one that resolves to a compound.

**The bug this closes, and it is worth 9 reactions on its own.** The metabolite audit walks
namespaces in a priority order and stops at the first that resolves to a cache compound.
That is the wrong stopping rule: a namespace can resolve to a compound record that carries
no formation energy while a different namespace on the SAME metabolite resolves to one that
does. yeast-GEM's hydrogen sulfide is the case. Its MetaNetX accession resolves to cache
compound 97968, for which ``standard_dg_formation`` returns nothing, so the audit filed all
three copies as ``no_usable_structure_available``. Its BiGG and ChEBI accessions both
resolve to compound 74380, ``[SH-]``, which has a group vector and a formation energy of
11.57 kJ/mol.

So the sweep here changes the stopping rule: try every accession, and accept the first that
yields a FINITE formation energy rather than the first that yields a record. The order is
still recorded, and every accession's answer is kept, so a disagreement between two
namespaces is visible rather than hidden behind whichever one was tried first.

Organism-agnostic: nothing here names a metabolite or a namespace, and the candidate list is
whatever the model's own annotations hold.

**What is NOT done here.** A compound whose composition disagrees with the GEM's formula is
recorded and REJECTED, not priced: two namespaces pointing at different molecules is a
mapping error, and pricing the wrong molecule is worse than leaving the reaction unpriced.
The comparison is on the heavy-atom composition, since the cache stores a microspecies whose
proton count is a function of pH and the GEM's is a function of its own charge convention.

Writes ``results/thermo_accession_sweep.json``, which ``thermo_pipeline_yeast9.py`` reads as
a resolver stage.

Run under the eQuilibrator environment::

    /scratch/projects/torchcell-scratch/envs/equilibrator/bin/python \
        experiments/026-metabolism-flux/scripts/thermo_accession_sweep.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import re
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
PIPELINE_JSON = osp.join(RESULTS, "thermo_pipeline_yeast9.json")
AUDIT_CSV = osp.join(RESULTS, "unresolved_metabolites.csv")
RECOMPUTATION_PARQUET = osp.join(
    DATA_ROOT, "data", "torchcell", "thermo_equilibrator", "metabolite_dgf_prime.parquet"
)
OUT_PATH = osp.join(RESULTS, "thermo_accession_sweep.json")

P_H = 7.2
IONIC_STRENGTH_M = 0.25
TEMPERATURE_K = 303.15
P_MG = 3.0

#: Tokens that make a formula a placeholder. A species whose formula names a family has no
#: accession that can help it, so it is not swept.
PLACEHOLDER_TOKENS = ("R", "X", "*")

#: The candidate set is every metabolite the eQuilibrator RECOMPUTATION could not price, and
#: that choice is not a detail. This file IS a pipeline stage, so a sweep that asked the
#: pipeline what is still unpriced would drop every metabolite it had priced on the previous
#: run and the run after that would unprice them again. The recomputation parquet is
#: strictly upstream of this script and is never written by it, which makes the sweep
#: idempotent and the rebuild order irrelevant.
#:
#: The audit table is NOT used for this, and the reason is a finding: it records hydrogen
#: sulfide as recovered, with an empty final blocker, while the recomputation carries no
#: value for it. Taking the audit's word would have skipped the one metabolite the sweep
#: exists to catch.

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def heavy_composition(formula: str) -> dict[str, int]:
    """Heavy-atom composition of a formula string, hydrogens dropped.

    Hydrogens are dropped because the cache stores a microspecies whose proton count depends
    on pH while the GEM's depends on its own charge convention, so an H count mismatch is
    expected and a heavy-atom mismatch is a different molecule.
    """
    out: dict[str, int] = {}
    for element, count in _FORMULA_TOKEN.findall(formula or ""):
        if not element or element == "H":
            continue
        out[element] = out.get(element, 0) + (int(count) if count else 1)
    return out


class AccessionAnswer(BaseModel):
    """What one accession of one metabolite resolved to."""

    namespace: str
    accession: str
    compound_id: int | None = None
    compound_formula: str | None = None
    has_group_vector: bool = False
    dgf_untransformed_kj_per_mol: float | None = None
    composition_matches: bool | None = None
    outcome: str


class SweepRecord(BaseModel):
    """One metabolite, every accession tried, and what the sweep concluded."""

    met_id: str
    met_ids: list[str] = Field(
        default_factory=list,
        description="Every metabolite this record prices; the pipeline stage reads this.",
    )
    name: str
    compartment: str
    formula: str
    charge: int
    n_reactions_blocked: int
    previous_reason: str
    answers: list[AccessionAnswer] = Field(default_factory=list)
    accepted_namespace: str | None = None
    accepted_compound_id: int | None = None
    dgf_prime_kj_per_mol: float | None = None
    sigma_kj_per_mol: float | None = None
    outcome: str = ""
    detail: str = ""


class SweepReport(BaseModel):
    """The sweep, and what it buys."""

    built_at: str
    conditions: dict[str, float]
    stopping_rule: str = (
        "accept the first accession whose compound yields a FINITE standard formation "
        "energy AND whose heavy-atom composition matches the GEM formula; the audit's rule "
        "was to accept the first accession that resolved to a record at all"
    )
    n_candidates: int
    n_priced: int
    n_metabolite_rows_priced: int
    n_rejected_on_composition: int
    outcomes: dict[str, int]
    records: list[SweepRecord] = Field(default_factory=list)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


def main() -> None:
    import numpy as np
    from equilibrator_api import Q_, ComponentContribution

    recomputation = pd.read_parquet(RECOMPUTATION_PARQUET)
    unpriced = {
        str(row.met_id): str(row.resolution_tier)
        for row in recomputation.itertuples()
        if not np.isfinite(row.dgf_prime_kj_per_mol)
        and not any(token in str(row.formula) for token in PLACEHOLDER_TOKENS)
    }
    audit = pd.read_csv(AUDIT_CSV).set_index("met_id")

    # How many reactions each candidate blocks is reporting only, so it may come from the
    # pipeline's own last result without making the candidate set depend on it.
    blocked: dict[str, int] = {}
    if osp.isfile(PIPELINE_JSON):
        with open(PIPELINE_JSON) as handle:
            pipeline = json.load(handle)
        for reaction in pipeline["reactions"]:
            if reaction["delta_r_g0_kj_per_mol"] is not None:
                continue
            for met_id in reaction["blocking_met_ids"]:
                blocked[met_id] = blocked.get(met_id, 0) + 1

    cc = ComponentContribution()
    cc.p_h = Q_(P_H)
    cc.ionic_strength = Q_(f"{IONIC_STRENGTH_M}M")
    cc.temperature = Q_(f"{TEMPERATURE_K}K")
    cc.p_mg = Q_(P_MG)

    records: list[SweepRecord] = []
    for met_id, reason in sorted(unpriced.items()):
        if met_id not in audit.index:
            continue
        row = audit.loc[met_id]
        formula = str(row["formula"]) if pd.notna(row["formula"]) else ""
        record = SweepRecord(
            met_id=met_id,
            met_ids=[met_id],
            name=str(row["name"]),
            compartment=str(row["compartment"]),
            formula=formula,
            charge=int(row["charge"]) if pd.notna(row["charge"]) else 0,
            n_reactions_blocked=blocked.get(met_id, 0),
            previous_reason=reason,
        )
        wanted = heavy_composition(formula)
        annotations = str(row["annotations"]) if pd.notna(row["annotations"]) else ""
        for token in annotations.split(";"):
            if "=" not in token:
                continue
            namespace, _, accession = token.partition("=")
            answer = AccessionAnswer(
                namespace=namespace, accession=accession, outcome="no_compound"
            )
            try:
                compound = cc.ccache.get_compound(f"{namespace}:{accession}")
            except Exception as exc:
                answer.outcome = f"lookup_raised:{type(exc).__name__}"
                record.answers.append(answer)
                continue
            if compound is None:
                record.answers.append(answer)
                continue
            answer.compound_id = int(compound.id)
            answer.compound_formula = compound.formula
            answer.has_group_vector = compound.group_vector is not None
            try:
                mu, sigma, _ = cc.standard_dg_formation(compound)
            except Exception as exc:
                answer.outcome = f"formation_raised:{type(exc).__name__}"
                record.answers.append(answer)
                continue
            if mu is None or not np.isfinite(float(mu)):
                answer.outcome = "no_formation_energy"
                record.answers.append(answer)
                continue
            answer.dgf_untransformed_kj_per_mol = float(mu)
            answer.composition_matches = heavy_composition(
                compound.formula or ""
            ) == wanted
            if not answer.composition_matches:
                answer.outcome = "composition_mismatch"
                record.answers.append(answer)
                continue
            answer.outcome = "accepted"
            record.answers.append(answer)
            if record.dgf_prime_kj_per_mol is None:
                correction = compound.transform(
                    Q_(P_H),
                    Q_(f"{IONIC_STRENGTH_M}M"),
                    Q_(f"{TEMPERATURE_K}K"),
                    Q_(P_MG),
                )
                record.accepted_namespace = namespace
                record.accepted_compound_id = int(compound.id)
                record.dgf_prime_kj_per_mol = float(mu) + float(
                    correction.m_as("kJ/mol")
                )
                if sigma is not None:
                    record.sigma_kj_per_mol = float(
                        np.linalg.norm(np.asarray(sigma))
                    )
        if record.dgf_prime_kj_per_mol is not None:
            record.outcome = "priced"
            record.detail = (
                f"accepted {record.accepted_namespace}, compound "
                f"{record.accepted_compound_id}"
            )
        elif any(a.outcome == "composition_mismatch" for a in record.answers):
            record.outcome = "rejected_on_composition"
            record.detail = (
                "an accession resolved to a priced compound whose heavy-atom composition "
                "is not the GEM's; that is a mapping error and pricing it would be worse "
                "than leaving the reaction unpriced"
            )
        elif not record.answers:
            record.outcome = "no_accessions"
            record.detail = "the GEM carries no annotation for this metabolite"
        else:
            record.outcome = "no_accession_carries_an_energy"
            record.detail = "; ".join(
                f"{a.namespace}:{a.outcome}" for a in record.answers
            )
        records.append(record)
        print(
            f"  {met_id} {record.name[:38]:38s} {record.outcome}"
            f" blocks {record.n_reactions_blocked}",
            flush=True,
        )

    # One record per CHEMICAL would be cleaner, but the audit is per metabolite row and the
    # compartment matters for nothing here (the transform is applied per compartment by the
    # pipeline stage that reads this), so each row keeps its own record.
    priced = [r for r in records if r.outcome == "priced"]
    outcomes: dict[str, int] = {}
    for record in records:
        outcomes[record.outcome] = outcomes.get(record.outcome, 0) + 1
    report = SweepReport(
        built_at=_timestamp(),
        conditions={
            "p_h": P_H,
            "ionic_strength_m": IONIC_STRENGTH_M,
            "temperature_k": TEMPERATURE_K,
            "p_mg": P_MG,
        },
        n_candidates=len(records),
        n_priced=len(priced),
        n_metabolite_rows_priced=sum(len(r.met_ids) for r in priced),
        n_rejected_on_composition=sum(
            1 for r in records if r.outcome == "rejected_on_composition"
        ),
        outcomes=dict(sorted(outcomes.items(), key=lambda kv: -kv[1])),
        records=records,
    )
    with open(OUT_PATH, "w") as handle:
        handle.write(report.model_dump_json(indent=1))
    print(f"wrote {OUT_PATH}")
    print(f"  {report.n_priced}/{report.n_candidates} priced: {report.outcomes}")


if __name__ == "__main__":
    main()
