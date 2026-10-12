# experiments/026-metabolism-flux/scripts/external_pka_method_selection.py
# [[experiments.026-metabolism-flux.scripts.external_pka_method_selection]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/external_pka_method_selection.py

r"""Choose the pKa method the yeast-GEM fill-in uses, on a set broad enough to choose on.

The 300-compound run in ``external_pka_benchmark.py`` established that an open-source
predictor can stand in for ChemAxon at all. It cannot settle WHICH one, for three reasons
that this script removes.

**1. It is 50 compounds per stratum.** A p90 on 50 compounds is the 5th largest of 50, so
the statistic that matters most for a fill-in, the tail, is the noisiest thing in the
table. The set here is at least 2,000, and the seed-17 300 are carried over verbatim so the
earlier numbers stay comparable and the earlier run is a subset rather than a competitor.

**2. It scored at one pH.** The flux layer does not run at one pH: the recomputation prices
every metabolite at its compartment's own pH, and the compartments in use span 5.0 to 7.8
(``build_equilibrator_thermo.COMPARTMENT_PH``; the layer's own
``CompartmentParameters`` adds 7.4). The transform's sensitivity to a pKa is largest where
that pKa sits near the pH, so a tool that is best at 7.2 is not automatically best at 5.0,
and the vacuole and the extracellular space are where the transport reactions live. Every
metric here is computed at pH 7.2 AND at each working compartment pH.

**3. It stratified only by pKa count and functional group.** The fill-in is spent on
lipids: 409 of the 665 unpriced yeast-GEM metabolites are tier b, dominated by
phosphatidylinositols, cardiolipins and triacylglycerols. Those are outside the earlier
sample by construction, which capped mass at 600 Da and heavy atoms at 50. The sample here
carries a lipid-like stratum and stratifies on net charge at pH 7 and molecular weight as
well.

ENSEMBLES ARE CANDIDATES, NOT A POST HOC RESCUE
-----------------------------------------------
Five ensembles are scored alongside the individual tools: the median of the top two and of
the top three tools per pKa RANK, a CASCADE over the top two and the top three (the best
tool where it answers, the next where it does not), and a per-group rule that uses each
functional group's best tool. Which tools are "top" is decided on the pinned seed-17 300
and the ensembles are then scored on the 1,700+ compounds that were not used to rank them,
so the ensemble is not selected and evaluated on the same rows.

The cascade is in the list because averaging cannot fix a COVERAGE gap: a median over tools
is undefined exactly where the best tool returned nothing, and returning nothing is a real
failure mode here rather than a rare one. Its cost per compound is reported as the sum over
its members, which is an upper bound: the second tool only runs where the first declined.

A rank is kept in an ensemble ladder when a MAJORITY of the contributing tools produced a
pKa at that rank, and its value is the median over the tools that did. Two tools therefore
need both, three need two. Taking the union of ranks instead would let the longest ladder
decide the length, and ladder length is exactly what the tools disagree about.

Subcommands, each under the environment that holds its dependencies.

``sample``
    Draw the stratified set from the eQuilibrator compound cache, pinning the seed-17 300.
    Runs under the eQuilibrator environment.

``score``
    Score every prediction file plus the ensembles, at every working pH, overall and per
    stratum. Runs under the eQuilibrator environment (it rebuilds the Legendre transform
    with ``equilibrator_assets``).

``decide``
    Apply the selection rule to the scored JSON and write the decision as a pydantic record
    with the numbers it was decided on. Runs anywhere.

``table``
    Print the scored JSON as the markdown tables the note carries.

Run::

    EQ=/scratch/projects/torchcell-scratch/envs/equilibrator/bin/python
    $EQ experiments/026-metabolism-flux/scripts/external_pka_method_selection.py sample --n 2000
    $EQ experiments/026-metabolism-flux/scripts/external_pka_method_selection.py score
    $EQ experiments/026-metabolism-flux/scripts/external_pka_method_selection.py decide

Outputs ``results/pka_selection_validation_set.json``,
``results/pka_method_selection_scores.json``,
``results/pka_method_selection_per_compound.csv`` and ``results/pka_method_selection.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import os.path as osp
import random
import sys
from datetime import datetime, timezone

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from external_pka_benchmark import (  # noqa: E402
    IONIC_STRENGTH_M,
    MAX_PH,
    MAX_PKAS,
    MID_PH,
    MIN_PH,
    P_H,
    P_MG,
    TEMPERATURE_K,
    PredictionSet,
    ValidationCompound,
    ValidationSet,
    _groups_of,
    _neutralize,
)

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
SAMPLE_PATH = osp.join(RESULTS, "pka_selection_validation_set.json")
PRED_PATH_TEMPLATE = osp.join(RESULTS, "pka_selection_predictions_{tool}.json")
SEED17_PATH = osp.join(RESULTS, "external_pka_validation_set.json")
SCORES_PATH = osp.join(RESULTS, "pka_method_selection_scores.json")
PER_COMPOUND_CSV = osp.join(RESULTS, "pka_method_selection_per_compound.csv")
DECISION_PATH = osp.join(RESULTS, "pka_method_selection.json")

#: The pHs the pipeline actually prices metabolites at. Seven come from
#: ``build_equilibrator_thermo.COMPARTMENT_PH`` (cytoplasm 7.2, extracellular and cell
#: envelope 5.0, mitochondrion 7.8, nucleus and lipid particle 7.2, peroxisome 7.0,
#: endoplasmic reticulum 7.1, Golgi 6.6, vacuole 6.2); 7.4 is the mitochondrial pH the flux
#: layer's own ``CompartmentParameters`` default carries, which differs from the build
#: table's 7.8 and is therefore also a pH a run can be at.
WORKING_PH: tuple[float, ...] = (5.0, 6.2, 6.6, 7.0, 7.1, 7.2, 7.4, 7.8)

#: Elements a component-contribution decomposition can reach. Same set as the 300-compound
#: sampler; the lipids of interest are all within it.
ORGANIC_ELEMENTS = {"C", "H", "N", "O", "P", "S"}

#: Eight consecutive acyclic CH2 groups. An acyl chain of that length is what separates a
#: membrane lipid from a small phosphorylated metabolite, and it is the regime the tier-b
#: fill-in runs in.
ACYL_CHAIN_SMARTS = "[CX4H2][CX4H2][CX4H2][CX4H2][CX4H2][CX4H2][CX4H2][CX4H2]"

#: A phosphate ester headgroup, the other half of the lipid-like definition.
PHOSPHO_HEADGROUP_SMARTS = "[PX4](=O)([OX2,OX1-])[OX2]"

#: Molecular-weight bin edges in Da. The top bin is where the yeast-GEM lipids sit: a
#: phosphatidylinositol 34:1 is 836 Da and a cardiolipin runs past 1,400.
MASS_EDGES: tuple[float, ...] = (150.0, 300.0, 500.0, 800.0)

#: Functional group priority when a compound carries several. Phosphate first because it is
#: the scarcest and the one the lipid fill-in needs; ``none`` for a compound carrying none
#: of the four.
GROUP_PRIORITY = ("phosphate", "phenol", "carboxylate", "amine")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


# ============================================================================== sample


def _mass_bin(mass: float) -> str:
    for i, edge in enumerate(MASS_EDGES):
        if mass < edge:
            return f"mass_{i}_lt_{edge:g}"
    return f"mass_{len(MASS_EDGES)}_ge_{MASS_EDGES[-1]:g}"


def _charge_bin(charge: int) -> str:
    if charge <= -3:
        return "charge_le_-3"
    if charge >= 1:
        return "charge_ge_+1"
    return f"charge_{charge}"


def _primary_group(groups: list[str]) -> str:
    for name in GROUP_PRIORITY:
        if name in groups:
            return name
    return "none"


def _describe(
    compound_id: int,
    inchi_key: str,
    major_ms_smiles: str,
    neutral_smiles: str,
    atom_bag: dict[str, int],
    cache_pkas: list[float],
    has_mg: bool,
) -> ValidationCompound:
    """One cache row as a fully labeled validation compound.

    The net charge is read off the cache's MAJOR MICROSPECIES rather than derived from the
    pKa list, because the cache defines that microspecies at pH 7 and it is the structure
    the reference transform is anchored to.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    major = Chem.MolFromSmiles(major_ms_smiles)
    neutral = Chem.MolFromSmiles(neutral_smiles)
    groups = _groups_of(neutral_smiles)
    lipid_like = bool(
        neutral is not None
        and neutral.HasSubstructMatch(Chem.MolFromSmarts(ACYL_CHAIN_SMARTS))
        and (
            neutral.HasSubstructMatch(Chem.MolFromSmarts(PHOSPHO_HEADGROUP_SMARTS))
            or neutral.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OX2,NX3]"))
        )
    )
    charge = int(Chem.GetFormalCharge(major)) if major is not None else 0
    mass = float(Descriptors.MolWt(neutral)) if neutral is not None else 0.0
    n_heavy = sum(v for k, v in atom_bag.items() if k not in ("H", "e-"))
    strata = {
        "pka_count": f"n_pka_{len(cache_pkas)}",
        "group": _primary_group(groups),
        "charge": _charge_bin(charge),
        "mass": _mass_bin(mass),
        "lipid": "lipid_like" if lipid_like else "not_lipid_like",
    }
    return ValidationCompound(
        compound_id=compound_id,
        inchi_key=inchi_key,
        major_ms_smiles=major_ms_smiles,
        neutral_smiles=neutral_smiles,
        formula_atom_bag=atom_bag,
        cache_pkas=cache_pkas,
        n_cache_pkas=len(cache_pkas),
        groups=groups,
        n_heavy_atoms=n_heavy,
        has_magnesium_constants=has_mg,
        stratum=f"n_pka_{len(cache_pkas)}",
        net_charge_ph7=charge,
        molecular_weight=mass,
        lipid_like=lipid_like,
        strata=strata,
    )


def cmd_sample(args: argparse.Namespace) -> None:
    """Draw the stratified selection set, with the seed-17 300 pinned in.

    The draw is a quota fill, not a shuffle. A plain random sample of the cache is ~0.1
    percent lipid-like and would answer nothing about the regime the fill-in runs in, so a
    share of the set is reserved for that stratum and the rest is spread evenly over the
    30 (pKa count x primary group) cells. Whatever a cell cannot supply is refilled from
    the general pool, and the realized counts per axis are written out rather than the
    intended ones.
    """
    from equilibrator_cache.api import create_compound_cache_from_zenodo
    from equilibrator_cache.models import Compound
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    pinned_ids: set[int] = set()
    if osp.isfile(SEED17_PATH):
        with open(SEED17_PATH) as handle:
            pinned_ids = {
                c["compound_id"]
                for c in json.load(handle)["compounds"]
            }
    print(f"pinning {len(pinned_ids)} compounds from {SEED17_PATH}")

    ccache = create_compound_cache_from_zenodo()
    query = (
        ccache.session.query(Compound)
        .filter(
            Compound.smiles.isnot(None),
            Compound.group_vector.isnot(None),
            Compound.dissociation_constants.isnot(None),
            Compound.mass < args.max_mass,
        )
        .order_by(Compound.id)
    )

    pool: list[ValidationCompound] = []
    pinned: list[ValidationCompound] = []
    n_scanned = 0
    for compound in query.yield_per(2000):
        n_scanned += 1
        if n_scanned > args.scan_limit:
            break
        pkas = sorted(
            (float(p) for p in (compound.dissociation_constants or []) if MIN_PH < float(p) < MAX_PH),
            reverse=True,
        )
        if not 1 <= len(pkas) <= MAX_PKAS:
            continue
        atom_bag = dict(compound.atom_bag or {})
        elements = {k for k in atom_bag if k != "e-"}
        if not elements or not elements <= ORGANIC_ELEMENTS:
            continue
        n_heavy = sum(v for k, v in atom_bag.items() if k not in ("H", "e-"))
        if n_heavy > args.max_heavy_atoms or n_heavy < 3:
            continue
        neutral = _neutralize(compound.smiles)
        if neutral is None:
            continue
        record = _describe(
            compound.id,
            compound.inchi_key or "",
            compound.smiles,
            neutral,
            atom_bag,
            pkas,
            bool(compound.magnesium_dissociation_constants),
        )
        if compound.id in pinned_ids:
            record.in_seed17_subset = True
            pinned.append(record)
        else:
            pool.append(record)
        if n_scanned % 20000 == 0:
            print(
                f"  scanned {n_scanned}, pool {len(pool)}, pinned {len(pinned)}",
                flush=True,
            )

    if len(pinned) != len(pinned_ids):
        # A pinned compound that the widened filter drops would silently change what the
        # earlier numbers were computed on, so it is an error rather than a shorter set.
        missing = pinned_ids - {c.compound_id for c in pinned}
        raise RuntimeError(
            f"{len(missing)} seed-17 compounds did not survive this scan: {sorted(missing)[:10]}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    chosen: list[ValidationCompound] = list(pinned)
    taken: set[int] = {c.compound_id for c in pinned}
    n_new = max(0, args.n - len(chosen))

    # 1. The lipid-like reservation, capped by what the cache actually holds.
    lipids = [c for c in pool if c.lipid_like and c.compound_id not in taken]
    n_lipid_target = int(round(args.lipid_fraction * n_new))
    for record in lipids[:n_lipid_target]:
        chosen.append(record)
        taken.add(record.compound_id)
    print(f"lipid-like: {len(lipids)} available, {min(len(lipids), n_lipid_target)} taken")

    # 2. Even quotas over (pKa count x primary group).
    cells: dict[tuple[str, str], list[ValidationCompound]] = {}
    for record in pool:
        if record.compound_id in taken:
            continue
        cells.setdefault(
            (record.strata["pka_count"], record.strata["group"]), []
        ).append(record)
    remaining = args.n - len(chosen)
    if cells and remaining > 0:
        quota = max(1, remaining // len(cells))
        for key in sorted(cells):
            for record in cells[key][:quota]:
                if len(chosen) >= args.n:
                    break
                chosen.append(record)
                taken.add(record.compound_id)

    # 3. Whatever the quotas could not supply, from the general pool.
    for record in pool:
        if len(chosen) >= args.n:
            break
        if record.compound_id in taken:
            continue
        chosen.append(record)
        taken.add(record.compound_id)

    chosen.sort(key=lambda c: c.compound_id)
    by_axis: dict[str, dict[str, int]] = {}
    for record in chosen:
        for axis, label in record.strata.items():
            by_axis.setdefault(axis, {})[label] = (
                by_axis.setdefault(axis, {}).get(label, 0) + 1
            )
    group_counts: dict[str, int] = {}
    for record in chosen:
        for g in record.groups:
            group_counts[g] = group_counts.get(g, 0) + 1

    vset = ValidationSet(
        built_at=_timestamp(),
        cache_path=str(ccache.session.bind.url),
        n_compounds=len(chosen),
        seed=args.seed,
        n_candidates_scanned=n_scanned,
        n_candidates_eligible=len(pool) + len(pinned),
        stratum_counts=by_axis.get("pka_count", {}),
        group_counts=group_counts,
        n_with_magnesium_constants=sum(1 for c in chosen if c.has_magnesium_constants),
        compounds=chosen,
        pinned_from=SEED17_PATH if pinned else None,
        n_pinned=len(pinned),
        stratum_counts_by_axis=by_axis,
    )
    os.makedirs(RESULTS, exist_ok=True)
    with open(SAMPLE_PATH, "w") as handle:
        handle.write(vset.model_dump_json(indent=1))
    print(f"wrote {SAMPLE_PATH}: n={vset.n_compounds}, pinned={vset.n_pinned}")
    for axis, counts in by_axis.items():
        print(f"  {axis}: {dict(sorted(counts.items()))}")
    print(f"  with Mg constants: {vset.n_with_magnesium_constants}")


def _load_sample() -> ValidationSet:
    with open(SAMPLE_PATH) as handle:
        return ValidationSet.model_validate_json(handle.read())


# =============================================================================== score


def _microspecies(atom_bag: dict[str, int], pkas: list[float], reference_n_h: int):
    """The microspecies of one pKa ladder, referenced to ``reference_n_h``.

    Split out of the transform so a ladder is decomposed ONCE and then evaluated at every
    working pH: ``create_microspecies_mappings`` is the expensive half and it does not
    depend on the pH the transform is taken at.
    """
    from equilibrator_assets.thermodynamics import create_microspecies_mappings
    from equilibrator_cache.models import Compound, CompoundMicrospecies

    compound = Compound(
        id=-1,
        atom_bag=dict(atom_bag),
        dissociation_constants=list(pkas),
        inchi_key="SELECTION-STANDIN-N",
    )
    compound.magnesium_dissociation_constants = []
    mappings = create_microspecies_mappings(compound, mid_ph=MID_PH)
    by_n_h = {m["number_protons"]: m for m in mappings}
    if reference_n_h not in by_n_h:
        return None
    offset = by_n_h[reference_n_h]["ddg_over_rt"]
    return [
        CompoundMicrospecies(
            compound_id=-1,
            charge=m["charge"],
            number_protons=m["number_protons"],
            number_magnesiums=m["number_magnesiums"],
            is_major=m["is_major"],
            ddg_over_rt=m["ddg_over_rt"] - offset,
        )
        for m in mappings
    ]


def _transform_at(microspecies, p_h: float) -> float:
    """The Legendre transform of a decomposed ladder at one pH, kJ/mol."""
    from equilibrator_cache.thermodynamic_constants import R

    ddg_over_rt = [
        -m.transform(
            pH=p_h,
            pMg=P_MG,
            ionic_strength_M=IONIC_STRENGTH_M,
            T_in_K=TEMPERATURE_K,
        )
        for m in microspecies
    ]
    rt = float(R.m_as("kJ/mol/K")) * TEMPERATURE_K
    return -rt * float(np.logaddexp.reduce(np.asarray(ddg_over_rt)))


def _ensemble_ladder(ladders: list[list[float]]) -> list[float]:
    """The majority-rank median of several macroscopic ladders.

    Rank ``i`` survives when more than half the contributing tools produced a pKa at that
    rank; its value is the median over the tools that did. Two tools therefore need both to
    agree that a rank exists and three need two of the three.
    """
    present = [ladder for ladder in ladders if ladder]
    if not present:
        return []
    need = len(ladders) / 2.0
    out: list[float] = []
    for rank in range(max(len(ladder) for ladder in present)):
        values = [ladder[rank] for ladder in present if len(ladder) > rank]
        if len(values) > need:
            out.append(float(np.median(values)))
    return sorted(out, reverse=True)


class MethodScore(BaseModel):
    """One method's scores, overall and per stratum, at every working pH."""

    method: str
    kind: str = Field(description="'tool' or 'ensemble'.")
    members: list[str] = Field(default_factory=list)
    n_attempted: int
    n_ok: int
    failure_rate: float
    frac_count_match: float
    pka_mae_paired: float | None
    pka_mae_n: int
    seconds_per_compound: float | None
    seconds_are_amortized: bool = False
    #: pH -> {mae, median, p90, max, n}, for pH 7.2 and every working compartment pH.
    by_ph: dict[str, dict[str, float]]
    #: The worst per-pH p90 over the working pHs; the statistic the rule is decided on.
    worst_working_p90: float | None
    worst_working_ph: str | None
    #: axis -> stratum -> {n, n_ok, frac_count_match, mae_ph72, p90_worst_working}
    by_stratum: dict[str, dict[str, dict[str, float | int | None]]]


class SelectionScores(BaseModel):
    """The whole method comparison."""

    built_at: str
    validation_set_path: str
    n_compounds: int
    n_pinned: int
    n_heldout: int
    conditions: dict[str, float]
    working_ph: list[float]
    ranking_split: str
    evaluation_split: str
    magnesium_gap: dict[str, float | str]
    methods: list[MethodScore]
    methods_heldout: list[MethodScore]
    ensemble_membership: dict[str, list[str]]
    per_group_best: dict[str, str]


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0.0}
    array = np.abs(np.asarray(values))
    return {
        "n": float(array.size),
        "mae": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def _score_method(
    method: str,
    kind: str,
    members: list[str],
    rows: list[dict],
    seconds_per_compound: float | None,
    amortized: bool,
) -> MethodScore:
    ok_rows = [r for r in rows if r["ok"]]
    by_ph: dict[str, dict[str, float]] = {}
    for p_h in (P_H, *WORKING_PH):
        key = f"{p_h:g}"
        if key in by_ph:
            continue
        by_ph[key] = _stats(
            [r["ddgf"][key] for r in ok_rows if r["ddgf"].get(key) is not None]
        )
    working = {
        f"{p:g}": by_ph[f"{p:g}"] for p in WORKING_PH if by_ph[f"{p:g}"].get("n")
    }
    worst_ph = max(working, key=lambda k: working[k]["p90"]) if working else None
    mae_values = [r["pka_mae"] for r in ok_rows if r["pka_mae"] is not None]

    by_stratum: dict[str, dict[str, dict[str, float | int | None]]] = {}
    axes = sorted({axis for r in rows for axis in r["strata"]})
    for axis in axes:
        labels = sorted({r["strata"][axis] for r in rows if axis in r["strata"]})
        block: dict[str, dict[str, float | int | None]] = {}
        for label in labels:
            subset = [r for r in rows if r["strata"].get(axis) == label]
            sub_ok = [r for r in subset if r["ok"]]
            ph72 = [
                r["ddgf"][f"{P_H:g}"]
                for r in sub_ok
                if r["ddgf"].get(f"{P_H:g}") is not None
            ]
            worst = None
            for p in WORKING_PH:
                vals = [
                    r["ddgf"][f"{p:g}"]
                    for r in sub_ok
                    if r["ddgf"].get(f"{p:g}") is not None
                ]
                if vals:
                    p90 = float(np.percentile(np.abs(np.asarray(vals)), 90))
                    worst = p90 if worst is None else max(worst, p90)
            block[label] = {
                "n": len(subset),
                "n_ok": len(sub_ok),
                "frac_count_match": (
                    sum(1 for r in sub_ok if r["count_matches"]) / len(sub_ok)
                    if sub_ok
                    else None
                ),
                "mae_ph72": float(np.abs(np.asarray(ph72)).mean()) if ph72 else None,
                "p90_worst_working": worst,
            }
        by_stratum[axis] = block

    return MethodScore(
        method=method,
        kind=kind,
        members=members,
        n_attempted=len(rows),
        n_ok=len(ok_rows),
        failure_rate=1.0 - (len(ok_rows) / len(rows) if rows else 0.0),
        frac_count_match=(
            sum(1 for r in ok_rows if r["count_matches"]) / len(ok_rows)
            if ok_rows
            else 0.0
        ),
        pka_mae_paired=float(np.mean(mae_values)) if mae_values else None,
        pka_mae_n=len(mae_values),
        seconds_per_compound=seconds_per_compound,
        seconds_are_amortized=amortized,
        by_ph=by_ph,
        worst_working_p90=working[worst_ph]["p90"] if worst_ph else None,
        worst_working_ph=worst_ph,
        by_stratum=by_stratum,
    )


def cmd_score(args: argparse.Namespace) -> None:
    """Score every prediction file and every ensemble, at pH 7.2 and each working pH."""
    vset = _load_sample()
    by_id = {c.compound_id: c for c in vset.compounds}

    predictions: dict[str, dict[int, list[float]]] = {}
    failures: dict[str, set[int]] = {}
    seconds: dict[str, tuple[float | None, bool]] = {}
    for tool in args.tools:
        path = PRED_PATH_TEMPLATE.format(tool=tool)
        if not osp.isfile(path):
            print(f"skip {tool}: no {path}")
            continue
        with open(path) as handle:
            pset = PredictionSet.model_validate_json(handle.read())
        ladders: dict[int, list[float]] = {}
        failed: set[int] = set()
        per_compound: list[float] = []
        for record in pset.records:
            if record.compound_id not in by_id:
                continue
            if record.ok and record.predicted_pkas:
                ladders[record.compound_id] = sorted(
                    record.predicted_pkas, reverse=True
                )
            else:
                failed.add(record.compound_id)
            if record.seconds is not None:
                per_compound.append(record.seconds)
        predictions[pset.tool] = ladders
        failures[pset.tool] = failed
        if per_compound:
            seconds[pset.tool] = (float(np.mean(per_compound)), False)
        else:
            n = max(1, pset.n_requested)
            seconds[pset.tool] = (pset.seconds_total / n, True)
        print(f"{pset.tool}: {len(ladders)} ladders, {len(failed)} failures")

    if not predictions:
        raise RuntimeError("no prediction file was found for any requested tool")

    # The reference transforms, one decomposition per compound reused at every pH.
    reference: dict[int, dict[str, float]] = {}
    micro_cache: dict[int, object] = {}
    for i, compound in enumerate(vset.compounds):
        reference_n_h = compound.formula_atom_bag.get("H", 0)
        micro = _microspecies(
            compound.formula_atom_bag, compound.cache_pkas, reference_n_h
        )
        micro_cache[compound.compound_id] = micro
        if micro is None:
            continue
        reference[compound.compound_id] = {
            f"{p:g}": _transform_at(micro, p) for p in (P_H, *WORKING_PH)
        }
        if (i + 1) % 250 == 0:
            print(f"  reference {i + 1}/{len(vset.compounds)}", flush=True)

    def _rows_for(ladders: dict[int, list[float]], failed: set[int]) -> list[dict]:
        out: list[dict] = []
        for compound in vset.compounds:
            cid = compound.compound_id
            if cid not in ladders and cid not in failed:
                continue
            pred = ladders.get(cid, [])
            reference_n_h = compound.formula_atom_bag.get("H", 0)
            ddgf: dict[str, float | None] = {}
            ok = bool(pred) and cid in reference
            if ok:
                micro = _microspecies(
                    compound.formula_atom_bag, pred, reference_n_h
                )
                if micro is None:
                    ok = False
                else:
                    for p in (P_H, *WORKING_PH):
                        key = f"{p:g}"
                        ddgf[key] = _transform_at(micro, p) - reference[cid][key]
            n = min(len(compound.cache_pkas), len(pred))
            pka_mae = (
                float(
                    np.mean(
                        [abs(compound.cache_pkas[i] - pred[i]) for i in range(n)]
                    )
                )
                if n
                else None
            )
            out.append(
                {
                    "compound_id": cid,
                    "inchi_key": compound.inchi_key,
                    "strata": compound.strata,
                    "pinned": compound.in_seed17_subset,
                    "n_cache_pkas": compound.n_cache_pkas,
                    "n_pred_pkas": len(pred),
                    "count_matches": len(pred) == compound.n_cache_pkas,
                    "cache_pkas": compound.cache_pkas,
                    "predicted_pkas": pred,
                    "pka_mae": pka_mae,
                    "ddgf": ddgf,
                    "ok": ok,
                }
            )
        return out

    tool_rows = {
        tool: _rows_for(ladders, failures[tool])
        for tool, ladders in predictions.items()
    }

    # Rank the individual tools on the PINNED subset only, so an ensemble built from that
    # ranking can be scored on rows that had no say in it.
    def _pinned_p90(rows: list[dict]) -> float:
        values: list[float] = []
        for p in WORKING_PH:
            key = f"{p:g}"
            sub = [
                abs(r["ddgf"][key])
                for r in rows
                if r["pinned"] and r["ok"] and r["ddgf"].get(key) is not None
            ]
            if sub:
                values.append(float(np.percentile(np.asarray(sub), 90)))
        return max(values) if values else float("inf")

    ranked = sorted(predictions, key=lambda t: _pinned_p90(tool_rows[t]))
    print("ranking on the pinned subset (worst working-pH p90):")
    for tool in ranked:
        print(f"  {tool:26s} {_pinned_p90(tool_rows[tool]):8.3f}")

    ensemble_membership: dict[str, list[str]] = {}
    if len(ranked) >= 2:
        ensemble_membership["ensemble_top2"] = ranked[:2]
        ensemble_membership["cascade_top2"] = ranked[:2]
    if len(ranked) >= 3:
        ensemble_membership["ensemble_top3"] = ranked[:3]
        ensemble_membership["cascade_top3"] = ranked[:3]

    # The per-group rule, also fitted on the pinned subset. A group the pinned subset does
    # not populate falls back to the OVERALL pinned best rather than being dropped: dropping
    # it turns a coverage question into a silent failure, and the 367 compounds carrying
    # none of the four groups are exactly the ones the pinned 300 has no example of.
    per_group_best: dict[str, str] = {}
    group_labels = sorted({c.strata["group"] for c in vset.compounds})
    for label in group_labels:
        best, best_value = None, float("inf")
        for tool in predictions:
            values: list[float] = []
            for p in WORKING_PH:
                key = f"{p:g}"
                sub = [
                    abs(r["ddgf"][key])
                    for r in tool_rows[tool]
                    if r["pinned"]
                    and r["ok"]
                    and r["strata"].get("group") == label
                    and r["ddgf"].get(key) is not None
                ]
                if sub:
                    values.append(float(np.percentile(np.asarray(sub), 90)))
            score = max(values) if values else float("inf")
            if score < best_value:
                best, best_value = tool, score
        per_group_best[label] = best if best is not None else ranked[0]
    print(f"per-group best (pinned subset): {per_group_best}")

    ensembles: dict[str, dict[int, list[float]]] = {}
    for name, members in ensemble_membership.items():
        ensembles[name] = {}
        cascade = name.startswith("cascade_")
        for compound in vset.compounds:
            cid = compound.compound_id
            if cascade:
                # The database-first pattern applied to predictors: the best tool where it
                # answers, the next where it does not. It closes the coverage half of the
                # problem, which averaging cannot: a median over tools is undefined exactly
                # where the best tool returned nothing.
                merged: list[float] = []
                for tool in members:
                    ladder = predictions[tool].get(cid, [])
                    if ladder:
                        merged = list(ladder)
                        break
            else:
                merged = _ensemble_ladder([predictions[t].get(cid, []) for t in members])
            if merged:
                ensembles[name][cid] = merged
    if per_group_best:
        ensembles["per_group_best"] = {}
        for compound in vset.compounds:
            tool = per_group_best.get(compound.strata["group"])
            if tool is None:
                continue
            ladder = predictions[tool].get(compound.compound_id)
            if ladder:
                ensembles["per_group_best"][compound.compound_id] = ladder

    all_ids = {c.compound_id for c in vset.compounds}
    ensemble_rows = {
        name: _rows_for(ladders, all_ids - set(ladders))
        for name, ladders in ensembles.items()
    }

    def _seconds_for(name: str) -> tuple[float | None, bool]:
        if name in seconds:
            return seconds[name]
        members = ensemble_membership.get(name) or list(predictions)
        parts = [seconds[m][0] for m in members if seconds.get(m, (None, False))[0]]
        amortized = any(seconds[m][1] for m in members if m in seconds)
        return (float(sum(parts)) if parts else None, amortized)

    methods: list[MethodScore] = []
    methods_heldout: list[MethodScore] = []
    for name, rows in list(tool_rows.items()) + list(ensemble_rows.items()):
        kind = "tool" if name in tool_rows else "ensemble"
        members = ensemble_membership.get(name, [])
        if name == "per_group_best":
            members = sorted(set(per_group_best.values()))
        sec, amortized = _seconds_for(name)
        methods.append(_score_method(name, kind, members, rows, sec, amortized))
        heldout = [r for r in rows if not r["pinned"]]
        if heldout:
            methods_heldout.append(
                _score_method(name, kind, members, heldout, sec, amortized)
            )

    result = SelectionScores(
        built_at=_timestamp(),
        validation_set_path=SAMPLE_PATH,
        n_compounds=vset.n_compounds,
        n_pinned=vset.n_pinned,
        n_heldout=vset.n_compounds - vset.n_pinned,
        conditions={
            "p_h_reference": P_H,
            "ionic_strength_m": IONIC_STRENGTH_M,
            "temperature_k": TEMPERATURE_K,
            "p_mg": P_MG,
        },
        working_ph=list(WORKING_PH),
        ranking_split=(
            "the pinned seed-17 300, which is where the ensemble membership and the "
            "per-group rule are fitted"
        ),
        evaluation_split=(
            "methods_heldout scores the compounds that are NOT pinned, so an ensemble is "
            "never scored on the rows that chose it"
        ),
        magnesium_gap={
            "n_sampled_with_magnesium_constants": float(
                vset.n_with_magnesium_constants
            ),
            "note": (
                "specified_pkas carries proton dissociation constants only, so every"
                " transform here is proton-only on both sides; a compound whose cache"
                " entry has magnesium constants keeps an unclosable gap on this route"
            ),
        },
        methods=methods,
        methods_heldout=methods_heldout,
        ensemble_membership=ensemble_membership,
        per_group_best=per_group_best,
    )
    with open(SCORES_PATH, "w") as handle:
        handle.write(result.model_dump_json(indent=1))

    ph_keys = [f"{P_H:g}"] + [f"{p:g}" for p in WORKING_PH]
    with open(PER_COMPOUND_CSV, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "compound_id",
                "inchi_key",
                "pinned",
                "pka_count",
                "group",
                "charge",
                "mass",
                "lipid",
                "n_cache_pkas",
                "n_pred_pkas",
                "count_matches",
                "cache_pkas",
                "predicted_pkas",
                "pka_mae",
                "ok",
                *[f"ddgf_ph{k}" for k in ph_keys],
            ]
        )
        for name, rows in list(tool_rows.items()) + list(ensemble_rows.items()):
            for row in rows:
                writer.writerow(
                    [
                        name,
                        row["compound_id"],
                        row["inchi_key"],
                        row["pinned"],
                        row["strata"].get("pka_count", ""),
                        row["strata"].get("group", ""),
                        row["strata"].get("charge", ""),
                        row["strata"].get("mass", ""),
                        row["strata"].get("lipid", ""),
                        row["n_cache_pkas"],
                        row["n_pred_pkas"],
                        row["count_matches"],
                        ";".join(f"{p:.2f}" for p in row["cache_pkas"]),
                        ";".join(f"{p:.2f}" for p in row["predicted_pkas"]),
                        "" if row["pka_mae"] is None else f"{row['pka_mae']:.4f}",
                        row["ok"],
                        *[
                            ""
                            if row["ddgf"].get(k) is None
                            else f"{row['ddgf'][k]:.4f}"
                            for k in ph_keys
                        ],
                    ]
                )
    print(f"wrote {SCORES_PATH} and {PER_COMPOUND_CSV}")
    for score in result.methods_heldout:
        print(
            f"  {score.method:26s} heldout n={score.n_attempted:5d}"
            f" fail={score.failure_rate:6.3f}"
            f" worst-working p90={_fmt(score.worst_working_p90)}"
            f" at pH {score.worst_working_ph}"
        )


# ============================================================================== decide


class SelectionRule(BaseModel):
    """The rule, written down before the numbers so it cannot be fitted to them."""

    statistic: str = (
        "the worst per-pH p90 of |dGf'0(predicted) - dGf'0(cache)| over the working "
        "compartment pHs"
    )
    why_p90: str = (
        "the fill-in is a tail risk: a method whose median is 0.2 kJ/mol and whose p90 is "
        "8 kJ/mol puts an 8 kJ/mol error on one reaction in ten, and one wrongly signed "
        "reaction potential is what a second-law hinge acts on"
    )
    why_worst_ph: str = (
        "a metabolite is priced at its compartment's pH, so the method has to hold at all "
        "of them; taking the mean over pHs would let a good cytosol score hide a bad "
        "vacuole one"
    )
    max_failure_rate: float = Field(
        default=0.02,
        description="A method that cannot answer is not a cheaper method, it is a gap.",
    )
    evaluated_on: str = (
        "the compounds NOT in the pinned seed-17 300, since that subset fitted the "
        "ensemble membership and the per-group rule"
    )
    tie_breaks: list[str] = Field(
        default_factory=lambda: [
            "median |ddGf'0| at pH 7.2",
            "seconds per compound",
        ]
    )
    ensembles_allowed: bool = True


class MethodSelection(BaseModel):
    """The decision, with every number it was decided on."""

    built_at: str
    rule: SelectionRule
    scores_path: str
    scores_sha256: str
    validation_set_path: str
    n_evaluated: int
    working_ph: list[float]
    selected: str
    selected_kind: str
    selected_members: list[str]
    selected_worst_working_p90: float
    selected_worst_working_ph: str
    selected_failure_rate: float
    selected_median_ph72: float | None
    selected_mae_ph72: float | None
    selected_frac_count_match: float
    selected_seconds_per_compound: float | None
    ranking: list[dict[str, float | str | None]]
    rejected_for_failure_rate: list[str]
    magnesium_gap: dict[str, float | str]
    caveats: list[str]


def _sha256_of(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def cmd_decide(args: argparse.Namespace) -> None:
    """Apply the rule to the scored JSON and write the decision as a record."""
    with open(SCORES_PATH) as handle:
        scores = SelectionScores.model_validate_json(handle.read())
    rule = SelectionRule(max_failure_rate=args.max_failure_rate)

    candidates = [m for m in scores.methods_heldout if m.worst_working_p90 is not None]
    eligible = [m for m in candidates if m.failure_rate <= rule.max_failure_rate]
    rejected = [
        f"{m.method} (failure rate {m.failure_rate:.3f})"
        for m in candidates
        if m.failure_rate > rule.max_failure_rate
    ]
    if not eligible:
        raise RuntimeError(
            "no method meets the failure-rate bound; the rule refuses to pick one rather "
            "than relaxing itself"
        )

    def _key(m: MethodScore) -> tuple[float, float, float]:
        median = m.by_ph.get(f"{P_H:g}", {}).get("median", float("inf"))
        return (
            float(m.worst_working_p90 or float("inf")),
            float(median),
            float(m.seconds_per_compound or float("inf")),
        )

    eligible.sort(key=_key)
    best = eligible[0]
    ranking = [
        {
            "method": m.method,
            "kind": m.kind,
            "worst_working_p90": m.worst_working_p90,
            "worst_working_ph": m.worst_working_ph,
            "mae_ph72": m.by_ph.get(f"{P_H:g}", {}).get("mae"),
            "median_ph72": m.by_ph.get(f"{P_H:g}", {}).get("median"),
            "max_ph72": m.by_ph.get(f"{P_H:g}", {}).get("max"),
            "failure_rate": m.failure_rate,
            "frac_count_match": m.frac_count_match,
            "pka_mae_paired": m.pka_mae_paired,
            "seconds_per_compound": m.seconds_per_compound,
        }
        for m in sorted(candidates, key=_key)
    ]

    decision = MethodSelection(
        built_at=_timestamp(),
        rule=rule,
        scores_path=SCORES_PATH,
        scores_sha256=_sha256_of(SCORES_PATH),
        validation_set_path=scores.validation_set_path,
        n_evaluated=best.n_attempted,
        working_ph=scores.working_ph,
        selected=best.method,
        selected_kind=best.kind,
        selected_members=best.members,
        selected_worst_working_p90=float(best.worst_working_p90 or 0.0),
        selected_worst_working_ph=str(best.worst_working_ph),
        selected_failure_rate=best.failure_rate,
        selected_median_ph72=best.by_ph.get(f"{P_H:g}", {}).get("median"),
        selected_mae_ph72=best.by_ph.get(f"{P_H:g}", {}).get("mae"),
        selected_frac_count_match=best.frac_count_match,
        selected_seconds_per_compound=best.seconds_per_compound,
        ranking=ranking,
        rejected_for_failure_rate=rejected,
        magnesium_gap=scores.magnesium_gap,
        caveats=[
            "The ground truth is ChemAxon's own pKa list as stored in the eQuilibrator "
            "cache, so this measures agreement with the convention the cache is built in, "
            "not agreement with experiment.",
            "Magnesium dissociation constants are absent from every external route, so a "
            "compound that binds magnesium keeps a gap this benchmark cannot close.",
            "pKa-count match is reported and is NOT part of the rule: a ladder can carry "
            "the right number of pKas and the wrong values, and the transform is what the "
            "flux layer consumes.",
        ],
    )
    with open(DECISION_PATH, "w") as handle:
        handle.write(decision.model_dump_json(indent=1))
    print(f"wrote {DECISION_PATH}")
    print(
        f"selected {decision.selected} ({decision.selected_kind}): worst working-pH p90 "
        f"{decision.selected_worst_working_p90:.3f} kJ/mol at pH "
        f"{decision.selected_worst_working_ph}, failure rate "
        f"{decision.selected_failure_rate:.4f}"
    )
    for row in ranking:
        print(
            f"  {str(row['method']):26s} p90 {_fmt(row['worst_working_p90'])}"
            f" @pH {row['worst_working_ph']}  median72 {_fmt(row['median_ph72'])}"
            f"  fail {_fmt(row['failure_rate'], 4)}"
        )


# =============================================================================== table


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def cmd_table(args: argparse.Namespace) -> None:
    """Print the scored JSON as the markdown tables the note carries."""
    with open(SCORES_PATH) as handle:
        scores = SelectionScores.model_validate_json(handle.read())
    split = scores.methods_heldout if args.heldout else scores.methods
    label = "held-out" if args.heldout else "whole set"

    print(
        f"Scored at {scores.built_at} over {scores.n_compounds} compounds"
        f" ({scores.n_pinned} pinned from the seed-17 set,"
        f" {scores.n_heldout} held out); table below is the {label}."
        f" I = {scores.conditions['ionic_strength_m']:g} M,"
        f" T = {scores.conditions['temperature_k']:g} K,"
        f" pMg {scores.conditions['p_mg']:g}."
    )
    print()
    print(
        "| method | kind | n | failure rate | right pKa count | pKa MAE |"
        " dGf'0 MAE pH 7.2 | median | p90 | max | worst working p90 | at pH | s/compound |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in sorted(split, key=lambda s: s.worst_working_p90 or 1e9):
        block = m.by_ph.get(f"{P_H:g}", {})
        print(
            f"| {m.method} | {m.kind} | {m.n_attempted} | {_fmt(m.failure_rate, 4)} |"
            f" {_fmt(m.frac_count_match)} | {_fmt(m.pka_mae_paired)} |"
            f" {_fmt(block.get('mae'))} | {_fmt(block.get('median'))} |"
            f" {_fmt(block.get('p90'))} | {_fmt(block.get('max'), 1)} |"
            f" {_fmt(m.worst_working_p90)} | {m.worst_working_ph} |"
            f" {_fmt(m.seconds_per_compound, 2)} |"
        )

    print()
    print("**dGf'0 p90 (kJ/mol) at each working compartment pH**")
    print()
    header = " | ".join(f"pH {p:g}" for p in scores.working_ph)
    print(f"| method | {header} |")
    print("|---" * (len(scores.working_ph) + 1) + "|")
    for m in sorted(split, key=lambda s: s.worst_working_p90 or 1e9):
        cells = " | ".join(
            _fmt(m.by_ph.get(f"{p:g}", {}).get("p90")) for p in scores.working_ph
        )
        print(f"| {m.method} | {cells} |")

    for axis in ("pka_count", "group", "charge", "mass", "lipid"):
        print()
        print(f"**By {axis}**")
        print()
        print(
            "| stratum | method | n | n scored | right pKa count | MAE pH 7.2 |"
            " worst working p90 |"
        )
        print("|---|---|---|---|---|---|---|")
        labels = sorted({k for m in split for k in m.by_stratum.get(axis, {})})
        for key in labels:
            for m in sorted(split, key=lambda s: s.method):
                block = m.by_stratum.get(axis, {}).get(key)
                if block is None:
                    continue
                print(
                    f"| {key} | {m.method} | {block['n']} | {block['n_ok']} |"
                    f" {_fmt(block['frac_count_match'])} | {_fmt(block['mae_ph72'])} |"
                    f" {_fmt(block['p90_worst_working'])} |"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_sample = sub.add_parser("sample", help="draw the stratified selection set")
    p_sample.add_argument("--n", type=int, default=2000)
    p_sample.add_argument("--seed", type=int, default=17)
    p_sample.add_argument(
        "--max-mass",
        type=float,
        default=1600.0,
        help="the 300-compound sampler capped at 600 Da, which excludes every lipid",
    )
    p_sample.add_argument("--max-heavy-atoms", type=int, default=120)
    p_sample.add_argument(
        "--lipid-fraction",
        type=float,
        default=0.25,
        help="share of the NEW compounds reserved for the lipid-like stratum",
    )
    p_sample.add_argument("--scan-limit", type=int, default=1000000)
    p_sample.set_defaults(func=cmd_sample)

    p_score = sub.add_parser("score", help="score every tool and ensemble")
    p_score.add_argument(
        "--tools",
        nargs="+",
        default=[
            "molgpka",
            "pkasolver",
            "unipka-simple-template",
            "qupkake-truncated",
        ],
    )
    p_score.set_defaults(func=cmd_score)

    p_decide = sub.add_parser("decide", help="apply the selection rule")
    p_decide.add_argument("--max-failure-rate", type=float, default=0.02)
    p_decide.set_defaults(func=cmd_decide)

    p_table = sub.add_parser("table", help="markdown tables from the scored JSON")
    p_table.add_argument("--heldout", action="store_true")
    p_table.set_defaults(func=cmd_table)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
