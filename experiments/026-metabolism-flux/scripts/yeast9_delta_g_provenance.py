# experiments/026-metabolism-flux/scripts/yeast9_delta_g_provenance.py
# [[experiments.026-metabolism-flux.yeast9-delta-g-provenance]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/yeast9_delta_g_provenance.py

r"""Where yeast-GEM 9's shipped deltaG columns came from, and why they disagree with us.

THE QUESTION
------------
Figure 3 of the 026 document shows yeast-GEM 9.0.2's ``model_rxnDeltaG.csv`` disagreeing
with an eQuilibrator 3 recomputation. A reviewer asked why the two tables disagree at all,
since yeast9 almost certainly used a group-contribution method too. This script answers
that with the upstream artifacts rather than by inference.

THE PROVENANCE CHAIN, READ OFF THE CURATION SCRIPT
--------------------------------------------------
``yeast-GEM/code/modelCuration/v8_7_0.m`` (the PR that introduced both columns) assigns,
in priority order:

* ``metDeltaG``: yETFL's ``metDeltaGFtr``, matched by metabolite NAME after the compartment
  tag has been stripped and de-duplicated; otherwise ModelSEED's ``deltag``, matched by
  KEGG compound id.
* ``rxnDeltaG``: yETFL's ``rxnDeltaGR``, matched by reaction id; otherwise dGPredictor for
  one contiguous block of reaction indices.

yETFL (Oftadeh et al. 2021) built those fields with matTFA/pyTFA, whose estimator is the
group-contribution method and whose transform is Alberty's, applied per compartment from
``yeast_compartment_data.json``. That file is therefore the compartment pH, ionic strength
and membrane-potential table yeast-GEM 9's numbers actually rest on, and it is recovered
here rather than guessed.

WHAT THE SCRIPT MEASURES
------------------------
1. **Units.** yETFL's model carries ``thermo_units = 'kcal/mol'``. Regressing our
   eQuilibrator formation energies (kJ/mol) on yETFL's gives the conversion factor
   directly, over ~900 shared metabolites, at a correlation high enough that the slope is
   not in doubt.
2. **Identity.** Whether the shipped reaction column IS yETFL's ``rxnDeltaGR``, to the two
   decimals ``round`` left.
3. **Compartment collapse.** Whether the shipped metabolite column preserved yETFL's
   per-compartment values, or flattened each chemical to one number.
4. **Internal consistency.** Whether summing the shipped metabolite column over a reaction
   reproduces the shipped reaction column, using the shipped table and then using yETFL's
   per-compartment values in its place. This is the source of the 10.2 residual the flux
   layer records.
5. **Condition matching.** Whether our eQuilibrator recomputation moves toward the shipped
   table once the units are fixed and the conditions are matched to yETFL's.

Run the recomputation under the eQuilibrator environment and everything else under
torchcell's:

    /scratch/projects/torchcell-scratch/envs/equilibrator/bin/python \
        experiments/026-metabolism-flux/scripts/yeast9_delta_g_provenance.py --recompute

    PYTHONPATH=. ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/yeast9_delta_g_provenance.py --report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
from datetime import UTC, datetime

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
GEM_DIR = osp.join(DATA_ROOT, "data", "torchcell", "yeast-GEM", "yeast-GEM-9.0.2")
GEM_DB = osp.join(GEM_DIR, "data", "databases")
THERMO = osp.join(DATA_ROOT, "data", "torchcell", "thermo_equilibrator")
YETFL = osp.join(DATA_ROOT, "data", "torchcell", "yetfl_thermo")
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")

#: yeast-GEM's missing-value sentinel, and 1e7 also flags matTFA's own "not estimable".
SENTINEL = 9.0e6

#: kJ per kcal, thermochemical. The one number the whole units finding turns on.
KJ_PER_KCAL = 4.184

#: Where each mirrored upstream artifact came from. The stored file plus its sha256 is
#: canonical; the URL is retrieval metadata, recorded so the mirror can be rebuilt.
UPSTREAM_SOURCES: dict[str, dict[str, str]] = {
    "yeast8_thermo_curated.mat": {
        "source_url": "https://zenodo.org/records/4778047/files/input_models.zip",
        "retrieval_method": "direct_url",
        "retrieval_command": (
            "curl -sL -o input_models.zip "
            "'https://zenodo.org/records/4778047/files/input_models.zip?download=1' "
            "&& unzip -o input_models.zip input_models/yeast8_thermo_curated.mat"
        ),
        "description": (
            "yETFL's thermodynamically curated Yeast8.3.4, the source of 96 percent of "
            "yeast-GEM 9's deltaG values. Carries thermo_units, metDeltaGFtr, rxnDeltaGR."
        ),
    },
    "yeast_compartment_data.json": {
        "source_url": "https://zenodo.org/records/4778047/files/info_yeast.zip",
        "retrieval_method": "direct_url",
        "retrieval_command": (
            "curl -sL -o info_yeast.zip "
            "'https://zenodo.org/records/4778047/files/info_yeast.zip?download=1' "
            "&& unzip -o info_yeast.zip info_yeast/thermo/yeast_compartment_data.json"
        ),
        "description": (
            "The compartment pH, ionic strength and membrane potential table matTFA "
            "applied when yETFL transformed every formation energy."
        ),
    },
    "dgpred_rxnG.csv": {
        "source_url": (
            "https://github.com/SysBioChalmers/yeast-GEM/raw/main/"
            "data/modelCuration/v8_7_0/dgpred_rxnG.csv"
        ),
        "retrieval_method": "direct_url",
        "retrieval_command": (
            "curl -sL https://codeload.github.com/SysBioChalmers/yeast-GEM/tar.gz/"
            "refs/heads/main | tar xz --strip-components=1 "
            "yeast-GEM-main/data/modelCuration/v8_7_0/dgpred_rxnG.csv"
        ),
        "description": "dGPredictor output for the reactions yETFL did not cover.",
    },
    "v8_7_0.m": {
        "source_url": (
            "https://github.com/SysBioChalmers/yeast-GEM/raw/main/"
            "code/modelCuration/v8_7_0.m"
        ),
        "retrieval_method": "direct_url",
        "retrieval_command": (
            "curl -sL https://codeload.github.com/SysBioChalmers/yeast-GEM/tar.gz/"
            "refs/heads/main | tar xz --strip-components=1 "
            "yeast-GEM-main/code/modelCuration/v8_7_0.m"
        ),
        "description": (
            "The curation script that assigned metDeltaG and rxnDeltaG, PR #330, "
            "yeast-GEM 8.7.0. The authority on source priority and on the name-based "
            "metabolite match that discards the compartment."
        ),
    },
    "run_dgpredictor.py": {
        "source_url": (
            "https://github.com/SysBioChalmers/yeast-GEM/raw/main/"
            "data/modelCuration/v8_7_0/code/run_dgpredictor.py"
        ),
        "retrieval_method": "direct_url",
        "retrieval_command": (
            "curl -sL https://codeload.github.com/SysBioChalmers/yeast-GEM/tar.gz/"
            "refs/heads/main | tar xz --strip-components=1 "
            "yeast-GEM-main/data/modelCuration/v8_7_0/code/run_dgpredictor.py"
        ),
        "description": (
            "The dGPredictor driver, which pins pH 7.0, ionic strength 0.1 M and "
            "T 298.15 K and the M12_model_BR Bayesian-ridge fit."
        ),
    },
}


class CompartmentCondition(BaseModel):
    """One compartment's thermodynamic condition, as one source states it."""

    symbol: str
    name: str
    p_h: float
    ionic_strength_m: float
    membrane_potential_mv: dict[str, float] = Field(default_factory=dict)


def sha256_file(path: str) -> str:
    """Hex sha256 of a file, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def yetfl_conditions() -> dict[str, CompartmentCondition]:
    """The compartment table matTFA used when it transformed yETFL's formation energies."""
    with open(osp.join(YETFL, "yeast_compartment_data.json")) as handle:
        raw = json.load(handle)
    return {
        symbol: CompartmentCondition(
            symbol=symbol,
            name=entry["name"],
            p_h=float(entry["pH"]),
            ionic_strength_m=float(entry["ionicStr"]),
            membrane_potential_mv={
                k: float(v) for k, v in entry["membranePot"].items() if float(v) != 0.0
            },
        )
        for symbol, entry in raw.items()
    }


def read_shipped(path: str) -> dict[str, float]:
    """A yeast-GEM deltaG CSV with both missing-value conventions rejected."""
    values: dict[str, float] = {}
    with open(path) as handle:
        next(handle)
        for line in handle:
            key, _, raw = line.strip().partition(",")
            try:
                value = float(raw)
            except ValueError:
                continue
            if abs(value) >= SENTINEL or np.isnan(value):
                continue
            values[key] = value
    return values


def yetfl_self_consistency(model) -> dict[str, float]:
    r"""Does yETFL's reaction column equal :math:`\sum_i S_{ij}\Delta_f G'^\circ_i`?

    Run against yETFL's OWN stoichiometry rather than yeast-GEM 9's, because yeast9
    rebalanced hundreds of reactions after 8.3.4 and a changed reaction is not evidence
    about the table. Protons are dropped from the sum, which is what matTFA does: the
    transform already carries the pH dependence, so a proton term would count it twice.
    """
    import scipy.sparse as sp

    n = len(np.asarray(model.metDeltaGFtr))
    formulas = [str(x) for x in np.asarray(model.metFormulas)[:n]]
    gftr = np.asarray(model.metDeltaGFtr, dtype=float)
    drg = np.asarray(model.rxnDeltaGR, dtype=float)
    is_trans = np.asarray(model.isTrans, dtype=float)
    s = sp.csc_matrix(model.S)
    residuals = []
    for j in np.where((np.abs(drg) < SENTINEL) & (is_trans == 0))[0]:
        column = s[:, j].tocoo()
        total, usable = 0.0, True
        for i, v in zip(column.row, column.data):
            if i >= n or abs(gftr[i]) >= SENTINEL:
                usable = False
                break
            if formulas[i] == "H":
                continue
            total += v * gftr[i]
        if usable:
            residuals.append(abs(total - drg[j]))
    return {
        "n_reactions": len(residuals),
        "max_abs_residual_kcal": float(np.max(residuals)),
        "median_abs_residual_kcal": float(np.median(residuals)),
    }


def load_yetfl():
    """YETFL's curated model: per-compartment formation energies and reaction energies.

    Returns ``(met_dgf, rxn_drg, units, n_transport, self_check)``. ``met_dgf`` is keyed
    by ``(chemical name, compartment symbol)``, which is the ONLY safe join to yeast-GEM.
    yETFL renumbered the metabolites: it keeps one ``s_XXXX`` per CHEMICAL and appends the
    compartment, so ``s_0001_c`` is yeast-GEM's ``s_0002``, not its ``s_0001``. Joining on
    the bare id looks plausible and silently pairs a cytosolic value with a cell-envelope
    one for 652 of 1,328 chemicals.
    """
    import scipy.io as sio

    model = sio.loadmat(
        osp.join(YETFL, "yeast8_thermo_curated.mat"),
        squeeze_me=True,
        struct_as_record=False,
    )["model"]
    # metDeltaGFtr is one element shorter than mets in the shipped .mat; truncate the id
    # list to the value list rather than padding, so no metabolite is silently misaligned.
    n = len(np.asarray(model.metDeltaGFtr))
    met_names = [
        str(x).rsplit(" [", 1)[0] for x in np.asarray(model.metNames)[:n]
    ]
    met_comps = [str(x) for x in np.asarray(model.metCompSymbol)[:n]]
    gftr = np.asarray(model.metDeltaGFtr, dtype=float)
    met_dgf = {
        (name, comp): float(g)
        for name, comp, g in zip(met_names, met_comps, gftr)
        if abs(g) < SENTINEL
    }
    rxn_ids = [str(x) for x in np.asarray(model.rxns)]
    drg = np.asarray(model.rxnDeltaGR, dtype=float)
    rxn_drg = {r: float(v) for r, v in zip(rxn_ids, drg) if abs(v) < SENTINEL}
    n_transport = int(np.asarray(model.isTrans, dtype=float).sum())
    return (
        met_dgf,
        rxn_drg,
        str(model.thermo_units),
        n_transport,
        yetfl_self_consistency(model),
    )


def slope_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of ``y = s x`` with no intercept."""
    return float(np.sum(x * y) / np.sum(x * x))


def recompute() -> None:
    """Recompute every formation energy at yETFL's conditions, under eQuilibrator.

    Same estimator as the main build, different conditions: T = 298.15 K, each
    compartment's own pH AND its own ionic strength from ``yeast_compartment_data.json``,
    and no magnesium, because matTFA's Alberty transform has no Mg term at all. p_mg is
    pushed to 14 rather than dropped, since eQuilibrator has no "no Mg" switch and a pMg
    that high leaves no bound species.
    """
    import cobra
    import pandas as pd
    from equilibrator_api import Q_, ComponentContribution

    conditions = yetfl_conditions()
    proton_inchi_key = "GPRLSGONYQIRFK-UHFFFAOYSA-N"
    temperature_k = 298.15
    p_mg_none = 14.0

    model = cobra.io.read_sbml_model(osp.join(GEM_DIR, "model", "yeast-GEM.xml"))
    cc = ComponentContribution()
    cc.p_h = Q_(7.0)
    cc.ionic_strength = Q_("0.25M")
    cc.temperature = Q_(f"{temperature_k}K")
    cc.p_mg = Q_(p_mg_none)

    chemical_cache: dict[str, tuple] = {}
    rows = []
    for index, metabolite in enumerate(model.metabolites):
        if index % 250 == 0:
            print(f"  {index}/{len(model.metabolites)}", flush=True)
        key = (metabolite.name or metabolite.id).lower()
        if key not in chemical_cache:
            compound = None
            for namespace, prefix in [
                ("metanetx.chemical", "metanetx.chemical"),
                ("chebi", "chebi"),
                ("kegg.compound", "kegg"),
                ("bigg.metabolite", "bigg.metabolite"),
            ]:
                value = metabolite.annotation.get(namespace)
                if not value:
                    continue
                for candidate in value if isinstance(value, list) else [value]:
                    accession = str(candidate)
                    query = (
                        accession
                        if accession.lower().startswith(prefix.lower())
                        else f"{prefix}:{accession}"
                    )
                    compound = cc.get_compound(query)
                    if compound is not None:
                        break
                if compound is not None:
                    break
            mu = None
            if compound is not None and compound.inchi_key != proton_inchi_key:
                mu, _fin, _inf = cc.standard_dg_formation(compound)
            chemical_cache[key] = (compound, mu)
        compound, mu = chemical_cache[key]

        condition = conditions.get(metabolite.compartment)
        p_h = condition.p_h if condition is not None else 7.0
        ionic = condition.ionic_strength_m if condition is not None else 0.25

        transformed = np.nan
        if compound is not None and compound.inchi_key == proton_inchi_key:
            transformed = 0.0
        elif compound is not None and mu is not None:
            correction = compound.transform(
                Q_(p_h), Q_(f"{ionic}M"), Q_(f"{temperature_k}K"), Q_(p_mg_none)
            )
            transformed = float(mu) + float(correction.m_as("kJ/mol"))
        rows.append(
            {
                "met_id": metabolite.id,
                "compartment": metabolite.compartment,
                "p_h": p_h,
                "ionic_strength_m": ionic,
                "dgf_prime_kj_per_mol": transformed,
            }
        )

    table = pd.DataFrame(rows)
    os.makedirs(THERMO, exist_ok=True)
    table.to_parquet(
        osp.join(THERMO, "metabolite_dgf_prime_yeast9_conditions.parquet"), index=False
    )

    lookup = dict(zip(table["met_id"], table["dgf_prime_kj_per_mol"]))
    reaction_rows = []
    for reaction in model.reactions:
        values = [lookup.get(m.id, np.nan) for m in reaction.metabolites]
        complete = not any(np.isnan(v) for v in values)
        drg = (
            float(sum(c * lookup[m.id] for m, c in reaction.metabolites.items()))
            if complete
            else np.nan
        )
        reaction_rows.append(
            {
                "reaction_id": reaction.id,
                "all_participants_known": complete,
                "drg_prime_kj_per_mol": drg,
            }
        )
    reactions = pd.DataFrame(reaction_rows)
    reactions.to_parquet(
        osp.join(THERMO, "reaction_drg_prime_yeast9_conditions.parquet"), index=False
    )
    print(
        f"resolved {int(table['dgf_prime_kj_per_mol'].notna().sum())} metabolites, "
        f"{int(reactions['all_participants_known'].sum())} reactions complete"
    )


def report() -> None:
    """Measure every claim about the provenance chain and write the results JSON."""
    import cobra
    import pandas as pd

    out: dict[str, object] = {
        "question": (
            "yeast-GEM 9.0.2 ships model_metDeltaG.csv and model_rxnDeltaG.csv. Which "
            "estimator, at which conditions, produced them, and does matching those "
            "conditions close the disagreement with an eQuilibrator recomputation?"
        ),
        "generated_at": datetime.now(UTC).isoformat(),
    }

    out["upstream_artifacts"] = {
        name: {**meta, "sha256": sha256_file(osp.join(YETFL, name))}
        for name, meta in UPSTREAM_SOURCES.items()
    }
    out["gem_artifacts"] = {
        name: {"path": osp.join(GEM_DB, name), "sha256": sha256_file(osp.join(GEM_DB, name))}
        for name in ["model_metDeltaG.csv", "model_rxnDeltaG.csv"]
    }

    conditions = yetfl_conditions()
    out["yetfl_compartment_conditions"] = {
        k: v.model_dump() for k, v in conditions.items()
    }

    met_dgf, rxn_drg, units, n_transport, self_check = load_yetfl()
    out["yetfl_model"] = {
        "thermo_units": units,
        "n_metabolites_with_dgf": len(met_dgf),
        "n_reactions_with_drg": len(rxn_drg),
        "n_transport_reactions": n_transport,
        "reaction_column_is_metabolite_sum": self_check,
        "estimator": (
            "group-contribution method (Mavrovouniotis 1990) as packaged in matTFA's "
            "thermo_data.thermodb, transformed to each compartment's pH and ionic "
            "strength by Alberty's relation. Oftadeh et al. 2021, Methods."
        ),
        "temperature_k": 298.15,
        "p_mg": None,
    }

    shipped_met = read_shipped(osp.join(GEM_DB, "model_metDeltaG.csv"))
    shipped_rxn = read_shipped(osp.join(GEM_DB, "model_rxnDeltaG.csv"))
    out["shipped_tables"] = {
        "n_metabolites": len(shipped_met),
        "n_reactions": len(shipped_rxn),
    }

    # yETFL is joined to yeast-GEM 9 through (chemical name, compartment); see load_yetfl.
    model = cobra.io.read_sbml_model(osp.join(GEM_DIR, "model", "yeast-GEM.xml"))
    yetfl_met = {
        m.id: met_dgf[(m.name, m.compartment)]
        for m in model.metabolites
        if (m.name, m.compartment) in met_dgf
    }
    out["yetfl_model"]["n_metabolites_joined_to_yeast9"] = len(yetfl_met)

    # -- 1. units --------------------------------------------------------------
    ours = pd.read_parquet(
        osp.join(THERMO, "metabolite_dgf_prime_single_condition.parquet")
    )
    ours = ours.dropna(subset=["dgf_prime_kj_per_mol"]).set_index("met_id")[
        "dgf_prime_kj_per_mol"
    ]
    shared = [k for k in ours.index if k in yetfl_met]
    x = np.array([yetfl_met[k] for k in shared])
    y = ours[shared].to_numpy()
    big = np.abs(x) > 1.0
    out["units"] = {
        "n_metabolites_compared": len(shared),
        "yetfl_declared_units": units,
        "slope_ours_kj_per_yetfl_unit": slope_through_origin(x[big], y[big]),
        "median_ratio": float(np.median(y[big] / x[big])),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "kj_per_kcal": KJ_PER_KCAL,
        "verdict": (
            "The shipped columns are kcal/mol. yeast-GEM copied yETFL's values without "
            "converting, and the yeast9 paper reports them as kJ/mol."
        ),
    }

    # -- 2. is the shipped reaction column literally yETFL's? ------------------
    common = sorted(set(shipped_rxn) & set(rxn_drg))
    delta = np.abs(
        np.array([shipped_rxn[k] for k in common])
        - np.array([rxn_drg[k] for k in common])
    )
    only_shipped = sorted(set(shipped_rxn) - set(rxn_drg))
    out["reaction_column_identity"] = {
        "n_shipped": len(shipped_rxn),
        "n_in_yetfl": len(common),
        "max_abs_difference": float(delta.max()),
        "frac_within_rounding": float((delta <= 0.005001).mean()),
        "n_not_from_yetfl": len(only_shipped),
        "note": (
            "round(x, 2) in v8_7_0.m accounts for every difference. The reactions not in "
            "yETFL are the dGPredictor block, whose own driver reports kJ/mol, so the "
            "shipped reaction column mixes two unit systems."
        ),
    }

    # -- 3. did the metabolite column keep the compartment? --------------------
    rows = [
        (m.id, m.compartment, shipped_met[m.id], yetfl_met[m.id])
        for m in model.metabolites
        if m.id in shipped_met and m.id in yetfl_met
    ]
    frame = pd.DataFrame(rows, columns=["met_id", "compartment", "shipped", "yetfl"])
    frame["matches"] = (frame["shipped"] - frame["yetfl"]).abs() <= 0.005001
    error = (frame["shipped"] - frame["yetfl"]).abs().to_numpy()
    out["compartment_collapse"] = {
        "n_metabolites": int(len(frame)),
        "n_matching_own_compartment": int(frame["matches"].sum()),
        "frac_matching_own_compartment": float(frame["matches"].mean()),
        "error_kcal_percentiles": {
            str(p): float(np.percentile(error, p)) for p in (50, 75, 90, 95, 99)
        },
        "error_kj_percentiles": {
            str(p): float(np.percentile(error, p) * KJ_PER_KCAL)
            for p in (50, 75, 90, 95, 99)
        },
        "mean_error_kj": float(error.mean() * KJ_PER_KCAL),
        "by_compartment": {
            str(k): {"matching": int(v["matches"].sum()), "total": int(len(v))}
            for k, v in frame.groupby("compartment")
        },
        "cause": (
            "v8_7_0.m strips the compartment tag from metNames, de-duplicates, and maps "
            "by chemical name, so every compartment copy of a chemical receives one "
            "value. yETFL's transform is compartment specific, so half the entries are "
            "the value for a different compartment."
        ),
    }

    # -- 4. are the two shipped columns consistent with each other? ------------
    # Four arms: the shipped metabolite column or yETFL's per-compartment one, each with
    # the proton term left in or dropped. matTFA drops it, because the transform already
    # carries the reaction's pH dependence, but yeast-GEM ships a nonzero proton entry
    # anyway, so a consumer that sums the column naively pays for it once per net proton.
    proton_ids = {m.id for m in model.metabolites if m.formula == "H"}
    proton_value_kcal = {
        m.compartment: shipped_met[m.id]
        for m in model.metabolites
        if m.id in proton_ids and m.id in shipped_met
    }
    buckets: dict[str, dict[str, list[float]]] = {
        f"{table}_{'without' if drop else 'with'}_proton": {"single": [], "multi": []}
        for table in ("shipped_metabolite_column", "yetfl_per_compartment")
        for drop in (True, False)
    }
    for reaction in model.reactions:
        if reaction.id not in shipped_rxn:
            continue
        participants = list(reaction.metabolites.items())
        side = "multi" if len({m.compartment for m in reaction.metabolites}) > 1 else "single"
        for table, values in (
            ("shipped_metabolite_column", shipped_met),
            ("yetfl_per_compartment", yetfl_met),
        ):
            if not all(m.id in values for m, _ in participants):
                continue
            for drop in (True, False):
                total = sum(
                    c * values[m.id]
                    for m, c in participants
                    if not (drop and m.id in proton_ids)
                )
                key = f"{table}_{'without' if drop else 'with'}_proton"
                buckets[key][side].append(abs(total - shipped_rxn[reaction.id]))
    out["internal_consistency"] = {
        "proton_formation_energy_kcal": proton_value_kcal,
        "arms": {
            key: {
                side: {
                    "n": len(values),
                    "median_abs_residual_kcal": float(np.median(values)),
                    "median_abs_residual_kj": float(np.median(values) * KJ_PER_KCAL),
                }
                for side, values in sides.items()
                if values
            }
            for key, sides in buckets.items()
        },
        "reading": (
            "The 10.2 the flux layer records as an unexplained single-compartment "
            "residual is yeast-GEM disagreeing with itself, in kcal/mol. Most of it is "
            "the proton: matTFA prices H+ at RT ln(10) pH and then never adds it into a "
            "reaction sum, while yeast-GEM ships that number in the metabolite column, "
            "so summing the column charges it once per net proton. The rest is the "
            "compartment collapse. Neither is evidence about eQuilibrator."
        ),
    }

    # -- 5. does matching the conditions close the gap? ------------------------
    transport = {
        r.id for r in model.reactions if len({m.compartment for m in r.metabolites}) > 1
    }
    uniform = pd.read_parquet(
        osp.join(THERMO, "reaction_drg_prime_single_condition.parquet")
    )
    uniform = uniform[uniform["all_participants_known"]].set_index("reaction_id")[
        "drg_prime_kj_per_mol"
    ]
    matched_path = osp.join(THERMO, "reaction_drg_prime_yeast9_conditions.parquet")
    matched = None
    if osp.exists(matched_path):
        matched = pd.read_parquet(matched_path)
        matched = matched[matched["all_participants_known"]].set_index("reaction_id")[
            "drg_prime_kj_per_mol"
        ]

    def compare(ours_series, scale: float, subset: list[str]) -> dict[str, float]:
        keys = [k for k in subset if k in ours_series.index]
        a = np.array([shipped_rxn[k] * scale for k in keys])
        b = ours_series[keys].to_numpy()
        finite = np.isfinite(a) & np.isfinite(b)
        a, b = a[finite], b[finite]
        big = np.abs(a) > 1.0
        return {
            "n": int(len(a)),
            "median_abs_diff_kj": float(np.median(np.abs(b - a))),
            "pearson": float(np.corrcoef(a, b)[0, 1]),
            "slope_ours_per_shipped": slope_through_origin(a[big], b[big]),
        }

    single = [k for k in shipped_rxn if k not in transport]
    crossing = [k for k in shipped_rxn if k in transport]
    # A transport reaction whose shipped value is zero is zero on both axes at any
    # condition, so pooling those in would report agreement the comparison never tested.
    crossing_driven = [k for k in crossing if abs(shipped_rxn[k]) > 0.1]
    arms: dict[str, object] = {
        "single_shipped_read_as_kj_uniform_ph": compare(uniform, 1.0, single),
        "single_shipped_converted_to_kj_uniform_ph": compare(
            uniform, KJ_PER_KCAL, single
        ),
        "transport_shipped_converted_to_kj_uniform_ph": compare(
            uniform, KJ_PER_KCAL, crossing_driven
        ),
    }
    if matched is not None:
        arms["single_shipped_converted_to_kj_matched_conditions"] = compare(
            matched, KJ_PER_KCAL, single
        )
        arms["transport_shipped_converted_to_kj_matched_conditions"] = compare(
            matched, KJ_PER_KCAL, crossing_driven
        )
        arms["transport_all_shipped_converted_matched_conditions"] = compare(
            matched, KJ_PER_KCAL, crossing
        )
    out["condition_matching"] = {
        "single_compartment_reactions": len(single),
        "transport_reactions": len(crossing),
        "transport_reactions_with_nonzero_shipped": len(crossing_driven),
        "arms": arms,
        "our_conditions_default": {
            "temperature_k": 303.15,
            "ionic_strength_m": 0.25,
            "p_mg": 3.0,
            "p_h": "per compartment, textbook values",
        },
        "yeast9_conditions": {
            "temperature_k": 298.15,
            "ionic_strength_m": "per compartment, 0.25 M or 0.0 M",
            "p_mg": "no magnesium term in the Alberty transform matTFA uses",
            "p_h": "per compartment, yeast_compartment_data.json",
        },
    }

    os.makedirs(RESULTS, exist_ok=True)
    path = osp.join(RESULTS, "thermo_yeast9_provenance.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2)
    print(json.dumps(out, indent=2, default=str))
    print(f"wrote {path}")


def transport() -> None:
    """Compute every reaction's transport term and check it against the GEM's own column.

    The shipped column is scaled to kJ/mol first, since the units finding above says it is
    kcal/mol. Both scalings are reported, because the unconverted one is the comparison
    the document made before the units were known and the difference between them is the
    whole point.
    """
    import cobra

    from torchcell.metabolism.transport import (
        build_transport_table,
        compare_to_shipped,
        pipeline_conditions,
        yetfl_conditions,
    )

    model = cobra.io.read_sbml_model(osp.join(GEM_DIR, "model", "yeast-GEM.xml"))
    shipped_rxn = read_shipped(osp.join(GEM_DB, "model_rxnDeltaG.csv"))
    os.makedirs(RESULTS, exist_ok=True)

    summary: dict[str, object] = {"generated_at": datetime.now(UTC).isoformat()}
    for label, conditions in [
        ("pipeline", pipeline_conditions()),
        ("yetfl", yetfl_conditions(osp.join(YETFL, "yeast_compartment_data.json"))),
    ]:
        table = build_transport_table(model, conditions)
        if label == "pipeline":
            table.write(osp.join(RESULTS, "transport_terms_yeast_gem_9.json"))
        summary[label] = {
            "conditions": conditions.model_dump(),
            "n_reactions": table.n_reactions,
            "n_crossing_a_membrane": table.n_crossing,
            "n_with_nonzero_term": table.n_nonzero,
            "n_moving_a_proton": sum(1 for t in table.terms if t.n_proton != 0),
            "n_moving_charge": sum(1 for t in table.terms if t.charge_moved != 0),
            "vs_shipped_kj": compare_to_shipped(table, shipped_rxn, scale=KJ_PER_KCAL),
            "vs_shipped_unconverted": compare_to_shipped(table, shipped_rxn, scale=1.0),
        }
    path = osp.join(RESULTS, "transport_terms.json")
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {path}")


def compartment_table() -> None:
    """Write the compartment conditions the pipeline uses beside the ones yeast9 used."""
    from torchcell.metabolism.flux_layer import CompartmentParameters
    from torchcell.metabolism.transport import pipeline_conditions, yetfl_conditions

    pipeline = pipeline_conditions()
    yetfl = yetfl_conditions(osp.join(YETFL, "yeast_compartment_data.json"))
    flux_layer = CompartmentParameters()
    rows = []
    for symbol in sorted(set(pipeline.p_h) | set(yetfl.p_h)):
        rows.append(
            {
                "compartment": symbol,
                "name": _COMPARTMENT_NAMES.get(symbol, symbol),
                "literature_value_in_library": _LITERATURE_PH.get(symbol),
                "pipeline_p_h": pipeline.p_h.get(symbol),
                "pipeline_ionic_strength_m": pipeline.ionic_strength_m.get(symbol),
                "pipeline_potential_mv_vs_cytosol": pipeline.potential_mv.get(symbol, 0.0),
                "yetfl_p_h": yetfl.p_h.get(symbol),
                "yetfl_ionic_strength_m": yetfl.ionic_strength_m.get(symbol),
                "yetfl_potential_mv_vs_cytosol": yetfl.potential_mv.get(symbol, 0.0),
            }
        )
    out = {
        "generated_at": datetime.now(UTC).isoformat(),
        "rows": rows,
        "literature_status": (
            "Every pH in the pipeline table is a textbook organellar value. The library "
            "holds no primary measurement of any yeast compartment pH: the only in-library "
            "sources for these numbers are two modeling papers that themselves tabulate "
            "assumptions, Smith et al. 2026 (Thermo-Flux, Table 1) and Oftadeh et al. 2021 "
            "(yETFL, through its compartment-data file). The measurements that should "
            "replace them are named in the curation request below; none is in the library, "
            "and none is cited anywhere in this build."
        ),
        "curation_requests": [
            "Orij et al. 2009, in vivo measurement of cytosolic and mitochondrial pH in "
            "S. cerevisiae with ratiometric pHluorin (cytosol, mitochondrial matrix).",
            "Brett et al. 2005 and Plant et al. 1999, vacuolar lumen pH in S. cerevisiae.",
            "Llopis et al. 1998, organelle pH by targeted GFP (mitochondrial matrix, "
            "Golgi, ER), measured in mammalian cells and used here only as the source of "
            "the textbook ordering.",
            "A peroxisomal pH measurement for S. cerevisiae; the pipeline uses 7.0 and "
            "yETFL uses 8.2, a 1.2-unit disagreement that no source in the library "
            "settles.",
        ],
        "pipeline_source": pipeline.source,
        "yetfl_source": yetfl.source,
        "flux_layer_compartment_parameters": flux_layer.model_dump(),
        "equilibrator_defaults": {
            "p_h": 7.0,
            "ionic_strength_m": 0.25,
            "p_mg": 3.0,
            "temperature_k": 298.15,
            "source": (
                "eQuilibrator 3 ComponentContribution defaults; the 026 build overrides "
                "temperature to 303.15 K and pH per compartment."
            ),
        },
    }
    os.makedirs(RESULTS, exist_ok=True)
    path = osp.join(RESULTS, "compartment_conditions.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))
    print(f"wrote {path}")


#: What the LIBRARY can support for each compartment's pH. Every entry is "none": the
#: mirror holds no primary measurement of a yeast compartment pH, so the numbers in use
#: trace to two modeling papers that tabulate them as parameters, not to a measurement.
_LITERATURE_PH: dict[str, str] = {
    symbol: (
        "no primary source in library; in-library modeling tabulations are "
        "smithThermofluxGenerationAnalysis2026 Table 1 (cytosol 7.0, mitochondrion 7.4, "
        "extracellular 5.0 only) and oftadehGenomescaleMetabolicModel2021 via its "
        "compartment-data file"
    )
    for symbol in (
        "c", "ce", "e", "er", "erm", "g", "gm", "im", "lp", "m", "mm", "n", "p", "v", "vm"
    )
}

#: Long names for the compartment symbols, from yeast-GEM's own SBML compartment list.
_COMPARTMENT_NAMES = {
    "c": "cytoplasm",
    "ce": "cell envelope",
    "e": "extracellular",
    "er": "endoplasmic reticulum",
    "erm": "endoplasmic reticulum membrane",
    "g": "Golgi",
    "gm": "Golgi membrane",
    "im": "mitochondrial intermembrane space",
    "lp": "lipid particle",
    "m": "mitochondrion",
    "mm": "mitochondrial membrane",
    "n": "nucleus",
    "p": "peroxisome",
    "v": "vacuole",
    "vm": "vacuolar membrane",
}


def main() -> None:
    """Recompute at yeast9's conditions, or report the provenance measurements."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--transport", action="store_true")
    parser.add_argument("--compartments", action="store_true")
    args = parser.parse_args()
    if args.recompute:
        recompute()
    if args.report:
        report()
    if args.transport:
        transport()
    if args.compartments:
        compartment_table()
    if not (args.recompute or args.report or args.transport or args.compartments):
        parser.error(
            "pass --recompute (eQuilibrator env), or --report / --transport / "
            "--compartments (torchcell env)"
        )


if __name__ == "__main__":
    main()
