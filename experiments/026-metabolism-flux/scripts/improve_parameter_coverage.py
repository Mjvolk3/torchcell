# experiments/026-metabolism-flux/scripts/improve_parameter_coverage.py
# [[experiments.026-metabolism-flux.scripts.improve_parameter_coverage]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/improve_parameter_coverage.py

r"""Push kcat, KM and DrG coverage across all of yeast-GEM 9.0.2, and price the push.

Three tables feed the flux layer and none of them covers the model. This script builds the
best table available for each, measures what every step buys, and refuses to report a
coverage number without the quality measure that makes it readable.

Four things are computed here that the per-predictor summaries cannot give:

* **A coverage ladder for DrG.** The shipped table and the eQuilibrator recomputation cover
  DIFFERENT reactions, so the union beats both. The ladder reports each step at the level
  the flux layer actually consumes, which is the reaction, not the metabolite.
* **Isomer-surrogate pricing.** 455 unpriced metabolites have a heavy-atom composition the
  cache can already decompose. A surrogate is only defensible when the compounds sharing
  that composition AGREE, so the spread within each composition is measured and carried as
  the surrogate's uncertainty rather than assumed away. Positional isomers of a lipid share
  a group vector exactly; constitutional isomers do not, and the spread is what separates
  the two cases empirically.
* **Consensus kinetics.** Across-predictor disagreement for kcat is larger than the spread
  of the quantity itself, so a single predictor is not a defensible table. The consensus is
  the per-unit median with the across-predictor spread carried alongside.
* **Flux-weighted coverage.** A raw count treats a lipid species that can never carry flux
  the same as a central-carbon reaction. Weighting by FVA width answers the question that
  matters: of the reactions that CAN carry flux, how many are parameterized.

Outputs ``results/parameter_coverage.json`` plus the merged tables under
``$DATA_ROOT/data/torchcell/thermo_equilibrator/`` and ``.../kinetics/_consensus/``.

Run with the eQuilibrator env, NOT the torchcell env, which has no ``equilibrator_api``::

    /scratch/projects/torchcell-scratch/envs/equilibrator/bin/python \
        experiments/026-metabolism-flux/scripts/improve_parameter_coverage.py
"""

import json
import os
import os.path as osp
import pickle
import re
import sqlite3
from collections import defaultdict
from typing import cast

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

DATA_ROOT = cast(str, os.environ["DATA_ROOT"])
EXPERIMENT_ROOT = cast(str, os.environ["EXPERIMENT_ROOT"])
GEM_DIR = osp.join(DATA_ROOT, "data", "torchcell", "yeast-GEM", "yeast-GEM-9.0.2")
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
THERMO_DIR = osp.join(DATA_ROOT, "data", "torchcell", "thermo_equilibrator")
KINETICS_DIR = osp.join(DATA_ROOT, "data", "torchcell", "kinetics")
CONSENSUS_DIR = osp.join(KINETICS_DIR, "_consensus")

#: yeast-GEM writes this sentinel where it has no value, rather than an empty cell.
SHIPPED_SENTINEL = 1e7

#: A formula carrying any of these is a polymer/placeholder, not a definite molecule, so it
#: has no composition to match on.
PLACEHOLDER_TOKENS = ("R", "X", "*")

#: Cap on cache compounds priced per composition. The spread is what is wanted, and it is
#: estimated from a sample rather than from every isomer of a common lipid formula.
MAX_SURROGATES_PER_COMPOSITION = 12

#: Across-surrogate spread below which a composition is taken to determine the energy. At
#: 303 K, RT ln(10) is 5.8 kJ/mol, so 5 kJ/mol is under one order of magnitude in the
#: equilibrium constant and cannot flip the sign of a reaction that is not already marginal.
TIGHT_SPREAD_KJ = 5.0

KCAT_PREDICTORS = ("dlkcat", "unikp", "eitlem", "turnup", "deepenzyme", "ezkinetics")
KM_PREDICTORS = ("unikp", "eitlem", "boost_km", "ezkinetics")


class CoverageStep(BaseModel):
    """One rung of a coverage ladder, always reported at both levels."""

    step: str
    n_metabolites_priced: int
    n_reactions_priced: int
    frac_reactions: float
    delta_reactions_vs_previous: int


def parse_formula(formula: str | None) -> dict[str, int] | None:
    """Heavy-atom composition, or None when the formula is absent or a placeholder."""
    if not formula:
        return None
    tokens = re.findall(r"([A-Z][a-z]?|\*)(\d*)", formula)
    bag: dict[str, int] = {}
    for element, count in tokens:
        if not element:
            continue
        if element in PLACEHOLDER_TOKENS:
            return None
        if element == "H":
            continue
        bag[element] = bag.get(element, 0) + (int(count) if count else 1)
    return bag or None


def composition_key(bag: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(bag.items()))


def cache_compounds_by_composition(
    ccache, wanted: set[tuple[tuple[str, int], ...]]
) -> dict[tuple[tuple[str, int], ...], list[int]]:
    """Cache compound ids per wanted composition, restricted to ones with a group vector.

    One streaming pass, because ``atom_bag`` is a pickled SQLAlchemy column with no index.
    A compound without a group vector cannot be priced by component contribution and is
    therefore useless as a surrogate, so it is dropped here rather than downstream.
    """
    path = ccache.session.get_bind().url.database
    connection = sqlite3.connect(path)
    found: dict[tuple[tuple[str, int], ...], list[int]] = defaultdict(list)
    for compound_id, atom_bag_blob, group_vector in connection.execute(
        "SELECT id, atom_bag, group_vector FROM compounds WHERE atom_bag IS NOT NULL"
    ):
        if group_vector is None:
            continue
        bag = pickle.loads(atom_bag_blob)
        key = tuple(
            sorted(
                (element, count)
                for element, count in bag.items()
                if element not in ("H", "e-")
            )
        )
        if key in wanted and len(found[key]) < MAX_SURROGATES_PER_COMPOSITION:
            found[key].append(compound_id)
    connection.close()
    return dict(found)


def reactions_priced(model, priced: set[str]) -> int:
    """Reactions whose every participant carries a formation energy."""
    return sum(
        1
        for reaction in model.reactions
        if all(m.id in priced for m in reaction.metabolites)
    )


def build_surrogates(met_table: pd.DataFrame) -> pd.DataFrame:
    """Price unpriced metabolites from cache compounds of identical heavy-atom composition.

    Returns one row per metabolite that gained a value, carrying the number of surrogates
    the composition matched and the spread across them. The spread is the whole point: a
    composition whose cache members agree to a few kJ/mol constrains the energy, and one
    whose members disagree by 100 does not. Nothing here decides a cutoff; the spread is
    reported so the consumer can.
    """
    from equilibrator_api import ComponentContribution, Q_

    unpriced = met_table[met_table["dgf_prime_kj_per_mol"].isna()]
    compositions: dict[str, tuple[tuple[str, int], ...]] = {}
    for row in unpriced.itertuples():
        bag = parse_formula(row.formula)
        if bag is not None:
            compositions[row.met_id] = composition_key(bag)

    cc = ComponentContribution()
    cc.ionic_strength = Q_("0.25M")
    cc.temperature = Q_("303.15K")
    cc.p_mg = Q_(3.0)

    wanted = set(compositions.values())
    matches = cache_compounds_by_composition(cc.ccache, wanted)

    # Price each distinct composition once, untransformed. The Legendre transform is applied
    # per metabolite afterwards, because two metabolites of the same composition can sit in
    # compartments at different pH.
    # `standard_dg_formation` returns (mu, sigma_fin, sigma_inf) where the sigmas are
    # covariance VECTORS, not scalars. Only mu is kept: the uncertainty carried for a
    # surrogate is the spread across the compounds sharing the composition, which is the
    # quantity that says whether the composition constrains the energy at all.
    per_composition: dict[tuple[tuple[str, int], ...], list[tuple[object, float]]] = {}
    for key, compound_ids in matches.items():
        priced: list[tuple[object, float]] = []
        for compound_id in compound_ids:
            compound = cc.ccache.get_compound_by_internal_id(compound_id)
            if compound is None:
                continue
            mu, _sigma_fin, _sigma_inf = cc.standard_dg_formation(compound)
            if mu is None:
                continue
            priced.append((compound, float(mu)))
        if priced:
            per_composition[key] = priced

    rows = []
    for met_id, key in compositions.items():
        priced = per_composition.get(key)
        if not priced:
            continue
        met_row = met_table.loc[met_table["met_id"] == met_id].iloc[0]
        p_h = float(met_row["p_h"])
        transformed = []
        for compound, mu in priced:
            correction = compound.transform(
                Q_(p_h), Q_("0.25M"), Q_("303.15K"), Q_(3.0)
            )
            transformed.append(mu + float(correction.m_as("kJ/mol")))
        values = np.asarray(transformed, dtype=float)
        rows.append(
            {
                "met_id": met_id,
                "dgf_prime_surrogate_kj_per_mol": float(np.median(values)),
                "n_surrogates": int(values.size),
                "surrogate_spread_kj_per_mol": float(values.max() - values.min()),
                "surrogate_sd_kj_per_mol": float(values.std(ddof=0)),
                "composition": "".join(f"{e}{c}" for e, c in key),
            }
        )
    return pd.DataFrame(rows)


def kinetics_consensus(parameter: str, predictors: tuple[str, ...]) -> pd.DataFrame:
    """Per catalytic unit, the median across predictors plus the across-predictor spread.

    Keyed on ``unit_id`` because that is what the capacity bound reads. The spread is
    carried because for kcat it is larger than the dynamic range of the quantity, which
    makes a bare median an overstatement of what is known.
    """
    frames = []
    for predictor in predictors:
        path = osp.join(KINETICS_DIR, predictor, "processed", f"{parameter}.parquet")
        if not osp.exists(path):
            continue
        frame = pd.read_parquet(path)[["unit_id", "reaction_id", "gene_id", parameter]]
        frame = frame.dropna(subset=[parameter])
        frame = frame[frame[parameter] > 0]
        frame["predictor"] = predictor
        frames.append(frame)
    stacked = pd.concat(frames, ignore_index=True)

    # Aggregate to the unit first: a unit with several substrates, or several subunit genes,
    # gets that predictor's median over them, so a promiscuous enzyme does not outvote a
    # specific one and a large complex does not vote once per subunit. Grouping on anything
    # finer than `unit_id` splits one catalytic unit across rows and inflates every count
    # past the number of units that exist.
    per_unit = (
        stacked.groupby(["unit_id", "predictor"])[parameter].median().reset_index()
    )
    logs = per_unit.assign(log10=np.log10(per_unit[parameter]))
    grouped = logs.groupby("unit_id")["log10"]
    out = grouped.agg(["median", "min", "max", "count"]).reset_index()
    out = out.rename(columns={"count": "n_predictors"})
    out[f"{parameter}_consensus"] = 10 ** out["median"]
    out["spread_decades"] = out["max"] - out["min"]
    reaction_of_unit = stacked.groupby("unit_id")["reaction_id"].first()
    out["reaction_id"] = out["unit_id"].map(reaction_of_unit)
    return out[
        [
            "unit_id",
            "reaction_id",
            f"{parameter}_consensus",
            "n_predictors",
            "spread_decades",
        ]
    ]


def main() -> None:
    import cobra

    os.makedirs(CONSENSUS_DIR, exist_ok=True)
    model = cobra.io.read_sbml_model(osp.join(GEM_DIR, "model", "yeast-GEM.xml"))
    n_reactions = len(model.reactions)
    n_metabolites = len(model.metabolites)

    met_table = pd.read_parquet(osp.join(THERMO_DIR, "metabolite_dgf_prime.parquet"))
    shipped_met = pd.read_csv(
        osp.join(GEM_DIR, "data", "databases", "model_metDeltaG.csv")
    ).rename(columns={"Var1": "met_id", "Var2": "dgf_shipped"})
    shipped_met = shipped_met[shipped_met["dgf_shipped"] < SHIPPED_SENTINEL]

    priced_recomputed = set(
        met_table.loc[met_table["dgf_prime_kj_per_mol"].notna(), "met_id"]
    )
    priced_shipped = set(shipped_met["met_id"])

    ladder: list[CoverageStep] = []

    def rung(name: str, priced: set[str]) -> None:
        n_rxn = reactions_priced(model, priced)
        previous = ladder[-1].n_reactions_priced if ladder else 0
        ladder.append(
            CoverageStep(
                step=name,
                n_metabolites_priced=len(priced),
                n_reactions_priced=n_rxn,
                frac_reactions=n_rxn / n_reactions,
                delta_reactions_vs_previous=n_rxn - previous,
            )
        )

    rung("shipped_only", priced_shipped)
    rung("recomputed_only", priced_recomputed)
    rung("union_shipped_recomputed", priced_shipped | priced_recomputed)

    surrogates = build_surrogates(met_table)
    surrogates.to_parquet(osp.join(THERMO_DIR, "metabolite_dgf_surrogate.parquet"))
    priced_surrogate = set(surrogates["met_id"])

    # The tight subset is the only surrogate rung that should ever be wired in. A
    # composition whose cache members span 50 kJ/mol does not determine the energy: at
    # 303 K, 50 kJ/mol is nine orders of magnitude in the equilibrium constant, so such a
    # value would set the sign of the second-law hinge essentially at random.
    tight = surrogates[surrogates["surrogate_spread_kj_per_mol"] < TIGHT_SPREAD_KJ]
    priced_tight = set(tight["met_id"])
    rung(
        "union_plus_tight_surrogate",
        priced_shipped | priced_recomputed | priced_tight,
    )
    rung(
        "union_plus_all_surrogate_NOT_RECOMMENDED",
        priced_shipped | priced_recomputed | priced_surrogate,
    )

    # Flux-weighted coverage. A reaction whose FVA width is zero cannot carry flux under the
    # wild-type medium, so parameterizing it changes nothing the layer can express.
    fva = pd.read_csv(osp.join(RESULTS, "fva_wildtype.csv"), index_col=0)
    carries_flux = set(fva.index[fva["width"] > 1e-9])
    recommended = priced_shipped | priced_recomputed | priced_tight

    def priced_among_flux_carrying(priced: set[str]) -> int:
        return sum(
            1
            for r in model.reactions
            if r.id in carries_flux and all(m.id in priced for m in r.metabolites)
        )

    flux_weighted = {
        "n_reactions_with_nonzero_fva_width": len(carries_flux),
        "n_of_those_priced_shipped": priced_among_flux_carrying(priced_shipped),
        "n_of_those_priced_recomputed": priced_among_flux_carrying(priced_recomputed),
        "n_of_those_priced_recommended": priced_among_flux_carrying(recommended),
    }
    flux_weighted["frac_flux_carrying_priced_shipped"] = (
        flux_weighted["n_of_those_priced_shipped"]
        / flux_weighted["n_reactions_with_nonzero_fva_width"]
    )
    flux_weighted["frac_flux_carrying_priced_recommended"] = (
        flux_weighted["n_of_those_priced_recommended"]
        / flux_weighted["n_reactions_with_nonzero_fva_width"]
    )

    kcat = kinetics_consensus("k_cat", KCAT_PREDICTORS)
    km = kinetics_consensus("K_M", KM_PREDICTORS)
    kcat.to_parquet(osp.join(CONSENSUS_DIR, "k_cat_consensus.parquet"))
    km.to_parquet(osp.join(CONSENSUS_DIR, "K_M_consensus.parquet"))

    n_units_total = 3728
    report = {
        "gem": {
            "n_reactions": n_reactions,
            "n_metabolites": n_metabolites,
            "n_catalytic_units": n_units_total,
        },
        "delta_g_ladder": [step.model_dump() for step in ladder],
        "surrogate": {
            "n_metabolites_gained": int(len(surrogates)),
            "median_n_surrogates_per_metabolite": float(
                surrogates["n_surrogates"].median()
            )
            if len(surrogates)
            else None,
            "spread_kj_per_mol_median": float(
                surrogates["surrogate_spread_kj_per_mol"].median()
            )
            if len(surrogates)
            else None,
            "spread_kj_per_mol_p90": float(
                surrogates["surrogate_spread_kj_per_mol"].quantile(0.9)
            )
            if len(surrogates)
            else None,
            "n_with_spread_under_tight_threshold": int(
                (surrogates["surrogate_spread_kj_per_mol"] < TIGHT_SPREAD_KJ).sum()
            )
            if len(surrogates)
            else 0,
            "tight_spread_threshold_kj": TIGHT_SPREAD_KJ,
        },
        "flux_weighted": flux_weighted,
        "kinetics_consensus": {
            "k_cat": {
                "n_units": int(kcat["unit_id"].nunique()),
                "coverage_frac": float(kcat["unit_id"].nunique() / n_units_total),
                "median_value": float(kcat["k_cat_consensus"].median()),
                "median_spread_decades": float(kcat["spread_decades"].median()),
                "n_units_all_predictors": int(
                    (kcat["n_predictors"] == len(KCAT_PREDICTORS)).sum()
                ),
            },
            "K_M": {
                "n_units": int(km["unit_id"].nunique()),
                "coverage_frac": float(km["unit_id"].nunique() / n_units_total),
                "median_value": float(km["K_M_consensus"].median()),
                "median_spread_decades": float(km["spread_decades"].median()),
                "n_units_all_predictors": int(
                    (km["n_predictors"] == len(KM_PREDICTORS)).sum()
                ),
            },
        },
    }
    with open(osp.join(RESULTS, "parameter_coverage.json"), "w") as handle:
        json.dump(report, handle, indent=2)

    for step in ladder:
        print(
            f"{step.step:32s} reactions {step.n_reactions_priced:5d} "
            f"({step.frac_reactions:6.2%})  delta {step.delta_reactions_vs_previous:+5d}"
        )
    print(json.dumps({k: report[k] for k in ("surrogate", "flux_weighted")}, indent=2))
    print(json.dumps(report["kinetics_consensus"], indent=2))


if __name__ == "__main__":
    main()
