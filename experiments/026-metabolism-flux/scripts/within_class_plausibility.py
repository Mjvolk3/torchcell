# experiments/026-metabolism-flux/scripts/within_class_plausibility.py
# [[experiments.026-metabolism-flux.scripts.within_class_plausibility]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/within_class_plausibility.py

r"""How many predictions land inside the spread of enzymes doing the same job, by model.

The predicted-against-measured panel in the agreement figure carries one shaded band, the
median within-EC4-class standard deviation, and its caption gives the fraction of points
inside it as a range across predictors. This script makes that fraction a first-class
result, computed four ways so that the answer does not depend on one arbitrary choice of
band.

**Four readings of "inside the within-class spread".**

1. **Shared band at each EC depth.** The band half-width is the median within-class
   standard deviation over all classes defined at EC1, EC2, EC3 or EC4. A class read to
   fewer digits pools unlike chemistry and is wider, so the fraction inside must rise as
   digits are dropped; how fast it rises says how much of a predictor's plausibility is
   coarse-class knowledge.
2. **Own-class band at EC4.** Each measured pair is scored against the standard deviation
   of ITS OWN EC4 class rather than the median over classes. Only pairs whose class holds
   at least three distinct enzymes in the Open Enzyme Database can be scored this way, so
   the count is smaller and is reported.
3. **The EC-median lookup**, scored the same way, which is the reference a predictor has
   to beat: it knows the enzyme's function and nothing else.
4. **The betaxanthin case study genes.** None of the nine yeast genes feeding the
   cassette has a measured yeast value in the Open Enzyme Database, so accuracy cannot be
   scored for them. What can be scored is class plausibility: each (gene, reaction) unit
   carries an EC number from yeast-GEM, that EC class has a measured distribution across
   organisms, and a prediction either sits inside one standard deviation of that class or
   it does not.

Every fraction carries a Wilson 95 percent interval, since the denominators are 24 to 57.

**Measured coverage of the catalytic units**, for the coverage figure. The Open Enzyme
Database supplies a measured wild-type value for a small fraction of yeast-GEM's 3,728
catalytic units, and EnzEngDB (``longEnzymeEngineeringDatabase2026``) supplies none: it was
read in full and queried on 2026-09-02 and holds no :math:`k_{cat}` or :math:`K_M` field,
no EC, no organism field and zero *S. cerevisiae* content. Both numbers are written here so
the coverage figure can draw the gap that motivates the predictors, a source contributing
nothing included.

Writes ``results/within_class_plausibility.json`` and
``results/within_class_plausibility_betaxanthin.csv``.
Run from the repo root with the torchcell env, after ``ec_class_variance.py`` and
``compare_predictors_to_measured.py``.
"""

import json
import os
import os.path as osp
import sys
from typing import cast

import cobra
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

from torchcell.metabolism.constraints import parse_catalytic_units

load_dotenv()

DATA_ROOT = cast(str, os.environ["DATA_ROOT"])
EXPERIMENT_ROOT = cast(str, os.environ["EXPERIMENT_ROOT"])
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
SCRIPTS = osp.dirname(osp.abspath(__file__))
KINETICS = osp.join(DATA_ROOT, "data", "torchcell", "kinetics")
OED_DIR = osp.join(DATA_ROOT, "data", "enzyme_kinetics", "open_enzyme_database")
ALL_RECORDS = osp.join(OED_DIR, "all", "oed_records.json")
YEAST_RECORDS = osp.join(OED_DIR, "scerevisiae", "oed_records.json")
GEM_XML = osp.join(
    DATA_ROOT, "data", "torchcell", "yeast-GEM", "yeast-GEM-9.0.2", "model", "yeast-GEM.xml"
)

sys.path.insert(0, SCRIPTS)
from ec_class_variance import (  # noqa: E402
    MIN_ENZYMES_PER_CLASS,
    ec_class_baseline,
    load_records,
    per_pair,
    within_class_spread,
)

KCAT_PREDICTORS = ("dlkcat", "unikp", "eitlem", "turnup", "deepenzyme", "ezkinetics")
KM_PREDICTORS = ("unikp", "eitlem", "boost_km", "ezkinetics")

#: The genes the betaxanthin case study runs on, as the agreement figure lists them: the
#: two native cassette members, then the aromatic-amino-acid genes supplying tyrosine.
#: CYP76AD1 and DOD are heterologous, have no yeast-GEM reaction and so no unit to score.
BETAXANTHIN_GENES: list[tuple[str, str]] = [
    ("YBR249C", "ARO4"),
    ("YPR060C", "ARO7"),
    ("YDR035W", "ARO3"),
    ("YDR127W", "ARO1"),
    ("YGL148W", "ARO2"),
    ("YBR166C", "TYR1"),
    ("YGL202W", "ARO8"),
    ("YHR137W", "ARO9"),
    ("YNL316C", "PHA2"),
]


class Fraction(BaseModel):
    """A fraction with its Wilson 95 percent interval and the count behind it."""

    n: int
    k: int
    frac: float
    ci95: tuple[float, float]

    @classmethod
    def of(cls, inside: np.ndarray) -> "Fraction":
        n = int(len(inside))
        k = int(inside.sum())
        if n == 0:
            return cls(n=0, k=0, frac=float("nan"), ci95=(float("nan"), float("nan")))
        z = 1.959964
        p = k / n
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return cls(n=n, k=k, frac=p, ci95=(float(center - half), float(center + half)))


class BetaxanthinUnit(BaseModel):
    """One (gene, reaction) unit of the betaxanthin supply pathway and its class."""

    gene_id: str
    common_name: str
    reaction_id: str
    reaction_name: str
    ec: str | None
    class_n_enzymes: int
    class_median_log10: float | None
    class_sd_log10: float | None
    class_min_log10: float | None
    class_max_log10: float | None
    predicted_log10: dict[str, float]
    inside_one_sd: dict[str, bool]


def wilson_json(fraction: Fraction) -> dict:
    return fraction.model_dump()


def load_predicted_units(predictor: str, parameter: str) -> pd.DataFrame:
    """Median log10 value per (gene, reaction) for one predictor."""
    path = osp.join(KINETICS, predictor, "processed", f"{parameter}.parquet")
    frame = pd.read_parquet(path)
    frame = frame[frame[parameter] > 0].copy()
    frame["log10"] = np.log10(frame[parameter])
    return frame.groupby(["gene_id", "reaction_id"], as_index=False)["log10"].median()


def load_scored_pairs(parameter: str) -> pd.DataFrame:
    """The measured pairs each predictor was scored on, with the pair's EC number."""
    pairs = pd.read_csv(osp.join(RESULTS, "predictor_vs_measured.csv"))
    pairs = pairs[pairs["parameter"] == parameter].copy()
    yeast = load_records(YEAST_RECORDS, parameter)
    ec_of_pair = yeast.groupby(["uniprot", "substrate"])["ec"].first().to_dict()
    pairs["ec"] = [
        ec_of_pair.get((u, s)) for u, s in zip(pairs["uniprot"], pairs["substrate"])
    ]
    pairs["residual"] = np.log10(pairs["predicted"]) - np.log10(pairs["measured"])
    return pairs


def class_table(pairs_all: pd.DataFrame, level: int) -> pd.DataFrame:
    """Per-class spread at one EC depth, keyed by the truncated EC string."""
    classes, _ = within_class_spread(pairs_all, level)
    return classes.set_index("ec")


def shared_band_fractions(
    scored: pd.DataFrame, predictors: list[str], bands: dict[int, float]
) -> dict:
    """Fraction of each predictor's pairs inside the shared median band, per EC depth."""
    out: dict = {}
    for predictor in predictors:
        subset = scored[scored["predictor"] == predictor]
        if subset.empty:
            continue
        out[predictor] = {
            f"ec{level}": wilson_json(Fraction.of(np.abs(subset["residual"]) < band))
            for level, band in bands.items()
        }
    return out


def own_class_fractions(
    scored: pd.DataFrame, predictors: list[str], ec4: pd.DataFrame
) -> dict:
    """Fraction inside each pair's OWN EC4 class standard deviation."""
    band_of = ec4["sd_decades"].to_dict()
    scorable = scored[scored["ec"].isin(band_of)].copy()
    scorable["band"] = scorable["ec"].map(band_of)
    out: dict = {
        "n_pairs_scorable": int(scorable.drop_duplicates(["uniprot", "substrate"]).shape[0]),
        "n_pairs_total": int(scored.drop_duplicates(["uniprot", "substrate"]).shape[0]),
        "min_enzymes_per_class": MIN_ENZYMES_PER_CLASS,
        "predictors": {},
    }
    for predictor in predictors:
        subset = scorable[scorable["predictor"] == predictor]
        if subset.empty:
            continue
        out["predictors"][predictor] = wilson_json(
            Fraction.of((np.abs(subset["residual"]) < subset["band"]).to_numpy())
        )
    return out


def baseline_fractions(
    scored: pd.DataFrame, reference_pairs: pd.DataFrame, bands: dict[int, float],
    ec4: pd.DataFrame,
) -> dict:
    """The EC-median lookup scored exactly like the predictors."""
    unique = scored.drop_duplicates(["uniprot", "substrate"])[
        ["uniprot", "substrate", "measured", "ec"]
    ]
    table, summary = ec_class_baseline(unique, reference_pairs)
    residual = (table["baseline_log10"] - np.log10(table["measured"])).to_numpy()
    band_of = ec4["sd_decades"].to_dict()
    own = table[table["ec"].isin(band_of)]
    own_residual = (own["baseline_log10"] - np.log10(own["measured"])).to_numpy()
    own_band = own["ec"].map(band_of).to_numpy()
    return {
        "summary": summary,
        "shared": {
            f"ec{level}": wilson_json(Fraction.of(np.abs(residual) < band))
            for level, band in bands.items()
        },
        "own_class": wilson_json(Fraction.of(np.abs(own_residual) < own_band)),
    }


def ec_codes(reaction: cobra.Reaction) -> list[str]:
    value = reaction.annotation.get("ec-code")
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def betaxanthin_units(
    model: cobra.Model, predictors: list[str], parameter: str, ec4: pd.DataFrame,
    reference_pairs: pd.DataFrame,
) -> list[BetaxanthinUnit]:
    """Every (gene, reaction) unit of the pathway genes, scored against its EC class.

    A reaction annotated with several EC numbers (ARO1 is a pentafunctional enzyme carrying
    five) is scored against the one whose class is largest in the database, and that choice
    is recorded in the row. A unit whose EC has fewer than three measured enzymes, or no EC
    at all, keeps its predictions and gets no band.
    """
    predicted = {p: load_predicted_units(p, parameter) for p in predictors}
    per_enzyme = reference_pairs.groupby(["ec", "uniprot"], as_index=False)["log10"].mean()
    rows: list[BetaxanthinUnit] = []
    for gene_id, common in BETAXANTHIN_GENES:
        for reaction in sorted(model.genes.get_by_id(gene_id).reactions, key=lambda r: r.id):
            candidates = [e for e in ec_codes(reaction) if e in ec4.index]
            chosen = (
                max(candidates, key=lambda e: int(ec4.loc[e, "n_enzymes"]))
                if candidates
                else (ec_codes(reaction)[0] if ec_codes(reaction) else None)
            )
            values: dict[str, float] = {}
            for name, frame in predicted.items():
                hit = frame[(frame["gene_id"] == gene_id) & (frame["reaction_id"] == reaction.id)]
                if not hit.empty:
                    values[name] = float(hit["log10"].iloc[0])
            if chosen in ec4.index:
                members = per_enzyme[per_enzyme["ec"] == chosen]["log10"]
                median = float(ec4.loc[chosen, "median_log10"])
                sd = float(ec4.loc[chosen, "sd_decades"])
                rows.append(
                    BetaxanthinUnit(
                        gene_id=gene_id, common_name=common, reaction_id=reaction.id,
                        reaction_name=reaction.name, ec=chosen,
                        class_n_enzymes=int(ec4.loc[chosen, "n_enzymes"]),
                        class_median_log10=median, class_sd_log10=sd,
                        class_min_log10=float(members.min()),
                        class_max_log10=float(members.max()),
                        predicted_log10=values,
                        inside_one_sd={k: bool(abs(v - median) < sd) for k, v in values.items()},
                    )
                )
            else:
                rows.append(
                    BetaxanthinUnit(
                        gene_id=gene_id, common_name=common, reaction_id=reaction.id,
                        reaction_name=reaction.name, ec=chosen, class_n_enzymes=0,
                        class_median_log10=None, class_sd_log10=None,
                        class_min_log10=None, class_max_log10=None,
                        predicted_log10=values, inside_one_sd={},
                    )
                )
    return rows


def measured_source_coverage(model: cobra.Model) -> dict:
    """How many of the 3,728 catalytic units any measured source can price."""
    units = parse_catalytic_units(model, [r.id for r in model.reactions])
    unit_gene = units.unit_gene_index.numpy()
    genes_of_unit: dict[int, set[str]] = {}
    for unit, gene in zip(unit_gene[0], unit_gene[1]):
        genes_of_unit.setdefault(int(unit), set()).add(units.gene_ids[int(gene)])
    # The predictor tables carry the gene-to-accession map the pipeline itself uses.
    accession = (
        pd.read_parquet(osp.join(KINETICS, "unikp", "processed", "k_cat.parquet"))
        .dropna(subset=["uniprot"])
        .groupby("gene_id")["uniprot"].first().to_dict()
    )
    out: dict = {"n_catalytic_units": int(units.n_units), "sources": {}}
    yeast = json.load(open(YEAST_RECORDS))
    for parameter, column in (("k_cat", "kcat_value"), ("K_M", "km_value")):
        measured = {
            r["uniprot"] for r in yeast
            if r.get("uniprot") and (r.get("enzymetype") or "").lower() == "wildtype"
            and r.get(column) is not None and r[column] > 0
        }
        covered = sum(
            1 for genes in genes_of_unit.values()
            if all(accession.get(g) in measured for g in genes)
        )
        out["sources"].setdefault("open_enzyme_database", {})[parameter] = {
            "n_units": int(covered),
            "frac": covered / units.n_units,
            "n_measured_wildtype_accessions": len(measured),
        }
    accuracy = json.load(open(osp.join(RESULTS, "predictor_vs_measured.json")))
    out["sources"]["enzengdb"] = {
        parameter: {"n_units": 0, "frac": 0.0} for parameter in ("k_cat", "K_M")
    } | {"why": accuracy["enzengdb"]}
    return out


def main() -> None:
    model = cobra.io.read_sbml_model(GEM_XML)
    report: dict = {"parameters": {}}
    betaxanthin_rows: list[dict] = []
    for parameter, predictors in (("k_cat", list(KCAT_PREDICTORS)), ("K_M", list(KM_PREDICTORS))):
        reference = per_pair(load_records(ALL_RECORDS, parameter))
        bands = {
            level: within_class_spread(reference, level)[1]["median_within_class_sd_decades"]
            for level in (1, 2, 3, 4)
        }
        ec4 = class_table(reference, 4)
        scored = load_scored_pairs(parameter)
        scored = scored[scored["ec"].notna()]
        present = [p for p in predictors if (scored["predictor"] == p).any()]
        # Which pathway genes have a measured yeast value at all. Measured, not assumed:
        # the answer decides whether accuracy or only plausibility can be scored for them.
        gene_ids = {g for g, _ in BETAXANTHIN_GENES}
        accessions = pd.read_parquet(
            osp.join(KINETICS, "unikp", "processed", f"{parameter}.parquet")
        )
        pathway_accessions = set(accessions[accessions["gene_id"].isin(gene_ids)]["uniprot"])
        measured_pathway = sorted(set(scored["uniprot"]) & pathway_accessions)
        units = betaxanthin_units(model, present, parameter, ec4, reference)
        scorable = [u for u in units if u.class_sd_log10 is not None]
        per_predictor_bx = {}
        for predictor in present:
            flags = np.array(
                [u.inside_one_sd[predictor] for u in scorable if predictor in u.inside_one_sd]
            )
            per_predictor_bx[predictor] = wilson_json(Fraction.of(flags))
        report["parameters"][parameter] = {
            "n_measured_pairs": int(scored.drop_duplicates(["uniprot", "substrate"]).shape[0]),
            "band_half_width_by_level": {f"ec{k}": v for k, v in bands.items()},
            "shared_band": shared_band_fractions(scored, present, bands),
            "own_class_band": own_class_fractions(scored, present, ec4),
            "ec_median_lookup": baseline_fractions(scored, reference, bands, ec4),
            "betaxanthin": {
                "n_units": len(units),
                "n_units_with_scorable_class": len(scorable),
                "pathway_accessions_with_measured_yeast_value": measured_pathway,
                "per_predictor_inside_one_sd": per_predictor_bx,
            },
        }
        for unit in units:
            row = unit.model_dump()
            row["parameter"] = parameter
            betaxanthin_rows.append(row)
    report["measured_source_coverage"] = measured_source_coverage(model)

    with open(osp.join(RESULTS, "within_class_plausibility.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    flat = pd.DataFrame(betaxanthin_rows)
    flat.to_csv(osp.join(RESULTS, "within_class_plausibility_betaxanthin.csv"), index=False)

    for parameter, block in report["parameters"].items():
        print(f"\n{parameter}: {block['n_measured_pairs']} measured pairs; bands "
              + ", ".join(f"{k} {v:.2f}" for k, v in block["band_half_width_by_level"].items()))
        print(f"  {'predictor':12s} {'ec1':>6s} {'ec2':>6s} {'ec3':>6s} {'ec4':>6s} {'own':>6s}")
        for name, levels in block["shared_band"].items():
            own = block["own_class_band"]["predictors"][name]
            print(f"  {name:12s} " + " ".join(f"{levels[k]['frac']:6.2f}" for k in ("ec1", "ec2", "ec3", "ec4"))
                  + f" {own['frac']:6.2f} (n={own['n']})")
        base = block["ec_median_lookup"]
        print(f"  {'EC median':12s} " + " ".join(f"{base['shared'][k]['frac']:6.2f}" for k in ("ec1", "ec2", "ec3", "ec4"))
              + f" {base['own_class']['frac']:6.2f}")
        bx = block["betaxanthin"]
        print(f"  betaxanthin units {bx['n_units']} ({bx['n_units_with_scorable_class']} with a class): "
              + ", ".join(f"{k} {v['k']}/{v['n']}" for k, v in bx["per_predictor_inside_one_sd"].items()))
    cov = report["measured_source_coverage"]
    print(f"\nmeasured coverage of {cov['n_catalytic_units']} units: "
          + json.dumps({k: {p: v.get('frac') for p, v in s.items() if p in ('k_cat', 'K_M')} for k, s in cov['sources'].items()}))
    print(f"wrote {osp.join(RESULTS, 'within_class_plausibility.json')}")


if __name__ == "__main__":
    main()
