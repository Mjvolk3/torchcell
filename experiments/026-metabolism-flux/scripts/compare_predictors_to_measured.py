# experiments/026-metabolism-flux/scripts/compare_predictors_to_measured.py
# [[experiments.026-metabolism-flux.scripts.compare_predictors_to_measured]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/compare_predictors_to_measured.py

r"""Predictor ACCURACY against measured yeast values, which agreement alone cannot give.

Every comparison in this work so far has been predictor against predictor, which measures
agreement and says nothing about whether any of them is right. The Open Enzyme Database
mirror carries 1,126 measured *S. cerevisiae* records with `kcat_value`, `km_value`, a
UniProt accession, a substrate SMILES and the assay temperature and pH. Joining those to the
predictor tables gives the first accuracy number in the document.

**The leakage caveat, stated up front and not buried.** Every one of these predictors was
trained on BRENDA and/or SABIO-RK, and the Open Enzyme Database is itself assembled from that
same literature. So an unknown and probably large fraction of the joined pairs was seen in
training. **These numbers are therefore an UPPER BOUND on held-out accuracy, not an
estimate of it.** They are still worth having: a predictor that cannot reproduce values it
may well have trained on has a problem that no held-out set would forgive, and the RANKING
between predictors is more trustworthy than any single value.

For CatPred the fraction is not unknown, because it released its training table with a
UniProt accession on every row: 50 of the 57 :math:`k_{cat}` pairs and 43 of the 57
:math:`K_M` pairs appear in CatPred-DB on a substrate whose structure matches, and the
released production weights were fit on train plus validation plus test
(``catpred_training_overlap.py`` -> ``results/catpred_training_overlap.json``). No zero is
claimed for the other six; their overlap is unmeasured, not absent.

**The Arnold-group database cannot be compared and this is measured, not assumed.**
`longEnzymeEngineeringDatabase2026` is EnzEngDB (Long et al., NAR, doi 10.1093/nar/gkaf1142).
It was read in full and queried read-only on 2026-09-02: it flattens every assay output into
one untyped `fitness_value` column by explicit design, so it holds **no kcat and no KM**; it
has no EC and no organism field; and it contains **zero** occurrences of `yeast`,
`cerevisiae` or `Saccharomyces` across the paper, all 140 public experiment records and all
1,342 analysis rows. Its six scaffolds are bacterial and archaeal, engineered for carbene and
nitrene chemistry that does not occur in yeast. The overlap with the 3,728 yeast-GEM
catalytic units is zero, so there is no comparison to draw rather than a weak one.

Writes ``results/predictor_vs_measured.json`` and ``results/predictor_vs_measured.csv``.
"""

import json
import os
import os.path as osp
from typing import cast

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

load_dotenv()

DATA_ROOT = cast(str, os.environ["DATA_ROOT"])
EXPERIMENT_ROOT = cast(str, os.environ["EXPERIMENT_ROOT"])
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
KINETICS_DIR = osp.join(DATA_ROOT, "data", "torchcell", "kinetics")
OED_PATH = osp.join(
    DATA_ROOT,
    "data",
    "enzyme_kinetics",
    "open_enzyme_database",
    "scerevisiae",
    "oed_records.json",
)

KCAT_PREDICTORS = (
    "dlkcat",
    "unikp",
    "eitlem",
    "turnup",
    "deepenzyme",
    "ezkinetics",
    "catpred",
)
KM_PREDICTORS = ("unikp", "eitlem", "boost_km", "ezkinetics", "catpred")

#: Assay temperature window kept, in Celsius. The GEM's own physiology is 30 C, and a kcat
#: measured at 65 C is a different quantity rather than a noisy version of the same one.
TEMP_MIN_C, TEMP_MAX_C = 20.0, 40.0


def load_measured(parameter: str) -> pd.DataFrame:
    """Measured OED rows for one parameter, wild-type only and near-physiological."""
    records = json.load(open(OED_PATH))
    column = {"k_cat": "kcat_value", "K_M": "km_value"}[parameter]
    rows = []
    for record in records:
        value = record.get(column)
        if value is None or not np.isfinite(value) or value <= 0:
            continue
        # Variants carry engineered turnover; the predictors are given the wild-type
        # sequence from the genome, so a variant row is a different enzyme.
        if (record.get("enzymetype") or "").lower() != "wildtype":
            continue
        temperature = record.get("temperature")
        if temperature is not None and not (TEMP_MIN_C <= temperature <= TEMP_MAX_C):
            continue
        rows.append(
            {
                "uniprot": record.get("uniprot"),
                "smiles": record.get("smiles"),
                "substrate": (record.get("substrate") or "").strip().lower(),
                "measured": float(value),
                "temperature": temperature,
                "ph": record.get("ph"),
            }
        )
    frame = pd.DataFrame(rows).dropna(subset=["uniprot"])
    # One enzyme-substrate pair can carry several literature values; the geometric median
    # over them is the target, since these are log-scale quantities.
    grouped = frame.groupby(["uniprot", "substrate"], as_index=False).agg(
        measured=("measured", lambda v: float(np.exp(np.log(v).mean()))),
        smiles=("smiles", "first"),
        n_reports=("measured", "size"),
        report_spread_decades=(
            "measured",
            lambda v: float(np.log10(v.max() / v.min())) if len(v) > 1 else 0.0,
        ),
    )
    return grouped


def load_predicted(predictor: str, parameter: str) -> pd.DataFrame:
    """One predictor's table, reduced to (uniprot, substrate) with its predicted value."""
    path = osp.join(KINETICS_DIR, predictor, "processed", f"{parameter}.parquet")
    if not osp.exists(path):
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame = frame.dropna(subset=[parameter])
    frame = frame[frame[parameter] > 0]
    frame["substrate"] = frame["substrate_name"].str.strip().str.lower()
    return frame.groupby(["uniprot", "substrate"], as_index=False).agg(
        predicted=(parameter, lambda v: float(np.exp(np.log(v).mean())))
    )


def score(joined: pd.DataFrame) -> dict:
    """Accuracy of predicted against measured, all of it on the log10 scale.

    A ratio is the only meaningful error for a quantity spanning six orders of magnitude,
    so the location statistic is the median absolute log10 ratio in decades and the
    tolerance statistic is the fraction within a factor of ten.
    """
    measured = np.log10(joined["measured"].to_numpy())
    predicted = np.log10(joined["predicted"].to_numpy())
    residual = predicted - measured
    return {
        "n_pairs": int(len(joined)),
        "spearman": float(stats.spearmanr(measured, predicted).statistic),
        "pearson_log10": float(stats.pearsonr(measured, predicted).statistic),
        "median_abs_error_decades": float(np.median(np.abs(residual))),
        "median_bias_decades": float(np.median(residual)),
        "frac_within_1_decade": float(np.mean(np.abs(residual) < 1.0)),
        "frac_within_half_decade": float(np.mean(np.abs(residual) < 0.5)),
    }


def main() -> None:
    report: dict = {
        "leakage_caveat": (
            "Every predictor trained on BRENDA and/or SABIO-RK, from which the Open Enzyme "
            "Database is itself assembled. An unknown, probably large fraction of these "
            "pairs was seen in training, so these are an UPPER BOUND on held-out accuracy."
        ),
        "enzengdb": {
            "citation_key": "longEnzymeEngineeringDatabase2026",
            "doi": "10.1093/nar/gkaf1142",
            "comparable": False,
            "reason": (
                "No kcat or KM field (all assay outputs flattened to one untyped "
                "fitness_value), no EC, no organism field, and zero S. cerevisiae content "
                "across the paper, 140 experiment records and 1,342 analysis rows. Overlap "
                "with the 3,728 yeast-GEM catalytic units is zero."
            ),
            "verified_on": "2026-09-02",
        },
        "temperature_window_c": [TEMP_MIN_C, TEMP_MAX_C],
        "parameters": {},
    }

    all_rows = []
    for parameter, predictors in (
        ("k_cat", KCAT_PREDICTORS),
        ("K_M", KM_PREDICTORS),
    ):
        measured = load_measured(parameter)
        per_predictor = {}
        for predictor in predictors:
            predicted = load_predicted(predictor, parameter)
            if predicted.empty:
                continue
            joined = measured.merge(predicted, on=["uniprot", "substrate"], how="inner")
            if len(joined) < 5:
                per_predictor[predictor] = {"n_pairs": int(len(joined))}
                continue
            per_predictor[predictor] = score(joined)
            frame = joined.assign(predictor=predictor, parameter=parameter)
            all_rows.append(frame)
        report["parameters"][parameter] = {
            "n_measured_pairs_available": int(len(measured)),
            "predictors": per_predictor,
        }

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(osp.join(RESULTS, "predictor_vs_measured.csv"), index=False)

    # Is the measurable subset representative? Accuracy can only be scored where a measured
    # value exists, and those are the well-studied enzymes. If the predictors agree far
    # better there than they do model-wide, then the accuracy above is computed on the easy
    # tail and does not transfer. This compares like with like: the same across-predictor
    # spread statistic, on the joined pairs against all pairs.
    for parameter, predictors in (("k_cat", KCAT_PREDICTORS), ("K_M", KM_PREDICTORS)):
        frames = []
        for predictor in predictors:
            frame = load_predicted(predictor, parameter)
            if frame.empty:
                continue
            frames.append(frame.assign(predictor=predictor))
        stacked = pd.concat(frames, ignore_index=True)
        stacked["log10"] = np.log10(stacked["predicted"])
        spread = (
            stacked.groupby(["uniprot", "substrate"])["log10"]
            .agg(["min", "max", "count"])
            .reset_index()
        )
        spread = spread[spread["count"] >= 2]
        spread["spread_decades"] = spread["max"] - spread["min"]
        measured_pairs = set(
            map(
                tuple,
                combined.loc[
                    combined["parameter"] == parameter, ["uniprot", "substrate"]
                ].to_numpy(),
            )
        )
        on_measured = spread[
            [tuple(x) in measured_pairs for x in spread[["uniprot", "substrate"]].to_numpy()]
        ]
        report["parameters"][parameter]["selection_effect"] = {
            "median_spread_all_pairs_decades": float(spread["spread_decades"].median()),
            "median_spread_measured_pairs_decades": float(
                on_measured["spread_decades"].median()
            ),
            "n_all_pairs": int(len(spread)),
            "n_measured_pairs": int(len(on_measured)),
        }
    with open(osp.join(RESULTS, "predictor_vs_measured.json"), "w") as handle:
        json.dump(report, handle, indent=2)

    for parameter, block in report["parameters"].items():
        print(f"\n{parameter}: {block['n_measured_pairs_available']} measured pairs available")
        print(
            f"  {'predictor':12s} {'n':>5s} {'spearman':>9s} {'MAE dec':>8s} "
            f"{'bias':>7s} {'<1dec':>7s}"
        )
        for name, s in sorted(
            block["predictors"].items(), key=lambda kv: -kv[1].get("spearman", -9)
        ):
            if "spearman" not in s:
                print(f"  {name:12s} {s['n_pairs']:5d}   too few pairs")
                continue
            print(
                f"  {name:12s} {s['n_pairs']:5d} {s['spearman']:+9.3f} "
                f"{s['median_abs_error_decades']:8.2f} {s['median_bias_decades']:+7.2f} "
                f"{s['frac_within_1_decade']:7.1%}"
            )
    print(f"\nwrote {osp.join(RESULTS, 'predictor_vs_measured.json')}")


if __name__ == "__main__":
    main()
