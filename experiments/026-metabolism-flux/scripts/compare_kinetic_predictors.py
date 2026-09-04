# experiments/026-metabolism-flux/scripts/compare_kinetic_predictors.py
# [[experiments.026-metabolism-flux.scripts.compare_kinetic_predictors]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/compare_kinetic_predictors.py

r"""Compare every kinetic predictor against every other, and against measured values.

Wu et al. report no accuracy numbers for these models, and the ranking lives in a
Supplementary Table the mirror does not hold, so choosing one on impression would be
inventing a result. Running all of them and comparing them is what replaces that.

TWO COMPARISONS THAT ANSWER DIFFERENT QUESTIONS
-------------------------------------------------
**Model against model, over all shared pairs.** Almost none of these pairs has a measured
value, so this is the only comparison available on the population the flux layer actually
consumes. Disagreement here is the honest uncertainty on any individual predicted value,
and it is the number that should be quoted when a single :math:`k_{cat}` is used.

**Model against measured, over the Open Enzyme Database.** This is NOT a generalization
estimate. The OED aggregates BRENDA and Sabio-RK, which is what these models were trained
on, so a matched pair is almost certainly inside their training sets. It measures
memorization and is reported as such. The join is on the (enzyme, substrate) PAIR with
canonicalized SMILES, because joining on enzyme alone compares a prediction for one
substrate against a measurement for another, which inflates one correlation here from
0.62 to 0.98 and deflates another from 0.62 to 0.12.
"""

import itertools
import json
import os
import os.path as osp
from typing import cast

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = cast(str, os.getenv("DATA_ROOT"))
EXPERIMENT_ROOT = cast(str, os.getenv("EXPERIMENT_ROOT"))
KINETICS = osp.join(DATA_ROOT, "data", "torchcell", "kinetics")
OED = osp.join(
    DATA_ROOT,
    "data/enzyme_kinetics/open_enzyme_database/scerevisiae/oed_records.json",
)
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
PAIR_KEYS = ["unit_id", "gene_id", "substrate_met_id"]


def canonical(smiles: object) -> str | None:
    """Canonical SMILES, so two spellings of one molecule join."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    if not isinstance(smiles, str):
        return None
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule) if molecule is not None else None


def log_stats(predicted: list[float], observed: list[float]) -> dict[str, float]:
    """Correlation and typical error in log10 space, where kinetic parameters live."""
    p = np.log10(np.asarray(predicted, dtype=float))
    o = np.log10(np.asarray(observed, dtype=float))
    keep = np.isfinite(p) & np.isfinite(o)
    p, o = p[keep], o[keep]
    if len(p) < 3:
        return {"n": int(len(p))}
    return {
        "n": int(len(p)),
        "pearson_log10": float(np.corrcoef(p, o)[0, 1]),
        "spearman": float(pd.Series(p).corr(pd.Series(o), method="spearman")),
        "median_abs_error_decades": float(np.median(np.abs(p - o))),
    }


def available() -> dict[str, dict[str, pd.DataFrame]]:
    """Every built table, as {parameter: {predictor: frame}}."""
    found: dict[str, dict[str, pd.DataFrame]] = {"k_cat": {}, "K_M": {}}
    if not osp.isdir(KINETICS):
        return found
    for predictor in sorted(os.listdir(KINETICS)):
        for parameter in ("k_cat", "K_M"):
            path = osp.join(KINETICS, predictor, "processed", f"{parameter}.parquet")
            if osp.exists(path):
                found[parameter][predictor] = pd.read_parquet(path)
    return found


def main() -> None:
    """Write the pairwise agreement matrix and the memorization check."""
    tables = available()

    pairwise: dict[str, list[dict[str, object]]] = {}
    for parameter, by_predictor in tables.items():
        rows = []
        for left, right in itertools.combinations(sorted(by_predictor), 2):
            a = by_predictor[left][PAIR_KEYS + [parameter]]
            b = by_predictor[right][PAIR_KEYS + [parameter]]
            merged = a.merge(b, on=PAIR_KEYS, suffixes=("_l", "_r"))
            if len(merged) < 3:
                continue
            stats = log_stats(
                merged[f"{parameter}_l"].tolist(), merged[f"{parameter}_r"].tolist()
            )
            rows.append({
                "left": left,
                "right": right,
                "median_left": float(merged[f"{parameter}_l"].median()),
                "median_right": float(merged[f"{parameter}_r"].median()),
                **stats,
            })
        pairwise[parameter] = rows

    records = [r for r in json.load(open(OED)) if r["enzymetype"] == "wildtype"]
    measured: dict[tuple[str, str], dict[str, list[float]]] = {}
    for record in records:
        smiles = canonical(record.get("smiles"))
        if smiles is None:
            continue
        entry = measured.setdefault(
            (record["uniprot"], smiles), {"k_cat": [], "K_M": []}
        )
        entry["k_cat"].append(record["kcat_value"])
        entry["K_M"].append(record["km_value"])

    against_measured: dict[str, dict[str, object]] = {}
    for parameter, by_predictor in tables.items():
        for predictor, frame in by_predictor.items():
            work = frame.copy()
            work["canonical"] = [canonical(s) for s in work["smiles"]]
            predicted, observed = [], []
            for row in work.itertuples():
                key = (row.uniprot, row.canonical)
                if key in measured and measured[key][parameter]:
                    predicted.append(getattr(row, parameter))
                    observed.append(float(np.median(measured[key][parameter])))
            against_measured[f"{predictor}_{parameter}"] = log_stats(predicted, observed)

    summary = {
        "model_vs_model": {
            "note": "All shared pairs. Almost none has a measured value, so this is the "
            "only comparison available on the population the flux layer consumes. The "
            "disagreement here is the uncertainty on any single predicted value.",
            **pairwise,
        },
        "vs_measured": {
            "note": "NOT held out. The Open Enzyme Database aggregates BRENDA and "
            "Sabio-RK, which both models trained on, so these pairs are almost certainly "
            "inside their training sets. This measures memorization, joined on the "
            "(enzyme, substrate) pair with canonicalized SMILES.",
            "n_measured_pairs_available": len(measured),
            **against_measured,
        },
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(osp.join(RESULTS, "kinetic_predictor_comparison.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    for parameter, rows in pairwise.items():
        if not rows:
            continue
        print(f"\n=== {parameter}: model against model ===")
        print(f"{'left':12s} {'right':12s} {'n':>6s} {'pearson':>8s} {'spearman':>9s} "
              f"{'|diff| dec':>11s}")
        for row in rows:
            print(f"{row['left']:12s} {row['right']:12s} {row['n']:6d} "
                  f"{row.get('pearson_log10', float('nan')):8.3f} "
                  f"{row.get('spearman', float('nan')):9.3f} "
                  f"{row.get('median_abs_error_decades', float('nan')):11.2f}")
    print("\n=== against measured (memorization, not generalization) ===")
    for name, stats in against_measured.items():
        if stats.get("n", 0) >= 3:
            print(f"  {name:22s} n={stats['n']:3d}  pearson {stats['pearson_log10']:6.3f}  "
                  f"median |err| {stats['median_abs_error_decades']:.2f} decades")


if __name__ == "__main__":
    main()
