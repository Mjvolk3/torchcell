# experiments/031-env-chemgen-inhibitor-tolerance/scripts/combination_conditions.py
# [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.combination_conditions]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-env-chemgen-inhibitor-tolerance/scripts/combination_conditions
"""Which served conditions dose more than one compound, and are their single agents
measured at the same dose in the same arm?

A two-compound condition is the environment-side analog of a double-gene perturbation: the
same additive null applies, and the same confound ruins it if the single agents were dosed
at other concentrations. This script scans the flattened records for environments carrying
more than one ``small_molecule`` perturbation, splits each into its agents and doses, and
asks whether every agent of the pair is ALSO dosed alone in that arm at the SAME value and
unit. A pair whose singles are dose-matched supports an additive null directly; one whose
singles sit at other doses confounds interaction with dose response.

Writes ``results/combination_conditions.csv`` (one row per multi-compound condition, with
per-agent dose-match flags and the gene count) and prints the dose-matched subset.
"""

from __future__ import annotations

import os
import os.path as osp

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "031-env-chemgen-inhibitor-tolerance", "results"
)
DATASETS = ["vanacloig2022", "hillenmeyer2008_hom", "hillenmeyer2008_het"]


def singles_index(df: pd.DataFrame) -> set[tuple[str, str, str]]:
    """Every (compound, dose value, dose unit) dosed ALONE in this arm."""
    s = df[df["n_small_molecules"] == 1]
    return set(zip(s["compound"], s["dose_value"], s["dose_unit"]))


def main() -> None:
    rows = []
    for name in DATASETS:
        df = pd.read_parquet(osp.join(RESULTS_DIR, f"records_{name}.parquet"))
        singles = singles_index(df)
        multi = df[df["n_small_molecules"] > 1]
        if multi.empty:
            print(f"{name}: no multi-compound condition")
            continue
        grouped = multi.groupby(["compound", "dose_value", "dose_unit"], sort=True)
        for (compound, dose, unit), block in grouped:
            agents = compound.split("|")
            doses = dose.split("|")
            units = unit.split("|")
            matched = [(a, d, u) in singles for a, d, u in zip(agents, doses, units)]
            rows.append(
                {
                    "dataset": name,
                    "n_agents": len(agents),
                    "pair": " + ".join(sorted(agents)),
                    "condition": compound,
                    "doses": dose,
                    "units": unit,
                    "n_records": len(block),
                    "n_genes": block["gene"].nunique(),
                    "n_agents_dosed_alone_same_dose": sum(matched),
                    "dose_matched": all(matched),
                }
            )
    out = pd.DataFrame(rows)
    path = osp.join(RESULTS_DIR, "combination_conditions.csv")
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print()
    for name, block in out.groupby("dataset"):
        ok = block[block["dose_matched"]]
        print(
            f"{name}: {len(block)} multi-compound conditions over "
            f"{block['pair'].nunique()} distinct pairs; {len(ok)} dose-matched over "
            f"{ok['pair'].nunique()} pairs, {int(ok['n_records'].sum())} records"
        )
    print(path)


if __name__ == "__main__":
    main()
