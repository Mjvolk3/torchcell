# experiments/033-env-chemgen-pooled/scripts/store_against_plan.py
# [[experiments.033-env-chemgen-pooled.scripts.store_against_plan]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/033-env-chemgen-pooled/scripts/store_against_plan
"""Check what the 031 plan assumed against what build 001 of the pooled store holds.

The 031 documents (``notes-tex/031-inhibitor-tolerance-data`` and
``notes-tex/031-unified-representation``) measured everything on the dev-tree loaders,
one row per served record, and mostly on the (gene, compound) pair. The model trains on
the built store, one entry per (genotype, environment) cell. This script reads the cell
table ``flatten_cells.py`` wrote and re-measures each plan claim on the store itself.

Ten tables, each one question:

``store_axes.csv``            what each source holds on the genotype, environment and
                              dose axes of the representation.
``unit_of_analysis.csv``      the store's cell against the plan's (gene, compound) pair:
                              counts, spread and skew at both units.
``ploidy_pairs.csv``          how many Hoepfner cells are measured in both ploidy arms,
                              and how the two arms agree.
``label_policy.csv``          which measurement of a repeated cell the label table holds.
``within_cell_pairs.csv``     every pair of screens that measured one environment, and
``within_cell_reliability.csv`` the reliability that repeat agreement gives per source.
``standardized_target.csv``   the pooled target after orienting and standardizing.
``vanacloig_folds.csv``       the 41 compound-cold folds with a ceiling from the store's
                              own standard errors, against the plan's.
``embedding_coverage_store.csv`` which cells have a compound vector in the plan's
                              embedding tables.
``cross_source_cells.csv``    shared (gene, compound) pairs and their agreement.
``environment_vocabulary.csv`` environments, compounds and doses per source.

``--plan-results`` is the ``results`` directory of experiment 031. It is a required
argument and not a default because 031 is on its own branch (PR #436) until it lands, so
the directory is not under this worktree's EXPERIMENT_ROOT. The commit it was read at is
written to ``store_against_plan_provenance.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import subprocess
from itertools import combinations

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import kurtosis, pearsonr, skew, spearmanr

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

BUILD_ROOT = "/db/experiments/033-env-chemgen-pooled-001-pooled-build"
LABEL_TABLE = osp.join(BUILD_ROOT, "processed", "label_df.parquet")
DEFAULT_CELLS = osp.join(
    DATA_ROOT,
    "experiments",
    "033-env-chemgen-pooled",
    "cell_table",
    "cell_table.parquet",
)
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "033-env-chemgen-pooled", "results")

#: Source order is the 031 planning order: application target first, then by size.
DISPLAY: dict[str, str] = {
    "EnvChemgenVanacloig2022Dataset": "Vanacloig 2022",
    "HetHillenmeyer2008Dataset": "Hillenmeyer HET",
    "EnvChemgenHoepfner2014Dataset": "Hoepfner 2014",
    "EnvChemgenWildenhain2015Dataset": "Wildenhain 2015",
}
#: The name each source carries in the 031 result files.
PLAN_NAME: dict[str, str] = {
    "EnvChemgenVanacloig2022Dataset": "vanacloig2022",
    "HetHillenmeyer2008Dataset": "hillenmeyer2008_het",
    "EnvChemgenHoepfner2014Dataset": "hoepfner2014",
    "EnvChemgenWildenhain2015Dataset": "wildenhain2015",
}
#: The sign a SICK strain carries in the served response, from each source's own
#: definition as recorded by its loader (031 ``dataset_joinability.POLARITY``).
SICK_SIGN: dict[str, int] = {
    "EnvChemgenVanacloig2022Dataset": -1,
    "HetHillenmeyer2008Dataset": +1,
    "EnvChemgenHoepfner2014Dataset": -1,
    "EnvChemgenWildenhain2015Dataset": -1,
}
#: A screen pair enters the reliability estimate only over at least this many genes.
MIN_GENES_PER_PAIR = 100
#: Columns of ``within_cell_pairs.csv``, declared so an empty table keeps its schema.
PAIR_COLUMNS = [
    "dataset",
    "functional_dose",
    "environment_id",
    "compound",
    "concentration",
    "unit",
    "screen_a",
    "screen_b",
    "n_genes",
    "pearson",
    "spearman",
]


def counts(series: pd.Series) -> str:
    """Distinct values with their cell counts, as ``value:count|value:count``."""
    tally = series.astype("string").fillna("unstated").value_counts().sort_index()
    return "|".join(f"{value}:{n}" for value, n in tally.items())


def measurements(table: pd.DataFrame) -> pd.DataFrame:
    """One row per folded measurement, keyed so repeated screen ids stay distinct."""
    frame = table[
        [
            "index",
            "dataset",
            "query_gene",
            "functional_dose",
            "environment_id",
            "inchikeys",
            "compound_names",
            "conc_values",
            "conc_units",
            "n_compounds",
            "n_with_inchikey",
            "responses",
            "response_ses",
            "screen_ids",
        ]
    ].explode(["responses", "response_ses", "screen_ids"], ignore_index=True)
    frame = frame.rename(
        columns={
            "responses": "response",
            "response_ses": "response_se",
            "screen_ids": "screen_id",
        }
    )
    frame["response"] = frame["response"].astype(float)
    frame["response_se"] = frame["response_se"].astype(float)
    frame["screen_id"] = frame["screen_id"].astype("string").fillna("unstated")
    occurrence = frame.groupby(["index", "screen_id"]).cumcount()
    frame["measurement_key"] = frame["screen_id"] + "#" + occurrence.astype(str)
    return frame


def store_axes(table: pd.DataFrame, plan_dose: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, display in DISPLAY.items():
        t = table[table["dataset"] == name]
        keys = t["inchikeys"].str.split("|").explode()
        single = t[t["n_compounds"] == 1]
        plan = plan_dose.loc[PLAN_NAME[name]]
        rows.append(
            {
                "dataset": display,
                "plan_records": int(plan["records"]),
                "measurements": int(t["n_measurements"].sum()),
                "cells": len(t),
                "genes": t["query_gene"].nunique(),
                "environments": t["environment_id"].nunique(),
                "plan_compounds": int(plan["compounds"]),
                "compounds": keys[keys != ""].nunique(),
                "cells_no_compound": int((t["n_compounds"] == 0).sum()),
                "cells_two_compounds": int((t["n_compounds"] >= 2).sum()),
                "cells_compound_without_inchikey": int(
                    (t["n_with_inchikey"] < t["n_compounds"]).sum()
                ),
                "functional_dose": counts(t["functional_dose"]),
                "ploidy": counts(t["ploidy"]),
                "perturbation_type": counts(t["perturbation_type"]),
                "base_medium": counts(t["base_medium"]),
                "temperature_c": counts(t["temperature_c"]),
                "aerobicity": counts(t["aerobicity"]),
                "duration_hours": counts(t["duration_hours"]),
                "duration_generations": counts(t["duration_generations"]),
                "dose_basis": counts(single["conc_bases"].replace("", pd.NA)),
                "frac_cells_with_dose_value": float(
                    (single["conc_values"] != "").sum() / len(t)
                ),
                "frac_cells_with_molar": float(t["log10_molar"].notna().mean()),
                "plan_frac_molar": float(plan["frac_convertible_to_molar"]),
            }
        )
    return pd.DataFrame(rows)


def pair_means(meas: pd.DataFrame, name: str) -> pd.Series:
    """Mean response per (gene, InChIKey) over every measurement, the plan's unit."""
    m = meas[
        (meas["dataset"] == name)
        & (meas["n_compounds"] == 1)
        & (meas["n_with_inchikey"] == 1)
    ]
    return m.groupby(["query_gene", "inchikeys"])["response"].mean()


def unit_of_analysis(
    table: pd.DataFrame, meas: pd.DataFrame, plan_dist: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for name, display in DISPLAY.items():
        store = table.loc[table["dataset"] == name, "label"].to_numpy()
        pairs = pair_means(meas, name).to_numpy()
        plan = plan_dist.loc[PLAN_NAME[name]]
        rows.append(
            {
                "dataset": display,
                "store_cells": len(store),
                "store_sd": float(np.std(store, ddof=1)),
                "store_skew": float(skew(store)),
                "pair_cells": len(pairs),
                "pair_sd": float(np.std(pairs, ddof=1)),
                "pair_skew": float(skew(pairs)),
                "plan_pair_cells": int(plan["cells"]),
                "plan_pair_sd": float(plan["sd"]),
                "plan_pair_skew": float(plan["skew"]),
                "store_cells_per_pair": len(store) / len(pairs),
            }
        )
    return pd.DataFrame(rows)


def ploidy_pairs(meas: pd.DataFrame) -> pd.DataFrame:
    """Hoepfner cells measured heterozygous AND homozygous, at three match levels."""
    h = meas[
        (meas["dataset"] == "EnvChemgenHoepfner2014Dataset")
        & (meas["n_compounds"] == 1)
    ]
    rows = []
    levels = {
        "same environment": ["environment_id"],
        "same compound and concentration": ["inchikeys", "conc_values"],
        "same compound": ["inchikeys"],
    }
    for level, keys in levels.items():
        arm = (
            h.groupby(["query_gene", *keys, "functional_dose"])["response"]
            .mean()
            .unstack("functional_dose")
        )
        both = arm.dropna(subset=[0.0, 0.5])
        row: dict[str, object] = {
            "level": level,
            "cells_homozygous": int(arm[0.0].notna().sum()),
            "cells_heterozygous": int(arm[0.5].notna().sum()),
            "cells_both": len(both),
        }
        if len(both) >= MIN_GENES_PER_PAIR:
            row["spearman"] = float(spearmanr(both[0.0], both[0.5])[0])
            row["pearson"] = float(pearsonr(both[0.0], both[0.5])[0])
            per_compound = [
                spearmanr(g[0.0], g[0.5])[0]
                for _, g in both.groupby(level="inchikeys")
                if len(g) >= MIN_GENES_PER_PAIR
            ]
            row["compounds_scored"] = len(per_compound)
            row["spearman_per_compound_median"] = float(np.median(per_compound))
            # a homozygous hit is a cell in the bottom 5 percent of its arm
            hom_hit = both[0.0] <= both[0.0].quantile(0.05)
            het_hit = both[0.5] <= both[0.5].quantile(0.05)
            row["hom_hits"] = int(hom_hit.sum())
            row["hom_hits_also_het_hits"] = int((hom_hit & het_hit).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def label_policy(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, display in DISPLAY.items():
        t = table[table["dataset"] == name]
        multi = t[t["n_measurements"] > 1]
        row: dict[str, object] = {
            "dataset": display,
            "cells": len(t),
            "cells_repeated": len(multi),
            "frac_cells_repeated": len(multi) / len(t),
        }
        if len(multi) > 0:
            label = multi["label"].to_numpy()
            first = multi["responses"].map(lambda r: r[0]).to_numpy(dtype=float)
            last = multi["responses"].map(lambda r: r[-1]).to_numpy(dtype=float)
            mean = multi["responses"].map(np.mean).to_numpy(dtype=float)
            spread = multi["responses"].map(np.ptp).to_numpy(dtype=float)
            row.update(
                {
                    "frac_label_is_last": float(np.isclose(label, last).mean()),
                    "frac_label_is_first": float(np.isclose(label, first).mean()),
                    "frac_label_is_mean": float(np.isclose(label, mean).mean()),
                    "median_range_over_source_sd": float(
                        np.median(spread) / np.std(t["label"], ddof=1)
                    ),
                    "pearson_label_vs_mean": float(pearsonr(label, mean)[0]),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def within_cell_pairs(meas: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Every pair of screens of one environment and ploidy arm, correlated over genes."""
    repeated = table.loc[table["n_measurements"] > 1, "index"]
    m = meas[meas["index"].isin(repeated)]
    rows = []
    for (name, env_id, dose), g in m.groupby(
        ["dataset", "environment_id", "functional_dose"], sort=False
    ):
        wide = g.pivot(index="query_gene", columns="measurement_key", values="response")
        for a, b in combinations(wide.columns, 2):
            both = wide[[a, b]].dropna()
            if len(both) < MIN_GENES_PER_PAIR:
                continue
            rows.append(
                {
                    "dataset": DISPLAY[name],
                    "functional_dose": dose,
                    "environment_id": env_id,
                    "compound": g["compound_names"].iloc[0],
                    "concentration": g["conc_values"].iloc[0],
                    "unit": g["conc_units"].iloc[0],
                    "screen_a": a,
                    "screen_b": b,
                    "n_genes": len(both),
                    "pearson": float(pearsonr(both[a], both[b])[0]),
                    "spearman": float(spearmanr(both[a], both[b])[0]),
                }
            )
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def within_cell_reliability(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (display, dose), g in pairs.groupby(["dataset", "functional_dose"]):
        rows.append(
            {
                "dataset": display,
                "functional_dose": dose,
                "environments": g["environment_id"].nunique(),
                "compounds": g["compound"].nunique(),
                "screen_pairs": len(g),
                "pearson_median": float(g["pearson"].median()),
                "pearson_q25": float(g["pearson"].quantile(0.25)),
                "pearson_q75": float(g["pearson"].quantile(0.75)),
                "spearman_median": float(g["spearman"].median()),
                "ceiling_r_truth": float(np.sqrt(max(g["pearson"].median(), 0.0))),
                "top_compound": g["compound"].value_counts().index[0],
                "top_compound_pairs": int(g["compound"].value_counts().iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def standardized_target(table: pd.DataFrame) -> pd.DataFrame:
    """The pooled target after the plan's transform, and what the transform leaves."""
    total_raw = sum(
        float(np.sum((t["label"] - t["label"].mean()) ** 2))
        for _, t in table.groupby("dataset")
    )
    rows = []
    for name, display in DISPLAY.items():
        y = table.loc[table["dataset"] == name, "label"].to_numpy()
        oriented = -SICK_SIGN[name] * y
        z = (oriented - oriented.mean()) / oriented.std(ddof=1)
        mad = np.median(np.abs(oriented - np.median(oriented)))
        robust = (oriented - np.median(oriented)) / (1.4826 * mad)
        tail = np.abs(z) > 3
        rows.append(
            {
                "dataset": display,
                "orientation": -SICK_SIGN[name],
                "cells": len(y),
                "share_of_cells": len(y) / len(table),
                "share_of_raw_squared_error": float(
                    np.sum((y - y.mean()) ** 2) / total_raw
                ),
                "excess_kurtosis": float(kurtosis(z)),
                "skew_oriented": float(skew(z)),
                "z_q001": float(np.quantile(z, 0.001)),
                "z_q01": float(np.quantile(z, 0.01)),
                "z_q99": float(np.quantile(z, 0.99)),
                "z_q999": float(np.quantile(z, 0.999)),
                "z_min": float(z.min()),
                "frac_cells_beyond_3sd": float(tail.mean()),
                "share_of_own_squared_error_beyond_3sd": float(
                    np.sum(z[tail] ** 2) / np.sum(z**2)
                ),
                "sd_over_robust_sd": float(oriented.std(ddof=1) / (1.4826 * mad)),
                "robust_z_q01": float(np.quantile(robust, 0.01)),
                "robust_z_q99": float(np.quantile(robust, 0.99)),
            }
        )
    return pd.DataFrame(rows)


def vanacloig_folds(table: pd.DataFrame, plan_noise: pd.DataFrame) -> pd.DataFrame:
    """One row per Vanacloig compound: the fold's cells and its ceiling from the store."""
    v = table[table["dataset"] == "EnvChemgenVanacloig2022Dataset"].copy()
    assert (v["n_measurements"] == 1).all(), "a Vanacloig cell folds two measurements"
    v["se"] = v["response_ses"].map(lambda s: s[0]).astype(float)
    rows = []
    for compound, g in v.groupby("compound_names"):
        reliability = 1.0 - float(np.mean(g["se"] ** 2)) / float(g["label"].var())
        rows.append(
            {
                "compound": compound,
                "inchikey": g["inchikeys"].iloc[0],
                "cells": len(g),
                "genes": g["query_gene"].nunique(),
                "dose_basis": g["conc_bases"].iloc[0],
                "frac_with_se": float(g["se"].notna().mean()),
                "reliability": reliability,
                "ceiling_r_truth": float(np.sqrt(max(reliability, 0.0))),
            }
        )
    folds = pd.DataFrame(rows).merge(
        plan_noise[["condition", "reliability", "ceiling_r_truth"]].rename(
            columns={
                "condition": "compound",
                "reliability": "plan_reliability",
                "ceiling_r_truth": "plan_ceiling_r_truth",
            }
        ),
        on="compound",
        how="left",
        validate="one_to_one",
    )
    folds["reliability_minus_plan"] = folds["reliability"] - folds["plan_reliability"]
    return folds.sort_values("ceiling_r_truth", ascending=False, ignore_index=True)


def embedding_coverage(table: pd.DataFrame, embed_dir: str) -> pd.DataFrame:
    cell_keys = table["inchikeys"].str.split("|")
    rows = []
    for file in sorted(os.listdir(embed_dir)):
        if not file.endswith(".npz"):
            continue
        embedded = set(
            np.load(osp.join(embed_dir, file), allow_pickle=True)["inchikey"]
        )
        has_vector = cell_keys.map(
            lambda keys, e=embedded: all(k in e for k in keys)
        ) & (table["n_compounds"] > 0)
        for name, display in DISPLAY.items():
            mask = table["dataset"] == name
            keys = cell_keys[mask].explode()
            keys = set(keys[keys != ""])
            rows.append(
                {
                    "encoder": file.removesuffix(".npz"),
                    "dataset": display,
                    "compounds": len(keys),
                    "compounds_embedded": len(keys & embedded),
                    "cells": int(mask.sum()),
                    "cells_with_every_vector": int(has_vector[mask].sum()),
                    "cells_no_compound": int(
                        (table.loc[mask, "n_compounds"] == 0).sum()
                    ),
                    "cells_compound_without_inchikey": int(
                        (
                            table.loc[mask, "n_with_inchikey"]
                            < table.loc[mask, "n_compounds"]
                        ).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def cross_source(meas: pd.DataFrame, plan_overlap: pd.DataFrame) -> pd.DataFrame:
    cells = {name: pair_means(meas, name) for name in DISPLAY}
    plan = plan_overlap.set_index(["a", "b"])
    rows = []
    for a, b in combinations(DISPLAY, 2):
        joined = cells[a].to_frame("a").join(cells[b].to_frame("b"), how="inner")
        orientation = SICK_SIGN[a] * SICK_SIGN[b]
        rho = float(spearmanr(joined["a"], joined["b"])[0])
        p = plan.loc[(PLAN_NAME[a], PLAN_NAME[b])]
        rows.append(
            {
                "a": DISPLAY[a],
                "b": DISPLAY[b],
                "shared_genes": len(
                    set(cells[a].index.get_level_values(0))
                    & set(cells[b].index.get_level_values(0))
                ),
                "shared_compounds": len(
                    set(cells[a].index.get_level_values(1))
                    & set(cells[b].index.get_level_values(1))
                ),
                "shared_pairs": len(joined),
                "spearman_as_served": rho,
                "spearman_oriented": rho * orientation,
                "plan_shared_pairs": int(p["shared_cells"]),
                "plan_spearman_oriented": float(p["spearman_oriented"]),
            }
        )
    return pd.DataFrame(rows)


def environment_vocabulary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, display in DISPLAY.items():
        t = table[table["dataset"] == name]
        per_env = t.groupby("environment_id").size()
        single = t[(t["n_compounds"] == 1) & (t["n_with_inchikey"] == 1)]
        doses = single.groupby("inchikeys")["conc_values"].nunique()
        envs = single.groupby("inchikeys")["environment_id"].nunique()
        rows.append(
            {
                "dataset": display,
                "environments": len(per_env),
                "cells_per_environment_min": int(per_env.min()),
                "cells_per_environment_median": float(per_env.median()),
                "cells_per_environment_max": int(per_env.max()),
                "compounds": len(doses),
                "doses_per_compound_median": float(doses.median()),
                "doses_per_compound_max": int(doses.max()),
                "environments_per_compound_median": float(envs.median()),
                "environments_per_compound_max": int(envs.max()),
                "compounds_with_one_environment": int((envs == 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-table", default=DEFAULT_CELLS)
    parser.add_argument("--plan-results", required=True)
    parser.add_argument("--results", default=RESULTS_DIR)
    args = parser.parse_args()
    os.makedirs(args.results, exist_ok=True)

    table = pd.read_parquet(args.cell_table)
    labels = pd.read_parquet(LABEL_TABLE).rename(
        columns={"environment_response": "label"}
    )
    table = table.merge(labels, on="index", how="left", validate="one_to_one")
    assert table["label"].notna().all(), "a cell has no label in the label table"
    print(f"cell table: {len(table):,} cells", flush=True)
    meas = measurements(table)
    print(f"measurements: {len(meas):,}", flush=True)

    plan = args.plan_results
    plan_dose = pd.read_csv(osp.join(plan, "dose_axis_summary.csv")).set_index(
        "dataset"
    )
    plan_dist = pd.read_csv(osp.join(plan, "dataset_distributions.csv")).set_index(
        "dataset"
    )
    plan_noise = pd.read_csv(osp.join(plan, "condition_noise_vanacloig2022.csv"))
    plan_overlap = pd.read_csv(osp.join(plan, "cross_dataset_pair_overlap.csv"))

    pairs = within_cell_pairs(meas, table)
    outputs = {
        "store_axes": store_axes(table, plan_dose),
        "unit_of_analysis": unit_of_analysis(table, meas, plan_dist),
        "ploidy_pairs": ploidy_pairs(meas),
        "label_policy": label_policy(table),
        "within_cell_pairs": pairs,
        "within_cell_reliability": within_cell_reliability(pairs),
        "standardized_target": standardized_target(table),
        "vanacloig_folds": vanacloig_folds(table, plan_noise),
        "embedding_coverage_store": embedding_coverage(
            table, osp.join(plan, "embeddings")
        ),
        "cross_source_cells": cross_source(meas, plan_overlap),
        "environment_vocabulary": environment_vocabulary(table),
    }
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for name, frame in outputs.items():
        frame.to_csv(osp.join(args.results, f"{name}.csv"), index=False)
        print(f"\n=== {name} ({len(frame)} rows)")
        print(frame.head(45).to_string(index=False))

    provenance = {
        "cell_table": args.cell_table,
        "cells": len(table),
        "measurements": len(meas),
        "store": BUILD_ROOT,
        "plan_results": osp.abspath(plan),
        "plan_commit": subprocess.run(
            ["git", "-C", plan, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }
    with open(osp.join(args.results, "store_against_plan_provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
