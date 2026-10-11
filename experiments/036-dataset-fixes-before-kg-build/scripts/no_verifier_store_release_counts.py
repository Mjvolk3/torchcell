# experiments/036-dataset-fixes-before-kg-build/scripts/no_verifier_store_release_counts.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.no_verifier_store_release_counts]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/no_verifier_store_release_counts
"""Derive the record-count oracle of each #889 store from the table its paper released.

Issue #889 enrolls 16 stores that no L0-L4 verifier reached. Each enrollment needs an
``expected_count`` in its ``torchcell.verification.runners`` registry entry, and that
number must come from the release, not from the store it judges. This script reads each
released table from the store's own ``raw/`` directory (sha256 checked against the pin
the registry carries), counts the rows the paper's own column definitions select, and
writes the counts beside the store's LMDB entry count:

- Costanzo 2016 Data File S1 (``SGA_DAmP``, ``SGA_ExE``, ``SGA_ExN_NxE``, ``SGA_NxN``):
  every row is one tested pair at one array type and temperature, DMF and DMI alike; the
  SMF spreadsheet gives one cell per (strain, temperature) with a stated fitness and SD.
- Kuzmin 2018 Data S1: column 5 ``Combined mutant type`` splits digenic from trigenic
  rows; the DMF store also holds one record per distinct double-mutant QUERY strain of
  the trigenic rows that states a query fitness.
- Kuzmin 2020 Tables S1 + S3: the same split, both screens kept; Table S5 gives the
  single-mutant query fitness and the double-mutant query strains.
- SGD: the inviable null S288C annotations of the per-gene JSON release.
- SynLethDB 2.0: the rows of ``Yeast_SL.csv`` / ``Yeast_SR.csv`` minus the rows the
  loader's ``dropped_records.json`` ledger names.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/no_verifier_store_release_counts.py

Output: ``experiments/036-dataset-fixes-before-kg-build/results/no_verifier_store_release_counts.json``.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from typing import Any

import lmdb
import pandas as pd
from dotenv import load_dotenv

from torchcell.verification.released import (
    KUZMIN_COMBINED_TYPE,
    drifted_files,
    sgd_inviable_null_annotations,
)
from torchcell.verification.runners import (
    GENE_ESSENTIALITY_DATASETS,
    GENE_INTERACTION_DATASETS,
    SYNTHETIC_PAIR_DATASETS,
)

RESULTS = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))),
    "results",
    "no_verifier_store_release_counts.json",
)


def store_entries(abs_root: str) -> int:
    """The number of entries in a store's LMDB, opened read-only."""
    env = lmdb.open(osp.join(abs_root, "processed", "lmdb"), readonly=True, lock=False)
    try:
        return int(env.stat()["entries"])
    finally:
        env.close()


def costanzo2016(data_root: str) -> dict[str, Any]:
    """Data File S1 row counts and the SMF spreadsheet's stated cells."""
    raw = osp.join(data_root, "data/torchcell/dmi_costanzo2016/raw")
    spec = GENE_INTERACTION_DATASETS["dmi_costanzo2016"]
    if drifted_files(raw, spec["released_files"]):
        raise ValueError("Costanzo 2016 Data File S1 drifted from its pin")
    per_file = {
        released.name: int(
            pd.read_csv(osp.join(raw, released.name), sep="\t", usecols=[0]).shape[0]
        )
        for released in spec["released_files"]
    }
    smf = pd.read_excel(
        osp.join(
            data_root,
            "data/torchcell/smf_costanzo2016/raw/strain_ids_and_single_mutant_fitness.xlsx",
        )
    )
    cells = []
    for temperature in (26, 30):
        fitness = f"Single mutant fitness ({temperature}°)"
        sd = f"Single mutant fitness ({temperature}°) stddev"
        stated = smf[smf[fitness].notna() & smf[sd].notna()]
        cells.append(
            stated[
                ["Strain ID", "Systematic gene name", "Allele/Gene name", fitness, sd]
            ]
            .set_axis(["strain", "orf", "allele", "fitness", "sd"], axis=1)
            .assign(temperature=temperature)
        )
    smf_cells = pd.concat(cells, ignore_index=True)
    return {
        "data_file_s1_rows_per_file": per_file,
        "data_file_s1_rows": sum(per_file.values()),
        "smf_strain_rows": int(smf.shape[0]),
        "smf_stated_cells": int(smf_cells.shape[0]),
        "smf_stated_cells_distinct": int(smf_cells.drop_duplicates().shape[0]),
    }


def kuzmin2018(data_root: str) -> dict[str, Any]:
    """Data S1 digenic / trigenic rows and the trigenic double-mutant query strains."""
    raw = osp.join(data_root, "data/torchcell/dmi_kuzmin2018/raw")
    spec = GENE_INTERACTION_DATASETS["dmi_kuzmin2018"]
    if drifted_files(raw, spec["released_files"]):
        raise ValueError("Kuzmin 2018 Data S1 drifted from its pin")
    table = pd.read_csv(osp.join(raw, "aao1729_data_s1.tsv"), sep="\t")
    digenic = table[table[KUZMIN_COMBINED_TYPE] == "digenic"]
    trigenic = table[table[KUZMIN_COMBINED_TYPE] == "trigenic"]
    query_strains = trigenic[trigenic["Query single/double mutant fitness"].notna()][
        "Query strain ID"
    ].nunique()
    return {
        "rows": int(table.shape[0]),
        "digenic_rows": int(digenic.shape[0]),
        "trigenic_rows": int(trigenic.shape[0]),
        "trigenic_query_strains_with_fitness": int(query_strains),
        "distinct_array_alleles": int(table["Array allele name"].nunique()),
        "array_alleles_with_array_fitness": int(
            table.groupby("Array allele name")["Array single mutant fitness"]
            .agg(lambda column: column.notna().any())
            .sum()
        ),
        "distinct_digenic_query_alleles": int(digenic["Query allele name"].nunique()),
        "digenic_query_alleles_with_query_fitness": int(
            digenic.groupby("Query allele name")["Query single/double mutant fitness"]
            .agg(lambda column: column.notna().any())
            .sum()
        ),
    }


def kuzmin2020(data_root: str) -> dict[str, Any]:
    """Tables S1 + S3 digenic / trigenic rows, and Table S5's query strains."""
    raw = osp.join(data_root, "data/torchcell/dmi_kuzmin2020/raw")
    spec = GENE_INTERACTION_DATASETS["dmi_kuzmin2020"]
    if drifted_files(raw, spec["released_files"]):
        raise ValueError("Kuzmin 2020 Tables S1/S3 drifted from their pins")
    out: dict[str, Any] = {}
    trigenic_frames = []
    for table_name in ("aaz5667-Table-S1.xlsx", "aaz5667-Table-S3.xlsx"):
        table = pd.read_excel(
            osp.join(raw, table_name),
            skiprows=1,
            usecols=[
                KUZMIN_COMBINED_TYPE,
                "Query strain ID",
                "Query single/double mutant fitness",
            ],
        )
        out[table_name] = {
            "digenic_rows": int((table[KUZMIN_COMBINED_TYPE] == "digenic").sum()),
            "trigenic_rows": int((table[KUZMIN_COMBINED_TYPE] == "trigenic").sum()),
        }
        trigenic_frames.append(table[table[KUZMIN_COMBINED_TYPE] == "trigenic"])
    trigenic = pd.concat(trigenic_frames, ignore_index=True)
    s5 = pd.read_excel(osp.join(raw, "aaz5667-Table-S5.xlsx"), skiprows=1)
    double = s5[s5["Mutant type"] == "Double mutant"]
    s5_stated = set(double[double["Fitness"].notna()]["Query Strain ID"])
    strains = trigenic.groupby("Query strain ID").agg(
        any_stated=(
            "Query single/double mutant fitness",
            lambda column: column.notna().any(),
        )
    )
    tm = strains.index.to_series().str.rsplit("_", n=1).str[-1]
    with_fitness = tm.isin(s5_stated) | strains["any_stated"]
    out["s5_double_mutant_rows"] = int(double.shape[0])
    out["s5_double_mutant_rows_with_fitness"] = len(s5_stated)
    out["digenic_rows"] = sum(
        out[t]["digenic_rows"]
        for t in ("aaz5667-Table-S1.xlsx", "aaz5667-Table-S3.xlsx")
    )
    out["trigenic_rows"] = sum(
        out[t]["trigenic_rows"]
        for t in ("aaz5667-Table-S1.xlsx", "aaz5667-Table-S3.xlsx")
    )
    out["trigenic_query_strains"] = int(strains.shape[0])
    out["trigenic_query_strains_with_fitness"] = int(with_fitness.sum())
    singles = s5[s5["Mutant type"] == "Single mutant"]
    out["s5_single_mutant_rows"] = int(singles.shape[0])
    out["s5_single_mutant_rows_with_fitness"] = int(singles["Fitness"].notna().sum())
    return out


def sgd(data_root: str) -> dict[str, Any]:
    """The SGD per-gene JSON release's inviable null S288C annotations."""
    spec = GENE_ESSENTIALITY_DATASETS["gene_essentiality_sgd"]
    annotations = sgd_inviable_null_annotations(
        osp.join(data_root, spec["sgd_genes_dir"])
    )
    return {
        "annotations": sum(annotations.values()),
        "genes": len({gene for gene, _ in annotations}),
        "distinct_gene_publication_pairs": len(annotations),
    }


def synlethdb(data_root: str) -> dict[str, Any]:
    """Each SynLethDB file's data rows and the loader ledger's drops."""
    out: dict[str, Any] = {}
    for name, spec in SYNTHETIC_PAIR_DATASETS.items():
        abs_root = osp.join(data_root, spec["root"])
        released = spec["released_file"]
        if drifted_files(osp.join(abs_root, "raw"), [released]):
            raise ValueError(f"{released.name} drifted from its pin")
        rows = int(pd.read_csv(osp.join(abs_root, "raw", released.name)).shape[0])
        with open(
            osp.join(abs_root, "preprocess", "dropped_records.json"), encoding="utf-8"
        ) as handle:
            ledger = json.load(handle)
        out[name] = {
            "released_rows": rows,
            "ledger_source_records": ledger["source_records"],
            "ledger_dropped": ledger["dropped_records"],
            "released_minus_dropped": rows - ledger["dropped_records"],
        }
    return out


def main() -> None:
    """Count every release, read every store's entry count, and write the JSON."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    stores = [
        "smf_costanzo2016",
        "dmf_costanzo2016",
        "dmi_costanzo2016",
        "smf_kuzmin2018",
        "dmf_kuzmin2018",
        "tmf_kuzmin2018",
        "dmi_kuzmin2018",
        "tmi_kuzmin2018",
        "smf_kuzmin2020",
        "dmf_kuzmin2020",
        "tmf_kuzmin2020",
        "dmi_kuzmin2020",
        "tmi_kuzmin2020",
        "gene_essentiality_sgd",
        "syn_leth_db_yeast",
        "syn_rescue_db_yeast",
    ]
    result = {
        "release": {
            "costanzo2016": costanzo2016(data_root),
            "kuzmin2018": kuzmin2018(data_root),
            "kuzmin2020": kuzmin2020(data_root),
            "sgd": sgd(data_root),
            "synlethdb": synlethdb(data_root),
        },
        "store_entries": {
            name: store_entries(osp.join(data_root, "data/torchcell", name))
            for name in stores
        },
    }
    os.makedirs(osp.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
