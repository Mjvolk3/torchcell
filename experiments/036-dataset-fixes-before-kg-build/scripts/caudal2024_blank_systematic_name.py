# experiments/036-dataset-fixes-before-kg-build/scripts/caudal2024_blank_systematic_name.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.caudal2024_blank_systematic_name]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/caudal2024_blank_systematic_name.py
"""Measure the Caudal 2024 blank-``systematic_name`` ledger and its served-record impact.

Issue #598. Reads, read-only:

- Datafile 1 from the mirror
  (``$DATA_ROOT/torchcell-library/caudalPantranscriptomeRevealsLarge2024/data/
  final_data_annotated_merged_04052022.tab.zip``, sha256 8b55ccd7...) through the
  loader's own ``read_caudal_table`` / ``restrict_to_built_isolates`` /
  ``resolve_gene_ids``, so the numbers are what a build applies;
- Peter's presence matrix through the genomes tier (the 1011-isolate panel);
- Supplementary Table 2 from the mirror (``si/41588_2024_1769_MOESM3_ESM.xlsx``), to
  check the S288C-homolog rule against the authors' gene table;
- the dev LMDB ``$DATA_ROOT/data/torchcell/caudal_pantranscriptome2024`` (pre-fix
  build), when present, for the per-isolate phenotype key counts it stores.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/``
``caudal2024_blank_systematic_name_ledger.json`` (the ledger as a build writes it) and
``caudal2024_blank_systematic_name_summary.json`` (every number the dendron note cites).

Run from the repo root:
    python experiments/036-dataset-fixes-before-kg-build/scripts/caudal2024_blank_systematic_name.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import pickle
from typing import Any

import lmdb
import pandas as pd
from dotenv import load_dotenv

from torchcell.datasets.scerevisiae import caudal2024 as m
from torchcell.sequence.genome.registry import PETER2018_1011, resolve

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"


def _dev_store_key_counts(store: str) -> dict[str, int]:
    """Strain -> number of expression_tpm keys in the dev LMDB (pre-fix build)."""
    env = lmdb.open(store, readonly=True, lock=False)
    out: dict[str, int] = {}
    with env.begin() as txn:
        for _, value in txn.cursor():
            record = pickle.loads(value)
            experiment = record["experiment"]
            strain = experiment["genotype"]["perturbations"][0]["strain_id"]
            out[strain] = len(experiment["phenotype"]["expression_tpm"])
    env.close()
    return out


def _homolog_rule_check(
    served_blank: pd.DataFrame,
    presence: pd.DataFrame,
    table_s2: pd.DataFrame,
    caudal_zip: str,
) -> dict[str, dict[str, Any]]:
    """For each S288C name served from a blank row: Table S2 agreement, Peter presence.

    Table S2 maps ``Annotation_Name`` -> ``systematic_name``; Peter presence is read, in
    the isolates served under that name, for the reference pangenome column of the name
    and for the accessory column the row's ``Annotation_Name`` numbers.
    """
    s2_map = dict(zip(table_s2["Annotation_Name"], table_s2["systematic_name"]))
    homolog_orfs = set(
        served_blank.loc[
            served_blank["gene_id"].astype(str).str.match(m._S288C_RE), "gene_id"
        ]
    )
    annotations = pd.read_csv(
        caudal_zip,
        encoding="latin-1",
        low_memory=False,
        usecols=["ORF", "Annotation_Name", "systematic_name"],
    )
    annotations = annotations[
        annotations["systematic_name"].isna() & annotations["ORF"].isin(homolog_orfs)
    ]
    ann_by_orf = annotations.drop_duplicates("ORF").set_index("ORF")["Annotation_Name"]
    out: dict[str, dict[str, Any]] = {}
    for orf, ann in ann_by_orf.items():
        strains = served_blank.loc[served_blank["gene_id"] == orf, "Strain"]
        number = str(ann).split("-", 1)[0]
        ref_cols = [
            c for c in presence.columns if m._orf_to_s288c(m._demangle_orf(c)) == orf
        ]
        acc_cols = [c for c in presence.columns if c.startswith(f"X{number}.")]
        out[str(orf)] = {
            "annotation_name": str(ann),
            "table_s2_systematic_name": str(s2_map.get(ann)),
            "agrees": s2_map.get(ann) == orf,
            "isolates": len(strains),
            "peter_reference_columns": ref_cols,
            "peter_reference_present": int(
                presence.loc[strains, ref_cols].to_numpy().sum()
            ),
            "peter_accessory_columns": acc_cols,
            "peter_accessory_present": int(
                presence.loc[strains, acc_cols].to_numpy().sum()
            ),
        }
    return out


def main() -> None:
    """Compute the ledger, the class profile and the served-record impact."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    key_dir = osp.join(data_root, "torchcell-library", m.CITATION_KEY)
    caudal_zip = osp.join(data_root, m.CAUDAL_ZIP_REL)
    os.makedirs(RESULTS, exist_ok=True)

    presence = pd.read_csv(
        resolve(PETER2018_1011, m.PRESENCE_NAME), sep="\t", index_col=0
    )
    presence.index = presence.index.astype(str)
    raw = m.read_caudal_table(caudal_zip)
    built = m.restrict_to_built_isolates(raw, set(presence.index))
    kept, ledger = m.resolve_gene_ids(built)
    with open(
        osp.join(RESULTS, "caudal2024_blank_systematic_name_ledger.json"), "w"
    ) as handle:
        handle.write(ledger.model_dump_json(indent=2))

    blank = built[built["systematic_name"].isna()]
    named = built[built["systematic_name"].notna()]
    served_blank = kept[kept["systematic_name"].isna()]
    class_profile: dict[str, dict[str, Any]] = {}
    for value, sub in blank.groupby("pan_absence"):
        class_profile[str(value)] = {
            "rows": len(sub),
            "distinct_orfs": int(sub["ORF"].nunique()),
            "rows_tpm_gt_0": int((sub["tpm"] > 0).sum()),
            "rows_tpm_gt_100": int((sub["tpm"] > 100).sum()),
            "isolates": int(sub["Strain"].nunique()),
        }
    excluded = raw[~raw["Strain"].isin(set(built["Strain"]))]
    excluded_blank = excluded[excluded["systematic_name"].isna()]

    # Served blank ids: isolates, median TPM, and the Peter presence of the pangenome
    # column a pangenome id names ('X' + demangled column == served id).
    peter_cols = {"X" + m._demangle_orf(c): c for c in presence.columns}
    served_profile: dict[str, dict[str, Any]] = {}
    for gene_id, sub in served_blank.groupby("gene_id"):
        entry: dict[str, Any] = {
            "isolates": int(sub["Strain"].nunique()),
            "median_tpm": float(sub["tpm"].median()),
            "rows_tpm_gt_0": int((sub["tpm"] > 0).sum()),
        }
        if str(gene_id) in peter_cols:
            col = peter_cols[str(gene_id)]
            entry["peter_column"] = col
            entry["peter_present_in_served_isolates"] = int(
                presence.loc[sub["Strain"], col].sum()
            )
        served_profile[str(gene_id)] = entry

    table_s2 = pd.read_excel(osp.join(key_dir, m.SI_TABLES_XLSX), sheet_name="Table S2")
    homolog_check = _homolog_rule_check(served_blank, presence, table_s2, caudal_zip)
    s2_names = set(table_s2["systematic_name"].astype(str))
    pangenome_ids = sorted(g for g in served_profile if not m._S288C_RE.match(g))

    # Per-isolate phenotype keys: new (named + served blank) vs the pre-fix dev store.
    new_keys = kept.groupby("Strain")["gene_id"].nunique()
    named_keys = named.groupby("Strain")["systematic_name"].nunique()
    gained = (new_keys - named_keys).astype(int)
    dev_lmdb = osp.join(
        data_root, "data/torchcell/caudal_pantranscriptome2024/processed/lmdb"
    )
    dev_counts = _dev_store_key_counts(dev_lmdb) if osp.isdir(dev_lmdb) else {}
    named_acc = named[~named["systematic_name"].astype(str).str.match(m._S288C_RE)]

    summary = {
        "inputs": {
            "caudal_zip": m.CAUDAL_ZIP_REL,
            "caudal_zip_sha256": m.CAUDAL_ZIP_SHA256,
            "peter_presence": m.PRESENCE_NAME,
            "table_s2": m.SI_TABLES_XLSX,
            "table_s2_sha256": m.SI_TABLES_XLSX_SHA256,
            "dev_lmdb": dev_lmdb if dev_counts else None,
        },
        "raw_rows": len(raw),
        "raw_strains": int(raw["Strain"].nunique()),
        "raw_blank_rows": int(raw["systematic_name"].isna().sum()),
        "excluded_strains": int(excluded["Strain"].nunique()),
        "excluded_blank_by_class": {
            str(k): int(v)
            for k, v in excluded_blank["pan_absence"].value_counts().items()
        },
        "built_isolates": int(built["Strain"].nunique()),
        "built_rows": len(built),
        "built_blank_rows": len(blank),
        "blank_tpm_fraction": float(blank["tpm"].sum() / built["tpm"].sum()),
        "per_strain_tpm_sum_median_all_rows": float(
            built.groupby("Strain")["tpm"].sum().median()
        ),
        "per_strain_tpm_sum_median_named_rows": float(
            named.groupby("Strain")["tpm"].sum().median()
        ),
        "ledger_counts": {str(k): v for k, v in ledger.counts.items()},
        "blank_class_profile": class_profile,
        "served_blank_rows": len(served_blank),
        "served_blank_ids": len(served_profile),
        "served_profile": served_profile,
        "homolog_rule_check": homolog_check,
        "pangenome_ids_in_table_s2": sorted(set(pangenome_ids) & s2_names),
        "isolates_gaining_keys": int((gained > 0).sum()),
        "keys_gained_max": int(gained.max()),
        "new_keys_per_isolate_median": float(new_keys.median()),
        "named_keys_per_isolate_median": float(named_keys.median()),
        "dev_store_records": len(dev_counts),
        "dev_store_keys_equal_named_keys": (
            all(dev_counts[s] == int(named_keys[s]) for s in dev_counts)
            if dev_counts
            else None
        ),
        "named_accessory_keys": int(named_acc["systematic_name"].nunique()),
        "named_accessory_keys_equal_X_plus_demangled_orf": bool(
            (
                named_acc["systematic_name"]
                == "X" + named_acc["ORF"].map(m._demangle_orf)
            ).all()
        ),
        "named_pan_absence_counts": ledger.named_pan_absence_counts,
    }
    with open(
        osp.join(RESULTS, "caudal2024_blank_systematic_name_summary.json"), "w"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ("served_profile", "homolog_rule_check")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
