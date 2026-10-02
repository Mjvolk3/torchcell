# experiments/036-dataset-fixes-before-kg-build/scripts/kuzmin2020_screen_source.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.kuzmin2020_screen_source]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/kuzmin2020_screen_source
"""Measure the Kuzmin 2020 main-vs-pilot screen overlap and what the screen tag fixes (#602).

Reads the released Tables S1, S3 and S5 from the dev raw directory
(``$DATA_ROOT/data/torchcell/dmf_kuzmin2020/raw``), runs every Kuzmin 2020 loader's
``preprocess_raw`` (no LMDB is built), then builds the experiment object of every
preprocessed row and hashes its ``model_dump`` exactly as the knowledge-graph adapter
does (``torchcell/adapters/cell_adapter.py``, ``_experiment_node``). Reports, per loader:

- raw (query strain, array strain) keys shared by Table S1 and Table S3, how many of
  those differ in the stored value, and how many carry two array allele names;
- preprocessed rows per ``screen_id``;
- distinct array strains and how many of them map to more than one
  ``Array allele name`` after preprocessing (0 is the fix);
- distinct experiment ids over all rows, and the number of (genotype, environment)
  groups that hold more than one record, split by whether those records differ in
  ``screen_id`` (every group with two records must now be a main + pilot pair);
- the same counts with ``screen_id`` removed from the dump (the old behavior), i.e.
  how many records would collapse onto one id without the screen tag.

With ``--dev-lmdb`` it also counts the records of the dev-tree LMDB stores and their
``screen_id`` values, for the post-rebuild check (run it after the slurm rebuild).

Writes ``experiments/036-dataset-fixes-before-kg-build/results/kuzmin2020_screen_source.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/kuzmin2020_screen_source.py
    python experiments/036-dataset-fixes-before-kg-build/scripts/kuzmin2020_screen_source.py --dev-lmdb
"""

import argparse
import hashlib
import json
import os
import os.path as osp
from collections import Counter
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from torchcell.datasets.scerevisiae import kuzmin2020 as k

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"
TABLES = ["aaz5667-Table-S1.xlsx", "aaz5667-Table-S3.xlsx", "aaz5667-Table-S5.xlsx"]
KEY = ["Query strain ID", "Array strain ID"]


def raw_overlap(df_s1: pd.DataFrame, df_s3: pd.DataFrame) -> dict[str, Any]:
    """Keys shared by S1 and S3 per combined-mutant type, as released."""
    out: dict[str, Any] = {}
    for kind in ("digenic", "trigenic"):
        a = df_s1[df_s1["Combined mutant type"] == kind]
        b = df_s3[df_s3["Combined mutant type"] == kind]
        m = a.merge(b, on=KEY, suffixes=("_s1", "_s3"))
        fit = "Double/triple mutant fitness"
        score = "Adjusted genetic interaction score (epsilon or tau)"
        out[kind] = {
            "rows_s1": len(a),
            "rows_s3": len(b),
            "rows_total": len(a) + len(b),
            "distinct_keys": len(pd.concat([a[KEY], b[KEY]]).drop_duplicates()),
            "keys_in_both": len(m),
            "fitness_differs": int((m[f"{fit}_s1"] != m[f"{fit}_s3"]).sum()),
            "fitness_and_sd_identical": int(
                (
                    (m[f"{fit}_s1"] == m[f"{fit}_s3"])
                    & (
                        m[f"{fit} standard deviation_s1"]
                        == m[f"{fit} standard deviation_s3"]
                    )
                ).sum()
            ),
            "score_and_p_identical": int(
                (
                    (m[f"{score}_s1"] == m[f"{score}_s3"])
                    & (m["P-value_s1"] == m["P-value_s3"])
                ).sum()
            ),
            "array_allele_name_differs": int(
                (m["Array allele name_s1"] != m["Array allele name_s3"]).sum()
            ),
        }
    a1 = df_s1.drop_duplicates("Array strain ID").set_index("Array strain ID")
    a3 = df_s3.drop_duplicates("Array strain ID").set_index("Array strain ID")
    shared = a1.index.intersection(a3.index)
    renamed = shared[
        a1.loc[shared, "Array allele name"] != a3.loc[shared, "Array allele name"]
    ]
    out["array_strains"] = {
        "s1": int(df_s1["Array strain ID"].nunique()),
        "s3": int(df_s3["Array strain ID"].nunique()),
        "shared": len(shared),
        "shared_with_two_names": len(renamed),
        "shared_with_two_names_case_only": int(
            (
                a1.loc[renamed, "Array allele name"].str.lower()
                == a3.loc[renamed, "Array allele name"].str.lower()
            ).sum()
        ),
        "examples": {
            s: [a1.loc[s, "Array allele name"], a3.loc[s, "Array allele name"]]
            for s in ["YAL001C_tsa508", "YBL041W_tsa1064", "YMR166C_dma3761"]
        },
    }
    return out


def experiment_ids(cls: Any, df: pd.DataFrame, extra: dict[str, Any]) -> dict[str, Any]:
    """Hash every row's experiment dump as the adapter does, with and without screen_id."""
    ids: list[str] = []
    ids_no_screen: list[str] = []
    group: dict[str, list[str | None]] = {}
    for _, row in df.iterrows():
        experiment, _, _ = cls.create_experiment(cls.__name__, row, **extra)
        dump = experiment.model_dump()
        ids.append(hashlib.sha256(json.dumps(dump).encode("utf-8")).hexdigest())
        screen = dump["phenotype"].pop("screen_id")
        ids_no_screen.append(
            hashlib.sha256(json.dumps(dump).encode("utf-8")).hexdigest()
        )
        ge = json.dumps([dump["genotype"], dump["environment"]])
        group.setdefault(ge, []).append(screen)
    multi = [v for v in group.values() if len(v) > 1]
    return {
        "records": len(ids),
        "distinct_experiment_ids": len(set(ids)),
        "distinct_experiment_ids_without_screen_id": len(set(ids_no_screen)),
        "genotype_environment_groups": len(group),
        "groups_with_more_than_one_record": len(multi),
        "group_sizes": dict(Counter(len(v) for v in multi)),
        "multi_groups_main_plus_pilot": sum(
            Counter(v) == Counter([k.SCREEN_ID_MAIN, k.SCREEN_ID_PILOT]) for v in multi
        ),
    }


def preprocessed(
    df_s1: pd.DataFrame, df_s3: pd.DataFrame, df_s5: pd.DataFrame
) -> dict[str, Any]:
    """Run each loader's preprocess and measure screen tags, names and experiment ids."""
    out: dict[str, Any] = {}
    loaders: list[tuple[Any, tuple[pd.DataFrame, ...]]] = [
        (k.SmfKuzmin2020Dataset, (df_s5,)),
        (k.DmfKuzmin2020Dataset, (df_s1, df_s3, df_s5)),
        (k.TmfKuzmin2020Dataset, (df_s1, df_s3)),
        (k.DmiKuzmin2020Dataset, (df_s1, df_s3)),
        (k.TmiKuzmin2020Dataset, (df_s1, df_s3)),
    ]
    for cls, tables in loaders:
        ds = cls.__new__(cls)
        df = ds.preprocess_raw(*(t.copy() for t in tables))
        entry: dict[str, Any] = {
            "rows": len(df),
            "rows_per_screen_id": {
                str(s): int(n) for s, n in df["screen_id"].value_counts().items()
            },
        }
        if "Array strain ID" in df.columns:
            arrays = df.dropna(subset=["Array strain ID"])
            names = arrays.groupby("Array strain ID")["Array allele name"].nunique()
            entry["array_strains"] = len(names)
            entry["array_strains_with_more_than_one_name"] = int((names > 1).sum())
        extra = (
            {"phenotype_reference_std": ds.phenotype_reference_std}
            if cls is k.TmfKuzmin2020Dataset
            else {}
        )
        print(f"{cls.__name__}: hashing {len(df)} experiments ...", flush=True)
        entry["experiment_ids"] = experiment_ids(cls, df, extra)
        out[cls.__name__] = entry
    return out


def dev_lmdb(data_root: str) -> dict[str, Any]:
    """Record counts and screen_id values of the five dev-tree stores (post-rebuild)."""
    out: dict[str, Any] = {}
    for cls, slug in [
        (k.SmfKuzmin2020Dataset, "smf_kuzmin2020"),
        (k.DmfKuzmin2020Dataset, "dmf_kuzmin2020"),
        (k.TmfKuzmin2020Dataset, "tmf_kuzmin2020"),
        (k.DmiKuzmin2020Dataset, "dmi_kuzmin2020"),
        (k.TmiKuzmin2020Dataset, "tmi_kuzmin2020"),
    ]:
        ds = cls(root=osp.join(data_root, "data/torchcell", slug))
        screens = Counter(
            str(ds[i]["experiment"]["phenotype"].get("screen_id"))
            for i in range(len(ds))
        )
        out[cls.__name__] = {"records": len(ds), "screen_id": dict(screens)}
        ds.close_lmdb()
    return out


def main() -> None:
    """Measure, then write the results JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-lmdb", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    raw_dir = osp.join(data_root, "data/torchcell/dmf_kuzmin2020/raw")
    df_s1, df_s3, df_s5 = (
        pd.read_excel(osp.join(raw_dir, name), skiprows=1) for name in TABLES
    )
    result: dict[str, Any] = {
        "script": "experiments/036-dataset-fixes-before-kg-build/scripts/"
        "kuzmin2020_screen_source.py",
        "raw_dir": raw_dir,
        "raw_overlap": raw_overlap(df_s1, df_s3),
        "preprocessed": preprocessed(df_s1, df_s3, df_s5),
    }
    if args.dev_lmdb:
        result["dev_lmdb"] = dev_lmdb(data_root)
    os.makedirs(RESULTS, exist_ok=True)
    path = osp.join(RESULTS, "kuzmin2020_screen_source.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
