# experiments/036-dataset-fixes-before-kg-build/scripts/wetmore2015_release_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.wetmore2015_release_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/wetmore2015_release_inventory
"""Measure Wetmore 2015's own E. coli release off the deposited bytes, and against Price.

Owner policy of 2026-10-10: a served compendium is a processing reference and a sanity
check, not the source of record. Wetmore et al. 2015 (mBio, doi:10.1128/mBio.00306-15)
released its own per-value gene fitness (statistics version 1.0.0), so the row is
loaded from that release (``RbTnseqWetmore2015Dataset``) and cross-checked against the
Price 2018 compendium's copy. This script reads the sha256-pinned files directly with
pandas and openpyxl, NOT through the loader's readers, so its counts are an independent
re-derivation of the loader's:

1. **Files.** Every file the row reads, re-hashed against its pin: Data Set S1 in the
   literature mirror and the release tables in the raw mirror.
2. **Samples.** Data Set S1's ``Expts_Keio``: condition samples and time-zero rows;
   ``Expt_Quality_Keio``'s ``u`` per sample, and whether the paper's five rules
   (gMed >= 50, mad12 <= 0.5, cor12 >= 0.1, |gccor| <= 0.2, |adjcor| <= 0.25) reproduce
   it; the failed samples with the rule each trips.
3. **The authors' withdrawal.** The condition samples whose ``Condition_1`` is sucrose
   or D-mannitol, which the compendium page's 2021 note withdraws.
4. **Genes and values.** Genes of the gene-fitness table, sample columns of the
   successful-sample table and of the all-sample table, and the t identity
   ``t = fit / sqrt(0.1^2 + max(se_obs, se_naive)^2)`` over every value.
5. **Against the compendium release.** For the samples both releases carry, the
   per-sample Pearson r of Wetmore 1.0.0 against Price 1.0.3 on the shared genes, and
   the largest single-gene difference, read off both ``fit_logratios`` tables.
6. **Against the compendium store.** Which shared samples the served Price 2018 dev store
   holds, and what this release carries that the store does not: the samples Price's
   own ledger drops (its ``preprocess/samples.json``), the sample its 1.0.3 re-analysis
   fails, and the per-value t the compendium's records do not store.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/
wetmore2015_release_inventory.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/wetmore2015_release_inventory.py
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
from typing import Any

import lmdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from torchcell.datasets.ecoli import wetmore2015 as w
from torchcell.verification.runners import stream_records

OUT = osp.join(
    "experiments",
    "036-dataset-fixes-before-kg-build",
    "results",
    "wetmore2015_release_inventory.json",
)
#: The paper's five quality rules (SUCCESS_RULES), restated here so the verdict is
#: re-derived rather than read back from the loader.
RULES = {
    "g_med_min": lambda q: q["gMed"] >= 50.0,
    "mad12_max": lambda q: q["mad12"] <= 0.5,
    "cor12_min": lambda q: q["cor12"] >= 0.1,
    "abs_gccor_max": lambda q: abs(q["gccor"]) <= 0.2,
    "abs_adjcor_max": lambda q: abs(q["adjcor"]) <= 0.25,
}
WITHDRAWN_CONDITIONS = ("Sucrose", "D-Mannitol")
SIGMA = 0.1


def sha256(path: str) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_columns(table: pd.DataFrame) -> dict[str, str]:
    """Sample name (the header's first token) to its column."""
    return {
        str(c).split(" ")[0]: str(c) for c in table.columns if str(c).startswith("set")
    }


def files(data_root: str) -> list[dict[str, Any]]:
    """Every file the row reads, with its pin and its re-hashed bytes."""
    rows = [
        {
            "mirror": "torchcell-library",
            "path": w.DATA_SET_S1,
            "pin": w.DATA_SET_S1_SHA256,
            "sha256": sha256(
                osp.join(data_root, w.LIBRARY_DIR_REL, w.CITATION_KEY, w.DATA_SET_S1)
            ),
        }
    ]
    for raw in w.RAW_FILES.values():
        rows.append(
            {
                "mirror": "torchcell-raw",
                "path": raw.relpath,
                "pin": raw.sha256,
                "sha256": sha256(osp.join(data_root, w.RAW_DIR_REL, raw.relpath)),
                "retrieved_at": raw.retrieved_at,
                "url": raw.url,
            }
        )
    for row in rows:
        row["intact"] = row["sha256"] == row["pin"]
    return rows


def samples(s1: str) -> dict[str, Any]:
    """Data Set S1's condition samples, their verdicts and the withdrawn ones."""
    experiments = pd.read_excel(s1, sheet_name=w.DATA_SET_S1_EXPERIMENTS_SHEET)
    quality = pd.read_excel(s1, sheet_name=w.DATA_SET_S1_QUALITY_SHEET).set_index(
        "name"
    )
    condition = experiments.loc[experiments["Group"] != "Time0"]
    verdicts: dict[str, bool] = {}
    failed: dict[str, list[str]] = {}
    for name in condition["name"]:
        q = quality.loc[name]
        trips = [rule for rule, ok in RULES.items() if not ok(q)]
        verdicts[str(name)] = bool(q["u"])
        if trips:
            failed[str(name)] = trips
    reproduced = all(verdicts[n] == (n not in failed) for n in verdicts)
    withdrawn = condition.loc[condition["Condition_1"].isin(WITHDRAWN_CONDITIONS)]
    return {
        "expts_rows": int(len(experiments)),
        "time_zero_rows": int((experiments["Group"] == "Time0").sum()),
        "condition_samples": int(len(condition)),
        "successful": sorted(n for n, ok in verdicts.items() if ok),
        "failed": failed,
        "paper_rules_reproduce_u": reproduced,
        "by_medium": {
            str(k): int(v) for k, v in condition["Media"].value_counts().items()
        },
        "successful_by_medium": {
            str(k): int(v)
            for k, v in condition.loc[condition["name"].map(verdicts)]["Media"]
            .value_counts()
            .items()
        },
        "withdrawn": {
            str(r["name"]): str(r["Condition_1"]) for _, r in withdrawn.iterrows()
        },
        "withdrawn_successful": sorted(
            str(n) for n in withdrawn["name"] if verdicts[str(n)]
        ),
    }


def values(raw: str) -> dict[str, Any]:
    """The release's gene and value counts and the t identity over every value."""

    def read(rel: str) -> pd.DataFrame:
        return pd.read_csv(osp.join(raw, rel), sep="\t")

    good, every = read(w.RELEASE_FITNESS), read(w.RELEASE_FITNESS_ALL)
    t, obs, naive = read(w.RELEASE_T), read(w.RELEASE_SE_OBS), read(w.RELEASE_SE_NAIVE)
    columns = sample_columns(every)
    names = sorted(columns)
    stack = {
        label: table[[sample_columns(table)[s] for s in names]].to_numpy(float)
        for label, table in (("fit", every), ("t", t), ("obs", obs), ("naive", naive))
    }
    se = np.maximum(stack["obs"], stack["naive"])
    deviation = np.abs(stack["t"] - stack["fit"] / np.sqrt(SIGMA**2 + se**2))
    good_columns = sample_columns(good)
    equal_on_good = max(
        float(np.max(np.abs(good[good_columns[s]] - every[columns[s]])))
        for s in good_columns
    )
    return {
        "genes": int(good["locusId"].nunique()),
        "b_number_sysnames": int(
            good["sysName"].astype(str).str.fullmatch(r"b\d{4}").sum()
        ),
        "good_table_samples": len(good_columns),
        "all_table_samples": len(columns),
        "all_table_time_zero_columns": sorted(
            s for s, c in columns.items() if c.endswith("Time0")
        ),
        "good_values_equal_all_table": equal_on_good,
        "t_identity_max_abs_deviation": float(deviation.max()),
        "naive_se_larger_fraction_successful": float(
            np.mean(
                stack["naive"][:, [names.index(s) for s in good_columns]]
                > stack["obs"][:, [names.index(s) for s in good_columns]]
            )
        ),
    }


def against_release(raw: str) -> dict[str, Any]:
    """Wetmore 1.0.0 (every sample) against Price 1.0.3 on the shared genes."""
    wet = pd.read_csv(osp.join(raw, w.RELEASE_FITNESS_ALL), sep="\t")
    price = pd.read_csv(osp.join(raw, w.SUPERSET_FITNESS), sep="\t")
    wc, pc = sample_columns(wet), sample_columns(price)
    shared_genes = sorted(set(wet["locusId"]) & set(price["locusId"]))
    left = wet.set_index("locusId").loc[shared_genes]
    right = price.set_index("locusId").loc[shared_genes]
    per_sample = {}
    for name in sorted(set(wc) & set(pc)):
        x = left[wc[name]].to_numpy(float)
        y = right[pc[name]].to_numpy(float)
        k = int(np.argmax(np.abs(x - y)))
        per_sample[name] = {
            "pearson_r": float(np.corrcoef(x, y)[0, 1]),
            "max_abs_difference": float(abs(x[k] - y[k])),
            "max_difference_sys_name": str(left["sysName"].iloc[k]),
        }
    rs = [v["pearson_r"] for v in per_sample.values()]
    return {
        "wetmore_genes": int(wet["locusId"].nunique()),
        "price_genes": int(price["locusId"].nunique()),
        "shared_genes": len(shared_genes),
        "shared_samples": len(per_sample),
        "min_pearson_r": min(rs),
        "median_pearson_r": float(np.median(rs)),
        "replicate_bar": w.COMPENDIUM_MIN_PEARSON_R,
        "below_bar": sorted(n for n, v in per_sample.items() if v["pearson_r"] < 0.87),
        "wetmore_samples_price_release_lacks": sorted(
            n for n in wc if n not in pc and not wc[n].endswith("Time0")
        ),
        "per_sample": per_sample,
    }


def against_store(data_root: str, successful: list[str]) -> dict[str, Any]:
    """What the served Price 2018 dev store holds of this paper's samples."""
    root = osp.join(data_root, w.SUPERSET_DATASET_ROOT_REL)
    with open(osp.join(root, "preprocess", "samples.json")) as handle:
        ledger = {row["name"]: row for row in json.load(handle)}
    with open(osp.join(root, "preprocess", "build_manifest.json")) as handle:
        manifest = json.load(handle)
    env = lmdb.open(osp.join(root, "processed", "lmdb"), readonly=True, lock=False)
    entries = env.stat()["entries"]
    env.close()
    phenotype = next(iter(stream_records(root)))["experiment"]["phenotype"]
    kept = {n for n, row in ledger.items() if row["drop_rule"] is None}
    carried = sorted(n for n, row in ledger.items() if row["source"] == w.CITATION_KEY)
    return {
        "store": w.SUPERSET_DATASET_ROOT_REL,
        "records": entries,
        "built_from_commit": manifest["torchcell_commit"],
        "built_from_dirty_tree": manifest["torchcell_dirty"],
        "table_s5_samples": len(ledger),
        "carried_from_wetmore": len(carried),
        "carried_and_kept": sorted(n for n in carried if n in kept),
        "carried_and_dropped": {
            n: ledger[n]["drop_rule"] for n in carried if n not in kept
        },
        "wetmore_successful_absent_from_store": sorted(
            n for n in successful if n not in kept
        ),
        "store_phenotype_fields": sorted(phenotype),
        "store_stores_a_t_statistic": "environment_response_t_statistic" in phenotype,
    }


def main() -> None:
    """Measure and write the inventory."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    raw = osp.join(data_root, w.RAW_DIR_REL)
    s1 = osp.join(data_root, w.LIBRARY_DIR_REL, w.CITATION_KEY, w.DATA_SET_S1)
    inventory_files = files(data_root)
    if not all(f["intact"] for f in inventory_files):
        raise RuntimeError(
            f"pins broken: {[f['path'] for f in inventory_files if not f['intact']]}"
        )
    sample_counts = samples(s1)
    out = {
        "citation_key": w.CITATION_KEY,
        "doi": w.PAPER_DOI,
        "files": inventory_files,
        "samples": sample_counts,
        "values": values(raw),
        "against_compendium_release": against_release(raw),
        "against_compendium_store": against_store(
            data_root, sample_counts["successful"]
        ),
    }
    os.makedirs(osp.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(out, handle, indent=2)
        handle.write("\n")
    print(OUT)


if __name__ == "__main__":
    main()
