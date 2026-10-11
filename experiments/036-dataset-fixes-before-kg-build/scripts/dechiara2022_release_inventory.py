# experiments/036-dataset-fixes-before-kg-build/scripts/dechiara2022_release_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.dechiara2022_release_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/dechiara2022_release_inventory

"""Inventory the De Chiara 2022 release and measure which growth block names its rows.

Supplementary Table 3 (``si/si4.xlsx``, MOESM4) carries every per-isolate phenotype the
paper released: 5 sporulation traits, 8 chronological-life-span traits, and the 43
growth environments twice, once as the control-normalized log2 values the paper
analysed ("Doubling time", "Yield") and once as "absolute values" (hours, cells). This
script measures, from the pinned bytes:

1. The non-missing count of every phenotype column, against Supplementary Table 10's
   retained-isolate counts.
2. Whether the normalized block and the absolute block describe the same isolate on
   each row. Where both agree, ``log2(absolute) - normalized`` is a per-environment
   constant (the back-transform the authors describe in the peer-review file); each
   row's 86-value profile is matched against every row of the other block, and the
   offset between a row and its best-matching partner is recorded in ALPHABETICAL
   name order, which is the order the misalignment follows.
3. Which block is the correctly named one, adjudicated by genotype: in the rows where
   the two blocks disagree, only one of them can still carry the textbook sugar
   associations (SUC2 for sucrose, the MAL1-locus genes for maltose) that both carry in
   the rows where they agree. Gene presence comes from Peter 2018's pangenome matrix
   in the genomes tier.
4. The chronological-life-span columns against Peter's genome: the HPF1-like ORF
   (pangenome ``X560``) reproduces Extended Data Fig. 7b's counts and P values from the
   released values, which checks those columns' names independently of the growth
   blocks.

Writes ``results/dechiara2022_release_inventory.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/dechiara2022_release_inventory.py
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import mannwhitneyu

load_dotenv()

from torchcell.sequence.genome.registry import PETER2018_1011, resolve  # noqa: E402

DATA_ROOT = os.environ["DATA_ROOT"]
CITATION_KEY = "dechiaraDomesticationReprogrammedBudding2022"
SI4 = osp.join(DATA_ROOT, "torchcell-library", CITATION_KEY, "si", "si4.xlsx")
SI4_SHA256 = "b2eb2e28cdd2c3037ce14bb1eb8f58ee84609157dd592b3904595b24daf8c99e"
ST3 = "Supplementary Table 3"
ST10 = "Supplementary Table 10"
RESULTS = osp.join(
    os.environ["EXPERIMENT_ROOT"], "036-dataset-fixes-before-kg-build", "results"
)
OUT = osp.join(RESULTS, "dechiara2022_release_inventory.json")

LIFE_CYCLE_COLUMNS = [
    "Sporulation in water",
    "Asci at 24h",
    "Dyads at 24h",
    "Asci at 72h",
    "Dyads at 72h",
    "CLS in water (day 8)",
    "CLS in water (day 20)",
    "CLS in SDC (day 7)",
    "CLS in SDC (day 21)",
    "CLS in SDC (day 35)",
    "CLS in crSDC (day 7)",
    "CLS in crSDC (day 21)",
    "CLS in crSDC (day 35)",
]
# Supplementary Table 10 spells the thirteen life-cycle phenotypes differently.
ST10_LIFE_CYCLE_NAMES = [
    "Sporulation_in_water",
    "Asci_at_24h",
    "Dyads_at_24h",
    "Asci_at_72h",
    "Dyads_at_72h",
    "CLS_in_water_.day_8.",
    "CLS_in_water_.day_20.",
    "cls_d7_sdc",
    "cls_d21_sdc",
    "cls_d35_sdc",
    "cls_d7_cr",
    "cls_d21_cr",
    "cls_d35_cr",
]
# (pangenome column, environment): presence of the gene should speed growth there.
GENOTYPE_TESTS = [
    ("X4646.YIL162W", "Sucrose_2"),
    ("X4188.YGR289C", "Maltose_2"),
    ("X4188.YGR289C", "Maltose_8"),
    ("X4187.YGR288W", "Maltose_32"),
]
HPF1_LIKE_COLUMN = "X560.snap_masked.10934.AEN_3"
HPF1_LIKE_TESTS = [
    "CLS in crSDC (day 7)",
    "CLS in crSDC (day 21)",
    "CLS in crSDC (day 35)",
]
MIN_SHARED = 10


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_corr(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    """Pearson r over the positions both rows carry, NaN under MIN_SHARED of them."""
    mask = ~np.isnan(a) & ~np.isnan(b)
    if int(mask.sum()) < MIN_SHARED:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _centered(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Subtract each environment's median, so a row profile carries no env main effect."""
    centered: npt.NDArray[np.float64] = matrix - np.nanmedian(matrix, axis=0)
    return centered


def _block_alignment(st3: pd.DataFrame, envs: list[str]) -> dict[str, Any]:
    """Match every absolute-block row to its best normalized-block row."""
    norm = np.concatenate(
        [
            st3[[f"{e}_doublingtime" for e in envs]].to_numpy(float),
            st3[[f"{e}_yield" for e in envs]].to_numpy(float),
        ],
        axis=1,
    )
    absolute = np.concatenate(
        [
            np.log2(st3[[f"{e}_doublingtime_hours" for e in envs]].to_numpy(float)),
            np.log2(st3[[f"{e}_yield_cellscount" for e in envs]].to_numpy(float)),
        ],
        axis=1,
    )
    norm_c, abs_c = _centered(norm), _centered(absolute)
    names = st3.index.tolist()
    alpha = {name: i for i, name in enumerate(sorted(names))}
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        if np.isnan(absolute[i]).all():
            rows.append({"name": name, "status": "no_absolute_values"})
            continue
        scores = np.array([_row_corr(norm_c[j], abs_c[i]) for j in range(len(names))])
        if np.isnan(scores).all():
            rows.append({"name": name, "status": "too_few_values_to_match"})
            continue
        best = int(np.nanargmax(scores))
        rows.append(
            {
                "name": name,
                "status": "matched",
                "r_self": _row_corr(norm_c[i], abs_c[i]),
                "best_normalized_row": names[best],
                "r_best": float(scores[best]),
                "alphabetical_offset": alpha[names[best]] - alpha[name],
            }
        )
    matched = [r for r in rows if r["status"] == "matched"]
    offsets = pd.Series([r["alphabetical_offset"] for r in matched]).value_counts()
    self_aligned = sorted(r["name"] for r in matched if r["alphabetical_offset"] == 0)
    shifted = sorted(r["name"] for r in matched if r["alphabetical_offset"] != 0)
    return {
        "n_rows": len(rows),
        "n_rows_without_absolute_values": sum(
            1 for r in rows if r["status"] == "no_absolute_values"
        ),
        "n_rows_too_few_values_to_match": sum(
            1 for r in rows if r["status"] == "too_few_values_to_match"
        ),
        "n_matched": len(matched),
        "alphabetical_offset_counts": {str(k): int(v) for k, v in offsets.items()},
        "n_self_aligned": len(self_aligned),
        "n_misaligned": len(shifted),
        "misaligned_names": shifted,
        "r_best_quantiles_1_5_50": [
            float(q)
            for q in np.nanquantile([r["r_best"] for r in matched], [0.01, 0.05, 0.5])
        ],
        "rows": matched,
    }


def _constant_offset(
    st3: pd.DataFrame, envs: list[str], aligned: list[str]
) -> list[dict[str, Any]]:
    """Per environment, log2(absolute) - normalized over the self-aligned rows."""
    out: list[dict[str, Any]] = []
    for env in envs:
        dt = (
            np.log2(st3.loc[aligned, f"{env}_doublingtime_hours"])
            - st3.loc[aligned, f"{env}_doublingtime"]
        )
        y = (
            np.log2(st3.loc[aligned, f"{env}_yield_cellscount"])
            - st3.loc[aligned, f"{env}_yield"]
        )
        out.append(
            {
                "environment": env,
                "doubling_time_offset_mean": float(dt.mean()),
                "doubling_time_offset_sd": float(dt.std()),
                "yield_offset_mean": float(y.mean()),
                "yield_offset_sd": float(y.std()),
            }
        )
    return out


def _genotype_adjudication(
    st3: pd.DataFrame, presence: pd.DataFrame, aligned: list[str], misaligned: list[str]
) -> list[dict[str, Any]]:
    """Two-sided Mann-Whitney of each block's value by gene presence, per row subset."""
    out: list[dict[str, Any]] = []
    blocks = {
        "normalized_doubling_time": "{env}_doublingtime",
        "absolute_doubling_time": "{env}_doublingtime_hours",
        "normalized_yield": "{env}_yield",
        "absolute_yield": "{env}_yield_cellscount",
    }
    for gene, env in GENOTYPE_TESTS:
        for subset_name, subset in (("aligned", aligned), ("misaligned", misaligned)):
            carried = presence.loc[subset, gene]
            for block, pattern in blocks.items():
                values = st3.loc[subset, pattern.format(env=env)]
                present = values[(carried == 1).to_numpy()].dropna()
                absent = values[(carried == 0).to_numpy()].dropna()
                out.append(
                    {
                        "gene_column": gene,
                        "environment": env,
                        "rows": subset_name,
                        "block": block,
                        "n_present": len(present),
                        "n_absent": len(absent),
                        "median_present": float(present.median()),
                        "median_absent": float(absent.median()),
                        "p_two_sided": float(mannwhitneyu(present, absent).pvalue),
                    }
                )
    return out


def _hpf1_like(st3: pd.DataFrame, presence: pd.DataFrame) -> list[dict[str, Any]]:
    """Extended Data Fig. 7b recomputed from the released CLS columns."""
    carried = presence.loc[st3.index, HPF1_LIKE_COLUMN]
    out: list[dict[str, Any]] = []
    for column in HPF1_LIKE_TESTS:
        values = st3[column]
        present = values[(carried == 1).to_numpy()].dropna()
        absent = values[(carried == 0).to_numpy()].dropna()
        out.append(
            {
                "column": column,
                "n_with_hpf1_like": len(present),
                "n_without": len(absent),
                "p_two_sided": float(mannwhitneyu(present, absent).pvalue),
            }
        )
    return out


def main() -> None:
    """Measure the release and write the inventory JSON."""
    observed = _sha256(SI4)
    if observed != SI4_SHA256:
        raise RuntimeError(f"{SI4}: sha256 {observed} != pinned {SI4_SHA256}")
    sheets = pd.read_excel(SI4, sheet_name=None, header=None)
    st3 = pd.read_excel(SI4, sheet_name=ST3, header=2, na_values=["NA"]).set_index(
        "Standard name"
    )
    st10_table = pd.read_excel(SI4, sheet_name=ST10, header=2).dropna(how="all")
    st10 = st10_table.set_index("Phenotype")["Total n. of isolates"]
    envs = [
        c[: -len("_doublingtime")] for c in st3.columns if c.endswith("_doublingtime")
    ]

    counts: dict[str, dict[str, int]] = {}
    for column, st10_name in zip(LIFE_CYCLE_COLUMNS, ST10_LIFE_CYCLE_NAMES):
        counts[column] = {
            "non_missing": int(st3[column].notna().sum()),
            "supplementary_table_10": int(st10[st10_name]),
        }
    for env in envs:
        for column, st10_name in (
            (f"{env}_doublingtime", f"{env}_doublingtime_hours"),
            (f"{env}_doublingtime_hours", f"{env}_doublingtime_hours"),
            (f"{env}_yield", f"{env}_yield_cellscount"),
            (f"{env}_yield_cellscount", f"{env}_yield_cellscount"),
        ):
            counts[column] = {
                "non_missing": int(st3[column].notna().sum()),
                "supplementary_table_10": int(st10[st10_name]),
            }

    presence = pd.read_csv(
        resolve(PETER2018_1011, "genesMatrix_PresenceAbsence.tab.gz"),
        sep="\t",
        index_col=0,
    )
    alignment = _block_alignment(st3, envs)
    misaligned = alignment["misaligned_names"]
    aligned = sorted(
        r["name"] for r in alignment["rows"] if r["alphabetical_offset"] == 0
    )
    sace = [n for n in st3.index if n.startswith("SACE_")]
    absolute_cols = [f"{e}_doublingtime_hours" for e in envs]
    result: dict[str, Any] = {
        "source": {"path": SI4, "sha256": observed},
        "sheets": {name: list(frame.shape) for name, frame in sheets.items()},
        "supplementary_table_3": {
            "n_isolates": int(len(st3)),
            "n_columns": int(st3.shape[1] + 1),
            "n_growth_environments": len(envs),
            "growth_environments": envs,
            "names_unique": bool(st3.index.is_unique),
            "n_names_in_peter_presence_matrix": int(
                st3.index.isin(presence.index).sum()
            ),
            "n_sace_isolates": len(sace),
            "n_sace_isolates_with_any_absolute_growth_value": int(
                st3.loc[sace, absolute_cols].notna().any(axis=1).sum()
            ),
            "clade_class_counts": {
                str(k): int(v) for k, v in st3["Clade_class"].value_counts().items()
            },
            "ploidy_counts": {
                str(k): int(v) for k, v in st3["Ploidy"].value_counts().items()
            },
            "max_value": {c: float(st3[c].max()) for c in LIFE_CYCLE_COLUMNS},
        },
        "column_counts": counts,
        "n_columns_where_count_differs_from_table_10": sum(
            1
            for v in counts.values()
            if v["non_missing"] != v["supplementary_table_10"]
        ),
        "block_alignment": {k: v for k, v in alignment.items() if k != "rows"},
        "absolute_minus_normalized_offset_on_aligned_rows": _constant_offset(
            st3, envs, aligned
        ),
        "genotype_adjudication": _genotype_adjudication(
            st3, presence, aligned, misaligned
        ),
        "hpf1_like_cls": _hpf1_like(st3, presence),
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"wrote {OUT}")
    print(
        f"aligned {alignment['n_self_aligned']}, misaligned {alignment['n_misaligned']}, "
        f"no absolute values {alignment['n_rows_without_absolute_values']}"
    )


if __name__ == "__main__":
    main()
