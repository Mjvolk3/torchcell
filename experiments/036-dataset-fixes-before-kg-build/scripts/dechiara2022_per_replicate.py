# experiments/036-dataset-fixes-before-kg-build/scripts/dechiara2022_per_replicate.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.dechiara2022_per_replicate]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/dechiara2022_per_replicate
"""Measure the per-replicate validation data of De Chiara 2022 that no store reads.

Tracking issue #1178 (manuscript #1176). Reads the sha256-pinned literature mirror of
``dechiaraDomesticationReprogrammedBudding2022`` and Peter 2018's Supplementary Table 1,
never writes a store. Measures:

1. Per Source Data sheet that releases one value per ``replicat`` (``si14.xlsx`` d,
   ``si9.xlsx`` e, ``si11.xlsx`` d and f): rows, numeric values, ``na`` cells, strains,
   and per strain its background, genotype, time points and replicate labels.
2. That ``si9.xlsx`` sheet e is a subset of ``si14.xlsx`` sheet d, value for value, so
   the distinct sporulation replicate values are those of si14 d alone.
3. The Extended Data Fig. 3d caption percentages, re-computed from the replicate values:
   the mean over every numeric ``replicat`` of every strain of one (background,
   genotype) at one time point. This is how the authors combined the two engineered
   strains and their technical triplicates.
4. Each strain against Supplementary Table 11 (engineered strains), and each background
   against Peter 2018 Table S1, by isolate name and by standardized name, with every
   disagreement listed.
5. That each replicate-design quote is a verbatim substring of its pinned file.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/dechiara2022_per_replicate.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/dechiara2022_per_replicate.py
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import warnings
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from torchcell.datasets.scerevisiae.dechiara2022 import CITATION_KEY

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"

#: Literature-mirror files read, with the sha256 its manifest.json records.
PINS: dict[str, str] = {
    "paper.md": "ca68d65691eaff4d6c1b6bf474a4ecef57a5bc41467b51ac993d816cfa033ea1",
    "si/si2.pdftotext.txt": (
        "ce323f8d2b0c2e59f4fef4551794599b350eb372ef247652aea814fef5075757"
    ),
    "si/si4.xlsx": "b2eb2e28cdd2c3037ce14bb1eb8f58ee84609157dd592b3904595b24daf8c99e",
    "si/si9.xlsx": "438d13504d07f0a2b2e026143efca599f357e61b3c1b46f83dd7b4e90f19278d",
    "si/si11.xlsx": "59b309c2fbefe0508897d1997f1e4270deb4353e4138c701df1bc4571da1232e",
    "si/si14.xlsx": "9ce3991473b16543a59f35491b3bdd234b1ef97f86aacbbdf2e4716e4e2744f4",
}
PETER_KEY = "peterGenomeEvolution10112018"
PETER_TABLES = (
    "si/si3.xls",
    "b11de13b50bcf91bb2a40cdbd0f2f35372bdef6fc9a018d7538cf5e5eea7f273",
)

#: Replicate-design statements, each a verbatim substring of the named pinned file.
QUOTES: list[tuple[str, str, str]] = [
    (
        "paper.md",
        "Methods, 'Validation of IME1 H78R and C-181T mutations'",
        "For each mutant, two independent biological replicates run in technical "
        "triplicates were considered.",
    ),
    (
        "paper.md",
        "Methods, 'Validation of IME1 H78R and C-181T mutations'",
        "sporulation efficiency was monitored using DAPI staining",
    ),
    (
        "paper.md",
        "Methods, 'Validation of IME1 H78R and C-181T mutations'",
        "Each cell displaying at least two distinct nuclei was counted as sporulating.",
    ),
    (
        "paper.md",
        "Methods, 'Chronological life span assay'",
        "Isogenic cultures were independently grown overnight in liquid SDC",
    ),
    (
        "paper.md",
        "Methods, 'Viability in all environments'",
        "Experiments were run in a single replicate and data were normalized to four "
        "internal controls distributed in each plate.",
    ),
    (
        "si/si2.pdftotext.txt",
        "Reporting Summary, 'Replication'",
        "Automatic growth curve were conducted in 4 replicates, median values were "
        "taken for each isolates.",
    ),
]

#: Extended Data Fig. 3d caption: (wild-type strains, edited strains, hours, caption
#: wild-type %, caption edited %), copied from the caption in paper.md. The YPS128
#: comparison is OS983 against the C-181T strains; YGL3969 (the ho/ura3 marker
#: derivative) is the control of the YGL3970 H78R panel, which the caption does not cite.
CAPTION_PAIRS: list[tuple[tuple[str, ...], tuple[str, ...], float, float, float]] = [
    (("OS983",), ("YGL4056", "YGL4059"), 9.0, 86.0, 4.0),
    (("OS983",), ("YGL4056", "YGL4059"), 24.0, 98.0, 58.0),
    (("OS1517",), ("YGL4074", "YGL4075"), 24.0, 1.7, 22.5),
    (("OS1517",), ("YGL4074", "YGL4075"), 168.0, 35.0, 59.0),
    (("YGL3975",), ("YGL3971",), 72.0, 37.0, 2.0),
    (("YGL3975",), ("YGL3971",), 168.0, 43.0, 4.0),
    (("YGL598",), ("YGL4076", "YGL4077"), 72.0, 0.3, 2.0),
    (("YGL598",), ("YGL4076", "YGL4077"), 240.0, 3.5, 7.7),
]

_SPO_COLUMNS = ["strain", "timepoint(h)", "replicat", "asci", "background", "genotype"]


def sha256(path: str) -> str:
    """The sha256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned(library: str, rel: str, expected: str) -> str:
    """The path of a mirror file after its sha256 is checked."""
    path = osp.join(library, rel)
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"{path}: sha256 {observed} != pinned {expected}")
    return path


def sporulation_blocks(path: str, sheet: str) -> pd.DataFrame:
    """The two side-by-side blocks of a sporulation sheet, stacked, blanks dropped."""
    frame = pd.read_excel(path, sheet_name=sheet)
    blocks = []
    for suffix in ("", ".1"):
        block = frame[[c + suffix for c in _SPO_COLUMNS]].dropna(
            subset=["strain" + suffix]
        )
        block.columns = _SPO_COLUMNS
        block = block.assign(block="left" if suffix == "" else "right")
        blocks.append(block)
    out = pd.concat(blocks, ignore_index=True)
    out["value"] = pd.to_numeric(out["asci"], errors="coerce")
    return out


def sheet_summary(frame: pd.DataFrame, time_col: str, value_col: str) -> dict[str, Any]:
    """Counts of one per-replicate sheet and the design of each strain in it."""
    numeric = pd.to_numeric(frame[value_col], errors="coerce")
    strains: dict[str, Any] = {}
    for strain, rows in frame.groupby("strain", sort=True):
        strains[str(strain)] = {
            "background": sorted(rows["background"].astype(str).unique().tolist()),
            "genotype": sorted(rows["genotype"].astype(str).unique().tolist()),
            "time_points": sorted(float(t) for t in rows[time_col].unique()),
            "replicat_labels": sorted(int(r) for r in rows["replicat"].unique()),
            "rows": len(rows),
            "na_cells": int(
                pd.to_numeric(rows[value_col], errors="coerce").isna().sum()
            ),
        }
    return {
        "rows": len(frame),
        "numeric_values": int(numeric.notna().sum()),
        "na_cells": int(numeric.isna().sum()),
        "strains": len(strains),
        "per_strain": strains,
    }


def caption_check(si14: pd.DataFrame) -> list[dict[str, Any]]:
    """Each Extended Data Fig. 3d caption percentage against the pooled replicate mean."""
    out = []
    for wt, edit, hours, cap_wt, cap_edit in CAPTION_PAIRS:
        row: dict[str, Any] = {"hours": hours}
        for label, strains, caption in (("wt", wt, cap_wt), ("edit", edit, cap_edit)):
            sel = si14[
                si14["strain"].isin(strains)
                & (si14["timepoint(h)"] == hours)
                & si14["value"].notna()
            ]
            row[label] = {
                "strains": list(strains),
                "background": sorted(sel["background"].unique().tolist()),
                "genotype": sorted(sel["genotype"].unique().tolist()),
                "n_values": len(sel),
                "pooled_mean": round(float(sel["value"].mean()), 3),
                "per_strain_mean": {
                    str(s): round(float(v), 3)
                    for s, v in sel.groupby("strain")["value"].mean().items()
                },
                "caption_percent": caption,
            }
        out.append(row)
    return out


def _cell(value: Any) -> str | None:
    """A workbook cell as text, a blank cell as None."""
    return None if pd.isna(value) else str(value)


def strain_identity(
    library: str, peter_tables: str, strains: set[str], backgrounds: set[str]
) -> dict[str, Any]:
    """Source-data strains against Supplementary Table 11; backgrounds against Peter."""
    st11 = pd.read_excel(
        osp.join(library, "si/si4.xlsx"),
        sheet_name="Supplementary Table 11",
        header=2,
        dtype=str,
    )
    st11 = st11.dropna(subset=["Strain name"]).set_index("Strain name")
    table_s1 = pd.read_excel(peter_tables, sheet_name="Table S1", header=3, dtype=str)
    by_isolate = dict(
        zip(table_s1["Isolate name"].str.strip(), table_s1["Standardized name"])
    )
    by_standard = dict(
        zip(table_s1["Standardized name"].str.strip(), table_s1["Isolate name"])
    )
    st11_rows = {
        s: (
            {
                "background": _cell(st11.loc[s, "Background"]),
                "ploidy": _cell(st11.loc[s, "Ploidy"]),
                "genotype": _cell(st11.loc[s, "Genotype"]),
                "additional_engineering": _cell(
                    st11.loc[s, "Additional genetic engineering"]
                ),
            }
            if s in st11.index
            else None
        )
        for s in sorted(strains)
    }
    peter: dict[str, Any] = {}
    for label in sorted(backgrounds):
        isolate, _, standard = label.partition("_")
        peter[label] = {
            "isolate_part": isolate,
            "standard_part": standard or None,
            "peter_standard_for_isolate": by_isolate.get(isolate),
            "peter_isolate_for_standard": by_standard.get(standard)
            if standard
            else None,
        }
    st11_names = {
        str(name).split(" (")[0]: str(row_name)
        for row_name, name in st11["Background"].items()
        if str(row_name) == str(row_name).upper() and "YGL" not in str(row_name)
    }
    return {
        "source_strains_in_supplementary_table_11": st11_rows,
        "backgrounds_against_peter_table_s1": peter,
        "supplementary_table_11_wild_type_rows": st11_names,
    }


def main() -> None:
    """Measure, check, and write the results JSON."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    library = osp.join(data_root, "torchcell-library", CITATION_KEY)
    paths = {rel: pinned(library, rel, sha) for rel, sha in PINS.items()}
    peter_tables = pinned(
        osp.join(data_root, "torchcell-library", PETER_KEY), *PETER_TABLES
    )

    quotes = []
    for rel, where, quote in QUOTES:
        with open(paths[rel], encoding="utf-8") as handle:
            text = handle.read()
        if quote not in text:
            raise RuntimeError(f"quote not verbatim in {rel}: {quote!r}")
        quotes.append(
            {"file": rel, "sha256": PINS[rel], "where": where, "quote": quote}
        )

    si14 = sporulation_blocks(paths["si/si14.xlsx"], "d")
    si9 = sporulation_blocks(paths["si/si9.xlsx"], "e")
    key = ["strain", "timepoint(h)", "replicat"]
    merged = si9.merge(si14, on=key, suffixes=("_si9", "_si14"), how="left")
    found = int(merged["background_si14"].notna().sum())
    same_value = int(
        (
            (merged["value_si9"] == merged["value_si14"])
            | (merged["value_si9"].isna() & merged["value_si14"].isna())
        ).sum()
    )
    if found != len(si9) or same_value != len(si9):
        raise RuntimeError(
            "si9 sheet e is not a value-for-value subset of si14 sheet d"
        )

    si11 = {
        sheet: pd.read_excel(paths["si/si11.xlsx"], sheet_name=sheet) for sheet in "df"
    }
    sheets = {
        "si14.xlsx d (Extended Data Fig. 3d)": sheet_summary(
            si14, "timepoint(h)", "asci"
        ),
        "si9.xlsx e (Fig. 4e)": sheet_summary(si9, "timepoint(h)", "asci"),
        "si11.xlsx d (Fig. 6f, VAR560 deletion)": sheet_summary(
            si11["d"], "timepoint(d)", "viability"
        ),
        "si11.xlsx f (Fig. 6d, WHI2 rescue)": sheet_summary(
            si11["f"], "timepoint(d)", "viability"
        ),
    }
    for sheet, frame in si11.items():
        sizes = frame.groupby(["strain", "environment", "timepoint(d)"]).size()
        if set(sizes.unique()) != {3}:
            raise RuntimeError(f"si11 {sheet}: a (strain, environment, day) is not x3")
    distinct = {
        "sporulation_values_si14_d": sheets["si14.xlsx d (Extended Data Fig. 3d)"][
            "numeric_values"
        ],
        "cls_values_si11_d": sheets["si11.xlsx d (Fig. 6f, VAR560 deletion)"][
            "numeric_values"
        ],
        "cls_values_si11_f": sheets["si11.xlsx f (Fig. 6d, WHI2 rescue)"][
            "numeric_values"
        ],
    }
    distinct["total_numeric"] = sum(distinct.values())
    distinct["total_rows"] = (
        sheets["si14.xlsx d (Extended Data Fig. 3d)"]["rows"]
        + sheets["si11.xlsx d (Fig. 6f, VAR560 deletion)"]["rows"]
        + sheets["si11.xlsx f (Fig. 6d, WHI2 rescue)"]["rows"]
    )

    strains = set(si14["strain"]) | set(si11["d"]["strain"]) | set(si11["f"]["strain"])
    backgrounds = (
        set(si14["background"])
        | set(si11["d"]["background"])
        | set(si11["f"]["background"])
    )
    captions = caption_check(si14)
    identity = strain_identity(library, peter_tables, strains, backgrounds)
    result: dict[str, Any] = {
        "citation_key": CITATION_KEY,
        "pins": PINS,
        "peter_table_s1": {
            "file": f"{PETER_KEY}/{PETER_TABLES[0]}",
            "sha256": PETER_TABLES[1],
        },
        "replicate_design_quotes": quotes,
        "si9_e_is_subset_of_si14_d": {
            "rows": len(si9),
            "found": found,
            "same_value": same_value,
        },
        "sheets": sheets,
        "distinct_replicate_values": distinct,
        "extended_data_fig_3d_caption": captions,
        "identity": identity,
    }
    os.makedirs(RESULTS, exist_ok=True)
    out = osp.join(RESULTS, "dechiara2022_per_replicate.json")
    with open(out, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps(distinct, indent=2))
    for row in captions:
        print(
            f"{row['hours']:>5} h  wt {row['wt']['strains']} {row['wt']['pooled_mean']}"
            f" (caption {row['wt']['caption_percent']})  edit {row['edit']['strains']}"
            f" {row['edit']['pooled_mean']} (caption {row['edit']['caption_percent']})"
        )
    print(json.dumps(identity["backgrounds_against_peter_table_s1"], indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
