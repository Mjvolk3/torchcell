#!/usr/bin/env python
# experiments/database/scripts/yoshikawa2009_yeastphenome_crosscheck.py
# [[experiments.database.scripts.yoshikawa2009_yeastphenome_crosscheck]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/yoshikawa2009_yeastphenome_crosscheck.py
r"""L4 cross-check of the Yoshikawa 2009 dev store against YeastPhenome's curation of it.

Issue #1000: yp-data v1.0 curates Yoshikawa 2009 (PMID 19054128) from the same Table S1
the ``EnvChemgenYoshikawa2009Dataset`` loader reads, so the two would duplicate if the
YeastPhenome ``SCREENS`` list were widened. yp-data's ``code.ipynb`` (pinned below) reads
the two culture columns of each condition, averages them per ORF, and divides each
stress condition by the nonstress mean ("Normalize by the untreated sample"). This
measures, read-only:

1. how many Yoshikawa rows the yeastphenome DEV store holds (its ``raw/`` holds one
   ``<PMID>_*_valuez.txt`` per kept study, so a Yoshikawa row needs a ``19054128_`` file);
2. the store's records recomputed the same way (per gene, the mean of the two cultures
   in a stress condition over the mean of the two nonstress cultures) against
   ``yoshikawa_shimizu_2009_value.txt``, joined on (ORF, condition), with the largest
   absolute difference, how many joined cells agree to 1e-12, Pearson's r, and the ORFs
   of the cells that do not agree (the 5% ethanol column included, which the loader
   stores since #1045);
3. for every cell that does not agree, whether it does once ONE released strain that
   yp-data's index lacks is pooled into that gene, recomputed from the released Table
   S1 the same way; this is the measurement that says what yp-data's ORF translation
   does, read off yp-data itself (no genome lookup; the loader stores every strain under
   its released ORF);
4. the same join against ``yoshikawa_shimizu_2009_valuez.txt`` (the per-screen NPV the
   yeastphenome loader would serve), Pearson and Spearman only.

A gene with two released strains (RPL22B: ``YFL034C-A`` and ``YFL035C-B``) is averaged
over both, as yp-data's ``groupby(index).mean()`` after ORF translation does.

The three yp-data files are fetched by hand (a cross-check input, not loader input) and
verified against the pins here::

    for f in yoshikawa_shimizu_2009_value.txt yoshikawa_shimizu_2009_valuez.txt \
        code.ipynb; do
      curl -s -o "$f" \
        "https://raw.githubusercontent.com/yeastphenome/yp-data/\
83e2917bf86955ec6ba66dc70ff2ed0fe24ecbe8/Datasets/19054128/$f"
    done

Run from the repo root::

    python experiments/database/scripts/yoshikawa2009_yeastphenome_crosscheck.py \
        --yp-dir <dir holding the three files>

Writes ``experiments/database/results/yoshikawa2009_yeastphenome_crosscheck.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import os.path as osp
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from torchcell.datasets.scerevisiae import yoshikawa2009 as y
from torchcell.verification.runners import load_records

YP_COMMIT = y.YP_DATA_COMMIT
YP_URL = f"https://raw.githubusercontent.com/yeastphenome/yp-data/{YP_COMMIT}/Datasets/{y.PMID}"
YP_FILES = {
    "value": (
        "yoshikawa_shimizu_2009_value.txt",
        "c4efc04469a8f0b117f6cb7b751de18656ea0c0f4ff22267cb901d30419b61d5",
    ),
    "valuez": (
        "yoshikawa_shimizu_2009_valuez.txt",
        "def047da355eb37444b93246fcc17fe5ca65dcd91fb661ffd736e36135ef6c2d",
    ),
    "code": (
        "code.ipynb",
        "0038c9a15b22739732a5ef51f7c0afebef913bd7c674ca62fc4a21a65f9a30ca",
    ),
}
#: yp-data's column, in its file's order -> the released condition it holds.
YP_COLUMNS = {
    "hap alpha | growth (exponential growth rate) | EtOH (ethanol [8%]) | "
    "Yoshikawa K~Shimizu H, 2009": "8% ethanol",
    "hap alpha | growth (exponential growth rate) | sodium chloride [1 M] | "
    "Yoshikawa K~Shimizu H, 2009": "1M NaCl",
    "hap alpha | growth (exponential growth rate) | EtOH (ethanol [5%]) | "
    "Yoshikawa K~Shimizu H, 2009": "5% ethanol",
}
YEASTPHENOME_RAW = "data/torchcell/yeastphenome/raw"
RESULTS = "experiments/database/results"
#: The notebook lines the recomputation follows, checked verbatim in ``code.ipynb``.
NOTEBOOK_STEPS = (
    "data_cols = [c for c in original_data.columns if 'specific growth rate' in c]",
    "original_data = original_data.groupby(original_data.index).mean()",
    "dataset_ids = [100, 100, 523, 523, 4, 4, 5, 5]",
    "original_data = original_data.div(original_data[100], axis=0)",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pinned(directory: Path) -> dict[str, Path]:
    """The three yp-data files, each verified against its pin."""
    paths: dict[str, Path] = {}
    for key, (filename, sha256) in YP_FILES.items():
        path = directory / filename
        got = _sha256(path)
        if got != sha256:
            raise RuntimeError(f"{path} sha256 {got} is not the pinned {sha256}")
        paths[key] = path
    notebook = json.loads(paths["code"].read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    missing = [step for step in NOTEBOOK_STEPS if step not in source]
    if missing:
        raise RuntimeError(f"code.ipynb no longer holds {missing}")
    return paths


def store_ratios(data_root: str) -> dict[tuple[str, str], float]:
    """(gene, stress condition) -> mean of its cultures over the nonstress mean."""
    rates: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in load_records(osp.join(data_root, y.DATASET_ROOT_REL)):
        experiment = record["experiment"]
        (perturbation,) = experiment["genotype"]["perturbations"]
        gene = perturbation["systematic_gene_name"]
        label = y.condition_label(experiment["environment"])
        rates[(gene, label)].append(
            float(experiment["phenotype"]["environment_response"])
        )
    means = {key: sum(v) / len(v) for key, v in rates.items()}
    return {
        (gene, label): mean / means[(gene, "non-stress")]
        for (gene, label), mean in means.items()
        if label != "non-stress"
    }


def compare_values(
    ratios: dict[tuple[str, str], float], frame: pd.DataFrame
) -> dict[str, Any]:
    """Join yp-data's curated ratio with the store's recomputation, cell by cell."""
    ours: list[float] = []
    theirs: list[float] = []
    disagreeing: dict[str, float] = {}
    yp_cells = 0
    for orf, row in frame.iterrows():
        for column, label in YP_COLUMNS.items():
            value = row[column]
            if isinstance(value, float) and math.isnan(value):
                continue
            yp_cells += 1
            stored = ratios.get((str(orf), label))
            if stored is None:
                continue
            ours.append(stored)
            theirs.append(float(value))
            if abs(stored - float(value)) > 1e-12:
                disagreeing[f"{orf} / {label}"] = stored - float(value)
    a, b = pd.Series(ours), pd.Series(theirs)
    joined_genes = {gene for gene, _ in ratios} & {str(o) for o in frame.index}
    return {
        "yp_orfs": int(frame.shape[0]),
        "yp_cells": yp_cells,
        "joined_cells": len(ours),
        "joined_genes": len(joined_genes),
        "store_cells_not_in_yp": len(ratios) - len(ours),
        "store_genes_not_in_yp": sorted(
            {gene for gene, _ in ratios} - {str(o) for o in frame.index}
        ),
        "max_abs_difference": float((a - b).abs().max()),
        "agree_to_1e-12": int(((a - b).abs() <= 1e-12).sum()),
        "pearson": float(a.corr(b)),
        "disagreeing_cells": dict(sorted(disagreeing.items())),
    }


def pooled_ratio(
    rows: dict[str, y.ReleaseRow], strains: list[str], label: str
) -> float:
    """yp-data's ratio for one gene over ``strains``: mean of the cultures, over nonstress."""

    def mean(condition: str) -> float:
        values = [v for s in strains for v in rows[s].conditions[condition].rates]
        return sum(values) / len(values)

    return mean(label) / mean("non-stress")


def explain_disagreements(
    disagreeing: dict[str, float], frame: pd.DataFrame, rows: dict[str, y.ReleaseRow]
) -> dict[str, Any]:
    """How many disagreeing cells yp-data reproduces with one absent strain pooled in.

    The candidates are the released strains (stored ORFs) that yp-data's index lacks:
    yp-data translated them onto another gene. A cell is reproduced when pooling exactly
    one of them into the cell's gene gives yp-data's value to 1e-12.
    """
    absent = sorted(orf for orf in rows if orf not in {str(o) for o in frame.index})
    column_of = {label: column for column, label in YP_COLUMNS.items()}
    reproduced: dict[str, list[str]] = {}
    unexplained: list[str] = []
    for cell in disagreeing:
        gene, label = cell.split(" / ")
        value = float(frame[column_of[label]].astype(float)[gene])
        matches = [
            strain
            for strain in absent
            if abs(pooled_ratio(rows, [gene, strain], label) - value) <= 1e-12
        ]
        if matches:
            reproduced[cell] = [gene, *matches]
        else:
            unexplained.append(cell)
    return {
        "released_strains_absent_from_yp": absent,
        "reproduced_with_one_absent_strain_pooled": len(reproduced),
        "genes": sorted({cell.split(" / ")[0] for cell in reproduced}),
        "pooled_strains": {cell: names for cell, names in sorted(reproduced.items())},
        "unexplained": sorted(unexplained),
    }


def compare_npv(
    ratios: dict[tuple[str, str], float], frame: pd.DataFrame
) -> dict[str, Any]:
    """Correlation of the store's ratios with yp-data's per-screen NPV."""
    ours: list[float] = []
    theirs: list[float] = []
    for orf, row in frame.iterrows():
        for column, label in YP_COLUMNS.items():
            value = row[column]
            stored = ratios.get((str(orf), label))
            if stored is None or (isinstance(value, float) and math.isnan(value)):
                continue
            ours.append(stored)
            theirs.append(float(value))
    a, b = pd.Series(ours), pd.Series(theirs)
    return {
        "joined_cells": len(ours),
        "pearson": float(a.corr(b)),
        "spearman": float(a.corr(b, method="spearman")),
    }


def main() -> None:
    """Measure the store's overlap with yp-data's Yoshikawa curation; write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--yp-dir", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    paths = read_pinned(args.yp_dir)
    rows = {
        row.orf.upper(): row
        for row in y.read_release(y.raw_mirror_dir(data_root) / y.TABLE_S1.raw_rel)
    }
    yp_raw = Path(data_root) / YEASTPHENOME_RAW
    ratios = store_ratios(data_root)
    results: dict[str, Any] = {
        "yeastphenome_dev_store_yoshikawa_files": sorted(
            p.name for p in yp_raw.glob(f"{y.PMID}_*")
        ),
        "yeastphenome_dev_store_raw_files": len(list(yp_raw.glob("*_valuez.txt"))),
        "store_ratio_cells": len(ratios),
        "yp_data": {"commit": YP_COMMIT, "url": YP_URL},
    }
    for key in ("value", "valuez"):
        frame = pd.read_csv(paths[key], sep="\t", index_col=0)
        if list(frame.columns) != list(YP_COLUMNS):
            raise RuntimeError(f"{paths[key]} columns are {list(frame.columns)}")
        compared = (
            compare_values(ratios, frame)
            if key == "value"
            else compare_npv(ratios, frame)
        )
        if key == "value":
            compared["disagreements_explained"] = explain_disagreements(
                compared["disagreeing_cells"], frame, rows
            )
        results[key] = {
            "file": YP_FILES[key][0],
            "sha256": YP_FILES[key][1],
            **compared,
        }
    os.makedirs(RESULTS, exist_ok=True)
    out = osp.join(RESULTS, "yoshikawa2009_yeastphenome_crosscheck.json")
    with open(out, "w") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")
    print(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
