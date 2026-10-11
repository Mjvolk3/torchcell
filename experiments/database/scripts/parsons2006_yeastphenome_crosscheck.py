#!/usr/bin/env python
# experiments/database/scripts/parsons2006_yeastphenome_crosscheck.py
# [[experiments.database.scripts.parsons2006_yeastphenome_crosscheck]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/parsons2006_yeastphenome_crosscheck.py
r"""L4 cross-check of the Parsons 2006 dev store against YeastPhenome's curation of it.

Issue #1000: yp-data v1.0 curates Parsons 2006 (PMID 16901791) from the same Table S7 the
``EnvChemgenParsons2006Dataset`` loader reads, so the two would duplicate if the
YeastPhenome ``SCREENS`` list were widened. This measures, read-only:

1. how many Parsons rows the yeastphenome DEV store holds (its ``raw/`` holds one
   ``<PMID>_*_valuez.txt`` per kept study, so a Parsons row needs a ``16901791_`` file);
2. the overlap of the Parsons store's records with yp-data's two curated Parsons files,
   ``parsons_boone_2006_value.txt`` (the curated values) and
   ``parsons_boone_2006_valuez.txt`` (the per-screen NPV the yeastphenome loader serves),
   joined on (source ORF, condition column), with Pearson and Spearman correlations,
   how many joined cells are equal or exactly negated (|sum| <= 1e-9; the "equal" count
   holds only the float-noise zeros, which are both), and the ORFs of the cells that
   are not negated.

yp-data's 82 columns are in Table S7's column order (checked by name for the
unambiguous ones below), so a column is joined by position. A record's source ORF is its
``constructed_orf.source_systematic_name`` when it has one, else its gene.

The two yp-data files are fetched by hand (they are a cross-check input, not loader input)
and verified against the pins here::

    for f in parsons_boone_2006_value.txt parsons_boone_2006_valuez.txt; do
      curl -s -o "$f" \
        "https://raw.githubusercontent.com/yeastphenome/yp-data/\
83e2917bf86955ec6ba66dc70ff2ed0fe24ecbe8/Datasets/16901791/$f"
    done

Run from the repo root::

    python experiments/database/scripts/parsons2006_yeastphenome_crosscheck.py \
        --yp-dir <dir holding the two files>

Writes ``experiments/database/results/parsons2006_yeastphenome_crosscheck.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import os.path as osp
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datasets.scerevisiae.parsons2006 import CONDITIONS
from torchcell.verification.runners import stream_records

YP_COMMIT = "83e2917bf86955ec6ba66dc70ff2ed0fe24ecbe8"
YP_URL = f"https://raw.githubusercontent.com/yeastphenome/yp-data/{YP_COMMIT}/Datasets/16901791"
YP_FILES = {
    "value": (
        "parsons_boone_2006_value.txt",
        "a269eaeb6c2ae11af702798fc439657138c49308f350795bf41e496383e8a197",
    ),
    "valuez": (
        "parsons_boone_2006_valuez.txt",
        "4448bb001c1564cfc3079b015229e197d5ec97b92b88d1ea273bb25f4fdc6514",
    ),
}
STORE_ROOT = "data/torchcell/env_chemgen_parsons2006"
YEASTPHENOME_RAW = "data/torchcell/yeastphenome/raw"
RESULTS = "experiments/database/results"
#: yp-data column index (1-based after ``orf``) -> a name fragment its header must hold,
#: for the columns whose spelling is the same in both releases.
ANCHORS = {
    1: "sulfometuron methyl",
    4: "benomyl",
    19: "tamoxifen",
    36: "cisplatin",
    64: "Extract 95-57",
    70: "Extract 00-132",
    82: "geldanamycin",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_yp(path: Path, sha256: str) -> pd.DataFrame:
    """One yp-data matrix (ORF index x 82 columns), its digest and its order checked."""
    got = _sha256(path)
    if got != sha256:
        raise RuntimeError(f"{path} sha256 {got} is not the pinned {sha256}")
    frame = pd.read_csv(path, sep="\t", index_col=0)
    if frame.shape[1] != len(CONDITIONS):
        raise RuntimeError(
            f"{path} has {frame.shape[1]} columns, not {len(CONDITIONS)}"
        )
    for index, fragment in ANCHORS.items():
        if fragment not in frame.columns[index - 1]:
            raise RuntimeError(
                f"{path} column {index} is {frame.columns[index - 1]!r}, expected to "
                f"name {fragment!r}"
            )
    return frame


def store_cells(data_root: str) -> dict[tuple[str, int], float]:
    """(source ORF, Table S7 column index) -> stored log2 ratio, for every record."""
    column_of = {
        resolved_compound(spec.lookup_label).name: index
        for index, spec in enumerate(CONDITIONS)
    }
    cells: dict[tuple[str, int], float] = {}
    for record in stream_records(osp.join(data_root, STORE_ROOT)):
        experiment = record["experiment"]
        (perturbation,) = experiment["genotype"]["perturbations"]
        constructed = perturbation.get("constructed_orf")
        orf = (
            constructed["source_systematic_name"]
            if constructed
            else perturbation["systematic_gene_name"]
        )
        (edit,) = experiment["environment"]["perturbations"]
        key = (orf, column_of[edit["compound"]["name"]])
        if key in cells:
            raise RuntimeError(f"two records for {key}")
        cells[key] = float(experiment["phenotype"]["environment_response"])
    return cells


def compare(
    cells: dict[tuple[str, int], float], frame: pd.DataFrame, *, by_orf: bool
) -> dict[str, Any]:
    """Join on (ORF, column); correlations and the equal / negated counts."""
    ours: list[float] = []
    theirs: list[float] = []
    orfs: list[str] = []
    yp_cells = 0
    for orf, row in frame.iterrows():
        for index, value in enumerate(row.tolist()):
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            yp_cells += 1
            stored = cells.get((str(orf), index))
            if stored is not None:
                ours.append(stored)
                theirs.append(float(value))
                orfs.append(str(orf))
    a = pd.Series(ours)
    b = pd.Series(theirs)
    tolerance = 1e-9
    negated = (a + b).abs() <= tolerance
    not_negated: dict[str, int] = {}
    for orf, flag in zip(orfs, negated.tolist(), strict=True):
        if not flag:
            not_negated[orf] = not_negated.get(orf, 0) + 1
    return {
        "yp_orfs": int(frame.shape[0]),
        "yp_cells": yp_cells,
        "joined_cells": len(ours),
        "store_cells_not_in_yp": len(cells) - len(ours),
        "pearson": float(a.corr(b)),
        "spearman": float(a.corr(b, method="spearman")),
        "equal": int(((a - b).abs() <= tolerance).sum()),
        "negated": int(negated.sum()),
        # only for the curated values, which are a sign flip of the release; the NPV
        # is a rescaling, so none of its cells is a negation
        "not_negated_cells_by_orf": dict(sorted(not_negated.items()))
        if by_orf
        else None,
    }


def main() -> None:
    """Measure the store's overlap with yp-data's Parsons curation; write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yp-dir", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]

    yp_raw = Path(data_root) / YEASTPHENOME_RAW
    parsons_files = sorted(p.name for p in yp_raw.glob("16901791_*"))
    cells = store_cells(data_root)
    results: dict[str, Any] = {
        "yeastphenome_dev_store_parsons_files": parsons_files,
        "yeastphenome_dev_store_raw_files": len(list(yp_raw.glob("*_valuez.txt"))),
        "store_records": len(cells),
        "yp_data": {"commit": YP_COMMIT, "url": YP_URL},
    }
    for key, (filename, sha256) in YP_FILES.items():
        frame = read_yp(args.yp_dir / filename, sha256)
        results[key] = {
            "file": filename,
            "sha256": sha256,
            **compare(cells, frame, by_orf=key == "value"),
        }
    os.makedirs(RESULTS, exist_ok=True)
    out = osp.join(RESULTS, "parsons2006_yeastphenome_crosscheck.json")
    with open(out, "w") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")
    print(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
