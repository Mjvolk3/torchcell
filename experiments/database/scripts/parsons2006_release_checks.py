#!/usr/bin/env python
# experiments/database/scripts/parsons2006_release_checks.py
# [[experiments.database.scripts.parsons2006_release_checks]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/parsons2006_release_checks.py
r"""Measurements on the Parsons 2006 release that the loader's docstring and note cite.

Reads the raw mirror (``$DATA_ROOT/torchcell-raw/parsonsExploringModeofActionBioactive2006``,
Tables S4, S7) through the loader's own parsers, plus two literature-mirror tables the
loader does not consume (S5 non-functional tags, S9 multidrug-resistant strains) and the
SGD essentiality gene set, all read-only:

- the sign: among cells whose Table S4 -log10 p exceeds 5, the fraction with a positive
  Table S7 ratio (the release says positive = depleted under the drug);
- the lone-tag halving: the SD of the Table S7 values of strains with one Table S5
  non-functional tag against the SD of every other strain (halving a lone tag would put
  the ratio below 1/sqrt(2) for any tag correlation, since a two-tag mean has SD
  s*sqrt((1+rho)/2) and a halved tag s/2);
- the Table S9 and SGD-essential overlaps with Table S7's ORFs.

Run from the repo root::

    python experiments/database/scripts/parsons2006_release_checks.py

Writes ``experiments/database/results/parsons2006_release_checks.json``.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from torchcell.datasets.scerevisiae import parsons2006 as p

RESULTS = "experiments/database/results"
ESSENTIAL = "data/torchcell/gene_essentiality_sgd/preprocess/gene_set.json"


def main() -> None:
    """Measure the release; write the JSON."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    raw = Path(data_root) / p.RAW_DIR_REL / "data"
    library = p.library_dir(data_root)
    ratios = p.parse_ratio_matrix(p.read_sheet(raw / p.TABLE_S7.filename))
    p_values = p.parse_p_value_matrix(p.read_sheet(raw / p.TABLE_S4.filename))

    strong = positive = 0
    for orf, row in zip(ratios.orfs, ratios.values, strict=True):
        for ratio, neg in zip(row, p_values.neg_log10[orf], strict=True):
            if ratio is not None and neg is not None and neg > 5:
                strong += 1
                positive += ratio > 0

    tags = p.read_sheet(library / "si/si6.xls")
    lone = {
        str(tag).strip().rsplit("_", 1)[0]
        for tag in tags.iloc[p.HEADER_ROW + 1 :, 0].tolist()
        if isinstance(tag, str)
    } & set(ratios.orfs)
    lone_values = [
        v
        for orf, row in zip(ratios.orfs, ratios.values, strict=True)
        if orf in lone
        for v in row
        if v is not None
    ]
    other_values = [
        v
        for orf, row in zip(ratios.orfs, ratios.values, strict=True)
        if orf not in lone
        for v in row
        if v is not None
    ]
    sd_lone = float(np.std(lone_values))
    sd_other = float(np.std(other_values))

    resistant = set(p.parse_orf_list(p.read_sheet(library / "si/si10.xls"), _S9))
    essential = set(json.loads((Path(data_root) / ESSENTIAL).read_text()))
    results: dict[str, Any] = {
        "table_s7_strains": len(ratios.orfs),
        "table_s7_ratios": ratios.n_numeric,
        "table_s7_blanks": ratios.n_blank,
        "sign": {
            "cells_neg_log10_p_above_5": strong,
            "positive": positive,
            "positive_fraction": positive / strong,
        },
        "lone_tag": {
            "strains": len(lone),
            "sd_lone_tag_strains": sd_lone,
            "sd_other_strains": sd_other,
            "sd_ratio": sd_lone / sd_other,
            "halving_upper_bound": 1 / np.sqrt(2),
        },
        "table_s9_strains": len(resistant),
        "table_s9_in_table_s7": len(resistant & set(ratios.orfs)),
        "sgd_essential_in_table_s7": sorted(essential & set(ratios.orfs)),
    }
    os.makedirs(RESULTS, exist_ok=True)
    out = osp.join(RESULTS, "parsons2006_release_checks.json")
    with open(out, "w") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")
    print(json.dumps(results, indent=2))
    print(f"wrote {out}")


_S9 = p.RawTable(
    table="S9",
    filename="mmc10.xls",
    library_rel="si/si10.xls",
    sha256="c98eb022013242e9bb5608c80009cf080d3f5c4e807217a442c774e2383af9ec",
    bytes=33280,
    title="Supplementary Table 9.  Multi-drug resistant deletion strains.",
)

if __name__ == "__main__":
    main()
