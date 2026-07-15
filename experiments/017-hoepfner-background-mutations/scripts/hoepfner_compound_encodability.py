# experiments/017-hoepfner-background-mutations/scripts/hoepfner_compound_encodability.py
# [[experiments.017-hoepfner-background-mutations.scripts.hoepfner_compound_encodability]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/017-hoepfner-background-mutations/scripts/hoepfner_compound_encodability
"""How many Hoepfner 2014 compounds are proprietary (no released structure)?

A second data-quality axis for `EnvChemgenHoepfner2014Dataset` (besides the HIP
background mutations): most of the profiled compounds are PROPRIETARY Novartis `CMBxxx`
entries with NO released SMILES, so they cannot be featurised by any molecular encoder
(SMILES/graph/fingerprint). Only the reference + novel-MoA compounds carry a structure
(Table S1). This quantifies the encodable fraction from the same sha256-pinned raw files
the LMDB is built from.

Paper provenance (paper.md, sha256-pinned mirror):
  - "In addition to 1641 proprietary compounds (named CMBxxx), we included 135 reference
    compounds with a previously reported molecular mechanism of action (Table S1)." (line 110)
  - "Compound structures for reference compounds and the novel compounds presented in this
    study are shown in Table S1." (line 58)
  - "nearly 1800 biologically active compounds" (line 24)
"""

import json
import os
import os.path as osp

from dotenv import load_dotenv

from torchcell.datasets.scerevisiae.hoepfner2014 import _COL_RE, _load_compound_meta

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RAW = osp.join(DATA_ROOT, "data/torchcell/env_chemgen_hoepfner2014/raw")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "017-hoepfner-background-mutations", "results")

PAPER_QUOTE = (
    "In addition to 1641 proprietary compounds (named CMBxxx), we included 135 reference "
    "compounds with a previously reported molecular mechanism of action (Table S1). "
    "[Hoepfner 2014 paper.md line 110]"
)


def main() -> None:
    meta = _load_compound_meta(osp.join(RAW, "Table_S1.xls"))
    has_smiles = lambda c: bool((meta.get(c) or {}).get("smiles"))  # noqa: E731
    has_name = lambda c: bool((meta.get(c) or {}).get("common_name"))  # noqa: E731

    cmbs: set[str] = set()
    cols = {"HIP": 0, "HOP": 0}
    cols_no_smiles = {"HIP": 0, "HOP": 0}
    for fname, assay in (("HIP_scores.txt", "HIP"), ("HOP_scores.txt", "HOP")):
        with open(osp.join(RAW, fname)) as handle:
            header = handle.readline().rstrip("\n").split("\t")
        for raw in header:
            m = _COL_RE.match(raw.strip().strip('"'))
            if m is not None and m.group("z") is None:
                cmb = m.group("cmb")
                cmbs.add(cmb)
                cols[assay] += 1
                if not has_smiles(cmb):
                    cols_no_smiles[assay] += 1

    n = len(cmbs)
    n_smiles = sum(1 for c in cmbs if has_smiles(c))
    n_name = sum(1 for c in cmbs if has_name(c))
    n_encodable_or_named = sum(1 for c in cmbs if has_smiles(c) or has_name(c))
    tot_cols = cols["HIP"] + cols["HOP"]
    tot_no_smiles = cols_no_smiles["HIP"] + cols_no_smiles["HOP"]

    out = {
        "paper_quote": PAPER_QUOTE,
        "paper_stated": {"proprietary": 1641, "reference": 135, "total": 1776,
                         "headline_text": "nearly 1800 biologically active compounds"},
        "measured_from_deposited_data": {
            "unique_compounds": n,
            "with_released_smiles_encodable": n_smiles,
            "with_common_name": n_name,
            "not_encodable_no_smiles": n - n_smiles,
            "proprietary_no_name_no_smiles": n - n_encodable_or_named,
            "pct_encodable": round(n_smiles / n * 100, 1),
            "pct_not_encodable": round((n - n_smiles) / n * 100, 1),
        },
        "experiment_columns": {
            "HIP": cols["HIP"], "HOP": cols["HOP"], "total": tot_cols,
            "using_non_encodable_compound": tot_no_smiles,
            "pct_using_non_encodable": round(tot_no_smiles / tot_cols * 100, 1),
        },
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(osp.join(RESULTS_DIR, "compound_encodability.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"unique compounds: {n}")
    print(f"  encodable (released SMILES): {n_smiles} ({n_smiles/n*100:.1f}%)")
    print(f"  NOT encodable (no SMILES):   {n - n_smiles} ({(n-n_smiles)/n*100:.1f}%)")
    print(f"  fully proprietary (no name/SMILES): {n - n_encodable_or_named}")
    print(f"experiment columns using a non-encodable compound: "
          f"{tot_no_smiles}/{tot_cols} ({tot_no_smiles/tot_cols*100:.1f}%)")
    print(f"wrote {osp.join(RESULTS_DIR, 'compound_encodability.json')}")


if __name__ == "__main__":
    main()
