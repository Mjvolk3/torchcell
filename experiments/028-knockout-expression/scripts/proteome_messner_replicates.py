# experiments/028-knockout-expression/scripts/proteome_messner_replicates.py
# [[experiments.028-knockout-expression.scripts.proteome_messner_replicates]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-knockout-expression/scripts/proteome_messner_replicates
"""The replicate ceiling of the Messner 2023 knockout proteome.

Messner measured one proteome per strain, so the panel has no replicate ceiling in the
sense Kemmeren's dye-swap pairs give, but 145 ORFs are present as two or three strains
of different origin, and the HIS3 control was measured 388 times. Two numbers follow:

  per protein   the Pearson across the duplicated ORFs between the two strains' log2
                ratios (over the HIS3 reference), the same quantity that bounds
                pearson_per_feature for a model scored per protein across strains;
                its mean sqrt is the ceiling in the sense of the expression strand
  per deletion  the Pearson over proteins between the two strains of one ORF, the
                strain-level replication

Run from the repo root:
    python experiments/028-knockout-expression/scripts/proteome_messner_replicates.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from cross_study_ko_expression import _load_lmdb  # noqa: E402
from proteome_expression_covariation import LMDB_PROTEOME  # noqa: E402

from torchcell.utils.paths import experiment_results_dir  # noqa: E402

MIN_PAIRS_PER_PROTEIN = 50


def main() -> None:
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    results_dir = experiment_results_dir("028-knockout-expression", __file__)
    records = _load_lmdb(osp.join(data_root, LMDB_PROTEOME))
    ref = records[0]["reference"]["phenotype_reference"]["protein_abundance"]
    ref_log = pd.Series({g: np.log2(v) for g, v in ref.items() if v > 0})
    per_orf: dict[str, list[pd.Series]] = {}
    for rec in records:
        perts = rec["experiment"]["genotype"]["perturbations"]
        if len(perts) != 1:
            continue
        ab = rec["experiment"]["phenotype"]["protein_abundance"]
        s = pd.Series({g: np.log2(v) for g, v in ab.items() if v > 0}) - ref_log
        per_orf.setdefault(perts[0]["systematic_gene_name"], []).append(s.dropna())
    dup = {o: v for o, v in per_orf.items() if len(v) >= 2}
    a = pd.DataFrame({o: v[0] for o, v in dup.items()}).T
    b = pd.DataFrame({o: v[1] for o, v in dup.items()}).T
    genes = a.columns.intersection(b.columns)
    a, b = a[genes], b[genes]
    both = a.notna() & b.notna()
    per_protein = {}
    for g in genes:
        m = both[g]
        if m.sum() >= MIN_PAIRS_PER_PROTEIN:
            per_protein[g] = float(np.corrcoef(a.loc[m, g], b.loc[m, g])[0, 1])
    pp = pd.Series(per_protein)
    per_strain = {}
    for o in a.index:
        m = both.loc[o]
        if m.sum() >= 200:
            per_strain[o] = float(np.corrcoef(a.loc[o, m], b.loc[o, m])[0, 1])
    ps = pd.Series(per_strain)
    out: dict[str, Any] = {
        "generated_by": "experiments/028-knockout-expression/scripts/proteome_messner_replicates.py",
        "n_duplicated_orfs": len(dup),
        "n_proteins_scored": int(len(pp)),
        "per_protein_median_r": float(pp.median()),
        "per_protein_mean_r": float(pp.mean()),
        "per_protein_frac_above_0.3": float((pp > 0.3).mean()),
        "ceiling_mean_sqrt_r": float(np.sqrt(pp.clip(lower=0)).mean()),
        "per_strain_median_r": float(ps.median()),
        "per_strain_iqr": [float(ps.quantile(0.25)), float(ps.quantile(0.75))],
        "per_strain_n": int(len(ps)),
    }
    print(json.dumps(out, indent=1))
    with open(osp.join(results_dir, "proteome_messner_replicates.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
