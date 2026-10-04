# experiments/019-simb-multimodal/scripts/expression_morphology_covariation.py
# [[experiments.019-simb-multimodal.scripts.expression_morphology_covariation]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/expression_morphology_covariation
"""Does the Kemmeren 2014 knockout transcriptome carry information about CalMorph
morphology of the same deletions, and the reverse?

The third side of the single-deletion triangle asked for Figure 3 on 2026-10-04. The
proteome-against-expression side is in review/2026-09-27-joint-review/01_data_ceilings.md
(ridge from the observed modality, 0.226 and 0.218 per feature) and the
proteome-against-morphology side in proteome_morphology_covariation.py (0.169 per
CalMorph feature, 0.280 on the features that move under deletion, 0.081 per protein in
the reverse direction). This script runs the same four reads with expression in the
proteome's place, with the same estimator, folds, penalties, seed and reliability
threshold, imported from that script so the three sides stay comparable:

  1. observed expression -> each CalMorph feature, split by replicate reliability;
  2. observed morphology -> each expression gene;
  3. response magnitude, Spearman across strains between the L2 norms of the per-feature
     z-scored responses;
  4. read 1 with the first principal component projected out of both sides.

Inputs: the Kemmeren 2014 LMDB (data/torchcell/microarray_kemmeren2014, log2 of deletion
over wild type, single deletions, duplicate records of one ORF averaged), the CalMorph
tables mt4718data.tsv and wt122data.tsv, and results/morphology_feature_ceiling.csv.

Writes results/expression_morphology_covariation.json.

    python experiments/019-simb-multimodal/scripts/expression_morphology_covariation.py \
        --morph-dir <dir holding mt4718data.tsv and wt122data.tsv>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
from typing import Any

import numpy as np
import pandas as pd
from proteome_morphology_covariation import (
    CEILING_CSV,
    DATA_ROOT,
    MORPH_DIR,
    RELIABLE,
    RESULTS,
    SEED,
    _col_pearson,
    _load_lmdb,
    _project_out_pc1,
    _ridge_oof,
    _zscore,
)
from scipy.stats import spearmanr

LMDB_EXPRESSION = osp.join(
    DATA_ROOT, "data/torchcell/microarray_kemmeren2014/processed/lmdb"
)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _kemmeren(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Deletion x gene log2 ratio over wild type, single deletions, one row per ORF."""
    rows: dict[str, list[pd.Series]] = {}
    for rec in records:
        perts = rec["experiment"]["genotype"]["perturbations"]
        if len(perts) != 1:
            continue
        orf = perts[0]["systematic_gene_name"]
        rows.setdefault(orf, []).append(
            pd.Series(rec["experiment"]["phenotype"]["expression_log2_ratio"])
        )
    return pd.DataFrame(
        {o: pd.concat(v, axis=1).mean(axis=1) for o, v in rows.items()}
    ).T


def _morphology(morph_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    mt = pd.read_csv(osp.join(morph_dir, "mt4718data.tsv"), sep="\t", index_col=0)
    ceiling = pd.read_csv(CEILING_CSV, index_col=0)
    mt.index = [str(i).upper() for i in mt.index]
    feats = [f for f in ceiling.index if f in mt.columns]
    return mt[feats].astype(float), ceiling.loc[feats]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--morph-dir", default=MORPH_DIR)
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    E = _kemmeren(_load_lmdb(LMDB_EXPRESSION))
    mt, ceiling = _morphology(args.morph_dir)
    strains = sorted(set(E.index) & set(mt.index))
    E = E.loc[strains]
    M = mt.loc[strains]
    # genes measured in every shared strain and not constant across them
    E = E.loc[:, E.notna().all() & (E.std(ddof=0) > 0)]
    Ez = _zscore(E)
    Mz = _zscore(M)
    reliability = ceiling["reliability"].reindex(M.columns)
    moving = reliability >= RELIABLE

    X, Y = Ez.to_numpy(), Mz.to_numpy()
    # 1. expression -> morphology
    pred, lam1 = _ridge_oof(X, Y, rng)
    r_em = pd.Series(_col_pearson(pred, Y), index=M.columns)
    perm = rng.permutation(X.shape[0])
    pred_null, _ = _ridge_oof(X[perm], Y, rng)
    r_em_null = pd.Series(_col_pearson(pred_null, Y), index=M.columns)
    # 4. first component removed from both sides
    Xg, Yg = _project_out_pc1(X), _project_out_pc1(Y)
    pred_g, _ = _ridge_oof(Xg, Yg, rng)
    r_em_g = pd.Series(_col_pearson(pred_g, Yg), index=M.columns)
    # 2. morphology -> expression
    pred_me, lam2 = _ridge_oof(Y, X, rng)
    r_me = pd.Series(_col_pearson(pred_me, X), index=E.columns)
    pred_me_null, _ = _ridge_oof(Y[perm], X, rng)
    r_me_null = pd.Series(_col_pearson(pred_me_null, X), index=E.columns)
    # 3. response magnitude
    mag_e = pd.Series(np.linalg.norm(X, axis=1), index=strains)
    mag_m = pd.Series(np.linalg.norm(Y, axis=1), index=strains)
    rho_mag = spearmanr(mag_e, mag_m)
    pc1_e = np.linalg.svd(X - X.mean(0), full_matrices=False)[0][:, 0]
    pc1_m = np.linalg.svd(Y - Y.mean(0), full_matrices=False)[0][:, 0]
    r_pc1 = float(abs(np.corrcoef(pc1_e, pc1_m)[0, 1]))

    def summ(r: pd.Series, mask: pd.Series | None = None) -> dict[str, float]:
        s = r if mask is None else r[mask.reindex(r.index).fillna(False)]
        s = s.dropna()
        return {
            "n": int(len(s)),
            "median": float(s.median()),
            "mean": float(s.mean()),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
            "max": float(s.max()),
            "frac_above_0.3": float((s > 0.3).mean()),
        }

    top = r_em.sort_values(ascending=False).head(15)
    out = {
        "generated_by": "experiments/019-simb-multimodal/scripts/expression_morphology_covariation.py",
        "inputs": {
            "expression_lmdb": LMDB_EXPRESSION,
            "morph_dir": args.morph_dir,
            "mt4718data_sha256": _sha256(osp.join(args.morph_dir, "mt4718data.tsv")),
        },
        "n_shared_strains": len(strains),
        "n_expression_genes": int(E.shape[1]),
        "n_morph_features": int(M.shape[1]),
        "n_moving_features": int(moving.sum()),
        "reliability_threshold": RELIABLE,
        "ridge_lambda_median": {
            "expression_to_morph": lam1,
            "morph_to_expression": lam2,
        },
        "expression_to_morphology": {
            "all_features": summ(r_em),
            "moving_features": summ(r_em, moving),
            "quiet_features": summ(r_em, ~moving),
            "null_all": summ(r_em_null),
            "first_component_removed_all": summ(r_em_g),
            "first_component_removed_moving": summ(r_em_g, moving),
            "top_features": [
                {
                    "feature": f,
                    "description": str(ceiling.loc[f, "description"]),
                    "r": float(v),
                    "reliability": float(reliability[f]),
                    "r_first_component_removed": float(r_em_g[f]),
                }
                for f, v in top.items()
            ],
        },
        "morphology_to_expression": {"all_genes": summ(r_me), "null": summ(r_me_null)},
        "response_magnitude": {
            "spearman": float(rho_mag.correlation),
            "p": float(rho_mag.pvalue),
            "n": len(strains),
        },
        "pc1_agreement_abs_r": r_pc1,
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(osp.join(RESULTS, "expression_morphology_covariation.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "inputs"}, indent=1)[:2600])


if __name__ == "__main__":
    main()
