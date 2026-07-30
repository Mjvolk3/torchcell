# experiments/019-simb-multimodal/scripts/morphology_noise_ceiling.py
# [[experiments.019-simb-multimodal.scripts.morphology_noise_ceiling]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/morphology_noise_ceiling
"""Estimate the ACHIEVABLE per-gene Pearson ceiling for Ohya-2005 CalMorph morphology.

Ohya 2005 gives ONE biological sample per deletion mutant (population-averaged over
cells) but 122 independent his3 wild-type replicates. So each mutant's 501-trait vector
is a single noisy measurement. We estimate, per feature, the reliability

    reliability_k = signal_var_k / total_var_k = 1 - noise_var_k / total_var_k

where noise_var_k = variance across the 122 WT replicates (measurement + replicate noise
of one genotype) and total_var_k = variance across the 4,718 mutants (biological signal +
that same single-replicate noise). This is broad-sense reliability (H^2). The maximum
achievable across-strain Pearson for a *perfect* predictor of the true signal against the
noisy target is

    ceiling_k = corr(signal, signal + noise) = sqrt(reliability_k).

Our metric `val/global/pearson_per_gene` = mean over features of the across-strain Pearson,
so its ceiling = mean_k ceiling_k over the features the model actually predicts (the 281
CALMORPH_LABELS minus the 3 dropped = 278). We compare that ceiling to the observed ~0.04.

Reads the sha256-pinned SCMD mirror (per manifest):
  $DATA_ROOT/torchcell-library/ohyaHighdimensionalLargescalePhenotyping2005a/data/
    mt4718data.tsv  (4,718 mutants x 501)   wt122data.tsv  (122 WT replicates x 501)

Run from repo root:  python experiments/019-simb-multimodal/scripts/morphology_noise_ceiling.py
"""

from __future__ import annotations

import os
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from torchcell.datamodels.calmorph_labels import CALMORPH_LABELS

DROPPED = [
    "A113_A",
    "D203",
    "D205",
]  # experiments/019 delta config multitask.drop_features.global


def _load(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    mt = pd.read_csv(osp.join(data_dir, "mt4718data.tsv"), sep="\t").set_index("ORF")
    wt = pd.read_csv(osp.join(data_dir, "wt122data.tsv"), sep="\t").set_index("NAME")
    return mt, wt


def _reliability(mt: pd.DataFrame, wt: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    noise_var = wt[feats].var(ddof=1)  # across 122 WT replicates
    total_var = mt[feats].var(ddof=1)  # across 4,718 mutants (signal + noise)
    rel = (1.0 - noise_var / total_var).clip(lower=0.0, upper=1.0)
    ceiling = np.sqrt(rel)
    return pd.DataFrame(
        {
            "noise_var": noise_var,
            "total_var": total_var,
            "reliability": rel,
            "ceiling": ceiling,
        }
    )


def _summary(name: str, df: pd.DataFrame) -> None:
    c = df["ceiling"]
    print(f"\n== {name} (n={len(df)}) ==")
    print(f"  mean ceiling (=max achievable mean per-gene Pearson): {c.mean():.4f}")
    print(
        f"  median ceiling: {c.median():.4f}   |  ceiling IQR: [{c.quantile(0.25):.3f}, {c.quantile(0.75):.3f}]"
    )
    print(f"  reliability mean: {df['reliability'].mean():.4f}")
    for thr in (0.05, 0.10, 0.20, 0.30, 0.50):
        print(
            f"    features with ceiling > {thr:.2f}: {(c > thr).sum():4d}  ({100 * (c > thr).mean():.1f}%)"
        )


def main() -> None:
    load_dotenv("/home/michaelvolk/Documents/projects/torchcell/.env")
    data_dir = osp.join(
        os.environ["DATA_ROOT"],
        "torchcell-library/ohyaHighdimensionalLargescalePhenotyping2005a/data",
    )
    mt, wt = _load(data_dir)
    cols = list(mt.columns)
    base = [k for k in CALMORPH_LABELS if k in cols]  # 281 base labels
    model_feats = [k for k in base if k not in DROPPED]  # the 278 the model predicts
    cv = [c for c in cols if c not in set(base)]  # 220 CV statistics
    print(
        f"loaded mt={mt.shape} wt={wt.shape} | base={len(base)} model={len(model_feats)} cv={len(cv)}"
    )

    rel_all = _reliability(mt, wt, cols)
    rel_model = rel_all.loc[model_feats]
    _summary("ALL 501 CalMorph features", rel_all)
    _summary("MODEL 278 base features (what pearson_per_gene scores)", rel_model)
    _summary("CV 220 statistics (not modeled)", rel_all.loc[cv])

    obs = 0.040  # observed best morph_002 val/global/pearson_per_gene (controlled _002 run)
    ceil = rel_model["ceiling"].mean()
    print("\n" + "=" * 64)
    print(f"OBSERVED morph per-gene (morph_002, 1,161 control): {obs:.3f}")
    print(f"CEILING  morph per-gene (278 feats, target noise) : {ceil:.3f}")
    print(f"fraction of ceiling realized: {obs / ceil:.1%}")
    print(f"headroom (ceiling - observed): {ceil - obs:.3f}")
    print("=" * 64)
    # top predictable features (where signal is real)
    top = rel_model.sort_values("ceiling", ascending=False).head(12)
    print("\nMost-reliable modeled features (ceiling | reliability | label):")
    for k, row in top.iterrows():
        print(
            f"  {k:10s} {row['ceiling']:.3f} | {row['reliability']:.3f} | {CALMORPH_LABELS.get(k, '')[:52]}"
        )


if __name__ == "__main__":
    main()
