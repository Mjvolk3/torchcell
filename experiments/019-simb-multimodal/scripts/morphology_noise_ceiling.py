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

import json
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


def _scalar_shortlist(rel: pd.DataFrame, mt: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Rank single features as candidate SCALAR morphology targets.

    A scalar target is only worth training on if two things hold at once: the measurement
    is reliable (high ceiling), and deletions actually move it (high spread relative to
    its own scale). Ranking on ceiling alone selects features that are precisely measured
    and nearly constant across mutants, which is the wrong end of the trade. The rank key
    is `ceiling * robust_cv`, where robust_cv = IQR / |median| across the 4,718 mutants,
    so a feature has to clear both bars.
    """
    feats = list(rel.index)
    q75 = mt[feats].quantile(0.75)
    q25 = mt[feats].quantile(0.25)
    median = mt[feats].median().abs()
    robust_cv = ((q75 - q25) / median.replace(0.0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )
    out = rel.assign(
        robust_cv=robust_cv,
        label=[CALMORPH_LABELS.get(k, "") for k in feats],
        score=rel["ceiling"] * robust_cv,
    )
    return out.sort_values("score", ascending=False).head(n)


def _observed_best(experiment_root: str) -> dict[str, float | str | None]:
    """Best morphology score on the committed leaderboard, or nothing if it is absent.

    Read rather than hardcoded: the earlier fixed 0.040 was the morph_002 control run and
    silently stopped being the best score once morph_v5 landed.
    """
    path = osp.join(
        experiment_root, "019-simb-multimodal", "results", "round_leaderboards.csv"
    )
    if not osp.exists(path):
        return {"source": None}
    board = pd.read_csv(path)
    morph = board[board["strand"] == "morphology"]
    morph = morph[~morph["is_collapsed"].fillna(False)]
    row = morph.loc[morph["primary_roll_max"].idxmax()]
    return {
        "source": path,
        "run_id": str(row["run_id"]),
        "roll_max": float(row["primary_roll_max"]),
        "epochs": float(row["epochs"]),
        "epoch_at_roll_max": float(row["primary_epoch_at_roll_max"]),
    }


def main() -> None:
    load_dotenv()
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

    ceil = rel_model["ceiling"].mean()
    observed = _observed_best(os.environ["EXPERIMENT_ROOT"])
    obs = observed.get("roll_max")
    print("\n" + "=" * 64)
    if obs is None:
        print("OBSERVED morph per-gene: no leaderboard; run pull_round_leaderboards.py")
    else:
        print(
            f"OBSERVED morph per-feature (run {observed['run_id']}, peak epoch "
            f"{observed['epoch_at_roll_max']:.0f} of {observed['epochs']:.0f}): {obs:.4f}"
        )
    print(f"CEILING  morph per-gene (278 feats, target noise) : {ceil:.3f}")
    if obs is not None:
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

    shortlist = _scalar_shortlist(rel_model, mt)
    print("\nScalar-target shortlist (ceiling x robust CV | ceiling | robust CV | label):")
    for k, row in shortlist.iterrows():
        print(
            f"  {k:10s} {row['score']:.3f} | {row['ceiling']:.3f} | {row['robust_cv']:.3f} | "
            f"{str(row['label'])[:44]}"
        )

    results_dir = osp.join(
        os.environ["EXPERIMENT_ROOT"], "019-simb-multimodal", "results"
    )
    os.makedirs(results_dir, exist_ok=True)
    rel_model.assign(label=[CALMORPH_LABELS.get(k, "") for k in rel_model.index]).to_csv(
        osp.join(results_dir, "morphology_feature_ceiling.csv")
    )
    payload = {
        "n_mutants": int(len(mt)),
        "n_wt_replicates": int(len(wt)),
        "n_base_features": len(base),
        "n_model_features": len(model_feats),
        "dropped_features": DROPPED,
        "ceiling_mean_model_features": float(ceil),
        "ceiling_median_model_features": float(rel_model["ceiling"].median()),
        "reliability_mean_model_features": float(rel_model["reliability"].mean()),
        "n_model_features_ceiling_above_0p5": int((rel_model["ceiling"] > 0.5).sum()),
        "ceiling_mean_all_501": float(rel_all["ceiling"].mean()),
        "observed_best": observed,
        "fraction_of_ceiling_realized": (None if obs is None else float(obs / ceil)),
        "scalar_shortlist": shortlist.reset_index()
        .rename(columns={"index": "feature"})
        .to_dict("records"),
    }
    with open(osp.join(results_dir, "morphology_noise_ceiling.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n-> {osp.join(results_dir, 'morphology_noise_ceiling.json')}")


if __name__ == "__main__":
    main()
