# experiments/023-metabolome-betaxanthin-joint/scripts/bx_aa_incremental_over_model.py
# [[experiments.023-metabolome-betaxanthin-joint.scripts.bx_aa_incremental_over_model]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/023-metabolome-betaxanthin-joint/scripts/bx_aa_incremental_over_model
"""Is the amino-acid to betaxanthin coupling REDUNDANT with genotype-only supervision?

THROWAWAY REVIEW SCRIPT for round-2 comments [67] and [80]. It does not replace
`betaxanthin_amino_acid_predictivity.py`; it reuses that script's loaders and its
`nineteen_given_fitness` residualization pattern with a different control variable.

THE QUESTION. `betaxanthin_amino_acid_predictivity.py` measures out-of-fold r = 0.298
from the 19 measured Mulleder amino acids to Cachera betaxanthin over 4,432 shared
deletions. That number is UNCONDITIONAL on genotype. Both the amino-acid pool and
betaxanthin are readouts of the same deletion, so the 0.298 does not separate

  (i) the amino-acid pool carries betaxanthin-relevant information a genotype-to-
      betaxanthin model does not already have, from
  (ii) both are shadows of one genotype axis the model already learns from the Cachera
      labels alone.

Only (i) leaves room for a shared encoder to beat betaxanthin-only training.

THE MEASUREMENT. Residualize BOTH sides on the model's own held-out betaxanthin
prediction and refit. If the residual r collapses to zero the coupling is redundant with
what genotype supervision already delivers; if it survives, there is signal the
genotype-to-betaxanthin map is not capturing.

WHERE THE MODEL PREDICTIONS COME FROM. `train_cgt_multitask._dump_test_predictions`,
enabled by `trainer.dump_test_predictions` on the `delta_grid_betaxanthin` grid, writes
per-gene test-split dumps to $DATA_ROOT/test-predictions/*.json. Every dump used here has
`active_heads == ["betaxanthin"]`, so the control variable is a BETAXANTHIN-ONLY model's
prediction on genes it never trained on, which is exactly the quantity a shared encoder
would have to beat.

Run from repo root:
  PYTHONPATH=. python experiments/023-metabolome-betaxanthin-joint/scripts/bx_aa_incremental_over_model.py
"""

from __future__ import annotations

import glob
import json
import os
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

ALPHAS = np.logspace(-2, 4, 25)
N_FOLDS = 5
CV_SEEDS = (0, 1, 2)


def _z(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return (a - a.mean(axis=0)) / a.std(axis=0)


def _residualize(y: np.ndarray, c: np.ndarray) -> np.ndarray:
    design = np.hstack([np.ones((len(c), 1)), c])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return y - design @ beta


def _cv_score(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Out-of-fold Pearson of a ridge fit, averaged over three fold shuffles."""
    scores = []
    for seed in CV_SEEDS:
        folds = KFold(N_FOLDS, shuffle=True, random_state=seed)
        oof = np.zeros_like(y)
        for train_idx, test_idx in folds.split(x):
            model = RidgeCV(alphas=ALPHAS).fit(x[train_idx], y[train_idx])
            oof[test_idx] = model.predict(x[test_idx])
        scores.append((pearsonr(oof, y)[0], spearmanr(oof, y)[0]))
    arr = np.asarray(scores)
    r = float(arr[:, 0].mean())
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se_z = 1.0 / np.sqrt(len(y) - 3)
    return {
        "pearson": r,
        "pearson_sd_across_shuffles": float(arr[:, 0].std(ddof=1)),
        "pearson_se_fisher": float((np.tanh(z + se_z) - np.tanh(z - se_z)) / 2.0),
        "spearman": float(arr[:, 1].mean()),
        "n": int(len(y)),
        "n_features": int(x.shape[1]),
    }


def load_dumps(data_root: str) -> list[dict]:
    """Betaxanthin-only test-prediction dumps, one per grid run."""
    paths = sorted(glob.glob(osp.join(data_root, "test-predictions", "*.json")))
    dumps = []
    for path in paths:
        with open(path) as fh:
            payload = json.load(fh)
        if payload.get("active_heads") != ["betaxanthin"]:
            continue
        rows = []
        for rec in payload["predictions"]["betaxanthin"]:
            # A handful of test records carry a null target (the head had no label for
            # that genotype); they cannot be scored and are dropped, counted below.
            if len(rec["genes"]) != 1 or rec["target"] is None:
                continue
            rows.append(
                {
                    "orf": rec["genes"][0],
                    "pred": float(rec["pred"][0]),
                    "target": float(rec["target"][0]),
                }
            )
        frame = pd.DataFrame(rows).groupby("orf", as_index=False).mean()
        dumps.append(
            {
                "path": path,
                "tags": payload.get("wandb_tags", []),
                "seed": payload.get("seed"),
                "n_test_records": payload.get("n_test_records"),
                "frame": frame,
            }
        )
    return dumps


def main() -> None:
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    experiment_root = os.environ["EXPERIMENT_ROOT"]

    tc = osp.join(data_root, "data", "torchcell")
    betaxanthin = pd.read_csv(
        osp.join(tc, "betaxanthin_cachera2023", "preprocess", "data.csv")
    )
    amino_acid = pd.read_csv(
        osp.join(tc, "amino_acid_mulleder2016", "preprocess", "data.csv")
    )
    names = [c for c in amino_acid.columns if c != "orf"]
    shared = betaxanthin.merge(amino_acid, on="orf", how="inner")

    dumps = load_dumps(data_root)
    per_run = []
    for dump in dumps:
        merged = dump["frame"].merge(shared, on="orf", how="inner")
        n = len(merged)
        if n < 100:
            per_run.append({"path": dump["path"], "n": n, "skipped": "too few genes"})
            continue

        level = merged["level"].to_numpy(dtype=float)
        pred = merged["pred"].to_numpy(dtype=float)
        dump_target = merged["target"].to_numpy(dtype=float)
        pool = _z(np.log(merged[names].to_numpy(dtype=float) + 1e-6))
        control = _z(pred.reshape(-1, 1))

        aa_alone = _cv_score(pool, level)
        resid_pool = _residualize(pool, control)
        resid_level = _residualize(level.reshape(-1, 1), control).ravel()
        aa_given_model = _cv_score(resid_pool, resid_level)

        per_run.append(
            {
                "path": dump["path"],
                "tags": dump["tags"],
                "seed": dump["seed"],
                "n_test_records": dump["n_test_records"],
                "n_scored": n,
                # sanity: the dump's stored target must be the same quantity as the
                # loader's `level`, otherwise the join is wrong.
                "dump_target_vs_level_pearson": float(pearsonr(dump_target, level)[0]),
                "model_pearson_on_scored_genes": float(pearsonr(pred, level)[0]),
                "aa_pool_alone": aa_alone,
                "aa_pool_given_model": aa_given_model,
                "delta": aa_alone["pearson"] - aa_given_model["pearson"],
                "model_pred_vs_pool_pearson": float(
                    pearsonr(
                        pred,
                        RidgeCV(alphas=ALPHAS).fit(pool, level).predict(pool),
                    )[0]
                ),
            }
        )

    live = [r for r in per_run if "aa_pool_alone" in r]
    summary = {
        "n_dumps": len(dumps),
        "n_scored_runs": len(live),
        "genes_shared_cachera_mulleder": int(len(shared)),
        "mean_n_scored": float(np.mean([r["n_scored"] for r in live])) if live else None,
        "mean_model_pearson": (
            float(np.mean([r["model_pearson_on_scored_genes"] for r in live]))
            if live
            else None
        ),
        "mean_aa_alone": (
            float(np.mean([r["aa_pool_alone"]["pearson"] for r in live])) if live else None
        ),
        "sd_aa_alone": (
            float(np.std([r["aa_pool_alone"]["pearson"] for r in live], ddof=1))
            if len(live) > 1
            else None
        ),
        "mean_aa_given_model": (
            float(np.mean([r["aa_pool_given_model"]["pearson"] for r in live]))
            if live
            else None
        ),
        "sd_aa_given_model": (
            float(np.std([r["aa_pool_given_model"]["pearson"] for r in live], ddof=1))
            if len(live) > 1
            else None
        ),
        "per_run": per_run,
    }

    out_dir = osp.join(experiment_root, "023-metabolome-betaxanthin-joint", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, "bx_aa_incremental_over_model.json")
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"dumps found: {len(dumps)}   scored: {len(live)}")
    print(f"shared Cachera x Mulleder deletions: {len(shared)}")
    print("\nrun                                   n   r_model  r_aa   r_aa|model  corr(pred,aa)")
    for r in live:
        tag = next((t for t in r["tags"] if t.startswith("s")), "?")
        print(
            f"{tag:<34s} {r['n_scored']:>4d}  {r['model_pearson_on_scored_genes']:+.4f}  "
            f"{r['aa_pool_alone']['pearson']:+.4f}  {r['aa_pool_given_model']['pearson']:+.4f}     "
            f"{r['model_pred_vs_pool_pearson']:+.4f}"
        )
    print(
        f"\nmean r_aa      = {summary['mean_aa_alone']:.4f} "
        f"(sd {summary['sd_aa_alone']:.4f})"
    )
    print(
        f"mean r_aa|model = {summary['mean_aa_given_model']:.4f} "
        f"(sd {summary['sd_aa_given_model']:.4f})"
    )
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
