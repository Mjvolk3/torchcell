# experiments/019-simb-multimodal/scripts/prediction_shrinkage_probe.py
# [[experiments.019-simb-multimodal.scripts.prediction_shrinkage_probe]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/prediction_shrinkage_probe
"""Is the rising validation loss a magnitude problem a single scalar can fix?

The expression and proteome heads reach their best per-feature Pearson on a validation
loss that has been rising for hundreds of epochs, and at that point no run beats the
per-gene mean on squared error (best normalized error 1.005). A predictor with correlation
r to the target minimizes squared error when sd(pred) / sd(target) = r; the runs sit near
twice that. If that is the whole story, ONE scalar that shrinks every prediction toward
the per-gene mean should bring the normalized error below 1 without touching Pearson.

This probe tests exactly that on the saved validation predictions
($DATA_ROOT/val-predictions/*.json, one file per checkpoint), with no GPU and no
retraining. The validation strains are cut into two halves (fixed seed); on one half it
fits a per-gene intercept (the half's target mean) and one global shrink factor c by least
squares on target-standardized residuals; on the other half it scores

    mean   the intercept alone                       (the per-gene-mean predictor)
    raw    intercept + (pred - pred mean)            (the model's own magnitudes)
    shrunk intercept + c * (pred - pred mean)        (one scalar)

as normalized squared error (per gene, squared error over the scoring half's target
variance, averaged over genes), then swaps the halves and averages. Pearson per feature
is unchanged by construction and is reported on the same halves.

    python experiments/019-simb-multimodal/scripts/prediction_shrinkage_probe.py

Writes results/prediction_shrinkage_probe.json.
"""
from __future__ import annotations

import glob
import json
import os
import os.path as osp

import numpy as np
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
RESULTS = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")
SEED = 0


def load(path: str) -> tuple[dict, np.ndarray, np.ndarray]:
    with open(path) as f:
        d = json.load(f)
    head = d["active_heads"][0]
    rows = d["predictions"][head]
    pred = np.array([r["pred"] for r in rows], dtype=np.float64)
    target = np.array([r["target"] for r in rows], dtype=np.float64)
    return d, pred, target


def colmean(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    return np.where(ok, x, 0.0).sum(0) / ok.sum(0).clip(min=1)


def half_scores(
    pred: np.ndarray, target: np.ndarray, fit: np.ndarray, score: np.ndarray
) -> dict[str, float]:
    ok_f = np.isfinite(target[fit])
    ok_s = np.isfinite(target[score])
    t_f, p_f = target[fit], pred[fit]
    t_s, p_s = target[score], pred[score]
    a = colmean(t_f, ok_f)  # per-gene intercept
    pm = p_f.mean(0)  # per-gene prediction mean on the fit half
    sd_f = np.sqrt(colmean((t_f - a) ** 2, ok_f)).clip(min=1e-8)
    # One scalar c: least squares of standardized target residual on standardized
    # centered prediction, pooled over genes and strains of the fit half.
    x = np.where(ok_f, (p_f - pm) / sd_f, 0.0)
    y = np.where(ok_f, (t_f - a) / sd_f, 0.0)
    c = float((x * y).sum() / (x * x).sum())
    var_s = colmean((t_s - colmean(t_s, ok_s)) ** 2, ok_s).clip(min=1e-8)
    keep = (ok_s.sum(0) >= 5) & (ok_f.sum(0) >= 5) & (var_s > 1e-8)

    def nmse(yhat: np.ndarray) -> float:
        return float((colmean((t_s - yhat) ** 2, ok_s) / var_s)[keep].mean())

    centered = p_s - pm
    # Pearson per feature on the scoring half (shrink-invariant).
    tc = np.where(ok_s, t_s - colmean(t_s, ok_s), 0.0)
    pc = np.where(ok_s, p_s - colmean(p_s, ok_s), 0.0)
    r = (tc * pc).sum(0) / np.sqrt((tc**2).sum(0) * (pc**2).sum(0)).clip(min=1e-12)
    sd_ratio = np.sqrt(colmean(pc**2, ok_s) / var_s)
    return {
        "c": c,
        "nmse_mean": nmse(np.broadcast_to(a, t_s.shape)),
        "nmse_raw": nmse(a + centered),
        "nmse_shrunk": nmse(a + c * centered),
        "pearson_per_feature": float(r[keep].mean()),
        "pred_sd_ratio": float(sd_ratio[keep].mean()),
        "n_features": int(keep.sum()),
    }


def main() -> None:
    out = []
    for path in sorted(glob.glob(osp.join(DATA_ROOT, "val-predictions", "*.json"))):
        d, pred, target = load(path)
        n = pred.shape[0]
        perm = np.random.default_rng(SEED).permutation(n)
        a_idx, b_idx = perm[: n // 2], perm[n // 2 :]
        halves = [half_scores(pred, target, a_idx, b_idx), half_scores(pred, target, b_idx, a_idx)]
        row = {k: float(np.mean([h[k] for h in halves])) for k in halves[0]}
        row.update(
            file=osp.basename(path),
            tags=d["wandb_tags"],
            n_strains=n,
            spread_over_r=row["pred_sd_ratio"] / row["pearson_per_feature"],
        )
        out.append(row)
        arm = next(t for t in d["wandb_tags"] if t[:2] in ("V_", "P_"))
        print(
            f"{arm:<12} n={n:<4} r={row['pearson_per_feature']:.3f} "
            f"sd_ratio={row['pred_sd_ratio']:.3f} c={row['c']:.3f}  nmse: "
            f"mean {row['nmse_mean']:.4f}  raw {row['nmse_raw']:.4f}  "
            f"shrunk {row['nmse_shrunk']:.4f}"
        )
    os.makedirs(RESULTS, exist_ok=True)
    with open(osp.join(RESULTS, "prediction_shrinkage_probe.json"), "w") as f:
        json.dump({"generated_by": osp.relpath(__file__), "seed": SEED, "rows": out}, f, indent=2)


if __name__ == "__main__":
    main()
