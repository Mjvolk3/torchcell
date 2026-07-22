# experiments/019-simb-multimodal/scripts/calmorph_variance_analysis.py
# [[experiments.019-simb-multimodal.scripts.calmorph_variance_analysis]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/calmorph_variance_analysis
"""Per-feature variance / dynamic-range analysis of the Ohya-2005 CalMorph target (WS10b).

The morphology training target (the served ``calmorph`` vector) is the 281 CalMorph
BASE parameters (``CALMORPH_LABELS``); the 220 coefficient-of-variation parameters
(``CALMORPH_STATISTICS``) are stored separately as ``calmorph_coefficient_of_variation``
-- together 501, the nominal CalMorph vocabulary. The target is RAW/unnormalized and the
281 features live on wildly different scales (cell-size counts ~1e4 vs 0-1 ratios), which
is why an un-normalized MSE loss is O(1e6).

This script computes, PER BASE FEATURE across the ~4718 mutants, the mean/std/variance,
min/max/range, and the coefficient of variation (std / |mean|), then flags NEAR-CONSTANT /
degenerate features -- the "values barely change across samples" set the user flagged. It
also cross-references the Ohya-2005 paper's own reliability verdict (Box-Cox + Shapiro-Wilk
kept 254 of 501 parameters; discarded 247 as non-normal / unreliable).

Reads the sha256-pinned raw mirror (identical to the served target modulo the ~23
gene-name-reconciled rows -- negligible for feature-level variance). Writes:

* ``results/calmorph_feature_variance.csv``  -- per-feature stats, sorted ascending by
  robust CV (degenerate first).
* ``results/calmorph_variance_summary.json`` -- thresholds + drop-candidate lists +
  keep/drop recommendation evidence.

Usage (from repo root):
    python experiments/019-simb-multimodal/scripts/calmorph_variance_analysis.py
"""

from __future__ import annotations

import json
import os
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from torchcell.datamodels.calmorph_labels import CALMORPH_LABELS, CALMORPH_STATISTICS

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

RAW_MUTANT = osp.join(
    DATA_ROOT,
    "torchcell-library",
    "ohyaHighdimensionalLargescalePhenotyping2005a",
    "data",
    "mt4718data.tsv",
)
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "019-simb-multimodal", "results")

# Degeneracy threshold on the ROBUST coefficient of variation (IQR / |median|). A feature
# whose middle-50% spread is < this fraction of its median barely moves across mutants;
# z-scoring it divides by a near-zero std and amplifies pure measurement noise to unit
# variance. Chosen at 0.01 (1%): well below any biologically informative CalMorph ratio,
# and it isolates the handful of structurally-degenerate parameters (near-constant ratios
# / bounded fractions) rather than merely tight ones.
ROBUST_CV_DEGENERATE = 0.01
# Hard-zero-variance guard: std == 0 exactly (a truly constant column) MUST be dropped or
# floored -- z-score is undefined.
ZERO_STD_EPS = 1e-12


def main() -> None:
    df = pd.read_csv(RAW_MUTANT, sep="\t")
    base_cols = [c for c in df.columns if c in CALMORPH_LABELS]
    cv_cols = [c for c in df.columns if c in CALMORPH_STATISTICS]
    assert len(base_cols) == 281, f"expected 281 base cols, got {len(base_cols)}"
    assert len(cv_cols) == 220, f"expected 220 CV cols, got {len(cv_cols)}"

    n_mutants = len(df)
    rows = []
    for col in base_cols:
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        mean = float(np.mean(v))
        std = float(np.std(v))
        vmin = float(np.min(v))
        vmax = float(np.max(v))
        median = float(np.median(v))
        q25 = float(np.percentile(v, 25))
        q75 = float(np.percentile(v, 75))
        iqr = q75 - q25
        cv = std / abs(mean) if abs(mean) > 0 else np.inf
        robust_cv = iqr / abs(median) if abs(median) > 0 else (0.0 if iqr == 0 else np.inf)
        rows.append(
            {
                "feature": col,
                "description": CALMORPH_LABELS[col],
                "mean": mean,
                "std": std,
                "variance": std**2,
                "min": vmin,
                "max": vmax,
                "range": vmax - vmin,
                "median": median,
                "iqr": iqr,
                "cv": cv,
                "robust_cv": robust_cv,
                "n": int(v.size),
            }
        )

    stats = pd.DataFrame(rows).sort_values("robust_cv", ascending=True).reset_index(drop=True)

    zero_std = stats[stats["std"] <= ZERO_STD_EPS]["feature"].tolist()
    degenerate = stats[stats["robust_cv"] < ROBUST_CV_DEGENERATE]["feature"].tolist()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = osp.join(RESULTS_DIR, "calmorph_feature_variance.csv")
    stats.to_csv(csv_path, index=False)

    summary = {
        "source_file": RAW_MUTANT,
        "n_mutants": n_mutants,
        "n_base_features": len(base_cols),
        "n_cv_features": len(cv_cols),
        "n_total_features": len(base_cols) + len(cv_cols),
        "robust_cv_degenerate_threshold": ROBUST_CV_DEGENERATE,
        "zero_std_features": zero_std,
        "n_zero_std": len(zero_std),
        "degenerate_features": degenerate,
        "n_degenerate": len(degenerate),
        "scale_span": {
            "min_mean_abs": float(stats["mean"].abs().min()),
            "max_mean_abs": float(stats["mean"].abs().max()),
            "min_std": float(stats["std"].min()),
            "max_std": float(stats["std"].max()),
        },
        "least_variable_top20": stats.head(20)[
            ["feature", "description", "mean", "std", "robust_cv"]
        ].to_dict(orient="records"),
        "ohya_paper_reliability": {
            "note": (
                "Ohya 2005 SI (si12.md, 'Supporting Text') fit a per-parameter Box-Cox "
                "power transform to the WILD-TYPE data (x divided by the WT mean, then "
                "F_{p,a}(x), then standardized y=(F-Mean)/SD), and applied a Shapiro-Wilk "
                "normality test. At P>=0.5 they KEPT 254 of 501 parameters and DISCARDED "
                "247 as non-normal / unreliable. This is the published precedent for "
                "dropping degenerate CalMorph parameters."
            ),
            "kept_parameters_at_p0.5": 254,
            "discarded_parameters": 247,
            "of_total": 501,
        },
        "recommendation": (
            "KEEP all 281 base features with a per-feature z-score computed on the TRAIN "
            "split only, using an epsilon-floored std (std + eps) so any near-constant / "
            "zero-std feature maps to ~0 instead of exploding. FLAG the degenerate list "
            "above for optional user-driven dropping -- do NOT silently drop. The Ohya "
            "paper's own 254/247 keep/discard is a normality verdict (for their abnormality "
            "test), not a variance floor, and it spans all 501 params (base+CV); our target "
            "is only the 281 base params, so we default to keep-with-epsilon and surface the "
            "candidates rather than importing their 247-drop wholesale."
        ),
    }
    json_path = osp.join(RESULTS_DIR, "calmorph_variance_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"n_mutants={n_mutants} base={len(base_cols)} cv={len(cv_cols)}")
    print(f"zero-std features ({len(zero_std)}): {zero_std}")
    print(f"degenerate (robust_cv<{ROBUST_CV_DEGENERATE}) [{len(degenerate)}]: {degenerate}")
    print("\nLeast-variable 15 base features (robust_cv ascending):")
    print(
        stats.head(15)[
            ["feature", "description", "mean", "std", "robust_cv", "range"]
        ].to_string(index=False)
    )
    print(
        f"\nScale span: |mean| in "
        f"[{stats['mean'].abs().min():.4g}, {stats['mean'].abs().max():.4g}]; "
        f"std in [{stats['std'].min():.4g}, {stats['std'].max():.4g}]"
    )


if __name__ == "__main__":
    main()
