# experiments/010-kuzmin-tmi/scripts/positive_interaction_retrieval.py
# [[experiments.010-kuzmin-tmi.scripts.positive_interaction_retrieval]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/positive_interaction_retrieval

"""Can the model retrieve POSITIVE trigenic interactions?

The companion measurement scored the negative side against the Kuzmin 2018 call,
which is one-sided by construction: "we used an established interaction magnitude
cut-off for digenic interactions $(p < 0.05, |\\varepsilon| > 0.08)$ and trigenic
interactions $(p < 0.05, \\tau < -0.08)$", with positives excluded because "We
focused exclusively on the analysis of deleterious negative trigenic
interactions."

Kuzmin 2020 and the 2021 protocol score positives symmetrically at
|tau| > 0.08 with p < 0.05, so the positive half of that call is the published
criterion here. It is askable on the same records: 20,426 of the 376,732 in this
build carry tau > +0.08 and 1,773 carry tau > +0.20.

The question is not academic. A panel built to find positive interactions is only
worth building if the ranking that picks it beats a model with no interaction
capacity, and that had never been measured on this side.

Everything except the criteria and the sort direction is shared with the negative
measurement, so this imports it rather than restating it. The ensemble row is
added here because a single checkpoint's tail is not the model's ranking.

Output: ``experiments/010-kuzmin-tmi/results/positive_interaction_retrieval.csv``
"""

import os
import os.path as osp
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import negative_interaction_retrieval as N  # noqa: E402

RESULTS_DIR = N.RESULTS_DIR
K_GRID = N.K_GRID
P_VALUE_MAX = N.P_VALUE_MAX

# The positive half of the Kuzmin 2020 symmetric call, its stringent tier, and
# two magnitude-only rows so the cost of dropping the p-value conjunct stays
# visible the same way it does on the negative side.
CRITERIA: list[tuple[str, float, bool]] = [
    ("K2020 positive tau>+0.08 & p<0.05", 0.08, True),
    ("stringent positive tau>+0.12 & p<0.05", 0.12, True),
    ("magnitude only tau>+0.08", 0.08, False),
    ("magnitude only tau>+0.20", 0.20, False),
]
HEADLINE = "K2020 positive tau>+0.08 & p<0.05"


def with_ensemble(preds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Add the mean of the three checkpoints, which is how the panel is ranked."""
    cgt = [v for k, v in preds.items() if k.startswith("CGT M")]
    if len(cgt) == 3:
        preds = dict(preds)
        preds["CGT ensemble"] = np.mean(cgt, axis=0)
    return preds


def retrieval(
    tau: np.ndarray, pval: np.ndarray, preds: dict[str, np.ndarray], split_name: str
) -> pd.DataFrame:
    """Precision among the K most POSITIVE predictions, against each criterion."""
    rows: list[dict[str, object]] = []
    for label, cut, use_p in CRITERIA:
        hit = (tau > cut) & (pval < P_VALUE_MAX) if use_p else (tau > cut)
        base = float(hit.mean())
        print(
            f"\n[{split_name}] {label}: {int(hit.sum())} of {tau.size} records, "
            f"base rate {base:.3%}"
        )
        for name, p in preds.items():
            order = np.argsort(p)[::-1]
            ap = float(average_precision_score(hit, p))
            line = [f"  {name:<18} AP {ap:.4f}"]
            for k in K_GRID:
                if k > tau.size:
                    continue
                prec = float(hit[order[:k]].mean())
                rows.append(
                    {
                        "split": split_name,
                        "criterion": label,
                        "tau_cut": cut,
                        "uses_p_value": use_p,
                        "model": name,
                        "k": k,
                        "precision": prec,
                        "base_rate": base,
                        "enrichment": prec / base if base > 0 else np.nan,
                        "average_precision": ap,
                    }
                )
                if k in (100, 1000):
                    line.append(f"P@{k} {prec:.3f} ({prec / base:.1f}x)")
            print("  ".join(line))
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tau_all, pval_all, rows = N.labels()
    print(f"random-split test: {rows.size} records")
    frames = [
        retrieval(
            tau_all[rows],
            pval_all[rows],
            with_ensemble(N.random_split_predictions(rows)),
            "random",
        )
    ]

    te_d, preds_d = N.disjoint_split_rows()
    print(f"\nquery-pair-disjoint test: {te_d.size} records")
    frames.append(
        retrieval(tau_all[te_d], pval_all[te_d], preds_d, "query_pair_disjoint")
    )

    out = pd.concat(frames, ignore_index=True)
    path = osp.join(RESULTS_DIR, "positive_interaction_retrieval.csv")
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
