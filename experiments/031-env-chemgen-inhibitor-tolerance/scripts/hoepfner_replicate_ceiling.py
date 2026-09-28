# experiments/031-env-chemgen-inhibitor-tolerance/scripts/hoepfner_replicate_ceiling.py
# [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.hoepfner_replicate_ceiling]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-env-chemgen-inhibitor-tolerance/scripts/hoepfner_replicate_ceiling
"""Can a prediction ceiling be recovered for Hoepfner, which serves no uncertainty?

Every other dataset in experiment 031 has a ceiling because its source serves a per-record
standard error. Hoepfner serves none, and its loader records that as a typed provenance gap
rather than a silent absence: *"the replicate t-test p-value is folded into the adjusted
score"*. This script asks whether the ceiling can be recovered another way, and answers two
different questions that are easy to confuse.

QUESTION 1: ARE THE TECHNICAL REPLICATES RECOVERABLE? No, and the deposited files settle it.
Each compound was profiled at n = 2 wells in the same plate, but the deposit carries ONE
column per experiment, an ``Ad.`` adjusted score in which those two wells are already
combined. ``audit_columns`` verifies this by parsing every column header of both score
matrices: every adjusted-score column carries a distinct experiment id, so no experiment
appears twice and there is no second replicate to correlate against. The companion
``z-score`` columns are a second normalization of the SAME numbers, so correlating them with
their own MADL column measures the normalization, not the measurement.

QUESTION 2: IS THERE ANY REPEATED MEASUREMENT AT ALL? Yes. Some compounds were screened at the
same concentration in more than one study, and a study is a separate screen with its own
control set. Two such screens are parallel measurements of the same condition, so their
agreement over genes estimates the reliability of ONE served value directly: for parallel
measurements the observed correlation IS the reliability, and the ceiling on correlation with
the noise-free response is its square root.

The estimate this yields is NOT the same quantity as the served-error ceilings elsewhere in
the document, and the difference matters in the conservative direction. A cross-screen repeat
carries batch variation that two wells in one plate do not, so it estimates the reliability of
a served value against everything that changes between screens. That is closer to what a model
trained across screens actually faces, and it is closer to Vanacloig's three-batch replicate
agreement than to a within-plate standard error.

Writes ``results/hoepfner_column_audit.csv`` (the column-type census that answers question 1)
and ``results/hoepfner_cross_screen_reliability.csv`` (one row per repeated condition pair).
"""

from __future__ import annotations

import os
import os.path as osp
import re
from itertools import combinations

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr, spearmanr

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "031-env-chemgen-inhibitor-tolerance", "results"
)
RAW_DIR = osp.join(
    DATA_ROOT, "torchcell-raw", "hoepfnerHighresolutionChemicalDissection2014"
)
SCORE_FILES = ("HIP_scores.txt", "HOP_scores.txt")
#: A deposited column is one of three kinds. ``Ad.`` is the adjusted score the loader stores,
#: already combining the two wells; ``MADL`` is the unadjusted score for the few experiments
#: that have no adjusted column; `` z-score`` is the companion gene-wise normalization.
COLUMN_KINDS = {
    "adjusted": re.compile(r"^Ad\. scores for Exp\. (?P<exp>.+?)(?P<z> z-score)?$"),
    "madl": re.compile(r"^MADL.*?(?P<exp>\S+)(?P<z> z-score)?$"),
}
#: The minimum genes in common before a condition pair is correlated at all.
MIN_GENES = 500


def audit_columns() -> pd.DataFrame:
    """Every column of both score matrices by kind, and whether any experiment repeats.

    This is the evidence for question 1: if no experiment id appears on two adjusted-score
    columns, the two technical replicates are not separately deposited.
    """
    rows = []
    for fname in SCORE_FILES:
        path = osp.join(RAW_DIR, fname)
        with open(path) as fh:
            header = fh.readline().rstrip("\n").split("\t")
        cols = [c.strip('"') for c in header[1:]]
        adjusted, madl, zscore = [], [], []
        for c in cols:
            if c.endswith(" z-score"):
                zscore.append(c)
            elif c.startswith("Ad. scores"):
                adjusted.append(c[len("Ad. scores for Exp. ") :])
            else:
                madl.append(c)
        counts = pd.Series(adjusted).value_counts()
        rows.append(
            {
                "file": fname,
                "columns": len(cols),
                "adjusted_score_columns": len(adjusted),
                "madl_columns": len(madl),
                "companion_zscore_columns": len(zscore),
                "distinct_experiment_ids": int(counts.size),
                "experiment_ids_appearing_twice": int((counts > 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def load_cells() -> pd.DataFrame:
    """Served Hoepfner records as (arm, compound, dose, screen, gene) -> response."""
    df = pd.read_parquet(
        osp.join(RESULTS_DIR, "records_hoepfner2014.parquet"),
        columns=[
            "gene",
            "compound",
            "inchikey",
            "response",
            "screen_id",
            "dose_value",
            "dose_unit",
            "perturbation_type",
        ],
    )
    # the two arms are different strain collections, so a repeat only counts within an arm
    df["arm"] = df["perturbation_type"].map(
        {"engineered_copy_number": "HIP", "kanmx_deletion": "HOP"}
    )
    return df


def cross_screen_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Agreement between two screens of the same compound at the same concentration."""
    rows = []
    keys = ["arm", "inchikey", "dose_value", "dose_unit"]
    for key, grp in df.groupby(keys, dropna=False):
        screens = sorted(grp["screen_id"].dropna().unique())
        if len(screens) < 2:
            continue
        profiles = {
            s: grp[grp["screen_id"] == s].set_index("gene")["response"] for s in screens
        }
        for a, b in combinations(screens, 2):
            joined = (
                profiles[a].to_frame("a").join(profiles[b].to_frame("b"), how="inner")
            )
            joined = joined[np.isfinite(joined["a"]) & np.isfinite(joined["b"])]
            if len(joined) < MIN_GENES:
                continue
            # For two PARALLEL measurements of the same condition the observed correlation
            # estimates the reliability of ONE of them, so the ceiling on correlation with
            # the noise-free response is its square root.
            r = float(pearsonr(joined["a"], joined["b"])[0])
            rho = float(spearmanr(joined["a"], joined["b"])[0])
            rows.append(
                {
                    "arm": key[0],
                    "inchikey": key[1],
                    "compound": grp["compound"].iloc[0],
                    "dose_value": key[2],
                    "dose_unit": key[3],
                    "screen_a": a,
                    "screen_b": b,
                    "n_genes": len(joined),
                    "pearson": r,
                    "spearman": rho,
                    "reliability_single": r,
                    "ceiling_r_truth": float(np.sqrt(r)) if r > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    audit = audit_columns()
    audit.to_csv(osp.join(RESULTS_DIR, "hoepfner_column_audit.csv"), index=False)
    print("COLUMN AUDIT -- are the two technical replicates separately deposited?")
    print(audit.to_string(index=False))
    repeated = int(audit["experiment_ids_appearing_twice"].sum())
    print(
        f"\n  {repeated} experiment ids appear on more than one adjusted-score column, "
        "so the two wells are folded into one column and the technical replicates are "
        "NOT recoverable."
    )

    df = load_cells()
    pairs = cross_screen_pairs(df)
    pairs.to_csv(
        osp.join(RESULTS_DIR, "hoepfner_cross_screen_reliability.csv"), index=False
    )
    print("\nCROSS-SCREEN REPEATS -- the same compound and dose in two studies")
    if pairs.empty:
        print("  none found")
        return
    print(
        pairs[
            [
                "arm",
                "compound",
                "dose_value",
                "screen_a",
                "screen_b",
                "n_genes",
                "pearson",
                "ceiling_r_truth",
            ]
        ].to_string(index=False)
    )
    for arm, g in pairs.groupby("arm"):
        print(
            f"\n{arm}: {len(g)} condition pairs, median reliability "
            f"{g['reliability_single'].median():.3f}, median ceiling "
            f"{g['ceiling_r_truth'].median():.3f}"
        )
    print(
        f"\nall: {len(pairs)} condition pairs over "
        f"{pairs['inchikey'].nunique()} compounds, median reliability "
        f"{pairs['reliability_single'].median():.3f}, median ceiling "
        f"{pairs['ceiling_r_truth'].median():.3f}"
    )


if __name__ == "__main__":
    main()
