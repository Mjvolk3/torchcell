# experiments/025-solid-growth/scripts/paired_bootstrap_025.py
# [[experiments.025-solid-growth.scripts.paired_bootstrap_025]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/paired_bootstrap_025
"""The arm Q test comparison: the scored transformer checkpoint against the nulls, paired.

Job 1640's epoch 7 checkpoint has a test score on arm Q (score_cgt_checkpoint_cpu.py,
results/cgt_checkpoint_pred_327csnlk_test.npy), and the six baselines have per-record test
predictions on the same 37,791 records (additive_baselines_025.py). A difference of two
Pearson correlations on the same records is a paired quantity, so its uncertainty comes from
resampling records, not from the two marginal standard errors: 2,000 resamples, percentile
interval, the same rule the 010 report used for the random-split margin.

The prespecified decision the document states is evaluated here and written out:
    D1  the transformer's test Pearson at its validation-selected epoch exceeds B1's
    D2  the paired-bootstrap 95 percent interval of (transformer minus B1) excludes zero
    D3  D1 and D2 hold over three seeds of the configuration     (not run)

Reads
    results/cgt_checkpoint_scores_327csnlk.json, results/cgt_checkpoint_pred_327csnlk_test{,_ids}.npy
    results/additive_baselines_025_pred_{B1_additive_gene,B2_additive_plus_pair,B5_gene_embedding_mlp_s0}_Q.npy
    results/subset_S0_indices.json.gz, results/query_pair_disjoint_splits_025.json.gz
    $DATA_ROOT/.../025-solid-growth/001-full-build/processed/label_df.parquet
Writes
    results/paired_bootstrap_025.json
    notes-tex/025-additive-baselines/tables/t6-armq-paired.tex
    notes-tex/025-additive-baselines/tables/t7-decision.tex

Run from the repository root:
    python experiments/025-solid-growth/scripts/paired_bootstrap_025.py
"""

from __future__ import annotations

import gzip
import json
import os
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from scipy.stats import pearsonr, spearmanr

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
REPO_ROOT = osp.dirname(EXPERIMENT_ROOT)
RESULTS = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results")
TABLES = osp.join(REPO_ROOT, "notes-tex", "025-additive-baselines", "tables")
BUILD = osp.join(DATA_ROOT, "data/torchcell/experiments/025-solid-growth/001-full-build")
RUN = "327csnlk"
N_BOOT = 2_000
RNG = np.random.default_rng(0)
BASELINES = {
    "B1 additive ridge": "additive_baselines_025_pred_B1_additive_gene_Q.npy",
    "B2 additive plus pair": "additive_baselines_025_pred_B2_additive_plus_pair_Q.npy",
    "B5 embedding MLP, seed 0": "additive_baselines_025_pred_B5_gene_embedding_mlp_s0_Q.npy",
}


class PairedGain(BaseModel):
    against: str
    pearson_baseline: float
    pearson_transformer: float
    gain: float
    ci_low: float
    ci_high: float
    residual_corr: float
    prediction_corr: float


class Decision(BaseModel):
    criterion: str
    statement: str
    value: str
    result: str


class Report(BaseModel):
    run: str
    epoch: int
    n_test: int
    n_boot: int
    transformer_test_pearson: float
    transformer_test_spearman: float
    gains: list[PairedGain]
    decisions: list[Decision]


def load_gz(name: str):
    with gzip.open(osp.join(RESULTS, name), "rt") as f:
        return json.load(f)


def paired_bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Percentile interval for r(y, b) - r(y, a), resampling records."""
    n = y.size
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        diffs[i] = pearsonr(y[idx], b[idx])[0] - pearsonr(y[idx], a[idx])[0]
    point = float(pearsonr(y, b)[0] - pearsonr(y, a)[0])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)


def main() -> None:
    scores = json.load(open(osp.join(RESULTS, f"cgt_checkpoint_scores_{RUN}.json")))
    ids = np.load(osp.join(RESULTS, f"cgt_checkpoint_pred_{RUN}_test_ids.npy"))
    cgt = np.load(osp.join(RESULTS, f"cgt_checkpoint_pred_{RUN}_test.npy"))
    subset = np.array(sorted(load_gz("subset_S0_indices.json.gz")), dtype=np.int64)
    test_ids = np.array(sorted(load_gz("query_pair_disjoint_splits_025.json.gz")["splits"]["test"]))
    if not np.array_equal(ids, test_ids):
        raise SystemExit("scored record ids are not the sorted arm Q test part")
    rows = np.searchsorted(subset, ids)
    if not (subset[rows] == ids).all():
        raise SystemExit("a test record id is absent from S0")

    label_df = pd.read_parquet(osp.join(BUILD, "processed", "label_df.parquet"), columns=["index", "gene_interaction"])
    y = label_df.set_index("index").loc[ids, "gene_interaction"].to_numpy(dtype=np.float64)

    r_cgt = float(pearsonr(y, cgt)[0])
    if abs(r_cgt - scores["parts"]["test"]["pearson"]) > 1e-6:
        raise SystemExit(f"recomputed test Pearson {r_cgt} differs from the scores file {scores['parts']['test']['pearson']}")

    gains = []
    for name, fname in BASELINES.items():
        base = np.load(osp.join(RESULTS, fname))[rows]
        point, lo, hi = paired_bootstrap(y, base, cgt)
        gains.append(PairedGain(
            against=name,
            pearson_baseline=float(pearsonr(y, base)[0]),
            pearson_transformer=r_cgt,
            gain=point, ci_low=lo, ci_high=hi,
            residual_corr=float(pearsonr(y - base, y - cgt)[0]),
            prediction_corr=float(pearsonr(base, cgt)[0]),
        ))
        print(f"vs {name:<26s} baseline {gains[-1].pearson_baseline:.4f} gain {point:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    b1 = gains[0]
    decisions = [
        Decision(criterion="D1", statement="transformer test Pearson at its validation-selected epoch exceeds B1's",
                 value=f"{r_cgt:.3f} against {b1.pearson_baseline:.3f}", result="PASS" if r_cgt > b1.pearson_baseline else "FAIL"),
        Decision(criterion="D2", statement="paired-bootstrap 95 percent interval of transformer minus B1 excludes zero",
                 value=f"{b1.gain:+.3f}, [{b1.ci_low:+.3f}, {b1.ci_high:+.3f}]",
                 result="PASS" if (b1.ci_low > 0) else ("FAIL" if b1.ci_high < 0 else "INCONCLUSIVE")),
        Decision(criterion="D3", statement="D1 and D2 hold over three seeds of the configuration",
                 value="one run, one checkpoint", result="NOT RUN"),
    ]
    report = Report(run=RUN, epoch=int(scores["epoch"]), n_test=int(ids.size), n_boot=N_BOOT,
                    transformer_test_pearson=r_cgt, transformer_test_spearman=float(spearmanr(y, cgt)[0]),
                    gains=gains, decisions=decisions)
    with open(osp.join(RESULTS, "paired_bootstrap_025.json"), "w") as f:
        f.write(json.dumps(report.model_dump(), indent=2))

    os.makedirs(TABLES, exist_ok=True)
    head = ("%% GENERATED by experiments/025-solid-growth/scripts/paired_bootstrap_025.py\n"
            "%% SOURCE: results/paired_bootstrap_025.json. Do not edit by hand.\n")
    rows_tex = ["\\begin{tabular}{lrrrrr}", "\\toprule",
                "Against & Baseline $r$ & Gain & 95\\% interval & Residual corr. & Prediction corr. \\\\", "\\midrule"]
    for g in gains:
        rows_tex.append(f"{g.against} & {g.pearson_baseline:.3f} & ${g.gain:+.3f}$ & $[{g.ci_low:+.3f}, {g.ci_high:+.3f}]$ & "
                        f"{g.residual_corr:.3f} & {g.prediction_corr:.3f} \\\\")
    rows_tex += ["\\bottomrule", "\\end{tabular}", ""]
    with open(osp.join(TABLES, "t6-armq-paired.tex"), "w") as f:
        f.write(head + "\n".join(rows_tex))
    dec_tex = ["\\begin{tabular}{l>{\\raggedright\\arraybackslash}p{78mm}>{\\raggedright\\arraybackslash}p{40mm}l}", "\\toprule",
               "Criterion & Statement & Value & Result \\\\", "\\midrule"]
    for d in decisions:
        dec_tex.append(f"{d.criterion} & {d.statement} & {d.value} & \\textbf{{{d.result}}} \\\\")
    dec_tex += ["\\bottomrule", "\\end{tabular}", ""]
    with open(osp.join(TABLES, "t7-decision.tex"), "w") as f:
        f.write(head + "\n".join(dec_tex))
    print("wrote", osp.join(RESULTS, "paired_bootstrap_025.json"), "and tables t6, t7")


if __name__ == "__main__":
    main()
