# experiments/028-gene-essentiality/scripts/target_ensemble_readout.py
# [[experiments.028-gene-essentiality.scripts.target_ensemble_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/target_ensemble_readout

r"""Seed ensembles of the betaxanthin and amino-acid arms, from the dumped predictions.

Round 71 measured that averaging seeds and folds is the largest single gain on the
essentiality endpoint (fold ensemble 0.930 to 0.953). The target-transfer arms dump
their pinned-test predictions per cell, so the same read costs no GPU: for each arm, the
mean prediction over its seeds on every pinned record, scored as the cells are (pinned
Spearman, the top-50 enrichment of high producers for betaxanthin; the mean per-column
Spearman for the amino acids), beside the mean of the single-seed scores. The ensemble
of the two disjoint seed triples (2700 to 2702, 2703 to 2705) gives the three-seed read
twice, which is the depth 027 ran at.

Each dump is resolved THROUGH its result file's ``test.dump_path`` and its ``cell`` is
checked against the result's, as the trainer's dump note requires. Writes
``results/target_ensemble_readout.json``. Run from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/028-gene-essentiality/scripts/target_ensemble_readout.py
"""

from __future__ import annotations

import json
import os
import os.path as osp

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
from scipy.stats import spearmanr

load_dotenv()
RESULTS = osp.join(os.environ["EXPERIMENT_ROOT"], "028-gene-essentiality", "results")
SEEDS = (2700, 2701, 2702, 2703, 2704, 2705)
TRIPLES = ((2700, 2701, 2702), (2703, 2704, 2705))
#: (arm stem, head); a stem takes ``_s<seed>.json``.
ARMS = [
    ("w134_bxctx_h12l1p1_e200", "betaxanthin"),
    ("w134_bxstore_h12l1p1_e200", "betaxanthin"),
    ("w136_bxratio_h12l1p1_e200", "betaxanthin"),
    ("w136_bxstorewd_h12l1p1_e200", "betaxanthin"),
    ("w137_bxdel_h12l1p1_e200", "betaxanthin"),
    ("w137_bxgpr_h12l1p1_e200", "betaxanthin"),
    ("w142_bxgrowth_h12l1p1_e200", "betaxanthin"),
    ("w142_bxprod_h12l1p1_e200", "betaxanthin"),
    ("w139_bxwide_h90l2p1_e200", "betaxanthin"),
    ("w140_bxwidestore_h90l2p1_e200", "betaxanthin"),
    ("w140_bxdelstore_h12l1p1_e200", "betaxanthin"),
    ("w135_aactx_h12l1p1_e200", "mulleder19"),
    ("w135_aastore_h12l1p1_e200", "mulleder19"),
    ("w138_aaprodstat_h12l1p1_e200", "mulleder19"),
    ("w138_aaprodfull_h12l1p1_e200", "mulleder19"),
    ("w141_aadel_h12l1p1_e200", "mulleder19"),
    ("w139_aawide_h90l2p1_e200", "mulleder19"),
]
TOP_K = 50
TOP_FRAC = 0.1


class Score(BaseModel):
    """One scoring of a prediction vector on the pinned records."""

    n: int
    spearman: float
    top50_enrichment: float | None = None
    top50_hits: float | None = None


class ArmEnsemble(BaseModel):
    """One arm's single-seed and ensembled reads."""

    stem: str
    head: str
    seeds: list[int]
    single: list[Score]
    mean_single_spearman: float
    n_constant: int = 0
    triples: list[Score]
    mean_triple_spearman: float | None
    all_seeds: Score | None


class Report(BaseModel):
    """The readout."""

    arms: list[ArmEnsemble]
    note: str


def load_dump(stem: str, seed: int) -> dict | None:
    result_path = osp.join(RESULTS, f"{stem}_s{seed}.json")
    if not osp.exists(result_path):
        return None
    with open(result_path) as fh:
        result = json.load(fh)
    with open(result["test"]["dump_path"]) as fh:
        dump = json.load(fh)
    assert dump["cell"] == result["cell"], (stem, seed)
    return dump


def score_scalar(pred: np.ndarray, true: np.ndarray) -> Score:
    finite = np.isfinite(pred) & np.isfinite(true)
    p, t = pred[finite], true[finite]
    rho = float(spearmanr(p, t).correlation)
    n_high = max(1, int(round(TOP_FRAC * t.size)))
    high = set(np.argsort(-t)[:n_high].tolist())
    top = np.argsort(-p)[:TOP_K].tolist()
    hits = sum(1 for i in top if i in high)
    return Score(
        n=int(t.size),
        spearman=rho,
        top50_enrichment=float((hits / TOP_K) / (n_high / t.size)),
        top50_hits=float(hits),
    )


def score_vector(pred: np.ndarray, true: np.ndarray) -> Score:
    rhos = []
    for j in range(true.shape[1]):
        finite = np.isfinite(pred[:, j]) & np.isfinite(true[:, j])
        if finite.sum() < 3:
            continue
        rhos.append(float(spearmanr(pred[finite, j], true[finite, j]).correlation))
    return Score(n=int(true.shape[0]), spearman=float(np.mean(rhos)))


def arrays(dump: dict, head: str) -> tuple[list[int], np.ndarray, np.ndarray]:
    recs = [r for r in dump["predictions"] if r["pinned"]]
    keys = [int(r["record_index"]) for r in recs]
    if head == "mulleder19":
        pred = np.array([r["pred"] for r in recs], dtype=float)
        true = np.array(
            [[np.nan if x is None else x for x in r["target"]] for r in recs],
            dtype=float,
        )
    else:
        pred = np.array([r["pred"] for r in recs], dtype=float)
        true = np.array(
            [np.nan if r["target"] is None else r["target"] for r in recs], dtype=float
        )
    return keys, pred, true


def ensemble(dumps: dict[int, dict], seeds: tuple[int, ...], head: str) -> Score | None:
    if any(s not in dumps for s in seeds):
        return None
    ref_keys, _, true = arrays(dumps[seeds[0]], head)
    preds = []
    for s in seeds:
        keys, pred, _ = arrays(dumps[s], head)
        assert keys == ref_keys, "pinned records differ between seeds"
        preds.append(pred)
    mean = np.mean(np.stack(preds), axis=0)
    return (
        score_vector(mean, true) if head == "mulleder19" else score_scalar(mean, true)
    )


def main() -> None:
    arms: list[ArmEnsemble] = []
    for stem, head in ARMS:
        dumps = {s: d for s in SEEDS if (d := load_dump(stem, s)) is not None}
        if not dumps:
            print(f"{stem}: no cells yet")
            continue
        single = [ensemble(dumps, (s,), head) for s in sorted(dumps)]
        single = [s for s in single if s is not None]
        triples = [t for tr in TRIPLES if (t := ensemble(dumps, tr, head)) is not None]
        full = ensemble(dumps, SEEDS, head)
        arm = ArmEnsemble(
            stem=stem,
            head=head,
            seeds=sorted(dumps),
            single=single,
            # A seed whose selected epoch is a constant predictor scores NaN; it is
            # counted in n_constant and left out of the mean.
            mean_single_spearman=float(np.nanmean([s.spearman for s in single])),
            n_constant=int(sum(np.isnan(s.spearman) for s in single)),
            triples=triples,
            mean_triple_spearman=(
                float(np.mean([t.spearman for t in triples])) if triples else None
            ),
            all_seeds=full,
        )
        arms.append(arm)
        line = (
            f"{stem}: {len(single)} seeds, single mean {arm.mean_single_spearman:+.4f}"
        )
        if triples:
            line += f"; triples {', '.join(f'{t.spearman:+.4f}' for t in triples)}"
        if full is not None:
            line += f"; all six {full.spearman:+.4f}"
            if full.top50_enrichment is not None:
                line += (
                    f" (top-50 enrichment {full.top50_enrichment:.2f}, single mean "
                    f"{np.mean([s.top50_enrichment for s in single]):.2f})"
                )
        print(line)
    report = Report(
        arms=arms,
        note=(
            "Mean prediction over seeds on the pinned records, scored as the cells are; "
            "every dump resolved through its result file and checked against it."
        ),
    )
    out = osp.join(RESULTS, "target_ensemble_readout.json")
    with open(out, "w") as fh:
        fh.write(report.model_dump_json(indent=2))
    print(out)


if __name__ == "__main__":
    main()
