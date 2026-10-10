# experiments/036-dataset-fixes-before-kg-build/scripts/balakrishnan2022_expression_duplication.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.balakrishnan2022_expression_duplication]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/balakrishnan2022_expression_duplication

"""Is any Balakrishnan 2022 profile already served by an E. coli expression store?

Duplication by CONTENT, measured on the dev-tree LMDBs: every Balakrishnan record
(mRNA number fractions) against every record of the three served E. coli RNA-seq
stores (Caglar 2017, PRECISE-1K and Public K-12 of Lamoureux 2023). A TPM profile is
turned into a number fraction over the shared b-numbers by dividing by its sum there
(TPM is a transcript fraction x 1e6 before restriction), and so is the Balakrishnan
profile, so the two are compared on one scale. Reported per store: shared genes, the
number of pairs whose renormalized profiles agree to 1e-6 relative in every shared gene
(a re-served profile), and the highest log10 Pearson r over all pairs, beside
Balakrishnan's own within-condition replicate r as the yardstick.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/balakrishnan2022_expression_duplication.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
from typing import Any

import numpy as np
import numpy.typing as npt
from dotenv import load_dotenv

load_dotenv()

from torchcell.verification.runners import load_records  # noqa: E402

DATA_ROOT = os.environ["DATA_ROOT"]
RESULTS = osp.join(
    os.environ["EXPERIMENT_ROOT"], "036-dataset-fixes-before-kg-build", "results"
)
OUT = osp.join(RESULTS, "balakrishnan2022_expression_duplication.json")
BALAKRISHNAN = "mrna_fraction_balakrishnan2022"
SERVED = (
    "rnaseq_caglar2017",
    "rnaseq_lamoureux2023",
    "rnaseq_public_k12_lamoureux2023",
)
EXACT_RTOL = 1e-6
REPLICATE_PAIRS = (("c5", "c0_1"), ("r0", "r0_1"), ("a2", "a2_1"), ("c3", "c3_1"))


def _profiles(name: str, key: str) -> list[dict[str, float]]:
    root = osp.join(DATA_ROOT, "data", "torchcell", name)
    return [
        {k: float(v) for k, v in rec["experiment"]["phenotype"][key].items()}
        for rec in load_records(root)
    ]


def _matrix(
    profiles: list[dict[str, float]], genes: list[str]
) -> npt.NDArray[np.float64]:
    """Rows renormalized to sum 1 over ``genes``."""
    m = np.array([[p[g] for g in genes] for p in profiles], dtype=np.float64)
    out: npt.NDArray[np.float64] = m / m.sum(axis=1, keepdims=True)
    return out


def _log_pearson(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    mask = (a > 0) & (b > 0)
    return float(np.corrcoef(np.log10(a[mask]), np.log10(b[mask]))[0, 1])


def main() -> dict[str, Any]:
    """Measure and write the duplication verdict."""
    bala = _profiles(BALAKRISHNAN, "mrna_number_fraction")
    root = osp.join(DATA_ROOT, "data", "torchcell", BALAKRISHNAN, "preprocess")
    import pandas as pd

    written = list(pd.read_csv(osp.join(root, "samples.csv"))["sample"])
    # The LMDB keys are the write index as a string, read back in byte order.
    order = sorted(str(i) for i in range(len(written)))
    samples = [written[int(key)] for key in order]
    by_sample = dict(zip(samples, bala, strict=True))
    genes_all = sorted(bala[0])
    whole = _matrix(bala, genes_all)
    replicate_r = {
        f"{a}~{b}": _log_pearson(whole[samples.index(a)], whole[samples.index(b)])
        for a, b in REPLICATE_PAIRS
    }
    out: dict[str, Any] = {
        "balakrishnan_records": len(bala),
        "balakrishnan_genes": len(genes_all),
        "balakrishnan_replicate_log10_pearson": replicate_r,
        "exact_rtol": EXACT_RTOL,
        "stores": {},
    }
    for name in SERVED:
        served = _profiles(name, "expression_tpm")
        shared: set[str] = set(genes_all)
        for profile in served:
            shared &= set(profile)
        genes = sorted(shared)
        if not genes:
            out["stores"][name] = {
                "records": len(served),
                "shared_genes": 0,
                "verdict": "no shared locus: the store is keyed in another strain's "
                "namespace, so no record can re-serve a Balakrishnan profile",
            }
            continue
        left = _matrix(list(by_sample.values()), genes)
        right = _matrix(served, genes)
        exact = 0
        best = (-1.0, "", -1)
        for i, row in enumerate(left):
            for j, other in enumerate(right):
                if np.allclose(row, other, rtol=EXACT_RTOL, atol=0.0):
                    exact += 1
                r = _log_pearson(row, other)
                if r > best[0]:
                    best = (r, samples[i], j)
        out["stores"][name] = {
            "records": len(served),
            "shared_genes": len(genes),
            "pairs": len(bala) * len(served),
            "exact_profile_matches": exact,
            "max_log10_pearson": best[0],
            "max_pair": {
                "balakrishnan_sample": best[1],
                "served_record_index": best[2],
            },
        }
    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(out, handle, indent=2)
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
