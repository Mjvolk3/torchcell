# experiments/028-gene-essentiality/scripts/verify_doc_numbers.py
# [[experiments.028-gene-essentiality.scripts.verify_doc_numbers]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/verify_doc_numbers

"""Every number typed into the document's prose, checked against the file it came from.

Tables and figures are written by scripts, so they cannot drift. Prose numbers are typed
by hand, and ``make check`` validates style, citations, provenance and float sizing but
never whether a number in a sentence still matches its source. A number that was right
when written and wrong after a rerun survives every gate the build has.

Each check names the claim, the quantity as the document states it, and how to recompute
it. Add one whenever a number is written into prose; a claim with no check here is a
claim nobody can rebuild. Exits non-zero on any mismatch.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        python experiments/028-gene-essentiality/scripts/verify_doc_numbers.py
"""

from __future__ import annotations

import glob
import json
import os
import os.path as osp
import sys
from typing import Callable

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")


class Check(BaseModel):
    """One number the prose states, and the recomputation that has to agree with it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    #: Where the number appears, as a section label plus the claim in a few words.
    claim: str
    #: The value as the document prints it.
    stated: float
    #: Recomputes the value from ``results/``.
    recompute: Callable[[], float]
    #: Allowed difference. Prose rounds, so this is the rounding step, not a tolerance
    #: for disagreement: 0.0006 for a three-decimal number, 0.5 for an integer.
    tolerance: float = 0.0006


def seed_mean(*patterns: str) -> Callable[[], float]:
    """The mean held-out AUROC over every cell matching the patterns."""

    def run() -> float:
        values = [
            float(json.load(open(p))["test"]["pinned"]["auroc"])
            for pat in patterns
            for p in sorted(glob.glob(osp.join(RESULTS, pat)))
        ]
        assert values, f"no cells matched {patterns}"
        return float(np.mean(values))

    return run


def field(name: str, *path: str | int) -> Callable[[], float]:
    """A scalar reached by walking keys and indices from a result file's root."""

    def run() -> float:
        node = json.load(open(osp.join(RESULTS, name)))
        for key in path:
            node = node[key]
        return float(node)

    return run


def calibration_growth(tier: str, scale: float) -> Callable[[], float]:
    """Median growth at one capacity scale under one turnover tier."""

    def run() -> float:
        rows = json.load(open(osp.join(RESULTS, f"capacity_calibration_{tier}.json")))["rows"]
        return float(next(r["growth_median"] for r in rows if r["capacity_scale"] == scale))

    return run


def store_stat(store_contains: str, key: str) -> Callable[[], float]:
    """One statistic of one store from the discriminability read."""

    def run() -> float:
        stores = json.load(open(osp.join(RESULTS, "store_discriminability.json")))["stores"]
        return float(next(s for s in stores if store_contains in s["store"])[key])

    return run


CHECKS: list[Check] = [
    # sec:store-discriminability -- the stores and the two cuts.
    Check(claim="store-discriminability: vertex store", stated=0.840,
          recompute=seed_mean("w52_pathlogsimplex_h12l1p1_e200_s*.json")),
    Check(claim="store-discriminability: clamp columns cut", stated=0.888,
          recompute=seed_mean("w84_pathlogsimplexdeclamp_h12l1p1_e200_s*.json",
                              "w85_pathlogsimplexdeclamp_h12l1p1_e200_s*.json")),
    Check(claim="store-discriminability: non-clamp cut (control)", stated=0.846,
          recompute=seed_mean("w86_pathlogsimplextopvar_h12l1p1_e200_s*.json")),
    Check(claim="store-discriminability: pFBA store", stated=0.908,
          recompute=seed_mean("w49_pathfbapfbalogfree_h12l1p1_e200_s*.json")),
    Check(claim="store-discriminability: dose, 25 columns", stated=0.874,
          recompute=seed_mean("w87_pathlogsimplexdeclamp25_h12l1p1_e200_s*.json")),
    Check(claim="store-discriminability: dose, 100 columns", stated=0.896,
          recompute=seed_mean("w87_pathlogsimplexdeclamp100_h12l1p1_e200_s*.json")),
    Check(claim="store-discriminability: cut store fold ensemble", stated=0.901,
          recompute=field("w90_pathlogsimplexdeclampcv_ensemble.json", "ensemble_auroc")),
    Check(claim="store-discriminability: vertex clamp columns in the widest ten",
          stated=10, tolerance=0.5,
          recompute=store_stat("simplex", "n_clamp_columns_in_top10")),
    Check(claim="store-discriminability: vertex columns ranking above 0.7 alone",
          stated=298, tolerance=0.5,
          recompute=store_stat("simplex", "n_columns_auroc_above_0_7")),
    Check(claim="store-discriminability: columns the clamp rule cuts from the vertex store",
          stated=429, tolerance=0.5,
          recompute=field("declamp_store.json", "stores", 0, "n_columns_dropped")),
    Check(claim="store-discriminability: columns the same rule cuts from pFBA",
          stated=1, tolerance=0.5,
          recompute=field("declamp_store.json", "stores", 1, "n_columns_dropped")),
    # sec:kcat-tier -- the capacity box and the three forced constructions.
    Check(claim="kcat-tier: measured median nominal ceiling", stated=34.6, tolerance=0.05,
          recompute=field("capacity_calibration_measured.json", "parameters",
                          "nominal_ceiling_quantiles", "0.5")),
    Check(claim="kcat-tier: predicted median nominal ceiling", stated=16.5, tolerance=0.05,
          recompute=field("capacity_calibration_predicted.json", "parameters",
                          "nominal_ceiling_quantiles", "0.5")),
    Check(claim="kcat-tier: measured growth at scale 100", stated=0.156,
          recompute=calibration_growth("measured", 100.0)),
    Check(claim="kcat-tier: predicted growth at scale 100", stated=0.103,
          recompute=calibration_growth("predicted", 100.0)),
    Check(claim="kcat-tier: measured growth at scale 1,000", stated=0.532,
          recompute=calibration_growth("measured", 1000.0)),
    Check(claim="kcat-tier: predicted growth at scale 1,000", stated=0.203,
          recompute=calibration_growth("predicted", 1000.0)),
    Check(claim="kcat-tier: hard forced column, twelve seeds", stated=0.699,
          recompute=seed_mean("w30_lpcapbio100_h12l1p1_e200_s*.json",
                              "w88_lpcapbio100_h12l1p1_e200_s*.json",
                              "w89_lpcapbio100_h12l1p1_e200_s*.json")),
    Check(claim="kcat-tier: forced column on the looser tier", stated=0.707,
          recompute=seed_mean("w83_lpcapbio100meas_h12l1p1_e200_s*.json",
                              "w88_lpcapbio100meas_h12l1p1_e200_s*.json")),
    Check(claim="kcat-tier: forced column on the elastic program", stated=0.715,
          recompute=seed_mean("w32_lpelbio100_h12l1p1_e200_s*.json",
                              "w89_lpelbio100_h12l1p1_e200_s*.json")),
    # sec:store-scale -- what sizes the store over the solid-growth build.
    Check(claim="store-scale: distinct (subset, medium) keys", stated=469474, tolerance=0.5,
          recompute=field("store_key_count.json", "n_distinct_keys")),
    Check(claim="store-scale: records in the build", stated=13525071, tolerance=0.5,
          recompute=field("store_key_count.json", "n_records")),
    Check(claim="store-scale: build wall time, minutes", stated=71, tolerance=0.6,
          recompute=lambda: field("solid_growth_store_SC_ipm.json", "seconds")() / 60.0),
]


def main() -> None:
    failed = 0
    for check in CHECKS:
        got = check.recompute()
        agrees = abs(got - check.stated) < check.tolerance
        failed += 0 if agrees else 1
        mark = "ok  " if agrees else "WRONG"
        print(f"{mark} {check.claim:58s} document {check.stated:,g}  files {got:,.4f}")
    print(f"\n{len(CHECKS)} checks, {failed} disagreeing with the document")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
