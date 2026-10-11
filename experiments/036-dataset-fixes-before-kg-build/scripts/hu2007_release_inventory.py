# experiments/036-dataset-fixes-before-kg-build/scripts/hu2007_release_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.hu2007_release_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/hu2007_release_inventory.py
"""Measure the Hu 2007 GSE4654 release and its agreement with Kemmeren 2014.

Every number in the 2026.10.11 section of ``[[torchcell.datasets.scerevisiae.hu2007]]``
that is not a verifier row comes from this script. It reads only sha256-pinned bytes
(the GSE4654 family SOFT and the Supplementary Table 1 text layer from the raw mirror)
and the two dev stores, read only: ``microarray_score_hu2007`` and
``microarray_kemmeren2014``.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/hu2007_release_inventory.json``.

    PYTHONPATH=<worktree> python experiments/036-dataset-fixes-before-kg-build/scripts/hu2007_release_inventory.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from scipy.stats import norm

from torchcell.data.experiment_dataset import verify_sha256
from torchcell.datasets.scerevisiae import hu2007 as h
from torchcell.verification.expression_score import (
    deletion_profiles,
    matched_profile_correlations,
)
from torchcell.verification.runners import load_records

RESULTS = Path(__file__).resolve().parent.parent / "results"


def release_inventory(data_root: str) -> dict[str, Any]:
    """Counts of the released values, the null spots and the p-value rule."""
    mirror = h.raw_mirror_dir(data_root)
    verify_sha256(mirror / h.SOFT.relpath, h.SOFT.sha256)
    verify_sha256(mirror / h.SI_TABLE1_TEXT.relpath, h.SI_TABLE1_TEXT.sha256)
    release = h.read_soft(mirror / h.SOFT.relpath)
    table = h.read_table1(mirror / h.SI_TABLE1_TEXT.relpath)
    worst = max(
        abs(2.0 * float(norm.sf(abs(x))) - p)
        for sample in release.samples
        for x, p in sample.values.values()
    )
    teto7 = {row.orf for row in table if row.condition == h.CONDITION_TETO7}
    own_teto7 = {
        sample.orf: sample.values.get(sample.orf, (None, None))[0]
        for sample in release.samples
        if sample.orf in teto7
    }
    return {
        "n_samples": len(release.samples),
        "n_elements": len(release.platform_ids),
        "n_released_scores": sum(len(s.values) for s in release.samples),
        "n_null_both": sum(s.null_both for s in release.samples),
        "n_null_score_p_one": sum(s.null_score_p_one for s in release.samples),
        "max_abs_p_minus_two_sided_normal": worst,
        "table1_rows": len(table),
        "table1_conditions": dict(Counter(row.condition for row in table)),
        "teto7_own_element_score": own_teto7,
    }


def kemmeren_cross_check(data_root: str) -> dict[str, Any]:
    """Matched vs mismatched per-strain Pearson r against the Kemmeren 2014 store."""
    hu = load_records(osp.join(data_root, h.DATASET_ROOT))
    kemmeren = load_records(osp.join(data_root, h.KEMMEREN_ROOT))
    matched, mismatched, n_genes = matched_profile_correlations(
        deletion_profiles(hu, "expression_score"),
        deletion_profiles(kemmeren, "expression_log2_ratio"),
    )
    r = np.array(list(matched.values()))
    other = np.array(mismatched)
    p95, p99 = (float(q) for q in np.percentile(other, [95, 99]))
    return {
        "n_shared_strains": int(r.size),
        "n_genes": n_genes,
        "matched_r_quantiles": {
            q: float(v)
            for q, v in zip(
                ("q05", "q25", "median", "q75", "q95"),
                np.percentile(r, [5, 25, 50, 75, 95]),
                strict=True,
            )
        },
        "mismatched_r_median": float(np.median(other)),
        "mismatched_r_p95": p95,
        "mismatched_r_p99": p99,
        "n_matched_above_mismatched_median": int((r > np.median(other)).sum()),
        "n_matched_above_mismatched_p95": int((r > p95).sum()),
        "n_matched_above_mismatched_p99": int((r > p99).sum()),
        "n_matched_negative": int((r < 0).sum()),
        "top_matched": sorted(matched.items(), key=lambda item: -item[1])[:15],
    }


def main() -> None:
    """Write the inventory JSON beside the experiment's other results."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    out = {
        "release": release_inventory(data_root),
        "kemmeren2014": kemmeren_cross_check(data_root),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "hu2007_release_inventory.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
