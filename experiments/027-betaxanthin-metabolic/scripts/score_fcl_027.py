# experiments/027-betaxanthin-metabolic/scripts/score_fcl_027.py
# [[experiments.027-betaxanthin-metabolic.cpu-baselines-2026-09-14]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/score_fcl_027
"""Score the 027 prediction dumps under Merzbacher's own three-class protocol.

020's ``evaluate_merzbacher_head_to_head.py`` IS the protocol, fixed before we had a model,
so nothing in it is edited here. It simply cannot read a 027 dump, for four measured
reasons:

1. ``load_dumps`` keeps a file only if ``"betaxanthin" in payload["predictions"]``. The 027
   ``predictions`` is a flat list, so the test is False on every file and the script exits.
2. ``gene_predictions`` reads ``dump["predictions"]["betaxanthin"]`` and takes
   ``rec["pred"][0]``; in 027 ``pred`` is a scalar on the betaxanthin head and a 19-vector
   on the ``mulleder19`` head.
3. ``setting_label`` raises unless a ``wandb_tags`` entry matches ``^s\\d+_``, and no 027
   dump carries a ``wandb_tags`` key at all. The label has to come from ``cell.arm`` plus
   ``cell.epochs``, and the seed from ``cell.seed``.
4. 027 records carry their own ``pinned`` boolean, which the 020 path obtains instead by
   intersecting with the released labels.

``normalize_dump`` is the whole fix: it rewrites a 027 payload into the shape 020's scoring
functions already accept, and is the identity on a 020 payload.

THE CONTROL RUNS FIRST AND IS NOT OPTIONAL. The 020 dumps are re-scored through this
adapter and every number in ``020-cachera-betaxanthin/results/merzbacher_head_to_head.json``
must reproduce exactly. A 027 number from a scorer that failed its own control is not
reported.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \\
        experiments/027-betaxanthin-metabolic/scripts/score_fcl_027.py
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
SCORER_DIR = osp.join(EXPERIMENT_ROOT, "020-cachera-betaxanthin", "scripts")
sys.path.insert(0, SCORER_DIR)

from build_merzbacher_split import (  # noqa: E402
    load_merzbacher_split,
    read_cachera_raw,
)
from evaluate_merzbacher_head_to_head import (  # noqa: E402
    AGG_KEYS,
    THEIR_BEST,
    aggregate,
    class_proportions,
    evaluate_one,
    load_dumps,
    scale_bins,
)

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results")
BASELINE_PATH = osp.join(
    EXPERIMENT_ROOT, "020-cachera-betaxanthin", "results", "merzbacher_baseline_analysis.json"
)
SPLIT_PATH = osp.join(
    EXPERIMENT_ROOT, "020-cachera-betaxanthin", "results", "merzbacher_nested_split.json"
)
CONTROL_DUMPS = osp.join(DATA_ROOT, "test-predictions")
CONTROL_RESULT = osp.join(
    EXPERIMENT_ROOT, "020-cachera-betaxanthin", "results", "merzbacher_head_to_head.json"
)
DEFAULT_DUMPS = osp.join(DATA_ROOT, "test-predictions-027-wave7")
#: 639 of Merzbacher's 639 test ORFs resolve to a record in the 027 build and are flagged
#: `pinned`; 629 of those carry a finite betaxanthin measurement.
EXPECTED_PINNED = 639


def normalize_dump(payload: dict[str, Any], dump_format: str) -> dict[str, Any] | None:
    """Put a dump into the shape 020's scoring functions read. None means "not scorable".

    The 020 format passes through untouched. The 027 format is rewritten: the flat
    ``predictions`` list becomes ``{"betaxanthin": [...]}`` with ``pred`` wrapped in a list,
    the setting label is synthesized from the cell's arm and epoch budget so
    ``setting_label``'s ``^s\\d+_`` anchor matches, and only the ``pinned`` records survive.
    """
    if dump_format == "020":
        return payload
    if payload.get("head") != "betaxanthin":
        return None
    cell = payload["cell"]
    records = [
        {
            "record_index": rec["record_index"],
            "genes": rec["genes"],
            "pred": [float(rec["pred"])],
        }
        for rec in payload["predictions"]
        if rec["pinned"]
    ]
    if len(records) != EXPECTED_PINNED:
        raise ValueError(
            f"{payload.get('_file')} carries {len(records)} pinned records, expected "
            f"{EXPECTED_PINNED}; the pin resolved against a different build"
        )
    return {
        "predictions": {"betaxanthin": records},
        # The label the by-setting table groups on. 027 has no grid index, so `s00_` is a
        # constant prefix that satisfies the anchor; the arm and the epoch budget are what
        # actually distinguish the cells.
        "wandb_tags": [f"s00_{cell['arm']}_e{cell['epochs']}"],
        "seed": cell["seed"],
        # `dist` reached the cell only from wave 7 on; an older dump simply has no entry,
        # and the 020 scorer records it as metadata rather than reading it.
        "dist": cell.get("dist"),
        "_file": payload["_file"],
        "_cell": cell,
    }


def read_dumps(path: str, dump_format: str) -> list[dict[str, Any]]:
    """Every scorable dump under ``path``, normalized."""
    if dump_format == "020":
        return [d for d in load_dumps(path) if normalize_dump(d, "020") is not None]
    files = sorted(f for f in os.listdir(path) if f.endswith(".json"))
    out: list[dict[str, Any]] = []
    for name in files:
        with open(osp.join(path, name)) as fh:
            payload = json.load(fh)
        payload["_file"] = osp.join(path, name)
        normalized = normalize_dump(payload, dump_format)
        if normalized is not None:
            out.append(normalized)
    return out


def build_payload(dumps: list[dict[str, Any]]) -> dict[str, Any]:
    """020's protocol, applied to whatever dumps are handed in.

    Every step below is lifted from ``evaluate_merzbacher_head_to_head.main`` unchanged: the
    ground truth is their released label, the bin scale is fitted on the train pool only,
    and the rank-matched class shares come from that same pool.
    """
    _, their_val = load_merzbacher_split()
    truth = {str(r["knockout"]): int(r["label"]) for _, r in their_val.iterrows()}
    counts = {c: sum(1 for v in truth.values() if v == c) for c in (0, 1, 2)}
    assert counts == {0: 109, 1: 431, 2: 100}, f"their released labels changed: {counts}"

    raw, _ = read_cachera_raw()
    with open(SPLIT_PATH) as fh:
        split = json.load(fh)
    pool = [raw[g] for g in split["split"]["train_val_pool"] if g in raw]
    lo, hi = float(np.nanmin(pool)), float(np.nanmax(pool))
    pool_props = class_proportions(scale_bins(np.asarray(pool, dtype=float), lo, hi))

    with open(BASELINE_PATH) as fh:
        baseline = json.load(fh)

    rows = [evaluate_one(d, truth, lo, hi, raw, pool_props) for d in dumps]
    by_setting: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_setting.setdefault(row["setting"], []).append(row)
    return {
        "protocol": "[[experiments.020-cachera-betaxanthin.merzbacher-comparison]]",
        "ground_truth": "Merzbacher released labels (yeast_production_validation_split.csv)",
        "bin_scale_fitted_on": {
            "split": "train_val_pool",
            "n_genes": len(pool),
            "min": lo,
            "max": hi,
        },
        "their_baseline_gene_level": baseline["gene_level"].get(THEIR_BEST),
        "their_best_model": THEIR_BEST,
        "their_mcc": baseline["per_fold_metrics_their_own"].get(THEIR_BEST, {}).get("mcc"),
        "pooled": aggregate(rows, AGG_KEYS),
        "by_setting": {k: aggregate(v, AGG_KEYS) for k, v in sorted(by_setting.items())},
        "per_run": rows,
    }


def run_control() -> dict[str, Any]:
    """Re-score the 020 dumps through this adapter and require the stored numbers back.

    Scored on EXACTLY the files the stored result names. ``$DATA_ROOT/test-predictions``
    has since gained four gilahyper dumps that carry a betaxanthin head and no grid setting
    tag, which ``setting_label`` refuses by design; they postdate the stored result, so
    including them would compare two different run sets rather than test the adapter.
    """
    with open(CONTROL_RESULT) as fh:
        stored = json.load(fh)
    named = {row["file"] for row in stored["per_run"]}
    available = {f for f in os.listdir(CONTROL_DUMPS) if f.endswith(".json")}
    dumps = [
        d
        for d in read_dumps(CONTROL_DUMPS, "020")
        if osp.basename(d["_file"]) in named
    ]
    assert len(dumps) == len(named), (
        f"{len(named) - len(dumps)} of the files the stored result names are missing from "
        f"{CONTROL_DUMPS}"
    )
    payload = build_payload(dumps)
    assert len(payload["per_run"]) == len(stored["per_run"]), (
        f"control scored {len(payload['per_run'])} runs, stored file has "
        f"{len(stored['per_run'])}"
    )
    mismatches: list[str] = []
    for new, old in zip(
        sorted(payload["per_run"], key=lambda r: r["file"]),
        sorted(stored["per_run"], key=lambda r: r["file"]),
        strict=True,
    ):
        assert new["file"] == old["file"]
        for key in AGG_KEYS + ("n_scored", "n_genes", "setting"):
            if new.get(key) != old.get(key):
                mismatches.append(f"{new['file']}:{key} {old.get(key)} -> {new.get(key)}")
    for key in AGG_KEYS:
        new_mean = payload["pooled"].get(key, {}).get("mean")
        old_mean = stored["pooled"].get(key, {}).get("mean")
        if new_mean != old_mean:
            mismatches.append(f"pooled:{key} {old_mean} -> {new_mean}")
    if mismatches:
        raise ValueError(
            "the adapter did not reproduce the 020 head-to-head exactly; refusing to "
            f"report any 027 number. {len(mismatches)} mismatches: {mismatches[:10]}"
        )
    print(
        f"control PASSED: {len(payload['per_run'])} 020 dumps reproduce "
        f"{CONTROL_RESULT} exactly on every aggregated metric"
    )
    return {
        "control_dumps": CONTROL_DUMPS,
        "control_result": CONTROL_RESULT,
        "n_runs": len(payload["per_run"]),
        "n_dump_files_present": len(available),
        "n_dump_files_named_by_stored_result": len(named),
        "reproduced_exactly": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", default=DEFAULT_DUMPS)
    ap.add_argument("--dump-format", default="027", choices=["020", "027"])
    ap.add_argument("--out", default=osp.join(RESULTS_DIR, "fcl_head_to_head_027.json"))
    ap.add_argument(
        "--skip-control",
        action="store_true",
        help="NEVER use for a reported run; present only so the control itself can be run",
    )
    args = ap.parse_args()

    control = None if args.skip_control else run_control()

    dumps = read_dumps(args.dumps, args.dump_format)
    if not dumps:
        raise SystemExit(f"no betaxanthin prediction dumps under {args.dumps}")
    payload = build_payload(dumps)
    payload["generated_by"] = (
        "experiments/027-betaxanthin-metabolic/scripts/score_fcl_027.py"
    )
    payload["dumps"] = args.dumps
    payload["dump_format"] = args.dump_format
    payload["control"] = control
    payload["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    their = payload["their_baseline_gene_level"]
    print(
        f"\n{len(payload['per_run'])} runs, {payload['per_run'][0]['n_scored']} genes scored"
    )
    header = (
        f"\n{'arm':<44s} {'n':>2s} {'MCC':>14s} {'acc':>14s} {'hi-rec':>14s} "
        f"{'frac-med':>14s} {'top50':>14s}"
    )
    print(header)
    for name, agg in sorted(payload["by_setting"].items()):

        def cell(key: str) -> str:
            a = agg.get(key)
            if not a:
                return "--"
            sem = f"+-{a['sem']:.3f}" if a["sem"] is not None else ""
            return f"{a['mean']:.3f}{sem}"

        print(
            f"{name:<44s} {agg['n_runs']:>2d} {cell('mcc'):>14s} {cell('accuracy'):>14s} "
            f"{cell('high_producer_recall'):>14s} "
            f"{cell('fraction_predicted_medium'):>14s} "
            f"{cell('topk_50_high_precision'):>14s}"
        )
    print(
        f"\nMERZBACHER {THEIR_BEST}: MCC {payload['their_mcc']:.3f} | accuracy "
        f"{their['gene_level_accuracy']:.3f} | high-producer recall "
        f"{their['high_producer_recall']:.3f} | fraction called medium "
        f"{their['fraction_predicted_medium']:.3f}"
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
