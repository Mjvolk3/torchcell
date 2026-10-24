# experiments/043-expression-proteome-morphology/scripts/alcf/aurora_record_retry_043.py
# [[experiments.043-expression-proteome-morphology.scripts.alcf.aurora_record_retry_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/alcf/aurora_record_retry_043.py
"""Record one retry of an Aurora cell in its W&B run (summary and config), visibly.

Every restart of a round aurora_2026_10_24 cell after the XPU page fault is written into
the run itself: ``ops/retries`` (count), ``ops/fault_utc`` (comma-separated fault times),
``ops/fault_last_step`` (the last logged optimizer step before the fault) and
``ops/fault_signature``, in the summary and as ``ops_*`` config keys.

    python aurora_record_retry_043.py a1024_lodo_hu_onto_hlgauss_equal_s0 \
        --fault-utc 2026-10-24T06:25:03Z
"""

from __future__ import annotations

import argparse

import wandb

PROJECT = "zhao-group/torchcell_043-expression-proteome-morphology"
SIGNATURE = (
    "Segmentation fault from GPU ... type: 0 (NotPresent), level: 1 (PDE), access: 1 "
    "(Write), banned: 1; Abort at intel-compute-runtime-25.18.33578.77 drm_neo.cpp:288"
)


def main() -> None:
    """Increment the retry count and append the fault time."""
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--fault-utc", required=True)
    a = ap.parse_args()
    run = wandb.Api().run(f"{PROJECT}/{a.run_id}")
    n = int(run.summary.get("ops/retries", 0)) + 1
    prev = str(run.summary.get("ops/fault_utc", ""))
    times = ",".join(t for t in (prev, a.fault_utc) if t)
    last_step = int(run.summary.get("_step", 0))
    for k, v in (
        ("ops/retries", n),
        ("ops/fault_utc", times),
        ("ops/fault_last_step", last_step),
        ("ops/fault_signature", SIGNATURE),
    ):
        run.summary[k] = v
        run.config[k.replace("/", "_")] = v
    run.summary.update()
    run.update()
    print(f"{a.run_id}: retries {n}, faults {times}, last logged step {last_step}")


if __name__ == "__main__":
    main()
