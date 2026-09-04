# experiments/027-betaxanthin-metabolic/scripts/grid_runner.py
# [[experiments.027-betaxanthin-metabolic.scripts.grid_runner]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/grid_runner.py

r"""Sharded work ledger for the 027 arms x seeds grid.

WHAT OPTUNA IS DOING HERE, AND WHAT IT IS NOT
----------------------------------------------
Every cell is ``enqueue_trial``-ed, so the sampler is never consulted for anything that
matters. Optuna is used for concurrency-safe claiming, durable per-trial attributes and
resume-after-kill, and for nothing else. There is no search: the design is a fully crossed
6 arms x N seeds factorial and the hyperparameters are fixed in ``conf/base.yaml``.

SHARDING, AND WHY A SHARED QUEUE IS NOT AN OPTION
--------------------------------------------------
Optuna's WAITING-trial pop RACES on SQLite across processes. Measured on IGB job 2332400 with
six workers against one study file: w0 and w1 both claimed trial 0, w2 and w3 both claimed
trial 1 -- six workers running THREE distinct cells. The same race fired on the Delta grids,
where ``s03_L2_maskon_lr0.001_energy`` ran four times and ``bx_ctrl s01`` twice, silently
halving coverage while every log looked healthy.

``experiments/019-simb-multimodal/scripts/delta_grid_common.sh`` STILL has this bug: it sets
``OPTUNA_WORKER_ID`` but points every worker at one ``OPTUNA_STORAGE`` file and never exports
``GRID_SHARD_COUNT``. Only the IGB launcher was fixed. 027's own Delta launcher
(``delta_bxfx_common.sh``) carries the fix, which is the main reason it does not source the
019 include.

Here each worker owns a DISJOINT, DETERMINISTIC slice in its OWN study in its OWN SQLite
file, so there is no shared object to race. The cost is no work stealing: a worker that
finishes early idles. That is the right trade because every cell in this grid costs the same
-- same architecture size, same epoch budget, same split.

THE SHARD RULE IS BY WHOLE SEED
--------------------------------
With A arms and W workers there are G = W // A groups; group ``g = WORKER_ID // A`` takes
every G-th ROUND, and within a group worker ``w`` runs arm ``w % A``. So at any instant G
COMPLETE seeds are in flight and they finish together, and a wall-clock kill leaves a
BALANCED prefix -- every arm at the same seed count, which is the only thing a paired
contrast can be computed on.

A flat ``cell_index % W`` stride would also cover every cell exactly once, but it would give
each worker one arm across scattered seeds, so a kill would leave controls at seeds the
treatments never reached: paired differences with no partner, which is not a smaller result
but an unreadable one.
"""

import json
import os
import os.path as osp
import sys
import time
from typing import Any, Literal, cast

import optuna
import wandb

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from train_bx import (  # noqa: E402
    EXP_DIR,
    RESULTS_DIR,
    arm_names,
    build_cell,
    load_config,
    resolve_pinned_test,
    run_cell,
)

# From 026 directly rather than re-exported through `train_bx`: importing `train_bx` above is
# what puts 026's scripts dir on `sys.path`, so this line must follow it.
from train_flux import build_dataset  # noqa: E402

#: wandb's own accepted values. Read from the environment and CHECKED, so a typo in a launcher
#: fails here instead of silently falling through to whatever wandb does with an unknown mode.
WandbMode = Literal["online", "offline", "disabled"]
WANDB_MODE = os.getenv("WANDB_MODE", "online")
assert WANDB_MODE in ("online", "offline", "disabled"), (
    f"WANDB_MODE={WANDB_MODE!r} is not one of online, offline, disabled."
)

WORKER_ID = int(os.getenv("OPTUNA_WORKER_ID", "0"))
SHARD_COUNT = int(os.getenv("GRID_SHARD_COUNT", "0"))
STORAGE = os.environ["OPTUNA_STORAGE"]
#: One round is one SEED across every arm. 24 by default; see README.md for the arithmetic
#: that sets it (measured paired SD 0.0554 -> a 2-sigma MDE of 0.032 at n=24, against 0.090
#: at n=3).
ROUNDS = int(os.getenv("GRID_ROUNDS", "24"))
#: Seeds are contiguous from a base that no earlier experiment used, so a 027 seed is never
#: confused with a 026 seed (101-110, 301-312) in a merged analysis.
SEED_BASE = int(os.getenv("GRID_SEED_BASE", "2700"))
STUDY_NAME = os.getenv(
    "OPTUNA_STUDY_NAME",
    f"bxfx_grid_000_w{WORKER_ID}" if SHARD_COUNT else "bxfx_grid_000",
)
#: Unix epoch after which no NEW cell is claimed. The slurm wall clock, minus teardown.
DEADLINE_EPOCH = int(os.getenv("GRID_DEADLINE_EPOCH", "0"))
#: Do not claim a cell with less than this much time left. A truncated run has no selected
#: epoch worth reporting, so the slot is better left idle than filled with an unreadable run.
#: 4.5 h is 200 epochs at the GilaHyper-measured 70.0 s/epoch for a flux arm, with the 1.44x
#: two-per-GPU penalty, plus margin. It governs only the FIRST claim; after one cell the guard
#: uses the longest duration actually observed, which is what matters if Delta runs slower.
GRID_MIN_TRIAL_S = int(os.getenv("GRID_MIN_TRIAL_S", "16200"))
#: Reserved after the last cell for the test pass, the per-gene dump and wandb teardown.
GRID_TEARDOWN_S = int(os.getenv("GRID_TEARDOWN_S", "600"))
#: Epoch override, for a wiring smoke test only. Unset in the real grid, where the budget is
#: `conf/base.yaml`'s 200 -- a run at a different epoch budget is a different experiment,
#: because the selection search widens with it.
GRID_EPOCHS = os.getenv("GRID_EPOCHS")

ARMS_ORDERED = arm_names()

if SHARD_COUNT:
    # Sharding must divide evenly into the arm count, or `_owns_cell` leaves cells owned by
    # nobody -- which looks exactly like a healthy short run in a slurm log.
    if SHARD_COUNT % len(ARMS_ORDERED) != 0:
        raise SystemExit(
            f"GRID_SHARD_COUNT={SHARD_COUNT} must be a multiple of the arm count "
            f"({len(ARMS_ORDERED)}: {ARMS_ORDERED}), so each worker owns exactly one arm."
        )
    if WORKER_ID >= SHARD_COUNT:
        raise SystemExit(
            f"OPTUNA_WORKER_ID={WORKER_ID} is outside GRID_SHARD_COUNT={SHARD_COUNT}."
        )


def _owns_cell(round_idx: int, arm_idx: int) -> bool:
    """Does THIS worker own cell ``(round, arm)``? Always true in shared-queue mode."""
    if not SHARD_COUNT:
        return True
    n_arms = len(ARMS_ORDERED)
    n_groups = SHARD_COUNT // n_arms
    return arm_idx == WORKER_ID % n_arms and round_idx % n_groups == WORKER_ID // n_arms


def get_study() -> optuna.Study:
    return optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        sampler=optuna.samplers.RandomSampler(seed=0),
        load_if_exists=True,
        direction="maximize",
    )


def owned_cells() -> list[tuple[str, int]]:
    """Every ``(arm, seed)`` this worker owns, ROUND-MAJOR."""
    return [
        (arm, SEED_BASE + r)
        for r in range(ROUNDS)
        for i, arm in enumerate(ARMS_ORDERED)
        if _owns_cell(r, i)
    ]


def enqueue_all(study: optuna.Study) -> int:
    """Enqueue every ``(arm, seed)`` cell this worker owns, ROUND-MAJOR.

    Idempotent: a requeue after a kill tops the queue back up rather than duplicating it.
    FAIL and PRUNED are DELIBERATELY absent from ``seen``, so a cell whose run died is
    re-enqueued rather than skipped -- counting a crashed trial as "already present" is how a
    resubmit silently runs a PARTIAL factorial and reports nothing wrong.

    An enqueued (WAITING) trial keeps its values in ``system_attrs["fixed_params"]`` until it
    runs, so ``.params`` alone would fail to recognize a queue that is merely waiting.
    """
    seen: set[tuple[str, int]] = set()
    for t in study.trials:
        if t.state.name in ("FAIL", "PRUNED"):
            continue
        params = t.params or t.system_attrs.get("fixed_params", {})
        if "arm" in params and "seed" in params:
            seen.add((params["arm"], int(params["seed"])))
    added = 0
    for arm, seed in owned_cells():
        if (arm, seed) in seen:
            continue
        study.enqueue_trial({"arm": arm, "seed": seed})
        seen.add((arm, seed))
        added += 1
    return added


def main() -> None:
    create_only = "--create-only" in sys.argv
    study = get_study()
    added = enqueue_all(study)
    print(
        f"[w{WORKER_ID}] study {STUDY_NAME} at {STORAGE}: enqueued {added} cell(s) "
        f"(shard {WORKER_ID}/{SHARD_COUNT or 'shared'}, rounds={ROUNDS}, "
        f"arms={ARMS_ORDERED})",
        flush=True,
    )
    if create_only:
        return

    cfg = load_config()
    dataset = build_dataset()
    print(f"[w{WORKER_ID}] dataset: {len(dataset)} aggregated genotypes", flush=True)
    pinned, pin_report = resolve_pinned_test(dataset, cfg)

    out_path = osp.join(RESULTS_DIR, f"bxfx_w{WORKER_ID}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    runs: list[dict[str, Any]] = []
    durations: list[float] = []

    def objective(trial: optuna.Trial) -> float:
        arm = trial.suggest_categorical("arm", ARMS_ORDERED)
        # Inclusive bounds, so the top of the range is the LAST seed, not one past it.
        # Inert while the queue is draining (an enqueued trial carries fixed values), but
        # it is the space a sampled trial would draw from and it must match `owned_cells`.
        seed = trial.suggest_int("seed", SEED_BASE, SEED_BASE + ROUNDS - 1)
        cell = build_cell(
            arm, int(seed), cfg, epochs=int(GRID_EPOCHS) if GRID_EPOCHS else None
        )
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "torchcell_027_bxfx"),
            group="bxfx_grid_000",
            name=f"{arm}-s{seed}",
            tags=["027", "bxfx", arm, f"seed{seed}"],
            config=cell.model_dump(),
            reinit=True,
            mode=cast(WandbMode, WANDB_MODE),
        )
        result = run_cell(
            cell,
            cfg,
            dataset,
            pinned,
            pin_report,
            on_epoch=lambda row: wandb.log(row, step=int(row["epoch"])),
        )
        pin = result["test"]["pinned"]
        run.summary.update(
            {
                # The PRIMARY endpoint, and the summary carries no bare maximum of anything.
                "test_spearman_pinned": pin["spearman"],
                "test_pearson_pinned": pin["pearson"],
                "n_test_pinned": pin["n"],
                "test_spearman_all": result["test"]["all"]["spearman"],
                "select_epoch": result["select_epoch"],
                "select_rule": result["select_rule"],
                "select_val_pearson": result["select_val_pearson"],
                "wall_time_s": result["wall_time_s"],
                "fcl_rf_test_spearman": cfg.baselines.fcl_rf_test_spearman,
            }
        )
        result["wandb_run_id"] = run.id
        result["wandb_url"] = run.url
        run.finish()
        durations.append(float(result["wall_time_s"]))
        runs.append(result)
        with open(out_path, "w") as fh:
            json.dump(
                {"worker": WORKER_ID, "shard_count": SHARD_COUNT, "runs": runs},
                fh,
                default=str,
            )
        print(f"[w{WORKER_ID}] checkpointed {len(runs)} -> {out_path}", flush=True)
        return float(pin["spearman"])

    def deadline_callback(study: optuna.Study, _: Any) -> None:
        """Stop before CLAIMING a cell that cannot finish.

        The budget uses the LONGEST observed cell, not the mean: a run killed at 90% has no
        selected epoch and no test pass, so it contributes nothing, and reserving for the
        average guarantees losing the tail.
        """
        if not DEADLINE_EPOCH:
            return
        left = DEADLINE_EPOCH - int(time.time()) - GRID_TEARDOWN_S
        need = max(int(max(durations)) if durations else 0, GRID_MIN_TRIAL_S)
        if left < need:
            print(
                f"[w{WORKER_ID}] stopping: {left}s left, next cell needs ~{need}s",
                flush=True,
            )
            study.stop()

    study.optimize(
        objective,
        # EXACTLY the cells this worker owns, NOT the whole grid. Under sharding a worker
        # enqueues 2 of the 144 cells, so `n_trials=144` would drain the queue and then let
        # the sampler INVENT 142 more from the `suggest_*` spaces -- running cells another
        # worker owns, duplicating them, and poisoning the balanced-prefix property the shard
        # rule exists to guarantee. (`metabolism_grid_runner.py` passes the full grid size and
        # is protected only by its deadline callback firing first.)
        n_trials=len(owned_cells()),
        # NO `catch=`. `catch=(Exception,)` turns a crash into a FAILED trial and the worker
        # marches on through its queue, exiting COMPLETED 0:0 with nothing to show -- which
        # is exactly how a broken arm burns an allocation invisibly. A 027 cell that raises
        # should kill its worker loudly; the other 71 workers are unaffected because each
        # owns its own study file.
        callbacks=[deadline_callback],
    )
    print(f"[w{WORKER_ID}] DONE: {len(runs)} cells -> {out_path}", flush=True)
    print(f"[w{WORKER_ID}] exp dir {EXP_DIR}", flush=True)


if __name__ == "__main__":
    main()
