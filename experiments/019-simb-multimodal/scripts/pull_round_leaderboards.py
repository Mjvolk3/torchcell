# experiments/019-simb-multimodal/scripts/pull_round_leaderboards.py
# [[experiments.019-simb-multimodal.scripts.pull_round_leaderboards]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/pull_round_leaderboards
"""One leaderboard across every phenotype strand run for the SIMB multimodal figure.

WHY THIS EXISTS. The six strands (expression, morphology, joint expression+morphology,
betaxanthin, beta-carotene, amino-acid metabolome, and the joint betaxanthin+metabolome
replication) each live in their own W&B project, were scored at different times, and are
quoted in prose from whichever run someone happened to be looking at. A retrospective that
compares them has to read them all the same way or it is comparing scoring conventions
rather than strands. This dumps one CSV, from which every retrospective number is read.

THE SCORING RULE, and it is named because it is biased. For each run this records:

  last      the metric at the final logged epoch. What `run.summary` holds, and what the
            expression round proved can sit ~1,300 epochs before the good checkpoint.
  roll_max  the maximum of a centered 5-epoch rolling mean of the validation metric. An
            UPWARD-BIASED order statistic whose bias grows with the number of epochs run,
            so it is only comparable between runs of similar length; `epochs` is carried
            alongside for exactly that reason. Same rule as
            `experiments/019-simb-multimodal/scripts/wave4b_convergence.py`.

Neither is "the score". Both are reported so a reader can see when they disagree, which is
itself one of the round's findings.

TRUNCATION IS RECORDED, NOT HIDDEN. `epochs` is the last logged epoch and `state` is W&B's,
so a run that was walltime-killed while still climbing is visible as such. Runs whose
metric is identically zero are collapsed predictors (the lr = 1e-3 cells), flagged by
`is_collapsed` rather than dropped, because dropping them would silently improve every
per-project maximum.

W&B history is DOWNSAMPLED to at most `samples` points per run, so `epoch_at_roll_max` is
approximate to that resolution. It is a locator, not a measurement.

Run from repo root (needs network; W&B login):
  PYTHONPATH=. python experiments/019-simb-multimodal/scripts/pull_round_leaderboards.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import signal

import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv

ENTITY = "zhao-group"
EXPERIMENT = "019-simb-multimodal"

# EVERY 019-023 project, not a selection. An earlier version of this script covered eight
# projects and 396 runs; the account is 28 projects and ~2,187 runs, and the omitted ones
# were not redundant. `torchcell_020_betaxanthin` and `_v3` train on 4,235 strains where
# `_v4` trains on 3,698 (the pinned Merzbacher test split removes genes), so the best
# betaxanthin score lives in a project the short list did not read.
#
# METRIC NAMES CHANGED TWICE, which is why each strand carries a list of ALIASES rather
# than one key. The honest per-feature validation correlation was called
# `val/per_gene/pearson_per_gene` for expression and `val/global/pearson_per_gene` for
# morphology before the rename to `val/<head>/pearson_per_feature`. A puller that knows
# only the new names silently returns nothing for the first eleven projects, which is a
# far worse failure than an error, because the leaderboard still looks populated.
#
# Beta-carotene's primary is SPEARMAN because its target is a subjective ordinal
# colony-color score, for which a Pearson ceiling is the wrong object (see
# pigment_noise_ceiling.py).
STRANDS: list[tuple[str, str, list[str], list[str]]] = [
    # --- expression ---------------------------------------------------------
    ("expression", "torchcell_019-simb-multimodal_cgt_multitask",
     ["val/per_gene/pearson_per_gene"], []),
    ("expression", "torchcell_019_expr", ["val/per_gene/pearson_per_gene"], []),
    ("expression", "torchcell_019_expr_v2", ["val/per_gene/pearson_per_gene"], []),
    ("expression", "torchcell_019_expr_v3", ["val/expression/pearson_per_feature"], []),
    ("expression", "torchcell_019_expr_v5", ["val/expression/pearson_per_feature"], []),
    ("expression", "torchcell_019_expr_v6", ["val/expression/pearson_per_feature"], []),
    ("expression", "torchcell_019_expr_v7", ["val/expression/pearson_per_feature"], []),
    ("expression", "torchcell_019_expr_v8", ["val/expression/pearson_per_feature"], []),
    ("expression_masked", "torchcell_019_expr_v9",
     ["val/expression/pearson_per_feature"], []),
    # --- morphology ---------------------------------------------------------
    ("morphology", "torchcell_019_morph_v2", ["val/global/pearson_per_gene"], []),
    ("morphology", "torchcell_019_morph_v3", ["val/morphology/pearson_per_feature"], []),
    ("morphology", "torchcell_019_morph_v5", ["val/morphology/pearson_per_feature"], []),
    # --- joint expression + morphology --------------------------------------
    ("expression_morphology_joint", "torchcell_019_expr_morph",
     ["val/global/pearson_per_gene"], ["val/per_gene/pearson_per_gene"]),
    ("expression_morphology_joint", "torchcell_019_expr_morph_v2",
     ["val/global/pearson_per_gene"], ["val/per_gene/pearson_per_gene"]),
    ("expression_morphology_joint", "torchcell_019_expr_morph_v3",
     ["val/morphology/pearson_per_feature"], ["val/expression/pearson_per_feature"]),
    ("expression_morphology_joint", "torchcell_019_expr_morph_v4",
     ["val/morphology/pearson_per_feature"], ["val/expression/pearson_per_feature"]),
    ("expression_morphology_joint", "torchcell_019_expr_morph_v5",
     ["val/morphology/pearson_per_feature"], ["val/expression/pearson_per_feature"]),
    # --- betaxanthin --------------------------------------------------------
    ("betaxanthin", "torchcell_020_betaxanthin",
     ["val/betaxanthin/pearson_per_feature"], ["val/betaxanthin/spearman_per_feature"]),
    ("betaxanthin", "torchcell_020_betaxanthin_v3",
     ["val/betaxanthin/pearson_per_feature"], ["val/betaxanthin/spearman_per_feature"]),
    ("betaxanthin", "torchcell_020_betaxanthin_v4",
     ["val/betaxanthin/pearson_per_feature"], ["val/betaxanthin/spearman_per_feature"]),
    ("betaxanthin", "torchcell_020_metabolism",
     ["val/betaxanthin/pearson_per_feature"], ["val/betaxanthin/spearman_per_feature"]),
    # --- beta-carotene ------------------------------------------------------
    ("beta_carotene", "torchcell_020_beta_carotene",
     ["val/beta_carotene/spearman_per_feature"], ["val/beta_carotene/pearson_per_feature"]),
    ("beta_carotene", "torchcell_020_beta_carotene_v3",
     ["val/beta_carotene/spearman_per_feature"], ["val/beta_carotene/pearson_per_feature"]),
    ("beta_carotene", "torchcell_021_beta_carotene_v4",
     ["val/beta_carotene/spearman_per_feature"], ["val/beta_carotene/pearson_per_feature"]),
    # --- amino acid ---------------------------------------------------------
    ("amino_acid", "torchcell_020_mulleder19",
     ["val/mulleder19/pearson_per_feature"], ["val/mulleder19/spearman_per_feature"]),
    ("amino_acid", "torchcell_020_mulleder19_v3",
     ["val/mulleder19/pearson_per_feature"], ["val/mulleder19/spearman_per_feature"]),
    ("amino_acid", "torchcell_022_mulleder19_v4",
     ["val/mulleder19/pearson_per_feature"], ["val/mulleder19/spearman_per_feature"]),
    # --- betaxanthin with a metabolome head ---------------------------------
    ("betaxanthin_amino_acid_joint", "torchcell_023_bx_m19_v1",
     ["val/betaxanthin/pearson_per_feature"],
     ["val/betaxanthin/spearman_per_feature", "val/mulleder19/pearson_per_feature"]),
]

# Config keys worth carrying into the leaderboard. Flat keys only; the nested Hydra tree is
# not reconstructed here because the training harness already flattens what varies into
# W&B config at launch.
CONFIG_KEYS = [
    "seed",
    "lr",
    "dropout",
    "num_layers",
    "hidden_channels",
    "target_norm",
    "graph_prior",
    "dist",
    "decoder",
    "n_train_supervised",
    "n_val_supervised",
    "n_test_supervised",
    "total_param_count",
    "perf/epoch_seconds",
]

ROLL_WINDOW = 5

# 500 sampled points locate a maximum on a curve that is itself already smoothed by a
# 5-point rolling mean; 2000 cost four times the requests for a locator that moves by a few
# epochs. The epoch of the peak is approximate to this resolution either way, and is
# documented as a locator rather than a measurement.
HISTORY_SAMPLES = 500

# PER-PROJECT CACHING, because this pull is long enough that a single hung request should
# not cost the whole run. Each project's rows are written to its own JSON under
# `results/_leaderboard_cache/` as soon as it completes, and a re-run skips any project
# already cached. Delete a cache file to refresh that project; delete the directory to
# refresh everything. The cache is deliberately keyed by project NAME only: the aliases and
# config keys are part of the script, so a change to either should be accompanied by
# clearing the cache, and the header comment says so.
CACHE_DIR = "_leaderboard_cache"

# Bumped whenever a row gains a field, so a cache written by an older version is refetched
# instead of silently producing a table with a column full of blanks. v2 added the mse and
# nmse columns.
CACHE_SCHEMA = 2

# HISTORY IS FETCHED FOR A BOUNDED CANDIDATE SET, NOT FOR EVERY RUN, and this is a real
# limitation rather than a detail. Every run's SUMMARY is read (that is one paginated query
# per project and is free), but one history request per run over 2,187 runs does not
# finish: the largest projects hold 496 and 295 runs whose histories run to 10,000 epochs.
#
# The candidate set per project is the union of
#   * the top CANDIDATES_BY_LAST runs by their final logged value, and
#   * the top CANDIDATES_BY_LENGTH runs by epochs reached,
# because a project's best `roll_max` is held either by a run that ended high or by a run
# that ran long and peaked late. Both are covered.
#
# WHAT THIS CAN MISS: a run that peaked high EARLY and then collapsed to a low final value
# in a short run. Those exist in the early rounds (the mean-collapse runs of
# `cgt_multitask` and `expr_v2`), but they peaked at 0.04 to 0.14, far below the strand
# maxima this table reports, so the risk is to a median rather than to a maximum. Every row
# carries `n_runs` and `n_history_fetched` so the sampling is visible wherever the number
# is quoted, and a project at or below the limit is exhaustive.
# HARD TIMEOUT PER HISTORY REQUEST. `wandb.Api(timeout=...)` covers the HTTP call, not the
# pagination loop underneath `run.history()`, so a single run can wedge the whole pull
# indefinitely; this happened twice, each time stalling for 40+ minutes on one project with
# the process alive and idle. SIGALRM is the right instrument here because the script is
# single-threaded and the goal is to ABANDON a stuck request rather than to wait it out: a
# thread-pool timeout would leave the request running and the interpreter would not exit.
# A skipped run is reported and simply has no roll_max, which is already how a
# summary-only run is represented.
HISTORY_TIMEOUT_S = 120

# Bound on materializing a project's run list. Generous, because 496 runs of summaries is a
# real amount of data; the point is that it terminates.
RUNS_TIMEOUT_S = 300

# ERROR METRICS ARE PULLED ALONGSIDE THE CORRELATION, and the reason is a finding rather
# than completeness. The expression diagnosis showed the best run reaching r = 0.236 while
# its `nmse` sat ABOVE 1, meaning it was worse than predicting each feature's training mean
# in squared error the whole time. A leaderboard that records only the correlation cannot
# distinguish an arm that fixed the ordering from one that fixed the ordering AND the
# magnitudes, which is exactly the distinction the next campaign has to make.
#
# What is recorded per run, derived from the primary metric's head (`val/<head>/...`):
#   <m>_min                  the best that error metric ever reached
#   <m>_at_primary_peak      its value AT the epoch the correlation peaked
# The second is the load-bearing one. A minimum reached at epoch 200 says nothing about the
# model that is actually selected, which is the one at the correlation peak.
#
# `nmse` is normalized so 1.0 is exactly "predict each feature's training mean", so
# `nmse_at_primary_peak > 1` is the precise statement that the selected model loses to the
# mean on squared error. These were added to the training path during the v8 round, so
# earlier projects simply do not log them and the absent-key skip handles it.
ERROR_METRICS = ("mse", "nmse")

CANDIDATES_BY_LAST = 8
CANDIDATES_BY_LENGTH = 5

# Approximate run counts, used ONLY to order the pull smallest-first so an interruption
# banks whole projects. A wrong value costs ordering, never correctness; a project missing
# from this map sorts first.
PROJECT_SIZE_HINT = {
    "torchcell_019-simb-multimodal_cgt_multitask": 496,
    "torchcell_019_expr": 26, "torchcell_019_expr_v2": 120,
    "torchcell_019_expr_v3": 60, "torchcell_019_expr_v5": 35,
    "torchcell_019_expr_v6": 158, "torchcell_019_expr_v7": 295,
    "torchcell_019_expr_v8": 163, "torchcell_019_expr_v9": 16,
    "torchcell_019_morph_v2": 71, "torchcell_019_morph_v3": 89,
    "torchcell_019_morph_v5": 23, "torchcell_019_expr_morph": 21,
    "torchcell_019_expr_morph_v2": 80, "torchcell_019_expr_morph_v3": 77,
    "torchcell_019_expr_morph_v4": 4, "torchcell_019_expr_morph_v5": 32,
    "torchcell_020_betaxanthin": 42, "torchcell_020_betaxanthin_v3": 53,
    "torchcell_020_betaxanthin_v4": 44, "torchcell_020_metabolism": 3,
    "torchcell_020_beta_carotene": 47, "torchcell_020_beta_carotene_v3": 50,
    "torchcell_021_beta_carotene_v4": 39, "torchcell_020_mulleder19": 36,
    "torchcell_020_mulleder19_v3": 30, "torchcell_022_mulleder19_v4": 31,
    "torchcell_023_bx_m19_v1": 46,
}


def _roll_max(values: np.ndarray, window: int = ROLL_WINDOW) -> tuple[float, int]:
    """Max of a centered rolling mean, and the index it occurs at."""
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return float("nan"), -1
    series = pd.Series(values).rolling(window, center=True, min_periods=1).mean()
    idx = int(series.idxmax())
    return float(series.iloc[idx]), idx


class _HistoryTimeout(Exception):
    """Raised by the SIGALRM handler when one history request overruns."""


def _timeout_handler(signum, frame):  # noqa: ARG001
    raise _HistoryTimeout


def fetch_history(run, key: str, samples: int):
    """`run.history` for one key, abandoned after HISTORY_TIMEOUT_S. None if it overran."""
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(HISTORY_TIMEOUT_S)
    try:
        return run.history(keys=["epoch", key], samples=samples)
    except _HistoryTimeout:
        return None
    finally:
        signal.alarm(0)


def _candidates(runs: list, primaries: list[str]) -> set[str]:
    """Run ids worth a history request, by final value and by length. See CACHE_DIR notes."""

    def last_value(run) -> float:
        for key in primaries:
            if key in run.summary:
                try:
                    return float(run.summary[key])
                except (TypeError, ValueError):
                    return float("-inf")
        return float("-inf")

    def epochs(run) -> float:
        try:
            return float(run.summary.get("epoch", -1))
        except (TypeError, ValueError):
            return -1.0

    by_last = sorted(runs, key=last_value, reverse=True)[:CANDIDATES_BY_LAST]
    by_length = sorted(runs, key=epochs, reverse=True)[:CANDIDATES_BY_LENGTH]
    return {r.id for r in by_last} | {r.id for r in by_length}


def _attach_error_metrics(run, primary: str, peak_epoch: float | None, row: dict) -> None:
    """Record each ERROR_METRIC's minimum and its value at the correlation peak.

    The head is taken from the primary key (`val/<head>/<stat>`), so this follows the
    metric rename automatically rather than hardcoding either generation of names. Older
    projects predate `mse`/`nmse` entirely, and the `in run.summary` check skips them
    without a request.
    """
    parts = primary.split("/")
    if len(parts) < 3:
        return
    head = parts[1]
    for name in ERROR_METRICS:
        key = f"val/{head}/{name}"
        if key not in run.summary:
            continue
        hist = fetch_history(run, key, HISTORY_SAMPLES)
        if hist is None or hist.empty or key not in hist:
            continue
        values = hist[key].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        row[f"{name}_min"] = float(np.nanmin(values))
        row[f"{name}_last"] = float(values[finite][-1])
        if peak_epoch is not None and "epoch" in hist:
            epochs = hist["epoch"].to_numpy(dtype=float)
            # Nearest logged epoch, because validation metrics and error metrics can be
            # logged on different cadences and an exact match is not guaranteed.
            pos = int(np.nanargmin(np.abs(epochs - peak_epoch)))
            if np.isfinite(values[pos]):
                row[f"{name}_at_primary_peak"] = float(values[pos])


def list_runs(api: wandb.Api, project: str) -> list:
    """Materialize a project's runs, with a timeout and shrinking page size.

    `list(api.runs(..., per_page=500))` asks the server for one enormous page, and on the
    496-run project that request never returned: the pull sat idle for over an hour with
    the process alive. The history timeout does not cover this, because no history call has
    been made yet. Smaller pages are more requests but each one is small enough to come
    back, and the whole pass is bounded so a wedged page costs one project rather than the
    run.
    """
    for per_page in (500, 100, 25):
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(RUNS_TIMEOUT_S)
        try:
            return list(api.runs(f"{ENTITY}/{project}", per_page=per_page))
        except _HistoryTimeout:
            print(
                f"    run listing timed out at per_page={per_page}, retrying smaller",
                flush=True,
            )
        finally:
            signal.alarm(0)
    raise TimeoutError(f"could not list runs for {project}")


def pull(api: wandb.Api, strand: str, project: str, primaries: list[str], extras: list[str]):
    rows = []
    n_missing_history = 0
    all_runs = list_runs(api, project)
    wanted = _candidates(all_runs, primaries)
    for run in all_runs:
        # The primary metric is whichever ALIAS this run actually logged. Summary keys are
        # already loaded, so this costs no extra request and avoids fetching history under
        # a name the run never used.
        primary = next((k for k in primaries if k in run.summary), primaries[0])
        keys = [primary, *extras]
        runtime = run.summary.get("_runtime")
        row: dict[str, object] = {
            "strand": strand,
            "project": project,
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "tags": ",".join(run.tags),
            "primary_metric": primary,
            "runtime_s": float(runtime) if runtime is not None else None,
            "runtime_h": float(runtime) / 3600.0 if runtime is not None else None,
        }
        for key in CONFIG_KEYS:
            row[key.replace("/", "_")] = run.config.get(key, run.summary.get(key))
        # ONE REQUEST PER KEY, deliberately. `run.history(keys=[a, b])` keeps only the
        # rows where BOTH a and b are present, so asking for the primary metric together
        # with an auxiliary head's metric returns an EMPTY frame for every run that does
        # not carry that head. In a paired design that is exactly the control arm, so the
        # joined form silently drops one side of every comparison the round exists to
        # make.
        row["epochs"] = None
        row["n_history_points"] = 0
        row["history_fetched"] = run.id in wanted
        row["n_runs_in_project"] = len(all_runs)
        if run.id not in wanted:
            # Summary-only row: it still carries config, tags and the final value, so it
            # counts toward run totals and toward n_train, but it is never a candidate for
            # the project maximum and is excluded from roll_max statistics by having no
            # primary_roll_max at all rather than a misleading zero.
            for key in keys:
                if key in run.summary:
                    tag = ("primary" if key == primaries[0]
                           else key.split("/")[-2] + "_" + key.split("/")[-1])
                    try:
                        row[f"{tag}_last"] = float(run.summary[key])
                    except (TypeError, ValueError):
                        pass
            rows.append(row)
            continue
        for key in keys:
            # A run that never logged this key has no history for it, and asking anyway is
            # what made this pull hang: W&B answers an unknown key with a full unsampled
            # scan. `summary` is already in memory, so the check is free.
            if key not in run.summary:
                n_missing_history += 1
                continue
            history = fetch_history(run, key, HISTORY_SAMPLES)
            if history is None:
                print(f"    TIMEOUT after {HISTORY_TIMEOUT_S}s: {run.id} {key}", flush=True)
                continue
            if history.empty or key not in history:
                continue
            epochs = (
                history["epoch"].to_numpy()
                if "epoch" in history
                else np.arange(len(history))
            )
            values = history[key].to_numpy(dtype=float)
            best, idx = _roll_max(values)
            tag = (
                "primary"
                if key == primary
                else key.split("/")[-2] + "_" + key.split("/")[-1]
            )
            finite = np.isfinite(values)
            row[f"{tag}_last"] = float(values[finite][-1]) if finite.any() else None
            row[f"{tag}_roll_max"] = best
            row[f"{tag}_epoch_at_roll_max"] = (
                float(epochs[idx]) if 0 <= idx < len(epochs) else None
            )
            if key == primary:
                row["epochs"] = float(np.nanmax(epochs))
                row["n_history_points"] = int(len(history))
                # A run whose validation metric never leaves zero is a collapsed
                # constant predictor, not a weak model. Flagged, never dropped.
                row["is_collapsed"] = bool(np.nanmax(np.abs(values)) < 1e-6)
                peak_epoch = float(epochs[idx]) if 0 <= idx < len(epochs) else None
                _attach_error_metrics(run, primary, peak_epoch, row)
        rows.append(row)
    n_fetched = sum(1 for r in rows if r.get("history_fetched"))
    print(f"    history fetched for {n_fetched} of {len(rows)} runs"
          + (f"; {n_missing_history} run/key pairs had none logged" if n_missing_history else ""),
          flush=True)
    return rows


def summarize(df: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {}
    for strand, group in df.groupby("strand"):
        # `primary_roll_max` is NaN for every summary-only run, and a project can consist
        # entirely of those: `cgt_multitask` logs the metric under a name none of its
        # candidate runs carry, so nothing in it has history. `idxmax()` on an all-NaN
        # column returns NaN and `.loc[nan]` raises, which is how the whole pull crashed
        # after every project had been fetched. Drop the NaNs first and skip an empty group.
        live = group[~group["is_collapsed"].fillna(False).astype(bool)]
        live = live[live["primary_roll_max"].notna()]
        best = live.loc[live["primary_roll_max"].idxmax()] if len(live) else None
        by_project = {}
        for proj, sub in group.groupby("project"):
            sub_live = sub[~sub["is_collapsed"].fillna(False).astype(bool)]
            sub_live = sub_live[sub_live["primary_roll_max"].notna()]
            if not len(sub_live):
                continue
            top = sub_live.loc[sub_live["primary_roll_max"].idxmax()]
            by_project[str(proj)] = {
                "n_runs": int(len(sub)),
                "best_roll_max": float(top["primary_roll_max"]),
                "best_run_id": str(top["run_id"]),
                "n_train_supervised": (
                    float(top["n_train_supervised"])
                    if pd.notna(top["n_train_supervised"]) else None
                ),
                "epochs": float(top["epochs"]) if pd.notna(top["epochs"]) else None,
                "runtime_h": float(top["runtime_h"]) if pd.notna(top["runtime_h"]) else None,
            }
        out[str(strand)] = {
            "by_project": by_project,
            "n_projects": int(group["project"].nunique()),
            "project": str(group["project"].iloc[0]),
            "metric": str(group["primary_metric"].iloc[0]),
            "n_runs": int(len(group)),
            "n_collapsed": int(group["is_collapsed"].fillna(False).sum()),
            "epochs_min": float(group["epochs"].min()) if group["epochs"].notna().any() else None,
            "epochs_max": float(group["epochs"].max()) if group["epochs"].notna().any() else None,
            "best_roll_max": float(best["primary_roll_max"]) if best is not None else None,
            "best_run_id": str(best["run_id"]) if best is not None else None,
            "best_run_epochs": float(best["epochs"]) if best is not None else None,
            "best_epoch_at_roll_max": (
                float(best["primary_epoch_at_roll_max"]) if best is not None else None
            ),
            "median_roll_max": (
                float(live["primary_roll_max"].median()) if len(live) else None
            ),
        }
    return out


def main() -> None:
    load_dotenv()
    experiment_root = os.environ["EXPERIMENT_ROOT"]
    results_dir = osp.join(experiment_root, EXPERIMENT, "results")
    os.makedirs(results_dir, exist_ok=True)

    api = wandb.Api(timeout=60)
    cache_dir = osp.join(results_dir, CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)

    # SMALLEST PROJECTS FIRST. Each project is cached only when it finishes, so a run
    # interrupted part-way through a 496-run project caches nothing at all. Ordering by size
    # makes progress monotone under interruption: every completed pass banks several whole
    # projects, and the large ones are attempted only once the rest are safe.
    ordered = sorted(STRANDS, key=lambda s: PROJECT_SIZE_HINT.get(s[1], 0))
    rows: list[dict[str, object]] = []
    for strand, project, primaries, extras in ordered:
        cache_path = osp.join(cache_dir, f"{project}.json")
        got = None
        if osp.exists(cache_path):
            with open(cache_path) as fh:
                blob = json.load(fh)
            if isinstance(blob, dict) and blob.get("schema") == CACHE_SCHEMA:
                got = blob["rows"]
                print(f"cached  {strand:30s} <- {project:44s} {len(got):4d} runs", flush=True)
            else:
                print(f"stale   {strand:30s} <- {project:44s} refetching", flush=True)
        if got is None:
            got = pull(api, strand, project, primaries, extras)
            with open(cache_path, "w") as fh:
                json.dump({"schema": CACHE_SCHEMA, "rows": got}, fh)
            print(f"pulled  {strand:30s} <- {project:44s} {len(got):4d} runs", flush=True)
        rows.extend(got)
    df = pd.DataFrame(rows)
    csv_path = osp.join(results_dir, "round_leaderboards.csv")
    df.to_csv(csv_path, index=False)

    summary = summarize(df)
    with open(osp.join(results_dir, "round_leaderboards_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"-> {csv_path}")


if __name__ == "__main__":
    main()
