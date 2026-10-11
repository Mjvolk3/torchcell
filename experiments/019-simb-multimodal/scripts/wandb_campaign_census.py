# experiments/019-simb-multimodal/scripts/wandb_campaign_census.py
# [[experiments.019-simb-multimodal.scripts.wandb_campaign_census]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/wandb_campaign_census
"""Census of every W&B run of the 019 SIMB multimodal campaign, before and after SIMB.

SCOPE. Every `zhao-group` project of the campaign (PROJECTS below): the July rounds that
led into SIMB and the September/October rounds after it. The four benchmark and profiling
projects (INFRA) are counted as one "infrastructure" group and never scored.

PHASE is per run, from its creation time (UTC): `before` = 2026-07-22 to 2026-07-31,
`after` = 2026-09-03 onward, `between` = anything in August or on 2026-09-01/02 (a few
`cgt_multitask` runs, and the start of `expr_v9`, which spans both).

HEADS AND METRIC NAMES. A head is scored on its validation Pearson per feature (per gene for
expression and proteome, per CalMorph feature for morphology), read under the first key the
run logged from HEAD_KEYS. The eleven oldest projects log the same statistic under the
pre-rename names `val/per_gene/pearson_per_gene` (expression) and
`val/global/pearson_per_gene` (morphology) (project_census.py documents the rename). The
first masked rounds (part of `expr_v9`) log only the per-mask-level keys; for those the
zero-revealed level `@k0` is read and the key is recorded. The key actually read is stored
per run, so nothing is silently aliased. Values from different stores (fig3_core against
fig3_proteome, 278 against 116 morphology features) are NOT directly comparable even under
one key; the dataset tag and feature count are carried per run for that reason.

THE SCORE is a window mean, never a maximum.
  * `final_window`: mean of the val metric over the last W epochs of the run's own budget,
    W = min(200, max(20, round(max_epochs / 6))), defined only for COMPLETE runs (last
    logged epoch >= max_epochs - 1). For the 1,200-epoch rounds this is epochs 1,000 to
    1,199, the registered window of the small-trunk readouts. The last epoch is the larger of
    the last sampled history epoch and the run summary's `epoch`.
  * `end_window`: for runs that FINISHED before their budget (EarlyStopping on val/loss with
    patience 20 was the trainer default through July, so nearly every July run stopped near
    epoch 20 to 60; later, time limits), the mean over the last min(200, max(5, round((stop + 1)
    / 6))) epochs up to the stop. A different statistic: a row is ranked on it
    (`score_kind` = "end", dagger in the table) only when none of its runs reached its budget.
  * `registered_window`: mean over epochs 1,000 to 1,199, for any run that reached 1,199
    (the same number as final_window when max_epochs is 1,200).
  * `raw_max`: maximum over the sampled history points. An upward-biased order statistic
    whose bias grows with the number of epochs; printed for reference, never ranked on.

COLLAPSE (the rule of collapse_census.py): a head LAUNCHES at the first epoch its
`val/<head>/pred_sd_ratio` reaches 0.05; it COLLAPSED if, after launch, the ratio stayed below
0.01 over a stretch of consecutive logged points spanning >= 50 epochs. With dense logging the
span equals the count of logged epochs; for runs longer than the sample cap the span is
measured in epochs between sampled points. Runs that never log the ratio (every July project
before expr_v8) have collapse = None, "not measurable", not False.

HISTORY IS SAMPLED. Each (run, key) is pulled with run.history(keys=["epoch", key],
samples=SAMPLES); W&B returns every row when the run has fewer, and an evenly thinned set
otherwise (the long v9/v13 budgets). The window mean over thinned points is still a mean
over the window; the raw max over thinned points is a lower bound on the max over all
epochs. Every raw pull is cached as JSON under --cache-dir, keyed by project and run id, and
reused unless the run's state or last step changed.

ARM. The arm tag a round writes (`S_ref_s3`, `C_prot_s4`, `M_fine`, ...; capitalized prefix,
split suffix `_s<k>` stripped) or a named W&B group; runs with neither (Optuna trials and
grid points of July) are their own one-run arm. The best arm per (project, phase, head) is
the arm with the highest mean final_window over its complete, non-collapsed runs; picking
the best of many arms is itself a selection, so the number of arms compared is recorded.

    python experiments/019-simb-multimodal/scripts/wandb_campaign_census.py --cache-dir <dir>

Writes results/wandb_campaign_census.json and notes-tex/figure-3-gate/tables/wandb_campaign_census.tex.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
import statistics as st
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import wandb

ENTITY = "zhao-group"
PROJECTS = [
    "torchcell_019-simb-multimodal_cgt_multitask",
    "torchcell_019_expr", "torchcell_019_expr_v2", "torchcell_019_expr_v3", "torchcell_019_expr_v5",
    "torchcell_019_expr_v6", "torchcell_019_expr_v7", "torchcell_019_expr_v8", "torchcell_019_expr_v9",
    "torchcell_019_expr_morph", "torchcell_019_expr_morph_v2", "torchcell_019_expr_morph_v3",
    "torchcell_019_expr_morph_v4", "torchcell_019_expr_morph_v5",
    "torchcell_019_morph_v2", "torchcell_019_morph_v3", "torchcell_019_morph_v5",
    "torchcell_019_expr_v10", "torchcell_019_expr_v11", "torchcell_019_expr_v12", "torchcell_019_expr_v13",
    "torchcell_019_expr_v15", "torchcell_019_expr_v17", "torchcell_019_expr_v18", "torchcell_019_expr_v21",
    "torchcell_019_expr_v21_canary", "torchcell_019_expr_v22",
    "torchcell_019_prot_v14", "torchcell_019_prot_v16", "torchcell_019_prot_v19", "torchcell_019_prot_v20",
    "torchcell_019_prot_v20_canary",
    "torchcell_019_morph_v24",
]
INFRA = ["torchcell_019_scaling", "torchcell_019_grid_bench", "torchcell_019_profile", "torchcell_019_small_bench"]
HEAD_KEYS = {
    "expression": ["val/expression/pearson_per_feature", "val/per_gene/pearson_per_gene", "val/expression/pearson_per_feature@k0"],
    "proteome": ["val/proteome/pearson_per_feature", "val/proteome/pearson_per_feature@k0"],
    "morphology": ["val/morphology/pearson_per_feature", "val/global/pearson_per_gene"],
}
BEFORE = ("2026-07-22", "2026-08-01")
AFTER_FROM = "2026-09-03"
REGISTERED = (1000, 1199)
LAUNCH, DEAD, RULE = 0.05, 0.01, 50
SAMPLES = 1500
AUTO_GROUP = re.compile(r"[0-9a-f]{64}")
ARM_TAG = re.compile(r"^[A-Z][A-Za-z0-9]*_[A-Za-z0-9_]+$")
REPO = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
RESULTS = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")
TABLES = osp.join(REPO, "notes-tex", "figure-3-gate", "tables")


def url(project: str, run_id: str) -> str:
    return f"https://wandb.ai/{ENTITY}/{project}/runs/{run_id}"


def phase_of(created: str) -> str:
    if BEFORE[0] <= created < BEFORE[1]:
        return "before"
    return "after" if created >= AFTER_FROM else "between"


def arm_of(run: Any) -> tuple[str, str]:
    """Arm label and how it was found ('tag', 'group' or 'run')."""
    for t in run.tags:
        if ARM_TAG.match(t):
            return re.sub(r"_s\d+(?=(_|$))", "", t), "tag"
    if run.group and not AUTO_GROUP.search(run.group) and run.group != run.name:
        return run.group, "group"
    return run.id, "run"


def dig(d: Any, *path: str) -> Any:
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def arm_config(c: dict[str, Any]) -> dict[str, Any]:
    mt = c.get("multitask") or {}
    dm = c.get("data_module") or {}
    cd = c.get("cell_dataset") or {}
    model = c.get("model") or {}
    return {
        "active_heads": mt.get("active_heads", c.get("active_heads")),
        "condition_head": mt.get("condition_head"),
        "condition_permute": mt.get("condition_permute"),
        "mask_schedule": mt.get("mask_schedule"),
        "decoder": mt.get("decoder", c.get("decoder")),
        "dist": mt.get("dist", c.get("dist")),
        "loss_fn": mt.get("loss_fn", c.get("loss_fn")),
        "use_gene_pool": dig(mt, "heads", "global", "use_gene_pool"),
        "global_output_dim": dig(mt, "heads", "global", "output_dim"),
        "require_modalities": dm.get("require_modalities"),
        "split_seed": dm.get("split_seed"),
        "batch_size": dm.get("batch_size"),
        "dataset_tag": cd.get("dataset_tag"),
        "node_embeddings": cd.get("node_embeddings"),
        "n_graphs": len(cd.get("graphs") or []),
        "hidden_channels": model.get("hidden_channels", c.get("hidden_channels")),
        "num_layers": model.get("num_transformer_layers", c.get("num_layers")),
        "perturb_cls": model.get("perturb_cls"),
        "lr": c.get("lr"),
        "seed": c.get("seed"),
        "max_epochs": dig(c, "trainer", "max_epochs"),
        "n_train_supervised": c.get("n_train_supervised"),
        "n_val_supervised": c.get("n_val_supervised"),
    }


def pull(run: Any, project: str, cache_dir: str) -> dict[str, Any]:
    """Raw per-run pull: metadata, config subset, and sampled histories of the needed keys."""
    path = osp.join(cache_dir, project, f"{run.id}.json")
    step = run.summary.get("_step")
    if osp.exists(path):
        cached = json.load(open(path))
        if cached["state"] == run.state and cached["summary_step"] == step and cached["samples"] == SAMPLES:
            return cached
    summary_keys = set(run.summary.keys())
    heads = {}
    for head, keys in HEAD_KEYS.items():
        key = next((k for k in keys if k in summary_keys), None)
        if key is None:
            continue
        ratio = f"val/{head}/pred_sd_ratio"
        hist = {}
        for k in (key, ratio) if ratio in summary_keys else (key,):
            hist[k] = sorted(
                (int(x["epoch"]), float(x[k]))
                for x in run.history(keys=["epoch", k], pandas=False, samples=SAMPLES)
                if x.get(k) is not None and x.get("epoch") is not None
            )
        heads[head] = {"key": key, "ratio_key": ratio if ratio in summary_keys else None, "history": hist}
    arm, arm_from = arm_of(run)
    out = {
        "project": project,
        "id": run.id,
        "name": run.name,
        "group": run.group,
        "tags": list(run.tags),
        "created_at": run.created_at,
        "state": run.state,
        "summary_step": step,
        "summary_epoch": run.summary.get("epoch"),
        "samples": SAMPLES,
        "arm": arm,
        "arm_from": arm_from,
        "config": arm_config(run.config),
        "val_keys_logged": sorted(k for k in summary_keys if k.startswith("val/") and ("pearson" in k or k == "val/loss")),
        "heads": heads,
    }
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    return out


def score_head(h: dict[str, Any], max_epochs: int | None, state: str, summary_epoch: Any) -> dict[str, Any]:
    val = h["history"][h["key"]]
    last_hist = val[-1][0] if val else -1
    # The sampled history can miss the last epoch of a long run; the summary's epoch cannot.
    last = max(last_hist, int(summary_epoch)) if isinstance(summary_epoch, (int, float)) else last_hist
    complete = max_epochs is not None and last >= max_epochs - 1
    stopped_early = (not complete) and state == "finished"
    w = min(200, max(20, round(max_epochs / 6))) if max_epochs else None
    final = None
    win = None
    if complete:
        win = (max_epochs - w, max_epochs - 1)
        pts = [v for e, v in val if win[0] <= e <= win[1]]
        final = st.mean(pts) if pts else None
    # END WINDOW for runs that ended normally before their budget (EarlyStopping on val/loss
    # with patience 20 was the trainer default through July; time limits later). The mean over
    # the last min(200, max(5, round((stop + 1) / 6))) epochs up to the stop. It is a
    # different statistic from the budget window and is always labeled as such.
    end = end_win = None
    if stopped_early and val:
        stop = last_hist
        we = min(200, max(5, round((stop + 1) / 6)))
        end_win = (max(0, stop - we + 1), stop)
        pts = [v for e, v in val if end_win[0] <= e <= end_win[1]]
        end = st.mean(pts) if pts else None
    # A run can pass epoch 1,199 with no logged (or sampled) val point inside 1,000 to 1,199,
    # e.g. a long budget thinned by the sample cap or a sparse validation interval. The window
    # is then undefined, and the reason is recorded rather than filled.
    reg_pts = [v for e, v in val if REGISTERED[0] <= e <= REGISTERED[1]]
    reg = st.mean(reg_pts) if last >= REGISTERED[1] and reg_pts else None
    reg_note = None if reg is not None else ("did not reach epoch 1199" if last < REGISTERED[1] else "no val point logged or sampled in epochs 1000 to 1199")
    final_note = None
    if complete and final is None:
        final_note = "no val point logged or sampled in the final window"
    collapsed = launch = None
    longest = 0
    if h["ratio_key"]:
        sp = h["history"][h["ratio_key"]]
        launch = next((e for e, v in sp if v >= LAUNCH), None)
        start = None
        for e, v in sp:
            if launch is not None and e > launch and v < DEAD:
                start = e if start is None else start
                longest = max(longest, e - start + 1)
            else:
                start = None
        collapsed = longest >= RULE
    return {
        "metric_key": h["key"],
        "n_points": len(val),
        "last_epoch": last,
        "complete": complete,
        "stopped_early": stopped_early,
        "end_window": end,
        "end_window_epochs": end_win,
        "final_window": final,
        "final_window_epochs": win,
        "final_window_note": final_note,
        "registered_window": reg,
        "registered_window_note": reg_note,
        "raw_max_sampled": max(v for _, v in val) if val else None,
        "last_value": val[-1][1] if val else None,
        "launch_epoch": launch,
        "longest_dead_span_epochs": longest if h["ratio_key"] else None,
        "collapsed": collapsed,
        "never_launched": (launch is None) if h["ratio_key"] else None,
    }


def summarize(v: list[float]) -> dict[str, Any]:
    return {"n": len(v), "mean": st.mean(v) if v else None, "sd": st.stdev(v) if len(v) > 1 else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=osp.join(RESULTS, "_leaderboard_cache", "wandb_campaign_census"))
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--from-cache", action="store_true", help="summarize the cached pulls only (no history requests)")
    args = ap.parse_args()
    api = wandb.Api(timeout=180)

    raw: list[dict[str, Any]] = []
    listing: dict[str, Any] = {}
    for project in PROJECTS:
        if args.from_cache:
            # Replay: summarize the cached raw pulls without touching the API for histories.
            d = osp.join(args.cache_dir, project)
            got = [json.load(open(osp.join(d, f))) for f in sorted(os.listdir(d))]
            raw += got
            listing[project] = {"n_cached": len(got), "note": "replayed from cache; listing not re-queried"}
            continue
        runs = list(api.runs(f"{ENTITY}/{project}", per_page=500, order="+created_at"))
        # Count check: the same query in the opposite order must return the same id set.
        desc = {r.id for r in api.runs(f"{ENTITY}/{project}", per_page=500, order="-created_at")}
        listing[project] = {"n_listed": len(runs), "n_desc": len(desc), "id_sets_agree": desc == {r.id for r in runs}}
        with ThreadPoolExecutor(args.threads) as ex:
            got = list(ex.map(lambda r: pull(r, project, args.cache_dir), runs))
        raw += got
        print(f"{project}: {len(got)} runs (desc listing {len(desc)})", flush=True)

    infra = []
    for project in INFRA:
        runs = list(api.runs(f"{ENTITY}/{project}", per_page=500))
        states: dict[str, int] = {}
        for r in runs:
            states[r.state] = states.get(r.state, 0) + 1
        infra.append({"project": project, "n_runs": len(runs), "states": states,
                      "first": min(r.created_at for r in runs), "last": max(r.created_at for r in runs)})

    rows = []
    for r in raw:
        heads = {h: score_head(v, r["config"]["max_epochs"], r["state"], r["summary_epoch"]) for h, v in r["heads"].items()}
        rows.append({k: v for k, v in r.items() if k != "heads"} | {"phase": phase_of(r["created_at"]), "url": url(r["project"], r["id"]), "heads_scored": heads})

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    no_val: dict[tuple[str, str], list[str]] = {}
    for r in rows:
        if not r["heads_scored"]:
            no_val.setdefault((r["project"], r["phase"]), []).append(r["id"])
        for h in r["heads_scored"]:
            groups.setdefault((r["project"], r["phase"], h), []).append(r)
    summary = []
    for (project, phase, head), rs in groups.items():
        sc = [(r, r["heads_scored"][head]) for r in rs]
        measurable = [s for _, s in sc if s["collapsed"] is not None]
        # Rank on the budget window when any run reached its budget; otherwise on the end
        # window of runs that stopped early (labeled `score_kind` = "end"), never mixing the two.
        kind = "budget" if any(s["complete"] and s["final_window"] is not None for _, s in sc) else "end"
        field = "final_window" if kind == "budget" else "end_window"
        eligible = [(r, s) for r, s in sc if s[field] is not None and s["collapsed"] is not True]
        if not eligible:
            kind = None
        arms: dict[str, list[float]] = {}
        arm_runs: dict[str, list[str]] = {}
        for r, s in eligible:
            arms.setdefault(r["arm"], []).append(s[field])
            arm_runs.setdefault(r["arm"], []).append(r["id"])
        best = max(arms, key=lambda a: st.mean(arms[a])) if arms else None
        top = max(eligible, key=lambda t: t[1][field], default=(None, None))
        mx = max(((r, s) for r, s in sc if s["raw_max_sampled"] is not None), key=lambda t: t[1]["raw_max_sampled"], default=(None, None))
        windows = sorted({tuple(s["final_window_epochs"]) for _, s in sc if s["final_window_epochs"]})
        summary.append({
            "project": project, "phase": phase, "head": head,
            "metric_keys": sorted({s["metric_key"] for _, s in sc}),
            "dataset_tags": sorted({str(r["config"]["dataset_tag"]) for r, _ in sc}),
            "global_output_dims": sorted({str(r["config"]["global_output_dim"]) for r, _ in sc}) if head == "morphology" else None,
            "n_runs": len(sc),
            "n_complete": sum(s["complete"] for _, s in sc),
            "n_stopped_early": sum(s["stopped_early"] for _, s in sc),
            "states": {k: sum(r["state"] == k for r, _ in sc) for k in sorted({r["state"] for r, _ in sc})},
            "median_last_epoch": st.median(s["last_epoch"] for _, s in sc),
            "max_epochs_values": sorted({r["config"]["max_epochs"] for r, _ in sc if r["config"]["max_epochs"] is not None}),
            "score_kind": kind,
            "n_collapse_measurable": len(measurable),
            "n_collapsed": sum(bool(s["collapsed"]) for _, s in sc),
            "n_never_launched": sum(bool(s["never_launched"]) for _, s in sc),
            "final_windows_used": windows,
            "n_arms_compared": len(arms),
            "best_arm": best,
            "best_arm_window": summarize(arms[best]) if best else None,
            "best_arm_runs": [url(project, i) for i in arm_runs[best]] if best else [],
            "best_single_run": {"id": top[0]["id"], "url": top[0]["url"], "arm": top[0]["arm"], field: top[1][field]} if top[0] else None,
            "raw_max_sampled": {"id": mx[0]["id"], "url": mx[0]["url"], "value": mx[1]["raw_max_sampled"], "epoch_last": mx[1]["last_epoch"],
                                "note": "upward-biased order statistic over sampled points; not a score"} if mx[0] else None,
            "first_created": min(r["created_at"] for r, _ in sc),
            "last_created": max(r["created_at"] for r, _ in sc),
        })
    order = {p: i for i, p in enumerate(PROJECTS)}
    ph = {"before": 0, "between": 1, "after": 2}
    summary.sort(key=lambda s: (ph[s["phase"]], order[s["project"]], s["head"]))

    for s in summary:
        b = s["best_arm_window"]
        print(f"{s['phase']:<8}{s['project'][14:]:<30}{s['head'][:5]:<6}runs {s['n_runs']:>4} done {s['n_complete']:>4} stop {s['n_stopped_early']:>4} "
              f"medlast {s['median_last_epoch']:>6} coll {s['n_collapsed']:>3}/{s['n_collapse_measurable']:<4} {str(s['score_kind']):<6} best {str(s['best_arm'])[:16]:<16} "
              + (f"{b['mean']:.4f} (n={b['n']}, arms {s['n_arms_compared']})" if b else "--")
              + (f" rawmax {s['raw_max_sampled']['value']:.4f}" if s["raw_max_sampled"] else ""))

    with open(osp.join(RESULTS, "wandb_campaign_census.json"), "w") as f:
        json.dump({
            "generated_by": "experiments/019-simb-multimodal/scripts/wandb_campaign_census.py",
            "entity": ENTITY,
            "phases": {"before": BEFORE, "after_from": AFTER_FROM, "between": "anything else"},
            "head_keys": HEAD_KEYS,
            "score": "final_window = mean val Pearson per feature over the last min(200, max(20, round(max_epochs / 6))) epochs of the budget, complete runs only; end_window (score_kind end) = mean over the last min(200, max(5, round((stop+1) / 6))) epochs of runs that finished before their budget, used only for rows where no run reached its budget; collapsed runs excluded",
            "registered_window": REGISTERED,
            "rule": {"launch": LAUNCH, "dead": DEAD, "collapse_span_epochs": RULE},
            "history_sampling": f"run.history(keys=['epoch', key], samples={SAMPLES}) per (run, key); long runs are thinned",
            "listing": listing,
            "no_val_metric_runs": {f"{p}|{ph_}": ids for (p, ph_), ids in no_val.items()},
            "infrastructure": infra,
            "summary": summary,
            "runs": rows,
        }, f, indent=1)

    write_table(summary, infra, no_val)


def tex(s: str) -> str:
    return s.replace("_", "\\_")


def write_table(summary: list[dict[str, Any]], infra: list[dict[str, Any]], no_val: dict[tuple[str, str], list[str]]) -> None:
    # Bold the largest budget-window value per head; end-window values (dagger) are a different
    # statistic and are never bolded.
    best_of: dict[str, dict[str, Any]] = {}
    for s in summary:
        b = s["best_arm_window"]
        if b and s["score_kind"] == "budget" and (s["head"] not in best_of or b["mean"] > best_of[s["head"]]["best_arm_window"]["mean"]):
            best_of[s["head"]] = s
    short = {"expression": "expr", "proteome": "prot", "morphology": "morph"}
    lines = [
        "%% GENERATED by experiments/019-simb-multimodal/scripts/wandb_campaign_census.py. Do not edit by hand.",
        f"%% SOURCE: W&B zhao-group, every torchcell_019* project (sampled run histories, samples={SAMPLES} per key)",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\footnotesize",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\caption[W\\&B census of the 019 campaign]{Every W\\&B run of the 019 campaign, one row per project, phase and"
        " scored head (project names drop the prefix \\texttt{torchcell\\_019\\_}; \\texttt{cgt\\_multitask} is"
        " \\texttt{torchcell\\_019-simb-multimodal\\_cgt\\_multitask}). Phase is the run's creation date (UTC): before SIMB"
        " (2026-07-22 to 07-31), between (August to 09-02) and after (09-03 onward). Complete means the run reached the"
        " last epoch of its own budget; stopped means it ended normally before that (EarlyStopping on the validation loss,"
        " patience 20, was the trainer default through July; time limits later). Collapsed counts runs whose validation"
        " prediction spread ratio, after first reaching 0.05, stayed below 0.01 over 50 or more epochs, out of the runs"
        " that log the ratio (n/a: never logged). The score is a window mean of the validation Pearson per feature, never"
        " a maximum: over the last $\\min(200, \\max(20, E/6))$ epochs of a budget of $E$ epochs for complete runs"
        " (epochs 1{,}000 to 1{,}199 for the 1{,}200-epoch rounds), or, marked $\\dagger$ where no run of the row reached"
        " its budget, over the last $\\min(200, \\max(5, S/6))$ epochs before a stop at epoch $S$. Collapsed runs are"
        " excluded. The best arm has the highest mean over its runs among the arms shown in parentheses, a selected"
        " maximum; the bold value is the largest budget-window value per head and not a like-for-like winner (stores,"
        " feature counts and conditioning differ between rows; see the text)."
        " \\src{experiments/019-simb-multimodal/scripts/wandb_campaign_census.py}}",
        "  \\label{tab:wandb-campaign-census}",
        "  \\begin{tabular}{@{}" + "".join(f">{{\\raggedright\\arraybackslash}}p{{{w}mm}}" if w else "r" for w in (25, 11, 9, 0, 13, 13, 27, 21, 17)) + "@{}}",
        "    \\toprule",
        "    project & phase & head & runs (n) & complete / stopped (n) & collapsed (n of measurable)"
        " & best arm by window mean val Pearson per feature (arms compared) & window mean val Pearson per feature,"
        " best arm (n runs) & created, first--last (UTC, MM-DD) \\\\",
        "    \\midrule",
    ]
    cur = None
    for s in summary:
        if cur is not None and s["phase"] != cur:
            lines.append("    \\midrule")
        cur = s["phase"]
        b = s["best_arm_window"]
        val = f"{b['mean']:.3f} ({b['n']})" if b else "--"
        if b and s["score_kind"] == "end":
            val += "$^\\dagger$"
        if b and best_of.get(s["head"]) is s:
            val = f"\\textbf{{{val}}}"
        coll = f"{s['n_collapsed']} of {s['n_collapse_measurable']}" if s["n_collapse_measurable"] else "n/a"
        arm = f"\\texttt{{{tex(str(s['best_arm'])[:20])}}} ({s['n_arms_compared']})" if s["best_arm"] else "--"
        name = s["project"].replace("torchcell_019-simb-multimodal_", "").replace("torchcell_019_", "")
        dates = f"{s['first_created'][5:10]}--{s['last_created'][5:10]}"
        lines.append(f"    \\texttt{{{tex(name)}}} & {s['phase']} & {short[s['head']]} & {s['n_runs']} & {s['n_complete']} / {s['n_stopped_early']}"
                     f" & {coll} & {arm} & {val} & {dates} \\\\")
    unscored = sorted({p for (p, _) in no_val} - {s["project"] for s in summary})
    lines.append("    \\midrule")
    n_inf = sum(i["n_runs"] for i in infra)
    names = ", ".join(tex(i["project"].replace("torchcell_019_", "")) for i in infra)
    lines.append(f"    infrastructure ({names}) & after & -- & {n_inf} & -- & -- & not scored (benchmarks and profiling) & -- & "
                 f"{min(i['first'] for i in infra)[5:10]}--{max(i['last'] for i in infra)[5:10]} \\\\")
    for p in unscored:
        n = sum(len(ids) for (q, _), ids in no_val.items() if q == p)
        lines.append(f"    \\texttt{{{tex(p.replace('torchcell_019_', ''))}}} & -- & -- & {n} & -- & -- & no validation Pearson logged & -- & -- \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]
    with open(osp.join(TABLES, "wandb_campaign_census.tex"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
