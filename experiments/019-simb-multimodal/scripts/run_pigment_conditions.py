# experiments/019-simb-multimodal/scripts/run_pigment_conditions.py
# [[experiments.019-simb-multimodal.scripts.run_pigment_conditions]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/run_pigment_conditions
"""Run the four pigment metabolome-transfer conditions and compute the two Deltas.

THE QUESTION. Does adding the Mulleder 19-amino-acid metabolome as an auxiliary task
improve production prediction -- and, decisively, does it help betaxanthin MORE than
beta-carotene?

    Delta_betaxanthin   = r(A2) - r(A1)      betaxanthin  + metabolome  vs  alone
    Delta_beta_carotene = r(A4) - r(A3)      beta-carotene + metabolome vs  alone

The hypothesis predicts an ASYMMETRY, not merely a lift: TYROSINE is betaxanthin's direct
precursor and is one of Mulleder's 19 measured amino acids, and the Cachera cassette
carries feedback-resistant ARO4-K229L / ARO7-G141S precisely to push tyrosine flux.
Beta-carotene's precursors (acetyl-CoA, GGPP) are measured NOWHERE genome-wide. So a lift
on betaxanthin with a null on beta-carotene is evidence that transfer travels through the
shared precursor rather than through generic multitask regularization; a lift on both
would mean the gain is generic; a null on both is a clean negative.

The four conditions differ ONLY in `multitask.active_heads`; every other config value,
including the split seed, comes from one shared base config (`gh_pigment_base.yaml`) and
is composed here rather than duplicated into four YAML files. An unmatched split
invalidates a Delta, so identity is enforced by construction, not by inspection.

METRICS. Pearson AND Spearman per target, taken at their PEAK over validation epochs
(``{metric}_max``) rather than at the last epoch -- these runs peak early and then
collapse toward the per-feature mean under MSE, so the last epoch understates the signal
(the same `BestMetricTracker` convention the rest of 019 uses). SPEARMAN IS PRIMARY FOR
BETA-CAROTENE: it is a subjective ordinal colony-colour score, so rank agreement is the
only meaningful notion of accuracy, and it is what the noise ceiling is measured in
(``pigment_noise_ceiling.py``).

Run (GPU 0 only -- GPUs 1-3 are held by another job)::

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python
        experiments/019-simb-multimodal/scripts/run_pigment_conditions.py

Env:
  PIGMENT_FAST_DEV_RUN=1  -- fit-check only: one train/val batch per condition, proving
                             every head trains at all before committing to full runs.
  PIGMENT_MAX_EPOCHS=<n>  -- override the base config's max_epochs.
  PIGMENT_CONDITIONS=A1,A3 -- run a subset.
  PIGMENT_SEEDS=42,7,1234 -- seeds to replicate over (each reshuffles split AND init).
  PIGMENT_OVERRIDES=k=v,... -- extra Hydra overrides applied to every condition.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from train_cgt_multitask import (  # type: ignore[import-not-found]  # noqa: E402
    run_training,
)

CONF_DIR = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), "..", "conf"))
BASE_CONFIG = "gh_pigment_base"

# condition -> (active heads, the production target whose metric is the outcome)
CONDITIONS: dict[str, tuple[list[str], str]] = {
    "A1": (["betaxanthin"], "betaxanthin"),
    "A2": (["betaxanthin", "mulleder19"], "betaxanthin"),
    "A3": (["beta_carotene"], "beta_carotene"),
    "A4": (["beta_carotene", "mulleder19"], "beta_carotene"),
}
# Spearman is primary for the ordinal beta-carotene score; Pearson for betaxanthin.
PRIMARY_METRIC = {"betaxanthin": "pearson", "beta_carotene": "spearman"}


def _metric(
    metrics: dict[str, float], head: str, kind: str, peak: bool
) -> float | None:
    """Read ``val/<head>/<kind>_per_feature`` (peak or last) from a run's metrics."""
    key = f"val/{head}/{kind}_per_feature"
    if peak:
        key = f"{key}_max"
    return metrics.get(key)


def run_condition(
    name: str, seed: int, max_epochs: int | None, fast_dev_run: bool
) -> dict[str, Any]:
    """Compose the base config with this condition's heads and train it."""
    heads, target = CONDITIONS[name]
    overrides = [
        f"multitask.active_heads=[{','.join(heads)}]",
        f"seed={seed}",
        f"wandb.tags=[ws11b,gilahyper,pigment,{name},seed{seed}]",
    ]
    if max_epochs is not None:
        overrides.append(f"trainer.max_epochs={max_epochs}")
    if fast_dev_run:
        overrides.append("trainer.fast_dev_run=true")
        overrides.append("trainer.early_stopping.enabled=false")
    # Extra Hydra overrides applied to EVERY condition, so they can never desynchronize
    # the four (e.g. PIGMENT_OVERRIDES="data_module.batch_size=128,seed=7").
    extra = os.environ.get("PIGMENT_OVERRIDES")
    if extra:
        overrides.extend(extra.split(","))
    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        cfg = compose(config_name=BASE_CONFIG, overrides=overrides)
    print("=" * 78)
    print(f"CONDITION {name} seed={seed}: active_heads={heads}  target={target}")
    print("=" * 78)
    metrics = run_training(cfg)
    out: dict[str, Any] = {
        "condition": name,
        "active_heads": heads,
        "target": target,
        "seed": int(cfg.seed),
        "n_epochs_run": metrics.get("epoch"),
    }
    for head in heads:
        for kind in ("pearson", "spearman"):
            out[f"{head}_{kind}_peak"] = _metric(metrics, head, kind, peak=True)
            out[f"{head}_{kind}_last"] = _metric(metrics, head, kind, peak=False)
    out["val_loss_last"] = metrics.get("val/loss")
    out["_all_metrics"] = {
        k: v for k, v in metrics.items() if k.startswith(("val/", "train/"))
    }
    return out


def _checkpoint_path(results_dir: str, fast_dev_run: bool) -> str:
    """Path of the incremental per-run store."""
    suffix = "_fast_dev_run" if fast_dev_run else ""
    return osp.join(results_dir, f"pigment_transfer_runs{suffix}_partial.json")


def _load_checkpoint(path: str) -> dict[str, dict[str, Any]]:
    """Load completed ``{seed: {condition: record}}`` runs, or an empty store."""
    if not osp.exists(path):
        return {}
    with open(path) as f:
        return dict(json.load(f))


def _save_checkpoint(path: str, store: dict[str, dict[str, Any]]) -> None:
    """Persist the per-run store after EVERY completed run.

    A 12-run sweep is hours long, and the aggregate report is only written at the end --
    so a single interruption used to discard every finished run. (It did: an external kill
    at run 8 lost seven completed runs whose metrics existed only in memory.) Each record
    is written as soon as it exists, and ``main`` skips any (seed, condition) already
    present, so a restart resumes instead of redoing the sweep.
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(store, f, indent=2)
    os.replace(tmp, path)


def _seed_deltas(
    results: dict[str, Any], peak: bool
) -> dict[str, dict[str, float | None]]:
    """Per-target Deltas within ONE seed (so both arms share one split)."""
    out: dict[str, dict[str, float | None]] = {}
    for target, (alone, joint) in (
        ("betaxanthin", ("A1", "A2")),
        ("beta_carotene", ("A3", "A4")),
    ):
        if alone not in results or joint not in results:
            continue
        entry: dict[str, float | None] = {}
        suffix = "peak" if peak else "last"
        for kind in ("pearson", "spearman"):
            a = results[alone].get(f"{target}_{kind}_{suffix}")
            j = results[joint].get(f"{target}_{kind}_{suffix}")
            entry[f"{kind}_alone"] = None if a is None else float(a)
            entry[f"{kind}_joint"] = None if j is None else float(j)
            entry[f"delta_{kind}"] = (
                None if a is None or j is None else float(j) - float(a)
            )
        entry["delta_primary"] = entry[f"delta_{PRIMARY_METRIC[target]}"]
        out[target] = entry
    return out


def _mean_sd(values: list[float]) -> dict[str, float | int | None]:
    """Mean, sample SD and n of a list (SD is None below 2 observations)."""
    n = len(values)
    if n == 0:
        return {"mean": None, "sd": None, "n": 0}
    mean = sum(values) / n
    if n < 2:
        return {"mean": mean, "sd": None, "n": n}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return {"mean": mean, "sd": var**0.5, "n": n}


def main() -> None:
    """Run every (seed, condition), compute both Deltas, write the results JSON."""
    fast_dev_run = os.environ.get("PIGMENT_FAST_DEV_RUN") == "1"
    max_epochs_env = os.environ.get("PIGMENT_MAX_EPOCHS")
    max_epochs = int(max_epochs_env) if max_epochs_env else None
    wanted = os.environ.get("PIGMENT_CONDITIONS")
    names = wanted.split(",") if wanted else list(CONDITIONS)
    seeds = [int(s) for s in os.environ.get("PIGMENT_SEEDS", "42").split(",")]

    # Several seeds because the val split is only ~345 genotypes: a Pearson there carries
    # an SE around 0.05, so a single-seed Delta of a few hundredths is indistinguishable
    # from noise. Each seed reshuffles BOTH the split and the model init; within a seed
    # all four conditions share one split exactly, which is what makes each Delta paired.
    here = osp.dirname(osp.abspath(__file__))
    results_dir = osp.abspath(osp.join(here, "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    ckpt = _checkpoint_path(results_dir, fast_dev_run)
    all_results: dict[str, dict[str, Any]] = _load_checkpoint(ckpt)
    peak = not fast_dev_run  # fast_dev_run disables callbacks, so no peak is recorded
    for seed in seeds:
        per_seed = all_results.setdefault(str(seed), {})
        for name in names:
            if name in per_seed:
                print(f"[resume] seed {seed} {name} already complete -- skipping")
                continue
            per_seed[name] = run_condition(name, seed, max_epochs, fast_dev_run)
            _save_checkpoint(ckpt, all_results)

    report: dict[str, Any] = {
        "base_config": BASE_CONFIG,
        "fast_dev_run": fast_dev_run,
        "seeds": seeds,
        "metric_taken_at": "peak" if peak else "last",
        "primary_metric_per_target": PRIMARY_METRIC,
        "runs": all_results,
    }

    per_seed_deltas = {s: _seed_deltas(r, peak) for s, r in all_results.items()}
    report["deltas_per_seed"] = per_seed_deltas

    summary: dict[str, Any] = {}
    for target in ("betaxanthin", "beta_carotene"):
        for kind in ("pearson", "spearman"):
            for field in (f"{kind}_alone", f"{kind}_joint", f"delta_{kind}"):
                vals = [
                    float(d[target][field])  # type: ignore[arg-type]
                    for d in per_seed_deltas.values()
                    if target in d and d[target].get(field) is not None
                ]
                summary.setdefault(target, {})[field] = _mean_sd(vals)
        prim = [
            float(d[target]["delta_primary"])  # type: ignore[arg-type]
            for d in per_seed_deltas.values()
            if target in d and d[target].get("delta_primary") is not None
        ]
        summary.setdefault(target, {})["delta_primary"] = _mean_sd(prim)
        summary[target]["primary_metric"] = PRIMARY_METRIC[target]
    report["summary_across_seeds"] = summary

    if "betaxanthin" in summary and "beta_carotene" in summary:
        d_bx = summary["betaxanthin"]["delta_primary"]["mean"]
        d_bc = summary["beta_carotene"]["delta_primary"]["mean"]
        report["headline"] = {
            "delta_betaxanthin_primary_mean": d_bx,
            "delta_betaxanthin_primary_sd": summary["betaxanthin"]["delta_primary"][
                "sd"
            ],
            "delta_beta_carotene_primary_mean": d_bc,
            "delta_beta_carotene_primary_sd": summary["beta_carotene"]["delta_primary"][
                "sd"
            ],
            "asymmetry": (
                None if d_bx is None or d_bc is None else float(d_bx) - float(d_bc)
            ),
            "hypothesis_direction_holds": (
                None if d_bx is None or d_bc is None else bool(d_bx > d_bc)
            ),
        }

    suffix = "_fast_dev_run" if fast_dev_run else ""
    out = osp.join(results_dir, f"pigment_transfer_runs{suffix}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 78)
    print("PIGMENT METABOLOME-TRANSFER RESULTS")
    print("=" * 78)
    tag = "peak" if peak else "last"
    for seed_key, res in all_results.items():
        for name, r in res.items():
            target = r["target"]
            p = r.get(f"{target}_pearson_{tag}")
            s = r.get(f"{target}_spearman_{tag}")
            print(
                f"  seed {seed_key:>5s}  {name} {str(r['active_heads']):40s} {target:14s} "
                f"pearson {p if p is None else round(float(p), 4)}  "
                f"spearman {s if s is None else round(float(s), 4)}"
            )
    print("\nsummary across seeds:")
    print(json.dumps(report.get("summary_across_seeds", {}), indent=2))
    print("\nheadline:")
    print(json.dumps(report.get("headline", {}), indent=2))
    print(f"\nWrote {out}")
    print(OmegaConf.to_yaml({"base_config": BASE_CONFIG}))


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
