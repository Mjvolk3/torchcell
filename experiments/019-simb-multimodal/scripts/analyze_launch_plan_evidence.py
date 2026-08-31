# experiments/019-simb-multimodal/scripts/analyze_launch_plan_evidence.py
# [[experiments.019-simb-multimodal.scripts.analyze_launch_plan_evidence]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/analyze_launch_plan_evidence
"""Evidence behind the 2026-08-31 expression launch plan.

Four measured inputs, one JSON output (results/launch_plan_evidence.json):

1. REPLICATE SPREAD at long budget. Every expression run past 9,000 epochs in
   round_leaderboards.csv shares one config (quantile head, lr 3e-4, dropout 0.1,
   L=6, hidden 90, mask prior, s1_pool, seed 0), so they are replicates. Their
   spread is the noise floor any arm comparison at this budget must beat, and the
   headline 0.2382 is the maximum of those draws, an upward-biased order statistic.
2. POWER. Replicates per arm needed to detect a given gap at that spread
   (two-sample and vs the existing n=8 baseline; 80% power, alpha 0.05).
3. WALL-CLOCK BUDGET. s/epoch (compute and wall) per card class from the same
   eight runs, converted to epochs per 5-day partition cap.
4. PACKING COST. Parsed from the raw logs of gh_packing_benchmark.sh (solo on one
   GilaHyper GPU vs two concurrent runs on another, same config resumed from the
   1445 checkpoint, train_eval_every=0), stored under results/packing_benchmark/ (.txt, since *.log is gitignored).

Also: parameter count of the incumbent model, summed from the 1445-last.ckpt
state dict and broken down by module prefix.

Run from the repo root:
    python experiments/019-simb-multimodal/scripts/analyze_launch_plan_evidence.py
"""

import json
import os.path as osp
import re
from collections import defaultdict

import pandas as pd
import torch
from scipy.stats import norm

EXP = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))))
RESULTS = osp.join(EXP, "results")
LEADERBOARD = osp.join(RESULTS, "round_leaderboards.csv")
PACK_DIR = osp.join(RESULTS, "packing_benchmark")
CKPT = (
    "/scratch/projects/torchcell-scratch/models/checkpoints/"
    "gilahyper-1445_447aa95e06414b55667966ee1f9a489200ccd9d2a8782fc216a61a74c9baaafb/"
    "1445-last.ckpt"
)
FIVE_DAYS_S = 5 * 86400
PACK_EPOCH_RE = re.compile(r"^Epoch (\d+): 100%\|[^|]*\| 39/39 \[(\d{2}):(\d{2})<")
# The wandb-logged trainable-parameter count for the incumbent config; the
# checkpoint state dict must reconcile to this exactly or the breakdown is wrong.
WANDB_TOTAL_PARAM_COUNT = 1_194_509
# Buffers (non-trainable) in the state dict, excluded from the parameter count.
BUFFER_KEYS = {"_norm_mean_per_gene", "_norm_std_per_gene", "loss.dist_heads.per_gene._taus"}


def replicate_spread(df: pd.DataFrame) -> dict:
    e = df[df.strand.astype(str).str.contains("expr", case=False, na=False)]
    n = e[e.epochs >= 9000].copy()
    # These runs must be one config for the replicate reading to hold; assert it.
    for col, want in [
        ("dist", "quantile"),
        ("lr", 0.0003),
        ("dropout", 0.1),
        ("num_layers", 6.0),
        ("hidden_channels", 90.0),
        ("graph_prior", "mask"),
        ("decoder", "s1_pool"),
        ("seed", 0.0),
    ]:
        vals = set(n[col])
        if vals != {want}:
            raise ValueError(f"9,000+ epoch runs are not one config: {col}={vals}")
    x = n.primary_roll_max
    sd = float(x.std(ddof=1))
    # Blom's approximation for the expected maximum of len(x) iid normal draws.
    z_max = float(norm.ppf((len(x) - 0.375) / (len(x) + 0.25)))
    return {
        "n": int(len(x)),
        "run_ids": sorted(n.run_id),
        "scores": sorted(round(v, 4) for v in x),
        "mean": round(float(x.mean()), 4),
        "sd": round(sd, 4),
        "min": round(float(x.min()), 4),
        "max": round(float(x.max()), 4),
        "expected_max_blom": round(float(x.mean()) + sd * z_max, 4),
        "observed_max_z": round((float(x.max()) - float(x.mean())) / sd, 2),
        "peak_epochs": sorted(int(v) for v in n.primary_epoch_at_roll_max),
        "power_80pct_alpha05": {
            # Two-sample n per arm: 2 * ((z_{1-a/2} + z_{0.8}) * sd / delta)^2,
            # with z_{0.975} + z_{0.8} = 1.960 + 0.842 = 2.802.
            f"delta_{d}": round(2 * (2.802 * sd / d) ** 2, 1)
            for d in (0.02, 0.03, 0.04, 0.05)
        },
        # Smallest detectable gap for k new replicates vs the n=8 baseline:
        # delta = 2.802 * sd * sqrt(1/k + 1/8).
        "detectable_vs_n8_baseline": {
            f"k_{k}": round(2.802 * sd * (1 / k + 1 / len(x)) ** 0.5, 4)
            for k in (2, 3, 4, 6, 7)
        },
    }


def wall_clock(df: pd.DataFrame) -> dict:
    e = df[df.strand.astype(str).str.contains("expr", case=False, na=False)]
    n = e[e.epochs >= 9000].copy()
    n["wall_s_ep"] = n.runtime_h * 3600 / n.epochs
    # perf_epoch_seconds is cleanly bimodal at identical config: the compute-time
    # clusters are the two card classes (RTX 6000 Ada ~17 s vs A100 ~27 s).
    n["card"] = ["ada" if v < 22 else "a100" for v in n.perf_epoch_seconds]
    out = {}
    for card, g in n.groupby("card"):
        w = float(g.wall_s_ep.median())
        out[card] = {
            "n_runs": int(len(g)),
            "compute_s_per_epoch_median": round(float(g.perf_epoch_seconds.median()), 1),
            "wall_s_per_epoch_median": round(w, 1),
            "epochs_in_5_days_solo": int(FIVE_DAYS_S / w),
            "days_for_9900_epochs_solo": round(9900 * w / 86400, 2),
        }
    return out


def packing(pack_dir: str) -> dict:
    def epoch_seconds(path: str) -> list[int]:
        # The progress bar redraws its 100% line several times per epoch (before
        # and after validation); the per-epoch MAX is the full wall time.
        per_epoch: dict[int, int] = defaultdict(int)
        with open(path) as f:
            for line in f.read().replace("\r", "\n").splitlines():
                m = PACK_EPOCH_RE.match(line)
                if m:
                    ep = int(m.group(1))
                    s = int(m.group(2)) * 60 + int(m.group(3))
                    per_epoch[ep] = max(per_epoch[ep], s)
        if not per_epoch:
            raise ValueError(f"no epoch lines parsed from {path}")
        return [per_epoch[e] for e in sorted(per_epoch)]

    solo = epoch_seconds(osp.join(pack_dir, "solo.txt"))
    packed = [
        epoch_seconds(osp.join(pack_dir, f"packed_{i}.txt")) for i in (1, 2)
    ]
    med = lambda v: float(pd.Series(v).median())  # noqa: E731
    solo_med = med(solo)
    packed_med = med(packed[0] + packed[1])
    mem = open(osp.join(pack_dir, "gpu1_mem.txt")).read()
    gpu1 = [ln for ln in mem.splitlines() if ln.startswith("1,")][0]
    both_mib = int(gpu1.split(",")[1].strip().split()[0])
    return {
        "solo_median_s_per_epoch": solo_med,
        "packed_median_s_per_epoch": packed_med,
        "n_epochs_measured": {"solo": len(solo), "packed_each": len(packed[0])},
        "per_run_slowdown": round(packed_med / solo_med, 2),
        "aggregate_speedup": round(2 * solo_med / packed_med, 2),
        "vram_mib_both_packed_runs": both_mib,
        "vram_gib_per_run": round(both_mib / 2 / 1024, 1),
    }


def param_breakdown(ckpt_path: str) -> dict:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    groups: dict[str, int] = defaultdict(int)
    seen: set[int] = set()
    aliased = 0
    total = 0
    for key, tensor in sd.items():
        n = tensor.numel()
        if key in BUFFER_KEYS:
            continue
        # `EquivariantPerturbationTransform` aliases layer 0 of its ModuleLists
        # (`self.cross_attn = self.cross_attn_layers[0]` and siblings), so the
        # same tensors appear under two state-dict keys; count each storage once.
        ptr = tensor.data_ptr()
        if ptr in seen:
            aliased += n
            continue
        seen.add(ptr)
        total += n
        parts = key.split(".")
        # Group by the module directly under the LightningModule's `model.`.
        prefix = ".".join(parts[:2]) if parts[0] == "model" else parts[0]
        groups[prefix] += n
    if total != WANDB_TOTAL_PARAM_COUNT:
        raise ValueError(
            f"deduped checkpoint total {total} != wandb total_param_count "
            f"{WANDB_TOTAL_PARAM_COUNT}"
        )
    return {
        "checkpoint": ckpt_path,
        "total_trainable": total,
        "matches_wandb_total_param_count": True,
        "aliased_params_excluded": aliased,
        "buffer_params_excluded": sum(
            t.numel() for k, t in sd.items() if k in BUFFER_KEYS
        ),
        "by_module": dict(sorted(groups.items(), key=lambda kv: -kv[1])),
    }


def main() -> None:
    df = pd.read_csv(LEADERBOARD)
    out = {
        "generated_by": "experiments/019-simb-multimodal/scripts/analyze_launch_plan_evidence.py",
        "replicate_spread_9900_epochs": replicate_spread(df),
        "wall_clock_by_card": wall_clock(df),
        "packing_benchmark": packing(PACK_DIR),
        "param_count": param_breakdown(CKPT),
    }
    dst = osp.join(RESULTS, "launch_plan_evidence.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
