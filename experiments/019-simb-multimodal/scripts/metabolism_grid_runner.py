# experiments/019-simb-multimodal/scripts/metabolism_grid_runner.py
# [[experiments.019-simb-multimodal.scripts.metabolism_grid_runner]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/metabolism_grid_runner
"""EIGHT SETTINGS, EACH TRAINED FOR THE WHOLE ALLOCATION -- the Delta metabolism runs.

This REPLACES the Optuna search (`optuna_metabolism_sweep.py`) for the Delta jobs. Two
measurements force the shape, and the second one is the important one.

(1) MORE SEARCH CANNOT HELP -- the arms are noisier than the gaps being ranked
-----------------------------------------------------------------------------
Replicate noise on `pearson_per_feature` is sigma = 0.030 (betaxanthin) and 0.025
(beta-carotene), measured directly from `_002` trials that re-proposed identical
hyperparameters AND an identical seed:

    betaxanthin    trials  3/10/11   0.4216 / 0.4095 / 0.3584     sigma = 0.030
    beta_carotene  trials 18/22      0.1993 / 0.1494              sigma = 0.025

`_002`'s top FIVE betaxanthin configs spanned 0.021 -- 0.68 sigma. A single-seed search can
rank AXES; it cannot pick a WINNER, and a third one would buy more unreadable trials. So the
budget goes to eight FIXED settings instead of a search.

(2) EVERY METABOLISM RUN SO FAR MAY HAVE BEEN SCORED IN THE DIP
---------------------------------------------------------------
This is the reason the runs are LONG rather than merely replicated. From the 019 expression
wave-6 measurements ([[scratch.2026.07.30.132219-exp-design]]), on the same model and the
same trainer:

    * Smoothed val Pearson peaks ~0.14 at epoch 85-136, FALLS to 0.08-0.11 by epoch 200-300,
      then RISES to a project-best 0.1980 at epoch 1367.
      "Every arm ever scored at 300-400 epochs was scored in the dip."
    * Nothing had converged at 1,500 epochs (eval-mode train Pearson still climbing,
      max == last in 4 of 4 runs).
    * val LOSS and val PEARSON move in OPPOSITE directions -- loss bottoms at epoch 103-136
      and then rises while Pearson keeps climbing. So "the loss stopped improving" is not
      evidence that the metric has.

Now compare the metabolism history: every `_002`/`_003`/`_004` run early-stopped on a
patience of 40-50 and finished at a median of ~71-135 epochs. If metabolism has the same
dip-then-climb shape, ALL of those runs stopped at or before the first bump -- and the 0.4301
we have been quoting is not this model's ceiling, it is its first local maximum.

That is a hypothesis, not a fact, because it has never been tested on these targets. Testing
it is cheap and is exactly what 48 h on four A40s is for: run the eight settings LONG and
look at the shape of the curve.

    max_epochs 10,000 (a ceiling, not a target) + early stopping OFF
    + `trainer.max_time_s`, a WALL-CLOCK budget handed to a Lightning `Timer`.

The time budget is what makes this runnable. It stops training gracefully, so `fit` returns
and the metric snapshot, the test pass and the prediction dump all still happen -- where a
slurm kill mid-`fit` would lose all three and leave the trial RUNNING rather than COMPLETE.
It also spends exactly the compute available instead of an epoch cap guessed in advance.

THE 8 RUNS: FOUR SETTINGS x TWO INIT SEEDS
-------------------------------------------
A 2x2 factorial, fully crossed, replicated twice. Replication is affordable here only because
the SPLIT IS NOW PINNED (`data_module.split_seed: 0`), so `seed` varies weight initialization
alone. 019 measured what that separation buys: with one knob driving both, between-seed sd was
0.0444 against a within-seed across-arm sd of 0.0058 -- the nuisance axis was 7.7x the signal
axis, and at n=4 the minimum detectable effect was 7.9x the effect being chased. Two runs of
the same setting are now a paired comparison rather than two draws from a wide nuisance
distribution.

WHAT IT COSTS, stated plainly: the absolute level belongs to ONE validation draw. Arm
RANKINGS transfer; the number does not. If a generalizable absolute level is wanted later,
the fix is K-fold with arms paired on fold, not a different single seed.

WHAT ROUND 0 STILL DOES NOT ANSWER: a 0.02 gap between settings. At two seeds the SE is
sigma/sqrt(2) = 0.021. The primary artifact of this round is the CURVE -- where the target
saturates -- not a leaderboard.

RUNTIME MODEL
-------------
Work is enqueued ROUND-MAJOR -- round r contributes one seed to EACH setting -- so a
wall-clock kill leaves every setting with the same number of seeds and the factorial stays
balanced. Seed-major would leave 12 seeds on two settings and none on six.

Every worker computes its trial's `max_time_s` from the job deadline when it CLAIMS the
trial, so the eight round-0 runs all end together just inside the wall clock, and a
replicate started later simply gets whatever is left.

Environment (set by the slurm launcher):
    ARM                    betaxanthin | mulleder19 | beta_carotene | bx_ctrl | bx_m19
    GRID_CONF_DIR          absolute path to the arm's Hydra conf dir      (required)
    GRID_BASE_CONFIG       Hydra base config name                        (required)
    OPTUNA_STORAGE         sqlite:///.../optuna_<exp>_<arm>_grid.db       (required)
    OPTUNA_STUDY_NAME      default: <arm>_grid_000
    GRID_ROUNDS            seed rounds to enqueue (default 3)
    GRID_DEADLINE_EPOCH    unix seconds -- the job's wall clock (required for long runs)
    GRID_MIN_TRIAL_S       don't claim work with less than this left (default 3600)
    GRID_TEARDOWN_S        reserve after training for test + dump (default 600)
    OPTUNA_WORKER_ID       0..N-1, for logging only
    NUM_WORKERS            dataloader workers per trial
    GRID_MANIFEST_DIR      where --create-only writes grid_manifest_<arm>.json (optional)

Modes:
    --create-only   create the study, enqueue every (setting, seed), write the manifest.
                    Run ONCE from the launcher before the workers, so a fresh SQLite file's
                    DDL is not raced by processes that each spend >15s importing torch.
    (no flag)       run as a worker: drain the queue until it is empty or the deadline hits.
"""

import itertools
import json
import os
import os.path as osp
import sys
import time
from typing import Any

import optuna
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
_TRAINER_DIR = osp.join(_REPO_ROOT, "experiments", "019-simb-multimodal", "scripts")
if _TRAINER_DIR not in sys.path:
    sys.path.insert(0, _TRAINER_DIR)

from train_cgt_multitask import run_training  # type: ignore[import-not-found]  # noqa: E402, I001

ARM = os.environ["ARM"]
CONF_DIR = os.environ["GRID_CONF_DIR"]
BASE_CONFIG = os.environ["GRID_BASE_CONFIG"]
STORAGE = os.environ["OPTUNA_STORAGE"]
ROUNDS = int(os.getenv("GRID_ROUNDS", "1"))
WORKER_ID = int(os.getenv("OPTUNA_WORKER_ID", "0"))
#: SHARDING -- how many workers split this arm's cells, or 0 for the shared-queue mode.
#:
#: WHY THIS EXISTS: optuna's WAITING-trial pop RACES on SQLite across processes. Measured on
#: IGB job 2332400, six workers against one study file: w0 and w1 both claimed trial 0
#: (`s00_ctrl` seed 0) and w2 and w3 both claimed trial 1 (`s01_auxm19` seed 0) -- six workers
#: running THREE distinct cells. It is not new and it is not IGB-specific: the same race is
#: visible in the Delta grids, where `s03_L2_maskon_lr0.001_energy` ran FOUR times and
#: `bx_ctrl s01` twice, silently halving effective coverage while every log looked healthy.
#:
#: With sharding on, a worker owns a DISJOINT, DETERMINISTIC slice and gets its OWN study in
#: its OWN SQLite file, so there is no cross-process claim to race. The cost is that there is
#: no work stealing: a worker that finishes early idles rather than helping. That is the right
#: trade here because every cell in this grid costs the same.
SHARD_COUNT = int(os.getenv("GRID_SHARD_COUNT", "0"))
STUDY_NAME = os.getenv(
    "OPTUNA_STUDY_NAME",
    f"{ARM}_grid_000_w{WORKER_ID}" if SHARD_COUNT else f"{ARM}_grid_000",
)
DEADLINE = float(os.getenv("GRID_DEADLINE_EPOCH", "0")) or None
#: Don't claim a trial with less than this much wall clock left -- an hour of training is
#: not enough to say anything about where a curve saturates, and the slot is better left idle
#: than filled with a run nobody can read.
MIN_TRIAL_S = float(os.getenv("GRID_MIN_TRIAL_S", "3600"))
#: Reserved AFTER training for the test pass, the prediction dump and wandb teardown. Taken
#: out of the trial's time budget rather than hoped for: the whole point of the graceful
#: `Timer` stop is that those three still happen.
TEARDOWN_S = float(os.getenv("GRID_TEARDOWN_S", "600"))
#: Restrict the grid to a comma-separated subset of setting names. Empty = the whole grid.
#: For running a SLICE of the design on an opportunistic GPU -- e.g. a single freed cabbi card
#: -- without forking the config or hand-writing overrides that could drift from the grid the
#: rest of the results come from. The names must exist, and a typo RAISES rather than silently
#: enqueuing nothing, because "the job ran and produced no trials" is indistinguishable from
#: success in a slurm log.
ONLY_SETTINGS = [s for s in os.getenv("GRID_ONLY_SETTINGS", "").split(",") if s]

#: Seeds, in the order rounds consume them. These are now INIT-ONLY: the configs pin
#: `data_module.split_seed: 0`, so `cfg.seed` no longer selects the partition. Round 0 runs
#: every setting at 42 (the seed the whole `_002`/`_003`/`_004` history used), round 1
#: replicates at 0, and any further round is spare capacity for a setting that finished early.
#: ONE SEED. The partition is pinned by `data_module.split_seed`, so `seed` varies weight
#: initialization only -- and the budget goes to 24 crossed SETTINGS rather than replicates,
#: because a full factorial already estimates every main effect on 12-vs-12 runs
#: (SE = sigma*sqrt(2/12) = 0.012) without spending a single run on a repeat.
#: 42 is the seed the whole `_002`/`_003`/`_004` history used.
#:
#: `bx_aa` INVERTS THAT TRADE, deliberately: it spends its whole budget on REPLICATES of one
#: cell rather than on crossed settings, because what it must estimate is a NOISE FLOOR. The
#: grid measured `bx_m19 - bx_ctrl` at +0.0203 in the L=6/mask-on/lr=1e-4 cell (against -0.049
#: and -0.060 at L=2), and a single paired difference with no replicate cannot be told apart
#: from init noise. Every crossed factor here would be a mechanism question asked ON TOP of an
#: effect not yet shown to exist.
SEEDS = {"bx_aa": [0, 1, 2, 3, 4, 5, 6, 7]}.get(ARM, [42])

#: Heads active per arm. The two `bx_*` arms are the CONTROLLED auxiliary-task pair: same
#: restricted instance set (`require_head_targets` in their config), same settings, same
#: seeds -- differing ONLY in whether the metabolome head is attached. `bx_m19 - bx_ctrl`
#: read within a (setting, seed) cell is then a PAIRED difference, which removes the split
#: and init variance that dominates an unpaired comparison at this sample size.
#:
#: `bx_aa` is the one arm whose active heads vary PER SETTING rather than per arm -- its three
#: settings ARE the control and the two auxiliary panels -- so this entry is the SUPERSET and
#: `_active_heads(setting)` is what `_overrides` actually uses. It stays listed here because
#: `_record` and the manifest want the union.
ACTIVE_HEADS = {
    "betaxanthin": ["betaxanthin"],
    "beta_carotene": ["beta_carotene"],
    "mulleder19": ["mulleder19"],
    "bx_ctrl": ["betaxanthin"],
    "bx_m19": ["betaxanthin", "mulleder19"],
    "bx_aa": ["betaxanthin", "mulleder19"],
}[ARM]


def _active_heads(setting: dict[str, Any]) -> list[str]:
    """Heads active for ONE cell. Differs from `ACTIVE_HEADS` only on `bx_aa`.

    The `aux` factor selects the head set: `none` is the control (betaxanthin alone), while
    `m19` and `tyr` both attach the metabolome head and differ only in HOW MANY of its 19
    features are scored -- see `_overrides` for why that distinction is made with
    `drop_features` and not by re-pinning the head's phenotype keys.
    """
    if setting.get("aux", "none") == "none":
        return ["betaxanthin"]
    return list(ACTIVE_HEADS)


#: The metric each arm is RANKED on. Beta-carotene is SPEARMAN and that is not a preference:
#: Ozaydin's readout is a subjective ordinal on -5..+5, so a Pearson asserts an interval
#: scale the measurement does not have. Both `bx_*` arms rank on BETAXANTHIN -- the metabolome
#: is an auxiliary task there, not a second objective.
OBJECTIVE_METRIC = {
    "betaxanthin": "val/betaxanthin/pearson_per_feature_max",
    "beta_carotene": "val/beta_carotene/spearman_per_feature_max",
    "mulleder19": "val/mulleder19/pearson_per_feature_max",
    "bx_ctrl": "val/betaxanthin/pearson_per_feature_max",
    "bx_m19": "val/betaxanthin/pearson_per_feature_max",
    "bx_aa": "val/betaxanthin/pearson_per_feature_max",
}[ARM]
RANK_PHENO = OBJECTIVE_METRIC.split("/")[1]
RANKED_SMOOTH3 = OBJECTIVE_METRIC.replace("_max", "_smooth3_max")

#: WHAT EVERY SETTING ON AN ARM HOLDS FIXED. Each entry is a measurement, not a default.
#:
#: `node_embeddings` -- prot_T5_all on all four arms. It led the kNN embedding probe with zero
#:   learned parameters and won the `_002` studies on betaxanthin and mulleder19. The `_003`/
#:   `_004` screens promoted `calm` / `random_1000` / `codon_frequency` on 17-19 trials at
#:   sigma 0.030, i.e. inside noise -- and a CONTROL embedding topping a 17-trial screen is
#:   evidence about the screen, not about the embedding.
#: `learnable_embedding` -- false, and structurally forced: every strain is a single deletion,
#:   so a strain-level split IS a gene-level split and a val gene's free row never receives a
#:   gradient (observed: train r 0.86 / val r 0.02).
#: `perturbation_head.num_heads` -- 6, the value every run up to and including `_003` used.
#:   `_004` rebound it to 9 IN THE SAME ROUND as it introduced the pinned split, and that
#:   round's best fell to 0.2469 from `_003`'s 0.4050. Two changes, one drop, no attribution.
#:   The pinned split is REQUIRED by the Merzbacher question, so the other change is the one
#:   that gets reverted; re-testing 9 heads is a follow-up, not a factor of this grid.
#: `dist` -- FROZEN per arm at its own measured best (mulleder19 sweeps it instead; see
#:   FACTORS). 019's `_007` settled the axis at n ~ 60 per mode: the five distributional modes
#:   are within noise of each other on accuracy while calibration succeeded. `point` (MSE) was
#:   never a candidate -- its optimum IS the conditional mean, i.e. the mean-collapse the
#:   metric punishes.
#: `hadamard` -- FROZEN at `off`. 019 wave 6 ran the whole pair-term ladder at 1000+ epochs
#:   (ranks 0/9/16/32/64/90 across six mechanisms) and the rank-0 BASELINE placed 2nd of 12;
#:   `replace` scored 0.1965 against the baseline's 0.2107. The mechanism argument still holds
#:   (at |S_b| = 1 the cross-attention softmax is over a one-element key set, so alpha == 1 and
#:   16,200 of 32,760 attention parameters are dead) -- it just does not buy accuracy, on the
#:   same single-deletion regime metabolism lives in.
#: `dropout` -- FROZEN at 0.1. Wave 6: 0.2 -> 0.1881 (below baseline), 0.3 -> COLLAPSED to
#:   exactly 0 (peaked 0.1262 at epoch 225, then nmse 1.002 / pred_sd_ratio 0, i.e. predicting
#:   each gene's mean). Higher dropout is not a lever here, it is a hazard.
#: `weight_decay` -- FROZEN at 1e-8, the expression line's round value, replacing three per-arm
#:   QMC samples (2.72e-6 / 2.11e-7 / 2.11e-5) that were spurious precision from a 19-trial
#:   screen. Wave 6 measured wd as null anyway (wd1e2 0.2041, wd1e4 0.2024 vs baseline 0.2107).
#: `node_embeddings` -- prot_T5_all. Pooled over every completed metabolism trial, betaxanthin
#:   reads 0.2407 for prot_T5_all against 0.0983 for the WIDTH-MATCHED random control, and 8 of
#:   the 10 best betaxanthin runs ever recorded use it. NOTE this deliberately differs from the
#:   expression line, which pinned `calm`: calm ranks 4th on betaxanthin (0.1816), so importing
#:   expression's choice would be a downgrade on the flagship arm. Do not "fix" this.
#: `attention_mask.layers` -- [1]. Wave 5 measured depth at 4 seeds each: [3] 0.1609 vs [1,3]
#:   0.1597. Null. Freezing it is also what frees `num_transformer_layers` to be swept, since
#:   layer 3 does not exist at L = 2.
FIXED: dict[str, dict[str, Any]] = {
    "betaxanthin": {
        "node_embeddings": "prot_T5_all",
        "weight_decay": 1e-8,
        "dist": "crps",  # best betaxanthin run ever recorded: 0.4301
        "target_norm": "zscore",
        "dropout": 0.1,
        "hadamard": "off",
        "mask_layers": [1],
        "perturbation_num_heads": 6,
    },
    "beta_carotene": {
        "node_embeddings": "prot_T5_all",
        "weight_decay": 1e-8,
        "dist": "quantile",
        "target_norm": "zscore",
        "dropout": 0.1,
        "hadamard": "off",
        "mask_layers": [1],
        "perturbation_num_heads": 6,
    },
    "mulleder19": {
        "node_embeddings": "prot_T5_all",
        "weight_decay": 1e-8,
        "dist": "quantile",  # swept on this arm -- see FACTORS
        "target_norm": "zscore",
        "dropout": 0.1,
        "hadamard": "off",
        "mask_layers": [1],
        "perturbation_num_heads": 6,
        # Rank of V in Sigma = diag(sigma^2) + V V^T. 8 fits inside F = 19; 32 cannot be
        # realized by a 19x19 covariance and `DistHead` has no guard. Inert unless dist=energy.
        "energy_rank": 8,
    },
}
FIXED["bx_ctrl"] = dict(FIXED["betaxanthin"])
FIXED["bx_m19"] = dict(FIXED["betaxanthin"])
#: `bx_aa` freezes the ENTIRE grid at the one cell where the auxiliary head was measured to
#: help, and varies nothing but the auxiliary panel and the seed. The three frozen values are
#: not defaults -- they are the coordinates of that cell (`s04_L6_maskon_lr0.0001` in the
#: `bx_ctrl`/`bx_m19` grid), and changing any one of them would make this round's paired
#: differences incomparable to the +0.0203 it exists to replicate.
FIXED["bx_aa"] = dict(FIXED["betaxanthin"])
FIXED["bx_aa"].update(
    {
        # L=6. At L=2 the auxiliary head COST betaxanthin in both fully-run cells (-0.049 at
        # s00, -0.060 at s02, both 1000 epochs, both arms finished); at L=6/mask-on it gained.
        # Depth is not a nuisance parameter here, it is the condition under which the effect
        # was seen at all, so it is pinned to where it was seen.
        "num_transformer_layers": 6,
        # mask ON. The other L=6 cell (s06, mask off) read -0.0153, so the gain does not
        # survive dropping the graph prior on the evidence we have.
        "attention_mask": "on",
        # 1e-4. The decade above it collapses the model to a constant predictor: EVERY
        # lr=1e-3 cell across projects 020/022/023 ended at pred_sd_ratio 0 and nmse 1.000,
        # 21 of 22 of them, and none recovered by epoch 999.
        "lr": 1e-4,
    }
)

#: THE 24-RUN GRID: a full factorial, one seed, every cell the same cost.
#:
#: DEPTH ORDER IS [2, 6, 4] AND THAT IS DELIBERATE. `itertools.product` varies the FIRST key
#: slowest, so the enqueue order is depth-major and the grid completes in three waves of eight,
#: each wave a COMPLETE 2x2x2 at one depth. All 24 cells cost the same (the third axis was
#: chosen to be compute-neutral -- `hidden {90,180}` was rejected for exactly this reason), so
#: if Delta runs slower than planned it is the LAST wave that gets squeezed. Ordering 2 before
#: 6 before 4 means a squeeze costs the INTERPOLATION point and keeps both complete blocks of
#: the contrast that matters: metabolism measured L=2 best (0.4301 vs 0.2764 at 4, 0.2396 at 6)
#: while the expression line fixed L=6, and those were the OLD architecture at ~100 epochs.
#:
#: `attention_mask` {on, off} -- the graph ablation, and it is genuinely UNMEASURED. `_010`
#:   froze masking on speed/memory/transfer and says so: its accuracy comparison was void
#:   because the scorer resolved `D2_mask` to the literal string "A0_baseline" and averaged it
#:   into the baseline. We just committed the architecture; this is the run that validates it.
#: `lr` {1e-4, 1e-3} -- a decade apart, bracketing metabolism's own config default (1e-3) and
#:   the direction longer training usually wants. Round numbers on purpose: every previous
#:   metabolism lr was a single QMC sample, and all of them sat inside one half-decade.
#: `target_norm` {zscore, yeo_johnson} -- the one axis our own rounds contradict (`_002`'s
#:   betaxanthin winner was yeo-johnson, `_003`'s was zscore), never replicated, never run past
#:   ~120 epochs.
FACTORS: dict[str, dict[str, list[Any]]] = {
    "betaxanthin": {
        "num_transformer_layers": [2, 6, 4],
        "attention_mask": ["on", "off"],
        "lr": [1e-4, 1e-3],
        "target_norm": ["zscore", "yeo_johnson"],
    }
}
FACTORS["beta_carotene"] = dict(FACTORS["betaxanthin"])
#: mulleder19 swaps `target_norm` for `dist`. It is the ONLY arm with a joint distribution to
#: model: at F = 1 the predictive covariance Sigma = diag(sigma^2) + V V^T has V of shape
#: [1, k], so V V^T is a scalar and `energy` degenerates into a noisier Monte-Carlo CRPS. Here
#: F = 19. 019's "the distributional axis is not the lever" was measured on expression, read by
#: a per-gene INDEPENDENT head -- a statement about MARGINALS, silent on modelling the joint.
FACTORS["mulleder19"] = {
    "num_transformer_layers": [2, 6, 4],
    "attention_mask": ["on", "off"],
    "lr": [1e-4, 1e-3],
    "dist": ["quantile", "energy"],
}
#: The controlled pair swaps `target_norm` for the ARM itself, so the 24 runs per side become
#: 12, and each (depth, mask, lr) cell is a PAIRED control-vs-joint comparison sharing the same
#: pinned split and the same initialization.
FACTORS["bx_ctrl"] = {
    "num_transformer_layers": [2, 6, 4],
    "attention_mask": ["on", "off"],
    "lr": [1e-4, 1e-3],
}
FACTORS["bx_m19"] = dict(FACTORS["bx_ctrl"])
#: ONE FACTOR, THREE LEVELS, and the replicate budget goes into `SEEDS` instead.
#:
#:   none  betaxanthin alone                          -- the control, shared by both contrasts
#:   m19   + all 19 amino acids                       -- reproduces the grid's `bx_m19` arm
#:   aro   + the 3 aromatic amino acids               -- the PRECURSOR BRANCH
#:
#: Why tyrosine and not another amino acid: betalains derive from L-tyrosine (tyrosine ->
#: L-DOPA -> betalamic acid), and a betaxanthin IS betalamic acid condensed with an amino
#: acid. So tyrosine is the chromophore precursor and the pool is the co-substrate -- the
#: mechanism predicts a coupling, which is what makes a null result here informative rather
#: than merely negative.
#:
#: The control is SHARED across both contrasts, so three cells per seed buy TWO paired
#: differences (m19-none and tyr-none) rather than one. That is the whole reason `tyr` fits in
#: a budget sized for a replication study.
FACTORS["bx_aa"] = {"aux": ["none", "m19", "aro"]}

#: Short factor names for the setting label (a W&B tag and a run-list column).
ABBREV = {"num_transformer_layers": "L", "lr": "lr", "dropout": "d"}

#: The 19 amino acids Mulleder 2016 measured, in the loader's order
#: (`torchcell/datasets/scerevisiae/mulleder2016.py::AMINO_ACIDS`). Duplicated here rather
#: than imported so the grid definition stays readable as a standalone artifact; the
#: import-time check below fails loudly if the two ever drift.
AMINO_ACIDS_19 = [
    "alanine", "aspartate", "glutamate", "phenylalanine", "glycine",
    "histidine", "isoleucine", "lysine", "leucine", "methionine",
    "asparagine", "proline", "glutamine", "arginine", "serine",
    "threonine", "valine", "tryptophan", "tyrosine",
]  # fmt: skip

#: Which of the 19 each `aux` panel SCORES. `none` never reaches this map.
#: `aro` IS THE AROMATIC BRANCH, NOT TYROSINE ALONE, and that is a MODEL CONSTRAINT rather
#: than a design preference. A tyrosine-only panel was tried first and rejected at runtime by
#: `MetabolomeVectorHead`: "output_dim must be >= 2, got 1; use ProductScalarHead for a
#: single-value phenotype" (IGB job 2332401). A one-feature VECTOR head is not something this
#: model builds; a single-value phenotype has to go through the scalar head, which would mean
#: declaring a NEW head -- and a new head is gated by `require_head_targets` through its own
#: `head_phenotype_keys`, restricting the arm to tyrosine-carrying strains and breaking the
#: shared denominator with `m19`.
#:
#: So the precursor contrast becomes the smallest PRINCIPLED panel of at least two: the three
#: aromatic amino acids, which share the shikimate/chorismate route. Tyrosine is the actual
#: betalain precursor (tyrosine -> L-DOPA -> betalamic acid); phenylalanine and tryptophan are
#: pathway siblings and are NOT precursors. The contrast is therefore WEAKER than "tyrosine
#: alone" would have been -- it asks whether the aromatic branch carries the signal, not
#: whether tyrosine specifically does, and a positive result would not on its own pin the
#: effect to the precursor. Recorded here so the readout is not over-read later.
AUX_PANELS: dict[str, list[str]] = {
    "m19": list(AMINO_ACIDS_19),
    "aro": ["tyrosine", "phenylalanine", "tryptophan"],
}

#: `MetabolomeVectorHead` refuses output_dim < 2 (see AUX_PANELS above).
MIN_VECTOR_HEAD_FEATURES = 2


def settings() -> list[dict[str, Any]]:
    """Enumerate the crossed factor levels, in a fixed order, as named settings.

    The name is self-describing but carries NO `=`: it travels into `wandb.tags` through a
    Hydra override, and `=` inside a list element is a parse error in Hydra's override
    grammar ("no viable alternative at input"), which fails at compose time rather than at
    run time -- i.e. it would kill every trial of the job.
    """
    factors = FACTORS[ARM]
    keys = list(factors)
    out = []
    for i, levels in enumerate(itertools.product(*(factors[k] for k in keys))):
        params = dict(zip(keys, levels, strict=True))
        # Built from whatever factors THIS arm has, rather than a fixed template: mulleder19
        # swaps target_norm for dist and the bx_* pair drops the axis entirely, so a hardcoded
        # name would either KeyError or label two different settings identically. Zero-padded
        # so the wave structure reads in sort order.
        bits = []
        for k, v in params.items():
            if k == "num_transformer_layers":
                bits.append(f"L{v}")
            elif k == "attention_mask":
                bits.append(f"mask{v}")
            elif k == "lr":
                bits.append(f"lr{v:g}")
            elif k == "target_norm":
                bits.append("zs" if v == "zscore" else "yj")
            elif k == "aux":
                bits.append("ctrl" if v == "none" else f"aux{v}")
            else:
                bits.append(str(v))
        params["name"] = f"s{i:02d}_" + "_".join(bits)
        out.append(params)
    return out


SETTINGS = settings()
if ONLY_SETTINGS:
    _known = {s["name"] for s in SETTINGS}
    _bad = [s for s in ONLY_SETTINGS if s not in _known]
    if _bad:
        raise SystemExit(
            f"GRID_ONLY_SETTINGS names {_bad} which do not exist on arm {ARM!r}.\n"
            f"available: {sorted(_known)}"
        )
    SETTINGS = [s for s in SETTINGS if s["name"] in ONLY_SETTINGS]
SETTING_NAMES = [s["name"] for s in SETTINGS]
SETTING_BY_NAME = {s["name"]: s for s in SETTINGS}


#: Keys that BOTH `_overrides` and the per-trial log line need. Validated at IMPORT (below)
#: rather than at trial time, so `--create-only` -- which every launcher runs first, before any
#: GPU work -- fails loudly on a mis-split axis.
REQUIRED_KEYS = (
    "node_embeddings",
    "lr",
    "weight_decay",
    "dist",
    "target_norm",
    "dropout",
    "hadamard",
    "mask_layers",
    "attention_mask",
    "num_transformer_layers",
    "perturbation_num_heads",
)


def _resolve(setting: dict[str, Any]) -> dict[str, Any]:
    """The EFFECTIVE config for one setting: the arm's frozen values overlaid by its factors.

    THE SINGLE SOURCE OF TRUTH, and it exists because having two of them was a live bug. When
    `lr` moved from FIXED into FACTORS, `_overrides` merged the two dicts and kept building the
    right config -- but the trial's log line still read `FIXED[ARM]["lr"]`, so EVERY trial died
    with `KeyError('lr')`. optuna's `catch=(Exception,)` swallowed it, so the job exited
    `COMPLETED 0:0`: green exit code, full walltime held, zero runs. It reached two Delta jobs
    and an IGB job before anything executed far enough to reveal it.

    An axis may live in either dict. Nothing may read one of them directly.
    """
    fixed = dict(FIXED[ARM])
    fixed.update({k: v for k, v in setting.items() if k != "name"})
    return fixed


# Fail at import if any setting cannot be fully resolved. This is the check that would have
# caught the bug above during `--create-only`, i.e. seconds after submit rather than inside the
# first trial of a two-day allocation.
for _s in SETTINGS:
    _missing = [k for k in REQUIRED_KEYS if k not in _resolve(_s)]
    if _missing:
        raise SystemExit(
            f"arm {ARM!r} setting {_s['name']!r} is missing {_missing}. "
            f"Every key in REQUIRED_KEYS must come from FIXED[{ARM!r}] or that arm's FACTORS."
        )

# Sharding must divide evenly into the setting count, or `_owns_cell` would leave cells owned
# by nobody -- which looks exactly like a healthy short run in a slurm log.
if SHARD_COUNT:
    if SHARD_COUNT % len(SETTING_NAMES) != 0:
        raise SystemExit(
            f"GRID_SHARD_COUNT={SHARD_COUNT} must be a multiple of the setting count "
            f"({len(SETTING_NAMES)} on arm {ARM!r}), so each worker owns exactly one setting."
        )
    if WORKER_ID >= SHARD_COUNT:
        raise SystemExit(
            f"OPTUNA_WORKER_ID={WORKER_ID} is outside GRID_SHARD_COUNT={SHARD_COUNT}."
        )

# Fail at import if the local copy of the amino-acid list has drifted from the loader's. The
# copy exists for readability, but a stale one would silently mis-name a dropped feature --
# and `drop_features` matches by NAME, so a typo drops nothing and the `tyr` cell would
# quietly score all 19, i.e. become a duplicate of `m19` while still being labelled `tyr`.
if ARM == "bx_aa":
    from torchcell.datasets.scerevisiae.mulleder2016 import (  # noqa: E402
        AMINO_ACIDS as _LOADER_AAS,
    )

    if list(_LOADER_AAS) != AMINO_ACIDS_19:
        raise SystemExit(
            "AMINO_ACIDS_19 has drifted from mulleder2016.AMINO_ACIDS.\n"
            f"  here:   {AMINO_ACIDS_19}\n  loader: {list(_LOADER_AAS)}"
        )
    for _name, _panel in AUX_PANELS.items():
        _bad = [a for a in _panel if a not in AMINO_ACIDS_19]
        if _bad:
            raise SystemExit(f"AUX_PANELS[{_name!r}] names non-amino-acids {_bad}.")
        # Enforce the MODEL's own constraint here, at import, rather than discovering it in
        # the first trial. `MetabolomeVectorHead.__init__` raises on output_dim < 2, and
        # optuna's `catch=(Exception,)` turns that into a FAILED cell instead of a crash --
        # so the worker races through its entire queue failing every cell, at ~11 s each,
        # while the slurm log shows nothing but healthy trial claims. Workers 2 and 5 of job
        # 2332401 burned all four of their cells that way before anything looked wrong.
        if len(_panel) < MIN_VECTOR_HEAD_FEATURES:
            raise SystemExit(
                f"AUX_PANELS[{_name!r}] has {len(_panel)} feature(s); MetabolomeVectorHead "
                f"requires >= {MIN_VECTOR_HEAD_FEATURES}. A single-value phenotype needs a "
                f"scalar head (ProductScalarHead), which is a different design -- see the "
                f"AUX_PANELS comment."
            )


def _overrides(
    setting: dict[str, Any], seed: int, max_time_s: float | None
) -> list[str]:
    """Translate one (setting, seed) cell into Hydra overrides.

    A factor's level takes precedence over the arm's frozen value for the same key, so an axis
    can move between FACTORS and FIXED (as `dist` and depth do across arms) without this
    function needing to know which arm it is serving.
    """
    fixed = _resolve(setting)
    active = _active_heads(setting)
    heads = ",".join(active)
    # Normalization is applied to EVERY active head, so the joint arm standardizes the
    # metabolome too -- an un-standardized 19-AA target would let a few large-magnitude amino
    # acids dominate the auxiliary loss and change what the shared encoder is pulled toward.
    zscore = fixed["target_norm"] == "zscore"
    normalize_list = [] if zscore else list(active)
    standardize_list = list(active) if zscore else []
    ov = [
        f"seed={seed}",
        f"multitask.active_heads=[{heads}]",
        f"cell_dataset.node_embeddings=[{fixed['node_embeddings']}]",
        "model.learnable_embedding.enabled=false",
        "multitask.decoder=s1_pool",
        f"multitask.dist={fixed['dist']}",
        f"multitask.normalize_vector_targets=[{','.join(normalize_list)}]",
        f"multitask.standardize_per_feature_target=[{','.join(standardize_list)}]",
        f"model.num_transformer_layers={fixed['num_transformer_layers']}",
        # THE GRAPH ABLATION. `off` leaves NO graph prior at all -- the mask is gone and the
        # KL is already 0 -- which is the honest control for "do the nine graphs do anything
        # under this architecture", a question `_010` never actually answered.
        f"model.attention_mask.enabled={'true' if fixed['attention_mask'] == 'on' else 'false'}",
        f"model.dropout={fixed['dropout']}",
        f"model.perturbation_head.num_heads={fixed['perturbation_num_heads']}",
        # THE PAIR TERM. `off` is the additive reference operator, which at |S_b| = 1 has no
        # pair-(p,i) term at all; `replace` swaps in h_i * (1 + gamma(c_b)), a rank-d = 90
        # bilinear interaction that is the exact identity at init.
        f"model.perturbation_head.hadamard={fixed['hadamard']}",
        # Encoder layers where the nine-relation hard mask is applied. NOT a lambda: the mask
        # is a structural constraint, so there is nothing to calibrate -- only where to apply
        # it. Layer indices must exist at `num_transformer_layers`, which is why L is frozen.
        f"model.attention_mask.layers=[{','.join(str(x) for x in fixed['mask_layers'])}]",
        f"regression_task.optimizer.lr={fixed['lr']:g}",
        f"regression_task.optimizer.weight_decay={fixed['weight_decay']}",
        f"data_module.num_workers={os.getenv('NUM_WORKERS', '4')}",
        # QUOTED elements. A tag like `s0_0.1_lam1.6e-05_2` contains `.` and `-`, which
        # Hydra's override grammar tries to lex as a number inside a list; quoting makes each
        # element an unambiguous string.
        f"wandb.tags=['delta','grid','{ARM}','{setting['name']}','seed-{seed}']",
    ]
    if "energy_rank" in fixed:
        ov.append(f"multitask.energy_rank={fixed['energy_rank']}")
    # THE AUXILIARY PANEL, and the mechanism is `drop_features` -- NOT `head_phenotype_keys`.
    # Two independent reasons, both of which cost a silently-wrong comparison if ignored:
    #
    # 1. INSTANCE-SET PARITY. `require_head_targets` resolves each head through
    #    `head_phenotype_keys` and takes the UNION over its keys. Re-pinning mulleder19 to
    #    [tyrosine] would therefore restrict the run to genotypes carrying TYROSINE, while
    #    the `m19` cell keeps genotypes carrying any of the 19 -- so the two panels would be
    #    scored on different strains and `tyr - none` would not be comparable to `m19 - none`.
    #    `drop_features` leaves the gate on all 19 and only narrows what the head SCORES.
    #
    # 2. LABEL DISAMBIGUATION. betaxanthin and the metabolome are BOTH `metabolite_level`, and
    #    the COO drops the dict keys, so the ONLY thing separating their value groups within a
    #    genotype is the group SIZE -- which the aligner enforces as distinct. Re-pinning would
    #    take mulleder19's raw_dim from 19 to 1, colliding with betaxanthin's 1 and making the
    #    assignment arbitrary. `drop_features` keeps raw_dim at 19 and moves only feat_dim.
    panel = AUX_PANELS.get(str(setting.get("aux", "none")))
    if panel is not None:
        dropped = [a for a in AMINO_ACIDS_19 if a not in panel]
        if dropped:
            ov.append(f"multitask.drop_features.mulleder19=[{','.join(dropped)}]")
        # The head's output_dim MUST equal the post-drop feature count; the trainer asserts it.
        ov.append(f"multitask.heads.mulleder19.output_dim={len(panel)}")
    if max_time_s is not None:
        ov.append(f"trainer.max_time_s={max_time_s:.0f}")
    return ov


def _trial_time_budget() -> float | None:
    """Seconds of TRAINING this trial gets, from the job's remaining wall clock.

    Computed at CLAIM time rather than once per job: the eight round-0 runs all start
    together and so all get (almost) the full window, while a replicate that starts later
    correctly gets only what is left. `TEARDOWN_S` is subtracted so the test pass, the
    prediction dump and the wandb teardown happen INSIDE the allocation -- the graceful stop
    is worth nothing if slurm kills the process during the work the stop exists to protect.
    """
    # `GRID_NO_TIME_CAP` keeps the DEADLINE gating which cells get CLAIMED (see
    # `deadline_callback`) while removing the per-trial time cap, so every cell that starts
    # runs the SAME number of epochs. That distinction is a correctness requirement for a
    # paired design, not a preference: the objective is a max over epochs, an upward-biased
    # order statistic whose bias grows with epochs run. If the clock cut one member of a pair
    # at 400 epochs and its partner reached 600, the difference between them would be part
    # effect and part unequal exposure, with no way to separate the two. Whole cells missing
    # from the tail costs power; unevenly truncated pairs costs validity.
    if DEADLINE is None or os.getenv("GRID_NO_TIME_CAP", "") == "1":
        return None
    return max(0.0, DEADLINE - time.time() - TEARDOWN_S)


def _record(trial: optuna.Trial, metrics: dict[str, float]) -> None:
    """Mirror the diagnostics into the trial so the study reads standalone.

    Recorded, never ranked. `pred_sd_ratio` is the mean-collapse diagnostic for a SCALAR head
    (`pearson_per_instance` is undefined at F = 1); `peak_epoch` / `n_val_epochs` are the
    covariates for the max-over-epochs bias, which on `_002` mulleder19 correlated with the
    objective at r = +0.75; the `test/*` keys exist only on arms running the pinned split.
    """
    for name, key in (
        ("pearson", f"val/{RANK_PHENO}/pearson_per_feature_max"),
        ("spearman", f"val/{RANK_PHENO}/spearman_per_feature_max"),
        ("ranked_smooth3", RANKED_SMOOTH3),
        ("pred_sd_ratio", f"val/{RANK_PHENO}/pred_sd_ratio_at_peak"),
        ("graph_reg_ratio", "val/graph_reg/ratio_to_data_at_peak"),
        ("n_val_epochs", "val/n_val_epochs"),
        ("peak_epoch", "val/peak_epoch"),
        ("test_pearson", f"test/{RANK_PHENO}/pearson_per_feature"),
        ("test_spearman", f"test/{RANK_PHENO}/spearman_per_feature"),
    ):
        trial.set_user_attr(name, metrics.get(key))
    # The auxiliary head's OWN score. "The metabolome helped betaxanthin" and "the metabolome
    # head learned anything" are different claims, and the second explains the first -- an
    # auxiliary head at r ~ 0 that still moves the primary metric would mean the gain came
    # from regularization, not from shared metabolic signal.
    for aux in ACTIVE_HEADS:
        if aux == RANK_PHENO:
            continue
        trial.set_user_attr(
            f"aux_{aux}_pearson", metrics.get(f"val/{aux}/pearson_per_feature_max")
        )
    for key in ("coverage_50", "coverage_80", "pit_ks"):
        trial.set_user_attr(
            f"calib_{key}", metrics.get(f"val/{RANK_PHENO}/calib/{key}_at_peak")
        )


def objective(trial: optuna.Trial) -> float:
    """Run one (setting, seed) cell.

    Both parameters are `suggest_categorical` over the FULL level set even though every trial
    is enqueued: optuna replays an enqueued trial by re-suggesting, and a suggest whose
    choices differ from the enqueued value's domain is a distribution mismatch, not a
    fixed value.
    """
    name = trial.suggest_categorical("setting", SETTING_NAMES)
    seed = trial.suggest_categorical("seed", SEEDS)
    setting = SETTING_BY_NAME[name]
    budget = _trial_time_budget()
    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        cfg = compose(
            config_name=BASE_CONFIG, overrides=_overrides(setting, seed, budget)
        )
    eff = _resolve(setting)
    print(
        f"[{ARM} w{WORKER_ID}] trial {trial.number}: {name} seed={seed} "
        f"emb={eff['node_embeddings']} L={eff['num_transformer_layers']} "
        f"mask={eff['attention_mask']} lr={eff['lr']:.3g} wd={eff['weight_decay']:.3g} "
        f"budget={'unbounded' if budget is None else f'{budget / 3600:.2f} h'}",
        flush=True,
    )
    print(OmegaConf.to_yaml(cfg.multitask), flush=True)
    metrics: dict[str, float] = run_training(cfg)
    torch.cuda.empty_cache()
    _record(trial, metrics)
    trial.set_user_attr("setting_params", {k: v for k, v in setting.items()})
    trial.set_user_attr("train_time_budget_s", budget)
    if OBJECTIVE_METRIC not in metrics:
        raise optuna.TrialPruned(
            f"{OBJECTIVE_METRIC} not logged (keys: {sorted(metrics)[:12]}...)"
        )
    return float(metrics[OBJECTIVE_METRIC])


def get_study() -> optuna.Study:
    """Create-or-load. The study is a WORK LEDGER, not a search.

    Every trial is enqueued, so the sampler is never consulted for anything that matters; it
    exists because `study.optimize` requires one. Optuna is used here for what it is genuinely
    good at in this setting -- concurrency-safe claiming across four processes on one SQLite
    file, durable per-trial attributes, and resume-after-kill -- and for nothing else.
    """
    return optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        sampler=optuna.samplers.RandomSampler(seed=0),
        load_if_exists=True,
        direction="maximize",
    )


def _enqueue_all(study: optuna.Study) -> int:
    """Enqueue every (setting, seed) cell, ROUND-MAJOR, skipping what already exists.

    Round-major is load-bearing: a job killed by the wall clock then leaves every setting with
    the SAME seed count, so the factorial stays balanced and the comparison stays valid.
    Seed-major would leave 12 seeds on two settings and none on six.

    Idempotent -- a requeue after a kill tops the queue back up rather than duplicating it.
    An enqueued (WAITING) trial keeps its values in `system_attrs["fixed_params"]` until it
    runs, so `.params` alone would fail to recognize the existing queue.
    """
    # FAIL and PRUNED are DELIBERATELY ABSENT from `seen`, so a cell whose run died is
    # re-enqueued rather than skipped. This is not hypothetical bookkeeping: the KeyError('lr')
    # bug left FAILED trials in the Delta studies, and counting them as "already present" would
    # have made the resubmit enqueue only the cells that had not yet reached the crash --
    # silently running a PARTIAL factorial and reporting nothing wrong. The predecessor sweep
    # driver had this exact carve-out; the rewrite dropped it.
    #
    # An enqueued (WAITING) trial keeps its values in `system_attrs["fixed_params"]` until it
    # runs, so `.params` alone would fail to recognize a queue that is merely waiting.
    seen = set()
    for t in study.trials:
        if t.state.name in ("FAIL", "PRUNED"):
            continue
        params = t.params or t.system_attrs.get("fixed_params", {})
        if "setting" in params and "seed" in params:
            seen.add((params["setting"], params["seed"]))
    added = 0
    for r in range(ROUNDS):
        seed = SEEDS[r % len(SEEDS)]
        for i, name in enumerate(SETTING_NAMES):
            if not _owns_cell(r, i):
                continue
            if (name, seed) in seen:
                continue
            study.enqueue_trial({"setting": name, "seed": seed})
            seen.add((name, seed))
            added += 1
    return added


def _owns_cell(round_idx: int, setting_idx: int) -> bool:
    """Does THIS worker own cell (round, setting)? Always true in shared-queue mode.

    THE SHARDING RULE IS BY WHOLE SEED, not by a flat stride over the cell list, and that
    choice is the whole point. With S settings and W workers there are G = W // S groups; group
    g = WORKER_ID // S takes every G-th ROUND, and within a group worker w runs setting
    `w % S`. So at any instant G COMPLETE seeds are in flight and they finish together.

    A flat `cell_index % W` stride would also cover every cell exactly once, but with W = 6 and
    S = 3 the stride is 6 -- two whole seeds -- so worker 0 would draw the CONTROL at seeds
    0, 2, 4, 6 and nothing else. Every worker would own one setting across scattered seeds, and
    a wall-clock stop would leave controls at seeds the treatments never reached: paired
    differences with no partner, which is not a smaller result but an unreadable one.

    Requires W % S == 0, checked at import.
    """
    if not SHARD_COUNT:
        return True
    n_settings = len(SETTING_NAMES)
    n_groups = SHARD_COUNT // n_settings
    return (
        setting_idx == WORKER_ID % n_settings
        and round_idx % n_groups == WORKER_ID // n_settings
    )


def _write_manifest() -> None:
    """Record the grid as an artifact, so a result can be read without re-reading this file."""
    out_dir = os.getenv("GRID_MANIFEST_DIR")
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    path = osp.join(out_dir, f"grid_manifest_{ARM}.json")
    with open(path, "w") as fh:
        json.dump(
            {
                "arm": ARM,
                "base_config": BASE_CONFIG,
                "study_name": STUDY_NAME,
                "objective_metric": OBJECTIVE_METRIC,
                "active_heads": ACTIVE_HEADS,
                "fixed": FIXED[ARM],
                "factors": FACTORS[ARM],
                "settings": SETTINGS,
                "rounds": ROUNDS,
                "seeds": SEEDS[:ROUNDS],
            },
            fh,
            indent=2,
        )
    print(f"[grid] wrote {path}", flush=True)


def main() -> None:
    study = get_study()
    if "--create-only" in sys.argv:
        added = _enqueue_all(study)
        print(
            f"[create-only] study={STUDY_NAME} arm={ARM} "
            f"{len(SETTINGS)} settings x {ROUNDS} rounds -> {added} newly enqueued "
            f"({len(study.trials)} total in study)",
            flush=True,
        )
        for s in SETTINGS:
            print(f"  {s['name']}", flush=True)
        _write_manifest()
        return

    # Stop CLAIMING new work once too little wall clock remains to say anything. A trial that
    # DOES start is bounded by `trainer.max_time_s` rather than by slurm, so it always ends
    # gracefully -- this guard is about not starting a run too short to read, not about
    # avoiding a kill.
    def deadline_callback(st: optuna.Study, _: optuna.trial.FrozenTrial) -> None:
        if DEADLINE is not None and time.time() > DEADLINE - MIN_TRIAL_S - TEARDOWN_S:
            print(
                f"[{ARM} w{WORKER_ID}] less than "
                f"{(MIN_TRIAL_S + TEARDOWN_S) / 60:.0f} min of usable wall clock left "
                f"-- not claiming further work.",
                flush=True,
            )
            st.stop()

    if DEADLINE is not None and time.time() > DEADLINE - MIN_TRIAL_S - TEARDOWN_S:
        print(f"[{ARM} w{WORKER_ID}] past the deadline before starting -- exiting.")
        return

    waiting = sum(1 for t in study.trials if t.state.name == "WAITING")
    print(
        f"[{ARM} w{WORKER_ID}] study={STUDY_NAME} objective={OBJECTIVE_METRIC} "
        f"{waiting} trials waiting, base={BASE_CONFIG}",
        flush=True,
    )
    # `n_trials` bounds this worker; the real stop is the empty queue or the deadline. optuna
    # runs an enqueued trial ahead of a sampled one, so the queue drains before any random
    # config could be invented -- and `_enqueue_all` sizes the queue to the grid exactly.
    study.optimize(
        objective,
        n_trials=len(SETTINGS) * ROUNDS,
        catch=(Exception,),
        callbacks=[deadline_callback],
    )

    complete = [t for t in study.trials if t.state.name == "COMPLETE" and t.values]
    if WORKER_ID == 0 and complete:
        by_setting: dict[str, list[float]] = {}
        for t in complete:
            by_setting.setdefault(t.params["setting"], []).append(t.values[0])
        print(f"\n[{ARM}] {len(complete)} completed runs")
        for name in SETTING_NAMES:
            vals = by_setting.get(name, [])
            if not vals:
                print(f"  {name:56s} n=0")
                continue
            mean = sum(vals) / len(vals)
            sd = (
                (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
                if len(vals) > 1
                else float("nan")
            )
            sem = sd / len(vals) ** 0.5 if len(vals) > 1 else float("nan")
            print(
                f"  {name:56s} n={len(vals):2d} mean={mean:.4f} sd={sd:.4f} sem={sem:.4f}"
            )


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
