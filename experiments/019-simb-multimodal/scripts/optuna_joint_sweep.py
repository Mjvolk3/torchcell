# experiments/019-simb-multimodal/scripts/optuna_joint_sweep.py
# [[experiments.019-simb-multimodal.scripts.optuna_joint_sweep]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/optuna_joint_sweep
"""DECODER x DISTRIBUTIONAL Optuna sweep driver (_003) for the Fig-3 multitask CGT.

Morphology sits at per-feature Pearson ~0.04 against a measured noise ceiling of 0.61
(6.5% of achievable), while expression is at its ~0.09-0.11 ceiling. Expression reads its
OWN token (S0 per-gene); morphology reads a SHARED pooled vector (S1), where a mean over
~6000 gene tokens dilutes a 1-2 gene perturbation before readout. PCA of the 278 CalMorph
targets gives an effective rank of 11.7, so capacity/embedding size is NOT the constraint
(confirmed by the _002 round: scaling d did not help). This sweep tests the DECODER instead,
over two orthogonal axes:

* STRUCTURAL (`multitask.decoder`, morphology head only)
  ``s1_pool`` -- GlobalHead: pool ``[h_CLS || mean_genes]``, fan out to all F features.
  ``s3_xattn`` -- CrossAttnHead: one LEARNED QUERY per feature cross-attending the full
  token set, so each output reads its own mixture (per-output support without a 1-1 map).
  Expression is per-token by construction (S0) and has no decoder lever.

* DISTRIBUTIONAL (`multitask.dist`, all vector heads)
  ``point`` -- MSE. Baseline; the MSE optimum IS the conditional mean, so it rewards
  mean-collapse (which drives per-feature Pearson to ~0).
  ``crps`` -- closed-form Gaussian CRPS, a proper scoring rule in y-units (no 1/sigma^2
  blow-up, so no sigma collapse; supersedes beta-NLL).
  ``quantile`` -- pinball over K=19 evenly spaced tau (distribution-free; no kernel or
  bandwidth to tune, unlike the earlier KDE attempts).

RANKING is on the PEAK of `val/<phenotype>/pearson_per_feature` -- the honest metric
(per-feature across strains, destroyed by mean-collapse), computed from `DistHead.point()`
so a CRPS run and an MSE run are compared on the same point-estimate correlation. Peak, not
last epoch, because these runs peak early then MSE-collapse toward the per-feature mean.

THE CONTROL: all three conditions run on the SAME instance set (`require_modalities` in the
base config), so expr_morph - morph isolates the auxiliary-task effect from a data-quantity
confound. Same split as _002, so _003 is directly comparable.

    CONDITION=morph      active_heads=[global]           single-obj: morphology
    CONDITION=expr       active_heads=[per_gene]         single-obj: expression (S0 control)
    CONDITION=expr_morph active_heads=[per_gene, global] MULTI-obj: (expr, morph) Pareto

Environment (set by the slurm launchers):
    CONDITION           expr | morph | expr_morph   (default: expr_morph)
    OPTUNA_STORAGE      sqlite:////<scratch>/.../optuna_019_<cond>_003.db   (required)
    OPTUNA_STUDY_NAME   default: <condition>_003
    OPTUNA_N_TRIALS     trials THIS worker runs (default: 20)
    OPTUNA_WORKER_ID    seeds the sampler (default: 0)
    JOINT_BASE_CONFIG   base Hydra config (default: cgt_decoder_003)
    WANDB_PROJECT_SUFFIX  W&B project suffix (default: v3)
    NUM_WORKERS         dataloader workers per trial (default: 4)
"""

import os
import os.path as osp
import sys

import optuna
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from train_cgt_multitask import run_training

CONF_DIR = osp.abspath(osp.join(osp.dirname(__file__), "../conf"))
BASE_CONFIG = os.getenv("JOINT_BASE_CONFIG", "cgt_embed_005")
STORAGE = os.environ["OPTUNA_STORAGE"]
CONDITION = os.getenv("CONDITION", "expr_morph")
STUDY_NAME = os.getenv("OPTUNA_STUDY_NAME", f"{CONDITION}_005")
N_TRIALS = int(os.getenv("OPTUNA_N_TRIALS", "20"))
WORKER_ID = int(os.getenv("OPTUNA_WORKER_ID", "0"))
PROJECT_SUFFIX = os.getenv("WANDB_PROJECT_SUFFIX", "v5")

ACTIVE_HEADS = {
    "expr": ["per_gene"],
    "morph": ["global"],
    "expr_morph": ["per_gene", "global"],
}[CONDITION]

# PEAK (max over training), not the last epoch — runs peak then MSE-collapse to the mean.
# Namespaced by PHENOTYPE (stable across decoder variants), computed from DistHead.point()
# so the ranking is loss-agnostic.
EXPR_METRIC = "val/expression/pearson_per_feature_max"
MORPH_METRIC = "val/morphology/pearson_per_feature_max"

PROFILES = {
    "baseline": {"lr": 3.0e-4, "dropout": 0.1, "weight_decay": 1.0e-8},
    "aggressive": {"lr": 1.0e-3, "dropout": 0.0, "weight_decay": 1.0e-4},
}

# `raw` is GONE as a lever: every vector head is standardized (see the base config's
# require_standardized_targets). An un-standardized 278-D CalMorph target makes any y-unit
# loss (MSE/CRPS/pinball) chase a few large-magnitude features while the metric weights all
# 278 equally — that would confound the decoder comparison.
TARGET_NORMS = ["zscore", "yeo_johnson"]

# ---- Node-embedding (cold-start) axis, round _005 ----
# Every option is COVERAGE-AUDITED: all 6607 genes, 0 missing, 0 zero-vectors. Variants that
# fail that bar are deliberately absent (one_hot_gene / fudt_downstream miss the 28
# mitochondrial Q0* genes; the esm2/prot_T5 `no_dubious` / `no_uncharacterized` variants emit
# 684/668 ALL-ZERO vectors, i.e. silently information-free for the hardest ORFs).
#   (""            -- DROPPED: learnable-only baseline, superseded by random_100.)
#   codon_frequency (64d)  -- only option smaller than hidden, so no lossy compression.
#   calm (768d), fudt_upstream (768d), prot_T5_all (1024d), esm2_..._all (1280d),
#   nt_window_5979 (2560d) -- real sequence/protein content.
#   random_100 (100d)      -- CONTROL: unique but meaningless. Paired with
#     learnable_embedding=false it isolates CONTENT from IDENTITY; with learnable=true it is
#     redundant with the free table, which is why `learnable` is swept alongside it.
# NOTE: "" (no content features) is DROPPED now that learnable_embedding is pinned False
# -- "" + no free table leaves genes with no node features at all. `random_100` is the
# strictly better version of that baseline: same "unique but meaningless" semantics, but
# explicit, fixed, and usable at cold start.
NODE_EMBEDDINGS = [
    "codon_frequency",
    "calm",
    "fudt_upstream",
    "prot_T5_all",
    "esm2_t33_650M_UR50D_all",
    "nt_window_5979",
    "random_100",
]


def _norm_choice(trial: optuna.Trial, head: str) -> str:
    """Per-modality target-norm lever, always standardized ({zscore, yeo_johnson})."""
    param = "expr_norm" if head == "per_gene" else "morph_norm"
    return trial.suggest_categorical(param, TARGET_NORMS)


def objective(trial: optuna.Trial) -> float | tuple[float, float]:
    # ---- Decoder study axes ----
    # `decoder` is only meaningful when the morphology (global) head is active; expression
    # is S0 per-token by construction. Suggesting it for the expr arm would add a phantom
    # dimension that splits the TPE search space without changing the model.
    if "global" in ACTIVE_HEADS:
        decoder = trial.suggest_categorical("decoder", ["s1_pool", "s3_xattn"])
    else:
        decoder = "s1_pool"
    # The joint arm drops `quantile` (K=19 params x 6127 genes x 278 features is a large
    # readout on a 4-GPU shared node); morph/expr sweep all three.
    dist_choices = (
        ["point", "crps"] if CONDITION == "expr_morph" else ["point", "crps", "quantile"]
    )
    dist = trial.suggest_categorical("dist", dist_choices)

    # ---- Cold-start axes (_005) ----
    node_emb = trial.suggest_categorical("node_embeddings", NODE_EMBEDDINGS)
    # learnable_embedding is NO LONGER SWEPT -- it is pinned False.
    #
    # Measured on the actual split (seed 0, the seed these runs use): only 26 of 539
    # validation genes (4.8%) and 14 of 519 test genes (2.7%) are ever perturbed in
    # training. Because every strain is a SINGLE deletion, splitting by strain IS
    # splitting by gene -- there is no warm-start regime in this dataset. So a free
    # per-gene row is still at initialization for ~95% of evaluation genes and cannot
    # contribute to the ranked (validation) metric; it can only memorize train, which is
    # exactly the train 0.9 / val 0.02 gap observed on morphology.
    #
    # Sweeping it therefore spent trials on a lever that provably cannot move val. The
    # honest version of "unique but meaningless vector" is `random_100`, which stays in
    # NODE_EMBEDDINGS as the explicit control -- and, combined with graph regularization,
    # is also the test of whether GRAPH POSITION alone carries the signal (a fixed random
    # vector makes each gene's neighborhood aggregate a computable fingerprint even for an
    # unseen gene).
    learnable = False

    hidden = trial.suggest_categorical("hidden_channels", [90, 180])
    # Widened from {2,4}: L=2 won _003, but under memorization depth mostly buys memorization.
    layers = trial.suggest_categorical("num_transformer_layers", [2, 4, 6, 8])
    # Re-centered and log-spaced for _006. Two model-side fixes changed what lambda MEANS:
    # it is no longer applied twice (the old effective weight was lambda^2 -- 1e-6 at the
    # swept 1e-3, i.e. the graph term was off), and normalization is now per-graph mean
    # degree instead of the summed degree over all graphs. 0.0 is dropped: the point of
    # this round is a WIDE, ENFORCED graph channel, and `val/graph_reg/ratio_to_data` (logged per
    # step) is what tells us which of these values actually constrain attention rather
    # than guessing the scale in advance.
    # CALIBRATED FROM THE MONITOR, not guessed. The first _006 launch used {1e-3 .. 1.0}
    # and `val/graph_reg/ratio_to_data` immediately read 0.99992 -- graph_reg_loss 6609 against an
    # expression loss of 0.54, i.e. the KL term was the entire objective and the task was
    # not being trained at all. At lambda=1e-2 the unweighted graph term is ~6.6e5, so
    # parity with the task loss sits near lambda ~ 8e-7:
    #
    #     lambda 1e-8 -> frac ~0.01     1e-7 -> ~0.11     1e-6 -> ~0.55     1e-5 -> ~0.92
    #
    # This grid therefore brackets "barely on" to "dominant" with the centre where the
    # graph term and the task term are comparable, which is the intent of the round.
    #
    # Worth recording: the old doubled-lambda code computed lambda^2, so the previously
    # swept 1e-3 landed at an effective 1e-6 -- accidentally inside this workable band.
    # Applying lambda once is still the correct fix; it just means the grid had to move
    # with it rather than being carried over.
    graph_reg = trial.suggest_categorical("graph_reg_lambda", [1e-8, 1e-7, 1e-6, 1e-5])
    profile_name = trial.suggest_categorical("hp_profile", ["baseline", "aggressive"])
    profile = PROFILES[profile_name]

    # Build the two per-head normalization lists from each active head's sampled lever.
    normalize_list: list[str] = []   # -> Yeo-Johnson (vector_norm_method)
    standardize_list: list[str] = []  # -> per-feature z-score
    for head in ACTIVE_HEADS:
        choice = _norm_choice(trial, head)
        if choice == "zscore":
            standardize_list.append(head)
        else:
            normalize_list.append(head)

    overrides = [
        f"multitask.active_heads=[{','.join(ACTIVE_HEADS)}]",
        f"cell_dataset.node_embeddings=[{node_emb}]",
        f"model.learnable_embedding.enabled={learnable}",
        f"multitask.decoder={decoder}",
        f"multitask.dist={dist}",
        f"multitask.normalize_vector_targets=[{','.join(normalize_list)}]",
        f"multitask.standardize_per_feature_target=[{','.join(standardize_list)}]",
        f"model.hidden_channels={hidden}",
        f"model.num_transformer_layers={layers}",
        f"model.dropout={profile['dropout']}",
        f"model.graph_regularization.graph_reg_lambda={graph_reg}",
        f"regression_task.optimizer.lr={profile['lr']}",
        f"regression_task.optimizer.weight_decay={profile['weight_decay']}",
        f"data_module.num_workers={os.getenv('NUM_WORKERS', '4')}",
        # One W&B project per condition: expr | morph | expr_morph. `_v3` marks this
        # decoder x distributional round, distinct from the _002 controlled runs.
        f"wandb.project=torchcell_019_{CONDITION}_{PROJECT_SUFFIX}",
        "wandb.tags=[ws-run,ctrl-split,decoder,optuna,single-gpu,"
        f"{CONDITION},{decoder},{dist},trial-{trial.number},{profile_name}]",
    ]

    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        cfg = compose(config_name=BASE_CONFIG, overrides=overrides)

    print(
        f"[{CONDITION} w{WORKER_ID}] trial {trial.number}: heads={ACTIVE_HEADS} "
        f"emb={node_emb or '(none)'} learnable={learnable} "
        f"decoder={decoder} dist={dist} hidden={hidden} layers={layers} "
        f"graph_reg={graph_reg} profile={profile_name} "
        f"norm+={normalize_list} std+={standardize_list}",
        flush=True,
    )
    print(OmegaConf.to_yaml(cfg.multitask), flush=True)

    metrics = run_training(cfg)
    torch.cuda.empty_cache()

    expr_r = metrics.get(EXPR_METRIC)
    morph_r = metrics.get(MORPH_METRIC)
    trial.set_user_attr("expr_pearson", expr_r)
    trial.set_user_attr("morph_pearson", morph_r)
    trial.set_user_attr("decoder", decoder)
    trial.set_user_attr("node_embeddings", node_emb or "(none)")
    trial.set_user_attr("learnable_embedding", learnable)
    trial.set_user_attr("dist", dist)
    # The mean-collapse diagnostic: per-instance stays high under collapse, so a large gap
    # vs per-feature is the signature. Recorded per trial so the sweep can be read for
    # "did the distributional loss actually prevent collapse?", not just the headline r.
    trial.set_user_attr(
        "morph_pearson_per_instance",
        metrics.get("val/morphology/pearson_per_instance_max"),
    )
    trial.set_user_attr(
        "expr_pearson_per_instance",
        metrics.get("val/expression/pearson_per_instance_max"),
    )

    if CONDITION == "expr_morph":
        # MULTI-OBJECTIVE: maximize BOTH honest metrics -> Optuna returns the Pareto front of
        # (expression, morphology). This is the scientific object: the configs on the frontier
        # of "good at both", NOT a hand-weighted scalar that hides the trade-off.
        if expr_r is None or morph_r is None:
            raise optuna.TrialPruned(
                f"expr_morph needs both metrics (expr={expr_r}, morph={morph_r})"
            )
        return expr_r, morph_r

    objective_metric = EXPR_METRIC if CONDITION == "expr" else MORPH_METRIC
    if objective_metric not in metrics:
        raise optuna.TrialPruned(
            f"{objective_metric} not logged (keys: {sorted(metrics)[:12]}...)"
        )
    return metrics[objective_metric]


def get_study() -> optuna.Study:
    """Create-or-load the study. expr_morph = MULTI-objective (maximize expr, maximize morph);
    expr/morph = single-objective. TPESampler handles both (MOTPE for the multi case).
    """
    sampler = optuna.samplers.TPESampler(seed=WORKER_ID, multivariate=True, group=True)
    common = dict(
        study_name=STUDY_NAME, storage=STORAGE, sampler=sampler, load_if_exists=True
    )
    if CONDITION == "expr_morph":
        return optuna.create_study(directions=["maximize", "maximize"], **common)
    return optuna.create_study(direction="maximize", **common)


def main() -> None:
    study = get_study()
    # --create-only: the slurm runs this ONCE (serialized) before the workers so they only
    # load_if_exists — avoids the fresh-DB DDL race. The directions logic lives HERE (one
    # place) so pre-create and workers always agree on single- vs multi-objective.
    if "--create-only" in sys.argv:
        print(f"[create-only] study={STUDY_NAME} directions={study.directions}", flush=True)
        return

    print(
        f"[{CONDITION} w{WORKER_ID}] study={STUDY_NAME} heads={ACTIVE_HEADS} "
        f"n_trials={N_TRIALS} multi_obj={CONDITION == 'expr_morph'} base={BASE_CONFIG}",
        flush=True,
    )
    study.optimize(objective, n_trials=N_TRIALS, catch=(Exception,))

    if WORKER_ID == 0:
        if CONDITION == "expr_morph":
            print(f"[expr_morph w0] Pareto front ({len(study.best_trials)} trials):")
            for t in study.best_trials[:10]:
                print(f"  t{t.number} (expr,morph)={[round(v, 4) for v in t.values]} {t.params}")
        else:
            print(f"[{CONDITION} w0] best={study.best_value:.4f} params={study.best_params}")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
