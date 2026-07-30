#!/bin/bash
# experiments/019-simb-multimodal/scripts/gh_expr_008_arm.sh
# [[experiments.019-simb-multimodal.scripts.gh_expr_008_arm]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/gh_expr_008_arm
#
# Run ONE arm x ONE seed of the _008 pair-routing round on GilaHyper.
#
# Not a slurm script: _008 runs on the single GPU left free while the 020/021/022
# metabolism jobs (1388-1390) hold GPUs 1-3, and it is driven interactively arm by arm
# rather than submitted as a batch sweep. Pin the GPU with CUDA_VISIBLE_DEVICES.
#
#   bash experiments/019-simb-multimodal/scripts/gh_expr_008_arm.sh A0_baseline 0
#
# ARMS -- the only thing that varies between them is listed here; every other
# hyperparameter is frozen in conf/cgt_expr_008.yaml at _006's measured winner.
#   A0_baseline    (nothing)                   the ~0.172 reference, re-run on GH
#   A1_free16      free_gene_dim=16            CAPACITY only: unties output-gene id from
#                                              h_k. Adds no information, so under the
#                                              structural claim this should move little.
#   A2_self        propagation hops=0          the cheapest PAIR term: a hop-0 indicator
#                                              of "is gene i itself the deleted gene".
#                                              The baseline cannot mark that at all.
#   A3_prop_h1     propagation hops=1          self + 1-hop graph routing
#   A4_prop_h2     propagation hops=2          self + 2-hop graph routing
#   A5_prop_h2_free16                          do capacity and information compose?
#
# A2 vs A3/A4 is the attribution that matters: if A2 captures most of the gain, the win
# is "identify the deleted gene", not "route along the graphs".
set -euo pipefail

ARM="${1:?usage: gh_expr_008_arm.sh <arm> <seed>}"
SEED="${2:?usage: gh_expr_008_arm.sh <arm> <seed>}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROP="model.perturbation_propagation.enabled"

case "$ARM" in
  A0_baseline)       OVERRIDES=() ;;
  A1_free16)         OVERRIDES=(multitask.free_gene_dim=16) ;;
  A2_self)           OVERRIDES=("$PROP=true" model.perturbation_propagation.hops=0) ;;
  A3_prop_h1)        OVERRIDES=("$PROP=true" model.perturbation_propagation.hops=1) ;;
  A4_prop_h2)        OVERRIDES=("$PROP=true" model.perturbation_propagation.hops=2) ;;
  A5_prop_h2_free16) OVERRIDES=("$PROP=true" model.perturbation_propagation.hops=2
                                multitask.free_gene_dim=16) ;;
  # Propagation restricted to the SHARP graphs. coexpression (avg degree 151) and
  # experimental (124) SATURATE at hop 2 -- they reach 22,742 and 15,475 genes, both > N
  # = 6,607 -- so their hop-2 features are near-constant across genes and carry no
  # pair-(p,i) information at all. They are also 75% of all 2.44M edges, so they dominate
  # hop 1 with a very diffuse 1/151 signal. regulatory (avg degree 6.0, reaching 36 genes
  # at 2 hops) is the sharp, directed, biologically-right channel for a deletion effect.
  A6_prop_sparse)    OVERRIDES=("$PROP=true" model.perturbation_propagation.hops=2
                                "model.perturbation_propagation.graphs=[regulatory_interaction,tflink,physical_interaction]") ;;
  # NODE-EMBEDDING arm. The kNN probe (knn_embedding_probe.py) measures 1.7x more
  # transferable signal in prot_T5 (0.1174) than in calm (0.0692) on this same split and
  # metric, yet every 019 round trains on calm -- a categorical axis whose _006 "win" was
  # sampled alongside different lr/dropout draws, so it was never a controlled comparison.
  # prot_T5 is ALSO the organism-transferable representation (any proteome), unlike the
  # curated yeast graphs.
  B0_protT5)         OVERRIDES=("cell_dataset.node_embeddings=[prot_T5_all]") ;;
  # CONCAT arm. h_i and c_b currently meet exactly ONCE, by addition, at
  # `norm1(H_genes + attended)` -- so every head sees only g(h_i + c_b), a function of
  # their SUM. This hands the per-gene head [h_pert ; h_i ; c] so it can represent
  # arbitrary (h_i, c_b) interactions. Still permutation-equivariant (one shared MLP per
  # token, no gene-indexed parameters) and needs no graphs, so it transfers across
  # organisms -- unlike free_gene_dim, which is a per-gene lookup.
  C0_concat)         OVERRIDES=(multitask.concat_context=true) ;;
  C1_concat_prop)    OVERRIDES=(multitask.concat_context=true "$PROP=true"
                                model.perturbation_propagation.hops=2) ;;
  # Rank-32 bilinear h_i^T W_r c appended to the head input -- the parameter-lean way to
  # get the inner-product pair term that the sum + LayerNorm destroys.
  D1_bilinear32)     OVERRIDES=(multitask.bilinear_rank=32) ;;
  # Perceiver post-perturbation mixing: genes -> 32 latents -> genes. Restores cross-gene
  # interaction AFTER the perturbation at O(N*M): ~4.9 GFLOP and ~0.12 GB attention vs
  # ~503 GFLOP / ~25 GB for dense post-pert self-attention at B=32.
  E0_perceiver32)    OVERRIDES=(model.post_perturbation_mixing.enabled=true
                                model.post_perturbation_mixing.num_latents=32) ;;
  E1_perceiver64)    OVERRIDES=(model.post_perturbation_mixing.enabled=true
                                model.post_perturbation_mixing.num_latents=64) ;;
  # ---- PHASE 1: STABILITY. The same A0 config scores 0.1527 / 0.0235 / 0.0661 across
  # seeds 0/1/2, and seed 1 never escapes ~0.015 in 83 epochs -- 2 of 3 seeds simply fail
  # to train. That fragility is ~6x the largest architecture effect we have measured, and
  # every _005.._007 hyperparameter was selected at seed 0 alone.
  S1_warmup)         OVERRIDES=(regression_task.lr_scheduler.type=CosineAnnealingWarmupRestarts regression_task.lr_scheduler.first_cycle_steps=140 regression_task.lr_scheduler.max_lr=3e-4 regression_task.lr_scheduler.min_lr=1e-6 regression_task.lr_scheduler.warmup_steps=10) ;;
  S2_rezero)         OVERRIDES=(model.perturbation_head.residual=rezero) ;;
  S3_both)           OVERRIDES=(regression_task.lr_scheduler.type=CosineAnnealingWarmupRestarts regression_task.lr_scheduler.first_cycle_steps=140 regression_task.lr_scheduler.max_lr=3e-4 regression_task.lr_scheduler.min_lr=1e-6 regression_task.lr_scheduler.warmup_steps=10 model.perturbation_head.residual=rezero) ;;
  # ---- DECODER / PERTURBATION arms. PerGeneHead currently sees gene i's token and
  # NOTHING ELSE -- it never receives z_S, the pooled perturbed-set summary that
  # PerturbationHead uses and that underpins the trigenic-interaction result.
  F0_pert_set)       OVERRIDES=(multitask.pert_set_context=true) ;;
  F2_film)           OVERRIDES=(multitask.film_on_pert_set=true) ;;
  F1_deep2)          OVERRIDES=(model.perturbation_head.num_layers=2) ;;
  F1_deep4)          OVERRIDES=(model.perturbation_head.num_layers=4) ;;
  F3_deep4_zS)       OVERRIDES=(model.perturbation_head.num_layers=4 multitask.pert_set_context=true) ;;
  # ---- ENCODER arms. Encoder layers cost 17 GFLOP (batch-1, amortized over all 32
  # strains) vs 34.3 for a perturbation-transform layer, so capacity is 2x cheaper here.
  # G0 fixes the SILENT bug: graph_reg named `physical`/`regulatory` but cell_graph emits
  # `physical_interaction`/`regulatory_interaction`, so 2 of 9 heads were never
  # regularized in _006/_007/_008 -- and they are the two SHARPEST graphs.
  G0_fix_graphreg)   OVERRIDES=("model.graph_regularization.regularized_heads.physical_interaction.layer=[1]"
                                model.graph_regularization.regularized_heads.physical_interaction.head=0
                                model.graph_regularization.regularized_heads.physical_interaction.lambda=2.0e-7
                                "model.graph_regularization.regularized_heads.regulatory_interaction.layer=[1]"
                                model.graph_regularization.regularized_heads.regulatory_interaction.head=1
                                model.graph_regularization.regularized_heads.regulatory_interaction.lambda=2.0e-7) ;;
  # D2: masking REPLACES the KL (lambda -> 0), which removes the need for attention
  # weights and lets the fused SDPA kernel run -- the memory lever.
  D2_mask)           OVERRIDES=(model.attention_mask.enabled=true
                                model.graph_regularization.graph_reg_lambda=0.0) ;;
  G1_layers6)        OVERRIDES=(model.num_transformer_layers=6) ;;
  G1_layers8)        OVERRIDES=(model.num_transformer_layers=8) ;;
  GEARS_crossgene)   OVERRIDES=(model.cross_gene.enabled=true model.cross_gene.rank=64) ;;
  H0_factor)         OVERRIDES=(multitask.response_basis_rank=32) ;;
  E0_perceiver_on)   OVERRIDES=(model.post_perturbation_mixing.enabled=true
                                model.post_perturbation_mixing.num_latents=32
                                'model.post_perturbation_mixing.gate_mode=on') ;;
  F1_attn8)          OVERRIDES=(model.perturbation_head.num_layers=8
                                model.perturbation_head.extra_layer_ffn_mult=0) ;;
  F1_attn4)          OVERRIDES=(model.perturbation_head.num_layers=4
                                model.perturbation_head.extra_layer_ffn_mult=0) ;;
  # ======================= WAVE 3 (2026.07.29) =======================
  # Everything above was scored with a BROKEN instrument: score_decoder_arms.py derived
  # the arm from only free_gene_dim + perturbation_propagation, so C0/D1/D2/E0/F1/G1/H0/
  # GEARS/S* all resolved to the literal string "A0_baseline" and were averaged INTO the
  # baseline. On top of that, a source edit at 2026-07-28 22:26:23 split the runs into two
  # code versions that were then pooled. Wave 3 therefore re-measures from scratch, on one
  # frozen source, with a co-launched in-wave reference and WITHIN-SEED pairing.
  #
  # THE FINDING THAT ORDERS THIS WAVE: the model never fits its own TRAINING set.
  # train/expression/pearson_per_feature ends at 0.19-0.246 and is still climbing at the
  # cap; train/expression/loss falls 8% over 196 epochs; grad_norm 0.02-0.13 against a clip
  # of 10.0, so clipping never engages. Every architecture delta in _008 (<=0.03) was
  # measured on a model explaining ~5% of train variance. So the LADDER runs FIRST, and the
  # architecture arms run at whatever optimizer it selects.
  #
  # L: optimizer ladder, 2 seeds, A0 architecture. Zero code -- every key is already
  # declared at conf/cgt_expr_008.yaml:196-209.
  L0_lr3e4)          OVERRIDES=(regression_task.optimizer.lr=3e-4) ;;
  L1_lr1e3)          OVERRIDES=(regression_task.optimizer.lr=1e-3) ;;
  L2_lr3e3)          OVERRIDES=(regression_task.optimizer.lr=3e-3) ;;
  L3_lr1e3_cosine)   OVERRIDES=(regression_task.optimizer.lr=1e-3
                                regression_task.lr_scheduler.type=CosineAnnealingWarmupRestarts
                                regression_task.lr_scheduler.first_cycle_steps=300
                                regression_task.lr_scheduler.max_lr=1e-3
                                regression_task.lr_scheduler.min_lr=1e-6
                                regression_task.lr_scheduler.warmup_steps=0) ;;
  # A1_ref: the in-wave paired REFERENCE. Not optional and not substitutable by an
  # earlier A0 -- it must share the source hash, the epoch budget and the GPU packing.
  # Unpaired comparison is dead on arrival: across-seed sd of an identical config is
  # 0.0652 (41% of the score), so the 1-seed unpaired MDE is 0.181.
  A1_ref)            OVERRIDES=() ;;
  # A2: the round's only statistically solid win (+0.0231 paired, per-seed
  # +0.0224/+0.0229/+0.0240, sd 0.00084) -- but all three of those runs are PRE-EDIT, so
  # the architecture direction the program is leaning on has never run on this code.
  A2_prop_sparse)    OVERRIDES=("$PROP=true" model.perturbation_propagation.hops=2
                                "model.perturbation_propagation.graphs=[regulatory_interaction,tflink,physical_interaction]") ;;
  # A3: the flagship positive (+0.0102), one seed, never paired. Read it as a probe of
  # stage-2 FFN CAPACITY, not of depth: at |S|=1 layer 1's cross-attention has a single
  # key, so its output is gene-independent.
  A3_deep2)          OVERRIDES=(model.perturbation_head.num_layers=2) ;;
  # A4: the 1-parameter structural probe. One learned scalar on a zero-valued sink column
  # makes alpha_i = sigmoid(q_i.k_p/sqrt(d_k) - b), reviving 16,380 dead parameters and
  # making the context query-dependent (verified: spread 2.98e-09 -> 0.0624, W_Q/W_K grad
  # 0.0 -> 56645.7). Cheapest possible falsification of the set-encoder thesis.
  A4_nullsink)       OVERRIDES=(model.perturbation_head.null_sink=true
                                model.perturbation_head.null_sink_bias_init=-4.0
                                model.perturbation_head.null_sink_trainable=true) ;;
  # A5: the EMPIRICAL PAIRED NULL. Numerically the exact identity (p_null ~ 2e-9, verified
  # to 5.96e-08) on the SAME code path as A4. The paired sd every power calculation rests
  # on was measured only on arms that are exact identities at INIT; nothing establishes
  # that pairing transfers to an arm whose forward pass differs from step 0. A5 tests that.
  # Its paired delta must be ~0 with a CI containing 0, or the paired design is dead.
  A5_sham)           OVERRIDES=(model.perturbation_head.null_sink=true
                                model.perturbation_head.null_sink_bias_init=-20.0
                                model.perturbation_head.null_sink_trainable=false) ;;
  # ======================= WAVE 4 (2026.07.29) -- base config cgt_expr_010 =======================
  # Hard graph masking is now FROZEN (all 9 graphs, one per head, KL off), so these arms inherit
  # it rather than sweeping it. Launch with TORCHCELL_CONFIG=cgt_expr_010.
  #
  # TAGS ARE DECLARED HERE, NOT DERIVED LATER. The scorer once inferred the arm from two config
  # keys and silently collapsed eleven arms into the string "A0_baseline"; a post-hoc lookup
  # table keyed on arm name repeats the same mistake one level up. Declaring mech/xfer in the
  # SAME case branch that sets the overrides means the arm's identity and its properties cannot
  # drift apart. `xfer-no` marks a mechanism that needs curated yeast graphs and therefore
  # cannot be part of the organism-general story no matter how well it scores.
  R_ref)             OVERRIDES=()
                     ARM_TAGS=(mech-baseline xfer-yes stage-wave4b) ;;

  # ------------------------------------------------------------------ WAVE 5 (cgt_expr_011)
  # Launch with TORCHCELL_CONFIG=cgt_expr_011, which pins the PARTITION (split_seed 0) so
  # `seed` varies initialization only, and turns on the eval-mode train metric. Two
  # hypotheses plus two config-only probes.
  #
  # ------------------------------------------------------------------ WAVE 6 (cgt_expr_012)
  # Launch with TORCHCELL_CONFIG=cgt_expr_012. Two axes, one seed, no replicates.
  #
  # AXIS 1 -- PAIR-TERM RANK. Every arm here differs ONLY in how much (perturbation, gene)
  # interaction the architecture can express, so the axis is monotone and the question is
  # "does pair rank predict performance, and where does it saturate":
  #     V_ref      additive g(h_i + c_b)                      rank 0
  #     V_sink     gated attention, one bounded scalar/head    rank 9
  #     V_basis*   explicit factor model sum_j b_ij(h_i) a_j(z_S)   rank r
  #     V_film     diagonal multiplicative at the readout      rank d = 90
  #     V_hadamard diagonal multiplicative at the OPERATOR     rank d = 90
  # Headroom is not the constraint: the measured rank-r reconstruction ceiling is 0.727 at
  # r=32 and 0.780 at r=64 (lowrank_output_ceiling.json), both far above the 0.198 best. So a
  # null here is about the MECHANISM, not about capacity.
  #
  # AXIS 2 -- REGULARIZATION. train 0.64-0.72 vs val 0.20 is a generalization gap, and
  # weight_decay is currently 1e-8, i.e. effectively off, while 86% of parameters sit in the
  # encoder + embedding preprocessor (369,639 + 393,480 of 887,879). Dropout direction is
  # already measured: dropout 0.1 beat dropout 0 by +0.0145 / +0.0565 paired.
  #
  # NOT IN THIS WAVE: post-perturbation graph masking (needs batched masked attention over
  # [B, N, d], not built) and the masked-label objective (measured orthogonal to genotype --
  # conditioning_gain_after_genotype.json retains 97.5-100.6% after removing a genotype
  # predictor -- so it cannot move the m=0 metric and is an imputation capability instead).
  V_ref)             OVERRIDES=()
                     ARM_TAGS=(mech-baseline pair-rank0 xfer-yes stage-wave6) ;;
  V_sink)            OVERRIDES=(model.perturbation_head.null_sink=true
                                model.perturbation_head.null_sink_bias_init=0.0
                                model.perturbation_head.null_sink_magnitude_match=true)
                     ARM_TAGS=(mech-nullsink pair-rank9 xfer-yes stage-wave6) ;;
  V_basis16)         OVERRIDES=(multitask.response_basis_rank=16)
                     ARM_TAGS=(mech-basis pair-rank16 xfer-yes stage-wave6) ;;
  V_basis32)         OVERRIDES=(multitask.response_basis_rank=32)
                     ARM_TAGS=(mech-basis pair-rank32 xfer-yes stage-wave6) ;;
  V_basis64)         OVERRIDES=(multitask.response_basis_rank=64)
                     ARM_TAGS=(mech-basis pair-rank64 xfer-yes stage-wave6) ;;
  V_film)            OVERRIDES=(multitask.film_on_pert_set=true)
                     ARM_TAGS=(mech-film pair-rank90 xfer-yes stage-wave6) ;;
  # TWO HADAMARD ARMS, because the two inits answer different questions. `replace` drops the
  # additive path entirely, so at init (gamma=0) the model sees NO perturbation and must
  # learn the whole pathway -- the literal "assertion instead of cross-attention", and a
  # strictly harder start than the reference. `add` keeps the additive context, so it is
  # bit-identical to the reference at init (verified: max|diff| = 0.000e+00) and a null
  # result there is interpretable as "the multiplicative term does not help" rather than
  # "the arm never got off the ground".
  V_hadamard)        OVERRIDES=(model.perturbation_head.hadamard=replace)
                     ARM_TAGS=(mech-hadamard pair-rank90 xfer-yes stage-wave6) ;;
  V_hadamard_add)    OVERRIDES=(model.perturbation_head.hadamard=add)
                     ARM_TAGS=(mech-hadamard pair-rank90 xfer-yes stage-wave6) ;;
  V_drop2)           OVERRIDES=(model.dropout=0.2)
                     ARM_TAGS=(mech-regularization xfer-yes stage-wave6) ;;
  V_drop3)           OVERRIDES=(model.dropout=0.3)
                     ARM_TAGS=(mech-regularization xfer-yes stage-wave6) ;;
  V_wd1e4)           OVERRIDES=(regression_task.optimizer.weight_decay=1e-4)
                     ARM_TAGS=(mech-regularization xfer-yes stage-wave6) ;;
  V_wd1e2)           OVERRIDES=(regression_task.optimizer.weight_decay=1e-2)
                     ARM_TAGS=(mech-regularization xfer-yes stage-wave6) ;;

  # THE WAVE-5 REFERENCE. Every pool gets its OWN co-launched W_ref, because the three GPU
  # types available (RTX 6000 Ada on GilaHyper, A100 on mmli, RTX 6000 on cabbi) are not
  # interchangeable -- _007 pooled A100 with Ada runs and their means differ (0.0242 vs
  # 0.0182). A paired reference is only a valid denominator for arms that ran beside it on the
  # same hardware under the same packing, so it is re-run per pool rather than shared.
  W_ref)             OVERRIDES=()
                     ARM_TAGS=(mech-baseline xfer-yes stage-wave5) ;;

  # H1 -- IS TRAINING ACTUALLY SATURATED? The `train/...` pearson series everyone has been
  # reading is accumulated over training batches, so it is measured WITH DROPOUT ACTIVE and
  # across weights still updating inside the epoch: a biased-low estimate of the model's real
  # fit. `traineval/...` is the eval-mode answer. These arms run a long cap so the train
  # asymptote is observable, which is a CAPACITY measurement -- saturating far below 1.0 means
  # the model cannot fit 1244 strains and more data will not help.
  # H1_nodrop exists because dropout is a confound for exactly this question: the model
  # underfits (train pf ~0.30) with grad_norm ~0.05 against a clip of 10.0, so regularizing it
  # is plausibly harmful. Paired against H1_ref at the same init seed.
  H1_ref)            OVERRIDES=()
                     ARM_TAGS=(mech-baseline hyp-h1 xfer-yes stage-wave5) ;;
  H1_nodrop)         OVERRIDES=(model.dropout=0.0)
                     ARM_TAGS=(mech-regularization hyp-h1 xfer-yes stage-wave5) ;;

  # H2 -- DOES MIXING GENES AFTER THE PERTURBATION TELL OTHER GENES HOW TO EXPRESS? The
  # encoder runs on the WILDTYPE graph at batch 1, so `H_genes` is strain-INDEPENDENT and
  # nothing routes information between gene tokens once the perturbation is applied. Gene i
  # therefore learns about a deletion only through a shared additive context vector. X_mix
  # opens that channel (Perceiver-style latents over the perturbed tokens).
  # Status of the prior evidence: E0_perceiver_on measured -0.0302 and GEARS_crossgene
  # -0.0243, but both were n=1 on seed 0 alone, i.e. no significance and taken before we knew
  # cross-seed comparison is invalid. H2 is effectively UNTESTED, which is why it gets the
  # largest pool and 9 paired replicates.
  # NOTE what is NOT here: the masked-label objective that would FORCE the channel to be used.
  # `model.observed_labels.enabled` builds ObservedLabelEncoder, but train_cgt_multitask.py
  # never passes observed_values/observed_mask, so the module receives nothing and the arm is
  # inert. Wiring it needs per-batch label masking AND restriction of the loss to the held-out
  # entries -- without the second part the model trivially copies the observed values and the
  # train fit explodes for no reason. Deferred to its own wave rather than shipped half-built.
  X_mix)             OVERRIDES=(model.post_perturbation_mixing.enabled=true
                                model.post_perturbation_mixing.num_latents=32
                                model.post_perturbation_mixing.gate_mode=on)
                     ARM_TAGS=(mech-postpert-mixing hyp-h2 xfer-yes stage-wave5) ;;
  X_mix64)           OVERRIDES=(model.post_perturbation_mixing.enabled=true
                                model.post_perturbation_mixing.num_latents=64
                                model.post_perturbation_mixing.gate_mode=on)
                     ARM_TAGS=(mech-postpert-mixing hyp-h2 xfer-yes stage-wave5) ;;
  # gate_mode=rezero starts the channel CLOSED (beta init 0), so this separates "the mechanism
  # does not help" from "the optimizer never opened the door". Both are logged via gate/.
  X_mix_rezero)      OVERRIDES=(model.post_perturbation_mixing.enabled=true
                                model.post_perturbation_mixing.num_latents=32
                                model.post_perturbation_mixing.gate_mode=rezero)
                     ARM_TAGS=(mech-postpert-mixing hyp-h2 xfer-yes stage-wave5) ;;

  # MASK DEPTH. Not the before/after-perturbation contrast that was asked for -- true
  # post-perturbation graph masking needs a new masked attention layer over the [batch, N, d]
  # perturbed tokens and is NOT built. These arms move the mask WITHIN the encoder, which is
  # config-only: layer 4 is the last encoder layer, i.e. as late as the graph can currently be
  # applied. All of these remain PRE-perturbation, so none of them can express
  # deletion-neighbour coupling; the point is only to test whether mask depth matters at all
  # before spending code on the post-perturbation version.
  # Encoder layers are 0-INDEXED, so the last of the four is 3 -- `layers=[4]` raises. The
  # default is [1]; layer 3 is as late as the graph can currently be applied.
  G_l3)              OVERRIDES=("model.attention_mask.layers=[3]")
                     ARM_TAGS=(mech-maskdepth xfer-yes stage-wave5) ;;
  G_l13)             OVERRIDES=("model.attention_mask.layers=[1,3]")
                     ARM_TAGS=(mech-maskdepth xfer-yes stage-wave5) ;;

  # PROBES from the 007 review: its top 12 runs cluster at lr 1e-4 (6 of 12) and the best two
  # are 1e-4 and 5e-4, yet the wave-3 ladder swept 3e-4 / 1e-3 / 3e-3 and NEVER tested below
  # 3e-4 before freezing 3e-4. Seed-0 order statistics, so a lead rather than a result.
  P_lr1e4)           OVERRIDES=(regression_task.optimizer.lr=1e-4)
                     ARM_TAGS=(mech-optimizer hyp-lr xfer-yes stage-wave5) ;;
  P_lr3e5)           OVERRIDES=(regression_task.optimizer.lr=3e-5)
                     ARM_TAGS=(mech-optimizer hyp-lr xfer-yes stage-wave5) ;;
  # The null sink RETRIED STARTING OPEN. The first attempt (bias_init -4.0) was chosen to keep
  # the arm attributable and that backfired: read from all four last.ckpt at epoch ~187,
  # null_bias had moved only -4.0 -> -3.92/-3.95, so the sink carried ~1.9% of the attention
  # mass versus ~1.8% at init. The paired delta was ~0, which LOOKED like a clean negative but
  # was really "the mechanism was never in the forward pass". bias_init 0.0 puts p_null at 0.5,
  # so the model must ACTIVELY CLOSE the sink for it to be useless -- the arm becomes
  # falsifiable in both directions. gate/null_bias is now logged every train epoch, so this is
  # visible during the run instead of post-mortem from a checkpoint.
  N_sink0)           OVERRIDES=(model.perturbation_head.null_sink=true
                                model.perturbation_head.null_sink_bias_init=0.0
                                model.perturbation_head.null_sink_trainable=true)
                     ARM_TAGS=(mech-nullsink xfer-yes stage-wave4) ;;
  # N_sink_mm SUPERSEDES N_sink0. Same open gate, but the attended context is rescaled by
  # 1/sigmoid(-bias_init) so the arm no longer carries a permanent ~2x attenuation alongside
  # its pair term. Measured on the real module (N=2048, d=90, 9 heads, |S|=1): the unmatched
  # sink gives ||c||/||c_ref|| = 0.5129, matched gives 1.0258, and the per-query CV -- the
  # pair term itself -- is 0.0971 in BOTH, versus 3.24e-07 for the reference. So matching
  # removes the confound without touching the mechanism. N_sink0 is kept only so the
  # already-launched unmatched runs remain identifiable.
  N_sink_mm)         OVERRIDES=(model.perturbation_head.null_sink=true
                                model.perturbation_head.null_sink_bias_init=0.0
                                model.perturbation_head.null_sink_trainable=true
                                model.perturbation_head.null_sink_magnitude_match=true)
                     ARM_TAGS=(mech-nullsink xfer-yes stage-wave4b) ;;
  *) echo "unknown arm '$ARM'" >&2; exit 1 ;;
esac

# PYTHONPATH pins the WORKTREE's torchcell: without it a script run from a worktree
# silently imports the PRIMARY checkout's package, and none of the _008 model code exists
# there -- the run would look like a baseline and quietly invalidate the comparison.
export PYTHONPATH="$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export WANDB__SERVICE_WAIT=600

# Explicit interpreter, NOT bare `python`. The slurm launchers get the env from
# `conda activate torchcell`; this script is invoked directly from a shell where the base
# env is active, and bare `python` there resolves to base conda -- which has no hydra, so
# every run died instantly with ModuleNotFoundError.
PYTHON="${TORCHCELL_PYTHON:-$HOME/miniconda3/envs/torchcell/bin/python}"

# WAVE-WIDE overrides (epoch budget, early stopping) applied identically to every arm.
# They live here rather than in each arm's case branch because a budget that differs
# between arms is a confound: it scores "how long did the plateau look flat" alongside
# "how good is the mechanism". Word-split on purpose so the launcher can pass several.
read -r -a EXTRA <<< "${TORCHCELL_EXTRA_OVERRIDES:-}"

# Tag list: config, arm, seed, plus whatever the arm DECLARED above. Arms predating wave 4 set
# no ARM_TAGS, so they keep exactly their old three tags and stay comparable.
TAGS="${TORCHCELL_CONFIG:-cgt_expr_008},$ARM,seed$SEED"
for t in ${ARM_TAGS[@]+"${ARM_TAGS[@]}"}; do TAGS="$TAGS,$t"; done

cd "$PROJECT_ROOT/experiments"
exec "$PYTHON" 019-simb-multimodal/scripts/train_cgt_multitask.py \
  --config-name "${TORCHCELL_CONFIG:-cgt_expr_008}" \
  seed="$SEED" \
  "wandb.tags=[$TAGS]" \
  "${OVERRIDES[@]}" \
  ${EXTRA[@]+"${EXTRA[@]}"}
