---
id: gicl51f5tth62lgegvvsxi5
title: Cgt_expr_v17_locality
desc: ''
updated: 1789786209563
created: 1789786209563
---

## 2026.09.18 - The perturbation-locality round, submitted as IGB 2409262

The v13 reference against two arms that give the deletion a location on the gene tokens. Built after the expression-fit review ([[experiments.019-simb-multimodal.expression-fit-review]]) and the B4 baseline ([[experiments.019-simb-multimodal.scripts.graph_retrieval_baseline]]): in the reference the deletion enters after the encoder as one 90-d vector added to every gene token, the deleted gene's own value is predicted at chance (a constant there is worth +0.014 val / +0.015 test), and the deleted gene's interaction-graph neighborhood is the strongest retrieval key measured (STRING-experimental adjacency 0.194 val / 0.175 test on gene-disjoint strains against the CGT dumps' 0.178 and 0.100 val / 0.097 and 0.134 test). The nine graphs reach the model only as an attention mask on encoder layers that run before the deletion exists, so a different layer pattern (more masked layers) cannot route the deletion; the post-perturbation propagation module can.

Arms (`gh_expr_008_arm.sh`, `L_*`), everything else v13's trunk, objective, mask schedule `[0, 10, 100, 1000]`, optimizer and E_full input:

| arm | override | isolates |
|---|---|---|
| `L_ref_s<k>` | none | the reference at 1,200 epochs |
| `L_self_s<k>` | `model.perturbation_propagation.enabled=true hops=0 gate_mode=on` | one hop-0 feature per token, "this gene is deleted in this strain" |
| `L_prop2_s<k>` | `enabled=true hops=2 gate_mode=on`, graphs `[]` = every gene-gene relation | self plus 1-hop and 2-hop reachability of the deletion along each of the nine graphs (19 features) |

L_self against L_ref is "identify the deleted gene"; L_prop2 against L_self is "route along the graphs". The gate is forced on because the one earlier test (wave-1 `A2_self`) ran with a ReZero gate closed at initialization, n = 1. `gate_mode` is passed on the command line as a string; in YAML the bare token `on` parses as boolean true.

Design: split seeds 0 to 3, init seeds 0 to 2, the three arms of one (split, seed) co-resident on one card, 36 runs as twelve tasks of three (`W5_STAGE=locality` in `igb_expr_wave5.slurm`), 1,200 epochs, `save_loss_min` on. Score = mean validation Pearson over epochs 1,000 to 1,200, paired t on the 12 differences per contrast, best-by-metric checkpoint read on test at the end; adopt only above +0.02 with the CI excluding 0, 3 of 4 partitions positive and the test sign agreeing. Read the expression metric on all held-out strains and on the gene-disjoint subset ([[experiments.019-simb-multimodal.scripts.split_gene_overlap_audit]]).

Smoke on GilaHyper CPU: fast-dev runs of both arms (job 2373; the composed configs carry `enabled: true`, `gate_mode: 'on'`, hops 0 and 2) and a one-epoch run of L_self (job 2374) that wrote `best`, `best-metric`, `best-loss` and `last`.

Submitted 2026-09-18 from the IGB checkout at 51c792acc (clean tracked tree) as array 2409262 on the `gpu` A40s, tasks 0 to 11, queued behind v13 (2397311, ~26 h left) and another group's pending array. W&B project `torchcell_019_expr_v17`, arm tags `pert-broadcast`, `pert-self-hop0`, `pert-prop-hop2`, `stage-locality`, `round-locality`.

Hypotheses (untested): L_self recovers at least the +0.014 / +0.015 of the constant substitution; L_prop2 recovers part of the gap to the graph kNN; neither changes the k>0 branches' early peak.
