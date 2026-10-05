# Reviewer 10 of 10: compute, logistics and the 20-day skeleton (2026-10-04)

Read-only audit by an independent agent; the cluster state was given to it as of 2026-10-04 and is not all in the repository; estimates and hypotheses are labeled as such.

I could not keep the throughput section to one table plus short lines. A small-model screen has never been measured, so every small-run rate below is labeled as a hypothesis. All of my reads were read-only, and nothing was submitted.

## 1. Measured throughput and capacity

Wall time per run comes from W&B `_runtime/epoch`. "Step only" is the median of `perf/epoch_seconds`.

| family, packing, card | s/epoch (wall) | 1,200 epochs | GPU-h/run, 1,200 / 300 epochs |
|---|---|---|---|
| v19 single-head, 6.67M params, 3 per A40 (gpu) | 43–46 (step only 38–39) | 14.4–15.3 h | ~5.9 / ~1.5 (derived) |
| v19 joint arm, same pack | 52–53 | 17.3–17.7 h | same pack |
| v19, 3 per RTX 6000 Ada (cabbi) | 41.5–61.6 | 13.8–20.5 h | ~6.8 / ~1.7 |
| v20 conditioned, 4 per Ada | 96–100 | ~33 h (given 33–35) | ~8.3 / ~2.1 |
| 025 interaction S3, 4×A100 mmli | 36 min | 130 epochs ≈ 3.3 d | n/a |
| 025, A40 on Delta, LMDB staged to node NVMe | 20–27 min | 30 epochs = 16 h | n/a |

Sources:

- v19 gpu rows: runs `rnosud90` and `qotaa6tf`, model config hidden 90, 6 layers, batch 32, `n_train_supervised` 1,107.
- v19 cabbi row: runs `kct7diof` and `ipwprg20`.
- v20 row: `u6p781n5` reached 157 epochs in 4.4 h.
- 025 rows: /Users/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-per-entry-dataset-token/notes/experiments.025-solid-growth.scripts.igb_mmli_cgt.md L132/L144 and /Users/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-per-entry-dataset-token/notes/experiments.025-solid-growth.scripts.delta_cgt.md L85/L184.

https://wandb.ai/zhao-group/torchcell_019_prot_v19/runs/qotaa6tf

https://wandb.ai/zhao-group/torchcell_019_prot_v20/runs/u6p781n5

**Small-model screen.** Hypothesis, unmeasured: at hidden 32 and 2 layers, a run of under 1M params costs 1 to 3 times less than v19, about 0.5–1.5 GPU-h per 300 epochs. That is roughly 16–48 runs per card-day. Host memory is 39.5 GB per 4-run pack (canary 2423182).

| resource, next 20 days | free cards | small 300-epoch runs per day (estimate) | full 1,200-epoch runs in window |
|---|---|---|---|
| IGB gpu (3 at once) | 3 now (v19 done) | 48–144 | ~240 |
| IGB cabbi | 0 until v20 ends, about Oct 13 (given), then 2 | 32–96 from about day 9 | ~80 |
| IGB mmli | 0–1 (given) | 0–48 | uncertain |
| Delta A40x4, 12 h class | 4 per node-job, after a 7.3 h median wait (skill table) | 32–96 per node-job | the balance of 28,833 GPU-h (given) is not the limit; the queue is |
| Mac M1 Max (10 cores, 32 GB, MPS available, torch 2.14) | 1 | 2–6 (hypothesis) | none; use for smoke tests |

**The Mac can probably train at small size, untested.** The trainer only picks CUDA-or-CPU for an early `.to(device)` (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/train_cgt_multitask.py L2827). Lightning takes `trainer.accelerator` from the config, so the run needs the overrides `trainer.accelerator=mps trainer.precision=32-true` (v19 sets `gpu` and `bf16-mixed`). I found no CUDA-only call in the model; `scatter_mean` is only used in HyperSAGNN.

**Data each resource needs for small runs:**

- **IGB:** has `fig3_proteome` (the both-label rows come from the `require_modalities` filter) and `fig3_core`. Nothing to move.
- **Delta:** has `fig3_core`, so expression and morphology screens could start there. It lacks `fig3_proteome`.
- **Mac:** has no 019 store; /Users/michaelvolk/Documents/projects/torchcell/data/torchcell/experiments holds no 019 tree. It needs `processed/` plus `data_module_cache/` copied from IGB. `Neo4jCellDataset` otherwise tries to rebuild from Neo4j, which lives on GilaHyper (offline). The graphs, ProtT5/ESM2 embeddings and SGD files are already on the Mac.
- **Shortest path:** rsync from biologin to the Mac (an allowed login-node action), then the Mac to Delta, because /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/sync_igb_fig3_core.sh assumes GilaHyper as the source. `fig3_core` is about 13 GB (script header); the size of `fig3_proteome` is not recorded.
- **Delta also needs:** a launcher for the `joint_clean` and `conditioned` stages built on /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/delta_grid_common.sh (only an old 4-GPU DDP launcher exists), a `.env` in the detached worktree (missing `.env` is what killed canary 2423054), a preflight and `--test-only`. Estimate: 2–3 days to the first Delta number.

## 2. What can start today with no GPU, and what is blocked on what

On the Mac now:

1. Ridge baseline for morphology on the Ohya tables plus ProtT5/ESM2 embeddings. The gate table says no linear baseline for morphology is on file (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/figure-3-gate/sections/3-remaining.tex).
2. Metrics restricted to the 116 reliable ("moving") CalMorph features, for the morphology baselines and kNN.
3. Count of deletion-keyed studies in SPELL (/Users/michaelvolk/Documents/projects/torchcell/data/sgd/spell, 1,211 entries).
4. Fitness (`smf_costanzo2016`) against the size of the Kemmeren expression response.
5. Reconcile the expression-morphology overlap (1,438 against 1,440).
6. Rescore the expression baselines under one scoring rule.
7. v20 readouts from W&B, after each login-node sync.

Blocked until the IGB store is copied, or runs as an IGB CPU job:

- Matched baselines on the both-label store.
- The proteome-expression side of the ridge triangle with a permuted null.
- The proteome strand of the pair-form graph probe.
- The 2026.09.08 proteome-expression covariation check.

Blocked on GilaHyper (new Neo4j builds):

- A joined proteome-plus-morphology store (panel A,e).
- Any natural-isolate store.

## 3. Risks and single points of failure

- **GilaHyper offline.** It holds the mirror, the builds and Neo4j, so no new store can be built. Fallback: IGB holds the only `fig3_proteome` copy, so make two copies now (Mac and Delta), and drop panel A,e.
- **cabbi held by another user.** v20 occupies the two free cards until about Oct 13. Fallback: cancel v20's pending tasks and resubmit them to gpu `%3` from the same pinned worktree at f09091e2f. That takes cards from the screen, so do it only if the screen finishes early.
- **Delta queue.** p90 waits are 32 h in the 12 h class and 79 h in the 20 h class (skill table). Fallback: keep every Delta job at 12 h or less, submit everything with no dependencies, and treat Delta as overflow, never the critical path.
- **Losing the commits behind running and finished jobs.** Landing the branch rebases its 169 commits, so f09091e2f (v20) and 24163737e (v19, W&B `source.git_hash`) leave the history once the branch ref is deleted. The running job is safe because it runs from its own detached IGB worktree. Fallback: tag both commits before landing.
- **Screen winners may not hold at full size.** Hypothesis, untested. Fallback: put a co-launched reference arm in every pack, and confirm on all 12 split seeds.
- **mmli occupancy.** Plan it at zero cards.

## 4. 20-day skeleton (Day 0 = Oct 4, deadline about Oct 24)

- **D0–1:** start the Mac analyses from section 2. Rsync `fig3_proteome` from IGB to the Mac. One-hour small-model packing benchmark on one gpu card (/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/gh_packing_benchmark.sh needs an IGB version). Mac MPS smoke test at 2 epochs.
- **D1–2:** port perturb_cls onto the 019 model, using the W30 notes' parity check. Detail: commit 9e2f59551 is 98 model lines, is not on main, and both lines have touched the model file since their merge base (237 and 119 lines changed). Estimate about 1 day. Build the operator-screen configs: null sink on and off, perturbed-CLS readout, softmax against sigmoid weight, 2-hop propagation, morphology on its 4,718 strains read from the perturbed CLS. Write the Delta launcher and sync the store there.
- **D2–7:** screen: small models, 12 split seeds, 300 epochs, on gpu ×3, Delta (12 h class) and mmli if a card frees.
- **G1, D4:** benchmark measured, and the no-GPU panels drawn. The "baseline and ceiling benchmark" framing needs no GPU and is the guaranteed floor.
- **Natural-isolate data, drop-dead D5:** if GilaHyper is not back with a build, drop it.
- **G2, D7:** read the screen. "Genotype-only improvement from an operator fix" is dropped if no arm beats its paired reference on at least 9 of 12 partitions.
- **D8–15:** confirm at most 2 winners at full size: 12 partitions × (winner + reference) = 24 runs, about 2.5 days on 3 A40s. Add cabbi after v20 ends.
- **G3, about D10:** v20 has 8 or more partitions. Cross-modal conditioning must beat the observed-modality ridge (0.226 and 0.218, /Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/019-simb-multimodal-expression/sections/6-checkpoint.tex and the 09.27 review), not the unconditioned arm. Drop-dead D11.
- **D16:** freeze. **D16–19:** test pass, figure, text.

## 5. Top three recommendations

1. **Build the benchmark framing on the Mac this week.** It is the only framing that is certain to exist. Cost if wrong: a few CPU hours.
2. **Screen on the IGB gpu partition first.** The store and the pinned-worktree routine are already proven there; Delta and the Mac come second. Cost if wrong: about 2 days of Delta setup that turns out unnecessary, or winners that do not hold at full size, which the full-size confirmation phase covers.
3. **Tag f09091e2f and 24163737e, then land the 019 branch, then port perturb_cls on a fresh branch.** New jobs then come from new detached worktrees. Leave the dataset token (603 trainer lines) and the Wasserstein loss (/Users/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-per-entry-dataset-token/torchcell/losses/point_dist_graph_reg.py, which was trained on 1.05M records) out of the window. The 025 result worked at 0.478 against 0.439 (/Users/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-per-entry-dataset-token/notes/experiments.025-solid-growth.mmli-campaign.md L17) on about a thousand times more rows than Figure 3's 1,107 training strains. Cost if wrong: if perturb_cls is the lever, it is in the screen; a deferred loss term costs one arm.

Files:

/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/igb_expr_wave5.slurm
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/train_cgt_multitask.py
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/sync_igb_fig3_core.sh
/Users/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/notes-tex/figure-3-gate/sections/3-remaining.tex
/Users/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-per-entry-dataset-token/notes/experiments.025-solid-growth.scripts.igb_mmli_cgt.md
