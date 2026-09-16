---
id: rzuldem67xanozz5ll3aq6c
title: 028 Gene Essentiality
desc: ''
updated: 1789537997385
created: 1789537997385
---

## 2026.09.16 - The essentiality line becomes its own experiment

Gene essentiality began as wave 9 of [[experiments.027-betaxanthin-metabolic]] (`train_bx.py --head essentiality`) and is moved here whole: the trainer (`scripts/train_essentiality.py`, the 027 script with paths repointed and the head fixed to essentiality), the union build (`$DATA_ROOT/data/torchcell/experiments/028-gene-essentiality/essentiality_build`, a copy of the 027 build so the other session's running wave-11 jobs keep theirs), the wave-9 to wave-11 result files, checkpoints (`$DATA_ROOT/checkpoints-028`) and per-gene dumps (`$DATA_ROOT/test-predictions-028`). W&B project: `zhao-group/torchcell_028_gene_essentiality`. The 027 history is in [[experiments.028-gene-essentiality.wave9-essentiality]].

There is no fitness objective anywhere in this experiment. The union build carries single-mutant fitness for 4,540 genotypes only because SGD's essential-gene list has no negative class; every fitness record arrives under a False essentiality mask and contributes no loss. The objective is one logit against a 0/1 label (binary cross entropy), selection is validation AUROC under a centered 5-epoch moving average, and the reported number is the AUROC on the released held-out genes at that epoch.

### What was measured before the move (027 waves 9 to 11)

- Pooled backbone at width 90 (958,501 parameters), released labels, 80 epochs: pinned test AUROC 0.793 / 0.788 / 0.578 at seeds 2700 to 2702 (jobs 2009, 2021, 2022); the third seed collapsed to a constant. Flux arm (biomass column of the projected flux, `FluxBiomassReadout`): 0.540 / 0.758 / 0.512, paired mean minus 0.117, 3 of 3 negative.
- Width 12, one block, one-layer preprocessor (16,874 parameters): 0.8011 / 0.7860 / 0.7958 (jobs 2045 to 2047), against FCL 0.742 and FBA 0.640 on the same 195 genes (`fcl_comparison.py`). Width 24: 0.7429 / 0.7398 / 0.8072.
- The validation curve at width 90 moves by a median of 0.034 to 0.072 between adjacent epochs; at widths 24 and 12 by 0.0067 (`curve_stability.py`).

### Three defects in the wave-9 protocol, and the split file that removes them

The data module drew train/validation at 80/10/10 over the 5,461 non-test records, of which 926 carry a label. Measured consequences: a validation set of 75 labeled genes with about 12 essential, on which one AUROC read has a Hanley-McNeil standard error near 0.07 (hypothesis for the bumpy curves, tested by wave 12); 10 percent of the labeled genes in a random test block nothing reports; 4,535 unlabeled records forwarded per epoch under a False mask, about 6 times the labeled work.

`scripts/build_essentiality_splits.py` writes one file per design under `results/splits/`: the per-gene labels, their provenance, and pinned train/val/test record indices. The trainer's `--split-file` pins all three splits and drops every other record (`CellDataModule(drop_unpinned=True)`, new in `torchcell/datamodules/cell.py`). The test block is always the released held-out genes (195 of 223 resolve in the build).

| split file | labels | train genes (ess) | val genes (ess) | test genes (ess) |
|---|---|---|---|---|
| `ess_fcl_val0.2_s0` | released, 1,121 | 629 (100) | 158 (25) | 195 (31) |
| `ess_genome_val0.2_s0` | released + SGD essential + viable single-mutant fitness, 5,795 | 4,368 (879) | 1,093 (220) | 195 (31) |

The genome source keeps the released label on every disagreement (12 genes, 11 of them SGD-essential that the released split calls nonessential, the same 11 `essentiality_label_agreement.py` found), so the held-out genes are scored on the comparator's labels under either source.

### FBA on four media

`scripts/fba_essentiality_baseline.py` (job 2072): cobrapy `single_gene_deletion` on yeast-GEM 9.0.2, the essentiality score being the negative knockout-to-wild-type growth ratio, rounded to 1e-3 so the ~900 deletions that grow at 0.99 to 1.00 of the wild type rank as the tie they are (without rounding two identical runs differed by 0.03 AUROC on the held-out genes, jobs 2069 and 2070).

| medium | WT growth (1/h) | lethal of 1,121 | AUROC all 1,121 | AUROC held-out 223 | AUROC our 195 [95% CI] |
|---|---|---|---|---|---|
| shipped bounds | 0.0858 | 186 | 0.707 | 0.633 | 0.654 [0.562, 0.748] |
| SM | 0.3138 | 160 | 0.707 | 0.642 | 0.666 [0.571, 0.759] |
| SC | 0.5428 | 91 | 0.719 | 0.653 | 0.666 [0.574, 0.759] |
| YPD approx | 0.5352 | 104 | 0.715 | 0.653 | 0.664 [0.572, 0.756] |

The medium moves the lethal count by a factor of two and the AUROC by 0.01. The published 0.69 on all labeled genes is reproduced to within 0.02 on our GEM copy and our bounds.

### Wave 12, queued 2026-09-16 00:55 (jobs 2071 smoke, 2073 to 2093)

Pooled arm, width 12, one block, one-layer preprocessor, 80 epochs, seeds 2700 to 2702, all scored on the same 195 genes:

| arm tag | change from the wave-11 width-12 cell | decides |
|---|---|---|
| `w12base` | none (wave-11 protocol in the 028 experiment) | the port: same numbers at the same seed |
| `w12fcl` | split file `ess_fcl_val0.2_s0` | whether a 158-gene stratified validation set smooths the curve and changes the selected epoch |
| `w12gen` | split file `ess_genome_val0.2_s0` | whether five times the labels move the held-out AUROC |
| `w12fclwd0`, `w12fclwd1e-2`, `w12fclwd1e-1` | AdamW weight decay 0, 1e-2, 1e-1 (base 1e-4) | the weight-decay dose response |
| `w12fclb32` | batch 32 (base 128) | whether 20 steps per epoch instead of 5 changes the curve |

Launcher: `scripts/gh_train_essentiality.slurm` through `scripts/freeze_and_submit.sh`, one environment per cell. Views: `scripts/wandb_views.py --contrast w12`.

### Wave 13, queued 2026-09-16 01:20 (jobs 2113 to 2130): the prediction routed through the module

Same width-12 cell, same fcl split and base hyperparameters as `w12fcl`, so every cell pairs by seed with a pooled cell. The flux arm is the wave-9 recipe (`flux_projection_anchored`, predicted kcat tier, union thermodynamics, YPD-approx bounds written onto the GEM, 20-epoch constraint warmup). Two readouts, two constraint scales, and the physics-off arm:

| arm tag | arm | readout | constraint scale | what the backbone's gradient passes through |
|---|---|---|---|---|
| `w13fluxbio` | anchored | growth column only | 0.1 | the flux layer only (wave-9 design) |
| `w13fluxbio_s0` | anchored | growth column only | 0 | the flux layer only, no physics penalty |
| `w13fluxctx` | anchored | context head + growth column | 0.1 | both paths |
| `w13fluxctx_s0` | anchored | context head + growth column | 0 | both paths, no physics penalty |
| `w13fluxoff` | stoichiometry only | growth column only | 0.1 | the flux layer with no thermodynamics or capacity |
| `w13fluxoffctx` | stoichiometry only | context head + growth column | 0.1 | both paths |

Per-module gradient norms are logged every epoch (`grad_norm_flux_layer_mean`, `grad_norm_readout_mean` beside the transformer's and the preprocessor's); on the CPU smoke of the context-plus-growth readout the flux layer carried 0.82 against the transformer's 0.68 at epoch 0 (anchored) and 6.6 against 0.45 (stoichiometry only). `--ess-readout` selects the readout (`ContextBiomassReadout` in `train_essentiality.py`).

## 2026.09.16 - Unified in one worktree; rounds 1 and 2 requeued with the 027 levers folded in

The 028 branch built in the other session was cherry-picked onto `feat/kinetics-equilibrator-datasets` (commits 3043dffd to cc60b17f, then 5808b618) and its worktree and branch were retired. The job ids in the two sections above (2071 to 2093, 2113 to 2130) were cancelled before any started, as were the 027 wave-12 cells (2094 to 2148) that duplicated them. The 027 essentiality line (`train_bx.py --head essentiality`, waves 9 to 12) ends here; its wave-11 `_nohead` reruns (jobs 2062 to 2068) are copied into this experiment's results and dumps when they land.

Folded into `train_essentiality.py` from the 027 side: the `gated` readout (`context_biomass` behind a zero-initialized gate, `ess_gate` logged per epoch, so at initialization the cell is the pooled model exactly), `--pos-weight-balanced` (772/126), an `ESS_GRAPH_REG` switch on the launcher, and the model change that removes the unused perturbation readout head (16,381 parameters at width 90, 1,201 at width 24, no gradient on any cell). Cells from here on are therefore not bit-identical at a seed to wave 11.

**Round 1, jobs 2149 to 2181 (33 cells)**, width 12, one block, preprocessor 1, 80 epochs, seeds 2700 to 2702. Beyond the wave-12 table above (base, fcl split, genome labels, weight decay 0 / 1e-2 / 1e-1, batch 32), all on the fcl split: learning rate 3e-4 and 3e-3, the class weight, and graph regularization off.

**Round 2, jobs 2182 to 2208 (27 cells)**, same cell on the fcl split. Beyond the wave-13 table above: the `gated` readout at constraint scale 0.1 and 0, and the shuffled-kcat control (`--flux-shuffle-kcat <seed>`) with the forced readout at scale 0.

Reads, pre-registered: (1) the bumpy-curve hypothesis is decided by `w12fcl` against `w12base`, median absolute validation step on 158 labeled genes against 75; (2) each round-1 lever earns a follow-up only when three of three seeds move the same way against `w12fcl`; (3) "not hurt" is `w13fluxgate*` against `w12fcl` paired by seed with the gate's trajectory beside it; "help" is `w13fluxbio_s0` above `w12fcl` and above `w13fluxshuf_s0`; the collapse's cause is `w13fluxbio` against `w13fluxbio_s0`. W&B project: `https://wandb.ai/zhao-group/torchcell_028_gene_essentiality`.

## 2026.09.16 - Round 1 read: the cells are under-trained, the class weight helps, weight decay and graph regularization do nothing

All 33 round-1 cells completed (jobs 2149 to 2181; `results/w12_readout.json`, figure `notes/assets/images/028-gene-essentiality/w12_readout_2026-09-16-10-32-55.svg`). Held-out AUROC on the 195 genes, seed mean over 2700 to 2702, with the pooled-logit gene bootstrap:

| arm | mean | sd | pooled [95% CI] | selected epochs | median val step |
|---|---:|---:|---|---|---:|
| base, wave-11 protocol, 75-gene validation | 0.759 | 0.020 | 0.760 [0.670, 0.844] | 18, 38, 14 | 0.0056 |
| fcl split, 158-gene validation | 0.725 | 0.019 | 0.738 [0.645, 0.826] | 66, 77, 77 | 0.0026 |
| fcl, weight decay 0 / 1e-2 / 1e-1 | 0.725 / 0.726 / 0.726 | 0.02 | 0.735 each | same | same |
| fcl, graph regularization off | 0.725 | 0.019 | 0.738 | same | same |
| fcl, lr 3e-4 / 3e-3 | 0.658 / 0.686 | 0.09 / 0.04 | 0.732 / 0.713 | 77, 77, 74 / 77, 77, 50 | |
| fcl, batch 32 | 0.765 | 0.005 | 0.764 [0.673, 0.852] | 74, 73, 77 | 0.0032 |
| fcl, class weight 772/126 | 0.759 | 0.016 | 0.769 [0.680, 0.858] | 77, 75, 53 | 0.0034 |
| genome labels, 1,093-gene validation | 0.704 | 0.024 | 0.705 [0.606, 0.802] | 77, 77, 69 | 0.0020 |

- **The bumpy-curve hypothesis holds.** The median epoch-to-epoch validation step halves from 0.0056 on 75 labeled genes to 0.0026 on 158 and 0.0020 on 1,093 (panel b of the figure). The old protocol's curve reads 0.8 on its own 75 genes and the split cells read 0.65 to 0.72 on theirs; the levels are not comparable, only the steps are.
- **The fcl split scores below the old protocol, and the reason is training genes, not validation.** The split trains on 629 labeled genes against the old draw's roughly 740 (80 percent of 926), and every fcl cell selected epoch 66 to 77 of 80 with its validation curve still rising (panel a), so the cells are under-trained at 80 epochs with 5 optimizer steps per epoch. Hypothesis (tested by round 3): the 0.034 gap closes with more optimization.
- **Weight decay does nothing at 0, 1e-2 or 1e-1**, to the third decimal, and neither does removing graph regularization: at 400 optimizer steps with lr 1e-3 a decay of 0.1 shrinks a weight by 4 percent, and the graph term's lambda is 6.5e-5. Both levers are closed on this endpoint.
- **Batch 32 moves three of three seeds up** (fcl 0.7455 / 0.7217 / 0.7085 to 0.7657 / 0.7701 / 0.7594; +0.020, +0.048, +0.051) and collapses the seed spread to 0.005. It is four times the optimizer steps per epoch, which is the same under-training story.
- **The class weight moves three of three seeds up** (to 0.7642 / 0.7722 / 0.7410; +0.019, +0.051, +0.033). Its curve also rises faster (panel a).
- **Genome-wide labels validate at 0.84 and test at 0.70.** The 1,093-gene validation set drawn from the genome-wide labels reads 0.841 / 0.836 / 0.835 at the selected epochs while the released held-out genes read 0.706 / 0.726 / 0.679, the widest validation-to-test gap of any arm; the released held-out genes are a different population from the genome-wide negatives, which is the same point as the bullet below.
- **The learning rate at 3e-4 and 3e-3 is worse than 1e-3 on the seed mean**, but with seed spreads of 0.09 and 0.04 the pooled numbers (0.732, 0.713) sit inside the fcl interval; neither is a lever to keep.
- **Genome-wide labels hurt** (0.704 against 0.725, three of three seeds down). Adding SGD's essential list and the viable single-mutant fitness records as nonessential gives 4,368 training genes but a worse ranking of the released held-out genes. Hypothesis (untested): the viable-fitness negatives are a different population from the released split's nonessential genes.
- **Gradient norms (panel d):** on the fcl cells the preprocessor's gradient starts near 1e-2 and rises about thirty-fold over 80 epochs while the head's stays flat near 0.4; the clip at 10 never binds.

Round 3, jobs 2209 to 2222 (14 cells), all width 12 on the fcl split with graph regularization on: batch 32 with the class weight at 80 and 200 epochs, batch 32 alone at 200 epochs, and the two winners on five-fold cross-validation files (`ess_fcl_fold{0-4}of5_s0`, 629 or 630 train genes and 157 or 158 validation genes each) at 200 epochs, whose fold models are averaged on the held-out genes exactly as FCL averages its five fold models. The ensemble read is `scripts/cv_ensemble_readout.py`.
