---
id: dq2s86b9s6w6zz6dysosbnf
title: Wave12 Essentiality Grid
desc: ''
updated: 1789538636950
created: 1789538636950
---

## 2026.09.16 - Design: two isolations, one per objective

Two objectives, stated by the user on 2026-09-16: the best essentiality AUROC the backbone can reach, and whether the metabolic module can help that prediction or at least not hurt it while receiving gradient. The grid is two isolations rather than a full cross, because the full cross of width, depth, learning rate, regularizer, module and readout is several hundred cells and the reads that decide the next step need six to eighteen.

### What the record says before any new cell runs

- **Stability tracks depth and preprocessor depth, not width, and the unstable cells are the ones that collapse.** Measured on the result histories (`wave9_ess_*`, `wave11_ess_pool_*`): the median absolute epoch-to-epoch step of validation AUROC is 0.005 to 0.008 on every 1-layer, preprocessor-1 cell (widths 12 and 24) and 0.032 to 0.072 on every 2-layer, preprocessor-2 cell (widths 54 and 90). The one pooled cell that collapsed to a constant (wave 9, width 90, seed 2702, spread ratio 1.8e-5) is a 2-layer cell; no 1-layer cell collapsed. Wave 11 moved the three factors together, so which of them buys the stability is not yet measured.
- **Selection noise is now the largest error on the small cells.** The width-24 cells at seeds 2700 and 2701 selected epochs 5 and 4 on validation peaks of 0.822 and 0.864 and scored 0.743 and 0.740 on test; seed 2702 selected epoch 47 and scored 0.807. Validation is 75 genes with about 12 essential, so a two-gene reordering moves it by 0.05.
- **Every flux cell on essentiality ran the penalty configuration that collapses the layer's input.** The wave-9 flux cells carried constraint scale 0.1 with 20 warmup epochs, the scale at which the betaxanthin dose-response measured a spread ratio of 1e-6, and two of three collapsed (spread 0.0005 and 0.0001). The penalty-free configuration was never run on this endpoint. The module has therefore been tested once, on the unstable backbone, with the penalties that starve it.
- **The comparator is FCL's held-out 0.742 on our 195 genes**, not the figure's 0.78, which is cross-validation on training-side genes ([[experiments.027-betaxanthin-metabolic.scripts.fcl_comparison]]).

### Phase A, the backbone (objective 1)

All pooled, 80 epochs, three seeds (2700 to 2702), graph regularization on and learning rate 1e-3 unless the row says otherwise. The reference cell is width 24, 8 heads, 1 layer, preprocessor 1, which is stable and has one seed at 0.807.

| id | width | heads | layers | preproc | change | what it isolates |
|---|---:|---:|---:|---:|---|---|
| A1 | 90 | 9 | 1 | 1 | | width at the shallow depth: if this is as stable as width 24, depth and preprocessor were the lever, not width |
| A2 | 12 | 4 | 2 | 2 | | the reverse: a narrow deep cell; if it is unstable, depth is the lever |
| A3 | 24 | 8 | 1 | 1 | graph reg off | the regularizer's contribution on this endpoint, never removed before |
| A4 | 24 | 8 | 1 | 1 | lr 3e-4 | learning rate, never moved on this endpoint |
| A5 | 24 | 8 | 1 | 1 | lr 3e-3 | |
| A6 | 24 | 8 | 1 | 1 | class weight 772/126 | the 16 percent positive rate, never rebalanced |

Reads, per row against the reference (the wave-11 width-24 cells, and their `_nohead` reruns 2066 to 2068): median absolute validation step, selected epoch, the spread ratio at selection, and test AUROC on the 195 genes with its gene bootstrap, paired by seed. A change earns a follow-up only if three of three seeds move the same way.

### Phase B, the module (objective 2)

All on the reference backbone (width 24, 8 heads, 1 layer, preprocessor 1), graph regularization on, 80 epochs, three seeds. Flux arms carry the wave-9 flux configuration (predicted kcat, union-external thermodynamics, 20 warmup epochs, dataset medium, mixture concentration space) and vary only what the row names. "Penalty-free" is constraint scale 0 with the density, active-balance and parsimony weights at their default 0.

| id | arm | readout | constraint scale | kcat | what it isolates |
|---|---|---|---:|---|---|
| B1 | flux_projection_anchored | biomass (forced) | 0 | as shipped | the module carrying the whole prediction, with nothing starving its input |
| B2 | flux_projection_anchored | biomass (forced) | 0.1 | as shipped | wave 9's configuration on the stable backbone: does the collapse follow the penalty or the backbone |
| B3 | flux_off | biomass (forced) | 0 | as shipped | stoichiometry alone, no thermodynamics or capacity |
| B4 | flux_projection_anchored | biomass (forced) | 0 | shuffled by seed | the control for B1: same architecture, no unit-to-value correspondence |
| B5 | flux_projection_anchored | gated (optional) | 0 | as shipped | can the module not hurt: at initialization this is the pooled model exactly |
| B6 | flux_projection_anchored | gated (optional) | 0.1 | as shipped | the same with the penalties on |

The gated readout is new (`FluxGatedEssentialityReadout` in `train_bx.py`): the pooled scalar head's logit plus a zero-initialized gate times the biomass affine. The gate is logged per epoch as `ess_gate`, and the per-module gradient report carries a `flux_layer` line, so "is it being used" and "are gradients passing through" are read off the history rather than argued.

Reads:

- **Not hurt** is B5 against the reference, paired by seed: a difference inside the seed spread with the gate having moved off zero is the module riding along at no cost; a difference below the reference with the gate open is the optimizer using a path that hurts; a gate that never leaves zero with the `flux_layer` gradient at zero is the module ignored, which is also an answer.
- **Help** is B1 above the reference on three of three seeds, and B1 above B4, since a shuffled kcat that scores the same says the physics content is not what is being used.
- **The collapse's cause** is B2 against B1 and against the wave-9 flux cells: collapse at 0.1 on the stable backbone puts it on the penalty; no collapse puts it on the backbone.

### Cost and order

Thirty-six cells at 65 to 80 minutes each on one GPU, four GPUs, so about eleven hours of wall time after the seven wave-11 reruns finish. Phase A is submitted first because its reference cells are already in the queue, and a two-epoch smoke of the gated flux arm gates phase B, since the readout is new code. Launcher: `gh_wave12_ess_grid.slurm`, parameterized by environment and frozen per cell by `freeze_and_submit.sh`.

## 2026.09.16 - Superseded by 028

Every cell above was cancelled before it started (jobs 2094 to 2148) and requeued under `experiments/028-gene-essentiality/` with the split-file protocol, see [[experiments.028-gene-essentiality]]. The gated readout, the class weight, the graph-regularization switch and the learning-rate cells live in `train_essentiality.py` from here on; `gh_wave12_ess_grid.slurm` is not used again.
