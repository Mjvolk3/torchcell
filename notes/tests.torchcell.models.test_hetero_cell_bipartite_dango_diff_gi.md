---
id: p3blf2y8mcxn6l0eiei3m5f
title: Test_hetero_cell_bipartite_dango_diff_gi
desc: ''
updated: 1790979583421
created: 1790979583421
---

## 2026.10.01 - Diffusion gene-interaction model on the four-gene fixture

Fixture: the four-gene, two-graph wildtype and perturbed batches imported from `tests/torchcell/models/test_hetero_cell_bipartite_dango_gi.py` (`_cell_graph`, `_batch`, `_sample`, `_multigraph`), with a tiny `GeneInteractionDiff`: hidden 8, 1 GIN layer, "sum" aggregation, LayerNorm, local predictor with 2 heads and 1 attention layer, gating, dropout 0, and a diffusion decoder of 1 block, 2 heads, linear schedule over T = 10, 4 sampling steps.

Expected values and derivations:

- `num_parameters`: gene_embedding 32, preprocessor 160, convs 322, gene_interaction_predictor 370, global_aggregator 113, decoder 1417, gate_mlp 42, total 2456 (the eager head, 81, is deleted). "concat" gives 2414; the linear decoder gives 17 and total 1056; an empty diffusion config gives the 4-block, 8-head decoder of 4921 with the model norm.
- Conditioning: z_c = [z_i_global, mean of wildtype embeddings of the sample's perturbed genes], [B, 16]; "z_i" aliases z_i_global and "combined_embeddings" and "z_p" alias z_c; z_w is reported but not used in z_c.
- Eval mode: the prediction is bit-identical to `decoder.sample(z_c)` under the same seed, and changing the genotypes changes it. Reordering the genes inside a genotype leaves z_c and the seeded sample bit-identical (two genes) or within 1e-6 (three genes).
- `compute_diffusion_loss` forwards `t_mode` ("zero" equals the clean reconstruction loss exactly); the linear decoder's loss is the MSE of its linear prediction and ignores `t_mode`.
- Linear decoder: weight [1, 2, 3] and bias 0.5 map [1, 0, -1] to -1.5 and [2, 2, 2] to 12.5; the linear model predicts in both train and eval mode.

Findings (all reproduced before pinning; lines in torchcell/models/hetero_cell_bipartite_dango_diff_gi.py unless stated):

- Training-mode zero placeholder (lines 242-258): zeros shaped and typed like `phenotype_values` ([B] -> [B, 1]; 0-dim -> [1, 1]; 3 targets for 2 genotypes -> [3, 1]; none -> [B, 1]), no grad_fn, independent of the decoder weights. In `DiffusionRegressionTask._shared_step` (torchcell/trainers/int_hetero_cell.py:1120) these zeros reach `train_transformed_metrics` (1235), `train_metrics` after the inverse transform (1276) and the train prediction samples (1293, 1306). Issue #614 item 5. The `training_mode` attribute is set to True and never read.
- Production batches pool every genotype (lines 223-229): the 006 diffusion script builds `CellDataModule` without `follow_batch` (experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_diff_gi.py:295-305), whose default is ["x", "x_pert"] (torchcell/datamodules/cell.py:385-386), so neither `perturbation_indices_ptr` nor `_batch` exists and every genotype's local conditioning is the mean over all perturbed genes of the batch (identical rows for genotypes [0, 1], [2, 3], [1]).
- An empty genotype (lines 220-222) is conditioned on the mean of the other genotypes' perturbed genes.
- Dead modules (lines 159-278): `gene_interaction_predictor` (13 tensors, 370 parameters) and `gate_mlp` (4 tensors, 42) are built and counted but never called, so they get no gradient; at shipped size the dead local predictor is 37507 parameters. With the decoder findings, the exact no-gradient set is those plus norm3, and the exactly-zero set is norm1, q_proj and k_proj.
- `sample()` (lines 304-333): in eval it runs the decoder loop twice (8 denoise calls for 4 steps; the returned sample is the second draw after seeding); in train mode it samples in train mode, so with `norm="batch"` the decoder BatchNorm running statistics change (norm1 `num_batches_tracked` 0 -> 4).
- The shipped config's `sampling_method` and `conditioning_type` (hetero_cell_bipartite_dango_diff_gi.yaml) are read by nothing: identical state_dict with, without, or with other values.

Left uncovered: `main` (lines 366-1376, out of scope by the phase brief) and two branch arcs (one genotype with no perturbation assignment; 2-D targets).

Mutation check (in-memory mutants, scratch only): 10 model mutants, all killed (z_c from the wildtype global, sum instead of mean, empty genotype zeros, train/eval branch flipped, hidden default doubled, sample drops num_samples, loss drops t_mode, gate left out of the count, config dropout ignored, single-batch path uses the first gene).

## 2026.10.01 - Audit follow-up: reach and two more pinned behaviors

Reach of the findings, as the independent audit gives it:

- Batch-wide pooling (G1) reaches the 006 script `experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_diff_gi.py`: both `CellDataModule` and `PerturbationSubsetDataModule` default `follow_batch` to ["x", "x_pert"], and the script passes neither. It reached the runs launched by `experiments/006-kuzmin-tmi/scripts/delta_hetero_cell_bipartite_dango_diff_gi-ddp_000.slurm`.
- The training-mode zero placeholder (G2) reaches `DiffusionRegressionTask._shared_step` (class at torchcell/trainers/int_hetero_cell.py:902) at lines 1120, 1235, 1276, 1293 and 1306.
- `sample()` running twice and moving BatchNorm statistics (G5) is latent: the shipped norm is "layer". The test now also pins the encoder side: the preprocessor's shared BatchNorm goes 0 -> 4 (two layers times two `forward_single` calls) and each conv-wrapper norm 0 -> 2.

Newly pinned:

- Eval predictions are one draw from the global RNG (Finding): the first denoise input is exactly the first `torch.randn(2, 1)` after seeding; one seed repeats bit for bit, seeds 0 and 1 differ. `val/inference_mse`, `test/inference_mse` and the val and test metrics (int_hetero_cell.py:1120, 1181-1193, 1235, 1276) are therefore single samples. On the untrained fixture the spread is about 1e-6 and 1e-4 per genotype; only the dependence is asserted.
- Every test runs inside `torch.random.fork_rng(devices=[])` (autouse fixture), so seeding does not leak.
