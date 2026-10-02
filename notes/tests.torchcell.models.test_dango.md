---
id: cdm6h3yvrgpp5s775q6izmx
title: Test_dango
desc: ''
updated: 1790916706022
created: 1790916706022
---

## 2026.10.01 - Dango model contract: closed forms, batch separation, findings

Paired with [[torchcell.models.dango]] (`torchcell/models/dango.py`), the Dango baseline (Zhang et al. 2020 reimplementation) trained by `experiments/005-kuzmin2018-tmi/scripts/dango.py` and `experiments/006-kuzmin-tmi/scripts/dango.py`. `main` (dango.py:551-976) is a demo and is not tested.

### Fixture

- Four genes, two networks named as the 006 configs name them: `string12_0_neighborhood` 0->1, 1->0, 1->2, 2->1, 3->2 and `string12_0_fusion` 0->3, 3->0 (source -> target; SAGEConv aggregates at the target).
- Batches are built as the scripts build them: one `HeteroData` per genotype with global `perturbation_indices` (the `Perturbation` graph processor), collated by `Batch.from_data_list(..., follow_batch=["perturbation_indices"])`.

### Expected values

- `DangoPreTrain`, SAGE `lin_l = I`, `lin_r = 0`, embedding X = [[1, 2], [3, -1], [-2, 1], [.5, .5]]: each layer is relu(mean of in-neighbor rows). Neighborhood h1 = [[3, 0], [0, 1.5], [1.75, 0], [0, 0]]; with a layer-2 bias (-1, 0), h2 = [[0, 1.5], [1.375, 0], [0, 0.75], [0, 0]]. Fusion h2 = [[1, 2], 0, 0, [.5, .5]]. Reconstruction is a raw linear output with gene_num columns (gene 0: [.1, 1.7, 1.8, -1.1]); no sigmoid.
- `MetaEmbedding` with score relu(e[0]): softmax across networks per gene; gene (2, 2) vs (2, -2) with equal scores gives (2, 0). The weights the source computes are captured by a recorder around `F.softmax` (see the 2026.10.02 section).
- `_global_attention_layer` with Q = K = 0, V = O = I, beta = 1: out_i = x_i + mean over the OTHER members of the set. A numpy masked multi-head oracle matches random seeded weights; a singleton set returns beta * O.bias + x (the NaN-to-zero path).
- `HyperSAGNN` closed form: triple [(1, 0), (0, 2), (-1, 1)] scores 28.75 / 3, pair [(2, 0), (0, 0)] scores 10, in one call.
- `Dango`: parameter counts 648 / 41 / 659 / 1348 for G = 4, H = 8, E = 2 (formula in the test); betas 0.01; batched genotypes equal each genotype alone; all six gene orders give one score; interaction-only backprop leaves exactly the four `recon_layers` tensors without a gradient.

### Issue #596

The silent-zero mechanism reported in #596 belongs to `hetero_cell_bipartite_dango_gi`, not to this model. Here, with `follow_batch=["perturbation_indices"]` (both dango scripts set it), a batch of two triples and a singleton equals each genotype alone; without it, `Dango.forward` raises `AttributeError` rather than predicting zero. The #596 batch-size logging defect (`_get_batch_size` in `int_hetero_cell`) has no counterpart in this model.

### Findings

- `HyperSAGNN.forward` sizes its output by the number of distinct set ids (dango.py:307-308, 350-352): a genotype with no `perturbation_indices` in the middle of a batch raises `index 2 is out of bounds for dimension 0 with size 2`; at the end of a batch it yields 2 predictions for 3 targets. Not shown reachable in the 006 data.
- `DangoPreTrain.lambda_values` (dango.py:80-96) knows only string9_1 / string11_0 names; every string12_0 network gets 1.0. No training script reads it (the scripts pass `determine_lambda_values()` to `DangoLoss`).

## 2026.10.02 - Audit 1 corrections and added pins

- `MetaEmbedding` weights: the earlier version asserted sum-to-one on the test's own numpy softmax, which could not fail. `test_meta_embedding_source_weights_sum_to_one_per_gene_and_weight_the_output` now records the tensor `F.softmax` returns inside `MetaEmbedding.forward` ([5 genes, 3 networks]), checks each gene's row sums to 1, checks it equals a numpy softmax of the hooked MLP scores over networks, and rebuilds the output from those weights.
- `reset_parameters`: with gene_num 200 and H 64, the embedding and both reconstruction weights have sample std within 0.1 * (1 +- 0.25), and the reconstruction biases are zero (the default `nn.Linear` init would give std 0.072).
- `Dango.forward` stage wiring: `integrated_embeddings == meta_embedding(network_embeddings)` and `scores == hyper_sagnn(integrated[perturbation_indices], perturbation_indices_batch)`.
- `HyperSAGNN` with unsorted set ids [1, 1, 0, 0]: the output is ordered by set id (output[0] is the set with id 0), each entry equal to that set run alone. PyG numbers graphs in order, so ids are ascending in practice.
- Duplicate gene in one genotype: the self mask is positional, so a copy attends to its twin. Closed form with a = (1, 0), b = (0, 2): layer 1 gives (1.5, 1) for each copy of a in [a, a, b], versus (1, 2) for a in the deduplicated pair. Not reachable through the `Perturbation` processor, which builds one index per gene name.
- `num_parameters` counts trainable parameters only: freezing HyperSAGNN gives 648 / 41 / 0 / 689.
- Finding 1 (empty genotype) raises `RuntimeError` mid-batch (the test already said so). Seen from the trainer, the trailing case makes `DangoLoss` raise at the log-cosh broadcast (pinned in [[tests.torchcell.trainers.test_int_dango]]).
- Every `torch.manual_seed` in the file now runs inside `torch.random.fork_rng()` through an autouse fixture; British spellings in docstrings were Americanized.
