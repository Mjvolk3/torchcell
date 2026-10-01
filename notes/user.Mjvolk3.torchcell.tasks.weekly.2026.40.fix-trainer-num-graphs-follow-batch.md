---
id: zhac2ko9n4086wjgm9lzqi7
title: fix-trainer-num-graphs-follow-batch
desc: ''
updated: 1790872939738
created: 1790872939738
---

## 2026.10.01

- [x] PR (branch `fix/trainer-num-graphs-follow-batch`): FIX issues #567 and #572, Refs #566. CGT, hetero and 019 trainers size a batch by `num_graphs`; the lazy collate writes `ptr`, `num_graphs` and `follow_batch` vectors and the 006 lazy scripts hand their list to the `LazyCollater`; 019 k = 0 comments and the `ObservedLabelEncoder` docstring state the `proj([0, 0])` offset. Notes: [[torchcell.trainers.int_transformer_cell]], [[torchcell.trainers.int_hetero_cell]], [[torchcell.datamodules.lazy_collate]], [[torchcell.models.equivariant_cell_graph_transformer]], [[experiments.019-simb-multimodal.scripts.train_cgt_multitask]], [[experiments.006-kuzmin-tmi.scripts.hetero_cell_bipartite_dango_gi_lazy]]; tests in [[tests.torchcell.trainers.test_int_transformer_cell]], [[tests.torchcell.trainers.test_int_transformer_cell_methods]], [[tests.torchcell.trainers.test_int_hetero_cell]], [[tests.torchcell.trainers.test_019_train_cgt_multitask]], [[tests.torchcell.datamodules.test_lazy_collate]].
