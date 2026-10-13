---
id: czislfek4q483wavubx9gf9
title: 025-graph-reg-sweep-readout
desc: ''
updated: 1791682498601
created: 1791682498601
---

## 2026.10.10

- [x] Every 025 graph-regularization run finished (65 of 80 round-2 runs; 15 failed at step one with CUDA OOM on the A40: KL 1 on layers 1-4 and all width-360 arms). `make refresh` after two readout fixes (NaN validation epochs counted, NaN seeds dropped from the paired test); the sweep figure now runs to KL 100: no turnover (0.444, 0.443 vs 0.447 at KL 1) [[experiments.025-solid-growth.scripts.graph_reg_round2_plan]]
- [x] Rounds 2 and 3 read into section 6 of the notes-tex document: reach and direction do not help either mechanism; KL 1 on layers 3-4 equals layer 1; at epoch 59 no penalty 0.39 to 0.41 vs KL 1 0.43 to 0.44; composite embedding does not train without the prior; same-seed replicates move by up to 0.011 at epoch 29
- [ ] Decide on the width-360 and layers-1-4 arms: smaller per-GPU batch with gradient accumulation, or drop
- [ ] Draw the mechanism and budget figures (round-2 wireframe rows 2 and 3) from the tables
- [ ] Pull best checkpoints for the discovery and overruling panels

## 2026.10.13

- [x] Figure 2 (manuscript and prelim) panel d, the additive null on the 010 triples [[experiments.010-kuzmin-tmi.scripts.fig2_additive_null_panel]], and panel e, do the graphs help [[experiments.025-solid-growth.scripts.fig2_graphs_help_panel]]; both third width, ready for the draw.io grid
- [ ] Refresh panels b (binned prediction) and c (state of the art) with the 025 S3 closure model: needs a TEST evaluation of checkpoints yb4gjh51, srl1xzdf, 8fuwsk48 (only validation is logged, 0.480 to 0.496)
- [ ] GAT vs CGT equivalence: unit test of a masked CGT layer against PyG TransformerConv with copied weights, then a three-seed training comparison under the 025 protocol
