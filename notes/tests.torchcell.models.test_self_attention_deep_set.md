---
id: k6fzbd56i3la1zegvs7tdjf
title: Test_self_attention_deep_set
desc: ''
updated: 1790549288124
created: 1790549288124
---

## 2026.09.27 - SelfAttentionDeepSet and the batch-wide softmax

Nine tests: the output shape, seeded determinism, the per-layer parameter count, each constructor error, and the attention identities. Coverage of the module from this file 78%; the rest is the `main()` smoke script. Findings: the attention softmax runs over every node in the batch rather than within each graph, so per-graph attention rows do not sum to 1 (closed form on a 1 x 1 slice: `p = e^(1/sqrt 2) / (e^(1/sqrt 2) + 1)`) and graph 0's set output changes when graph 1's features change (self_attention_deep_set.py lines 39 to 41), which means the pooling is not a per-set function; the same single-layer branch problem as `DeepSet` (lines 94 to 112). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.30 - Phase 12: permutation identities and exact wiring

Nine to eighteen tests, 78 to 99 percent. Two heads give the exact concatenated rows; relabeling the graphs swaps the two `x_set` rows; sum pooling feeds each graph's node sum and a gap in the labels pools a zero row; dropout at p=1 gives exact zeros in training; a width-preserving first layer gets the skip connection; `main()` prints 10064 parameters. The node-permutation identities use `assert_close`, since the float summation order changes under a permutation.

Findings: `norm="instance"` builds but every forward raises (line 88); with one set layer `x_set` is `hidden_channels` wide with no dropout (116-118); `main()` turns on autograd anomaly detection and never turns it off (177).
