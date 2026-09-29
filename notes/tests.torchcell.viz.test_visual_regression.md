---
id: 4zcq5v9q92zd88subh4y8ii
title: Test_visual_regression
desc: ''
updated: 1790648896942
created: 1790648896942
---

## 2026.09.28 - The visual regression plotter on tiny prediction arrays (Phase 9)

17 tests. Correlation scatter with title, identity line and exact Pearson digits (three points: `3/sqrt(28/3) = 0.98198`, shown as 0.982), the bfloat16 path, subsampling under `np.random.seed(0)` pinned to the exact chosen points, the fewer-than-two-points skip and the failed-correlation report; distribution metrics with the Jensen-Shannon value 0.1323 (`KL(p||m) = 0.08229`, `KL(q||m) = ln 1.2 = 0.18232`) and the histogram bar positions; UMAP through a fake reducer (kwargs, fitted rows, embedding, NaN rows dropped, colors by label); `visualize_model_outputs` plotting only informative targets. Finding: `log_sample_metrics` with `stage=""` logs keys with a leading slash (`/MSE_target_0`, lines 258 to 290) while the plot methods drop the separator. Coverage 11.4% to 100%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]].
