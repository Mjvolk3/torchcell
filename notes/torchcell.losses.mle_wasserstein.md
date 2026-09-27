---
id: kfdd4e9dpxpavz61xl56bib
title: Mle_wasserstein
desc: ''
updated: 1790546348134
created: 1790546348134
---

## 2026.09.27 - The scheduled temperature now reaches the unbuffered SupCR

`MleWassSupCR.forward` computed the scheduled temperature every call and logged it under `loss_dict["temperature"]`, but passed it only to the buffered `BufferedWeightedSupCRCell`; with `use_buffer=False` the plain `WeightedSupCRCell` ran at the constructor's `supcr_temperature` (pinned by [[tests.torchcell.losses.test_mle_wasserstein_buffers]] in Phase 6 of [[plan.test-suite-buildout.2026.09.25]]). The unbuffered branch now writes the scheduled value into `self.supcr_loss.supcr.temperature` before the call, the same mechanism the buffered cell uses internally; the identical branch in `mle_dist_supcr.py` received the same fix. Still pinned and not changed: `weights=None` sizes the Wasserstein buffer at one column, so `MleWassSupCR(use_buffer=True)` with default weights raises on a two-dimensional target; below `min_samples` the buffered losses return `zeros(2)` whatever the dimension count; `BufferedWeightedSupCRCell` scales by `1 - w + 0.5 w`.
