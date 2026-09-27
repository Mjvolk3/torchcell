---
id: yxg76ikt43od3pg86jgfj3h
title: Test_mle_wasserstein_buffers
desc: ''
updated: 1790543610430
created: 1790543610430
---

## 2026.09.27 - Ring buffers and the composite MleWassSupCR on CPU

`tests/torchcell/losses/test_mle_wasserstein.py` covers the schedulers and `WeightedWassersteinLoss`; this file covers the buffered losses and the composite. A ring buffer of size 4 written with rows (1, 2, 3) then (4, 5) holds (5, 2, 3, 4) with pointer 1, total 4 and the full flag set. Wasserstein values use the translation identity (debiased Sinkhorn at p = 2 on a cloud shifted by 1 is 0.5 per dimension whatever the cloud), so mixing buffer rows into the batch cannot change them; SupCR values come from the `_naive_supcr` reference on the exact row set the buffered loss concatenates. The composite with `use_buffer=False` on predictions = targets + 1 (16 rows, 2 dimensions) gives mse 1.0 and Wasserstein 0.5 per dimension and SupCR at the constructor temperature, combined as `1.0 * 1.0 + 0.1 * 0.5 + 0.001 * S`. Findings: `weights=None` sizes the Wasserstein buffer at one column, so `MleWassSupCR(use_buffer=True)` with default weights raises `RuntimeError` on the first two-dimensional forward (mle_wasserstein.py line 191); below `min_samples` both buffered losses return `zeros(2)` regardless of the dimension count (lines 266 and 410); `BufferedWeightedSupCRCell` scales by 1 - w + 0.5 w, which halves the loss at the default `buffer_weight` of 1.0 (line 429); with `use_buffer=False` the scheduled temperature is logged in `loss_dict` but never applied, SupCR runs at `supcr_temperature` (lines 596 to 601 and 675). Phase 6 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.27 - The temperature finding fixed; the tie rule changed two references

After the fix PR ([[torchcell.losses.mle_wasserstein]]) the composite test with `use_buffer=False` asserts SupCR at the scheduled temperature 1.0 and that it differs from the constructor value. The two buffered SupCR tests concatenate the batch with its own buffered copy, so every label is tied; under the corrected tie rule (the anchor's own term excluded) their `_naive_supcr` references are 1.1907942 at temperature 1.0 and 1.2760278 at 0.5 (before: 1.2417051 and 1.3407291). The other three findings (one-column default buffer, `zeros(2)` below `min_samples`, the `1 - w + 0.5 w` scale) stay pinned.
