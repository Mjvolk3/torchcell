---
id: 0ffzc4o0pvn2k9bm6afe81d
title: Distributional
desc: ''
updated: 1785532045851
created: 1785532045851
---

## 2026.07.31 - Make "score only the genes still hidden" one reduction that every distributional mode obeys

The round's teacher-forced masked-label objective (v9) hands the model `m` true gene values and asks it to predict the rest. That measures nothing unless the loss ignores the revealed genes: a revealed gene is model INPUT at that step, so scoring it rewards copying input to output -- train loss collapses, train Pearson inflates, and none of it transfers to validation, where nothing is revealed. `masked_mean` is that restriction, and it lives here rather than in the training script so the point path and the five probabilistic modes reduce through ONE definition instead of five copies that can drift apart silently (commit `73224a55`, +49/-5).

- **Why this module is distributional at all.** Under MSE the optimal point prediction is the conditional mean, so `point` actively *rewards* mean-collapse -- exactly the failure the expression task keeps hitting. The five probabilistic modes replace it with proper scoring rules in y-units: `gaussian` / `laplace` closed-form CRPS (the Laplace twin isolates whether a quantile head's edge is the distributional loss or merely a median point estimate), `quantile` pinball over K=19 taus, `energy` the multivariate generalization with a global low-rank `V [F, k]`, and `nll_gaussian` deliberately UNPATCHED as the negative control carrying the `1/sigma^2` sigma-collapse pathology. `pit_values` / `coverage` / `pit_ks` are the one diagnostic that puts all of them on the same `[0, 1]` scale, since the losses themselves are not comparable to each other. Default `rank=32` is not a guess: `residual_covariance_diagnostic.json` puts the participation-ratio effective rank of the reproducible residual structure at **32.78** (cumulative variance 37.6% at k=8, 59.1% at k=32).
- **What the reduction has to satisfy.** `feature_mask` is `[B, F]` (True = score this entry) while `elem` may be `[B, F]` (point) or `[B, F, K]` (quantile), so the mask broadcasts along trailing axes -- a quantile knot is scored exactly when its gene is. An empty selection returns a graph-connected zero rather than a NaN, so DDP find-unused stays happy.

```python
def loss(self, params, target, mask=None, feature_mask=None) -> torch.Tensor:
    if mask is not None:                      # row mask [B] as before
        params, target = params[mask], target[mask]
        if feature_mask is not None:
            feature_mask = feature_mask[mask]  # kept aligned with the rows
    if self.mode == "point":
        return masked_mean((params - target) ** 2, feature_mask)
    ...
```

- **`feature_mask=None` is bit-identical to the previous reduction**, which is the whole reason wave-6 arms already in flight were not silently re-baselined. `verify_masked_objective.json` C2 (quantile head, B=6, F=40, K=19): implicit `0.54242224`, explicit `None` `0.54242224`, all-True mask `0.54242224`. Note `masked_mean` on an all-True mask is a *weighted* sum/denominator while `None` short-circuits to `elem.mean()`; the contract asserts they agree, it is not assumed.
- **A revealed gene contributes exactly zero gradient** -- the copying failure in its most direct form. C3: `|grad|` hidden `0.4941`, revealed `0.000e+00`.
- **The point path reuses the same function, it does not reimplement it.** `MultitaskLoss._elementwise_masked` in `torchcell/models/equivariant_cell_graph_transformer.py` imports `masked_mean` *inside* the function -- a deferred import on purpose, so that models module keeps carrying no runtime dependency on `torchcell.losses` (whose `__init__` pulls in unrelated losses) while still sharing one definition.
- **Known gap, from reading the code, not measured:** `energy` mode ignores `feature_mask`. `energy_score` collapses all F features into one per-row scalar through the Euclidean norm, so there is no per-feature term left to select; a v9 arm run with `dist=energy` would score revealed genes. Likewise `DistHead.pit` takes only the row mask, so calibration diagnostics still pool every feature. No v9 arm has been run with `dist=energy`, so this has cost nothing yet.
- **Why the exactness mattered before spending GPU.** The CPU oracle (`masked_conditioning_oracle.json`, 1,482 strains x 6,169 genes, seed 0, 5 draws) bounds what ridge-from-revealed-genes alone can reach: val Pearson **0.408** (m=10), **0.676** (m=100), **0.793** (m=1000) -- at m=1000 the gene-gene signal by itself exceeds the 0.7746 replicate-based genotype ceiling (`expression_ceiling_replicate.json`). With numbers that large available for free from the labels, a leaky reduction would have produced a headline result that meant nothing; only the `k=0` column, where nothing is revealed, is comparable to prior arms.
