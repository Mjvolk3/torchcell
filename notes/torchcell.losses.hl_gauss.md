---
id: hwzvynxzzhgbtc0uuq35tvq
title: Hl_gauss
desc: ''
updated: 1792641736805
created: 1792641736805
---

## 2026.10.21 - HL-Gauss loss and per-dataset standardization

Source: `torchcell/losses/hl_gauss.py`; tests `tests/torchcell/losses/test_hl_gauss.py` (13). Design: decision O4 of the 043 primer (`notes-tex/multimodal/043-datasets-primer/sections/output-design.tex`, "Losses"); consumer [[torchcell.models.cell_model_043]] and [[experiments.043-expression-proteome-morphology.scripts.train_043]].

**`HLGaussLoss`.** A standardized target $z$ becomes a histogram over $K$ bins on $[\ell, h]$ (defaults $K = 32$, $[-4, 4]$, bin width $w = 0.25$) by integrating $\mathcal N(z, \sigma^2)$ with $\sigma = 0.75\,w = 0.1875$:

$$
p_k(z) = \Phi\!\left(\frac{e_{k+1} - z}{\sigma}\right) - \Phi\!\left(\frac{e_k - z}{\sigma}\right),
\qquad e_0 = -\infty,\; e_K = +\infty,
$$

so the two edge bins are open and $\sum_k p_k = 1$ exactly (no renormalization step; a target beyond $\pm 4$ puts its mass in the edge bin). The loss is the cross-entropy $-\sum_k p_k(z) \log \mathrm{softmax}(\ell)_k$, averaged over the measured entries. Readouts:

- point: $\hat z = \sum_k q_k c_k$ over bin centers $c_k$ (the open edge bins use their nominal midpoints $\pm 3.875$);
- uncertainty: the entropy of $q$;
- classification: accuracy within one bin (argmax bin against the target's bin).

Missing entries are `NaN`. The soft target is built from the zero-filled safe target and the elementwise loss is multiplied by the measured mask, so no `NaN` reaches the value or the gradient; a batch with nothing measured returns a connected zero. Shapes: any leading shape, `[B, G, K]` against `[B, G]`, `[B, K]` against `[B]`.

**`StandardizationStats`.** Per dataset, $z = (y - c_g - \mu)/s$: an optional per-gene center $c_g$ (the panel datasets, centered per gene over their training isolates or segregants), then a pooled $\mu$, $s$ over every training value. A gene with no training value has no center and no target (`None` in the record, `NaN` in $z$). Fitted by `StandardizationStats.fit` on training rows only, and pinned by the sha256 of its sorted-key JSON; the trainer writes it beside the checkpoints and logs the sha256.

Tested behavior (13 tests): soft target sums to one and matches the normal CDF; missing entries carry zero gradient; an all-missing batch is a connected zero; minimizing the loss recovers the target to 0.05; centered and pooled fits reproduce hand-computed means and spreads; the sha256 changes with the content and not with a copy.
