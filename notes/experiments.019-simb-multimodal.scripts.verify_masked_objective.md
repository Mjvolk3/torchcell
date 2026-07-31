---
id: ry36x0ycza6qrb6e86lcm76
title: Verify_masked_objective
desc: ''
updated: 1785532031333
created: 1785532031333
---

## 2026.07.31 - Refuse to launch the masked objective until a revealed gene provably cannot move a weight

The teacher-forced masked-label objective (v9) reveals `m` true gene values and scores only the still-hidden genes, and every way it can be wrong is silent -- it keeps producing a plausible number that means nothing. The failure that matters most is copying: if a revealed entry can push gradient, the model is being trained to reproduce its own input, train loss collapses and train metrics inflate for free, and none of it transfers to validation, where nothing is revealed. This script turns that worry into an executable gate -- it imports the shipped `DistHead.loss` and the real `MultitaskCGTTask._observed_feature_mask` / `_to_token_space` (not a re-implementation), writes `results/verify_masked_objective.json`, and exits non-zero with "do not launch" if any contract fails.

```bash
python experiments/019-simb-multimodal/scripts/verify_masked_objective.py
# -> results/verify_masked_objective.json ; "all contracts hold -- safe to launch"
```

All seven contracts pass in `experiments/019-simb-multimodal/results/verify_masked_objective.json` (fixtures: B=6 rows, F=40 features, K=19 quantile knots, `torch.manual_seed(0)`).

- **C3 -- a revealed gene contributes exactly zero gradient.** With the first half of features marked hidden, backprop through `head.loss(p, target, feature_mask=hidden)` gives `|grad| hidden=0.4941 revealed=0.000e+00`. Exactly 0.0, not merely small; this is the copying failure tested in its most direct form.
- **C2 -- `feature_mask=None` is bit-identical to the previous reduction**, so every arm already running is unaffected and the new argument cannot silently re-baseline the project: `none=0.54242224 explicit_none=0.54242224 all_true=0.54242224` (implicit `None`, explicit `None`, and an all-True mask agree to 8 decimals). C1 separately pins that a mask selecting `elem[0, :3]` reduces to exactly those entries (`got 1.000000 vs expected 1.000000`).
- **C6 -- the token-space scatter is the inverse of the head's gather.** `_to_token_space` writes `[B, F]` observations into `[B, N]` node space through `col_idx`; gathering back with `index_select(1, col)` reproduces `target * obs` exactly, and `msk.sum() == obs.sum()` shows no observation mass lands off the mapped nodes. An off-by-one here would feed the model the right values attached to the wrong genes and still train happily.
- **C4/C5 -- the validation sweep is an unmasking trajectory, not K unrelated draws.** One shared per-row random key with `scores.argsort(dim=1)[:, :n_reveal]` makes the observed sets nested (`M_k subset M_{k+1}` for every consecutive pair), with exact per-row reveal counts `[0, 5, 10, 20, 40]` for reveal requests `(0, 5, 10, 20, -1)` -- `-1` means reveal all.
- **C7 -- unsupervised rows reveal nothing.** Gating the observed mask by the per-row supervision mask leaves `revealed entries on unsupervised rows = 0`; teacher-forcing zeros on a row with no ground truth would teach "unmeasured == 0".
- **Why the gate is worth its cost (trainer-side context, not asserted here):** at `k=0` the observed set is empty and the forward pass is identical to the unconditioned model, so `val/<phenotype>/pearson_per_feature@k0` stays directly comparable to every previous round while the `k>0` numbers report an imputation capability. That comparability is only meaningful if C3 holds. Relatedly, the trainer refuses a schedule containing `-1` (reveal-all leaves an empty hidden set, hence nothing to score) and points at a largest-partial schedule such as `[0, 10, 100, 1000]`.
