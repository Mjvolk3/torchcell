---
id: je8a0nejapqvembbg4n84dt
title: Test_losses_dango
desc: ''
updated: 1791269896800
created: 1791269896800
---

## 2026.10.06 - Phase 21: flipped schedule, refusals and edge behaviors

Named `test_losses_dango.py` because `tests/torchcell/models/test_dango.py` holds the basename; it needs the pair entry `"torchcell/losses/dango.py" = "tests/torchcell/losses/test_losses_dango.py"`.

Fixture: one edge type "string" (lambda 0.1), reconstruction [[1, 0], [0, 0.5]] against adjacency [[1, 0], [1, 0]]: weighted MSE (1 + 0.1 * 0.25) / 4 = 0.25625; interaction log-cosh of [0.5, -1] against 0.

- `LinearUntilFlipped(20)`: alpha 1 - e / 20 (5 -> 0.75, 19 -> 0.05), 0 from epoch 20; DangoLoss with it at epoch 5 of T = 10 gives 0.5 *0.25625 + 0.5* I.
- Two networks average their losses; a missing lambda defaults to 1.0.
- Refusals: the abstract scheduler, reduction 'avg', a non-scheduler (a falsy one silently becomes PreThenPost(10)).

Findings:

- `reduction` is validated and never read (losses/dango.py:201-207): 'none' and 'sum' give the scalar 'mean' result.
- No matching edge type returns the float 0.0 and `forward` raises `AttributeError: 'float' object has no attribute 'device'` (line 341).
- Latent: log(cosh(r)) overflows for |r| above about 89 in float32 (inf loss, NaN gradient at r = 100); gene-interaction residuals are far smaller, so no reported run is affected.

## 2026.10.06 - Phase 21 audit 2

The base-forward `pass` test was deleted (it covered a line, not a contract; the abstract refusal is pinned in `test_abstract_scheduler_and_refusals`). Reach of the reduction finding: latent, the 005/006 dango.py scripts hard-code reduction "mean". The default-scheduler check is now typed (isinstance narrowing) instead of a type ignore.
