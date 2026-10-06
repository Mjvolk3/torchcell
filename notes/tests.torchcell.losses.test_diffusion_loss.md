---
id: 3ygn4qrs2pg80jwqd5uzxtz
title: Test_diffusion_loss
desc: ''
updated: 1791269888987
created: 1791269888987
---

## 2026.10.06 - Phase 21: weighting, x0 term and timestep sampler

Fixture: a stand-in model whose `compute_diffusion_loss` returns 0.75 and whose decoder returns x_t = targets + noise and x0_hat = x_t, so x0_hat - targets is the sampled noise. Under a fixed seed the loss draws t, then the noise; the test regenerates both.

- total = lambda_diffusion *0.75 + lambda_x0* mean(noise^2) with the exact call arguments (t_mode, predict_x0=True, the same t and noise); lambda_x0 0 skips the decoder, lambda_diffusion 0 skips the model loss; a custom x0 loss (L1) replaces the MSE.
- `_sample_t`: 'full' is randint(0, T); 'partial' is randint(0, max(1, T // 10)) (T = 100, 25, 5 give [0, 10), [0, 2), only 0); 'zero' returns zeros and draws nothing.
- Refusals: both lambdas <= 0, a model without `compute_diffusion_loss` or `diffusion_decoder`, and a missing context, each with the exact message. The model is a registered submodule.

Finding: an unknown t_mode (e.g. 'parital') silently samples the full range (diffusion_loss.py:90), against the class docstring's "no silent fallbacks".

## 2026.10.06 - Phase 21 audit 2

The zero-mode test asserts dtype long (an int32 mutant now fails; `torch.equal` ignores dtype). Reach of the unknown-t_mode finding: latent, the only diffusion config sets t_mode "full".
