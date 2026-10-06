---
id: oc7hw75phe0ejrx7onea5kx
title: Test_diffusion_decoder
desc: ''
updated: 1790979575088
created: 1790979575088
---

## 2026.10.01 - Schedules, q-sample, denoiser oracle and DDIM on a one-dimensional toy

Fixture: no data, only hand-written tensors. Schedules are recomputed in float64 numpy from the formulas the docstrings cite; the denoiser is compared with a float64 numpy re-implementation on a decoder whose parameters and BatchNorm running statistics are all set from `np.random.default_rng(3)` (norm "layer", "batch" and "none"); the DDIM loop is replayed in numpy with `denoise` replaced by `pred_x0 = 0.5 x + 0.01 t`.

Expected values and derivations:

- Linear schedule, T = 10: beta = linspace(1e-4, 0.02, 10), abar_0 = 0.9999, abar_9 = 0.903739; every float32 buffer within 1e-6.
- Time embedding, dim 6: frequencies [1, 1e-2, 1e-4]; rows are [sin, sin, sin, cos, cos, cos]; t = 0 gives [0, 0, 0, 1, 1, 1].
- q-sample, x_0 = [1, 2, -1], t = [0, 3, 9], eps = [0.5, -1, 2]: rows 1 and 2 follow sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps; row 0 returns x_0 and noise 0.
- DDIM step lists: T = 10 default gives 10 calls (9..0); T = 1000 default gives 50 calls on floor(linspace(0, 999, 50)) reversed (999, 978, 958, ..., 61, 40, 20, 0; truncation, not rounding); T = 3 with steps 5 gives 5 calls on [2, 1, 1, 0, 0]; T = 1 gives one call at t = 0.
- Zero start (torch.randn patched to zeros), steps 9, 6, 3, 0: every state fed to the stand-in and the final sample match the numpy recurrence to 1e-6.
- Loss: "full" and "random" replay t = randint(0, 10, (3,)) and then eps = randn(3, 1) under the same seed; "zero", "t0", "partial", "small" at T = 10 all equal the clean reconstruction loss exactly (T // 10 = 1); "partial" at T = 50 draws t from [0, 5).
- Parameter counts: tiny (input 16, hidden 8, 1 block, 2 heads) 136 + 16 + 1168 + 97 = 1417; shipped 006 decoder (input 128, hidden 64, 2 blocks, 4 heads) 8256 + 128 + 2 * 66688 + 4353 = 146113.

Findings (all reproduced before pinning; source lines in torchcell/models/diffusion_decoder.py):

- Single-key cross-attention (lines 92-97): softmax over one score is exactly 1, so the output is out_proj(v_proj(context)) whatever the query; norm1, q_proj and k_proj receive exactly zero gradient and norm3 (lines 123-133) is never called. In the shipped decoder that is 16896 parameters that never move and 256 that are never used. Mutating the attention scale is an equivalent mutant for the same reason.
- Cosine schedule in float32 (lines 283-289): the same expression in float64 matches the formula to 1e-12, but the float32 betas miss it by 3.1e-5 at t = 998 (more than 5e-6 asserted), and sqrt(1 - abar_0) is 0.0100008 instead of 0.01.
- Cosine lower clip (line 290): the cited schedule (Nichol and Dhariwal 2021) clips only from above at 0.999; this code raises the first 13 betas at T = 1000 from 4.13e-5.. to 1e-4 (abar off the formula by up to 3.84e-4) and clips the last beta from 1 to 0.9999 (abar_{T-1} = 2.43e-10).
- t = 0 means two things (lines 309-327 vs 434-447): q-sample returns x_0 at t = 0, but the DDIM step into t = 0 builds sqrt(abar_0) p + sqrt(1 - abar_0) eps_hat; with a constant stand-in p = 2 from a zero start the network receives 1.93862 at t = 0, an input training never produces.
- `predict_x0` is never read by `denoise` (lines 336-374); `parameterization="eps"` passes the constructor and then every `denoise` raises `NotImplementedError("Parameterization eps not fully implemented")`, so the eps branch of `loss` (lines 504-508) is unreachable.
- Time embedding (lines 33-37): an odd dim returns dim - 1 columns (a decoder with hidden 5 fails in the time MLP with "mat1 and mat2 shapes cannot be multiplied (2x4 and 5x10)"); dim 2 raises `ZeroDivisionError` at the first forward.
- Unknown `norm` in `DenoisingBlock` (lines 130-133) silently builds Identity norms ("instance": 365 parameters instead of 389).
- `sample` drops an unsatisfiable `num_samples` (lines 407-409): 2 contexts with num_samples 3 return 2 rows.

Left uncovered: lines 504-510, the eps branch of `loss`, unreachable because `denoise` raises first.

Mutation check (in-memory mutants, scratch only): 10 decoder mutants, 9 killed; the survivor (attention scale inverted) is equivalent because of the one-key softmax finding above.

## 2026.10.01 - Audit follow-up

- Float32 cosine schedule (D8): negligible in effect. The test now pins only the platform-independent statement: the float64 replica matches the formula to 1e-12, and the float32 buffer misses it by more than 1e-6 and less than 1e-4. The exact miss (3.1e-5 at t = 998 here) depends on the platform's float32 cos.
- t = 0 inconsistency: 1.93862 belongs to the constant stand-in from a zero start. The general closed form, now also asserted, is that the t = 0 input minus sqrt(abar_0) p equals sqrt(1 - abar_0) eps_hat, about 0.01 eps_hat.
- The linear-schedule test now checks the module's `alphas_cumprod[-1]` (0.9037394) instead of the oracle against itself.
- New Finding: a 1-D x_0 broadcasts in `forward_diffusion` (diffusion_decoder.py:313-321). x_0 = [1, 2, 3], t = [1, 2, 3] gives x_t of shape [3, 3] with x_t[i, j] = sqrt(abar_{t_i}) x_0[j] + sqrt(1 - abar_{t_i}) eps[j].
- Every test runs inside `torch.random.fork_rng(devices=[])` (autouse fixture); `_decoder` takes `**overrides: Any`, so the `type: ignore` is gone.
