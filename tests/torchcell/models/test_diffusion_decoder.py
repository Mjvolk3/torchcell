# tests/torchcell/models/test_diffusion_decoder.py
# [[tests.torchcell.models.test_diffusion_decoder]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_diffusion_decoder.py
"""``DiffusionDecoder`` and its parts: schedules, q-sample, the denoiser, DDIM, the loss.

Fixtures. No data: every input is a hand-written tensor. Schedules are recomputed in
float64 numpy from the formulas the docstrings cite and compared with the float32
buffers. The denoiser is checked against a float64 numpy re-implementation (``_np_*``
below) on a decoder whose every parameter (and BatchNorm running statistic) is set from
a seeded numpy generator, so the oracle never calls the code under test. The DDIM loop
is checked on a one-dimensional toy: ``denoise`` is replaced by the affine stand-in
``pred_x0 = 0.5 x + 0.01 t`` and the recurrence is replayed in numpy.

Derivations used throughout.

* Linear schedule, T = 10: beta_t = linspace(1e-4, 0.02, 10), abar_t = prod_{s<=t}
  (1 - beta_s); abar_0 = 0.9999.
* Cosine schedule (Nichol and Dhariwal 2021): f(x) = cos(((x / T) + s) / (1 + s) pi / 2)^2
  on x = 0..T, abar = f / f(0), beta_t = 1 - abar_{t+1} / abar_t, then the code clips to
  [1e-4, 0.9999] and rebuilds abar = cumprod(1 - beta).
* q-sample: x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps, except t = 0 returns x_0 and a
  zero noise (the "pure reconstruction" override).
* DDIM (eta 0): eps_hat = (x - sqrt(abar_t) p) / sqrt(1 - abar_t), x_prev =
  sqrt(abar_prev) p + sqrt(1 - abar_prev) eps_hat, and the last step returns p.
* Step list: floor(linspace(0, T - 1, S)) reversed, S = min(sampling_steps, T) when not
  given (torch's integer linspace truncates; for T = 1000, S = 50 this gives 0, 20, 40,
  61, ..., 978, 999).
* Parameter count, input 16, hidden 8, 1 layer, 2 heads, mlp_ratio 4: context_proj 136 +
  input_proj 16 + block (3 LayerNorms 48 + q/k/v/out 4 * 72 = 288 + MLP 288 + 264 = 552 +
  time MLP 144 + 136 = 280 = 1168) + output head (LayerNorm 16 + 72 + 9 = 97) = 1417.
  Shipped 006 decoder (input 128, hidden 64, 2 layers, 4 heads): 8256 + 128 + 2 * 66688
  + 4353 = 146113.

Findings pinned here (each test docstring starts ``Finding:``): the single-key
cross-attention makes the query path dead (q_proj, k_proj and norm1 get exactly zero
gradient; norm3 is never used); the cosine schedule's float32 betas miss the float64
formula by more than 1e-6 and its lower clip rewrites the first 13 betas at T = 1000;
the t = 0 override of q-sample disagrees with what DDIM feeds the network at t = 0;
``predict_x0`` is ignored and ``parameterization="eps"`` builds but cannot run;
an odd or 2-wide hidden dimension breaks the time embedding; an unknown ``norm`` builds
an Identity; ``sample`` silently ignores a ``num_samples`` it cannot honor.
"""

import math
import re
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from numpy.typing import NDArray
from scipy.special import erf

from torchcell.models.diffusion_decoder import (
    CrossAttention,
    DenoisingBlock,
    DiffusionDecoder,
    SinusoidalTimeEmbedding,
)

Arr = NDArray[Any]

LINEAR_BETAS_T10 = np.linspace(1e-4, 0.02, 10)
LINEAR_ABAR_T10 = np.cumprod(1.0 - LINEAR_BETAS_T10)


@pytest.fixture(autouse=True)
def _restore_global_rng() -> Iterator[None]:
    """Run every test inside ``fork_rng`` so its seeding does not leak out."""
    with torch.random.fork_rng(devices=[]):
        yield


def _decoder(seed: int = 0, **overrides: Any) -> DiffusionDecoder:
    torch.manual_seed(seed)
    kwargs: dict[str, Any] = {
        "input_dim": 4,
        "hidden_dim": 4,
        "num_layers": 1,
        "num_heads": 2,
        "dropout": 0.0,
        "num_timesteps": 10,
        "beta_schedule": "linear",
    }
    kwargs.update(overrides)
    return DiffusionDecoder(**kwargs)


def _cosine_formula(T: int, s: float = 0.008) -> tuple[Arr, Arr]:
    """Return (unclipped betas, abar = f / f(0) at x = 1..T) in float64."""
    x = np.linspace(0.0, T, T + 1)
    f = np.cos(((x / T) + s) / (1 + s) * np.pi * 0.5) ** 2
    abar = f / f[0]
    return 1.0 - abar[1:] / abar[:-1], abar[1:]


# ------------------------------------------------------------------ numpy oracle


def _np_linear(x: Arr, w: dict[str, Arr], name: str) -> Arr:
    out: Arr = x @ w[f"{name}.weight"].T + w[f"{name}.bias"]
    return out


def _np_gelu(x: Arr) -> Arr:
    out: Arr = 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))
    return out


def _np_layer_norm(x: Arr, w: dict[str, Arr], name: str) -> Arr:
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    out: Arr = (x - mu) / np.sqrt(var + 1e-5) * w[f"{name}.weight"] + w[f"{name}.bias"]
    return out


def _np_norm(x: Arr, w: dict[str, Arr], name: str, norm: str) -> Arr:
    if norm == "layer":
        return _np_layer_norm(x, w, name)
    if norm == "batch":
        out: Arr = (x - w[f"{name}.running_mean"]) / np.sqrt(
            w[f"{name}.running_var"] + 1e-5
        ) * w[f"{name}.weight"] + w[f"{name}.bias"]
        return out
    return x


def _np_time_embedding(t: Arr, dim: int) -> Arr:
    half = dim // 2
    freqs = np.exp(-np.arange(half) * math.log(10000.0) / (half - 1))
    arg = t[:, None].astype(np.float64) * freqs[None, :]
    return np.concatenate([np.sin(arg), np.cos(arg)], axis=-1)


def _np_cross_attention(x: Arr, ctx: Arr, w: dict[str, Arr], p: str, heads: int) -> Arr:
    """Generic multi-head attention with one query row and one key row per sample."""
    q = _np_linear(x, w, f"{p}.q_proj")
    k = _np_linear(ctx, w, f"{p}.k_proj")
    v = _np_linear(ctx, w, f"{p}.v_proj")
    b, dim = x.shape
    hd = dim // heads
    out = np.zeros_like(x)
    for i in range(b):
        for h in range(heads):
            sl = slice(h * hd, (h + 1) * hd)
            score = np.array([q[i, sl] @ k[i, sl] / math.sqrt(hd)])
            attn = np.exp(score - score.max()) / np.exp(score - score.max()).sum()
            out[i, sl] = attn[0] * v[i, sl]
    return _np_linear(out, w, f"{p}.out_proj")


def _np_denoise(
    x_t: Arr, ctx: Arr, t: Arr, w: dict[str, Arr], norm: str, heads: int, dim: int
) -> Arr:
    c = _np_linear(ctx, w, "context_proj")
    h = _np_linear(x_t, w, "input_proj")
    te = _np_time_embedding(t, dim)
    tm = _np_linear(
        _np_gelu(_np_linear(te, w, "blocks.0.time_mlp.0")), w, "blocks.0.time_mlp.2"
    )
    h = h + tm
    h = h + _np_cross_attention(
        _np_norm(h, w, "blocks.0.norm1", norm), c, w, "blocks.0.cross_attn", heads
    )
    m = _np_gelu(
        _np_linear(_np_norm(h, w, "blocks.0.norm2", norm), w, "blocks.0.mlp.0")
    )
    h = h + _np_linear(m, w, "blocks.0.mlp.3")
    o = _np_layer_norm(h, w, "output_proj.0")
    o = _np_gelu(_np_linear(o, w, "output_proj.1"))
    return _np_linear(o, w, "output_proj.3")


def _set_from_numpy(module: nn.Module, seed: int) -> dict[str, Arr]:
    """Fill every parameter and BatchNorm running statistic from a numpy generator."""
    rng = np.random.default_rng(seed)
    values: dict[str, Arr] = {}
    with torch.no_grad():
        for name, p in module.named_parameters():
            arr = rng.normal(0.0, 0.5, size=tuple(p.shape))
            p.copy_(torch.from_numpy(arr))
            values[name] = arr
        for name, buf in module.named_buffers():
            if name.endswith("running_mean"):
                arr = rng.normal(0.0, 0.3, size=tuple(buf.shape))
            elif name.endswith("running_var"):
                arr = rng.uniform(0.5, 2.0, size=tuple(buf.shape))
            else:
                continue
            buf.copy_(torch.from_numpy(arr))
            values[name] = arr
    return values


# ------------------------------------------------------------------ time embedding


def test_time_embedding_is_sin_block_then_cos_block_on_geometric_frequencies() -> None:
    """Dim 6: half 3, log scale ln(1e4) / 2, frequencies [1, 1e-2, 1e-4]; the row for t
    is [sin t, sin 0.01 t, sin 1e-4 t, cos t, cos 0.01 t, cos 1e-4 t]. t = 0 gives
    [0, 0, 0, 1, 1, 1]. Long timesteps are accepted and promoted to float.
    """
    t = torch.tensor([0, 1, 999])
    out = SinusoidalTimeEmbedding(6)(t)
    tt = np.array([0.0, 1.0, 999.0])[:, None] * np.array([1.0, 1e-2, 1e-4])[None, :]
    expected = np.concatenate([np.sin(tt), np.cos(tt)], axis=-1)
    assert out.dtype == torch.float32
    torch.testing.assert_close(
        out, torch.tensor(expected, dtype=torch.float32), atol=2e-5, rtol=0.0
    )
    assert out[0].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_time_embedding_drops_a_column_for_odd_dims_and_divides_by_zero_at_two() -> (
    None
):
    """Finding: the docstring promises ``[batch_size, dim]``, but an odd ``dim`` returns
    2 * (dim // 2) columns (dim 5 -> 4), so a decoder with an odd ``hidden_dim`` builds
    and then fails in its first time-MLP matmul; ``dim`` 2 makes ``half_dim - 1`` zero
    and the first forward raises ``ZeroDivisionError`` (diffusion_decoder.py:33-37).
    Pinned until the constructor refuses an odd or 2-wide dim.
    """
    assert SinusoidalTimeEmbedding(5)(torch.tensor([1, 2])).shape == (2, 4)
    with pytest.raises(ZeroDivisionError, match=re.escape("float division by zero")):
        SinusoidalTimeEmbedding(2)(torch.tensor([1]))
    odd = _decoder(hidden_dim=5, num_heads=1)
    with pytest.raises(
        RuntimeError,
        match=re.escape("mat1 and mat2 shapes cannot be multiplied (2x4 and 5x10)"),
    ):
        odd.denoise(torch.zeros(2, 1), torch.zeros(2, 4), torch.tensor([1, 2]))


# ------------------------------------------------------------------ cross-attention


def test_single_key_cross_attention_is_out_of_v_of_context_whatever_the_query() -> None:
    """Finding: the class docstring says the diffusion state queries the graph
    embeddings, but there is one key per sample, so softmax over one score is exactly 1
    and the output is out_proj(v_proj(context)) for every query (diffusion_decoder.py:
    92-97). Hand-set weights, dim 4, 2 heads: the output equals W_o (W_v c + b_v) + b_o
    in float64 numpy, and changing the query input or the q/k weights leaves it
    bit-identical. Pinned until the attention has more than one key (or is replaced by
    a projection that says what it is).
    """
    attn = CrossAttention(4, num_heads=2, dropout=0.0).eval()
    w = _set_from_numpy(attn, seed=1)
    ctx = torch.tensor([[[0.5, -1.0, 2.0, 0.0]], [[1.0, 1.0, -0.5, 0.25]]])
    x = torch.tensor([[[3.0, 0.0, -2.0, 1.0]], [[-1.0, 0.5, 0.5, 0.0]]])
    out = attn(x, ctx)
    v = ctx.squeeze(1).double().numpy() @ w["v_proj.weight"].T + w["v_proj.bias"]
    expected = v @ w["out_proj.weight"].T + w["out_proj.bias"]
    torch.testing.assert_close(
        out.squeeze(1).double(), torch.from_numpy(expected), atol=1e-5, rtol=0.0
    )
    again = attn(torch.full_like(x, 7.0), ctx)
    with torch.no_grad():
        attn.q_proj.weight.mul_(-3.0)
        attn.k_proj.bias.add_(5.0)
    after = attn(x, ctx)
    assert torch.equal(again, out)
    assert torch.equal(after, out)


def test_cross_attention_broadcasts_one_context_takes_the_first_or_fails() -> None:
    """X batch 2 with context batch 1: the context is expanded, so both rows equal the
    single-context output. x batch 1 with context batch 3: only context[0] is used.
    x batch 2 with context batch 3: neither branch applies and the head reshape fails
    with torch's view message (12 + 12 = 24 elements cannot be [2, 1, 2, 4]).
    """
    attn = CrossAttention(8, num_heads=2, dropout=0.0).eval()
    ctx = torch.arange(24, dtype=torch.float32).view(3, 1, 8) / 10.0
    x2 = torch.zeros(2, 1, 8)
    one = attn(x2, ctx[:1])
    assert torch.equal(one[0], one[1])
    single = attn(x2[:1], ctx[:1])
    torch.testing.assert_close(one[:1], single, atol=1e-6, rtol=0.0)
    assert torch.equal(attn(torch.zeros(1, 1, 8), ctx), single)
    with pytest.raises(
        RuntimeError,
        match=re.escape("shape '[2, 1, 2, 4]' is invalid for input of size 24"),
    ):
        attn(x2, ctx)
    with pytest.raises(
        AssertionError, match=re.escape("dim must be divisible by num_heads")
    ):
        CrossAttention(6, num_heads=4)


# ------------------------------------------------------------------ denoiser


@pytest.mark.parametrize("norm", ["layer", "batch", "none"])
def test_denoise_matches_a_float64_numpy_reimplementation(norm: str) -> None:
    """Every parameter and running statistic set from ``np.random.default_rng(3)``; eval
    mode. The oracle (``_np_denoise``) applies, per sample: context_proj, input_proj,
    + time MLP of the sinusoidal embedding, + cross-attention of norm1(h) over the
    projected context, + MLP of norm2(h), then LayerNorm -> Linear -> GELU -> Linear.
    "layer" and "batch" use their normalizations (batch: running statistics), anything
    else is the identity. Inputs: x_t = [[0.3], [-1.2]], t = [0, 7], context 2 x 4.
    """
    dec = _decoder(norm=norm).eval()
    w = _set_from_numpy(dec, seed=3)
    x_t = np.array([[0.3], [-1.2]])
    ctx = np.array([[0.5, -1.0, 2.0, 0.0], [1.0, 1.0, -0.5, 0.25]])
    t = np.array([0, 7])
    out = dec.denoise(
        torch.tensor(x_t, dtype=torch.float32),
        torch.tensor(ctx, dtype=torch.float32),
        torch.from_numpy(t),
    )
    expected = _np_denoise(x_t, ctx, t, w, norm, heads=2, dim=4)
    assert out.shape == (2, 1)
    torch.testing.assert_close(
        out.double(), torch.from_numpy(expected), atol=1e-5, rtol=0.0
    )


def test_unknown_norm_silently_builds_identity_norms() -> None:
    """Finding: ``DenoisingBlock`` maps any norm other than "layer" or "batch" to
    ``nn.Identity`` (diffusion_decoder.py:130-133), so ``norm="instance"`` builds without
    a word and with 3 * 8 = 24 fewer parameters per block than "layer" (hidden 4: 389 vs
    365). Pinned until unknown norms are refused.
    """
    inst = _decoder(norm="instance")
    layer = _decoder(norm="layer")
    block = inst.blocks[0]
    assert isinstance(block, DenoisingBlock)
    assert [type(m) for m in (block.norm1, block.norm2, block.norm3)] == [
        nn.Identity
    ] * 3
    assert sum(p.numel() for p in inst.parameters()) == 365
    assert sum(p.numel() for p in layer.parameters()) == 389


def test_predict_x0_flag_is_ignored_and_eps_parameterization_cannot_run() -> None:
    """Finding: ``denoise`` documents ``predict_x0=False`` as "predict noise" but never
    reads the flag (diffusion_decoder.py:336-374): the two calls are bit-identical.
    ``parameterization="eps"`` passes the constructor's check (lines 221-223) yet every
    ``denoise`` raises ``NotImplementedError``, so the eps branch of ``loss``
    (lines 504-508) is unreachable. Under "x0", ``loss(predict_x0=False)`` runs the full
    forward and only then fails its assertion. Pinned until eps is implemented or
    refused at construction and the flag is removed.
    """
    dec = _decoder().eval()
    x_t, ctx, t = torch.tensor([[0.3]]), torch.ones(1, 4), torch.tensor([4])
    assert torch.equal(
        dec.denoise(x_t, ctx, t, predict_x0=True),
        dec.denoise(x_t, ctx, t, predict_x0=False),
    )
    eps = _decoder(parameterization="eps")
    assert eps.parameterization == "eps"
    message = re.escape("Parameterization eps not fully implemented")
    with pytest.raises(NotImplementedError, match=message):
        eps.loss(torch.zeros(2, 1), torch.zeros(2, 4), predict_x0=False)
    with pytest.raises(
        AssertionError, match=re.escape("x0 parameterization requires predict_x0=True")
    ):
        dec.loss(torch.zeros(2, 1), torch.zeros(2, 4), predict_x0=False)


# ------------------------------------------------------------------ schedules


def test_linear_schedule_buffers_match_the_float64_formula() -> None:
    """Beta = linspace(1e-4, 0.02, 10) (step 0.0199 / 9); alphas = 1 - beta; abar =
    cumprod; sqrt and sqrt(1 - abar). Float32 buffers agree to 1e-6. abar_0 = 0.9999,
    abar_9 = 0.903739... (product of the ten alphas).
    """
    dec = _decoder()
    np.testing.assert_allclose(dec.betas.double().numpy(), LINEAR_BETAS_T10, atol=1e-8)
    np.testing.assert_allclose(
        dec.alphas.double().numpy(), 1 - LINEAR_BETAS_T10, atol=1e-7
    )
    np.testing.assert_allclose(
        dec.alphas_cumprod.double().numpy(), LINEAR_ABAR_T10, atol=1e-6
    )
    np.testing.assert_allclose(
        dec.sqrt_alphas_cumprod.double().numpy(), np.sqrt(LINEAR_ABAR_T10), atol=1e-6
    )
    np.testing.assert_allclose(
        dec.sqrt_one_minus_alphas_cumprod.double().numpy(),
        np.sqrt(1 - LINEAR_ABAR_T10),
        atol=1e-6,
    )
    assert dec.alphas_cumprod[-1].item() == pytest.approx(0.9037394, abs=1e-6)


def test_cosine_schedule_code_is_the_formula_but_float32_misses_it_beyond_1e6() -> None:
    """Finding: at T = 1000 the cosine betas are computed in float32 (linspace and cos,
    diffusion_decoder.py:283-289). Run in float64, the same expression matches the numpy
    formula to 1e-12, so the gap is precision only: near x = T, f(x) = cos^2(~pi/2) is a
    difference of nearly equal float32 numbers and the ratio abar_{t+1} / abar_t loses
    digits. Platform-independent statement pinned: the float32 buffer does NOT match
    the formula to 1e-6 (the error is driven by rounding the float32 cos argument near
    pi / 2, not by one library's cos) and does match it to 1e-4. The size of the miss
    depends on the platform's float32 cos (3.1e-5 at t = 998 on the reference machine),
    so no tighter band is asserted. sqrt(1 - abar_0) is 0.0100008 instead of 0.01:
    float32(1 - 1e-4) is a correctly rounded IEEE subtraction on every platform, and
    1 minus it is 1.0002e-4. The effect on training is negligible (audit D8). Pinned
    until the schedule is built in float64 and cast once.
    """
    T = 1000
    dec = _decoder(num_timesteps=T, beta_schedule="cosine")
    raw, _ = _cosine_formula(T)
    clipped = np.clip(raw, 1e-4, 0.9999)
    x = torch.linspace(0, T, T + 1, dtype=torch.float64)
    f = torch.cos(((x / T) + 0.008) / (1 + 0.008) * torch.pi * 0.5) ** 2
    f = f / f[0]
    np.testing.assert_allclose(
        torch.clip(1 - f[1:] / f[:-1], 0.0001, 0.9999).numpy(), clipped, atol=1e-12
    )
    gap = np.abs(dec.betas.double().numpy() - clipped)
    assert 1e-6 < gap.max() < 1e-4
    assert dec.sqrt_one_minus_alphas_cumprod[0].item() == pytest.approx(
        0.0100008, abs=5e-7
    )
    assert math.sqrt(clipped[0]) == pytest.approx(0.01, abs=1e-12)


def test_cosine_schedule_clips_the_first_13_betas_up_and_the_last_one_down() -> None:
    """Finding: the docstring cites Nichol and Dhariwal (2021), whose schedule clips
    beta only from above (at 0.999). This code also clips from below at 1e-4
    (diffusion_decoder.py:290): at T = 1000 the first 13 formula betas (4.13e-5,
    4.61e-5, ..., all below 1e-4) are raised to 1e-4, so abar no longer equals f(t)/f(0)
    (largest gap 3.8e-4). The last formula beta is 1 (f(T) ~ 0) and is clipped to 0.9999,
    so abar_{T-1} = 2.43e-10. With T = 10 no beta is below 1e-4 and only the last is
    clipped. Pinned until the clip matches the cited schedule or the docstring says why.
    """
    raw, formula_abar = _cosine_formula(1000)
    assert int((raw < 1e-4).sum()) == 13
    assert np.flatnonzero(raw < 1e-4).tolist() == list(range(13))
    assert raw[0] == pytest.approx(4.128422e-5, rel=1e-5)
    assert raw[-1] == pytest.approx(1.0, abs=1e-12)
    dec = _decoder(num_timesteps=1000, beta_schedule="cosine")
    torch.testing.assert_close(
        dec.betas[:13], torch.full((13,), 1e-4), atol=0.0, rtol=0.0
    )
    assert dec.betas[13].item() == pytest.approx(raw[13], abs=1e-7)
    assert dec.betas[-1].item() == pytest.approx(0.9999, abs=1e-7)
    clipped_abar = np.cumprod(1 - np.clip(raw, 1e-4, 0.9999))
    assert np.abs(clipped_abar - formula_abar).max() == pytest.approx(3.84e-4, abs=1e-5)
    assert dec.alphas_cumprod[-1].item() == pytest.approx(2.4284e-10, rel=1e-3)
    raw10, _ = _cosine_formula(10)
    small = _decoder(num_timesteps=10, beta_schedule="cosine")
    np.testing.assert_allclose(small.betas[:-1].double().numpy(), raw10[:-1], atol=1e-6)
    assert small.betas[-1].item() == pytest.approx(0.9999, abs=1e-7)


def test_constructor_refusals_name_the_bad_value() -> None:
    """num_timesteps 0, schedule "quad" and parameterization "v" are refused by name."""
    with pytest.raises(
        AssertionError, match=re.escape("num_timesteps must be >= 1, got 0")
    ):
        _decoder(num_timesteps=0)
    with pytest.raises(ValueError, match=re.escape("Unknown beta schedule: quad")):
        _decoder(beta_schedule="quad")
    with pytest.raises(AssertionError, match=re.escape("Unknown parameterization: v")):
        _decoder(parameterization="v")


def test_parameter_counts_by_hand_for_the_tiny_and_the_shipped_006_decoder() -> None:
    """Tiny: 136 + 16 + 1168 + 97 = 1417 (module docstring). Shipped 006 (input 128 =
    2 x 64 hidden channels, hidden 64, 2 layers, 4 heads, mlp_ratio 4): context_proj
    128 * 64 + 64 = 8256, input_proj 128, per block 3 * 128 + 4 * 4160 + (16640 + 16448)
    + (8320 + 8256) = 66688, head 128 + 4160 + 65 = 4353; total 146113.
    """
    tiny = DiffusionDecoder(input_dim=16, hidden_dim=8, num_layers=1, num_heads=2)
    assert sum(p.numel() for p in tiny.parameters()) == 1417
    shipped = DiffusionDecoder(
        input_dim=128, hidden_dim=64, num_layers=2, num_heads=4, dropout=0.0
    )
    assert sum(p.numel() for p in shipped.parameters()) == 146113


# ------------------------------------------------------------------ q-sample


def test_forward_diffusion_is_the_closed_form_and_t0_returns_x0_with_zero_noise() -> (
    None
):
    """Linear T = 10, x_0 = [1, 2, -1], t = [0, 3, 9], eps = [0.5, -1, 2]:
    row 0 (t = 0) -> x_0 = 1 and noise 0; rows 1, 2 -> sqrt(abar_t) x_0 + sqrt(1 - abar_t)
    eps with abar from the float64 formula; their noise is returned unchanged.
    """
    dec = _decoder()
    x0 = torch.tensor([[1.0], [2.0], [-1.0]])
    eps = torch.tensor([[0.5], [-1.0], [2.0]])
    t = torch.tensor([0, 3, 9])
    x_t, noise = dec.forward_diffusion(x0, t, eps)
    ab = LINEAR_ABAR_T10
    expected = [
        1.0,
        math.sqrt(ab[3]) * 2.0 + math.sqrt(1 - ab[3]) * -1.0,
        math.sqrt(ab[9]) * -1.0 + math.sqrt(1 - ab[9]) * 2.0,
    ]
    torch.testing.assert_close(
        x_t.view(-1).double(),
        torch.tensor(expected, dtype=torch.float64),
        atol=1e-6,
        rtol=0.0,
    )
    assert x_t[0, 0].item() == 1.0
    assert noise.view(-1).tolist() == [0.0, -1.0, 2.0]


def test_forward_diffusion_draws_standard_normal_noise_when_none_is_given() -> None:
    """With ``noise=None`` the noise is ``randn_like(x_0)`` from the global generator:
    seeding 11 and drawing ``torch.randn(2, 1)`` reproduces it, and x_t follows it.
    """
    dec = _decoder()
    x0 = torch.tensor([[1.0], [2.0]])
    t = torch.tensor([5, 2])
    torch.manual_seed(11)
    x_t, noise = dec.forward_diffusion(x0, t)
    torch.manual_seed(11)
    expected_noise = torch.randn(2, 1)
    assert torch.equal(noise, expected_noise)
    ab = LINEAR_ABAR_T10
    expected = [
        math.sqrt(ab[5]) * 1.0 + math.sqrt(1 - ab[5]) * expected_noise[0, 0].item(),
        math.sqrt(ab[2]) * 2.0 + math.sqrt(1 - ab[2]) * expected_noise[1, 0].item(),
    ]
    torch.testing.assert_close(
        x_t.view(-1).double(),
        torch.tensor(expected, dtype=torch.float64),
        atol=1e-6,
        rtol=0.0,
    )


def test_a_one_dimensional_x0_broadcasts_to_a_batch_by_batch_matrix() -> None:
    """Finding: ``forward_diffusion`` documents x_0 as [batch, 1] but does not check it
    (diffusion_decoder.py:313-321). A 1-D x_0 of length 3 with t = [1, 2, 3] and 1-D
    noise broadcasts the [3, 1] coefficients against [3]: x_t is [3, 3] with
    x_t[i, j] = sqrt(abar_{t_i}) x_0[j] + sqrt(1 - abar_{t_i}) eps[j], every sample
    mixed with every other's target. The noise comes back 1-D, unchanged (no t = 0).
    Pinned until x_0 of the wrong rank is refused.
    """
    dec = _decoder()
    x0 = torch.tensor([1.0, 2.0, 3.0])
    eps = torch.tensor([0.1, 0.2, 0.3])
    x_t, noise = dec.forward_diffusion(x0, torch.tensor([1, 2, 3]), eps)
    ab = LINEAR_ABAR_T10[[1, 2, 3]]
    expected = (
        np.sqrt(ab)[:, None] * x0.double().numpy()[None, :]
        + np.sqrt(1 - ab)[:, None] * eps.double().numpy()[None, :]
    )
    assert x_t.shape == (3, 3)
    torch.testing.assert_close(
        x_t.double(), torch.from_numpy(expected), atol=1e-6, rtol=0.0
    )
    assert torch.equal(noise, eps)


# ------------------------------------------------------------------ DDIM sampling


Recorder = list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def _affine_denoise(calls: Recorder) -> Callable[..., torch.Tensor]:
    def fake(
        x: torch.Tensor, context: torch.Tensor, t: torch.Tensor, predict_x0: bool = True
    ) -> torch.Tensor:
        calls.append((x.clone(), context.clone(), t.clone()))
        return 0.5 * x + 0.01 * t.to(x.dtype).unsqueeze(1)

    return fake


def _np_ddim(x_start: Arr, ts: list[int], abar: Arr) -> list[Arr]:
    """Return the state fed to each step and, last, the returned sample."""
    states, x = [], x_start.astype(np.float64)
    for i, t in enumerate(ts):
        states.append(x)
        p = 0.5 * x + 0.01 * t
        if i < len(ts) - 1:
            tp = ts[i + 1]
            eps_hat = (x - math.sqrt(abar[t]) * p) / math.sqrt(1 - abar[t])
            x = math.sqrt(abar[tp]) * p + math.sqrt(1 - abar[tp]) * eps_hat
        else:
            x = p
    states.append(x)
    return states


def test_ddim_from_a_zero_start_is_the_deterministic_mean_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linear T = 10, sampling_steps 4 -> t = 9, 6, 3, 0. ``torch.randn`` returns zeros,
    so the start is x = 0 and the loop is deterministic (DDIM, eta 0). With the stand-in
    p = 0.5 x + 0.01 t the numpy replay gives the state at each step and the final
    sample, which is p at t = 0 of the last state. Each call sees t as a long tensor of the batch size
    and the context unchanged.
    """
    dec = _decoder()
    calls: Recorder = []
    monkeypatch.setattr(dec, "denoise", _affine_denoise(calls))

    def zeros(*size: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.zeros(*size, device=device)

    monkeypatch.setattr(torch, "randn", zeros)
    ctx = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 1.0]])
    out = dec.sample(ctx, sampling_steps=4)
    ts = [9, 6, 3, 0]
    states = _np_ddim(np.zeros((2, 1)), ts, LINEAR_ABAR_T10)
    assert [c[2].tolist() for c in calls] == [[t, t] for t in ts]
    assert all(c[2].dtype == torch.long for c in calls)
    assert all(torch.equal(c[1], ctx) for c in calls)
    for (x_in, _, _), state in zip(calls, states[:-1], strict=True):
        torch.testing.assert_close(
            x_in.double(), torch.from_numpy(state), atol=1e-6, rtol=0.0
        )
    torch.testing.assert_close(
        out.double(), torch.from_numpy(states[-1]), atol=1e-6, rtol=0.0
    )
    # Hand check of the first move, 9 -> 6, from x = 0: p = 0.09.
    eps9 = (0 - math.sqrt(LINEAR_ABAR_T10[9]) * 0.09) / math.sqrt(
        1 - LINEAR_ABAR_T10[9]
    )
    x6 = math.sqrt(LINEAR_ABAR_T10[6]) * 0.09 + math.sqrt(1 - LINEAR_ABAR_T10[6]) * eps9
    assert calls[1][0][0, 0].item() == pytest.approx(x6, abs=1e-6)


def test_ddim_from_a_seeded_start_replays_the_same_recurrence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seed 7: the start is the first ``torch.randn(2, 1)`` after seeding, and the loop
    is the same recurrence from it; the sample is a pure function of that draw.
    """
    dec = _decoder()
    calls: Recorder = []
    monkeypatch.setattr(dec, "denoise", _affine_denoise(calls))
    torch.manual_seed(7)
    start = torch.randn(2, 1)
    torch.manual_seed(7)
    out = dec.sample(torch.zeros(2, 4), sampling_steps=4)
    states = _np_ddim(start.double().numpy(), [9, 6, 3, 0], LINEAR_ABAR_T10)
    assert torch.equal(calls[0][0], start)
    torch.testing.assert_close(
        out.double(), torch.from_numpy(states[-1]), atol=1e-6, rtol=0.0
    )


def test_ddim_step_lists_and_the_number_of_denoise_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Steps actually taken (one ``denoise`` call each):

    * T = 10, steps not given: min(50, 10) = 10 -> t = 9, 8, ..., 0.
    * T = 1000, steps not given: 50 calls on reversed floor(linspace(0, 999, 50)), which
      starts 999, 978, 958 and ends 61, 40, 20, 0. It truncates rather than rounds:
      999 / 49 * 48 = 978.6 -> 978 and 999 / 49 * 3 = 61.2 -> 61.
    * T = 3, steps 5 given explicitly: no cap, so floor(linspace(0, 2, 5)) = [0, 0, 1, 1,
      2] reversed: 5 calls on 3 distinct timesteps.
    * T = 1: a single call at t = 0.
    """
    taken: dict[tuple[int, int | None], list[int]] = {}
    for T, steps in [(10, None), (1000, None), (3, 5), (1, None)]:
        dec = _decoder(num_timesteps=T)
        calls: Recorder = []
        monkeypatch.setattr(dec, "denoise", _affine_denoise(calls))
        dec.sample(torch.zeros(1, 4), sampling_steps=steps)
        taken[(T, steps)] = [int(c[2][0]) for c in calls]
    assert taken[(10, None)] == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    long_run = taken[(1000, None)]
    assert len(long_run) == 50
    assert long_run[:3] == [999, 978, 958]
    assert long_run[-4:] == [61, 40, 20, 0]
    assert long_run == [int(999 / 49 * k) for k in range(49, -1, -1)]
    assert taken[(3, 5)] == [2, 1, 1, 0, 0]
    assert taken[(1, None)] == [0]


def test_sampler_feeds_t0_a_noisy_state_that_training_never_shows_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding: ``forward_diffusion`` treats t = 0 as clean (x_t = x_0, lines 309-327),
    so the network is only ever trained at t = 0 on x_0 itself. The DDIM step into
    t_prev = 0 (lines 434-447) instead builds sqrt(abar_0) p + sqrt(1 - abar_0) eps_hat
    from the schedule, whose index 0 is one step of noise (abar_0 = 0.9999 here, so the
    noise weight is 0.01). With a constant stand-in p = 2 from a zero start (linear
    T = 10, steps 9, 6, 3, 0) eps_hat stays at -2 sqrt(abar_9) / sqrt(1 - abar_9) =
    -6.13, and the t = 0 call receives 2 sqrt(abar_0) + 0.01 eps_hat = 1.93862, not the
    2.0 that q-sample gives at t = 0 (float32 loop vs float64 replay agree to 1e-5).
    The 1.93862 is a property of the constant stand-in from a zero start. The general
    closed form, which holds for any denoiser, is that the t = 0 input minus
    sqrt(abar_0) p equals sqrt(1 - abar_0) eps_hat, about 0.01 eps_hat, where eps_hat is
    the noise implied at the previous step; this is asserted as well. Pinned until the
    sampler and q-sample agree on what t = 0 means.
    """
    dec = _decoder()
    seen: list[float] = []

    def constant(
        x: torch.Tensor, context: torch.Tensor, t: torch.Tensor, predict_x0: bool = True
    ) -> torch.Tensor:
        seen.append(float(x[0, 0]))
        return torch.full_like(x, 2.0)

    monkeypatch.setattr(dec, "denoise", constant)
    monkeypatch.setattr(
        torch, "randn", lambda *s, device=None: torch.zeros(*s, device=device)
    )
    out = dec.sample(torch.zeros(1, 4), sampling_steps=4)
    ab, x = LINEAR_ABAR_T10, 0.0
    for t, tp in [(9, 6), (6, 3), (3, 0)]:
        eps_hat = (x - math.sqrt(ab[t]) * 2.0) / math.sqrt(1 - ab[t])
        x = math.sqrt(ab[tp]) * 2.0 + math.sqrt(1 - ab[tp]) * eps_hat
    assert seen[-1] == pytest.approx(x, abs=1e-5)
    assert x == pytest.approx(1.93862, abs=1e-5)
    eps_at_3 = (seen[2] - math.sqrt(ab[3]) * 2.0) / math.sqrt(1 - ab[3])
    assert seen[-1] - math.sqrt(ab[0]) * 2.0 == pytest.approx(
        math.sqrt(1 - ab[0]) * eps_at_3, abs=1e-5
    )
    assert math.sqrt(1 - ab[0]) == pytest.approx(0.01, abs=1e-6)
    clean, _ = dec.forward_diffusion(torch.tensor([[2.0]]), torch.tensor([0]))
    assert clean.item() == 2.0
    assert out.item() == 2.0


def test_sample_batch_size_rules_and_the_silently_ignored_num_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding: with 2 contexts, ``num_samples=3`` returns 2 rows (the request is
    dropped without a word, diffusion_decoder.py:407-409). One context with
    ``num_samples=3`` is expanded to 3 identical rows; 3 contexts with ``num_samples=1``
    keep only context[0]. Pinned until an unsatisfiable ``num_samples`` is refused.
    """
    dec = _decoder()
    calls: Recorder = []
    monkeypatch.setattr(dec, "denoise", _affine_denoise(calls))
    ctx3 = torch.arange(12, dtype=torch.float32).view(3, 4)
    assert dec.sample(ctx3[:2], num_samples=3, sampling_steps=2).shape == (2, 1)
    assert torch.equal(calls[-1][1], ctx3[:2])
    calls.clear()
    assert dec.sample(ctx3[:1], num_samples=3, sampling_steps=2).shape == (3, 1)
    assert torch.equal(calls[-1][1], ctx3[:1].expand(3, -1))
    calls.clear()
    assert dec.sample(ctx3, num_samples=1, sampling_steps=2).shape == (1, 1)
    assert torch.equal(calls[-1][1], ctx3[:1])


def test_real_sampler_is_seeded_row_equivariant_and_moves_with_the_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real denoiser (seeded init, eval). Same seed -> bit-identical samples. From a zero
    start, permuting the context rows permutes the samples (no cross-row mixing under
    LayerNorm in eval). Changing one context row moves that row's sample and leaves the
    other row bit-identical. ``sample`` runs under ``no_grad``.
    """
    dec = _decoder(num_timesteps=20).eval()
    ctx = torch.tensor([[0.5, -1.0, 2.0, 0.0], [1.0, 1.0, -0.5, 0.25]])
    torch.manual_seed(4)
    a = dec.sample(ctx)
    torch.manual_seed(4)
    b = dec.sample(ctx)
    assert torch.equal(a, b)
    assert not a.requires_grad
    monkeypatch.setattr(
        torch, "randn", lambda *s, device=None: torch.zeros(*s, device=device)
    )
    base = dec.sample(ctx)
    swapped = dec.sample(ctx[[1, 0]])
    torch.testing.assert_close(swapped, base[[1, 0]], atol=1e-6, rtol=0.0)
    moved_ctx = ctx.clone()
    moved_ctx[1] = torch.tensor([-2.0, 3.0, 0.0, 1.0])
    moved = dec.sample(moved_ctx)
    assert torch.equal(moved[0], base[0])
    assert abs(moved[1, 0].item() - base[1, 0].item()) > 1e-3


# ------------------------------------------------------------------ loss


def test_loss_draws_t_then_noise_and_scores_the_x0_prediction() -> None:
    """t_mode "full", seed 5: the loss draws t = randint(0, 10, (3,)) and then eps =
    randn_like(x_0), noises x_0 with the closed form (t = 0 rows stay clean), and returns
    mean((denoise(x_t, c, t) - x_0)^2). Replaying those two draws under the same seed
    reproduces the loss exactly. "random" is the same mode.
    """
    dec = _decoder().eval()
    x0 = torch.tensor([[1.0], [-0.5], [2.0]])
    ctx = torch.tensor(
        [[0.5, -1.0, 2.0, 0.0], [1.0, 1.0, -0.5, 0.25], [0.0, 2.0, 1.0, -1.0]]
    )
    for mode in ["full", "random"]:
        torch.manual_seed(5)
        loss = dec.loss(x0, ctx, t_mode=mode)
        torch.manual_seed(5)
        t = torch.randint(0, 10, (3,))
        eps = torch.randn(3, 1)
        ab = torch.tensor(LINEAR_ABAR_T10, dtype=torch.float32)[t].view(-1, 1)
        x_t = torch.where(
            (t == 0).view(-1, 1), x0, ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        )
        pred = dec.denoise(x_t, ctx, t)
        expected = ((pred - x0) ** 2).mean()
        torch.testing.assert_close(loss, expected, atol=1e-6, rtol=0.0)


def test_loss_zero_mode_is_clean_reconstruction_and_partial_collapses_at_t10() -> None:
    """Modes "zero"/"t0": t = 0 everywhere, so x_t = x_0 and the loss is
    mean((denoise(x_0, c, 0) - x_0)^2) whatever the noise draw. "partial"/"small" draw t
    from [0, max(1, T // 10)); at T = 10 that is {0}, so partial equals zero mode
    exactly. At T = 50, partial draws from [0, 5): seed 2 replays randint(0, 5, (3,)).
    An unknown mode is refused by name.
    """
    dec = _decoder().eval()
    x0 = torch.tensor([[1.0], [-0.5], [2.0]])
    ctx = torch.tensor(
        [[0.5, -1.0, 2.0, 0.0], [1.0, 1.0, -0.5, 0.25], [0.0, 2.0, 1.0, -1.0]]
    )
    clean = ((dec.denoise(x0, ctx, torch.zeros(3, dtype=torch.long)) - x0) ** 2).mean()
    for mode in ["zero", "t0", "partial", "small"]:
        torch.testing.assert_close(
            dec.loss(x0, ctx, t_mode=mode), clean, atol=0.0, rtol=0.0
        )
    dec50 = _decoder(num_timesteps=50).eval()
    torch.manual_seed(2)
    loss = dec50.loss(x0, ctx, t_mode="partial")
    torch.manual_seed(2)
    t = torch.randint(0, 5, (3,))
    eps = torch.randn(3, 1)
    x_t, _ = dec50.forward_diffusion(x0, t, eps)
    torch.testing.assert_close(
        loss, ((dec50.denoise(x_t, ctx, t) - x0) ** 2).mean(), atol=1e-6, rtol=0.0
    )
    assert int(t.max()) < 5
    with pytest.raises(ValueError, match=re.escape("Unknown t_mode: bogus")):
        dec.loss(x0, ctx, t_mode="bogus")


def test_loss_gradient_skips_norm3_and_is_exactly_zero_on_the_query_path() -> None:
    """Finding: backward of the loss (train mode, dropout 0, t_mode "full") leaves
    ``blocks.0.norm3`` with no gradient at all (built at diffusion_decoder.py:125/129,
    never called), and gives exactly zero gradient to norm1, q_proj and k_proj: they only
    feed the score of a one-key softmax, whose derivative is y (g - g y) = 0 for y = 1.
    Every other parameter gets a finite, nonzero gradient. In the shipped 006 decoder
    (2 blocks, hidden 64) that is 2 * (4160 + 4160 + 128) = 16896 parameters that never
    move and 2 * 128 = 256 that are never used. Pinned until the attention has a real
    key set and norm3 is used or removed.
    """
    dec = _decoder().train()
    x0 = torch.tensor([[1.0], [-0.5], [2.0]])
    ctx = torch.tensor(
        [[0.5, -1.0, 2.0, 0.0], [1.0, 1.0, -0.5, 0.25], [0.0, 2.0, 1.0, -1.0]]
    )
    torch.manual_seed(5)
    dec.loss(x0, ctx).backward()
    none = sorted(n for n, p in dec.named_parameters() if p.grad is None)
    zero = sorted(
        n for n, p in dec.named_parameters() if p.grad is not None and not p.grad.any()
    )
    assert none == ["blocks.0.norm3.bias", "blocks.0.norm3.weight"]
    assert zero == [
        "blocks.0.cross_attn.k_proj.bias",
        "blocks.0.cross_attn.k_proj.weight",
        "blocks.0.cross_attn.q_proj.bias",
        "blocks.0.cross_attn.q_proj.weight",
        "blocks.0.norm1.bias",
        "blocks.0.norm1.weight",
    ]
    for n, p in dec.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n
    shipped = DiffusionDecoder(input_dim=128, hidden_dim=64, num_layers=2, num_heads=4)
    frozen = sum(
        p.numel()
        for n, p in shipped.named_parameters()
        if ".q_proj." in n or ".k_proj." in n or ".norm1." in n
    )
    unused = sum(p.numel() for n, p in shipped.named_parameters() if ".norm3." in n)
    assert (frozen, unused) == (16896, 256)


# ------------------------------------------------------------------ Phase 24: eps loss


class _NoisePredictor:
    """Stands in for ``denoise``: returns ``0.5 * x_t`` and records its arguments."""

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, bool]] = []

    def __call__(
        self,
        x_t: torch.Tensor,
        context: torch.Tensor,
        t: torch.Tensor,
        predict_x0: bool,
    ) -> torch.Tensor:
        self.calls.append((x_t, t, predict_x0))
        return 0.5 * x_t


def test_eps_loss_scores_the_prediction_against_the_added_noise() -> None:
    """With ``denoise`` replaced (it raises for "eps", see the Finding above), the eps
    branch of ``loss`` (diffusion_decoder.py:504-508) runs: t_mode "full", seed 4 draws
    t = randint(0, 10, (3,)) = [0, 4, 1] then eps = randn(3, 1); x_t is the closed-form
    q-sample (the t = 0 row keeps x_0 and gets zero noise), the stand-in predicts
    0.5 x_t, and the loss is mean((0.5 x_t - noise)^2) with that zeroed noise, so
    scoring against the raw eps instead of ``actual_noise`` would fail on row 0.
    ``predict_x0=False`` reaches the stand-in unchanged.
    """
    dec = _decoder(parameterization="eps").eval()
    fake = _NoisePredictor()
    object.__setattr__(dec, "denoise", fake)
    x0 = torch.tensor([[1.0], [-0.5], [2.0]])
    ctx = torch.zeros(3, 4)
    torch.manual_seed(4)
    loss = dec.loss(x0, ctx, predict_x0=False, t_mode="full")
    torch.manual_seed(4)
    t = torch.randint(0, 10, (3,))
    assert t.tolist() == [0, 4, 1]
    eps = torch.randn(3, 1)
    assert not torch.equal(((0.5 * x0[:1] - 0.0) ** 2), ((0.5 * x0[:1] - eps[:1]) ** 2))
    ab = torch.tensor(LINEAR_ABAR_T10, dtype=torch.float32)[t].view(-1, 1)
    zero = (t == 0).view(-1, 1)
    x_t = torch.where(zero, x0, ab.sqrt() * x0 + (1 - ab).sqrt() * eps)
    noise = torch.where(zero, torch.zeros_like(eps), eps)
    expected = ((0.5 * x_t - noise) ** 2).mean()
    torch.testing.assert_close(loss, expected, atol=1e-6, rtol=0.0)
    ((seen_x_t, seen_t, seen_flag),) = fake.calls
    assert torch.equal(seen_t, t)
    torch.testing.assert_close(seen_x_t, x_t, atol=1e-6, rtol=0.0)
    assert seen_flag is False


def test_eps_loss_refuses_predict_x0_and_an_unknown_parameterization_is_named() -> None:
    """Under "eps", ``predict_x0=True`` (the default) fails the assertion after the
    forward; a parameterization rewritten after construction to "v" (the constructor
    refuses it) reaches the final ``else`` and raises ValueError naming it.
    """
    dec = _decoder(parameterization="eps").eval()
    object.__setattr__(dec, "denoise", _NoisePredictor())
    x0, ctx = torch.zeros(2, 1), torch.zeros(2, 4)
    with pytest.raises(
        AssertionError,
        match=re.escape("eps parameterization requires predict_x0=False"),
    ):
        dec.loss(x0, ctx)
    dec.parameterization = "v"
    with pytest.raises(ValueError, match=re.escape("Unknown parameterization: v")):
        dec.loss(x0, ctx, predict_x0=False)
