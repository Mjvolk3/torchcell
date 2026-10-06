# tests/torchcell/losses/test_diffusion_loss.py
# [[tests.torchcell.losses.test_diffusion_loss]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/losses/test_diffusion_loss.py
"""``DiffusionLoss``: the weighting, the x0 term and the timestep sampler, exactly.

Fixture: ``_FakeModel``, an nn.Module whose ``compute_diffusion_loss`` returns 0.75 and
records its arguments, and whose ``_FakeDecoder`` (num_timesteps T) returns
x_t = targets + noise from ``forward_diffusion`` and x0_hat = x_t from ``denoise``, so
x0_hat - targets is exactly the sampled noise. Under a fixed seed the loss draws t
first and then the noise, so the test regenerates both with the same seed:
x0 MSE = mean(noise^2), L1 = mean(|noise|), total = lambda_diffusion * 0.75 +
lambda_x0 * x0 term. The sampler is checked against ``torch.randint`` under the same
seed for each mode.
"""

import re
from collections.abc import Iterator

import pytest
import torch
import torch.nn as nn

from torchcell.losses.diffusion_loss import DiffusionLoss

TARGETS = torch.tensor([[0.5, -1.0], [2.0, 0.0], [1.0, 1.0]])
CONTEXT = torch.tensor([[1.0], [2.0], [3.0]])


@pytest.fixture(autouse=True)
def _fork_rng() -> Iterator[None]:
    """Every test runs inside its own RNG fork so seeding never leaks."""
    with torch.random.fork_rng():
        yield


class _FakeDecoder(nn.Module):
    def __init__(self, num_timesteps: int) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.diffusion_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.denoise_calls: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]
        ] = []

    def forward_diffusion(
        self, targets: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.diffusion_calls.append((targets, t, noise))
        return targets + noise, noise

    def denoise(
        self,
        x_t: torch.Tensor,
        context: torch.Tensor,
        t: torch.Tensor,
        *,
        predict_x0: bool,
    ) -> torch.Tensor:
        self.denoise_calls.append((x_t, context, t, predict_x0))
        return x_t


class _FakeModel(nn.Module):
    def __init__(self, num_timesteps: int = 100) -> None:
        super().__init__()
        self.diffusion_decoder = _FakeDecoder(num_timesteps)
        self.loss_calls: list[tuple[torch.Tensor, torch.Tensor, str]] = []

    def compute_diffusion_loss(
        self, targets: torch.Tensor, context: torch.Tensor, *, t_mode: str
    ) -> torch.Tensor:
        self.loss_calls.append((targets, context, t_mode))
        return torch.tensor(0.75)


def _expected_t_and_noise(seed: int, high: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    t = torch.randint(0, high, (3,))
    return t, torch.randn_like(TARGETS)


def test_both_terms_closed_form_and_call_arguments() -> None:
    """lambda_diffusion 0.5, lambda_x0 2, mode 'full', T = 100, seed 0:
    total = 0.5 * 0.75 + 2 * mean(noise^2); the decoder sees t ~ randint(0, 100).
    """
    model = _FakeModel(100)
    loss = DiffusionLoss(model, lambda_diffusion=0.5, lambda_x0=2.0)
    torch.manual_seed(0)
    total, parts = loss(torch.zeros(3, 2), TARGETS, CONTEXT, epoch=7)
    t, noise = _expected_t_and_noise(0, 100)
    x0 = noise.pow(2).mean()
    assert sorted(parts) == ["diffusion_loss", "total_loss", "x0_mse"]
    assert parts["diffusion_loss"].item() == 0.75
    torch.testing.assert_close(parts["x0_mse"], x0)
    torch.testing.assert_close(total, 0.5 * torch.tensor(0.75) + 2 * x0)
    assert parts["total_loss"] is total
    (seen_targets, seen_context, seen_mode) = model.loss_calls[0]
    assert seen_targets is TARGETS and seen_context is CONTEXT and seen_mode == "full"
    d_targets, d_t, d_noise = model.diffusion_decoder.diffusion_calls[0]
    assert d_targets is TARGETS
    assert torch.equal(d_t, t) and torch.equal(d_noise, noise)
    x_t, ctx, t_seen, predict_x0 = model.diffusion_decoder.denoise_calls[0]
    assert torch.equal(x_t, TARGETS + noise) and ctx is CONTEXT
    assert t_seen is d_t and predict_x0 is True


def test_diffusion_only_skips_the_decoder_and_x0_only_skips_the_model_loss() -> None:
    """lambda_x0 0 (default): only diffusion_loss, total = 3 * 0.75 = 2.25. lambda_diffusion
    0 with lambda_x0 1: no model loss call, total = mean(noise^2).
    """
    model = _FakeModel()
    total, parts = DiffusionLoss(model, lambda_diffusion=3.0)(TARGETS, TARGETS, CONTEXT)
    assert total.item() == pytest.approx(2.25)
    assert sorted(parts) == ["diffusion_loss", "total_loss"]
    assert model.diffusion_decoder.diffusion_calls == []

    model = _FakeModel()
    torch.manual_seed(1)
    total, parts = DiffusionLoss(model, lambda_diffusion=0.0, lambda_x0=1.0)(
        TARGETS, TARGETS, CONTEXT
    )
    _, noise = _expected_t_and_noise(1, 100)
    assert model.loss_calls == []
    assert sorted(parts) == ["total_loss", "x0_mse"]
    torch.testing.assert_close(total, noise.pow(2).mean())


def test_custom_x0_loss_replaces_the_mse() -> None:
    """x0_loss = L1: x0 term = mean(|noise|); total = 0.75 + 0.5 mean(|noise|)."""
    model = _FakeModel()
    loss = DiffusionLoss(model, lambda_x0=0.5, x0_loss=nn.L1Loss())
    torch.manual_seed(2)
    total, parts = loss(TARGETS, TARGETS, CONTEXT)
    _, noise = _expected_t_and_noise(2, 100)
    torch.testing.assert_close(parts["x0_mse"], noise.abs().mean())
    torch.testing.assert_close(total, torch.tensor(0.75) + 0.5 * noise.abs().mean())


@pytest.mark.parametrize(
    ("num_timesteps", "mode", "high"),
    [(100, "full", 100), (100, "partial", 10), (25, "partial", 2), (5, "partial", 1)],
)
def test_sample_t_ranges(num_timesteps: int, mode: str, high: int) -> None:
    """'full' draws randint(0, T); 'partial' draws randint(0, max(1, T // 10)):
    T = 100 -> [0, 10), T = 25 -> [0, 2), T = 5 -> only 0. Same draws as torch.randint
    under the same seed, dtype long.
    """
    loss = DiffusionLoss(_FakeModel(num_timesteps), lambda_diffusion=1.0)
    torch.manual_seed(3)
    t = loss._sample_t(64, torch.device("cpu"), mode)
    torch.manual_seed(3)
    expected = torch.randint(0, high, (64,))
    assert t.dtype == torch.long
    assert torch.equal(t, expected)
    assert int(t.max().item()) < high


def test_sample_t_zero_mode_draws_nothing() -> None:
    """'zero' returns zeros(B) and leaves the RNG untouched."""
    loss = DiffusionLoss(_FakeModel(), lambda_diffusion=1.0)
    torch.manual_seed(4)
    t = loss._sample_t(5, torch.device("cpu"), "zero")
    after = torch.rand(1)
    torch.manual_seed(4)
    assert torch.equal(t, torch.zeros(5, dtype=torch.long))
    assert t.dtype == torch.long
    assert torch.equal(after, torch.rand(1))


def test_unknown_t_mode_silently_samples_the_full_range() -> None:
    """Finding: a misspelled t_mode is not refused; it samples randint(0, T).

    The class docstring promises "no silent fallbacks", but ``_sample_t`` treats any
    mode other than 'zero' and 'partial' as 'full' (diffusion_loss.py:90), so
    t_mode='parital' trains on the full noise range without a warning. Reach: latent;
    the only diffusion config sets t_mode "full". Pinned until unknown modes raise.
    """
    loss = DiffusionLoss(_FakeModel(100), lambda_diffusion=1.0)
    torch.manual_seed(5)
    t = loss._sample_t(64, torch.device("cpu"), "parital")
    torch.manual_seed(5)
    assert torch.equal(t, torch.randint(0, 100, (64,)))


def test_refusals_exact_messages() -> None:
    """Both lambdas <= 0, a model without either hook, and a missing context."""
    with pytest.raises(
        ValueError,
        match=re.escape("At least one of {lambda_diffusion, lambda_x0} > 0."),
    ):
        DiffusionLoss(_FakeModel(), lambda_diffusion=0.0, lambda_x0=-1.0)
    no_loss = nn.Module()
    no_loss.diffusion_decoder = _FakeDecoder(10)
    with pytest.raises(
        AttributeError, match=re.escape("model must implement compute_diffusion_loss")
    ):
        DiffusionLoss(no_loss)
    no_decoder = _FakeModel()
    del no_decoder.diffusion_decoder
    with pytest.raises(
        AttributeError, match=re.escape("model must have diffusion_decoder")
    ):
        DiffusionLoss(no_decoder)
    with pytest.raises(
        ValueError, match=re.escape("context (conditioning) tensor is required.")
    ):
        DiffusionLoss(_FakeModel())(TARGETS, TARGETS, None)


def test_the_wrapped_model_is_a_registered_submodule() -> None:
    """``self.model = model`` registers it: its decoder's parameters are the loss's."""
    model = _FakeModel()
    model.diffusion_decoder.scale = nn.Parameter(torch.ones(2))
    loss = DiffusionLoss(model)
    assert dict(loss.named_children())["model"] is model
    assert [n for n, _ in loss.named_parameters()] == ["model.diffusion_decoder.scale"]
