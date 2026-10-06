# tests/torchcell/losses/test_losses_dango.py
# [[tests.torchcell.losses.test_losses_dango]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/losses/test_losses_dango.py
"""``torchcell.losses.dango``: the flipped schedule, refusals and three edge behaviors.

Named ``test_losses_dango.py`` because ``tests/torchcell/models/test_dango.py`` holds the
basename. The trainer tests (tests/torchcell/trainers/test_int_dango.py) already pin
PreThenPost, LinearUntilUniform and the reconstruction closed form inside a training
step; this file covers what they do not.

Fixture: one edge type "string" with lambda 0.1, a reconstruction [[1, 0], [0, 0.5]]
against adjacency [[1, 0], [1, 0]] (squared errors 0, 0, 1, 0.25; nonzero targets at
(0, 0) and (1, 0)): weighted MSE = (0 + 1 + 0.1 * (0 + 0.25)) / 4 = 0.25625.
Interaction predictions [0.5, -1] against [0, 0]: log-cosh mean =
(log cosh 0.5 + log cosh 1) / 2.
LinearUntilFlipped(T): alpha = 1 - e / T below T, then 0; the total is
alpha * recon + (1 - alpha) * interaction.
"""

import math
import re

import pytest
import torch

from torchcell.losses.dango import (
    SCHEDULER_MAP,
    DangoLoss,
    DangoLossSched,
    LinearUntilFlipped,
    LinearUntilUniform,
    PreThenPost,
)

RECON = {"string": torch.tensor([[1.0, 0.0], [0.0, 0.5]])}
ADJ = {"string": torch.tensor([[1.0, 0.0], [1.0, 0.0]])}
PRED = torch.tensor([0.5, -1.0])
TARGET = torch.zeros(2)
RECON_LOSS = (1.0 + 0.1 * 0.25) / 4
INTERACTION = (math.log(math.cosh(0.5)) + math.log(math.cosh(1.0))) / 2


@pytest.mark.parametrize(
    ("epoch", "alpha"),
    [(0, 1.0), (5, 0.75), (10, 0.5), (19, 0.05), (20, 0.0), (35, 0.0)],
)
def test_linear_until_flipped_alpha_and_components(epoch: int, alpha: float) -> None:
    """T = 20: alpha = 1 - e / 20 (5 -> 0.75, 19 -> 0.05), then 0 from epoch 20 on."""
    recon, inter = torch.tensor(2.0), torch.tensor(6.0)
    total, a, w_recon, w_inter = LinearUntilFlipped(transition_epoch=20).forward(
        recon, inter, epoch
    )
    assert a == pytest.approx(alpha)
    assert w_recon.item() == pytest.approx(alpha * 2.0)
    assert w_inter.item() == pytest.approx((1 - alpha) * 6.0)
    assert total.item() == pytest.approx(alpha * 2.0 + (1 - alpha) * 6.0)


def test_scheduler_map_names_each_class() -> None:
    """The config-name map holds exactly the three schedules."""
    assert SCHEDULER_MAP == {
        "PreThenPost": PreThenPost,
        "LinearUntilUniform": LinearUntilUniform,
        "LinearUntilFlipped": LinearUntilFlipped,
    }
    assert LinearUntilFlipped().transition_epoch == 20


def test_dango_loss_with_flipped_schedule_closed_form() -> None:
    """Epoch 5 of LinearUntilFlipped(10): alpha 0.5, total = 0.5 * 0.25625 + 0.5 * I."""
    loss = DangoLoss(["string"], {"string": 0.1}, LinearUntilFlipped(10))
    total, parts = loss(PRED, TARGET, RECON, ADJ, current_epoch=5)
    assert parts["reconstruction_loss"].item() == pytest.approx(RECON_LOSS)
    assert parts["interaction_loss"].item() == pytest.approx(INTERACTION, abs=1e-7)
    assert parts["alpha"].item() == 0.5
    assert total.item() == pytest.approx(0.5 * RECON_LOSS + 0.5 * INTERACTION, abs=1e-7)


def test_abstract_scheduler_and_refusals() -> None:
    """The ABC cannot be built; a bad reduction or a non-scheduler raise exactly."""
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Can't instantiate abstract class DangoLossSched without an implementation "
            "for abstract method 'forward'"
        ),
    ):
        DangoLossSched()  # type: ignore[abstract, unused-ignore]
    with pytest.raises(ValueError, match=re.escape("Invalid reduction mode: avg")):
        DangoLoss(["string"], {}, reduction="avg")
    with pytest.raises(
        TypeError, match=re.escape("scheduler must be an instance of DangoLossSched")
    ):
        DangoLoss(["string"], {}, scheduler="PreThenPost")  # type: ignore[arg-type, unused-ignore]
    default = DangoLoss(["string"], {}).scheduler
    assert isinstance(default, PreThenPost)
    assert default.transition_epoch == 10


@pytest.mark.parametrize("reduction", ["none", "sum"])
def test_reduction_is_validated_and_then_ignored(reduction: str) -> None:
    """Finding: ``reduction`` is accepted but never read.

    ``__init__`` validates and stores it (losses/dango.py:201-207), yet both terms always
    reduce by the mean (lines 239-242 and 299): 'none' and 'sum' give the same scalar
    total and components as 'mean'. Reach: latent; the 005/006 dango.py scripts
    hard-code reduction "mean". Pinned until the reduction is applied or the argument
    is removed.
    """
    mean = DangoLoss(["string"], {"string": 0.1}, LinearUntilUniform(10))
    other = DangoLoss(["string"], {"string": 0.1}, LinearUntilUniform(10), reduction)
    total_mean, parts_mean = mean(PRED, TARGET, RECON, ADJ, 4)
    total_other, parts_other = other(PRED, TARGET, RECON, ADJ, 4)
    assert other.reduction == reduction
    assert total_other.dim() == 0
    assert torch.equal(total_other, total_mean)
    for key, value in parts_mean.items():
        assert torch.equal(parts_other[key], value), key


def test_no_matching_edge_type_crashes_on_the_float_zero() -> None:
    """Finding: with no edge type in both dicts the loss raises AttributeError.

    ``compute_reconstruction_loss`` starts at the float 0.0 and returns it untouched
    when nothing matches (losses/dango.py:262-284); ``forward`` then reads
    ``recon_loss.device`` (line 341). The return is typed Tensor and the comment calls
    the path unused. Pinned until the empty case returns a tensor or raises a clear error.
    """
    loss = DangoLoss(["string"], {"string": 0.1})
    assert loss.compute_reconstruction_loss({"other": RECON["string"]}, ADJ) == 0.0
    with pytest.raises(
        AttributeError, match=re.escape("'float' object has no attribute 'device'")
    ):
        loss(PRED, TARGET, {"other": RECON["string"]}, ADJ)


def test_networks_are_averaged_and_default_lambda_is_one() -> None:
    """Two networks: "string" (lambda 0.1, 0.25625) and "go" (no lambda -> 1.0,
    errors 0, 1, 1, 0.25 -> (1 + 1 * (1 + 0.25)) / 4 = 0.5625); the mean is 0.409375.
    """
    loss = DangoLoss(["string", "go"], {"string": 0.1})
    recon = {**RECON, "go": torch.tensor([[1.0, 1.0], [0.0, 0.5]])}
    adj = {**ADJ, "go": torch.tensor([[1.0, 0.0], [1.0, 0.0]])}
    value = loss.compute_reconstruction_loss(recon, adj)
    assert isinstance(value, torch.Tensor)
    assert value.item() == pytest.approx((RECON_LOSS + 0.5625) / 2)


def test_log_cosh_overflows_for_large_residuals() -> None:
    """Finding (latent): ``log(cosh(r))`` overflows in float32 for |r| above about 89.

    At r = 80 it equals r - log 2 = 79.306853 (the stable asymptote), but at r = 100
    ``cosh`` is inf, the interaction loss is inf and its gradient is NaN
    (losses/dango.py:299). Gene-interaction residuals are far smaller, so reported runs
    are not affected; a stable form is |r| + softplus(-2|r|) - log 2. Pinned until the
    stable form is used.
    """
    loss = DangoLoss(["string"], {})
    big = loss.compute_interaction_loss(torch.tensor([80.0]), torch.zeros(1))
    assert big.item() == pytest.approx(80.0 - math.log(2.0), abs=1e-4)
    pred = torch.tensor([100.0, 0.0], requires_grad=True)
    inf = loss.compute_interaction_loss(pred, torch.zeros(2))
    assert math.isinf(inf.item())
    inf.backward()
    assert pred.grad is not None
    assert math.isnan(pred.grad[0].item())
    assert pred.grad[1].item() == 0.0
