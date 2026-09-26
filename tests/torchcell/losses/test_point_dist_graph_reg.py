# tests/torchcell/losses/test_point_dist_graph_reg.py
# [[tests.torchcell.losses.test_point_dist_graph_reg]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/losses/test_point_dist_graph_reg.py
"""``PointDistGraphReg`` composition on a three-sample batch with the distribution term off.

Predictions [1, 2, 3] against targets [1, 1, 1]: log-cosh mean is 0.5862611926 (see
``test_logcosh.py``), MSE is (0 + 1 + 4) / 3 = 5/3. With ``lambda_point`` 2.0 and a graph
regularization value 0.4 at ``lambda`` 0.5, the total is 2 * 0.5862611926 + 0.5 * 0.4 =
1.3725223852. The distribution loss stays off (``lambda`` 0) so the value is a hand
computation; its buffered variants are exercised in ``test_mle_dist_supcr.py``.
"""

import pytest
import torch

from torchcell.losses.point_dist_graph_reg import PointDistGraphReg

PREDICTIONS = torch.tensor([[1.0], [2.0], [3.0]])
TARGETS = torch.tensor([[1.0], [1.0], [1.0]])
LOGCOSH_MEAN = 0.5862611926
MSE_MEAN = 5.0 / 3.0


def _loss(point_type: str = "logcosh", lambda_point: float = 2.0) -> PointDistGraphReg:
    return PointDistGraphReg(
        point_estimator={"type": point_type, "lambda": lambda_point},
        distribution_loss={"type": "dist", "lambda": 0.0},
        graph_regularization={"lambda": 0.5},
        ddp={"use_ddp_gather": False},
    )


def test_logcosh_point_plus_graph_regularization() -> None:
    """Total = 2 * 0.5862611926 + 0.5 * 0.4 = 1.3725223852, every part reported."""
    total, parts = _loss()(PREDICTIONS, TARGETS, {"graph_reg_loss": torch.tensor(0.4)})
    assert total.item() == pytest.approx(1.3725223852, abs=1e-6)
    assert parts["point_loss"] == pytest.approx(LOGCOSH_MEAN, abs=1e-6)
    assert parts["weighted_point"] == pytest.approx(2 * LOGCOSH_MEAN, abs=1e-6)
    assert parts["dist_loss"] == 0.0 and parts["weighted_dist"] == 0.0
    assert parts["graph_reg_loss"] == pytest.approx(0.4, abs=1e-6)
    assert parts["weighted_graph_reg"] == pytest.approx(0.2, abs=1e-6)
    assert parts["total_loss"] == pytest.approx(1.3725223852, abs=1e-6)
    # normalized shares: weighted point / total, and the unweighted point / (point + reg)
    assert parts["norm_weighted_point"] == pytest.approx(
        2 * LOGCOSH_MEAN / 1.3725223852, abs=1e-6
    )
    assert parts["norm_unweighted_point"] == pytest.approx(
        LOGCOSH_MEAN / (LOGCOSH_MEAN + 0.4), abs=1e-6
    )
    assert parts["norm_weighted_graph_reg"] == pytest.approx(
        0.2 / 1.3725223852, abs=1e-6
    )


def test_mse_point_estimator() -> None:
    """The mse branch uses WeightedMSELoss: (0 + 1 + 4) / 3 = 5/3, times lambda 2."""
    total, parts = _loss("mse")(PREDICTIONS, TARGETS, {})
    assert parts["point_loss"] == pytest.approx(MSE_MEAN, abs=1e-6)
    assert total.item() == pytest.approx(2 * MSE_MEAN, abs=1e-6)
    assert parts["graph_reg_loss"] == 0.0


def test_missing_graph_reg_key_adds_nothing() -> None:
    """Without "graph_reg_loss" in representations the total is the weighted point loss."""
    total, parts = _loss()(PREDICTIONS, TARGETS, {})
    assert total.item() == pytest.approx(2 * LOGCOSH_MEAN, abs=1e-6)
    assert parts["weighted_graph_reg"] == 0.0


def test_forward_count_increments_per_call() -> None:
    """The registered counter tracks calls (used for DDP gather intervals)."""
    loss = _loss()
    assert loss.forward_count.item() == 0
    loss(PREDICTIONS, TARGETS, {})
    loss(PREDICTIONS, TARGETS, {})
    assert loss.forward_count.item() == 2


def test_gradient_reaches_the_predictions() -> None:
    """D total / d p = lambda_point * tanh(p - t) / 3 for the log-cosh branch."""
    predictions = PREDICTIONS.clone().requires_grad_(True)
    total, _ = _loss()(predictions, TARGETS, {})
    total.backward()
    assert predictions.grad is not None
    expected = 2 * torch.tensor([[0.0], [0.7615941560], [0.9640275801]]) / 3
    torch.testing.assert_close(predictions.grad, expected, atol=1e-6, rtol=0)


def test_flat_keyword_configuration_matches_the_nested_form() -> None:
    """The deprecated flat kwargs configure the same components."""
    flat = PointDistGraphReg(
        point_loss_type="logcosh",
        lambda_point=2.0,
        lambda_dist=0.0,
        lambda_graph_reg=0.5,
        use_ddp_gather=False,
    )
    total, _ = flat(PREDICTIONS, TARGETS, {"graph_reg_loss": torch.tensor(0.4)})
    assert total.item() == pytest.approx(1.3725223852, abs=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"point_estimator": {"type": "huber"}}, "Unknown point_loss_type: huber"),
        (
            {
                "distribution_loss": {"type": "kl", "lambda": 0.1},
                "ddp": {"use_ddp_gather": False},
            },
            "Unknown dist_loss_type: kl",
        ),
    ],
)
def test_unknown_component_types_raise(kwargs: dict[str, object], message: str) -> None:
    """A misspelled component type fails at construction, naming the value."""
    with pytest.raises(ValueError, match=message):
        PointDistGraphReg(**kwargs)  # type: ignore[arg-type]
