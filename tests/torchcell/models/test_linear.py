# tests/torchcell/models/test_linear.py
# [[tests.torchcell.models.test_linear]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_linear.py
"""``SimpleLinearModel.forward``: scatter aggregation then one Linear, hand-set weights.

2026.10.06, Phase 21. Fixture: ``SimpleLinearModel(2, 1, scatter)`` with weight
``[[1, 2]]`` and bias 0. Nodes ``x = [[1, 2], [3, 4], [5, 6]]`` with ``batch = [0, 0, 1]``.

* add: set 0 = ``[1 + 3, 2 + 4] = [4, 6]``, set 1 = ``[5, 6]``; outputs
  ``4 + 12 = 16`` and ``5 + 12 = 17``.
* mean: set 0 = ``[2, 3]``, set 1 = ``[5, 6]``; outputs ``2 + 6 = 8`` and 17.

``main`` is a print-only demo and is left uncovered.
"""

import re

import pytest
import torch

from torchcell.models.linear import SimpleLinearModel

X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
BATCH = torch.tensor([0, 0, 1])


def _model(scatter: str) -> SimpleLinearModel:
    model = SimpleLinearModel(2, 1, scatter=scatter)
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[1.0, 2.0]]))
        model.linear.bias.zero_()
    return model


@pytest.mark.parametrize(
    ("scatter", "expected"), [("add", [[16.0], [17.0]]), ("mean", [[8.0], [17.0]])]
)
def test_forward_aggregates_per_set_then_projects(
    scatter: str, expected: list[list[float]]
) -> None:
    """One output row per set, values from the module docstring."""
    out = _model(scatter)(X, BATCH)
    torch.testing.assert_close(out, torch.tensor(expected), rtol=0.0, atol=0.0)


def test_default_scatter_is_add() -> None:
    """With no ``scatter`` argument the model sums each set.

    Weight ``[[1, 2]]``, bias 0 on the module fixture: set 0 sums to ``[4, 6]``, giving
    ``4 + 2 * 6 = 16``; set 1 is ``[5, 6]``, giving ``5 + 12 = 17``. A mean default
    would give 8 for set 0, a max default would raise.
    """
    model = SimpleLinearModel(2, 1)
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[1.0, 2.0]]))
        model.linear.bias.zero_()
    torch.testing.assert_close(
        model(X, BATCH), torch.tensor([[16.0], [17.0]]), rtol=0.0, atol=0.0
    )


def test_max_scatter_crashes_on_the_argmax_tuple() -> None:
    """Finding: ``scatter="max"`` cannot run.

    ``torch_scatter.scatter_max`` returns ``(values, argmax)``; linear.py:37 assigns
    the tuple to ``x`` and passes it to ``nn.Linear``, which refuses it. Pinned until
    the branch keeps ``scatter_max(...)[0]``.
    """
    message = "linear(): argument 'input' (position 1) must be Tensor, not tuple"
    with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
        _model("max")(X, BATCH)


def test_unknown_scatter_skips_aggregation_silently() -> None:
    """Finding: an unrecognized mode (``"sum"``) is not refused; nodes are not pooled.

    No branch matches, so the Linear runs per node and the output has one row per
    NODE: ``[1 + 4, 3 + 8, 5 + 12] = [5, 11, 17]``, shape ``[3, 1]`` instead of
    ``[2, 1]``. Pinned until the constructor validates ``scatter``.
    """
    out = _model("sum")(X, BATCH)
    torch.testing.assert_close(
        out, torch.tensor([[5.0], [11.0], [17.0]]), rtol=0.0, atol=0.0
    )
