# tests/torchcell/models/test_mlp.py
# [[tests.torchcell.models.test_mlp]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_mlp.py
"""``Mlp``: the exact layer stack for every option, refusals, and a closed-form forward.

2026.10.06, Phase 21. Layer stacks are compared as nested lists of
``(type name, in, out)`` descriptors read off ``model.model``; parameter counts are
worked by hand (``Linear(a, b)`` has ``a * b + b``; ``BatchNorm1d`` / ``LayerNorm`` /
affine ``InstanceNorm1d`` of width ``w`` have ``2 * w``).

Closed-form forward: ``Mlp(2, 2, 1, num_layers=2, activation="relu")`` with
``W1 = [[1, -1], [2, 0]]``, ``b1 = [0, -1]``, ``W2 = [[1, 1]]``, ``b2 = 0.5``.
For ``x = [1, 2]``: ``W1 x + b1 = [-1, 1]``, relu ``[0, 1]``, output 1.5. For
``x = [3, 1]``: ``[2, 5]``, relu ``[2, 5]``, output 7.5. ``forward`` squeezes the
trailing size-1 axis, so the result is ``[1.5, 7.5]`` of shape ``[2]``.
"""

import re
from typing import Any

import pytest
import torch
from torch import nn

from torchcell.models import act_register
from torchcell.models.mlp import Mlp


def _describe(module: nn.Module) -> Any:
    """A comparable descriptor: Sequentials recurse, leaves give (type, dims)."""
    if isinstance(module, nn.Sequential):
        return [_describe(m) for m in module]
    if isinstance(module, nn.Linear):
        return ("Linear", module.in_features, module.out_features)
    if isinstance(module, (nn.BatchNorm1d, nn.InstanceNorm1d)):
        return (type(module).__name__, module.num_features, module.affine)
    if isinstance(module, nn.LayerNorm):
        return ("LayerNorm", module.normalized_shape)
    if isinstance(module, nn.Dropout):
        return ("Dropout", module.p)
    return type(module).__name__


def _layer(model: Mlp, *path: int) -> nn.Module:
    """``model.model[path[0]][path[1]]...`` with the ``Sequential`` checks mypy needs."""
    module: nn.Module = model.model
    for i in path:
        assert isinstance(module, nn.Sequential)
        module = module[i]
    return module


def _linear(model: Mlp, block: int) -> nn.Linear:
    """The ``Linear`` opening block ``block``."""
    layer = _layer(model, block, 0)
    assert isinstance(layer, nn.Linear)
    return layer


def _n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_three_layers_with_batch_norm_and_activations() -> None:
    """Norm + activation on every hidden block, a bare final Linear, then dropout, then
    the output activation.

    ``Mlp(4, 3, 2, 3, 0.25, "batch", "relu", "sigmoid")``: parameters
    ``(4*3+3) + 2*3 + (3*3+3) + 2*3 + (3*2+2) = 15 + 6 + 12 + 6 + 8 = 47``. The relu
    module is the registry's single instance, shared by both hidden blocks.
    """
    model = Mlp(
        4, 3, 2, 3, 0.25, norm="batch", activation="relu", output_activation="sigmoid"
    )
    assert _describe(model.model) == [
        [("Linear", 4, 3), ("BatchNorm1d", 3, True), "ReLU"],
        [("Linear", 3, 3), ("BatchNorm1d", 3, True), "ReLU"],
        [("Linear", 3, 2)],
        ("Dropout", 0.25),
        "Sigmoid",
    ]
    assert _n_params(model) == 47
    assert _layer(model, 0, 2) is act_register["relu"]
    assert _layer(model, 1, 2) is act_register["relu"]


@pytest.mark.parametrize(
    ("norm", "descriptor", "norm_params"),
    [
        ("instance", ("InstanceNorm1d", 5, True), 10),
        ("layer", ("LayerNorm", (5,)), 10),
        (None, None, 0),
    ],
)
def test_norm_choice_places_one_norm_per_hidden_block(
    norm: str | None, descriptor: Any, norm_params: int
) -> None:
    """``Mlp(3, 5, 1, 2, norm=...)``: one norm after the first Linear, none after the last.

    Instance norm is built ``affine=True`` (2 * 5 = 10 parameters), layer norm has
    10, no norm has 0; the Linears contribute ``(3*5+5) + (5*1+1) = 26``.
    """
    model = Mlp(3, 5, 1, 2, norm=norm, activation="tanh")
    first = [("Linear", 3, 5)] + ([descriptor] if descriptor else []) + ["Tanh"]
    assert _describe(model.model) == [first, [("Linear", 5, 1)], ("Dropout", 0.0)]
    assert _n_params(model) == 26 + norm_params


def test_single_layer_maps_in_to_out_with_norm_and_activation_and_no_dropout() -> None:
    """Finding: with ``num_layers=1`` ``dropout_prob`` is silently ignored.

    The one block is ``Linear(in, out)`` followed by the hidden-layer norm and
    activation (mlp.py:62-65), and the loop breaks before any ``Dropout`` is appended,
    so ``dropout_prob=0.5`` builds no dropout at all; ``hidden_channels`` is unused.
    Pinned until a single-layer model either applies or refuses ``dropout_prob``.
    """
    model = Mlp(
        6, 999, 4, 1, 0.5, norm="layer", activation="gelu", output_activation="tanh"
    )
    assert _describe(model.model) == [
        [("Linear", 6, 4), ("LayerNorm", (4,)), "GELU"],
        "Tanh",
    ]
    assert _n_params(model) == (6 * 4 + 4) + 2 * 4


def test_zero_layers_builds_an_identity_stack() -> None:
    """``num_layers=0`` builds an empty ``Sequential``: the input comes back squeezed."""
    model = Mlp(3, 3, 3, 0)
    assert _describe(model.model) == []
    x = torch.tensor([[1.0], [2.0]])
    assert torch.equal(model(x), torch.tensor([1.0, 2.0]))


def test_dropout_acts_on_the_output_not_before_the_final_block() -> None:
    """Finding: the dropout sits AFTER the final Linear, contrary to the docstring.

    The docstring says "Dropout probability applied before the final block", but
    mlp.py:71-72 appends ``Dropout`` after the last ``Linear``, so in training mode
    the prediction itself is dropped and rescaled. With ``p = 1.0`` every training
    output is exactly 0 whatever the weights; a dropout before the final block would
    leave the bias ``b2 = 0.5``. Pinned until the dropout precedes the final Linear.
    """
    model = _closed_form_model(dropout=1.0)
    model.train()
    out = model(torch.tensor([[1.0, 2.0], [3.0, 1.0]]))
    assert torch.equal(out, torch.zeros(2))


def _closed_form_model(dropout: float = 0.0) -> Mlp:
    model = Mlp(2, 2, 1, 2, dropout_prob=dropout, activation="relu")
    with torch.no_grad():
        _linear(model, 0).weight.copy_(torch.tensor([[1.0, -1.0], [2.0, 0.0]]))
        _linear(model, 0).bias.copy_(torch.tensor([0.0, -1.0]))
        _linear(model, 1).weight.copy_(torch.tensor([[1.0, 1.0]]))
        _linear(model, 1).bias.copy_(torch.tensor([0.5]))
    return model


def test_forward_closed_form_squeezes_a_single_output() -> None:
    """Hand-set weights give ``[1.5, 7.5]`` of shape ``[2]`` (module docstring)."""
    model = _closed_form_model()
    model.eval()
    out = model(torch.tensor([[1.0, 2.0], [3.0, 1.0]]))
    assert out.shape == (2,)
    torch.testing.assert_close(out, torch.tensor([1.5, 7.5]), rtol=0.0, atol=0.0)


def test_forward_keeps_a_multi_output_axis() -> None:
    """``out_channels = 2`` is not squeezed: identity-weight Linear gives ``x`` back."""
    model = Mlp(2, 0, 2, 1)
    with torch.no_grad():
        _linear(model, 0).weight.copy_(torch.eye(2))
        _linear(model, 0).bias.zero_()
    x = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    assert torch.equal(model(x), x)


@pytest.mark.parametrize(
    ("kwargs", "exc", "message"),
    [
        ({"norm": "group"}, AssertionError, "Invalid norm type"),
        ({"activation": "swish"}, AssertionError, "Invalid activation type"),
        ({"output_activation": "swish"}, KeyError, "'swish'"),
    ],
)
def test_invalid_options_are_refused(
    kwargs: dict[str, Any], exc: type[Exception], message: str
) -> None:
    """Norm and activation are asserted against the allowed sets; the output activation
    is not validated and fails as a bare ``KeyError`` on the registry lookup.
    """
    with pytest.raises(exc, match=f"^{re.escape(message)}$"):
        Mlp(2, 2, 1, 2, **kwargs)
