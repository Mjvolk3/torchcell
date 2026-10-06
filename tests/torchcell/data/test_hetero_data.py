# tests/torchcell/data/test_hetero_data.py
# [[tests.torchcell.data.test_hetero_data]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_hetero_data.py
"""``custom_size_repr`` branch by branch and the patched ``HeteroData.__repr__``.

2026.10.06, Phase 21. Every expectation is a literal string written by hand from the
source rules: a 0-dim tensor prints its item, other tensors and arrays their size as a
list, a nested tensor the size of its zero-padded form, a string in single quotes,
any other sequence its length as ``[n]``, an empty mapping ``{}``, a one-item mapping
whose value is not a mapping inline as ``{ k=v }``, any other mapping one entry per
line indented by ``indent + 2`` with a trailing comma and a closing brace at
``indent``; a ``dict`` holding a list or tuple value, or more than four items,
collapses to ``dict(len=n)``. Keys lose their single quotes. ``torch_sparse`` and
``torch_frame`` are not installed, so the ``SparseTensor`` / ``TensorFrame`` branches
are driven with stand-ins patched over the module's names.
"""

import numpy as np
import pytest
import torch

import torchcell.data.hetero_data as hd
from torchcell.data.hetero_data import HeteroData, custom_size_repr, hetero_repr


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (torch.tensor(2.5), "k=2.5"),
        (torch.zeros(3, 2), "k=[3, 2]"),
        (np.zeros((4, 1, 2)), "k=[4, 1, 2]"),
        ("abc", "k='abc'"),
        ([1, 2, 3], "k=[3]"),
        ((1, 2), "k=[2]"),
        ({}, "k={}"),
        ({"a": torch.zeros(5)}, "k={ a=[5] }"),
        (7, "k=7"),
        (None, "k=None"),
    ],
    ids=[
        "scalar",
        "tensor",
        "ndarray",
        "str",
        "list",
        "tuple",
        "empty",
        "one",
        "int",
        "none",
    ],
)
def test_leaf_and_small_mapping_branches(value: object, expected: str) -> None:
    """Each leaf rule from the module docstring, at indent 0."""
    assert custom_size_repr("k", value) == expected


def test_nested_tensor_reports_its_padded_size() -> None:
    """Rows of 2 and 4 by 3 pad to ``[2, 4, 3]``."""
    nested = torch.nested.nested_tensor([torch.zeros(2, 3), torch.zeros(4, 3)])
    assert custom_size_repr("n", nested) == "n=[2, 4, 3]"


def test_multi_item_mapping_is_multiline_with_indent() -> None:
    """At indent 2: entries at 4 spaces, each with a trailing comma, brace at 2.

    The inner ``{"b": tensor}`` is a one-item non-mapping value, so it stays inline.
    """
    value = {"a": {"b": torch.zeros(1)}, "c": 1}
    assert (
        custom_size_repr("m", value, indent=2)
        == "  m={\n    a={ b=[1] },\n    c=1,\n  }"
    )


def test_one_item_mapping_of_a_mapping_is_multiline() -> None:
    """A single item whose value is itself a mapping does not inline."""
    assert custom_size_repr("m", {"a": {"b": 1}}) == "m={\n  a={ b=1 },\n}"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"a": [1, 2], "b": 3}, "  d=dict(len=2)"),
        ({"a": (1,)}, "  d=dict(len=1)"),
        ({str(i): i for i in range(5)}, "  d=dict(len=5)"),
    ],
)
def test_dict_with_sequences_or_many_items_collapses(
    value: dict[str, object], expected: str
) -> None:
    """A ``dict`` with a list/tuple value or more than four items prints its length."""
    assert custom_size_repr("d", value, indent=2) == expected


def test_four_item_dict_is_expanded() -> None:
    """Exactly four scalar items is under the cutoff and expands."""
    value = {"a": 1, "b": 2, "c": 3, "d": 4}
    assert custom_size_repr("d", value) == "d={\n  a=1,\n  b=2,\n  c=3,\n  d=4,\n}"


def test_collapsed_dict_keeps_tuple_key_quotes() -> None:
    """Finding: the ``dict(len=n)`` early return skips the key quote stripping.

    hetero_data.py:26/28 format the raw key, while every other path goes through
    ``str(key).replace("'", "")`` (line 63). An edge-type tuple key therefore prints
    as ``('a', 'b')=dict(len=1)`` when collapsed but ``(a, b)=...`` otherwise. Pinned
    until the early returns use the stripped key.
    """
    key = ("a", "b")
    assert custom_size_repr(key, {"x": [1]}) == "('a', 'b')=dict(len=1)"
    assert custom_size_repr(key, 1) == "(a, b)=1"


def test_sparse_tensor_and_tensor_frame_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``sizes()`` minus its closing bracket plus ``nnz``; ``Class([rows, cols])``."""

    class FakeSparse:
        def sizes(self) -> list[int]:
            return [3, 4]

        def nnz(self) -> int:
            return 5

    class FakeFrame:
        num_rows = 6
        num_cols = 2

    monkeypatch.setattr(hd, "SparseTensor", FakeSparse)
    monkeypatch.setattr(hd, "TensorFrame", FakeFrame)
    assert custom_size_repr("adj", FakeSparse()) == "adj=[3, 4, nnz=5]"
    assert custom_size_repr("tf", FakeFrame()) == "tf=FakeFrame([6, 2])"


def test_hetero_repr_lists_global_then_node_then_edge_stores() -> None:
    """Global attributes, then node stores, then edge stores, each at indent 2.

    The patch is installed at import: ``HeteroData.__repr__`` is ``hetero_repr``.
    """
    data = HeteroData()
    data.ids = ["a", "b"]
    data.meta = {"k": torch.zeros(2)}
    data["gene"].x = torch.zeros(3, 2)
    data["gene"].y = torch.zeros(3)
    data["gene", "r", "gene"].edge_index = torch.zeros(2, 5, dtype=torch.long)
    assert HeteroData.__repr__ is hetero_repr
    assert repr(data) == (
        "HeteroData(\n"
        "  ids=[2],\n"
        "  meta={ k=[2] },\n"
        "  gene={\n    x=[3, 2],\n    y=[3],\n  },\n"
        "  (gene, r, gene)={ edge_index=[2, 5] }\n"
        ")"
    )


def test_empty_hetero_data_repr() -> None:
    """No stores: no inner newlines."""
    assert repr(HeteroData()) == "HeteroData()"
