# tests/torchcell/loader/test_dense_padding_data_loader.py
# [[tests.torchcell.loader.test_dense_padding_data_loader]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/loader/test_dense_padding_data_loader.py
"""Dense padding collation of two hand-written ``HeteroData`` graphs.

Graph 1 has three ``gene`` nodes with ``x = [[1], [2], [3]]``, ``flag = [T, F, T]``,
``ids = [10, 11, 12]`` and two edges ``0 -> 1, 1 -> 2``; graph 2 has two nodes with
``x = [[4], [5]]``, ``flag = [F, T]``, ``ids = [20, 21]`` and one edge ``0 -> 1``. Both carry a
graph-level scalar ``y`` (1.0 / 2.0), a Python int ``label`` (3 / 4) and a string ``name``.

Padding to the longer graph appends one row to graph 2. Floats pad with
``FLOAT_PADDING_VALUE = 1e-5`` (not zero), integers with ``-1``; bool tensors go through a
long ``-1`` pad, get their mask from it, and come back as bool with ``False`` in the pad.
``edge_index`` pads along its edge dimension, so graph 2's ``[[0], [1]]`` becomes
``[[0, -1], [1, -1]]``. Every mask is ``value != padding`` and lands in ``mask_dict``
under the store key (``"gene"``, the edge-type tuple, or the attribute name itself for
the global store). ``num_nodes`` sums to 5 with the per-graph list ``[3, 2]`` kept.

2026.09.30, Phase 16: the recursion, key-name, storage and dataset branches. A dict or
list of int tensors ``[1, 2]`` and ``[3]`` collated with ``non_float_padding_value=-7``
pads the bare tensors with -7 but the nested ones with the module default -1, because
the recursive calls (``dense_padding_data_loader.py:190`` and ``:204``) do not forward the
padding arguments. A key containing ``edge_index`` (so ``hyperedge_index`` too) pads
``[[0, 1], [2, 3]]`` and ``[[4], [5]]`` along the edge axis to ``[[4, -1], [5, -1]]``. A
float64 tensor takes the non-float32 branch yet still pads with 1e-5 and stays float64.
A stored value equal to the pad (an id of -1, a feature of 1e-5) is masked False like
the pad itself. The shared-memory worker path, the two TensorFrame branches (torch_frame
is not installed, so ``torch_geometric.typing.torch_frame`` is ``object``) and the
``OnDiskDataset`` indirection (two ``Data`` graphs with ``x = [[1], [2]]`` and ``[[3]]``,
written to ``tmp_path``) are pinned against the dense batch they produce.
"""

from collections import namedtuple
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.utils.data
import torch_geometric.typing as pyg_typing
from torch_geometric.data import Data, HeteroData, OnDiskDataset
from torch_geometric.typing import TensorFrame

from torchcell.loader import dense_padding_data_loader as dpl
from torchcell.loader.dense_padding_data_loader import (
    FLOAT_PADDING_VALUE,
    NON_FLOAT_PADDING_VALUE,
    DensePaddingCollater,
    DensePaddingDataLoader,
    dense_padded_collate,
    dense_padded_from_data_list,
)

EDGE = ("gene", "interacts", "gene")


def _graph(
    xs: list[list[float]],
    flags: list[bool],
    ids: list[int],
    edge_index: list[list[int]],
    y: float,
    label: int,
    name: str,
) -> HeteroData:
    data = HeteroData()
    data["gene"].x = torch.tensor(xs)
    data["gene"].flag = torch.tensor(flags)
    data["gene"].ids = torch.tensor(ids)
    data["gene"].num_nodes = len(xs)
    data["gene"].extra = {"a": torch.tensor(ids)}
    data["gene"].seq = [torch.tensor(ids), torch.tensor(flags)]
    data[EDGE].edge_index = torch.tensor(edge_index)
    data.y = torch.tensor(y)
    data.label = label
    data.name = name
    return data


def _pair() -> list[HeteroData]:
    return [
        _graph(
            [[1.0], [2.0], [3.0]],
            [True, False, True],
            [10, 11, 12],
            [[0, 1], [1, 2]],
            1.0,
            3,
            "a",
        ),
        _graph([[4.0], [5.0]], [False, True], [20, 21], [[0], [1]], 2.0, 4, "b"),
    ]


def test_float_attributes_pad_with_1e_minus_5_and_mask_the_pad() -> None:
    """Finding: the float pad is ``1e-5``, not zero (``dense_padding_data_loader.py:19``).

    ``x`` becomes ``[2, 3, 1]`` with graph 2's third row equal to the padding constant, and
    the mask is ``value != 1e-5``: all True except that row.
    """
    batch = dense_padded_from_data_list(_pair())
    assert FLOAT_PADDING_VALUE == 1e-5
    expected = torch.tensor(
        [[[1.0], [2.0], [3.0]], [[4.0], [5.0], [FLOAT_PADDING_VALUE]]]
    )
    torch.testing.assert_close(batch["gene"].x, expected)
    assert batch.mask_dict["gene"]["x"].tolist() == [
        [[True], [True], [True]],
        [[True], [True], [False]],
    ]


def test_integer_bool_and_uint8_attributes_pad_with_minus_one_then_zero() -> None:
    """``ids`` keeps ``-1`` in the pad; ``flag`` is masked from the ``-1`` pad and returned as bool False."""
    batch = dense_padded_from_data_list(_pair())
    assert NON_FLOAT_PADDING_VALUE == -1
    assert batch["gene"].ids.tolist() == [[10, 11, 12], [20, 21, -1]]
    assert batch.mask_dict["gene"]["ids"].tolist() == [
        [True, True, True],
        [True, True, False],
    ]
    assert batch["gene"].flag.dtype == torch.bool
    assert batch["gene"].flag.tolist() == [[True, False, True], [False, True, False]]
    assert batch.mask_dict["gene"]["flag"].tolist() == [
        [True, True, True],
        [True, True, False],
    ]

    u1, u2 = HeteroData(), HeteroData()
    u1["gene"].u = torch.tensor([1, 2], dtype=torch.uint8)
    u2["gene"].u = torch.tensor([3], dtype=torch.uint8)
    padded = dense_padded_from_data_list([u1, u2])
    assert padded["gene"].u.dtype == torch.uint8
    assert padded["gene"].u.tolist() == [[1, 2], [3, 0]]
    assert padded.mask_dict["gene"]["u"].tolist() == [[True, True], [True, False]]


def test_edge_index_pads_along_the_edge_dimension() -> None:
    """``[2, E]`` per graph becomes ``[2, 2, 2]`` with ``-1`` columns for the missing edge."""
    batch = dense_padded_from_data_list(_pair())
    assert batch[EDGE].edge_index.tolist() == [[[0, 1], [1, 2]], [[0, -1], [1, -1]]]
    assert batch.mask_dict[EDGE]["edge_index"].tolist() == [
        [[True, True], [True, True]],
        [[True, False], [True, False]],
    ]


def test_num_nodes_sum_num_graphs_and_no_batch_vectors() -> None:
    """Finding: ``follow_batch`` is accepted and ignored; no ``batch``, ``ptr`` or ``x_batch`` exists.

    ``dense_padded_collate`` never reads ``follow_batch`` (``dense_padding_data_loader.py:219``),
    so unlike PyG's sparse collate the dense batch carries only the padded tensors. What it
    does keep: ``num_nodes`` summed to 5, the per-graph list ``[3, 2]`` and ``num_graphs``.
    """
    batch = dense_padded_from_data_list(_pair(), follow_batch=["x"])
    assert batch.num_graphs == 2
    assert batch["gene"].num_nodes == 5
    assert batch["gene"]._num_nodes == [3, 2]
    assert not hasattr(batch["gene"], "batch")
    assert not hasattr(batch["gene"], "ptr")
    assert not hasattr(batch["gene"], "x_batch")


def test_scalars_numbers_strings_mappings_and_sequences_collate_by_type() -> None:
    """Global scalar ``y`` stacks with a mask under its own name; ints become a tensor with
    no mask; strings stay a list; a dict and a list of tensors recurse element-wise.
    """
    batch = dense_padded_from_data_list(_pair())
    assert batch.y.tolist() == [1.0, 2.0]
    assert batch.mask_dict["y"].tolist() == [True, True]
    assert batch.label.tolist() == [3, 4]
    assert "label" not in batch.mask_dict
    assert batch.name == ["a", "b"]
    assert "name" not in batch.mask_dict
    assert batch["gene"].extra["a"].tolist() == [[10, 11, 12], [20, 21, -1]]
    assert batch.mask_dict["gene"]["extra"]["a"].tolist() == [
        [True, True, True],
        [True, True, False],
    ]
    seq = batch["gene"].seq
    assert seq[0].tolist() == [[10, 11, 12], [20, 21, -1]]
    assert seq[1].tolist() == [[True, False, True], [False, True, False]]
    assert [m.tolist() for m in batch.mask_dict["gene"]["seq"]] == [
        [[True, True, True], [True, True, False]],
        [[True, True, True], [True, True, False]],
    ]


def test_exclude_keys_drops_the_attribute_and_its_mask() -> None:
    """``ids`` is absent from the store and from ``mask_dict``; ``x`` is kept unchanged,
    padded with the module's float pad value 1e-5 in the missing third node of graph 1.
    """
    batch = dense_padded_from_data_list(_pair(), exclude_keys=["ids"])
    assert not hasattr(batch["gene"], "ids")
    assert "ids" not in batch.mask_dict["gene"]
    torch.testing.assert_close(
        batch["gene"].x,
        torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [1e-5]]]),
        atol=0,
        rtol=0,
    )


def test_collate_accepts_a_generator_the_data_class_itself_and_skips_ptr() -> None:
    """A generator is materialized; collating into ``HeteroData`` gives a plain
    ``HeteroData``; a ``ptr`` attribute is dropped rather than padded.
    """
    pair = _pair()
    for data in pair:
        data["gene"].ptr = torch.tensor([0, 1])
    # The signature says list, but dense_padded_collate materializes any iterable.
    generator = cast(list[Any], (data for data in pair))
    out, masks = dense_padded_collate(HeteroData, generator)
    assert type(out) is HeteroData
    assert not hasattr(out["gene"], "ptr")
    assert "ptr" not in masks["gene"]
    assert out["gene"].ids.tolist() == [[10, 11, 12], [20, 21, -1]]
    assert out["gene"].num_nodes == 5
    assert masks["gene"]["ids"].tolist() == [[True, True, True], [True, True, False]]


def test_worker_process_branch_writes_the_same_batch_into_shared_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a worker info present the values are concatenated into a pre-allocated shared
    tensor; every attribute equals the main-process result exactly.
    """
    plain = dense_padded_from_data_list(_pair())
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: object())
    shared = dense_padded_from_data_list(_pair())
    torch.testing.assert_close(shared["gene"].x, plain["gene"].x)
    assert torch.equal(shared["gene"].flag, plain["gene"].flag)
    assert torch.equal(shared["gene"].ids, plain["gene"].ids)
    assert torch.equal(shared[EDGE].edge_index, plain[EDGE].edge_index)
    assert torch.equal(shared.y, plain.y)


def test_sparse_and_nested_tensor_attributes_are_rejected() -> None:
    """Finding: the nested-tensor ``NotImplementedError`` is unreachable.

    A sparse attribute raises the module's own ``NotImplementedError``. A nested tensor is
    handed to ``_dense_pad_tensor`` BEFORE the ``is_nested`` check
    (``dense_padding_data_loader.py:124-134``), and ``pad_sequence`` fails on it with a
    torch ``RuntimeError`` first.
    """
    s1, s2 = HeteroData(), HeteroData()
    s1["gene"].adj = torch.eye(2).to_sparse()
    s2["gene"].adj = torch.eye(2).to_sparse()
    with pytest.raises(NotImplementedError, match="SparseTensors is not supported"):
        dense_padded_from_data_list([s1, s2])
    n1, n2 = HeteroData(), HeteroData()
    n1["gene"].x = torch.nested.nested_tensor(
        [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
    )
    n2["gene"].x = torch.nested.nested_tensor(
        [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
    )
    with pytest.raises(RuntimeError, match="NestedTensorImpl doesn't support sizes"):
        dense_padded_from_data_list([n1, n2])


def test_collater_dispatches_plain_python_batches_by_element_type() -> None:
    """Tensors stack, floats and ints become tensors, strings stay, dicts and tuples recurse."""
    collate = DensePaddingCollater(_pair())
    assert collate([torch.tensor([1, 2]), torch.tensor([3, 4])]).tolist() == [
        [1, 2],
        [3, 4],
    ]
    floats = collate([1.5, 2.5])
    assert floats.dtype == torch.float
    assert floats.tolist() == [1.5, 2.5]
    ints = collate([1, 2])
    assert ints.dtype == torch.int64
    assert ints.tolist() == [1, 2]
    assert collate(["a", "b"]) == ["a", "b"]
    assert collate([{"k": 1}, {"k": 2}])["k"].tolist() == [1, 2]
    a, b = collate([(1, 2.0), (3, 4.0)])
    assert a.tolist() == [1, 3]
    assert b.tolist() == [2.0, 4.0]
    Pair = namedtuple("Pair", "a b")
    named = collate([Pair(1, 2.0), Pair(3, 4.0)])
    assert isinstance(named, Pair)
    assert named.a.tolist() == [1, 3]
    with pytest.raises(
        TypeError, match="DataLoader found invalid type: '<class 'object'>'"
    ):
        collate([object(), object()])


def test_data_loader_yields_one_dense_batch_and_drops_a_passed_collate_fn() -> None:
    """A ``collate_fn`` kwarg is discarded in favor of the collater; batch 2 gives one batch."""
    loader = DensePaddingDataLoader(
        _pair(), batch_size=2, collate_fn="ignored", follow_batch=["x"]
    )
    assert loader.follow_batch == ["x"]
    assert loader.collator.follow_batch == ["x"]
    assert loader.collate_fn == loader.collator.collate_fn
    batches: list[Any] = list(loader)
    assert len(batches) == 1
    assert batches[0].num_graphs == 2
    torch.testing.assert_close(
        batches[0]["gene"].x,
        torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [FLOAT_PADDING_VALUE]]]),
    )


def test_nested_mappings_and_lists_ignore_a_custom_integer_pad() -> None:
    """Finding: the recursion drops the caller's padding values.

    A bare tensor honors ``non_float_padding_value=-7``; the same tensors inside a dict or
    a list are padded with the default -1 because ``_dense_padded_collate`` recurses without
    forwarding ``float_padding_value`` / ``non_float_padding_value``
    (``dense_padding_data_loader.py:190-192`` and ``:204-206``). Pinned until the
    recursive calls pass them on.
    """
    ones, three = torch.tensor([1, 2]), torch.tensor([3])
    bare, bare_mask = dpl._dense_padded_collate(
        "k", [ones, three], [], [], non_float_padding_value=-7
    )
    assert bare.tolist() == [[1, 2], [3, -7]]
    assert bare_mask.tolist() == [[True, True], [True, False]]
    nested, nested_mask = dpl._dense_padded_collate(
        "k", [{"a": ones}, {"a": three}], [], [], non_float_padding_value=-7
    )
    assert nested["a"].tolist() == [[1, 2], [3, -1]]
    assert nested_mask["a"].tolist() == [[True, True], [True, False]]
    listed, _ = dpl._dense_padded_collate(
        "k", [[ones], [three]], [], [], non_float_padding_value=-7
    )
    assert listed[0].tolist() == [[1, 2], [3, -1]]


def test_any_key_containing_edge_index_pads_along_the_edge_axis() -> None:
    """``hyperedge_index`` matches the substring test: ``[2, E]`` pads columns, not rows."""
    value, mask = dpl._dense_padded_collate(
        "hyperedge_index",
        [torch.tensor([[0, 1], [2, 3]]), torch.tensor([[4], [5]])],
        [],
        [],
    )
    assert value.tolist() == [[[0, 1], [2, 3]], [[4, -1], [5, -1]]]
    assert mask.tolist() == [
        [[True, True], [True, True]],
        [[True, False], [True, False]],
    ]


def test_float64_keeps_its_dtype_and_pads_with_the_float_value() -> None:
    """float64 is not ``torch.float`` so it takes the integer-style branch, but the pad
    is still chosen by ``is_floating_point``: 1e-5, in float64.
    """
    value, mask = dpl._dense_padded_collate(
        "x",
        [
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            torch.tensor([3.0], dtype=torch.float64),
        ],
        [],
        [],
    )
    assert value.dtype == torch.float64
    assert value.tolist() == [[1.0, 2.0], [3.0, 1e-5]]
    assert mask.tolist() == [[True, True], [True, False]]


def test_a_real_value_equal_to_the_pad_is_masked_as_padding() -> None:
    """Finding: the mask is ``value != pad`` (``dense_padding_data_loader.py:161``).

    A genuine id of -1 and a genuine feature of exactly 1e-5 are indistinguishable from
    padding, so both are masked False in an otherwise full row. Pinned until the mask is
    built from the per-graph lengths instead of from the values.
    """
    a, b = HeteroData(), HeteroData()
    a["gene"].ids = torch.tensor([-1, 5])
    b["gene"].ids = torch.tensor([7, 8])
    a["gene"].x = torch.tensor([[1e-5], [2.0]])
    b["gene"].x = torch.tensor([[3.0], [4.0]])
    batch = dense_padded_from_data_list([a, b])
    assert batch.mask_dict["gene"]["ids"].tolist() == [[False, True], [True, True]]
    assert batch.mask_dict["gene"]["x"].tolist() == [
        [[False], [True]],
        [[True], [True]],
    ]


def test_worker_branch_without_pt20_reads_a_flag_torch_geometric_lacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding: the pre-2.0 fallback reads ``torch_geometric.typing.WITH_PT112``.

    The installed torch_geometric defines ``WITH_PT113`` and no ``WITH_PT112``, so in a
    worker with ``WITH_PT20`` false the collate raises ``AttributeError``
    (``dense_padding_data_loader.py:151``). Pinned until the branch is removed or reads
    a flag that exists.
    """
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: object())
    monkeypatch.setattr(pyg_typing, "WITH_PT20", False)
    with pytest.raises(AttributeError, match="has no attribute 'WITH_PT112'"):
        dense_padded_from_data_list(_pair())


@pytest.mark.parametrize("with_pt112", [True, False])
def test_legacy_shared_storage_paths_give_the_same_batch(
    monkeypatch: pytest.MonkeyPatch, with_pt112: bool
) -> None:
    """With the pre-2.0 flags forced, both typed-storage allocations produce exactly the
    main-process batch: ``x`` padded with 1e-5 and ``ids`` with -1.
    """
    plain = dense_padded_from_data_list(_pair())
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: object())
    monkeypatch.setattr(pyg_typing, "WITH_PT20", False)
    monkeypatch.setattr(pyg_typing, "WITH_PT112", with_pt112, raising=False)
    shared = dense_padded_from_data_list(_pair())
    torch.testing.assert_close(shared["gene"].x, plain["gene"].x, atol=0, rtol=0)
    assert shared["gene"].ids.tolist() == [[10, 11, 12], [20, 21, -1]]
    assert torch.equal(shared[EDGE].edge_index, plain[EDGE].edge_index)


def test_tensor_frames_are_refused_by_collate_and_break_the_collater() -> None:
    """The attribute collate raises its own ``NotImplementedError``.

    Finding: the collater's TensorFrame branch calls ``torch_frame.cat``, and without
    torch_frame installed ``torch_frame`` is the placeholder ``object``, so a batch of
    frames raises ``AttributeError`` (``dense_padding_data_loader.py:373``). Pinned until
    that branch refuses the way the attribute collate does.
    """
    with pytest.raises(
        NotImplementedError,
        match=r"^Dense padding collation for TensorFrames is not supported \(tested\) yet\.$",
    ):
        dpl._dense_padded_collate("t", [TensorFrame(), TensorFrame()], [], [])
    with pytest.raises(
        AttributeError, match="type object 'object' has no attribute 'cat'"
    ):
        DensePaddingCollater([])([TensorFrame()])


def test_on_disk_dataset_is_loaded_by_index_through_multi_get(tmp_path: Path) -> None:
    """The loader iterates ``range(len(dataset))`` and the collater fetches the graphs
    with ``multi_get``; the dense batch is ``[[[1], [2]], [[3], [1e-5]]]``.
    """
    dataset = OnDiskDataset(str(tmp_path / "disk"), log=False)
    dataset.extend(
        [Data(x=torch.tensor([[1.0], [2.0]])), Data(x=torch.tensor([[3.0]]))]
    )
    loader = DensePaddingDataLoader(dataset, batch_size=2)
    indices: object = loader.dataset
    assert indices == range(2)
    batches: list[Any] = list(loader)
    assert len(batches) == 1
    torch.testing.assert_close(
        batches[0].x,
        torch.tensor([[[1.0], [2.0]], [[3.0], [FLOAT_PADDING_VALUE]]]),
        atol=0,
        rtol=0,
    )
    direct = loader.collator.collate_fn([1])
    torch.testing.assert_close(direct.x, torch.tensor([[[3.0]]]), atol=0, rtol=0)
    dataset.close()
