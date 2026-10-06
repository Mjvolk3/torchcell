# tests/torchcell/trainers/test_coo_targets
# [[tests.torchcell.trainers.test_coo_targets]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/trainers/test_coo_targets
"""The vectorized COO target decode equals the per-graph loop it replaces."""

import pytest
import torch

from torchcell.trainers.coo_targets import decode_head_targets, decode_head_targets_loop

EXPR = "expression_log2_ratio"
PROT = "protein_abundance"
FIT = "fitness"


def _batch(
    graphs: list[list[tuple[str, int, list[float]]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[list[str]]]:
    """Collate graphs given as lists of (type name, experiment index, values)."""
    values: list[float] = []
    type_idx: list[int] = []
    val_batch: list[int] = []
    samp_idx: list[int] = []
    per_graph_types: list[list[str]] = []
    for b, experiments in enumerate(graphs):
        names: list[str] = []
        for name, samp, vals in experiments:
            if name not in names:
                names.append(name)
            values.extend(vals)
            type_idx.extend([names.index(name)] * len(vals))
            val_batch.extend([b] * len(vals))
            samp_idx.extend([samp] * len(vals))
        per_graph_types.append(names)
    return (
        torch.tensor(values, dtype=torch.float32),
        torch.tensor(type_idx, dtype=torch.long),
        torch.tensor(val_batch, dtype=torch.long),
        torch.tensor(samp_idx, dtype=torch.long),
        per_graph_types,
    )


def _vec(n: int, offset: float) -> list[float]:
    return [offset + 0.1 * i for i in range(n)]


def _mixed_graphs() -> list[list[tuple[str, int, list[float]]]]:
    """Five genotypes with the structures the 019 stores produce."""
    return [
        # expression and proteome, type names in a different local order per graph
        [(EXPR, 0, _vec(6, 1.0)), (PROT, 1, _vec(4, 9.0))],
        [(PROT, 0, _vec(4, 5.0)), (EXPR, 1, _vec(6, 2.0))],
        # proteome only
        [(PROT, 0, _vec(4, 7.0))],
        # nothing this head reads, plus two fitness replicates
        [(FIT, 0, [0.8]), (FIT, 1, [0.6])],
        # an expression group of the WRONG width beside a correct one, and fitness
        [(EXPR, 0, _vec(3, 4.0)), (EXPR, 1, _vec(6, 3.0)), (FIT, 2, [0.5])],
    ]


@pytest.mark.parametrize(
    ("names", "raw_dim", "is_scalar", "keep", "width"),
    [
        ({EXPR}, 6, False, None, 6),
        ({EXPR}, 6, False, torch.tensor([True, False, True, True, False, True]), 4),
        ({PROT}, 4, False, None, 4),
        ({FIT}, 1, True, None, 1),
        ({"absent_label"}, 6, False, None, 6),
    ],
)
def test_vectorized_decode_matches_loop(
    names: set[str],
    raw_dim: int,
    is_scalar: bool,
    keep: torch.Tensor | None,
    width: int,
) -> None:
    values, type_idx, val_batch, samp_idx, types = _batch(_mixed_graphs())
    shape = (5,) if is_scalar else (5, width)
    loop_t, loop_m = decode_head_targets_loop(
        values,
        type_idx,
        val_batch,
        samp_idx,
        types,
        names,
        raw_dim,
        is_scalar,
        keep,
        torch.zeros(shape),
    )
    fast_t, fast_m = decode_head_targets(
        values,
        type_idx,
        val_batch,
        samp_idx,
        types,
        names,
        raw_dim,
        is_scalar,
        keep,
        torch.zeros(shape),
    )
    assert torch.equal(fast_m, loop_m)
    assert torch.equal(fast_t, loop_t)


def test_expected_rows_and_values() -> None:
    """Pin the decode itself, not only agreement between the two forms."""
    values, type_idx, val_batch, samp_idx, types = _batch(_mixed_graphs())
    target, mask = decode_head_targets(
        values,
        type_idx,
        val_batch,
        samp_idx,
        types,
        {EXPR},
        6,
        False,
        None,
        torch.zeros(5, 6),
    )
    assert mask.tolist() == [True, True, False, False, True]
    assert torch.allclose(target[0], torch.tensor(_vec(6, 1.0)))
    assert torch.allclose(target[1], torch.tensor(_vec(6, 2.0)))
    assert torch.allclose(target[4], torch.tensor(_vec(6, 3.0)))
    assert torch.equal(target[2], torch.zeros(6))
    fit, fit_mask = decode_head_targets(
        values,
        type_idx,
        val_batch,
        samp_idx,
        types,
        {FIT},
        1,
        True,
        None,
        torch.zeros(5),
    )
    assert fit_mask.tolist() == [False, False, False, True, True]
    assert fit[3].item() == pytest.approx(0.7)
    assert fit[4].item() == pytest.approx(0.5)


@pytest.mark.parametrize("decode", [decode_head_targets, decode_head_targets_loop])
def test_two_groups_of_the_head_width_raise(decode) -> None:  # type: ignore[no-untyped-def]
    graphs = [[(EXPR, 0, _vec(6, 1.0)), (EXPR, 1, _vec(6, 2.0))]]
    values, type_idx, val_batch, samp_idx, types = _batch(graphs)
    with pytest.raises(ValueError, match="exactly one measurement per genotype"):
        decode(
            values,
            type_idx,
            val_batch,
            samp_idx,
            types,
            {EXPR},
            6,
            False,
            None,
            torch.zeros(1, 6),
        )


@pytest.mark.parametrize("decode", [decode_head_targets, decode_head_targets_loop])
def test_width_mismatch_raises_instead_of_broadcasting(decode) -> None:  # type: ignore[no-untyped-def]
    graphs = [[(EXPR, 0, _vec(6, 1.0))]]
    values, type_idx, val_batch, samp_idx, types = _batch(graphs)
    with pytest.raises(ValueError, match="would BROADCAST"):
        decode(
            values,
            type_idx,
            val_batch,
            samp_idx,
            types,
            {EXPR},
            6,
            False,
            None,
            torch.zeros(1, 5),
        )
