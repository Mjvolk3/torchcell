# tests/torchcell/models/test_gpu_edge_mask_generator.py
# [[tests.torchcell.models.test_gpu_edge_mask_generator]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_gpu_edge_mask_generator.py
"""`GPUEdgeMaskGenerator` on CPU (the class takes any `torch.device`; nothing in it
requires CUDA), against hand-derived masks and a plain-loop oracle written here.

Cell graph: five genes, three edge types.

* physical: edge_index [[0,1,3,0,1,2,3,4],[1,2,4,0,1,2,3,4]] (three edges then one self
  loop per node, the `to_cell_data` layout), 8 edges.
* regulatory: [[2,4,0,1,2,3,4],[0,2,0,1,2,3,4]], 7 edges.
* gpr (gene -> reaction, `hyperedge_index` only): not gene-gene, so the generator
  ignores it.

Incidence (edge positions touching each gene; a self loop is listed once):
physical 0:[0,3] 1:[0,1,4] 2:[1,5] 3:[2,6] 4:[2,7];
regulatory 0:[0,2] 1:[3] 2:[0,1,4] 3:[5] 4:[1,6].
Padded to width 3 with -1. Elements: physical 2+3+2+2+2 = 11, regulatory 2+1+3+1+2 = 9.

Masks (an edge is kept iff neither endpoint is perturbed):

* {1}: physical [F,F,T,T,F,T,T,T], regulatory [T,T,T,F,T,T,T]
* {0,3}: physical [F,T,F,F,T,T,F,T], regulatory [F,T,F,T,T,F,T]
* {4}: physical [T,T,F,T,T,T,T,F], regulatory [T,F,T,T,T,T,F]
"""

import re

import pytest
import torch
from torch_geometric.data import HeteroData

from torchcell.models.gpu_edge_mask_generator import GPUEdgeMaskGenerator

PHYS = ("gene", "physical_interaction", "gene")
REG = ("gene", "regulatory_interaction", "gene")
GPR = ("gene", "gpr", "reaction")
CPU = torch.device("cpu")

T, F = True, False
HAND = {
    (1,): {PHYS: [F, F, T, T, F, T, T, T], REG: [T, T, T, F, T, T, T]},
    (0, 3): {PHYS: [F, T, F, F, T, T, F, T], REG: [F, T, F, T, T, F, T]},
    (4,): {PHYS: [T, T, F, T, T, T, T, F], REG: [T, F, T, T, T, T, F]},
}


def _cell_graph() -> HeteroData:
    cg = HeteroData()
    cg["gene"].num_nodes = 5
    cg[PHYS].edge_index = torch.tensor(
        [[0, 1, 3, 0, 1, 2, 3, 4], [1, 2, 4, 0, 1, 2, 3, 4]]
    )
    cg[REG].edge_index = torch.tensor([[2, 4, 0, 1, 2, 3, 4], [0, 2, 0, 1, 2, 3, 4]])
    cg[GPR].hyperedge_index = torch.tensor([[0, 1, 3], [0, 0, 1]])
    return cg


@pytest.fixture
def generator() -> GPUEdgeMaskGenerator:
    """The generator on CPU over the five-gene graph."""
    return GPUEdgeMaskGenerator(_cell_graph(), CPU)


def _oracle(edge_index: torch.Tensor, perturbed: list[int]) -> list[bool]:
    """Straightforward loop: keep edge e iff neither endpoint is perturbed."""
    gone = set(perturbed)
    src, dst = edge_index.tolist()
    return [s not in gone and d not in gone for s, d in zip(src, dst, strict=True)]


def _as_lists(
    masks: dict[tuple[str, str, str], torch.Tensor],
) -> dict[tuple[str, str, str], list[bool]]:
    for mask in masks.values():
        assert mask.dtype == torch.bool
        assert mask.device == CPU
    return {k: v.tolist() for k, v in masks.items()}


def test_incidence_cache_lists_each_edge_once_per_endpoint(
    generator: GPUEdgeMaskGenerator,
) -> None:
    """The docstring incidence lists; the self loop (i, i) appears once under i; gpr has
    no entry.
    """
    cache = {et: [t.tolist() for t in v] for et, v in generator.incidence_cache.items()}
    assert cache == {
        PHYS: [[0, 3], [0, 1, 4], [1, 5], [2, 6], [2, 7]],
        REG: [[0, 2], [3], [0, 1, 4], [5], [1, 6]],
    }


def test_padded_incidence_tensors_and_validity(generator: GPUEdgeMaskGenerator) -> None:
    """Width = longest list (3); padding is -1 and is exactly the invalid positions."""
    assert generator.incidence_tensors[PHYS].tolist() == [
        [0, 3, -1],
        [0, 1, 4],
        [1, 5, -1],
        [2, 6, -1],
        [2, 7, -1],
    ]
    assert generator.incidence_tensors[REG].tolist() == [
        [0, 2, -1],
        [3, -1, -1],
        [0, 1, 4],
        [5, -1, -1],
        [1, 6, -1],
    ]
    for et in [PHYS, REG]:
        assert (
            generator.incidence_masks[et].tolist()
            == (generator.incidence_tensors[et] >= 0).tolist()
        )


def test_buffers_are_exactly_the_all_true_base_masks(
    generator: GPUEdgeMaskGenerator,
) -> None:
    """Two registered buffers, all True, of length 8 and 7; nothing else in the state
    dict (the incidence tensors are plain attributes).
    """
    assert {n: b.tolist() for n, b in generator.named_buffers()} == {
        "base_mask_gene__physical_interaction__gene": [T] * 8,
        "base_mask_gene__regulatory_interaction__gene": [T] * 7,
    }
    assert list(generator.state_dict()) == [
        "base_mask_gene__physical_interaction__gene",
        "base_mask_gene__regulatory_interaction__gene",
    ]


@pytest.mark.parametrize("perturbed", list(HAND))
def test_single_mask_equals_the_hand_derived_set(
    generator: GPUEdgeMaskGenerator, perturbed: tuple[int, ...]
) -> None:
    """One genotype at a time; also equal to the loop oracle; base buffers untouched."""
    masks = _as_lists(generator.generate_single_mask(torch.tensor(list(perturbed))))
    assert masks == HAND[perturbed]
    cg = _cell_graph()
    assert masks == {
        et: _oracle(cg[et].edge_index, list(perturbed)) for et in [PHYS, REG]
    }
    base = generator.get_buffer("base_mask_gene__physical_interaction__gene")
    assert base.tolist() == [T] * 8


@pytest.mark.parametrize(
    "method", ["generate_batch_masks", "generate_batch_masks_vectorized"]
)
def test_batch_is_the_per_sample_masks_concatenated_in_order(
    generator: GPUEdgeMaskGenerator, method: str
) -> None:
    """Batch [{1}, {0,3}, {4}]: sample k occupies positions k*E .. k*E+E-1, so the
    output is the three hand masks concatenated (24 physical, 21 regulatory). A wrong
    per-sample offset would move a False into the neighboring sample's block.
    """
    batch = [torch.tensor(list(p)) for p in HAND]
    masks = _as_lists(getattr(generator, method)(batch, 3))
    assert masks == {et: [v for p in HAND for v in HAND[p][et]] for et in [PHYS, REG]}


def test_vectorized_equals_loop_and_oracle_on_random_batches(
    generator: GPUEdgeMaskGenerator,
) -> None:
    """60 seeded batches of 1..4 samples, 0..3 perturbed genes each (duplicates allowed,
    empty samples allowed): both batch paths equal the concatenated oracle exactly.
    """
    cg = _cell_graph()
    gen = torch.Generator().manual_seed(0)
    for _ in range(60):
        size = int(torch.randint(1, 5, (1,), generator=gen).item())
        batch = [
            torch.randint(
                0,
                5,
                (int(torch.randint(0, 4, (1,), generator=gen).item()),),
                generator=gen,
            )
            for _ in range(size)
        ]
        expected = {
            et: [v for p in batch for v in _oracle(cg[et].edge_index, p.tolist())]
            for et in [PHYS, REG]
        }
        assert _as_lists(generator.generate_batch_masks(batch, size)) == expected
        assert (
            _as_lists(generator.generate_batch_masks_vectorized(batch, size))
            == expected
        )


def test_duplicate_perturbation_is_idempotent(generator: GPUEdgeMaskGenerator) -> None:
    """[1, 1] masks exactly what [1] masks, in all three paths."""
    expected = HAND[(1,)]
    assert _as_lists(generator.generate_single_mask(torch.tensor([1, 1]))) == expected
    assert (
        _as_lists(generator.generate_batch_masks([torch.tensor([1, 1])], 1)) == expected
    )
    assert (
        _as_lists(generator.generate_batch_masks_vectorized([torch.tensor([1, 1])], 1))
        == expected
    )


def test_out_of_range_indices_are_refused_by_both_batch_paths(
    generator: GPUEdgeMaskGenerator,
) -> None:
    """Loop path names the first bad index and the sample; vectorized names all bad
    indices of the batch in order; both report the first edge type checked.
    """
    with pytest.raises(
        IndexError,
        match="^"
        + re.escape(
            "Perturbation index 5 out of bounds [0, 4]. Edge type: "
            "('gene', 'physical_interaction', 'gene'), Sample perturbation indices: [5]"
        )
        + "$",
    ):
        generator.generate_batch_masks([torch.tensor([5])], 1)
    with pytest.raises(
        IndexError,
        match="^"
        + re.escape(
            "Perturbation index -1 out of bounds [0, 4]. Edge type: "
            "('gene', 'physical_interaction', 'gene'), Sample perturbation indices: [-1, 2]"
        )
        + "$",
    ):
        generator.generate_batch_masks([torch.tensor([-1, 2])], 1)
    with pytest.raises(
        IndexError,
        match="^"
        + re.escape(
            "Perturbation indices [5, -1] out of bounds [0, 4]. Edge type: "
            "('gene', 'physical_interaction', 'gene')"
        )
        + "$",
    ):
        generator.generate_batch_masks_vectorized(
            [torch.tensor([5, -1]), torch.tensor([2])], 2
        )
    # a negative index alone must be refused too, not wrapped to gene 4's row
    with pytest.raises(
        IndexError,
        match="^"
        + re.escape(
            "Perturbation indices [-1] out of bounds [0, 4]. Edge type: "
            "('gene', 'physical_interaction', 'gene')"
        )
        + "$",
    ):
        generator.generate_batch_masks_vectorized(
            [torch.tensor([2]), torch.tensor([-1])], 2
        )


def test_single_mask_has_no_bounds_check(generator: GPUEdgeMaskGenerator) -> None:
    """Finding: `generate_single_mask` indexes the Python incidence list directly
    (`gpu_edge_mask_generator.py:420`) with no bounds check, so index -1 silently masks
    gene 4's edges (identical to [4]) and index 5 raises a bare list IndexError, while
    both batch paths refuse -1 and 5 with a message. Latent: only the 006 `test_*.py`
    diagnostic scripts use the generator; trainer use was removed in 53c257c22. Pinned
    until the single-sample path applies the same bounds check.
    """
    assert _as_lists(generator.generate_single_mask(torch.tensor([-1]))) == HAND[(4,)]
    with pytest.raises(IndexError, match="^list index out of range$"):
        generator.generate_single_mask(torch.tensor([5]))


def test_batch_size_argument_is_ignored_by_the_loop_path(
    generator: GPUEdgeMaskGenerator,
) -> None:
    """Finding: `generate_batch_masks` never reads `batch_size` (one block per list
    element), but the vectorized path uses it: with no perturbations it returns
    `batch_size` blocks regardless of the list (`gpu_edge_mask_generator.py:295`), and
    with perturbations a mismatch fails inside torch (`repeat_interleave`, line 316; the
    message is torch-owned and pinned at torch's current wording). Latent: only the 006
    `test_*.py` diagnostic scripts use the generator; trainer use was removed in
    53c257c22. Pinned until both paths check `batch_size ==
    len(batch_perturbation_indices)`.
    """
    one = [torch.tensor([1])]
    assert _as_lists(generator.generate_batch_masks(one, 3)) == HAND[(1,)]
    empty = [torch.tensor([], dtype=torch.long)]
    assert _as_lists(generator.generate_batch_masks_vectorized(empty, 3)) == {
        PHYS: [T] * 24,
        REG: [T] * 21,
    }
    assert _as_lists(generator.generate_batch_masks(empty, 3)) == {
        PHYS: [T] * 8,
        REG: [T] * 7,
    }
    with pytest.raises(
        RuntimeError,
        match="^"
        + re.escape(
            "repeats must have the same size as input along dim, but got "
            "repeats.size(0) = 2 and input.size(0) = 1"
        )
        + "$",
    ):
        generator.generate_batch_masks_vectorized(
            [torch.tensor([1]), torch.tensor([2])], 1
        )


def test_empty_batch_differs_between_paths(generator: GPUEdgeMaskGenerator) -> None:
    """Finding: an empty batch returns {} from the vectorized path
    (`gpu_edge_mask_generator.py:276-277`) but raises from the loop path's
    `torch.cat([])` (line 253). Latent: only the 006 `test_*.py` diagnostic scripts use
    the generator; trainer use was removed in 53c257c22. Pinned until the loop path
    handles an empty batch.
    """
    assert generator.generate_batch_masks_vectorized([], 0) == {}
    with pytest.raises(
        ValueError, match=re.escape("torch.cat(): expected a non-empty list of Tensors")
    ):
        generator.generate_batch_masks([], 0)


def test_module_to_moves_base_masks_but_not_the_incidence(
    generator: GPUEdgeMaskGenerator,
) -> None:
    """Finding: only the base masks are buffers, so `.to(device)` moves them while
    `self.device`, `incidence_cache` and `incidence_tensors` stay on the construction
    device (`gpu_edge_mask_generator.py:116,165`; shown with the meta device on CPU).
    The class docstring says the incidence cache lives on the GPU; it lives wherever the
    generator was built. Latent: only the 006 `test_*.py` diagnostic scripts use the
    generator; trainer use was removed in 53c257c22. Pinned until the incidence tensors
    are registered as buffers.
    """
    moved = generator.to("meta")
    assert moved.base_mask_gene__physical_interaction__gene.device.type == "meta"
    assert moved.device == CPU
    assert moved.incidence_tensors[PHYS].device == CPU
    assert moved.incidence_masks[REG].device == CPU
    assert moved.incidence_cache[PHYS][0].device == CPU


def test_edge_type_with_no_edges() -> None:
    """A gene-gene type with zero edges: zero-width incidence, empty masks in every
    path, and the other type is unaffected.
    """
    cg = HeteroData()
    cg["gene"].num_nodes = 3
    empty = ("gene", "empty", "gene")
    cg[empty].edge_index = torch.zeros((2, 0), dtype=torch.long)
    cg[PHYS].edge_index = torch.tensor([[0, 1], [1, 2]])
    gen = GPUEdgeMaskGenerator(cg, CPU)
    assert tuple(gen.incidence_tensors[empty].shape) == (3, 0)
    assert tuple(gen.incidence_tensors[PHYS].shape) == (3, 2)
    expected = {empty: [], PHYS: [F, F]}
    assert _as_lists(gen.generate_single_mask(torch.tensor([1]))) == expected
    assert _as_lists(gen.generate_batch_masks([torch.tensor([1])], 1)) == expected
    assert _as_lists(
        gen.generate_batch_masks_vectorized([torch.tensor([1]), torch.tensor([0])], 2)
    ) == {empty: [], PHYS: [F, F, F, T]}


def test_memory_usage_arithmetic(generator: GPUEdgeMaskGenerator) -> None:
    """Cache: 11 + 9 = 20 int64 = 160 B; masks: 8 + 7 = 15 bool = 15 B; in MiB."""
    usage = generator.get_memory_usage()
    assert usage == {
        "incidence_cache_mb": 160 / 1024**2,
        "base_masks_mb": 15 / 1024**2,
        "total_mb": 160 / 1024**2 + 15 / 1024**2,
    }
