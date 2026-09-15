# tests/torchcell/metabolism/test_extract_targets_width_collision.py

"""Tests for 026's ``extract_targets`` width-1 collision guard.

The claim under test: ``extract_targets`` separates two heads sharing a phenotype label by
the WIDTH of their value group, because the phenotype dict keys are dropped in collation.
Two width-1 groups under one label on one genotype therefore both match a width-1 head and
would be averaged into one target. The guard refuses that batch and names the group sizes;
everything else about the decode is unchanged, which the first two tests pin.
"""

import importlib
import os
import os.path as osp
import sys
from typing import Any

import pytest
import torch
from dotenv import load_dotenv

load_dotenv()
if "DATA_ROOT" not in os.environ:
    pytest.skip(
        "DATA_ROOT is not set; train_flux.py reads it at import",
        allow_module_level=True,
    )

REPO = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
os.environ["EXPERIMENT_ROOT"] = osp.join(REPO, "experiments")
sys.path.insert(0, osp.join(REPO, "experiments", "026-metabolism-flux", "scripts"))
train_flux: Any = importlib.import_module("train_flux")


class _Genes:
    """The four COO tensors plus the per-graph label names ``extract_targets`` reads."""

    def __init__(
        self,
        values: list[float],
        type_indices: list[int],
        values_batch: list[int],
        sample_indices: list[int],
        types: list[list[str]],
    ) -> None:
        self.phenotype_values = torch.tensor(values, dtype=torch.float)
        self.phenotype_type_indices = torch.tensor(type_indices, dtype=torch.long)
        self.phenotype_values_batch = torch.tensor(values_batch, dtype=torch.long)
        self.phenotype_sample_indices = torch.tensor(sample_indices, dtype=torch.long)
        self._types = types

    def __getitem__(self, key: str) -> list[list[str]]:
        assert key == "phenotype_types"
        return self._types


class _Batch:
    def __init__(self, genes: _Genes, num_graphs: int) -> None:
        self._genes = genes
        self.num_graphs = num_graphs

    def __getitem__(self, key: str) -> _Genes:
        assert key == "gene"
        return self._genes


def _two_groups(sizes: tuple[int, ...]) -> _Batch:
    """One genotype carrying ``len(sizes)`` ``metabolite_level`` groups of those widths."""
    values: list[float] = []
    samples: list[int] = []
    for sample, size in enumerate(sizes):
        values.extend(float(sample + 1) for _ in range(size))
        samples.extend([sample] * size)
    genes = _Genes(
        values=values,
        type_indices=[0] * len(values),
        values_batch=[0] * len(values),
        sample_indices=samples,
        types=[["metabolite_level"]],
    )
    return _Batch(genes, num_graphs=1)


def test_one_width_one_group_beside_a_wide_one_decodes_as_before():
    batch = _two_groups((1, 19))
    out = train_flux.extract_targets(
        batch,
        {
            "betaxanthin": {"label": "metabolite_level", "width": 1},
            "mulleder19": {"label": "metabolite_level", "width": 19},
        },
    )
    bx, bx_mask = out["betaxanthin"]
    aa, aa_mask = out["mulleder19"]
    assert bool(bx_mask.all()) and bool(aa_mask.all())
    assert float(bx[0, 0]) == 1.0
    assert torch.equal(aa[0], torch.full((19,), 2.0))


def test_a_head_whose_width_no_group_matches_is_masked_false():
    batch = _two_groups((19,))
    out = train_flux.extract_targets(
        batch, {"betaxanthin": {"label": "metabolite_level", "width": 1}}
    )
    _, mask = out["betaxanthin"]
    assert not bool(mask.any())


def test_two_width_one_groups_under_one_label_are_refused_not_averaged():
    batch = _two_groups((1, 1))
    with pytest.raises(AssertionError, match="width-1 'metabolite_level' groups"):
        train_flux.extract_targets(
            batch, {"betaxanthin": {"label": "metabolite_level", "width": 1}}
        )


def test_a_wide_head_reads_past_a_width_one_collision_it_does_not_consume():
    """The guard is scoped to the head that would do the averaging, and to width 1.

    ``fig6_build`` carries exactly one colliding genotype, record 3363, where two Cachera
    strains aggregate to one deletion key and contribute two width-1 betaxanthin values.
    The isobutanol head reads a width-20 column there and averages nothing, so refusing it
    would make a correct run impossible for no benefit.
    """
    batch = _two_groups((1, 1, 19))
    out = train_flux.extract_targets(
        batch, {"mulleder19": {"label": "metabolite_level", "width": 19}}
    )
    target, mask = out["mulleder19"]
    assert bool(mask.all())
    assert torch.equal(target[0], torch.full((19,), 3.0))


def test_two_groups_of_the_same_wider_width_are_averaged_as_replicates():
    """Averaging same-width wide groups is the documented behavior and is left alone."""
    batch = _two_groups((19, 19))
    out = train_flux.extract_targets(
        batch, {"mulleder19": {"label": "metabolite_level", "width": 19}}
    )
    target, _ = out["mulleder19"]
    assert torch.equal(target[0], torch.full((19,), 1.5))


def test_the_guard_is_per_label_and_ignores_a_second_label():
    """Two width-1 groups under DIFFERENT labels are not a collision; nothing is averaged."""
    genes = _Genes(
        values=[1.0, 2.0],
        type_indices=[0, 1],
        values_batch=[0, 0],
        sample_indices=[0, 1],
        types=[["metabolite_level", "visual_score"]],
    )
    batch = _Batch(genes, num_graphs=1)
    out = train_flux.extract_targets(
        batch,
        {
            "betaxanthin": {"label": "metabolite_level", "width": 1},
            "carotenoid": {"label": "visual_score", "width": 1},
        },
    )
    assert float(out["betaxanthin"][0][0, 0]) == 1.0
    assert float(out["carotenoid"][0][0, 0]) == 2.0
