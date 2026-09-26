"""SyntheticTokenOffset: the smoke test's shifted twin of a source, under its own token."""

import pytest
import torch
from torch_geometric.data import Batch, HeteroData

from torchcell.transforms.coo_regression_to_classification import (
    COOLabelNormalizationTransform,
)
from torchcell.transforms.synthetic_token_offset import SyntheticTokenOffset

TYPES = ["fitness", "gene_interaction"]


def _record(values: list[float], types: list[int], tokens: list[int]) -> HeteroData:
    d = HeteroData()
    d["gene"].num_nodes = 4
    d["gene"].perturbation_indices = torch.tensor([1, 2])
    d["gene"].phenotype_types = TYPES
    d["gene"].phenotype_values = torch.tensor(values)
    d["gene"].phenotype_type_indices = torch.tensor(types)
    d["gene"].phenotype_sample_indices = torch.tensor(list(range(len(values))))
    d["gene"].phenotype_dataset_indices = torch.tensor(tokens)
    return d


def test_clones_only_the_named_labels_of_mapped_sources_under_their_twins() -> None:
    t = SyntheticTokenOffset(
        delta=0.3, token_map={0: 7, 1: 8}, labels=["gene_interaction"]
    )
    # token 3 (fitness) is not cloned: wrong label; token 2 (gi) has no twin
    out = t(_record([0.9, 0.1, -0.2, 0.5], [0, 1, 1, 1], [3, 0, 1, 2]))
    g = out["gene"]
    assert g.phenotype_values.tolist() == pytest.approx([0.9, 0.1, -0.2, 0.5, 0.4, 0.1])
    assert g.phenotype_type_indices.tolist() == [0, 1, 1, 1, 1, 1]
    assert g.phenotype_dataset_indices.tolist() == [3, 0, 1, 2, 7, 8]
    assert g.phenotype_sample_indices.tolist() == [0, 1, 2, 3, 1, 2]
    assert g.phenotype_synthetic.tolist() == [False] * 4 + [True] * 2
    assert t.twin_of(torch.tensor([0, 1, 2, 9])).tolist() == [7, 8, -1, -1]


def test_control_keeps_the_original_token_and_nan_rows_are_not_cloned() -> None:
    t = SyntheticTokenOffset(
        delta=0.3, token_map={0: 7}, labels=["gene_interaction"], control=True
    )
    out = t(_record([0.9, float("nan")], [0, 1], [3, 0]))
    assert out["gene"].phenotype_values.shape == (2,)
    assert out["gene"].phenotype_synthetic.tolist() == [False, False]
    out = t(_record([0.9, 0.1], [0, 1], [3, 0]))
    assert out["gene"].phenotype_dataset_indices.tolist() == [3, 0, 0]


def test_original_scale_clone_uses_the_normalizer() -> None:
    stats = {
        "gene_interaction": {
            "mean": 0.0,
            "std": 0.05,
            "min": -1.0,
            "max": 1.0,
            "q25": -0.02,
            "q75": 0.02,
        }
    }
    norm = COOLabelNormalizationTransform(
        None,  # type: ignore[arg-type]
        {"gene_interaction": {"strategy": "standard"}},
        fit_stats=stats,
    )
    t = SyntheticTokenOffset(
        delta=0.3, token_map={0: 7}, labels=["gene_interaction"], normalizer=norm
    )
    d = norm(_record([0.9, 0.1], [0, 1], [3, 0]))
    out = t(d)
    g = out["gene"]
    assert g.phenotype_values[-1].item() == pytest.approx(0.1 / (0.05 + 1e-8) + 0.3)
    assert g.phenotype_values_original[-1].item() == pytest.approx(
        0.1 + 0.3 * 0.05, rel=1e-4
    )
    # follow_batch gives one batch row per value, clones included
    b = Batch.from_data_list([out, out], follow_batch=["phenotype_values"])
    assert b["gene"].phenotype_values_batch.tolist() == [0, 0, 0, 1, 1, 1]


def test_bad_maps_and_fit_stats_are_rejected() -> None:
    with pytest.raises(ValueError, match="cannot also be a source"):
        SyntheticTokenOffset(delta=0.3, token_map={0: 1, 1: 2}, labels=["fitness"])
    with pytest.raises(ValueError, match="no entry for label"):
        COOLabelNormalizationTransform(
            None,  # type: ignore[arg-type]
            {"fitness": {"strategy": "standard"}},
            fit_stats={"gene_interaction": {}},
        )
    with pytest.raises(ValueError, match="one of fit_table"):
        COOLabelNormalizationTransform(
            None,  # type: ignore[arg-type]
            {"fitness": {"strategy": "standard"}},
            fit_indices=[0],
            fit_stats={"fitness": {}},
        )
