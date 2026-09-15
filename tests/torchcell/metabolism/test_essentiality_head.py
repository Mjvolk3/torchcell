# tests/torchcell/metabolism/test_essentiality_head.py

"""Tests for 027's gene-essentiality head, on a SYNTHETIC batch.

No built CellDataset carries the `is_essential` label (measured 2026-09-14: the only two
builds holding `GeneEssentialitySgdDataset` converted it to `fitness`, and their
`phenotype_label_index` keys are `fitness` and `gene_interaction` only), so the head is
exercised against a hand-built COO batch rather than against a loader. What is pinned here:

1. The decode reads the boolean label as a width-1 0/1 target, with no standardization.
2. The released FCL split is read with its label INVERSION applied, and its class counts
   are asserted against the paper's own Methods sentence.
3. AUROC behaves as the rank statistic the comparator table reports.
4. `FluxBiomassReadout` reads exactly the GEM's objective column.
"""

import importlib
import os
import os.path as osp
import sys
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from dotenv import load_dotenv

load_dotenv()
if "DATA_ROOT" not in os.environ:
    pytest.skip(
        "DATA_ROOT is not set; train_bx.py reads it at import", allow_module_level=True
    )

REPO = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
os.environ["EXPERIMENT_ROOT"] = osp.join(REPO, "experiments")
sys.path.insert(
    0, osp.join(REPO, "experiments", "027-betaxanthin-metabolic", "scripts")
)
train_bx: Any = importlib.import_module("train_bx")

B, N, D, R = 4, 5, 8, 12


class _Genes:
    def __init__(self, values: list[float], batch: list[int]) -> None:
        self.phenotype_values = torch.tensor(values, dtype=torch.float)
        self.phenotype_type_indices = torch.zeros(len(values), dtype=torch.long)
        self.phenotype_values_batch = torch.tensor(batch, dtype=torch.long)
        self.phenotype_sample_indices = torch.zeros(len(values), dtype=torch.long)
        self.perturbation_indices = torch.tensor([0, 1, 2, 3])
        self.perturbation_indices_batch = torch.tensor([0, 1, 2, 3])

    def __getitem__(self, key: str) -> list[list[str]]:
        assert key == "phenotype_types"
        return [[train_bx.ESSENTIALITY_LABEL]] * B


class _Batch:
    def __init__(self, genes: _Genes) -> None:
        self._genes = genes
        self.num_graphs = B

    def __getitem__(self, key: str) -> _Genes:
        assert key == "gene"
        return self._genes


def _essentiality_batch() -> _Batch:
    """Four genotypes: essential, nonessential, essential, nonessential."""
    return _Batch(_Genes(values=[1.0, 0.0, 1.0, 0.0], batch=[0, 1, 2, 3]))


def _spec() -> dict[str, dict[str, Any]]:
    return {
        train_bx.ESSENTIALITY_HEAD: {"label": train_bx.ESSENTIALITY_LABEL, "width": 1}
    }


def test_the_label_decodes_as_a_width_one_zero_one_target():
    out = train_bx.targets_in_space(_essentiality_batch(), _spec(), False)
    target, mask = out[train_bx.ESSENTIALITY_HEAD]
    assert target.shape == (B, 1) and bool(mask.all())
    assert target[:, 0].tolist() == [1.0, 0.0, 1.0, 0.0]
    # No transform is declared, so the value is the stored boolean and nothing rescales it.
    assert _spec()[train_bx.ESSENTIALITY_HEAD].get("transform") is None


def test_essentiality_is_a_scalar_head_and_a_binary_one():
    assert train_bx.ESSENTIALITY_HEAD in train_bx.SCALAR_HEADS
    assert train_bx.ESSENTIALITY_HEAD in train_bx.BINARY_HEADS
    assert train_bx.HEAD not in train_bx.BINARY_HEADS
    assert train_bx.AUX_HEAD not in train_bx.BINARY_HEADS
    assert train_bx.ISOBUTANOL_HEAD not in train_bx.BINARY_HEADS


def test_auroc_is_the_rank_statistic_the_comparator_reports():
    assert (
        train_bx.binary_auroc(np.array([0.1, 0.2, 0.9, 0.8]), np.array([0, 0, 1, 1]))
        == 1.0
    )
    assert (
        train_bx.binary_auroc(np.array([0.9, 0.8, 0.1, 0.2]), np.array([0, 0, 1, 1]))
        == 0.0
    )
    # Ties take mid-ranks, so a constant predictor scores exactly chance rather than 0 or 1.
    assert train_bx.binary_auroc(np.ones(4), np.array([0, 0, 1, 1])) == 0.5
    assert np.isnan(train_bx.binary_auroc(np.array([0.1, 0.2]), np.array([0, 0])))


def test_binary_block_reports_per_class_and_the_majority_rate():
    logit = np.array([2.0, -2.0, 2.0, -2.0, -1.0])
    label = np.array([1, 0, 1, 0, 1])
    out = train_bx.binary_block(logit, label)
    assert out["n"] == 5 and out["n_essential"] == 3 and out["n_nonessential"] == 2
    assert out["accuracy"] == pytest.approx(4 / 5)
    assert out["recall_essential"] == pytest.approx(2 / 3)
    assert out["recall_nonessential"] == pytest.approx(1.0)
    assert out["precision_essential"] == pytest.approx(1.0)
    assert out["majority_class_rate"] == pytest.approx(3 / 5)


def test_the_released_fcl_split_is_read_with_its_label_inversion():
    path = osp.join(os.environ["DATA_ROOT"], train_bx.FCL_ESSENTIALITY_SPLIT)
    if not osp.exists(path):
        pytest.skip("the Merzbacher FCL split is not mirrored on this machine")
    labels, report = train_bx.read_fcl_essentiality_split(path)
    # Supplementary Table S4: N=964 nonessential, N=157 essential.
    assert report["n_essential"] == 157 and report["n_nonessential"] == 964
    assert report["n_labeled"] == 1121 and report["n_test_genes"] == 223
    assert set(labels.values()) == {0, 1}


def test_the_comparators_are_the_yeast_numbers_not_the_ecoli_ones():
    c = train_bx.FclEssentialityComparators()
    assert c.fba_roc_auc == 0.69 and c.fcl_test_roc_auc == 0.73
    assert c.fcl_cv_roc_auc == 0.78 and c.fcl_cv_roc_auc_sd == 0.065
    # FBA's accuracy is BELOW the majority-class rate, which is why AUROC is the headline.
    assert c.fba_accuracy < c.majority_class_rate_test
    assert c.majority_class_rate_test == pytest.approx(192 / 223)


class _FluxBackbone(nn.Module):
    """Stands in for the metabolism transformer, emitting a flux vector directly."""

    def __init__(self, flux: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("tokens", torch.zeros(B, N, D))
        self.flux_layer: nn.Module | None = nn.Linear(1, 1)
        self._flux = flux

    @property
    def num_parameters(self) -> dict[str, int]:
        return {"total": 2}

    def forward(
        self, cell_graph: Any, batch: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return {}, {
            "H_genes_pert": self.tokens,
            "head_outputs": {},
            "graph_reg_loss": torch.zeros(()),
            "flux": {"v": self._flux},
        }


def test_the_biomass_readout_reads_exactly_the_objective_column():
    torch.manual_seed(0)
    flux = torch.randn(B, R)
    column = 7
    readout = train_bx.FluxBiomassReadout(_FluxBackbone(flux), column)
    _, reps = readout(None, _essentiality_batch())
    assert torch.equal(reps["biomass_flux"], flux[:, column : column + 1])
    out = reps["head_outputs"][train_bx.ESSENTIALITY_HEAD]
    assert out.shape == (B, 1, 1)
    # The prediction is the affine of that one column and nothing else.
    expected = readout.readout(flux[:, column : column + 1]).unsqueeze(1)
    assert torch.equal(out, expected)
    assert readout.num_parameters[f"{train_bx.ESSENTIALITY_HEAD}_head"] == 2


def test_the_biomass_reaction_is_the_gems_objective():
    assert train_bx.BIOMASS_REACTION == "r_2111"


class _GeneKeyed:
    """A batch whose genotypes delete zero, one or two genes."""

    def __init__(self) -> None:
        # rows: 0 deletes gene 1, 1 deletes gene 3, 2 deletes genes 4 and 5, 3 deletes none
        self.deletion_indices = torch.tensor([1, 3, 4, 5])
        self.deletion_indices_batch = torch.tensor([0, 1, 2, 2])
        self.perturbation_indices = torch.tensor([1, 3, 4, 5])
        self.perturbation_indices_batch = torch.tensor([0, 1, 2, 2])


class _GeneKeyedBatch:
    def __init__(self, n: int = 4, device: str = "cpu") -> None:
        self._gene = _GeneKeyed()
        if device != "cpu":
            for name in (
                "deletion_indices",
                "deletion_indices_batch",
                "perturbation_indices",
                "perturbation_indices_batch",
            ):
                setattr(self._gene, name, getattr(self._gene, name).to(device))
        self.num_graphs = n

    def __getitem__(self, key: str) -> _GeneKeyed:
        assert key == "gene"
        return self._gene


def test_the_gene_keyed_target_joins_by_the_single_deleted_gene():
    value = torch.zeros(6)
    known = torch.zeros(6, dtype=torch.bool)
    value[1], known[1] = 1.0, True  # gene 1 is essential and known
    value[3], known[3] = 0.0, True  # gene 3 is nonessential and known
    value[4], known[4] = 1.0, True  # gene 4 is known but its genotype is a double
    target, mask = train_bx.gene_keyed_target(_GeneKeyedBatch(), value, known)
    assert target.shape == (4, 1) and mask.shape == (4, 1)
    assert mask[:, 0].tolist() == [True, True, False, False]
    assert target[0, 0].item() == 1.0 and target[1, 0].item() == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU to mix devices")
def test_the_target_lands_on_the_batch_device_not_the_label_device():
    """The labels live on the host and the batch does not, which crashed two cells.

    ``value`` and ``known`` are built once from the released split and stay on the CPU,
    while the indices that address them come off a CUDA batch. Allocating the target beside
    the LABELS therefore indexed a CPU tensor with CUDA rows, which every CPU-only test
    passes and every real run fails 30 seconds in (jobs 1990, 1991, 1997, 1998).
    """
    value = torch.zeros(6)
    known = torch.zeros(6, dtype=torch.bool)
    value[1], known[1] = 1.0, True
    batch = _GeneKeyedBatch(device="cuda")
    target, mask = train_bx.gene_keyed_target(batch, value, known)
    assert target.device.type == "cuda" and mask.device.type == "cuda"
    assert mask[:, 0].tolist() == [True, False, False, False]
    assert target[0, 0].item() == 1.0


def test_a_gene_absent_from_the_split_is_masked_out():
    value = torch.zeros(6)
    known = torch.zeros(6, dtype=torch.bool)
    value[1], known[1] = 1.0, True
    # gene 3 is deliberately left unknown: the split does not name it.
    target, mask = train_bx.gene_keyed_target(_GeneKeyedBatch(), value, known)
    assert mask[:, 0].tolist() == [True, False, False, False]
    assert target[1, 0].item() == 0.0


def test_targets_in_space_routes_a_gene_keyed_head_past_extract_targets():
    value = torch.zeros(6)
    known = torch.zeros(6, dtype=torch.bool)
    value[1], known[1] = 1.0, True
    spec = {
        train_bx.ESSENTIALITY_HEAD: {
            "label": train_bx.ESSENTIALITY_LABEL,
            "width": 1,
            "gene_label": value,
            "gene_known": known,
        }
    }
    # The batch carries NO phenotype tensors at all; a gene-keyed head must not need them.
    out = train_bx.targets_in_space(_GeneKeyedBatch(), spec, False)
    target, mask = out[train_bx.ESSENTIALITY_HEAD]
    assert mask[:, 0].tolist() == [True, False, False, False]
    assert target[0, 0].item() == 1.0
