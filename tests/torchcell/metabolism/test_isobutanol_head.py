# tests/torchcell/metabolism/test_isobutanol_head.py

"""Tests for 027's wave-9 pieces: the isobutanol decode and the context precursor ablation.

Three claims are pinned here, each of which failed silently in an earlier design.

1. **The biosensor is a COLUMN of a wider block, not a width-1 group.** On ``fig6_build``
   the deduplicator merged the Lopez screen into the Mulleder record, so a width-1 head
   would have read betaxanthin colony color. The decode takes column
   ``ISOBUTANOL_COLUMN`` of the width-20 block and moves it into log10 of the fold change.
2. **The held-back panel is masked at the one place targets enter the run**, by gene node
   index rather than by a gene-name string the batch does not carry.
3. **``ContextPrecursorConcentrationReadout`` differs from the flux precursor readout only
   in where ``log c`` comes from.** Handed the same concentration and the same affine, the
   two emit the same prediction bit for bit.
"""

import importlib
import math
import os
import os.path as osp
import sys
from typing import Any

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

B, N, D, M, P = 3, 5, 8, 12, 2
LO, HI = 1e-7, 1.0


class _Genes:
    """The COO phenotype tensors plus the perturbation indices the mask reads."""

    def __init__(
        self,
        values: list[float],
        sample_indices: list[int],
        values_batch: list[int],
        perturbation_indices: list[int],
        perturbation_indices_batch: list[int],
    ) -> None:
        self.phenotype_values = torch.tensor(values, dtype=torch.float)
        self.phenotype_type_indices = torch.zeros(len(values), dtype=torch.long)
        self.phenotype_values_batch = torch.tensor(values_batch, dtype=torch.long)
        self.phenotype_sample_indices = torch.tensor(sample_indices, dtype=torch.long)
        self.perturbation_indices = torch.tensor(perturbation_indices, dtype=torch.long)
        self.perturbation_indices_batch = torch.tensor(
            perturbation_indices_batch, dtype=torch.long
        )

    def __getitem__(self, key: str) -> list[list[str]]:
        assert key == "phenotype_types"
        return [["metabolite_level"]] * B


class _Batch:
    def __init__(self, genes: _Genes) -> None:
        self._genes = genes
        self.num_graphs = B

    def __getitem__(self, key: str) -> _Genes:
        assert key == "gene"
        return self._genes


def _block_batch() -> _Batch:
    """Three genotypes: two carry the width-20 block, the third carries only width 19."""
    width = train_bx.ISOBUTANOL_BLOCK_WIDTH
    values: list[float] = []
    samples: list[int] = []
    batch: list[int] = []
    for row, (n, base) in enumerate([(width, 1.0), (width, 10.0), (19, 5.0)]):
        values.extend(base + i for i in range(n))
        samples.extend([0] * n)
        batch.extend([row] * n)
    return _Batch(
        _Genes(
            values=values,
            sample_indices=samples,
            values_batch=batch,
            # genotype 0 deletes gene 0, genotype 1 deletes gene 3, genotype 2 gene 4.
            perturbation_indices=[0, 3, 4],
            perturbation_indices_batch=[0, 1, 2],
        )
    )


def _spec(heldout: torch.Tensor | None = None) -> dict[str, dict[str, Any]]:
    spec: dict[str, Any] = {
        "label": "metabolite_level",
        "width": train_bx.ISOBUTANOL_BLOCK_WIDTH,
        "column": train_bx.ISOBUTANOL_COLUMN,
        "transform": "log10",
    }
    if heldout is not None:
        spec["heldout_gene_flags"] = heldout
    return {train_bx.ISOBUTANOL_HEAD: spec}


def test_the_isobutanol_column_is_where_the_graph_processors_key_sort_puts_it():
    assert train_bx.ISOBUTANOL_BLOCK_WIDTH == len(train_bx.AA_KEYS) + 1 == 20
    assert train_bx.ISOBUTANOL_BLOCK_KEYS == sorted([*train_bx.AA_KEYS, "isobutanol"])
    assert train_bx.ISOBUTANOL_BLOCK_KEYS[train_bx.ISOBUTANOL_COLUMN] == "isobutanol"
    assert train_bx.ISOBUTANOL_COLUMN == 8


def test_the_decode_takes_one_column_and_moves_it_into_log10_fold_change():
    out = train_bx.targets_in_space(_block_batch(), _spec(), False)
    target, mask = out[train_bx.ISOBUTANOL_HEAD]
    assert target.shape == (B, 1) and mask.shape == (B, 1)
    # Rows 0 and 1 carry the width-20 block, row 2 carries width 19 and matches nothing.
    assert mask[:, 0].tolist() == [True, True, False]
    assert target[0, 0].item() == pytest.approx(math.log10(1.0 + 8))
    assert target[1, 0].item() == pytest.approx(math.log10(10.0 + 8))


def test_a_held_back_gene_is_masked_out_and_nothing_else_is():
    flags = torch.zeros(6, dtype=torch.bool)
    flags[3] = True  # the gene genotype 1 deletes
    out = train_bx.targets_in_space(_block_batch(), _spec(flags), False)
    _, mask = out[train_bx.ISOBUTANOL_HEAD]
    assert mask[:, 0].tolist() == [True, False, False]


def test_heldout_rows_is_an_or_over_a_genotypes_perturbed_genes():
    flags = torch.zeros(6, dtype=torch.bool)
    flags[4] = True
    rows = train_bx.heldout_rows(_block_batch(), flags)
    assert rows.tolist() == [False, False, True]


def test_a_non_positive_fold_change_is_refused_rather_than_clipped():
    batch = _block_batch()
    batch["gene"].phenotype_values[train_bx.ISOBUTANOL_COLUMN] = -1.0
    with pytest.raises(AssertionError, match="non-positive observed value"):
        train_bx.targets_in_space(batch, _spec(), False)


class _Backbone(nn.Module):
    """The pooled backbone, or the flux arm's model when ``flux`` is given."""

    def __init__(self, tokens: torch.Tensor, flux: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("tokens", tokens)
        self.flux_layer: nn.Module | None = None
        self.anchor = nn.Parameter(torch.zeros(1))
        self._flux = flux

    @property
    def num_parameters(self) -> dict[str, int]:
        return {"total": 1}

    def forward(
        self, cell_graph: Any, batch: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        reps: dict[str, Any] = {
            "H_genes_pert": self.tokens,
            "head_outputs": {},
            "graph_reg_loss": torch.zeros(()),
        }
        if self._flux is not None:
            reps["flux"] = {"thermo_log_c": self._flux}
        return {}, reps


def _precursor_fixture() -> tuple[torch.Tensor, Any, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    tokens = torch.randn(B, N, D)
    genes = _Genes(
        values=[0.0],
        sample_indices=[0],
        values_batch=[0],
        perturbation_indices=[0, 2, 1, 3],
        perturbation_indices_batch=[0, 0, 1, 2],
    )
    indices = torch.tensor([[0, 1], [2, 2]])
    log_w = torch.tensor([[math.log(0.65), math.log(0.18)], [0.0, -math.inf]])
    return tokens, _Batch(genes), indices, log_w


def test_context_precursor_readout_matches_the_flux_one_on_the_same_log_c():
    tokens, batch, indices, log_w = _precursor_fixture()
    context_side = train_bx.ContextPrecursorConcentrationReadout(
        _Backbone(tokens), indices, log_w, 2, D, M, LO, HI
    )
    with torch.no_grad():
        context_side.log_c_base.normal_()
    _, reps = context_side(None, batch)
    pred = reps["head_outputs"][train_bx.HEAD]
    assert pred.shape == (B, 1, 2)
    assert "flux" not in reps

    flux_side = train_bx.FluxPrecursorConcentrationReadout(
        _Backbone(tokens, flux=reps["context_log_c"].detach()), indices, log_w, 2
    )
    with torch.no_grad():
        flux_side.readout.weight.copy_(context_side.readout.weight)
        flux_side.readout.bias.copy_(context_side.readout.bias)
    _, flux_reps = flux_side(None, batch)
    assert torch.equal(pred, flux_reps["head_outputs"][train_bx.HEAD])


def test_context_precursor_readout_refuses_a_model_with_a_flux_layer():
    tokens, _, indices, log_w = _precursor_fixture()
    backbone = _Backbone(tokens)
    backbone.flux_layer = nn.Linear(1, 1)
    with pytest.raises(AssertionError, match="pooled backbone"):
        train_bx.ContextPrecursorConcentrationReadout(
            backbone, indices, log_w, 2, D, M, LO, HI
        )


def test_the_precursor_readouts_name_their_head_and_count_their_parameters():
    tokens, _, indices, log_w = _precursor_fixture()
    flux_side = train_bx.FluxPrecursorConcentrationReadout(
        _Backbone(tokens, flux=torch.zeros(B, M)),
        indices,
        log_w,
        2,
        head_name=train_bx.ISOBUTANOL_HEAD,
    )
    # Linear(P=2, param_dim=2): 4 weights + 2 biases.
    assert flux_side.num_parameters[f"{train_bx.ISOBUTANOL_HEAD}_head"] == 6
    context_side = train_bx.ContextPrecursorConcentrationReadout(
        _Backbone(tokens), indices, log_w, 2, D, M, LO, HI
    )
    # Plus the concentration head: M*(D+1) weights and biases, and the base M.
    assert context_side.num_parameters[f"{train_bx.HEAD}_head"] == 6 + M * (D + 2)


def test_the_anchor_sets_are_the_ligand_alone_and_the_valine_branch():
    one = train_bx.isobutanol_anchors(train_bx.IB_ANCHORS_ONE)
    three = train_bx.isobutanol_anchors(train_bx.IB_ANCHORS_THREE)
    assert one == {"alpha_isopropylmalate": "s_0162"}
    assert set(three) == {"alpha_isopropylmalate", "keto_isovalerate", "valine"}
    assert three["keto_isovalerate"] == "s_0232" and three["valine"] == "s_1056"
    with pytest.raises(AssertionError, match="--ib-anchors"):
        train_bx.isobutanol_anchors("two")


def test_build_cell_derives_the_target_space_and_refuses_a_layerless_precursor_arm():
    cfg = train_bx.load_config()
    cell = train_bx.build_cell(
        "flux_projection_anchored",
        2700,
        cfg,
        epochs=1,
        crps=True,
        head=train_bx.ISOBUTANOL_HEAD,
        ib_readout=train_bx.IB_READOUT_PRECURSOR_CONCENTRATION,
    )
    assert cell.target_log10 is True and cell.target_log10_molar is False
    assert train_bx.dataset_paths(cell.head)[0].endswith("fig6_build")
    assert train_bx.dataset_paths(train_bx.HEAD)[0].endswith("fig6_pigment_transfer")
    pooled = train_bx.build_cell(
        "pooled", 2700, cfg, epochs=1, crps=True, head=train_bx.ISOBUTANOL_HEAD
    )
    assert pooled.target_log10 is True


def test_the_supervise_column_mask_is_validated_against_the_measured_columns():
    cfg = train_bx.load_config()
    with pytest.raises(AssertionError, match="not Mulleder columns"):
        train_bx.build_cell(
            "pooled",
            2700,
            cfg,
            epochs=1,
            head=train_bx.AUX_HEAD,
            aa_supervise_columns=["alanine", "isobutanol"],
        )
    with pytest.raises(AssertionError, match="holds out"):
        train_bx.build_cell(
            "pooled",
            2700,
            cfg,
            epochs=1,
            head=train_bx.AUX_HEAD,
            aa_supervise_columns=list(train_bx.AA_KEYS),
        )
    cell = train_bx.build_cell(
        "pooled",
        2700,
        cfg,
        epochs=1,
        head=train_bx.AUX_HEAD,
        aa_readout=train_bx.AA_READOUT_CONTEXT_CONCENTRATION,
        aa_supervise_columns=[k for k in train_bx.AA_KEYS if k != "valine"],
    )
    assert cell.aa_supervise_columns is not None
    assert len(cell.aa_supervise_columns) == len(train_bx.AA_KEYS) - 1
