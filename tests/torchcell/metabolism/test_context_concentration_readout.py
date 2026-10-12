# tests/torchcell/metabolism/test_context_concentration_readout.py

"""Tests for 027's context-only concentration readout, the is-the-module-on ablation.

The claim under test: ``ContextConcentrationReadout`` and ``FluxConcentrationReadout``
differ ONLY in where the natural-log concentration comes from. Handed the same ``log c``,
they emit the same standardized prediction bit for bit; the context readout computes that
``log c`` exactly as ``FluxLayer.reaction_potential`` does in ANCHORED mode; and it is built
with no flux layer at all. The metabolite axis it is built on is checked against the axis
the flux layer would have had, on the real GEM when the checkout is present.
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

#: The experiment scripts are read from THIS checkout, whatever `.env` names, so the test
#: exercises the code it sits beside.
REPO = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
os.environ["EXPERIMENT_ROOT"] = osp.join(REPO, "experiments")
sys.path.insert(
    0, osp.join(REPO, "experiments", "027-betaxanthin-metabolic", "scripts")
)
# Imported by name at run time rather than by statement: the experiment script is not a
# package and has no stub, so strict mypy cannot resolve it; ``Any`` says so explicitly.
train_bx: Any = importlib.import_module("train_bx")

from torchcell.models.cell_graph_transformer_metabolism import (  # noqa: E402
    perturbed_gene_pool,
)

GEM_CHECKOUT = osp.join(
    os.environ["DATA_ROOT"], "data/torchcell/yeast-GEM/yeast-GEM-9.0.2"
)


class _Backbone(nn.Module):
    """Stands in for the metabolism transformer: fixed gene tokens, no flux layer.

    With ``flux`` given it plays the flux arm's model instead, emitting that tensor as
    ``thermo_log_c`` the way the layer would, so the two readouts can be fed one ``log c``.
    """

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


class _Genes:
    def __init__(self, indices: torch.Tensor, batch: torch.Tensor) -> None:
        self.perturbation_indices = indices
        self.perturbation_indices_batch = batch


B, N, D, M = 4, 6, 8, 12
LO, HI = 1e-7, 1.0


def _fixture() -> tuple[torch.Tensor, dict[str, Any], torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    tokens = torch.randn(B, N, D)
    batch = {
        "gene": _Genes(
            torch.tensor([0, 2, 1, 3, 4, 5]), torch.tensor([0, 0, 1, 2, 2, 3])
        )
    }
    # Three "amino acids" over up to two compartments each; the third has one compartment
    # and a -inf padding weight, which is the case the log-sum-exp has to get right.
    indices = torch.tensor([[0, 1], [2, 3], [4, 4]])
    log_w = torch.tensor(
        [
            [math.log(0.65), math.log(0.18)],
            [math.log(0.65), math.log(0.05)],
            [0.0, -math.inf],
        ]
    )
    return tokens, batch, indices, log_w


def _context_readout() -> tuple[train_bx.ContextConcentrationReadout, dict[str, Any]]:
    tokens, batch, indices, log_w = _fixture()
    readout = train_bx.ContextConcentrationReadout(
        _Backbone(tokens), indices, log_w, True, D, M, LO, HI
    )
    with torch.no_grad():
        readout.log_c_base.normal_()
        readout.log_sigma.normal_()
    readout.set_target_standardization(
        torch.tensor([-3.0, -2.5, -4.0]), torch.tensor([0.5, 0.7, 0.9])
    )
    return readout, batch


def test_context_readout_is_the_anchored_parameterization_on_the_pooled_context():
    readout, batch = _context_readout()
    _, reps = readout(None, batch)
    log_c = reps["context_log_c"]
    context = perturbed_gene_pool(
        readout.model.tokens,
        batch["gene"].perturbation_indices,
        batch["gene"].perturbation_indices_batch,
    )
    lo, hi = math.log(LO), math.log(HI)
    expected = lo + (hi - lo) * torch.sigmoid(
        readout.log_c_base.unsqueeze(0) + readout.log_c_from_context(context)
    )
    assert torch.equal(log_c, expected)
    assert log_c.shape == (B, M)
    assert bool((log_c > lo).all()) and bool((log_c < hi).all())
    assert "flux" not in reps


def test_context_readout_and_flux_readout_agree_bit_for_bit_on_the_same_log_c():
    readout, batch = _context_readout()
    _, reps = readout(None, batch)
    pred_context = reps["head_outputs"][train_bx.AUX_HEAD]
    assert pred_context.shape == (B, 3, 2)

    tokens, _, indices, log_w = _fixture()
    flux_side = train_bx.FluxConcentrationReadout(
        _Backbone(tokens, flux=reps["context_log_c"].detach()), indices, log_w, True
    )
    with torch.no_grad():
        flux_side.log_sigma.copy_(readout.log_sigma)
    flux_side.set_target_standardization(readout.target_mean, readout.target_sd)
    _, flux_reps = flux_side(None, batch)
    assert torch.equal(pred_context, flux_reps["head_outputs"][train_bx.AUX_HEAD])


def test_context_readout_gradient_reaches_the_concentration_head_and_only_read_rows():
    readout, batch = _context_readout()
    _, reps = readout(None, batch)
    reps["head_outputs"][train_bx.AUX_HEAD][..., 0].sum().backward()
    grad = readout.log_c_base.grad
    assert grad is not None
    read = {0, 1, 2, 3, 4}
    for row in range(M):
        if row in read:
            assert grad[row] != 0.0
        else:
            assert grad[row] == 0.0
    assert readout.log_c_from_context.weight.grad is not None


def test_context_readout_builds_without_a_flux_layer_and_counts_its_head():
    readout, _ = _context_readout()
    assert readout.flux_layer is None
    cell = train_bx.build_cell(
        "pooled",
        0,
        train_bx.load_config(),
        crps=True,
        head=train_bx.AUX_HEAD,
        aa_readout=train_bx.AA_READOUT_CONTEXT_CONCENTRATION,
    )
    assert cell.aa_readout == "context_concentration"
    assert cell.target_log10_molar is True
    assert train_bx._sampling_meta(readout, cell) == {
        "sampling": {"flux_draws": 1, "latent_stochastic": False}
    }
    # Linear(D, M) is M*D weights + M biases, the base is M, log_sigma is one per column.
    counts = readout.num_parameters
    assert counts[f"{train_bx.AUX_HEAD}_head"] == M * (D + 2) + 3
    assert counts["total"] == 1 + M * (D + 2) + 3

    tokens, _, indices, log_w = _fixture()
    flux_side = train_bx.FluxConcentrationReadout(
        _Backbone(tokens, flux=torch.zeros(B, M)), indices, log_w, True
    )
    assert flux_side.num_parameters[f"{train_bx.AUX_HEAD}_head"] == 3


def test_context_readout_refuses_a_model_with_a_flux_layer():
    tokens, _, indices, log_w = _fixture()
    backbone = _Backbone(tokens)
    backbone.flux_layer = nn.Linear(1, 1)
    with pytest.raises(AssertionError, match="pooled backbone"):
        train_bx.ContextConcentrationReadout(
            backbone, indices, log_w, True, D, M, LO, HI
        )


@pytest.mark.skipif(
    not osp.exists(GEM_CHECKOUT), reason="yeast-GEM checkout not present"
)
def test_gem_metabolite_index_is_the_flux_layers_metabolite_axis():
    from torchcell.metabolism.constraints import build_gem_tensors
    from torchcell.metabolism.yeast_GEM import YeastGEM

    source = YeastGEM()
    gem = build_gem_tensors(
        source.model, with_thermo=False, with_independent_rows=False
    )
    layer_index = {m: i for i, m in enumerate(gem.met_ids)}
    index = train_bx.gem_metabolite_index()
    assert index == layer_index
    assert len(index) == 2806
    for aa in train_bx.AA_KEYS:
        assert train_bx.AMINO_ACID_TO_GEM_METABOLITE[aa] in index
