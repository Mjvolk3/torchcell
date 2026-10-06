# tests/torchcell/models/test_hetero_cell_bipartite_dango_diff_gi.py
# [[tests.torchcell.models.test_hetero_cell_bipartite_dango_diff_gi]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_hetero_cell_bipartite_dango_diff_gi.py
"""``GeneInteractionDiff`` (the 006 diffusion gene-interaction model) and ``LinearDecoder``.

Fixture. The four-gene, two-graph wildtype and perturbed batches of the eager sibling's
test module are imported, not copied (``_cell_graph``, ``_batch``, ``_sample``,
``_multigraph``): genes 0..3, graphs ``physical`` and ``regulatory``, samples collated
with ``follow_batch=["perturbation_indices"]`` so ``perturbation_indices_ptr`` exists.
``_diff`` builds the tiny model: hidden 8, 1 GIN conv layer, "sum" graph aggregation,
LayerNorm, local predictor with 2 heads and 1 attention layer, gating, dropout 0, and a
diffusion decoder of 1 block, 2 heads, linear schedule over T = 10, 4 sampling steps.

Parameter count by component (encoder parts derived in the sibling module docstring):
gene_embedding 32, preprocessor 160, convs 322, gene_interaction_predictor 370,
global_aggregator 113, gate_mlp 42; the eager head ``global_interaction_predictor`` (81)
is deleted; the decoder (input 2 x 8 = 16, hidden 8) is 136 + 16 + 1168 + 97 = 1417
(derived in tests/torchcell/models/test_diffusion_decoder.py). Total 2456.

The conditioning vector is z_c = [z_i_global, mean of the wildtype embeddings of the
sample's perturbed genes], width 16. In training mode the diffusion model returns a
zero placeholder; in eval mode it returns ``decoder.sample(z_c)``.

Findings pinned here: the training-mode zero placeholder and the trainer metrics it
reaches (issue #614 item 5); the production loader's ``follow_batch`` gives every
genotype in a batch the same batch-wide local conditioning; an empty genotype is
conditioned on the other genotypes' genes; the local predictor and gate MLP are built,
counted and never used; ``sample()`` runs the decoder loop twice in eval and mutates
BatchNorm statistics in train; the shipped config's ``sampling_method`` and
``conditioning_type`` are read by nothing.
"""

import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

from tests.torchcell.models.test_hetero_cell_bipartite_dango_gi import (
    _batch,
    _cell_graph,
    _multigraph,
    _sample,
)
from torchcell.models.diffusion_decoder import DenoisingBlock, DiffusionDecoder
from torchcell.models.hetero_cell_bipartite_dango_diff_gi import (
    GeneInteractionDiff,
    LinearDecoder,
)
from torchcell.models.hetero_cell_bipartite_dango_gi import GeneInteractionPredictor

N_GENES = 4
HIDDEN = 8
TINY_DIFFUSION = {
    "num_layers": 1,
    "num_heads": 2,
    "num_timesteps": 10,
    "beta_schedule": "linear",
    "sampling_steps": 4,
}
SHIPPED_YAML = (
    Path(__file__).resolve().parents[3]
    / "experiments/006-kuzmin-tmi/conf/hetero_cell_bipartite_dango_diff_gi.yaml"
)


@pytest.fixture(autouse=True)
def _restore_global_rng() -> Iterator[None]:
    """Run every test inside ``fork_rng`` so its seeding does not leak out."""
    with torch.random.fork_rng(devices=[]):
        yield


def _diff(seed: int = 0, **overrides: Any) -> GeneInteractionDiff:
    torch.manual_seed(seed)
    kwargs: dict[str, Any] = {
        "gene_num": N_GENES,
        "hidden_channels": HIDDEN,
        "num_layers": 1,
        "gene_multigraph": _multigraph(),
        "dropout": 0.0,
        "norm": "layer",
        "gene_encoder_config": {
            "encoder_type": "gin",
            "graph_aggregation_method": "sum",
        },
        "local_predictor_config": {"num_heads": 2, "num_attention_layers": 1},
        "diffusion_config": dict(TINY_DIFFUSION),
    }
    kwargs.update(overrides)
    return GeneInteractionDiff(**kwargs)


def _diffusion(model: GeneInteractionDiff) -> DiffusionDecoder:
    decoder = model.decoder
    assert isinstance(decoder, DiffusionDecoder)
    return decoder


def _counting(
    decoder: DiffusionDecoder, calls: list[int]
) -> Callable[..., torch.Tensor]:
    real = decoder.denoise

    def wrapped(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(1)
        return real(*args, **kwargs)

    return wrapped


def _block(decoder: DiffusionDecoder) -> DenoisingBlock:
    block = decoder.blocks[0]
    assert isinstance(block, DenoisingBlock)
    return block


def _bn_state(norm: nn.Module) -> tuple[int, torch.Tensor]:
    """Return (num_batches_tracked, running_mean) of a BatchNorm1d."""
    assert isinstance(norm, nn.BatchNorm1d)
    assert norm.num_batches_tracked is not None and norm.running_mean is not None
    return int(norm.num_batches_tracked.item()), norm.running_mean


# ------------------------------------------------------------------ construction


def test_parameter_counts_by_component_match_the_hand_derivation() -> None:
    """Counts in the module docstring; ``total`` is the sum of the components and equals
    the model-wide count, so no parameter sits outside the reported components. The
    eager MLP head is deleted and ``diffusion_decoder`` is the same object as
    ``decoder``.
    """
    model = _diff()
    assert model.num_parameters == {
        "gene_embedding": 32,
        "preprocessor": 160,
        "convs": 322,
        "gene_interaction_predictor": 370,
        "global_aggregator": 113,
        "decoder": 1417,
        "gate_mlp": 42,
        "total": 2456,
    }
    assert sum(p.numel() for p in model.parameters()) == 2456
    assert not hasattr(model, "global_interaction_predictor")
    assert model.diffusion_decoder is model.decoder


def test_concat_combination_and_the_linear_decoder_change_only_their_parts() -> None:
    """Combination "concat" drops gate_mlp (2456 - 42 = 2414, no key). ``decoder_type="linear"``
    swaps the 1417-parameter decoder for Linear(16, 1) = 17 (2456 - 1417 + 17 = 1056)
    and sets no ``diffusion_decoder`` alias. An empty diffusion config gives the
    decoder defaults: hidden = hidden_channels 8, 4 blocks of 8 heads, T = 1000,
    50 sampling steps, cosine schedule, x0, and the MODEL norm ("batch" here builds
    BatchNorm1d in every block): 136 + 16 + 4 * 1168 + 97 = 4921.
    """
    concat = _diff(
        local_predictor_config={
            "num_heads": 2,
            "num_attention_layers": 1,
            "combination_method": "concat",
        }
    )
    assert "gate_mlp" not in concat.num_parameters
    assert concat.num_parameters["total"] == 2414
    linear = _diff(decoder_type="linear")
    assert isinstance(linear.decoder, LinearDecoder)
    assert linear.num_parameters["decoder"] == 17
    assert linear.num_parameters["total"] == 1056
    assert not hasattr(linear, "diffusion_decoder")
    default = _diff(diffusion_config=None, norm="batch")
    dec = _diffusion(default)
    assert (dec.hidden_dim, len(dec.blocks), dec.num_timesteps) == (8, 4, 1000)
    assert dec.default_sampling_steps == 50
    assert dec.parameterization == "x0"
    first = _block(dec)
    assert isinstance(first.norm1, nn.BatchNorm1d)
    assert first.cross_attn.num_heads == 8
    assert default.num_parameters["decoder"] == 4921


def test_every_read_diffusion_key_reaches_the_decoder_and_two_shipped_keys_do_not() -> (
    None
):
    """Each key the constructor reads lands on the decoder: hidden 12, 3 blocks, 3 heads,
    dropout 0.25, T 20, mlp_ratio 2, linear betas from 0.001 to 0.05, 7 sampling steps,
    parameterization "eps".

    Finding: the shipped 006 config also sets ``sampling_method: "ddim"`` and
    ``conditioning_type: "cross_attention"`` (hetero_cell_bipartite_dango_diff_gi.yaml,
    diffusion_config), which no code reads (hetero_cell_bipartite_dango_diff_gi.py:
    134-151): building with the shipped dict, or with those two keys set to "ddpm" and
    "film", gives the same state_dict as building without them. The shipped dict on the
    tiny encoder gives a 138945-parameter decoder (context_proj 16 * 64 + 64 = 1088,
    input_proj 128, 2 * 66688, head 4353). Pinned until the keys are wired or removed.
    """
    custom = _diffusion(
        _diff(
            diffusion_config={
                "hidden_dim": 12,
                "num_layers": 3,
                "num_heads": 3,
                "dropout": 0.25,
                "num_timesteps": 20,
                "mlp_ratio": 2.0,
                "beta_schedule": "linear",
                "beta_start": 0.001,
                "beta_end": 0.05,
                "sampling_steps": 7,
                "parameterization": "eps",
            }
        )
    )
    assert (custom.hidden_dim, len(custom.blocks), custom.num_timesteps) == (12, 3, 20)
    block = _block(custom)
    assert block.cross_attn.num_heads == 3
    assert block.cross_attn.dropout.p == 0.25
    hidden = block.mlp[0]
    assert isinstance(hidden, nn.Linear) and hidden.out_features == 24
    assert custom.betas[0].item() == pytest.approx(0.001)
    assert custom.betas[-1].item() == pytest.approx(0.05)
    assert (custom.default_sampling_steps, custom.parameterization) == (7, "eps")
    container = OmegaConf.to_container(
        OmegaConf.load(SHIPPED_YAML).model.diffusion_config
    )
    assert isinstance(container, dict)
    shipped = {str(k): v for k, v in container.items()}
    assert (shipped["sampling_method"], shipped["conditioning_type"]) == (
        "ddim",
        "cross_attention",
    )
    read_only = {
        k: v
        for k, v in shipped.items()
        if k not in ("sampling_method", "conditioning_type")
    }
    a = _diff(seed=2, diffusion_config=shipped).state_dict()
    b = _diff(seed=2, diffusion_config=read_only).state_dict()
    c = _diff(
        seed=2,
        diffusion_config={
            **read_only,
            "sampling_method": "ddpm",
            "conditioning_type": "film",
        },
    ).state_dict()
    assert list(a) == list(b) == list(c)
    assert all(torch.equal(a[k], b[k]) and torch.equal(a[k], c[k]) for k in a)
    assert _diff(diffusion_config=shipped).num_parameters["decoder"] == 138945


# ------------------------------------------------------------------ forward modes


def test_training_forward_returns_a_detached_zero_placeholder() -> None:
    """Finding: in training mode the diffusion model's forward returns zeros shaped like
    the targets, not a prediction (hetero_cell_bipartite_dango_diff_gi.py:242-258),
    although the docstring says "Phenotype predictions". The trainer's training step
    (torchcell/trainers/int_hetero_cell.py:1120) takes these as predictions and feeds
    them to ``train_transformed_metrics`` (1235), to ``train_metrics`` after the inverse
    transform (1276) and to the train prediction samples (1293, 1306); only
    ``train/loss`` comes from the diffusion loss. Targets [B] -> zeros [B, 1] of the
    targets' dtype; a 0-dim target -> [1, 1]; 3 targets for 2 genotypes -> [3, 1]; no
    targets -> [2, 1] from the batch size. The tensor has no grad_fn and is unchanged
    when every decoder parameter is set to 5. Pinned until training mode returns a
    prediction (issue #614 item 5; the trainer side is fixed separately).
    """
    model = _diff().train()
    cell_graph = _cell_graph()
    cases: list[tuple[torch.Tensor | None, tuple[int, int], torch.dtype]] = [
        (torch.tensor([0.5, -1.0]), (2, 1), torch.float32),
        (torch.tensor(0.5, dtype=torch.float64), (1, 1), torch.float64),
        (torch.tensor([1.0, 2.0, 3.0]), (3, 1), torch.float32),
        (None, (2, 1), torch.float32),
    ]
    for values, shape, dtype in cases:
        batch = _batch([[0, 1], [2, 3]])
        if values is not None:
            batch["gene"].phenotype_values = values
        pred, rep = model(cell_graph, batch)
        assert torch.equal(pred, torch.zeros(shape, dtype=dtype))
        assert pred.dtype == dtype
        assert pred.grad_fn is None and not pred.requires_grad
        assert rep["z_c"].requires_grad
    with torch.no_grad():
        for p in model.decoder.parameters():
            p.fill_(5.0)
    pred, _ = model(cell_graph, _batch([[0, 1], [2, 3]]))
    assert torch.equal(pred, torch.zeros(2, 1))
    assert model.training_mode is True
    model.eval()
    assert model.training_mode is True


def test_eval_forward_is_the_seeded_decoder_sample_of_z_c_and_moves_with_genotype() -> (
    None
):
    """Eval mode: under seed 3 the prediction is bit-identical to ``decoder.sample`` of
    the returned z_c under seed 3, so the forward adds nothing after the decoder;
    the same seed repeats it exactly. Changing the genotypes from [[0, 1], [2, 3]] to
    [[0, 2], [1, 3]] changes z_c and both samples under the same seed.
    """
    model = _diff().eval()
    cell_graph = _cell_graph()
    with torch.no_grad():
        torch.manual_seed(3)
        pred, rep = model(cell_graph, _batch([[0, 1], [2, 3]]))
        torch.manual_seed(3)
        direct = _diffusion(model).sample(rep["z_c"])
        torch.manual_seed(3)
        again, _ = model(cell_graph, _batch([[0, 1], [2, 3]]))
        torch.manual_seed(3)
        other, rep_other = model(cell_graph, _batch([[0, 2], [1, 3]]))
    assert pred.shape == (2, 1)
    assert torch.equal(pred, direct)
    assert torch.equal(pred, again)
    assert not torch.allclose(rep_other["z_c"], rep["z_c"], atol=1e-4)
    assert (pred - other).abs().min().item() > 1e-4


def test_an_eval_prediction_is_one_draw_from_the_global_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding: in eval mode the forward returns a single ``decoder.sample`` draw whose
    starting noise is ``torch.randn(B, 1)`` from the global RNG
    (hetero_cell_bipartite_dango_diff_gi.py:259-261, diffusion_decoder.py:422), with no
    averaging over draws and no generator of its own. The state fed to the first denoise
    call is exactly the first ``torch.randn(2, 1)`` after seeding (seed 0 and seed 1 give
    different starts); under one seed two eval forwards are bit-identical, under seeds 0
    and 1 the predictions are not. On this untrained fixture the spread is small
    (about 1e-6 and 1e-4 per genotype) because the random denoiser barely reads x_t;
    the size is not a contract, the dependence is. ``DiffusionRegressionTask._shared_step``
    computes ``val/inference_mse``, ``test/inference_mse`` and the val and test metrics
    from this forward (torchcell/trainers/int_hetero_cell.py:1120, 1181-1193, 1235,
    1276), so every reported validation and test number is a single sample that depends
    on how much RNG was consumed before it. Pinned until evaluation averages draws or
    samples from a fixed generator.
    """
    model = _diff().eval()
    dec = _diffusion(model)
    starts: list[torch.Tensor] = []
    real = dec.denoise

    def recording(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        starts.append(x.clone())
        return real(x, *args, **kwargs)

    monkeypatch.setattr(dec, "denoise", recording)
    cell_graph, batch = _cell_graph(), _batch([[0, 1], [2, 3]])
    preds: list[torch.Tensor] = []
    firsts: list[torch.Tensor] = []
    with torch.no_grad():
        for seed in (0, 0, 1):
            starts.clear()
            torch.manual_seed(seed)
            pred, _ = model(cell_graph, batch)
            preds.append(pred)
            firsts.append(starts[0])
    for seed, first in zip((0, 0, 1), firsts, strict=True):
        torch.manual_seed(seed)
        assert torch.equal(first, torch.randn(2, 1))
    assert not torch.equal(firsts[0], firsts[2])
    assert torch.equal(preds[0], preds[1])
    assert not torch.equal(preds[0], preds[2])


def test_conditioning_path_is_aggregated_perturbed_graph_beside_wildtype_gene_means() -> (
    None
):
    """Representations, batch [[0, 1], [2, 3]]: z_i_global [2, 8] =
    global_aggregator(forward_single(batch), batch.gene.batch); z_w [1, 8] =
    global_aggregator(forward_single(cell_graph), zeros, dim_size=1); pert_gene_embs row
    i = mean of the wildtype embeddings of sample i's perturbed genes (rows 0, 1 and
    rows 2, 3 of forward_single(cell_graph)); z_c [2, 16] = [z_i_global, pert_gene_embs].
    "z_i" is z_i_global and "combined_embeddings" and "z_p" are z_c (same objects); the
    wildtype global z_w is reported but does not enter z_c.
    """
    model = _diff().train()
    cell_graph, batch = _cell_graph(), _batch([[0, 1], [2, 3]])
    _, rep = model(cell_graph, batch)
    assert set(rep) == {
        "z_i_global",
        "z_i",
        "z_w",
        "pert_gene_embs",
        "z_c",
        "combined_embeddings",
        "z_p",
    }
    assert rep["z_i"] is rep["z_i_global"]
    assert rep["combined_embeddings"] is rep["z_c"] and rep["z_p"] is rep["z_c"]
    with torch.no_grad():
        z_w = model.forward_single(cell_graph)
        z_i = model.forward_single(batch)
        z_i_global = model.global_aggregator(z_i, index=batch["gene"].batch)
        z_w_global = model.global_aggregator(
            z_w, index=torch.zeros(4, dtype=torch.long), dim_size=1
        )
    pert = torch.stack([z_w[[0, 1]].mean(0), z_w[[2, 3]].mean(0)])
    assert rep["z_c"].shape == (2, 16) and rep["z_w"].shape == (1, 8)
    torch.testing.assert_close(rep["z_i_global"], z_i_global, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(rep["z_w"], z_w_global, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(rep["pert_gene_embs"], pert, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        rep["z_c"], torch.cat([z_i_global, pert], dim=-1), atol=1e-6, rtol=0.0
    )


def test_production_follow_batch_gives_every_genotype_the_batch_wide_gene_mean() -> (
    None
):
    """Finding: the 006 diffusion script builds ``CellDataModule`` without
    ``follow_batch`` (experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_diff_gi.py:
    295-305), whose default is ["x", "x_pert"] (torchcell/datamodules/cell.py:385-386),
    so batches carry neither ``perturbation_indices_ptr`` nor ``_batch``. The model
    then takes the "single batch" branch (hetero_cell_bipartite_dango_diff_gi.py:
    223-229): every genotype's local conditioning is the mean of the wildtype embeddings
    of ALL perturbed genes in the batch (here genes 0, 1, 2, 3, 1 for genotypes [0, 1],
    [2, 3], [1]), so the three rows are identical. Collated with
    ``follow_batch=["perturbation_indices"]`` the rows are the per-genotype means.
    z_i_global stays per-genotype in both. Pinned until the model refuses a batch with
    no perturbation assignment (or the script follows ``perturbation_indices``).
    """
    model = _diff().eval()
    samples = [_sample([0, 1]), _sample([2, 3]), _sample([1])]
    loader = DataLoader(samples, batch_size=3, follow_batch=["x", "x_pert"])
    production = next(iter(loader))
    assert not hasattr(production["gene"], "perturbation_indices_ptr")
    assert not hasattr(production["gene"], "perturbation_indices_batch")
    assert production["gene"].perturbation_indices.tolist() == [0, 1, 2, 3, 1]
    with torch.no_grad():
        z_w = model.forward_single(_cell_graph())
        _, prod_rep = model(_cell_graph(), production)
        _, ptr_rep = model(_cell_graph(), _batch([[0, 1], [2, 3], [1]]))
    pooled = z_w[[0, 1, 2, 3, 1]].mean(0).expand(3, -1)
    torch.testing.assert_close(prod_rep["pert_gene_embs"], pooled, atol=1e-6, rtol=0.0)
    per_sample = torch.stack([z_w[[0, 1]].mean(0), z_w[[2, 3]].mean(0), z_w[1]])
    torch.testing.assert_close(
        ptr_rep["pert_gene_embs"], per_sample, atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        prod_rep["z_i_global"], ptr_rep["z_i_global"], atol=1e-6, rtol=0.0
    )


def test_a_genotype_without_perturbations_is_conditioned_on_the_others_genes() -> None:
    """Finding: for a sample with no perturbed genes (ptr [0, 2, 2, 3] for genotypes
    [0, 1], [], [2]) the fallback (hetero_cell_bipartite_dango_diff_gi.py:220-222) gives
    it the mean over the whole batch's perturbed genes, here genes 0, 1 and 2, instead
    of refusing or using a defined empty value. Rows 0 and 2 are their own means.
    Pinned until an empty genotype is refused or given an explicit representation.
    """
    model = _diff().eval()
    batch = _batch([[0, 1], [], [2]])
    assert batch["gene"].perturbation_indices_ptr.tolist() == [0, 2, 2, 3]
    with torch.no_grad():
        z_w = model.forward_single(_cell_graph())
        _, rep = model(_cell_graph(), batch)
    expected = torch.stack([z_w[[0, 1]].mean(0), z_w[[0, 1, 2]].mean(0), z_w[2]])
    torch.testing.assert_close(rep["pert_gene_embs"], expected, atol=1e-6, rtol=0.0)


def test_a_genotypes_prediction_does_not_depend_on_its_gene_order() -> None:
    """Listing each genotype's genes in another order ([[1, 0], [3, 2]] for [[0, 1],
    [2, 3]]) changes only the order of ``perturbation_indices``; the per-sample mean of
    two rows is exact either way (a + b == b + a), so z_c and the seeded eval samples are
    bit-identical. A three-gene genotype in orders [0, 1, 2] and [2, 0, 1] agrees to
    1e-6 (float sums of three terms may round differently).
    """
    model = _diff().eval()
    cell_graph = _cell_graph()
    with torch.no_grad():
        torch.manual_seed(8)
        pred, rep = model(cell_graph, _batch([[0, 1], [2, 3]]))
        torch.manual_seed(8)
        pred_r, rep_r = model(cell_graph, _batch([[1, 0], [3, 2]]))
        torch.manual_seed(8)
        tri, rep_tri = model(cell_graph, _batch([[0, 1, 2]]))
        torch.manual_seed(8)
        tri_r, rep_tri_r = model(cell_graph, _batch([[2, 0, 1]]))
    assert torch.equal(rep_r["z_c"], rep["z_c"])
    assert torch.equal(pred_r, pred)
    torch.testing.assert_close(rep_tri_r["z_c"], rep_tri["z_c"], atol=1e-6, rtol=0.0)
    torch.testing.assert_close(tri_r, tri, atol=1e-5, rtol=0.0)


# ------------------------------------------------------------------ loss and gradients


def test_compute_diffusion_loss_forwards_t_mode_and_the_linear_loss_is_mse() -> None:
    """Diffusion: ``t_mode="zero"`` reaches the decoder, so the loss is the clean
    reconstruction mean((denoise(y, z_c, 0) - y)^2) exactly. Under seed 6, "full" equals
    ``decoder.loss(y, z_c, t_mode="full")`` under seed 6. Linear: the loss is
    mse(Linear(z_c), y) and ``t_mode`` is ignored.
    """
    model = _diff().eval()
    y = torch.tensor([[0.5], [-1.0]])
    with torch.no_grad():
        _, rep = model(_cell_graph(), _batch([[0, 1], [2, 3]]))
        z_c = rep["z_c"]
        dec = _diffusion(model)
        clean = (
            (dec.denoise(y, z_c, torch.zeros(2, dtype=torch.long)) - y) ** 2
        ).mean()
        torch.testing.assert_close(
            model.compute_diffusion_loss(y, z_c, t_mode="zero"),
            clean,
            atol=0.0,
            rtol=0.0,
        )
        torch.manual_seed(6)
        full = model.compute_diffusion_loss(y, z_c, t_mode="full")
        torch.manual_seed(6)
        assert torch.equal(full, dec.loss(y, z_c, t_mode="full"))
    linear = _diff(decoder_type="linear")
    with torch.no_grad():
        _, rep_l = linear(_cell_graph(), _batch([[0, 1], [2, 3]]))
        expected = ((linear.decoder(rep_l["z_c"]) - y) ** 2).mean()
        assert torch.equal(
            linear.compute_diffusion_loss(y, rep_l["z_c"], t_mode="bogus"), expected
        )


def test_diffusion_loss_never_reaches_the_local_predictor_or_the_gate() -> None:
    """Finding: the class docstring says only the MLP head is replaced, but forward
    never calls ``gene_interaction_predictor`` or ``gate_mlp``
    (hetero_cell_bipartite_dango_diff_gi.py:159-278); both are built, counted in
    ``num_parameters["total"]`` and get no gradient. Backward of
    ``compute_diffusion_loss`` (train mode, t_mode "full", seed 1) leaves exactly these
    with ``grad is None``: the 13 local-predictor tensors (370 parameters), the 4
    gate tensors (42) and the decoder's norm3 (16); norm1, q_proj and k_proj get
    exactly zero (one-key attention, see the decoder tests). Every other parameter,
    the whole encoder included, gets a finite, nonzero gradient. At the shipped size
    (hidden 64, 8 heads, 2 attention layers, "concat" so no gate) the dead local
    predictor is 4160 + 2 * (4 * 4160 + 1) + 65 = 37507 parameters.
    """
    model = _diff().train()
    batch = _batch([[0, 1], [2, 3]])
    _, rep = model(_cell_graph(), batch)
    torch.manual_seed(1)
    model.compute_diffusion_loss(torch.tensor([[0.5], [-1.0]]), rep["z_c"]).backward()
    params = dict(model.named_parameters())
    none = {n for n, p in params.items() if p.grad is None}
    zero = {n for n, p in params.items() if p.grad is not None and not p.grad.any()}
    local = {n for n in params if n.startswith("gene_interaction_predictor.")}
    gate = {n for n in params if n.startswith("gate_mlp.")}
    assert (len(local), len(gate)) == (13, 4)
    assert none == local | gate | {
        "decoder.blocks.0.norm3.weight",
        "decoder.blocks.0.norm3.bias",
    }
    assert sum(params[n].numel() for n in none) == 370 + 42 + 16
    assert zero == {
        f"decoder.blocks.0.{m}.{k}"
        for m in ("norm1", "cross_attn.q_proj", "cross_attn.k_proj")
        for k in ("weight", "bias")
    }
    for n, p in params.items():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n
    shipped_local = GeneInteractionPredictor(hidden_dim=64, num_heads=8, num_layers=2)
    assert sum(p.numel() for p in shipped_local.parameters()) == 37507


# ------------------------------------------------------------------ linear decoder


def test_linear_decoder_closed_form_and_sample_ignores_its_kwargs() -> None:
    """LinearDecoder(3, 1) with weight [1, 2, 3], bias 0.5: [1, 0, -1] -> 1 - 3 + 0.5 =
    -1.5 and [2, 2, 2] -> 12 + 0.5 = 12.5. ``sample`` is the same forward and accepts
    and ignores any keyword (``num_samples`` included).
    """
    lin = LinearDecoder(3, 1)
    with torch.no_grad():
        lin.proj.weight.copy_(torch.tensor([[1.0, 2.0, 3.0]]))
        lin.proj.bias.fill_(0.5)
    z = torch.tensor([[1.0, 0.0, -1.0], [2.0, 2.0, 2.0]])
    assert lin(z).tolist() == [[-1.5], [12.5]]
    assert lin.sample(z, num_samples=9, steps=3).tolist() == [[-1.5], [12.5]]


def test_linear_model_predicts_in_both_modes_and_sample_ignores_num_samples() -> None:
    """With ``decoder_type="linear"`` and hand-set weights (weight 0.1 * [1..16], bias
    -0.2) the prediction is z_c @ w + b in train mode AND in eval mode (no zero
    placeholder), and ``sample(..., num_samples=5)`` returns the same 2 rows.
    """
    model = _diff(decoder_type="linear")
    assert isinstance(model.decoder, LinearDecoder)
    with torch.no_grad():
        model.decoder.proj.weight.copy_(0.1 * torch.arange(1.0, 17.0).view(1, 16))
        model.decoder.proj.bias.fill_(-0.2)
    cell_graph, batch = _cell_graph(), _batch([[0, 1], [2, 3]])
    for mode in (True, False):
        model.train(mode)
        pred, rep = model(cell_graph, batch)
        expected = rep["z_c"] @ (0.1 * torch.arange(1.0, 17.0)) - 0.2
        torch.testing.assert_close(pred.view(-1), expected, atol=1e-5, rtol=0.0)
    sampled = model.sample(cell_graph, batch, num_samples=5)
    torch.testing.assert_close(sampled, pred, atol=0.0, rtol=0.0)


# ------------------------------------------------------------------ model.sample


def test_sample_runs_the_loop_twice_in_eval_and_mutates_batch_norm_in_train(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding: ``GeneInteractionDiff.sample`` (lines 304-333) calls ``self.forward``,
    which in eval mode already runs ``decoder.sample`` and throws the result away, then
    samples again: 2 * 4 = 8 denoise calls for 4 sampling steps, and the returned sample
    is the SECOND draw after seeding (equal to seeding, discarding one
    ``decoder.sample(z_c)`` and drawing another). In train mode forward returns the zero
    placeholder and the decoder loop runs once (4 calls) but in train mode: with
    ``norm="batch"`` the decoder's BatchNorm running statistics are updated by sampling
    (norm1 ``num_batches_tracked`` 0 -> 4; norm3, never called, stays 0), although the
    method runs under ``no_grad`` and is documented as inference. The encoder's
    BatchNorm statistics move too: the preprocessor's one shared BatchNorm runs twice
    per ``forward_single`` and ``forward_single`` runs twice (wildtype and batch), so its
    ``num_batches_tracked`` goes 0 -> 4; each conv-wrapper norm runs once per
    ``forward_single``, 0 -> 2. Latent in the shipped run (norm "layer").
    ``num_samples=1`` keeps the first genotype. Pinned until ``sample`` reuses one forward and sets eval.
    """
    cell_graph, batch = _cell_graph(), _batch([[0, 1], [2, 3]])
    model = _diff().eval()
    dec = _diffusion(model)
    with torch.no_grad():
        _, rep = model(cell_graph, batch)
    calls: list[int] = []
    monkeypatch.setattr(dec, "denoise", _counting(dec, calls))
    torch.manual_seed(9)
    out = model.sample(cell_graph, batch)
    assert len(calls) == 8
    monkeypatch.undo()
    with torch.no_grad():
        torch.manual_seed(9)
        dec.sample(rep["z_c"])
        second = dec.sample(rep["z_c"])
    assert torch.equal(out, second)
    assert model.sample(cell_graph, batch, num_samples=1).shape == (1, 1)

    bn = _diff(norm="batch").train()
    bn_dec = _diffusion(bn)
    block = _block(bn_dec)
    train_calls: list[int] = []
    monkeypatch.setattr(bn_dec, "denoise", _counting(bn_dec, train_calls))
    modules = dict(bn.named_modules())
    encoder_norms = [
        "preprocessor.mlp.1",
        "convs.0.convs.('gene', 'physical', 'gene').norm.module",
        "convs.0.convs.('gene', 'regulatory', 'gene').norm.module",
    ]
    assert [_bn_state(modules[n])[0] for n in encoder_norms] == [0, 0, 0]
    before = _bn_state(block.norm1)[1].clone()
    bn.sample(cell_graph, batch)
    assert [_bn_state(modules[n])[0] for n in encoder_norms] == [4, 2, 2]
    assert len(train_calls) == 4
    assert _bn_state(block.norm1)[0] == 4
    assert _bn_state(block.norm2)[0] == 4
    assert _bn_state(block.norm3)[0] == 0
    assert not torch.equal(_bn_state(block.norm1)[1], before)
    bn.eval()
    bn.sample(cell_graph, batch)
    assert _bn_state(block.norm1)[0] == 4


# ------------------------------------------------------------------ refusals


def test_unknown_decoder_types_are_refused_by_name() -> None:
    """Construction refuses "mlp" by name. A decoder type changed after construction
    (the only way to reach the branches at lines 262-263 and 301-302) is refused by
    ``forward`` and ``compute_diffusion_loss`` with their own message.
    """
    with pytest.raises(ValueError, match=re.escape("Unknown decoder_type: mlp")):
        _diff(decoder_type="mlp")
    model = _diff()
    _, rep = model(_cell_graph(), _batch([[0, 1], [2, 3]]))
    model.decoder_type = "mlp"
    with pytest.raises(ValueError, match=re.escape("Unknown decoder type: mlp")):
        model(_cell_graph(), _batch([[0, 1], [2, 3]]))
    with pytest.raises(ValueError, match=re.escape("Unknown decoder type: mlp")):
        model.compute_diffusion_loss(torch.zeros(2, 1), rep["z_c"])
