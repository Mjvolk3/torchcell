# tests/torchcell/models/test_dcell_opt.py
# [[tests.torchcell.models.test_dcell_opt]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/models/test_dcell_opt.py
"""``DCellOpt`` declares its root key, so ``DCellLoss`` counts exactly two auxiliaries.

Fixture: the conftest three-term hierarchy (root GO:0, children GO:1 and GO:2), seed 0,
``min_subsystem_size=2``, ``subsystem_ratio=0.5``; batch knocks out gene 0, genes 2 and
3, and gene 1 (B = 3); target y = [0.5, -0.5, 0.25]. The seeded heads give

* GO:0 (root) = [0.6433955, 0.6312478, 0.6461661],
  MSE = (0.1433955^2 + 1.1312478^2 + 0.3961661^2) / 3 = 0.4857438;
* GO:1 = [-0.3495494, -0.4098770, -0.3280257],
  MSE = (0.8495494^2 + 0.0901230^2 + 0.5780257^2) / 3 = 0.3546567;
* GO:2 = [0.0388609, 0.3137608, 0.0388609],
  MSE = (0.4611391^2 + 0.8137608^2 + 0.2111391^2) / 3 = 0.3064786.

Paper objective (sum over the two non-root terms): 0.4857438 + 0.3 * (0.3546567 +
0.3064786) = 0.4857438 + 0.1983406 = 0.6840844. ``DCellOpt`` builds ``predictions`` and
``GO:0`` by two separate indexing calls, so they are equal but distinct tensors; an
identity-based root skip counted GO:0 as a third auxiliary term (0.6840844 + 0.3 *
0.4857438 = 0.8298076).
"""

import math
import re

import pytest
import torch
from torch_geometric.data import HeteroData

from tests.torchcell.conftest import make_dcell_batch, make_dcell_graph
from torchcell.losses.dcell import DCellLoss
from torchcell.models.dcell import DCell
from torchcell.models.dcell_opt import DCellOpt

TARGET = torch.tensor([0.5, -0.5, 0.25])


def test_dcell_opt_loss_counts_only_the_non_root_heads() -> None:
    """Root key GO:0 is declared; the loss is the paper value 0.6840844, not 0.8298076."""
    graph = make_dcell_graph()
    torch.manual_seed(0)
    model = DCellOpt(graph, min_subsystem_size=2, subsystem_ratio=0.5, output_size=1)
    predictions, outputs = model(graph, make_dcell_batch([[0], [2, 3], [1]]))
    linear = outputs["linear_outputs"]
    assert outputs["root_key"] == "GO:0"
    assert sorted(linear) == ["GO:0", "GO:1", "GO:2", "GO:ROOT"]
    assert linear["GO:0"] is not predictions
    torch.testing.assert_close(linear["GO:0"], predictions)
    expected_heads = {
        "GO:0": [0.6433955, 0.6312478, 0.6461661],
        "GO:1": [-0.3495494, -0.4098770, -0.3280257],
        "GO:2": [0.0388609, 0.3137608, 0.0388609],
    }
    for key, values in expected_heads.items():
        torch.testing.assert_close(
            linear[key].detach(), torch.tensor(values), atol=1e-6, rtol=0
        )
    total, parts = DCellLoss(alpha=0.3, aux_reduction="sum")(
        predictions, outputs, TARGET
    )
    assert abs(parts["primary_loss"].item() - 0.4857438) < 1e-6
    assert abs(parts["auxiliary_loss"].item() - 0.6611353) < 1e-6
    assert abs(total.item() - 0.6840844) < 1e-6


# ---------------------------------------------------------------------------
# 2026.10.01 (phase 19): the optimized path against the reference DCell.
#
# Three ontologies over the conftest genes (term 1 annotates genes {0, 1}, term 2
# genes {2, 3}, the root none):
#
# * ``fixture``: ``make_dcell_graph()``; with min 2 / ratio 0.5 every output dim is
#   max(2, ceil(0.5 * n)) = 2 for counts [0, 2, 2]; input dims {0: 2 + 2 + 1, 1: 2, 2: 2}.
# * ``unequal``: the same graph with ``term_gene_counts = [0, 2, 6]``, so term 2's
#   output is max(2, ceil(0.5 * 6)) = 3 while term 1 keeps 2; the root's input is
#   2 + 3 + 1 = 6. Child concatenation order now matters (a swap changes the columns
#   each child occupies).
# * ``dag``: the three-level DAG of ``test_dcell.py`` (edges child -> parent (1, 0),
#   (2, 0), (2, 1); strata {0: [0], 1: [1], 2: [2]}; counts [5, 4, 2]) with min 1 /
#   ratio 0.3: outputs {0: 2, 1: 2, 2: 1}, inputs {0: 2 + 1 + 1, 1: 1 + 2, 2: 2}.
#
# The reference model is given non-trivial weights (standard normal for every
# parameter, normal running means, running variances in [0.5, 1.5)) and its state is
# copied into DCellOpt through the explicit name map ``_NAME_MAP`` (the two classes
# name their parameters identically, ModuleDict key "k" vs ModuleList index k, but the
# map is written out so a renamed parameter fails the key-set assertion). DCellOpt's
# seven index buffers are NOT in the reference state and must be the only missing keys.
# ---------------------------------------------------------------------------

_SUBSYSTEM_SUFFIXES = (
    "linear.weight",
    "linear.bias",
    "batch_norm.weight",
    "batch_norm.bias",
    "batch_norm.running_mean",
    "batch_norm.running_var",
    "batch_norm.num_batches_tracked",
)
_NAME_MAP = {
    **{
        f"subsystems.{k}.{s}": f"subsystems.{k}.{s}"
        for k in range(3)
        for s in _SUBSYSTEM_SUFFIXES
    },
    **{
        f"linear_heads.{k}.{s}": f"linear_heads.{k}.{s}"
        for k in range(3)
        for s in ("weight", "bias")
    },
}
_OPT_ONLY_BUFFERS = [
    "has_subsystem",
    "children_indices",
    "num_children",
    "stratum_masks",
    "term_row_indices_tensor",
    "term_num_genes_tensor",
    "term_output_dims_tensor",
]
_BATCH = [[0], [2, 3], [1]]


def _unequal_graph() -> HeteroData:
    graph = make_dcell_graph()
    graph["gene_ontology"].term_gene_counts = torch.tensor([0, 2, 6])
    return graph


def _dag_graph() -> HeteroData:
    graph = make_dcell_graph()
    go = graph["gene_ontology"]
    go.strata = torch.tensor([0, 1, 2])
    go.stratum_to_terms = {
        0: torch.tensor([0]),
        1: torch.tensor([1]),
        2: torch.tensor([2]),
    }
    go.term_gene_counts = torch.tensor([5, 4, 2])
    graph["gene_ontology", "is_child_of", "gene_ontology"].edge_index = torch.tensor(
        [[1, 2, 2], [0, 0, 1]]
    )
    return graph


_GRAPHS = {
    "fixture": (make_dcell_graph, 2, 0.5),
    "unequal": (_unequal_graph, 2, 0.5),
    "dag": (_dag_graph, 1, 0.3),
}


def _paired(name: str) -> tuple[HeteroData, DCell, DCellOpt]:
    """Reference with seeded non-trivial weights, and DCellOpt loaded from it."""
    build, min_size, ratio = _GRAPHS[name]
    graph = build()
    with torch.random.fork_rng():
        torch.manual_seed(0)
        ref = DCell(graph, min_subsystem_size=min_size, subsystem_ratio=ratio)
        with torch.no_grad():
            for param in ref.parameters():
                param.copy_(torch.randn_like(param))
            for buf_name, buf in ref.named_buffers():
                if buf_name.endswith("running_mean"):
                    buf.copy_(torch.randn_like(buf))
                elif buf_name.endswith("running_var"):
                    buf.copy_(torch.rand_like(buf) + 0.5)
        torch.manual_seed(123)  # different init, so only the copy can make them agree
        opt = DCellOpt(graph, min_subsystem_size=min_size, subsystem_ratio=ratio)
    ref_state = ref.state_dict()
    assert set(ref_state) == set(_NAME_MAP)
    result = opt.load_state_dict(
        {_NAME_MAP[k]: v.clone() for k, v in ref_state.items()}, strict=False
    )
    assert result.unexpected_keys == []
    assert result.missing_keys == _OPT_ONLY_BUFFERS
    return graph, ref, opt


@pytest.mark.parametrize("graph_name", ["fixture", "unequal", "dag"])
@pytest.mark.parametrize(
    ("mode", "perturbed"), [("eval", [[1]]), ("eval", _BATCH), ("train", _BATCH)]
)
def test_optimized_forward_equals_reference_on_every_term(
    graph_name: str, mode: str, perturbed: list[list[int]]
) -> None:
    """Root prediction, every ``GO:k`` head and every subsystem activation agree exactly.

    B = 1 runs in eval mode (BatchNorm1d refuses a single sample in training); B = 3
    knocks out gene 0, genes {2, 3} and gene 1, so each sample reaches a different
    leaf input. Train mode normalizes over the batch, so the two models must also see
    the same batch rows in the same order; the running statistics they update must then
    agree too. Both models run the same float ops in the same order, so the agreement
    is bitwise (``torch.equal``), not within a tolerance.
    """
    graph, ref, opt = _paired(graph_name)
    ref.train(mode == "train")
    opt.train(mode == "train")
    batch = make_dcell_batch(perturbed)
    ref_pred, ref_out = ref(graph, batch)
    opt_pred, opt_out = opt(graph, batch)
    assert opt_pred.shape == (len(perturbed),)
    assert torch.equal(opt_pred, ref_pred)
    assert sorted(opt_out["linear_outputs"]) == sorted(ref_out["linear_outputs"])
    assert sorted(ref_out["linear_outputs"]) == ["GO:0", "GO:1", "GO:2", "GO:ROOT"]
    for key, value in ref_out["linear_outputs"].items():
        assert torch.equal(opt_out["linear_outputs"][key], value), key
    assert sorted(opt_out["term_activations"]) == [0, 1, 2]
    for term, act in ref_out["term_activations"].items():
        assert opt_out["term_activations"][term].shape == act.shape, term
        assert torch.equal(opt_out["term_activations"][term], act), term
    assert opt_out["root_key"] == ref_out["root_key"] == "GO:0"
    for name, buf in ref.state_dict().items():
        assert torch.equal(opt.state_dict()[_NAME_MAP[name]], buf), name


def test_optimized_backward_equals_reference_gradients() -> None:
    """The DCell objective root MSE + 0.3 * sum of the two non-root head MSEs
    (``DCellLoss(alpha=0.3, aux_reduction="sum")``) backpropagates the same gradient
    into every parameter of both models (unequal-children graph, train mode, B = 3).
    """
    graph, ref, opt = _paired("unequal")
    batch = make_dcell_batch(_BATCH)
    loss_fn = DCellLoss(alpha=0.3, aux_reduction="sum")
    ref_total, _ = loss_fn(*ref(graph, batch), TARGET)
    opt_total, _ = loss_fn(*opt(graph, batch), TARGET)
    assert torch.equal(opt_total, ref_total)
    ref_total.backward()
    opt_total.backward()
    opt_params = dict(opt.named_parameters())
    for name, param in ref.named_parameters():
        grad = opt_params[_NAME_MAP[name]].grad
        assert grad is not None and param.grad is not None, name
        assert torch.equal(grad, param.grad), name


def test_padded_activation_tensor_holds_each_term_in_its_own_width() -> None:
    """``all_activations_tensor`` is [B, 3, max_output_dim = 3] on the unequal graph;
    term 2 fills all three columns, terms 0 and 1 fill two and leave column 2 at zero;
    ``activation_mask`` is True for every (sample, term).
    """
    graph, ref, opt = _paired("unequal")
    opt.eval()
    ref.eval()
    batch = make_dcell_batch(_BATCH)
    _, ref_out = ref(graph, batch)
    _, out = opt(graph, batch)
    padded = out["all_activations_tensor"]
    assert opt.max_output_dim == 3
    assert padded.shape == (3, 3, 3)
    assert torch.equal(out["activation_mask"], torch.ones(3, 3, dtype=torch.bool))
    assert torch.equal(padded[:, 2, :], ref_out["term_activations"][2])
    for term in (0, 1):
        assert torch.equal(padded[:, term, :2], ref_out["term_activations"][term])
        assert torch.equal(padded[:, term, 2], torch.zeros(3))
    assert out["stratum_outputs"] == {}


def test_gene_states_extracted_in_parallel_equal_the_per_term_loop() -> None:
    """``_extract_gene_states_parallel([0, 1, 2])`` is [B, 3, max_genes = 2]: term 0
    annotates no gene, so its row is all zero (the per-term path returns [B, 1] zeros);
    terms 1 and 2 hold their two gene states. Hand values for knockouts
    {0}, {2, 3}, {1}: term 1 (genes 0, 1) = [[0, 1], [1, 1], [1, 0]], term 2 (genes
    2, 3) = [[1, 1], [0, 0], [1, 1]].

    Nothing in ``torchcell/`` or ``experiments/`` calls this helper (forward goes through
    ``_extract_gene_states_for_term``); the test pins the helper's own contract only.
    """
    graph = make_dcell_graph()
    with torch.random.fork_rng():
        torch.manual_seed(0)
        opt = DCellOpt(graph, min_subsystem_size=2, subsystem_ratio=0.5)
    batch = make_dcell_batch(_BATCH)
    states = opt._extract_gene_states_parallel(torch.tensor([0, 1, 2]), batch)
    term1 = torch.tensor([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    term2 = torch.tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
    expected = torch.stack([torch.zeros(3, 2), term1, term2], dim=1)
    assert torch.equal(states, expected)
    for j, term in enumerate((1, 2)):
        loop = opt._extract_gene_states_for_term(torch.tensor(term), batch)
        assert torch.equal(states[:, j + 1, :], loop)
    assert torch.equal(
        opt._extract_gene_states_for_term(torch.tensor(0), batch), torch.zeros(3, 1)
    )
    # a subset in another order keeps the caller's order and pads to its own max
    subset = opt._extract_gene_states_parallel(torch.tensor([2, 0]), batch)
    assert torch.equal(subset, torch.stack([term2, torch.zeros(3, 2)], dim=1))
    # only the gene-less root: max_genes = 0, so the result has no gene columns
    assert opt._extract_gene_states_parallel(torch.tensor([0]), batch).shape == (
        3,
        1,
        0,
    )


@pytest.mark.parametrize(
    ("graph_name", "order", "expected"),
    [
        # fixture: dims {0: (5, 2), 1: (2, 2), 2: (2, 2)}
        ("fixture", [0, 1, 2], [[0], [1, 2]]),
        ("fixture", [2, 0, 1], [[2, 1], [0]]),
        # unequal: dims {0: (6, 2), 1: (2, 2), 2: (2, 3)}, every term its own group
        ("unequal", [2, 0, 1], [[2], [0], [1]]),
        # dag: dims {0: (4, 2), 1: (3, 2), 2: (2, 1)}
        ("dag", [1, 2, 0, 1], [[1, 1], [2], [0]]),
    ],
)
def test_terms_group_by_input_and_output_dimension_in_first_seen_order(
    graph_name: str, order: list[int], expected: list[list[int]]
) -> None:
    """Groups are keyed by (input dim, output dim), listed in the order a key is first
    seen, and keep the caller's term order (and repeats) inside a group. Nothing in
    ``torchcell/`` or ``experiments/`` calls this helper; the test pins its own
    contract only.
    """
    _, _, opt = _paired(graph_name)
    groups = opt._group_terms_by_dimensions(torch.tensor(order))
    assert [g.tolist() for g in groups] == expected
    assert all(g.dtype == torch.long for g in groups)


def test_has_subsystem_is_true_for_every_term_so_the_skip_branches_never_fire() -> None:
    """Finding: ``_build_term_module`` runs for every term (dcell_opt.py:105-106) and
    sets ``has_subsystem[term] = True`` unconditionally (dcell_opt.py:312), so the
    "Skip if no subsystem" branches in ``_group_terms_by_dimensions`` (:474) and
    ``_process_stratum_parallel`` (:531) cannot be reached from a constructed model;
    neither ``_group_terms_by_dimensions`` nor ``_extract_gene_states_parallel`` is
    called anywhere in ``torchcell/`` or ``experiments/``. Forcing the flag off shows
    what the branch would do: the term is dropped from its group and skipped in its
    stratum (term 1 gets no activation and no head output), and a full forward then
    fails at the parent's Linear with "mat1 and mat2 shapes cannot be multiplied (3x3
    and 5x2)", since the root receives 2 + 1 columns where it was built for 2 + 2 + 1.
    Latent: every 006 config uses ``model_version: dcell``, not ``dcell_opt``. Pinned
    until the dead helpers are removed or ``has_subsystem`` can be False.
    """
    _, _, opt = _paired("fixture")
    assert torch.equal(opt.has_subsystem, torch.ones(3, dtype=torch.bool))
    opt.has_subsystem[1] = False
    groups = opt._group_terms_by_dimensions(torch.tensor([0, 1, 2]))
    assert [g.tolist() for g in groups] == [[0], [2]]
    opt.eval()
    batch = make_dcell_batch(_BATCH)
    all_act = torch.zeros(3, 3, opt.max_output_dim)
    mask = torch.zeros(3, 3, dtype=torch.bool)
    heads = torch.zeros(3, 3, 1)
    opt._process_stratum_parallel(torch.tensor([1, 2]), batch, all_act, mask, heads)
    assert mask.tolist() == [[False, False, True]] * 3
    assert torch.equal(all_act[:, 1, :], torch.zeros(3, 2))
    assert torch.equal(heads[:, 1, :], torch.zeros(3, 1))
    # term 2's stored activation is its subsystem applied to its gene states
    genes_2 = torch.tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
    assert torch.equal(all_act[:, 2, :], opt.subsystems[2](genes_2))
    assert torch.equal(heads[:, 2, :], opt.linear_heads[2](opt.subsystems[2](genes_2)))
    # a stratum whose only term is flagged off returns before writing anything
    only = torch.zeros(3, 3, opt.max_output_dim)
    only_mask = torch.zeros(3, 3, dtype=torch.bool)
    opt._process_stratum_parallel(
        torch.tensor([1]), batch, only, only_mask, torch.zeros(3, 3, 1)
    )
    assert not only_mask.any()
    assert torch.equal(only, torch.zeros(3, 3, 2))
    with pytest.raises(
        RuntimeError,
        match=re.escape("mat1 and mat2 shapes cannot be multiplied (3x3 and 5x2)"),
    ):
        opt(make_dcell_graph(), batch)


def test_process_stratum_with_no_terms_writes_nothing() -> None:
    """An empty stratum returns before touching the three buffers."""
    _, _, opt = _paired("fixture")
    batch = make_dcell_batch(_BATCH)
    all_act = torch.full((3, 3, 2), 7.0)
    mask = torch.zeros(3, 3, dtype=torch.bool)
    heads = torch.full((3, 3, 1), 7.0)
    opt._process_stratum_parallel(
        torch.tensor([], dtype=torch.long), batch, all_act, mask, heads
    )
    assert torch.equal(all_act, torch.full((3, 3, 2), 7.0))
    assert not mask.any()
    assert torch.equal(heads, torch.full((3, 3, 1), 7.0))


@pytest.mark.parametrize(
    ("graph_name", "expected"),
    [
        # subsystem = in*out + out (Linear) + 2*out (BatchNorm affine) = out*(in + 3);
        # head = out + 1.
        # fixture: 2*(5+3) + 2*(2+3) + 2*(2+3) = 16 + 10 + 10 = 36; heads 3 * 3 = 9.
        ("fixture", {"subsystems": 36, "dcell_linear": 9}),
        # unequal: 2*(6+3) + 2*(2+3) + 3*(2+3) = 18 + 10 + 15 = 43; heads 3 + 3 + 4 = 10.
        ("unequal", {"subsystems": 43, "dcell_linear": 10}),
        # dag: 2*(4+3) + 2*(3+3) + 1*(2+3) = 14 + 12 + 5 = 31; heads 3 + 3 + 2 = 8.
        ("dag", {"subsystems": 31, "dcell_linear": 8}),
    ],
)
def test_num_parameters_matches_the_hand_count_and_the_reference(
    graph_name: str, expected: dict[str, int]
) -> None:
    """Buffers are not parameters, so ``total`` equals ``dcell``; the dict equals the
    reference model's ``num_parameters`` key for key.
    """
    _, ref, opt = _paired(graph_name)
    dcell = expected["subsystems"] + expected["dcell_linear"]
    assert opt.num_parameters == {
        **expected,
        "dcell": dcell,
        "total": dcell,
        "num_go_terms": 3,
        "num_subsystems": 3,
    }
    assert opt.num_parameters == ref.num_parameters


def test_profiling_prints_one_line_per_stratum_and_leaves_outputs_unchanged(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``enable_profiling`` toggles ``_profile_mode`` with its two messages; with it on,
    forward prints the stratum 1 line (2 terms) before the stratum 0 line (1 term),
    then the total; predictions are bitwise the unprofiled ones.
    """
    graph, _, opt = _paired("fixture")
    opt.eval()
    batch = make_dcell_batch(_BATCH)
    plain, _ = opt(graph, batch)
    capsys.readouterr()
    opt.enable_profiling()
    assert opt._profile_mode is True
    assert capsys.readouterr().out == (
        "DCellOpt profiling enabled - will log stratum processing times\n"
    )
    profiled, _ = opt(graph, batch)
    assert torch.equal(profiled, plain)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0:2] == ["", "Stratum processing times (PARALLEL):"]
    assert re.fullmatch(
        r"  Stratum 1: 2 terms in \d+\.\d{4}s \(\d+\.\d{6}s per term\)", lines[2]
    )
    assert re.fullmatch(
        r"  Stratum 0: 1 terms in \d+\.\d{4}s \(\d+\.\d{6}s per term\)", lines[3]
    )
    assert re.fullmatch(r"  Total: \d+\.\d{4}s", lines[4])
    assert len(lines) == 5
    opt.enable_profiling(False)
    assert opt._profile_mode is False
    assert capsys.readouterr().out == "DCellOpt profiling disabled\n"
    opt(graph, batch)
    assert capsys.readouterr().out == ""


def test_an_ontology_without_a_stratum_zero_is_refused_at_forward() -> None:
    """Root at stratum 1 and leaves at stratum 2 build (``stratum_masks`` has an all-False
    row 0), every term is processed, then forward refuses with the exact message.
    """
    graph = make_dcell_graph()
    graph["gene_ontology"].stratum_to_terms = {
        1: torch.tensor([0]),
        2: torch.tensor([1, 2]),
    }
    with torch.random.fork_rng():
        torch.manual_seed(0)
        opt = DCellOpt(graph, min_subsystem_size=2, subsystem_ratio=0.5)
    assert opt.stratum_masks.tolist() == [
        [False, False, False],
        [True, False, False],
        [False, True, True],
    ]
    opt.eval()
    with pytest.raises(ValueError, match=re.escape("No root terms found in stratum 0")):
        opt(graph, make_dcell_batch(_BATCH))


def test_output_dim_is_the_ceil_formula() -> None:
    """max(min_subsystem_size, ceil(ratio * n)) on the dag graph counts [5, 4, 2] with
    ratio 0.3: ceil(1.5) = 2, ceil(1.2) = 2, ceil(0.6) = 1.
    """
    _, _, opt = _paired("dag")
    assert opt.term_output_dims == {
        t: max(1, math.ceil(0.3 * n)) for t, n in enumerate([5, 4, 2])
    }
    assert opt.term_output_dims == {0: 2, 1: 2, 2: 1}
    assert opt.term_input_dims == {0: 4, 1: 3, 2: 2}
    assert opt.children_indices.tolist() == [[1, 2], [2, -1], [-1, -1]]
    assert opt.num_children.tolist() == [2, 1, 0]
    assert opt.term_output_dims_tensor.tolist() == [2, 2, 1]


def _seeded_pair(graph: HeteroData, output_size: int) -> tuple[DCell, DCellOpt]:
    """Reference with standard-normal weights; DCellOpt loaded from it by name."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        ref = DCell(
            graph, min_subsystem_size=2, subsystem_ratio=0.5, output_size=output_size
        )
        with torch.no_grad():
            for param in ref.parameters():
                param.copy_(torch.randn_like(param))
        torch.manual_seed(123)
        opt = DCellOpt(
            graph, min_subsystem_size=2, subsystem_ratio=0.5, output_size=output_size
        )
    result = opt.load_state_dict(ref.state_dict(), strict=False)
    assert result.missing_keys == _OPT_ONLY_BUFFERS
    return ref.eval(), opt.eval()


def test_with_two_roots_only_the_first_feeds_the_prediction() -> None:
    """Finding: stratum 0 = [0, 2] (term 2 is a second root: no parent edge, edges only
    1 -> 0). Both models build and run both roots, but the prediction is the head of
    ``root_terms[0]`` alone (dcell.py:341, dcell_opt.py:677-678): ``GO:0`` and
    ``GO:ROOT`` are the prediction, while root 2's head ``GO:2`` is computed, differs
    from it, and reaches the loss only as an auxiliary term. No warning is given.
    DCellOpt equals the reference on every head. Not measured whether any served GO
    hierarchy has more than one root at stratum 0. Pinned until several roots are
    refused or combined.
    """
    graph = make_dcell_graph()
    graph["gene_ontology"].stratum_to_terms = {
        0: torch.tensor([0, 2]),
        1: torch.tensor([1]),
    }
    graph["gene_ontology", "is_child_of", "gene_ontology"].edge_index = torch.tensor(
        [[1], [0]]
    )
    ref, opt = _seeded_pair(graph, 1)
    assert opt.term_input_dims == {0: 2 + 1, 1: 2, 2: 2}
    batch = make_dcell_batch(_BATCH)
    ref_pred, ref_out = ref(graph, batch)
    pred, out = opt(graph, batch)
    assert out["root_key"] == ref_out["root_key"] == "GO:0"
    assert torch.equal(pred, out["linear_outputs"]["GO:0"])
    assert torch.equal(out["linear_outputs"]["GO:ROOT"], pred)
    assert not torch.equal(out["linear_outputs"]["GO:2"], pred)
    assert torch.equal(pred, ref_pred)
    for key, value in ref_out["linear_outputs"].items():
        assert torch.equal(out["linear_outputs"][key], value), key


def test_output_size_two_matches_the_reference_head_for_head() -> None:
    """``output_size=2``: every head is Linear(out, 2), so the prediction and each
    ``GO:k`` output are [B, 2] (``squeeze(-1)`` leaves them) and DCellOpt equals the
    reference bitwise; the parameter count grows by one weight row plus one bias per
    head, (2 + 1) * 3 = 9 more than at output size 1 (54 = 45 + 9).
    """
    graph = make_dcell_graph()
    ref, opt = _seeded_pair(graph, 2)
    batch = make_dcell_batch(_BATCH)
    ref_pred, ref_out = ref(graph, batch)
    pred, out = opt(graph, batch)
    assert pred.shape == (3, 2)
    assert torch.equal(pred, ref_pred)
    for key, value in ref_out["linear_outputs"].items():
        assert value.shape == (3, 2), key
        assert torch.equal(out["linear_outputs"][key], value), key
    assert opt.num_parameters["total"] == ref.num_parameters["total"] == 54
