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

import torch

from tests.torchcell.conftest import make_dcell_batch, make_dcell_graph
from torchcell.losses.dcell import DCellLoss
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
