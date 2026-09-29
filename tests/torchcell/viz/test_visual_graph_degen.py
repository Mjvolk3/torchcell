# tests/torchcell/viz/test_visual_graph_degen.py
# [[tests.torchcell.viz.test_visual_graph_degen]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/viz/test_visual_graph_degen.py
"""``VisGraphDegen`` on a two-node graph with hand-computed metrics.

Features ``X = [[1, 2], [3, 4]]``: column means ``[2, 3]``, centered rows
``[-1, -1]`` and ``[1, 1]``, Frobenius norm ``sqrt(4) = 2``.

Adjacency ``A = [[1, 1], [0, 1]]`` (node 0 reads itself and node 1). Powers:
``A^2 = [[1, 2], [0, 1]]``, ``A^3 = [[1, 3], [0, 1]]``. Per hop, ``A^k X`` and the
unbiased per-row variance over the two features:

* hop 1: ``[[4, 6], [3, 4]]`` -> variances ``2, 0.5`` -> mean ``1.25``
* hop 2: ``[[7, 10], [3, 4]]`` -> ``4.5, 0.5`` -> ``2.5``
* hop 3: ``[[10, 14], [3, 4]]`` -> ``8, 0.5`` -> ``4.25``

``local_bottleneck_score`` is ``last / (first + 1e-6) = 4.25 / 1.250001``; with ``k=1``
it is ``1.25 / 1.250001``.
"""

from typing import Any

import pytest
import torch
import wandb

from torchcell.viz.visual_graph_degen import VisGraphDegen

ADJ = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])


def test_compute_smoothness_is_the_frobenius_norm_of_centered_features() -> None:
    torch.testing.assert_close(VisGraphDegen.compute_smoothness(X), torch.tensor(2.0))
    constant = torch.full((3, 2), 5.0)
    torch.testing.assert_close(
        VisGraphDegen.compute_smoothness(constant), torch.tensor(0.0)
    )


def test_local_bottleneck_score_ratio_of_third_to_first_hop_diversity() -> None:
    score = VisGraphDegen.local_bottleneck_score(ADJ, X)
    torch.testing.assert_close(score, torch.tensor(4.25 / 1.250001))
    one_hop = VisGraphDegen.local_bottleneck_score(ADJ, X, k=1)
    torch.testing.assert_close(one_hop, torch.tensor(1.25 / 1.250001))


def test_log_metrics_logs_both_scores_under_the_prefix(  # test-quality: allow log_metrics returns None; its only output is the recorded wandb.log payload
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both metrics are logged as plain floats under ``<prefix>/oversmoothing`` and
    ``<prefix>/oversquashing``; the default prefix is ``train_sample``.
    """
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(wandb, "log", lambda payload: logged.append(payload))
    VisGraphDegen(ADJ, X).log_metrics()
    VisGraphDegen(ADJ, X).log_metrics(wandb_key_prefix="val_sample")
    assert logged == [
        {
            "train_sample/oversmoothing": pytest.approx(2.0),
            "train_sample/oversquashing": pytest.approx(4.25 / 1.250001, rel=1e-6),
        },
        {
            "val_sample/oversmoothing": pytest.approx(2.0),
            "val_sample/oversquashing": pytest.approx(4.25 / 1.250001, rel=1e-6),
        },
    ]
    assert all(isinstance(v, float) for payload in logged for v in payload.values())
