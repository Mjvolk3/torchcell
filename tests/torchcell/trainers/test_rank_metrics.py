# tests/torchcell/trainers/test_rank_metrics
# [[tests.torchcell.trainers.test_rank_metrics]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/trainers/test_rank_metrics
"""average_rank equals scipy's rankdata with averaged ties and omitted NaNs."""

import numpy as np
import torch
from scipy.stats import rankdata

from torchcell.trainers.rank_metrics import average_rank


def test_matches_scipy_on_continuous_values() -> None:
    x = torch.randn(200, 37, generator=torch.Generator().manual_seed(0))
    expected = rankdata(x.numpy(), axis=0)
    assert np.array_equal(average_rank(x).numpy(), expected.astype(np.float32))


def test_matches_scipy_with_heavy_ties() -> None:
    """An ordinal target: eleven distinct values over 300 strains."""
    g = torch.Generator().manual_seed(1)
    x = torch.randint(-5, 6, (300, 9), generator=g).float()
    expected = rankdata(x.numpy(), axis=0)
    assert np.array_equal(average_rank(x).numpy(), expected.astype(np.float32))


def test_nan_entries_stay_nan_and_are_omitted_from_the_ranking() -> None:
    g = torch.Generator().manual_seed(2)
    x = torch.randint(0, 6, (120, 11), generator=g).float()
    x[torch.rand(120, 11, generator=g) < 0.3] = float("nan")
    x[:, 4] = float("nan")  # a column with nothing measured
    expected = rankdata(x.numpy(), axis=0, nan_policy="omit").astype(np.float32)
    got = average_rank(x).numpy()
    assert np.array_equal(np.isnan(got), np.isnan(expected))
    assert np.array_equal(got[~np.isnan(got)], expected[~np.isnan(expected)])


def test_known_small_case() -> None:
    x = torch.tensor([[3.0], [1.0], [3.0], [2.0], [float("nan")]])
    ranks = average_rank(x).squeeze(1)
    assert ranks[:4].tolist() == [3.5, 1.0, 3.5, 2.0]
    assert torch.isnan(ranks[4])
