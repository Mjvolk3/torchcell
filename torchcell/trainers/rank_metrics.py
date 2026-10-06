# torchcell/trainers/rank_metrics
# [[torchcell.trainers.rank_metrics]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/trainers/rank_metrics
# Test file: tests/torchcell/trainers/test_rank_metrics.py
"""Column-wise average ranks in torch, on whatever device the input is on.

The epoch Spearman of the 019 trainer ranked ``[strains, features]`` matrices with
``scipy.stats.rankdata`` on the CPU: 1,100 strains by 6,127 genes, twice per reduction,
three reductions per epoch. On a batch-128 run that was 17 percent of the training
process's wall time (2026-10-06 stack samples). :func:`average_rank` computes the same
ranks with one sort and two scatters, so the reduction can run on the GPU.
"""

import torch


def average_rank(x: torch.Tensor) -> torch.Tensor:
    """Average ranks of each column of ``x`` (``[N, F]``), ties sharing their mean rank.

    Matches ``scipy.stats.rankdata(x, axis=0, nan_policy="omit")``: a NaN entry stays NaN
    and the finite entries of its column are ranked among themselves, from 1.

    Args:
        x: ``[N, F]`` values.

    Returns:
        ``[N, F]`` float32 ranks on ``x``'s device.
    """
    n = x.shape[0]
    nan = torch.isnan(x)
    # torch.sort places NaN after every number, so finite entries take positions 1..k.
    sorted_vals, order = torch.sort(x.double(), dim=0, stable=True)
    pos = torch.arange(1, n + 1, device=x.device, dtype=torch.float64)
    pos = pos.unsqueeze(1).expand_as(sorted_vals)
    # A new tie group starts wherever the sorted value changes (NaN != NaN, so each NaN is
    # its own group; those entries are overwritten below).
    change = torch.ones_like(sorted_vals, dtype=torch.bool)
    change[1:] = sorted_vals[1:] != sorted_vals[:-1]
    group = change.cumsum(dim=0) - 1  # [N, F], group index within the column
    sums = torch.zeros_like(sorted_vals).scatter_add_(0, group, pos)
    counts = torch.zeros_like(sorted_vals).scatter_add_(0, group, torch.ones_like(pos))
    mean_pos = (sums / counts.clamp(min=1)).gather(0, group)
    ranks = torch.empty_like(mean_pos).scatter_(0, order, mean_pos)
    ranks[nan] = float("nan")
    return ranks.float()
