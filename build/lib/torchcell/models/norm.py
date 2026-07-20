# torchcell/models/norm
# [[torchcell.models.norm]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/models/norm
# Test file: tests/torchcell/models/test_norm.py
"""Re-exports of PyTorch Geometric normalization layers used across torchcell models."""

from torch_geometric.nn import (
    BatchNorm,
    GraphNorm,
    InstanceNorm,
    LayerNorm,
    MeanSubtractionNorm,
    PairNorm,
)

norm_register = {
    "batch": BatchNorm,
    "layer": LayerNorm,
    "graph": GraphNorm,
    "instance": InstanceNorm,
    "pair": PairNorm,
    "mean_subtraction": MeanSubtractionNorm,
}
