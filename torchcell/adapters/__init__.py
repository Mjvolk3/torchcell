from .costanzo2016_adapter import SmfCostanzo2016Adapter, DmfCostanzo2016Adapter
from .kuzmin2018_adapter import (
    SmfKuzmin2018Adapter,
    DmfKuzmin2018Adapter,
    TmfKuzmin2018Adapter,
)
from .cell_adapter import CellAdapter

cell_adapters = ["CellAdapter"]

# Experiment Adapters

# Dataset Adapters
costanzo2016_adapters = ["SmfCostanzo2016Adapter", "DmfCostanzo2016Adapter"]

kuzmin2018_adapters = [
    "SmfKuzmin2018Adapter",
    "DmfKuzmin2018Adapter",
    "TmfKuzmin2018Adapter",
]

__all__ = cell_adapters + costanzo2016_adapters + kuzmin2018_adapters
