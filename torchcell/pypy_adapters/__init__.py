from .costanzo2016_pypy_adapter import DmfCostanzo2016Adapter, SmfCostanzo2016Adapter
from .kuzmin2018_pypy_adapter import (
    DmfKuzmin2018Adapter,
    SmfKuzmin2018Adapter,
    TmfKuzmin2018Adapter,
)

kuzmin2018_adapters = [
    "SmfKuzmin2018Adapter",
    "DmfKuzmin2018Adapter",
    "TmfKuzmin2018Adapter",
]
costanzo2016_adapters = ["SmfCostanzo2016Adapter", "DmfCostanzo2016Adapter"]

__all__ = kuzmin2018_adapters + costanzo2016_adapters
