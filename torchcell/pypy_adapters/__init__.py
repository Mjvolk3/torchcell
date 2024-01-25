from .kuzmin2018_pypy_adapter import (
    SmfKuzmin2018Adapter,
    DmfKuzmin2018Adapter,
    TmfKuzmin2018Adapter,
)
from .costanzo2016_pypy_adapter import SmfCostanzo2016Adapter, DmfCostanzo2016Adapter


kuzmin2018_adapters = [
    "SmfKuzmin2018Adapter",
    "DmfKuzmin2018Adapter",
    "TmfKuzmin2018Adapter",
]
costanzo2016_adapters = ["SmfCostanzo2016Adapter", "DmfCostanzo2016Adapter"]

__all__ = kuzmin2018_adapters + costanzo2016_adapters
