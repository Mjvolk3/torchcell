"""PyPy-compatible BioCypher adapters for Costanzo 2016 and Kuzmin 2018 datasets."""

from .costanzo2016_pypy_adapter import DmfCostanzo2016Adapter as DmfCostanzo2016Adapter
from .costanzo2016_pypy_adapter import SmfCostanzo2016Adapter as SmfCostanzo2016Adapter
from .kuzmin2018_pypy_adapter import DmfKuzmin2018Adapter as DmfKuzmin2018Adapter
from .kuzmin2018_pypy_adapter import SmfKuzmin2018Adapter as SmfKuzmin2018Adapter
from .kuzmin2018_pypy_adapter import TmfKuzmin2018Adapter as TmfKuzmin2018Adapter

kuzmin2018_adapters = [
    "SmfKuzmin2018Adapter",
    "DmfKuzmin2018Adapter",
    "TmfKuzmin2018Adapter",
]
costanzo2016_adapters = ["SmfCostanzo2016Adapter", "DmfCostanzo2016Adapter"]

__all__ = kuzmin2018_adapters + costanzo2016_adapters
