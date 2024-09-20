from .cell_adapter import CellAdapter
from .costanzo2016_adapter import (
    SmfCostanzo2016Adapter,
    DmfCostanzo2016Adapter,
    DmiCostanzo2016Adapter,
)
from .kuzmin2018_adapter import (
    SmfKuzmin2018Adapter,
    DmfKuzmin2018Adapter,
    TmfKuzmin2018Adapter,
    DmiKuzmin2018Adapter,
    TmiKuzmin2018Adapter,
)
from .kuzmin2020_adapter import (
    SmfKuzmin2020Adapter,
    DmfKuzmin2020Adapter,
    TmfKuzmin2020Adapter,
    DmiKuzmin2020Adapter,
    TmiKuzmin2020Adapter,
)

from .sgd_adapter import GeneEssentialitySgdAdapter

from .synth_leth_db_adapter import (
    SynthLethalityYeastSynthLethDbAdapter,
    SynthRescueYeastSynthLethDbAdapter,
)

cell_adapters = ["CellAdapter"]

costanzo2016_adapters = [
    "SmfCostanzo2016Adapter",
    "DmfCostanzo2016Adapter",
    "DmiCostanzo2016Adapter",
]

kuzmin2018_adapters = [
    "SmfKuzmin2018Adapter",
    "DmfKuzmin2018Adapter",
    "TmfKuzmin2018Adapter",
    "DmiKuzmin2018Adapter",
    "TmiKuzmin2018Adapter",
]

kuzmin2020_adapters = [
    "SmfKuzmin2020Adapter",
    "DmfKuzmin2020Adapter",
    "TmfKuzmin2020Adapter",
    "DmiKuzmin2020Adapter",
    "TmiKuzmin2020Adapter",
]

gene_essentiality_adapters = ["GeneEssentialitySgdAdapter"]

synth_leth_db_adapters = [
    "SynthLethalityYeastSynthLethDbAdapter",
    "SynthRescueYeastSynthLethDbAdapter",
]


__all__ = (
    cell_adapters
    + costanzo2016_adapters
    + kuzmin2018_adapters
    + kuzmin2020_adapters
    + gene_essentiality_adapters
    + synth_leth_db_adapters
)
