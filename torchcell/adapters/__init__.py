"""BioCypher adapters mapping torchcell datasets into knowledge-graph nodes and edges."""

from .cell_adapter import CellAdapter as CellAdapter
from .costanzo2016_adapter import DmfCostanzo2016Adapter as DmfCostanzo2016Adapter
from .costanzo2016_adapter import DmiCostanzo2016Adapter as DmiCostanzo2016Adapter
from .costanzo2016_adapter import SmfCostanzo2016Adapter as SmfCostanzo2016Adapter
from .kuzmin2018_adapter import DmfKuzmin2018Adapter as DmfKuzmin2018Adapter
from .kuzmin2018_adapter import DmiKuzmin2018Adapter as DmiKuzmin2018Adapter
from .kuzmin2018_adapter import SmfKuzmin2018Adapter as SmfKuzmin2018Adapter
from .kuzmin2018_adapter import TmfKuzmin2018Adapter as TmfKuzmin2018Adapter
from .kuzmin2018_adapter import TmiKuzmin2018Adapter as TmiKuzmin2018Adapter
from .kuzmin2020_adapter import DmfKuzmin2020Adapter as DmfKuzmin2020Adapter
from .kuzmin2020_adapter import DmiKuzmin2020Adapter as DmiKuzmin2020Adapter
from .kuzmin2020_adapter import SmfKuzmin2020Adapter as SmfKuzmin2020Adapter
from .kuzmin2020_adapter import TmfKuzmin2020Adapter as TmfKuzmin2020Adapter
from .kuzmin2020_adapter import TmiKuzmin2020Adapter as TmiKuzmin2020Adapter
from .ohya2005_adapter import ScmdOhya2005Adapter as ScmdOhya2005Adapter
from .sgd_adapter import GeneEssentialitySgdAdapter as GeneEssentialitySgdAdapter
from .synth_leth_db_adapter import (
    SynthLethalityYeastSynthLethDbAdapter as SynthLethalityYeastSynthLethDbAdapter,
)
from .synth_leth_db_adapter import (
    SynthRescueYeastSynthLethDbAdapter as SynthRescueYeastSynthLethDbAdapter,
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

ohya2005_adapters = ["ScmdOhya2005Adapter"]


__all__ = (
    cell_adapters
    + costanzo2016_adapters
    + kuzmin2018_adapters
    + kuzmin2020_adapters
    + gene_essentiality_adapters
    + synth_leth_db_adapters
    + ohya2005_adapters
)
