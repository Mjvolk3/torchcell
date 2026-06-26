"""Mapping from S. cerevisiae dataset classes to their BioCypher adapters."""

from torchcell.adapters import (
    DmfCostanzo2016Adapter,
    DmfKuzmin2018Adapter,
    DmfKuzmin2020Adapter,
    DmiCostanzo2016Adapter,
    DmiKuzmin2018Adapter,
    DmiKuzmin2020Adapter,
    GeneEssentialitySgdAdapter,
    SmfCostanzo2016Adapter,
    SmfKuzmin2018Adapter,
    SmfKuzmin2020Adapter,
    SynthLethalityYeastSynthLethDbAdapter,
    SynthRescueYeastSynthLethDbAdapter,
    TmfKuzmin2018Adapter,
    TmfKuzmin2020Adapter,
    TmiKuzmin2018Adapter,
    TmiKuzmin2020Adapter,
)
from torchcell.adapters.ohya2005_adapter import ScmdOhya2005Adapter
from torchcell.datasets.scerevisiae.costanzo2016 import (
    DmfCostanzo2016Dataset,
    DmiCostanzo2016Dataset,
    SmfCostanzo2016Dataset,
)
from torchcell.datasets.scerevisiae.kuzmin2018 import (
    DmfKuzmin2018Dataset,
    DmiKuzmin2018Dataset,
    SmfKuzmin2018Dataset,
    TmfKuzmin2018Dataset,
    TmiKuzmin2018Dataset,
)
from torchcell.datasets.scerevisiae.kuzmin2020 import (
    DmfKuzmin2020Dataset,
    DmiKuzmin2020Dataset,
    SmfKuzmin2020Dataset,
    TmfKuzmin2020Dataset,
    TmiKuzmin2020Dataset,
)
from torchcell.datasets.scerevisiae.ohya2005 import ScmdOhya2005Dataset
from torchcell.datasets.scerevisiae.sgd import GeneEssentialitySgdDataset
from torchcell.datasets.scerevisiae.synth_leth_db import (
    SynthLethalityYeastSynthLethDbDataset,
    SynthRescueYeastSynthLethDbDataset,
)

dataset_adapter_map = {
    SmfCostanzo2016Dataset: SmfCostanzo2016Adapter,
    DmfCostanzo2016Dataset: DmfCostanzo2016Adapter,
    DmiCostanzo2016Dataset: DmiCostanzo2016Adapter,
    SmfKuzmin2018Dataset: SmfKuzmin2018Adapter,
    DmfKuzmin2018Dataset: DmfKuzmin2018Adapter,
    TmfKuzmin2018Dataset: TmfKuzmin2018Adapter,
    DmiKuzmin2018Dataset: DmiKuzmin2018Adapter,
    TmiKuzmin2018Dataset: TmiKuzmin2018Adapter,
    SmfKuzmin2020Dataset: SmfKuzmin2020Adapter,
    DmfKuzmin2020Dataset: DmfKuzmin2020Adapter,
    TmfKuzmin2020Dataset: TmfKuzmin2020Adapter,
    DmiKuzmin2020Dataset: DmiKuzmin2020Adapter,
    TmiKuzmin2020Dataset: TmiKuzmin2020Adapter,
    GeneEssentialitySgdDataset: GeneEssentialitySgdAdapter,
    SynthLethalityYeastSynthLethDbDataset: SynthLethalityYeastSynthLethDbAdapter,
    SynthRescueYeastSynthLethDbDataset: SynthRescueYeastSynthLethDbAdapter,
    ScmdOhya2005Dataset: ScmdOhya2005Adapter,
}
