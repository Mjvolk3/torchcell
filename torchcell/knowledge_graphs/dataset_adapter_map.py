"""Mapping from S. cerevisiae dataset classes to their BioCypher adapters."""

from torchcell.adapters import (
    AminoAcidMulleder2016Adapter,
    BetaxanthinCachera2023Adapter,
    CarotenoidOzaydin2013Adapter,
    CaudalPanTranscriptome2024Adapter,
    DmfCostanzo2016Adapter,
    DmfKuzmin2018Adapter,
    DmfKuzmin2020Adapter,
    DmiCostanzo2016Adapter,
    DmiKuzmin2018Adapter,
    DmiKuzmin2020Adapter,
    DmMicroarraySameith2015Adapter,
    FattyAcidXue2025Adapter,
    GeneEssentialitySgdAdapter,
    IsobutanolScreenLopez2024Adapter,
    IsobutanolValidatedLopez2024Adapter,
    MetaboliteDaSilveira2014Adapter,
    MetaboliteZelezniak2018Adapter,
    MicroarrayKemmeren2014Adapter,
    OrganicAcidYoshida2012Adapter,
    ProteomeZelezniak2018Adapter,
    ScmdOhnuki2018Adapter,
    ScmdOhnuki2022Adapter,
    SmfCostanzo2016Adapter,
    SmfKuzmin2018Adapter,
    SmfKuzmin2020Adapter,
    SmMicroarraySameith2015Adapter,
    SynthLethalityYeastSynthLethDbAdapter,
    SynthRescueYeastSynthLethDbAdapter,
    TmfKuzmin2018Adapter,
    TmfKuzmin2020Adapter,
    TmiKuzmin2018Adapter,
    TmiKuzmin2020Adapter,
)
from torchcell.adapters.ohya2005_adapter import ScmdOhya2005Adapter
from torchcell.datasets.scerevisiae.cachera2023 import BetaxanthinCachera2023Dataset
from torchcell.datasets.scerevisiae.caudal2024 import CaudalPanTranscriptome2024Dataset
from torchcell.datasets.scerevisiae.costanzo2016 import (
    DmfCostanzo2016Dataset,
    DmiCostanzo2016Dataset,
    SmfCostanzo2016Dataset,
)
from torchcell.datasets.scerevisiae.dasilveira2014 import (
    MetaboliteDaSilveira2014Dataset,
)
from torchcell.datasets.scerevisiae.kemmeren2014 import MicroarrayKemmeren2014Dataset
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
from torchcell.datasets.scerevisiae.lopez2024 import (
    IsobutanolScreenLopez2024Dataset,
    IsobutanolValidatedLopez2024Dataset,
)
from torchcell.datasets.scerevisiae.mulleder2016 import AminoAcidMulleder2016Dataset
from torchcell.datasets.scerevisiae.ohnuki2018 import ScmdOhnuki2018Dataset
from torchcell.datasets.scerevisiae.ohnuki2022 import ScmdOhnuki2022Dataset
from torchcell.datasets.scerevisiae.ohya2005 import ScmdOhya2005Dataset
from torchcell.datasets.scerevisiae.ozaydin2013 import CarotenoidOzaydin2013Dataset
from torchcell.datasets.scerevisiae.sameith2015 import (
    DmMicroarraySameith2015Dataset,
    SmMicroarraySameith2015Dataset,
)
from torchcell.datasets.scerevisiae.sgd import GeneEssentialitySgdDataset
from torchcell.datasets.scerevisiae.synth_leth_db import (
    SynthLethalityYeastSynthLethDbDataset,
    SynthRescueYeastSynthLethDbDataset,
)
from torchcell.datasets.scerevisiae.xue2025 import FattyAcidXue2025Dataset
from torchcell.datasets.scerevisiae.yoshida2012 import OrganicAcidYoshida2012Dataset
from torchcell.datasets.scerevisiae.zelezniak2018 import (
    MetaboliteZelezniak2018Dataset,
    ProteomeZelezniak2018Dataset,
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
    MicroarrayKemmeren2014Dataset: MicroarrayKemmeren2014Adapter,
    SmMicroarraySameith2015Dataset: SmMicroarraySameith2015Adapter,
    DmMicroarraySameith2015Dataset: DmMicroarraySameith2015Adapter,
    CaudalPanTranscriptome2024Dataset: CaudalPanTranscriptome2024Adapter,
    ScmdOhnuki2018Dataset: ScmdOhnuki2018Adapter,
    ScmdOhnuki2022Dataset: ScmdOhnuki2022Adapter,
    CarotenoidOzaydin2013Dataset: CarotenoidOzaydin2013Adapter,
    BetaxanthinCachera2023Dataset: BetaxanthinCachera2023Adapter,
    MetaboliteDaSilveira2014Dataset: MetaboliteDaSilveira2014Adapter,
    MetaboliteZelezniak2018Dataset: MetaboliteZelezniak2018Adapter,
    ProteomeZelezniak2018Dataset: ProteomeZelezniak2018Adapter,
    AminoAcidMulleder2016Dataset: AminoAcidMulleder2016Adapter,
    OrganicAcidYoshida2012Dataset: OrganicAcidYoshida2012Adapter,
    IsobutanolScreenLopez2024Dataset: IsobutanolScreenLopez2024Adapter,
    IsobutanolValidatedLopez2024Dataset: IsobutanolValidatedLopez2024Adapter,
    FattyAcidXue2025Dataset: FattyAcidXue2025Adapter,
}
