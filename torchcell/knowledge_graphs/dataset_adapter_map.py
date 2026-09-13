"""Mapping from S. cerevisiae dataset classes to their BioCypher adapters."""

from torchcell.adapters import (
    AminoAcidMulleder2016Adapter,
    BetaxanthinCachera2023Adapter,
    Bloom2019Adapter,
    CarotenoidOzaydin2013Adapter,
    CaudalPanTranscriptome2024Adapter,
    DmfCostanzo2016Adapter,
    DmfKuzmin2018Adapter,
    DmfKuzmin2020Adapter,
    DmiCostanzo2016Adapter,
    DmiKuzmin2018Adapter,
    DmiKuzmin2020Adapter,
    DmMicroarraySameith2015Adapter,
    EnvChemgenAuesukaree2009Adapter,
    EnvChemgenCostanzo2021Adapter,
    EnvChemgenHoepfner2014Adapter,
    EnvChemgenMota2024Adapter,
    EnvChemgenVanacloig2022Adapter,
    EnvChemgenWildenhain2015Adapter,
    FattyAcidXue2025Adapter,
    GeneEssentialitySgdAdapter,
    HetHillenmeyer2008Adapter,
    HomHillenmeyer2008Adapter,
    IsobutanolScreenLopez2024Adapter,
    IsobutanolValidatedLopez2024Adapter,
    Lian2019Adapter,
    MetaboliteDaSilveira2014Adapter,
    MetaboliteZelezniak2018Adapter,
    MicroarrayKemmeren2014Adapter,
    Mormino2022Adapter,
    NadalRibellesPerturbSeq2025Adapter,
    OrganicAcidYoshida2012Adapter,
    ProteomeMessner2023Adapter,
    ProteomeZelezniak2018Adapter,
    ScmdOhnuki2018Adapter,
    ScmdOhnuki2022Adapter,
    SmfBaryshnikova2010Adapter,
    SmfCostanzo2016Adapter,
    SmfKuzmin2018Adapter,
    SmfKuzmin2020Adapter,
    SmfODuibhir2014Adapter,
    Smith2006Adapter,
    Smith2016Adapter,
    SmMicroarraySameith2015Adapter,
    SynthLethalityYeastSynthLethDbAdapter,
    SynthRescueYeastSynthLethDbAdapter,
    TmfKuzmin2018Adapter,
    TmfKuzmin2020Adapter,
    TmiKuzmin2018Adapter,
    TmiKuzmin2020Adapter,
)
from torchcell.adapters.ohya2005_adapter import ScmdOhya2005Adapter
from torchcell.datasets.scerevisiae.auesukaree2009 import (
    EnvChemgenAuesukaree2009Dataset,
)
from torchcell.datasets.scerevisiae.baryshnikova2010 import SmfBaryshnikova2010Dataset
from torchcell.datasets.scerevisiae.bloom2019 import Bloom2019Dataset
from torchcell.datasets.scerevisiae.cachera2023 import BetaxanthinCachera2023Dataset
from torchcell.datasets.scerevisiae.caudal2024 import CaudalPanTranscriptome2024Dataset
from torchcell.datasets.scerevisiae.costanzo2016 import (
    DmfCostanzo2016Dataset,
    DmiCostanzo2016Dataset,
    SmfCostanzo2016Dataset,
)
from torchcell.datasets.scerevisiae.costanzo2021 import EnvChemgenCostanzo2021Dataset
from torchcell.datasets.scerevisiae.dasilveira2014 import (
    MetaboliteDaSilveira2014Dataset,
)
from torchcell.datasets.scerevisiae.hillenmeyer2008 import (
    HetHillenmeyer2008Dataset,
    HomHillenmeyer2008Dataset,
)
from torchcell.datasets.scerevisiae.hoepfner2014 import EnvChemgenHoepfner2014Dataset
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
from torchcell.datasets.scerevisiae.lian2019 import CrisprMagicLian2019Dataset
from torchcell.datasets.scerevisiae.lopez2024 import (
    IsobutanolScreenLopez2024Dataset,
    IsobutanolValidatedLopez2024Dataset,
)
from torchcell.datasets.scerevisiae.messner2023 import ProteomeMessner2023Dataset
from torchcell.datasets.scerevisiae.mormino2022 import CrispriMormino2022Dataset
from torchcell.datasets.scerevisiae.mota2024 import EnvChemgenMota2024Dataset
from torchcell.datasets.scerevisiae.mulleder2016 import AminoAcidMulleder2016Dataset
from torchcell.datasets.scerevisiae.nadal_ribelles2025 import (
    NadalRibellesPerturbSeq2025Dataset,
)
from torchcell.datasets.scerevisiae.oduibhir2014 import SmfODuibhir2014Dataset
from torchcell.datasets.scerevisiae.ohnuki2018 import ScmdOhnuki2018Dataset
from torchcell.datasets.scerevisiae.ohnuki2022 import ScmdOhnuki2022Dataset
from torchcell.datasets.scerevisiae.ohya2005 import ScmdOhya2005Dataset
from torchcell.datasets.scerevisiae.ozaydin2013 import CarotenoidOzaydin2013Dataset
from torchcell.datasets.scerevisiae.sameith2015 import (
    DmMicroarraySameith2015Dataset,
    SmMicroarraySameith2015Dataset,
)
from torchcell.datasets.scerevisiae.sgd import GeneEssentialitySgdDataset
from torchcell.datasets.scerevisiae.smith2006 import FattyAcidSmith2006Dataset
from torchcell.datasets.scerevisiae.smith2016 import CrispriChemgenSmith2016Dataset
from torchcell.datasets.scerevisiae.synth_leth_db import (
    SynthLethalityYeastSynthLethDbDataset,
    SynthRescueYeastSynthLethDbDataset,
)
from torchcell.datasets.scerevisiae.vanacloig2022 import EnvChemgenVanacloig2022Dataset
from torchcell.datasets.scerevisiae.wildenhain2015 import (
    EnvChemgenWildenhain2015Dataset,
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
    SmfODuibhir2014Dataset: SmfODuibhir2014Adapter,
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
    ProteomeMessner2023Dataset: ProteomeMessner2023Adapter,
    ProteomeZelezniak2018Dataset: ProteomeZelezniak2018Adapter,
    AminoAcidMulleder2016Dataset: AminoAcidMulleder2016Adapter,
    OrganicAcidYoshida2012Dataset: OrganicAcidYoshida2012Adapter,
    IsobutanolScreenLopez2024Dataset: IsobutanolScreenLopez2024Adapter,
    IsobutanolValidatedLopez2024Dataset: IsobutanolValidatedLopez2024Adapter,
    FattyAcidXue2025Dataset: FattyAcidXue2025Adapter,
    NadalRibellesPerturbSeq2025Dataset: NadalRibellesPerturbSeq2025Adapter,
    Bloom2019Dataset: Bloom2019Adapter,
    SmfBaryshnikova2010Dataset: SmfBaryshnikova2010Adapter,
    EnvChemgenCostanzo2021Dataset: EnvChemgenCostanzo2021Adapter,
    EnvChemgenAuesukaree2009Dataset: EnvChemgenAuesukaree2009Adapter,
    EnvChemgenMota2024Dataset: EnvChemgenMota2024Adapter,
    FattyAcidSmith2006Dataset: Smith2006Adapter,
    CrispriChemgenSmith2016Dataset: Smith2016Adapter,
    CrisprMagicLian2019Dataset: Lian2019Adapter,
    CrispriMormino2022Dataset: Mormino2022Adapter,
    EnvChemgenVanacloig2022Dataset: EnvChemgenVanacloig2022Adapter,
    EnvChemgenWildenhain2015Dataset: EnvChemgenWildenhain2015Adapter,
    EnvChemgenHoepfner2014Dataset: EnvChemgenHoepfner2014Adapter,
    HetHillenmeyer2008Dataset: HetHillenmeyer2008Adapter,
    HomHillenmeyer2008Dataset: HomHillenmeyer2008Adapter,
}
