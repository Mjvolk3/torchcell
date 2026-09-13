"""BioCypher adapters mapping torchcell datasets into knowledge-graph nodes and edges."""

from .auesukaree2009_adapter import (
    EnvChemgenAuesukaree2009Adapter as EnvChemgenAuesukaree2009Adapter,
)
from .baryshnikova2010_adapter import (
    SmfBaryshnikova2010Adapter as SmfBaryshnikova2010Adapter,
)
from .bloom2019_adapter import Bloom2019Adapter as Bloom2019Adapter
from .cachera2023_adapter import (
    BetaxanthinCachera2023Adapter as BetaxanthinCachera2023Adapter,
)
from .caudal2024_adapter import (
    CaudalPanTranscriptome2024Adapter as CaudalPanTranscriptome2024Adapter,
)
from .cell_adapter import CellAdapter as CellAdapter
from .costanzo2016_adapter import DmfCostanzo2016Adapter as DmfCostanzo2016Adapter
from .costanzo2016_adapter import DmiCostanzo2016Adapter as DmiCostanzo2016Adapter
from .costanzo2016_adapter import SmfCostanzo2016Adapter as SmfCostanzo2016Adapter
from .costanzo2021_adapter import (
    EnvChemgenCostanzo2021Adapter as EnvChemgenCostanzo2021Adapter,
)
from .dasilveira2014_adapter import (
    MetaboliteDaSilveira2014Adapter as MetaboliteDaSilveira2014Adapter,
)
from .hillenmeyer2008_adapter import (
    HetHillenmeyer2008Adapter as HetHillenmeyer2008Adapter,
)
from .hillenmeyer2008_adapter import (
    HomHillenmeyer2008Adapter as HomHillenmeyer2008Adapter,
)
from .hoepfner2014_adapter import (
    EnvChemgenHoepfner2014Adapter as EnvChemgenHoepfner2014Adapter,
)
from .kemmeren2014_adapter import (
    MicroarrayKemmeren2014Adapter as MicroarrayKemmeren2014Adapter,
)
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
from .lian2019_adapter import Lian2019Adapter as Lian2019Adapter
from .lopez2024_adapter import (
    IsobutanolScreenLopez2024Adapter as IsobutanolScreenLopez2024Adapter,
)
from .lopez2024_adapter import (
    IsobutanolValidatedLopez2024Adapter as IsobutanolValidatedLopez2024Adapter,
)
from .messner2023_adapter import (
    ProteomeMessner2023Adapter as ProteomeMessner2023Adapter,
)
from .mormino2022_adapter import Mormino2022Adapter as Mormino2022Adapter
from .mota2024_adapter import EnvChemgenMota2024Adapter as EnvChemgenMota2024Adapter
from .mulleder2016_adapter import (
    AminoAcidMulleder2016Adapter as AminoAcidMulleder2016Adapter,
)
from .nadal_ribelles2025_adapter import (
    NadalRibellesPerturbSeq2025Adapter as NadalRibellesPerturbSeq2025Adapter,
)
from .oduibhir2014_adapter import SmfODuibhir2014Adapter as SmfODuibhir2014Adapter
from .ohnuki2018_adapter import ScmdOhnuki2018Adapter as ScmdOhnuki2018Adapter
from .ohnuki2022_adapter import ScmdOhnuki2022Adapter as ScmdOhnuki2022Adapter
from .ohya2005_adapter import ScmdOhya2005Adapter as ScmdOhya2005Adapter
from .ozaydin2013_adapter import (
    CarotenoidOzaydin2013Adapter as CarotenoidOzaydin2013Adapter,
)
from .sameith2015_adapter import (
    DmMicroarraySameith2015Adapter as DmMicroarraySameith2015Adapter,
)
from .sameith2015_adapter import (
    SmMicroarraySameith2015Adapter as SmMicroarraySameith2015Adapter,
)
from .sgd_adapter import GeneEssentialitySgdAdapter as GeneEssentialitySgdAdapter
from .smith2006_adapter import Smith2006Adapter as Smith2006Adapter
from .smith2016_adapter import Smith2016Adapter as Smith2016Adapter
from .synth_leth_db_adapter import (
    SynthLethalityYeastSynthLethDbAdapter as SynthLethalityYeastSynthLethDbAdapter,
)
from .synth_leth_db_adapter import (
    SynthRescueYeastSynthLethDbAdapter as SynthRescueYeastSynthLethDbAdapter,
)
from .vanacloig2022_adapter import (
    EnvChemgenVanacloig2022Adapter as EnvChemgenVanacloig2022Adapter,
)
from .wildenhain2015_adapter import (
    EnvChemgenWildenhain2015Adapter as EnvChemgenWildenhain2015Adapter,
)
from .xue2025_adapter import FattyAcidXue2025Adapter as FattyAcidXue2025Adapter
from .yoshida2012_adapter import (
    OrganicAcidYoshida2012Adapter as OrganicAcidYoshida2012Adapter,
)
from .zelezniak2018_adapter import (
    MetaboliteZelezniak2018Adapter as MetaboliteZelezniak2018Adapter,
)
from .zelezniak2018_adapter import (
    ProteomeZelezniak2018Adapter as ProteomeZelezniak2018Adapter,
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

oduibhir2014_adapters = ["SmfODuibhir2014Adapter"]

expression_adapters = [
    "MicroarrayKemmeren2014Adapter",
    "SmMicroarraySameith2015Adapter",
    "DmMicroarraySameith2015Adapter",
    "CaudalPanTranscriptome2024Adapter",
    "NadalRibellesPerturbSeq2025Adapter",
]

morphology_adapters = ["ScmdOhnuki2018Adapter", "ScmdOhnuki2022Adapter"]

metabolite_adapters = [
    "CarotenoidOzaydin2013Adapter",
    "BetaxanthinCachera2023Adapter",
    "MetaboliteDaSilveira2014Adapter",
    "OrganicAcidYoshida2012Adapter",
    "IsobutanolScreenLopez2024Adapter",
    "IsobutanolValidatedLopez2024Adapter",
    "FattyAcidXue2025Adapter",
]

proteome_metabolome_adapters = [
    "MetaboliteZelezniak2018Adapter",
    "ProteomeZelezniak2018Adapter",
    "ProteomeMessner2023Adapter",
    "AminoAcidMulleder2016Adapter",
]

segregant_adapters = ["Bloom2019Adapter"]

environment_adapters = [
    "EnvChemgenCostanzo2021Adapter",
    "EnvChemgenAuesukaree2009Adapter",
    "EnvChemgenMota2024Adapter",
    "Smith2006Adapter",
    "Smith2016Adapter",
    "Lian2019Adapter",
    "Mormino2022Adapter",
    "EnvChemgenVanacloig2022Adapter",
    "EnvChemgenWildenhain2015Adapter",
    "EnvChemgenHoepfner2014Adapter",
    "HetHillenmeyer2008Adapter",
    "HomHillenmeyer2008Adapter",
]

baryshnikova_adapters = ["SmfBaryshnikova2010Adapter"]


__all__ = (
    cell_adapters
    + costanzo2016_adapters
    + kuzmin2018_adapters
    + kuzmin2020_adapters
    + gene_essentiality_adapters
    + synth_leth_db_adapters
    + ohya2005_adapters
    + oduibhir2014_adapters
    + expression_adapters
    + morphology_adapters
    + metabolite_adapters
    + proteome_metabolome_adapters
    + segregant_adapters
    + environment_adapters
    + baryshnikova_adapters
)
