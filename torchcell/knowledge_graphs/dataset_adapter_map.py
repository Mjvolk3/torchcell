"""Mapping from yeast and bacterial dataset classes to their BioCypher adapters."""

from torchcell.adapters import (
    AminoAcidCooper2010Adapter,
    AminoAcidMulleder2016Adapter,
    BetaxanthinCachera2023Adapter,
    Bloom2019Adapter,
    CarbonSourceTong2020Adapter,
    CarotenoidOzaydin2013Adapter,
    CaudalPanTranscriptome2024Adapter,
    CrispriArrayYunus2026Adapter,
    CrispriChemgenChoe2025Adapter,
    CrispriCrossRachwalski2024Adapter,
    CrispriDifferentialProteomeYunus2026Adapter,
    CrispriGuideFitnessWang2018Adapter,
    CrispriKnockdownCui2018Adapter,
    CrispriKnockdownYunus2026Adapter,
    CrispriScreenRousset2018Adapter,
    DmfCostanzo2016Adapter,
    DmfKuzmin2018Adapter,
    DmfKuzmin2020Adapter,
    DmiCostanzo2016Adapter,
    DmiKuzmin2018Adapter,
    DmiKuzmin2020Adapter,
    DmMicroarraySameith2015Adapter,
    EnvChemgenAuesukaree2009Adapter,
    EnvChemgenCostanzo2021Adapter,
    EnvChemgenGirgis2009Adapter,
    EnvChemgenHoepfner2014Adapter,
    EnvChemgenMota2024Adapter,
    EnvChemgenShiver2016Adapter,
    EnvChemgenVanacloig2022Adapter,
    EnvChemgenWang2015Adapter,
    EnvChemgenWildenhain2015Adapter,
    FattyAcidXue2025Adapter,
    GeneEssentialityGoodall2018Adapter,
    GeneEssentialityPrice2018EcoliAdapter,
    GeneEssentialitySgdAdapter,
    GeneInteractionBabu2014Adapter,
    GrowthAucRapp2026Adapter,
    GrowthRateCampos2018Adapter,
    GrowthRateChoe2019Adapter,
    GrowthRateSchmidt2016Adapter,
    HetHillenmeyer2008Adapter,
    HomHillenmeyer2008Adapter,
    IsobutanolScreenLopez2024Adapter,
    IsobutanolValidatedLopez2024Adapter,
    IsopentenolTiterFoo2014Adapter,
    IsoprenolSelectionMenasalvas2025Adapter,
    IsoprenolTiterCarruthers2025Adapter,
    IsoprenolTiterDeSiqueira2025Adapter,
    IsoprenolToleranceLim2025Adapter,
    IsoprenylAcetateTiterKang2026Adapter,
    Lian2019Adapter,
    MetaboliteDaSilveira2014Adapter,
    MetaboliteIntensityRapp2026Adapter,
    MetaboliteZelezniak2018Adapter,
    MetabolomeFuhrer2017Adapter,
    MetabolomeRapp2026Adapter,
    MetabolomeSchastnaya2021Adapter,
    MicroarrayKemmeren2014Adapter,
    Mormino2022Adapter,
    NadalRibellesPerturbSeq2025Adapter,
    OrganicAcidYoshida2012Adapter,
    PhageRbTnseqMutalik2020Adapter,
    ProteinTurnoverGupta2024Adapter,
    ProteomeBanerjee2025Adapter,
    ProteomeCaglar2017Adapter,
    ProteomeCarruthers2025Adapter,
    ProteomeDeSiqueira2025Adapter,
    ProteomeLim2025Adapter,
    ProteomeLog10PercentDeSiqueira2025Adapter,
    ProteomeMessner2023Adapter,
    ProteomeMori2021Adapter,
    ProteomePercentDeSiqueira2025Adapter,
    ProteomeSchmidt2016Adapter,
    ProteomeSrmSet1Schmidt2016Adapter,
    ProteomeSrmSet2Schmidt2016Adapter,
    ProteomeZelezniak2018Adapter,
    PutidaPrecise321Lim2022Adapter,
    RbTnseqBorchert2023Adapter,
    RbTnseqBorchert2024Adapter,
    RbTnseqPrice2018EcoliAdapter,
    RnaseqCaglar2017Adapter,
    RnaseqLamoureux2023Adapter,
    RnaseqPublicK12Lamoureux2023Adapter,
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
    TargetedMetabolomeRapp2026Adapter,
    TmfKuzmin2018Adapter,
    TmfKuzmin2020Adapter,
    TmiKuzmin2018Adapter,
    TmiKuzmin2020Adapter,
    TranscriptionFactorKnockoutChoe2019Adapter,
)
from torchcell.adapters.ohya2005_adapter import ScmdOhya2005Adapter
from torchcell.datasets.ecoli.babu2014 import GeneInteractionBabu2014Dataset
from torchcell.datasets.ecoli.caglar2017 import (
    ProteomeCaglar2017Dataset,
    RnaseqCaglar2017Dataset,
)
from torchcell.datasets.ecoli.campos2018 import GrowthRateCampos2018Dataset
from torchcell.datasets.ecoli.choe2019_growth_rate import (
    GrowthRateChoe2019Dataset,
    TranscriptionFactorKnockoutChoe2019Dataset,
)
from torchcell.datasets.ecoli.choe2025 import CrispriChemgenChoe2025Dataset
from torchcell.datasets.ecoli.cui2018 import CrispriKnockdownCui2018Dataset
from torchcell.datasets.ecoli.foo2014 import IsopentenolTiterFoo2014Dataset
from torchcell.datasets.ecoli.fuhrer2017 import MetabolomeFuhrer2017Dataset
from torchcell.datasets.ecoli.girgis2009 import EnvChemgenGirgis2009Dataset
from torchcell.datasets.ecoli.goodall2018 import GeneEssentialityGoodall2018Dataset
from torchcell.datasets.ecoli.gupta2024 import ProteinTurnoverGupta2024Dataset
from torchcell.datasets.ecoli.lamoureux2023 import RnaseqLamoureux2023Dataset
from torchcell.datasets.ecoli.lamoureux2023_public_k12 import (
    RnaseqPublicK12Lamoureux2023Dataset,
)
from torchcell.datasets.ecoli.mori2021 import ProteomeMori2021Dataset
from torchcell.datasets.ecoli.mutalik2020 import PhageRbTnseqMutalik2020Dataset
from torchcell.datasets.ecoli.price2018 import (
    GeneEssentialityPrice2018EcoliDataset,
    RbTnseqPrice2018EcoliDataset,
)
from torchcell.datasets.ecoli.rachwalski2024 import CrispriCrossRachwalski2024Dataset
from torchcell.datasets.ecoli.rapp2026 import MetabolomeRapp2026Dataset
from torchcell.datasets.ecoli.rapp2026_platforms import (
    GrowthAucRapp2026Dataset,
    MetaboliteIntensityRapp2026Dataset,
    TargetedMetabolomeRapp2026Dataset,
)
from torchcell.datasets.ecoli.rousset2018 import CrispriScreenRousset2018Dataset
from torchcell.datasets.ecoli.schastnaya2021 import MetabolomeSchastnaya2021Dataset
from torchcell.datasets.ecoli.schmidt2016 import ProteomeSchmidt2016Dataset
from torchcell.datasets.ecoli.schmidt2016_growth_rate import (
    GrowthRateSchmidt2016Dataset,
)
from torchcell.datasets.ecoli.schmidt2016_srm import (
    ProteomeSrmSet1Schmidt2016Dataset,
    ProteomeSrmSet2Schmidt2016Dataset,
)
from torchcell.datasets.ecoli.shiver2016 import EnvChemgenShiver2016Dataset
from torchcell.datasets.ecoli.tong2020 import CarbonSourceTong2020Dataset
from torchcell.datasets.ecoli.wang2015 import EnvChemgenWang2015Dataset
from torchcell.datasets.ecoli.wang2018 import CrispriGuideFitnessWang2018Dataset
from torchcell.datasets.pputida.banerjee2025 import ProteomeBanerjee2025Dataset
from torchcell.datasets.pputida.borchert2023 import RbTnseqBorchert2023Dataset
from torchcell.datasets.pputida.borchert2024 import RbTnseqBorchert2024Dataset
from torchcell.datasets.pputida.carruthers2025 import (
    IsoprenolTiterCarruthers2025Dataset,
    ProteomeCarruthers2025Dataset,
)
from torchcell.datasets.pputida.desiqueira2025 import (
    IsoprenolTiterDeSiqueira2025Dataset,
    ProteomeDeSiqueira2025Dataset,
    ProteomeLog10PercentDeSiqueira2025Dataset,
    ProteomePercentDeSiqueira2025Dataset,
)
from torchcell.datasets.pputida.kang2026 import IsoprenylAcetateTiterKang2026Dataset
from torchcell.datasets.pputida.lim2022 import PutidaPrecise321Lim2022Dataset
from torchcell.datasets.pputida.lim2025 import (
    IsoprenolToleranceLim2025Dataset,
    ProteomeLim2025Dataset,
)
from torchcell.datasets.pputida.menasalvas2025 import (
    IsoprenolSelectionMenasalvas2025Dataset,
)
from torchcell.datasets.pputida.yunus2026 import (
    CrispriArrayYunus2026Dataset,
    CrispriDifferentialProteomeYunus2026Dataset,
    CrispriKnockdownYunus2026Dataset,
)
from torchcell.datasets.scerevisiae.auesukaree2009 import (
    EnvChemgenAuesukaree2009Dataset,
)
from torchcell.datasets.scerevisiae.baryshnikova2010 import SmfBaryshnikova2010Dataset
from torchcell.datasets.scerevisiae.bloom2019 import Bloom2019Dataset
from torchcell.datasets.scerevisiae.cachera2023 import BetaxanthinCachera2023Dataset
from torchcell.datasets.scerevisiae.caudal2024 import CaudalPanTranscriptome2024Dataset
from torchcell.datasets.scerevisiae.cooper2010 import AminoAcidCooper2010Dataset
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
    AminoAcidCooper2010Dataset: AminoAcidCooper2010Adapter,
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
    # Bacteria (plan.bacteria-ontology-genome step 9): E. coli, then P. putida.
    GeneInteractionBabu2014Dataset: GeneInteractionBabu2014Adapter,
    RnaseqCaglar2017Dataset: RnaseqCaglar2017Adapter,
    ProteomeCaglar2017Dataset: ProteomeCaglar2017Adapter,
    CrispriChemgenChoe2025Dataset: CrispriChemgenChoe2025Adapter,
    CrispriKnockdownCui2018Dataset: CrispriKnockdownCui2018Adapter,
    GrowthRateCampos2018Dataset: GrowthRateCampos2018Adapter,
    GrowthRateChoe2019Dataset: GrowthRateChoe2019Adapter,
    IsopentenolTiterFoo2014Dataset: IsopentenolTiterFoo2014Adapter,
    TranscriptionFactorKnockoutChoe2019Dataset: TranscriptionFactorKnockoutChoe2019Adapter,
    MetabolomeFuhrer2017Dataset: MetabolomeFuhrer2017Adapter,
    EnvChemgenGirgis2009Dataset: EnvChemgenGirgis2009Adapter,
    GeneEssentialityGoodall2018Dataset: GeneEssentialityGoodall2018Adapter,
    ProteinTurnoverGupta2024Dataset: ProteinTurnoverGupta2024Adapter,
    RnaseqLamoureux2023Dataset: RnaseqLamoureux2023Adapter,
    RnaseqPublicK12Lamoureux2023Dataset: RnaseqPublicK12Lamoureux2023Adapter,
    ProteomeMori2021Dataset: ProteomeMori2021Adapter,
    PhageRbTnseqMutalik2020Dataset: PhageRbTnseqMutalik2020Adapter,
    RbTnseqPrice2018EcoliDataset: RbTnseqPrice2018EcoliAdapter,
    GeneEssentialityPrice2018EcoliDataset: GeneEssentialityPrice2018EcoliAdapter,
    CrispriCrossRachwalski2024Dataset: CrispriCrossRachwalski2024Adapter,
    MetabolomeRapp2026Dataset: MetabolomeRapp2026Adapter,
    GrowthAucRapp2026Dataset: GrowthAucRapp2026Adapter,
    MetaboliteIntensityRapp2026Dataset: MetaboliteIntensityRapp2026Adapter,
    TargetedMetabolomeRapp2026Dataset: TargetedMetabolomeRapp2026Adapter,
    CrispriScreenRousset2018Dataset: CrispriScreenRousset2018Adapter,
    MetabolomeSchastnaya2021Dataset: MetabolomeSchastnaya2021Adapter,
    EnvChemgenShiver2016Dataset: EnvChemgenShiver2016Adapter,
    ProteomeSchmidt2016Dataset: ProteomeSchmidt2016Adapter,
    ProteomeSrmSet1Schmidt2016Dataset: ProteomeSrmSet1Schmidt2016Adapter,
    ProteomeSrmSet2Schmidt2016Dataset: ProteomeSrmSet2Schmidt2016Adapter,
    GrowthRateSchmidt2016Dataset: GrowthRateSchmidt2016Adapter,
    CarbonSourceTong2020Dataset: CarbonSourceTong2020Adapter,
    EnvChemgenWang2015Dataset: EnvChemgenWang2015Adapter,
    CrispriGuideFitnessWang2018Dataset: CrispriGuideFitnessWang2018Adapter,
    ProteomeBanerjee2025Dataset: ProteomeBanerjee2025Adapter,
    RbTnseqBorchert2023Dataset: RbTnseqBorchert2023Adapter,
    RbTnseqBorchert2024Dataset: RbTnseqBorchert2024Adapter,
    IsoprenolTiterCarruthers2025Dataset: IsoprenolTiterCarruthers2025Adapter,
    ProteomeCarruthers2025Dataset: ProteomeCarruthers2025Adapter,
    ProteomeDeSiqueira2025Dataset: ProteomeDeSiqueira2025Adapter,
    ProteomePercentDeSiqueira2025Dataset: ProteomePercentDeSiqueira2025Adapter,
    ProteomeLog10PercentDeSiqueira2025Dataset: (
        ProteomeLog10PercentDeSiqueira2025Adapter
    ),
    IsoprenolTiterDeSiqueira2025Dataset: IsoprenolTiterDeSiqueira2025Adapter,
    IsoprenylAcetateTiterKang2026Dataset: IsoprenylAcetateTiterKang2026Adapter,
    PutidaPrecise321Lim2022Dataset: PutidaPrecise321Lim2022Adapter,
    IsoprenolToleranceLim2025Dataset: IsoprenolToleranceLim2025Adapter,
    ProteomeLim2025Dataset: ProteomeLim2025Adapter,
    IsoprenolSelectionMenasalvas2025Dataset: IsoprenolSelectionMenasalvas2025Adapter,
    CrispriArrayYunus2026Dataset: CrispriArrayYunus2026Adapter,
    CrispriDifferentialProteomeYunus2026Dataset: (
        CrispriDifferentialProteomeYunus2026Adapter
    ),
    CrispriKnockdownYunus2026Dataset: CrispriKnockdownYunus2026Adapter,
}
