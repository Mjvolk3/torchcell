"""BioCypher adapters mapping torchcell datasets into knowledge-graph nodes and edges."""

from .auesukaree2009_adapter import (
    EnvChemgenAuesukaree2009Adapter as EnvChemgenAuesukaree2009Adapter,
)
from .baba2006_adapter import (
    GeneEssentialityBaba2006Adapter as GeneEssentialityBaba2006Adapter,
)
from .baba2006_growth_adapter import (
    GrowthYieldBaba2006Adapter as GrowthYieldBaba2006Adapter,
)
from .babu2014_adapter import (
    GeneInteractionBabu2014Adapter as GeneInteractionBabu2014Adapter,
)
from .balakrishnan2022_adapter import (
    MrnaFractionBalakrishnan2022Adapter as MrnaFractionBalakrishnan2022Adapter,
)
from .banerjee2024_titer_adapter import (
    IsoprenolTiterBanerjee2024Adapter as IsoprenolTiterBanerjee2024Adapter,
)
from .banerjee2025_proteome_adapter import (
    ProteomeBanerjee2025Adapter as ProteomeBanerjee2025Adapter,
)
from .baryshnikova2010_adapter import (
    SmfBaryshnikova2010Adapter as SmfBaryshnikova2010Adapter,
)
from .bloom2019_adapter import Bloom2019Adapter as Bloom2019Adapter
from .boocock2025_adapter import Boocock2025Adapter as Boocock2025Adapter
from .borchert2023_adapter import (
    RbTnseqBorchert2023Adapter as RbTnseqBorchert2023Adapter,
)
from .borchert2024_adapter import (
    RbTnseqBorchert2024Adapter as RbTnseqBorchert2024Adapter,
)
from .brunk2016_exometabolite_adapter import (
    ExometaboliteBrunk2016Adapter as ExometaboliteBrunk2016Adapter,
)
from .brunk2016_metabolome_adapter import (
    MetabolomeBrunk2016Adapter as MetabolomeBrunk2016Adapter,
)
from .brunk2016_proteome_adapter import (
    ProteomeBrunk2016Adapter as ProteomeBrunk2016Adapter,
)
from .brunk2016_titer_adapter import (
    BiofuelTiterBrunk2016Adapter as BiofuelTiterBrunk2016Adapter,
)
from .butland2008_adapter import (
    GeneInteractionButland2008Adapter as GeneInteractionButland2008Adapter,
)
from .cachera2023_adapter import (
    BetaxanthinCachera2023Adapter as BetaxanthinCachera2023Adapter,
)
from .caglar2017_doubling_time_adapter import (
    DoublingTimeCaglar2017Adapter as DoublingTimeCaglar2017Adapter,
)
from .caglar2017_protein_fold_change_adapter import (
    ProteinFoldChangeCaglar2017Adapter as ProteinFoldChangeCaglar2017Adapter,
)
from .caglar2017_proteome_adapter import (
    ProteomeCaglar2017Adapter as ProteomeCaglar2017Adapter,
)
from .caglar2017_rnaseq_adapter import (
    RnaseqCaglar2017Adapter as RnaseqCaglar2017Adapter,
)
from .campos2018_adapter import (
    GrowthRateCampos2018Adapter as GrowthRateCampos2018Adapter,
)
from .campos2018_morphology_adapter import (
    MorphologyCampos2018Adapter as MorphologyCampos2018Adapter,
)
from .carruthers2025_campaign_proteome_adapter import (
    CampaignProteomeCarruthers2025Adapter as CampaignProteomeCarruthers2025Adapter,
)
from .carruthers2025_proteome_adapter import (
    ProteomeCarruthers2025Adapter as ProteomeCarruthers2025Adapter,
)
from .carruthers2025_proteome_fold_change_adapter import (
    ProteomeFoldChangeCarruthers2025Adapter as ProteomeFoldChangeCarruthers2025Adapter,
)
from .carruthers2025_titer_adapter import (
    IsoprenolTiterCarruthers2025Adapter as IsoprenolTiterCarruthers2025Adapter,
)
from .caudal2024_adapter import (
    CaudalPanTranscriptome2024Adapter as CaudalPanTranscriptome2024Adapter,
)
from .cell_adapter import CellAdapter as CellAdapter
from .choe2019_growth_rate_adapter import (
    GrowthRateChoe2019Adapter as GrowthRateChoe2019Adapter,
)
from .choe2019_tf_knockout_adapter import (
    TranscriptionFactorKnockoutChoe2019Adapter as TranscriptionFactorKnockoutChoe2019Adapter,
)
from .choe2025_adapter import (
    CrispriChemgenChoe2025Adapter as CrispriChemgenChoe2025Adapter,
)
from .cooper2010_adapter import AminoAcidCooper2010Adapter as AminoAcidCooper2010Adapter
from .costanzo2016_adapter import DmfCostanzo2016Adapter as DmfCostanzo2016Adapter
from .costanzo2016_adapter import DmiCostanzo2016Adapter as DmiCostanzo2016Adapter
from .costanzo2016_adapter import SmfCostanzo2016Adapter as SmfCostanzo2016Adapter
from .costanzo2021_adapter import (
    EnvChemgenCostanzo2021Adapter as EnvChemgenCostanzo2021Adapter,
)
from .cui2018_adapter import (
    CrispriKnockdownCui2018Adapter as CrispriKnockdownCui2018Adapter,
)
from .dasilveira2014_adapter import (
    MetaboliteDaSilveira2014Adapter as MetaboliteDaSilveira2014Adapter,
)
from .dechiara2022_adapter import (
    CellYieldDeChiara2022Adapter as CellYieldDeChiara2022Adapter,
)
from .dechiara2022_adapter import (
    DoublingTimeDeChiara2022Adapter as DoublingTimeDeChiara2022Adapter,
)
from .dechiara2022_adapter import (
    LifeCycleDeChiara2022Adapter as LifeCycleDeChiara2022Adapter,
)
from .desiqueira2025_proteome_adapter import (
    ProteomeDeSiqueira2025Adapter as ProteomeDeSiqueira2025Adapter,
)
from .desiqueira2025_proteome_log10_percent_adapter import (
    ProteomeLog10PercentDeSiqueira2025Adapter as ProteomeLog10PercentDeSiqueira2025Adapter,
)
from .desiqueira2025_proteome_percent_adapter import (
    ProteomePercentDeSiqueira2025Adapter as ProteomePercentDeSiqueira2025Adapter,
)
from .desiqueira2025_titer_adapter import (
    IsoprenolTiterDeSiqueira2025Adapter as IsoprenolTiterDeSiqueira2025Adapter,
)
from .fang2025_adapter import (
    CrispriGuideFfaEnrichmentFang2025Adapter as CrispriGuideFfaEnrichmentFang2025Adapter,
)
from .fitness_browser_ecoli_adapter import (
    RbTnseqFitnessBrowserEcoliAdapter as RbTnseqFitnessBrowserEcoliAdapter,
)
from .fitness_browser_putida_adapter import (
    RbTnseqFitnessBrowserPutidaAdapter as RbTnseqFitnessBrowserPutidaAdapter,
)
from .foo2014_adapter import (
    IsopentenolTiterFoo2014Adapter as IsopentenolTiterFoo2014Adapter,
)
from .fuhrer2017_adapter import (
    MetabolomeFuhrer2017Adapter as MetabolomeFuhrer2017Adapter,
)
from .gerdes2003_adapter import (
    GeneEssentialityGerdes2003Adapter as GeneEssentialityGerdes2003Adapter,
)
from .girgis2009_adapter import (
    EnvChemgenGirgis2009Adapter as EnvChemgenGirgis2009Adapter,
)
from .goodall2018_adapter import (
    GeneEssentialityGoodall2018Adapter as GeneEssentialityGoodall2018Adapter,
)
from .goodall2018_graded_adapter import (
    GradedGeneEssentialityGoodall2018Adapter as GradedGeneEssentialityGoodall2018Adapter,
)
from .gupta2024_adapter import (
    ProteinTurnoverGupta2024Adapter as ProteinTurnoverGupta2024Adapter,
)
from .guzman2018_adapter import RnaseqGuzman2018Adapter as RnaseqGuzman2018Adapter
from .hackett2020_adapter import (
    InductionTimecourseHackett2020Adapter as InductionTimecourseHackett2020Adapter,
)
from .hale2024_adapter import Hale2024Adapter as Hale2024Adapter
from .hawkins2020_adapter import (
    MismatchCrispriFitnessHawkins2020Adapter as MismatchCrispriFitnessHawkins2020Adapter,
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
from .hu2007_adapter import MicroarrayScoreHu2007Adapter as MicroarrayScoreHu2007Adapter
from .ishii2007_flux_adapter import FluxIshii2007Adapter as FluxIshii2007Adapter
from .ishii2007_metabolome_adapter import (
    MetabolomeIshii2007Adapter as MetabolomeIshii2007Adapter,
)
from .ishii2007_proteome_adapter import (
    ProteomeIshii2007Adapter as ProteomeIshii2007Adapter,
)
from .jackson2020_adapter import (
    TfDeletionPseudobulkJackson2020Adapter as TfDeletionPseudobulkJackson2020Adapter,
)
from .kang2026_adapter import (
    IsoprenylAcetateTiterKang2026Adapter as IsoprenylAcetateTiterKang2026Adapter,
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
from .lamoureux2023_adapter import (
    RnaseqLamoureux2023Adapter as RnaseqLamoureux2023Adapter,
)
from .lamoureux2023_growth_adapter import (
    GrowthRateLamoureux2023Adapter as GrowthRateLamoureux2023Adapter,
)
from .lamoureux2023_public_k12_adapter import (
    RnaseqPublicK12Lamoureux2023Adapter as RnaseqPublicK12Lamoureux2023Adapter,
)
from .li2014_adapter import (
    ProteinSynthesisRateLi2014Adapter as ProteinSynthesisRateLi2014Adapter,
)
from .li2021_flux_adapter import FluxLi2021Adapter as FluxLi2021Adapter
from .lian2019_adapter import Lian2019Adapter as Lian2019Adapter
from .lim2022_adapter import (
    PutidaPrecise321Lim2022Adapter as PutidaPrecise321Lim2022Adapter,
)
from .lim2025_proteome_adapter import ProteomeLim2025Adapter as ProteomeLim2025Adapter
from .lim2025_proteome_fold_change_adapter import (
    ProteomeFoldChangeLim2025Adapter as ProteomeFoldChangeLim2025Adapter,
)
from .lim2025_tolerance_adapter import (
    IsoprenolToleranceLim2025Adapter as IsoprenolToleranceLim2025Adapter,
)
from .lopez2024_adapter import (
    IsobutanolScreenLopez2024Adapter as IsobutanolScreenLopez2024Adapter,
)
from .lopez2024_adapter import (
    IsobutanolValidatedLopez2024Adapter as IsobutanolValidatedLopez2024Adapter,
)
from .menasalvas2025_adapter import (
    IsoprenolSelectionMenasalvas2025Adapter as IsoprenolSelectionMenasalvas2025Adapter,
)
from .menasalvas2025_metabolite_growth_adapter import (
    MetaboliteGrowthPhaseMenasalvas2025Adapter as MetaboliteGrowthPhaseMenasalvas2025Adapter,
)
from .menasalvas2025_metabolite_production_adapter import (
    MetaboliteProductionPhaseMenasalvas2025Adapter as MetaboliteProductionPhaseMenasalvas2025Adapter,
)
from .menasalvas2025_proteome_adapter import (
    ProteomeMenasalvas2025Adapter as ProteomeMenasalvas2025Adapter,
)
from .messner2021_adapter import (
    ProteomeMessner2021Adapter as ProteomeMessner2021Adapter,
)
from .messner2023_adapter import (
    ProteomeMessner2023Adapter as ProteomeMessner2023Adapter,
)
from .mohiuddin2022_adapter import (
    PromoterReporterMohiuddin2022Adapter as PromoterReporterMohiuddin2022Adapter,
)
from .mori2021_adapter import ProteomeMori2021Adapter as ProteomeMori2021Adapter
from .mormino2022_adapter import Mormino2022Adapter as Mormino2022Adapter
from .mota2024_adapter import EnvChemgenMota2024Adapter as EnvChemgenMota2024Adapter
from .muenzner2024_adapter import (
    ProteinTurnoverMuenzner2024Adapter as ProteinTurnoverMuenzner2024Adapter,
)
from .muenzner2024_adapter import (
    ProteomeMuenzner2024Adapter as ProteomeMuenzner2024Adapter,
)
from .mulleder2016_adapter import (
    AminoAcidMulleder2016Adapter as AminoAcidMulleder2016Adapter,
)
from .mutalik2020_adapter import (
    PhageRbTnseqMutalik2020Adapter as PhageRbTnseqMutalik2020Adapter,
)
from .nadal_ribelles2025_adapter import (
    NadalRibellesPerturbSeq2025Adapter as NadalRibellesPerturbSeq2025Adapter,
)
from .nguessan2025_adapter import Nguessan2025Adapter as Nguessan2025Adapter
from .nichols2011_adapter import (
    EnvChemgenNichols2011Adapter as EnvChemgenNichols2011Adapter,
)
from .niu2019_adapter import (
    CrisprPineneToleranceNiu2019Adapter as CrisprPineneToleranceNiu2019Adapter,
)
from .niu2019_pinene_adapter import (
    PineneFoldChangeNiu2019Adapter as PineneFoldChangeNiu2019Adapter,
)
from .oduibhir2014_adapter import SmfODuibhir2014Adapter as SmfODuibhir2014Adapter
from .ohnuki2018_adapter import ScmdOhnuki2018Adapter as ScmdOhnuki2018Adapter
from .ohnuki2022_adapter import ScmdOhnuki2022Adapter as ScmdOhnuki2022Adapter
from .ohya2005_adapter import ScmdOhya2005Adapter as ScmdOhya2005Adapter
from .ozaydin2013_adapter import (
    CarotenoidOzaydin2013Adapter as CarotenoidOzaydin2013Adapter,
)
from .parsons2006_adapter import (
    EnvChemgenParsons2006Adapter as EnvChemgenParsons2006Adapter,
)
from .peter2018_adapter import (
    IsolateGrowthPeter2018Adapter as IsolateGrowthPeter2018Adapter,
)
from .piotrowski2017_adapter import (
    EnvChemgenPiotrowski2017Adapter as EnvChemgenPiotrowski2017Adapter,
)
from .potel2018_adapter import (
    PhosphositeIntensityPotel2018Adapter as PhosphositeIntensityPotel2018Adapter,
)
from .price2018_ecoli_adapter import (
    RbTnseqPrice2018EcoliAdapter as RbTnseqPrice2018EcoliAdapter,
)
from .price2018_ecoli_essentiality_adapter import (
    GeneEssentialityPrice2018EcoliAdapter as GeneEssentialityPrice2018EcoliAdapter,
)
from .puddu2019_adapter import (
    GenomeArchitecturePuddu2019Adapter as GenomeArchitecturePuddu2019Adapter,
)
from .rachwalski2024_adapter import (
    CrispriCrossRachwalski2024Adapter as CrispriCrossRachwalski2024Adapter,
)
from .rapp2026_adapter import MetabolomeRapp2026Adapter as MetabolomeRapp2026Adapter
from .rapp2026_growth_adapter import (
    GrowthAucRapp2026Adapter as GrowthAucRapp2026Adapter,
)
from .rapp2026_intensity_adapter import (
    MetaboliteIntensityRapp2026Adapter as MetaboliteIntensityRapp2026Adapter,
)
from .rapp2026_targeted_adapter import (
    TargetedMetabolomeRapp2026Adapter as TargetedMetabolomeRapp2026Adapter,
)
from .rousset2018_adapter import (
    CrispriScreenRousset2018Adapter as CrispriScreenRousset2018Adapter,
)
from .royet2025_adapter import (
    EnvMetalTnseqRoyet2025Adapter as EnvMetalTnseqRoyet2025Adapter,
)
from .royet2025_graded_adapter import (
    GradedGeneEssentialityRoyet2025Adapter as GradedGeneEssentialityRoyet2025Adapter,
)
from .sameith2015_adapter import (
    DmMicroarraySameith2015Adapter as DmMicroarraySameith2015Adapter,
)
from .sameith2015_adapter import (
    SmMicroarraySameith2015Adapter as SmMicroarraySameith2015Adapter,
)
from .schastnaya2021_adapter import (
    MetabolomeSchastnaya2021Adapter as MetabolomeSchastnaya2021Adapter,
)
from .schmidt2016_adapter import (
    ProteomeSchmidt2016Adapter as ProteomeSchmidt2016Adapter,
)
from .schmidt2016_growth_rate_adapter import (
    GrowthRateSchmidt2016Adapter as GrowthRateSchmidt2016Adapter,
)
from .schmidt2016_s23_growth_rate_adapter import (
    GrowthRateS23Schmidt2016Adapter as GrowthRateS23Schmidt2016Adapter,
)
from .schmidt2016_srm_set1_adapter import (
    ProteomeSrmSet1Schmidt2016Adapter as ProteomeSrmSet1Schmidt2016Adapter,
)
from .schmidt2016_srm_set2_adapter import (
    ProteomeSrmSet2Schmidt2016Adapter as ProteomeSrmSet2Schmidt2016Adapter,
)
from .sgd_adapter import GeneEssentialitySgdAdapter as GeneEssentialitySgdAdapter
from .shiver2016_adapter import (
    EnvChemgenShiver2016Adapter as EnvChemgenShiver2016Adapter,
)
from .smith2006_adapter import Smith2006Adapter as Smith2006Adapter
from .smith2016_adapter import Smith2016Adapter as Smith2016Adapter
from .synth_leth_db_adapter import (
    SynthLethalityYeastSynthLethDbAdapter as SynthLethalityYeastSynthLethDbAdapter,
)
from .synth_leth_db_adapter import (
    SynthRescueYeastSynthLethDbAdapter as SynthRescueYeastSynthLethDbAdapter,
)
from .teteneva2024_adapter import (
    RbTnseqTeteneva2024Adapter as RbTnseqTeteneva2024Adapter,
)
from .thompson2019_valerolactam_growth_adapter import (
    LactamGrowthRateThompson2019Adapter as LactamGrowthRateThompson2019Adapter,
)
from .thompson2019_valerolactam_titer_adapter import (
    ValerolactamTiterThompson2019Adapter as ValerolactamTiterThompson2019Adapter,
)
from .tong2020_adapter import CarbonSourceTong2020Adapter as CarbonSourceTong2020Adapter
from .vaishnav2022_adapter import (
    PooledPromoterExpressionVaishnav2022Adapter as PooledPromoterExpressionVaishnav2022Adapter,
)
from .vanacloig2022_adapter import (
    EnvChemgenVanacloig2022Adapter as EnvChemgenVanacloig2022Adapter,
)
from .wang2015_adapter import EnvChemgenWang2015Adapter as EnvChemgenWang2015Adapter
from .wang2015_growth_adapter import GrowthWang2015Adapter as GrowthWang2015Adapter
from .wang2018_adapter import (
    CrispriGuideFitnessWang2018Adapter as CrispriGuideFitnessWang2018Adapter,
)
from .wang2024_adapter import EnvChemgenWang2024Adapter as EnvChemgenWang2024Adapter
from .wetmore2015_adapter import RbTnseqWetmore2015Adapter as RbTnseqWetmore2015Adapter
from .wildenhain2015_adapter import (
    EnvChemgenWildenhain2015Adapter as EnvChemgenWildenhain2015Adapter,
)
from .xue2025_adapter import FattyAcidXue2025Adapter as FattyAcidXue2025Adapter
from .yoshida2012_adapter import (
    OrganicAcidYoshida2012Adapter as OrganicAcidYoshida2012Adapter,
)
from .yoshikawa2009_adapter import (
    EnvChemgenYoshikawa2009Adapter as EnvChemgenYoshikawa2009Adapter,
)
from .yunus2026_array_adapter import (
    CrispriArrayYunus2026Adapter as CrispriArrayYunus2026Adapter,
)
from .yunus2026_differential_adapter import (
    CrispriDifferentialProteomeYunus2026Adapter as CrispriDifferentialProteomeYunus2026Adapter,
)
from .yunus2026_knockdown_adapter import (
    CrispriKnockdownYunus2026Adapter as CrispriKnockdownYunus2026Adapter,
)
from .yunus2026_panel_proteome_adapter import (
    CrispriPanelProteomeYunus2026Adapter as CrispriPanelProteomeYunus2026Adapter,
)
from .yunus2026_titer_adapter import (
    IsoprenolTiterYunus2026Adapter as IsoprenolTiterYunus2026Adapter,
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
    "MicroarrayScoreHu2007Adapter",
    "SmMicroarraySameith2015Adapter",
    "DmMicroarraySameith2015Adapter",
    "CaudalPanTranscriptome2024Adapter",
    "NadalRibellesPerturbSeq2025Adapter",
    "TfDeletionPseudobulkJackson2020Adapter",
    "InductionTimecourseHackett2020Adapter",
    "PooledPromoterExpressionVaishnav2022Adapter",
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
    "ProteomeMessner2021Adapter",
    "ProteomeMessner2023Adapter",
    "ProteomeMuenzner2024Adapter",
    "ProteinTurnoverMuenzner2024Adapter",
    "AminoAcidMulleder2016Adapter",
    "AminoAcidCooper2010Adapter",
]

segregant_adapters = [
    "Bloom2019Adapter",
    "Boocock2025Adapter",
    "Nguessan2025Adapter",
    "Hale2024Adapter",
]

natural_isolate_adapters = [
    "IsolateGrowthPeter2018Adapter",
    "LifeCycleDeChiara2022Adapter",
    "DoublingTimeDeChiara2022Adapter",
    "CellYieldDeChiara2022Adapter",
]

genome_architecture_adapters = ["GenomeArchitecturePuddu2019Adapter"]

environment_adapters = [
    "EnvChemgenCostanzo2021Adapter",
    "EnvChemgenAuesukaree2009Adapter",
    "EnvChemgenMota2024Adapter",
    "Smith2006Adapter",
    "Smith2016Adapter",
    "Lian2019Adapter",
    "Mormino2022Adapter",
    "EnvChemgenVanacloig2022Adapter",
    "EnvChemgenPiotrowski2017Adapter",
    "EnvChemgenWildenhain2015Adapter",
    "EnvChemgenHoepfner2014Adapter",
    "HetHillenmeyer2008Adapter",
    "HomHillenmeyer2008Adapter",
    "EnvChemgenParsons2006Adapter",
    "EnvChemgenYoshikawa2009Adapter",
]

baryshnikova_adapters = ["SmfBaryshnikova2010Adapter"]

ecoli_adapters = [
    "GeneInteractionBabu2014Adapter",
    "MrnaFractionBalakrishnan2022Adapter",
    "GeneInteractionButland2008Adapter",
    "RnaseqCaglar2017Adapter",
    "ProteomeCaglar2017Adapter",
    "DoublingTimeCaglar2017Adapter",
    "ProteinFoldChangeCaglar2017Adapter",
    "CrispriChemgenChoe2025Adapter",
    "CrispriKnockdownCui2018Adapter",
    "GrowthRateCampos2018Adapter",
    "MorphologyCampos2018Adapter",
    "GrowthRateChoe2019Adapter",
    "MetabolomeBrunk2016Adapter",
    "ExometaboliteBrunk2016Adapter",
    "ProteomeBrunk2016Adapter",
    "BiofuelTiterBrunk2016Adapter",
    "IsopentenolTiterFoo2014Adapter",
    "TranscriptionFactorKnockoutChoe2019Adapter",
    "MetabolomeFuhrer2017Adapter",
    "EnvChemgenGirgis2009Adapter",
    "GeneEssentialityBaba2006Adapter",
    "GrowthYieldBaba2006Adapter",
    "GeneEssentialityGerdes2003Adapter",
    "GeneEssentialityGoodall2018Adapter",
    "GradedGeneEssentialityGoodall2018Adapter",
    "RnaseqGuzman2018Adapter",
    "GeneEssentialityPrice2018EcoliAdapter",
    "ProteinTurnoverGupta2024Adapter",
    "ProteinSynthesisRateLi2014Adapter",
    "FluxLi2021Adapter",
    "MetabolomeIshii2007Adapter",
    "ProteomeIshii2007Adapter",
    "FluxIshii2007Adapter",
    "MismatchCrispriFitnessHawkins2020Adapter",
    "RnaseqLamoureux2023Adapter",
    "GrowthRateLamoureux2023Adapter",
    "PromoterReporterMohiuddin2022Adapter",
    "ProteomeMori2021Adapter",
    "PhosphositeIntensityPotel2018Adapter",
    "PhageRbTnseqMutalik2020Adapter",
    "CrisprPineneToleranceNiu2019Adapter",
    "PineneFoldChangeNiu2019Adapter",
    "RbTnseqPrice2018EcoliAdapter",
    "RbTnseqWetmore2015Adapter",
    "RbTnseqFitnessBrowserEcoliAdapter",
    "RbTnseqTeteneva2024Adapter",
    "CrispriCrossRachwalski2024Adapter",
    "MetabolomeRapp2026Adapter",
    "GrowthAucRapp2026Adapter",
    "MetaboliteIntensityRapp2026Adapter",
    "TargetedMetabolomeRapp2026Adapter",
    "CrispriScreenRousset2018Adapter",
    "MetabolomeSchastnaya2021Adapter",
    "EnvChemgenShiver2016Adapter",
    "EnvChemgenNichols2011Adapter",
    "ProteomeSchmidt2016Adapter",
    "GrowthRateS23Schmidt2016Adapter",
    "CarbonSourceTong2020Adapter",
    "EnvChemgenWang2015Adapter",
    "GrowthWang2015Adapter",
    "CrispriGuideFfaEnrichmentFang2025Adapter",
    "CrispriGuideFitnessWang2018Adapter",
    "EnvChemgenWang2024Adapter",
]

pputida_adapters = [
    "ProteomeBanerjee2025Adapter",
    "RbTnseqBorchert2023Adapter",
    "RbTnseqBorchert2024Adapter",
    "RbTnseqFitnessBrowserPutidaAdapter",
    "CampaignProteomeCarruthers2025Adapter",
    "IsoprenolTiterCarruthers2025Adapter",
    "ProteomeCarruthers2025Adapter",
    "ProteomeFoldChangeCarruthers2025Adapter",
    "ProteomeDeSiqueira2025Adapter",
    "ProteomeLog10PercentDeSiqueira2025Adapter",
    "ProteomePercentDeSiqueira2025Adapter",
    "IsoprenolTiterDeSiqueira2025Adapter",
    "IsoprenolTiterBanerjee2024Adapter",
    "IsoprenylAcetateTiterKang2026Adapter",
    "PutidaPrecise321Lim2022Adapter",
    "IsoprenolToleranceLim2025Adapter",
    "ProteomeLim2025Adapter",
    "ProteomeFoldChangeLim2025Adapter",
    "IsoprenolSelectionMenasalvas2025Adapter",
    "ValerolactamTiterThompson2019Adapter",
    "LactamGrowthRateThompson2019Adapter",
    "MetaboliteGrowthPhaseMenasalvas2025Adapter",
    "MetaboliteProductionPhaseMenasalvas2025Adapter",
    "ProteomeMenasalvas2025Adapter",
    "EnvMetalTnseqRoyet2025Adapter",
    "GradedGeneEssentialityRoyet2025Adapter",
    "CrispriArrayYunus2026Adapter",
    "CrispriDifferentialProteomeYunus2026Adapter",
    "CrispriKnockdownYunus2026Adapter",
    "CrispriPanelProteomeYunus2026Adapter",
    "IsoprenolTiterYunus2026Adapter",
]


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
    + natural_isolate_adapters
    + genome_architecture_adapters
    + environment_adapters
    + baryshnikova_adapters
    + ecoli_adapters
    + pputida_adapters
)
