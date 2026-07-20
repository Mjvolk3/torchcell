# torchcell/datasets/scerevisiae/__init__.py
# [[torchcell.datasets.scerevisiae.__init__]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/__init__.py
# Test file: tests/torchcell/datasets/scerevisiae/test___init__.py
"""S. cerevisiae fitness and morphology datasets (Costanzo, Kuzmin, Ohya, Sameith)."""

from .auesukaree2009 import (
    EnvChemgenAuesukaree2009Dataset as EnvChemgenAuesukaree2009Dataset,
)
from .cachera2023 import BetaxanthinCachera2023Dataset as BetaxanthinCachera2023Dataset
from .caudal2024 import (
    CaudalPanTranscriptome2024Dataset as CaudalPanTranscriptome2024Dataset,
)
from .costanzo2016 import DmfCostanzo2016Dataset as DmfCostanzo2016Dataset
from .costanzo2016 import SmfCostanzo2016Dataset as SmfCostanzo2016Dataset
from .costanzo2021 import EnvChemgenCostanzo2021Dataset as EnvChemgenCostanzo2021Dataset
from .dasilveira2014 import (
    MetaboliteDaSilveira2014Dataset as MetaboliteDaSilveira2014Dataset,
)
from .hillenmeyer2008 import HetHillenmeyer2008Dataset as HetHillenmeyer2008Dataset
from .hillenmeyer2008 import HomHillenmeyer2008Dataset as HomHillenmeyer2008Dataset
from .hoepfner2014 import EnvChemgenHoepfner2014Dataset as EnvChemgenHoepfner2014Dataset
from .kuzmin2018 import DmfKuzmin2018Dataset as DmfKuzmin2018Dataset
from .kuzmin2018 import SmfKuzmin2018Dataset as SmfKuzmin2018Dataset
from .kuzmin2018 import TmfKuzmin2018Dataset as TmfKuzmin2018Dataset
from .lian2019 import CrisprMagicLian2019Dataset as CrisprMagicLian2019Dataset
from .lopez2024 import (
    IsobutanolScreenLopez2024Dataset as IsobutanolScreenLopez2024Dataset,
)
from .lopez2024 import (
    IsobutanolValidatedLopez2024Dataset as IsobutanolValidatedLopez2024Dataset,
)
from .messner2023 import ProteomeMessner2023Dataset as ProteomeMessner2023Dataset
from .mormino2022 import CrispriMormino2022Dataset as CrispriMormino2022Dataset
from .mota2024 import EnvChemgenMota2024Dataset as EnvChemgenMota2024Dataset
from .mulleder2016 import AminoAcidMulleder2016Dataset as AminoAcidMulleder2016Dataset
from .nadal_ribelles2025 import (
    NadalRibellesPerturbSeq2025Dataset as NadalRibellesPerturbSeq2025Dataset,
)
from .oduibhir2014 import SmfODuibhir2014Dataset as SmfODuibhir2014Dataset
from .ohnuki2018 import ScmdOhnuki2018Dataset as ScmdOhnuki2018Dataset
from .ohnuki2022 import ScmdOhnuki2022Dataset as ScmdOhnuki2022Dataset
from .ohya2005 import ScmdOhya2005Dataset as ScmdOhya2005Dataset
from .ozaydin2013 import CarotenoidOzaydin2013Dataset as CarotenoidOzaydin2013Dataset
from .sameith2015 import (
    DmMicroarraySameith2015Dataset as DmMicroarraySameith2015Dataset,
)
from .sameith2015 import (
    SmMicroarraySameith2015Dataset as SmMicroarraySameith2015Dataset,
)
from .smith2006 import FattyAcidSmith2006Dataset as FattyAcidSmith2006Dataset
from .vanacloig2022 import (
    EnvChemgenVanacloig2022Dataset as EnvChemgenVanacloig2022Dataset,
)
from .wildenhain2015 import (
    EnvChemgenWildenhain2015Dataset as EnvChemgenWildenhain2015Dataset,
)
from .xue2025 import FattyAcidXue2025Dataset as FattyAcidXue2025Dataset
from .yeastphenome import YeastPhenomeDataset as YeastPhenomeDataset
from .yoshida2012 import OrganicAcidYoshida2012Dataset as OrganicAcidYoshida2012Dataset
from .zelezniak2018 import (
    MetaboliteZelezniak2018Dataset as MetaboliteZelezniak2018Dataset,
)
from .zelezniak2018 import ProteomeZelezniak2018Dataset as ProteomeZelezniak2018Dataset

# from .baryshnikovna2010 import Baryshnikovna2010Dataset

costanzo_datasets = ["SmfCostanzo2016Dataset", "DmfCostanzo2016Dataset"]

kuzmin_datasets = [
    "SmfKuzmin2018Dataset",
    "DmfKuzmin2018Dataset",
    "TmfKuzmin2018Dataset",
]

ohya_datasets = ["ScmdOhya2005Dataset"]

ohnuki_datasets = ["ScmdOhnuki2018Dataset", "ScmdOhnuki2022Dataset"]

sameith_datasets = ["SmMicroarraySameith2015Dataset", "DmMicroarraySameith2015Dataset"]

smith_datasets = ["FattyAcidSmith2006Dataset"]

lian_datasets = ["CrisprMagicLian2019Dataset"]

mormino_datasets = ["CrispriMormino2022Dataset"]

ozaydin_datasets = ["CarotenoidOzaydin2013Dataset"]

cachera_datasets = ["BetaxanthinCachera2023Dataset"]

mulleder_datasets = ["AminoAcidMulleder2016Dataset"]

zelezniak_datasets = ["ProteomeZelezniak2018Dataset", "MetaboliteZelezniak2018Dataset"]

messner_datasets = ["ProteomeMessner2023Dataset"]

vanacloig_datasets = ["EnvChemgenVanacloig2022Dataset"]

mota_datasets = ["EnvChemgenMota2024Dataset"]

wildenhain_datasets = ["EnvChemgenWildenhain2015Dataset"]

hoepfner_datasets = ["EnvChemgenHoepfner2014Dataset"]

auesukaree_datasets = ["EnvChemgenAuesukaree2009Dataset"]

costanzo2021_datasets = ["EnvChemgenCostanzo2021Dataset"]

hillenmeyer_datasets = ["HetHillenmeyer2008Dataset", "HomHillenmeyer2008Dataset"]

dasilveira_datasets = ["MetaboliteDaSilveira2014Dataset"]

yoshida_datasets = ["OrganicAcidYoshida2012Dataset"]

oduibhir_datasets = ["SmfODuibhir2014Dataset"]

lopez_datasets = [
    "IsobutanolScreenLopez2024Dataset",
    "IsobutanolValidatedLopez2024Dataset",
]

xue_datasets = ["FattyAcidXue2025Dataset"]

nadal_ribelles_datasets = ["NadalRibellesPerturbSeq2025Dataset"]

yeastphenome_datasets = ["YeastPhenomeDataset"]

__all__ = (
    costanzo_datasets
    + kuzmin_datasets
    + ohya_datasets
    + ohnuki_datasets
    + sameith_datasets
    + smith_datasets
    + ozaydin_datasets
    + cachera_datasets
    + mulleder_datasets
    + zelezniak_datasets
    + messner_datasets
    + vanacloig_datasets
    + mota_datasets
    + wildenhain_datasets
    + hoepfner_datasets
    + auesukaree_datasets
    + costanzo2021_datasets
    + hillenmeyer_datasets
    + dasilveira_datasets
    + yoshida_datasets
    + oduibhir_datasets
    + lian_datasets
    + mormino_datasets
    + lopez_datasets
    + xue_datasets
    + nadal_ribelles_datasets
    + yeastphenome_datasets
)
