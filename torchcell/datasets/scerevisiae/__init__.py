# torchcell/datasets/scerevisiae/__init__.py
# [[torchcell.datasets.scerevisiae.__init__]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/__init__.py
# Test file: tests/torchcell/datasets/scerevisiae/test___init__.py
"""S. cerevisiae fitness and morphology datasets (Costanzo, Kuzmin, Ohya, Sameith)."""

from .costanzo2016 import DmfCostanzo2016Dataset as DmfCostanzo2016Dataset
from .costanzo2016 import SmfCostanzo2016Dataset as SmfCostanzo2016Dataset
from .kuzmin2018 import DmfKuzmin2018Dataset as DmfKuzmin2018Dataset
from .kuzmin2018 import SmfKuzmin2018Dataset as SmfKuzmin2018Dataset
from .kuzmin2018 import TmfKuzmin2018Dataset as TmfKuzmin2018Dataset
from .ohya2005 import ScmdOhya2005Dataset as ScmdOhya2005Dataset
from .sameith2015 import (
    DmMicroarraySameith2015Dataset as DmMicroarraySameith2015Dataset,
)
from .sameith2015 import (
    SmMicroarraySameith2015Dataset as SmMicroarraySameith2015Dataset,
)

# from .baryshnikovna2010 import Baryshnikovna2010Dataset

costanzo_datasets = ["SmfCostanzo2016Dataset", "DmfCostanzo2016Dataset"]

kuzmin_datasets = [
    "SmfKuzmin2018Dataset",
    "DmfKuzmin2018Dataset",
    "TmfKuzmin2018Dataset",
]

ohya_datasets = ["ScmdOhya2005Dataset"]

sameith_datasets = ["SmMicroarraySameith2015Dataset", "DmMicroarraySameith2015Dataset"]

__all__ = costanzo_datasets + kuzmin_datasets + ohya_datasets + sameith_datasets
