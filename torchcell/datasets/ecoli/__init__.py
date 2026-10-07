# torchcell/datasets/ecoli/__init__.py
# [[torchcell.datasets.ecoli.__init__]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/__init__.py
"""E. coli dataset loaders (K-12 MG1655 and BW25113, B REL606), one module per paper.

Each loader is ``<firstauthor><year>.py``, decorated with ``@register_dataset`` and
subclassing ``ExperimentDataset``; importing it here populates ``dataset_registry``.
The shared skeleton (genome, locus-tag reconciliation, assembly pin, injection rule)
is ``torchcell.datasets.bacteria_common``.
"""

from .caglar2017 import ProteomeCaglar2017Dataset as ProteomeCaglar2017Dataset
from .caglar2017 import RnaseqCaglar2017Dataset as RnaseqCaglar2017Dataset
from .fuhrer2017 import MetabolomeFuhrer2017Dataset as MetabolomeFuhrer2017Dataset
from .goodall2018 import (
    GeneEssentialityGoodall2018Dataset as GeneEssentialityGoodall2018Dataset,
)
from .lamoureux2023 import RnaseqLamoureux2023Dataset as RnaseqLamoureux2023Dataset
from .price2018 import RbTnseqPrice2018EcoliDataset as RbTnseqPrice2018EcoliDataset
from .tong2020 import CarbonSourceTong2020Dataset as CarbonSourceTong2020Dataset
from .wang2015 import EnvChemgenWang2015Dataset as EnvChemgenWang2015Dataset
