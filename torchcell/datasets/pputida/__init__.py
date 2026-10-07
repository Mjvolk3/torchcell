# torchcell/datasets/pputida/__init__.py
# [[torchcell.datasets.pputida.__init__]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/pputida/__init__.py
"""P. putida KT2440 dataset loaders, one module per paper.

Each loader is ``<firstauthor><year>.py``, decorated with ``@register_dataset`` and
subclassing ``ExperimentDataset``; importing it here populates ``dataset_registry``.
The shared skeleton (genome, locus-tag reconciliation, assembly pin, injection rule)
is ``torchcell.datasets.bacteria_common``.
"""

from .carruthers2025 import (
    IsoprenolTiterCarruthers2025Dataset as IsoprenolTiterCarruthers2025Dataset,
)
from .carruthers2025 import (
    ProteomeCarruthers2025Dataset as ProteomeCarruthers2025Dataset,
)
from .lim2022 import PutidaPrecise321Lim2022Dataset as PutidaPrecise321Lim2022Dataset
