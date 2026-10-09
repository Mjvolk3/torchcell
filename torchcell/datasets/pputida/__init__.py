# torchcell/datasets/pputida/__init__.py
# [[torchcell.datasets.pputida.__init__]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/pputida/__init__.py
"""P. putida KT2440 dataset loaders, one module per paper.

Each loader is ``<firstauthor><year>.py``, decorated with ``@register_dataset`` and
subclassing ``ExperimentDataset``; importing it here populates ``dataset_registry``.
The shared skeleton (genome, locus-tag reconciliation, assembly pin, injection rule)
is ``torchcell.datasets.bacteria_common``.

What this package holds:

- ``carruthers2025`` -- ``IsoprenolTiterCarruthers2025Dataset`` and
  ``ProteomeCarruthers2025Dataset``: the CRISPRi isoprenol production campaign, 465
  product-titer strains on the IY1449b chassis plus the released 19-sample Top3 proteome
  panel.
- ``desiqueira2025`` -- ``IsoprenolTiterDeSiqueira2025Dataset``,
  ``ProteomeDeSiqueira2025Dataset``, ``ProteomePercentDeSiqueira2025Dataset`` and
  ``ProteomeLog10PercentDeSiqueira2025Dataset``: the acetate-tolerization panel, whose
  proteome is released on three normalizations, loaded only where
  the genotype is representable (the wild type and the pre-tolerized parent).
- ``kang2026`` -- ``IsoprenylAcetateTiterKang2026Dataset``: the isoprenyl acetate
  production campaign on the PIPA chassis, across the paper's three titer columns.
- ``lim2022`` -- ``PutidaPrecise321Lim2022Dataset``: putidaPRECISE321, the KT2440
  transcriptome compendium as one record per sample (180 of 321), each naming the study
  its profile was first published in.
- ``menasalvas2025`` -- ``IsoprenolSelectionMenasalvas2025Dataset``: the
  biosensor-coupled CRISPRi selection, one record per enriched knockdown target.
"""

from .banerjee2025 import ProteomeBanerjee2025Dataset as ProteomeBanerjee2025Dataset
from .borchert2023 import RbTnseqBorchert2023Dataset as RbTnseqBorchert2023Dataset
from .borchert2024 import RbTnseqBorchert2024Dataset as RbTnseqBorchert2024Dataset
from .carruthers2025 import (
    IsoprenolTiterCarruthers2025Dataset as IsoprenolTiterCarruthers2025Dataset,
)
from .carruthers2025 import (
    ProteomeCarruthers2025Dataset as ProteomeCarruthers2025Dataset,
)
from .desiqueira2025 import (
    IsoprenolTiterDeSiqueira2025Dataset as IsoprenolTiterDeSiqueira2025Dataset,
)
from .desiqueira2025 import (
    ProteomeDeSiqueira2025Dataset as ProteomeDeSiqueira2025Dataset,
)
from .desiqueira2025 import (
    ProteomeLog10PercentDeSiqueira2025Dataset as ProteomeLog10PercentDeSiqueira2025Dataset,
)
from .desiqueira2025 import (
    ProteomePercentDeSiqueira2025Dataset as ProteomePercentDeSiqueira2025Dataset,
)
from .kang2026 import (
    IsoprenylAcetateTiterKang2026Dataset as IsoprenylAcetateTiterKang2026Dataset,
)
from .lim2022 import PutidaPrecise321Lim2022Dataset as PutidaPrecise321Lim2022Dataset
from .lim2025 import (
    IsoprenolToleranceLim2025Dataset as IsoprenolToleranceLim2025Dataset,
)
from .lim2025 import ProteomeLim2025Dataset as ProteomeLim2025Dataset
from .menasalvas2025 import (
    IsoprenolSelectionMenasalvas2025Dataset as IsoprenolSelectionMenasalvas2025Dataset,
)
from .yunus2026 import CrispriArrayYunus2026Dataset as CrispriArrayYunus2026Dataset
from .yunus2026 import (
    CrispriDifferentialProteomeYunus2026Dataset as CrispriDifferentialProteomeYunus2026Dataset,
)
from .yunus2026 import (
    CrispriKnockdownYunus2026Dataset as CrispriKnockdownYunus2026Dataset,
)
