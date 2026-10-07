# torchcell/datasets/ecoli/__init__.py
# [[torchcell.datasets.ecoli.__init__]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/__init__.py
"""E. coli dataset loaders (K-12 MG1655 and BW25113, B REL606), one module per paper.

Each loader is ``<firstauthor><year>.py``, decorated with ``@register_dataset`` and
subclassing ``ExperimentDataset``; importing it here populates ``dataset_registry``.
The shared skeleton (genome, locus-tag reconciliation, assembly pin, injection rule)
is ``torchcell.datasets.bacteria_common``.

What this package holds. A module may carry a dataset class, a provenance record, or
both: a row whose data another row's loader subsumes, or whose records the schema cannot
yet express, is a RECORD of that finding rather than nothing, and is imported for its
sourcing layer.

- ``caglar2017`` -- ``ProteomeCaglar2017Dataset`` and ``RnaseqCaglar2017Dataset``: the
  REL606 multi-omic growth panel, which became loadable once the *E. coli* B assembly
  set joined the tier (this module began as the provenance record of that blocker).
- ``fuhrer2017`` -- ``MetabolomeFuhrer2017Dataset``: the Keio deletion metabolome,
  FIA-TOF-MS ion z-scores per BW25113 strain (BioStudies S-BSST5).
- ``goodall2018`` -- ``GeneEssentialityGoodall2018Dataset``: BW25113 gene-level TraDIS
  essentiality calls.
- ``lamoureux2023`` -- ``RnaseqLamoureux2023Dataset``: PRECISE-1K, one record per MG1655
  RNA-seq library of the samples whose genotype and environment the release states.
- ``tong2020`` -- ``CarbonSourceTong2020Dataset``: Keio and sRNA-library deletion growth
  on thirty carbon sources, against two assembly pins.
- ``wang2015`` -- ``EnvChemgenWang2015Dataset``: Keio transporter deletions scored for
  isoprenol tolerance.
- ``mutalik2020`` -- the phage-resistance RB-TnSeq raw mirror and sourcing layer; the
  dataset class waits on a loader using the ``PhagePerturbation`` environment leaf.
- ``wetmore2015`` -- a subsumption record: its *E. coli* experiments are carried by the
  Price 2018 compendium, which is the loader that will serve them.
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
