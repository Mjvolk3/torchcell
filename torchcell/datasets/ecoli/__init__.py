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

- ``cui2018`` -- ``CrispriKnockdownCui2018Dataset``: the genome-wide dCas9 knockdown
  screen, one record per (guide, screened strain) over the two dCas9-dose regimes.
- ``caglar2017`` -- ``ProteomeCaglar2017Dataset`` and ``RnaseqCaglar2017Dataset``: the
  REL606 multi-omic growth panel, which became loadable once the *E. coli* B assembly
  set joined the tier (this module began as the provenance record of that blocker).
- ``campos2018`` -- ``GrowthRateCampos2018Dataset``: the imaged Keio collection. Its
  released phenotype is 26 features of ONE medium (19 morphological, 2 growth, 5 cell
  cycle), of which only the Gompertz maximal growth rate has a phenotype class, served
  as a ko/wt growth-rate ratio; the module records the exact mismatch for the other 25.
- ``choe2025`` -- ``CrispriChemgenChoe2025Dataset``: the genome-scale MG1655 CRISPRi
  guide library scored against twelve antibiotics, one record per (guide, drug).
- ``fuhrer2017`` -- ``MetabolomeFuhrer2017Dataset``: the Keio deletion metabolome,
  FIA-TOF-MS ion z-scores per BW25113 strain (BioStudies S-BSST5).
- ``girgis2009`` -- ``EnvChemgenGirgis2009Dataset``: the 17-antibiotic transposon
  selection read by microarray genetic footprinting, one combined z-score per
  (MG1655 gene, drug).
- ``goodall2018`` -- ``GeneEssentialityGoodall2018Dataset``: BW25113 gene-level TraDIS
  essentiality calls.
- ``gupta2024`` -- ``ProteinTurnoverGupta2024Dataset``: per-protein total turnover in 13
  NCM3722 growth conditions, the first consumer of ``ProteinTurnoverPhenotype``.
- ``lamoureux2023`` -- ``RnaseqLamoureux2023Dataset``: PRECISE-1K, one record per MG1655
  RNA-seq library of the samples whose genotype and environment the release states.
- ``lamoureux2023_public_k12`` -- ``RnaseqPublicK12Lamoureux2023Dataset``: the Public K-12
  arm of the same release, one record per reprocessed public MG1655 RNA-seq library, keyed
  by its SRA experiment accession.
- ``shiver2016`` -- ``EnvChemgenShiver2016Dataset``: the neglected-antibiotic
  chemical-genomic screen, KEIO deletion fitness-scores across the 57 conditions of
  its own batches in the integrated S1 Dataset matrix.
- ``tong2020`` -- ``CarbonSourceTong2020Dataset``: Keio and sRNA-library deletion growth
  on thirty carbon sources, against two assembly pins.
- ``wang2018`` -- ``CrispriGuideFitnessWang2018Dataset``: the genome-scale pooled CRISPRi
  library, one signed log2 fitness per (guide, screen) over five screens.
- ``wang2015`` -- ``EnvChemgenWang2015Dataset``: Keio transporter deletions scored for
  isoprenol tolerance.
- ``rapp2026`` -- ``MetabolomeRapp2026Dataset``: the metabolome of a CRISPRi library
  covering every iML1515 gene, FI-MS feature fold changes per MG1655 b-number.
- ``price2018`` -- ``RbTnseqPrice2018EcoliDataset``: the RB-TnSeq fitness compendium,
  the loader that serves the experiments Wetmore 2015 first reported.
- ``rousset2018`` -- ``CrispriScreenRousset2018Dataset``: per-sgRNA dCas9 knockdown
  log2FC from the three phage challenges and the lambda transduction assay in FR-E01.
  Its fifth released screen, growth over 17 generations, is Cui 2018's screen released
  again (r 1.0000 over the 54,326 spacers they share), so it is accounted for in the
  retention ledger rather than stored twice.
- ``schastnaya2021`` -- ``MetabolomeSchastnaya2021Dataset``: the KEIO deletion arm of a
  phosphosite screen, FIA-TOF-MS ion log2 fold changes against an MG1655 wild type on
  four carbon sources. Its 171 phosphomutant columns are a genomic codon substitution,
  for which the schema has no bacterial perturbation leaf, so they are accounted for in
  the retention ledger rather than stored under a leaf that would misstate them.
- ``schmidt2016`` -- ``ProteomeSchmidt2016Dataset``: the condition-dependent BW25113
  proteome, one record per loaded growth condition in absolute protein copies per cell.
- ``mori2021`` -- ``ProteomeMori2021Dataset``: the DIA/SWATH absolute proteome, one
  record per loaded MG1655 (EQ353) calibration sample in protein mass fractions. Seven of
  its 66 released samples are loaded; the other 59 are dropped on their medium and
  ledgered, because only the calibration samples' Neidhardt MOPS minimal is an object
  ``MEDIA_LIBRARY`` states. Its values are an independent measurement of Schmidt 2016's
  quantity, not a re-release: zero of the 1,812 proteins their nearest conditions share
  agree to 1e-6, at Pearson r 0.796 on log10 mass fraction.
- ``mutalik2020`` -- ``PhageRbTnseqMutalik2020Dataset``: the phage-resistance RB-TnSeq
  screen, one record per (gene, experiment) over 68 phage challenges and 10 no-phage
  controls, each challenge's environment carrying a ``PhagePerturbation`` at the MOI
  the S13 Table states.
- ``wetmore2015`` -- a subsumption record: its *E. coli* experiments are carried by the
  Price 2018 compendium above, which is the loader that serves them.
"""

from .caglar2017 import ProteomeCaglar2017Dataset as ProteomeCaglar2017Dataset
from .caglar2017 import RnaseqCaglar2017Dataset as RnaseqCaglar2017Dataset
from .campos2018 import GrowthRateCampos2018Dataset as GrowthRateCampos2018Dataset
from .choe2025 import CrispriChemgenChoe2025Dataset as CrispriChemgenChoe2025Dataset
from .cui2018 import CrispriKnockdownCui2018Dataset as CrispriKnockdownCui2018Dataset
from .fuhrer2017 import MetabolomeFuhrer2017Dataset as MetabolomeFuhrer2017Dataset
from .girgis2009 import EnvChemgenGirgis2009Dataset as EnvChemgenGirgis2009Dataset
from .goodall2018 import (
    GeneEssentialityGoodall2018Dataset as GeneEssentialityGoodall2018Dataset,
)
from .gupta2024 import (
    ProteinTurnoverGupta2024Dataset as ProteinTurnoverGupta2024Dataset,
)
from .lamoureux2023 import RnaseqLamoureux2023Dataset as RnaseqLamoureux2023Dataset
from .lamoureux2023_public_k12 import (
    RnaseqPublicK12Lamoureux2023Dataset as RnaseqPublicK12Lamoureux2023Dataset,
)
from .mori2021 import ProteomeMori2021Dataset as ProteomeMori2021Dataset
from .mutalik2020 import (
    PhageRbTnseqMutalik2020Dataset as PhageRbTnseqMutalik2020Dataset,
)
from .price2018 import RbTnseqPrice2018EcoliDataset as RbTnseqPrice2018EcoliDataset
from .rapp2026 import MetabolomeRapp2026Dataset as MetabolomeRapp2026Dataset
from .rousset2018 import (
    CrispriScreenRousset2018Dataset as CrispriScreenRousset2018Dataset,
)
from .schastnaya2021 import (
    MetabolomeSchastnaya2021Dataset as MetabolomeSchastnaya2021Dataset,
)
from .schmidt2016 import ProteomeSchmidt2016Dataset as ProteomeSchmidt2016Dataset
from .shiver2016 import EnvChemgenShiver2016Dataset as EnvChemgenShiver2016Dataset
from .tong2020 import CarbonSourceTong2020Dataset as CarbonSourceTong2020Dataset
from .wang2015 import EnvChemgenWang2015Dataset as EnvChemgenWang2015Dataset
from .wang2018 import (
    CrispriGuideFitnessWang2018Dataset as CrispriGuideFitnessWang2018Dataset,
)
