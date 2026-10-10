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
- ``babu2014`` -- ``GeneInteractionBabu2014Dataset``: the genome-wide eSGA digenic
  interaction map, one record per released (donor, recipient) pair with its signed
  colony-size S score. The first consumer of ``BacterialGeneInteractionExperiment``,
  and the loader that re-releases 727 of Butland 2008's measurements (every record
  names its screen set in ``screen_id``); it does NOT subsume that release, which
  ``butland2008`` below serves in full. Its 3,420 hypomorph-involving pairs are dropped:
  a
  3'-UTR cassette hypomorph has no bacterial gene-perturbation leaf.
- ``butland2008`` -- ``GeneInteractionButland2008Dataset``: the UNFILTERED eSGA
  matrix those 39 screens actually released, one record per (query strain, recipient
  isolate) cell of Supplementary Table 4. It is NOT subsumed by Babu 2014, which carries
  727 of these 314,847 scores, and the build proves the partition against the served
  Babu store in both directions before writing a record. Its 149 SPA-tag recipient rows
  share Babu's hypomorph blocker and its two colony-size sheets have no phenotype class.
- ``caglar2017`` -- ``ProteomeCaglar2017Dataset``, ``RnaseqCaglar2017Dataset`` and
  ``ProteinFoldChangeCaglar2017Dataset``: the
  REL606 multi-omic growth panel, which became loadable once the *E. coli* B assembly
  set joined the tier (this module began as the provenance record of that blocker).
- ``caglar2017_doubling_time`` -- ``DoublingTimeCaglar2017Dataset``: the same paper's
  Table S5, 55 absolute doubling times, one per released biological-replicate growth
  curve. A separate module so the served RNA-seq and proteome stores' schema closure
  does not move; the dataset is what the environment-response phenotype's interval
  carrier, replicate id and absolute-reference branch were added for (#776).
- ``brunk2016`` -- ``MetabolomeBrunk2016Dataset``, ``ExometaboliteBrunk2016Dataset``, ``ProteomeBrunk2016Dataset`` and
  ``BiofuelTiterBrunk2016Dataset``: the four released scales of one batch
  fermentation of nine mevalonate-pathway DH1 strains sampled 0 to 72 h. The
  sampling hour is on the environment, so each strain contributes one record per
  sample rather than one record in total.
- ``campos2018`` -- ``GrowthRateCampos2018Dataset`` and
  ``MorphologyCampos2018Dataset``: the imaged Keio collection, split by readout. Its
  released phenotype is 28 named features of ONE medium (Appendix Table S1's 21
  morphological, 2 growth and 5 cell cycle symbols); the Gompertz maximal growth rate is
  served as a ko/wt growth-rate ratio and the 26 morphology symbols as a
  ``BacterialMorphologyPhenotype`` against the ``campos2018`` assay vocabulary. Only the
  saturating optical density has no phenotype class left.
- ``choe2019_growth_rate`` -- ``GrowthRateChoe2019Dataset`` and
  ``TranscriptionFactorKnockoutChoe2019Dataset``: the two writable arms of the
  genome-reduced ALE campaign. The first serves the two designed deletions built on the
  reduced parent MS56 as parent-relative fitness, the second the two Keio BW25113
  deletions whose growth rate the Supplementary Fig. 6 legend states. Every evolved
  strain is refused: the release gives population allele frequencies rather than clone
  genotypes, and no perturbation leaf holds a called bacterial variant.
- ``choe2025`` -- ``CrispriChemgenChoe2025Dataset``: the genome-scale MG1655 CRISPRi
  guide library scored against twelve antibiotics, one record per (guide, drug).
- ``foo2014`` -- ``IsopentenolTiterFoo2014Dataset``: the nine released isopentenol
  titers of the tolerance-engineering campaign, one per production strain on the PS
  chassis. Isopentenol is isoprenol under its older name, so these are the only
  released E. coli isoprenol titers; its GEO series is one strain under two
  isopentenol doses and so belongs to a separate class, not this one.
- ``fuhrer2017`` -- ``MetabolomeFuhrer2017Dataset``: the Keio deletion metabolome,
  FIA-TOF-MS ion z-scores per BW25113 strain (BioStudies S-BSST5).
- ``girgis2009`` -- ``EnvChemgenGirgis2009Dataset``: the 17-antibiotic transposon
  selection read by microarray genetic footprinting, one combined z-score per
  (MG1655 gene, drug).
- ``goodall2018`` -- ``GeneEssentialityGoodall2018Dataset``: BW25113 gene-level TraDIS
  essentiality calls.
- ``gupta2024`` -- ``ProteinTurnoverGupta2024Dataset``: per-protein total turnover in 13
  NCM3722 growth conditions, the first consumer of ``ProteinTurnoverPhenotype``.
- ``hawkins2020`` -- ``MismatchCrispriFitnessHawkins2020Dataset``: mismatch-CRISPRi, one
  record per released sgRNA with a measured relative fitness over 317 BW25113 essential
  genes. The guide's mismatch design rides on the perturbation's description; the
  released PREDICTED sgRNA activity is a model output and is not stored.
- ``lamoureux2023`` -- ``RnaseqLamoureux2023Dataset``: PRECISE-1K, one record per MG1655
  RNA-seq library of the samples whose genotype and environment the release states.
- ``lamoureux2023_growth`` -- ``GrowthRateLamoureux2023Dataset``: the SAME release's
  ``Growth Rate (1/hr)`` metadata column, as 89 absolute ``EnvironmentResponsePhenotype``
  rates against a declared base condition. 354 cells are released, the expression
  loader's own genotype and environment rules keep 103, and the 14 that read exactly 0.0
  are dropped as indistinguishable from an unrecorded cell.
- ``lamoureux2023_public_k12`` -- ``RnaseqPublicK12Lamoureux2023Dataset``: the Public K-12
  arm of the same release, one record per reprocessed public MG1655 RNA-seq library, keyed
  by its SRA experiment accession.
- ``niu2019`` -- ``CrisprPineneToleranceNiu2019Dataset``: CRISPRa and CRISPRi of the
  pinene-response genes in the designed parent BW25113(PT5-dxs), one record per stored
  strain's OD600 ratio against the no-guide control under 0.5% pinene.
- ``shiver2016`` -- ``EnvChemgenShiver2016Dataset``: the neglected-antibiotic
  chemical-genomic screen, KEIO deletion fitness-scores across the 57 conditions of
  its own batches in the integrated S1 Dataset matrix.
- ``tong2020`` -- ``CarbonSourceTong2020Dataset``: Keio and sRNA-library deletion growth
  on thirty carbon sources, against two assembly pins.
- ``fang2025`` -- ``CrispriGuideFfaEnrichmentFang2025Dataset``: the two-round
  CRISPRi-FACS free-fatty-acid enrichment screen over the SAME 55,671-guide library
  ``wang2018`` serves, 15,708 records.
- ``wang2018`` -- ``CrispriGuideFitnessWang2018Dataset``: the genome-scale pooled CRISPRi
  library, one signed log2 fitness per (guide, screen) over five screens.
- ``wang2015`` -- ``EnvChemgenWang2015Dataset``: Keio transporter deletions scored for
  isoprenol tolerance.
- ``wang2015_growth`` -- ``GrowthWang2015Dataset``: the SAME Table S3's other column,
  the plain-medium endpoint OD600, as 46 ``FitnessPhenotype`` ratios of each deletion to
  BW25113. It is not a second copy of the isoprenol arm: the stored log2 there is a
  function of both columns, and measured over the 46 mutants neither column is
  recoverable from it.
- ``rapp2026`` -- ``MetabolomeRapp2026Dataset``: the metabolome of a CRISPRi library
  covering every iML1515 gene, FI-MS feature fold changes per MG1655 b-number.
- ``rapp2026_platforms`` -- the same release's three other per-strain families, each on
  its own scale and so its own dataset: ``GrowthAucRapp2026Dataset`` (the paper's
  trapezoid AUC over the released OD600 curves, against the control strains),
  ``TargetedMetabolomeRapp2026Dataset`` (the targeted LC-MS/MS screen, a second
  platform) and ``MetaboliteIntensityRapp2026Dataset`` (the absolute FI-MS intensity of
  each accumulating feature, with its per-replicate standard error).
- ``rachwalski2024`` -- ``CrispriCrossRachwalski2024Dataset``: the mobile CRISPRi
  collection's normalized colony growth, crossed into the lpp deletion and into the whole
  Keio collection, so a record can carry an essential-gene knockdown and a cataloged
  deletion at once.
- ``price2018`` -- ``RbTnseqPrice2018EcoliDataset``: the RB-TnSeq fitness compendium,
  the loader that serves the experiments Wetmore 2015 first reported; and
  ``GeneEssentialityPrice2018EcoliDataset``, its Supplementary Table 1 likely-essential
  gene list, a TnSeq no-insertion call over the genes the fitness analysis could not
  value, so the two gene sets are disjoint.
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
- ``schmidt2016_srm`` -- ``ProteomeSrmSet1Schmidt2016Dataset`` and
  ``ProteomeSrmSet2Schmidt2016Dataset``: the same paper's Tables S2 and S3, the 41-protein
  SRM + stable-isotope-dilution panel the proteome-wide estimates were anchored on. A
  DIFFERENT assay of proteins the Table S6 block also covers, so each arm carries its own
  ``measurement_type`` and the build asserts that zero shared (accession, condition) cells
  agree with the stored block. The two arms are separate classes because their released
  dispersions are of different things (two SRM injections against three grown cultures)
  and one dataset carries one ``measurement_type``.
- ``schmidt2016_growth_rate`` -- ``GrowthRateSchmidt2016Dataset``: the same paper's Table
  S24, the only gene-perturbation phenotype it releases. Six records: the KEIO ``rimI``,
  ``rimJ`` and ``rimL`` deletions in glucose and in acetate, each a ``FitnessPhenotype``
  ratio to the wild-type row of its own medium.
- ``schmidt2016_s23_growth_rate`` -- ``GrowthRateS23Schmidt2016Dataset``: the same
  paper's Table S23, the per-condition growth rates. 15 records of the 26 released rows,
  one ``EnvironmentResponsePhenotype`` per kept (condition, BW25113) row carrying the
  ABSOLUTE rate in h^-1 against the released Glucose row. The 11 dropped rows are the 4
  rows of the two strains L1 cannot tell apart from BW25113 on a wild-type genotype, plus
  the 7 rows the proteome loader's own medium, growth-phase and culture-mode rules cover.
- ``mohiuddin2022`` -- ``PromoterReporterMohiuddin2022Dataset``: the promoter-GFP
  reporter library, one ``PromoterActivityPhenotype`` record per (arm, plate, well, read
  hour) over 1,930 wells, four arms and nine hourly reads. The only expression-side
  antibiotic-response set for this host: the three drugs are read as induced
  transcription rather than as a fitness cost, and the untreated arm is stored as data
  because it is the divisor the released fold changes use. Its environment carries the
  drug only from hour five, when the dose went in.
- ``wang2024`` -- ``EnvChemgenWang2024Dataset``: the rifampicin dose-by-time Tn-seq
  screen, one ``EnvironmentResponsePhenotype`` log2 fold change per (MG1655 gene,
  condition) over three doses (2, 32 and 160 mg/L, 0.25x, 4x and 20x MIC) at 1 and 3 h.
  24,763 of the 26,514 released cells; the drops are genes with no reads in either pool
  and five b-numbers the pinned annotation does not carry.
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
- ``teteneva2024`` -- the lake-water RB-TnSeq record: the host gate is closed and the
  loader is blocked. W3110 now has its own assembly set in the genomes tier
  (``ecoli_K12_W3110_ASM1024v1``, ten members deposited 2026-10-09), but its GenBank
  deposit carries no ``locus_tag`` at all, so the GenBank-first ingest does not reach
  it and no genome class or schema vocabulary names it yet. The paper is in neither
  mirror and filing it is a curation decision, so every value a loader needs is a typed
  gap rather than a guess.
- ``cai2023`` -- the MCF2Chem record: an aggregation NOT admitted. Its 8,888 production
  records are transcriptions of review tables, not measurements, and they are released
  only through a web server whose API answered 502 the day the row was settled, so there
  is no artifact to hash. The module holds the inventory, the quotes and the five
  required titer fields the release cannot fill.
- ``li2024`` -- the D2Cell 2026 record: a secondary source NOT admitted. Its database
  workbook is a Qwen1.5-110B transcription of abstracts and full texts (10,525 E. coli
  rows over 2,030 source DOIs, no source sentence per value, 383 distinct titer units),
  and its training split is a binary label construction with rule-designated
  negatives. The per-row DOIs are kept as a lead list of primary papers.
- ``typas2008`` -- the eSGA companion record: NOT loaded, because the release prints no
  interaction score. Its 42 released pairs are the TERMS neg (sick), neg (lethal) and
  pos, its one numeric table is a marker co-transduction check, and its 12 by 12 cross is
  four heat-map panels. The deposit IS mirrored in full, and Babu 2014 holds only 2 of
  the 42 pairs, so the row is blocked rather than subsumed.
"""

from .babu2014 import GeneInteractionBabu2014Dataset as GeneInteractionBabu2014Dataset
from .brunk2016 import BiofuelTiterBrunk2016Dataset as BiofuelTiterBrunk2016Dataset
from .brunk2016 import ExometaboliteBrunk2016Dataset as ExometaboliteBrunk2016Dataset
from .brunk2016 import MetabolomeBrunk2016Dataset as MetabolomeBrunk2016Dataset
from .brunk2016 import ProteomeBrunk2016Dataset as ProteomeBrunk2016Dataset
from .butland2008 import (
    GeneInteractionButland2008Dataset as GeneInteractionButland2008Dataset,
)
from .caglar2017 import (
    ProteinFoldChangeCaglar2017Dataset as ProteinFoldChangeCaglar2017Dataset,
)
from .caglar2017 import ProteomeCaglar2017Dataset as ProteomeCaglar2017Dataset
from .caglar2017 import RnaseqCaglar2017Dataset as RnaseqCaglar2017Dataset
from .caglar2017_doubling_time import (
    DoublingTimeCaglar2017Dataset as DoublingTimeCaglar2017Dataset,
)
from .campos2018 import GrowthRateCampos2018Dataset as GrowthRateCampos2018Dataset
from .campos2018 import MorphologyCampos2018Dataset as MorphologyCampos2018Dataset
from .choe2019_growth_rate import GrowthRateChoe2019Dataset as GrowthRateChoe2019Dataset
from .choe2019_growth_rate import (
    TranscriptionFactorKnockoutChoe2019Dataset as TranscriptionFactorKnockoutChoe2019Dataset,
)
from .choe2025 import CrispriChemgenChoe2025Dataset as CrispriChemgenChoe2025Dataset
from .cui2018 import CrispriKnockdownCui2018Dataset as CrispriKnockdownCui2018Dataset
from .fang2025 import (
    CrispriGuideFfaEnrichmentFang2025Dataset as CrispriGuideFfaEnrichmentFang2025Dataset,
)
from .foo2014 import IsopentenolTiterFoo2014Dataset as IsopentenolTiterFoo2014Dataset
from .fuhrer2017 import MetabolomeFuhrer2017Dataset as MetabolomeFuhrer2017Dataset
from .girgis2009 import EnvChemgenGirgis2009Dataset as EnvChemgenGirgis2009Dataset
from .goodall2018 import (
    GeneEssentialityGoodall2018Dataset as GeneEssentialityGoodall2018Dataset,
)
from .gupta2024 import (
    ProteinTurnoverGupta2024Dataset as ProteinTurnoverGupta2024Dataset,
)
from .hawkins2020 import (
    MismatchCrispriFitnessHawkins2020Dataset as MismatchCrispriFitnessHawkins2020Dataset,
)
from .ishii2007 import FluxIshii2007Dataset as FluxIshii2007Dataset
from .ishii2007 import MetabolomeIshii2007Dataset as MetabolomeIshii2007Dataset
from .ishii2007 import ProteomeIshii2007Dataset as ProteomeIshii2007Dataset
from .lamoureux2023 import RnaseqLamoureux2023Dataset as RnaseqLamoureux2023Dataset
from .lamoureux2023_growth import (
    GrowthRateLamoureux2023Dataset as GrowthRateLamoureux2023Dataset,
)
from .lamoureux2023_public_k12 import (
    RnaseqPublicK12Lamoureux2023Dataset as RnaseqPublicK12Lamoureux2023Dataset,
)
from .mohiuddin2022 import (
    PromoterReporterMohiuddin2022Dataset as PromoterReporterMohiuddin2022Dataset,
)
from .mori2021 import ProteomeMori2021Dataset as ProteomeMori2021Dataset
from .mutalik2020 import (
    PhageRbTnseqMutalik2020Dataset as PhageRbTnseqMutalik2020Dataset,
)
from .niu2019 import (
    CrisprPineneToleranceNiu2019Dataset as CrisprPineneToleranceNiu2019Dataset,
)
from .price2018 import (
    GeneEssentialityPrice2018EcoliDataset as GeneEssentialityPrice2018EcoliDataset,
)
from .price2018 import RbTnseqPrice2018EcoliDataset as RbTnseqPrice2018EcoliDataset
from .rachwalski2024 import (
    CrispriCrossRachwalski2024Dataset as CrispriCrossRachwalski2024Dataset,
)
from .rapp2026 import MetabolomeRapp2026Dataset as MetabolomeRapp2026Dataset
from .rapp2026_platforms import GrowthAucRapp2026Dataset as GrowthAucRapp2026Dataset
from .rapp2026_platforms import (
    MetaboliteIntensityRapp2026Dataset as MetaboliteIntensityRapp2026Dataset,
)
from .rapp2026_platforms import (
    TargetedMetabolomeRapp2026Dataset as TargetedMetabolomeRapp2026Dataset,
)
from .rousset2018 import (
    CrispriScreenRousset2018Dataset as CrispriScreenRousset2018Dataset,
)
from .schastnaya2021 import (
    MetabolomeSchastnaya2021Dataset as MetabolomeSchastnaya2021Dataset,
)
from .schmidt2016 import ProteomeSchmidt2016Dataset as ProteomeSchmidt2016Dataset
from .schmidt2016_growth_rate import (
    GrowthRateSchmidt2016Dataset as GrowthRateSchmidt2016Dataset,
)
from .schmidt2016_s23_growth_rate import (
    GrowthRateS23Schmidt2016Dataset as GrowthRateS23Schmidt2016Dataset,
)
from .schmidt2016_srm import (
    ProteomeSrmSet1Schmidt2016Dataset as ProteomeSrmSet1Schmidt2016Dataset,
)
from .schmidt2016_srm import (
    ProteomeSrmSet2Schmidt2016Dataset as ProteomeSrmSet2Schmidt2016Dataset,
)
from .shiver2016 import EnvChemgenShiver2016Dataset as EnvChemgenShiver2016Dataset
from .tong2020 import CarbonSourceTong2020Dataset as CarbonSourceTong2020Dataset
from .wang2015 import EnvChemgenWang2015Dataset as EnvChemgenWang2015Dataset
from .wang2015_growth import GrowthWang2015Dataset as GrowthWang2015Dataset
from .wang2018 import (
    CrispriGuideFitnessWang2018Dataset as CrispriGuideFitnessWang2018Dataset,
)
from .wang2024 import EnvChemgenWang2024Dataset as EnvChemgenWang2024Dataset
