# torchcell/datasets/pputida/yunus2026
# [[torchcell.datasets.pputida.yunus2026]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/pputida/yunus2026
# Test file: tests/torchcell/datasets/pputida/test_yunus2026.py
"""Yunus 2026 CRISPRi knockdown panel on an isoprenol-producing P. putida KT2440 chassis.

Yunus, Carruthers, Chen, Gin, Baidoo, Petzold, Garcia Martin, Adams, Mukhopadhyay and
Lee 2026 (Metab. Eng., doi:10.1016/j.ymben.2025.11.007) screened CRISPRi knockdowns
chosen by FluxRETAP against knockdowns chosen by intuition, in the engineered
isoprenol-producing strain ``IY1452``, and built the arrays with VAMMPIRE. Seven released
tables of the one supplementary file carry a per-strain number, and this module serves
all of them, as three dataset classes because ``ExperimentDataset.transform_item``
validates against ONE ``experiment_class``:

- :class:`CrispriKnockdownYunus2026Dataset` -- Supplementary Table S3, the 125-sample
  shotgun-proteomics screen: one record per CRISPRi strain, the relative expression of
  its OWN target protein against the control strain.
- :class:`CrispriArrayYunus2026Dataset` -- Supplementary Tables S8-S12, the sgRNA-array
  position study: one record per multiplexed construct, the relative expression of each
  of the five representative proteins the panel measured in it, over three biological
  replicates with the sample SD.
- :class:`CrispriDifferentialProteomeYunus2026Dataset` -- Supplementary Tables S4 and S5,
  the ``PP_4188`` strain's global fold change against the control: ONE record carrying
  the 305 of 338 released protein keys that resolve to a locus of the pinned assembly.
- :class:`IsoprenolTiterYunus2026Dataset` -- the HAND-DEPOSITED Supplementary Note 1
  input table, the only released per-strain isoprenol titer: one record per deposited
  strain row, referenced to the released control row's own titer.
- :class:`CrispriPanelProteomeYunus2026Dataset` -- the 253 UniProt accession columns of
  that same deposited table, as an ABSOLUTE per-strain Top3 signal, referenced to the
  control row's own profile.

THE THIRD FAMILY IS THE SAME QUANTITY TYPE ON A DIFFERENT SCALE, WHICH IS WHY IT IS A
THIRD DATASET. Tables S4 and S5 release a ``Fold Change`` against the control strain from
the same DIA-NN Top3 quantification, so it is the quantity ``ProteinFoldChangePhenotype``
already holds here. It is not a 103rd record of the Table S3 family, because the two
differ in exactly the way ``measurement_type`` exists to record: Table S3 is ONE
proteomics sample per strain with no uncertainty and no test released, while this is a
thresholded differential over three biological replicates with a t-test beside it. The
shared ``verify_protein_dataset`` asserts one ``measurement_type`` per dataset, so a
different scale is a different dataset.

TABLES S4 AND S5's P-VALUE IS STORED; WHAT IS READ AND NOT STORED IS ONE REVERSIBLE
TRANSFORM OF IT AND ONE PRESENTATION INDEX. The released ``P-Value (Equal Variance)``
column goes into ``protein_fold_change_p_value`` for all 305 stored keys, unadjusted:
the Fig. 5B caption states it is a "student's t-test $p$ -value" with an applied
threshold of 0.05, and no multiple-testing correction appears anywhere in the paper or
the SI, so ``p_value_adjustment_method`` is ``None`` and the adjusted map is a typed gap.
The one FDR in this paper is DIA-NN's IDENTIFICATION filter ("a global FDR = 0.01 at both
the precursor and protein group levels"), applied before any contrast was computed, and
it is recorded in :data:`SOURCED_VALUES` precisely so it is never mistaken for a
correction of these p-values. ``(-Log10(P-Value))`` is the same number by exact
arithmetic (``p = 10 ** -x``) and ``Rank`` is a presentation index of the released sort
order (:data:`DIFFERENTIAL_NOT_STORED`); both are read and used as build oracles instead
(:func:`parse_differential`), so no asserted column is parsed past: ``log2(Fold Change)``
reproduces the released log2 on 338 of 338 rows, the released ``-log10 p`` recovers the
printed p to its own precision, the ranks are ``1..n`` in row order, and every row clears
the thresholds its direction implies.

THE KNOCKED-DOWN GENE IS THE GENOTYPE, NOT A MEASURED KEY, AND IT IS IN THE TABLES UNDER
ANOTHER NAME. No row of either table carries ``PP_4188`` in its ``Protein`` column, which
the build asserts, so this record's profile and the Table S3 record for the same strain
share no protein and cannot disagree about one. But the protein IS there: Table S4 row 40
is ``Kgdb`` (``Q88FB0``, "Dihydrolipoyllysine-residue succinyltransferase component of
2-oxoglutarate dehydrogenase complex") at a fold change of 0.238537433, and that is the
enzyme Table S2 names for ``PP_4188`` ("2-oxoglutarate dehydrogenase
dihydrolipoyltranssuccinylase subunit"), which the pinned annotation resolves from the
symbol ``sucB``. ``Kgdb`` is a retired symbol this assembly carries no gene row for, so
it lands in the dropped set and nothing clashes today. A UniProt-to-locus-tag crosswalk
that recovered it would put 0.238537433 beside Table S3's 0.2213 for the same strain, on
a DIFFERENT ``measurement_type``, which is the two independent runs this module already
documents rather than a contradiction. ``Kgda`` (``Q88FA9``, the E1 component) is
``PP_4189`` by the same route.

THE ISOPRENOL TITERS ARE LOADED, FROM A MANUAL DEPOSIT. This is a production campaign
and its headline readout is isoprenol titer. The publisher releases the per-strain
titers as bar charts only (Fig. 4B-J, Supplementary Fig. S6 for the intuition group,
S7 for OD600, S9 for the multiplexed strains) and the prose states exactly two numbers,
``1469 mg/L`` for ``PP_4188`` and ``958 mg/L`` for ``PP_0168``, never the control's. But
Supplementary Note 1 links the per-strain INPUT table of its Pearson analysis
(``strain``, ``isoprenol_production``, then 253 UniProt protein columns) as a Benchling
share page, and that page carries a ``Control`` row. The share page is a JavaScript
single-page app whose data loads through an authenticated internal API (``/1/api/...``
answers HTTP 401 "permission denied - not logged in"; the public share page answers 200
with no data in the HTML, measured 2026-10-07), so the owner opened it in a browser and
PASTED the rendered table; both pages are deposited under ``data/benchling/`` with
``RetrievalMethod.manual_browser`` and their sha256, and :data:`BENCHLING_PASTE_CAVEAT`
is carried into every record's provenance because a DISPLAYED value is not an export.
That ``Control`` row is a real released control titer, which is what makes
``ProductTiterExperimentReference`` satisfiable without inventing a denominator.
``mmc1.docx`` is still the publisher's ONLY supplementary component: ``mmc1..mmc4`` x
``{docx,xlsx,pdf,zip,csv}`` were probed on ``ars.els-cdn.com`` on 2026-10-07 and only
``mmc1.docx`` exists (HTTP 206; every other combination 404).

WHAT THE DEPOSITED ROW LABELS ARE, AND THE SIX ROWS THAT ARE DROPPED. The 132 row labels
are CRISPRi TARGET labels, not strain names, and they are reconciled against what this
loader already knows: Table S3's ``CRISPRi target gene`` labels and tags plus Tables S1
and S2's locus tags. Measured 2026-10-09: 123 match a released target exactly, 6 more
match after stripping a trailing ``" (S)"``, 2 more after also stripping a trailing
``_NT<digit>``, and exactly one label matches nothing -- ``Control``, which is the
reference. The ``_NT<digit>`` token is the guide VARIANT number Table S3 already uses
(Table S7 gives ``PP_0339_NT1`` and ``PP_0339_NT2`` distinct spacers; Table S3 screens
``PP_1607_NT2`` and ``PP_1607_NT4`` as two strains), so it is matched as part of the
guide key rather than read as a non-targeting filler; either reading names the same
single perturbed locus. The ``" (S)"`` marker is DIFFERENT: no mirrored byte defines it
(:data:`BENCHLING_MARKER_SEARCH` records the search), and each of those six labels also
appears WITHOUT the marker as its own row, so the six marked rows are DROPPED by a named
rule and no target is lost. Both families therefore store 125 records.

ONE CROSS-SOURCE AGREEMENT AND ONE MEASURED DISAGREEMENT, BOTH ASSERTED AT BUILD TIME.
The deposit is an independent retrieval of the same campaign, so the two titers the
Results print are the available join (:func:`assert_benchling_titers_match_the_results_text`,
written into ``preprocess/benchling_proofs.json``). ``PP_0168`` is 957.246595 in the
deposit against the printed 958 mg/L, a difference of 0.7534 mg/L, inside the 1 mg/L the
paper prints to, and that bound is asserted. ``PP_4188`` is 1494.98874 against the
printed 1469 mg/L, a difference of 25.98874 mg/L (1.77 %), which is NOT inside it. No
mirrored byte says which number Fig. 4E was drawn from, so the difference is recorded
exactly and the deposited per-strain column is what every record stores.

THE DEPOSITED PANEL IS KEYED BY UNIPROT ACCESSION, SO IT GOES THROUGH THE GOA CROSSWALK.
``ProteinAbundancePhenotype`` keys by locus tag and the deposited columns are UniProt
accessions, so each is resolved through the assembly set's own GOA proteome file, the
same sha256-pinned member the genome reads its GO from. Measured 2026-10-09: 207 of 253
accessions (0.8182) reach exactly one KT2440 locus, 2 reach several (``Q877U6``,
``Q877V8``) and 44 reach none; the floor sits just below the measured fraction and the 46
are dropped from every profile by two named rules, listed in
``preprocess/dropped_accessions.csv``. 5,978 of 33,396 released abundance cells are
exactly 0 and none is blank: a released 0 is a present measurement and is kept verbatim.

THE PEARSON OUTPUT PAGE IS RECORDED AND NEVER STORED. The other deposited file is the
analysis page's result frame: ``Protein``, ``Correlation_with_isoprenol_production``,
``p_value``, one row per protein. That is a derived statistic over a strain panel rather
than a measurement of any strain, and no phenotype class models a per-protein
correlation with its own test, so it is in the mirror manifest and in :data:`NOT_LOADED`
and no record stores it. Measured on the pinned bytes: 2,659 rows with p up to 0.9989,
so it is the UNFILTERED frame of Supplementary Note 1's script rather than its p < 0.05
output, and it covers a WIDER protein set than the panel (252 of the panel's 253
accessions appear in it, one does not), so neither deposited file completes the other.

WHAT ONE STORED NUMBER IS, AND WHY IT IS A ``ProteinFoldChangePhenotype``. Every
released table here reports a RATIO: the target protein's abundance in the CRISPRi strain
divided by its abundance in the control strain, from the same DIA-NN Top3 quantification.
The record carries the strain's own number and the reference carries ``1.0``, which is
the ratio's denominator by definition rather than a measured quantity, so experiment /
reference reproduces the released value exactly and nothing is imputed.
``ProteinAbundancePhenotype`` is the WRONG class for that and says so in its own
docstring ("absolute per-strain quantity on a log signal scale, NOT a ratio"), which is
the mislabeling issue #770 records; these three datasets are on the relative sibling
instead. ``fold_change_scale`` is :data:`FOLD_CHANGE_SCALE` (linear, MEASURED two ways,
see that constant), ``reference_basis`` carries the source's own clause for the
denominator (:data:`REFERENCE_BASIS`, :data:`DIFFERENTIAL_REFERENCE_BASIS`), and
``measurement_type`` names the quantification (:data:`MEASUREMENT_TYPE`), which is the
axis that keeps these numbers from ever being compared with an absolute Top3 signal such
as the Carruthers 2025 proteome's.

A RELEASED ``0`` IS A MEASUREMENT, NOT A MISSING VALUE, AND IT IS NOT AN ``n.d.``. Table
S3 writes the verbatim cell ``0`` on 37 of its 102 numeric rows and Tables S9-S12 write
it on 49 of their 93 replicate cells, which makes 13 of the 51 (construct, protein) means
exactly ``0.0``. Those are complete knockdowns: the protein WAS detected in the control
strain (so the ratio has a denominator) and was not detected in the CRISPRi strain. The
paper counts them, and the arithmetic proves it: its census reproduces exactly on linear
thresholds over the 102 numeric rows (68 at ``<= 0.5``, 51 at ``< 0.05``, 16 in
``(0.5, 0.9]``, 13 in ``(0.9, 1.0)``, 5 above ``1.0``, summing to 102), and the 37 zeros
sit inside the 51 "downregulated by more than 95 %" (the ``linear_scale_census`` entry
of :data:`SOURCED_VALUES` carries the quote). The 23 ``n.d.`` rows are the
different, DROPPED case: the paper states their expression "was not detected in the
control strain", so the ratio has no denominator at all.

THE CHASSIS IS A DEFERRAL THIS PAPER DOES NOT CLOSE. The background is the strain the
paper names, ``IY1452``, described only as "a highly genetically engineered
isoprenol-producing strain (IY1452) (Banerjee et al., 2024)". Banerjee 2024 (Metab. Eng.
82:157-170, doi:10.1016/j.ymben.2024.02.004) is NOT in the literature mirror, so the
allele-level genotype and the construction are typed ``ProvenanceGap``s with
``deferred_pending_source_review`` and ``resolve_with`` naming that paper. Carruthers
2025 IS mirrored and states a genotype for ``IY1449b`` / ``IY1452b``, and that genotype
is deliberately NOT borrowed: ``IY1452`` and ``IY1452b`` are different designations and
no mirrored byte says they are the same strain. ``parents`` is ``["KT2440"]``, which the
paper does state ("a highly engineered Pseudomonas putida KT2440 strain").

EVERY GUIDE IS MAPPED TO ITS SPACER, AND THE PAIRED OLIGOS CHECK EACH OTHER.
Supplementary Table S7 releases 204 forward/reverse oligo pairs. Each forward oligo is
``TCTGGGTCTCTTAGC`` + spacer + ``GTTTGGAGACCATCG`` and each reverse oligo is
``CGATGGTCTCCAAAC`` + spacer + ``GCTAAGAGACCCAGA``; the build asserts both flank pairs
and that the reverse spacer is the reverse complement of the forward one, which holds
for 204 of 204. 203 spacers are 22 nt as the Methods state ("single guide RNAs (sgRNAs)
with 22 nucleotides"); ``PP4650_sgRNA_NT1`` is 21 nt and is stored verbatim rather than
corrected. An oligo label is either a ``PP_`` tag or a gene symbol the pinned annotation
resolves (``accA`` -> ``PP_1607``, ``gltA`` -> ``PP_4194``, ``bioB`` -> ``PP_0362``,
``birA`` -> ``PP_0437``, ``pta`` -> ``PP_0774``, ``edd`` -> ``PP_1010``, ``hsdR`` ->
``PP_4740``). The ``NT<n>`` token is a guide VARIANT number, not a non-targeting filler:
``PP_0339_NT1`` and ``PP_0339_NT2`` carry DISTINCT spacers, and Table S3 screens
``PP_1607_NT2`` and ``PP_1607_NT4`` as two separate strains, so the variant is matched
as part of the key.

FOUR KEPT RECORDS CARRY NO SPACER, each for a measured reason (all four are in
``preprocess/guide_assignment.csv``): ``PP_5064`` and ``PP_4678`` have no oligo in Table
S7 at all; ``PP_1444`` has two variant oligos (``NT1``, ``NT2``) with distinct spacers
while its Table S3 strain carries no variant label, so which guide it holds is not
stated; and ``PP_1319`` has two oligos BOTH labeled ``NT1`` (``PP1319_NT1_sgRNA`` and
``PP1319_sgRNA_NT1``) with distinct spacers, which the variant key cannot separate.

ONE CROSS-SOURCE DISAGREEMENT, KEPT RATHER THAN SILENTLY REPAIRED. Table S7's only
oligo for the Table S1 target ``PP_3744`` is labeled ``glgC``, while Table S1 names that
gene's enzyme ``GlcC`` ("transcriptional dual regulator GlcC-Glycolate"). Measured on
the pinned annotation, ``glcC`` resolves to ``PP_3744`` and ``glgC`` resolves to no
locus of this assembly, so the label is one letter away from the gene Table S1 names.
The loader does NOT remap it: ``PP_3744``'s strain is a Table S3 ``n.d.`` row and is
dropped anyway, and the finding is recorded instead of a correction.

A SECOND SOURCE INCONSISTENCY, FOR THE RECORD. The Abstract names the best knockdown
``PP_4118`` and the Results, the Discussion and Supplementary Fig. S8 all name
``PP_4188``. ``PP_4188`` is the FluxRETAP target Table S2 lists as ``SucB``,
"2-oxoglutarate dehydrogenase dihydrolipoyltranssuccinylase subunit", which agrees with
the Abstract's own gloss "a gene encoding alpha-ketoglutarate dehydrogenase";
``PP_4118`` appears nowhere else in the paper and in no SI table. Both spellings are
recorded in :data:`SOURCED_VALUES` and no record is written from the Abstract.

REPLICATES AND UNCERTAINTY DIFFER BETWEEN THE TWO FAMILIES, WHICH IS WHY THEY ARE TWO
DATASETS. Table S3 is "shotgun proteomics on 125 samples carrying different sgRNAs", and
the table holds exactly 125 rows, one per strain, so one sample per strain; the build
asserts that row count, which is what makes ``n_replicates = 1`` an arithmetic reading
of the source rather than an assumption. Table S3 releases no uncertainty and the
replicate DESIGN behind one sample is not stated, so ``protein_fold_change_se`` is a typed
gap. Tables S8-S12 release three per-replicate values per construct (``R1``, ``R2``,
``R3``) and the Fig. 3J-N caption states "Error bars represent standard deviation from
three biological replicates", so those records carry ``n_replicates = 3`` and an SE
derived as the sample SD over sqrt(3). The two families also disagree numerically for a
genotype they share -- Table S3 gives ``PP_4188`` 0.2213 while Table S8's three
replicates mean 0.2509 -- which is the evidence that they are separate runs and must not
be pooled into one record.

NOT LOADED, with the reason: Supplementary Tables S4 and S5's ``(-Log10(P-Value))`` and
``Rank`` columns, the first because it is the stored p-value's own reversible transform
and the second because it is a presentation index of the released sort order (both are
read and asserted as build oracles instead); Supplementary Table S6 (plasmids) and the
non-sgRNA rows of Table S7 (the three sequencing primers) are genotype metadata rather
than measurements; Tables S1 and S2 are the target lists, carried in
``preprocess/target_lists.csv``; Fig. 5A's TCA metabolite concentrations, Fig. 3C/D's
RFP and OD600, and Fig. 3F's growth curve are released as figures only; PRIDE
``PXD062697`` holds the raw DIA spectra, which no loader here consumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import os.path as osp
import re
import shutil
import statistics
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal
from xml.etree import ElementTree

import pandas as pd
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import M9
from torchcell.datamodels.schema import (
    BACTERIAL_ASSEMBLY_SETS,
    AssemblyReferenceGenome,
    BacterialAssemblySet,
    BacterialCrisprInterferencePerturbation,
    BacterialGeneNamespace,
    BacterialProteinAbundanceExperiment,
    BacterialProteinAbundanceExperimentReference,
    BacterialProteinFoldChangeExperiment,
    BacterialProteinFoldChangeExperimentReference,
    BacterialReferenceStrain,
    BacterialStrainBackground,
    Concentration,
    ConcentrationUnit,
    CrisprConstruct,
    CultureEnvironment,
    CultureFormat,
    EndpointRule,
    Environment,
    EnvironmentPhysicalPerturbation,
    Experiment,
    ExperimentReference,
    FoldChangeScale,
    Genotype,
    PhysicalFactor,
    ProductTiterExperiment,
    ProductTiterExperimentReference,
    ProductTiterPhenotype,
    ProteinAbundancePhenotype,
    ProteinFoldChangePhenotype,
    Publication,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.bacteria_common import (
    LocusTagReconciliation,
    UniProtResolution,
    assembly_reference,
    bacterial_genome,
    reconcile_locus_tags,
    resolve_uniprot_accessions,
    uniprot_locus_crosswalk,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ROLE_SI_DATA,
    ArtifactRecord,
    Manifest,
    ProcessingRecord,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.base import GeneNameStatus
from torchcell.sequence.genome.pputida.kt2440 import PPutidaKT2440Genome
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
    audit_sourced_value,
    library_available,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "yunusPredictiveCRISPRmediatedGene2026"
PAPER_DOI = "10.1016/j.ymben.2025.11.007"
PAPER_PII = "S1096717625001740"
PAPER_TITLE = (
    "Predictive CRISPR-mediated gene downregulation for enhanced production of "
    "sustainable aviation fuel precursor in Pseudomonas putida"
)
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
LIBRARY_DIR_REL = f"torchcell-library/{CITATION_KEY}"

KNOCKDOWN_ROOT_REL = "data/torchcell/crispri_knockdown_yunus2026"
ARRAY_ROOT_REL = "data/torchcell/crispri_array_yunus2026"
DIFFERENTIAL_ROOT_REL = "data/torchcell/crispri_differential_proteome_yunus2026"
TITER_ROOT_REL = "data/torchcell/isoprenol_titer_yunus2026"
PANEL_PROTEOME_ROOT_REL = "data/torchcell/crispri_panel_proteome_yunus2026"

#: The MinerU OCR of the publisher PDF; every ``SourcedValue`` quotes these bytes.
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "32ab4cd3753a930c6ad983809e7083a06159b7b0feb6b5252ace180273bbe563"

#: Supplementary file 1, the ONLY supplementary component the publisher serves.
SI_DOCX = "si1.docx"
#: The raw mirror's data directory, under which every consumed file sits.
DATA_DIR_REL = "data"
#: The raw-mirror path of the publisher's supplementary component.
SI_MIRROR_RELPATH = f"{DATA_DIR_REL}/{SI_DOCX}"
SI_DOCX_SHA256 = "daa2c91d0ec7b4560e086517bbdbcbf845c060f0f201c294c5ddef2399b963a9"
SI_DOCX_BYTES = 3517548
SI_SOURCE_URL = f"https://ars.els-cdn.com/content/image/1-s2.0-{PAPER_PII}-mmc1.docx"
_SI_RETRIEVED_AT = "2026-10-07T11:44:21.268547+00:00"

#: Banenjee 2024, the paper that built ``IY1452`` and that this paper defers its
#: genotype to. NOT in the literature mirror, which is why the chassis alleles are gaps.
CHASSIS_SOURCE_DOI = "10.1016/j.ymben.2024.02.004"
CHASSIS_SOURCE_CITATION = (
    "Banerjee, D., Yunus, I.S., Wang, X., Kim, Jinho, Srinivasan, A., Menchavez, R., "
    "Chen, Y., Gin, J.W., Petzold, C.J., Martin, H.G., Magnuson, J.K., Adams, P.D., "
    "Simmons, B.A., Mukhopadhyay, A., Kim, Joonhoon, Lee, T.S., 2024. Genome-scale and "
    "pathway engineering for the sustainable aviation fuel precursor isoprenol "
    "production in Pseudomonas putida. Metab. Eng. 82, 157-170."
)

#: The ProteomeXchange / PRIDE deposit of the raw DIA spectra (not consumed).
PRIDE_ACCESSION = "PXD062697"
#: Supplementary Note 1's Benchling share pages: the Pearson analysis and its input
#: table, the only released per-strain isoprenol numbers. Not scriptable (see docstring).
BENCHLING_ANALYSIS_URL = "https://benchling.com/s/etr-mta4DgBAatF0hFYzy275"
BENCHLING_INPUT_URL = "https://benchling.com/s/etr-l20YX8nWcCM66vIvFUZf"

# --------------------------------------------------------------------------- #
# The manual deposit of the two Benchling pages (issues #699 and #788 item 2)
# --------------------------------------------------------------------------- #
#: Where the owner deposited the two pasted tables inside the raw mirror.
BENCHLING_DIR_REL = f"{DATA_DIR_REL}/benchling"
#: The per-strain input table of Supplementary Note 1's Pearson analysis: the ONLY
#: released per-strain isoprenol titer of this paper, beside 253 protein columns.
BENCHLING_TITER_FILENAME = "strain_isoprenol_production_protein_abundance.tsv"
BENCHLING_TITER_REL = f"{BENCHLING_DIR_REL}/{BENCHLING_TITER_FILENAME}"
BENCHLING_TITER_SHA256 = (
    "4b5d70de2b858b93a5359a3d5f3c0c26db4c1b9387187cf39ec97456906c0838"
)
BENCHLING_TITER_BYTES = 269377
#: The Pearson OUTPUT page: one row per protein, a derived statistic. Recorded, never
#: stored as a phenotype (:data:`NOT_LOADED`).
BENCHLING_CORRELATION_FILENAME = "protein_correlation_with_isoprenol_production.tsv"
BENCHLING_CORRELATION_REL = f"{BENCHLING_DIR_REL}/{BENCHLING_CORRELATION_FILENAME}"
BENCHLING_CORRELATION_SHA256 = (
    "0e05f26b2686a2039dae70e2e87368f23e4c7a7f54fc9048c67adc5b4c986212"
)
BENCHLING_CORRELATION_BYTES = 80862
BENCHLING_RETRIEVED_AT = "2026-10-09"
#: Verbatim from ``data/benchling/DEPOSIT.md``: who produced the bytes and when.
BENCHLING_RETRIEVED_BY = (
    "the owner (mjvolk3); retrieved_at: 2026-10-09 (scratch notes created 02:54 and "
    "02:55 local)"
)
#: Verbatim from ``DEPOSIT.md``: the manual recipe a rebuild re-runs by hand before
#: verifying the two sha256 digests above. It is a COPY-PASTE of a rendered page, not an
#: export, which is why it is the whole recipe there is.
BENCHLING_MANUAL_RECIPE = (
    "HOW THE BYTES WERE OBTAINED, exactly: the owner opened the pages in a browser on "
    '2026-10-09 and could NOT use a CSV export ("couldn\'t download"), so each '
    "notebook table was COPIED from the rendered page and PASTED as tab-separated text "
    "into a scratch note; the driver session stripped the note frontmatter and wrote "
    "the body unchanged as the .tsv files here."
)
#: Verbatim from ``DEPOSIT.md``: what the numbers in these files ARE. Carried into the
#: ``Provenance.method`` of both families, because a displayed value is not an export
#: and nothing downstream may treat its precision as the instrument's.
BENCHLING_PASTE_CAVEAT = (
    "These are therefore the page's DISPLAYED values, not an export: numeric precision "
    "is whatever the page rendered (e.g. p-values appear as `1.30e-18`, correlations "
    "to 9 decimals, abundances as integers or with 2 decimals)."
)
#: Verbatim from ``DEPOSIT.md``: which page became which file, and the owner's own
#: caveat that the mapping is not independently verified.
BENCHLING_PAGE_MAPPING = (
    'page-to-file mapping, AS STATED BY THE OWNER ("first notebook" = the first link '
    'above, "second" = the second): first page -> '
    "`protein_correlation_with_isoprenol_production.tsv` (columns Protein, "
    "Correlation_with_isoprenol_production, p_value; 2,659 data rows); second page -> "
    "`strain_isoprenol_production_protein_abundance.tsv` (columns strain, "
    "isoprenol_production, then 253 UniProt protein columns; 132 strain rows). The "
    "mapping is not independently verified from the page titles."
)
#: Why the retrieval method is ``manual_browser`` and not a script, verbatim.
BENCHLING_DEPOSIT_NOTE = (
    "retrieval_method: manual_browser. The share pages are a JavaScript single-page "
    "app whose data loads through an authenticated internal API (HTTP 401 to scripts, "
    "measured 2026-10-07, issues #699 and #788 item 2)."
)
#: The deposit's own record and checksum files, named in every retrieval record.
BENCHLING_DEPOSIT_RECORD = f"{BENCHLING_DIR_REL}/DEPOSIT.md"
BENCHLING_CHECKSUMS = f"{BENCHLING_DIR_REL}/SHA256SUMS.txt"

#: The deposited table's shape, measured on the pinned bytes 2026-10-09
#: (``scripts`` of this module's note; re-measured by the data-gated tests).
BENCHLING_TITER_ROWS = 132
BENCHLING_TITER_COLUMNS = 255
BENCHLING_ACCESSIONS = 253
BENCHLING_CORRELATION_ROWS = 2659
#: Reconciliation of the 132 deposited row labels against what this loader already knows
#: (Table S3's ``CRISPRi target gene`` labels and tags plus Tables S1 and S2's locus
#: tags), measured 2026-10-09: 123 match a known label exactly, 6 more match after
#: stripping the trailing ``" (S)"``, 2 more after also stripping a trailing
#: ``_NT<digit>``, and exactly one label never matches -- ``Control``, which is the
#: reference row rather than a strain.
BENCHLING_EXACT_LABELS = 123
BENCHLING_MARKER_LABELS = 6
BENCHLING_VARIANT_LABELS = 2
#: The 6 marked labels, verbatim. Their unmarked twins are ALL in the table as well
#: (measured: every one of these six base tags also appears as a bare row), so the marker
#: separates two rows for one target and dropping the marked one loses no gene.
BENCHLING_MARKED_LABELS: tuple[str, ...] = (
    "PP_0751 (S)",
    "PP_0815 (S)",
    "PP_1240 (S)",
    "PP_1769 (S)",
    "PP_4191 (S)",
    "PP_4635 (S)",
)
#: What was searched for a definition of that marker, and what was found. Recorded as
#: the drop rule's own evidence: a marker nothing defines cannot be read, and guessing
#: at it would invent a second strain identity.
BENCHLING_MARKER_SEARCH = (
    "searched for a definition of the ' (S)' marker on 2026-10-09 and found none: "
    "paper.md (sha256 32ab4cd3753a930c6ad983809e7083a06159b7b0feb6b5252ace180273bbe563) "
    "contains the string '(S)' zero times and names none of the six tags; the pinned "
    "mmc1.docx contains it exactly once, inside the chemical name "
    "'ADP-dependent (S)-NAD(P)H-hydrate dehydratase' in Supplementary Table S5, which "
    "defines nothing about a strain; Supplementary Note 1 states only the two Benchling "
    "links and the Pearson script; and Supplementary Tables S6 and S7 carry each of the "
    "six tags as a plasmid or oligo name with no marker beside it"
)
#: Records written by each deposited family: the 132 rows less the control row and less
#: the 6 marked rows. Both families key on the same strain rows, so the two counts agree
#: by construction and the verifier asserts each.
BENCHLING_RECORDS = 125
#: UniProt accessions of the deposited panel reaching exactly one KT2440 locus through
#: the GOA proteome crosswalk: 207 of 253 (0.8182), measured 2026-10-09. 2 reach several
#: (``Q877U6``, ``Q877V8``) and 44 reach none. The floor sits just below the measured
#: fraction; every dropped accession is listed in ``preprocess/dropped_accessions.csv``.
PANEL_RESOLVED_ACCESSIONS = 207
PANEL_MIN_RESOLVED_FRACTION = 0.81
#: The two titers the Results state as a number, and what the deposited column gives for
#: the same two strains (measured 2026-10-09). ``PP_0168`` agrees to within the 1 mg/L
#: the paper prints; ``PP_4188`` does NOT, and the difference is recorded rather than
#: reconciled.
TITER_ORACLE_TOL_MG_PER_L = 1.0
TITER_ORACLE_AGREES = "PP_0168"
TITER_ORACLE_DISAGREES = "PP_4188"

KT2440_NAMESPACE: BacterialGeneNamespace = "pputida_kt2440_locus_tag"
KT2440_STRAIN: BacterialReferenceStrain = "KT2440"
KT2440_ASSEMBLY_SET: BacterialAssemblySet = "pputida_KT2440_ASM756v2"
if BACTERIAL_ASSEMBLY_SETS[KT2440_STRAIN] != KT2440_ASSEMBLY_SET:
    raise RuntimeError(
        f"{KT2440_STRAIN} is assembly set {BACTERIAL_ASSEMBLY_SETS[KT2440_STRAIN]!r}, "
        f"not the {KT2440_ASSEMBLY_SET!r} this loader pins"
    )
#: The strain label every record is written against, verbatim from the Results.
CHASSIS_STRAIN = "IY1452"

#: What one stored number IS. Named so a ratio to a control strain can never be
#: compared with an absolute Top3 signal (Carruthers 2025's
#: ``dia_nn_top3_peptide_signal_mean``).
MEASUREMENT_TYPE = "dia_nn_top3_relative_to_control_strain"
#: What one stored number of the Tables S4 + S5 differential IS. The released header's own
#: word is ``Fold Change``, the test beside it is an equal-variance Student's t-test, and
#: the replicate design behind it is three biological replicates, so it is a DIFFERENT
#: scale from Table S3's single-sample ``Relative expression level`` and gets its own
#: measurement_type and its own dataset class (the shared protein verifier asserts one
#: measurement_type per dataset).
DIFFERENTIAL_MEASUREMENT_TYPE = (
    "dia_nn_top3_fold_change_relative_to_control_strain_equal_variance_t_test"
)
#: What one stored number of the DEPOSITED Benchling panel IS: an ABSOLUTE per-strain
#: Top3 signal as the share page RENDERED it, not a ratio and not an export. A distinct
#: string from every other proteome family's (the two ratio scales above, Carruthers
#: 2025's ``dia_nn_top3_peptide_signal_mean`` and
#: ``dia_nn_top3_percent_of_proteome_mean``), so heterogeneous proteomics is never
#: pooled; ``benchling_displayed`` is the half of the name that records the caveat.
PANEL_PROTEOME_MEASUREMENT_TYPE = "dia_nn_top3_signal_benchling_displayed"
#: The scale EVERY released number of this paper is on: a plain, untransformed ratio of
#: the strain to the control strain. Measured, not assumed, two ways. (1) Tables S4 and
#: S5 release ``Fold Change`` AND ``Log2(Fold Change)`` side by side and
#: ``log2(Fold Change)`` reproduces the second column on 338 of 338 rows
#: (:func:`parse_differential`), so the first column is the linear ratio. (2) The Results
#: census of Table S3 reproduces exactly on LINEAR thresholds over the 102 numeric rows
#: (68 at <= 0.5, 51 at < 0.05, 16 in (0.5, 0.9], 13 in (0.9, 1.0), 5 above 1.0, summing
#: to 102); on a log2 scale 0.5 would be a 1.41-fold INCREASE, not "downregulated by
#: 50 %".
FOLD_CHANGE_SCALE = FoldChangeScale.linear
#: The reference strain's value on that scale: the ratio's denominator, by definition.
#: Derived from the scale rather than written down, so the two can never disagree.
REFERENCE_RELATIVE_EXPRESSION = FOLD_CHANGE_SCALE.neutral_value

#: The deposited table's own label for the control strain's row. It is a REAL released
#: row (isoprenol 845.73 and a full protein profile), so both deposited families
#: reference a measured control rather than an invented denominator.
BENCHLING_CONTROL_LABEL = "Control"
#: The marker six deposited row labels carry and that NO mirrored byte defines; the
#: search that established that is in :data:`BENCHLING_MARKER_SEARCH`.
BENCHLING_SOLID_MARKER = " (S)"

#: isoprenol (3-methyl-3-buten-1-ol), the campaign's product. Its row in
#: ``compound_identity_table.json`` landed with PR #729; the key is pinned here and checked
#: against that row so a later curation cannot silently disagree with this module. No titer
#: is released, so no record stores the compound.
ISOPRENOL_INCHIKEY = "CPJRRXSHAYUTGL-UHFFFAOYSA-N"

# --------------------------------------------------------------------------- #
# Supplementary-file structure (asserted, never assumed)
# --------------------------------------------------------------------------- #
#: Number of tables in the pinned ``mmc1.docx`` (S1-S12 plus two protocol tables).
SI_TABLE_COUNT = 14
#: 0-based table index of each Supplementary Table inside that file.
TABLE_INDEX: dict[str, int] = {
    "S1": 0,
    "S2": 1,
    "S3": 2,
    "S4": 3,
    "S5": 4,
    "S6": 5,
    "S7": 6,
    "S8": 7,
    "S9": 8,
    "S10": 9,
    "S11": 10,
    "S12": 11,
}
TABLE_S3_HEADER = ("Strain name", "CRISPRi target gene", "Relative expression level")
TABLE_S7_HEADER = ("Oligo name", "Sequence (5' to 3')")
#: Tables S4 and S5 share one header. ``P-Value (Equal Variance)``, ``(-Log10(P-Value))``
#: and ``Rank`` are read as build oracles and are NOT stored; see
#: :data:`DIFFERENTIAL_NOT_STORED`.
TABLE_S4_S5_HEADER = (
    "Protein.Group",
    "Protein.Names",
    "Protein",
    "Protein.Description",
    "Fold Change",
    "Log2(Fold Change)",
    "P-Value (Equal Variance)",
    "(-Log10(P-Value))",
    "Rank",
)
#: ``(table, direction)`` of the two halves of the PP_4188 differential, and the row
#: count each one holds in the pinned docx.
DIFFERENTIAL_PANELS: tuple[tuple[str, str, int], ...] = (
    ("S4", "downregulated", 145),
    ("S5", "upregulated", 193),
)
#: The strain both tables are measured on, verbatim from their captions ("from PP_4188
#: strain"); it is the Table S3 strain of the same name.
DIFFERENTIAL_STRAIN_TARGET = "PP_4188"
TABLE_S1_HEADER = (
    "No",
    "Target Gene",
    "Enzyme",
    "Enzyme description",
    "KEGG Orthology",
    "Metabolic Pathway",
    "iPath3",
)
TABLE_S2_HEADER = (
    "No.",
    "Target Gene",
    "Enzyme",
    "Enzyme description",
    "KEGG Orthology",
    "Metabolic pathways",
    "iPath3",
)
#: Table S3 holds one row per proteomics sample, and the Results state 125 samples.
TABLE_S3_ROWS = 125
#: The value Table S3 writes where the control strain had no detected expression.
NOT_DETECTED = "n.d."
#: ``(table, measured protein)`` of the five array-position panels (Fig. 3J-N).
ARRAY_PANELS: tuple[tuple[str, str], ...] = (
    ("S8", "PP_4188"),
    ("S9", "PP_0812"),
    ("S10", "PP_4160"),
    ("S11", "PP_0168"),
    ("S12", "PP_0528"),
)
#: Replicate labels every array panel releases, in order.
ARRAY_REPLICATES: tuple[str, ...] = ("R1", "R2", "R3")

#: sha256 of each parsed table (see :func:`table_digest`): the extraction check. A
#: changed digest means the docx or the parser moved, and the build stops.
TABLE_DIGESTS: dict[str, str] = {
    "S1": "ec9e0428ec8093940bc0b3b2b1210edc848c91cf715bef9994eccc194cc92800",
    "S2": "8648b678f58e330055121649788ac75b3d802608a2fba0a9f0f1abdd0c19d661",
    "S3": "8e46cb5226b69d367ce48a0793811851d8952da5b7d23f516d797696f7955881",
    "S4": "0a36b87dc4d8b19e9821867a926b9093a85bba8a5442e4ebc58a44e339533786",
    "S5": "82b1a035525763afcddbf52a300ef8371efa20604cd6a40301e0f9273925ad00",
    "S7": "54f8b3ad02b77b1ce8d004e95a3faa69e48fd77c254134a5b0e734770aae9143",
    "S8": "38d44066a69c78914e9e7459eeb30b69fce8327f22e57fa5f3f1b63dca80e35a",
    "S9": "637afca27730c99820ad962208503e88949960d570fe99120e5bb95a004d00ad",
    "S10": "b514b55a5d2c22c4fc855c68e6655758ed4389699272c77ba7904161b9d39d09",
    "S11": "b1074ad9b5df5b5e13debeaa373987e4234a39b649b67227ff8ac8668f13fb93",
    "S12": "99b1557883f1db8125f25d10e1d44a744e137b51fecd8187aa798b8cbe8a34f3",
}

#: The BASIC-method flanks every Table S7 sgRNA oligo carries around its spacer.
OLIGO_FORWARD_PREFIX = "TCTGGGTCTCTTAGC"
OLIGO_FORWARD_SUFFIX = "GTTTGGAGACCATCG"
OLIGO_REVERSE_PREFIX = "CGATGGTCTCCAAAC"
OLIGO_REVERSE_SUFFIX = "GCTAAGAGACCCAGA"
_DNA_COMPLEMENT = str.maketrans("ACGT", "TGCA")

OLIGO_NAME_RE = re.compile(r"^(?P<oligo>IY\d+)_(?P<label>.+)_(?P<side>[FR])$")
VARIANT_RE = re.compile(r"^NT\d$")
LOCUS_TAG_RE = re.compile(r"^PP_?(?P<number>\d{4})$")
S3_TARGET_RE = re.compile(r"^(?P<tag>PP_\d{4})(?:_(?P<variant>NT\d))?$")
#: The trailing guide-variant token of a released label, as a SUFFIX to strip when
#: reconciling a deposited row against the target lists.
VARIANT_SUFFIX_RE = re.compile(r"_NT\d$")
FOUR_DIGITS_RE = re.compile(r"\d{4}")

#: Oligo labels that name no gene of this assembly, each with the measured reason.
NON_GENE_OLIGO_LABELS: dict[str, str] = {
    "RFP": "the red fluorescent protein reporter of the Fig. 3B-D CRISPRi test, not a "
    "KT2440 gene",
    "BFP": "a fluorescent-protein reporter, not a KT2440 gene",
    "nontarget": "the non-targeting control guide, which perturbs no gene",
    "glgC": "one letter from the 'GlcC' Table S1 names for PP_3744: 'glcC' resolves to "
    "PP_3744 on the pinned annotation and 'glgC' resolves to no locus, so the label is "
    "kept unmapped rather than corrected",
}


# --------------------------------------------------------------------------- #
# Sourced values: every number below quotes sha256-pinned mirrored bytes
# --------------------------------------------------------------------------- #
def _paper(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote in the pinned ``paper.md`` OCR mirror."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
            page=page,
        ),
    )


_METHODS_STRAINS = "Methods 2.1, 'Strains, plasmids, media, and growth conditions'"
_METHODS_CRISPRI = "Methods 2.3, 'Construction of CRISPRi plasmids'"
_METHODS_PROTEOMICS = "Methods 2.6, 'Proteomics analysis'"
_METHODS_ISOPRENOL = "Methods 2.2, 'Routine isoprenol extraction and analysis'"
_RESULTS_VAMMPIRE = "Results 3.2, 'VAMMPIRE'"
_RESULTS_PREDICTIVE = "Results 3.3, 'Predictive CRISPRi downregulation'"


def _si_docx(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim paragraph or cell of the pinned ``mmc1.docx`` bytes."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SI_MIRROR_RELPATH,
            citation_key=CITATION_KEY,
            sha256=SI_DOCX_SHA256,
            method="stdlib WordprocessingML read of the deposited mmc1.docx (raw mirror)",
            page=page,
        ),
    )


SOURCED_VALUES: dict[str, SourcedValue] = {
    "chassis_strain": _paper(
        CHASSIS_STRAIN,
        "we transformed the CRISPRi plasmid into a highly genetically engineered "
        "isoprenol-producing strain (IY1452) (Banerjee et al., 2024) (Fig. 4A).",
        page=_RESULTS_PREDICTIVE,
        note="the only statement of the background; its allele-level genotype is "
        "deferred to Banerjee 2024, which is not in the literature mirror, so the "
        "alleles are typed gaps rather than borrowed from another paper's IY1452b",
    ),
    "host_strain": _paper(
        "KT2440",
        "The highest isoprenol titer to date in Pseudomonas was achieved by "
        "heterologous expression of the IPP-bypass pathway in a highly engineered "
        "Pseudomonas putida KT2440 strain developed by our group (Banerjee et al., "
        "2024).",
        page="Results 3.1, 'Identification of target genes'",
        note="the reference strain IY1452's alleles are edits against, and the only "
        "sourced statement of its parent",
    ),
    "effector": _paper(
        "dCas9",
        "A $\\mathrm { P _ { n a g A a } }$ promoter (Banerjee et al., 2024) was used "
        "to drive dCas9 expression as it is functional in P. putida KT2440 but inactive "
        "in E. coli, thereby minimizing the expression burden of dCas9 during cloning.",
        page=_RESULTS_VAMMPIRE,
        note="the CRISPRi effector; the vector it sits on is pIY989 "
        "(SOURCED_VALUES['crispri_plasmid'])",
    ),
    "crispri_plasmid": _paper(
        "pIY989",
        "The resulting plasmid was digested with BsaI and cloned into pIY989 plasmid "
        "(JBx_249567).",
        page=_METHODS_CRISPRI,
    ),
    "guide_length": _paper(
        22,
        "For CRISPRi-mediated gene downregulation, single guide RNAs (sgRNAs) with 22 "
        "nucleotides were designed using the web tool CRISPOR (Concordet and Haeussler, "
        "2018) to target the non-template strand with $3 ^ { \\prime } { \\cdot } "
        "\\mathrm { N G G } { \\cdot } 5 ^ { \\prime }$ protospacer adjacent motif "
        "(PAM) sequence.",
        page=_METHODS_CRISPRI,
        note="203 of the 204 released spacers are 22 nt; PP4650_sgRNA_NT1 is 21 nt and "
        "is stored verbatim",
    ),
    "oligo_table": _paper(
        "Supplementary Table S7",
        "Hybridized oligos used for CRISPRi-mediated gene downregulation are listed in "
        "Supplementary Table S7.",
        page=_METHODS_CRISPRI,
    ),
    "production_culture": _paper(
        {
            "medium": "M9",
            "glucose_percent": 2.0,
            "kanamycin_mg_per_l": 50.0,
            "gentamicin_mg_per_l": 10.0,
            "arabinose_percent": 0.2,
        },
        "For isoprenol production, cultures were inoculated at an $\\mathrm { O D } _ { "
        "6 0 0 }$ of 0.2 in 5 mL M9 medium with $2 \\%$ glucose and antibiotics "
        "(kanamycin $5 0 \\mathrm { m g / L }$ , gentamicin $1 0 \\mathrm { m g / L } )$ "
        ") and induced with $0 . 2 ~ \\%$ L-arabinose $^ { 4 \\mathrm { ~ h ~ } }$ "
        "after inoculation.",
        page=_METHODS_STRAINS,
        note="the only medium with a stated composition for a producing culture; the "
        "Methods write both percentages without a basis, and w/v is the convention for "
        "a solid solute, which is the inference recorded here and in the note. The "
        "proteomics Methods name no medium of their own and the samples were extracted "
        "at 48 h, which is the production culture's own endpoint",
    ),
    "temperature_c": _paper(
        30.0,
        "putida KT2440 seed cultures were grown from single colonies in 5 mL LB at $3 0 "
        "\\ { } ^ { \\circ } \\mathrm { C } ,$ $1 8 0 ~ \\mathrm { r p m }$ , overnight. "
        "Unless specified, $1 0 0 ~ \\mu \\mathrm { L }$ of the overnight culture was "
        "transferred to $5 ~ \\mathrm { m L }$ M9 minimal medium and incubated under "
        "the same conditions overnight. This step was repeated once to adapt the cells "
        "to M9 medium.",
        page=_METHODS_STRAINS,
        note="'the same conditions' is what carries 30 C and 180 rpm to the M9 "
        "cultures; 180 rpm is a CultureEnvironment field and Experiment.environment is "
        "annotated Environment, so it is recorded here and in the note, not typed",
    ),
    "duration_hours": _paper(
        48.0,
        "All samples were extracted at $^ { 4 8 \\mathrm { ~ h ~ } }$ . $\\mathrm { O D "
        "} _ { 6 0 0 }$ at $^ { 4 8 \\mathrm { ~ h ~ } }$ is shown in Supplementary "
        "Fig. S7.",
        page="Fig. 4 caption",
        note="the Fig. 4 strains are the Table S3 strains; the Supplementary Fig. S5 "
        "and S8 captions state the same 48 h for the array panels and for PP_4188",
    ),
    "aerobicity": _paper(
        "aerobic",
        "putida KT2440 seed cultures were grown from single colonies in 5 mL LB at $3 0 "
        "\\ { } ^ { \\circ } \\mathrm { C } ,$ $1 8 0 ~ \\mathrm { r p m }$ , overnight.",
        page=_METHODS_STRAINS,
        note="shaken tube cultures, the standard aerobic configuration; the source "
        "never uses the word",
    ),
    "screen_samples": _paper(
        TABLE_S3_ROWS,
        "To show the effectiveness of downregulation of different genes, we performed "
        "shotgun proteomics on 125 samples carrying different sgRNAs (Supplementary "
        "Table S3).",
        page=_RESULTS_VAMMPIRE,
        note="Table S3 holds exactly 125 rows and 125 distinct strain names, which the "
        "build asserts; 125 samples over 125 strains is one sample per strain, so "
        "n_replicates = 1 is arithmetic rather than an assumption",
    ),
    "not_detected_count": _paper(
        23,
        "The expression levels of twenty-three genes could not be determined as their "
        "gene expression was not detected in the control strain.",
        page=_RESULTS_VAMMPIRE,
        note="the sourced reason the 23 'n.d.' rows of Table S3 are dropped: with no "
        "control-strain expression the ratio has no denominator",
    ),
    "array_replicates": _paper(
        3,
        "Relative expression levels of selected target genes were measured when their "
        "corresponding sgRNAs were placed in different positions within multisgRNA "
        "arrays. Each panel shows one representative gene (PP_4188, PP_0812, PP_4160, "
        "PP_0168, PP_0528) with its repression level plotted across different array "
        "contexts. The complete list of target genes from each panel is shown in "
        "Supplementary Table S8–12. Error bars represent standard deviation from "
        "three biological replicates.",
        page="Fig. 3 caption (panels J-N)",
        note="the replicate count AND the uncertainty type of the Tables S8-S12 "
        "family: three biological replicates, sample standard deviation",
    ),
    "quantification": _paper(
        "DIA-NN Top 3",
        "Protein quantities were plotted using the Top 3 method, which averages the MS "
        "signal of the three most intense tryptic peptides.",
        page=_METHODS_PROTEOMICS,
        note="the quantification both released ratios are formed from",
    ),
    "relative_expression_basis": _paper(
        "the control strains",
        "(I) Summary of relative expression levels of target genes in comparison to "
        "the control strains.",
        page="Fig. 3 caption (panel I)",
        note="the DENOMINATOR of every Table S3 and Tables S8-S12 number, in the "
        "source's own words; it is what ProteinFoldChangePhenotype.reference_basis "
        "records, and the control strain is the nontarget-sgRNA strain of the same "
        "campaign (SOURCED_VALUES['control_strain_is_nontarget'])",
    ),
    "control_strain_is_nontarget": _paper(
        "nontarget",
        "(G) Protein counts of PP_1607 in both nontarget (control) and PP_1607 strains.",
        page="Fig. 3 caption (panel G)",
        note="the only statement of WHAT the control strain is: the strain carrying the "
        "non-targeting sgRNA, which is also the NON_GENE_OLIGO_LABELS 'nontarget' guide",
    ),
    "linear_scale_census": _paper(
        {"down_50": 68, "down_95": 51, "down_10_50": 16, "down_10": 13, "up": 5},
        "Proteomics analysis revealed that 68 genes were downregulated by at least $5 0 "
        "\\%$ , 51 of which were downregulated by more than $9 5 ~ \\%$ (Fig. 3). "
        "Sixteen genes were downregulated by $1 0 { - } 5 0 ~ \\%$ . Thirteen genes were "
        "downregulated only by $1 0 ~ \\%$ . Five genes were upregulated.",
        page=_RESULTS_VAMMPIRE,
        note="the evidence that FOLD_CHANGE_SCALE is linear and that a released '0' is a "
        "MEASUREMENT rather than a placeholder. The five buckets reproduce exactly on "
        "linear thresholds over Table S3's 102 numeric rows (68 at <= 0.5, 51 at < 0.05, "
        "16 in (0.5, 0.9], 13 in (0.9, 1.0), 5 above 1.0) and sum to 102; the 37 rows "
        "whose verbatim cell is '0' sit inside the 51 'more than 95 %' bucket, i.e. "
        "downregulated by 100 %. On a log2 scale 0.5 would be a 1.41-fold increase",
    ),
    "differential_basis_and_test": _paper(
        "the control strain",
        "(B) Volcano plot representing the results of shotgun proteomic analysis from "
        "strain PP_4188 strain in comparison to the control strain. Horizontal dashed "
        "line represents the applied significance threshold of a student’s t-test "
        "$p$ -value $= 0 . 0 5$ . Vertical dashed lines represent the applied thresholds "
        "of an absolute fold change ${ \\geq } 1$ .",
        page="Fig. 5 caption (panel B)",
        note="three things at once: the DENOMINATOR of Tables S4 and S5 ('in comparison "
        "to the control strain'), WHAT their 'P-Value (Equal Variance)' column is (a "
        "Student's t-test p-value), and that the 0.05 the parser asserts is the applied "
        "threshold. The t-test p-value is RAW: no multiple-testing correction is named "
        "anywhere in the paper or the SI, so p_value_adjustment_method stays None "
        "(SOURCED_VALUES['identification_fdr'] is the other FDR in this paper, and it is "
        "not one)",
    ),
    "identification_fdr": _paper(
        0.01,
        "The main DIA-NN reports were filtered with a global $\\mathrm { F D R } = 0 . 0 "
        "1$ at both the precursor and protein group levels.",
        page=_METHODS_PROTEOMICS,
        note="recorded so this FDR is never mistaken for a correction of the Tables S4 "
        "and S5 p-values: it is an IDENTIFICATION FDR on precursors and protein groups, "
        "applied before any contrast was computed. Nothing in the paper or the SI "
        "adjusts the released per-protein t-test p-values, which is why they are stored "
        "as protein_fold_change_p_value and the adjusted map is None",
    ),
    "search_database": _paper(
        "P. putida KT2440 UniProt proteome + heterologous proteins + contaminants",
        "The DIA-NN search used the latest P. putida KT2440 Uniprot proteome FASTA "
        "sequences, along with sequences for heterologous proteins and common "
        "contaminants.",
        page=_METHODS_PROTEOMICS,
    ),
    "raw_spectra_deposit": _paper(
        PRIDE_ACCESSION,
        "The generated mass spectrometry proteomics data have been deposited to the "
        "ProteomeXchange Consortium via the PRIDE partner repository with the dataset "
        "identifier PXD062697",
        page=_METHODS_PROTEOMICS,
        note="raw DIA files; no loader here consumes raw spectra",
    ),
    "titer_quantification": _paper(
        "GC-FID",
        "Isoprenol was sampled from the top ethyl acetate layer and measured using "
        "GC-FID, with concentration determined using serially diluted isoprenol "
        "standards.",
        page=_METHODS_ISOPRENOL,
        note="how the UNLOADED titer family was measured; recorded because the titers "
        "are the campaign's headline readout and are released only as bar charts",
    ),
    "best_titer_mg_per_l": _paper(
        1469.0,
        "The highest recorded isoprenol titer of $1 4 6 9 \\mathrm { m g / L }$ (Fig. "
        "4E) was achieved by downregulating PP_4188, a gene identified by FluxRETAP",
        page=_RESULTS_PREDICTIVE,
        note="one of only two titers the paper states as a number; no control titer is "
        "stated anywhere, which is why no ProductTiter family is built",
    ),
    "titer_unit": _paper(
        "mg/L",
        "The highest recorded isoprenol titer of $1 4 6 9 \\mathrm { m g / L }$ (Fig. "
        "4E) was achieved by downregulating PP_4188, a gene identified by FluxRETAP",
        page="Results section '# 3.3. Predictive CRISPRi downregulation for improved "
        "isoprenol production'",
        note="the unit of every isoprenol number this paper states, and therefore of "
        "the deposited 'isoprenol_production' column, which the Benchling page labels "
        "with no unit of its own. ConcentrationUnit carries no mg/L member and 1 mg/L "
        "is exactly 1 ug/mL, so the released number is stored verbatim under ug_per_ml "
        "and no arithmetic touches a source value",
    ),
    "titer_replicates": _paper(
        (3, 6),
        "All samples were extracted at $^ { 4 8 \\mathrm { ~ h ~ } }$ . $\\mathrm { O D "
        "} _ { 6 0 0 }$ at $^ { 4 8 \\mathrm { ~ h ~ } }$ is shown in Supplementary "
        "Fig. S7. Error bars represent standard deviation from 3 to 6 biological "
        "replicates.",
        page="Fig. 4 caption",
        note="the replicate DESIGN of the isoprenol titers, released only as a RANGE. "
        "The deposited column is one number per strain with no SD and no replicate "
        "count, and the SD exists only as the figure's error bars, so a back-solve is "
        "precluded; CLAUDE.md's range rule then takes the conservative lower end, 3, "
        "which never overstates the replicate support. Supplementary Figs. S6, S7 "
        "state the same 3 to 6 and Supplementary Fig. S9 states '3~6'",
    ),
    "best_intuition_titer_mg_per_l": _paper(
        958.0,
        "while in our intuition-based group, the highest titer was $9 5 8 \\mathrm { m "
        "g / L }$ obtained from the PP_0168 downregulated strain",
        page=_RESULTS_PREDICTIVE,
    ),
    "abstract_best_target": _paper(
        "PP_4118",
        "The highest isoprenol titer of nearly $1 . 5 ~ \\mathrm { g } / \\mathrm { L "
        "}$ was achieved by knocking down PP_4118 (a gene encoding $\\alpha$ "
        "-ketoglutarate dehydrogenase).",
        page="Abstract",
        note="the Abstract's spelling; the Results, the Discussion and Supplementary "
        "Fig. S8 all write PP_4188, which Table S2 lists as SucB, "
        "'2-oxoglutarate dehydrogenase dihydrolipoyltranssuccinylase subunit' -- the "
        "enzyme the Abstract itself names. PP_4118 appears nowhere else in the paper",
    ),
    "results_best_target": _paper(
        "PP_4188",
        "PP_4188 encodes $\\alpha$ -ketoglutarate dehydrogenase, a key enzyme in the "
        "TCA cycle.",
        page="Results 3.4, 'Metabolic and proteomic insights'",
    ),
    "array_sizes": _paper(
        {"one": 197, "two": 41, "three": 24, "four": 11, "five": 2},
        "Using this method, we constructed 197 CRISPRi plasmids harboring a sgRNA, 41 "
        "plasmids with two sgRNAs, 24 plasmids with three sgRNAs, 11 plasmids with four "
        "sgRNAs, and two plasmids with five sgRNAs.",
        page=_RESULTS_VAMMPIRE,
        note="the campaign's construct census; Tables S3 and S8-S12 together release a "
        "number for a subset of it",
    ),
}

#: Values quoted from the DEPOSITED ``mmc1.docx`` rather than from the OCR ``paper.md``.
#: They are a separate dict because ``audit_sourced_value`` reads its artifact as TEXT to
#: find the quote, and a ``.docx`` is a zip of deflated XML, so a quote inside one can
#: never be found that way. These three are audited instead against the paragraphs and
#: cells :func:`read_docx_tables` and the SI paragraph reader return, in
#: ``tests/torchcell/datasets/pputida/test_yunus2026.py``, which is the same check
#: against the same sha256-pinned bytes.
SI_SOURCED_VALUES: dict[str, SourcedValue] = {
    "differential_replicates": _si_docx(
        3,
        "Supplementary Figure S8. Relative expression level of PP_4188 gene in the "
        "control and PP_4188 strains. Proteins were extracted at 48 h. Error bars "
        "represent standard deviation from three biological replicates.",
        page="Supplementary Figure S8 caption",
        note="the replicate design of the PP_4188 strain's proteomics, and the only "
        "statement of it. Tables S4 and S5 are 'from PP_4188 strain' and the same 48 h "
        "extraction, and their 'P-Value (Equal Variance)' column needs a per-group "
        "replicate set, but neither table releases a per-replicate value or an "
        "uncertainty, so protein_fold_change_se is a typed gap",
    ),
    "differential_down_caption": _si_docx(
        145,
        "Supplementary Table S4. List of downregulated genes from PP_4188 strain",
        page="Supplementary Table S4 caption",
        note="the caption names the strain and the direction; the row count is measured "
        "on the pinned docx and asserted by DIFFERENTIAL_PANELS",
    ),
    "differential_up_caption": _si_docx(
        193,
        "Supplementary Table S5. List of upregulated proteins from PP_4188 strain",
        page="Supplementary Table S5 caption",
        note="the caption names the strain and the direction; the row count is measured "
        "on the pinned docx and asserted by DIFFERENTIAL_PANELS",
    ),
}

CHASSIS_STRAIN_SOURCE = SOURCED_VALUES["chassis_strain"]
HOST_STRAIN_SOURCE = SOURCED_VALUES["host_strain"]
EFFECTOR = SOURCED_VALUES["effector"]
GUIDE_LENGTH: int = int(SOURCED_VALUES["guide_length"].value)
PRODUCTION_CULTURE: dict[str, Any] = dict(SOURCED_VALUES["production_culture"].value)
TEMPERATURE_C = SOURCED_VALUES["temperature_c"]
DURATION_HOURS = SOURCED_VALUES["duration_hours"]
AEROBICITY = SOURCED_VALUES["aerobicity"]
SCREEN_SAMPLES: int = int(SOURCED_VALUES["screen_samples"].value)
NOT_DETECTED_COUNT: int = int(SOURCED_VALUES["not_detected_count"].value)
ARRAY_N_REPLICATES: int = int(SOURCED_VALUES["array_replicates"].value)
TITER_UNIT_SOURCE = SOURCED_VALUES["titer_unit"]
TITER_REPLICATES_SOURCE = SOURCED_VALUES["titer_replicates"]
#: The released replicate range, and the count taken from it. Back-solving is precluded
#: (the deposit carries no SD and no companion statistic), so CLAUDE.md's range rule
#: takes the conservative LOWER end: a smaller n never overstates precision.
TITER_REPLICATE_RANGE: tuple[int, int] = (
    int(TITER_REPLICATES_SOURCE.value[0]),
    int(TITER_REPLICATES_SOURCE.value[1]),
)
TITER_N_REPLICATES: int = TITER_REPLICATE_RANGE[0]
#: The deposited panel is one displayed number per (strain, protein) with no replicate
#: column and no uncertainty, so the replicate support of one stored abundance is one
#: sample. The same conservative reading as :data:`TITER_N_REPLICATES`, and the same
#: count Table S3's family carries for the same reason.
PANEL_N_REPLICATES = 1
DIFFERENTIAL_N_REPLICATES: int = int(SI_SOURCED_VALUES["differential_replicates"].value)

#: ``reference_basis`` of the Table S3 and Tables S8-S12 families: WHAT the denominator
#: is, carrying the source's own clause rather than a paraphrase of it.
REFERENCE_BASIS = (
    "the same protein in the control strain, which carries the non-targeting sgRNA: "
    "'relative expression levels of target genes in comparison to the control strains' "
    "(Fig. 3I), 'nontarget (control)' (Fig. 3G)"
)
#: ``reference_basis`` of the Tables S4 and S5 differential, from the Fig. 5B caption.
DIFFERENTIAL_REFERENCE_BASIS = (
    "the same protein in the control strain, which carries the non-targeting sgRNA: "
    "'shotgun proteomic analysis from strain PP_4188 strain in comparison to the "
    "control strain' (Fig. 5B)"
)
#: The multiple-testing correction behind the Tables S4 and S5 p-values. ``None``
#: because none is named: the Fig. 5B caption calls the column a "student's t-test
#: $p$ -value" with an applied threshold of 0.05 and no adjustment, and the only FDR in
#: the paper is DIA-NN's IDENTIFICATION filter on precursors and protein groups
#: (SOURCED_VALUES['identification_fdr']), applied before any contrast was computed. The
#: class requires the method whenever adjusted p-values are stored and forbids it
#: otherwise, so the raw p-values go in ``protein_fold_change_p_value`` and the adjusted
#: map stays None.
DIFFERENTIAL_P_VALUE_ADJUSTMENT: str | None = None
if [n for _, _, n in DIFFERENTIAL_PANELS] != [
    int(SI_SOURCED_VALUES["differential_down_caption"].value),
    int(SI_SOURCED_VALUES["differential_up_caption"].value),
]:
    raise RuntimeError(
        "DIFFERENTIAL_PANELS no longer holds the row counts its captions are sourced for"
    )
#: What Tables S4 and S5 release that this loader reads but does NOT store, and why.
#: The ``P-Value (Equal Variance)`` column IS stored now, as
#: ``protein_fold_change_p_value`` on ``ProteinFoldChangePhenotype``; what is left is
#: one reversible transform of it and one presentation index.
DIFFERENTIAL_NOT_STORED: tuple[str, ...] = (
    "the '(-Log10(P-Value))' column: the same quantity as the stored "
    "protein_fold_change_p_value by exact arithmetic (p = 10 ** -x), so storing both "
    "would store one number twice. It is read and asserted as a build oracle instead: "
    "10 ** -(-Log10(P-Value)) reproduces the released p to its own printed precision on "
    "338 of 338 rows (worst relative disagreement 3.2489e-3, Table S4's '1.30E-06' "
    "against a released -log10 of 5.884647992)",
    "the 'Rank' column: a presentation index of the released sort order (1 to 145 by "
    "ascending fold change in Table S4, 1 to 193 by descending fold change in Table "
    "S5, both measured), so it carries nothing the stored fold change does not. Read "
    "and asserted as a build oracle",
)

#: What the paper releases that no loader here consumes, with the reason for each.
NOT_LOADED: tuple[str, ...] = (
    f"the Supplementary Note 1 Benchling input-table page ({BENCHLING_INPUT_URL}) is "
    f"DEPOSITED "
    f"as {BENCHLING_TITER_REL} ({BENCHLING_TITER_BYTES} B, sha256 "
    f"{BENCHLING_TITER_SHA256}) and IS loaded: it is the only released per-strain "
    "isoprenol titer of this paper, and its control row supplies the reference titer "
    "whose absence previously kept the ProductTiter family unbuilt. "
    f"IsoprenolTiterYunus2026Dataset stores the titer and "
    "CrispriPanelProteomeYunus2026Dataset stores the 253 protein columns beside it. "
    f"RetrievalMethod.manual_browser; {BENCHLING_DEPOSIT_NOTE} MANUAL RECIPE, re-run "
    f"on rebuild: {BENCHLING_MANUAL_RECIPE} {BENCHLING_PASTE_CAVEAT}",
    f"the Supplementary Note 1 Benchling analysis page ({BENCHLING_ANALYSIS_URL}) is "
    f"DEPOSITED "
    f"as {BENCHLING_CORRELATION_REL} ({BENCHLING_CORRELATION_BYTES} B, sha256 "
    f"{BENCHLING_CORRELATION_SHA256}) and is RECORDED ONLY, never stored as a "
    "phenotype: its three columns are Protein, "
    "Correlation_with_isoprenol_production and p_value, one row per protein, which is "
    "a DERIVED statistic over a strain panel rather than a measurement of any one "
    "strain, and no phenotype class models a per-protein correlation with its own "
    f"test. Measured on the pinned bytes: {BENCHLING_CORRELATION_ROWS} rows, p up to "
    "0.9989 (so it is the UNFILTERED result frame of Supplementary Note 1's script, "
    "not its p < 0.05 output), over a protein set WIDER than the deposited panel's -- "
    f"{BENCHLING_ACCESSIONS - 1} of the panel's {BENCHLING_ACCESSIONS} accessions "
    "appear in it and one does not, so the two pages are not input and output of each "
    "other on one protein set",
    "the per-strain isoprenol titers as PLOTTED (Fig. 4B-J, Supplementary Figs. S6, S7 "
    "and S9) and their error bars: the figures are the only place the SD of a titer "
    "appears, and a bar chart is not machine-readable, so every stored titer carries a "
    "typed gap on its uncertainty rather than a number read off a figure",
    "Supplementary Tables S4 and S5's -log10 p-value and rank columns: the Fold Change "
    "AND the 'P-Value (Equal Variance)' columns ARE loaded, by "
    f"CrispriDifferentialProteomeYunus2026Dataset ({DIFFERENTIAL_NOT_STORED[0]})",
    "Supplementary Table S6 (plasmids) and the three sequencing primers of Table S7 "
    "(IY77, IY169, IY425): genotype and method metadata, not measurements",
    "Fig. 5A (TCA metabolite concentrations at 24, 48 and 72 h), Fig. 3C/D (terminal "
    "OD600 and RFP fluorescence) and Fig. 3F (the PP_1607 growth curve): figures only",
    f"PRIDE {PRIDE_ACCESSION}: the raw DIA mass-spectrometry files, which no loader "
    "here consumes",
    f"mmc2 and beyond do not exist: mmc1..mmc4 x docx/xlsx/pdf/zip/csv were probed on "
    f"ars.els-cdn.com for PII {PAPER_PII} on 2026-10-07 and only mmc1.docx answered "
    "(HTTP 206); every other combination answered 404",
)


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (both mirrors live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/yunusPredictiveCRISPRmediatedGene2026``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def library_dir(data_root: str | None = None) -> Path:
    """The literature mirror of this key (where ``paper.md`` and the SI were captured)."""
    return Path(data_root or _data_root()) / LIBRARY_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: ``{raw file name: pinned sha256}``, the build-time check of every consumed file.
DATA_SHA256: dict[str, str] = {SI_DOCX: SI_DOCX_SHA256}


def si_retrieval() -> RetrievalRecord:
    """How ``mmc1.docx`` is fetched: a plain GET of the Elsevier asset CDN.

    Directly scriptable, so the recorded retrieval re-runs as-is; the literature
    mirror's own record for this key carries the same URL, retriever and digest.
    """
    return RetrievalRecord(
        method=RetrievalMethod.direct_url,
        source_url=SI_SOURCE_URL,
        retriever="torchcell.literature.retrieve.elsevier_mmc",
        params={"pii": PAPER_PII, "filename": "mmc1.docx"},
        sha256=SI_DOCX_SHA256,
        retrieved_at=_SI_RETRIEVED_AT,
    )


def extraction_record() -> ProcessingRecord:
    """The deterministic recipe that turns the docx into the parsed tables."""
    return ProcessingRecord(
        processor="torchcell.datasets.pputida.yunus2026.read_docx_tables",
        tool="python-stdlib (zipfile + xml.etree.ElementTree)",
        version="1",
        params={
            "member": "word/document.xml",
            "cell_text": "the concatenated w:t runs of each w:p, joined by a space",
            "table_count": SI_TABLE_COUNT,
            "table_digests": TABLE_DIGESTS,
        },
        input_sha256=[SI_DOCX_SHA256],
    )


def benchling_deposits() -> tuple[tuple[str, str, str, str], ...]:
    """``(relpath, role, sha256, what it is)`` of the two hand-deposited Benchling tables.

    The digests are read HERE rather than captured in a module constant, so a test that
    re-points the pins at a synthetic deposit re-points this too.
    """
    return (
        (
            BENCHLING_TITER_REL,
            ROLE_RAW_DATA,
            BENCHLING_TITER_SHA256,
            "the per-strain input table of Supplementary Note 1's Pearson analysis: "
            "'strain', 'isoprenol_production' and 253 UniProt protein columns over "
            f"{BENCHLING_TITER_ROWS} strain rows. The only released per-strain "
            "isoprenol titer of this paper",
        ),
        (
            BENCHLING_CORRELATION_REL,
            ROLE_SI_DATA,
            BENCHLING_CORRELATION_SHA256,
            "the Pearson OUTPUT table: 'Protein', "
            "'Correlation_with_isoprenol_production', 'p_value' over "
            f"{BENCHLING_CORRELATION_ROWS} protein rows. RECORDED ONLY -- a derived "
            "per-protein statistic over a strain panel, which no phenotype class "
            "models, so no record stores it",
        ),
    )


def benchling_artifact_records(root: Path) -> list[ArtifactRecord]:
    """The two manual-deposit records, verified in place under ``root``.

    The owner pasted these bytes out of a rendered share page and deposited them, so
    this function never copies or fetches: it asserts each file is present with its
    pinned sha256 and describes it. An absent or altered file raises WITH the manual
    recipe, which is the only way to produce it again.
    """
    records: list[ArtifactRecord] = []
    for relpath, role, expected, description in benchling_deposits():
        path = root / relpath
        if not path.exists():
            raise RuntimeError(
                f"{path} is missing; the Benchling tables are a manual deposit. "
                f"MANUAL RECIPE: {BENCHLING_MANUAL_RECIPE}"
            )
        got = _sha256(path)
        if got != expected:
            raise RuntimeError(
                f"{path} has sha256 {got}, pinned {expected}; the deposited bytes "
                "changed and must become a NEW provenance record. MANUAL RECIPE: "
                f"{BENCHLING_MANUAL_RECIPE}"
            )
        records.append(
            ArtifactRecord(
                path=relpath,
                role=role,
                bytes=path.stat().st_size,
                sha256=expected,
                source=BENCHLING_INPUT_URL
                if relpath == BENCHLING_TITER_REL
                else BENCHLING_ANALYSIS_URL,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.manual_browser,
                    source_url=BENCHLING_INPUT_URL
                    if relpath == BENCHLING_TITER_REL
                    else BENCHLING_ANALYSIS_URL,
                    retriever="manual",
                    params={
                        "retrieval_command": BENCHLING_MANUAL_RECIPE,
                        "paste_caveat": BENCHLING_PASTE_CAVEAT,
                        "page_mapping": BENCHLING_PAGE_MAPPING,
                        "deposit_note": BENCHLING_DEPOSIT_NOTE,
                        "retrieved_by": BENCHLING_RETRIEVED_BY,
                        "deposit_record": BENCHLING_DEPOSIT_RECORD,
                        "checksums": BENCHLING_CHECKSUMS,
                        "contents": description,
                    },
                    sha256=expected,
                    retrieved_at=BENCHLING_RETRIEVED_AT,
                ),
            )
        )
    return records


def retrieve_raw_files(into: str | Path) -> dict[str, Path]:
    """Re-run the recorded retrieval into ``into`` and verify the pinned sha256."""
    from torchcell.literature.provenance import run_retriever

    target = Path(into)
    target.mkdir(parents=True, exist_ok=True)
    record = si_retrieval()
    payload = run_retriever(record)
    got = hashlib.sha256(payload).hexdigest()
    if got != SI_DOCX_SHA256:
        raise RuntimeError(
            f"{SI_SOURCE_URL} now yields sha256 {got}, pinned {SI_DOCX_SHA256}; "
            "upstream changed and must become a NEW provenance record"
        )
    path = target / SI_DOCX
    path.write_bytes(payload)
    return {SI_DOCX: path}


def deposit_raw_mirror(*, source: str | Path, data_root: str | None = None) -> Path:
    """Write the raw mirror (``mmc1.docx``) and its ``manifest.json``.

    Idempotent by sha256: an existing mirror file whose digest matches is left alone and
    a differing one raises rather than being overwritten. The source is verified BEFORE
    anything is written, so a refusal leaves no partial deposit.

    The two Benchling tables are NOT copied here: the owner deposited those bytes by
    hand under ``data/benchling/`` (:func:`benchling_artifact_records`), so they are
    verified in place and described, and an absent or drifted one raises with the manual
    recipe.
    """
    src = Path(source)
    got = _sha256(src)
    if got != SI_DOCX_SHA256:
        raise RuntimeError(f"{src} sha256 mismatch: got {got}, want {SI_DOCX_SHA256}")
    root = raw_mirror_dir(data_root)
    dest = root / SI_MIRROR_RELPATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if _sha256(dest) != SI_DOCX_SHA256:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    else:
        shutil.copy2(src, dest)
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=PAPER_TITLE,
        files=[
            ArtifactRecord(
                path=SI_MIRROR_RELPATH,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=SI_DOCX_SHA256,
                source=SI_SOURCE_URL,
                retrieval=si_retrieval(),
                processing=extraction_record(),
            ),
            *benchling_artifact_records(root),
        ],
        si_data_sources=[
            SI_SOURCE_URL,
            BENCHLING_ANALYSIS_URL,
            BENCHLING_INPUT_URL,
            f"https://www.ebi.ac.uk/pride/archive/projects/{PRIDE_ACCESSION}",
        ],
        si_expected=list(NOT_LOADED),
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return root


def load_manifest(data_root: str | None = None) -> Manifest:
    """Read the raw mirror's ``manifest.json``."""
    path = raw_mirror_dir(data_root) / "manifest.json"
    return Manifest.model_validate_json(path.read_text())


def manifest_sha256(manifest: Manifest, relpath: str) -> str:
    """The recorded sha256 of one mirror file."""
    for record in manifest.files:
        if record.path == relpath:
            return record.sha256
    raise KeyError(f"{relpath} is not in the raw-mirror manifest")


# --------------------------------------------------------------------------- #
# Reading the supplementary tables out of the .docx
# --------------------------------------------------------------------------- #
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class TableExtractionError(RuntimeError):
    """The pinned docx no longer parses to the structure this loader was written on."""


def _paragraph_text(paragraph: ElementTree.Element) -> str:
    """The visible text of one ``w:p``: its ``w:t`` runs concatenated."""
    return "".join(node.text or "" for node in paragraph.iter(f"{_W}t")).strip()


def read_docx_tables(path: str | Path) -> list[list[list[str]]]:
    """Every table of a ``.docx`` as ``[table][row][cell]`` of stripped text.

    Pure stdlib, so the extraction adds no dependency and is byte-deterministic for a
    pinned file: ``word/document.xml`` is read, each ``w:tbl`` walked in document order,
    and each ``w:tc``'s paragraphs joined by a single space.
    """
    with zipfile.ZipFile(path) as archive:
        document = archive.read("word/document.xml")
    body = ElementTree.fromstring(document).find(f"{_W}body")
    if body is None:
        raise TableExtractionError(f"{path}: word/document.xml has no w:body")
    tables: list[list[list[str]]] = []
    for table in body.findall(f"{_W}tbl"):
        rows: list[list[str]] = []
        for row in table.findall(f"{_W}tr"):
            rows.append(
                [
                    " ".join(_paragraph_text(p) for p in cell.findall(f"{_W}p")).strip()
                    for cell in row.findall(f"{_W}tc")
                ]
            )
        tables.append(rows)
    return tables


def table_digest(rows: Sequence[Sequence[str]]) -> str:
    """sha256 of one parsed table: tab-joined cells, newline-joined rows."""
    payload = "\n".join("\t".join(cell for cell in row) for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def supplementary_tables(path: str | Path) -> dict[str, list[list[str]]]:
    """``{'S1': rows, ...}`` for the twelve Supplementary Tables, structure asserted.

    The table COUNT, each consumed table's HEADER and each consumed table's parsed
    digest are checked, so a re-released docx or a changed parser stops the build
    instead of silently shifting a column.
    """
    tables = read_docx_tables(path)
    if len(tables) != SI_TABLE_COUNT:
        raise TableExtractionError(
            f"{osp.basename(str(path))} holds {len(tables)} tables, pinned "
            f"{SI_TABLE_COUNT}"
        )
    out = {name: tables[index] for name, index in TABLE_INDEX.items()}
    headers = {
        "S1": TABLE_S1_HEADER,
        "S2": TABLE_S2_HEADER,
        "S3": TABLE_S3_HEADER,
        "S7": TABLE_S7_HEADER,
    }
    for name, expected in headers.items():
        got = tuple(out[name][0])
        if got != expected:
            raise TableExtractionError(
                f"Table {name} header is {got!r}, pinned {expected!r}"
            )
    for name, protein in ARRAY_PANELS:
        expected_array = ("", "Replicate", f"Relative expression level of {protein}")
        got = tuple(out[name][0])
        if got != expected_array:
            raise TableExtractionError(
                f"Table {name} header is {got!r}, pinned {expected_array!r}"
            )
    for name, pinned in TABLE_DIGESTS.items():
        digest = table_digest(out[name])
        if digest != pinned:
            raise TableExtractionError(
                f"Table {name} parsed sha256 {digest}, pinned {pinned}"
            )
    return out


class ScreenRow(BaseModel):
    """One Table S3 row: a strain, its CRISPRi target, and the released ratio."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strain: str
    target: str
    locus_tag: str
    variant: str | None
    relative_expression: float | None

    @property
    def not_detected(self) -> bool:
        """True for an ``n.d.`` row: the control strain had no detected expression."""
        return self.relative_expression is None


def parse_table_s3(rows: Sequence[Sequence[str]]) -> list[ScreenRow]:
    """Parse Table S3 into typed rows; the row count and the ``n.d.`` count are checked."""
    parsed: list[ScreenRow] = []
    for row in rows[1:]:
        strain, target, value = (cell.strip() for cell in row[:3])
        match = S3_TARGET_RE.match(target)
        if match is None:
            raise TableExtractionError(
                f"Table S3 target {target!r} is not PP_<4 digits>[_NT<n>]"
            )
        parsed.append(
            ScreenRow(
                strain=strain,
                target=target,
                locus_tag=match.group("tag"),
                variant=match.group("variant"),
                relative_expression=None if value == NOT_DETECTED else float(value),
            )
        )
    if len(parsed) != TABLE_S3_ROWS:
        raise TableExtractionError(
            f"Table S3 holds {len(parsed)} rows; the Results state {SCREEN_SAMPLES} "
            "samples and the pinned table has one row per sample"
        )
    strains = {row.strain for row in parsed}
    if len(strains) != len(parsed):
        raise TableExtractionError("Table S3 repeats a strain name")
    n_missing = sum(1 for row in parsed if row.not_detected)
    if n_missing != NOT_DETECTED_COUNT:
        raise TableExtractionError(
            f"Table S3 holds {n_missing} 'n.d.' rows; the Results state "
            f"{NOT_DETECTED_COUNT}"
        )
    return parsed


class DifferentialRow(BaseModel):
    """One row of Table S4 or S5: a protein and its fold change against the control."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    table: str
    direction: str
    accession: str
    entry_name: str
    protein: str
    description: str
    fold_change: float
    log2_fold_change: float
    #: The released ``P-Value (Equal Variance)``, an unadjusted Student's t-test
    #: p-value, STORED as ``protein_fold_change_p_value``.
    p_value: float
    #: Read for the build oracle and not stored: the reversible transform of
    #: :attr:`p_value` (:data:`DIFFERENTIAL_NOT_STORED`).
    neg_log10_p_value: float
    rank: int


#: How far ``log2(Fold Change)`` may sit from the released ``Log2(Fold Change)``.
#: Measured on the pinned docx: zero of 338 rows disagree beyond this.
_LOG2_TOL = 1e-6
#: How far ``10 ** -(-Log10(P-Value))`` may sit from the released p, relatively. The
#: released p is printed to three significant figures for the 23 cells in scientific
#: notation, so the worst measured disagreement is 3.2488e-3 (Table S4's '1.30E-06'
#: against a released -log10 of 5.884647992, i.e. p = 1.3042e-6).
_P_VALUE_TOL = 5e-3


def parse_differential(
    rows: Sequence[Sequence[str]], *, table: str, direction: str
) -> list[DifferentialRow]:
    """Parse Table S4 or S5 into typed rows, with every derived column asserted.

    Four oracles on the deposited bytes, each measured before it was written here:
    ``log2(Fold Change)`` reproduces the released log2 column exactly (0 of 338 rows
    disagree at 1e-6); ``10 ** -(-Log10(P-Value))`` reproduces the released p to its own
    printed precision; the ranks are ``1..n`` in released row order; and every released
    row clears the thresholds its direction implies (fold change below 1 for the
    downregulated table and above 1 for the upregulated one, every |log2| at least 1 and
    every p below 0.05). A column shift or a re-released table stops the build.
    """
    parsed: list[DifferentialRow] = []
    for row in rows[1:]:
        cells = [cell.strip() for cell in row[:9]]
        parsed.append(
            DifferentialRow(
                table=table,
                direction=direction,
                accession=cells[0],
                entry_name=cells[1],
                protein=cells[2],
                description=cells[3],
                fold_change=float(cells[4]),
                log2_fold_change=float(cells[5]),
                p_value=float(cells[6]),
                neg_log10_p_value=float(cells[7]),
                rank=int(cells[8]),
            )
        )
    if not parsed:
        raise TableExtractionError(f"Table {table} holds no rows")
    if len({entry.protein for entry in parsed}) != len(parsed):
        raise TableExtractionError(f"Table {table} repeats a protein key")
    if [entry.rank for entry in parsed] != list(range(1, len(parsed) + 1)):
        raise TableExtractionError(
            f"Table {table}'s Rank column is not 1..{len(parsed)} in released row order"
        )
    for entry in parsed:
        if abs(math.log2(entry.fold_change) - entry.log2_fold_change) > _LOG2_TOL:
            raise TableExtractionError(
                f"Table {table}/{entry.protein}: log2({entry.fold_change}) is "
                f"{math.log2(entry.fold_change)} and the released Log2(Fold Change) "
                f"reads {entry.log2_fold_change}; the two columns are not one quantity"
            )
        recovered = 10.0**-entry.neg_log10_p_value
        if abs(recovered - entry.p_value) > _P_VALUE_TOL * entry.p_value:
            raise TableExtractionError(
                f"Table {table}/{entry.protein}: the released -log10 p "
                f"{entry.neg_log10_p_value} recovers p {recovered} against a released "
                f"{entry.p_value}, beyond the printed precision of the p column"
            )
        if entry.p_value >= 0.05 or abs(entry.log2_fold_change) < 1.0:
            raise TableExtractionError(
                f"Table {table}/{entry.protein}: p {entry.p_value} and log2 fold change "
                f"{entry.log2_fold_change} do not clear the thresholds every released "
                "row of this table clears"
            )
        if direction == "downregulated" and entry.fold_change >= 1.0:
            raise TableExtractionError(
                f"Table {table}/{entry.protein}: fold change {entry.fold_change} in the "
                "downregulated table"
            )
        if direction == "upregulated" and entry.fold_change <= 1.0:
            raise TableExtractionError(
                f"Table {table}/{entry.protein}: fold change {entry.fold_change} in the "
                "upregulated table"
            )
    return parsed


def read_differential(
    tables: Mapping[str, list[list[str]]],
    panels: Sequence[tuple[str, str, int]] | None = None,
) -> list[DifferentialRow]:
    """Both halves of the PP_4188 differential, with each panel's row count asserted."""
    rows: list[DifferentialRow] = []
    for table, direction, expected in panels or DIFFERENTIAL_PANELS:
        panel = parse_differential(tables[table], table=table, direction=direction)
        if len(panel) != expected:
            raise TableExtractionError(
                f"Table {table} holds {len(panel)} rows, pinned {expected}"
            )
        rows.extend(panel)
    keys = [entry.protein for entry in rows]
    if len(set(keys)) != len(keys):
        raise TableExtractionError(
            "a protein key appears in both the downregulated and the upregulated table; "
            "one strain cannot move both ways"
        )
    return rows


class ArrayCell(BaseModel):
    """One (construct, measured protein, replicate) cell of Tables S8-S12."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    construct_name: str
    guide_targets: tuple[str, ...]
    protein: str
    replicate: str
    relative_expression: float


def parse_construct(name: str) -> tuple[str, ...]:
    """The guide targets of an array construct name, e.g. ``PP_4188_0528_4160``.

    The released names abbreviate every target after the first to its four digits, so
    the tags are rebuilt from the digit groups and the name is re-serialized and
    compared: a name that does not round-trip stops the build rather than losing a guide.
    """
    digits = FOUR_DIGITS_RE.findall(name)
    if not digits or f"PP_{'_'.join(digits)}" != name:
        raise TableExtractionError(
            f"array construct {name!r} is not PP_<4 digits>(_<4 digits>)*"
        )
    return tuple(f"PP_{group}" for group in digits)


def parse_array_panel(rows: Sequence[Sequence[str]], protein: str) -> list[ArrayCell]:
    """Parse one Fig. 3J-N panel (Tables S8-S12) into typed per-replicate cells."""
    cells: list[ArrayCell] = []
    for row in rows[1:]:
        construct, replicate, value = (cell.strip() for cell in row[:3])
        if replicate not in ARRAY_REPLICATES:
            raise TableExtractionError(
                f"{protein} panel replicate label {replicate!r} is not in "
                f"{ARRAY_REPLICATES}"
            )
        cells.append(
            ArrayCell(
                construct_name=construct,
                guide_targets=parse_construct(construct),
                protein=protein,
                replicate=replicate,
                relative_expression=float(value),
            )
        )
    seen = {(cell.construct_name, cell.replicate) for cell in cells}
    if len(seen) != len(cells):
        raise TableExtractionError(
            f"{protein} panel repeats a (construct, replicate); a repeated replicate "
            "would shrink the SE"
        )
    return cells


class TargetListRow(BaseModel):
    """One row of Table S1 or S2: a screened target and its annotation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    table: str
    selection: str
    number: str
    locus_tag: str
    enzyme: str
    enzyme_description: str
    kegg_orthology: str
    metabolic_pathway: str


def parse_target_list(
    rows: Sequence[Sequence[str]], *, table: str
) -> list[TargetListRow]:
    """Parse Table S1 (intuition) or S2 (FluxRETAP) into typed annotation rows."""
    selection = "intuition" if table == "S1" else "fluxretap"
    out: list[TargetListRow] = []
    for row in rows[1:]:
        cells = [cell.strip() for cell in row] + [""] * (7 - len(row))
        if LOCUS_TAG_RE.match(cells[1]) is None:
            raise TableExtractionError(
                f"Table {table} target {cells[1]!r} is not a PP_ locus tag"
            )
        out.append(
            TargetListRow(
                table=table,
                selection=selection,
                number=cells[0],
                locus_tag=cells[1],
                enzyme=cells[2],
                enzyme_description=cells[3],
                kegg_orthology=cells[4],
                metabolic_pathway=cells[5],
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Guide spacers (Table S7) mapped onto the pinned annotation
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Reading the two hand-deposited Benchling tables
# --------------------------------------------------------------------------- #
class BenchlingStrainRow(BaseModel):
    """One row of the deposited panel: a row label, a titer, and 253 abundances."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    isoprenol_production: float
    abundance: dict[str, float]

    @property
    def is_control(self) -> bool:
        """Whether this is the control strain's row, which both families reference."""
        return self.label == BENCHLING_CONTROL_LABEL

    @property
    def marked(self) -> bool:
        """Whether the label carries the undefined ``" (S)"`` marker."""
        return self.label.endswith(BENCHLING_SOLID_MARKER)


class BenchlingStrainTable(BaseModel):
    """The whole deposited panel, with its control row separated out."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    accessions: tuple[str, ...]
    rows: tuple[BenchlingStrainRow, ...]
    zero_cells: int

    @property
    def control(self) -> BenchlingStrainRow:
        """The control row; the reader refuses a table without exactly one."""
        return next(row for row in self.rows if row.is_control)

    @property
    def strains(self) -> tuple[BenchlingStrainRow, ...]:
        """Every non-control row, in released order."""
        return tuple(row for row in self.rows if not row.is_control)

    @property
    def titer_by_label(self) -> dict[str, float]:
        """``label -> isoprenol_production`` over every released row."""
        return {row.label: row.isoprenol_production for row in self.rows}


def read_benchling_strain_table(path: str | Path) -> BenchlingStrainTable:
    """Parse the deposited ``strain, isoprenol_production, <accessions>`` TSV.

    Every refusal is a changed release rather than a repair: a header whose first two
    columns are not the two Supplementary Note 1's own script reads
    (``data.columns[2:]`` are the proteins), a repeated accession or strain label, a
    missing or repeated control row, a blank cell, or a non-finite number.
    """
    text = Path(path).read_text()
    lines = [line for line in text.split("\n") if line != ""]
    if not lines:
        raise TableExtractionError(f"{path} is empty")
    header = lines[0].split("\t")
    if header[:2] != ["strain", "isoprenol_production"]:
        raise TableExtractionError(
            f"{path} header starts {header[:2]!r}, not ['strain', "
            "'isoprenol_production']; Supplementary Note 1's script reads the proteins "
            "as data.columns[2:], so those two columns are the file's contract"
        )
    accessions = tuple(header[2:])
    if not accessions:
        raise TableExtractionError(f"{path} carries no protein column")
    if len(set(accessions)) != len(accessions):
        repeated = sorted({a for a in accessions if accessions.count(a) > 1})
        raise TableExtractionError(f"{path} repeats the protein column(s) {repeated}")
    rows: list[BenchlingStrainRow] = []
    zero_cells = 0
    for line in lines[1:]:
        cells = line.split("\t")
        if len(cells) != len(header):
            raise TableExtractionError(
                f"{path} row {cells[0]!r} has {len(cells)} cells, header has "
                f"{len(header)}"
            )
        values: dict[str, float] = {}
        for accession, cell in zip(accessions, cells[2:], strict=True):
            if cell == "":
                raise TableExtractionError(
                    f"{path} row {cells[0]!r} column {accession} is blank; the "
                    "deposited panel releases a number in every cell and a blank would "
                    "be an absence this loader has no sourced rule for"
                )
            value = float(cell)
            if not math.isfinite(value):
                raise TableExtractionError(
                    f"{path} row {cells[0]!r} column {accession} is {cell!r}"
                )
            if value == 0.0:
                zero_cells += 1
            values[accession] = value
        titer = float(cells[1])
        if not math.isfinite(titer) or titer < 0.0:
            raise TableExtractionError(
                f"{path} row {cells[0]!r} has isoprenol_production {cells[1]!r}"
            )
        rows.append(
            BenchlingStrainRow(
                label=cells[0], isoprenol_production=titer, abundance=values
            )
        )
    labels = [row.label for row in rows]
    if len(set(labels)) != len(labels):
        repeated = sorted({label for label in labels if labels.count(label) > 1})
        raise TableExtractionError(
            f"{path} repeats the strain row(s) {repeated}; one row is one strain and a "
            "repeat would store two titers for one identity"
        )
    controls = [row for row in rows if row.is_control]
    if len(controls) != 1:
        raise TableExtractionError(
            f"{path} carries {len(controls)} rows labeled "
            f"{BENCHLING_CONTROL_LABEL!r}; that row is the reference of both deposited "
            "families and nothing else supplies one"
        )
    return BenchlingStrainTable(
        accessions=accessions, rows=tuple(rows), zero_cells=zero_cells
    )


class BenchlingCorrelationRow(BaseModel):
    """One row of the deposited Pearson output: a protein, its r and its p."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protein: str
    correlation: float
    p_value: float


def read_benchling_correlation_table(
    path: str | Path,
) -> tuple[BenchlingCorrelationRow, ...]:
    """Parse the deposited correlation TSV, which is RECORDED and never stored.

    It is read so the deposit is proven parseable and its shape asserted; no record is
    written from it (:data:`NOT_LOADED`).
    """
    text = Path(path).read_text()
    lines = [line for line in text.split("\n") if line != ""]
    header = lines[0].split("\t")
    expected = ["Protein", "Correlation_with_isoprenol_production", "p_value"]
    if header != expected:
        raise TableExtractionError(f"{path} header is {header!r}, not {expected!r}")
    rows: list[BenchlingCorrelationRow] = []
    for line in lines[1:]:
        cells = line.split("\t")
        if len(cells) != 3:
            raise TableExtractionError(
                f"{path} row {cells[0]!r} has {len(cells)} cells"
            )
        correlation = float(cells[1])
        p_value = float(cells[2])
        if not -1.0 <= correlation <= 1.0:
            raise TableExtractionError(
                f"{path} row {cells[0]!r} has correlation {correlation}, outside [-1, 1]"
            )
        if not 0.0 <= p_value <= 1.0:
            raise TableExtractionError(
                f"{path} row {cells[0]!r} has p_value {p_value}, outside [0, 1]"
            )
        rows.append(
            BenchlingCorrelationRow(
                protein=cells[0], correlation=correlation, p_value=p_value
            )
        )
    proteins = [row.protein for row in rows]
    if len(set(proteins)) != len(proteins):
        repeated = sorted({p for p in proteins if proteins.count(p) > 1})
        raise TableExtractionError(f"{path} repeats the protein row(s) {repeated}")
    return tuple(rows)


class LabelReconciliation(BaseModel):
    """How each deposited row label reaches a target this loader already knows.

    Three documented routes and nothing else: the label as released, the label without
    the undefined ``" (S)"`` marker, and the label without a trailing ``_NT<digit>``
    guide-variant token. ``unmatched`` is what no route reaches, which must be the
    control row alone.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    known_targets: int
    exact: tuple[str, ...]
    after_marker_strip: tuple[str, ...]
    after_variant_strip: tuple[str, ...]
    unmatched: tuple[str, ...]

    @property
    def rows(self) -> int:
        """Labels reconciled."""
        return (
            len(self.exact)
            + len(self.after_marker_strip)
            + len(self.after_variant_strip)
            + len(self.unmatched)
        )


def reconcile_benchling_labels(
    labels: Sequence[str], known: Iterable[str]
) -> LabelReconciliation:
    """Split the deposited row labels by which documented route reaches a known target.

    The row labels are CRISPRi target labels, not strain names, so they are reconciled
    against what the supplementary tables already name: Table S3's
    ``CRISPRi target gene`` column (its labels AND the locus tags they parse to) plus
    Tables S1 and S2's locus tags.
    """
    targets = set(known)
    exact: list[str] = []
    marker: list[str] = []
    variant: list[str] = []
    unmatched: list[str] = []
    for label in labels:
        unmarked = (
            label[: -len(BENCHLING_SOLID_MARKER)]
            if label.endswith(BENCHLING_SOLID_MARKER)
            else label
        )
        if label in targets:
            exact.append(label)
        elif unmarked in targets:
            marker.append(label)
        elif VARIANT_SUFFIX_RE.sub("", unmarked) in targets:
            variant.append(label)
        else:
            unmatched.append(label)
    return LabelReconciliation(
        known_targets=len(targets),
        exact=tuple(exact),
        after_marker_strip=tuple(marker),
        after_variant_strip=tuple(variant),
        unmatched=tuple(unmatched),
    )


def benchling_target(label: str) -> tuple[str, str | None]:
    """``(locus tag, guide variant)`` of one deposited row label.

    The ``NT<n>`` token is the SAME guide-variant number Table S3 uses, measured on this
    paper's own bytes: Table S7 gives ``PP_0339_NT1`` and ``PP_0339_NT2`` distinct
    spacers and Table S3 screens ``PP_1607_NT2`` and ``PP_1607_NT4`` as two strains, so
    it is matched as part of the key rather than read as a non-targeting filler. Either
    reading names the same single perturbed locus; only the spacer lookup differs.
    """
    match = S3_TARGET_RE.match(label)
    if match is None:
        raise TableExtractionError(
            f"deposited row label {label!r} is not PP_<4 digits> with an optional "
            "_NT<digit> variant"
        )
    return match.group("tag"), match.group("variant")


def reverse_complement(sequence: str) -> str:
    """Reverse complement of an unambiguous ACGT sequence."""
    return sequence.translate(_DNA_COMPLEMENT)[::-1]


class GuideOligo(BaseModel):
    """One Table S7 sgRNA oligo pair: its label, its spacer and its two oligo ids."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    forward_oligo: str
    reverse_oligo: str
    spacer: str


class GuideAssignment(BaseModel):
    """What a record's CRISPRi target knows about its guide, and why."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    locus_tag: str
    variant: str | None
    spacer: str | None
    oligo_label: str | None
    reason: str


class GuideLibrary(BaseModel):
    """Table S7's sgRNA oligos keyed by ``(locus tag, variant)``, with the misses."""

    model_config = ConfigDict(extra="forbid")

    by_key: dict[str, GuideOligo]
    spacers_by_tag: dict[str, list[str]]
    unmapped_labels: dict[str, list[str]]
    n_pairs: int
    n_off_length: list[str]

    @staticmethod
    def key(locus_tag: str, variant: str | None) -> str:
        """The ``(tag, variant)`` key as one string, so the model stays JSON-dumpable."""
        return f"{locus_tag}|{variant or ''}"

    def assign(self, locus_tag: str, variant: str | None) -> GuideAssignment:
        """The spacer of one screened target, or the measured reason there is none.

        An exact ``(tag, variant)`` hit wins. A target with no variant label resolves
        only when the tag carries exactly one distinct spacer across all its variants;
        two distinct spacers mean the source does not state which guide the strain
        holds, and the spacer stays ``None`` rather than being picked.
        """
        exact = self.by_key.get(self.key(locus_tag, variant))
        if exact is not None:
            return GuideAssignment(
                locus_tag=locus_tag,
                variant=variant,
                spacer=exact.spacer,
                oligo_label=exact.label,
                reason="exact (locus tag, variant) oligo",
            )
        spacers = self.spacers_by_tag.get(locus_tag, [])
        if len(set(spacers)) == 1:
            only = next(
                oligo for oligo in self.by_key.values() if oligo.spacer == spacers[0]
            )
            return GuideAssignment(
                locus_tag=locus_tag,
                variant=variant,
                spacer=only.spacer,
                oligo_label=only.label,
                reason="the tag's only oligo spacer",
            )
        if not spacers:
            return GuideAssignment(
                locus_tag=locus_tag,
                variant=variant,
                spacer=None,
                oligo_label=None,
                reason="no Table S7 oligo names this locus",
            )
        return GuideAssignment(
            locus_tag=locus_tag,
            variant=variant,
            spacer=None,
            oligo_label=None,
            reason=f"{len(set(spacers))} distinct spacers name this locus and the "
            "source does not state which guide this strain carries",
        )


def _oligo_label_parts(label: str) -> tuple[str, str | None]:
    """``(gene label, variant)`` of a Table S7 oligo name's middle token.

    ``PP4549_NT1_sgRNA`` and ``PP0103_sgRNA_NT1`` are both written, so the ``sgRNA``
    and ``NT<n>`` tokens are removed wherever they sit and the remainder is the gene.
    """
    parts = label.split("_")
    variants = [part for part in parts if VARIANT_RE.match(part)]
    head = [part for part in parts if part != "sgRNA" and not VARIANT_RE.match(part)]
    return "_".join(head), (variants[0] if variants else None)


def build_guide_library(
    rows: Sequence[Sequence[str]], genome: PPutidaKT2440Genome
) -> GuideLibrary:
    """Read Table S7's sgRNA oligos into a guide library keyed on the pinned annotation.

    Every pair is validated against the BASIC flanks and against its own partner: the
    reverse oligo's spacer must be the reverse complement of the forward oligo's, which
    is an independent check that neither sequence was mis-transcribed. A label is a
    ``PP_`` tag or a gene symbol the annotation resolves; a label that resolves to no
    locus is kept in ``unmapped_labels`` and never remapped.
    """
    sides: dict[str, dict[str, tuple[str, str]]] = {}
    for row in rows[1:]:
        name, sequence = row[0].strip(), row[1].strip().upper()
        match = OLIGO_NAME_RE.match(name)
        if match is None:
            continue
        sides.setdefault(match.group("label"), {})[match.group("side")] = (
            match.group("oligo"),
            sequence,
        )
    by_key: dict[str, GuideOligo] = {}
    spacers_by_tag: dict[str, list[str]] = {}
    unmapped: dict[str, list[str]] = {}
    off_length: list[str] = []
    n_pairs = 0
    for label, pair in sorted(sides.items()):
        if set(pair) != {"F", "R"}:
            continue
        (forward_id, forward), (reverse_id, reverse) = pair["F"], pair["R"]
        if not (
            forward.startswith(OLIGO_FORWARD_PREFIX)
            and forward.endswith(OLIGO_FORWARD_SUFFIX)
            and reverse.startswith(OLIGO_REVERSE_PREFIX)
            and reverse.endswith(OLIGO_REVERSE_SUFFIX)
        ):
            raise TableExtractionError(
                f"Table S7 oligo pair {label!r} does not carry the BASIC flanks"
            )
        spacer = forward[len(OLIGO_FORWARD_PREFIX) : -len(OLIGO_FORWARD_SUFFIX)]
        partner = reverse[len(OLIGO_REVERSE_PREFIX) : -len(OLIGO_REVERSE_SUFFIX)]
        if reverse_complement(spacer) != partner:
            raise TableExtractionError(
                f"Table S7 pair {label!r}: the reverse oligo's spacer is not the "
                f"reverse complement of the forward oligo's ({spacer} / {partner})"
            )
        n_pairs += 1
        if len(spacer) != GUIDE_LENGTH:
            off_length.append(f"{label} ({len(spacer)} nt)")
        head, variant = _oligo_label_parts(label)
        tag_match = LOCUS_TAG_RE.match(head)
        if tag_match is not None:
            locus_tag = f"PP_{tag_match.group('number')}"
        else:
            resolution = genome.resolve_gene_name(head)
            if resolution.systematic_name is None or resolution.status not in (
                GeneNameStatus.CURRENT,
                GeneNameStatus.RENAMED,
                GeneNameStatus.NON_GENE_FEATURE,
            ):
                unmapped.setdefault(head, []).append(label)
                continue
            locus_tag = resolution.systematic_name
        oligo = GuideOligo(
            label=label,
            forward_oligo=forward_id,
            reverse_oligo=reverse_id,
            spacer=spacer,
        )
        by_key[GuideLibrary.key(locus_tag, variant)] = oligo
        spacers_by_tag.setdefault(locus_tag, []).append(spacer)
    unexpected = sorted(set(unmapped) - set(NON_GENE_OLIGO_LABELS))
    if unexpected:
        raise TableExtractionError(
            f"Table S7 oligo labels {unexpected} name no locus of "
            f"{BACTERIAL_ASSEMBLY_SETS['KT2440']} and are not documented in "
            "NON_GENE_OLIGO_LABELS"
        )
    return GuideLibrary(
        by_key=by_key,
        spacers_by_tag=spacers_by_tag,
        unmapped_labels=unmapped,
        n_pairs=n_pairs,
        n_off_length=off_length,
    )


# --------------------------------------------------------------------------- #
# Genotype and environment builders
# --------------------------------------------------------------------------- #
def publication() -> Publication:
    """This paper, by DOI (no PubMed id is carried in the mirrored metadata)."""
    return Publication(doi=PAPER_DOI, doi_url=f"https://doi.org/{PAPER_DOI}")


def _chassis_gap(field: str, note: str) -> ProvenanceGap:
    """A deferral of the chassis genotype to the unmirrored Banerjee 2024."""
    return ProvenanceGap(
        field=field,
        reason=ProvenanceGapReason.deferred_pending_source_review,
        looked_in=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
            page=_RESULTS_PREDICTIVE,
        ),
        resolve_with=Provenance(
            source_uri=f"https://doi.org/{CHASSIS_SOURCE_DOI}",
            citation_key="banerjee2024",
            method=CHASSIS_SOURCE_CITATION,
            page="the paper that constructed IY1452; NOT in the literature mirror",
        ),
        note=note,
    )


def chassis_background() -> BacterialStrainBackground:
    """``IY1452``: the isoprenol-producing chassis, named but never described here.

    ``alleles`` is empty and ``genotype_statement`` / ``construction`` are typed gaps,
    because this paper states only that the strain is "highly genetically engineered"
    and cites Banerjee 2024 for it. Carruthers 2025's mirrored genotype for ``IY1449b``
    / ``IY1452b`` is deliberately NOT borrowed: no mirrored byte says ``IY1452`` and
    ``IY1452b`` are the same strain.
    """
    return BacterialStrainBackground(
        name=CHASSIS_STRAIN,
        reference_strain=KT2440_STRAIN,
        assembly_set=KT2440_ASSEMBLY_SET,
        parents=["KT2440"],
        construction=None,
        genotype_statement=None,
        alleles=[],
        provenance=[CHASSIS_STRAIN_SOURCE, HOST_STRAIN_SOURCE],
        provenance_gaps=[
            _chassis_gap(
                "genotype_statement",
                "this paper writes no genotype for IY1452: the heterologous isoprenol "
                "pathway it carries and every chromosomal edit behind 'highly "
                "genetically engineered' are stated only in Banerjee 2024",
            ),
            _chassis_gap(
                "construction",
                "no construction step for IY1452 appears in this paper's Methods, "
                "which describe only the transformation of the CRISPRi plasmid into it",
            ),
        ],
    )


def chassis_reference() -> AssemblyReferenceGenome:
    """The assembly-pinned reference genome every record of this paper is written against."""
    return assembly_reference(KT2440_STRAIN, background=chassis_background())


def production_environment() -> Environment:
    """M9 with 2 % glucose at 30 C for 48 h, induced with 0.2 % L-arabinose.

    A plain ``Environment``, not a ``CultureEnvironment``: ``Experiment.environment`` is
    annotated ``Environment`` and pydantic serializes by the DECLARED type, so a shaking
    speed or a working volume would be dumped away without an error. The 180 rpm and the
    5 mL tube are recorded in :data:`SOURCED_VALUES` and in the note instead.

    The medium object is the library's ``M9`` salts, which is what the Methods name;
    the 2 % glucose that makes it a growth medium is carried as the carbon-source
    physical factor, because adding an ``M9_GLUCOSE_2PCT_YUNUS2026`` entry edits
    ``media.py``, a value-surface file whose change blocks an incremental admission.
    That library addition is raised in the PR.
    """
    return Environment(
        media=M9,
        temperature=Temperature(value=float(TEMPERATURE_C.value)),
        perturbations=[
            EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.carbon_source,
                agent=resolved_compound("D-glucose"),
                magnitude=Concentration(
                    value=float(PRODUCTION_CULTURE["glucose_percent"]),
                    unit=ConcentrationUnit.percent_w_v,
                ),
            ),
            SmallMoleculePerturbation(
                compound=resolved_compound("L-arabinose"),
                concentration=Concentration(
                    value=float(PRODUCTION_CULTURE["arabinose_percent"]),
                    unit=ConcentrationUnit.percent_w_v,
                ),
            ),
            SmallMoleculePerturbation(
                compound=resolved_compound("kanamycin"),
                concentration=Concentration(
                    value=float(PRODUCTION_CULTURE["kanamycin_mg_per_l"]),
                    unit=ConcentrationUnit.ug_per_ml,
                ),
            ),
            SmallMoleculePerturbation(
                compound=resolved_compound("gentamicin"),
                concentration=Concentration(
                    value=float(PRODUCTION_CULTURE["gentamicin_mg_per_l"]),
                    unit=ConcentrationUnit.ug_per_ml,
                ),
            ),
        ],
        aerobicity=str(AEROBICITY.value),
        duration_hours=float(DURATION_HOURS.value),
    )


def crispri_perturbation(
    locus_tag: str, gene_name: str, assignment: GuideAssignment
) -> BacterialCrisprInterferencePerturbation:
    """One dCas9 knockdown, carrying its Table S7 spacer when the source states it."""
    return BacterialCrisprInterferencePerturbation(
        systematic_gene_name=locus_tag,
        perturbed_gene_name=gene_name,
        gene_namespace=KT2440_NAMESPACE,
        crispr=CrisprConstruct(
            effector=str(EFFECTOR.value), guide_sequence=assignment.spacer, n_guides=1
        ),
    )


def relative_expression_phenotype(
    values: Mapping[str, float], *, n_replicates: int
) -> ProteinFoldChangePhenotype:
    """One record's relative target-protein expression, keyed by locus tag.

    ``n_replicates = 1`` carries no SE at all (the source releases none and states no
    replicate design behind one sample), which is a typed gap; with more than one
    replicate the SE is the sample SD over sqrt(n), the uncertainty type the Fig. 3
    caption states.
    """
    if not values:
        raise RuntimeError("a relative-expression record needs at least one protein")
    for tag, value in values.items():
        if not math.isfinite(value):
            raise RuntimeError(f"{tag}: non-finite relative expression {value}")
    return ProteinFoldChangePhenotype(
        protein_fold_change=dict(values),
        fold_change_scale=FOLD_CHANGE_SCALE,
        reference_basis=REFERENCE_BASIS,
        protein_fold_change_se=None,
        n_replicates={tag: n_replicates for tag in values},
        measurement_type=MEASUREMENT_TYPE,
        provenance_gaps=[
            ProvenanceGap(
                field="protein_fold_change_se",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="Supplementary Table S3 releases one number per strain with no "
                "uncertainty, and the replicate design behind one proteomics sample is "
                "not stated; there is nothing to derive an SE from",
            ),
            ProvenanceGap(
                field="protein_fold_change_p_value",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="Supplementary Table S3 releases no test result beside its ratio: "
                "one proteomics sample per strain cannot carry a p-value. The Tables S4 "
                "and S5 differential does, and that is a different dataset",
            ),
        ],
    )


def array_phenotype(
    means: Mapping[str, float],
    standard_errors: Mapping[str, float],
    counts: Mapping[str, int],
) -> ProteinFoldChangePhenotype:
    """One array construct's per-protein mean relative expression with its SE."""
    if set(means) != set(standard_errors) or set(means) != set(counts):
        raise RuntimeError("mean, SE and replicate-count keys disagree")
    return ProteinFoldChangePhenotype(
        protein_fold_change=dict(means),
        fold_change_scale=FOLD_CHANGE_SCALE,
        reference_basis=REFERENCE_BASIS,
        protein_fold_change_se=dict(standard_errors),
        n_replicates=dict(counts),
        measurement_type=MEASUREMENT_TYPE,
        provenance_gaps=[
            ProvenanceGap(
                field="protein_fold_change_p_value",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="Supplementary Tables S8-S12 release three per-replicate ratios "
                "per construct and no test against the control, so there is no p-value "
                "to store and none can be derived without the control's own replicates",
            )
        ],
    )


def differential_phenotype(
    values: Mapping[str, float], p_values: Mapping[str, float] | None = None
) -> ProteinFoldChangePhenotype:
    """The PP_4188 strain's released fold changes and t-test p-values, by locus tag.

    ``p_values`` carries the released ``P-Value (Equal Variance)`` column, which the
    Fig. 5B caption states is a "student's t-test $p$ -value" against the control
    strain. It is UNADJUSTED: no multiple-testing correction appears anywhere in the
    paper or the SI, so it goes in ``protein_fold_change_p_value`` and
    ``p_value_adjustment_method`` stays :data:`DIFFERENTIAL_P_VALUE_ADJUSTMENT`
    (``None``), which the class requires of a record with no adjusted map.

    ``protein_fold_change_se`` is a typed gap rather than a derived number: the
    replicate COUNT is sourced (three biological replicates, Supplementary Fig. S8's
    caption), but neither Table S4 nor Table S5 releases a per-replicate value or a
    spread, so there is nothing to divide by sqrt(n). A p-value is a test result, not a
    dispersion, and inverting it into an SE would need the per-group means this release
    does not carry.
    """
    if not values:
        raise RuntimeError("a differential record needs at least one protein")
    for tag, value in values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"{tag}: fold change {value} is not a positive ratio")
    if p_values is not None and set(p_values) != set(values):
        raise RuntimeError("p-value and fold-change keys disagree")
    return ProteinFoldChangePhenotype(
        protein_fold_change=dict(values),
        fold_change_scale=FOLD_CHANGE_SCALE,
        reference_basis=DIFFERENTIAL_REFERENCE_BASIS,
        protein_fold_change_se=None,
        protein_fold_change_p_value=None if p_values is None else dict(p_values),
        protein_fold_change_p_value_adjusted=None,
        p_value_adjustment_method=DIFFERENTIAL_P_VALUE_ADJUSTMENT,
        n_replicates=dict.fromkeys(values, DIFFERENTIAL_N_REPLICATES),
        measurement_type=DIFFERENTIAL_MEASUREMENT_TYPE,
        provenance_gaps=[
            ProvenanceGap(
                field="protein_fold_change_se",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="Supplementary Tables S4 and S5 release a fold change, its log2, a "
                "p-value and a rank, and no per-replicate value or spread. The "
                f"replicate count is sourced ({DIFFERENTIAL_N_REPLICATES} biological "
                "replicates, Supplementary Fig. S8's caption) but the SD exists only as "
                "that figure's error bars, so there is nothing to derive an SE from",
            ),
            ProvenanceGap(
                field="protein_fold_change_p_value_adjusted",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="only the unadjusted t-test p-value is released. The one FDR in "
                "this paper is DIA-NN's IDENTIFICATION filter ('a global FDR = 0.01 at "
                "both the precursor and protein group levels'), applied before any "
                "contrast was computed, so it is not a correction of these p-values and "
                "no adjusted column exists to store",
            ),
        ],
    )


def differential_reference_phenotype(
    values: Mapping[str, float],
) -> ProteinFoldChangePhenotype:
    """The control strain on the released fold-change scale: 1.0 for every key.

    Not a measurement, the same way :func:`reference_phenotype` is not: 1.0 is the fold
    change's denominator by definition, which is what
    ``ProteinFoldChangePhenotype.neutral_reference`` returns, and it carries no p-value
    because a strain tested against itself has no contrast.
    """
    return differential_phenotype(
        dict.fromkeys(sorted(values), REFERENCE_RELATIVE_EXPRESSION)
    )


def reference_phenotype(
    tags: Iterable[str], *, n_replicates: int, with_se: bool
) -> ProteinFoldChangePhenotype:
    """The control strain on the released scale: 1.0 for every measured protein.

    Not a measurement -- it is the fold change's denominator, and the only value the
    released numbers are expressed against, so experiment / reference reproduces the
    source's number exactly. ``REFERENCE_RELATIVE_EXPRESSION`` is
    ``FOLD_CHANGE_SCALE.neutral_value``, so a reader cannot mistake it for a quantified
    abundance and the two can never disagree.
    """
    keys = sorted(set(tags))
    if not keys:
        raise RuntimeError("a reference needs the record's measured proteins")
    values = {tag: REFERENCE_RELATIVE_EXPRESSION for tag in keys}
    if with_se:
        return ProteinFoldChangePhenotype(
            protein_fold_change=values,
            fold_change_scale=FOLD_CHANGE_SCALE,
            reference_basis=REFERENCE_BASIS,
            protein_fold_change_se={tag: 0.0 for tag in keys},
            n_replicates={tag: n_replicates for tag in keys},
            measurement_type=MEASUREMENT_TYPE,
        )
    return relative_expression_phenotype(values, n_replicates=n_replicates)


def titer_environment() -> CultureEnvironment:
    """The production culture as a ``CultureEnvironment``, for the titer family.

    ``ProductTiterExperiment.environment`` is annotated ``CultureEnvironment`` and
    pydantic serializes by the DECLARED type, so this slot is what finally carries the
    5 mL working volume, the 180 rpm and the OD600 0.2 inoculum the Methods state and
    that :func:`production_environment` has to leave in ``SOURCED_VALUES``. Everything
    else is that function's object unchanged, so the two families cannot drift apart on
    the medium, the temperature, the doses or the duration.

    ``vessel`` is a typed gap, not a guess: the Methods say "5 mL M9 medium" and name no
    container, so the volume is sourced and the vessel is not.
    """
    base = production_environment()
    return CultureEnvironment(
        media=base.media,
        temperature=base.temperature,
        perturbations=list(base.perturbations),
        aerobicity=base.aerobicity,
        duration_hours=base.duration_hours,
        culture_format=CultureFormat(
            vessel=None,
            working_volume_ul=5000.0,
            shaking_rpm=180.0,
            inoculum_od600=0.2,
            endpoint=EndpointRule.fixed_duration,
            provenance=[SOURCED_VALUES["production_culture"], TEMPERATURE_C],
            provenance_gaps=[
                ProvenanceGap(
                    field="vessel",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                    note="the Methods state the working volume ('5 mL M9 medium') and "
                    "never the container it is in, so the volume is sourced and the "
                    "vessel is not invented",
                )
            ],
        ),
    )


def isoprenol_titer_phenotype(value: float) -> ProductTiterPhenotype:
    """One deposited isoprenol number as a titer, stored verbatim.

    The deposited column carries no unit of its own; the unit is the paper's, quoted in
    :data:`TITER_UNIT_SOURCE`. ``ConcentrationUnit`` has no ``mg/L`` member and 1 mg/L
    is exactly 1 ug/mL, so the released number is stored unchanged under ``ug_per_ml``
    and no arithmetic touches a source value.

    ``titer_uncertainty`` is a typed gap on both halves: the deposit releases one number
    per strain and the SD exists only as Fig. 4's error bars. ``n_samples`` is the
    conservative LOWER end of the released 3-to-6 range (:data:`TITER_N_REPLICATES`),
    because there is no companion statistic to back-solve the per-strain count from.
    """
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError(f"isoprenol titer {value} is not a measured amount")
    return ProductTiterPhenotype(
        product=resolved_compound("isoprenol"),
        titer=value,
        titer_unit=ConcentrationUnit.ug_per_ml,
        titer_uncertainty=None,
        titer_uncertainty_type=None,
        n_samples=TITER_N_REPLICATES,
        sample_unit=SampleUnit.biological_replicate,
        quantification_method=str(SOURCED_VALUES["titer_quantification"].value),
        provenance_gaps=[
            ProvenanceGap(
                field=field,
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="the deposited table releases one isoprenol number per strain and "
                "no spread; the standard deviation of a titer appears only as Fig. 4's "
                "error bars, which are not machine-readable, so there is nothing to "
                "store and nothing to derive an SE from",
            )
            for field in ("titer_uncertainty", "titer_uncertainty_type", "titer_se")
        ]
        + [
            ProvenanceGap(
                field=field,
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="this campaign reports titer only; neither a yield on substrate "
                "nor a volumetric productivity is released for any strain",
            )
            for field in (
                "product_yield",
                "product_yield_unit",
                "productivity",
                "productivity_unit",
            )
        ],
    )


def panel_proteome_phenotype(values: Mapping[str, float]) -> ProteinAbundancePhenotype:
    """One deposited strain's protein profile, keyed by locus tag.

    An ABSOLUTE per-strain Top3 signal, which is what ``ProteinAbundancePhenotype``
    asks for, unlike the two ratio families in this module. A released ``0`` is a
    present measurement and is kept verbatim: nothing is imputed and no key is dropped
    for being zero.

    ``n_replicates`` is 1 per key, which is the arithmetic of the deposit: one displayed
    number per (strain, protein), no replicate column and no uncertainty, so one sample
    is the support of one stored abundance. ``protein_abundance_se`` is a typed gap for
    the same reason.
    """
    if not values:
        raise RuntimeError("a panel-proteome record needs at least one protein")
    for tag, value in values.items():
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError(f"{tag}: abundance {value} is not a measured signal")
    return ProteinAbundancePhenotype(
        protein_abundance=dict(values),
        protein_abundance_se=None,
        n_replicates=dict.fromkeys(values, PANEL_N_REPLICATES),
        measurement_type=PANEL_PROTEOME_MEASUREMENT_TYPE,
        provenance_gaps=[
            ProvenanceGap(
                field="protein_abundance_se",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="the deposited page renders one number per (strain, protein) with "
                "no replicate column and no uncertainty, and the replicate design "
                "behind one proteomics sample of this screen is not stated; there is "
                f"nothing to derive an SE from. {BENCHLING_PASTE_CAVEAT}",
            )
        ],
    )


def check_isoprenol_identity() -> None:
    """Stop if the compound table gains an isoprenol row with another InChIKey.

    The campaign's titers are not released, so no record stores the compound; the pinned
    key is checked against the table's row (landed in PR #729) so a later curation cannot
    silently disagree with this module.
    """
    compound = resolved_compound("isoprenol")
    if compound.inchikey is not None and compound.inchikey != ISOPRENOL_INCHIKEY:
        raise RuntimeError(
            f"the compound table now gives isoprenol {compound.inchikey}, not the "
            f"pinned {ISOPRENOL_INCHIKEY}"
        )


def standard_names(genome: PPutidaKT2440Genome, tags: Iterable[str]) -> dict[str, str]:
    """Locus tag -> the annotation's own gene symbol for it, falling back to the tag."""
    exact, _ = genome.feature_index["symbol"]
    by_tag: dict[str, str] = {}
    for symbol, loci in exact.items():
        for locus in loci:
            by_tag.setdefault(str(locus), str(symbol))
    return {tag: by_tag.get(tag, tag) for tag in tags}


# --------------------------------------------------------------------------- #
# Retention ledger
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One retention rule, what it removed, and the items it removed.

    ``scope`` says WHAT the rule removed. A ``"record"`` rule removes whole records and
    is counted by :meth:`DropLog.check`; a ``"protein_accession"`` rule removes a
    measured KEY from every record's profile, so it carries ``n_records = 0`` and its
    arithmetic is the resolution split rather than the record totals.
    """

    rule: str
    scope: Literal["record", "protein_accession"] = "record"
    description: str
    n_records: int
    items: list[str] = []


class DropLog(BaseModel):
    """Every retention rule applied to one build, with the arithmetic it must satisfy."""

    dataset: str
    source_rows: int
    candidate_records: int
    kept_records: int
    dropped_records: int
    rules: list[DropRule]
    reconciliation: LocusTagReconciliation | None = None
    notes: list[str] = []

    def check(self) -> None:
        """The rules must account for every candidate record that was not written."""
        if self.kept_records + self.dropped_records != self.candidate_records:
            raise RuntimeError(
                f"{self.dataset}: {self.kept_records} kept + {self.dropped_records} "
                f"dropped != {self.candidate_records} candidates"
            )
        accounted = sum(rule.n_records for rule in self.rules)
        if accounted != self.dropped_records:
            raise RuntimeError(
                f"{self.dataset}: rules total {accounted}, {self.dropped_records} "
                "records are missing from the build"
            )


# --------------------------------------------------------------------------- #
# Cross-source proofs over the deposited panel (build time)
# --------------------------------------------------------------------------- #
def assert_benchling_titers_match_the_results_text(
    table: BenchlingStrainTable,
) -> list[str]:
    """Join the deposited titer column onto the two titers the Results state.

    The paper states exactly two isoprenol numbers, and the deposit is an independent
    retrieval of the same campaign, so they are the one cross-source join available.

    Measured 2026-10-09 on the pinned bytes: ``PP_0168`` is 957.246595 against the
    printed 958 mg/L, a difference of 0.7534 mg/L, which is inside the 1 mg/L the paper
    prints and is ASSERTED. ``PP_4188`` is 1494.98874 against the printed 1469 mg/L, a
    difference of 25.98874 mg/L (1.77 %), which is NOT inside it; that difference is
    recorded exactly and is not reconciled by preferring either source, because no
    mirrored byte says which number the figure was drawn from.
    """
    titers = table.titer_by_label
    proofs: list[str] = []
    for label, sourced in (
        (TITER_ORACLE_AGREES, SOURCED_VALUES["best_intuition_titer_mg_per_l"]),
        (TITER_ORACLE_DISAGREES, SOURCED_VALUES["best_titer_mg_per_l"]),
    ):
        if label not in titers:
            raise RuntimeError(
                f"the deposited panel has no row for {label}, whose titer the Results "
                f"state as a number ('{sourced.quote}'); the cross-source join that "
                "checks the deposit against the paper cannot run"
            )
        stated = float(sourced.value)
        deposited = titers[label]
        difference = abs(deposited - stated)
        if label == TITER_ORACLE_AGREES:
            if difference > TITER_ORACLE_TOL_MG_PER_L:
                raise RuntimeError(
                    f"the deposited titer for {label} is {deposited} mg/L and the "
                    f"Results print {stated} mg/L, a difference of {difference} mg/L, "
                    f"above the {TITER_ORACLE_TOL_MG_PER_L} mg/L the paper prints to; "
                    "the deposit is no longer the same campaign's numbers"
                )
            proofs.append(
                f"the deposited titer for {label} is {deposited} mg/L and the Results "
                f"print {stated} mg/L ('{sourced.quote}'), a difference of "
                f"{difference} mg/L, inside the {TITER_ORACLE_TOL_MG_PER_L} mg/L the "
                "paper prints to"
            )
            continue
        proofs.append(
            f"MEASURED DISAGREEMENT, kept: the deposited titer for {label} is "
            f"{deposited} mg/L and the Results print {stated} mg/L "
            f"('{sourced.quote}'), a difference of {difference} mg/L "
            f"({difference / stated:.4%}). The deposited column is what every record "
            "stores, because it is the per-strain release; no mirrored byte states "
            "which of the two numbers Fig. 4E was drawn from, so the difference is "
            "recorded rather than repaired"
        )
    return proofs


def assert_benchling_deposit_shape(
    table: BenchlingStrainTable, correlation: Sequence[BenchlingCorrelationRow]
) -> list[str]:
    """Assert the two deposited files still have the shape the pins describe.

    Also records what the join between them IS: the correlation table covers a WIDER
    protein set than the panel, so it is not the panel's own Pearson output and neither
    file can be used to fill a gap in the other.
    """
    if len(table.rows) != BENCHLING_TITER_ROWS:
        raise TableExtractionError(
            f"the deposited panel has {len(table.rows)} rows, pinned "
            f"{BENCHLING_TITER_ROWS}"
        )
    if len(table.accessions) != BENCHLING_ACCESSIONS:
        raise TableExtractionError(
            f"the deposited panel has {len(table.accessions)} protein columns, pinned "
            f"{BENCHLING_ACCESSIONS}"
        )
    if len(correlation) != BENCHLING_CORRELATION_ROWS:
        raise TableExtractionError(
            f"the deposited correlation table has {len(correlation)} rows, pinned "
            f"{BENCHLING_CORRELATION_ROWS}"
        )
    shared = {row.protein for row in correlation} & set(table.accessions)
    return [
        f"the deposited panel is {len(table.rows)} rows x "
        f"{len(table.accessions) + 2} columns with {table.zero_cells} of "
        f"{len(table.rows) * len(table.accessions)} abundance cells a released 0 and "
        "no blank cell; a released 0 is a present measurement and is kept verbatim",
        f"the deposited correlation table is {len(correlation)} protein rows and shares "
        f"{len(shared)} of the panel's {len(table.accessions)} accessions, so it is a "
        "statistic over a WIDER protein set than the panel and neither file completes "
        "the other. It is recorded and no record stores it",
    ]


# --------------------------------------------------------------------------- #
# Shared dataset plumbing
# --------------------------------------------------------------------------- #
class _Yunus2026Dataset(ExperimentDataset):
    """Shared skeleton: the pinned docx, the KT2440 genome, and the record schema."""

    REFERENCE_STRAIN: ClassVar[BacterialReferenceStrain] = KT2440_STRAIN
    #: Every screened target must resolve to a locus of the pinned assembly. Measured on
    #: the pinned docx: 123 of 123 Table S3 tags and 11 of 11 array targets are current
    #: standard locus tags, so anything below 1.0 means the annotation or the released
    #: names moved and the build stops.
    MIN_RESOLVED_FRACTION: ClassVar[float] = 1.0

    def __init__(
        self,
        root: str,
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the KT2440 genome resolves guide labels and screened targets."""
        self.pputida_genome = pputida_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialProteinFoldChangeExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialProteinFoldChangeExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The one consumed file: the publisher's only supplementary component."""
        return [SI_DOCX]

    def download(self) -> None:
        """Link the mirrored docx into ``raw/`` after checking the manifest and sha256."""
        data_root = _data_root()
        manifest = load_manifest(data_root)
        check_manifest_pin(
            SI_MIRROR_RELPATH,
            manifest_sha256(manifest, SI_MIRROR_RELPATH),
            SI_DOCX_SHA256,
        )
        src = raw_mirror_dir(data_root) / SI_MIRROR_RELPATH
        if not src.exists():
            raise RuntimeError(f"required raw artifact missing from mirror: {src}")
        os.makedirs(self.raw_dir, exist_ok=True)
        link_verified(src, osp.join(self.raw_dir, SI_DOCX), SI_DOCX_SHA256)
        log.info("Yunus 2026 SI linked into %s (sha256 verified)", self.raw_dir)

    def _genome(self) -> PPutidaKT2440Genome:
        """The injected KT2440 genome, or one opened from the genomes tier.

        A genome of another assembly set is refused: this paper's identifiers are
        GenBank ``PP_`` locus tags of ``pputida_KT2440_ASM756v2`` and mean nothing
        against any other annotation.
        """
        genome = self.pputida_genome
        if genome is None:
            genome = bacterial_genome("pputida", "KT2440")
            self.pputida_genome = genome
        if genome.ASSEMBLY_SET != KT2440_ASSEMBLY_SET:
            raise ValueError(
                f"{type(self).__name__} needs the {KT2440_ASSEMBLY_SET} genome, got "
                f"{genome.ASSEMBLY_SET}"
            )
        return genome

    def _tables(self) -> dict[str, list[list[str]]]:
        """The pinned supplementary tables, after verifying the raw file's sha256."""
        verify_raw_files(self.raw_dir, DATA_SHA256)
        return supplementary_tables(osp.join(self.raw_dir, SI_DOCX))

    def _resolve_targets(
        self, genome: PPutidaKT2440Genome, tags: Sequence[str]
    ) -> tuple[dict[str, str], LocusTagReconciliation]:
        """Reconcile every screened locus tag against the pinned annotation."""
        stored, report = reconcile_locus_tags(
            genome, pd.Series(list(tags), dtype=object), label=self.name
        )
        report.require_resolved(self.MIN_RESOLVED_FRACTION)
        if report.outside_namespace:
            raise RuntimeError(
                f"{self.name}: screened targets outside {KT2440_NAMESPACE}: "
                f"{report.outside_namespace}"
            )
        return dict(zip(tags, stored.tolist(), strict=True)), report

    def _write_target_lists(self, tables: Mapping[str, list[list[str]]]) -> None:
        """Tables S1 and S2 (the target lists) as a preprocess CSV; not records."""
        rows = parse_target_list(tables["S1"], table="S1") + parse_target_list(
            tables["S2"], table="S2"
        )
        pd.DataFrame([row.model_dump() for row in rows]).to_csv(
            osp.join(self.preprocess_dir, "target_lists.csv"), index=False
        )

    def _write_drop_log(self, drop_log: DropLog) -> None:
        """Validate and write ``preprocess/dropped_records.json``."""
        drop_log.check()
        Path(self.preprocess_dir, "dropped_records.json").write_text(
            drop_log.model_dump_json(indent=2)
        )

    def _write_guide_library(self, library: GuideLibrary) -> None:
        """The parsed Table S7 library and its misses, as a preprocess JSON."""
        Path(self.preprocess_dir, "guide_library.json").write_text(
            library.model_dump_json(indent=2)
        )

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


class _Yunus2026BenchlingDataset(_Yunus2026Dataset):
    """Shared skeleton for the two families built from the hand-deposited tables.

    They consume the publisher's docx (for the target lists, the guide library and the
    cross-source assertions) AND the two deposited TSVs, so ``raw/`` carries all three
    and every one is checked against the mirror manifest and its pinned sha256.
    """

    def _benchling_pins(self) -> dict[str, str]:
        """``{raw file name: pinned sha256}`` of every file this family consumes."""
        pins = {SI_DOCX: SI_DOCX_SHA256}
        for relpath, _, expected, _ in benchling_deposits():
            pins[osp.basename(relpath)] = expected
        return pins

    @property
    def raw_file_names(self) -> list[str]:
        """The docx plus the two hand-deposited Benchling tables."""
        return sorted(self._benchling_pins())

    def download(self) -> None:
        """Link the docx and both deposited tables after checking manifest and sha256."""
        data_root = _data_root()
        manifest = load_manifest(data_root)
        mirror = raw_mirror_dir(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        deposits = [(SI_MIRROR_RELPATH, SI_DOCX_SHA256)] + [
            (relpath, expected) for relpath, _, expected, _ in benchling_deposits()
        ]
        for relpath, expected in deposits:
            check_manifest_pin(relpath, manifest_sha256(manifest, relpath), expected)
            src = mirror / relpath
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            link_verified(src, osp.join(self.raw_dir, osp.basename(relpath)), expected)
        log.info(
            "Yunus 2026 SI and both Benchling deposits linked into %s (sha256 verified)",
            self.raw_dir,
        )

    def _benchling_inputs(
        self,
    ) -> tuple[
        dict[str, list[list[str]]],
        BenchlingStrainTable,
        tuple[BenchlingCorrelationRow, ...],
        list[str],
    ]:
        """The parsed docx tables, both deposited tables, and the build-time proofs."""
        verify_raw_files(self.raw_dir, self._benchling_pins())
        tables = supplementary_tables(osp.join(self.raw_dir, SI_DOCX))
        panel = read_benchling_strain_table(
            osp.join(self.raw_dir, BENCHLING_TITER_FILENAME)
        )
        correlation = read_benchling_correlation_table(
            osp.join(self.raw_dir, BENCHLING_CORRELATION_FILENAME)
        )
        proofs = assert_benchling_deposit_shape(panel, correlation)
        proofs.extend(assert_benchling_titers_match_the_results_text(panel))
        return tables, panel, correlation, proofs

    def _reconcile_rows(
        self, tables: Mapping[str, list[list[str]]], panel: BenchlingStrainTable
    ) -> LabelReconciliation:
        """Reconcile every deposited row label against the released target lists.

        The control row is the only label no route may reach: it is the reference, not a
        strain. Anything else unmatched is a released label this loader cannot key and
        the build stops rather than storing a titer against a guessed gene.
        """
        known = {row.target for row in parse_table_s3(tables["S3"])} | {
            row.locus_tag for row in parse_table_s3(tables["S3"])
        }
        known |= {
            row.locus_tag
            for table in ("S1", "S2")
            for row in parse_target_list(tables[table], table=table)
        }
        reconciliation = reconcile_benchling_labels(
            [row.label for row in panel.rows], known
        )
        if reconciliation.unmatched != (BENCHLING_CONTROL_LABEL,):
            raise RuntimeError(
                f"{self.name}: the deposited labels {list(reconciliation.unmatched)} "
                "reach no target of Tables S1, S2 or S3 through the released label, "
                "the ' (S)' marker strip or the _NT<digit> variant strip; only "
                f"{BENCHLING_CONTROL_LABEL!r} may be unmatched"
            )
        return reconciliation

    @staticmethod
    def _marker_drop_rule(panel: BenchlingStrainTable) -> DropRule:
        """The rule that drops the rows carrying the undefined ``" (S)"`` marker."""
        marked = [row.label for row in panel.strains if row.marked]
        return DropRule(
            rule="row_label_carries_an_undefined_marker",
            scope="record",
            description=(
                f"the row label ends in {BENCHLING_SOLID_MARKER!r}, a marker NO "
                f"mirrored byte defines: {BENCHLING_MARKER_SEARCH}. Each of these "
                "labels also appears WITHOUT the marker as its own row, so the marker "
                "separates two rows for one target and keeping the marked one would "
                "store a second titer and a second proteome against a strain identity "
                "this release never states. The unmarked twin is kept, so no target is "
                "lost"
            ),
            n_records=len(marked),
            items=marked,
        )

    def _reconciliation_rows(
        self, reconciliation: LabelReconciliation
    ) -> list[dict[str, Any]]:
        """The reconciliation as one row per deposited label, for ``preprocess/``."""
        return [
            {"label": label, "route": route}
            for route, labels in (
                ("exact", reconciliation.exact),
                ("after_marker_strip", reconciliation.after_marker_strip),
                ("after_variant_strip", reconciliation.after_variant_strip),
                ("unmatched_reference_row", reconciliation.unmatched),
            )
            for label in labels
        ]


# --------------------------------------------------------------------------- #
# Family 1: the 125-sample knockdown screen (Supplementary Table S3)
# --------------------------------------------------------------------------- #
@register_dataset
class CrispriKnockdownYunus2026Dataset(_Yunus2026Dataset):
    """Yunus 2026 per-strain relative expression of its own CRISPRi target (Table S3)."""

    def __init__(
        self,
        root: str = KNOCKDOWN_ROOT_REL,
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with this family's default dev-tree root."""
        super().__init__(
            root, io_workers, pputida_genome, transform, pre_transform, **kwargs
        )

    @post_process
    def process(self) -> None:
        """One record per Table S3 strain with a released ratio; write LMDB."""
        check_isoprenol_identity()
        tables = self._tables()
        rows = parse_table_s3(tables["S3"])
        genome = self._genome()
        library = build_guide_library(tables["S7"], genome)

        tags = sorted({row.locus_tag for row in rows})
        stored_by_tag, report = self._resolve_targets(genome, tags)
        common = standard_names(genome, stored_by_tag.values())

        kept = [row for row in rows if not row.not_detected]
        dropped = [row for row in rows if row.not_detected]
        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        reference_genome = chassis_reference()
        environment = production_environment()
        pub = publication()

        assignments: list[GuideAssignment] = []
        table_rows: list[dict[str, Any]] = []
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for row in tqdm(kept, desc="yunus2026-knockdown"):
                tag = stored_by_tag[row.locus_tag]
                assignment = library.assign(tag, row.variant)
                assignments.append(assignment)
                value = row.relative_expression
                assert value is not None  # noqa: S101 - kept rows are numeric by filter
                experiment = BacterialProteinFoldChangeExperiment(
                    dataset_name=self.name,
                    genotype=Genotype(
                        perturbations=[
                            crispri_perturbation(tag, common[tag], assignment)
                        ]
                    ),
                    environment=environment,
                    phenotype=relative_expression_phenotype(
                        {tag: value}, n_replicates=1
                    ),
                )
                reference = BacterialProteinFoldChangeExperimentReference(
                    dataset_name=self.name,
                    genome_reference=reference_genome,
                    environment_reference=environment.model_copy(),
                    phenotype_reference=reference_phenotype(
                        [tag], n_replicates=1, with_se=False
                    ),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, pub, itxn),
                )
                table_rows.append(
                    {
                        "record": idx,
                        "strain": row.strain,
                        "target": row.target,
                        "locus_tag": tag,
                        "variant": row.variant,
                        "gene_name": common[tag],
                        "relative_expression": value,
                        "guide_spacer": assignment.spacer,
                        "guide_oligo_label": assignment.oligo_label,
                        "guide_reason": assignment.reason,
                    }
                )
                idx += 1
        env.close()
        interned_env.close()

        pd.DataFrame(table_rows).to_csv(
            osp.join(self.preprocess_dir, "table_s3.csv"), index=False
        )
        pd.DataFrame([a.model_dump() for a in assignments]).to_csv(
            osp.join(self.preprocess_dir, "guide_assignment.csv"), index=False
        )
        self._write_guide_library(library)
        self._write_target_lists(tables)
        self._write_drop_log(
            DropLog(
                dataset=self.name,
                source_rows=len(rows),
                candidate_records=len(rows),
                kept_records=idx,
                dropped_records=len(dropped),
                rules=[
                    DropRule(
                        rule="control_strain_expression_not_detected",
                        description="Table S3 reports 'n.d.' for the strain: the "
                        "target protein was not detected in the control strain, so the "
                        "released ratio has no denominator and there is no number to "
                        f"store ('{SOURCED_VALUES['not_detected_count'].quote}')",
                        n_records=len(dropped),
                        items=[f"{row.strain} ({row.target})" for row in dropped],
                    )
                ],
                reconciliation=report,
                notes=[
                    "one record per Table S3 strain; the control strain is the "
                    "phenotype_reference at the ratio's denominator (1.0), not a record",
                    f"n_replicates = 1 for every record: '{SOURCED_VALUES['screen_samples'].quote}' "
                    "and the table holds exactly that many rows, one per strain",
                    f"{sum(1 for a in assignments if a.spacer is None)} of {idx} "
                    "records carry no guide spacer; the reason per target is in "
                    "preprocess/guide_assignment.csv",
                    "the isoprenol titers this campaign is about are released only as "
                    "bar charts and are not records (see the module docstring)",
                ],
            )
        )
        log.info(
            "Yunus2026 knockdown: %d records from %d Table S3 rows (%d 'n.d.' dropped); "
            "%d targets, %d with a sourced spacer; name statuses %s",
            idx,
            len(rows),
            len(dropped),
            len(tags),
            sum(1 for a in assignments if a.spacer is not None),
            {status.value: n for status, n in report.status_histogram.items()},
        )


# --------------------------------------------------------------------------- #
# Family 2: the sgRNA-array position study (Supplementary Tables S8-S12)
# --------------------------------------------------------------------------- #
@register_dataset
class CrispriArrayYunus2026Dataset(_Yunus2026Dataset):
    """Yunus 2026 multiplexed-array relative expression panels (Tables S8-S12)."""

    def __init__(
        self,
        root: str = ARRAY_ROOT_REL,
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with this family's default dev-tree root."""
        super().__init__(
            root, io_workers, pputida_genome, transform, pre_transform, **kwargs
        )

    @post_process
    def process(self) -> None:
        """One record per array construct, over the proteins the panels measured in it."""
        check_isoprenol_identity()
        tables = self._tables()
        cells: list[ArrayCell] = []
        for table, protein in ARRAY_PANELS:
            cells.extend(parse_array_panel(tables[table], protein))
        genome = self._genome()
        library = build_guide_library(tables["S7"], genome)

        tags = sorted(
            {tag for cell in cells for tag in cell.guide_targets}
            | {cell.protein for cell in cells}
        )
        stored_by_tag, report = self._resolve_targets(genome, tags)
        common = standard_names(genome, stored_by_tag.values())

        by_construct: dict[str, dict[str, list[float]]] = {}
        guides: dict[str, tuple[str, ...]] = {}
        for cell in cells:
            by_construct.setdefault(cell.construct_name, {}).setdefault(
                cell.protein, []
            ).append(cell.relative_expression)
            guides[cell.construct_name] = cell.guide_targets
        observed = {
            len(values)
            for proteins in by_construct.values()
            for values in proteins.values()
        }
        if observed != {ARRAY_N_REPLICATES}:
            raise TableExtractionError(
                f"the array panels hold replicate counts {sorted(observed)}; the Fig. 3 "
                f"caption states {ARRAY_N_REPLICATES}"
            )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        reference_genome = chassis_reference()
        environment = production_environment()
        pub = publication()

        table_rows: list[dict[str, Any]] = []
        assignments: list[GuideAssignment] = []
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for construct, proteins in tqdm(
                sorted(by_construct.items()), desc="yunus2026-array"
            ):
                means: dict[str, float] = {}
                standard_errors: dict[str, float] = {}
                counts: dict[str, int] = {}
                for protein, values in proteins.items():
                    tag = stored_by_tag[protein]
                    means[tag] = statistics.fmean(values)
                    standard_errors[tag] = statistics.stdev(values) / math.sqrt(
                        len(values)
                    )
                    counts[tag] = len(values)
                perturbations: list[Any] = []
                for target in guides[construct]:
                    tag = stored_by_tag[target]
                    assignment = library.assign(tag, None)
                    assignments.append(assignment)
                    perturbations.append(
                        crispri_perturbation(tag, common[tag], assignment)
                    )
                experiment = BacterialProteinFoldChangeExperiment(
                    dataset_name=self.name,
                    genotype=Genotype(perturbations=perturbations),
                    environment=environment,
                    phenotype=array_phenotype(means, standard_errors, counts),
                )
                reference = BacterialProteinFoldChangeExperimentReference(
                    dataset_name=self.name,
                    genome_reference=reference_genome,
                    environment_reference=environment.model_copy(),
                    phenotype_reference=reference_phenotype(
                        means, n_replicates=ARRAY_N_REPLICATES, with_se=True
                    ),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, pub, itxn),
                )
                table_rows.append(
                    {
                        "record": idx,
                        "construct": construct,
                        "n_guides": len(perturbations),
                        "guide_targets": ";".join(
                            stored_by_tag[t] for t in guides[construct]
                        ),
                        "measured_proteins": ";".join(sorted(means)),
                        "n_measured_proteins": len(means),
                        "n_replicates": ARRAY_N_REPLICATES,
                    }
                )
                idx += 1
        env.close()
        interned_env.close()

        pd.DataFrame(table_rows).to_csv(
            osp.join(self.preprocess_dir, "array_constructs.csv"), index=False
        )
        pd.DataFrame(
            [
                {
                    "construct": cell.construct_name,
                    "protein": stored_by_tag[cell.protein],
                    "replicate": cell.replicate,
                    "relative_expression": cell.relative_expression,
                }
                for cell in cells
            ]
        ).to_csv(osp.join(self.preprocess_dir, "array_replicates.csv"), index=False)
        pd.DataFrame([a.model_dump() for a in assignments]).to_csv(
            osp.join(self.preprocess_dir, "guide_assignment.csv"), index=False
        )
        self._write_guide_library(library)
        self._write_target_lists(tables)
        self._write_drop_log(
            DropLog(
                dataset=self.name,
                source_rows=len(cells),
                candidate_records=len(by_construct),
                kept_records=idx,
                dropped_records=0,
                rules=[],
                reconciliation=report,
                notes=[
                    "nothing is dropped: every released (construct, protein, replicate) "
                    "cell of Tables S8-S12 is in a record",
                    "one record per CONSTRUCT, with every protein the five panels "
                    "measured in it; a panel measures its representative protein even "
                    "in a construct that carries no guide for it (PP_4192_0812_4160_0168 "
                    "is measured for PP_4188), which is the panel's own control and is "
                    "stored as measured",
                    f"n_replicates = {ARRAY_N_REPLICATES} and the SE is the sample SD "
                    f"over sqrt(n): '{SOURCED_VALUES['array_replicates'].quote}'",
                    "Table S3 reports a DIFFERENT value for a genotype this family also "
                    "holds (PP_4188: 0.2213 there, mean 0.2509 here), which is why the "
                    "two families are separate datasets and are never pooled",
                ],
            )
        )
        log.info(
            "Yunus2026 array: %d constructs from %d released cells; %d (construct, "
            "protein) measurements; %d guide slots, %d with a sourced spacer",
            idx,
            len(cells),
            sum(len(p) for p in by_construct.values()),
            len(assignments),
            sum(1 for a in assignments if a.spacer is not None),
        )


# --------------------------------------------------------------------------- #
# Family 3: the PP_4188 strain's global differential (Supplementary Tables S4, S5)
# --------------------------------------------------------------------------- #
@register_dataset
class CrispriDifferentialProteomeYunus2026Dataset(_Yunus2026Dataset):
    """Yunus 2026 global fold change against the control for the PP_4188 strain.

    ONE record carrying every released protein key that resolves to a locus of the
    pinned assembly. Tables S4 and S5 release 145 downregulated and 193 upregulated
    proteins "from PP_4188 strain", as a fold change against the control strain on the
    same DIA-NN Top3 quantification the other two families use, so this is the same
    QUANTITY TYPE :class:`CrispriKnockdownYunus2026Dataset` already stores.

    It is its own dataset class, not a 103rd record of that family, for two reasons the
    verifier makes concrete. The scale is different: Table S3's
    ``Relative expression level`` is one proteomics sample per strain (n = 1, no
    uncertainty released) while this is a thresholded differential over three biological
    replicates, so ``measurement_type`` differs and the shared
    ``verify_protein_dataset`` asserts one ``measurement_type`` per dataset. And the
    record is one profile of 305 proteins rather than one strain's own target, so pooling
    them would put two meanings of "the measured protein set" in one store.
    """

    #: The MEASURED keys here are DIA-NN protein names, not screened locus tags, so the
    #: base class's 1.0 does not apply. Measured on the pinned docx: 305 of 338 (0.9024)
    #: resolve to a locus of this assembly, 196 as a current tag and 109 through a gene
    #: symbol. The threshold sits just below that. The 33 that do not are title-cased
    #: UniProt gene symbols the GenBank annotation of this assembly carries no symbol
    #: for; a real drop means the keying changed.
    MIN_RESOLVED_FRACTION: ClassVar[float] = 0.90

    def __init__(
        self,
        root: str = DIFFERENTIAL_ROOT_REL,
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with this family's default dev-tree root."""
        super().__init__(
            root, io_workers, pputida_genome, transform, pre_transform, **kwargs
        )

    @post_process
    def process(self) -> None:
        """Build the one PP_4188 differential record; write LMDB."""
        check_isoprenol_identity()
        tables = self._tables()
        rows = read_differential(tables)
        genome = self._genome()
        library = build_guide_library(tables["S7"], genome)

        screen = parse_table_s3(tables["S3"])
        self._assert_the_knocked_down_gene_is_not_a_measured_key(rows, screen)

        keys = sorted({entry.protein for entry in rows})
        stored, report = reconcile_locus_tags(
            genome, pd.Series(keys, dtype=object), label=self.name
        )
        report.require_resolved(self.MIN_RESOLVED_FRACTION)
        outside = set(report.outside_namespace)
        stored_by_key = dict(zip(keys, stored.tolist(), strict=True))
        kept = {key: stored_by_key[key] for key in keys if key not in outside}
        if not kept:
            raise RuntimeError(f"{self.name}: every released protein key was dropped")
        if len(set(kept.values())) != len(kept):
            raise RuntimeError(
                f"{self.name}: two released keys resolve to one locus tag; a fold change "
                "cannot be attributed to either"
            )

        target = stored_by_key.get(
            DIFFERENTIAL_STRAIN_TARGET, DIFFERENTIAL_STRAIN_TARGET
        )
        common = standard_names(genome, [target, *kept.values()])
        assignment = library.assign(target, None)
        values = {
            kept[entry.protein]: entry.fold_change
            for entry in rows
            if entry.protein in kept
        }
        p_values = {
            kept[entry.protein]: entry.p_value
            for entry in rows
            if entry.protein in kept
        }

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        environment = production_environment()
        pub = publication()
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            experiment = BacterialProteinFoldChangeExperiment(
                dataset_name=self.name,
                genotype=Genotype(
                    perturbations=[
                        crispri_perturbation(target, common[target], assignment)
                    ]
                ),
                environment=environment,
                phenotype=differential_phenotype(values, p_values),
            )
            reference = BacterialProteinFoldChangeExperimentReference(
                dataset_name=self.name,
                genome_reference=chassis_reference(),
                environment_reference=environment.model_copy(),
                phenotype_reference=differential_reference_phenotype(values),
            )
            txn.put(b"0", self._intern_record(experiment, reference, pub, itxn))
        env.close()
        interned_env.close()

        pd.DataFrame(
            [
                {
                    **entry.model_dump(),
                    "locus_tag": kept.get(entry.protein),
                    "stored": entry.protein in kept,
                }
                for entry in rows
            ]
        ).to_csv(osp.join(self.preprocess_dir, "differential.csv"), index=False)
        pd.DataFrame([assignment.model_dump()]).to_csv(
            osp.join(self.preprocess_dir, "guide_assignment.csv"), index=False
        )
        self._write_guide_library(library)
        self._write_target_lists(tables)
        self._write_drop_log(
            DropLog(
                dataset=self.name,
                source_rows=len(rows),
                candidate_records=1,
                kept_records=1,
                dropped_records=0,
                rules=[],
                reconciliation=report,
                notes=[
                    f"one record over {len(values)} of {len(rows)} released protein "
                    f"keys; {len(outside)} keys resolve to no locus of "
                    f"{KT2440_ASSEMBLY_SET} and are dropped from the profile, listed in "
                    "preprocess/differential.csv with stored=False. They are "
                    "title-cased UniProt gene symbols the GenBank annotation of this "
                    "assembly carries no symbol for, and a UniProt-to-locus-tag "
                    "crosswalk in the genomes tier would recover them",
                    f"the knocked-down gene {DIFFERENTIAL_STRAIN_TARGET} is the "
                    "GENOTYPE, not a measured key: neither table releases a row whose "
                    "Protein column is it, which the build asserts. It IS in the tables "
                    "under the UniProt symbol 'Kgdb' (Q88FB0, 'Dihydrolipoyllysine-"
                    "residue succinyltransferase component of 2-oxoglutarate "
                    "dehydrogenase complex'), which is the enzyme Table S2 names for "
                    f"{DIFFERENTIAL_STRAIN_TARGET} and which the pinned annotation "
                    "resolves from the symbol 'sucB'; that key is in the dropped set, so "
                    "nothing clashes today, and a crosswalk that recovers it would put "
                    "0.238537433 beside Table S3's 0.2213 on a DIFFERENT "
                    "measurement_type",
                    f"n_replicates = {DIFFERENTIAL_N_REPLICATES} for every key: "
                    f"'{SI_SOURCED_VALUES['differential_replicates'].quote}'",
                    f"the released 'P-Value (Equal Variance)' is stored for all "
                    f"{len(p_values)} keys as protein_fold_change_p_value, UNADJUSTED: "
                    f"'{SOURCED_VALUES['differential_basis_and_test'].quote}'. No "
                    "multiple-testing correction is named anywhere in the paper or the "
                    "SI, so p_value_adjustment_method is None and the adjusted map is a "
                    "typed gap; the paper's one FDR is DIA-NN's identification filter, "
                    f"'{SOURCED_VALUES['identification_fdr'].quote}', applied before any "
                    "contrast was computed",
                    *DIFFERENTIAL_NOT_STORED,
                ],
            )
        )
        log.info(
            "Yunus2026 differential: 1 record over %d of %d released keys (%d dropped "
            "outside %s); %d down + %d up",
            len(values),
            len(rows),
            len(outside),
            KT2440_NAMESPACE,
            sum(1 for entry in rows if entry.direction == "downregulated"),
            sum(1 for entry in rows if entry.direction == "upregulated"),
        )

    @staticmethod
    def _assert_the_knocked_down_gene_is_not_a_measured_key(
        rows: Sequence[DifferentialRow], screen: Sequence[ScreenRow]
    ) -> None:
        """The knocked-down gene is this record's GENOTYPE, so it must not be a key.

        Measured on the pinned docx: no row of Table S4 or S5 carries
        ``PP_4188`` in its ``Protein`` column, so the profile and the Table S3 record for
        the same strain share no protein and cannot disagree about one. Table S3 must
        still hold that strain, because this record's genotype is read from it.
        """
        clashing = [
            entry.protein
            for entry in rows
            if entry.protein == DIFFERENTIAL_STRAIN_TARGET
        ]
        if clashing:
            raise TableExtractionError(
                f"Tables S4/S5 now release a row for {DIFFERENTIAL_STRAIN_TARGET} "
                "itself; it is this record's genotype and a strain's own knocked-down "
                "gene cannot also be one of its measured fold changes without deciding "
                "which of the two released numbers for it wins"
            )
        if not any(row.locus_tag == DIFFERENTIAL_STRAIN_TARGET for row in screen):
            raise TableExtractionError(
                f"Table S3 no longer screens {DIFFERENTIAL_STRAIN_TARGET}; this "
                "record's genotype is the strain that table names"
            )


# --------------------------------------------------------------------------- #
# Family 4: the per-strain isoprenol titer (the deposited Benchling input table)
# --------------------------------------------------------------------------- #
@register_dataset
class IsoprenolTiterYunus2026Dataset(_Yunus2026BenchlingDataset):
    """Yunus 2026 per-strain isoprenol titer, from the hand-deposited Benchling table.

    One record per deposited strain row, with the ``Control`` row as the
    ``phenotype_reference``. The reference is a REAL released control strain titer
    (845.73 mg/L on the pinned bytes), which is what makes this family buildable at all:
    the paper's prose states two strain titers and never the control's, and a
    ``ProductTiterExperimentReference`` requires one.

    The titer is stored verbatim under ``ug_per_ml``, since the paper's unit is mg/L
    (:data:`TITER_UNIT_SOURCE`) and 1 mg/L is exactly 1 ug/mL. The uncertainty is a
    typed gap and ``n_samples`` is the conservative lower end of the released 3-to-6
    replicate range; both decisions are in :func:`isoprenol_titer_phenotype`.
    """

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return ProductTiterExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return ProductTiterExperimentReference

    def __init__(
        self,
        root: str = TITER_ROOT_REL,
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with this family's default dev-tree root."""
        super().__init__(
            root, io_workers, pputida_genome, transform, pre_transform, **kwargs
        )

    @post_process
    def process(self) -> None:
        """One record per deposited strain row; write LMDB."""
        check_isoprenol_identity()
        tables, panel, _, proofs = self._benchling_inputs()
        genome = self._genome()
        library = build_guide_library(tables["S7"], genome)
        reconciliation = self._reconcile_rows(tables, panel)

        candidates = [row for row in panel.strains if not row.marked]
        marker_rule = self._marker_drop_rule(panel)
        tags = sorted({benchling_target(row.label)[0] for row in candidates})
        stored_by_tag, report = self._resolve_targets(genome, tags)
        common = standard_names(genome, stored_by_tag.values())

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        reference_genome = chassis_reference()
        environment = titer_environment()
        pub = publication()
        reference_titer = isoprenol_titer_phenotype(panel.control.isoprenol_production)

        assignments: list[GuideAssignment] = []
        table_rows: list[dict[str, Any]] = []
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for row in tqdm(candidates, desc="yunus2026-titer"):
                label_tag, variant = benchling_target(row.label)
                tag = stored_by_tag[label_tag]
                assignment = library.assign(tag, variant)
                assignments.append(assignment)
                experiment = ProductTiterExperiment(
                    dataset_name=self.name,
                    genotype=Genotype(
                        perturbations=[
                            crispri_perturbation(tag, common[tag], assignment)
                        ]
                    ),
                    environment=environment,
                    phenotype=isoprenol_titer_phenotype(row.isoprenol_production),
                )
                reference = ProductTiterExperimentReference(
                    dataset_name=self.name,
                    genome_reference=reference_genome,
                    environment_reference=environment.model_copy(),
                    phenotype_reference=reference_titer.model_copy(),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, pub, itxn),
                )
                table_rows.append(
                    {
                        "record": idx,
                        "label": row.label,
                        "locus_tag": tag,
                        "variant": variant,
                        "gene_name": common[tag],
                        "isoprenol_production_mg_per_l": row.isoprenol_production,
                        "guide_spacer": assignment.spacer,
                        "guide_oligo_label": assignment.oligo_label,
                        "guide_reason": assignment.reason,
                    }
                )
                idx += 1
        env.close()
        interned_env.close()

        pd.DataFrame(table_rows).to_csv(
            osp.join(self.preprocess_dir, "benchling_titers.csv"), index=False
        )
        pd.DataFrame(self._reconciliation_rows(reconciliation)).to_csv(
            osp.join(self.preprocess_dir, "label_reconciliation.csv"), index=False
        )
        pd.DataFrame([a.model_dump() for a in assignments]).to_csv(
            osp.join(self.preprocess_dir, "guide_assignment.csv"), index=False
        )
        Path(self.preprocess_dir, "benchling_proofs.json").write_text(
            json.dumps(proofs, indent=2)
        )
        self._write_guide_library(library)
        self._write_target_lists(tables)
        self._write_drop_log(
            DropLog(
                dataset=self.name,
                source_rows=len(panel.rows),
                candidate_records=len(panel.strains),
                kept_records=idx,
                dropped_records=marker_rule.n_records,
                rules=[marker_rule],
                reconciliation=report,
                notes=[
                    f"one record per deposited strain row; the {BENCHLING_CONTROL_LABEL!r} "
                    f"row is the phenotype_reference at {panel.control.isoprenol_production} "
                    "mg/L and is not a record. It is a REAL released control titer, "
                    "which is why this family is buildable: the paper's prose states two "
                    "strain titers and never the control's",
                    f"the deposited row labels are CRISPRi TARGET labels, not strain "
                    f"names. Reconciled against Tables S1, S2 and S3: "
                    f"{len(reconciliation.exact)} match a released target exactly, "
                    f"{len(reconciliation.after_marker_strip)} after stripping "
                    f"{BENCHLING_SOLID_MARKER!r}, "
                    f"{len(reconciliation.after_variant_strip)} after also stripping a "
                    f"trailing _NT<digit>, and {len(reconciliation.unmatched)} matches "
                    "nothing: the control row. Per-label routes are in "
                    "preprocess/label_reconciliation.csv",
                    "the _NT<digit> token is the guide VARIANT number Table S3 already "
                    "uses, measured on this paper's own bytes (Table S7 gives "
                    "PP_0339_NT1 and PP_0339_NT2 distinct spacers, and Table S3 screens "
                    "PP_1607_NT2 and PP_1607_NT4 as two strains), so it is matched as "
                    "part of the guide key; either reading names the same single "
                    "perturbed locus",
                    f"n_samples = {TITER_N_REPLICATES} for every record: the Fig. 4 "
                    f"caption releases a RANGE ('{TITER_REPLICATES_SOURCE.quote}') and "
                    "no per-record count, the deposit carries no companion statistic to "
                    "back-solve one from, so the conservative lower end of the range is "
                    "taken and the uncertainty itself is a typed gap",
                    f"the titer unit is the paper's, not the page's: "
                    f"'{TITER_UNIT_SOURCE.quote}'. ConcentrationUnit has no mg/L member "
                    "and 1 mg/L is exactly 1 ug/mL, so the number is stored verbatim",
                    f"{sum(1 for a in assignments if a.spacer is None)} of {idx} "
                    "records carry no guide spacer; the reason per target is in "
                    "preprocess/guide_assignment.csv",
                    f"RetrievalMethod.manual_browser. {BENCHLING_PASTE_CAVEAT}",
                    *proofs,
                ],
            )
        )
        log.info(
            "Yunus2026 titer: %d records from %d deposited rows (%d marked rows "
            "dropped, 1 control row is the reference); reference titer %s mg/L",
            idx,
            len(panel.rows),
            marker_rule.n_records,
            panel.control.isoprenol_production,
        )


# --------------------------------------------------------------------------- #
# Family 5: the per-strain protein panel (the same deposited table's 253 columns)
# --------------------------------------------------------------------------- #
@register_dataset
class CrispriPanelProteomeYunus2026Dataset(_Yunus2026BenchlingDataset):
    """Yunus 2026 per-strain Top3 protein panel, from the hand-deposited table.

    The same strain rows the titer family stores, carrying the 253 UniProt accession
    columns beside the titer. One record per deposited strain row, with the ``Control``
    row's own profile as the ``phenotype_reference`` -- a real measured control
    proteome, unlike the two ratio families in this module, whose reference is the
    ratio's denominator.

    The accessions are UniProt ids and ``ProteinAbundancePhenotype`` keys by locus tag,
    so every column goes through the assembly set's GOA proteome crosswalk; the
    accessions that reach no locus or several are dropped from every profile by two
    named rules and listed in ``preprocess/dropped_accessions.csv``.
    """

    #: The MEASURED keys here are UniProt accessions, not screened locus tags, so the
    #: base class's 1.0 does not apply to them (it still does to the screened targets,
    #: which :meth:`_resolve_targets` checks). Measured 2026-10-09: 207 of 253
    #: accessions (0.8182) reach exactly one locus of this assembly through
    #: ``109.P_putida_KT2440.goa``; the floor sits just below that.
    MIN_ACCESSION_RESOLVED_FRACTION: ClassVar[float] = PANEL_MIN_RESOLVED_FRACTION

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset.

        The base class produces a fold change, which the three CRISPRi ratio families
        of this module store. This family's number is an ABSOLUTE per-strain Top3
        signal (``PANEL_PROTEOME_MEASUREMENT_TYPE``), so it overrides back to the
        abundance pair rather than inheriting a ratio's classes.
        """
        return BacterialProteinAbundanceExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialProteinAbundanceExperimentReference

    def __init__(
        self,
        root: str = PANEL_PROTEOME_ROOT_REL,
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with this family's default dev-tree root."""
        super().__init__(
            root, io_workers, pputida_genome, transform, pre_transform, **kwargs
        )

    def _resolve_accessions(
        self, genome: PPutidaKT2440Genome
    ) -> tuple[UniProtResolution, Any]:
        """Split the 253 deposited accessions through the GOA proteome crosswalk."""
        crosswalk = uniprot_locus_crosswalk(genome)
        panel = read_benchling_strain_table(
            osp.join(self.raw_dir, BENCHLING_TITER_FILENAME)
        )
        resolution = resolve_uniprot_accessions(
            crosswalk, panel.accessions, label=self.name
        )
        resolution.require_resolved(self.MIN_ACCESSION_RESOLVED_FRACTION)
        if resolution.collisions:
            raise RuntimeError(
                f"{self.name}: {resolution.collisions} name one locus from several "
                "accessions, so a stored abundance would be two proteins'"
            )
        return resolution, crosswalk

    @post_process
    def process(self) -> None:
        """One record per deposited strain row, keyed by locus tag; write LMDB."""
        check_isoprenol_identity()
        tables, panel, _, proofs = self._benchling_inputs()
        genome = self._genome()
        library = build_guide_library(tables["S7"], genome)
        reconciliation = self._reconcile_rows(tables, panel)
        resolution, crosswalk = self._resolve_accessions(genome)
        locus_of = dict(resolution.resolved)

        candidates = [row for row in panel.strains if not row.marked]
        marker_rule = self._marker_drop_rule(panel)
        tags = sorted({benchling_target(row.label)[0] for row in candidates})
        stored_by_tag, report = self._resolve_targets(genome, tags)
        common = standard_names(genome, stored_by_tag.values())

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        reference_genome = chassis_reference()
        environment = production_environment()
        pub = publication()
        reference_profile = panel_proteome_phenotype(
            {
                locus_of[accession]: value
                for accession, value in panel.control.abundance.items()
                if accession in locus_of
            }
        )

        assignments: list[GuideAssignment] = []
        table_rows: list[dict[str, Any]] = []
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for row in tqdm(candidates, desc="yunus2026-panel-proteome"):
                label_tag, variant = benchling_target(row.label)
                tag = stored_by_tag[label_tag]
                assignment = library.assign(tag, variant)
                assignments.append(assignment)
                values = {
                    locus_of[accession]: value
                    for accession, value in row.abundance.items()
                    if accession in locus_of
                }
                experiment = BacterialProteinAbundanceExperiment(
                    dataset_name=self.name,
                    genotype=Genotype(
                        perturbations=[
                            crispri_perturbation(tag, common[tag], assignment)
                        ]
                    ),
                    environment=environment,
                    phenotype=panel_proteome_phenotype(values),
                )
                reference = BacterialProteinAbundanceExperimentReference(
                    dataset_name=self.name,
                    genome_reference=reference_genome,
                    environment_reference=environment.model_copy(),
                    phenotype_reference=reference_profile.model_copy(),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, pub, itxn),
                )
                table_rows.append(
                    {
                        "record": idx,
                        "label": row.label,
                        "locus_tag": tag,
                        "variant": variant,
                        "gene_name": common[tag],
                        "n_proteins": len(values),
                        "n_zero": sum(1 for value in values.values() if value == 0.0),
                        "isoprenol_production_mg_per_l": row.isoprenol_production,
                    }
                )
                idx += 1
        env.close()
        interned_env.close()

        pd.DataFrame(table_rows).to_csv(
            osp.join(self.preprocess_dir, "benchling_panel.csv"), index=False
        )
        pd.DataFrame(self._reconciliation_rows(reconciliation)).to_csv(
            osp.join(self.preprocess_dir, "label_reconciliation.csv"), index=False
        )
        pd.DataFrame(
            [
                {"accession": accession, "reason": reason, "loci": loci}
                for accession, reason, loci in (
                    *(
                        (accession, "accession_names_several_loci", "|".join(loci))
                        for accession, loci in sorted(resolution.multi_locus.items())
                    ),
                    *(
                        (accession, "no_locus_tag_in_the_goa_proteome_file", "")
                        for accession in resolution.unmapped
                    ),
                )
            ]
        ).to_csv(osp.join(self.preprocess_dir, "dropped_accessions.csv"), index=False)
        Path(self.preprocess_dir, "benchling_proofs.json").write_text(
            json.dumps(proofs, indent=2)
        )
        self._write_guide_library(library)
        self._write_target_lists(tables)
        self._write_drop_log(
            DropLog(
                dataset=self.name,
                source_rows=len(panel.rows),
                candidate_records=len(panel.strains),
                kept_records=idx,
                dropped_records=marker_rule.n_records,
                rules=[
                    marker_rule,
                    DropRule(
                        rule="no_locus_tag_in_the_goa_proteome_file",
                        scope="protein_accession",
                        description=(
                            "the deposited panel is keyed by UniProt accession and the "
                            "only mirrored statement of accession -> locus tag is the "
                            f"assembly set's GOA proteome file {crosswalk.member} "
                            f"(sha256 {crosswalk.sha256}), which carries no locus tag "
                            "for these accessions. None has a gene node to key an "
                            "abundance to, so each is dropped from every profile"
                        ),
                        n_records=0,
                        items=list(resolution.unmapped),
                    ),
                    DropRule(
                        rule="accession_names_several_loci",
                        scope="protein_accession",
                        description=(
                            "the GOA file gives the accession more than one locus tag, "
                            "so one abundance column stands for several genes and "
                            "cannot be attributed to any of them"
                        ),
                        n_records=0,
                        items=sorted(resolution.multi_locus),
                    ),
                ],
                reconciliation=report,
                notes=[
                    f"one record per deposited strain row over "
                    f"{len(reference_profile.protein_abundance)} of "
                    f"{len(panel.accessions)} accessions "
                    f"({len(locus_of) / len(panel.accessions):.4f} resolved through "
                    f"{crosswalk.member}); {len(resolution.unmapped)} carry no locus "
                    f"tag and {len(resolution.multi_locus)} carry several, all listed "
                    "in preprocess/dropped_accessions.csv",
                    f"the {BENCHLING_CONTROL_LABEL!r} row's own profile is the "
                    "phenotype_reference: a REAL measured control proteome, which is "
                    "what this family stores rather than the ratio denominator the "
                    "Table S3 and Tables S4/S5 families carry",
                    f"{panel.zero_cells} of "
                    f"{len(panel.rows) * len(panel.accessions)} released abundance "
                    "cells are exactly 0 and none is blank; a released 0 is a present "
                    "measurement and is kept verbatim, never imputed and never dropped",
                    f"measurement_type = {PANEL_PROTEOME_MEASUREMENT_TYPE!r}: "
                    f"'{SOURCED_VALUES['quantification'].quote}' This is an ABSOLUTE "
                    "per-strain signal, so it is a different scale from this module's "
                    "two ratio families and from Carruthers 2025's two proteome scales, "
                    "and the shared protein verifier asserts one measurement_type per "
                    "dataset",
                    f"n_replicates = {PANEL_N_REPLICATES} for every key: the deposit "
                    "renders one number per (strain, protein) with no replicate column "
                    "and no uncertainty, so one sample is the support of one stored "
                    "abundance",
                    f"RetrievalMethod.manual_browser. {BENCHLING_PASTE_CAVEAT}",
                    *proofs,
                ],
            )
        )
        log.info(
            "Yunus2026 panel proteome: %d records over %d of %d accessions "
            "(%d unmapped, %d multi-locus); %d marked rows dropped",
            idx,
            len(locus_of),
            len(panel.accessions),
            len(resolution.unmapped),
            len(resolution.multi_locus),
            marker_rule.n_records,
        )


# --------------------------------------------------------------------------- #
# Verification (L0-L4) of a built LMDB
# --------------------------------------------------------------------------- #
DATASETS: dict[str, dict[str, Any]] = {
    "crispri_knockdown_yunus2026": {
        "cls": CrispriKnockdownYunus2026Dataset,
        "root": KNOCKDOWN_ROOT_REL,
        "page": "Supplementary Table S3 (mmc1.docx)",
        "method": "relative expression of each strain's own CRISPRi target protein "
        "against the control strain (DIA-NN Top3); reference = the control strain at "
        "the ratio's denominator (1.0)",
    },
    "crispri_array_yunus2026": {
        "cls": CrispriArrayYunus2026Dataset,
        "root": ARRAY_ROOT_REL,
        "page": "Supplementary Tables S8-S12 (mmc1.docx)",
        "method": "per-construct mean relative expression of the five representative "
        "proteins over three biological replicates, with the sample SD over sqrt(n); "
        "reference = the control strain at the ratio's denominator (1.0)",
    },
    "crispri_differential_proteome_yunus2026": {
        "cls": CrispriDifferentialProteomeYunus2026Dataset,
        "root": DIFFERENTIAL_ROOT_REL,
        "page": "Supplementary Tables S4 and S5 (mmc1.docx)",
        "method": "the PP_4188 strain's released per-protein Fold Change against the "
        "control strain (DIA-NN Top3) over three biological replicates, with the "
        "released unadjusted equal-variance t-test p-value; reference = the control "
        "strain at the ratio's denominator (1.0)",
    },
}


def verifier_provenance(name: str) -> Provenance:
    """The verifier's own provenance record for one of the two families."""
    spec = DATASETS[name]
    return Provenance(
        source_uri=SI_SOURCE_URL,
        citation_key=CITATION_KEY,
        sha256=SI_DOCX_SHA256,
        method=str(spec["method"]),
        page=str(spec["page"]),
    )


def _l4_reference_is_the_ratio_denominator(
    records: Sequence[Mapping[str, Any]],
) -> LevelResult:
    """L4: every reference value is exactly ``REFERENCE_RELATIVE_EXPRESSION``.

    The released numbers are ratios to the control strain, so the reference may hold
    the denominator and nothing else; a reference that drifted off 1.0 would silently
    rescale every record in the dataset. The denominator is read from the scale
    (``FoldChangeScale.linear.neutral_value``), so the rule cannot disagree with the
    ``fold_change_scale`` the records were written on.
    """
    bad: list[str] = []
    n = 0
    for index, record in enumerate(records):
        levels = record["reference"]["phenotype_reference"]["protein_fold_change"]
        for tag, value in levels.items():
            n += 1
            if float(value) != REFERENCE_RELATIVE_EXPRESSION:
                bad.append(f"record {index} {tag}={value}")
    return LevelResult(
        level=Level.L4,
        name="reference_is_the_ratio_denominator",
        passed=not bad,
        message=(
            f"all {n} reference values equal {REFERENCE_RELATIVE_EXPRESSION}"
            if not bad
            else f"{len(bad)} reference values are not the ratio's denominator"
        ),
        details={"n_values": n, "examples": bad[:10]},
    )


# --------------------------------------------------------------------------- #
# The two deposited families: their own L0-L4 battery
# --------------------------------------------------------------------------- #
#: The two families built from the hand-deposited Benchling tables, each with the
#: ``verify_build`` family name that runs its battery.
BIOPRODUCTION_DATASETS: dict[str, dict[str, Any]] = {
    "isoprenol_titer_yunus2026": {
        "cls": IsoprenolTiterYunus2026Dataset,
        "root": TITER_ROOT_REL,
        "family": "titer",
        "page": f"{BENCHLING_TITER_REL} (manual deposit of {BENCHLING_INPUT_URL})",
        "method": "the deposited 'isoprenol_production' column, one number per strain "
        f"row, stored verbatim under ug_per_ml because the paper's unit is mg/L "
        f"('{SOURCED_VALUES['titer_unit'].quote}') and 1 mg/L is exactly 1 ug/mL; "
        f"reference = the released {BENCHLING_CONTROL_LABEL!r} row's own titer. "
        f"{BENCHLING_PASTE_CAVEAT}",
    },
    "crispri_panel_proteome_yunus2026": {
        "cls": CrispriPanelProteomeYunus2026Dataset,
        "root": PANEL_PROTEOME_ROOT_REL,
        "family": "panel_proteome",
        "page": f"{BENCHLING_TITER_REL} (manual deposit of {BENCHLING_INPUT_URL})",
        "method": "the deposited table's 253 UniProt accession columns as an ABSOLUTE "
        "per-strain DIA-NN Top3 signal, re-keyed to locus tags through the assembly "
        f"set's GOA proteome file; reference = the released {BENCHLING_CONTROL_LABEL!r} "
        f"row's own profile. {BENCHLING_PASTE_CAVEAT}",
    },
}

Family = Literal["titer", "panel_proteome"]


def bioproduction_provenance(name: str) -> Provenance:
    """The verifier's own provenance record for one deposited family."""
    spec = BIOPRODUCTION_DATASETS[name]
    return Provenance(
        source_uri=BENCHLING_TITER_REL,
        citation_key=CITATION_KEY,
        sha256=BENCHLING_TITER_SHA256,
        method=str(spec["method"]),
        page=str(spec["page"]),
    )


def _kept_records(dataset_root: str) -> int:
    """The record count this build's own drop log accounts for."""
    drops = DropLog.model_validate_json(
        Path(dataset_root, "preprocess", "dropped_records.json").read_text()
    )
    drops.check()
    return drops.kept_records


def _deposited_panel(data_root: str | None) -> BenchlingStrainTable:
    """Re-read the deposited table out of the raw mirror, verifying its sha256."""
    path = raw_mirror_dir(data_root) / BENCHLING_TITER_REL
    got = _sha256(path)
    if got != BENCHLING_TITER_SHA256:
        raise RuntimeError(
            f"{path} has sha256 {got}, pinned {BENCHLING_TITER_SHA256}; the oracle "
            "cannot read a deposit that drifted"
        )
    return read_benchling_strain_table(path)


def _l3_titer_reference_is_one_released_control(
    records: Sequence[Mapping[str, Any]],
) -> LevelResult:
    """L3: every record references ONE titer, and it is a measured amount.

    The deposit releases a single control row, so a second reference titer would mean
    the build invented one; a reference of 1.0 would mean it used a ratio denominator
    where this family stores an absolute titer.
    """
    values = {
        float(record["reference"]["phenotype_reference"]["titer"]) for record in records
    }
    passed = len(values) == 1 and all(value > 0.0 for value in values)
    return LevelResult(
        level=Level.L3,
        name="titer_reference_is_one_released_control",
        passed=passed,
        message=(
            f"all {len(records)} records reference the one released control titer "
            f"{sorted(values)}"
            if passed
            else f"{len(values)} distinct reference titers {sorted(values)[:5]}"
        ),
        details={"reference_titers": sorted(values)},
    )


def _l4_titers_are_the_deposited_column(
    records: Sequence[Mapping[str, Any]], data_root: str | None = None
) -> LevelResult:
    """L4: the stored titers are exactly the deposited column, control excluded.

    Read back out of the sha256-verified deposit: the multiset of stored titers must
    equal the multiset of deposited titers for the rows the drop rules kept, and the
    reference must be the control row's own number. Nothing is rescaled, because 1 mg/L
    is exactly 1 ug/mL.
    """
    panel = _deposited_panel(data_root)
    expected = sorted(
        row.isoprenol_production for row in panel.strains if not row.marked
    )
    stored = sorted(
        float(record["experiment"]["phenotype"]["titer"]) for record in records
    )
    references = {
        float(record["reference"]["phenotype_reference"]["titer"]) for record in records
    }
    control = panel.control.isoprenol_production
    passed = stored == expected and references == {control}
    return LevelResult(
        level=Level.L4,
        name="titers_are_the_deposited_column",
        passed=passed,
        message=(
            f"{len(stored)} stored titers are the deposited column verbatim and the "
            f"reference is the control row's {control}"
            if passed
            else f"{len(stored)} stored vs {len(expected)} deposited titers; "
            f"references {sorted(references)[:5]}, control {control}"
        ),
        details={
            "n_stored": len(stored),
            "n_deposited": len(expected),
            "control_titer": control,
            "reference_titers": sorted(references),
        },
    )


def _l1_panel_key_set_is_shared(records: Sequence[Mapping[str, Any]]) -> LevelResult:
    """L1: every record carries the SAME protein key set.

    The deposit gives every strain row all 253 columns, so after the accession
    crosswalk every record's key set is the same resolved set; a record with a different
    set would mean a column went missing for one strain.
    """
    sets = {
        frozenset(record["experiment"]["phenotype"]["protein_abundance"])
        for record in records
    }
    passed = len(sets) == 1
    sizes = sorted(len(keys) for keys in sets)
    return LevelResult(
        level=Level.L1,
        name="panel_key_set_is_shared",
        passed=passed,
        message=(
            f"all {len(records)} records carry the same {sizes[0]} protein keys"
            if passed
            else f"{len(sets)} distinct key sets, sizes {sizes[:5]}"
        ),
        details={"n_key_sets": len(sets), "key_set_sizes": sizes[:5]},
    )


def _l3_panel_reference_is_a_measured_control(
    records: Sequence[Mapping[str, Any]],
) -> LevelResult:
    """L3: the reference is the control strain's MEASURED profile, not a denominator.

    This family stores an absolute signal, so its reference is a real control proteome.
    A reference of all ``REFERENCE_RELATIVE_EXPRESSION`` would mean the ratio families'
    denominator leaked into an absolute scale.
    """
    profiles = {
        json.dumps(
            record["reference"]["phenotype_reference"]["protein_abundance"],
            sort_keys=True,
        )
        for record in records
    }
    distinct_values = {
        float(value)
        for record in records
        for value in record["reference"]["phenotype_reference"][
            "protein_abundance"
        ].values()
    }
    passed = len(profiles) == 1 and distinct_values != {REFERENCE_RELATIVE_EXPRESSION}
    return LevelResult(
        level=Level.L3,
        name="panel_reference_is_a_measured_control",
        passed=passed,
        message=(
            f"all {len(records)} records reference one control profile of "
            f"{len(distinct_values)} distinct measured values"
            if passed
            else f"{len(profiles)} distinct reference profiles over "
            f"{len(distinct_values)} distinct values"
        ),
        details={
            "n_reference_profiles": len(profiles),
            "n_distinct_reference_values": len(distinct_values),
        },
    )


def _l4_panel_profiles_are_the_deposited_columns(
    records: Sequence[Mapping[str, Any]], data_root: str | None = None
) -> LevelResult:
    """L4: the stored profiles are the deposited columns, re-keyed and nothing else.

    Read back out of the sha256-verified deposit and re-crosswalked: the set of stored
    profiles must be exactly the set the deposit's kept rows imply, so a released 0
    survives and no value is imputed.
    """
    panel = _deposited_panel(data_root)
    genome = bacterial_genome("pputida", KT2440_STRAIN, data_root)
    resolution = resolve_uniprot_accessions(
        uniprot_locus_crosswalk(genome, data_root),
        panel.accessions,
        label="yunus2026-panel-oracle",
    )
    locus_of = dict(resolution.resolved)
    expected = sorted(
        json.dumps(
            {
                locus_of[accession]: value
                for accession, value in row.abundance.items()
                if accession in locus_of
            },
            sort_keys=True,
        )
        for row in panel.strains
        if not row.marked
    )
    stored = sorted(
        json.dumps(
            record["experiment"]["phenotype"]["protein_abundance"], sort_keys=True
        )
        for record in records
    )
    passed = stored == expected
    zeros = sum(
        1
        for record in records
        for value in record["experiment"]["phenotype"]["protein_abundance"].values()
        if float(value) == 0.0
    )
    return LevelResult(
        level=Level.L4,
        name="panel_profiles_are_the_deposited_columns",
        passed=passed,
        message=(
            f"{len(stored)} stored profiles over {len(locus_of)} of "
            f"{len(panel.accessions)} accessions are the deposited columns verbatim, "
            f"{zeros} released zeros kept"
            if passed
            else f"{len(stored)} stored vs {len(expected)} deposited profiles"
        ),
        details={
            "n_stored": len(stored),
            "n_deposited": len(expected),
            "n_resolved_accessions": len(locus_of),
            "n_zeros_kept": zeros,
        },
    )


def titer_report(
    records: Sequence[dict[str, Any]], expected_count: int, data_root: str | None = None
) -> VerificationReport:
    """The deposited titer family's L0-L4 report over already-loaded records."""
    from torchcell.verification.product_titer import verify_product_titer_dataset

    report = verify_product_titer_dataset(
        [dict(record) for record in records],
        dataset_name=IsoprenolTiterYunus2026Dataset.__name__,
        provenance=bioproduction_provenance("isoprenol_titer_yunus2026"),
        expected_count=expected_count,
        titer_unit=ConcentrationUnit.ug_per_ml.value,
        titer_unit_detail=(
            f"the paper releases mg/L ('{TITER_UNIT_SOURCE.quote}') and "
            "ConcentrationUnit has no mg/L member; 1 mg/L == 1 ug/mL exactly, so the "
            "deposited number is stored verbatim under the numerically identical unit"
        ),
        se_tol=0.0,
        pathway_gene_counts=(0,),
        product_names=("isoprenol",),
    )
    report.add(_l3_titer_reference_is_one_released_control(records))
    report.add(_l4_titers_are_the_deposited_column(records, data_root))
    return report


def panel_proteome_report(
    records: Sequence[dict[str, Any]], expected_count: int, data_root: str | None = None
) -> VerificationReport:
    """The deposited protein-panel family's L0-L4 report over already-loaded records."""
    from torchcell.verification.protein import verify_protein_dataset

    report = verify_protein_dataset(
        [dict(record) for record in records],
        dataset_name=CrispriPanelProteomeYunus2026Dataset.__name__,
        provenance=bioproduction_provenance("crispri_panel_proteome_yunus2026"),
        expected_count=expected_count,
        allow_duplicate_orfs=True,
    )
    report.add(_l1_panel_key_set_is_shared(records))
    report.add(_l3_panel_reference_is_a_measured_control(records))
    report.add(_l4_panel_profiles_are_the_deposited_columns(records, data_root))
    return report


def verify_build(
    dataset_root: str, data_root: str | None = None, *, family: Family
) -> VerificationReport:
    """Run one deposited family's L0-L4 gate over a built tree and write the report.

    ``family`` is ``"titer"`` or ``"panel_proteome"``. The report is written to
    ``<dataset_root>/preprocess/verification_report.json``. The expected record count is
    read from the build's OWN drop log, whose arithmetic is re-checked here, so the
    count the verifier asserts is the count the build accounted for.
    """
    from torchcell.verification.runners import load_records

    records = load_records(dataset_root)
    expected = _kept_records(dataset_root)
    build = {"titer": titer_report, "panel_proteome": panel_proteome_report}[family]
    report = build(records, expected, data_root)
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    os.makedirs(osp.dirname(out), exist_ok=True)
    Path(out).write_text(report.model_dump_json(indent=2))
    return report


def _l3_scale_and_basis_are_one_contrast(
    records: Sequence[Mapping[str, Any]],
) -> LevelResult:
    """L3: one ``(fold_change_scale, reference_basis)`` pair across the dataset.

    The pair is what makes two columns comparable, so two pairs in one store would mean
    two different contrasts pooled under one dataset name: a linear ratio averaged with
    a log2 one, or a ratio to the control strain averaged with a ratio to something
    else. Scale is also asserted to be the one this loader measured on the pinned docx.
    """
    pairs = sorted(
        {
            (
                str(record["experiment"]["phenotype"]["fold_change_scale"]),
                str(record["experiment"]["phenotype"]["reference_basis"]),
            )
            for record in records
        }
    )
    scales = {scale for scale, _ in pairs}
    passed = len(pairs) == 1 and scales == {str(FOLD_CHANGE_SCALE)}
    return LevelResult(
        level=Level.L3,
        name="scale_and_basis_are_one_contrast",
        passed=passed,
        message=(
            f"all {len(records)} records are {pairs[0][0]!r} against one basis"
            if passed
            else f"{len(pairs)} distinct (scale, basis) pairs, scales {sorted(scales)}"
        ),
        details={
            "n_pairs": len(pairs),
            "scales": sorted(scales),
            "expected_scale": str(FOLD_CHANGE_SCALE),
            "bases": sorted({basis for _, basis in pairs}),
        },
    )


def _l3_p_values_are_unadjusted(records: Sequence[Mapping[str, Any]]) -> LevelResult:
    """L3: a stored p-value is an unadjusted probability and names no correction.

    Nothing in this paper adjusts the released per-protein t-test p-values, so a record
    that grew an adjusted map or an adjustment method would be carrying a correction no
    mirrored byte states. A dataset with no p-values at all passes trivially.
    """
    n_p = 0
    bad: list[str] = []
    for index, record in enumerate(records):
        phenotype = record["experiment"]["phenotype"]
        p_values = phenotype["protein_fold_change_p_value"] or {}
        n_p += len(p_values)
        for tag, value in p_values.items():
            if not 0.0 < float(value) <= 1.0:
                bad.append(f"record {index} {tag}={value}")
        if phenotype["protein_fold_change_p_value_adjusted"] is not None:
            bad.append(f"record {index} carries an adjusted p-value map")
        if phenotype["p_value_adjustment_method"] != DIFFERENTIAL_P_VALUE_ADJUSTMENT:
            bad.append(
                f"record {index} names the correction "
                f"{phenotype['p_value_adjustment_method']!r}"
            )
    return LevelResult(
        level=Level.L3,
        name="p_values_are_unadjusted_probabilities",
        passed=not bad,
        message=(
            f"{n_p} stored p-values are unadjusted probabilities naming no correction"
            if not bad
            else f"{len(bad)} p-value violations"
        ),
        details={"n_p_values": n_p, "examples": bad[:10]},
    )


def run_verification(name: str, data_root: str | None = None) -> VerificationReport:
    """Run the shared protein verifier on the fold-change label (L0-L3), the bacterial
    L4 containment, the one-contrast and unadjusted-p-value rules, the
    reference-denominator rule and the provenance audit of every ``SOURCED_VALUES``
    entry on a built dev LMDB; write ``preprocess/verification_report.json``.
    """
    from torchcell.verification.protein import protein_gene_set, verify_protein_dataset
    from torchcell.verification.runners import (
        _gene_set_for_reference,
        _write_report,
        load_records,
    )

    base = data_root or _data_root()
    spec = DATASETS[name]
    abs_root = osp.join(base, str(spec["root"]))
    records = load_records(abs_root)
    drops = DropLog.model_validate_json(
        Path(abs_root, "preprocess", "dropped_records.json").read_text()
    )
    report = verify_protein_dataset(
        records,
        dataset_name=spec["cls"].__name__,
        provenance=verifier_provenance(name),
        expected_count=drops.kept_records,
        allow_duplicate_orfs=True,
        label_key="protein_fold_change",
        se_key="protein_fold_change_se",
    )
    references = {
        json.dumps(r["reference"]["genome_reference"], sort_keys=True) for r in records
    }
    if len(references) != 1:
        raise ValueError(f"{len(references)} distinct genome references; expected 1")
    universe = _gene_set_for_reference(json.loads(references.pop()), base)
    measured = protein_gene_set(records) | {
        tag
        for record in records
        for tag in record["experiment"]["phenotype"]["protein_fold_change"]
    }
    missing = sorted(measured - universe)
    report.add(
        LevelResult(
            level=Level.L4,
            name="gene_containment_kt2440_locus_tags",
            passed=not missing,
            message=f"{len(measured) - len(missing)} of {len(measured)} perturbed and "
            "measured loci are KT2440 GenBank gene rows",
            details={
                "n_measured": len(measured),
                "n_universe": len(universe),
                "missing_examples": missing[:20],
            },
        )
    )
    report.add(_l3_scale_and_basis_are_one_contrast(records))
    report.add(_l3_p_values_are_unadjusted(records))
    report.add(_l4_reference_is_the_ratio_denominator(records))
    library = Path(base) / "torchcell-library"
    if library_available(library):
        for value in SOURCED_VALUES.values():
            report.add(audit_sourced_value(value, library))
    else:
        log.warning(
            "the literature mirror is not at %s, so the %d provenance audits of "
            "SOURCED_VALUES did not run; the L0-L4 record gate below is unaffected",
            library,
            len(SOURCED_VALUES),
        )
    _write_report(report, osp.join(abs_root, "preprocess"))
    return report


#: Every dataset class this module serves, for the CLI and for callers that iterate.
ALL_DATASETS: dict[str, dict[str, Any]] = {**DATASETS, **BIOPRODUCTION_DATASETS}


def print_table_digests(path: str | Path) -> dict[str, str]:
    """Print the parsed-table digests of a docx, for pinning :data:`TABLE_DIGESTS`."""
    tables = read_docx_tables(path)
    digests = {name: table_digest(tables[TABLE_INDEX[name]]) for name in TABLE_DIGESTS}
    for name, digest in digests.items():
        print(f'    "{name}": "{digest}",')
    return digests


def main(argv: list[str] | None = None) -> int:
    """CLI: ``deposit`` the raw mirror, ``build`` a dev LMDB, ``verify`` it, or
    ``digests`` to print the parsed-table digests of the pinned docx.
    """
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        prog="python -m torchcell.datasets.pputida.yunus2026"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    deposit = sub.add_parser("deposit", help="deposit the raw mirror")
    deposit.add_argument(
        "--retrieve-into",
        default=None,
        help="re-run the recorded Elsevier retrieval into this directory and deposit "
        "those bytes; without it the literature mirror's captured SI file is used",
    )
    build = sub.add_parser("build", help="build (or load) a dev-tree LMDB")
    build.add_argument("--dataset", choices=sorted(ALL_DATASETS), default=None)
    verify = sub.add_parser("verify", help="run L0-L4 on the built dev-tree LMDBs")
    verify.add_argument("--dataset", choices=sorted(ALL_DATASETS), default=None)
    digests = sub.add_parser("digests", help="print the parsed-table digests")
    digests.add_argument("--path", default=None)
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = _data_root()
    if args.command == "deposit":
        if args.retrieve_into is not None:
            source: Path = retrieve_raw_files(args.retrieve_into)[SI_DOCX]
        else:
            source = library_dir(data_root) / "si" / SI_DOCX
        print(deposit_raw_mirror(source=source, data_root=data_root))
        return 0
    if args.command == "digests":
        path = args.path or raw_mirror_dir(data_root) / SI_MIRROR_RELPATH
        print_table_digests(path)
        return 0
    names = [args.dataset] if args.dataset else sorted(ALL_DATASETS)
    if args.command == "build":
        genome = bacterial_genome("pputida", "KT2440", data_root)
        for name in names:
            spec = ALL_DATASETS[name]
            dataset = spec["cls"](
                root=osp.join(data_root, str(spec["root"])), pputida_genome=genome
            )
            print(f"{spec['cls'].__name__}: len = {len(dataset)}")
        return 0
    ok = True
    for name in names:
        if name in BIOPRODUCTION_DATASETS:
            spec = BIOPRODUCTION_DATASETS[name]
            report = verify_build(
                osp.join(data_root, str(spec["root"])),
                data_root,
                family=str(spec["family"]),  # type: ignore[arg-type]
            )
        else:
            report = run_verification(name, data_root)
        print(report.summary())
        ok = ok and report.passed
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
