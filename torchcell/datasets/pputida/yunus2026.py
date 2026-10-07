# torchcell/datasets/pputida/yunus2026
# [[torchcell.datasets.pputida.yunus2026]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/pputida/yunus2026
# Test file: tests/torchcell/datasets/pputida/test_yunus2026.py
"""Yunus 2026 CRISPRi knockdown panel on an isoprenol-producing P. putida KT2440 chassis.

Yunus, Carruthers, Chen, Gin, Baidoo, Petzold, Garcia Martin, Adams, Mukhopadhyay and
Lee 2026 (Metab. Eng., doi:10.1016/j.ymben.2025.11.007) screened CRISPRi knockdowns
chosen by FluxRETAP against knockdowns chosen by intuition, in the engineered
isoprenol-producing strain ``IY1452``, and built the arrays with VAMMPIRE. Two released
tables of the one supplementary file carry a per-strain number, and this module serves
both, as two dataset classes because ``ExperimentDataset.transform_item`` validates
against ONE ``experiment_class``:

- :class:`CrispriKnockdownYunus2026Dataset` -- Supplementary Table S3, the 125-sample
  shotgun-proteomics screen: one record per CRISPRi strain, the relative expression of
  its OWN target protein against the control strain.
- :class:`CrispriArrayYunus2026Dataset` -- Supplementary Tables S8-S12, the sgRNA-array
  position study: one record per multiplexed construct, the relative expression of each
  of the five representative proteins the panel measured in it, over three biological
  replicates with the sample SD.

THE ISOPRENOL TITERS ARE NOT LOADED, BECAUSE THEY ARE NOT RELEASED. This is a
production campaign and its headline readout is isoprenol titer, yet no mirrored byte
carries a per-strain titer. Fig. 4B-J, Supplementary Fig. S6 (intuition group),
Supplementary Fig. S7 (OD600) and Supplementary Fig. S9 (multiplexed strains) are bar
charts; the text states exactly two titers, ``1469 mg/L`` for the ``PP_4188`` knockdown
and ``958 mg/L`` for ``PP_0168`` (:data:`SOURCED_VALUES`), and never states the control
strain's titer. A ``ProductTiterExperimentReference`` requires a reference titer, so the
titer family cannot be built without inventing the denominator, and it is not built.
Supplementary Note 1 does link the per-strain input table of its Pearson analysis
(``isoprenol_production`` beside the protein columns) as a Benchling share page, which is
a JavaScript single-page app whose data loads through an authenticated internal API
(``/1/api/...`` answers HTTP 401 "permission denied - not logged in"; the public share
page itself answers 200 with no data in the HTML, measured 2026-10-07). The manual
recipe is in the raw mirror's ``si_expected`` and in the note. ``mmc1.docx`` is the
publisher's ONLY supplementary component: ``mmc1..mmc4`` x ``{docx,xlsx,pdf,zip,csv}``
were probed on ``ars.els-cdn.com`` on 2026-10-07 and only ``mmc1.docx`` exists (HTTP
206; every other combination 404).

WHAT ONE STORED NUMBER IS, AND WHY IT IS A ``ProteinAbundancePhenotype``. Both released
tables report a RATIO: the target protein's abundance in the CRISPRi strain divided by
its abundance in the control strain, from the same DIA-NN Top3 quantification. The
record carries the strain's own number and the reference carries ``1.0``, which is the
ratio's denominator by definition rather than a measured quantity, so experiment /
reference reproduces the released value exactly and nothing is imputed.
``measurement_type`` names the scale (:data:`MEASUREMENT_TYPE`), which is the axis that
keeps these numbers from ever being compared with an absolute Top3 signal such as the
Carruthers 2025 proteome's. ``ProteinAbundancePhenotype``'s docstring asks for an
absolute per-strain quantity, so this is a deliberate, documented stretch of that class
and the PR asks for the typed ``abundance_basis`` axis that would make it exact.

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
replicate DESIGN behind one sample is not stated, so ``protein_abundance_se`` is a typed
gap. Tables S8-S12 release three per-replicate values per construct (``R1``, ``R2``,
``R3``) and the Fig. 3J-N caption states "Error bars represent standard deviation from
three biological replicates", so those records carry ``n_replicates = 3`` and an SE
derived as the sample SD over sqrt(3). The two families also disagree numerically for a
genotype they share -- Table S3 gives ``PP_4188`` 0.2213 while Table S8's three
replicates mean 0.2509 -- which is the evidence that they are separate runs and must not
be pooled into one record.

NOT LOADED, with the reason: Supplementary Tables S4 and S5 (145 downregulated and 193
upregulated proteins of the ``PP_4188`` strain against the control, as fold change,
log2 fold change and a t-test p-value) are a derived differential statistic for a single
strain with no per-replicate values released, and no phenotype class models a
per-protein differential with its own test; Supplementary Table S6 (plasmids) and the
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
from typing import Any, ClassVar
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
    BacterialReferenceStrain,
    BacterialStrainBackground,
    Concentration,
    ConcentrationUnit,
    CrisprConstruct,
    Environment,
    EnvironmentPhysicalPerturbation,
    Experiment,
    ExperimentReference,
    Genotype,
    PhysicalFactor,
    ProteinAbundancePhenotype,
    Publication,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.bacteria_common import (
    LocusTagReconciliation,
    assembly_reference,
    bacterial_genome,
    reconcile_locus_tags,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
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

#: The MinerU OCR of the publisher PDF; every ``SourcedValue`` quotes these bytes.
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "32ab4cd3753a930c6ad983809e7083a06159b7b0feb6b5252ace180273bbe563"

#: Supplementary file 1, the ONLY supplementary component the publisher serves.
SI_DOCX = "si1.docx"
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
#: The reference strain's value on that scale: the ratio's denominator, by definition.
REFERENCE_RELATIVE_EXPRESSION = 1.0

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

#: What the paper releases that no loader here consumes, with the reason for each.
NOT_LOADED: tuple[str, ...] = (
    "the per-strain isoprenol titers (Fig. 4B-J, Supplementary Figs. S6, S7 and S9): "
    "released as bar charts only. The text states exactly two titers (1469 mg/L for "
    "PP_4188, 958 mg/L for PP_0168) and never the control strain's, so a "
    "ProductTiterExperimentReference has no reference titer and the family is not built",
    f"the Supplementary Note 1 Benchling pages ({BENCHLING_ANALYSIS_URL} and "
    f"{BENCHLING_INPUT_URL}), whose input table is 'strain, isoprenol_production, "
    "<protein columns>' -- the only released per-strain titers. NOT deposited: the "
    "share page is a JavaScript single-page app and its data loads through an "
    "authenticated internal API (benchling.com/1/api/... answers HTTP 401 'permission "
    "denied - not logged in'; the share page answers 200 with no data in the HTML), "
    "measured 2026-10-07. MANUAL RECIPE: open the input-table link in a browser, sign "
    "in or use the share view, export the notebook table to CSV, deposit it under "
    "data/ with RetrievalMethod.manual_browser and the sha256 of the bytes that arrive",
    "Supplementary Tables S4 and S5 (145 downregulated and 193 upregulated proteins of "
    "the PP_4188 strain, as fold change, log2 fold change and a t-test p-value): a "
    "derived differential statistic for a single strain with no per-replicate values "
    "released, and no phenotype class models a per-protein differential with its own "
    "test",
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


#: The raw-mirror path of the one consumed file.
SI_MIRROR_RELPATH = f"data/{SI_DOCX}"
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
            )
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
) -> ProteinAbundancePhenotype:
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
    return ProteinAbundancePhenotype(
        protein_abundance=dict(values),
        protein_abundance_se=None,
        n_replicates={tag: n_replicates for tag in values},
        measurement_type=MEASUREMENT_TYPE,
        provenance_gaps=[
            ProvenanceGap(
                field="protein_abundance_se",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="Supplementary Table S3 releases one number per strain with no "
                "uncertainty, and the replicate design behind one proteomics sample is "
                "not stated; there is nothing to derive an SE from",
            )
        ],
    )


def array_phenotype(
    means: Mapping[str, float],
    standard_errors: Mapping[str, float],
    counts: Mapping[str, int],
) -> ProteinAbundancePhenotype:
    """One array construct's per-protein mean relative expression with its SE."""
    if set(means) != set(standard_errors) or set(means) != set(counts):
        raise RuntimeError("mean, SE and replicate-count keys disagree")
    return ProteinAbundancePhenotype(
        protein_abundance=dict(means),
        protein_abundance_se=dict(standard_errors),
        n_replicates=dict(counts),
        measurement_type=MEASUREMENT_TYPE,
    )


def reference_phenotype(
    tags: Iterable[str], *, n_replicates: int, with_se: bool
) -> ProteinAbundancePhenotype:
    """The control strain on the released scale: 1.0 for every measured protein.

    Not a measurement -- it is the ratio's denominator, and the only value the released
    numbers are expressed against, so experiment / reference reproduces the source's
    number exactly. ``REFERENCE_RELATIVE_EXPRESSION`` names it so a reader cannot
    mistake it for a quantified abundance.
    """
    keys = sorted(set(tags))
    if not keys:
        raise RuntimeError("a reference needs the record's measured proteins")
    values = {tag: REFERENCE_RELATIVE_EXPRESSION for tag in keys}
    if with_se:
        return ProteinAbundancePhenotype(
            protein_abundance=values,
            protein_abundance_se={tag: 0.0 for tag in keys},
            n_replicates={tag: n_replicates for tag in keys},
            measurement_type=MEASUREMENT_TYPE,
        )
    return relative_expression_phenotype(values, n_replicates=n_replicates)


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
    """One retention rule, what it removed, and the items it removed."""

    rule: str
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
        return BacterialProteinAbundanceExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialProteinAbundanceExperimentReference

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
                experiment = BacterialProteinAbundanceExperiment(
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
                reference = BacterialProteinAbundanceExperimentReference(
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
                experiment = BacterialProteinAbundanceExperiment(
                    dataset_name=self.name,
                    genotype=Genotype(perturbations=perturbations),
                    environment=environment,
                    phenotype=array_phenotype(means, standard_errors, counts),
                )
                reference = BacterialProteinAbundanceExperimentReference(
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
    rescale every record in the dataset.
    """
    bad: list[str] = []
    n = 0
    for index, record in enumerate(records):
        levels = record["reference"]["phenotype_reference"]["protein_abundance"]
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


def run_verification(name: str, data_root: str | None = None) -> VerificationReport:
    """Run the protein-abundance family verifier (L0-L3), the bacterial L4 containment,
    the reference-denominator rule and the provenance audit of every
    ``SOURCED_VALUES`` entry on a built dev LMDB; write
    ``preprocess/verification_report.json``.
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
        for tag in record["experiment"]["phenotype"]["protein_abundance"]
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
    build.add_argument("--dataset", choices=sorted(DATASETS), default=None)
    verify = sub.add_parser("verify", help="run L0-L4 on the built dev-tree LMDBs")
    verify.add_argument("--dataset", choices=sorted(DATASETS), default=None)
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
    names = [args.dataset] if args.dataset else sorted(DATASETS)
    if args.command == "build":
        genome = bacterial_genome("pputida", "KT2440", data_root)
        for name in names:
            spec = DATASETS[name]
            dataset = spec["cls"](
                root=osp.join(data_root, str(spec["root"])), pputida_genome=genome
            )
            print(f"{spec['cls'].__name__}: len = {len(dataset)}")
        return 0
    ok = True
    for name in names:
        report = run_verification(name, data_root)
        print(report.summary())
        ok = ok and report.passed
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
