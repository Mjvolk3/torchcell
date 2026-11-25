# torchcell/datasets/scerevisiae/smith2016
# [[torchcell.datasets.scerevisiae.smith2016]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/smith2016
# Test file: tests/torchcell/datasets/scerevisiae/test_smith2016.py
"""Smith 2016 quantitative CRISPRi chemical-genetic screen (per-guide, per-drug fitness).

Smith JD et al. 2016 (Genome Biol 17:45, doi:10.1186/s13059-016-0900-9;
citation_key ``smithQuantitativeCRISPRInterference2016``) built a regulatable dCas9-Mxi1
CRISPRi platform in S. cerevisiae (BY4741, one pRS416-derived CEN/URA3 plasmid carrying
TetR, an ATc-inducible RPR1-TetO gRNA locus and constitutive dCas9-Mxi1). gRNA tiling
libraries against 20 drug-partner genes were grown as POOLS in competitive-growth assays,
+/- ATc and +/- a small-molecule inhibitor, and each guide's abundance was tracked by
amplicon sequencing.

RECORD = one biological replicate of one (pool x guide x drug-condition), as released
(2026.11.24, #1178). Two record shapes, one ``EnvironmentResponseExperiment`` class:

- an A RECORD per Additional file 10 row: the ATc-induced fold change A of ONE replicate
  pair of cultures (+ATc, -ATc). Every Additional file 10 row is replicate 1 of its
  condition, so ``replicate_id = "1"``. Two checks ``process()`` re-runs on every build:
  Additional file 5 lists replicate 1 for each of the 67 (pool, drug, concentration)
  groups, and in the 1% DMSO and 20 uM fluconazole groups the row's ``-ATC_count`` /
  ``+ATC_count`` equal Additional files 6 / 7's ``Rep-1`` counts (2,148 of 2,148
  comparable cells). Measured once beside them (2026.11.24, not re-run by the build
  because Additional file 9 is not read): those counts also equal Additional file 9's
  ``Count_perfect`` for replicate 1 in all 67 groups.
- a CULTURE RECORD per replicate culture that Additional file 6 (1% DMSO, eight
  replicates, gene-tiling pools) or Additional file 7 (20 uM fluconazole, three
  replicates in the gene-tiling pools, two in the mutant pools) releases and whose A
  Additional file 10 does not: the culture's guide fitness ``f`` (``Rep-k_noATC_f_mean``
  / ``Rep-k_plusATC_f_mean``) with its released ``f_var``. The source releases no A for
  replicates 2-8, only the two cultures' ``f``; ``A_k = f(+ATc) - f(-ATc)`` is left to
  model time (policy: store the per-culture value the source releases, never a loader
  ratio). The +ATc culture's environment carries anhydrotetracycline at the dose
  Additional file 5 states for every sequenced pooled sample (column ``ATC (250ng/ml)``);
  the -ATc culture's environment is the drug alone.

The combined statistics of Additional files 6 / 7 (``Combined_A_mean``,
``Combined_A_var``, ``Combined_*ATC_*``) are the authors' summary over the released
replicates and are not stored: the replicates are. Measured on the pinned files,
``Combined_A_mean`` is the plain mean of the per-replicate ``f(+ATc) - f(-ATc)`` (max
|difference| 0.0009 over 451 + 637 guides) and ``Combined_A_var`` their sample variance
(ddof 1), not the inverse-variance combination the Methods describe; that is a
modeling-time derivation over these records.

Shared by both shapes:

- GENOTYPE: a ``CrisprInterferencePerturbation`` whose ``crispr`` construct carries the
  effector ``dCas9-Mxi1``, the 18/20 nt ``Specificity_sequence`` spacer (joined from
  Additional file 4 by guide name) and the screen ``library_pool``. The spacer keeps
  sibling guides of one gene distinct; the pool keeps the SAME spacer's two independent
  pool measurements distinct. The target is stored AS RELEASED: ``systematic_gene_name``
  is the row's ``ORF`` cell and ``perturbed_gene_name`` its ``Gene`` cell (the ORF where
  that is blank); no genome is consulted (#1045, owner rule 2026-10-12).
- ENVIRONMENT: the shared ``media.SC_URA`` object (the paper's "SCM-Ura"; Smith releases
  no SCM-Ura recipe, so the shipped library gaps are the honest state) carrying the row's
  drug at its released concentration as a ``SmallMoleculePerturbation``, 30 C, aerobic,
  ``duration_generations = 20``.
- PHENOTYPE: ``measurement_type=log2_ratio``, ``assay_type=pooled_competitive_growth_barcode``,
  ``n_samples = 1``, ``replicate_id`` = the source's replicate label. An A record holds
  ``environment_response = A`` with the released ``var(A)`` VERBATIM as
  ``UncertaintyType.variance`` (``UNITS``), reference = the uninduced (-ATc) baseline in
  the same drug environment, A = 0. A culture record holds ``environment_response = f``
  with the released ``f_var`` as ``UncertaintyType.variance`` (``CULTURE_UNITS``),
  reference = the pool's median guide in the same culture environment, f = 0. Both
  variances are the source's per-measurement read-resampling variance ("used the
  variance s^2 of the simulated values as a variance estimate"), not a dispersion over
  replicates, so they are data and stay.

SOURCED VALUES (module-level ``SourcedValue``s anchored to the sha256 of
``smithQuantitativeCRISPRInterference2016/paper.md``):

- ``duration_generations = 20.0``: "This amounted to approximately 20 culture doublings
  from the beginning of the experiment." Exposure duration is part of environment
  identity, and the previous build left it None. ``duration_hours`` is a typed
  ``ProvenanceGap``: the cultures were held in log phase by repeated dilution and no wall
  time is reported.
- ``assay_type``: "the short specificity-determining regions of gRNAs (that is, the
  sequence complementary to the target) can act as unique identifiers of individual
  strains".
- the 1% DMSO vehicle-control dose, from the Methods sentence that also gives the eight
  and three replicate experiments for the tiling pools and for 20 uM fluconazole.

``n_samples = 1`` because every record is one replicate. Until 2026.11.24 this docstring
said the 1% DMSO and 20 uM fluconazole rows of Additional file 10 were the paper's
inverse-variance combination of the eight and three replicates; measured, they are
replicate 1 alone (see RECORD), so the released ``var(A)`` is one replicate's variance
and ``derive_se`` dividing by 1 was right for that reason.

An A record carries NO ATc in its environment: A is the contrast between the induced and
uninduced cultures, and its reference is the uninduced (-ATc) culture. A culture record
is one culture, so its environment states whether ATc was in it. The dose is Additional
file 5's: that workbook lists every sequenced pooled-screen sample (pools broad_tiling,
gene_tiling18 & gene_tiling20, mutant18 & mutant20) with the column ``ATC (250ng/ml)``,
README '"yes" if ATC was included, or "no" if it was not'; ``process()`` reads the dose
from that header. (Before 2026.11.24 this docstring said the 250 ng/mL was not the pooled
screen's; Additional file 5 is the pooled screen's sample sheet.) No solvent is asserted:
"Drugs were dissolved in DMSO" sits in the individual-strain growth-assay section, and the
pooled section does not restate it, so the 1% DMSO control condition is served as a
compound in its own right and the other drugs carry no ``Solvent``.

NOT STORED (``dropped_records.json`` lists each under ``replicates_not_stored``): the
second replicate of 1% DMSO in the mutant pools, which only Additional file 9 releases,
as read counts (no ``MeasurementType`` holds a read count, and computing ``f`` from the
counts would be a loader derivation); and Additional file 5's Rapamycin 10 nM samples of
the gene-tiling pools, for which no file releases a count, fitness or effect.

The screening doses were set by a rule the SI states ("Concentration(s) used for library
screening. These concentrations were selected to inhibit growth of a wild-type strain by
~20%", Additional file 8 ReadMe, sha256 a22ef00ace51...), but ``DoseBasis`` has no IC20
member and adding one changes a class in all 36 served dataset closures, so the rule is
recorded here and in the dendron note instead of in ``Concentration.basis``.

RECORDS (``preprocess/dropped_records.json`` is the accounting): every released row of an
identified compound is a record. AN UNIDENTIFIED COMPOUND IS HELD OUT (2026.10.15,
database constitution principle 2; owner: "if you cannot tell me what goes into the flask
so to speak, then we have no hope of modeling!"). The rows whose drug is a vendor catalog
code -- 1181-0519, 4130-1276 and 0KPI-0099 (ChemDiv), 9121982, 9125678, 7312221,
CBF-666774, 6630449 and 9150499 (ChemBridge), ST016598 (TimTec) -- are held out under
``UNIDENTIFIED_COMPOUND_RULE``: the pinned identity table marks each PROPRIETARY, and
Additional file 8 ("All chemical compounds used in this study are listed in Additional
file 8.", the ``COMPOUND_LIST`` quote) gives only the code and the vendor, no structure,
SMILES or CAS. Measured on the pinned Additional file 10 by the build (2026.10.15): 7,410
of 14,463 rows held out over 10 codes, 7,053 stored. From 2026.10.12 to 2026.10.15 these
rows were stored under the bare code with a typed ``inchikey`` gap (#1045); a gap is for a
missing attribute of an identified compound, never its identity. The ledger lists each
code with its doses and rows under ``unidentified_compounds_held_out``; the EMPTY
admission table ``RELEASED_IDENTITIES`` admits a code the day a public structure for it
is mirrored (a store rebuild). The tenth public compound, ``NSC-180973``, resolves
through the pinned identity table's synonym for tamoxifen, the alias the paper's own
Additional file 8 releases. The two other rules that can leave a row without a record are
a row with no ``A`` or ``var(A)`` and a guide with no spacer in Additional file 4; both
are 0 on the pinned files. An ``ORF`` cell that is not ORF-shaped is counted under
``stored_on_released_gene_name_leaf`` (0 of 14,463 rows) as released (#1045).

DUPLICATE-BY-POOL: 272 (guide, drug, concentration) triples appear twice, always the pool
pair (``broad_tiling``, ``gene_tiling_20bp``) sharing a 20 bp perfect-match spacer. Fitness
is median-centred WITHIN a pool, so the two A values differ by mean 0.95 and up to 7.8 log2
units: they are two independent pooled measurements, not replicates to average and not a
duplicate to drop. ``library_pool`` therefore joins the strain identity.

DATA SOURCE: Additional file 10 ``13059_2016_900_MOESM10_ESM.xlsx`` sheet 'Fitness and
Effect Data' (14,463 rows), Additional file 4 ``13059_2016_900_MOESM4_ESM.xlsx`` sheet
'gRNAs' (the guide -> spacer join), Additional file 5 ``13059_2016_900_MOESM5_ESM.xlsx``
sheet 'Metadata' (the sequenced samples: replicate labels and the ATc dose), and
Additional files 6 / 7 ``13059_2016_900_MOESM6_ESM.xlsx`` sheet 'DMSO_replicate_data' and
``13059_2016_900_MOESM7_ESM.xlsx`` sheet 'Fluconazole_replicate_data' (per-replicate
counts and fitness), all deposited in the raw mirror
``$DATA_ROOT/torchcell-raw/smithQuantitativeCRISPRInterference2016/`` with a
``manifest.json`` recording their Springer ESM URLs and hashes (Additional files 10 and
4 re-retrieved 2026-09-12; 5, 6 and 7 retrieved 2026-11-25 UTC, Additional file 5's
bytes identical to the literature mirror's copy).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import os.path as osp
import re
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
    verify_sha256,
)
from torchcell.datamodels.compound_identity import (
    CompoundResolutionStatus,
    resolve_compound_identity,
    resolved_compound,
)
from torchcell.datamodels.media import SC_URA
from torchcell.datamodels.schema import (
    AssayType,
    Compound,
    Concentration,
    ConcentrationUnit,
    CrisprConstruct,
    CrisprInterferencePerturbation,
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    Publication,
    ReferenceGenome,
    ReleasedIdentifierPerturbation,
    ReleasedKind,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.scerevisiae.hoepfner2014 import ReleasedIdentity
from torchcell.literature.manifest import (
    ROLE_SI_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DOI = "10.1186/s13059-016-0900-9"

CITATION_KEY = "smithQuantitativeCRISPRInterference2016"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

_ESM_BASE = (
    "https://static-content.springer.com/esm/art%3A10.1186%2Fs13059-016-0900-9/"
    "MediaObjects/"
)
EFFECT_FILENAME = "13059_2016_900_MOESM10_ESM.xlsx"
EFFECT_REL = f"si/si_data/{EFFECT_FILENAME}"
_SYSTEMATIC_RE = re.compile(r"^(Y[A-P][LR]\d{3}[WC](-[A-Z])?|Q\d{4})$")

EFFECT_SHEET = "Fitness and Effect Data"
EFFECT_SHA256 = "02962e51e492b0505e8595fc1c80fab5fca8a8c8f05e05969dbff18ddff71cd0"
GUIDE_FILENAME = "13059_2016_900_MOESM4_ESM.xlsx"
GUIDE_REL = f"si/si_data/{GUIDE_FILENAME}"
GUIDE_SHEET = "gRNAs"
GUIDE_SHA256 = "e5eb4e3c7856782e36edff5ef55e680cb43fa8e9944bf8e40d7cb7b4f376c1e5"
SI_RETRIEVED_AT = "2026-09-12"
SAMPLES_FILENAME = "13059_2016_900_MOESM5_ESM.xlsx"
SAMPLES_REL = f"si/si_data/{SAMPLES_FILENAME}"
SAMPLES_SHEET = "Metadata"
SAMPLES_SHA256 = "9477066f538248c18b256ed2fec811bbd5914b2039bd2f78baa41000e70eaf9c"
DMSO_REPLICATES_FILENAME = "13059_2016_900_MOESM6_ESM.xlsx"
DMSO_REPLICATES_REL = f"si/si_data/{DMSO_REPLICATES_FILENAME}"
DMSO_REPLICATES_SHEET = "DMSO_replicate_data"
DMSO_REPLICATES_SHA256 = (
    "7f70c60de3702e49caa5142b73b32b2de64ad1cbfd4db47e3a58a76b616ae676"
)
FLUCONAZOLE_REPLICATES_FILENAME = "13059_2016_900_MOESM7_ESM.xlsx"
FLUCONAZOLE_REPLICATES_REL = f"si/si_data/{FLUCONAZOLE_REPLICATES_FILENAME}"
FLUCONAZOLE_REPLICATES_SHEET = "Fluconazole_replicate_data"
FLUCONAZOLE_REPLICATES_SHA256 = (
    "3d26fb3e6e613e21e427abeaa7a8cc6a9053b34d4770ecc185db7900053a592b"
)
REPLICATES_RETRIEVED_AT = "2026-11-25"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "346be6968eced82706cffb163a76dfc5adf381e602bd11006fea608436ca7b2f"

#: Drug label of the vehicle-only control condition (its own served compound).
DMSO = "DMSO"


def _paper(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """Bind a value to a verbatim quote in the sha256-pinned paper OCR mirror."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
            page="Methods, 'Growth assays' / 'Pooled competitive growth assays' / "
            "'ATc-induced fold change'",
        ),
    )


UNIDENTIFIED_COMPOUND_RULE = (
    "compound_has_no_released_name_or_structure_not_representable"
)

UNIDENTIFIED_COMPOUND_POLICY = (
    "the Drug label is a ChemDiv / ChemBridge / TimTec catalog code the pinned compound "
    "identity table marks PROPRIETARY: Additional file 8 releases only the code and its "
    "vendor, no name, structure, SMILES or CAS, so what went into the flask cannot be "
    "stated and the experiment cannot be represented (database constitution principle 2, "
    "owner ruling 2026-10-15). Its rows are HELD OUT, never stored under the bare code "
    "with an inchikey gap; each code is listed with its doses and rows, and is admitted "
    "the day a public identity enters RELEASED_IDENTITIES"
)

#: The ADMISSION TABLE, keyed by the released Drug label: EMPTY today. A drug the
#: identity table marks PROPRIETARY is stored iff it has an entry here. Only a public
#: code-to-structure release (the vendor's or a primary's), mirrored and sha256-pinned,
#: fills it; an identity inferred from chemical-genetic profile similarity is a hypothesis
#: and never goes here. A filled entry changes the stored records, so it rebuilds the
#: store. The ledger's ``unidentified_compounds_held_out`` is the request list.
RELEASED_IDENTITIES: dict[str, ReleasedIdentity] = {}

_POOLED_QUOTE = (
    "Briefly, $7 0 0 ~ \\mu \\mathrm { L }$ yeast cultures were grown $+ / -$ a drug "
    "listed in Additional file 8, and $+ / -$ ATc) in 48 well plates at $3 0 ~ ^ { \\circ "
    "} \\mathrm { C }$ with orbital shaking in Infinite plate readers (Tecan)."
)

TEMPERATURE_C = _paper(30.0, _POOLED_QUOTE)
AEROBICITY = _paper(
    "aerobic",
    _POOLED_QUOTE,
    note="48-well plates with orbital shaking in a plate reader; no anaerobic handling "
    "is described",
)
DURATION_GENERATIONS = _paper(
    20.0,
    "This amounted to approximately 20 culture doublings from the beginning of the "
    "experiment.",
    note="a competitive-growth screen doses exposure in doublings, not hours; the "
    "cultures were held in log phase by repeated dilution, so no wall time is reported "
    "and duration_hours is a typed ProvenanceGap",
)
ASSAY = _paper(
    AssayType.pooled_competitive_growth_barcode,
    "the short specificity-determining regions of gRNAs (that is, the sequence "
    "complementary to the target) can act as unique identifiers of individual strains",
)
EFFECTOR = _paper(
    "dCas9-Mxi1",
    "catalytically inactive Streptococcus pyogenes Cas9 (dCas9) to which the Mxi1 "
    "transcriptional repressor was fused at the C-terminus",
)
MEDIUM = _paper(
    "SCM-Ura",
    "all colonies were washed off plates with SCM-Ura liquid media",
    note="served as the shared media.SC_URA object (synthetic complete minus uracil, "
    "URA3 plasmid selection); Smith releases no SCM-Ura recipe, so SC_URA's own "
    "component gaps are the honest state",
)
DMSO_PERCENT = _paper(
    1.0,
    "For the control condition $( 1 ~ \\% ~ \\mathrm { D M S O } )$ , we had eight "
    "replicate experiments for the tiling pools, and three replicate experiments for $2 "
    "0 ~ \\mu \\mathrm { M }$ fluconazole.",
    note="the released Concentration cell for the DMSO control is the v/v fraction 0.01; "
    "the Methods give it as 1%",
)
REPLICATE_STRUCTURE = _paper(
    {"1% DMSO": 8, "20 uM fluconazole": 3},
    "We combined the $R$ replicates $k _ { I } , . . . , k _ { R }$ in a natural way to "
    "obtain the variance",
    note="the replicates are released one by one in Additional files 6 (eight, 1% DMSO) "
    "and 7 (three, 20 uM fluconazole; two in the mutant pools), and each is its own "
    "record. Additional file 10's row for these conditions is replicate 1, not the "
    "combination (measured 2026.11.24: its counts equal the Rep-1 counts on every row "
    "both files release), so its var(A) is one replicate's",
)
REPLICATE_DOSE = _paper(
    {DMSO: 0.01, "Fluconazole": "20uM"},
    "For the control condition $( 1 ~ \\% ~ \\mathrm { D M S O } )$ , we had eight "
    "replicate experiments for the tiling pools, and three replicate experiments for $2 "
    "0 ~ \\mu \\mathrm { M }$ fluconazole.",
    note="the dose of each per-replicate workbook's condition, written as the released "
    "Concentration cell of Additional file 10 so one condition has one environment: "
    'Additional file 6 is the 1% DMSO control (\'Counts and analysis for 8 "no drug" '
    "replicates') and Additional file 7 the 20 uM fluconazole one ('Analysis of "
    "replicates  in 20 µM Fluconazole '); neither workbook carries a concentration "
    "column",
)
BIOLOGICAL_REPLICATES = _paper(
    "biological",
    "The guide counts following competitive growth were highly reproducible between "
    "biological replicates, indicating that the assay is robust (Additional file 2: "
    "Figure S3A-B, Additional files 6 and 7).",
    note="Additional files 6 and 7 are the per-replicate tables this loader stores, so "
    "each replicate there is an independent pooled culture pair, not a repeat read",
)
ATC_COMPOUND = _paper(
    "anhydrotetracycline",
    "ATc: Anhydrotetracycline",
    note="the dose is read at build time from Additional file 5's 'ATC (250ng/ml)' "
    "column header (sheet 'Metadata', sha256 9477066f538248c1...), the sample sheet of "
    "every sequenced pooled culture",
)
CULTURE_FITNESS = _paper(
    "log2 median-centred guide count",
    "we calculated the $\\log _ { 2 }$ -scale median centered counts $l _ { i j k } = "
    "\\log _ { 2 } ( c _ { i j k } + 0 . 5 )$ - mediani’ in pool $_ j ( \\log _ { 2 } ( "
    "c _ { i ^ { \\prime } j k } + 0 . 5 ) )$ ).",
    note="Additional files 6 / 7 release l as 'Rep-k_<arm>_f_mean' ('Replicate k no ATc "
    "mean fitness f') with its resampling variance 'Rep-k_<arm>_f_var'",
)
COMPOUND_LIST = _paper(
    "Additional file 8",
    "All chemical compounds used in this study are listed in Additional file 8.",
    note="Additional file 8 (13059_2016_900_MOESM8_ESM.xlsx, sha256 a22ef00ace51...) "
    "gives each compound as 'Name of compound (alias in parentheses)' with its 'Vendor'; "
    "for the ten catalog codes the name IS the code (ChemDiv, Chembridge, TimTec) and no "
    "structure column exists, so the paper releases no identity for them",
)
UPSTREAM_FILTER = _paper(
    "30 reads in the -ATc control",
    "gRNAs with fewer than 30 reads following growth in the minus ATc control condition "
    "were excluded from this analysis, as their effect size estimates had large variance "
    "across conditions.",
    note="applied by the primary before release; this loader adds no count filter",
)
DURATION_HOURS_GAP = ProvenanceGap(
    field="duration_hours",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="the pooled cultures were maintained in log phase by repeated dilution over "
    "approximately 20 doublings; no wall-clock exposure time is reported, so the "
    "duration is carried in generations only",
)

UNITS = (
    "ATc-induced fold change A = f(+ATc) - f(-ATc): the difference in log2 "
    "median-centred guide read-count fitness between the induced (+ATc, dCas9-Mxi1 "
    "CRISPRi ON) and uninduced (-ATc) pooled cultures in this drug condition; negative = "
    "CRISPRi repression of the target causes a growth/fitness defect (drug-sensitivity)"
)

CULTURE_UNITS = (
    "guide fitness f in ONE pooled culture: log2(c + 0.5) minus the median over the "
    "pool's guides of log2(c' + 0.5), c = the guide's perfect-match read count after "
    "approximately 20 doublings; 0 = the pool's median guide. The culture is induced "
    "(+ATc, dCas9-Mxi1 CRISPRi ON) when its environment carries anhydrotetracycline and "
    "uninduced (-ATc) otherwise; A = f(+ATc) - f(-ATc) of one replicate is a "
    "model-time derivation over the two records of one replicate"
)

#: Additional file 10 ``#Pool`` -> the ``Pool Name`` Additional file 5 lists the
#: sequenced sample under. The 18 bp and 20 bp halves of one library were sequenced as
#: one sample, so two pools share one sample row and one replicate label.
POOL_TO_SAMPLE_POOL = {
    "broad_tiling": "broad_tiling",
    "gene_tiling_18bp": "gene_tiling18 & gene_tiling20",
    "gene_tiling_20bp": "gene_tiling18 & gene_tiling20",
    "mutant18": "mutant18 & mutant20",
    "mutant20": "mutant18 & mutant20",
}

#: The per-replicate workbook each multi-replicate condition's cultures come from, keyed
#: by the Additional file 10 ``Drug`` label: (file, sheet, how many replicate blocks the
#: sheet carries).
REPLICATE_WORKBOOKS = {
    DMSO: (DMSO_REPLICATES_FILENAME, DMSO_REPLICATES_SHEET, 8),
    "Fluconazole": (FLUCONAZOLE_REPLICATES_FILENAME, FLUCONAZOLE_REPLICATES_SHEET, 3),
}

#: The pin of each per-replicate workbook, by file name (recorded in the ledger).
REPLICATE_SHA256S = {
    DMSO_REPLICATES_FILENAME: DMSO_REPLICATES_SHA256,
    FLUCONAZOLE_REPLICATES_FILENAME: FLUCONAZOLE_REPLICATES_SHA256,
}

#: The replicate label every Additional file 10 row carries (proved on every build by
#: ``_check_effect_rows_are_replicate_one``).
FIRST_REPLICATE = "1"

#: A released replicate culture this loader stores no record for, and why. Measured on
#: the pinned Additional files 5 and 9 (2026.11.24); the vendor-code conditions are held
#: out by ``UNIDENTIFIED_COMPOUND_RULE`` for every replicate, so they are not listed.
REPLICATE_NOT_STORED_REASONS = {
    ("mutant18 & mutant20", DMSO, "1.0percent_v/v", "2"): (
        "released only as read counts (Additional file 9's Count_all / Count_perfect): "
        "no MeasurementType holds a read count, and computing this culture's fitness f "
        "from the counts would be a loader derivation, not a released value. Additional "
        "file 6 covers the eight 1% DMSO replicates of the gene-tiling pools only"
    ),
    ("gene_tiling18 & gene_tiling20", "Rapamycin", "10.0nM", "1"): (
        "Additional file 5 lists the sequenced sample pair, and no released file carries "
        "a count, a fitness or an effect for it: Additional file 9 holds no 10 nM "
        "Rapamycin rows and Additional file 10 releases only the 2.2 nM and 4.375 nM "
        "doses"
    ),
}

_CONC_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*(M|mM|uM|nM)\s*$")
_UNIT_MAP = {
    "M": ConcentrationUnit.molar,
    "mM": ConcentrationUnit.millimolar,
    "uM": ConcentrationUnit.micromolar,
    "nM": ConcentrationUnit.nanomolar,
}


def drug_is_identified(drug: str) -> bool:
    """Whether a Drug label names identified material (``UNIDENTIFIED_COMPOUND_RULE``).

    Identified unless the pinned identity table marks the label PROPRIETARY (a vendor
    catalog code with no released structure) and ``RELEASED_IDENTITIES`` has no entry
    for it. A public name the table has not resolved yet is identified.
    """
    if drug in RELEASED_IDENTITIES:
        return True
    status = resolve_compound_identity(name=drug).status
    return status is not CompoundResolutionStatus.PROPRIETARY


def released_compound(drug: str) -> Compound:
    """The drug's ``Compound``: the resolver, or an admission-table entry's identity."""
    released = RELEASED_IDENTITIES.get(drug)
    if released is None:
        return resolved_compound(drug)
    compound = resolved_compound(
        released.name or drug, smiles=released.smiles, derive_from_smiles=True
    )
    if released.inchikey is not None and compound.inchikey is None:
        compound = compound.model_copy(
            update={
                "inchikey": released.inchikey,
                "provenance_gaps": [
                    gap for gap in compound.provenance_gaps if gap.field != "inchikey"
                ],
            }
        )
    return compound


def parse_concentration(drug: str, raw: Any) -> Concentration:
    """Parse the released ``Concentration`` cell into a typed dose (fail loud, never guess)."""
    if drug == DMSO:
        return Concentration(
            value=DMSO_PERCENT.value, unit=ConcentrationUnit.percent_v_v
        )
    match = _CONC_RE.match(str(raw))
    if match is None:
        raise ValueError(f"unparseable concentration {raw!r} for drug {drug!r}")
    return Concentration(value=float(match.group(1)), unit=_UNIT_MAP[match.group(2)])


_ATC_HEADER_RE = re.compile(r"\(\s*(?P<value>[0-9]*\.?[0-9]+)\s*ng/m[lL]\s*\)")


def dose_key(drug: str, raw: Any) -> str:
    """The TYPED dose of a condition, as a string, so two spellings key as one.

    Additional file 5 writes a dose with the trailing space Additional file 10 omits
    (``'20uM '`` against ``'20uM'``), so a condition is matched between the two files by
    the ``Concentration`` the cell parses to, never by the cell's characters.
    """
    dose = parse_concentration(drug, raw)
    if dose.unit is None:
        raise ValueError(f"the parsed dose {raw!r} of {drug!r} carries no unit")
    return f"{dose.value}{dose.unit.value}"


def atc_column(columns: list[str]) -> str:
    """Additional file 5's induction column, the one whose header states the ATc dose."""
    named = [name for name in columns if _ATC_HEADER_RE.search(name)]
    if len(named) != 1:
        raise ValueError(
            "Additional file 5 must carry exactly one ATc-dose column header, "
            f"found {named!r}"
        )
    return named[0]


def atc_concentration(column: str) -> Concentration:
    """The induction dose that column header states, in the schema's ug/mL.

    ``ATC (250ng/ml)`` -> 0.25 ug/mL. The dose is released only inside the header, so it
    is parsed rather than hardcoded: a different pinned file would carry its own number
    or fail loud.
    """
    match = _ATC_HEADER_RE.search(column)
    if match is None:
        raise ValueError(f"no ATc dose in Additional file 5 column header {column!r}")
    return Concentration(
        value=float(match.group("value")) / 1000.0, unit=ConcentrationUnit.ug_per_ml
    )


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror + build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/smithQuantitativeCRISPRInterference2016``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deposit_raw_mirror(
    *,
    effect_path: str | Path,
    guide_path: str | Path,
    samples_path: str | Path,
    dmso_replicates_path: str | Path,
    fluconazole_replicates_path: str | Path,
    retrieved_at: str = SI_RETRIEVED_AT,
    replicates_retrieved_at: str = REPLICATES_RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror from the five retrieved SI workbooks plus its ``manifest.json``.

    Idempotent by sha256. ``static-content.springer.com`` is a directly scriptable CDN, so
    the recorded ``springer_esm`` retrieval re-runs as-is; the effect and guide URLs were
    re-retrieved on 2026-09-12 and the three per-replicate workbooks (Additional files 5,
    6 and 7) on 2026-11-25, each reproducing the pinned hash exactly. Additional file 5's
    bytes are identical to the literature mirror's copy of it.
    """
    root = raw_mirror_dir(data_root)
    files: list[ArtifactRecord] = []
    deposits = (
        (effect_path, EFFECT_REL, EFFECT_FILENAME, EFFECT_SHA256, retrieved_at),
        (guide_path, GUIDE_REL, GUIDE_FILENAME, GUIDE_SHA256, retrieved_at),
        (
            samples_path,
            SAMPLES_REL,
            SAMPLES_FILENAME,
            SAMPLES_SHA256,
            replicates_retrieved_at,
        ),
        (
            dmso_replicates_path,
            DMSO_REPLICATES_REL,
            DMSO_REPLICATES_FILENAME,
            DMSO_REPLICATES_SHA256,
            replicates_retrieved_at,
        ),
        (
            fluconazole_replicates_path,
            FLUCONAZOLE_REPLICATES_REL,
            FLUCONAZOLE_REPLICATES_FILENAME,
            FLUCONAZOLE_REPLICATES_SHA256,
            replicates_retrieved_at,
        ),
    )
    # Every workbook is verified before anything is written, so a refusal leaves no
    # mirror directory and no partial deposit behind.
    for source, _, _, expected, _ in deposits:
        verify_sha256(source, expected)
    for source, relpath, filename, expected, retrieved in deposits:
        dest = root / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != expected:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(source, dest)
        url = f"{_ESM_BASE}{filename}"
        files.append(
            ArtifactRecord(
                path=relpath,
                role=ROLE_SI_DATA,
                bytes=dest.stat().st_size,
                sha256=expected,
                source=url,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.springer_esm,
                    source_url=url,
                    retriever="torchcell.literature.retrieve.springer_esm",
                    params={"url": url},
                    sha256=expected,
                    retrieved_at=retrieved,
                ),
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=(
            "Quantitative CRISPR interference screens in yeast identify chemical-genetic "
            "interactions and new rules for guide RNA design"
        ),
        files=files,
        si_data_sources=[
            f"{_ESM_BASE}{EFFECT_FILENAME}",
            f"{_ESM_BASE}{GUIDE_FILENAME}",
            f"{_ESM_BASE}{SAMPLES_FILENAME}",
            f"{_ESM_BASE}{DMSO_REPLICATES_FILENAME}",
            f"{_ESM_BASE}{FLUCONAZOLE_REPLICATES_FILENAME}",
        ],
        si_expected=[
            "Additional file 10 (ATc effects A, drug effects D, no-drug ATc effects A0 "
            "for each gRNA in every tested condition)",
            "Additional file 4 (the five gRNA libraries and their Specificity_sequence "
            "spacers)",
            "Additional file 5 (the sequenced samples: pool, drug, dose, replicate "
            "label and whether ATc was in the culture, with its dose in the header)",
            "Additional file 6 (counts and analysis for the eight 'no drug' 1% DMSO "
            "replicates of the gene-tiling pools)",
            "Additional file 7 (counts and analysis of the 20 uM fluconazole "
            "replicates: three for the gene-tiling pools, two for the mutant pools)",
        ],
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
# Retention bookkeeping
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One retention rule, the records it removed, and the items it removed them for."""

    rule: str
    scope: Literal["compound", "guide", "row"]
    description: str
    n_records: int
    items: list[str] = []


class GappedCompound(BaseModel):
    """A drug stored under its released label with a typed identity gap; not a drop."""

    drug: str
    compound_name: str
    gap_field: str
    gap_reason: ProvenanceGapReason
    n_records: int


class HeldOutDrug(BaseModel):
    """One drug held out under ``UNIDENTIFIED_COMPOUND_RULE``, with its doses and rows."""

    drug: str
    concentrations: list[str]
    n_records: int


class ReplicateNotStored(BaseModel):
    """One released replicate culture that reaches no record, and why (#1178).

    Not a drop rule: the rules above are about Additional file 10's rows. This is the
    accounting of the replicate axis itself, so a replicate the source releases and this
    loader cannot represent is named rather than left silent.
    """

    sample_pool: str
    drug: str
    concentration: str
    replicate: str
    reason: str


class StoredReplicates(BaseModel):
    """What the replicate axis produced, per source file (#1178)."""

    source: str
    sheet: str
    sha256: str
    n_guides: int
    n_replicates: int
    n_records: int
    records_per_replicate: dict[str, int]


class DropLog(BaseModel):
    """Every retention rule applied to a build, in order, plus the gapped compounds.

    ``compound_identity_gaps`` are not drops: those rows are stored, each identified
    compound carrying its typed gap. ``unidentified_compounds_held_out`` are the rows of
    the ``UNIDENTIFIED_COMPOUND_RULE`` rule, per drug, with the quote that grounds it.

    ``source_records``, ``kept_records``, ``dropped_records`` and ``rules`` account for
    Additional file 10's rows, one A record each (#1178). ``culture_replicates`` accounts
    for the per-replicate culture records of Additional files 6 and 7, and
    ``stored_records`` is the store's length: ``kept_records`` plus every file's
    ``n_records``.
    """

    dataset: str
    source_records: int
    kept_records: int
    dropped_records: int
    rules: list[DropRule]
    compound_identity_gaps: list[GappedCompound] = []
    unidentified_compounds_held_out: list[HeldOutDrug] = []
    unidentified_compound_sources: list[SourcedValue] = []
    culture_replicates: list[StoredReplicates] = []
    replicates_not_stored: list[ReplicateNotStored] = []
    stored_records: int = 0


@register_dataset
class CrispriChemgenSmith2016Dataset(ExperimentDataset):
    """Smith 2016 per-guide dCas9-Mxi1 CRISPRi chemical-genetic env x geno dataset."""

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/crispri_chemgen_smith2016",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (no genome: every target is stored as released)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return EnvironmentResponseExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return EnvironmentResponseExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The five sha256-pinned SI workbooks (effects, guides, samples, replicates)."""
        return [
            EFFECT_FILENAME,
            GUIDE_FILENAME,
            SAMPLES_FILENAME,
            DMSO_REPLICATES_FILENAME,
            FLUCONAZOLE_REPLICATES_FILENAME,
        ]

    def download(self) -> None:
        """Link the mirror files into ``raw/`` after verifying each against its pin.

        ``EFFECT_SHA256`` and its four siblings are the pins; the manifest is the
        retrieval record and must carry the same digests, else
        ``ManifestPinMismatchError`` names both.
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        for relpath, filename, expected in (
            (EFFECT_REL, EFFECT_FILENAME, EFFECT_SHA256),
            (GUIDE_REL, GUIDE_FILENAME, GUIDE_SHA256),
            (SAMPLES_REL, SAMPLES_FILENAME, SAMPLES_SHA256),
            (DMSO_REPLICATES_REL, DMSO_REPLICATES_FILENAME, DMSO_REPLICATES_SHA256),
            (
                FLUCONAZOLE_REPLICATES_REL,
                FLUCONAZOLE_REPLICATES_FILENAME,
                FLUCONAZOLE_REPLICATES_SHA256,
            ),
        ):
            check_manifest_pin(relpath, manifest_sha256(manifest, relpath), expected)
            src = raw_mirror_dir(data_root) / relpath
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            link_verified(src, osp.join(self.raw_dir, filename), expected)
        log.info(
            "Smith 2016 SI workbooks linked into %s (sha256 verified)", self.raw_dir
        )

    @staticmethod
    def _released_target(orf_cell: Any, gene_cell: Any) -> tuple[str, str, bool]:
        """``(systematic_gene_name, perturbed_gene_name, placed)`` from the row's cells.

        The ``ORF`` cell (stripped) is the systematic name and the ``Gene`` cell the
        perturbed name, the ORF again where it is blank. No genome is consulted.
        ``placed`` is False for a cell that is not ORF-shaped, which goes on the
        released-identifier leaf (#1045; 0 of the 20 released targets).
        """
        orf = str(orf_cell).strip()
        gene = "" if pd.isna(gene_cell) else str(gene_cell).strip()
        return orf, gene if gene else orf, bool(_SYSTEMATIC_RE.match(orf))

    def _spacer_map(self) -> dict[str, str]:
        """Guide name -> ``Specificity_sequence`` spacer (unique per name; 0 conflicts)."""
        guides = pd.read_excel(
            osp.join(self.raw_dir, GUIDE_FILENAME), sheet_name=GUIDE_SHEET
        )
        return {
            str(name): str(spacer)
            for name, spacer in zip(
                guides["Guide_name"], guides["Specificity_sequence"], strict=True
            )
        }

    @staticmethod
    def _target_perturbation(
        orf: str, gene: str, construct: CrisprConstruct, placed: bool
    ) -> CrisprInterferencePerturbation | ReleasedIdentifierPerturbation:
        """The genotype leaf of one guide: CRISPRi, or the released-identifier leaf."""
        if placed:
            return CrisprInterferencePerturbation(
                systematic_gene_name=orf, perturbed_gene_name=gene, crispr=construct
            )
        return ReleasedIdentifierPerturbation(
            systematic_gene_name=orf,
            gene_namespace="scerevisiae_s288c_sgd_systematic",
            perturbed_gene_name=gene,
            identifier_kind="gene identifier of the guide table, as released",
            released_kind=ReleasedKind.crispr_interference,
            crispr=construct,
        )

    def _target_map(self) -> dict[str, tuple[str, str]]:
        """Guide name -> its released ``(ORF, Gene_name)`` in Additional file 4.

        The per-replicate workbooks carry only a pool and a guide name, so the target
        comes from the guide design table, as released. Measured on the pinned files: a
        guide name repeated there (63 of them, a broad-tiling guide also in a gene-tiling
        set) agrees on ORF, gene and spacer in every case, and where Additional file 10
        also names the guide's target the two files agree on every row.
        """
        guides = pd.read_excel(
            osp.join(self.raw_dir, GUIDE_FILENAME), sheet_name=GUIDE_SHEET
        )
        targets: dict[str, tuple[str, str]] = {}
        for name, orf, gene in zip(
            guides["Guide_name"], guides["ORF"], guides["Gene_name"], strict=True
        ):
            released = (str(orf).strip(), "" if pd.isna(gene) else str(gene).strip())
            held = targets.setdefault(str(name), released)
            if held != released:
                raise ValueError(
                    f"Additional file 4 gives guide {name!r} two targets: "
                    f"{held!r} and {released!r}"
                )
        return targets

    def _environment(
        self, drug: str, conc_raw: Any, *, atc: Concentration | None = None
    ) -> Environment:
        """SC-Ura liquid carrying the row's drug at its released dose, 30 C, 20 doublings.

        ``atc`` is the induction dose of ONE culture (a culture record); an A record
        contrasts the induced and uninduced cultures, so it carries none.
        """
        perturbations: list[Any] = [
            SmallMoleculePerturbation(
                compound=released_compound(drug),
                concentration=parse_concentration(drug, conc_raw),
            )
        ]
        if atc is not None:
            perturbations.append(
                SmallMoleculePerturbation(
                    compound=resolved_compound(ATC_COMPOUND.value), concentration=atc
                )
            )
        return Environment(
            media=SC_URA,
            temperature=Temperature(value=TEMPERATURE_C.value),
            perturbations=perturbations,
            aerobicity=AEROBICITY.value,
            duration_generations=DURATION_GENERATIONS.value,
            provenance_gaps=[DURATION_HOURS_GAP],
        )

    def _reference(
        self, environment: Environment, units: str = UNITS
    ) -> EnvironmentResponseExperimentReference:
        """The readout's own zero in this environment, over the control replicates.

        For an A record the uninduced (-ATc) baseline: no ATc-induced fold change, A = 0.
        For a culture record the pool's median guide in that same culture, f = 0 by the
        median centring. ``replicate_id`` stays None: a reference is the aggregate
        baseline of the condition, not one replicate of it.
        """
        return EnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae",
                strain="BY4741",
                assembly_set=SGD_S288C_R64,
                assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
            ),
            environment_reference=environment.model_copy(),
            phenotype_reference=EnvironmentResponsePhenotype(
                measurement_type=MeasurementType.log2_ratio,
                assay_type=ASSAY.value,
                environment_response=0.0,
                units=units,
            ),
        )

    def _samples(
        self,
    ) -> tuple[Concentration, dict[tuple[str, str, str], list[int]], dict[str, str]]:
        """Additional file 5: the induction dose, the replicate inventory, the spellings.

        The replicate labels are the source's own (``Replicate``, "Replicate
        Identifier"), keyed by the sample's pool name, its drug and its TYPED dose
        (:func:`dose_key`). The third return is that dose key's released spelling, for
        the ledger.
        """
        samples = pd.read_excel(
            osp.join(self.raw_dir, SAMPLES_FILENAME), sheet_name=SAMPLES_SHEET
        )
        column = atc_column([str(name) for name in samples.columns])
        inventory: dict[tuple[str, str, str], set[int]] = {}
        spellings: dict[str, str] = {}
        for pool, drug, dose, replicate in zip(
            samples["Pool Name"],
            samples["Drug"],
            samples["Drug concentration"],
            samples["Replicate"],
            strict=True,
        ):
            named = str(drug).strip()
            typed = dose_key(named, str(dose).strip())
            spellings[typed] = str(dose).strip()
            key = (str(pool).strip(), named, typed)
            inventory.setdefault(key, set()).add(int(replicate))
        return (
            atc_concentration(column),
            {key: sorted(labels) for key, labels in inventory.items()},
            spellings,
        )

    def _replicate_frame(self, filename: str, sheet: str, n_replicates: int) -> Any:
        """One per-replicate workbook, every statistic column coerced to a number.

        The blank cells of a guide the condition released no data for are the literal
        string ``nan`` in these workbooks, so every column but the pool and the guide is
        coerced; a cell where the count, the fitness and its variance disagree about
        being present is refused rather than patched.
        """
        frame = pd.read_excel(osp.join(self.raw_dir, filename), sheet_name=sheet)
        for replicate in range(1, n_replicates + 1):
            for arm in ("noATC", "plusATC"):
                names = [
                    f"Rep-{replicate}_{arm}_{field}"
                    for field in ("count", "f_mean", "f_var")
                ]
                for name in names:
                    frame[name] = pd.to_numeric(frame[name], errors="coerce")
                present = frame[names].notna()
                partial = present.any(axis=1) & ~present.all(axis=1)
                if bool(partial.any()):
                    guides = frame.loc[partial, "Guide"].tolist()
                    raise ValueError(
                        f"{filename} replicate {replicate} {arm} releases a partial "
                        f"measurement for {guides!r}"
                    )
        return frame

    @staticmethod
    def _check_effect_rows_are_replicate_one(
        df: Any,
        inventory: dict[tuple[str, str, str], list[int]],
        replicate_frames: dict[str, Any],
    ) -> None:
        """Refuse unless every Additional file 10 row is replicate 1 of its condition.

        Two checks, both over the pinned bytes, because ``replicate_id = "1"`` on an A
        record is an assertion about which culture pair it came from:

        1. Additional file 5 lists the (pool, drug, dose) sample and its replicate labels
           start at 1, so a condition whose first released replicate is not 1 is refused.
        2. where a per-replicate workbook also covers that (pool, guide), Additional file
           10's ``-ATC_count`` and ``+ATC_count`` equal its ``Rep-1`` counts. Measured
           2026.11.24 on the pinned files: 440 of 440 DMSO rows and 634 of 646
           fluconazole rows (the other 12 have no Rep-1 cells at all), 0 mismatches.
        """
        for keys, _ in df.groupby(["#Pool", "Drug", "Concentration"]):
            pool, drug, dose = (str(value).strip() for value in keys)
            if pool not in POOL_TO_SAMPLE_POOL:
                raise ValueError(f"Additional file 10 names an unknown pool {pool!r}")
            key = (POOL_TO_SAMPLE_POOL[pool], drug, dose_key(drug, dose))
            labels = inventory.get(key)
            if labels is None:
                raise ValueError(f"Additional file 5 lists no sample for {key!r}")
            if labels[0] != 1:
                raise ValueError(
                    f"Additional file 5's first replicate of {key!r} is {labels[0]}, "
                    "so Additional file 10's row is not replicate 1"
                )
        for drug, frame in replicate_frames.items():
            counts = frame.set_index(["#Pool", "Guide"])[
                ["Rep-1_noATC_count", "Rep-1_plusATC_count"]
            ]
            rows = df[df["Drug"].map(lambda value: str(value).strip()) == drug]
            for _, row in rows.iterrows():
                index = (str(row["#Pool"]), str(row["Guide"]))
                if index not in counts.index:
                    continue
                first = counts.loc[index]
                for released, pinned in (
                    (row["-ATC_count"], first["Rep-1_noATC_count"]),
                    (row["+ATC_count"], first["Rep-1_plusATC_count"]),
                ):
                    if pd.isna(pinned):
                        continue
                    if float(released) != float(pinned):
                        raise ValueError(
                            f"Additional file 10's {drug} row for {index!r} carries "
                            f"count {released} where replicate 1 carries {pinned}, so "
                            "the row is not replicate 1"
                        )

    @post_process
    def process(self) -> None:
        """Build one record per released replicate measurement; write the LMDB (#1178).

        An A record per Additional file 10 row (replicate 1 of its condition) and a
        culture record per replicate culture Additional files 6 and 7 release.
        """
        verify_raw_files(
            self.raw_dir,
            {
                EFFECT_FILENAME: EFFECT_SHA256,
                GUIDE_FILENAME: GUIDE_SHA256,
                SAMPLES_FILENAME: SAMPLES_SHA256,
                DMSO_REPLICATES_FILENAME: DMSO_REPLICATES_SHA256,
                FLUCONAZOLE_REPLICATES_FILENAME: FLUCONAZOLE_REPLICATES_SHA256,
            },
        )
        df = pd.read_excel(
            osp.join(self.raw_dir, EFFECT_FILENAME), sheet_name=EFFECT_SHEET
        )
        df["A"] = pd.to_numeric(df["A"], errors="coerce")
        df["var(A)"] = pd.to_numeric(df["var(A)"], errors="coerce")
        spacer_map = self._spacer_map()
        target_map = self._target_map()
        atc_dose, inventory, dose_spellings = self._samples()
        replicate_frames = {
            drug: self._replicate_frame(filename, sheet, n_replicates)
            for drug, (filename, sheet, n_replicates) in REPLICATE_WORKBOOKS.items()
        }
        self._check_effect_rows_are_replicate_one(df, inventory, replicate_frames)
        publication = Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")

        source_records = len(df)
        gapped_records: dict[str, int] = {}

        env_cache: dict[tuple[str, str], dict[str, Any]] = {}

        def prepared(drug: str, conc_raw: Any) -> dict[str, Any]:
            """The A-record environment of one drug condition, and its A = 0 reference."""
            key = (drug, str(conc_raw))
            if key not in env_cache:
                environment = self._environment(drug, conc_raw)
                env_cache[key] = {
                    "environment": environment,
                    "reference": self._reference(environment),
                }
            return env_cache[key]

        culture_cache: dict[tuple[str, str, bool], dict[str, Any]] = {}

        def prepared_culture(
            drug: str, conc_raw: Any, *, induced: bool
        ) -> dict[str, Any]:
            """One culture's environment (ATc when induced), and its f = 0 reference."""
            key = (drug, str(conc_raw), induced)
            if key not in culture_cache:
                environment = self._environment(
                    drug, conc_raw, atc=atc_dose if induced else None
                )
                culture_cache[key] = {
                    "environment": environment,
                    "reference": self._reference(environment, CULTURE_UNITS),
                }
            return culture_cache[key]

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        n_nan = n_no_spacer = 0
        held_out: dict[str, int] = {}
        held_out_doses: dict[str, set[str]] = {}
        released_targets: set[str] = set()
        no_spacer_guides: set[str] = set()
        stored_replicates: list[StoredReplicates] = []
        stored_samples: set[tuple[str, str, str, str]] = set()
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="smith2016"):
                drug = str(row["Drug"])
                if not drug_is_identified(drug):
                    held_out[drug] = held_out.get(drug, 0) + 1
                    held_out_doses.setdefault(drug, set()).add(
                        str(row["Concentration"])
                    )
                    continue
                effect, variance = row["A"], row["var(A)"]
                if pd.isna(effect) or pd.isna(variance):
                    n_nan += 1
                    continue
                orf, gene, placed = self._released_target(row["ORF"], row["Gene"])
                if not placed:
                    released_targets.add(orf)
                guide_name = str(row["Guide"])
                spacer = spacer_map.get(guide_name)
                if spacer is None:
                    n_no_spacer += 1
                    no_spacer_guides.add(guide_name)
                    continue
                item = prepared(drug, row["Concentration"])
                compound = item["environment"].perturbations[0].compound
                if compound.provenance_gaps:
                    gapped_records[drug] = gapped_records.get(drug, 0) + 1
                pool = str(row["#Pool"])
                construct = CrisprConstruct(
                    effector=EFFECTOR.value,
                    guide_sequence=spacer,
                    n_guides=1,
                    library_pool=pool,
                )
                target = self._target_perturbation(orf, gene, construct, placed)
                stored_samples.add(
                    (
                        POOL_TO_SAMPLE_POOL[pool],
                        drug.strip(),
                        dose_key(drug, str(row["Concentration"]).strip()),
                        FIRST_REPLICATE,
                    )
                )
                experiment = EnvironmentResponseExperiment(
                    dataset_name=self.name,
                    genotype=Genotype(perturbations=[target]),
                    environment=item["environment"],
                    phenotype=EnvironmentResponsePhenotype(
                        measurement_type=MeasurementType.log2_ratio,
                        assay_type=ASSAY.value,
                        environment_response=float(effect),
                        environment_response_uncertainty=float(variance),
                        environment_response_uncertainty_type=UncertaintyType.variance,
                        n_samples=1,
                        sample_unit=SampleUnit.pooled,
                        replicate_id=FIRST_REPLICATE,
                        units=UNITS,
                    ),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(
                        experiment, item["reference"], publication, itxn
                    ),
                )
                idx += 1
            kept_effect_records = idx
            for drug, frame in replicate_frames.items():
                filename, sheet, n_replicates = REPLICATE_WORKBOOKS[drug]
                dose = REPLICATE_DOSE.value[drug]
                per_replicate: dict[str, int] = {}
                for _, row in tqdm(
                    frame.iterrows(), total=len(frame), desc=f"smith2016 {drug}"
                ):
                    pool = str(row["#Pool"])
                    if pool not in POOL_TO_SAMPLE_POOL:
                        raise ValueError(f"{filename} names an unknown pool {pool!r}")
                    guide_name = str(row["Guide"])
                    spacer = spacer_map.get(guide_name)
                    released = target_map.get(guide_name)
                    if spacer is None or released is None:
                        raise ValueError(
                            f"{filename} releases replicates for guide {guide_name!r}, "
                            "which Additional file 4 does not list"
                        )
                    orf, gene, placed = self._released_target(*released)
                    if not placed:
                        released_targets.add(orf)
                    construct = CrisprConstruct(
                        effector=EFFECTOR.value,
                        guide_sequence=spacer,
                        n_guides=1,
                        library_pool=pool,
                    )
                    target = self._target_perturbation(orf, gene, construct, placed)
                    for replicate in range(1, n_replicates + 1):
                        for arm, induced in (("noATC", False), ("plusATC", True)):
                            fitness = row[f"Rep-{replicate}_{arm}_f_mean"]
                            variance = row[f"Rep-{replicate}_{arm}_f_var"]
                            if pd.isna(fitness):
                                continue
                            item = prepared_culture(drug, dose, induced=induced)
                            experiment = EnvironmentResponseExperiment(
                                dataset_name=self.name,
                                genotype=Genotype(perturbations=[target]),
                                environment=item["environment"],
                                phenotype=EnvironmentResponsePhenotype(
                                    measurement_type=MeasurementType.log2_ratio,
                                    assay_type=ASSAY.value,
                                    environment_response=float(fitness),
                                    environment_response_uncertainty=float(variance),
                                    environment_response_uncertainty_type=(
                                        UncertaintyType.variance
                                    ),
                                    n_samples=1,
                                    sample_unit=SampleUnit.pooled,
                                    replicate_id=str(replicate),
                                    units=CULTURE_UNITS,
                                ),
                            )
                            txn.put(
                                f"{idx}".encode(),
                                self._intern_record(
                                    experiment, item["reference"], publication, itxn
                                ),
                            )
                            idx += 1
                            label = str(replicate)
                            per_replicate[label] = per_replicate.get(label, 0) + 1
                            stored_samples.add(
                                (
                                    POOL_TO_SAMPLE_POOL[pool],
                                    drug,
                                    dose_key(drug, dose),
                                    label,
                                )
                            )
                stored_replicates.append(
                    StoredReplicates(
                        source=filename,
                        sheet=sheet,
                        sha256=REPLICATE_SHA256S[filename],
                        n_guides=len(frame),
                        n_replicates=n_replicates,
                        n_records=sum(per_replicate.values()),
                        records_per_replicate=dict(sorted(per_replicate.items())),
                    )
                )
        env.close()
        interned_env.close()

        rules = [
            DropRule(
                rule=UNIDENTIFIED_COMPOUND_RULE,
                scope="compound",
                description=UNIDENTIFIED_COMPOUND_POLICY,
                n_records=sum(held_out.values()),
                items=sorted(held_out),
            ),
            DropRule(
                rule="row_has_no_A_or_var_A",
                scope="row",
                description=(
                    "the released row carries no ATc-induced fold change or no variance "
                    "for it (0 today; every row of Additional file 10 is complete)"
                ),
                n_records=n_nan,
                items=[],
            ),
            DropRule(
                rule="stored_on_released_gene_name_leaf",
                scope="row",
                description=(
                    "the ORF cell is not ORF-shaped, so the target is stored on "
                    "ReleasedIdentifierPerturbation verbatim with its construct (#1045), "
                    "never dropped; listed here (0 today; all 20 released targets are "
                    "ORF-shaped)"
                ),
                n_records=0,
                items=sorted(released_targets),
            ),
            DropRule(
                rule="guide_has_no_released_spacer",
                scope="guide",
                description=(
                    "the guide name is absent from Additional file 4, so no spacer exists "
                    "to carry the strain identity (0 today; all 977 screened guides join)"
                ),
                n_records=n_no_spacer,
                items=sorted(no_spacer_guides),
            ),
        ]
        compound_identity_gaps = []
        for drug, count in sorted(gapped_records.items()):
            compound = (
                env_cache[next(k for k in env_cache if k[0] == drug)]["environment"]
                .perturbations[0]
                .compound
            )
            (gap,) = compound.provenance_gaps
            compound_identity_gaps.append(
                GappedCompound(
                    drug=drug,
                    compound_name=compound.name,
                    gap_field=gap.field,
                    gap_reason=gap.reason,
                    n_records=count,
                )
            )
        # The replicate axis: every (pool, drug, dose, replicate) Additional file 5
        # releases either reached a record or is named with its reason. A condition whose
        # drug is held out under UNIDENTIFIED_COMPOUND_RULE is accounted by that rule for
        # all of its replicates, so it is not listed twice.
        replicates_not_stored: list[ReplicateNotStored] = []
        for (pool, drug, typed), labels in sorted(inventory.items()):
            if not drug_is_identified(drug):
                continue
            for released_label in labels:
                key = (pool, drug, typed, str(released_label))
                if key in stored_samples:
                    continue
                reason = REPLICATE_NOT_STORED_REASONS.get(key)
                if reason is None:
                    raise RuntimeError(
                        f"released replicate {key!r} reached no record and no reason is "
                        "recorded for it"
                    )
                replicates_not_stored.append(
                    ReplicateNotStored(
                        sample_pool=pool,
                        drug=drug,
                        concentration=dose_spellings[typed],
                        replicate=str(released_label),
                        reason=reason,
                    )
                )
        drop_log = DropLog(
            dataset=self.name,
            source_records=source_records,
            kept_records=kept_effect_records,
            dropped_records=source_records - kept_effect_records,
            rules=rules,
            compound_identity_gaps=compound_identity_gaps,
            unidentified_compounds_held_out=[
                HeldOutDrug(
                    drug=drug,
                    concentrations=sorted(held_out_doses[drug]),
                    n_records=count,
                )
                for drug, count in sorted(held_out.items())
            ],
            unidentified_compound_sources=[COMPOUND_LIST],
            culture_replicates=stored_replicates,
            replicates_not_stored=replicates_not_stored,
            stored_records=idx,
        )
        with open(osp.join(self.preprocess_dir, "dropped_records.json"), "w") as handle:
            handle.write(drop_log.model_dump_json(indent=2))
        accounted = sum(rule.n_records for rule in rules)
        if accounted != drop_log.dropped_records:
            raise RuntimeError(
                f"drop accounting mismatch: rules total {accounted}, "
                f"{drop_log.dropped_records} records missing from the build"
            )
        culture_records = sum(entry.n_records for entry in stored_replicates)
        if kept_effect_records + culture_records != idx:
            raise RuntimeError(
                f"record accounting mismatch: {kept_effect_records} A records plus "
                f"{culture_records} culture records is not the {idx} written"
            )
        log.info(
            "Smith2016: wrote %d records -- %d of %d Additional file 10 rows (%d under a "
            "typed compound-identity gap on %d drugs) plus %d replicate cultures; held "
            "out %d rows of %d unidentified vendor-code drugs; %d with no A/var(A), %d "
            "targets on the released-identifier leaf, %d guides with no spacer, %d "
            "released replicates not stored",
            idx,
            kept_effect_records,
            source_records,
            sum(gapped_records.values()),
            len(gapped_records),
            culture_records,
            sum(held_out.values()),
            len(held_out),
            n_nan,
            len(released_targets),
            n_no_spacer,
            len(replicates_not_stored),
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


def main() -> None:
    """Build/load the dataset for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    root = osp.join(data_root, "data/torchcell/crispri_chemgen_smith2016")
    dataset = CrispriChemgenSmith2016Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])
    print(
        json.dumps(
            json.loads(
                Path(osp.join(root, "preprocess/dropped_records.json")).read_text()
            )["rules"],
            indent=2,
        )[:2000]
    )


if __name__ == "__main__":
    main()
