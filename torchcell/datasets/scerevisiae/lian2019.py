# torchcell/datasets/scerevisiae/lian2019
# [[torchcell.datasets.scerevisiae.lian2019]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/lian2019
# Test file: tests/torchcell/datasets/scerevisiae/test_lian2019.py
"""Lian 2019 MAGIC genome-wide CRISPRa/i/d furfural-tolerance screen (per-guide enrichment).

Lian et al. 2019 (Nat Commun 10:5794, doi:10.1038/s41467-019-13621-4; PMID 31857575) built
MAGIC: three genome-scale gRNA libraries driven by three ORTHOGONAL Cas effectors in one
pooled CRISPR-AID host, so a single cell carries exactly one guide of exactly one MODE.

- CRISPRa (activation)   -- ``dLbCas12a-VP``   (37,817 guides, 23 nt spacer)
- CRISPRi (interference) -- ``dSpCas9-RD1152`` (37,870 guides, 20 nt spacer)
- CRISPRd (deletion)     -- ``SaCas9``         (24,806 designs, 21 nt spacer + 100 nt donor)

The unique guide is a genetic barcode; furfural tolerance is mapped by tracking each
guide's enrichment (furfural-selected vs untreated) by NGS. Screening is ITERATIVE in
accumulating integrated host strains: round 1 is screened in bAID (5 mM furfural), round
2 in R1 = bAID-X3::SIZ1i (10 mM), round 3 in R2 = bAID-X3::SIZ1i-X4::NAT1a (15 mM). The
round's accumulated cassettes are the HOST's, constant across every record of that round
and shared with its reference strain, so they ride on the reference's typed
``StrainBackground`` as ``IntegratedCassette`` entries and the record's ``Genotype``
holds the one screened guide. A round-2/3 record is still the strain actually in the
tube: the background states what else it carries, and at which locus.

RECORD = one (guide x round) ``StrainEnvironmentResponseExperiment``:

- GENOTYPE: the library member as a ``CrisprActivation`` / ``CrisprInterference`` /
  ``CrisprDeletion`` perturbation (target gene, this guide's spacer, its effector), and
  nothing else. The common name is the GENOME's own standard name for the resolved ORF,
  so one gene carries one spelling across datasets.
- ENVIRONMENT: a ``CultureEnvironment`` over the shared ``media.SED_URA_G418`` object
  carrying furfural (5/10/15 mM by round) as a ``SmallMoleculePerturbation``, 30 C,
  aerobic, 50 mL in a shaken 250 mL baffled flask.
- REFERENCE GENOME: a ``StrainReferenceGenome`` whose ``StrainBackground`` is the round's
  host (BY4742's four auxotrophies + bAID's Delta-site CRISPR-AID cassette + the round's
  integrated gRNA cassettes), each element sourced or a typed gap.
- PHENOTYPE: ``measurement_type=log2_ratio``,
  ``assay_type=pooled_competitive_growth_barcode``, ``environment_response`` = mean
  log2(after/before) over 3 biological triplicates, uncertainty = SD (``sample_sd``,
  n=3 -> SE = SD/sqrt(3)). Reference = no-enrichment baseline (log2FC 0) in the round's
  host.

THE MEDIUM IS SED-URA/G418, NOT SED/G418 (corrected this build, all records). Methods,
verbatim: "The iMAGIC libraries in triplicates were inoculated into 50 mL SED-URA/G418
medium with or without furfural in a 250 mL baffled flask." The same paragraph shows
SED/G418 is the medium for a DIFFERENT experiment ("SED-URA/G418 (plasmid-bearing strains)
or SED/G418 (integrated strains)"), and the enrichment data come from the plasmid-borne
pooled library. This is not a naming nit: SED/G418 lacks the uracil dropout that selects
the guide plasmid, so the previous build recorded a different selection regime from the
one that produced the data. ``SED_URA_G418``'s components (YNB w/o AA 0.17%, monosodium
L-glutamate 0.1%, CSM-URA 0.077%, glucose 2%, G418 200 ug/mL, dropout uracil) are all
sourced from the Methods sentence quoted in ``media.py``.

THE CRISPRd GUIDE/DONOR SPLIT (corrected this build, 62,793 records). The previous build
stored the 44 nt amplicon barcode -- the FIRST 44 nt of the 121 nt design cassette -- in
``crispr.guide_sequence``, where the schema documents a "~20 nt" spacer, and left
``donor_sequence`` None. The paper states the layout: "the homologous recombination donor
was integrated to the 5'-end of the targeting sequences" and "Homology-directed repair
resulted in the deletion of 28 bp nucleotides in the coding sequences, including both the
targeting sequences and the protospacer adjacent motif sequences". The boundary is
MEASURED, not assumed, against the S288C genome:

- for 193 of 200 sampled designs whose two 50 nt halves both map uniquely, the gap between
  the first arm's end and the second arm's start is exactly 28 bp, and the last 21 nt of
  the design starts exactly at that gap (offset 0);
- for ALL 24,706 non-control designs, the LAST 21 nt is present in the genome with a
  canonical SaCas9 ``NNGRRT`` PAM immediately 3' of it (0 exceptions).

So the design cassette is ``donor (100 nt: two 50 nt homology arms flanking the 28 bp
deletion) + guide spacer (21 nt)``, and this build stores the last 21 nt as
``crispr.guide_sequence`` and everything before it as ``CrisprDeletionPerturbation.
donor_sequence``. The split is read from the PUBLISHED Supplementary Data 3
(``41467_2019_13621_MOESM5_ESM.xlsx``), not from a lab copy: its ``Sequence`` column is
element-for-element identical to the archived design file, and its first 44 nt reproduce
the enrichment table's ``d`` barcode for all 24,806 rows, which is the positional join
this loader asserts at build time.

RECORDS DROPPED (rule + counts in ``preprocess/dropped_records.json``):

- 300 random negative-control guides (100 per library) and 16 guides whose gene name is a
  source-corrupted Excel date artifact present IDENTICALLY in the reference and the design
  library, so it is unrecoverable from our inputs.
- guides whose target gene does not resolve to a current R64 gene (ncRNA/rDNA features
  absent from the ORF genome).
- (guide, round) cells with no enrichment value, and the guide-round in which a guide
  targets its OWN integrated background gene (redundant, and it would collapse to an empty
  strain signature once the background is subtracted).
Nothing else is dropped. In particular, 150 groups of CRISPRd designs (318 records) share
a resolved gene AND a 21 nt spacer, because the spacer maps to more than one site at a
multicopy locus (tRNA genes, paralogs) and the designs differ only in their donor arms.
They are distinct strains and are all KEPT: the verifier's genotype signature reads
``donor_sequence``, which is what tells them apart.

THE INTEGRATION LOCUS IS STATED, and the previous build's docstring was wrong to say it
"is not stated anywhere in the release" (corrected this build). Supplementary Table 11,
"Strains constructed in this study", gives the strain table verbatim:

    BY4742 | MATα his3∆1 leu2∆0 lys2∆0 ura3∆0
    bAID   | BY4742-Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]

So bAID is BY4742 with the CRISPR-AID cassette integrated at the DELTA site (the Ty1
delta repeat family), carrying KanMX plus four orthogonal effector cassettes. The same
table gives the round strains (``R1`` = bAID-X3::SIZ1i, ``R2`` =
bAID-X3::SIZ1i-X4::NAT1a), and the pre-selected landing pads X2-XII5 are named in the
Results. Every one of those is an INTEGRATION at a named site, which no ``AlleleEdit``
describes, so each is a typed ``IntegratedCassette`` on the background:
``locus`` is the site verbatim and ``locus_systematic_gene_name`` stays None, because a
delta repeat family and an intergenic landing pad are not R64 ORFs.

The strain string is the SI's own genotype column (``bAID``,
``bAID-X3::SIZ1i``, ``bAID-X3::SIZ1i-X4::NAT1a``) rather than its terse ``R1`` / ``R2``
label, so the join key says what the strain is. It still does not join the BY4742
datasets, which is correct: these strains are not BY4742, and the background now states
the difference allele by allele and cassette by cassette.

Exposure duration is a typed ``ProvenanceGap`` on both ``duration_hours`` and
``duration_generations``: the pooled cultures were harvested at mid-log ("1 OD of the
mid-log phase growing cells from each of the untreated and stressed libraries were
collected"), and neither a wall time nor a doubling count is reported.

DESIGNED VS REALIZED: every CRISPR perturbation here is a pooled-library DESIGN asserted
as realized. That is a documented deferral (memory
``designed-vs-realized-perturbation-material-entity``), not something this build measured,
and it is unmarked in the record because no certainty axis exists yet.

DATA SOURCE. The furfural per-guide enrichment is NOT in any Nature supplement (only the
design libraries, Supplementary Data 1-3, and the guide reference, Supplementary Data 4,
were released; the Fig 2 profiles are excluded from Source Data). It is reprocessed from
raw reads (NCBI SRA PRJNA504483, 21 runs) against the 100,493-guide reference by the
versioned pipeline in ``experiments/016-lian-magic-reprocess/``, and validated against the
paper's hits (PDR1i round-3 rank 1, SLX5i round-1 rank 1, SAP30d round-1 rank 2). The
derived ``guide_enrichment_final.tsv`` and the published Supplementary Data 3 both live in
the raw mirror ``$DATA_ROOT/torchcell-raw/lianMultifunctionalGenomewideCRISPR2019/`` with a
``manifest.json`` recording the Springer ESM retrieval for the design file and the
reprocessing chain (inputs + pipeline) for the derived table.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import os.path as osp
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
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
    verify_sha256,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import SED_URA_G418
from torchcell.datamodels.schema import (
    AssayType,
    Concentration,
    ConcentrationUnit,
    CrisprActivationPerturbation,
    CrisprConstruct,
    CrisprDeletionPerturbation,
    CrisprInterferencePerturbation,
    CultureEnvironment,
    CultureFormat,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    IntegratedCassette,
    MatingType,
    MeasurementType,
    Publication,
    SampleUnit,
    SmallMoleculePerturbation,
    StrainBackground,
    StrainEnvironmentResponseExperiment,
    StrainEnvironmentResponseExperimentReference,
    StrainReferenceGenome,
    Temperature,
    UncertaintyType,
    Zygosity,
)
from torchcell.datamodels.strain_background import (
    BRACHMANN_1998,
    STANDARD_BY_GENOTYPES,
    standard_allele,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.scerevisiae.smith2006 import canonical_common_names
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ROLE_SI_DATA,
    ArtifactRecord,
    Manifest,
    ProcessingRecord,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DOI = "10.1038/s41467-019-13621-4"
PMID = "31857575"

CITATION_KEY = "lianMultifunctionalGenomewideCRISPR2019"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

TSV_FILENAME = "guide_enrichment_final.tsv"
TSV_REL = f"data/{TSV_FILENAME}"
TSV_SHA256 = "f9af849f97a2d460c3a6d628308491ec3966c6cc2a7f6cad130848d2bad32647"

_ESM_BASE = (
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-13621-4/"
    "MediaObjects/"
)
#: Supplementary Data 3 -- the CRISPRd design library (24,806 rows, 121 nt cassettes).
DESIGN_D_FILENAME = "41467_2019_13621_MOESM5_ESM.xlsx"
DESIGN_D_REL = f"si/si_data/{DESIGN_D_FILENAME}"
DESIGN_D_SHA256 = "737074a76b9eee2dc015be8b17e29b4fbe65c8be5565e6fcbe71505dca4109e2"
#: Supplementary Data 4 -- the 100,493-guide reference the reprocessing mapped against.
REFERENCE_FILENAME = "41467_2019_13621_MOESM6_ESM.xlsx"
REFERENCE_SHA256 = "4e3f225ae0194462252049aae28f57c1428a99000ffdeef9e418e240778435a3"
SI_RETRIEVED_AT = "2026-09-12"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "63fe2b7101fc48feb297f9e34b83d108b74f03f28bbc280e08c7219bc975086c"

#: Supplementary Information 1 as MinerU extracted it from the publisher PDF. It carries
#: Supplementary Table 11 ("Strains constructed in this study"), which is where bAID's
#: integration site and every round strain's genotype are stated.
SI1_MD = "si/si1.md"
SI1_MD_SHA256 = "b2bcfe2e672674438216472e3e06903c93d4ee54cd8b6fd9b5f964ad2a3d32db"

#: The amplicon barcode the reprocessing counted for a CRISPRd guide: read[27:71].
D_BARCODE_LEN = 44
#: SaCas9 spacer length, measured against the genome (see the module docstring).
D_SPACER_LEN = 21


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
            page="Methods, 'iMAGIC screening of furfural tolerance' / 'Design and "
            "construction of the MAGIC libraries' / 'Strains and media'",
        ),
    )


def _si1(value: Any, quote: str, *, page: str, note: str | None = None) -> SourcedValue:
    """Bind a value to a verbatim quote in the sha256-pinned SI 1 OCR mirror."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SI1_MD,
            citation_key=CITATION_KEY,
            sha256=SI1_MD_SHA256,
            method="MinerU OCR of the publisher Supplementary Information PDF "
            "(torchcell-library mirror)",
            page=page,
        ),
    )


_SCREEN_QUOTE = (
    "The iMAGIC libraries in triplicates were inoculated into $5 0 \\mathrm { m L }$ "
    "SED-URA/G418 medium with or without furfural in a $2 5 0 \\mathrm { m L }$ baffled "
    "flask."
)
_CULTIVATION_QUOTE = (
    "Yeast strains were cultivated in complex medium consisting of $2 \\%$ peptone, $1 "
    "\\%$ yeast extract, and $2 \\%$ glucose (YPD) or synthetic complete medium consisting "
    "of $0 . 1 7 \\%$ yeast nitrogen base, $0 . 1 \\%$ mono-sodium glutamate, $0 . 0 7 7 "
    "\\%$ CSM-URA, and $2 \\%$ glucose (SED-URA) at $3 0 ^ { \\circ } \\mathrm { C } ,$ . "
    "$2 5 0 \\mathrm { r p m }$ ."
)

MEDIUM = _paper(
    "SED-URA/G418",
    _SCREEN_QUOTE,
    note="served as the shared media.SED_URA_G418 object. The previous build stored "
    "'SED/G418', which is the medium for the INTEGRATED validation strains in the same "
    "paragraph, not for the plasmid-borne pooled library the enrichment comes from; "
    "SED/G418 lacks the uracil dropout that selects the guide plasmid",
)
TEMPERATURE_C = _paper(30.0, _CULTIVATION_QUOTE)
AEROBICITY = _paper(
    "aerobic",
    _SCREEN_QUOTE,
    note="a shaken baffled flask, the standard aerobic configuration; the validation "
    "cultures of the same screen are explicitly 'cultivated under aerobic conditions'",
)
FURFURAL_MM = _paper(
    {1: 5.0, 2: 10.0, 3: 15.0},
    "5, 10, and $1 5 \\mathrm { m M }$ furfural were used for the first, second, and "
    "third round of iMAGIC screening, respectively.",
)
N_REPLICATES = _paper(
    3,
    "biological triplicates for untreated and furfural stressed libraries",
    note="the triplicate untreated and furfural-stressed libraries; the stored SD is the "
    "sample SD across them, so SE = SD/sqrt(3)",
)
ASSAY = _paper(
    AssayType.pooled_competitive_growth_barcode,
    "The reads of $4 3 \\mathrm { b p }$ between SNR52p and SUP4t that contains a unique "
    "sequence in all three CRISPR-AID libraries (Supplementary Table 12) were extracted "
    "from the NGS data",
    note="a pooled library grown competitively and read out by amplifying each strain's "
    "unique guide sequence as a barcode",
)
HOST = _paper(
    "bAID",
    "The CRISPR-AID strain (bAID) was constructed by integrating PmeI-digested $\\mathrm "
    "{ \\ p A I D } 6 ^ { 8 }$ into the genome of BY4742 and selection for G418 resistance.",
    note="the host's construction sentence: BY4742 plus an integrated pAID6. The "
    "integration SITE is stated in Supplementary Table 11 (BAID_GENOTYPE), so the "
    "cassette is carried as a typed IntegratedCassette on the reference's "
    "StrainBackground",
)
BY4742_GENOTYPE = _si1(
    "MATα his3∆1 leu2∆0 lys2∆0 ura3∆0",
    "<td rowspan=1 colspan=1>BY4742</td><td rowspan=1 colspan=1>MATα his3∆1 leu2∆0 "
    "lys2∆0 ura3∆0</td>",
    page="Supplementary Table 11, 'Strains constructed in this study' (si1.md line 104)",
    note="the parent strain's genotype as the SI states it; the stored alleles are the "
    "STANDARD_ALLELES spellings his3Δ1, leu2Δ0, lys2Δ0, ura3Δ0 (the OCR writes the "
    "delta as the mathematical operator ∆)",
)
BAID_GENOTYPE = _si1(
    "BY4742-Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]",
    "<td rowspan=1 colspan=1>bAID</td><td rowspan=1 colspan=1>BY4742-Delta::KanMX-"
    "[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]</td>",
    page="Supplementary Table 11, 'Strains constructed in this study' (si1.md line 104)",
    note="the integration SITE of the CRISPR-AID cassette, which the previous build's "
    "docstring wrongly called unstated: the SI's strain table writes it as the Delta "
    "site (the Ty1 delta repeat family), carrying KanMX plus the four orthogonal "
    "effector cassettes",
)
ROUND_STRAIN_GENOTYPES = _si1(
    ["bAID", "bAID-X3::SIZ1i", "bAID-X3::SIZ1i-X4::NAT1a"],
    "<td rowspan=1 colspan=1>R1</td><td rowspan=1 colspan=1>bAID-X3::SIZ1i</td></tr>"
    "<tr><td rowspan=1 colspan=1>R2</td><td rowspan=1 colspan=1>"
    "bAID-X3::SIZ1i-X4::NAT1a</td>",
    page="Supplementary Table 11, 'Strains constructed in this study' (si1.md line 104)",
    note="the three round HOSTS in round order (ROUND_STRAIN). The SI names the second "
    "and third R1 and R2: they are the strains obtained AFTER rounds 1 and 2 and are "
    "therefore the hosts of rounds 2 and 3 (ROUND_PARENT). The stored strain string is "
    "the SI's own genotype column, which is self-describing, rather than the terse "
    "'R1' / 'R2' label. A LIST in round order, not a round-keyed dict: this value is "
    "embedded in every record's background provenance, and a dict with integer keys "
    "would come back from JSON with string keys, so the record would not round-trip",
)

#: The strain each round is screened in, by round (``ROUND_STRAIN_GENOTYPES`` indexed).
ROUND_STRAIN: dict[int, str] = {
    rnd: name for rnd, name in enumerate(ROUND_STRAIN_GENOTYPES.value, start=1)
}
ROUND_PARENT = _paper(
    {2: "R1", 3: "R2"},
    "Then we used the NAT1a and SIZ1i-integrated strain (R2) as the parent strain for "
    "the third round of genome-wide screening and continued to observe highly enriched "
    "guide sequences (Fig. 2e).",
    note="round 3 was screened in R2 (SIZ1i + NAT1a). The round-2 host is stated in the "
    "SI figure legends: 'the second round iMAGIC screening identified targets when "
    "integrated into the X4 locus of R1 strain (SIZ1i)' (ROUND2_HOST)",
)
ROUND2_HOST = _si1(
    {"host": "R1", "integration_locus": "X4"},
    "Supplementary Figure 4. Verification of the second round iMAGIC screening "
    "identified targets when integrated into the X4 locus of R1 strain (SIZ1i).",
    page="Supplementary Figure 4 legend (si1.md line 21)",
    note="round 2 was screened in R1, whose only integration is SIZ1i; the X4 locus "
    "named here is where the round-2 HITS were then integrated, which is why NAT1a sits "
    "at X4 in R2",
)
ROUND3_HOST = _si1(
    {"host": "R2", "integration_locus": "XI1"},
    "Supplementary Figure 6. Verification of the third round iMAGIC screening identified "
    "targets when integrated into the XI1 locus of the R2 strain (SIZ1i-NAT1a).",
    page="Supplementary Figure 6 legend (si1.md line 28)",
    note="round 3 was screened in R2 (SIZ1i at X3 + NAT1a at X4); XI1 is where the "
    "round-3 hit PDR1i was then integrated, which is R3 and is not screened here",
)
MARKERLESS_INTEGRATION = _paper(
    "marker-less",
    "The gRNA expression cassettes identified by MAGIC screening were integrated into "
    "the predefined loci (Supplementary Table 1) in a CRISPR-assisted and marker-less "
    "manner.",
    note="the round gRNA cassettes carry NO selection marker, so IntegratedCassette."
    "marker is None for them; bAID's own cassette does carry one (KanMX)",
)
INTEGRATION_LOCI = _paper(
    ["X2", "X3", "X4", "XI1", "XI2", "XI3", "XII1", "XII2", "XII4", "XII5"],
    "Ten gRNA plasmids based on SaCas9 were constructed to integrate heterologous "
    "cassettes into X2, X3, X4, XI1, XI2, XI3, XII1, XII2, XII4, and XII5 loci, "
    "respectively (Supplementary Table 1).",
    note="the pre-selected landing pads, 'flanked by highly expressed essential genes'; "
    "they are intergenic sites, not ORFs, so IntegratedCassette."
    "locus_systematic_gene_name is None for every one of them",
)
DONOR_LAYOUT = _paper(
    {"donor_nt": 100, "spacer_nt": D_SPACER_LEN, "deleted_bp": 28},
    "the homologous recombination donor was integrated to the $5 ^ { \\prime }$ -end of "
    "the targeting sequences9.",
    note="measured against S288C, not assumed: the two 50 nt donor arms flank a gap of "
    "exactly 28 bp in 193/193 resolvable sampled designs, the last 21 nt starts at that "
    "gap, and for all 24,706 non-control designs the last 21 nt sits in the genome with a "
    "canonical SaCas9 NNGRRT PAM immediately 3' (0 exceptions)",
)
DELETION_SIZE = _paper(
    28,
    "Homology-directed repair resulted in the deletion of 28 bp nucleotides in the coding "
    "sequences, including both the targeting sequences and the protospacer adjacent motif "
    "sequences",
)
HARVEST = _paper(
    "mid-log phase",
    "1 OD of the mid-log phase growing cells from each of the untreated and stressed "
    "libraries were collected and the plasmids were extracted for NGS analysis.",
    note="the cultures were harvested at mid-log, and neither a wall time nor a doubling "
    "count is reported, so both duration fields are typed ProvenanceGaps",
)

SCREEN_VESSEL = _paper(
    {"vessel": "250 mL baffled flask", "working_volume_ul": 50000.0},
    _SCREEN_QUOTE,
    note="50 mL of medium in a 250 mL baffled flask, the pooled screening culture these "
    "records come from",
)
SHAKING_RPM = _paper(
    250.0,
    _CULTIVATION_QUOTE,
    note="the cultivation sentence that also gives the temperature: 30 C, 250 rpm. It "
    "is the paper's one statement of agitation, and the screening flasks are baffled, "
    "which is the shaken configuration",
)

_SCREEN_METHODS_PAGE = Provenance(
    source_uri=PAPER_MD,
    citation_key=CITATION_KEY,
    sha256=PAPER_MD_SHA256,
    page="Methods, 'iMAGIC screening of furfural tolerance'",
)

DURATION_GAPS = [
    ProvenanceGap(
        field="duration_hours",
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=_SCREEN_METHODS_PAGE,
        note="the pooled cultures were harvested at mid-log phase; no wall-clock exposure "
        "time is reported for the screening flasks",
    ),
    ProvenanceGap(
        field="duration_generations",
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=_SCREEN_METHODS_PAGE,
        note="no doubling count is reported for the screening flasks either",
    ),
]

ENDPOINT_GAP = ProvenanceGap(
    field="endpoint",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_SCREEN_METHODS_PAGE,
    note="the cultures were read at mid-log phase ('1 OD of the mid-log phase growing "
    "cells ... were collected'), which is neither a fixed duration, nor a fixed number "
    "of generations, nor control saturation. EndpointRule has no member for a "
    "phase-triggered harvest, so the field is a typed gap rather than a chosen rule",
)
INOCULUM_GAP = ProvenanceGap(
    field="inoculum_od600",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_SCREEN_METHODS_PAGE,
    note="the same Methods paragraph gives an initial OD of 0.05, but for the "
    "INDIVIDUALLY constructed validation strains in culture tubes, not for the pooled "
    "library flasks these records come from; the pooled flasks' inoculum is not stated",
)
PRE_CULTURE_GAP = ProvenanceGap(
    field="pre_culture",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_SCREEN_METHODS_PAGE,
    note="the screening paragraph says only that the libraries 'were inoculated into 50 "
    "mL SED-URA/G418 medium' and names no pre-culture. The library-construction "
    "paragraph does say the transformants were 'cultured in 50 mL SED-URA/G418 medium "
    "for ~2 days' before pooling, but no PreCultureSource member describes a ~2 day "
    "culture of unstated phase: filing it as overnight_culture or log_phase_culture "
    "would assert a time or a phase the source does not give",
)
AUXOTROPH_SUPPLEMENT_GAP = ProvenanceGap(
    field="auxotroph_supplements",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=Provenance(
        source_uri=PAPER_MD,
        citation_key=CITATION_KEY,
        sha256=PAPER_MD_SHA256,
        page="Methods, 'Strains, media, and cultivation conditions'",
    ),
    note="the paper names no supplement added BESIDE the medium for the host's four "
    "BY4742 auxotrophies; SED-URA's CSM-URA is a complete supplement mixture minus "
    "uracil, so the complementation rides inside media rather than beside it, and "
    "uracil is deliberately withheld to select the guide plasmid",
)

UNITS = (
    "log2(furfural-selected / untreated guide-barcode abundance); positive = the "
    "perturbation confers furfural tolerance"
)

#: Orthogonal Cas effector per modality (Table 1).
EFFECTOR = {"a": "dLbCas12a-VP", "i": "dSpCas9-RD1152", "d": "SaCas9"}
PERT_CLASS: dict[str, Any] = {
    "a": CrisprActivationPerturbation,
    "i": CrisprInterferencePerturbation,
    "d": CrisprDeletionPerturbation,
}
#: Integrated gRNA cassettes the round's HOST strain already carries, accumulated per
#: round: round 1 is screened in bAID (none), round 2 in R1 (SIZ1i at X3), round 3 in R2
#: (SIZ1i at X3 + NAT1a at X4). Each entry is ``(designation, locus, target ORF)``; the
#: designation and the locus are the SI's own (ROUND_STRAIN_GENOTYPES), and the ORF is
#: recorded so a guide that targets its own round background is still detectable
#: (SIZ1 = YDR409W, NAT1 = YDL040C).
ROUND_BACKGROUND: dict[int, list[tuple[str, str, str]]] = {
    1: [],
    2: [("SIZ1i", "X3", "YDR409W")],
    3: [("SIZ1i", "X3", "YDR409W"), ("NAT1a", "X4", "YDL040C")],
}


_BY4742_ALLELE_NOTE = (
    "Supplementary Table 11 states BY4742's genotype string ('MATα his3∆1 leu2∆0 "
    "lys2∆0 ura3∆0', BY4742_GENOTYPE), so WHICH alleles the strain carries is sourced; "
    "how each was constructed (the delta0 designer deletions vs the his3-delta1 "
    "internal deletion, which is what STANDARD_ALLELES encodes as an AlleleEdit) is "
    "stated only by Brachmann 1998, which is not mirrored"
)


def _baid_cassette() -> IntegratedCassette:
    """BAID's integrated CRISPR-AID cassette, at the Delta site, carrying KanMX.

    The site and the element list are read from Supplementary Table 11's bAID row
    (``BAID_GENOTYPE``), which the previous build's docstring wrongly called unstated.
    ``locus_systematic_gene_name`` stays None: the Delta site is the Ty1 delta repeat
    family, not an R64 ORF, so there is no systematic name to give it.
    """
    return IntegratedCassette(
        name="Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]",
        locus="Delta",
        elements=["KanMX", "dLbCpf1-VP", "Csy4", "dSpCas9-RD1152", "SaCas9"],
        marker="KanMX",
        zygosity=Zygosity.haploid,
        provenance=[BAID_GENOTYPE, HOST],
    )


def _round_cassette(designation: str, locus: str) -> IntegratedCassette:
    """One round's integrated gRNA expression cassette, marker-less, at a landing pad.

    ``elements`` is the single designation the source writes (``"SIZ1i"``), not a
    decomposition into promoter / spacer / terminator: the SI states the cassette by
    this name only, and naming parts it does not would be invention.
    """
    return IntegratedCassette(
        name=f"{locus}::{designation}",
        locus=locus,
        elements=[designation],
        marker=None,
        zygosity=Zygosity.haploid,
        provenance=[ROUND_STRAIN_GENOTYPES, MARKERLESS_INTEGRATION, INTEGRATION_LOCI],
    )


#: The strain each round's host was built FROM, as the SI's accumulating genotype column
#: shows it: bAID is built from BY4742, R1 from bAID, R2 from R1.
ROUND_PARENT_STRAIN: dict[int, str] = {1: "BY4742", 2: "bAID", 3: "bAID-X3::SIZ1i"}


def round_background(rnd: int) -> StrainBackground:
    """The typed background of the strain round ``rnd`` was screened in.

    BY4742's four auxotrophies plus every cassette the host carries integrated: bAID's
    CRISPR-AID cassette at the Delta site in all three rounds, and the round's
    accumulated gRNA cassettes (``ROUND_BACKGROUND``) on top.

    The auxotrophies are asserted with a ``deferred_pending_source_review`` gap naming
    Brachmann 1998 even though Supplementary Table 11 states the genotype string
    (``BY4742_GENOTYPE``): the SI states WHICH alleles BY4742 carries, not how any of
    them was made, and ``STANDARD_ALLELES`` reads each designation as a specific edit
    kind (``his3Δ1`` partial, the three ``Δ0`` alleles full) that only Brachmann 1998
    states. The strain name is the SI's own genotype column, not its terse ``R1`` /
    ``R2`` label, so the string says what the strain is.
    """
    alleles = [
        standard_allele(
            allele_name, zygosity, resolve_with=BRACHMANN_1998, note=_BY4742_ALLELE_NOTE
        )
        for allele_name, zygosity in STANDARD_BY_GENOTYPES["BY4742"].alleles.items()
    ]
    integrations = [_baid_cassette()] + [
        _round_cassette(designation, locus)
        for designation, locus, _ in ROUND_BACKGROUND[rnd]
    ]
    construction = HOST.quote
    if rnd > 1:
        construction = (
            f"{HOST.quote} {MARKERLESS_INTEGRATION.quote} Integrated here: "
            + ", ".join(c.name for c in integrations[1:])
        )
    return StrainBackground(
        name=ROUND_STRAIN[rnd],
        parents=[ROUND_PARENT_STRAIN[rnd]],
        construction=construction,
        mating_type=MatingType.alpha,
        ploidy="haploid",
        alleles=alleles,
        integrations=integrations,
        provenance=[BY4742_GENOTYPE, BAID_GENOTYPE, HOST],
    )


def crispr_perturbation(
    orf: str, common: str, mod: str, guide: str | None, donor: str | None = None
) -> Any:
    """Build the modality-appropriate CRISPR perturbation for a target gene."""
    construct = CrisprConstruct(
        effector=EFFECTOR[mod],
        guide_sequence=guide,
        n_guides=1 if guide is not None else None,
    )
    cls = PERT_CLASS[mod]
    if mod == "d":
        return cls(
            systematic_gene_name=orf,
            perturbed_gene_name=common,
            crispr=construct,
            donor_sequence=donor,
        )
    return cls(systematic_gene_name=orf, perturbed_gene_name=common, crispr=construct)


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror + build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/lianMultifunctionalGenomewideCRISPR2019``."""
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
    enrichment_path: str | Path,
    design_d_path: str | Path,
    retrieved_at: str = SI_RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror (derived enrichment + published CRISPRd design) + manifest.

    Idempotent by sha256. The design file is a directly scriptable Springer ESM whose
    pinned hash was reproduced on 2026-09-12. The enrichment table is DERIVED, not
    retrieved: its ``ProcessingRecord`` names the versioned reprocessing pipeline and the
    sha256 of its inputs (the Supplementary Data 4 guide reference and the raw SRA
    project), so the rebuild chain is recorded rather than a fabricated download URL.
    """
    root = raw_mirror_dir(data_root)
    design_url = f"{_ESM_BASE}{DESIGN_D_FILENAME}"
    reference_url = f"{_ESM_BASE}{REFERENCE_FILENAME}"
    files: list[ArtifactRecord] = []
    deposits = (
        (enrichment_path, TSV_REL, TSV_SHA256),
        (design_d_path, DESIGN_D_REL, DESIGN_D_SHA256),
    )
    # Both sources are verified before anything is written, so a refusal leaves no
    # mirror directory and no partial deposit behind.
    for source, _, expected in deposits:
        verify_sha256(source, expected)
    for source, relpath, expected in deposits:
        dest = root / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != expected:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(source, dest)
    files.append(
        ArtifactRecord(
            path=TSV_REL,
            role=ROLE_RAW_DATA,
            bytes=(root / TSV_REL).stat().st_size,
            sha256=TSV_SHA256,
            source="derived: SRA PRJNA504483 reprocessing",
            processing=ProcessingRecord(
                processor="experiments/016-lian-magic-reprocess/scripts/reproduce.sh",
                tool="torchcell lian-magic-reprocess pipeline",
                version="2026-07-13",
                params={
                    "sra_project": "PRJNA504483",
                    "n_runs": 21,
                    "barcode_window": "read[27:70] (43 bp activation) | read[27:71] "
                    "(44 bp interference/deletion), forward, exact match",
                    "normalization": "CPM(+1) per library; per round per replicate "
                    "log2(furfural-after / untreated-before); mean +- SD over triplicates",
                    "validation": "PDR1i round-3 rank 1, SLX5i round-1 rank 1, SAP30d "
                    "round-1 rank 2 against the paper's reported hits",
                    "reference_url": reference_url,
                },
                input_sha256=[REFERENCE_SHA256],
            ),
        )
    )
    files.append(
        ArtifactRecord(
            path=DESIGN_D_REL,
            role=ROLE_SI_DATA,
            bytes=(root / DESIGN_D_REL).stat().st_size,
            sha256=DESIGN_D_SHA256,
            source=design_url,
            retrieval=RetrievalRecord(
                method=RetrievalMethod.springer_esm,
                source_url=design_url,
                retriever="torchcell.literature.retrieve.springer_esm",
                params={"url": design_url},
                sha256=DESIGN_D_SHA256,
                retrieved_at=retrieved_at,
            ),
        )
    )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=(
            "Multi-functional genome-wide CRISPR system for high throughput "
            "genotype-phenotype mapping"
        ),
        files=files,
        si_data_sources=[
            design_url,
            reference_url,
            "https://www.ncbi.nlm.nih.gov/bioproject/PRJNA504483/",
        ],
        si_expected=[
            "Supplementary Data 3 (CRISPRd design library) -- consumed here for the "
            "guide/donor split; its Sequence column is element-for-element identical to "
            "the archived lab design file, and the same holds for Supplementary Data 1 "
            "and 2 against the CRISPRa and CRISPRi lab files",
            "Supplementary Data 4 (100,493-guide reference) -- an INPUT to the derived "
            "enrichment table, recorded in its processing record rather than mirrored "
            "here, since the loader does not read it",
            "the per-guide furfural enrichment itself was NEVER released; it is "
            "reprocessed from SRA PRJNA504483",
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


def split_deletion_cassette(sequence: str) -> tuple[str, str]:
    """``(guide spacer, HR donor)`` for one CRISPRd design cassette.

    The paper puts the donor 5' of the targeting sequence, and the boundary is measured
    against S288C (module docstring): the last :data:`D_SPACER_LEN` nt is the SaCas9
    spacer and everything before it is the donor.
    """
    text = sequence.strip().upper()
    if len(text) <= D_SPACER_LEN:
        raise ValueError(f"deletion cassette too short to split: {text!r}")
    return text[-D_SPACER_LEN:], text[:-D_SPACER_LEN]


# --------------------------------------------------------------------------- #
# Retention bookkeeping
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One retention rule, the records it removed, and the items it removed them for."""

    rule: str
    scope: Literal["guide", "guide_round", "strain"]
    description: str
    n_records: int
    items: list[str] = []


class DropLog(BaseModel):
    """Every retention rule applied to a build, in the order they were applied."""

    dataset: str
    source_records: int
    kept_records: int
    dropped_records: int
    rules: list[DropRule]


@register_dataset
class CrisprMagicLian2019Dataset(ExperimentDataset):
    """Lian 2019 MAGIC per-guide CRISPRa/i/d furfural-enrichment env x geno dataset."""

    def __init__(
        self,
        root: str = "data/torchcell/crispr_magic_lian2019",
        io_workers: int = 0,
        genome: SCerevisiaeGenome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; a genome is REQUIRED for common-name -> current-R64-ORF resolution."""
        self.genome = genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return StrainEnvironmentResponseExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return StrainEnvironmentResponseExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The derived enrichment table and the published CRISPRd design library."""
        return [TSV_FILENAME, DESIGN_D_FILENAME]

    def download(self) -> None:
        """Link the mirror files into ``raw/`` after verifying each against its pin.

        ``TSV_SHA256`` and ``DESIGN_D_SHA256`` are the pins; the manifest is the
        retrieval record and must carry the same digests, else
        ``ManifestPinMismatchError`` names both.
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        for relpath, filename, expected in (
            (TSV_REL, TSV_FILENAME, TSV_SHA256),
            (DESIGN_D_REL, DESIGN_D_FILENAME, DESIGN_D_SHA256),
        ):
            check_manifest_pin(relpath, manifest_sha256(manifest, relpath), expected)
            src = raw_mirror_dir(data_root) / relpath
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            link_verified(src, osp.join(self.raw_dir, filename), expected)
        log.info("Lian 2019 artifacts linked into %s (sha256 verified)", self.raw_dir)

    def _resolver(self) -> Callable[[str], str | None]:
        """Common/standard gene name -> current-R64 ORF via the SHARED genome resolver."""
        if self.genome is None:
            raise RuntimeError(
                "CrisprMagicLian2019Dataset requires a genome; inject SCerevisiaeGenome(...)"
            )
        genome = self.genome
        gene_set = {gene.upper() for gene in genome.gene_set}
        cache: dict[str, str | None] = {}

        def resolve(name: str) -> str | None:
            key = str(name).strip()
            if key not in cache:
                resolution = genome.resolve_gene_name(key)
                cache[key] = (
                    resolution.systematic_name
                    if resolution.is_current_gene
                    and resolution.systematic_name in gene_set
                    else None
                )
            return cache[key]

        return resolve

    def _deletion_cassettes(self, table: pd.DataFrame) -> list[tuple[str, str]]:
        """Per CRISPRd row of the enrichment table, its ``(spacer, donor)`` split.

        The published design library and the enrichment table's ``d`` block are the same
        24,806 rows in the same order. That positional join is ASSERTED here, not assumed:
        every row's released barcode must equal the first 44 nt of its design cassette.
        """
        design = pd.read_excel(osp.join(self.raw_dir, DESIGN_D_FILENAME))
        sequences = design["Sequence"].astype(str).str.upper().tolist()
        barcodes = table.loc[table["mod"] == "d", "spacer"].astype(str).str.upper()
        if len(sequences) != len(barcodes):
            raise RuntimeError(
                f"CRISPRd design rows ({len(sequences)}) do not match the enrichment "
                f"table's d rows ({len(barcodes)})"
            )
        mismatched = [
            i
            for i, (barcode, sequence) in enumerate(
                zip(barcodes, sequences, strict=True)
            )
            if barcode != sequence[:D_BARCODE_LEN]
        ]
        if mismatched:
            raise RuntimeError(
                f"{len(mismatched)} CRISPRd rows do not positionally join the design "
                f"library (first offenders: {mismatched[:5]})"
            )
        return [split_deletion_cassette(sequence) for sequence in sequences]

    def _culture_format(self) -> CultureFormat:
        """50 mL in a shaken 250 mL baffled flask, harvested at mid-log.

        ``endpoint`` is a typed gap rather than a value: the cultures were read when
        they reached mid-log phase, and ``EndpointRule`` has no member for that (its
        three are a fixed time, a fixed doubling count, and control saturation), so
        choosing one would assert a rule the source did not use. ``inoculum_od600`` is
        None for the same kind of reason: the OD 0.05 the paper states is for the
        individually constructed validation strains, not for the pooled screening
        flasks these records come from.
        """
        return CultureFormat(
            vessel="250 mL baffled flask",
            working_volume_ul=50000.0,
            shaking_rpm=250.0,
            inoculum_od600=None,
            endpoint=None,
            provenance=[SCREEN_VESSEL, SHAKING_RPM, HARVEST],
            provenance_gaps=[ENDPOINT_GAP, INOCULUM_GAP],
        )

    def _environment(self, furfural_mm: float) -> CultureEnvironment:
        """SED-URA/G418 liquid carrying furfural (mM), 30 C, aerobic, in a flask."""
        return CultureEnvironment(
            media=SED_URA_G418,
            temperature=Temperature(value=TEMPERATURE_C.value),
            perturbations=[
                SmallMoleculePerturbation(
                    compound=resolved_compound("furfural"),
                    concentration=Concentration(
                        value=furfural_mm, unit=ConcentrationUnit.millimolar
                    ),
                )
            ],
            aerobicity=AEROBICITY.value,
            culture_format=self._culture_format(),
            pre_culture=None,
            auxotroph_supplements=None,
            provenance_gaps=[*DURATION_GAPS, PRE_CULTURE_GAP, AUXOTROPH_SUPPLEMENT_GAP],
        )

    def _genome_reference(self, rnd: int) -> StrainReferenceGenome:
        """The typed background of the strain round ``rnd`` was screened in."""
        background = round_background(rnd)
        return StrainReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain=background.name,
            ploidy="haploid",
            background=background,
        )

    def _reference(
        self, rnd: int, environment: CultureEnvironment
    ) -> StrainEnvironmentResponseExperimentReference:
        """No-enrichment baseline: a guide that neither enriches nor depletes -> log2FC 0."""
        return StrainEnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=self._genome_reference(rnd),
            environment_reference=environment.model_copy(),
            phenotype_reference=EnvironmentResponsePhenotype(
                measurement_type=MeasurementType.log2_ratio,
                assay_type=ASSAY.value,
                environment_response=0.0,
                n_samples=N_REPLICATES.value,
                sample_unit=SampleUnit.biological_replicate,
                units=UNITS,
            ),
        )

    @post_process
    def process(self) -> None:
        """Build one env x geno -> log2-enrichment record per (guide, round); write LMDB."""
        verify_raw_files(
            self.raw_dir, {TSV_FILENAME: TSV_SHA256, DESIGN_D_FILENAME: DESIGN_D_SHA256}
        )
        table = pd.read_csv(osp.join(self.raw_dir, TSV_FILENAME), sep="\t")
        resolve = self._resolver()
        assert self.genome is not None
        canonical = canonical_common_names(self.genome)
        cassettes = self._deletion_cassettes(table)

        # Attach the CRISPRd (spacer, donor) split back onto the enrichment rows; every
        # other modality already releases its true spacer.
        spacers = table["spacer"].astype(str).str.upper().tolist()
        donors: list[str | None] = [None] * len(table)
        d_positions = [i for i, mod in enumerate(table["mod"]) if mod == "d"]
        for position, (spacer, donor) in zip(d_positions, cassettes, strict=True):
            spacers[position] = spacer
            donors[position] = donor
        table = table.assign(
            true_spacer=pd.Series(spacers, index=table.index),
            donor=pd.Series(donors, index=table.index, dtype=object),
        )

        background_orfs = {
            rnd: {orf for _, _, orf in ROUND_BACKGROUND[rnd]} for rnd in (1, 2, 3)
        }
        rounds = (1, 2, 3)

        # --- pass 1: gene resolution -------------------------------------------- #
        # Two CRISPRd designs can share a gene AND a 21 nt spacer (multicopy loci where
        # the spacer maps to more than one site) and differ only in their donor arms. They
        # are distinct strains and are all KEPT: the verifier's genotype signature reads
        # ``donor_sequence``, so the donor is what tells them apart.
        resolved: list[str | None] = []
        for _, row in table.iterrows():
            if bool(row["is_control"]) or bool(row["corrupted_gene"]):
                resolved.append(None)
                continue
            resolved.append(resolve(row["gene"]))

        publication = Publication(
            pubmed_id=PMID,
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PMID}/",
            doi=DOI,
            doi_url=f"https://doi.org/{DOI}",
        )
        # One environment and one strain-resolved reference per round. The round's
        # accumulated integrated gRNA cassettes are on the reference's typed background
        # (round_background), NOT in the record's Genotype: they are constant across
        # every record of that round and shared with its reference strain, which is the
        # line the schema draws between a background and a genotype. The Genotype
        # therefore holds exactly the ONE screened guide.
        prepared: dict[int, dict[str, Any]] = {}
        for rnd in rounds:
            environment = self._environment(FURFURAL_MM.value[rnd])
            prepared[rnd] = {
                "environment": environment,
                "reference": self._reference(rnd, environment),
            }

        # --- pass 2: write ------------------------------------------------------- #
        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        n_ctrl = n_corrupt = n_unresolved = n_nan = n_bg_self = 0
        unresolved_genes: set[str] = set()
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for position, (_, row) in tqdm(
                enumerate(table.iterrows()), total=len(table), desc="lian2019"
            ):
                if bool(row["is_control"]):
                    n_ctrl += 1
                    continue
                if bool(row["corrupted_gene"]):
                    n_corrupt += 1
                    continue
                orf = resolved[position]
                if orf is None:
                    n_unresolved += 1
                    unresolved_genes.add(str(row["gene"]))
                    continue
                mod = str(row["mod"])
                common = canonical.get(orf, str(row["gene"]))
                spacer = row["true_spacer"]
                donor = row["donor"]
                for rnd in rounds:
                    mean = row[f"r{rnd}_log2fc_mean"]
                    if pd.isna(mean):
                        n_nan += 1
                        continue
                    if orf in background_orfs[rnd]:
                        n_bg_self += 1
                        continue
                    sd = row[f"r{rnd}_log2fc_sd"]
                    has_sd = not pd.isna(sd) and not math.isinf(float(sd))
                    item = prepared[rnd]
                    experiment = StrainEnvironmentResponseExperiment(
                        dataset_name=self.name,
                        genotype=Genotype(
                            perturbations=[
                                crispr_perturbation(orf, common, mod, spacer, donor)
                            ]
                        ),
                        environment=item["environment"],
                        phenotype=EnvironmentResponsePhenotype(
                            measurement_type=MeasurementType.log2_ratio,
                            assay_type=ASSAY.value,
                            environment_response=float(mean),
                            environment_response_uncertainty=(
                                float(sd) if has_sd else None
                            ),
                            environment_response_uncertainty_type=(
                                UncertaintyType.sample_sd if has_sd else None
                            ),
                            n_samples=N_REPLICATES.value,
                            sample_unit=SampleUnit.biological_replicate,
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
        env.close()
        interned_env.close()

        source_records = len(table) * len(rounds)
        rules = [
            DropRule(
                rule="random_negative_control_guide",
                scope="guide",
                description=(
                    "one of the 300 random negative-control guides (100 per library); it "
                    "targets no gene, so there is no genotype to key a record to"
                ),
                n_records=n_ctrl * len(rounds),
                items=[],
            ),
            DropRule(
                rule="source_corrupted_gene_name",
                scope="guide",
                description=(
                    "the gene name is an Excel date/serial artifact present IDENTICALLY "
                    "in the released reference and the design library, so the target is "
                    "unrecoverable from our inputs"
                ),
                n_records=n_corrupt * len(rounds),
                items=[],
            ),
            DropRule(
                rule="target_gene_is_not_a_current_genome_gene",
                scope="guide",
                description=(
                    "the guide targets an ncRNA / rDNA feature or a name absent from the "
                    "current R64 ORF genome, so no gene entity exists to key the record to"
                ),
                n_records=n_unresolved * len(rounds),
                items=sorted(unresolved_genes)[:200],
            ),
            DropRule(
                rule="guide_round_has_no_enrichment_value",
                scope="guide_round",
                description=(
                    "the guide was not detected in this round's before/after libraries, "
                    "so the round has no log2 enrichment for it"
                ),
                n_records=n_nan,
                items=[],
            ),
            DropRule(
                rule="guide_targets_its_own_round_background",
                scope="guide_round",
                description=(
                    "in this round the guide's target gene is already an integrated "
                    "background perturbation, so the foreground edit is redundant and the "
                    "strain signature would collapse to empty once the background is "
                    "subtracted"
                ),
                n_records=n_bg_self,
                items=[],
            ),
        ]
        drop_log = DropLog(
            dataset=self.name,
            source_records=source_records,
            kept_records=idx,
            dropped_records=source_records - idx,
            rules=rules,
        )
        with open(osp.join(self.preprocess_dir, "dropped_records.json"), "w") as handle:
            handle.write(drop_log.model_dump_json(indent=2))
        accounted = sum(rule.n_records for rule in rules)
        if accounted != drop_log.dropped_records:
            raise RuntimeError(
                f"drop accounting mismatch: rules total {accounted}, "
                f"{drop_log.dropped_records} records missing from the build"
            )
        log.info(
            "Lian2019: wrote %d (guide x round) records; dropped %d control + %d "
            "corrupted + %d unresolved-gene guides (%d distinct genes); %d guide-rounds "
            "undetected; %d skipped (guide targets its own round background)",
            idx,
            n_ctrl,
            n_corrupt,
            n_unresolved,
            len(unresolved_genes),
            n_nan,
            n_bg_self,
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
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    root = osp.join(data_root, "data/torchcell/crispr_magic_lian2019")
    dataset = CrisprMagicLian2019Dataset(root=root, genome=genome)
    print(f"len = {len(dataset)}")
    print(
        json.dumps(
            json.loads(
                Path(osp.join(root, "preprocess/dropped_records.json")).read_text()
            )["rules"],
            indent=2,
        )[:2500]
    )


if __name__ == "__main__":
    main()
