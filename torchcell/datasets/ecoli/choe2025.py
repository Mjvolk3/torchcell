# torchcell/datasets/ecoli/choe2025
# [[torchcell.datasets.ecoli.choe2025]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/choe2025
# Test file: tests/torchcell/datasets/ecoli/test_choe2025.py
"""Choe 2025 genome-scale CRISPRi knockdown of E. coli under twelve antibiotics.

Choe, Lee, Kim, Hwang, Jeong, Palsson, Cho and Cho 2025 (iScience 28:112435,
doi:10.1016/j.isci.2025.112435; citation_key ``choeRapidIdentificationKey2025``)
transformed a 39,591-member sgRNA library, one guide per 100 nt of all 4,198 MG1655
coding sequences, into *E. coli* K-12 MG1655 carrying pdCas9, grew the pool in LB and in
LB plus a sub-inhibitory dose of each of twelve antibiotics, and read each guide's
abundance by amplicon sequencing. Table S1 releases the per-guide normalized abundance
(reads per million mapped reads) of 39,580 guides in two library samples and in two
replicate samples of each of the thirteen culture conditions.

RECORD = one (guide x antibiotic) ``BacterialEnvironmentResponseExperiment``:

- GENOTYPE: one ``BacterialCrisprInterferencePerturbation`` on the
  ``ecoli_k12_mg1655_bnumber`` namespace, whose ``crispr`` construct carries the
  effector ``dCas9``, the 20 nt spacer joined from Table S6 by the guide's released
  genomic coordinates, and ``n_guides`` = the number of Table S1 guides of that gene.
  The spacer is what keeps sibling guides of one gene distinct strains
  (``verification.environment_response._genotype_signature``).
- ENVIRONMENT: ``CHOE2025_LB_SELECTION`` (the shared ``LB`` Miller broth plus the
  35 ug/mL chloramphenicol and 100 ug/mL ampicillin the screening cultures carry for
  plasmid maintenance, both ``selection_agent`` components) at 37 C, shaken, aerobic,
  plus one ``SmallMoleculePerturbation``: the condition's antibiotic at its Table S3
  dose, with the 0.4% (v/v) DMSO vehicle as a typed ``Solvent`` for the six drugs whose
  Table S3 row states it.
- PHENOTYPE: ``EnvironmentResponsePhenotype``, ``log2_ratio`` by
  ``pooled_competitive_growth_barcode``: ``log2(a_abx / a_LB)``, where ``a_c`` is the
  guide's mean abundance over the two replicate samples of condition ``c``, which is the
  paper's own definition of an sgRNA's abundance under a condition. The reference is the
  SAME clone in untreated LB, log2(1) = 0. ``n_samples = 2``.

WHY THE LB CONTROL IS THE DENOMINATOR, NOT THE INITIAL LIBRARY. The paper's gene-level
statistic, the enrichment ratio ER, divides by the initial library (the pool WITHOUT
dCas9), so ER in plain LB measures the knockdown's own fitness cost and ER under a drug
conflates that cost with the drug effect. The paper's own antibiotic-response statistic
is the ratio BETWEEN those two, released in Table S4 as "Log2 ER ratio fold-change
(Abx/LB)", and the initial library cancels out of it: log2(ER_abx / ER_LB) =
log2(a_abx / a_LB). That is a treatment-over-control barcode abundance ratio, which is
exactly what ``MeasurementType.log2_ratio`` names, and it makes the reference a real
control environment (untreated LB) whose response is 0. Storing log2(ER) instead would
make the thirteenth, untreated condition a record with no environmental edit, which the
environment-response verifier's L3 ``environment_perturbed`` rule correctly refuses.

WHY ``EnvironmentResponsePhenotype`` AND NOT ``FitnessPhenotype``. The stored number is
signed (234,744 of the 466,569 values are negative, 13,176 of them below -1; the minimum
is -7.2275, the maximum +5.3219 and the median -0.0031) and its baseline is 0, not 1. ``FitnessPhenotype`` is a strictly positive
growth ratio that CLAMPS non-positive values and whose verifier requires a 1.0 reference,
so it cannot carry a log2 fold change.

WHY PER GUIDE AND NOT PER GENE. Table S2's ER is one number per (gene, condition), a
median the authors took over the gene's guides; Table S1 is the per-guide release the ER
is computed from, and the ``+guide`` sequence basis of this row is a guide, not a gene.
The two replicate columns of a condition are averaged because that average IS the paper's
definition of an sgRNA's abundance ("its mean abundance (reads per million mapped reads)
from two replicate measurements under a given condition"); pairing replicate 1 of an
antibiotic with replicate 1 of LB would assert a pairing the release never states, so the
replicate count rides on ``n_samples = 2`` instead.

ARITHMETIC CHECKS RUN AT BUILD TIME (``preprocess/extraction.json``), each of which
stops the build:

1. ``spacer_matches_the_released_window``: for every one of the 39,577 Table S1 guides
   that Table S6 names, the Table S6 spacer is the first 20 nt of the 23 nt MG1655
   window the row's Start and End delimit, in the orientation whose remaining 3 nt are
   an NGG PAM (19,183 on the reverse read, 20,394 on the forward read, none both ways,
   none neither). Spacer, coordinates and pinned assembly are therefore mutually
   consistent, measured, not assumed.
2. ``enrichment_ratio_reproduced``: the median over a gene's Table S1 guides of
   (condition mean abundance / library mean abundance) equals Table S2's released ER for
   all 4,192 genes with no zero-abundance library guide, in all 13 conditions (54,496
   cells, max absolute difference 5.0e-5, which is Table S2's 4-decimal rounding). The
   six genes carrying a guide absent from the initial library (``sbp``, ``nmpC``,
   ``ybgK``, ``yrhA``, ``ftsX``, ``yiiR``) are excluded from the CHECK, not from the
   dataset: that guide's ratio is infinite, which moves an even-length median, and the
   paper does not state how it handled it. The library columns are used for nothing else;
   the stored statistic does not divide by them.
3. ``library_coverage_reproduced``: 39,580 Table S1 rows minus the 6 whose two library
   samples are both zero = 39,574, the paper's own "39,574 sgRNA sequences were present
   in the initial library".

IDENTIFIERS. Table S1 names a guide's gene by symbol only; Table S2 is the released
symbol-to-b-number table and its 4,198 symbols are exactly Table S1's. Each released
b-number goes through ``reconcile_locus_tags`` against MG1655 GCA_000005845.2 (4,194 of
4,198 resolve: 4,070 current loci, 124 pseudogene loci that resolve to themselves, 4
through an ECK ``gene_synonym``). ``perturbed_gene_name`` is the pinned annotation's own
symbol for the stored locus when it resolves back to it, so one locus carries one
spelling; 213 of the 4,194 resolved symbols are an older spelling of the current one
(``yadD`` for ``rpnC``, ``ligT`` for ``thpR``). No ECK crosswalk is needed: the screen
host IS MG1655, so the released b-numbers are already in the stored namespace.

RECORDS DROPPED (rule, counts and items in ``preprocess/dropped_records.json``):

1. ``b_number_is_not_in_the_mg1655_annotation`` (17 guides, 4 genes): b0017 (insL1),
   b4590 (ybfK), b4694 (yagP) are no locus, symbol or synonym of GCA_000005845.2, and
   b4659 (yabP) matches both b0056 and b0057.
2. ``guide_window_does_not_overlap_the_resolved_locus`` (301 guides, 66 genes): the
   row's released Start-End lies outside the span of the locus its released b-number
   resolves to, so the stored gene identity would not be supported by the stored
   coordinates. The RefSeq NC_000913.3 spans the paper designed against are identical to
   the GenBank ones for every one of them, so this is not an annotation-version
   difference. Two groups dominate: the IS families, whose released symbol does not name
   one locus (58 guides under ``insC1`` and 38 under ``insH1``, against loci of 366 and
   981 bp, which at one guide per 100 nt is arithmetically impossible), and the Qin
   prophage cluster b1548-b1578 (``ydfK``, ``rem``, ``dicA``, ``flxA``, ``hokD``, ...),
   where the released symbol-to-b-number column and the released coordinate columns
   disagree by a shift.
3. ``guide_strand_disagrees_with_the_resolved_locus`` (12 guides: 9 ``lomR``, 3
   ``insO``): the window overlaps the locus but the released Strand is the opposite of
   the locus's, so the row and the resolved locus name different genes.
4. ``guide_has_no_spacer_in_table_s6`` (1 guide, ``ypjA`` 2778354-2778376): Table S1
   releases three coordinate pairs Table S6 does not; the other two belong to genes rule
   1 already removed. A guide with no spacer has no strain identity in a guide library.
5. ``abundance_is_zero_in_the_untreated_lb_control`` (283 guides): both LB replicates are
   0, so the log2 ratio has a zero denominator in all twelve conditions. A pseudocount
   would fabricate the value.
6. ``abundance_is_zero_in_the_antibiotic`` (1,023 cells): both replicates of that one
   antibiotic are 0 for a guide whose LB control is not, so that one record's log2 ratio
   is negative infinity. This is the strongest depletion a pooled screen can show and it
   has no finite log2; the honest answer is one dropped record, never a pseudocount.

39,580 - 17 - 301 - 12 - 1 - 283 = 38,966 guides kept; 38,966 x 12 - 1,023 = 466,569
records over 4,156 MG1655 loci.

DATA SOURCE. Table S1 (``mmc2.xlsx``), Table S2 (``mmc3.xlsx``) and Table S6
(``mmc5.xlsx``), all three from the PMC Article Datasets bucket prefix ``PMC12063145.1``
(``pmc_cloud``, scriptable: re-retrieved bit-identically on 2026-10-07), deposited in
``$DATA_ROOT/torchcell-raw/choeRapidIdentificationKey2025/data/`` with a
``manifest.json``. The ENA study PRJEB33267 holds the raw amplicon reads; the loader
consumes the released abundance tables, not the reads.

NOT ASSERTED. dCas9 sits under a tetracycline-inducible promoter (Figure S1A) and the
paper never names an inducer or a dose for the screen, so no inducer is put on the
environment. Each condition was harvested at its own OD600 (Table S3, 1.10 to 2.00) and
no wall time or doubling count is given, so ``duration_hours`` and
``duration_generations`` are typed gaps whose note carries the condition's sampling
OD600. The treatment doses are all 0.04 to 0.36 of the Table S3 lethal dose, but the
paper states no rule relating the two, so ``DoseBasis.fixed`` records them as the
explicit doses Table S3 prints. Table S3 gives sulfamethizole in mg/ml; 0.2 mg/ml is
stored as 200 ug/mL, a unit conversion inside the paper's own mass-per-volume dimension
(``ConcentrationUnit`` has no mg/mL member). Six of the twelve compounds (CCCP, polymyxin
B, pyocyanin, rifampicin, puromycin, phleomycin) have no row in the shared
compound-identity table, so ``resolved_compound`` returns them name-only with an
``inchikey`` gap, which is the honest shipped state and the Wang 2015 precedent.
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
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
    write_verified,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import LB
from torchcell.datamodels.schema import (
    BACTERIAL_ASSEMBLY_SETS,
    AssayType,
    AssemblyReferenceGenome,
    BacterialCrisprInterferencePerturbation,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    Concentration,
    ConcentrationUnit,
    CrisprConstruct,
    DoseBasis,
    Environment,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    Media,
    MediaComponent,
    MediaComponentRole,
    Publication,
    SmallMoleculePerturbation,
    Solvent,
    Temperature,
)
from torchcell.datasets.bacteria_common import (
    STRAIN_GENE_NAMESPACES,
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
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.literature.provenance import run_retriever
from torchcell.literature.retrieve import pmc_cloud_url
from torchcell.sequence.genome.bacterial import BacterialGenome
from torchcell.sequence.genome.base import GeneNameStatus
from torchcell.sequence.genome.ecoli.k12 import EcoliK12Genome, EcoliK12StrainName
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
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "choeRapidIdentificationKey2025"
PAPER_DOI = "10.1016/j.isci.2025.112435"
PAPER_TITLE = (
    "Rapid identification of key antibiotic resistance genes in E. coli using "
    "high-resolution genome-scale CRISPRi screening"
)
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
LIBRARY_DIR_REL = f"torchcell-library/{CITATION_KEY}"
DATASET_ROOT_REL = "data/torchcell/crispri_chemgen_choe2025"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "d29923ce2276b17882e89407b683a1edce74d4da74535f7398f7e68c86abcbf7"
#: MinerU OCR of the SI PDF; the anchor of every Table S3 quote.
SI1_MD = "si/si1.md"
SI1_MD_SHA256 = "bc26a5be98957b07ed65addd027fe7240a6ec52ad2710dcd9b7a3a245e1e8895"

#: The PMC Article Datasets bucket prefix of this article (version 1).
PMC_PREFIX = "PMC12063145.1"
#: When the three workbooks were re-retrieved and re-hashed before deposit.
RETRIEVED_AT = "2026-10-07"

TABLE_S1_FILE = "mmc2.xlsx"
TABLE_S2_FILE = "mmc3.xlsx"
TABLE_S6_FILE = "mmc5.xlsx"

#: The ENA study holding the raw amplicon reads (retrieval metadata, not a build input).
ENA_STUDY_URL = "https://www.ebi.ac.uk/ena/browser/view/PRJEB33267"


class RawFile(BaseModel):
    """One file the loader consumes: its pinned bytes and how they were retrieved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    sha256: str
    bytes: int
    description: str

    @property
    def mirror_relpath(self) -> str:
        """Path inside the raw mirror (``data/<name>``)."""
        return f"data/{self.name}"

    @property
    def bucket_key(self) -> str:
        """The PMC Article Datasets bucket key the bytes came from."""
        return f"{PMC_PREFIX}/{self.name}"

    @property
    def source_url(self) -> str:
        """HTTPS URL of the bucket object."""
        return pmc_cloud_url(self.bucket_key)

    @property
    def retrieval(self) -> RetrievalRecord:
        """The re-runnable retrieval of these bytes."""
        return RetrievalRecord(
            method=RetrievalMethod.pmc_cloud,
            source_url=self.source_url,
            retriever="torchcell.literature.retrieve.pmc_cloud_object",
            params={"key": self.bucket_key},
            sha256=self.sha256,
            retrieved_at=RETRIEVED_AT,
        )


RAW_FILES: tuple[RawFile, ...] = (
    RawFile(
        name=TABLE_S1_FILE,
        sha256="7d1224ccf399fb5f89fb33915ee01680cfc13a98ef8df1aa94d87e646339a8cc",
        bytes=9020670,
        description="Table S1, normalized sgRNA counts (reads per million mapped "
        "reads): 39,580 guides x (2 initial-library samples + 2 replicate samples of "
        "each of 13 culture conditions), with each guide's gene symbol, genomic Start "
        "and End, strand and CDS fragment",
    ),
    RawFile(
        name=TABLE_S2_FILE,
        sha256="23eba0baea3e2d4e4041af6db61eaee6b425667d01597824c7aee564d416c681",
        bytes=703998,
        description="Table S2, enrichment ratio (ER) of 4,198 targeted genes in 13 "
        "conditions, with the released gene symbol to b-number map, guide count, Keio "
        "essentiality flag, COG category and product function",
    ),
    RawFile(
        name=TABLE_S6_FILE,
        sha256="4f8642f007e403fb2676942d07308482ba14c87836f4626dcf5effe9fabe59f9",
        bytes=1999427,
        description="Table S6, the synthesized 39,591-oligo sgRNA library: each oligo's "
        "name (orientation, index, Start, End, CDS fragment, gene index) and its 79 nt "
        "sequence, from which the 20 nt spacer is sliced",
    ),
)
#: ``{raw file name: pinned sha256}``, the build-time check of every consumed file.
DATA_SHA256: dict[str, str] = {f.name: f.sha256 for f in RAW_FILES}
RAW_FILES_BY_NAME: dict[str, RawFile] = {f.name: f for f in RAW_FILES}

#: What the article releases that the loader deliberately does not consume.
NOT_MIRRORED = (
    "ENA PRJEB33267 (raw CRISPRi amplicon reads): the released per-guide abundance "
    f"tables are what the records are built from; the reads are at {ENA_STUDY_URL}",
    "mmc1.pdf (Supplementary Information): quoted through the literature mirror's OCR "
    "si/si1.md for Table S3's doses; no per-record value is in it",
    "mmc4.xlsx (Table S4, log2 ER ratios of the 1,085 significantly changed genes): a "
    "gene-level subset derived from Table S2 by log2, which is reproduced exactly from "
    "Table S2 (yaaX under CCCP: mean log2 ER -0.58824, log2 fold-change 0.78227), so "
    "loading it would store the same measurement twice",
    "mmc6.zip, mmc7.zip, mmc8.zip (Data S1 to S3, the in-house sgRNA design scripts): "
    "code, not data",
    "mmc5.xlsx Table S5 primers and the KEIO single-knockout growth curves (Figures "
    "S5 to S10): the knockout area-under-curve arm is a separate assay on BW25113 "
    "strains, released only as figures",
)


def _paper(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
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
            page=page,
        ),
    )


def _si(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """Bind a value to a verbatim quote in the SI OCR (``si/si1.md``)."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SI1_MD,
            citation_key=CITATION_KEY,
            sha256=SI1_MD_SHA256,
            method="MinerU OCR of the SI PDF (torchcell-library mirror)",
            page="Table S3. A list of antibiotics and treatment conditions",
        ),
    )


# --------------------------------------------------------------------------- #
# Sourced values (verbatim quotes, pinned sha256)
# --------------------------------------------------------------------------- #
_STRAINS = "STAR Methods, METHOD DETAILS, 'Bacterial strains'"
_LIBRARY = "STAR Methods, METHOD DETAILS, 'Design of sgRNA library'"
_CONSTRUCTION = "STAR Methods, METHOD DETAILS, 'sgRNA library construction'"
_POPULATIONS = (
    "STAR Methods, METHOD DETAILS, 'Generation of antibiotics treated populations'"
)
_SEQUENCING = "STAR Methods, METHOD DETAILS, 'sgRNA library sequencing'"
_QUANT = (
    "STAR Methods, QUANTIFICATION AND STATISTICAL ANALYSIS, "
    "'Quantification of gene fitness'"
)
_RESULTS = (
    "Results, 'Genome-wide fitness effect of a gene knockdown using high-resolution "
    "CRISPRi screening'"
)
_KEY_RESOURCES = "STAR Methods, KEY RESOURCES TABLE"

_SCREEN_CULTURE_QUOTE = (
    "Then, pre-cultured cells were re-inoculated to an $\\mathsf { O D } _ { 6 0 0 } "
    "0 . 0 5$ in $5 0 \\mathsf { m l }$ LB medium or LB media with antibiotics "
    "containing a sublethal dose of antibiotics (Table S3), and cultured shaking with "
    "250 rpm at $3 7 ^ { \\circ } \\mathrm { C }$ until the required cell density was "
    "achieved."
)

SOURCED_VALUES: dict[str, SourcedValue] = {
    "host_strain": _paper(
        "MG1655",
        "Purified sgRNA library plasmids (100 ng) were transformed into E. coli K-12 "
        "MG1655 competent cells containing the pdCas9 plasmid through electroporation.",
        page=_POPULATIONS,
        note="the screened pool is MG1655, so the released b-numbers are already in the "
        "stored namespace and no ECK crosswalk is involved",
    ),
    "effector": _paper(
        "dCas9",
        "dCas9 and sgRNA were expressed with separate plasmids, pdCas9 plasmid73 (cat. "
        "#46569; Addgene, Cambridge, MA, USA) and pgRNA-bacteria plasmid4 (cat. #44251; "
        "Addgene) targeting mrfp (Figure S1A)",
        page=_STRAINS,
        note="the stored CrisprConstruct.effector is the paper's own name for the "
        "catalytically dead Cas9 it expresses from pdCas9",
    ),
    "library_design": _paper(
        (4198, 39591),
        "we designed and synthesized a library of evenly spaced sgRNAs targeting every "
        "100 bp of the 4,198 coding sequences (CDSs) in E. coli (targeting 39,591 "
        "positions, Figure 1A)",
        page=_RESULTS,
        note="the 4,198 genes are exactly Table S1's and Table S2's gene sets; the "
        "39,591 oligos are Table S6's rows",
    ),
    "library_coverage": _paper(
        39574,
        "Of the sgRNAs, 39,574 sgRNA sequences were present in the initial library "
        "(lib), representing $9 9 . 9 6 \\%$ coverage (Table S1).",
        page=_RESULTS,
        note="reproduced by the build: 39,580 Table S1 rows minus the 6 whose two "
        "library samples are both zero",
    ),
    "guide_spacing": _paper(
        100,
        "As mentioned above, we designed sgRNAs for the CRISPRi system with a ratio of "
        "one within 100 nt of all CDSs in E. coli K-12 MG1655.",
        page=_CONSTRUCTION,
    ),
    "oligo_layout": _paper(
        ("GCTCAGTCCTAGGTATAATACTAGTA", 20, "GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAG"),
        "A pool of 39,591 oligos consisting of a 20 nt random spacer and additional "
        "sequences for the ${ } ^ { 5 ^ { \\prime } }$ and ${ \\mathfrak { z } } ^ { "
        "\\prime }$ extensions was synthesized in the form of ${ } ^ { 5 ^ { \\prime } "
        "}$ -GCTCAGTCC TAGGTATAATACTAGTA-20 nt spacer-"
        "GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAG- ${ \\cdot } 3 ^ { \\prime }$ by CustomArray",
        page=_CONSTRUCTION,
        note="the 26 nt prefix, the 20 nt spacer and the 33 nt suffix the loader slices "
        "a Table S6 oligo on; the OCR breaks the prefix across a line",
    ),
    "pam_rule": _paper(
        23,
        "candidate sgRNAs were acquired by dissecting all genic regions into 23 mers "
        "that began with the protospacer adjacent motif (PAM) sequence",
        page=_LIBRARY,
        note="why a Table S1 Start-End window is 23 nt: 20 nt spacer plus a 3 nt PAM. "
        "The build checks every spacer against its window in both orientations",
    ),
    "unique_alignment": _paper(
        "one locus",
        "Sequences that aligned to more than one locus in the genome were excluded.",
        page=_LIBRARY,
        note="the design rule a guide whose window does not overlap its released "
        "locus violates; the IS-family rows are where it breaks down",
    ),
    "cds_source": _paper(
        "NC_000913.3",
        "gRNAs positioned in the CDS were extracted from the CDS list (CDSs extracted "
        "from the NC_000913.3 RefSeq annotation)",
        page=_LIBRARY,
        note="the RefSeq view of the same replicon the pinned GenBank assembly "
        "GCA_000005845.2 carries as U00096.3; the build measured the gene spans to be "
        "identical in both for every flagged guide",
    ),
    "read_mapping": _paper(
        "NC_000913.3",
        "Raw reads were mapped to the E. coli MG1655 reference genome (NC_000913.3) "
        "using the CLC Genomics Workbench (CLC Bio, Aarhus, Denmark), with a length and "
        "similarity fraction of 0.9.",
        page=_SEQUENCING,
    ),
    "abundance_definition": _paper(
        2,
        "The abundance change of each sgRNA was calculated by dividing its mean "
        "abundance (reads per million mapped reads) from two replicate measurements "
        "under a given condition by its mean abundance in the initial library.",
        page=_QUANT,
        note="the two replicate columns a condition's abundance is the mean of, and the "
        "source of n_samples = 2. The paper does not say whether a replicate is "
        "biological or technical, so sample_unit is a typed gap",
    ),
    "enrichment_ratio_definition": _paper(
        "median",
        "The enrichment ratio (ER) of a gene is the median of abundance changes of all "
        "sgRNAs targeting that gene.",
        page=_QUANT,
        note="the gene-level statistic the build reproduces from Table S1 as its "
        "fidelity check; the stored records are per guide",
    ),
    "medium_and_selection": _paper(
        ("LB", 35.0, 100.0),
        "E. coli cells for sgRNA library construction and for CRISPRi screening were "
        "each cultivated in LB medium supplemented with $1 0 0 \\mu \\mathrm { g / m l "
        "}$ ampicillin and $3 5 \\mu \\ g / \\mathrm { m l }$ chloramphenicol plus $1 0 "
        "0 \\mu \\mathrm { g / m l }$ ampicillin, respectively.",
        page=_STRAINS,
        note="the screening cultures are the 'respectively' arm: LB plus 35 ug/mL "
        "chloramphenicol and 100 ug/mL ampicillin, the two selection agents of the "
        "stored medium",
    ),
    "lb_miller": _paper(
        "LB Miller broth",
        "<td>LB Miller broth</td><td>BD Difco</td><td>244620</td></tr>",
        page=_KEY_RESOURCES,
        note="the formulation named, which is the shared media-library LB object",
    ),
    "temperature": _paper(37.0, _SCREEN_CULTURE_QUOTE, page=_POPULATIONS),
    "aerobicity": _paper(
        "aerobic",
        _SCREEN_CULTURE_QUOTE,
        page=_POPULATIONS,
        note="50 ml cultures shaken at 250 rpm; no anaerobic chamber or gas control is "
        "described",
    ),
    "harvest_rule": _paper(
        "sampling OD600 per condition",
        _SCREEN_CULTURE_QUOTE,
        page=_POPULATIONS,
        note="the cultures are harvested by density, not by time: Table S3 gives each "
        "condition's sampling OD600 (1.10 to 2.00) and the paper gives no wall time or "
        "doubling count, so duration_hours and duration_generations are typed gaps",
    ),
    "twelve_antibiotics": _paper(
        12,
        "Next, we exposed the culture to sub-inhibitory concentrations of 12 different "
        "antibiotics to identify genes affecting cellular responses to antibiotics "
        "(Figure S3; Table S3)",
        page=_RESULTS,
    ),
    "assay_type": _paper(
        AssayType.pooled_competitive_growth_barcode.value,
        "The number of reads fall onto each gRNA were counted (Table S1) and ratio of "
        "gRNA abundance compared to initial library was calculated. The enrichment "
        "ratio (ER) was determined as the median of all gRNA abundance ratios in a gene "
        "(Table S2).",
        page=_SEQUENCING,
        note="a pooled competitive-growth screen read out by amplicon sequencing of the "
        "sgRNA spacer, which is the clone's barcode",
    ),
    "ena_study": _paper(
        "PRJEB33267",
        "Original data of CRISPRi amplicon sequencing have been deposited at European "
        "Nucleotide Archive as ENA: PRJEB33267 (https://www.ebi. "
        "ac.uk/ena/browser/view/PRJEB33267) and are publicly available as of the date "
        "of publication.",
        page="RESOURCE AVAILABILITY, 'Data and code availability'",
        note="retrieval metadata for the raw reads, which this loader does not consume",
    ),
}

# --------------------------------------------------------------------------- #
# The screened conditions (Table S3) and the released column labels
# --------------------------------------------------------------------------- #
#: The untreated control column of Table S1, the denominator of every stored ratio.
CONTROL_LABEL = "LB"
#: Its Table S3 row, quoted for the control's own identity.
CONTROL_ROW = _si(
    CONTROL_LABEL,
    "<tr><td>LB</td><td>LB</td><td>-</td><td></td><td>-</td><td>1.80</td><td></td></tr>",
    note="the untreated control condition: LB with no additive and no solvent, "
    "harvested at OD600 1.80",
)
CONTROL_SAMPLING_OD600 = 1.80

#: The DMSO vehicle fraction of the six Table S3 rows whose Solvent/Media is DMSO/LB.
DMSO_PERCENT = 0.4


class ScreenCondition(BaseModel):
    """One antibiotic arm of the screen, as Table S1, S2 and S3 name and dose it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    s1_label: str
    """Table S1 column prefix; its two samples are ``<s1_label>_1`` and ``_2``."""
    s2_column: str
    """Table S2 ER column header for the same condition."""
    compound_label: str
    """Label handed to ``resolved_compound`` for the typed chemical identity."""
    dose: float
    unit: ConcentrationUnit
    dmso: bool
    """True when Table S3's Solvent/Media column is ``0.4% DMSO/LB``."""
    sampling_od600: float
    mode_of_action: str
    """Table S3's own words for the drug's mode of action, verbatim."""
    table_s3_row: str
    """The Table S3 row, verbatim from the SI OCR, that states all of the above."""

    @property
    def dose_sourced(self) -> SourcedValue:
        """The dose bound to its Table S3 row."""
        return _si(
            (self.dose, self.unit.value),
            self.table_s3_row,
            note=f"{self.s1_label}: the Treatment and Unit columns of its Table S3 row"
            + (
                ", with 0.2 mg/ml converted to 200 ug/mL (ConcentrationUnit has no "
                "mg/mL member and the conversion stays inside the paper's own "
                "mass-per-volume dimension)"
                if self.s1_label == "Sulfamethizole"
                else ""
            ),
        )


CONDITIONS: tuple[ScreenCondition, ...] = (
    ScreenCondition(
        s1_label="CCCP",
        s2_column="CCCP",
        compound_label="CCCP",
        dose=5.3,
        unit=ConcentrationUnit.ug_per_ml,
        dmso=True,
        sampling_od600=1.22,
        mode_of_action="Inhibits proton motif force generation",
        table_s3_row=(
            "<tr><td>CCCP</td><td>0.4% DMSO/LB</td><td>16</td><td>5.3</td>"
            "<td>μg/ml</td><td>1.22</td>"
            "<td>Inhibits proton motif force generation</td></tr>"
        ),
    ),
    ScreenCondition(
        s1_label="Polymyxin B",
        s2_column="PolymyxinB",
        compound_label="polymyxin B",
        dose=0.5,
        unit=ConcentrationUnit.ug_per_ml,
        dmso=False,
        sampling_od600=1.53,
        mode_of_action="Destabilizes the outer membrane",
        table_s3_row=(
            "<tr><td>Polymyxin B</td><td>LB/LB</td><td>2.7</td><td>0.5</td>"
            "<td>μg/ml</td><td>1.53</td>"
            "<td>Destabilizes the outer membrane</td></tr>"
        ),
    ),
    ScreenCondition(
        s1_label="Pyocyanine",
        s2_column="Pyocyanin",
        compound_label="pyocyanin",
        dose=5.0,
        unit=ConcentrationUnit.ug_per_ml,
        dmso=True,
        sampling_od600=1.50,
        mode_of_action="Induces superoxide stress",
        table_s3_row=(
            "<tr><td>Pyocyanin</td><td>0.4% DMSO/LB</td><td>26</td><td>5</td>"
            "<td>μg/ml</td><td>1.50</td><td>Induces superoxide stress</td></tr>"
        ),
    ),
    ScreenCondition(
        s1_label="Rifampicin",
        s2_column="Rifampicin",
        compound_label="rifampicin",
        dose=3.2,
        unit=ConcentrationUnit.ug_per_ml,
        dmso=True,
        sampling_od600=1.32,
        mode_of_action="Transcription inhibitor",
        table_s3_row=(
            "<tr><td>Rifampicin</td><td>0.4% DMSO/LB</td><td>9.5</td><td>3.2</td>"
            "<td>μg/ml</td><td>1.32</td><td>Transcription inhibitor</td></tr>"
        ),
    ),
    ScreenCondition(
        s1_label="Sulfamethizole",
        s2_column="Sulfamethizole",
        compound_label="sulfamethizole",
        dose=200.0,
        unit=ConcentrationUnit.ug_per_ml,
        dmso=True,
        sampling_od600=1.36,
        mode_of_action="Folic acid biosynthesis inhibitor",
        table_s3_row=(
            "<tr><td>Sulfamethizole</td><td>0.4% DMSO/LB</td><td>0.6</td><td>0.2</td>"
            "<td>mg/ml</td><td>1.36</td>"
            "<td>Folic acid biosynthesis inhibitor</td></tr>"
        ),
    ),
    ScreenCondition(
        s1_label="Verapamil",
        s2_column="Verapamil",
        compound_label="verapamil",
        dose=3.6,
        unit=ConcentrationUnit.millimolar,
        dmso=False,
        sampling_od600=1.50,
        mode_of_action="Calcium channel blocker",
        table_s3_row=(
            "<tr><td>Verapamil</td><td>LB/LB</td><td>10.8</td><td>3.6</td>"
            "<td>mM</td><td>1.50</td><td>Calcium channel blocker</td></tr>"
        ),
    ),
    ScreenCondition(
        s1_label="Erythromycin",
        s2_column="Erythromycin",
        compound_label="erythromycin",
        dose=13.4,
        unit=ConcentrationUnit.ug_per_ml,
        dmso=True,
        sampling_od600=1.12,
        mode_of_action="Translation inhibitor",
        table_s3_row=(
            "<tr><td>Erythromycin</td><td>0.4% DMSO/LB</td><td>40.3</td><td>13.4</td>"
            "<td>μg/ml</td><td>1.12</td><td>Translation inhibitor</td></tr>"
        ),
    ),
    ScreenCondition(
        s1_label="Puromycin",
        s2_column="Puromycin",
        compound_label="puromycin",
        dose=26.0,
        unit=ConcentrationUnit.ug_per_ml,
        dmso=False,
        sampling_od600=2.00,
        mode_of_action="Translation inhibitor",
        table_s3_row=(
            "<tr><td>Puromycin</td><td>LB/LB</td><td>77.8</td><td>26</td>"
            "<td>μg/ml</td><td>2.00</td><td>Translation inhibitor</td></tr>"
        ),
    ),
    ScreenCondition(
        s1_label="Phleomycin",
        s2_column="Phleomycin",
        compound_label="phleomycin",
        dose=5.0,
        unit=ConcentrationUnit.ug_per_ml,
        dmso=False,
        sampling_od600=1.12,
        mode_of_action="DNA damage; intercalates DNA",
        table_s3_row=(
            "<tr><td>Phleomycin</td><td>LB/LB</td><td>121.2</td><td>5</td>"
            "<td>μg/ml</td><td>1.12</td>"
            "<td>DNA damage; intercalates DNA</td></tr>"
        ),
    ),
    ScreenCondition(
        s1_label="Mitomycin C",
        s2_column="Mitomycin",
        compound_label="mitomycin C",
        dose=26.5,
        unit=ConcentrationUnit.ug_per_ml,
        dmso=True,
        sampling_od600=1.10,
        mode_of_action="DNA damage; DNA replication inhibitor",
        table_s3_row=(
            "<tr><td>Mitomycin C</td><td>0.4% DMSO/LB</td><td>79.6</td><td>26.5</td>"
            "<td>μg/ml</td><td>1.10</td>"
            "<td>DNA damage; DNA replication inhibitor</td></tr>"
        ),
    ),
    ScreenCondition(
        s1_label="MMS",
        s2_column="MMS",
        compound_label="methyl methanesulfonate",
        dose=0.04,
        unit=ConcentrationUnit.percent_w_v,
        dmso=False,
        sampling_od600=1.30,
        mode_of_action="DNA damage; methylates DNA",
        table_s3_row=(
            "<tr><td>MMS</td><td>LB/LB</td><td>0.11</td><td>0.04</td>"
            "<td>% (w/v)</td><td>1.30</td>"
            "<td>DNA damage; methylates DNA</td></tr>"
        ),
    ),
    ScreenCondition(
        s1_label="Novobiocin",
        s2_column="Novobiocin",
        compound_label="novobiocin",
        dose=35.0,
        unit=ConcentrationUnit.ug_per_ml,
        dmso=False,
        sampling_od600=1.28,
        mode_of_action="DNA damage; DNA gyrase inhibitor",
        table_s3_row=(
            "<tr><td>Novobiocin</td><td>LB/LB</td><td>104.1</td><td>35</td>"
            "<td>μg/ml</td><td>1.28</td>"
            "<td>DNA damage; DNA gyrase inhibitor</td></tr>"
        ),
    ),
)
CONDITIONS_BY_LABEL: dict[str, ScreenCondition] = {c.s1_label: c for c in CONDITIONS}

#: Every Table S1 condition column prefix in sheet order: the control, then the twelve.
S1_CONDITION_LABELS: tuple[str, ...] = (
    CONTROL_LABEL,
    *(c.s1_label for c in CONDITIONS),
)
#: Every Table S2 ER column header in sheet order.
S2_CONDITION_COLUMNS: tuple[str, ...] = (
    CONTROL_LABEL,
    *(c.s2_column for c in CONDITIONS),
)
#: The two initial-library (no dCas9) sample columns of Table S1.
LIBRARY_COLUMNS: tuple[str, str] = ("Lib1", "Lib2")

# --------------------------------------------------------------------------- #
# Record-level constants
# --------------------------------------------------------------------------- #
MEASUREMENT_TYPE = MeasurementType.log2_ratio
ASSAY_TYPE = AssayType.pooled_competitive_growth_barcode
RESPONSE_UNITS = (
    "log2(mean sgRNA abundance in the antibiotic / mean sgRNA abundance in untreated "
    "LB), abundance in reads per million mapped reads averaged over the condition's two "
    "replicate samples (Table S1)"
)
GENE_NAMESPACE = STRAIN_GENE_NAMESPACES["MG1655"]
EFFECTOR = "dCas9"
N_REPLICATES = 2

PUBLICATION = Publication(doi=PAPER_DOI, doi_url=f"https://doi.org/{PAPER_DOI}")

#: Fraction of the released b-numbers that must resolve to one MG1655 locus. Measured on
#: the release: 4,194 of 4,198 (0.999). Below it the build stops and reports.
MIN_RESOLVED_FRACTION = 0.99

#: The paper's own count of guides present in the initial library, which the build
#: reproduces from Table S1 (a module constant so a test can state its own).
REPORTED_LIBRARY_COVERAGE = int(SOURCED_VALUES["library_coverage"].value)

#: Largest absolute difference tolerated between a reproduced ER and Table S2's released
#: value. Measured on the release: 5.0e-5, which is Table S2's 4-decimal rounding.
ER_TOLERANCE = 1e-3

#: The frozen record-count oracle: 38,966 kept guides x 12 antibiotics, minus the 1,023
#: (guide, antibiotic) cells whose antibiotic abundance is zero.
EXPECTED_RECORDS = 466569


# --------------------------------------------------------------------------- #
# The screening medium
# --------------------------------------------------------------------------- #
def _selection_component(label: str, dose: float) -> MediaComponent:
    """One selection agent of the screening medium at its stated ug/mL dose."""
    return MediaComponent(
        compound=resolved_compound(label),
        role=MediaComponentRole.selection_agent,
        concentration=Concentration(value=dose, unit=ConcentrationUnit.ug_per_ml),
        provenance=[SOURCED_VALUES["medium_and_selection"]],
        note=f"maintains the {'pdCas9' if label == 'chloramphenicol' else 'sgRNA'} "
        "plasmid; present in every condition including the untreated control",
    )


#: The screening medium: the shared ``LB`` Miller broth plus the two plasmid-maintenance
#: selection agents the Methods state. ``base_medium="LB"`` keeps it joined to the shared
#: library entry rather than standing alone.
CHOE2025_LB_SELECTION = Media(
    name="LB, Miller, with 35 ug/mL chloramphenicol and 100 ug/mL ampicillin "
    "(Choe 2025 CRISPRi screening cultures)",
    state="liquid",
    is_synthetic=False,
    base_medium="LB",
    components=[
        *LB.components,
        _selection_component("chloramphenicol", 35.0),
        _selection_component("ampicillin", 100.0),
    ],
    provenance=[SOURCED_VALUES["lb_miller"], SOURCED_VALUES["medium_and_selection"]],
)


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirrors and build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/choeRapidIdentificationKey2025``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def library_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-library/choeRapidIdentificationKey2025``."""
    return Path(data_root or _data_root()) / LIBRARY_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retrieve_raw_files(dest_dir: str | Path) -> dict[str, Path]:
    """Run each file's recorded retriever and write the verified bytes to ``dest_dir``.

    The recorded ``RetrievalRecord`` is what runs (``run_retriever``), so this IS the
    re-runnable retrieval; a byte mismatch raises before anything is written.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for raw in RAW_FILES:
        path = dest / raw.name
        write_verified(run_retriever(raw.retrieval), path, raw.sha256, raw.source_url)
        out[raw.name] = path
    return out


def deposit_raw_mirror(
    *, sources: Mapping[str, str | Path], data_root: str | None = None
) -> Path:
    """Write the raw mirror from already-retrieved files plus its ``manifest.json``.

    ``sources`` maps every name in ``RAW_FILES`` to a local file. Idempotent by sha256: a
    mirror file with the pinned hash is left alone, and one with any other hash raises
    rather than being overwritten.
    """
    missing = sorted(set(DATA_SHA256) - set(sources))
    if missing:
        raise KeyError(f"no source given for {missing}")
    root = raw_mirror_dir(data_root)
    records: list[ArtifactRecord] = []
    for raw in RAW_FILES:
        src = Path(sources[raw.name])
        got = _sha256(src)
        if got != raw.sha256:
            raise RuntimeError(
                f"{src} sha256 mismatch: got {got}, expected {raw.sha256}"
            )
        dest = root / raw.mirror_relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != raw.sha256:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(src, dest)
        records.append(
            ArtifactRecord(
                path=raw.mirror_relpath,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=raw.sha256,
                source=raw.source_url,
                retrieval=raw.retrieval,
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=PAPER_TITLE,
        files=records,
        si_data_sources=[raw.source_url for raw in RAW_FILES],
        si_expected=list(NOT_MIRRORED),
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
# Reading the three workbooks
# --------------------------------------------------------------------------- #
class TableLayoutError(ValueError):
    """A released workbook's header is not the one the loader was written against."""


#: Table S1's five identifier columns, then one column per sample.
S1_ID_COLUMNS: tuple[str, ...] = ("Gene", "Start", "End", "Strand", "Fragment")
#: Table S1's group label over every sample column, which names the released unit.
S1_VALUE_GROUP = "gRNAs per million mapped reads"
#: Table S2's seven identifier columns, then one ER column per condition.
S2_ID_COLUMNS: tuple[str, ...] = (
    "Name",
    "b number",
    "Strand",
    "gRNA #",
    "KEIO*",
    "COG†",
    "Function",
)
#: Table S6's four columns.
S6_COLUMNS: tuple[str, ...] = ("#", "Primer Name", "Sequence", "Length")

#: ``<orientation><index>_<start>_<end>_<fragment>_<gene index>``, the Table S6 name.
S6_NAME = re.compile(
    r"^(?P<orientation>[FR])(?P<index>\d+)_(?P<start>\d+)_(?P<end>\d+)"
    r"_(?P<fragment>\d+_\d+)_(?P<gene>\d+)$"
)
#: The constant flanks of a Table S6 oligo; the 20 nt spacer sits between them.
OLIGO_PREFIX = "GCTCAGTCCTAGGTATAATACTAGTA"
OLIGO_SUFFIX = "GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAG"
SPACER_LENGTH = 20
#: A released Start-End window: the spacer plus its PAM.
WINDOW_LENGTH = SPACER_LENGTH + 3

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(sequence: str) -> str:
    """Reverse complement of an unambiguous DNA string."""
    return sequence[::-1].translate(_COMPLEMENT)


def _expect_header(got: Sequence[Any], want: Sequence[str], what: str) -> None:
    """Refuse a sheet whose header row is not the expected one."""
    if list(got) != list(want):
        raise TableLayoutError(f"{what} header {list(got)} is not {list(want)}")


def read_table_s1(path: str | Path) -> pd.DataFrame:
    """Table S1 as one row per guide: the five identifier columns plus every sample.

    The sheet's header spans four rows: a title, then the identifier names beside the
    ``gRNAs per million mapped reads`` group label, then the ``w/o dCas9 (lib)`` and
    ``w/ dCas9 (stress)`` sub-groups, then the sample labels. The identifier names are
    read from row 1 and the sample labels from row 3, and both are asserted against
    ``S1_ID_COLUMNS``, ``LIBRARY_COLUMNS`` and ``S1_CONDITION_LABELS``.
    """
    raw = pd.read_excel(path, sheet_name=0, header=None, engine="openpyxl")

    def row(index: int) -> list[str]:
        return [
            "" if value is None or pd.isna(value) else str(value)
            for value in raw.iloc[index]
        ]

    n_id = len(S1_ID_COLUMNS)
    identifiers, samples_row = row(1), row(3)
    _expect_header(identifiers[:n_id], S1_ID_COLUMNS, "Table S1 identifier")
    _expect_header(
        identifiers[n_id : n_id + 1], [S1_VALUE_GROUP], "Table S1 value-group"
    )
    samples = [
        f"{label}_{replicate}" for label in S1_CONDITION_LABELS for replicate in (1, 2)
    ]
    _expect_header(samples_row[n_id:], [*LIBRARY_COLUMNS, *samples], "Table S1 sample")
    frame = raw.iloc[4:].reset_index(drop=True)
    frame.columns = pd.Index([*S1_ID_COLUMNS, *samples_row[n_id:]])
    frame["Gene"] = frame["Gene"].astype(str)
    frame["Strand"] = frame["Strand"].astype(str)
    frame["Fragment"] = frame["Fragment"].astype(str)
    for column in ("Start", "End"):
        frame[column] = pd.to_numeric(frame[column]).astype(int)
    for column in (*LIBRARY_COLUMNS, *samples):
        frame[column] = pd.to_numeric(frame[column]).astype(float)
    repeated = frame[frame.duplicated(subset=["Start", "End"])]
    if len(repeated):
        raise TableLayoutError(
            f"Table S1 repeats {len(repeated)} (Start, End) windows, so a window is not "
            "a guide identity"
        )
    return frame


def read_table_s2(path: str | Path) -> pd.DataFrame:
    """Table S2 as one row per gene: the released symbol, b-number, guide count and ER."""
    raw = pd.read_excel(path, sheet_name=0, header=None, engine="openpyxl")
    identifiers = [
        "" if value is None or pd.isna(value) else str(value) for value in raw.iloc[1]
    ]
    conditions = [
        "" if value is None or pd.isna(value) else str(value) for value in raw.iloc[2]
    ]
    _expect_header(
        identifiers[: len(S2_ID_COLUMNS)], S2_ID_COLUMNS, "Table S2 identifier"
    )
    _expect_header(
        conditions[len(S2_ID_COLUMNS) :], S2_CONDITION_COLUMNS, "Table S2 ER"
    )
    frame = raw.iloc[3:].reset_index(drop=True)
    frame.columns = pd.Index([*S2_ID_COLUMNS, *S2_CONDITION_COLUMNS])
    frame["Name"] = frame["Name"].astype(str)
    frame["b number"] = frame["b number"].astype(str)
    frame["gRNA #"] = pd.to_numeric(frame["gRNA #"]).astype(int)
    for column in S2_CONDITION_COLUMNS:
        frame[column] = pd.to_numeric(frame[column]).astype(float)
    repeated = sorted(frame["Name"][frame["Name"].duplicated()])
    if repeated:
        raise TableLayoutError(f"Table S2 repeats gene symbols {repeated}")
    return frame


def read_table_s6(path: str | Path) -> pd.DataFrame:
    """Table S6 as one row per oligo, with the 20 nt spacer sliced from its flanks.

    Every oligo must carry the two constant flanks the Methods state, and every name
    must parse to an orientation, an index, a window and a CDS fragment.
    """
    raw = pd.read_excel(path, sheet_name=0, header=None, engine="openpyxl")
    _expect_header(
        ["" if pd.isna(v) else str(v) for v in raw.iloc[1]], S6_COLUMNS, "Table S6"
    )
    frame = raw.iloc[2:].reset_index(drop=True)
    frame.columns = pd.Index(S6_COLUMNS)
    frame["Sequence"] = frame["Sequence"].astype(str)
    oligo_length = len(OLIGO_PREFIX) + SPACER_LENGTH + len(OLIGO_SUFFIX)
    malformed = frame[
        ~frame["Sequence"].str.startswith(OLIGO_PREFIX)
        | ~frame["Sequence"].str.endswith(OLIGO_SUFFIX)
        | (frame["Sequence"].str.len() != oligo_length)
    ]
    if len(malformed):
        raise TableLayoutError(
            f"{len(malformed)} Table S6 oligos do not carry the stated 26 nt prefix, "
            f"20 nt spacer and 33 nt suffix, e.g. {malformed['Primer Name'].iloc[0]!r}"
        )
    parsed = [S6_NAME.match(str(name)) for name in frame["Primer Name"]]
    unparsed = [
        str(name)
        for name, match in zip(frame["Primer Name"], parsed, strict=True)
        if match is None
    ]
    if unparsed:
        raise TableLayoutError(f"Table S6 names do not parse: {unparsed[:5]}")
    frame["orientation"] = [m["orientation"] for m in parsed if m is not None]
    frame["start"] = [int(m["start"]) for m in parsed if m is not None]
    frame["end"] = [int(m["end"]) for m in parsed if m is not None]
    frame["fragment"] = [m["fragment"] for m in parsed if m is not None]
    frame["spacer"] = frame["Sequence"].str[
        len(OLIGO_PREFIX) : len(OLIGO_PREFIX) + SPACER_LENGTH
    ]
    repeated = frame[frame.duplicated(subset=["start", "end"])]
    if len(repeated):
        raise TableLayoutError(f"Table S6 repeats {len(repeated)} (start, end) windows")
    return frame


# --------------------------------------------------------------------------- #
# Fidelity checks against the release's own numbers
# --------------------------------------------------------------------------- #
class SpacerCheck(BaseModel):
    """How every joined spacer sits in the MG1655 window its row delimits."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_joined: int
    n_reverse_read: int
    """Spacer == reverse complement of the window's first 20 nt, PAM NGG."""
    n_forward_read: int
    """Spacer == the window's first 20 nt, PAM NGG."""
    n_both: int
    n_neither: int
    examples_neither: tuple[str, ...]


class ErCheck(BaseModel):
    """Reproduction of Table S2's released ER from Table S1's per-guide abundance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_cells: int
    n_genes: int
    n_conditions: int
    max_abs_difference: float
    tolerance: float
    excluded_genes: tuple[str, ...]
    """Genes carrying a guide absent from the initial library (an infinite ratio)."""


class CoverageCheck(BaseModel):
    """Reproduction of the paper's own initial-library coverage count."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_guide_rows: int
    n_absent_from_library: int
    n_present_in_library: int
    reported: int


def spacer_by_window(s6: pd.DataFrame) -> dict[tuple[int, int], str]:
    """``{(start, end): spacer}`` from Table S6, the join key Table S1 rows carry."""
    return {
        (int(start), int(end)): str(spacer)
        for start, end, spacer in zip(s6["start"], s6["end"], s6["spacer"], strict=True)
    }


def check_spacers(frame: pd.DataFrame, genome: BacterialGenome[Any]) -> SpacerCheck:
    """Check every joined spacer against the genomic window its row delimits.

    A row passes when the window is 23 nt and, in exactly one of its two orientations,
    its first 20 nt are the spacer and its last 3 are an NGG PAM. The build refuses a row
    that passes in neither orientation: spacer, coordinates and pinned assembly must
    agree, which is what proves the join is the right one.
    """
    sequence = str(next(iter(genome.fasta_dna.values())).seq)
    joined = frame[frame["spacer"].notna()]
    reverse = forward = both = 0
    neither: list[str] = []
    for start, end, spacer in zip(
        joined["Start"], joined["End"], joined["spacer"], strict=True
    ):
        window = sequence[int(start) - 1 : int(end)]
        if len(window) != WINDOW_LENGTH:
            neither.append(f"{start}-{end} window is {len(window)} nt")
            continue
        flipped = reverse_complement(window)
        reverse_ok = (
            flipped[:SPACER_LENGTH] == spacer
            and flipped[SPACER_LENGTH + 1 : WINDOW_LENGTH] == "GG"
        )
        forward_ok = (
            window[:SPACER_LENGTH] == spacer
            and window[SPACER_LENGTH + 1 : WINDOW_LENGTH] == "GG"
        )
        reverse += reverse_ok
        forward += forward_ok
        both += reverse_ok and forward_ok
        if not (reverse_ok or forward_ok):
            neither.append(f"{start}-{end} {spacer} vs {window}")
    check = SpacerCheck(
        n_joined=len(joined),
        n_reverse_read=reverse,
        n_forward_read=forward,
        n_both=both,
        n_neither=len(neither),
        examples_neither=tuple(neither[:10]),
    )
    if check.n_neither:
        raise TableLayoutError(
            f"{check.n_neither} Table S6 spacers are not the 20 nt of their released "
            f"MG1655 window with an NGG PAM, e.g. {check.examples_neither[:3]}"
        )
    return check


def check_enrichment_ratios(
    s1: pd.DataFrame, s2: pd.DataFrame, *, tolerance: float = ER_TOLERANCE
) -> ErCheck:
    """Reproduce Table S2's ER from Table S1 and refuse a disagreement.

    ER is the median over a gene's guides of (the condition's mean abundance / the
    initial library's mean abundance). Genes carrying a guide with zero library abundance
    are excluded: that guide's ratio is infinite, which moves an even-length median, and
    the paper does not state how it handled it.
    """
    library = s1[list(LIBRARY_COLUMNS)].mean(axis=1)
    excluded = tuple(sorted(set(s1.loc[library == 0, "Gene"])))
    keep = ~s1["Gene"].isin(excluded)
    released = s2.set_index("Name")
    worst = 0.0
    n_cells = 0
    for label, column in zip(S1_CONDITION_LABELS, S2_CONDITION_COLUMNS, strict=True):
        condition = s1[[f"{label}_1", f"{label}_2"]].mean(axis=1)
        ratio = pd.DataFrame(
            {"Gene": s1["Gene"][keep], "ratio": (condition / library)[keep]}
        )
        reproduced = ratio.groupby("Gene")["ratio"].median()
        difference = (reproduced - released[column].reindex(reproduced.index)).abs()
        n_cells += len(difference)
        worst = max(worst, float(difference.max()))
    check = ErCheck(
        n_cells=n_cells,
        n_genes=int(s1.loc[keep, "Gene"].nunique()),
        n_conditions=len(S1_CONDITION_LABELS),
        max_abs_difference=worst,
        tolerance=tolerance,
        excluded_genes=excluded,
    )
    if worst > tolerance:
        raise TableLayoutError(
            f"reproduced enrichment ratios differ from Table S2 by up to {worst:.6g}, "
            f"above {tolerance}"
        )
    return check


def check_library_coverage(s1: pd.DataFrame) -> CoverageCheck:
    """Reproduce the paper's "39,574 sgRNA sequences were present in the initial library"."""
    absent = int((s1[list(LIBRARY_COLUMNS)].mean(axis=1) == 0).sum())
    check = CoverageCheck(
        n_guide_rows=len(s1),
        n_absent_from_library=absent,
        n_present_in_library=len(s1) - absent,
        reported=REPORTED_LIBRARY_COVERAGE,
    )
    if check.n_present_in_library != check.reported:
        raise TableLayoutError(
            f"{check.n_present_in_library} guides have a non-zero initial-library "
            f"abundance; the paper reports {check.reported}"
        )
    return check


# --------------------------------------------------------------------------- #
# Identifiers: the released b-number to the pinned MG1655 locus
# --------------------------------------------------------------------------- #
class LocusResolutionError(ValueError):
    """Too few released b-numbers resolve to a locus of the pinned assembly."""


def canonical_symbol(genome: BacterialGenome[Any], tag: str) -> str:
    """The genome's own gene symbol for ``tag`` when it resolves back to ``tag``.

    One spelling per locus is what keeps a perturbation from splitting into two graph
    nodes, so the symbol comes from the pinned annotation and not from the release's Gene
    column. A locus with no symbol, or whose symbol resolves elsewhere, is named by its
    tag.
    """
    symbol = genome.genbank.loci[tag].symbol
    if symbol is None:
        return tag
    return symbol if genome.resolve_gene_name(symbol).systematic_name == tag else tag


class GeneIdentity(BaseModel):
    """One released gene symbol and the MG1655 locus its b-number resolves to."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    released_symbol: str
    b_number: str
    locus_tag: str
    symbol: str
    """The pinned annotation's own symbol, which is what records store."""
    start: int
    end: int
    strand: str
    released_guide_count: int
    """Table S2's own ``gRNA #`` for the gene."""


class GuideIdentity(BaseModel):
    """One Table S1 guide the build keeps, with everything a record needs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row: int
    locus_tag: str
    symbol: str
    spacer: str
    n_guides: int
    """Table S1 guides of this gene, which is what the record's ``n_guides`` carries."""
    control_abundance: float
    antibiotic_abundance: dict[str, float]


def resolve_genes(
    s2: pd.DataFrame, genome: BacterialGenome[Any], *, label: str
) -> tuple[dict[str, GeneIdentity], tuple[str, ...], LocusTagReconciliation]:
    """Map each released gene symbol to its MG1655 locus through its released b-number.

    Returns the per-symbol identities, the released symbols whose b-number resolves to no
    single locus, and the reconciliation report.
    """
    stored, report = reconcile_locus_tags(genome, s2["b number"], label=label)
    report.require_resolved(MIN_RESOLVED_FRACTION)
    loci = genome.genbank.loci
    unresolved: list[str] = []
    identities: dict[str, GeneIdentity] = {}
    for symbol, b_number, tag, count in zip(
        s2["Name"], s2["b number"], stored, s2["gRNA #"], strict=True
    ):
        if tag not in loci:
            unresolved.append(str(symbol))
            continue
        locus = loci[tag]
        identities[str(symbol)] = GeneIdentity(
            released_symbol=str(symbol),
            b_number=str(b_number),
            locus_tag=tag,
            symbol=canonical_symbol(genome, tag),
            start=locus.start,
            end=locus.end,
            strand=locus.strand,
            released_guide_count=int(count),
        )
    return identities, tuple(sorted(unresolved)), report


# --------------------------------------------------------------------------- #
# Retention bookkeeping
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One retention rule, the records it removed, and the items it removed them for."""

    rule: str
    scope: Literal["guide", "cell"]
    description: str
    n_guides: int
    n_records: int
    items: list[str] = []


class DropLog(BaseModel):
    """Every retention rule applied to a build, in the order they were applied."""

    dataset: str
    guide_rows: int
    antibiotics: int
    source_records: int
    kept_guides: int
    kept_records: int
    dropped_records: int
    rules: list[DropRule]


class IdentifierLedger(BaseModel):
    """What the identifier step did, written to ``preprocess/``."""

    dataset: str
    released_genes: int
    resolved_genes: int
    kept_genes: int
    reconciliation: LocusTagReconciliation
    released_symbol_differs_from_annotation: dict[str, str]
    """``{released symbol: the pinned annotation's symbol for the same locus}``."""
    released_guide_count_differs_from_table_s1: dict[str, dict[str, int]]
    """``{gene: {"table_s1": n, "table_s2": n}}`` where the two counts disagree."""


class RetentionResult(BaseModel):
    """The guides a build keeps and the per-rule drop lists."""

    model_config = ConfigDict(extra="forbid")

    kept: list[GuideIdentity]
    unresolved_b_number: list[str]
    outside_locus: list[str]
    strand_disagreement: list[str]
    missing_spacer: list[str]
    zero_control: int
    zero_antibiotic_cells: int
    guide_counts: dict[str, dict[str, int]]


def _guide_label(gene: str, start: int, end: int) -> str:
    """How a dropped guide is named in the drop log."""
    return f"{gene}:{start}-{end}"


def select_guides(
    s1: pd.DataFrame,
    spacers: Mapping[tuple[int, int], str],
    identities: Mapping[str, GeneIdentity],
    unresolved: Sequence[str],
) -> RetentionResult:
    """Apply the six retention rules to Table S1 and build the kept guides' identities.

    The rules run in the module docstring's order, so a guide appears under the first
    rule that removes it. The initial-library columns are not consulted: the stored
    statistic divides the antibiotic abundance by the untreated LB abundance.
    """
    unresolved_set = set(unresolved)
    counts = Counter(s1["Gene"])
    kept: list[GuideIdentity] = []
    dropped_unresolved: list[str] = []
    dropped_outside: list[str] = []
    dropped_strand: list[str] = []
    dropped_spacer: list[str] = []
    zero_control = 0
    zero_cells = 0
    # Read every column out as a typed Python list once. A per-cell ``.iat`` read is
    # typed as the whole pandas scalar union, which strict mypy refuses to hand to
    # ``int`` or ``float``, and 39,580 x 14 of them is the slow way besides.
    genes: list[str] = [str(value) for value in s1["Gene"].tolist()]
    starts: list[int] = [int(value) for value in s1["Start"].tolist()]
    ends: list[int] = [int(value) for value in s1["End"].tolist()]
    strands: list[str] = [str(value) for value in s1["Strand"].tolist()]
    control: list[float] = [
        float(value)
        for value in s1[[f"{CONTROL_LABEL}_1", f"{CONTROL_LABEL}_2"]]
        .mean(axis=1)
        .tolist()
    ]
    means: dict[str, list[float]] = {
        condition.s1_label: [
            float(value)
            for value in s1[[f"{condition.s1_label}_1", f"{condition.s1_label}_2"]]
            .mean(axis=1)
            .tolist()
        ]
        for condition in CONDITIONS
    }
    for row in range(len(s1)):
        gene = genes[row]
        start = starts[row]
        end = ends[row]
        item = _guide_label(gene, start, end)
        if gene in unresolved_set or gene not in identities:
            dropped_unresolved.append(item)
            continue
        identity = identities[gene]
        if start > identity.end or end < identity.start:
            dropped_outside.append(item)
            continue
        if strands[row] != identity.strand:
            dropped_strand.append(item)
            continue
        spacer = spacers.get((start, end))
        if spacer is None:
            dropped_spacer.append(item)
            continue
        if control[row] == 0.0:
            zero_control += 1
            continue
        abundance: dict[str, float] = {}
        for condition in CONDITIONS:
            value = means[condition.s1_label][row]
            if value == 0.0:
                zero_cells += 1
                continue
            abundance[condition.s1_label] = value
        kept.append(
            GuideIdentity(
                row=row,
                locus_tag=identity.locus_tag,
                symbol=identity.symbol,
                spacer=str(spacer),
                n_guides=counts[gene],
                control_abundance=control[row],
                antibiotic_abundance=abundance,
            )
        )
    return RetentionResult(
        kept=kept,
        unresolved_b_number=dropped_unresolved,
        outside_locus=dropped_outside,
        strand_disagreement=dropped_strand,
        missing_spacer=dropped_spacer,
        zero_control=zero_control,
        zero_antibiotic_cells=zero_cells,
        guide_counts={
            gene: {"table_s1": counts[gene], "table_s2": identity.released_guide_count}
            for gene, identity in identities.items()
            if counts[gene] != identity.released_guide_count
        },
    )


def drop_log(dataset: str, s1_rows: int, result: RetentionResult) -> DropLog:
    """The retention ledger, with every dropped record accounted for by a rule."""
    n_abx = len(CONDITIONS)
    rules = [
        DropRule(
            rule="b_number_is_not_in_the_mg1655_annotation",
            scope="guide",
            description="the released b-number of the guide's gene is no locus tag, "
            "symbol or synonym of GCA_000005845.2, or matches more than one locus, so "
            "no locus of the pinned assembly can be written for the knockdown",
            n_guides=len(result.unresolved_b_number),
            n_records=len(result.unresolved_b_number) * n_abx,
            items=result.unresolved_b_number,
        ),
        DropRule(
            rule="guide_window_does_not_overlap_the_resolved_locus",
            scope="guide",
            description="the row's released Start-End window lies outside the span of "
            "the locus its released b-number resolves to, so the stored gene identity "
            "would not be supported by the stored coordinates. The RefSeq NC_000913.3 "
            "spans the library was designed against are identical to the GenBank ones "
            "for every one of these, so it is not an annotation-version difference",
            n_guides=len(result.outside_locus),
            n_records=len(result.outside_locus) * n_abx,
            items=result.outside_locus,
        ),
        DropRule(
            rule="guide_strand_disagrees_with_the_resolved_locus",
            scope="guide",
            description="the window overlaps the locus but the row's released Strand is "
            "the opposite of the locus's, so the row and the resolved locus name "
            "different genes",
            n_guides=len(result.strand_disagreement),
            n_records=len(result.strand_disagreement) * n_abx,
            items=result.strand_disagreement,
        ),
        DropRule(
            rule="guide_has_no_spacer_in_table_s6",
            scope="guide",
            description="Table S6 releases no oligo for the row's (Start, End) window, "
            "so the guide has no spacer and therefore no strain identity in a guide "
            "library",
            n_guides=len(result.missing_spacer),
            n_records=len(result.missing_spacer) * n_abx,
            items=result.missing_spacer,
        ),
        DropRule(
            rule="abundance_is_zero_in_the_untreated_lb_control",
            scope="guide",
            description="both untreated-LB replicates of the guide are 0, so the log2 "
            "ratio has a zero denominator in all twelve conditions; a pseudocount would "
            "fabricate the value",
            n_guides=result.zero_control,
            n_records=result.zero_control * n_abx,
            items=[],
        ),
        DropRule(
            rule="abundance_is_zero_in_the_antibiotic",
            scope="cell",
            description="both replicates of this one antibiotic are 0 for a guide whose "
            "untreated-LB control is not, so the log2 ratio is negative infinity. This "
            "is the strongest depletion a pooled screen can show and it has no finite "
            "log2 value",
            n_guides=0,
            n_records=result.zero_antibiotic_cells,
            items=[],
        ),
    ]
    source_records = s1_rows * n_abx
    kept_records = sum(len(guide.antibiotic_abundance) for guide in result.kept)
    ledger = DropLog(
        dataset=dataset,
        guide_rows=s1_rows,
        antibiotics=n_abx,
        source_records=source_records,
        kept_guides=len(result.kept),
        kept_records=kept_records,
        dropped_records=source_records - kept_records,
        rules=rules,
    )
    accounted = sum(rule.n_records for rule in rules)
    if accounted != ledger.dropped_records:
        raise RuntimeError(
            f"drop accounting mismatch: rules total {accounted}, "
            f"{ledger.dropped_records} records missing from the build"
        )
    return ledger


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def _sample_unit_gap() -> ProvenanceGap:
    """The replicate TYPE: the paper says only "two replicate measurements"."""
    return ProvenanceGap(
        field="sample_unit",
        reason=ProvenanceGapReason.not_reported_by_primary,
        note="the Methods state 'two replicate measurements under a given condition' "
        "and never say whether a replicate is a separate culture or a re-sequencing of "
        "one, so the count is stored and the unit is not guessed",
    )


def _uncertainty_gap() -> ProvenanceGap:
    """Table S1 releases the two replicate abundances and no dispersion of the ratio."""
    return ProvenanceGap(
        field="environment_response_uncertainty",
        reason=ProvenanceGapReason.not_reported_by_primary,
        note="Table S1 releases the two replicate abundances of each condition and the "
        "paper reports no dispersion of the ratio they define; a sample SD computed from "
        "two values by this loader would be our statistic, not a released one",
    )


def log2_response(guide: GuideIdentity, label: str) -> float:
    """log2 of the guide's antibiotic abundance over its untreated-LB abundance."""
    return math.log2(guide.antibiotic_abundance[label] / guide.control_abundance)


def response_phenotype(value: float) -> EnvironmentResponsePhenotype:
    """One guide's log2 treatment-over-control abundance ratio."""
    return EnvironmentResponsePhenotype(
        measurement_type=MEASUREMENT_TYPE,
        assay_type=ASSAY_TYPE,
        environment_response=value,
        n_samples=N_REPLICATES,
        units=RESPONSE_UNITS,
        provenance_gaps=[_sample_unit_gap(), _uncertainty_gap()],
    )


def _duration_gaps(sampling_od600: float) -> list[ProvenanceGap]:
    """The harvest rule: a sampling OD600, not a wall time or a doubling count."""
    note = (
        "the screening cultures were inoculated at OD600 0.05 and harvested at the "
        f"condition's own sampling OD600 of {sampling_od600:.2f} (Table S3); the paper "
        "states no wall time and no doubling count"
    )
    return [
        ProvenanceGap(
            field=field, reason=ProvenanceGapReason.not_reported_by_primary, note=note
        )
        for field in ("duration_hours", "duration_generations")
    ]


def control_environment() -> Environment:
    """Untreated LB with the two selection agents: the denominator's own environment."""
    return Environment(
        media=CHOE2025_LB_SELECTION,
        temperature=Temperature(value=float(SOURCED_VALUES["temperature"].value)),
        aerobicity=str(SOURCED_VALUES["aerobicity"].value),
        provenance_gaps=_duration_gaps(CONTROL_SAMPLING_OD600),
    )


def environment(condition: ScreenCondition) -> Environment:
    """The screening medium plus one antibiotic at its Table S3 dose."""
    solvent = (
        Solvent(
            name="DMSO",
            percent=DMSO_PERCENT,
            compound=resolved_compound("dimethyl sulfoxide"),
        )
        if condition.dmso
        else None
    )
    return Environment(
        media=CHOE2025_LB_SELECTION,
        temperature=Temperature(value=float(SOURCED_VALUES["temperature"].value)),
        perturbations=[
            SmallMoleculePerturbation(
                compound=resolved_compound(condition.compound_label),
                concentration=Concentration(
                    value=condition.dose, unit=condition.unit, basis=DoseBasis.fixed
                ),
                solvent=solvent,
            )
        ],
        aerobicity=str(SOURCED_VALUES["aerobicity"].value),
        provenance_gaps=_duration_gaps(condition.sampling_od600),
    )


def knockdown_genotype(guide: GuideIdentity) -> Genotype:
    """The one CRISPRi knockdown this guide is, on its MG1655 locus tag."""
    return Genotype(
        perturbations=[
            BacterialCrisprInterferencePerturbation(
                systematic_gene_name=guide.locus_tag,
                perturbed_gene_name=guide.symbol,
                gene_namespace=GENE_NAMESPACE,
                crispr=CrisprConstruct(
                    effector=EFFECTOR,
                    guide_sequence=guide.spacer,
                    n_guides=guide.n_guides,
                ),
            )
        ]
    )


def build_experiment(
    dataset_name: str,
    guide: GuideIdentity,
    condition: ScreenCondition,
    genotype: Genotype,
    env: Environment,
) -> BacterialEnvironmentResponseExperiment:
    """The record of one guide under one antibiotic."""
    return BacterialEnvironmentResponseExperiment(
        dataset_name=dataset_name,
        genotype=genotype,
        environment=env,
        phenotype=response_phenotype(log2_response(guide, condition.s1_label)),
    )


def build_reference(
    dataset_name: str, genome_reference: AssemblyReferenceGenome
) -> BacterialEnvironmentResponseExperimentReference:
    """The same pool in untreated LB: the clone's ratio to itself, log2(1) = 0."""
    return BacterialEnvironmentResponseExperimentReference(
        dataset_name=dataset_name,
        genome_reference=genome_reference,
        environment_reference=control_environment(),
        phenotype_reference=response_phenotype(0.0),
    )


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class CrispriChemgenChoe2025Dataset(ExperimentDataset):
    """Choe 2025 per-guide CRISPRi abundance response to twelve antibiotics."""

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = "MG1655"

    def __init__(
        self,
        root: str = DATASET_ROOT_REL,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        ecoli_genome: EcoliK12Genome | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; ``ecoli_genome`` is injected by the build entry points."""
        self.ecoli_genome = ecoli_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialEnvironmentResponseExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialEnvironmentResponseExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Every consumed workbook, linked from the raw mirror."""
        return [raw.name for raw in RAW_FILES]

    def download(self) -> None:
        """Link each mirror file into ``raw/`` after checking the manifest and sha256.

        The mirror plus ``DATA_SHA256`` is canonical; the PMC bucket URL is retrieval
        metadata ``retrieve_raw_files`` re-runs, never a build input.
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        for raw in RAW_FILES:
            check_manifest_pin(
                raw.mirror_relpath,
                manifest_sha256(manifest, raw.mirror_relpath),
                raw.sha256,
            )
            src = raw_mirror_dir(data_root) / raw.mirror_relpath
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            link_verified(src, osp.join(self.raw_dir, raw.name), raw.sha256)
        log.info("Choe 2025 raw files linked into %s (sha256 verified)", self.raw_dir)

    def _raw(self, name: str) -> str:
        return osp.join(self.raw_dir, name)

    def _genome(self) -> EcoliK12Genome:
        """The injected genome, or the reference strain's default cache (a direct run)."""
        if self.ecoli_genome is None:
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        expected = BACTERIAL_ASSEMBLY_SETS[self.REFERENCE_STRAIN]
        if self.ecoli_genome.ASSEMBLY_SET != expected:
            raise ValueError(
                f"{type(self).__name__} needs the {expected} genome, got "
                f"{self.ecoli_genome.ASSEMBLY_SET}"
            )
        return self.ecoli_genome

    @post_process
    def process(self) -> None:
        """Parse the three workbooks into per-(guide, antibiotic) records; write LMDB."""
        verify_raw_files(self.raw_dir, DATA_SHA256)
        s1 = read_table_s1(self._raw(TABLE_S1_FILE))
        s2 = read_table_s2(self._raw(TABLE_S2_FILE))
        s6 = read_table_s6(self._raw(TABLE_S6_FILE))
        released_genes = set(s2["Name"])
        if set(s1["Gene"]) != released_genes:
            raise TableLayoutError(
                "Table S1's gene symbols are not Table S2's: "
                f"{len(set(s1['Gene']) - released_genes)} only in S1, "
                f"{len(released_genes - set(s1['Gene']))} only in S2"
            )

        genome = self._genome()
        coverage = check_library_coverage(s1)
        er = check_enrichment_ratios(s1, s2)
        identities, unresolved, reconciliation = resolve_genes(
            s2, genome, label=self.name
        )
        spacer_map = spacer_by_window(s6)
        joined = s1.assign(
            spacer=pd.Series(
                [
                    spacer_map.get((int(start), int(end)))
                    for start, end in zip(s1["Start"], s1["End"], strict=True)
                ],
                index=s1.index,
                dtype="object",
            )
        )
        spacers = check_spacers(joined, genome)
        result = select_guides(s1, spacer_map, identities, unresolved)
        ledger = drop_log(self.name, len(s1), result)

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self._write_ledgers(
            ledger, result, identities, reconciliation, coverage, er, spacers
        )

        reference = build_reference(
            self.name, assembly_reference(self.REFERENCE_STRAIN)
        )
        environments = {
            condition.s1_label: environment(condition) for condition in CONDITIONS
        }
        env_out, interned_env = self._open_write_lmdb(
            osp.join(self.processed_dir, "lmdb")
        )
        idx = 0
        with env_out.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for guide in tqdm(result.kept, desc="choe2025"):
                genotype = knockdown_genotype(guide)
                for condition in CONDITIONS:
                    if condition.s1_label not in guide.antibiotic_abundance:
                        continue
                    experiment = build_experiment(
                        self.name,
                        guide,
                        condition,
                        genotype,
                        environments[condition.s1_label],
                    )
                    txn.put(
                        f"{idx}".encode(),
                        self._intern_record(experiment, reference, PUBLICATION, itxn),
                    )
                    idx += 1
        env_out.close()
        interned_env.close()
        if idx != ledger.kept_records:
            raise RuntimeError(
                f"wrote {idx} records, the drop ledger expects {ledger.kept_records}"
            )
        log.info(
            "Choe 2025: wrote %d records from %d guides over %d loci; dropped %d "
            "unresolved, %d outside-locus, %d strand-disagreeing, %d spacerless, %d "
            "zero-control guides and %d zero-antibiotic cells",
            idx,
            len(result.kept),
            len({guide.locus_tag for guide in result.kept}),
            len(result.unresolved_b_number),
            len(result.outside_locus),
            len(result.strand_disagreement),
            len(result.missing_spacer),
            result.zero_control,
            result.zero_antibiotic_cells,
        )

    def _write_ledgers(
        self,
        ledger: DropLog,
        result: RetentionResult,
        identities: Mapping[str, GeneIdentity],
        reconciliation: LocusTagReconciliation,
        coverage: CoverageCheck,
        er: ErCheck,
        spacers: SpacerCheck,
    ) -> None:
        """The drop log, the identifier ledger and the three fidelity checks."""
        out = Path(self.preprocess_dir)
        (out / "dropped_records.json").write_text(ledger.model_dump_json(indent=2))
        identifier_ledger = IdentifierLedger(
            dataset=self.name,
            released_genes=reconciliation.unique_names,
            resolved_genes=len(identities),
            kept_genes=len({guide.locus_tag for guide in result.kept}),
            reconciliation=reconciliation,
            released_symbol_differs_from_annotation={
                identity.released_symbol: identity.symbol
                for identity in identities.values()
                if identity.released_symbol != identity.symbol
            },
            released_guide_count_differs_from_table_s1=result.guide_counts,
        )
        (out / "identifier_reconciliation.json").write_text(
            identifier_ledger.model_dump_json(indent=2)
        )
        (out / "extraction.json").write_text(
            json.dumps(
                {
                    "raw_sha256": DATA_SHA256,
                    "library_coverage": coverage.model_dump(),
                    "enrichment_ratio": er.model_dump(),
                    "spacer_window": spacers.model_dump(),
                },
                indent=2,
                default=list,
            )
        )

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled by ``build_experiment``."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Verification (L0-L4) of a built LMDB
# --------------------------------------------------------------------------- #
VERIFIER_PROVENANCE = Provenance(
    source_uri=(
        f"$DATA_ROOT/{RAW_DIR_REL}/data/{TABLE_S1_FILE} (Table S1, per-guide abundance) "
        f"+ data/{TABLE_S2_FILE} (Table S2, the b-number map and the ER oracle) "
        f"+ data/{TABLE_S6_FILE} (Table S6, the guide spacers); raw mirror with "
        "manifest.json, pmc_cloud retrieval re-verified 2026-10-07"
    ),
    citation_key=CITATION_KEY,
    sha256=RAW_FILES_BY_NAME[TABLE_S1_FILE].sha256,
    method=(
        "One BacterialEnvironmentResponseExperiment per (Table S1 guide, antibiotic). "
        "measurement_type=log2_ratio, assay_type=pooled_competitive_growth_barcode, "
        "environment_response = log2(a_abx / a_LB) where a_c is the guide's mean "
        "abundance (reads per million mapped reads) over condition c's two replicate "
        "samples, the paper's own definition of an sgRNA's abundance. The initial-library "
        "columns cancel out of this ratio (log2(ER_abx/ER_LB) = log2(a_abx/a_LB)) and are "
        "used only for the build's ER reproduction check. n_samples=2 with sample_unit a "
        "typed gap (the Methods say only 'two replicate measurements'); the ratio's "
        "dispersion is a typed gap. Genotype = BacterialCrisprInterferencePerturbation on "
        "the MG1655 locus the released b-number resolves to, effector dCas9, "
        "guide_sequence = the 20 nt Table S6 spacer joined by the row's released genomic "
        "window, n_guides = the gene's Table S1 guide count; the spacer is what keeps "
        "sibling guides of one gene distinct strains. Environment = LB Miller plus the "
        "35 ug/mL chloramphenicol and 100 ug/mL ampicillin of the screening cultures, at "
        "37 C, shaken, aerobic, plus the condition's antibiotic at its Table S3 dose "
        "(DoseBasis.fixed; sulfamethizole's 0.2 mg/ml stored as 200 ug/mL) with the 0.4% "
        "DMSO vehicle typed for the six rows that state it. No inducer is asserted (dCas9 "
        "is under a tetracycline-inducible promoter and no dose is given for the screen); "
        "duration_hours and duration_generations are typed gaps carrying the condition's "
        "sampling OD600. Reference = the same clone in untreated LB, log2(1) = 0. Build "
        "checks: every spacer is the 20 nt of its released MG1655 window with an NGG PAM; "
        "Table S2's ER reproduced to 5.0e-5 over 4,192 genes x 13 conditions; the paper's "
        "own 39,574 library-present guides reproduced. DROPPED: 17 guides on an "
        "unresolvable b-number, 301 whose window misses the resolved locus (IS families "
        "and the Qin prophage cluster), 12 with the opposite strand, 1 with no Table S6 "
        "spacer, 283 with a zero untreated-LB control, and 1,023 (guide, antibiotic) "
        "cells with a zero antibiotic abundance (negative infinity, never a pseudocount)"
    ),
    page=(
        "iScience 2025 28:112435 (doi:10.1016/j.isci.2025.112435); Table S1 "
        f"sha256={RAW_FILES_BY_NAME[TABLE_S1_FILE].sha256}; Table S2 "
        f"sha256={RAW_FILES_BY_NAME[TABLE_S2_FILE].sha256}; Table S6 "
        f"sha256={RAW_FILES_BY_NAME[TABLE_S6_FILE].sha256}; paper.md "
        f"sha256={PAPER_MD_SHA256}; si/si1.md sha256={SI1_MD_SHA256}"
    ),
    retrieved=RETRIEVED_AT,
)


def stored_tags_are_loci(
    records: Iterable[Mapping[str, Any]], genome: BacterialGenome[Any]
) -> LevelResult:
    """SUPPLEMENTARY L1: every stored tag resolves to itself as a locus of the genome.

    The shared ``canonical_gene_names`` rule requires status ``current``, which a
    pseudogene locus never has (the bacterial resolver returns ``non_gene_feature``,
    naming the same tag). This row accepts a gene or a pseudogene locus that resolves to
    itself and counts the pseudogenes. It is added beside the shared row, never in its
    place.
    """
    tags = sorted(
        {
            str(perturbation["systematic_gene_name"])
            for record in records
            for perturbation in record["experiment"]["genotype"]["perturbations"]
        }
    )
    statuses: Counter[str] = Counter()
    elsewhere: list[str] = []
    for tag in tags:
        resolution = genome.resolve_gene_name(tag)
        statuses[str(resolution.status.value)] += 1
        if resolution.systematic_name != tag or resolution.status not in (
            GeneNameStatus.CURRENT,
            GeneNameStatus.NON_GENE_FEATURE,
        ):
            elsewhere.append(tag)
    return LevelResult(
        level=Level.L1,
        name="stored_tags_are_loci_of_the_pinned_assembly",
        passed=not elsewhere,
        message=(
            f"SUPPLEMENTARY: {len(tags)} stored tags, statuses {dict(statuses)}; "
            f"{len(elsewhere)} do not resolve to themselves"
        ),
        details={"statuses": dict(statuses), "not_a_locus": elsewhere[:20]},
    )


def run_verification(data_root: str | None = None) -> VerificationReport:
    """Run the environment-response family verifier (L0-L4) on the built dev LMDB.

    The gene universe and the resolver are MG1655's, the host the records are written
    against; the dataset is streamed, so 466,569 records are never materialized. Two
    supplementary rows are appended: the pseudogene-tolerant locus check and the
    provenance audit of every ``SOURCED_VALUES`` entry. The report is written to
    ``preprocess/verification_report.json``.
    """
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset_streaming,
    )
    from torchcell.verification.runners import (
        _gene_set_for_reference,
        _genome_for_reference,
        _write_report,
        stream_records,
    )

    base = data_root or _data_root()
    abs_root = osp.join(base, DATASET_ROOT_REL)
    drops = DropLog.model_validate_json(
        Path(abs_root, "preprocess", "dropped_records.json").read_text()
    )
    first = next(iter(stream_records(abs_root)))
    reference = first["reference"]["genome_reference"]
    genome = _genome_for_reference(reference, base)
    report = verify_environment_response_dataset_streaming(
        stream_records(abs_root),
        dataset_name=CrispriChemgenChoe2025Dataset.__name__,
        provenance=VERIFIER_PROVENANCE,
        expected_count=drops.kept_records,
        sgd_genes=_gene_set_for_reference(reference, base),
        resolve_gene_name=genome.resolve_gene_name,
    )
    report.add(stored_tags_are_loci(stream_records(abs_root), genome))
    library = Path(base) / "torchcell-library"
    for value in SOURCED_VALUES.values():
        report.add(audit_sourced_value(value, library))
    for condition in CONDITIONS:
        report.add(audit_sourced_value(condition.dose_sourced, library))
    report.add(audit_sourced_value(CONTROL_ROW, library))
    _write_report(report, osp.join(abs_root, "preprocess"))
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI: ``deposit`` the raw mirror, ``build`` the dev LMDB, or ``verify`` it."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(prog="python -m torchcell.datasets.ecoli.choe2025")
    sub = parser.add_subparsers(dest="command", required=True)
    deposit = sub.add_parser(
        "deposit", help="retrieve (optional) and deposit the raw mirror"
    )
    deposit.add_argument("--download-dir", required=True)
    deposit.add_argument(
        "--retrieve",
        action="store_true",
        help="run the recorded PMC bucket retrievers into --download-dir first",
    )
    sub.add_parser("build", help="build (or load) the dev-tree LMDB")
    sub.add_parser("verify", help="run L0-L4 on the built dev-tree LMDB")
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = _data_root()
    if args.command == "deposit":
        download = Path(args.download_dir)
        if args.retrieve:
            retrieve_raw_files(download)
        sources: dict[str, str | Path] = {name: download / name for name in DATA_SHA256}
        print(deposit_raw_mirror(sources=sources, data_root=data_root))
        return 0
    if args.command == "build":
        dataset = CrispriChemgenChoe2025Dataset(
            root=osp.join(data_root, DATASET_ROOT_REL)
        )
        print(f"len = {len(dataset)}")
        return 0
    report = run_verification(data_root)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
