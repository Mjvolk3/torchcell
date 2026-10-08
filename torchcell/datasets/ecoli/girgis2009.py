# torchcell/datasets/ecoli/girgis2009
# [[torchcell.datasets.ecoli.girgis2009]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/girgis2009
# Test file: tests/torchcell/datasets/ecoli/test_girgis2009.py
"""Girgis 2009: the 17-antibiotic transposon selection of E. coli K-12 MG1655.

Girgis, Hottes and Tavazoie 2009 (PLoS ONE 4(5):e5629, doi:10.1371/journal.pone.0005629,
PMC2680486, citation key ``girgisGeneticArchitectureIntrinsic2009``) propagated a library
of about 5 x 10^5 single-insertion transposon mutants for several days in each of 17
antibiotics at concentrations that impaired but did not abolish growth of the parent, then
read the surviving population by microarray genetic footprinting: the transposon-adjacent
genomic DNA was amplified, labeled and hybridized against labeled genomic DNA on spotted
arrays. Each locus therefore gets a signed per-drug score saying whether disrupting it
helped or hurt in that drug.

RECORD = one (gene x antibiotic) ``BacterialEnvironmentResponseExperiment``, from Dataset
S5, "Combined Z-scores for all loci". 3,976 released loci x 17 antibiotics = 67,592 cells;
63,766 are stored (the arithmetic is in RETENTION below).

PHENOTYPE. ``EnvironmentResponsePhenotype``, ``measurement_type=z_score``,
``assay_type=other``. The number is the COMBINED z-score: a z-score is computed for every
(hybridization x reference set) pair, where ``z = (x - mu) / sigma``, ``x = log2(r)``, ``r``
is the normalized transposon/genomic-DNA ratio and ``mu``/``sigma`` come from one of two
reference sets (five hybridizations of the unselected library, six selections in the same
medium without antibiotics); when every z-score of a drug agrees in sign the locus is
assigned the one CLOSEST TO ZERO, and when they disagree it is assigned 0, "indicating no
consistent fitness effect". A positive score means the disruption is beneficial in that
drug (``SOURCED_VALUES["sign_convention"]``).

WHY NOT ``FitnessPhenotype``. The score is signed and routinely negative: over the 63,766
stored records, 9,600 are positive, 11,523 negative and 42,643 exactly zero.
``FitnessPhenotype`` is a strictly positive ko/wt ratio that CLAMPS non-positive values and
whose verifier wants a 1.0 reference, so it would erase half the measurements and all of
the zeros. The zeros are themselves measured statements, not missing data: the combination
rule assigns 0 when the z-scores disagree in sign, which is the paper's way of saying the
effect is not reproducible. A cell with too few usable hybridizations is released as ``ND``
instead and is not a record.

WHY ``measurement_type=z_score`` AND ``assay_type=other``. ``z_score`` is exactly what the
Methods compute. ``AssayType`` has no member for microarray genetic footprinting: the
readout is a two-color hybridization of transposon-adjacent DNA to spotted arrays, so it is
pooled competitive growth but NOT ``pooled_competitive_growth_barcode`` (there is no
molecular barcode, the insertion's own genomic position is what hybridizes), and it is
neither a colony array nor a liquid growth curve. ``other`` with the readout spelled out in
``units`` is the honest typing; the PR records that a ``genetic_footprinting_microarray``
member would be more precise.

CONDITIONS. The 17 drugs of Table 1, each as a ``SmallMoleculePerturbation`` at its printed
ug/mL dose on one shared medium, with ``screen_id`` = ``"<code> day <day>"`` (the drug code
of the released column plus Table 1's selection day verbatim, e.g. ``"dox day 2/3"``). The
dose and the day are the condition's identity and both come from that drug's own Table 1
row, quoted verbatim in :data:`DRUGS`.

``n_samples`` is the number of independent replicate SELECTIONS hybridized for that drug
(``sample_unit=biological_replicate``): 3 for ampicillin, lomefloxacin, sulfamonomethoxine
and doxycycline, 2 for the other 13. It is NOT read off Table 1's "# Samples" column, which
disagrees with the released data for two drugs; it is derived from the release and checked
two ways at build time (see TABLE 1 DISAGREEMENT).

MEDIUM. One medium for all 17 conditions: M9 salts with 0.4% glucose, 0.1% casamino acids,
1 mM MgSO4, 0.1 mM CaCl2 and 1.5 uM thiamine, at 37 C, shaken and aerobic. It derives from
the ``M9`` library key, so it joins there, but the four salts carry ``concentration=None``:
the paper states "M9 salts [80]" by reference to Ausubel's Current Protocols and prints no
amounts, and the ``M9`` library object's amounts are Borchert 2024's and Kang 2026's, so
asserting them here would fabricate numbers this paper never gave. Every condition's base
medium has a library entry, so no condition is dropped for a medium reason. Casamino acids
is an acid hydrolysate of casein with no structure to resolve
(``ComponentDefinition.intrinsically_undefined``), which is why ``is_synthetic`` is False.
Nine of the 17 antibiotics are not in the curated compound table (amikacin, ampicillin,
cefoxitin, doxycycline hyclate, fusidic acid, gentamycin, nitrofurantoin, piperacillin,
streptomycin), so ``resolved_compound`` keeps the paper's label and attaches a typed gap on
``inchikey``; the table is not edited here.

STRAIN. MG1655 (``REFERENCE_STRAIN``), pinned to GCA_000005845.2. The library's parent is
MG1655 ``delta-lacZ``, carried as a ``BacterialStrainBackground`` whose
``genotype_statement`` is the paper's own string; ``alleles`` is empty and that is a
statement, not an omission, because neither the paper nor Text S1 says whether the lacZ
lesion is a full deletion, an internal one or a cassette replacement, and ``AlleleEdit``
has no member for an unspecified edit.

GENOTYPE. One ``TransposonInsertionPerturbation`` per record, at the GENE level: a spot on
the array stands for every mutant whose transposon-adjacent DNA hybridizes to that gene, so
``barcode``, ``insertion_position`` and ``insertion_strand`` are all None, and so is
``transposon`` -- this paper names no transposon, deferring the library and the footprinting
method to Girgis et al. 2007 (PLoS Genet 3:1644), which is not in the mirror. The leaf has
no ``provenance_gaps`` field, so those four absences are typed in
:data:`PERTURBATION_FIELD_GAPS` and written to ``preprocess/perturbation_field_gaps.json``.
The score is also not strictly a property of the gene: the legend of every heatmap figure
says "transposon insertions in or near a gene", so a spot's signal can include insertions in
the neighbourhood (``SOURCED_VALUES["gene_level"]``).

IDENTIFIERS. ``UNIQID`` holds 2009-vintage MG1655 b-numbers, every one a lowercase b
followed by four digits (:data:`B_NUMBER_PATTERN`).
Against the pinned GCA_000005845.2 annotation 3,821 of the 3,976 ARE locus tags (3,758
current genes, 63 pseudogene loci) and are stored as released, so no record carries a
``DerivedIdentifierMapping``. The other 155 are dropped (see RETENTION).

RETENTION (rules, counts and items in ``preprocess/dropped_records.json``).

1. ``b_number_is_not_a_locus_tag_of_the_pinned_annotation`` -- 155 loci x 17 drugs = 2,635
   cells. 110 of those b-numbers the current annotation carries as a ``/gene_synonym`` of a
   DIFFERENT locus (39 of a current gene, 71 of a pseudogene) and 45 it does not carry at
   all. Storing the merged locus needs a ``DerivedIdentifierMapping`` and
   ``DerivedIdentifierRoute`` has no member for a retired tag of the pinned strain's OWN
   namespace (issue #753), and storing the released tag would put a record on a locus the
   assembly does not have, so the rows are dropped and each is ledgered with the
   annotation's own resolution note.
2. ``no_combined_score_released`` -- 1,191 cells of the kept loci whose Dataset S5 entry is
   ``ND``, which the sheet defines as fewer than two usable repetitions, so the paper
   assigned no score.

2,635 + 1,191 = 3,826 dropped; 67,592 - 3,826 = 63,766 records.

TABLE 1 DISAGREEMENT. Table 1 prints "# Samples" 3 for streptomycin and 2 for
sulfamonomethoxine, and the released data says the opposite. Datasets S2 to S4 carry two STR
hybridization columns and three SLF columns; Table 1's own day footnote marks SLF (with DOX
and LOM) as "Two samples from day 2 and one from day 3"; and Text S1 sets the significance
cut at 2.15 for a drug hybridized twice and 1.5 for one hybridized three times, which is
what the released Dataset S1 shows (STR's smallest significant |z| is 2.1505 with nothing
below 2.15; SLF has 8 significant loci below 2.15, the smallest 1.5246). The release wins:
``n_samples`` comes from the hybridization columns, corrected for the four drugs whose
repetitions 2 and 3 Text S1 says were technical and averaged, and ``process()`` refuses a
build whose Dataset S1 values disagree with the threshold that count implies. The two
printed cells look transposed between the STR and SLF rows; both disagreements are recorded
in ``preprocess/replicate_structure.json``.

UNCERTAINTY. None exists per record. Datasets S3 and S4 release a per-gene ``Stdev`` column,
but that is the dispersion of the REFERENCE hybridizations, which is already the denominator
of the z-score and carries a heuristic global component of 0.15 added to every standard
deviation; it is not an error bar on the response. The combined score is additionally a
MINIMUM over the drug's z-scores, a deliberately conservative order statistic with no
released spread. ``environment_response_uncertainty`` and ``environment_response_se`` carry
``not_reported_by_primary`` gaps.

BUILD-TIME CHECKS (no fallbacks; each raises). Every sheet's title cell and header row must
read as expected; the four sheets must carry the same 3,976 UNIQIDs in the same order with
the same descriptions; every non-zero Dataset S1 value must equal its Dataset S5 cell and
clear its drug's threshold; and every Dataset S5 value must be reproducible from Datasets S3
and S4 by the paper's own combination rule. The last check passes on 66,240 of the 66,287
numeric cells exactly; the other 47 are cells where the rule gives ``|z| < 0.001`` and the
sheet prints 0, which ``MAX_ROUNDED_TO_ZERO`` bounds.

REFERENCE. One per condition: the unperturbed MG1655 ``delta-lacZ`` parent in the same drug,
scoring 0, which on this axis is "no consistent fitness effect" relative to the reference
hybridizations.
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
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Final

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
from torchcell.datamodels.media import M9
from torchcell.datamodels.schema import (
    BACTERIAL_ASSEMBLY_SETS,
    AssayType,
    AssemblyReferenceGenome,
    BacterialAssemblySet,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    BacterialStrainBackground,
    ComponentDefinition,
    Concentration,
    ConcentrationUnit,
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
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
    TransposonInsertionPerturbation,
)
from torchcell.datasets.bacteria_common import (
    LAYER_LOCUS_TAG,
    LOCUS_TAG_PATTERNS,
    STRAIN_GENE_NAMESPACES,
    LocusTagReconciliation,
    assembly_reference,
    bacterial_genome,
    reconcile_locus_tags,
    resolution_layer,
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
from torchcell.sequence.genome.ecoli.k12 import EcoliK12Genome, EcoliK12StrainName
from torchcell.verification.report import Provenance, VerificationReport
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
    audit_sourced_value,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "girgisGeneticArchitectureIntrinsic2009"
PAPER_DOI = "10.1371/journal.pone.0005629"
PMCID = "PMC2680486"
#: The PubMed id of doi:10.1371/journal.pone.0005629, read from the NCBI E-utilities
#: ``esearch`` on that DOI and confirmed by ``esummary`` on 2026-10-07.
PUBMED_ID = "19462005"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
LIBRARY_DIR_REL = f"torchcell-library/{CITATION_KEY}"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "974add90e9f1737adb55ef2e4131c1eaf1d1a0931d03e34962a1d99e60afac0d"
TEXT_S1_MD = "si/si1.md"
TEXT_S1_MD_SHA256 = "36a4fec61377744520f578041252c16b1f9aa9829e57366fbff1385717082dfe"

DATASET_S1 = "si20.xls"
DATASET_S3 = "si22.xls"
DATASET_S4 = "si23.xls"
DATASET_S5 = "si24.xls"
#: When the literature mirror captured the SI (copied from its ``manifest.json``).
DATA_RETRIEVED_AT = "2026-10-07T11:42:35.675278+00:00"


class RawFile(BaseModel):
    """One file the loader consumes: its pinned bytes and how they were retrieved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    sha256: str
    bytes: int
    description: str
    retrieval: RetrievalRecord

    @property
    def mirror_relpath(self) -> str:
        """Path inside the raw mirror (``data/<name>``)."""
        return f"data/{self.name}"


def _pmc_file(name: str, obj: str, sha256: str, size: int, description: str) -> RawFile:
    """A publisher SI file of the PMC Article Datasets bucket (``pmc_cloud``)."""
    key = f"{PMCID}.1/{obj}"
    return RawFile(
        name=name,
        sha256=sha256,
        bytes=size,
        description=description,
        retrieval=RetrievalRecord(
            method=RetrievalMethod.pmc_cloud,
            source_url=f"https://pmc-oa-opendata.s3.amazonaws.com/{key}",
            retriever="torchcell.literature.retrieve.pmc_cloud_object",
            params={"key": key},
            sha256=sha256,
            retrieved_at=DATA_RETRIEVED_AT,
        ),
    )


RAW_FILES: tuple[RawFile, ...] = (
    _pmc_file(
        DATASET_S1,
        "pone.0005629.s020.xls",
        "b6c4daed3db9cbdd6e9683e5ace995467cac79cacc8ca3c7a8f181675e7edf38",
        902144,
        "Dataset S1: the combined z-scores that cleared each drug's significance "
        "threshold, every other cell zeroed; read only to prove which threshold, and so "
        "how many hybridizations, each drug had",
    ),
    _pmc_file(
        DATASET_S3,
        "pone.0005629.s022.xls",
        "7f190bec346242ce0d883ad22803d0eca7f09e350fee994f281f51f66915ee21",
        3575808,
        "Dataset S3: per-hybridization z-scores against the five hybridizations of the "
        "unselected library; the first of the two reference sets the combined score "
        "minimizes over",
    ),
    _pmc_file(
        DATASET_S4,
        "pone.0005629.s023.xls",
        "6e76fc1c450cf9f63f1a5ecb06c9e7caeddcbdbe099458704a964ac4cd8c7d8b",
        3580416,
        "Dataset S4: per-hybridization z-scores against the six no-antibiotic "
        "selections; the second reference set",
    ),
    _pmc_file(
        DATASET_S5,
        "pone.0005629.s024.xls",
        "823c6aaffe26a2419a0fa5bbf0b0e72dc6cd14a390995a28696011a5e1ab9da5",
        1237504,
        "Dataset S5: the combined z-score of every locus in every one of the 17 "
        "antibiotics; the values this loader stores",
    ),
)
#: ``{raw file name: pinned sha256}``, the build-time check of every consumed file.
DATA_SHA256: dict[str, str] = {f.name: f.sha256 for f in RAW_FILES}
RAW_FILES_BY_NAME: dict[str, RawFile] = {f.name: f for f in RAW_FILES}

#: Released data deliberately not mirrored (the loader does not read it).
NOT_MIRRORED = (
    "Dataset S2 (pone.0005629.s021.xls): the normalized transposon/genomic-DNA ratios "
    "the z-scores were computed from; the loader stores the combined z-score and "
    "reconstructs it from Datasets S3 and S4, so the ratios are never read",
    "Text S1 and Figures S1-S11 and Tables S1-S7 (pone.0005629.s001-s019.pdf): the "
    "additional methods, the gel and dose-response images, the per-class heatmaps and "
    "the MIC tables. Text S1 is quoted from its OCR in the LITERATURE mirror "
    "(si/si1.md), which is where every SourcedValue anchored to it points; none of them "
    "is a build input",
)


def _paper(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to a verbatim quote in the sha256-pinned Girgis ``paper.md``."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
        ),
    )


def _text_s1(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to a verbatim quote in the pinned OCR of Text S1."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=TEXT_S1_MD,
            citation_key=CITATION_KEY,
            sha256=TEXT_S1_MD_SHA256,
            method="MinerU OCR of Text S1, Additional Materials and Methods "
            "(torchcell-library mirror)",
            page="Text S1 (pone.0005629.s001.pdf)",
        ),
    )


# --------------------------------------------------------------------------- #
# Sourced values (verbatim quotes, pinned sha256)
# --------------------------------------------------------------------------- #
_Q_STRAIN_AND_MEDIUM = (
    "All experiments were performed using $E$ . coli MG1655 [79]. Transposon insertion "
    "mutants were generated in a MG1655 DlacZ strain as described in a previous study "
    "[25]. All experiments were conducted in M9 salts [80] supplemented with "
    "$0 . 4 \\%$ glucose, $0 . 1 \\%$ casamino acids, 1 mM "
    "$\\mathrm { M g S O _ { 4 } }$ , $0 . 1 \\ \\mathrm { m M } \\ \\mathrm { C a C l "
    "_ { 2 } }$ , and $1 . 5 ~ \\mu \\mathrm { M }$ thiamine."
)
_Q_COMBINATION_RULE = (
    "To identify the most reproducible fitness effects, we considered all of the "
    "z-scores for each gene for a given antibiotic. (Antibiotics with two and three "
    "hybridizations had four and six z-scores, respectively.) When all of the "
    "$\\mathbf { Z }$ -scores had the same sign, we assigned the gene the "
    "$\\mathbf { Z }$ -score in the set that was closest to zero (representing the "
    "smallest effect). When a gene had $\\mathbf { Z }$ -scores of different signs, the "
    "gene was assigned a score of 0, indicating no consistent fitness effect."
)
_Q_THRESHOLDS = (
    "Based on the distribution, a cutoff of 2.15 (positive or negative) was chosen as "
    "the threshold for antibiotics for which we had two hybridizations. For those "
    "antibiotics with three hybridizations, the same technique was used except that two "
    "samples from one reference set and one from the other were designated as data, "
    "giving a cutoff of 1.5 for two false positives per antibiotic."
)

SOURCED_VALUES: dict[str, SourcedValue] = {
    "reference_strain": _paper(
        "MG1655",
        _Q_STRAIN_AND_MEDIUM,
        note="the strain every experiment was done in; pinned to GCA_000005845.2, the "
        "assembly of the ecoli_K12_MG1655_ASM584v2 set, whose locus tags are the "
        "b-numbers Dataset S5 keys on",
    ),
    "library_parent": _text_s1(
        "MG1655 delta-lacZ",
        "the library’s parental strain (MG1655 ∆lacZ), were pelleted and "
        "washed with M9 media.",
        note="the unperturbed reference of every record: the strain the transposon "
        "library was built in. The paper writes the same strain as 'MG1655 DlacZ' in "
        "its Methods (SOURCED_VALUES['library_construction'])",
    ),
    "library_construction": _paper(
        "a MG1655 DlacZ strain",
        "Transposon insertion mutants were generated in a MG1655 DlacZ strain as "
        "described in a previous study [25].",
        note="the OCR renders the delta as 'D'; Text S1 writes it as a delta sign. No "
        "allele is typed from it: neither source says whether the lesion is a full "
        "deletion, an internal one or a cassette replacement",
    ),
    "medium": _paper(
        "M9 salts with 0.4% glucose, 0.1% casamino acids, 1 mM MgSO4, 0.1 mM CaCl2 and "
        "1.5 uM thiamine",
        _Q_STRAIN_AND_MEDIUM,
        note="one medium for all 17 conditions. The four M9 salts carry no amount: the "
        "paper cites Ausubel's Current Protocols [80] for them and prints none",
    ),
    "temperature_c": _paper(
        37.0,
        "Unless otherwise noted, cultures were shaken at $3 7 ^ { \\circ } \\mathrm { C "
        "}$ .",
    ),
    "aerobicity": _paper(
        "aerobic",
        "Using an aerobic environment – a condition where many antibiotics, "
        "especially aminoglycosides, are particularly effective [32] – likely "
        "increased the number and types of beneficial mutations identified.",
        note="shaken cultures; the Results name the regime explicitly because it is "
        "what makes the aminoglycoside hits appear",
    ),
    "library_size": _paper(
        500_000,
        "An aliquot of a library containing ${ \\sim } 5 \\times 1 0 ^ { 5 }$ mutants "
        "each with a single transposon insertion [25] was taken from frozen stock, "
        "grown overnight in LB, pelleted, washed, and resuspended at $2 \\%$ inoculum "
        "in fresh M9-media containing an antibiotic at the chosen concentration "
        "(Table 1).",
        note="one insertion per mutant, which is why a record carries exactly one "
        "TransposonInsertionPerturbation",
    ),
    "subinhibitory_doses": _paper(
        "concentrations that impaired but did not completely inhibit growth",
        "we used antibiotic concentrations that impaired but did not completely inhibit "
        "the growth of the wildtype strain (Table 1).",
        note="why the doses of DRUGS are what they are; Figure S1 holds the "
        "dose-response curves they were read off",
    ),
    "serial_transfer": _paper(
        "2% of the culture transferred daily",
        "Each day, an aliquot was frozen, and $2 \\%$ of the culture was transferred to "
        "fresh media to continue the selection.",
        note="the selection is dated in DAYS of transfer and no hours or generations "
        "per day are stated, which is why Environment.duration_hours is gapped and the "
        "day lives in the phenotype's screen_id",
    ),
    "assay": _paper(
        "microarray genetic footprinting",
        "Genetic footprinting and subsequent hybridization to DNA spotted arrays were "
        "performed as described in Girgis et al. [25].",
        note="the readout: transposon-adjacent genomic DNA amplified and hybridized "
        "against labeled genomic DNA on spotted arrays. AssayType has no member for it, "
        "so assay_type is `other` and units says what was measured",
    ),
    "gene_level": _paper(
        "insertions in or near a gene",
        "Yellow (blue) indicates that transposon insertions in or near a gene were "
        "beneficial (deleterious).",
        note="a record is one array spot, standing for every mutant whose "
        "transposon-adjacent DNA hybridizes to that gene, insertions NEAR it included; "
        "no single insertion site exists for it",
    ),
    "replicate_selections": _paper(
        2,
        "Samples from at least two independent replicate selections were hybridized for "
        "each antibiotic. As controls, six samples from independent selections in the "
        "absence of any drug were hybridized.",
        note="the floor, not the per-drug count: a sample is an independent SELECTION, "
        "which is why sample_unit is biological_replicate. The per-drug count comes from "
        "the released hybridization columns and is checked against the thresholds",
    ),
    "z_score_definition": _paper(
        "z = (log2(ratio) - mean) / sd of the reference hybridizations",
        "Two $\\mathbf { Z }$ -scores were calculated for each ratio, r, where "
        "$\\mathrm { { z } = ( x - \\mu ) / \\sigma }$ , $\\mathbf { \\sigma } _ { "
        "\\mathbf { X } } = \\log _ { 2 } ( \\mathbf { r } )$ , and $\\mu$ and "
        "$\\sigma$ are the mean and standard deviation, respectively, of the "
        "$\\log _ { 2 }$ ratios for the gene from reference hybridizations.",
        note="r is the normalized transposon/genomic-DNA ratio; the two reference sets "
        "are the five unselected-library hybridizations and the six no-antibiotic "
        "selections",
    ),
    "combination_rule": _paper(
        "the z-score closest to zero when all agree in sign, else 0",
        _Q_COMBINATION_RULE,
        note="what a Dataset S5 cell IS, and why a 0 is a measured statement rather "
        "than a missing value. process() reproduces every cell from Datasets S3 and S4 "
        "by this rule",
    ),
    "sign_convention": _paper(
        "positive = the disruption is beneficial in that drug",
        "For example, disruption of a locus was classified as beneficial in "
        "aminoglycosides if the gene had positive $\\mathbf { Z }$ -scores in all four "
        "drugs, and the $\\mathbf { Z }$ -scores reached the significance level for at "
        "least two drugs.",
    ),
    "technical_replicates_averaged": _text_s1(
        ("PIP", "FOX", "TET", "TRM"),
        "For pipercillin, cefoxitin, tetracycline, and trimethoprim, two separate "
        "hybridizations were done for one of the samples. The data from the two arrays "
        "was similar. Corresponding values were averaged and treated as a single "
        "repetition during subsequent analysis.",
        note="these four drugs have three hybridization columns in Datasets S3 and S4 "
        "but TWO repetitions: columns R2 and R3 are two arrays of one sample and are "
        "averaged before the combination rule, which is what the reconstruction check "
        "and n_samples both follow",
    ),
    "min_two_repetitions": _text_s1(
        2,
        "For a gene to be assigned a non-zero z-score, data from at least two "
        "repetitions was needed.",
        note="why a Dataset S5 cell can read ND: fewer than two usable repetitions, so "
        "no score was assigned. Those cells are dropped under "
        "`no_combined_score_released`",
    ),
    "significance_thresholds": _text_s1(
        {2: 2.15, 3: 1.5},
        _Q_THRESHOLDS,
        note="the cut Dataset S1 applied, keyed by the drug's repetition count. "
        "process() uses it as a CHECK on the per-drug count read off the hybridization "
        "columns: no significant |z| may fall below the implied cut, and a "
        "three-repetition drug must have at least one between 1.5 and 2.15",
    ),
    "false_positive_budget": _paper(
        2,
        "The significance threshold was set so that two false positives are expected "
        "per antibiotic. False positives were estimated by treating randomly chosen "
        "reference samples as data and repeating the analysis procedure. (See Text S1.)",
    ),
    "global_sd_component": _text_s1(
        0.15,
        "The value of 0.15 was chosen heuristically based on simulations of the number "
        "of false positives expected at significance thresholds that allowed the "
        "correct classification of loci known to be affected by the experimental "
        "perturbation. Smaller values gave unacceptably high false positive rates.",
        note="added to every reference standard deviation before dividing, so the "
        "Stdev column of Datasets S3 and S4 is a tuned denominator rather than a "
        "measured spread; it is not an uncertainty of the stored response",
    ),
    "ratio_floor": _text_s1(
        0.05,
        "Ratios smaller than 0.05 were set to 0.05.",
        note="a floor on the ratio before the log, so the most depleted mutants share "
        "one bounded z-score rather than running to minus infinity",
    ),
    "array_normalization": _text_s1(
        5000,
        "Finally, the ratio for each spot was calculated as the normalized transposon "
        "signal divided by the normalized genomic DNA signal. Ratios from the duplicate "
        "spots on each slide were averaged. To facilitate between array comparisons, "
        "arrays were normalized so that the sum of the ratios for the set of genes "
        "present on all arrays (3334 genes) was 5000 (arbitrarily chosen).",
        note="the units of the score's input; the normalization is per array, which is "
        "what makes the per-hybridization z-scores comparable",
    ),
    "released_datasets": _paper(
        "Dataset S5",
        "Supplementary information contains normalized ratios (Dataset S2), "
        "$\\mathbf { Z }$ -scores relative to the unselected library (Dataset S3), "
        "$\\mathbf { Z }$ -scores relative to the enrichments performed in the media "
        "without antibiotics (Dataset S4), combined $\\mathbf { Z }$ -scores "
        "(Dataset S5), and the combined z-scores considered significant (Dataset S1).",
        note="which released table is which; the loader stores Dataset S5 and reads S1, "
        "S3 and S4 as checks",
    ),
}

REFERENCE_STRAIN_NAME: Final[EcoliK12StrainName] = "MG1655"
MG1655_ASSEMBLY_SET: Final[BacterialAssemblySet] = "ecoli_K12_MG1655_ASM584v2"
MG1655_NAMESPACE = STRAIN_GENE_NAMESPACES[REFERENCE_STRAIN_NAME]
LIBRARY_PARENT: Final[str] = SOURCED_VALUES["library_parent"].value
TEMPERATURE_C: Final[float] = SOURCED_VALUES["temperature_c"].value
AEROBICITY: Final[str] = SOURCED_VALUES["aerobicity"].value
#: ``{repetitions: significance cut}`` as Text S1 states it.
SIGNIFICANCE_THRESHOLDS: Final[dict[int, float]] = SOURCED_VALUES[
    "significance_thresholds"
].value
#: Drug codes whose hybridization columns R2 and R3 are two arrays of ONE sample.
AVERAGED_CODES: Final[tuple[str, ...]] = SOURCED_VALUES[
    "technical_replicates_averaged"
].value
#: Every released ``UNIQID`` matches this; a value that does not fails the format check.
B_NUMBER_PATTERN = re.compile(r"^b\d{4}$")
#: Below this fraction of distinct ``UNIQID``s resolving on the pinned MG1655 annotation
#: the build stops: the release keys on MG1655 b-numbers, so a lower fraction would mean
#: the wrong annotation rather than a decade of merged loci (measured 3,931 of 3,976 =
#: 0.9887 resolving to one locus, of which 3,821 are locus tags in their own right).
MIN_RESOLVED_FRACTION = 0.98
#: The largest ``|combined z|`` the reconstruction may produce for a cell Dataset S5
#: prints as 0 (measured 0.000976 over the 47 such cells).
MAX_ROUNDED_TO_ZERO = 1e-3

_PAPER_LOOKED_IN = Provenance(
    source_uri=PAPER_MD,
    citation_key=CITATION_KEY,
    sha256=PAPER_MD_SHA256,
    method="full Results, Materials and Methods and Text S1 read",
)
_GIRGIS2007 = Provenance(
    source_uri="https://doi.org/10.1371/journal.pgen.0030154",
    citation_key="girgisComprehensiveGeneticCharacterization2007",
    method="this paper's own deferral: 'Genetic footprinting and subsequent "
    "hybridization to DNA spotted arrays were performed as described in Girgis et al. "
    "[25]', where [25] is Girgis, Liu, Ryu and Tavazoie 2007, PLoS Genet 3:1644, which "
    "built the library and names the transposon; it is not in the mirror",
    page="PLoS Genet 3(9):e154",
)

DURATION_GAP = ProvenanceGap(
    field="duration_hours",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_PAPER_LOOKED_IN,
    note="the selection is dated in DAYS of 2% serial transfer (Table 1's Day column) "
    "and no hours or generations per transfer are stated; three drugs are dated '2/3*', "
    "two samples from day 2 and one from day 3, which has no single value at all. The "
    "day is kept verbatim in the phenotype's screen_id",
)

_RESPONSE_UNCERTAINTY_NOTE = (
    "Dataset S5 releases one combined z-score per cell and no dispersion. The Stdev "
    "column of Datasets S3 and S4 is the standard deviation of the REFERENCE "
    "hybridizations with a heuristic global component of 0.15 added, which is the "
    "z-score's own denominator, not an error on the response; and the combined score is "
    "a minimum over the drug's z-scores, an order statistic with no released spread"
)
PHENOTYPE_GAPS: tuple[ProvenanceGap, ...] = (
    ProvenanceGap(
        field="environment_response_uncertainty",
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=_PAPER_LOOKED_IN,
        note=_RESPONSE_UNCERTAINTY_NOTE,
    ),
    ProvenanceGap(
        field="environment_response_se",
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=_PAPER_LOOKED_IN,
        note="no uncertainty is released, so no standard error can be derived",
    ),
)

_PERTURBATION_GAP_NOTES: dict[str, tuple[ProvenanceGapReason, Provenance, str]] = {
    "barcode": (
        ProvenanceGapReason.not_reported_by_primary,
        _PAPER_LOOKED_IN,
        "genetic footprinting counts transposon-adjacent genomic DNA by hybridization, "
        "so the library carries no molecular barcode: the insertion's own position is "
        "its read-out identity",
    ),
    "insertion_position": (
        ProvenanceGapReason.not_reported_by_primary,
        _PAPER_LOOKED_IN,
        "a record is one array spot standing for every mutant whose "
        "transposon-adjacent DNA hybridizes to that gene, insertions near it included, "
        "so no single insertion site exists for it and none is released",
    ),
    "insertion_strand": (
        ProvenanceGapReason.not_reported_by_primary,
        _PAPER_LOOKED_IN,
        "the arrays report hybridization intensity per gene, never an orientation",
    ),
    "transposon": (
        ProvenanceGapReason.deferred_pending_source_review,
        _PAPER_LOOKED_IN,
        "this paper names no transposon; it defers the library and the footprinting "
        "method to Girgis et al. 2007, which is not in the mirror",
    ),
}
#: Typed absences of the transposon leaf's own fields. The leaf carries no
#: ``provenance_gaps`` slot, so they live here and in
#: ``preprocess/perturbation_field_gaps.json``, and every record's value is None.
PERTURBATION_FIELD_GAPS: tuple[ProvenanceGap, ...] = tuple(
    ProvenanceGap(
        field=field,
        reason=reason,
        looked_in=looked_in
        if reason is not ProvenanceGapReason.deferred_pending_source_review
        else None,
        resolve_with=_GIRGIS2007
        if reason is ProvenanceGapReason.deferred_pending_source_review
        else None,
        note=note,
    )
    for field, (reason, looked_in, note) in _PERTURBATION_GAP_NOTES.items()
)

PUBLICATION = Publication(
    pubmed_id=PUBMED_ID,
    pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PUBMED_ID}/",
    doi=PAPER_DOI,
    doi_url=f"https://doi.org/{PAPER_DOI}",
)


# --------------------------------------------------------------------------- #
# The 17 antibiotics of Table 1
# --------------------------------------------------------------------------- #
class DrugSpec(BaseModel):
    """One antibiotic of Table 1: its dose, its selection day and its replicate count.

    ``code`` is the released Dataset S5 column (lowercase of Table 1's Code).
    ``hybridizations`` is the number of per-hybridization columns Datasets S3 and S4
    carry for it, and ``repetitions`` the number of independent SELECTIONS behind them,
    which is one fewer for the four drugs whose repetitions 2 and 3 Text S1 says were two
    arrays of one sample. ``table1_samples`` is what Table 1 PRINTS in its "# Samples"
    column, kept so the two cells that disagree with the release stay visible.
    ``row_quote`` is that drug's Table 1 row verbatim from the pinned ``paper.md``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    name: str
    compound_label: str
    dose_ug_per_ml: float
    day: str
    hybridizations: int
    table1_samples: int
    drug_class: str | None
    cellular_target: str
    action: str
    row_quote: str

    @property
    def repetitions(self) -> int:
        """Independent replicate selections behind the drug's combined z-score."""
        return self.hybridizations - (1 if self.code.upper() in AVERAGED_CODES else 0)

    @property
    def threshold(self) -> float:
        """The significance cut Dataset S1 applied to this drug."""
        return SIGNIFICANCE_THRESHOLDS[self.repetitions]

    @property
    def screen_id(self) -> str:
        """The condition's own label: the released column plus Table 1's day."""
        return f"{self.code} day {self.day.rstrip('*')}"

    def dose_sourced(self) -> SourcedValue:
        """The drug's whole Table 1 row, bound to the pinned ``paper.md``."""
        return _paper(
            self.dose_ug_per_ml,
            self.row_quote,
            note=f"Table 1: {self.name} at {self.dose_ug_per_ml} ug/ml, hybridized on "
            f"day {self.day}; the Dose column header is 'Dose (μg/ ml)'",
        )


def _drug(
    code: str,
    name: str,
    dose: float,
    day: str,
    hybridizations: int,
    table1_samples: int,
    drug_class: str | None,
    target: str,
    action: str,
    row_quote: str,
    *,
    compound_label: str | None = None,
) -> DrugSpec:
    """One ``DrugSpec`` from its Table 1 row."""
    return DrugSpec(
        code=code,
        name=name,
        compound_label=compound_label or name.lower(),
        dose_ug_per_ml=dose,
        day=day,
        hybridizations=hybridizations,
        table1_samples=table1_samples,
        drug_class=drug_class,
        cellular_target=target,
        action=action,
        row_quote=row_quote,
    )


#: The 17 conditions, in the column order of Dataset S5. ``hybridizations`` is measured
#: on the released Datasets S3 and S4 and checked at build time against both the column
#: headers and the Dataset S1 thresholds; Table 1's own "# Samples" cell is carried
#: beside it in ``table1_samples``, and disagrees for STR and SLF.
DRUGS: tuple[DrugSpec, ...] = (
    _drug(
        "amk",
        "Amikacin",
        1.0,
        "4",
        2,
        2,
        "Aminoglycoside",
        "Protein synthesis, 30S",
        "Bactericidal",
        "<tr><td>Amikacin</td><td>AMK</td><td>1</td><td>4</td><td>2</td>"
        "<td>Aminoglycoside</td><td>Protein synthesis, 30S</td><td>Bactericidal</td>"
        "</tr>",
    ),
    _drug(
        "gen",
        "Gentamycin",
        0.1,
        "2",
        2,
        2,
        "Aminoglycoside",
        "Protein synthesis, 30S",
        "Bactericidal",
        "<tr><td>Gentamycin</td><td>GEN</td><td>0.1</td><td>2</td><td>2</td>"
        "<td>Aminoglycoside</td><td>Protein synthesis, 30S</td><td>Bactericidal</td>"
        "</tr>",
    ),
    _drug(
        "str",
        "Streptomycin",
        3.0,
        "2",
        2,
        3,
        "Aminoglycoside",
        "Protein synthesis, 30 S",
        "Bactericidal",
        "<tr><td>Streptomycin</td><td>STR</td><td>3</td><td>2</td><td>3</td>"
        "<td>Aminoglycoside</td><td>Protein synthesis, 30 S</td><td>Bactericidal</td>"
        "</tr>",
    ),
    _drug(
        "tob",
        "Tobramycin",
        0.25,
        "2",
        2,
        2,
        "Aminoglycoside",
        "Protein synthesis, 30S",
        "Bactericidal",
        "<tr><td>Tobramycin</td><td>TOB</td><td>0.25</td><td>2</td><td>2</td>"
        "<td>Aminoglycoside</td><td>Protein synthesis, 30S</td><td>Bactericidal</td>"
        "</tr>",
    ),
    _drug(
        "lom",
        "Lomefloxacin",
        0.05,
        "2/3*",
        3,
        3,
        "Quinolone",
        "DNA gyrase",
        "Bactericidal",
        "<tr><td>Lomefloxacin</td><td>LOM</td><td>0.05</td><td>2/3*</td><td>3</td>"
        "<td>Quinolone</td><td>DNA gyrase</td><td>Bactericidal</td></tr>",
    ),
    _drug(
        "nal",
        "Nalidixic acid",
        4.0,
        "2",
        2,
        2,
        "Quinolone",
        "DNA gyrase",
        "Bactericidal",
        "<tr><td>Nalidixic acid</td><td>NAL</td><td>4</td><td>2</td><td>2</td>"
        "<td>Quinolone</td><td>DNA gyrase</td><td>Bactericidal</td></tr>",
    ),
    _drug(
        "slf",
        "Sulfamonomethoxine",
        0.5,
        "2/3*",
        3,
        2,
        "Sulfonamide",
        "Folic acid biosynthesis",
        "Bacteriostatic",
        "<tr><td>Sulfamonomethoxine</td><td>SLF</td><td>0.5</td><td>2/3*</td><td>2</td>"
        "<td>Sulfonamide</td><td>Folic acid biosynthesis</td><td>Bacteriostatic</td>"
        "</tr>",
    ),
    _drug(
        "dox",
        "Doxycycline hyclate",
        0.5,
        "2/3*",
        3,
        3,
        "Tetracycline",
        "Protein synthesis, 30S",
        "Bacteriostatic",
        "<tr><td>Doxycycline hyclate</td><td>DOX</td><td>0.5</td><td>2/3*</td>"
        "<td>3</td><td>Tetracycline</td><td>Protein synthesis, 30S</td>"
        "<td>Bacteriostatic</td></tr>",
    ),
    _drug(
        "tet",
        "Tetracycline",
        0.25,
        "2",
        3,
        2,
        "Tetracycline",
        "Protein synthesis, 30S",
        "Bacteriostatic",
        "<tr><td>Tetracycline</td><td>TET</td><td>0.25</td><td>2</td><td>2</td>"
        "<td>Tetracycline</td><td>Protein synthesis, 30S</td><td>Bacteriostatic</td>"
        "</tr>",
    ),
    _drug(
        "blm",
        "Bleomycin",
        1.0,
        "2",
        2,
        2,
        "Peptide",
        "Nucleic acid",
        "Bacteriostatic",
        "<tr><td>Bleomycin</td><td>BLM</td><td>1</td><td>2</td><td>2</td>"
        "<td>Peptide</td><td>Nucleic acid</td><td>Bacteriostatic</td></tr>",
    ),
    _drug(
        "fox",
        "Cefoxitin",
        1.0,
        "3",
        3,
        2,
        "β-lactam",
        "Cell wall biosynthesis",
        "Bactericidal",
        "<tr><td>Cefoxitin</td><td>FOX</td><td>1</td><td>3</td><td>2</td>"
        "<td>β-lactam</td><td>Cell wall biosynthesis</td><td>Bactericidal</td>"
        "</tr>",
    ),
    _drug(
        "ery",
        "Erythromycin",
        2.0,
        "3",
        2,
        2,
        "Macrolide",
        "Protein synthesis, 50S",
        "Bacteriostatic",
        "<tr><td>Erythromycin</td><td>ERY</td><td>2</td><td>3</td><td>2</td>"
        "<td>Macrolide</td><td>Protein synthesis, 50S</td><td>Bacteriostatic</td></tr>",
    ),
    _drug(
        "nit",
        "Nitrofurantoin",
        4.0,
        "3",
        2,
        2,
        "Nitroheterocyclic",
        "Multiple Targets",
        "Bactericidal",
        "<tr><td>Nitrofurantoin</td><td>NIT</td><td>4</td><td>3</td><td>2</td>"
        "<td>Nitroheterocyclic</td><td>Multiple Targets</td><td>Bactericidal</td></tr>",
    ),
    _drug(
        "amp",
        "Ampicillin",
        3.0,
        "2",
        3,
        3,
        "β-lactam",
        "Cell wall biosynthesis",
        "Bactericidal",
        "<tr><td>Ampicillin</td><td>AMP</td><td>3</td><td>2</td><td>3</td>"
        "<td>β-lactam</td><td>Cell wall biosynthesis</td><td>Bactericidal</td>"
        "</tr>",
    ),
    _drug(
        "pip",
        "Piperacillin",
        1.0,
        "4",
        3,
        2,
        "β-lactam",
        "Cell wall biosynthesis",
        "Bactericidal",
        "<tr><td>Piperacillin</td><td>PIP</td><td>1</td><td>4</td><td>2</td>"
        "<td>β-lactam</td><td>Cell wall biosynthesis</td><td>Bactericidal</td>"
        "</tr>",
    ),
    _drug(
        "trm",
        "Trimethoprim",
        0.5,
        "4",
        3,
        2,
        "DHFR Inhibitor",
        "Folic acid biosynthesis",
        "Bacteriostatic",
        "<tr><td>Trimethoprim</td><td>TRM</td><td>0.5</td><td>4</td><td>2</td>"
        "<td>DHFR Inhibitor</td><td>Folic acid biosynthesis</td><td>Bacteriostatic</td>"
        "</tr>",
    ),
    _drug(
        "fus",
        "Fusidic acid",
        180.0,
        "4",
        2,
        2,
        None,
        "Protein synthesis, 50S",
        "Bacteriostatic",
        "<tr><td>Fusidic acid</td><td>FUS</td><td>180</td><td>4</td><td>2</td><td></td>"
        "<td>Protein synthesis, 50S</td><td>Bacteriostatic</td></tr>",
    ),
)
DRUGS_BY_CODE: dict[str, DrugSpec] = {d.code: d for d in DRUGS}
DRUG_CODES: tuple[str, ...] = tuple(d.code for d in DRUGS)


# --------------------------------------------------------------------------- #
# Raw mirror (the loader reads the mirror, never a live URL)
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirrors and the build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/girgisGeneticArchitectureIntrinsic2009``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def library_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-library/girgisGeneticArchitectureIntrinsic2009``."""
    return Path(data_root or _data_root()) / LIBRARY_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retrieve_raw_files(
    dest_dir: str | Path, names: Sequence[str] | None = None
) -> dict[str, Path]:
    """Run each file's recorded retriever and write the verified bytes to ``dest_dir``.

    The recorded ``RetrievalRecord`` is what runs (``run_retriever``), so this is the
    re-runnable retrieval itself; a byte mismatch raises before anything is written.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for raw in RAW_FILES:
        if names is not None and raw.name not in names:
            continue
        path = dest / raw.name
        write_verified(
            run_retriever(raw.retrieval),
            path,
            raw.sha256,
            raw.retrieval.source_url or raw.name,
        )
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
                source=raw.retrieval.source_url,
                retrieval=raw.retrieval,
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title="Genetic Architecture of Intrinsic Antibiotic Susceptibility",
        files=records,
        si_data_sources=[r.retrieval.source_url or r.name for r in RAW_FILES],
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
# Reading the released sheets
# --------------------------------------------------------------------------- #
class SheetFormatError(ValueError):
    """A released sheet whose title, header, ids or a cell is not what the loader reads."""


class CombinationRuleError(ValueError):
    """A Dataset S5 cell the paper's own combination rule does not reproduce."""


class ThresholdError(ValueError):
    """A Dataset S1 value that disagrees with its drug's significance threshold."""


class SheetSpec(BaseModel):
    """One released sheet: where its header is and what its first cell must say."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str
    sheet_name: str
    title: str
    header_row: int
    id_column: str
    description_column: str


SHEETS: dict[str, SheetSpec] = {
    DATASET_S1: SheetSpec(
        file=DATASET_S1,
        sheet_name="S1- Scores of Sig",
        title="Dataset S1:  Z-scores for loci with a significant effect on antibiotic "
        "susceptibility.",
        header_row=7,
        id_column="UNIQID",
        description_column="NAME",
    ),
    DATASET_S3: SheetSpec(
        file=DATASET_S3,
        sheet_name="S3-z relative unselected",
        title="Dataset S3: z-scores for individual hybridization computed relative to "
        "five hybridizations of the original, unselected library.",
        header_row=5,
        id_column="bnum",
        description_column="Description",
    ),
    DATASET_S4: SheetSpec(
        file=DATASET_S4,
        sheet_name="S4-z relative no antibiotic",
        title="Dataset S4: z-scores for individual hybridizations computed relative to "
        "six hybridization of the library cultured in the same media (M9 with glucose "
        "and casamino acids) without antibiotics.",
        header_row=4,
        id_column="bnum",
        description_column="Description",
    ),
    DATASET_S5: SheetSpec(
        file=DATASET_S5,
        sheet_name="S5-Scores of All",
        title="Dataset S5:  Combined Z-scores for all loci --see Materials and Methods "
        "for calculation details.",
        header_row=3,
        id_column="UNIQID",
        description_column="NAME",
    ),
}
#: The sheets' own marker for a cell with fewer than two usable repetitions.
NO_DATA_MARKER = "ND"
#: Datasets S3 and S4 mark an unusable array this way, and the combined score is then
#: computed from the remaining hybridizations.
LOW_QUALITY_MARKER = "LQ"


def read_sheet(path: str | Path, spec: SheetSpec) -> pd.DataFrame:
    """One released sheet as a frame, refusing an unexpected title or header.

    The title is read from the raw first cell and the header from ``spec.header_row``,
    so a re-released file with a shifted header fails here instead of silently loading
    the wrong row as column names.
    """
    sheets = pd.read_excel(path, sheet_name=None, header=None, nrows=1)
    if list(sheets) != [spec.sheet_name]:
        raise SheetFormatError(f"{spec.file}: sheets {list(sheets)}")
    title = sheets[spec.sheet_name].iat[0, 0]
    if title != spec.title:
        raise SheetFormatError(f"{spec.file}: title {title!r}, not {spec.title!r}")
    frame = pd.read_excel(path, sheet_name=spec.sheet_name, header=spec.header_row)
    for column in (spec.id_column, spec.description_column):
        if column not in frame.columns:
            raise SheetFormatError(f"{spec.file}: no {column!r} column")
    ids = [str(value) for value in frame[spec.id_column].tolist()]
    bad = [value for value in ids if B_NUMBER_PATTERN.match(value) is None]
    if bad:
        raise SheetFormatError(
            f"{spec.file}: {len(bad)} ids are not b-numbers: {bad[:5]}"
        )
    return frame


def hybridization_columns(frame: pd.DataFrame, code: str) -> list[str]:
    """The per-hybridization columns of one drug in Dataset S3 or S4, in order.

    They are named ``<CODE>_R<n>``; the four drugs with a repeated array carry three.
    """
    prefix = f"{code.upper()}_R"
    return sorted(
        (str(column) for column in frame.columns if str(column).startswith(prefix)),
        key=lambda column: int(column.removeprefix(prefix)),
    )


def _cell(value: Any) -> float | None:
    """A sheet cell as a finite float, or None for ``ND``/``LQ``/blank."""
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip() in (NO_DATA_MARKER, LOW_QUALITY_MARKER):
            return None
        raise SheetFormatError(f"unexpected text cell {value!r}")
    if isinstance(value, bool):
        raise SheetFormatError(f"unexpected boolean cell {value!r}")
    if not isinstance(value, (int, float)):
        raise SheetFormatError(f"unexpected cell {value!r}")
    if pd.isna(value):
        return None
    if not math.isfinite(value):
        raise SheetFormatError(f"non-finite cell {value!r}")
    return float(value)


def combined_z(z_scores: Sequence[float]) -> float:
    """The paper's combination rule over one drug's z-scores.

    The score closest to zero when every z-score has the same sign, else 0: "When a gene
    had Z-scores of different signs, the gene was assigned a score of 0, indicating no
    consistent fitness effect."
    """
    signs = {(1 if z > 0 else -1 if z < 0 else 0) for z in z_scores}
    if len(signs) != 1 or 0 in signs:
        return 0.0
    return min(z_scores, key=abs)


def drug_z_scores(
    reference_rows: Sequence[Sequence[Any]], spec: DrugSpec
) -> list[float]:
    """Every z-score of one drug in one gene, over both reference sets.

    ``reference_rows`` holds the drug's hybridization cells of Dataset S3 and of Dataset
    S4, each in column order. An ``LQ`` array drops out of its set. For the four drugs
    Text S1 names, the second and third hybridizations of a set are two arrays of one
    sample and are averaged into a single repetition before the rule is applied.
    """
    averaged = spec.code.upper() in AVERAGED_CODES
    out: list[float] = []
    for row in reference_rows:
        values = [v for v in (_cell(cell) for cell in row) if v is not None]
        if not values:
            continue
        if averaged and len(values) > 1:
            out.append(values[0])
            out.append(sum(values[1:]) / len(values[1:]))
        else:
            out.extend(values)
    return out


class CombinationCheck(BaseModel):
    """How Dataset S5 compares with the rule re-applied to Datasets S3 and S4."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    drug: str
    repetitions: int
    n_cells: int
    n_exact: int
    n_rounded_to_zero: int
    n_no_data: int
    disagreements: list[str]


def check_combination_rule(
    s3: pd.DataFrame, s4: pd.DataFrame, s5: pd.DataFrame
) -> list[CombinationCheck]:
    """Re-derive every Dataset S5 cell from Datasets S3 and S4 by the paper's rule.

    A cell matches exactly, or is a cell Dataset S5 prints as 0 for which the rule gives
    ``|z| < MAX_ROUNDED_TO_ZERO`` (the sheet's own display floor), or it is a
    disagreement. ``ND`` cells are counted and skipped.
    """
    checks: list[CombinationCheck] = []
    for spec in DRUGS:
        cols3 = hybridization_columns(s3, spec.code)
        cols4 = hybridization_columns(s4, spec.code)
        for frame_name, columns in (("S3", cols3), ("S4", cols4)):
            if len(columns) != spec.hybridizations:
                raise SheetFormatError(
                    f"Dataset {frame_name}: {spec.code} has {len(columns)} "
                    f"hybridization columns, not {spec.hybridizations}"
                )
        released = s5[spec.code].tolist()
        rows3 = s3[cols3].to_numpy(dtype=object).tolist()
        rows4 = s4[cols4].to_numpy(dtype=object).tolist()
        exact = rounded = no_data = 0
        disagreements: list[str] = []
        for gene, value, row3, row4 in zip(
            s5[SHEETS[DATASET_S5].id_column].tolist(),
            released,
            rows3,
            rows4,
            strict=True,
        ):
            target = _cell(value)
            if target is None:
                no_data += 1
                continue
            z_scores = drug_z_scores((row3, row4), spec)
            if not z_scores:
                disagreements.append(
                    f"{gene}: Dataset S5 gives {target} with no usable z-score"
                )
                continue
            predicted = combined_z(z_scores)
            if abs(predicted - target) < 1e-9:
                exact += 1
            elif target == 0.0 and abs(predicted) < MAX_ROUNDED_TO_ZERO:
                rounded += 1
            else:
                disagreements.append(
                    f"{gene}: Dataset S5 gives {target}, the rule gives {predicted} "
                    f"from {z_scores}"
                )
        checks.append(
            CombinationCheck(
                drug=spec.code,
                repetitions=spec.repetitions,
                n_cells=len(released),
                n_exact=exact,
                n_rounded_to_zero=rounded,
                n_no_data=no_data,
                disagreements=disagreements,
            )
        )
    return checks


class ThresholdCheck(BaseModel):
    """What Dataset S1 says about one drug's significance cut, hence its repetitions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    drug: str
    hybridizations: int
    repetitions: int
    threshold: float
    table1_samples: int
    n_significant: int
    min_abs: float | None
    n_below_two_repetition_cut: int


def check_thresholds(s1: pd.DataFrame, s5: pd.DataFrame) -> list[ThresholdCheck]:
    """Prove each drug's repetition count from Dataset S1, and S1's agreement with S5.

    Every non-zero Dataset S1 value must equal its Dataset S5 cell (S1 is S5 with the
    insignificant cells zeroed) and must clear the cut its repetition count implies. A
    three-repetition drug must additionally carry at least one significant value below
    the two-repetition cut, because that is what distinguishes the 1.5 threshold from
    the 2.15 one.
    """
    two_cut = SIGNIFICANCE_THRESHOLDS[2]
    checks: list[ThresholdCheck] = []
    for spec in DRUGS:
        magnitudes: list[float] = []
        below = 0
        for gene, sig, full in zip(
            s1[SHEETS[DATASET_S1].id_column].tolist(),
            s1[spec.code].tolist(),
            s5[spec.code].tolist(),
            strict=True,
        ):
            significant, combined = _cell(sig), _cell(full)
            if (significant is None) != (combined is None):
                raise ThresholdError(
                    f"{spec.code} {gene}: Dataset S1 and S5 disagree on no-data"
                )
            if significant is None or significant == 0.0:
                continue
            if combined is None or abs(significant - combined) > 1e-12:
                raise ThresholdError(
                    f"{spec.code} {gene}: Dataset S1 gives {significant}, S5 {combined}"
                )
            magnitudes.append(abs(significant))
            below += abs(significant) < two_cut
        if magnitudes and min(magnitudes) < spec.threshold:
            raise ThresholdError(
                f"{spec.code}: a significant |z| of {min(magnitudes)} is below the "
                f"{spec.threshold} cut of a {spec.repetitions}-repetition drug"
            )
        if spec.repetitions == 3 and below == 0:
            raise ThresholdError(
                f"{spec.code}: read as a 3-repetition drug but no significant |z| is "
                f"below the {two_cut} two-repetition cut"
            )
        if spec.repetitions == 2 and below:
            raise ThresholdError(
                f"{spec.code}: read as a 2-repetition drug but {below} significant |z| "
                f"are below {two_cut}"
            )
        checks.append(
            ThresholdCheck(
                drug=spec.code,
                hybridizations=spec.hybridizations,
                repetitions=spec.repetitions,
                threshold=spec.threshold,
                table1_samples=spec.table1_samples,
                n_significant=len(magnitudes),
                min_abs=min(magnitudes) if magnitudes else None,
                n_below_two_repetition_cut=below,
            )
        )
    return checks


def check_sheets_align(frames: Mapping[str, pd.DataFrame]) -> list[str]:
    """Refuse sheets whose ids or descriptions differ, and return the shared ids."""
    reference = SHEETS[DATASET_S5]
    ids = [str(v) for v in frames[DATASET_S5][reference.id_column].tolist()]
    descriptions = [
        str(v) for v in frames[DATASET_S5][reference.description_column].tolist()
    ]
    if len(set(ids)) != len(ids):
        raise SheetFormatError(f"{DATASET_S5}: {len(ids) - len(set(ids))} repeated ids")
    for name, frame in frames.items():
        spec = SHEETS[name]
        other = [str(v) for v in frame[spec.id_column].tolist()]
        if other != ids:
            raise SheetFormatError(
                f"{name}: {spec.id_column} does not match {DATASET_S5}'s UNIQID order "
                f"({len(other)} vs {len(ids)} rows)"
            )
        if [str(v) for v in frame[spec.description_column].tolist()] != descriptions:
            raise SheetFormatError(f"{name}: descriptions differ from {DATASET_S5}'s")
    missing = sorted(set(DRUG_CODES) - set(map(str, frames[DATASET_S5].columns)))
    if missing:
        raise SheetFormatError(f"{DATASET_S5}: no column for {missing}")
    return ids


# --------------------------------------------------------------------------- #
# Identifiers and the retention ledger
# --------------------------------------------------------------------------- #
#: The retention rules, in application order.
DROP_RULE_DESCRIPTIONS: dict[str, str] = {
    "b_number_is_not_a_locus_tag_of_the_pinned_annotation": "the released b-number is "
    "not a locus tag of GCA_000005845.2: the annotation either carries it as a "
    "/gene_synonym of a different locus (a merge since 2009) or does not carry it at "
    "all. Storing the merged locus needs a DerivedIdentifierMapping and "
    "DerivedIdentifierRoute has no member for a retired tag of the pinned strain's own "
    "namespace (issue #753), and storing the released tag would place a record on a "
    "locus the assembly does not have, so every one of the drug's cells for that gene "
    "is dropped; each item gives the annotation's own resolution note",
    "no_combined_score_released": "Dataset S5 gives the cell as 'ND', which the sheet "
    "defines as fewer than two usable repetitions, so the paper assigned no combined "
    "z-score and there is no measurement to store",
}


class DropRule(BaseModel):
    """One retention rule, the cells it removed, and why."""

    rule: str
    description: str
    n_records: int
    items: list[str] = []


class DrugLedger(BaseModel):
    """Per-drug counts: released cells, kept records and their sign split."""

    drug: str
    screen_id: str
    source_cells: int
    no_data: int
    dropped_identifier: int
    kept_records: int
    n_positive: int
    n_negative: int
    n_zero: int


class DropLog(BaseModel):
    """The retention ledger of one build, in the order the rules were applied."""

    dataset: str
    source_loci: int
    kept_loci: int
    source_records: int
    kept_records: int
    dropped_records: int
    drugs: list[DrugLedger]
    rules: list[DropRule]


class IdentifierLedger(BaseModel):
    """The reconciliation of every released b-number, plus the stop threshold.

    ``not_a_locus_tag`` lists each dropped b-number with the resolution note the
    annotation gave it, which is what names the merge or the retirement.
    """

    reconciliation: LocusTagReconciliation
    min_resolved_fraction: float
    n_locus_tags: int
    not_a_locus_tag: dict[str, str]


class ReplicateStructure(BaseModel):
    """The per-drug repetition count, its two checks, and Table 1's disagreements."""

    thresholds: dict[int, float]
    averaged_codes: list[str]
    per_drug: list[ThresholdCheck]
    table1_disagreements: list[str]


def canonical_symbol(genome: EcoliK12Genome, tag: str) -> str:
    """The annotation's own gene symbol of ``tag`` when it resolves back to ``tag``,
    else the tag itself, so the stored pair always round-trips through the resolver.
    """
    symbol = genome.genbank.loci[tag].symbol
    if symbol is None:
        return tag
    resolution = genome.resolve_gene_name(symbol)
    return symbol if resolution.systematic_name == tag else tag


def resolve_b_numbers(
    genome: EcoliK12Genome, b_numbers: Sequence[str], *, label: str
) -> tuple[dict[str, str], IdentifierLedger]:
    """The kept b-numbers mapped to the annotation's symbol, plus the ledger.

    A b-number is kept when it IS a locus tag of the pinned annotation (a current gene
    or a pseudogene locus), which is the only case in which the record's stored tag is
    the one the source released and no ``DerivedIdentifierMapping`` is needed. Stops
    (``LocusTagResolutionError``) below :data:`MIN_RESOLVED_FRACTION`.
    """
    stored, report = reconcile_locus_tags(
        genome, pd.Series(list(b_numbers), dtype=object), label=label
    )
    report.require_resolved(MIN_RESOLVED_FRACTION)
    pattern = LOCUS_TAG_PATTERNS[report.gene_namespace]
    kept: dict[str, str] = {}
    unplaced: dict[str, str] = {}
    for b_number in dict.fromkeys(b_numbers):
        resolution = genome.resolve_gene_name(b_number)
        if resolution_layer(genome, resolution) == LAYER_LOCUS_TAG:
            if pattern.match(b_number) is None:
                raise SheetFormatError(
                    f"{b_number} is a locus of {report.assembly_set} but not a "
                    f"{report.gene_namespace} tag"
                )
            kept[b_number] = canonical_symbol(genome, b_number)
        else:
            unplaced[b_number] = (
                f"{resolution.status.value}: {resolution.note or 'no note'}"
            )
    ledger = IdentifierLedger(
        reconciliation=report,
        min_resolved_fraction=MIN_RESOLVED_FRACTION,
        n_locus_tags=len(kept),
        not_a_locus_tag=unplaced,
    )
    del stored
    return kept, ledger


# --------------------------------------------------------------------------- #
# Media, environment, genotype, phenotype (pure, no files)
# --------------------------------------------------------------------------- #
_M9_SALT_NOTE = (
    "an M9 salt; the paper names the salts only as 'M9 salts [80]', a reference to "
    "Ausubel's Current Protocols, and prints no amount, so none is asserted here"
)


def _m9_salts(statement: SourcedValue) -> list[MediaComponent]:
    """The four M9 salts of the ``M9`` library object, with no amount asserted."""
    return [
        MediaComponent(
            compound=component.compound,
            role=component.role,
            concentration=None,
            provenance=[statement],
            note=_M9_SALT_NOTE,
        )
        for component in M9.components
    ]


def girgis_medium() -> Media:
    """The one medium of all 17 selections, derived from the ``M9`` library key."""
    statement = SOURCED_VALUES["medium"]
    return Media(
        name="M9 salts with 0.4% glucose, 0.1% casamino acids, 1 mM MgSO4, 0.1 mM "
        "CaCl2 and 1.5 uM thiamine, salt amounts not stated (Girgis 2009), liquid",
        state="liquid",
        is_synthetic=False,
        base_medium="M9",
        components=[
            *_m9_salts(statement),
            MediaComponent(
                compound=resolved_compound("D-glucose"),
                role=MediaComponentRole.carbon_source,
                concentration=Concentration(
                    value=0.4, unit=ConcentrationUnit.percent_w_v
                ),
                provenance=[statement],
                note="the paper writes 0.4% with no w/v or v/v marker and glucose is a "
                "solid, so it is read as w/v",
            ),
            MediaComponent(
                compound=resolved_compound("casamino acids"),
                role=MediaComponentRole.complex_ingredient,
                concentration=Concentration(
                    value=0.1, unit=ConcentrationUnit.percent_w_v
                ),
                definition=ComponentDefinition.intrinsically_undefined,
                provenance=[statement],
                note="an acid hydrolysate of casein, so there is no structure to "
                "resolve; it is what makes is_synthetic False",
            ),
            MediaComponent(
                compound=resolved_compound("magnesium sulfate"),
                role=MediaComponentRole.bulk_salt,
                concentration=Concentration(
                    value=1.0, unit=ConcentrationUnit.millimolar
                ),
                provenance=[statement],
            ),
            MediaComponent(
                compound=resolved_compound("calcium chloride"),
                role=MediaComponentRole.bulk_salt,
                concentration=Concentration(
                    value=0.1, unit=ConcentrationUnit.millimolar
                ),
                provenance=[statement],
            ),
            MediaComponent(
                compound=resolved_compound("thiamine"),
                role=MediaComponentRole.vitamin,
                concentration=Concentration(
                    value=1.5, unit=ConcentrationUnit.micromolar
                ),
                provenance=[statement],
            ),
        ],
        provenance=[statement],
    )


GIRGIS_MEDIUM = girgis_medium()
"""The selections' medium, built once: identical across all 17 conditions."""


def antibiotic(spec: DrugSpec) -> SmallMoleculePerturbation:
    """One drug of Table 1 at its printed dose, as the condition's edit."""
    return SmallMoleculePerturbation(
        compound=resolved_compound(spec.compound_label),
        concentration=Concentration(
            value=spec.dose_ug_per_ml, unit=ConcentrationUnit.ug_per_ml
        ),
    )


def environment(spec: DrugSpec) -> Environment:
    """The selection culture of one drug: the shared medium plus that drug."""
    return Environment(
        media=GIRGIS_MEDIUM,
        temperature=Temperature(value=TEMPERATURE_C),
        perturbations=[antibiotic(spec)],
        aerobicity=AEROBICITY,
        provenance_gaps=[DURATION_GAP],
    )


def insertion_genotype(locus_tag: str, symbol: str) -> Genotype:
    """The gene-level transposon disruption of one MG1655 locus."""
    return Genotype(
        perturbations=[
            TransposonInsertionPerturbation(
                systematic_gene_name=locus_tag,
                perturbed_gene_name=symbol,
                gene_namespace=MG1655_NAMESPACE,
            )
        ]
    )


UNITS = (
    "combined z-score of log2(normalized transposon signal / normalized genomic DNA "
    "signal) against two reference sets (five hybridizations of the unselected library "
    "and six no-antibiotic selections): the z-score closest to zero among the drug's "
    "hybridizations when they all agree in sign, else 0 for no consistent fitness "
    "effect. Positive = disrupting the gene is beneficial in that drug"
)
UNITS_REFERENCE = (
    "the unperturbed parent in the same drug: 0 on the combined z-score axis, no "
    "consistent fitness effect relative to the reference hybridizations"
)


def phenotype(value: float, spec: DrugSpec) -> EnvironmentResponsePhenotype:
    """One released combined z-score of one gene in one drug."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.z_score,
        assay_type=AssayType.other,
        environment_response=value,
        n_samples=spec.repetitions,
        sample_unit=SampleUnit.biological_replicate,
        units=UNITS,
        screen_id=spec.screen_id,
        provenance_gaps=list(PHENOTYPE_GAPS),
    )


def reference_phenotype(spec: DrugSpec) -> EnvironmentResponsePhenotype:
    """The parent strain of the same condition: a combined z-score of 0."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.z_score,
        assay_type=AssayType.other,
        environment_response=0.0,
        n_samples=spec.repetitions,
        sample_unit=SampleUnit.biological_replicate,
        units=UNITS_REFERENCE,
        screen_id=spec.screen_id,
    )


def library_background() -> BacterialStrainBackground:
    """MG1655 ``delta-lacZ``, the strain the transposon library was built in.

    ``alleles`` is empty and that is a statement: the two sources write the genotype as
    a delta, and neither says whether the lacZ lesion is a full deletion, an internal
    one or a cassette replacement. ``AlleleEdit`` has no member for an unspecified edit,
    so typing one would assert a mechanism the paper never gave; the genotype string is
    kept verbatim instead.
    """
    return BacterialStrainBackground(
        name=LIBRARY_PARENT,
        reference_strain=REFERENCE_STRAIN_NAME,
        assembly_set=MG1655_ASSEMBLY_SET,
        parents=["MG1655"],
        construction="the transposon insertion mutants were generated in it, one "
        "insertion per mutant, in the study this paper defers to (Girgis 2007)",
        genotype_statement="MG1655 ∆lacZ",
        alleles=[],
        provenance=[
            SOURCED_VALUES["library_parent"],
            SOURCED_VALUES["library_construction"],
        ],
    )


def reference_genome(data_root: str | None = None) -> AssemblyReferenceGenome:
    """MG1655 pinned to its GenBank assembly, carrying the library's background."""
    return assembly_reference(
        REFERENCE_STRAIN_NAME, background=library_background(), data_root=data_root
    )


def build_experiment(
    dataset_name: str,
    locus_tag: str,
    symbol: str,
    value: float,
    spec: DrugSpec,
    env: Environment,
) -> BacterialEnvironmentResponseExperiment:
    """The record of one (gene, drug) cell of Dataset S5."""
    return BacterialEnvironmentResponseExperiment(
        dataset_name=dataset_name,
        genotype=insertion_genotype(locus_tag, symbol),
        environment=env,
        phenotype=phenotype(value, spec),
    )


def build_reference(
    dataset_name: str, genome: AssemblyReferenceGenome, spec: DrugSpec, env: Environment
) -> BacterialEnvironmentResponseExperimentReference:
    """The unperturbed parent of one condition, scoring 0."""
    return BacterialEnvironmentResponseExperimentReference(
        dataset_name=dataset_name,
        genome_reference=genome,
        environment_reference=env.model_copy(),
        phenotype_reference=reference_phenotype(spec),
    )


def stored_cells(
    s5: pd.DataFrame, b_numbers: Sequence[str], kept: Mapping[str, str]
) -> Iterator[tuple[str, str, DrugSpec, float]]:
    """Every stored cell, gene-major then drug-major, in LMDB order.

    A gene whose b-number is not in ``kept`` contributes nothing, and so does a cell
    Dataset S5 released as ``ND``.
    """
    columns = {spec.code: s5[spec.code].tolist() for spec in DRUGS}
    for row, b_number in enumerate(b_numbers):
        symbol = kept.get(b_number)
        if symbol is None:
            continue
        for spec in DRUGS:
            value = _cell(columns[spec.code][row])
            if value is None:
                continue
            yield b_number, symbol, spec, value


def build_drop_log(
    dataset_name: str,
    s5: pd.DataFrame,
    b_numbers: Sequence[str],
    kept: Mapping[str, str],
) -> DropLog:
    """The retention ledger of one build, refusing rules that miss a dropped cell."""
    columns = {spec.code: s5[spec.code].tolist() for spec in DRUGS}
    dropped_identifier = sorted(set(b_numbers) - set(kept))
    no_data: list[str] = []
    drugs: list[DrugLedger] = []
    for spec in DRUGS:
        signs: Counter[str] = Counter()
        missing = identifier = 0
        for b_number, cell in zip(b_numbers, columns[spec.code], strict=True):
            if b_number not in kept:
                identifier += 1
                continue
            value = _cell(cell)
            if value is None:
                missing += 1
                no_data.append(f"{b_number} {spec.code}")
                continue
            signs["positive" if value > 0 else "negative" if value < 0 else "zero"] += 1
        drugs.append(
            DrugLedger(
                drug=spec.code,
                screen_id=spec.screen_id,
                source_cells=len(b_numbers),
                no_data=missing,
                dropped_identifier=identifier,
                kept_records=sum(signs.values()),
                n_positive=signs["positive"],
                n_negative=signs["negative"],
                n_zero=signs["zero"],
            )
        )
    rules = [
        DropRule(
            rule="b_number_is_not_a_locus_tag_of_the_pinned_annotation",
            description=DROP_RULE_DESCRIPTIONS[
                "b_number_is_not_a_locus_tag_of_the_pinned_annotation"
            ],
            n_records=len(dropped_identifier) * len(DRUGS),
            items=dropped_identifier,
        ),
        DropRule(
            rule="no_combined_score_released",
            description=DROP_RULE_DESCRIPTIONS["no_combined_score_released"],
            n_records=len(no_data),
            items=no_data,
        ),
    ]
    source_records = len(b_numbers) * len(DRUGS)
    kept_records = sum(drug.kept_records for drug in drugs)
    drop_log = DropLog(
        dataset=dataset_name,
        source_loci=len(b_numbers),
        kept_loci=len(kept),
        source_records=source_records,
        kept_records=kept_records,
        dropped_records=source_records - kept_records,
        drugs=drugs,
        rules=rules,
    )
    if sum(rule.n_records for rule in rules) != drop_log.dropped_records:
        raise RuntimeError("drop rules do not account for every dropped cell")
    return drop_log


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class EnvChemgenGirgis2009Dataset(ExperimentDataset):
    """Combined z-scores of MG1655 transposon mutants in 17 antibiotics."""

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = REFERENCE_STRAIN_NAME

    def __init__(
        self,
        root: str = "data/torchcell/ecoli_env_chemgen_girgis2009",
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
        """Every consumed file, linked from the raw mirror."""
        return [raw.name for raw in RAW_FILES]

    def download(self) -> None:
        """Link each mirror file into ``raw/`` after checking the manifest and sha256.

        The mirror plus ``DATA_SHA256`` is canonical; the PMC bucket URLs are retrieval
        metadata that ``retrieve_raw_files`` re-runs, never a build input.
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
        log.info("Girgis 2009 raw files linked into %s (sha256 verified)", self.raw_dir)

    def _genome(self) -> EcoliK12Genome:
        """The injected genome, or the reference strain's default cache (a direct run);
        a genome of another assembly set is refused.
        """
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
        """Parse Dataset S5 into per-gene, per-drug records + LMDB, checking it first."""
        verify_raw_files(self.raw_dir, DATA_SHA256)
        frames = {
            name: read_sheet(osp.join(self.raw_dir, name), spec)
            for name, spec in SHEETS.items()
        }
        b_numbers = check_sheets_align(frames)
        thresholds = check_thresholds(frames[DATASET_S1], frames[DATASET_S5])
        combination = check_combination_rule(
            frames[DATASET_S3], frames[DATASET_S4], frames[DATASET_S5]
        )
        failed = [c for c in combination if c.disagreements]
        if failed:
            raise CombinationRuleError(
                f"{sum(len(c.disagreements) for c in failed)} Dataset S5 cells are not "
                f"reproduced by the combination rule: "
                f"{[(c.drug, c.disagreements[0]) for c in failed[:3]]}"
            )

        genome = self._genome()
        kept, identifiers = resolve_b_numbers(
            genome, b_numbers, label=f"{self.name} Dataset S5 UNIQID"
        )
        drop_log = build_drop_log(self.name, frames[DATASET_S5], b_numbers, kept)
        log.info(
            "Girgis 2009: %d loci x %d drugs = %d cells -> %d records; dropped %s",
            drop_log.source_loci,
            len(DRUGS),
            drop_log.source_records,
            drop_log.kept_records,
            {rule.rule: rule.n_records for rule in drop_log.rules},
        )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self._write_ledgers(drop_log, identifiers, thresholds, combination)

        environments = {spec.code: environment(spec) for spec in DRUGS}
        genome_reference = reference_genome()
        references = {
            spec.code: build_reference(
                self.name, genome_reference, spec, environments[spec.code]
            )
            for spec in DRUGS
        }
        env_out, interned_env = self._open_write_lmdb(
            osp.join(self.processed_dir, "lmdb")
        )
        index = 0
        with env_out.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for locus_tag, symbol, spec, value in tqdm(
                stored_cells(frames[DATASET_S5], b_numbers, kept),
                total=drop_log.kept_records,
                desc="girgis2009",
            ):
                experiment = build_experiment(
                    self.name, locus_tag, symbol, value, spec, environments[spec.code]
                )
                txn.put(
                    f"{index}".encode(),
                    self._intern_record(
                        experiment, references[spec.code], PUBLICATION, itxn
                    ),
                )
                index += 1
        env_out.close()
        interned_env.close()
        if index != drop_log.kept_records:
            raise RuntimeError(
                f"wrote {index} records, the ledger counted {drop_log.kept_records}"
            )
        log.info("Wrote %d Girgis 2009 environment-response experiments to LMDB", index)

    def _write_ledgers(
        self,
        drop_log: DropLog,
        identifiers: IdentifierLedger,
        thresholds: Sequence[ThresholdCheck],
        combination: Sequence[CombinationCheck],
    ) -> None:
        """The drop log, the identifier ledger, the replicate structure, the
        reconstruction check and the typed absences of the transposon leaf's fields.
        """
        out = Path(self.preprocess_dir)
        (out / "dropped_records.json").write_text(drop_log.model_dump_json(indent=2))
        (out / "identifier_reconciliation.json").write_text(
            identifiers.model_dump_json(indent=2)
        )
        (out / "replicate_structure.json").write_text(
            ReplicateStructure(
                thresholds=SIGNIFICANCE_THRESHOLDS,
                averaged_codes=list(AVERAGED_CODES),
                per_drug=list(thresholds),
                table1_disagreements=[
                    f"{check.drug}: Table 1 prints # Samples {check.table1_samples}, "
                    f"the release has {check.hybridizations} hybridization columns and "
                    f"{check.repetitions} repetitions (Dataset S1 cut {check.threshold})"
                    for check in thresholds
                    if check.table1_samples != check.repetitions
                ],
            ).model_dump_json(indent=2)
        )
        (out / "combination_rule.json").write_text(
            json.dumps([check.model_dump() for check in combination], indent=2)
        )
        (out / "perturbation_field_gaps.json").write_text(
            json.dumps(
                [gap.model_dump(mode="json") for gap in PERTURBATION_FIELD_GAPS],
                indent=2,
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
# Verification (L0-L4) of a built tree
# --------------------------------------------------------------------------- #
DATASET_ROOT_REL = "data/torchcell/ecoli_env_chemgen_girgis2009"
#: Records of the full build: 3,821 kept loci x 17 drugs, minus 1,191 ND cells.
EXPECTED_RECORDS = 63766


def verify_build(
    dataset_root: str,
    *,
    genome: EcoliK12Genome | None = None,
    data_root: str | None = None,
    expected_count: int | None = None,
) -> VerificationReport:
    """Run the environment-response L0-L4 gate on a built tree and write its report.

    The LMDB is streamed once. Every record is checked against the MG1655 genome its
    references pin: the resolver of the canonical-name rule, and as the L4 universe every
    GenBank locus of the assembly, pseudogenes included. One resolver cannot serve two
    strains, which is why the host-aware gene set is read from the record's own pinned
    assembly rather than from the yeast runner's S288C default. Every module-level
    ``SourcedValue`` is additionally audited against its pinned artifact when the
    literature mirror is mounted. The report is written to
    ``preprocess/verification_report.json``.

    ``expected_count`` defaults to :data:`EXPECTED_RECORDS`, read when the function runs
    rather than when it is defined, so a build of another size states its own count.
    """
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset_streaming,
    )
    from torchcell.verification.runners import stream_records
    from torchcell.verification.sourced import library_available

    base = data_root or _data_root()
    if expected_count is None:
        expected_count = EXPECTED_RECORDS
    if genome is None:
        genome = bacterial_genome("ecoli", REFERENCE_STRAIN_NAME, base)
    report = verify_environment_response_dataset_streaming(
        stream_records(dataset_root),
        dataset_name=osp.basename(osp.normpath(dataset_root)),
        provenance=Provenance(
            source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/data/{DATASET_S5}",
            citation_key=CITATION_KEY,
            sha256=DATA_SHA256[DATASET_S5],
            method="Dataset S5, the combined z-score of every locus in each of the 17 "
            "antibiotics; one BacterialEnvironmentResponseExperiment per (gene, drug), "
            "gene-level TransposonInsertionPerturbation against MG1655, reference = the "
            "unperturbed MG1655 delta-lacZ parent at 0",
            page="Dataset S5 (pone.0005629.s024.xls), sheet 'S5-Scores of All'",
            retrieved=DATA_RETRIEVED_AT,
        ),
        expected_count=expected_count,
        sgd_genes=set(genome.genbank.loci),
        resolve_gene_name=genome.resolve_gene_name,
    )
    library_root = Path(base) / "torchcell-library"
    if library_available(library_root):
        for value in SOURCED_VALUES.values():
            report.add(audit_sourced_value(value, library_root))
        for spec in DRUGS:
            report.add(audit_sourced_value(spec.dose_sourced(), library_root))
    preprocess = osp.join(dataset_root, "preprocess")
    os.makedirs(preprocess, exist_ok=True)
    with open(osp.join(preprocess, "verification_report.json"), "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI: ``deposit`` the raw mirror, ``build`` the dev LMDB, or ``verify`` it."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        prog="python -m torchcell.datasets.ecoli.girgis2009"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    deposit = sub.add_parser("deposit", help="deposit the raw mirror")
    deposit.add_argument(
        "--retrieve-into",
        default=None,
        help="re-run every recorded PMC retrieval into this directory and deposit those "
        "bytes; without it the literature mirror's captured SI files are used",
    )
    sub.add_parser("build", help="build (or load) the dev-tree LMDB")
    sub.add_parser("verify", help="run L0-L4 on the built dev-tree LMDB")
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = _data_root()
    if args.command == "deposit":
        sources: dict[str, str | Path]
        if args.retrieve_into is not None:
            sources = dict(retrieve_raw_files(args.retrieve_into))
        else:
            sources = {
                name: library_dir(data_root) / "si" / name for name in DATA_SHA256
            }
        print(deposit_raw_mirror(sources=sources, data_root=data_root))
        return 0
    root = osp.join(data_root, DATASET_ROOT_REL)
    if args.command == "build":
        dataset = EnvChemgenGirgis2009Dataset(root=root)
        print(f"len = {len(dataset)}")
        return 0
    report = verify_build(root, data_root=data_root)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
