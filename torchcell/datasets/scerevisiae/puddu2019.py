# torchcell/datasets/scerevisiae/puddu2019
# [[torchcell.datasets.scerevisiae.puddu2019]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/puddu2019
# Test file: tests/torchcell/datasets/scerevisiae/test_puddu2019.py
"""Puddu 2019: the realized genome architecture of 8,835 yeast knockout-collection colonies.

Puddu et al. 2019 (Nature 573:416, doi:10.1038/s41586-019-1549-9; citation key
``pudduGenomeArchitectureStability2019``) whole-genome-sequenced the homozygous diploid
yeast knockout collection (YKOC) and released, per sequenced colony, the copy number of
the rDNA and CUP1 repeat arrays, of mtDNA, of the 2 micron plasmid and of the five Ty
families, a telomere-length proxy, the copy number of all 16 chromosomes, their summed
deviation from diploidy and a count of chromosomal rearrangements.

WHY THIS ROW EXISTS. Half the served yeast rows (Costanzo 2016, Kuzmin 2018/2020,
Kemmeren 2014, Messner 2023, Ohya 2005, Hillenmeyer 2008, Hoepfner 2014) are screens OF
this collection and all of them assume "S288C with one ORF deleted". This release
measures what those strains actually carry, so it is the genotype backbone for them: the
designed perturbation is the deletion, and the realized genome content is what the
sequencing found. The Hoepfner 2014 background already records the gap this closes
("The schema has no per-chromosome copy number, so the aneuploidy and the amplification
cannot be recorded per strain").

RELEASE. Supplementary Tables 1-6, one workbook (``si/si3.xlsx``, sha256
fa369c18a3..., retrieved from the PMC Article Datasets bucket as
``EMS83962-supplement-Supp_Tab.xlsx``). Measured on those bytes:

- Table 1: 9,014 colonies, each with its European Nucleotide Archive sample accession,
  over 4,535 deletion-strain ids, plus one stray trailing cell holding a bare ``2``.
  171 of those colonies are not in Table 3 (sequenced, no released estimates).
- Table 2: the wild-type distribution of 10 features (mean, SD, 3-SD interval) and the
  number of genes and colonies outside it.
- Table 3: 8,843 colonies x 28 numeric estimates. No missing cell, no negative value,
  every ``GCR`` integral, and ``ANtot`` equals the sum over the 16 chromosomes of
  ``|copy number - 2|`` to within 3.1e-15, and every colony carries the same accession
  Table 1 gives it.
- Table 4: 4,436 knockout genes, listing only the per-gene values that deviate from
  wild type. A DERIVED summary of Table 3, so it is not served; it is read for the
  (systematic name, common name) pairs and as an L4 cross-check.
- Table 5: the 21 other strains of the paper's validation experiments (ts alleles,
  reporters). Not a YKOC screen, so not served.
- Table 6: 8,831 colonies with the barcodes actually detected in them plus the
  barcode-CONFIRMED deletion as a systematic name.

RECORD. One ``GenomeArchitectureExperiment`` per sequenced knockout colony: 8,835
records over 4,458 deletion strains and 4,437 released deletion names. The 8,835
knockout colonies split by strain into 258 strains with 1 colony, 4,097 with 2, 30 with
3, 72 with 4 and 1 with 5, which is the Extended Data Fig. 1c legend exactly, and every
one of them is stored.
Colonies are NOT averaged into strains: two colonies of one knockout are two
independently isolated samples whose architectures differ, which is the paper's own unit
of analysis.

The 8 wild-type BY4743 colonies of Table 3 (strain labels ``Del4733_WT-1`` ...
``Del4736_WT-4``) are not perturbation records; they are the REFERENCE. Table 2's
published wild-type mean is their mean (re-derived from Table 3's bytes: rDNA 124.9905
vs the published 124.99, CUP1 14.0340 vs 14.03, mtDNA 17.5041 vs 17.5) and its published
SD is their n-denominator SD (rDNA 7.846 vs the published 7.85), so the reference carries
Table 2's values with ``n_samples = 8`` and their per-colony rows go to
``preprocess/wild_type_colonies.csv``.

GENOTYPE (designed) and ARCHITECTURE (realized). The designed edit is a homozygous
kanMX whole-ORF deletion in BY4743, stored as ``BarcodedKanMxDeletionPerturbation``
carrying the barcodes Table 6 DETECTED in that colony rather than the collection's
expected tags, which is realized detail of its own: of 8,828 served colonies with a
Table 6 row, 5,115 carry an UPTAG matching the reference, 1,023 carry a different one,
1,153 a heterozygous tag and 1,291 none at all. The deleted gene is the colony's own
barcode-confirmed systematic name; for 7 colonies Table 6 has no row and the gene comes
from Table 4's (systematic name, common name) pair for the strain label, with a typed
gap on both barcode fields. Two colonies of one strain can carry DIFFERENT confirmed
deletions (15 strain ids resolve to two genes), so the gene is read per colony, never
per strain.

ORF HISTORY. The release names each deletion as the collection annotated it around
2000, and the loader stores that name verbatim as ``systematic_gene_name``, with no
genome lookup: the colony's Table 6 ``Confirmed_deletion``, or, for the 7 colonies with
no Table 6 row, the systematic name Table 4 pairs with the strain label.
``perturbed_gene_name`` is the common name Table 4 pairs with that systematic name (a
release-internal pairing; 933 Table 4 rows pair a name with itself), or the systematic
name again when Table 4 has no row for it (1 name, YKL075C). Measured on 2026-10-11
against R64-4-1, 24 of the 4,437 released names have since been renamed or absorbed into
a neighbor and 5 are retired (YAR037W, YAR040C, YAR043C, YCL013W, YCL026C, 9 colonies
over 5 strains). Those 9 colonies are stored under the names the authors released, like
every other colony: the genome is a separate layer that meets these records downstream
through the gene set, so a released name the current annotation no longer carries simply
has no gene edge, and no ``ConstructedOrf`` is built at load time.

SNVs and indels have no released table:
Figure 4 analyses them and the Methods state the calling pipeline, but none of the six
supplementary tables carries a variant column (measured: Tables 1-6 hold 3, 6, 31, 12, 5
and 9 columns, listed above), and the per-gene browser at ``sgv.gurdon.cam.ac.uk`` is a
live URL, which is retrieval metadata and not a source. So no
``SequenceVariantPerturbation`` is emitted and the secondary point mutations of these
strains stay unrecorded. The raw reads (9,015 ENA samples, study ERP109205) are not
deposited; the accession travels on every record instead.

NO GENOMES-TIER SET. The genomes tier holds sequence data (assemblies, FASTA panels,
matrices). This release has none: its three publisher files are an SI guide
(``si/si1.docx``, 16,832 bytes, a six-line table of contents), one western-blot image
(``si/si2.tif``, 11,697,348 bytes) and the table workbook (``si/si3.xlsx``, 3,098,936
bytes). The derived per-colony calls are 8,843 x 28 = 247,604 numbers, which are records,
not reference sequence, so they are served through the dataset route and no tier
directory is created.

RAW MIRROR. ``$DATA_ROOT/torchcell-raw/pudduGenomeArchitectureStability2019/si/si3.xlsx``
with a ``pmc_cloud`` retrieval record (the same bucket object, URL and sha256 the
literature mirror recorded on 2026-10-11).
"""

from __future__ import annotations

import json
import logging
import math
import os
import os.path as osp
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.genome_architecture_features import (
    PUDDU2019_WGS_COVERAGE,
    ArchitectureBasis,
)
from torchcell.datamodels.media import YPAD
from torchcell.datamodels.schema import (
    BarcodedKanMxDeletionPerturbation,
    CultureEnvironment,
    CultureFormat,
    EukaryoticStrainBackground,
    Experiment,
    ExperimentReference,
    GenomeArchitectureExperiment,
    GenomeArchitectureExperimentReference,
    GenomeArchitecturePhenotype,
    Genotype,
    Media,
    Publication,
    SampleUnit,
    StrainConstruction,
    StrainReferenceGenome,
    Temperature,
    UncertaintyType,
)
from torchcell.datamodels.strain_background import (
    BRACHMANN_1998,
    KANMX4_CASSETTE,
    standard_background,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_SI_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)
from torchcell.sequence import GeneSet
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

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "pudduGenomeArchitectureStability2019"
PAPER_DOI = "10.1038/s41586-019-1549-9"
PAPER_TITLE = (
    "Genome architecture and stability in the Saccharomyces cerevisiae knockout "
    "collection"
)
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "ba2a94392af73ce98c22dc3916dd25eca7ec91bb8ce9d1f716e16651eb11071f"
SI_TABLES_XLSX = "si/si3.xlsx"
SI_TABLES_XLSX_SHA256 = (
    "fa369c18a31c9ec7f0ade8b63f0be6a595821ac360aa72b1e1dcd46ed8994214"
)
SI_TABLES_FILENAME = "si3.xlsx"
#: The PMC Article Datasets object the workbook was retrieved from, and when.
SI_SOURCE_URL = (
    "https://pmc-oa-opendata.s3.amazonaws.com/PMC6774800.1/"
    "EMS83962-supplement-Supp_Tab.xlsx"
)
SI_BUCKET_KEY = "PMC6774800.1/EMS83962-supplement-Supp_Tab.xlsx"
SI_RETRIEVED_AT = "2026-10-11T10:42:42.804041+00:00"

#: Sheet names, and the header row of each is the first row whose first cell starts with
#: a single ``#`` (the workbook writes its legend rows with ``##``).
ACCESSION_SHEET = "SupplementaryTable1"
WILD_TYPE_SHEET = "SupplementaryTable2"
COLONY_SHEET = "SupplementaryTable3"
GENE_SHEET = "SupplementaryTable4"
BARCODE_SHEET = "SupplementaryTable6"

ENA_STUDY = "ERP109205"
BACKGROUND_STRAIN = "BY4743"
ASSAY = PUDDU2019_WGS_COVERAGE.name
TEMPERATURE_C = 30.0
CULTURE_VOLUME_UL = 1800.0

#: Released rows, measured on the pinned workbook (see the module docstring).
N_ACCESSION_ROWS = 9014
#: Table 1's last row holds a bare ``2`` in the strain column and no colony id: a stray
#: cell, not a record. It is refused by count rather than dropped by a filter, so a
#: re-released workbook with more unparseable rows fails instead of losing colonies.
N_ACCESSION_STRAY_ROWS = 1
N_COLONIES_RELEASED = 8843
N_WILD_TYPE_COLONIES = 8
#: Served records: every one of the 8,835 knockout colonies, under its released name.
N_SERVED_RECORDS = 8835
N_KNOCKOUT_COLONIES = 8835
N_GENE_ROWS = 4436
N_BARCODE_ROWS = 8831
#: Colonies per deletion strain, as the Extended Data Fig. 1c legend states them for the
#: knockout strains. MEASURED on the 8,835 knockout colonies: exactly these five strata,
#: 4,458 strains in all. Table 3's 4,462 strain ids are these plus the four wild-type
#: strain ids, which carry two colonies each and are the reference. The served strata
#: are these same five, since every knockout colony is stored.
COLONIES_PER_STRAIN_LEGEND = {1: 258, 2: 4097, 3: 30, 4: 72, 5: 1}
N_KNOCKOUT_STRAINS = 4458

#: The paper's own summary statistics, re-derived from Table 3 by this loader's ledger.
PAPER_RHO0_COLONIES = 303
PAPER_ANEUPLOID_FRACTION = 0.52


def _sv(
    value: object,
    quote: str,
    *,
    uri: str = PAPER_MD,
    sha: str = PAPER_MD_SHA256,
    page: str | None = None,
    note: str | None = None,
) -> SourcedValue:
    """A SourcedValue pinned to a mirrored Puddu 2019 artifact (quote + sha256)."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri=uri,
            citation_key=CITATION_KEY,
            sha256=sha,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)"
            if uri == PAPER_MD
            else "Supplementary Tables workbook (PMC Article Datasets copy)",
            page=page,
        ),
        quote=quote,
        note=note,
    )


def _cell(
    value: object, quote: str, sheet: str, note: str | None = None
) -> SourcedValue:
    """A SourcedValue on one legend cell of the Supplementary Tables workbook."""
    return _sv(
        value,
        quote,
        uri=SI_TABLES_XLSX,
        sha=SI_TABLES_XLSX_SHA256,
        page=f"sheet {sheet!r}",
        note=note,
    )


# Methods, "Duplication of the YKOC and colony isolation".
COLLECTION_QUOTE = (
    "The homozygous diploid YKOC was acquired from EUROSCARF in 2001, propagated for "
    "two batch expansions (10–20 cell generations) in liquid complete synthetic medium"
)
COLONY_ISOLATION_QUOTE = (
    "From each well, single colonies were isolated on YPAD medium and grown for 2–8 "
    "days at $3 0 ^ { \\circ } \\mathrm { C }$ ."
)
CULTURE_QUOTE = (
    "Two colonies per knockout were inoculated in $1 . 8 \\mathrm { m l }$ YPAD and "
    "grown to saturation for 2–5 days, before collecting by centrifugation."
)
PLATFORM_QUOTE = "Sequencing of the YKOC was carried out on Illumina HiSeqX machines."
ALIGNMENT_QUOTE = (
    "Sequencing reads were aligned to the yeast reference genome "
    "(S288c_R64-1-1_Ensembl) using bwa mem with the following options: -t 16 -p -T 0."
)
DELETION_VALIDATION_QUOTE = (
    "An ORF coverage ${ < } 1 5 \\%$ of the relevant chromosome-wide median was deemed "
    "to indicate full gene deletion."
)
BARCODE_QUOTE = (
    "For each sample, the detected barcodes—observed at least 5 times and for a "
    "frequency ${ > } 2 0 \\%$ —are reported in Supplementary Table\xa06, and noted if "
    "they do not match with the expected barcodes."
)
REASSIGNMENT_QUOTE = (
    "Any sample that did not carry the expected deletion was reassigned to the correct "
    "knockout strain using information encoded in the unique barcodes, when possible."
)
REPEAT_COPY_NUMBER_QUOTE = (
    "Estimates of copy numbers were obtained by dividing the median coverage across "
    "the appropriate genomic region (rDNA, chromosome XII:452000–459000; CUP1, "
    "chromosome VIII: 212986–213525) by the genome-wide median coverage and "
    "multiplying by two (this is because the S288C reference genome contains two "
    "copies of the minimal repeat unit)."
)
TY_COPY_NUMBER_QUOTE = (
    "Median coverage was calculated across the following regions, and normalized by "
    "the genome-wide median to obtain the estimate of the copy number (in number of "
    "copies per haploid genome; in the case of the $2 \\mu$ plasmid these estimates "
    "were multiplied by the expected ploidy to obtain the number of copies per cell; "
    "see ‘Quantification of mtDNA copy number’)."
)
TELOMERE_QUOTE = (
    "A synthetic measure of telomere length was obtained by counting the number of "
    "telomeric reads in both orientations and read pairs, and normalizing by the total "
    "number of reads."
)
PLOIDY_QUOTE = (
    "This was obtained by dividing the coverage across a 10-kb window centred on the "
    "centromere by the genome-wide median and multiplying by 2 (assuming largely "
    "diploid samples)."
)
PLOIDY_ROUNDING_QUOTE = (
    "Ploidy estimates of $< 0 . 2$ and ${ > } 0 . 8$ units were rounded to the closest "
    "unit (see ‘Determination of thresholds for ‘normal’ ploidy’ for details about how "
    "these thresholds were calculated); all others were maintained as rational numbers."
)
MTDNA_QUOTE = (
    "Estimates of copy number were obtained by dividing the median coverage across a "
    "region of the COX1 gene (mito: 14000– 20000) by the genome-wide median and "
    "multiplying by the expected ploidy of the sample"
)
RHO0_QUOTE = "$r h o ^ { 0 }$ colonies were defined as having less than one mtDNA copy."
REARRANGEMENT_QUOTE = (
    "Finally, the number of rearrangements was calculated by counting the number of "
    "regions with abnormal ploidy (greater than (centromere ploidy $+ 0 . 5 )$ ) or "
    "smaller than (centromere ploidy − 0.5)) larger than $2 0 \\mathrm { k b }$ ."
)
BATCH_CORRECTION_QUOTE = (
    "To correct for batch effects caused by different library preparations and "
    "sequencing runs, we calculated for every genomic feature (rDNA, CUP1, Ty1–Ty5, "
    "mtDNA, $2 \\mu$ or telomeres), and for each sequencing run or lane, the median of "
    "the estimates of copy numbers of the approximately 80–96 strains sequenced in that "
    "lane."
)
PLOIDY_CORRECTION_QUOTE = (
    "The copy number of CUP1 and rDNA was corrected for the observed ploidy of "
    "chromosome VIII and XII, respectively, multiplying the raw copy-number value by "
    "the ratio between the expected and observed ploidy of the relevant chromosome."
)
DIPLOIDY_DEFINITION_QUOTE = (
    "We define a strain as deviating from diploidy when any of the 16 chromosomes "
    "deviates from diploidy"
)
REPLICATE_QUOTE = (
    "$n = 8$ biologically independent samples for wildtype strains; $n = 1$ sample for "
    "258 of the knockout strains; $n = 2$ biologically independent samples for 4,097 of "
    "the knockout strains; $n = 3$ biologically independent samples for 30 of the "
    "knockout strains; $n = 4$ biologically independent samples for 72 of the knockout "
    "strains; and $n = 5$ biologically independent samples for 1 of the knockout strains."
)
ANEUPLOIDY_FRACTION_QUOTE = (
    "Notably, about $5 2 \\%$ of the colonies that we sequenced displayed some "
    "deviation from $2 n$ ploidy"
)
MUTATION_CALLING_QUOTE = (
    "Mutation-calling was performed against the EF4.69 revision of the yeast genome "
    "with samtools mpileup with the following options:"
)
ENA_QUOTE = (
    "The primary sequencing data have been deposited in the European Nucleotide Archive "
    "in the study ERP109205, under the accession numbers detailed in Supplementary "
    "Table\xa01."
)
WILD_TYPE_STRAIN_QUOTE = (
    "we estimated the sequence divergence of two colonies of the same knockout strain "
    "from each other, and from the wild-type strain BY4743"
)

#: Every paper-OCR quote this loader pins, for the L3 provenance audit.
PAPER_SOURCED_VALUES: dict[str, SourcedValue] = {
    "collection": _sv(
        "EUROSCARF homozygous diploid YKOC, acquired 2001",
        COLLECTION_QUOTE,
        page="Methods, 'Duplication of the YKOC and colony isolation'",
    ),
    "colony_isolation": _sv(
        "single colonies on YPAD, 2-8 days, 30 C",
        COLONY_ISOLATION_QUOTE,
        page="Methods, 'Duplication of the YKOC and colony isolation'",
    ),
    "culture": _sv(
        CULTURE_VOLUME_UL,
        CULTURE_QUOTE,
        page="Methods, 'Duplication of the YKOC and colony isolation'",
        note="the measured sample is this 1.8 ml liquid YPAD culture, grown to "
        "saturation; the 2-5 day range is per colony and is not released, so "
        "duration_hours is a typed gap",
    ),
    "platform": _sv(
        "Illumina HiSeqX", PLATFORM_QUOTE, page="Methods, 'Genomic DNA extractions'"
    ),
    "alignment": _sv(
        "S288c_R64-1-1_Ensembl",
        ALIGNMENT_QUOTE,
        page="Methods, 'Sequencing read alignment'",
    ),
    "deletion_validation": _sv(
        0.15, DELETION_VALIDATION_QUOTE, page="Methods, 'Validation of gene knockout'"
    ),
    "barcodes": _sv(
        "detected barcodes, >= 5 observations and > 20% frequency",
        BARCODE_QUOTE,
        page="Methods, 'Validation of gene knockout'",
    ),
    "reassignment": _sv(
        "barcode reassignment",
        REASSIGNMENT_QUOTE,
        page="Methods, 'Re-assignment of gene knockouts'",
    ),
    "repeat_copy_number": _sv(
        ArchitectureBasis.copies_per_haploid_genome.value,
        REPEAT_COPY_NUMBER_QUOTE,
        page="Methods, 'Quantification of rDNA and CUP1-repeat copy number'",
    ),
    "ty_copy_number": _sv(
        ArchitectureBasis.copies_per_cell.value,
        TY_COPY_NUMBER_QUOTE,
        page="Methods, 'Quantification of Ty1-Ty5, 2u and definition of genetic "
        "mating type'",
        note="the Ty families are per haploid genome; the 2 micron plasmid and mtDNA "
        "are multiplied by the expected ploidy, so they are per cell",
    ),
    "telomeres": _sv(
        ArchitectureBasis.reads_per_million.value,
        TELOMERE_QUOTE,
        page="Methods, 'Quantification of telomere length'",
    ),
    "ploidy": _sv(
        ArchitectureBasis.chromosome_copies_per_cell.value,
        PLOIDY_QUOTE,
        page="Methods, 'Quantification of chromosome aneuploidy'",
    ),
    "ploidy_rounding": _sv(
        0.2,
        PLOIDY_ROUNDING_QUOTE,
        page="Methods, 'Quantification of chromosome aneuploidy'",
    ),
    "mtdna": _sv(
        ArchitectureBasis.copies_per_cell.value,
        MTDNA_QUOTE,
        page="Methods, 'Quantification of mtDNA copy number'",
    ),
    "rho0": _sv(1.0, RHO0_QUOTE, page="Methods, 'Quantification of mtDNA copy number'"),
    "rearrangements": _sv(
        ArchitectureBasis.rearrangement_count.value,
        REARRANGEMENT_QUOTE,
        page="Methods, 'Quantification of chromosomal rearrangements'",
    ),
    "batch_correction": _sv(
        "per-lane median normalization",
        BATCH_CORRECTION_QUOTE,
        page="Methods, 'Correction of batch effects'",
    ),
    "ploidy_correction": _sv(
        "rDNA and CUP1 corrected for the ploidy of chromosomes XII and VIII",
        PLOIDY_CORRECTION_QUOTE,
        page="Methods, 'rDNA and CUP1 copy-number correction'",
    ),
    "diploidy_definition": _sv(
        2.0,
        DIPLOIDY_DEFINITION_QUOTE,
        page="Methods, 'Determination of thresholds for normal ploidy'",
        note="the reference value of every chromosome copy number: the expected ploidy "
        "of this collection is 2n, which is what a deviation is measured against",
    ),
    "replicates": _sv(
        COLONIES_PER_STRAIN_LEGEND,
        REPLICATE_QUOTE,
        page="Extended Data Fig. 1c legend",
        note="the 8,835 knockout colonies reproduce these five strata exactly (258 / "
        "4,097 / 30 / 72 / 1 strains with 1 / 2 / 3 / 4 / 5 colonies, 4,458 strains); "
        "Table 3's 4,462 strain ids are these plus the four wild-type ids, which the "
        "legend counts separately as its 8 wild-type samples",
    ),
    "aneuploid_fraction": _sv(
        PAPER_ANEUPLOID_FRACTION,
        ANEUPLOIDY_FRACTION_QUOTE,
        page="main text",
        note="measured on Table 3: 4,607 of 8,843 colonies (52.10%) carry a chromosome "
        "away from 2.0",
    ),
    "mutation_calling": _sv(
        "samtools mpileup / bcftools call",
        MUTATION_CALLING_QUOTE,
        page="Methods, 'Analysis of SNVs and indels'",
        note="the pipeline is stated and the calls are NOT released as a table; this "
        "quote is what the refusal of SNV perturbations rests on",
    ),
    "ena": _sv(ENA_STUDY, ENA_QUOTE, page="Data availability"),
    "wild_type_strain": _sv(
        BACKGROUND_STRAIN, WILD_TYPE_STRAIN_QUOTE, page="main text"
    ),
}

#: Legend cells of the workbook this loader pins (audited as sheet cell strings).
WORKBOOK_SOURCED_VALUES: dict[str, SourcedValue] = {
    "colony_id": _cell(
        "SDname", "Unique colony ID", COLONY_SHEET, note="the stored sample_id"
    ),
    "accession": _cell(
        "ERSno",
        "Accession code to download data from the European Nucleotide Archive",
        COLONY_SHEET,
        note="the stored sample_accession",
    ),
    "deletion_strain": _cell(
        "Deletion",
        "Unique ID identifying the KO strain from which colonies were derived",
        COLONY_SHEET,
    ),
    "telomere_units": _cell(
        ArchitectureBasis.reads_per_million.value,
        "Indirect measure of telomere length (in number of telomeric reads per million)",
        COLONY_SHEET,
    ),
    "ploidy_deviation": _cell(
        ArchitectureBasis.ploidy_deviation_units.value,
        "Sum of all the deviation from diploidy for the 16 chromosomes",
        COLONY_SHEET,
    ),
    "rearrangements": _cell(
        ArchitectureBasis.rearrangement_count.value,
        "Number of chromosomal rearrangements detected",
        COLONY_SHEET,
    ),
    "confirmed_deletion": _cell(
        "Confirmed_deletion",
        "Confirmed gene deletion (systematic name)",
        BARCODE_SHEET,
        note="the realized deleted gene of one colony, read from its barcodes",
    ),
    "uptag": _cell(
        "UPTAG_seqs", "Detected UPTAG sequences (comma separated list)", BARCODE_SHEET
    ),
    "dntag": _cell(
        "DNTAG_seqs", "Detected DNTAG sequence (comma separated list)", BARCODE_SHEET
    ),
    "wild_type_mean": _cell("WT_mean", "Average of wild-type samples", WILD_TYPE_SHEET),
    "wild_type_sd": _cell(
        UncertaintyType.sample_sd.value,
        "Standard deviation of wild-type samples",
        WILD_TYPE_SHEET,
        note="measured: the published number is the n-denominator SD of the 8 "
        "wild-type colonies of Table 3 (rDNA 7.846 printed as 7.85), i.e. a standard "
        "deviation of observations and not a standard error",
    ),
}

#: Table 2's feature label -> the Table 3 column (and assay symbol) it summarizes.
WILD_TYPE_FEATURE_COLUMN: dict[str, str] = {
    "rDNA": "rDNA",
    "CUP1": "CUP1",
    "mtDNA": "mitochondria",
    "2mic": "2-micron",
    "Ty1": "Ty1",
    "Ty2": "Ty2",
    "Ty3": "Ty3",
    "TEL": "Telomeres",
    "ANEUP": "ANtot",
    "CR": "GCR",
}

#: Features of the assay that Table 2 gives no wild-type statistic for.
WILD_TYPE_MISSING_FEATURES = ("Ty4", "Ty5")

#: Strain-label prefix of the four wild-type strain ids of Table 3.
WILD_TYPE_LABEL = re.compile(r"^Del\d+_WT-\d+$")
#: ``Del<number>_<label>`` as Table 3 writes a strain id.
STRAIN_ID = re.compile(r"^Del(?P<number>\d+)_(?P<label>.+)$")


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/pudduGenomeArchitectureStability2019``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def library_workbook(data_root: str | None = None) -> Path:
    """The literature mirror's copy of the Supplementary Tables workbook."""
    return (
        Path(data_root or _data_root())
        / "torchcell-library"
        / CITATION_KEY
        / SI_TABLES_XLSX
    )


def deposit_raw_mirror(*, workbook: str | Path, data_root: str | None = None) -> Path:
    """Deposit the Supplementary Tables workbook as the raw mirror, with its manifest.

    Run once from the literature mirror's copy: the same bucket object the literature
    retrieval recorded, so the raw record carries that retrieval verbatim. Idempotent by
    sha256; a file already there with a different hash raises.
    """
    source = Path(workbook)
    sha = sha256_file(source)
    if sha != SI_TABLES_XLSX_SHA256:
        raise ValueError(
            f"{source} has sha256 {sha}, the loader pins {SI_TABLES_XLSX_SHA256}"
        )
    root = raw_mirror_dir(data_root)
    dest = root / SI_TABLES_XLSX
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if sha256_file(dest) != sha:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    else:
        shutil.copy2(source, dest)
    record = ArtifactRecord(
        path=SI_TABLES_XLSX,
        role=ROLE_SI_DATA,
        bytes=dest.stat().st_size,
        sha256=sha,
        source=SI_SOURCE_URL,
        original_filename="EMS83962-supplement-Supp_Tab.xlsx",
        retrieval=RetrievalRecord(
            method=RetrievalMethod.pmc_cloud,
            source_url=SI_SOURCE_URL,
            retriever="torchcell.literature.retrieve.pmc_cloud_object",
            params={"key": SI_BUCKET_KEY},
            sha256=sha,
            retrieved_at=SI_RETRIEVED_AT,
        ),
    )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=PAPER_TITLE,
        files=[record],
        si_data_sources=[SI_SOURCE_URL],
        si_expected=[
            "Supplementary Tables 1-6 (colony accessions, the wild-type distribution, "
            "per-colony repetitive-DNA and ploidy estimates, the per-gene summary, the "
            "other strains used, and the detected deletion barcodes)"
        ],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return root


def load_manifest(data_root: str | None = None) -> Manifest:
    """Read the raw mirror's ``manifest.json``."""
    return Manifest.model_validate_json(
        (raw_mirror_dir(data_root) / "manifest.json").read_text()
    )


def manifest_sha256(manifest: Manifest, relpath: str) -> str:
    """The recorded sha256 of one mirror file."""
    for record in manifest.files:
        if record.path == relpath:
            return record.sha256
    raise KeyError(f"{relpath} is not in the raw-mirror manifest")


# --------------------------------------------------------------------------- #
# Workbook reading
# --------------------------------------------------------------------------- #
class ReleaseShapeError(ValueError):
    """The workbook does not have the shape the paper and this loader state."""


def read_sheet(path: str | Path, sheet: str) -> pd.DataFrame:
    """One supplementary table, with its legend rows dropped.

    The workbook writes legend rows with a leading ``##`` and the header row with a
    single ``#``; everything below the header is data. Refuses a sheet with no such
    header row rather than guessing one.
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=None, dtype=object)
    first = raw[0].astype(str)
    header_rows = [
        i
        for i, value in enumerate(first)
        if value.startswith("#") and not value.startswith("##")
    ]
    if not header_rows:
        raise ReleaseShapeError(f"sheet {sheet!r} has no '#' header row")
    index = header_rows[0]
    columns = [
        str(c).lstrip("#")
        for c in raw.iloc[index].tolist()
        if c is not None and str(c) != "nan"
    ]
    frame = raw.iloc[index + 1 :, : len(columns)].copy()
    frame.columns = columns
    frame = frame.dropna(how="all").reset_index(drop=True)
    return frame


def read_colonies(path: str | Path) -> pd.DataFrame:
    """Table 3: one row per sequenced colony, every value checked against its basis.

    Refuses a shape other than the released one, a duplicated colony id, a missing cell,
    a negative or non-finite estimate, a non-integral rearrangement count, and an
    ``ANtot`` that is not the sum of its own chromosome deviations.
    """
    frame = read_sheet(path, COLONY_SHEET)
    if len(frame) != N_COLONIES_RELEASED:
        raise ReleaseShapeError(
            f"Table 3 has {len(frame)} colonies, expected {N_COLONIES_RELEASED}"
        )
    symbols = list(PUDDU2019_WGS_COVERAGE.by_symbol)
    expected = {"SDname", "ERSno", "Deletion", *symbols}
    if set(frame.columns) != expected:
        raise ReleaseShapeError(
            f"Table 3 columns {sorted(set(frame.columns) ^ expected)} differ from the "
            "assay vocabulary"
        )
    frame["SDname"] = frame["SDname"].astype(str)
    if not frame["SDname"].is_unique:
        raise ReleaseShapeError("a colony id appears twice in Table 3")
    frame["ERSno"] = frame["ERSno"].astype(str)
    frame["Deletion"] = frame["Deletion"].astype(str)
    for symbol in symbols:
        frame[symbol] = frame[symbol].astype(float)
        if frame[symbol].isna().any():
            raise ReleaseShapeError(f"Table 3 column {symbol} has a missing cell")
        for value in frame[symbol].tolist():
            PUDDU2019_WGS_COVERAGE.validate_value(symbol, value)
    chromosomes = sorted(
        PUDDU2019_WGS_COVERAGE.symbols_with_basis(
            ArchitectureBasis.chromosome_copies_per_cell
        )
    )
    deviation = (frame[chromosomes] - 2.0).abs().sum(axis=1)
    worst = float((deviation - frame["ANtot"]).abs().max())
    if worst > 1e-6:
        raise ReleaseShapeError(
            f"Table 3 ANtot disagrees with its chromosome deviations by {worst}"
        )
    return frame


def read_accessions(path: str | Path) -> dict[str, str]:
    """Table 1: ``{colony id: ENA sample accession}``."""
    frame = read_sheet(path, ACCESSION_SHEET)
    stray = int(frame["SD_code"].isna().sum())
    if stray != N_ACCESSION_STRAY_ROWS:
        raise ReleaseShapeError(
            f"Table 1 has {stray} rows with no colony id, expected "
            f"{N_ACCESSION_STRAY_ROWS}"
        )
    frame = frame.dropna(subset=["SD_code"])
    if len(frame) != N_ACCESSION_ROWS:
        raise ReleaseShapeError(
            f"Table 1 has {len(frame)} colony rows, expected {N_ACCESSION_ROWS}"
        )
    frame["SD_code"] = frame["SD_code"].astype(str)
    accessions = dict(
        zip(frame["SD_code"], frame["ERS accession number"].astype(str), strict=True)
    )
    if len(accessions) != len(frame):
        raise ReleaseShapeError("Table 1 lists a colony twice")
    return accessions


def read_confirmed_deletions(path: str | Path) -> pd.DataFrame:
    """Table 6: per colony, the confirmed deletion and the barcodes detected in it."""
    frame = read_sheet(path, BARCODE_SHEET)
    if len(frame) != N_BARCODE_ROWS:
        raise ReleaseShapeError(
            f"Table 6 has {len(frame)} rows, expected {N_BARCODE_ROWS}"
        )
    frame["SDname"] = frame["SDname"].astype(str)
    if not frame["SDname"].is_unique:
        raise ReleaseShapeError("a colony id appears twice in Table 6")
    return frame


def read_gene_names(path: str | Path) -> dict[str, str]:
    """Table 4: ``{common name: systematic name}``, the release's own gene mapping."""
    frame = read_sheet(path, GENE_SHEET)
    if len(frame) != N_GENE_ROWS:
        raise ReleaseShapeError(
            f"Table 4 has {len(frame)} rows, expected {N_GENE_ROWS}"
        )
    mapping: dict[str, str] = {}
    for systematic, common in zip(
        frame["Sysname"].astype(str), frame["common_name"].astype(str), strict=True
    ):
        if common in mapping and mapping[common] != systematic:
            raise ReleaseShapeError(
                f"Table 4 maps the common name {common!r} to two systematic names"
            )
        mapping[common] = systematic
    return mapping


def workbook_cell_strings(path: str | Path) -> frozenset[str]:
    """Every cell of every sheet of the workbook, as a string.

    What the L3 quote audit compares a pinned legend cell against: a legend cell is not
    text in a document, so ``audit_sourced_value``'s substring search over file bytes
    cannot see it (the xlsx is a zip), and a cell must match a cell exactly.
    """
    cells: set[str] = set()
    sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
    for frame in sheets.values():
        for value in frame.to_numpy().ravel().tolist():
            if value is None:
                continue
            text = str(value)
            if text != "nan":
                cells.add(text)
    return frozenset(cells)


def wild_type_statistic(distribution: pd.DataFrame, feature: str, column: str) -> float:
    """One cell of Table 2 as a float (``WT_mean`` or ``WT_StDev`` of one feature)."""
    return float(str(distribution.at[feature, column]))


def read_wild_type_distribution(path: str | Path) -> pd.DataFrame:
    """Table 2: the wild-type mean and SD of the 10 features it covers."""
    frame = read_sheet(path, WILD_TYPE_SHEET)
    frame["Repeat"] = frame["Repeat"].astype(str)
    missing = set(WILD_TYPE_FEATURE_COLUMN) - set(frame["Repeat"])
    if missing:
        raise ReleaseShapeError(f"Table 2 is missing the features {sorted(missing)}")
    return frame.set_index("Repeat")


# --------------------------------------------------------------------------- #
# Environment, background, reference
# --------------------------------------------------------------------------- #
def puddu_media() -> Media:
    """Liquid YPAD, the medium the sequenced culture was grown in.

    The library ``YPAD`` object is the solid form; the measured culture is 1.8 ml of
    liquid YPAD, so this is that object's composition in the liquid state with the
    paper's own sentence added. ``base_medium`` names the library key, so the two join.
    """
    return Media(
        name="YPAD (YPD + adenine), liquid",
        state="liquid",
        is_synthetic=False,
        base_medium="YPAD",
        components=list(YPAD.components),
        provenance=[*YPAD.provenance, PAPER_SOURCED_VALUES["culture"]],
    )


def puddu_environment() -> CultureEnvironment:
    """The 1.8 ml liquid YPAD culture at 30 C, grown to saturation."""
    return CultureEnvironment(
        media=puddu_media(),
        temperature=Temperature(value=TEMPERATURE_C),
        aerobicity="aerobic",
        duration_hours=None,
        culture_format=CultureFormat(
            working_volume_ul=CULTURE_VOLUME_UL,
            provenance=[
                PAPER_SOURCED_VALUES["culture"],
                PAPER_SOURCED_VALUES["colony_isolation"],
            ],
        ),
        provenance_gaps=[
            ProvenanceGap(
                field="duration_hours",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="the culture was 'grown to saturation for 2–5 days' and which "
                "colony took which is not released, so no single duration is stated; "
                "the release also carries no EndpointRule member for 'until the culture "
                "saturated' (the enum's until_control_saturation is about a vehicle "
                "control), so the statement rides on culture_format.provenance",
            )
        ],
    )


def puddu_background() -> EukaryoticStrainBackground:
    """BY4743 with its typed alleles, the collection's diploid background.

    The alleles are the standard BY4743 set, asserted with a pending-review gap naming
    Brachmann 1998 (not mirrored), exactly as the Hillenmeyer and Hoepfner backgrounds
    do. The paper states the background strain name and that the collection is the
    homozygous diploid YKOC from EUROSCARF; how the diploids were made is Giaever 2002.
    """
    alleles = standard_background(
        BACKGROUND_STRAIN,
        resolve_with=BRACHMANN_1998,
        note="the BY4741 x BY4742 genotype is not in a mirrored source; the background "
        "name BY4743 is sourced to the paper's own hyper-mutator comparison",
    ).alleles
    return EukaryoticStrainBackground(
        reference_strain="S288C",
        assembly_set=SGD_S288C_R64,
        name=BACKGROUND_STRAIN,
        parents=["BY4741", "BY4742"],
        construction="homozygous diploid yeast knockout collection, acquired from "
        "EUROSCARF in 2001 and propagated for two batch expansions; how the homozygous "
        "diploids were built (mating MATa x MATalpha deletants) is Giaever 2002, which "
        "is not mirrored",
        mating_type=None,
        ploidy="diploid",
        alleles=alleles,
        provenance=[
            PAPER_SOURCED_VALUES["wild_type_strain"],
            PAPER_SOURCED_VALUES["collection"],
        ],
        provenance_gaps=[
            ProvenanceGap(
                field="mating_type",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="the paper states the collection is homozygous diploid and never "
                "writes the MAT locus state of BY4743",
            )
        ],
    )


def reference_genome() -> StrainReferenceGenome:
    """BY4743 as the diploid baseline every chromosome copy number is read against."""
    return StrainReferenceGenome(
        species="Saccharomyces cerevisiae",
        strain=BACKGROUND_STRAIN,
        ploidy="diploid",
        background=puddu_background(),
        assembly_set=SGD_S288C_R64,
        assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
    )


def wild_type_profile(distribution: pd.DataFrame) -> GenomeArchitecturePhenotype:
    """Table 2's wild-type profile: the reference every record is compared against.

    The 10 features Table 2 covers carry its published mean and SD; the 16 chromosomes
    carry the expected diploid copy number the Methods define a deviation against. Ty4
    and Ty5 have no Table 2 row, so they are a per-key typed gap rather than a value
    computed here.
    """
    values: dict[str, float] = {}
    dispersion: dict[str, float] = {}
    for feature, column in WILD_TYPE_FEATURE_COLUMN.items():
        values[column] = wild_type_statistic(distribution, feature, "WT_mean")
        dispersion[column] = wild_type_statistic(distribution, feature, "WT_StDev")
    for symbol in PUDDU2019_WGS_COVERAGE.symbols_with_basis(
        ArchitectureBasis.chromosome_copies_per_cell
    ):
        values[symbol] = 2.0
    # The published ANEUP mean is the wild-type colonies' own mean deviation from 2n,
    # which is NOT zero: those colonies deviate slightly too. Stored beside the expected
    # per-chromosome value of 2.0 it would assert that a sum of zero deviations equals
    # it, which is false, so the aggregate is dropped from the reference profile (its
    # published mean and SD are in the ledger) and the gap below says so.
    aggregate_mean = values.pop("ANtot")
    aggregate_sd = dispersion.pop("ANtot")
    phenotype = GenomeArchitecturePhenotype(
        assay=ASSAY,
        genome_architecture=values,
        genome_architecture_sd=dispersion,
        uncertainty_type=UncertaintyType.sample_sd,
        n_samples=N_WILD_TYPE_COLONIES,
        sample_unit=SampleUnit.colony,
        provenance_gaps=[
            ProvenanceGap(
                field="genome_architecture",
                reason=ProvenanceGapReason.not_reported_by_primary,
                keys=list(WILD_TYPE_MISSING_FEATURES),
                note="Table 2 gives a wild-type mean and SD for 10 features and no row "
                "for Ty4 or Ty5, although Table 3 releases both per colony",
            ),
            ProvenanceGap(
                field="genome_architecture",
                reason=ProvenanceGapReason.not_reported_by_primary,
                keys=["ANtot"],
                note="Table 2's wild-type ANEUP mean is "
                f"{aggregate_mean} chromosome units (SD {aggregate_sd}), i.e. the "
                "wild-type colonies themselves deviate slightly from 2n, so the "
                "aggregate cannot be stored beside the expected per-chromosome value "
                "of 2.0 without asserting a sum that is false; the published pair is "
                "in preprocess/release_ledger.json",
            ),
        ],
    )
    return phenotype


def publication() -> Publication:
    """The Nature paper."""
    return Publication(doi=PAPER_DOI, doi_url=f"https://doi.org/{PAPER_DOI}")


# --------------------------------------------------------------------------- #
# Genotype (the DESIGNED edit, with the barcodes actually detected)
# --------------------------------------------------------------------------- #
_NO_TAG_NOTE = (
    "Table 6 reports no detected tag for this colony (the UPTAG/DNTAG note reads 'No "
    "tags detected', or the tag column is empty); the deletion itself was confirmed by "
    "loss of ORF coverage"
)
_NO_ROW_NOTE = (
    "Table 6 carries no row for this colony, so no detected barcode is released for it; "
    "its deleted gene is the systematic name Table 4 pairs with the strain label"
)


def _tags(cell: object) -> str | None:
    """The detected tag string of one Table 6 cell, or None when nothing was detected.

    A cell holding several comma-separated tags is kept verbatim: the release states
    both, and picking one would assert a resolution the source did not make.
    """
    if cell is None:
        return None
    text = str(cell).strip()
    if not text or text in {"nan", "NA", "None"}:
        return None
    return text


def deletion_perturbation(
    *,
    systematic_gene_name: str,
    perturbed_gene_name: str,
    strain_id: str,
    uptag: str | None,
    dntag: str | None,
    has_barcode_row: bool,
) -> BarcodedKanMxDeletionPerturbation:
    """The colony's designed edit: a homozygous kanMX deletion with its detected tags.

    ``systematic_gene_name`` is the released deletion name verbatim, and
    ``perturbed_gene_name`` is the common name Table 4 pairs with it (the systematic name
    again when Table 4 pairs none). Neither is the strain label: the label is a strain
    designation and in 2 colonies it names a different gene than the barcodes
    confirmed. The label itself is kept verbatim in ``construction.strain_accession``
    and in ``preprocess/data.csv``. No ``ConstructedOrf`` is attached: what the release
    names is what is stored, and the genome meets it downstream.
    """
    gaps: list[ProvenanceGap] = []
    note = _NO_ROW_NOTE if not has_barcode_row else _NO_TAG_NOTE
    for field, tag in (("barcode", uptag), ("downtag_barcode", dntag)):
        if tag is None:
            gaps.append(
                ProvenanceGap(
                    field=field,
                    reason=ProvenanceGapReason.not_reported_by_primary,
                    note=note,
                )
            )
    return BarcodedKanMxDeletionPerturbation(
        systematic_gene_name=systematic_gene_name,
        perturbed_gene_name=perturbed_gene_name,
        barcode=uptag,
        downtag_barcode=dntag,
        collection="homozygous diploid YKOC (EUROSCARF, acquired 2001)",
        cassette=str(KANMX4_CASSETTE.value),
        construction=StrainConstruction(strain_accession=strain_id),
        constructed_orf=None,
        provenance_gaps=gaps,
    )


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #
class ReleaseLedger(BaseModel):
    """Every released row, accounted for (``preprocess/release_ledger.json``)."""

    model_config = ConfigDict(extra="forbid")

    workbook_sha256: str
    n_accession_rows: int
    n_colonies_released: int
    n_colonies_served: int
    n_wild_type_colonies: int
    n_barcode_rows: int
    n_gene_rows: int
    n_strains_served: int
    colonies_per_strain: dict[str, int]
    n_colonies_without_barcode_row: int
    n_colonies_gene_from_table4: int
    n_colonies_label_disagrees_with_barcode: int
    n_genes_without_table4_row: int
    uptag_notes: dict[str, int]
    n_rho0_colonies: int
    paper_rho0_colonies: int
    n_aneuploid_colonies: int
    aneuploid_fraction: float
    wild_type_mean_published: dict[str, float]
    wild_type_mean_from_table3: dict[str, float]
    wild_type_sd_published: dict[str, float]
    wild_type_sd_from_table3: dict[str, float]
    n_genes_served: int

    @model_validator(mode="after")
    def _accounted(self) -> ReleaseLedger:
        if (
            self.n_colonies_served + self.n_wild_type_colonies
            != self.n_colonies_released
        ):
            raise ValueError("served + wild type must equal released")
        return self


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@register_dataset
class GenomeArchitecturePuddu2019Dataset(ExperimentDataset):
    """Puddu 2019: one record per sequenced knockout colony, 8,835 records."""

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/genome_architecture_puddu2019",
        io_workers: int = 0,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Build or load the store under ``root``. No genome is consulted."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return GenomeArchitectureExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return GenomeArchitectureExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The Supplementary Tables workbook, the only file this loader consumes."""
        return [SI_TABLES_FILENAME]

    def download(self) -> None:
        """Link the workbook out of the raw mirror, sha256-verified.

        The mirror manifest must record the digest this loader pins
        (``ManifestPinMismatchError`` otherwise), so a re-deposited file with other
        bytes cannot be read as this release.
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        check_manifest_pin(
            SI_TABLES_XLSX,
            manifest_sha256(manifest, SI_TABLES_XLSX),
            SI_TABLES_XLSX_SHA256,
        )
        os.makedirs(self.raw_dir, exist_ok=True)
        link_verified(
            raw_mirror_dir(data_root) / SI_TABLES_XLSX,
            osp.join(self.raw_dir, SI_TABLES_FILENAME),
            SI_TABLES_XLSX_SHA256,
        )
        log.info("Puddu 2019 workbook linked into %s (sha256 verified)", self.raw_dir)

    # ---- build ---------------------------------------------------------------
    @post_process
    def process(self) -> None:
        """Build one record per sequenced knockout colony, strain-major."""
        verify_raw_files(self.raw_dir, {SI_TABLES_FILENAME: SI_TABLES_XLSX_SHA256})
        os.makedirs(self.preprocess_dir, exist_ok=True)
        workbook = osp.join(self.raw_dir, SI_TABLES_FILENAME)
        colonies = read_colonies(workbook)
        accessions = read_accessions(workbook)
        barcodes = read_confirmed_deletions(workbook).set_index("SDname")
        gene_of_common = read_gene_names(workbook)
        distribution = read_wild_type_distribution(workbook)

        missing = sorted(set(colonies["SDname"]) - set(accessions))
        if missing:
            raise ReleaseShapeError(
                f"{len(missing)} Table 3 colonies have no Table 1 accession: {missing}"
            )
        disagree = [
            colony
            for colony, released in zip(
                colonies["SDname"], colonies["ERSno"], strict=True
            )
            if accessions[colony] != released
        ]
        if disagree:
            raise ReleaseShapeError(
                f"{len(disagree)} colonies have different accessions in Tables 1 and 3"
            )

        symbols = list(PUDDU2019_WGS_COVERAGE.by_symbol)
        reference = GenomeArchitectureExperimentReference(
            dataset_name=self.name,
            genome_reference=reference_genome(),
            environment_reference=puddu_environment(),
            phenotype_reference=wild_type_profile(distribution),
        )
        environment = puddu_environment()
        pub = publication()

        wild_type = colonies[colonies["Deletion"].str.match(WILD_TYPE_LABEL)]
        if len(wild_type) != N_WILD_TYPE_COLONIES:
            raise ReleaseShapeError(
                f"{len(wild_type)} wild-type colonies in Table 3, expected "
                f"{N_WILD_TYPE_COLONIES}"
            )
        served = colonies.drop(index=wild_type.index).reset_index(drop=True)
        if len(served) != N_KNOCKOUT_COLONIES:
            raise ReleaseShapeError(
                f"{len(served)} knockout colonies in Table 3, expected "
                f"{N_KNOCKOUT_COLONIES}"
            )
        per_strain = served["Deletion"].value_counts().value_counts()
        strata = dict(
            zip(per_strain.index.astype(int).tolist(), per_strain.astype(int).tolist())
        )
        if strata != COLONIES_PER_STRAIN_LEGEND:
            raise ReleaseShapeError(
                f"colonies per knockout strain {sorted(strata.items())} differ from the "
                f"Extended Data Fig. 1c legend {sorted(COLONIES_PER_STRAIN_LEGEND.items())}"
            )
        if served["Deletion"].nunique() != N_KNOCKOUT_STRAINS:
            raise ReleaseShapeError(
                f"{served['Deletion'].nunique()} knockout strains, expected "
                f"{N_KNOCKOUT_STRAINS}"
            )

        released_genes = self._released_gene_of_colony(served, barcodes, gene_of_common)
        common_of_gene = {
            systematic: common for common, systematic in gene_of_common.items()
        }
        gene_from_table4 = 0
        label_disagrees = 0
        without_row = 0
        gene_names: set[str] = set()
        rows: list[dict[str, Any]] = []
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for row in tqdm(
                served.to_dict("records"), total=len(served), desc="colonies"
            ):
                colony = str(row["SDname"])
                strain_id = str(row["Deletion"])
                label = self._strain_label(strain_id)
                has_row = colony in barcodes.index
                if has_row:
                    gene = str(barcodes.loc[colony, "Confirmed_deletion"])
                    uptag = _tags(barcodes.loc[colony, "UPTAG_seqs"])
                    dntag = _tags(barcodes.loc[colony, "DNTAG_seqs"])
                    if gene_of_common.get(label, gene) != gene:
                        label_disagrees += 1
                else:
                    without_row += 1
                    if label not in gene_of_common:
                        raise ReleaseShapeError(
                            f"colony {colony} has no Table 6 row and its label "
                            f"{label!r} is not a Table 4 common name"
                        )
                    gene = gene_of_common[label]
                    gene_from_table4 += 1
                    uptag = None
                    dntag = None
                if released_genes[colony] != gene:
                    raise ReleaseShapeError(
                        f"colony {colony}: released deletion {gene!r} differs from "
                        f"{released_genes[colony]!r}"
                    )
                perturbation = deletion_perturbation(
                    systematic_gene_name=gene,
                    perturbed_gene_name=common_of_gene.get(gene, gene),
                    strain_id=strain_id,
                    uptag=uptag,
                    dntag=dntag,
                    has_barcode_row=has_row,
                )
                gene_names.add(gene)
                phenotype = GenomeArchitecturePhenotype(
                    assay=ASSAY,
                    genome_architecture={
                        symbol: float(row[symbol]) for symbol in symbols
                    },
                    n_samples=1,
                    sample_unit=SampleUnit.colony,
                    sample_id=colony,
                    sample_accession=str(row["ERSno"]),
                )
                experiment = GenomeArchitectureExperiment(
                    dataset_name=self.name,
                    genotype=Genotype(perturbations=[perturbation]),
                    environment=environment,
                    phenotype=phenotype,
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, pub, itxn),
                )
                rows.append(
                    {
                        "colony_id": colony,
                        "strain_id": strain_id,
                        "systematic_gene_name": gene,
                        "perturbed_gene_name": common_of_gene.get(gene, gene),
                        "strain_label": label,
                        "ena_sample_accession": str(row["ERSno"]),
                        "barcode_row": has_row,
                    }
                )
                idx += 1
        env.close()
        interned_env.close()
        if idx != N_SERVED_RECORDS:
            raise ReleaseShapeError(f"wrote {idx} records, expected {N_SERVED_RECORDS}")
        self._gene_names = gene_names
        pd.DataFrame(rows).to_csv(
            osp.join(self.preprocess_dir, "data.csv"), index=False
        )
        wild_type.to_csv(
            osp.join(self.preprocess_dir, "wild_type_colonies.csv"), index=False
        )
        self._write_ledger(
            colonies=colonies,
            served=served,
            wild_type=wild_type,
            barcodes=barcodes,
            distribution=distribution,
            gene_names=gene_names,
            without_row=without_row,
            gene_from_table4=gene_from_table4,
            label_disagrees=label_disagrees,
            n_genes_without_table4_row=len(gene_names - set(common_of_gene)),
            served_rows=rows,
        )
        log.info(
            "Wrote %d Puddu 2019 colony records over %d genes", idx, len(gene_names)
        )

    def _released_gene_of_colony(
        self,
        served: pd.DataFrame,
        barcodes: pd.DataFrame,
        gene_of_common: dict[str, str],
    ) -> dict[str, str]:
        """``{colony id: released systematic deletion name}`` for the served colonies.

        The colony's own Table 6 row is the authority (the barcodes confirm WHICH gene
        that colony lacks); a colony with no Table 6 row falls back to the systematic
        name Table 4 pairs with its strain label, which is release-internal too.
        """
        out: dict[str, str] = {}
        for colony, strain_id in zip(
            served["SDname"].astype(str), served["Deletion"].astype(str), strict=True
        ):
            if colony in barcodes.index:
                out[colony] = str(barcodes.loc[colony, "Confirmed_deletion"])
                continue
            label = self._strain_label(strain_id)
            if label not in gene_of_common:
                raise ReleaseShapeError(
                    f"colony {colony} has no Table 6 row and its label {label!r} is "
                    "not a Table 4 common name"
                )
            out[colony] = gene_of_common[label]
        return out

    @staticmethod
    def _strain_label(strain_id: str) -> str:
        """The gene label inside a ``Del<number>_<label>`` strain id."""
        match = STRAIN_ID.match(strain_id)
        if match is None:
            raise ReleaseShapeError(
                f"strain id {strain_id!r} is not Del<number>_<label>"
            )
        return match.group("label")

    def _write_ledger(
        self,
        *,
        colonies: pd.DataFrame,
        served: pd.DataFrame,
        wild_type: pd.DataFrame,
        barcodes: pd.DataFrame,
        distribution: pd.DataFrame,
        gene_names: set[str],
        without_row: int,
        gene_from_table4: int,
        label_disagrees: int,
        n_genes_without_table4_row: int,
        served_rows: list[dict[str, Any]],
    ) -> None:
        """Account for every released row, and re-derive the paper's own statistics."""
        per_strain = pd.Series([row["strain_id"] for row in served_rows]).value_counts()
        histogram = per_strain.value_counts().to_dict()
        served_colonies = [str(row["colony_id"]) for row in served_rows]
        notes = (
            barcodes["UPTAG_notes"]
            .reindex(barcodes.index.intersection(served_colonies))
            .astype(str)
            .value_counts()
            .to_dict()
        )
        chromosomes = sorted(
            PUDDU2019_WGS_COVERAGE.symbols_with_basis(
                ArchitectureBasis.chromosome_copies_per_cell
            )
        )
        aneuploid = int((colonies[chromosomes] != 2.0).any(axis=1).sum())
        published_mean = {
            column: wild_type_statistic(distribution, feature, "WT_mean")
            for feature, column in WILD_TYPE_FEATURE_COLUMN.items()
        }
        published_sd = {
            column: wild_type_statistic(distribution, feature, "WT_StDev")
            for feature, column in WILD_TYPE_FEATURE_COLUMN.items()
        }
        measured_mean = {
            column: float(wild_type[column].mean())
            for column in WILD_TYPE_FEATURE_COLUMN.values()
        }
        measured_sd = {
            column: float(_population_sd(wild_type[column].tolist()))
            for column in WILD_TYPE_FEATURE_COLUMN.values()
        }
        ledger = ReleaseLedger(
            workbook_sha256=SI_TABLES_XLSX_SHA256,
            n_accession_rows=N_ACCESSION_ROWS,
            n_colonies_released=len(colonies),
            n_colonies_served=len(served_rows),
            n_wild_type_colonies=len(wild_type),
            n_barcode_rows=len(barcodes),
            n_gene_rows=N_GENE_ROWS,
            n_strains_served=int(per_strain.size),
            colonies_per_strain={str(k): int(v) for k, v in sorted(histogram.items())},
            n_colonies_without_barcode_row=without_row,
            n_colonies_gene_from_table4=gene_from_table4,
            n_colonies_label_disagrees_with_barcode=label_disagrees,
            n_genes_without_table4_row=n_genes_without_table4_row,
            uptag_notes={str(k): int(v) for k, v in notes.items()},
            n_rho0_colonies=int((colonies["mitochondria"] < 1.0).sum()),
            paper_rho0_colonies=PAPER_RHO0_COLONIES,
            n_aneuploid_colonies=aneuploid,
            aneuploid_fraction=aneuploid / len(colonies),
            wild_type_mean_published=published_mean,
            wild_type_mean_from_table3=measured_mean,
            wild_type_sd_published=published_sd,
            wild_type_sd_from_table3=measured_sd,
            n_genes_served=len(gene_names),
        )
        with open(osp.join(self.preprocess_dir, "release_ledger.json"), "w") as handle:
            handle.write(ledger.model_dump_json(indent=2))

    # ---- gene set --------------------------------------------------------------
    def compute_gene_set(self) -> GeneSet:
        """Systematic names of the deleted genes, cached from the build."""
        names: set[str] | None = getattr(self, "_gene_names", None)
        if names is not None:
            return GeneSet(names)
        return super().compute_gene_set()

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing happens inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Records are assembled inside process()."""
        raise NotImplementedError(
            "GenomeArchitecturePuddu2019Dataset builds its records in process()"
        )


def _population_sd(values: list[float]) -> float:
    """The n-denominator standard deviation, which is what Table 2 publishes."""
    if not values:
        raise ValueError("a standard deviation needs at least one value")
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def verify_build(dataset_root: str, data_root: str | None = None) -> Any:
    """L0-L4 report for a built store (``build_dataset_lmdb --verify`` calls this)."""
    from torchcell.verification import runners

    return runners.verify_genome_architecture_dataset(
        "genome_architecture_puddu2019",
        data_root or _data_root(),
        abs_root=dataset_root,
    )


def main() -> None:
    """Build/load the dataset and print one record."""
    data_root = _data_root()
    dataset = GenomeArchitecturePuddu2019Dataset(
        root=osp.join(data_root, "data/torchcell/genome_architecture_puddu2019")
    )
    print(f"len = {len(dataset)}")
    record = dataset[0]
    print(record["experiment"].genotype.perturbations[0])
    print(json.dumps(record["experiment"].phenotype.model_dump(mode="json"))[:400])


if __name__ == "__main__":
    main()
