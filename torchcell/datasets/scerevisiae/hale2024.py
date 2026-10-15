# torchcell/datasets/scerevisiae/hale2024
# [[torchcell.datasets.scerevisiae.hale2024]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/hale2024
# Test file: tests/torchcell/datasets/scerevisiae/test_hale2024.py
"""Hale 2024: CRISPRi guides on 169 barcoded BYx3S segregants, fitness per lineage.

Hale JJ et al. 2024, "Genome-scale analysis of interactions between genetic
perturbations and natural variation", Nat Commun 15:4234, doi:10.1038/s41467-024-48626-1
(citation key ``haleGenomescaleAnalysisInteractions2024``). The first dataset whose
genotype is a SEGREGANT BACKGROUND plus an ENGINEERED PERTURBATION:
``PerturbedSegregantGenotype`` holds the inferred haplotype mosaic and the designed
guide side by side. Note: ``[[torchcell.datasets.scerevisiae.hale2024]]``.

THE CROSS. BY4716 (BY) x 322134S (3S), NOT BY x RM: "We utilized 169 haploid progeny
(segregants) from a cross between the BY4716 (BY) lab and 322134S (3S) clinical strains".
The BY parent itself was also transformed with the library ("into each segregant and the
BY parent"); its released genotype column ``x_BY_BYx`` is BY at every marker, so it is
served with ``background_role="parent_1"`` and a one-parent mosaic that the schema
validator holds to that parent.

GENOTYPE. Background = the HMM hard calls of Supplemental Data 12 (14,395 markers over the
whole 970-strain panel, 0 = BY, 1 = 3S), not the 5,717-marker Supplemental Data 2 the
paper's analyses read: Supplemental Data 2's markers are a strict subset of Supplemental
Data 12's and agree at every called cell (1,039,423 of 1,039,423 for the 182 shared
columns, measured), so the finer matrix gives tighter block boundaries with no
disagreement. The call method is the one the paper defers to (Matsui 2022, ref. 35): an R
HMM over ~10x WGS allele fractions, chromosome ends (30 kb) omitted, adjacent
non-recombining SNPs collapsed. Each column is run-length encoded per chromosome over its
CALLED markers (an ``NA`` cell is skipped, so a run spans an uncalled marker exactly as a
block spans the positions between markers); ``n_markers`` counts called markers and the
re-expansion at the column's called markers reproduces it exactly (the build asserts it,
the verifier re-checks it). Chromosome 17 is the mitochondrion (``chrmt``). Every block
has ``posterior=1.0``: the release is hard calls. Parent 1 = BY pinned to the SGD S288C
R64-4-1 sequence ("The S288C reference differs from the BY strain by only roughly 125
SNPs"); parent 2 = 3S, which has NO deposited assembly: it is pinned to the authors' own
3S-vs-S288C variant table (Supplemental Data 13: 43,558 SNVs and 1,041 single-base indels,
every SNV and deletion ref base equal to R64-4-1, measured) in the genomes-tier set
``hale2024_322134S_variants``. That ref is a variant panel, not an assembly; it is the most
the release offers and the gap is an open owner decision in the dendron note.

Perturbation = one ``CrisprInterferencePerturbation`` per lineage: the target ORF as the
lineage table releases it (``gene_orf``, verbatim; no genome is consulted, the data is
the data), the released ``gene_name`` (or the ORF again for the 162,753 gene rows whose
``gene_name`` cell is ``NA``), and a ``CrisprConstruct`` with the ``dCas9-Mxi1``
effector and the 20-nt guide. Control gRNAs ("Control gRNAs show 'NA' under
the ORF column", 83 guides) perturb no gene, so they are never a record (the Hawkins 2020
and Cui 2018 precedent); they are the normalization reference and are counted in the
dropped-record ledger. Guides that bind near a BY/3S polymorphism are SERVED: the paper
excludes them from its own analyses, but the measured lineage is real and the background
mosaic says which allele the binding site carries.

RECORD = one double-barcode lineage in one assay (``replicate_id`` = the paper's double
barcode ``<geno_bc>_<guide_bc>``, ``screen_id`` = ``ATC1`` / ``ATC2`` / ``CON``). A lineage
is one independently integrated transformant of a (segregant, guide) design, so lineages
of one design are biological replicates; nothing is averaged that the source did not
average. Supplemental Data 6 = ATC1 (the primary induced assay), 7 = ATC2 (its replicate),
8 = CON (uninduced); the released analysis code names the first two the other way round
and says so ("Note that the names of ATC1 and ATC2 have been switched in the main text"),
and the labels here follow the SI description and the main text.

PHENOTYPE. ``EnvironmentResponsePhenotype`` with ``MeasurementType.relative_fitness``: the
released ``fitness`` column, which is PyFitSeq's Malthusian lineage fitness relative to
the population (time axis 0, 2, 4, 6, 10 generations, ``fitseq.py -t 0 2 4 6 10``) minus a
per-assay constant (the median of the per-guide mean ``raw_fitness`` of the 77 control
gRNAs with non-negative means, then the CON mean). The constant was recomputed from the
release to within 5.1e-15 for all three assays (measured), so the stored value is the
paper's own normalized statistic. Uncertainty = the released ``error``, stored as
``UncertaintyType.standard_error``: PyFitSeq's "Estimation_Error" is ``1/sqrt`` of the
second derivative of the negative log-likelihood at the fitted fitness (``fitseq.py`` at
the repository's last commit, deposited in the raw mirror), the observed-information
standard error. ``n_samples = 1`` lineage, ``sample_unit = biological_replicate``.
Reference: 0.0 = the mean fitness of all CON lineages ("such that the mean fitness of all
lineages in CON was exactly zero").

ENVIRONMENT. SC minus uracil (``media.SC_URA``; "synthetic complete medium containing
glucose ... and lacking uracil"), 21 C with shaking, 10 generations of serial batch
culture; ATC1 and ATC2 carry anhydrotetracycline at 250 ng/mL (stored as 0.25 ug/mL), CON
carries none. ``duration_hours`` is a typed gap: the time points are 24 h transfers, but
no exposure duration in hours is stated.

RAW MIRROR. ``$DATA_ROOT/torchcell-raw/haleGenomescaleAnalysisInteractions2024/``: the
seven Supplemental Data files the loader and verifier read (copied out of the literature
mirror's ``si/``, whose PMC-cloud retrieval is recorded; byte-identical to the Figshare
release 10.6084/m9.figshare.24578995.v1, measured), five members of the Figshare
``code.zip`` (``zip_member`` retriever, container sha256 pinned) and ``fitseq.py`` at
``darachm/fitseq`` commit 340194a.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import os.path as osp
import pickle
import shutil
import zipfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    check_manifest_pin,
    compute_sha256_hash,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.data.data import ExperimentReferenceIndex
from torchcell.data.experiment_dataset import canonical_json
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import SC_URA
from torchcell.datamodels.pydant import ModelStrict
from torchcell.datamodels.schema import (
    ArtifactRef,
    AssayType,
    Concentration,
    ConcentrationUnit,
    CrisprConstruct,
    CrisprInterferencePerturbation,
    Environment,
    EnvironmentPerturbationType,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    HaplotypeBlock,
    MeasurementType,
    PerturbedSegregantGenotype,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SegregantGenotype,
    SegregantParent,
    SegregantPerturbationExperiment,
    SegregantPerturbationExperimentReference,
    SmallMoleculePerturbation,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence import GeneSet
from torchcell.sequence.genome.registry import (
    HALE2024_3S_VARIANTS,
    ROLE_VARIANTS,
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
    GenomeManifest,
    assembly_set_dir,
    deposit_assembly_set,
    load_genome_manifest,
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

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "haleGenomescaleAnalysisInteractions2024"
PAPER_DOI = "10.1038/s41467-024-48626-1"
TITLE = (
    "Genome-scale analysis of interactions between genetic perturbations and natural "
    "variation"
)
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
LIBRARY_DIR_REL = f"torchcell-library/{CITATION_KEY}"
MATSUI_KEY = "matsuiInterplayAdditivityDominance2022"

PMC_BASE = "https://pmc-oa-opendata.s3.amazonaws.com/"
FIGSHARE_DOI = "10.6084/m9.figshare.24578995.v1"
FIGSHARE_ARTICLE_URL = "https://api.figshare.com/v2/articles/24578995"
FIGSHARE_FILE_URL = "https://ndownloader.figshare.com/files/{file_id}"

PAPER_MD_SHA256 = "3ff168708557588639b54b38fb27d11049d01f2c0f9b6e2c3677f2fa20afad1b"
SI_DESCRIPTION_SHA256 = (
    "f31850db672164f2953c9a9ffb2c28a6880d5add10593a964af3acc893dbf2f0"
)
MATSUI_PAPER_MD_SHA256 = (
    "09f8d82d1594920a2725c3f7dc2eee2a690562cddaf46325dfc78533d9de29f5"
)


class ReleasedFile(ModelStrict):
    """One released Supplemental Data file: its mirror copy and both public sources.

    ``library_path`` is the literature mirror's ``si/`` copy (PMC-cloud retrieval
    recorded there), ``pmc_object`` its key in the PMC article dataset, and
    ``figshare_file_id`` / ``figshare_md5`` the same bytes on Figshare (md5 as the
    Figshare API reports it). ``sha256`` is what every copy hashes to.
    """

    name: str
    supplemental_data: int
    library_path: str
    pmc_object: str
    figshare_file_id: int
    figshare_md5: str
    sha256: str


def _released(
    name: str, sd: int, si: str, moesm: int, fid: int, md5: str, sha: str
) -> ReleasedFile:
    ext = name.split(".", 1)[1].replace(".zip", "")
    return ReleasedFile(
        name=name,
        supplemental_data=sd,
        library_path=f"si/{si}",
        pmc_object=f"PMC11102447.1/41467_2024_48626_MOESM{moesm}_ESM.{ext}",
        figshare_file_id=fid,
        figshare_md5=md5,
        sha256=sha,
    )


SD1 = "supplemental_data1.txt"
SD2 = "supplemental_data2.txt"
SD3 = "supplemental_data3.csv"
SD6 = "supplemental_data6.csv.zip"
SD7 = "supplemental_data7.csv.zip"
SD8 = "supplemental_data8.csv.zip"
SD12 = "supplemental_data12.txt"
SD13 = "supplemental_data13.txt"

RELEASED_FILES: dict[str, ReleasedFile] = {
    f.name: f
    for f in (
        _released(SD1, 1, "si4.txt", 4, 43263429, "cc4b69fb338a69f7fff738d76b519ada", "348b9e0ce2edb864caa09d343e52d9506c9aba5c193832e5bd1db050fb97c2f3"),
        _released(SD2, 2, "si5.txt", 5, 43263435, "a290ed14f9f32c0183f07b6f33b3d757", "09c2ba7f4ebbf1b408ecdd3307f12f8293edc4d7644a0b5e67b51c2924fe6a41"),
        _released(SD3, 3, "si6.csv", 6, 43263441, "69cf1351388e1f53bf869bcc0c3d6f28", "291427b76cf7f88154dc197e59379f34d3d93e61db142a91a74ae5a1e044ee2a"),
        _released(SD6, 6, "si9.csv", 9, 43268520, "551d8ea599579b358dc74c495c7dd202", "8e60ce440713273e6845874fc7eaf4d386174f4f12cbc6e622d76c83e1172a67"),
        _released(SD7, 7, "si10.csv", 10, 43268634, "5b90e55ad7285422d07ac1e6f89cab3a", "cee3da9f0d72ef2d5ef38107f21d2cafb1af20b6db20bd848b8da87ac77bd6e6"),
        _released(SD8, 8, "si11.csv", 11, 43268748, "fc0713a80c6d030295bcb7af9481f8d0", "52318279a16056b46f88e4d86c0b7806743807d62ec62f9c5d1a2a31aced5592"),
        _released(SD12, 12, "si15.txt", 15, 43266903, "75e42853c41cd3364c9affbedfa492e9", "da390fe4ba67761ae789ce1d9c27bec8a798d56cdbae6b7d932da89b8e9f8c15"),
    )
}  # fmt: skip
#: The 3S-vs-S288C variant table goes to the genomes tier, not the raw mirror.
VARIANT_FILE = _released(
    SD13,
    13,
    "si16.txt",
    16,
    43268475,
    "55febad53e5f8f0f0e9a5f4d3a107fd2",
    "5a69877677639778d82acf70c6d6d2025636b1ac2a382d8baab5cf9a1dd103fe",
)
#: The loader reads these; the verifier also reads SD2 and SD3.
LOADER_FILES = [SD1, SD12, SD6, SD7, SD8]

CODE_ZIP_FILE_ID = 43268517
CODE_ZIP_MD5 = "65370d4b7616a56be7176483973ee861"
CODE_ZIP_SHA256 = "dc8dac03e1d4a49817842acf7aa864cbe5d0fb80950979135dc569d2fa53ab95"
#: Mirror-relative path -> member path inside the Figshare ``code.zip``.
CODE_MEMBERS: dict[str, str] = {
    "code/README.txt": "code/README.txt",
    "code/analysis_code.R": "code/analysis_code.R",
    "code/EIGHT_fitseq.job": "code/data_processing/EIGHT_fitseq.job",
    "code/SEVEN_fitseqInput_clusterV10.py": (
        "code/data_processing/lib/SEVEN_fitseqInput_clusterV10.py"
    ),
    "code/NINE_addFitSeqData_V2.py": "code/data_processing/NINE_addFitSeqData_V2.py",
}
FITSEQ_REPO = "darachm/fitseq"
FITSEQ_COMMIT = "340194a359f121bf4a535db0a577e767f3279445"
FITSEQ_REL = "code/fitseq.py"
FITSEQ_URL = (
    f"https://raw.githubusercontent.com/{FITSEQ_REPO}/{FITSEQ_COMMIT}/fitseq/fitseq.py"
)

# --------------------------------------------------------------------------- #
# The cross, the assays, the release shape
# --------------------------------------------------------------------------- #
CROSS = "BYx3S"
BY_NAME = "BY"
THREE_S_NAME = "3S"
BY_GENOTYPE_NAME = "x_BY_BYx"
#: Matsui 2022 (the paper Hale defers the segregants to) states both parents' engineered
#: genotypes verbatim; Hale states the same deletions and landing pad in prose.
BY_ENGINEERED = "BY fcy1Δ flo8Δ flo11Δ ura3Δ ho YBR209W::pGal1-Cre - Lox71 - Lox2272/71"
THREE_S_ENGINEERED = (
    "3S fcy1Δ flo8Δ flo11Δ ura3Δ ho YBR209W::pGal1-Cre - Lox71 - Lox2272/71"
)
S288C_ASSEMBLY_FSA = "S288C_reference_sequence_R64-4-1_20230830.fsa"

#: Supplemental Data 12 chromosome code -> the genome's chromosome name (17 = mito).
CHROMOSOMES: dict[int, str] = {
    1: "chrI", 2: "chrII", 3: "chrIII", 4: "chrIV", 5: "chrV", 6: "chrVI",
    7: "chrVII", 8: "chrVIII", 9: "chrIX", 10: "chrX", 11: "chrXI", 12: "chrXII",
    13: "chrXIII", 14: "chrXIV", 15: "chrXV", 16: "chrXVI", 17: "chrmt",
}  # fmt: skip


class AssaySpec(ModelStrict):
    """One released fitness assay: its lineage table and whether the guide was induced."""

    screen_id: Literal["ATC1", "ATC2", "CON"]
    released_file: str
    zip_member: str
    induced: bool
    #: rows of the released table, and rows whose ``gene_orf`` names a gene (measured)
    released_rows: int
    gene_rows: int


ASSAYS: list[AssaySpec] = [
    AssaySpec(screen_id="ATC1", released_file=SD6, zip_member="supplemental_data6.csv", induced=True, released_rows=2959500, gene_rows=2923436),
    AssaySpec(screen_id="ATC2", released_file=SD7, zip_member="supplemental_data7.csv", induced=True, released_rows=2892633, gene_rows=2857376),
    AssaySpec(screen_id="CON", released_file=SD8, zip_member="supplemental_data8.csv", induced=False, released_rows=2947298, gene_rows=2911823),
]  # fmt: skip
EXPECTED_RECORDS = sum(a.gene_rows for a in ASSAYS)  # 8,692,635
N_BACKGROUNDS = 170  # 169 segregants + the BY parent (measured in every assay)
#: Records are committed in chunks so one write transaction never holds the whole store.
COMMIT_EVERY = 250_000

LINEAGE_COLUMNS = [
    "geno_bc",
    "guide_bc",
    "fitness",
    "error",
    "genotype",
    "guide_seq",
    "gene_orf",
    "gene_name",
    "guide_category",
]


# --------------------------------------------------------------------------- #
# Sourced values (verbatim from the mirror bytes; audited by the verifier)
# --------------------------------------------------------------------------- #
def _paper(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri="paper.md",
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
            page="Results and Methods ('Fitness assay', 'Fitness estimation')",
        ),
    )


def _si(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri="si/si3.md",
            citation_key=CITATION_KEY,
            sha256=SI_DESCRIPTION_SHA256,
            method="MinerU OCR of 'Description of Additional Supplementary Files'",
            page="Supplemental Data 1-13 descriptions",
        ),
    )


def _matsui(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri="paper.md",
            citation_key=MATSUI_KEY,
            sha256=MATSUI_PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
            page="Methods ('Generation of haploid segregants', 'Whole-genome "
            "sequencing of parental haploid segregants')",
        ),
    )


TEMPERATURE_C = _paper(
    21.0,
    "assays were grown for $2 4 \\mathrm { h }$ at $2 1 ^ { \\circ } \\mathrm { C }$ "
    "with shaking.",
    note="21 C is also in the PDF text layer ('grown for 24 h at 21 °C with shaking')",
)
AEROBICITY = _paper(
    "aerobic",
    "assays were grown for $2 4 \\mathrm { h }$ at $2 1 ^ { \\circ } \\mathrm { C }$ "
    "with shaking.",
    note="shaken flasks; no anaerobic handling is described",
)
ATC_DOSE_UG_PER_ML = _paper(
    0.25,
    "ATC was added to experimental flasks (assays) at a final concentration of $2 5 0 "
    "\\mathrm { n g / m L }$ .",
    note="250 ng/mL stored as 0.25 ug/mL (ConcentrationUnit has no ng/mL member)",
)
NO_ATC_IN_CON = _paper(False, "ATC was not added to the control assay.")
MEDIUM = _paper(
    "SC-Ura",
    "All assays were done in synthetic complete medium containing glucose as the "
    "fermentable carbon source and lacking uracil, which maintained selection for the "
    "URA3 marker generated by integration of CRISPRi plasmids.",
    note="served as the shared media.SC_URA object; the glucose concentration is not "
    "stated by Hale, so SC's own recipe provenance is the honest state",
)
DURATION_GENERATIONS = _paper(
    10.0,
    "We performed the assays for 10 generations in serial batch culture, diluting 1:4 "
    "roughly every two generations, with a bottleneck size of ${ \\sim } 2 .",
)
EFFECTOR = _paper(
    "dCas9-Mxi1",
    "Each plasmid in this library contained an endonuclease-dead Cas9 (dCas9) fused "
    "with the Mxi1 repressor domain, an inducible single guide RNA (gRNA) targeting "
    "dCas9 to a specific promoter, a unique 20- nucleotide barcode, a loxP site to "
    "enable chromosomal integration, and the remaining portion of URA3.",
    note="the effector spelling Smith 2016 serves for the same Smith-lab system",
)
RELATIVE_TO_POPULATION = _paper(
    MeasurementType.relative_fitness,
    "PyFitSeq estimates the fitnesses of lineages relative to the mean fitness of the "
    "whole population.",
)
CON_MEAN_ZERO = _paper(
    0.0,
    "Finally, to aid in interpretation of the fitness values, we adjusted all fitness "
    "estimates in all assays by the same value, such that the mean fitness of all "
    "lineages in CON was exactly zero.",
)
CONTROL_GUIDES = _paper(
    83,
    "This normalization was performed using 83 control gRNAs that target intergenic and "
    "noncoding regions as the common reference point across the three assays.",
)
CONTROL_GUIDES_REMOVED = _paper(
    6,
    "Six of the 83 control gRNAs were removed at this step, as their average fitness was "
    "below zero and they had inconsistent effects between the experimental and control "
    "assays.",
)
CROSS_PARENTS = _paper(
    (BY_NAME, THREE_S_NAME),
    "We utilized 169 haploid progeny (segregants) from a cross between the BY4716 (BY) "
    "lab and 322134S (3S) clinical strains of Saccharomyces cerevisiae, each of which "
    "contained a barcode that marked a segregant35,36.",
)
BY_PARENT_TRANSFORMED = _paper(
    BY_GENOTYPE_NAME,
    "We integrated this library of barcoded CRISPRi plasmids into each segregant and the "
    "BY parent using large-scale transformations and Cre/loxP-mediated site-specific "
    "recombination.",
)
BY_IS_S288C = _si(
    SGD_S288C_R64,
    "The S288C reference differs from the BY strain by only roughly 125 SNPs.",
)
ALLELE_CODING = _si(
    {0: BY_NAME, 1: THREE_S_NAME},
    "with 0 indicating a BY allele and 1 indicating a 3S allele at that locus.",
)
LINEAGE_ROW = _si(
    "double barcode lineage", "Each row represents a unique double barcode lineage."
)
ERROR_COLUMN = _si(
    UncertaintyType.standard_error,
    "error: Reported error for the fitness estimate from PyFitSeq software.",
    note="the kind of error is settled by fitseq.py (FITSEQ_ERROR_QUOTE)",
)
CONTROL_ORF_NA = _si("NA", "Control gRNAs show 'NA' under the ORF column.")
HMM_CALLS = _matsui(
    "HMM",
    "This vector of initial genotype calls was then corrected with a Hidden Markov "
    "Model (HMM), implemented using the HMM package version $1 .",
)
CHROMOSOME_ENDS = _matsui(
    30_000,
    "All SNP markers within the first and last $3 0 \\mathrm { k b }$ of each "
    "chromosome were omitted because we observed higher sequencing error rate and/or "
    "lower read mapping quality for these specific regions.",
)
COLLAPSED_MARKERS = _matsui(
    "rmdup",
    "Adjacent SNPs in the HMM-corrected genotype calls that lack recombination in the "
    "segregants were collapsed into a single SNP, reducing the number of SNP markers in "
    "subsequent analysis from 43,865 to 7742.",
    note="the released matrices are named genotype_array_HMM_rmdup_* in "
    "analysis_code.R",
)
PARENT_GENOTYPES = _matsui(
    (BY_ENGINEERED, THREE_S_ENGINEERED),
    "Opposite mating types of BY fcy1Δ flo8Δ flo11Δ ura3Δ ho "
    "YBR209W::pGal1-Cre - Lox71 - Lox2272/71 and 3S fcy1Δ flo8Δ flo11Δ ura3Δ ho "
    "YBR209W::pGal1-Cre - Lox71 - Lox2272/71 were next mated, creating two BY/3S "
    "heterozygous diploids.",
)


def _code(
    value: Any, path: str, quote: str, *, note: str | None = None
) -> SourcedValue:
    """A value anchored to a deposited code file (root: ``$DATA_ROOT/torchcell-raw``)."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=path,
            citation_key=CITATION_KEY,
            sha256=CODE_SHA256[path],
            method="deposited release code (raw mirror code/)",
            page=path,
        ),
    )


#: sha256 of each deposited code file (measured at deposit; the manifest pins the same).
CODE_SHA256: dict[str, str] = {
    "code/README.txt": "8699bd78cac639024f2b8e01daf50edf62e14549950d3ce84e85323fba6a6959",
    "code/analysis_code.R": "bb75e634034a32741c901b62cbb78c306148c6676c6d6236d9565be163c525a0",
    "code/EIGHT_fitseq.job": "d1772dfa3bf64f6bb8b454a2124b62b9e23d604ce4f2a3627cbc3d4d4f614a72",
    "code/SEVEN_fitseqInput_clusterV10.py": "58a3b771d3bdbefd25410cfa7e2fb2f5bdc3f0331114c9775cb830c70d058682",
    "code/NINE_addFitSeqData_V2.py": "bbccc35858891cb25eeaa21553ca5f69559f1a0d5317e38c1c4c93180e21ba61",
    FITSEQ_REL: "576f84d5b2a9b67315ad0b174c4334c116bd593ce9c6719766c0524ddeda1454",
}

FITSEQ_TIME_AXIS = _code(
    [0.0, 2.0, 4.0, 6.0, 10.0],
    "code/EIGHT_fitseq.job",
    "-t 0 2 4 6 10",
    note="the five fitted time points T0, T1, T2, T3, T5 in generations; the ATC tables "
    "also release a sixth (T7) normalized count that the fit did not use "
    "(SEVEN_fitseqInput: '[:5]')",
)
FITSEQ_ERROR = _code(
    UncertaintyType.standard_error,
    FITSEQ_REL,
    "[ 1/np.sqrt(i) ",
    note="Estimation_Error = 1/sqrt(second derivative) of calculate_likelihood_of_"
    "fitness_vector, which 'return -likelihood_log_lineage': the observed-information "
    "standard error of the maximum-likelihood fitness",
)
FITSEQ_NEGATIVE_LOG_LIKELIHOOD = _code(
    "negative log-likelihood", FITSEQ_REL, "return -likelihood_log_lineage"
)
NORMALIZATION_MEDIAN = _code(
    "control-gRNA median",
    "code/analysis_code.R",
    "atc2$fitness = atc2$raw_fitness - atc2.med",
)
NORMALIZATION_CON_MEAN = _code(
    "CON mean",
    "code/analysis_code.R",
    "con3$fitness = con3$fitness - mean(con3$fitness)",
)
ASSAY_NAMES_SWAPPED = _code(
    "SD6 = ATC1",
    "code/analysis_code.R",
    "Note that the names of ATC1 and ATC2 have been switched in the main text for "
    "clarity. Supplemental Data 6 is labelled ATC1, but this same file is referred to as "
    "ATC2 in this script.",
)

SOURCED_VALUES: list[SourcedValue] = [
    TEMPERATURE_C, AEROBICITY, ATC_DOSE_UG_PER_ML, NO_ATC_IN_CON, MEDIUM,
    DURATION_GENERATIONS, EFFECTOR, RELATIVE_TO_POPULATION, CON_MEAN_ZERO,
    CONTROL_GUIDES, CONTROL_GUIDES_REMOVED, CROSS_PARENTS, BY_PARENT_TRANSFORMED,
    BY_IS_S288C, ALLELE_CODING, LINEAGE_ROW, ERROR_COLUMN, CONTROL_ORF_NA, HMM_CALLS, CHROMOSOME_ENDS,
    COLLAPSED_MARKERS, PARENT_GENOTYPES,
]  # fmt: skip
#: Anchored to the raw mirror's ``code/`` (library root ``$DATA_ROOT/torchcell-raw``).
CODE_SOURCED_VALUES: list[SourcedValue] = [
    FITSEQ_TIME_AXIS, FITSEQ_ERROR, FITSEQ_NEGATIVE_LOG_LIKELIHOOD,
    NORMALIZATION_MEDIAN, NORMALIZATION_CON_MEAN, ASSAY_NAMES_SWAPPED,
]  # fmt: skip

DURATION_HOURS_GAP = ProvenanceGap(
    field="duration_hours",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="each time point is a 24 h transfer and the fit spans T0 to T5, but no "
    "exposure duration in hours is stated; the duration is carried in generations",
)
CALL_METHOD = (
    "HMM-corrected hard calls (R HMM package; Matsui 2022 Methods) from ~10x "
    "whole-genome sequencing, chromosome ends (30 kb) omitted, adjacent non-recombining "
    "SNPs collapsed (genotype_array_HMM_rmdup_version3); released as Hale 2024 "
    "Supplemental Data 12 (0 = BY, 1 = 3S); NA cells uncalled"
)
UNITS = (
    "PyFitSeq Malthusian fitness of one double-barcode lineage relative to its pooled "
    "population (generations 0, 2, 4, 6, 10), minus the assay's control-gRNA median and "
    "the CON lineage mean (released 'fitness'); uncertainty = PyFitSeq estimation error"
)
UNITS_REFERENCE = (
    "0 = the mean relative fitness of all CON lineages; every assay is first aligned on "
    "the median of its 77 retained control gRNAs"
)


# --------------------------------------------------------------------------- #
# Haplotype blocks from the released hard calls
# --------------------------------------------------------------------------- #
class CalledMarkers(BaseModel):
    """One column's CALLED markers in genome order: chromosome, position, 0/1 call."""

    chromosomes: list[str]
    positions: list[int]
    calls: list[int]


def read_genotype_matrix(path: str | Path) -> pd.DataFrame:
    """Supplemental Data 2 / 12: ``c``, ``p``, ``gp`` then one 0/1/NA column per strain.

    Positions are strictly increasing within each chromosome (measured on both
    releases); the loader still sorts by (chromosome, position) and asserts the sort is
    the identity, so a reordered re-release fails loudly instead of yielding phantom
    blocks.
    """
    frame = pd.read_csv(path, sep="\t")
    if list(frame.columns[:3]) != ["c", "p", "gp"]:
        raise ValueError(f"{path}: first columns are {list(frame.columns[:3])}")
    unknown = set(frame["c"].unique()) - set(CHROMOSOMES)
    if unknown:
        raise ValueError(f"{path}: chromosome codes {sorted(unknown)} not 1..17")
    order = frame.sort_values(["c", "p"], kind="stable").index
    if not (order == frame.index).all():
        raise ValueError(f"{path}: markers are not in (chromosome, position) order")
    if frame.duplicated(["c", "p"]).any():
        raise ValueError(f"{path}: duplicate (chromosome, position) markers")
    return frame


def called_markers(frame: pd.DataFrame, column: str) -> CalledMarkers:
    """The non-NA markers of one strain's column; every call must be 0 or 1."""
    values = frame[column].to_numpy(dtype=float)
    keep = ~np.isnan(values)
    calls = values[keep]
    if not np.isin(calls, (0.0, 1.0)).all():
        raise ValueError(f"{column}: calls outside {{0, 1, NA}}")
    return CalledMarkers(
        chromosomes=[CHROMOSOMES[int(c)] for c in frame["c"].to_numpy()[keep]],
        positions=[int(p) for p in frame["p"].to_numpy()[keep]],
        calls=[int(v) for v in calls],
    )


def encode_blocks(markers: CalledMarkers) -> list[HaplotypeBlock]:
    """Run-length encode called markers per chromosome (0 -> parent 1, 1 -> parent 2)."""
    blocks: list[HaplotypeBlock] = []
    n = len(markers.calls)
    start = 0
    for i in range(1, n + 1):
        if (
            i == n
            or markers.calls[i] != markers.calls[start]
            or markers.chromosomes[i] != markers.chromosomes[start]
        ):
            blocks.append(
                HaplotypeBlock(
                    chromosome=markers.chromosomes[start],
                    start=markers.positions[start],
                    end=markers.positions[i - 1],
                    parent=1 if markers.calls[start] == 0 else 2,
                    posterior=1.0,
                    n_markers=i - start,
                )
            )
            start = i
    return blocks


def expand_blocks(blocks: list[HaplotypeBlock], markers: CalledMarkers) -> list[int]:
    """Inverse of ``encode_blocks`` at the same called markers (0/1 per marker)."""
    out: list[int] = []
    i = 0
    for block in blocks:
        end = i + block.n_markers
        if (
            end > len(markers.calls)
            or markers.chromosomes[i] != block.chromosome
            or markers.chromosomes[end - 1] != block.chromosome
            or markers.positions[i] != block.start
            or markers.positions[end - 1] != block.end
        ):
            raise ValueError(f"block {block} does not align with the markers at {i}")
        out.extend([block.parent - 1] * block.n_markers)
        i = end
    if i != len(markers.calls):
        raise ValueError(f"blocks cover {i} of {len(markers.calls)} called markers")
    return out


def read_segregant_table(path: str | Path) -> pd.DataFrame:
    """Supplemental Data 1: ``seg_name`` (one segregant barcode) -> ``genotype`` name."""
    frame = pd.read_csv(path)
    expected = ["seg_name", "geno_bc", "rev_geno_bc", "genotype", "genotype_data"]
    if list(frame.columns) != expected:
        raise ValueError(f"{path}: columns {list(frame.columns)} != {expected}")
    if frame["seg_name"].duplicated().any():
        raise ValueError(f"{path}: duplicate seg_name rows")
    return frame


def read_lineage_table(zip_path: str | Path, member: str) -> pd.DataFrame:
    """One assay's released lineage table, read out of its zip container.

    ``round_trip`` keeps every stored float the exact decimal the release prints, so
    the verifier's re-read equals it bit for bit. ``gene_orf`` and ``gene_name`` stay
    strings so the release's literal ``NA`` is seen as missing, not as a gene.
    """
    with zipfile.ZipFile(zip_path) as archive, archive.open(member) as handle:
        return pd.read_csv(
            handle,
            usecols=LINEAGE_COLUMNS,
            dtype={
                "gene_orf": str,
                "gene_name": str,
                "genotype": str,
                "guide_seq": str,
            },
            float_precision="round_trip",
        )


def background_role(genotype: str) -> Literal["segregant", "parent_1"]:
    """The BY parent's column is the parental control; every other column a segregant."""
    return "parent_1" if genotype == BY_GENOTYPE_NAME else "segregant"


# --------------------------------------------------------------------------- #
# Raw-mirror and genomes-tier deposits
# --------------------------------------------------------------------------- #
def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_root() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/haleGenomescaleAnalysisInteractions2024``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def raw_relpath(name: str) -> str:
    """Mirror-relative path of one released data file."""
    return f"data/{name}"


def _library_record(library_manifest: Manifest, relpath: str) -> ArtifactRecord:
    for record in library_manifest.files:
        if record.path == relpath:
            return record
    raise KeyError(f"{relpath} is not in the literature mirror manifest")


def _copy_verified(src: Path, dest: Path, sha256: str) -> None:
    """Copy ``src`` to ``dest`` unless an identical file is there; never overwrite."""
    if _sha256(src) != sha256:
        raise RuntimeError(f"{src} does not hash to the pinned {sha256}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if _sha256(dest) != sha256:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        return
    shutil.copy2(src, dest)


def _pmc_retrieval(
    released: ReleasedFile, library_record: ArtifactRecord
) -> RetrievalRecord:
    """The literature mirror's own PMC-cloud retrieval of the same bytes."""
    if library_record.retrieval is None:
        raise RuntimeError(
            f"{released.library_path}: no retrieval record in the mirror"
        )
    if library_record.retrieval.params.get("key") != released.pmc_object:
        raise RuntimeError(
            f"{released.library_path}: mirror retrieved "
            f"{library_record.retrieval.params.get('key')!r}, expected "
            f"{released.pmc_object!r}"
        )
    return library_record.retrieval


def deposit_raw_mirror(
    *,
    code_zip_path: str | Path,
    fitseq_path: str | Path,
    retrieved_at_code: str,
    retrieved_at_fitseq: str,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror (data out of the literature mirror, code, ``manifest.json``).

    Idempotent by sha256: an existing file with the pinned hash is left alone and a
    differing one raises. The seven data files are copied out of the literature
    mirror's ``si/`` after its manifest's pin is checked; their retrieval record IS the
    mirror's PMC-cloud record (same bytes as Figshare, listed as an alternative source).
    ``code_zip_path`` is the Figshare ``code.zip`` (container sha256 pinned) and
    ``fitseq_path`` the GitHub ``fitseq.py`` at ``FITSEQ_COMMIT``.
    """
    root_dir = data_root or _data_root()
    root = raw_mirror_dir(root_dir)
    library = Path(root_dir) / LIBRARY_DIR_REL
    library_manifest = Manifest.model_validate_json(
        (library / "manifest.json").read_text()
    )
    files: list[ArtifactRecord] = []
    for released in RELEASED_FILES.values():
        record = _library_record(library_manifest, released.library_path)
        if record.sha256 != released.sha256:
            raise RuntimeError(
                f"{released.library_path}: mirror pins {record.sha256}, loader pins "
                f"{released.sha256}"
            )
        relpath = raw_relpath(released.name)
        _copy_verified(library / released.library_path, root / relpath, released.sha256)
        retrieval = _pmc_retrieval(released, record)
        files.append(
            ArtifactRecord(
                path=relpath,
                role=ROLE_RAW_DATA,
                bytes=(root / relpath).stat().st_size,
                sha256=released.sha256,
                source=retrieval.source_url,
                retrieval=retrieval,
            )
        )

    if _sha256(code_zip_path) != CODE_ZIP_SHA256:
        raise RuntimeError(f"{code_zip_path} is not the pinned Figshare code.zip")
    code_url = FIGSHARE_FILE_URL.format(file_id=CODE_ZIP_FILE_ID)
    with zipfile.ZipFile(code_zip_path) as archive:
        for relpath, member in CODE_MEMBERS.items():
            data = archive.read(member)
            sha = hashlib.sha256(data).hexdigest()
            if sha != CODE_SHA256[relpath]:
                raise RuntimeError(
                    f"{member}: sha256 {sha} != pinned {CODE_SHA256[relpath]}"
                )
            dest = root / relpath
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if _sha256(dest) != sha:
                    raise RuntimeError(
                        f"{dest} exists with a different sha256; refusing"
                    )
            else:
                dest.write_bytes(data)
            files.append(
                ArtifactRecord(
                    path=relpath,
                    role=ROLE_RAW_DATA,
                    bytes=len(data),
                    sha256=sha,
                    source=code_url,
                    retrieval=RetrievalRecord(
                        method=RetrievalMethod.direct_url,
                        source_url=code_url,
                        retriever="torchcell.literature.retrieve.zip_member",
                        params={
                            "url": code_url,
                            "member": member,
                            "container_sha256": CODE_ZIP_SHA256,
                        },
                        sha256=sha,
                        retrieved_at=retrieved_at_code,
                    ),
                )
            )
    _copy_verified(Path(fitseq_path), root / FITSEQ_REL, CODE_SHA256[FITSEQ_REL])
    files.append(
        ArtifactRecord(
            path=FITSEQ_REL,
            role=ROLE_RAW_DATA,
            bytes=(root / FITSEQ_REL).stat().st_size,
            sha256=CODE_SHA256[FITSEQ_REL],
            source=FITSEQ_URL,
            retrieval=RetrievalRecord(
                method=RetrievalMethod.direct_url,
                source_url=FITSEQ_URL,
                retriever="torchcell.literature.retrieve.direct_url",
                params={
                    "url": FITSEQ_URL,
                    "repo": FITSEQ_REPO,
                    "commit": FITSEQ_COMMIT,
                    "path": "fitseq/fitseq.py",
                },
                sha256=CODE_SHA256[FITSEQ_REL],
                retrieved_at=retrieved_at_fitseq,
            ),
        )
    )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=TITLE,
        files=files,
        si_data_sources=[
            PMC_BASE + "PMC11102447.1/",
            f"https://doi.org/{FIGSHARE_DOI}",
            FIGSHARE_ARTICLE_URL,
            f"https://github.com/{FITSEQ_REPO}/tree/{FITSEQ_COMMIT}",
            "https://www.ncbi.nlm.nih.gov/bioproject/PRJNA986287",
        ],
        si_expected=[
            f"Supplemental Data {f.supplemental_data} ({f.name})"
            for f in RELEASED_FILES.values()
        ],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return root


def deposit_variant_set(data_root: str | None = None) -> str:
    """Deposit the 3S variant table as the genomes-tier set ``HALE2024_3S_VARIANTS``.

    The file is copied out of the literature mirror (its PMC-cloud retrieval becomes
    the tier record's retrieval) and ``deposit_assembly_set`` writes the manifest after
    re-hashing the bytes on disk. A set that already has a manifest is left alone.
    """
    root_dir = data_root or _data_root()
    directory = Path(assembly_set_dir(HALE2024_3S_VARIANTS, root_dir))
    if (directory / "manifest.json").exists():
        return str(directory / "manifest.json")
    library = Path(root_dir) / LIBRARY_DIR_REL
    library_manifest = Manifest.model_validate_json(
        (library / "manifest.json").read_text()
    )
    record = _library_record(library_manifest, VARIANT_FILE.library_path)
    _copy_verified(
        library / VARIANT_FILE.library_path, directory / SD13, VARIANT_FILE.sha256
    )
    retrieval = _pmc_retrieval(VARIANT_FILE, record)
    manifest = GenomeManifest(
        assembly_set=HALE2024_3S_VARIANTS,
        organism="Saccharomyces cerevisiae",
        strain_or_population="322134S (3S), the clinical parent of the BYx3S cross",
        source="Hale et al. 2024 Supplemental Data 13",
        release="2024",
        citation_key=CITATION_KEY,
        source_url=retrieval.source_url,
        files=[
            ArtifactRecord(
                path=SD13,
                role=ROLE_VARIANTS,
                bytes=(directory / SD13).stat().st_size,
                sha256=VARIANT_FILE.sha256,
                source=retrieval.source_url,
                retrieval=retrieval,
            )
        ],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
        notes=(
            "A VARIANT TABLE, NOT AN ASSEMBLY: 'Table of all SNPs present in the 3S "
            "strain, relative to S288C' (four tab-separated columns, no header: "
            "'<n>_S288C' chromosome with 17 = mitochondrion, position, S288C base, 3S "
            "base). Measured 2026-10-11: 44,599 rows = 43,558 SNVs + 270 single-base "
            "deletions (3S base '-') + 771 single-base insertions (S288C base '-'); every "
            "SNV and deletion S288C base equals SGD R64-4-1 (sgd_S288C_R64-4-1_20230830) "
            "at its position. No public 3S assembly exists (NCBI assembly search for "
            "322134S: 0 hits; absent from Peter 2018 Table S1 and SGRP2), so this table "
            "is the 3S parent's sequence reference for the Hale 2024 segregant mosaics; "
            "larger indels, structural variants and 3S-specific sequence are not "
            "represented. Byte-identical to Figshare file 43268475 of "
            f"{FIGSHARE_DOI} (md5 {VARIANT_FILE.figshare_md5})."
        ),
    )
    return deposit_assembly_set(manifest, root_dir)


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


def tier_file_ref(
    assembly_set: str,
    filename: str,
    data_root: str | None = None,
    media_type: str | None = None,
) -> ArtifactRef:
    """File-level ``ArtifactRef`` of one genomes-tier file, pinned by its manifest."""
    record = load_genome_manifest(assembly_set, data_root).record(filename)
    return ArtifactRef(
        tier="genomes",
        key=assembly_set,
        path=record.path,
        sha256=record.sha256,
        bytes=record.bytes,
        media_type=media_type,
    )


def build_parents(
    data_root: str | None = None,
) -> tuple[SegregantParent, SegregantParent]:
    """BY pinned to the S288C R64-4-1 sequence, 3S to its variant table."""
    by = SegregantParent(
        name=BY_NAME,
        peter_strain_id=None,
        assembly_ref=tier_file_ref(SGD_S288C_R64, S288C_ASSEMBLY_FSA, data_root),
        engineered_background=BY_ENGINEERED,
    )
    three_s = SegregantParent(
        name=THREE_S_NAME,
        peter_strain_id=None,
        assembly_ref=tier_file_ref(
            HALE2024_3S_VARIANTS, SD13, data_root, "text/tab-separated-values"
        ),
        engineered_background=THREE_S_ENGINEERED,
    )
    return by, three_s


def run_verification(data_root: str | None = None) -> Any:
    """L0-L4 for the built store; writes ``preprocess/verification_report.json``."""
    from torchcell.verification.runners import verify_segregant_perturbation_dataset

    return verify_segregant_perturbation_dataset(
        "crispri_segregant_hale2024", data_root or _data_root()
    )


class DropRule(BaseModel):
    """One rule that kept released rows out of the store, with its count and items."""

    rule: str
    scope: Literal["guide", "row"]
    description: str
    n_records: int
    per_assay: dict[str, int]
    items: list[str]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@register_dataset
class CrispriSegregantHale2024Dataset(ExperimentDataset):
    """Hale 2024: one record per (double-barcode lineage, assay), 8,692,635 records."""

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/crispri_segregant_hale2024",
        io_workers: int = 0,
        genome: SCerevisiaeGenome | None = None,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """``genome`` is kept for ``verify_build``'s L4 containment row; the build itself
        reads none (every target is stored as the lineage table releases it).
        """
        self.genome = genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return SegregantPerturbationExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return SegregantPerturbationExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The seven released data files (the verifier also reads SD2 and SD3)."""
        return list(RELEASED_FILES)

    def download(self) -> None:
        """Symlink the manifest-listed mirror files into ``raw/`` and verify each sha256."""
        data_root = _data_root()
        manifest = load_manifest(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        for name, released in RELEASED_FILES.items():
            relpath = raw_relpath(name)
            check_manifest_pin(
                relpath, manifest_sha256(manifest, relpath), released.sha256
            )
            src = raw_mirror_dir(data_root) / relpath
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            link_verified(src, osp.join(self.raw_dir, name), released.sha256)
        log.info("Hale 2024 raw files linked into %s (sha256 verified)", self.raw_dir)

    # ---- environment / phenotype / reference ---------------------------------
    @staticmethod
    def environment_for(induced: bool) -> Environment:
        """SC-Ura, 21 C, 10 generations; ATC at 250 ng/mL when the guide is induced."""
        perturbations: list[EnvironmentPerturbationType] = (
            [
                SmallMoleculePerturbation(
                    compound=resolved_compound("anhydrotetracycline"),
                    concentration=Concentration(
                        value=ATC_DOSE_UG_PER_ML.value, unit=ConcentrationUnit.ug_per_ml
                    ),
                )
            ]
            if induced
            else []
        )
        return Environment(
            media=SC_URA,
            temperature=Temperature(value=TEMPERATURE_C.value),
            perturbations=perturbations,
            aerobicity=AEROBICITY.value,
            duration_generations=DURATION_GENERATIONS.value,
            provenance_gaps=[DURATION_HOURS_GAP],
        )

    @staticmethod
    def phenotype_for(
        screen_id: str, fitness: float, error: float, replicate_id: str
    ) -> EnvironmentResponsePhenotype:
        """One lineage's released fitness and PyFitSeq standard error in one assay."""
        return EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.relative_fitness,
            assay_type=AssayType.pooled_competitive_growth_barcode,
            environment_response=fitness,
            environment_response_uncertainty=error,
            environment_response_uncertainty_type=UncertaintyType.standard_error,
            n_samples=1,
            sample_unit=SampleUnit.biological_replicate,
            units=UNITS,
            screen_id=screen_id,
            replicate_id=replicate_id,
        )

    def reference_for(
        self, spec: AssaySpec, environment: Environment
    ) -> SegregantPerturbationExperimentReference:
        """0 = the CON lineage mean; the reference environment is the assay's own."""
        return SegregantPerturbationExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae",
                strain="S288C",
                ploidy="haploid",
                assembly_set=SGD_S288C_R64,
                assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
            ),
            environment_reference=environment,
            phenotype_reference=EnvironmentResponsePhenotype(
                measurement_type=MeasurementType.relative_fitness,
                assay_type=AssayType.pooled_competitive_growth_barcode,
                environment_response=CON_MEAN_ZERO.value,
                units=UNITS_REFERENCE,
                screen_id=spec.screen_id,
            ),
        )

    # ---- genotypes ------------------------------------------------------------
    def build_backgrounds(
        self,
        names: list[str],
        matrix: pd.DataFrame,
        parents: tuple[SegregantParent, SegregantParent],
        matrix_sha256: str,
    ) -> dict[str, SegregantGenotype]:
        """One mosaic per named column, each asserted to round-trip to its calls."""
        out: dict[str, SegregantGenotype] = {}
        for name in names:
            if name not in matrix.columns:
                raise ValueError(f"{name}: no column in Supplemental Data 12")
            markers = called_markers(matrix, name)
            blocks = encode_blocks(markers)
            if expand_blocks(blocks, markers) != markers.calls:
                raise ValueError(f"{name}: block round trip failed")
            out[name] = SegregantGenotype(
                cross=CROSS,
                segregant_id=name,
                parent_1=parents[0],
                parent_2=parents[1],
                blocks=blocks,
                call_method=CALL_METHOD,
                marker_matrix_sha256=matrix_sha256,
            )
        return out

    @staticmethod
    def perturbation_for(
        orf: str, gene_name: str, guide_seq: str
    ) -> CrisprInterferencePerturbation:
        """The designed knockdown: released ORF and gene name + dCas9-Mxi1 with this guide."""
        return CrisprInterferencePerturbation(
            systematic_gene_name=orf,
            perturbed_gene_name=gene_name,
            crispr=CrisprConstruct(
                effector=EFFECTOR.value, guide_sequence=guide_seq, n_guides=1
            ),
        )

    # ---- build ----------------------------------------------------------------
    def _pointer(self, obj: BaseModel, hint: str, itxn: Any) -> Any:
        """``obj``'s dump, replaced by an interned ``$ref`` pointer when large enough."""
        holder: dict[str, Any] = {"x": obj.model_dump()}
        self._maybe_intern(holder, "x", obj, hint, itxn)
        return holder["x"]

    @staticmethod
    def _interned_pointer(obj: BaseModel, hint: str, itxn: Any) -> dict[str, str]:
        """Intern ``obj`` regardless of size and return its ``$ref`` pointer.

        The same put-if-absent row ``_maybe_intern`` writes, without its 512-byte floor:
        a perturbation (485 canonical bytes) and the publication (226) are constant across
        8.7M records, and inline they keep each record above LMDB's half-page node limit.
        """
        digest = compute_sha256_hash(canonical_json(obj))
        if itxn.get(digest.encode()) is None:
            itxn.put(digest.encode(), pickle.dumps(obj.model_dump()))
        return {"$ref": digest, "name": hint}

    def iter_assay_rows(self, spec: AssaySpec) -> Iterator[tuple[Any, ...]]:
        """(geno_bc, guide_bc, fitness, error, genotype, guide_seq, gene_orf, gene_name)
        per row; the two gene cells are the release's strings, ``NA`` read as missing.
        """
        frame = read_lineage_table(
            osp.join(self.raw_dir, spec.released_file), spec.zip_member
        )
        if len(frame) != spec.released_rows:
            raise ValueError(
                f"{spec.screen_id}: {len(frame)} released rows, expected {spec.released_rows}"
            )
        if set(frame["guide_category"].unique()) != {"E"}:
            raise ValueError(f"{spec.screen_id}: non-essential-library rows present")
        if frame[["fitness", "error"]].isna().any().any():
            raise ValueError(f"{spec.screen_id}: NaN fitness or error in the release")
        yield from zip(
            frame["geno_bc"].astype(str),
            frame["guide_bc"].astype(str),
            frame["fitness"].astype(float),
            frame["error"].astype(float),
            frame["genotype"].astype(str),
            frame["guide_seq"].astype(str),
            frame["gene_orf"],
            frame["gene_name"],
            strict=True,
        )

    @post_process
    def process(self) -> None:
        """Build one record per (lineage, assay) with gene-targeting guides."""
        # The module constants are the pins (download() checked the mirror manifest
        # records the same digests), so a file edited in raw/ after linking is refused
        # before any byte of it is read.
        verify_raw_files(
            self.raw_dir,
            {name: RELEASED_FILES[name].sha256 for name in self.raw_file_names},
        )
        data_root = _data_root()
        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        segregant_table = read_segregant_table(osp.join(self.raw_dir, SD1))
        matrix = read_genotype_matrix(osp.join(self.raw_dir, SD12))
        backgrounds = self.build_backgrounds(
            sorted(set(segregant_table["genotype"])),
            matrix,
            build_parents(data_root),
            RELEASED_FILES[SD12].sha256,
        )
        seg_to_genotype = dict(
            zip(segregant_table["geno_bc"], segregant_table["genotype"], strict=True)
        )
        environments = {
            True: self.environment_for(True),
            False: self.environment_for(False),
        }
        references: dict[str, SegregantPerturbationExperimentReference] = {
            a.screen_id: self.reference_for(a, environments[a.induced]) for a in ASSAYS
        }
        publication = Publication(doi=PAPER_DOI, doi_url=f"https://doi.org/{PAPER_DOI}")

        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        with interned_env.begin(write=True) as itxn:
            env_ptr = {
                k: self._pointer(v, SC_URA.stated_name, itxn)
                for k, v in environments.items()
            }
            ref_ptr = {
                k: self._pointer(v, self.name, itxn) for k, v in references.items()
            }
            pub_ptr = self._interned_pointer(publication, PAPER_DOI, itxn)
            bg_ptr = {k: self._pointer(v, k, itxn) for k, v in backgrounds.items()}
        role_of = {name: background_role(name) for name in backgrounds}
        perturbations: dict[tuple[str, str], CrisprInterferencePerturbation] = {}
        # A perturbation is interned once per (ORF, guide): inline it pushed every record
        # past LMDB's half-page node limit onto its own 4 KB overflow page (measured: 2,299
        # bytes per record, 8,692,635 overflow pages, 34 GB), and 8,046 guides are a
        # constant set exactly as the backgrounds are.
        pert_ptr: dict[tuple[str, str], Any] = {}

        idx = 0
        member_ranges: dict[str, tuple[int, int]] = {}
        dropped_control: dict[str, int] = {}
        control_guides: set[str] = set()
        target_genes: set[str] = set()
        seen_lineages: dict[str, set[str]] = {}
        checked: set[tuple[str, str]] = set()
        for spec in ASSAYS:
            start = idx
            n_control = 0
            lineages: set[str] = set()
            induced = spec.induced
            rows = tqdm(
                self.iter_assay_rows(spec),
                total=spec.released_rows,
                desc=spec.screen_id,
            )
            while chunk := list(itertools.islice(rows, COMMIT_EVERY)):
                # Interned rows commit before the records that point at them (LIFO).
                with (
                    env.begin(write=True) as txn,
                    interned_env.begin(write=True) as itxn,
                ):
                    for (
                        geno_bc,
                        guide_bc,
                        fitness,
                        error,
                        genotype,
                        guide_seq,
                        orf,
                        gene_name,
                    ) in chunk:
                        if seg_to_genotype.get(geno_bc) != genotype:
                            raise ValueError(
                                f"{spec.screen_id}: segregant barcode {geno_bc} is "
                                f"{seg_to_genotype.get(geno_bc)!r} in SD1, {genotype!r} here"
                            )
                        if not isinstance(orf, str):
                            n_control += 1
                            control_guides.add(guide_seq)
                            continue
                        key = (orf, guide_seq)
                        if key not in perturbations:
                            # The released gene name, or the ORF again when the cell
                            # is NA (the release names no gene for 162,753 gene rows).
                            released_name = (
                                gene_name if isinstance(gene_name, str) else orf
                            )
                            perturbations[key] = self.perturbation_for(
                                orf, released_name, guide_seq
                            )
                            pert_ptr[key] = self._interned_pointer(
                                perturbations[key], orf, itxn
                            )
                            target_genes.add(orf)
                        replicate_id = f"{geno_bc}_{guide_bc}"
                        if replicate_id in lineages:
                            raise ValueError(
                                f"{spec.screen_id}: lineage {replicate_id} twice"
                            )
                        lineages.add(replicate_id)
                        phenotype = self.phenotype_for(
                            spec.screen_id, fitness, error, replicate_id
                        )
                        role = role_of[genotype]
                        genotype_model = PerturbedSegregantGenotype(
                            background=backgrounds[genotype],
                            background_role=role,
                            perturbations=[perturbations[key]],
                        )
                        experiment = {
                            "experiment_type": "segregant_perturbation",
                            "dataset_name": self.name,
                            "genotype": {
                                "background": bg_ptr[genotype],
                                "background_role": role,
                                "perturbations": [pert_ptr[key]],
                            },
                            "environment": env_ptr[induced],
                            "phenotype": phenotype.model_dump(),
                        }
                        if (spec.screen_id, genotype) not in checked:
                            # The composed dict must be exactly the pydantic dump with
                            # the constants swapped for their pointers.
                            checked.add((spec.screen_id, genotype))
                            full = SegregantPerturbationExperiment(
                                dataset_name=self.name,
                                genotype=genotype_model,
                                environment=environments[induced],
                                phenotype=phenotype,
                            ).model_dump()
                            full["genotype"]["background"] = bg_ptr[genotype]
                            full["genotype"]["perturbations"] = [pert_ptr[key]]
                            full["environment"] = env_ptr[induced]
                            if full != experiment:
                                raise ValueError(
                                    f"{spec.screen_id}/{genotype}: composed record drifted"
                                )
                        record = {
                            "experiment": experiment,
                            "reference": ref_ptr[spec.screen_id],
                            "publication": pub_ptr,
                        }
                        txn.put(f"{idx}".encode(), pickle.dumps(record))
                        idx += 1
            member_ranges[spec.screen_id] = (start, idx)
            dropped_control[spec.screen_id] = n_control
            seen_lineages[spec.screen_id] = lineages
            if idx - start != spec.gene_rows:
                raise ValueError(
                    f"{spec.screen_id}: wrote {idx - start} records, expected {spec.gene_rows}"
                )
        env.close()
        interned_env.close()
        if idx != EXPECTED_RECORDS:
            raise ValueError(f"wrote {idx} records, expected {EXPECTED_RECORDS}")

        self._write_reference_index(references, member_ranges)
        self._write_preprocess(
            backgrounds=backgrounds,
            target_genes=target_genes,
            member_ranges=member_ranges,
            seen_lineages=seen_lineages,
            rules=[
                DropRule(
                    rule="control_guide_targets_no_gene",
                    scope="guide",
                    description=(
                        "the lineage carries one of the control gRNAs that target "
                        "intergenic and noncoding regions ('Control gRNAs show 'NA' under "
                        "the ORF column'); a guide that perturbs no gene is never a "
                        "record, it is the reference the assays are normalized on"
                    ),
                    n_records=sum(dropped_control.values()),
                    per_assay=dropped_control,
                    items=sorted(control_guides),
                )
            ],
        )
        log.info("Wrote %d Hale 2024 records", idx)

    def _write_reference_index(
        self,
        references: dict[str, SegregantPerturbationExperimentReference],
        member_ranges: dict[str, tuple[int, int]],
    ) -> None:
        """One reference per assay over its contiguous record range (no record re-read)."""
        index = [
            ExperimentReferenceIndex(
                reference=references[spec.screen_id],
                member_indices=list(range(*member_ranges[spec.screen_id])),
            )
            for spec in ASSAYS
        ]
        with open(
            osp.join(self.preprocess_dir, "experiment_reference_index.json"), "w"
        ) as fh:
            json.dump([eri.model_dump() for eri in index], fh)

    def _write_preprocess(
        self,
        *,
        backgrounds: dict[str, SegregantGenotype],
        target_genes: set[str],
        member_ranges: dict[str, tuple[int, int]],
        seen_lineages: dict[str, set[str]],
        rules: list[DropRule],
    ) -> None:
        with open(osp.join(self.preprocess_dir, "target_genes.json"), "w") as fh:
            json.dump(sorted(target_genes), fh)
        with open(osp.join(self.preprocess_dir, "dropped_records.json"), "w") as fh:
            json.dump([r.model_dump() for r in rules], fh, indent=2)
        with open(osp.join(self.preprocess_dir, "block_counts.json"), "w") as fh:
            json.dump(
                {k: v.n_blocks for k, v in sorted(backgrounds.items())}, fh, indent=2
            )
        lineage_sets = list(seen_lineages.values())
        summary = {
            "records": member_ranges[ASSAYS[-1].screen_id][1],
            "member_ranges": member_ranges,
            "target_genes": len(target_genes),
            "lineages_per_assay": {k: len(v) for k, v in seen_lineages.items()},
            "lineages_in_all_assays": len(set.intersection(*lineage_sets)),
            "lineages_in_any_assay": len(set.union(*lineage_sets)),
        }
        with open(osp.join(self.preprocess_dir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)

    # ---- gene set --------------------------------------------------------------
    def compute_gene_set(self) -> GeneSet:
        """The current R64 ORFs the served guides target (written during ``process``)."""
        path = osp.join(self.preprocess_dir, "target_genes.json")
        with open(path) as fh:
            genes = json.load(fh)
        if not genes:
            raise ValueError("no target gene was served")
        return GeneSet(genes)

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing happens inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Records are assembled inside process()."""
        raise NotImplementedError(
            "CrispriSegregantHale2024Dataset builds records in process()"
        )


def main() -> None:
    """Build/load the dataset for interactive debugging."""
    data_root = _data_root()
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    dataset = CrispriSegregantHale2024Dataset(
        root=osp.join(data_root, "data/torchcell/crispri_segregant_hale2024"),
        genome=genome,
    )
    print(f"len = {len(dataset)}")
    record = dataset[0]
    print(record["experiment"].genotype.background.segregant_id)
    print(record["experiment"].phenotype.environment_response)


if __name__ == "__main__":
    main()
