# torchcell/datasets/scerevisiae/nguessan2025
# [[torchcell.datasets.scerevisiae.nguessan2025]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/nguessan2025
# Test file: tests/torchcell/datasets/scerevisiae/test_nguessan2025.py
"""N'Guessan 2025 one-pot single-cell RNA-seq of BY x RM segregants (eLife,
doi:10.7554/eLife.93906; citation key ``nguessanRefiningResolutionYeast2025``).

One pooled panel: batch 1 of the Nguyen Ba 2022 barcoded bulk QTL (BB-QTL) cross, 4,489
F2 segregants of BY4741 x RM11-1a, grown together in YPD to log phase and profiled in a
single 10x 3' run. CellRanger kept 18,233 barcodes. Each cell was genotyped from its
transcribed SNPs by the authors' HMM and assigned to the batch-1 strain whose
reference-panel genotype is closest (expected distance), kept only when the match beats
the best match in random batches of the same size (a permutation test). The release
holds the assignment, not the per-cell genotype: ``significant_best_match`` is the
0-based batch-1 index of the matched strain, or NA.

A record is one SEGREGANT, the same design as Boocock 2025's pooled panel: the UMI
counts of its significantly assigned cells summed (pseudobulk) and the reference-panel
genotype of the strain the cells were assigned to. Measured on the release: 14,263 of
the 18,233 cells carry a significant match, to 2,951 distinct strains (a median of 3
cells and a mean of 4.83, Table 1's "Average number of cell barcodes per F2 segregant");
the other 3,970 cells are refused (no assignment, which includes every doublet by the
paper's own definition).

GENOTYPE (``SegregantGenotype``, never ``SequenceVariantPerturbation``). The
reference-panel genotypes are Nguyen Ba 2022's: "an inferred posterior probability of
that strain having the RM allele" at 41,594 SNPs (Dryad doi:10.5061/dryad.1rn8pk0vd,
``geno_data_*.txt.gz``, rows keyed by ``segregant_info.txt`` ``Number``). They are stored
the way Boocock 2025 stores a released posterior: runs of constant called parent and
constant P(called parent) on a ``POSTERIOR_STEP`` grid (parent 2 = RM, the allele the
posterior is a probability of). A marker at exactly 0.5 has no called parent; it is
written as parent 1 at posterior 0.5, which states the same distribution, and the round
trip treats it as a tie. Parents: BY4741 is pinned to the S288C R64-4-1 reference
sequence, RM11-1a to the Peter 2018 assembly ``AAA`` (Peter 2018 Table S1), the same two
pins Bloom 2019 and Boocock 2025 use; no new assembly set.

**The Dryad deposit is walled** (HTTP 401 on the files API, 403 on the download route;
issue #1014), so the four Dryad files are a by-hand ``manual_browser`` deposit: the
owner downloaded the four-file selection in a browser on 2026-10-12, the zip Dryad
returned is kept beside its four members (``DEPOSIT.md`` in the mirror directory), and
every member hashes to the sha256 Dryad lists for it. ``download()`` refuses while any
is absent. The Dryad README states the formats (tab-separated; ``segregant_info.txt``
has ``Number`` and ``Batch`` columns; each ``geno_data`` row is ``Number`` then one
posterior per SNP) but not the SNP list's columns. ``SNP_LIST_COLUMNS`` is the header
measured on the deposited file, which also closes every chromosome with an ``END`` row
that is not a SNP: 41,610 rows, 16 of them ``END``, leave 41,594 SNPs, the number of
posteriors on every geno row. The k-th posterior of a geno row is taken to be the k-th
SNP row in file order (``snp<k>``, k from 1); the counts agree and the README says
nothing more, so the L4 ``mat_fixation`` and ``cis_concordance`` rules test it. The map
from the release's batch-1 index to ``Number`` is the batch-1 rows of
``segregant_info.txt`` in file order, the order the authors' code reads batch 1 in (the
first 4,489 rows of its panel); on the deposit those are ``Number`` 0 to 4,488 in
order, and L4 ``cis_concordance`` tests the map against a permutation null rather than
assuming it.

PHENOTYPE (``SingleCellExpressionPhenotype``, the single-cell UMI-count leaf shared with
Boocock 2025 and Jackson 2020, #1016): UMI counts over the 6,240 released genes, summed
over the segregant's cells, with each summed cell's barcode, library and UMI total. The
authors kept every feature "EXPRESSED IN AT LEAST 1 CELL", so a dropped feature has zero
UMIs in every cell and a cell's row sum is its UMI total over every feature (Table 1's
median 5,158). No cell-cycle stage is released, so
``cell_cycle_stage_counts`` is a typed gap. Every barcode carries CellRanger's GEM-group
suffix ``-1``, the single library, which is ``library_cell_counts``' one label. The
released features are S288C CDS only, so ``marker_umi_count`` is a typed gap. The
reference is the panel pseudobulk: every assigned cell summed, an absolute baseline.

ENVIRONMENT. YPD (1% yeast extract, 2% peptone, 2% dextrose), library ``YPD_LIQUID``
with the paper's own sentences; ``duration_hours=4`` from the passage to log phase. No
growth temperature is stated (only the 37 C spheroplasting step), so temperature is a
typed gap, never a guessed 30 C.

NOT LOADED (each in the dendron note): the eQTL, QTL and variance-partitioning results
(Supplementary files 1 to 3; derived analyses, not phenotypes); the per-cell HMM
genotypes (not released); the BB-QTL fitness of these strains (Nguyen Ba 2022's own
measurement, a separate candidate row); raw reads (SRA PRJNA1022775).

RAW MIRROR. ``$DATA_ROOT/torchcell-raw/nguessanRefiningResolutionYeast2025/`` with a
``manifest.json``: the four release files from GitHub ``arnaud00013/sc-eQTL`` at the
commit the paper archives (``1e0e2109``), the README and the lineage-assignment script
quoted below, the Dryad dataset metadata and file listing (readable without a token), and
the four walled Dryad files with the zip they arrived in, as ``manual_browser`` records.
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import os
import os.path as osp
import re
import shutil
import tarfile
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from torchcell.data import (
    ExperimentDataset,
    Organism,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.media import YPD_LIQUID, restated
from torchcell.datamodels.pydant import ModelStrict
from torchcell.datamodels.schema import (
    ArtifactRef,
    Environment,
    Experiment,
    ExperimentReference,
    HaplotypeBlock,
    Media,
    Publication,
    ReferenceGenome,
    SegregantExpressionExperiment,
    SegregantExpressionExperimentReference,
    SegregantGenotype,
    SegregantParent,
    SingleCellExpressionPhenotype,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.scerevisiae.bloom2019 import (
    PARENT_PETER_ID,
    Marker,
    ParentAssemblyRefs,
)
from torchcell.datasets.scerevisiae.boocock2025 import (
    POSTERIOR_STEP,
    expand_posterior_blocks,
    feature_universe_sha256,
)
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence import GeneSet
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "nguessanRefiningResolutionYeast2025"
PAPER_DOI = "10.7554/eLife.93906"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
#: sha256 of the mirrored MinerU ``paper.md`` (the library key's ``manifest.json``).
PAPER_MD_SHA256 = "97cf5c178323798ef84fe4763f9a3d7361b6d37907fbfe320d607996feab861b"
SRA_BIOPROJECT = "PRJNA1022775"

CODE_REPO = "arnaud00013/sc-eQTL"
#: The commit the paper's Software Heritage citation archives (swh:1:rev:1e0e2109...).
CODE_COMMIT = "1e0e2109014ab62cc6636fb2359186dc25d43f8d"

EXPRESSION_PARTS = (
    "Matrix_gene_expression_barcodes_1_to_9000.csv.tar.gz",
    "Matrix_gene_expression_barcodes_9001_to_18233.csv.tar.gz",
)
GENE_TABLE = "Table_expressed_genes.csv"
ASSIGNMENT_TABLE = (
    "Table_single_cell_barcodes_mapping_to_reference_panel_strains_0_based_index.csv"
)
RELEASE_FILES = (*EXPRESSION_PARTS, GENE_TABLE, ASSIGNMENT_TABLE)
#: Repository files whose lines this module and its verifier quote.
CODE_FILES = (
    "README.md",
    "III_Genotype_analysis/generate_dist_cell_to_lineage_corrected_gen.py",
)

DRYAD_DOI = "10.5061/dryad.1rn8pk0vd"
DRYAD_VERSION_ID = 175928
DRYAD_DATASET_API = (
    "https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.1rn8pk0vd"
)
DRYAD_FILES_API = (
    f"https://datadryad.org/api/v2/versions/{DRYAD_VERSION_ID}/files?per_page=100"
)
DRYAD_METADATA = "dataset_metadata.json"
DRYAD_LISTING = "version_175928_files.json"
README_NAME = "README.txt"
SNP_LIST_NAME = "SNP_list.txt"
SEGREGANT_INFO_NAME = "segregant_info.txt"
GENO_NAME = "geno_data_1.txt.gz"


class DryadFile(ModelStrict):
    """One walled Dryad file: its files-API id and the size and sha-256 Dryad lists."""

    name: str
    file_id: int
    bytes: int
    sha256: str

    @property
    def download_url(self) -> str:
        """The files-API download route (HTTP 401 without a bearer token, 2026-10-11)."""
        return f"https://datadryad.org/api/v2/files/{self.file_id}/download"


#: The four Dryad files the loader consumes, with Dryad's own digests (``digestType``
#: ``sha-256`` in ``/api/v2/versions/175928/files``); the by-hand deposit must hash to
#: these. ``geno_data_1`` holds 20,000 strains and is expected to hold batch 1.
DRYAD_FILES: dict[str, DryadFile] = {
    f.name: f
    for f in (
        DryadFile(
            name=README_NAME,
            file_id=1474448,
            bytes=3400,
            sha256="a4514d32d5fd3058130b05ce4172c10fd3ce20238810e52ab13bffd38ca5612f",
        ),
        DryadFile(
            name=SNP_LIST_NAME,
            file_id=1474445,
            bytes=1159378,
            sha256="333550aa3a7eecd2c90b5550c36cc7f59257d878142a423fa24402724c3b59b6",
        ),
        DryadFile(
            name=SEGREGANT_INFO_NAME,
            file_id=1474444,
            bytes=5691263,
            sha256="0d46317ec9e8c37ae903d1337699baeaacd7f2c8329812498474719dd9c765bc",
        ),
        DryadFile(
            name=GENO_NAME,
            file_id=1474423,
            bytes=5078870207,
            sha256="25dc87034c470d7c109b61c8a52897a1e854953909a416f34e89d3ff7a36a366",
        ),
    )
}
DRYAD_ISSUE = 1014
#: The by-hand deposit of the four files (#1014), as ``data/dryad/DEPOSIT.md`` in the
#: raw mirror records it: the owner downloaded the four-file selection in a browser.
DRYAD_RETRIEVED_AT = "2026-10-12"
DRYAD_RETRIEVED_BY = (
    "the owner (mjvolk3), browser download; the zip pushed to GilaHyper by rsync"
)
#: The zip Dryad returned for the selection, kept unchanged under the name it arrived
#: with (the browser's " (1)" suffix included) beside its four members, which it holds
#: stored, uncompressed. Dryad lists no digest for an assembled selection, so its
#: sha256 is our measurement at the deposit (slurm job 4195, 2026-10-12).
DRYAD_ARCHIVE_NAME = "doi_10_5061_dryad_1rn8pk0vd__v20220520 (1).zip"
DRYAD_ARCHIVE_BYTES = 5085724902
DRYAD_ARCHIVE_SHA256 = (
    "2fcaf5bc37bdf895846ecafd971902200956905cf297de6d04c8ffc5c286af0e"
)
#: Our own records beside the deposited bytes, named in each deposit record's params.
DRYAD_DEPOSIT_RECORD = "DEPOSIT.md"
DRYAD_CHECKSUMS = "SHA256SUMS.txt"
DRYAD_RECIPE = (
    f"manual browser download (owner only; issue #{DRYAD_ISSUE}): datadryad.org answers "
    "HTTP 401 'Unauthorized, must have current bearer token' on /api/v2/files/<id>/"
    "download and HTTP 403 on /downloads/file_stream/<id> (measured 2026-10-11), so: "
    f"open https://doi.org/{DRYAD_DOI} in a browser, select the four files "
    f"{README_NAME}, {SNP_LIST_NAME}, {SEGREGANT_INFO_NAME} and {GENO_NAME} (not the "
    "24.7 GB dataset) and download the selection, which Dryad returns as one zip "
    "(doi_10_5061_dryad_1rn8pk0vd__v20220520.zip); put the zip unchanged under "
    "data/dryad/, unzip its four members beside it under their Dryad names, and check "
    "each member's sha256 against this record (Dryad's own digest, files API version "
    f"{DRYAD_VERSION_ID})"
)

#: Grid levels per unit posterior (20 for a step of 0.05); levels run 10..20.
_LEVELS_PER_UNIT = round(1 / POSTERIOR_STEP)

#: Strains in batch 1 of the reference panel ("a single batch (batch 1) of 4489 F2
#: segregants"; the authors' code takes the first 4,489 rows of the panel).
BATCH1_STRAINS = 4489
N_BARCODES = 18233
N_GENES = 6240
#: Measured on the release (the inventory script): cells with a significant match and
#: the distinct strains they match. These are the record and cell counts of the build.
N_ASSIGNED_CELLS = 14263
N_SEGREGANTS = 2951
#: The single library: CellRanger's GEM-group suffix on every released barcode.
LIBRARY_LABEL = "1"
GROWTH_DURATION_HOURS = 4.0

#: The header of ``SNP_list.txt`` as deposited (measured 2026-10-12 on the file Dryad
#: lists at sha256 ``333550aa...``; its README says only "Table with index, genome
#: position, and sequence information"). The reader refuses any other header.
SNP_LIST_COLUMNS = (
    "Index",
    "SNP",
    "Chromosome",
    "Position (bp)",
    "BY allele",
    "RM allele",
)
#: The ``SNP`` value of the row that closes each chromosome in ``SNP_list.txt``. It is
#: not a SNP: its position and both alleles are ``0``, and the geno rows carry no
#: posterior for it (16 such rows; 41,610 rows less 16 is the 41,594 posteriors of a
#: geno row).
SNP_END = "END"
#: The SNP list's chromosome numbers (``1``..``16``) -> the S288C names markers carry.
_ROMAN = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX",
    "X", "XI", "XII", "XIII", "XIV", "XV", "XVI",
]  # fmt: skip
CHROMOSOME_TOKEN: dict[str, str] = {str(i): f"chr{r}" for i, r in enumerate(_ROMAN, 1)}
_CHROM_ORDER = {f"chr{r}": i for i, r in enumerate(_ROMAN)}

CALL_METHOD = (
    "reference-panel genotype of the matched batch-1 strain: Nguyen Ba 2022's inferred "
    "posterior probability of the RM allele at 41,594 SNPs (Dryad "
    f"doi:{DRYAD_DOI} {GENO_NAME}), the strain being the cell's significant best match "
    f"({ASSIGNMENT_TABLE} significant_best_match, a 0-based index into batch 1; expected "
    "distance plus a permutation test against random batches), encoded as runs of "
    "constant called parent (P(RM) > 0.5 calls RM) and constant P(called parent) on a "
    f"{POSTERIOR_STEP} grid; an exact 0.5 is written as parent 1 at 0.5. Block "
    f"coordinates are the SNP positions as released ({SNP_LIST_NAME}), on the reference "
    "the Dryad README names: S288C_reference_genome_R64-3-1_20210421"
)
CROSS = "BBQTL_BYxRM"


def _paper_sv(value: object, quote: str, *, note: str | None = None) -> SourcedValue:
    """A SourcedValue anchored to the mirrored N'Guessan 2025 ``paper.md``."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri="paper.md", citation_key=CITATION_KEY, sha256=PAPER_MD_SHA256
        ),
        quote=quote,
        note=note,
    )


# Quotes are verbatim substrings of the MinerU ``paper.md`` (LaTeX markup kept; the
# OCR's non-breaking spaces written as \xa0).
SOURCED: dict[str, SourcedValue] = {
    "panel": _paper_sv(
        "batch 1 of the BB-QTL panel: 4,489 F2 segregants of BY4741 x RM11-1a",
        "We analyzed cells from a single batch (batch 1) of 4489\xa0 F2 segregants "
        "obtained from a F1 cross between the yeast laboratory strain BY4741 and the "
        "vineyard strain RM11-1a generated in a previous study (Nguyen Ba et\xa0al., "
        "2022).",
    ),
    "genotypes_from_panel": _paper_sv(
        "reference-panel genotypes are Nguyen Ba 2022's",
        "Genotypes and fitness data used were the same ones obtained in the previous "
        "study.",
    ),
    "allele_coded": _paper_sv(
        "the genotype is the frequency of the RM allele",
        "Thus, the genotype of the segregants can be represented by the frequency of "
        "only one of the parental alleles, which is RM in the dataset.",
    ),
    "n_barcodes": _paper_sv(
        N_BARCODES, "The barcode filtering retained 18,233 barcodes."
    ),
    "assignment": _paper_sv(
        "a cell is kept on its best batch-1 match only when the match beats random "
        "batches of the same size",
        "Next, we assigned the single cell to its best match in the studied batch of "
        "4489 trains only if this match is better than the best match in randomly "
        "generated batches of the same size",
    ),
    "cells_per_segregant": _paper_sv(
        4.83,
        "Average number of cell barcodes per F2 segregant                         "
        "                                      4.83",
    ),
    "median_genes": _paper_sv(
        1311,
        "Median number of covered genes per cell                                  "
        "                                      1,311",
    ),
    "median_umi": _paper_sv(
        5158,
        "Median number of                                                         "
        "                                     5,158unique molecular identifiers per "
        "cell",
    ),
    "doublets": _paper_sv(
        "a doublet has no significant genotype association",
        "The doublets were defined as scRNA-seq barcodes with high coverage $\\geq$ "
        "third quartile) without significant genotype association to any F2 "
        "segregants",
    ),
    "medium": _paper_sv(
        "YPD: 1% yeast extract, 2% peptone, 2% dextrose",
        "inoculated approximately $5 ^ { \\star } 1 0 ^ { 6 }$ cells in YPD $1 \\%$ "
        "yeast extract, $2 \\%$ peptone, $2 \\%$ dextrose) to saturation.",
    ),
    "passage": _paper_sv(
        GROWTH_DURATION_HOURS,
        "The next day, about $\\scriptstyle 1 0 ^ { 7 }$ cells were passaged to $5 "
        "\\mathsf { m L }$ of fresh YPD and grew for 4\xa0hours to bring cells to log "
        "phase.",
    ),
    "log_phase": _paper_sv(
        "one time point, log phase",
        "We note that the cells analyzed here were grown in bulk and assayed for their "
        "transcriptome in log phase.",
    ),
    "spheroplasting": _paper_sv(
        "37 C is the spheroplasting step, not growth",
        "The cells were incubated at $3 7 ^ { \\circ } \\mathsf { C }$ for "
        "approximately 10\xa0minutes at which point spheroplasting was verified",
    ),
    "chemistry": _paper_sv(
        "10x Chromium Next GEM single-cell 3' kit, NextSeq 500",
        "Chromium Next GEM Single-cell $3 ^ { \\prime }$ Reagent Kit to prepare the "
        "sequencing libraries and sequenced on a NextSeq 500 high-output flow cell.",
    ),
    "cellranger": _paper_sv(
        "CellRanger 3.1.0 count", "CellRanger version 3.1.0 (Zheng et\xa0al., 2017)."
    ),
    "release": _paper_sv(
        "the per-cell expression matrices are released on GitHub",
        "The single-cell barcodes expression data are also available at "
        "https://github.com/ arnaud00013/sc-eQTL as an archive file named "
        "Matrix_gene_expression_barcodes_1_to_9000.csv.tar.",
    ),
    "reads": _paper_sv(SRA_BIOPROJECT, "accession number PRJNA1022775."),
}


def _raw_sv(
    value: object, relpath: str, sha256: str, quote: str, *, note: str | None = None
) -> SourcedValue:
    """A SourcedValue anchored to a file of this key's RAW mirror (audited there)."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri=relpath, citation_key=CITATION_KEY, sha256=sha256
        ),
        quote=quote,
        note=note,
    )


README_REL = "code/README.md"
README_SHA256 = "143d18b8d322f3f8c33f2af5ac1b71f43586c345b8d3143760b7fab0bdd31902"
LINEAGE_SCRIPT_REL = (
    "code/III_Genotype_analysis/generate_dist_cell_to_lineage_corrected_gen.py"
)
LINEAGE_SCRIPT_SHA256 = (
    "397b32c21ecf3a281fb2750d8b8d320564059a0dcdbfdc945fc196d3f8e3bff8"
)
DRYAD_METADATA_REL = f"data/dryad/{DRYAD_METADATA}"
DRYAD_METADATA_SHA256 = (
    "ddaeabd1173433c461f444be51f44a264ced361edebdd540cc89dcfe3448b412"
)

#: Quotes from the release itself (raw mirror): the column meaning of the assignment
#: table, the gene filter, how batch 1 is indexed, and the Dryad file formats.
RELEASE_SOURCED: dict[str, SourcedValue] = {
    "best_match_index": _raw_sv(
        "best_match is a 0-based index of the closest reference-panel strain",
        README_REL,
        README_SHA256,
        "the column **best_match** indicates the **0-based index** of the closest "
        "reference panel strains to each single cell",
    ),
    "significant_match": _raw_sv(
        "significant_best_match is NA when the relatedness is not significant",
        README_REL,
        README_SHA256,
        "the column **significant_best_match** represents the same information but "
        "with missing values (NA) when the relatedness between the single cell and its "
        "closest strain is not statistically significant",
    ),
    "reference_genotypes": _raw_sv(
        f"reference-panel genotypes: Dryad doi:{DRYAD_DOI}",
        README_REL,
        README_SHA256,
        "The reference panel genotype can be downloaded at "
        "https://datadryad.org/stash/dataset/doi:10.5061/dryad.1rn8pk0vd.",
    ),
    "gene_filter": _raw_sv(
        "released genes: nonzero in at least one cell",
        LINEAGE_SCRIPT_REL,
        LINEAGE_SCRIPT_SHA256,
        "#ONLY SELECT GENES THAT ARE EXPRESSED IN AT LEAST 1 CELL",
    ),
    "batch1_rows": _raw_sv(
        "batch 1 is the first 4,489 rows of the panel; best_match indexes them",
        LINEAGE_SCRIPT_REL,
        LINEAGE_SCRIPT_SHA256,
        "np.nanargmin(mtx_dist_corrected_genotype_vs_all_reference_lineages[:,0:"
        "nb_lineages_in_batch1],axis=1)",
    ),
    "other_batches": _raw_sv(
        "random batches are drawn from the rows after the first 4,489",
        LINEAGE_SCRIPT_REL,
        LINEAGE_SCRIPT_SHA256,
        "np.random.randint(4489, nb_rows_all_genotype_df, 4489)",
    ),
    "geno_rows": _raw_sv(
        "a geno_data row is Number then one posterior per SNP",
        DRYAD_METADATA_REL,
        DRYAD_METADATA_SHA256,
        "The first column contains the identifying number of that strain, as in the "
        "'Number'",
    ),
    "geno_posterior": _raw_sv(
        "geno_data values are P(RM allele)",
        DRYAD_METADATA_REL,
        DRYAD_METADATA_SHA256,
        "The genotype is encoded at the approximately 42,000 single-nucleotide "
        "polymorphisms between the two parental strains as an inferred posterior "
        "probability of that strain having the RM allele at that locus.",
    ),
    "geno_files": _raw_sv(
        "20,000 strains per geno file",
        DRYAD_METADATA_REL,
        DRYAD_METADATA_SHA256,
        "Each file contains the genotype of 20,000 strains.",
    ),
    "segregant_info": _raw_sv(
        "segregant_info.txt: Number, Batch, Set, Plate, Well, barcode",
        DRYAD_METADATA_REL,
        DRYAD_METADATA_SHA256,
        "Unique indeces, library location (Batch, Set, Plate, Well) and barcode "
        "sequence of each of the",
    ),
    "snp_reference": _raw_sv(
        "SNP coordinates on S288C R64-3-1",
        DRYAD_METADATA_REL,
        DRYAD_METADATA_SHA256,
        "on which this dataset is based is S288C_reference_genome_R64-3-1_20210421 by "
        "the Saccharomyces",
    ),
}

MEDIUM: Media = restated(YPD_LIQUID, SOURCED["medium"], SOURCED["passage"])
"""YPD of the scRNA-seq culture: library ``YPD_LIQUID`` (the same three ingredients at
the same percentages) plus the paper's two sentences. Same ``media_identity``."""

_PAPER_PROVENANCE = Provenance(
    source_uri="paper.md", citation_key=CITATION_KEY, sha256=PAPER_MD_SHA256
)
TEMPERATURE_GAP = ProvenanceGap(
    field="temperature",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_PAPER_PROVENANCE,
    note="the Methods state the YPD saturation culture and the 4 h passage to log phase "
    "but no temperature; the only temperature in the protocol is the 37 C "
    "spheroplasting step after growth",
)
CELL_CYCLE_GAP = ProvenanceGap(
    field="cell_cycle_stage_counts",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_PAPER_PROVENANCE,
    note="the release assigns no cell-cycle stage to any cell; the paper controls for "
    "the cell cycle only by building consensus profiles",
)
MARKER_GAP = ProvenanceGap(
    field="marker_umi_count",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_PAPER_PROVENANCE,
    note="CellRanger 'mapped the scRNA-seq reads to the reference genome' with no "
    "cassette sequence named, and the released feature table "
    "(Table_expressed_genes.csv) lists 6,240 S288C CDS features and no drug-cassette "
    "transcript, so no cassette is counted apart from the genes",
)


# --------------------------------------------------------------------------- #
# Reading the GitHub release
# --------------------------------------------------------------------------- #
class ExpressionRelease(ModelStrict):
    """The two released matrices joined: cells x genes UMI counts, row-aligned."""

    model_config = {"arbitrary_types_allowed": True, "frozen": True}

    barcodes: list[str]
    genes: list[str]
    counts: Any  # scipy.sparse CSR, int64, cells x genes

    @property
    def cell_totals(self) -> np.ndarray[Any, Any]:
        """UMIs per cell over the released genes (every feature, see the docstring)."""
        return np.asarray(self.counts.sum(axis=1)).ravel().astype(np.int64)


def read_expression_part(path: str | Path) -> tuple[list[str], list[str], Any]:
    """One ``Matrix_gene_expression_barcodes_*.csv.tar.gz``: barcodes, genes, CSR counts.

    The tarball holds one CSV whose header names the genes and whose rows are a barcode
    followed by one UMI count per gene; any other layout, a non-integral or negative
    count, raises.
    """
    import scipy.sparse as sp

    with tarfile.open(path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        if len(members) != 1:
            raise ValueError(f"{path}: expected one CSV member, found {len(members)}")
        handle = tar.extractfile(members[0])
        if handle is None:
            raise ValueError(f"{path}: member {members[0].name} is not readable")
        frame = pd.read_csv(handle, index_col=0)
    values = frame.to_numpy()
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{path}: counts are {values.dtype}, not integers")
    if (values < 0).any():
        raise ValueError(f"{path}: negative UMI counts")
    return (
        [str(b) for b in frame.index],
        [str(g) for g in frame.columns],
        sp.csr_matrix(values.astype(np.int64)),
    )


def read_expression(paths: Sequence[str | Path]) -> ExpressionRelease:
    """Join the released parts (same gene columns, disjoint barcodes, release order)."""
    import scipy.sparse as sp

    barcodes: list[str] = []
    genes: list[str] | None = None
    blocks = []
    for path in paths:
        part_barcodes, part_genes, counts = read_expression_part(path)
        if genes is not None and part_genes != genes:
            raise ValueError(f"{path}: gene columns differ from the first part")
        genes = part_genes
        barcodes.extend(part_barcodes)
        blocks.append(counts)
    if genes is None:
        raise ValueError("no expression parts given")
    if len(set(barcodes)) != len(barcodes):
        raise ValueError("a barcode appears in more than one row")
    return ExpressionRelease(
        barcodes=barcodes, genes=genes, counts=sp.vstack(blocks, format="csr")
    )


def read_gene_table(path: str | Path) -> list[str]:
    """``Table_expressed_genes.csv``: the released genes in matrix-column order.

    Columns ``Expressed_gene_original_index`` (CellRanger feature index),
    ``sc_eQTL_gene_id`` (0..n-1, the matrix column) and ``Gene_name`` (``<ORF>_CDS``);
    returns the ORF names with the ``_CDS`` suffix removed.
    """
    table = pd.read_csv(path, sep="\t")
    expected = ["Expressed_gene_original_index", "sc_eQTL_gene_id", "Gene_name"]
    if list(table.columns) != expected:
        raise ValueError(f"{path}: columns {list(table.columns)} != {expected}")
    if table["sc_eQTL_gene_id"].tolist() != list(range(len(table))):
        raise ValueError(f"{path}: sc_eQTL_gene_id is not 0..{len(table) - 1}")
    names = [str(n) for n in table["Gene_name"]]
    if not all(n.endswith("_CDS") for n in names):
        raise ValueError(f"{path}: a Gene_name lacks the _CDS suffix")
    return [n.removesuffix("_CDS") for n in names]


class CellAssignment(ModelStrict):
    """One released cell: its best batch-1 match and, if significant, that match."""

    barcode: str
    best_match: int
    significant_best_match: int | None


def read_assignment(path: str | Path) -> list[CellAssignment]:
    """The barcode-to-strain table; a significant match must equal the best match."""
    table = pd.read_csv(path, dtype={"single_cell_barcode": str})
    expected = ["single_cell_barcode", "best_match", "significant_best_match"]
    if list(table.columns) != expected:
        raise ValueError(f"{path}: columns {list(table.columns)} != {expected}")
    out: list[CellAssignment] = []
    for barcode, best, significant in table.itertuples(index=False):
        match = None if pd.isna(significant) else int(significant)
        if match is not None and match != int(best):
            raise ValueError(f"{barcode}: significant match {match} != best {best}")
        out.append(
            CellAssignment(
                barcode=str(barcode), best_match=int(best), significant_best_match=match
            )
        )
    return out


def library_of(barcode: str) -> str:
    """The CellRanger GEM-group suffix of a barcode (``...-1`` -> ``1``)."""
    match = re.fullmatch(r"[ACGT]+-(\d+)", barcode)
    if match is None:
        raise ValueError(f"barcode {barcode!r} is not <ACGT...>-<GEM group>")
    return match.group(1)


def assigned_cells(assignment: Sequence[CellAssignment]) -> dict[int, list[int]]:
    """Batch-1 index -> the release rows of the cells significantly assigned to it."""
    cells: dict[int, list[int]] = {}
    for row, cell in enumerate(assignment):
        if cell.significant_best_match is not None:
            cells.setdefault(cell.significant_best_match, []).append(row)
    return cells


# --------------------------------------------------------------------------- #
# Reading the Dryad reference panel (format per the Dryad README; #1014)
# --------------------------------------------------------------------------- #
def read_batch_numbers(path: str | Path, batch: int = 1) -> list[int]:
    """``segregant_info.txt``: the ``Number`` of every strain of ``batch``, file order.

    The release's batch-1 index i is the i-th of these (see the module docstring).
    """
    table = pd.read_csv(path, sep="\t")
    missing = {"Number", "Batch"} - set(table.columns)
    if missing:
        raise ValueError(
            f"{path}: no {sorted(missing)} column (columns {list(table.columns)})"
        )
    numbers = [int(n) for n in table.loc[table["Batch"] == batch, "Number"]]
    if len(set(numbers)) != len(numbers):
        raise ValueError(f"{path}: a Number repeats within batch {batch}")
    return numbers


def read_snp_markers(path: str | Path) -> tuple[list[Marker], list[int]]:
    """``SNP_list.txt`` as sorted ``Marker``s plus the geno column of each.

    The deposited layout (module docstring): ``Index`` 0..n-1 over every row, one
    ``END`` row per chromosome after all of that chromosome's SNPs (position and alleles
    ``0``), and the SNPs named ``snp1``, ``snp2``, ... in file order. ``Marker.index``
    is the row's ``Index`` and ``Marker.column`` the SNP's 0-based column among the
    posteriors of a geno row (its rank among the non-``END`` rows). The returned list is
    sorted by (chromosome, position); the second value maps each sorted marker back to
    its geno column. Any other header, an ``Index`` that is not 0..n-1, a chromosome
    outside ``1``..``16``, a SNP out of sequence, a malformed ``END`` row, or a
    chromosome without exactly one ``END`` after its SNPs raises.
    """
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if tuple(table.columns) != SNP_LIST_COLUMNS:
        raise ValueError(
            f"{path}: header {tuple(table.columns)} is not the deposited "
            f"{SNP_LIST_COLUMNS}"
        )
    rows = list(table.itertuples(index=False, name=None))
    if [int(row[0]) for row in rows] != list(range(len(rows))):
        raise ValueError(f"{path}: Index is not 0..{len(rows) - 1}")
    unknown = sorted({row[2] for row in rows} - set(CHROMOSOME_TOKEN))
    if unknown:
        raise ValueError(f"{path}: chromosomes {unknown} are not 1..16")
    markers: list[Marker] = []
    closed: set[str] = set()
    for idx, snp, chrom, pos, by, rm in rows:
        if snp == SNP_END:
            if (pos, by, rm) != ("0", "0", "0") or chrom in closed:
                raise ValueError(
                    f"{path}: Index {idx} is not the one END row of chromosome {chrom}"
                )
            closed.add(chrom)
            continue
        if chrom in closed:
            raise ValueError(f"{path}: {snp} follows the END row of chromosome {chrom}")
        if snp != f"snp{len(markers) + 1}":
            raise ValueError(
                f"{path}: Index {idx} is {snp}, expected snp{len(markers) + 1}"
            )
        markers.append(
            Marker(
                chromosome=CHROMOSOME_TOKEN[chrom],
                position=int(pos),
                ref=by,
                alt=rm,
                index=int(idx),
                column=len(markers),
            )
        )
    open_chromosomes = sorted({row[2] for row in rows} - closed)
    if open_chromosomes:
        raise ValueError(f"{path}: chromosomes {open_chromosomes} have no END row")
    ordered = sorted(markers, key=lambda m: (_CHROM_ORDER[m.chromosome], m.position))
    return ordered, [m.column for m in ordered]


def read_reference_genotypes(
    path: str | Path, numbers: set[int], n_markers: int
) -> dict[int, np.ndarray[Any, Any]]:
    """Stream ``geno_data_*.txt.gz`` and keep the posterior rows of ``numbers``.

    A row is ``Number`` then ``n_markers`` posteriors, tab-separated. A kept row of the
    wrong length, a value outside [0, 1] or a non-finite value raises, and so does a
    requested Number the file does not hold (the error names how many are missing, so a
    batch-1 strain stored in another geno file is refused, not skipped).
    """
    out: dict[int, np.ndarray[Any, Any]] = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            first, _, rest = line.partition("\t")
            number = int(float(first))
            if number not in numbers:
                continue
            values = np.array(rest.rstrip("\n").split("\t"), dtype=np.float64)
            if values.size != n_markers:
                raise ValueError(
                    f"{path}: strain {number} has {values.size} posteriors, "
                    f"expected {n_markers}"
                )
            if (
                not np.isfinite(values).all()
                or (values < 0).any()
                or (values > 1).any()
            ):
                raise ValueError(
                    f"{path}: strain {number} holds a value outside [0, 1]"
                )
            out[number] = values
            if len(out) == len(numbers):
                break
    missing = numbers - set(out)
    if missing:
        raise ValueError(
            f"{path}: {len(missing)} requested strains are not in this file "
            f"(first {sorted(missing)[:5]}); deposit the geno file that holds them"
        )
    return out


# --------------------------------------------------------------------------- #
# Posterior blocks
# --------------------------------------------------------------------------- #
def encode_reference_posterior(
    posterior: np.ndarray[Any, Any], markers: Sequence[Marker]
) -> list[HaplotypeBlock]:
    """Runs of constant called parent and constant posterior level, per chromosome.

    ``posterior`` is P(RM) = P(parent 2) at the sorted ``markers``. The called parent is
    2 above 0.5 and 1 otherwise, so an exact 0.5 is parent 1 at posterior 0.5, the same
    distribution; the level is P(called parent) on the ``POSTERIOR_STEP`` grid. The
    Boocock encoder raises on 0.5 instead, because its release never holds one.
    """
    parent = np.where(posterior > 0.5, 2, 1).astype(np.int8)
    q = np.where(parent == 2, posterior, 1.0 - posterior)
    level = np.rint(q * _LEVELS_PER_UNIT).astype(np.int64)
    chrom = np.fromiter((_CHROM_ORDER[m.chromosome] for m in markers), dtype=np.int64)
    change = np.ones(len(markers), dtype=bool)
    change[1:] = (
        (parent[1:] != parent[:-1])
        | (level[1:] != level[:-1])
        | (chrom[1:] != chrom[:-1])
    )
    # Python ints, so a Sequence index is an int for every numpy stub version.
    starts: list[int] = np.flatnonzero(change).tolist()
    ends = [*(s - 1 for s in starts[1:]), len(markers) - 1]
    return [
        HaplotypeBlock(
            chromosome=markers[s].chromosome,
            start=markers[s].position,
            end=markers[e].position,
            parent=int(parent[s]),  # type: ignore[arg-type]
            posterior=round(int(level[s]) / _LEVELS_PER_UNIT, 2),
            n_markers=e - s + 1,
        )
        for s, e in zip(starts, ends)
    ]


def round_trip_error(
    blocks: Sequence[HaplotypeBlock],
    markers: Sequence[Marker],
    posterior: np.ndarray[Any, Any],
) -> tuple[bool, float]:
    """(called parents agree, worst |P(called parent) - stored level|) for one mosaic.

    A marker at exactly 0.5 agrees with either parent at level 0.5.
    """
    parent, level = expand_posterior_blocks(blocks, markers)
    called = np.where(posterior > 0.5, 2, 1)
    q = np.where(called == 2, posterior, 1.0 - posterior)
    tie = posterior == 0.5
    parents_ok = bool(np.all((parent == called) | (tie & (level == 0.5))))
    return parents_ok, float(np.max(np.abs(q - level))) if len(q) else 0.0


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_root() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/nguessanRefiningResolutionYeast2025``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def github_url(repo_path: str) -> str:
    """The raw URL of one repository file at the pinned commit."""
    return f"https://raw.githubusercontent.com/{CODE_REPO}/{CODE_COMMIT}/{repo_path}"


def raw_relpaths() -> dict[str, str]:
    """``raw/`` file name -> mirror-relative path, for every file the loader reads."""
    rel = {name: f"data/github/{name}" for name in RELEASE_FILES}
    rel.update({name: f"data/dryad/{name}" for name in DRYAD_FILES})
    return rel


def check_dryad_archive(path: str | Path, files: Mapping[str, DryadFile]) -> None:
    """The arrival zip holds exactly ``files``, each at its listed size, stored.

    Reads the zip's central directory only (its four members' bytes are verified by
    their own sha256 records). A missing, extra, resized or compressed member raises.
    """
    with zipfile.ZipFile(path) as archive:
        members = {
            info.filename: (info.file_size, info.compress_size)
            for info in archive.infolist()
        }
    expected = {name: (f.bytes, f.bytes) for name, f in files.items()}
    if members != expected:
        raise ValueError(
            f"{path}: members {members} are not the stored deposit {expected}"
        )


def deposit_raw_mirror(
    *,
    release_dir: str | Path,
    code_dir: str | Path,
    dryad_metadata_dir: str | Path,
    retrieved_at: str,
    data_root: str | None = None,
) -> Path:
    """Copy the retrieved files into the raw mirror and write its ``manifest.json``.

    Run by hand after the retrievals in the module docstring; idempotent by sha256 (a
    present file with the recorded hash is left alone, a differing one raises).
    ``retrieved_at`` dates the GitHub and Dryad-metadata retrievals. The four walled
    Dryad files and the zip they arrived in are the owner's by-hand deposit
    (``DRYAD_RETRIEVED_AT``): ``manual_browser`` records pinned to Dryad's listed
    sha256 (the zip to ours), written whether or not their bytes are on disk, so a
    rebuild names what to deposit; a present one must hash to its pin, and a present
    zip must hold exactly the four members (:func:`check_dryad_archive`).
    """
    root = raw_mirror_dir(data_root)
    rel = raw_relpaths()
    sources: dict[str, tuple[Path | None, RetrievalRecord, int]] = {}

    def record(
        method: RetrievalMethod,
        source_url: str,
        retriever: str,
        sha: str,
        when: str = retrieved_at,
        **params: Any,
    ) -> RetrievalRecord:
        return RetrievalRecord(
            method=method,
            source_url=source_url,
            retriever=retriever,
            params=params,
            sha256=sha,
            retrieved_at=when,
        )

    deposit_params: dict[str, Any] = {
        "retrieval_command": DRYAD_RECIPE,
        "retrieved_by": DRYAD_RETRIEVED_BY,
        "dryad_version_id": DRYAD_VERSION_ID,
        "received_in": f"data/dryad/{DRYAD_ARCHIVE_NAME}",
        "deposit_record": f"data/dryad/{DRYAD_DEPOSIT_RECORD}",
        "checksums": f"data/dryad/{DRYAD_CHECKSUMS}",
    }

    for name in RELEASE_FILES:
        src = Path(release_dir) / name
        url = github_url(name)
        sources[rel[name]] = (
            src,
            record(
                RetrievalMethod.direct_url,
                url,
                "torchcell.literature.retrieve.direct_url",
                _sha256(src),
                url=url,
                repo=CODE_REPO,
                commit=CODE_COMMIT,
                path=name,
            ),
            src.stat().st_size,
        )
    for repo_path in CODE_FILES:
        src = Path(code_dir) / repo_path
        url = github_url(repo_path)
        sources[f"code/{repo_path}"] = (
            src,
            record(
                RetrievalMethod.direct_url,
                url,
                "torchcell.literature.retrieve.direct_url",
                _sha256(src),
                url=url,
                repo=CODE_REPO,
                commit=CODE_COMMIT,
                path=repo_path,
            ),
            src.stat().st_size,
        )
    for name, url in (
        (DRYAD_METADATA, DRYAD_DATASET_API),
        (DRYAD_LISTING, DRYAD_FILES_API),
    ):
        src = Path(dryad_metadata_dir) / name
        sources[f"data/dryad/{name}"] = (
            src,
            record(
                RetrievalMethod.direct_url,
                url,
                "torchcell.literature.retrieve.direct_url",
                _sha256(src),
                url=url,
            ),
            src.stat().st_size,
        )
    for name, dryad in DRYAD_FILES.items():
        sources[rel[name]] = (
            None,
            record(
                RetrievalMethod.manual_browser,
                f"https://doi.org/{DRYAD_DOI}",
                "manual",
                dryad.sha256,
                DRYAD_RETRIEVED_AT,
                **deposit_params,
                dryad_file_id=dryad.file_id,
                dryad_download_url=dryad.download_url,
            ),
            dryad.bytes,
        )
    archive_rel = f"data/dryad/{DRYAD_ARCHIVE_NAME}"
    sources[archive_rel] = (
        None,
        record(
            RetrievalMethod.manual_browser,
            f"https://doi.org/{DRYAD_DOI}",
            "manual",
            DRYAD_ARCHIVE_SHA256,
            DRYAD_RETRIEVED_AT,
            **{k: v for k, v in deposit_params.items() if k != "received_in"},
            members=list(DRYAD_FILES),
        ),
        DRYAD_ARCHIVE_BYTES,
    )
    files: list[ArtifactRecord] = []
    for relpath, (source, retrieval, size) in sources.items():
        dest = root / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != retrieval.sha256:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        elif source is not None:
            shutil.copy2(source, dest)
        if relpath == archive_rel and dest.exists():
            check_dryad_archive(dest, DRYAD_FILES)
        files.append(
            ArtifactRecord(
                path=relpath,
                role=ROLE_RAW_DATA,
                bytes=size,
                sha256=retrieval.sha256,
                source=retrieval.source_url,
                retrieval=retrieval,
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=(
            "Refining the resolution of the yeast genotype-phenotype map using "
            "single-cell RNA-sequencing"
        ),
        files=files,
        si_data_sources=[
            f"https://github.com/{CODE_REPO}/tree/{CODE_COMMIT}",
            f"https://doi.org/{DRYAD_DOI}",
            f"https://www.ncbi.nlm.nih.gov/bioproject/{SRA_BIOPROJECT}",
        ],
        si_expected=[
            f"GitHub {CODE_REPO} @ {CODE_COMMIT[:8]}: the two expression matrices, the "
            "gene table and the barcode-to-strain assignment",
            f"Dryad {DRYAD_DOI} (Nguyen Ba 2022 reference panel): {README_NAME}, "
            f"{SNP_LIST_NAME}, {SEGREGANT_INFO_NAME}, {GENO_NAME}; walled (HTTP 401 on "
            "/api/v2/files/<id>/download, 403 on /downloads/file_stream/<id>, "
            f"2026-10-11), deposited by hand {DRYAD_RETRIEVED_AT} per issue "
            f"#{DRYAD_ISSUE} with the zip they arrived in ({DRYAD_ARCHIVE_NAME}). Every "
            "record depends on them.",
            f"SRA {SRA_BIOPROJECT}: raw reads, not deposited (no loader reads reads)",
        ],
        provenance_complete=all((root / relpath).exists() for relpath in sources),
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
    for artifact in manifest.files:
        if artifact.path == relpath:
            return artifact.sha256
    raise KeyError(f"{relpath} is not in the raw-mirror manifest")


def missing_dryad_files(data_root: str | None = None) -> list[str]:
    """The walled Dryad files not yet deposited in the raw mirror (sorted)."""
    root = raw_mirror_dir(data_root)
    rel = raw_relpaths()
    return sorted(name for name in DRYAD_FILES if not (root / rel[name]).exists())


# --------------------------------------------------------------------------- #
# Parents
# --------------------------------------------------------------------------- #
#: The source's words for each parent, verbatim (Methods, "Yeast strains and
#: segregants"); the marker genotypes are in Nguyen Ba 2022, not mirrored (#1015).
PARENT_BACKGROUND = {
    "BY4741": "the yeast laboratory strain BY4741",
    "RM11-1a": "the vineyard strain RM11-1a",
}


def build_parents(
    assemblies: ParentAssemblyRefs,
) -> tuple[SegregantParent, SegregantParent]:
    """BY4741 on the S288C reference; RM11-1a on its 1011-panel assembly (``AAA``)."""
    rm_id = PARENT_PETER_ID["RMx"]
    if rm_id is None or rm_id not in assemblies.index:
        raise RuntimeError(f"RM11-1a: Peter id {rm_id} is not in the member index")
    tarball = assemblies.tarball
    return (
        SegregantParent(
            name="BY4741",
            peter_strain_id=None,
            assembly_ref=assemblies.s288c,
            engineered_background=PARENT_BACKGROUND["BY4741"],
        ),
        SegregantParent(
            name="RM11-1a",
            peter_strain_id=rm_id,
            assembly_ref=ArtifactRef(
                tier=tarball.tier,
                key=tarball.key,
                path=tarball.path,
                member=assemblies.index[rm_id],
                sha256=tarball.sha256,
                bytes=tarball.bytes,
                media_type=tarball.media_type,
            ),
            engineered_background=PARENT_BACKGROUND["RM11-1a"],
        ),
    )


# --------------------------------------------------------------------------- #
# Phenotype
# --------------------------------------------------------------------------- #
def pseudobulk(
    release: ExpressionRelease, rows: Sequence[int], universe_sha256: str
) -> SingleCellExpressionPhenotype:
    """Sum the release ``rows`` (cells) into one phenotype."""
    selected = release.counts[list(rows)]
    summed = np.asarray(selected.sum(axis=0)).ravel()
    nonzero = np.flatnonzero(summed)
    cell_libraries = [library_of(release.barcodes[row]) for row in rows]
    libraries: dict[str, int] = {}
    for label in cell_libraries:
        libraries[label] = libraries.get(label, 0) + 1
    return SingleCellExpressionPhenotype(
        umi_count={release.genes[j]: int(summed[j]) for j in nonzero},
        marker_umi_count=None,
        total_umis=int(summed.sum()),
        n_features=len(release.genes),
        feature_universe_sha256=universe_sha256,
        n_cells=len(rows),
        cell_cycle_stage_counts=None,
        library_cell_counts=libraries,
        cell_barcodes=[release.barcodes[row] for row in rows],
        cell_libraries=cell_libraries,
        cell_total_umis=[int(n) for n in np.asarray(selected.sum(axis=1)).ravel()],
        provenance_gaps=[CELL_CYCLE_GAP, MARKER_GAP],
    )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@register_dataset
class Nguessan2025Dataset(ExperimentDataset):
    """N'Guessan 2025: 2,951 BB-QTL batch-1 segregants, their cells summed."""

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/nguessan2025",
        io_workers: int = 0,
        genome: SCerevisiaeGenome | None = None,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """A genome is REQUIRED (the gene set is the genes the markers span)."""
        self.genome = genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return SegregantExpressionExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return SegregantExpressionExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The four GitHub release files and the four walled Dryad files."""
        return list(raw_relpaths())

    def download(self) -> None:
        """Symlink the manifest-listed mirror files into ``raw/`` and verify each sha256.

        Refuses while any walled Dryad file is undeposited (issue #1014).
        """
        data_root = _data_root()
        missing = missing_dryad_files(data_root)
        if missing:
            raise RuntimeError(
                f"the walled Dryad files {missing} are not deposited in "
                f"{raw_mirror_dir(data_root) / 'data/dryad'}; every record needs the "
                f"reference-panel genotypes (issue #{DRYAD_ISSUE})"
            )
        manifest = load_manifest(data_root)
        rel = raw_relpaths()
        os.makedirs(self.raw_dir, exist_ok=True)
        for name in self.raw_file_names:
            src = raw_mirror_dir(data_root) / rel[name]
            link_verified(
                src, osp.join(self.raw_dir, name), manifest_sha256(manifest, rel[name])
            )

    @staticmethod
    def environment() -> Environment:
        """YPD, 4 h passage to log phase; the temperature is a typed gap."""
        return Environment(
            media=MEDIUM,
            temperature=None,
            aerobicity="aerobic",
            duration_hours=GROWTH_DURATION_HOURS,
            provenance_gaps=[TEMPERATURE_GAP],
        )

    def read_release(self) -> tuple[ExpressionRelease, list[CellAssignment]]:
        """The joined matrix and the assignment, checked against each other."""
        release = read_expression([osp.join(self.raw_dir, n) for n in EXPRESSION_PARTS])
        genes = read_gene_table(osp.join(self.raw_dir, GENE_TABLE))
        if genes != release.genes:
            raise ValueError("the gene table and the matrix columns differ")
        assignment = read_assignment(osp.join(self.raw_dir, ASSIGNMENT_TABLE))
        if [cell.barcode for cell in assignment] != release.barcodes:
            raise ValueError("the assignment table and the matrix rows differ")
        return release, assignment

    @post_process
    def process(self) -> None:
        """Build one record per significantly assigned batch-1 strain."""
        data_root = _data_root()
        manifest = load_manifest(data_root)
        rel = raw_relpaths()
        pins = {
            name: manifest_sha256(manifest, rel[name]) for name in self.raw_file_names
        }
        verify_raw_files(self.raw_dir, pins)
        os.makedirs(self.preprocess_dir, exist_ok=True)

        release, assignment = self.read_release()
        self._check_release(release, assignment)
        cells = assigned_cells(assignment)
        numbers = read_batch_numbers(osp.join(self.raw_dir, SEGREGANT_INFO_NAME))
        if len(numbers) != BATCH1_STRAINS:
            raise ValueError(
                f"segregant_info.txt holds {len(numbers)} batch-1 strains, expected "
                f"{BATCH1_STRAINS}"
            )
        markers, order = read_snp_markers(osp.join(self.raw_dir, SNP_LIST_NAME))
        needed = {numbers[i] for i in cells}
        genotypes = read_reference_genotypes(
            osp.join(self.raw_dir, GENO_NAME), needed, len(markers)
        )
        parents = build_parents(ParentAssemblyRefs.from_tier(data_root))
        universe = feature_universe_sha256(release.genes)
        environment = self.environment()
        all_rows = sorted(r for rows in cells.values() for r in rows)
        reference = SegregantExpressionExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae",
                strain="S288C",
                ploidy="haploid",
                assembly_set=SGD_S288C_R64,
                assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
            ),
            environment_reference=environment,
            phenotype_reference=pseudobulk(release, all_rows, universe),
        )
        publication = Publication(doi=PAPER_DOI, doi_url=f"https://doi.org/{PAPER_DOI}")

        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        index_rows: list[dict[str, Any]] = []
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for idx, experiment in enumerate(
                self._records(
                    release,
                    cells,
                    numbers,
                    markers,
                    order,
                    genotypes,
                    parents,
                    pins[GENO_NAME],
                    universe,
                    environment,
                )
            ):
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, publication, itxn),
                )
                index_rows.append(
                    {
                        "segregant_id": experiment.genotype.segregant_id,
                        "n_cells": experiment.phenotype.n_cells,
                        "n_blocks": experiment.genotype.n_blocks,
                    }
                )
        env.close()
        interned_env.close()
        if len(index_rows) != N_SEGREGANTS:
            raise ValueError(
                f"wrote {len(index_rows)} records, expected {N_SEGREGANTS}"
            )
        pd.DataFrame(index_rows).to_csv(
            osp.join(self.preprocess_dir, "data.csv"), index=False
        )

    def _records(
        self,
        release: ExpressionRelease,
        cells: dict[int, list[int]],
        numbers: list[int],
        markers: list[Marker],
        order: list[int],
        genotypes: dict[int, np.ndarray[Any, Any]],
        parents: tuple[SegregantParent, SegregantParent],
        geno_sha: str,
        universe: str,
        environment: Environment,
    ) -> Iterator[SegregantExpressionExperiment]:
        """One record per assigned strain, in batch-1 index order."""
        for index in sorted(cells):
            number = numbers[index]
            posterior = genotypes[number][order]
            blocks = encode_reference_posterior(posterior, markers)
            parents_ok, error = round_trip_error(blocks, markers, posterior)
            if not parents_ok or error > POSTERIOR_STEP / 2 + 1e-9:
                raise ValueError(f"strain {number}: posterior round trip failed")
            yield SegregantExpressionExperiment(
                dataset_name=self.name,
                genotype=SegregantGenotype(
                    cross=CROSS,
                    segregant_id=str(number),
                    parent_1=parents[0],
                    parent_2=parents[1],
                    blocks=blocks,
                    call_method=CALL_METHOD,
                    marker_matrix_sha256=geno_sha,
                ),
                environment=environment,
                phenotype=pseudobulk(release, cells[index], universe),
            )

    @staticmethod
    def _check_release(
        release: ExpressionRelease, assignment: Sequence[CellAssignment]
    ) -> None:
        """Barcode, gene, assignment and strain counts are the release's."""
        if len(release.barcodes) != N_BARCODES:
            raise ValueError(f"{len(release.barcodes)} barcodes, expected {N_BARCODES}")
        if len(release.genes) != N_GENES:
            raise ValueError(f"{len(release.genes)} genes, expected {N_GENES}")
        cells = assigned_cells(assignment)
        n_assigned = sum(len(rows) for rows in cells.values())
        if n_assigned != N_ASSIGNED_CELLS or len(cells) != N_SEGREGANTS:
            raise ValueError(
                f"{n_assigned} assigned cells over {len(cells)} strains, expected "
                f"{N_ASSIGNED_CELLS} over {N_SEGREGANTS}"
            )
        if max(cell.best_match for cell in assignment) >= BATCH1_STRAINS:
            raise ValueError("a best match indexes past batch 1")

    # ---- gene set ----------------------------------------------------------------
    @staticmethod
    def extract_systematic_gene_names(genotype: dict[str, Any]) -> list[str]:
        """Not applicable: a mosaic genotype has no gene-keyed perturbations."""
        raise NotImplementedError(
            "a SegregantGenotype carries no gene perturbations; "
            "Nguessan2025Dataset.compute_gene_set returns the S288C genes its markers span"
        )

    def compute_gene_set(self) -> GeneSet:
        """S288C genes overlapping the reference panel's marker span, per chromosome."""
        if self.genome is None:
            raise RuntimeError("Nguessan2025Dataset requires a genome for its gene set")
        markers, _ = read_snp_markers(osp.join(self.raw_dir, SNP_LIST_NAME))
        spans: dict[str, tuple[int, int]] = {}
        for chrom in _CHROM_ORDER:
            positions = [m.position for m in markers if m.chromosome == chrom]
            if positions:
                spans[chrom] = (min(positions), max(positions))
        gene_set = GeneSet()
        db = self.genome.db
        for gene_id in self.genome.gene_set:
            feature = db[gene_id]
            span = spans.get(str(feature.chrom))
            if span is not None and feature.start <= span[1] and feature.end >= span[0]:
                gene_set.add(gene_id)
        if not gene_set:
            raise ValueError("no S288C gene overlaps the marker span")
        return gene_set

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing happens inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Records are assembled inside process()."""
        raise NotImplementedError("Nguessan2025Dataset builds its records in process()")


def sourced_values() -> list[SourcedValue]:
    """Every quote of this paper the records rest on (the medium's two included).

    The medium's library quotes (Tong and Boone's YEPD recipe, Hoepfner 2014) belong to
    ``media.py`` and are not re-audited here: the YEPD recipe quote elides with "..."
    and is not a verbatim substring of its source (found 2026-10-11, recorded in the
    dendron note).
    """
    return list(SOURCED.values())


def verify_build(dataset_root: str, *, data_root: str | None = None) -> Any:
    """L0-L4 report for a built store (``segregant_expression_reference_panel``)."""
    import json

    from torchcell.sequence.genome.registry import PETER2018_1011, resolve
    from torchcell.verification import runners
    from torchcell.verification.segregant_expression_reference_panel import (
        gene_midpoints_from_genome,
        verify_reference_panel_expression_streaming,
    )

    root = data_root or _data_root()
    spec = runners.SEGREGANT_EXPRESSION_DATASETS["nguessan2025"]
    genome = SCerevisiaeGenome(
        genome_root=osp.join(root, "data/sgd/genome"),
        go_root=osp.join(root, "data/go"),
        overwrite=False,
    )
    index_path = resolve(PETER2018_1011, spec["assembly_index"], data_root=root)
    assembly_index = dict(
        line.split("\t") for line in Path(index_path).read_text().splitlines()
    )
    with open(osp.join(dataset_root, "preprocess", "gene_set.json")) as handle:
        gene_set = set(json.load(handle))
    raw_dir = osp.join(dataset_root, "raw")
    genes = read_gene_table(osp.join(raw_dir, GENE_TABLE))
    return verify_reference_panel_expression_streaming(
        runners.stream_records(dataset_root),
        dataset_name="nguessan2025",
        provenance=spec["provenance"],
        expected_count=spec["expected_count"],
        raw_dir=raw_dir,
        library_root=osp.join(root, spec["library"]),
        raw_root=osp.join(root, "torchcell-raw"),
        assembly_index=assembly_index,
        genome=genome,
        sgd_genes=runners._sgd_gene_set(root),
        gene_set=gene_set,
        gene_midpoints=gene_midpoints_from_genome(genome, genes),
        sourced=sourced_values(),
        release_sourced=list(RELEASE_SOURCED.values()),
    )


def main() -> None:
    """Build/load the dataset for interactive debugging."""
    data_root = _data_root()
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    dataset = Nguessan2025Dataset(
        root=osp.join(data_root, "data/torchcell/nguessan2025"), genome=genome
    )
    print(f"len = {len(dataset)}")


if __name__ == "__main__":
    main()
