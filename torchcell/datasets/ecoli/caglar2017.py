# torchcell/datasets/ecoli/caglar2017
# [[torchcell.datasets.ecoli.caglar2017]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/caglar2017
# Test file: tests/torchcell/datasets/ecoli/test_caglar2017.py
"""Caglar 2017 E. coli molecular phenotype: the raw mirror and the strain gate.

Caglar et al. 2017 (Scientific Reports, doi:10.1038/srep45303) measured mRNA (RNA-seq),
protein (LC-MS/MS) and 13C central-carbon flux ratios of one wild-type strain across 34
conditions: four carbon sources, Mg2+ and Na+ series, exponential and stationary phase,
and two starvation time courses. The processed data are the paper's Supplementary
Tables S1 (sample sheet), S2 (normalized mRNA), S3 (normalized protein) and S4 (flux
ratios).

**This module holds no loader, and that is the finding.** The strain is ``REL606``, an
*E. coli* **B** strain (``STRAIN``, ``LINEAGE``), and the paper's identifiers are REL606
identifiers: Table S2 is keyed by ``ECB_`` locus tags of GenBank CP000819.1 and Table S3
by retired RefSeq ``YP_`` protein accessions of NC_012967.1 (``IDENTIFIER_FORMS``). The
genomes tier holds K-12 MG1655, K-12 BW25113 and KT2440 only
(``BacterialReferenceStrain``), so no record of this paper can carry an honest
``AssemblyReferenceGenome``. Writing one against MG1655 would assert that a B-strain
locus is a K-12 b-number, which is the cross-strain inference plan D5/D9 refuse.
``require_pinnable_strain`` is the gate a loader calls first; it raises
``UnpinnedStrainError`` carrying ``STRAIN_GAP`` (a typed ``ProvenanceGap`` on
``genome_reference``) until ``REL606`` is in the tier's vocabulary.

``REL606_TIER_ADDITION`` records the exact addition that opens the gate: NCBI assembly
ASM1798v1 (``GCA_000017985.1`` / ``GCF_000017985.1``), its nine members with the md5
NCBI publishes and the sha256 measured on retrieval, the locus-tag pattern measured from
the GenBank flat file, and the schema, registry and genome edits it needs.

What IS done here: the raw mirror. ``deposit_raw_mirror`` writes
``$DATA_ROOT/torchcell-raw/caglarColiMolecularPhenotype2017/`` from bytes produced by
the recorded retrievers (``raw_file_specs``): Tables S1 to S4 from the PMC Article
Datasets bucket, and the NCBI protein records of every ``YP_`` accession in Table S3,
which are the only scriptable bridge from the protein table to ``ECB_`` locus tags (no
``YP_`` accession appears in the current GenBank or RefSeq annotation of the assembly).
``identifier_coverage`` re-measures all of that from the mirror and the assembly files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import shutil
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_args

from Bio import SeqIO
from pydantic import BaseModel, ConfigDict, Field

from torchcell.datamodels.schema import (
    BACTERIAL_LOCUS_TAG_PATTERNS,
    BacterialReferenceStrain,
)
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)
from torchcell.literature.provenance import run_retriever
from torchcell.literature.retrieve import pmc_cloud_url
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "caglarColiMolecularPhenotype2017"
PAPER_DOI = "10.1038/srep45303"
PAPER_TITLE = "The E. coli molecular phenotype under different growth conditions"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "0878d5e7d49bcea4570aa2db225318a8469f4755645f1ddf73563effe7b3109b"
SI1_MD = "si/si1.md"
SI1_MD_SHA256 = "1b7b8ed0f8b21c1909568f8a05d6a92d99bffaf851aa474a3a8673ebc2da4bdc"

#: The article version in the PMC Article Datasets bucket that serves the SI tables.
PMC_ARTICLE = "PMC5394689.1"
#: The date the raw-mirror bytes were produced by the recorded retrievers.
RAW_RETRIEVED_AT = "2026-10-07"

_OCR_METHOD = "MinerU OCR of the publisher PDF (torchcell-library mirror)"


def _paper(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote in the sha256-pinned paper OCR."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method=_OCR_METHOD,
            page=page,
        ),
    )


def _si(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """Bind a value to a verbatim quote in the OCR of the Supplementary Information PDF."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SI1_MD,
            citation_key=CITATION_KEY,
            sha256=SI1_MD_SHA256,
            method=_OCR_METHOD,
            page="List of Supplementary Tables",
        ),
    )


# --------------------------------------------------------------------------- #
# The strain, the reference the authors mapped to, and the identifier forms
# --------------------------------------------------------------------------- #
STRAIN = _paper(
    "REL606",
    "E. coli B REL606 was inoculated from a freezer stock",
    page="Methods, Cell Growth",
    note="every sample comes from this one strain: 'We grew multiple cultures of E. "
    "coli REL606, from the same stock' (Results) and 'used the exact same $E$ . coli "
    "genotype throughout' (Discussion)",
)
LINEAGE = _paper(
    "E. coli B",
    "the REL606 Escherichia coli B genome",
    page="Methods, RNA-seq",
    note="a B strain, not K-12: no deposited assembly set (MG1655, BW25113) is its genome",
)
SAME_STOCK = _paper(
    True,
    "We grew multiple cultures of E. coli REL606, from the same stock, under a variety "
    "of different growth conditions.",
    page="Results, Experimental design and data collection",
)
SAME_GENOTYPE = _paper(
    True, "used the exact same $E$ . coli genotype throughout", page="Discussion"
)
RNASEQ_REFERENCE = _paper(
    "NC_012967.1",
    "we implemented a custom analysis pipeline using the REL606 Escherichia coli B "
    "genome (GenBank:NC_012967.1) as the reference sequence",
    page="Methods, RNA-seq",
    note="NC_012967.1 is the RefSeq copy of GenBank CP000819.1, the one replicon of "
    "assembly GCA_000017985.1 / GCF_000017985.1 (ASM1798v1)",
)
PROTEOME_REFERENCE = _paper(
    "REL606 protein sequence database",
    "Spectra were searched against an $E _ { \\ast }$ . coli strain REL606 protein "
    "sequence database",
    page="Methods, Proteomics",
)
IDENTIFIER_FORMS = _si(
    {"mrna": "ECB_", "protein": "YP_"},
    "Gene id (ECB number for mRNA and YP number for proteins), and corresponding gene "
    "name",
    note="stated for Table S8; Tables S2 and S3 carry the same two forms in their first "
    "column (identifier_coverage measures 4,196 ECB_ and 4,196 YP_ ids)",
)
MATCHED_GENES = _paper(
    4196,
    "This resulted in 4196 matching mRNA and protein counts for each sample.",
    page="Methods, Normalization and quality control of RNA and protein counts",
)

# --------------------------------------------------------------------------- #
# Replicate structure (what a loader's n_samples and SampleUnit come from)
# --------------------------------------------------------------------------- #
BIOLOGICAL_REPLICATES = _paper(
    3,
    "For each experimental condition, bacteria were grown in three biological "
    "replicates.",
    page="Figure 1 legend",
    note="Table S2 and Table S3 are per SAMPLE (one column per biological replicate "
    "culture), not per-condition means",
)
REPLICATE_DAYS = _paper(
    "separate day",
    "Each of the three biological replicates was performed on a separate day.",
    page="Methods, Cell Growth",
)
TECHNICAL_REPLICATE_COLUMNS = _si(
    ("RNA_Data_Freq", "Protein_Data_Freq"),
    "number of RNA samples (technical replicates), number of protein samples "
    "(technical replicates)",
    note="the Table S1 columns that count technical replicates per sample; measured "
    "on the mirror: RNA 1 for all 152 RNA samples, protein 1 for 93 and 2 for 12 of "
    "the 105 protein samples",
)
FLUX_REPLICATES = _paper(
    3,
    "For each condition, flux samples were analyzed in triplicate (except one, which "
    "was analyzed in duplicate only), and 13 different flux ratios were measured for "
    "each sample.",
    page="Results, Metabolic flux ratios under salt stress",
    note="the lower end of the stated structure is 2; which condition had 2 is not "
    "named in the mirror (a gap for the flux record's n_samples)",
)
FLUX_AVERAGED = _paper(
    "mean over replicates",
    "The flux ratios were then averaged across replicates",
    page="Results, Metabolic flux ratios under salt stress",
)
FLUX_TECHNICAL_INJECTIONS = _paper(
    3, "three technical replicates of each vial", page="Methods, Flux analysis"
)
DOUBLING_TIME_REPLICATES = _paper(
    3,
    "Means and confidence intervals were calculated from three replicate growth "
    "curves for all conditions except for gluconate and lactate, which had "
    "measurements for only two replicates.",
    page="Methods, Cell Growth",
)

# --------------------------------------------------------------------------- #
# The data home and how the values were processed
# --------------------------------------------------------------------------- #
PROCESSED_TABLES = _paper(
    ("S2", "S3", "S4"),
    "final processed data are available as Supplementary\u00a0Tables\u00a0S2,\u00a0S3\u00a0"
    "and\u00a0S4.",
    page="Results, Experimental design and data collection",
    note="the OCR separates these words with no-break spaces (U+00A0), kept verbatim",
)
NORMALIZATION = _paper(
    "DESeq2 size factors, then log-transformed",
    "we normalized read counts using size-factors calculated via $\\mathrm { D E S e q "
    "} 2 ^ { 1 7 }$",
    page="Methods, Normalization and quality control of RNA and protein counts",
    note="'All resulting data sets were checked for quality, normalized, and "
    "log-transformed.' (Results). The base of the log is not stated in the mirror; the "
    "Methods defer to ref 10 (the glucose-starvation paper), which is not mirrored",
)
TABLE_S2 = _si(
    (4196, 152),
    "Supplementary Table S2: Normalized mRNA counts. Includes data for 4196 distinct "
    "proteins each for 152 samples.",
)
TABLE_S3 = _si(
    (4196, 105),
    "Supplementary Table S3: Normalized protein counts. Includes data for 4196 "
    "distinct proteins each for 105 samples.",
)
TABLE_S4 = _si(
    13,
    "Supplementary Table S4: Mean flux ratios for 13 branches each, measured for "
    "varying Mg $^ { 2 + }$ and Na $^ +$ concentrations in exponential and stationary "
    "phase.",
)
GEO_ACCESSION = _paper(
    "GSE94117",
    "accession GSE94117 for all other experiments",
    page="Methods, Statistical analysis and data availability",
)
PRIDE_ACCESSION = _paper(
    "PXD005721",
    "accession PXD005721 for all other experiments",
    page="Methods, Statistical analysis and data availability",
)

# --------------------------------------------------------------------------- #
# Environment (for the loader the gate is waiting on)
# --------------------------------------------------------------------------- #
BASE_MEDIUM = _paper(
    "DM500",
    "Davis Minimal medium supplemented with $2 \\mu \\mathrm { g } / 1$ thiamine $( "
    "\\mathrm { D M } ) ^ { 3 6 }$ and limiting glucose at $5 0 0 \\mathrm { m g / l }$ "
    "(DM500)",
    page="Methods, Cell Growth",
    note="MEDIA_LIBRARY key DM500 (base DAVIS_MINIMAL); the DM salts defer to ref 36, "
    "Lenski 1991",
)
CARBON_SOURCE_SWAP = _paper(
    0.5,
    "the Davis Minimal (DM) medium used was supplemented with $0 . 5 { \\mathrm { g } } "
    "/ { \\mathrm { L } }$ of the specified compound (glycerol, lactate, or gluconate) "
    "instead of glucose.",
    page="Methods, Cell Growth",
    note="g/L of the replacing carbon source",
)
MAGNESIUM_SERIES = _paper(
    0.83,
    "$\\mathbf { M } \\mathbf { g } ^ { 2 + }$ concentrations were varied by changing "
    "the amount of $\\mathrm { M g S O _ { 4 } }$ added to DM media from the concentration "
    "of $0 . 8 3 \\mathrm { m M }$ that is normally present.",
    page="Methods, Cell Growth",
    note="mM MgSO4 in DM; Table S1's Mg_mM column states 0.8 for the base level",
)
SODIUM_SERIES = _paper(
    5,
    "The base recipe for DM already contains ${ \\sim } 5 \\mathrm { m M N a ^ { + } }$ "
    "due to the inclusion of sodium citrate",
    page="Methods, Cell Growth",
    note="mM Na+ at base; NaCl is added to reach each higher level",
)


# --------------------------------------------------------------------------- #
# The strain gate
# --------------------------------------------------------------------------- #
class TierMember(BaseModel):
    """One file the REL606 assembly set would hold, as fetched and checked on 2026-10-07."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    url: str
    role: str = Field(description="genomes-tier role: annotation | sequence | index")
    bytes: int
    md5: str = Field(description="NCBI's md5checksums.txt value, matched on retrieval")
    sha256: str


class AssemblyTierAddition(BaseModel):
    """The genomes-tier and schema addition that would make REL606 pinnable.

    Every number here was measured on the bytes retrieved on ``retrieved_at``; the
    sha256 values are retrieval evidence, not a deposited manifest (a deposit re-fetches
    and ``deposit_assembly_set`` re-hashes).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    assembly_set: str = Field(
        description="proposed id, after ecoli_K12_MG1655_ASM584v2"
    )
    organism: str = Field(description="'# Organism name' of the assembly report")
    strain: str
    assembly_name: str
    genbank_accession: str
    refseq_accession: str
    genbank_replicon: str
    refseq_replicon: str
    replicon_length_bp: int
    gene_namespace: str = Field(description="proposed BacterialGeneNamespace member")
    locus_tag_pattern: str = Field(
        description="matches all GenBank gene features and no deposited namespace"
    )
    genbank_gene_features: int
    genbank_url: str
    refseq_url: str
    md5_checksum_files: dict[str, str] = Field(
        description="URL of each directory's md5checksums.txt -> sha256 of the copy used"
    )
    members: tuple[TierMember, ...]
    edits_needed: tuple[str, ...]
    retrieved_at: str


_NCBI = "https://ftp.ncbi.nlm.nih.gov/genomes/all"
_GCA_DIR = f"{_NCBI}/GCA/000/017/985/GCA_000017985.1_ASM1798v1"
_GCF_DIR = f"{_NCBI}/GCF/000/017/985/GCF_000017985.1_ASM1798v1"


def _member(
    directory: str, name: str, role: str, size: int, md5: str, sha: str
) -> TierMember:
    return TierMember(
        path=name, url=f"{directory}/{name}", role=role, bytes=size, md5=md5, sha256=sha
    )


REL606_TIER_ADDITION = AssemblyTierAddition(
    assembly_set="ecoli_B_REL606_ASM1798v1",
    organism="Escherichia coli B str. REL606 (E. coli)",
    strain="REL606",
    assembly_name="ASM1798v1",
    genbank_accession="GCA_000017985.1",
    refseq_accession="GCF_000017985.1",
    genbank_replicon="CP000819.1",
    refseq_replicon="NC_012967.1",
    replicon_length_bp=4629812,
    gene_namespace="ecoli_b_rel606_locus_tag",
    locus_tag_pattern=r"^ECB_[rt]?\d{5}$",
    genbank_gene_features=4383,
    genbank_url=f"{_GCA_DIR}/",
    refseq_url=f"{_GCF_DIR}/",
    md5_checksum_files={
        f"{_GCA_DIR}/md5checksums.txt": (
            "c7a6602a83d3e5573c46f8a0c8af1ff81716f686b6888cd162926afe485b578c"
        ),
        f"{_GCF_DIR}/md5checksums.txt": (
            "cff986bda5f799e5308ed7341c7af3fb74486fb163b3b486e604146b8e64eaf2"
        ),
    },
    members=(
        _member(
            _GCA_DIR,
            "GCA_000017985.1_ASM1798v1_genomic.gbff.gz",
            "annotation",
            3239604,
            "9c93575508d0eb2559106185330e483b",
            "aacf2559815f959c9417984ce1632228fd94caeac4b62b7910f714e310542e6b",
        ),
        _member(
            _GCA_DIR,
            "GCA_000017985.1_ASM1798v1_genomic.fna.gz",
            "sequence",
            1375449,
            "84ac547b7fa22c6291aa519f2d1ce444",
            "070a03fc2e2813853d5327608ee3ebcb4b0b2fe7faa239169921b3362b24adfa",
        ),
        _member(
            _GCA_DIR,
            "GCA_000017985.1_ASM1798v1_genomic.gff.gz",
            "annotation",
            270806,
            "673a2039153095efd66128386d2d9b80",
            "b928f83a99ea3ec7e64137f36490c37aa4585689de1abb884ff9cbaa4e1199d5",
        ),
        _member(
            _GCA_DIR,
            "GCA_000017985.1_ASM1798v1_protein.faa.gz",
            "sequence",
            889482,
            "732ed5bf0131047db14f5c019a8286db",
            "40f1748bf2e86f5a43d7a8bb1515a0b3812f66f27f4d7fb9dc62a0c348962663",
        ),
        _member(
            _GCA_DIR,
            "GCA_000017985.1_ASM1798v1_feature_table.txt.gz",
            "index",
            173353,
            "52ce70f6a9e672ea4f044474c24fc8e2",
            "5cba47c018f4a5180eb1af9f06e4b9103837f894a08f05fb5f6f70e8795379ec",
        ),
        _member(
            _GCA_DIR,
            "GCA_000017985.1_ASM1798v1_assembly_report.txt",
            "index",
            1172,
            "e89ce0ff1c31066a8639ec94e40e66a2",
            "51968f440a6497669ad8ccf703c437d5a8055990d2c7e27194b9cc1ffeeda369",
        ),
        _member(
            _GCF_DIR,
            "GCF_000017985.1_ASM1798v1_genomic.gbff.gz",
            "annotation",
            3428153,
            "15b4bbd969d274c99919bcd5d57d4d87",
            "b90a8ab7a8f1e9e736952b6e17017a9cd6bc6567cb11b1ecdc2e7c895695a26b",
        ),
        _member(
            _GCF_DIR,
            "GCF_000017985.1_ASM1798v1_genomic.gff.gz",
            "annotation",
            433859,
            "9a52f090f2e985a44e5e6752018b0b23",
            "27c302a37ac517de79999cc8438c744367e5c12ad4ac60b34f55bfd753214f25",
        ),
        _member(
            _GCF_DIR,
            "GCF_000017985.1_ASM1798v1_gene_ontology.gaf.gz",
            "annotation",
            158466,
            "42ac1448a06841ec7d7e86aab3f916e0",
            "4cbd6f5767d0f8651346891af174eaf9fd3353c25b6916fdee1f3d6c399374b1",
        ),
    ),
    edits_needed=(
        "torchcell/sequence/genome/registry.py: ECOLI_B_REL606 = "
        "'ecoli_B_REL606_ASM1798v1'",
        "scripts/provision_bacterial_genomes.py: a REL606 set of the nine members "
        "above (the GCF directory lists a _gene_ontology.gaf.gz, so it is a member), "
        "fetched by direct_url, md5-checked, deposited by deposit_assembly_set",
        "torchcell/datamodels/schema.py: 'REL606' in BacterialReferenceStrain; "
        "BACTERIAL_ASSEMBLY_SETS['REL606']; the set id in BacterialAssemblySet; "
        "ASSEMBLY_SET_ACCESSIONS[set] = ('GCA_000017985.1', 'GCF_000017985.1'); "
        "'ecoli_b_rel606_locus_tag' in BacterialGeneNamespace with pattern "
        "^ECB_[rt]?\\d{5}$ in BACTERIAL_LOCUS_TAG_PATTERNS (disjoint from the three "
        "existing patterns and from yeast systematic names)",
        "torchcell/sequence/genome/ecoli/: an E. coli B REL606 genome class beside "
        "EcoliK12Genome (the K12 classes and the ECK crosswalk do not apply to a B "
        "strain)",
        "torchcell/datasets/bacteria_common.py: REL606 in HOST_STRAINS['ecoli'], "
        "STRAIN_GENE_NAMESPACES and BACTERIAL_GENOME_CLASSES",
        "torchcell/verification/runners.py: a REL606 gene universe beside "
        "_ecoli_k12_gene_set (4,383 GenBank gene features)",
    ),
    retrieved_at="2026-10-07",
)

STRAIN_GAP = ProvenanceGap(
    field="genome_reference",
    reason=ProvenanceGapReason.deferred_pending_source_review,
    looked_in=Provenance(
        source_uri=PAPER_MD,
        citation_key=CITATION_KEY,
        sha256=PAPER_MD_SHA256,
        method=_OCR_METHOD,
        page="Methods, Cell Growth and RNA-seq",
    ),
    resolve_with=Provenance(
        source_uri=REL606_TIER_ADDITION.genbank_url,
        sha256=REL606_TIER_ADDITION.members[0].sha256,
        method="deposit NCBI assembly GCA_000017985.1 / GCF_000017985.1 (ASM1798v1) "
        "into the genomes tier as ecoli_B_REL606_ASM1798v1",
        page="GCA_000017985.1_ASM1798v1_genomic.gbff.gz",
        retrieved=REL606_TIER_ADDITION.retrieved_at,
    ),
    note="The paper's strain is E. coli B REL606 and its identifiers are REL606 ECB_ "
    "locus tags and NC_012967.1 YP_ proteins; the genomes tier holds only K-12 MG1655, "
    "K-12 BW25113 and KT2440, so no assembly pin is honest. The paper reports the "
    "genome; the gap is the tier's, recoverable by the deposit named in resolve_with.",
)


class StrainPinFinding(BaseModel):
    """Whether this paper's strain can be pinned to a deposited assembly set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strain: str
    lineage: str
    tier_strains: tuple[str, ...]
    pinnable: bool
    gap: ProvenanceGap | None
    tier_addition: AssemblyTierAddition | None


class UnpinnedStrainError(RuntimeError):
    """The paper's strain has no assembly set in the genomes tier."""

    def __init__(self, finding: StrainPinFinding) -> None:
        """Carry the finding (its ``gap`` and ``tier_addition``) on the error."""
        self.finding = finding
        super().__init__(
            f"{CITATION_KEY}: strain {finding.strain} ({finding.lineage}) is not one of "
            f"the tier's strains {list(finding.tier_strains)}; "
            f"{finding.gap.note if finding.gap is not None else ''}"
        )


def strain_pin_finding() -> StrainPinFinding:
    """The paper's strain against the schema's ``BacterialReferenceStrain`` vocabulary."""
    tier_strains: tuple[str, ...] = get_args(BacterialReferenceStrain)
    pinnable = STRAIN.value in tier_strains
    return StrainPinFinding(
        strain=STRAIN.value,
        lineage=LINEAGE.value,
        tier_strains=tier_strains,
        pinnable=pinnable,
        gap=None if pinnable else STRAIN_GAP,
        tier_addition=None if pinnable else REL606_TIER_ADDITION,
    )


def require_pinnable_strain() -> StrainPinFinding:
    """The gate a Caglar 2017 loader calls before building any record."""
    finding = strain_pin_finding()
    if not finding.pinnable:
        raise UnpinnedStrainError(finding)
    return finding


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
class RawFileSpec(BaseModel):
    """One raw-mirror file: where it lives, its pin, and the retriever that made it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relpath: str
    sha256: str
    description: str
    retrieval: RetrievalRecord


#: Supplementary table -> (PMC object, sha256, what it is). The SI lists the tables as
#: S1 to S14 after the SI PDF, so PMC object ``-s<N+1>`` is Table S<N>; the content
#: agrees (the S2 and S3 shapes equal the SI's 4196 x 152 and 4196 x 105).
SI_TABLES: dict[str, tuple[str, str, str]] = {
    "S1": (
        "srep45303-s2.csv",
        "1486290bf6a340ae64ee20c915435c0a00ff5eede489de1f56b62733b66f8940",
        "Supplementary Table S1 (tableS1_meta_data.csv): one row per sample with "
        "carbon source, Mg2+ and Na+ levels, growth phase, batch, technical-replicate "
        "counts and doubling time",
    ),
    "S2": (
        "srep45303-s3.csv",
        "df3e28237ea8e03c68ec93c577be1b1032ccc2d1f60372f45a72a6107de94d95",
        "Supplementary Table S2 (tableS2_mRNA_normalized_raw_data.csv): normalized, "
        "log-transformed mRNA counts, 4196 ECB_ genes x 152 samples",
    ),
    "S3": (
        "srep45303-s4.csv",
        "a391afb5784edebf2e44c52d622b868eaf7096258658f7473640090fb1429b11",
        "Supplementary Table S3 (tableS3_protein_normalized_raw_data.csv): normalized, "
        "log-transformed protein counts, 4196 YP_ proteins x 105 samples",
    ),
    "S4": (
        "srep45303-s5.csv",
        "eb9e6746dc7813327e463ad1ee5fc5f6eed17495bf59ef3d2ff9175c38ce5073",
        "Supplementary Table S4 (tableS4_fluxData.csv): mean and SDE of 13 branch flux "
        "ratios per salt, concentration and phase",
    ),
}

#: NCBI E-utilities batch size for the YP_ protein records (GET, one URL per batch).
#: Measured 2026-10-07: a 200-record batch took about 85 to 95 s against the 120 s
#: timeout of the direct_url retriever; the 42 batches of 100 took 38 to 85 s each.
YP_BATCH_SIZE = 100
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

#: sha256 of each GenPept batch of Table S3's YP_ accessions, in table order, as
#: retrieved on RAW_RETRIEVED_AT. A second retrieval of batches 0 and 41 was
#: byte-identical, so the URL is a reproducible retrieval, not a moving one.
YP_BATCH_SHA256: tuple[str, ...] = (
    "2cf634697c26238d0421d87458d93d1f683bfa2ee3f0aeb1d49de49e5e464d53",
    "a8aee1572d8096584b2d4647065ddef8d21b7e5bfe40495eaa83e26b2da96780",
    "f36c65eb33d9da5ef5bc673835ff37bd6f6f1f4f3a709514b1f3c0de72d669ac",
    "2edd91e904d97d58db4fb68b32526e21355d880d1cb4bbc2dfd24a01e0eaea9d",
    "03c311f5e4cfaa06fd13877f72ef1f793cdf84163b1962ee01d16b56a5cf775e",
    "e714cfcc2446c8c0fe7272b6349bbb1b8f2d1a87673c136bf71eba3672476c1d",
    "f8798c0725438cf41661fe160ba4f609e166fc2ad191f462ab511a3022174d08",
    "0938f934992518750d84a99aa6ca8f4aff3b011916d62d9108da9102d035997f",
    "7a4b54840a35d5e1e7a800daa839cf1a9a3ed02e80abea8e8c2f0eff5dc9677a",
    "f66c0601555b7f8b19d289ee8ab1a9381c51f6f18705de81365a8d49b37e92d5",
    "8bb28eaaebb6cfa47fbb47fdf528c70ea396e6d6a6be0d890f6ca92bc2266982",
    "a264ac610977f0ad381dea2df3dabfcd39cc1623bdd488f55f252004728d3469",
    "f3ab3016232436a0f95f92d7fbfdf701275fc8f72c19f77797126f27a7fe3d6d",
    "2f44565ad628f1c32bcc8b00ee6e71ad6cc55c2ad119ad480073d0e38b8e640b",
    "ae360be5e5f27e4601bae44539e09ddd62c363bd195511a872428b8413c7f2f9",
    "540e05de73de93eefb8155bcc5e94dbb098fbaa2163ac1b6dbf330a99c2a4a05",
    "934b9ed33d6171484469751b9b565041ad76dfe1fd3a85ba9f6e0dede3938184",
    "89b7a382b9f0775fddff175b7431670f13786cae156bd117f2338564d9edce74",
    "641faca533aeb55b6c8200aacaad0bc5d0addf95c11ebfd6c0b974bbf22ba979",
    "863854f5fea4f3f85f0957edb9d367f4dbbd507f17899207fcda56a447558ca3",
    "e8880562a2e1f37d6fb1baf3b4e78d1cc7558afb7652e2ca410a36e14bed4b6c",
    "31b60b1cc3e6a28306de274f3be9d62e991029ce7677fd545d69c3fbe07d552b",
    "7c8295750ed805a32922779007ae74190fe0bcd282448108f59be2bad2e3c985",
    "3b07b8b94e6866e84d2caadc3050ded36ba2f38339f4f34acd63a2a96d83b583",
    "acf357b3f0b39dc8b95846d3431a207f2d2077d32377e3281c2fc23e10b15dd3",
    "18356152fbfb36f763e344a274f52b575365be569898c3158e4d519889c51a0f",
    "77a9e511f2349b56f540dd5dc7c69c17780cdd18074d98711294fe0e7b9f474b",
    "bc38b788f3a35dc0985f5b2fe54219079fdb92c29784a3761f1d537d700f3b51",
    "6989074acd7a54d0d042baf87c7b3afe41cc84df94ba4e4f947b74d5c8aed2de",
    "ea2ab9083ddcd85309f310a88800a771af1e9ad97603e780dd89774cdb272a31",
    "f3f2529499a0968d529ef47986896ecc6eeb8e212eaacc7cc1a88c9e8ae6545b",
    "e7eef3e8baf471f5b22b2c730885bb85fcf0e2fe969aad6a8fb262a2c68d0503",
    "2bd3ba9416284f601af94d32bc2c1d25c693f9ecfa9d583c62b0ec9d9144a025",
    "3f12fec04dc067965f8cb5739e9c8a30dd811899d807b8b22f79fbf442a1a2d9",
    "5539f56f18f93cd545cbe6a4a230f0f7ed17836feaf44e0102cc7511c9a1f33d",
    "0f9bfb72c65d2989526cb69c02b336830cffa92469fe1490f616a9f8f781bd58",
    "f1cecb2f33fabe75f61a879e4b42af99c40803df099cbdfcb16e7a9da391965d",
    "1e1c85455aa8bb5ba1d450d73e32d33071f98a2741d754c113c5818b882d0707",
    "4cb163db9981307e9c6c7c0201d98ba46028253a37c8667aa8720484f1e0e2e6",
    "eb033a796b24f0e0ec9094cba8f08bd4a09b69519ae9949a5c25f2f02f6a48ad",
    "e6965153cd89ed5e2a866f80ce3dd7f74aae248611f3a53c1bc02beeef02dc94",
    "96281a84a76b69243e0cfb5c75572f7f76fb731bb3fe1f9ab22c74ff9f6aa53b",
)


def si_table_relpath(table: str) -> str:
    """Mirror path of one supplementary table."""
    return f"data/{SI_TABLES[table][0]}"


def yp_batch_relpath(index: int) -> str:
    """Mirror path of one GenPept batch."""
    return f"ncbi_protein/yp_batch_{index:02d}.gp"


def yp_batch_url(accessions: Iterable[str]) -> str:
    """The efetch GET URL returning the GenPept records of ``accessions``."""
    return f"{EFETCH}?db=protein&rettype=gp&retmode=text&id={','.join(accessions)}"


def _si_table_spec(table: str) -> RawFileSpec:
    obj, sha, description = SI_TABLES[table]
    key = f"{PMC_ARTICLE}/{obj}"
    return RawFileSpec(
        relpath=si_table_relpath(table),
        sha256=sha,
        description=description,
        retrieval=RetrievalRecord(
            method=RetrievalMethod.pmc_cloud,
            source_url=pmc_cloud_url(key),
            retriever="torchcell.literature.retrieve.pmc_cloud_object",
            params={"key": key},
            sha256=sha,
            retrieved_at=RAW_RETRIEVED_AT,
        ),
    )


def si_table_specs() -> list[RawFileSpec]:
    """The four supplementary tables, in table order."""
    return [_si_table_spec(table) for table in SI_TABLES]


def yp_batch_specs(protein_ids: list[str]) -> list[RawFileSpec]:
    """One GenPept batch per ``YP_BATCH_SIZE`` accessions of Table S3, in table order.

    The URLs are a function of Table S3's first column, which is pinned, so the specs
    are reproducible from the mirror; the count must equal the pinned batch count.
    """
    batches = [
        protein_ids[start : start + YP_BATCH_SIZE]
        for start in range(0, len(protein_ids), YP_BATCH_SIZE)
    ]
    if len(batches) != len(YP_BATCH_SHA256):
        raise ValueError(
            f"{len(protein_ids)} protein ids make {len(batches)} batches; "
            f"{len(YP_BATCH_SHA256)} are pinned"
        )
    specs = []
    for index, (batch, sha) in enumerate(zip(batches, YP_BATCH_SHA256, strict=True)):
        url = yp_batch_url(batch)
        specs.append(
            RawFileSpec(
                relpath=yp_batch_relpath(index),
                sha256=sha,
                description=(
                    f"NCBI GenPept records of Table S3 YP_ accessions "
                    f"{batch[0]}..{batch[-1]} ({len(batch)}); each record names its "
                    "REL606 /locus_tag"
                ),
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.direct_url,
                    source_url=url,
                    retriever="torchcell.literature.retrieve.direct_url",
                    params={"url": url},
                    sha256=sha,
                    retrieved_at=RAW_RETRIEVED_AT,
                ),
            )
        )
    return specs


def read_table_ids(path: str | Path) -> list[str]:
    """First column of Table S2 or S3 (the gene or protein identifier), in file order."""
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        return [row[0] for row in reader]


def raw_file_specs(protein_ids: list[str]) -> list[RawFileSpec]:
    """Every raw-mirror file, in the order the manifest lists them."""
    return si_table_specs() + yp_batch_specs(protein_ids)


def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror lives under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/caglarColiMolecularPhenotype2017``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def retrieve_raw_files(staging_dir: str | Path) -> Path:
    """Run every recorded retriever into ``staging_dir`` and check each pin.

    The tables come first because the protein batches are built from Table S3's ids.
    A file already staged with its pinned sha256 is not fetched again; bytes that do not
    match a pin raise (upstream drift is reported, never followed).
    """
    staging = Path(staging_dir)
    for spec in si_table_specs():
        _retrieve_one(spec, staging)
    protein_ids = read_table_ids(staging / si_table_relpath("S3"))
    for spec in yp_batch_specs(protein_ids):
        _retrieve_one(spec, staging)
    return staging


def _retrieve_one(spec: RawFileSpec, staging: Path) -> None:
    dest = staging / spec.relpath
    if dest.exists() and sha256_file(dest) == spec.sha256:
        return
    body = run_retriever(spec.retrieval)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    got = sha256_file(dest)
    if got != spec.sha256:
        raise RuntimeError(
            f"{spec.relpath}: retrieved sha256 {got} differs from the pin {spec.sha256}"
        )


def raw_manifest(specs: list[RawFileSpec], sizes: dict[str, int]) -> Manifest:
    """The raw mirror's ``manifest.json`` content for ``specs`` (``created_at`` unset)."""
    return Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=PAPER_TITLE,
        files=[
            ArtifactRecord(
                path=spec.relpath,
                role=ROLE_RAW_DATA,
                bytes=sizes[spec.relpath],
                sha256=spec.sha256,
                source=spec.retrieval.source_url,
                retrieval=spec.retrieval,
            )
            for spec in specs
        ],
        si_data_sources=[
            f"https://pmc-oa-opendata.s3.amazonaws.com/{PMC_ARTICLE}/",
            EFETCH,
        ],
        si_expected=[
            "GEO GSE94117 (raw reads, per-gene read counts) -- not mirrored; Table S2 is "
            "the processed matrix the paper names",
            "PRIDE PXD005721 (raw spectra) -- not mirrored; Table S3 is the processed "
            "matrix the paper names",
            "Texas Data Repository doi:10.18738/T8/UG3TUR (raw GC-MS) -- not mirrored; "
            "Table S4 is the processed flux-ratio table",
        ],
        provenance_complete=True,
    )


def deposit_raw_mirror(*, source_dir: str | Path, data_root: str | None = None) -> Path:
    """Write the raw mirror from retrieved files (``retrieve_raw_files``) + its manifest.

    Every staged file, every existing mirror file and an existing manifest are checked
    before anything is written. Idempotent by sha256: a mirror file already holding its
    pinned bytes is left alone, and one holding other bytes raises rather than being
    overwritten. An existing ``manifest.json`` whose file records equal the new ones is
    left untouched (its ``created_at`` is the first deposit's); one that differs raises.
    """
    source = Path(source_dir)
    root = raw_mirror_dir(data_root)
    specs = raw_file_specs(read_table_ids(source / si_table_relpath("S3")))
    for spec in specs:
        staged = source / spec.relpath
        got = sha256_file(staged)
        if got != spec.sha256:
            raise RuntimeError(f"{staged}: sha256 {got}, pinned {spec.sha256}")
        dest = root / spec.relpath
        if dest.exists() and sha256_file(dest) != spec.sha256:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    manifest = raw_manifest(
        specs, {spec.relpath: (source / spec.relpath).stat().st_size for spec in specs}
    )
    path = root / "manifest.json"
    existing = Manifest.model_validate_json(path.read_text()) if path.exists() else None
    if existing is not None and existing.model_dump(
        exclude={"created_at"}
    ) != manifest.model_dump(exclude={"created_at"}):
        raise RuntimeError(f"{path} records other files; refusing to overwrite")
    for spec in specs:
        dest = root / spec.relpath
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / spec.relpath, dest)
    if existing is None:
        manifest.created_at = datetime.now(UTC).isoformat()
        path.write_text(manifest.model_dump_json(indent=2))
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
# Identifier coverage (the measurement behind the strain finding)
# --------------------------------------------------------------------------- #
class GenBankIdentifiers(BaseModel):
    """The identifiers one GenBank flat file carries."""

    model_config = ConfigDict(extra="forbid")

    locus_tags: set[str]
    old_locus_tags: set[str]
    protein_ids: set[str]


def genbank_identifiers(gbff_gz: str | Path) -> GenBankIdentifiers:
    """Locus tags (gene features), ``old_locus_tag`` values and CDS ``protein_id``s."""
    locus_tags: set[str] = set()
    old_locus_tags: set[str] = set()
    protein_ids: set[str] = set()
    with gzip.open(gbff_gz, "rt") as handle:
        for record in SeqIO.parse(handle, "genbank"):  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
            for feature in record.features:
                if feature.type == "gene":
                    locus_tags.update(feature.qualifiers["locus_tag"])
                    old_locus_tags.update(feature.qualifiers.get("old_locus_tag", []))
                elif feature.type == "CDS":
                    protein_ids.update(feature.qualifiers.get("protein_id", []))
    return GenBankIdentifiers(
        locus_tags=locus_tags, old_locus_tags=old_locus_tags, protein_ids=protein_ids
    )


def genpept_locus_tags(path: str | Path) -> dict[str, str]:
    """``YP_`` accession.version -> the ``/locus_tag`` of its one CDS, from a GenPept file.

    A record with no CDS, or with a CDS naming no locus tag or several, is refused.
    """
    out: dict[str, str] = {}
    with open(path) as handle:
        for record in SeqIO.parse(handle, "genbank"):  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
            tags = [
                feature.qualifiers.get("locus_tag", [])
                for feature in record.features
                if feature.type == "CDS"
            ]
            if len(tags) != 1 or len(tags[0]) != 1:
                raise ValueError(f"{record.id}: CDS locus tags {tags}, expected one")
            if record.id in out:
                raise ValueError(f"{record.id} appears twice")
            out[record.id] = tags[0][0]
    return out


class IdentifierCoverage(BaseModel):
    """How the paper's identifiers land on REL606 and on the deposited namespaces."""

    model_config = ConfigDict(extra="forbid")

    mrna_ids: int
    mrna_ecb: int
    mrna_in_genbank_locus_tags: int
    mrna_in_refseq_old_locus_tags: int
    protein_ids: int
    protein_yp: int
    protein_in_genbank_protein_ids: int
    protein_in_refseq_protein_ids: int
    protein_resolved_by_ncbi_record: int
    protein_locus_tag_in_genbank: int
    protein_row_aligned_with_mrna: int
    deposited_namespace_matches: dict[str, int] = Field(
        description="paper identifiers matching each deposited namespace's pattern"
    )
    rel606_pattern_matches: int


def identifier_coverage(
    raw_root: str | Path, genbank_gbff: str | Path, refseq_gbff: str | Path
) -> IdentifierCoverage:
    """Measure the paper's identifiers against the REL606 annotation and the tier.

    ``raw_root`` is the raw mirror (or a staging directory with the same layout);
    the two flat files are the GCA and GCF ``_genomic.gbff.gz`` of ASM1798v1.
    """
    root = Path(raw_root)
    mrna = read_table_ids(root / si_table_relpath("S2"))
    protein = read_table_ids(root / si_table_relpath("S3"))
    genbank = genbank_identifiers(genbank_gbff)
    refseq = genbank_identifiers(refseq_gbff)
    yp_to_tag: dict[str, str] = {}
    for spec in yp_batch_specs(protein):
        for accession, tag in genpept_locus_tags(root / spec.relpath).items():
            if accession in yp_to_tag:
                raise ValueError(f"{accession} appears in two batches")
            yp_to_tag[accession] = tag
    rel606 = re.compile(REL606_TIER_ADDITION.locus_tag_pattern)
    every_id = mrna + protein
    return IdentifierCoverage(
        mrna_ids=len(mrna),
        mrna_ecb=sum(name.startswith("ECB_") for name in mrna),
        mrna_in_genbank_locus_tags=sum(name in genbank.locus_tags for name in mrna),
        mrna_in_refseq_old_locus_tags=sum(
            name in refseq.old_locus_tags for name in mrna
        ),
        protein_ids=len(protein),
        protein_yp=sum(name.startswith("YP_") for name in protein),
        protein_in_genbank_protein_ids=sum(
            name in genbank.protein_ids for name in protein
        ),
        protein_in_refseq_protein_ids=sum(
            name in refseq.protein_ids for name in protein
        ),
        protein_resolved_by_ncbi_record=sum(name in yp_to_tag for name in protein),
        protein_locus_tag_in_genbank=sum(
            yp_to_tag[name] in genbank.locus_tags
            for name in protein
            if name in yp_to_tag
        ),
        protein_row_aligned_with_mrna=sum(
            yp_to_tag.get(p) == m for m, p in zip(mrna, protein, strict=True)
        ),
        deposited_namespace_matches={
            namespace: sum(bool(re.match(pattern, name)) for name in every_id)
            for namespace, pattern in BACTERIAL_LOCUS_TAG_PATTERNS.items()
        },
        rel606_pattern_matches=sum(bool(rel606.match(name)) for name in mrna),
    )


class AnnotationSummary(BaseModel):
    """The REL606 annotation facts ``REL606_TIER_ADDITION`` states, re-measured."""

    model_config = ConfigDict(extra="forbid")

    genbank_gene_features: int
    genbank_tag_prefixes: dict[str, int] = Field(
        description="locus-tag prefix (the tag less its trailing digits) -> genes"
    )
    genbank_pseudogenes: int
    genbank_pattern_matches: int
    refseq_gene_features: int
    refseq_tag_prefixes: dict[str, int]
    refseq_genes_with_old_locus_tag: int
    gaf_rows: int
    gaf_objects: int
    gaf_evidence: dict[str, int]


def _gene_qualifiers(gbff_gz: str | Path) -> list[dict[str, list[str]]]:
    with gzip.open(gbff_gz, "rt") as handle:
        return [
            dict(feature.qualifiers)
            for record in SeqIO.parse(handle, "genbank")  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
            for feature in record.features
            if feature.type == "gene"
        ]


def _prefixes(tags: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tag in tags:
        prefix = re.sub(r"\d+$", "", tag)
        counts[prefix] = counts.get(prefix, 0) + 1
    return dict(sorted(counts.items()))


def annotation_summary(
    genbank_gbff: str | Path, refseq_gbff: str | Path, refseq_gaf: str | Path
) -> AnnotationSummary:
    """Gene features, tag forms and GO rows of the ASM1798v1 GenBank and RefSeq files."""
    genbank = _gene_qualifiers(genbank_gbff)
    refseq = _gene_qualifiers(refseq_gbff)
    genbank_tags = [q["locus_tag"][0] for q in genbank]
    pattern = re.compile(REL606_TIER_ADDITION.locus_tag_pattern)
    objects: set[str] = set()
    evidence: dict[str, int] = {}
    rows = 0
    with gzip.open(refseq_gaf, "rt") as handle:
        for line in handle:
            if line.startswith("!"):
                continue
            columns = line.rstrip("\n").split("\t")
            rows += 1
            objects.add(columns[1])
            evidence[columns[6]] = evidence.get(columns[6], 0) + 1
    return AnnotationSummary(
        genbank_gene_features=len(genbank),
        genbank_tag_prefixes=_prefixes(genbank_tags),
        genbank_pseudogenes=sum("pseudo" in q for q in genbank),
        genbank_pattern_matches=sum(bool(pattern.match(tag)) for tag in genbank_tags),
        refseq_gene_features=len(refseq),
        refseq_tag_prefixes=_prefixes(q["locus_tag"][0] for q in refseq),
        refseq_genes_with_old_locus_tag=sum("old_locus_tag" in q for q in refseq),
        gaf_rows=rows,
        gaf_objects=len(objects),
        gaf_evidence=dict(sorted(evidence.items())),
    )


def main() -> None:
    """Retrieve and deposit the raw mirror, or measure identifier coverage."""
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    deposit = sub.add_parser("deposit", help="retrieve into --staging, then deposit")
    deposit.add_argument("--staging", required=True)
    measure = sub.add_parser("measure", help="identifier coverage on the raw mirror")
    measure.add_argument("--genbank-gbff", required=True)
    measure.add_argument("--refseq-gbff", required=True)
    measure.add_argument("--refseq-gaf", required=True)
    args = parser.parse_args()
    if args.command == "deposit":
        print(deposit_raw_mirror(source_dir=retrieve_raw_files(args.staging)))
    else:
        coverage = identifier_coverage(
            raw_mirror_dir(), args.genbank_gbff, args.refseq_gbff
        )
        print(coverage.model_dump_json(indent=2))
        summary = annotation_summary(
            args.genbank_gbff, args.refseq_gbff, args.refseq_gaf
        )
        print(summary.model_dump_json(indent=2))
    print(
        strain_pin_finding().model_dump_json(indent=2, include={"strain", "pinnable"})
    )


if __name__ == "__main__":
    main()
