# torchcell/datasets/ecoli/zhang2015
# [[torchcell.datasets.ecoli.zhang2015]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/zhang2015
# Test file: tests/torchcell/datasets/ecoli/test_zhang2015.py
"""CeCaFDB 2015: a provenance record of a flux compendium this ontology does not admit.

Zhang et al. 2015 (Nucleic Acids Res. 43:D549-D557, doi:10.1093/nar/gku1137, PMC4383945,
citation key ``zhangCeCaFDBCuratedDatabase2015``) is row 54 of the bacterial schedule,
``status="aggregation"``: a manually curated database of 581 published 13C-MFA flux
distributions over 36 organisms, of which E. coli is 297 cases and P. putida 3.

DECISION: NOT LOADED. This module registers no dataset. It deposits and pins the
release, measures it, and records why, in the MCF2Chem 2023 pattern (``cai2023.py``).

**What the release is, measured** (:func:`release_inventory` on the deposited bytes).
Unlike MCF2Chem, CeCaFDB DOES release a per-record artifact: the Download page
(:data:`DOWNLOAD_INDEX`) links one Excel workbook per source reference, and every
workbook names that reference in its ``Experiment Name (ID)`` cell. For E. coli and
P. putida that is 33 workbooks, 31 + 2 references, 297 + 3 cases, all deposited under
``$DATA_ROOT/torchcell-raw/zhangCeCaFDBCuratedDatabase2015/`` with sha256 and a
re-runnable retrieval. Attribution per record is therefore available; it is not what
blocks the row.

**Why it is refused anyway.**

1. **No interval, anywhere.** ``FluxPhenotype`` exists to carry a fitted flux WITH the
   fit's interval (``net_flux_lower``/``net_flux_upper``/``confidence_level``). Measured:
   0 of 33 workbooks carry any column beyond the per-case value columns, and none holds
   an uncertainty term; the paper's description of a case's data fields
   (:data:`DATA_FIELDS`) names no uncertainty and the paper uses none of "standard
   deviation", "confidence" or "uncertainty". A stored map would be a point estimate
   whose interval is a typed gap on every one of the 300 records.
2. **The values are re-typed, not re-measured.** Every value is in one unit,
   ``relative flux``, renormalized by the curators to substrate uptake = 100
   (:data:`RELATIVE_FLUX`), and each source's lumped reactions were split onto KEGG
   reactions by the curators (:data:`LUMPED_SPLIT`). The two landed aggregation loaders
   are admissible because the aggregator recomputed from raw data (Lim 2022, Borchert
   2024); CeCaFDB, like MCF2Chem, transcribes. A value here is the curators' rendering
   of a source figure or table, and the source itself is the measurement to load.
3. **The genotype layer is corrupted where it matters most.** 190 of the 297 E. coli
   cases are Haverkorn van Rijsewijk 2011, whose ``Strains`` row runs ``BW25113``,
   ``BW25114``, ... ``BW25302``: one consecutive integer run of 190 labels, while the
   same workbook's ``Genotype`` row gives every case the BW25113 genotype string with a
   regulator name appended. The strain label therefore does not name the strain its
   own genotype row describes, and the deletion is recoverable only from free text. That paper is its own schedule row (69, "Haverkorn van
   Rijsewijk 2011", Supplementary Tables 2 and 3), so loading it through CeCaFDB would
   both duplicate row 69 and inherit the corruption.
4. **References are not always in the file.** ``FluxExperimentReference`` requires the
   parent's map (the reason Li 2021, #800, was refused). Five E. coli workbooks and one
   P. putida workbook hold a single case, so no reference can be built from the release.

**Duplication, measured.** The 33 source DOIs (resolved through PubMed, pinned in
:data:`WORKBOOKS`) intersect the served loaders in zero DOIs and the bacteria candidate
table in one (Haverkorn van Rijsewijk 2011, row 69). Ishii 2007, the one served E. coli
flux map, is not among CeCaFDB's references.

Script and committed results:
``experiments/036-dataset-fixes-before-kg-build/scripts/cecafdb2015_release_inventory.py``.
Finding: [[torchcell.datasets.ecoli.zhang2015]].
"""

from __future__ import annotations

import hashlib
import html
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import xlrd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from torchcell.literature.manifest import (
    ROLE_PAPER_PDF,
    ROLE_PAPER_TEXT,
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
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
CITATION_KEY: Final = "zhangCeCaFDBCuratedDatabase2015"
PAPER_DOI: Final = "10.1093/nar/gku1137"
PAPER_PMCID: Final = "PMC4383945"
PAPER_TITLE: Final = (
    "CeCaFDB: a curated database for the documentation, visualization and comparative "
    "analysis of central carbon metabolic flux distributions explored by 13C-fluxomics"
)
RAW_DIR_REL: Final = f"torchcell-raw/{CITATION_KEY}"
RETRIEVED_AT: Final = "2026-10-10"

#: The accession the paper names; it answers over plain http. Over https the
#: certificate does not verify, and with verification off the host serves a different
#: application (CeCaFLUX); the inventory script's ``probe`` records both.
ACCESSION_URL: Final = "http://www.cecafdb.org"
DOWNLOAD_PAGE_URL: Final = f"{ACCESSION_URL}/download_action"
WORKBOOK_URL_PREFIX: Final = f"{ACCESSION_URL}/download_files/"

PMC_PREFIX: Final = f"{PAPER_PMCID}.1"
PAPER_TEXT_NAME: Final = f"{PMC_PREFIX}.txt"
PAPER_TEXT_REL: Final = f"paper/{PAPER_TEXT_NAME}"
PAPER_TEXT_SHA256: Final = (
    "c6bd5a739bd773c0e22c0ed14174310add54d19e384ccdb525651ca0a924a88d"
)
PAPER_PDF_NAME: Final = f"{PMC_PREFIX}.pdf"
PAPER_PDF_REL: Final = f"paper/{PAPER_PDF_NAME}"
PAPER_PDF_SHA256: Final = (
    "e476a508e0adad99ae0b6952c4a557072e18e3fcf72b2373645c964731b4ead5"
)
DOWNLOAD_INDEX_REL: Final = "data/download_action.html"
DOWNLOAD_INDEX_SHA256: Final = (
    "58825a4937a4857b34eb74af6ea6566fa8dd31f2bb65c87d13815801fe615687"
)

ECOLI: Final = "Escherichia coli"
PPUTIDA: Final = "Pseudomonas putida"
#: The two hosts the bacterial schedule admits; the other 34 organisms are out of scope.
IN_SCOPE_SPECIES: Final = (ECOLI, PPUTIDA)

#: Strain names the genomes tier resolves; a case whose ``Strains`` cell names none of
#: them would need a new assembly set before any genotype could be written.
GENOME_TIER_STRAINS: Final = ("MG1655", "BW25113", "W3110", "REL606", "KT2440")

#: Template words that would mark an uncertainty column or note in a workbook.
INTERVAL_TERMS: Final = re.compile(
    r"±|\bSD\b|\bS\.D\.|standard deviation|confidence|interval|lower bound|"
    r"upper bound|\berror\b",
    re.IGNORECASE,
)

PAPER: Final = Provenance(
    source_uri=PAPER_TEXT_REL, citation_key=CITATION_KEY, sha256=PAPER_TEXT_SHA256
)


# --------------------------------------------------------------------------- #
# What the paper says about the release
# --------------------------------------------------------------------------- #
CASE_COUNT: Final = SourcedValue(
    value={"cases": 581, "organisms": 36},
    provenance=PAPER.model_copy(update={"page": "Data Source"}),
    quote=(
        "Currently, the database encompasses 581 cases of flux distributions from 36 "
        "organisms."
    ),
)

REFERENCE_COUNT: Final = SourcedValue(
    value=118,
    provenance=PAPER.model_copy(update={"page": "Data Source"}),
    quote=(
        "Ultimately, a total of 118 references were collected as a preliminary source "
        "for our database."
    ),
)

ECOLI_TABLE_ROW: Final = SourcedValue(
    value={"references": 32, "cases": 297, "flux_graphs": 297},
    provenance=PAPER.model_copy(update={"page": "Table 1"}),
    quote="Escherichia coli\t32\t297\t297",
    note=(
        "The paper's E. coli row says 32 references; the Download page links 31 E. coli "
        "references whose workbooks hold 297 cases, and the site's own statistics table "
        "says 31. The case count agrees; the reference count does not, and the release "
        "is what is measured."
    ),
)

PPUTIDA_TABLE_ROW: Final = SourcedValue(
    value={"references": 2, "cases": 3, "flux_graphs": 3},
    provenance=PAPER.model_copy(update={"page": "Table 1"}),
    quote="Pseudomonas putida\t2\t3\t3",
)

LUMPED_SPLIT: Final = SourcedValue(
    value="curator_split_onto_kegg_reactions",
    provenance=PAPER.model_copy(update={"page": "Data Source"}),
    quote=(
        "The lumped reactions in these studies were broken down into their original "
        "forms, as in the KEGG Reaction Database (36), and the flux value was mapped to "
        "its precisely corresponding reaction."
    ),
    note=(
        "The reaction set of a stored map is the curators' KEGG rendering, not the "
        "network the source study fitted."
    ),
)

RELATIVE_FLUX: Final = SourcedValue(
    value="relative_to_substrate_uptake_100",
    provenance=PAPER.model_copy(update={"page": "Browse flux distribution"}),
    quote=(
        "The ‘Flux value’, the core of the database, displays the quantity of "
        "the flux value relative to the substrate uptake rate. In the case of multiple "
        "substrates, the sum of all the substrates uptake rates is set to 100."
    ),
    note="Measured: every value row of all 33 in-scope workbooks has unit 'relative flux'.",
)

DATA_FIELDS: Final = SourcedValue(
    value=(
        "experiment_name",
        "strain",
        "culture_medium",
        "carbon_source",
        "growth_rate",
        "specific_rate",
        "case_specific_description",
    ),
    provenance=PAPER.model_copy(update={"page": "Data Content"}),
    quote=(
        "Each individual case of flux distribution includes several data fields "
        "describing the origin of the information, the process measure, a "
        "case-specific description and the flux distribution."
    ),
    note=(
        "The paragraph this sentence opens defines each field in turn; none is an "
        "uncertainty, and the paper contains none of 'standard deviation', "
        "'confidence' or 'uncertainty' (counted by the inventory script)."
    ),
)

STRAIN_FIELD: Final = SourcedValue(
    value="strain_if_mentioned",
    provenance=PAPER.model_copy(update={"page": "Data Content"}),
    quote=(
        "The ‘Strain’ denotes the specific strain name of the organism if it "
        "was mentioned in the reference."
    ),
)

DOWNLOAD_FORMAT: Final = SourcedValue(
    value="one_excel_file_per_reference",
    provenance=PAPER.model_copy(update={"page": "Data Download and Submission"}),
    quote=(
        "Clicking the corresponding reference name will display the links for the "
        "corresponding flux distribution that is stored in an Excel file."
    ),
    note="Measured: 118 references on the Download page, one .xls each.",
)

INTERVAL_GAP: Final = ProvenanceGap(
    field="FluxPhenotype.net_flux_lower/net_flux_upper/confidence_level",
    reason=ProvenanceGapReason.not_carried_by_curation,
    looked_in=PAPER.model_copy(
        update={"page": "Data Content; all 33 in-scope workbooks"}
    ),
    note=(
        "No workbook carries a value column beyond the per-case fluxes and no cell "
        "names an uncertainty. Where a source study reported one, the curation did not "
        "carry it; the source is where it would be read."
    ),
)


class FieldFit(BaseModel):
    """One required or interval field of the flux family against the release."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str = Field(description="schema class and field")
    carried: bool = Field(description="whether every in-scope record has a value")
    evidence: str = Field(description="the measurement, from release_inventory()")


#: The flux family field by field. ``carried`` is asserted against the measured
#: inventory by the data tests, so this table cannot drift from the bytes.
FLUX_FIELD_FIT: Final[tuple[FieldFit, ...]] = (
    FieldFit(
        field="FluxPhenotype.net_flux",
        carried=True,
        evidence=(
            "per-case numbers keyed by the 'Reactionname' KEGG code: 8,144 E. coli and "
            "88 P. putida numeric cells, beside 717 and 6 'X'/'x' cells"
        ),
    ),
    FieldFit(
        field="FluxPhenotype.net_flux_lower",
        carried=False,
        evidence="0 of 33 workbooks carry a column beyond the per-case values",
    ),
    FieldFit(
        field="FluxPhenotype.net_flux_upper",
        carried=False,
        evidence="0 of 33 workbooks carry a column beyond the per-case values",
    ),
    FieldFit(
        field="FluxPhenotype.confidence_level",
        carried=False,
        evidence="0 workbook cells and 0 paper words name an uncertainty",
    ),
    FieldFit(
        field="FluxPhenotype.measurement_type",
        carried=True,
        evidence="one unit in all 33 workbooks, 'relative flux' (uptake = 100)",
    ),
    FieldFit(
        field="FluxExperimentReference.phenotype_reference",
        carried=False,
        evidence=(
            "5 E. coli and 1 P. putida workbooks hold a single case, so no parent map "
            "is in the release for them"
        ),
    ),
    FieldFit(
        field="FluxExperiment.genotype",
        carried=False,
        evidence=(
            "33 of 297 E. coli cases name a genomes-tier strain; the 190 Haverkorn van "
            "Rijsewijk 2011 cases carry a 190-long consecutive strain-label run and the "
            "deletion only as free text"
        ),
    ),
)


# --------------------------------------------------------------------------- #
# The deposited release
# --------------------------------------------------------------------------- #
class WorkbookPin(BaseModel):
    """One source reference's workbook as the Download page links it, pinned."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str = Field(description="the server's file name, an md5-like hex + .xls")
    sha256: str
    n_bytes: int
    species: str = Field(description="the Download page's species cell, verbatim")
    reference_code: str = Field(description="the page's per-species reference id")
    pmid: str = Field(description="the source paper, resolved through PubMed")
    doi: str = Field(description="the source paper's DOI, from the PubMed record")

    @property
    def relpath(self) -> str:
        """Path inside the raw mirror."""
        return f"data/xls/{self.file}"

    @property
    def source_url(self) -> str:
        """The URL the bytes were retrieved from."""
        return WORKBOOK_URL_PREFIX + self.file

    @property
    def retrieval(self) -> RetrievalRecord:
        """The re-runnable retrieval of these bytes."""
        return RetrievalRecord(
            method=RetrievalMethod.direct_url,
            source_url=self.source_url,
            retriever="torchcell.literature.retrieve.direct_url",
            params={"url": self.source_url},
            sha256=self.sha256,
            retrieved_at=RETRIEVED_AT,
        )


def _wb(
    file: str, sha: str, n: int, species: str, code: str, pmid: str, doi: str
) -> WorkbookPin:
    return WorkbookPin(
        file=file,
        sha256=sha,
        n_bytes=n,
        species=species,
        reference_code=code,
        pmid=pmid,
        doi=doi,
    )


#: Every in-scope workbook, in the Download page's order. 31 E. coli + 2 P. putida.
WORKBOOKS: Final[tuple[WorkbookPin, ...]] = (
    _wb(
        "9bf6f02a5cb0a2ef191e49c3066892df.xls",
        "1683435c8ff3de542f94e95805356b69aee69cb62563845ed81703e49cf1af6b",
        39424,
        ECOLI,
        "00000002",
        "14751266",
        "10.1016/j.ab.2003.10.036",
    ),
    _wb(
        "4874ffc349232c5560e279c46b3c4e71.xls",
        "4c3ce642e1de5271dc325be5c8b1121fcbafb7eaad3c8b0890923f65cb88068d",
        32768,
        ECOLI,
        "00000010",
        "9251207",
        "10.1128/aem.63.8.3205-3210.1997",
    ),
    _wb(
        "acdb0a2aec25785603f8f660d675d759.xls",
        "2e7ea6689b9ff9f8d3668e7ff63285c986575cde763a48ed949e429c83456c6e",
        33792,
        ECOLI,
        "00000023",
        "18806003",
        "10.1128/AEM.01327-08",
    ),
    _wb(
        "1133122c4f24c965362a340b0484ccd8.xls",
        "8c389dc6cb5395b0d92ba38bd44a4fc30b0fcd4e83cb21f59e6f79003f575764",
        34816,
        ECOLI,
        "00000015",
        "12802531",
        "10.1007/s00253-003-1357-9",
    ),
    _wb(
        "5f74af4dea14b5fdfd52e9a145b4e21d.xls",
        "5e6e7f8d372c8ca1a47037a29a0547e2b7b20a6b1f972fd438f7bb866d03b2fc",
        37888,
        ECOLI,
        "00000017",
        "14661115",
        "10.1007/s00253-003-1458-5",
    ),
    _wb(
        "cb0ba738d380eaf0315571b68ccb4ae3.xls",
        "da6647d199e4340e7f1d661dc7f16543a2a7a0046081469a82f0f2570ccf2f35",
        38912,
        ECOLI,
        "00000020",
        "20809933",
        "10.1186/1752-0509-4-122",
    ),
    _wb(
        "151676066e8ff9c88e83c7cbfbb540fd.xls",
        "a503ff22cb4ee948361efc456ca36b3289e0adc2ecfca35418be9232bf97a71e",
        34816,
        ECOLI,
        "00000007",
        "17972325",
        "10.1002/bit.21675",
    ),
    _wb(
        "cb4919a56a1ffc2028b86989a8702ab4.xls",
        "1b352f28626e2b445001fb3832285034b2f3ea974721bf976652d194dbb4cf38",
        33792,
        ECOLI,
        "00000001",
        "23893473",
        "10.1002/bit.24997",
    ),
    _wb(
        "81eff1c98c8de770a28d450f5416625a.xls",
        "b7561b5bbec701a9bc618347cb7a3c9c60b070294aceb81cdaa8e7f6450a64e5",
        34816,
        ECOLI,
        "00000006",
        "15176872",
        "10.1021/bp0342755",
    ),
    _wb(
        "a400c72d0727b2f4345faf55c90c2197.xls",
        "5b8dc31f56f027f2e13622a7ef6a3162c8463468308106dcaa59f3431835d725",
        41472,
        ECOLI,
        "00000027",
        "19785030",
        "10.1002/btpr.290",
    ),
    _wb(
        "fa21cfa7353ec0e00630865d7177d4e3.xls",
        "0ca2f7092de640319b369373ae04245b82a1e65ce2d5b334df4341d41efe444f",
        38912,
        ECOLI,
        "00000013",
        "12670695",
        "10.1016/S0378-1097(03)00133-2",
    ),
    _wb(
        "cf696a2d11f1aa65a058384d4c95d75f.xls",
        "e2200cdbf313808dbef4aaef52cf4dc34412e50482975b51920b02025646fda9",
        36352,
        ECOLI,
        "00000008",
        "15158257",
        "10.1016/j.femsle.2004.04.003",
    ),
    _wb(
        "7be596adbde3170344c20d3547dd747d.xls",
        "36197912d1d75b24c81d6b7013a67e79154361ded1118721997b3ec115081f65",
        38400,
        ECOLI,
        "00000021",
        "8988566",
        "10.1111/j.1574-6976.1996.tb00255.x",
    ),
    _wb(
        "15d5fb8184f2eeea9701ddb793994ff5.xls",
        "f9674c1f1bf3866aefb56bc33e6fded38ff6c25ae748da1d0fda1f2ec1df8aec",
        35328,
        ECOLI,
        "00000014",
        "11741855",
        "10.1128/JB.184.1.152-164.2002",
    ),
    _wb(
        "e795eaedd12fdd1827bc15de5882000a.xls",
        "b024f35ea091c3e332d948e7ca34750ca1024c2c4af9ace2718c2b9dd7f64a4f",
        37376,
        ECOLI,
        "00000025",
        "14645264",
        "10.1128/JB.185.24.7053-7067.2003",
    ),
    _wb(
        "0e5562f5406d2acd8017f6c1f11401f0.xls",
        "f190943f4ce85b64b13e68339eb2d7513324607a09135dc9b0f88ec0e39cec77",
        35840,
        ECOLI,
        "00000030",
        "15838044",
        "10.1128/JB.187.9.3171-3179.2005",
    ),
    _wb(
        "7325d945e23f4ebb78ecfc7b400e5c7b.xls",
        "83af082c4db8583968b11d3bdddb7e1f06108f4150d80b47f1c51dbc61217550",
        36864,
        ECOLI,
        "00000024",
        "18223071",
        "10.1128/JB.01353-07",
    ),
    _wb(
        "bebdbfa2f15a02ca8e3f5d9944585146.xls",
        "5c222e31848aa12f40b9c6dbd19c0469ec6cc68f105d62a26ecc124755b26ab5",
        39424,
        ECOLI,
        "00000022",
        "19561129",
        "10.1128/JB.00174-09",
    ),
    _wb(
        "1baf177c35411f3f8b41953093d1e799.xls",
        "1c6f9272b69139f40193564862e16dee9dc9b1e30047a1b018546f9ad1cc4d8f",
        35840,
        ECOLI,
        "00000016",
        "14660605",
        "10.1074/jbc.M311657200",
    ),
    _wb(
        "42cb0edef24890d8c3afe3264e11d9cf.xls",
        "939f4b37b9e60a1034232d5587222bb16113bb419cbbb481fe282cac5af84aff",
        39936,
        ECOLI,
        "00000018",
        "16319065",
        "10.1074/jbc.M510016200",
    ),
    _wb(
        "0b59c7b4aed90dfcbe26c32b718566be.xls",
        "f587b0162b7143adb8d226dcda786bacb0696148bacf1111d937e7041a3d0f4d",
        41984,
        ECOLI,
        "00000019",
        "12568740",
        "10.1016/s0168-1656(02)00316-4",
    ),
    _wb(
        "2efde048fe3f8f240561e71135dfa6b7.xls",
        "b0341cb711fa86e90d47f8cd2b2a7ae2afa26fc868ea0b95b524a0a5bd4360e5",
        38400,
        ECOLI,
        "00000009",
        "16310273",
        "10.1016/j.jbiotec.2005.09.016",
    ),
    _wb(
        "63fadee5dfb21399c37f97ea22bef41c.xls",
        "81c05fdb743154b0b627435f873279e6d27b3894a5b2ab334680b3f4a2ab4597",
        34816,
        ECOLI,
        "00000011",
        "17207877",
        "10.1016/j.jbiotec.2006.11.015",
    ),
    _wb(
        "8b41acce483fbe44c85aaa956a5da995.xls",
        "2a989bc5c27d494f1c9e540fac4c60c16cceb210d79baecd0f393c6b778e3f05",
        35328,
        ECOLI,
        "00000029",
        "17055605",
        "10.1016/j.jbiotec.2006.09.004",
    ),
    _wb(
        "0e382b093f803f384fbb8b2a5f22a48b.xls",
        "b21a05d4f72ba3f54e4c0eccd0071090ec3d2c0591e1a5a6aefbfd8bf439bb09",
        36352,
        ECOLI,
        "00000012",
        "17462663",
        "10.1016/j.chroma.2007.04.011",
    ),
    _wb(
        "d5ff4f17cf23bdef2dac20757ae9e4af.xls",
        "0bcec06c110570445fe8ce8dfc41f994369d29fa9ba7cd8e203c1d6ef0808482",
        34304,
        ECOLI,
        "00000003",
        "12850130",
        "10.1016/s1096-7176(03)00023-5",
    ),
    _wb(
        "5cd96ed0b226edb4c431f349ad70d9da.xls",
        "afc1781bacfb9b03ef27b38f68bc603a3dfa73f99630756db324cf30a26dec6e",
        34816,
        ECOLI,
        "00000004",
        "24021936",
        "10.1016/j.ymben.2013.08.006",
    ),
    _wb(
        "007d4e2b7b05355b44180de78129798a.xls",
        "2f43513721a0ca3208fc2c31c419065fc52ce58b579f215a04660ff9665fc4ad",
        36352,
        ECOLI,
        "00000028",
        "22973998",
        "10.1186/1475-2859-11-127",
    ),
    _wb(
        "c02a28f1d398cfca740d8a896961e712.xls",
        "5903021b087dc8f3c0d2a6894a3d56d145e98521ddacb0274901bb0e029ee63f",
        34304,
        ECOLI,
        "00000026",
        "16849805",
        "10.1099/mic.0.28765-0",
    ),
    _wb(
        "afd9edb5e6ccc478603f7e675f0a6edf.xls",
        "9b7f920a587d9caa15a0950be9f04ccba3c0d894c4d9f71fad0b58810508e19f",
        174592,
        ECOLI,
        "00000031",
        "21451587",
        "10.1038/msb.2011.9",
    ),
    _wb(
        "ff31a805cace3d0b8e3bc6506cede2e5.xls",
        "a32f67ff65b2e248f9be824e45a8163463dc2e0cd8d389919e385cb53c61e05a",
        35328,
        ECOLI,
        "00000005",
        "19478804",
        "10.1038/nprot.2009.58",
    ),
    _wb(
        "dbabc44061463de39594c4801ae7716e.xls",
        "66a603af4cec7e6c8b37509fc78e9d0561d6bf92a66654bc40cbf531e2adff43",
        36864,
        PPUTIDA,
        "00000001",
        "15716428",
        "10.1128/JB.187.5.1581-1590.2005",
    ),
    _wb(
        "268c7f8ba9e5f628f9f59e09e082bbd2.xls",
        "a26a07ffa1829f97e538f188dd793ecc9fbdfa0aebac3c2e17b3e6df7ff9bbae",
        35840,
        PPUTIDA,
        "00000002",
        "19560494",
        "10.1016/j.jbiotec.2009.06.023",
    ),
)
WORKBOOKS_BY_FILE: Final[dict[str, WorkbookPin]] = {w.file: w for w in WORKBOOKS}


class OtherFile(BaseModel):
    """A non-workbook file of the deposit: the paper and the Download page."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relpath: str
    sha256: str
    role: str
    retrieval: RetrievalRecord


def _pmc(name: str, sha: str) -> RetrievalRecord:
    key = f"{PMC_PREFIX}/{name}"
    return RetrievalRecord(
        method=RetrievalMethod.pmc_cloud,
        source_url=pmc_cloud_url(key),
        retriever="torchcell.literature.retrieve.pmc_cloud_object",
        params={"key": key},
        sha256=sha,
        retrieved_at=RETRIEVED_AT,
    )


OTHER_FILES: Final[tuple[OtherFile, ...]] = (
    OtherFile(
        relpath=PAPER_TEXT_REL,
        sha256=PAPER_TEXT_SHA256,
        role=ROLE_PAPER_TEXT,
        retrieval=_pmc(PAPER_TEXT_NAME, PAPER_TEXT_SHA256),
    ),
    OtherFile(
        relpath=PAPER_PDF_REL,
        sha256=PAPER_PDF_SHA256,
        role=ROLE_PAPER_PDF,
        retrieval=_pmc(PAPER_PDF_NAME, PAPER_PDF_SHA256),
    ),
    OtherFile(
        relpath=DOWNLOAD_INDEX_REL,
        sha256=DOWNLOAD_INDEX_SHA256,
        role=ROLE_RAW_DATA,
        retrieval=RetrievalRecord(
            method=RetrievalMethod.direct_url,
            source_url=DOWNLOAD_PAGE_URL,
            retriever="torchcell.literature.retrieve.direct_url",
            params={"url": DOWNLOAD_PAGE_URL},
            sha256=DOWNLOAD_INDEX_SHA256,
            retrieved_at=RETRIEVED_AT,
        ),
    ),
)

#: What the release holds that is deliberately not deposited.
NOT_MIRRORED: Final = (
    "The 85 workbooks of the 34 organisms other than E. coli and P. putida: outside "
    "the bacterial schedule's two hosts",
    "The per-case web pages and Cytoscape views: renderings of the same workbooks",
)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/zhangCeCaFDBCuratedDatabase2015``."""
    if data_root is None:
        load_dotenv()
        data_root = os.environ["DATA_ROOT"]
    return Path(data_root) / RAW_DIR_REL


def _deposit(src: Path, dest: Path, sha: str) -> None:
    """Copy ``src`` to ``dest`` after checking its hash; never overwrite other bytes."""
    got = _sha256(src)
    if got != sha:
        raise RuntimeError(f"{src} sha256 mismatch: got {got}, expected {sha}")
    if dest.exists():
        if _sha256(dest) != sha:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def retrieve_release(dest_dir: str | Path) -> dict[str, Path]:
    """Re-run every recorded retrieval into ``dest_dir``, verifying each sha256.

    Returns ``{mirror relpath: local file}``, the ``sources`` argument of
    :func:`deposit_raw_mirror`.
    """
    dest = Path(dest_dir)
    out: dict[str, Path] = {}
    records = [(w.relpath, w.retrieval) for w in WORKBOOKS] + [
        (f.relpath, f.retrieval) for f in OTHER_FILES
    ]
    for relpath, record in records:
        data = run_retriever(record)
        got = hashlib.sha256(data).hexdigest()
        if got != record.sha256:
            raise RuntimeError(
                f"{record.source_url} sha256 drift: got {got}, expected {record.sha256}"
            )
        path = dest / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        out[relpath] = path
    return out


def deposit_raw_mirror(
    *, sources: Mapping[str, str | Path], data_root: str | None = None
) -> Path:
    """Write the raw mirror plus its ``manifest.json`` from already-retrieved files.

    ``sources`` maps every mirror relpath to a local file. Idempotent by sha256.
    """
    expected = {w.relpath for w in WORKBOOKS} | {f.relpath for f in OTHER_FILES}
    missing = sorted(expected - set(sources))
    if missing:
        raise KeyError(f"no source given for {missing}")
    root = raw_mirror_dir(data_root)
    records: list[ArtifactRecord] = []
    for w in WORKBOOKS:
        _deposit(Path(sources[w.relpath]), root / w.relpath, w.sha256)
        records.append(
            ArtifactRecord(
                path=w.relpath,
                role=ROLE_RAW_DATA,
                bytes=w.n_bytes,
                sha256=w.sha256,
                source=w.source_url,
                original_filename=w.file,
                retrieval=w.retrieval,
            )
        )
    for f in OTHER_FILES:
        _deposit(Path(sources[f.relpath]), root / f.relpath, f.sha256)
        records.append(
            ArtifactRecord(
                path=f.relpath,
                role=f.role,
                bytes=(root / f.relpath).stat().st_size,
                sha256=f.sha256,
                source=f.retrieval.source_url,
                original_filename=Path(f.relpath).name,
                retrieval=f.retrieval,
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=PAPER_TITLE,
        files=records,
        si_data_sources=[DOWNLOAD_PAGE_URL],
        si_expected=list(NOT_MIRRORED),
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return root


# --------------------------------------------------------------------------- #
# Reading the release
# --------------------------------------------------------------------------- #
class IndexRow(BaseModel):
    """One reference row of the Download page."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    species: str
    reference: str
    reference_code: str
    files: list[str]


_SPECIES_BLOCK: Final = re.compile(
    r'<td>([^<]+)</td>\s*<td colspan="2">\s*<ul>(.*?)</ul>', re.S
)
_REFERENCE_ITEM: Final = re.compile(
    r'<li class="flip">(.*?)</li>\s*<p class="flip" id="(\d+)".*?'
    r'<div class="\2"[^>]*>(.*?)</div>',
    re.S,
)
_FILE_LINK: Final = re.compile(r'href="\./download_files/([0-9a-f]+\.xls)"')


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_download_index(page: str) -> list[IndexRow]:
    """Every (species, reference, workbook) row of the Download page, in page order."""
    rows: list[IndexRow] = []
    for block in _SPECIES_BLOCK.finditer(page):
        species = _squash(html.unescape(block.group(1)))
        for item in _REFERENCE_ITEM.finditer(block.group(2)):
            rows.append(
                IndexRow(
                    species=species,
                    reference=_squash(html.unescape(item.group(1))),
                    reference_code=item.group(2),
                    files=_FILE_LINK.findall(item.group(3)),
                )
            )
    return rows


Cell = str | float


def read_grid(path: str | Path) -> list[list[Cell]]:
    """The single sheet of a CeCaFDB workbook as rows of cell values."""
    book = xlrd.open_workbook(str(path))
    if book.nsheets != 1:
        raise ValueError(f"{path} has {book.nsheets} sheets, expected 1")
    sheet = book.sheet_by_index(0)
    return [
        [sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)
    ]


class ParsedWorkbook(BaseModel):
    """The template fields of one workbook and its value block, as released."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_name: str
    coordinator: str
    case_ids: list[str]
    strains: list[str]
    genotypes: list[str]
    carbon_sources: list[str]
    growth_rates: list[str]
    descriptions: list[str]
    reactions: list[str] = Field(description="the 'Reaction' column, an equation")
    reaction_codes: list[str] = Field(description="the 'Reactionname' column")
    units: list[str]
    values: list[list[Cell]] = Field(description="reaction x case, '' when empty")
    n_extra_value_columns: int = Field(
        description="columns past the case columns that carry any value-row cell"
    )
    n_interval_term_cells: int = Field(
        description="cells anywhere in the sheet matching INTERVAL_TERMS"
    )


def _label_row(
    grid: Sequence[Sequence[Cell]], label: str, stop: int | None = None
) -> int:
    """Index of the one row above stop whose first cell is label.

    The template's casing varies between workbooks (genotype/Genotype), and a
    free-text block BELOW the value table can repeat a label (one workbook has a second
    carbon source row at row 108), so template fields are looked up above the value
    header only.
    """
    hits = [
        i
        for i, row in enumerate(grid[:stop])
        if str(row[0]).strip().casefold() == label.casefold()
    ]
    if len(hits) != 1:
        raise ValueError(f"expected one {label!r} row, found {len(hits)}")
    return hits[0]


def parse_workbook(grid: Sequence[Sequence[Cell]]) -> ParsedWorkbook:
    """Parse the fixed CeCaFDB template (``V1.0F``) of one workbook."""
    header = _label_row(grid, "Reaction")
    measurements = _label_row(grid, "Measurements", header)
    if [str(c) for c in grid[header][:3]] != ["Reaction", "Reactionname", "Unit"]:
        raise ValueError(f"unexpected value header {grid[header][:3]}")
    width = max(len(row) for row in grid)
    case_cols = [
        c for c in range(3, len(grid[measurements])) if grid[measurements][c] != ""
    ]
    if not case_cols or case_cols != list(range(3, 3 + len(case_cols))):
        raise ValueError(f"case columns are not contiguous from column 3: {case_cols}")
    first, last = 1, case_cols[-1] - 2

    def per_case(label: str) -> list[str]:
        row = grid[_label_row(grid, label, header)]
        return [str(row[c]).strip() for c in range(first, last + 1)]

    value_rows: list[int] = []
    r = header + 1
    while r < len(grid) and str(grid[r][0]).strip():
        value_rows.append(r)
        r += 1
    extra = [
        c
        for c in range(3, width)
        if c not in case_cols
        and any(c < len(grid[i]) and grid[i][c] != "" for i in value_rows)
    ]
    n_terms = sum(1 for row in grid for cell in row if INTERVAL_TERMS.search(str(cell)))
    return ParsedWorkbook(
        experiment_name=_squash(
            str(grid[_label_row(grid, "Experiment Name (ID)", header)][1])
        ),
        coordinator=str(grid[_label_row(grid, "Coordinator", header)][1]).strip(),
        case_ids=[str(grid[measurements][c]) for c in case_cols],
        strains=per_case("Strains"),
        genotypes=per_case("Genotype"),
        carbon_sources=per_case("Carbon source"),
        growth_rates=per_case("Growth rate"),
        descriptions=per_case("Case-specific description"),
        reactions=[str(grid[i][0]).strip() for i in value_rows],
        reaction_codes=[str(grid[i][1]).strip() for i in value_rows],
        units=[str(grid[i][2]).strip() for i in value_rows],
        values=[[grid[i][c] for c in case_cols] for i in value_rows],
        n_extra_value_columns=len(extra),
        n_interval_term_cells=n_terms,
    )


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
_INTEGER_SUFFIX: Final = re.compile(r"^(.*?)(\d+)$")


def longest_label_run(labels: Sequence[str]) -> int:
    """Longest run of adjacent labels sharing a prefix with suffixes n, n+1, n+2, ...

    A spreadsheet fill-series turns ``BW25113`` dragged across 190 columns into
    ``BW25113 ... BW25302``; a run longer than 2 is that signature. Labels without an
    integer suffix break a run.
    """
    best = 0
    run = 0
    prev: tuple[str, int] | None = None
    for label in labels:
        m = _INTEGER_SUFFIX.match(label)
        cur = (m.group(1), int(m.group(2))) if m else None
        if cur is not None and prev is not None and cur == (prev[0], prev[1] + 1):
            run += 1
        else:
            run = 1 if cur is not None else 0
        best = max(best, run)
        prev = cur
    return best


def names_genome_tier_strain(label: str) -> bool:
    """Whether a ``Strains`` cell names a strain the genomes tier resolves."""
    return any(re.search(rf"\b{s}\b", label) for s in GENOME_TIER_STRAINS)


class WorkbookInventory(BaseModel):
    """What one workbook was measured to hold."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str
    species: str
    reference_code: str
    pmid: str
    doi: str
    experiment_name: str
    index_reference_matches: bool = Field(
        description="the workbook's Experiment Name equals the Download page reference"
    )
    n_cases: int
    n_reactions: int
    n_numeric_values: int
    n_placeholder_values: int = Field(
        description="non-empty, non-numeric value cells (the release uses 'X'/'x')"
    )
    units: list[str]
    n_extra_value_columns: int
    n_interval_term_cells: int
    n_cases_naming_genome_tier_strain: int
    longest_strain_label_run: int
    distinct_strain_labels: int
    distinct_genotypes: int


def inventory_workbook(
    pin: WorkbookPin, parsed: ParsedWorkbook, index_reference: str
) -> WorkbookInventory:
    """Measure one parsed workbook (the pure half)."""
    flat = [v for row in parsed.values for v in row if v != ""]
    numeric = [v for v in flat if isinstance(v, float)]
    return WorkbookInventory(
        file=pin.file,
        species=pin.species,
        reference_code=pin.reference_code,
        pmid=pin.pmid,
        doi=pin.doi,
        experiment_name=parsed.experiment_name,
        index_reference_matches=parsed.experiment_name == _squash(index_reference),
        n_cases=len(parsed.case_ids),
        n_reactions=len(parsed.reactions),
        n_numeric_values=len(numeric),
        n_placeholder_values=len(flat) - len(numeric),
        units=sorted(set(parsed.units)),
        n_extra_value_columns=parsed.n_extra_value_columns,
        n_interval_term_cells=parsed.n_interval_term_cells,
        n_cases_naming_genome_tier_strain=sum(
            names_genome_tier_strain(s) for s in parsed.strains
        ),
        longest_strain_label_run=longest_label_run(parsed.strains),
        distinct_strain_labels=len(set(parsed.strains)),
        distinct_genotypes=len(set(parsed.genotypes)),
    )


class SpeciesTotals(BaseModel):
    """Per-host totals over the in-scope workbooks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    references: int
    cases: int
    numeric_values: int
    placeholder_values: int
    single_case_references: int = Field(
        description="workbooks with one case, so no in-release reference map"
    )
    cases_naming_genome_tier_strain: int


class ReleaseInventory(BaseModel):
    """The whole in-scope release, measured, and the decision it supports."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_key: str
    n_index_references: int = Field(description="all organisms, Download page")
    n_index_workbooks: int
    workbooks: list[WorkbookInventory]
    totals: dict[str, SpeciesTotals]
    n_workbooks_with_interval: int = Field(
        description="workbooks with an extra value column or an uncertainty term"
    )
    units: list[str]
    all_attributed: bool = Field(
        description="every workbook names the reference the Download page files it under"
    )
    largest_reference_case_fraction: float = Field(
        description="share of E. coli cases from the single largest reference"
    )
    loadable_as_flux_with_interval: bool


def inventory_release(
    index: Sequence[IndexRow], parsed: Mapping[str, ParsedWorkbook]
) -> ReleaseInventory:
    """Build the release inventory from the parsed index and workbooks (pure)."""
    in_scope = [r for r in index if r.species in IN_SCOPE_SPECIES]
    by_file = {r.files[0]: r for r in in_scope if len(r.files) == 1}
    if len(by_file) != len(in_scope) or set(by_file) != set(WORKBOOKS_BY_FILE):
        raise ValueError("Download page no longer lists exactly the pinned workbooks")
    inventories = [
        inventory_workbook(pin, parsed[pin.file], by_file[pin.file].reference)
        for pin in WORKBOOKS
    ]
    totals: dict[str, SpeciesTotals] = {}
    for species in IN_SCOPE_SPECIES:
        rows = [w for w in inventories if w.species == species]
        totals[species] = SpeciesTotals(
            references=len(rows),
            cases=sum(w.n_cases for w in rows),
            numeric_values=sum(w.n_numeric_values for w in rows),
            placeholder_values=sum(w.n_placeholder_values for w in rows),
            single_case_references=sum(w.n_cases == 1 for w in rows),
            cases_naming_genome_tier_strain=sum(
                w.n_cases_naming_genome_tier_strain for w in rows
            ),
        )
    with_interval = sum(
        w.n_extra_value_columns > 0 or w.n_interval_term_cells > 0 for w in inventories
    )
    ecoli = [w for w in inventories if w.species == ECOLI]
    return ReleaseInventory(
        citation_key=CITATION_KEY,
        n_index_references=len(index),
        n_index_workbooks=sum(len(r.files) for r in index),
        workbooks=inventories,
        totals=totals,
        n_workbooks_with_interval=with_interval,
        units=sorted({u for w in inventories for u in w.units}),
        all_attributed=all(w.index_reference_matches for w in inventories),
        largest_reference_case_fraction=max(w.n_cases for w in ecoli)
        / sum(w.n_cases for w in ecoli),
        loadable_as_flux_with_interval=with_interval > 0,
    )


def release_inventory(raw_dir: str | Path | None = None) -> ReleaseInventory:
    """Read the deposited Download page and workbooks, verify hashes, inventory them."""
    root = Path(raw_dir) if raw_dir is not None else raw_mirror_dir()
    index_path = root / DOWNLOAD_INDEX_REL
    if _sha256(index_path) != DOWNLOAD_INDEX_SHA256:
        raise ValueError(f"{index_path} sha256 drift")
    parsed: dict[str, ParsedWorkbook] = {}
    for pin in WORKBOOKS:
        path = root / pin.relpath
        if _sha256(path) != pin.sha256:
            raise ValueError(f"{path} sha256 drift")
        parsed[pin.file] = parse_workbook(read_grid(path))
    index = parse_download_index(index_path.read_text(encoding="utf-8"))
    return inventory_release(index, parsed)
