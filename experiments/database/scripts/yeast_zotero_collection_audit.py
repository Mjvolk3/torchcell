#!/usr/bin/env python
# experiments/database/scripts/yeast_zotero_collection_audit.py
# [[experiments.database.scripts.yeast_zotero_collection_audit]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/yeast_zotero_collection_audit.py
r"""Audit of the yeast Zotero collection: every item with no loader, classified and gated.

The group collection ``database/Saccharomyces-cerevisiae`` (Zotero key ``KCG3NWH7``) holds
papers whose datasets the KG could carry. This script holds the audit of every item in it
that had no loader under ``torchcell/datasets/scerevisiae/`` on 2026-10-11 (excluding yeast
schedule rows 1 to 5, loaded in parallel that day), as typed data: the category, the
released artifacts, the record count measured from bytes, the genotype basis, the phenotype
class or the schema gap, the overlap risk, and the verbatim quotes behind the judgments. It
then runs the candidate gates (:mod:`torchcell.candidates.gates`, the evaluators the
``candidate-gate`` CLI runs) for every key that is a dataset candidate and, with
``--write``, stores the verdicts in ``database/candidates/``.

Categories:

- ``a`` loadable now, no schema change
- ``b`` loadable with a named additive schema gap (an issue is filed)
- ``c`` method, tool, protocol or database paper with no genotype x environment ->
  phenotype records of its own; no verdict is written (the verdict's G1 vocabulary has no
  kind for a paper that releases no records, and calling it a transcription would be false)
- ``d`` a genomes or sequence resource for the genomes tier
- ``e`` an aggregation (decision 16 of [[plan.dataset-admission-pipeline.2026.10.10]])
- ``f`` blocked at source (walled, vanished, or never released per record)
- ``g`` wrong organism; no verdict is written

Where the evaluators cannot express a judgment off a row's fields, the gate is composed
from the same record types and its reason says ``composed``:

- an off-schedule key has no row in the yeast candidate table, so its row is built here
  (``RowSpec``) and the deterministic G2 to G4 read it like a table row;
- a schedule row whose text names no host (the token scan of ``gates.HOSTS`` then refuses
  it as ``absent_unnamed``) gets the host the paper names (``HostSpec``, with the quote on
  G2): a host whose token maps to a tier set is probed exactly as ``gates.probe_host``
  does, and a host no set holds is ``absent_provisionable`` (blocked:provision);
- a release whose per-record values were never released or are walled is a G3
  blocked:wall carrying the manual recipe (no manifest entry records an unfetched file);
- a gap the phenotype-class check cannot see (a genotype-side gap beside an existing
  phenotype class) is a G5 ``gap`` naming its issue.

Measurements behind the entries were taken on 2026-10-11 from the literature mirror (SI
captured that day by ``scripts/lit_capture_si.py``, report
``torchcell-library/_sync_reports/si_20261011T104400Z.json``), the raw mirror, the dev
stores (read-only) and listing probes of the external repositories; the note
[[database.yeast-zotero-collection-audit.2026.10.11]] carries the narrative.

Run from the repo root (a worktree: ``PYTHONPATH=<worktree>``)::

    python experiments/database/scripts/yeast_zotero_collection_audit.py           # print
    python experiments/database/scripts/yeast_zotero_collection_audit.py --write   # store

``--write`` writes ``database/candidates/<key>.json`` for every gated key and the note
tables to ``experiments/database/results/yeast_zotero_collection_audit.md``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import date
from pathlib import Path
from typing import Final, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from torchcell.candidates import gates, store
from torchcell.candidates.cli import load_table_module, table_rows
from torchcell.candidates.gates import CandidateRow, GateRun, SourceKindFinding
from torchcell.candidates.verdict import (
    AggregationRecord,
    CandidateVerdict,
    G2Host,
    G2Outcome,
    G2Record,
    G5Record,
    GateName,
    GateResult,
    InventoryRecord,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

SCRIPT: Final = Path(__file__).resolve()
REPO: Final = SCRIPT.parents[3]
RESULTS: Final = SCRIPT.parent.parent / "results"
TABLES_OUT: Final = RESULTS / "yeast_zotero_collection_audit.md"
COLLECTION: Final = "database/Saccharomyces-cerevisiae"
COLLECTION_KEY: Final = "KCG3NWH7"

Category = Literal["a", "b", "c", "d", "e", "f", "g"]
CATEGORY_LABEL: Final[dict[str, str]] = {
    "a": "loadable now",
    "b": "schema gap",
    "c": "not a dataset",
    "d": "genomes tier",
    "e": "aggregation",
    "f": "blocked at source",
    "g": "wrong organism",
}
GATED: Final[frozenset[str]] = frozenset({"a", "b", "d", "e", "f"})


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Released(_Frozen):
    """One released artifact: what it is, where, whether we hold it, how it is reached."""

    file: str
    where: str
    sha256: str | None = Field(
        default=None, description="sha256 when its bytes were read"
    )
    access: Literal["scriptable", "walled", "vanished", "unknown"]


class RowSpec(_Frozen):
    """The candidate-row fields for a key the yeast table does not rank."""

    name: str
    klass: str
    seq_basis: str
    url: str
    accession: str
    text: str = Field(description="what the G2 token scan reads (host names)")


class HostSpec(_Frozen):
    """A host the paper names: its label and the ``gates.HOSTS`` token, if any."""

    label: str
    token: str | None = Field(description="a gates.HOSTS name; None = no tier set")


class AuditEntry(_Frozen):
    """One collection item without a loader, with every judgment the audit made."""

    citation_key: str
    zotero_item: str
    short: str = Field(description="author year, as the note tables print it")
    category: Category
    schedule_row: str | None = Field(
        description="the yeast candidate table row name; None when off-schedule"
    )
    row: RowSpec | None = Field(default=None, description="off-schedule row")
    finding: SourceKindFinding | None = Field(
        default=None, description="a G1 judgment the row fields cannot make"
    )
    hosts: tuple[HostSpec, ...] | None = Field(
        default=None, description="hosts the paper names, when the row names none"
    )
    mosaic: bool = Field(default=False, description="a segregant panel (two parents)")
    g3_wall: str | None = Field(
        default=None, description="the manual recipe of values no retriever can reach"
    )
    phenotype_class: str | None = None
    proposed_class: str | None = None
    g5_gap: str | None = Field(
        default=None, description="a gap the class-existence check cannot see"
    )
    issue: int | None = None
    blocked_issue: int | None = Field(
        default=None, description="the by-hand issue of a blocked source"
    )
    records: int | None = Field(description="measured from bytes; None = not measured")
    records_unit: str
    records_how: str
    genotype_basis: str
    released: tuple[Released, ...] = ()
    overlap: str
    summary: str
    evidence: tuple[tuple[GateName, SourcedValue], ...] = ()


def quote(
    key: str, source_uri: str, sha256: str, text: str, supports: str
) -> SourcedValue:
    """A verbatim quote off a mirror file, and what it supports."""
    return SourcedValue(
        value=supports,
        provenance=Provenance(source_uri=source_uri, citation_key=key, sha256=sha256),
        quote=text,
    )


PETER: Final = "Peter 2018 isolates"


# --------------------------------------------------------------------------- #
# The audit (measured from bytes on 2026-10-11)
# --------------------------------------------------------------------------- #
_TURCO_KEY: Final = "turcoGlobalAnalysisYeast2023"
_TURCO_EVIDENCE: Final = (
    quote(
        "turcoGlobalAnalysisYeast2023",
        "paper.md",
        "b666d48f699a69bb2d95ff0d8e5ffe9514fba9127096c0b559bbb6dd9301bf8a",
        "a data library that aggregates and annotates all published screens of the YKO collection",
        "classification (e): an aggregation of published screens",
    ),
    quote(
        "turcoGlobalAnalysisYeast2023",
        "paper.md",
        "b666d48f699a69bb2d95ff0d8e5ffe9514fba9127096c0b559bbb6dd9301bf8a",
        "devised a reproducible computational pipeline for extracting, formatting, and normalizing data from each publication",
        "value_origin = derived_by_aggregator: values are extracted from publications, not re-measured",
    ),
    quote(
        "turcoGlobalAnalysisYeast2023",
        "paper.md",
        "b666d48f699a69bb2d95ff0d8e5ffe9514fba9127096c0b559bbb6dd9301bf8a",
        "we used the mode as a reference to normalize each screen using a modified $z$ score transformation",
        "normalization (NPV)",
    ),
    quote(
        "turcoGlobalAnalysisYeast2023",
        "paper.md",
        "b666d48f699a69bb2d95ff0d8e5ffe9514fba9127096c0b559bbb6dd9301bf8a",
        "The original phenotypic values, along with Python code for data extraction and processing, are available at https://github.com/yeastphenome/yp-data and Zenodo at https://doi.org/10.5281/zenodo.7714347.",
        "data availability: per-record original values",
    ),
)
_LOEGLER_KEY: Final = "loeglerGenotypePhenotype10862025"
_LOEGLER_EVIDENCE: Final = (
    quote(
        "loeglerGenotypePhenotype10862025",
        "paper.md",
        "3d2a5caedf52019d42af7a1f2fe9a9273b2d889d60fba07c14c078ab041b0bf8",
        "Genomes and annotations, gene-based pangenome, graph-based pangenomes, SV matrix and phenotypes generated are available on Zenodo",
        "assembly + phenotype deposit on Zenodo",
    ),
    quote(
        "loeglerGenotypePhenotype10862025",
        "paper.md",
        "3d2a5caedf52019d42af7a1f2fe9a9273b2d889d60fba07c14c078ab041b0bf8",
        "This resource complements previous phenotypic data spanning 241 colony growth traits (used here as organismal trait proxies), and 8,150 molecular traits, including transcriptomic and proteomic measurements",
        "phenotypes are re-used, not new (refs 19-22)",
    ),
    quote(
        "loeglerGenotypePhenotype10862025",
        "paper.md",
        "3d2a5caedf52019d42af7a1f2fe9a9273b2d889d60fba07c14c078ab041b0bf8",
        "19. Peter, J. et al. Genome evolution across 1,011 Saccharomyces cerevisiae isolates.",
        "identity of ref 19 (growth traits source)",
    ),
)
_KHAIWAL_KEY: Final = "khaiwalPredictingNaturalVariation2025"
_KHAIWAL_ROW: Final = "Khaiwal 2025 (33 new phenotypes on the 1,011 panel)"

AUDIT: Final[tuple[AuditEntry, ...]] = (
    # ---------------- (a) loadable now ----------------
    AuditEntry(
        citation_key="piotrowskiFunctionalAnnotationChemical2017",
        zotero_item="K8PYLHCJ",
        short="Piotrowski 2017",
        category="a",
        schedule_row="Piotrowski 2017 (MOSAIC diagnostic panel)",
        phenotype_class="EnvironmentResponsePhenotype",
        records=3438421,
        records_unit="strain x compound z-scores",
        records_how="si/si13.xlsx (Supplementary Data Set 5) sheet All_conditions: 291 strains x 11,984 conditions, numeric cells only (48,923 'NA' dropped); replicate-averaged over 3 technical PCR replicates",
        genotype_basis="291 diagnostic kanMX deletions in the pdr1 pdr3 snq2 drug-sensitized SGA background (pool v2.0 lists 310; not the ~157 the schedule row assumed)",
        released=(
            Released(
                file="si/si13.xlsx (NIHMS884855 Suppl_Dataset_5.xlsx)",
                where="mirror (PMC open-data bucket)",
                sha256="96e07f956f7edeb07359fc8b1fa5c72a167bf9bb18d1e8a680c76362e97aaefd",
                access="scriptable",
            ),
            Released(
                file="si/si11.xlsx (Data Set 4: condition metadata, SMILES)",
                where="mirror",
                sha256="eba460fe354bafd4e57c2a8438201cdc9399cf97582e3d86adf3e9362f821f87",
                access="scriptable",
            ),
        ),
        overlap="none held: no Piotrowski store; vanacloig2022.py cites it for methods only; yp-data has no folder for PMID 28759014. Same sensitized lineage as env_chemgen_vanacloig2022 (different screens).",
        summary="largest chemical-genetic matrix in the collection, already in the mirror",
        evidence=(
            (
                "G3",
                quote(
                    "piotrowskiFunctionalAnnotationChemical2017",
                    "paper.md",
                    "9314a0dd932c1b20b4dc4297de4ce9452b09f314bf100f05f2fe718fb3415fc1",
                    "All high-confidence chemical-genetic data and associated compound information has been deposited to PubChem",
                    "data availability: PubChem holds only the high-confidence subset; the full matrix is the SI",
                ),
            ),
            (
                "G3",
                quote(
                    "piotrowskiFunctionalAnnotationChemical2017",
                    "paper.md",
                    "9314a0dd932c1b20b4dc4297de4ce9452b09f314bf100f05f2fe718fb3415fc1",
                    "The genomic extractions for each plate were amplified in triplicate using three unique multiplex primer plates (three technical replicates).",
                    "replicates: 3 technical PCR replicates averaged into each DS5 value",
                ),
            ),
            (
                "G5",
                quote(
                    "piotrowskiFunctionalAnnotationChemical2017",
                    "paper.md",
                    "9314a0dd932c1b20b4dc4297de4ce9452b09f314bf100f05f2fe718fb3415fc1",
                    "The MATα query strain was crossed to an ordered array of MATa xxxΔ::kanMX deletion mutants",
                    "genotype basis: kanMX deletions introduced into the 3-delta background by SGA",
                ),
            ),
        ),
    ),
    AuditEntry(
        citation_key="parsonsExploringModeofActionBioactive2006",
        zotero_item="3PRLBH7Y",
        short="Parsons 2006",
        category="a",
        schedule_row="Parsons 2006 (bioactive-compound profiling)",
        phenotype_class="EnvironmentResponsePhenotype",
        records=329254,
        records_unit="strain x compound log2(control/treated) ratios",
        records_how="si/si8.xls (Table S7, mmc8): 4,111 ORFs x 82 conditions, numeric cells (7,848 blank noise-removed cells dropped); companion si/si5.xls (Table S4) holds 339,135 -log10 p over 4,232 ORFs",
        genotype_basis="MATa haploid YKO (Research Genetics; BY4741 inferred), pooled barcode microarray",
        released=(
            Released(
                file="si/si8.xls (mmc8.xls, Table S7 raw matrix)",
                where="mirror (Elsevier mmc)",
                sha256="564e94c963424687edc24ed23b6e3c6882b70c246a2524057cf1dc5f393c848f",
                access="scriptable",
            ),
            Released(
                file="si/si5.xls (mmc5.xls, Table S4 p-values)",
                where="mirror",
                sha256="0446eae63b5b08af7a1a94fa3b8d42d70f1c75b55a60e0aa6e03fea189753a01",
                access="scriptable",
            ),
        ),
        overlap="none held; latent: yp-data v1.0 curates the same Table S7 (PMID 16901791), so yeastphenome SCREENS must skip it (#1000). The ratio is control over treated: a loader negates or records the orientation.",
        summary="full-collection chemical-genetic profiles, 82 conditions",
        evidence=(
            (
                "G3",
                quote(
                    "parsonsExploringModeofActionBioactive2006",
                    "paper.md",
                    "dfc222acef11c4d66807946cb36f8e99f73de5d3af2715967674bc029193c886",
                    "For the original data set used to generate the cluster see Table S7 and for the complete clustered data set see Table S8.",
                    "Table S7 (si8.xls) is the original per-strain matrix; S8 is the derived clustered matrix",
                ),
            ),
            (
                "G3",
                quote(
                    "parsonsExploringModeofActionBioactive2006",
                    "paper.md",
                    "dfc222acef11c4d66807946cb36f8e99f73de5d3af2715967674bc029193c886",
                    "applied to a data matrix X of 4111 deletion mutant strains by 82 compounds",
                    "stated 4,111 strains matches the measured 4,111 ORF rows of Table S7",
                ),
            ),
            (
                "G5",
                quote(
                    "parsonsExploringModeofActionBioactive2006",
                    "paper.md",
                    "dfc222acef11c4d66807946cb36f8e99f73de5d3af2715967674bc029193c886",
                    "ratios of the control divided by sample intensities were computed for each tag",
                    "statistic orientation is log2(control/treated), the opposite of the log2_ratio docstring's treatment/control",
                ),
            ),
        ),
    ),
    AuditEntry(
        citation_key="yoshikawaComprehensivePhenotypicAnalysis2009",
        zotero_item="DCF6RHD6",
        short="Yoshikawa 2009",
        category="a",
        schedule_row=None,
        row=RowSpec(
            name="Yoshikawa 2009 (ethanol and salt stress growth of the deletion collection)",
            klass="Tolerance / robustness",
            seq_basis="S288C-KO",
            url="https://doi.org/10.1111/j.1567-1364.2008.00456.x",
            accession="FEMS Yeast Res Table S1 (publisher walled); yp-data v1.0 raw_data copy",
            text="haploid MATalpha YKO in BY4742",
        ),
        phenotype_class="EnvironmentResponsePhenotype",
        records=18916,
        records_unit="strain x condition growth records (two replicate rates + a call each)",
        records_how="FYR_456_sm_tableS1.xlsx (yp-data v1.0 raw_data copy, sha256 f8567d2c..., read in audit scratch): 4,729 ORFs x 4 conditions; the sensitive and tolerant counts reproduce the paper's Table 2; 14,187 without the 5% ethanol condition the paper never describes",
        genotype_basis="haploid MATalpha YKO (BY4742; some BY4739)",
        released=(
            Released(
                file="Table S1 (publisher original)",
                where="Wiley/OUP supporting information",
                access="walled",
            ),
            Released(
                file="Datasets/19054128/raw_data/FYR_456_sm_tableS1.xlsx",
                where="github.com/yeastphenome/yp-data at v1.0",
                sha256="f8567d2cbfe974a87cdb8629086884bc5e3c280513b99ab944065fb53d873871",
                access="scriptable",
            ),
        ),
        overlap="none held; latent: yp-data v1.0 curates the same Table S1 (PMID 19054128). Off-schedule: schedule row 85 is Yoshikawa 2011 (G3, xylose), a different paper not in Zotero.",
        summary="ethanol and NaCl growth of the deletion collection; G3 is pending until a loader deposits Table S1",
        evidence=(
            (
                "G3",
                quote(
                    "yoshikawaComprehensivePhenotypicAnalysis2009",
                    "paper.md",
                    "6552adf5b44fabc293ff1980069b67c9a6598a0d60c98e90c8ce424c5120f7fa",
                    "Table S1. Specific growth rates of single gene deletion strains.",
                    "the per-strain values live only in Supporting Information Table S1",
                ),
            ),
            (
                "G2",
                quote(
                    "yoshikawaComprehensivePhenotypicAnalysis2009",
                    "paper.md",
                    "6552adf5b44fabc293ff1980069b67c9a6598a0d60c98e90c8ce424c5120f7fa",
                    "We evaluated the growth of single gene deletion strains derived from S. cerevisiae BY4742",
                    "genotype basis: BY4742 haploid deletion collection",
                ),
            ),
            (
                "G5",
                quote(
                    "yoshikawaComprehensivePhenotypicAnalysis2009",
                    "paper.md",
                    "6552adf5b44fabc293ff1980069b67c9a6598a0d60c98e90c8ce424c5120f7fa",
                    "We simultaneously obtained 720 sets of information regarding the specific growth rates of the standard strain from the cultivation of all the deletion strains in 90 microtiter plates twice under each condition.",
                    "replicates: two independent cultures per condition",
                ),
            ),
        ),
    ),
    AuditEntry(
        citation_key="messnerUltrafastProteomicsScanning2021",
        zotero_item="XTEKHQBM",
        short="Messner 2021",
        category="a",
        schedule_row=None,
        row=RowSpec(
            name="Messner 2021 (Scanning SWATH yeast drug-response proteomes)",
            klass="Modality / backbone",
            seq_basis="reference-only",
            url="https://doi.org/10.1038/s41587-021-00860-4",
            accession="PRIDE PXD023613",
            text="prototrophic BY4741",
        ),
        phenotype_class="ProteinAbundancePhenotype",
        records=95,
        records_unit="drug-screen proteome runs (protein x run values not counted)",
        records_how="PRIDE PXD023613 file listing: 95 `*_plate_*.wiff` runs, 71 drug-treated over 18 agents + 24 DMSO; the protein values sit in the 741 MB DIA-NN report 20200914_sswath_yeast_drugscreen.tsv",
        genotype_basis="wild type only (prototrophic BY4741)",
        released=(
            Released(
                file="20200914_sswath_yeast_drugscreen.tsv (DIA-NN report)",
                where="PRIDE PXD023613 (FTP)",
                access="scriptable",
            ),
        ),
        overlap="none: proteome_messner2023 holds no drug or DMSO samples (yeast5k metadata grep 0)",
        summary="16 drugs + DMSO on one background, quadruplicate",
        evidence=(
            (
                "G3",
                quote(
                    "messnerUltrafastProteomicsScanning2021",
                    "paper.md",
                    "07756a757d7378503f2b284c3d0ad2896446a9bee94435120ebf6715484b0e20",
                    "The data generated for the benchmarks and the drug response screen in yeast have been deposited at the ProteomeXchange Consortium via the",
                    "data availability: the yeast drug screen is deposited",
                ),
            ),
            (
                "G3",
                quote(
                    "messnerUltrafastProteomicsScanning2021",
                    "paper.md",
                    "07756a757d7378503f2b284c3d0ad2896446a9bee94435120ebf6715484b0e20",
                    "partner repository, with the dataset identifier PXD023613",
                    "PRIDE accession",
                ),
            ),
            (
                "G2",
                quote(
                    "messnerUltrafastProteomicsScanning2021",
                    "paper.md",
                    "07756a757d7378503f2b284c3d0ad2896446a9bee94435120ebf6715484b0e20",
                    "The auxotrophic S. cerevisiae strain BY4741",
                    "host strain",
                ),
            ),
        ),
    ),
    # ---------------- (b) schema-gated ----------------
    AuditEntry(
        citation_key="leutertRegulatoryLandscapeYeast2023",
        zotero_item="3ZCWNIHX",
        short="Leutert 2023",
        category="b",
        schedule_row="Leutert 2023 (phosphoproteome x 101 conditions)",
        hosts=(HostSpec(label="BY4741 (S288C derivative)", token="S288C"),),
        phenotype_class="PhosphositeFoldChangePhenotype",
        proposed_class="PhosphositeFoldChangePhenotype",
        issue=921,
        records=529400,
        records_unit="phosphosite x treatment log2 fold changes",
        records_how="si/si9.xlsx (Table S6) sheet p_site_diff_reg read with keep_default_na=False: 5,294 sites x 100 treatments vs untreated, each with p and BH-adjusted p; 0 duplicated pairs",
        genotype_basis="wild type only (BY4741; BY4743 for the diploid condition)",
        released=(
            Released(
                file="si/si9.xlsx (MOESM9, Table S6)",
                where="mirror (Springer ESM)",
                sha256="12412d7697e36534699d53c0c75dbb1a138ea0f31519010afd673aea446acebe",
                access="scriptable",
            ),
            Released(
                file="si/si7.xlsx (MOESM7, Table S4 total proteome, 30 treatments)",
                where="mirror",
                sha256="fc5304c701fe8b8afe7362cd1ff0962043dfafa6ab987391d86a87a686fa5cc2",
                access="scriptable",
            ),
        ),
        overlap="none: no phosphoproteome store; the total-proteome arm (Table S4) fits ProteinFoldChangePhenotype today. The schedule's PXD034997 is the total proteome; the phospho set is PXD035050.",
        summary="the phosphosite leaf of #921 (Potel 2018) fits both releases",
        evidence=(
            (
                "G2",
                quote(
                    "leutertRegulatoryLandscapeYeast2023",
                    "paper.md",
                    "705eb7cc0197ab8dc16a07c41db4c3d8528bdb5447dfdc8e3a82636e0e9b11d4",
                    "The S288C-derivative BY4741 (MATa his3Δ1 leu2Δ0 met15Δ0 ura3Δ0) S. cerevisiae strain was used for all treatments, except for the diploid condition where BY4743 was used.",
                    "genotype basis and host strains",
                ),
            ),
            (
                "G5",
                quote(
                    "leutertRegulatoryLandscapeYeast2023",
                    "paper.md",
                    "705eb7cc0197ab8dc16a07c41db4c3d8528bdb5447dfdc8e3a82636e0e9b11d4",
                    "The MS proteomics data have been deposited to the ProteomeXchange Consortium via the PRIDE82 partner repository with the dataset identifiers PXD035029 for the ultradeep reference phosphoproteomic DDA data, PXD035050 for the quantitative phosphoproteomic DIA data, and PXD034997 for the quantitative proteomics DIA data.",
                    "data availability and which PRIDE accession holds which data",
                ),
            ),
            (
                "G5",
                quote(
                    "leutertRegulatoryLandscapeYeast2023",
                    "paper.md",
                    "705eb7cc0197ab8dc16a07c41db4c3d8528bdb5447dfdc8e3a82636e0e9b11d4",
                    "Significant differential expression was then calculated for each treatment against the untreated samples",
                    "Table S6 fold changes are treatment vs untreated (reference_basis)",
                ),
            ),
        ),
    ),
    AuditEntry(
        citation_key="vaishnavEvolutionEvolvabilityEngineering2022",
        zotero_item="CTSIU75E",
        short="Vaishnav 2022",
        category="b",
        schedule_row="Vaishnav 2022 (regulatory DNA fitness landscape)",
        phenotype_class="PromoterActivityPhenotype",
        proposed_class="a yeast promoter-variant perturbation (sequence-level genotype)",
        g5_gap="PromoterActivityPhenotype exists, but no yeast genotype holds the variable promoter sequence, its well_id is a required str a pooled sort-seq library has none of, and ReporterReadout has no sort-seq member",
        issue=991,
        records=75487,
        records_unit="sequence x expression-level records (MolEvol set; 20,616,659 + 30,722,376 random sequences stated, not counted)",
        records_how="GSE163045_MolEvol_seq_data_SCUra_only.splitByOrigID.meanEL.all.txt.gz (sha256 9193507a..., read in audit scratch): 75,488 lines minus the header; the full random libraries are 0.67 to 3.59 GB files on GEO and Zenodo",
        genotype_basis="promoter library: an 80 bp variable insert in a YFP reporter promoter on a CEN plasmid (Y8205 and S288C::ura3)",
        released=(
            Released(
                file="GSE163045 and GSE163866 supplementary files",
                where="GEO (ftp.ncbi.nlm.nih.gov listing; acc.cgi serves a reCAPTCHA)",
                access="scriptable",
            ),
            Released(
                file="record 4436477 (31 files)", where="Zenodo", access="scriptable"
            ),
        ),
        overlap="none in yeast; the defined-medium test set is labeled NBT_S288CdU_SCUra, which may be the de Boer 2020 set (inference, unmeasured)",
        summary="sequence-to-expression landscape; si3.xlsx and the native-promoter tsv are predictions",
        evidence=(
            (
                "G5",
                quote(
                    "vaishnavEvolutionEvolvabilityEngineering2022",
                    "paper.md",
                    "80aa5de145c36b3617809db881b313bc0be69267259e5296bb617efd8bd97f09",
                    "Data generated for this study are available at the NCBI GEO with accession numbers GSE163045 and GSE163866",
                    "the exact GEO accessions (the schedule listed GEO as unconfirmed; now confirmed)",
                ),
            ),
            (
                "G5",
                quote(
                    "vaishnavEvolutionEvolvabilityEngineering2022",
                    "paper.md",
                    "80aa5de145c36b3617809db881b313bc0be69267259e5296bb617efd8bd97f09",
                    "cloned into the promoter of a YFP gene within a CEN plasmid",
                    "genotype basis: a plasmid-borne reporter with a variable promoter insert, which is why no host-gene perturbation class applies",
                ),
            ),
            (
                "G5",
                quote(
                    "vaishnavEvolutionEvolvabilityEngineering2022",
                    "paper.md",
                    "80aa5de145c36b3617809db881b313bc0be69267259e5296bb617efd8bd97f09",
                    "A total of 20,616,659 random sequences (defined medium) and 30,722,376 random sequences (complex medium), along with their experimentally measured expression",
                    "the library sizes, STATED not measured (the files holding them are 0.7-3.6 GB)",
                ),
            ),
        ),
    ),
    AuditEntry(
        citation_key=_KHAIWAL_KEY,
        zotero_item="8HLUB5DG",
        short="Khaiwal 2025",
        category="b",
        schedule_row=None,
        row=RowSpec(
            name=_KHAIWAL_ROW,
            klass="Natural variation",
            seq_basis="isolate-WGS",
            url="https://doi.org/10.1038/s44320-025-00136-y",
            accession="Mol Syst Biol Dataset EV1 (si/si3.xlsx)",
            text="1,011 isolates of the Peter 2018 panel; BY4743 normalization control",
        ),
        finding=SourceKindFinding(
            row_name=_KHAIWAL_ROW,
            citation_key=_KHAIWAL_KEY,
            source_kind="primary",
            reason="the candidate is the 33 phenotypes EV1 marks 'Not Published' and releases here; the paper's 190 other traits are re-served from four earlier studies and its model scores are predictions, so neither is part of the candidate",
            evidence=(
                quote(
                    "khaiwalPredictingNaturalVariation2025",
                    "paper.md",
                    "4172e8d113c7af1db8d99fb6828aa918aa69ec4198e5f7e7ed0386a915e35b0b",
                    "33 unpublished phenotypes of which 22 phenotypes belong to growth measured in eight different antifungal drugs",
                    "new antifungal measurements",
                ),
                quote(
                    "khaiwalPredictingNaturalVariation2025",
                    "paper.md",
                    "4172e8d113c7af1db8d99fb6828aa918aa69ec4198e5f7e7ed0386a915e35b0b",
                    "and 11 independently measured phenotypes, including cell size, DNA content, cell flocculation frequency, and CLS measured in rapamycin which are released in this study",
                    "new cell-feature and CLS measurements",
                ),
                quote(
                    "khaiwalPredictingNaturalVariation2025",
                    "paper.md",
                    "4172e8d113c7af1db8d99fb6828aa918aa69ec4198e5f7e7ed0386a915e35b0b",
                    "we compiled 190 published life history traits measured across the 1011 S. cerevisiae collection in four main studies",
                    "aggregation: most traits re-used (Peter 2018, De Chiara 2022, D'Angiolo 2023, Galardini 2019)",
                ),
            ),
            aggregation=None,
        ),
        phenotype_class="CellFeaturePhenotype",
        proposed_class="CellFeaturePhenotype and ChronologicalLifespanPhenotype",
        issue=992,
        records=28435,
        records_unit="isolate x trait values for the 33 new phenotypes",
        records_how="si/si3.xlsx (Dataset EV1): phenotypes annotated 'Not Published' (33), numeric cells over 1,011 strain rows: 22 antifungal traits 20,104, 8 cytometry features 6,616, 3 lifespan timepoints 1,715",
        genotype_basis="natural isolates (the 1,011 panel)",
        released=(
            Released(
                file="si/si3.xlsx (MOESM3, Dataset EV1)",
                where="mirror (PMC open-data bucket)",
                sha256="de4c39640239dcf76ac839cf2d143fc96d5b6b01907929ea0ec9f5d259722055",
                access="scriptable",
            ),
        ),
        overlap="the 190 re-served traits duplicate Peter 2018 (35), De Chiara 2022 (99), Galardini 2019 (47) and D'Angiolo 2023 (9): load those from the originals; the 33 new traits overlap no store. The antifungal data carry 'do not distribute' terms in their CEFIPRA README: privacy review first.",
        summary="22 antifungal traits fit EnvironmentResponsePhenotype; 11 cytometry and lifespan traits have no class",
        evidence=(
            (
                "G2",
                quote(
                    "khaiwalPredictingNaturalVariation2025",
                    "paper.md",
                    "4172e8d113c7af1db8d99fb6828aa918aa69ec4198e5f7e7ed0386a915e35b0b",
                    "223 traits measured across 1011 genome-sequenced Saccharomyces cerevisiae strains",
                    "genotype panel",
                ),
            ),
        ),
    ),
    AuditEntry(
        citation_key="gaschGenomicExpressionPrograms2000",
        zotero_item="GNF43TXA",
        short="Gasch 2000",
        category="b",
        schedule_row="Gasch 2000 (environmental stress response)",
        hosts=(
            HostSpec(label="DBY7286 (background not stated in the paper)", token=None),
        ),
        phenotype_class="MicroarrayExpressionPhenotype",
        g5_gap="MicroarrayExpressionPhenotype requires linear per-gene expression and per-gene n_replicates; every release carries log2 ratios against time zero only",
        issue=989,
        records=1032274,
        records_unit="gene x array log2 ratios (173 arrays x 6,152 genes)",
        records_how="complete_dataset.txt (Wayback capture 20060110024849, sha256 3050664e..., read in audit scratch): 173 array columns, 155 wild type + 8 deletion + 3 overexpression + 7 DeRisi 1997",
        genotype_basis="wild type (DBY7286, DBY8768) plus yap1 and msn2 msn4 deletions and MSN2/MSN4/YAP1 overexpression",
        released=(
            Released(
                file="complete_dataset.txt (the authors' file)",
                where="Wayback Machine copy of genome-www.stanford.edu/yeast_stress (original site gone)",
                sha256="3050664e0deba976b70b954676675cce74d3288198d850abb6fb123bac58daa7",
                access="scriptable",
            ),
            Released(
                file="Gasch_2000_PMID_11102521.zip (SGD-reprocessed, 138 columns, no mutant arrays)",
                where="sgd-archive.yeastgenome.org (SPELL)",
                sha256="31f41d08b1f7e2bcded2933518eb99809b39c2a1a34f8c9e9b56f6b61ee8d5ab",
                access="scriptable",
            ),
        ),
        overlap="none: no SPELL store and $DATA_ROOT/data/sgd/spell is empty (#982); Kemmeren and Sameith are deletion-vs-pool arrays",
        summary="canonical stress-response arrays; also blocked at G2 until DBY7286's background is sourced",
        evidence=(
            (
                "G2",
                quote(
                    "gaschGenomicExpressionPrograms2000",
                    "paper.md",
                    "e22b3e01f5664a00b9a6fdcd4c78bacd5803e5c31f285f45b98c47a871a98fbb",
                    "Wild-type DBY7286 cells harboring each plasmid",
                    "host strain, background not stated",
                ),
            ),
            (
                "G5",
                quote(
                    "gaschGenomicExpressionPrograms2000",
                    "paper.md",
                    "e22b3e01f5664a00b9a6fdcd4c78bacd5803e5c31f285f45b98c47a871a98fbb",
                    "The complete data set, as well as supplemental materials, is available at http://www-genome.stanford.edu/yeast stress.",
                    "data availability: the only stated location is the Stanford lab site, which is now 404 / unresolvable",
                ),
            ),
            (
                "G5",
                quote(
                    "gaschGenomicExpressionPrograms2000",
                    "paper.md",
                    "e22b3e01f5664a00b9a6fdcd4c78bacd5803e5c31f285f45b98c47a871a98fbb",
                    "the ratios represent the expression level at each time point relative to the expression level in the unshocked, time-zero sample",
                    "phenotype is a log ratio against the time-zero sample of the same culture (environment contrast)",
                ),
            ),
        ),
    ),
    AuditEntry(
        citation_key="zhangCombiningMechanisticMachine2020",
        zotero_item="B46K3SBE",
        short="Zhang 2020",
        category="b",
        schedule_row=None,
        row=RowSpec(
            name="Zhang 2020 (combinatorial promoter library for tryptophan)",
            klass="Metabolite / precursor",
            seq_basis="engineered-chassis",
            url="https://doi.org/10.1038/s41467-020-17910-1",
            accession="GitHub sorpet/Zhang_and_Petersen_et_al_2019 (raw mirror)",
            text="CEN.PK2-1C platform strain",
        ),
        hosts=(HostSpec(label="CEN.PK2-1C", token="CEN.PK"),),
        # Both G5 gaps are closed by PR #1055: PromoterDrivenGeneIntegrationPerturbation
        # states which promoter drives each re-integrated gene, and
        # BiosensorFluorescencePhenotype holds a sensor's signal with the sensed
        # metabolite named, so the gate evaluates the class instead of a composed gap.
        phenotype_class="BiosensorFluorescencePhenotype",
        issue=990,
        records=341,
        records_unit="promoter-combination strain records (GFP synthesis rate) + 6 HPLC titers",
        records_how="raw mirror ice.csv (299 library strains with a full five-promoter genotype, 275 distinct) + recommendations.csv (42 strains, disjoint); edd.csv carries 75,026 GFP rows, 3 replicate lines per strain; si/si1.md holds the 6 titers",
        genotype_basis="engineered strain: CEN.PK2-1C platform with five genes re-integrated under one of six promoters each (6^5 designs)",
        released=(
            Released(
                file="data/ice.csv.zip, data/edd.csv.zip, data/recommendations.csv, data/biosensor_val.csv",
                where="raw mirror (GitHub commit f95f7d10); manifest sha256 all reproduce",
                sha256="d1ea7cb4c6189d444ab0d0a24c5e5dbd48689b634cf52d6afc1435841e3c9081",
                access="scriptable",
            ),
        ),
        overlap="none found",
        summary="combinatorial promoter design with a biosensor readout; CEN.PK has no assembly set",
        evidence=(
            (
                "G2",
                quote(
                    "zhangCombiningMechanisticMachine2020",
                    "paper.md",
                    "56c4b1ec3e1c916cb491dddfe3fd08d86d513b2d943fde82efe1186fed12b15f",
                    "S. cerevisiae strains were derived from CEN.PK2-1C",
                    "host background",
                ),
            ),
            (
                "G5",
                quote(
                    "zhangCombiningMechanisticMachine2020",
                    "paper.md",
                    "56c4b1ec3e1c916cb491dddfe3fd08d86d513b2d943fde82efe1186fed12b15f",
                    "The genotype and time series datasets are available at The Joint BioEnergy Institute",
                    "data availability: genotype and time series are deposited, which is exactly what the raw mirror copies",
                ),
            ),
            (
                "G5",
                quote(
                    "zhangCombiningMechanisticMachine2020",
                    "si/si1.md",
                    "ac76272e7e71d3a2e0ff5169c7302c3e9065c4bb4d4568602dd457d82f1b601c",
                    "The 30 selected native yeast promoters, and their position in the combinatorial cluster",
                    "the promoter-number to systematic-gene-name mapping that makes the ice.csv genotype strings resolvable",
                ),
            ),
        ),
    ),
    # ---------------- (d) genomes tier ----------------
    AuditEntry(
        citation_key="odonnellTelomeretotelomereAssemblies1422023",
        zotero_item="HSFSW32M",
        short="O'Donnell 2023",
        category="d",
        schedule_row=None,
        row=RowSpec(
            name="O'Donnell 2023 (telomere-to-telomere assemblies of 142 strains)",
            klass="Natural variation",
            seq_basis="isolate-WGS",
            url="https://doi.org/10.1038/s41588-023-01459-y",
            accession="ENA PRJEB59869",
            text="142 strains, ScRAP assemblies",
        ),
        hosts=(
            HostSpec(
                label="ScRAP 142-strain T2T assembly set (O'Donnell 2023)", token=None
            ),
        ),
        records=197,
        records_unit="nuclear assemblies (0 phenotype records)",
        records_how="si/si4.xlsx sheet 'Supp Table 1': 197 rows (142 haploid or collapsed, 42 diploid, 6 triploid, 7 tetraploid haplotypes); 136 mitochondrial assemblies stated",
        genotype_basis="natural isolates and lab strains; 124 of the 142 strains are 1,011-panel isolates, 123 of them with a Peter 2018 assembly already in the tier",
        released=(
            Released(
                file="assemblies PRJEB59413 (unphased), PRJEB59129 / PRJEB59231 / PRJEB59232 (haplotypes), PRJEB59230 (mitochondria)",
                where="ENA",
                access="scriptable",
            ),
            Released(
                file="si/si4.xlsx (Supplementary Tables 1-17)",
                where="mirror (PMC open-data bucket)",
                sha256="6e1ca79129743ee03ad6d7241d7636b3f8aaadc67f79cd4cd84dce61f765e20f",
                access="scriptable",
            ),
        ),
        overlap="genomes only; 71 of its assemblies are reused verbatim by Loegler 2025. A second mirror key, odonnell142TelomeretotelomereAssemblies2023, holds the same DOI (personal library; #271 keeps such pairs).",
        summary="T2T assemblies (median 17 contigs vs 3,259 for Peter 2018), phased haplotypes, a 4,809-row SV list, Ty and telomere annotation; no phenotypes",
        evidence=(
            (
                "G2",
                quote(
                    "odonnellTelomeretotelomereAssemblies1422023",
                    "paper.md",
                    "af1fbc3c3bcc8d4219dc70664d0916141ff14f987137e628febf394f735fc1da",
                    "The assembly/annotation accessions are PRJEB59413, PRJEB59129, PRJEB59231, PRJEB59232 and PRJEB59230 for unphased nuclear, HP1, HP2, HP (for polyploids) and mitochondrial assemblies, respectively.",
                    "accessions per assembly class",
                ),
            ),
            (
                "G2",
                quote(
                    "odonnellTelomeretotelomereAssemblies1422023",
                    "paper.md",
                    "af1fbc3c3bcc8d4219dc70664d0916141ff14f987137e628febf394f735fc1da",
                    "The panel comprises 197 nuclear and 136 mitochondrial genome assemblies, including 100 newly sequenced genomes",
                    "assembly count (matches the 197 rows measured in Supp Table 1)",
                ),
            ),
            (
                "G2",
                quote(
                    "odonnellTelomeretotelomereAssemblies1422023",
                    "paper.md",
                    "af1fbc3c3bcc8d4219dc70664d0916141ff14f987137e628febf394f735fc1da",
                    "The rationale for the selection of the 100 strains de novo sequenced in this study was based mainly on the knowledge gained from the 1,011 genome project",
                    "strains chosen from the Peter 2018 panel (overlap)",
                ),
            ),
        ),
    ),
    AuditEntry(
        citation_key=_LOEGLER_KEY,
        zotero_item="2EYKDJAT",
        short="Loegler 2025",
        category="d",
        schedule_row=None,
        row=RowSpec(
            name="Loegler 2025 (1,086 near telomere-to-telomere genomes)",
            klass="Natural variation",
            seq_basis="isolate-WGS",
            url="https://doi.org/10.1038/s41586-025-09637-0",
            accession="Zenodo 15698884; ENA PRJEB77686, PRJEB81147",
            text="1,086 isolates",
        ),
        finding=SourceKindFinding(
            row_name="Loegler 2025 (1,086 near telomere-to-telomere genomes)",
            citation_key=_LOEGLER_KEY,
            source_kind="aggregation",
            reason="the genomes are new; every one of its 8,153 traits (TableS11) is re-served from earlier releases for GWAS, so as a phenotype dataset it is an aggregation whose originals load on their own",
            evidence=_LOEGLER_EVIDENCE,
            aggregation=AggregationRecord(
                n_source_studies=4,
                attribution_field="TableS11 Pheno_Type (Growth, Transcriptomics, Proteomics) and the paper's refs 19-22 (Peter 2018, Caudal 2024, Muenzner 2024, Teyssonniere 2024); 65 SC*/YNB* growth conditions carry no attributable source",
                value_origin="derived_by_aggregator",
                n_sources_mirrored=3,
                net_new_vs_served=None,
                evidence=_LOEGLER_EVIDENCE,
            ),
        ),
        hosts=(
            HostSpec(
                label="1,086-isolate near-T2T assembly set (Loegler 2025)", token=None
            ),
        ),
        records=None,
        records_unit="isolate x trait values (re-served)",
        records_how="not measured: the per-isolate values are only in Zenodo Phenotypes_8391Traits.tar.gz (79.6 MB); measured instead: si/si3.xlsx TableS11 8,153 traits (232 growth, 5,816 transcript, 2,105 protein), TableS2 1,482 assemblies over 1,086 isolates",
        genotype_basis="natural isolates: 995 are 1,011-panel isolates, 91 are not (90 `XTRA_*` plus JLL)",
        released=(
            Released(
                file="GenomeAssemblies.tar.gz (16.8 GB), SVcalls.tar.gz, pangenomes, Phenotypes_8391Traits.tar.gz",
                where="Zenodo 15698884",
                access="scriptable",
            ),
            Released(
                file="si/si3.xlsx (Supplementary Tables S1-S15)",
                where="mirror (PMC open-data bucket)",
                sha256="ae46f4663ad1e9dcba901096623d598f0de4faaa6426015b3a9ed4c31402bd55",
                access="scriptable",
            ),
        ),
        overlap="HIGH for phenotypes: growth duplicates Peter 2018 (landed as PR #993), transcripts Caudal 2024, proteins Muenzner 2024 (wave 1) and Teyssonniere 2024. Genomes: 994 isolates already have a Peter assembly; 71 assemblies are O'Donnell 2023's.",
        summary="near-T2T genomes for 1,086 isolates, an SV VCF and graph pangenomes; phenotypes all re-served",
        evidence=(
            (
                "G2",
                quote(
                    "loeglerGenotypePhenotype10862025",
                    "paper.md",
                    "3d2a5caedf52019d42af7a1f2fe9a9273b2d889d60fba07c14c078ab041b0bf8",
                    "Sequencing reads associated with this work are available at the European Nucleotide Archive under the accessions PRJEB77686 and PRJEB81147.",
                    "raw read deposit",
                ),
            ),
            (
                "G2",
                quote(
                    "loeglerGenotypePhenotype10862025",
                    "paper.md",
                    "3d2a5caedf52019d42af7a1f2fe9a9273b2d889d60fba07c14c078ab041b0bf8",
                    "resulting in 1,086 isolates overall",
                    "isolate count (matches 1,086 distinct names measured in TableS2)",
                ),
            ),
        ),
    ),
    # ---------------- (e) aggregation ----------------
    AuditEntry(
        citation_key=_TURCO_KEY,
        zotero_item="KQWRRBY4",
        short="Turco 2023",
        category="e",
        schedule_row="Turco 2023 (Yeast Phenome)",
        finding=SourceKindFinding(
            row_name="Turco 2023 (Yeast Phenome)",
            citation_key=_TURCO_KEY,
            source_kind="aggregation",
            reason="Yeast Phenome re-serves every published YKO screen after extracting each paper's values and rescaling each screen to a mode-referenced NPV; its own measurements are spot assays for two ORFs",
            evidence=_TURCO_EVIDENCE,
            aggregation=AggregationRecord(
                n_source_studies=517,
                attribution_field="the release's Datasets/<PMID>/ folder (also the last field of every value-file column header); the loader stores it as Publication.pubmed_id",
                value_origin="derived_by_aggregator",
                n_sources_mirrored=17,
                net_new_vs_served=507,
                evidence=_TURCO_EVIDENCE,
            ),
        ),
        phenotype_class="EnvironmentResponsePhenotype",
        records=296777,
        records_unit="strain x screen NPVs the loader keeps (83 screens, 47 studies)",
        records_how="the 47 dev-store `raw/*_valuez.txt` files (each equal to its SCREENS sha256 pin), the loader's column filter; equals verification_report.json L1. The release: 517 PMID folders at yp-data v1.0 (GitHub API listing, sha256 34fdc938...); the paper states 531 studies and 14,495 screens",
        genotype_basis="YKO collections: haploid BY4741 and BY4742, homozygous diploid BY4743 (het screens dropped)",
        released=(
            Released(
                file="yp-data v1.0 (commit 83e2917b), Datasets/<PMID>/",
                where="GitHub / Zenodo 7714347",
                access="scriptable",
            ),
            Released(
                file="NPV bulk downloads", where="Zenodo 7714354", access="scriptable"
            ),
        ),
        overlap="the yeastphenome dev store IS this loader (built, not served). 10 source studies are their own dev stores and are excluded; Parsons 2006 and Yoshikawa 2009 would duplicate if SCREENS widened (#1000). 517 sources, 17 mirrored, 507 net new; decomposition not measured.",
        summary="the knockout phenome compendium; a loader exists (dev store yeastphenome, not in the adapter map)",
    ),
    # ---------------- (f) blocked at source ----------------
    AuditEntry(
        citation_key="leeMappingCellularResponse2014",
        zotero_item="6BWAYPY5",
        short="Lee 2014",
        category="f",
        schedule_row="Lee 2014 (HIP-HOP fitness signatures)",
        g3_wall="the per-strain FD matrices (stated 4,810 hom + 1,095 het strains x 3,356 screens) were served only by the UBC HIPHOP portal (refused a TCP connection 2026-10-11); none of the ten Science workbooks (legacy copies in data/torchcell/env_chemgen_lee2014/raw/) holds them. Recipe: request fitness_defect_matrix_hom.txt and _het.txt from the Giaever/Nislow lab (#994)",
        phenotype_class="EnvironmentResponsePhenotype",
        blocked_issue=994,
        records=None,
        records_unit="strain x screen FD scores",
        records_how="not measurable: every sheet of the ten workbooks was opened; the closest are 455 clearance scores for selected HIP hits (s2) and median FD per signature gene (s4)",
        genotype_basis="heterozygous (HIP) and homozygous (HOP) diploid YKO pools, BY4743",
        released=(
            Released(
                file="HIPHOP portal download (FD matrices)",
                where="chemogenomics.pharmacy.ubc.ca",
                access="vanished",
            ),
            Released(
                file="1250217s2.xlsx (sha256 shown) and nine more Science workbooks",
                where="legacy dev-store raw dir (not in any mirror manifest)",
                sha256="6a74854a3e7ef3802e7cc3ed355e98bc168882387f36bff4159b8ff8c262a0c7",
                access="walled",
            ),
        ),
        overlap="env_chemgen_lee2014 holds raw/ only (no LMDB); yeastphenome has no Lee values",
        summary="the largest chemical-genetic matrix stays blocked: the FD matrices are in no release we hold",
        evidence=(
            (
                "G3",
                quote(
                    "leeMappingCellularResponse2014",
                    "paper.md",
                    "08b07462d8cd9a709c54db1e4f88faa4bc47c373f5dabcaeb7ec49666e82ac38",
                    "(available at www.sciencemag.org/344/6180/208/suppl/DC1 and at http://chemogenomics.pharmacy.ubc.ca/HIPHOP/index.php?q=&f=fixed&fr=0&restr=download)",
                    "data availability: Science SI plus the HIPHOP portal download page (the probed URL)",
                ),
            ),
            (
                "G3",
                quote(
                    "leeMappingCellularResponse2014",
                    "paper.md",
                    "08b07462d8cd9a709c54db1e4f88faa4bc47c373f5dabcaeb7ec49666e82ac38",
                    "The scores are used throughout the manuscript and are searchable on our webpage (http://chemogenomics.pharmacy.ubc.ca/HIPHOP/).",
                    "the FD scores were released via the lab portal, not the SI tables",
                ),
            ),
            (
                "G3",
                quote(
                    "leeMappingCellularResponse2014",
                    "paper.md",
                    "08b07462d8cd9a709c54db1e4f88faa4bc47c373f5dabcaeb7ec49666e82ac38",
                    "and for each screen, the median FD score is 0 and the MAD is 1",
                    "statistic: FD is a robust z-score (sensitivity_score)",
                ),
            ),
        ),
    ),
    AuditEntry(
        citation_key="guzmanUltrafastLabelfreeQuantification2024",
        zotero_item="MU4HHLNG",
        short="Guzman 2024",
        category="f",
        schedule_row=None,
        row=RowSpec(
            name="Guzman 2024 (Astral knockout proteomes, 104 strains)",
            klass="Modality / backbone",
            seq_basis="S288C-KO",
            url="https://doi.org/10.1038/s41587-023-02099-7",
            accession="PRIDE PXD046386; si/si13.xlsx",
            text="BY4743 yeast knockout collection",
        ),
        g3_wall="the protein matrix is released (si/si13.xlsx) but the strains are labeled KO1 to KO104 and no file (paper, 15 SI files, PRIDE SDRF) names the deleted gene. Recipe: request the KO-number to ORF key from the Olsen lab (#998)",
        phenotype_class="ProteinAbundancePhenotype",
        blocked_issue=998,
        records=1375123,
        records_unit="protein x knockout-run values (genotypes unreleased)",
        records_how="si/si13.xlsx sheet 'KO PG collection ': 4,887 non-contaminant protein groups x 312 KO run columns, non-null cells",
        genotype_basis="104 BY4743 knockout-collection strains, identities unreleased",
        released=(
            Released(
                file="si/si13.xlsx (MOESM13)",
                where="mirror (PMC open-data bucket)",
                sha256="6772df03faa6c9feb1d12dfcf3230248895e0a755424403d1fe01c0f884db1b5",
                access="scriptable",
            ),
        ),
        overlap="proteome_messner2023 may share genes (unverifiable without the key); different strains and samples",
        summary="1,375,123 values with no genotype key",
        evidence=(
            (
                "G3",
                quote(
                    "guzmanUltrafastLabelfreeQuantification2024",
                    "paper.md",
                    "aec8c783e23d75ed23906986554181419475f27b0f7440ae3592fc498ce51009",
                    "we applied this to analyze a yeast strain library consisting of 104 gene knockouts targeting cell cycle, proteasome, DNA damage response and kinase genes",
                    "stated KO count (104, matches 104 KO numbers in si13) and gene classes, with no gene list",
                ),
            ),
            (
                "G2",
                quote(
                    "guzmanUltrafastLabelfreeQuantification2024",
                    "paper.md",
                    "aec8c783e23d75ed23906986554181419475f27b0f7440ae3592fc498ce51009",
                    "S288C isogenic yeast strains (MATα) wild-type, BY4743 and BY4743 yeast knockout collection from Horizon Discovery",
                    "genotype basis and host strain (and the MATalpha vs diploid inconsistency)",
                ),
            ),
        ),
    ),
    AuditEntry(
        citation_key="hoMassiveQTLAnalysis2021",
        zotero_item="7DQ7JSCP",
        short="Ho 2021",
        category="f",
        schedule_row=None,
        row=RowSpec(
            name="Ho 2021 (industrial traits of the RM11-1a x YJM975 segregants)",
            klass="Natural variation",
            seq_basis="segregant-WGS",
            url="https://doi.org/10.1186/s13068-021-02059-w",
            accession="Additional file 3; genotypes from She & Jarosz 2018",
            text="RM11-1a x YJM975 F6 segregants",
        ),
        hosts=(
            HostSpec(label="RM11-1a (1,011-panel isolate AAA)", token=PETER),
            HostSpec(label="YJM975 (1,011-panel isolate YBA)", token=PETER),
        ),
        mosaic=True,
        g3_wall="phenotypes are released, but the genotypes come from She & Jarosz 2018, whose genotype matrix and id crosswalk were only in a Dropbox folder now deleted. Recipe in #70",
        phenotype_class="MetabolitePhenotype",
        blocked_issue=70,
        records=20377,
        records_unit="segregant x trait values (genotypes blocked)",
        records_how="si/si_data/13068_2021_2059_MOESM3_ESM.xlsx sheet 'For the mapping experiments': 1,152 rows x 20 trait columns, numeric cells (25,317 with the five NOIS ester columns)",
        genotype_basis="segregants (F6 advanced intercross RM11-1a x YJM975)",
        released=(
            Released(
                file="si/si_data/13068_2021_2059_MOESM3_ESM.xlsx",
                where="mirror (Springer ESM)",
                sha256="7857557eaeb21591edf16141cb9a13c6eb4219343c05564be67808fc80256e74",
                access="scriptable",
            ),
        ),
        overlap="none (bloom2019 crosses do not use YJM975); same segregants as She 2018",
        summary="phenotypes in hand, genotypes vanished with She 2018's Dropbox (#70)",
        evidence=(
            (
                "G3",
                quote(
                    "hoMassiveQTLAnalysis2021",
                    "paper.md",
                    "8913bfee34532387fcd246b02d837abc1e1ea94faca3e4f4eeacca1d02cc5187",
                    "segregants for the mapping experiments are included in the Additional fle 3",
                    "per-segregant phenotypes released in Additional file 3 (OCR drops the 'fi' ligature)",
                ),
            ),
            (
                "G3",
                quote(
                    "hoMassiveQTLAnalysis2021",
                    "paper.md",
                    "8913bfee34532387fcd246b02d837abc1e1ea94faca3e4f4eeacca1d02cc5187",
                    "genotypic data of the panel were obtained from She & Jarosz (2018)",
                    "genotypes are not released by this paper; deferral to She & Jarosz",
                ),
            ),
            (
                "G2",
                quote(
                    "hoMassiveQTLAnalysis2021",
                    "paper.md",
                    "8913bfee34532387fcd246b02d837abc1e1ea94faca3e4f4eeacca1d02cc5187",
                    "segregants were obtained from a cross of two phenotypically divergent S. cerevisiae strains, RM11-1a, a natural vineyard isolate",
                    "genotype basis: segregants of RM11-1a x YJM975",
                ),
            ),
        ),
    ),
    AuditEntry(
        citation_key="sheMappingCausalVariants2018",
        zotero_item="6E5RS4AS",
        short="She 2018",
        category="f",
        schedule_row="She 2018 (natural variant effects across a large isolate panel)",
        hosts=(
            HostSpec(label="RM11-1a (1,011-panel isolate AAA)", token=PETER),
            HostSpec(label="YJM975 (1,011-panel isolate YBA)", token=PETER),
        ),
        mosaic=True,
        g3_wall="the per-segregant genotype and phenotype matrices were released only in a Dropbox folder that reads 'File Deleted' (2026-10-11T10:57:58Z); raw reads remain on SRA PRJNA387489. Recipe in #70",
        phenotype_class="EnvironmentResponsePhenotype",
        blocked_issue=70,
        records=None,
        records_unit="segregant x condition colony-area z-scores",
        records_how="not measurable: the Cell SI holds only Table S3 (727 causal-variant rows over 28 condition labels)",
        genotype_basis="segregants (1,125 F6 RM11-1a x YJM975), not the ~500 isolates the schedule row says",
        released=(
            Released(
                file="Dropbox 'additional data'", where="dropbox.com", access="vanished"
            ),
            Released(file="PRJNA387489 reads", where="SRA", access="scriptable"),
        ),
        overlap="none (bloom2019 crosses do not use YJM975)",
        summary="per-record data vanished with the authors' Dropbox; schedule row design is wrong",
        evidence=(
            (
                "G3",
                quote(
                    "sheMappingCausalVariants2018",
                    "paper.md",
                    "418f04347b9368eea84a22f8a6b190fd3d867d98ecdbce8a6c47f13490928b9b",
                    "All genotype and phenotype data are shared freely in the additional data",
                    "per-record data were deposited only in the 'additional data' (Dropbox) link",
                ),
            ),
            (
                "G2",
                quote(
                    "sheMappingCausalVariants2018",
                    "paper.md",
                    "418f04347b9368eea84a22f8a6b190fd3d867d98ecdbce8a6c47f13490928b9b",
                    "a vineyard in California (RM-11a) (Torok et al., 1996) and a patient in Italy (YJM975a)",
                    "cross parents",
                ),
            ),
            (
                "G2",
                quote(
                    "sheMappingCausalVariants2018",
                    "paper.md",
                    "418f04347b9368eea84a22f8a6b190fd3d867d98ecdbce8a6c47f13490928b9b",
                    "haploid progeny and performed whole-genome sequencing on clonal expansions of each individual",
                    "genotype basis: sequenced F6 segregants",
                ),
            ),
        ),
    ),
    AuditEntry(
        citation_key="zackrissonScanomaticHighResolutionMicrobial2016",
        zotero_item="QZ9CKZQQ",
        short="Zackrisson 2016",
        category="f",
        schedule_row="Zackrisson 2016 (Scan-o-matic growth curves)",
        g3_wall="the one genotype x environment screen (BY4742 YKO, glucose +/- 0.85 M NaCl, n = 3) appears only as figure points; the 21 SI files are figures, a methods PDF and movies. Recipe: author request (#999)",
        phenotype_class="EnvironmentResponsePhenotype",
        blocked_issue=999,
        records=None,
        records_unit="strain x condition log2 relative doubling times",
        records_how="not measurable: no per-strain file in the release",
        genotype_basis="haploid YKO (BY4742)",
        released=(
            Released(
                file="21 SI files (figures, File S1, movies)",
                where="mirror (PMC open-data bucket)",
                access="scriptable",
            ),
        ),
        overlap="none held",
        summary="a platform paper; its demonstration screen is figure-only",
        evidence=(
            (
                "G3",
                quote(
                    "zackrissonScanomaticHighResolutionMicrobial2016",
                    "paper.md",
                    "4dc6fecc7e422221b95772827e47ade65b055dfb8c4bc9e690945cdd77feac12",
                    "The authors state that all data necessary for confirming the conclusions presented in the article are represented fully within the article.",
                    "data availability: no deposited dataset",
                ),
            ),
            (
                "G3",
                quote(
                    "zackrissonScanomaticHighResolutionMicrobial2016",
                    "paper.md",
                    "4dc6fecc7e422221b95772827e47ade65b055dfb8c4bc9e690945cdd77feac12",
                    "colonies corresponding to single yeast gene knockouts of the haploid BY4742 deletion collection (Giaever et al. 2002), with WT control colonies interleaved in every fourth position",
                    "genotype basis of the only genotype x environment screen",
                ),
            ),
        ),
    ),
    AuditEntry(
        citation_key="lianCombinatorialMetabolicEngineering2017",
        zotero_item="WFBK2TWS",
        short="Lian 2017",
        category="f",
        schedule_row="Lian 2017 (CRISPR-AID, beta-carotene)",
        hosts=(HostSpec(label="CEN.PK2-1C", token="CEN.PK"),),
        g3_wall="the guide genotypes are released (SI Tables 3-5) but beta-carotene and cellulase values are bar heights only; the data statement routes the rest to the corresponding author. Recipe: author request (#997)",
        phenotype_class="MetabolitePhenotype",
        blocked_issue=997,
        records=None,
        records_unit="strain x titer records",
        records_how="not measurable: pdftotext of si/si1.pdf; no table carries a phenotype number",
        genotype_basis="engineered strain: CEN.PK2-1C with the CRISPR-AID cassette, combinatorial a/i/d guides",
        released=(
            Released(
                file="si/si1.pdf (Supplementary Information); si/si2.pdf is the peer-review file",
                where="mirror (PMC open-data bucket)",
                sha256="9f81bc16c67ef2a0fa1bad6345fdc742d0af879d59071118bf9683d5ac05d930",
                access="scriptable",
            ),
        ),
        overlap="none: crispr_magic_lian2019 is the same platform on BY4742 with guide-enrichment records",
        summary="figure-only values; CEN.PK also has no assembly set",
        evidence=(
            (
                "G2",
                quote(
                    "lianCombinatorialMetabolicEngineering2017",
                    "paper.md",
                    "c0ef82dbfdf1f577b87428253cf8207988bad0c9e2927ab8ccb67878f71943d1",
                    "S. cerevisiae CEN.PK2-1C strain (EUROSCARF, Frankfurt, Germany) was used as the host",
                    "host strain / genotype basis",
                ),
            ),
            (
                "G3",
                quote(
                    "lianCombinatorialMetabolicEngineering2017",
                    "paper.md",
                    "c0ef82dbfdf1f577b87428253cf8207988bad0c9e2927ab8ccb67878f71943d1",
                    "available within the article and its Supplementary Information file or from the corresponding author upon reasonable request",
                    "data availability: no deposited per-record values, the rest is on author request (classification f)",
                ),
            ),
        ),
    ),
    AuditEntry(
        citation_key="deloacheEnzymecoupledBiosensorEnables2015",
        zotero_item="WV48S84B",
        short="DeLoache 2015",
        category="f",
        schedule_row=None,
        row=RowSpec(
            name="DeLoache 2015 (betaxanthin biosensor for (S)-reticuline)",
            klass="Metabolite / precursor",
            seq_basis="engineered-chassis",
            url="https://doi.org/10.1038/nchembio.1816",
            accession="Nat Chem Biol SI",
            text="base strain BY4741",
        ),
        g3_wall="SI Table 1 lists 54 strains with construct GenBank accessions and 0 phenotype values; fluorescence and LC/MS titers are figure-only and the mutant library was screened by eye. Recipe: author request (#996)",
        phenotype_class="MetabolitePhenotype",
        blocked_issue=996,
        records=None,
        records_unit="strain x signal records",
        records_how="not measurable: 54 strain rows in SI Table 1 (pdftotext), none with a value",
        genotype_basis="engineered strain: BY4741 with integrated heterologous BIA pathway constructs and CYP76AD1 point mutants",
        released=(
            Released(
                file="si/si1.pdf (Supplementary Text and Figures)",
                where="mirror (Zotero attachment)",
                sha256="fa5922dc23976e7c9edbf4cdce8e37141ff1074196191b80804760e7c902779b",
                access="unknown",
            ),
        ),
        overlap="none: betaxanthin_cachera2023 is a deletion-collection screen",
        summary="a method paper with a 54-strain panel and figure-only values",
        evidence=(
            (
                "G2",
                quote(
                    "deloacheEnzymecoupledBiosensorEnables2015",
                    "paper.md",
                    "9eef4f50c4b49de068087ace81fd5502db09ba5d5daeb3846a3d64ba4ee86fec",
                    "The base S. cerevisiae strain for all experiments in this article was BY4741",
                    "host strain",
                ),
            ),
            (
                "G3",
                quote(
                    "deloacheEnzymecoupledBiosensorEnables2015",
                    "paper.md",
                    "9eef4f50c4b49de068087ace81fd5502db09ba5d5daeb3846a3d64ba4ee86fec",
                    "the most intensely yellow colonies were isolated for further analysis",
                    "the variant library was screened by visual inspection, so no per-variant signal was ever recorded for the library",
                ),
            ),
            (
                "G3",
                quote(
                    "deloacheEnzymecoupledBiosensorEnables2015",
                    "paper.md",
                    "9eef4f50c4b49de068087ace81fd5502db09ba5d5daeb3846a3d64ba4ee86fec",
                    "We narrowed our set of hits to six by removing duplicate genotypes",
                    "only six first-round mutants carry a released genotype, which is why the loadable unit is a strain panel and not a library",
                ),
            ),
        ),
    ),
    AuditEntry(
        citation_key="trikkaIterativeCarotenogenicScreens2015",
        zotero_item="3WAM7IER",
        short="Trikka 2015",
        category="f",
        schedule_row="Trikka 2015 (carotenogenic heterozygous screen)",
        hosts=(
            HostSpec(label="BY4741 deletion library (S288C)", token="S288C"),
            HostSpec(label="EG60 (carotenogenic parent)", token=None),
        ),
        g3_wall="Additional file 1 is a 100-gene list with 0 numeric cells; the about 4,700-cross screen was a visual selection never scored per strain, and carotenoid amounts are figure-only. Recipe: author request (#995)",
        phenotype_class="VisualScorePhenotype",
        blocked_issue=995,
        records=None,
        records_unit="per-cross carotenoid scores",
        records_how="not measurable: si/si1.doc text extracted; 100 ORF cells, 0 numeric cells",
        genotype_basis="heterozygous EG60 x BY4741 hybrids (not BY4743)",
        released=(
            Released(
                file="si/si1.doc (Additional file 1)",
                where="mirror (Springer ESM)",
                sha256="e5587f1eeb0148bca6d7c54bbc4038aaf0b495bbdfe6ad9d3a75c2c58e715b9f",
                access="scriptable",
            ),
        ),
        overlap="none: carotenoid_ozaydin2013 is a different screen on haploid BY4741",
        summary="hit list only; EG60 has no assembly set",
        evidence=(
            (
                "G2",
                quote(
                    "trikkaIterativeCarotenogenicScreens2015",
                    "paper.md",
                    "b48ed2c4eb19263f6f01e103cafc1e8da6678a79082e0269a303fa3ba8d76e9a",
                    "This strain was created by transforming EG60 wild type Mat α haploid cells with the linearized vector YEplac195-YB/I/E",
                    "screening parent: EG60 background, so diploids are EG60 x BY4741 hybrids",
                ),
            ),
            (
                "G3",
                quote(
                    "trikkaIterativeCarotenogenicScreens2015",
                    "paper.md",
                    "b48ed2c4eb19263f6f01e103cafc1e8da6678a79082e0269a303fa3ba8d76e9a",
                    "Additional file 1: Table S1. List of gene deletions selected by the carotenoid screen.",
                    "Additional file 1 is a selected-gene list",
                ),
            ),
            (
                "G3",
                quote(
                    "trikkaIterativeCarotenogenicScreens2015",
                    "paper.md",
                    "b48ed2c4eb19263f6f01e103cafc1e8da6678a79082e0269a303fa3ba8d76e9a",
                    "Visual inspection of the crosses revealed a palette of red color intensities.",
                    "the screen readout was visual inspection, not a released score",
                ),
            ),
        ),
    ),
    # ---------------- (c) not a dataset ----------------
    AuditEntry(
        citation_key="brettnerUltraHighthroughputMassively2024",
        zotero_item="VP8TSA3J",
        short="Brettner 2024",
        category="c",
        schedule_row="Brettner 2024 (ultra-high-throughput yeast scRNA-seq)",
        records=None,
        records_unit="cells",
        records_how="not pursued: GEO GSE251966 holds unfiltered matrices of three technical validation runs",
        genotype_basis="wild-type S288C / BY4741 and two reporter strains, no perturbation axis",
        released=(Released(file="GSE251966", where="GEO", access="scriptable"),),
        overlap="none",
        summary="method (yeast SPLiT-seq); its value is as the chemistry a Perturb-seq proposal names, as schedule row 30 says",
    ),
    AuditEntry(
        citation_key="wagihSGAtoolsOnestopAnalysis2013",
        zotero_item="I45F26KL",
        short="Wagih 2013",
        category="c",
        schedule_row=None,
        records=None,
        records_unit="",
        records_how="tool paper",
        genotype_basis="none",
        overlap="none",
        summary="SGAtools, a web tool for scoring colony-size screens",
    ),
    AuditEntry(
        citation_key="kuzminSyntheticGeneticArray2016",
        zotero_item="WKGW6AJH",
        short="Kuzmin 2016",
        category="c",
        schedule_row=None,
        records=None,
        records_unit="",
        records_how="protocol",
        genotype_basis="none",
        overlap="none",
        summary="SGA protocol (Cold Spring Harb Protoc)",
    ),
    AuditEntry(
        citation_key="yantongSyntheticGeneticArray2006",
        zotero_item="ZXC7VLW7",
        short="Tong 2006",
        category="c",
        schedule_row=None,
        records=None,
        records_unit="",
        records_how="protocol (methods chapter; the Zotero item has no DOI)",
        genotype_basis="none",
        overlap="none",
        summary="SGA methods chapter",
    ),
    AuditEntry(
        citation_key="engelNewDataCollaborations2022",
        zotero_item="H47A584B",
        short="Engel 2022",
        category="c",
        schedule_row=None,
        records=None,
        records_unit="",
        records_how="database update paper",
        genotype_basis="none",
        overlap="SGD content is already consumed: GeneEssentialitySgdDataset (sgd.py, which cites no release paper) and the genomes-tier set sgd_S288C_R64-4-1_20230830",
        summary="SGD release paper; nothing new to load, a citation for sgd.py",
    ),
    AuditEntry(
        citation_key="wangSynLethDB20Webbased2022",
        zotero_item="VI2GJEBI",
        short="Wang 2022 (SynLethDB 2.0)",
        category="c",
        schedule_row=None,
        records=None,
        records_unit="",
        records_how="database paper of a served aggregation",
        genotype_basis="none",
        overlap="already loaded: synth_leth_db.py names it SYNLETHDB_CITATION_KEY; dev stores synth_lethality_yeast_synth_leth_db and synth_rescue_yeast_synth_leth_db. Its aggregation verdict is the piece-3 backfill, not this audit",
        summary="the release paper of the served SynLethDB aggregation (not missed)",
    ),
    AuditEntry(
        citation_key="longEnzymeEngineeringDatabase2026",
        zotero_item="DKNER94S",
        short="Long 2026 (EnzEngDB)",
        category="c",
        schedule_row=None,
        records=None,
        records_unit="",
        records_how="database paper",
        genotype_basis="none",
        overlap="none",
        summary="enzyme sequence-function database across organisms; no cell genotype x environment records",
    ),
    AuditEntry(
        citation_key="yuanOpenEnzymeDatabase2026",
        zotero_item="3IL3E6NI",
        short="Yuan 2026 (OED)",
        category="c",
        schedule_row=None,
        records=None,
        records_unit="",
        records_how="database paper",
        genotype_basis="none",
        overlap="none",
        summary="enzyme kinetics repository; no cell genotype x environment records",
    ),
    # ---------------- (g) wrong organism ----------------
    AuditEntry(
        citation_key="aranguizMachineLearningReveals2025",
        zotero_item="KE773CUE",
        short="Aranguiz 2025",
        category="g",
        schedule_row=None,
        records=15,
        records_unit="S. cerevisiae strain x condition values",
        records_how="si/si12.xlsx source data: 5 S288C deletion strains x 3 oxidants; the screen itself is 284 species x 2 doses (si/si3.xlsx), one value per species",
        genotype_basis="a species panel (284 Saccharomycotina species); S. cerevisiae only as 5 deletion strains",
        overlap="none",
        summary="a cross-species screen, not an S. cerevisiae genotype panel; 15 S. cerevisiae values are not worth a loader",
    ),
    AuditEntry(
        citation_key="suthersGenomescaleMetabolicReconstruction2020",
        zotero_item="USUXQJR3",
        short="Suthers 2020",
        category="g",
        schedule_row=None,
        records=None,
        records_unit="",
        records_how="genome-scale model paper",
        genotype_basis="none",
        overlap="none",
        summary="Issatchenkia orientalis SD108 (Pichia kudriavzevii) model; likely mis-filed, the empty collection database/Pichia-kudriavzevii exists (moving it is the owner's call)",
    ),
)


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #
def candidate_row(entry: AuditEntry, rows: dict[str, CandidateRow]) -> CandidateRow:
    """The table row for a ranked key, else the row built from its ``RowSpec``."""
    if entry.schedule_row is not None:
        return rows[entry.schedule_row]
    if entry.row is None:
        raise ValueError(f"{entry.citation_key}: off-schedule without a RowSpec")
    spec = entry.row
    return CandidateRow(
        table="yeast",
        name=spec.name,
        organism="S. cerevisiae",
        status="candidate",
        klass=spec.klass,
        seq_basis=spec.seq_basis,
        url=spec.url,
        accession=spec.accession,
        accession_confirmed=None,
        text=spec.text,
    )


def composed_host(spec: HostSpec, data_root: str) -> G2Host:
    """A named host: probed through its ``gates.HOSTS`` token, or absent from the tier."""
    if spec.token is None:
        return G2Host(
            name=spec.label, assembly_set=None, resolvable=False, readable=False
        )
    probe = gates.probe_host(spec.token, data_root, verify=False)
    return G2Host(
        name=spec.label,
        assembly_set=probe.assembly_set,
        resolvable=probe.resolvable,
        readable=probe.readable,
    )


def worst_outcome(hosts: tuple[G2Host, ...]) -> G2Outcome:
    """The worst host decides, as ``G2Record`` checks it."""
    if any(h.assembly_set is None or not h.resolvable for h in hosts):
        return "absent_provisionable"
    if any(not h.readable for h in hosts):
        return "resolvable_not_ingestible"
    return "resolvable_readable"


def composed_g2(entry: AuditEntry, data_root: str) -> GateRun:
    """G2 over the hosts the paper names (the row text names none)."""
    assert entry.hosts is not None
    hosts = tuple(composed_host(h, data_root) for h in entry.hosts)
    record = G2Record(
        branch="haplotype_mosaic" if entry.mosaic else "assembly_set",
        outcome=worst_outcome(hosts),
        hosts=hosts,
    )
    phrases = "; ".join(
        f"{h.name} -> {h.assembly_set}" if h.assembly_set else f"{h.name}: no tier set"
        for h in hosts
    )
    outcome, kind = gates.G2_GATE[record.outcome]
    return GateRun(
        result=GateResult(
            gate="G2",
            outcome=outcome,
            blocked_kind=kind,
            reason=f"{record.branch} (composed: hosts named by the paper): {phrases}",
        ),
        record=record,
    )


def composed_g3(entry: AuditEntry, inventory: InventoryRecord | None) -> GateRun:
    """G3 blocked:wall: per-record values that no retriever can reach, and the recipe."""
    assert entry.g3_wall is not None
    record = InventoryRecord(
        citation_key=entry.citation_key,
        trees_read=inventory.trees_read if inventory is not None else ("library",),
        files=inventory.files if inventory is not None else (),
        walls=(entry.g3_wall,),
    )
    return GateRun(
        result=GateResult(
            gate="G3",
            outcome="blocked",
            blocked_kind="wall",
            reason=f"composed: {entry.g3_wall}",
        ),
        record=record,
    )


def composed_g5(entry: AuditEntry) -> GateRun:
    """G5 gap beside an existing phenotype class (a genotype-side or field-level gap)."""
    assert entry.g5_gap is not None and entry.phenotype_class is not None
    record = G5Record(
        phenotype_class=entry.phenotype_class,
        class_exists=gates.phenotype_class_exists(entry.phenotype_class),
        proposed_class=entry.proposed_class,
    )
    return GateRun(
        result=GateResult(
            gate="G5",
            outcome="gap",
            issue=entry.issue,
            reason=f"composed: {entry.g5_gap}",
        ),
        record=record,
    )


def with_evidence(verdict: CandidateVerdict, entry: AuditEntry) -> CandidateVerdict:
    """The verdict with the entry's quotes added to the measured gates they support."""
    data = verdict.model_dump()
    for gate in data["gates"]:
        if gate["outcome"] == "unmeasured":
            continue
        extra = [v.model_dump() for g, v in entry.evidence if g == gate["gate"]]
        gate["evidence"] = [*gate["evidence"], *extra]
    return CandidateVerdict.model_validate(data)


def gate_entry(
    entry: AuditEntry,
    rows: dict[str, CandidateRow],
    index: tuple[gates.ModuleKeys, ...],
    root: str,
    commit: str,
    today: str,
) -> CandidateVerdict:
    """G1 to G5 for one key with the stop rule, as the CLI's ``gate`` composes them."""
    row = candidate_row(entry, rows)
    g1 = gates.evaluate_g1(row, entry.finding)
    runs: dict[GateName, GateRun] = {}
    steps = {
        "G2": lambda: (
            composed_g2(entry, root)
            if entry.hosts is not None
            else gates.evaluate_g2(row, root, source_kind=g1.source_kind, verify=False)
        ),
        "G3": lambda: (
            composed_g3(entry, gates.inventory(entry.citation_key, root))
            if entry.g3_wall is not None
            else gates.evaluate_g3(gates.inventory(entry.citation_key, root))
        ),
        "G4": lambda: gates.evaluate_g4_key(
            row, entry.citation_key, index, doi_to_pmid=None, aggregator_pmids=None
        ),
    }
    if entry.g5_gap is not None:
        steps["G5"] = lambda: composed_g5(entry)
    elif entry.phenotype_class is not None:
        steps["G5"] = lambda: gates.evaluate_g5(
            entry.phenotype_class,  # type: ignore[arg-type]
            (),
            issue=entry.issue,
            proposed_class=entry.proposed_class,
        )
    stopped = g1.result.outcome in ("fail", "blocked")
    for name, step in steps.items():
        if stopped:
            break
        run = step()
        runs[name] = run  # type: ignore[index]
        stopped = run.result.outcome in ("fail", "blocked")
    verdict = gates.compose_verdict(
        row=row,
        citation_key=entry.citation_key,
        g1=g1,
        runs=runs,
        zotero=gates.zotero_item(entry.citation_key, root),
        decided_at=today,
        torchcell_commit=commit,
    )
    return with_evidence(verdict, entry)


# --------------------------------------------------------------------------- #
# Note tables
# --------------------------------------------------------------------------- #
def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _count(entry: AuditEntry) -> str:
    if entry.records is None:
        return "not measured"
    return f"{entry.records:,} {entry.records_unit}"


def _rank(entry: AuditEntry, ranks: dict[str, int]) -> str:
    if entry.schedule_row is None:
        return "off-schedule"
    return f"#{ranks[entry.schedule_row]}"


def _mirror(entry: AuditEntry) -> str:
    return f"`$DATA_ROOT/torchcell-library/{entry.citation_key}/`"


def _outcome(entry: AuditEntry, verdicts: dict[str, CandidateVerdict]) -> str:
    verdict = verdicts.get(entry.citation_key)
    return verdict.outcome if verdict is not None else "no verdict"


def _issue(entry: AuditEntry) -> str:
    number = entry.issue if entry.issue is not None else entry.blocked_issue
    return f"#{number}" if number is not None else ""


def render_tables(verdicts: dict[str, CandidateVerdict], ranks: dict[str, int]) -> str:
    """The note's tables: loadable now, schema-gated, held, and not a dataset."""
    by_count = lambda e: (-(e.records or 0), e.citation_key)  # noqa: E731
    loadable = sorted((e for e in AUDIT if e.category == "a"), key=by_count)
    gated = sorted((e for e in AUDIT if e.category == "b"), key=by_count)
    held = [e for e in AUDIT if e.category in ("e", "f")]
    rest = [e for e in AUDIT if e.category in ("c", "d", "g")]
    out = [
        "### Loadable now\n",
        "| rank | key | Zotero item | schedule | records (measured) | genotype basis "
        "| phenotype class | overlap risk | verdict | mirror |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, e in enumerate(loadable, start=1):
        out.append(
            f"| {i} | {e.citation_key} | {e.zotero_item} | {_rank(e, ranks)} | "
            f"{_cell(_count(e))} | {_cell(e.genotype_basis)} | {e.phenotype_class} | "
            f"{_cell(e.overlap)} | {_outcome(e, verdicts)} | {_mirror(e)} |"
        )
    out += [
        "\n### Schema-gated\n",
        "| key | Zotero item | schedule | records (measured) | genotype basis | gap | "
        "issue | verdict | mirror |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in gated:
        gap = e.g5_gap or f"no {e.phenotype_class}"
        out.append(
            f"| {e.citation_key} | {e.zotero_item} | {_rank(e, ranks)} | "
            f"{_cell(_count(e))} | {_cell(e.genotype_basis)} | {_cell(gap)} | "
            f"{_issue(e)} | {_outcome(e, verdicts)} | {_mirror(e)} |"
        )
    out += [
        "\n### Aggregation and blocked at source\n",
        "| key | Zotero item | schedule | category | records (measured) | why | issue "
        "| verdict | mirror |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in held:
        out.append(
            f"| {e.citation_key} | {e.zotero_item} | {_rank(e, ranks)} | "
            f"{CATEGORY_LABEL[e.category]} | {_cell(_count(e))} | {_cell(e.summary)} | "
            f"{_issue(e)} | {_outcome(e, verdicts)} | {_mirror(e)} |"
        )
    out += [
        "\n### Not a dataset, wrong organism, or genomes tier\n",
        "| key | Zotero item | schedule | category | why | verdict | mirror |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in rest:
        out.append(
            f"| {e.citation_key} | {e.zotero_item} | {_rank(e, ranks)} | "
            f"{CATEGORY_LABEL[e.category]} | {_cell(e.summary)} | "
            f"{_outcome(e, verdicts)} | {_mirror(e)} |"
        )
    return "\n".join(out) + "\n"


def render_details(verdicts: dict[str, CandidateVerdict]) -> str:
    """Per key: the count and how it was measured, released artifacts, overlap, gates."""
    out: list[str] = []
    for e in AUDIT:
        out.append(f"#### {e.short} (`{e.citation_key}`)\n")
        out.append(f"- category: {e.category} ({CATEGORY_LABEL[e.category]})")
        out.append(f"- records: {_count(e)}; {e.records_how}")
        out.append(f"- genotype basis: {e.genotype_basis}")
        out.append(f"- overlap: {e.overlap}")
        for r in e.released:
            sha = f"`{r.sha256}`" if r.sha256 else "bytes not read"
            out.append(f"- released: {r.file} ({r.where}; {r.access}; {sha})")
        verdict = verdicts.get(e.citation_key)
        if verdict is not None:
            gates_line = ", ".join(
                f"{g.gate} {g.outcome}{':' + g.blocked_kind if g.blocked_kind else ''}"
                for g in verdict.gates
            )
            out.append(f"- verdict: {verdict.outcome} ({gates_line})")
        out.append("")
    return "\n".join(out)


def git_commit() -> str:
    """HEAD of the tree the gates read."""
    return subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def main() -> None:
    """Gate every dataset-candidate key; print, or with --write store, the results."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    load_dotenv(REPO / ".env")
    root = os.environ["DATA_ROOT"]
    table = load_table_module("yeast")
    rows = {r.name: r for r in table_rows("yeast", table)}
    ranked, _ = table.ranked()
    ranks = {c.name: i for i, c in enumerate(ranked, start=1)}
    index = gates.module_index()
    commit = git_commit()
    today = date.today().isoformat()
    verdicts: dict[str, CandidateVerdict] = {}
    for entry in AUDIT:
        if entry.category not in GATED:
            continue
        verdict = gate_entry(entry, rows, index, root, commit, today)
        verdicts[entry.citation_key] = verdict
        print(f"{entry.citation_key}: {verdict.outcome}")
        for g in verdict.gates:
            print(f"  {g.gate} {g.outcome}: {g.reason[:160]}")
        if args.write:
            print(f"  wrote {store.write_verdict(verdict)}")
    tables = render_tables(verdicts, ranks) + "\n" + render_details(verdicts)
    if args.write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        TABLES_OUT.write_text(tables, encoding="utf-8")
        print(f"wrote {TABLES_OUT}")
    else:
        print(tables)


if __name__ == "__main__":
    main()
