# torchcell/datasets/pputida/thompson2019_valerolactam
# [[torchcell.datasets.pputida.thompson2019_valerolactam]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/pputida/thompson2019_valerolactam
# Test file: tests/torchcell/datasets/pputida/test_thompson2019_valerolactam.py
"""Thompson 2019 valerolactam catabolism and production in P. putida KT2440.

Thompson et al. 2019 (Metab. Eng. Commun. 9:e00098, doi:10.1016/j.mec.2019.e00098)
found that KT2440 grows on valerolactam, identified the lactam hydrolase ``oplBA`` by
secretome proteomics after RB-TnSeq failed to, and deleted ``oplBA``, ``davT`` and
``alr`` in a lysine-fed producer carrying ``pBADT-davBA-ORF26``. The paper releases
three kinds of measurement, and this module measured each against what is already
served before writing anything:

- **RB-TnSeq fitness: SUBSUMED, no records here.** The Methods describe one condition
  (10 mM valerolactam as the sole carbon source) and the Results name "two valerolactam
  RB-TnSeq experiments". Both are samples of the Borchert 2024 compendium that
  ``RbTnseqBorchert2024Dataset`` already serves: ``set6IT064`` (library ``Putida_ML5``,
  48-well Tecan Infinite F200, the vessel and reader the Methods state) and
  ``set7IT045`` (``Putida_ML5_JBEI``), both ``2-Piperidinone`` 10 mM in the carbon-source
  group. The paper's own release is the Fitness Browser alone. The 5-aminovalerate
  samples in Fig. S2 are NOT this paper's: "All non-valerolactam fitness experiments are
  from Thompson et. al 2019", the lysine paper (schedule row 55). The measurement is
  ``experiments/036-dataset-fixes-before-kg-build/scripts/
  thompsonOmicsdrivenIdentificationElimination2019_release_inventory.py``.
- **Valerolactam titer: one dataset class here,
  :class:`ValerolactamTiterThompson2019Dataset`.** The Results state eight titers, four
  strains at 24 and 48 h, in prose (Fig. 3B has no source-data file).
- **Maximal growth rate: one dataset class here,
  :class:`LactamGrowthRateThompson2019Dataset`.** Supplementary Table S1 releases nine
  rates, three strains on three carbon sources.

FOUR TITERS ARE STORED AND FOUR ARE REFUSED, AND THE REFUSAL IS A SCHEMA FINDING. The 24 h
family stores all four strains, and its reference is the wild-type producer's own 24 h
titer (0.43 mg/L). The 48 h family has no reference number: the paper states "no
valerolactam could be detected after 48 h" for the wild type, which is a value below
the assay's detection floor, not a zero and not a missing measurement.
``ProductTiterExperimentReference`` requires a ``phenotype_reference`` whose ``titer`` is
a required non-negative float, and ``ProductTiterPhenotype`` has no censoring field, so
the 48 h wild type cannot be written as anything true and the three 48 h engineered
titers (9.27, 85.19, 91.97 mg/L) have no denominator. Writing 0.0 would state a
measurement nobody made; borrowing the 24 h reference would compare two different
sampling times. The Kang 2026 and Yunus 2026 loaders refuse a titer family without a
released reference for the same reason. The four refusals are counted in
``preprocess/build_accounting.json`` and the schema gap is filed as an issue.

THE davT LOCUS IS NOT IN THE PAPER OR THE ANNOTATION, AND THE GENOMES TIER CLOSES IT.
The paper names no locus tag anywhere. ``oplB``, ``oplA``, ``alr``, ``davB`` and ``davA``
resolve through the pinned GenBank annotation (``PP_3514``, ``PP_3515``, ``PP_3722``,
``PP_0383``, ``PP_0382``). ``davT`` does not ("not found in GCA_000007565.2_ASM756v2").
The genomes tier's pinned UniProt GOA proteome file for the same assembly set carries
``davT`` as the DB-object symbol of exactly one protein, with the locus tag ``PP_0214``
in its synonym column and the product name "5-aminovalerate aminotransferase DavT",
which is the enzyme the paper describes ("davT, which catalyzes the first step in 5AVA
catabolism"). :func:`goa_symbol_locus` reads that file and refuses anything but one
locus.

THE PLASMID. ``pBADT-davBA-ORF26`` carries ``davB`` and ``davA``, "two genes endogenous
to P. putida", and ``ORF26``, "a promiscuous acyl-coA ligase from Streptomyces
aizunensis", under "an arabinose inducible promoter" on "a pBBR ori plasmid". Each is a
``HeterologousPathwayPerturbation``: the two native genes carry their KT2440 locus tag
and ``is_heterologous=False`` (an extra native copy), ``ORF26`` its source name.

THE GROWTH MEDIUM IS THIS PAPER'S OWN RECIPE. The Methods write out a "modified MOPS
minimal medium" component by component (LaBauve and Wargo 2012). It differs from the
library's ``MOPS_MINIMAL`` (Neidhardt, as Price 2018 tabulates it) in stated amounts,
most visibly 32.5 uM against 0.5 uM calcium chloride, and it adds 8 uM iron(II)
chloride. Serving ``MOPS_MINIMAL`` would misstate the recipe, so
:data:`MOPS_MODIFIED_THOMPSON2019` is built here, kept out of ``MEDIA_LIBRARY`` (the
Ishii 2007 and Choe 2019 pattern), with ``base_medium="MOPS_MINIMAL"`` so it still joins
the MOPS family.

QUOTES. Every quoted statement is verbatim in the PMC full text
(``paper/PMC6838509.1.txt``) or in the text layer of the Supplementary Information PDF
(``si/mmc1.txt``, rendered by pypdf from ``si/mmc1.pdf``); both are sha256-pinned in the
raw mirror ``$DATA_ROOT/torchcell-raw/thompsonOmicsdrivenIdentificationElimination2019/``,
which this module's :func:`deposit_raw_mirror` writes. The paper is in neither Zotero
library, so the literature mirror holds nothing for this key.
"""

from __future__ import annotations

import json
import logging
import os
import os.path as osp
import re
import shutil
from collections.abc import Callable, Iterable
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
    verify_sha256,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import LB
from torchcell.datamodels.schema import (
    AssayType,
    BacterialDeletionPerturbation,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    BacterialGeneNamespace,
    BacterialReferenceStrain,
    Compound,
    Concentration,
    ConcentrationUnit,
    CultureEnvironment,
    CultureFormat,
    EndpointRule,
    Environment,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    HeterologousPathwayPerturbation,
    MeasurementType,
    Media,
    MediaComponent,
    MediaComponentRole,
    PhysicalFactor,
    ProductTiterExperiment,
    ProductTiterExperimentReference,
    ProductTiterPhenotype,
    Publication,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.bacteria_common import (
    LocusTagReconciliation,
    assembly_reference,
    bacterial_genome,
    reconcile_locus_tags,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_PAPER_PDF,
    ROLE_PAPER_TEXT,
    ROLE_SI_PDF,
    ROLE_SI_TEXT,
    ArtifactRecord,
    Manifest,
    ProcessingRecord,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.literature.retrieve import pmc_cloud_url
from torchcell.sequence.genome.bacterial import GAF_COLUMNS
from torchcell.sequence.genome.pputida.kt2440 import PPutidaKT2440Genome
from torchcell.sequence.genome.registry import load_genome_manifest, resolve
from torchcell.verification.levels import (
    l0_structural,
    l1_count,
    l2_value_fidelity,
    l3_convention,
    l4_cross_source,
)
from torchcell.verification.report import Provenance, VerificationReport, sha256_file
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

log = logging.getLogger(__name__)

DOI = "10.1016/j.mec.2019.e00098"
PMCID = "PMC6838509"
PUBMED_ID = "31720214"
TITLE = (
    "Omics-driven identification and elimination of valerolactam catabolism in "
    "Pseudomonas putida KT2440 for increased product titer"
)
CITATION_KEY = "thompsonOmicsdrivenIdentificationElimination2019"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
#: PMC Article Datasets bucket prefix of the article's one open-access version.
PMC_PREFIX = f"{PMCID}.1"
RETRIEVED_AT = "2026-10-10"
#: Where the strains and plasmids are deposited (not retrievable by script).
JBEI_REGISTRY_FOLDER = "https://public-registry.jbei.org/folders/456"
#: The paper's only release of its RB-TnSeq values.
FITNESS_BROWSER = "https://fit.genomics.lbl.gov/cgi-bin/org.cgi?orgId=Putida"

KT2440_NAMESPACE: BacterialGeneNamespace = "pputida_kt2440_locus_tag"
REFERENCE_STRAIN: BacterialReferenceStrain = "KT2440"

#: Version of pypdf that rendered ``si/mmc1.txt`` from ``si/mmc1.pdf``.
SI_TEXT_PYPDF_VERSION = "6.19.0"
#: How the SI text layer was rendered: every page's ``extract_text()``, joined by "\n".
SI_TEXT_PAGE_SEPARATOR = "\n"


class RawFile(BaseModel):
    """One file of the raw mirror: its pinned bytes, role and origin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relpath: str
    bucket_name: str | None
    role: str
    sha256: str
    bytes: int
    description: str

    @property
    def bucket_key(self) -> str:
        """The PMC Article Datasets bucket key the bytes came from."""
        if self.bucket_name is None:
            raise ValueError(f"{self.relpath} is derived, not retrieved")
        return f"{PMC_PREFIX}/{self.bucket_name}"


PAPER_TEXT = RawFile(
    relpath="paper/PMC6838509.1.txt",
    bucket_name="PMC6838509.1.txt",
    role=ROLE_PAPER_TEXT,
    sha256="f0244882bdfc6c4789079dbe76a14c47be04acb3a7ab6083f5a80f8f23c2da84",
    bytes=46473,
    description="The article's full text as PMC renders it, the anchor of every paper "
    "quote in this module. The publisher's own text, not OCR, so a quote matches "
    "character for character including the narrow no-break spaces before units",
)
PAPER_XML = RawFile(
    relpath="paper/PMC6838509.1.xml",
    bucket_name="PMC6838509.1.xml",
    role=ROLE_PAPER_TEXT,
    sha256="0a1f88d7987bfe5b0f71eb14ced2987aa22f5c20413a45c02ed6d4e257cb7ed4",
    bytes=97776,
    description="The article's JATS XML as PMC serves it; the structured form of the "
    "same text, mirrored because it carries Table 1 as a table",
)
PAPER_PDF = RawFile(
    relpath="paper/PMC6838509.1.pdf",
    bucket_name="PMC6838509.1.pdf",
    role=ROLE_PAPER_PDF,
    sha256="ed6c14563b2b48d10f269b1b1abc624b4cbfbd634c49cbf2e19b4a1b53238b2f",
    bytes=1194524,
    description="The article PDF as PMC serves it. Mirrored because the paper is in "
    "neither Zotero library, so the literature mirror holds no paper.pdf for this key",
)
SI_PDF = RawFile(
    relpath="si/mmc1.pdf",
    bucket_name="mmc1.pdf",
    role=ROLE_SI_PDF,
    sha256="73032e933ca57b6a0d20badd8dea04b89ba27d6039c2748b1d6f034085a4483f",
    bytes=2303927,
    description="Appendix A, 'Lactam Production in Putida revision SI_V2': Figures S1 "
    "to S5 and Table S1, the specific growth rates. The paper's only supplementary file",
)
SI_TEXT = RawFile(
    relpath="si/mmc1.txt",
    bucket_name=None,
    role=ROLE_SI_TEXT,
    sha256="197a98d3c122da930a6b4ae3b000767950e626d48f194487d07eefe7008067b1",
    bytes=3760,
    description="The text layer of si/mmc1.pdf, rendered by pypdf (every page's "
    "extract_text joined by a newline). Table S1 is read from it and the SI quotes are "
    "cut from it; the PDF stays the canonical artifact",
)
RAW_FILES: tuple[RawFile, ...] = (PAPER_TEXT, PAPER_XML, PAPER_PDF, SI_PDF, SI_TEXT)
RETRIEVED_FILES: tuple[RawFile, ...] = tuple(
    raw for raw in RAW_FILES if raw.bucket_name is not None
)


def raw_file(relpath: str) -> RawFile:
    """The ``RawFile`` of one mirror path, read from :data:`RAW_FILES` at call time.

    A dataset names the files it reads by PATH rather than holding the record objects,
    so a run whose pins are pointed at other bytes (a synthetic mirror in a test) reads
    one set of pins everywhere instead of two.
    """
    for raw in RAW_FILES:
        if raw.relpath == relpath:
            return raw
    raise KeyError(f"{relpath} is not a file of this key's raw mirror")


# --------------------------------------------------------------------------- #
# Verbatim quotes (PMC full text, or the SI text layer)
# --------------------------------------------------------------------------- #
_Q_WT_OPLBA = (
    "Wild type P.\xa0putida produced 0.43 mg/L valerolactam after 24 h, but "
    "no valerolactam could be detected after 48 h, presumably due to host "
    "consumption (Fig.\xa03B). Simple deletion of the oplBA locus resulted in a 10-fold "
    "increase of production at 24 h to 4.47 mg/L and a 48-h titer of "
    "9.27 mg/L (Fig.\xa03B)."
)
_Q_DAVT_ALR = (
    "Additional deletion of davT resulted in an increase of titer to 19.29 mg/L "
    "and 85.19 mg/L, and by deleting the amino acid racemase alr titers increased "
    "to 63.66 mg/L and 91.97 mg/L at 24 and 48 h, respectively"
)
_Q_FIG3B = (
    "Valerolactam production from different P.\xa0putida strains grown in LB medium "
    "supplemented with 25 mM L-lysine and 0.2% (w/v) arabinose at 24 and "
    "48 h post inoculation. Error bars show 95% cI, n = 3."
)
_Q_PRODUCTION = (
    "Production cultures of 10 mL of LB supplemented with kanamycin, 25 mM "
    "L-lysine, and 0.2% w/v arabinose were then inoculated 1:100 with overnight "
    "cultures and then grown at 30 °C shaking at 250 rpm. Samples for "
    "valerolactam production were taken at 24 and 48 h post-inoculation"
)
_Q_PRODUCER = (
    "To assess valerolactam production in strains of P.\xa0putida overnight cultures "
    "of strains harboring pBADT-davBA-ORF26 were grown in 3 mL of LB supplemented "
    "with kanamycin and grown at 30"
)
_Q_KANAMYCIN = (
    "Cultures were supplemented with kanamycin (50 mg/L, Sigma Aldrich, USA), "
    "gentamicin (30 mg/L, Fisher Scientific, USA), or carbenicillin "
    "(100 mg/L, Sigma Aldrich, USA), when indicated."
)
_Q_LB_MILLER = (
    "General E.\xa0coli cultures were grown in lysogeny broth (LB) Miller medium (BD "
    "Biosciences, USA) at 37 °C while P. putida was grown at 30 °C."
)
_Q_DAVBA_NATIVE = (
    "via DavBA, two genes endogenous to P.\xa0putida, and then cyclizes it via a "
    "promiscuous coA-ligase (Zhang"
)
_Q_ORF26 = "promiscuous acyl-coA ligase from Streptomyces aizunensis"
_Q_PATHWAY = (
    "we expressed the davBA-ORF26 pathway via a arabinose-inducible broad host range "
    "vector pBADT"
)
_Q_PLASMID = (
    "The biosynthetic pathway genes are shown in green and were overexpressed "
    "heterologously from a pBBR ori plasmid using an arabinose inducible promoter."
)
_Q_QUANTIFICATION = (
    "The HPLC system was coupled to an Agilent Technologies 6520 quadrupole "
    "time-of-flight mass spectrometer (QTOF MS) with a 1:6 post-column split."
)
_Q_CALIBRATION = (
    "Lactams were quantified by comparison with 8-point calibration curves of authentic "
    "chemical standards from 0.78125 μM to 100 μM."
)
_Q_ABSTRACT_UNDETECTABLE = (
    "Deletion of oplBA, as well as pathways that compete for precursors L-lysine or "
    "5-aminovalerate, increased the titer of valerolactam from undetectable after "
    "48 h of production to ~90 mg/L."
)
_Q_STRAIN_TABLE = (
    "P.\xa0putida KT2440\t\tATCC 47054\t\nP.\xa0putida ΔdavT\t\tThompson et\xa0al. "
    "(2019b)\t\nP.\xa0putida ΔoplBA\tJPUB_013576\tThis work\t\nP.\xa0putida "
    "ΔoplBAΔdavT\tJPUB_013577\tThis work\t\nP.\xa0putida ΔoplBAΔdavTΔalr\t"
    "JPUB_013578\tThis work\t\n"
)
_Q_DAVT_ROLE = "davT, which catalyzes the first step in 5AVA catabolism."
_Q_DELETIONS = (
    "Construction of P.\xa0putida deletion mutants was performed as described "
    "previously (Thompson et\xa0al., 2019a)."
)
_Q_GROWTH_ASSAY = (
    "These cultures were then washed twice with MOPS minimal media without any added "
    "carbon and diluted 1:100 into 500 μL of MOPS medium with 10 mM of a "
    "carbon source in 48-well plates (Falcon, 353072). Plates were sealed with a "
    "gas-permeable microplate adhesive film (VWR, USA), and then optical density and "
    "fluorescence were monitored for 48 h in an Biotek Synergy 4 plate reader "
    "(BioTek, USA) at 30 °C with fast continuous shaking. Optical density was "
    "measured at 600 nm."
)
_Q_FIG2 = (
    "Growth of wild-type, ΔdavT, or ΔoplBA in minimal media supplemented with either "
    "10 mM glucose (A), 5-aminovaleroate (B), or valerolactam (C)."
)
_Q_TABLE_S1_POINTER = (
    "Growth rates of strains on all carbon sources are reported in Table\xa0S1."
)
_Q_MOPS = (
    "When indicated, P. putida and E. coli were grown on modified MOPS minimal medium, "
    "which is comprised of 32.5 μM CaCl2, 0.29 mM K2SO4, 1.32 mM K2HPO4, "
    "8 μM FeCl2, 40 mM MOPS, 4 mM tricine, 0.01 mM FeSO4, "
    "9.52 mM NH4Cl, 0.52 mM MgCl2, 50 mM NaCl, 0.03 μM "
    "(NH4)6Mo7O24, 4 μM H3BO3, 0.3 μM CoCl2, 0.1 μM CuSO4, "
    "0.8 μM MnCl2, and 0.1 μM ZnSO4"
)
_Q_RBTNSEQ_METHODS = (
    "Libraries were then washed once in MOPS minimal medium with no carbon source, and "
    "then diluted 1:50 in MOPS minimal medium with 10 mM valerolactam."
)
_Q_RBTNSEQ_TWO = (
    "Interestingly, fitness data from two valerolactam RB-TnSeq experiments in "
    "P.\xa0putida KT2440 show oplBA mutants having no significant fitness defects "
    "(Fig.\xa0S2B)."
)
_Q_RBTNSEQ_RELEASE = (
    "All fitness data is publically available at http://fit.genomics.lbl.gov."
)
_Q_REGISTRY = (
    "All strains and plasmids created in this work are available through the public "
    "instance of the JBEI registry. (public-registry.jbei.org/folders/456)."
)
#: SI text-layer quotes (``si/mmc1.txt``).
_Q_SI_TABLE_S1_TITLE = (
    "Table S1: Specific growth rates of P. putida and valerolactam catabolic mutants "
    "on various \ncarbon sources."
)
_Q_SI_TABLE_S1_HEADER = "Strain Carbon Source Maximal Growth Rate (1/hr)"
_Q_SI_FIG_S2_ATTRIBUTION = (
    "All non-valerolactam fitness experiments are from \nThompson et. al 2019."
)

PAPER_QUOTES: tuple[str, ...] = (
    _Q_WT_OPLBA,
    _Q_DAVT_ALR,
    _Q_FIG3B,
    _Q_PRODUCTION,
    _Q_PRODUCER,
    _Q_KANAMYCIN,
    _Q_LB_MILLER,
    _Q_DAVBA_NATIVE,
    _Q_ORF26,
    _Q_PATHWAY,
    _Q_PLASMID,
    _Q_QUANTIFICATION,
    _Q_CALIBRATION,
    _Q_ABSTRACT_UNDETECTABLE,
    _Q_STRAIN_TABLE,
    _Q_DAVT_ROLE,
    _Q_DELETIONS,
    _Q_GROWTH_ASSAY,
    _Q_FIG2,
    _Q_TABLE_S1_POINTER,
    _Q_MOPS,
    _Q_RBTNSEQ_METHODS,
    _Q_RBTNSEQ_TWO,
    _Q_RBTNSEQ_RELEASE,
    _Q_REGISTRY,
)
SI_QUOTES: tuple[str, ...] = (
    _Q_SI_TABLE_S1_TITLE,
    _Q_SI_TABLE_S1_HEADER,
    _Q_SI_FIG_S2_ATTRIBUTION,
)


def _paper(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote of the pinned PMC full text."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_TEXT.relpath,
            citation_key=CITATION_KEY,
            sha256=PAPER_TEXT.sha256,
            method="PMC full text (raw mirror), publisher's own text layer",
            page=page,
            retrieved=RETRIEVED_AT,
        ),
    )


def _si(value: Any, quote: str, *, page: str, note: str | None = None) -> SourcedValue:
    """Bind a value to a verbatim quote of the pinned SI text layer."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SI_TEXT.relpath,
            citation_key=CITATION_KEY,
            sha256=SI_TEXT.sha256,
            method=f"text layer of si/mmc1.pdf rendered by pypdf {SI_TEXT_PYPDF_VERSION}",
            page=page,
            retrieved=RETRIEVED_AT,
        ),
    )


_RESULTS_PRODUCTION = (
    "Results 2.3, 'Host engineering for increased valerolactam production'"
)
_METHODS_MEDIA = "Methods 4.1, 'Media, chemicals, and culture conditions'"
_METHODS_GROWTH = "Methods 4.3, 'Plate based growth assays'"
_METHODS_PRODUCTION = "Methods 4.4, 'Production assays and lactam quantification'"

TEMPERATURE_C = _paper(30.0, _Q_PRODUCTION, page=_METHODS_PRODUCTION)
GROWTH_TEMPERATURE_C = _paper(30.0, _Q_GROWTH_ASSAY, page=_METHODS_GROWTH)
TITER_REPLICATES = _paper(
    3,
    _Q_FIG3B,
    page="Figure 3 caption, panel B",
    note="'n = 3' with 95% confidence-interval error bars; each n is one 10 mL "
    "production culture (Methods 4.4), so the unit is a biological replicate",
)
QUANTIFICATION = _paper(
    "HPLC-QTOF MS",
    _Q_QUANTIFICATION,
    page=_METHODS_PRODUCTION,
    note="HILIC HPLC coupled to an Agilent 6520 QTOF MS, quantified against 8-point "
    "standard curves",
)
KANAMYCIN_MG_PER_L = _paper(50.0, _Q_KANAMYCIN, page=_METHODS_MEDIA)
LYSINE_MM = _paper(25.0, _Q_PRODUCTION, page=_METHODS_PRODUCTION)
ARABINOSE_PERCENT_W_V = _paper(0.2, _Q_PRODUCTION, page=_METHODS_PRODUCTION)
PRODUCTION_VOLUME_ML = _paper(10.0, _Q_PRODUCTION, page=_METHODS_PRODUCTION)
PRODUCTION_SHAKING_RPM = _paper(250.0, _Q_PRODUCTION, page=_METHODS_PRODUCTION)
GROWTH_VOLUME_UL = _paper(500.0, _Q_GROWTH_ASSAY, page=_METHODS_GROWTH)
GROWTH_DURATION_HOURS = _paper(48.0, _Q_GROWTH_ASSAY, page=_METHODS_GROWTH)
CARBON_SOURCE_MM = _paper(10.0, _Q_GROWTH_ASSAY, page=_METHODS_GROWTH)
WT_48H_NOT_DETECTED = _paper(
    "not detected",
    _Q_WT_OPLBA,
    page=_RESULTS_PRODUCTION,
    note="a value below the 8-point calibration floor ('" + _Q_CALIBRATION + "'), "
    "which the abstract restates as 'undetectable'; not a zero",
)
RBTNSEQ_SUBSUMED = _paper(
    {"valerolactam_rbtnseq_experiments": 2, "release": "http://fit.genomics.lbl.gov"},
    _Q_RBTNSEQ_TWO,
    page="Results 2.1, 'Identification of a lactam hydrolase in P. putida'",
    note="both are Borchert 2024 compendium samples served by RbTnseqBorchert2024Dataset "
    "(set6IT064, set7IT045; 2-Piperidinone 10 mM, carbon source); the 5AVA samples of "
    "Fig. S2 are the lysine paper's ('" + _Q_SI_FIG_S2_ATTRIBUTION + "')",
)

#: Standard InChIKey of valerolactam (2-piperidinone), DERIVED from the structure SMILES
#: ``O=C1CCCCN1`` with ``rdkit.Chem.inchi.MolToInchiKey`` (rdkit 2026.03.6, run
#: 2026-10-10). Recorded for the ``compound_identity_table.json`` curation that will fill
#: it and deliberately NOT set on the ``Compound``: curating a row is a human act. The
#: Borchert 2024 compendium names the same molecule ``2-Piperidinone``, so that row is
#: what would join this product to the compendium's two valerolactam samples.
VALEROLACTAM_INCHIKEY = "XUWHAWMETYGRKB-UHFFFAOYSA-N"
PRODUCT_NAME = "valerolactam"


# --------------------------------------------------------------------------- #
# Strains
# --------------------------------------------------------------------------- #
class StrainSpec(BaseModel):
    """One strain of Table 1: its gene deletions, by the symbols the paper writes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    deletions: tuple[str, ...]
    jbei_part_id: str | None


STRAINS: dict[str, StrainSpec] = {
    spec.name: spec
    for spec in (
        StrainSpec(name="KT2440", deletions=(), jbei_part_id=None),
        StrainSpec(name="ΔdavT", deletions=("davT",), jbei_part_id=None),
        StrainSpec(
            name="ΔoplBA", deletions=("oplB", "oplA"), jbei_part_id="JPUB_013576"
        ),
        StrainSpec(
            name="ΔoplBAΔdavT",
            deletions=("oplB", "oplA", "davT"),
            jbei_part_id="JPUB_013577",
        ),
        StrainSpec(
            name="ΔoplBAΔdavTΔalr",
            deletions=("oplB", "oplA", "davT", "alr"),
            jbei_part_id="JPUB_013578",
        ),
    )
}
#: The symbol the GenBank annotation does not carry, resolved through the GOA file.
GOA_RESOLVED_SYMBOLS: frozenset[str] = frozenset({"davT"})
#: The two native genes of the production plasmid.
PLASMID_NATIVE_SYMBOLS: tuple[str, ...] = ("davB", "davA")
PLASMID_NAME = "pBADT-davBA-ORF26"
PLASMID_JBEI_PART_ID = "JPUB_013587"
PATHWAY_NAME = "valerolactam from L-lysine via 5-aminovalerate (davBA-ORF26)"
PROMOTER_NAME = "arabinose inducible promoter"
ORF26_ORGANISM = "Streptomyces aizunensis"
NATIVE_ORGANISM = "Pseudomonas putida KT2440"


def all_symbols() -> tuple[str, ...]:
    """Every gene symbol a record names: the deletions, then the plasmid's native pair."""
    deleted = sorted({s for spec in STRAINS.values() for s in spec.deletions})
    return (*deleted, *PLASMID_NATIVE_SYMBOLS)


# --------------------------------------------------------------------------- #
# Released values
# --------------------------------------------------------------------------- #
class TiterCell(BaseModel):
    """One (strain, sampling time) the Results state a titer for."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strain: str
    hours: float
    titer_mg_per_l: float | None
    quote: str


#: The eight titers the Results state. The 24 h family is stored; the 48 h family is
#: refused, because its reference (the wild type at 48 h) is below the detection floor
#: and ``ProductTiterPhenotype.titer`` is a required non-negative float with no
#: censoring field (:data:`TITER_48H_REFUSED`).
TITER_CELLS: tuple[TiterCell, ...] = (
    TiterCell(strain="KT2440", hours=24.0, titer_mg_per_l=0.43, quote=_Q_WT_OPLBA),
    TiterCell(strain="KT2440", hours=48.0, titer_mg_per_l=None, quote=_Q_WT_OPLBA),
    TiterCell(strain="ΔoplBA", hours=24.0, titer_mg_per_l=4.47, quote=_Q_WT_OPLBA),
    TiterCell(strain="ΔoplBA", hours=48.0, titer_mg_per_l=9.27, quote=_Q_WT_OPLBA),
    TiterCell(
        strain="ΔoplBAΔdavT", hours=24.0, titer_mg_per_l=19.29, quote=_Q_DAVT_ALR
    ),
    TiterCell(
        strain="ΔoplBAΔdavT", hours=48.0, titer_mg_per_l=85.19, quote=_Q_DAVT_ALR
    ),
    TiterCell(
        strain="ΔoplBAΔdavTΔalr", hours=24.0, titer_mg_per_l=63.66, quote=_Q_DAVT_ALR
    ),
    TiterCell(
        strain="ΔoplBAΔdavTΔalr", hours=48.0, titer_mg_per_l=91.97, quote=_Q_DAVT_ALR
    ),
)
#: The strain every titer record's reference is, and the sampling time it is read at.
TITER_REFERENCE_STRAIN = "KT2440"
#: Sampling times, in hours, whose records are written.
TITER_HOURS_STORED: tuple[float, ...] = (24.0,)
TITER_48H_REFUSED = (
    "the four 48 h titers are NOT records. ProductTiterExperimentReference requires a "
    "phenotype_reference and ProductTiterPhenotype.titer is a required non-negative "
    f"float, and the 48 h reference measurement is '{WT_48H_NOT_DETECTED.value}': the "
    f"Results state '{_Q_WT_OPLBA}' and the abstract restates it as 'undetectable'. "
    "That is a value below the 8-point calibration floor, which no field of "
    "ProductTiterPhenotype can carry (it has no Censoring slot, unlike "
    "ProteinTurnoverPhenotype). Storing 0.0 would state a measurement nobody made and "
    "reusing the 24 h reference would compare two sampling times, so the wild type's "
    "48 h cell and the three engineered 48 h titers that would be measured against it "
    "are all refused. The sibling Kang 2026 and Yunus 2026 loaders refuse a titer "
    "family for the same missing-denominator reason"
)


class GrowthRateRow(BaseModel):
    """One released Table S1 row: a strain on a carbon source, and its maximal rate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strain: str
    carbon_source: str
    rate_per_hour: float


#: Table S1's three carbon sources, as the SI writes them -> the compound name the
#: shared identity layer resolves. ``5AVA`` is the SI's abbreviation.
TABLE_S1_CARBON: dict[str, str] = {
    "5AVA": "5-aminovaleric acid",
    "Glucose": "D-glucose",
    "Valerolactam": PRODUCT_NAME,
}
#: Table S1's strain labels, as the SI writes them -> the Table 1 strain.
TABLE_S1_STRAINS: dict[str, str] = {
    "WT": "KT2440",
    "ΔdavT": "ΔdavT",
    "ΔoplBA": "ΔoplBA",
}
#: The row count Table S1 releases: three strains x three carbon sources.
EXPECTED_GROWTH_ROWS = len(TABLE_S1_STRAINS) * len(TABLE_S1_CARBON)
#: The carbon source whose record is every growth record's reference: the one all three
#: strains grow on identically ("All strains showed identical growth on glucose as a
#: sole carbon source"), which is the comparison Figs. 2A to 2C are read against.
GROWTH_REFERENCE_CARBON = "Glucose"
_TABLE_S1_ROW = re.compile(
    r"^(?P<strain>\S+)?\s*(?P<carbon>5AVA|Glucose|Valerolactam)\s+"
    r"(?P<rate>[0-9]+(?:\.[0-9]+)?)\s*$"
)


def read_table_s1(si_text_path: str | Path) -> list[GrowthRateRow]:
    """Read Table S1's nine growth rates out of the pinned SI text layer.

    The table is a text-layer block: a row names a strain only when the strain changes,
    so the strain carries down the three carbon sources beneath it, which is how the PDF
    renders a merged cell. A row whose rate does not parse, or a strain or carbon source
    the SI does not name, raises rather than being skipped.
    """
    strain: str | None = None
    rows: list[GrowthRateRow] = []
    for line in Path(si_text_path).read_text(encoding="utf-8").splitlines():
        match = _TABLE_S1_ROW.match(line.strip())
        if match is None:
            continue
        if match.group("strain") is not None:
            label = match.group("strain")
            if label not in TABLE_S1_STRAINS:
                raise RuntimeError(f"Table S1 names strain {label!r}, which is unknown")
            strain = TABLE_S1_STRAINS[label]
        if strain is None:
            raise RuntimeError(f"a Table S1 row precedes any strain label: {line!r}")
        rows.append(
            GrowthRateRow(
                strain=strain,
                carbon_source=match.group("carbon"),
                rate_per_hour=float(match.group("rate")),
            )
        )
    if len(rows) != EXPECTED_GROWTH_ROWS:
        raise RuntimeError(
            f"Table S1 parsed {len(rows)} rows, expected {EXPECTED_GROWTH_ROWS}"
        )
    return rows


# --------------------------------------------------------------------------- #
# The medium this paper states, built here and kept out of MEDIA_LIBRARY
# --------------------------------------------------------------------------- #
_MM = ConcentrationUnit.millimolar
_UM = ConcentrationUnit.micromolar


def _mops_component(
    name: str, role: MediaComponentRole, value: float, unit: ConcentrationUnit
) -> MediaComponent:
    """One component of this paper's modified MOPS medium, from the Methods recipe."""
    return MediaComponent(
        compound=resolved_compound(name),
        role=role,
        concentration=Concentration(value=value, unit=unit),
        provenance=[_paper(value, _Q_MOPS, page=_METHODS_MEDIA)],
    )


MOPS_MODIFIED_THOMPSON2019 = Media(
    name="modified MOPS minimal medium, no carbon source (Thompson 2019, LaBauve and "
    "Wargo 2012 formulation)",
    state="liquid",
    is_synthetic=True,
    base_medium="MOPS_MINIMAL",
    components=[
        _mops_component("calcium chloride", MediaComponentRole.bulk_salt, 32.5, _UM),
        _mops_component("potassium sulfate", MediaComponentRole.bulk_salt, 0.29, _MM),
        _mops_component(
            "dipotassium hydrogen phosphate", MediaComponentRole.bulk_salt, 1.32, _MM
        ),
        _mops_component(
            "iron(II) chloride", MediaComponentRole.trace_element, 8.0, _UM
        ),
        _mops_component(
            "3-(N-morpholino)propanesulfonic acid", MediaComponentRole.buffer, 40.0, _MM
        ),
        _mops_component("tricine", MediaComponentRole.buffer, 4.0, _MM),
        _mops_component(
            "iron(II) sulfate", MediaComponentRole.trace_element, 0.01, _MM
        ),
        _mops_component(
            "ammonium chloride", MediaComponentRole.nitrogen_source, 9.52, _MM
        ),
        _mops_component("magnesium chloride", MediaComponentRole.bulk_salt, 0.52, _MM),
        _mops_component("sodium chloride", MediaComponentRole.bulk_salt, 50.0, _MM),
        _mops_component(
            "ammonium heptamolybdate", MediaComponentRole.trace_element, 0.03, _UM
        ),
        _mops_component("boric acid", MediaComponentRole.trace_element, 4.0, _UM),
        _mops_component("cobalt chloride", MediaComponentRole.trace_element, 0.3, _UM),
        _mops_component("copper sulfate", MediaComponentRole.trace_element, 0.1, _UM),
        _mops_component("MnCl2", MediaComponentRole.trace_element, 0.8, _UM),
        _mops_component("zinc sulfate", MediaComponentRole.trace_element, 0.1, _UM),
    ],
    provenance=[_paper("modified MOPS minimal medium", _Q_MOPS, page=_METHODS_MEDIA)],
)
"""This paper's modified MOPS, carbon-source free, kept in this module.

It is NOT the library's ``MOPS_MINIMAL`` (Neidhardt as Price 2018 tabulates it): the
stated amounts differ (32.5 uM against 0.5 uM calcium chloride, 0.29 against 0.276 mM
potassium sulfate, 9.52 against 9.5 mM ammonium chloride, 0.52 against 0.525 mM
magnesium salt, which this recipe weighs as the chloride) and this one adds 8 uM
iron(II) chloride. ``base_medium`` still names ``MOPS_MINIMAL`` so the MOPS family
joins; the object stays out of ``MEDIA_LIBRARY`` for the Ishii 2007 and Choe 2019
reason, that a per-paper recipe nobody else states should not invite a second loader to
attach amounts to it.

One component carries a typed identity gap rather than a structure: ``iron(II) chloride``
has no row in the committed ``compound_identity_table.json`` under that name or any
synonym measured here (``FeCl2``, ``iron chloride``, ``ferrous chloride``,
``iron dichloride``), so ``resolved_compound`` returns the name with a ``ProvenanceGap``
on ``inchikey``. Curating that row is a human act and is raised in the PR.
"""


def publication() -> Publication:
    """The paper every record cites."""
    return Publication(
        pubmed_id=PUBMED_ID,
        pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PUBMED_ID}/",
        doi=DOI,
        doi_url=f"https://doi.org/{DOI}",
    )


def _dose(
    compound: str, value: float, unit: ConcentrationUnit
) -> SmallMoleculePerturbation:
    """One added small molecule at a stated dose, through the shared compound layer."""
    return SmallMoleculePerturbation(
        compound=resolved_compound(compound),
        concentration=Concentration(value=value, unit=unit),
    )


def production_environment(hours: float) -> CultureEnvironment:
    """The LB + 25 mM L-lysine + 0.2% arabinose production culture at one sampling time.

    A ``CultureEnvironment`` because ``ProductTiterExperiment.environment`` is annotated
    as one: the 10 mL culture shaken at 250 rpm travels on the record. The medium is the
    library's ``LB`` (Miller, which the Methods name), and the lysine, the arabinose and
    the kanamycin are its three stated additions.
    """
    return CultureEnvironment(
        media=LB,
        temperature=Temperature(value=float(TEMPERATURE_C.value)),
        culture_format=CultureFormat(
            vessel=None,
            working_volume_ul=float(PRODUCTION_VOLUME_ML.value) * 1000.0,
            shaking_rpm=float(PRODUCTION_SHAKING_RPM.value),
            endpoint=EndpointRule.fixed_duration,
            provenance=[PRODUCTION_VOLUME_ML, PRODUCTION_SHAKING_RPM],
            provenance_gaps=[
                ProvenanceGap(
                    field="vessel",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                    note="the Methods state the 10 mL working volume and the 250 rpm "
                    f"shaking ('{_Q_PRODUCTION}') but never the vessel: no flask, tube "
                    "or plate is named for the production culture, and the 3 mL "
                    f"overnight culture it is inoculated from ('{_Q_PRODUCER}') names "
                    "none either, so the slot stays empty rather than borrowing one",
                )
            ],
        ),
        perturbations=[
            _dose("L-lysine", float(LYSINE_MM.value), ConcentrationUnit.millimolar),
            _dose(
                "L-arabinose",
                float(ARABINOSE_PERCENT_W_V.value),
                ConcentrationUnit.percent_w_v,
            ),
            _dose(
                "kanamycin",
                float(KANAMYCIN_MG_PER_L.value),
                ConcentrationUnit.ug_per_ml,
            ),
        ],
        aerobicity="aerobic",
        duration_hours=hours,
    )


def growth_environment(carbon_source: str) -> Environment:
    """The 48-well plate growth assay on one 10 mM carbon source.

    The medium is this paper's carbon-source-free modified MOPS, so the carbon source is
    an ``EnvironmentPhysicalPerturbation(factor=carbon_source)`` naming the molecule and
    its 10 mM dose, which is the convention every carbon-free medium in the library
    carries.
    """
    return Environment(
        media=MOPS_MODIFIED_THOMPSON2019,
        temperature=Temperature(value=float(GROWTH_TEMPERATURE_C.value)),
        perturbations=[
            EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.carbon_source,
                magnitude=Concentration(
                    value=float(CARBON_SOURCE_MM.value),
                    unit=ConcentrationUnit.millimolar,
                ),
                agent=resolved_compound(TABLE_S1_CARBON[carbon_source]),
            )
        ],
        aerobicity="aerobic",
        duration_hours=float(GROWTH_DURATION_HOURS.value),
    )


# --------------------------------------------------------------------------- #
# Gene symbols -> locus tags, including the one the annotation does not carry
# --------------------------------------------------------------------------- #
class GoaSymbolResolution(BaseModel):
    """One symbol the pinned GOA proteome file places at exactly one locus."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    locus_tag: str
    accession: str
    product_name: str
    assembly_set: str
    member: str
    sha256: str


def goa_symbol_locus(
    genome: PPutidaKT2440Genome, symbol: str, data_root: str | None = None
) -> GoaSymbolResolution:
    """The one locus tag the assembly set's GOA proteome file gives ``symbol``.

    ``davT`` is a symbol the pinned GenBank annotation of GCA_000007565.2 does not
    carry, so ``resolve_gene_name`` returns ``retired``. The genomes tier's own GOA file
    for the same assembly set names it: the symbol is a DB-object symbol (column 3) and
    the locus tag is in that row's synonym column, which is the same column and the same
    locus-tag pattern :func:`torchcell.datasets.bacteria_common.uniprot_locus_crosswalk`
    reads. More than one locus, or none, raises: the point of reading the file is that
    the mapping is stated, not inferred.
    """
    spec = genome.ASSEMBLY.go_source
    if spec.identifier_pattern is None:
        raise ValueError(f"{spec.member} names no identifier_pattern")
    path = resolve(spec.assembly_set, spec.member, data_root=data_root)
    pinned = load_genome_manifest(spec.assembly_set, data_root).record(spec.member)
    pattern = re.compile(spec.identifier_pattern)
    found: dict[str, tuple[str, str]] = {}
    with open(path) as handle:
        for line in handle:
            if line.startswith("!"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) != GAF_COLUMNS:
                raise ValueError(
                    f"{spec.member}: a GAF 2.x row has {GAF_COLUMNS} columns, got "
                    f"{len(columns)}"
                )
            if columns[2] != symbol:
                continue
            tags = {
                token
                for value in columns[10].split("|")
                for token in value.split("/")
                if pattern.fullmatch(token)
            }
            if len(tags) != 1:
                raise RuntimeError(
                    f"{spec.member}: the {symbol!r} row names {sorted(tags)} locus "
                    "tags; exactly one is required"
                )
            found[columns[1]] = (next(iter(tags)), columns[9])
    if len(found) != 1:
        raise RuntimeError(
            f"{spec.member}: {symbol!r} is the symbol of {len(found)} proteins "
            f"({sorted(found)}); exactly one is required"
        )
    accession, (locus_tag, product_name) = next(iter(found.items()))
    return GoaSymbolResolution(
        symbol=symbol,
        locus_tag=locus_tag,
        accession=accession,
        product_name=product_name,
        assembly_set=spec.assembly_set,
        member=spec.member,
        sha256=pinned.sha256,
    )


class SymbolResolution(BaseModel):
    """Every symbol this paper names, and the locus tag each one is stored as."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    locus_tags: dict[str, str]
    reconciliation: LocusTagReconciliation
    goa: tuple[GoaSymbolResolution, ...]


def resolve_symbols(
    genome: PPutidaKT2440Genome, data_root: str | None = None
) -> SymbolResolution:
    """Map every symbol the paper names to a locus tag of the pinned assembly.

    The annotation decides every symbol it carries; the pinned GOA file decides the ones
    it does not (:data:`GOA_RESOLVED_SYMBOLS`). A symbol in neither raises, because the
    paper names no locus tag of its own to fall back to.
    """
    symbols = all_symbols()
    annotated = [s for s in symbols if s not in GOA_RESOLVED_SYMBOLS]
    stored, report = reconcile_locus_tags(
        genome, pd.Series(annotated), label="thompson2019_valerolactam"
    )
    locus_tags = dict(zip(annotated, stored, strict=True))
    unresolved = [s for s, tag in locus_tags.items() if tag == s]
    if unresolved:
        raise RuntimeError(
            f"the pinned annotation resolves no locus for {unresolved}; the paper names "
            "no locus tag, so nothing can be stored for them"
        )
    goa = tuple(
        goa_symbol_locus(genome, symbol, data_root)
        for symbol in sorted(GOA_RESOLVED_SYMBOLS)
    )
    for resolution in goa:
        locus_tags[resolution.symbol] = resolution.locus_tag
    return SymbolResolution(locus_tags=locus_tags, reconciliation=report, goa=goa)


def strain_genotype(strain: str, locus_tags: dict[str, str]) -> Genotype:
    """One strain's genotype: its deletions plus the production plasmid's three genes.

    Every strain carries ``pBADT-davBA-ORF26`` -- the titer records because the Methods
    grow only plasmid-bearing strains, and that is the whole point of the comparison.
    """
    perturbations: list[Any] = [
        BacterialDeletionPerturbation(
            systematic_gene_name=locus_tags[symbol],
            perturbed_gene_name=symbol,
            gene_namespace=KT2440_NAMESPACE,
            cassette=None,
            collection=None,
        )
        for symbol in STRAINS[strain].deletions
    ]
    perturbations.extend(plasmid_perturbations(locus_tags))
    return Genotype(perturbations=perturbations)


def plasmid_perturbations(
    locus_tags: dict[str, str],
) -> list[HeterologousPathwayPerturbation]:
    """The three genes of ``pBADT-davBA-ORF26``, each as its own typed perturbation.

    ``davB`` and ``davA`` are extra copies of KT2440's own genes ("two genes endogenous
    to P. putida"), so each carries its locus tag and ``is_heterologous=False``;
    ``ORF26`` is the *Streptomyces aizunensis* acyl-CoA ligase and carries its source
    name, since it has no locus in this host.
    """
    genes = [
        HeterologousPathwayPerturbation(
            systematic_gene_name=locus_tags[symbol],
            perturbed_gene_name=symbol,
            gene_namespace=KT2440_NAMESPACE,
            pathway_name=PATHWAY_NAME,
            source_organism=NATIVE_ORGANISM,
            is_heterologous=False,
            localization="episomal_plasmid",
            construct_name=PLASMID_NAME,
            integration_locus=None,
            promoter_name=PROMOTER_NAME,
            copy_number=1.0,
        )
        for symbol in PLASMID_NATIVE_SYMBOLS
    ]
    genes.append(
        HeterologousPathwayPerturbation(
            systematic_gene_name="ORF26",
            perturbed_gene_name="ORF26",
            gene_namespace=KT2440_NAMESPACE,
            pathway_name=PATHWAY_NAME,
            source_organism=ORF26_ORGANISM,
            is_heterologous=True,
            localization="episomal_plasmid",
            construct_name=PLASMID_NAME,
            integration_locus=None,
            promoter_name=PROMOTER_NAME,
            copy_number=1.0,
        )
    )
    return genes


def deletion_genotype(strain: str, locus_tags: dict[str, str]) -> Genotype:
    """One strain's genotype WITHOUT the plasmid: the growth-assay arm.

    Figs. 2A to 2C grow the wild type and the two single-locus mutants with no
    production plasmid, so a growth record must not carry one.
    """
    return Genotype(
        perturbations=[
            BacterialDeletionPerturbation(
                systematic_gene_name=locus_tags[symbol],
                perturbed_gene_name=symbol,
                gene_namespace=KT2440_NAMESPACE,
                cassette=None,
                collection=None,
            )
            for symbol in STRAINS[strain].deletions
        ]
    )


# --------------------------------------------------------------------------- #
# Phenotypes
# --------------------------------------------------------------------------- #
def valerolactam() -> Compound:
    """The product, through the shared compound-identity layer.

    ``valerolactam`` has no row in the committed ``compound_identity_table.json``, so
    the resolver returns the honest typed absence: the canonical name with a
    ``ProvenanceGap`` on ``inchikey``. The key the curation will fill is
    :data:`VALEROLACTAM_INCHIKEY`.
    """
    return resolved_compound(PRODUCT_NAME)


def titer_phenotype(titer_mg_per_l: float) -> ProductTiterPhenotype:
    """One stated titer in ``ug/mL``, with the typed absences the paper forces.

    The released numbers are mg/L and 1 mg/L is exactly 1 ug/mL, so no arithmetic is
    applied to a source value. The replicate DESIGN is sourced (n = 3, 95% confidence
    interval) but no interval NUMBER is released: Fig. 3B has no source-data file and
    the prose gives only the means. ``ProductTiterPhenotype`` forbids an unlabelled
    uncertainty, so both the number and its type are typed gaps rather than a
    half-filled pair, and ``UncertaintyType.ci95`` takes a half-width this paper never
    prints.
    """
    design_note = (
        "the spread is released only as error bars: Fig. 3B has no source-data file "
        f"and the Results give the means alone. The design is sourced ('{_Q_FIG3B}') "
        "and is carried in n_samples + sample_unit"
    )
    return ProductTiterPhenotype(
        product=valerolactam(),
        titer=titer_mg_per_l,
        titer_unit=ConcentrationUnit.ug_per_ml,
        n_samples=int(TITER_REPLICATES.value),
        sample_unit=SampleUnit.biological_replicate,
        quantification_method=str(QUANTIFICATION.value),
        provenance_gaps=[
            ProvenanceGap(
                field="titer_uncertainty",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note=design_note,
            ),
            ProvenanceGap(
                field="titer_uncertainty_type",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="the error bars are a 95% confidence interval over biological "
                f"triplicates ('{_Q_FIG3B}'); UncertaintyType.ci95 takes the "
                "half-width, which is nowhere printed, so this is a typed absence",
            ),
            ProvenanceGap(
                field="product_yield",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="no yield on the 25 mM lysine fed is released for any strain",
            ),
            ProvenanceGap(
                field="product_yield_unit",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="no yield on the 25 mM lysine fed is released for any strain",
            ),
            ProvenanceGap(
                field="productivity",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="no volumetric productivity is released for any strain or time",
            ),
            ProvenanceGap(
                field="productivity_unit",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="no volumetric productivity is released for any strain or time",
            ),
        ],
    )


#: What one stored growth number IS: an ABSOLUTE maximal specific growth rate in h^-1,
#: so ``MeasurementType.growth_rate`` (a member of ``ABSOLUTE_MEASUREMENT_TYPES``) and
#: the reference states the reference condition's own measured rate rather than 0.
GROWTH_UNITS = "maximal specific growth rate, 1/hr"


def growth_phenotype(rate_per_hour: float) -> EnvironmentResponsePhenotype:
    """One released Table S1 rate, stored verbatim with no uncertainty.

    Table S1 releases ONE number per (strain, carbon source) and no dispersion column.
    The growth curves of Figs. 2A to 2C are n = 3 with a 95% confidence band, but the
    band is on the OD curve rather than on the fitted rate, and no per-replicate rate is
    released, so ``n_samples`` is a typed gap rather than a 3 carried over from a
    different quantity.
    """
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.growth_rate,
        assay_type=AssayType.liquid_od_growth,
        environment_response=rate_per_hour,
        units=GROWTH_UNITS,
        provenance_gaps=[
            ProvenanceGap(
                field=field,
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="Table S1 releases one rate per strain and carbon source with no "
                f"dispersion column ('{_Q_SI_TABLE_S1_HEADER}'), and no per-replicate "
                "rate is released anywhere; the n = 3 of the Fig. 2 growth curves is "
                "the replicate count of the OD curve, not of this fitted rate",
            )
            for field in (
                "environment_response_uncertainty",
                "environment_response_uncertainty_type",
                "environment_response_se",
                "n_samples",
                "sample_unit",
            )
        ],
    )


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror and the build tree live there)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/thompsonOmicsdrivenIdentificationElimination2019``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


SI_EXPECTED: list[str] = [
    "paper/PMC6838509.1.txt and paper/PMC6838509.1.xml (PMC open-access full text) "
    "carry every quote this loader's paper values cite: the eight titers of Results "
    "2.3, the production and growth-assay Methods, the modified MOPS recipe and Table 1",
    "si/mmc1.pdf (Appendix A, publisher mmc1.pdf) is the paper's ONLY supplementary "
    "file: Figures S1 to S5 and Table S1. Table S1's nine growth rates are read from "
    "si/mmc1.txt, its pypdf text layer, which is deposited beside it so a quote is "
    "auditable against bytes rather than against a re-render",
    "the per-replicate titers and the 95% confidence intervals of Fig. 3B were NEVER "
    "released: there is no source-data file and the SI holds no titer table, which is "
    "why the titer uncertainty is a typed gap rather than a deferred one",
    f"RB-TnSeq fitness is published only on the Fitness Browser ({FITNESS_BROWSER}), "
    "with no per-experiment accession. Nothing is deposited for it here: the bytes that "
    "carry this paper's two valerolactam samples (set6IT064, set7IT045) are the "
    "Borchert 2024 compendium release under that key's own raw mirror, and "
    "RbTnseqBorchert2024Dataset already serves them",
    f"{JBEI_REGISTRY_FOLDER} -- the strains and plasmids are deposited in the JBEI "
    "registry, so no cassette sequence is retrievable by script; Table 1's genotype "
    "labels and the Results' plasmid description are the authority this loader types "
    "against",
]


def deposit_raw_mirror(
    sources: dict[str, str | Path],
    *,
    retrieved_at: str = RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror and its ``manifest.json`` from local copies of each file.

    ``sources`` maps every ``RAW_FILES`` relpath to a local file. Idempotent by sha256:
    a mirror file whose digest matches is left alone and a differing one raises rather
    than being overwritten. Every source is verified BEFORE anything is written, so a
    refusal leaves no partial deposit.
    """
    for raw in RAW_FILES:
        if raw.relpath not in sources:
            raise RuntimeError(f"no source given for {raw.relpath}")
        verify_sha256(sources[raw.relpath], raw.sha256)
    root = raw_mirror_dir(data_root)
    records: list[ArtifactRecord] = []
    for raw in RAW_FILES:
        dest = root / raw.relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if sha256_file(dest) != raw.sha256:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(sources[raw.relpath], dest)
        records.append(_artifact_record(raw, dest, retrieved_at))
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
        files=records,
        si_data_sources=[
            pmc_cloud_url(PMC_PREFIX),
            FITNESS_BROWSER,
            JBEI_REGISTRY_FOLDER,
        ],
        si_expected=SI_EXPECTED,
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return root


def _artifact_record(raw: RawFile, dest: Path, retrieved_at: str) -> ArtifactRecord:
    """The manifest record of one deposited file: retrieved, or derived from one."""
    if raw.bucket_name is None:
        return ArtifactRecord(
            path=raw.relpath,
            role=raw.role,
            bytes=dest.stat().st_size,
            sha256=raw.sha256,
            source=f"derived from {SI_PDF.relpath}",
            processing=ProcessingRecord(
                processor="pypdf.PdfReader(...).pages[*].extract_text",
                tool="pypdf",
                version=SI_TEXT_PYPDF_VERSION,
                params={"page_separator": SI_TEXT_PAGE_SEPARATOR},
                input_sha256=[SI_PDF.sha256],
            ),
        )
    url = pmc_cloud_url(raw.bucket_key)
    return ArtifactRecord(
        path=raw.relpath,
        role=raw.role,
        bytes=dest.stat().st_size,
        sha256=raw.sha256,
        source=url,
        original_filename=raw.bucket_name,
        retrieval=RetrievalRecord(
            method=RetrievalMethod.pmc_cloud,
            source_url=url,
            retriever="torchcell.literature.retrieve.pmc_cloud_object",
            params={"key": raw.bucket_key},
            sha256=raw.sha256,
            retrieved_at=retrieved_at,
        ),
    )


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


def _link_mirror_files(raw_dir: str, pins: Iterable[RawFile]) -> None:
    """Link each pinned mirror file into ``raw/`` after checking it against the manifest."""
    data_root = _data_root()
    manifest = load_manifest(data_root)
    os.makedirs(raw_dir, exist_ok=True)
    for raw in pins:
        check_manifest_pin(
            raw.relpath, manifest_sha256(manifest, raw.relpath), raw.sha256
        )
        src = raw_mirror_dir(data_root) / raw.relpath
        if not src.exists():
            raise RuntimeError(f"required raw artifact missing from mirror: {src}")
        link_verified(src, osp.join(raw_dir, osp.basename(raw.relpath)), raw.sha256)


def verify_quotes(data_root: str | None = None) -> dict[str, int]:
    """Assert every quote is verbatim in the pinned file it cites.

    The two consumed text artifacts are hashed first, so a quote is trusted only against
    the pinned bytes. Returns the number of quotes checked per artifact.
    """
    root = raw_mirror_dir(data_root)
    checked: dict[str, int] = {}
    for relpath, quotes in (
        (PAPER_TEXT.relpath, PAPER_QUOTES),
        (SI_TEXT.relpath, SI_QUOTES),
    ):
        raw = raw_file(relpath)
        path = root / raw.relpath
        digest = sha256_file(path)
        if digest != raw.sha256:
            raise RuntimeError(
                f"{path} hashes {digest}, not the pinned {raw.sha256}; every quote must "
                "be re-read before it is trusted"
            )
        text = path.read_text(encoding="utf-8")
        missing = [quote for quote in quotes if quote not in text]
        if missing:
            raise RuntimeError(
                f"{len(missing)} of {len(quotes)} quotes are not verbatim in "
                f"{raw.relpath}: {missing[:2]}"
            )
        checked[raw.relpath] = len(quotes)
    return checked


# --------------------------------------------------------------------------- #
# Build accounting
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One retention rule, what it removed, and the items it removed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule: str
    description: str
    n_records: int
    items: tuple[str, ...] = ()


class BuildAccounting(BaseModel):
    """Everything a reader needs to audit one build's retention arithmetic."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    source_rows: int
    candidate_records: int
    kept_records: int
    dropped_records: int
    rules: list[DropRule] = []
    symbol_locus_tags: dict[str, str] = {}
    goa_resolved: list[GoaSymbolResolution] = []
    notes: list[str] = []

    def check(self) -> None:
        """The rules must account for every candidate record that was not written."""
        if self.kept_records + self.dropped_records != self.candidate_records:
            raise RuntimeError(
                f"{self.dataset}: {self.kept_records} kept + {self.dropped_records} "
                f"dropped != {self.candidate_records} candidates"
            )
        accounted = sum(rule.n_records for rule in self.rules)
        if accounted != self.dropped_records:
            raise RuntimeError(
                f"{self.dataset}: rules total {accounted}, {self.dropped_records} "
                "records are missing from the build"
            )


def _write_accounting(accounting: BuildAccounting, preprocess_dir: str) -> None:
    """Validate and write ``preprocess/build_accounting.json``."""
    accounting.check()
    os.makedirs(preprocess_dir, exist_ok=True)
    with open(osp.join(preprocess_dir, "build_accounting.json"), "w") as handle:
        handle.write(accounting.model_dump_json(indent=2))


# --------------------------------------------------------------------------- #
# The datasets
# --------------------------------------------------------------------------- #
#: Four of the eight stated titers: the 24 h column, every strain.
EXPECTED_TITER_RECORDS = sum(
    1 for cell in TITER_CELLS if cell.hours in TITER_HOURS_STORED
)
#: Table S1 in full: nine rates, nothing dropped.
EXPECTED_GROWTH_RECORDS = EXPECTED_GROWTH_ROWS


class _Thompson2019Dataset(ExperimentDataset):
    """Shared raw-file, genome and quote handling for the two released families."""

    REFERENCE_STRAIN: ClassVar[Literal["KT2440"]] = "KT2440"
    #: Every gene symbol the paper names must reach a locus of the pinned assembly:
    #: five through the annotation and ``davT`` through the pinned GOA file. A value
    #: below 1.0 means the annotation moved and the build stops rather than dropping a
    #: gene the paper deleted.
    MIN_RESOLVED_FRACTION: ClassVar[float] = 1.0
    #: The mirror paths this family's build reads; resolved to pins at call time.
    RAW_PATHS: ClassVar[tuple[str, ...]] = (PAPER_TEXT.relpath,)

    @classmethod
    def raw_pins(cls) -> tuple[RawFile, ...]:
        """This family's pinned mirror records, read from the module at call time."""
        return tuple(raw_file(relpath) for relpath in cls.RAW_PATHS)

    def __init__(
        self,
        root: str,
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the KT2440 genome resolves every gene symbol to a locus tag."""
        self.pputida_genome = pputida_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def raw_file_names(self) -> list[str]:
        """The deposited mirror files this family reads."""
        return [osp.basename(relpath) for relpath in self.RAW_PATHS]

    def download(self) -> None:
        """Link the pinned mirror files into ``raw/`` after verifying each one."""
        _link_mirror_files(self.raw_dir, self.raw_pins())

    def _genome(self) -> PPutidaKT2440Genome:
        """The injected KT2440 genome, or one opened from the genomes tier."""
        if self.pputida_genome is None:
            self.pputida_genome = bacterial_genome("pputida", self.REFERENCE_STRAIN)
        return self.pputida_genome

    def _prepare(self) -> SymbolResolution:
        """Verify every consumed file and quote, then resolve every gene symbol."""
        verify_raw_files(
            self.raw_dir,
            {osp.basename(raw.relpath): raw.sha256 for raw in self.raw_pins()},
        )
        verify_quotes()
        resolution = resolve_symbols(self._genome())
        resolution.reconciliation.require_resolved(self.MIN_RESOLVED_FRACTION)
        if resolution.reconciliation.outside_namespace:
            raise RuntimeError(
                f"{self.name}: locus tags outside {KT2440_NAMESPACE}: "
                f"{resolution.reconciliation.outside_namespace}"
            )
        return resolution

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for both families."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for both families."""
        raise NotImplementedError


@register_dataset
class ValerolactamTiterThompson2019Dataset(_Thompson2019Dataset):
    """Thompson 2019 valerolactam titers: the 24 h column of the four-strain ladder."""

    def __init__(
        self,
        root: str = "data/torchcell/valerolactam_titer_thompson2019",
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with the titer family's default dev-tree root."""
        super().__init__(
            root, io_workers, pputida_genome, transform, pre_transform, **kwargs
        )

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return ProductTiterExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return ProductTiterExperimentReference

    @post_process
    def process(self) -> None:
        """Write one titer record per strain at 24 h; refuse the 48 h family."""
        resolution = self._prepare()
        reference_genome = assembly_reference(REFERENCE_STRAIN)
        stored = [cell for cell in TITER_CELLS if cell.hours in TITER_HOURS_STORED]
        refused = [cell for cell in TITER_CELLS if cell.hours not in TITER_HOURS_STORED]
        baseline = next(
            cell
            for cell in stored
            if cell.strain == TITER_REFERENCE_STRAIN and cell.titer_mg_per_l is not None
        )
        reference = ProductTiterExperimentReference(
            dataset_name=self.name,
            genome_reference=reference_genome,
            environment_reference=production_environment(baseline.hours),
            phenotype_reference=titer_phenotype(float(baseline.titer_mg_per_l or 0.0)),
        )
        pub = publication()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        rows: list[dict[str, Any]] = []
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for cell in tqdm(stored, desc="thompson2019-valerolactam-titer"):
                if cell.titer_mg_per_l is None:
                    raise RuntimeError(
                        f"{cell.strain} at {cell.hours} h has no stated titer but is in "
                        "the stored set"
                    )
                experiment = ProductTiterExperiment(
                    dataset_name=self.name,
                    genotype=strain_genotype(cell.strain, resolution.locus_tags),
                    environment=production_environment(cell.hours),
                    phenotype=titer_phenotype(cell.titer_mg_per_l),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, pub, itxn),
                )
                idx += 1
                rows.append(
                    {
                        "strain": cell.strain,
                        "hours": cell.hours,
                        "titer_mg_per_l": cell.titer_mg_per_l,
                        "deletions": " ".join(STRAINS[cell.strain].deletions),
                        "jbei_part_id": STRAINS[cell.strain].jbei_part_id,
                        "quote": cell.quote,
                    }
                )
        env.close()
        interned_env.close()

        pd.DataFrame(rows).to_csv(
            osp.join(self.preprocess_dir, "titer_rows.csv"), index=False
        )
        pd.DataFrame(
            [
                {
                    "strain": cell.strain,
                    "hours": cell.hours,
                    "titer_mg_per_l": cell.titer_mg_per_l,
                    "reason": TITER_48H_REFUSED,
                }
                for cell in refused
            ]
        ).to_csv(osp.join(self.preprocess_dir, "refused_titers.csv"), index=False)
        _write_accounting(
            BuildAccounting(
                dataset=self.name,
                source_rows=len(TITER_CELLS),
                candidate_records=len(TITER_CELLS),
                kept_records=idx,
                dropped_records=len(refused),
                rules=[
                    DropRule(
                        rule="no_released_reference_titer_at_this_time",
                        description=TITER_48H_REFUSED,
                        n_records=len(refused),
                        items=tuple(
                            f"{cell.strain}@{cell.hours:g}h" for cell in refused
                        ),
                    )
                ],
                symbol_locus_tags=resolution.locus_tags,
                goa_resolved=list(resolution.goa),
                notes=[
                    "every record carries the production plasmid pBADT-davBA-ORF26: the "
                    "Methods grow only plasmid-bearing strains for the titer assay "
                    f"('{_Q_PRODUCER}')",
                    f"the reference of every record is {TITER_REFERENCE_STRAIN} at "
                    f"{baseline.hours:g} h, {baseline.titer_mg_per_l} mg/L, which is "
                    "the paper's own fold-change denominator ('a 10-fold increase of "
                    "production at 24 h')",
                    "RB-TnSeq is subsumed, not loaded: " + str(RBTNSEQ_SUBSUMED.note),
                ],
            ),
            self.preprocess_dir,
        )
        log.info(
            "Thompson2019 valerolactam titer: %d records, %d refused; davT -> %s",
            idx,
            len(refused),
            resolution.locus_tags["davT"],
        )


@register_dataset
class LactamGrowthRateThompson2019Dataset(_Thompson2019Dataset):
    """Thompson 2019 Table S1: maximal growth rates on three carbon sources."""

    RAW_PATHS: ClassVar[tuple[str, ...]] = (PAPER_TEXT.relpath, SI_TEXT.relpath)

    def __init__(
        self,
        root: str = "data/torchcell/lactam_growth_rate_thompson2019",
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with the growth family's default dev-tree root."""
        super().__init__(
            root, io_workers, pputida_genome, transform, pre_transform, **kwargs
        )

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialEnvironmentResponseExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialEnvironmentResponseExperimentReference

    @post_process
    def process(self) -> None:
        """Write one record per released Table S1 rate; nothing is dropped."""
        resolution = self._prepare()
        rows = read_table_s1(
            osp.join(self.raw_dir, osp.basename(raw_file(SI_TEXT.relpath).relpath))
        )
        reference_genome = assembly_reference(REFERENCE_STRAIN)
        pub = publication()
        baseline = {
            row.strain: row
            for row in rows
            if row.carbon_source == GROWTH_REFERENCE_CARBON
        }
        if set(baseline) != set(TABLE_S1_STRAINS.values()):
            raise RuntimeError(
                f"Table S1 releases a {GROWTH_REFERENCE_CARBON} rate for "
                f"{sorted(baseline)}; every strain needs one to be a reference"
            )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        ledger: list[dict[str, Any]] = []
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for row in tqdm(rows, desc="thompson2019-valerolactam-growth"):
                genotype = deletion_genotype(row.strain, resolution.locus_tags)
                reference = BacterialEnvironmentResponseExperimentReference(
                    dataset_name=self.name,
                    genome_reference=reference_genome,
                    environment_reference=growth_environment(GROWTH_REFERENCE_CARBON),
                    phenotype_reference=growth_phenotype(
                        baseline[row.strain].rate_per_hour
                    ),
                )
                experiment = BacterialEnvironmentResponseExperiment(
                    dataset_name=self.name,
                    genotype=genotype,
                    environment=growth_environment(row.carbon_source),
                    phenotype=growth_phenotype(row.rate_per_hour),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, pub, itxn),
                )
                idx += 1
                ledger.append(
                    {
                        "strain": row.strain,
                        "carbon_source": row.carbon_source,
                        "compound": TABLE_S1_CARBON[row.carbon_source],
                        "rate_per_hour": row.rate_per_hour,
                        "reference_rate_per_hour": baseline[row.strain].rate_per_hour,
                        "deletions": " ".join(STRAINS[row.strain].deletions),
                    }
                )
        env.close()
        interned_env.close()

        pd.DataFrame(ledger).to_csv(
            osp.join(self.preprocess_dir, "growth_rows.csv"), index=False
        )
        _write_accounting(
            BuildAccounting(
                dataset=self.name,
                source_rows=len(rows),
                candidate_records=len(rows),
                kept_records=idx,
                dropped_records=0,
                rules=[],
                symbol_locus_tags=resolution.locus_tags,
                goa_resolved=list(resolution.goa),
                notes=[
                    "nothing is dropped: every row Table S1 releases is a record",
                    "no record carries the production plasmid: Figs. 2A to 2C grow the "
                    f"wild type and the two single mutants ('{_Q_FIG2}')",
                    f"the reference of each record is the SAME strain on "
                    f"{GROWTH_REFERENCE_CARBON}, the carbon source all three grow on "
                    "identically ('All strains showed identical growth on glucose as a "
                    "sole carbon source'), so three records are their own reference and "
                    "carry an identical value by construction",
                    "the three zero rates are released zeros, not gaps: ΔdavT on "
                    "5-aminovalerate ('the davT mutant predictably was unable to "
                    "grow'), and both mutants on valerolactam ('both the oplBA and "
                    "davT mutants showed no measurable growth after 40 h')",
                ],
            ),
            self.preprocess_dir,
        )
        log.info(
            "Thompson2019 lactam growth: %d records over %d strains and %d carbon "
            "sources",
            idx,
            len(TABLE_S1_STRAINS),
            len(TABLE_S1_CARBON),
        )


# --------------------------------------------------------------------------- #
# Verification, L0 to L4
# --------------------------------------------------------------------------- #
def _strain_of_record(record: dict[str, Any], *, with_plasmid: bool) -> str:
    """The strain a stored record belongs to, matched on its deletion set alone.

    A ``Genotype`` is a set of typed edits and names no strain, so the join back to
    :data:`STRAINS` is by the deleted gene symbols; exactly one strain must match.
    """
    perturbations = record["experiment"]["genotype"]["perturbations"]
    deleted = {
        pert["perturbed_gene_name"]
        for pert in perturbations
        if pert["perturbation_type"] == "bacterial_deletion"
    }
    plasmid = {
        pert["perturbed_gene_name"]
        for pert in perturbations
        if pert["perturbation_type"] == "heterologous_pathway"
    }
    expected_plasmid = {*PLASMID_NATIVE_SYMBOLS, "ORF26"} if with_plasmid else set()
    if plasmid != expected_plasmid:
        raise AssertionError(
            f"a record carries plasmid genes {sorted(plasmid)}, expected "
            f"{sorted(expected_plasmid)}"
        )
    matches = [name for name, spec in STRAINS.items() if set(spec.deletions) == deleted]
    if len(matches) != 1:
        raise AssertionError(
            f"a record's deletions {sorted(deleted)} match {matches}; the genotype no "
            "longer identifies a strain"
        )
    return matches[0]


def verify_titer_build(
    dataset_root: str, data_root: str | None = None
) -> VerificationReport:
    """Run L0 to L4 over the built titer tree and write its report.

    There is no ``run_product_titer`` runner in ``torchcell/verification/runners.py``,
    so the levels are assembled here as the sibling Kang 2026 loader does. L4 re-reads
    the pinned PMC full text and joins each stored titer back to the quote that states
    it, which is the only release this number has.
    """
    from torchcell.verification.runners import load_records

    records = load_records(dataset_root)
    experiments = [record["experiment"] for record in records]
    phenotypes = [exp["phenotype"] for exp in experiments]
    report = VerificationReport(
        dataset_name=ValerolactamTiterThompson2019Dataset.__name__,
        provenance=Provenance(
            source_uri=PAPER_TEXT.relpath,
            citation_key=CITATION_KEY,
            sha256=PAPER_TEXT.sha256,
            method="valerolactam titer in mg/L by HPLC-QTOF MS, stored verbatim as "
            "ug/mL: the 24 h column of the four-strain ladder the Results state",
            page=_RESULTS_PRODUCTION,
            retrieved=RETRIEVED_AT,
        ),
    )
    report.add(l0_structural(experiments, ProductTiterExperiment.model_validate))
    report.add(l1_count(len(records), EXPECTED_TITER_RECORDS))
    report.add(l2_value_fidelity([p["titer"] for p in phenotypes], minimum=0.0))
    report.add(
        l3_convention(
            "titer_unit_is_the_sources_mg_per_l_as_ug_per_ml",
            all(
                p["titer_unit"] == ConcentrationUnit.ug_per_ml.value for p in phenotypes
            ),
            detail="1 mg/L == 1 ug/mL exactly, so the stated number is stored verbatim",
        )
    )
    report.add(
        l3_convention(
            "every_uncertainty_is_a_typed_gap_not_a_guess",
            all(
                p["titer_uncertainty"] is None
                and p["titer_uncertainty_type"] is None
                and {"titer_uncertainty", "titer_uncertainty_type"}
                <= {gap["field"] for gap in p["provenance_gaps"]}
                for p in phenotypes
            ),
            detail="the design is sourced (n = 3 biological replicates, 95% CI) and no "
            "interval number is released anywhere in the mirror",
        )
    )
    report.add(
        l3_convention(
            "every_record_carries_the_production_plasmid",
            all(
                sum(
                    1
                    for pert in exp["genotype"]["perturbations"]
                    if pert["perturbation_type"] == "heterologous_pathway"
                    and pert["construct_name"] == PLASMID_NAME
                )
                == len(PLASMID_NATIVE_SYMBOLS) + 1
                for exp in experiments
            ),
            detail="the titer assay grows only strains harboring pBADT-davBA-ORF26",
        )
    )
    report.add(
        l3_convention(
            "the_reference_is_the_wild_type_at_the_same_sampling_time",
            all(
                record["reference"]["environment_reference"]["duration_hours"]
                == record["experiment"]["environment"]["duration_hours"]
                for record in records
            ),
            detail="no record is measured against a different sampling time, which is "
            "also why the 48 h family is refused",
        )
    )
    report.add(_titer_l4(records, data_root))
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def _titer_l4(records: list[dict[str, Any]], data_root: str | None) -> Any:
    """L4: every stored titer against the quote that states it, re-read from the mirror."""
    pinned = raw_file(PAPER_TEXT.relpath)
    path = raw_mirror_dir(data_root) / pinned.relpath
    digest = sha256_file(path)
    if digest != pinned.sha256:
        raise AssertionError(f"{path} hashes {digest}, not the pinned sha256")
    text = path.read_text(encoding="utf-8")
    stated = {
        (cell.strain, cell.hours): cell
        for cell in TITER_CELLS
        if cell.hours in TITER_HOURS_STORED
    }
    shared = []
    for record in records:
        experiment = record["experiment"]
        strain = _strain_of_record(record, with_plasmid=True)
        hours = experiment["environment"]["duration_hours"]
        cell = stated[(strain, hours)]
        if cell.titer_mg_per_l is None:
            raise AssertionError(f"{strain} at {hours} h is stored but states no titer")
        if cell.quote not in text:
            raise AssertionError(
                f"the quote for {strain} at {hours} h is not verbatim in "
                f"{PAPER_TEXT.relpath}"
            )
        number = f"{cell.titer_mg_per_l} mg/L"
        if number not in cell.quote:
            raise AssertionError(
                f"{number!r} is not in the quote the {strain} record cites"
            )
        shared.append(
            (
                f"{strain}@{hours:g}h",
                experiment["phenotype"]["titer"],
                cell.titer_mg_per_l,
            )
        )
    if len(shared) != len(stated):
        raise AssertionError(f"{len(shared)} records for {len(stated)} stated titers")
    return l4_cross_source(shared, tol=1e-9).model_copy(
        update={"name": "stored_titer_vs_results_prose"}
    )


def verify_growth_build(
    dataset_root: str, data_root: str | None = None
) -> VerificationReport:
    """Run the environment-response L0 to L4 gate over the built growth tree.

    ``reference_centered=False`` because the stored number is an ABSOLUTE maximal
    specific growth rate in h^-1; the absolute branch refuses the request for any record
    whose ``measurement_type`` is not in ``ABSOLUTE_MEASUREMENT_TYPES``.
    ``expected_unperturbed`` is 0: every record carries its carbon source as a typed
    environment perturbation.
    """
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset,
    )
    from torchcell.verification.runners import load_records

    records = load_records(dataset_root)
    report = verify_environment_response_dataset(
        records,
        dataset_name=LactamGrowthRateThompson2019Dataset.__name__,
        provenance=Provenance(
            source_uri=SI_TEXT.relpath,
            citation_key=CITATION_KEY,
            sha256=SI_TEXT.sha256,
            method="Supplementary Table S1: one BacterialEnvironmentResponseExperiment "
            "per released (strain, carbon source) maximal specific growth rate in h^-1, "
            "MeasurementType.growth_rate; the reference is the same strain on glucose",
            page="Table S1",
            retrieved=RETRIEVED_AT,
        ),
        expected_count=EXPECTED_GROWTH_RECORDS,
        reference_centered=False,
        expected_unperturbed=0,
    )
    report.add(_growth_l4(records, data_root))
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def _growth_l4(records: list[dict[str, Any]], data_root: str | None) -> Any:
    """L4: every stored rate against Table S1, re-read from the pinned SI text layer."""
    path = raw_mirror_dir(data_root) / raw_file(SI_TEXT.relpath).relpath
    released = {
        (row.strain, TABLE_S1_CARBON[row.carbon_source]): row.rate_per_hour
        for row in read_table_s1(path)
    }
    shared = []
    for record in records:
        experiment = record["experiment"]
        strain = _strain_of_record(record, with_plasmid=False)
        agents = [
            pert["agent"]["name"]
            for pert in experiment["environment"]["perturbations"]
            if pert["perturbation_type"] == "environment_physical"
        ]
        if len(agents) != 1:
            raise AssertionError(
                f"a record names {len(agents)} carbon sources; exactly one is required"
            )
        key = (strain, agents[0])
        if key not in released:
            raise AssertionError(f"{key} is not a row Table S1 releases")
        shared.append(
            (
                f"{strain}|{agents[0]}",
                experiment["phenotype"]["environment_response"],
                released[key],
            )
        )
    if len(shared) != len(released):
        raise AssertionError(f"{len(shared)} records for {len(released)} released rows")
    return l4_cross_source(shared, tol=1e-9).model_copy(
        update={"name": "stored_growth_rate_vs_table_s1"}
    )


def main() -> None:
    """Build both dev-tree datasets and run their verifiers, for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = _data_root()
    genome = bacterial_genome("pputida", "KT2440", data_root)
    for cls, rel, verifier in (
        (
            ValerolactamTiterThompson2019Dataset,
            "data/torchcell/valerolactam_titer_thompson2019",
            verify_titer_build,
        ),
        (
            LactamGrowthRateThompson2019Dataset,
            "data/torchcell/lactam_growth_rate_thompson2019",
            verify_growth_build,
        ),
    ):
        root = osp.join(data_root, rel)
        dataset = cls(root=root, pputida_genome=genome)
        print(f"{cls.__name__}: len = {len(dataset)}")
        dataset.close_lmdb()
        print(
            json.dumps(
                json.loads(
                    Path(root, "preprocess", "build_accounting.json").read_text()
                ),
                indent=2,
            )
        )
        print(verifier(root, data_root).summary())


if __name__ == "__main__":
    main()
