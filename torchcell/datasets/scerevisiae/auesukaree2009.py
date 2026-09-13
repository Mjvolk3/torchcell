# torchcell/datasets/scerevisiae/auesukaree2009
# [[torchcell.datasets.scerevisiae.auesukaree2009]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/auesukaree2009
# Test file: tests/torchcell/datasets/scerevisiae/test_auesukaree2009.py
"""Auesukaree 2009 multi-stress tolerance screen (env x geno -> categorical response).

Auesukaree et al. 2009 (J Appl Genet 50(3):301-310, doi:10.1007/BF03195688; PMC2747848)
screened the 4828-strain S. cerevisiae nonessential HAPLOID single-deletion collection
(Saccharomyces Genome Deletion Project / Research Genetics-Invitrogen, BY4742 MATalpha
background) for growth SENSITIVITY to six environmental stresses on solid YPD: three
alcohols (10% ethanol, 16% methanol, 7% 1-propanol), heat (37 C), high osmolarity
(1 M NaCl), and oxidative stress (5 mM H2O2). This is a plain haploid deletion collection
(NOT a HIP/HOP heterozygous screen), so every screened strain is a single-gene KanMX
deletion in BY4742 and there is no engineered-heterozygous / dosage complication.

READOUT -- a CATEGORICAL sensitivity call by serial-dilution spot assay
(``assay_type=spot_dilution``, ``_ASSAY``), and the comparison is WITHIN a strain, not
against the parent: ``_SCORING`` quotes the Methods definition, "as compared to those
under non-stress conditions". So the record's reference is the SAME deletion strain on the
UNPERTURBED plate (shared ``media.YPD_AGAR``, 30 C, no perturbation, 72 h), and
``MEASUREMENT_UNITS`` says exactly that. ``ReferenceGenome`` cannot express "the same
deletion strain", so ``genome_reference`` stays BY4742 and the within-strain comparator is
stated in ``units`` and in the dendron note. The reference phenotype is
``ResponseCategory.no_change`` with the source's own word "tolerant" in ``category_label``;
the experiment is ``ResponseCategory.sensitive``, label "sensitive".

``sensitive`` is deliberately NOT mapped to ``reduced``: the paper reports only an
UNGRADED hit call, so a severity grade would be invented. The reference maps to
``no_change`` because "tolerant" here means indistinguishable from the unstressed plate.

The paper reports ONLY the SENSITIVE hits (enumerated per stress in Tables 1-6, grouped by
functional class); the non-sensitive majority is NOT listed. So this dataset carries one
categorical ``sensitive`` record per (sensitive deletion x stress) pair. No mutant
conferred INCREASED tolerance in this study, so there is no ``resistant`` record.

ENVIRONMENT -- the medium is the SHARED ``media.YPD_AGAR`` object. The paper independently
restates the YPD recipe (``_MEDIUM``: "YPD medium (1% yeast extract, 2% peptone, 2%
glucose)") and this screen is stamped onto PLATES, so the plate object is the right member
of the family: ``media.YPD`` is the join anchor a loader does not use directly, ``YPD_AGAR``
is the plate that states its agar row, and ``YPD_LIQUID`` is the culture. Auesukaree's
plates therefore share one media node with Mota 2024's, instead of sitting on a name-only
``Media(name="YPD")`` that joins nothing. The edit is:
- three alcohols + NaCl + H2O2 -> ``SmallMoleculePerturbation`` (an added chemical
  SPECIES; NaCl / H2O2 are named compounds, not abstract "physical stresses" -- their
  osmotic / oxidative framing is a CONSEQUENCE and belongs to the phenotype or to the
  compound's ChEBI role, M1);
- heat -> NO perturbation object, the raised ``Environment.temperature`` (37 C vs the
  30 C base) alone, temperature having one canonical home (M2).

SOURCING -- every metadata number is a ``SourcedValue`` quoting the mirrored OCR
``paper.md`` (``_MEDIUM``, ``_SCREEN_CONDITIONS``, ``_DOSES``, ``_N_SAMPLES``, ``_ASSAY``,
``_SCORING``, ``_BACKGROUND``), with two readings recorded as notes rather than as quotes:
- ``percent_v/v`` for the three alcohols. The paper writes "10% ethanol" and never writes
  "v/v" or "w/v" anywhere (grep over the full text layer: 0 hits). Alcohol percentages in
  yeast stress work are conventionally v/v; that convention is the reason for the unit and
  it is recorded as such, not as something the source said.
- ``sample_unit=biological_replicate``. "All experiments ... were performed in triplicate"
  does not say biological or technical; each replicate is an independent screening
  experiment of the whole collection, which is the biological-replicate reading.

DATA SOURCE (the sensitive-gene lists themselves):
- The per-stress lists are the article Tables 1-6, extracted deterministically from the
  BORN-DIGITAL text layer of the mirrored publisher PDF via ``pdftotext -layout``
  (poppler; the version used is recorded in the raw mirror's manifest). There is no
  separate data deposit -- the data IS the article tables. Each table row declares its
  functional-class member count in parentheses (e.g. "Vacuolar function (16)"); the parser
  uses these declared counts as per-class self-checksums, so extraction drift is caught at
  build time. The MinerU OCR ``paper.md`` in the same mirror is lossy for the TABLES (its
  heat Table 4 "Unknown function" cell is truncated by two ORFs, YOR364w and YPL144w), so
  the PDF text layer is the source of record for the gene lists while ``paper.md`` is the
  anchor for the prose quotes.

SOURCE COUNT NOTES (documented, not guessed):
- Per-stress listed sensitive genes match the abstract/Figure headline counts EXACTLY for
  five stresses: ethanol 95, 1-propanol 125, heat 178, NaCl 42, H2O2 30.
- METHANOL is the one internal source discrepancy: the abstract says 54 methanol-sensitive
  mutants, but Table 2 lists 55 distinct genes and its class parentheticals also sum to 55
  (11+7+4+5+2+5+2+2+4+5+1+7). The Table is the per-gene DATA and is authoritative over the
  summary headline, so 55 methanol records are stored and the 1-gene discrepancy is flagged.

GENE RESOLUTION -- every token goes through the SHARED, layered
``SCerevisiaeGenome.resolve_gene_name``; ``AMBIGUOUS`` is a HARD STOP, never a silent
first-match. Measured over the 333 listed tokens: 283 RENAMED, 48 CURRENT, 2 AMBIGUOUS,
0 dropped. The two ambiguous ones are adjudicated by source evidence in
``_AMBIGUOUS_ADJUDICATIONS``; every other token is stored verbatim in
``perturbed_gene_name`` (it round-trips to its own systematic name). Final: ethanol 95,
methanol 55, 1-propanol 125, heat 178, NaCl 42, H2O2 30 = 525 categorical records, none
dropped.
"""

import hashlib
import logging
import os
import os.path as osp
import pickle
import re
import shutil
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lmdb
from pydantic import BaseModel, Field
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import YPD_AGAR
from torchcell.datamodels.schema import (
    AssayType,
    BiologicPerturbation,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    KanMxDeletionPerturbation,
    MeasurementType,
    Publication,
    ReferenceGenome,
    ResponseCategory,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_PAPER_PDF,
    ArtifactRecord,
    Manifest,
    ProcessingRecord,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome
from torchcell.sequence.genome.scerevisiae.s288c import GeneNameStatus
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DOI = "10.1007/BF03195688"
CITATION_KEY = "auesukareeGenomewideIdentificationGenes2009"

# Two artifacts of the SAME paper, both in the library mirror and both pinned:
#   paper.pdf -- born-digital; its text layer carries the COMPLETE Tables 1-6 (the data).
#   paper.md  -- the MinerU OCR; lossy for the tables, but it is flowing text and so is
#                the anchor every prose SourcedValue quote is checked against.
_PDF_FILENAME = "paper.pdf"
_PDF_SHA256 = "01b945443c0ce41642c76fd737e12b4c31cacb5f384049a5c0a7e4bf9e1eb5a1"
_PAPER_MD = "paper.md"
_PAPER_MD_SHA256 = "d0f3885d1f5027fc29beab7a4327ff377d2bc9c42dd5b87f580049eeda4223b2"

_DROPPED_FILENAME = "dropped_records.json"


def _sv(value: Any, quote: str, note: str | None = None) -> SourcedValue:
    """A value sourced from the mirrored OCR of the paper."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=_PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=_PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (library mirror)",
            page="J Appl Genet 2009 50(3):301-310, Materials and methods",
        ),
    )


_BACKGROUND = _sv(
    "BY4742",
    "The 4828 nonessential haploid S. cerevisiae deletion strains generated by the "
    "Saccharomyces Genome Deletion Project (Winzeler et al. 1999) were obtained from "
    "Research Genetics/Invitrogen.",
    note="a plain nonessential HAPLOID deletion collection, not HIP/HOP; the BY4742 "
    "auxotrophies are the collection background and are carried by "
    "ReferenceGenome(strain='BY4742'), not modelled as perturbations",
)

_MEDIUM = _sv(
    YPD_AGAR,
    "YPD medium ( $1 \\%$ yeast extract, $2 \\%$ peptone, $2 \\%$ glucose), with the "
    "optional addition of $2 0 0 \\ \\mathrm { ~ m g \\ L ^ { - 1 } }$ geneticin "
    "(Sigma-Aldrich), was used as a rich medium for yeast growth.",
    note="component-for-component the shared media.YPD core (yeast extract 1% w/v, "
    "peptone 2% w/v, D-glucose 2% w/v), which is why the library object is used verbatim; "
    "the SOLID member YPD_AGAR is the one this screen stamps onto. The "
    "geneticin is explicitly OPTIONAL and belongs to the pre-screen replicator plates, "
    "not to the scored plates (see _SCREEN_CONDITIONS), so it is not a component here",
)

_SCREEN_CONDITIONS = _sv(
    (30.0, 37.0, 72.0),
    "After incubation at $3 0 ^ { \\circ } \\mathrm { C }$ for 2 days, strains were "
    "stamped onto YPD plates and YPD plates containing $10 \\%$ ethanol, $1 6 \\%$ "
    "methanol, $7 \\%$ propanol, $1 \\ : \\mathrm { M } \\ : \\mathrm { N a C l }$ , or "
    "$5 \\mathrm { m M }$ $\\mathrm { H } _ { 2 } \\mathrm { O } _ { 2 }$ , and grown at "
    "$3 0 ^ { \\circ } \\mathrm { C }$ for a further 3 days. Screening for "
    "heat-sensitive deletion mutants was performed at $3 7 ^ { \\circ } \\mathrm { C } .$",
    note="(base temperature C, heat temperature C, duration hours): the scored plates "
    "are plain YPD (no geneticin), 30 C for the five chemical stresses and 37 C for "
    "heat, 3 days = 72 h",
)

_DOSES = _sv(
    {
        "ethanol": 10.0,
        "methanol": 16.0,
        "1-propanol": 7.0,
        "sodium chloride": 1.0,
        "hydrogen peroxide": 5.0,
    },
    "We found that the maximum non-inhibitory concentrations and temperature were: "
    "$10 \\%$ ethanol, $1 6 \\%$ methanol, $7 \\%$ 1-propanol, $1 \\ : \\mathrm { M } \\ "
    ": \\mathrm { N a C l }$ , 5 mM $\\mathrm { H } _ { 2 } \\mathrm { O } _ { 2 }$ , and "
    "$3 7 ^ { \\circ } \\mathrm { C }$ (Figure 1).",
    note="each dose is the MAXIMUM NON-INHIBITORY value for the parent strain, fixed by "
    "the spot test of Figure 1. The alcohol percentages are stored as percent v/v; the "
    "paper writes neither 'v/v' nor 'w/v' anywhere, so the basis is the standard "
    "alcohol-stress convention, a reading and not a quote",
)

_N_SAMPLES = _sv(
    3,
    "All experiments for the identification of stress-sensitive mutants were performed "
    "in triplicate.",
    note="the paper does not say whether the triplicate is biological or technical; "
    "each replicate is an independent screening experiment of the whole collection, "
    "which is the sample_unit=biological_replicate reading, an interpretation",
)

_ASSAY = _sv(
    AssayType.spot_dilution,
    "Appropriate conditions for the evaluation of sensitivity to ethanol, methanol, "
    "1-propanol, NaCl, $\\mathrm { H } _ { 2 } \\mathrm { O } _ { 2 }$ , and heat, were "
    "determined by the serial-dilution spot test.",
)

_SCORING = _sv(
    ResponseCategory.sensitive,
    "Deletion mutants showing significantly reduced growth on plates under stress "
    "conditions, as compared to those under non-stress conditions, were defined as "
    "sensitive mutants.",
    note="the comparator is the SAME strain on the non-stress plate, which is why the "
    "reference environment is the UNPERTURBED base environment rather than a copy of "
    "the stressed one. 'sensitive' is an ungraded hit call, so it maps to "
    "ResponseCategory.sensitive and never to a graded member such as reduced",
)

#: The source's own words, kept verbatim in ``category_label`` alongside the typed call.
_SENSITIVE_LABEL = "sensitive"
_REFERENCE_LABEL = "tolerant"

MEASUREMENT_UNITS = (
    "serial-dilution spot-assay growth call, each deletion strain on the stress plate "
    "vs the SAME strain on the matched non-stress YPD plate after 3 days: 'sensitive' = "
    "significantly reduced growth under stress vs non-stress conditions; the unlisted "
    "majority is 'tolerant'"
)

# Per-table caption stress token (as it appears in the PDF captions) -> expected number of
# LISTED sensitive genes (the parenthetical class counts sum; a build-time self-checksum).
# Ethanol/1-propanol/heat/NaCl/H2O2 match the abstract headline; methanol Table 2 lists 55
# (abstract headline says 54 -- documented source discrepancy, Table is authoritative).
_EXPECTED_LISTED: dict[str, int] = {
    "ethanol": 95,
    "methanol": 55,
    "1-propanol": 125,
    "heat": 178,
    "NaCl": 42,
    "H2O2": 30,
}

_BASE_TEMPERATURE_C, _HEAT_TEMPERATURE_C, _DURATION_HOURS = _SCREEN_CONDITIONS.value

# Per-stress environment spec. ``kind`` selects how the edit is modeled:
#   - "small_molecule": an added chemical SPECIES.
#   - "temperature": NO perturbation object; the edit is the raised
#     ``Environment.temperature`` (37 C) alone -- temperature has ONE canonical home (M2).
_STRESS_SPECS: list[dict[str, Any]] = [
    {
        "stress": "ethanol",
        "kind": "small_molecule",
        "compound_name": "ethanol",
        "unit": ConcentrationUnit.percent_v_v,
        "temperature": _BASE_TEMPERATURE_C,
    },
    {
        "stress": "methanol",
        "kind": "small_molecule",
        "compound_name": "methanol",
        "unit": ConcentrationUnit.percent_v_v,
        "temperature": _BASE_TEMPERATURE_C,
    },
    {
        "stress": "1-propanol",
        "kind": "small_molecule",
        "compound_name": "1-propanol",
        "unit": ConcentrationUnit.percent_v_v,
        "temperature": _BASE_TEMPERATURE_C,
    },
    {"stress": "heat", "kind": "temperature", "temperature": _HEAT_TEMPERATURE_C},
    {
        "stress": "NaCl",
        "kind": "small_molecule",
        "compound_name": "sodium chloride",
        "unit": ConcentrationUnit.molar,
        "temperature": _BASE_TEMPERATURE_C,
    },
    {
        "stress": "H2O2",
        "kind": "small_molecule",
        "compound_name": "hydrogen peroxide",
        "unit": ConcentrationUnit.millimolar,
        "temperature": _BASE_TEMPERATURE_C,
    },
]


class AmbiguousAdjudication(BaseModel):
    """One ``AMBIGUOUS`` gene token resolved by source evidence, never by first-match."""

    token: str = Field(description="the paper's gene name, verbatim")
    candidates: list[str] = Field(description="the systematic names the alias maps to")
    systematic_name: str = Field(description="the one this dataset stores")
    stored_gene_name: str = Field(
        description="the genome's canonical common name for that gene; the ambiguous "
        "source token cannot be stored because it resolves to no single gene"
    )
    evidence: str = Field(description="why the other candidate(s) are excluded")


#: Two of the 333 listed tokens are aliases of more than one current gene. The shared
#: resolver refuses to choose (status AMBIGUOUS, systematic_name None) and this loader
#: raises on any token not listed here, so no ambiguity is ever silently first-matched.
_AMBIGUOUS_ADJUDICATIONS: dict[str, AmbiguousAdjudication] = {
    "PPA1": AmbiguousAdjudication(
        token="PPA1",
        candidates=["YBR011C", "YHR026W"],
        systematic_name="YHR026W",
        stored_gene_name="VMA16",
        evidence=(
            "YBR011C is IPP1, an ESSENTIAL gene: Costanzo 2021's Data File S1 carries it "
            "only as a temperature-sensitive strain ('YBR011C, IPP1, ipp1-5001, "
            "tsa1062'), so it cannot be in the 4828-strain NONESSENTIAL haploid deletion "
            "collection this paper screened. YHR026W is VMA16/PPA1, the V-ATPase V0 "
            "subunit c'', and the paper lists PPA1 in the 'Vacuolar function' class in "
            "BOTH tables it appears in (Table 3 1-propanol: 'Vacuolar function (16) "
            "ARP5, BRO1, FAB1, PEP12, PIB2, PPA1, ...'; Table 4 heat: 'Vacuolar function "
            "(16) BRO1, CUP5, DID4, PEP7, PEP12, PPA1, ...')"
        ),
    ),
    "FEN1": AmbiguousAdjudication(
        token="FEN1",
        candidates=["YCR034W", "YKL113C"],
        systematic_name="YCR034W",
        stored_gene_name="ELO2",
        evidence=(
            "both candidates are nonessential, so essentiality does not decide. The SAME "
            "table that lists FEN1 lists the other candidate SEPARATELY under its own "
            "name: Table 2 (methanol) has 'Vacuolar function (11) CHC1, FEN1, VAM3, "
            "VMA6, VMA8, VMA21, VPH2, VPS1, VPS4, VPS16, VPS25' and 'Cell rescue, "
            "defense and virulence (5) AFT2, FLC1, FYV6, RAD27, SOD1'. RAD27 IS "
            "YKL113C, so FEN1 in that table is the other candidate, YCR034W (ELO2)"
        ),
    ),
}

#: Retention rule: a token is kept only when it resolves to a LIVE R64 gene.
_KEPT_STATUSES = frozenset({GeneNameStatus.CURRENT, GeneNameStatus.RENAMED})

DROP_RULE = (
    "a listed gene token resolves through SCerevisiaeGenome.resolve_gene_name to a "
    "status other than CURRENT or RENAMED, and is not one of the two AMBIGUOUS tokens "
    "adjudicated by source evidence in _AMBIGUOUS_ADJUDICATIONS (an unlisted AMBIGUOUS "
    "token raises rather than being first-matched). Measured over this release: 0 tokens "
    "are dropped"
)

RAW_MIRROR_REL = f"torchcell-raw/{CITATION_KEY}"
_PDF_RAW_RELPATH = f"paper/{_PDF_FILENAME}"

_CAPTION_RE = re.compile(
    r"^\s*Table [1-6]\. Classification of genes whose deletions result in "
    r"(.+?) sensitivity\s*$"
)
# A functional-class row: "<Class words> (<count>)   <gene, gene, ...>".
_CLASS_RE = re.compile(r"^\s*[A-Za-z].*?\((\d+)\)\s{2,}(\S.*)$")
_CONTINUATION_RE = re.compile(r"^\s{2,}\S")


def _tokenize(cell: str) -> list[str]:
    """Split a table gene cell into gene tokens (comma/whitespace separated)."""
    return [t for t in re.split(r"[,\s]+", cell.strip()) if t]


def poppler_version() -> str:
    """The ``pdftotext`` version string, recorded as extraction provenance."""
    result = subprocess.run(
        ["pdftotext", "-v"], capture_output=True, text=True, check=True
    )
    return (result.stderr or result.stdout).strip().split("\n")[0]


def raw_mirror_dir(data_root: str | None = None) -> str:
    """``$DATA_ROOT/torchcell-raw/auesukareeGenomewideIdentificationGenes2009``."""
    return osp.join(data_root or os.environ["DATA_ROOT"], RAW_MIRROR_REL)


def deposit_raw_mirror(
    *, source_pdf: str, retrieved_at: str, data_root: str | None = None
) -> str:
    """Copy the ONE consumed file (``paper.pdf``) into the raw mirror + write its manifest.

    The article tables ARE the data, so the raw mirror holds the born-digital PDF and
    nothing else. Its bytes came from the Zotero attachment the library mirror records
    (item ``IGDTEZJV``, attachment ``VIJCFVIA``); PMC file downloads use a JS
    proof-of-work and are not scriptable, which is why Zotero is the retrieval route. The
    ``pdftotext`` version this build extracted with is recorded as the processing step.
    """
    root = Path(raw_mirror_dir(data_root))
    dest = root / _PDF_RAW_RELPATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(Path(source_pdf))
    if digest != _PDF_SHA256:
        raise RuntimeError(
            f"{source_pdf} sha256 {digest} != pinned {_PDF_SHA256}; refusing to deposit"
        )
    if dest.exists():
        if sha256_file(dest) != _PDF_SHA256:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    else:
        dest.write_bytes(Path(source_pdf).read_bytes())
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=(
            "Genome-wide identification of genes involved in tolerance to various "
            "environmental stresses in Saccharomyces cerevisiae"
        ),
        library_id="6582362",
        zotero_item_key="IGDTEZJV",
        files=[
            ArtifactRecord(
                path=_PDF_RAW_RELPATH,
                role=ROLE_PAPER_PDF,
                bytes=dest.stat().st_size,
                sha256=_PDF_SHA256,
                source="zotero:attachment:VIJCFVIA",
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.zotero_attachment,
                    source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC2747848/",
                    retriever="torchcell.literature.zotero.ZoteroLibrary.download_artifact",
                    params={
                        "library_id": "6582362",
                        "zotero_item_key": "IGDTEZJV",
                        "attachment_key": "VIJCFVIA",
                        "citation_key": CITATION_KEY,
                    },
                    sha256=_PDF_SHA256,
                    retrieved_at=retrieved_at,
                ),
                processing=ProcessingRecord(
                    processor=(
                        "torchcell.datasets.scerevisiae.auesukaree2009."
                        "EnvChemgenAuesukaree2009Dataset._parse_tables"
                    ),
                    tool="pdftotext",
                    version=poppler_version(),
                    params={"args": ["-layout"]},
                    input_sha256=[_PDF_SHA256],
                ),
            )
        ],
        si_data_sources=[],
        si_expected=[
            "none -- the article Tables 1-6 ARE the data; this paper released no "
            "supplementary data file"
        ],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return str(root)


class DropLog(BaseModel):
    """The build's retention accounting, written beside ``processed/``."""

    dataset: str
    rule: str
    n_listed_tokens: int
    n_kept_records: int
    n_dropped_records: int
    dropped_tokens: dict[str, int] = Field(default_factory=dict)
    adjudicated: list[AmbiguousAdjudication] = Field(default_factory=list)


@register_dataset
class EnvChemgenAuesukaree2009Dataset(ExperimentDataset):
    """Auesukaree 2009 six-stress env x geno -> categorical sensitivity dataset."""

    def __init__(
        self,
        root: str = "data/torchcell/env_chemgen_auesukaree2009",
        io_workers: int = 0,
        genome: SCerevisiaeGenome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset; a genome is REQUIRED for common-name -> ORF mapping."""
        self.genome = genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return EnvironmentResponseExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return EnvironmentResponseExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The mirrored publisher PDF (born-digital source of the sensitive-gene tables)."""
        return [_PDF_FILENAME]

    def download(self) -> None:
        """Copy the pinned PDF from the raw mirror into ``raw_dir``; verify its sha256."""
        os.makedirs(self.raw_dir, exist_ok=True)
        dest = osp.join(self.raw_dir, _PDF_FILENAME)
        if not osp.exists(dest):
            src = osp.join(raw_mirror_dir(), _PDF_RAW_RELPATH)
            if not osp.exists(src):
                raise RuntimeError(
                    f"raw-mirror PDF not found: {src}. Deposit it with "
                    "deposit_raw_mirror() from the library mirror's paper.pdf."
                )
            shutil.copyfile(src, dest)
        digest = hashlib.sha256(open(dest, "rb").read()).hexdigest()
        if digest != _PDF_SHA256:
            raise RuntimeError(
                f"{_PDF_FILENAME} sha256 mismatch: got {digest}, expected {_PDF_SHA256}"
            )
        log.info("Verified %s (sha256 %s)", dest, _PDF_SHA256)

    def _parse_tables(self) -> dict[str, list[str]]:
        """Extract {stress: [gene tokens]} from the born-digital PDF text layer.

        Uses ``pdftotext -layout`` (poppler); each functional-class row declares its member
        count in parentheses, used as a per-class self-checksum against extraction drift.
        """
        pdf_path = osp.join(self.raw_dir, _PDF_FILENAME)
        text = subprocess.run(
            ["pdftotext", "-layout", pdf_path, "-"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        lines = text.split("\n")
        tables: dict[str, list[str]] = {}
        i = 0
        while i < len(lines):
            caption = _CAPTION_RE.match(lines[i])
            if not caption:
                i += 1
                continue
            stress = caption.group(1)
            i += 1
            genes: list[str] = []
            cur_declared = 0
            cur_count = 0
            while i < len(lines):
                line = lines[i]
                if line.strip() == "" or _CAPTION_RE.match(line):
                    break
                class_row = _CLASS_RE.match(line)
                if class_row:
                    cur_declared = int(class_row.group(1))
                    toks = _tokenize(class_row.group(2))
                    cur_count = len(toks)
                    genes.extend(toks)
                elif _CONTINUATION_RE.match(line) and cur_count < cur_declared:
                    toks = _tokenize(line)
                    cur_count += len(toks)
                    genes.extend(toks)
                else:
                    # not a table row (footer / prose) -> table body ended
                    break
                i += 1
            tables[stress] = genes
        return tables

    def _resolve_token(self, token: str) -> tuple[str, str] | None:
        """``(current systematic name, the common name to store)``, or None to drop.

        An ``AMBIGUOUS`` token is a hard stop: it is resolved only from the curated,
        evidence-carrying ``_AMBIGUOUS_ADJUDICATIONS`` table, and an unlisted one raises.
        For an adjudicated token the genome's canonical common name is stored, because
        the source token itself resolves to no single gene and so could not round-trip.
        """
        if self.genome is None:
            raise RuntimeError(
                "EnvChemgenAuesukaree2009Dataset requires a genome for gene-name "
                "resolution; inject SCerevisiaeGenome(...)"
            )
        resolution = self.genome.resolve_gene_name(token)
        if resolution.status is GeneNameStatus.AMBIGUOUS:
            adjudication = _AMBIGUOUS_ADJUDICATIONS.get(token.upper())
            if adjudication is None:
                raise RuntimeError(
                    f"gene token {token!r} is AMBIGUOUS "
                    f"(candidates {resolution.candidates}) and has no evidence-based "
                    "adjudication in _AMBIGUOUS_ADJUDICATIONS; refusing to first-match"
                )
            return adjudication.systematic_name, adjudication.stored_gene_name
        if resolution.status not in _KEPT_STATUSES:
            return None
        return str(resolution.systematic_name), token

    def _resolve_stress(
        self, stress: str, tokens: list[str]
    ) -> tuple[dict[str, str], dict[str, int]]:
        """Resolve one stress's tokens to ``{ORF: stored name}``; checksum + count drops."""
        expected = _EXPECTED_LISTED[stress]
        if len(tokens) != expected:
            raise RuntimeError(
                f"{stress}: parsed {len(tokens)} listed genes, expected {expected} "
                f"(table extraction self-checksum failed)"
            )
        resolved: dict[str, str] = {}
        dropped: dict[str, int] = {}
        for token in tokens:
            hit = self._resolve_token(token)
            if hit is None:
                dropped[token] = dropped.get(token, 0) + 1
                continue
            orf, stored = hit
            resolved.setdefault(orf, stored)
        log.info(
            "Auesukaree2009 %s: %d listed -> %d unique-ORF records (dropped %d: %s)",
            stress,
            len(tokens),
            len(resolved),
            sum(dropped.values()),
            sorted(dropped),
        )
        return resolved, dropped

    def _environment(self, spec: dict[str, Any]) -> Environment:
        """Aerobic solid-YPD plate carrying the edit (an added small molecule, or heat)."""
        perturbations: list[
            SmallMoleculePerturbation
            | EnvironmentPhysicalPerturbation
            | BiologicPerturbation
        ] = []
        if spec["kind"] == "small_molecule":
            perturbations.append(
                SmallMoleculePerturbation(
                    compound=resolved_compound(spec["compound_name"]),
                    concentration=Concentration(
                        value=_DOSES.value[spec["compound_name"]], unit=spec["unit"]
                    ),
                )
            )
        return Environment(
            media=YPD_AGAR,
            temperature=Temperature(value=spec["temperature"]),
            perturbations=perturbations,
            aerobicity="aerobic",
            duration_hours=_DURATION_HOURS,
        )

    def _reference(self) -> EnvironmentResponseExperimentReference:
        """The matched NON-STRESS plate: unperturbed YPD at 30 C, growth unchanged.

        The paper's own definition compares a mutant on the stress plate to that mutant on
        the non-stress plate, so the reference environment carries NO perturbation and the
        base 30 C temperature -- including for the heat records, whose comparator is the
        same collection at 30 C.
        """
        return EnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain=_BACKGROUND.value
            ),
            environment_reference=Environment(
                media=YPD_AGAR,
                temperature=Temperature(value=_BASE_TEMPERATURE_C),
                perturbations=[],
                aerobicity="aerobic",
                duration_hours=_DURATION_HOURS,
            ),
            phenotype_reference=EnvironmentResponsePhenotype(
                measurement_type=MeasurementType.categorical,
                assay_type=_ASSAY.value,
                category=ResponseCategory.no_change,
                category_label=_REFERENCE_LABEL,
                n_samples=_N_SAMPLES.value,
                sample_unit=SampleUnit.biological_replicate,
                units=MEASUREMENT_UNITS,
            ),
        )

    def _experiment(
        self, *, orf: str, gene_name: str, environment: Environment
    ) -> EnvironmentResponseExperiment:
        """Build one env x geno -> categorical-sensitivity experiment for (gene, stress)."""
        genotype = Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=gene_name
                )
            ]
        )
        phenotype = EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.categorical,
            assay_type=_ASSAY.value,
            category=_SCORING.value,
            category_label=_SENSITIVE_LABEL,
            n_samples=_N_SAMPLES.value,
            sample_unit=SampleUnit.biological_replicate,
            units=MEASUREMENT_UNITS,
        )
        return EnvironmentResponseExperiment(
            dataset_name=self.name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )

    @post_process
    def process(self) -> None:
        """Parse the six stress tables into categorical records; write LMDB + drop log."""
        tables = self._parse_tables()
        publication = Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")
        pub_dump = publication.model_dump()
        ref_dump = self._reference().model_dump()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        n_listed = 0
        dropped_tokens: dict[str, int] = {}
        with env.begin(write=True) as txn:
            for spec in _STRESS_SPECS:
                stress = spec["stress"]
                if stress not in tables:
                    raise RuntimeError(
                        f"stress table not found in PDF: {stress!r} "
                        f"(parsed: {sorted(tables)})"
                    )
                environment = self._environment(spec)
                n_listed += len(tables[stress])
                orf_names, dropped = self._resolve_stress(stress, tables[stress])
                for token, count in dropped.items():
                    dropped_tokens[token] = dropped_tokens.get(token, 0) + count
                for orf, gene_name in tqdm(sorted(orf_names.items()), desc=f"{stress}"):
                    experiment = self._experiment(
                        orf=orf, gene_name=gene_name, environment=environment
                    )
                    txn.put(
                        f"{idx}".encode(),
                        pickle.dumps(
                            {
                                "experiment": experiment.model_dump(),
                                "reference": ref_dump,
                                "publication": pub_dump,
                            }
                        ),
                    )
                    idx += 1
        env.close()

        drop_log = DropLog(
            dataset=self.name,
            rule=DROP_RULE,
            n_listed_tokens=n_listed,
            n_kept_records=idx,
            n_dropped_records=sum(dropped_tokens.values()),
            dropped_tokens=dropped_tokens,
            adjudicated=list(_AMBIGUOUS_ADJUDICATIONS.values()),
        )
        with open(osp.join(self.root, _DROPPED_FILENAME), "w") as handle:
            handle.write(drop_log.model_dump_json(indent=2))
        log.info(
            "Wrote %d Auesukaree2009 environment-response experiments to LMDB "
            "(%d listed tokens, %d dropped)",
            idx,
            n_listed,
            drop_log.n_dropped_records,
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
    root = osp.join(data_root, "data/torchcell/env_chemgen_auesukaree2009")
    dataset = EnvChemgenAuesukaree2009Dataset(root=root, genome=genome)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()
