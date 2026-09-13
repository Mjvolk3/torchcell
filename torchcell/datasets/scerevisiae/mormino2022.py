# torchcell/datasets/scerevisiae/mormino2022
# [[torchcell.datasets.scerevisiae.mormino2022]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/mormino2022
# Test file: tests/torchcell/datasets/scerevisiae/test_mormino2022.py
"""Mormino 2022 Haa1-biosensor CRISPRi acetic-acid screen (12 isolated strains).

Mormino, Lenitz, Siewers & Nygard 2022 (Microb Cell Fact 21:214,
doi:10.1186/s12934-022-01938-7; PMID 36284296; PMC9571444) integrated a Haa1-based
acetic-acid biosensor into a pooled S. cerevisiae CRISPRi strain library (dCas9-Mxi1
repressing each essential / respiratory-growth-essential gene; library from Smith JD et
al. 2017, derived from BY4742) to make the CRISPRi Biosensor Library (CBL). FACS enriched
cells with the highest reporter signal, and a higher biosensor signal marks cells more
sensitive to acetic acid. Table 1 reports the 12 individually isolated strains.

RECORD = one isolated strain as an ``EnvironmentResponseExperiment``:

- GENOTYPE: one ``CrisprInterferencePerturbation`` (target gene, effector ``dCas9-Mxi1``,
  ``n_guides=1`` -- an isolated strain carries exactly one gRNA; the "1-16 gRNAs each" in
  the paper is a per-GENE property of the library, not of a strain). ``guide_sequence``
  stays None: Mormino releases no per-strain spacer, the guides live upstream in the
  Smith 2017 library. PLUS the biosensor cassette every screened strain carries, as two
  ``GeneAdditionPerturbation``s integrated at the HO locus from pMM4_14L: the synthetic-TF
  construct BM3R1-HAA1-mTurquoise2 and the sfpHluorin expression cassette. They are a
  CONSTANT background (present in every strain and in the comparator), so the verifier
  entry lists them in ``background_genes`` and they leave the strain identity, the L1
  uniqueness key and the L4 gene set unchanged.
- ENVIRONMENT: the shared ``media.SC`` object (synthetic complete, whose sourced 0.77 g/L
  CSM / 6.9 g/L YNB w/o AA / 20 g/L glucose come from THIS paper) carrying three typed
  edits: acetic acid at 50 mM, pH 3.5 as an ``EnvironmentPhysicalPerturbation``, and the
  CRISPRi inducer anhydrotetracycline at 2 ug/mL with its DMSO vehicle as a typed
  ``Solvent``. 30 C, aerobic.
- PHENOTYPE: ``measurement_type=categorical``, ``assay_type=biosensor_readout``. Table 1's
  call is relative to the CBL pool: ``+`` (reporter expression 30% higher) ->
  ``ResponseCategory.enhanced``, ``=`` (similar) -> ``ResponseCategory.no_change``;
  ``category_label`` keeps the source's own symbol. ``n_samples=2``,
  ``sample_unit=biological_replicate``. Reference = the CBL pool (``no_change``).

CORRECTIONS THIS BUILD MAKES over the previous one, each sourced:

- The medium was stored as ``Media(name="SD, pH 3.5")``. The paper says SC, and SD
  (minimal) and SC (complete) are different media: "yeast cells were cultivated in
  synthetic complete medium (SC)". pH also left the free-text name and became a typed
  ``PhysicalFactor.ph`` edit, so it can be compared with any other dataset's pH.
- ATc (2 ug/mL) and its DMSO vehicle were absent although both are released. Without ATc
  the CRISPRi genotype is not repressed, so a record omitting it describes a different
  strain state.
- The comparator was named as the CC23 control strain. Table 1's own footnote says the
  comparator is the CBL: "*Reporter expression 30% higher (+) or similar (=) compared to
  the CBL". CC23 is the comparator for the separate 150 mM / sfpHluorin experiments.
- ``n_samples`` was None although "Screening of the pooled and single cell cultures sorted
  by FACS was performed in two biological replicates". The 3/5/6/7 replicate counts the
  previous docstring cited belong to different experiments.
- ``assay_type`` was None where ``biosensor_readout`` is exactly this assay, and the
  temperature was called unsourced although "Plates with 200 uL-cultures were cultivated
  at 30 C and 85% humidity, shaking at 995 rpm".

SCOPE (deliberate, not silent): only Table 1's 12 strains carry a machine-readable
per-strain call. The genome-wide enrichment is figure-only (Figs 2-6, bar charts and
growth curves, no released data matrix), so no numeric FI / sfpHluorin / growth value is
ingested; digitizing bars would violate provenance. Table 1's ``Growth`` column
(``ns`` / ``-`` / ``ND``) is a second readout against a DIFFERENT comparator (the CC23
control strain) and its ``E or RE`` essentiality column is an annotation, not a
measurement; neither is stored.

DATA SOURCE: Table 1 is embedded in the sha256-pinned OCR ``paper.md`` as a machine-
readable HTML table, and every one of the 12 rows is CHECKED against it at build time, so
the stored literal is auditable rather than merely transcribed. Both ``paper.pdf`` (BMC
counter URL, re-retrieved and sha256-verified 2026-09-12) and ``paper.md`` live in the raw
mirror ``$DATA_ROOT/torchcell-raw/morminoIdentificationAceticAcid2022/``; the manifest is
marked ``provenance_complete=False`` because the MinerU version and DPI of the 2026-07-12
OCR run were not recorded at the time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import os.path as osp
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import SC
from torchcell.datamodels.schema import (
    AssayType,
    Concentration,
    ConcentrationUnit,
    CrisprConstruct,
    CrisprInterferencePerturbation,
    Environment,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    GeneAdditionPerturbation,
    Genotype,
    MeasurementType,
    PhysicalFactor,
    Publication,
    ReferenceGenome,
    ResponseCategory,
    SampleUnit,
    SmallMoleculePerturbation,
    Solvent,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.scerevisiae.smith2006 import canonical_common_names
from torchcell.literature.manifest import (
    ROLE_PAPER_OCR,
    ROLE_PAPER_PDF,
    ArtifactRecord,
    Manifest,
    ProcessingRecord,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DOI = "10.1186/s12934-022-01938-7"
PMID = "36284296"

CITATION_KEY = "morminoIdentificationAceticAcid2022"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

PDF_FILENAME = "paper.pdf"
PDF_REL = "paper.pdf"
PDF_SHA256 = "388f8e922b0b94fba3a41965035eeee0f6180a110869073a426df96b5e63746a"
PDF_URL = (
    "https://microbialcellfactories.biomedcentral.com/counter/pdf/"
    "10.1186/s12934-022-01938-7"
)
PAPER_MD = "paper.md"
PAPER_MD_REL = "paper.md"
PAPER_MD_SHA256 = "f5d38e486148527bfba9dc9e40a9eb06ba051766aeca3e8ff67663551bf043c3"
RETRIEVED_AT = "2026-09-12"


def _paper(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
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
            page="Methods, 'Cultivation and screening conditions'; Table 1",
        ),
    )


_SC_QUOTE = (
    "yeast cells were cultivated in synthetic complete medium (SC) $( 0 . 7 7 \\textrm { "
    "g L } ^ { - 1 }$ complete supplement mix drop out (CSM), $6 . 9 \\ \\mathrm { g \\ "
    "L ^ { - 1 } }$ yeast nitrogen base without amino acids $( \\mathrm { Y N B ~ w / o ~ "
    "}$ AA), $2 0 \\ \\mathrm { g \\ L ^ { - 1 } }$ glucose, $\\mathrm { p H } ~ 5 . 5$ "
    ", 4.5 or 3.5)"
)
_SCREEN_QUOTE = (
    "Screening of the pooled and single cell cultures sorted by FACS was performed in two "
    "biological replicates, in SC medium at 0 and $5 0 ~ \\mathrm { m M }$ acetic acid at "
    "$\\mathrm { p H }$ 3.5."
)

MEDIUM = _paper(
    "SC",
    _SC_QUOTE,
    note="served as the shared media.SC object, whose 0.77 g/L CSM / 6.9 g/L YNB w/o AA / "
    "20 g/L glucose are sourced from THIS sentence; the previous build stored the wrong "
    "base, 'SD, pH 3.5'",
)
ACETIC_ACID_MM = _paper(50.0, _SCREEN_QUOTE)
MEDIUM_PH = _paper(
    3.5,
    _SCREEN_QUOTE,
    note="Media has no ph field (it sits in 36 served dataset closures, so adding one is "
    "a full rebuild), so the pH rides as a typed PhysicalFactor edit rather than inside a "
    "free-text medium name",
)
ACETIC_ACID_TITRATION = _paper(
    "NaOH-titrated to the medium pH",
    "added acetic acid was diluted into distilled water at a concentration of $5 0 0 ~ "
    "\\mathrm { m M }$ and the $\\mathsf { p H }$ of the solution was adjusted to the "
    "$\\mathrm { p H }$ of the medium through NaOH titration",
    note="the acid stock was pH-matched to the medium before addition, so the 50 mM dose "
    "and the pH 3.5 factor are independent edits rather than one acidification",
)
ATC_UG_PER_ML = _paper(
    2.0,
    "containing $2 ~ \\mu \\ g \\mathrm { ~ m L } ^ { - 1 }$ ATc (SC-ATc)",
    note="anhydrotetracycline is the CRISPRi inducer: without it the dCas9-Mxi1 genotype "
    "is not repressed, so a record omitting it describes a different strain state. "
    "'Acetic acid, as well as ATc, were added to the media at the beginning of the "
    "cultivation.'",
)
ATC_VEHICLE = _paper(
    "DMSO",
    "ATc stock solution was prepared by dissolving ATc into DMSO to a concentration of",
    note="the vehicle is released for THIS screen (125 ug/mL stock), so the Solvent is "
    "asserted; the final DMSO fraction in the medium is not stated and stays None",
)
TEMPERATURE_C = _paper(
    30.0,
    "-cultures were cultivated at $3 0 ~ ^ { \\circ } \\mathrm { C }$ and $8 5 \\%$ "
    "humidity, shaking at 995 rpm",
)
AEROBICITY = _paper(
    "aerobic",
    "-cultures were cultivated at $3 0 ~ ^ { \\circ } \\mathrm { C }$ and $8 5 \\%$ "
    "humidity, shaking at 995 rpm",
    note="shaken microbioreactor plates; no anaerobic handling is described",
)
N_REPLICATES = _paper(2, _SCREEN_QUOTE)
COMPARATOR = _paper(
    "CBL",
    "\\*Reporter expression $3 0 \\%$ higher $( + )$ or similar $\\left( = \\right)$ "
    "compared to the CBL",
    note="Table 1's own footnote. The comparator is the CRISPRi Biosensor Library pool, "
    "NOT the CC23 control strain (CC23 is the comparator for the Growth column and for "
    "the separate 150 mM / sfpHluorin experiments), and the threshold is 30%",
)
CBL_DEFINITION = _paper(
    "CRISPRi Biosensor Library (CBL)",
    "the biosensor library was named the CRISPRi Biosensor Library, for short CBL",
)
BIOSENSOR_CASSETTE = _paper(
    "pMM4_14L integrated at HO",
    "was integrated into the HO locus of the CRISPRi library strains",
    note="the sTF construct BM3R1-HAA1-mTurquoise2 under the RET2 promoter together with "
    "the sfpHluorin expression cassette (pMM4_14L); present in every library strain AND "
    "in the comparator, so it is a constant background",
)
LIBRARY = _paper(
    "Smith 2017 CRISPRi library, derived from BY4742",
    "Te CRISPRi strain library (derived from BY4742) screened originally contained 9,078 "
    "strains",
    note="1-16 gRNAs per gene is a property of the LIBRARY; an isolated strain carries "
    "exactly one gRNA, so n_guides=1",
)

EFFECTOR = "dCas9-Mxi1"

UNITS = (
    "Haa1 acetic-acid biosensor reporter (mCherry RFP) call from Table 1, relative to the "
    "CRISPRi Biosensor Library (CBL) pool: '+' = reporter expression 30% higher than the "
    "CBL (a higher biosensor signal marks cells more sensitive to acetic acid), "
    "'=' = similar to the CBL"
)

#: Table 1 symbol -> (typed call, the source's own symbol).
RFP_CATEGORY: dict[str, tuple[ResponseCategory, str]] = {
    "+": (ResponseCategory.enhanced, "+"),
    "=": (ResponseCategory.no_change, "="),
}
#: The CBL pool is the comparator, so its own call on this axis is "similar".
REFERENCE_CATEGORY = ResponseCategory.no_change
REFERENCE_CATEGORY_LABEL = "="

#: Table 1 "Properties of isolated strains": (strain id, RFP call, Growth call, target).
#: The Growth column is kept ONLY so each row can be checked verbatim against the OCR
#: table; it is a second readout against a different comparator and is not stored.
TABLE_1: list[tuple[str, str, str, str]] = [
    ("#3", "+", "ns", "QCR8"),
    ("#8", "+", "−", "TIF34"),
    ("#13", "+", "", "MSN5"),
    ("#15", "=", "ND", "NDC1"),
    ("#17", "+", "", "PAP1"),
    ("#32", "=", "", "CBP2"),
    ("#33", "+", "", "COX10"),
    ("#35", "+", "", "TRA1"),
    ("#37", "=", "ns", "UBA2"),
    ("#43", "=", "ND", "RPS30B"),
    ("#46", "=", "ND", "HSH49"),
    ("#49", "=", "ND", "LCB1"),
]


def table_1_row_fragment(row: tuple[str, str, str, str]) -> str:
    """The verbatim OCR-table cell run for one Table 1 row (the build-time audit key)."""
    strain, rfp, growth, gene = row
    return f"<td>{strain}</td><td>{rfp}</td><td>{growth}</td><td>{gene}</td>"


def audit_table_1(paper_md: str) -> None:
    """Assert every stored Table 1 row appears VERBATIM in the OCR'd article table.

    The values are a literal in this module, so the build re-derives nothing; this check
    is what makes the literal auditable instead of merely transcribed. A row that no
    longer matches means the OCR changed under us and raises rather than being ingested.
    """
    missing = [
        table_1_row_fragment(row)
        for row in TABLE_1
        if table_1_row_fragment(row) not in paper_md
    ]
    if missing:
        raise RuntimeError(
            f"{len(missing)} Table 1 rows are not present verbatim in paper.md: {missing}"
        )


def biosensor_cassette() -> list[GeneAdditionPerturbation]:
    """The pMM4_14L biosensor cassette every CBL strain carries, integrated at HO."""
    return [
        GeneAdditionPerturbation(
            systematic_gene_name="BM3R1-HAA1-mTurquoise2",
            perturbed_gene_name="BM3R1-HAA1-mTurquoise2",
            source_organism=(
                "synthetic fusion (BM3R1 from Bacillus megaterium, HAA1 from "
                "Saccharomyces cerevisiae, mTurquoise2 a synthetic fluorescent protein)"
            ),
            is_heterologous=True,
            localization="chromosomal_integration",
            construct_name="pMM4_14L",
            integration_locus="HO",
        ),
        GeneAdditionPerturbation(
            systematic_gene_name="sfpHluorin",
            perturbed_gene_name="sfpHluorin",
            source_organism="synthetic (superfolder pHluorin, an Aequorea victoria GFP variant)",
            is_heterologous=True,
            localization="chromosomal_integration",
            construct_name="pMM4_14L",
            integration_locus="HO",
        ),
    ]


#: The systematic-name slots of the constant biosensor cassette, for the verifier entry.
BACKGROUND_GENES = frozenset({"BM3R1-HAA1-mTurquoise2", "sfpHluorin"})


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror + build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/morminoIdentificationAceticAcid2022``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deposit_raw_mirror(
    *,
    pdf_path: str | Path,
    paper_md_path: str | Path,
    retrieved_at: str = RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror (article PDF + its OCR) plus its ``manifest.json``.

    Idempotent by sha256. The BMC counter URL is directly scriptable and reproduced the
    pinned PDF hash exactly on 2026-09-12. ``paper.md`` is the DERIVED artifact the build
    audits Table 1 against; the MinerU version and DPI of the 2026-07-12 OCR run were not
    recorded at the time, so the processing record says ``unrecorded`` and the manifest is
    marked ``provenance_complete=False`` rather than carrying a fabricated version.
    """
    root = raw_mirror_dir(data_root)
    root.mkdir(parents=True, exist_ok=True)
    files: list[ArtifactRecord] = []
    for source, relpath, expected in (
        (pdf_path, PDF_REL, PDF_SHA256),
        (paper_md_path, PAPER_MD_REL, PAPER_MD_SHA256),
    ):
        got = _sha256(source)
        if got != expected:
            raise RuntimeError(
                f"{source} sha256 mismatch: got {got}, expected {expected}"
            )
        dest = root / relpath
        if dest.exists():
            if _sha256(dest) != expected:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(source, dest)
    pdf_dest = root / PDF_REL
    md_dest = root / PAPER_MD_REL
    files.append(
        ArtifactRecord(
            path=PDF_REL,
            role=ROLE_PAPER_PDF,
            bytes=pdf_dest.stat().st_size,
            sha256=PDF_SHA256,
            source=PDF_URL,
            retrieval=RetrievalRecord(
                method=RetrievalMethod.direct_url,
                source_url=PDF_URL,
                retriever="torchcell.literature.retrieve.direct_url",
                params={"url": PDF_URL},
                sha256=PDF_SHA256,
                retrieved_at=retrieved_at,
            ),
        )
    )
    files.append(
        ArtifactRecord(
            path=PAPER_MD_REL,
            role=ROLE_PAPER_OCR,
            bytes=md_dest.stat().st_size,
            sha256=PAPER_MD_SHA256,
            source="mineru-ocr",
            processing=ProcessingRecord(
                processor="torchcell.literature.ocr.run_mineru",
                tool="mineru",
                version="unrecorded",
                params={
                    "note": (
                        "the OCR was run 2026-07-12, before the tool version and DPI were "
                        "recorded per artifact; the artifact is sha256-pinned and the "
                        "runner is versioned source, but the exact version is unknown and "
                        "is NOT fabricated here"
                    )
                },
                input_sha256=[PDF_SHA256],
            ),
        )
    )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=(
            "Identification of acetic acid sensitive strains through biosensor-based "
            "screening of a Saccharomyces cerevisiae CRISPRi library"
        ),
        library_id="6582362",
        zotero_item_key="XKUDX4UN",
        files=files,
        si_data_sources=[PDF_URL],
        si_expected=[
            "none -- the article Table 1 IS the data; no supplementary data file was "
            "released for this paper"
        ],
        provenance_complete=False,
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


class DropLog(BaseModel):
    """The build's retention accounting (no rule drops a Mormino record)."""

    dataset: str
    source_records: int
    kept_records: int
    dropped_records: int
    rules: list[dict[str, Any]]


@register_dataset
class CrispriMormino2022Dataset(ExperimentDataset):
    """Mormino 2022 CRISPRi acetic-acid biosensor screen (12 isolated strains)."""

    def __init__(
        self,
        root: str = "data/torchcell/crispri_mormino2022",
        io_workers: int = 0,
        genome: SCerevisiaeGenome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; a genome is REQUIRED for common-name -> current-R64-ORF resolution."""
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
        """The article PDF and the OCR the Table 1 literal is audited against."""
        return [PDF_FILENAME, PAPER_MD]

    def download(self) -> None:
        """Link the manifest-listed mirror files into ``raw/`` and verify their sha256."""
        data_root = _data_root()
        manifest = load_manifest(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        for relpath in (PDF_REL, PAPER_MD_REL):
            expected = manifest_sha256(manifest, relpath)
            src = raw_mirror_dir(data_root) / relpath
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            got = _sha256(src)
            if got != expected:
                raise RuntimeError(
                    f"{relpath} sha256 mismatch: got {got}, expected {expected}"
                )
            dest = osp.join(self.raw_dir, osp.basename(relpath))
            if not osp.exists(dest):
                os.symlink(src, dest)
        log.info(
            "Mormino 2022 artifacts linked into %s (sha256 verified)", self.raw_dir
        )

    def _resolve(self, common: str) -> str:
        """Common gene name -> current-R64 ORF via the SHARED genome resolver."""
        if self.genome is None:
            raise RuntimeError(
                "CrispriMormino2022Dataset requires a genome; inject SCerevisiaeGenome(...)"
            )
        resolution = self.genome.resolve_gene_name(common)
        if not resolution.is_current_gene or resolution.systematic_name is None:
            raise RuntimeError(
                f"Mormino2022: gene {common!r} did not resolve to a current R64 gene "
                f"({resolution.status})"
            )
        return resolution.systematic_name

    def _environment(self) -> Environment:
        """SC at pH 3.5 with 50 mM acetic acid and 2 ug/mL ATc, 30 C, aerobic."""
        return Environment(
            media=SC,
            temperature=Temperature(value=TEMPERATURE_C.value),
            perturbations=[
                SmallMoleculePerturbation(
                    compound=resolved_compound("acetic acid"),
                    concentration=Concentration(
                        value=ACETIC_ACID_MM.value, unit=ConcentrationUnit.millimolar
                    ),
                ),
                SmallMoleculePerturbation(
                    compound=resolved_compound("anhydrotetracycline"),
                    concentration=Concentration(
                        value=ATC_UG_PER_ML.value, unit=ConcentrationUnit.ug_per_ml
                    ),
                    solvent=Solvent(
                        name=ATC_VEHICLE.value, compound=resolved_compound("DMSO")
                    ),
                ),
                EnvironmentPhysicalPerturbation(
                    factor=PhysicalFactor.ph,
                    magnitude=Concentration(
                        value=MEDIUM_PH.value, unit=ConcentrationUnit.ph
                    ),
                ),
            ],
            aerobicity=AEROBICITY.value,
        )

    def _reference(
        self, environment: Environment
    ) -> EnvironmentResponseExperimentReference:
        """The CBL pool: the comparator every Table 1 call is made against."""
        return EnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain="BY4742"
            ),
            environment_reference=environment.model_copy(),
            phenotype_reference=EnvironmentResponsePhenotype(
                measurement_type=MeasurementType.categorical,
                assay_type=AssayType.biosensor_readout,
                category=REFERENCE_CATEGORY,
                category_label=REFERENCE_CATEGORY_LABEL,
                n_samples=N_REPLICATES.value,
                sample_unit=SampleUnit.biological_replicate,
                units=UNITS,
            ),
        )

    @post_process
    def process(self) -> None:
        """Build one categorical env x geno record per Table 1 isolated strain; write LMDB."""
        paper_md = Path(osp.join(self.raw_dir, PAPER_MD)).read_text(encoding="utf-8")
        audit_table_1(paper_md)
        assert self.genome is not None
        canonical = canonical_common_names(self.genome)
        publication = Publication(
            pubmed_id=PMID,
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PMID}/",
            doi=DOI,
            doi_url=f"https://doi.org/{DOI}",
        )
        environment = self._environment()
        reference = self._reference(environment)

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for _strain, rfp, _growth, common in tqdm(TABLE_1, desc="mormino2022"):
                orf = self._resolve(common)
                category, label = RFP_CATEGORY[rfp]
                experiment = EnvironmentResponseExperiment(
                    dataset_name=self.name,
                    genotype=Genotype(
                        perturbations=[
                            CrisprInterferencePerturbation(
                                systematic_gene_name=orf,
                                perturbed_gene_name=canonical.get(orf, orf),
                                crispr=CrisprConstruct(
                                    effector=EFFECTOR, guide_sequence=None, n_guides=1
                                ),
                            ),
                            *biosensor_cassette(),
                        ]
                    ),
                    environment=environment,
                    phenotype=EnvironmentResponsePhenotype(
                        measurement_type=MeasurementType.categorical,
                        assay_type=AssayType.biosensor_readout,
                        category=category,
                        category_label=label,
                        n_samples=N_REPLICATES.value,
                        sample_unit=SampleUnit.biological_replicate,
                        units=UNITS,
                    ),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, publication, itxn),
                )
                idx += 1
        env.close()
        interned_env.close()

        drop_log = DropLog(
            dataset=self.name,
            source_records=len(TABLE_1),
            kept_records=idx,
            dropped_records=len(TABLE_1) - idx,
            rules=[
                {
                    "rule": "none",
                    "scope": "row",
                    "description": (
                        "no retention rule removes a Mormino record: acetic acid, "
                        "anhydrotetracycline and DMSO all resolve to a structure "
                        "identifier, all 12 targets resolve to current R64 genes, and "
                        "every Table 1 row is verified verbatim against the OCR table"
                    ),
                    "n_records": 0,
                    "items": [],
                }
            ],
        )
        with open(osp.join(self.preprocess_dir, "dropped_records.json"), "w") as handle:
            handle.write(drop_log.model_dump_json(indent=2))
        log.info("Wrote %d Mormino2022 CRISPRi env-response records to LMDB", idx)

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
    root = osp.join(data_root, "data/torchcell/crispri_mormino2022")
    dataset = CrispriMormino2022Dataset(root=root, genome=genome)
    print(f"len = {len(dataset)}")
    print(json.dumps(dataset[0]["experiment"], indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
