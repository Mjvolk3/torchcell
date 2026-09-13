# torchcell/datasets/scerevisiae/lian2019
# [[torchcell.datasets.scerevisiae.lian2019]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/lian2019
# Test file: tests/torchcell/datasets/scerevisiae/test_lian2019.py
"""Lian 2019 MAGIC genome-wide CRISPRa/i/d furfural-tolerance screen (per-guide enrichment).

Lian et al. 2019 (Nat Commun 10:5794, doi:10.1038/s41467-019-13621-4; PMID 31857575) built
MAGIC: three genome-scale gRNA libraries driven by three ORTHOGONAL Cas effectors in one
pooled CRISPR-AID host, so a single cell carries exactly one guide of exactly one MODE.

- CRISPRa (activation)   -- ``dLbCas12a-VP``   (37,817 guides, 23 nt spacer)
- CRISPRi (interference) -- ``dSpCas9-RD1152`` (37,870 guides, 20 nt spacer)
- CRISPRd (deletion)     -- ``SaCas9``         (24,806 designs, 21 nt spacer + 100 nt donor)

The unique guide is a genetic barcode; furfural tolerance is mapped by tracking each
guide's enrichment (furfural-selected vs untreated) by NGS. Screening is ITERATIVE in
accumulating integrated backgrounds: round 1 = bAID host (5 mM furfural); round 2 = +SIZ1i
(10 mM); round 3 = +SIZ1i +NAT1a (15 mM). A round-2/3 record is therefore a genuine
2-/3-perturbation mixed-modality combo, the strain actually in the tube.

RECORD = one (guide x round) ``EnvironmentResponseExperiment``:

- GENOTYPE: the library member as a ``CrisprActivation`` / ``CrisprInterference`` /
  ``CrisprDeletion`` perturbation (target gene, this guide's spacer, its effector), plus
  the round's integrated background perturbations. The common name is the GENOME's own
  standard name for the resolved ORF, so one gene carries one spelling across datasets.
- ENVIRONMENT: the shared ``media.SED_URA_G418`` object carrying furfural (5/10/15 mM by
  round) as a ``SmallMoleculePerturbation``, 30 C, aerobic.
- PHENOTYPE: ``measurement_type=log2_ratio``,
  ``assay_type=pooled_competitive_growth_barcode``, ``environment_response`` = mean
  log2(after/before) over 3 biological triplicates, uncertainty = SD (``sample_sd``,
  n=3 -> SE = SD/sqrt(3)). Reference = no-enrichment baseline (log2FC 0) in the bAID host.

THE MEDIUM IS SED-URA/G418, NOT SED/G418 (corrected this build, all records). Methods,
verbatim: "The iMAGIC libraries in triplicates were inoculated into 50 mL SED-URA/G418
medium with or without furfural in a 250 mL baffled flask." The same paragraph shows
SED/G418 is the medium for a DIFFERENT experiment ("SED-URA/G418 (plasmid-bearing strains)
or SED/G418 (integrated strains)"), and the enrichment data come from the plasmid-borne
pooled library. This is not a naming nit: SED/G418 lacks the uracil dropout that selects
the guide plasmid, so the previous build recorded a different selection regime from the
one that produced the data. ``SED_URA_G418``'s components (YNB w/o AA 0.17%, monosodium
L-glutamate 0.1%, CSM-URA 0.077%, glucose 2%, G418 200 ug/mL, dropout uracil) are all
sourced from the Methods sentence quoted in ``media.py``.

THE CRISPRd GUIDE/DONOR SPLIT (corrected this build, 62,793 records). The previous build
stored the 44 nt amplicon barcode -- the FIRST 44 nt of the 121 nt design cassette -- in
``crispr.guide_sequence``, where the schema documents a "~20 nt" spacer, and left
``donor_sequence`` None. The paper states the layout: "the homologous recombination donor
was integrated to the 5'-end of the targeting sequences" and "Homology-directed repair
resulted in the deletion of 28 bp nucleotides in the coding sequences, including both the
targeting sequences and the protospacer adjacent motif sequences". The boundary is
MEASURED, not assumed, against the S288C genome:

- for 193 of 200 sampled designs whose two 50 nt halves both map uniquely, the gap between
  the first arm's end and the second arm's start is exactly 28 bp, and the last 21 nt of
  the design starts exactly at that gap (offset 0);
- for ALL 24,706 non-control designs, the LAST 21 nt is present in the genome with a
  canonical SaCas9 ``NNGRRT`` PAM immediately 3' of it (0 exceptions).

So the design cassette is ``donor (100 nt: two 50 nt homology arms flanking the 28 bp
deletion) + guide spacer (21 nt)``, and this build stores the last 21 nt as
``crispr.guide_sequence`` and everything before it as ``CrisprDeletionPerturbation.
donor_sequence``. The split is read from the PUBLISHED Supplementary Data 3
(``41467_2019_13621_MOESM5_ESM.xlsx``), not from a lab copy: its ``Sequence`` column is
element-for-element identical to the archived design file, and its first 44 nt reproduce
the enrichment table's ``d`` barcode for all 24,806 rows, which is the positional join
this loader asserts at build time.

RECORDS DROPPED (rule + counts in ``preprocess/dropped_records.json``):

- 300 random negative-control guides (100 per library) and 16 guides whose gene name is a
  source-corrupted Excel date artifact present IDENTICALLY in the reference and the design
  library, so it is unrecoverable from our inputs.
- guides whose target gene does not resolve to a current R64 gene (ncRNA/rDNA features
  absent from the ORF genome).
- (guide, round) cells with no enrichment value, and the guide-round in which a guide
  targets its OWN integrated background gene (redundant, and it would collapse to an empty
  strain signature once the background is subtracted).
Nothing else is dropped. In particular, 150 groups of CRISPRd designs (318 records) share
a resolved gene AND a 21 nt spacer, because the spacer maps to more than one site at a
multicopy locus (tRNA genes, paralogs) and the designs differ only in their donor arms.
They are distinct strains and are all KEPT: the verifier's genotype signature reads
``donor_sequence``, which is what tells them apart.

THE HOST STAYS ``ReferenceGenome(strain="bAID")``, a documented decision. bAID is BY4742
plus an integrated pAID6 carrying the three Cas effector cassettes ("The CRISPR-AID strain
(bAID) was constructed by integrating PmeI-digested pAID6 into the genome of BY4742 and
selection for G418 resistance."). The integration LOCUS is not stated anywhere in the
release, so a gene-keyed ``GeneAdditionPerturbation`` for the cassettes cannot be sourced,
and the three effectors are already carried per record on ``CrisprConstruct.effector``.
The cost is recorded rather than hidden: the strain string ``bAID`` does not join the
BY4742 datasets.

Exposure duration is a typed ``ProvenanceGap`` on both ``duration_hours`` and
``duration_generations``: the pooled cultures were harvested at mid-log ("1 OD of the
mid-log phase growing cells from each of the untreated and stressed libraries were
collected"), and neither a wall time nor a doubling count is reported.

DESIGNED VS REALIZED: every CRISPR perturbation here is a pooled-library DESIGN asserted
as realized. That is a documented deferral (memory
``designed-vs-realized-perturbation-material-entity``), not something this build measured,
and it is unmarked in the record because no certainty axis exists yet.

DATA SOURCE. The furfural per-guide enrichment is NOT in any Nature supplement (only the
design libraries, Supplementary Data 1-3, and the guide reference, Supplementary Data 4,
were released; the Fig 2 profiles are excluded from Source Data). It is reprocessed from
raw reads (NCBI SRA PRJNA504483, 21 runs) against the 100,493-guide reference by the
versioned pipeline in ``experiments/016-lian-magic-reprocess/``, and validated against the
paper's hits (PDR1i round-3 rank 1, SLX5i round-1 rank 1, SAP30d round-1 rank 2). The
derived ``guide_enrichment_final.tsv`` and the published Supplementary Data 3 both live in
the raw mirror ``$DATA_ROOT/torchcell-raw/lianMultifunctionalGenomewideCRISPR2019/`` with a
``manifest.json`` recording the Springer ESM retrieval for the design file and the
reprocessing chain (inputs + pipeline) for the derived table.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import os.path as osp
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import SED_URA_G418
from torchcell.datamodels.schema import (
    AssayType,
    Concentration,
    ConcentrationUnit,
    CrisprActivationPerturbation,
    CrisprConstruct,
    CrisprDeletionPerturbation,
    CrisprInterferencePerturbation,
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.scerevisiae.smith2006 import canonical_common_names
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ROLE_SI_DATA,
    ArtifactRecord,
    Manifest,
    ProcessingRecord,
    RetrievalMethod,
    RetrievalRecord,
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

DOI = "10.1038/s41467-019-13621-4"
PMID = "31857575"

CITATION_KEY = "lianMultifunctionalGenomewideCRISPR2019"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

TSV_FILENAME = "guide_enrichment_final.tsv"
TSV_REL = f"data/{TSV_FILENAME}"
TSV_SHA256 = "f9af849f97a2d460c3a6d628308491ec3966c6cc2a7f6cad130848d2bad32647"

_ESM_BASE = (
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-13621-4/"
    "MediaObjects/"
)
#: Supplementary Data 3 -- the CRISPRd design library (24,806 rows, 121 nt cassettes).
DESIGN_D_FILENAME = "41467_2019_13621_MOESM5_ESM.xlsx"
DESIGN_D_REL = f"si/si_data/{DESIGN_D_FILENAME}"
DESIGN_D_SHA256 = "737074a76b9eee2dc015be8b17e29b4fbe65c8be5565e6fcbe71505dca4109e2"
#: Supplementary Data 4 -- the 100,493-guide reference the reprocessing mapped against.
REFERENCE_FILENAME = "41467_2019_13621_MOESM6_ESM.xlsx"
REFERENCE_SHA256 = "4e3f225ae0194462252049aae28f57c1428a99000ffdeef9e418e240778435a3"
SI_RETRIEVED_AT = "2026-09-12"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "63fe2b7101fc48feb297f9e34b83d108b74f03f28bbc280e08c7219bc975086c"

#: The amplicon barcode the reprocessing counted for a CRISPRd guide: read[27:71].
D_BARCODE_LEN = 44
#: SaCas9 spacer length, measured against the genome (see the module docstring).
D_SPACER_LEN = 21


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
            page="Methods, 'iMAGIC screening of furfural tolerance' / 'Design and "
            "construction of the MAGIC libraries' / 'Strains and media'",
        ),
    )


_SCREEN_QUOTE = (
    "The iMAGIC libraries in triplicates were inoculated into $5 0 \\mathrm { m L }$ "
    "SED-URA/G418 medium with or without furfural in a $2 5 0 \\mathrm { m L }$ baffled "
    "flask."
)
_CULTIVATION_QUOTE = (
    "Yeast strains were cultivated in complex medium consisting of $2 \\%$ peptone, $1 "
    "\\%$ yeast extract, and $2 \\%$ glucose (YPD) or synthetic complete medium consisting "
    "of $0 . 1 7 \\%$ yeast nitrogen base, $0 . 1 \\%$ mono-sodium glutamate, $0 . 0 7 7 "
    "\\%$ CSM-URA, and $2 \\%$ glucose (SED-URA) at $3 0 ^ { \\circ } \\mathrm { C } ,$ . "
    "$2 5 0 \\mathrm { r p m }$ ."
)

MEDIUM = _paper(
    "SED-URA/G418",
    _SCREEN_QUOTE,
    note="served as the shared media.SED_URA_G418 object. The previous build stored "
    "'SED/G418', which is the medium for the INTEGRATED validation strains in the same "
    "paragraph, not for the plasmid-borne pooled library the enrichment comes from; "
    "SED/G418 lacks the uracil dropout that selects the guide plasmid",
)
TEMPERATURE_C = _paper(30.0, _CULTIVATION_QUOTE)
AEROBICITY = _paper(
    "aerobic",
    _SCREEN_QUOTE,
    note="a shaken baffled flask, the standard aerobic configuration; the validation "
    "cultures of the same screen are explicitly 'cultivated under aerobic conditions'",
)
FURFURAL_MM = _paper(
    {1: 5.0, 2: 10.0, 3: 15.0},
    "5, 10, and $1 5 \\mathrm { m M }$ furfural were used for the first, second, and "
    "third round of iMAGIC screening, respectively.",
)
N_REPLICATES = _paper(
    3,
    "biological triplicates for untreated and furfural stressed libraries",
    note="the triplicate untreated and furfural-stressed libraries; the stored SD is the "
    "sample SD across them, so SE = SD/sqrt(3)",
)
ASSAY = _paper(
    AssayType.pooled_competitive_growth_barcode,
    "The reads of $4 3 \\mathrm { b p }$ between SNR52p and SUP4t that contains a unique "
    "sequence in all three CRISPR-AID libraries (Supplementary Table 12) were extracted "
    "from the NGS data",
    note="a pooled library grown competitively and read out by amplifying each strain's "
    "unique guide sequence as a barcode",
)
HOST = _paper(
    "bAID",
    "The CRISPR-AID strain (bAID) was constructed by integrating PmeI-digested $\\mathrm "
    "{ \\ p A I D } 6 ^ { 8 }$ into the genome of BY4742 and selection for G418 resistance.",
    note="the integration LOCUS is not stated anywhere in the release, so the three Cas "
    "effector cassettes cannot be sourced as gene-keyed GeneAdditionPerturbations; they "
    "are carried per record on CrisprConstruct.effector instead, and the cost of keeping "
    "the engineered-host strain string (bAID does not join the BY4742 datasets) is "
    "recorded rather than hidden",
)
DONOR_LAYOUT = _paper(
    {"donor_nt": 100, "spacer_nt": D_SPACER_LEN, "deleted_bp": 28},
    "the homologous recombination donor was integrated to the $5 ^ { \\prime }$ -end of "
    "the targeting sequences9.",
    note="measured against S288C, not assumed: the two 50 nt donor arms flank a gap of "
    "exactly 28 bp in 193/193 resolvable sampled designs, the last 21 nt starts at that "
    "gap, and for all 24,706 non-control designs the last 21 nt sits in the genome with a "
    "canonical SaCas9 NNGRRT PAM immediately 3' (0 exceptions)",
)
DELETION_SIZE = _paper(
    28,
    "Homology-directed repair resulted in the deletion of 28 bp nucleotides in the coding "
    "sequences, including both the targeting sequences and the protospacer adjacent motif "
    "sequences",
)
HARVEST = _paper(
    "mid-log phase",
    "1 OD of the mid-log phase growing cells from each of the untreated and stressed "
    "libraries were collected and the plasmids were extracted for NGS analysis.",
    note="the cultures were harvested at mid-log, and neither a wall time nor a doubling "
    "count is reported, so both duration fields are typed ProvenanceGaps",
)

DURATION_GAPS = [
    ProvenanceGap(
        field="duration_hours",
        reason=ProvenanceGapReason.not_reported_by_primary,
        note="the pooled cultures were harvested at mid-log phase; no wall-clock exposure "
        "time is reported for the screening flasks",
    ),
    ProvenanceGap(
        field="duration_generations",
        reason=ProvenanceGapReason.not_reported_by_primary,
        note="no doubling count is reported for the screening flasks either",
    ),
]

UNITS = (
    "log2(furfural-selected / untreated guide-barcode abundance); positive = the "
    "perturbation confers furfural tolerance"
)

#: Orthogonal Cas effector per modality (Table 1).
EFFECTOR = {"a": "dLbCas12a-VP", "i": "dSpCas9-RD1152", "d": "SaCas9"}
PERT_CLASS: dict[str, Any] = {
    "a": CrisprActivationPerturbation,
    "i": CrisprInterferencePerturbation,
    "d": CrisprDeletionPerturbation,
}
#: Integrated backgrounds accumulated per round. R1 = bAID (none); R2 = +SIZ1i;
#: R3 = +SIZ1i +NAT1a. SIZ1 = YDR409W, NAT1 = YDL040C.
ROUND_BACKGROUND: dict[int, list[tuple[str, str, str]]] = {
    1: [],
    2: [("YDR409W", "SIZ1", "i")],
    3: [("YDR409W", "SIZ1", "i"), ("YDL040C", "NAT1", "a")],
}


def crispr_perturbation(
    orf: str, common: str, mod: str, guide: str | None, donor: str | None = None
) -> Any:
    """Build the modality-appropriate CRISPR perturbation for a target gene."""
    construct = CrisprConstruct(
        effector=EFFECTOR[mod],
        guide_sequence=guide,
        n_guides=1 if guide is not None else None,
    )
    cls = PERT_CLASS[mod]
    if mod == "d":
        return cls(
            systematic_gene_name=orf,
            perturbed_gene_name=common,
            crispr=construct,
            donor_sequence=donor,
        )
    return cls(systematic_gene_name=orf, perturbed_gene_name=common, crispr=construct)


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror + build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/lianMultifunctionalGenomewideCRISPR2019``."""
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
    enrichment_path: str | Path,
    design_d_path: str | Path,
    retrieved_at: str = SI_RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror (derived enrichment + published CRISPRd design) + manifest.

    Idempotent by sha256. The design file is a directly scriptable Springer ESM whose
    pinned hash was reproduced on 2026-09-12. The enrichment table is DERIVED, not
    retrieved: its ``ProcessingRecord`` names the versioned reprocessing pipeline and the
    sha256 of its inputs (the Supplementary Data 4 guide reference and the raw SRA
    project), so the rebuild chain is recorded rather than a fabricated download URL.
    """
    root = raw_mirror_dir(data_root)
    design_url = f"{_ESM_BASE}{DESIGN_D_FILENAME}"
    reference_url = f"{_ESM_BASE}{REFERENCE_FILENAME}"
    files: list[ArtifactRecord] = []
    for source, relpath, expected in (
        (enrichment_path, TSV_REL, TSV_SHA256),
        (design_d_path, DESIGN_D_REL, DESIGN_D_SHA256),
    ):
        got = _sha256(source)
        if got != expected:
            raise RuntimeError(
                f"{source} sha256 mismatch: got {got}, expected {expected}"
            )
        dest = root / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != expected:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(source, dest)
    files.append(
        ArtifactRecord(
            path=TSV_REL,
            role=ROLE_RAW_DATA,
            bytes=(root / TSV_REL).stat().st_size,
            sha256=TSV_SHA256,
            source="derived: SRA PRJNA504483 reprocessing",
            processing=ProcessingRecord(
                processor="experiments/016-lian-magic-reprocess/scripts/reproduce.sh",
                tool="torchcell lian-magic-reprocess pipeline",
                version="2026-07-13",
                params={
                    "sra_project": "PRJNA504483",
                    "n_runs": 21,
                    "barcode_window": "read[27:70] (43 bp activation) | read[27:71] "
                    "(44 bp interference/deletion), forward, exact match",
                    "normalization": "CPM(+1) per library; per round per replicate "
                    "log2(furfural-after / untreated-before); mean +- SD over triplicates",
                    "validation": "PDR1i round-3 rank 1, SLX5i round-1 rank 1, SAP30d "
                    "round-1 rank 2 against the paper's reported hits",
                    "reference_url": reference_url,
                },
                input_sha256=[REFERENCE_SHA256],
            ),
        )
    )
    files.append(
        ArtifactRecord(
            path=DESIGN_D_REL,
            role=ROLE_SI_DATA,
            bytes=(root / DESIGN_D_REL).stat().st_size,
            sha256=DESIGN_D_SHA256,
            source=design_url,
            retrieval=RetrievalRecord(
                method=RetrievalMethod.springer_esm,
                source_url=design_url,
                retriever="torchcell.literature.retrieve.springer_esm",
                params={"url": design_url},
                sha256=DESIGN_D_SHA256,
                retrieved_at=retrieved_at,
            ),
        )
    )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=(
            "Multi-functional genome-wide CRISPR system for high throughput "
            "genotype-phenotype mapping"
        ),
        files=files,
        si_data_sources=[
            design_url,
            reference_url,
            "https://www.ncbi.nlm.nih.gov/bioproject/PRJNA504483/",
        ],
        si_expected=[
            "Supplementary Data 3 (CRISPRd design library) -- consumed here for the "
            "guide/donor split; its Sequence column is element-for-element identical to "
            "the archived lab design file, and the same holds for Supplementary Data 1 "
            "and 2 against the CRISPRa and CRISPRi lab files",
            "Supplementary Data 4 (100,493-guide reference) -- an INPUT to the derived "
            "enrichment table, recorded in its processing record rather than mirrored "
            "here, since the loader does not read it",
            "the per-guide furfural enrichment itself was NEVER released; it is "
            "reprocessed from SRA PRJNA504483",
        ],
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


def split_deletion_cassette(sequence: str) -> tuple[str, str]:
    """``(guide spacer, HR donor)`` for one CRISPRd design cassette.

    The paper puts the donor 5' of the targeting sequence, and the boundary is measured
    against S288C (module docstring): the last :data:`D_SPACER_LEN` nt is the SaCas9
    spacer and everything before it is the donor.
    """
    text = sequence.strip().upper()
    if len(text) <= D_SPACER_LEN:
        raise ValueError(f"deletion cassette too short to split: {text!r}")
    return text[-D_SPACER_LEN:], text[:-D_SPACER_LEN]


# --------------------------------------------------------------------------- #
# Retention bookkeeping
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One retention rule, the records it removed, and the items it removed them for."""

    rule: str
    scope: Literal["guide", "guide_round", "strain"]
    description: str
    n_records: int
    items: list[str] = []


class DropLog(BaseModel):
    """Every retention rule applied to a build, in the order they were applied."""

    dataset: str
    source_records: int
    kept_records: int
    dropped_records: int
    rules: list[DropRule]


@register_dataset
class CrisprMagicLian2019Dataset(ExperimentDataset):
    """Lian 2019 MAGIC per-guide CRISPRa/i/d furfural-enrichment env x geno dataset."""

    def __init__(
        self,
        root: str = "data/torchcell/crispr_magic_lian2019",
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
        """The derived enrichment table and the published CRISPRd design library."""
        return [TSV_FILENAME, DESIGN_D_FILENAME]

    def download(self) -> None:
        """Link the manifest-listed mirror files into ``raw/`` and verify their sha256."""
        data_root = _data_root()
        manifest = load_manifest(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        for relpath, filename in (
            (TSV_REL, TSV_FILENAME),
            (DESIGN_D_REL, DESIGN_D_FILENAME),
        ):
            expected = manifest_sha256(manifest, relpath)
            src = raw_mirror_dir(data_root) / relpath
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            got = _sha256(src)
            if got != expected:
                raise RuntimeError(
                    f"{filename} sha256 mismatch: got {got}, expected {expected}"
                )
            dest = osp.join(self.raw_dir, filename)
            if not osp.exists(dest):
                os.symlink(src, dest)
        log.info("Lian 2019 artifacts linked into %s (sha256 verified)", self.raw_dir)

    def _resolver(self) -> Callable[[str], str | None]:
        """Common/standard gene name -> current-R64 ORF via the SHARED genome resolver."""
        if self.genome is None:
            raise RuntimeError(
                "CrisprMagicLian2019Dataset requires a genome; inject SCerevisiaeGenome(...)"
            )
        genome = self.genome
        gene_set = {gene.upper() for gene in genome.gene_set}
        cache: dict[str, str | None] = {}

        def resolve(name: str) -> str | None:
            key = str(name).strip()
            if key not in cache:
                resolution = genome.resolve_gene_name(key)
                cache[key] = (
                    resolution.systematic_name
                    if resolution.is_current_gene
                    and resolution.systematic_name in gene_set
                    else None
                )
            return cache[key]

        return resolve

    def _deletion_cassettes(self, table: pd.DataFrame) -> list[tuple[str, str]]:
        """Per CRISPRd row of the enrichment table, its ``(spacer, donor)`` split.

        The published design library and the enrichment table's ``d`` block are the same
        24,806 rows in the same order. That positional join is ASSERTED here, not assumed:
        every row's released barcode must equal the first 44 nt of its design cassette.
        """
        design = pd.read_excel(osp.join(self.raw_dir, DESIGN_D_FILENAME))
        sequences = design["Sequence"].astype(str).str.upper().tolist()
        barcodes = table.loc[table["mod"] == "d", "spacer"].astype(str).str.upper()
        if len(sequences) != len(barcodes):
            raise RuntimeError(
                f"CRISPRd design rows ({len(sequences)}) do not match the enrichment "
                f"table's d rows ({len(barcodes)})"
            )
        mismatched = [
            i
            for i, (barcode, sequence) in enumerate(
                zip(barcodes, sequences, strict=True)
            )
            if barcode != sequence[:D_BARCODE_LEN]
        ]
        if mismatched:
            raise RuntimeError(
                f"{len(mismatched)} CRISPRd rows do not positionally join the design "
                f"library (first offenders: {mismatched[:5]})"
            )
        return [split_deletion_cassette(sequence) for sequence in sequences]

    def _environment(self, furfural_mm: float) -> Environment:
        """SED-URA/G418 liquid carrying furfural (mM), 30 C, aerobic."""
        return Environment(
            media=SED_URA_G418,
            temperature=Temperature(value=TEMPERATURE_C.value),
            perturbations=[
                SmallMoleculePerturbation(
                    compound=resolved_compound("furfural"),
                    concentration=Concentration(
                        value=furfural_mm, unit=ConcentrationUnit.millimolar
                    ),
                )
            ],
            aerobicity=AEROBICITY.value,
            provenance_gaps=list(DURATION_GAPS),
        )

    def _reference(
        self, environment: Environment
    ) -> EnvironmentResponseExperimentReference:
        """No-enrichment baseline: a guide that neither enriches nor depletes -> log2FC 0."""
        return EnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain=HOST.value
            ),
            environment_reference=environment.model_copy(),
            phenotype_reference=EnvironmentResponsePhenotype(
                measurement_type=MeasurementType.log2_ratio,
                assay_type=ASSAY.value,
                environment_response=0.0,
                n_samples=N_REPLICATES.value,
                sample_unit=SampleUnit.biological_replicate,
                units=UNITS,
            ),
        )

    @post_process
    def process(self) -> None:
        """Build one env x geno -> log2-enrichment record per (guide, round); write LMDB."""
        table = pd.read_csv(osp.join(self.raw_dir, TSV_FILENAME), sep="\t")
        resolve = self._resolver()
        assert self.genome is not None
        canonical = canonical_common_names(self.genome)
        cassettes = self._deletion_cassettes(table)

        # Attach the CRISPRd (spacer, donor) split back onto the enrichment rows; every
        # other modality already releases its true spacer.
        spacers = table["spacer"].astype(str).str.upper().tolist()
        donors: list[str | None] = [None] * len(table)
        d_positions = [i for i, mod in enumerate(table["mod"]) if mod == "d"]
        for position, (spacer, donor) in zip(d_positions, cassettes, strict=True):
            spacers[position] = spacer
            donors[position] = donor
        table = table.assign(
            true_spacer=pd.Series(spacers, index=table.index),
            donor=pd.Series(donors, index=table.index, dtype=object),
        )

        background_orfs = {
            rnd: {orf for orf, _, _ in ROUND_BACKGROUND[rnd]} for rnd in (1, 2, 3)
        }
        rounds = (1, 2, 3)

        # --- pass 1: gene resolution -------------------------------------------- #
        # Two CRISPRd designs can share a gene AND a 21 nt spacer (multicopy loci where
        # the spacer maps to more than one site) and differ only in their donor arms. They
        # are distinct strains and are all KEPT: the verifier's genotype signature reads
        # ``donor_sequence``, so the donor is what tells them apart.
        resolved: list[str | None] = []
        for _, row in table.iterrows():
            if bool(row["is_control"]) or bool(row["corrupted_gene"]):
                resolved.append(None)
                continue
            resolved.append(resolve(row["gene"]))

        publication = Publication(
            pubmed_id=PMID,
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PMID}/",
            doi=DOI,
            doi_url=f"https://doi.org/{DOI}",
        )
        prepared: dict[int, dict[str, Any]] = {}
        for rnd in rounds:
            environment = self._environment(FURFURAL_MM.value[rnd])
            prepared[rnd] = {
                "environment": environment,
                "reference": self._reference(environment),
                "background": [
                    crispr_perturbation(orf, canonical.get(orf, common), mod, None)
                    for orf, common, mod in ROUND_BACKGROUND[rnd]
                ],
            }

        # --- pass 2: write ------------------------------------------------------- #
        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        n_ctrl = n_corrupt = n_unresolved = n_nan = n_bg_self = 0
        unresolved_genes: set[str] = set()
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for position, (_, row) in tqdm(
                enumerate(table.iterrows()), total=len(table), desc="lian2019"
            ):
                if bool(row["is_control"]):
                    n_ctrl += 1
                    continue
                if bool(row["corrupted_gene"]):
                    n_corrupt += 1
                    continue
                orf = resolved[position]
                if orf is None:
                    n_unresolved += 1
                    unresolved_genes.add(str(row["gene"]))
                    continue
                mod = str(row["mod"])
                common = canonical.get(orf, str(row["gene"]))
                spacer = row["true_spacer"]
                donor = row["donor"]
                for rnd in rounds:
                    mean = row[f"r{rnd}_log2fc_mean"]
                    if pd.isna(mean):
                        n_nan += 1
                        continue
                    if orf in background_orfs[rnd]:
                        n_bg_self += 1
                        continue
                    sd = row[f"r{rnd}_log2fc_sd"]
                    has_sd = not pd.isna(sd) and not math.isinf(float(sd))
                    item = prepared[rnd]
                    experiment = EnvironmentResponseExperiment(
                        dataset_name=self.name,
                        genotype=Genotype(
                            perturbations=[
                                crispr_perturbation(orf, common, mod, spacer, donor),
                                *item["background"],
                            ]
                        ),
                        environment=item["environment"],
                        phenotype=EnvironmentResponsePhenotype(
                            measurement_type=MeasurementType.log2_ratio,
                            assay_type=ASSAY.value,
                            environment_response=float(mean),
                            environment_response_uncertainty=(
                                float(sd) if has_sd else None
                            ),
                            environment_response_uncertainty_type=(
                                UncertaintyType.sample_sd if has_sd else None
                            ),
                            n_samples=N_REPLICATES.value,
                            sample_unit=SampleUnit.biological_replicate,
                            units=UNITS,
                        ),
                    )
                    txn.put(
                        f"{idx}".encode(),
                        self._intern_record(
                            experiment, item["reference"], publication, itxn
                        ),
                    )
                    idx += 1
        env.close()
        interned_env.close()

        source_records = len(table) * len(rounds)
        rules = [
            DropRule(
                rule="random_negative_control_guide",
                scope="guide",
                description=(
                    "one of the 300 random negative-control guides (100 per library); it "
                    "targets no gene, so there is no genotype to key a record to"
                ),
                n_records=n_ctrl * len(rounds),
                items=[],
            ),
            DropRule(
                rule="source_corrupted_gene_name",
                scope="guide",
                description=(
                    "the gene name is an Excel date/serial artifact present IDENTICALLY "
                    "in the released reference and the design library, so the target is "
                    "unrecoverable from our inputs"
                ),
                n_records=n_corrupt * len(rounds),
                items=[],
            ),
            DropRule(
                rule="target_gene_is_not_a_current_genome_gene",
                scope="guide",
                description=(
                    "the guide targets an ncRNA / rDNA feature or a name absent from the "
                    "current R64 ORF genome, so no gene entity exists to key the record to"
                ),
                n_records=n_unresolved * len(rounds),
                items=sorted(unresolved_genes)[:200],
            ),
            DropRule(
                rule="guide_round_has_no_enrichment_value",
                scope="guide_round",
                description=(
                    "the guide was not detected in this round's before/after libraries, "
                    "so the round has no log2 enrichment for it"
                ),
                n_records=n_nan,
                items=[],
            ),
            DropRule(
                rule="guide_targets_its_own_round_background",
                scope="guide_round",
                description=(
                    "in this round the guide's target gene is already an integrated "
                    "background perturbation, so the foreground edit is redundant and the "
                    "strain signature would collapse to empty once the background is "
                    "subtracted"
                ),
                n_records=n_bg_self,
                items=[],
            ),
        ]
        drop_log = DropLog(
            dataset=self.name,
            source_records=source_records,
            kept_records=idx,
            dropped_records=source_records - idx,
            rules=rules,
        )
        with open(osp.join(self.preprocess_dir, "dropped_records.json"), "w") as handle:
            handle.write(drop_log.model_dump_json(indent=2))
        accounted = sum(rule.n_records for rule in rules)
        if accounted != drop_log.dropped_records:
            raise RuntimeError(
                f"drop accounting mismatch: rules total {accounted}, "
                f"{drop_log.dropped_records} records missing from the build"
            )
        log.info(
            "Lian2019: wrote %d (guide x round) records; dropped %d control + %d "
            "corrupted + %d unresolved-gene guides (%d distinct genes); %d guide-rounds "
            "undetected; %d skipped (guide targets its own round background)",
            idx,
            n_ctrl,
            n_corrupt,
            n_unresolved,
            len(unresolved_genes),
            n_nan,
            n_bg_self,
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
    root = osp.join(data_root, "data/torchcell/crispr_magic_lian2019")
    dataset = CrisprMagicLian2019Dataset(root=root, genome=genome)
    print(f"len = {len(dataset)}")
    print(
        json.dumps(
            json.loads(
                Path(osp.join(root, "preprocess/dropped_records.json")).read_text()
            )["rules"],
            indent=2,
        )[:2500]
    )


if __name__ == "__main__":
    main()
