# torchcell/datasets/scerevisiae/mota2024
# [[torchcell.datasets.scerevisiae.mota2024]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/mota2024
# Test file: tests/torchcell/datasets/scerevisiae/test_mota2024.py
"""Mota 2024 acetic/butyric/octanoic acid chemogenomic screen (env x geno -> response).

Mota et al. 2024 (Microbial Cell Factories, doi:10.1186/s12934-024-02309-0; PMC10903034)
screened the BY4741 Euroscarf HAPLOID single-deletion collection for susceptibility to
three linear monocarboxylic weak acids at EQUIVALENT moderate inhibitory concentrations --
75 mM acetic acid (C2), 14 mM butyric acid (C4), and 0.30 mM octanoic acid (C8) -- on solid
YPD at pH 4.5, 30 C, scored by spot assay against the parental strain. This is a plain
haploid deletion collection (NOT a HIP/HOP heterozygous screen), so every screened strain
is a single-gene KanMX deletion in BY4741.

READOUT -- an ORDINAL susceptibility grade, and it is stored as one. The source scale is
``0`` < ``+`` < ``++`` (``_SCORING``, ``_FIGURE_S2``), an ORDER whose spacing means
nothing, which is exactly ``MeasurementType.ordinal``: the rank rides on
``environment_response`` (0.0 / 1.0 / 2.0), the shared call on ``category``
(``no_change`` / ``reduced`` / ``severely_reduced``) and the source's own symbol on
``category_label`` (``"0"`` / ``"+"`` / ``"++"``). The previous build set
``measurement_type=categorical`` with ``environment_response=None``, which threw the order
away and left L2 value fidelity checking nothing. The reference is the PARENTAL strain,
which the paper scores ``0`` = "an absence of a detectable susceptibility phenotype", so
the reference rank is 0.0 and L3 reference_zero holds numerically.

The comparator here is mutant-vs-PARENTAL (unlike Auesukaree's within-strain call): the
wild type was inoculated into empty wells of the SAME acid plates (``_SCREEN_DESIGN``), so
the reference environment carries the same acid, the same pH and the same medium.

ENVIRONMENT -- the medium is the SHARED ``media.YPD_AGAR`` (YPD plus the paper's own
20 g/L agar row, ``base_medium="YPD"``), and the acidification is a TYPED perturbation,
never part of the medium's name: ``EnvironmentPhysicalPerturbation(factor=ph,
magnitude=4.5 pH, agent=hydrochloric acid)``, with the acid that sets it sourced by
``_MEDIUM`` ("acidified with HCl until pH 4.5"). The previous build emitted a free-text
``Media(name="YPD, pH 4.5")`` with zero components, a node nothing joins. The weak acid
itself is a ``SmallMoleculePerturbation`` at its sourced molar concentration; both
perturbations ride on the treated AND the reference environment.

SOURCING -- every metadata number is a ``SourcedValue`` quoting the mirrored OCR
``paper.md``: ``_MEDIUM``, ``_TEMPERATURE``, ``_PH``, ``_CONCENTRATIONS``, ``_DURATION``,
``_SCORING``, ``_FIGURE_S2``, ``_BACKGROUND``, ``_ASSAY``. Two absences are TYPED, not
silent: the screen's ``n_samples`` and ``sample_unit`` carry
``ProvenanceGap(not_reported_by_primary)`` -- the paper's ">= 3 independent experiments"
statements attach to the CFU-viability and intracellular-pH assays, never to the disruptome
screen, which states no replicate count at all.

DURATION -- 48.0 h, and the rule is written down rather than left implicit. The paper gives
a RANGE for when the acid plates were photographed ("36-48 h"), but the SCORING definition
this dataset stores is anchored at a point: "(++) if no growth was observed after 48 h of
incubation". ``duration_hours`` is the time at which the stored call was made, so 48.0 is
the sourced value and the 24 h control-plate reading and the 36-48 h photograph range are
recorded in ``_DURATION``'s note.

GENE RESOLUTION -- every token, INCLUDING one that merely looks systematic, goes through
the SHARED ``SCerevisiaeGenome.resolve_gene_name``. The previous resolver returned any
``Y..####[WC]``-shaped token unchecked, which stored four RETIRED systematic names on seven
records (L4 gene containment read 0.993, below the rule's notice but above its 0.90 floor).
All four are RENAMED, not retired: YGR272C -> YGR271C-A, YJL021C -> YJL020C,
YML010W-A -> YML009W-B, YML013C-A -> YML012C-A. CURRENT and RENAMED are kept under the
CURRENT systematic name; RETIRED is dropped, counted and written to
``dropped_records.json`` beside ``processed/``.

SOURCE ARTIFACTS / QUIRKS handled deterministically (``_DEDUP_RULE``):
- RNR4 (YGR180C) is listed TWICE in every table (a source duplicate the paper's headline
  totals count); acetic/butyric both rows are ``+``, octanoic the two rows conflict
  (``+`` and ``++``).
- EFG1 and YGR272C are listed as SEPARATE rows in all three tables but are ONE gene, and
  the SI says so itself: its annotation of YGR272C reads "it was merged with an adjacent
  ORF into a single reading frame, designated YGR271C-A". Resolving both to YGR271C-A
  merges them.
  Records are deduplicated per (resolved ORF, acid) keeping the MORE SEVERE score; on a tie
  the lexicographically smallest source token wins. Where two DIFFERENT tokens claim one
  gene the genome's canonical common name is stored in ``perturbed_gene_name`` instead of
  either token, so one gene carries one spelling across the three acids.
- Six gene tokens are genuinely RETIRED in R64-4-1 and are DROPPED (never guessed): REF1,
  RLM2, SBR2 (all three acids), ILM2 (butyric), VPS236 (butyric, octanoic), SIW15
  (octanoic) -- 13 records. Resolving them is a flagged follow-up (likely SI typos:
  VPS236->VPS36?, SIW15->SIW14?, ILM2->ILM1?, RLM2->RLM1?).

Final: acetic 372, butyric 415, octanoic 483 = 1270 records (1289 raw susceptible rows
- 3 RNR4 duplicates - 3 EFG1/YGR272C merges - 13 retired-token drops).

DATA SOURCE -- the BMC open-access supplementary spreadsheets (Additional file 1 = acetic
Table S1, file 2 = butyric Table S2, file 3 = octanoic Table S3) are deposited into
``$DATA_ROOT/torchcell-raw/motaSharedMoreSpecific2024/`` and sha256-pinned. The mirror is
the source of record; ``static-content.springer.com`` is scriptable and re-yielded all
three files bit-identically on 2026-09-12, which is recorded as the manifest's
``last_check``, but a build reads the mirror and only falls to the network when the mirror
is absent.
"""

import hashlib
import logging
import os
import os.path as osp
import pickle
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lmdb
import openpyxl
from pydantic import BaseModel, Field
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import YPD_AGAR
from torchcell.datamodels.schema import (
    AssayType,
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
    PhysicalFactor,
    Publication,
    ReferenceGenome,
    ResponseCategory,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_SI_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
    SourceCheck,
    sha256_file,
)
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome
from torchcell.sequence.genome.scerevisiae.s288c import GeneNameStatus
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DOI = "10.1186/s12934-024-02309-0"
CITATION_KEY = "motaSharedMoreSpecific2024"

_PAPER_MD = "paper.md"
_PAPER_MD_SHA256 = "a19769f757fd912139551f39736dd2b67581cb03f83a7a9b9385e28516b1f1b6"

# BMC open-access supplementary spreadsheets (scriptable static-content CDN).
_ESM_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1186%2Fs12934-024-02309-0/MediaObjects/"
    "12934_2024_2309_MOESM{n}_ESM.xlsx"
)

RAW_MIRROR_REL = f"torchcell-raw/{CITATION_KEY}"
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
            page="Microb Cell Fact 2024 23:57, Methods",
        ),
    )


_BACKGROUND = _sv(
    "BY4741",
    "Te haploid parental strain S. cerevisiae BY4741 (MATa, his3∆1, leu2∆0, met15∆0, "
    "ura3∆0) and the collection of derived single deletion mutants, obtained from "
    "Euroscarf (Frankfurt, Germany), were used for the chemogenomic analysis.",
    note="the OCR's 'Te' is the publisher PDF's dropped-ligature 'The'; the quote is "
    "kept verbatim as the mirrored artifact renders it",
)

_MEDIUM = _sv(
    (YPD_AGAR, "hydrochloric acid"),
    "in liquid YPD medium containing, $2 0 ~ \\mathrm { g / L }$ glucose (Merck, "
    "Darmstadt, Germany), $1 0 ~ \\mathrm { g / L }$ yeast extract and $2 0 ~ \\mathrm "
    "{ g / L }$ peptone, both from BD Biosciences (Franklin Lakes, NJ, USA) acidifed "
    "with HCl until $\\mathrm { p H } 4 . 5$ . Solid media were prepared by addition of "
    "$2 0 \\ \\mathrm { g / L }$ agar (NZYTech, Lisbon, Portugal).",
    note="the three YPD ingredients at the shared media.YPD percentages plus the 20 g/L "
    "agar row = media.YPD_AGAR. The pH is NOT folded into the medium: it is the typed "
    "EnvironmentPhysicalPerturbation(factor=ph) whose agent is the HCl this sentence "
    "names",
)

_PH = _sv(
    4.5,
    "acidifed with HCl until $\\mathrm { p H } 4 . 5$",
    note="carried on BOTH the treated and the reference environment -- the control "
    "wells sit on the same pH 4.5 plates",
)

_TEMPERATURE = _sv(
    30.0,
    "the entire BY4741 Euroscarf deletion mutant collection was screened for "
    "susceptibility to the selected equivalent inhibitory concentrations of $7 5 ~ "
    "\\mathrm { ~ m M }$ acetic acid, $1 4 ~ \\mathrm { m M }$ butyric acid, or $0 . 3 ~ "
    "\\mathrm { m M }$ octanoic acid, at $3 0 ~ \\mathrm { ^ { \\circ } C }$ , in YPD "
    "medium at $\\mathrm { p H } ~ 4 . 5$ .",
)

_CONCENTRATIONS = _sv(
    {"acetic acid": 75.0, "butyric acid": 14.0, "octanoic acid": 0.30},
    "Equivalent mild growth inhibitory concentrations for the parental strain were "
    "found by the supplementation of YPD solid medium at $\\mathsf { p H } 4 . 5$ with "
    "$7 5 ~ \\mathrm { m M }$ $( 4 . 5 8 ~ \\mathrm { g / L } )$ of acetic acid, or $1 4 "
    "~ \\mathrm { m M }$ $( 1 . 2 3 ~ \\mathrm { g / L } )$ of butyric acid, or $0 . 3 ~ "
    "\\mathrm { m M }$ $( 0 . 0 4 ~ \\mathrm { g / L } )$ of octanoic acid (Fig.\xa0 3).",
    note="the octanoic dose is written '0.3 mM' in the Methods and '0.30 mM' in the "
    "Figure S2 caption; the same number, stored as 0.30",
)

_SCREEN_DESIGN = _sv(
    "wild type in empty wells of the same plates",
    "To be used as control, the wild-type strain was prepared individually under the "
    "same conditions as the deletion mutants and was inoculated in empty wells, "
    "according to the original display of the haploid yeast deletion mutant collection "
    "plates.",
    note="the comparator is the PARENTAL strain on the SAME acid plate, which is why "
    "the reference environment keeps the acid and the pH perturbation",
)

_ASSAY = _sv(
    AssayType.spot_dilution,
    "Using a 96-pin replica platter, the cell suspensions were spotted onto the surface "
    "of YPD solid medium supplemented, or not, with the selected concentrations and "
    "incubated at $3 0 ~ \\mathrm { ^ { \\circ } C } .$",
    note="the collection screen itself is a single-density 96-pin stamp read as spot "
    "growth; the serial dilutions were used to CHOOSE the doses, not to score the "
    "collection. spot_dilution is the closest AssayType member and the distinction is "
    "recorded here",
)

_DURATION = _sv(
    48.0,
    "When observed, the susceptibility phenotype of each single deletion mutant was "
    "scored as $( + )$ if the mutant strain showed, compared with the parental strain, "
    "a slight to moderate growth inhibition after the standardized incubation time, and "
    "$( + + )$ if no growth was observed after $4 8 \\ \\mathrm { h }$ of incubation "
    "(visual criteria illustrated in the Additional fle\xa0 17: Figure S2).",
    note="RULE: duration_hours is the time at which the STORED call was made, and the "
    "(++) definition anchors that at 48 h. The Methods separately give a RANGE for when "
    "the acid plates were photographed -- 'Photographs were taken after 24 h of "
    "incubation for control plates (YPD medium) or 36-48 h in the presence of the "
    "acids' -- so 36-48 h is the photograph window and 24 h is the control-plate "
    "reading; neither is the scoring anchor",
)

_SCORING = _sv(
    {"+": ResponseCategory.reduced, "++": ResponseCategory.severely_reduced},
    "When observed, the susceptibility phenotype of each single deletion mutant was "
    "scored as $( + )$ if the mutant strain showed, compared with the parental strain, "
    "a slight to moderate growth inhibition after the standardized incubation time, and "
    "$( + + )$ if no growth was observed after $4 8 \\ \\mathrm { h }$ of incubation",
)

_FIGURE_S2 = _sv(
    ResponseCategory.no_change,
    "Two levels of susceptibility were defned: $( + )$ when the growth inhibition of "
    "the single mutant strains was minor to moderate or $( + + )$ referring to total "
    "growth inhibition compared to wild-type. ${ } ^ { \\prime \\prime } 0 ^ { \\prime "
    "\\prime }$ corresponds to an absence of a detectable susceptibility phenotype.",
    note="the source's own third grade, '0', is the reference's call: rank 0.0, "
    "ResponseCategory.no_change, category_label '0'",
)

#: The ordinal scale, in the source's own symbols. The rank is what
#: ``environment_response`` carries; the spacing between ranks means nothing, which is
#: what ``MeasurementType.ordinal`` records.
_RANK = {"0": 0.0, "+": 1.0, "++": 2.0}
_REFERENCE_SYMBOL = "0"

MEASUREMENT_UNITS = (
    "ordinal spot-assay susceptibility grade vs the parental BY4741 strain on the same "
    "plate after 48 h on solid YPD (pH 4.5, HCl): 0 = no detectable susceptibility, "
    "1 = (+) minor to moderate growth inhibition, 2 = (++) total growth inhibition. The "
    "rank is an ORDER, not a quantity: the spacing between grades has no meaning"
)

_DEDUP_RULE = (
    "one record per (resolved systematic ORF, acid): where two source rows claim the "
    "same gene, the MORE SEVERE grade wins (++ > +) and a tie is broken by the "
    "lexicographically smallest source token. Where two DIFFERENT tokens claim one gene "
    "the genome's canonical common name is stored instead of either token, so one gene "
    "carries one spelling"
)

DROP_RULE = (
    "a gene token resolves through SCerevisiaeGenome.resolve_gene_name to a status "
    "other than CURRENT or RENAMED (RETIRED: absent from R64-4-1); the token's row is "
    "dropped and counted"
)

#: Retention rule: a token is kept only when it resolves to a LIVE R64 gene.
_KEPT_STATUSES = frozenset({GeneNameStatus.CURRENT, GeneNameStatus.RENAMED})

# Per-acid source spec: MOESM index, filename, sha256, compound name.
_ACID_SPECS: list[dict[str, Any]] = [
    {
        "acid": "acetic",
        "n": 1,
        "filename": "12934_2024_2309_MOESM1_ESM.xlsx",
        "sha256": "b23ad28141e70b307048fc69475aedd4e3cf880118ae9d0d806b6d9f91205e42",
        "compound_name": "acetic acid",
    },
    {
        "acid": "butyric",
        "n": 2,
        "filename": "12934_2024_2309_MOESM2_ESM.xlsx",
        "sha256": "a7a1aaee1c76e52d8fe435326790c89170ab43ec96b92ef272903d8e78a1e81f",
        "compound_name": "butyric acid",
    },
    {
        "acid": "octanoic",
        "n": 3,
        "filename": "12934_2024_2309_MOESM3_ESM.xlsx",
        "sha256": "27f1508641ad5e7cc29ab8611739d4940da355c1dffbe9c9a908c267cdf5d455",
        "compound_name": "octanoic acid",
    },
]


def _screen_gaps() -> list[ProvenanceGap]:
    """The two replicate-design fields the disruptome screen never reports.

    A fresh list per phenotype: ``ProvenanceGap`` objects are shared-safe, but pydantic
    would otherwise alias one list across every record.
    """
    return [
        ProvenanceGap(
            field="n_samples", reason=ProvenanceGapReason.not_reported_by_primary
        ),
        ProvenanceGap(
            field="sample_unit", reason=ProvenanceGapReason.not_reported_by_primary
        ),
    ]


def raw_mirror_dir(data_root: str | None = None) -> str:
    """``$DATA_ROOT/torchcell-raw/motaSharedMoreSpecific2024``."""
    return osp.join(data_root or os.environ["DATA_ROOT"], RAW_MIRROR_REL)


def _raw_relpath(spec: dict[str, Any]) -> str:
    """Mirror-relative path of one supplementary spreadsheet."""
    return f"si/{spec['filename']}"


def deposit_raw_mirror(
    *,
    source_dir: str,
    retrieved_at: str,
    checked_at: str | None = None,
    data_root: str | None = None,
) -> str:
    """Copy the three consumed spreadsheets into the raw mirror + write their manifest.

    Idempotent by sha256. ``checked_at`` records an ISO timestamp at which the Springer
    ESM URLs were re-run and each produced the pinned sha256, so the manifest carries a
    real ``SourceCheck`` rather than an assumption that the URL still works.
    """
    root = Path(raw_mirror_dir(data_root))
    files: list[ArtifactRecord] = []
    for spec in _ACID_SPECS:
        src = Path(source_dir) / spec["filename"]
        digest = sha256_file(src)
        if digest != spec["sha256"]:
            raise RuntimeError(
                f"{src} sha256 {digest} != pinned {spec['sha256']}; refusing to deposit"
            )
        dest = root / _raw_relpath(spec)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if sha256_file(dest) != spec["sha256"]:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            dest.write_bytes(src.read_bytes())
        url = _ESM_URL.format(n=spec["n"])
        files.append(
            ArtifactRecord(
                path=_raw_relpath(spec),
                role=ROLE_SI_DATA,
                bytes=dest.stat().st_size,
                sha256=spec["sha256"],
                source=url,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.springer_esm,
                    source_url=url,
                    retriever="torchcell.literature.retrieve.springer_esm",
                    params={"url": url},
                    sha256=spec["sha256"],
                    retrieved_at=retrieved_at,
                    last_check=(
                        None
                        if checked_at is None
                        else SourceCheck(
                            checked_at=checked_at,
                            produced_sha256=spec["sha256"],
                            matches=True,
                        )
                    ),
                ),
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=(
            "Shared and more specific genetic determinants and pathways underlying "
            "yeast tolerance to acetic, butyric, and octanoic acids"
        ),
        library_id="6582362",
        zotero_item_key="4JMAVP2G",
        files=files,
        si_data_sources=[_ESM_URL.format(n=spec["n"]) for spec in _ACID_SPECS],
        si_expected=[
            "Additional file 1: Table S1 (acetic acid susceptible mutants)",
            "Additional file 2: Table S2 (butyric acid susceptible mutants)",
            "Additional file 3: Table S3 (octanoic acid susceptible mutants)",
        ],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return str(root)


class DroppedToken(BaseModel):
    """One source gene token the retention rule removed."""

    token: str = Field(description="the 'Gene/ORF name' cell, verbatim")
    status: str = Field(description="GeneNameStatus the shared resolver returned")
    acids: list[str] = Field(description="the acid tables the token appears in")
    n_records: int = Field(description="records lost with this token")


class MergedGene(BaseModel):
    """One gene claimed by more than one source row, and how the merge was resolved."""

    systematic_name: str
    acid: str
    source_tokens: list[str]
    source_scores: list[str]
    kept_score: str
    stored_gene_name: str


class DropLog(BaseModel):
    """The build's retention + dedup accounting, written beside ``processed/``."""

    dataset: str
    rule: str
    dedup_rule: str
    n_raw_rows: int
    n_kept_records: int
    n_dropped_records: int
    n_merged_records: int
    dropped: list[DroppedToken] = Field(default_factory=list)
    merged: list[MergedGene] = Field(default_factory=list)


@register_dataset
class EnvChemgenMota2024Dataset(ExperimentDataset):
    """Acetic/butyric/octanoic acid chemogenomic env x geno -> ordinal susceptibility."""

    def __init__(
        self,
        root: str = "data/torchcell/env_chemgen_mota2024",
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
        """The three BMC supplementary spreadsheets required before processing."""
        return [spec["filename"] for spec in _ACID_SPECS]

    def download(self) -> None:
        """Take the three spreadsheets from the raw mirror; verify each pinned sha256.

        The mirror is the source of record. The Springer ESM CDN is scriptable and is
        used only when the mirror does not hold the file yet, so a rebuild does not
        depend on a live URL surviving.
        """
        import urllib.request

        os.makedirs(self.raw_dir, exist_ok=True)
        for spec in _ACID_SPECS:
            dest = osp.join(self.raw_dir, spec["filename"])
            if not osp.exists(dest):
                mirror = osp.join(raw_mirror_dir(), _raw_relpath(spec))
                if osp.exists(mirror):
                    shutil.copyfile(mirror, dest)
                else:
                    url = _ESM_URL.format(n=spec["n"])
                    log.info("raw mirror missing %s; fetching %s", spec["acid"], url)
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "Mozilla/5.0"}
                    )
                    with urllib.request.urlopen(req, timeout=180) as resp:
                        data = resp.read()
                    with open(dest, "wb") as handle:
                        handle.write(data)
            digest = hashlib.sha256(open(dest, "rb").read()).hexdigest()
            if digest != spec["sha256"]:
                raise RuntimeError(
                    f"{spec['filename']} sha256 mismatch: got {digest}, "
                    f"expected {spec['sha256']}"
                )

    def _parse_acid(
        self, spec: dict[str, Any]
    ) -> tuple[dict[str, tuple[str, str]], dict[str, str], list[MergedGene], int]:
        """Parse one acid spreadsheet into ``{ORF: (score symbol, stored gene name)}``.

        Returns the kept records, the dropped tokens keyed by their resolver status, the
        merge accounting, and the raw susceptible-row count. Deduplication follows
        ``_DEDUP_RULE``; retention follows ``DROP_RULE``.
        """
        if self.genome is None:
            raise RuntimeError(
                "EnvChemgenMota2024Dataset requires a genome for gene-name resolution; "
                "inject SCerevisiaeGenome(...)"
            )
        resolve = self.genome.resolve_gene_name
        attributes = self.genome.gene_attribute_table
        standard_name = dict(zip(attributes["ID"], attributes["gene"], strict=True))

        path = osp.join(self.raw_dir, spec["filename"])
        workbook = openpyxl.load_workbook(path, read_only=True)
        sheet = workbook[workbook.sheetnames[0]]
        rows = list(sheet.iter_rows(values_only=True))
        header_idx = next(
            i for i, row in enumerate(rows) if row and row[0] == "Gene/ORF name"
        )
        claims: dict[str, list[tuple[str, str]]] = {}
        dropped: dict[str, str] = {}
        n_raw = 0
        for row in rows[header_idx + 1 :]:
            if not row or row[0] is None or row[2] is None:
                continue
            token = str(row[0]).replace("\xa0", " ").strip()
            score = str(row[2]).strip()
            if not token or score not in _RANK or score == _REFERENCE_SYMBOL:
                continue
            n_raw += 1
            resolution = resolve(token)
            if resolution.status not in _KEPT_STATUSES:
                dropped[token] = resolution.status.value
                continue
            claims.setdefault(str(resolution.systematic_name), []).append(
                (score, token)
            )

        best: dict[str, tuple[str, str]] = {}
        merged: list[MergedGene] = []
        for orf, entries in claims.items():
            score, token = min(entries, key=lambda e: (-_RANK[e[0]], e[1]))
            tokens = sorted({t for _, t in entries})
            if len(tokens) > 1:
                canonical = standard_name.get(orf)
                token = canonical if isinstance(canonical, str) else orf
            if len(entries) > 1:
                merged.append(
                    MergedGene(
                        systematic_name=orf,
                        acid=spec["acid"],
                        source_tokens=tokens,
                        source_scores=sorted(s for s, _ in entries),
                        kept_score=score,
                        stored_gene_name=token,
                    )
                )
            best[orf] = (score, token)
        log.info(
            "Mota2024 %s acid: %d raw susceptible rows -> %d records "
            "(dropped %d retired tokens: %s; merged %d genes)",
            spec["acid"],
            n_raw,
            len(best),
            len(dropped),
            sorted(dropped),
            len(merged),
        )
        return best, dropped, merged, n_raw

    def _perturbations(self, spec: dict[str, Any]) -> list[Any]:
        """The acid at its sourced molarity plus the typed pH edit that sets the plate."""
        return [
            SmallMoleculePerturbation(
                compound=resolved_compound(spec["compound_name"]),
                concentration=Concentration(
                    value=_CONCENTRATIONS.value[spec["compound_name"]],
                    unit=ConcentrationUnit.millimolar,
                ),
            ),
            EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.ph,
                magnitude=Concentration(value=_PH.value, unit=ConcentrationUnit.ph),
                agent=resolved_compound("hydrochloric acid"),
            ),
        ]

    def _environment(self, spec: dict[str, Any]) -> Environment:
        """Aerobic solid-YPD plate, acidified to pH 4.5 with HCl, carrying the acid."""
        return Environment(
            media=YPD_AGAR,
            temperature=Temperature(value=_TEMPERATURE.value),
            perturbations=self._perturbations(spec),
            aerobicity="aerobic",
            duration_hours=_DURATION.value,
        )

    def _reference(
        self, environment: Environment
    ) -> EnvironmentResponseExperimentReference:
        """Parental BY4741 on the SAME acid plate: the source's own grade 0.

        ``n_samples`` / ``sample_unit`` are TYPED GAPS: the paper's replicate statements
        belong to the CFU-viability and pHi assays, and the disruptome screen states no
        replicate count.
        """
        return EnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain=_BACKGROUND.value
            ),
            environment_reference=environment.model_copy(),
            phenotype_reference=EnvironmentResponsePhenotype(
                measurement_type=MeasurementType.ordinal,
                assay_type=_ASSAY.value,
                environment_response=_RANK[_REFERENCE_SYMBOL],
                category=_FIGURE_S2.value,
                category_label=_REFERENCE_SYMBOL,
                units=MEASUREMENT_UNITS,
                provenance_gaps=_screen_gaps(),
            ),
        )

    def _experiment(
        self, *, orf: str, gene_name: str, score: str, environment: Environment
    ) -> EnvironmentResponseExperiment:
        """Build one env x geno -> ordinal-susceptibility experiment for (gene, acid)."""
        genotype = Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=gene_name
                )
            ]
        )
        phenotype = EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.ordinal,
            assay_type=_ASSAY.value,
            environment_response=_RANK[score],
            category=_SCORING.value[score],
            category_label=score,
            units=MEASUREMENT_UNITS,
            provenance_gaps=_screen_gaps(),
        )
        return EnvironmentResponseExperiment(
            dataset_name=self.name,
            genotype=genotype,
            environment=environment,
            phenotype=phenotype,
        )

    @post_process
    def process(self) -> None:
        """Parse the three acid spreadsheets into ordinal records; write LMDB + drop log."""
        publication = Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")
        pub_dump = publication.model_dump()

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        idx = 0
        n_raw_total = 0
        dropped_by_token: dict[str, DroppedToken] = {}
        merged_all: list[MergedGene] = []
        with env.begin(write=True) as txn:
            for spec in _ACID_SPECS:
                environment = self._environment(spec)
                ref_dump = self._reference(environment).model_dump()
                orf_scores, dropped, merged, n_raw = self._parse_acid(spec)
                n_raw_total += n_raw
                merged_all.extend(merged)
                for token, status in dropped.items():
                    entry = dropped_by_token.get(token)
                    if entry is None:
                        dropped_by_token[token] = DroppedToken(
                            token=token,
                            status=status,
                            acids=[spec["acid"]],
                            n_records=1,
                        )
                    else:
                        entry.acids.append(spec["acid"])
                        entry.n_records += 1
                for orf, (score, gene_name) in tqdm(
                    sorted(orf_scores.items()), desc=f"{spec['acid']} acid"
                ):
                    experiment = self._experiment(
                        orf=orf,
                        gene_name=gene_name,
                        score=score,
                        environment=environment,
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
            dedup_rule=_DEDUP_RULE,
            n_raw_rows=n_raw_total,
            n_kept_records=idx,
            n_dropped_records=sum(e.n_records for e in dropped_by_token.values()),
            n_merged_records=n_raw_total
            - sum(e.n_records for e in dropped_by_token.values())
            - idx,
            dropped=sorted(dropped_by_token.values(), key=lambda e: e.token),
            merged=merged_all,
        )
        with open(osp.join(self.root, _DROPPED_FILENAME), "w") as handle:
            handle.write(drop_log.model_dump_json(indent=2))
        log.info(
            "Wrote %d Mota2024 environment-response experiments to LMDB "
            "(%d raw rows, %d dropped, %d merged)",
            idx,
            n_raw_total,
            drop_log.n_dropped_records,
            drop_log.n_merged_records,
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
    root = osp.join(data_root, "data/torchcell/env_chemgen_mota2024")
    dataset = EnvChemgenMota2024Dataset(root=root, genome=genome)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()
