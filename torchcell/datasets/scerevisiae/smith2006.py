# torchcell/datasets/scerevisiae/smith2006
# [[torchcell.datasets.scerevisiae.smith2006]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/smith2006
# Test file: tests/torchcell/datasets/scerevisiae/test_smith2006.py
"""Smith 2006 fatty-acid clear-zone screen (env x geno -> ordinal categorical response).

Smith et al. 2006 (Mol Syst Biol 2:2006.0009, doi:10.1038/msb4100051; PMID 16738555;
PMC1681483) screened the entire ``matalpha`` haploid viable S. cerevisiae gene-deletion
set (BY4742, Resgen/Invitrogen KanMX deletions) for defects in peroxisomal fatty-acid
beta-oxidation. Two readouts on solid agar omnitrays:

- CLEAR ZONE (``AssayType.halo_zone``): a strain clears the turbid fatty-acid medium
  (oleate = YPBO, myristate = YPBM) as it metabolizes the fatty acid. Zone size is scored
  4/3/2/1 (larger / wild type / less / small-or-absent).
- ACETATE GROWTH (``AssayType.colony_size_array``): beta-oxidation needs a functional
  mitochondrial electron transport chain, so growth of the pinned patch on the
  nonfermentable carbon source acetate (YPBA) is the control, scored 3/2/1.

RECORD = one (deletion strain x condition) ``EnvironmentResponseExperiment``:

- GENOTYPE: one ``KanMxDeletionPerturbation`` in the BY4742 background. The common name
  is the GENOME's own standard name for the resolved ORF, not the 2005-era Standard Name
  column, so one gene carries one spelling across every torchcell dataset.
- ENVIRONMENT: the shared ``MEDIA_LIBRARY`` medium (``YPBO`` / ``YPBM`` / ``YPBA``),
  whose typed components carry the published recipe INCLUDING the fatty acid or acetate
  as its ``carbon_source`` component. The experimental EDIT is therefore a carbon-source
  change, recorded as ``EnvironmentPhysicalPerturbation(factor=carbon_source, agent=...)``
  -- the same convention condition-SGA (Costanzo 2021) uses for its galactose arm. It is
  NOT a ``SmallMoleculePerturbation``: none of these three species is a stress dosed on
  top of a complete medium, each IS the plate's sole carbon and energy source.
- PHENOTYPE: ``measurement_type=ordinal``. The released 1-4 score is stored verbatim on
  ``environment_response``; ``category`` is the typed cross-screen ``ResponseCategory``
  call and ``category_label`` keeps this screen's own word for it.

SOURCED VALUES (every number is a module-level ``SourcedValue`` anchored to the sha256 of
``smithExpressionFunctionalProfiling2006/paper.md``, or a typed ``ProvenanceGap``):

- ``duration_hours = 72.0``. The source gives a RANGE, "Plates were incubated for 3-4 days
  at 30 C". No companion statistic permits a back-solve, so the CLAUDE.md range rule takes
  the CONSERVATIVE end (3 days = 72 h). The previous build asserted an unmarked 84.0 h
  midpoint.
- ``n_samples = 3``, ``sample_unit=biological_replicate``: "Colonies were replicated in
  triplicate onto acetate, oleate or myristate agar omnitrays." The quadruplicate YEPD
  pinning is within-plate technical replication of the source colony and is not counted.
- ``Temperature(30.0)`` and ``aerobicity="aerobic"`` from the same incubation sentence.

RECORDS DROPPED (rule + counts written to ``preprocess/dropped_records.json``):

1. ``strain_failed_the_yepd_growth_control``. Column 7 of the released table flags strains
   with no growth (NG, 263) or low growth (LG, 208) on the YEPD plate the whole set was
   pinned on before replication, plus one NG/CONT. A clear zone needs a growing patch, so
   these scores measure the growth failure, not beta-oxidation. MEASURED on the released
   table: among NG strains 98.1% / 98.9% / 96.2% score the lowest value on oleate /
   myristate / acetate against 2.0% / 2.2% / 0.3% of unflagged strains, and 79.8% / 76.0%
   of LG strains score the lowest clear-zone value with 70.7% scoring the middle acetate
   value. The readout is confounded rather than merely noisy, and nothing in the record
   could carry the flag (``EnvironmentResponsePhenotype`` has no QC-flag field), so the
   strains are dropped and counted.
2. ``systematic_name_does_not_resolve_to_a_current_orf`` and
   ``alias_resolution_collides_with_a_directly_present_orf``. A name already a current R64
   id is taken directly; an old systematic name follows the genome alias table ONLY when
   its target is not already claimed by a direct-id row, so a dubious ORF absorbed into a
   verified neighbour (YOR240W -> YOR239W, present as ABP140) is dropped, not mislabelled.

NOT INGESTED (deliberate, recorded here rather than silently): the ``Adherence`` column
(79 strains, a second released phenotype the paper analyses separately) and the
``retested myristate`` column. 27 strains carry the retest flag, whose legend describes a
DIFFERENT assay ("growing strains overnight in YEPD without agar, washing them with water,
and spotting 2 mL of the suspension ... onto YPBA and YPBM agar plates"); the release does
not state whether the myristate column holds the retest or the original pinning score, so
the stored ``assay_type`` stays ``halo_zone`` for all of them and the ambiguity is flagged
for review rather than guessed.

DATA SOURCE: Supplementary Table 1, ``msb4100051-s1.xls`` (legacy BIFF .xls, header on
0-based row 23, 4770 strain rows), deposited in the raw mirror
``$DATA_ROOT/torchcell-raw/smithExpressionFunctionalProfiling2006/`` with a
``manifest.json`` recording the Europe PMC supplementary-bundle retrieval and the file's
sha256. The loader reads the mirror and verifies that hash; the URL is retrieval metadata.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import os.path as osp
import re
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import YPBA, YPBM, YPBO
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
    SampleUnit,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DOI = "10.1038/msb4100051"
PMID = "16738555"
PMCID = "PMC1681483"

CITATION_KEY = "smithExpressionFunctionalProfiling2006"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

XLS_FILENAME = "msb4100051-s1.xls"
XLS_REL = f"data/{XLS_FILENAME}"
XLS_SHA256 = "7048663ffa4890478724e6e371f434baccc7160e6d8250df9a777a26c6b283a4"
XLS_RETRIEVED_AT = "2026-09-12"

#: Europe PMC serves the article's supplementary files as ONE zip, re-packed per request.
SUPPLEMENTARY_ZIP_URL = (
    f"https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/supplementaryFiles"
)

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "eb5ab21b842365e2138528bbce936bd68134dd97ee99eb9a58502c25ca2948c6"

#: 0-based header row of the single sheet; the 4770 strain rows sit below it.
HEADER_ROW = 23
SYSTEMATIC_COL = "Systematic Name"
STANDARD_COL = "Standard Name"
YEPD_QC_COL = "Glucose (YEPD)"

#: Column-7 flags that mark a strain as having failed the YEPD growth control.
YEPD_FAIL_FLAGS = frozenset({"NG", "LG", "NG/CONT"})


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
            page="Materials and methods, 'Myristate screen and generation of the "
            "fitness data set'",
        ),
    )


_INCUBATION_QUOTE = (
    "Plates were incubated for 3-4 days at $3 0 ^ { \\circ } \\mathrm { C }$"
)

TEMPERATURE_C = _paper(30.0, _INCUBATION_QUOTE)
AEROBICITY = _paper(
    "aerobic",
    _INCUBATION_QUOTE,
    note="agar omnitrays incubated on the bench; no anaerobic chamber is described, and "
    "beta-oxidation and growth on a nonfermentable carbon source both require oxygen",
)
DURATION_HOURS = _paper(
    72.0,
    _INCUBATION_QUOTE,
    note="the source gives a RANGE (3-4 days) and no per-plate value, and no companion "
    "statistic permits a back-solve, so the CLAUDE.md range rule takes the CONSERVATIVE "
    "lower end, 3 days = 72 h (the previous build asserted an unmarked 84.0 h midpoint)",
)
N_REPLICATES = _paper(
    3,
    "Colonies were replicated in triplicate onto acetate, oleate or myristate agar "
    "omnitrays.",
    note="the triplicate replicate PLATES are the independent replicates; the "
    "quadruplicate YEPD pinning is within-plate technical replication of one source "
    "colony and is not counted",
)
PINNING_QUOTE = _paper(
    "quadruplicate YEPD pinning",
    "The entire deletion set was pinned in quadruplicate on YEPD agar",
    note="the rich-medium plate the whole set was pinned on before replication onto the "
    "fatty-acid plates; column 7 of the released table flags the strains that failed it",
)
CLEAR_ZONE_SCALE = _paper(
    {
        4.0: "larger than wild type",
        3.0: "wild type",
        2.0: "less than wild type",
        1.0: "small or not detectable",
    },
    "Clear zone sizes around cell patches were scored as 4 for larger than wild type, 3 "
    "for wild type, 2 for less than wild type and 1 for small or not detectable.",
)
GROWTH_SCALE = _paper(
    {3.0: "wild type", 2.0: "moderate", 1.0: "little/no growth"},
    "Growth was scored as 3, 2 or 1 for patches with wild type, moderate or little/no "
    "growth, respectively.",
    note="an undocumented acetate value of 2.5 also appears in the released table and is "
    "kept VERBATIM on environment_response, labelled 'intermediate'",
)

_CLEAR_ZONE_UNITS = (
    "clear-zone size around the cell patch on turbid fatty-acid agar, scored by visual "
    "inspection: 4=larger than wild type, 3=wild type, 2=less than wild type, 1=small or "
    "not detectable"
)
_GROWTH_UNITS = (
    "growth of the cell patch on nonfermentable acetate, scored by visual inspection: "
    "3=wild-type, 2=moderate, 1=little/no growth (an undocumented 2.5=intermediate is "
    "kept verbatim from the released table)"
)

#: Ordinal score -> (typed ResponseCategory, this screen's own word for the call).
CLEAR_ZONE_CATEGORY: dict[float, tuple[ResponseCategory, str]] = {
    4.0: (ResponseCategory.enhanced, "enhanced"),
    3.0: (ResponseCategory.no_change, "wild_type"),
    2.0: (ResponseCategory.reduced, "reduced"),
    1.0: (ResponseCategory.severely_reduced, "defective"),
}
GROWTH_CATEGORY: dict[float, tuple[ResponseCategory, str]] = {
    3.0: (ResponseCategory.no_change, "wild_type"),
    2.5: (ResponseCategory.mildly_reduced, "intermediate"),
    2.0: (ResponseCategory.reduced, "moderate"),
    1.0: (ResponseCategory.severely_reduced, "poor"),
}

#: The parental BY4742 baseline: the wild-type call on either readout.
REFERENCE_CATEGORY = ResponseCategory.no_change
REFERENCE_CATEGORY_LABEL = "wild_type"

#: One entry per screened condition: the released column, the shared medium object, the
#: carbon source that medium's recipe names, the assay, and the ordinal legend.
CONDITION_SPECS: list[dict[str, Any]] = [
    {
        "column": "Oleate (YPBO)",
        "media": YPBO,
        "carbon_source": "oleic acid",
        "carbon_percent": 0.1,
        "assay_type": AssayType.halo_zone,
        "category_map": CLEAR_ZONE_CATEGORY,
        "units": _CLEAR_ZONE_UNITS,
    },
    {
        "column": "Myristae (YPBM)",  # verbatim header typo for "Myristate"
        "media": YPBM,
        "carbon_source": "myristic acid",
        "carbon_percent": 0.125,
        "assay_type": AssayType.halo_zone,
        "category_map": CLEAR_ZONE_CATEGORY,
        "units": _CLEAR_ZONE_UNITS,
    },
    {
        "column": "Acetate (YPBA)",
        "media": YPBA,
        "carbon_source": "acetate",
        "carbon_percent": 2.0,
        "assay_type": AssayType.colony_size_array,
        "category_map": GROWTH_CATEGORY,
        "units": _GROWTH_UNITS,
    },
]

_SYSTEMATIC_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$")


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror + build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/smithExpressionFunctionalProfiling2006``."""
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
    xls_path: str | Path,
    retrieved_at: str = XLS_RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror from the already-retrieved .xls plus its ``manifest.json``.

    Idempotent by sha256: an existing mirror file with the recorded hash is left alone, a
    differing one raises. The Europe PMC REST endpoint serves ALL of the article's
    supplementary files as one zip that is re-packed per request, so the container's
    sha256 is not stable (measured: two retrievals on 2026-09-12 produced different
    container hashes and the SAME member hash). The recorded retrieval therefore calls
    ``retrieve.zip_member`` with ``container_sha256=None``, which is the case that
    function documents, and the pinned MEMBER sha256 is the anchor a rebuild verifies.
    """
    root = raw_mirror_dir(data_root)
    got = _sha256(xls_path)
    if got != XLS_SHA256:
        raise RuntimeError(
            f"{xls_path} sha256 mismatch: got {got}, expected {XLS_SHA256}"
        )
    dest = root / XLS_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if _sha256(dest) != XLS_SHA256:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    else:
        shutil.copy2(xls_path, dest)
    retrieval = RetrievalRecord(
        method=RetrievalMethod.pmc_oa_api,
        source_url=SUPPLEMENTARY_ZIP_URL,
        retriever="torchcell.literature.retrieve.zip_member",
        params={
            "url": SUPPLEMENTARY_ZIP_URL,
            "member": XLS_FILENAME,
            # The Europe PMC supplementaryFiles endpoint re-zips the bundle on every
            # request, so the CONTAINER sha256 is not stable and cannot be pinned
            # (measured 2026-09-12: two retrievals, two container hashes, one member
            # hash). ``zip_member`` takes None for exactly this case; the pinned member
            # sha256 below is the anchor, and it is verified on every rebuild.
            "container_sha256": None,
        },
        sha256=XLS_SHA256,
        retrieved_at=retrieved_at,
    )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=(
            "Expression and functional profiling reveal distinct gene classes involved "
            "in fatty acid metabolism"
        ),
        files=[
            ArtifactRecord(
                path=XLS_REL,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=XLS_SHA256,
                source=SUPPLEMENTARY_ZIP_URL,
                retrieval=retrieval,
            )
        ],
        si_data_sources=[SUPPLEMENTARY_ZIP_URL],
        si_expected=[
            "Supplementary Table 1 (msb4100051-s1.xls) -- the per-strain growth, clear "
            "zone and adhesion table this dataset is built from"
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


# --------------------------------------------------------------------------- #
# Retention bookkeeping
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One retention rule, the records it removed, and the items it removed them for."""

    rule: str
    scope: Literal["strain", "cell"]
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


class ResolvedStrains(BaseModel):
    """The strain rows a build keeps, plus the per-rule drop lists."""

    orf: list[str]
    common: list[str]
    row_index: list[int]
    dropped_unresolved: list[str]
    dropped_alias_collision: list[str]
    dropped_duplicate: list[str]
    dropped_yepd_qc: list[str]


def canonical_common_names(genome: SCerevisiaeGenome) -> dict[str, str]:
    """``systematic name -> the genome's own standard (common) name``.

    One spelling per gene is what keeps a perturbation from splitting into two graph
    nodes, and taking it from the genome rather than from the 2005-era Standard Name
    column is what makes the spelling identical across datasets. Only a standard name
    that resolves BACK to the gene is used, so the stored pair round-trips through the
    resolver (the L1 ``canonical_gene_names`` rule).
    """
    canonical: dict[str, str] = {}
    for standard in genome.feature_index["standard_to_ids"]:
        resolution = genome.resolve_gene_name(standard)
        if resolution.is_current_gene and resolution.systematic_name is not None:
            canonical.setdefault(resolution.systematic_name, standard)
    return canonical


def resolve_strains(df: pd.DataFrame, genome: SCerevisiaeGenome) -> ResolvedStrains:
    """Map every released strain row onto a current R64 ORF, or drop it with a reason.

    A name already a current R64 id resolves to itself; an old systematic name follows
    the genome alias table to its current id ONLY when that target is not already a
    directly-present ORF, so a dubious ORF absorbed into a verified neighbour is dropped
    rather than mislabelled as the neighbour's deletion. A strain flagged NG/LG on the
    YEPD growth control is dropped by rule 1 of the module docstring.
    """
    names = [str(v).upper().strip() for v in df[SYSTEMATIC_COL].tolist()]
    qc = [str(v).upper().strip() for v in df[YEPD_QC_COL].tolist()]
    ids = set(genome.gene_attribute_table["ID"])
    alias_map = genome.alias_to_systematic
    direct = {name for name in names if name in ids}
    canonical = canonical_common_names(genome)

    orfs: list[str] = []
    commons: list[str] = []
    rows: list[int] = []
    unresolved: list[str] = []
    collision: list[str] = []
    duplicate: list[str] = []
    yepd: list[str] = []
    seen: set[str] = set()
    for row, (name, flag) in enumerate(zip(names, qc, strict=True)):
        if name in ids:
            orf: str | None = name
        elif _SYSTEMATIC_RE.match(name):
            candidates = alias_map.get(name, [])
            if candidates and candidates[0] in ids and candidates[0] not in direct:
                orf = candidates[0]
            elif candidates and candidates[0] in direct:
                collision.append(name)
                continue
            else:
                unresolved.append(name)
                continue
        else:
            unresolved.append(name)
            continue
        assert orf is not None
        if orf in seen:
            duplicate.append(name)
            continue
        seen.add(orf)
        if flag in YEPD_FAIL_FLAGS:
            yepd.append(orf)
            continue
        orfs.append(orf)
        commons.append(canonical.get(orf, orf))
        rows.append(row)
    return ResolvedStrains(
        orf=orfs,
        common=commons,
        row_index=rows,
        dropped_unresolved=sorted(unresolved),
        dropped_alias_collision=sorted(collision),
        dropped_duplicate=sorted(duplicate),
        dropped_yepd_qc=sorted(yepd),
    )


@register_dataset
class FattyAcidSmith2006Dataset(ExperimentDataset):
    """Smith 2006 fatty-acid clear-zone env x geno -> ordinal response dataset."""

    def __init__(
        self,
        root: str = "data/torchcell/env_chemgen_smith2006",
        io_workers: int = 0,
        genome: SCerevisiaeGenome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset; a genome is REQUIRED for systematic-name resolution."""
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
        """The mirrored Supplementary Table 1 .xls (per-strain ordinal score table)."""
        return [XLS_FILENAME]

    def download(self) -> None:
        """Link the manifest-listed mirror file into ``raw/`` and verify its sha256.

        The mirror plus its recorded sha256 is canonical; the Europe PMC URL is retrieval
        metadata that ``deposit_raw_mirror`` records, never a live build dependency.
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        expected = manifest_sha256(manifest, XLS_REL)
        src = raw_mirror_dir(data_root) / XLS_REL
        if not src.exists():
            raise RuntimeError(f"required raw artifact missing from mirror: {src}")
        got = _sha256(src)
        if got != expected:
            raise RuntimeError(
                f"{XLS_FILENAME} sha256 mismatch: got {got}, expected {expected}"
            )
        os.makedirs(self.raw_dir, exist_ok=True)
        dest = osp.join(self.raw_dir, XLS_FILENAME)
        if not osp.exists(dest):
            os.symlink(src, dest)
        log.info("Smith 2006 raw table linked into %s (sha256 verified)", self.raw_dir)

    def _read_table(self) -> pd.DataFrame:
        """Read the legacy BIFF .xls Supplementary Table 1 (header on 0-based row 23)."""
        return pd.read_excel(
            osp.join(self.raw_dir, XLS_FILENAME), engine="xlrd", header=HEADER_ROW
        )

    def _environment(self, spec: dict[str, Any]) -> Environment:
        """The plate: the shared medium object plus its typed carbon-source edit."""
        return Environment(
            media=spec["media"],
            temperature=Temperature(value=TEMPERATURE_C.value),
            perturbations=[
                EnvironmentPhysicalPerturbation(
                    factor=PhysicalFactor.carbon_source,
                    magnitude=Concentration(
                        value=spec["carbon_percent"], unit=ConcentrationUnit.percent_w_v
                    ),
                    agent=resolved_compound(spec["carbon_source"]),
                )
            ],
            aerobicity=AEROBICITY.value,
            duration_hours=DURATION_HOURS.value,
        )

    def _reference(
        self, spec: dict[str, Any], environment: Environment
    ) -> EnvironmentResponseExperimentReference:
        """Parental BY4742 baseline: the wild-type call on this readout.

        The numeric ``environment_response`` is left None. The scale is an absolute
        visual ordinal (1-4), not a difference from the parent, so the wild type's value
        on it is 3 and asserting 0 would be false; the baseline is carried as a category.
        """
        return EnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain="BY4742"
            ),
            environment_reference=environment.model_copy(),
            phenotype_reference=EnvironmentResponsePhenotype(
                measurement_type=MeasurementType.ordinal,
                assay_type=spec["assay_type"],
                category=REFERENCE_CATEGORY,
                category_label=REFERENCE_CATEGORY_LABEL,
                n_samples=N_REPLICATES.value,
                sample_unit=SampleUnit.biological_replicate,
                units=spec["units"],
            ),
        )

    def _phenotype(
        self, spec: dict[str, Any], score: float
    ) -> EnvironmentResponsePhenotype:
        """One ordinal score with its typed call and this screen's own word for it."""
        mapped = spec["category_map"].get(score)
        if mapped is None:
            raise RuntimeError(
                f"unmapped ordinal score {score!r} in column {spec['column']!r}"
            )
        category, label = mapped
        return EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.ordinal,
            assay_type=spec["assay_type"],
            environment_response=score,
            category=category,
            category_label=label,
            n_samples=N_REPLICATES.value,
            sample_unit=SampleUnit.biological_replicate,
            units=spec["units"],
        )

    @post_process
    def process(self) -> None:
        """Parse the ordinal score table into per-(strain, condition) records; write LMDB."""
        if self.genome is None:
            raise RuntimeError(
                "FattyAcidSmith2006Dataset requires a genome for systematic-name "
                "resolution; inject SCerevisiaeGenome(...)"
            )
        df = self._read_table()
        strains = resolve_strains(df, self.genome)
        n_conditions = len(CONDITION_SPECS)
        source_records = len(df) * n_conditions
        publication = Publication(
            pubmed_id=PMID,
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PMID}/",
            doi=DOI,
            doi_url=f"https://doi.org/{DOI}",
        )

        prepared: list[dict[str, Any]] = [
            {
                "spec": spec,
                "environment": self._environment(spec),
                "reference": self._reference(spec, self._environment(spec)),
            }
            for spec in CONDITION_SPECS
        ]

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        n_blank_cells = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for orf, common, row in tqdm(
                zip(strains.orf, strains.common, strains.row_index, strict=True),
                total=len(strains.orf),
                desc="smith2006",
            ):
                record = df.iloc[row]
                genotype = Genotype(
                    perturbations=[
                        KanMxDeletionPerturbation(
                            systematic_gene_name=orf, perturbed_gene_name=common
                        )
                    ]
                )
                for item in prepared:
                    spec = cast(dict[str, Any], item["spec"])
                    raw = record[spec["column"]]
                    if pd.isna(raw):
                        n_blank_cells += 1
                        continue
                    experiment = EnvironmentResponseExperiment(
                        dataset_name=self.name,
                        genotype=genotype,
                        environment=item["environment"],
                        phenotype=self._phenotype(spec, float(raw)),
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

        rules = [
            DropRule(
                rule="strain_failed_the_yepd_growth_control",
                scope="strain",
                description=(
                    "column 7 of the released table flags the strain as NG (no growth), "
                    "LG (low growth) or NG/CONT on the YEPD plate the whole deletion set "
                    "was pinned on before replication onto the fatty-acid plates. A clear "
                    "zone needs a growing patch, so the score measures the growth-control "
                    "failure rather than beta-oxidation: among NG strains 98.1/98.9/96.2 "
                    "percent take the lowest oleate/myristate/acetate value against "
                    "2.0/2.2/0.3 percent of unflagged strains. Nothing in "
                    "EnvironmentResponsePhenotype can carry the flag, so the strain is "
                    "dropped rather than served unflagged"
                ),
                n_records=len(strains.dropped_yepd_qc) * n_conditions,
                items=strains.dropped_yepd_qc,
            ),
            DropRule(
                rule="systematic_name_does_not_resolve_to_a_current_orf",
                scope="strain",
                description=(
                    "the 2005-era systematic name is neither a current R64 gene id nor a "
                    "genome alias of one, so no gene entity exists to key the record to"
                ),
                n_records=len(strains.dropped_unresolved) * n_conditions,
                items=strains.dropped_unresolved,
            ),
            DropRule(
                rule="alias_resolution_collides_with_a_directly_present_orf",
                scope="strain",
                description=(
                    "the old name aliases to an ORF the SAME release also carries "
                    "directly (a dubious ORF absorbed into a verified neighbour, e.g. "
                    "YOR240W -> YOR239W present as ABP140); forcing the alias would store "
                    "a distinct dubious-ORF strain mislabelled as the neighbour's deletion"
                ),
                n_records=len(strains.dropped_alias_collision) * n_conditions,
                items=strains.dropped_alias_collision,
            ),
            DropRule(
                rule="second_row_for_an_already_resolved_orf",
                scope="strain",
                description=(
                    "a later row resolves to an ORF an earlier row already claimed; the "
                    "release's own legend says duplicated strains were already collapsed, "
                    "so a residual duplicate cannot be told apart from the kept strain"
                ),
                n_records=len(strains.dropped_duplicate) * n_conditions,
                items=strains.dropped_duplicate,
            ),
            DropRule(
                rule="condition_cell_is_blank",
                scope="cell",
                description=(
                    "the strain has no score in this condition's column (the release "
                    "carries all three scores for every row, so this is 0 today)"
                ),
                n_records=n_blank_cells,
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
            "Smith2006: wrote %d records from %d strains (dropped %d YEPD-QC, %d "
            "unresolved, %d alias-collision, %d duplicate strains)",
            idx,
            len(strains.orf),
            len(strains.dropped_yepd_qc),
            len(strains.dropped_unresolved),
            len(strains.dropped_alias_collision),
            len(strains.dropped_duplicate),
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
    root = osp.join(data_root, "data/torchcell/env_chemgen_smith2006")
    dataset = FattyAcidSmith2006Dataset(root=root, genome=genome)
    print(f"len = {len(dataset)}")
    print(dataset[0])
    print(
        json.dumps(
            json.loads(
                Path(osp.join(root, "preprocess/dropped_records.json")).read_text()
            )["rules"],
            indent=2,
        )[:2000]
    )


if __name__ == "__main__":
    main()
