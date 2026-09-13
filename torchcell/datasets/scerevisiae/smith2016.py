# torchcell/datasets/scerevisiae/smith2016
# [[torchcell.datasets.scerevisiae.smith2016]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/smith2016
# Test file: tests/torchcell/datasets/scerevisiae/test_smith2016.py
"""Smith 2016 quantitative CRISPRi chemical-genetic screen (per-guide, per-drug fitness).

Smith JD et al. 2016 (Genome Biol 17:45, doi:10.1186/s13059-016-0900-9;
citation_key ``smithQuantitativeCRISPRInterference2016``) built a regulatable dCas9-Mxi1
CRISPRi platform in S. cerevisiae (BY4741, one pRS416-derived CEN/URA3 plasmid carrying
TetR, an ATc-inducible RPR1-TetO gRNA locus and constitutive dCas9-Mxi1). gRNA tiling
libraries against 20 drug-partner genes were grown as POOLS in competitive-growth assays,
+/- ATc and +/- a small-molecule inhibitor, and each guide's abundance was tracked by
amplicon sequencing.

RECORD = one (pool x guide x drug-condition) ``EnvironmentResponseExperiment``:

- GENOTYPE: a ``CrisprInterferencePerturbation`` whose ``crispr`` construct carries the
  effector ``dCas9-Mxi1``, the 18/20 nt ``Specificity_sequence`` spacer (joined from
  Additional file 4 by guide name) and the screen ``library_pool``. The spacer keeps
  sibling guides of one gene distinct; the pool keeps the SAME spacer's two independent
  pool measurements distinct. The common name is the GENOME's own standard name for the
  resolved ORF, so one gene carries one spelling across datasets.
- ENVIRONMENT: the shared ``media.SC_URA`` object (the paper's "SCM-Ura"; Smith releases
  no SCM-Ura recipe, so the shipped library gaps are the honest state) carrying the row's
  drug at its released concentration as a ``SmallMoleculePerturbation``, 30 C, aerobic,
  ``duration_generations = 20``.
- PHENOTYPE: ``measurement_type=log2_ratio``, ``assay_type=pooled_competitive_growth_barcode``,
  ``environment_response = A`` (the ATc-induced fold change), uncertainty = the released
  ``var(A)`` stored VERBATIM as ``UncertaintyType.variance``. Reference = the uninduced
  (-ATc) baseline in the same drug environment, A = 0.

SOURCED VALUES (module-level ``SourcedValue``s anchored to the sha256 of
``smithQuantitativeCRISPRInterference2016/paper.md``):

- ``duration_generations = 20.0``: "This amounted to approximately 20 culture doublings
  from the beginning of the experiment." Exposure duration is part of environment
  identity, and the previous build left it None. ``duration_hours`` is a typed
  ``ProvenanceGap``: the cultures were held in log phase by repeated dilution and no wall
  time is reported.
- ``assay_type``: "the short specificity-determining regions of gRNAs (that is, the
  sequence complementary to the target) can act as unique identifiers of individual
  strains".
- the 1% DMSO vehicle-control dose, from the Methods sentence that also gives the eight
  and three replicate experiments for the tiling pools and for 20 uM fluconazole.

``n_samples = 1`` is KEPT and is a deliberate, documented choice rather than an oversight.
The released ``var(A)`` is the variance of the SINGLE released A estimate: for the 1% DMSO
and 20 uM fluconazole conditions the paper has already inverse-variance-combined the eight
and three replicate experiments into that one estimate ("We combined the R replicates
... to obtain the variance"), so ``derive_se`` must divide by 1, not by 8 or 3. Recording
the replicate count instead would silently shrink the stored SE by sqrt(8). The replicate
structure is recorded here, in ``REPLICATE_STRUCTURE``, rather than in a field whose
arithmetic it would corrupt.

ATc is NOT asserted as an environment dose. The induction is what turns the CRISPRi
genotype on, but the pooled-screen Methods say only "grown +/- a drug listed in Additional
file 8, and +/- ATc"; the single released number, 250 ng/mL, appears in the qPCR section
and in Additional file 5's sample-metadata header, neither of which is the pooled screen.
The induced-vs-uninduced definition rides in the phenotype ``units`` string and in the
uninduced reference. No solvent is asserted either: "Drugs were dissolved in DMSO" sits in
the individual-strain growth-assay section, and the pooled section does not restate it, so
the 1% DMSO control condition is served as a compound in its own right and the other
drugs carry no ``Solvent``. Both are flagged for review rather than guessed.

The screening doses were set by a rule the SI states ("Concentration(s) used for library
screening. These concentrations were selected to inhibit growth of a wild-type strain by
~20%", Additional file 8 ReadMe, sha256 a22ef00ace51...), but ``DoseBasis`` has no IC20
member and adding one changes a class in all 36 served dataset closures, so the rule is
recorded here and in the dendron note instead of in ``Concentration.basis``.

RECORDS DROPPED (rule + counts in ``preprocess/dropped_records.json``): 7,410 of 14,463
(51.2%) whose drug is a vendor catalog code -- 1181-0519, 4130-1276 and 0KPI-0099
(ChemDiv), 9121982, 9125678, 7312221, CBF-666774, 6630449 and 9150499 (ChemBridge),
ST016598 (TimTec). The primary released no structure, SMILES or CAS for any of them, so
``resolve_compound_identity`` returns ``PROPRIETARY`` with no InChIKey, ChEBI id or CID
and the compound entity cannot be encoded or joined. Per the Hoepfner precedent those
records are dropped, not served under a bare name. Kept: 7,053. The tenth public compound,
``NSC-180973``, resolves through the pinned identity table's synonym for tamoxifen -- the
alias the paper's own Additional file 8 releases -- so its 456 records are kept.

DUPLICATE-BY-POOL: 272 (guide, drug, concentration) triples appear twice, always the pool
pair (``broad_tiling``, ``gene_tiling_20bp``) sharing a 20 bp perfect-match spacer. Fitness
is median-centred WITHIN a pool, so the two A values differ by mean 0.95 and up to 7.8 log2
units: they are two independent pooled measurements, not replicates to average and not a
duplicate to drop. ``library_pool`` therefore joins the strain identity.

DATA SOURCE: Additional file 10 ``13059_2016_900_MOESM10_ESM.xlsx`` sheet 'Fitness and
Effect Data' (14,463 rows) and Additional file 4 ``13059_2016_900_MOESM4_ESM.xlsx`` sheet
'gRNAs' (the guide -> spacer join), both deposited in the raw mirror
``$DATA_ROOT/torchcell-raw/smithQuantitativeCRISPRInterference2016/`` with a
``manifest.json`` recording their Springer ESM URLs (re-retrieved and sha256-verified
2026-09-12) and hashes.
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
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.compound_identity import (
    resolve_compound_identity,
    resolved_compound,
)
from torchcell.datamodels.media import SC_URA
from torchcell.datamodels.schema import (
    AssayType,
    Concentration,
    ConcentrationUnit,
    CrisprConstruct,
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
    ROLE_SI_DATA,
    ArtifactRecord,
    Manifest,
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

DOI = "10.1186/s13059-016-0900-9"

CITATION_KEY = "smithQuantitativeCRISPRInterference2016"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

_ESM_BASE = (
    "https://static-content.springer.com/esm/art%3A10.1186%2Fs13059-016-0900-9/"
    "MediaObjects/"
)
EFFECT_FILENAME = "13059_2016_900_MOESM10_ESM.xlsx"
EFFECT_REL = f"si/si_data/{EFFECT_FILENAME}"
EFFECT_SHEET = "Fitness and Effect Data"
EFFECT_SHA256 = "02962e51e492b0505e8595fc1c80fab5fca8a8c8f05e05969dbff18ddff71cd0"
GUIDE_FILENAME = "13059_2016_900_MOESM4_ESM.xlsx"
GUIDE_REL = f"si/si_data/{GUIDE_FILENAME}"
GUIDE_SHEET = "gRNAs"
GUIDE_SHA256 = "e5eb4e3c7856782e36edff5ef55e680cb43fa8e9944bf8e40d7cb7b4f376c1e5"
SI_RETRIEVED_AT = "2026-09-12"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "346be6968eced82706cffb163a76dfc5adf381e602bd11006fea608436ca7b2f"

#: Drug label of the vehicle-only control condition (its own served compound).
DMSO = "DMSO"


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
            page="Methods, 'Growth assays' / 'Pooled competitive growth assays' / "
            "'ATc-induced fold change'",
        ),
    )


_POOLED_QUOTE = (
    "Briefly, $7 0 0 ~ \\mu \\mathrm { L }$ yeast cultures were grown $+ / -$ a drug "
    "listed in Additional file 8, and $+ / -$ ATc) in 48 well plates at $3 0 ~ ^ { \\circ "
    "} \\mathrm { C }$ with orbital shaking in Infinite plate readers (Tecan)."
)

TEMPERATURE_C = _paper(30.0, _POOLED_QUOTE)
AEROBICITY = _paper(
    "aerobic",
    _POOLED_QUOTE,
    note="48-well plates with orbital shaking in a plate reader; no anaerobic handling "
    "is described",
)
DURATION_GENERATIONS = _paper(
    20.0,
    "This amounted to approximately 20 culture doublings from the beginning of the "
    "experiment.",
    note="a competitive-growth screen doses exposure in doublings, not hours; the "
    "cultures were held in log phase by repeated dilution, so no wall time is reported "
    "and duration_hours is a typed ProvenanceGap",
)
ASSAY = _paper(
    AssayType.pooled_competitive_growth_barcode,
    "the short specificity-determining regions of gRNAs (that is, the sequence "
    "complementary to the target) can act as unique identifiers of individual strains",
)
EFFECTOR = _paper(
    "dCas9-Mxi1",
    "catalytically inactive Streptococcus pyogenes Cas9 (dCas9) to which the Mxi1 "
    "transcriptional repressor was fused at the C-terminus",
)
MEDIUM = _paper(
    "SCM-Ura",
    "all colonies were washed off plates with SCM-Ura liquid media",
    note="served as the shared media.SC_URA object (synthetic complete minus uracil, "
    "URA3 plasmid selection); Smith releases no SCM-Ura recipe, so SC_URA's own "
    "component gaps are the honest state",
)
DMSO_PERCENT = _paper(
    1.0,
    "For the control condition $( 1 ~ \\% ~ \\mathrm { D M S O } )$ , we had eight "
    "replicate experiments for the tiling pools, and three replicate experiments for $2 "
    "0 ~ \\mu \\mathrm { M }$ fluconazole.",
    note="the released Concentration cell for the DMSO control is the v/v fraction 0.01; "
    "the Methods give it as 1%",
)
REPLICATE_STRUCTURE = _paper(
    {"1% DMSO": 8, "20 uM fluconazole": 3},
    "We combined the $R$ replicates $k _ { I } , . . . , k _ { R }$ in a natural way to "
    "obtain the variance",
    note="these replicate experiments are ALREADY inverse-variance-combined into the "
    "single released (A, var(A)) pair, so n_samples stays 1 and derive_se divides by 1. "
    "Recording 8 or 3 here would shrink the stored SE by sqrt(8) or sqrt(3) against a "
    "variance that is already the combined estimate's",
)
UPSTREAM_FILTER = _paper(
    "30 reads in the -ATc control",
    "gRNAs with fewer than 30 reads following growth in the minus ATc control condition "
    "were excluded from this analysis, as their effect size estimates had large variance "
    "across conditions.",
    note="applied by the primary before release; this loader adds no count filter",
)
DURATION_HOURS_GAP = ProvenanceGap(
    field="duration_hours",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="the pooled cultures were maintained in log phase by repeated dilution over "
    "approximately 20 doublings; no wall-clock exposure time is reported, so the "
    "duration is carried in generations only",
)

UNITS = (
    "ATc-induced fold change A = f(+ATc) - f(-ATc): the difference in log2 "
    "median-centred guide read-count fitness between the induced (+ATc, dCas9-Mxi1 "
    "CRISPRi ON) and uninduced (-ATc) pooled cultures in this drug condition; negative = "
    "CRISPRi repression of the target causes a growth/fitness defect (drug-sensitivity)"
)

_CONC_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*(M|mM|uM|nM)\s*$")
_UNIT_MAP = {
    "M": ConcentrationUnit.molar,
    "mM": ConcentrationUnit.millimolar,
    "uM": ConcentrationUnit.micromolar,
    "nM": ConcentrationUnit.nanomolar,
}


def parse_concentration(drug: str, raw: Any) -> Concentration:
    """Parse the released ``Concentration`` cell into a typed dose (fail loud, never guess)."""
    if drug == DMSO:
        return Concentration(
            value=DMSO_PERCENT.value, unit=ConcentrationUnit.percent_v_v
        )
    match = _CONC_RE.match(str(raw))
    if match is None:
        raise ValueError(f"unparseable concentration {raw!r} for drug {drug!r}")
    return Concentration(value=float(match.group(1)), unit=_UNIT_MAP[match.group(2)])


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror + build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/smithQuantitativeCRISPRInterference2016``."""
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
    effect_path: str | Path,
    guide_path: str | Path,
    retrieved_at: str = SI_RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror from the two retrieved SI workbooks plus its ``manifest.json``.

    Idempotent by sha256. ``static-content.springer.com`` is a directly scriptable CDN, so
    the recorded ``springer_esm`` retrieval re-runs as-is; both URLs were re-retrieved on
    2026-09-12 and reproduced the pinned hashes exactly.
    """
    root = raw_mirror_dir(data_root)
    files: list[ArtifactRecord] = []
    for source, relpath, filename, expected in (
        (effect_path, EFFECT_REL, EFFECT_FILENAME, EFFECT_SHA256),
        (guide_path, GUIDE_REL, GUIDE_FILENAME, GUIDE_SHA256),
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
        url = f"{_ESM_BASE}{filename}"
        files.append(
            ArtifactRecord(
                path=relpath,
                role=ROLE_SI_DATA,
                bytes=dest.stat().st_size,
                sha256=expected,
                source=url,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.springer_esm,
                    source_url=url,
                    retriever="torchcell.literature.retrieve.springer_esm",
                    params={"url": url},
                    sha256=expected,
                    retrieved_at=retrieved_at,
                ),
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=(
            "Quantitative CRISPR interference screens in yeast identify chemical-genetic "
            "interactions and new rules for guide RNA design"
        ),
        files=files,
        si_data_sources=[
            f"{_ESM_BASE}{EFFECT_FILENAME}",
            f"{_ESM_BASE}{GUIDE_FILENAME}",
        ],
        si_expected=[
            "Additional file 10 (ATc effects A, drug effects D, no-drug ATc effects A0 "
            "for each gRNA in every tested condition)",
            "Additional file 4 (the five gRNA libraries and their Specificity_sequence "
            "spacers)",
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
    scope: Literal["compound", "guide", "row"]
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
class CrispriChemgenSmith2016Dataset(ExperimentDataset):
    """Smith 2016 per-guide dCas9-Mxi1 CRISPRi chemical-genetic env x geno dataset."""

    def __init__(
        self,
        root: str = "data/torchcell/crispri_chemgen_smith2016",
        io_workers: int = 0,
        genome: SCerevisiaeGenome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; a genome is REQUIRED for target-ORF -> current-R64 resolution."""
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
        """The two sha256-pinned SI workbooks (effect table + guide->spacer map)."""
        return [EFFECT_FILENAME, GUIDE_FILENAME]

    def download(self) -> None:
        """Link the manifest-listed mirror files into ``raw/`` and verify their sha256."""
        data_root = _data_root()
        manifest = load_manifest(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        for relpath, filename in (
            (EFFECT_REL, EFFECT_FILENAME),
            (GUIDE_REL, GUIDE_FILENAME),
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
        log.info(
            "Smith 2016 SI workbooks linked into %s (sha256 verified)", self.raw_dir
        )

    def _resolver(self) -> Callable[[str], str | None]:
        """Target ORF -> current-R64 ORF via the SHARED genome resolver."""
        if self.genome is None:
            raise RuntimeError(
                "CrispriChemgenSmith2016Dataset requires a genome; "
                "inject SCerevisiaeGenome(...)"
            )
        genome = self.genome
        gene_set = {gene.upper() for gene in genome.gene_set}

        def resolve(name: str) -> str | None:
            resolution = genome.resolve_gene_name(str(name).strip())
            if resolution.is_current_gene and resolution.systematic_name in gene_set:
                return resolution.systematic_name
            return None

        return resolve

    def _spacer_map(self) -> dict[str, str]:
        """Guide name -> ``Specificity_sequence`` spacer (unique per name; 0 conflicts)."""
        guides = pd.read_excel(
            osp.join(self.raw_dir, GUIDE_FILENAME), sheet_name=GUIDE_SHEET
        )
        return {
            str(name): str(spacer)
            for name, spacer in zip(
                guides["Guide_name"], guides["Specificity_sequence"], strict=True
            )
        }

    def _environment(self, drug: str, conc_raw: Any) -> Environment:
        """SC-Ura liquid carrying the row's drug at its released dose, 30 C, 20 doublings."""
        return Environment(
            media=SC_URA,
            temperature=Temperature(value=TEMPERATURE_C.value),
            perturbations=[
                SmallMoleculePerturbation(
                    compound=resolved_compound(drug),
                    concentration=parse_concentration(drug, conc_raw),
                )
            ],
            aerobicity=AEROBICITY.value,
            duration_generations=DURATION_GENERATIONS.value,
            provenance_gaps=[DURATION_HOURS_GAP],
        )

    def _reference(
        self, environment: Environment
    ) -> EnvironmentResponseExperimentReference:
        """Uninduced (-ATc) baseline: no ATc-induced fold change -> A = 0 (log2 ratio 0)."""
        return EnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain="BY4741"
            ),
            environment_reference=environment.model_copy(),
            phenotype_reference=EnvironmentResponsePhenotype(
                measurement_type=MeasurementType.log2_ratio,
                assay_type=ASSAY.value,
                environment_response=0.0,
                units=UNITS,
            ),
        )

    @post_process
    def process(self) -> None:
        """Build one env x geno -> A record per (pool, guide, drug, condition); write LMDB."""
        df = pd.read_excel(
            osp.join(self.raw_dir, EFFECT_FILENAME), sheet_name=EFFECT_SHEET
        )
        df["A"] = pd.to_numeric(df["A"], errors="coerce")
        df["var(A)"] = pd.to_numeric(df["var(A)"], errors="coerce")
        resolve = self._resolver()
        assert self.genome is not None
        canonical = canonical_common_names(self.genome)
        spacer_map = self._spacer_map()
        publication = Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")

        source_records = len(df)
        drugs = sorted({str(v) for v in df["Drug"]})
        unidentified = sorted(
            drug
            for drug in drugs
            if not resolve_compound_identity(name=drug).identified
        )
        unidentified_set = set(unidentified)

        env_cache: dict[tuple[str, str], dict[str, Any]] = {}

        def prepared(drug: str, conc_raw: Any) -> dict[str, Any]:
            key = (drug, str(conc_raw))
            if key not in env_cache:
                environment = self._environment(drug, conc_raw)
                env_cache[key] = {
                    "environment": environment,
                    "reference": self._reference(environment),
                }
            return env_cache[key]

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        n_nan = n_unresolved = n_no_spacer = n_unidentified = 0
        unresolved_genes: set[str] = set()
        no_spacer_guides: set[str] = set()
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="smith2016"):
                effect, variance = row["A"], row["var(A)"]
                if pd.isna(effect) or pd.isna(variance):
                    n_nan += 1
                    continue
                drug = str(row["Drug"])
                if drug in unidentified_set:
                    n_unidentified += 1
                    continue
                orf = resolve(row["ORF"])
                if orf is None:
                    n_unresolved += 1
                    unresolved_genes.add(str(row["ORF"]))
                    continue
                guide_name = str(row["Guide"])
                spacer = spacer_map.get(guide_name)
                if spacer is None:
                    n_no_spacer += 1
                    no_spacer_guides.add(guide_name)
                    continue
                item = prepared(drug, row["Concentration"])
                experiment = EnvironmentResponseExperiment(
                    dataset_name=self.name,
                    genotype=Genotype(
                        perturbations=[
                            CrisprInterferencePerturbation(
                                systematic_gene_name=orf,
                                perturbed_gene_name=canonical.get(orf, orf),
                                crispr=CrisprConstruct(
                                    effector=EFFECTOR.value,
                                    guide_sequence=spacer,
                                    n_guides=1,
                                    library_pool=str(row["#Pool"]),
                                ),
                            )
                        ]
                    ),
                    environment=item["environment"],
                    phenotype=EnvironmentResponsePhenotype(
                        measurement_type=MeasurementType.log2_ratio,
                        assay_type=ASSAY.value,
                        environment_response=float(effect),
                        environment_response_uncertainty=float(variance),
                        environment_response_uncertainty_type=UncertaintyType.variance,
                        n_samples=1,
                        sample_unit=SampleUnit.pooled,
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

        rules = [
            DropRule(
                rule="drug_is_a_vendor_catalog_code_with_no_released_structure",
                scope="compound",
                description=(
                    "the Drug label is a ChemDiv / ChemBridge / TimTec catalog id and the "
                    "primary released no structure, SMILES or CAS for it, so "
                    "resolve_compound_identity returns PROPRIETARY with no InChIKey, ChEBI "
                    "id or PubChem CID and the compound entity cannot be encoded or joined"
                ),
                n_records=n_unidentified,
                items=unidentified,
            ),
            DropRule(
                rule="row_has_no_A_or_var_A",
                scope="row",
                description=(
                    "the released row carries no ATc-induced fold change or no variance "
                    "for it (0 today; every row of Additional file 10 is complete)"
                ),
                n_records=n_nan,
                items=[],
            ),
            DropRule(
                rule="target_orf_is_not_a_current_genome_gene",
                scope="row",
                description=(
                    "the target ORF does not resolve to a gene of the current R64 "
                    "annotation (0 today; all 20 targets are current)"
                ),
                n_records=n_unresolved,
                items=sorted(unresolved_genes),
            ),
            DropRule(
                rule="guide_has_no_released_spacer",
                scope="guide",
                description=(
                    "the guide name is absent from Additional file 4, so no spacer exists "
                    "to carry the strain identity (0 today; all 977 screened guides join)"
                ),
                n_records=n_no_spacer,
                items=sorted(no_spacer_guides),
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
            "Smith2016: wrote %d of %d rows; dropped %d on %d unidentified vendor-code "
            "drugs, %d with no A/var(A), %d unresolved targets, %d guides with no spacer",
            idx,
            source_records,
            n_unidentified,
            len(unidentified),
            n_nan,
            n_unresolved,
            n_no_spacer,
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
    root = osp.join(data_root, "data/torchcell/crispri_chemgen_smith2016")
    dataset = CrispriChemgenSmith2016Dataset(root=root, genome=genome)
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
