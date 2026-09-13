# torchcell/datasets/scerevisiae/wildenhain2015
# [[torchcell.datasets.scerevisiae.wildenhain2015]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/wildenhain2015
# Test file: tests/torchcell/datasets/scerevisiae/test_wildenhain2015.py
"""Wildenhain 2015 chemical-genetic matrix (CGM): env x geno -> z-score response.

Wildenhain et al. 2015 (Cell Systems, doi:10.1016/j.cels.2015.12.003) screened a panel of
haploid non-essential S. cerevisiae deletion strains (Euroscarf collection, isogenic to
BY4741) against thousands of compounds from four chemical libraries at a single 20 uM
screening concentration, reading out growth inhibition as a normalized-OD600 Z-score. This
loader builds ONLY the chemical-genetic matrix (strain x compound -> z-score); the separate
128x128 cryptagen chemical-chemical synergy layer is out of scope.

Every screened strain is one haploid gene replacement in BY4741 (NOT a HIP/HOP engineered
heterozygote), so there is no engineered-dosage complication. The paper's "195 sentinel
strains" is a dataset-level selection note whose roster is in the unmirrored Table S3; the
RELEASED data covers 242 distinct systematic ORFs, all current R64 genes, and all 242 are
served.

DATA SOURCE. PubChem BioAssay AID 1159580 (the paper's ACCESSION NUMBERS section; the
chemgrid.org/cgm portal is an interactive PHP site with no bulk export). Two artifacts are
mirrored and sha256-pinned: the per-AID datapoint export ``1159580.csv.gz``, read out of the
byte-stable NCBI FTP range archive with the container's OWN sha256 asserted first
(``zip_member``), and the AID's PUG-REST description JSON, which is the only source for the
screening protocol's medium, temperature, incubation, replicate structure and column
definitions. Both are read from the raw mirror at build time; the URLs are retrieval
metadata, not live dependencies.

WHAT THE NUMBER IS. The stored response is the released per-screen ``z_score``, averaged
over the screens of a (strain, compound) cell exactly as the paper constructs the matrix
("Z scores were calculated and averaged for the replicate screens"). The z-score is
standardized WITHIN a screen ("Z-Score calculated based on kernel density distribution from
normalized average reads per screen"), so the reference baseline of 0 is that screen's own
normalized-growth center rather than a measured wild-type value, and the reference
environment is the compound's own environment. The released ``sym == 'wild type'`` rows are
strain measurements, not that baseline, and are not ingested (see the dendron note).

SCREEN COUNTING (measured, and it corrects the previous build). A (strain, compound) cell
recurs across the four libraries, and the export ALSO re-emits the same datapoint under two
gene-symbol spellings (``MDH1`` / ``mdh1``): of the 46,195 cells with more than one released
row, 33,483 contain byte-identical duplicates. Within a cell, two rows sharing a z_score
share every other data column too, and no two genuinely distinct datapoints of a cell share
a z_score, so the z string IS the datapoint key. After deduplication the screens-per-cell
histogram is {1: 412,368, 2: 14,579, 3: 1,617, 4: 7, 5: 2}. ``n_samples`` is therefore the
number of contributing SCREENS with ``sample_unit=screen`` (the independent unit; the two
OD reads inside a screen are the technical duplicate the z already averages), and the
uncertainty is the sample SD across those screens, or a typed ``ProvenanceGap`` for the
single-screen cells. The previous build's ``n_samples = 2 x rows`` counted OD reads of
re-exported duplicates.

RECORDS DROPPED (rule + count in ``preprocess/dropped_records.json``): cells whose compound
carries no structure identifier (the 5 SID-only compounds, which have no CID and no SMILES).
Everything else is served: all 5,173 CIDs resolve to an InChIKey and a canonical PubChem
name through the pinned compound-identity table.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import logging
import os
import os.path as osp
import re
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Literal

from pydantic import BaseModel
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import SC
from torchcell.datamodels.schema import (
    AssayType,
    BarcodedKanMxDeletionPerturbation,
    Compound,
    Concentration,
    ConcentrationUnit,
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
    ResponseCategory,
    SampleUnit,
    SmallMoleculePerturbation,
    Solvent,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.scerevisiae.gene_name_reconcile import default_genome
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
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

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "wildenhainPredictionSynergismChemicalGenetic2015"
PAPER_DOI = "10.1016/j.cels.2015.12.003"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

FTP_ZIP_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/CSV/Data/1159001_1160000.zip"
)
ZIP_MEMBER = "1159001_1160000/1159580.csv.gz"
#: sha256 of the FTP range archive itself, asserted BEFORE the member is read, so a
#: re-packed upstream archive fails loudly instead of yielding different member bytes.
CONTAINER_SHA256 = "d1fd5dc2bf7c526ad9845e0a14ae9981256fb820aaf4228b48a3ba0724ee59b0"
DATA_FILENAME = "1159580.csv.gz"
DATA_SHA256 = "c461c679b63ac56045cef0f03ed9bcbb8e7f9c12146f1fc7cc8ac0c113188d64"
DATA_REL = f"data/{DATA_FILENAME}"

AID_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/1159580/description/JSON"
AID_FILENAME = "aid_1159580_description.json"
AID_SHA256 = "23c5f8c56af94786cfe8e22c93fdde0b719ca2165975305944557ab39087b0e4"
AID_REL = f"data/{AID_FILENAME}"
RETRIEVED_AT = "2026-09-13"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "f46409eb8f23412c9c1015d0f8f5bb581bfddfe2796d319d407585e23c757ac2"


def _aid(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """Bind a value to a verbatim quote in the sha256-pinned AID 1159580 description."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=AID_REL,
            citation_key=CITATION_KEY,
            sha256=AID_SHA256,
            method="PubChem PUG-REST assay description JSON (torchcell-raw mirror)",
            page="PC_AssayContainer[0].assay.descr.protocol",
            retrieved=RETRIEVED_AT,
        ),
    )


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
            page="RESULTS 'Generation of a Chemical-Genetic Matrix' / EXPERIMENTAL PROCEDURES",
        ),
    )


# --------------------------------------------------------------------------- #
# Sourced environment + phenotype constants
# --------------------------------------------------------------------------- #
MEDIUM = _aid(
    "SC",
    "All strains were grown and screened in synthetic complete (SC) medium with 2% "
    "glucose.",
    note="the paper's own medium sentence has 'All fungal species' as its subject, "
    "which in context follows the list of fungal PATHOGEN isolates; the AID protocol "
    "says 'All strains', which is unambiguous for the S. cerevisiae screens. The shared "
    "SC library object carries D-glucose at 20 g/L, i.e. the same 2% w/v",
)
TEMPERATURE_C = _aid(
    30.0,
    "Plates were incubated at 30 C without shaking for approximately 18 h or until "
    "culture saturation was achieved for the solvent controls.",
    note="the PAPER states no incubation temperature anywhere (its only temperature is "
    "37 C, for HeLa/HEK cells), so this value exists only because the AID description "
    "is now mirrored and hashed; it is sourced, not a default",
)
DURATION_HOURS = _aid(
    18.0,
    "Plates were incubated at 30 C without shaking for approximately 18 h or until "
    "culture saturation was achieved for the solvent controls.",
)
SCREEN_CONCENTRATION_UM = _paper(
    20.0,
    "We carried out over 600 growth-based screens in duplicate at a compound "
    "concentration of $2 0 \\mu \\mathsf { M }$ .",
    note="the AID protocol states the same dose as a preparation: 'Strains were seeded "
    "at 50,000 cells per well in a volume of 100 L in 96 well plates followed by "
    "addition of 2 L of 1 mM compound stock for final concentration of 20 M.'",
)
SOLVENT = _aid(
    "DMSO",
    "DMSO solvent only controls and 10 uM cycloheximide positive controls were seeded "
    "in columns 1 and 12 of each assay plate.",
    note="the vehicle is sourced; its FINAL fraction in the well is not released (the "
    "stock's own solvent fraction is unstated), so Solvent.percent stays unset rather "
    "than being back-computed from the 2 uL into 100 uL dilution",
)
ASSAY = _aid(
    AssayType.liquid_od_growth,
    "Cultures were resuspended by shaking on the robotic platform prior to reading OD600 "
    "values on either Tecan M1000 or Tecan Sunrise plate readers.",
)
SCREEN_REPLICATION = _aid(
    "technical duplicate",
    "Screens were conducted in technical duplicate",
    note="the duplicate is the pair of OD READS inside one screen, which the z-score "
    "already averages ('normalized average reads per screen'); the independent unit of "
    "replication is therefore the SCREEN, which is what n_samples counts",
)
Z_SCORE_DEFINITION = _aid(
    MeasurementType.z_score,
    "Z-Score calculated based on kernel density distribution from normalized average "
    "reads per screen",
    note="standardized WITHIN a screen, so 0 is that screen's normalized-growth center; "
    "the companion statement is 'Z-factors for growth inhibition were calculated using "
    "the median and the interquartile range (IQR) by fitting a normal distribution with "
    "N(1,IQR) to the experimental data.'",
)
Z_AVERAGED = _paper(
    "mean over the replicate screens",
    "Z scores were calculated and averaged for the replicate screens.",
)
COLLECTION = _aid(
    "Euroscarf deletion collection",
    "All S. cerevisiae deletion strains were obtained from the Euroscarf deletion "
    "collection.",
    note="the paper says the same: 'S. cerevisiae deletion strains were obtained from "
    "the Euroscarf deletion set and are isogenic to BY4741 (Table S3).' Neither source "
    "names the deletion cassette's marker; kanMX is a property of the Euroscarf MATa "
    "collection sourced to Winzeler 1999 / Giaever 2002, which are not mirrored",
)
PARENT_STRAIN = _aid(
    "BY4741", "The wild type parental strain for this collection is BY 4741"
)
NON_REPLICATE_COLUMN = _aid(
    "non replicate",
    "test for non-replicates between first and second replicate",
    note="a screen whose two OD reads disagreed; counting SCREENS rather than reads is "
    "what keeps this flag from corrupting n_samples. The release also states 'Data "
    "points with high variation between replicates (> 3 MAD) were removed as "
    "inconsistent outliers.', so a retained flagged screen passed that filter",
)
BIOACTIVITY_COLUMN = _aid(
    "bioactivity",
    "sensitive if compound decreases fitness or resistant if compound increases fitness "
    "compared to negative control",
    note="together with the released PUBCHEM_ACTIVITY_OUTCOME this is the source of the "
    "ResponseCategory mapping below",
)

MEASUREMENT_UNITS = (
    "released PubChem AID 1159580 z_score of growth inhibition, standardized within a "
    "screen from the normalized OD600 average of a technical-duplicate read pair "
    "(20 uM compound in DMSO, SC + 2% glucose, 30 C, ~18 h); negative = growth "
    "inhibition, averaged over the contributing screens of the cell"
)

#: Released activity outcome + bioactivity -> the shared ResponseCategory axis. The two
#: columns' own definitions are the source (``BIOACTIVITY_COLUMN``): an Inactive datapoint
#: is one the screen could not distinguish from the negative control, an Active one is a
#: called hit whose DIRECTION the bioactivity column gives, and Inconclusive is PubChem's
#: own "no call" verdict -- which is ``not_determined``, never silently an Inactive.
OUTCOME_CATEGORY: dict[tuple[str, str], ResponseCategory] = {
    ("Inactive", ""): ResponseCategory.no_change,
    ("Active", "sensitive"): ResponseCategory.sensitive,
    ("Active", "resistant"): ResponseCategory.resistant,
    ("Inconclusive", "sensitive"): ResponseCategory.not_determined,
    ("Inconclusive", "resistant"): ResponseCategory.not_determined,
    ("Inconclusive", ""): ResponseCategory.not_determined,
}

#: Systematic ORF pattern; non-strain tokens (NA / NULL control rows) are dropped.
_SYSTEMATIC_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$")


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror + build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/wildenhainPredictionSynergismChemicalGenetic2015``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_relpaths() -> dict[str, str]:
    """Short name -> mirror-relative path for every file the loader reads."""
    return {DATA_FILENAME: DATA_REL, AID_FILENAME: AID_REL}


def deposit_raw_mirror(
    *,
    csv_path: str | Path,
    aid_path: str | Path,
    retrieved_at: str = RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror from already-retrieved files and its ``manifest.json``.

    Idempotent by sha256 (an existing file with the recorded hash is left alone, a
    differing one raises). The datapoint export's retrieval pins the FTP container FIRST
    and then reads one member out of it, so a re-packed upstream archive is detected
    rather than silently followed.
    """
    root = raw_mirror_dir(data_root)
    sources: dict[str, tuple[Path, str, RetrievalRecord]] = {
        DATA_REL: (
            Path(csv_path),
            DATA_SHA256,
            RetrievalRecord(
                method=RetrievalMethod.direct_url,
                source_url=FTP_ZIP_URL,
                retriever="torchcell.literature.retrieve.zip_member",
                params={
                    "url": FTP_ZIP_URL,
                    "member": ZIP_MEMBER,
                    "container_sha256": CONTAINER_SHA256,
                },
                sha256=DATA_SHA256,
                retrieved_at=retrieved_at,
            ),
        ),
        AID_REL: (
            Path(aid_path),
            AID_SHA256,
            RetrievalRecord(
                method=RetrievalMethod.pubchem_api,
                source_url=AID_URL,
                retriever="torchcell.literature.retrieve.direct_url",
                params={"url": AID_URL},
                sha256=AID_SHA256,
                retrieved_at=retrieved_at,
            ),
        ),
    }
    files: list[ArtifactRecord] = []
    for relpath, (src, expected, retrieval) in sources.items():
        got = _sha256(src)
        if got != expected:
            raise RuntimeError(f"{src} sha256 mismatch: got {got}, expected {expected}")
        dest = root / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != expected:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(src, dest)
        files.append(
            ArtifactRecord(
                path=relpath,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=expected,
                source=retrieval.source_url,
                retrieval=retrieval,
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title="Prediction of Synergism from Chemical-Genetic Interactions by Machine Learning",
        files=files,
        si_data_sources=[
            "https://pubchem.ncbi.nlm.nih.gov/bioassay/1159580",
            FTP_ZIP_URL,
            AID_URL,
        ],
        si_expected=[
            "Tables S1/S2 (the four compound libraries) and Table S3 (the 195 sentinel "
            "strains) -- cell.com supplementary files are not scriptable, so they are "
            "NOT mirrored and the 195-vs-242 strain split cannot be reconstructed"
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
# Retention bookkeeping + the collapsed matrix cell
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One retention rule, the records it removed, and the items it removed them for."""

    rule: str
    scope: Literal["compound", "library_row", "cell"]
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


class MatrixCell(BaseModel):
    """One (strain, compound) cell of the CGM, collapsed over its released screens."""

    orf: str
    identity: str
    pubchem_cid: int | None
    smiles: str | None
    #: z_score string -> (non-replicate flag, activity outcome, bioactivity), one entry
    #: per DISTINCT released datapoint (the z string is the datapoint key; see the
    #: module docstring for the measurement behind that).
    screens: dict[str, tuple[str, str, str]] = {}

    @property
    def z_values(self) -> list[float]:
        """The distinct per-screen z-scores contributing to this cell."""
        return [float(z) for z in self.screens]

    @property
    def n_screens(self) -> int:
        """Number of contributing screens (the independent unit of replication)."""
        return len(self.screens)

    @property
    def all_screens_non_replicating(self) -> bool:
        """Whether EVERY contributing screen's two OD reads failed to replicate."""
        return all(flag == "1" for flag, _, _ in self.screens.values())

    def category(self) -> tuple[ResponseCategory, str]:
        """The released curation verdict on the shared axis + its verbatim source words.

        A cell whose screens DISAGREE has no single released call, so it resolves to
        ``not_determined`` and the label keeps every word the release used.
        """
        outcomes = sorted({outcome for _, outcome, _ in self.screens.values()})
        activities = sorted({bio for _, _, bio in self.screens.values()})
        if len(outcomes) == 1 and len(activities) == 1:
            label = (
                outcomes[0] if not activities[0] else f"{outcomes[0]} / {activities[0]}"
            )
            return OUTCOME_CATEGORY[(outcomes[0], activities[0])], label
        words = [*outcomes, *[bio for bio in activities if bio]]
        return ResponseCategory.not_determined, " / ".join(words)


def _canonical_common_names(genome: SCerevisiaeGenome) -> dict[str, str]:
    """``systematic name -> the genome's own standard (common) name``.

    The release spells 16 ORFs two ways (``TOR1`` and ``Tor1``), which splits one
    perturbation into two graph nodes. Taking the spelling from the genome instead of
    from the source is what makes it one node, and identical across datasets. Only a
    standard name that resolves BACK to the gene is used.
    """
    canonical: dict[str, str] = {}
    for standard in genome.feature_index["standard_to_ids"]:
        resolution = genome.resolve_gene_name(standard)
        if resolution.is_current_gene and resolution.systematic_name is not None:
            canonical.setdefault(resolution.systematic_name, standard)
    return canonical


@register_dataset
class EnvChemgenWildenhain2015Dataset(ExperimentDataset):
    """Wildenhain 2015 chemical-genetic matrix: env x geno -> growth-inhibition z-score."""

    def __init__(
        self,
        root: str = "data/torchcell/env_chemgen_wildenhain2015",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (the genome is loaded lazily inside ``process``)."""
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
        """The datapoint export and the AID description the protocol is sourced from."""
        return [DATA_FILENAME, AID_FILENAME]

    def download(self) -> None:
        """Link the manifest-listed mirror files into ``raw/`` and verify each sha256.

        The mirror is canonical. The FTP container is 151 MB and its sha256 is recorded
        with the member's, so ``deposit_raw_mirror``'s retrieval re-runs the download and
        detects a re-packed archive; a build never depends on that URL being alive.
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        for name, relpath in raw_relpaths().items():
            src = raw_mirror_dir(data_root) / relpath
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            expected = manifest_sha256(manifest, relpath)
            got = _sha256(src)
            if got != expected:
                raise RuntimeError(
                    f"{name} sha256 mismatch: got {got}, expected {expected}"
                )
            dest = osp.join(self.raw_dir, name)
            if not osp.exists(dest):
                os.symlink(src, dest)
        log.info(
            "Wildenhain 2015 raw files linked into %s (sha256 verified)", self.raw_dir
        )

    # ---- source reading -------------------------------------------------------- #
    def _collapse_matrix(self) -> tuple[dict[tuple[str, str], MatrixCell], int, int]:
        """Read the datapoint CSV; collapse to one cell per (ORF, compound identity).

        Identity is the PubChem CID when present, else the SID. Repeated releases of the
        SAME datapoint (identical z) collapse; distinct screens accumulate. Returns the
        cells plus the strain-row and non-strain-row counts.
        """
        path = osp.join(self.raw_dir, DATA_FILENAME)
        cells: dict[tuple[str, str], MatrixCell] = {}
        n_rows = 0
        n_non_strain = 0
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            idx = {name: i for i, name in enumerate(header)}
            next(reader)  # RESULT_TYPE definition row
            for row in reader:
                if not row:
                    continue
                orf = row[idx["orf"]].strip()
                if not _SYSTEMATIC_RE.match(orf):
                    n_non_strain += 1
                    continue
                z_raw = row[idx["z_score"]].strip()
                if not z_raw:
                    continue
                n_rows += 1
                cid = row[idx["PUBCHEM_CID"]].strip()
                sid = row[idx["PUBCHEM_SID"]].strip()
                identity = f"CID {cid}" if cid else f"SID {sid}"
                key = (orf, identity)
                cell = cells.get(key)
                if cell is None:
                    cell = MatrixCell(
                        orf=orf,
                        identity=identity,
                        pubchem_cid=int(cid) if cid else None,
                        smiles=row[idx["PUBCHEM_EXT_DATASOURCE_SMILES"]].strip()
                        or None,
                    )
                    cells[key] = cell
                cell.screens[z_raw] = (
                    row[idx["non replicate"]].strip(),
                    row[idx["PUBCHEM_ACTIVITY_OUTCOME"]].strip(),
                    row[idx["bioactivity"]].strip(),
                )
        log.info(
            "Wildenhain2015: %d strain datapoints (%d non-strain control rows) -> "
            "%d (ORF, compound) cells",
            n_rows,
            n_non_strain,
            len(cells),
        )
        return cells, n_rows, n_non_strain

    # ---- environment / phenotype builders -------------------------------------- #
    def _compound(self, cell: MatrixCell) -> Compound:
        """The compound's canonical identity: the table's PubChem name + InChIKey."""
        return resolved_compound(
            cell.identity, pubchem_cid=cell.pubchem_cid, smiles=cell.smiles
        )

    def _environment(self, compound: Compound) -> Environment:
        """Aerobic SC (2% glucose) liquid culture carrying the compound at 20 uM in DMSO."""
        return Environment(
            media=SC,
            temperature=Temperature(value=TEMPERATURE_C.value),
            perturbations=[
                SmallMoleculePerturbation(
                    compound=compound,
                    concentration=Concentration(
                        value=SCREEN_CONCENTRATION_UM.value,
                        unit=ConcentrationUnit.micromolar,
                    ),
                    solvent=Solvent(
                        name=SOLVENT.value,
                        compound=resolved_compound("dimethyl sulfoxide"),
                    ),
                )
            ],
            aerobicity="aerobic",
            duration_hours=DURATION_HOURS.value,
            provenance_gaps=[
                ProvenanceGap(
                    field="duration_generations",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                    note="an 18 h liquid OD growth to saturation doses exposure in "
                    "hours, not doublings, and neither the paper nor the AID protocol "
                    "reports a doubling count",
                )
            ],
        )

    def _reference(
        self, environment: Environment
    ) -> EnvironmentResponseExperimentReference:
        """The screen's own normalized-growth center: z = 0, in the same compound well.

        The z-score is standardized WITHIN a screen, so its 0 is that screen's center,
        and the environment the baseline was measured in IS the compound environment.
        """
        return EnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain=PARENT_STRAIN.value
            ),
            environment_reference=environment,
            phenotype_reference=EnvironmentResponsePhenotype(
                measurement_type=Z_SCORE_DEFINITION.value,
                assay_type=ASSAY.value,
                environment_response=0.0,
                units=(
                    MEASUREMENT_UNITS
                    + "; the reference 0 is the screen's own normalized-growth center "
                    "by construction of the z-score, NOT a measured wild-type value"
                ),
            ),
        )

    def _phenotype(self, cell: MatrixCell) -> EnvironmentResponsePhenotype:
        """The screen-averaged z-score with its across-screen dispersion, or a typed gap."""
        z_values = cell.z_values
        category, label = cell.category()
        common: dict[str, Any] = {
            "measurement_type": Z_SCORE_DEFINITION.value,
            "assay_type": ASSAY.value,
            "environment_response": fmean(z_values),
            "category": category,
            "category_label": label,
            "n_samples": cell.n_screens,
            "sample_unit": SampleUnit.screen,
            "units": MEASUREMENT_UNITS,
        }
        if cell.n_screens == 1:
            return EnvironmentResponsePhenotype(
                **common,
                provenance_gaps=[
                    ProvenanceGap(
                        field=field,
                        reason=ProvenanceGapReason.not_reported_by_primary,
                        note="one released screen for this (strain, compound) cell; a "
                        "dispersion across screens is undefined at n=1 and the release "
                        "carries no per-screen error",
                    )
                    for field in (
                        "environment_response_uncertainty",
                        "environment_response_se",
                    )
                ],
            )
        dispersion = stdev(z_values)
        if dispersion == 0.0:
            raise RuntimeError(
                f"{cell.orf}/{cell.identity}: {cell.n_screens} distinct screens with a "
                "sample SD of exactly 0, which the datapoint-key measurement says "
                "cannot happen"
            )
        return EnvironmentResponsePhenotype(
            **common,
            environment_response_uncertainty=dispersion,
            environment_response_uncertainty_type=UncertaintyType.sample_sd,
        )

    def _genotype(self, systematic: str, common: str) -> Genotype:
        """One Euroscarf haploid deletion, carrying the collection it came from."""
        return Genotype(
            perturbations=[
                BarcodedKanMxDeletionPerturbation(
                    systematic_gene_name=systematic,
                    perturbed_gene_name=common,
                    collection=COLLECTION.value,
                )
            ]
        )

    # ---- build ------------------------------------------------------------------ #
    @post_process
    def process(self) -> None:
        """Collapse the datapoint export into the CGM matrix; write LMDB."""
        cells, n_rows, n_non_strain = self._collapse_matrix()
        source_records = len(cells)

        genome = default_genome()
        gene_set = {gene.upper() for gene in genome.gene_set}
        canonical = _canonical_common_names(genome)
        orfs = sorted({cell.orf for cell in cells.values()})
        off_genome = sorted(
            orf
            for orf in orfs
            if not (
                (resolution := genome.resolve_gene_name(orf)).is_current_gene
                and resolution.systematic_name in gene_set
            )
        )
        if off_genome:
            raise RuntimeError(
                f"{len(off_genome)} released ORFs are not current R64 genes: "
                f"{off_genome[:10]}; the release measured 242 current genes when this "
                "loader was written, so a new drop rule is needed, not a silent skip"
            )
        common_names = {orf: canonical.get(orf, orf) for orf in orfs}

        # Resolve each DISTINCT compound identity once; a compound carrying no InChIKey,
        # CID or ChEBI id cannot be encoded, so its cells are dropped.
        compounds: dict[str, Compound] = {}
        for cell in cells.values():
            if cell.identity not in compounds:
                compounds[cell.identity] = self._compound(cell)
        unidentified = sorted(
            identity
            for identity, compound in compounds.items()
            if compound.inchikey is None
            and compound.pubchem_cid is None
            and compound.chebi_id is None
        )
        unidentified_set = set(unidentified)
        n_dropped = sum(
            1 for cell in cells.values() if cell.identity in unidentified_set
        )
        names = [
            compound.name
            for identity, compound in compounds.items()
            if identity not in unidentified_set
        ]
        if len(set(names)) != len(names):
            raise RuntimeError(
                "two distinct compound identities resolve to the same canonical name, "
                "which would merge two conditions into one record key"
            )

        publication = Publication(doi=PAPER_DOI, doi_url=f"https://doi.org/{PAPER_DOI}")
        environments: dict[str, Environment] = {}
        references: dict[str, EnvironmentResponseExperimentReference] = {}

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        n_all_non_replicating = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for key in tqdm(sorted(cells), desc="Wildenhain2015 CGM"):
                cell = cells[key]
                if cell.identity in unidentified_set:
                    continue
                if cell.identity not in environments:
                    environments[cell.identity] = self._environment(
                        compounds[cell.identity]
                    )
                    references[cell.identity] = self._reference(
                        environments[cell.identity]
                    )
                if cell.all_screens_non_replicating:
                    n_all_non_replicating += 1
                experiment = EnvironmentResponseExperiment(
                    dataset_name=self.name,
                    genotype=self._genotype(cell.orf, common_names[cell.orf]),
                    environment=environments[cell.identity],
                    phenotype=self._phenotype(cell),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(
                        experiment, references[cell.identity], publication, itxn
                    ),
                )
                idx += 1
        env.close()
        interned_env.close()

        drop_log = DropLog(
            dataset=self.name,
            source_records=source_records,
            kept_records=idx,
            dropped_records=source_records - idx,
            rules=[
                DropRule(
                    rule="compound_without_a_structure_identifier",
                    scope="compound",
                    description=(
                        "the released row carries no PUBCHEM_CID and no SMILES, so the "
                        "compound has no InChIKey, CID or ChEBI id to be keyed or "
                        "joined by; only its submitter SID is known"
                    ),
                    n_records=n_dropped,
                    items=unidentified,
                )
            ],
        )
        with open(osp.join(self.preprocess_dir, "dropped_records.json"), "w") as handle:
            handle.write(drop_log.model_dump_json(indent=2))
        if drop_log.dropped_records != n_dropped:
            raise RuntimeError(
                f"drop accounting mismatch: rule total {n_dropped}, "
                f"{drop_log.dropped_records} records missing from the build"
            )
        log.info(
            "Wrote %d Wildenhain2015 records (%d dropped for an unidentifiable "
            "compound; %d non-strain rows ignored; %d cells whose every screen is "
            "non-replicate flagged)",
            idx,
            n_dropped,
            n_non_strain,
            n_all_non_replicating,
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
    root = osp.join(data_root, "data/torchcell/env_chemgen_wildenhain2015")
    dataset = EnvChemgenWildenhain2015Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()
