# torchcell/datasets/scerevisiae/wildenhain2015
# [[torchcell.datasets.scerevisiae.wildenhain2015]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/wildenhain2015
# Test file: tests/torchcell/datasets/scerevisiae/test_wildenhain2015.py
"""Wildenhain 2015 chemical-genetic matrix (CGM): env x geno -> z-score response.

Wildenhain et al. 2015 (Cell Systems, doi:10.1016/j.cels.2015.12.003) screened a panel of
haploid non-essential S. cerevisiae deletion strains ("sentinels", isogenic to BY4741,
Euroscarf collection) against 4,915 unique compounds from four chemical libraries at a
single 20 uM screening concentration, reading out growth inhibition as a normalized-OD600
Z-score. This loader builds ONLY the chemical-genetic matrix (strain x compound -> z-score);
the separate 128x128 cryptagen chemical-chemical synergy layer is out of scope.

A "sentinel" is an ORDINARY single-gene haploid deletion strain (NOT a HIP/HOP engineered
heterozygous strain), so there is no engineered-dosage complication: every screened strain
is one KanMX gene replacement in the BY4741 reference background (the standard Euroscarf
MATa deletion collection). The 195-strain "sentinel panel" is a dataset-level selection
note, not a per-record schema element; the RELEASED data actually covers 242 distinct
systematic ORFs (extra strains screened across the four libraries), all valid SGD R64 genes.

This maps onto the WS15 environment-perturbation ontology: ``EnvironmentResponseExperiment``
= single-deletion ``Genotype`` (a ``KanMxDeletionPerturbation`` in the BY4741 reference
background) x aerobic ``Environment`` carrying a ``SmallMoleculePerturbation`` (the compound
at 20 uM in DMSO) -> ``EnvironmentResponsePhenotype`` (``measurement_type=z_score``,
``environment_response`` the signed growth-inhibition Z-score). The parent BY4741 strain in
the same compound environment is the reference; its Z-score is 0 (the standardized control
baseline).

PROVENANCE / SOURCING (all sha256-pinned; see class constants + the runner Provenance):
- 195-strain sentinel panel, 4,915 unique compounds, 20 uM, Z-score averaged over the
  duplicate screens: RESULTS "Generation of a Chemical-Genetic Matrix" -- *"we screened a
  total of 4,915 unique compounds derived from four different chemical libraries against a
  panel of 195 non-essential deletion strains"* and *"We carried out over 600 growth-based
  screens in duplicate at a compound concentration of 20 uM ... Z scores were calculated and
  averaged for the replicate screens."* (paper.md line 68 + 70).
- Strain background (haploid, isogenic to BY4741, Euroscarf; NOT HIP/HOP): EXPERIMENTAL
  PROCEDURES -- *"S. cerevisiae deletion strains were obtained from the Euroscarf deletion
  set and are isogenic to BY4741"* (paper.md line 217).
- Medium / temperature / duration / solvent / duplicate structure (PubChem AID 1159580
  protocol): *"All strains were grown and screened in synthetic complete (SC) medium with 2%
  glucose."*, *"final concentration of 20 M. Screens were conducted in technical duplicate
  ... DMSO solvent only controls ... Plates were incubated at 30 C without shaking for
  approximately 18 h ... reading OD600 values"*, with per-record columns *"Raw OD read 1 =
  first replicate"*, *"Raw OD read 2 = second replicate"*, *"Z_score = ... per screen"*.
  Hence ``n_samples=2`` (the duplicate replicate screens, ``sample_unit=technical_replicate``)
  underlies each released Z-score; ``units`` records the readout definition.

DATA SOURCE (structured, scriptable, byte-stable):
- Accession: PubChem BioAssay AID 1159580 (paper.md line 237, "ACCESSION NUMBERS"). The
  additional http://chemgrid.org/cgm portal is an interactive PHP site (no bulk-downloadable
  processed matrix), so the PubChem datapoint export is the canonical scriptable full-data
  artifact. The per-AID CSV is pulled from the byte-stable NCBI FTP range archive and its
  inner ``1159580.csv.gz`` (fixed 2022-12-16 mtime) is sha256-pinned; its content is
  bit-identical to the PUG-REST ``/assay/aid/1159580/CSV`` export (492,126 datapoints).
- Each released row carries PUBCHEM_CID + PUBCHEM_EXT_DATASOURCE_SMILES, so compounds map
  directly to PubChem CIDs (99.9% of rows) and SMILES -- no live lookup needed.

BUILD / SOURCE QUIRKS handled deterministically (no fabrication):
- Rows whose ``orf`` is a non-strain control token (``NA`` / ``NULL``, 7,296 rows) are
  dropped -- only systematic-ORF strains enter the matrix.
- A given (strain, compound) can recur across the four libraries: 46,195 (ORF, compound)
  cells have 2-3 released screen rows with distinct Z-scores. These are collapsed to one
  matrix cell by AVERAGING the per-screen Z-scores -- exactly the paper's stated matrix
  construction (*"Z scores ... averaged for the replicate screens"*) -- and ``n_samples`` is
  set to ``2 x (number of screen rows)`` to count all contributing duplicate-screen reads
  (2 for the 89% single-screen cells). This averaging is a documented, deterministic
  reconstruction of the paper's DEFINED quantity from the canonical artifact (the Vanacloig
  precedent), NOT a value copied verbatim.
- Compound identity is the PubChem CID when present (the paper's unique-compound axis), else
  the PubChem SID (367 records lack a CID). The typed ``Compound`` carries this identifier as
  its ``name`` (``"CID <cid>"`` / ``"SID <sid>"``) because the released structured artifact
  provides NO human compound name; ``Compound.pubchem_cid`` carries the integer CID and
  ``Compound.smiles`` the released SMILES. Final matrix: 242 strains x compounds = 428,573
  records.
"""

import csv
import gzip
import hashlib
import io
import logging
import os
import os.path as osp
import pickle
import re
import urllib.request
import zipfile
from collections.abc import Callable
from statistics import fmean
from typing import Any

import lmdb
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.schema import (
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
    KanMxDeletionPerturbation,
    MeasurementType,
    Media,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SmallMoleculePerturbation,
    Solvent,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DOI = "10.1016/j.cels.2015.12.003"
# Byte-stable NCBI FTP range archive holding the per-AID datapoint export. The inner
# 1159580.csv.gz (fixed 2022-12-16 mtime) is the canonical raw artifact and is sha256-pinned.
_FTP_ZIP_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/CSV/Data/1159001_1160000.zip"
)
_ZIP_MEMBER = "1159001_1160000/1159580.csv.gz"
_RAW_FILENAME = "1159580.csv.gz"
_RAW_SHA256 = "c461c679b63ac56045cef0f03ed9bcbb8e7f9c12146f1fc7cc8ac0c113188d64"

# Systematic ORF pattern (protein-coding); non-strain tokens (NA/NULL) are dropped.
_SYSTEMATIC_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$")

SCREEN_CONCENTRATION_UM = 20.0
MEASUREMENT_UNITS = (
    "Z-score of growth inhibition from normalized OD600 (20 uM compound vs DMSO control, "
    "SC + 2% glucose, 30 C, ~18 h); negative = growth inhibition, averaged over the "
    "duplicate replicate screens"
)


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
        """Initialize the dataset. ORFs are already systematic, so no genome is required."""
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
        """The single PubChem AID 1159580 datapoint archive required before processing."""
        return [_RAW_FILENAME]

    def download(self) -> None:
        """Fetch the FTP range zip, extract AID 1159580's csv.gz, verify its pinned sha256."""
        os.makedirs(self.raw_dir, exist_ok=True)
        dest = osp.join(self.raw_dir, _RAW_FILENAME)
        if osp.exists(dest):
            return
        log.info("Downloading Wildenhain2015 PubChem AID archive from %s", _FTP_ZIP_URL)
        req = urllib.request.Request(
            _FTP_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            zip_bytes = resp.read()
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            member = zf.read(_ZIP_MEMBER)
        digest = hashlib.sha256(member).hexdigest()
        if digest != _RAW_SHA256:
            raise RuntimeError(
                f"{_RAW_FILENAME} sha256 mismatch: got {digest}, expected {_RAW_SHA256}"
            )
        with open(dest, "wb") as handle:
            handle.write(member)
        log.info("Wrote %s (%d bytes, sha256 verified)", dest, len(member))

    def _collapse_matrix(self) -> dict[tuple[str, str], dict[str, Any]]:
        """Read the datapoint CSV; collapse to one cell per (ORF, compound-identity).

        Identity is the PubChem CID when present, else the PubChem SID. Per-screen
        Z-scores for a recurring (strain, compound) cell are averaged (the paper's matrix
        construction), and the contributing screen count is tracked for ``n_samples``.
        """
        path = osp.join(self.raw_dir, _RAW_FILENAME)
        cells: dict[tuple[str, str], dict[str, Any]] = {}
        n_rows = 0
        n_control = 0
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
                    n_control += 1
                    continue
                z_raw = row[idx["z_score"]].strip()
                if not z_raw:
                    continue
                cid = row[idx["PUBCHEM_CID"]].strip()
                sid = row[idx["PUBCHEM_SID"]].strip()
                smiles = row[idx["PUBCHEM_EXT_DATASOURCE_SMILES"]].strip()
                sym = row[idx["sym"]].strip()
                identity = f"CID {cid}" if cid else f"SID {sid}"
                key = (orf, identity)
                cell = cells.get(key)
                if cell is None:
                    cells[key] = {
                        "orf": orf,
                        "sym": sym or orf,
                        "compound_name": identity,
                        "pubchem_cid": int(cid) if cid else None,
                        "smiles": smiles or None,
                        "z_values": [float(z_raw)],
                    }
                else:
                    cell["z_values"].append(float(z_raw))
                n_rows += 1
        log.info(
            "Wildenhain2015: parsed %d strain datapoints (%d non-strain control rows "
            "dropped) -> %d (ORF, compound) cells",
            n_rows,
            n_control,
            len(cells),
        )
        return cells

    def _environment(self, cell: dict[str, Any]) -> Environment:
        """Aerobic SC (2% glucose) liquid culture carrying the compound at 20 uM in DMSO."""
        return Environment(
            media=Media(
                name="synthetic complete (SC), 2% glucose",
                state="liquid",
                is_synthetic=True,
            ),
            temperature=Temperature(value=30.0),
            perturbations=[
                SmallMoleculePerturbation(
                    compound=Compound(
                        name=cell["compound_name"],
                        pubchem_cid=cell["pubchem_cid"],
                        smiles=cell["smiles"],
                    ),
                    concentration=Concentration(
                        value=SCREEN_CONCENTRATION_UM, unit=ConcentrationUnit.micromolar
                    ),
                    solvent=Solvent(name="DMSO"),
                )
            ],
            aerobicity="aerobic",
            duration_hours=18.0,
        )

    def _reference(
        self, environment: Environment
    ) -> EnvironmentResponseExperimentReference:
        """Parent BY4741 baseline in the same compound environment: control Z-score = 0."""
        phenotype_reference = EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.z_score,
            environment_response=0.0,
            units=MEASUREMENT_UNITS,
        )
        return EnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain="BY4741"
            ),
            environment_reference=environment.model_copy(),
            phenotype_reference=phenotype_reference,
        )

    def _experiment(
        self, cell: dict[str, Any], environment: Environment
    ) -> EnvironmentResponseExperiment:
        """Build one env x geno -> z-score experiment for a (strain, compound) cell."""
        genotype = Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=cell["orf"], perturbed_gene_name=cell["sym"]
                )
            ]
        )
        z_values = cell["z_values"]
        phenotype = EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.z_score,
            environment_response=fmean(z_values),
            n_samples=2 * len(z_values),
            sample_unit=SampleUnit.technical_replicate,
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
        """Collapse the datapoint export into the CGM matrix; write LMDB."""
        cells = self._collapse_matrix()
        publication = Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")
        pub_dump = publication.model_dump()

        # One reference per compound environment (Z-score control baseline is 0); cache
        # the reference dump per compound identity to avoid rebuilding it per strain.
        ref_cache: dict[str, dict[str, Any]] = {}

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e12))
        idx = 0
        with env.begin(write=True) as txn:
            for (orf, identity), cell in tqdm(
                sorted(cells.items()), desc="Wildenhain2015 CGM"
            ):
                environment = self._environment(cell)
                ref_dump = ref_cache.get(identity)
                if ref_dump is None:
                    ref_dump = self._reference(environment).model_dump()
                    ref_cache[identity] = ref_dump
                experiment = self._experiment(cell, environment)
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
        log.info(
            "Wrote %d Wildenhain2015 environment-response experiments to LMDB", idx
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
