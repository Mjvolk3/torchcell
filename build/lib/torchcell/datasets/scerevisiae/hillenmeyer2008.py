# torchcell/datasets/scerevisiae/hillenmeyer2008
# [[torchcell.datasets.scerevisiae.hillenmeyer2008]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/hillenmeyer2008
# Test file: tests/torchcell/datasets/scerevisiae/test_hillenmeyer2008.py
"""Hillenmeyer 2008 "chemical genomic portrait of yeast" (FitDb): env x geno -> fitness.

Hillenmeyer et al. 2008 (Science 320:362; doi:10.1126/science.1150021; citation key
``hillenmeyerChemicalGenomicPortrait2008``) is the genome-scale HIP/HOP fitness compendium:
the heterozygous (HIP) and homozygous (HOP) barcoded deletion collections competitively
grown under ~400 chemical + environmental conditions, read out as a per-strain fitness
defect. It is the earlier, broader companion to the Hoepfner 2014 HIP-HOP atlas.

TWO DATASETS (the readouts are incomparable -- a log-ratio must never be mixed with a
z-score in one measurement_type -- so HET and HOM are SEPARATE dataset classes,
``HetHillenmeyer2008Dataset`` and ``HomHillenmeyer2008Dataset``, cf. the Smf/Dmf split):
- ``het.ratio_result_nm.pub`` -- HET (HIP) fitness-defect LOG-RATIO: 5984 strains x 726
  arrays, ``measurement_type=log2_ratio`` (FD log-ratio = log2(mean control intensity /
  treatment intensity), averaged over up+down tags; positive = fitness defect).
- ``hom.z_result_nm.pub`` -- HOM (HOP) fitness-defect Z-SCORE: 4769 strains x 418 arrays,
  ``measurement_type=z_score`` (no HOM log-ratio matrix was archived). z = (mu_c - x_t)/sigma_c.

GENOTYPE (as in Hoepfner): the collections are diploid (BY4743-derived). HET = one of the
two autosomal copies deleted (KanMX) -> reduced dosage ->
``EngineeredCopyNumberPerturbation(copy_number=1, reference_copy_number=2, marker='KanMX')``
in a ``ReferenceGenome(ploidy='diploid')``, incl. essential genes. HOM = both copies deleted
-> ``KanMxDeletionPerturbation`` in a diploid. One strain per gene (no allelic series).

ENVIRONMENT -- parsed from each column header
``filename:cond1:conc1:unit1:cond2:conc2:unit2:generations:pool:scanner``. The biological
environment is (cond1[+cond2], concentration, generations); ``pool``/``scanner``/``filename``
are technical. Conditions are classified per SOM Table S1 (environmental stress vs small
molecule); the bulk are small molecules, and a curated set of RULES handles the rest:
- ``<N> degrees C`` / ``37c, 45c`` -> raised/lowered ``Environment.temperature`` (M2; no
  perturbation), baseline 30 C. ``37c, 45c`` (heat-shock cycle) -> the peak 45 C.
- ``pH<x>`` -> ``EnvironmentPhysicalPerturbation(factor=ph, magnitude=<x> pH)``.
- amino-acid / vitamin ``... dropout`` / ``... drop-out`` -> ``EnvironmentPhysicalPerturbation
  (factor=nutrient_dropout, agent=Compound(<nutrient>))`` -- the removed nutrient is the SAME
  compound entity a dataset ADDING it would use (join-able).
- ``no drug irradiated`` -> ``EnvironmentPhysicalPerturbation(factor=radiation)`` (no dose
  released -> qualitative, magnitude None).
- ``minimal media`` / ``synthetic complete`` -> base-medium swap on ``Environment.media``
  (no perturbation; recognized as an edit by L3 via non-baseline media).
- ``YP glycerol`` -> ``EnvironmentPhysicalPerturbation(factor=carbon_source, agent=glycerol)``.
- everything else -> ``SmallMoleculePerturbation(Compound(name=cond1), Concentration(conc,unit))``;
  a second compound (cond2) adds a second small-molecule perturbation (combination treatment).

The ``generations`` field is the readout timepoint in doublings of competitive growth and is
stored verbatim on ``Environment.duration_generations`` (source values 0/5/10/15/20 and the
early-timepoint labels -5/-10); distinct generation counts are distinct environments.

REPLICATE AGGREGATION -- the atlas is per-array and per-barcode-probe, but the ontology's
record is per (strain, environment). Two kinds of replicate are aggregated into one record
(MEAN score, ``n_samples`` = number of measurements, sample SD -> derived SE for n>=2):
(1) arrays sharing an identical biological environment (same compound+concentration+
generations+...); (2) multiple barcode-probe ROWS of the same gene deletion (e.g.
``YBR020W:chr00_18`` and ``YBR020W:chr2_2`` are the same strain). This is why L1 uniqueness
(one record per strain x condition) holds. ORFs are validated against SGD R64 (grouping on
the systematic name, so probe rows collapse); non-R64 names are dropped and logged.

DATA SOURCE: live FitDb portals are DNS-dead and the Science SI 403s; the Stanford static
supplement is recovered from the Internet Archive (per-file snapshot timestamps), deposited
to the raw mirror, and sha256-pinned. See ``[[fitdb-hillenmeyer2008-wayback-data-source]]``.
"""

import hashlib
import json
import logging
import math
import os
import os.path as osp
import pickle
import re
from collections.abc import Callable
from typing import Any

import lmdb
from tqdm import tqdm

from torchcell.data import ExperimentDataset, post_process
from torchcell.datamodels.schema import (
    Compound,
    Concentration,
    ConcentrationUnit,
    DoseBasis,
    EngineeredCopyNumberPerturbation,
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
    Media,
    PhysicalFactor,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.dataset_registry import register_dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DOI = "10.1126/science.1150021"

_BASE_MEDIUM = "YPD"
_BASELINE_TEMPERATURE_C = 30.0

# Wayback-mirrored Stanford supplement matrices (sha256-pinned).
_MATRICES: dict[str, dict[str, Any]] = {
    "het": {
        "filename": "het.ratio_result_nm.pub",
        "sha256": "c0ddefaeb44c9760481dc443d1c2e82570bad4eb1a5458acd74b169bec47cb44",
        "measurement_type": MeasurementType.log2_ratio,
        "collection": "heterozygous",
        "units": (
            "HIP fitness-defect log-ratio = log2(mean control intensity / treatment "
            "intensity), mean of up+down tags (Hillenmeyer 2008); positive = fitness "
            "defect (strain depleted under treatment); mean over replicate arrays + barcode probes"
        ),
    },
    "hom": {
        "filename": "hom.z_result_nm.pub",
        "sha256": "0b7d5e4dad0e5b4336b1dac97fb32ef14d7d0bf9f5d1b0d5ddb382c393362b71",
        "measurement_type": MeasurementType.z_score,
        "collection": "homozygous",
        "units": (
            "HOP fitness-defect z-score = (mean control - treatment)/SD control "
            "(Hillenmeyer 2008); positive = fitness defect; mean over replicate arrays + barcode probes"
        ),
    },
}

# S288C R64 gene universe (systematic ORF + RNA-coding names) for R64 resolution.
_SGD_GENE_FASTAS = (
    "data/sgd/genome/S288C_reference_genome_R64-4-1_20230830/"
    "orf_coding_all_R64-4-1_20230830.fasta",
    "data/sgd/genome/S288C_reference_genome_R64-4-1_20230830/"
    "rna_coding_R64-4-1_20230830.fasta",
)

_UNIT_MAP = {
    "um": ConcentrationUnit.micromolar,
    "nm": ConcentrationUnit.nanomolar,
    "mm": ConcentrationUnit.millimolar,
    "m": ConcentrationUnit.molar,
    "ug/ml": ConcentrationUnit.ug_per_ml,
    "%": ConcentrationUnit.percent_v_v,
}

_TEMP_RE = re.compile(r"(\d+(?:\.\d+)?)\s*degrees?\s*c", re.I)
_TEMP_CYCLE_RE = re.compile(r"^\s*(\d+)\s*c\s*,\s*(\d+)\s*c", re.I)
_PH_RE = re.compile(r"^\s*ph\s*([\d.]+)\s*$", re.I)
_DROPOUT_RE = re.compile(
    r"(.+?)\s*(?:partial\s*)?(?:drop-?out)(?:\s*control media)?\s*$", re.I
)
_MEDIA_SWAPS = {"minimal media", "synthetic complete"}
# Missing-value tokens in the score matrices (upper-cased) -- skipped when aggregating.
_MISSING = {"", "NA", "NAN", "NULL"}


def _load_sgd_genes(data_root: str) -> set[str]:
    """S288C R64 systematic-name universe from the ORF + RNA-coding FASTA headers."""
    genes: set[str] = set()
    for rel in _SGD_GENE_FASTAS:
        with open(osp.join(data_root, rel)) as handle:
            for line in handle:
                if line.startswith(">"):
                    genes.add(line[1:].split()[0])
    return genes


def _concentration(conc: str, unit: str) -> Concentration:
    """Parse a (value, unit) dose; fall back to a fixed-dose basis when the unit is absent."""
    unit_enum = _UNIT_MAP.get(unit.strip().lower()) if unit.strip() else None
    if conc.strip() and unit_enum is not None:
        return Concentration(value=float(conc), unit=unit_enum)
    return Concentration(basis=DoseBasis.fixed)


class _ColumnSpec:
    """Parsed environment for one matrix column + the grouping key of biological replicates."""

    __slots__ = ("index", "group_key", "environment")

    def __init__(self, index: int, group_key: str, environment: Environment):
        self.index = index
        self.group_key = group_key
        self.environment = environment


class _Hillenmeyer2008Base(ExperimentDataset):
    """Shared FitDb HIP/HOP loader. HET (log2_ratio) and HOM (z_score) are SEPARATE datasets
    (incomparable readouts, distinct collections), one concrete subclass each.
    """

    _matrix_key: str = ""  # 'het' | 'hom' -- set by concrete subclasses

    def __init__(
        self,
        root: str,
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
        """The one Wayback-mirrored score matrix this dataset reads."""
        return [_MATRICES[self._matrix_key]["filename"]]

    def download(self) -> None:
        """Verify the manually deposited (Wayback-mirrored) matrix; check pinned sha256."""
        spec = _MATRICES[self._matrix_key]
        dest = osp.join(self.raw_dir, spec["filename"])
        if not osp.exists(dest):
            raise RuntimeError(
                f"{spec['filename']} not found in {self.raw_dir}. FitDb portals are "
                "DNS-dead and the Science SI 403s; recover the Stanford supplement from "
                "the Internet Archive (see fitdb-hillenmeyer2008-wayback-data-source) and "
                "deposit it, then rebuild (sha256 verified)."
            )
        digest = hashlib.sha256(open(dest, "rb").read()).hexdigest()
        if digest != spec["sha256"]:
            raise RuntimeError(
                f"{spec['filename']} sha256 mismatch: got {digest}, "
                f"expected {spec['sha256']}"
            )

    def _perturbations(
        self, cond: str, conc: str, unit: str, cond2: str, conc2: str, unit2: str
    ) -> tuple[list[Any], str, float]:
        """Classify one column's condition -> (perturbations, media_name, temperature).

        Rule-based per SOM Table S1: temperature / pH / nutrient-dropout / radiation /
        media-swap / carbon-source are typed physical edits; everything else is a small
        molecule. A second compound (cond2) is a combination-treatment small molecule.
        """
        cl = cond.strip()
        low = cl.lower()

        cycle = _TEMP_CYCLE_RE.match(cl)
        if cycle:
            return (
                [],
                _BASE_MEDIUM,
                float(max(int(cycle.group(1)), int(cycle.group(2)))),
            )
        temp_match = _TEMP_RE.search(cl)
        if temp_match:
            return [], _BASE_MEDIUM, float(temp_match.group(1))

        ph_match = _PH_RE.match(cl)
        if ph_match:
            pert = EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.ph,
                magnitude=Concentration(
                    value=float(ph_match.group(1)), unit=ConcentrationUnit.ph
                ),
            )
            return [pert], _BASE_MEDIUM, _BASELINE_TEMPERATURE_C

        if "irradiated" in low:
            pert = EnvironmentPhysicalPerturbation(factor=PhysicalFactor.radiation)
            return [pert], _BASE_MEDIUM, _BASELINE_TEMPERATURE_C

        if "dropout" in low or "drop-out" in low:
            nutrient = _DROPOUT_RE.match(cl)
            name = nutrient.group(1).strip() if nutrient else cl
            pert = EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.nutrient_dropout, agent=Compound(name=name)
            )
            return [pert], _BASE_MEDIUM, _BASELINE_TEMPERATURE_C

        if low in _MEDIA_SWAPS:
            return [], cl, _BASELINE_TEMPERATURE_C

        if low == "yp glycerol":
            pert = EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.carbon_source, agent=Compound(name="glycerol")
            )
            return [pert], _BASE_MEDIUM, _BASELINE_TEMPERATURE_C

        perts: list[Any] = [
            SmallMoleculePerturbation(
                compound=Compound(name=cl), concentration=_concentration(conc, unit)
            )
        ]
        if cond2.strip():
            perts.append(
                SmallMoleculePerturbation(
                    compound=Compound(name=cond2.strip()),
                    concentration=_concentration(conc2, unit2),
                )
            )
        return perts, _BASE_MEDIUM, _BASELINE_TEMPERATURE_C

    def _column_specs(self, header: list[str]) -> list[_ColumnSpec]:
        """Parse every data column into its environment + biological-replicate group key."""
        specs: list[_ColumnSpec] = []
        for index, raw in enumerate(header):
            if index == 0:
                continue  # 'Orf'
            parts = raw.strip().split(":")
            while len(parts) < 10:
                parts.append("")
            _fname, c1, co1, u1, c2, co2, u2, gen, _pool, _scanner = parts[:10]
            perts, media, temp = self._perturbations(c1, co1, u1, c2, co2, u2)
            env = Environment(
                media=Media(name=media, state="liquid", is_synthetic=False),
                temperature=Temperature(value=temp),
                perturbations=perts,
                aerobicity="aerobic",
                duration_generations=float(gen.replace("gen", "")) if gen else None,
            )
            # Group replicate arrays by the CANONICAL environment (the same identity the L1
            # verifier keys on), not the raw header strings -- so two columns that parse to
            # the same environment (e.g. concentration '6.9' vs '6.90') aggregate together
            # instead of becoming an L1 duplicate.
            group_key = json.dumps(env.model_dump(), sort_keys=True, default=str)
            specs.append(_ColumnSpec(index, group_key, env))
        return specs

    def _genotype(self, collection: str, orf: str) -> Genotype:
        """HET -> heterozygous engineered-CNV (copy 1 of 2); HOM -> homozygous deletion."""
        if collection == "heterozygous":
            return Genotype(
                perturbations=[
                    EngineeredCopyNumberPerturbation(
                        systematic_gene_name=orf,
                        perturbed_gene_name=orf,
                        copy_number=1,
                        reference_copy_number=2,
                        marker="KanMX",
                    )
                ]
            )
        return Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=orf
                )
            ]
        )

    def _reference_dump(self, spec: dict[str, Any]) -> dict[str, Any]:
        """One shared per-matrix reference: the untreated diploid-collection control, 0."""
        strain = (
            "BY4743 heterozygous deletion collection"
            if spec["collection"] == "heterozygous"
            else "BY4743 homozygous deletion collection"
        )
        return EnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=ReferenceGenome(
                species="Saccharomyces cerevisiae", strain=strain, ploidy="diploid"
            ),
            environment_reference=Environment(
                media=Media(name=_BASE_MEDIUM, state="liquid", is_synthetic=False),
                temperature=Temperature(value=_BASELINE_TEMPERATURE_C),
                aerobicity="aerobic",
            ),
            phenotype_reference=EnvironmentResponsePhenotype(
                measurement_type=spec["measurement_type"],
                environment_response=0.0,
                units=spec["units"],
            ),
        ).model_dump()

    def _iter_matrix(
        self,
        key: str,
        spec: dict[str, Any],
        sgd_genes: set[str],
        pub_dump: dict[str, Any],
    ) -> Any:
        """Stream one matrix; aggregate replicate arrays per (strain, environment)."""
        path = osp.join(self.raw_dir, spec["filename"])
        ref_dump = self._reference_dump(spec)
        dropped: set[str] = set()
        n_written = 0
        with open(path) as handle:
            header = handle.readline().rstrip("\n").split("\t")
            specs = self._column_specs(header)
            # group columns (arrays) sharing an identical biological environment
            groups: dict[str, list[_ColumnSpec]] = {}
            for cs in specs:
                groups.setdefault(cs.group_key, []).append(cs)
            group_list = list(groups.values())
            log.info(
                "Hillenmeyer2008 %s: %d arrays -> %d unique environments",
                key,
                len(specs),
                len(group_list),
            )
            # Group ALL matrix rows by systematic ORF. A gene deletion appears in multiple
            # rows when it has multiple barcode probes (e.g. YBR020W:chr00_18 and
            # YBR020W:chr2_2) -- these are the SAME strain, so their measurements aggregate
            # together (with the replicate arrays) into one record per (strain, environment).
            rows_by_orf: dict[str, list[list[str]]] = {}
            for line in handle:
                cells = line.rstrip("\n").split("\t")
                orf = cells[0].strip().strip('"').split(":")[0]
                if orf not in sgd_genes:
                    dropped.add(orf)
                    continue
                rows_by_orf.setdefault(orf, []).append(cells)
            for orf, orf_rows in tqdm(
                rows_by_orf.items(), desc=f"Hillenmeyer2008 {key}"
            ):
                genotype = self._genotype(spec["collection"], orf)
                for members in group_list:
                    values = [
                        float(cells[cs.index])
                        for cells in orf_rows
                        for cs in members
                        if cs.index < len(cells)
                        and cells[cs.index].strip().upper() not in _MISSING
                    ]
                    if not values:
                        continue
                    n = len(values)
                    mean = sum(values) / n
                    sd = (
                        math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
                        if n >= 2
                        else None
                    )
                    phenotype = EnvironmentResponsePhenotype(
                        measurement_type=spec["measurement_type"],
                        environment_response=mean,
                        n_samples=n,
                        sample_unit=SampleUnit.biological_replicate,
                        environment_response_uncertainty=sd,
                        environment_response_uncertainty_type=(
                            UncertaintyType.sample_sd if sd is not None else None
                        ),
                        units=spec["units"],
                    )
                    experiment = EnvironmentResponseExperiment(
                        dataset_name=self.name,
                        genotype=genotype,
                        environment=members[0].environment,
                        phenotype=phenotype,
                    )
                    n_written += 1
                    yield pickle.dumps(
                        {
                            "experiment": experiment.model_dump(),
                            "reference": ref_dump,
                            "publication": pub_dump,
                        }
                    )
        log.info(
            "Hillenmeyer2008 %s: wrote %d records; dropped %d non-R64 ORF names",
            key,
            n_written,
            len(dropped),
        )

    @post_process
    def process(self) -> None:
        """Aggregate this dataset's matrix into per-(strain, environment) records; write LMDB."""
        from dotenv import load_dotenv

        load_dotenv()
        data_root = os.environ["DATA_ROOT"]
        sgd_genes = _load_sgd_genes(data_root)
        pub_dump = Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}").model_dump()
        spec = _MATRICES[self._matrix_key]

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e12))
        batch_size = 500_000
        idx = 0
        txn = env.begin(write=True)
        for value in self._iter_matrix(self._matrix_key, spec, sgd_genes, pub_dump):
            txn.put(f"{idx}".encode(), value)
            idx += 1
            if idx % batch_size == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.commit()
        env.close()
        log.info(
            "Wrote %d Hillenmeyer2008 %s environment-response records to LMDB",
            idx,
            self._matrix_key,
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


@register_dataset
class HetHillenmeyer2008Dataset(_Hillenmeyer2008Base):
    """FitDb HIP (heterozygous, engineered-CNV) fitness-defect log2-ratio dataset."""

    _matrix_key = "het"

    def __init__(
        self,
        root: str = "data/torchcell/env_chemgen_hillenmeyer2008_het",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the HET (HIP) dataset."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)


@register_dataset
class HomHillenmeyer2008Dataset(_Hillenmeyer2008Base):
    """FitDb HOP (homozygous, KanMX deletion) fitness-defect z-score dataset."""

    _matrix_key = "hom"

    def __init__(
        self,
        root: str = "data/torchcell/env_chemgen_hillenmeyer2008_hom",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the HOM (HOP) dataset."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)


def main() -> None:
    """Build/load both datasets for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    for cls, sub in (
        (HetHillenmeyer2008Dataset, "het"),
        (HomHillenmeyer2008Dataset, "hom"),
    ):
        root = osp.join(data_root, f"data/torchcell/env_chemgen_hillenmeyer2008_{sub}")
        dataset = cls(root=root)
        print(f"{sub}: len = {len(dataset)}")
        print(dataset[0])


if __name__ == "__main__":
    main()
