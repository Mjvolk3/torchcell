# torchcell/datasets/ecoli/wang2015_growth
# [[torchcell.datasets.ecoli.wang2015_growth]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/wang2015_growth
# Test file: tests/torchcell/datasets/ecoli/test_wang2015_growth.py
r"""Wang 2015 Table S3, the no-isoprenol column: the plain-medium growth of 46 deletions.

Wang et al. 2015 (Sci Rep 5:16505, doi:10.1038/srep16505) Supplementary Table S3 has
exactly three columns -- ``Strains``, ``Cell growth (OD600) / No isoprenol`` and
``Cell growth (OD600) / 0.5% (v/v) isoprenol`` -- each value cell printed ``mean +/- SD``
over 47 rows (BW25113 plus 46 ``BWD`` Keio deletions). ``wang2015.py`` serves the
isoprenol arm: a log2 relative tolerance that is a ratio OF the two columns. This module
serves the FIRST column in its own right as :class:`GrowthWang2015Dataset`, one
``BacterialFitnessExperiment`` per deletion strain in PLAIN 2YT: **46 records and one
reference**, the BW25113 row being the reference the ratio is taken against.

Rank 12 of the E. coli SI audit called this "a recorded decision rather than a release
defect", and the decision is recorded here. It is not a second copy of a stored number:
measured over the 46 mutants, Pearson r(``od600_without``, the stored log2) is 0.5454
against 0.9774 for ``od600_with``; 0 of 46 stored values equal any no-isoprenol value;
and ``max|log2(od600_without_strain / od600_without_parent) - stored| = 0.6632``. The
stored log2 is a function of BOTH columns, so neither column is recoverable from it.

WHY ``FitnessPhenotype`` AND NOT THE ABSOLUTE ENVIRONMENT-RESPONSE BRANCH. The readout
is a 12 h endpoint OD600 of a DELETION strain against its parent in the same medium,
which is a genotype ratio, and ``FitnessPhenotype`` is the genotype-ratio carrier. The
absolute branch PR #836 landed does NOT reach it: that branch "REFUSES any record whose
``measurement_type`` is not in ``ABSOLUTE_MEASUREMENT_TYPES``"
(``verification/environment_response.py``), the set is
``{growth_rate, colony_size}``, and ``MeasurementType`` has no optical-density member, so
an endpoint OD600 would have to be mislabeled a growth rate to take it.
``FitnessPhenotype`` also carries no ``measurement_type`` field at all, which is the
second reason this readout belongs to the fitness family rather than the typed-axis one.

THE CLAMP NEVER FIRES, MEASURED. ``FitnessPhenotype.validate_fitness`` clamps a
non-positive value to 0.0, which would destroy a measurement. The 46 ratios run
0.877108 to 1.030120 with a median of 0.956024 and none non-positive, so nothing is
clamped and no information is lost. (That is the same guard
``schmidt2016_growth_rate.py`` states for its six Table S24 ratios.) Honest caveat, and
it is a reading rather than a measurement: ``FitnessPhenotype.fitness`` is documented
``ko_growth_rate/wt_growth_rate`` and this ratio is an endpoint BIOMASS ratio, not a rate
ratio. The number is a legitimate same-environment deletion-over-parent growth ratio,
and it is a looser reading of the field's description than Schmidt's real h^-1 rates.

THE UNCERTAINTY IS EXACT ARITHMETIC, NOT A GAP. Unlike the stored log2 record -- whose
dispersion is a typed gap because a ratio of ratios mixes four means -- this ratio mixes
only two, so the Schmidt Table S24 arithmetic applies verbatim:

- ``fitness_uncertainty`` is ``sd_without(strain) / od600_without(parent)`` with
  ``fitness_uncertainty_type=sample_sd`` and ``n_samples=2``. This is exact: it is the
  sample standard deviation of the n released ratio observations
  ``{od600_r / mean(parent)}``, so the released statistic's kind and n survive the
  change of units. Measured range 0.00120 to 0.04096, and 0 of 46 ``sd_without`` cells
  are zero.
- ``fitness_se`` is supplied explicitly as the delta-method standard error of the ratio,
  ``fitness * sqrt((SE_strain/mean_strain)^2 + (SE_parent/mean_parent)^2)`` with
  ``SE_x = SD_x / sqrt(2)``. Measured range 0.01261 to 0.03142. It is always at least as
  large as the auto-derived ``fitness_uncertainty / sqrt(n)``, which conditions on the
  parent mean as a fixed denominator and so understates the spread; the build asserts
  that direction over every record, so the stored SE can never be the optimistic one.

``n_samples = 2`` is STATED, not inferred. Table S3 releases no n; the Fig. 1C caption
does, verbatim: "The growth inhibition of the wild type strain is at approximate $5 0 \%$
as a reference. Results are the means of two biological replicates." That is
``wang2015.SOURCED_VALUES["n_samples"]``, reused here rather than re-sourced, and the
text cites Table S3 and Fig. 1C together. The kind is the table's own note, "Note: The
results are presented as means $\pm$ standard divisions." (a typo for deviations),
corroborated by Figs. 3C, 4 and 5, "Error bars represent the standard deviations of two
biological replicates."

THE ENVIRONMENT IS THE STORED DATASET'S MINUS THE ISOPRENOL. 2YT at pH 7.0 and 30 C,
shaken, 12 h: the Methods' own "4 mL of fresh media with (or without) isoprenol" is what
makes the no-isoprenol column an independently grown culture rather than a time point of
the same one. Isoprenol is ABSENT here, not a gap; the pH perturbation keeps its agent
gap, because the paper names no acid or base.

A SEPARATE MODULE FROM ``wang2015.py``, on purpose, and the reason is measured.
``build_manifest`` keys a built store's staleness on the schema closure of the loader
MODULE's own ``torchcell.datamodels`` imports (``provenance/schema_deps.py:
loader_closure``). ``wang2015.py``'s closure is 69 symbols and equals the 69 the served
``env_chemgen_wang2015`` store records; adding ``BacterialFitnessExperiment``,
``BacterialFitnessExperimentReference`` and ``FitnessPhenotype`` to its imports raises it
to 72, which would mark that store STALE for a change touching none of its 46 records.
The pinned artifact, the SI parsers, the strain resolver, the media object and the
sourced values are imported FROM ``wang2015`` instead, so there is one copy of each.

DATA. The one consumed file is ``si1.pdf``, the same pinned SI ``wang2015.py`` consumes
from the same raw mirror under the same sha256, read with the same
``pdftotext -layout -enc UTF-8`` recipe and checked against the same parsed-Table-S3
digest. Nothing is retrieved here that that module does not already record.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import os.path as osp
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.media import YT_2X
from torchcell.datamodels.schema import (
    BacterialFitnessExperiment,
    BacterialFitnessExperimentReference,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentPhysicalPerturbation,
    Experiment,
    ExperimentReference,
    FitnessPhenotype,
    PhysicalFactor,
    SampleUnit,
    Temperature,
    UncertaintyType,
)
from torchcell.datasets.bacteria_common import (
    BACTERIAL_ASSEMBLY_SETS,
    assembly_reference,
    bacterial_genome,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.ecoli import wang2015 as wg
from torchcell.sequence.genome.ecoli.k12 import EcoliK12Genome, EcoliK12StrainName
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)
from torchcell.verification.sourced import SourcedValue

log = logging.getLogger(__name__)

DATASET_ROOT_REL = "data/torchcell/growth_wang2015"

#: Build oracles, every one measured on the pinned bytes before being written here.
EXPECTED_SOURCE_ROWS = 47
EXPECTED_RECORDS = 46
EXPECTED_REFERENCES = 1
#: Measured over the 46 released ratios: the clamp in ``validate_fitness`` never fires.
MIN_FITNESS = 0.877108
MAX_FITNESS = 1.030120
#: Tolerance of the declared fitness range against the recomputed one.
FITNESS_RANGE_ATOL = 1e-6

#: The released column this module consumes, by its printed header.
COLUMN = "Cell growth (OD600) / No isoprenol"
#: What the stored ratio is, for the verifier's method line and the note.
FITNESS_DEFINITION = (
    "OD600 of the deletion strain in plain 2YT after 12 h over BW25113's in the same "
    "medium (Table S3's 'No isoprenol' column)"
)

#: Reused from ``wang2015`` rather than re-sourced, so there is one copy of each quote.
N_SAMPLES: int = int(wg.N_SAMPLES.value)
PARENT_STRAIN = wg.PARENT_STRAIN

SOURCED_VALUES: dict[str, SourcedValue] = {
    "background_strain": wg.SOURCED_VALUES["background_strain"],
    "collection": wg.SOURCED_VALUES["collection"],
    "cassette": wg.SOURCED_VALUES["cassette"],
    "data_location": wg.SOURCED_VALUES["data_location"],
    "medium": wg.SOURCED_VALUES["medium"],
    "medium_ph": wg.SOURCED_VALUES["medium_ph"],
    "temperature_c": wg.SOURCED_VALUES["temperature_c"],
    "duration_hours": wg.SOURCED_VALUES["duration_hours"],
    "aerobicity": wg.SOURCED_VALUES["aerobicity"],
    "n_samples": wg.SOURCED_VALUES["n_samples"],
    "uncertainty_type": wg.SOURCED_VALUES["uncertainty_type"],
    "uncertainty_type_corroboration": wg.SOURCED_VALUES[
        "uncertainty_type_corroboration"
    ],
    "independent_cultures": wg.SOURCED_VALUES["culture_format"],
    "growth_inhibition_definition": wg.SOURCED_VALUES["growth_inhibition_definition"],
}

#: The one sourced value this module adds: what the no-isoprenol column IS. The
#: growth-inhibition definition names it as the DENOMINATOR of the paper's own
#: normalization, which is the statement that it is an independently grown plain-medium
#: culture and not a derived quantity.
NO_ISOPRENOL_IS_THE_DENOMINATOR = SourcedValue(
    value=FITNESS_DEFINITION,
    provenance=wg.SOURCED_VALUES["growth_inhibition_definition"].provenance,
    quote=wg.SOURCED_VALUES["growth_inhibition_definition"].quote,
    note="the paper's only defined normalization of this column divides BY it, never "
    "toward a plate reference, so the column is an absolute endpoint OD600 of a "
    "separately grown culture ('4 mL of fresh media with (or without) isoprenol'). This "
    "module stores it as a ratio to the BW25113 row of the SAME column, which is the "
    "deletion-over-parent comparison FitnessPhenotype carries",
)
SOURCED_VALUES["no_isoprenol_column"] = NO_ISOPRENOL_IS_THE_DENOMINATOR


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def environment() -> Environment:
    """Plain 2YT at pH 7.0 and 30 C, shaken for 12 h: the no-isoprenol arm.

    This is ``wang2015.environment()`` MINUS the isoprenol
    ``SmallMoleculePerturbation``. The pH perturbation keeps its agent gap, because the
    paper names no acid or base ("adjusted to pH 7.0").
    """
    return Environment(
        media=YT_2X,
        temperature=Temperature(value=float(wg.TEMPERATURE_C.value)),
        perturbations=[
            EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.ph,
                magnitude=Concentration(
                    value=float(wg.MEDIUM_PH.value), unit=ConcentrationUnit.ph
                ),
                provenance_gaps=[wg.PH_AGENT_GAP],
            )
        ],
        aerobicity=str(wg.AEROBICITY.value),
        duration_hours=float(wg.DURATION_HOURS.value),
    )


def standard_error(sd: float) -> float:
    """The standard error of a mean of ``N_SAMPLES`` released replicates."""
    return sd / math.sqrt(N_SAMPLES)


def fitness_phenotype(row: wg.TableS3Row, parent: wg.TableS3Row) -> FitnessPhenotype:
    """One deletion strain's plain-medium OD600 as a ratio to the BW25113 row.

    ``fitness_uncertainty`` is the released SD divided by the parent mean, the sample
    standard deviation of the n released ratio observations; ``fitness_se`` is the
    delta-method standard error of the ratio, which also carries the parent's own
    spread. The second is never smaller than what the first derives, and the build
    refuses the other direction.
    """
    fitness = row.od600_without / parent.od600_without
    uncertainty = row.sd_without / parent.od600_without
    propagated = fitness * math.sqrt(
        (standard_error(row.sd_without) / row.od600_without) ** 2
        + (standard_error(parent.sd_without) / parent.od600_without) ** 2
    )
    conditioned = uncertainty / math.sqrt(N_SAMPLES)
    if propagated < conditioned:
        raise RuntimeError(
            f"{row.strain}: the propagated SE {propagated} is smaller than the "
            f"parent-conditioned {conditioned}"
        )
    return FitnessPhenotype(
        fitness=fitness,
        fitness_se=propagated,
        fitness_uncertainty=uncertainty,
        fitness_uncertainty_type=UncertaintyType.sample_sd,
        n_samples=N_SAMPLES,
        sample_unit=SampleUnit.biological_replicate,
    )


def reference_phenotype(parent: wg.TableS3Row) -> FitnessPhenotype:
    """BW25113 against itself: 1.0, with its own relative spread."""
    return FitnessPhenotype(
        fitness=1.0,
        fitness_uncertainty=parent.sd_without / parent.od600_without,
        fitness_uncertainty_type=UncertaintyType.sample_sd,
        n_samples=N_SAMPLES,
        sample_unit=SampleUnit.biological_replicate,
    )


def build_experiment(
    dataset_name: str,
    identity: wg.StrainIdentity,
    parent: wg.TableS3Row,
    env: Environment,
) -> BacterialFitnessExperiment:
    """The record of one Keio strain in plain 2YT."""
    return BacterialFitnessExperiment(
        dataset_name=dataset_name,
        genotype=wg.deletion_genotype(identity),
        environment=env,
        phenotype=fitness_phenotype(identity.row, parent),
    )


def build_reference(
    dataset_name: str, genome_reference: Any, env: Environment, parent: wg.TableS3Row
) -> BacterialFitnessExperimentReference:
    """BW25113 in the same plain medium: the denominator of every stored ratio."""
    return BacterialFitnessExperimentReference(
        dataset_name=dataset_name,
        genome_reference=genome_reference,
        environment_reference=env.model_copy(),
        phenotype_reference=reference_phenotype(parent),
    )


class FitnessRange(BaseModel):
    """The recomputed span of the stored ratios, checked against the declared one."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_records: int
    minimum: float
    maximum: float
    median: float
    n_non_positive: int = Field(
        description="ratios the FitnessPhenotype clamp would destroy; must be 0"
    )
    n_zero_sd: int = Field(
        description="released no-isoprenol SD cells that are exactly 0"
    )


def check_fitness_range(
    mutants: Sequence[wg.TableS3Row], parent: wg.TableS3Row
) -> FitnessRange:
    """Recompute the ratio span and refuse a clamp, a zero SD or a drifted range.

    The declared ``MIN_FITNESS`` / ``MAX_FITNESS`` are what makes the "clamp never
    fires" claim a checked oracle rather than a sentence: a re-extracted table whose
    ratios moved stops the build instead of silently clamping a record to 0.0.
    """
    ratios = sorted(row.od600_without / parent.od600_without for row in mutants)
    span = FitnessRange(
        n_records=len(ratios),
        minimum=ratios[0],
        maximum=ratios[-1],
        median=(ratios[len(ratios) // 2 - 1] + ratios[len(ratios) // 2]) / 2
        if len(ratios) % 2 == 0
        else ratios[len(ratios) // 2],
        n_non_positive=sum(1 for ratio in ratios if ratio <= 0.0),
        n_zero_sd=sum(1 for row in mutants if row.sd_without == 0.0),
    )
    if span.n_non_positive:
        raise RuntimeError(
            f"{span.n_non_positive} released ratios are non-positive, which "
            "FitnessPhenotype would clamp to 0.0 and destroy"
        )
    if span.n_zero_sd:
        raise RuntimeError(
            f"{span.n_zero_sd} released no-isoprenol SD cells are exactly 0, which is "
            "not a measured dispersion"
        )
    if (
        abs(span.minimum - MIN_FITNESS) > FITNESS_RANGE_ATOL
        or abs(span.maximum - MAX_FITNESS) > FITNESS_RANGE_ATOL
    ):
        raise RuntimeError(
            f"the released ratios span {span.minimum} to {span.maximum}; the module "
            f"declares {MIN_FITNESS} to {MAX_FITNESS}"
        )
    return span


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class GrowthWang2015Dataset(ExperimentDataset):
    """Plain-medium growth of 46 Keio transporter deletions (Wang 2015, Table S3)."""

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = "BW25113"

    def __init__(
        self,
        root: str = DATASET_ROOT_REL,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        ecoli_genome: EcoliK12Genome | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; ``ecoli_genome`` is injected by the build entry points."""
        self.ecoli_genome = ecoli_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialFitnessExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialFitnessExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Every mirrored raw file, the same list ``wang2015`` declares."""
        return [f.name for f in wg.RAW_FILES]

    def download(self) -> None:
        """Link each mirror file into ``raw/`` after checking the manifest and sha256.

        The pins, the mirror layout and the retrieval recipe are ``wang2015``'s; this
        reads the same files from the same place under the same hashes.
        """
        data_root = os.environ["DATA_ROOT"]
        manifest = wg.load_manifest(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        for raw in wg.RAW_FILES:
            check_manifest_pin(
                raw.mirror_relpath,
                wg.manifest_sha256(manifest, raw.mirror_relpath),
                raw.sha256,
            )
            src = wg.raw_mirror_dir(data_root) / raw.mirror_relpath
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            link_verified(src, osp.join(self.raw_dir, raw.name), raw.sha256)
        log.info("Wang 2015 raw files linked into %s (sha256 verified)", self.raw_dir)

    def _raw(self, name: str) -> str:
        """Absolute path of one linked raw file."""
        return osp.join(self.raw_dir, name)

    def _genome(self) -> EcoliK12Genome:
        """The injected genome, or the reference strain's default cache."""
        if self.ecoli_genome is None:
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        expected = BACTERIAL_ASSEMBLY_SETS[self.REFERENCE_STRAIN]
        if self.ecoli_genome.ASSEMBLY_SET != expected:
            raise ValueError(
                f"{type(self).__name__} needs the {expected} genome, got "
                f"{self.ecoli_genome.ASSEMBLY_SET}"
            )
        return self.ecoli_genome

    @post_process
    def process(self) -> None:
        """Parse Table S3's no-isoprenol column into 46 fitness records + the LMDB."""
        verify_raw_files(self.raw_dir, wg.DATA_SHA256)
        text = wg.layout_text(self._raw(wg.SI_PDF))
        entries = wg.parse_table_s2(
            wg.table_section(text, wg.TABLE_S2_MARKER, wg.TABLE_S3_MARKER)
        )
        rows = wg.parse_table_s3(
            wg.table_section(text, wg.TABLE_S3_MARKER, wg.TABLE_S4_MARKER)
        )
        digest = wg.table_s3_digest(rows)
        if digest != wg.TABLE_S3_SHA256:
            raise wg.TableExtractionError(
                f"parsed Table S3 sha256 {digest}, pinned {wg.TABLE_S3_SHA256}"
            )
        if len(rows) != EXPECTED_SOURCE_ROWS:
            raise wg.TableExtractionError(
                f"Table S3 holds {len(rows)} rows, the module declares "
                f"{EXPECTED_SOURCE_ROWS}"
            )
        parent, mutants = wg.split_parent(rows)
        keio = wg.keio_entries_for(mutants, entries)
        span = check_fitness_range(mutants, parent)

        genome = self._genome()
        resolved, unresolved, ledger = wg.resolve_strains(
            genome, mutants, keio, label=self.name
        )
        drop_log = wg.DropLog(
            dataset=self.name,
            table_rows=len(rows),
            reference_rows=[PARENT_STRAIN],
            source_records=len(mutants),
            kept_records=len(resolved),
            dropped_records=len(mutants) - len(resolved),
            rules=[unresolved],
        )
        if sum(rule.n_records for rule in drop_log.rules) != drop_log.dropped_records:
            raise RuntimeError("drop rules do not account for every dropped strain")
        if drop_log.kept_records != EXPECTED_RECORDS:
            raise RuntimeError(
                f"{drop_log.kept_records} records survive strain resolution, the module "
                f"declares {EXPECTED_RECORDS}"
            )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env = environment()
        reference = build_reference(
            self.name, assembly_reference(self.REFERENCE_STRAIN), env, parent
        )
        env_out, interned_env = self._open_write_lmdb(
            osp.join(self.processed_dir, "lmdb")
        )
        with env_out.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, identity in enumerate(tqdm(resolved, desc="wang2015 growth")):
                experiment = build_experiment(self.name, identity, parent, env)
                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, wg.PUBLICATION, itxn),
                )
        env_out.close()
        interned_env.close()

        self._write_ledgers(drop_log, ledger, digest, parent, resolved, span)
        log.info(
            "Wang 2015 Table S3 no-isoprenol column: %d records (+ %d %s reference); "
            "fitness %.6f to %.6f, median %.6f",
            drop_log.kept_records,
            EXPECTED_REFERENCES,
            PARENT_STRAIN,
            span.minimum,
            span.maximum,
            span.median,
        )

    def _write_ledgers(
        self,
        drop_log: wg.DropLog,
        ledger: wg.IdentifierLedger,
        digest: str,
        parent: wg.TableS3Row,
        resolved: Sequence[wg.StrainIdentity],
        span: FitnessRange,
    ) -> None:
        """The drop log, the identifier ledger, the extraction record and the table."""
        out = Path(self.preprocess_dir)
        (out / "dropped_records.json").write_text(drop_log.model_dump_json(indent=2))
        (out / "identifier_reconciliation.json").write_text(
            ledger.model_dump_json(indent=2)
        )
        (out / "extraction.json").write_text(
            json.dumps(
                {
                    "si_pdf_sha256": wg.SI_PDF_SHA256,
                    "pdftotext": wg.pdftotext_version(),
                    "pdftotext_args": list(wg.PDFTOTEXT_ARGS),
                    "table_s3_sha256": digest,
                    "column": COLUMN,
                    "fitness_definition": FITNESS_DEFINITION,
                    "parent_od600_without": parent.od600_without,
                    "parent_sd_without": parent.sd_without,
                    "n_samples": N_SAMPLES,
                    "fitness_range": span.model_dump(),
                },
                indent=2,
            )
        )
        (out / "not_stored.json").write_text(
            json.dumps(
                {
                    "dataset": self.name,
                    "issue": 826,
                    "items": [
                        {
                            "released": "Table S3 '0.5% (v/v) isoprenol' OD600 and SD",
                            "n_values": len(resolved) + 1,
                            "why_not_stored": "that column is the isoprenol arm, served "
                            "as EnvChemgenWang2015Dataset's log2 relative tolerance; "
                            "storing it here would be the same measurement twice",
                        }
                    ],
                },
                indent=2,
            )
        )
        (out / "sourced_values.json").write_text(
            json.dumps(
                {
                    name: value.model_dump(mode="json")
                    for name, value in SOURCED_VALUES.items()
                },
                indent=2,
            )
        )
        table = [
            {
                "record": None,
                "strain": parent.strain,
                "keio_token": None,
                "locus_tag": None,
                "symbol": None,
                "od600_without": parent.od600_without,
                "sd_without": parent.sd_without,
                "fitness": 1.0,
                "fitness_uncertainty": parent.sd_without / parent.od600_without,
                "fitness_se": None,
            }
        ] + [
            {
                "record": index,
                "strain": identity.row.strain,
                "keio_token": identity.keio_token,
                "locus_tag": identity.locus_tag,
                "symbol": identity.symbol,
                "od600_without": identity.row.od600_without,
                "sd_without": identity.row.sd_without,
                "fitness": fitness_phenotype(identity.row, parent).fitness,
                "fitness_uncertainty": fitness_phenotype(
                    identity.row, parent
                ).fitness_uncertainty,
                "fitness_se": fitness_phenotype(identity.row, parent).fitness_se,
            }
            for index, identity in enumerate(resolved)
        ]
        pd.DataFrame(table).astype({"record": "Int64"}).to_csv(
            out / "no_isoprenol.csv", index=False
        )

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled by ``build_experiment``."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# L0-L4 verification
# --------------------------------------------------------------------------- #
VERIFIER_PROVENANCE = Provenance(
    source_uri=f"{wg.SI_SOURCE_URL} (Table S3, 'No isoprenol' column)",
    citation_key=wg.CITATION_KEY,
    sha256=wg.SI_PDF_SHA256,
    method=FITNESS_DEFINITION,
    page="Sci Rep 5:16505, Supplementary Table S3",
)


def verify_build(
    dataset_root: str,
    data_root: str | None = None,
    *,
    genome: EcoliK12Genome | None = None,
    expected_count: int = EXPECTED_RECORDS,
) -> VerificationReport:
    """Run the fitness L0-L4 gate over a built tree and write the report.

    The gene universe and the canonical-name resolver are the record's own pinned
    BW25113 assembly, not the yeast default, which is what the landed Campos 2018 and
    Schmidt 2016 Table S24 loaders do for the same collection. Every ``SOURCED_VALUES``
    entry is additionally audited against the literature mirror, as ``wang2015``'s own
    verifier does.
    """
    from torchcell.datasets.bacteria_common import bacterial_genome
    from torchcell.verification.fitness import verify_fitness_dataset
    from torchcell.verification.runners import load_records
    from torchcell.verification.sourced import audit_sourced_value

    records = load_records(dataset_root)
    if genome is None:
        genome = bacterial_genome(
            "ecoli", GrowthWang2015Dataset.REFERENCE_STRAIN, data_root
        )
    report = verify_fitness_dataset(
        records,
        dataset_name=osp.basename(osp.normpath(dataset_root)),
        provenance=VERIFIER_PROVENANCE,
        expected_count=expected_count,
        sgd_genes=set(genome.genbank.loci),
        gene_universe_label=f"BW25113 ({genome.ASSEMBLY_SET})",
        resolve_gene_name=genome.resolve_gene_name,
    )
    report.add(l2_se_is_never_the_conditioned_one(records))
    if data_root is not None:
        library = Path(data_root) / "torchcell-library"
        for value in SOURCED_VALUES.values():
            report.add(audit_sourced_value(value, library))
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def l2_se_is_never_the_conditioned_one(
    records: Sequence[Mapping[str, Any]],
) -> LevelResult:
    """L2: every stored ``fitness_se`` is at least the parent-conditioned derivation.

    The conditioned derivation (``fitness_uncertainty / sqrt(n)``) treats the parent
    mean as a fixed denominator and so understates the spread. The build asserts the
    direction record by record; this rule re-asserts it on the STORED numbers, so a
    re-serialized store cannot carry the optimistic SE.
    """
    optimistic: list[str] = []
    for record in records:
        phenotype = record["experiment"]["phenotype"]
        stored = phenotype["fitness_se"]
        derived = phenotype["fitness_uncertainty"] / math.sqrt(phenotype["n_samples"])
        if stored + 1e-12 < derived:
            optimistic.append(f"{stored} < {derived}")
    return LevelResult(
        level=Level.L2,
        name="se_is_not_the_conditioned_derivation",
        passed=not optimistic,
        message=(
            f"all {len(records)} stored fitness_se values are at least the "
            "parent-conditioned derivation"
            if not optimistic
            else f"{len(optimistic)} records store an SE below the conditioned "
            f"derivation: {optimistic[:5]}"
        ),
        details={"n_records": len(records), "optimistic": optimistic[:20]},
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: ``build`` the dev-tree LMDB, or ``verify`` an already built one."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        prog="python -m torchcell.datasets.ecoli.wang2015_growth"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="build (or load) the dev-tree LMDB")
    sub.add_parser("verify", help="run L0-L4 on the built dev-tree LMDB")
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    root = osp.join(data_root, DATASET_ROOT_REL)
    if args.command == "build":
        dataset = GrowthWang2015Dataset(root=root)
        print(f"len = {len(dataset)}")
        print(Path(root, "preprocess", "extraction.json").read_text())
        return 0
    report = verify_build(root, data_root)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
