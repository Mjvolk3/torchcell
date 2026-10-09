# torchcell/datasets/private_torchcell/volk2021_inhibitor_bioscreen.py
# [[torchcell.datasets.private_torchcell.volk2021_inhibitor_bioscreen]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/private_torchcell/volk2021_inhibitor_bioscreen.py
# Test file: tests/torchcell/datasets/private_torchcell/test_volk2021_inhibitor_bioscreen.py
"""The 2021 Bioscreen C inhibitor runs on strain bAID, one record per inoculated well.

PRIVATE (``visibility = Visibility.private``): in-house measurements behind the 2021
preliminary exam (library key ``volkPreliminaryExamReport2021``), never published in a
``tc-data`` release and served only by a build run with ``--include-private``.

Five runs (:mod:`torchcell.datasets.private_torchcell.bioscreen`): ex21, the six
single-inhibitor titrations; ex23, the 63 combinations at one dose each; ex26, ex27 and
ex28, the furfural, formic acid and 5-HMF isoboles against acetic acid. Every inoculated
well the plate layout assigns is a record, the uninhibited (``WT``) wells included, so
each run's baseline is served with its replicates.

RECORD SHAPE. ``StrainEnvironmentResponseExperiment`` with

- genotype: ``Genotype(perturbations=[])``. Every well is the same strain, the CRISPR-AID
  host bAID (the thesis calls it ``BY4742-iAID6``; ``volk2021_sources.STRAIN_NAME`` and
  ``BAID_CONSTRUCTION``), so there is no edit relative to the background, and the
  background rides on the reference's ``StrainReferenceGenome(strain="bAID",
  background=baid_background())``: BY4742's four auxotrophies plus the integrated
  ``Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]`` cassette.
- environment: ``CultureEnvironment`` of liquid YPD (``YPD_LIQUID`` restated with the
  archive's own YPD statements; the recipe is ``MEDIUM_RECIPE_GAP``) carrying one
  ``SmallMoleculePerturbation`` per inhibitor present, at the DISPENSED dose in g/L
  (``bioscreen.layout``: ex23 at ``EX23_G_PER_L``, ex21 from the dispensed volumes with
  the FF 18 and FA 0.125 corrections, the isoboles by the 039 grid rule), basis
  ``fixed``. 30.0 C, the logged tray temperature (median 30.01 C on every run,
  ``TRAY_TEMPERATURE_C``; the set point is ``TEMPERATURE_SET_POINT_GAP``). The run
  length (``RUN_EVENTS``: 71.97, 84.97, 95.99, 95.98, 95.98 h) is ``duration_hours``, so
  each run is its own environment and its own reference. Aerobic: the Bioscreen reads
  wells under a lid in room air; no mirrored source states the oxygen regime, and
  ``aerobicity`` is a required string that cannot carry a gap.
- phenotype: ``EnvironmentResponsePhenotype``, ``assay_type=liquid_od_growth``. A grown
  well is ``relative_growth_rate`` = the run's mean WT generation time / the well's
  (``bioscreen.relative_growth_rate``; the raw-curve derivation, the same method on all
  five runs). A well that never grew is ``categorical`` / ``severely_reduced`` ("growth
  or signal essentially abolished"), ``category_label`` "no growth within <h> h",
  ``environment_response=None``. ``n_samples=1`` (one well), ``screen_id`` =
  ``<run>:well<n>``. The Bioscreen software's trait calls are a documented check in the
  note, never a served value.
- reference: the run's inhibitor-free environment, ``relative_growth_rate`` 1.0 with the
  sample SD of the WT wells' own relative growth rates and their count. Those wells are
  ``biological_replicate`` samples when every grown WT well comes from a different
  biological replicate (the isoboles: one per plate), else ``technical_replicate``
  (ex21's 17 grown WT wells and ex23's 8 come from three pre-cultures).

RECORDS DROPPED (``preprocess/dropped_records.json``): ex23's three uninoculated
``blank`` wells, and the wells of a run's export the layout assigns to no condition
(ex21's 91-100 and 191-200).

Rebuild: a NEW private dataset in a new class (``CultureEnvironment`` family, new
``relative_growth_rate`` / ``preliminary_report`` vocabulary from PR #778), so the served
graph gains it through a full rebuild run with ``--include-private``.
"""

from __future__ import annotations

import json
import logging
import os
import os.path as osp
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from torchcell.data import (
    ExperimentDataset,
    Visibility,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import YPD_LIQUID, restated
from torchcell.datamodels.schema import (
    AssayType,
    Compound,
    Concentration,
    ConcentrationUnit,
    CultureEnvironment,
    CultureFormat,
    DoseBasis,
    EndpointRule,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    Media,
    Publication,
    ResponseCategory,
    SampleUnit,
    SmallMoleculePerturbation,
    SourceType,
    StrainEnvironmentResponseExperiment,
    StrainEnvironmentResponseExperimentReference,
    StrainReferenceGenome,
    Temperature,
    UncertaintyType,
)
from torchcell.datamodels.strain_background import (
    BAID_GENOTYPE,
    BAID_STRAIN,
    baid_background,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.private_torchcell import bioscreen as b
from torchcell.datasets.private_torchcell import volk2021_sources as s
from torchcell.literature.manifest import Manifest
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)
from torchcell.verification.sourced import SourcedValue

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Raw files the build consumes, pinned to the raw-mirror manifest
# --------------------------------------------------------------------------- #
#: archive-relative path -> sha256 of every raw-mirror file the build reads: the five
#: exports (traits) and the three tables the layouts come from (ex23's well map, the
#: ex21 stocks and titration volumes). Equal to the raw mirror's ``manifest.json``
#: records, which equal the archive's ``MANIFEST.tsv``.
CONSUMED_SHA256: dict[str, str] = {
    b.RAW_CSV["ex21"]: (
        "b4127ab7279ea2fc4d814f8eeb79f6ebd875d21bdc150b441960c5126f69bdf6"
    ),
    b.RAW_CSV["ex23"]: (
        "691e6e90b479789596c262a7aa9187d3c571b827c3f436c5c1dde8ef7d1624a9"
    ),
    b.RAW_CSV["ex26"]: (
        "f21eab6d29642732d80d0746bf8e98a4866b5c9656d6b2f94a75608c450cba1f"
    ),
    b.RAW_CSV["ex27"]: (
        "0dd71ff2cc2c827f6fbd3db715ce2ad3496f6012368a35c360306a429a17640f"
    ),
    b.RAW_CSV["ex28"]: (
        "4a8f0989e6960cb58c65e9337a482a50fc39e83b2e610df374bc34b4266204f7"
    ),
    b.EX23_PREPROCESSED: (
        "e96523b4a0c65522b78073ee016a4b0c4966e5022860a80c4dc4afdbed6faf44"
    ),
    b.INHIBITORS_XLSX: s.ARCHIVE_SHA256[b.INHIBITORS_XLSX],
    b.TITRATION_ARRAY_XLSX: s.ARCHIVE_SHA256[b.TITRATION_ARRAY_XLSX],
}

#: The runs a build covers, in record order.
RUNS: tuple[b.Run, ...] = tuple(b.Run)

#: The report PDF in the library mirror: the deposited document a record cites.
REPORT_PDF = "paper.pdf"

SPECIES = "Saccharomyces cerevisiae"
TEMPERATURE_C = 30.0
#: The readout definition every record carries in ``units``.
UNITS = (
    "relative growth rate = mean generation time of the run's grown uninhibited (WT) "
    "wells / this well's generation time; generation time = 1 / the largest "
    "least-squares slope of log2(OD600 - baseline) over any 3 h window whose readings "
    "all exceed baseline + 0.05, baseline = median of the first three readings; a well "
    "grew when its baseline-subtracted OD600 rose by at least 0.3"
)

#: Name passed to the compound-identity resolver for each inhibitor: the report's own
#: name (``INHIBITOR_LABELS``), except HMF, whose report name the table lacks and whose
#: identity the stock sheet's molar mass fixes (``HMF_IDENTITY``).
RESOLVER_NAME: dict[b.Inhibitor, str] = {
    b.Inhibitor.FF: s.INHIBITOR_LABELS.value["FF"],
    b.Inhibitor.AA: s.INHIBITOR_LABELS.value["AA"],
    b.Inhibitor.HMF: s.HMF_IDENTITY.value,
    b.Inhibitor.FA: s.INHIBITOR_LABELS.value["FA"],
    b.Inhibitor.LVA: s.INHIBITOR_LABELS.value["LVA"],
    b.Inhibitor.LA: s.INHIBITOR_LABELS.value["LA"],
}

#: The sourced initial OD600 of each run's culture.
_INITIAL_OD: dict[b.Run, SourcedValue] = {
    b.Run.ex21: s.EX21_INITIAL_OD,
    b.Run.ex23: s.EX23_INITIAL_OD,
    b.Run.ex26: s.ISOBOLE_INITIAL_OD,
    b.Run.ex27: s.ISOBOLE_INITIAL_OD,
    b.Run.ex28: s.ISOBOLE_INITIAL_OD,
}


# --------------------------------------------------------------------------- #
# Environment, phenotype, reference
# --------------------------------------------------------------------------- #
def medium() -> Media:
    """Liquid YPD with the archive's own statements that the runs are in YPD."""
    return restated(YPD_LIQUID, s.EX23_MEDIUM, s.ISOBOLE_MEDIUM)


def compound(inhibitor: b.Inhibitor) -> Compound:
    """The inhibitor's identity through the resolver, with lactic acid's typed gap.

    Lactic acid's resolver gap is replaced by ``LACTIC_ACID_IDENTITY_GAP``: the stock
    sheet does not state the enantiomer, which is terminal, not a pending curation.
    """
    resolved = resolved_compound(RESOLVER_NAME[inhibitor])
    if inhibitor is b.Inhibitor.LA:
        return resolved.model_copy(
            update={"provenance_gaps": [s.LACTIC_ACID_IDENTITY_GAP]}
        )
    return resolved


def small_molecule(
    inhibitor: b.Inhibitor, dose_g_per_l: float
) -> SmallMoleculePerturbation:
    """One inhibitor at its dispensed dose, g/L; the stock diluent is a gap."""
    return SmallMoleculePerturbation(
        compound=compound(inhibitor),
        concentration=Concentration(
            value=dose_g_per_l, unit=ConcentrationUnit.g_per_l, basis=DoseBasis.fixed
        ),
        solvent=None,
        provenance_gaps=[s.SOLVENT_GAP],
    )


def culture_format(run: b.Run) -> CultureFormat:
    """Bioscreen C wells read at fixed times; volume sourced for the isoboles only."""
    isobole = run in b.ISOBOLE_RUNS
    provenance = [s.INSTRUMENT, _INITIAL_OD[run], s.RUN_EVENTS[run.value]]
    if isobole:
        provenance += [s.ISOBOLE_WELL_UL, s.ISOBOLE_INOCULUM_UL]
    return CultureFormat(
        vessel="Bioscreen C 100-well plate",
        working_volume_ul=s.ISOBOLE_WELL_UL.value if isobole else None,
        shaking_rpm=None,
        inoculum_od600=_INITIAL_OD[run].value,
        endpoint=EndpointRule.fixed_duration,
        provenance=provenance,
        provenance_gaps=(
            [s.SHAKING_GAP] if isobole else [s.SHAKING_GAP, s.WELL_VOLUME_GAP]
        ),
    )


def environment(run: b.Run, doses: dict[b.Inhibitor, float]) -> CultureEnvironment:
    """YPD at 30 C for the run's length, with every inhibitor dosed above zero."""
    return CultureEnvironment(
        media=medium(),
        temperature=Temperature(value=TEMPERATURE_C),
        perturbations=[
            small_molecule(inhibitor, doses[inhibitor])
            for inhibitor in b.INHIBITORS
            if doses[inhibitor] > 0
        ],
        aerobicity="aerobic",
        duration_hours=s.RUN_EVENTS[run.value].value,
        duration_generations=None,
        culture_format=culture_format(run),
        pre_culture=None,
        auxotroph_supplements=None,
        provenance_gaps=[
            s.DURATION_GENERATIONS_GAP,
            s.PRE_CULTURE_GAP,
            s.AUXOTROPH_SUPPLEMENTS_GAP,
        ],
    )


def no_growth_label(run: b.Run) -> str:
    """``no growth within <run hours> h``, the run length rounded to the hour."""
    return f"no growth within {round(s.RUN_EVENTS[run.value].value)} h"


def phenotype(
    well: b.Well, wt_generation_time_h: float
) -> EnvironmentResponsePhenotype:
    """One well: its relative growth rate, or the no-growth call."""
    screen_id = f"{well.run.value}:well{well.well}"
    rate = b.relative_growth_rate(well, wt_generation_time_h)
    if rate is None:
        return EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.categorical,
            assay_type=AssayType.liquid_od_growth,
            environment_response=None,
            category=ResponseCategory.severely_reduced,
            category_label=no_growth_label(well.run),
            n_samples=1,
            sample_unit=SampleUnit.biological_replicate,
            units=UNITS,
            screen_id=screen_id,
        )
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.relative_growth_rate,
        assay_type=AssayType.liquid_od_growth,
        environment_response=rate,
        n_samples=1,
        sample_unit=SampleUnit.biological_replicate,
        units=UNITS,
        screen_id=screen_id,
    )


def genome_reference() -> StrainReferenceGenome:
    """bAID, haploid, with its typed background."""
    return StrainReferenceGenome(
        species=SPECIES,
        strain=BAID_STRAIN,
        ploidy="haploid",
        background=baid_background(),
    )


def reference_phenotype(
    wells: list[b.Well], wt_generation_time_h: float
) -> EnvironmentResponsePhenotype:
    """1.0, with the sample SD of the grown WT wells' relative growth rates.

    The unit is ``biological_replicate`` when each grown WT well comes from a different
    biological replicate, else ``technical_replicate``. Fewer than two grown WT wells
    leave no SD, which raises: every run here has at least two.
    """
    grown = [
        (w.biological_replicate_id, gt)
        for w in wells
        if w.is_wild_type and (gt := w.generation_time_h) is not None
    ]
    if len(grown) < 2:
        raise ValueError(f"{wells[0].run}: {len(grown)} grown WT wells, need >= 2")
    rates = [wt_generation_time_h / gt for _, gt in grown]
    replicates = {replicate for replicate, _ in grown}
    unit = (
        SampleUnit.biological_replicate
        if len(replicates) == len(grown)
        else SampleUnit.technical_replicate
    )
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.relative_growth_rate,
        assay_type=AssayType.liquid_od_growth,
        environment_response=1.0,
        environment_response_uncertainty=float(np.std(rates, ddof=1)),
        environment_response_uncertainty_type=UncertaintyType.sample_sd,
        n_samples=len(grown),
        sample_unit=unit,
        units=UNITS,
    )


def reference(
    run: b.Run, wells: list[b.Well], dataset_name: str
) -> StrainEnvironmentResponseExperimentReference:
    """The run's inhibitor-free YPD and its WT wells as the 1.0 baseline."""
    wt = b.wild_type_generation_time(wells)
    return StrainEnvironmentResponseExperimentReference(
        dataset_name=dataset_name,
        genome_reference=genome_reference(),
        environment_reference=environment(run, b.no_doses()),
        phenotype_reference=reference_phenotype(wells, wt),
    )


def publication() -> Publication:
    """The preliminary-exam report, identified by its deposited PDF and sha256."""
    return Publication(
        source_type=SourceType.preliminary_report,
        title=b.REPORT_TITLE,
        identifier=f"{REPORT_PDF} sha256:{b.REPORT_PDF_SHA256}",
        doi=None,
        pubmed_id=None,
    )


def check_library_manifest(manifest: Manifest) -> None:
    """The library mirror records the report the publication names, by title and hash."""
    if manifest.title != b.REPORT_TITLE:
        raise RuntimeError(
            f"library manifest title {manifest.title!r} != {b.REPORT_TITLE!r}"
        )
    recorded = {f.path: f.sha256 for f in manifest.files}
    check_manifest_pin(REPORT_PDF, recorded[REPORT_PDF], b.REPORT_PDF_SHA256)


# --------------------------------------------------------------------------- #
# Wells -> records
# --------------------------------------------------------------------------- #
def layout_wells(
    plate: b.PlateLayout, generation_times: dict[int, float | None]
) -> list[b.Well]:
    """Every layout well with its raw-curve generation time."""
    return [
        b.Well(
            **designed.model_dump(),
            generation_time_h=generation_times[designed.well],
            grew=generation_times[designed.well] is not None,
            trait_source=b.TraitSource.raw_curve,
        )
        for designed in plate.wells
    ]


class RunRecords(BaseModel):
    """One run's experiments, in well order, and the reference they all share."""

    model_config = ConfigDict(frozen=True)

    run: b.Run
    experiments: list[StrainEnvironmentResponseExperiment]
    reference: StrainEnvironmentResponseExperimentReference
    wt_generation_time_h: float


def run_records(run: b.Run, wells: list[b.Well], dataset_name: str) -> RunRecords:
    """One experiment per well (WT wells included) against the run's reference."""
    wt = b.wild_type_generation_time(wells)
    return RunRecords(
        run=run,
        experiments=[
            StrainEnvironmentResponseExperiment(
                dataset_name=dataset_name,
                genotype=Genotype(perturbations=[]),
                environment=environment(run, well.doses_g_per_l),
                phenotype=phenotype(well, wt),
            )
            for well in wells
        ],
        reference=reference(run, wells, dataset_name),
        wt_generation_time_h=wt,
    )


class DropRule(BaseModel):
    """One retention rule and the wells it removed (``<run>:well<n>``)."""

    rule: str
    description: str
    n_records: int
    items: list[str]


class DropLog(BaseModel):
    """The wells of the five exports and which ones became records."""

    dataset: str
    source_records: int
    kept_records: int
    dropped_records: int
    rules: list[DropRule]


def ex23_blank_wells(data_dir: Path) -> list[int]:
    """ex23 wells the well map names ``blank`` (uninoculated YPD)."""
    return sorted(row.well for row in b.read_ex23_map(data_dir) if row.name == b.BLANK)


def dropped_wells(
    run: b.Run,
    plate: b.PlateLayout,
    measured: dict[int, float | None],
    blanks: list[int],
) -> tuple[list[str], list[str]]:
    """``(blank wells, wells no condition is assigned to)`` as ``<run>:well<n>``.

    ``measured`` holds every well of the run's export; ``blanks`` the run's
    uninoculated wells (ex23's only). Every other exported well the layout does not
    list is unassigned.
    """
    assigned = {w.well for w in plate.wells}
    blank_items = [f"{run.value}:well{w}" for w in blanks]
    unassigned = [
        f"{run.value}:well{w}"
        for w in sorted(measured)
        if w not in assigned and w not in blanks
    ]
    return blank_items, unassigned


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@register_dataset
class InhibitorBioscreenVolk2021Dataset(ExperimentDataset):
    """bAID in YPD with dosed inhibitors: relative growth rate per Bioscreen C well."""

    visibility = Visibility.private
    #: Every record is the unedited bAID host; the gene set is legitimately empty.
    has_gene_perturbations = False

    def __init__(
        self,
        root: str = "data/torchcell/inhibitor_bioscreen_volk2021",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return StrainEnvironmentResponseExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return StrainEnvironmentResponseExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The consumed raw-mirror files, at their archive paths under ``raw/``."""
        return list(CONSUMED_SHA256)

    def download(self) -> None:
        """Link each consumed raw-mirror file into ``raw/`` after checking its pin.

        The raw mirror's ``manifest.json`` must record each file under the code's pin,
        and the file must hash to it; ``bioscreen.deposit_raw_mirror`` rebuilds the
        mirror from the thesis archive, which is never a build dependency.
        """
        mirror = b.raw_mirror_dir(os.environ["DATA_ROOT"])
        manifest = Manifest.model_validate_json((mirror / "manifest.json").read_text())
        recorded = {f.path: f.sha256 for f in manifest.files}
        for rel, pin in CONSUMED_SHA256.items():
            check_manifest_pin(rel, recorded[rel], pin)
            dest = osp.join(self.raw_dir, rel)
            os.makedirs(osp.dirname(dest), exist_ok=True)
            link_verified(mirror / rel, dest, pin)
        log.info("volk2021 raw files linked into %s (sha256 verified)", self.raw_dir)

    @post_process
    def process(self) -> None:
        """Derive every well's trait from its raw curve and write one record per well."""
        verify_raw_files(self.raw_dir, CONSUMED_SHA256)
        library = b.library_dir(os.environ["DATA_ROOT"])
        check_library_manifest(
            Manifest.model_validate_json((library / "manifest.json").read_text())
        )
        data_dir = Path(self.raw_dir)
        pub = publication()
        source_records = 0
        blank_items: list[str] = []
        unassigned_items: list[str] = []
        per_run: list[RunRecords] = []
        for run in RUNS:
            plate = b.layout(run, data_dir)
            measured = b.raw_curve_generation_times(run, data_dir)
            duration = b.run_duration_h(data_dir, run)
            if abs(duration - s.RUN_EVENTS[run.value].value) > 0.01:
                raise RuntimeError(
                    f"{run}: export spans {duration:.4f} h, RUN_EVENTS records "
                    f"{s.RUN_EVENTS[run.value].value} h"
                )
            source_records += len(measured)
            blank, unassigned = dropped_wells(
                run,
                plate,
                measured,
                ex23_blank_wells(data_dir) if run is b.Run.ex23 else [],
            )
            blank_items += blank
            unassigned_items += unassigned
            per_run.append(run_records(run, layout_wells(plate, measured), self.name))

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for records in per_run:
                for experiment in records.experiments:
                    txn.put(
                        f"{idx}".encode(),
                        self._intern_record(experiment, records.reference, pub, itxn),
                    )
                    idx += 1
        env.close()
        interned_env.close()

        rules = [
            DropRule(
                rule="blank_well",
                description="uninoculated YPD (ex23 well map 'blank'): no strain was "
                "inoculated, so the well measures no strain",
                n_records=len(blank_items),
                items=blank_items,
            ),
            DropRule(
                rule="well_not_in_layout",
                description="a well of the export that the run's plate layout assigns "
                "to no condition (ex21: wells 91-100 and 191-200 lie outside the six "
                "inhibitor blocks)",
                n_records=len(unassigned_items),
                items=unassigned_items,
            ),
        ]
        drop_log = DropLog(
            dataset=self.name,
            source_records=source_records,
            kept_records=idx,
            dropped_records=source_records - idx,
            rules=rules,
        )
        if sum(r.n_records for r in rules) != drop_log.dropped_records:
            raise RuntimeError(
                f"drop accounting mismatch: rules total "
                f"{sum(r.n_records for r in rules)}, {drop_log.dropped_records} missing"
            )
        with open(osp.join(self.preprocess_dir, "dropped_records.json"), "w") as handle:
            handle.write(drop_log.model_dump_json(indent=2))
        with open(
            osp.join(self.preprocess_dir, "wild_type_generation_time.json"), "w"
        ) as handle:
            json.dump(
                {r.run.value: r.wt_generation_time_h for r in per_run}, handle, indent=2
            )
        log.info("Wrote %d volk2021 Bioscreen records to LMDB", idx)

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# L0-L4 verification of a built store (#827)
#
# The dataset carries its own gate, as the bioproduction loaders do, instead of a row in
# ``runners.ENVIRONMENT_RESPONSE_DATASETS``: that registry's rules are written for a
# deletion-collection screen (one record per (ORF, condition), a reference centered on 0,
# one measurement type, a gene universe to contain) and FIVE of them do not describe this
# dataset, which is one unedited strain measured in 977 wells. The measurements that say
# so are in the dendron note's 2026.10.09 section.
# --------------------------------------------------------------------------- #
#: Records a build must produce per run: every inoculated well the layout assigns.
EXPECTED_RECORDS: dict[str, int] = {
    "ex21": 180,
    "ex23": 197,
    "ex26": 200,
    "ex27": 200,
    "ex28": 200,
}
#: Records per run whose readout is the no-growth CALL rather than a rate: the wells whose
#: raw curve never rose by ``bioscreen.GROWTH_RISE``.
EXPECTED_NO_GROWTH: dict[str, int] = {
    "ex21": 63,
    "ex23": 128,
    "ex26": 155,
    "ex27": 124,
    "ex28": 147,
}
#: Inhibitor-free (``WT``) wells per run: the baseline every rate of that run is taken
#: against, served as records of their own.
EXPECTED_WILD_TYPE_WELLS: dict[str, int] = {
    "ex21": 18,
    "ex23": 8,
    "ex26": 2,
    "ex27": 2,
    "ex28": 2,
}
#: L4, the independent second source: the fraction of wells on which the SERVED
#: raw-curve grew / did-not-grow call equals the Bioscreen software's, for the three runs
#: the software processed. Declared rather than floored -- observed must EQUAL these, so
#: a derivation change fails here instead of drifting under a threshold.
SOFTWARE_GREW_AGREEMENT: dict[str, float] = {
    "ex21": 0.977778,
    "ex23": 0.939086,
    "ex26": 0.985000,
}
#: Tolerance of the agreement oracles above (the values are pinned to 6 decimals).
AGREEMENT_TOL = 5e-6
#: The parent strain and the integrated cassette, split off the SOURCED genotype string
#: Lian 2019's Supplementary Table 11 gives for bAID
#: (``BY4742-Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]``), so the L3 row
#: checks the served background against the SOURCE and not against the constructor that
#: wrote it.
BAID_PARENT, BAID_CASSETTE_NAME = BAID_GENOTYPE.value.split("-", 1)


def _run_of(screen_id: str) -> str:
    """``ex23`` from ``ex23:well150``."""
    return screen_id.split(":well")[0]


def _per_run(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """The records of each run, keyed by the run of their ``screen_id``."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        run = _run_of(record["experiment"]["phenotype"]["screen_id"])
        grouped.setdefault(run, []).append(record)
    return grouped


def l1_wells_per_run(records: list[dict[str, Any]]) -> LevelResult:
    """L1: the records of each run are exactly the wells its layout assigns."""
    observed = {run: len(group) for run, group in _per_run(records).items()}
    holds = observed == EXPECTED_RECORDS
    return LevelResult(
        level=Level.L1,
        name="wells_per_run",
        passed=holds,
        message=(
            f"records per run {observed} == the layouts' wells"
            if holds
            else f"records per run {observed} != {EXPECTED_RECORDS}"
        ),
        details={"observed": observed, "expected": EXPECTED_RECORDS},
    )


def l2_readout_split(records: list[dict[str, Any]]) -> LevelResult:
    """L2: a record carries a rate or a no-growth call, never both or neither.

    The dataset's readout is MIXED by construction: a well that never grew has no
    generation time, so there is no rate to store and the call is the measurement. The
    per-run no-growth counts are oracles, and a record holding both a numeric response
    and a category (or neither) is a defect.
    """
    no_growth: dict[str, int] = {run: 0 for run in EXPECTED_NO_GROWTH}
    malformed: list[dict[str, Any]] = []
    for record in records:
        phenotype = record["experiment"]["phenotype"]
        run = _run_of(phenotype["screen_id"])
        rate = phenotype["environment_response"]
        category = phenotype.get("category")
        measurement = str(phenotype["measurement_type"])
        rated = measurement == MeasurementType.relative_growth_rate.value
        called = measurement == MeasurementType.categorical.value
        if rated and (rate is None or category is not None):
            malformed.append({"screen_id": phenotype["screen_id"], "rule": "rate"})
        if called and (rate is not None or category is None):
            malformed.append({"screen_id": phenotype["screen_id"], "rule": "call"})
        if not rated and not called:
            malformed.append({"screen_id": phenotype["screen_id"], "rule": measurement})
        if called:
            no_growth[run] += 1
    holds = not malformed and no_growth == EXPECTED_NO_GROWTH
    return LevelResult(
        level=Level.L2,
        name="readout_split",
        passed=holds,
        message=(
            f"{sum(no_growth.values())} no-growth calls and "
            f"{len(records) - sum(no_growth.values())} rates, split per run {no_growth}"
            if holds
            else f"{len(malformed)} records malformed; no-growth per run {no_growth} "
            f"vs declared {EXPECTED_NO_GROWTH}"
        ),
        details={
            "no_growth_per_run": no_growth,
            "expected_no_growth_per_run": EXPECTED_NO_GROWTH,
            "n_malformed": len(malformed),
            "malformed": malformed[:20],
        },
    )


def l3_reference_one(records: list[dict[str, Any]]) -> LevelResult:
    """L3: the reference is the run's own WT baseline, a relative growth rate of 1.0.

    The env-response verifier's ``reference_zero`` rule does not apply: this readout is a
    RATIO against the wild type measured in the same run, so its control is 1.0, not 0.
    The reference also carries the sample SD of the grown WT wells' own rates and their
    count, so the baseline's spread is served with it.
    """
    worst = 0.0
    counts: dict[str, int] = {}
    bad: list[str] = []
    for record in records:
        reference = record["reference"]["phenotype_reference"]
        run = _run_of(record["experiment"]["phenotype"]["screen_id"])
        value = reference["environment_response"]
        if value is None:
            bad.append(f"{run}: no reference response")
            continue
        worst = max(worst, abs(float(value) - 1.0))
        counts[run] = int(reference["n_samples"])
        if (
            reference["environment_response_uncertainty"] is None
            or float(reference["environment_response_uncertainty"]) <= 0.0
            or str(reference["environment_response_uncertainty_type"])
            != UncertaintyType.sample_sd.value
            or str(reference["measurement_type"])
            != MeasurementType.relative_growth_rate.value
        ):
            bad.append(f"{run}: reference carries no positive sample SD")
    holds = not bad and worst == 0.0
    return LevelResult(
        level=Level.L3,
        name="reference_one",
        passed=holds,
        message=(
            f"reference relative growth rate == 1.0 for all {len(records)} records, "
            f"with the sample SD of {counts} grown WT wells per run"
            if holds
            else f"max|v-1|={worst:.3g}; {len(set(bad))} reference defects"
        ),
        details={
            "worst_abs_deviation": worst,
            "n_wild_type_wells": counts,
            "defects": sorted(set(bad))[:20],
        },
    )


def l3_wild_type_wells(records: list[dict[str, Any]]) -> LevelResult:
    """L3: the only records with no environmental edit are the declared ``WT`` wells.

    Every other well carries at least one dosed inhibitor, and the inhibitor-free wells
    are served as records rather than collapsed into the reference, so a run's baseline
    keeps its replicates. Declared per run, so a layout change cannot quietly add or
    drop one.
    """
    observed = {run: 0 for run in EXPECTED_WILD_TYPE_WELLS}
    for record in records:
        if record["experiment"]["environment"]["perturbations"]:
            continue
        run = _run_of(record["experiment"]["phenotype"]["screen_id"])
        observed[run] = observed.get(run, 0) + 1
    holds = observed == EXPECTED_WILD_TYPE_WELLS
    return LevelResult(
        level=Level.L3,
        name="wild_type_wells",
        passed=holds,
        message=(
            f"{sum(observed.values())} inhibitor-free wells, {observed} per run; every "
            "other record carries a dosed inhibitor"
            if holds
            else f"inhibitor-free wells {observed} != declared "
            f"{EXPECTED_WILD_TYPE_WELLS}"
        ),
        details={"observed": observed, "expected": EXPECTED_WILD_TYPE_WELLS},
    )


def l3_no_growth_label(records: list[dict[str, Any]]) -> LevelResult:
    """L3: every no-growth call is ``severely_reduced`` labelled with its RUN's length.

    No run was 48 h; 48 h is the lag the Bioscreen software assigns to a well that never
    grew. A label naming the wrong window would read as an endpoint the run never had.
    """
    expected = {run.value: no_growth_label(run) for run in RUNS}
    observed: dict[str, set[str]] = {}
    wrong_category: list[str] = []
    for record in records:
        phenotype = record["experiment"]["phenotype"]
        if str(phenotype["measurement_type"]) != MeasurementType.categorical.value:
            continue
        run = _run_of(phenotype["screen_id"])
        observed.setdefault(run, set()).add(str(phenotype["category_label"]))
        if str(phenotype["category"]) != ResponseCategory.severely_reduced.value:
            wrong_category.append(phenotype["screen_id"])
    labels = {run: sorted(values) for run, values in sorted(observed.items())}
    holds = not wrong_category and all(
        labels.get(run) == [expected[run]] for run in expected
    )
    return LevelResult(
        level=Level.L3,
        name="no_growth_label",
        passed=holds,
        message=(
            f"every no-growth call is severely_reduced, labelled {labels}"
            if holds
            else f"{len(wrong_category)} wrong categories; labels {labels} != {expected}"
        ),
        details={
            "labels_per_run": labels,
            "expected_per_run": expected,
            "wrong_category": wrong_category[:20],
        },
    )


def l3_strain_background(records: list[dict[str, Any]]) -> LevelResult:
    """L3: every record's reference carries bAID with its typed pAID6 integration.

    The genotype is empty (no edit relative to the host), so the strain's identity lives
    entirely on the reference's background: BY4742 as the parent and the
    ``Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]`` cassette as a typed
    integration with its source quotes. If that object were dropped, the records would
    describe a wild-type BY4742 screen.
    """
    cassette = BAID_CASSETTE_NAME
    defects: list[str] = []
    for record in records:
        reference = record["reference"]["genome_reference"]
        background = reference["background"]
        integrations = [str(entry["name"]) for entry in background["integrations"]]
        if (
            str(reference["strain"]) != BAID_STRAIN
            or str(background["name"]) != BAID_STRAIN
            or list(background["parents"]) != [BAID_PARENT]
            or integrations != [cassette]
            or not background["integrations"][0]["provenance"]
        ):
            defects.append(record["experiment"]["phenotype"]["screen_id"])
    holds = not defects
    return LevelResult(
        level=Level.L3,
        name="strain_background",
        passed=holds,
        message=(
            f"all {len(records)} references carry {BAID_STRAIN} (parent {BAID_PARENT}) "
            f"with the integration {cassette!r}"
            if holds
            else f"{len(defects)} records lack the bAID background or its integration"
        ),
        details={
            "strain": BAID_STRAIN,
            "parent": BAID_PARENT,
            "integration": cassette,
            "defects": defects[:20],
        },
    )


def l4_software_agreement(records: list[dict[str, Any]], mirror: Path) -> LevelResult:
    """L4: the served grew / did-not-grow call against the Bioscreen software's.

    The software's own trait files are an INDEPENDENT derivation of the same curves (it
    worked from estimated cell counts through a calibration, this loader from the raw
    OD600), and it processed ex21, ex23 and ex26. Agreement is the cross-source row;
    the two disagree in SCALE on ex21 and ex23 (the note's table), which is why the
    served value is one source throughout and never a blend.
    """
    grouped = _per_run(records)
    observed: dict[str, float] = {}
    for run_name in SOFTWARE_GREW_AGREEMENT:
        software = b.software_generation_times(b.Run(run_name), mirror)
        group = grouped[run_name]
        agree = 0
        for record in group:
            phenotype = record["experiment"]["phenotype"]
            well = int(phenotype["screen_id"].split("well")[1])
            served_grew = (
                str(phenotype["measurement_type"])
                == MeasurementType.relative_growth_rate.value
            )
            if served_grew == (software[well] is not None):
                agree += 1
        observed[run_name] = agree / len(group)
    holds = all(
        abs(observed[run] - declared) <= AGREEMENT_TOL
        for run, declared in SOFTWARE_GREW_AGREEMENT.items()
    )
    return LevelResult(
        level=Level.L4,
        name="software_trait_agreement",
        passed=holds,
        message=(
            "the served grew/no-grew call equals the Bioscreen software's on "
            + ", ".join(f"{run} {value:.4f}" for run, value in sorted(observed.items()))
            if holds
            else f"agreement {observed} != declared {SOFTWARE_GREW_AGREEMENT}"
        ),
        details={"observed": observed, "declared": SOFTWARE_GREW_AGREEMENT},
    )


#: Where the records came from, recorded with the verification report.
VERIFICATION_PROVENANCE = Provenance(
    source_uri=(
        f"$DATA_ROOT/{b.RAW_DIR_REL} (the raw mirror of {b.ARCHIVE_ROOT}, every file "
        "checked against the archive's MANIFEST.tsv sha256 on deposit)"
    ),
    citation_key=b.CITATION_KEY,
    sha256=b.REPORT_PDF_SHA256,
    method=UNITS,
    page=(
        "the five Bioscreen C OD600 exports (bioscreen.RAW_CSV), the ex23 well map "
        "(MV_ex23_preprocessed.csv) and the ex21 stock + titration tables"
    ),
    retrieved=b.RETRIEVED_AT,
)


def verify_build(dataset_root: str, data_root: str | None = None) -> VerificationReport:
    """Run the L0-L4 gate on a built store; write ``preprocess/verification_report.json``.

    ``data_root`` locates the raw mirror the L4 row reads the Bioscreen software's trait
    files from (they are a cross-source oracle, not a build input, so the loader never
    links them into ``raw/``); it defaults to ``$DATA_ROOT``.
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType
    from torchcell.verification.levels import (
        l0_structural,
        l1_completeness,
        l1_count,
        l2_value_fidelity,
    )
    from torchcell.verification.runners import load_records

    records = load_records(dataset_root)
    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python
    mirror = b.raw_mirror_dir(data_root or os.environ["DATA_ROOT"])
    expected_ids = [
        f"{run.value}:well{well.well}"
        for run in RUNS
        for well in b.layout(run, Path(dataset_root) / "raw").wells
    ]
    rates = [
        record["experiment"]["phenotype"]["environment_response"]
        for record in records
        if record["experiment"]["phenotype"]["environment_response"] is not None
    ]
    report = VerificationReport(
        dataset_name=InhibitorBioscreenVolk2021Dataset.__name__,
        provenance=VERIFICATION_PROVENANCE,
    )
    report.add(l0_structural((record["experiment"] for record in records), validate))
    report.add(l1_count(len(records), sum(EXPECTED_RECORDS.values())))
    report.add(
        l1_completeness(
            [record["experiment"]["phenotype"]["screen_id"] for record in records],
            expected_ids,
            allow_extra=False,
        )
    )
    report.add(l1_wells_per_run(records))
    report.add(l2_value_fidelity(rates, minimum=0.0))
    report.add(l2_readout_split(records))
    report.add(l3_reference_one(records))
    report.add(l3_wild_type_wells(records))
    report.add(l3_no_growth_label(records))
    report.add(l3_strain_background(records))
    report.add(l4_software_agreement(records, mirror))
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main() -> None:
    """Build/load the dataset under ``$DATA_ROOT`` and verify it (L0-L4)."""
    from dotenv import load_dotenv

    load_dotenv()
    root = osp.join(
        os.environ["DATA_ROOT"], "data/torchcell/inhibitor_bioscreen_volk2021"
    )
    dataset = InhibitorBioscreenVolk2021Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])
    print(verify_build(root, os.environ["DATA_ROOT"]).summary())


if __name__ == "__main__":
    main()
