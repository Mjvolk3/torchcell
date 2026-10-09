# tests/torchcell/datasets/private_torchcell/test_volk2021_inhibitor_bioscreen.py
# [[tests.torchcell.datasets.private_torchcell.test_volk2021_inhibitor_bioscreen]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/private_torchcell/test_volk2021_inhibitor_bioscreen.py
"""The private 2021 Bioscreen dataset: synthetic records, the gates, the real runs.

A synthetic Bioscreen export (two WT wells doubling every 2.0 and 2.5 h, a furfural well
doubling every 4.0 h, a flat furfural well, and the export's own ``Blank`` column) goes
through ``bioscreen.read_raw`` -> ``generation_time`` -> the loader's ``layout_wells``
and ``run_records``, and the records are checked field by field against the doubling
times the curves were built with. Mirror-backed tests read the raw mirror under
``$DATA_ROOT/torchcell-raw/volkPreliminaryExamReport2021`` and skip without it; they pin
the counts the layouts give and the ex23 single-inhibitor means measured 2026-10-08.

Every record has an empty genotype, so the loader declares ``has_gene_perturbations =
False`` and a synthetic build serves its wells with an empty gene set
(``test_a_synthetic_build_serves_every_well_with_an_empty_gene_set``).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from torchcell.data import GenotypeAggregator, Visibility
from torchcell.datamodels.schema import (
    AssayType,
    Genotype,
    MeasurementType,
    ResponseCategory,
    SampleUnit,
    SmallMoleculePerturbation,
    SourceType,
    StrainEnvironmentResponseExperiment,
    UncertaintyType,
)
from torchcell.datamodels.strain_background import BAID_STRAIN
from torchcell.datasets.private_torchcell import bioscreen as b
from torchcell.datasets.private_torchcell import volk2021_inhibitor_bioscreen as v
from torchcell.datasets.private_torchcell import volk2021_sources as s
from torchcell.knowledge_graphs.dataset_adapter_map import (
    PRIVATE_DATASET_ADAPTER_MAP,
    PrivateDatasetRefused,
    build_adapter_map,
    dataset_adapter_map,
    refuse_private_datasets,
)
from torchcell.literature.manifest import ArtifactRecord, Manifest
from torchcell.sequence import GeneSet

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/scratch/projects/torchcell-scratch"))
MIRROR = b.raw_mirror_dir(DATA_ROOT)
needs_mirror = pytest.mark.skipif(
    not (MIRROR / "manifest.json").is_file(), reason="the raw mirror is not present"
)

HOURS: b.FloatArray = np.arange(0.0, 96.0, 0.25, dtype=np.float64)


def _exponential(
    doubling_h: float, hours: b.FloatArray = HOURS, onset_h: float = 4.0
) -> b.FloatArray:
    """Flat at 0.1 until ``onset_h``, then ``0.1 + 0.06 * 2**((t - onset)/d)``, capped."""
    rise = np.where(
        hours >= onset_h, 0.06 * 2.0 ** ((hours - onset_h) / doubling_h), 0.0
    )
    return np.minimum(0.1 + rise, 2.0)


def _write_export(
    path: Path, wells: dict[int, b.FloatArray], hours: b.FloatArray = HOURS
) -> None:
    """A Bioscreen C export: UTF-16LE, Label/Info lines, Time + Blank + wells 1..200."""
    columns = [str(w) for w in range(1, 201)]
    lines = [
        "\ufeffLabel," + ",".join('""' for _ in range(201)),
        "Info," + ",".join('""' for _ in range(201)),
        "Time,Blank," + ",".join(columns),
    ]
    for i, hour in enumerate(hours):
        h, rem = divmod(int(round(hour * 3600)), 3600)
        m, sec = divmod(rem, 60)
        ods = [wells.get(w, np.full_like(hours, 0.1))[i] for w in range(1, 201)]
        lines.append(
            f"{h:02d}:{m:02d}:{sec:02d},0.000," + ",".join(f"{od:.9f}" for od in ods)
        )
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-16le"))


def _layout(doses: dict[int, float]) -> b.PlateLayout:
    """Synthetic ex26 wells: ``{well: furfural g/L}``, 0 meaning WT."""
    wells = []
    for well, dose in sorted(doses.items()):
        d = b.no_doses() | {b.Inhibitor.FF: dose}
        wells.append(
            b.WellLayout(
                run=b.Run.ex26,
                plate=b.plate_of(well),
                well=well,
                condition=b.WILD_TYPE if dose == 0 else f"FF{dose:g}",
                doses_g_per_l=d,
                biological_replicate_id=b.plate_of(well),
            )
        )
    return b.PlateLayout(run=b.Run.ex26, wells=wells)


@pytest.fixture
def synthetic(tmp_path: Path) -> v.RunRecords:
    """Wells 1 and 101 WT (2.0 h, 2.5 h), well 2 furfural grown (4.0 h), well 3 flat."""
    export = tmp_path / "synthetic.csv"
    _write_export(
        export, {1: _exponential(2.0), 101: _exponential(2.5), 2: _exponential(4.0)}
    )
    raw = b.read_raw(export)
    # the Blank column is dropped
    assert [int(c) for c in raw.columns] == list(range(1, 201))
    hours = raw.index.to_numpy(dtype=np.float64)
    measured = {
        int(w): b.generation_time(hours, raw[w].to_numpy(dtype=np.float64))
        for w in raw.columns
    }
    plate = _layout({1: 0.0, 2: 1.0, 3: 1.0, 101: 0.0})
    return v.run_records(b.Run.ex26, v.layout_wells(plate, measured), "Synthetic")


# --------------------------------------------------------------------------- #
# Synthetic records
# --------------------------------------------------------------------------- #
def test_wild_type_mean_is_the_mean_of_the_two_doubling_times(
    synthetic: v.RunRecords,
) -> None:
    assert synthetic.wt_generation_time_h == pytest.approx(2.25, rel=1e-6)
    assert [e.phenotype.screen_id for e in synthetic.experiments] == [
        "ex26:well1",
        "ex26:well2",
        "ex26:well3",
        "ex26:well101",
    ]


def test_a_grown_inhibitor_well_is_its_relative_growth_rate(
    synthetic: v.RunRecords,
) -> None:
    grown = synthetic.experiments[1]
    assert isinstance(grown.genotype, Genotype)
    assert grown.genotype.perturbations == []
    (perturbation,) = grown.environment.perturbations
    assert isinstance(perturbation, SmallMoleculePerturbation)
    assert perturbation.concentration.unit is not None
    assert perturbation.concentration.basis is not None
    assert perturbation.compound.name == "furfural"
    assert perturbation.compound.inchikey == "HYBBIBNJHNGZAN-UHFFFAOYSA-N"
    assert perturbation.concentration.value == 1.0
    assert perturbation.concentration.unit.value == "g/L"
    assert perturbation.concentration.basis.value == "fixed"
    assert perturbation.solvent is None
    assert perturbation.gapped_fields() == {"solvent"}
    phenotype = grown.phenotype
    assert phenotype.measurement_type is MeasurementType.relative_growth_rate
    assert phenotype.assay_type is AssayType.liquid_od_growth
    assert phenotype.environment_response == pytest.approx(2.25 / 4.0, rel=1e-6)
    assert phenotype.category is None
    assert phenotype.n_samples == 1
    assert phenotype.sample_unit is SampleUnit.biological_replicate
    assert phenotype.units == v.UNITS


def test_a_flat_well_is_the_no_growth_call(synthetic: v.RunRecords) -> None:
    flat = synthetic.experiments[2].phenotype
    assert flat.measurement_type is MeasurementType.categorical
    assert flat.category is ResponseCategory.severely_reduced
    assert flat.category_label == "no growth within 96 h"
    assert flat.environment_response is None
    assert flat.assay_type is AssayType.liquid_od_growth
    assert flat.screen_id == "ex26:well3"


def test_a_wild_type_well_is_a_record_with_no_inhibitor(
    synthetic: v.RunRecords,
) -> None:
    wt = synthetic.experiments[0]
    assert wt.environment.perturbations == []
    assert wt.phenotype.environment_response == pytest.approx(2.25 / 2.0, rel=1e-6)
    assert wt.environment == synthetic.reference.environment_reference


def test_the_reference_is_bAID_and_the_wild_type_spread(
    synthetic: v.RunRecords,
) -> None:
    reference = synthetic.reference
    genome = reference.genome_reference
    assert genome.strain == BAID_STRAIN and genome.ploidy == "haploid"
    assert [c.name for c in genome.background.integrations] == [
        "Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]"
    ]
    phenotype = reference.phenotype_reference
    assert phenotype.environment_response == 1.0
    rates = [2.25 / 2.0, 2.25 / 2.5]
    assert phenotype.environment_response_uncertainty == pytest.approx(
        float(np.std(rates, ddof=1)), rel=1e-6
    )
    assert phenotype.environment_response_uncertainty_type is UncertaintyType.sample_sd
    assert phenotype.n_samples == 2
    # wells 1 and 101 sit on plates 1 and 2, two biological replicates
    assert phenotype.sample_unit is SampleUnit.biological_replicate
    environment = reference.environment_reference
    assert environment.temperature is not None
    assert environment.temperature.value == 30.0
    assert environment.duration_hours == 95.99
    assert environment.media.base_medium == "YPD"
    assert environment.culture_format is not None
    assert environment.culture_format.working_volume_ul == 200.0
    assert environment.culture_format.inoculum_od600 == 0.2
    assert environment.culture_format.gapped_fields() == {"shaking_rpm"}
    assert environment.gapped_fields() == {
        "duration_generations",
        "pre_culture",
        "auxotroph_supplements",
    }


def test_wild_type_wells_of_one_replicate_are_technical(
    synthetic: v.RunRecords,
) -> None:
    wells = [
        b.Well(
            **w.model_dump(),
            generation_time_h=2.0,
            grew=True,
            trait_source=b.TraitSource.raw_curve,
        )
        for w in _layout({1: 0.0, 11: 0.0}).wells
    ]
    reference = v.reference_phenotype(wells, 2.0)
    assert reference.sample_unit is SampleUnit.technical_replicate
    with pytest.raises(ValueError, match="1 grown WT wells, need >= 2"):
        v.reference_phenotype(wells[:1], 2.0)


def test_the_empty_genotype_round_trips_and_keys_in_the_genotype_aggregator(
    synthetic: v.RunRecords,
) -> None:
    """Every record's genotype is ``[]``: it validates, survives JSON, and keys.

    ``GenotypeAggregator`` keys the empty set to one bucket. The (genotype, environment)
    ``GenotypeEnvironmentAggregator`` lives on ``exp/033-env-chemgen-pooled`` and is not
    importable here.
    """
    experiment = synthetic.experiments[1]
    restored = StrainEnvironmentResponseExperiment.model_validate_json(
        experiment.model_dump_json()
    )
    assert restored == experiment
    aggregator = GenotypeAggregator.__new__(GenotypeAggregator)
    keys = {
        aggregator.aggregate_check(
            {"experiment": e, "experiment_reference": synthetic.reference}
        )
        for e in synthetic.experiments
    }
    assert keys == {hashlib.sha256(str([]).encode()).hexdigest()}
    raw = aggregator.aggregate_key_raw({"experiment": experiment.model_dump()})
    assert raw in keys


# --------------------------------------------------------------------------- #
# Identity, publication, visibility
# --------------------------------------------------------------------------- #
def test_compound_identities_and_their_gaps() -> None:
    names = {i: v.compound(i) for i in b.INHIBITORS}
    assert names[b.Inhibitor.HMF].name == "5-(hydroxymethyl)furfural"
    assert names[b.Inhibitor.HMF].inchikey == "NOEGNKMFWQHSLB-UHFFFAOYSA-N"
    assert names[b.Inhibitor.LVA].inchikey == "JOOXCMJARBKPKM-UHFFFAOYSA-N"
    assert names[b.Inhibitor.AA].inchikey == "QTBSBXVTEAMEQO-UHFFFAOYSA-N"
    assert names[b.Inhibitor.FA].inchikey is None
    assert [g.reason.value for g in names[b.Inhibitor.FA].provenance_gaps] == [
        "deferred_pending_source_review"
    ]
    assert names[b.Inhibitor.LA].inchikey is None
    assert names[b.Inhibitor.LA].provenance_gaps == [s.LACTIC_ACID_IDENTITY_GAP]


def test_the_publication_is_the_deposited_report() -> None:
    publication = v.publication()
    assert publication.source_type is SourceType.preliminary_report
    assert (
        publication.title == "Machine Learning for Engineering Improved Yeast Fitness"
    )
    assert publication.identifier == f"paper.pdf sha256:{b.REPORT_PDF_SHA256}"
    assert publication.doi is None and publication.pubmed_id is None


def _library_manifest(sha256: str, title: str) -> Manifest:
    return Manifest(
        citation_key=b.CITATION_KEY,
        doi=None,
        title=title,
        files=[ArtifactRecord(path="paper.pdf", role="pdf", bytes=1, sha256=sha256)],
        provenance_complete=True,
        created_at="2026-10-08T00:00:00+00:00",
    )


def test_the_library_manifest_must_record_the_cited_report() -> None:
    v.check_library_manifest(_library_manifest(b.REPORT_PDF_SHA256, b.REPORT_TITLE))
    with pytest.raises(RuntimeError, match="title"):
        v.check_library_manifest(_library_manifest(b.REPORT_PDF_SHA256, "Other"))
    with pytest.raises(Exception, match="paper.pdf"):
        v.check_library_manifest(_library_manifest("0" * 64, b.REPORT_TITLE))


def test_the_dataset_is_private_and_mapped_only_privately() -> None:
    cls = v.InhibitorBioscreenVolk2021Dataset
    assert cls.visibility is Visibility.private
    assert cls in PRIVATE_DATASET_ADAPTER_MAP
    assert cls not in dataset_adapter_map
    assert cls not in build_adapter_map()
    assert cls in build_adapter_map(include_private=True)
    with pytest.raises(
        PrivateDatasetRefused, match="InhibitorBioscreenVolk2021Dataset"
    ):
        refuse_private_datasets([cls])
    refuse_private_datasets([cls], include_private=True)


def test_no_growth_labels_round_the_run_length() -> None:
    assert [v.no_growth_label(run) for run in b.Run] == [
        "no growth within 72 h",
        "no growth within 85 h",
        "no growth within 96 h",
        "no growth within 96 h",
        "no growth within 96 h",
    ]


# --------------------------------------------------------------------------- #
# The dataset build (blocked) and the real runs
# --------------------------------------------------------------------------- #
def _synthetic_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A DATA_ROOT whose raw mirror holds one synthetic ex26 export and a manifest."""
    data_root = tmp_path / "data_root"
    rel = b.RAW_CSV["ex26"]
    export = b.raw_mirror_dir(data_root) / rel
    export.parent.mkdir(parents=True)
    # the export must span RUN_EVENTS' 95.99 h, which the build checks
    hours = np.append(HOURS, 95.99)
    curves = {w: _exponential(d, hours) for w, d in ((1, 2.0), (101, 2.5), (2, 4.0))}
    _write_export(export, curves, hours)
    sha = hashlib.sha256(export.read_bytes()).hexdigest()
    (b.raw_mirror_dir(data_root) / "manifest.json").write_text(
        Manifest(
            citation_key=b.CITATION_KEY,
            doi=None,
            title=b.REPORT_TITLE,
            files=[
                ArtifactRecord(
                    path=rel, role="raw_data", bytes=export.stat().st_size, sha256=sha
                )
            ],
            provenance_complete=True,
            created_at="2026-10-08T00:00:00+00:00",
        ).model_dump_json()
    )
    library = b.library_dir(data_root)
    library.mkdir(parents=True)
    (library / "manifest.json").write_text(
        _library_manifest(b.REPORT_PDF_SHA256, b.REPORT_TITLE).model_dump_json()
    )
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setattr(v, "RUNS", (b.Run.ex26,))
    monkeypatch.setattr(v, "CONSUMED_SHA256", {rel: sha})
    return data_root


def test_a_synthetic_build_serves_every_well_with_an_empty_gene_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every record is the unedited host, so the loader declares
    ``has_gene_perturbations = False`` and the build's gene set is empty by design.
    """
    _synthetic_data_root(tmp_path, monkeypatch)
    dataset = v.InhibitorBioscreenVolk2021Dataset(root=str(tmp_path / "build"))
    assert len(dataset) == 200
    assert v.InhibitorBioscreenVolk2021Dataset.has_gene_perturbations is False
    assert dataset.gene_set == GeneSet()
    assert all(
        dataset.transform_item(dataset[i])["experiment"].genotype.perturbations == []
        for i in range(len(dataset))
    )


@needs_mirror
def test_record_counts_per_run_from_the_layouts() -> None:
    """ex21 180 (6 inhibitors x 3 blocks x 10), ex23 197 (200 - 3 blanks), isoboles 200."""
    expected = {
        b.Run.ex21: (180, 117),
        b.Run.ex23: (197, 69),
        b.Run.ex26: (200, 45),
        b.Run.ex27: (200, 76),
        b.Run.ex28: (200, 53),
    }
    blank_items: list[str] = []
    unassigned: list[str] = []
    for run, (n, n_grown) in expected.items():
        plate = b.layout(run, MIRROR)
        measured = b.raw_curve_generation_times(run, MIRROR)
        records = v.run_records(run, v.layout_wells(plate, measured), "X")
        assert len(records.experiments) == n, run
        grown = [e for e in records.experiments if e.phenotype.environment_response]
        assert len(grown) == n_grown, run
        blanks = v.ex23_blank_wells(MIRROR) if run is b.Run.ex23 else []
        dropped = v.dropped_wells(run, plate, measured, blanks)
        blank_items += dropped[0]
        unassigned += dropped[1]
    assert blank_items == ["ex23:well80", "ex23:well87", "ex23:well157"]
    assert unassigned == [f"ex21:well{w}" for w in [*range(91, 101), *range(191, 201)]]


@needs_mirror
def test_ex23_single_inhibitor_means_from_the_served_records() -> None:
    """Mean relative growth rate of the three single-inhibitor wells, measured 2026-10-08."""
    plate = b.layout(b.Run.ex23, MIRROR)
    wells = v.layout_wells(plate, b.raw_curve_generation_times(b.Run.ex23, MIRROR))
    records = v.run_records(b.Run.ex23, wells, "X")
    measured = {}
    for inhibitor in b.INHIBITORS:
        values = [
            e.phenotype.environment_response
            for w, e in zip(wells, records.experiments, strict=True)
            if w.present() == [inhibitor]
        ]
        assert len(values) == 3 and None not in values, inhibitor
        measured[inhibitor.value] = float(np.mean([x for x in values if x is not None]))
    assert measured == pytest.approx(
        {
            "FF": 0.5069,
            "AA": 0.9980,
            "HMF": 0.4796,
            "FA": 0.9958,
            "LVA": 0.8689,
            "LA": 0.9824,
        },
        abs=5e-4,
    )
    assert records.wt_generation_time_h == pytest.approx(1.3453, abs=5e-4)


@needs_mirror
def test_the_mirror_manifest_records_every_consumed_pin() -> None:
    manifest = Manifest.model_validate_json((MIRROR / "manifest.json").read_text())
    recorded = {f.path: f.sha256 for f in manifest.files}
    assert {rel: recorded[rel] for rel in v.CONSUMED_SHA256} == v.CONSUMED_SHA256
    assert json.loads((MIRROR / "manifest.json").read_text())["title"] == (
        b.REPORT_TITLE
    )


# --------------------------------------------------------------------------- #
# 2026.10.09 - the L0-L4 gate the dataset carries (#827)
# --------------------------------------------------------------------------- #
def _synthetic_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Build the synthetic ex26 store and point every oracle of the gate at it.

    The synthetic export grows wells 1 and 101 (the two inhibitor-free grid wells) and
    well 2, so the run is 200 records, 197 no-growth calls, 2 wild-type wells, and a
    software trait file written to agree on every well makes the L4 row 1.0.
    """
    data_root = _synthetic_data_root(tmp_path, monkeypatch)
    root = tmp_path / "build"
    dataset = v.InhibitorBioscreenVolk2021Dataset(root=str(root))
    assert len(dataset) == 200
    dataset.close_lmdb()
    traits = b.raw_mirror_dir(data_root) / b.EX26_SOFTWARE_TRAITS
    traits.parent.mkdir(parents=True, exist_ok=True)
    grew = {1: 2.0, 101: 2.5, 2: 4.0}
    traits.write_text(
        "Container Name\tGT\n"
        + "".join(f"Well {w}\t{grew.get(w, float('nan'))}\n" for w in range(1, 201))
    )
    monkeypatch.setattr(v, "EXPECTED_RECORDS", {"ex26": 200})
    monkeypatch.setattr(v, "EXPECTED_NO_GROWTH", {"ex26": 197})
    monkeypatch.setattr(v, "EXPECTED_WILD_TYPE_WELLS", {"ex26": 2})
    monkeypatch.setattr(v, "SOFTWARE_GREW_AGREEMENT", {"ex26": 1.0})
    return root, data_root


def test_verify_build_covers_l0_to_l4_and_writes_its_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every row of the gate, in order, on a store built from a synthetic export."""
    from torchcell.verification.report import Level, VerificationReport

    root, data_root = _synthetic_verification(tmp_path, monkeypatch)
    report = v.verify_build(str(root), str(data_root))
    assert [(r.level.name, r.name, r.passed) for r in report.results] == [
        ("L0", "structural", True),
        ("L1", "count", True),
        ("L1", "completeness", True),
        ("L1", "wells_per_run", True),
        ("L2", "value_fidelity", True),
        ("L2", "readout_split", True),
        ("L3", "reference_one", True),
        ("L3", "wild_type_wells", True),
        ("L3", "no_growth_label", True),
        ("L3", "strain_background", True),
        ("L4", "software_trait_agreement", True),
    ]
    assert report.passed
    assert report.levels_covered == {Level.L0, Level.L1, Level.L2, Level.L3, Level.L4}
    written = root / "preprocess" / "verification_report.json"
    reread = VerificationReport.model_validate_json(written.read_text())
    assert reread == report
    assert reread.provenance.citation_key == b.CITATION_KEY
    assert reread.provenance.sha256 == b.REPORT_PDF_SHA256
    assert reread.provenance.method == v.UNITS


def test_the_software_agreement_row_is_an_oracle_and_not_a_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One well called differently by the software drops the agreement below the pin.

    The row fails on 0.995 against a declared 1.0, so a change in either derivation is
    reported rather than absorbed by a threshold.
    """
    root, data_root = _synthetic_verification(tmp_path, monkeypatch)
    traits = b.raw_mirror_dir(data_root) / b.EX26_SOFTWARE_TRAITS
    traits.write_text(traits.read_text().replace("Well 3\tnan", "Well 3\t5.0"))
    report = v.verify_build(str(root), str(data_root))
    (row,) = [r for r in report.results if r.name == "software_trait_agreement"]
    assert row.passed is False
    assert row.details["observed"] == {"ex26": 0.995}
    assert row.details["declared"] == {"ex26": 1.0}
    assert report.passed is False


def _records(root: Path) -> list[dict[str, Any]]:
    from torchcell.verification.runners import load_records

    return load_records(str(root))


def test_the_gate_fails_a_record_holding_both_a_rate_and_a_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L2 ``readout_split``: a no-growth call that also carries a number is a defect."""
    root, _ = _synthetic_verification(tmp_path, monkeypatch)
    records = _records(root)
    (called,) = [
        r
        for r in records
        if str(r["experiment"]["phenotype"]["measurement_type"])
        == MeasurementType.categorical.value
    ][:1]
    called["experiment"]["phenotype"]["environment_response"] = 0.5
    row = v.l2_readout_split(records)
    assert row.passed is False
    assert row.details["n_malformed"] == 1
    assert row.details["malformed"][0]["rule"] == "call"


def test_the_gate_fails_a_reference_centered_on_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L3 ``reference_one``: 0 is the log2-ratio convention, not this ratio's control."""
    root, _ = _synthetic_verification(tmp_path, monkeypatch)
    records = _records(root)
    records[0]["reference"]["phenotype_reference"]["environment_response"] = 0.0
    row = v.l3_reference_one(records)
    assert row.passed is False
    assert row.details["worst_abs_deviation"] == 1.0


def test_the_gate_fails_when_an_inhibitor_free_well_is_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L3 ``wild_type_wells``: the declared baseline wells must all be served."""
    root, _ = _synthetic_verification(tmp_path, monkeypatch)
    records = _records(root)
    kept = [
        r
        for r in records
        if r["experiment"]["environment"]["perturbations"]
        or r["experiment"]["phenotype"]["screen_id"] != "ex26:well1"
    ]
    row = v.l3_wild_type_wells(kept)
    assert row.passed is False
    assert row.details["observed"] == {"ex26": 1}
    assert row.details["expected"] == {"ex26": 2}


def test_the_gate_fails_a_no_growth_label_naming_the_wrong_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L3 ``no_growth_label``: the software's 48 h lag is not any run's length."""
    root, _ = _synthetic_verification(tmp_path, monkeypatch)
    records = _records(root)
    for record in records:
        phenotype = record["experiment"]["phenotype"]
        if str(phenotype["measurement_type"]) == MeasurementType.categorical.value:
            phenotype["category_label"] = "no growth within 48 h"
    row = v.l3_no_growth_label(records)
    assert row.passed is False
    assert row.details["labels_per_run"] == {"ex26": ["no growth within 48 h"]}
    assert row.details["expected_per_run"] == {"ex26": "no growth within 96 h"}


def test_the_gate_fails_a_background_without_the_baid_integration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L3 ``strain_background``: with an empty genotype, the cassette is the strain.

    The oracle is the genotype string Lian 2019's Supplementary Table 11 states, split
    into the parent and the integration, so dropping the cassette object leaves the
    records describing an unedited BY4742 and the row says so.
    """
    root, _ = _synthetic_verification(tmp_path, monkeypatch)
    records = _records(root)
    assert (v.BAID_PARENT, v.BAID_CASSETTE_NAME) == (
        "BY4742",
        "Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]",
    )
    assert v.l3_strain_background(records).passed is True
    records[0]["reference"]["genome_reference"]["background"]["integrations"] = []
    row = v.l3_strain_background(records)
    assert row.passed is False
    # the reference is INTERNED, so one object backs every record of the run: emptying
    # it on one record empties it for all 200, which is what the row reports.
    assert row.message == "200 records lack the bAID background or its integration"
    assert row.details["defects"][:2] == ["ex26:well1", "ex26:well2"]


@needs_mirror
def test_the_gate_oracles_equal_what_the_raw_mirror_gives() -> None:
    """The declared per-run counts and L4 agreements, re-derived from the raw files.

    The gate's oracles are numbers, so they can drift from the data without any test
    noticing. This derives all four from the mirror itself: the layouts' well counts,
    the wells whose raw curve never rose, the inhibitor-free wells, and the fraction of
    wells on which the raw-curve call equals the Bioscreen software's.
    """
    records_per_run: dict[str, int] = {}
    no_growth: dict[str, int] = {}
    wild_type: dict[str, int] = {}
    agreement: dict[str, float] = {}
    for run in b.Run:
        wells = v.layout_wells(
            b.layout(run, MIRROR), b.raw_curve_generation_times(run, MIRROR)
        )
        records_per_run[run.value] = len(wells)
        no_growth[run.value] = sum(1 for w in wells if w.generation_time_h is None)
        wild_type[run.value] = sum(1 for w in wells if not w.present())
        if run.value in v.SOFTWARE_GREW_AGREEMENT:
            software = b.software_generation_times(run, MIRROR)
            agree = sum(
                1
                for w in wells
                if (w.generation_time_h is not None) == (software[w.well] is not None)
            )
            agreement[run.value] = agree / len(wells)
    assert records_per_run == v.EXPECTED_RECORDS
    assert no_growth == v.EXPECTED_NO_GROWTH
    assert wild_type == v.EXPECTED_WILD_TYPE_WELLS
    assert agreement == pytest.approx(v.SOFTWARE_GREW_AGREEMENT, abs=v.AGREEMENT_TOL)
    assert sum(records_per_run.values()) == 977
