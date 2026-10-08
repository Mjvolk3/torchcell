# tests/torchcell/knowledge_graphs/test_build_time_projection.py
# [[tests.torchcell.knowledge_graphs.test_build_time_projection]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/knowledge_graphs/test_build_time_projection.py
"""The KG build-time projection: models, calibration, projection, the record gatherer.

Synthetic calibration (``TIMINGS``): three adapters with hand-picked totals over
``FULL`` record counts chosen so every rate is a short decimal::

    adapter                  dataset full   kg_full records   total_sec   rate s/rec
    SmfCostanzo2016Adapter        2,000           2,000          40.0       0.02
    DmiCostanzo2016Adapter    1,000,000         100,000         300.0       0.003
    SmfKuzmin2018Adapter            500             500          10.0       0.02

``kg_full`` caps only dmf/dmi Costanzo (100,000), so the Dmi row is the one where
``min(cap, full)`` matters. Projections of that calibration are worked in each test.

The committed job-966 file (``experiments/database/scripts/build966_timings.json``) is
read as-is: 33 adapters whose ``total_sec`` sum to the measured 33,180 s, so the
``kg_full`` self-check must land at 0 % error, and the uncapped figure is
``33180 - (147 + 861) + (147 + 861) / 100000 * 20705612 = 240884.56896`` s.

``gather_dataset_full_records`` runs against a fake ``lmdb`` module placed in
``sys.modules`` (the function imports it locally), recording every ``open`` call; no
store is touched.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from torchcell.knowledge_graphs import build_time_projection as btp
from torchcell.knowledge_graphs.build_time_projection import (
    ADAPTER_TO_DATASET,
    DATASET_FULL_RECORDS,
    DATASET_LMDB_SUBPATH,
    KG_FULL_CONFIG,
    AdapterRate,
    AdapterTiming,
    CalibrationTimings,
    DatasetSize,
    SubsetConfig,
    calibrate,
    dataset_sizes,
    gather_dataset_full_records,
    load_timings,
    project_build_time,
)

REPO = Path(__file__).resolve().parents[3]
CALIBRATION_FILE = (
    REPO / "experiments" / "database" / "scripts" / "build966_timings.json"
)

FULL: dict[str, int] = {
    "SmfCostanzo2016Dataset": 2000,
    "DmiCostanzo2016Dataset": 1_000_000,
    "SmfKuzmin2018Dataset": 500,
}

TIMINGS = CalibrationTimings(
    generation_total_sec=350.0,
    adapters=[
        AdapterTiming(
            adapter="SmfCostanzo2016Adapter",
            node_sec=15.0,
            edge_sec=25.0,
            total_sec=40.0,
            n_nodes=16000,
        ),
        AdapterTiming(
            adapter="DmiCostanzo2016Adapter",
            node_sec=140.0,
            edge_sec=160.0,
            total_sec=300.0,
            n_nodes=900000,
        ),
        AdapterTiming(
            adapter="SmfKuzmin2018Adapter",
            node_sec=4.0,
            edge_sec=6.0,
            total_sec=10.0,
            n_nodes=4000,
        ),
    ],
)


def _error_summary(exc: ValidationError) -> list[tuple[str, tuple[int | str, ...]]]:
    return [(e["type"], tuple(e["loc"])) for e in exc.errors()]


# --- pydantic models -------------------------------------------------------


def test_adapter_timing_refuses_a_non_integer_node_count_and_a_missing_total() -> None:
    """``n_nodes`` is an int (``"many"`` fails int parsing) and ``total_sec`` is required."""
    with pytest.raises(ValidationError) as exc:
        AdapterTiming.model_validate(
            {"adapter": "A", "node_sec": 1.0, "edge_sec": 2.0, "n_nodes": "many"}
        )
    assert _error_summary(exc.value) == [
        ("missing", ("total_sec",)),
        ("int_parsing", ("n_nodes",)),
    ]


def test_dataset_size_source_is_a_closed_literal_defaulting_to_lmdb() -> None:
    """``source`` accepts only ``lmdb`` / ``estimated``; omitted, it is ``lmdb``."""
    row = DatasetSize(dataset="D", full_records=3, lmdb_subpath="data/torchcell/d")
    assert row.source == "lmdb"
    assert (
        DatasetSize(
            dataset="D", full_records=3, lmdb_subpath="x", source="estimated"
        ).source
        == "estimated"
    )
    with pytest.raises(ValidationError) as exc:
        DatasetSize.model_validate(
            {"dataset": "D", "full_records": 3, "lmdb_subpath": "x", "source": "guess"}
        )
    assert _error_summary(exc.value) == [("literal_error", ("source",))]


def test_subset_config_refuses_a_fractional_cap() -> None:
    """A cap of 1.5 records is not an int; pydantic refuses it rather than truncating."""
    with pytest.raises(ValidationError) as exc:
        SubsetConfig.model_validate({"subset_size": 1.5})
    assert _error_summary(exc.value) == [("int_from_float", ("subset_size",))]
    with pytest.raises(ValidationError) as exc2:
        SubsetConfig.model_validate({"per_dataset": {"D": 2.5}})
    assert _error_summary(exc2.value) == [("int_from_float", ("per_dataset", "D"))]


def test_calibration_timings_subsets_default_is_a_fresh_empty_dict() -> None:
    """``subsets`` defaults to ``{}`` and each instance gets its own dict. This pins a
    pydantic guarantee (``Field(default_factory=dict)``), kept so a switch to a shared
    mutable default would fail here.
    """
    a = CalibrationTimings(generation_total_sec=1.0, adapters=[])
    b = CalibrationTimings(generation_total_sec=1.0, adapters=[])
    a.subsets["X"] = [1, 2]
    assert b.subsets == {}


# --- SubsetConfig ------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "dataset", "full", "cap", "effective"),
    [
        # no caps at all: full
        (SubsetConfig(), "D", 1234, None, 1234),
        # global cap below full: the cap
        (SubsetConfig(subset_size=1000), "D", 1234, 1000, 1000),
        # global cap above full: full (min)
        (SubsetConfig(subset_size=5000), "D", 1234, 5000, 1234),
        # per-dataset cap overrides the global one, in both directions
        (SubsetConfig(subset_size=1000, per_dataset={"D": 10}), "D", 1234, 10, 10),
        (SubsetConfig(subset_size=10, per_dataset={"D": 1000}), "D", 1234, 1000, 1000),
        # per-dataset None means uncapped for that dataset even under a global cap
        (SubsetConfig(subset_size=10, per_dataset={"D": None}), "D", 1234, None, 1234),
        # a per-dataset entry for another dataset does not apply
        (SubsetConfig(subset_size=10, per_dataset={"E": None}), "D", 1234, 10, 10),
        # cap 0 is a cap, not "uncapped"
        (SubsetConfig(subset_size=0), "D", 1234, 0, 0),
        # kg_full: dmf/dmi Costanzo capped at 100k, everything else full
        (KG_FULL_CONFIG, "DmfCostanzo2016Dataset", 20705612, 100000, 100000),
        (KG_FULL_CONFIG, "DmiCostanzo2016Dataset", 20705612, 100000, 100000),
        (KG_FULL_CONFIG, "SmfCostanzo2016Dataset", 20484, None, 20484),
    ],
)
def test_cap_for_and_effective_records(
    config: SubsetConfig, dataset: str, full: int, cap: int | None, effective: int
) -> None:
    """``cap_for``: per_dataset entry (even ``None``) wins, else ``subset_size``;
    ``effective_records = full`` when uncapped else ``min(cap, full)``.
    """
    assert config.cap_for(dataset) == cap
    assert config.effective_records(dataset, full) == effective


def test_kg_full_config_is_a_copy_of_the_calibration_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutating the config's dict must not reach the module constant it was built from
    (the entry is added through monkeypatch, so it is removed after the test).
    """
    monkeypatch.setitem(KG_FULL_CONFIG.per_dataset, "SmfKuzmin2018Dataset", 5)
    assert btp.CALIBRATION_PER_DATASET_CAPS == {
        "DmfCostanzo2016Dataset": 100000,
        "DmiCostanzo2016Dataset": 100000,
    }
    monkeypatch.undo()
    assert KG_FULL_CONFIG.subset_size is None
    assert KG_FULL_CONFIG.per_dataset == {
        "DmfCostanzo2016Dataset": 100000,
        "DmiCostanzo2016Dataset": 100000,
    }
    assert KG_FULL_CONFIG.per_dataset is not btp.CALIBRATION_PER_DATASET_CAPS


# --- load_timings ------------------------------------------------------------


def test_load_timings_reads_a_hand_written_file(tmp_path: Path) -> None:
    """Every field round-trips; an unknown top-level key (``finish``) is ignored."""
    path = tmp_path / "t.json"
    path.write_text(
        json.dumps(
            {
                "generation_total_sec": 12.5,
                "finish": "2026-07-19 10:57:39",
                "subsets": {"DmiCostanzo2016Dataset": [20705612, 100000]},
                "adapters": [
                    {
                        "adapter": "SmfKuzmin2018Adapter",
                        "node_sec": 5.0,
                        "edge_sec": 7.5,
                        "total_sec": 12.5,
                        "n_nodes": 99,
                    }
                ],
            }
        )
    )
    got = load_timings(str(path))
    assert got == CalibrationTimings(
        generation_total_sec=12.5,
        adapters=[
            AdapterTiming(
                adapter="SmfKuzmin2018Adapter",
                node_sec=5.0,
                edge_sec=7.5,
                total_sec=12.5,
                n_nodes=99,
            )
        ],
        subsets={"DmiCostanzo2016Dataset": [20705612, 100000]},
    )
    assert load_timings(path) == got


def test_load_timings_refuses_an_adapter_without_its_timing(tmp_path: Path) -> None:
    """The adapter entry's missing ``edge_sec`` is reported at its list position."""
    path = tmp_path / "t.json"
    path.write_text(
        json.dumps(
            {
                "generation_total_sec": 1.0,
                "adapters": [
                    {"adapter": "A", "node_sec": 1.0, "total_sec": 1.0, "n_nodes": 1}
                ],
            }
        )
    )
    with pytest.raises(ValidationError) as exc:
        load_timings(path)
    assert _error_summary(exc.value) == [("missing", ("adapters", 0, "edge_sec"))]


# --- calibrate ---------------------------------------------------------------


def test_calibrate_divides_each_total_by_its_kg_full_record_count() -> None:
    """Rates 40/2000 = 0.02, 300/min(100000, 1e6) = 0.003, 10/500 = 0.02 (table above)."""
    assert calibrate(TIMINGS, FULL) == [
        AdapterRate(
            adapter="SmfCostanzo2016Adapter",
            dataset="SmfCostanzo2016Dataset",
            total_sec_966=40.0,
            records_966=2000,
            rate_s_per_rec=40.0 / 2000,
        ),
        AdapterRate(
            adapter="DmiCostanzo2016Adapter",
            dataset="DmiCostanzo2016Dataset",
            total_sec_966=300.0,
            records_966=100000,
            rate_s_per_rec=300.0 / 100000,
        ),
        AdapterRate(
            adapter="SmfKuzmin2018Adapter",
            dataset="SmfKuzmin2018Dataset",
            total_sec_966=10.0,
            records_966=500,
            rate_s_per_rec=10.0 / 500,
        ),
    ]


def test_calibrate_honors_an_explicit_calibration_config() -> None:
    """Calibrated under ``subset_size=250``: records min(250, full) = 250, 250, 250, so
    rates 40/250 = 0.16, 300/250 = 1.2, 10/250 = 0.04.
    """
    rates = calibrate(TIMINGS, FULL, calibration_config=SubsetConfig(subset_size=250))
    assert [(r.records_966, r.rate_s_per_rec) for r in rates] == [
        (250, 40.0 / 250),
        (250, 300.0 / 250),
        (250, 10.0 / 250),
    ]


def test_calibrate_refuses_an_unmapped_adapter_and_an_unknown_dataset() -> None:
    """An adapter outside ``ADAPTER_TO_DATASET`` and a dataset with no full count are
    both KeyErrors naming the missing key (no silent zero rate).
    """
    unknown = CalibrationTimings(
        generation_total_sec=1.0,
        adapters=[
            AdapterTiming(
                adapter="NoSuchAdapter",
                node_sec=0.5,
                edge_sec=0.5,
                total_sec=1.0,
                n_nodes=1,
            )
        ],
    )
    with pytest.raises(KeyError, match=r"^'NoSuchAdapter'$"):
        calibrate(unknown, FULL)
    with pytest.raises(KeyError, match=r"^'DmiCostanzo2016Dataset'$"):
        calibrate(TIMINGS, {"SmfCostanzo2016Dataset": 2000})


def test_calibrate_refuses_a_dataset_with_zero_records() -> None:
    """A zero-record dataset has no per-record rate: the division raises."""
    with pytest.raises(ZeroDivisionError, match=r"^float division by zero$"):
        calibrate(TIMINGS, {**FULL, "SmfKuzmin2018Dataset": 0})


# --- project_build_time ----------------------------------------------------------


def test_projecting_the_calibration_config_reproduces_the_measured_total() -> None:
    """kg_full: 0.02*2000 + 0.003*100000 + 0.02*500 = 40 + 300 + 10 = 350 s; error 0 %.
    Rows sorted descending: Dmi 300 (85.714 %), Smf 40 (11.429 %), Kuzmin 10 (2.857 %).
    """
    proj = project_build_time(
        calibrate(TIMINGS, FULL),
        FULL,
        KG_FULL_CONFIG,
        label="kg_full",
        measured_sec=350.0,
    )
    assert proj.label == "kg_full"
    assert proj.subset_config == KG_FULL_CONFIG
    assert proj.total_sec == pytest.approx(350.0, rel=1e-12)
    assert proj.total_hours == pytest.approx(350.0 / 3600, rel=1e-12)
    assert proj.measured_sec == 350.0
    assert proj.error_pct == pytest.approx(0.0, abs=1e-10)
    got = [
        (c.adapter, c.dataset, c.records, c.projected_sec, c.pct_of_total)
        for c in proj.contributions
    ]
    expected = [
        ("DmiCostanzo2016Adapter", "DmiCostanzo2016Dataset", 100000, 300.0, 600 / 7),
        ("SmfCostanzo2016Adapter", "SmfCostanzo2016Dataset", 2000, 40.0, 80 / 7),
        ("SmfKuzmin2018Adapter", "SmfKuzmin2018Dataset", 500, 10.0, 20 / 7),
    ]
    assert [g[:3] for g in got] == [e[:3] for e in expected]
    assert [g[3] for g in got] == pytest.approx([e[3] for e in expected], rel=1e-12)
    assert [g[4] for g in got] == pytest.approx([e[4] for e in expected], rel=1e-12)
    assert [c.projected_hours for c in proj.contributions] == pytest.approx(
        [300 / 3600, 40 / 3600, 10 / 3600], rel=1e-12
    )


def test_uncapped_projection_scales_the_capped_rate_to_the_full_count() -> None:
    """Uncapped: Dmi 0.003 * 1,000,000 = 3000 s, Smf 40, Kuzmin 10; total 3050 s.
    No measured time: ``measured_sec`` and ``error_pct`` stay ``None``; default label.
    """
    proj = project_build_time(calibrate(TIMINGS, FULL), FULL, SubsetConfig())
    assert proj.label == "projection"
    assert proj.measured_sec is None
    assert proj.error_pct is None
    assert proj.total_sec == pytest.approx(3050.0, rel=1e-12)
    assert [(c.adapter, c.records) for c in proj.contributions] == [
        ("DmiCostanzo2016Adapter", 1_000_000),
        ("SmfCostanzo2016Adapter", 2000),
        ("SmfKuzmin2018Adapter", 500),
    ]
    assert sum(c.pct_of_total for c in proj.contributions) == pytest.approx(100.0)


def test_mixed_caps_ties_keep_input_order_and_error_is_signed() -> None:
    """``subset_size=100`` with Dmi uncapped: Smf 0.02*100 = 2, Dmi 0.003*1e6 = 3000,
    Kuzmin 0.02*100 = 2; total 3004 s. The 2 s tie keeps the calibration order (stable
    sort: Smf before Kuzmin). Against measured 3500 s: 100*(3004-3500)/3500 = -14.1714 %.
    """
    proj = project_build_time(
        calibrate(TIMINGS, FULL),
        FULL,
        SubsetConfig(subset_size=100, per_dataset={"DmiCostanzo2016Dataset": None}),
        measured_sec=3500.0,
    )
    assert [(c.adapter, c.records) for c in proj.contributions] == [
        ("DmiCostanzo2016Adapter", 1_000_000),
        ("SmfCostanzo2016Adapter", 100),
        ("SmfKuzmin2018Adapter", 100),
    ]
    assert proj.total_sec == pytest.approx(3004.0, rel=1e-12)
    assert proj.error_pct == pytest.approx(100 * (3004 - 3500) / 3500, rel=1e-12)


def test_a_zero_total_projection_reports_zero_percent_rather_than_dividing() -> None:
    """``subset_size=0`` processes no record: total 0 s and every share 0.0 %."""
    proj = project_build_time(
        calibrate(TIMINGS, FULL), FULL, SubsetConfig(subset_size=0)
    )
    assert proj.total_sec == 0.0
    assert [
        (c.records, c.projected_sec, c.pct_of_total) for c in proj.contributions
    ] == [(0, 0.0, 0.0), (0, 0.0, 0.0), (0, 0.0, 0.0)]


# --- the committed calibration and constants --------------------------------------


def test_the_committed_job_966_file_self_checks_at_zero_error() -> None:
    """The 33 per-adapter totals sum to the measured 33,180 s, so projecting kg_full with
    the committed counts gives 33,180 s at 0 % error; uncapped gives
    32172 + 1008/100000 * 20705612 = 240884.56896 s.
    """
    timings = load_timings(CALIBRATION_FILE)
    assert timings.generation_total_sec == 33180.0
    assert [t.adapter for t in timings.adapters] == list(ADAPTER_TO_DATASET)
    assert sum(t.total_sec for t in timings.adapters) == 33180.0
    assert timings.subsets == {
        "DmiCostanzo2016Dataset": [
            DATASET_FULL_RECORDS["DmiCostanzo2016Dataset"],
            100000,
        ]
    }
    rates = calibrate(timings, DATASET_FULL_RECORDS)
    full = project_build_time(
        rates, DATASET_FULL_RECORDS, KG_FULL_CONFIG, measured_sec=33180.0
    )
    assert full.total_sec == pytest.approx(33180.0, rel=1e-12)
    assert full.error_pct == pytest.approx(0.0, abs=1e-9)
    uncapped = project_build_time(rates, DATASET_FULL_RECORDS, SubsetConfig())
    assert uncapped.total_sec == pytest.approx(
        32172 + 1008 / 100000 * 20705612, rel=1e-12
    )
    assert uncapped.contributions[0].adapter == "DmiCostanzo2016Adapter"


def test_dataset_sizes_cover_the_same_datasets_as_the_adapter_map() -> None:
    """One typed row per committed count, keyed in the same order, with the subpath.
    The committed total is 44,357,773 records: the two Costanzo double-mutant sets,
    2 x 20,705,612 = 41,411,224, plus 2,946,549 for the other 31 datasets.
    """
    sizes = dataset_sizes()
    assert list(sizes) == list(DATASET_FULL_RECORDS)
    assert set(sizes) == set(DATASET_LMDB_SUBPATH) == set(ADAPTER_TO_DATASET.values())
    assert sizes["DmiCostanzo2016Dataset"] == DatasetSize(
        dataset="DmiCostanzo2016Dataset",
        full_records=20705612,
        lmdb_subpath="data/torchcell/dmi_costanzo2016",
        source="lmdb",
    )
    assert sum(s.full_records for s in sizes.values()) == 44_357_773
    assert (
        sum(
            s.full_records
            for name, s in sizes.items()
            if name not in ("DmfCostanzo2016Dataset", "DmiCostanzo2016Dataset")
        )
        == 2_946_549
    )


BACTERIAL_DATASETS = {
    "CarbonSourceTong2020Dataset",
    "CrispriArrayYunus2026Dataset",
    "CrispriGuideFitnessWang2018Dataset",
    "CrispriKnockdownCui2018Dataset",
    "CrispriKnockdownYunus2026Dataset",
    "CrispriScreenRousset2018Dataset",
    "EnvChemgenShiver2016Dataset",
    "EnvChemgenWang2015Dataset",
    "GeneEssentialityGoodall2018Dataset",
    "GrowthRateCampos2018Dataset",
    "IsoprenolSelectionMenasalvas2025Dataset",
    "IsoprenolTiterCarruthers2025Dataset",
    "IsoprenolTiterDeSiqueira2025Dataset",
    "IsoprenolToleranceLim2025Dataset",
    "IsoprenylAcetateTiterKang2026Dataset",
    "MetabolomeFuhrer2017Dataset",
    "MetabolomeRapp2026Dataset",
    "ProteinTurnoverGupta2024Dataset",
    "ProteomeMori2021Dataset",
    "ProteomeCaglar2017Dataset",
    "ProteomeCarruthers2025Dataset",
    "ProteomeDeSiqueira2025Dataset",
    "ProteomeLim2025Dataset",
    "ProteomeSchmidt2016Dataset",
    "PutidaPrecise321Lim2022Dataset",
    "RbTnseqBorchert2024Dataset",
    "RbTnseqPrice2018EcoliDataset",
    "RnaseqCaglar2017Dataset",
    "RnaseqLamoureux2023Dataset",
}
"""The E. coli and P. putida datasets mapped in plan step 9 and after, none calibrated
yet."""


def test_adapter_to_dataset_is_the_inverse_of_the_served_adapter_map() -> None:
    """Every pair here is a (dataset, adapter) pair of ``dataset_adapter_map``.

    Finding: the projection covers 33 of the 79 datasets in ``dataset_adapter_map``.
    The 46 absent ones (``ADAPTER_TO_DATASET``, build_time_projection.py:81-115) include
    the six EnvChemgen chemogenomic sets, both Hillenmeyer 2008 sets, the Nadal-Ribelles
    Perturb-seq set and the 28 bacterial datasets mapped in plan.bacteria-ontology-genome
    step 9 and after, so a projection of a build that serves them omits their
    generation time
    entirely (``calibrate`` would raise ``KeyError`` on a timing file that lists them).
    Their record counts were not measured here. Pinned until those adapters are
    calibrated and added to the three tables.
    """
    from torchcell.knowledge_graphs.dataset_adapter_map import dataset_adapter_map

    inverse = {a.__name__: d.__name__ for d, a in dataset_adapter_map.items()}
    assert {a: inverse[a] for a in ADAPTER_TO_DATASET} == ADAPTER_TO_DATASET
    assert len(dataset_adapter_map) == 79
    served = {d.__name__ for d in dataset_adapter_map}
    assert served - set(ADAPTER_TO_DATASET.values()) == BACTERIAL_DATASETS | {
        "AminoAcidCooper2010Dataset",
        "Bloom2019Dataset",
        "CrisprMagicLian2019Dataset",
        "CrispriChemgenSmith2016Dataset",
        "CrispriMormino2022Dataset",
        "EnvChemgenAuesukaree2009Dataset",
        "EnvChemgenCostanzo2021Dataset",
        "EnvChemgenHoepfner2014Dataset",
        "EnvChemgenMota2024Dataset",
        "EnvChemgenVanacloig2022Dataset",
        "EnvChemgenWildenhain2015Dataset",
        "FattyAcidSmith2006Dataset",
        "HetHillenmeyer2008Dataset",
        "HomHillenmeyer2008Dataset",
        "NadalRibellesPerturbSeq2025Dataset",
        "ProteomeMessner2023Dataset",
        "SmfBaryshnikova2010Dataset",
        "SmfODuibhir2014Dataset",
    }


DATASET_MODULES: dict[str, str] = {
    "SmfCostanzo2016Dataset": "costanzo2016",
    "DmfCostanzo2016Dataset": "costanzo2016",
    "DmiCostanzo2016Dataset": "costanzo2016",
    "SmfKuzmin2018Dataset": "kuzmin2018",
    "DmfKuzmin2018Dataset": "kuzmin2018",
    "TmfKuzmin2018Dataset": "kuzmin2018",
    "DmiKuzmin2018Dataset": "kuzmin2018",
    "TmiKuzmin2018Dataset": "kuzmin2018",
    "SmfKuzmin2020Dataset": "kuzmin2020",
    "DmfKuzmin2020Dataset": "kuzmin2020",
    "TmfKuzmin2020Dataset": "kuzmin2020",
    "DmiKuzmin2020Dataset": "kuzmin2020",
    "TmiKuzmin2020Dataset": "kuzmin2020",
    "GeneEssentialitySgdDataset": "sgd",
    "SynthLethalityYeastSynthLethDbDataset": "synth_leth_db",
    "SynthRescueYeastSynthLethDbDataset": "synth_leth_db",
    "ScmdOhya2005Dataset": "ohya2005",
    "MicroarrayKemmeren2014Dataset": "kemmeren2014",
    "SmMicroarraySameith2015Dataset": "sameith2015",
    "DmMicroarraySameith2015Dataset": "sameith2015",
    "CaudalPanTranscriptome2024Dataset": "caudal2024",
    "ScmdOhnuki2018Dataset": "ohnuki2018",
    "ScmdOhnuki2022Dataset": "ohnuki2022",
    "CarotenoidOzaydin2013Dataset": "ozaydin2013",
    "BetaxanthinCachera2023Dataset": "cachera2023",
    "MetaboliteDaSilveira2014Dataset": "dasilveira2014",
    "MetaboliteZelezniak2018Dataset": "zelezniak2018",
    "ProteomeZelezniak2018Dataset": "zelezniak2018",
    "AminoAcidMulleder2016Dataset": "mulleder2016",
    "OrganicAcidYoshida2012Dataset": "yoshida2012",
    "IsobutanolScreenLopez2024Dataset": "lopez2024",
    "IsobutanolValidatedLopez2024Dataset": "lopez2024",
    "FattyAcidXue2025Dataset": "xue2025",
}


def test_lmdb_subpaths_differ_from_loader_default_roots_only_for_synthleth() -> None:
    """Finding: the ``DATASET_LMDB_SUBPATH`` comment (build_time_projection.py:159-161)
    says the subpaths match each loader's default ``root`` except DmfCostanzo. Measured:
    DmfCostanzo now matches (its default root was fixed in 9eadc39e2), and the two
    SynLethDB datasets do NOT: the loaders default to ``syn_leth_db_yeast`` /
    ``syn_rescue_db_yeast`` (synth_leth_db.py:488, 621) while the table names
    ``synth_lethality_yeast_synth_leth_db`` / ``synth_rescue_yeast_synth_leth_db``, the
    KG conf path (knowledge_graphs/conf/scerevisiae_global_kg.yaml:72). Consequence,
    checked read-only on GilaHyper 2026.10.06: those two dev-tree directories hold only
    ``raw/`` (no ``processed/lmdb``), so without ``build_tree`` the gatherer raises on
    ``lmdb.open`` for them; with ``build_tree`` it falls back to the database tree (14000
    and 6948 entries, the committed constants), while the dev default-root LMDBs the live
    rebuild reads (``build_dataset_lmdb.dataset_default_root``) hold 13996 and 6942.
    Reach: only ``experiments/database/scripts/project_build_time.py`` uses this table;
    latent. Pinned until the table and the loaders agree (or the comment names the
    SynLethDB exception).
    """
    differ = {}
    for dataset, subpath in DATASET_LMDB_SUBPATH.items():
        module = importlib.import_module(
            f"torchcell.datasets.scerevisiae.{DATASET_MODULES[dataset]}"
        )
        cls = getattr(module, dataset)
        default = inspect.signature(cls.__init__).parameters["root"].default
        if default != subpath:
            differ[dataset] = (default, subpath)
    assert differ == {
        "SynthLethalityYeastSynthLethDbDataset": (
            "data/torchcell/syn_leth_db_yeast",
            "data/torchcell/synth_lethality_yeast_synth_leth_db",
        ),
        "SynthRescueYeastSynthLethDbDataset": (
            "data/torchcell/syn_rescue_db_yeast",
            "data/torchcell/synth_rescue_yeast_synth_leth_db",
        ),
    }


# --- gather_dataset_full_records ---------------------------------------------------


class _FakeTxn:
    def __init__(self, entries: int) -> None:
        self.entries = entries

    def __enter__(self) -> _FakeTxn:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def stat(self) -> dict[str, int]:
        return {"entries": self.entries}


class _FakeEnv:
    def __init__(self, entries: int, log: list[tuple[str, Any]]) -> None:
        self.entries = entries
        self.log = log

    def begin(self) -> _FakeTxn:
        return _FakeTxn(self.entries)

    def close(self) -> None:
        self.log.append(("close", self.entries))


def _install_fake_lmdb(
    monkeypatch: pytest.MonkeyPatch, entries_by_path: dict[str, int]
) -> list[tuple[str, Any]]:
    log: list[tuple[str, Any]] = []

    def fake_open(path: str, **kwargs: Any) -> _FakeEnv:
        log.append(("open", (path, kwargs)))
        return _FakeEnv(entries_by_path[path], log)

    fake = types.ModuleType("lmdb")
    setattr(fake, "open", fake_open)
    monkeypatch.setitem(sys.modules, "lmdb", fake)
    return log


def test_gather_prefers_the_dev_tree_and_falls_back_to_the_build_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two datasets: ``a`` has a dev-tree LMDB dir, ``b`` does not, so ``b`` is opened at
    ``<build_tree>/<leaf>/processed/lmdb`` (leaf = last subpath component). Each env is
    opened read-only without a lock and closed; the counts come from ``stat()``.
    """
    data_root = tmp_path / "dev"
    build = tmp_path / "build"
    dev_a = data_root / "data/torchcell/a_ds" / "processed" / "lmdb"
    dev_a.mkdir(parents=True)
    monkeypatch.setattr(
        btp,
        "DATASET_LMDB_SUBPATH",
        {"ADataset": "data/torchcell/a_ds", "BDataset": "data/torchcell/b_ds"},
    )
    build_b = str(build / "b_ds" / "processed" / "lmdb")
    log = _install_fake_lmdb(monkeypatch, {str(dev_a): 7, build_b: 11})
    sizes = gather_dataset_full_records(str(data_root), build_tree=str(build))
    assert sizes == {
        "ADataset": DatasetSize(
            dataset="ADataset", full_records=7, lmdb_subpath="data/torchcell/a_ds"
        ),
        "BDataset": DatasetSize(
            dataset="BDataset", full_records=11, lmdb_subpath="data/torchcell/b_ds"
        ),
    }
    kwargs = {"readonly": True, "lock": False, "subdir": True, "max_dbs": 0}
    assert log == [
        ("open", (str(dev_a), kwargs)),
        ("close", 7),
        ("open", (build_b, kwargs)),
        ("close", 11),
    ]


def test_gather_without_a_build_tree_opens_the_missing_dev_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No fallback without ``build_tree``: the absent dev path itself goes to
    ``lmdb.open`` (a real read-only open of a missing dir raises there).
    """
    monkeypatch.setattr(
        btp, "DATASET_LMDB_SUBPATH", {"BDataset": "data/torchcell/b_ds"}
    )
    dev_b = str(tmp_path / "data/torchcell/b_ds" / "processed" / "lmdb")
    log = _install_fake_lmdb(monkeypatch, {dev_b: 3})
    assert gather_dataset_full_records(str(tmp_path))["BDataset"].full_records == 3
    assert log[0] == (
        "open",
        (dev_b, {"readonly": True, "lock": False, "subdir": True, "max_dbs": 0}),
    )
