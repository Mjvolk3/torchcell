# torchcell/knowledge_graphs/build_time_projection.py
# [[torchcell.knowledge_graphs.build_time_projection]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/build_time_projection
"""Project the tcdb knowledge-graph BUILD (CSV-generation) time for any subset config.

The KG build's wall time is dominated by adapter CSV generation -- each adapter
iterates its dataset's records and writes BioCypher node/edge CSVs. Dataset
*import* (constructing the in-memory pydantic records from the LMDB) is ~0 relative
to generation, so this model deliberately covers ONLY adapter generation.

Model (linear in records processed)
-----------------------------------
Per adapter ``i`` the generation time is ~linear in the number of records it
actually iterates::

    rate_i          = total_sec_i / records_966_i          # calibrate
    projected_sec_i = rate_i * effective_records_i(config)  # project
    total           = Σ_i projected_sec_i

``effective_records_i(config) = min(cap_i, full_records_i)`` (full when uncapped),
computed by :meth:`SubsetConfig.effective_records` -- the SAME min(cap, full) rule
the build applies via ``torchcell.knowledge_graphs.subset.subset_dataset``. Both
calibration and projection route through it, so projecting the calibration config
reproduces the measured total by construction.

Calibration (job 966, config ``kg_full``)
-----------------------------------------
Job 966 ran ``kg_full``: subsetting OFF (``subset.size: null``) except the two
giant Costanzo double-mutant sets (20.7M records each), each capped to 100k via
``subset.per_dataset``. So ``records_966_i`` is 100000 for dmf/dmi Costanzo and the
full dataset record count for every other adapter -- exactly what
``KG_FULL_CONFIG.effective_records`` yields. The per-adapter ``total_sec`` values
sum to the measured ``generation_total_sec`` (33180 s = 9.22 h), so the ``kg_full``
projection self-check lands at ~0 % error.

Two calibration caveats, both intentionally captured per-adapter
----------------------------------------------------------------
1. Serialization-heavy adapters (Caudal RNA-seq, Kemmeren + Sameith microarray)
   legitimately carry a high per-record rate -- each record serializes a large
   expression vector. That cost is real and is captured in that adapter's own
   ``rate_i``; it is not smoothed away.
2. ``DmfCostanzo2016Dataset``'s source default ``root`` is a copy-paste of the smf
   path (``data/torchcell/smf_costanzo2016``), so in job 966 the Dmf adapter loaded
   the SMF LMDB (20,484 records -> 163,883 nodes, identical to the Smf adapter) and,
   since its 100k cap exceeded that, processed it in full. Its rate is therefore
   modelled per the ``kg_full`` intent (records_966 = 100000) but is NOT a faithful
   double-mutant-fitness rate. For an uncapped projection the Dmi rate is the
   trustworthy double-mutant analogue; the Dmf uncapped figure is a lower bound.
   See ``DmfCostanzo2016Dataset`` in ``torchcell/datasets/scerevisiae/costanzo2016.py``.

The gathered full record counts (``DATASET_FULL_RECORDS``) are a committed constant
so the tool is self-contained; :func:`gather_dataset_full_records` regenerates them
from the built LMDBs (read-only ``stat``, no dataset rebuild).
"""

from __future__ import annotations

import json
import os.path as osp
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# --- committed calibration constants --------------------------------------

CALIBRATION_JOB = 966
"""Slurm job whose per-adapter timings calibrate this model (config ``kg_full``)."""

# The kg_full ``subset.per_dataset`` block job 966 ran: only the two giant Costanzo
# double-mutant sets are capped (to 100k); everything else builds in full. Keyed by
# DATASET class name, mirroring the hydra config.
CALIBRATION_PER_DATASET_CAPS: dict[str, int] = {
    "DmfCostanzo2016Dataset": 100000,
    "DmiCostanzo2016Dataset": 100000,
}

# adapter class name -> dataset class name (inverse of
# ``torchcell.knowledge_graphs.dataset_adapter_map.dataset_adapter_map``). Timings
# are keyed by adapter; subset configs (and DATASET_FULL_RECORDS) by dataset class.
ADAPTER_TO_DATASET: dict[str, str] = {
    "SmfCostanzo2016Adapter": "SmfCostanzo2016Dataset",
    "DmfCostanzo2016Adapter": "DmfCostanzo2016Dataset",
    "DmiCostanzo2016Adapter": "DmiCostanzo2016Dataset",
    "SmfKuzmin2018Adapter": "SmfKuzmin2018Dataset",
    "DmfKuzmin2018Adapter": "DmfKuzmin2018Dataset",
    "TmfKuzmin2018Adapter": "TmfKuzmin2018Dataset",
    "DmiKuzmin2018Adapter": "DmiKuzmin2018Dataset",
    "TmiKuzmin2018Adapter": "TmiKuzmin2018Dataset",
    "SmfKuzmin2020Adapter": "SmfKuzmin2020Dataset",
    "DmfKuzmin2020Adapter": "DmfKuzmin2020Dataset",
    "TmfKuzmin2020Adapter": "TmfKuzmin2020Dataset",
    "DmiKuzmin2020Adapter": "DmiKuzmin2020Dataset",
    "TmiKuzmin2020Adapter": "TmiKuzmin2020Dataset",
    "GeneEssentialitySgdAdapter": "GeneEssentialitySgdDataset",
    "SynthLethalityYeastSynthLethDbAdapter": "SynthLethalityYeastSynthLethDbDataset",
    "SynthRescueYeastSynthLethDbAdapter": "SynthRescueYeastSynthLethDbDataset",
    "ScmdOhya2005Adapter": "ScmdOhya2005Dataset",
    "MicroarrayKemmeren2014Adapter": "MicroarrayKemmeren2014Dataset",
    "SmMicroarraySameith2015Adapter": "SmMicroarraySameith2015Dataset",
    "DmMicroarraySameith2015Adapter": "DmMicroarraySameith2015Dataset",
    "CaudalPanTranscriptome2024Adapter": "CaudalPanTranscriptome2024Dataset",
    "ScmdOhnuki2018Adapter": "ScmdOhnuki2018Dataset",
    "ScmdOhnuki2022Adapter": "ScmdOhnuki2022Dataset",
    "CarotenoidOzaydin2013Adapter": "CarotenoidOzaydin2013Dataset",
    "BetaxanthinCachera2023Adapter": "BetaxanthinCachera2023Dataset",
    "MetaboliteDaSilveira2014Adapter": "MetaboliteDaSilveira2014Dataset",
    "MetaboliteZelezniak2018Adapter": "MetaboliteZelezniak2018Dataset",
    "ProteomeZelezniak2018Adapter": "ProteomeZelezniak2018Dataset",
    "AminoAcidMulleder2016Adapter": "AminoAcidMulleder2016Dataset",
    "OrganicAcidYoshida2012Adapter": "OrganicAcidYoshida2012Dataset",
    "IsobutanolScreenLopez2024Adapter": "IsobutanolScreenLopez2024Dataset",
    "IsobutanolValidatedLopez2024Adapter": "IsobutanolValidatedLopez2024Dataset",
    "FattyAcidXue2025Adapter": "FattyAcidXue2025Dataset",
}

# dataset class name -> FULL record count = LMDB ``stat()['entries']`` of the full
# (uncapped) dataset. Gathered read-only from the dev tree
# ($DATA_ROOT/data/torchcell/<name>/processed/lmdb) on 2026-07-19; dmi cross-checks
# against the job-966 ``subsets`` entry (20,705,612). Regenerate with
# :func:`gather_dataset_full_records`. NOTE: the dmf/dmi values are the TRUE full
# double-mutant counts (dmf's buggy default root is bypassed here on purpose).
DATASET_FULL_RECORDS: dict[str, int] = {
    "SmfCostanzo2016Dataset": 20484,
    "DmfCostanzo2016Dataset": 20705612,
    "DmiCostanzo2016Dataset": 20705612,
    "SmfKuzmin2018Dataset": 1539,
    "DmfKuzmin2018Dataset": 410399,
    "TmfKuzmin2018Dataset": 91111,
    "DmiKuzmin2018Dataset": 410399,
    "TmiKuzmin2018Dataset": 91111,
    "SmfKuzmin2020Dataset": 472,
    "DmfKuzmin2020Dataset": 632797,
    "TmfKuzmin2020Dataset": 301798,
    "DmiKuzmin2020Dataset": 632797,
    "TmiKuzmin2020Dataset": 301798,
    "GeneEssentialitySgdDataset": 1329,
    "SynthLethalityYeastSynthLethDbDataset": 14000,
    "SynthRescueYeastSynthLethDbDataset": 6948,
    "ScmdOhya2005Dataset": 4718,
    "MicroarrayKemmeren2014Dataset": 1484,
    "SmMicroarraySameith2015Dataset": 82,
    "DmMicroarraySameith2015Dataset": 72,
    "CaudalPanTranscriptome2024Dataset": 943,
    "ScmdOhnuki2018Dataset": 1112,
    "ScmdOhnuki2022Dataset": 1979,
    "CarotenoidOzaydin2013Dataset": 4474,
    "BetaxanthinCachera2023Dataset": 4735,
    "MetaboliteDaSilveira2014Dataset": 127,
    "MetaboliteZelezniak2018Dataset": 95,
    "ProteomeZelezniak2018Dataset": 97,
    "AminoAcidMulleder2016Dataset": 4678,
    "OrganicAcidYoshida2012Dataset": 17,
    "IsobutanolScreenLopez2024Dataset": 4554,
    "IsobutanolValidatedLopez2024Dataset": 224,
    "FattyAcidXue2025Dataset": 176,
}

# dataset class name -> processed-LMDB subpath under DATA_ROOT, for re-gathering the
# full record counts. Matches each loader's default ``root`` EXCEPT DmfCostanzo,
# whose source default root is the (buggy) smf path -- here it is the true dmf tree.
DATASET_LMDB_SUBPATH: dict[str, str] = {
    "SmfCostanzo2016Dataset": "data/torchcell/smf_costanzo2016",
    "DmfCostanzo2016Dataset": "data/torchcell/dmf_costanzo2016",
    "DmiCostanzo2016Dataset": "data/torchcell/dmi_costanzo2016",
    "SmfKuzmin2018Dataset": "data/torchcell/smf_kuzmin2018",
    "DmfKuzmin2018Dataset": "data/torchcell/dmf_kuzmin2018",
    "TmfKuzmin2018Dataset": "data/torchcell/tmf_kuzmin2018",
    "DmiKuzmin2018Dataset": "data/torchcell/dmi_kuzmin2018",
    "TmiKuzmin2018Dataset": "data/torchcell/tmi_kuzmin2018",
    "SmfKuzmin2020Dataset": "data/torchcell/smf_kuzmin2020",
    "DmfKuzmin2020Dataset": "data/torchcell/dmf_kuzmin2020",
    "TmfKuzmin2020Dataset": "data/torchcell/tmf_kuzmin2020",
    "DmiKuzmin2020Dataset": "data/torchcell/dmi_kuzmin2020",
    "TmiKuzmin2020Dataset": "data/torchcell/tmi_kuzmin2020",
    "GeneEssentialitySgdDataset": "data/torchcell/gene_essentiality_sgd",
    "SynthLethalityYeastSynthLethDbDataset": "data/torchcell/synth_lethality_yeast_synth_leth_db",
    "SynthRescueYeastSynthLethDbDataset": "data/torchcell/synth_rescue_yeast_synth_leth_db",
    "ScmdOhya2005Dataset": "data/torchcell/scmd_ohya2005",
    "MicroarrayKemmeren2014Dataset": "data/torchcell/microarray_kemmeren2014",
    "SmMicroarraySameith2015Dataset": "data/torchcell/sm_microarray_sameith2015",
    "DmMicroarraySameith2015Dataset": "data/torchcell/dm_microarray_sameith2015",
    "CaudalPanTranscriptome2024Dataset": "data/torchcell/caudal_pantranscriptome2024",
    "ScmdOhnuki2018Dataset": "data/torchcell/scmd_ohnuki2018",
    "ScmdOhnuki2022Dataset": "data/torchcell/scmd_ohnuki2022",
    "CarotenoidOzaydin2013Dataset": "data/torchcell/carotenoid_ozaydin2013",
    "BetaxanthinCachera2023Dataset": "data/torchcell/betaxanthin_cachera2023",
    "MetaboliteDaSilveira2014Dataset": "data/torchcell/metabolite_dasilveira2014",
    "MetaboliteZelezniak2018Dataset": "data/torchcell/metabolite_zelezniak2018",
    "ProteomeZelezniak2018Dataset": "data/torchcell/proteome_zelezniak2018",
    "AminoAcidMulleder2016Dataset": "data/torchcell/amino_acid_mulleder2016",
    "OrganicAcidYoshida2012Dataset": "data/torchcell/organic_acid_yoshida2012",
    "IsobutanolScreenLopez2024Dataset": "data/torchcell/isobutanol_screen_lopez2024",
    "IsobutanolValidatedLopez2024Dataset": "data/torchcell/isobutanol_validated_lopez2024",
    "FattyAcidXue2025Dataset": "data/torchcell/ffa_xue2025",
}


# --- pydantic models ------------------------------------------------------


class AdapterTiming(BaseModel):
    """One adapter's measured CSV-generation timing from a calibration build."""

    adapter: str
    node_sec: float
    edge_sec: float
    total_sec: float
    n_nodes: int


class CalibrationTimings(BaseModel):
    """Parsed calibration file (per-adapter timings + the measured total)."""

    generation_total_sec: float
    adapters: list[AdapterTiming]
    subsets: dict[str, list[int]] = Field(default_factory=dict)


class SubsetConfig(BaseModel):
    """A KG-build subset config, mirroring the hydra ``kg_*.yaml`` ``subset`` block.

    ``subset_size`` caps every dataset (``None`` => full); ``per_dataset`` overrides
    that per DATASET class name (value ``None`` => uncapped for that dataset).
    """

    subset_size: int | None = None
    per_dataset: dict[str, int | None] = Field(default_factory=dict)

    def cap_for(self, dataset: str) -> int | None:
        """Return the record cap for ``dataset`` (``None`` => uncapped/full)."""
        if dataset in self.per_dataset:
            return self.per_dataset[dataset]
        return self.subset_size

    def effective_records(self, dataset: str, full_records: int) -> int:
        """Records actually processed = ``min(cap, full)``; ``full`` when uncapped."""
        cap = self.cap_for(dataset)
        if cap is None:
            return full_records
        return min(cap, full_records)


class DatasetSize(BaseModel):
    """Full (uncapped) record count for one dataset, with its provenance source."""

    dataset: str
    full_records: int
    lmdb_subpath: str
    source: Literal["lmdb", "estimated"] = "lmdb"


class AdapterRate(BaseModel):
    """Per-adapter calibrated generation rate (seconds per record processed)."""

    adapter: str
    dataset: str
    total_sec_966: float
    records_966: int
    rate_s_per_rec: float


class AdapterContribution(BaseModel):
    """One adapter's projected contribution to a build under some subset config."""

    adapter: str
    dataset: str
    records: int
    rate_s_per_rec: float
    projected_sec: float
    projected_hours: float
    pct_of_total: float


class BuildTimeProjection(BaseModel):
    """Projected build (CSV-generation) time for a subset config, with a breakdown."""

    label: str
    subset_config: SubsetConfig
    total_sec: float
    total_hours: float
    contributions: list[AdapterContribution]
    measured_sec: float | None = None
    error_pct: float | None = None


# The calibration config = the exact subset config job 966 (``kg_full``) ran.
KG_FULL_CONFIG = SubsetConfig(
    subset_size=None, per_dataset=dict(CALIBRATION_PER_DATASET_CAPS)
)


# --- API ------------------------------------------------------------------


def load_timings(path: str | Path) -> CalibrationTimings:
    """Load and validate a calibration-timings JSON file (e.g. job 966)."""
    raw = json.loads(Path(path).read_text())
    return CalibrationTimings.model_validate(raw)


def dataset_sizes() -> dict[str, DatasetSize]:
    """Return the committed full record counts as typed :class:`DatasetSize` rows."""
    return {
        name: DatasetSize(
            dataset=name,
            full_records=DATASET_FULL_RECORDS[name],
            lmdb_subpath=DATASET_LMDB_SUBPATH[name],
            source="lmdb",
        )
        for name in DATASET_FULL_RECORDS
    }


def calibrate(
    timings: CalibrationTimings,
    dataset_full_records: dict[str, int],
    *,
    calibration_config: SubsetConfig | None = None,
) -> list[AdapterRate]:
    """Fit a per-record rate for each adapter from the calibration build.

    ``records_966`` for each adapter is what it processed under the calibration
    config -- ``min(cap, full)`` -- so dmf/dmi Costanzo use 100000 and every other
    adapter its full record count. ``rate = total_sec / records_966``.
    """
    config = KG_FULL_CONFIG if calibration_config is None else calibration_config
    rates: list[AdapterRate] = []
    for t in timings.adapters:
        dataset = ADAPTER_TO_DATASET[t.adapter]
        records_966 = config.effective_records(dataset, dataset_full_records[dataset])
        rates.append(
            AdapterRate(
                adapter=t.adapter,
                dataset=dataset,
                total_sec_966=t.total_sec,
                records_966=records_966,
                rate_s_per_rec=t.total_sec / records_966,
            )
        )
    return rates


def project_build_time(
    rates: list[AdapterRate],
    dataset_full_records: dict[str, int],
    subset_config: SubsetConfig,
    *,
    label: str = "projection",
    measured_sec: float | None = None,
) -> BuildTimeProjection:
    """Project total build time for ``subset_config`` from calibrated ``rates``.

    ``projected_sec_i = rate_i * min(cap_i, full_records_i)`` summed over adapters.
    The breakdown is sorted by descending contribution. When ``measured_sec`` is
    given (e.g. the ``kg_full`` self-check), the percent error is reported.
    """
    rows: list[AdapterContribution] = []
    for r in rates:
        records = subset_config.effective_records(
            r.dataset, dataset_full_records[r.dataset]
        )
        projected = r.rate_s_per_rec * records
        rows.append(
            AdapterContribution(
                adapter=r.adapter,
                dataset=r.dataset,
                records=records,
                rate_s_per_rec=r.rate_s_per_rec,
                projected_sec=projected,
                projected_hours=projected / 3600.0,
                pct_of_total=0.0,  # filled once the total is known
            )
        )
    total_sec = sum(row.projected_sec for row in rows)
    for row in rows:
        row.pct_of_total = 100.0 * row.projected_sec / total_sec if total_sec else 0.0
    rows.sort(key=lambda row: row.projected_sec, reverse=True)
    error_pct = (
        None
        if measured_sec is None
        else 100.0 * (total_sec - measured_sec) / measured_sec
    )
    return BuildTimeProjection(
        label=label,
        subset_config=subset_config,
        total_sec=total_sec,
        total_hours=total_sec / 3600.0,
        contributions=rows,
        measured_sec=measured_sec,
        error_pct=error_pct,
    )


def gather_dataset_full_records(
    data_root: str, build_tree: str | None = None
) -> dict[str, DatasetSize]:
    """Regenerate full record counts by reading each dataset's LMDB ``stat`` entries.

    Read-only (``lock=False``); never rebuilds a dataset. Prefers the dev tree
    (``data_root``); falls back to ``build_tree`` when a dataset's dev-tree LMDB is
    absent. Use this to refresh :data:`DATASET_FULL_RECORDS` after a rebuild.
    """
    import lmdb  # local import: the projection path itself touches no LMDB

    sizes: dict[str, DatasetSize] = {}
    for dataset, subpath in DATASET_LMDB_SUBPATH.items():
        dev = osp.join(data_root, subpath, "processed", "lmdb")
        lmdb_dir = dev
        if not osp.isdir(dev) and build_tree is not None:
            leaf = subpath.split("/")[-1]
            lmdb_dir = osp.join(build_tree, leaf, "processed", "lmdb")
        env = lmdb.open(lmdb_dir, readonly=True, lock=False, subdir=True, max_dbs=0)
        with env.begin() as txn:
            entries = int(txn.stat()["entries"])
        env.close()
        sizes[dataset] = DatasetSize(
            dataset=dataset, full_records=entries, lmdb_subpath=subpath, source="lmdb"
        )
    return sizes
