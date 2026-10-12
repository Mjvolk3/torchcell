# tests/torchcell/datasets/scerevisiae/test_kuzmin_released_zero_sd.py
# [[tests.torchcell.datasets.scerevisiae.test_kuzmin_released_zero_sd]]
"""A fitness SD the release gives as exactly 0 is stored as released (#894-#930).

Owner decision, 2026-10-12: a released record is never dropped as a remedy, and the
uncertainty a source releases is stored exactly as released ("the data is the data");
it is gapped only when the source releases none. Kuzmin 2018 Additional Data S1 and
Kuzmin 2020 Tables S1 and S3 give a combined-mutant fitness standard deviation of
exactly 0 on 9 and 46 rows, Costanzo 2016 Data File S1 on 2,781. The SIs define the
column only by its name ("12. Combined mutant fitness standard deviation") and state
nothing else about these rows, and the released P-value is also exactly 0 on every
Kuzmin one (``experiments/036-dataset-fixes-before-kg-build/results/
keep_records_887_sd_zero_inventory.json``). So the record keeps the row, the released 0
as ``fitness_uncertainty`` / ``fitness_std``, its ``sample_sd`` label, and the SE that
arithmetic gives (0 / sqrt(4) = 0): no drop, no substituted SE, no gap.

The hermetic tests pin each loader's rule on a released 0. The ``--data`` tests read
the real released tables and pin the counts: 4 digenic and 5 trigenic rows in Kuzmin
2018, 18 + 15 digenic and 6 + 7 trigenic rows (Tables S1 + S3) in Kuzmin 2020, each
present in its loader's preprocessed frame and built into a record carrying the 0.
"""

from __future__ import annotations

import os
import os.path as osp
from typing import Any

import pandas as pd
import pytest

from torchcell.datamodels.schema import FitnessPhenotype, SampleUnit, UncertaintyType
from torchcell.datasets.scerevisiae import costanzo2016 as c16
from torchcell.datasets.scerevisiae import kuzmin2018 as k18
from torchcell.datasets.scerevisiae import kuzmin2020 as k20

#: The fields every released-zero record must carry, as released.
_AS_RELEASED = {
    "fitness_std": 0.0,
    "fitness_uncertainty": 0.0,
    "fitness_uncertainty_type": UncertaintyType.sample_sd,
    "n_samples": 4,
    "sample_unit": SampleUnit.colony,
    "fitness_se": 0.0,
    "provenance_gaps": [],
}


def _stored(phenotype: FitnessPhenotype) -> dict[str, Any]:
    """The uncertainty fields of a phenotype, as ``_AS_RELEASED`` names them."""
    return {name: getattr(phenotype, name) for name in _AS_RELEASED}


@pytest.mark.parametrize("module", [k18, k20], ids=["kuzmin2018", "kuzmin2020"])
def test_a_released_zero_sd_is_stored_as_released(module: Any) -> None:
    """``_combined_mutant_uncertainty(0.0)`` labels the 0 a sample SD over 4 colonies,
    ``_reported_sd(0.0)`` keeps it, and the phenotype derives an SE of 0.0. Nothing is
    dropped to None and no ``ProvenanceGap`` is declared.
    """
    assert module._combined_mutant_uncertainty(0.0) == {
        "fitness_uncertainty": 0.0,
        "fitness_uncertainty_type": UncertaintyType.sample_sd,
        "n_samples": 4,
        "sample_unit": SampleUnit.colony,
    }
    assert module._reported_sd(0.0) == 0.0
    phenotype = FitnessPhenotype(
        fitness=0.2869,
        fitness_std=module._reported_sd(0.0),
        **module._combined_mutant_uncertainty(0.0),
    )
    assert _stored(phenotype) == _AS_RELEASED


def _costanzo_row(sd: float) -> pd.Series:
    """One preprocessed Costanzo 2016 DMF row (KanMX query x KanMX array, 30 C)."""
    return pd.Series(
        {
            "Query Strain ID": "YAL002W_sn273",
            "Query allele name": "vps8",
            "Query Systematic Name": "YAL002W",
            "query_perturbation_type": "KanMX_deletion",
            "Array Strain ID": "YBL001C_dma94",
            "Array allele name": "ecm15",
            "Array Systematic Name": "YBL001C",
            "array_perturbation_type": "KanMX_deletion",
            "Temperature": 30,
            "Double mutant fitness": 0.8973,
            "Double mutant fitness standard deviation": sd,
        }
    )


def test_costanzo_dmf_keeps_a_released_zero_sd() -> None:
    """#930: Costanzo 2016's DMF record keeps a released 0 as a sample SD over 4
    colonies with SE 0; a nonzero SD (0.0269) gives SE 0.0269 / 2.
    """
    zero, _, _ = c16.DmfCostanzo2016Dataset.create_experiment(
        "DmfCostanzo2016Dataset", _costanzo_row(0.0), 0.05, 0.05
    )
    assert _stored(zero.phenotype) == _AS_RELEASED
    nonzero, _, _ = c16.DmfCostanzo2016Dataset.create_experiment(
        "DmfCostanzo2016Dataset", _costanzo_row(0.0269), 0.05, 0.05
    )
    assert nonzero.phenotype.fitness_se == pytest.approx(0.0269 / 2)


# ---------------------------------------------------------------------------------------
# Data-gated tests on the real released tables.
# ---------------------------------------------------------------------------------------

_RAW_2018 = osp.join(
    os.environ["DATA_ROOT"], "data/torchcell/dmf_kuzmin2018/raw/aao1729_data_s1.tsv"
)
_S4_2018 = osp.join(
    os.environ["DATA_ROOT"], "torchcell-raw", k18.CITATION_KEY, "data", k18.S4_NAME
)
_RAW_2020 = osp.join(os.environ["DATA_ROOT"], "data/torchcell/dmf_kuzmin2020/raw")
_TABLES_2020 = [
    "aaz5667-Table-S1.xlsx",
    "aaz5667-Table-S3.xlsx",
    "aaz5667-Table-S5.xlsx",
]
_SD_2018 = "Combined mutant fitness standard deviation"
_SD_2020 = "Double/triple mutant fitness standard deviation"
_KEY = ["Query strain ID", "Array strain ID"]


@pytest.mark.data
@pytest.mark.skipif(
    not (osp.exists(_RAW_2018) and osp.exists(_S4_2018)),
    reason=f"requires the Kuzmin 2018 Data S1 at {_RAW_2018} and S4 at {_S4_2018}",
)
def test_kuzmin2018_keeps_its_9_released_zero_sds() -> None:
    """Data S1 releases 4 digenic and 5 trigenic rows with an SD of exactly 0, each with
    a released P-value of exactly 0. The Dmf frame keeps the 4 and the Tmf frame the 5,
    and every one of them builds into a record carrying the 0 as released.
    """
    raw = pd.read_csv(_RAW_2018, sep="\t")
    zero = raw[raw[_SD_2018] == 0]
    assert zero["Combined mutant type"].value_counts().to_dict() == {
        "digenic": 4,
        "trigenic": 5,
    }
    assert set(zero["P-value"]) == {0.0}

    dmf = k18.DmfKuzmin2018Dataset.__new__(k18.DmfKuzmin2018Dataset)
    tmf = k18.TmfKuzmin2018Dataset.__new__(k18.TmfKuzmin2018Dataset)
    dmf_frame = dmf.preprocess_raw(raw.copy(), pd.read_excel(_S4_2018))
    tmf_frame = tmf.preprocess_raw(raw.copy())
    digenic = dmf_frame.merge(zero[zero["Combined mutant type"] == "digenic"][_KEY])
    digenic = digenic[digenic["record_kind"] == k18.RECORD_KIND_DIGENIC_ARRAY_CROSS]
    trigenic = tmf_frame.merge(zero[zero["Combined mutant type"] == "trigenic"][_KEY])
    assert (len(digenic), len(trigenic)) == (4, 5)
    for _, row in digenic.iterrows():
        experiment, _, _ = k18.DmfKuzmin2018Dataset.create_experiment(
            "DmfKuzmin2018Dataset",
            row,
            phenotype_reference_std=dmf.phenotype_reference_std,
        )
        assert _stored(experiment.phenotype) == _AS_RELEASED
    for _, row in trigenic.iterrows():
        experiment, _, _ = k18.TmfKuzmin2018Dataset.create_experiment(
            "TmfKuzmin2018Dataset",
            row,
            phenotype_reference_std=tmf.phenotype_reference_std,
        )
        assert _stored(experiment.phenotype) == _AS_RELEASED


@pytest.mark.data
@pytest.mark.skipif(
    not all(osp.exists(osp.join(_RAW_2020, name)) for name in _TABLES_2020),
    reason=f"requires the Kuzmin 2020 Tables S1, S3 and S5 in {_RAW_2020}",
)
def test_kuzmin2020_keeps_its_33_released_zero_sd_digenic_rows() -> None:
    """Tables S1 and S3 release 18 and 15 digenic rows with an SD of exactly 0, and 6 and
    7 trigenic ones, each with a released P-value of exactly 0. The Dmf frame keeps all
    33 digenic rows and the Tmf frame all 13 trigenic ones, each tagged with the screen
    it came from, and every one builds into a record carrying the 0 as released
    (Tmf labels its released SD since 2026.10.12).
    """
    df_s1, df_s3, df_s5 = (
        pd.read_excel(osp.join(_RAW_2020, name), skiprows=1) for name in _TABLES_2020
    )
    counts = {}
    for label, table in (("S1", df_s1), ("S3", df_s3)):
        zero = table[table[_SD_2020] == 0]
        counts[label] = zero["Combined mutant type"].value_counts().to_dict()
        assert set(zero["P-value"]) == {0.0}
    assert counts == {
        "S1": {"digenic": 18, "trigenic": 6},
        "S3": {"digenic": 15, "trigenic": 7},
    }

    dataset = k20.DmfKuzmin2020Dataset.__new__(k20.DmfKuzmin2020Dataset)
    frame = dataset.preprocess_raw(df_s1.copy(), df_s3.copy(), df_s5.copy())
    rows = frame[
        (frame["record_kind"] == k20.RECORD_KIND_DIGENIC_ARRAY_CROSS)
        & (frame["fitness_std"] == 0)
    ]
    assert rows["screen_id"].value_counts().to_dict() == {
        k20.SCREEN_ID_MAIN: 18,
        k20.SCREEN_ID_PILOT: 15,
    }
    for _, row in rows.iterrows():
        experiment, _, _ = k20.DmfKuzmin2020Dataset.create_experiment(
            "DmfKuzmin2020Dataset", row
        )
        assert _stored(experiment.phenotype) == _AS_RELEASED

    tmf = k20.TmfKuzmin2020Dataset.__new__(k20.TmfKuzmin2020Dataset)
    trigenic = tmf.preprocess_raw(df_s1.copy(), df_s3.copy())
    zero = trigenic[trigenic["fitness_std"] == 0]
    assert zero["screen_id"].value_counts().to_dict() == {
        k20.SCREEN_ID_MAIN: 6,
        k20.SCREEN_ID_PILOT: 7,
    }
    for _, row in zero.iterrows():
        experiment, _, _ = k20.TmfKuzmin2020Dataset.create_experiment(
            "TmfKuzmin2020Dataset", row, phenotype_reference_std=0.05
        )
        assert _stored(experiment.phenotype) == _AS_RELEASED
