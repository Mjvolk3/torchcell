# tests/torchcell/datasets/scerevisiae/test_kuzmin_released_zero_sd.py
# [[tests.torchcell.datasets.scerevisiae.test_kuzmin_released_zero_sd]]
"""A combined-mutant SD the release gives as exactly 0 is stored as released (#894-#897).

Owner decision, 2026-10-12: a released record is never dropped as a remedy, and the
uncertainty a source releases is stored exactly as released; it is gapped only when the
source releases none. Kuzmin 2018 Additional Data S1 and Kuzmin 2020 Tables S1 and S3
give a combined-mutant fitness standard deviation of exactly 0 on 9 and 46 rows. The SIs
define the column only by its name ("12. Combined mutant fitness standard deviation") and
state nothing else about these rows, and the released P-value on every one of them is
also exactly 0 (``experiments/036-dataset-fixes-before-kg-build/results/
keep_records_887_sd_zero_inventory.json``). So the record keeps the row, the released 0
as ``fitness_uncertainty`` / ``fitness_std``, its ``sample_sd`` label, and the SE that
arithmetic gives (0 / sqrt(4) = 0): no drop, no substituted SE, no gap.

The hermetic tests pin the two loaders' uncertainty helpers on a released 0. The
``--data`` tests read the real released tables and pin the counts: 4 digenic and 5
trigenic rows in Kuzmin 2018, 18 + 15 digenic rows (Tables S1 + S3) in Kuzmin 2020,
each present in its loader's preprocessed frame and built into a record carrying the 0.
"""

from __future__ import annotations

import os
import os.path as osp
from typing import Any

import pandas as pd
import pytest

from torchcell.datamodels.schema import FitnessPhenotype, SampleUnit, UncertaintyType
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


# ---------------------------------------------------------------------------------------
# Data-gated tests on the real released tables.
# ---------------------------------------------------------------------------------------

_RAW_2018 = osp.join(
    os.environ["DATA_ROOT"], "data/torchcell/dmf_kuzmin2018/raw/aao1729_data_s1.tsv"
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
    not osp.exists(_RAW_2018), reason=f"requires the Kuzmin 2018 Data S1 at {_RAW_2018}"
)
def test_kuzmin2018_keeps_its_9_released_zero_sds() -> None:
    """Data S1 releases 4 digenic and 5 trigenic rows with an SD of exactly 0, each with
    a released P-value of exactly 0. The Dmf frame keeps the 4 and the Tmf frame the 5,
    and every one of them builds into a record that carries the 0 as released.
    """
    raw = pd.read_csv(_RAW_2018, sep="\t")
    zero = raw[raw[_SD_2018] == 0]
    assert zero["Combined mutant type"].value_counts().to_dict() == {
        "digenic": 4,
        "trigenic": 5,
    }
    assert set(zero["P-value"]) == {0.0}

    for cls, kind, n in (
        (k18.DmfKuzmin2018Dataset, "digenic", 4),
        (k18.TmfKuzmin2018Dataset, "trigenic", 5),
    ):
        dataset = cls.__new__(cls)
        frame = dataset.preprocess_raw(raw.copy())
        keys = zero[zero["Combined mutant type"] == kind][_KEY]
        rows = frame.merge(keys, on=_KEY)
        if cls is k18.DmfKuzmin2018Dataset:
            rows = rows[rows["record_kind"] == k18.RECORD_KIND_DIGENIC_ARRAY_CROSS]
        assert len(rows) == n
        for _, row in rows.iterrows():
            experiment, _, _ = cls.create_experiment(
                cls.__name__,
                row,
                phenotype_reference_std=dataset.phenotype_reference_std,
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
    33 digenic rows, tagged with the screen they came from, and every one of them builds
    into a record carrying the 0 as released. (``TmfKuzmin2020Dataset`` keeps the 13
    trigenic rows but stores every released SD only as ``fitness_std``, with no labeled
    uncertainty; that is a separate finding, not pinned here.)
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
