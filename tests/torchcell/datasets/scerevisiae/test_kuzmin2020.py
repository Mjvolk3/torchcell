# tests/torchcell/datasets/scerevisiae/test_kuzmin2020
# [[tests.torchcell.datasets.scerevisiae.test_kuzmin2020]]
"""Record-construction tests for the Kuzmin 2020 double-mutant fitness loader.

Calls ``preprocess_raw`` / ``create_experiment`` directly on the raw supplementary
tables, so nothing is built into LMDB. Covers both record kinds: the digenic query x
array crosses, which must stay byte-identical to what is already served, and the
double-mutant QUERY strains of the trigenic screens, whose fitness and bootstrap SD
come from Table S5 through a join on the tm number. Skipped when the ``$DATA_ROOT``
raw mirror is absent (CI without the data).
"""

import os
import os.path as osp
from typing import Any

import pandas as pd
import pytest
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
if DATA_ROOT is None:
    pytest.skip("requires DATA_ROOT data (absent in CI)", allow_module_level=True)

_RAW_DIR = osp.join(DATA_ROOT, "data/torchcell/dmf_kuzmin2020/raw")
_RAW_FILES = ["aaz5667-Table-S1.xlsx", "aaz5667-Table-S3.xlsx", "aaz5667-Table-S5.xlsx"]

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not all(osp.exists(osp.join(_RAW_DIR, f)) for f in _RAW_FILES),
        reason=f"requires the Kuzmin 2020 raw supplementary tables in {_RAW_DIR}",
    ),
]

# The digenic row used by the regression test, addressed by strain ids.
_DIGENIC_QUERY_STRAIN = "YAL015C+YDL227C_tm461"
_DIGENIC_ARRAY_STRAIN = "YBL007C_dma91"
_DIGENIC_FITNESS = 0.9695
_DIGENIC_FITNESS_SD = 0.0465

# A double-mutant query strain of the trigenic screens (Table S5 entry "tm72").
_QUERY_STRAIN = "YAL015C+YOL043C_tm72"
_QUERY_STRAIN_FITNESS = 1.0133
_QUERY_STRAIN_SD = 0.008


@pytest.fixture(scope="module")
def raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tables S1, S3, and S5 as released."""
    df_s1, df_s3, df_s5 = (
        pd.read_excel(osp.join(_RAW_DIR, f), skiprows=1) for f in _RAW_FILES
    )
    return df_s1, df_s3, df_s5


@pytest.fixture(scope="module")
def built(
    raw: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> tuple[Any, pd.DataFrame]:
    """The loader instance and its preprocessed frame."""
    from torchcell.datasets.scerevisiae.kuzmin2020 import DmfKuzmin2020Dataset

    df_s1, df_s3, df_s5 = raw
    dataset = DmfKuzmin2020Dataset.__new__(DmfKuzmin2020Dataset)
    return dataset, dataset.preprocess_raw(df_s1.copy(), df_s3.copy(), df_s5.copy())


@pytest.mark.slow
def test_trigenic_query_strains_are_kept_one_record_each(
    raw: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    built: tuple[Any, pd.DataFrame],
) -> None:
    """The trigenic rows are no longer dropped, and collapse to one record/strain."""
    from torchcell.datasets.scerevisiae.kuzmin2020 import (
        RECORD_KIND_DIGENIC_ARRAY_CROSS,
        RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN,
    )

    df_s1, df_s3, _ = raw
    _, df = built
    combined = pd.concat([df_s1, df_s3])
    trigenic = combined[combined["Combined mutant type"] == "trigenic"]

    counts = df["record_kind"].value_counts().to_dict()
    assert counts[RECORD_KIND_DIGENIC_ARRAY_CROSS] == int(
        (combined["Combined mutant type"] == "digenic").sum()
    )
    assert counts[RECORD_KIND_DIGENIC_ARRAY_CROSS] == 632_797
    assert counts[RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN] == 201

    # One record per strain, not one per trigenic row.
    assert len(trigenic) == 301_798
    query_rows = df[df["record_kind"] == RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN]
    assert query_rows["Query strain ID"].is_unique
    assert query_rows["fitness"].notna().all()


@pytest.mark.slow
def test_table_s5_join_is_on_the_tm_number(
    raw: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    built: tuple[Any, pd.DataFrame],
) -> None:
    """The corrected join matches; the old full-strain-id join matched nothing."""
    from torchcell.datasets.scerevisiae.kuzmin2020 import (
        RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN,
    )

    df_s1, df_s3, df_s5 = raw
    combined = pd.concat([df_s1, df_s3])
    trigenic = combined[combined["Combined mutant type"] == "trigenic"]
    s5_double = df_s5[df_s5["Mutant type"] == "Double mutant"]

    # The defect: S5 keys on the bare tm number, S1/S3 on "<ORF>+<ORF>_tm<N>".
    assert not set(trigenic["Query strain ID"]) & set(s5_double["Query Strain ID"])
    tm = trigenic["Query strain ID"].str.rsplit("_", n=1).str[-1]
    matched = trigenic[tm.isin(set(s5_double["Query Strain ID"]))]
    assert len(matched) == 301_798

    # Every query-strain record's SD is the S5 "St.dev." for its tm number.
    _, df = built
    query_rows = df[df["record_kind"] == RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN]
    s5_sd = s5_double.set_index("Query Strain ID")["St.dev."]
    from_s5 = (
        query_rows["Query strain ID"].str.rsplit("_", n=1).str[-1].map(s5_sd).to_numpy()
    )
    assert query_rows["fitness_std"].to_numpy() == pytest.approx(from_s5)
    assert query_rows["fitness_std"].notna().all()


@pytest.mark.slow
def test_query_strain_record_is_the_query_pair(built: tuple[Any, pd.DataFrame]) -> None:
    """Genotype is the two QUERY genes; fitness and bootstrap SD come from S5."""
    from torchcell.datamodels.schema import Genotype, SampleUnit, UncertaintyType
    from torchcell.datasets.scerevisiae.kuzmin2020 import (
        N_SAMPLES_QUERY_STRAIN_FITNESS,
        RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN,
        DmfKuzmin2020Dataset,
    )

    _, df = built
    rows = df[
        (df["record_kind"] == RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN)
        & (df["Query strain ID"] == _QUERY_STRAIN)
    ]
    assert len(rows) == 1
    row = rows.iloc[0]

    experiment, _, _ = DmfKuzmin2020Dataset.create_experiment("dmf_kuzmin2020", row)
    genotype = experiment.genotype
    assert isinstance(genotype, Genotype)
    perturbations = genotype.perturbations
    assert len(perturbations) == 2
    assert [p.systematic_gene_name for p in perturbations] == ["YAL015C", "YOL043C"]
    # kuzmin2020 names perturbed genes without the "_delta" allele suffix, as its
    # digenic / Tmf / Tmi records do.
    assert [p.perturbed_gene_name for p in perturbations] == ["ntg1", "ntg2"]
    assert {p.model_dump()["strain_id"] for p in perturbations} == {_QUERY_STRAIN}
    assert row["Array systematic name"] not in {
        p.systematic_gene_name for p in perturbations
    }

    phenotype = experiment.phenotype
    assert phenotype.fitness == pytest.approx(_QUERY_STRAIN_FITNESS)
    assert phenotype.fitness_std == pytest.approx(_QUERY_STRAIN_SD)
    # A bootstrap SD of the estimator is already an SE: used as-is, not divided.
    assert phenotype.fitness_uncertainty_type == UncertaintyType.bootstrap_se
    assert phenotype.fitness_se == pytest.approx(_QUERY_STRAIN_SD)
    assert phenotype.n_samples == N_SAMPLES_QUERY_STRAIN_FITNESS == 12
    assert phenotype.sample_unit == SampleUnit.colony


@pytest.mark.slow
def test_digenic_record_unchanged(built: tuple[Any, pd.DataFrame]) -> None:
    """Regression: a digenic record equals the object the old path produced."""
    from torchcell.datamodels.media import SGA_TM_SELECTION
    from torchcell.datamodels.schema import (
        Environment,
        FitnessExperiment,
        FitnessPhenotype,
        Genotype,
        SampleUnit,
        SgaKanMxDeletionPerturbation,
        Temperature,
        UncertaintyType,
    )
    from torchcell.datasets.scerevisiae.kuzmin2020 import (
        RECORD_KIND_DIGENIC_ARRAY_CROSS,
        DmfKuzmin2020Dataset,
    )

    _, df = built
    rows = df[
        (df["record_kind"] == RECORD_KIND_DIGENIC_ARRAY_CROSS)
        & (df["Query strain ID"] == _DIGENIC_QUERY_STRAIN)
        & (df["Array strain ID"] == _DIGENIC_ARRAY_STRAIN)
    ]
    assert len(rows) == 1

    experiment, _, _ = DmfKuzmin2020Dataset.create_experiment(
        "dmf_kuzmin2020", rows.iloc[0]
    )

    expected = FitnessExperiment(
        dataset_name="dmf_kuzmin2020",
        genotype=Genotype(
            perturbations=[
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name="YAL015C",
                    perturbed_gene_name="ntg1",
                    strain_id=_DIGENIC_QUERY_STRAIN,
                ),
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name="YBL007C",
                    perturbed_gene_name="sla1",
                    strain_id=_DIGENIC_ARRAY_STRAIN,
                ),
            ]
        ),
        environment=Environment(
            media=SGA_TM_SELECTION, temperature=Temperature(value=26)
        ),
        phenotype=FitnessPhenotype(
            fitness=_DIGENIC_FITNESS,
            fitness_std=_DIGENIC_FITNESS_SD,
            fitness_uncertainty=_DIGENIC_FITNESS_SD,
            fitness_uncertainty_type=UncertaintyType.sample_sd,
            n_samples=4,
            sample_unit=SampleUnit.colony,
        ),
    )
    assert experiment.model_dump() == expected.model_dump()
