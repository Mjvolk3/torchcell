# tests/torchcell/datasets/scerevisiae/test_kuzmin2018
# [[tests.torchcell.datasets.scerevisiae.test_kuzmin2018]]
"""Record-construction tests for the Kuzmin 2018 double-mutant fitness loader.

Calls ``preprocess_raw`` / ``create_experiment`` directly on the raw Data S1 table,
so nothing is built into LMDB. Covers both record kinds: the digenic query x array
crosses, which must stay byte-identical to what is already served, and the
double-mutant QUERY strains of the trigenic screens (one record per distinct strain,
which the loader used to drop entirely). Skipped when the ``$DATA_ROOT`` raw mirror
is absent (CI without the data).
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

_RAW = osp.join(DATA_ROOT, "data/torchcell/dmf_kuzmin2018/raw/aao1729_data_s1.tsv")

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not osp.exists(_RAW),
        reason=f"requires the Kuzmin 2018 raw Data S1 TSV at {_RAW} (absent in CI)",
    ),
]

# The digenic row used by the regression test, addressed by strain ids (stable under
# any reordering). Values read off the raw table.
_DIGENIC_QUERY_STRAIN = "YAR002W+YDL227C_tm3180"
_DIGENIC_ARRAY_STRAIN = "YAL048C_dma5203"
_DIGENIC_FITNESS = 0.8103
_DIGENIC_FITNESS_SD = 0.0463

# A double-mutant query strain of the trigenic screens, with its released fitness.
_QUERY_STRAIN = "YAR002W+YML107C_tm2550"
_QUERY_STRAIN_FITNESS = 0.5128


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    """The raw Kuzmin 2018 Data S1 table."""
    return pd.read_csv(_RAW, sep="\t")


@pytest.fixture(scope="module")
def built(raw: pd.DataFrame) -> tuple[Any, pd.DataFrame]:
    """The loader instance (for phenotype_reference_std) and its preprocessed frame."""
    from torchcell.datasets.scerevisiae.kuzmin2018 import DmfKuzmin2018Dataset

    dataset = DmfKuzmin2018Dataset.__new__(DmfKuzmin2018Dataset)
    return dataset, dataset.preprocess_raw(raw.copy())


def test_trigenic_query_strains_are_kept_one_record_each(
    raw: pd.DataFrame, built: tuple[Any, pd.DataFrame]
) -> None:
    """The trigenic rows are no longer dropped, and collapse to one record/strain."""
    from torchcell.datasets.scerevisiae.kuzmin2018 import (
        RECORD_KIND_DIGENIC_ARRAY_CROSS,
        RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN,
    )

    _, df = built
    trigenic = raw[raw["Combined mutant type"] == "trigenic"]
    n_query_strains = trigenic.dropna(subset=["Query single/double mutant fitness"])[
        "Query strain ID"
    ].nunique()

    counts = df["record_kind"].value_counts().to_dict()
    assert counts[RECORD_KIND_DIGENIC_ARRAY_CROSS] == int(
        (raw["Combined mutant type"] == "digenic").sum()
    )
    assert counts[RECORD_KIND_DIGENIC_ARRAY_CROSS] == 410_399
    assert counts[RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN] == n_query_strains == 172

    # One record per strain, NOT one per trigenic row (91,111 rows, 182 strains, of
    # which 172 report a fitness).
    assert len(trigenic) == 91_111
    query_rows = df[df["record_kind"] == RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN]
    assert query_rows["Query strain ID"].is_unique
    assert query_rows["Query single/double mutant fitness"].notna().all()


def test_query_strain_record_is_the_query_pair(built: tuple[Any, pd.DataFrame]) -> None:
    """Genotype is the two QUERY genes, both tagged with the full query strain id."""
    from torchcell.datamodels.schema import Genotype
    from torchcell.datasets.scerevisiae.kuzmin2018 import (
        RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN,
        DmfKuzmin2018Dataset,
    )

    dataset, df = built
    rows = df[
        (df["record_kind"] == RECORD_KIND_DOUBLE_MUTANT_QUERY_STRAIN)
        & (df["Query strain ID"] == _QUERY_STRAIN)
    ]
    assert len(rows) == 1
    row = rows.iloc[0]

    experiment, _, _ = DmfKuzmin2018Dataset.create_experiment(
        "dmf_kuzmin2018", row, phenotype_reference_std=dataset.phenotype_reference_std
    )
    genotype = experiment.genotype
    assert isinstance(genotype, Genotype)
    perturbations = genotype.perturbations
    assert len(perturbations) == 2
    assert [p.systematic_gene_name for p in perturbations] == ["YAR002W", "YML107C"]
    assert [p.perturbed_gene_name for p in perturbations] == [
        "nup60_delta",
        "pml39_delta",
    ]
    assert {p.model_dump()["strain_id"] for p in perturbations} == {_QUERY_STRAIN}
    # The array gene of the trigenic rows this value was read off is NOT in the
    # genotype.
    assert row["Array systematic name"] not in {
        p.systematic_gene_name for p in perturbations
    }

    # Released value, no uncertainty (the SD lives in Data File S4, not a raw file
    # of this loader).
    assert experiment.phenotype.fitness == pytest.approx(_QUERY_STRAIN_FITNESS)
    assert experiment.phenotype.fitness_std is None
    assert experiment.phenotype.fitness_se is None
    assert experiment.phenotype.fitness_uncertainty is None
    assert experiment.phenotype.fitness_uncertainty_type is None


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
    from torchcell.datasets.scerevisiae.kuzmin2018 import (
        RECORD_KIND_DIGENIC_ARRAY_CROSS,
        DmfKuzmin2018Dataset,
    )

    dataset, df = built
    rows = df[
        (df["record_kind"] == RECORD_KIND_DIGENIC_ARRAY_CROSS)
        & (df["Query strain ID"] == _DIGENIC_QUERY_STRAIN)
        & (df["Array strain ID"] == _DIGENIC_ARRAY_STRAIN)
    ]
    assert len(rows) == 1

    experiment, _, _ = DmfKuzmin2018Dataset.create_experiment(
        "dmf_kuzmin2018",
        rows.iloc[0],
        phenotype_reference_std=dataset.phenotype_reference_std,
    )

    expected = FitnessExperiment(
        dataset_name="dmf_kuzmin2018",
        genotype=Genotype(
            perturbations=[
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name="YAR002W",
                    perturbed_gene_name="nup60_delta",
                    strain_id=_DIGENIC_QUERY_STRAIN,
                ),
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name="YAL048C",
                    perturbed_gene_name="gem1_delta",
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
