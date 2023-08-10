# tests/torchcell/sequence/test_sequence.py
import pandas as pd
import pytest
from tqdm import tqdm

from torchcell.sequence import DnaSelectionResult, DnaWindowResult
from torchcell.data_prior.sequence import SCerevisiaeGenome


@pytest.fixture
def genome():
    return SCerevisiaeGenome()


def test_gene_list(genome: SCerevisiaeGenome) -> None:
    gene_list = genome.gene_set
    assert isinstance(gene_list, list)
    assert len(gene_list) == len(set(gene_list)), "Duplicate genes found"


def test_feature_types(genome: SCerevisiaeGenome) -> None:
    feature_types = genome.feature_types
    assert isinstance(feature_types, list)


def test_gene_attribute_table(genome: SCerevisiaeGenome) -> None:
    gene_attribute_table = genome.gene_attribute_table
    assert isinstance(gene_attribute_table, pd.DataFrame)


def test_get_seq(genome: SCerevisiaeGenome) -> None:
    result = genome.get_seq(chr=0, start=10, end=120, strand="+")
    assert isinstance(result, DnaSelectionResult)


def test_select_feature_window(genome: SCerevisiaeGenome) -> None:
    # Assuming "YFL039C" is a valid feature in your data
    result = genome.select_feature_window("YFL039C", 20000, is_max_size=False)
    assert isinstance(result, DnaWindowResult)


def test_select_feature_window_with_max_size(genome: SCerevisiaeGenome) -> None:
    for gene in tqdm(genome.gene_set):
        result = genome.select_feature_window(gene, 60000, is_max_size=True)
        assert isinstance(result, DnaWindowResult)
        assert result.start_window < result.start
        assert result.end_window > result.end
        assert result.start < result.end
        assert result.size == 60000


def test_select_feature_window_without_max_size(genome: SCerevisiaeGenome) -> None:
    for gene in tqdm(genome.gene_set):
        result = genome.select_feature_window(gene, 60000, is_max_size=False)
        assert isinstance(result, DnaWindowResult)
        assert result.start_window < result.start
        assert result.end_window > result.end
        assert result.start < result.end
        assert (result.end - result.start) <= (result.end_window - result.start_window)


# Tests for validation
def test_dna_selection_result_validation() -> None:
    with pytest.raises(ValueError):
        DnaSelectionResult(seq="", chromosome=1, start=5, end=3, strand="+")

    with pytest.raises(ValueError):
        DnaSelectionResult(seq="ATCG", chromosome=-1, start=5, end=10, strand="+")

    with pytest.raises(ValueError):
        DnaSelectionResult(seq="ATCG", chromosome=1, start=-5, end=10, strand="+")

    with pytest.raises(ValueError):
        DnaSelectionResult(seq="ATCG", chromosome=1, start=5, end=-10, strand="+")

    with pytest.raises(ValueError):
        DnaSelectionResult(seq="ATCG", chromosome=1, start=5, end=10, strand="x")


def test_dna_window_result_validation() -> None:
    with pytest.raises(ValueError):
        DnaWindowResult(
            seq="ATCG",
            chromosome=1,
            start=5,
            end=10,
            strand="+",
            size=5,
            start_window=10,
            end_window=5,
        )

    with pytest.raises(ValueError):
        DnaWindowResult(
            seq="ATCG",
            chromosome=1,
            start=5,
            end=10,
            strand="+",
            size=5,
            start_window=-1,
            end_window=5,
        )

    with pytest.raises(ValueError):
        DnaWindowResult(
            seq="ATCG",
            chromosome=1,
            start=5,
            end=10,
            strand="+",
            size=5,
            start_window=10,
            end_window=-1,
        )


if __name__ == "__main__":
    genome = SCerevisiaeGenome()
    test_select_feature_window_with_max_size(genome)
    pass
