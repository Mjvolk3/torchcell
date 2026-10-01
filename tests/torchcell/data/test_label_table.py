# tests/torchcell/data/test_label_table.py
# [[tests.torchcell.data.test_label_table]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/data/test_label_table.py
"""``label_table`` on a three-record no-merge build under ``tmp_path``.

``tests/torchcell/data/test_label_policy.py`` covers the policy and the success paths of
``triple_roles``; this file pins the remaining branches, the record normalizer, and the
table builder on a store written by hand in the processed-LMDB shape: key ``str(i)``,
value ``json.dumps([{"experiment": model_dump(), "experiment_reference": model_dump()},
...])``.

Records (default ``LabelPolicy(name="kuzmin-first")``, precedence kuzmin2018 >
kuzmin2020 > costanzo2016@30 > costanzo2016@26 > converted_zero):

* ``0`` Costanzo 30 C fitness 0.9 (sd 0.1, n 4, strain ``YAL001C_dma1``) and an SGD
  converted 0: the zero yields to the measurement, so fitness 0.9 from
  ``costanzo2016@30`` with 2 entries available, 1 combined; no interaction.
* ``1`` Kuzmin 2018 fitness 0.5, Costanzo 30 C fitness 0.7, Kuzmin 2018 interaction
  -0.2 (p 0.01): fitness 0.5 from ``kuzmin2018`` (2 available), interaction -0.2.
* ``2`` Costanzo 30 C interaction 0.1 (p 0.5) only: no fitness (0 entries).

``perturbation_count_index.json`` is ``{"1": [0, 2], "2": [1]}``, so the full pass
covers indices 0, 1, 2. The kuzmin-first ``policy_id`` is ``b2f26bb6ef86d11a``
(``LabelPolicy(name="kuzmin-first").policy_id`` in a shell), so the cache path is
``<root>/label_tables/b2f26bb6ef86d11a.parquet``.
"""

from __future__ import annotations

import json
import math
import os.path as osp
from pathlib import Path
from typing import Any

import lmdb
import pandas as pd
import pytest

from torchcell.data import label_table
from torchcell.data.label_policy import LabelEntry, LabelPolicy
from torchcell.data.label_table import (
    TripleRoles,
    _finite,
    build_label_table,
    entries_of_record,
    label_table_path,
    triple_roles,
)
from torchcell.datamodels.schema import (
    Environment,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    GeneInteractionExperiment,
    GeneInteractionExperimentReference,
    GeneInteractionPhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    ReferenceGenome,
    SgaKanMxDeletionPerturbation,
    Temperature,
)

POLICY = LabelPolicy(name="kuzmin-first")
POLICY_ID = "b2f26bb6ef86d11a"
MEDIA = Media(name="YPD", state="solid", is_synthetic=False)
AT_30 = Environment(media=MEDIA, temperature=Temperature(value=30.0))
NO_TEMP = Environment(media=MEDIA)
GENOME = ReferenceGenome(species="Saccharomyces cerevisiae", strain="S288C")


def _perturbations(genes: dict[str, str | None]) -> Genotype:
    """One perturbation per gene; a strain id makes it the SGA variant."""
    perts: list[Any] = []
    for gene, strain in genes.items():
        if strain is None:
            perts.append(
                KanMxDeletionPerturbation(
                    systematic_gene_name=gene, perturbed_gene_name=gene
                )
            )
        else:
            perts.append(
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=gene,
                    perturbed_gene_name=gene,
                    strain_id=strain,
                )
            )
    return Genotype(perturbations=perts)


def _fitness(
    dataset: str,
    environment: Environment,
    genes: dict[str, str | None],
    fitness: float,
    std: float | None = None,
    n: int | None = None,
) -> dict[str, Any]:
    return {
        "experiment": FitnessExperiment(
            dataset_name=dataset,
            genotype=_perturbations(genes),
            environment=environment,
            phenotype=FitnessPhenotype(fitness=fitness, fitness_std=std, n_samples=n),
        ),
        "experiment_reference": FitnessExperimentReference(
            dataset_name=dataset,
            genome_reference=GENOME,
            environment_reference=environment,
            phenotype_reference=FitnessPhenotype(fitness=1.0),
        ),
    }


def _interaction(
    dataset: str,
    environment: Environment,
    genes: dict[str, str | None],
    score: float,
    p: float,
) -> dict[str, Any]:
    return {
        "experiment": GeneInteractionExperiment(
            dataset_name=dataset,
            genotype=_perturbations(genes),
            environment=environment,
            phenotype=GeneInteractionPhenotype(
                gene_interaction=score, gene_interaction_p_value=p
            ),
        ),
        "experiment_reference": GeneInteractionExperimentReference(
            dataset_name=dataset,
            genome_reference=GENOME,
            environment_reference=environment,
            phenotype_reference=GeneInteractionPhenotype(gene_interaction=0.0),
        ),
    }


def _serialize(records: list[dict[str, Any]]) -> bytes:
    return json.dumps(
        [{k: v.model_dump() for k, v in r.items()} for r in records]
    ).encode()


RECORD_0 = [
    _fitness("DmfCostanzo2016Dataset", AT_30, {"YAL001C": "YAL001C_dma1"}, 0.9, 0.1, 4),
    _fitness("GeneEssentialitySgdDataset", NO_TEMP, {"YAL001C": None}, 0.0),
]
RECORD_1 = [
    _fitness(
        "TmfKuzmin2018Dataset",
        NO_TEMP,
        {"YAL001C": "YAL001C_dma1", "YAL002W": "YAL001C+YAL002W_tm7"},
        0.5,
    ),
    _fitness(
        "DmfCostanzo2016Dataset",
        AT_30,
        {"YAL001C": "YAL001C_dma1", "YAL002W": "YAL002W_dma2"},
        0.7,
        0.2,
        4,
    ),
    _interaction(
        "TmfKuzmin2018Dataset",
        NO_TEMP,
        {"YAL001C": "YAL001C_dma1", "YAL002W": "YAL001C+YAL002W_tm7"},
        -0.2,
        0.01,
    ),
]
RECORD_2 = [
    _interaction(
        "DmfCostanzo2016Dataset",
        AT_30,
        {"YAL003W": "YAL003W_dma3", "YAL004W": "YAL003W_dma3"},
        0.1,
        0.5,
    )
]
RECORDS = [RECORD_0, RECORD_1, RECORD_2]

ENTRIES_0 = [
    LabelEntry(
        source="costanzo2016@30",
        label="fitness",
        value=0.9,
        sd=0.1,
        n_samples=4,
        p_value=None,
        strain_id="YAL001C_dma1",
    ),
    LabelEntry(source="converted_zero", label="fitness", value=0.0),
]
ENTRIES_1 = [
    LabelEntry(source="kuzmin2018", label="fitness", value=0.5),
    LabelEntry(
        source="costanzo2016@30", label="fitness", value=0.7, sd=0.2, n_samples=4
    ),
    LabelEntry(source="kuzmin2018", label="gene_interaction", value=-0.2, p_value=0.01),
]
ENTRIES_2 = [
    LabelEntry(
        source="costanzo2016@30",
        label="gene_interaction",
        value=0.1,
        p_value=0.5,
        strain_id="YAL003W_dma3",
    )
]


def _row(
    index: int, fitness: tuple[Any, ...] | None, interaction: tuple[Any, ...] | None
) -> dict[str, Any]:
    """A table row from (value, source, p, sd, n_entries, n_combined) per label."""
    row: dict[str, Any] = {"index": index}
    for label, choice in (("fitness", fitness), ("gene_interaction", interaction)):
        value, source, p, sd, n_entries, n_combined = choice or (
            None,
            None,
            None,
            None,
            0,
            0,
        )
        row[label] = value
        row[f"{label}_source"] = source
        row[f"{label}_p"] = p
        row[f"{label}_sd"] = sd
        row[f"{label}_n_entries"] = n_entries
        row[f"{label}_n_combined"] = n_combined
    return row


EXPECTED_ROWS = [
    _row(0, (0.9, "costanzo2016@30", None, 0.1, 2, 1), None),
    _row(
        1, (0.5, "kuzmin2018", None, None, 2, 1), (-0.2, "kuzmin2018", 0.01, None, 1, 1)
    ),
    _row(2, None, (0.1, "costanzo2016@30", 0.5, None, 1, 1)),
]


def _dumps(entries: list[LabelEntry]) -> list[dict[str, Any]]:
    return [e.model_dump() for e in entries]


@pytest.fixture
def build_root(tmp_path: Path) -> str:
    """A no-merge build: ``processed/lmdb`` with three records plus the count index."""
    processed = tmp_path / "processed"
    processed.mkdir()
    env = lmdb.open(str(processed / "lmdb"), map_size=int(1e8))
    with env.begin(write=True) as txn:
        for i, records in enumerate(RECORDS):
            txn.put(str(i).encode(), _serialize(records))
    env.close()
    with open(processed / "perturbation_count_index.json", "w") as f:
        json.dump({"1": [0, 2], "2": [1]}, f)
    return str(tmp_path)


# ------------------------------------------------------------------ triple_roles
def test_triple_roles_reads_a_tsa_array_and_a_bare_query_partner() -> None:
    """The tsa plate names the array; one tm token among the other two is enough."""
    roles = triple_roles(
        [
            {"systematic_gene_name": "YAL003W", "strain_id": "YAL003W_tsa12"},
            {"systematic_gene_name": "YAL001C", "strain_id": "YAL001C_tm9"},
            {"systematic_gene_name": "YAL002W", "strain_id": None},
        ]
    )
    assert roles == TripleRoles(
        array_gene="YAL003W",
        query_genes=("YAL001C", "YAL002W"),
        query_strain_id="YAL001C_tm9",
    )


def test_triple_roles_is_none_when_the_gene_count_is_not_three() -> None:
    """Four genes, or three perturbations naming only two genes, are not a triple."""
    four = [
        {"systematic_gene_name": g, "strain_id": f"{g}_dma1" if i == 0 else f"{g}_tm1"}
        for i, g in enumerate(["YAL001C", "YAL002W", "YAL003W", "YAL004W"])
    ]
    assert triple_roles(four) is None
    repeated = [
        {"systematic_gene_name": "YAL001C", "strain_id": "YAL001C_dma1"},
        {"systematic_gene_name": "YAL002W", "strain_id": "YAL002W_tm1"},
        {"systematic_gene_name": "YAL002W", "strain_id": "YAL002W_tm1"},
    ]
    assert triple_roles(repeated) is None


def test_triple_roles_is_none_without_exactly_one_array_strain() -> None:
    """No dma/tsa strain, or two of them, leaves the array gene undecided."""
    none_array = [
        {"systematic_gene_name": "YAL001C", "strain_id": "YAL001C_tm1"},
        {"systematic_gene_name": "YAL002W", "strain_id": "YAL002W_tm1"},
        {"systematic_gene_name": "YAL003W", "strain_id": ""},
    ]
    assert triple_roles(none_array) is None
    two_arrays = [
        {"systematic_gene_name": "YAL001C", "strain_id": "YAL001C_dma1"},
        {"systematic_gene_name": "YAL002W", "strain_id": "YAL002W_tsa2"},
        {"systematic_gene_name": "YAL003W", "strain_id": "YAL003W_tm1"},
    ]
    assert triple_roles(two_arrays) is None


def test_triple_roles_is_none_when_the_query_pair_has_no_single_tm_token() -> None:
    """Two different tm strains, or none at all, do not name one query strain."""
    two_tokens = [
        {"systematic_gene_name": "YAL001C", "strain_id": "YAL001C_dma1"},
        {"systematic_gene_name": "YAL002W", "strain_id": "YAL002W_tm1"},
        {"systematic_gene_name": "YAL003W", "strain_id": "YAL003W_tm2"},
    ]
    assert triple_roles(two_tokens) is None
    no_token = [
        {"systematic_gene_name": "YAL001C", "strain_id": "YAL001C_dma1"},
        {"systematic_gene_name": "YAL002W", "strain_id": "YAL002W"},
        {"systematic_gene_name": "YAL003W"},
    ]
    assert triple_roles(no_token) is None


# ------------------------------------------------------------ helpers and paths
def test_label_table_path_is_the_policy_id_parquet_under_label_tables() -> None:
    """``<root>/label_tables/<policy_id>.parquet`` with the kuzmin-first id."""
    assert POLICY.policy_id == POLICY_ID
    assert label_table_path("/build", POLICY) == (
        f"/build/label_tables/{POLICY_ID}.parquet"
    )


def test_finite_accepts_numbers_and_numeric_strings_only() -> None:
    """None, non-numeric, NaN and infinities are None; ints and strings become floats."""
    assert _finite(None) is None
    assert _finite(3) == 3.0
    assert _finite("0.5") == 0.5
    assert _finite("abc") is None
    assert _finite([1.0]) is None
    assert _finite(math.nan) is None
    assert _finite(math.inf) is None
    assert _finite(-math.inf) is None


def test_entries_of_record_normalizes_each_stored_entry() -> None:
    """Sources, labels, statistics and the shared strain id, in stored order."""
    assert _dumps(entries_of_record(_serialize(RECORD_0))) == _dumps(ENTRIES_0)
    assert _dumps(entries_of_record(_serialize(RECORD_1))) == _dumps(ENTRIES_1)
    assert _dumps(entries_of_record(_serialize(RECORD_2))) == _dumps(ENTRIES_2)


def test_entries_of_record_skips_a_missing_value_and_refuses_an_unknown_dataset() -> (
    None
):
    """A null fitness drops the entry; a dataset outside source_key is an error."""
    items = json.loads(_serialize(RECORD_0))
    items[0]["experiment"]["phenotype"]["fitness"] = None
    assert _dumps(entries_of_record(json.dumps(items).encode())) == _dumps(
        ENTRIES_0[1:]
    )
    unknown = _fitness("ToyDataset", AT_30, {"YAL001C": None}, 0.5)
    with pytest.raises(ValueError, match="no source key for dataset 'ToyDataset'"):
        entries_of_record(_serialize([unknown]))


def test_entries_of_record_refuses_an_experiment_type_no_policy_ranks() -> None:
    """A stored type other than fitness or gene interaction is refused (issue #527),
    where it used to be read as fitness because "interaction" was not in its name.
    """
    items = json.loads(_serialize(RECORD_0))
    items[0]["experiment"]["experiment_type"] = "calmorph"
    with pytest.raises(ValueError, match=r"^no label for experiment type 'calmorph'$"):
        entries_of_record(json.dumps(items).encode())


# ------------------------------------------------------------------ the table
def test_rows_reads_the_requested_indices_and_skips_absent_keys(
    build_root: str,
) -> None:
    """Indices 0, 1 and the absent 5 give the two rows of records 0 and 1."""
    label_table._init(build_root)
    try:
        rows = label_table._rows(([0, 1, 5], POLICY.model_dump(mode="json")))
    finally:
        assert label_table._ENV is not None
        label_table._ENV.close()
        label_table._ENV = None
    assert rows == EXPECTED_ROWS[:2]
    assert label_table._BUILD_ROOT == build_root


def test_build_label_table_gives_one_row_per_record_and_caches_it(
    build_root: str,
) -> None:
    """Three rows in index order; the cached parquet reads back to the same table."""
    table = build_label_table(POLICY, build_root, workers=1, chunk=2, cache=False)
    expected = pd.DataFrame(EXPECTED_ROWS)
    pd.testing.assert_frame_equal(table, expected)
    path = label_table_path(build_root, POLICY)
    assert not osp.exists(path)

    cached = build_label_table(POLICY, build_root, workers=1, chunk=2)
    pd.testing.assert_frame_equal(cached, expected)
    assert osp.exists(path)
    reread = build_label_table(POLICY, build_root, workers=1, chunk=2)
    pd.testing.assert_frame_equal(reread, pd.read_parquet(path))
    assert reread["index"].tolist() == [0, 1, 2]
    assert reread["fitness_source"].tolist() == ["costanzo2016@30", "kuzmin2018", None]


def test_build_label_table_subset_run_writes_the_cache_a_full_run_then_trusts(
    build_root: str,
) -> None:
    """Finding: the docstring says a cached table is reused only when it covered the
    whole build, but ``label_table.py:231`` tests ``indices is not None`` AFTER the
    full-build branch has filled ``indices`` in, so a subset run with ``cache=True``
    writes its partial table to the policy path and the next full run reads it back.
    """
    subset = build_label_table(POLICY, build_root, indices=[1], workers=1)
    pd.testing.assert_frame_equal(subset, pd.DataFrame(EXPECTED_ROWS[1:2]))
    assert osp.exists(label_table_path(build_root, POLICY))
    full = build_label_table(POLICY, build_root, workers=1)
    assert full["index"].tolist() == [1]
