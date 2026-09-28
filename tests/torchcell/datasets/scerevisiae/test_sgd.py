# tests/torchcell/datasets/scerevisiae/test_sgd.py
# [[tests.torchcell.datasets.scerevisiae.test_sgd]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_sgd.py
"""Hermetic build of the SGD gene-essentiality loader on a stub graph and a fake Entrez.

The loader has no raw files; ``process()`` walks ``scerevisiae_graph.G_raw`` and fetches
each phenotype's publication through ``Bio.Entrez``. Here ``G_raw`` is a three-node
networkx graph and ``Entrez.efetch``/``Entrez.read`` are replaced on ``Bio.Entrez`` itself
(the module object the loader holds), so ``get_publication_info`` runs for real against
canned records; ``time.sleep`` and ``random.uniform`` are patched the same way. ``DATA_ROOT``
points into ``tmp_path``; a ``data/sgd/genome/genes/`` directory with 100 ``.json`` files
satisfies the "genes already downloaded" check, so ``main_get_all_genes`` is not called.

Graph:
    YAL001C  four phenotype_details: inviable/null/S288C (PubMed 111, locus TFC3) -> record;
             viable/null/S288C, inviable/reduction of function/S288C, inviable/null/W303
             -> skipped
    YBR001C  no phenotype_details -> nothing
    YCR001W  two inviable/null/S288C entries (PubMed 222 and 333) -> two records

Canned Entrez: 111 has the DOI ``10.1/aaa`` in ELocationID; 222 has a ``pii`` there and the
DOI ``10.2/bbb`` in ArticleIdList; 333 has no DOI, so ``doi``/``doi_url`` are None.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, cast

import networkx as nx
import pytest
from Bio import Entrez

from torchcell.datamodels.schema import (
    Environment,
    GeneEssentialityExperiment,
    GeneEssentialityExperimentReference,
    GeneEssentialityPhenotype,
    Genotype,
    Media,
    Publication,
    ReferenceGenome,
    SgaKanMxDeletionPerturbation,
    Temperature,
)
from torchcell.datasets.scerevisiae import sgd as m
from torchcell.graph import SCerevisiaeGraph


class _Elem(str):
    """A string carrying Entrez-style ``.attributes`` (as ``Bio.Entrez`` StringElement)."""

    attributes: dict[str, str]

    def __new__(cls, value: str, attributes: dict[str, str]) -> _Elem:
        obj = super().__new__(cls, value)
        obj.attributes = attributes
        return obj


def _article(elocation: list[_Elem], article_ids: list[_Elem]) -> dict[str, Any]:
    return {
        "PubmedArticle": [
            {
                "MedlineCitation": {"Article": {"ELocationID": elocation}},
                "PubmedData": {"ArticleIdList": article_ids},
            }
        ]
    }


_RECORDS: dict[str, dict[str, Any]] = {
    "111": _article([_Elem("10.1/aaa", {"EIdType": "doi"})], []),
    "222": _article(
        [_Elem("S0000-1", {"EIdType": "pii"})],
        [_Elem("22222", {"IdType": "pubmed"}), _Elem("10.2/bbb", {"IdType": "doi"})],
    ),
    "333": _article([], [_Elem("33333", {"IdType": "pubmed"})]),
}


class _Handle:
    def __init__(self, pubmed_id: str) -> None:
        self.pubmed_id = pubmed_id

    def close(self) -> None:
        pass


def _install_entrez(monkeypatch: pytest.MonkeyPatch, fail_first: int = 0) -> list[str]:
    """Patch efetch/read on the module Entrez; the first ``fail_first`` calls raise."""
    fetched: list[str] = []
    state = {"failures": fail_first}

    def efetch(**kwargs: Any) -> _Handle:
        assert kwargs == {
            "db": "pubmed",
            "id": kwargs["id"],
            "rettype": "xml",
            "retmode": "text",
        }
        if state["failures"] > 0:
            state["failures"] -= 1
            raise RuntimeError("boom")
        fetched.append(kwargs["id"])
        return _Handle(kwargs["id"])

    def read(handle: _Handle) -> dict[str, Any]:
        return _RECORDS[handle.pubmed_id]

    monkeypatch.setattr(Entrez, "efetch", efetch)
    monkeypatch.setattr(Entrez, "read", read)
    return fetched


def _phenotype(pubmed_id: str, locus: str) -> dict[str, Any]:
    return {
        "mutant_type": "null",
        "strain": {"display_name": "S288C"},
        "phenotype": {"display_name": "inviable"},
        "reference": {"pubmed_id": int(pubmed_id)},
        "locus": {"display_name": locus},
    }


def _graph() -> SCerevisiaeGraph:
    g = nx.Graph()
    g.add_node(
        "YAL001C",
        phenotype_details=[
            _phenotype("111", "TFC3"),
            {**_phenotype("111", "TFC3"), "phenotype": {"display_name": "viable"}},
            {**_phenotype("111", "TFC3"), "mutant_type": "reduction of function"},
            {**_phenotype("111", "TFC3"), "strain": {"display_name": "W303"}},
        ],
    )
    g.add_node("YBR001C")
    g.add_node(
        "YCR001W",
        phenotype_details=[_phenotype("222", "YCR001W"), _phenotype("333", "YCR001W")],
    )

    class _StubGraph:
        G_raw = g

    return cast(SCerevisiaeGraph, _StubGraph())


def _genes_dir(data_root: Path, n_files: int) -> None:
    genes = data_root / "data" / "sgd" / "genome" / "genes"
    genes.mkdir(parents=True)
    for k in range(n_files):
        (genes / f"gene{k}.json").write_text("{}")


@pytest.fixture
def dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> m.GeneEssentialitySgdDataset:
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    _genes_dir(data_root, 100)
    _install_entrez(monkeypatch)

    def _no_gene_download() -> None:
        raise AssertionError("main_get_all_genes called with 100 gene files present")

    monkeypatch.setattr(m, "main_get_all_genes", _no_gene_download)
    return m.GeneEssentialitySgdDataset(
        root=str(tmp_path / "gene_essentiality_sgd"), scerevisiae_graph=_graph()
    )


_ENVIRONMENT = Environment(
    media=Media(name="YEPD", state="solid", is_synthetic=False),
    temperature=Temperature(value=30),
)
_REFERENCE = GeneEssentialityExperimentReference(
    dataset_name="GeneEssentialitySgdDataset",
    genome_reference=ReferenceGenome(
        species="Saccharomyces cerevisiae", strain="S288C"
    ),
    environment_reference=_ENVIRONMENT,
    phenotype_reference=GeneEssentialityPhenotype(is_essential=False),
).model_dump()


def _experiment(gene: str, locus: str) -> dict[str, Any]:
    return GeneEssentialityExperiment(
        dataset_name="GeneEssentialitySgdDataset",
        genotype=Genotype(
            perturbations=[
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=gene,
                    perturbed_gene_name=locus,
                    strain_id="S288C",
                )
            ]
        ),
        environment=_ENVIRONMENT,
        phenotype=GeneEssentialityPhenotype(is_essential=True),
    ).model_dump()


def test_one_record_per_inviable_null_s288c_phenotype(
    dataset: m.GeneEssentialitySgdDataset,
) -> None:
    """Three records: YAL001C once (three of its four phenotype entries fail a filter) and
    YCR001W twice, once per inviable entry, each with its own publication. The DOI is
    taken from ELocationID first, then ArticleIdList, else left None.
    """
    assert len(dataset) == 3
    assert dataset[0]["experiment"] == _experiment("YAL001C", "TFC3")
    assert dataset[1]["experiment"] == _experiment("YCR001W", "YCR001W")
    assert dataset[2]["experiment"] == _experiment("YCR001W", "YCR001W")
    assert dataset[0]["reference"] == _REFERENCE
    assert [dataset[i]["publication"] for i in range(3)] == [
        Publication(
            pubmed_id="111",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/111/",
            doi="10.1/aaa",
            doi_url="https://doi.org/10.1/aaa",
        ).model_dump(),
        Publication(
            pubmed_id="222",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/222/",
            doi="10.2/bbb",
            doi_url="https://doi.org/10.2/bbb",
        ).model_dump(),
        Publication(
            pubmed_id="333",
            pubmed_url="https://pubmed.ncbi.nlm.nih.gov/333/",
            doi=None,
            doi_url=None,
        ).model_dump(),
    ]


def test_side_files(dataset: m.GeneEssentialitySgdDataset) -> None:
    """No raw files, no ``data.csv``; the gene set is the two genes with records and one
    reference covers all three.
    """
    assert dataset.raw_file_names == []
    assert dataset.experiment_class is GeneEssentialityExperiment
    assert dataset.reference_class is GeneEssentialityExperimentReference
    preprocess = Path(dataset.root) / "preprocess"
    assert not (preprocess / "data.csv").exists()
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YCR001W",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [[0, 1, 2]]
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    assert manifest["dataset_name"] == "gene_essentiality_sgd"
    assert manifest["loader_class"] == "GeneEssentialitySgdDataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.sgd"
    assert {"GeneEssentialityExperiment", "SgaKanMxDeletionPerturbation"} <= set(
        manifest["closure"]
    )


def test_gene_download_is_triggered_below_100_gene_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With 99 gene JSON files under ``$DATA_ROOT/data/sgd/genome/genes`` the loader calls
    ``main_get_all_genes`` exactly once before building; the build itself is unchanged.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    _genes_dir(data_root, 99)
    _install_entrez(monkeypatch)
    calls: list[None] = []
    monkeypatch.setattr(m, "main_get_all_genes", lambda: calls.append(None))
    dataset = m.GeneEssentialitySgdDataset(
        root=str(tmp_path / "gene_essentiality_sgd"), scerevisiae_graph=_graph()
    )
    assert calls == [None]
    assert len(dataset) == 3


def test_get_publication_info_retries_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two failing fetches then success: sleeps 1 s and 2 s (``base_delay * 2**attempt``
    with the random jitter pinned to 0), prints one retry line per failure, and returns
    the parsed record.
    """
    fetched = _install_entrez(monkeypatch, fail_first=2)
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    monkeypatch.setattr(random, "uniform", lambda a, b: 0.0)
    assert m.get_publication_info("111") == {
        "pubmed_id": "111",
        "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/111/",
        "doi": "10.1/aaa",
        "doi_url": "https://doi.org/10.1/aaa",
    }
    assert sleeps == [1, 2]
    assert fetched == ["111"]
    assert capsys.readouterr().out == (
        "Attempt 1 failed. Retrying in 1.00 seconds...\n"
        "Attempt 2 failed. Retrying in 2.00 seconds...\n"
    )


def test_get_publication_info_gives_up_after_five_attempts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Five failures: four sleeps (1, 2, 4, 8 s), a final error line, and None."""
    _install_entrez(monkeypatch, fail_first=5)
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    monkeypatch.setattr(random, "uniform", lambda a, b: 0.0)
    assert m.get_publication_info("999") is None
    assert sleeps == [1, 2, 4, 8]
    assert capsys.readouterr().out.splitlines()[-1] == (
        "Error fetching info for PubMed ID 999: boom"
    )


def test_create_experiment_raises_when_publication_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(m, "get_publication_info", lambda pubmed_id: None)
    with pytest.raises(
        ValueError,
        match="Unable to retrieve publication information for PubMed ID: 999",
    ):
        m.GeneEssentialitySgdDataset.create_experiment(
            "GeneEssentialitySgdDataset", "YAL001C", _phenotype("999", "TFC3")
        )
