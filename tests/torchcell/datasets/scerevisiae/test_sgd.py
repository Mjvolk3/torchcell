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

2026.09.30 (Phase 15): the same three-node graph and canned Entrez, plus

- the gene-file threshold: ``process`` counts only ``*.json`` names under
  ``$DATA_ROOT/data/sgd/genome/genes`` (line 168) and fetches below 100 (line 173), so
  a missing directory counts 0 and 99 ``.json`` plus one ``.txt`` still counts 99; both
  call the (faked) ``main_get_all_genes`` once, 100 ``.json`` calls it zero times. With
  ``DATA_ROOT`` unset the directory falls back to the relative ``data/data/sgd/genome/
  genes`` under the working directory (line 165).
- the content-hash collapse the essentiality showcase page reports (1,329 stored records,
  1,140 graph nodes): the node id is ``sha256(json.dumps(experiment.model_dump()))``
  (``cell_adapter.py`` line 527), the experiment excludes the publication, so the two
  YCR001W records (PubMed 222 and 333) hash to one id: 3 records, 2 ids.
- the assumed environment the showcase page documents (``# HACK ... all meta data is
  guessed``, lines 215 to 242): YEPD, solid, not synthetic, 30 C, already pinned on every
  record by the full-dump equality of the first test, with the reference phenotype
  ``is_essential=False`` pinned again through ``transform_item``.
- ``main`` with the genome, the graph and the dataset replaced by recorders.

Findings pinned here: ``main`` builds the genome with ``overwrite=True`` (line 294), the
setting that races a concurrent rebuild, and constructs the dataset at the class default
``root="data/torchcell/gene_essentiality_sgd"``, relative to the working directory rather
than under ``$DATA_ROOT`` (line 303). The gene-directory check falls back to the relative
path ``"data"`` when ``DATA_ROOT`` is unset instead of refusing (line 165).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, cast

import networkx as nx
import pandas as pd
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


def test_stored_items_retype_through_the_essentiality_classes(
    dataset: m.GeneEssentialitySgdDataset,
) -> None:
    """``transform_item`` rebuilds each stored item through the declared classes (lines
    129 to 136): a ``GeneEssentialityExperiment`` whose phenotype is essential, a
    ``GeneEssentialityExperimentReference`` whose phenotype is not, and a ``Publication``,
    each dumping back to exactly the stored dictionary. A class wired to the wrong schema
    (for example a fitness experiment) would fail on the first record.
    """
    for index in range(3):
        item = dataset[index]
        typed = dataset.transform_item(item)
        assert type(typed["experiment"]) is GeneEssentialityExperiment
        assert type(typed["reference"]) is GeneEssentialityExperimentReference
        assert type(typed["publication"]) is Publication
        assert typed["experiment"].model_dump() == item["experiment"]
        assert typed["reference"].model_dump() == item["reference"]
        assert typed["publication"].model_dump() == item["publication"]
        assert typed["experiment"].phenotype.is_essential is True
        assert typed["reference"].phenotype_reference.is_essential is False


def test_publication_is_outside_the_content_hash_so_repeat_annotations_collapse(
    dataset: m.GeneEssentialitySgdDataset,
) -> None:
    """The KG node id of an experiment is the sha256 of its JSON dump
    (``cell_adapter.py`` line 527), and the stored experiment carries no publication. The
    two YCR001W records differ only in their publication (PubMed 222 and 333), so the
    three records map to two node ids: the loader-level cause of the 1,329 stored records
    becoming 1,140 graph nodes.
    """
    ids = [
        hashlib.sha256(
            json.dumps(
                GeneEssentialityExperiment(**dataset[i]["experiment"]).model_dump()
            ).encode("utf-8")
        ).hexdigest()
        for i in range(3)
    ]
    assert ids[1] == ids[2]
    assert ids[0] != ids[1]
    assert len(set(ids)) == 2
    assert dataset[1]["publication"]["pubmed_id"] == "222"
    assert dataset[2]["publication"]["pubmed_id"] == "333"


@pytest.mark.parametrize(
    ("n_json", "n_other", "make_dir", "n_fetches", "message"),
    [
        (0, 0, False, 1, "SGD gene data incomplete or missing (found 0 files)."),
        (99, 1, True, 1, "SGD gene data incomplete or missing (found 99 files)."),
        (100, 0, True, 0, "SGD gene data already exists (100 gene files found)"),
    ],
)
def test_gene_file_threshold_counts_json_files_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    n_json: int,
    n_other: int,
    make_dir: bool,
    n_fetches: int,
    message: str,
) -> None:
    """The fetch runs when fewer than 100 ``*.json`` files sit under
    ``$DATA_ROOT/data/sgd/genome/genes``: a missing directory counts 0, a ``.txt`` beside
    99 ``.json`` files is not counted (99 < 100, fetch), and exactly 100 ``.json`` files
    skip it. The logged count is the ``.json`` count; the build is three records each time.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    if make_dir:
        _genes_dir(data_root, n_json)
        for k in range(n_other):
            (
                data_root / "data" / "sgd" / "genome" / "genes" / f"note{k}.txt"
            ).write_text("")
    _install_entrez(monkeypatch)
    calls: list[None] = []
    monkeypatch.setattr(m, "main_get_all_genes", lambda: calls.append(None))
    with caplog.at_level(logging.INFO, logger=m.log.name):
        dataset = m.GeneEssentialitySgdDataset(
            root=str(tmp_path / "gene_essentiality_sgd"), scerevisiae_graph=_graph()
        )
    assert len(calls) == n_fetches
    assert message in [r.getMessage() for r in caplog.records if r.name == m.log.name]
    assert len(dataset) == 3


def test_unset_data_root_reads_genes_under_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: with ``DATA_ROOT`` unset the gene directory is the RELATIVE path
    ``data/data/sgd/genome/genes`` (``os.environ.get("DATA_ROOT", "data")``, line 165), so
    100 ``.json`` files under ``<cwd>/data/data/sgd/genome/genes`` satisfy the check and
    no fetch runs. Pinned until the check refuses an unset ``DATA_ROOT``.
    """
    monkeypatch.delenv("DATA_ROOT")
    monkeypatch.chdir(tmp_path)
    _genes_dir(tmp_path / "data", 100)
    _install_entrez(monkeypatch)
    calls: list[None] = []
    monkeypatch.setattr(m, "main_get_all_genes", lambda: calls.append(None))
    dataset = m.GeneEssentialitySgdDataset(
        root=str(tmp_path / "gene_essentiality_sgd"), scerevisiae_graph=_graph()
    )
    assert calls == []
    assert len(dataset) == 3


def test_no_raw_stage(dataset: m.GeneEssentialitySgdDataset) -> None:
    """The loader has no raw files: ``download`` writes nothing into ``raw/`` and
    ``preprocess_raw`` hands back the very frame it was given, unchanged.
    """
    raw = Path(dataset.raw_dir)
    before = sorted(p.name for p in raw.iterdir()) if raw.exists() else []
    dataset.download()
    after = sorted(p.name for p in raw.iterdir()) if raw.exists() else []
    assert after == before
    frame = pd.DataFrame({"gene": ["YAL001C"], "is_essential": [True]})
    returned = dataset.preprocess_raw(frame)
    assert returned is frame
    assert returned.to_dict("list") == {"gene": ["YAL001C"], "is_essential": [True]}


def test_main_wires_genome_graph_and_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` builds the genome from ``$DATA_ROOT/data/sgd/genome`` and ``data/go``, the
    graph from the SGD, STRING and TFLink roots with that genome, and the dataset with that
    graph, then prints the dataset. The three classes are recorders, so nothing is built.

    Finding: the genome is built with ``overwrite=True`` (line 294) and the dataset at its
    default relative ``root`` (line 303, no ``root`` argument). Pinned until ``main`` uses
    ``overwrite=False`` and a root under ``$DATA_ROOT``.
    """
    import torchcell.graph as graph_module
    import torchcell.sequence.genome.scerevisiae.s288c as s288c_module

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    calls: list[tuple[str, dict[str, Any]]] = []

    class _Genome:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("genome", kwargs))

    class _Graph:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("graph", kwargs))

    class _Dataset:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("dataset", kwargs))

        def __repr__(self) -> str:
            return "GeneEssentialitySgdDataset(3)"

    monkeypatch.setattr(s288c_module, "SCerevisiaeGenome", _Genome)
    monkeypatch.setattr(graph_module, "SCerevisiaeGraph", _Graph)
    monkeypatch.setattr(m, "GeneEssentialitySgdDataset", _Dataset)
    m.main()
    assert [name for name, _ in calls] == ["genome", "graph", "dataset"]
    assert calls[0][1] == {
        "genome_root": f"{tmp_path}/data/sgd/genome",
        "go_root": f"{tmp_path}/data/go",
        "overwrite": True,
    }
    graph_kwargs = calls[1][1]
    assert set(graph_kwargs) == {"sgd_root", "string_root", "tflink_root", "genome"}
    assert graph_kwargs["sgd_root"] == f"{tmp_path}/data/sgd/genome"
    assert graph_kwargs["string_root"] == f"{tmp_path}/data/string"
    assert graph_kwargs["tflink_root"] == f"{tmp_path}/data/tflink"
    assert type(graph_kwargs["genome"]) is _Genome
    assert list(calls[2][1]) == ["scerevisiae_graph"]
    assert type(calls[2][1]["scerevisiae_graph"]) is _Graph
    assert capsys.readouterr().out == "GeneEssentialitySgdDataset(3)\n"
