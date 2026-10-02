# tests/torchcell/paper/test_ontology_usage.py
# [[tests.torchcell.paper.test_ontology_usage]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/paper/test_ontology_usage.py
"""``torchcell.paper.ontology_usage`` on a hand-built repository and on the committed one.

The hand-built repository holds two release snapshots (the newer one must win), three
loader modules (one registered class each, one unregistered duplicate in a legacy
module, one helper class), and a registry with a documented supported query, an
undocumented one and a deprecated documented one, so each rule is pinned by an exact
value: which release is read, which module a dataset's API link names, which datasets
get a page link, and which query's page wins.

The checks against the committed repository pin the two links the explorer shows for
``SmfCostanzo2016Dataset`` (both were fetched from the live documentation site on
2026-10-02 and returned 200) and that every schema class named in a closure is either
a class of the ontology graph or the ``ModelStrict`` configuration mixin.
"""

import json
from pathlib import Path

import pytest

from torchcell.paper.ontology_graph import build_ontology_graph
from torchcell.paper.ontology_usage import (
    DOCS_URL,
    build_ontology_usage,
    dataset_pages,
    loader_modules,
)

REPO = Path(__file__).resolve().parents[3]
DOCS = "https://docs.example/"
SHA = "0" * 64


def _snapshot(release: str, built_at: str, datasets: dict[str, int]) -> str:
    return json.dumps(
        {
            "release": release,
            "version": "9.9",
            "torchcell_commit": "abc1234",
            "torchcell_version": "9.9.0",
            "torchcell_tag": None,
            "built_at": built_at,
            "neo4j_version": "5",
            "biocypher_version": "0",
            "store_host": "test",
            "n_nodes": None,
            "datasets": {
                name: {
                    "dataset_class": name,
                    "n_experiments": n,
                    "content_sha256": SHA,
                    "import_mode": "full",
                    "admitted_at": built_at,
                }
                for name, n in datasets.items()
            },
            "graph_schema": {},
            "events": [],
            "composite_sha256": SHA,
        }
    )


def _query(
    query_id: str, title: str, status: str, docs_page: str | None
) -> dict[str, object]:
    return {
        "id": query_id,
        "title": title,
        "cql_path": f"queries/{query_id}.cql",
        "converter": None,
        "phenotype_classes": [],
        "status": status,
        "since_kg_version": "9.9",
        "deprecated_in": "9.9" if status == "deprecated" else None,
        "validated_release": None,
        "dataset_composite": None,
        "docs_page": docs_page,
    }


def _cql(*datasets: str) -> str:
    return "\nUNION ALL\n".join(
        "MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)\n"
        f"WHERE dataset.id = '{name}'\nRETURN e.serialized_data AS experiment"
        for name in datasets
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    releases = tmp_path / "database" / "releases"
    releases.mkdir(parents=True)
    (releases / "2026.01.01-old.json").write_text(
        _snapshot("2026.01.01-old", "2026-01-01T00:00:00+00:00", {"AlphaDataset": 1})
    )
    (releases / "2026.01.01-old.closures.json").write_text(
        json.dumps({"AlphaDataset": {"Stale": SHA}})
    )
    (releases / "2026.02.01-new.json").write_text(
        _snapshot(
            "2026.02.01-new",
            "2026-02-01T00:00:00+00:00",
            {"BetaDataset": 20, "AlphaDataset": 10, "GammaDataset": 30},
        )
    )
    (releases / "2026.02.01-new.closures.json").write_text(
        json.dumps(
            {
                "AlphaDataset": {"Genotype": SHA, "Experiment": SHA, "Fitness": SHA},
                "BetaDataset": {"Genotype": SHA, "Experiment": SHA},
                "GammaDataset": {"Experiment": SHA, "Morphology": SHA},
            }
        )
    )

    loaders = tmp_path / "torchcell" / "datasets" / "scerevisiae"
    loaders.mkdir(parents=True)
    (loaders / "alpha2020.py").write_text(
        "class Helper:\n    pass\n\n\n@register_dataset\nclass AlphaDataset:\n    pass\n"
    )
    (loaders / "beta_gamma2021.py").write_text(
        "@register_dataset\nclass BetaDataset:\n    pass\n\n\n"
        "@register_dataset\nclass GammaDataset:\n    pass\n"
    )
    (loaders / "alpha2020_deprecated.py").write_text("class AlphaDataset:\n    pass\n")

    kg = tmp_path / "torchcell" / "knowledge_graphs"
    (kg / "queries").mkdir(parents=True)
    (kg / "supported_queries").mkdir()
    (kg / "queries" / "a_documented.cql").write_text(
        _cql("AlphaDataset", "BetaDataset")
    )
    (kg / "queries" / "b_also_documented.cql").write_text(_cql("BetaDataset"))
    (kg / "queries" / "c_undocumented.cql").write_text(_cql("GammaDataset"))
    (kg / "queries" / "d_deprecated.cql").write_text(_cql("GammaDataset"))
    (kg / "supported_queries" / "registry.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "queries": [
                    _query(
                        "a_documented",
                        "Alpha and beta",
                        "supported",
                        "docs/source/datasets/scerevisiae/alpha-beta.md",
                    ),
                    _query(
                        "b_also_documented",
                        "Beta alone",
                        "supported",
                        "docs/source/datasets/scerevisiae/beta.md",
                    ),
                    _query("c_undocumented", "Gamma", "supported", None),
                    _query(
                        "d_deprecated",
                        "Old gamma",
                        "deprecated",
                        "docs/source/datasets/scerevisiae/gamma-old.md",
                    ),
                ],
            }
        )
    )
    return tmp_path


def test_loader_modules_keep_registered_classes_only(repo: Path) -> None:
    assert loader_modules(repo) == {
        "AlphaDataset": "alpha2020",
        "BetaDataset": "beta_gamma2021",
        "GammaDataset": "beta_gamma2021",
    }


def test_loader_modules_refuse_a_dataset_registered_twice(repo: Path) -> None:
    (repo / "torchcell/datasets/scerevisiae/zeta.py").write_text(
        "@register_dataset\nclass AlphaDataset:\n    pass\n"
    )
    with pytest.raises(
        ValueError, match="AlphaDataset is registered in alpha2020.py and zeta.py"
    ):
        loader_modules(repo)


def test_dataset_pages_come_from_supported_documented_queries(repo: Path) -> None:
    assert dataset_pages(repo, DOCS) == {
        "AlphaDataset": (
            "https://docs.example/datasets/scerevisiae/alpha-beta.html",
            "Alpha and beta",
        ),
        # BetaDataset is on two pages; the first query in registry order wins.
        "BetaDataset": (
            "https://docs.example/datasets/scerevisiae/alpha-beta.html",
            "Alpha and beta",
        ),
    }


def test_dataset_pages_refuse_a_page_outside_the_docs_source(repo: Path) -> None:
    path = repo / "torchcell/knowledge_graphs/supported_queries/registry.json"
    registry = json.loads(path.read_text())
    registry["queries"][0]["docs_page"] = "notes/alpha-beta.md"
    path.write_text(json.dumps(registry))
    with pytest.raises(
        ValueError, match="a_documented: docs_page is not under docs/source/"
    ):
        dataset_pages(repo, DOCS)


def test_usage_reads_the_newest_release(repo: Path) -> None:
    usage = build_ontology_usage(repo, DOCS)
    assert usage.release == "2026.02.01-new"
    assert usage.kg_version == "9.9"
    assert [d.name for d in usage.datasets] == [
        "AlphaDataset",
        "BetaDataset",
        "GammaDataset",
    ]
    alpha, beta, gamma = usage.datasets
    assert alpha.model_dump() == {
        "name": "AlphaDataset",
        "n_experiments": 10,
        "classes": ["Experiment", "Fitness", "Genotype"],
        "api_url": "https://docs.example/generated/"
        "torchcell.datasets.scerevisiae.alpha2020.AlphaDataset.html",
        "page_url": "https://docs.example/datasets/scerevisiae/alpha-beta.html",
        "page_title": "Alpha and beta",
    }
    assert beta.n_experiments == 20
    assert gamma.api_url.endswith("scerevisiae.beta_gamma2021.GammaDataset.html")
    # Its only documented query is deprecated and its supported query has no page.
    assert (gamma.page_url, gamma.page_title) == (None, None)


def test_datasets_using_inverts_the_closures(repo: Path) -> None:
    usage = build_ontology_usage(repo, DOCS)
    assert usage.datasets_using("Experiment") == [
        "AlphaDataset",
        "BetaDataset",
        "GammaDataset",
    ]
    assert usage.datasets_using("Genotype") == ["AlphaDataset", "BetaDataset"]
    assert usage.datasets_using("Morphology") == ["GammaDataset"]
    assert usage.datasets_using("Stale") == []


def test_a_served_dataset_without_a_registered_loader_fails(repo: Path) -> None:
    (repo / "torchcell/datasets/scerevisiae/alpha2020.py").write_text("")
    with pytest.raises(KeyError, match="AlphaDataset"):
        build_ontology_usage(repo, DOCS)


def test_committed_repository_links() -> None:
    usage = build_ontology_usage(REPO)
    by_name = {d.name: d for d in usage.datasets}
    smf = by_name["SmfCostanzo2016Dataset"]
    assert smf.api_url == (
        "https://mjvolk3.github.io/torchcell/generated/"
        "torchcell.datasets.scerevisiae.costanzo2016.SmfCostanzo2016Dataset.html"
    )
    assert smf.page_url == (
        "https://mjvolk3.github.io/torchcell/datasets/scerevisiae/essentiality-smf.html"
    )
    assert smf.page_title == "Gene essentiality and single-mutant fitness"
    assert "FitnessPhenotype" in smf.classes
    assert all(d.api_url.startswith(DOCS_URL + "generated/") for d in usage.datasets)
    assert len({d.api_url for d in usage.datasets}) == len(usage.datasets)


def test_committed_closures_name_ontology_classes() -> None:
    usage = build_ontology_usage(REPO)
    graph_classes = set(build_ontology_graph().classes)
    named = {name for d in usage.datasets for name in d.classes}
    assert named - graph_classes == {"ModelStrict"}
