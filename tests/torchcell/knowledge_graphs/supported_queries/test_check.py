# tests/torchcell/knowledge_graphs/supported_queries/test_check.py
# [[tests.torchcell.knowledge_graphs.supported_queries.test_check]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/knowledge_graphs/supported_queries/test_check.py
"""The supported-query drift check against a synthetic release, one test per drift kind,
plus ``validate``, the CLI, and the real registry against the committed snapshot.

The synthetic checkout (``_repo``) holds a two-class schema surface (``Phenotype`` and
``GrowthPhenotype(Phenotype)``), a BioCypher config in which ``growth phenotype`` and
``colony phenotype`` are ``is_a: phenotypic feature``, a converter module, and one query::

    MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
    WHERE dataset.id = 'DsA'
    MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
    WHERE phen.graph_level = 'global'
    RETURN e.serialized_data AS e_serialized

The release ``2026.09.21-aaaaaaaa`` serves ``DsA`` (content hash ``"a" * 64``) and ``DsB``
(``"b" * 64``); both closures record the checkout's fingerprints of ``GrowthPhenotype``
and ``Phenotype``. The query's composite is sha256 of ``"a" * 64`` plus a newline =
``44c2336f...``. Each drift test changes one input and asserts the exact drift list, with
exit code 1 for a ``supported`` query and 0 for a ``deprecated`` one.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from torchcell.knowledge_graphs.kg_manifest import GraphSchemaEntry
from torchcell.knowledge_graphs.release_snapshot import (
    KgReleaseSnapshot,
    SnapshotDataset,
    load_snapshot,
    snapshot_paths,
    write_snapshot,
)
from torchcell.knowledge_graphs.supported_queries.__main__ import main
from torchcell.knowledge_graphs.supported_queries.check import (
    GraphLabels,
    QueryDrift,
    check_repo,
    graph_labels,
    pascal_label,
    run_check,
    schema_is_a,
    selected_composite,
    validate_query,
)
from torchcell.knowledge_graphs.supported_queries.registry import (
    QueryRegistry,
    SupportedQuery,
    registry_path,
)
from torchcell.provenance.schema_deps import SchemaSurface, load_surface_from_sources

REPO = Path(__file__).resolve().parents[4]
RELEASE = "2026.09.21-aaaaaaaa"
COMPOSITE_A = "44c2336fedab8ff6a85c74c2b94165377b0981f526adb9487895ca6314165e86"
COMPOSITE_C = "4635042acefc14343e21753cde7f1465c323970e205d422248232ff8e2a0fad2"
CONVERTER = "torchcell.datamodels.conv.GrowthConverter"

SCHEMA = """from pydantic import BaseModel


class Phenotype(BaseModel):
    graph_level: str


class GrowthPhenotype(Phenotype):
    fitness: float
"""
PYDANT = "class ModelStrict:\n    pass\n"
CONFIG = """dataset:
    represented_as: node
experiment:
    represented_as: node
    is_a: information content entity
growth phenotype:
    is_a: phenotypic feature
    represented_as: node
colony phenotype:
    is_a: [phenotypic feature]
    represented_as: node
experiment member of:
    is_a: part of
    represented_as: edge
phenotype member of:
    is_a: participates in
    represented_as: edge
"""
QUERY = """// synthetic query
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'DsA'
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
WHERE phen.graph_level = 'global'
RETURN e.serialized_data AS e_serialized
"""


def _surface(schema: str = SCHEMA) -> SchemaSurface:
    return load_surface_from_sources(
        {
            "torchcell/datamodels/schema.py": schema,
            "torchcell/datamodels/pydant.py": PYDANT,
        }
    )


def _graph_schema(colony_props: list[str]) -> dict[str, GraphSchemaEntry]:
    return {
        "dataset": GraphSchemaEntry(kind="node"),
        "experiment": GraphSchemaEntry(kind="node", properties=["serialized_data"]),
        "growth phenotype": GraphSchemaEntry(
            kind="node", properties=["fitness", "graph_level", "serialized_data"]
        ),
        "colony phenotype": GraphSchemaEntry(kind="node", properties=colony_props),
        "experiment member of": GraphSchemaEntry(
            kind="edge", source=["experiment"], target=["dataset"]
        ),
        "phenotype member of": GraphSchemaEntry(
            kind="edge", source=["growth phenotype"], target=["experiment"]
        ),
    }


def _snapshot(
    hash_a: str = "a" * 64, colony_props: list[str] | None = None
) -> KgReleaseSnapshot:
    def dataset(name: str, digest: str) -> SnapshotDataset:
        return SnapshotDataset(
            dataset_class=name,
            n_experiments=10,
            content_sha256=digest,
            import_mode="full",
            admitted_at="2026-09-21T20:25:23+00:00",
        )

    return KgReleaseSnapshot(
        release=RELEASE,
        version="1.2",
        torchcell_commit="a" * 40,
        torchcell_version="1.2.0",
        torchcell_tag=None,
        built_at="2026-09-21T20:25:23+00:00",
        neo4j_version="5.26.28",
        biocypher_version="0.15.2",
        store_host="gilahyper",
        n_nodes=100,
        datasets={"DsA": dataset("DsA", hash_a), "DsB": dataset("DsB", "b" * 64)},
        graph_schema=_graph_schema(
            ["graph_level", "serialized_data"] if colony_props is None else colony_props
        ),
        events=[],
        composite_sha256="0" * 64,
    )


def _closures() -> dict[str, dict[str, str]]:
    surface = _surface()
    closure = {
        name: surface.fingerprints[name] for name in ("GrowthPhenotype", "Phenotype")
    }
    return {"DsA": dict(closure), "DsB": dict(closure)}


def _query(status: str = "supported", **overrides: Any) -> SupportedQuery:
    fields: dict[str, Any] = {
        "id": "growth",
        "title": "Growth",
        "cql_path": "queries/growth.cql",
        "converter": CONVERTER,
        "phenotype_classes": ["GrowthPhenotype"],
        "status": status,
        "since_kg_version": "1.2",
        "deprecated_in": "1.3" if status == "deprecated" else None,
        "validated_release": RELEASE,
        "dataset_composite": COMPOSITE_A,
        "docs_page": None,
    }
    fields.update(overrides)
    return SupportedQuery(**fields)


def _repo(root: Path, query_text: str | None = QUERY) -> Path:
    files = {
        "torchcell/datamodels/schema.py": SCHEMA,
        "torchcell/datamodels/pydant.py": PYDANT,
        "torchcell/datamodels/conv.py": "class GrowthConverter:\n    pass\n",
        "biocypher/config/torchcell_schema_config.yaml": CONFIG,
    }
    if query_text is not None:
        files["torchcell/knowledge_graphs/queries/growth.cql"] = query_text
    for relpath, text in files.items():
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _run(
    root: Path,
    query: SupportedQuery,
    snapshot: KgReleaseSnapshot | None = None,
    surface: SchemaSurface | None = None,
    closures: dict[str, dict[str, str]] | None = None,
) -> tuple[int, list[QueryDrift]]:
    report = run_check(
        QueryRegistry(queries=[query]),
        snapshot or _snapshot(),
        _closures() if closures is None else closures,
        surface or _surface(),
        CONFIG,
        root,
        "c" * 40,
    )
    (result,) = report.results
    return report.exit_code, result.drifts


def _drift(kind: str, detail: str) -> QueryDrift:
    return QueryDrift.model_validate(
        {"query_id": "growth", "kind": kind, "detail": detail}
    )


STATUS_EXIT = [("supported", 1), ("deprecated", 0)]


def test_no_drift_exits_zero(tmp_path: Path) -> None:
    assert _run(_repo(tmp_path), _query()) == (0, [])


@pytest.mark.parametrize(("status", "exit_code"), STATUS_EXIT)
def test_file_missing(tmp_path: Path, status: str, exit_code: int) -> None:
    root = _repo(tmp_path, query_text=None)
    assert _run(root, _query(status)) == (
        exit_code,
        [
            _drift(
                "file_missing",
                "torchcell/knowledge_graphs/queries/growth.cql is not in the checkout",
            )
        ],
    )


@pytest.mark.parametrize(("status", "exit_code"), STATUS_EXIT)
def test_missing_node_label(tmp_path: Path, status: str, exit_code: int) -> None:
    root = _repo(tmp_path, QUERY.replace("(e:Experiment)", "(e:Experiment:Assay)"))
    assert _run(root, _query(status)) == (
        exit_code,
        [_drift("missing_node_label", f"Assay is not a node label of {RELEASE}")],
    )


@pytest.mark.parametrize(("status", "exit_code"), STATUS_EXIT)
def test_missing_relationship_type(tmp_path: Path, status: str, exit_code: int) -> None:
    root = _repo(tmp_path, QUERY.replace("PhenotypeMemberOf", "PhenotypeOf"))
    assert _run(root, _query(status)) == (
        exit_code,
        [
            _drift(
                "missing_relationship_type",
                f"PhenotypeOf is not a relationship type of {RELEASE}",
            )
        ],
    )


@pytest.mark.parametrize(("status", "exit_code"), STATUS_EXIT)
def test_missing_property_on_a_concrete_label(
    tmp_path: Path, status: str, exit_code: int
) -> None:
    root = _repo(tmp_path, QUERY.replace("e.serialized_data", "e.payload"))
    assert _run(root, _query(status)) == (
        exit_code,
        [
            _drift(
                "missing_property",
                f"Experiment.payload: Experiment nodes of {RELEASE} do not all carry "
                "payload",
            )
        ],
    )


@pytest.mark.parametrize(("status", "exit_code"), STATUS_EXIT)
def test_missing_property_on_one_class_under_an_ancestor_label(
    tmp_path: Path, status: str, exit_code: int
) -> None:
    # colony phenotype loses graph_level, so not every PhenotypicFeature node carries it
    snapshot = _snapshot(colony_props=["serialized_data"])
    assert _run(_repo(tmp_path), _query(status), snapshot=snapshot) == (
        exit_code,
        [
            _drift(
                "missing_property",
                f"PhenotypicFeature.graph_level: PhenotypicFeature nodes of {RELEASE} "
                "do not all carry graph_level",
            )
        ],
    )


@pytest.mark.parametrize(("status", "exit_code"), STATUS_EXIT)
def test_missing_dataset(tmp_path: Path, status: str, exit_code: int) -> None:
    # DsA is still selected, so the composite over the served selection is unchanged
    root = _repo(
        tmp_path, QUERY.replace("dataset.id = 'DsA'", "dataset.id IN ['DsA', 'DsZ']")
    )
    assert _run(root, _query(status)) == (
        exit_code,
        [_drift("missing_dataset", f"DsZ is not served by {RELEASE}")],
    )


@pytest.mark.parametrize(("status", "exit_code"), STATUS_EXIT)
def test_contract_changed(tmp_path: Path, status: str, exit_code: int) -> None:
    moved = _surface(SCHEMA.replace("    graph_level: str\n", "    graph_level: int\n"))
    assert _run(_repo(tmp_path), _query(status), surface=moved) == (
        exit_code,
        [
            _drift(
                "contract_changed",
                "GrowthPhenotype: contract of Phenotype differs from the closure "
                f"{RELEASE} recorded for DsA",
            )
        ],
    )


@pytest.mark.parametrize(("status", "exit_code"), STATUS_EXIT)
def test_contract_changed_when_the_class_is_gone(
    tmp_path: Path, status: str, exit_code: int
) -> None:
    gone = _surface(SCHEMA.replace("GrowthPhenotype", "YieldPhenotype"))
    assert _run(_repo(tmp_path), _query(status), surface=gone) == (
        exit_code,
        [
            _drift(
                "contract_changed",
                "GrowthPhenotype: contract of GrowthPhenotype differs from the closure "
                f"{RELEASE} recorded for DsA (GrowthPhenotype absent from the "
                "checkout's schema)",
            )
        ],
    )


@pytest.mark.parametrize(("status", "exit_code"), STATUS_EXIT)
def test_contract_changed_when_no_selected_dataset_serves_the_class(
    tmp_path: Path, status: str, exit_code: int
) -> None:
    closures = {"DsA": {"Phenotype": _closures()["DsA"]["Phenotype"]}, "DsB": {}}
    assert _run(_repo(tmp_path), _query(status), closures=closures) == (
        exit_code,
        [
            _drift(
                "contract_changed",
                "GrowthPhenotype is in the recorded closure of none of the selected "
                f"datasets in {RELEASE}",
            )
        ],
    )


@pytest.mark.parametrize(("status", "exit_code"), STATUS_EXIT)
def test_composite_changed(tmp_path: Path, status: str, exit_code: int) -> None:
    snapshot = _snapshot(hash_a="c" * 64)
    assert _run(_repo(tmp_path), _query(status), snapshot=snapshot) == (
        exit_code,
        [
            _drift(
                "composite_changed",
                f"selected datasets compose to {COMPOSITE_C} in {RELEASE}; recorded "
                f"{COMPOSITE_A} (validated on {RELEASE})",
            )
        ],
    )


@pytest.mark.parametrize(("status", "exit_code"), STATUS_EXIT)
def test_converter_missing(tmp_path: Path, status: str, exit_code: int) -> None:
    root = _repo(tmp_path)
    query = _query(status, converter="torchcell.datamodels.conv.OtherConverter")
    absent = _query(status, converter="torchcell.datamodels.gone.GrowthConverter")
    assert _run(root, query) == (
        exit_code,
        [
            _drift(
                "converter_missing",
                "torchcell.datamodels.conv defines no top-level class OtherConverter",
            )
        ],
    )
    assert _run(root, absent) == (
        exit_code,
        [
            _drift(
                "converter_missing",
                "converter module torchcell.datamodels.gone is not in the checkout "
                "(gone.py absent)",
            )
        ],
    )


def test_never_validated_is_a_composite_drift(tmp_path: Path) -> None:
    query = _query(validated_release=None, dataset_composite=None)
    assert _run(_repo(tmp_path), query) == (
        1,
        [
            _drift(
                "composite_changed",
                f"selected datasets compose to {COMPOSITE_A} in {RELEASE}; never validated",
            )
        ],
    )


def test_report_names_release_commit_and_issue_title(tmp_path: Path) -> None:
    report = run_check(
        QueryRegistry(queries=[_query(converter="torchcell.datamodels.conv.Other")]),
        _snapshot(),
        _closures(),
        _surface(),
        CONFIG,
        _repo(tmp_path),
        "c" * 40,
    )
    (result,) = report.results
    assert result.issue_title == (
        f"Before the next KG build: supported query growth drifts against {RELEASE}"
    )
    assert result.issue_body == (
        f"Supported query `growth` (Growth) drifts against KG release `{RELEASE}` "
        "(version 1.2, built from `aaaaaaaa`).\n"
        "\n"
        "- Query: `torchcell/knowledge_graphs/queries/growth.cql`, status `supported`\n"
        f"- Last validated on: `{RELEASE}`\n"
        f"- Checkout commit: `{'c' * 40}`\n"
        "\n"
        "Drift:\n"
        "\n"
        "- `converter_missing`: torchcell.datamodels.conv defines no top-level class "
        "Other\n"
        "\n"
        "After the next KG build, re-validate with `python -m "
        "torchcell.knowledge_graphs.supported_queries validate growth --release <new "
        "release>` (or deprecate the query) and close this issue by hand.\n"
    )
    assert report.report_markdown.startswith(
        f"## Supported queries against `{RELEASE}` (KG 1.2)\n"
        "\n"
        f"Checkout `{'c' * 40}`: 1 queries, 1 drifted, 1 failing.\n"
        "\n"
        "- `growth` (supported): 1 drift(s)\n"
        "\n"
        f"### {result.issue_title}\n"
    )
    assert (report.release, report.release_version, report.exit_code) == (
        RELEASE,
        "1.2",
        1,
    )


def test_graph_labels_concrete_ancestor_and_edges() -> None:
    labels = graph_labels(_graph_schema(["serialized_data"]), schema_is_a(CONFIG))
    implicit = ["id", "preferred_id"]
    assert labels == GraphLabels(
        node_properties={
            "ColonyPhenotype": sorted(["serialized_data", *implicit]),
            "Dataset": implicit,
            "Experiment": sorted(["serialized_data", *implicit]),
            "GrowthPhenotype": sorted(
                ["fitness", "graph_level", "serialized_data", *implicit]
            ),
            "InformationContentEntity": sorted(["serialized_data", *implicit]),
            "PhenotypicFeature": sorted(["serialized_data", *implicit]),
        },
        relationship_types=["ExperimentMemberOf", "PhenotypeMemberOf"],
    )


def test_schema_is_a_reads_strings_and_lists() -> None:
    assert schema_is_a(CONFIG) == {
        "experiment": ["information content entity"],
        "growth phenotype": ["phenotypic feature"],
        "colony phenotype": ["phenotypic feature"],
        "experiment member of": ["part of"],
        "phenotype member of": ["participates in"],
    }
    assert pascal_label("experiment reference of") == "ExperimentReferenceOf"


def test_selected_composite_ignores_unserved_datasets() -> None:
    assert selected_composite(_snapshot(), ["DsA", "DsZ"]) == COMPOSITE_A


# --------------------------------------------------------------------------- repo + CLI


def _git_repo(root: Path, query: SupportedQuery) -> Path:
    _repo(root)
    write_snapshot(_snapshot(), _closures(), root)
    path = registry_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    QueryRegistry(queries=[query]).save(path)
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "fixture"], check=True)
    return root


def test_check_repo_reads_the_latest_snapshot(tmp_path: Path) -> None:
    root = _git_repo(tmp_path, _query())
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    report = check_repo(root, QueryRegistry.load(registry_path(root)), None)
    assert (report.release, report.checkout_commit, report.exit_code) == (
        RELEASE,
        head,
        0,
    )


def test_validate_records_release_composite_and_since(tmp_path: Path) -> None:
    unvalidated = _query(
        since_kg_version=None, validated_release=None, dataset_composite=None
    )
    root = _git_repo(tmp_path, unvalidated)
    registry = QueryRegistry.load(registry_path(root))
    updated, blocking = validate_query(registry, "growth", root, RELEASE)
    assert blocking == []
    assert updated.get("growth") == _query()


def test_validate_refuses_a_query_with_a_blocking_drift(tmp_path: Path) -> None:
    root = _git_repo(tmp_path, _query(validated_release=None, dataset_composite=None))
    (root / "torchcell/knowledge_graphs/queries/growth.cql").write_text(
        QUERY.replace("= 'DsA'", "IN ['DsA', 'DsZ']"), encoding="utf-8"
    )
    registry = QueryRegistry.load(registry_path(root))
    updated, blocking = validate_query(registry, "growth", root, RELEASE)
    assert updated == registry
    assert blocking == [_drift("missing_dataset", f"DsZ is not served by {RELEASE}")]


def test_cli_check_json_list_and_validate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _git_repo(tmp_path, _query(converter="torchcell.datamodels.conv.Other"))
    assert main(["--repo-root", str(root), "check", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert (payload["release"], payload["exit_code"]) == (RELEASE, 1)
    assert [d["kind"] for d in payload["results"][0]["drifts"]] == ["converter_missing"]

    (root / "torchcell/knowledge_graphs/queries/growth.cql").unlink()
    assert main(["--repo-root", str(root), "list"]) == 0
    assert capsys.readouterr().out == (
        f"growth  supported  since 1.2  validated {RELEASE} (44c2336fedab)  "
        "queries/growth.cql: file absent\n"
    )
    assert main(["--repo-root", str(root), "check"]) == 1
    assert "`file_missing`: torchcell/knowledge_graphs/queries/growth.cql" in (
        capsys.readouterr().out
    )


def test_cli_validate_writes_the_registry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _git_repo(tmp_path, _query(validated_release=None, dataset_composite=None))
    assert (
        main(["--repo-root", str(root), "validate", "growth", "--release", RELEASE])
        == 0
    )
    assert capsys.readouterr().out == (
        f"growth: validated on {RELEASE}, dataset_composite {COMPOSITE_A}, "
        "since_kg_version 1.2\n"
    )
    assert QueryRegistry.load(registry_path(root)).get("growth") == _query()
    (root / "torchcell/knowledge_graphs/queries/growth.cql").write_text(
        QUERY.replace("Dataset)", "Collection)"), encoding="utf-8"
    )
    assert (
        main(["--repo-root", str(root), "validate", "growth", "--release", RELEASE])
        == 1
    )
    assert capsys.readouterr().out == (
        f"growth does not hold on {RELEASE}; nothing recorded:\n"
        f"- missing_node_label: Collection is not a node label of {RELEASE}\n"
        "- contract_changed: GrowthPhenotype is in the recorded closure of none of "
        f"the selected datasets in {RELEASE}\n"
    )


# --------------------------------------------------------------------------- the real repo


def test_real_registry_holds_on_the_committed_snapshot() -> None:
    release = "2026.09.21-ab6d8c5d"
    registry = QueryRegistry.load(registry_path(REPO))
    report = check_repo(REPO, registry, release)
    assert [(r.query_id, r.drifts) for r in report.results] == [
        (q.id, []) for q in sorted(registry.queries, key=lambda q: q.id)
    ]
    assert report.exit_code == 0
    snapshot = load_snapshot(snapshot_paths(REPO, release)[0])
    assert {q.id: q.validated_release for q in registry.queries} == dict.fromkeys(
        [
            "amino_acid_betaxanthin",
            "essentiality_smf",
            "expression_proteome_morphology",
            "solid_growth_025",
        ],
        snapshot.release,
    )


def test_pre_commit_hook_fires_on_the_listed_paths_only() -> None:
    config = yaml.safe_load(
        (REPO / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    (hook,) = [
        hook
        for repo in config["repos"]
        for hook in repo["hooks"]
        if hook["id"] == "supported-queries"
    ]
    pattern = re.compile(hook["files"])
    paths = {
        "torchcell/datamodels/schema.py": True,
        "torchcell/datamodels/pydant.py": True,
        "biocypher/config/torchcell_schema_config.yaml": True,
        "torchcell/adapters/cell_adapter.py": True,
        "torchcell/knowledge_graphs/queries/solid_growth_025.cql": True,
        "torchcell/knowledge_graphs/supported_queries/registry.json": True,
        "torchcell/knowledge_graphs/supported_queries/check.py": True,
        "database/releases/2026.09.21-ab6d8c5d.closures.json": True,
        "torchcell/datamodels/media.py": False,
        "torchcell/adapters/kemmeren2014_adapter.py": False,
        "torchcell/knowledge_graphs/queries/__init__.py": False,
        "torchcell/knowledge_graphs/releases.py": False,
        "database/releases/README.md": False,
    }
    assert {path: bool(pattern.search(path)) for path in paths} == paths
    assert (hook["entry"], hook["pass_filenames"]) == (
        "bash scripts/run-supported-queries.sh",
        False,
    )
