# torchcell/paper/ontology_usage.py
# [[torchcell.paper.ontology_usage]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/paper/ontology_usage.py
# Test file: tests/torchcell/paper/test_ontology_usage.py

"""Which served datasets use each schema class, and where each dataset is documented.

The ontology explorer shows the schema; this module adds the other half, the data that
is stored in it. For the newest committed release snapshot (``database/releases/``) it
reads each served dataset's schema closure, the set of schema classes its records can
hold, and gives every dataset two links on the documentation site:

- ``page_url``: the dataset page (what the experiments measured, a stored record, the
  distributions, the query), when a supported query with a ``docs_page`` selects the
  dataset. The query's Cypher names its datasets, so the mapping is read from the
  ``.cql`` file and the registry, never typed here.
- ``api_url``: the API reference page of the loader class, which every dataset has. The
  module that defines the class is found by parsing the loader sources, so no loader
  (and no torch) is imported.

Everything is read from files committed to the repository, so the explorer built in CI
agrees with the release snapshot at that commit.
"""

from __future__ import annotations

import ast
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from torchcell.knowledge_graphs.release_snapshot import load_closures, load_snapshots
from torchcell.knowledge_graphs.supported_queries.cypher_deps import (
    extract_dependencies,
)
from torchcell.knowledge_graphs.supported_queries.registry import (
    QueryRegistry,
    registry_path,
)

DOCS_URL = "https://mjvolk3.github.io/torchcell/"
DOCS_SOURCE_PREFIX = "docs/source/"
LOADERS_RELPATH = "torchcell/datasets/scerevisiae"
LOADERS_PACKAGE = "torchcell.datasets.scerevisiae"
REGISTER_DECORATOR = "register_dataset"


class DatasetUsage(BaseModel):
    """One served dataset: its size, the schema classes it uses, and its docs links."""

    model_config = ConfigDict(frozen=True)

    name: str
    n_experiments: int
    classes: list[str]
    api_url: str
    page_url: str | None
    page_title: str | None


class OntologyUsage(BaseModel):
    """Schema-class usage of every dataset served by one release."""

    model_config = ConfigDict(frozen=True)

    release: str
    kg_version: str
    datasets: list[DatasetUsage]

    def datasets_using(self, class_name: str) -> list[str]:
        """Names of the datasets whose closure holds ``class_name``, alphabetical."""
        return [d.name for d in self.datasets if class_name in d.classes]


def _is_registered(node: ast.ClassDef) -> bool:
    return any(
        isinstance(decorator, ast.Name) and decorator.id == REGISTER_DECORATOR
        for decorator in node.decorator_list
    )


def loader_modules(repo_root: Path) -> dict[str, str]:
    """Map every ``@register_dataset`` class to the loader module that defines it.

    The decorator is what makes a class a dataset; a legacy module that redefines a
    class name without registering it is not a second home for that dataset.
    """
    modules: dict[str, str] = {}
    for path in sorted((repo_root / LOADERS_RELPATH).glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and _is_registered(node):
                if node.name in modules:
                    raise ValueError(
                        f"dataset {node.name} is registered in "
                        f"{modules[node.name]}.py and {path.name}"
                    )
                modules[node.name] = path.stem
    return modules


def dataset_pages(repo_root: Path, docs_url: str) -> dict[str, tuple[str, str]]:
    """Map dataset class to ``(page url, page title)`` from the supported queries.

    A dataset selected by several documented queries takes the page of the first query
    in registry order.
    """
    pages: dict[str, tuple[str, str]] = {}
    registry = QueryRegistry.load(registry_path(repo_root))
    for query in registry.queries:
        if query.status != "supported" or query.docs_page is None:
            continue
        if not query.docs_page.startswith(DOCS_SOURCE_PREFIX):
            raise ValueError(f"{query.id}: docs_page is not under {DOCS_SOURCE_PREFIX}")
        relative = Path(query.docs_page.removeprefix(DOCS_SOURCE_PREFIX))
        url = docs_url + relative.with_suffix(".html").as_posix()
        cypher = query.cql_file(repo_root).read_text(encoding="utf-8")
        for dataset in extract_dependencies(cypher).dataset_ids:
            pages.setdefault(dataset, (url, query.title))
    return pages


def build_ontology_usage(repo_root: Path, docs_url: str = DOCS_URL) -> OntologyUsage:
    """Usage of the schema by the newest release snapshot committed under ``repo_root``."""
    snapshot = load_snapshots(repo_root)[-1]
    closures = load_closures(repo_root, snapshot.release)
    modules = loader_modules(repo_root)
    pages = dataset_pages(repo_root, docs_url)
    datasets: list[DatasetUsage] = []
    for name in sorted(snapshot.datasets):
        page = pages.get(name)
        module = modules[name]
        datasets.append(
            DatasetUsage(
                name=name,
                n_experiments=snapshot.datasets[name].n_experiments,
                classes=sorted(closures[name]),
                api_url=f"{docs_url}generated/{LOADERS_PACKAGE}.{module}.{name}.html",
                page_url=page[0] if page else None,
                page_title=page[1] if page else None,
            )
        )
    return OntologyUsage(
        release=snapshot.release, kg_version=snapshot.version, datasets=datasets
    )
