# torchcell/knowledge_graphs/supported_queries/registry.py
# [[torchcell.knowledge_graphs.supported_queries.registry]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/supported_queries/registry.py
# Test file: tests/torchcell/knowledge_graphs/supported_queries/test_supported_queries_registry.py
"""The supported-query registry: which shipped ``.cql`` queries the project stands behind.

A **supported query** is a Cypher file under ``torchcell/knowledge_graphs/queries/`` that a
showcase page, an experiment or a collaborator runs against the served knowledge graph,
together with the converter that reads its records and the phenotype classes those
records carry. The registry (``registry.json`` beside this module, committed and shipped
as package data) records each one's lifecycle:

- ``status``: ``supported`` (a drift fails the check) or ``deprecated`` (a drift is
  reported and never fails; ``deprecated_in`` names the KG version it was deprecated in).
- ``since_kg_version``: the KG version the query was first validated on.
- ``validated_release`` and ``dataset_composite``: the release the query was last
  validated on and the sha256 over the sorted ``content_sha256`` of the datasets it
  selects in that release, computed the way ``release_snapshot.composite_sha256`` does.

The dependencies (labels, relationship types, properties, dataset ids, graph levels,
media names) are NOT stored: :func:`torchcell.knowledge_graphs.supported_queries
.cypher_deps.extract_dependencies` reads them from the ``.cql`` each time, so a stored copy
can never go stale. See [[plan.data-release-program.2026.09.29]], Decisions 3 and 4.

``save`` is byte-stable: queries sorted by id, fields in model order, ``indent=2``, one
trailing newline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

__all__ = [
    "QueryStatus",
    "KG_PACKAGE_RELPATH",
    "REGISTRY_RELPATH",
    "QueryDependencies",
    "SupportedQuery",
    "QueryRegistry",
    "registry_path",
]

QueryStatus = Literal["supported", "deprecated"]

# The package the ``cql_path`` of every query is relative to.
KG_PACKAGE_RELPATH = "torchcell/knowledge_graphs"
REGISTRY_RELPATH = f"{KG_PACKAGE_RELPATH}/supported_queries/registry.json"

_QUERY_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_DOTTED = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class QueryDependencies(BaseModel):
    """What a query reads from the graph, as extracted from its Cypher text.

    Every field is a sorted list without duplicates. ``label_properties`` holds
    ``Label.prop`` for each property read through a variable bound to ``Label``;
    ``properties`` holds the bare names of every property read through a bound variable.
    """

    model_config = ConfigDict(extra="forbid")

    node_labels: list[str]
    relationship_types: list[str]
    properties: list[str]
    label_properties: list[str]
    dataset_ids: list[str]
    graph_levels: list[str]
    media_names: list[str]
    parameters: list[str]

    @model_validator(mode="after")
    def _sorted_unique(self) -> QueryDependencies:
        for name, values in self:
            if values != sorted(set(values)):
                raise ValueError(f"{name} must be sorted and unique, got {values}")
        return self


class SupportedQuery(BaseModel):
    """One registered query and where it stands in the release lifecycle."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    cql_path: str  # relative to torchcell/knowledge_graphs, e.g. queries/x.cql
    converter: str | None  # dotted class path, e.g. torchcell.datamodels.x.Converter
    phenotype_classes: list[str]
    status: QueryStatus
    since_kg_version: str | None
    deprecated_in: str | None
    validated_release: str | None
    dataset_composite: str | None
    docs_page: str | None

    @field_validator("id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        if not _QUERY_ID.match(value):
            raise ValueError(f"query id must match {_QUERY_ID.pattern}: {value!r}")
        return value

    @field_validator("cql_path")
    @classmethod
    def _relative_cql(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".cql":
            raise ValueError(
                f"cql_path must be a relative .cql path inside {KG_PACKAGE_RELPATH}: "
                f"{value!r}"
            )
        return value

    @field_validator("converter")
    @classmethod
    def _dotted_converter(cls, value: str | None) -> str | None:
        if value is not None and not _DOTTED.match(value):
            raise ValueError(f"converter must be a dotted class path: {value!r}")
        return value

    @field_validator("phenotype_classes")
    @classmethod
    def _sorted_classes(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError(f"phenotype_classes must be sorted and unique: {value}")
        return value

    @field_validator("dataset_composite")
    @classmethod
    def _composite_shape(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256.match(value):
            raise ValueError(
                f"dataset_composite must be a sha256 hex digest: {value!r}"
            )
        return value

    @model_validator(mode="after")
    def _lifecycle(self) -> SupportedQuery:
        if (self.status == "deprecated") != (self.deprecated_in is not None):
            raise ValueError(
                f"{self.id}: deprecated_in is set exactly when status is deprecated "
                f"(status {self.status}, deprecated_in {self.deprecated_in})"
            )
        if (self.validated_release is None) != (self.dataset_composite is None):
            raise ValueError(
                f"{self.id}: validated_release and dataset_composite are recorded together"
            )
        return self

    def cql_file(self, repo_root: Path) -> Path:
        """The ``.cql`` file in the checkout at ``repo_root``."""
        return repo_root / KG_PACKAGE_RELPATH / self.cql_path


class QueryRegistry(BaseModel):
    """Every registered query; ``schema_version`` versions this file's shape."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    queries: list[SupportedQuery]

    @model_validator(mode="after")
    def _unique_ids(self) -> QueryRegistry:
        ids = [query.id for query in self.queries]
        duplicates = sorted({qid for qid in ids if ids.count(qid) > 1})
        if duplicates:
            raise ValueError(f"duplicate query ids: {duplicates}")
        return self

    def get(self, query_id: str) -> SupportedQuery:
        """The query registered as ``query_id``; KeyError names the known ids."""
        for query in self.queries:
            if query.id == query_id:
                return query
        raise KeyError(
            f"no supported query {query_id!r}; registered: "
            f"{sorted(q.id for q in self.queries)}"
        )

    def replace(self, query: SupportedQuery) -> QueryRegistry:
        """A registry with the entry of the same id swapped for ``query``."""
        self.get(query.id)
        return QueryRegistry(
            queries=[query if q.id == query.id else q for q in self.queries]
        )

    def dumps(self) -> str:
        """The byte-stable file text: queries sorted by id, one trailing newline."""
        data = self.model_dump(mode="json")
        data["queries"] = sorted(data["queries"], key=lambda q: q["id"])
        return json.dumps(data, indent=2) + "\n"

    def save(self, path: Path) -> None:
        """Write :meth:`dumps` to ``path``."""
        path.write_text(self.dumps(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> QueryRegistry:
        """Read and validate a registry file."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def registry_path(repo_root: Path) -> Path:
    """``registry.json`` in the checkout at ``repo_root``."""
    return repo_root / REGISTRY_RELPATH
