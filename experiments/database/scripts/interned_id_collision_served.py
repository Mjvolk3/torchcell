# experiments/database/scripts/interned_id_collision_served.py
# [[experiments.database.scripts.interned_id_collision_served]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/interned_id_collision_served
r"""Measure the interned-id collision on the SERVED knowledge graph, read-only.

Companion of ``interned_id_collision_counts.py`` (which reads a BioCypher output
directory): the same shape asked of the live store through bolt. Every query is
anchored on the ``InternedConstant`` label (a few hundred thousand nodes), so the
measurement never scans ``Experiment`` or ``Genotype``. The per-dataset
attribution walks ``InternedConstant -[GenotypeMemberOf]-> Experiment
-[ExperimentMemberOf]-> Dataset``. A sample of the experiment blobs of each
affected dataset is parsed for its genotype pointers and each pointer is resolved
against the store, which measures the direction the query loader depends on (a
pointer with no constant) on a sample.

Usage (from the repo root, the torchcell environment; ``NEO4J_URI``, ``NEO4J_USER``
and ``NEO4J_PASSWORD`` from ``.env``)::

    python experiments/database/scripts/interned_id_collision_served.py \\
        --database torchcell \\
        --out experiments/database/results/interned_id_collision_served.json

Read-only: every statement is a ``MATCH`` or ``SHOW`` run in a read transaction.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase
from pydantic import BaseModel, ConfigDict, Field

from torchcell.datamodels.interned_constant import POINTER_KEY

GENOTYPE_KINDS = ("genotype", "genotype.background")


class EdgeToInterned(BaseModel):
    """Rows of one relationship type whose genotype-side endpoint is an interned constant."""

    model_config = ConfigDict(frozen=True)

    edge: str
    side: str
    n_rows_total: int
    n_rows_to_interned: int
    n_distinct_interned: int


class ServedDatasetLoss(BaseModel):
    """One dataset's genotype nodes that exist only as interned constants."""

    model_config = ConfigDict(frozen=True)

    dataset: str
    lost_by_kind: dict[str, int]
    n_genotype_member_of_rows: int
    n_sampled_experiments: int
    n_sampled_pointers: int
    n_sampled_pointers_unresolved: int = Field(
        description="pointers of the sampled blobs that name no InternedConstant node"
    )


class ServedCollision(BaseModel):
    """The served store's measurement."""

    model_config = ConfigDict(frozen=True)

    uri: str
    database: str
    n_nodes_by_label: dict[str, int]
    n_interned_by_kind: dict[str, int]
    edges: list[EdgeToInterned]
    datasets: list[ServedDatasetLoss]
    elapsed_s: float


def _rows(session: Any, query: str, **params: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = session.read_transaction(
        lambda tx: [r.data() for r in tx.run(query, **params)]
    )
    return rows


def measure(
    uri: str, auth: tuple[str, str] | None, database: str, sample: int
) -> ServedCollision:
    t0 = time.monotonic()
    driver = GraphDatabase.driver(uri, auth=auth)
    with driver.session(database=database, default_access_mode="READ") as session:
        labels = (
            "Genotype",
            "SegregantGenotype",
            "PerturbedSegregantGenotype",
            "InternedConstant",
            "Perturbation",
            "BacterialPerturbation",
            "Experiment",
        )
        n_nodes = {
            label: _rows(session, f"MATCH (n:{label}) RETURN count(n) AS n")[0]["n"]
            for label in labels
        }
        by_kind = {
            r["kind"]: r["n"]
            for r in _rows(
                session,
                "MATCH (c:InternedConstant) RETURN c.kind AS kind, count(c) AS n",
            )
        }
        edges: list[EdgeToInterned] = []
        for edge, pattern in (
            (
                "PerturbationMemberOf",
                "(c:InternedConstant)<-[r:PerturbationMemberOf]-()",
            ),
            ("GenotypeMemberOf", "(c:InternedConstant)-[r:GenotypeMemberOf]->()"),
            (
                "SegregantBackgroundOf",
                "(c:InternedConstant)-[r:SegregantBackgroundOf]->()",
            ),
        ):
            total = _rows(session, f"MATCH ()-[r:{edge}]->() RETURN count(r) AS n")[0][
                "n"
            ]
            hit = _rows(
                session, f"MATCH {pattern} RETURN count(r) AS n, count(DISTINCT c) AS d"
            )[0]
            edges.append(
                EdgeToInterned(
                    edge=edge,
                    side="end" if edge == "PerturbationMemberOf" else "start",
                    n_rows_total=total,
                    n_rows_to_interned=hit["n"],
                    n_distinct_interned=hit["d"],
                )
            )
        per_dataset = _rows(
            session,
            "MATCH (c:InternedConstant)-[r:GenotypeMemberOf]->(e:Experiment)"
            "-[:ExperimentMemberOf]->(d:Dataset) "
            "RETURN d.id AS dataset, c.kind AS kind, count(DISTINCT c) AS lost, "
            "count(r) AS rows ORDER BY dataset, kind",
        )
        datasets: list[ServedDatasetLoss] = []
        grouped: dict[str, dict[str, Any]] = {}
        for row in per_dataset:
            entry = grouped.setdefault(row["dataset"], {"kinds": {}, "rows": 0})
            entry["kinds"][row["kind"]] = row["lost"]
            entry["rows"] += row["rows"]
        for name, entry in grouped.items():
            blobs = _rows(
                session,
                "MATCH (d:Dataset {id: $name})<-[:ExperimentMemberOf]-(e:Experiment) "
                "RETURN e.serialized_data AS blob LIMIT $sample",
                name=name,
                sample=sample,
            )
            refs: list[str] = []
            for blob in blobs:
                dump = json.loads(blob["blob"])
                genotype = dump.get("genotype")
                if isinstance(genotype, dict):
                    if POINTER_KEY in genotype:
                        refs.append(genotype[POINTER_KEY])
                    background = genotype.get("background")
                    if isinstance(background, dict) and POINTER_KEY in background:
                        refs.append(background[POINTER_KEY])
            resolved = _rows(
                session,
                "UNWIND $refs AS ref MATCH (c:InternedConstant {id: ref}) "
                "RETURN count(c) AS n",
                refs=refs,
            )[0]["n"]
            datasets.append(
                ServedDatasetLoss(
                    dataset=name,
                    lost_by_kind=entry["kinds"],
                    n_genotype_member_of_rows=entry["rows"],
                    n_sampled_experiments=len(blobs),
                    n_sampled_pointers=len(refs),
                    n_sampled_pointers_unresolved=len(refs) - resolved,
                )
            )
    driver.close()
    return ServedCollision(
        uri=uri,
        database=database,
        n_nodes_by_label=n_nodes,
        n_interned_by_kind=by_kind,
        edges=edges,
        datasets=datasets,
        elapsed_s=time.monotonic() - t0,
    )


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687")
    )
    parser.add_argument("--database", default="torchcell")
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    user, password = (
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "torchcell"),
    )
    auth = (user, password) if user and password else None
    result = measure(args.uri, auth, args.database, args.sample)
    print(result.model_dump_json(indent=2))
    Path(args.out).write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
