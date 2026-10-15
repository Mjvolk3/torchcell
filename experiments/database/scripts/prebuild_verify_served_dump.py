# experiments/database/scripts/prebuild_verify_served_dump.py
# [[experiments.database.scripts.prebuild_verify_served_dump]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/prebuild_verify_served_dump
r"""Dump the served KG 3.0 records of a few datasets, read-only, for a content comparison.

Lane P of the ontology unification (2026-10-15) rebuilds a sample of dev stores from
the integrated branch and has to show that nothing reachable in the served KG 3.0
became unreachable. Node ids re-key by design (P1 and P2 change every perturbation,
genotype, reference and experiment dump), so the comparison is by CONTENT, and the
content lives in the ``Experiment`` blob (``serialized_data``, a pointered dump whose
``$ref`` entries name ``InternedConstant`` nodes). This script streams, per dataset,
every experiment under the ``Dataset`` node through its ``ExperimentMemberOf`` edges
(no property index is needed, the same walk ``live_experiment_ids`` does) and writes
one JSONL row per experiment with its id and blob, plus ONE file of every
``InternedConstant`` node (about 21 k in 3.0, anchored on the label) so the pointers
resolve offline. The genotype nodes reached by ``GenotypeMemberOf`` are listed with
their labels, which is how the lane N defect (a genotype that exists only as an
``InternedConstant``) shows per record.

Read-only: every statement is a ``MATCH`` in a read transaction. Run under slurm
(``prebuild_verify_served_dump.slurm``), never from the login shell::

    python experiments/database/scripts/prebuild_verify_served_dump.py \\
        --datasets SmfKuzmin2018Dataset,BetaxanthinCachera2023Dataset \\
        --out-dir <dir>
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

EXPERIMENTS_QUERY = (
    "MATCH (d:Dataset {id: $name})<-[:ExperimentMemberOf]-(e:Experiment) "
    "OPTIONAL MATCH (e)<-[:GenotypeMemberOf]-(g) "
    "WITH e, collect({id: g.id, labels: labels(g)}) AS genotypes "
    "RETURN e.id AS id, e.serialized_data AS blob, genotypes"
)
CONSTANTS_QUERY = (
    "MATCH (c:InternedConstant) RETURN c.id AS id, c.kind AS kind, "
    "c.serialized_data AS payload"
)


def dump_dataset(session: Any, name: str, out: Path) -> int:
    """Write one JSONL row per served experiment of ``name``; return the row count."""
    n = 0
    with out.open("w", encoding="utf-8") as handle:
        for record in session.run(EXPERIMENTS_QUERY, name=name):
            handle.write(
                json.dumps(
                    {
                        "id": record["id"],
                        "blob": record["blob"],
                        "genotypes": record["genotypes"],
                    }
                )
                + "\n"
            )
            n += 1
    return n


def dump_constants(session: Any, out: Path) -> int:
    """Write every ``InternedConstant`` node as one JSONL row; return the count."""
    n = 0
    with out.open("w", encoding="utf-8") as handle:
        for record in session.run(CONSTANTS_QUERY):
            handle.write(
                json.dumps(
                    {
                        "id": record["id"],
                        "kind": record["kind"],
                        "payload": record["payload"],
                    }
                )
                + "\n"
            )
            n += 1
    return n


def main() -> None:
    """CLI entry point."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687")
    )
    parser.add_argument("--database", default="torchcell")
    parser.add_argument("--datasets", required=True, help="comma-separated classes")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    driver = GraphDatabase.driver(
        args.uri,
        auth=(
            os.getenv("NEO4J_USER", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "torchcell"),
        ),
    )
    summary: dict[str, Any] = {"database": args.database, "datasets": {}}
    with driver.session(database=args.database, default_access_mode="READ") as session:
        t0 = time.time()
        n_constants = session.read_transaction(
            lambda tx: dump_constants(tx, out_dir / "interned_constants.jsonl")
        )
        summary["interned_constants"] = n_constants
        print(f"interned constants: {n_constants} in {time.time() - t0:.0f}s")
        for name in args.datasets.split(","):
            t0 = time.time()
            n = session.read_transaction(
                lambda tx, name=name: dump_dataset(tx, name, out_dir / f"{name}.jsonl")
            )
            summary["datasets"][name] = n
            print(f"{name}: {n} served experiments in {time.time() - t0:.0f}s")
    driver.close()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
