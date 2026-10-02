# experiments/036-dataset-fixes-before-kg-build/scripts/sm_media_served_check.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.sm_media_served_check]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/sm_media_served_check
"""Dump every served Media node with its composition counts (issue #143 close-out check).

Issue #143 reported the synthetic minimal (SM) media nodes as bare stubs (name + state,
empty components, dropouts and provenance). PR #425 sourced the three SM formulations on
main at d1bb7f4b4a, and the full rebuild of 2026-10-02 (job 3198, release
2026.10.02-833970cd) is the first served store built from that code. This script reads the
served graph under slurm and writes one row per Media node: name, state, base medium, and
the number of components, dropouts and provenance entries in ``serialized_data``, so the
note can state from the store itself whether any SM node is still a stub.

Run under slurm (``gh_sm_media_served_check.slurm``); never from a shell on the login node.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import os.path as osp
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

QUERY = """
MATCH (m:Media)
OPTIONAL MATCH (m)-[r]-()
RETURN m.name AS name, m.state AS state, m.serialized_data AS serialized_data,
       count(r) AS n_edges
ORDER BY name, state
"""


def media_rows(
    uri: str, user: str, password: str, database: str
) -> list[dict[str, Any]]:
    """Return one row per served Media node with its composition counts."""
    rows: list[dict[str, Any]] = []
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            for record in session.run(QUERY):
                payload = json.loads(record["serialized_data"])
                rows.append(
                    {
                        "name": record["name"],
                        "state": record["state"],
                        "n_edges": record["n_edges"],
                        "base_medium": (payload.get("base_medium") or {}).get("name")
                        if isinstance(payload.get("base_medium"), dict)
                        else payload.get("base_medium"),
                        "n_components": len(payload.get("components") or []),
                        "n_dropouts": len(payload.get("dropouts") or []),
                        "n_provenance": len(payload.get("provenance") or []),
                        "is_synthetic": payload.get("is_synthetic"),
                    }
                )
    finally:
        driver.close()
    return rows


def main() -> None:
    """Write the Media census CSV and print the SM rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="torchcell")
    parser.add_argument("--out", required=True, help="CSV path to write")
    args = parser.parse_args()
    load_dotenv()
    rows = media_rows(
        os.environ["NEO4J_URI"],
        os.environ["NEO4J_USER"],
        os.environ["NEO4J_PASSWORD"],
        args.database,
    )
    os.makedirs(osp.dirname(args.out), exist_ok=True)
    fields = [
        "name",
        "state",
        "n_edges",
        "base_medium",
        "n_components",
        "n_dropouts",
        "n_provenance",
        "is_synthetic",
    ]
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} Media nodes -> {args.out}")
    for row in rows:
        if "SM" in str(row["name"]) or "minimal" in str(row["name"]).lower():
            print(row)
    stubs = [r for r in rows if r["n_components"] == 0 and r["n_provenance"] == 0]
    print(f"stub media (no components, no provenance): {len(stubs)}")
    for row in stubs:
        print("  STUB", row)


if __name__ == "__main__":
    main()
