# experiments/036-dataset-fixes-before-kg-build/scripts/artifact_tier_live_radiant.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.artifact_tier_live_radiant]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/artifact_tier_live_radiant

"""The wiring diagram, run for real against the public hosts.

A fresh process with no ``TC_DATA_URL``, no key and an EMPTY artifact data root queries
the served knowledge graph on Radiant for a few Caudal 2024 records, lets the raw stage's
resolvability gate check every pointer those records carry against the public tc-data
endpoint, lists the distinct pointers, and materializes one file into the cache, verified
against its pinned sha256. Nothing here reads a local tier: every byte comes from Taiga
through Radiant. Writes ``results/artifact_tier_live_radiant.json``.

Run from the repo root:
    python experiments/036-dataset-fixes-before-kg-build/scripts/artifact_tier_live_radiant.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import shutil
import tempfile
import time
from datetime import UTC, datetime

from torchcell import __version__
from torchcell.artifacts.walk import distinct_refs
from torchcell.data.neo4j_query_raw import Neo4jQueryRaw
from torchcell.datamodels.schema import ArtifactRef
from torchcell.endpoint_http import DEFAULT_URL

RADIANT_URI = "neo4j+s://torchcell-database.ncsa.illinois.edu:7687"
RESULTS = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))),
    "results",
    "artifact_tier_live_radiant.json",
)
QUERY = """
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'CaudalPanTranscriptome2024Dataset'
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
WITH DISTINCT e, ref ORDER BY e.id LIMIT 6
RETURN e, ref
"""


def main() -> None:
    """Query, gate, list the pointers, materialize one file; record every number."""
    for var in ("TC_DATA_URL", "TC_DATA_API_KEY"):
        os.environ.pop(var, None)
    work = tempfile.mkdtemp(prefix="artifact-tier-live-")
    empty_tier = tempfile.mkdtemp(prefix="empty-data-root-")
    try:
        t0 = time.perf_counter()
        raw = Neo4jQueryRaw(
            uri=RADIANT_URI,
            username="torchcell",
            password="torchcell",
            root_dir=work,
            query=QUERY,
            artifact_data_root=empty_tier,
        )
        n = len(raw)
        gate_seconds = time.perf_counter() - t0
        refs: dict[str, ArtifactRef] = {}
        for i in range(n):
            rec = raw._get_record_by_index(i)
            for ref in distinct_refs(
                [rec["experiment"], rec["experiment_reference"]]
            ).values():
                refs[str(ref)] = ref
        first = refs[sorted(refs)[0]]
        t0 = time.perf_counter()
        path = raw.materialize(first)
        materialize_seconds = time.perf_counter() - t0
        result = {
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "torchcell_version": __version__,
            "graph": RADIANT_URI,
            "tc_data": DEFAULT_URL,
            "records": n,
            "query_and_gate_seconds": round(gate_seconds, 1),
            "distinct_refs": sorted(refs),
            "materialized": {
                "ref": str(first),
                "bytes": path.stat().st_size,
                "seconds": round(materialize_seconds, 2),
                "cache_path_relative": osp.relpath(path, empty_tier),
            },
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(empty_tier, ignore_errors=True)
    os.makedirs(osp.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)
        handle.write("\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
