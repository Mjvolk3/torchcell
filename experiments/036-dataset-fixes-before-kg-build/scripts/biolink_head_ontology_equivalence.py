# experiments/036-dataset-fixes-before-kg-build/scripts/biolink_head_ontology_equivalence.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.biolink_head_ontology_equivalence]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/biolink_head_ontology_equivalence
"""Compare the BioCypher ontology built from the local Biolink mirror vs the live URL.

Issue #619: the KG build now reads the head ontology from
``biocypher/ontology/biolink-model-v3.2.1.owl.ttl`` instead of BioCypher's default
GitHub URL. This script builds BioCypher's ``Ontology`` twice over
``biocypher/config/torchcell_schema_config.yaml``, once per source, and compares, for
every schema class, the ancestor list ``get_ancestors`` returns (the list the Neo4j batch
writer turns into the ``:LABEL`` column). It also re-fetches the URL and records the
sha256 of both byte streams.

The generation container installs ``biocypher==0.15.2``; the dev env carries an older
release, so run this with the 0.15.2 wheel on ``PYTHONPATH`` (the version used is
written into the result). Run from the repo root:

    PYTHONPATH=<repo>:<dir with biocypher 0.15.2> python \
        experiments/036-dataset-fixes-before-kg-build/scripts/biolink_head_ontology_equivalence.py

Output: ``experiments/036-dataset-fixes-before-kg-build/results/biolink_head_ontology_equivalence.json``.
"""

import hashlib
import json
import os.path as osp
from pathlib import Path

from biocypher._mapping import OntologyMapping
from biocypher._ontology import Ontology

import biocypher
from torchcell.knowledge_graphs.head_ontology import (
    BIOLINK_ROOT_NODE,
    BIOLINK_SOURCE_URL,
    REPO_ONTOLOGY_PATH,
)
from torchcell.literature.manifest import sha256_file
from torchcell.literature.retrieve import direct_url

SCHEMA = "biocypher/config/torchcell_schema_config.yaml"
OUT = (
    "experiments/036-dataset-fixes-before-kg-build/results/"
    "biolink_head_ontology_equivalence.json"
)


def ancestors_by_class(url: str) -> dict[str, list[str]]:
    """Every schema class -> its ``get_ancestors`` list under the given head ontology."""
    mapping = OntologyMapping(config_file=SCHEMA)
    ontology = Ontology(
        head_ontology={"url": url, "root_node": BIOLINK_ROOT_NODE},
        ontology_mapping=mapping,
    )
    return {
        name: [str(a) for a in ontology.get_ancestors(name)]
        for name in sorted(mapping.extended_schema)
    }


def main() -> None:
    """Build both ontologies, compare per-class ancestry, write the JSON result."""
    local = ancestors_by_class(osp.abspath(REPO_ONTOLOGY_PATH))
    remote = ancestors_by_class(BIOLINK_SOURCE_URL)
    differing = sorted(
        k for k in set(local) | set(remote) if local.get(k) != remote.get(k)
    )
    result = {
        "biocypher_version": biocypher.__version__,  # type: ignore[attr-defined]  # untyped package
        "schema_config": SCHEMA,
        "local_path": REPO_ONTOLOGY_PATH,
        "local_sha256": sha256_file(Path(REPO_ONTOLOGY_PATH)),
        "remote_url": BIOLINK_SOURCE_URL,
        "remote_sha256": hashlib.sha256(direct_url(BIOLINK_SOURCE_URL)).hexdigest(),
        "n_schema_classes_local": len(local),
        "n_schema_classes_remote": len(remote),
        "n_classes_with_differing_ancestors": len(differing),
        "differing_classes": differing,
        "ancestors_local": local,
    }
    Path(OUT).write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"biocypher {result['biocypher_version']}: {len(local)} classes, "
        f"{len(differing)} differ; local sha256 {result['local_sha256']}, "
        f"remote sha256 {result['remote_sha256']}"
    )


if __name__ == "__main__":
    main()
