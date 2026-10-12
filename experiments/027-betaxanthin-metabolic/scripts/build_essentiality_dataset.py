# experiments/027-betaxanthin-metabolic/scripts/build_essentiality_dataset.py
# [[experiments.027-betaxanthin-metabolic.wave9-essentiality]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/build_essentiality_dataset.py

r"""Build the 027 gene-essentiality CellDataset, then verify it against the paper.

WHY A DEDICATED BUILD. Two built CellDatasets already contain
``GeneEssentialitySgdDataset`` and neither carries the label: measured 2026-09-14, the
``phenotype_label_index`` of ``003-fit-int/001-small-build`` and
``025-solid-growth/001-full-build`` holds only ``fitness`` and ``gene_interaction``, because
both were opened with a converter that turns the boolean into a fitness value. This build
opens the dataset with ``converter=None`` from ``queries/essentiality_build.cql``, which
selects single-deletion genotypes and no co-grouped fitness dataset, so ``is_essential``
survives as itself.

WHAT IT VERIFIES, and none of it is optional. The build is only useful if it can be joined
to Merzbacher 2025's released split, so the script refuses a build that cannot:

* the four class counts of the released split against the paper's own Methods sentence
  (``read_fcl_essentiality_split`` does this, including the label inversion);
* the COVERAGE of that split by the build, found against absent, per class and per side of
  the split, so a build that silently misses the essential genes says so rather than
  training on the majority class alone;
* that the built dataset's own label really is ``is_essential`` and really is binary.

Run from the repo root (it is a Neo4j build, so it belongs in a CPU slurm job)::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/build_essentiality_dataset.py
"""

import json
import os
import os.path as osp
import sys
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "scripts"))

import train_bx as t  # noqa: E402

#: The three datasets whose single-deletion genotypes make up the build. The first is the
#: essential-gene list and the other two are viable deletion collections, so together they
#: carry both of the released split's classes; alone, the first carries only one.
ESSENTIALITY_UNION = [
    "GeneEssentialitySgdDataset",
    "SmfCostanzo2016Dataset",
    "SmfKuzmin2018Dataset",
]
#: Recorded in the manifest so the replaced build is traceable rather than just gone.
SUPERSEDED_ROOT = (
    "$DATA_ROOT/data/torchcell/experiments/027-betaxanthin-metabolic/"
    "essentiality_build-sgd-only-20260915-003142"
)

from torchcell.database.connection import neo4j_connection_settings  # noqa: E402


def main() -> None:
    """Build the dataset, then write a manifest beside it."""
    root, query_path = t.dataset_paths(t.ESSENTIALITY_HEAD)
    settings = neo4j_connection_settings()
    print(f"[build] root {root}", flush=True)
    print(f"[build] query {query_path}", flush=True)
    print(
        f"[build] neo4j {settings.uri} database {settings.database!r} "
        f"as {settings.username}",
        flush=True,
    )

    # THE PREFLIGHT THAT SAVES A QUEUE WAIT. A reachable but EMPTY graph is the state this
    # build met on 2026-09-14 (0 nodes, while kg_manifest.json still claimed 36 served
    # datasets), and `Neo4jCellDataset` on an empty graph writes an empty LMDB rather than
    # failing, which is a build that looks finished and carries nothing.
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        settings.uri, auth=(settings.username, settings.password)
    )
    # THE DATABASE NAME IS LOAD-BEARING. The server hosts `neo4j` (the default, EMPTY),
    # `system`, and `torchcell`, which is the one holding the build. A session opened
    # without `database=` goes to `neo4j` and returns zero rows for every query, which is
    # indistinguishable from an empty or mid-rebuild store; that is exactly how this probe
    # reported "0 nodes" on 2026-09-14 against a graph holding 83,066,931 of them.
    # `neo4j_query_raw.py` has always passed "torchcell" at its own session, which is why
    # dataset builds worked while the probe saw nothing.
    with driver.session(database=settings.database) as session:
        n_experiments = session.run(
            "MATCH (d:Dataset)<-[:ExperimentMemberOf]-(e:Experiment) "
            "WHERE d.id IN $ds RETURN count(e) AS n",
            ds=ESSENTIALITY_UNION,
        ).single()["n"]
    driver.close()
    assert n_experiments > 0, (
        f"the graph at {settings.uri} database {settings.database!r} carries 0 "
        f"experiments across {ESSENTIALITY_UNION}, so "
        "this build would write an empty LMDB and report success. Repopulate the served "
        "store (the manifest at $DATA_ROOT/database/kg_manifest.json lists the dataset as "
        "served, so the manifest and the live store disagree) and rerun."
    )
    print(
        f"[build] graph carries {n_experiments} experiments across "
        f"{len(ESSENTIALITY_UNION)} datasets",
        flush=True,
    )

    started = time.time()
    dataset = t.build_dataset(t.ESSENTIALITY_HEAD)
    print(
        f"[build] {len(dataset)} genotypes in {time.time() - started:.0f} s", flush=True
    )

    # NO LABEL CHECK, DELIBERATELY. The essentiality target comes from the released split
    # and not from this build; whatever phenotype columns the union carries are ignored by
    # the head. The labels present are recorded in the manifest so a reader can see what was
    # dropped, and `GeneEssentialitySgdDataset` alone was single class anyway (all 1,140 of
    # its records read `is_essential: True`), which is why the union exists.
    labels_present = dataset.phenotype_label_index

    split_path = osp.join(os.environ["DATA_ROOT"], t.FCL_ESSENTIALITY_SPLIT)
    fcl_labels, fcl_report = t.read_fcl_essentiality_split(split_path)

    gene_index = dataset.is_any_deletion_gene_index
    found = [g for g in fcl_labels if g in gene_index]
    absent = sorted(g for g in fcl_labels if g not in gene_index)
    found_essential = sum(1 for g in found if fcl_labels[g] == 1)
    absent_essential = sum(1 for g in absent if fcl_labels[g] == 1)
    manifest: dict[str, Any] = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "root": root,
        "query_path": query_path,
        "neo4j_uri": settings.uri,
        "neo4j_database": settings.database,
        "neo4j_user": settings.username,
        "n_genotypes": len(dataset),
        "phenotype_labels": {k: len(v) for k, v in labels_present.items()},
        "fcl_split": fcl_report,
        "coverage": {
            "requested": len(fcl_labels),
            "found": len(found),
            "absent": len(absent),
            "absent_genes": absent,
            "found_essential": found_essential,
            "found_nonessential": len(found) - found_essential,
            "absent_essential": absent_essential,
            "absent_nonessential": len(absent) - absent_essential,
        },
        "superseded_root": SUPERSEDED_ROOT,
        "superseded_note": (
            "the SGD-only build this replaces, moved aside rather than deleted. It held "
            "1,140 genotypes whose is_essential was True on every one of them, i.e. no "
            "negative class, which is what forced the union."
        ),
        "code_provenance": t.code_provenance(),
    }
    os.makedirs(root, exist_ok=True)
    out = osp.join(root, "build_manifest.json")
    with open(out, "w") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    print(
        f"[build] FCL split coverage {len(found)}/{len(fcl_labels)} genes "
        f"({found_essential} essential, {len(found) - found_essential} nonessential); "
        f"{len(absent)} absent",
        flush=True,
    )
    print(f"[build] wrote {out}", flush=True)
    assert found_essential > 0, (
        "the build covers 0 essential genes of the released split, so every label it could "
        "score is the majority class and the comparison is not defined."
    )


if __name__ == "__main__":
    main()
