# experiments/database/scripts/organism_summary.py
# [[experiments.database.scripts.organism_summary]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/organism_summary
r"""One row per organism of a knowledge-graph release: datasets, records, experiment
types, the reference genomes kept beside the graph, and the gene count of the primary
reference. The dissertation's organism table (thesis ``common/organisms_table.py``,
``tab:organisms``) is rendered from the JSON this writes, so no number on that table is
typed by hand.

Every value is read, never typed:

- ``database/releases/<release>.json`` (the release snapshot the build stamped): the
  organism and record count (``n_experiments``) of every dataset in the graph.
- ``experiments/database/scripts/dataset_probe_cache.json`` (probe variant ``probes-v1``):
  the experiment type each dataset's records instantiate. A dataset of the release that
  the cache does not hold is listed under ``unprobed_datasets`` and counted in no type.
- ``$DATA_ROOT/torchcell-genomes/<set>/manifest.json``: the assembly sets of each organism,
  with the strain or population each one is and the roles of its files. A set whose files
  carry a ``sequence`` or ``container`` role is an assembly; a set of ``variants`` alone is
  a variant table against a reference, kept in the tier but not an assembly.
- the genome classes of the primary reference (the one the organism's gene sets are
  written against): ``SCerevisiaeGenome`` (SGD S288C R64-4-1), ``EcoliK12MG1655Genome``
  (K-12 MG1655 ASM584v2), ``PPutidaKT2440Genome`` (KT2440 ASM756v2); ``len(gene_set)``
  is the gene count. The caches these classes keep live under ``$DATA_ROOT/data`` and
  are built from the registry-resolved, sha256-verified tier files.

Usage (from the repo root, torchcell env)::

    python experiments/database/scripts/organism_summary.py \\
        [--release 2026.10.17-4295f7ca] [--out experiments/database/results/organism_summary.json]

Prints a markdown table of the same rows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import os.path as osp
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from torchcell.sequence.genome.registry import (
    ROLE_CONTAINER,
    ROLE_SEQUENCE,
    genomes_root,
    load_genome_manifest,
)

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.abspath(osp.join(HERE, "..", "..", ".."))
RELEASES = osp.join(REPO, "database", "releases")
PROBE_CACHE = osp.join(HERE, "dataset_probe_cache.json")
PROBE_VARIANT = "probes-v1"
DEFAULT_RELEASE = "2026.10.17-4295f7ca"  # knowledge graph 4.0
DEFAULT_OUT = osp.join(
    REPO, "experiments", "database", "results", "organism_summary.json"
)

#: release organism key -> the manifest's organism name and the row order of the table
ORGANISMS: dict[str, str] = {
    "scerevisiae": "Saccharomyces cerevisiae",
    "ecoli": "Escherichia coli",
    "pputida": "Pseudomonas putida",
}


def _yeast_genes(data_root: str) -> tuple[str, int]:
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data", "sgd", "genome"),
        go_root=osp.join(data_root, "data", "go"),
        overwrite=False,
    )
    return "SCerevisiaeGenome", len(genome.gene_set)


def _ecoli_genes(data_root: str) -> tuple[str, int]:
    from torchcell.sequence.genome.ecoli.k12 import EcoliK12MG1655Genome

    genome = EcoliK12MG1655Genome(
        genome_root=osp.join(data_root, "data", "ecoli", "mg1655", "genome")
    )
    return "EcoliK12MG1655Genome", len(genome.gene_set)


def _pputida_genes(data_root: str) -> tuple[str, int]:
    from torchcell.sequence.genome.pputida.kt2440 import PPutidaKT2440Genome

    genome = PPutidaKT2440Genome(
        genome_root=osp.join(data_root, "data", "pputida", "kt2440", "genome")
    )
    return "PPutidaKT2440Genome", len(genome.gene_set)


#: the primary reference of each organism: its assembly set and the genome class that reads it
PRIMARY = {
    "scerevisiae": ("sgd_S288C_R64-4-1_20230830", _yeast_genes),
    "ecoli": ("ecoli_K12_MG1655_ASM584v2", _ecoli_genes),
    "pputida": ("pputida_KT2440_ASM756v2", _pputida_genes),
}


def sha256_of(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def release_rows(release: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = osp.join(RELEASES, f"{release}.json")
    snap = json.loads(Path(path).read_text())
    assert snap["release"] == release, (snap["release"], release)
    return snap, snap["datasets"]


def probe_types(
    datasets: dict[str, dict[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Dataset class -> experiment type for every probed dataset, and the cache's identity."""
    raw = json.loads(Path(PROBE_CACHE).read_text())
    variants = {v["variant"] for v in raw.values()}
    if variants != {PROBE_VARIANT}:
        raise SystemExit(f"unexpected probe variants {variants}")
    by_class = {v["record"]["dataset_class"]: v["record"] for v in raw.values()}
    types = {k: by_class[k]["experiment_type"] for k in datasets if k in by_class}
    identity = {
        "path": osp.relpath(PROBE_CACHE, REPO),
        "variant": PROBE_VARIANT,
        "sha256": sha256_of(PROBE_CACHE),
        "modified": dt.datetime.fromtimestamp(osp.getmtime(PROBE_CACHE))
        .date()
        .isoformat(),
    }
    return types, identity


def genome_sets(data_root: str) -> dict[str, list[dict[str, Any]]]:
    """The assembly sets of the tier, grouped by the release's organism key."""
    by_name = {name: key for key, name in ORGANISMS.items()}
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    root = genomes_root(data_root)
    for assembly_set in sorted(os.listdir(root)):
        if not osp.isfile(osp.join(root, assembly_set, "manifest.json")):
            continue
        m = load_genome_manifest(assembly_set, data_root)
        key = by_name.get(m.organism)
        if key is None:  # the GO release, and organisms the graph does not hold
            continue
        roles = sorted({f.role for f in m.files})
        out[key].append(
            {
                "assembly_set": m.assembly_set,
                "strain": m.strain_or_population,
                "source": m.source,
                "release": m.release,
                "roles": roles,
                "is_assembly": bool({ROLE_SEQUENCE, ROLE_CONTAINER} & set(roles)),
            }
        )
    return out


def summarize(release: str, data_root: str) -> dict[str, Any]:
    snap, datasets = release_rows(release)
    types, probe_identity = probe_types(datasets)
    sets = genome_sets(data_root)
    organisms: dict[str, Any] = {}
    for key, name in ORGANISMS.items():
        mine = {k: v for k, v in datasets.items() if v["organism"] == key}
        assert mine, f"release {release} holds no {key} dataset"
        unprobed = sorted(k for k in mine if k not in types)
        etypes = sorted({types[k] for k in mine if k in types})
        primary_set, count_genes = PRIMARY[key]
        primary = next(s for s in sets[key] if s["assembly_set"] == primary_set)
        genome_class, n_genes = count_genes(data_root)
        organisms[key] = {
            "name": name,
            "datasets": len(mine),
            "records": sum(v["n_experiments"] for v in mine.values()),
            "experiment_types": etypes,
            "n_experiment_types": len(etypes),
            "unprobed_datasets": unprobed,
            # the primary reference first, then the other assemblies, then variant tables
            "reference_genomes": sorted(
                sets[key],
                key=lambda s: (
                    s["assembly_set"] != primary_set,
                    not s["is_assembly"],
                    s["strain"],
                ),
            ),
            "primary_reference": {
                **primary,
                "genome_class": genome_class,
                "genes": n_genes,
            },
        }
    return {
        "release": snap["release"],
        "version": snap["version"],
        "built_at": snap["built_at"],
        "torchcell_version": snap["torchcell_version"],
        "torchcell_tag": snap.get("torchcell_tag"),
        "release_snapshot": {
            "path": osp.relpath(osp.join(RELEASES, f"{release}.json"), REPO),
            "sha256": sha256_of(osp.join(RELEASES, f"{release}.json")),
        },
        "probe_cache": probe_identity,
        "genomes_root": genomes_root(data_root),
        "measured_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "organisms": organisms,
        "total": {
            "datasets": sum(o["datasets"] for o in organisms.values()),
            "records": sum(o["records"] for o in organisms.values()),
        },
    }


def markdown(summary: dict[str, Any]) -> str:
    head = "| Organism | Reference genomes | Genes | Datasets | Experiment types | Records |"
    rows = [head, "| :-- | :-- | --: | --: | --: | --: |"]
    for o in summary["organisms"].values():
        refs = "; ".join(
            s["strain"] for s in o["reference_genomes"] if s["is_assembly"]
        )
        rows.append(
            f"| *{o['name']}* | {refs} | {o['primary_reference']['genes']:,} | {o['datasets']} "
            f"| {o['n_experiment_types']} | {o['records']:,} |"
        )
    t = summary["total"]
    rows.append(f"| **Total** | | | **{t['datasets']}** | | **{t['records']:,}** |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--release", default=DEFAULT_RELEASE, help="release id, <date>-<commit8>"
    )
    ap.add_argument("--out", default=DEFAULT_OUT, help="JSON to write")
    args = ap.parse_args(argv)
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    summary = summarize(args.release, data_root)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(markdown(summary))
    print(f"\nwrote {osp.relpath(args.out, REPO)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
