# experiments/034-showcase-datasets/scripts/query_amino_acid_betaxanthin.py
# [[experiments.034-showcase-datasets.scripts.query_amino_acid_betaxanthin]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/034-showcase-datasets/scripts/query_amino_acid_betaxanthin
"""Run the supported amino-acid + betaxanthin query against the served graph and record it.

Builds a no-merge ``Neo4jCellDataset`` over
``torchcell/knowledge_graphs/queries/amino_acid_betaxanthin.cql``: no converter (every
entry is already a ``MetabolitePhenotype``), no deduplicator, and ``GenotypeAggregator``
groups every entry of one perturbed gene set under one processed record. Because the
aggregation key is the full perturbed gene set, a Mulleder and a Cooper record of the same
deletion share a processed record, while a Cachera record of that deletion (which also
carries the four Btx-cassette genes) is a record of its own.

Writes what the showcase page embeds:

- ``experiments/034-showcase-datasets/results/amino_acid_betaxanthin_query.json``: entries
  per ``dataset_name`` at each stage (raw query, processed records), the processed record
  count, records by the datasets their entries come from, the deleted genes shared across
  datasets, the metabolite keys each dataset returns, one processed record of each shape,
  the served release reported by ``python -m torchcell.knowledge_graphs.releases status
  --json``, and the slurm job id;
- ``docs/source/showcase/_generated/amino-acid-betaxanthin/query_results.md``: the same
  numbers as a MyST fragment.

The served graph is queried only under slurm
(``experiments/034-showcase-datasets/scripts/gh_query_amino_acid_betaxanthin.slurm``); the
dataset cache lives under ``$DATA_ROOT/data/torchcell/showcase_amino_acid_betaxanthin``.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lmdb
from dotenv import load_dotenv

import torchcell
from torchcell.data.genotype_aggregate import GenotypeAggregator
from torchcell.data.graph_processor import SubgraphRepresentation
from torchcell.data.neo4j_cell import Neo4jCellDataset
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

SCRIPT = "experiments/034-showcase-datasets/scripts/query_amino_acid_betaxanthin.py"
EXPERIMENT_DIR = Path("experiments/034-showcase-datasets")
RESULTS_JSON = EXPERIMENT_DIR / "results" / "amino_acid_betaxanthin_query.json"
FRAGMENT_DIR = Path("docs/source/showcase/_generated/amino-acid-betaxanthin")
QUERY_PATH = (
    Path(torchcell.__file__).parent
    / "knowledge_graphs"
    / "queries"
    / "amino_acid_betaxanthin.cql"
)
DATASET_SUBDIR = "data/torchcell/showcase_amino_acid_betaxanthin"
DATASETS = (
    "AminoAcidMulleder2016Dataset",
    "AminoAcidCooper2010Dataset",
    "BetaxanthinCachera2023Dataset",
)
SHORT = {
    "AminoAcidMulleder2016Dataset": "Mulleder",
    "AminoAcidCooper2010Dataset": "Cooper",
    "BetaxanthinCachera2023Dataset": "Cachera",
}

# The processed-record dump keeps these paths of each entry's experiment dict; the
# metabolite dicts are cut to their first keys by ``_cut_levels``.
ENTRY_FIELDS = (
    "dataset_name",
    "experiment_type",
    "environment.media.name",
    "environment.temperature",
    "phenotype.measurement_type",
)
LEVELS_SHOWN = 4


def _pick(record: dict[str, Any], path: str) -> Any:
    """Value at a dotted path with ``[i]`` list indices, e.g. ``a.b[0].c``."""
    node: Any = record
    for part in path.split("."):
        name, _, index = part.partition("[")
        node = node[name]
        if index:
            node = node[int(index.rstrip("]"))]
    return node


def _entries(payload: Any) -> list[dict[str, Any]]:
    """Raw stages store one dict per key; the processed stage a list of entries."""
    return payload if isinstance(payload, list) else [payload]


def _deleted_genes(experiment: dict[str, Any]) -> list[str]:
    """Systematic names of the deletion perturbations of one experiment dict."""
    return sorted(
        p["systematic_gene_name"]
        for p in experiment["genotype"]["perturbations"]
        if p["perturbation_type"].endswith("deletion")
    )


def stage_counts(lmdb_dir: str) -> dict[str, dict[str, int]]:
    """Entries per ``dataset_name`` and ``experiment_type`` in one pipeline-stage LMDB."""
    counts: dict[str, dict[str, int]] = {}
    env = lmdb.open(lmdb_dir, readonly=True, lock=False)
    with env.begin() as txn:
        for _, value in txn.cursor():
            for entry in _entries(json.loads(value.decode())):
                e = entry["experiment"]
                by_type = counts.setdefault(e["dataset_name"], {})
                by_type[e["experiment_type"]] = by_type.get(e["experiment_type"], 0) + 1
    env.close()
    return counts


def processed_records(processed_lmdb: str) -> dict[int, list[dict[str, Any]]]:
    """Every processed record's entry list, keyed by index."""
    out: dict[int, list[dict[str, Any]]] = {}
    env = lmdb.open(processed_lmdb, readonly=True, lock=False)
    with env.begin() as txn:
        for key, value in txn.cursor():
            out[int(key.decode())] = _entries(json.loads(value.decode()))
    env.close()
    return out


def releases_status() -> dict[str, Any]:
    """``releases status --json`` output, or the command's failure, verbatim."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchcell.knowledge_graphs.releases",
            "status",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    result: dict[str, Any] = {"returncode": proc.returncode}
    if proc.returncode == 0:
        result["status"] = json.loads(proc.stdout)
    else:
        result["stderr"] = proc.stderr[-4000:]
    return result


def served_release(releases: dict[str, Any]) -> dict[str, Any] | None:
    """The ``latest``-aliased database's row of ``releases status``, cut to the page."""
    if releases["returncode"] != 0:
        return None
    rows = [db for dbs in releases["status"].values() for db in dbs]
    latest = [db for db in rows if "latest" in db["aliases"]]
    assert len(latest) == 1, f"expected one database aliased latest, got {len(latest)}"
    db = latest[0]
    release = db["release"]
    return {
        "database": db["name"],
        "release": release["release"],
        "version": release["version"],
        "torchcell_commit": release["torchcell_commit"],
        "torchcell_version": release.get("torchcell_version"),
        "built_at": release["built_at"],
        "n_datasets": db["n_datasets"],
        "n_nodes": db["n_nodes"],
        "datasets": {name: release["datasets"][name] for name in DATASETS},
    }


def analyze(records: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    """Record composition, cross-dataset deleted-gene overlap, metabolite keys, examples."""
    composition: Counter[str] = Counter()
    entry_counts: Counter[str] = Counter()
    entries_per_record: Counter[int] = Counter()
    genes_by_dataset: dict[str, set[str]] = {name: set() for name in DATASETS}
    keys_by_dataset: dict[str, Counter[str]] = {name: Counter() for name in DATASETS}
    perturbations_by_dataset: dict[str, Counter[str]] = {
        name: Counter() for name in DATASETS
    }
    # First record of each composition, the example the fragment dumps.
    examples: dict[str, int] = {}
    for idx in sorted(records):
        entries = records[idx]
        entries_per_record[len(entries)] += 1
        names = sorted(
            {e["experiment"]["dataset_name"] for e in entries}, key=DATASETS.index
        )
        comp = " + ".join(f"`{n}`" for n in names)
        composition[comp] += 1
        per_name = Counter(e["experiment"]["dataset_name"] for e in entries)
        entry_counts[" + ".join(f"{per_name[n]} {SHORT[n]}" for n in names)] += 1
        examples.setdefault(comp, idx)
        for e in entries:
            exp = e["experiment"]
            name = exp["dataset_name"]
            deleted = _deleted_genes(exp)
            assert len(deleted) == 1, f"record {idx}: {len(deleted)} deletions"
            genes_by_dataset[name].add(deleted[0])
            keys_by_dataset[name].update(list(exp["phenotype"]["metabolite_level"]))
            perturbations_by_dataset[name][
                " + ".join(
                    sorted(
                        p["perturbation_type"] for p in exp["genotype"]["perturbations"]
                    )
                )
            ] += 1
    mul, coo, cac = (genes_by_dataset[n] for n in DATASETS)
    cachera = f"`{DATASETS[2]}`"
    cachera_mixed = sum(
        n for comp, n in composition.items() if cachera in comp and comp != cachera
    )
    return {
        "records_mixing_cachera_with_amino_acids": cachera_mixed,
        "records_by_entry_count": {
            str(k): v for k, v in sorted(entries_per_record.items())
        },
        "records_by_dataset_composition": dict(composition.most_common()),
        "records_by_dataset_entry_counts": {
            k: v for k, v in sorted(entry_counts.items(), key=lambda kv: -kv[1])
        },
        "deleted_genes": {
            "per_dataset": {SHORT[n]: len(g) for n, g in genes_by_dataset.items()},
            "mulleder_and_cooper": len(mul & coo),
            "mulleder_and_cachera": len(mul & cac),
            "cooper_and_cachera": len(coo & cac),
            "all_three": len(mul & coo & cac),
            "union": len(mul | coo | cac),
        },
        "metabolite_keys": {
            SHORT[n]: dict(sorted(c.items())) for n, c in keys_by_dataset.items()
        },
        "perturbation_types": {
            SHORT[n]: dict(c.most_common()) for n, c in perturbations_by_dataset.items()
        },
        "example_indices": examples,
    }


def _cut_levels(level: dict[str, Any]) -> str:
    """The first ``LEVELS_SHOWN`` items of a metabolite dict, with the cut counted."""
    items = list(level.items())
    shown = ", ".join(f"{k!r}: {v!r}" for k, v in items[:LEVELS_SHOWN])
    rest = len(items) - LEVELS_SHOWN
    return "{" + shown + (f", ... {rest} more" if rest > 0 else "") + "}"


def dump_processed_record(entries: list[dict[str, Any]]) -> str:
    """Every entry of one processed record, cut to ``ENTRY_FIELDS`` plus the levels."""
    lines = [f"# processed record: a list of {len(entries)} entries"]
    for i, item in enumerate(entries):
        exp = item["experiment"]
        lines.append(f"[{i}] experiment:")
        for p in exp["genotype"]["perturbations"]:
            variant = f", variant={p['variant']!r}" if p.get("variant") else ""
            lines.append(
                f"    genotype.perturbations: {p['perturbation_type']} "
                f"{p['systematic_gene_name']} ({p['perturbed_gene_name']}{variant})"
            )
        for path in ENTRY_FIELDS:
            lines.append(f"    {path} = {_pick(exp, path)!r}")
        ph = exp["phenotype"]
        lines.append(
            f"    phenotype.metabolite_level = {_cut_levels(ph['metabolite_level'])}"
        )
        lines.append(f"    phenotype.n_replicates = {_cut_levels(ph['n_replicates'])}")
        lines.append("    # ... every other field cut")
    return "\n".join(lines)


def main() -> None:
    """Query, aggregate, count, dump, write the JSON and the fragment."""
    load_dotenv()
    data_root = os.getenv("DATA_ROOT")
    assert data_root is not None, "DATA_ROOT must be set in .env"
    job_id = os.getenv("SLURM_JOB_ID", "none")
    started = datetime.now(UTC).isoformat(timespec="seconds")

    query = QUERY_PATH.read_text()
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    print(f"gene_set: {len(genome.gene_set)} genes")

    dataset_root = osp.join(data_root, DATASET_SUBDIR)
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=None,
        incidence_graphs=None,
        node_embeddings=None,
        converter=None,
        deduplicator=None,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )
    n_processed = len(dataset)
    print(f"dataset length: {n_processed}")

    stages = {
        stage: stage_counts(osp.join(dataset_root, stage, "lmdb"))
        for stage in ("raw", "processed")
    }
    records = processed_records(osp.join(dataset.processed_dir, "lmdb"))
    assert len(records) == n_processed
    analysis = analyze(records)
    releases = releases_status()
    dataset.close_lmdb()

    examples = {
        comp: {"index": idx, "dump": dump_processed_record(records[idx])}
        for comp, idx in analysis.pop("example_indices").items()
    }
    summary: dict[str, Any] = {
        "script": SCRIPT,
        "slurm_job_id": job_id,
        "started_utc": started,
        "query_path": "torchcell/knowledge_graphs/queries/amino_acid_betaxanthin.cql",
        "dataset_root": dataset_root,
        "pipeline": "(no converter) -> (no deduplicator) -> GenotypeAggregator",
        "gene_set_size": len(genome.gene_set),
        "stage_entries": stages,
        "processed_records": n_processed,
        **analysis,
        "example_records": examples,
        "served_release": served_release(releases),
        "releases_status": releases,
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ("releases_status", "example_records")
            },
            indent=2,
        )
    )
    write_fragment(summary)
    print("finished")


def _release_lines(summary: dict[str, Any]) -> list[str]:
    """Lines naming the served release and the three datasets' content hashes."""
    rel = summary["served_release"]
    if rel is None:
        return [
            "`python -m torchcell.knowledge_graphs.releases status --json` exited "
            f"{summary['releases_status']['returncode']} inside the job; its stderr is in "
            "`experiments/034-showcase-datasets/results/amino_acid_betaxanthin_query.json`.",
            "",
        ]
    lines = [
        f"Served database `{rel['database']}` (alias `latest`): release `{rel['release']}`, "
        f"version `{rel['version']}`, built {rel['built_at']} from torchcell commit "
        f"`{rel['torchcell_commit'][:8]}`, {rel['n_datasets']} datasets, "
        f"{rel['n_nodes']:,} nodes.",
        "",
        "| dataset | experiments served | content sha256 |",
        "|---|---:|---|",
    ]
    for name, d in rel["datasets"].items():
        lines.append(
            f"| `{name}` | {d['n_experiments']:,} | `{d['content_sha256'][:16]}...` |"
        )
    return lines + [""]


def write_fragment(summary: dict[str, Any]) -> None:
    """The ``query_results.md`` MyST fragment from the summary."""
    FRAGMENT_DIR.mkdir(parents=True, exist_ok=True)
    st = summary["stage_entries"]
    dg = summary["deleted_genes"]
    lines = [
        f"<!-- generated by {SCRIPT}, slurm job {summary['slurm_job_id']}, "
        f"{summary['started_utc']}; do not edit -->",
        "",
        *_release_lines(summary),
        f"Slurm job `{summary['slurm_job_id']}`; `$gene_set` bound to "
        f"`SCerevisiaeGenome.gene_set` ({summary['gene_set_size']:,} genes); pipeline "
        f"{summary['pipeline']}.",
        "",
        "**Entries at each stage** (one entry = one experiment and its reference)",
        "",
        "| dataset_name | query (raw) | in processed records | perturbations per entry |",
        "|---|---|---|---|",
    ]
    for name in DATASETS:
        cells = [
            ", ".join(f"{n:,} `{t}`" for t, n in st[stage].get(name, {}).items()) or "0"
            for stage in ("raw", "processed")
        ]
        perts = "; ".join(
            f"{t} ({n:,})"
            for t, n in summary["perturbation_types"][SHORT[name]].items()
        )
        lines.append(f"| `{name}` | " + " | ".join(cells) + f" | {perts} |")
    lines += [
        "",
        f"`GenotypeAggregator` groups those entries into "
        f"{summary['processed_records']:,} processed records, one per perturbed gene set. "
        "Records by the datasets their entries come from:",
        "",
        "| datasets in the record | records |",
        "|---|---:|",
    ]
    for comp, n in summary["records_by_dataset_composition"].items():
        lines.append(f"| {comp} | {n:,} |")
    lines += [
        "",
        "Records by the number of entries of each dataset they hold:",
        "",
        "| entries in the record | records |",
        "|---|---:|",
    ]
    for comp, n in summary["records_by_dataset_entry_counts"].items():
        lines.append(f"| {comp} | {n:,} |")
    per = dg["per_dataset"]
    lines += [
        "",
        f"Deleted genes returned: Mulleder {per['Mulleder']:,}, Cooper "
        f"{per['Cooper']:,}, Cachera {per['Cachera']:,}; {dg['union']:,} in the union. "
        f"Mulleder and Cooper share {dg['mulleder_and_cooper']:,}, Mulleder and Cachera "
        f"{dg['mulleder_and_cachera']:,}, Cooper and Cachera {dg['cooper_and_cachera']:,}, "
        f"all three {dg['all_three']:,}. Records holding a Cachera entry beside an amino-acid "
        f"entry: {summary['records_mixing_cachera_with_amino_acids']:,}; a Cachera gene set "
        "also holds the four cassette genes, so it never equals a single-deletion gene set.",
        "",
        "Metabolite keys each dataset returns (key: entries carrying it):",
        "",
        "| dataset | keys |",
        "|---|---|",
    ]
    for short, keys in summary["metabolite_keys"].items():
        lines.append(
            f"| {short} | " + ", ".join(f"`{k}` {n:,}" for k, n in keys.items()) + " |"
        )
    for comp, ex in summary["example_records"].items():
        lines += [
            "",
            f"Processed record {ex['index']}, the first record holding {comp}, each "
            "entry's `experiment` cut to the fields shown:",
            "",
            "```text",
            ex["dump"],
            "```",
        ]
    lines.append("")
    (FRAGMENT_DIR / "query_results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
