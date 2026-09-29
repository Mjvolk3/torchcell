# experiments/034-showcase-datasets/scripts/query_essentiality_smf.py
# [[experiments.034-showcase-datasets.scripts.query_essentiality_smf]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/034-showcase-datasets/scripts/query_essentiality_smf
"""Run the supported essentiality + SMF query against the served graph and record it.

Builds a no-merge ``Neo4jCellDataset`` over
``torchcell/knowledge_graphs/queries/essentiality_smf.cql``: ``CompositeFitnessConverter``
turns every SGD essential gene into a fitness-0 record, no deduplicator runs, and
``GenotypeAggregator`` groups every entry of one gene set under one processed record.
This is the build shape ``torchcell.data.label_policy`` is written for: every source
entry survives, and a ``LabelPolicy`` chooses among them at read time.

Writes what the showcase page embeds:

- ``experiments/034-showcase-datasets/results/essentiality_smf_query.json``: records per
  ``dataset_name`` at each stage (raw query, conversion, processed entries), the
  processed record count, how many records carry a ``CONVERTED_ZERO`` entry and with
  which other sources, what ``label_df`` and the default ``LabelPolicy`` pick for them,
  ``label_df['fitness'].describe()``, one processed record, the served release reported
  by ``python -m torchcell.knowledge_graphs.releases status --json``, and the slurm job id;
- ``docs/source/showcase/_generated/essentiality-smf/query_results.md``: the same numbers
  as a MyST fragment.

The served graph is queried only under slurm
(``experiments/034-showcase-datasets/scripts/gh_query_essentiality_smf.slurm``); the
dataset cache lives under ``$DATA_ROOT/data/torchcell/showcase_essentiality_smf``.
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
from torchcell.data.label_policy import CONVERTED_ZERO, LabelPolicy
from torchcell.data.label_table import entries_of_record
from torchcell.data.neo4j_cell import Neo4jCellDataset
from torchcell.datamodels.fitness_composite_conversion import CompositeFitnessConverter
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

SCRIPT = "experiments/034-showcase-datasets/scripts/query_essentiality_smf.py"
EXPERIMENT_DIR = Path("experiments/034-showcase-datasets")
RESULTS_JSON = EXPERIMENT_DIR / "results" / "essentiality_smf_query.json"
FRAGMENT_DIR = Path("docs/source/showcase/_generated/essentiality-smf")
QUERY_PATH = (
    Path(torchcell.__file__).parent
    / "knowledge_graphs"
    / "queries"
    / "essentiality_smf.cql"
)
DATASET_SUBDIR = "data/torchcell/showcase_essentiality_smf"
DATASETS = ("GeneEssentialitySgdDataset", "SmfCostanzo2016Dataset")

# The processed-record dump keeps these paths of each entry's experiment dict.
ENTRY_FIELDS = (
    "dataset_name",
    "experiment_type",
    "genotype.perturbations[0].systematic_gene_name",
    "genotype.perturbations[0].perturbation_type",
    "genotype.perturbations[0].strain_id",
    "environment.temperature.value",
    "phenotype.fitness",
    "phenotype.fitness_std",
)


def _pick(record: dict[str, Any], path: str) -> Any:
    """Value at a dotted path with ``[i]`` list indices, e.g. ``a.b[0].c``."""
    node: Any = record
    for part in path.split("."):
        name, _, index = part.partition("[")
        node = node[name]
        if index:
            node = node[int(index.rstrip("]"))]
    return node


def stage_counts(lmdb_dir: str) -> dict[str, dict[str, int]]:
    """Entries per ``dataset_name`` and ``experiment_type`` in one pipeline-stage LMDB.

    Raw and conversion stages store one JSON dict per key; the processed stage stores a
    JSON list of entries per key. Both are counted per entry.
    """
    counts: dict[str, dict[str, int]] = {}
    env = lmdb.open(lmdb_dir, readonly=True, lock=False)
    with env.begin() as txn:
        for _, value in txn.cursor():
            payload = json.loads(value.decode())
            entries = payload if isinstance(payload, list) else [payload]
            for entry in entries:
                e = entry["experiment"]
                by_type = counts.setdefault(e["dataset_name"], {})
                by_type[e["experiment_type"]] = by_type.get(e["experiment_type"], 0) + 1
    env.close()
    return counts


def processed_entries(processed_lmdb: str) -> dict[int, bytes]:
    """Every processed record's raw JSON bytes, keyed by index."""
    out: dict[int, bytes] = {}
    env = lmdb.open(processed_lmdb, readonly=True, lock=False)
    with env.begin() as txn:
        for key, value in txn.cursor():
            out[int(key.decode())] = bytes(value)
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
    """The ``latest``-aliased database's row of ``releases status``, cut to what the page shows."""
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


def analyze(records: dict[int, bytes], label_df: Any) -> dict[str, Any]:
    """Per-record sources, what ``label_df`` holds, and what the default policy picks."""
    policy = LabelPolicy(name="default")
    fitness_by_index = dict(
        zip(label_df["index"].astype(int), label_df["fitness"].astype(float))
    )
    entries_per_record: Counter[int] = Counter()
    composition: Counter[str] = Counter()
    policy_source: Counter[str] = Counter()
    with_zero = 0
    zero_alone = 0
    zero_with_measurement = 0
    label_df_zero_with_measurement = 0
    policy_refused_zero = 0
    # The example is the shortest record in which label_df kept the converted 0 over a
    # measured Costanzo fitness: the case the label policy exists for.
    example: tuple[int, int] | None = None
    for idx in sorted(records):
        entries = entries_of_record(records[idx])
        entries_per_record[len(entries)] += 1
        sources = sorted({e.source for e in entries})
        composition[" + ".join(sources)] += 1
        choice = policy.select(entries, "fitness")
        assert choice is not None, f"record {idx} has no admissible fitness"
        policy_source[choice.source] += 1
        if CONVERTED_ZERO not in sources:
            continue
        with_zero += 1
        if len(sources) == 1:
            zero_alone += 1
            continue
        zero_with_measurement += 1
        if choice.source != CONVERTED_ZERO:
            policy_refused_zero += 1
        if fitness_by_index[idx] == 0.0:
            label_df_zero_with_measurement += 1
            if example is None or len(entries) < example[0]:
                example = (len(entries), idx)
    assert example is not None, "label_df never kept a converted 0 over a measurement"
    example_index = example[1]
    return {
        "records_by_entry_count": {
            str(k): v for k, v in sorted(entries_per_record.items())
        },
        "records_by_source_composition": dict(composition.most_common()),
        "records_with_converted_zero": with_zero,
        "converted_zero_only": zero_alone,
        "converted_zero_with_measurement": zero_with_measurement,
        "label_df_fitness_zero_despite_measurement": label_df_zero_with_measurement,
        "default_policy_refused_converted_zero": policy_refused_zero,
        "default_policy_chosen_source": dict(policy_source.most_common()),
        "default_policy_id": policy.policy_id,
        "example_index": example_index,
    }


def dump_processed_record(raw: bytes, label_fitness: float) -> str:
    """Every entry of one processed record, cut to ``ENTRY_FIELDS``, plus the policy pick."""
    entries = json.loads(raw)
    lines = [f"# processed record: a list of {len(entries)} entries"]
    for i, item in enumerate(entries):
        lines.append(f"[{i}] experiment:")
        for path in ENTRY_FIELDS:
            lines.append(f"    {path} = {_pick(item['experiment'], path)!r}")
        lines.append("    # ... every other field cut")
    choice = LabelPolicy(name="default").select(entries_of_record(raw), "fitness")
    assert choice is not None
    lines.append(f"label_df['fitness'] = {label_fitness!r}")
    lines.append(
        f"LabelPolicy(name='default').select(entries, 'fitness') = "
        f"{choice.value!r} from {choice.source!r} "
        f"({choice.n_entries_combined} of {choice.n_entries_available} entries combined)"
    )
    return "\n".join(lines)


def main() -> None:
    """Query, convert, aggregate, count, dump, write the JSON and the fragment."""
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
        converter=CompositeFitnessConverter,
        deduplicator=None,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )
    n_processed = len(dataset)
    print(f"dataset length: {n_processed}")

    stages = {
        stage: stage_counts(osp.join(dataset_root, stage, "lmdb"))
        for stage in ("raw", "conversion", "processed")
    }
    label_df = dataset.label_df
    records = processed_entries(osp.join(dataset.processed_dir, "lmdb"))
    assert len(records) == n_processed
    analysis = analyze(records, label_df)
    example = analysis["example_index"]
    label_fitness = float(label_df.loc[label_df["index"] == example, "fitness"].iloc[0])
    describe = {k: float(v) for k, v in label_df["fitness"].describe().items()}
    releases = releases_status()
    dataset.close_lmdb()

    summary: dict[str, Any] = {
        "script": SCRIPT,
        "slurm_job_id": job_id,
        "started_utc": started,
        "query_path": "torchcell/knowledge_graphs/queries/essentiality_smf.cql",
        "dataset_root": dataset_root,
        "pipeline": "CompositeFitnessConverter -> (no deduplicator) -> GenotypeAggregator",
        "gene_set_size": len(genome.gene_set),
        "stage_entries": stages,
        "processed_records": n_processed,
        "label_df_fitness_describe": describe,
        **{k: v for k, v in analysis.items() if k != "example_index"},
        "example_record": {
            "index": example,
            "dump": dump_processed_record(records[example], label_fitness),
        },
        "served_release": served_release(releases),
        "releases_status": releases,
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "releases_status"}, indent=2
        )
    )
    write_fragment(summary)
    print("finished")


def _release_line(summary: dict[str, Any]) -> list[str]:
    """Lines naming the served release and the two datasets' content hashes."""
    rel = summary["served_release"]
    if rel is None:
        return [
            "`python -m torchcell.knowledge_graphs.releases status --json` exited "
            f"{summary['releases_status']['returncode']} inside the job; its stderr is in "
            "`experiments/034-showcase-datasets/results/essentiality_smf_query.json`.",
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
    lines = [
        f"<!-- generated by {SCRIPT}, slurm job {summary['slurm_job_id']}, "
        f"{summary['started_utc']}; do not edit -->",
        "",
        *_release_line(summary),
        f"Slurm job `{summary['slurm_job_id']}`; `$gene_set` bound to "
        f"`SCerevisiaeGenome.gene_set` ({summary['gene_set_size']:,} genes); pipeline "
        f"{summary['pipeline']}.",
        "",
        "**Entries at each stage** (one entry = one experiment and its reference)",
        "",
        "| dataset_name | query (raw) | after conversion | in processed records |",
        "|---|---|---|---|",
    ]
    for name in DATASETS:
        cells = [
            ", ".join(f"{n:,} `{t}`" for t, n in st[stage].get(name, {}).items()) or "0"
            for stage in ("raw", "conversion", "processed")
        ]
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"`GenotypeAggregator` groups those entries into "
        f"{summary['processed_records']:,} processed records, one per gene set. "
        "Records by the sources their entries come from (`source_key`):",
        "",
        "| sources in the record | records |",
        "|---|---:|",
    ]
    for comp, n in summary["records_by_source_composition"].items():
        lines.append(f"| {comp} | {n:,} |")
    lines += [
        "",
        f"{summary['records_with_converted_zero']:,} records carry a `{CONVERTED_ZERO}` "
        f"entry: {summary['converted_zero_only']:,} have nothing else and "
        f"{summary['converted_zero_with_measurement']:,} also carry a measured Costanzo "
        "fitness. `label_df` keeps the last non-missing entry of each record, which is 0 "
        f"for {summary['label_df_fitness_zero_despite_measurement']:,} of those "
        f"{summary['converted_zero_with_measurement']:,}. The default `LabelPolicy` "
        f"(`policy_id` `{summary['default_policy_id']}`) refuses the converted 0 in "
        f"{summary['default_policy_refused_converted_zero']:,} of them and takes it only "
        "where it is the sole entry. Source the default policy picks, over every record:",
        "",
        "| chosen source | records |",
        "|---|---:|",
    ]
    for src, n in summary["default_policy_chosen_source"].items():
        lines.append(f"| `{src}` | {n:,} |")
    lines += [
        "",
        "`label_df['fitness'].describe()` over every processed record:",
        "",
        "| statistic | value |",
        "|---|---:|",
    ]
    for k, v in summary["label_df_fitness_describe"].items():
        lines.append(f"| {k} | {int(v):,} |" if k == "count" else f"| {k} | {v:.4f} |")
    ex = summary["example_record"]
    lines += [
        "",
        f"Processed record {ex['index']}, the shortest record in which `label_df` kept "
        "the converted 0 over a measured Costanzo fitness, each entry's `experiment` cut "
        "to the fields shown:",
        "",
        "```text",
        ex["dump"],
        "```",
        "",
    ]
    (FRAGMENT_DIR / "query_results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
