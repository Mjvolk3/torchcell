# experiments/034-showcase-datasets/scripts/served_dataset_counts.py
# [[experiments.034-showcase-datasets.scripts.served_dataset_counts]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/034-showcase-datasets/scripts/served_dataset_counts
"""The size of the collection on offer, from the newest committed release snapshot.

Reads the newest ``database/releases/<release>.json`` (the snapshot the KG build wrote,
``torchcell.knowledge_graphs.release_snapshot``) and writes
``docs/source/datasets/_generated/served_counts.md``: one line naming the release, the
number of datasets it serves and the total experiments, the organisms it covers, then
one table per organism (in ``Organism`` order) with one row per dataset (experiments
served, the dataset page that shows it, when one exists). The page column comes from the
supported-query registry (``docs_page`` of each query and the datasets its Cypher
selects), so a new dataset page appears here once it is registered.

The organisms are the release's own record when it has one: a snapshot written since the
organism switch (2026.10.12) carries ``organisms`` and a per-dataset ``organism``, and a
yeast-only release then reads as yeast-only here. A snapshot from before the switch
records neither, and the fragment says so and takes each dataset's organism from its
loader class (``organism``, the attribute the build selects by). No number on the
fragment is typed by hand. Run from the repo root::

    python experiments/034-showcase-datasets/scripts/served_dataset_counts.py
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from torchcell.data.experiment_dataset import Organism, dataset_organism
from torchcell.database.build_dataset_lmdb import resolve_dataset_class

REPO = Path(__file__).resolve().parents[3]
RELEASES = REPO / "database" / "releases"
REGISTRY = (
    REPO / "torchcell" / "knowledge_graphs" / "supported_queries" / "registry.json"
)
QUERIES = REPO / "torchcell" / "knowledge_graphs"
FRAGMENT = REPO / "docs" / "source" / "datasets" / "_generated" / "served_counts.md"
SCRIPT = "experiments/034-showcase-datasets/scripts/served_dataset_counts.py"
DATASET_ID = re.compile(r"dataset\.id = '([A-Za-z0-9_]+)'")


def newest_snapshot() -> tuple[str, dict[str, object]]:
    """The release snapshot with the latest ``built_at``."""
    candidates = [
        p for p in RELEASES.glob("*.json") if not p.name.endswith(".closures.json")
    ]
    if not candidates:
        raise FileNotFoundError(f"no release snapshot under {RELEASES}")
    loaded = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in candidates}
    name = max(loaded, key=lambda n: str(loaded[n]["built_at"]))
    return name, loaded[name]


def pages_by_dataset() -> dict[str, str]:
    """Dataset class -> docs page (relative to ``docs/source/datasets``), from the registry."""
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    pages: dict[str, str] = {}
    for query in registry["queries"]:
        page = query.get("docs_page")
        if not page:
            continue
        rel = Path(page).relative_to("docs/source/datasets").with_suffix("")
        cql = (QUERIES / query["cql_path"]).read_text(encoding="utf-8")
        for dataset in DATASET_ID.findall(cql):
            pages[dataset] = rel.as_posix()
    return pages


def dataset_organisms(snapshot: dict[str, object]) -> tuple[dict[str, str], str]:
    """Dataset -> organism, and the sentence saying where the organisms came from.

    A snapshot that records ``organisms`` must record every dataset's ``organism`` (a
    missing one is a KeyError, never filled in). One that records no set predates the
    switch, and each dataset's organism is the one its loader class declares.
    """
    datasets = snapshot["datasets"]
    assert isinstance(datasets, dict)
    recorded = snapshot.get("organisms")
    if recorded is not None:
        assert isinstance(recorded, list)
        organisms = {name: str(entry["organism"]) for name, entry in datasets.items()}
        outside = sorted(n for n, o in organisms.items() if o not in recorded)
        assert not outside, f"datasets outside the recorded organism set: {outside}"
        return organisms, (
            f"The release was built for the organism set `{','.join(recorded)}` and "
            "records each dataset's organism."
        )
    organisms = {
        name: dataset_organism(resolve_dataset_class(name)).value for name in datasets
    }
    return organisms, (
        "The release predates the organism switch and records no organism set; each "
        "dataset's organism below is the one its loader class declares."
    )


def fragment(name: str, snapshot: dict[str, object], pages: dict[str, str]) -> str:
    """The served_counts.md text."""
    datasets = snapshot["datasets"]
    assert isinstance(datasets, dict)
    organisms, provenance = dataset_organisms(snapshot)
    rows = sorted(datasets.items(), key=lambda kv: -int(kv[1]["n_experiments"]))
    total = sum(int(v["n_experiments"]) for _, v in rows)
    n_nodes = snapshot["n_nodes"]
    assert isinstance(n_nodes, int)
    groups = [
        (organism.value, [(d, e) for d, e in rows if organisms[d] == organism.value])
        for organism in Organism
    ]
    groups = [(organism, members) for organism, members in groups if members]
    covered = "; ".join(
        f"`{organism}` {len(members)} datasets, "
        f"{sum(int(e['n_experiments']) for _, e in members):,} experiments"
        for organism, members in groups
    )
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    lines = [
        f"<!-- generated by {SCRIPT} from database/releases/{name} at {stamp}; do not edit -->",
        "",
        f"Release `{snapshot['release']}` (knowledge-graph version {snapshot['version']}, built "
        f"{str(snapshot['built_at'])[:10]} from torchcell {snapshot['torchcell_version']}) serves "
        f"**{len(rows)} datasets** and **{total:,} experiments** over "
        f"{n_nodes:,} graph nodes, by organism: {covered}. {provenance} One experiment is "
        "one genotype, environment and phenotype record with its reference; the guide's "
        "supported-datasets table gives each dataset's genotype count, environments, "
        "phenotype shape and graph role.",
    ]
    for organism, members in groups:
        subtotal = sum(int(e["n_experiments"]) for _, e in members)
        lines += [
            "",
            f"**{organism}**",
            "",
            "| dataset | experiments served | page |",
            "|---|---:|---|",
        ]
        for dataset, entry in members:
            page = pages.get(dataset)
            link = f"{{doc}}`{page}`" if page else ""
            lines.append(f"| `{dataset}` | {int(entry['n_experiments']):,} | {link} |")
        lines.append(f"| **{organism} total** | **{subtotal:,}** | |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write the fragment and print the totals."""
    name, snapshot = newest_snapshot()
    text = fragment(name, snapshot, pages_by_dataset())
    FRAGMENT.parent.mkdir(parents=True, exist_ok=True)
    FRAGMENT.write_text(text, encoding="utf-8")
    print(FRAGMENT.relative_to(REPO))
    print(text.splitlines()[2])


if __name__ == "__main__":
    main()
