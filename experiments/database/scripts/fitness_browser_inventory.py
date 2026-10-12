#!/usr/bin/env python
# experiments/database/scripts/fitness_browser_inventory.py
# [[experiments.database.scripts.fitness_browser_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/fitness_browser_inventory.py
r"""G3 inventory of the Fitness Browser deposit, read from the bytes.

Opens the pinned ``feba.db`` (``torchcell.datasets.fitness_browser.FEBA``) read-only and
reports, per organism and per experiment: the experiment's publication (``pubId`` and the
``Publication`` row), set, description, group, conditions, medium, temperature, library,
generations, the number of loci with both a fitness and a t value, whether Price 2018
Supplementary Table 5 lists the sample (joined on ``orgId`` and the experiment name), and
which served compendium or own-release store keeps it, by that loader's own sample rule:

- Keio: ``RbTnseqPrice2018EcoliDataset`` (``price2018.classify_samples`` over Table S5)
  and ``RbTnseqWetmore2015Dataset`` (``wetmore2015.classify_samples`` over Data Set S1);
- Putida: ``RbTnseqBorchert2024Dataset`` (``borchert2024.drop_reason`` over its release).

The served sets are computed from the pinned release bytes through the loaders' own
functions, never from a dev store (the stores can be mid-rebuild). Nothing is written to
any mirror or store.

Run from the repo root (the sha256 check reads all 10.2 GB once)::

    python experiments/database/scripts/fitness_browser_inventory.py

Writes ``experiments/database/results/fitness_browser_inventory.json`` and the markdown
tables of [[torchcell.candidates.aggregation-decomposition]]'s 2026.10.12 section to
``experiments/database/results/fitness_browser_inventory_tables.md``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd
from pydantic import BaseModel, ConfigDict

from torchcell.data import file_sha256
from torchcell.datasets import fitness_browser as fb
from torchcell.datasets.ecoli import price2018
from torchcell.datasets.ecoli import wetmore2015 as wetmore
from torchcell.datasets.pputida import borchert2024 as b24

RESULTS: Final = Path("experiments/database/results")
JSON_OUT: Final = RESULTS / "fitness_browser_inventory.json"
TABLES_OUT: Final = RESULTS / "fitness_browser_inventory_tables.md"
NO_PUB: Final = "no-pubId"


class ExperimentRow(BaseModel):
    """One browser experiment as the inventory reports it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    org_id: str
    exp_name: str
    pub_id: str | None
    set_name: str
    exp_desc: str
    exp_group: str
    conditions: tuple[fb.FbCondition, ...]
    media: str
    media_strength: float
    temperature_c: float | None
    mutant_library: str
    date_started: str
    n_generations: float | None
    n_loci_with_fit_and_t: int
    in_price2018_table_s5: bool
    in_borchert2024_release: bool
    served_by: tuple[str, ...]


class OrganismSummary(BaseModel):
    """One organism's counts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    org_id: str
    organism: str
    n_experiments: int
    n_publications: int
    n_no_pub_id: int
    experiments_by_pub_id: dict[str, int]
    n_loci_with_values: int
    loci_per_experiment: tuple[int, ...]
    n_in_price2018_table_s5: int
    n_served: int


class Inventory(BaseModel):
    """The whole inventory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    script: str
    measured_at: str
    deposit: fb.DepositPin
    sha256_verified_this_run: bool
    publications: dict[str, fb.FbPublication]
    n_organisms: int
    n_experiments: int
    n_publications_cited: int
    organisms: tuple[OrganismSummary, ...]
    experiments: tuple[ExperimentRow, ...]
    table_s5_not_in_browser: dict[str, list[str]]


def table_s5_names(data_root: str) -> set[tuple[str, str]]:
    """``(orgId, name)`` of every Price 2018 Supplementary Table 5 row, all organisms."""
    path = price2018.source_path("si3.xlsx", data_root)
    sheet = pd.read_excel(path, sheet_name=price2018.TABLE_S5_SHEET, header=None)
    header_rows = sheet.index[sheet[0] == "orgId"]
    if len(header_rows) != 1:
        raise ValueError("Table S5: expected one 'orgId' header row")
    start = int(header_rows[0])
    table = sheet.iloc[start + 1 :].copy()
    table.columns = sheet.iloc[start].tolist()
    return {
        (str(o), str(n)) for o, n in zip(table["orgId"], table["name"], strict=True)
    }


def served_keio(data_root: str) -> dict[str, set[str]]:
    """Keio sample names each Keio store keeps, by its own classification."""
    table_s5 = wetmore.read_superset_experiments(
        price2018.source_path("si3.xlsx", data_root)
    )
    carried = frozenset(wetmore.subsumption_record(data_root).carried)
    price = {
        s.name
        for s in price2018.classify_samples(table_s5, carried)
        if s.drop_rule is None
    }
    s1 = wetmore.loader_source_path("si1.xlsx", data_root)
    wet = {
        s.name
        for s in wetmore.classify_samples(
            pd.read_excel(s1, sheet_name=wetmore.DATA_SET_S1_EXPERIMENTS_SHEET),
            pd.read_excel(s1, sheet_name=wetmore.DATA_SET_S1_QUALITY_SHEET),
        )
        if s.drop_rule is None
    }
    return {"RbTnseqPrice2018EcoliDataset": price, "RbTnseqWetmore2015Dataset": wet}


def borchert_release(data_root: str) -> tuple[set[str], set[str]]:
    """(every Borchert 2024 sample, the kept ones), off its pinned release."""
    path = b24.raw_mirror_dir(data_root) / b24.DATA_RELPATH
    if file_sha256(path) != b24.DATA_SHA256:
        raise SystemExit(f"{path}: sha256 differs from borchert2024.DATA_SHA256")
    release = b24.read_release(path)
    names = {s.exp_name for s in release.samples}
    kept = {s.exp_name for s in release.samples if b24.drop_reason(s) is None}
    return names, kept


def build_inventory(
    feba: fb.Feba,
    *,
    verified: bool,
    s5: set[tuple[str, str]],
    served: Mapping[str, Mapping[str, set[str]]],
    borchert_names: set[str],
) -> Inventory:
    """Every organism and experiment of the deposit, with its membership columns."""
    publications = feba.publications()
    rows: list[ExperimentRow] = []
    summaries: list[OrganismSummary] = []
    for organism in feba.organisms():
        experiments = feba.experiments(organism.org_id)
        counts = feba.gene_counts(organism.org_id)
        org_served = served.get(organism.org_id, {})
        org_rows = [
            ExperimentRow(
                org_id=e.org_id,
                exp_name=e.exp_name,
                pub_id=e.pub_id,
                set_name=e.set_name,
                exp_desc=e.exp_desc,
                exp_group=e.exp_group,
                conditions=e.conditions,
                media=e.media,
                media_strength=e.media_strength,
                temperature_c=e.temperature_c,
                mutant_library=e.mutant_library,
                date_started=e.date_started,
                n_generations=e.n_generations,
                n_loci_with_fit_and_t=counts.get(e.exp_name, 0),
                in_price2018_table_s5=(e.org_id, e.exp_name) in s5,
                in_borchert2024_release=(
                    e.org_id == "Putida" and e.exp_name in borchert_names
                ),
                served_by=tuple(
                    sorted(
                        cls for cls, names in org_served.items() if e.exp_name in names
                    )
                ),
            )
            for e in experiments
        ]
        rows.extend(org_rows)
        by_pub = Counter(r.pub_id or NO_PUB for r in org_rows)
        summaries.append(
            OrganismSummary(
                org_id=organism.org_id,
                organism=f"{organism.genus} {organism.species} {organism.strain}",
                n_experiments=len(org_rows),
                n_publications=len({r.pub_id for r in org_rows if r.pub_id}),
                n_no_pub_id=by_pub.get(NO_PUB, 0),
                experiments_by_pub_id=dict(sorted(by_pub.items())),
                n_loci_with_values=max(counts.values(), default=0),
                loci_per_experiment=tuple(sorted(set(counts.values()))),
                n_in_price2018_table_s5=sum(r.in_price2018_table_s5 for r in org_rows),
                n_served=sum(1 for r in org_rows if r.served_by),
            )
        )
    in_browser = {(r.org_id, r.exp_name) for r in rows}
    missing: dict[str, list[str]] = {}
    for org, name in sorted(s5 - in_browser):
        missing.setdefault(org, []).append(name)
    cited = {r.pub_id for r in rows if r.pub_id}
    return Inventory(
        script="experiments/database/scripts/fitness_browser_inventory.py",
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        deposit=fb.FEBA,
        sha256_verified_this_run=verified,
        publications=publications,
        n_organisms=len(summaries),
        n_experiments=len(rows),
        n_publications_cited=len(cited),
        organisms=tuple(summaries),
        experiments=tuple(rows),
        table_s5_not_in_browser=missing,
    )


def _by_publication(
    rows: Sequence[ExperimentRow], columns: Sequence[tuple[str, Any]]
) -> list[str]:
    """Markdown rows: one per pubId, the count, then one count per column predicate."""
    groups: dict[str, list[ExperimentRow]] = {}
    for r in rows:
        groups.setdefault(r.pub_id or NO_PUB, []).append(r)
    out = []
    for pub, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        cells = [str(sum(1 for r in members if pred(r))) for _, pred in columns]
        media = Counter(r.media for r in members)
        out.append(
            f"| {pub} | {len(members)} | "
            + " | ".join(cells)
            + " | "
            + "; ".join(f"{m} {n}" for m, n in media.most_common())
            + " |"
        )
    return out


def markdown_tables(inv: Inventory) -> str:
    """The note's tables: every organism, then Keio and Putida by publication."""
    lines = [
        f"Deposit: `{inv.deposit.relpath}`, sha256 `{inv.deposit.sha256}` "
        f"(verified this run: {inv.sha256_verified_this_run}), "
        f"{inv.n_organisms} organisms, {inv.n_experiments:,} experiments, "
        f"{inv.n_publications_cited} publications cited by at least one experiment "
        f"({len(inv.publications)} rows in `Publication`).",
        "",
        "| orgId | organism | experiments | publications | no pubId | loci with values | "
        "in Price 2018 Table S5 | served by a torchcell store |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for o in inv.organisms:
        lines.append(
            f"| {o.org_id} | {o.organism} | {o.n_experiments} | {o.n_publications} | "
            f"{o.n_no_pub_id} | {o.n_loci_with_values} | {o.n_in_price2018_table_s5} | "
            f"{o.n_served} |"
        )
    total_s5 = sum(o.n_in_price2018_table_s5 for o in inv.organisms)
    lines += [
        f"| **total** | | {inv.n_experiments} | | "
        f"{sum(o.n_no_pub_id for o in inv.organisms)} | | {total_s5} | "
        f"{sum(o.n_served for o in inv.organisms)} |",
        "",
        "Price 2018 Table S5 rows absent from the browser, per organism: "
        + (
            "; ".join(f"{k} {len(v)}" for k, v in inv.table_s5_not_in_browser.items())
            or "none"
        )
        + ".",
        "",
    ]
    keio = [r for r in inv.experiments if r.org_id == "Keio"]
    lines += [
        "**Keio (E. coli BW25113)**, by browser publication:",
        "",
        "| pubId | experiments | in Table S5 | kept by RbTnseqPrice2018EcoliDataset | "
        "kept by RbTnseqWetmore2015Dataset | media |",
        "|---|---|---|---|---|---|",
        *_by_publication(
            keio,
            [
                ("s5", lambda r: r.in_price2018_table_s5),
                ("price", lambda r: "RbTnseqPrice2018EcoliDataset" in r.served_by),
                ("wetmore", lambda r: "RbTnseqWetmore2015Dataset" in r.served_by),
            ],
        ),
        "",
    ]
    putida = [r for r in inv.experiments if r.org_id == "Putida"]
    lines += [
        "**Putida (P. putida KT2440)**, by browser publication:",
        "",
        "| pubId | experiments | in Table S5 | in the Borchert 2024 release | "
        "kept by RbTnseqBorchert2024Dataset | media |",
        "|---|---|---|---|---|---|",
        *_by_publication(
            putida,
            [
                ("s5", lambda r: r.in_price2018_table_s5),
                ("b24", lambda r: r.in_borchert2024_release),
                ("b24 kept", lambda r: "RbTnseqBorchert2024Dataset" in r.served_by),
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Measure, write the JSON and the tables, print the totals."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the 10.2 GB sha256 (the size is still checked); the JSON says so",
    )
    args = parser.parse_args(argv)
    root = fb.data_root(args.data_root)
    path = fb.feba_path(root)
    if args.no_verify and path.stat().st_size != fb.FEBA.n_bytes:
        raise SystemExit(f"{path}: size differs from the pin")
    feba = fb.Feba(path, verify=not args.no_verify)
    borchert_names, borchert_kept = borchert_release(root)
    inventory = build_inventory(
        feba,
        verified=not args.no_verify,
        s5=table_s5_names(root),
        served={
            "Keio": served_keio(root),
            "Putida": {"RbTnseqBorchert2024Dataset": borchert_kept},
        },
        borchert_names=borchert_names,
    )
    feba.close()
    RESULTS.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(inventory.model_dump_json(indent=1))
    TABLES_OUT.write_text(markdown_tables(inventory))
    print(
        json.dumps(
            {
                "organisms": inventory.n_organisms,
                "experiments": inventory.n_experiments,
                "publications_cited": inventory.n_publications_cited,
                "in_table_s5": sum(
                    o.n_in_price2018_table_s5 for o in inventory.organisms
                ),
                "json": str(JSON_OUT),
                "tables": str(TABLES_OUT),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
