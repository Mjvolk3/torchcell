# experiments/036-dataset-fixes-before-kg-build/scripts/yeast_gene_id_currency.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.yeast_gene_id_currency]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/yeast_gene_id_currency
"""Classify every released Costanzo 2016 / Kuzmin 2020 ORF R64-4-1 no longer carries.

Issues #893, #930, #979 (Costanzo 2016) and #897-#900 (Kuzmin 2020). Reads the releases
from the dev raw directories, takes every systematic name the way each loader derives it
(the strain id up to its first ``_``, or the SMF ``Systematic gene name`` column), and
resolves each against the pinned R64-4-1 genome (``SCerevisiaeGenome`` on the genomes
tier; no SGD call). For every name that is not a live locus it records:

- the R64 resolution (status, locus, the locus's feature type and ``orf_classification``
  from its GFF row);
- the decision of ``torchcell.datasets.systematic_name_currency`` (renamed, merged or
  retired) and that the genome agrees with it, and that the measured set of non-current
  names equals the set the tables decide (no name undecided, no table entry unused);
- the release's own evidence: the strain ids naming the released ORF and the strain ids
  naming its current locus, in each release (a merged ORF sits beside a separate
  deletion strain of its locus; a renamed ORF's strain number recurs under the current
  name in Costanzo 2016);
- rows per store naming it in a checked column, the rows each store refuses, and the
  Kuzmin 2020 rows whose stored common name gains the prime of ``imp2'``.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/yeast_gene_id_currency.json``.
``--table`` renders that file as the markdown tables of the dataset notes. Run under
slurm (the four Data File S1 tables are 2 GB)::

    sbatch experiments/036-dataset-fixes-before-kg-build/scripts/gh_yeast_gene_id_currency.slurm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import urllib.parse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from torchcell.datasets.gene_alias_resolution import GeneNameRefusalReason
from torchcell.datasets.scerevisiae import costanzo2016 as costanzo
from torchcell.datasets.scerevisiae import kuzmin2020 as kuzmin
from torchcell.datasets.systematic_name_currency import (
    SGA_SYSTEMATIC_REFUSALS,
    SGA_SYSTEMATIC_RENAMES,
    sga_systematic_name_currency,
)
from torchcell.sequence.genome.registry import SGD_S288C_R64, resolve
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

EXPERIMENT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS = osp.join(EXPERIMENT_DIR, "results", "yeast_gene_id_currency.json")
GFF_NAME = "saccharomyces_cerevisiae_R64-4-1_20230830.gff"
SGA_FILES = ["SGA_DAmP.txt", "SGA_ExE.txt", "SGA_ExN_NxE.txt", "SGA_NxN.txt"]
KUZMIN_TABLES = ["aaz5667-Table-S1.xlsx", "aaz5667-Table-S3.xlsx"]
KUZMIN_S5 = "aaz5667-Table-S5.xlsx"
STORES = [
    "smf_costanzo2016",
    "dmf_costanzo2016",
    "dmi_costanzo2016",
    "smf_kuzmin2020",
    "dmf_kuzmin2020",
    "tmf_kuzmin2020",
    "dmi_kuzmin2020",
    "tmi_kuzmin2020",
]


class StoreCount(BaseModel):
    """What the currency check does to one store's input frame."""

    model_config = ConfigDict(extra="forbid")

    store: str
    n_rows_in: int
    n_rows_refused: int
    n_rows_out: int
    n_rows_renamed: int
    n_rows_prime_restored: int
    rows_by_name: dict[str, int]


class NameRow(BaseModel):
    """One released ORF R64-4-1 does not carry as a live locus."""

    model_config = ConfigDict(extra="forbid")

    released_name: str
    releases: list[str]
    r64_status: str
    r64_locus: str | None
    locus_feature_type: str | None
    locus_orf_classification: str | None
    decision: str
    decision_reason: str | None
    genome_agrees: bool
    strains_of_name: dict[str, list[str]]
    strains_of_locus: dict[str, list[str]]
    rows_by_store: dict[str, int]


class Report(BaseModel):
    """The measured classification and per-store counts."""

    model_config = ConfigDict(extra="forbid")

    annotation: str
    gff_sha256: str
    n_names: int
    n_renamed: int
    n_merged: int
    n_retired: int
    n_per_store_occurrences: int
    tables_cover_exactly_the_measured_names: bool
    names: list[NameRow]
    stores: list[StoreCount]


def _gff_rows(path: str) -> dict[str, tuple[str, dict[str, str]]]:
    """Locus id -> (feature type, decoded attributes) for every GFF row with an ID."""
    rows: dict[str, tuple[str, dict[str, str]]] = {}
    with open(path) as handle:
        for line in handle:
            if line.startswith("##FASTA"):
                break
            fields = line.rstrip("\n").split("\t")
            if line.startswith("#") or len(fields) < 9:
                continue
            attributes = dict(
                part.split("=", 1) for part in fields[8].split(";") if "=" in part
            )
            if "ID" in attributes:
                rows.setdefault(
                    attributes["ID"],
                    (
                        fields[2],
                        {k: urllib.parse.unquote(v) for k, v in attributes.items()},
                    ),
                )
    return rows


def _orf(strain_id: str) -> str:
    """The systematic name a loader reads off a strain id (up to the first ``_``)."""
    return strain_id.split("_")[0]


def _costanzo_sga(raw: str) -> pd.DataFrame:
    """Query and array strain ids of every Data File S1 row, in loader order."""
    frames = [
        pd.read_csv(
            osp.join(raw, name),
            sep="\t",
            usecols=["Query Strain ID", "Array Strain ID"],
        )
        for name in SGA_FILES
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame["Query Systematic Name"] = frame["Query Strain ID"].map(_orf)
    frame["Array Systematic Name"] = frame["Array Strain ID"].map(_orf)
    return frame


def _rows_by_name(
    frame: pd.DataFrame, columns: Sequence[str], names: set[str]
) -> dict[str, int]:
    """Rows naming each of ``names`` in any of ``columns`` (a row counts once per name)."""
    counts: Counter[str] = Counter()
    for name in names:
        carried = pd.Series(False, index=frame.index)
        for column in columns:
            carried |= frame[column] == name
        if carried.any():
            counts[name] = int(carried.sum())
    return dict(sorted(counts.items()))


def _store_count(
    store: str,
    genome: SCerevisiaeGenome,
    parts: Iterable[tuple[pd.DataFrame, Sequence[str]]],
    names: set[str],
    allele_column: str | None,
) -> StoreCount:
    """Apply the loader's currency check to each (frame, columns) part and count."""
    currency = sga_systematic_name_currency(genome, store)
    rows: Counter[str] = Counter()
    n_prime = 0
    for frame, columns in parts:
        rows.update(_rows_by_name(frame, columns, names))
        kept = currency.apply(frame, columns)
        if allele_column is not None:
            n_prime += int(
                kept[allele_column].map(kuzmin._stored_gene_name).eq("imp2'").sum()
            )
    ledger = currency.ledger(store)
    return StoreCount(
        store=store,
        n_rows_in=ledger.n_rows_in,
        n_rows_refused=ledger.n_rows_refused,
        n_rows_out=ledger.n_rows_out,
        n_rows_renamed=sum(o.n_rows for o in ledger.renamed),
        n_rows_prime_restored=n_prime,
        rows_by_name=dict(sorted(rows.items())),
    )


def _strains(ids: Iterable[str]) -> dict[str, list[str]]:
    """Systematic name -> the distinct strain ids naming it."""
    by_name: dict[str, set[str]] = defaultdict(set)
    for strain in ids:
        for part in str(strain).split("+"):
            by_name[_orf(part)].add(part)
    return {name: sorted(strains) for name, strains in by_name.items()}


def measure(data_root: str) -> Report:
    """Run the classification and the per-store counts on the releases."""
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    gff_path = str(resolve(SGD_S288C_R64, GFF_NAME, data_root=data_root))
    gff = _gff_rows(gff_path)
    with open(gff_path, "rb") as handle:
        gff_sha256 = hashlib.sha256(handle.read()).hexdigest()

    torchcell = osp.join(data_root, "data/torchcell")
    smf_c = costanzo.SmfCostanzo2016Dataset.__new__(costanzo.SmfCostanzo2016Dataset)
    smf_c_frame = smf_c.preprocess_raw(
        pd.read_excel(
            osp.join(
                torchcell,
                "smf_costanzo2016/raw/strain_ids_and_single_mutant_fitness.xlsx",
            )
        )
    )
    sga = _costanzo_sga(osp.join(torchcell, "dmf_costanzo2016/raw"))
    k_raw = osp.join(torchcell, "dmf_kuzmin2020/raw")
    s1, s3 = (pd.read_excel(osp.join(k_raw, n), skiprows=1) for n in KUZMIN_TABLES)
    s5 = pd.read_excel(osp.join(k_raw, KUZMIN_S5), skiprows=1)
    k_frames: dict[str, pd.DataFrame] = {}
    for store, cls, tables in (
        ("smf_kuzmin2020", kuzmin.SmfKuzmin2020Dataset, (s5,)),
        ("dmf_kuzmin2020", kuzmin.DmfKuzmin2020Dataset, (s1, s3, s5)),
        ("tmf_kuzmin2020", kuzmin.TmfKuzmin2020Dataset, (s1, s3)),
        ("dmi_kuzmin2020", kuzmin.DmiKuzmin2020Dataset, (s1, s3)),
        ("tmi_kuzmin2020", kuzmin.TmiKuzmin2020Dataset, (s1, s3)),
    ):
        loader = cls.__new__(cls)
        k_frames[store] = loader.preprocess_raw(*(t.copy() for t in tables))

    sga_cols = ["Query Systematic Name", "Array Systematic Name"]
    digenic = ["Query systematic name no ho", "Array systematic name"]
    pair = ["Query systematic name_1", "Query systematic name_2"]
    triple = [*pair, "Array systematic name"]
    dmf_k = k_frames["dmf_kuzmin2020"]
    is_digenic = dmf_k["record_kind"] == kuzmin.RECORD_KIND_DIGENIC_ARRAY_CROSS
    plans: dict[str, list[tuple[pd.DataFrame, Sequence[str]]]] = {
        "smf_costanzo2016": [(smf_c_frame, ["Systematic gene name"])],
        "dmf_costanzo2016": [(sga, sga_cols)],
        "dmi_costanzo2016": [(sga, sga_cols)],
        "smf_kuzmin2020": [(k_frames["smf_kuzmin2020"], ["ORF1"])],
        "dmf_kuzmin2020": [(dmf_k[is_digenic], digenic), (dmf_k[~is_digenic], pair)],
        "tmf_kuzmin2020": [(k_frames["tmf_kuzmin2020"], triple)],
        "dmi_kuzmin2020": [(k_frames["dmi_kuzmin2020"], digenic)],
        "tmi_kuzmin2020": [(k_frames["tmi_kuzmin2020"], triple)],
    }

    def non_current(frame: pd.DataFrame, columns: Sequence[str]) -> set[str]:
        names = set(pd.unique(frame[list(columns)].to_numpy().ravel()).astype(str))
        out = set()
        for name in names:
            resolution = genome.resolve_gene_name(name)
            live = resolution.status.value == "current" or (
                resolution.status.value == "non_gene_feature"
                and resolution.systematic_name == name
            )
            if not live:
                out.add(name)
        return out

    names_by_store = {
        store: set().union(*(non_current(f, c) for f, c in parts))
        for store, parts in plans.items()
    }
    measured = set().union(*names_by_store.values())
    by_release = {
        release: set().union(
            *(names for store, names in names_by_store.items() if release in store)
        )
        for release in ("costanzo2016", "kuzmin2020")
    }
    decided = set(SGA_SYSTEMATIC_RENAMES) | set(SGA_SYSTEMATIC_REFUSALS)

    stores = [
        _store_count(
            store,
            genome,
            plans[store],
            names_by_store[store],
            None
            if store.endswith("costanzo2016") or store == "smf_kuzmin2020"
            else "Array allele name",
        )
        for store in STORES
    ]
    rows_by_store = {s.store: s.rows_by_name for s in stores}

    costanzo_strains = _strains(
        list(smf_c_frame["Strain ID"])
        + list(sga["Query Strain ID"].unique())
        + list(sga["Array Strain ID"].unique())
    )
    kuzmin_strains = _strains(
        list(s1["Query strain ID"])
        + list(s1["Array strain ID"])
        + list(s3["Query strain ID"])
        + list(s3["Array strain ID"])
    )

    def release_strains(name: str | None) -> dict[str, list[str]]:
        if name is None:
            return {}
        return {
            "costanzo2016": costanzo_strains.get(name, []),
            "kuzmin2020": kuzmin_strains.get(name, []),
        }

    rows: list[NameRow] = []
    for name in sorted(measured):
        resolution = genome.resolve_gene_name(name)
        locus = (
            resolution.systematic_name if resolution.systematic_name != name else None
        )
        feature, attributes = gff.get(locus, (None, {})) if locus else (None, {})
        if name in SGA_SYSTEMATIC_RENAMES:
            decision, reason = "renamed", None
            agrees = (
                resolution.status.value == "renamed"
                and locus == SGA_SYSTEMATIC_RENAMES[name].systematic_name
            )
        elif name in SGA_SYSTEMATIC_REFUSALS:
            refusal = SGA_SYSTEMATIC_REFUSALS[name]
            reason = refusal.reason.value
            if refusal.reason is GeneNameRefusalReason.RETIRED_SYSTEMATIC_NAME:
                decision = "retired"
                agrees = resolution.status.value == "retired"
            else:
                decision = "merged"
                agrees = locus == refusal.locus
        else:
            decision, reason, agrees = "undecided", None, False
        rows.append(
            NameRow(
                released_name=name,
                releases=[r for r, names in by_release.items() if name in names],
                r64_status=resolution.status.value,
                r64_locus=locus,
                locus_feature_type=feature,
                locus_orf_classification=attributes.get("orf_classification"),
                decision=decision,
                decision_reason=reason,
                genome_agrees=agrees,
                strains_of_name=release_strains(name),
                strains_of_locus=release_strains(locus),
                rows_by_store={
                    s: n
                    for s, counts in rows_by_store.items()
                    if (n := counts.get(name, 0))
                },
            )
        )
    decisions = Counter(row.decision for row in rows)
    return Report(
        annotation="S288C R64-4-1 (2023-08-30), genomes tier " + SGD_S288C_R64,
        gff_sha256=gff_sha256,
        n_names=len(rows),
        n_renamed=decisions["renamed"],
        n_merged=decisions["merged"],
        n_retired=decisions["retired"],
        n_per_store_occurrences=sum(len(names) for names in by_release.values()),
        tables_cover_exactly_the_measured_names=measured == decided,
        names=rows,
        stores=stores,
    )


def render_tables(report: Mapping[str, Any]) -> str:
    """The per-name and per-store markdown tables of the dataset notes."""
    lines = [
        "| Released ORF | Releases | R64-4-1 | Current locus (type, class) | Decision | "
        "Strains of the ORF / of its locus |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["names"]:
        locus = row["r64_locus"] or "-"
        if row["r64_locus"]:
            locus += (
                f" ({row['locus_feature_type']}, {row['locus_orf_classification']})"
            )
        strains = "; ".join(
            f"{release}: {', '.join(row['strains_of_name'][release]) or '-'} / "
            f"{', '.join(row['strains_of_locus'].get(release, [])) or '-'}"
            for release in ("costanzo2016", "kuzmin2020")
            if row["strains_of_name"][release] or row["strains_of_locus"].get(release)
        )
        lines.append(
            f"| {row['released_name']} | {', '.join(row['releases'])} | "
            f"{row['r64_status']} | {locus} | {row['decision']} | {strains} |"
        )
    lines += [
        "",
        "| Store | Rows in | Refused | Rows out | Renamed rows | imp2' rows | Names |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for store in report["stores"]:
        lines.append(
            f"| {store['store']} | {store['n_rows_in']:,} | {store['n_rows_refused']:,} | "
            f"{store['n_rows_out']:,} | {store['n_rows_renamed']:,} | "
            f"{store['n_rows_prime_restored']:,} | {len(store['rows_by_name'])} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Measure and write the results file, or render it with ``--table``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", action="store_true")
    args = parser.parse_args(argv)
    if args.table:
        with open(RESULTS) as handle:
            print(render_tables(json.load(handle)))
        return 0
    load_dotenv()
    report = measure(os.environ["DATA_ROOT"])
    os.makedirs(osp.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    print(
        f"{report.n_names} names ({report.n_per_store_occurrences} store occurrences): "
        f"{report.n_renamed} renamed, {report.n_merged} merged, {report.n_retired} "
        f"retired; tables cover exactly the measured names: "
        f"{report.tables_cover_exactly_the_measured_names}"
    )
    print(render_tables(json.loads(report.model_dump_json())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
