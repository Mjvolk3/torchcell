# experiments/036-dataset-fixes-before-kg-build/scripts/manual_problems_ledger_measurements.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.manual_problems_ledger_measurements]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/manual_problems_ledger_measurements
"""Measure the numbers the manual-problems ledger states about the machine, not the papers.

The ledger (``notes/database.manual-problems-ledger.2026.10.md``) quotes most of its
content verbatim from issues and PR bodies. The rest is measured here, read-only:

- **record counts** of the dev stores whose owner decision or retrieval the ledger names,
  read as the LMDB entry count of ``$DATA_ROOT/data/torchcell/<store>/processed/lmdb``;
- **Zotero coverage**: every ``CITATION_KEY`` a bacterial loader module declares (plus the
  extra keys of settled records that live only in the raw mirror), and whether
  ``$DATA_ROOT/torchcell-library/<key>/manifest.json`` exists and names a Zotero item;
- **mirror presence** of the files the walled-retrieval rows ask the owner to deposit;
- **retired dev-store copies** (``*.superseded.*`` and ``deprecated-*`` beside the live
  stores, and the ``scripts/deprecate.sh`` graveyard) and the two ``/db`` leftovers, by
  ``du`` (``--disk``; slow, about two minutes on GilaHyper);
- **#734's residual hole**: which dev-store ``build_manifest.json`` files record no
  module-level binding (an all-caps closure key such as ``CHEBI_ID_PATTERN``), the gap PR
  #831 left "for your call";
- **PubChem reachability** from this host (``--probe-pubchem``): a handful of sequential
  PUG REST name lookups and their HTTP status, the measurement behind #726's host
  question. Each request takes about 5 s on GilaHyper because of the resolver stall.

Nothing is written outside ``experiments/036-dataset-fixes-before-kg-build/results/``.

Usage, from the repository root::

    PYTHONPATH=. python experiments/036-dataset-fixes-before-kg-build/scripts/manual_problems_ledger_measurements.py
    PYTHONPATH=. python experiments/036-dataset-fixes-before-kg-build/scripts/manual_problems_ledger_measurements.py --disk --probe-pubchem
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import os.path as osp
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request

import lmdb
from dotenv import load_dotenv
from pydantic import BaseModel

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
EXPERIMENT_DIR = osp.dirname(SCRIPT_DIR)
REPO = osp.dirname(osp.dirname(EXPERIMENT_DIR))
RESULTS = osp.join(
    EXPERIMENT_DIR, "results", "manual_problems_ledger_measurements.json"
)

#: Dev stores the ledger names, with the ledger row each count supports.
STORES: dict[str, str] = {
    "ecoli_carbon_source_tong2020": "#788 item 3 (CarPE concentrations)",
    "mrna_fraction_balakrishnan2022": "#853, #906",
    "metabolome_ishii2007": "#699",
    "proteome_ishii2007": "#699",
    "flux_ishii2007": "#699",
    "isoprenyl_acetate_titer_kang2026": "#788 item 4 (Kang reference titer)",
    "rbtnseq_borchert2024": "#726 / #860, #908",
    "rbtnseq_borchert2023": "#913",
    "crispri_guide_fitness_wang2018": "#879",
    "pputida_env_metal_tnseq_royet2025": "#909",
    "proteome_brunk2016": "#904",
    "isoprenol_selection_menasalvas2025": "#915",
    "protein_fold_change_caglar2017": "#914",
    "ecoli_crispri_rousset2018": "#905",
    "crispri_knockdown_cui2018": "#905",
    "inhibitor_bioscreen_volk2021": "#912",
    "ffa_xue2025": "#887",
    "tmi_kuzmin2018": "#896",
    "dmf_kuzmin2018": "#894",
    "tmf_kuzmin2018": "#895",
    "dmf_kuzmin2020": "#897",
    "gene_essentiality_sgd": "#901",
    "syn_leth_db_yeast": "#902",
    "syn_rescue_db_yeast": "#903",
}

#: Keys of settled records whose loaders declare no CITATION_KEY of their own.
EXTRA_KEYS: tuple[str, ...] = (
    "randMetabolicPathwayCatabolizing2017",
    "thompsonMassivelyParallelFitness2019",
    "tetenevaGenomewideScreenGenetic2024",
    "banerjeeGenomescalePathwayEngineering2024",
    "nicholsPhenotypicLandscapeBacterial2011",
)

#: Files the walled-retrieval rows ask for, relative to the mirror root named first.
EXPECTED_DEPOSITS: dict[str, tuple[str, str]] = {
    "#699 Ishii 2007 SOM": (
        "torchcell-raw",
        "ishiiMultipleHighThroughputAnalyses2007/si",
    ),
    "#853 Balakrishnan 2022 Table S6": (
        "torchcell-raw",
        "balakrishnanPrinciplesGeneRegulation2022/data/NIHMS1856867-supplement-Table_S6.xlsx",
    ),
    "#788 item 3 Tong 2020 CarPE": (
        "torchcell-raw",
        "tongGeneDispensabilityEscherichia2020/data/carpe",
    ),
    "#691 Nichols 2011 paper": (
        "torchcell-library",
        "nicholsPhenotypicLandscapeBacterial2011",
    ),
}

#: The two /db leftovers the build/ops rows name.
DB_LEFTOVERS: tuple[str, ...] = (
    "/db/database/biocypher-out/2026-10-09_07-20-33",
    "/db/database/data.superseded.2026-10-06_04-41-16",
)

PUBCHEM_NAMES: tuple[str, ...] = (
    "potassium chloride",
    "sodium dihydrogen phosphate",
    "nicotinic acid",
    "cyanocobalamin",
    "valerolactam",
    "isoprenol",
)

CITATION_KEY_LINE = re.compile(
    r'^[A-Z_]*CITATION_KEY[A-Z_]*\s*(?::\s*\w+\s*)?=\s*"([A-Za-z0-9]+)"', re.MULTILINE
)


class StoreCount(BaseModel):
    """One dev store's LMDB entry count and the ledger row it supports."""

    store: str
    ledger_row: str
    records: int | None  # None when the store has no processed/lmdb


class ZoteroCoverage(BaseModel):
    """Whether a bacterial citation key has a Zotero-backed library mirror."""

    citation_key: str
    modules: list[str]
    in_library_mirror: bool
    zotero_item_key: str | None
    in_raw_mirror: bool


class DepositCheck(BaseModel):
    """Whether a file a walled-retrieval row asks for is on disk yet."""

    row: str
    path: str
    present: bool


class DiskUsage(BaseModel):
    """``du`` of one set of retired copies or one /db leftover."""

    what: str
    directories: int
    du_human: str


class PubchemProbe(BaseModel):
    """One PUG REST name lookup from this host and its HTTP status."""

    name: str
    http_status: int
    seconds: float


class Measurements(BaseModel):
    """Everything this script measured, written to the results JSON."""

    measured_at: str
    hostname: str
    git_head: str
    stores: list[StoreCount]
    zotero: list[ZoteroCoverage]
    deposits: list[DepositCheck]
    manifests_without_bindings: list[str]
    disk: list[DiskUsage] | None
    pubchem: list[PubchemProbe] | None


def lmdb_entries(store_dir: str) -> int | None:
    path = osp.join(store_dir, "processed", "lmdb")
    if not osp.isdir(path):
        return None
    env = lmdb.open(path, readonly=True, lock=False, readahead=False, max_readers=1)
    entries = int(env.stat()["entries"])
    env.close()
    return entries


def zotero_coverage(data_root: str) -> list[ZoteroCoverage]:
    modules: dict[str, set[str]] = {}
    for module in sorted(
        glob.glob(osp.join(REPO, "torchcell/datasets/ecoli/*.py"))
        + glob.glob(osp.join(REPO, "torchcell/datasets/pputida/*.py"))
    ):
        with open(module) as handle:
            text = handle.read()
        for match in CITATION_KEY_LINE.finditer(text):
            modules.setdefault(match.group(1), set()).add(osp.basename(module))
    for key in EXTRA_KEYS:
        modules.setdefault(key, set())
    rows = []
    for key in sorted(modules):
        manifest = osp.join(data_root, "torchcell-library", key, "manifest.json")
        item = None
        if osp.exists(manifest):
            with open(manifest) as handle:
                item = json.load(handle)["zotero_item_key"]
        rows.append(
            ZoteroCoverage(
                citation_key=key,
                modules=sorted(modules[key]),
                in_library_mirror=osp.exists(manifest),
                zotero_item_key=item,
                in_raw_mirror=osp.isdir(osp.join(data_root, "torchcell-raw", key)),
            )
        )
    return rows


BINDING_KEY = re.compile(r"_?[A-Z][A-Z0-9_]+")


def manifests_without_bindings(dev: str) -> list[str]:
    """Store directories whose manifest closure names no module-level binding."""
    missing = []
    for manifest in sorted(
        glob.glob(osp.join(dev, "*", "preprocess", "build_manifest.json"))
    ):
        with open(manifest) as handle:
            closure = json.load(handle)["closure"]
        if not any(BINDING_KEY.fullmatch(key) for key in closure):
            missing.append(osp.basename(osp.dirname(osp.dirname(manifest))))
    return missing


def du_paths(paths: list[str]) -> str:
    if not paths:
        return "0"
    out = subprocess.run(
        ["du", "-csh", *paths], check=True, capture_output=True, text=True
    ).stdout
    return out.strip().splitlines()[-1].split()[0]


def disk_usage(data_root: str) -> list[DiskUsage]:
    dev = osp.join(data_root, "data", "torchcell")
    superseded = sorted(
        glob.glob(osp.join(dev, "*", "*.superseded.*"))
        + glob.glob(osp.join(dev, "*.superseded*"))
    )
    deprecated = sorted(
        glob.glob(osp.join(dev, "*", "deprecated-*"))
        + glob.glob(osp.join(dev, "*.deprecated-*"))
    )
    graveyard = "/scratch/projects/torchcell-deprecated"
    rows = [
        DiskUsage(
            what=f"{dev}/**/*.superseded.*",
            directories=len(superseded),
            du_human=du_paths(superseded),
        ),
        DiskUsage(
            what=f"{dev}/**/deprecated-*",
            directories=len(deprecated),
            du_human=du_paths(deprecated),
        ),
        DiskUsage(
            what=graveyard,
            directories=len(os.listdir(graveyard)),
            du_human=du_paths([graveyard]),
        ),
    ]
    for path in DB_LEFTOVERS:
        rows.append(DiskUsage(what=path, directories=1, du_human=du_paths([path])))
    return rows


def probe_pubchem() -> list[PubchemProbe]:
    probes = []
    for name in PUBCHEM_NAMES:
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{urllib.parse.quote(name)}/property/InChIKey/JSON"
        )
        start = dt.datetime.now()
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                status = response.status
        except urllib.error.HTTPError as error:  # a 429 or 404 is the measurement
            status = error.code
        probes.append(
            PubchemProbe(
                name=name,
                http_status=status,
                seconds=round((dt.datetime.now() - start).total_seconds(), 2),
            )
        )
    return probes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--disk", action="store_true", help="du the retired copies (slow)"
    )
    parser.add_argument(
        "--probe-pubchem", action="store_true", help="probe PubChem from this host"
    )
    args = parser.parse_args()

    load_dotenv(osp.join(REPO, ".env"))
    data_root = os.environ["DATA_ROOT"]
    dev = osp.join(data_root, "data", "torchcell")

    stores = [
        StoreCount(store=s, ledger_row=row, records=lmdb_entries(osp.join(dev, s)))
        for s, row in STORES.items()
    ]
    deposits = [
        DepositCheck(
            row=row,
            path=osp.join(data_root, root, rel),
            present=osp.exists(osp.join(data_root, root, rel)),
        )
        for row, (root, rel) in EXPECTED_DEPOSITS.items()
    ]
    head = subprocess.run(
        ["git", "-C", REPO, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = Measurements(
        measured_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        hostname=os.uname().nodename,
        git_head=head,
        stores=stores,
        zotero=zotero_coverage(data_root),
        deposits=deposits,
        manifests_without_bindings=manifests_without_bindings(dev),
        disk=disk_usage(data_root) if args.disk else None,
        pubchem=probe_pubchem() if args.probe_pubchem else None,
    )
    os.makedirs(osp.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as handle:
        handle.write(result.model_dump_json(indent=2) + "\n")

    for s in result.stores:
        print(f"{s.store:40s} {s.records!s:>10s}  {s.ledger_row}")
    absent = [z for z in result.zotero if z.zotero_item_key is None]
    print(f"Zotero: {len(result.zotero)} keys, {len(absent)} without a Zotero item")
    for z in absent:
        print(f"  {z.citation_key} raw={z.in_raw_mirror} modules={z.modules}")
    for d in result.deposits:
        print(f"deposit {d.row}: present={d.present}")
    print(f"manifests without binding nodes: {result.manifests_without_bindings}")
    for u in result.disk or []:
        print(f"disk {u.du_human:>6s} over {u.directories} dirs: {u.what}")
    for p in result.pubchem or []:
        print(f"pubchem {p.http_status} {p.seconds}s {p.name}")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
