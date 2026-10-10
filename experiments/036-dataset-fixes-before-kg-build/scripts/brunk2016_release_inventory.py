# experiments/036-dataset-fixes-before-kg-build/scripts/brunk2016_release_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.brunk2016_release_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/brunk2016_release_inventory

r"""What Brunk 2016 released, what each arm serves, and whether it duplicates a sibling.

Three measurements, each regenerable, behind the Brunk 2016 claims in the PR and the
note. Every one reads the sha256-pinned bytes of the raw mirror; only ``--network``
re-probes the two retrieval routes.

1. **Access.** The Elsevier CDN components and the PMC author-manuscript object, with
   their status codes, so "scriptable" is a measurement and not a claim.
2. **Inventory.** Per arm, off the pinned workbooks: the sample grid, the columns by
   the unit in their own header, the protein keys by the released mapping route that
   keys them, and the records the loader serves after its drops.
3. **Duplication.** The proteome arm against the two landed E. coli proteomes whose key
   space it shares, Schmidt 2016 and Ishii 2007: the loci in common, and the Pearson r
   of the wild-type profiles over them. A shared locus is not a shared measurement, and
   the r is what says so.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/brunk2016_release_inventory.py
    python experiments/036-dataset-fixes-before-kg-build/scripts/brunk2016_release_inventory.py --network
"""

import argparse
import json
import os
import os.path as osp
import statistics
from typing import Any

import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import torchcell.datasets.ecoli.brunk2016 as brunk  # noqa: E402

RESULTS = osp.join(
    os.environ["EXPERIMENT_ROOT"], "036-dataset-fixes-before-kg-build", "results"
)
OUT = osp.join(RESULTS, "brunk2016_release_inventory.json")

#: The landed E. coli proteomes whose key space this one shares.
SIBLING_PROTEOMES: tuple[str, ...] = ("proteome_schmidt2016", "proteome_ishii2007")


def probe_access() -> dict[str, Any]:
    """The status of each retrieval route, and whether the body is a challenge page."""
    probes = {raw.name: raw.source_url for raw in brunk.RETRIEVED_FILES}
    out: dict[str, Any] = {}
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        for name, url in probes.items():
            response = client.get(url)
            body = (
                response.text[:4000]
                if response.headers.get("content-type", "").startswith("text")
                else ""
            )
            out[name] = {
                "url": url,
                "status": response.status_code,
                "bytes": len(response.content),
                "javascript_challenge": "_cf_chl_opt" in body
                or "Just a moment" in body,
            }
    return out


def measure_metabolite_arms(workbook: str) -> dict[str, Any]:
    """The metabolomics workbook: its sample grid, its columns and its served records."""
    columns = brunk.read_metabolite_columns(workbook)
    samples = brunk.read_metabolite_samples(workbook)
    hours = sorted(samples["Hour"].unique().tolist())
    per_block: dict[str, Any] = {}
    for block_name in ("micromolar", "grams_per_litre"):
        block = getattr(columns, block_name)
        filled = int(samples[list(block)].notna().sum().sum())
        with_any = int((samples[list(block)].notna().sum(axis=1) > 0).sum())
        per_block[block_name] = {
            "columns": list(block),
            "filled_cells": filled,
            "samples_with_at_least_one_value": with_any,
        }
    fuel = {column: int(samples[column].notna().sum()) for column in columns.product}
    return {
        "strains": sorted(samples["Strain"].unique().tolist()),
        "hours": hours,
        "samples": int(len(samples)),
        "columns_by_unit": {
            "micromolar": len(columns.micromolar),
            "grams_per_litre": len(columns.grams_per_litre),
            "fuel": len(columns.product),
            "no_unit_in_the_header": len(columns.unitless),
        },
        "no_unit_columns": list(columns.unitless),
        "per_block": per_block,
        "fuel_cells": fuel,
        "served_records": {
            "metabolome": brunk.EXPECTED_METABOLOME_RECORDS,
            "exometabolite": brunk.EXPECTED_EXOMETABOLITE_RECORDS,
            "titer": brunk.EXPECTED_TITER_RECORDS,
        },
    }


def measure_proteome_arm(workbook: str) -> dict[str, Any]:
    """The proteomics workbook: its samples, and its proteins by mapping route."""
    keys = brunk.read_protein_keys(workbook)
    areas = brunk.read_protein_areas(workbook)
    host = [key for key in keys if key.organism == brunk.HOST_SPECIES]
    by_route: dict[str, list[str]] = {}
    for key in host:
        by_route.setdefault(key.route or "refused", []).append(key.protein)
    hours = sorted({str(hour) for hour in areas["Hour"].unique()})
    return {
        "samples": int(areas.groupby(["Strain", "Hour"]).ngroups),
        "hour_labels": hours,
        "proteins": len({key.protein for key in keys}),
        "host_proteins": len(host),
        "host_proteins_by_route": {
            route: sorted(names) for route, names in by_route.items()
        },
        "non_host_proteins": sorted(
            {key.protein for key in keys if key.organism != brunk.HOST_SPECIES}
        ),
        "served_records": brunk.EXPECTED_PROTEOME_RECORDS,
    }


def _wild_type_profile(records: list[dict[str, Any]]) -> dict[str, float]:
    """The mean abundance per locus over the records that perturb no gene."""
    totals: dict[str, list[float]] = {}
    for record in records:
        if record["experiment"]["genotype"]["perturbations"]:
            continue
        for locus, value in record["experiment"]["phenotype"][
            "protein_abundance"
        ].items():
            totals.setdefault(str(locus), []).append(float(value))
    return {locus: statistics.fmean(values) for locus, values in totals.items()}


def _symbol_map(strain: str) -> dict[str, str]:
    """``locus tag -> gene symbol`` for one pinned assembly, from its own annotation."""
    from torchcell.datasets.bacteria_common import bacterial_genome

    table = bacterial_genome("ecoli", strain).gene_attribute_table
    return {
        str(tag): str(symbol)
        for tag, symbol in zip(table["locus_tag"], table["gene"])
        if isinstance(symbol, str) and symbol
    }


def measure_duplication(data_root: str) -> dict[str, Any]:
    """The proteome arm against each landed sibling, joined on the gene SYMBOL.

    The stores do not share a key space: this arm is pinned to MG1655 and keyed by
    b-number, the two siblings are pinned to BW25113 and keyed by its locus tags, so a
    locus-level join finds nothing and says nothing. The join is therefore each
    assembly's own gene symbol, and the number reported is the Pearson r of the two
    wild-type profiles over the symbols in common.
    """
    from torchcell.verification.runners import load_records

    mine = _wild_type_profile(
        load_records(osp.join(data_root, "data/torchcell/proteome_brunk2016"))
    )
    mg1655 = _symbol_map("MG1655")
    bw25113 = _symbol_map("BW25113")
    by_symbol = {
        mg1655[locus]: value for locus, value in mine.items() if locus in mg1655
    }
    out: dict[str, Any] = {
        "brunk_wild_type_loci": len(mine),
        "brunk_wild_type_symbols": len(by_symbol),
        "key_namespaces": {
            "proteome_brunk2016": "MG1655 b-number",
            "siblings": "BW25113 locus tag",
        },
    }
    for sibling in SIBLING_PROTEOMES:
        root = osp.join(data_root, "data/torchcell", sibling)
        if not osp.exists(osp.join(root, "processed", "lmdb")):
            out[sibling] = {"state": "not built under this DATA_ROOT"}
            continue
        records = load_records(root)
        theirs = _wild_type_profile(records)
        their_symbols = {
            bw25113[locus]: value for locus, value in theirs.items() if locus in bw25113
        }
        shared = sorted(set(by_symbol) & set(their_symbols))
        row: dict[str, Any] = {
            "measurement_type": records[0]["experiment"]["phenotype"][
                "measurement_type"
            ],
            "their_loci": len(theirs),
            "shared_locus_tags": len(set(mine) & set(theirs)),
            "shared_symbols": len(shared),
        }
        if len(shared) >= 3:
            row["pearson_r_over_shared_symbols"] = round(
                float(
                    pd.Series([by_symbol[symbol] for symbol in shared]).corr(
                        pd.Series([their_symbols[symbol] for symbol in shared])
                    )
                ),
                4,
            )
        out[sibling] = row
    return out


def main() -> int:
    """Measure, print a summary and write the JSON beside the other 036 results."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network", action="store_true", help="re-probe the two retrieval routes"
    )
    args = parser.parse_args()

    data_root = os.environ["DATA_ROOT"]
    mirror = brunk.raw_mirror_dir(data_root)
    metabolomics = str(mirror / brunk.METABOLOMICS_REL)
    proteomics = str(mirror / brunk.PROTEOMICS_REL)
    for path, pinned in (
        (metabolomics, brunk.METABOLOMICS_SHA256),
        (proteomics, brunk.PROTEOMICS_SHA256),
    ):
        digest = brunk._sha256(path)
        if digest != pinned:
            raise RuntimeError(f"{path} is not the pinned workbook ({digest})")

    report: dict[str, Any] = {
        "citation_key": brunk.CITATION_KEY,
        "doi": brunk.DOI,
        "pmcid": brunk.PMCID,
        "quotes_checked": brunk.verify_quotes(data_root),
        "table_s1": brunk.read_table_s1(mirror / brunk.SI_OCR_REL).model_dump(),
        "metabolite_arms": measure_metabolite_arms(metabolomics),
        "proteome_arm": measure_proteome_arm(proteomics),
        "duplication_against_landed_proteomes": measure_duplication(data_root),
    }
    if args.network:
        report["access"] = probe_access()

    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
