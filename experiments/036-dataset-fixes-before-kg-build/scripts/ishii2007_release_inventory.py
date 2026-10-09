# experiments/036-dataset-fixes-before-kg-build/scripts/ishii2007_release_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.ishii2007_release_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/ishii2007_release_inventory

r"""What Ishii 2007 released, what is scriptable, and what each layer would serve.

Three measurements, each regenerable, behind the Ishii 2007 claims in PR and note:

1. **Access.** The publisher supplement, the article PDF, the publisher's own
   free-access ``ijkey`` link printed on the project web site, the retired
   ``sciencemag.org`` SOM URL, and the PMC id converter. The row's accession read
   "Science supporting online material; no repository accession found"; this says
   precisely which of those answer, with what.
2. **Inventory.** A HEAD of every file the paper's reference-21 project web site links,
   with its byte size and ``Last-Modified``, so "the release" is a list and not a
   gesture.
3. **Layers.** Per layer, off the sha256-pinned workbook in the raw mirror: the sample
   columns, the targets, the filled cells, and the records a loader would serve after
   the structural drops. This is what replaces the row's ``dim=4300`` estimate.

Steps 1 and 2 reach the network and are skipped without ``--network``; step 3 reads only
the mirror. Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/ishii2007_release_inventory.py
    python experiments/036-dataset-fixes-before-kg-build/scripts/ishii2007_release_inventory.py --network
"""

import argparse
import json
import os
import os.path as osp
from typing import Any

import httpx
import xlrd
from dotenv import load_dotenv

load_dotenv()

import torchcell.datasets.ecoli.ishii2007 as ishii  # noqa: E402

RESULTS = osp.join(
    os.environ["EXPERIMENT_ROOT"], "036-dataset-fixes-before-kg-build", "results"
)
OUT = osp.join(RESULTS, "ishii2007_release_inventory.json")

#: Every access route to the publisher's supporting online material we know of.
ACCESS_PROBES: dict[str, str] = {
    "science_org_supplement": ishii.PUBLISHER_SUPPLEMENT_URL,
    "science_org_article": "https://www.science.org/doi/10.1126/science.1132067",
    "science_org_pdf_with_publisher_ijkey": (
        "https://www.science.org/doi/pdf/10.1126/science.1132067"
        "?ijkey=9Ix5kSaD9ei06&keytype=ref"
    ),
    "sciencemag_som_url_printed_in_the_paper": (
        "https://www.sciencemag.org/cgi/content/full/1132067/DC1"
    ),
    "project_web_site": ishii.PROJECT_SITE,
}
PMC_IDCONV = (
    "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
    f"?ids={ishii.PAPER_DOI.replace('/', '%2F')}&format=json&tool=torchcell"
)

#: Every file the project web site links, consumed or not.
RELEASED_FILES: tuple[str, ...] = (
    "Quantitative_data.xls",
    "Metabolome_UK_data.xls",
    "DNAArray_data.xls",
    "DNAArray_raw-data.zip",
    "2D-DIGE_ratio_data.xls",
    "2D-DIGE_ID_list.xls",
    "Flux_GC-MS_data.xls",
)

#: ``layer -> (sheet, first data row, name row, series row, paired Conc/CV)``.
LAYERS: dict[str, tuple[str, int, int, int | None, bool]] = {
    "mRNA (qRT-PCR, copy number/ug-total RNA)": ("mRNA", 4, 2, 1, True),
    "protein (LC-MS/MS, mg/g-DCW)": ("Protein", 4, 2, 1, True),
    "metabolite (CE-TOFMS, mM)": ("Metabolite", 3, 2, 1, False),
    "flux (13C MFA, percent of glucose uptake)": ("Flux", 2, 1, None, False),
}


def probe_access() -> dict[str, Any]:
    """HTTP status, final URL and whether the body is a JS challenge, per route."""
    out: dict[str, Any] = {}
    with httpx.Client(follow_redirects=True, timeout=45.0) as client:
        for name, url in ACCESS_PROBES.items():
            response = client.get(url)
            body = response.text[:4000]
            out[name] = {
                "url": url,
                "status": response.status_code,
                "final_url": str(response.url),
                "javascript_challenge": "_cf_chl_opt" in body
                or "Just a moment" in body,
            }
        response = client.get(PMC_IDCONV)
        out["pmc_id_converter"] = {
            "url": PMC_IDCONV,
            "status": response.status_code,
            "record": response.json()["records"][0],
        }
    return out


def probe_inventory() -> list[dict[str, Any]]:
    """A HEAD of every released file: bytes, type and Last-Modified."""
    rows: list[dict[str, Any]] = []
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        for name in RELEASED_FILES:
            url = f"{ishii.PROJECT_SITE}{name}"
            response = client.head(url)
            rows.append(
                {
                    "file": name,
                    "url": url,
                    "status": response.status_code,
                    "bytes": int(response.headers.get("content-length", -1)),
                    "content_type": response.headers.get("content-type"),
                    "last_modified": response.headers.get("last-modified"),
                    "consumed_by_the_loader": name in ishii.DATA_SHA256,
                }
            )
    return rows


def _filled(sheet: Any, rows: list[int], column: int) -> int:
    return sum(
        1
        for row in rows
        if isinstance(sheet.cell_value(row, column), float)
        or str(sheet.cell_value(row, column)).strip() not in ("", "-")
    )


def measure_layers(book: Any) -> dict[str, Any]:
    """Per layer: sample columns by arm, targets, filled cells and servable records."""
    out: dict[str, Any] = {}
    for label, (name, first, name_row, series_row, paired) in LAYERS.items():
        sheet = book.sheet_by_name(name)
        columns = ishii.read_sample_columns(
            sheet, paired=paired, series_row=series_row, name_row=name_row
        )
        rows = [row for row, _ in ishii.read_row_labels(sheet, first)]
        kinds: dict[str, int] = {}
        for column in columns:
            kinds[column.kind] = kinds.get(column.kind, 0) + 1
        dilution_rates = ishii.check_dilution_rate_arm(book, columns)
        kept, ledger = ishii.classify_columns(
            sheet, columns, rows, dataset=label, dilution_rates=dilution_rates
        )
        out[label] = {
            "sheet": name,
            "sample_columns": len(columns),
            "sample_columns_by_arm": kinds,
            "targets": len(rows),
            "filled_cells": sum(_filled(sheet, rows, c.column) for c in columns),
            "servable_records": len(kept),
            "dilution_rates_served": sorted(
                {ishii.culture_dilution_rate(column, dilution_rates) for column in kept}
            ),
            "dropped": {
                rule.rule: list(rule.sample_ids)
                for rule in ledger.rules
                if rule.sample_ids
            },
        }
    return out


def main() -> int:
    """Measure, print a summary and write the JSON beside the other 036 results."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        action="store_true",
        help="run the access probes and the released-file inventory",
    )
    args = parser.parse_args()

    mirror = (
        ishii.raw_mirror_dir()
        / ishii.RAW_FILES_BY_NAME[ishii.QUANTITATIVE_FILE].mirror_relpath
    )
    digest = ishii._sha256(mirror)
    if digest != ishii.DATA_SHA256[ishii.QUANTITATIVE_FILE]:
        raise RuntimeError(f"{mirror} is not the pinned workbook ({digest})")
    book = xlrd.open_workbook(str(mirror), formatting_info=True)

    report: dict[str, Any] = {
        "citation_key": ishii.CITATION_KEY,
        "doi": ishii.PAPER_DOI,
        "workbook_sha256": digest,
        "layers": measure_layers(book),
        "metabolite_protocol_rows": ishii.EXPECTED_PROTOCOL_ROWS,
        "loaded_datasets": ishii.EXPECTED_RECORDS,
        "dilution_rate_arm_per_hour": list(ishii.DILUTION_RATE_ARM),
        "reference_dilution_rate_per_hour": ishii.DILUTION_RATE_PER_HOUR,
        "retired_drop_rules": [dict(rule) for rule in ishii.RETIRED_DROP_RULES],
    }
    if args.network:
        report["access"] = probe_access()
        report["released_files"] = probe_inventory()

    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    for label, values in report["layers"].items():
        print(
            f"{label}: {values['sample_columns']} columns, {values['targets']} targets, "
            f"{values['filled_cells']} filled, {values['servable_records']} records"
        )
    if args.network:
        for name, values in report["access"].items():
            if name == "pmc_id_converter":
                print(f"PMC: {values['record'].get('errmsg', values['record'])}")
            else:
                print(
                    f"{name}: HTTP {values['status']} "
                    f"(js_challenge={values['javascript_challenge']})"
                )
        for row in report["released_files"]:
            print(
                f"{row['file']}: HTTP {row['status']} {row['bytes']} B "
                f"consumed={row['consumed_by_the_loader']}"
            )
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
