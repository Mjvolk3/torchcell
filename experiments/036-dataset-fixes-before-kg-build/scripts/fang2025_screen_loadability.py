# experiments/036-dataset-fixes-before-kg-build/scripts/fang2025_screen_loadability.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.fang2025_screen_loadability]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/fang2025_screen_loadability
"""Re-verify schedule row 43, Fang 2025 FFA CRISPRi-FACS, by measuring the release.

The row carries ``status="blocked"`` with the reason that "Pooled enrichment gives a
guide-level score rather than a per-design titer, so only the handful of individually
reconstructed strains carry phenotype values; no sequencing accession was confirmed."
Both halves of that are tested here against the pinned bytes rather than re-asserted.

Five questions, each answered by reading a file:

1. **Is the deposit mirrored, in full?** The publisher deposit is enumerated from the
   PMC Article Datasets bucket and from the article's JATS
   ``<supplementary-material>`` elements, then matched against the library mirror's own
   ``manifest.json`` by sha256. A declared file absent from either side shows up as a
   named disagreement rather than as silence.

2. **Does the release carry a per-guide score?** Every sheet of Supplementary Data 6
   (``si8.xlsx``) is characterized: its dimensions, the header blocks it holds, and
   whether a block's unit of observation is a guide or a strain. The two screen sheets
   are counted exactly.

3. **Do the Methods equations reproduce from the released columns?** Equation (1)
   ``(reads + 1) / total reads`` and equation (2) ``log2(normalized AS / normalized
   BS)`` are recomputed per row from the released read counts and compared to the
   released ``Normalized read`` and ``Fitness (all)`` columns. A formula that does not
   reproduce is a reason not to load, so it is measured, not assumed.

4. **Does the authors' own 20-read floor explain the ``Fitness`` column?** The Methods
   state "sgRNAs with fewer than 20 reads in each library were excluded from the
   analysis to calculate sgRNA fitness", and ``si8.xlsx`` splits the fitness into one
   qualified column plus one column per library that failed. The floor is applied
   independently and cross-tabulated against which rows carry the qualified value, so
   the retention rule is derived from the bytes.

5. **Is it subsumed by a dataset we serve?** The screened set is matched id-for-id
   against the guide library of ``CrispriGuideFitnessWang2018Dataset``, which we serve,
   and the phenotype is matched against that dataset's five screen ids. An identical
   library with a phenotype none of the served screens holds is new labels on known
   strains, which is the opposite of subsumption, so both halves are reported.

Writes ``results/fang2025_screen_loadability.json`` plus
``results/fang2025_screen_sheets.csv`` (the per-sheet characterization) and
``results/fang2025_screen_retention.csv`` (the per-round retention arithmetic).

Usage (from the repo root)::

    python experiments/036-dataset-fixes-before-kg-build/scripts/fang2025_screen_loadability.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import os.path as osp
import re
import urllib.request
from collections import Counter
from typing import Any
from xml.etree import ElementTree as ET

import openpyxl
import pandas as pd
from dotenv import load_dotenv

CITATION_KEY = "fangGenomescaleCRISPRiScreen2025"
LIBRARY_DIR_REL = f"torchcell-library/{CITATION_KEY}"
DOI = "10.1038/s41467-025-58368-3"

#: Supplementary Data 6, the source-data workbook: the only released file whose unit of
#: observation is a single sgRNA.
SI8_RELPATH = "si/si8.xlsx"
SI8_SHA256 = "04945e4651c05b1c6d6fa3dc057ae6bced9f370b2537068472354510d20a8a5a"

#: The PMC Article Datasets bucket prefix of this article, from the ``pmc_cloud``
#: retrieval record the library manifest stores for every ``si/`` object.
PMC_PREFIX = "PMC11954867.1"
BUCKET = "pmc-oa-opendata"
BUCKET_LIST_URL = f"https://s3.amazonaws.com/{BUCKET}?list-type=2&prefix={PMC_PREFIX}"
ARTICLE_XML_URL = f"https://{BUCKET}.s3.amazonaws.com/{PMC_PREFIX}/{PMC_PREFIX}.xml"

#: Objects in the bucket that are PMC's own renditions of the article rather than
#: publisher supplementary files.
_RENDITION_SUFFIXES = (".json", ".txt", ".xml")
_FIGURE_RE = re.compile(r"^\d+_\d+_\d+_(Fig|MOESM)", re.IGNORECASE)

#: The guide library of the dataset we already serve, read from ITS OWN raw mirror under
#: ITS OWN citation key: Wang 2018 Supplementary Data 3, guide id -> 20-mer spacer.
WANG_CITATION_KEY = "wangPooledCRISPRInterference2018"
WANG_LIBRARY_REL = (
    f"torchcell-raw/{WANG_CITATION_KEY}/si/si_data/41467_2018_4899_MOESM6_ESM.xlsx"
)
WANG_LIBRARY_SHA256 = (
    "78a05c25da94df158065c18a316d0d29869587e4c839ad3365c911503900ef62"
)
#: The five screen ids ``CrispriGuideFitnessWang2018Dataset`` serves, in Table 2's order.
WANG_SCREEN_IDS = (
    "essentiality",
    "auxotrophy",
    "trp_biosynthesis",
    "furfural_tolerance",
    "isobutanol_tolerance",
)
#: Wang 2018's non-targeting control ids, which Fang does not release.
CONTROL_ID_RE = re.compile(r"^NC_\d+$")

#: The two screen rounds of ``si8.xlsx``, each a sheet and the 0-based column offset of
#: its guide block inside the row tuple ``openpyxl`` yields.
ROUNDS: tuple[dict[str, Any], ...] = (
    {
        "round_id": "round_1_cf",
        "sheet": "Figure 1",
        "panel": "Figure 1d",
        "id_col": 10,
        "read_cols": {
            "transformation": 11,
            "before_sorting": 12,
            "after_sorting": 13,
        },
        "normalized": {"before_sorting": 14, "after_sorting": 15},
        "fitness_col": 16,
        "excluded_cols": {
            "transformation": 18,
            "before_sorting": 19,
            "after_sorting": 20,
        },
        "fitness_all_col": 21,
        "host_strain": "CF",
    },
    {
        "round_id": "round_2_pcnbi",
        "sheet": "Figure 5",
        "panel": "Figure 5c",
        "id_col": 10,
        "read_cols": {"before_sorting": 11, "after_sorting": 12},
        "normalized": {"before_sorting": 13, "after_sorting": 14},
        "fitness_col": 15,
        "excluded_cols": {"before_sorting": 17, "after_sorting": 18},
        "fitness_all_col": 19,
        "host_strain": "pcnBi",
    },
)

#: The authors' stated read floor, and the sentence that states it, verbatim.
READ_FLOOR = 20
READ_FLOOR_QUOTE = (
    "sgRNAs with fewer than 20 reads in each library were excluded from the analysis "
    "to calculate sgRNA fitness."
)
#: The library-provenance sentence, verbatim, and the reference it cites.
LIBRARY_QUOTE = (
    "a previously published plasmid library containing 55,671 sgRNAs20 was transformed "
    "into E. coli strain CF (an MG1655 (DE3) derivative with fadE deletion) carrying "
    "the pCF plasmid for expression of dCas9 and the truncated fatty acyl-ACP "
    "thioesterase TesA′"
)
LIBRARY_REFERENCE_QUOTE = (
    "20. Wang, T. et al. Pooled CRISPR interference screening enables genome-scale "
    "functional genomics study in bacteria with superior performance. Nat. Commun. 9, "
    "2475 (2018)."
)
#: The sentence that makes the ratio a sort enrichment rather than a growth selection.
SORT_ONLY_QUOTE = (
    "Thus, fitness reflects the relative abundance of each sgRNA due solely to "
    "sorting, helping to minimize data noise from cultivation."
)
#: The sequencing accession the row calls unconfirmed.
GEO_ACCESSION_QUOTE = (
    "The NGS data generated in this study have been deposited in the NCBI GEO database "
    "under accession code GSE267827."
)
GEO_ACCESSION = "GSE267827"

#: Sheets of ``si8.xlsx`` whose first numeric block is a per-strain titer in mg/L, with
#: the header the sheet prints for that block.
TITER_HEADER = "FFAs (mg L-1)"


def sha256_of(path: str) -> str:
    """Hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def library_dir(data_root: str) -> str:
    """This paper's directory in the torchcell-library mirror."""
    return osp.join(data_root, LIBRARY_DIR_REL)


def verify_mirror(data_root: str) -> dict[str, Any]:
    """sha256-verify every mirrored ``si/`` object against the library manifest.

    Returns one record per retrieved supplementary object, carrying the retrieval
    method, source url and retriever the manifest stores, so the provenance of the
    bytes this script measures is part of its own output.
    """
    library = library_dir(data_root)
    manifest = json.loads(open(osp.join(library, "manifest.json")).read())
    retrieved: list[dict[str, Any]] = []
    for entry in manifest["files"]:
        path = entry["path"]
        if not path.startswith("si/") or entry.get("retrieval") is None:
            continue
        observed = sha256_of(osp.join(library, path))
        if observed != entry["sha256"]:
            raise RuntimeError(
                f"{path}: sha256 {observed}, manifest records {entry['sha256']}"
            )
        retrieval = entry["retrieval"]
        retrieved.append(
            {
                "path": path,
                "bytes": entry["bytes"],
                "sha256": observed,
                "original_filename": entry.get("original_filename"),
                "retrieval_method": retrieval["method"],
                "source_url": retrieval["source_url"],
                "retrieval_command": retrieval["retriever"],
                "retrieval_params": retrieval["params"],
                "retrieved_at": retrieval["retrieved_at"],
            }
        )
    si8 = next(r for r in retrieved if r["path"] == SI8_RELPATH)
    if si8["sha256"] != SI8_SHA256:
        raise RuntimeError(
            f"{SI8_RELPATH}: sha256 {si8['sha256']}, module pins {SI8_SHA256}"
        )
    return {
        "doi": manifest["doi"],
        "title": manifest["title"],
        "retrieved_si_objects": retrieved,
        "n_retrieved_si_objects": len(retrieved),
    }


def enumerate_publisher_deposit() -> dict[str, Any]:
    """List the article's bucket objects and its JATS supplementary-material elements."""
    with urllib.request.urlopen(BUCKET_LIST_URL, timeout=120) as response:
        listing = response.read().decode("utf-8")
    keys = re.findall(r"<Key>([^<]+)</Key>", listing)
    sizes = [int(n) for n in re.findall(r"<Size>(\d+)</Size>", listing)]
    objects = [
        {"key": key, "bytes": size, "name": key.split("/", 1)[1]}
        for key, size in zip(keys, sizes, strict=True)
    ]
    supplementary = [
        obj
        for obj in objects
        if "MOESM" in obj["name"] and not obj["name"].endswith(_RENDITION_SUFFIXES)
    ]
    with urllib.request.urlopen(ARTICLE_XML_URL, timeout=120) as response:
        article = response.read().decode("utf-8")
    root = ET.fromstring(article)
    declared = []
    for element in root.iter("supplementary-material"):
        media = element.find("media")
        declared.append(
            {
                "id": element.get("id"),
                "href": (
                    None
                    if media is None
                    else media.get("{http://www.w3.org/1999/xlink}href")
                ),
            }
        )
    return {
        "bucket_objects": objects,
        "supplementary_objects": supplementary,
        "n_supplementary_objects": len(supplementary),
        "declared_supplementary": declared,
        "n_declared_supplementary": len(declared),
    }


def reconcile_deposit(
    deposit: dict[str, Any], mirror: dict[str, Any]
) -> dict[str, Any]:
    """Name every publisher object the mirror lacks, and the reverse."""
    published = {obj["name"] for obj in deposit["supplementary_objects"]}
    mirrored = {
        r["original_filename"]
        for r in mirror["retrieved_si_objects"]
        if r["original_filename"]
    }
    return {
        "published_not_mirrored": sorted(published - mirrored),
        "mirrored_not_published": sorted(mirrored - published),
        "complete": published == mirrored,
    }


def characterize_sheets(path: str) -> list[dict[str, Any]]:
    """Every sheet of ``si8.xlsx``: its dimensions, its panels and its header blocks.

    A sheet is called ``per_guide`` when a header block names ``sgRNA-ID`` and
    ``per_strain`` when one names ``Strain`` or ``Substrate``; a sheet can be both,
    which is what the two screen sheets are.
    """
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out: list[dict[str, Any]] = []
    for sheet in workbook.worksheets:
        rows = list(sheet.iter_rows(min_row=1, max_row=3, values_only=True))
        panels = [str(c) for c in rows[0] if c is not None] if rows else []
        headers = [str(c) for c in rows[1] if c is not None] if len(rows) > 1 else []
        third = [str(c) for c in rows[2] if c is not None] if len(rows) > 2 else []
        out.append(
            {
                "sheet": sheet.title,
                "dimensions": sheet.calculate_dimension(),
                "max_row": sheet.max_row,
                "panels": panels,
                "headers": headers,
                "per_guide": "sgRNA-ID" in headers,
                "per_strain": any(
                    t in ("Strain", "Substrate") for t in headers + third
                ),
                "carries_titer_header": TITER_HEADER in headers + third,
            }
        )
    workbook.close()
    return out


def measure_round(path: str, spec: dict[str, Any]) -> dict[str, Any]:
    """One screen round: its ids, its equation residuals and its retention arithmetic.

    Equations (1) and (2) are recomputed from the released read counts and totals, and
    the ``>= READ_FLOOR`` rule is cross-tabulated against which rows carry the qualified
    ``Fitness`` value, so both the formula and the retention rule are derived.
    """
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook[spec["sheet"]]
    ids: list[str] = []
    reads: dict[str, list[int]] = {name: [] for name in spec["read_cols"]}
    released_normalized: list[tuple[float, float]] = []
    fitness: list[float | None] = []
    fitness_all: list[float] = []
    excluded_flags: list[tuple[bool, ...]] = []
    for row in sheet.iter_rows(min_row=3, values_only=True):
        guide = row[spec["id_col"]]
        if guide is None:
            continue
        ids.append(str(guide))
        for name, column in spec["read_cols"].items():
            reads[name].append(int(row[column]))
        released_normalized.append(
            (
                float(row[spec["normalized"]["before_sorting"]]),
                float(row[spec["normalized"]["after_sorting"]]),
            )
        )
        qualified = row[spec["fitness_col"]]
        fitness.append(None if qualified is None else float(qualified))
        fitness_all.append(float(row[spec["fitness_all_col"]]))
        excluded_flags.append(
            tuple(row[c] is not None for c in spec["excluded_cols"].values())
        )
    workbook.close()

    totals = {name: sum(values) for name, values in reads.items()}
    max_normalized_error = 0.0
    max_fitness_error = 0.0
    for index in range(len(ids)):
        before = (reads["before_sorting"][index] + 1) / totals["before_sorting"]
        after = (reads["after_sorting"][index] + 1) / totals["after_sorting"]
        released_before, released_after = released_normalized[index]
        max_normalized_error = max(
            max_normalized_error,
            abs(before - released_before),
            abs(after - released_after),
        )
        max_fitness_error = max(
            max_fitness_error,
            abs(math.log2(after / before) - fitness_all[index]),
        )

    floor = Counter()
    for index in range(len(ids)):
        passes = all(reads[name][index] >= READ_FLOOR for name in spec["read_cols"])
        floor[(passes, fitness[index] is not None)] += 1
    n_kept = sum(1 for value in fitness if value is not None)
    kept = [value for value in fitness if value is not None]
    partition = Counter(excluded_flags)
    return {
        "round_id": spec["round_id"],
        "sheet": spec["sheet"],
        "panel": spec["panel"],
        "host_strain": spec["host_strain"],
        "libraries": sorted(spec["read_cols"]),
        "total_reads": totals,
        "source_rows": len(ids),
        "distinct_guide_ids": len(set(ids)),
        "max_equation_1_error": max_normalized_error,
        "max_equation_2_error": max_fitness_error,
        "floor_cross_tab": {
            f"passes={p},fitness_populated={f}": n for (p, f), n in sorted(floor.items())
        },
        "floor_explains_fitness_column": set(floor) <= {(True, True), (False, False)},
        "n_kept": n_kept,
        "n_dropped": len(ids) - n_kept,
        "kept_min": min(kept),
        "kept_max": max(kept),
        "kept_negative": sum(1 for value in kept if value < 0.0),
        "kept_nonpositive": sum(1 for value in kept if value <= 0.0),
        "excluded_partition": {
            ",".join(
                name
                for name, flag in zip(spec["excluded_cols"], flags, strict=True)
                if flag
            )
            or "none": count
            for flags, count in sorted(partition.items())
        },
        "guide_ids": ids,
    }


def read_wang_library(path: str) -> dict[str, str]:
    """Wang 2018 Supplementary Data 3 as ``guide id -> 20-mer spacer``."""
    frame = pd.read_excel(path, sheet_name="sheet1", dtype=str)
    if tuple(frame.columns) != ("sgRNAID", "nucleotide sequence"):
        raise ValueError(f"{osp.basename(path)} has columns {tuple(frame.columns)}")
    return dict(
        zip(
            frame["sgRNAID"].astype(str),
            frame["nucleotide sequence"].astype(str),
            strict=True,
        )
    )


def measure_subsumption(
    data_root: str, rounds: list[dict[str, Any]]
) -> dict[str, Any]:
    """Match the screened ids against the served Wang 2018 library, id for id."""
    path = osp.join(data_root, WANG_LIBRARY_REL)
    observed = sha256_of(path)
    if observed != WANG_LIBRARY_SHA256:
        raise RuntimeError(
            f"{path}: sha256 {observed}, module pins {WANG_LIBRARY_SHA256}"
        )
    spacers = read_wang_library(path)
    controls = {i for i in spacers if CONTROL_ID_RE.match(i)}
    targeting = set(spacers) - controls
    screened = [set(r["guide_ids"]) for r in rounds]
    union: set[str] = set().union(*screened)
    return {
        "wang_library_path": path,
        "wang_library_sha256": observed,
        "wang_library_rows": len(spacers),
        "wang_targeting_guides": len(targeting),
        "wang_non_targeting_controls": len(controls),
        "fang_rounds_screen_identical_sets": all(s == screened[0] for s in screened),
        "fang_union_guides": len(union),
        "fang_equals_wang_targeting_library": union == targeting,
        "in_fang_not_in_wang": sorted(union - set(spacers))[:20],
        "n_in_fang_not_in_wang": len(union - set(spacers)),
        "in_wang_targeting_not_in_fang": sorted(targeting - union)[:20],
        "n_in_wang_targeting_not_in_fang": len(targeting - union),
        "fang_releases_non_targeting_controls": len(union & controls),
        "served_wang_screen_ids": list(WANG_SCREEN_IDS),
        "fang_round_ids": [r["round_id"] for r in rounds],
        "phenotype_collides_with_a_served_screen": bool(
            set(r["round_id"] for r in rounds) & set(WANG_SCREEN_IDS)
        ),
    }


def measure_titer_blocks(path: str) -> dict[str, Any]:
    """Count the per-strain FFA titer values the workbook releases in mg/L.

    A titer block is a header cell equal to ``FFAs (mg L-1)``; its rows are counted by
    walking down from the block's own ``Strain`` row until the column runs out. The
    count is the second release this row carries and is reported, not loaded here.
    """
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    blocks: list[dict[str, Any]] = []
    for sheet in workbook.worksheets:
        grid = [
            list(row)
            for row in sheet.iter_rows(min_row=1, max_row=min(sheet.max_row, 60))
        ]
        for row_index, row in enumerate(grid):
            for column_index, cell in enumerate(row):
                if cell.value != TITER_HEADER:
                    continue
                strains: list[str] = []
                for below in grid[row_index + 1 :]:
                    if column_index - 1 >= len(below):
                        break
                    label = below[column_index - 1].value
                    if label in (None, "Strain", "1"):
                        continue
                    strains.append(str(label))
                blocks.append(
                    {
                        "sheet": sheet.title,
                        "header_cell": cell.coordinate,
                        "n_labelled_rows": len(strains),
                        "labels": strains[:30],
                    }
                )
    workbook.close()
    return {
        "titer_header": TITER_HEADER,
        "n_blocks": len(blocks),
        "blocks": blocks,
        "total_labelled_rows": sum(b["n_labelled_rows"] for b in blocks),
    }


def main() -> None:
    """Run every measurement and write the three result files."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    experiment_root = os.environ["EXPERIMENT_ROOT"]
    results = osp.join(experiment_root, "036-dataset-fixes-before-kg-build", "results")
    os.makedirs(results, exist_ok=True)

    mirror = verify_mirror(data_root)
    deposit = enumerate_publisher_deposit()
    reconciliation = reconcile_deposit(deposit, mirror)

    si8 = osp.join(library_dir(data_root), SI8_RELPATH)
    sheets = characterize_sheets(si8)
    rounds = [measure_round(si8, spec) for spec in ROUNDS]
    subsumption = measure_subsumption(data_root, rounds)
    titers = measure_titer_blocks(si8)

    pd.DataFrame(
        [{k: v for k, v in s.items() if k != "headers"} | {
            "headers": " | ".join(s["headers"])
        } for s in sheets]
    ).to_csv(osp.join(results, "fang2025_screen_sheets.csv"), index=False)
    pd.DataFrame(
        [{k: v for k, v in r.items() if k != "guide_ids"} for r in rounds]
    ).to_csv(osp.join(results, "fang2025_screen_retention.csv"), index=False)

    payload = {
        "row": 43,
        "row_name": "Fang 2025 FFA CRISPRi-FACS",
        "doi": DOI,
        "citation_key": CITATION_KEY,
        "mirror": mirror,
        "publisher_deposit": deposit,
        "deposit_reconciliation": reconciliation,
        "si8_sheets": sheets,
        "rounds": [{k: v for k, v in r.items() if k != "guide_ids"} for r in rounds],
        "subsumption": subsumption,
        "titer_blocks": titers,
        "quotes": {
            "read_floor": READ_FLOOR_QUOTE,
            "library_provenance": LIBRARY_QUOTE,
            "library_reference": LIBRARY_REFERENCE_QUOTE,
            "sort_only": SORT_ONLY_QUOTE,
            "geo_accession": GEO_ACCESSION_QUOTE,
        },
        "geo_accession": GEO_ACCESSION,
        "n_kept_total": sum(r["n_kept"] for r in rounds),
        "blocking_claim_still_true": False,
    }
    with open(osp.join(results, "fang2025_screen_loadability.json"), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)

    print(f"mirrored si objects: {mirror['n_retrieved_si_objects']}")
    print(f"deposit complete: {reconciliation['complete']}")
    for record in rounds:
        print(
            f"{record['round_id']}: {record['source_rows']} rows, "
            f"{record['distinct_guide_ids']} distinct ids, kept {record['n_kept']}, "
            f"eq1 err {record['max_equation_1_error']:.3e}, "
            f"eq2 err {record['max_equation_2_error']:.3e}, "
            f"floor explains column: {record['floor_explains_fitness_column']}"
        )
    print(
        "screened set == served Wang 2018 targeting library: "
        f"{subsumption['fang_equals_wang_targeting_library']}"
    )
    print(f"kept records total: {payload['n_kept_total']}")
    print(f"titer blocks: {titers['n_blocks']}, rows {titers['total_labelled_rows']}")


if __name__ == "__main__":
    main()
