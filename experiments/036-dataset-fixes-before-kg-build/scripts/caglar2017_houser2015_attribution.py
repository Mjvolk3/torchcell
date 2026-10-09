# experiments/036-dataset-fixes-before-kg-build/scripts/caglar2017_houser2015_attribution.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.caglar2017_houser2015_attribution]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/caglar2017_houser2015_attribution
"""Measure the Houser 2015 re-release inside Caglar 2017, and the attribution that fixes it (#771).

#771 measured that 27 of Caglar 2017's 152 mRNA records and 27 of its 105 protein records
are Houser 2015's glucose time course, released again. Houser 2015 is not mirrored, so it
is an attribution gap rather than a duplication: before the fix those 54 records asserted
Caglar 2017 as the source of measurements Houser 2015 published. This script measures, on
the pinned raw bytes and the rebuilt dev stores:

1. **The overlap, from Table S1 joined to the two omics tables.** How many samples carry
   ``experiment == 'glucose_time_course'``, and how many of those are columns of Table S2
   (mRNA) and Table S3 (protein). It also counts the 9 repeated-time-course samples that
   are in neither table, so the ledger is complete.
2. **The attribution the stores now carry**, read off each record's ``publication.doi``:
   per dataset, how many records cite Houser 2015 and how many cite Caglar 2017, against
   the Table S1 split.
3. **Every attribution quote re-read from the pinned OCR**, because a hash pin does not
   make a transcription verbatim (#758). The three sentences are the Results deferral,
   reference 10 itself, and the Data availability split.
4. **Whether Houser 2015 is mirrored**: the `torchcell-library` and `torchcell-raw` keys
   are looked for on disk. If it ever is mirrored, the attribution becomes a true
   duplication question and the superset rule applies, which is why this is measured
   rather than asserted.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/caglar2017_houser2015_attribution.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/caglar2017_houser2015_attribution.py
"""

import hashlib
import json
import os
import os.path as osp
from collections import Counter
from typing import Any

import pandas as pd
from dotenv import load_dotenv

import torchcell.datasets.ecoli.caglar2017 as c

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"
#: Dataset class name -> dev-tree slug.
FAMILIES = {
    "RnaseqCaglar2017Dataset": ("rnaseq_caglar2017", "S2"),
    "ProteomeCaglar2017Dataset": ("proteome_caglar2017", "S3"),
}
#: Houser 2015's mirror keys, if it is ever mirrored (it is not today).
HOUSER2015_CANDIDATE_KEYS = ("houserControlledMeasurementComparative2015", "houser2015")


def _sha256(path: str) -> str:
    """Hex sha256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def released_overlap(raw_dir: str) -> dict[str, Any]:
    """The glucose-time-course samples, and which omics tables carry them as columns."""
    sheet_path = osp.join(raw_dir, c.SI_TABLES["S1"][0])
    frame = pd.read_csv(sheet_path, dtype=str, keep_default_na=False)
    experiment = frame[c.COL_EXPERIMENT]
    sample = frame[c.COL_SAMPLE]
    time_course = set(sample[experiment == c.HOUSER2015_EXPERIMENT])
    repeated = sorted(
        sample[experiment.str.startswith(f"{c.HOUSER2015_EXPERIMENT} (repeated")]
    )
    out: dict[str, Any] = {
        "table_s1_rows": int(len(frame)),
        "experiment_values": dict(sorted(Counter(experiment).items())),
        "glucose_time_course_samples": len(time_course),
        "repeated_time_course_samples": repeated,
    }
    for table in ("S2", "S3"):
        columns = set(
            pd.read_csv(osp.join(raw_dir, c.SI_TABLES[table][0]), nrows=0).columns
        )
        out[f"table_{table.lower()}"] = {
            "sample_columns": len(columns & set(sample)),
            "glucose_time_course_columns": len(columns & time_course),
        }
    return out


def store_attribution(root: str, cls: Any) -> dict[str, Any]:
    """Per-record ``publication.doi`` of one rebuilt dev store."""
    dataset = cls(root=root)
    dois: Counter[str] = Counter()
    for index in range(len(dataset)):
        publication = dataset[index]["publication"]
        dump = (
            publication.model_dump()
            if hasattr(publication, "model_dump")
            else publication
        )
        dois[str(dump["doi"])] += 1
    records = len(dataset)
    dataset.close_lmdb()
    ledger = json.loads(
        open(osp.join(root, "preprocess", "source_study_attribution.json")).read()
    )
    return {
        "root": root,
        "records": records,
        "records_by_publication_doi": dict(sorted(dois.items())),
        "ledger_samples_by_study": ledger["samples_by_study"],
        "ledger_rows": len(ledger["samples"]),
    }


def main() -> None:
    """Measure the released overlap and the stores' attribution, then write the JSON."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    library = osp.join(data_root, "torchcell-library")
    raw_dir = str(c.raw_mirror_dir(data_root) / "data")

    quotes = {}
    for name in c.HOUSER2015_EVIDENCE:
        value = getattr(c, name)
        path = osp.join(
            library, value.provenance.citation_key, value.provenance.source_uri
        )
        quotes[name] = {
            "sha256_matches_the_pin": _sha256(path) == value.provenance.sha256,
            "quote_found_verbatim": value.quote in open(path, encoding="utf-8").read(),
            "page": value.provenance.page,
            "quote": value.quote,
        }

    result: dict[str, Any] = {
        "released_overlap": released_overlap(raw_dir),
        "dev_stores": {
            name: store_attribution(
                osp.join(data_root, "data/torchcell", slug), c.DATASET_CLASSES[family]
            )
            for name, (slug, _) in FAMILIES.items()
            for family in ["rnaseq" if name.startswith("Rnaseq") else "proteome"]
        },
        "attribution_quotes": quotes,
        "houser2015": {
            "doi": c.HOUSER2015_DOI,
            "pubmed_id": c.HOUSER2015_PUBMED_ID,
            "is_mirrored_flag": c.HOUSER2015_IS_MIRRORED,
            "library_keys_on_disk": [
                key
                for key in HOUSER2015_CANDIDATE_KEYS
                if osp.isdir(osp.join(library, key))
            ],
            "raw_keys_on_disk": [
                key
                for key in HOUSER2015_CANDIDATE_KEYS
                if osp.isdir(osp.join(data_root, "torchcell-raw", key))
            ],
        },
    }
    overlap = result["released_overlap"]
    result["verdict"] = {
        "every_glucose_time_course_sample_is_in_both_omics_tables": (
            overlap["table_s2"]["glucose_time_course_columns"]
            == overlap["table_s3"]["glucose_time_course_columns"]
            == overlap["glucose_time_course_samples"]
        ),
        "every_record_cites_one_of_the_two_studies": all(
            set(store["records_by_publication_doi"]) <= {c.PAPER_DOI, c.HOUSER2015_DOI}
            for store in result["dev_stores"].values()
        ),
        "the_houser_record_count_equals_the_table_s1_split": all(
            store["records_by_publication_doi"].get(c.HOUSER2015_DOI, 0)
            == store["ledger_samples_by_study"].get("houser2015", 0)
            for store in result["dev_stores"].values()
        ),
        "every_attribution_quote_is_verbatim_in_the_pinned_bytes": all(
            q["sha256_matches_the_pin"] and q["quote_found_verbatim"]
            for q in quotes.values()
        ),
        "houser2015_is_still_unmirrored": (
            not result["houser2015"]["library_keys_on_disk"]
            and not result["houser2015"]["raw_keys_on_disk"]
        ),
    }

    os.makedirs(RESULTS, exist_ok=True)
    out_path = osp.join(RESULTS, "caglar2017_houser2015_attribution.json")
    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
