# experiments/037-yeastract/scripts/yeastract_publication_manifest
# [[experiments.037-yeastract.scripts.yeastract_publication_manifest]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/037-yeastract/scripts/yeastract_publication_manifest

"""
YEASTRACT+ publication manifest.

One row per PubMed id in the S. cerevisiae regulation flat file of the
YEASTRACT+ 2022 release (`YeastractPlus_2022_Scerevisiae_regs_flatfile.tsv.gz`
inside `$DATA_ROOT/data/safe_keeping/YeastractPlus_2022_all_regs.zip`). The flat
file has no header; a row is one piece of evidence that a transcription factor
regulates a target gene, and its fifth column is the PMID the evidence is
credited to. The journal citation, DOI and PMC id are resolved from the PMID
through NCBI E-utilities `esummary`; the raw responses are stored next to the
manifest so the lookup is on record.

Each paper is also checked against the SPELL publication manifest
(experiments/015-spell/results/spell_publications.csv) by PMID, and against
REANALYSES, the papers known to reprocess the data of another paper.

Outputs (experiments/037-yeastract/results/):
    yeastract_pubmed_esummary.json   raw esummary records + retrieval metadata
    yeastract_publications.csv       the manifest, one row per PMID

Usage:
    python experiments/037-yeastract/scripts/yeastract_publication_manifest.py
"""

import csv
import gzip
import hashlib
import io
import json
import os
import os.path as osp
import time
import zipfile
from datetime import UTC, datetime

import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

YEASTRACT_ZIP = osp.join(
    DATA_ROOT, "data", "safe_keeping", "YeastractPlus_2022_all_regs.zip"
)
YEASTRACT_ZIP_SHA256 = (
    "2f2e9f8b3c905ae333814f04161d0ffd2fabf00cc9878c1bd86a8f5e366ebb3e"
)
YEASTRACT_MEMBER = "YeastractPlus_2022_Scerevisiae_regs_flatfile.tsv.gz"
YEASTRACT_MEMBER_SHA256 = (
    "d879c1f4997c4521a68edcad5f8814add6186db4522839507b373b9b1a12f95f"
)
# The file carries no header row. These names are read off the values.
COLUMNS = [
    "tf",
    "tf_name",
    "target",
    "target_name",
    "pmid",
    "sign",
    "evidence",
    "assay",
    "condition_group",
    "condition_subgroup",
    "condition",
]

SPELL_CSV = osp.join(EXPERIMENT_ROOT, "015-spell", "results", "spell_publications.csv")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "037-yeastract", "results")
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
ESUMMARY_BATCH = 200
NAME_SUFFIXES = {"Jr", "Sr", "2nd", "3rd"}


class Reanalysis(BaseModel):
    """A paper known to reprocess another paper's data, with the sentence that says so."""

    source_pmid: str
    quote: str
    quote_source: str


# A paper that reprocesses another paper's measurements, so its YEASTRACT rows
# trace to that paper's data. Each entry carries the verbatim sentence that says
# so. The list holds only what has been read; it is not a survey of all papers.
REANALYSES = {
    "20385592": Reanalysis(
        source_pmid="17417638",
        quote=(
            "Recently, Hu and colleagues published a comprehensive study covering "
            "269 TF knockout mutants for the yeast Saccharomyces cerevisiae. "
            "However, the information that can be extracted from this valuable "
            "dataset is limited by the method employed to process the microarray "
            "data. Here, we present a reanalysis of the original data using "
            "improved statistical techniques freely available from the "
            "BioConductor project."
        ),
        quote_source=(
            "PubMed abstract of PMID 20385592, efetch db=pubmed rettype=abstract, "
            "retrieved 2026-10-04"
        ),
    )
}


class PubmedRecord(BaseModel):
    """The fields of one PubMed esummary record that the manifest keeps."""

    pmid: str
    doi: str | None
    pmcid: str | None
    title: str
    journal: str
    pubdate: str
    volume: str
    issue: str
    pages: str
    authors: list[str]


class YeastractPublication(BaseModel):
    """One row of the manifest: a PMID, its citation, and its weight in the flat file."""

    pmid: str
    pubmed_found: bool
    first_author: str | None
    year: int | None
    doi: str | None
    pmcid: str | None
    title: str | None
    journal: str | None
    pubdate: str | None
    volume: str | None
    issue: str | None
    pages: str | None
    authors: str | None
    n_rows: int
    n_tfs: int
    n_targets: int
    n_pairs: int
    n_direct: int
    n_indirect: int
    n_evidence_na: int
    n_positive: int
    n_negative: int
    n_sign_na: int
    n_assays: int
    top_assay: str
    n_conditions: int
    top_condition_group: str
    in_spell: bool
    spell_study: str | None
    spell_geo: str | None
    reanalysis_of_pmid: str | None
    reanalysis_of_spell_study: str | None
    reanalysis_of_spell_geo: str | None
    pubmed_url: str
    doi_url: str | None
    pmc_url: str | None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_flatfile() -> pd.DataFrame:
    with open(YEASTRACT_ZIP, "rb") as f:
        archive = f.read()
    assert sha256_bytes(archive) == YEASTRACT_ZIP_SHA256, YEASTRACT_ZIP
    member = gzip.decompress(
        zipfile.ZipFile(io.BytesIO(archive)).read(YEASTRACT_MEMBER)
    )
    assert sha256_bytes(member) == YEASTRACT_MEMBER_SHA256, YEASTRACT_MEMBER
    df = pd.read_csv(
        io.BytesIO(member),
        sep="\t",
        header=None,
        names=COLUMNS,
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )
    assert df.pmid.str.fullmatch(r"\d+").all()
    assert not df.duplicated().any()
    assert set(df.evidence) == {"Direct", "Indirect", "N/A"}
    assert set(df.sign) == {"Positive", "Negative", "N/A"}
    return df


def fetch_esummary(pmids: list[str]) -> dict:
    """Raw esummary records keyed by PMID, with retrieval metadata.

    A PMID PubMed has no summary for comes back as an `error` record. It is
    kept as returned, so the manifest can report it; it is not dropped.
    """
    records: dict[str, dict] = {}
    for start in range(0, len(pmids), ESUMMARY_BATCH):
        batch = pmids[start : start + ESUMMARY_BATCH]
        response = requests.post(
            ESUMMARY_URL,
            data={"db": "pubmed", "id": ",".join(batch), "retmode": "json"},
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()["result"]
        for pmid in batch:
            records[pmid] = result[pmid]
        time.sleep(0.5)
    return {
        "source_url": ESUMMARY_URL,
        "retrieval_method": "direct_url",
        "retrieval_command": (
            f"POST {ESUMMARY_URL} db=pubmed retmode=json "
            f"id=<{ESUMMARY_BATCH} PMIDs per request>"
        ),
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_pmids": len(pmids),
        "n_without_record": sum("error" in r for r in records.values()),
        "records": records,
    }


def parse_esummary(record: dict) -> PubmedRecord | None:
    if "error" in record:
        return None
    ids = {a["idtype"]: a["value"] for a in record["articleids"]}
    return PubmedRecord(
        pmid=record["uid"],
        doi=ids.get("doi"),
        pmcid=ids.get("pmc"),
        title=record["title"],
        journal=record["source"],
        pubdate=record["pubdate"],
        volume=record["volume"],
        issue=record["issue"],
        pages=record["pages"],
        authors=[a["name"] for a in record["authors"]],
    )


def surname(name: str) -> str:
    """Surname from an esummary name, `Surname Initials` with an optional suffix."""
    parts = name.split(" ")
    if parts[-1] in NAME_SUFFIXES:
        parts = parts[:-1]
    assert len(parts) >= 2, name
    return " ".join(parts[:-1])


def top_value(column: pd.Series) -> str:
    """Most frequent value; ties go to the alphabetically first."""
    counts = column.value_counts()
    return sorted(counts[counts == counts.max()].index)[0]


def build_publication(
    pmid: str, rows: pd.DataFrame, pubmed: PubmedRecord | None, spell: pd.DataFrame
) -> YeastractPublication:
    reanalysis = REANALYSES.get(pmid)
    source = reanalysis.source_pmid if reanalysis else None
    if source is not None:
        assert source in spell.index, source
    return YeastractPublication(
        pmid=pmid,
        pubmed_found=pubmed is not None,
        first_author=surname(pubmed.authors[0]) if pubmed else None,
        year=int(pubmed.pubdate[:4]) if pubmed else None,
        doi=pubmed.doi if pubmed else None,
        pmcid=pubmed.pmcid if pubmed else None,
        title=pubmed.title if pubmed else None,
        journal=pubmed.journal if pubmed else None,
        pubdate=pubmed.pubdate if pubmed else None,
        volume=pubmed.volume if pubmed else None,
        issue=pubmed.issue if pubmed else None,
        pages=pubmed.pages if pubmed else None,
        authors="; ".join(pubmed.authors) if pubmed else None,
        n_rows=len(rows),
        n_tfs=rows.tf.nunique(),
        n_targets=rows.target.nunique(),
        n_pairs=len(rows[["tf", "target"]].drop_duplicates()),
        n_direct=int((rows.evidence == "Direct").sum()),
        n_indirect=int((rows.evidence == "Indirect").sum()),
        n_evidence_na=int((rows.evidence == "N/A").sum()),
        n_positive=int((rows.sign == "Positive").sum()),
        n_negative=int((rows.sign == "Negative").sum()),
        n_sign_na=int((rows.sign == "N/A").sum()),
        n_assays=rows.assay.nunique(),
        top_assay=top_value(rows.assay),
        n_conditions=rows.condition.nunique(),
        top_condition_group=top_value(rows.condition_group),
        in_spell=pmid in spell.index,
        spell_study=spell.study_name.get(pmid),
        spell_geo=spell.geo_accession.get(pmid) or None,
        reanalysis_of_pmid=source,
        reanalysis_of_spell_study=spell.study_name[source] if source else None,
        reanalysis_of_spell_geo=(spell.geo_accession[source] or None)
        if source
        else None,
        pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        doi_url=f"https://doi.org/{pubmed.doi}" if pubmed and pubmed.doi else None,
        pmc_url=(
            f"https://pmc.ncbi.nlm.nih.gov/articles/{pubmed.pmcid}/"
            if pubmed and pubmed.pmcid
            else None
        ),
    )


def main() -> None:
    flat = read_flatfile()
    pmids = sorted(flat.pmid.unique())
    print(f"flat-file rows:      {len(flat)}")
    print(f"distinct PMIDs:      {len(pmids)}")

    spell = pd.read_csv(SPELL_CSV, dtype=str, keep_default_na=False).set_index("pmid")
    assert spell.index.is_unique

    esummary = fetch_esummary(pmids)
    esummary["yeastract_zip"] = osp.basename(YEASTRACT_ZIP)
    esummary["yeastract_zip_sha256"] = YEASTRACT_ZIP_SHA256
    esummary["yeastract_member"] = YEASTRACT_MEMBER
    esummary["yeastract_member_sha256"] = YEASTRACT_MEMBER_SHA256
    os.makedirs(RESULTS_DIR, exist_ok=True)
    esummary_path = osp.join(RESULTS_DIR, "yeastract_pubmed_esummary.json")
    with open(esummary_path, "w") as f:
        json.dump(esummary, f, indent=1, sort_keys=True)

    publications = [
        build_publication(pmid, rows, parse_esummary(esummary["records"][pmid]), spell)
        for pmid, rows in flat.groupby("pmid", sort=True)
    ]
    df = pd.DataFrame([p.model_dump() for p in publications])
    df["year"] = df.year.astype("Int64")
    assert df.n_rows.sum() == len(flat)
    csv_path = osp.join(RESULTS_DIR, "yeastract_publications.csv")
    df.to_csv(csv_path, index=False)

    print(f"without PubMed record: {int((~df.pubmed_found).sum())}")
    print(f"with DOI:            {int(df.doi.notna().sum())}")
    print(f"with PMC id:         {int(df.pmcid.notna().sum())}")
    print(f"in SPELL by PMID:    {int(df.in_spell.sum())}")
    print(f"reanalysis of SPELL: {int(df.reanalysis_of_spell_study.notna().sum())}")
    print(esummary_path)
    print(csv_path)


if __name__ == "__main__":
    main()
