# experiments/015-spell/scripts/spell_publication_manifest
# [[experiments.015-spell.scripts.spell_publication_manifest]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/015-spell/scripts/spell_publication_manifest

"""
SPELL publication manifest.

One row per study folder in the SGD SPELL archive
(`$DATA_ROOT/data/sgd/spell/<FirstAuthor>_<year>_PMID_<pmid>/`). The README in
each folder gives the PMID, the GEO accession (or `N/A`) and the PCL dataset
table. The DOI, PMC id and journal citation are not in the README, so they are
resolved from the PMID through NCBI E-utilities `esummary`; the raw responses
are stored next to the manifest so the lookup is on record.

Outputs (experiments/015-spell/results/):
    spell_pubmed_esummary.json   raw esummary records + retrieval metadata
    spell_publications.csv       the manifest, one row per study

Usage:
    python experiments/015-spell/scripts/spell_publication_manifest.py
"""

import glob
import hashlib
import json
import os
import os.path as osp
import re
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

SPELL_DIR = osp.join(DATA_ROOT, "data/sgd/spell")
SPELL_ARCHIVE = osp.join(SPELL_DIR, "all_spell_datasets.tar.gz")
SPELL_ARCHIVE_URL = (
    "http://sgd-archive.yeastgenome.org/expression/microarray/all_spell_datasets.tar.gz"
)
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "015-spell", "results")
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
ESUMMARY_BATCH = 200

PCL_HEADER = "PCL filename\tshort description\t# conditions\ttags\t# channels"


class SpellDataset(BaseModel):
    pcl_filename: str
    short_description: str
    n_conditions: int
    tags: list[str]
    n_channels: int


class SpellReadme(BaseModel):
    study_name: str
    first_author: str
    folder_year: int
    pmid: str
    geo_accession: str | None
    geo_id_field: str
    geo_series_in_pcl_names: list[str]
    datasets: list[SpellDataset]


class PubmedRecord(BaseModel):
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


class SpellPublication(BaseModel):
    study_name: str
    first_author: str
    folder_year: int
    pmid: str
    doi: str | None
    pmcid: str | None
    geo_accession: str | None
    geo_id_field: str
    geo_series_in_pcl_names: str
    has_geo: bool
    title: str
    journal: str
    pubdate: str
    volume: str
    issue: str
    pages: str
    authors: str
    n_datasets: int
    n_conditions: int
    channels: str
    tags: str
    pcl_files: str
    url: str
    pubmed_url: str
    doi_url: str | None
    pmc_url: str | None
    geo_url: str | None


def parse_readme(study_dir: str) -> SpellReadme:
    study_name = osp.basename(study_dir)
    name_match = re.fullmatch(r"(.+)_(\d{4})_PMID_(\d+)", study_name)
    assert name_match is not None, study_name
    (readme_path,) = glob.glob(osp.join(study_dir, "*.README"))
    with open(readme_path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    pmid_match = re.search(r"^PMID:\s*(\d+)\s*$", text, re.M)
    assert pmid_match is not None, readme_path
    assert pmid_match.group(1) == name_match.group(3), readme_path

    # The field holds a bare series id in some READMEs and a PCL filename that
    # starts with one in others (`GSE9136.final.pcl`, `GSE34330GPL8154.sfp.pcl`).
    geo_match = re.search(r"^GEO ID:[ \t]*(\S+)[ \t]*$", text, re.M)
    assert geo_match is not None, readme_path
    geo_value = geo_match.group(1)
    series_match = re.match(r"GSE\d+", geo_value)
    assert geo_value == "N/A" or series_match is not None, geo_value

    # One dataset row per PCL file, each followed by its own
    # `File last modified:` line.
    table = text.split(PCL_HEADER, 1)[1]
    datasets = []
    for line in table.splitlines():
        if not line.strip() or line.startswith("File last modified:"):
            continue
        pcl, description, n_conditions, tags, n_channels = line.split("\t")
        datasets.append(
            SpellDataset(
                pcl_filename=pcl.strip(),
                short_description=description.strip(),
                n_conditions=int(n_conditions),
                tags=[t.strip() for t in tags.split("|") if t.strip()],
                n_channels=int(n_channels),
            )
        )
    assert len(datasets) > 0, readme_path

    return SpellReadme(
        study_name=study_name,
        first_author=name_match.group(1).replace("_", " "),
        folder_year=int(name_match.group(2)),
        pmid=pmid_match.group(1),
        geo_accession=series_match.group(0) if series_match else None,
        geo_id_field=geo_value,
        geo_series_in_pcl_names=sorted(
            {g for d in datasets for g in re.findall(r"GSE\d+", d.pcl_filename)}
        ),
        datasets=datasets,
    )


def fetch_esummary(pmids: list[str]) -> dict:
    """Raw esummary records keyed by PMID, with retrieval metadata."""
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
            assert "error" not in result[pmid], (pmid, result[pmid])
            records[pmid] = result[pmid]
        time.sleep(0.5)
    return {
        "source_url": ESUMMARY_URL,
        "retrieval_method": "direct_url",
        "retrieval_command": (
            f"POST {ESUMMARY_URL} db=pubmed retmode=json "
            f"id=<{ESUMMARY_BATCH} PMIDs per request>"
        ),
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_pmids": len(pmids),
        "records": records,
    }


def parse_esummary(record: dict) -> PubmedRecord:
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


def build_publication(readme: SpellReadme, pubmed: PubmedRecord) -> SpellPublication:
    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{readme.pmid}/"
    doi_url = f"https://doi.org/{pubmed.doi}" if pubmed.doi else None
    tags = sorted({t for d in readme.datasets for t in d.tags})
    channels = sorted({d.n_channels for d in readme.datasets})
    return SpellPublication(
        study_name=readme.study_name,
        first_author=readme.first_author,
        folder_year=readme.folder_year,
        pmid=readme.pmid,
        doi=pubmed.doi,
        pmcid=pubmed.pmcid,
        geo_accession=readme.geo_accession,
        geo_id_field=readme.geo_id_field,
        geo_series_in_pcl_names="|".join(readme.geo_series_in_pcl_names),
        has_geo=readme.geo_accession is not None,
        title=pubmed.title,
        journal=pubmed.journal,
        pubdate=pubmed.pubdate,
        volume=pubmed.volume,
        issue=pubmed.issue,
        pages=pubmed.pages,
        authors="; ".join(pubmed.authors),
        n_datasets=len(readme.datasets),
        n_conditions=sum(d.n_conditions for d in readme.datasets),
        channels="|".join(str(c) for c in channels),
        tags="|".join(tags),
        pcl_files="|".join(d.pcl_filename for d in readme.datasets),
        url=doi_url if doi_url else pubmed_url,
        pubmed_url=pubmed_url,
        doi_url=doi_url,
        pmc_url=(
            f"https://pmc.ncbi.nlm.nih.gov/articles/{pubmed.pmcid}/"
            if pubmed.pmcid
            else None
        ),
        geo_url=(
            f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={readme.geo_accession}"
            if readme.geo_accession
            else None
        ),
    )


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    study_dirs = sorted(
        d for d in glob.glob(osp.join(SPELL_DIR, "*_PMID_*")) if osp.isdir(d)
    )
    readmes = [parse_readme(d) for d in study_dirs]
    pmids = [r.pmid for r in readmes]
    assert len(set(pmids)) == len(pmids)
    print(f"SPELL study folders: {len(readmes)}")

    esummary = fetch_esummary(pmids)
    esummary["spell_archive_url"] = SPELL_ARCHIVE_URL
    esummary["spell_archive_sha256"] = sha256_file(SPELL_ARCHIVE)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    esummary_path = osp.join(RESULTS_DIR, "spell_pubmed_esummary.json")
    with open(esummary_path, "w") as f:
        json.dump(esummary, f, indent=1, sort_keys=True)

    publications = [
        build_publication(r, parse_esummary(esummary["records"][r.pmid]))
        for r in readmes
    ]
    df = pd.DataFrame([p.model_dump() for p in publications])
    csv_path = osp.join(RESULTS_DIR, "spell_publications.csv")
    df.to_csv(csv_path, index=False)

    print(f"with DOI:            {int(df.doi.notna().sum())}")
    print(f"with PMC id:         {int(df.pmcid.notna().sum())}")
    print(f"with GEO accession:  {int(df.has_geo.sum())}")
    print(f"without GEO:         {int((~df.has_geo).sum())}")
    print(f"datasets (PCL):      {int(df.n_datasets.sum())}")
    print(f"conditions:          {int(df.n_conditions.sum())}")
    print(f"archive sha256:      {esummary['spell_archive_sha256']}")
    print(esummary_path)
    print(csv_path)


if __name__ == "__main__":
    main()
