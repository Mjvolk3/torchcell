# experiments/015-spell/scripts/spell_allele_reuse
# [[experiments.015-spell.scripts.spell_allele_reuse]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/015-spell/scripts/spell_allele_reuse

"""
Temperature-sensitive and DAmP alleles of Costanzo 2016 and Kuzmin 2020, and
where the same alleles or genes turn up in SPELL column headers.

Allele sources:
    Costanzo 2016  strain_ids_and_single_mutant_fitness.xlsx (Data File S1):
                   strain ids ending `_tsq<n>` / `_tsa<n>` (temperature-sensitive,
                   query / array) and `_damp<n>` (DAmP).
    Kuzmin 2020    Table S1: array strain ids ending `_tsa<n>`. Its query
                   strains are all deletions and it has no DAmP strains.

A SPELL header REUSES AN ALLELE when it contains the allele name as a whole
token (`cdc28-4`), and it NAMES A NON-DELETION ALLELE OF A SHARED GENE when it
contains a `gene-suffix` token for a gene that has a temperature-sensitive or
DAmP allele in either source, without that exact allele name. Both are read
from header text; neither was checked against a paper's strain table.

Outputs (experiments/015-spell/results/):
    sga_conditional_alleles.csv   one row per (source, strain id)
    spell_allele_reuse.csv        one row per SPELL condition with a match

Usage:
    python experiments/015-spell/scripts/spell_allele_reuse.py
"""

import glob
import hashlib
import os
import os.path as osp
import re

import openpyxl
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

from spell_knockout_coverage import GFF_PATH, SPELL_DIR, load_gene_names

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "015-spell", "results")

# The Costanzo raw zip under $DATA_ROOT/data/torchcell/smf_costanzo2016/raw is
# truncated on this machine, so the strain table is read from an older local
# copy of Data File S1. It is pinned by sha256 here and has NOT been compared
# with the copy the loader extracts on the build machine.
COSTANZO_STRAIN_TABLE = osp.expanduser(
    "~/Documents/projects/data/scerevisiae/costanzo2016/raw/"
    "strain_ids_and_single_mutant_fitness.xlsx"
)
COSTANZO_STRAIN_TABLE_SHA256 = (
    "3b9d3351cdde8ee90832797193a1e8838be3ce6778b3761a9e57db2d3a6a9ccc"
)
KUZMIN_TABLE_S1 = osp.join(
    DATA_ROOT, "data/torchcell/smf_kuzmin2020/raw/aaz5667-Table-S1.xlsx"
)

ALLELE_CLASS = {"tsq": "ts", "tsa": "ts", "damp": "damp"}
ALLELE_TOKEN = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{3}[0-9]{1,3}-[A-Za-z0-9]+)(?![A-Za-z0-9])")


class ConditionalAllele(BaseModel):
    source: str
    strain_id: str
    systematic_name: str
    allele_name: str
    allele_class: str  # "ts" or "damp"


class AlleleReuse(BaseModel):
    study_name: str
    pcl_filename: str
    condition_name: str
    token: str
    systematic_name: str
    match: str  # "same_allele" or "same_gene_other_allele"
    sga_allele_classes: str
    sga_sources: str


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strain_suffix(strain_id: str) -> str:
    return re.sub(r"\d+$", "", strain_id.rsplit("_", 1)[1])


def costanzo_alleles() -> list[ConditionalAllele]:
    assert sha256_file(COSTANZO_STRAIN_TABLE) == COSTANZO_STRAIN_TABLE_SHA256
    table = pd.read_excel(COSTANZO_STRAIN_TABLE)
    alleles = []
    for row in table.itertuples(index=False):
        suffix = strain_suffix(row[0])
        if suffix in ALLELE_CLASS:
            alleles.append(
                ConditionalAllele(
                    source="costanzo2016",
                    strain_id=row[0],
                    systematic_name=row[1],
                    allele_name=str(row[2]),
                    allele_class=ALLELE_CLASS[suffix],
                )
            )
    return alleles


def kuzmin_alleles() -> list[ConditionalAllele]:
    sheet = openpyxl.load_workbook(KUZMIN_TABLE_S1, read_only=True)["Sheet1"]
    strains: dict[str, str] = {}
    for index, row in enumerate(sheet.iter_rows(values_only=True)):
        if index < 2:
            continue
        for strain_id, allele in ((row[0], row[1]), (row[2], row[3])):
            if "+" not in strain_id:
                strains[strain_id] = allele
    return [
        ConditionalAllele(
            source="kuzmin2020",
            strain_id=strain_id,
            systematic_name=strain_id.rsplit("_", 1)[0],
            allele_name=str(allele),
            allele_class=ALLELE_CLASS[strain_suffix(strain_id)],
        )
        for strain_id, allele in sorted(strains.items())
        if strain_suffix(strain_id) in ALLELE_CLASS
    ]


def main() -> None:
    names = load_gene_names(GFF_PATH)
    alleles = costanzo_alleles() + kuzmin_alleles()
    alleles_df = pd.DataFrame([a.model_dump() for a in alleles])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    alleles_path = osp.join(RESULTS_DIR, "sga_conditional_alleles.csv")
    alleles_df.to_csv(alleles_path, index=False)

    for source, sub in alleles_df.groupby("source"):
        for allele_class, cls in sub.groupby("allele_class"):
            print(
                f"{source:13s} {allele_class:4s} strains {len(cls):5d}  "
                f"alleles {cls.allele_name.str.lower().nunique():5d}  "
                f"genes {cls.systematic_name.nunique():5d}"
            )
    ts = alleles_df[alleles_df.allele_class == "ts"]
    damp = alleles_df[alleles_df.allele_class == "damp"]
    print(f"union ts:   alleles {ts.allele_name.str.lower().nunique()}  genes {ts.systematic_name.nunique()}")
    print(f"union damp: alleles {damp.allele_name.str.lower().nunique()}  genes {damp.systematic_name.nunique()}")
    kuzmin_only = set(
        alleles_df[alleles_df.source == "kuzmin2020"].allele_name.str.lower()
    ) - set(alleles_df[alleles_df.source == "costanzo2016"].allele_name.str.lower())
    print(f"kuzmin2020 ts alleles not in costanzo2016: {len(kuzmin_only)}")

    by_allele = alleles_df.assign(key=alleles_df.allele_name.str.lower()).groupby("key")
    allele_info = {
        key: (
            sub.systematic_name.iloc[0],
            "|".join(sorted(set(sub.allele_class))),
            "|".join(sorted(set(sub.source))),
        )
        for key, sub in by_allele
    }
    by_gene = alleles_df.groupby("systematic_name")
    gene_info = {
        gene: ("|".join(sorted(set(sub.allele_class))), "|".join(sorted(set(sub.source))))
        for gene, sub in by_gene
    }

    reuse: list[AlleleReuse] = []
    for pcl_path in sorted(glob.glob(osp.join(SPELL_DIR, "*_PMID_*", "*.pcl"))):
        study_name = osp.basename(osp.dirname(pcl_path))
        with open(pcl_path, encoding="utf-8", errors="replace") as f:
            headers = f.readline().rstrip("\n").split("\t")[3:]
        for header in headers:
            for token in sorted(set(ALLELE_TOKEN.findall(header))):
                key = token.lower()
                gene_part, suffix = key.split("-", 1)
                if suffix in ("del", "delta", "deletion", "null", "ko"):
                    continue
                if key in allele_info:
                    gene, classes, sources = allele_info[key]
                    match = "same_allele"
                elif names.get(gene_part.upper()) in gene_info:
                    gene = names[gene_part.upper()]
                    classes, sources = gene_info[gene]
                    match = "same_gene_other_allele"
                else:
                    continue
                reuse.append(
                    AlleleReuse(
                        study_name=study_name,
                        pcl_filename=osp.basename(pcl_path),
                        condition_name=header,
                        token=token,
                        systematic_name=gene,
                        match=match,
                        sga_allele_classes=classes,
                        sga_sources=sources,
                    )
                )

    reuse_df = pd.DataFrame([r.model_dump() for r in reuse])
    reuse_path = osp.join(RESULTS_DIR, "spell_allele_reuse.csv")
    reuse_df.to_csv(reuse_path, index=False)
    for match, sub in reuse_df.groupby("match"):
        print(
            f"{match:24s} conditions {sub.condition_name.nunique():4d}  "
            f"studies {sub.study_name.nunique():3d}  "
            f"tokens {sub.token.str.lower().nunique():3d}  "
            f"genes {sub.systematic_name.nunique():3d}"
        )
    print(alleles_path)
    print(reuse_path)


if __name__ == "__main__":
    main()
