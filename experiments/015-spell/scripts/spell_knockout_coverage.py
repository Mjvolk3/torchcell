# experiments/015-spell/scripts/spell_knockout_coverage
# [[experiments.015-spell.scripts.spell_knockout_coverage]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/015-spell/scripts/spell_knockout_coverage

"""
Gene-deletion coverage of each SPELL study, estimated from PCL column headers.

A SPELL condition is described only by its free-text column header, so deletion
coverage is an ESTIMATE from text: a header counts as a deletion condition when
it carries a deletion marker (`-del`, a delta sign, `delta`, `deletion`,
`knockout`, `null`, `KO`, or a lowercase gene name with a `D` suffix) attached
to a token that resolves to a yeast gene in the SGD R64-4-1 GFF (systematic
name, standard name, or an alias that names exactly one gene).

It misses deletions written without a marker (`ptc1 vs wt`) and it can count a
marked allele that is not a full deletion (`SIC1Δ3P`). The per-condition file is
written so every call can be audited.

A second, separate count covers headers that are nothing but gene names
(`ade1`, `EFB1/YAL003W`, `dig1, dig2 (haploid)`). Those name a perturbed gene
without saying how it was perturbed: Hughes 2000 labels deletions this way and
Mnaimneh 2004 labels tet-promoter alleles this way. They are reported as
"gene-named" and never added to the deletion count.

Deleted genes are also compared with the 1,484 mutants of Kemmeren 2014
Table S1, the deletion compendium torchcell already serves. Genotypes (the set
of genes deleted in one condition) are compared with Kemmeren 2014 and Sameith
2015 together: a single-gene genotype is covered when its gene is in Kemmeren
Table S1, and any genotype is covered when it occurs in SPELL's own Sameith
2015 folder, which stands in here for the torchcell Sameith loaders.

Outputs (experiments/015-spell/results/):
    spell_knockout_conditions.csv  one row per deletion or gene-named condition
    spell_knockout_coverage.csv    one row per study

Usage:
    python experiments/015-spell/scripts/spell_knockout_coverage.py
"""

import glob
import hashlib
import os
import os.path as osp
import re
from collections import defaultdict
from urllib.parse import unquote

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

SPELL_DIR = osp.join(DATA_ROOT, "data/sgd/spell")
GFF_PATH = osp.join(
    DATA_ROOT,
    "data/sgd/genome/S288C_reference_genome_R64-4-1_20230830",
    "saccharomyces_cerevisiae_R64-4-1_20230830.gff",
)
KEMMEREN_TABLE_S1 = osp.join(
    DATA_ROOT, "data/torchcell/microarray_kemmeren2014/raw/kemmeren2014_table_s1.xlsx"
)
SAMEITH_STUDY = "Sameith_2015_PMID_26700642"
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "015-spell", "results")

GENE_FEATURES = {
    "gene",
    "ncRNA_gene",
    "tRNA_gene",
    "snoRNA_gene",
    "snRNA_gene",
    "rRNA_gene",
    "telomerase_RNA_gene",
    "pseudogene",
    "transposable_element_gene",
    "blocked_reading_frame",
}

GENE_TOKEN = r"[A-Za-z][A-Za-z0-9]{2,9}(?:-[A-Za-z])?"
DELTA = "[Δ∆]"
# Each pattern captures the token the deletion marker is attached to.
MARKER_PATTERNS = [
    re.compile(rf"({GENE_TOKEN})-del\b", re.I),
    re.compile(rf"({GENE_TOKEN}){DELTA}"),
    re.compile(rf"{DELTA}({GENE_TOKEN})"),
    re.compile(rf"({GENE_TOKEN})[ \-_]?[\[(]?delta[\])]?(?![A-Za-z])", re.I),
    re.compile(rf"\bdelta[ \-_]({GENE_TOKEN})", re.I),
    re.compile(rf"({GENE_TOKEN})[ \-_](?:deletion|knockout|null|ko)\b", re.I),
    re.compile(rf"(?:deletion|knockout) of ({GENE_TOKEN})", re.I),
    # hog1D, msn2D: lowercase gene name, capital D. Case-sensitive on purpose.
    re.compile(r"(?<![A-Za-z0-9])([a-z]{3}[0-9]{1,3})D(?![a-z0-9])"),
]


class KnockoutCondition(BaseModel):
    study_name: str
    pcl_filename: str
    condition_index: int
    condition_name: str
    call: str  # "deletion" (marker attached to a gene) or "gene_named"
    genes: list[str]


class StudyCoverage(BaseModel):
    study_name: str
    n_conditions_in_pcl: int
    n_ko_conditions: int
    n_ko_genes: int
    n_ko_genotypes: int
    n_ko_single_genotypes: int
    n_ko_multi_genotypes: int
    n_ko_genotypes_new: int
    n_ko_multi_genotypes_new: int
    n_ko_genes_not_in_kemmeren: int
    n_named_conditions: int
    n_named_genes: int
    n_named_genotypes_new: int
    ko_genes: str
    ko_genes_not_in_kemmeren: str
    named_genes: str


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gene_names(gff_path: str) -> dict[str, str]:
    """Upper-cased name -> systematic name. Ambiguous aliases are dropped."""
    systematic: set[str] = set()
    standard: dict[str, str] = {}
    alias_targets: dict[str, set[str]] = defaultdict(set)
    with open(gff_path) as f:
        for line in f:
            if line.startswith("##FASTA"):
                break
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if fields[2] not in GENE_FEATURES:
                continue
            attrs = dict(a.split("=", 1) for a in fields[8].split(";") if "=" in a)
            orf = attrs["ID"]
            systematic.add(orf.upper())
            if "gene" in attrs:
                standard[unquote(attrs["gene"]).upper()] = orf
            for alias in attrs.get("Alias", "").split(","):
                alias = unquote(alias).strip().upper()
                if re.fullmatch(r"[A-Z][A-Z0-9]{2,9}(-[A-Z])?", alias):
                    alias_targets[alias].add(orf)

    names = {
        alias: next(iter(targets))
        for alias, targets in alias_targets.items()
        if len(targets) == 1
    }
    names.update(standard)
    names.update({orf: orf for orf in systematic})
    return names


def deleted_genes(header: str, names: dict[str, str]) -> list[str]:
    genes: set[str] = set()
    for pattern in MARKER_PATTERNS:
        for match in pattern.finditer(header):
            token = match.group(1).upper()
            if token in names:
                genes.add(names[token])
    return sorted(genes)


def named_genes(header: str, names: dict[str, str]) -> list[str]:
    """Genes of a header that is only gene names, once parentheticals are dropped."""
    text = re.sub(r"\(.*?\)", " ", header).replace('"', " ").replace("'", " ")
    tokens = [t for t in re.split(r"[,/ ]+", text.strip()) if t]
    if not tokens or any(t.upper() not in names for t in tokens):
        return []
    return sorted({names[t.upper()] for t in tokens})


def load_kemmeren_genes(names: dict[str, str]) -> set[str]:
    table = pd.read_excel(KEMMEREN_TABLE_S1)
    return {names[orf.upper()] for orf in table["orf name"].astype(str)}


def main() -> None:
    names = load_gene_names(GFF_PATH)
    kemmeren = load_kemmeren_genes(names)
    print(f"Kemmeren Table S1 sha256: {sha256_file(KEMMEREN_TABLE_S1)}")
    print(f"Kemmeren genes: {len(kemmeren)}")
    print(f"GFF sha256: {sha256_file(GFF_PATH)}")
    print(f"gene names: {len(names)} -> {len(set(names.values()))} genes")

    conditions: list[KnockoutCondition] = []
    by_study: dict[str, tuple[int, list[KnockoutCondition]]] = {}
    study_dirs = sorted(
        d for d in glob.glob(osp.join(SPELL_DIR, "*_PMID_*")) if osp.isdir(d)
    )
    for study_dir in study_dirs:
        study_name = osp.basename(study_dir)
        study_conditions = []
        n_headers = 0
        for pcl_path in sorted(glob.glob(osp.join(study_dir, "*.pcl"))):
            with open(pcl_path, encoding="utf-8", errors="replace") as f:
                headers = f.readline().rstrip("\n").split("\t")[3:]
            n_headers += len(headers)
            for index, header in enumerate(headers):
                genes, call = deleted_genes(header, names), "deletion"
                if not genes:
                    genes, call = named_genes(header, names), "gene_named"
                if genes:
                    study_conditions.append(
                        KnockoutCondition(
                            study_name=study_name,
                            pcl_filename=osp.basename(pcl_path),
                            condition_index=index,
                            condition_name=header,
                            call=call,
                            genes=genes,
                        )
                    )
        conditions.extend(study_conditions)
        by_study[study_name] = (n_headers, study_conditions)

    sameith = {
        tuple(c.genes) for c in by_study[SAMEITH_STUDY][1] if c.call == "deletion"
    }

    def is_new(genotype: tuple[str, ...]) -> bool:
        if genotype in sameith:
            return False
        return len(genotype) > 1 or genotype[0] not in kemmeren

    coverage: list[StudyCoverage] = []
    for study_name, (n_headers, study_conditions) in by_study.items():
        deletions = [c for c in study_conditions if c.call == "deletion"]
        genotypes = {tuple(c.genes) for c in deletions}
        named = [c for c in study_conditions if c.call == "gene_named"]
        ko_genes = sorted({g for c in deletions for g in c.genes})
        named_gene_set = sorted({g for c in named for g in c.genes})
        coverage.append(
            StudyCoverage(
                study_name=study_name,
                n_conditions_in_pcl=n_headers,
                n_ko_conditions=len(deletions),
                n_ko_genes=len(ko_genes),
                n_ko_genotypes=len(genotypes),
                n_ko_single_genotypes=sum(len(g) == 1 for g in genotypes),
                n_ko_multi_genotypes=sum(len(g) > 1 for g in genotypes),
                n_ko_genotypes_new=sum(is_new(g) for g in genotypes),
                n_ko_multi_genotypes_new=sum(
                    is_new(g) and len(g) > 1 for g in genotypes
                ),
                n_ko_genes_not_in_kemmeren=len(set(ko_genes) - kemmeren),
                n_named_conditions=len(named),
                n_named_genes=len(named_gene_set),
                n_named_genotypes_new=sum(
                    is_new(g) for g in {tuple(c.genes) for c in named}
                ),
                ko_genes="|".join(ko_genes),
                ko_genes_not_in_kemmeren="|".join(sorted(set(ko_genes) - kemmeren)),
                named_genes="|".join(named_gene_set),
            )
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    conditions_df = pd.DataFrame(
        [{**c.model_dump(), "genes": "|".join(c.genes)} for c in conditions]
    )
    conditions_path = osp.join(RESULTS_DIR, "spell_knockout_conditions.csv")
    conditions_df.to_csv(conditions_path, index=False)
    coverage_df = pd.DataFrame([c.model_dump() for c in coverage])
    coverage_path = osp.join(RESULTS_DIR, "spell_knockout_coverage.csv")
    coverage_df.to_csv(coverage_path, index=False)

    with_ko = coverage_df[coverage_df.n_ko_genes > 0]
    all_genes = {g for genes in with_ko.ko_genes for g in genes.split("|")}
    print(f"studies:                      {len(coverage_df)}")
    print(f"conditions (PCL headers):     {int(coverage_df.n_conditions_in_pcl.sum())}")
    print(f"studies with a deletion call: {len(with_ko)}")
    print(f"deletion conditions:          {int(coverage_df.n_ko_conditions.sum())}")
    print(f"distinct deleted genes:       {len(all_genes)}")
    print(f"  not in Kemmeren Table S1:   {len(all_genes - kemmeren)}")
    all_genotypes = {
        tuple(c.genes) for c in conditions if c.call == "deletion"
    }
    new_genotypes = {g for g in all_genotypes if is_new(g)}
    print(f"distinct deletion genotypes:  {len(all_genotypes)}")
    print(f"  single-gene:                {sum(len(g) == 1 for g in all_genotypes)}")
    print(f"  multi-gene:                 {sum(len(g) > 1 for g in all_genotypes)}")
    print(f"  not in Kemmeren or Sameith: {len(new_genotypes)}")
    print(f"    single-gene:              {sum(len(g) == 1 for g in new_genotypes)}")
    print(f"    multi-gene:               {sum(len(g) > 1 for g in new_genotypes)}")
    print(f"gene-named conditions:        {int(coverage_df.n_named_conditions.sum())}")
    print(f"studies with gene-named:      {int((coverage_df.n_named_conditions > 0).sum())}")
    print(conditions_path)
    print(coverage_path)


if __name__ == "__main__":
    main()
