# experiments/044-organism-similarity/scripts/04_emapper.py
# [[experiments.044-organism-similarity.scripts.04_emapper]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/044-organism-similarity/scripts/04_emapper.py

"""EC and KEGG KO repertoires from one uniform eggNOG-mapper run.

``gh_04_emapper.slurm`` annotates the concatenated per-gene proteomes (eggNOG 5.0.2,
diamond mode) in one run; this script parses ``all_proteomes.emapper.annotations``:

- per protein: organism key, EC list, KO list, COG category and the eggNOG OG list,
  written to ``$DATA_ROOT/data/organism_similarity/emapper/annotations.parquet`` (the
  Eukaryota-level OG of each protein feeds the single-copy-OG proxy in script 06);
- per organism: EC level 4 (``1.1.1.1``, complete numbers only), EC level 3
  (``1.1.1``), KO and COG-category term counts (number of proteins carrying the term),
  long format in ``results/ec_profiles.parquet``; EC coverage (fraction of per-gene
  proteins with any EC) in ``results/ec_coverage.csv``;
- per organism pair: ``ec4_jaccard``, ``ec3_jaccard`` (set Jaccard), ``ec4_braycurtis``
  (Bray-Curtis DISSIMILARITY of the level-4 count vectors) and ``ko_jaccard``, merged
  into ``results/pairwise_similarity.csv``.

The eggNOG 5.0.2 download record (``fetch_eggnog_db.sh``) is copied to
``results/eggnog_db_provenance.tsv``; the run used emapper 2.1.12 with diamond 2.2.8
(``gh_04_emapper.slurm`` log, job 5901).

Run from the repo root with the torchcell env.
"""

from __future__ import annotations

import os.path as osp
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from orgsim_manifest import RESULTS_DIR, WORK_DIR, load_manifest  # noqa: E402

EMAPPER_DIR = osp.join(WORK_DIR, "emapper")
ANNOT = osp.join(EMAPPER_DIR, "all_proteomes.emapper.annotations")
ANNOT_PARQUET = osp.join(EMAPPER_DIR, "annotations.parquet")
PROFILES = osp.join(RESULTS_DIR, "ec_profiles.parquet")
COVERAGE = osp.join(RESULTS_DIR, "ec_coverage.csv")
PAIRWISE_CSV = osp.join(RESULTS_DIR, "pairwise_similarity.csv")
NEW_COLS = ["ec4_jaccard", "ec3_jaccard", "ec4_braycurtis", "ko_jaccard"]


def split_terms(value: str) -> list[str]:
    """An emapper multi-value cell; ``-`` is empty."""
    if value == "-" or value == "":
        return []
    return [v for v in value.split(",") if v]


def read_annotations() -> pd.DataFrame:
    """The emapper annotation table, one row per annotated protein."""
    header: list[str] | None = None
    rows: list[list[str]] = []
    with open(ANNOT) as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            cols = line.rstrip("\n").split("\t")
            if line.startswith("#query"):
                header = [c.lstrip("#") for c in cols]
                continue
            rows.append(cols)
    assert header is not None, f"no header in {ANNOT}"
    df = pd.DataFrame(rows, columns=header)
    df["org"] = df["query"].str.split("|", n=1).str[0]
    return df


def jaccard(a: set[str], b: set[str]) -> float:
    """Set Jaccard."""
    union = a | b
    return len(a & b) / len(union) if union else np.nan


def braycurtis(a: dict[str, int], b: dict[str, int]) -> float:
    """Bray-Curtis dissimilarity of two count vectors."""
    terms = set(a) | set(b)
    num = sum(abs(a.get(t, 0) - b.get(t, 0)) for t in terms)
    den = sum(a.get(t, 0) + b.get(t, 0) for t in terms)
    return num / den if den else np.nan


def main() -> None:
    shutil.copyfile(
        osp.join(WORK_DIR, "..", "eggnog", "eggnog_db_provenance.tsv"),
        osp.join(RESULTS_DIR, "eggnog_db_provenance.tsv"),
    )
    manifest = load_manifest()
    keys = [o.key for o in manifest.present()]
    n_genes = {o.key: o.prep.n_proteins_prepped for o in manifest.present() if o.prep}
    ann = read_annotations()
    assert set(ann["org"]) <= set(keys), set(ann["org"]) - set(keys)
    ann["ec_list"] = ann["EC"].map(split_terms)
    ann["ko_list"] = ann["KEGG_ko"].map(
        lambda v: [k.removeprefix("ko:") for k in split_terms(v)]
    )
    ann[
        [
            "query",
            "org",
            "seed_ortholog",
            "eggNOG_OGs",
            "max_annot_lvl",
            "COG_category",
            "EC",
            "KEGG_ko",
            "Preferred_name",
        ]
    ].to_parquet(ANNOT_PARQUET)

    records = []
    sets: dict[str, dict[str, set[str]]] = {}
    counts4: dict[str, dict[str, int]] = {}
    coverage = []
    for key in keys:
        sub = ann[ann["org"] == key]
        ec4: dict[str, int] = {}
        ec3: dict[str, int] = {}
        ko: dict[str, int] = {}
        cog: dict[str, int] = {}
        for ecs, kos, c in zip(sub["ec_list"], sub["ko_list"], sub["COG_category"]):
            full = {e for e in ecs if len(e.split(".")) == 4 and "-" not in e}
            for e in full:
                ec4[e] = ec4.get(e, 0) + 1
            for e3 in {
                ".".join(e.split(".")[:3])
                for e in ecs
                if len(e.split(".")) >= 3 and "-" not in e.split(".")[:3]
            }:
                ec3[e3] = ec3.get(e3, 0) + 1
            for k in set(kos):
                ko[k] = ko.get(k, 0) + 1
            if c != "-":
                for letter in set(c):
                    cog[letter] = cog.get(letter, 0) + 1
        for level, d in (("ec4", ec4), ("ec3", ec3), ("ko", ko), ("cog", cog)):
            records += [
                {"org": key, "level": level, "term": t, "n_proteins": n}
                for t, n in sorted(d.items())
            ]
        sets[key] = {"ec4": set(ec4), "ec3": set(ec3), "ko": set(ko)}
        counts4[key] = ec4
        n_ec = int((sub["ec_list"].map(len) > 0).sum())
        coverage.append(
            {
                "org": key,
                "n_proteins": n_genes[key],
                "n_annotated_any": len(sub),
                "n_with_ec": n_ec,
                "ec_coverage": n_ec / n_genes[key],
                "n_ec4_terms": len(ec4),
                "n_ec3_terms": len(ec3),
                "n_ko_terms": len(ko),
            }
        )
    pd.DataFrame(records).to_parquet(PROFILES)
    cov = pd.DataFrame(coverage)
    cov.to_csv(COVERAGE, index=False)
    print(cov.to_string(index=False))

    pw = pd.read_csv(PAIRWISE_CSV)
    pw = pw.drop(columns=[c for c in NEW_COLS if c in pw.columns])
    pw["ec4_jaccard"] = [
        jaccard(sets[i]["ec4"], sets[j]["ec4"]) for i, j in zip(pw.org_i, pw.org_j)
    ]
    pw["ec3_jaccard"] = [
        jaccard(sets[i]["ec3"], sets[j]["ec3"]) for i, j in zip(pw.org_i, pw.org_j)
    ]
    pw["ec4_braycurtis"] = [
        braycurtis(counts4[i], counts4[j]) for i, j in zip(pw.org_i, pw.org_j)
    ]
    pw["ko_jaccard"] = [
        jaccard(sets[i]["ko"], sets[j]["ko"]) for i, j in zip(pw.org_i, pw.org_j)
    ]
    pw.to_csv(PAIRWISE_CSV, index=False)
    print(f"appended {NEW_COLS} to {PAIRWISE_CSV}")


if __name__ == "__main__":
    main()
