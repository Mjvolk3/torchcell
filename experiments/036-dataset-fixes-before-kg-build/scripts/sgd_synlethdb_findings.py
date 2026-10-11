# experiments/036-dataset-fixes-before-kg-build/scripts/sgd_synlethdb_findings.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.sgd_synlethdb_findings]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/sgd_synlethdb_findings
"""Measure the SGD essentiality and SynLethDB findings of #901, #902 and #903.

Reads only pinned bytes and built stores, never the loaders' own summaries:

- the SGD per-gene API dump ``$DATA_ROOT/data/sgd/genome/genes`` (refused off the
  ``SGD_GENES_DIGEST`` pin): every inviable / null / S288C annotation, its allele, its
  condition properties and its note; the (gene, PubMed id) keys the old loader collapsed
  onto one whole-gene deletion, and the keys left under the whole-gene allele rule;
- the S288C single-mutant fitness store ``smf_costanzo2016``: the full-deletion single
  mutants it grew (fitness > 0), against which each essential gene set is checked (the
  #889 L4 rule), split by what the SGD annotations say about the gene;
- ``Yeast_SL.csv`` and ``Yeast_SR.csv`` (sha256-pinned in the loader): per-source row
  counts, distinct PubMed ids per source (the aggregation's attribution column), and the
  SR rows that state a score;
- each of the three dev stores, and with ``--before`` each superseded store: records,
  environments with a gapped or a named medium, temperature gaps, and the SR records
  that state or gap a score.

Writes ``results/sgd_synlethdb_findings.json``. Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/sgd_synlethdb_findings.py \
        [--before gene_essentiality_sgd=<dir> --before syn_leth_db_yeast=<dir> ...]
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from collections import Counter
from typing import Any

from dotenv import find_dotenv, load_dotenv

from torchcell.data import file_sha256
from torchcell.datasets.scerevisiae import sgd
from torchcell.datasets.scerevisiae import synth_leth_db as s
from torchcell.verification.runners import stream_records

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"
STORES = ("gene_essentiality_sgd", "syn_leth_db_yeast", "syn_rescue_db_yeast")
_FULL_DELETION_TYPES = frozenset({"sga_kanmx_deletion", "sga_natmx_deletion"})


def sgd_census(genes_dir: str) -> dict[str, Any]:
    """The annotation-level counts of the pinned SGD dump."""
    sgd.verify_sgd_dump(genes_dir)
    annotations = sgd.inviable_null_annotations(genes_dir)
    kept, dropped = sgd.split_by_allele(annotations)
    collapsed = Counter((a.systematic_name, a.pubmed_id) for a in annotations)
    duplicated = {key: n for key, n in collapsed.items() if n > 1}
    after = Counter((a.systematic_name, a.conditions, a.pubmed_id) for a in kept)
    variant_spelling = [
        a.allele for a in kept if a.allele != f"{a.locus_display_name.lower()}-Δ"
    ]
    return {
        "genes_dir": genes_dir,
        "sgd_genes_digest": sgd.SGD_GENES_DIGEST,
        "n_annotations": len(annotations),
        "n_distinct_annotation_ids": len({a.sgd_annotation_id for a in annotations}),
        "n_whole_gene_deletion": len(kept),
        "n_other_allele": len(dropped),
        "whole_gene_deletion_variant_spellings": variant_spelling,
        "other_alleles": [
            {
                "gene": a.systematic_name,
                "locus": a.locus_display_name,
                "allele": a.allele,
                "pubmed_id": a.pubmed_id,
                "conditions": [f"{k}: {t}" for k, t in a.conditions],
            }
            for a in dropped
        ],
        "old_collapse_duplicate_keys": len(duplicated),
        "old_collapse_extra_records": sum(n - 1 for n in duplicated.values()),
        "old_collapse_duplicates": {
            f"{gene} PMID {pmid}": [
                a.allele
                for a in annotations
                if (a.systematic_name, a.pubmed_id) == (gene, pmid)
            ]
            for gene, pmid in sorted(duplicated)
        },
        "duplicate_keys_after_allele_rule": sum(1 for n in after.values() if n > 1),
        "conditions_all": {
            " + ".join(f"{k}: {t}" for k, t in cond) or "(none)": n
            for cond, n in Counter(a.conditions for a in annotations).most_common()
        },
        "conditions_kept": {
            " + ".join(f"{k}: {t}" for k, t in cond) or "(none)": n
            for cond, n in Counter(a.conditions for a in kept).most_common()
        },
        "experiment_types": dict(Counter(a.experiment for a in annotations)),
        "n_with_note": sum(1 for a in annotations if a.note),
        "_annotations": annotations,
        "_kept": kept,
    }


def viable_deletions(data_root: str) -> dict[str, str]:
    """``gene -> strain id`` of every full-deletion single mutant smf_costanzo2016 grew."""
    out: dict[str, str] = {}
    root = osp.join(data_root, "data/torchcell/smf_costanzo2016")
    for record in stream_records(root):
        perturbations = record["experiment"]["genotype"]["perturbations"]
        fitness = record["experiment"]["phenotype"]["fitness"]
        if len(perturbations) != 1 or fitness is None or not fitness > 0:
            continue
        (p,) = perturbations
        if p["perturbation_type"] in _FULL_DELETION_TYPES:
            out.setdefault(str(p["systematic_gene_name"]), str(p.get("strain_id")))
    return out


def contradiction_census(
    census: dict[str, Any], viable: dict[str, str]
) -> dict[str, Any]:
    """Essential genes grown as viable deletions, before and after, by annotation kind."""
    annotations = census["_annotations"]
    kept = census["_kept"]
    before = {a.systematic_name for a in annotations} & set(viable)
    after = {a.systematic_name for a in kept} & set(viable)
    unconditional = {a.systematic_name for a in kept if not a.conditions} & set(viable)
    by_gene: dict[str, dict[str, Any]] = {}
    for gene in sorted(before):
        mine = [a for a in annotations if a.systematic_name == gene]
        kinds = sorted(
            {
                "other allele"
                if not sgd.is_whole_gene_deletion(a.allele, a.locus_display_name)
                else ("conditional" if a.conditions else "unconditional deletion")
                for a in mine
            }
        )
        by_gene[gene] = {
            "locus": mine[0].locus_display_name,
            "viable_strain": viable[gene],
            "annotation_kinds": kinds,
            "pubmed_ids": sorted({a.pubmed_id for a in mine}),
            "conditions": sorted({f"{k}: {t}" for a in mine for k, t in a.conditions}),
        }
    return {
        "n_viable_full_deletions_smf_costanzo2016": len(viable),
        "n_contradicted_before": len(before),
        "n_contradicted_after": len(after),
        "n_contradicted_after_unconditional_only": len(unconditional),
        "after_conditional_only": sorted(after - unconditional),
        "after_unconditional": sorted(unconditional),
        "by_gene": by_gene,
    }


def synlethdb_census(data_root: str) -> dict[str, Any]:
    """Per-source counts and stated scores of the pinned SL and SR files.

    ``source_pubmed_ids`` splits a multi-paper cell (``24125552;19918932``) into its
    PubMed ids: the number of source studies the aggregation attributes, per file and
    over both files (plan decision 15's ``n_source_studies``).
    """
    out: dict[str, Any] = {}
    pmids: dict[str, set[str]] = {}
    for label, slug, name, sha in (
        ("SL", "syn_leth_db_yeast", s.SL_CSV_NAME, s.SL_CSV_SHA256),
        ("SR", "syn_rescue_db_yeast", s.SR_CSV_NAME, s.SR_CSV_SHA256),
    ):
        path = osp.join(data_root, "data/torchcell", slug, "raw", name)
        observed = file_sha256(path)
        if observed != sha:
            raise ValueError(f"{path}: sha256 {observed} is not the pinned {sha}")
        df = s._read_synlethdb_csv(path)
        scored = df[df["r.statistic_score"].notna()]
        pmids[label] = {
            p.strip() for cell in df["r.pubmed_id"] for p in cell.split(";")
        }
        out[label] = {
            "path": path,
            "sha256": observed,
            "n_rows": len(df),
            "n_distinct_pubmed_ids": int(df["r.pubmed_id"].nunique()),
            "n_source_pubmed_ids": len(pmids[label]),
            "n_multi_pubmed_rows": int(df["r.pubmed_id"].str.contains(";").sum()),
            "rows_by_source": df["r.source"].value_counts().to_dict(),
            "distinct_pubmed_ids_by_source": df.groupby("r.source")["r.pubmed_id"]
            .nunique()
            .to_dict(),
            "n_cell_line_stated": int(df["r.cell_line"].notna().sum()),
            "n_scored": len(scored),
            "scored_by_source": scored["r.source"].value_counts().to_dict(),
            "n_scored_pubmed_ids": int(scored["r.pubmed_id"].nunique()),
            "scored_top_pubmed_ids": scored["r.pubmed_id"]
            .value_counts()
            .head(10)
            .to_dict(),
            "score_min": float(scored["r.statistic_score"].min()),
            "score_max": float(scored["r.statistic_score"].max()),
            "score_by_source": {
                str(source): sorted(set(map(float, group)))[:5]
                for source, group in scored.groupby("r.source")["r.statistic_score"]
            },
        }
    out["n_source_pubmed_ids_sl_or_sr"] = len(pmids["SL"] | pmids["SR"])
    return out


def store_census(root: str) -> dict[str, Any]:
    """Records and environment / score gaps of one built store."""
    n = 0
    media: Counter[str] = Counter()
    temperature: Counter[str] = Counter()
    perturbations: Counter[str] = Counter()
    scores: Counter[str] = Counter()
    keys: Counter[tuple[Any, ...]] = Counter()
    for record in stream_records(root):
        n += 1
        experiment = record["experiment"]
        environment = experiment["environment"]
        gaps = {g["field"]: g["reason"] for g in environment.get("provenance_gaps", [])}
        medium = environment["media"]
        media[
            f"gap:{gaps['media']}" if medium is None else f"named:{medium['name']}"
        ] += 1
        temperature[
            f"gap:{gaps['temperature']}"
            if environment["temperature"] is None
            else f"{environment['temperature']['value']} C"
        ] += 1
        for p in environment.get("perturbations", []):
            perturbations[f"{p['perturbation_type']}:{p.get('description')}"] += 1
        phenotype = experiment["phenotype"]
        for field in (
            "synthetic_rescue_statistic_score",
            "synthetic_lethality_statistic_score",
        ):
            if field in phenotype:
                phenotype_gaps = {g["field"] for g in phenotype["provenance_gaps"]}
                scores[
                    "gapped"
                    if field in phenotype_gaps
                    else ("none" if phenotype[field] is None else "stated")
                ] += 1
        names = tuple(
            sorted(
                p["systematic_gene_name"]
                for p in experiment["genotype"]["perturbations"]
            )
        )
        keys[
            (
                names,
                json.dumps(environment, sort_keys=True),
                str(record["publication"]["pubmed_id"]),
            )
        ] += 1
    return {
        "processed": osp.realpath(osp.join(root, "processed")),
        "n_records": n,
        "media": dict(media.most_common()),
        "temperature": dict(temperature.most_common()),
        "environment_perturbations": dict(perturbations.most_common()),
        "scores": dict(scores),
        "duplicate_gene_environment_publication_keys": sum(
            1 for c in keys.values() if c > 1
        ),
        "duplicate_extra_records": sum(c - 1 for c in keys.values() if c > 1),
    }


def main() -> None:
    """Measure, then write ``results/sgd_synlethdb_findings.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before",
        action="append",
        default=[],
        help="<store>=<dir holding the superseded processed/ and preprocess/>",
    )
    args = parser.parse_args()
    load_dotenv(find_dotenv(usecwd=True))
    data_root = os.environ["DATA_ROOT"]

    census = sgd_census(osp.join(data_root, sgd.SGD_GENES_RELPATH))
    contradictions = contradiction_census(census, viable_deletions(data_root))
    stores: dict[str, Any] = {
        name: {"after": store_census(osp.join(data_root, "data/torchcell", name))}
        for name in STORES
    }
    for spec in args.before:
        name, root = spec.split("=", 1)
        stores[name]["before"] = store_census(root)
    result = {
        "sgd_dump": {k: v for k, v in census.items() if not k.startswith("_")},
        "essential_vs_smf_costanzo2016": contradictions,
        "synlethdb_release": synlethdb_census(data_root),
        "stores": stores,
    }
    os.makedirs(RESULTS, exist_ok=True)
    out = osp.join(RESULTS, "sgd_synlethdb_findings.json")
    with open(out, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, default=str)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
