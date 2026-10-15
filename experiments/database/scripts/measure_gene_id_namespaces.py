# experiments/database/scripts/measure_gene_id_namespaces.py
# [[experiments.database.scripts.measure_gene_id_namespaces]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/measure_gene_id_namespaces
"""Measure the gene identifier spaces the served stores actually carry (P1, ontology-unify).

``GeneIdNamespace`` (``torchcell/datamodels/schema.py``) is the closed vocabulary of
host identifier spaces a ``GenePerturbation.systematic_gene_name`` can be written in.
Its members are measured, never assumed (constitution, principle 5): this script opens
every served dev store read-only, reads every perturbation of every record, and
classifies the stored identifier by form. It also measures the locus-tag forms of the
two yeast assembly sets the genomes tier deposits (S288C R64 and CEN.PK113-7D), so the
yeast patterns in ``GENE_ID_NAMESPACE_PATTERNS`` come from the same deposited bytes the
bacterial patterns did.

Inputs: the newest release snapshot (``database/releases/<release>.json``) names the
served datasets; each loader's default ``root`` under ``$DATA_ROOT`` is its dev store.
Outputs, under ``experiments/database/results/``:

- ``gene_id_namespace_vocabulary.json``: per dataset, per perturbation type, the count
  of identifiers in each form, with up to five examples, plus the reference strain the
  store names and the measured yeast locus-tag forms.
- ``gene_id_namespace_vocabulary.md``: the same as a table.

Usage (from the repo root, the torchcell environment)::

    python experiments/database/scripts/measure_gene_id_namespaces.py [--workers 8]
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import os.path as osp
import re
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from torchcell.database.build_dataset_lmdb import (
    dataset_default_root,
    resolve_dataset_class,
)
from torchcell.datamodels.schema import BACTERIAL_LOCUS_TAG_PATTERNS
from torchcell.provenance.schema_change import served_release
from torchcell.sequence.genome import registry
from torchcell.verification.runners import stream_records

EXAMPLES_PER_CELL = 5

#: Identifier forms, tried in order; the first match names the form.
FORMS: dict[str, str] = {
    "sgd_systematic": r"^(Y[A-P][LR]\d{3}[WC](-[A-Z])?|Q\d{4}|YNC[A-Q]\d{4}[WC])$",
    # The two CEN.PK forms below are confirmed against the deposited GenBank by
    # ``measure_yeast_locus_tags``; the pattern here is what the schema will pin.
    "cenpk113_7d_locus_tag": r"^SCEN_(?:[A-P]\d{5}|[QR]\d+|[A-R]trna\d{1,2}[A-Z]?)$",
    **BACTERIAL_LOCUS_TAG_PATTERNS,
    # Peter 2018 pangenome ids as Caudal 2024 stores them (``X<n>-<name>``).
    "peter2018_pangenome_x_id": r"^X\d+-\S+$",
    # Peter 2018 pangenome ORF ids of the accessory set (``EC1118_1F14_0012g``).
    "peter2018_pangenome_orf": r"^[A-Za-z0-9]+_[A-Za-z0-9]+_\d{4}g$",
    "variant_site_id": r"^[^\s]+:\d+$",
    "curie": r"^[A-Za-z0-9_.]+:[A-Za-z0-9_.\-]+$",
}


def classify(name: str) -> str:
    """The first form in ``FORMS`` that ``name`` matches, else ``other``."""
    for form, pattern in FORMS.items():
        if re.match(pattern, name):
            return form
    return "other"


def scan_store(args: tuple[str, str]) -> dict[str, Any]:
    """Classify every perturbation identifier of one dev store."""
    dataset_class, abs_root = args
    counts: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    examples: dict[str, dict[str, list[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    strains: collections.Counter[str] = collections.Counter()
    n_records = 0
    n_without_perturbations = 0
    for record in stream_records(abs_root):
        n_records += 1
        genome = record["reference"]["genome_reference"]
        strains[f"{genome['species']} / {genome['strain']}"] += 1
        genotype = record["experiment"]["genotype"]
        if "perturbations" not in genotype:
            # a segregant genotype (Bloom 2019) is a haplotype mosaic with no
            # gene-keyed perturbation, so it carries no identifier to classify
            n_without_perturbations += 1
            continue
        for perturbation in genotype["perturbations"]:
            kind = str(perturbation["perturbation_type"])
            name = str(perturbation["systematic_gene_name"])
            form = classify(name)
            counts[kind][form] += 1
            bucket = examples[kind][form]
            if len(bucket) < EXAMPLES_PER_CELL and name not in bucket:
                bucket.append(name)
    return {
        "dataset_class": dataset_class,
        "root": abs_root,
        "n_records": n_records,
        "n_records_without_perturbations": n_without_perturbations,
        "strains": dict(strains),
        "perturbation_types": {
            kind: {
                "forms": dict(forms),
                "examples": {form: examples[kind][form] for form in forms},
            }
            for kind, forms in counts.items()
        },
    }


def measure_yeast_locus_tags(data_root: str) -> dict[str, Any]:
    """Every locus tag of the deposited CEN.PK113-7D GenBank, and every gene-class
    feature id of the SGD R64 GFF, against the yeast patterns.
    """
    out: dict[str, Any] = {}
    cenpk = registry.resolve(
        registry.SCEREVISIAE_CENPK113_7D,
        "GCA_002571405.2_ASM257140v2_genomic.gbff.gz",
        data_root=data_root,
    )
    tags: set[str] = set()
    with gzip.open(cenpk, "rt", encoding="utf-8") as handle:
        for line in handle:
            match = re.search(r'/locus_tag="([^"]+)"', line)
            if match:
                tags.add(match.group(1))
    cenpk_pattern = FORMS["cenpk113_7d_locus_tag"]
    unmatched = sorted(tag for tag in tags if not re.match(cenpk_pattern, tag))
    shapes = collections.Counter(re.sub(r"\d+", "N", tag) for tag in tags)
    out["cenpk113_7d"] = {
        "assembly_set": registry.SCEREVISIAE_CENPK113_7D,
        "file": osp.basename(cenpk),
        "pattern": cenpk_pattern,
        "n_locus_tags": len(tags),
        "n_unmatched": len(unmatched),
        "unmatched_examples": unmatched[:20],
        "shapes": dict(sorted(shapes.items(), key=lambda kv: -kv[1])[:12]),
    }
    gff = registry.resolve(
        registry.SGD_S288C_R64,
        "saccharomyces_cerevisiae_R64-4-1_20230830.gff",
        data_root=data_root,
    )
    gene_types = {
        "gene",
        "pseudogene",
        "transposable_element_gene",
        "ncRNA_gene",
        "tRNA_gene",
        "snoRNA_gene",
        "rRNA_gene",
        "snRNA_gene",
        "telomerase_RNA_gene",
    }
    by_type: collections.Counter[str] = collections.Counter()
    unmatched_sgd: list[tuple[str, str]] = []
    sgd_pattern = FORMS["sgd_systematic"]
    with open(gff, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) < 9 or columns[2] not in gene_types:
                continue
            match = re.search(r"(?:^|;)ID=([^;]+)", columns[8])
            if match is None:
                continue
            by_type[columns[2]] += 1
            if not re.match(sgd_pattern, match.group(1)):
                unmatched_sgd.append((columns[2], match.group(1)))
    out["sgd_s288c_r64"] = {
        "assembly_set": registry.SGD_S288C_R64,
        "file": osp.basename(gff),
        "pattern": sgd_pattern,
        "n_gene_features": sum(by_type.values()),
        "by_type": dict(by_type),
        "n_unmatched": len(unmatched_sgd),
        "unmatched": unmatched_sgd[:20],
    }
    return out


def render_markdown(result: dict[str, Any]) -> str:
    """The measurement as a markdown table, one row per (dataset, type, form)."""
    lines = [
        "# Gene identifier namespaces in the served stores",
        "",
        f"Release `{result['release']}`: {len(result['stores'])} served datasets, "
        f"read from the dev stores under `$DATA_ROOT`. Generated by "
        "`experiments/database/scripts/measure_gene_id_namespaces.py`.",
        "",
        "## Identifier forms per served dataset and perturbation type",
        "",
        "| dataset | perturbation_type | form | count | examples |",
        "|---|---|---|---|---|",
    ]
    totals: collections.Counter[str] = collections.Counter()
    for store in result["stores"]:
        for kind, cell in sorted(store["perturbation_types"].items()):
            for form, count in sorted(cell["forms"].items()):
                totals[form] += count
                lines.append(
                    f"| {store['dataset_class']} | `{kind}` | {form} | {count} | "
                    f"{', '.join(f'`{e}`' for e in cell['examples'][form])} |"
                )
    lines += ["", "## Totals per form", "", "| form | identifiers |", "|---|---|"]
    for form, count in totals.most_common():
        lines.append(f"| {form} | {count} |")
    lines += [
        "",
        "## Reference strain per served dataset",
        "",
        "| dataset | records | species / strain |",
        "|---|---|---|",
    ]
    for store in result["stores"]:
        strains = "; ".join(f"{k} ({v})" for k, v in store["strains"].items())
        lines.append(f"| {store['dataset_class']} | {store['n_records']} | {strains} |")
    yeast = result["yeast_locus_tags"]
    lines += ["", "## Yeast locus-tag forms in the genomes tier", ""]
    for key, block in yeast.items():
        lines.append(
            f"- `{block['assembly_set']}` (`{block['file']}`): pattern "
            f"`{block['pattern']}`; "
        )
        if key == "cenpk113_7d":
            lines.append(
                f"  {block['n_locus_tags']} locus tags, {block['n_unmatched']} "
                f"unmatched {block['unmatched_examples']}; shapes {block['shapes']}"
            )
        else:
            lines.append(
                f"  {block['n_gene_features']} gene-class features "
                f"{block['by_type']}, {block['n_unmatched']} unmatched "
                f"{block['unmatched']}"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--results-dir",
        default=osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results"),
    )
    args = parser.parse_args(argv)
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    repo = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
    release, served_classes = served_release(Path(repo))
    served = sorted(served_classes)
    jobs: list[tuple[str, str]] = []
    for name in served:
        cls = resolve_dataset_class(name)
        abs_root = osp.join(data_root, dataset_default_root(cls))
        if not osp.isdir(osp.join(abs_root, "processed", "lmdb")):
            raise FileNotFoundError(f"{name}: no dev store at {abs_root}")
        jobs.append((name, abs_root))
    print(
        f"{len(jobs)} served stores; scanning with {args.workers} workers", flush=True
    )
    with Pool(args.workers) as pool:
        stores = []
        for store in pool.imap_unordered(scan_store, jobs):
            print(
                f"  {store['dataset_class']}: {store['n_records']} records", flush=True
            )
            stores.append(store)
    stores.sort(key=lambda s: s["dataset_class"])
    result = {
        "release": release,
        "forms": FORMS,
        "stores": stores,
        "yeast_locus_tags": measure_yeast_locus_tags(data_root),
    }
    os.makedirs(args.results_dir, exist_ok=True)
    json_path = osp.join(args.results_dir, "gene_id_namespace_vocabulary.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    md_path = osp.join(args.results_dir, "gene_id_namespace_vocabulary.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(result))
    print(f"wrote {json_path}\nwrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
