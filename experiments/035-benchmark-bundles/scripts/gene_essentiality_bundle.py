# experiments/035-benchmark-bundles/scripts/gene_essentiality_bundle.py
# [[experiments.035-benchmark-bundles.scripts.gene_essentiality_bundle]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/035-benchmark-bundles/scripts/gene_essentiality_bundle
r"""Build the ``gene-essentiality-sgd`` benchmark bundle from two tc-data archives.

The task is binary: one record per gene, one target ``is_essential``.

- Label 1: the gene has an SGD ``inviable`` null-mutant record, that is, it is in the
  gene set of ``GeneEssentialitySgdDataset``.
- Label 0: the gene is not in that set and has a KanMX or NatMX deletion strain with a
  measured fitness in ``SmfCostanzo2016Dataset``, that is, its deletion mutant grew.
- A gene in neither group is not in the benchmark: nothing stored says whether its
  null mutant is viable. A gene in both (SGD calls it inviable and Costanzo measured a
  deletion strain) is labeled 1, and the count is recorded.

The essentiality store holds only ``inviable`` records, so the negatives have to come
from another dataset; this rule is PROVISIONAL and is the project owner's to confirm.

Split: within each class the genes are sorted, shuffled with ``random.Random(seed)``
and cut 80/10/10 into train, validation and test, so each split keeps the class ratio.

Inputs are the archives in a tc-data store (``index.json`` plus one ``.tar.xz`` per
dataset). Each archive is verified against the sha256 in the index before it is read,
and only ``preprocess/gene_set.json`` (essentiality) and ``preprocess/data.csv`` (SMF)
are read from it. Outputs: the bundle under ``--datasets-root`` (it holds the labels and
is not committed) and ``experiments/035-benchmark-bundles/results/
gene_essentiality_bundle.json``, a :class:`BundleProvenance` with every count and hash.

Run from the repo root::

    python experiments/035-benchmark-bundles/scripts/gene_essentiality_bundle.py \\
        --store <tc-data store> --datasets-root <TC_BENCH_DATASETS_ROOT>
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from torchcell.benchmark.bundle import BenchmarkDataset, sha256_bytes, write_bundle
from torchcell.benchmark.submission import Split

SLUG = "gene-essentiality-sgd"
VERSION = "1"
TARGET = "is_essential"
SEED = 0
FRACTIONS = {Split.TRAIN: 0.8, Split.VAL: 0.1, Split.TEST: 0.1}
ESSENTIALITY_SLUG = "gene_essentiality_sgd"
SMF_SLUG = "smf_costanzo2016"
DELETION_TYPES = frozenset({"KanMX_deletion", "NatMX_deletion"})
RESULTS = Path(__file__).resolve().parents[1] / "results"


class ArchiveUsed(BaseModel):
    """One tc-data archive the bundle was built from."""

    slug: str
    dataset_class: str
    torchcell_version: str
    archive: str
    archive_sha256: str
    member: str
    member_sha256: str


class BundleProvenance(BaseModel):
    """Where the bundle's records and labels came from, and how they were split."""

    script: str
    built_at: datetime
    label_rule: str
    seed: int
    fractions: dict[str, float]
    archives: list[ArchiveUsed]
    n_essential: int
    n_nonessential: int
    n_essential_with_a_deletion_strain: int
    n_per_split: dict[str, dict[str, int]]
    dataset: BenchmarkDataset


def read_member(store: Path, slug: str, member: str) -> tuple[bytes, ArchiveUsed]:
    """One file of the newest archive of ``slug``, after verifying the archive's hash."""
    index = json.loads((store / "index.json").read_text(encoding="utf-8"))
    rows = [row for row in index["artifacts"] if row["slug"] == slug]
    row = max(rows, key=lambda r: r["packaged_at"])
    data = (store / slug / row["archive"]).read_bytes()
    if sha256_bytes(data) != row["archive_sha256"]:
        raise ValueError(f"{row['archive']} does not match the sha256 in index.json")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as archive:
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError(f"{member} is not a file in {row['archive']}")
        content = extracted.read()
    return content, ArchiveUsed(
        slug=slug,
        dataset_class=row["dataset_class"],
        torchcell_version=row["torchcell_version"],
        archive=row["archive"],
        archive_sha256=row["archive_sha256"],
        member=member,
        member_sha256=sha256_bytes(content),
    )


def stratified_split(genes: set[str], seed: int) -> dict[Split, list[str]]:
    """``genes`` sorted, shuffled with ``seed`` and cut by ``FRACTIONS``."""
    ordered = sorted(genes)
    random.Random(seed).shuffle(ordered)
    n_val = round(len(ordered) * FRACTIONS[Split.VAL])
    n_test = round(len(ordered) * FRACTIONS[Split.TEST])
    n_train = len(ordered) - n_val - n_test
    return {
        Split.TRAIN: ordered[:n_train],
        Split.VAL: ordered[n_train : n_train + n_val],
        Split.TEST: ordered[n_train + n_val :],
    }


def main() -> None:
    """Build the bundle and write its provenance record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    args = parser.parse_args()

    gene_set_bytes, essentiality = read_member(
        args.store, ESSENTIALITY_SLUG, "preprocess/gene_set.json"
    )
    smf_bytes, smf = read_member(args.store, SMF_SLUG, "preprocess/data.csv")
    essential = set(json.loads(gene_set_bytes))
    deleted = {
        row["Systematic gene name"]
        for row in csv.DictReader(io.StringIO(smf_bytes.decode("utf-8")))
        if row["perturbation_type"] in DELETION_TYPES
    }
    nonessential = deleted - essential

    by_class = {
        1.0: stratified_split(essential, SEED),
        0.0: stratified_split(nonessential, SEED),
    }
    splits = {
        split: sorted(by_class[1.0][split] + by_class[0.0][split]) for split in Split
    }
    values = {
        (gene, TARGET): label
        for label, class_splits in by_class.items()
        for split in (Split.VAL, Split.TEST)
        for gene in class_splits[split]
    }
    built_at = datetime.now(UTC)
    dataset = write_bundle(
        args.datasets_root,
        slug=SLUG,
        title="Gene essentiality (SGD)",
        description=(
            "Predict whether the null mutant of a gene is inviable. Essential genes "
            "are those SGD annotates as inviable; non-essential genes are those with "
            "a deletion strain measured in the Costanzo 2016 single-mutant fitness "
            "screens. A prediction is a score, larger meaning more likely essential."
        ),
        loader_class=essentiality.dataset_class,
        citation_key="cherrySGDSaccharomycesGenome1998",
        version=VERSION,
        task="binary",
        primary_metric="auroc",
        splits=splits,
        values=values,
        docs_url="https://mjvolk3.github.io/torchcell/datasets/scerevisiae/essentiality-smf.html",
        tc_data_slug=ESSENTIALITY_SLUG,
        built_at=built_at,
    )
    provenance = BundleProvenance(
        script="experiments/035-benchmark-bundles/scripts/gene_essentiality_bundle.py",
        built_at=built_at,
        label_rule=(
            "1 if the gene is in the GeneEssentialitySgdDataset gene set; 0 if it is "
            "not and has a KanMX or NatMX deletion strain in SmfCostanzo2016Dataset; "
            "otherwise excluded"
        ),
        seed=SEED,
        fractions={split.value: fraction for split, fraction in FRACTIONS.items()},
        archives=[essentiality, smf],
        n_essential=len(essential),
        n_nonessential=len(nonessential),
        n_essential_with_a_deletion_strain=len(essential & deleted),
        n_per_split={
            split.value: {
                "essential": len(by_class[1.0][split]),
                "nonessential": len(by_class[0.0][split]),
            }
            for split in Split
        },
        dataset=dataset,
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "gene_essentiality_bundle.json"
    out.write_text(provenance.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(provenance.model_dump_json(indent=2, exclude={"dataset"}))
    print(f"bundle: {args.datasets_root / SLUG}\nprovenance: {out}")


if __name__ == "__main__":
    main()
