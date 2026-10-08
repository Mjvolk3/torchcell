# experiments/035-benchmark-bundles/scripts/gene_essentiality_bundle.py
# [[experiments.035-benchmark-bundles.scripts.gene_essentiality_bundle]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/035-benchmark-bundles/scripts/gene_essentiality_bundle
r"""Build the ``gene-essentiality-sgd`` benchmark bundle from SGD alone.

The task is binary: one record per gene, one target ``is_essential``. Both labels come
from the Saccharomyces Genome Database, so no other dataset is spent on the labels and
a method may use any of them, single-mutant fitness included, as input.

- Universe: the ORF rows of ``SGD_features.tab`` (Verified, Uncharacterized, Dubious).
- Label 1: the gene is in the gene set of the served ``GeneEssentialitySgdDataset``
  (the tc-data archive's ``preprocess/gene_set.json``): SGD annotates a ``null`` mutant
  of the gene in strain ``S288C`` as ``inviable``. Taking the archive, not a fresh SGD
  read, keeps the benchmark's positives equal to the dataset the knowledge graph serves;
  the fresh read is compared with it and any drift is recorded.
- Label 0: the gene is in the universe, not in that set, and SGD annotates a ``null``
  mutant in ``S288C`` as ``viable`` (the mirrored ``phenotype_details/<ORF>.json``).
- Excluded: a gene with neither annotation; nothing stored says whether its null
  mutant lives. A gene with both is labeled 1 and counted.

Split: within each class the genes are sorted, shuffled with ``random.Random(seed)``
and cut 80/10/10 into train, validation and test, so each split keeps the class ratio.

Inputs, every one verified by hash before it is read: the SGD mirror written by
``sgd_phenotype_mirror.py`` (each file against its ``provenance.json`` entry) and the
tc-data store (the archive against ``index.json``). Outputs: the bundle under
``--datasets-root`` (it holds the labels and is not committed), with a public
:class:`BundleProvenance` in its ``benchmark.json``; ``results/gene_essentiality_bundle.json``,
the full build record; and the upload test case, ``results/smf_baseline_predictions.csv``
plus ``results/smf_baseline_metadata.json``: a Costanzo 2016 deletion-fitness lookup
scored on the validation and test genes (score ``1 - min fitness`` over the gene's
KanMX and NatMX deletion strains at 30 C; a gene with no deletion strain measured
scores 1, since the deletion collection lacks the genes whose deletion did not live).
That file is the first thing submitted to the board, through ``/admin/baselines``.

Run from the repo root::

    python experiments/035-benchmark-bundles/scripts/gene_essentiality_bundle.py \\
        --store <tc-data store> --mirror <SGD raw mirror> \\
        --datasets-root <TC_BENCH_DATASETS_ROOT>
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import tarfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from torchcell.benchmark.bundle import (
    BenchmarkDataset,
    BundleProvenance,
    SourceRecord,
    sha256_bytes,
    write_bundle,
)
from torchcell.benchmark.submission import PREDICTION_COLUMNS, Split

SLUG = "gene-essentiality-sgd"
VERSION = "2"
TARGET = "is_essential"
SEED = 0
FRACTIONS = {Split.TRAIN: 0.8, Split.VAL: 0.1, Split.TEST: 0.1}
ESSENTIALITY_SLUG = "gene_essentiality_sgd"
SMF_SLUG = "smf_costanzo2016"
DELETION_TYPES = frozenset({"KanMX_deletion", "NatMX_deletion"})
BASELINE_TEMPERATURE = "30"
MISSING_SCORE = 1.0
SCRIPT = "experiments/035-benchmark-bundles/scripts/gene_essentiality_bundle.py"
RESULTS = Path(__file__).resolve().parents[1] / "results"
LABEL_RULE = (
    "1 if the gene is in the served GeneEssentialitySgdDataset gene set (an SGD null "
    "mutant in S288C annotated inviable); 0 if it is an SGD ORF not in that set with "
    "an SGD null mutant in S288C annotated viable; otherwise excluded. Both labels "
    "come from SGD; no fitness dataset is used."
)
SPLIT_RULE = (
    "within each class, genes sorted, shuffled with random.Random(0), cut 80/10/10 "
    "into train, validation, test"
)


class ArchiveUsed(BaseModel):
    """One tc-data archive the bundle was built from."""

    slug: str
    dataset_class: str
    torchcell_version: str
    archive: str
    archive_sha256: str
    member: str
    member_sha256: str


class SgdDrift(BaseModel):
    """The fresh SGD read against the served dataset's gene set."""

    fresh_inviable: int
    in_served_set_not_fresh: int
    fresh_not_in_served_set: int
    both_viable_and_inviable: int


class BaselineCoverage(BaseModel):
    """How many scored genes the deletion-fitness lookup covered."""

    scored_genes: int
    with_deletion_fitness: int
    without: int
    missing_score: float
    predictions_sha256: str


class BundleBuild(BaseModel):
    """The full build record (``results/gene_essentiality_bundle.json``)."""

    script: str
    built_at: datetime
    label_rule: str
    split_rule: str
    seed: int
    fractions: dict[str, float]
    mirror_provenance_sha256: str
    universe: int
    universe_by_qualifier: dict[str, int]
    archives: list[ArchiveUsed]
    drift: SgdDrift
    n_essential: int
    n_essential_outside_universe: int
    n_nonessential: int
    n_excluded: int
    n_per_split: dict[str, dict[str, int]]
    baseline: BaselineCoverage
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


def verified_file(mirror: Path, entry: dict[str, object]) -> bytes:
    """A mirrored file's bytes, refused if they differ from ``provenance.json``."""
    path = mirror / str(entry["path"])
    data = path.read_bytes()
    if sha256_bytes(data) != entry["sha256"]:
        raise ValueError(f"{path} does not match the sha256 in provenance.json")
    return data


def null_s288c_calls(phenotypes: list[dict[str, object]]) -> set[str]:
    """The phenotype names SGD gives a ``null`` mutant of the locus in ``S288C``."""
    calls: set[str] = set()
    for row in phenotypes:
        strain = row.get("strain")
        phenotype = row.get("phenotype")
        if (
            row.get("mutant_type") == "null"
            and isinstance(strain, dict)
            and strain.get("display_name") == "S288C"
            and isinstance(phenotype, dict)
        ):
            calls.add(str(phenotype["display_name"]))
    return calls


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


def content_sha256(hashes: list[str]) -> str:
    """sha256 over the sorted hashes, newline-joined, trailing newline."""
    return sha256_bytes(("\n".join(sorted(hashes)) + "\n").encode("utf-8"))


def smf_baseline(
    smf_csv: bytes, scored: dict[Split, list[str]]
) -> tuple[bytes, BaselineCoverage]:
    """The deletion-fitness lookup as a predictions file in template order."""
    best: dict[str, float] = defaultdict(lambda: float("inf"))
    for row in csv.DictReader(io.StringIO(smf_csv.decode("utf-8"))):
        if (
            row["perturbation_type"] in DELETION_TYPES
            and row["Temperature"] == BASELINE_TEMPERATURE
        ):
            gene = row["Systematic gene name"]
            best[gene] = min(best[gene], float(row["Single mutant fitness"]))
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(PREDICTION_COLUMNS)
    covered = 0
    total = 0
    for split in (Split.VAL, Split.TEST):
        for gene in scored[split]:
            total += 1
            if gene in best:
                covered += 1
                score = 1.0 - best[gene]
            else:
                score = MISSING_SCORE
            writer.writerow([gene, split.value, TARGET, repr(score)])
    data = out.getvalue().encode("utf-8")
    return data, BaselineCoverage(
        scored_genes=total,
        with_deletion_fitness=covered,
        without=total - covered,
        missing_score=MISSING_SCORE,
        predictions_sha256=sha256_bytes(data),
    )


def main() -> None:
    """Build the bundle, its provenance, the build record, and the upload test case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    args = parser.parse_args()

    provenance_bytes = (args.mirror / "provenance.json").read_bytes()
    mirror_provenance = json.loads(provenance_bytes)
    features_entry = mirror_provenance["features"]
    features_bytes = verified_file(args.mirror, features_entry)
    universe: dict[str, str] = {}
    for row in csv.reader(io.StringIO(features_bytes.decode("utf-8")), delimiter="\t"):
        if len(row) >= 4 and row[1] == "ORF":
            universe[row[3]] = row[2]
    by_qualifier: dict[str, int] = defaultdict(int)
    for qualifier in universe.values():
        by_qualifier[qualifier] += 1

    detail_entries = {
        Path(str(e["path"])).stem: e for e in mirror_provenance["phenotype_details"]
    }
    missing = sorted(set(universe) - set(detail_entries))
    if missing:
        raise ValueError(
            f"{len(missing)} ORFs have no phenotype file in the mirror, "
            f"for example {missing[:3]}; rerun sgd_phenotype_mirror.py"
        )
    fresh_viable: set[str] = set()
    fresh_inviable: set[str] = set()
    detail_hashes: list[str] = []
    for orf in sorted(universe):
        entry = detail_entries[orf]
        calls = null_s288c_calls(json.loads(verified_file(args.mirror, entry)))
        detail_hashes.append(str(entry["sha256"]))
        if "viable" in calls:
            fresh_viable.add(orf)
        if "inviable" in calls:
            fresh_inviable.add(orf)

    gene_set_bytes, essentiality = read_member(
        args.store, ESSENTIALITY_SLUG, "preprocess/gene_set.json"
    )
    smf_bytes, smf = read_member(args.store, SMF_SLUG, "preprocess/data.csv")
    essential = set(json.loads(gene_set_bytes))
    nonessential = (fresh_viable & set(universe)) - essential
    excluded = set(universe) - essential - nonessential
    drift = SgdDrift(
        fresh_inviable=len(fresh_inviable),
        in_served_set_not_fresh=len(essential - fresh_inviable),
        fresh_not_in_served_set=len(fresh_inviable - essential),
        both_viable_and_inviable=len(fresh_viable & essential),
    )

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
    retrieved = datetime.fromisoformat(
        str(mirror_provenance["phenotype_details"][0]["retrieved_at"])
    )
    provenance = BundleProvenance(
        script=SCRIPT,
        label_rule=LABEL_RULE,
        split_rule=SPLIT_RULE,
        sources=[
            SourceRecord(
                name="SGD_features.tab, the SGD chromosomal feature table",
                role="universe",
                source_url=str(features_entry["source_url"]),
                retrieval_method="direct_url",
                retrieved_at=datetime.fromisoformat(
                    str(features_entry["retrieved_at"])
                ),
                sha256=str(features_entry["sha256"]),
                bytes=int(str(features_entry["bytes"])),
                note=f"{len(universe)} ORFs: "
                + ", ".join(f"{n} {q}" for q, n in sorted(by_qualifier.items())),
            ),
            SourceRecord(
                name="SGD locus phenotype annotations, one file per ORF",
                role="label 0 (viable)",
                source_url="https://www.yeastgenome.org/backend/locus/{ORF}/phenotype_details",
                retrieval_method="direct_url",
                retrieved_at=retrieved,
                sha256=content_sha256(detail_hashes),
                n_files=len(detail_hashes),
                note=(
                    "mirrored with experiments/035-benchmark-bundles/scripts/"
                    "sgd_phenotype_mirror.py; per-file hashes in the mirror's "
                    f"provenance.json (sha256 {sha256_bytes(provenance_bytes)[:12]})"
                ),
            ),
            SourceRecord(
                name=f"tc-data archive {essentiality.archive} of {essentiality.dataset_class}",
                role="label 1 (inviable)",
                retrieval_method="tc_data_archive",
                sha256=essentiality.archive_sha256,
                bytes=None,
                note=(
                    f"preprocess/gene_set.json, sha256 {essentiality.member_sha256[:12]}, "
                    f"torchcell {essentiality.torchcell_version}; the gene set the "
                    "knowledge graph serves"
                ),
            ),
        ],
        notes=[
            f"{drift.both_viable_and_inviable} genes carry both a viable and an inviable "
            "SGD call; labeled 1.",
            f"Fresh SGD read: {drift.fresh_inviable} inviable; "
            f"{drift.fresh_not_in_served_set} not in the served set, "
            f"{drift.in_served_set_not_fresh} in the served set no longer inviable.",
            f"{len(excluded)} ORFs excluded: no viable or inviable null call in S288C.",
        ],
    )
    built_at = datetime.now(UTC)
    dataset = write_bundle(
        args.datasets_root,
        slug=SLUG,
        title="Gene essentiality (SGD)",
        description=(
            "Predict whether the null mutant of a gene is inviable. Both labels come "
            "from SGD: essential genes are those with a null mutant annotated inviable "
            "in S288C, non-essential genes those with a null mutant annotated viable. "
            "A prediction is a score, larger meaning more likely essential."
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
        provenance=provenance,
    )

    predictions, coverage = smf_baseline(smf_bytes, splits)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "smf_baseline_predictions.csv").write_bytes(predictions)
    metadata = {
        "method_name": "Costanzo 2016 deletion fitness",
        "description": (
            "A lookup, not a model: the gene's single-mutant fitness in the Costanzo "
            "2016 screens at 30 C, minimum over its KanMX and NatMX deletion strains, "
            "scored as 1 - fitness. A gene with no deletion strain measured scores 1: "
            "the deletion collection lacks the genes whose deletion did not live."
        ),
        "model_family": "lookup",
        "encoding": "none",
        "code_url": "https://github.com/Mjvolk3/torchcell/blob/main/" + SCRIPT,
        "uses_external_data": True,
        "external_data_description": (
            f"SmfCostanzo2016Dataset via tc-data archive {smf.archive} "
            f"(sha256 {smf.archive_sha256[:12]}), preprocess/data.csv"
        ),
        "hyperparameters": {
            "temperature_c": 30,
            "aggregate": "min over deletion strains",
            "missing_score": MISSING_SCORE,
        },
    }
    (RESULTS / "smf_baseline_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    build = BundleBuild(
        script=SCRIPT,
        built_at=built_at,
        label_rule=LABEL_RULE,
        split_rule=SPLIT_RULE,
        seed=SEED,
        fractions={split.value: fraction for split, fraction in FRACTIONS.items()},
        mirror_provenance_sha256=sha256_bytes(provenance_bytes),
        universe=len(universe),
        universe_by_qualifier=dict(sorted(by_qualifier.items())),
        archives=[essentiality, smf],
        drift=drift,
        n_essential=len(essential),
        n_essential_outside_universe=len(essential - set(universe)),
        n_nonessential=len(nonessential),
        n_excluded=len(excluded),
        n_per_split={
            split.value: {
                "essential": len(by_class[1.0][split]),
                "nonessential": len(by_class[0.0][split]),
            }
            for split in Split
        },
        baseline=coverage,
        dataset=dataset,
    )
    out = RESULTS / "gene_essentiality_bundle.json"
    out.write_text(build.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(build.model_dump_json(indent=2, exclude={"dataset"}))
    print(f"bundle: {args.datasets_root / SLUG}\nbuild record: {out}")
    print(f"upload test case: {RESULTS / 'smf_baseline_predictions.csv'}")


if __name__ == "__main__":
    main()
