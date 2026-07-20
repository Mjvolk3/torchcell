# torchcell/knowledge_graphs/subset
# [[torchcell.knowledge_graphs.subset]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/subset
"""Prefilter and subsample datasets for KG builds.

A build selects which records reach the adapters in two composable steps:

    prefilter -> sample

``RecordFilter`` declares WHICH records are eligible (by perturbed genes, by number of
perturbations); ``size`` then takes a random sample of the eligible pool, or all of it
when ``size`` is null. Both return the SAME dataset class -- ``dataset[indices]`` is a
PyG ``index_select``, a lazy index view that copies no data and keeps
``experiment_reference_index`` and the on-disk paths intact -- so adapters are unaware
that anything was narrowed.

WHY PREFILTER. A random cap cannot serve a join. Sameith 2015 measures expression for 72
double-knockout strains but reports no fitness; the matching double-mutant fitness sits
in Costanzo 2016, which the build caps at 100k of 20.7M records. At 0.48% that sample
hits 0 of the 72 pairs (measured), while the full Costanzo screen covers 57 of them.
Naming the records is the only way to get them.

The same mechanism serves fast test builds: prefilter to single mutants, or to one
pathway's genes, and sample from that -- a miniature that is representative of what you
care about rather than of the whole corpus.

COST. Filtering reads LMDB values directly and pulls only the fields a criterion needs
out of the pickled dict, skipping pydantic validation -- ~5 min for the 20.7M-record
Costanzo double-mutant set (measured), seconds elsewhere. Validation then runs only on
the surviving records, through the normal ``dataset[indices]`` path.
"""

import logging
import os.path as osp
import pickle
import random
from typing import Any

import lmdb
from pydantic import BaseModel, model_validator

log = logging.getLogger(__name__)


class RecordFilter(BaseModel):
    """Declarative prefilter over raw dataset records, applied before sampling.

    Every criterion that is set must hold (logical AND); an all-unset filter matches
    nothing and is rejected, so a typo in a config cannot silently pass everything
    through. Criteria read the raw record dict, never a validated pydantic object.

    Attributes:
        gene_sets: keep a record whose perturbed gene set EQUALS one of these sets.
            Order-independent (compared as frozensets), which matters because screens
            store a pair under both query/array orientations.
        gene_sets_file: path to a gene-set file, merged into ``gene_sets``. Relative
            paths resolve against the installed ``torchcell`` package.
        any_genes: keep a record perturbing at least one of these genes.
        all_genes: keep a record perturbing all of these genes.
        n_perturbations: keep a record whose perturbation count is one of these
            (e.g. ``[1]`` singles, ``[2]`` doubles).
    """

    gene_sets: list[list[str]] | None = None
    gene_sets_file: str | None = None
    any_genes: list[str] | None = None
    all_genes: list[str] | None = None
    n_perturbations: list[int] | None = None

    @model_validator(mode="after")
    def _merge_file_and_require_a_criterion(self) -> "RecordFilter":
        """Fold ``gene_sets_file`` into ``gene_sets`` and reject an empty filter."""
        if self.gene_sets_file is not None:
            loaded = load_gene_sets(self.gene_sets_file)
            self.gene_sets = (self.gene_sets or []) + loaded
        if not any(
            (self.gene_sets, self.any_genes, self.all_genes, self.n_perturbations)
        ):
            raise ValueError(
                "RecordFilter has no criteria set; it would match nothing. "
                "Set gene_sets/gene_sets_file, any_genes, all_genes, or n_perturbations."
            )
        return self

    def matches(self, genes: frozenset[str], n_perturbations: int) -> bool:
        """Return whether one record's perturbation summary satisfies every criterion."""
        if self.gene_sets is not None and genes not in {
            frozenset(gs) for gs in self.gene_sets
        }:
            return False
        if self.any_genes is not None and not (genes & set(self.any_genes)):
            return False
        if self.all_genes is not None and not set(self.all_genes) <= genes:
            return False
        if self.n_perturbations is not None and n_perturbations not in set(
            self.n_perturbations
        ):
            return False
        return True


def load_gene_sets(path: str) -> list[list[str]]:
    """Load gene sets from a text file: one set per line, whitespace-separated genes.

    Blank lines and ``#`` comments are ignored, so a generating script can stamp
    provenance into the file it emits. A relative path resolves against the installed
    ``torchcell`` package, because the KG build runs in a container that pip-installs
    torchcell from git -- the repo checkout is absent there, but the package is not.
    """
    if not osp.isabs(path):
        path = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), path)
    gene_sets: list[list[str]] = []
    with open(path) as fh:
        for line in fh:
            stripped = line.split("#", 1)[0].strip()
            if stripped:
                gene_sets.append(stripped.split())
    log.info("load_gene_sets: %d sets from %s", len(gene_sets), path)
    return gene_sets


def _perturbation_summary(record: dict[str, Any]) -> tuple[frozenset[str], int]:
    """Return the perturbed gene names and perturbation count for one raw record."""
    perturbations = record["experiment"]["genotype"]["perturbations"]
    return frozenset(p["systematic_gene_name"] for p in perturbations), len(
        perturbations
    )


def select_indices(lmdb_path: str, record_filter: RecordFilter) -> list[int]:
    """Return the indices of records satisfying ``record_filter``, in one LMDB pass."""
    indices: list[int] = []
    env = lmdb.open(
        lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
    )
    with env.begin() as txn:
        total = txn.stat()["entries"]
        for key, value in txn.cursor():
            genes, n_perturbations = _perturbation_summary(pickle.loads(value))
            if record_filter.matches(genes, n_perturbations):
                indices.append(int(key.decode()))
    env.close()
    indices.sort()
    log.info("prefilter: %s -> %d of %d records", lmdb_path, len(indices), total)
    return indices


def subset_dataset(
    dataset: Any,
    size: int | None,
    seed: int = 42,
    prefilter_indices: list[int] | None = None,
) -> Any:
    """Return the dataset narrowed to a prefiltered pool, then randomly sampled.

    ``prefilter_indices=None`` leaves the pool as the whole dataset; ``size=None`` takes
    the whole pool. With both unset the dataset is returned unchanged (the real build).
    The return is always the same dataset class -- an index view, not a copy.
    """
    if size is None and prefilter_indices is None:
        return dataset
    pool = list(range(len(dataset))) if prefilter_indices is None else prefilter_indices
    if size is None or size >= len(pool):
        indices = sorted(pool)
        log.info(
            "subset: %s %d -> %d records (all of the prefiltered pool)",
            type(dataset).__name__,
            len(dataset),
            len(indices),
        )
    else:
        indices = sorted(random.Random(seed).sample(pool, size))
        log.info(
            "subset: %s %d -> %d records (%d sampled from a pool of %d, seed=%d)",
            type(dataset).__name__,
            len(dataset),
            len(indices),
            size,
            len(pool),
            seed,
        )
    return dataset[indices]
