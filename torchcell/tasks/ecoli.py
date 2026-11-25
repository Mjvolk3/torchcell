# torchcell/tasks/ecoli.py
# [[torchcell.tasks.ecoli]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/tasks/ecoli.py
"""The four *E. coli* benchmark tasks and the helpers that match records to genes.

A task is built in three steps, each a function here, so that a script which walks
through them one at a time and :func:`load_task` run the same code:

1. :func:`match_records` resolves every perturbed gene of every record against the
   genome the records name and says whether the record is a benchmark sample.
2. :func:`build_task` keeps the samples, reads their labels and attaches the genome.
3. :func:`load_task` adds the pinned splits shipped in ``torchcell/tasks/data`` after
   checking the file's sha256 and that its sample keys equal the built samples.

A record is a benchmark sample when it perturbs the number of genes the task is about
(one, or two) and each of them resolves to a gene of the genome that has a protein
sequence. Every other record is kept in ``task.excluded`` with its reason; the
datasets themselves are never filtered.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib
import io
import os
import os.path as osp
from importlib import resources
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from torchcell.sequence.genome.base import GeneNameStatus
from torchcell.sequence.genome.standard import genome_for
from torchcell.tasks.task import BenchmarkTask, TaskSpec

_ECOLI = "torchcell.datasets.ecoli"
ECOLI_TASKS: dict[str, TaskSpec] = {
    spec.name: spec
    for spec in (
        TaskSpec(
            name="ecoli-growth-rate-campos2018",
            title="Growth rate of single-gene deletions (Campos 2018)",
            dataset_slug="ecoli_growth_rate_campos2018",
            dataset_module=f"{_ECOLI}.campos2018",
            dataset_class="GrowthRateCampos2018Dataset",
            assembly_set="ecoli_K12_BW25113_ASM75055v1",
            kind="single_gene",
            label="fitness",
            n_targets_note="1 target: maximal growth rate relative to the parent strain",
        ),
        TaskSpec(
            name="ecoli-morphology-campos2018",
            title="Cell shape and cell cycle of single-gene deletions (Campos 2018)",
            dataset_slug="ecoli_morphology_campos2018",
            dataset_module=f"{_ECOLI}.campos2018",
            dataset_class="MorphologyCampos2018Dataset",
            assembly_set="ecoli_K12_BW25113_ASM75055v1",
            kind="single_gene",
            label="morphology",
            n_targets_note="15 targets: feature means; 6 are nan for strains imaged "
            "without a nucleoid channel",
        ),
        TaskSpec(
            name="ecoli-gene-interaction-babu2014",
            title="Genetic interaction score of double mutants (Babu 2014)",
            dataset_slug="gene_interaction_babu2014",
            dataset_module=f"{_ECOLI}.babu2014",
            dataset_class="GeneInteractionBabu2014Dataset",
            assembly_set="ecoli_K12_MG1655_ASM584v2",
            kind="gene_pair",
            label="gene_interaction",
            n_targets_note="1 target: eSGA S-score of the pair",
        ),
        TaskSpec(
            name="ecoli-metabolome-fuhrer2017",
            title="Metabolome of single-gene deletions (Fuhrer 2017)",
            dataset_slug="metabolome_fuhrer2017",
            dataset_module=f"{_ECOLI}.fuhrer2017",
            dataset_class="MetabolomeFuhrer2017Dataset",
            assembly_set="ecoli_K12_BW25113_ASM75055v1",
            kind="single_gene",
            label="metabolite_level",
            n_targets_note="7,534 targets: modified z-score of each detected ion",
        ),
    )
}

#: sha256 of each pinned split file in ``torchcell/tasks/data`` (the gzip bytes).
SPLIT_SHA256: dict[str, str] = {
    "ecoli-growth-rate-campos2018": "610640d6bcc07d9ae386502c0e8da25c4e38400b29ff8901eba910c12da8f7ae",
    "ecoli-morphology-campos2018": "610640d6bcc07d9ae386502c0e8da25c4e38400b29ff8901eba910c12da8f7ae",
    "ecoli-gene-interaction-babu2014": "57f69cbc513b3b610e4cd620755df221d64ce53a84dee0db314df94279aaaa99",
    "ecoli-metabolome-fuhrer2017": "eb984703fd287706f8160d5e2dc10267368ba018b9a711ac0f4521a4ed0dfe73",
}

#: Statuses under which a stored identifier names exactly one gene of the genome.
_RESOLVED = frozenset({GeneNameStatus.CURRENT, GeneNameStatus.RENAMED})


def dataset_root(slug: str) -> str:
    """Where the dataset's store is read from: ``$DATA_ROOT/data/torchcell/<slug>``."""
    return osp.join(os.environ.get("DATA_ROOT", "."), "data/torchcell", slug)


def open_dataset(spec: TaskSpec, root: str | None = None) -> Any:
    """Open the task's dataset with its loader.

    With ``TC_DATA_URL`` set the loader downloads the built store. Without it the
    loader builds the store from the publisher's files, which is slow.
    """
    loader = getattr(importlib.import_module(spec.dataset_module), spec.dataset_class)
    return loader(root=root or dataset_root(spec.dataset_slug))


def match_records(dataset: Any, genome: Any, n_genes: int) -> pd.DataFrame:
    """Match every record of ``dataset`` to genes of ``genome``.

    Each perturbed gene is stored under the identifier its source released.
    ``genome.resolve_gene_name`` maps it to a locus tag of the genome and reports how
    (``current``, ``renamed``, ``non_gene_feature``, ``retired``, ``ambiguous``).

    Args:
        dataset: A loaded torchcell dataset.
        genome: The genome its records name (``genome_for(dataset)``).
        n_genes: Number of perturbed genes a benchmark sample has (1 or 2).

    Returns:
        One row per record: ``record_index``, ``n_perturbations``, and joined by ``|``
        in perturbation order ``stored`` (identifiers as released), ``symbol``,
        ``locus_tag`` (resolved), ``status``; then ``eligible`` and
        ``exclusion_reason`` (empty when eligible).
    """
    resolution = {name: genome.resolve_gene_name(name) for name in dataset.gene_set}
    has_protein: dict[str, bool] = {}
    rows = []
    for index in range(len(dataset)):
        perturbations = dataset[index]["experiment"]["genotype"]["perturbations"]
        stored = [p["systematic_gene_name"] for p in perturbations]
        resolved = [resolution[name] for name in stored]
        reason = ""
        if len(stored) != n_genes:
            reason = f"perturbs {len(stored)} genes, the task is about {n_genes}"
        elif any(r.status not in _RESOLVED for r in resolved):
            reason = "a gene does not resolve to a gene of the genome"
        else:
            for r in resolved:
                locus = str(r.systematic_name)
                if locus not in has_protein:
                    protein = genome[locus].protein
                    has_protein[locus] = protein is not None and len(protein.seq) > 0
            if not all(has_protein[str(r.systematic_name)] for r in resolved):
                reason = "a gene has no protein sequence in the genome"
            elif len({str(r.systematic_name) for r in resolved}) != n_genes:
                reason = "the perturbed genes resolve to the same gene"
        rows.append(
            {
                "record_index": index,
                "n_perturbations": len(stored),
                "stored": "|".join(stored),
                "symbol": "|".join(
                    str(p["perturbed_gene_name"]) for p in perturbations
                ),
                "locus_tag": "|".join(str(r.systematic_name or "") for r in resolved),
                "status": "|".join(r.status.value for r in resolved),
                "eligible": reason == "",
                "exclusion_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def read_labels(
    dataset: Any, spec: TaskSpec, record_index: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.float64], list[str]]:
    """Labels of the given records as ``(n, n_targets)`` and the target names.

    A scalar label is one column named after the label. A label that maps names to
    values (morphology features, ion z-scores) is one column per name, in the order of
    first appearance, with ``nan`` where a record does not hold that name.
    """
    values = [
        dataset[int(i)]["experiment"]["phenotype"][spec.label] for i in record_index
    ]
    if not isinstance(values[0], dict):
        return np.asarray(values, dtype=np.float64).reshape(-1, 1), [spec.label]
    targets: dict[str, int] = {}
    for value in values:
        for name in value:
            targets.setdefault(name, len(targets))
    y = np.full((len(values), len(targets)), np.nan, dtype=np.float64)
    for row, value in enumerate(values):
        for name, level in value.items():
            if level is not None:
                y[row, targets[name]] = float(level)
    return y, list(targets)


def build_task(name: str, root: str | None = None) -> BenchmarkTask:
    """Build a task from its dataset and genome, without splits.

    Args:
        name: A key of :data:`ECOLI_TASKS`.
        root: Dataset root; default ``$DATA_ROOT/data/torchcell/<slug>``.

    Returns:
        The task with ``samples``, ``genes``, ``y``, ``targets``, ``excluded`` and
        ``genome`` set and no splits.

    Raises:
        KeyError: ``name`` is not a task; the message lists the task names.
        ValueError: The records name another genome assembly than the task pins.
    """
    if name not in ECOLI_TASKS:
        raise KeyError(f"no task {name!r}; the tasks are {sorted(ECOLI_TASKS)}")
    spec = ECOLI_TASKS[name]
    dataset = open_dataset(spec, root)
    genome = genome_for(dataset)
    if genome.ASSEMBLY_SET != spec.assembly_set:
        raise ValueError(
            f"{name} records name {genome.ASSEMBLY_SET}, the task pins {spec.assembly_set}"
        )
    n_genes = 1 if spec.kind == "single_gene" else 2
    matched = match_records(dataset, genome, n_genes)
    samples = matched[matched["eligible"]].drop(
        columns=["eligible", "exclusion_reason"]
    )
    samples = samples.reset_index(drop=True)
    samples.insert(1, "key", samples["locus_tag"])
    excluded = matched[~matched["eligible"]].drop(columns=["eligible"])
    genes = np.array([tags.split("|") for tags in samples["locus_tag"]], dtype=str)
    y, targets = read_labels(
        dataset, spec, samples["record_index"].to_numpy(dtype=np.int64)
    )
    return BenchmarkTask(
        spec, samples, genes, y, targets, excluded.reset_index(drop=True), genome
    )


def write_splits(path: str, task: BenchmarkTask, splits: pd.DataFrame) -> str:
    """Write a split file (gzip CSV, byte-stable) and return its sha256.

    The file has ``record_index``, ``key`` and one column per split holding ``train``,
    ``val`` or ``test`` for every sample of ``task``.
    """
    frame = pd.concat([task.samples[["record_index", "key"]], splits], axis=1)
    text = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, filename="") as handle:
        handle.write(text)
    data = buffer.getvalue()
    with open(path, "wb") as out:
        out.write(data)
    return hashlib.sha256(data).hexdigest()


def read_splits(name: str) -> pd.DataFrame:
    """The pinned split file of a task, verified against :data:`SPLIT_SHA256`."""
    data = (
        resources.files("torchcell.tasks").joinpath(f"data/{name}.splits.csv.gz")
    ).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SPLIT_SHA256[name]:
        raise ValueError(
            f"split file of {name} has sha256 {digest}, pinned {SPLIT_SHA256[name]}"
        )
    return pd.read_csv(io.BytesIO(gzip.decompress(data)), dtype=str)


def load_task(name: str, root: str | None = None) -> BenchmarkTask:
    """The processed benchmark task ``name`` with its pinned splits.

    Opens the dataset (downloading the built store when ``TC_DATA_URL`` is set, building
    it from the publisher's files otherwise), loads the genome its records name (fetched
    when absent), matches records to genes, reads
    the labels and attaches the pinned splits. The samples built here must equal the
    samples the split file was written for, row for row, or this raises.

    Args:
        name: A key of :data:`ECOLI_TASKS`.
        root: Dataset root; default ``$DATA_ROOT/data/torchcell/<slug>``.

    Returns:
        The task; ``task.split(split_name)`` gives train and validation indices and a
        locked test part.

    Raises:
        KeyError: ``name`` is not a task.
        ValueError: The pinned split file fails its sha256 check, or the samples built
            here differ from the samples the split file was written for.
    """
    task = build_task(name, root)
    pinned = read_splits(name)
    built = task.samples[["record_index", "key"]].astype(str).reset_index(drop=True)
    if not built.equals(pinned[["record_index", "key"]]):
        raise ValueError(
            f"the samples built for {name} differ from the pinned split file's "
            f"({len(built)} built, {len(pinned)} pinned)"
        )
    task._splits = pinned.drop(columns=["record_index", "key"])
    return task
