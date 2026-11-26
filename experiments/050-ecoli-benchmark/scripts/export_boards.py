# experiments/050-ecoli-benchmark/scripts/export_boards.py
# [[experiments.050-ecoli-benchmark.scripts.export_boards]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/050-ecoli-benchmark/scripts/export_boards
"""Export the E. coli tasks and their linear-probe predictions for the benchmark boards.

The benchmark service (``tc-bench``, on the website branch) serves one board per
dataset bundle, and a bundle has one split. This script writes, for the seed-0 split of
every split family of a task, a board directory the bundle builder on that branch reads:

    $DATA_ROOT/data/torchcell/experiments/050-ecoli-benchmark/boards/<slug>/
        board.json                      slug, title, description, loader, citation, targets
        splits.csv                      record_id,split            (every sample)
        labels.csv                      record_id,split,target,value   (validation and test)
        submissions/<name>/predictions.csv   record_id,split,target,prediction
        submissions/<name>/metadata.json     the board's submission metadata

A record id is the row of the source dataset (``record_index``). A target is named
as the task names it, with characters the board does not accept replaced
(:func:`board_target`; ``board.json`` keeps the original names). A label that was not
measured is left out, and so is its prediction. A probe whose predictions are constant
within a split and target (one-hot features on a gene that was not in training) cannot
be scored by the board's grader and is skipped, with a line saying so.

Tasks with more than ``MAX_TARGETS`` targets are not exported: the metabolome task has
7,534, which is about 5.4 million prediction rows per upload against the service's
16 MB limit.

    python experiments/050-ecoli-benchmark/scripts/export_boards.py
"""

from __future__ import annotations

import csv
import glob
import json
import os
import os.path as osp
import re

import numpy as np
from dotenv import load_dotenv

from torchcell.datasets.client import DatasetClient
from torchcell.endpoint_http import DEFAULT_URL
from torchcell.tasks import load_task
from torchcell.tasks.ecoli import ECOLI_TASKS

ROOT = "data/torchcell/experiments/050-ecoli-benchmark"
MAX_TARGETS = 100
SITE = "https://torchcell-database.ncsa.illinois.edu/staging/benchmark/datasets/ecoli"
CODE = "https://github.com/Mjvolk3/torchcell/blob/main/experiments/050-ecoli-benchmark/scripts/linear_probes.py"
PAGE = {
    "ecoli-growth-rate-campos2018": (
        "growth-rate-campos2018",
        "E. coli growth rate (Campos 2018)",
        "camposGenomewidePhenotypicAnalysis2018",
    ),
    "ecoli-morphology-campos2018": (
        "morphology-campos2018",
        "E. coli cell shape and cell cycle (Campos 2018)",
        "camposGenomewidePhenotypicAnalysis2018",
    ),
    "ecoli-gene-interaction-babu2014": (
        "gene-interaction-babu2014",
        "E. coli genetic interactions (Babu 2014)",
        "babuQuantitativeGenomeWideGenetic2014",
    ),
    "ecoli-metabolome-fuhrer2017": (
        "metabolome-fuhrer2017",
        "E. coli metabolome (Fuhrer 2017)",
        "fuhrerGenomewideLandscapeGene2017",
    ),
}
FAMILY = {
    "random": "random split",
    "seqid30": "sequence identity 30 % split",
    "seqid50": "sequence identity 50 % split",
    "position": "chromosome position split",
    "gene_holdout": "gene holdout, random",
    "seqid30_gene_holdout": "gene holdout, sequence identity 30 %",
    "seqid50_gene_holdout": "gene holdout, sequence identity 50 %",
    "position_gene_holdout": "gene holdout, chromosome position",
}
ENCODING = {
    "random": "Random features (1,280)",
    "one_hot": "One-hot gene",
    "codon_frequency": "Codon frequency (64)",
    "calm": "CaLM, mean-pooled",
    "genslm_25m": "GenSLM 25M, mean-pooled",
    "genslm_250m": "GenSLM 250M, mean-pooled",
    "nt_v1_2500m": "Nucleotide Transformer v1 2.5B multi-species, mean-pooled",
    "nt_v2_500m": "Nucleotide Transformer v2 500M multi-species, mean-pooled",
    "esm2": "ESM-2 650M, mean-pooled",
    "prot_t5": "ProtT5-XL-UniRef50, mean-pooled",
}
COMBINE = {"single": "", "concat": ", genes concatenated", "sum": ", genes summed"}


def board_target(name: str) -> str:
    """A target name the board accepts (letters, digits and ``_.:|+-`` only).

    ``<L>`` becomes ``mean_L`` (the angle brackets mark a population mean), ``%2N``
    becomes ``pct_2N``, ``/`` becomes ``_per_`` and any other character ``_``.
    """
    text = name
    if text.startswith("<") and text.endswith(">"):
        text = "mean_" + text[1:-1]
    text = text.replace("%", "pct_").replace("/", "_per_")
    return re.sub(r"[^A-Za-z0-9_.:|+\-]", "_", text)


def main() -> None:
    """Write one board directory per exported task and seed-0 split."""
    load_dotenv(".env")
    data_root = os.environ["DATA_ROOT"]
    boards = osp.join(data_root, ROOT, "boards")
    client = DatasetClient(DEFAULT_URL)
    for name, spec in ECOLI_TASKS.items():
        task = load_task(name)
        if task.y.shape[1] > MAX_TARGETS:
            print(f"{name}: {task.y.shape[1]:,} targets, not exported")
            continue
        stem, title, citation = PAGE[name]
        artifact = client.select(spec.dataset_slug)
        if artifact is None:
            raise SystemExit(f"{client.url} serves no {spec.dataset_slug} archive")
        source = {
            "archive": artifact.archive,
            "archive_sha256": artifact.archive_sha256,
            "archive_bytes": artifact.archive_bytes,
            "content_sha256": artifact.content_sha256,
            "url": f"{client.url}/datasets",
        }
        record_id = [str(i) for i in task.samples["record_index"]]
        targets = [board_target(t) for t in task.targets]
        if len(set(targets)) != len(targets):
            raise SystemExit(f"{name}: two targets get the same board name")
        for split_name in [s for s in task.split_names if s.endswith("_s0")]:
            family = split_name[: -len("_s0")]
            split = task.split(split_name)
            test = split.unlock_test(reason="writing the board's private labels")
            part = {int(i): "train" for i in split.train}
            part.update({int(i): "val" for i in split.val})
            part.update({int(i): "test" for i in test})
            slug = f"{name}-{family.replace('_', '-')}-s0"
            out = osp.join(boards, slug)
            os.makedirs(out, exist_ok=True)
            with open(osp.join(out, "board.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "slug": slug,
                        "title": f"{title}, {FAMILY[family]}",
                        "description": f"{spec.title}. {spec.n_targets_note}. Split {split_name} of the task {name}: {FAMILY[family]}.",
                        "loader_class": spec.dataset_class,
                        "citation_key": citation,
                        "tc_data_slug": spec.dataset_slug,
                        "docs_url": f"{SITE}/{stem}/",
                        "task": name,
                        "split": split_name,
                        "targets": targets,
                        "target_names": dict(zip(targets, task.targets)),
                        "source": source,
                    },
                    handle,
                    indent=2,
                )
                handle.write("\n")
            with open(
                osp.join(out, "splits.csv"), "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(["record_id", "split"])
                writer.writerows([record_id[i], part[i]] for i in range(len(task)))
            scored = np.concatenate([split.val, test])
            with open(
                osp.join(out, "labels.csv"), "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(["record_id", "split", "target", "value"])
                for i in scored:
                    for j, target in enumerate(targets):
                        if not np.isnan(task.y[i, j]):
                            writer.writerow(
                                [
                                    record_id[i],
                                    part[int(i)],
                                    target,
                                    repr(float(task.y[i, j])),
                                ]
                            )
            pattern = osp.join(
                data_root, ROOT, "predictions", name, split_name, "*.npz"
            )
            written = 0
            for path in sorted(glob.glob(pattern)):
                feature, combine = osp.basename(path)[: -len(".npz")].split(".")
                saved = np.load(path)
                if not (
                    np.array_equal(saved["val_index"], split.val)
                    and np.array_equal(saved["test_index"], test)
                ):
                    raise SystemExit(
                        f"{path} was written for other indices than {split_name}"
                    )
                blocks = (
                    (split.val, saved["val_prediction"]),
                    (test, saved["test_prediction"]),
                )
                if any(
                    np.ptp(block[:, j]) == 0
                    for _, block in blocks
                    for j in range(block.shape[1])
                ):
                    print(f"{slug}: {feature}.{combine} predicts a constant, skipped")
                    continue
                sub = osp.join(out, "submissions", f"{feature}.{combine}")
                os.makedirs(sub, exist_ok=True)
                with open(
                    osp.join(sub, "predictions.csv"), "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["record_id", "split", "target", "prediction"])
                    for index, block in blocks:
                        for row, i in enumerate(index):
                            for j, target in enumerate(targets):
                                if not np.isnan(task.y[i, j]):
                                    writer.writerow(
                                        [
                                            record_id[i],
                                            part[int(i)],
                                            target,
                                            repr(float(block[row, j])),
                                        ]
                                    )
                with open(
                    osp.join(sub, "metadata.json"), "w", encoding="utf-8"
                ) as handle:
                    json.dump(
                        {
                            "method_name": f"Linear probe: {ENCODING[feature].split(',')[0].split(' (')[0]}{COMBINE[combine]}",
                            "description": "Ordinary least squares on a frozen representation of each deleted gene, features standardized on train, minimum-norm solution when the fit is not unique, no hyperparameter, test scored once.",
                            "model_family": "ordinary least squares",
                            "encoding": ENCODING[feature] + COMBINE[combine],
                            "code_url": CODE,
                            "data_scope": "split_only",
                            "hyperparameters": {},
                        },
                        handle,
                        indent=2,
                    )
                    handle.write("\n")
                written += 1
            print(f"{slug}: {len(scored):,} scored records, {written} submissions")


if __name__ == "__main__":
    main()
