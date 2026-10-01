# tests/torchcell/trainers/test_019_train_cgt_multitask.py
# [[tests.torchcell.trainers.test_019_train_cgt_multitask]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/trainers/test_019_train_cgt_multitask.py
"""Batch sizing and target/mask decoding of the 019 multitask CGT script (issue #567).

``experiments/019-simb-multimodal/scripts/train_cgt_multitask.py`` is a script, not a
package module, so it is loaded by path with ``dotenv.load_dotenv`` stubbed (it runs at
import). ``MultitaskCGTTask._batch_size`` and ``_extract_targets_and_masks`` are called
unbound on a namespace carrying only the attributes they read, so no model is built.

The batch has three genotypes: 0 perturbs genes {1, 2}, 1 perturbs {3}, 2 is the wild
type and perturbs nothing, so ``perturbation_indices_batch`` is [0, 0, 1]. Genotypes 0
and 2 carry a fitness value and a 2-feature morphology vector; genotype 1 carries none.
The model emits ``num_graphs`` = 3 rows, so every target and mask has 3 rows and row 2's
measurements are decoded. With the old ``max + 1`` sizing (2) the mask had 2 rows against
a 3-row target and row 2 was never decoded.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch_geometric.data import HeteroData

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "019-simb-multimodal"
    / "scripts"
    / "train_cgt_multitask.py"
)


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The 019 script loaded by path, with ``load_dotenv`` a no-op."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    spec = importlib.util.spec_from_file_location("train_cgt_multitask_019", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "train_cgt_multitask_019", module)
    spec.loader.exec_module(module)
    return module


def _wild_type_last() -> HeteroData:
    batch = HeteroData()
    gene = batch["gene"]
    gene.perturbation_indices = torch.tensor([1, 2, 3])
    gene.perturbation_indices_batch = torch.tensor([0, 0, 1])
    # COO phenotypes: row 0 has fitness 0.5 (sample 0) and morphology [1, 2] (sample 1);
    # row 2 has fitness -0.25 (sample 0) and morphology [3, 4] (sample 1).
    gene.phenotype_values = torch.tensor([0.5, 1.0, 2.0, -0.25, 3.0, 4.0])
    gene.phenotype_type_indices = torch.tensor([0, 1, 1, 0, 1, 1])
    gene.phenotype_values_batch = torch.tensor([0, 0, 0, 2, 2, 2])
    gene.phenotype_sample_indices = torch.tensor([0, 1, 1, 0, 1, 1])
    gene.phenotype_types = [
        ["fitness", "morphology"],
        ["fitness", "morphology"],
        ["fitness", "morphology"],
    ]
    batch.num_graphs = 3  # a collated PyG Batch carries it
    return batch


def _task_attrs() -> SimpleNamespace:
    return SimpleNamespace(
        head_phenotypes={"gene_interaction": ["fitness"], "global": ["morphology"]},
        head_align={
            "gene_interaction": {"is_scalar": True, "raw_dim": 1},
            "global": {"is_scalar": False, "raw_dim": 2},
        },
        dist_heads={},
        norm_heads=set(),
    )


def test_batch_size_is_num_graphs_with_a_trailing_wild_type(script: ModuleType) -> None:
    """3 genotypes, although max([0, 0, 1]) + 1 = 2."""
    assert script.MultitaskCGTTask._batch_size(_task_attrs(), _wild_type_last()) == 3


def test_targets_and_masks_cover_the_trailing_wild_type_row(script: ModuleType) -> None:
    """Rows 0 and 2 are supervised with their exact values; row 1 is masked out."""
    task_cls = script.MultitaskCGTTask
    attrs = _task_attrs()
    batch = _wild_type_last()
    bsz = task_cls._batch_size(attrs, batch)
    head_outputs = {"gene_interaction": torch.zeros(3), "global": torch.zeros(3, 2)}
    targets, masks = task_cls._extract_targets_and_masks(
        attrs, batch, head_outputs, bsz
    )
    assert masks["gene_interaction"].tolist() == [True, False, True]
    assert masks["global"].tolist() == [True, False, True]
    assert targets["gene_interaction"].tolist() == [0.5, 0.0, -0.25]
    assert targets["global"].tolist() == [[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]]
