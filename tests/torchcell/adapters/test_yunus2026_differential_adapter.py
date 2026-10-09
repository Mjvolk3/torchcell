# tests/torchcell/adapters/test_yunus2026_differential_adapter.py
# [[tests.torchcell.adapters.test_yunus2026_differential_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_yunus2026_differential_adapter.py
"""``torchcell/adapters/yunus2026_differential_adapter.py``: the conf it loads, the
gate's view of it, and (``--data``) the graph it emits from its dev-tree LMDB.

Yunus 2026 PP_4188 differential proteome: 1 IY1452 strain,
BacterialProteinFoldChangeExperiment over 305 released protein keys, each carrying its
released unadjusted equal-variance t-test p-value. One
BacterialCrisprInterferencePerturbation carrying a CrisprConstruct, served as `bacterial
perturbation`, and the environment carries EnvironmentPhysicalPerturbation and
SmallMoleculePerturbation edits.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import torchcell.adapters
from tests.torchcell.adapters._adapter_init_harness import (
    assert_construction,
    assert_missing_conf,
)
from tests.torchcell.adapters._bacterial_adapter_cases import (
    adapter_module,
    assert_conf_registered_and_declared,
    assert_dev_store_graph,
    assert_gate_resolves_own_files,
    case_for,
)
from torchcell.datasets.pputida.yunus2026 import (
    CrispriDifferentialProteomeYunus2026Dataset,
)

CONF_NAME = "crispri_differential_proteome_yunus2026_adapter.yaml"
CASE = case_for(CrispriDifferentialProteomeYunus2026Dataset)


def test_init_serves_the_exact_conf_and_wires_the_base_adapter(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert_construction(CASE.case, monkeypatch, capsys)


def test_init_refuses_a_missing_conf_before_wandb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_missing_conf(CASE.case, adapter_module(CASE), monkeypatch)


def test_conf_methods_are_registered_and_their_classes_declared() -> None:
    assert_conf_registered_and_declared(CASE)


def test_kg_manifest_resolves_the_dataset_to_its_own_module_and_conf() -> None:
    assert_gate_resolves_own_files(CASE)


@pytest.mark.data
def test_dev_store_emits_a_closed_declared_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_dev_store_graph(CASE, monkeypatch)


def test_conf_serves_the_fold_change_phenotype_and_not_the_absolute_one() -> None:
    """The conf's phenotype pair is the RELATIVE family, exactly.

    Issue #770: this dataset stores a ratio to a control strain, which
    ``ProteinAbundancePhenotype`` forbids in its own docstring, so the conf must enable
    the fold-change node methods and neither of the protein-abundance ones.
    """
    conf = yaml.safe_load(
        (Path(torchcell.adapters.__file__).parent / "conf" / CONF_NAME).read_text(
            encoding="utf-8"
        )
    )
    names = [m["method_name"] for m in conf["cell_adapter"]["node_methods"]]
    assert [n for n in names if "phenotype" in n] == [
        "protein fold change phenotype (chunked)",
        "protein fold change phenotype reference",
    ]
    assert "protein abundance phenotype (chunked)" not in names
    assert "protein abundance phenotype reference" not in names
