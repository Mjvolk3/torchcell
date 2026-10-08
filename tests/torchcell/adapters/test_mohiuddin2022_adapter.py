# tests/torchcell/adapters/test_mohiuddin2022_adapter.py
# [[tests.torchcell.adapters.test_mohiuddin2022_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_mohiuddin2022_adapter.py
"""``torchcell/adapters/mohiuddin2022_adapter.py``: the conf it loads, the gate's view
of it, and (``--data``) the graph it emits from its dev-tree LMDB.

Mohiuddin 2022 promoter-GFP readings: 69,480 ``PromoterActivityExperiment`` records, the
first consumer of the ``promoter activity phenotype`` node class. Both optional pairs
are on -- every genotype carries the episomal reporter, and the treated arms carry their
antibiotic from hour five -- so this conf is the perturbation-bearing shape.

The property pinned hardest is that the conf names the NEW phenotype methods and not an
environment-response or expression method: this family exists because neither of those
can carry an untreated reporter reading, and a conf naming the wrong pair would make
BioCypher drop the phenotype nodes without an error.
"""

from __future__ import annotations

import pytest
import yaml

from tests.torchcell.adapters._adapter_init_harness import (
    assert_construction,
    assert_missing_conf,
)
from tests.torchcell.adapters._bacterial_adapter_cases import (
    REPO_ROOT,
    adapter_module,
    assert_conf_registered_and_declared,
    assert_dev_store_graph,
    assert_gate_resolves_own_files,
    case_for,
)
from torchcell.datasets.ecoli.mohiuddin2022 import PromoterReporterMohiuddin2022Dataset

CASE = case_for(PromoterReporterMohiuddin2022Dataset)


def _conf_method_names() -> list[str]:
    conf = yaml.safe_load(
        (REPO_ROOT / "torchcell/adapters/conf" / CASE.case.conf_name).read_text(
            encoding="utf-8"
        )
    )
    return [
        method["method_name"]
        for method in conf["cell_adapter"]["node_methods"]
        + conf["cell_adapter"]["edge_methods"]
    ]


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


def test_conf_names_the_promoter_activity_pair_and_no_other_phenotype() -> None:
    """A conf naming the wrong phenotype pair drops every phenotype node silently."""
    names = _conf_method_names()
    phenotype = [n for n in names if "phenotype" in n]
    assert phenotype == [
        "promoter activity phenotype (chunked)",
        "promoter activity phenotype reference",
        "phenotype to experiment (chunked)",
        "phenotype to experiment reference",
    ]


def test_conf_enables_both_the_gene_and_the_environment_perturbation_pairs() -> None:
    """Every record carries the reporter; the treated arms carry their antibiotic."""
    names = _conf_method_names()
    assert "bacterial perturbation (chunked)" in names
    assert "perturbation to genotype (chunked)" in names
    assert "environment perturbation (chunked)" in names
    assert "environment perturbation to environment (chunked)" in names
    assert [n for n in names if "crispr" in n] == []
    assert [n for n in names if "phage" in n] == []


@pytest.mark.data
@pytest.mark.timeout(1800)
def test_dev_store_emits_a_closed_declared_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Over the suite's 300 s default, because this store has 17,370 REFERENCES.

    A record's reference is its own well in the untreated arm at its own read hour,
    which is what makes the released fold change recoverable from one record, and it
    makes the reference index 284 MB (30.2 s to load, measured). Six reference
    collectors then each walk all 17,370 entries and sha256 a ``model_dump`` that
    carries the whole LB media object. The cost is bought deliberately; the only
    cheaper reference would drop the untreated reading the fold change divides by.
    """
    assert_dev_store_graph(CASE, monkeypatch)
