# tests/torchcell/adapters/test_bacterial_adapters.py
# [[tests.torchcell.adapters.test_bacterial_adapters]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/test_bacterial_adapters.py
"""The bacterial adapters as a set (plan.bacteria-ontology-genome step 9).

Each adapter's own checks live in its paired ``test_<module>.py``. Here: every
registered E. coli and P. putida dataset class is in ``dataset_adapter_map`` with its
adapter and nothing else is; each conf is named after its loader's root slug; and
``kg_bacteria.yaml`` names exactly the registered bacterial classes, in full, as a
generation-only rehearsal.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import yaml

from tests.torchcell.adapters._bacterial_adapter_cases import (
    BACTERIAL,
    BACTERIAL_PACKAGES,
    KG_BACTERIA,
    registered_bacterial_classes,
)
from torchcell.datasets.dataset_registry import dataset_registry
from torchcell.knowledge_graphs.dataset_adapter_map import dataset_adapter_map


def test_every_registered_bacterial_dataset_is_mapped_to_its_adapter() -> None:
    expected = {b.case.dataset_cls: b.case.adapter_cls for b in BACTERIAL}
    # 59 + 1 (Teteneva W3110) + 6 (#844's six P. putida classes) + 2 (#834's two)
    # + 3 (#826: Schmidt Table S23, Wang's no-isoprenol column, Lamoureux's growth
    # rate) + 1 (#770's four Lim 2025 proteome contrasts) + 1 (Wang 2024 rifampicin
    # Tn-seq) + 1 (row 53, Royet 2025's metal Tn-seq) + 2 (Thompson 2019 valerolactam:
    # its titer and its growth-rate families) + 1 (Hawkins 2020 mismatch-CRISPRi)
    # + 4 (Brunk 2016's four released scales) + 1 (#854: Balakrishnan 2022 mRNA number
    # fractions) + 1 (#857, Li 2014 synthesis rates), re-derived from the merged conf and map.
    assert (
        len(expected)
        == len(BACTERIAL)
        == 59 + 1 + 6 + 2 + 3 + 1 + 1 + 1 + 2 + 1 + 4 + 1 + 1
    )
    assert registered_bacterial_classes() == set(expected)
    mapped = {
        ds: ad
        for ds, ad in dataset_adapter_map.items()
        if ds.__module__.startswith(BACTERIAL_PACKAGES)
    }
    assert mapped == expected


def test_each_adapter_has_its_own_module() -> None:
    """One class per module, so ``kg_manifest`` reads each dataset's own conf."""
    modules = [b.case.adapter_cls.__module__ for b in BACTERIAL]
    assert len(set(modules)) == 59 + 1 + 6 + 2 + 3 + 1 + 1 + 1 + 2 + 1 + 4 + 1 + 1
    assert modules == [f"torchcell.adapters.{b.module}" for b in BACTERIAL]


def test_each_conf_is_named_after_the_dataset_root_slug() -> None:
    """``<slug>_adapter.yaml`` where ``data/torchcell/<slug>`` is the loader's root."""
    for bacterial in BACTERIAL:
        params = inspect.signature(bacterial.case.dataset_cls.__init__).parameters
        slug = Path(params["root"].default).name
        assert bacterial.case.conf_name == f"{slug}_adapter.yaml"


def test_kg_bacteria_names_exactly_the_registered_bacterial_datasets() -> None:
    conf = yaml.safe_load(KG_BACTERIA.read_text(encoding="utf-8"))
    names = conf["datasets"]
    assert (
        len(names) == len(set(names)) == 59 + 1 + 6 + 2 + 3 + 1 + 1 + 1 + 2 + 1 + 4 + 1 + 1
    )
    classes = [dataset_registry[name] for name in names]
    assert all(cls in dataset_adapter_map for cls in classes)
    assert set(classes) == registered_bacterial_classes()
    assert names == [b.case.dataset_cls.__name__ for b in BACTERIAL]
    assert conf["import_mode"] == "full"
    assert conf["subset"] == {
        "size": None,
        "seed": 42,
        "per_dataset": {},
        "prefilter": {},
    }
    assert conf["defaults"] == ["default", "_self_"]
