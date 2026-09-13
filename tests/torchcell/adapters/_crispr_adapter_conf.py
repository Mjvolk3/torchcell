# tests/torchcell/adapters/_crispr_adapter_conf.py
"""Shared helpers for the CRISPR-dataset adapter conf tests (not a test module)."""

from __future__ import annotations

import inspect
import re

import yaml

from torchcell.adapters.cell_adapter import CellAdapter


def adapter_method_names() -> tuple[set[str], set[str]]:
    """Every node and edge method name ``CellAdapter.__init__`` registers.

    Read from the source because building the tables means constructing an adapter,
    which starts a wandb run.
    """
    source = inspect.getsource(CellAdapter.__init__)
    node_block, edge_block = source.split("self.edge_methods = [", 1)
    node_block = node_block.split("self.node_methods = [", 1)[1]
    pattern = r'"([^"]+)",\s*\n?\s*self\._'
    return set(re.findall(pattern, node_block)), set(re.findall(pattern, edge_block))


def conf_methods(path: str) -> tuple[list[str], list[str]]:
    """The node and edge method names an adapter conf enables."""
    with open(path) as handle:
        conf = yaml.safe_load(handle)["cell_adapter"]
    return (
        [method["method_name"] for method in conf["node_methods"]],
        [method["method_name"] for method in conf["edge_methods"]],
    )
