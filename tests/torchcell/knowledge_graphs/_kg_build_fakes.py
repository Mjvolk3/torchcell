# tests/torchcell/knowledge_graphs/_kg_build_fakes.py
# [[tests.torchcell.knowledge_graphs._kg_build_fakes]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/knowledge_graphs/_kg_build_fakes.py
"""Shared stand-ins for the BioCypher build scripts' tests (not a test module).

The build scripts (``create_kg``, ``create_scerevisiae_kg``,
``create_scerevisiae_kg_small``, ``gene_interactions_scerevisae_kg``) orchestrate three
things the tests replace at the module boundary: ``BioCypher`` (``FakeBioCypher`` records
its constructor kwargs and every write call in order), ``wandb`` (``FakeWandb`` records
``init``, ``log`` payloads, ``run.log_code`` and ``finish``, and exposes the config the
scripts read back through ``wandb.config.<section>[key]``), and the wall clock
(``FixedDatetime.now`` is pinned to 2026-09-27 08:30:05 so the output directory is a
fixed string, ``TickingTime.time`` returns 0.0, 1.0, 2.0, ... so every measured duration
is exactly 1.0). ``FakeDataset`` / ``FakeAdapter`` are the loader and adapter stand-ins
the scripts wire together; each records its constructor kwargs.

``import_build_module`` imports a build module and restores the whole process
environment afterwards: the scripts assign ``certifi.where()`` to ``SSL_CERT_FILE`` at
import and ``create_kg`` runs ``load_dotenv()`` there, and neither mutation may leak into
the rest of the session (the reason ``test_import_all`` never imports them).
"""

from __future__ import annotations

import importlib
import os
import os.path as osp
from collections.abc import Iterator
from datetime import datetime
from types import ModuleType, SimpleNamespace
from typing import Any

FIXED_NOW = datetime(2026, 9, 27, 8, 30, 5)
TIME_STR = "2026-09-27_08-30-05"


def import_build_module(name: str) -> ModuleType:
    """Import ``torchcell.knowledge_graphs.<name>`` with the environment restored.

    The scripts assign ``SSL_CERT_FILE`` at import, and ``create_kg`` also runs
    ``load_dotenv()`` there, which adds every key of a repo ``.env`` that is not already
    set; the whole environment is snapshotted before the import and put back after.
    """
    saved = dict(os.environ)
    try:
        return importlib.import_module(f"torchcell.knowledge_graphs.{name}")
    finally:
        os.environ.clear()
        os.environ.update(saved)


def module_dir(module: ModuleType) -> str:
    """The directory of a loaded module's file (what ``wandb.run.log_code`` receives)."""
    path = module.__file__
    if path is None:
        raise AssertionError(f"{module.__name__} has no __file__")
    return osp.dirname(path)


class FakeBioCypher:
    """Records the constructor kwargs and every write call, in order."""

    instances: list[FakeBioCypher] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._output_directory = str(kwargs["output_directory"])
        self.calls: list[tuple[Any, ...]] = []
        FakeBioCypher.instances.append(self)

    def write_nodes(self, nodes: Any) -> None:
        self.calls.append(("write_nodes", list(nodes)))

    def write_edges(self, edges: Any) -> None:
        self.calls.append(("write_edges", list(edges)))

    def write_import_call(self) -> None:
        self.calls.append(("write_import_call",))

    def write_schema_info(self, as_node: bool) -> None:
        self.calls.append(("write_schema_info", as_node))


class _FakeRun:
    def __init__(self) -> None:
        self.log_code_calls: list[str] = []

    def log_code(self, path: str) -> None:
        self.log_code_calls.append(path)


class FakeWandb:
    """Records ``init`` kwargs, ``log`` payloads and ``finish``; serves ``config``."""

    def __init__(self) -> None:
        self.init_kwargs: list[dict[str, Any]] = []
        self.logged: list[dict[str, Any]] = []
        self.finish_calls = 0
        self.run = _FakeRun()
        self.config = SimpleNamespace()

    def init(self, **kwargs: Any) -> None:
        self.init_kwargs.append(kwargs)
        self.config = SimpleNamespace(**kwargs["config"])

    def log(self, payload: dict[str, Any]) -> None:
        self.logged.append(payload)

    def finish(self) -> None:
        self.finish_calls += 1


class FixedDatetime:
    """``datetime`` stand-in whose ``now`` is pinned to ``FIXED_NOW``."""

    @staticmethod
    def now() -> datetime:
        return FIXED_NOW


class TickingTime:
    """``time`` stand-in whose ``time`` returns 0.0, 1.0, 2.0, ... per call."""

    def __init__(self) -> None:
        self.calls = 0

    def time(self) -> float:
        value = float(self.calls)
        self.calls += 1
        return value


class FakeDataset:
    """A loader stand-in: records its kwargs, has a fixed length, indexes to a view."""

    n_records: int = 3
    instances: list[FakeDataset] = []

    def __init__(self, root: str, **kwargs: Any) -> None:
        self.root = root
        self.kwargs = kwargs
        self.selected: list[int] | None = None
        type(self).instances.append(self)

    def __len__(self) -> int:
        return len(self.selected) if self.selected is not None else self.n_records

    def __getitem__(self, indices: list[int]) -> FakeDataset:
        view = type(self).__new__(type(self))
        view.root = self.root
        view.kwargs = self.kwargs
        view.selected = list(indices)
        return view


class FakeAdapter:
    """An adapter stand-in: records its kwargs, yields one node and one edge per record."""

    instances: list[FakeAdapter] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        type(self).instances.append(self)

    def get_nodes(self) -> Iterator[tuple[str, str, int]]:
        for i in range(len(self.kwargs["dataset"])):
            yield ("node", type(self).__name__, i)

    def get_edges(self) -> Iterator[tuple[str, str, int]]:
        for i in range(len(self.kwargs["dataset"])):
            yield ("edge", type(self).__name__, i)


def reset_instances(*classes: Any) -> None:
    """Clear the per-class ``instances`` lists before a run."""
    for cls in classes:
        cls.instances = []
