# torchcell/tasks/__init__.py
# [[torchcell.tasks.__init__]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/tasks/__init__.py
"""Standardized benchmark tasks: one dataset, its genome, its labels and pinned splits.

The short import a benchmark script uses::

    from torchcell.tasks import load_task

    task = load_task("ecoli-growth-rate-campos2018")
    split = task.split("random_s0")
    # features: your own array with one row per sample, in the order of task.genes
    x_train, y_train = features[split.train], task.y[split.train]

``split.test`` is locked: it raises until ``split.unlock_test(reason=...)`` is called,
and that call warns. The modules:

- :mod:`torchcell.tasks.splits`: the split functions (random, grouped, pair).
- :mod:`torchcell.tasks.task`: the task and split objects, the test lock, the metrics.
- :mod:`torchcell.tasks.ecoli`: the four *E. coli* tasks and the helper functions that
  match a dataset's records to genes of its genome.
"""

from torchcell.tasks.ecoli import ECOLI_TASKS, load_task
from torchcell.tasks.task import (
    BenchmarkTask,
    Split,
    TaskSpec,
    TestSplitLocked,
    TestSplitWarning,
    regression_metrics,
)

__all__ = [
    "ECOLI_TASKS",
    "BenchmarkTask",
    "Split",
    "TaskSpec",
    "TestSplitLocked",
    "TestSplitWarning",
    "load_task",
    "regression_metrics",
]
