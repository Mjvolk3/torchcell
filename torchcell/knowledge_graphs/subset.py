# torchcell/knowledge_graphs/subset
# [[torchcell.knowledge_graphs.subset]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/subset
"""Random-subset a dataset for fast test builds -- the small-KG subsetting mechanism.

A KG test build should be a faithful *miniature* of the real build: same datasets,
same adapters, same pipeline, just fewer records per dataset so it finishes in
minutes. We subset AFTER construction by index-selecting ``size`` random records.
The dataset's ``experiment_reference_index`` is dataset-level (a handful of shared
references), so a subset keeps the full reference set while capping only the
experiment records the adapter emits -- experiment->reference edges stay consistent.

``size=None`` returns the dataset unchanged: the real KG build flips subsetting off
by setting ``subset.size: null`` in the config, running the identical script.
"""

import logging
import random
from typing import Any

log = logging.getLogger(__name__)


def subset_dataset(dataset: Any, size: int | None, seed: int = 42) -> Any:
    """Return a random subset of at most ``size`` records (seeded, reproducible).

    Returns the dataset unchanged when ``size`` is None or already >= len(dataset).
    """
    if size is None:
        return dataset
    n = len(dataset)
    if size >= n:
        log.info(
            "subset: %s has %d <= %d records; using full dataset",
            type(dataset).__name__,
            n,
            size,
        )
        return dataset
    indices = sorted(random.Random(seed).sample(range(n), size))
    log.info(
        "subset: %s %d -> %d records (seed=%d)", type(dataset).__name__, n, size, seed
    )
    return dataset[indices]
