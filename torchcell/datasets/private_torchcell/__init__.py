# torchcell/datasets/private_torchcell/__init__
# [[torchcell.datasets.private_torchcell]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/private_torchcell

"""In-house wet-lab datasets, which stay in-house.

Every dataset in this package sets ``visibility = Visibility.private``
(``torchcell.data.experiment_dataset``). A private dataset is:

- never served on the public knowledge graph. The build refuses any dataset whose class
  ``visibility`` is private unless it is run with ``--include-private``, and the public
  ``dataset_adapter_map`` does not list it; private loaders register their adapters in
  ``PRIVATE_DATASET_ADAPTER_MAP`` instead
  (``torchcell.knowledge_graphs.dataset_adapter_map``).
- never published in a ``tc-data`` release. ``scripts/package_dataset_lmdb.py`` refuses
  a dataset directory whose build manifest names a private loader class, so there is no
  path by which one reaches the artifact store.

The data itself is in-house: measurements made in our own lab, whose source document is
a dissertation or a preliminary-exam report with no DOI
(``Publication.source_type`` ``dissertation`` / ``preliminary_report`` / ``in_house``,
identified by title plus the deposited document's mirror-relative path and sha256).

Filtering is exact, not heuristic: every record carries ``dataset_name``, the dataset
class's own name, so the private records in any store are exactly those whose
``dataset_name`` names a class in this package.
"""

from .volk2021_inhibitor_bioscreen import (
    InhibitorBioscreenVolk2021Dataset as InhibitorBioscreenVolk2021Dataset,
)

private_datasets = ["InhibitorBioscreenVolk2021Dataset"]
