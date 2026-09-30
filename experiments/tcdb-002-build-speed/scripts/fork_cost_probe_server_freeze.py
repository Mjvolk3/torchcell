# experiments/tcdb-002-build-speed/scripts/fork_cost_probe_server_freeze.py
# [[experiments.tcdb-002-build-speed.scripts.fork_cost_probe_server_freeze]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/fork_cost_probe_server_freeze
"""Forkserver preload hook for fork_cost_probe.py: freeze the server's heap.

Imported last in the forkserver's preload list, after torchcell, so everything the
server imported moves into the GC's permanent generation before any worker is forked
from it: the same guard CellAdapter.get_data_by_type applies to a fork parent.
"""

import gc

gc.collect()
gc.freeze()
