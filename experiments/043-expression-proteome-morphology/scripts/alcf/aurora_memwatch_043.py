# experiments/043-expression-proteome-morphology/scripts/alcf/aurora_memwatch_043.py
# [[experiments.043-expression-proteome-morphology.scripts.alcf.aurora_memwatch_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/alcf/aurora_memwatch_043.py
"""train_043.py with a thread that prints XPU memory every ``MEMWATCH_SECONDS`` (60).

Aurora cells at micro-batch 2 die on a GPU write page fault 40 to 85 min into epoch 1
(PBS 8959387/8959388) at no heavy micro-batch. Hypothesis under test: reserved device
memory grows (allocator fragmentation from score matrices of varying size) until an
allocation lands unmapped. Each line: elapsed s, memory_allocated, memory_reserved,
max_memory_allocated, max_memory_reserved (GB), and the device's free / total.

    python aurora_memwatch_043.py <train_043.py hydra overrides>
"""

from __future__ import annotations

import os
import os.path as osp
import sys
import threading
import time

import torch

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))


def _watch(period: float) -> None:
    t0 = time.time()
    while True:
        time.sleep(period)
        if not torch.xpu.is_initialized():
            continue
        free, total = torch.xpu.mem_get_info()
        print(
            f"[memwatch] {time.time() - t0:.0f} s allocated "
            f"{torch.xpu.memory_allocated() / 1e9:.2f} reserved "
            f"{torch.xpu.memory_reserved() / 1e9:.2f} max_allocated "
            f"{torch.xpu.max_memory_allocated() / 1e9:.2f} max_reserved "
            f"{torch.xpu.max_memory_reserved() / 1e9:.2f} device_free "
            f"{free / 1e9:.2f} of {total / 1e9:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    threading.Thread(
        target=_watch,
        args=(float(os.environ.get("MEMWATCH_SECONDS", "60")),),
        daemon=True,
    ).start()
    import train_043

    sys.argv[0] = train_043.__file__
    train_043._hydra_main()
