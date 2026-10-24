# experiments/043-expression-proteome-morphology/scripts/alcf/aurora_memwatch_043.py
# [[experiments.043-expression-proteome-morphology.scripts.alcf.aurora_memwatch_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/alcf/aurora_memwatch_043.py
"""train_043.py with XPU memory logged to W&B, and optionally a periodic empty_cache.

Aurora cells at micro-batch 2 die on a GPU write page fault 40 to 85 min into epoch 1
(PBS 8959387/8959388) at no heavy micro-batch. Hypothesis under test: reserved device
memory grows (allocator fragmentation from score matrices of varying size) until an
allocation lands unmapped.

The optimizer's ``step`` is wrapped: every ``MEMWATCH_EVERY`` optimizer steps (set it to
100 micro-steps / accumulate_steps) it logs ``ops/mem_alloc_gb``, ``ops/mem_reserved_gb``
and ``ops/mem_max_reserved_gb`` at the trainer's step (the step counter equals the number
of optimizer steps taken), prints the same line, and, with ``EMPTY_CACHE_EVERY=n``, calls
``torch.xpu.empty_cache()`` every n optimizer steps. No trainer code changes.

    MEMWATCH_EVERY=25 python aurora_memwatch_043.py <train_043.py hydra overrides>
"""

from __future__ import annotations

import os
import os.path as osp
import sys
import time

import torch

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

EVERY = int(os.environ.get("MEMWATCH_EVERY", "25"))
EMPTY_EVERY = int(os.environ.get("EMPTY_CACHE_EVERY", "0"))
T0 = time.time()
_n = 0
_orig_step = torch.optim.AdamW.step


def _step(self: torch.optim.AdamW, *args: object, **kwargs: object) -> object:
    """AdamW.step, then the memory log and the optional cache release."""
    global _n
    out = _orig_step(self, *args, **kwargs)
    _n += 1
    if EMPTY_EVERY and _n % EMPTY_EVERY == 0:
        torch.xpu.empty_cache()
    if _n % EVERY == 0:
        import wandb

        row = {
            "ops/mem_alloc_gb": torch.xpu.memory_allocated() / 1e9,
            "ops/mem_reserved_gb": torch.xpu.memory_reserved() / 1e9,
            "ops/mem_max_reserved_gb": torch.xpu.max_memory_reserved() / 1e9,
            "ops/elapsed_s": time.time() - T0,
        }
        print(
            f"[memwatch] step {_n} "
            + " ".join(f"{k.split('/')[1]} {v:.2f}" for k, v in row.items()),
            flush=True,
        )
        if wandb.run is not None:
            wandb.log(row, step=_n)
    return out


torch.optim.AdamW.step = _step  # type: ignore[method-assign]

if __name__ == "__main__":
    import train_043

    sys.argv[0] = train_043.__file__
    train_043._hydra_main()
