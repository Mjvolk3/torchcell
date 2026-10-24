# experiments/043-expression-proteome-morphology/scripts/alcf/aurora_sdpa_probe_043.py
# [[experiments.043-expression-proteome-morphology.scripts.alcf.aurora_sdpa_probe_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/alcf/aurora_sdpa_probe_043.py
"""One SDPA forward+backward at the 043 shapes on one device, for the Aurora page fault.

The 043 trunk runs SDPA over ``[C, 9, 6609, 16]`` with a ``[1, 9, 6609, 6609]`` bool head
mask and dropout 0.1; the mark operator runs ``[B, 9, 6609, 16]`` queries against
``[B, 9, M+1, 16]`` keys with a ``[B, 1, 1, M+1]`` bool mask. Jobs 8957194 and 8957239 died
on a GPU write page fault. One case per process (a GPU fault aborts the process), so the
driver (aurora_sdpa_probe_043.pbs) loops over cases and records which ones survive.

    python aurora_sdpa_probe_043.py --batch 6 --lq 6609 --lk 6609 --mask head --dropout 0.1
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def main() -> None:
    """Run the case and print one JSON line."""
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--heads", type=int, default=9)
    p.add_argument("--head-dim", type=int, default=16)
    p.add_argument("--lq", type=int, required=True)
    p.add_argument("--lk", type=int, required=True)
    p.add_argument("--mask", choices=["none", "head", "key"], required=True)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--backend", default="default")
    p.add_argument("--device", default="xpu")
    a = p.parse_args()
    dev = torch.device(a.device)
    acc = torch.xpu if dev.type == "xpu" else torch.cuda
    B, h, hd = a.batch, a.heads, a.head_dim
    q = torch.randn(B, h, a.lq, hd, device=dev, requires_grad=True)
    k = torch.randn(B, h, a.lk, hd, device=dev, requires_grad=True)
    v = torch.randn(B, h, a.lk, hd, device=dev, requires_grad=True)
    if a.mask == "head":
        mask: torch.Tensor | None = (
            torch.rand(1, h, a.lq, a.lk, device=dev) < 0.05
        ) | torch.eye(a.lq, a.lk, dtype=torch.bool, device=dev)
    elif a.mask == "key":
        mask = torch.rand(B, 1, 1, a.lk, device=dev) < 0.5
        mask[..., 0] = True
    else:
        mask = None
    acc.synchronize()
    acc.reset_peak_memory_stats()
    base = acc.memory_allocated()
    backends = (
        [b for b in SDPBackend]
        if a.backend == "default"
        else [getattr(SDPBackend, a.backend)]
    )
    t0 = time.perf_counter()
    with sdpa_kernel(backends), torch.autocast(dev.type, dtype=torch.bfloat16):
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=a.dropout
        )
    out.float().sum().backward()
    acc.synchronize()
    dt = time.perf_counter() - t0
    print(
        json.dumps(
            {
                **vars(a),
                "elements_scores": B * h * a.lq * a.lk,
                "over_int32": B * h * a.lq * a.lk >= 2**31,
                "seconds": round(dt, 3),
                "peak_extra_gb": round((acc.max_memory_allocated() - base) / 1e9, 3),
                "ok": True,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
