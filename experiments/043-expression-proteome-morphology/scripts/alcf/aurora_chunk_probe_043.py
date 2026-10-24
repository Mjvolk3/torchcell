# experiments/043-expression-proteome-morphology/scripts/alcf/aurora_chunk_probe_043.py
# [[experiments.043-expression-proteome-morphology.scripts.alcf.aurora_chunk_probe_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/alcf/aurora_chunk_probe_043.py
"""Is query_chunked_sdpa exact on XPU? One call against query blocks, at the trunk shape.

Job 8959230: every chunked 043 run on Aurora stopped on a non-finite gradient norm within
2 to 68 steps. This compares, under bf16 autocast and dropout 0, the one SDPA call with
(a) query blocks without checkpointing, (b) query blocks with contiguous slices, and (c)
query_chunked_sdpa (blocks under checkpointing), on q/k/v laid out as the model lays them
out ([B, N, H, hd] transposed to [B, H, N, hd], so a query slice is non-contiguous). It
prints the maximum absolute difference of the output and of each input gradient, and
whether anything is non-finite.
"""

from __future__ import annotations

import json
import sys

import torch
import torch.nn.functional as F

from torchcell.models.equivariant_cell_graph_transformer import query_chunked_sdpa


def blocks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    chunk: int,
    contiguous: bool,
    p: float = 0.0,
) -> torch.Tensor:
    """Query blocks, no checkpointing."""
    outs = []
    for s in range(0, q.shape[-2], chunk):
        qc = q[..., s : s + chunk, :]
        m = None if mask is None else mask[..., s : s + chunk, :]
        if contiguous:
            qc = qc.contiguous()
            m = None if m is None else m.contiguous()
        outs.append(F.scaled_dot_product_attention(qc, k, v, attn_mask=m, dropout_p=p))
    return torch.cat(outs, dim=-2)


def main() -> None:
    """Run the four variants on xpu (or cuda) and print one JSON line per variant."""
    dev = torch.device("xpu" if torch.xpu.is_available() else "cuda")
    p = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    torch.manual_seed(0)
    B, N, H, hd, chunk = 2, 6609, 9, 16, 1024
    x = torch.randn(B, N, H * hd, device=dev)
    proj = torch.nn.Linear(H * hd, 3 * H * hd).to(dev)
    mask = (torch.rand(1, H, N, N, device=dev) < 0.05) | torch.eye(
        N, dtype=torch.bool, device=dev
    )
    mask[..., 0] = True
    w = torch.randn(B, H, N, hd, device=dev)

    def run(kind: str) -> tuple[torch.Tensor, list[torch.Tensor]]:
        proj.zero_grad()
        xx = x.clone().requires_grad_(True)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            qkv = proj(xx).view(B, N, 3, H, hd)
            q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
            if kind == "one":
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=p
                )
            elif kind == "blocks":
                out = blocks(q, k, v, mask, chunk, contiguous=False, p=p)
            elif kind == "blocks_contig":
                out = blocks(q, k, v, mask, chunk, contiguous=True, p=p)
            else:
                out = query_chunked_sdpa(q, k, v, mask, p, chunk)
        (out.float() * w).sum().backward()
        assert xx.grad is not None and proj.weight.grad is not None
        return out.float().detach(), [xx.grad.clone(), proj.weight.grad.clone()]

    ref_out, ref_g = run("one")
    for kind in ("one", "blocks", "blocks_contig", "checkpointed"):
        out, g = run(kind)
        print(
            json.dumps(
                {
                    "variant": kind,
                    "dropout": p,
                    "out_finite": bool(torch.isfinite(out).all()),
                    "grads_finite": [bool(torch.isfinite(t).all()) for t in g],
                    "max_abs_out": float((out - ref_out).abs().max()),
                    "max_abs_grad": [
                        float((a - b).abs().max())
                        for a, b in zip(g, ref_g, strict=True)
                    ],
                    "ref_grad_scale": [float(t.abs().max()) for t in ref_g],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
