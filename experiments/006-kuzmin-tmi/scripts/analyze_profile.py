#!/usr/bin/env python
"""
Analyze PyTorch profiler trace files to identify training bottlenecks.

Usage:
    python analyze_profile.py <trace_file.pt.trace.json>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def analyze_trace(trace_file):
    """Analyze PyTorch profiler trace and print bottleneck summary."""
    print(f"Loading trace file: {trace_file}")
    print("(This may take 30-60 seconds for large files...)\n")

    with open(trace_file, 'r') as f:
        data = json.load(f)

    # Extract trace events
    events = data.get('traceEvents', [])
    print(f"Total trace events: {len(events):,}\n")

    # Categorize events by phase and name
    cpu_time = defaultdict(float)
    cuda_time = defaultdict(float)

    # Parse events
    for event in events:
        name = event.get('name', '')
        dur = event.get('dur', 0)  # Duration in microseconds
        cat = event.get('cat', '')
        ph = event.get('ph', '')  # Event phase

        # Only look at complete events (X phase)
        if ph != 'X':
            continue

        # Categorize by CPU vs CUDA
        if 'cuda' in cat.lower() or 'kernel' in cat.lower():
            cuda_time[name] += dur
        else:
            cpu_time[name] += dur

    # Convert to milliseconds and sort
    cpu_time_ms = {k: v/1000 for k, v in cpu_time.items()}
    cuda_time_ms = {k: v/1000 for k, v in cuda_time.items()}

    # Calculate totals
    total_cpu = sum(cpu_time_ms.values())
    total_cuda = sum(cuda_time_ms.values())

    print("="*80)
    print("PROFILER SUMMARY")
    print("="*80)
    print(f"Total CPU time:  {total_cpu:,.2f} ms")
    print(f"Total CUDA time: {total_cuda:,.2f} ms")
    print(f"CPU/CUDA ratio:  {total_cpu/total_cuda:.2f}x" if total_cuda > 0 else "N/A")
    print()

    # Top CPU operations
    print("="*80)
    print("TOP 20 CPU OPERATIONS (Data Loading & Python Overhead)")
    print("="*80)
    print(f"{'Operation':<60} {'Time (ms)':>15} {'%':>5}")
    print("-"*80)

    for name, time_ms in sorted(cpu_time_ms.items(), key=lambda x: x[1], reverse=True)[:20]:
        pct = (time_ms / total_cpu * 100) if total_cpu > 0 else 0
        print(f"{name[:60]:<60} {time_ms:>15,.2f} {pct:>5.1f}%")

    print()

    # Top CUDA operations
    print("="*80)
    print("TOP 20 CUDA OPERATIONS (GPU Kernels)")
    print("="*80)
    print(f"{'Operation':<60} {'Time (ms)':>15} {'%':>5}")
    print("-"*80)

    for name, time_ms in sorted(cuda_time_ms.items(), key=lambda x: x[1], reverse=True)[:20]:
        pct = (time_ms / total_cuda * 100) if total_cuda > 0 else 0
        print(f"{name[:60]:<60} {time_ms:>15,.2f} {pct:>5.1f}%")

    print()

    # Look for specific bottleneck indicators
    print("="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)

    # Data loading indicators
    data_loading_ops = [
        'enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__',
        'LazySubgraphRepresentation',
        '_process_gene_interactions',
        'collate',
        'batch.to',
    ]

    data_loading_time = 0
    for op in data_loading_ops:
        for name, time_ms in cpu_time_ms.items():
            if op in name:
                data_loading_time += time_ms

    # DDP communication
    ddp_comm_time = 0
    for name, time_ms in cuda_time_ms.items():
        if 'nccl' in name.lower() or 'allreduce' in name.lower():
            ddp_comm_time += time_ms

    # Model forward/backward
    model_forward_time = 0
    for name, time_ms in cuda_time_ms.items():
        if any(x in name.lower() for x in ['conv', 'matmul', 'gemm', 'attention']):
            model_forward_time += time_ms

    print(f"Data Loading Time:     {data_loading_time:>10,.2f} ms  ({data_loading_time/total_cpu*100:>5.1f}% of CPU)")
    print(f"DDP Communication:     {ddp_comm_time:>10,.2f} ms  ({ddp_comm_time/total_cuda*100:>5.1f}% of CUDA)" if total_cuda > 0 else "N/A")
    print(f"Model Forward/Back:    {model_forward_time:>10,.2f} ms  ({model_forward_time/total_cuda*100:>5.1f}% of CUDA)" if total_cuda > 0 else "N/A")
    print()

    # Bottleneck determination
    if data_loading_time > total_cuda * 0.5:
        print("⚠️  BOTTLENECK: DATA LOADING")
        print("    Data loading time is significant compared to GPU computation.")
        print("    Consider: saving masks to disk, increasing num_workers, or prefetching.")
    elif ddp_comm_time > total_cuda * 0.3:
        print("⚠️  BOTTLENECK: DDP COMMUNICATION")
        print("    DDP gradient synchronization is taking significant time.")
        print("    Consider: gradient accumulation, larger batch size, or gradient compression.")
    elif total_cpu > total_cuda * 2:
        print("⚠️  BOTTLENECK: CPU OVERHEAD")
        print("    CPU operations dominate. GPU is underutilized.")
    else:
        print("✅  BALANCED: No obvious bottleneck detected.")
        print("    Training appears well-optimized.")

    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    trace_file = Path(sys.argv[1])
    if not trace_file.exists():
        print(f"Error: File not found: {trace_file}")
        sys.exit(1)

    analyze_trace(trace_file)
