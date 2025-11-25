#!/usr/bin/env python
"""
Detailed PyTorch profiler trace analysis with operation categorization.

Usage:
    python analyze_profile_detailed.py <trace_file.pt.trace.json>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def categorize_operation(name, category):
    """Categorize operations into meaningful groups."""
    name_lower = name.lower()
    cat_lower = category.lower()

    # Data loading operations
    if any(x in name_lower for x in ['dataloader', 'getitem', 'collate', 'worker']):
        return 'data_loading'

    # Graph processing
    if any(x in name_lower for x in ['lazysubgraph', 'subgraph', 'process_gene', 'mask']):
        return 'graph_processing'

    # DDP communication
    if any(x in name_lower for x in ['nccl', 'allreduce', 'allgather', 'broadcast', 'dist']):
        return 'ddp_communication'

    # Model forward pass
    if any(x in name_lower for x in ['forward', 'conv', 'gin', 'gat', 'attention', 'embedding']):
        return 'model_forward'

    # Loss computation
    if any(x in name_lower for x in ['loss', 'mse', 'wasserstein', 'supcr', 'criterion']):
        return 'loss_computation'

    # Backward pass
    if any(x in name_lower for x in ['backward', 'autograd', 'accumulategrad']):
        return 'backward'

    # Optimizer
    if any(x in name_lower for x in ['optimizer', 'adamw', 'step']):
        return 'optimizer'

    # Memory operations
    if any(x in name_lower for x in ['cudamalloc', 'cudafree', 'allocat']):
        return 'memory'

    # Tensor operations
    if any(x in name_lower for x in ['aten::', 'matmul', 'mul', 'add', 'gather', 'scatter', 'index']):
        return 'tensor_ops'

    # CUDA kernels
    if 'cuda' in cat_lower or 'kernel' in cat_lower or 'culaunch' in name_lower:
        return 'cuda_kernels'

    return 'other'


def analyze_trace(trace_file):
    """Analyze PyTorch profiler trace with detailed categorization."""
    print(f"Loading trace file: {trace_file}")
    print("(This may take 30-60 seconds for large files...)\n")

    with open(trace_file, 'r') as f:
        data = json.load(f)

    events = data.get('traceEvents', [])
    print(f"Total trace events: {len(events):,}\n")

    # Categorize events
    category_times = defaultdict(float)
    operation_times = defaultdict(float)

    for event in events:
        name = event.get('name', '')
        dur = event.get('dur', 0)  # Microseconds
        cat = event.get('cat', '')
        ph = event.get('ph', '')

        # Only look at complete events
        if ph != 'X' or dur == 0:
            continue

        # Categorize
        category = categorize_operation(name, cat)
        category_times[category] += dur
        operation_times[name] += dur

    # Convert to milliseconds
    category_times_ms = {k: v/1000 for k, v in category_times.items()}
    operation_times_ms = {k: v/1000 for k, v in operation_times.items()}

    total_time = sum(category_times_ms.values())

    print("="*80)
    print("OPERATION CATEGORY BREAKDOWN")
    print("="*80)
    print(f"{'Category':<25} {'Time (ms)':>15} {'%':>10} {'Description':<30}")
    print("-"*80)

    category_descriptions = {
        'data_loading': 'DataLoader, batching, workers',
        'graph_processing': 'Graph operations, masking',
        'ddp_communication': 'DDP gradient sync',
        'model_forward': 'Model forward pass',
        'loss_computation': 'Loss calculation',
        'backward': 'Gradient computation',
        'optimizer': 'Parameter updates',
        'memory': 'CUDA memory allocation',
        'tensor_ops': 'Tensor operations (aten::)',
        'cuda_kernels': 'GPU kernel execution',
        'other': 'Miscellaneous operations'
    }

    for category in sorted(category_times_ms.keys(), key=lambda x: category_times_ms[x], reverse=True):
        time_ms = category_times_ms[category]
        pct = (time_ms / total_time * 100) if total_time > 0 else 0
        desc = category_descriptions.get(category, '')
        print(f"{category:<25} {time_ms:>15,.2f} {pct:>9.1f}% {desc:<30}")

    print()
    print(f"{'Total time':<25} {total_time:>15,.2f} ms")
    print()

    # Top operations per category
    print("="*80)
    print("TOP 10 OPERATIONS PER CATEGORY")
    print("="*80)

    # Group operations by category
    ops_by_category = defaultdict(list)
    for op_name, time_ms in operation_times_ms.items():
        cat = categorize_operation(op_name, '')
        ops_by_category[cat].append((op_name, time_ms))

    for category in sorted(category_times_ms.keys(), key=lambda x: category_times_ms[x], reverse=True)[:5]:
        if category == 'other':
            continue

        print()
        print(f"--- {category.upper().replace('_', ' ')} ---")
        print(f"{'Operation':<70} {'Time (ms)':>15}")
        print("-"*86)

        ops = sorted(ops_by_category[category], key=lambda x: x[1], reverse=True)[:10]
        for op_name, time_ms in ops:
            # Truncate long names
            display_name = op_name[:70] if len(op_name) <= 70 else op_name[:67] + "..."
            print(f"{display_name:<70} {time_ms:>15,.2f}")

    print()
    print("="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)

    # Calculate key metrics
    data_loading_pct = (category_times_ms.get('data_loading', 0) / total_time * 100) if total_time > 0 else 0
    graph_proc_pct = (category_times_ms.get('graph_processing', 0) / total_time * 100) if total_time > 0 else 0
    ddp_comm_pct = (category_times_ms.get('ddp_communication', 0) / total_time * 100) if total_time > 0 else 0
    model_fwd_pct = (category_times_ms.get('model_forward', 0) / total_time * 100) if total_time > 0 else 0
    backward_pct = (category_times_ms.get('backward', 0) / total_time * 100) if total_time > 0 else 0
    cuda_pct = (category_times_ms.get('cuda_kernels', 0) / total_time * 100) if total_time > 0 else 0

    print(f"Data Loading:          {data_loading_pct:>6.1f}%  ({category_times_ms.get('data_loading', 0):,.0f} ms)")
    print(f"Graph Processing:      {graph_proc_pct:>6.1f}%  ({category_times_ms.get('graph_processing', 0):,.0f} ms)")
    print(f"Model Forward:         {model_fwd_pct:>6.1f}%  ({category_times_ms.get('model_forward', 0):,.0f} ms)")
    print(f"Loss Computation:      {(category_times_ms.get('loss_computation', 0) / total_time * 100):>6.1f}%  ({category_times_ms.get('loss_computation', 0):,.0f} ms)")
    print(f"Backward Pass:         {backward_pct:>6.1f}%  ({category_times_ms.get('backward', 0):,.0f} ms)")
    print(f"DDP Communication:     {ddp_comm_pct:>6.1f}%  ({category_times_ms.get('ddp_communication', 0):,.0f} ms)")
    print(f"Optimizer:             {(category_times_ms.get('optimizer', 0) / total_time * 100):>6.1f}%  ({category_times_ms.get('optimizer', 0):,.0f} ms)")
    print(f"CUDA Kernels:          {cuda_pct:>6.1f}%  ({category_times_ms.get('cuda_kernels', 0):,.0f} ms)")
    print()

    # Identify bottleneck
    cpu_bound = data_loading_pct + graph_proc_pct + model_fwd_pct + backward_pct
    gpu_bound = cuda_pct

    print("BOTTLENECK IDENTIFICATION:")
    if data_loading_pct > 15:
        print(f"⚠️  DATA LOADING ({data_loading_pct:.1f}%)")
        print("    → Consider: increase num_workers, prefetch_factor, or persistent_workers")

    if graph_proc_pct > 10:
        print(f"⚠️  GRAPH PROCESSING ({graph_proc_pct:.1f}%)")
        print("    → Consider: optimize mask computation or cache graph structures")

    if ddp_comm_pct > 15:
        print(f"⚠️  DDP COMMUNICATION ({ddp_comm_pct:.1f}%)")
        print("    → Consider: gradient accumulation, larger batch size, or gradient compression")

    if cpu_bound > gpu_bound * 2:
        print(f"⚠️  CPU OVERHEAD ({cpu_bound:.1f}% CPU vs {gpu_bound:.1f}% GPU)")
        print("    → GPU is underutilized. Increase batch size or reduce CPU preprocessing.")

    if cuda_pct > 50:
        print(f"✅  GPU COMPUTE BOUND ({cuda_pct:.1f}%)")
        print("    → This is ideal - GPU is well utilized.")

    if data_loading_pct < 5 and graph_proc_pct < 5:
        print(f"✅  DATA LOADING OPTIMIZED ({data_loading_pct + graph_proc_pct:.1f}%)")
        print("    → Data loading is not a bottleneck. No need to save masks to disk.")

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
