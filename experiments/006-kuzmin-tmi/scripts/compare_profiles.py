#!/usr/bin/env python
"""
Compare two PyTorch profiler traces to identify performance differences.

Usage:
    python compare_profiles.py <trace1.json> <trace2.json>

Example:
    python compare_profiles.py dango_profile.json lazy_hetero_profile.json
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
    if any(x in name_lower for x in ['lazysubgraph', 'subgraph', 'process_gene', 'mask', 'incidence']):
        return 'graph_processing'

    # DDP communication
    if any(x in name_lower for x in ['nccl', 'allreduce', 'allgather', 'broadcast', 'dist']):
        return 'ddp_communication'

    # Model forward pass
    if any(x in name_lower for x in ['forward', 'conv', 'gin', 'gat', 'attention', 'embedding', 'sage', 'dango']):
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
    """Analyze a single trace file."""
    with open(trace_file, 'r') as f:
        data = json.load(f)

    events = data.get('traceEvents', [])

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

    return category_times_ms, operation_times_ms, total_time


def compare_traces(trace1_file, trace2_file):
    """Compare two trace files and show differences."""
    print(f"Loading traces...")
    print(f"  Trace 1: {trace1_file.name}")
    print(f"  Trace 2: {trace2_file.name}")
    print()

    cat1, ops1, total1 = analyze_trace(trace1_file)
    cat2, ops2, total2 = analyze_trace(trace2_file)

    # Get all categories
    all_categories = sorted(set(list(cat1.keys()) + list(cat2.keys())))

    print("="*100)
    print(f"{'PERFORMANCE COMPARISON':<50}")
    print("="*100)
    print(f"{'Total Time':<30} {trace1_file.stem:<30} {trace2_file.stem:<30} {'Speedup':<10}")
    print("-"*100)
    print(f"{'Total (ms)':<30} {total1:>25,.2f} ms  {total2:>25,.2f} ms  {total1/total2 if total2 > 0 else 0:>7.2f}x")
    print(f"{'Iteration Speed (approx)':<30} {1000/total1 if total1 > 0 else 0:>22.2f} it/s  {1000/total2 if total2 > 0 else 0:>22.2f} it/s")
    print()

    print("="*100)
    print(f"{'CATEGORY BREAKDOWN':<50}")
    print("="*100)
    print(f"{'Category':<30} {trace1_file.stem + ' (%)':<25} {trace2_file.stem + ' (%)':<25} {'Difference':<15} {'Ratio':<10}")
    print("-"*100)

    for category in all_categories:
        time1 = cat1.get(category, 0)
        time2 = cat2.get(category, 0)
        pct1 = (time1 / total1 * 100) if total1 > 0 else 0
        pct2 = (time2 / total2 * 100) if total2 > 0 else 0
        diff = pct1 - pct2
        ratio = (time1 / time2) if time2 > 0 else float('inf') if time1 > 0 else 1.0

        # Highlight significant differences
        marker = ""
        if abs(diff) > 5:
            marker = "⚠️ " if diff > 0 else "✅ "

        print(f"{marker}{category:<28} {time1:>10,.0f} ({pct1:>5.1f}%)  "
              f"{time2:>10,.0f} ({pct2:>5.1f}%)  "
              f"{diff:>+6.1f}%  "
              f"{ratio:>8.2f}x")

    print()
    print("="*100)
    print("BOTTLENECK ANALYSIS")
    print("="*100)

    # Identify key differences
    significant_diffs = []
    for category in all_categories:
        time1 = cat1.get(category, 0)
        time2 = cat2.get(category, 0)
        pct1 = (time1 / total1 * 100) if total1 > 0 else 0
        pct2 = (time2 / total2 * 100) if total2 > 0 else 0
        diff = pct1 - pct2

        if abs(diff) > 5:  # More than 5% difference
            significant_diffs.append((category, diff, pct1, pct2))

    significant_diffs.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\nKey differences (>{5}% change):")
    for category, diff, pct1, pct2 in significant_diffs:
        if diff > 0:
            print(f"  ⚠️  {category}: {trace1_file.stem} uses {diff:.1f}% more time ({pct1:.1f}% vs {pct2:.1f}%)")
        else:
            print(f"  ✅  {category}: {trace1_file.stem} uses {-diff:.1f}% less time ({pct1:.1f}% vs {pct2:.1f}%)")

    # Overall assessment
    print(f"\n{'OVERALL ASSESSMENT':=^100}")
    speedup = total2 / total1 if total1 > 0 else 0
    if speedup > 1.5:
        print(f"✅ {trace1_file.stem} is {speedup:.2f}x FASTER than {trace2_file.stem}")
    elif speedup < 0.67:
        print(f"⚠️  {trace1_file.stem} is {1/speedup:.2f}x SLOWER than {trace2_file.stem}")
    else:
        print(f"≈  Performance is comparable (speedup: {speedup:.2f}x)")

    # Specific recommendations
    print(f"\n{'RECOMMENDATIONS':=^100}")

    # Check data loading
    data_loading_1 = cat1.get('data_loading', 0) / total1 * 100 if total1 > 0 else 0
    data_loading_2 = cat2.get('data_loading', 0) / total2 * 100 if total2 > 0 else 0

    if data_loading_1 > data_loading_2 + 5:
        print(f"• {trace1_file.stem} has higher data loading overhead ({data_loading_1:.1f}% vs {data_loading_2:.1f}%)")
        print(f"  → Consider optimizations from {trace2_file.stem}'s data pipeline")

    # Check graph processing
    graph_proc_1 = cat1.get('graph_processing', 0) / total1 * 100 if total1 > 0 else 0
    graph_proc_2 = cat2.get('graph_processing', 0) / total2 * 100 if total2 > 0 else 0

    if graph_proc_1 > graph_proc_2 + 5:
        print(f"• {trace1_file.stem} spends more time in graph processing ({graph_proc_1:.1f}% vs {graph_proc_2:.1f}%)")
        print(f"  → Examine {trace2_file.stem}'s graph handling approach")

    # Check model complexity
    model_fwd_1 = cat1.get('model_forward', 0) / total1 * 100 if total1 > 0 else 0
    model_fwd_2 = cat2.get('model_forward', 0) / total2 * 100 if total2 > 0 else 0

    if model_fwd_1 > model_fwd_2 + 5:
        print(f"• {trace1_file.stem} has heavier model computation ({model_fwd_1:.1f}% vs {model_fwd_2:.1f}%)")
        print(f"  → Model architecture may be more complex")

    print("="*100)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    trace1_file = Path(sys.argv[1])
    trace2_file = Path(sys.argv[2])

    if not trace1_file.exists():
        print(f"Error: File not found: {trace1_file}")
        sys.exit(1)

    if not trace2_file.exists():
        print(f"Error: File not found: {trace2_file}")
        sys.exit(1)

    compare_traces(trace1_file, trace2_file)