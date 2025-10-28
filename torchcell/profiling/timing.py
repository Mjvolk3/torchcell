"""
Generic method timing decorator for profiling any function/method.
Toggle with TORCHCELL_DEBUG_TIMING=1 environment variable.
Set via shell: TORCHCELL_DEBUG_TIMING=1 python script.py
Or in SLURM: --env TORCHCELL_DEBUG_TIMING=1
"""

import os
import time
import functools
from typing import Dict, List
from collections import defaultdict

# Global state
_TIMINGS: Dict[str, List[float]] = defaultdict(list)
_PROFILE_ENABLED = os.getenv('TORCHCELL_DEBUG_TIMING', '0') == '1'


def time_method(func):
    """
    Decorator that times any function/method execution when profiling is enabled.
    Works with functions, instance methods, class methods, and static methods.
    Zero overhead when disabled (single boolean check).

    Usage:
        @time_method
        def my_function():
            pass

        class MyClass:
            @time_method
            def my_method(self):
                pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not _PROFILE_ENABLED:
            return func(*args, **kwargs)

        # Build qualified name: Class.method or just function_name
        qualname = func.__qualname__  # e.g., "SubgraphRepresentation.process"

        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        _TIMINGS[qualname].append(elapsed)

        return result
    return wrapper


def print_timing_summary(title: str = "Method Timing Profile", filter_class: str = None):
    """
    Print formatted timing summary with standard deviations.

    Args:
        title: Custom title for the report
        filter_class: Optional class name to filter methods (e.g., "SubgraphRepresentation")
    """
    if not _TIMINGS:
        if _PROFILE_ENABLED:
            print(f"\n[TIMING] Profiling enabled but no timing data collected")
        return

    import numpy as np

    # Filter timings if class filter provided
    if filter_class:
        # Exact match: class name must be followed by a dot (e.g., "SubgraphRepresentation.")
        # This prevents "SubgraphRepresentation" from matching "LazySubgraphRepresentation"
        filtered_timings = {k: v for k, v in _TIMINGS.items() if k.startswith(filter_class + ".")}
    else:
        filtered_timings = _TIMINGS

    if not filtered_timings:
        print(f"\n[TIMING] No timing data for class '{filter_class}'")
        return

    print("\n" + "="*80)
    print(title)
    print("="*80)
    print(f"{'Method':<50} {'Calls':>8} {'Mean (ms)':>12} {'Std (ms)':>11}")
    print("-"*80)

    # Sort by total time descending
    for name, times in sorted(filtered_timings.items(), key=lambda x: sum(x[1]), reverse=True):
        times_ms = np.array(times) * 1000
        mean_ms = times_ms.mean()
        std_ms = times_ms.std()
        print(f"{name:<50} {len(times):>8} {mean_ms:>12.4f} {std_ms:>11.4f}")

    print("="*80 + "\n")


def reset_timings():
    """Clear all timing data."""
    _TIMINGS.clear()


def get_timings() -> Dict[str, List[float]]:
    """Get raw timing data as a dictionary."""
    return dict(_TIMINGS)


def get_timing_summary() -> Dict[str, Dict[str, float]]:
    """
    Get timing summary statistics as a dictionary.

    Returns:
        Dict mapping method names to stats (total_ms, mean_ms, count, etc.)
    """
    if not _PROFILE_ENABLED:
        return {}

    summary = {}
    for name, times in _TIMINGS.items():
        total_ms = sum(times) * 1000
        count = len(times)
        summary[name] = {
            'total_ms': total_ms,
            'mean_ms': total_ms / count,
            'count': count,
            'min_ms': min(times) * 1000,
            'max_ms': max(times) * 1000,
        }
    return summary


def print_comparison_table(baseline_class: str, optimized_class: str, title: str = "Graph Processor Comparison"):
    """
    Print side-by-side comparison of two processor classes.

    Args:
        baseline_class: Name of baseline class (e.g., "SubgraphRepresentation")
        optimized_class: Name of optimized class (e.g., "LazySubgraphRepresentation")
        title: Custom title for the report
    """
    if not _TIMINGS:
        if _PROFILE_ENABLED:
            print(f"\n[TIMING] Profiling enabled but no timing data collected")
        return

    import numpy as np

    # Group timing data by method name
    baseline_methods = {}
    optimized_methods = {}

    for qualname, times in _TIMINGS.items():
        if baseline_class in qualname:
            method_name = qualname.split('.')[-1]  # Extract method name
            times_ms = np.array(times) * 1000
            baseline_methods[method_name] = {
                'mean': times_ms.mean(),
                'std': times_ms.std(),
                'count': len(times)
            }
        elif optimized_class in qualname:
            method_name = qualname.split('.')[-1]
            times_ms = np.array(times) * 1000
            optimized_methods[method_name] = {
                'mean': times_ms.mean(),
                'std': times_ms.std(),
                'count': len(times)
            }

    # Get all method names
    all_methods = sorted(set(baseline_methods.keys()) | set(optimized_methods.keys()))

    print("\n" + "="*100)
    print(title)
    print("="*100)
    print(f"{'Method':<35} {baseline_class[:12]:>12} {optimized_class[:12]:>12} {'Change':>12} {'Speedup':>10}")
    print("-"*100)

    baseline_total = 0
    optimized_total = 0

    for method in all_methods:
        baseline_time = baseline_methods.get(method, {}).get('mean', 0)
        optimized_time = optimized_methods.get(method, {}).get('mean', 0)

        baseline_total += baseline_time
        optimized_total += optimized_time

        if baseline_time > 0 and optimized_time > 0:
            change = optimized_time - baseline_time
            speedup = baseline_time / optimized_time if optimized_time > 0 else 0

            # Add visual indicator
            if speedup >= 2.0:
                indicator = "⭐"
            elif speedup >= 1.2:
                indicator = "✓"
            elif speedup < 0.9:
                indicator = "⚠"
            else:
                indicator = ""

            print(f"{method:<35} {baseline_time:>11.2f}ms {optimized_time:>11.2f}ms {change:>11.2f}ms {speedup:>9.2f}x {indicator}")
        elif baseline_time > 0:
            print(f"{method:<35} {baseline_time:>11.2f}ms {'N/A':>12} {'N/A':>12} {'N/A':>10}")
        elif optimized_time > 0:
            print(f"{method:<35} {'N/A':>12} {optimized_time:>11.2f}ms {'N/A':>12} {'N/A':>10}")

    print("-"*100)
    change_total = optimized_total - baseline_total
    speedup_total = baseline_total / optimized_total if optimized_total > 0 else 0
    print(f"{'TOTAL':<35} {baseline_total:>11.2f}ms {optimized_total:>11.2f}ms {change_total:>11.2f}ms {speedup_total:>9.2f}x")
    print("="*100 + "\n")

    print("Legend: ⭐ = 2x+ speedup, ✓ = 1.2x+ speedup, ⚠ = slowdown")
