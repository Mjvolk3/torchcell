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


def print_timing_summary(title: str = "Method Timing Profile"):
    """
    Print formatted timing summary with standard deviations.

    Args:
        title: Custom title for the report
    """
    if not _TIMINGS:
        if _PROFILE_ENABLED:
            print(f"\n[TIMING] Profiling enabled but no timing data collected")
        return

    import numpy as np

    print("\n" + "="*80)
    print(title)
    print("="*80)
    print(f"{'Method':<50} {'Calls':>8} {'Mean (ms)':>12} {'Std (ms)':>11}")
    print("-"*80)

    # Sort by total time descending
    for name, times in sorted(_TIMINGS.items(), key=lambda x: sum(x[1]), reverse=True):
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
