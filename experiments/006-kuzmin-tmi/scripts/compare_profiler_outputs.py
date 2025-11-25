#!/usr/bin/env python3
"""
Compare PyTorch profiler outputs from dango.py and hetero_cell_bipartite_dango_gi.py
to identify performance bottlenecks, especially in subgraphing operations.

Usage:
    python compare_profiler_outputs.py --dango-profile <path> --hetero-profile <path>
"""

import argparse
import json
import os
import os.path as osp
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Add base directory to Python path
BASE_DIR = "/home/michaelvolk/Documents/projects/torchcell"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Load environment from base directory
load_dotenv(osp.join(BASE_DIR, ".env"))
DATA_ROOT = os.getenv("DATA_ROOT", "/scratch/projects/torchcell")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR", osp.join(DATA_ROOT, "assets/images"))

# Import timestamp for consistent naming
from torchcell.timestamp import timestamp


def load_chrome_trace(filepath: str) -> Dict:
    """Load Chrome trace JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_key_operations(trace_data: Dict) -> pd.DataFrame:
    """Extract key operations from Chrome trace format."""
    events = []

    if 'traceEvents' in trace_data:
        for event in trace_data['traceEvents']:
            if event.get('ph') == 'X':  # Complete events
                events.append({
                    'name': event.get('name', ''),
                    'duration_us': event.get('dur', 0),
                    'timestamp': event.get('ts', 0),
                    'category': event.get('cat', ''),
                    'args': event.get('args', {})
                })

    return pd.DataFrame(events)


def analyze_subgraph_operations(df: pd.DataFrame) -> pd.DataFrame:
    """Identify and analyze subgraph-related operations."""
    # Look for operations that might be related to subgraphing
    subgraph_keywords = [
        'subgraph', 'select', 'index', 'gather', 'scatter',
        'edge_index', 'node_', 'batch', 'to_hetero', 'hetero',
        'message', 'aggregate', 'update', 'conv'
    ]

    subgraph_ops = df[df['name'].str.lower().str.contains('|'.join(subgraph_keywords), na=False)]

    if not subgraph_ops.empty:
        # Aggregate by operation name
        agg_stats = subgraph_ops.groupby('name').agg({
            'duration_us': ['sum', 'mean', 'std', 'count']
        }).round(2)
        agg_stats.columns = ['total_time_us', 'mean_time_us', 'std_time_us', 'count']
        agg_stats['percent_of_total'] = (agg_stats['total_time_us'] / df['duration_us'].sum() * 100).round(2)
        return agg_stats.sort_values('total_time_us', ascending=False)

    return pd.DataFrame()


def analyze_gpu_utilization(df: pd.DataFrame) -> Dict:
    """Analyze GPU utilization from profiler data."""
    # Separate CPU and GPU operations
    cpu_ops = df[df['category'].str.contains('cpu', case=False, na=False)]
    gpu_ops = df[df['category'].str.contains('cuda|gpu', case=False, na=False)]

    total_time = df['duration_us'].sum()
    cpu_time = cpu_ops['duration_us'].sum()
    gpu_time = gpu_ops['duration_us'].sum()

    return {
        'total_time_us': total_time,
        'cpu_time_us': cpu_time,
        'gpu_time_us': gpu_time,
        'cpu_percentage': (cpu_time / total_time * 100) if total_time > 0 else 0,
        'gpu_percentage': (gpu_time / total_time * 100) if total_time > 0 else 0,
        'gpu_utilization': (gpu_time / total_time * 100) if total_time > 0 else 0
    }


def get_top_operations(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Get top N most time-consuming operations."""
    op_stats = df.groupby('name').agg({
        'duration_us': ['sum', 'mean', 'count']
    }).round(2)
    op_stats.columns = ['total_time_us', 'mean_time_us', 'count']
    op_stats['percent_of_total'] = (op_stats['total_time_us'] / df['duration_us'].sum() * 100).round(2)

    return op_stats.sort_values('total_time_us', ascending=False).head(n)


def compare_profiles(dango_path: str, hetero_path: str) -> Dict:
    """Compare two profile outputs and generate analysis."""
    results = {}

    # Load profiles
    print("Loading Dango profile...")
    dango_trace = load_chrome_trace(dango_path)
    dango_df = extract_key_operations(dango_trace)

    print("Loading HeteroCell profile...")
    hetero_trace = load_chrome_trace(hetero_path)
    hetero_df = extract_key_operations(hetero_trace)

    # Analyze GPU utilization
    results['dango_gpu'] = analyze_gpu_utilization(dango_df)
    results['hetero_gpu'] = analyze_gpu_utilization(hetero_df)

    # Analyze subgraph operations
    results['dango_subgraph'] = analyze_subgraph_operations(dango_df)
    results['hetero_subgraph'] = analyze_subgraph_operations(hetero_df)

    # Get top operations
    results['dango_top_ops'] = get_top_operations(dango_df)
    results['hetero_top_ops'] = get_top_operations(hetero_df)

    # Calculate overall statistics
    results['summary'] = {
        'dango_total_ops': len(dango_df),
        'hetero_total_ops': len(hetero_df),
        'dango_unique_ops': dango_df['name'].nunique(),
        'hetero_unique_ops': hetero_df['name'].nunique(),
        'dango_mean_op_time': dango_df['duration_us'].mean(),
        'hetero_mean_op_time': hetero_df['duration_us'].mean(),
        'speedup_factor': dango_df['duration_us'].sum() / hetero_df['duration_us'].sum() if hetero_df['duration_us'].sum() > 0 else 0
    }

    return results


def generate_comparison_plots(results: Dict, output_dir: str):
    """Generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)

    # GPU Utilization Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # GPU utilization bar chart
    models = ['Dango', 'HeteroCell']
    gpu_utils = [results['dango_gpu']['gpu_utilization'], results['hetero_gpu']['gpu_utilization']]

    axes[0].bar(models, gpu_utils, color=['#2E86AB', '#A23B72'])
    axes[0].set_ylabel('GPU Utilization (%)')
    axes[0].set_title('GPU Utilization Comparison')
    axes[0].set_ylim(0, 100)

    # Add value labels on bars
    for i, v in enumerate(gpu_utils):
        axes[0].text(i, v + 1, f'{v:.1f}%', ha='center')

    # Time distribution pie chart
    dango_times = [results['dango_gpu']['cpu_time_us'], results['dango_gpu']['gpu_time_us']]
    hetero_times = [results['hetero_gpu']['cpu_time_us'], results['hetero_gpu']['gpu_time_us']]

    axes[1].pie([sum(dango_times), sum(hetero_times)],
                labels=['Dango', 'HeteroCell'],
                autopct='%1.1f%%',
                colors=['#2E86AB', '#A23B72'])
    axes[1].set_title('Total Execution Time Distribution')

    plt.suptitle('Performance Comparison: Dango vs HeteroCell', fontsize=14, fontweight='bold')
    plt.tight_layout()

    filename = osp.join(output_dir, f"gpu_comparison_{timestamp()}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved GPU comparison plot to {filename}")

    # Top operations comparison
    if not results['dango_top_ops'].empty and not results['hetero_top_ops'].empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Dango top operations
        top_5_dango = results['dango_top_ops'].head(5)
        axes[0].barh(range(len(top_5_dango)), top_5_dango['percent_of_total'])
        axes[0].set_yticks(range(len(top_5_dango)))
        axes[0].set_yticklabels([name[:30] for name in top_5_dango.index])
        axes[0].set_xlabel('% of Total Time')
        axes[0].set_title('Dango: Top 5 Time-Consuming Operations')
        axes[0].invert_yaxis()

        # HeteroCell top operations
        top_5_hetero = results['hetero_top_ops'].head(5)
        axes[1].barh(range(len(top_5_hetero)), top_5_hetero['percent_of_total'])
        axes[1].set_yticks(range(len(top_5_hetero)))
        axes[1].set_yticklabels([name[:30] for name in top_5_hetero.index])
        axes[1].set_xlabel('% of Total Time')
        axes[1].set_title('HeteroCell: Top 5 Time-Consuming Operations')
        axes[1].invert_yaxis()

        plt.suptitle('Top Time-Consuming Operations', fontsize=14, fontweight='bold')
        plt.tight_layout()

        filename = osp.join(output_dir, f"top_operations_{timestamp()}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved top operations plot to {filename}")

    plt.close('all')


def print_analysis_report(results: Dict):
    """Print detailed analysis report."""
    print("\n" + "="*80)
    print("PROFILER COMPARISON REPORT")
    print("="*80)

    # Summary statistics
    print("\n## SUMMARY STATISTICS")
    print("-"*40)
    summary = results['summary']
    print(f"Dango - Total Operations: {summary['dango_total_ops']:,}")
    print(f"Dango - Unique Operations: {summary['dango_unique_ops']:,}")
    print(f"Dango - Mean Operation Time: {summary['dango_mean_op_time']:.2f} μs")
    print()
    print(f"HeteroCell - Total Operations: {summary['hetero_total_ops']:,}")
    print(f"HeteroCell - Unique Operations: {summary['hetero_unique_ops']:,}")
    print(f"HeteroCell - Mean Operation Time: {summary['hetero_mean_op_time']:.2f} μs")
    print()
    if summary['speedup_factor'] > 1:
        print(f"🚀 Dango is {summary['speedup_factor']:.2f}x faster overall")
    else:
        print(f"⚠️ HeteroCell is {1/summary['speedup_factor']:.2f}x faster overall")

    # GPU Utilization
    print("\n## GPU UTILIZATION")
    print("-"*40)
    dango_gpu = results['dango_gpu']
    hetero_gpu = results['hetero_gpu']

    print(f"Dango:")
    print(f"  - GPU Utilization: {dango_gpu['gpu_utilization']:.1f}%")
    print(f"  - CPU Time: {dango_gpu['cpu_percentage']:.1f}%")
    print(f"  - GPU Time: {dango_gpu['gpu_percentage']:.1f}%")

    print(f"\nHeteroCell:")
    print(f"  - GPU Utilization: {hetero_gpu['gpu_utilization']:.1f}%")
    print(f"  - CPU Time: {hetero_gpu['cpu_percentage']:.1f}%")
    print(f"  - GPU Time: {hetero_gpu['gpu_percentage']:.1f}%")

    # Subgraph operations analysis
    print("\n## SUBGRAPH OPERATIONS ANALYSIS")
    print("-"*40)

    if not results['dango_subgraph'].empty:
        print("\nDango - Top Subgraph-Related Operations:")
        print(results['dango_subgraph'].head(5).to_string())
    else:
        print("\nDango - No significant subgraph operations detected")

    if not results['hetero_subgraph'].empty:
        print("\nHeteroCell - Top Subgraph-Related Operations:")
        print(results['hetero_subgraph'].head(5).to_string())

        # Identify potential bottlenecks
        hetero_subgraph_time = results['hetero_subgraph']['total_time_us'].sum()
        dango_subgraph_time = results['dango_subgraph']['total_time_us'].sum() if not results['dango_subgraph'].empty else 0

        if hetero_subgraph_time > dango_subgraph_time * 2:
            print(f"\n⚠️ BOTTLENECK DETECTED: HeteroCell spends {hetero_subgraph_time/1e6:.2f}s on subgraph operations")
            print(f"   vs Dango's {dango_subgraph_time/1e6:.2f}s (difference: {(hetero_subgraph_time-dango_subgraph_time)/1e6:.2f}s)")
    else:
        print("\nHeteroCell - No significant subgraph operations detected")

    # Top operations
    print("\n## TOP TIME-CONSUMING OPERATIONS")
    print("-"*40)

    print("\nDango - Top 5 Operations:")
    if not results['dango_top_ops'].empty:
        print(results['dango_top_ops'].head(5).to_string())

    print("\nHeteroCell - Top 5 Operations:")
    if not results['hetero_top_ops'].empty:
        print(results['hetero_top_ops'].head(5).to_string())

    print("\n" + "="*80)


def find_latest_profiles(base_dir: str = None) -> Tuple[Optional[str], Optional[str]]:
    """Find the latest profile files for both models."""
    if base_dir is None:
        base_dir = DATA_ROOT

    profile_dir = osp.join(base_dir, "data/torchcell/experiments/006-kuzmin-tmi/profiler_output")

    dango_profiles = []
    hetero_profiles = []

    if osp.exists(profile_dir):
        for root, dirs, files in os.walk(profile_dir):
            for file in files:
                if file.endswith('.json'):
                    full_path = osp.join(root, file)
                    if 'dango_' in root and 'hetero' not in root:
                        dango_profiles.append(full_path)
                    elif 'hetero_dango_gi_' in root:
                        hetero_profiles.append(full_path)

    # Get most recent files
    dango_latest = max(dango_profiles, key=os.path.getmtime) if dango_profiles else None
    hetero_latest = max(hetero_profiles, key=os.path.getmtime) if hetero_profiles else None

    return dango_latest, hetero_latest


def main():
    parser = argparse.ArgumentParser(description='Compare PyTorch profiler outputs')
    parser.add_argument('--dango-profile', type=str, help='Path to Dango profile JSON')
    parser.add_argument('--hetero-profile', type=str, help='Path to HeteroCell profile JSON')
    parser.add_argument('--auto-find', action='store_true',
                       help='Automatically find latest profile files')
    parser.add_argument('--output-dir', type=str, default=ASSET_IMAGES_DIR,
                       help='Directory for output plots')

    args = parser.parse_args()

    # Find profile files
    if args.auto_find:
        print("Automatically finding latest profile files...")
        dango_path, hetero_path = find_latest_profiles(DATA_ROOT)

        if not dango_path or not hetero_path:
            print("Error: Could not find profile files. Please run both models with profiling enabled first.")
            return

        print(f"Found Dango profile: {dango_path}")
        print(f"Found HeteroCell profile: {hetero_path}")
    else:
        if not args.dango_profile or not args.hetero_profile:
            print("Error: Please provide both --dango-profile and --hetero-profile paths, or use --auto-find")
            return

        dango_path = args.dango_profile
        hetero_path = args.hetero_profile

    # Check files exist
    if not osp.exists(dango_path):
        print(f"Error: Dango profile not found at {dango_path}")
        return

    if not osp.exists(hetero_path):
        print(f"Error: HeteroCell profile not found at {hetero_path}")
        return

    # Run comparison
    results = compare_profiles(dango_path, hetero_path)

    # Generate report
    print_analysis_report(results)

    # Generate plots
    generate_comparison_plots(results, args.output_dir)

    # Save detailed results to CSV for further analysis
    output_path = osp.join(args.output_dir, f"profile_comparison_{timestamp()}.csv")

    # Create summary DataFrame
    summary_df = pd.DataFrame([{
        'metric': 'Total Operations',
        'dango': results['summary']['dango_total_ops'],
        'hetero': results['summary']['hetero_total_ops']
    }, {
        'metric': 'GPU Utilization (%)',
        'dango': results['dango_gpu']['gpu_utilization'],
        'hetero': results['hetero_gpu']['gpu_utilization']
    }, {
        'metric': 'Mean Op Time (μs)',
        'dango': results['summary']['dango_mean_op_time'],
        'hetero': results['summary']['hetero_mean_op_time']
    }])

    summary_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()