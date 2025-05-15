# experiments/005-kuzmin2018-tmi/scripts/go_term_filter_analysis
# [[experiments.005-kuzmin2018-tmi.scripts.go_term_filter_analysis]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/005-kuzmin2018-tmi/scripts/go_term_filter_analysis
# Test file: experiments/005-kuzmin2018-tmi/scripts/test_go_term_filter_analysis.py


import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import seaborn as sns
from datetime import datetime
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.graph.graph import (
    SCerevisiaeGraph,
    filter_by_contained_genes,
    filter_by_date,
)
import torchcell.timestamp as tc_timestamp

# Load environment variables
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR", "notes/assets/images")

def analyze_containment_filter(graph, n_values=None):
    """Analyze how the number of GO terms changes with different n values for contained genes filter."""
    if n_values is None:
        n_values = [1, 2, 3, 4, 5, 6]
    
    G_go = graph.G_go
    original_count = len(G_go.nodes())
    
    results = []
    for n in n_values:
        filtered_G = filter_by_contained_genes(G_go, n=n, gene_set=graph.genome.gene_set)
        num_nodes = len(filtered_G.nodes())
        
        results.append({
            'n': n,
            'num_nodes': num_nodes
        })
    
    return pd.DataFrame(results)

def analyze_go_terms_by_time(graph):
    """Analyze how the number of GO terms changes over time using monthly date cutoffs."""
    G_go = graph.G_go
    original_count = len(G_go.nodes())
    
    # Generate dates for analysis (monthly from 2000 to present)
    years = range(2000, 2026)
    months = range(1, 13)
    
    results = []
    # Add a point for no filtering
    results.append({
        'date': 'No filter',
        'year_month': 'No filter',
        'num_nodes': original_count
    })
    
    # Test each monthly cutoff
    for year in years:
        for month in months:
            cutoff_date = f"{year}-{month:02d}-01"
            
            # Skip future dates
            if datetime.strptime(cutoff_date, "%Y-%m-%d") > datetime.now():
                continue
                
            # Apply date filter
            filtered_G = filter_by_date(G_go, cutoff_date=cutoff_date)
            num_nodes = len(filtered_G.nodes())
            
            results.append({
                'date': cutoff_date,
                'year_month': f"{year}-{month:02d}",
                'num_nodes': num_nodes
            })
    
    return pd.DataFrame(results)

def plot_containment_filter(df):
    """Plot GO term counts vs minimum contained genes threshold."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='n', y='num_nodes', marker='o', linewidth=3)
    plt.xlabel('Minimum Number of Contained Genes (n)', fontsize=14)
    plt.ylabel('Number of GO Terms Retained', fontsize=14)
    plt.suptitle('GO Term Count vs. Minimum Contained Genes Threshold', fontsize=14, y=0.98)
    plt.title('GO terms removed if they contain fewer than n genes', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(df['n'], fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    timestamp = tc_timestamp.timestamp()
    filepath = osp.join(ASSET_IMAGES_DIR, f"go_term_containment_filter_{timestamp}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved containment filter plot to {filepath}")
    
    return filepath

def plot_go_terms_by_time(df):
    """Plot how GO term counts change over time."""
    # Filter out "No filter" entry and convert to datetime for plotting
    df_time = df[df['date'] != 'No filter'].copy()
    df_time['date_obj'] = pd.to_datetime(df_time['date'])
    df_time = df_time.sort_values('date_obj')
    
    # Sample every 12 months for readable tick labels
    df_time_sampled = df_time.iloc[::12].copy()
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_time, x='date_obj', y='num_nodes', marker='', linewidth=2.5)
    plt.xlabel('Date Cutoff', fontsize=14)
    plt.ylabel('Number of GO Terms', fontsize=14)
    plt.suptitle('GO Terms by Date', fontsize=14, y=0.98)
    plt.title('GO terms removed if they were annotated after the cutoff date', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis tick labels
    plt.xticks(df_time_sampled['date_obj'], df_time_sampled['year_month'], rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    timestamp = tc_timestamp.timestamp()
    filepath = osp.join(ASSET_IMAGES_DIR, f"go_terms_by_time_{timestamp}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved GO terms by time plot to {filepath}")
    
    return filepath

def main():
    """Main function to run the analysis."""
    # Load genome and graph
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    
    print(f"Original GO graph has {len(graph.G_go.nodes())} nodes")
    
    # 1. Analyze the effect of the containment filter (n=1 to n=6)
    n_values = [1, 2, 3, 4, 5, 6]
    containment_df = analyze_containment_filter(graph, n_values)
    
    # 2. Analyze GO terms by time (monthly cutoffs)
    go_time_df = analyze_go_terms_by_time(graph)
    
    # Generate plots
    containment_plot_path = plot_containment_filter(containment_df)
    time_plot_path = plot_go_terms_by_time(go_time_df)
    
    # Print summary information
    print("\nContainment Filter Analysis (n=1 to n=6):")
    print(containment_df)
    
    print("\nGO Terms by Time Analysis (sample):")
    print(go_time_df.iloc[::12])  # Show yearly samples
    
    print(f"\nPlots saved to:\n{containment_plot_path}\n{time_plot_path}")

if __name__ == "__main__":
    main()