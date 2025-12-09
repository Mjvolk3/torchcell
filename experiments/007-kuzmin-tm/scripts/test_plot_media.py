#!/usr/bin/env python3
"""
Test script to debug media comparison plotting.
"""

print("Starting script...")

import os
print("Imported os")

import os.path as osp
print("Imported osp")

import pandas as pd
print("Imported pandas")

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
print("Set matplotlib backend")

import matplotlib.pyplot as plt
print("Imported pyplot")

import numpy as np
print("Imported numpy")

from scipy import stats
print("Imported scipy.stats")

from dotenv import load_dotenv
print("Imported load_dotenv")

from datetime import datetime
print("Imported datetime")

print("\nAll imports successful!")

# Set up paths
results_dir = "/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/results/cobra-fba-growth"
print(f"Results directory: {results_dir}")

# Check if matched file exists
matched_file = osp.join(results_dir, "matched_fba_experimental_fixed.parquet")
print(f"Checking for matched file: {matched_file}")
print(f"File exists: {osp.exists(matched_file)}")

if osp.exists(matched_file):
    print("\nLoading matched data...")
    df = pd.read_parquet(matched_file)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Unique phenotype types: {df['phenotype_type'].unique()}")

# Check for media-specific files
for media in ['minimal', 'YNB', 'YPD']:
    singles_file = osp.join(results_dir, f'singles_deletions_{media}.parquet')
    print(f"\n{media} singles file exists: {osp.exists(singles_file)}")
    if osp.exists(singles_file):
        singles_df = pd.read_parquet(singles_file)
        print(f"  Loaded {len(singles_df)} singles for {media}")

print("\nTest complete!")