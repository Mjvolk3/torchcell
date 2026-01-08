#!/usr/bin/env python
"""Quick script to view inference parquet data."""

import os
import os.path as osp
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")

# Path to the parquet file
parquet_path = osp.join(
    DATA_ROOT,
    "data/torchcell/experiments/006-kuzmin-tmi/inference_1/inferred",
    "models-checkpoints-gilahyper-647_c3851d6b804e8ee6da66a874d51dbcd70f0805ce0a19f6e89486add3d1f06484-scl42ay6-best-pearson-epoch=20-val-gene_interaction-Pearson=0.4149.parquet"
)

print(f"Reading: {parquet_path}")
print(f"File exists: {osp.exists(parquet_path)}")

if osp.exists(parquet_path):
    # Read parquet
    df = pd.read_parquet(parquet_path)

    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 10 rows:")
    print(df.head(10))

    print(f"\nData types:")
    print(df.dtypes)

    print(f"\nPrediction statistics:")
    print(df['prediction'].describe())
else:
    print("File not found!")
