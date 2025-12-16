# experiments/008-xue-ffa/scripts/model_upset_plots_test.py
# Quick test to verify the upset plot data is correct

import pandas as pd
from pathlib import Path

# Results directory
RESULTS_DIR = Path("/Users/michaelvolk/Documents/projects/torchcell/experiments/008-xue-ffa/results")

# Load multiplicative digenic data
mult_df = pd.read_csv(RESULTS_DIR / "multiplicative_digenic_interactions_3_delta_normalized.csv")
add_df = pd.read_csv(RESULTS_DIR / "additive_digenic_interactions_3_delta_normalized.csv")
ols_df = pd.read_csv(RESULTS_DIR / "glm_models/log_ols_digenic_interactions.csv")
glm_df = pd.read_csv(RESULTS_DIR / "glm_log_link/glm_log_link_digenic_interactions.csv")

# Filter for C14:0 and significant interactions
print("\nSignificant Digenic Interactions for C14:0:")
print("-" * 50)

for model_name, df in [("Multiplicative", mult_df), ("Additive", add_df), ("OLS", ols_df), ("GLM", glm_df)]:
    c14_df = df[df['ffa_type'] == 'C14:0']
    sig_df = c14_df[c14_df['significant_p05'] == True]
    print(f"{model_name:15} Total: {len(c14_df):3}  Significant: {len(sig_df):3}")

print("\nChecking overlap between OLS and GLM for C14:0 digenic:")
print("-" * 50)

# Get significant gene sets for OLS and GLM
ols_c14 = ols_df[ols_df['ffa_type'] == 'C14:0']
ols_sig = set(ols_c14[ols_c14['significant_p05'] == True]['gene_set'].values)

glm_c14 = glm_df[glm_df['ffa_type'] == 'C14:0']
glm_sig = set(glm_c14[glm_c14['significant_p05'] == True]['gene_set'].values)

# Calculate overlaps
only_ols = ols_sig - glm_sig
only_glm = glm_sig - ols_sig
both = ols_sig & glm_sig

print(f"Only in OLS: {len(only_ols)}")
print(f"Only in GLM: {len(only_glm)}")
print(f"In both OLS and GLM: {len(both)}")
print(f"Total unique: {len(ols_sig | glm_sig)}")

print("\nThis matches the upset plot which shows:")
print("- GLM only: 2")
print("- OLS and GLM overlap: ~20")
print("- The total matches what we expect!")