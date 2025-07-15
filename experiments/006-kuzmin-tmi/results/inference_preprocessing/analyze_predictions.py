import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from datetime import datetime

# Load custom matplotlib style
style_path = "/Users/michaelvolk/Documents/projects/torchcell/torchcell/torchcell.mplstyle"
matplotlib.style.use(style_path)

# Read the CSV file
csv_path = "/Users/michaelvolk/Documents/projects/torchcell/data/torchcell/experiments/006-kuzmin-tmi/inference_0/inference_predictions_2025-07-07-12-48-31.csv"
df = pd.read_csv(csv_path)

# Sort by prediction value from highest (positive) to lowest (negative)
df_sorted = df.sort_values('prediction', ascending=False)

# Save top 200 predictions
output_dir = "/Users/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/results/inference_preprocessing"
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
top_200_path = os.path.join(output_dir, f"top_200_predictions_{timestamp}.csv")
df_sorted.head(200).to_csv(top_200_path, index=False)

# Print top 10 predictions
print("Top 10 Predictions (Highest to Lowest):")
print("=" * 80)
for idx, row in df_sorted.head(10).iterrows():
    print(f"Rank {idx+1}: {row['gene1']}, {row['gene2']}, {row['gene3']} → {row['prediction']:.6f}")

# Plot histogram
plt.figure(figsize=(12, 7.4))
plt.hist(df['prediction'], bins=1000, edgecolor='black')
plt.xlabel('Prediction Value')
plt.ylabel('Frequency')
plt.title('Distribution of Inference Predictions')
plt.grid(True, alpha=0.3)

# Save histogram
histogram_path = os.path.join(output_dir, f"predictions_histogram_{timestamp}.png")
plt.savefig(histogram_path, dpi=300)
plt.close()

print(f"\nSaved top 200 predictions to: {top_200_path}")
print(f"Saved histogram to: {histogram_path}")