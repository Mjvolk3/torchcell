#!/bin/bash
# experiments/014-genes-enriched-at-extreme-tmi/scripts/014-genes-enriched-at-extreme-tmi
# [[experiments.014-genes-enriched-at-extreme-tmi.scripts.014-genes-enriched-at-extreme-tmi]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/014-genes-enriched-at-extreme-tmi/scripts/014-genes-enriched-at-extreme-tmi


# Step 1: Data processing (slow - only run once or when parameters change)
echo "Step 1: Data processing..."
python experiments/014-genes-enriched-at-extreme-tmi/scripts/analyze_extreme_interactions.py

# Step 2: Visualization (fast - can run many times)
echo ""
echo "Step 2: Creating visualizations..."
python experiments/014-genes-enriched-at-extreme-tmi/scripts/visualize_extreme_interactions.py

