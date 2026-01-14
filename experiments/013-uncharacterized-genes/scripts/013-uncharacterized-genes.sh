#!/bin/bash

# Step 1: Count and classify uncharacterized/dubious genes
python experiments/013-uncharacterized-genes/scripts/count_dubious_and_uncharacterized_genes.py

# Step 2: Analyze triple interactions involving uncharacterized genes
python experiments/013-uncharacterized-genes/scripts/triple_interaction_enrichment_of_uncharacterized_genes.py

# Step 3: Analyze essential and uncharacterized gene overlap
python experiments/013-uncharacterized-genes/scripts/uncharacterized_essential_overlap.py