#!/bin/bash
# Case Study: 2,3-Butanediol Production (Ng et al. 2012)
# Predicting trigenic interactions for ADH/ALD knockouts
# Run from project root: bash experiments/010-kuzmin-tmi/scripts/case_study_ngProduction23butanediolSaccharomyces2012.bash

# Step 1: Lookup Costanzo2016 SMF/DMF/ε data for ADH/ALD genes
python experiments/010-kuzmin-tmi/scripts/adh_ald_costanzo2016_lookup.py

# Step 2: Run CellGraphTransformer inference for triple knockouts
python experiments/010-kuzmin-tmi/scripts/trigenic_interaction_adh1_adh3_adh5.py
