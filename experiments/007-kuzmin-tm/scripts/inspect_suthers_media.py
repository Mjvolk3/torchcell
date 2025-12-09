#!/usr/bin/env python3
"""
Inspect the Suthers et al. I. orientalis model to understand how they set up YPD media.
Map their approach to S. cerevisiae Yeast9 model.
"""

import cobra
import pandas as pd
from cobra.io import load_json_model, load_model
import os

def inspect_exchange_reactions(model, media_type="all"):
    """Inspect exchange reactions and their bounds."""

    exchanges = []

    for rxn in model.exchanges:
        # Get metabolite if available
        mets = list(rxn.metabolites.keys())
        met_name = mets[0].name if mets else "Unknown"
        met_id = mets[0].id if mets else "Unknown"

        # Check if reaction is open (allows uptake)
        is_uptake = rxn.lower_bound < 0

        exchanges.append({
            'reaction_id': rxn.id,
            'reaction_name': rxn.name,
            'metabolite_id': met_id,
            'metabolite_name': met_name,
            'lower_bound': rxn.lower_bound,
            'upper_bound': rxn.upper_bound,
            'uptake_allowed': is_uptake,
            'uptake_rate': abs(rxn.lower_bound) if is_uptake else 0
        })

    df = pd.DataFrame(exchanges)

    # Filter for active uptake reactions
    active_uptake = df[df['uptake_allowed'] == True].sort_values('uptake_rate', ascending=False)

    return df, active_uptake

def categorize_nutrients(active_df):
    """Categorize nutrients into groups."""

    categories = {
        'carbon': [],
        'nitrogen': [],
        'amino_acids': [],
        'vitamins': [],
        'nucleotides': [],
        'inorganic': [],
        'other': []
    }

    for _, row in active_df.iterrows():
        name_lower = row['metabolite_name'].lower()

        # Carbon sources
        if any(c in name_lower for c in ['glucose', 'fructose', 'glycerol', 'ethanol', 'acetate']):
            categories['carbon'].append(row)
        # Nitrogen sources
        elif any(n in name_lower for n in ['ammonium', 'ammonia', 'nh4', 'nh3']):
            categories['nitrogen'].append(row)
        # Amino acids
        elif any(aa in name_lower for aa in ['alanine', 'arginine', 'asparagine', 'aspartate',
                                              'cysteine', 'glutamate', 'glutamine', 'glycine',
                                              'histidine', 'isoleucine', 'leucine', 'lysine',
                                              'methionine', 'phenylalanine', 'proline', 'serine',
                                              'threonine', 'tryptophan', 'tyrosine', 'valine']):
            categories['amino_acids'].append(row)
        # Vitamins/cofactors (YNB components)
        elif any(v in name_lower for v in ['thiamine', 'riboflavin', 'nicotinate', 'pyridoxin',
                                           'folate', 'pantothenate', 'biotin', 'inositol',
                                           'aminobenzoate', 'vitamin']):
            categories['vitamins'].append(row)
        # Nucleotides
        elif any(n in name_lower for n in ['adenine', 'guanine', 'cytosine', 'thymine', 'uracil']):
            categories['nucleotides'].append(row)
        # Inorganic
        elif any(i in name_lower for i in ['phosphate', 'sulfate', 'oxygen', 'water', 'h2o',
                                           'iron', 'magnesium', 'calcium', 'potassium', 'sodium']):
            categories['inorganic'].append(row)
        else:
            categories['other'].append(row)

    return categories

def main():
    print("=" * 70)
    print("Inspecting Suthers et al. I. orientalis Model Media Setup")
    print("=" * 70)

    # Load the I. orientalis model
    model_path = "/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/1-s2.0-S2214030120300481-mmc8/iIsor850.json"

    print(f"\nLoading model from: {model_path}")
    isor_model = load_json_model(model_path)

    print(f"Model: {isor_model.id}")
    print(f"Reactions: {len(isor_model.reactions)}")
    print(f"Metabolites: {len(isor_model.metabolites)}")
    print(f"Genes: {len(isor_model.genes)}")
    print(f"Exchange reactions: {len(isor_model.exchanges)}")

    # Inspect exchange reactions
    print("\n" + "=" * 70)
    print("ANALYZING EXCHANGE REACTIONS (DEFAULT MEDIA)")
    print("=" * 70)

    all_exchanges, active_uptake = inspect_exchange_reactions(isor_model)

    print(f"\nTotal exchange reactions: {len(all_exchanges)}")
    print(f"Active uptake reactions (lower_bound < 0): {len(active_uptake)}")

    # Categorize active nutrients
    categories = categorize_nutrients(active_uptake)

    print("\n" + "-" * 50)
    print("NUTRIENT CATEGORIES IN DEFAULT MEDIA:")
    print("-" * 50)

    for category, nutrients in categories.items():
        if nutrients:
            print(f"\n{category.upper()} ({len(nutrients)} compounds):")
            for nut in nutrients[:5]:  # Show first 5
                print(f"  {nut['metabolite_name']}: {nut['uptake_rate']:.4f} mmol/gDW/h")
            if len(nutrients) > 5:
                print(f"  ... and {len(nutrients)-5} more")

    # Check for specific YNB components mentioned in paper
    print("\n" + "-" * 50)
    print("YNB COMPONENTS (from paper):")
    print("-" * 50)

    ynb_components = ['thiamine', 'riboflavin', 'nicotinate', 'pyridoxin',
                      'folate', 'pantothenate', '4-aminobenzoate', 'myo-inositol']

    for component in ynb_components:
        found = active_uptake[active_uptake['metabolite_name'].str.contains(component, case=False, na=False)]
        if not found.empty:
            for _, row in found.iterrows():
                print(f"  {component}: {row['metabolite_name']} = {row['uptake_rate']:.4f} mmol/gDW/h")
        else:
            print(f"  {component}: NOT FOUND in active uptake")

    # Check amino acids
    print("\n" + "-" * 50)
    print("AMINO ACIDS STATUS:")
    print("-" * 50)

    amino_acids = ['alanine', 'arginine', 'asparagine', 'aspartate', 'cysteine',
                   'glutamate', 'glutamine', 'glycine', 'histidine', 'isoleucine',
                   'leucine', 'lysine', 'methionine', 'phenylalanine', 'proline',
                   'serine', 'threonine', 'tryptophan', 'tyrosine', 'valine']

    aa_found = []
    aa_not_found = []

    for aa in amino_acids:
        found = active_uptake[active_uptake['metabolite_name'].str.contains(aa, case=False, na=False)]
        if not found.empty:
            aa_found.append(aa)
        else:
            aa_not_found.append(aa)

    print(f"Amino acids with uptake allowed: {len(aa_found)}/20")
    if aa_found:
        print(f"  Found: {', '.join(aa_found[:10])}")
    if aa_not_found:
        print(f"  Not found: {', '.join(aa_not_found[:10])}")

    # Analyze uptake rate patterns
    print("\n" + "-" * 50)
    print("UPTAKE RATE PATTERNS:")
    print("-" * 50)

    # Group by uptake rate
    rate_groups = active_uptake.groupby('uptake_rate').size()

    print("\nCommon uptake rates (mmol/gDW/h):")
    for rate, count in rate_groups.sort_values(ascending=False).head(10).items():
        if rate > 0:
            print(f"  {rate:.4f}: {count} metabolites")

    # Check for the 0.165 rate mentioned in paper (5% of 3.3)
    supplement_rate = 0.165
    tolerance = 0.01

    supplements = active_uptake[
        (active_uptake['uptake_rate'] > supplement_rate - tolerance) &
        (active_uptake['uptake_rate'] < supplement_rate + tolerance)
    ]

    if not supplements.empty:
        print(f"\nMetabolites with uptake rate near {supplement_rate} (YNB/YPD supplements):")
        for _, row in supplements.iterrows():
            print(f"  {row['metabolite_name']}: {row['uptake_rate']:.4f}")

    # Map to S. cerevisiae
    print("\n" + "=" * 70)
    print("MAPPING TO S. CEREVISIAE YEAST9 MODEL")
    print("=" * 70)

    # Load Yeast9 for comparison
    yeast9_path = "/home/michaelvolk/Documents/yeast-GEM/model/yeast-GEM.xml"

    if os.path.exists(yeast9_path):
        print(f"\nLoading Yeast9 model for comparison...")
        yeast9 = load_model(yeast9_path)

        # Find corresponding exchange reactions
        print("\nSearching for corresponding YNB components in Yeast9...")

        yeast9_exchanges = {rxn.id: rxn for rxn in yeast9.exchanges}

        mapping = []
        for component in ynb_components:
            # Search in Yeast9
            found_in_yeast9 = []
            for rxn_id, rxn in yeast9_exchanges.items():
                if component.lower() in rxn.name.lower() or component.lower() in rxn.id.lower():
                    found_in_yeast9.append((rxn.id, rxn.name))

            if found_in_yeast9:
                print(f"  {component}: Found in Yeast9 as {found_in_yeast9[0]}")
                mapping.append({
                    'nutrient': component,
                    'yeast9_rxn': found_in_yeast9[0][0],
                    'rate': supplement_rate
                })
            else:
                print(f"  {component}: NOT FOUND in Yeast9 exchanges")

        # Save mapping
        print("\n" + "-" * 50)
        print("RECOMMENDED YPD SETUP FOR YEAST9:")
        print("-" * 50)
        print("\n# Based on Suthers et al. approach")
        print("# YPD = YNB + 20 amino acids + glucose")
        print("# Supplement rate = 0.165 mmol/gDW/h (5% of glucose)")
        print("\ndef setup_ypd_media(model, glucose_rate=10.0):")
        print("    # Reset exchanges")
        print("    for rxn in model.exchanges:")
        print("        rxn.lower_bound = 0")
        print("    ")
        print("    # Basic nutrients")
        print("    model.reactions.r_1714.lower_bound = -glucose_rate  # glucose")
        print("    model.reactions.r_1992.lower_bound = -1000  # oxygen")
        print("    model.reactions.r_1654.lower_bound = -1000  # NH4+")
        print("    ")
        print("    # YNB components (0.165 mmol/gDW/h)")
        print("    supplement_rate = glucose_rate * 0.05")

        if mapping:
            for m in mapping:
                print(f"    model.reactions.{m['yeast9_rxn']}.lower_bound = -supplement_rate  # {m['nutrient']}")

        print("    ")
        print("    # Add 20 amino acids at same rate")
        print("    # ... (amino acid exchange reactions)")

    # Save analysis results
    output_file = "/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/results/cobra-fba-growth/suthers_media_analysis.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("SUTHERS MODEL MEDIA ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Active uptake reactions: {len(active_uptake)}\n\n")
        f.write(active_uptake.to_string())

    print(f"\nFull analysis saved to: {output_file}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()