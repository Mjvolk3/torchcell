#!/usr/bin/env python3
"""
Analyze how to set up YPD media based on Suthers model structure.
Look for all exchange reactions and determine appropriate setup.
"""

import cobra
import pandas as pd
from cobra.io import load_json_model, load_model
import os

def find_all_nutrient_exchanges(model):
    """Find all exchange reactions that could be used for YPD media."""

    exchanges_data = []

    # YNB components from paper
    ynb_keywords = ['thiamine', 'riboflavin', 'nicotinate', 'nicotinic', 'pyridoxin',
                    'pyridoxine', 'folate', 'folic', 'pantothenate', 'pantothenic',
                    'aminobenzoate', 'paba', 'inositol', 'biotin']

    # Amino acids
    aa_keywords = ['alanine', 'arginine', 'asparagine', 'aspartate', 'aspartic',
                   'cysteine', 'glutamate', 'glutamic', 'glutamine', 'glycine',
                   'histidine', 'isoleucine', 'leucine', 'lysine', 'methionine',
                   'phenylalanine', 'proline', 'serine', 'threonine', 'tryptophan',
                   'tyrosine', 'valine']

    for rxn in model.exchanges:
        mets = list(rxn.metabolites.keys())
        if mets:
            met = mets[0]
            met_name = met.name.lower()
            met_id = met.id.lower()
            rxn_name = rxn.name.lower()

            # Check category
            category = 'other'
            matched_name = ''

            # Check if it's a YNB component
            for ynb in ynb_keywords:
                if ynb in met_name or ynb in met_id or ynb in rxn_name:
                    category = 'YNB'
                    matched_name = ynb
                    break

            # Check if it's an amino acid
            if category == 'other':
                for aa in aa_keywords:
                    if aa in met_name or aa in met_id or aa in rxn_name:
                        category = 'amino_acid'
                        matched_name = aa
                        break

            # Check basic categories
            if category == 'other':
                if 'glucose' in met_name or 'glucose' in met_id:
                    category = 'carbon'
                    matched_name = 'glucose'
                elif 'ammonium' in met_name or 'nh4' in met_id or 'nh3' in met_id:
                    category = 'nitrogen'
                    matched_name = 'ammonium'

            exchanges_data.append({
                'reaction_id': rxn.id,
                'reaction_name': rxn.name,
                'metabolite_id': met.id,
                'metabolite_name': met.name,
                'category': category,
                'matched': matched_name,
                'current_lb': rxn.lower_bound,
                'current_ub': rxn.upper_bound
            })

    return pd.DataFrame(exchanges_data)

def main():
    print("=" * 70)
    print("Analyzing Suthers Model for YPD Media Setup")
    print("=" * 70)

    # Load model
    model_path = "/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/1-s2.0-S2214030120300481-mmc8/iIsor850.json"
    print(f"\nLoading I. orientalis model...")
    model = load_json_model(model_path)

    # Find all potential exchanges
    print("\nAnalyzing all exchange reactions...")
    exchanges_df = find_all_nutrient_exchanges(model)

    # Group by category
    print("\n" + "-" * 50)
    print("EXCHANGE REACTIONS BY CATEGORY:")
    print("-" * 50)

    for category in ['YNB', 'amino_acid', 'carbon', 'nitrogen', 'other']:
        cat_df = exchanges_df[exchanges_df['category'] == category]
        print(f"\n{category.upper()}: {len(cat_df)} reactions")

        if category in ['YNB', 'amino_acid'] and not cat_df.empty:
            for _, row in cat_df.head(30).iterrows():
                status = "OPEN" if row['current_lb'] < 0 else "CLOSED"
                print(f"  {row['reaction_id']}: {row['metabolite_name']} [{status}]")
                if category == 'YNB':
                    print(f"    -> Matches: {row['matched']}")

    # Generate YPD setup code
    print("\n" + "=" * 70)
    print("GENERATING YPD SETUP CODE FOR I. ORIENTALIS MODEL")
    print("=" * 70)

    ynb_rxns = exchanges_df[exchanges_df['category'] == 'YNB']
    aa_rxns = exchanges_df[exchanges_df['category'] == 'amino_acid']

    print("\ndef setup_ypd_media_isor(model, glucose_rate=10.0):")
    print('    """Set up YPD media following Suthers et al., 2020."""')
    print("    ")
    print("    # Reset all exchanges")
    print("    for rxn in model.exchanges:")
    print("        rxn.lower_bound = 0")
    print("    ")
    print("    # Supplement rate (5% of glucose as per paper)")
    print("    supplement_rate = glucose_rate * 0.05  # 0.165 when glucose=3.3")
    print("    ")
    print("    # Basic nutrients")

    # Find glucose exchange
    glucose_rxns = exchanges_df[exchanges_df['matched'] == 'glucose']
    if not glucose_rxns.empty:
        print(f"    model.reactions.{glucose_rxns.iloc[0]['reaction_id']}.lower_bound = -glucose_rate")

    # Find ammonium exchange
    nh4_rxns = exchanges_df[exchanges_df['matched'] == 'ammonium']
    if not nh4_rxns.empty:
        print(f"    model.reactions.{nh4_rxns.iloc[0]['reaction_id']}.lower_bound = -1000")

    print("    ")
    print("    # YNB components (if available)")

    if not ynb_rxns.empty:
        for _, rxn in ynb_rxns.iterrows():
            print(f"    if '{rxn['reaction_id']}' in model.reactions:")
            print(f"        model.reactions.{rxn['reaction_id']}.lower_bound = -supplement_rate  # {rxn['matched']}")
    else:
        print("    # Note: YNB components not found as separate exchanges in this model")

    print("    ")
    print("    # 20 Amino acids")

    if not aa_rxns.empty:
        for _, rxn in aa_rxns.iterrows():
            print(f"    model.reactions.{rxn['reaction_id']}.lower_bound = -supplement_rate  # {rxn['metabolite_name']}")
    else:
        print("    # Note: Amino acid exchanges not found in this model")

    # Now map to Yeast9
    print("\n" + "=" * 70)
    print("MAPPING TO YEAST9 S. CEREVISIAE MODEL")
    print("=" * 70)

    yeast9_path = "/home/michaelvolk/Documents/yeast-GEM/model/yeast-GEM.xml"
    if os.path.exists(yeast9_path):
        print("\nLoading Yeast9 model...")
        yeast9 = load_model(yeast9_path)

        # Analyze Yeast9 exchanges
        yeast9_exchanges = find_all_nutrient_exchanges(yeast9)

        print("\nYeast9 exchange reactions by category:")
        for category in ['YNB', 'amino_acid', 'carbon', 'nitrogen']:
            cat_df = yeast9_exchanges[yeast9_exchanges['category'] == category]
            print(f"  {category}: {len(cat_df)} reactions")

        # Generate setup code for Yeast9
        print("\n" + "-" * 50)
        print("YPD SETUP CODE FOR YEAST9:")
        print("-" * 50)

        ynb_y9 = yeast9_exchanges[yeast9_exchanges['category'] == 'YNB']
        aa_y9 = yeast9_exchanges[yeast9_exchanges['category'] == 'amino_acid']

        print("\ndef setup_ypd_yeast9(model, glucose_rate=10.0):")
        print('    """YPD media for Yeast9 based on Suthers approach."""')
        print("    ")
        print("    # Reset exchanges")
        print("    for rxn in model.exchanges:")
        print("        rxn.lower_bound = 0")
        print("    ")
        print("    supplement_rate = glucose_rate * 0.05")
        print("    ")
        print("    # Core nutrients")
        print("    model.reactions.r_1714.lower_bound = -glucose_rate  # glucose")
        print("    model.reactions.r_1992.lower_bound = -1000  # oxygen")
        print("    model.reactions.r_1654.lower_bound = -1000  # ammonium")
        print("    model.reactions.r_2100.lower_bound = -1000  # water")
        print("    model.reactions.r_2005.lower_bound = -1000  # phosphate")
        print("    model.reactions.r_2060.lower_bound = -1000  # sulfate")
        print("    ")

        if not ynb_y9.empty:
            print("    # YNB components found in Yeast9:")
            for _, rxn in ynb_y9.head(10).iterrows():
                print(f"    model.reactions.{rxn['reaction_id']}.lower_bound = -supplement_rate  # {rxn['matched']}")

        if not aa_y9.empty:
            print("    ")
            print("    # Amino acids found in Yeast9:")
            unique_aa = aa_y9['matched'].unique()
            for aa in sorted(unique_aa)[:20]:  # Get first 20 unique amino acids
                aa_rxn = aa_y9[aa_y9['matched'] == aa].iloc[0]
                print(f"    model.reactions.{aa_rxn['reaction_id']}.lower_bound = -supplement_rate  # {aa}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()