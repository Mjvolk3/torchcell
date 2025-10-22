#!/usr/bin/env python3
"""
Set up YNB and YPD media conditions for Yeast9 S. cerevisiae model.
Based on Suthers et al., 2020 approach:
- YNB: Minimal + vitamins/cofactors
- YPD: YNB + 20 amino acids
- Supplements at 5% of glucose uptake rate
"""

import cobra
from cobra.io import read_sbml_model
import pandas as pd
import numpy as np


def reset_media(model):
    """Reset all exchange reactions to closed state."""
    for rxn in model.exchanges:
        rxn.lower_bound = 0
    return model


def setup_minimal_media(model, glucose_rate=10.0):
    """Set up minimal media with just glucose, NH4, O2, and inorganics."""

    # Reset first
    reset_media(model)

    # Essential nutrients - matching YeastGEM default medium
    # All trace elements are required for proper biomass formation
    exchanges = {
        'r_1714': -glucose_rate,    # D-glucose exchange (carbon source)
        'r_1654': -1000.0,          # ammonium exchange (nitrogen source)
        'r_1992': -1000.0,          # oxygen exchange
        'r_2100': -1000.0,          # H2O exchange
        'r_2005': -1000.0,          # phosphate exchange
        'r_2060': -1000.0,          # sulfate exchange
        'r_1832': -1000.0,          # H+ exchange
        'r_1861': -1000.0,          # iron(2+) exchange
        'r_4593': -1000.0,          # chloride exchange
        'r_2049': -1000.0,          # sodium exchange
        'r_2020': -1000.0,          # potassium exchange
        'r_4594': -1000.0,          # Cu2(+) exchange
        'r_4595': -1000.0,          # Mn(2+) exchange
        'r_4596': -1000.0,          # Zn(2+) exchange
        'r_4597': -1000.0,          # Mg(2+) exchange
        'r_4600': -1000.0,          # Ca(2+) exchange
    }

    # Set bounds
    for rxn_id, bound in exchanges.items():
        if rxn_id in model.reactions:
            model.reactions.get_by_id(rxn_id).lower_bound = bound
        else:
            print(f"Warning: {rxn_id} not found in model")

    return model


def setup_ynb_media(model, glucose_rate=10.0):
    """
    Set up YNB (Yeast Nitrogen Base) media.
    YNB = minimal media + vitamins/cofactors
    Based on Suthers et al., 2020
    """

    # Start with minimal media
    setup_minimal_media(model, glucose_rate)

    # Supplement rate: 5% of glucose rate (Suthers et al., 2020)
    supplement_rate = glucose_rate * 0.05

    # YNB vitamin/cofactor components
    # These are the 8 components mentioned in Suthers paper
    ynb_exchanges = {
        # Vitamins and cofactors - VERIFIED REACTION IDs
        'r_2067': 'thiamine',           # thiamine(1+) exchange
        'r_2038': 'riboflavin',         # riboflavin exchange
        'r_1967': 'nicotinate',         # nicotinate exchange
        'r_2028': 'pyridoxine',         # pyridoxine exchange
        'r_1625': 'folate',             # 5-formyltetrahydrofolic acid exchange
        'r_1548': 'pantothenate',       # (R)-pantothenate exchange
        'r_1604': '4-aminobenzoate',    # 4-aminobenzoate exchange
        'r_1947': 'myo-inositol',       # myo-inositol exchange

        # Additional common YNB components
        'r_1671': 'biotin',             # biotin exchange
    }

    # Open YNB component exchanges
    components_added = []
    components_missing = []

    for rxn_id, name in ynb_exchanges.items():
        if rxn_id in model.reactions:
            model.reactions.get_by_id(rxn_id).lower_bound = -supplement_rate
            components_added.append(name)
        else:
            # Try alternative search by name
            found = False
            for rxn in model.exchanges:
                if name.lower() in rxn.name.lower():
                    rxn.lower_bound = -supplement_rate
                    components_added.append(f"{name} (via {rxn.id})")
                    found = True
                    break
            if not found:
                components_missing.append(name)

    return model, components_added, components_missing


def setup_ypd_media(model, glucose_rate=10.0):
    """
    Set up YPD (Yeast Peptone Dextrose) media.
    YPD = YNB + 20 amino acids
    Based on Suthers et al., 2020 approximation
    """

    # Start with YNB media
    model, ynb_added, ynb_missing = setup_ynb_media(model, glucose_rate)

    # Supplement rate: 5% of glucose rate
    supplement_rate = glucose_rate * 0.05

    # 20 standard amino acids - VERIFIED REACTION IDs
    amino_acid_exchanges = {
        'r_1873': 'L-alanine',          # L-alanine exchange
        'r_1879': 'L-arginine',         # L-arginine exchange
        'r_1880': 'L-asparagine',       # L-asparagine exchange
        'r_1881': 'L-aspartate',        # L-aspartate exchange
        'r_1883': 'L-cysteine',         # L-cysteine exchange
        'r_1889': 'L-glutamate',        # L-glutamate exchange
        'r_1891': 'L-glutamine',        # L-glutamine exchange
        'r_1810': 'glycine',            # L-glycine exchange
        'r_1893': 'L-histidine',        # L-histidine exchange
        'r_1897': 'L-isoleucine',       # L-isoleucine exchange
        'r_1899': 'L-leucine',          # L-leucine exchange
        'r_1900': 'L-lysine',           # L-lysine exchange
        'r_1902': 'L-methionine',       # L-methionine exchange
        'r_1903': 'L-phenylalanine',    # L-phenylalanine exchange
        'r_1904': 'L-proline',          # L-proline exchange
        'r_1906': 'L-serine',           # L-serine exchange
        'r_1911': 'L-threonine',        # L-threonine exchange
        'r_1912': 'L-tryptophan',       # L-tryptophan exchange
        'r_1913': 'L-tyrosine',         # L-tyrosine exchange
        'r_1914': 'L-valine',           # L-valine exchange
    }

    # Open amino acid exchanges
    aa_added = []
    aa_missing = []

    for rxn_id, name in amino_acid_exchanges.items():
        if rxn_id in model.reactions:
            model.reactions.get_by_id(rxn_id).lower_bound = -supplement_rate
            aa_added.append(name)
        else:
            # Try alternative search
            found = False
            for rxn in model.exchanges:
                aa_simple = name.replace('L-', '').lower()
                if aa_simple in rxn.name.lower() and 'exchange' in rxn.name.lower():
                    rxn.lower_bound = -supplement_rate
                    aa_added.append(f"{name} (via {rxn.id})")
                    found = True
                    break
            if not found:
                aa_missing.append(name)

    return model, ynb_added, ynb_missing, aa_added, aa_missing


def get_media_summary(model):
    """Get summary of current media setup."""

    open_exchanges = []

    for rxn in model.exchanges:
        if rxn.lower_bound < 0:
            mets = list(rxn.metabolites.keys())
            met_name = mets[0].name if mets else "Unknown"

            open_exchanges.append({
                'reaction_id': rxn.id,
                'metabolite': met_name,
                'uptake_rate': abs(rxn.lower_bound)
            })

    df = pd.DataFrame(open_exchanges)

    # Categorize by uptake rate
    if not df.empty:
        df['category'] = 'other'
        df.loc[df['uptake_rate'] > 900, 'category'] = 'unlimited'
        df.loc[(df['uptake_rate'] > 5) & (df['uptake_rate'] < 20), 'category'] = 'carbon'
        df.loc[(df['uptake_rate'] > 0.1) & (df['uptake_rate'] < 1), 'category'] = 'supplement'

    return df


def compare_media_conditions(model):
    """Compare growth rates across different media conditions."""

    results = []

    # Test minimal media
    model_min = model.copy()
    setup_minimal_media(model_min)
    solution_min = model_min.optimize()
    results.append({
        'media': 'Minimal',
        'growth_rate': solution_min.objective_value if solution_min.status == 'optimal' else 0,
        'open_exchanges': len([r for r in model_min.exchanges if r.lower_bound < 0])
    })

    # Test YNB media
    model_ynb = model.copy()
    setup_ynb_media(model_ynb)
    solution_ynb = model_ynb.optimize()
    results.append({
        'media': 'YNB',
        'growth_rate': solution_ynb.objective_value if solution_ynb.status == 'optimal' else 0,
        'open_exchanges': len([r for r in model_ynb.exchanges if r.lower_bound < 0])
    })

    # Test YPD media
    model_ypd = model.copy()
    setup_ypd_media(model_ypd)
    solution_ypd = model_ypd.optimize()
    results.append({
        'media': 'YPD',
        'growth_rate': solution_ypd.objective_value if solution_ypd.status == 'optimal' else 0,
        'open_exchanges': len([r for r in model_ypd.exchanges if r.lower_bound < 0])
    })

    return pd.DataFrame(results)


def main():
    """Demo usage of media setup functions."""

    print("=" * 70)
    print("Media Setup for Yeast9 S. cerevisiae Model")
    print("Based on Suthers et al., 2020 approach")
    print("=" * 70)

    # Load model
    model_path = "/home/michaelvolk/Documents/yeast-GEM/model/yeast-GEM.xml"
    print(f"\nLoading model from: {model_path}")
    model = read_sbml_model(model_path)

    print(f"Model: {model.id}")
    print(f"Reactions: {len(model.reactions)}")
    print(f"Genes: {len(model.genes)}")

    # Test YNB setup
    print("\n" + "-" * 50)
    print("Setting up YNB media...")
    model_ynb = model.copy()
    model_ynb, ynb_added, ynb_missing = setup_ynb_media(model_ynb, glucose_rate=10.0)

    print(f"YNB components added: {len(ynb_added)}")
    if ynb_added:
        print(f"  Added: {', '.join(ynb_added[:5])}...")
    if ynb_missing:
        print(f"  Missing: {', '.join(ynb_missing)}")

    # Test YPD setup
    print("\n" + "-" * 50)
    print("Setting up YPD media...")
    model_ypd = model.copy()
    model_ypd, ynb_added, ynb_missing, aa_added, aa_missing = setup_ypd_media(model_ypd, glucose_rate=10.0)

    print(f"YPD setup complete:")
    print(f"  YNB components: {len(ynb_added)} added")
    print(f"  Amino acids: {len(aa_added)} added")
    if aa_missing:
        print(f"  Missing amino acids: {', '.join(aa_missing)}")

    # Compare media conditions
    print("\n" + "-" * 50)
    print("Comparing media conditions...")
    comparison = compare_media_conditions(model)
    print("\n", comparison.to_string(index=False))

    # Show media summary for YPD
    print("\n" + "-" * 50)
    print("YPD media summary:")
    ypd_summary = get_media_summary(model_ypd)

    if not ypd_summary.empty:
        print(f"\nTotal open exchanges: {len(ypd_summary)}")

        for category in ['carbon', 'supplement', 'unlimited', 'other']:
            cat_df = ypd_summary[ypd_summary['category'] == category]
            if not cat_df.empty:
                print(f"\n{category.upper()}: {len(cat_df)} metabolites")
                if category == 'supplement':
                    # Show some examples
                    for _, row in cat_df.head(5).iterrows():
                        print(f"  {row['metabolite']}: {row['uptake_rate']:.3f} mmol/gDW/h")

    print("\n" + "=" * 70)
    print("Media setup complete!")
    print("\nUsage in your scripts:")
    print("  from setup_media_conditions import setup_ynb_media, setup_ypd_media")
    print("  model_ypd, *_ = setup_ypd_media(model, glucose_rate=10.0)")
    print("=" * 70)


if __name__ == "__main__":
    main()