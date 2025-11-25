#!/usr/bin/env python3
"""
Test media setup to diagnose why wild-type growth is 0.
"""

import sys
sys.path.append('/home/michaelvolk/Documents/projects/torchcell')
from torchcell.metabolism.yeast_GEM import YeastGEM

sys.path.append('/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/scripts')
from setup_media_conditions import setup_minimal_media, setup_ynb_media, setup_ypd_media

def diagnose_media():
    """Diagnose media setup issues."""

    print("Loading Yeast9 model...")
    yeast_gem = YeastGEM()
    model = yeast_gem.model

    print(f"Model: {model.id}")
    print(f"Reactions: {len(model.reactions)}")
    print(f"Exchanges: {len(model.exchanges)}")

    # Test default model
    print("\n" + "="*50)
    print("DEFAULT MODEL STATE")
    print("="*50)

    # Check default medium
    default_medium = model.medium
    print(f"Default medium has {len(default_medium)} exchanges open")

    # Show first few exchanges
    print("\nFirst 10 default exchanges:")
    for i, (rxn_id, flux) in enumerate(list(default_medium.items())[:10]):
        rxn = model.reactions.get_by_id(rxn_id)
        print(f"  {rxn_id}: {rxn.name} = {flux}")

    # Test growth with default medium
    solution = model.optimize()
    print(f"\nDefault wild-type growth: {solution.objective_value:.4f}")
    print(f"Solution status: {solution.status}")

    # Test minimal media
    print("\n" + "="*50)
    print("MINIMAL MEDIA TEST")
    print("="*50)

    model_min = model.copy()
    setup_minimal_media(model_min, glucose_rate=10.0)

    # Check which exchanges are open
    open_exchanges = []
    for rxn in model_min.exchanges:
        if rxn.lower_bound < 0:
            open_exchanges.append((rxn.id, rxn.name, rxn.lower_bound))

    print(f"Open exchanges: {len(open_exchanges)}")
    for rxn_id, name, bound in open_exchanges[:20]:
        print(f"  {rxn_id}: {name} = {bound}")

    # Check if critical exchanges exist
    critical = {
        'r_1714': 'glucose',
        'r_1654': 'ammonium',
        'r_1992': 'oxygen',
        'r_2100': 'water',
        'r_2005': 'phosphate',
        'r_2060': 'sulfate'
    }

    print("\nCritical exchange status:")
    for rxn_id, name in critical.items():
        if rxn_id in model_min.reactions:
            rxn = model_min.reactions.get_by_id(rxn_id)
            print(f"  {name} ({rxn_id}): lower={rxn.lower_bound}, upper={rxn.upper_bound}")
        else:
            print(f"  {name} ({rxn_id}): NOT FOUND")

    solution_min = model_min.optimize()
    print(f"\nMinimal media growth: {solution_min.objective_value:.4f}")
    print(f"Solution status: {solution_min.status}")

    # If no growth, check what's limiting
    if solution_min.objective_value < 0.01:
        print("\n⚠ NO GROWTH - Checking shadow prices...")

        # Check shadow prices to see what's limiting
        if hasattr(solution_min, 'shadow_prices'):
            limiting = sorted(solution_min.shadow_prices.items(),
                            key=lambda x: abs(x[1]), reverse=True)[:10]
            print("Top limiting metabolites:")
            for met_id, price in limiting:
                if abs(price) > 0.01:
                    met = model_min.metabolites.get_by_id(met_id)
                    print(f"  {met.name} ({met_id}): {price:.4f}")

    # Test YPD media
    print("\n" + "="*50)
    print("YPD MEDIA TEST")
    print("="*50)

    model_ypd = model.copy()
    model_ypd, ynb_added, ynb_missing, aa_added, aa_missing = setup_ypd_media(model_ypd, glucose_rate=10.0)

    print(f"YNB added: {len(ynb_added)}")
    print(f"Amino acids added: {len(aa_added)}")

    solution_ypd = model_ypd.optimize()
    print(f"\nYPD media growth: {solution_ypd.objective_value:.4f}")
    print(f"Solution status: {solution_ypd.status}")

    # Try using default medium setup
    print("\n" + "="*50)
    print("USING MODEL'S DEFAULT MEDIUM")
    print("="*50)

    model_default = model.copy()
    # Don't reset exchanges, use default medium
    solution_default = model_default.optimize()
    print(f"Growth with model's default medium: {solution_default.objective_value:.4f}")

    # Try manually setting glucose exchange
    print("\n" + "="*50)
    print("MANUAL GLUCOSE SETUP TEST")
    print("="*50)

    model_manual = model.copy()

    # Find glucose exchange by searching
    glucose_rxns = [r for r in model_manual.exchanges
                    if 'glucose' in r.name.lower() or 'glc' in r.id.lower()]

    print(f"Found {len(glucose_rxns)} glucose-related exchanges:")
    for rxn in glucose_rxns:
        print(f"  {rxn.id}: {rxn.name}")
        print(f"    Current bounds: [{rxn.lower_bound}, {rxn.upper_bound}]")

    # Try to set the first glucose exchange
    if glucose_rxns:
        glucose_rxns[0].lower_bound = -10.0
        print(f"\nSet {glucose_rxns[0].id} lower bound to -10.0")

        solution_manual = model_manual.optimize()
        print(f"Growth with manual glucose: {solution_manual.objective_value:.4f}")

    # Find chloride and sodium exchanges
    print("\n" + "="*50)
    print("FINDING MISSING NUTRIENTS")
    print("="*50)

    # Look for chloride exchange
    chloride_rxns = [r for r in model.exchanges
                     if 'chloride' in r.name.lower() or 'cl' in r.id.lower()]
    print(f"\nChloride-related exchanges:")
    for rxn in chloride_rxns:
        print(f"  {rxn.id}: {rxn.name}")
        print(f"    Bounds: [{rxn.lower_bound}, {rxn.upper_bound}]")

    # Look for sodium exchange
    sodium_rxns = [r for r in model.exchanges
                   if 'sodium' in r.name.lower() or 'na' in r.id.lower()]
    print(f"\nSodium-related exchanges:")
    for rxn in sodium_rxns:
        print(f"  {rxn.id}: {rxn.name}")
        print(f"    Bounds: [{rxn.lower_bound}, {rxn.upper_bound}]")

    # Test minimal media with chloride and sodium
    print("\n" + "="*50)
    print("MINIMAL MEDIA + CHLORIDE + SODIUM TEST")
    print("="*50)

    model_fixed = model.copy()
    setup_minimal_media(model_fixed, glucose_rate=10.0)

    # Add chloride if found
    if 'r_4635' in model_fixed.reactions:  # chloride exchange
        model_fixed.reactions.r_4635.lower_bound = -1000.0
        print("Added chloride exchange")

    # Add sodium if found
    if 'r_2049' in model_fixed.reactions:  # sodium exchange
        model_fixed.reactions.r_2049.lower_bound = -1000.0
        print("Added sodium exchange")

    solution_fixed = model_fixed.optimize()
    print(f"\nMinimal + Cl + Na growth: {solution_fixed.objective_value:.4f}")
    print(f"Solution status: {solution_fixed.status}")

if __name__ == "__main__":
    diagnose_media()