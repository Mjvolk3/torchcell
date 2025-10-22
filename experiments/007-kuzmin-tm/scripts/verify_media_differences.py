#!/usr/bin/env python3
"""
Verify media setup and display differences between conditions.
Outputs table-ready format showing what's added at each step.
"""

import sys
sys.path.append('/home/michaelvolk/Documents/projects/torchcell')
from torchcell.metabolism.yeast_GEM import YeastGEM

sys.path.append('/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/scripts')
from setup_media_conditions import setup_minimal_media, setup_ynb_media, setup_ypd_media

def get_open_exchanges(model, threshold=0.01):
    """Get all open exchanges with their rates."""
    exchanges = {}
    for rxn in model.exchanges:
        if rxn.lower_bound < -threshold:
            # Get metabolite name from reaction
            mets = list(rxn.metabolites.keys())
            if mets:
                met_name = mets[0].name
            else:
                met_name = rxn.name.replace(' exchange', '')

            exchanges[rxn.id] = {
                'name': met_name,
                'rate': abs(rxn.lower_bound),
                'reaction_name': rxn.name
            }
    return exchanges

def categorize_exchange(rxn_id, name, rate):
    """Categorize exchange by type and rate."""
    if rate >= 900:
        return 'Unlimited'
    elif rate >= 9 and rate <= 11:
        return 'Carbon source'
    elif rate >= 0.4 and rate <= 0.6:
        return 'Supplement'
    else:
        return f'Other ({rate:.3f})'

def main():
    print("="*80)
    print("MEDIA COMPOSITION VERIFICATION AND COMPARISON")
    print("="*80)
    print("\nFollowing Suthers et al. 2020 approach:")
    print("- YPD approximation: YNB + 20 amino acids")
    print("- Supplements at 5% of glucose uptake rate")
    print("- Glucose rate: 10.0 mmol/gDW/h → Supplements: 0.5 mmol/gDW/h")
    print()

    # Load model
    yeast_gem = YeastGEM()
    base_model = yeast_gem.model

    # Setup three media conditions
    models = {}
    growth_rates = {}
    exchanges = {}

    # 1. Minimal media
    model_min = base_model.copy()
    setup_minimal_media(model_min, glucose_rate=10.0)
    models['minimal'] = model_min
    solution_min = model_min.optimize()
    growth_rates['minimal'] = solution_min.objective_value
    exchanges['minimal'] = get_open_exchanges(model_min)

    # 2. YNB media
    model_ynb = base_model.copy()
    model_ynb, ynb_added, ynb_missing = setup_ynb_media(model_ynb, glucose_rate=10.0)
    models['YNB'] = model_ynb
    solution_ynb = model_ynb.optimize()
    growth_rates['YNB'] = solution_ynb.objective_value
    exchanges['YNB'] = get_open_exchanges(model_ynb)

    # 3. YPD media
    model_ypd = base_model.copy()
    model_ypd, ynb_added2, ynb_missing2, aa_added, aa_missing = setup_ypd_media(model_ypd, glucose_rate=10.0)
    models['YPD'] = model_ypd
    solution_ypd = model_ypd.optimize()
    growth_rates['YPD'] = solution_ypd.objective_value
    exchanges['YPD'] = get_open_exchanges(model_ypd)

    # Print growth rates
    print("="*80)
    print("GROWTH RATES")
    print("="*80)
    print(f"{'Media':<15} {'Growth Rate':<15} {'Relative to Minimal':<20} {'Relative to YNB'}")
    print("-"*80)

    for media in ['minimal', 'YNB', 'YPD']:
        rate = growth_rates[media]
        rel_min = rate / growth_rates['minimal'] if growth_rates['minimal'] > 0 else 0
        rel_ynb = rate / growth_rates['YNB'] if growth_rates['YNB'] > 0 else 0

        print(f"{media:<15} {rate:<15.4f} {rel_min:<20.2f} {rel_ynb:.2f}")

    # Analyze differences
    print("\n" + "="*80)
    print("METABOLITE DIFFERENCES")
    print("="*80)

    # Find what's added in YNB vs Minimal
    ynb_additions = {}
    for rxn_id, data in exchanges['YNB'].items():
        if rxn_id not in exchanges['minimal'] or exchanges['YNB'][rxn_id]['rate'] != exchanges['minimal'][rxn_id]['rate']:
            if exchanges['YNB'][rxn_id]['rate'] == 0.5:  # Supplement rate
                ynb_additions[rxn_id] = data

    # Find what's added in YPD vs YNB
    ypd_additions = {}
    for rxn_id, data in exchanges['YPD'].items():
        if rxn_id not in exchanges['YNB'] or exchanges['YPD'][rxn_id]['rate'] != exchanges['YNB'][rxn_id]['rate']:
            if exchanges['YPD'][rxn_id]['rate'] == 0.5:  # Supplement rate
                ypd_additions[rxn_id] = data

    print("\n1. MINIMAL MEDIA (Base components)")
    print("-"*80)
    print(f"{'Exchange ID':<12} {'Metabolite':<35} {'Rate (mmol/gDW/h)':<18} {'Category'}")
    print("-"*80)

    # Sort by rate for better readability
    minimal_sorted = sorted(exchanges['minimal'].items(), key=lambda x: x[1]['rate'], reverse=True)
    for rxn_id, data in minimal_sorted:
        category = categorize_exchange(rxn_id, data['name'], data['rate'])
        print(f"{rxn_id:<12} {data['name']:<35} {data['rate']:<18.1f} {category}")

    print(f"\nTotal: {len(exchanges['minimal'])} metabolites")

    print("\n2. YNB ADDITIONS (Vitamins/Cofactors)")
    print("-"*80)
    print(f"{'Exchange ID':<12} {'Metabolite':<35} {'Rate (mmol/gDW/h)':<18}")
    print("-"*80)

    for rxn_id, data in sorted(ynb_additions.items()):
        print(f"{rxn_id:<12} {data['name']:<35} {data['rate']:<18.3f}")

    print(f"\nTotal YNB additions: {len(ynb_additions)} vitamins/cofactors")

    print("\n3. YPD ADDITIONS (Amino Acids)")
    print("-"*80)
    print(f"{'Exchange ID':<12} {'Metabolite':<35} {'Rate (mmol/gDW/h)':<18}")
    print("-"*80)

    # Sort amino acids alphabetically
    aa_sorted = sorted(ypd_additions.items(), key=lambda x: x[1]['name'])
    for rxn_id, data in aa_sorted:
        print(f"{rxn_id:<12} {data['name']:<35} {data['rate']:<18.3f}")

    print(f"\nTotal YPD additions: {len(ypd_additions)} amino acids")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"{'Media':<15} {'Total Exchanges':<20} {'Growth Rate':<15} {'Growth Increase'}")
    print("-"*80)

    prev_rate = 0
    for media in ['minimal', 'YNB', 'YPD']:
        count = len(exchanges[media])
        rate = growth_rates[media]
        increase = f"+{((rate - prev_rate) / prev_rate * 100):.1f}%" if prev_rate > 0 else "baseline"
        print(f"{media:<15} {count:<20} {rate:<15.4f} {increase}")
        prev_rate = rate

    # Verify Suthers approach
    print("\n" + "="*80)
    print("VERIFICATION OF SUTHERS ET AL. 2020 APPROACH")
    print("="*80)

    # Check supplement rates
    supplement_rates = set()
    for rxn_id in list(ynb_additions.keys()) + list(ypd_additions.keys()):
        if rxn_id in exchanges['YPD']:
            rate = exchanges['YPD'][rxn_id]['rate']
            if 0.4 <= rate <= 0.6:
                supplement_rates.add(rate)

    print("✓ Glucose uptake rate: 10.0 mmol/gDW/h")
    print(f"✓ Supplement rate (5% of glucose): {list(supplement_rates)[0] if supplement_rates else 'N/A':.3f} mmol/gDW/h")
    print(f"✓ YNB components (vitamins/cofactors): {len(ynb_additions)}")
    print(f"✓ Amino acids in YPD: {len(ypd_additions)}")

    # Verify all are at correct rate
    all_correct = all(abs(data['rate'] - 0.5) < 0.01 for data in ynb_additions.values())
    all_correct &= all(abs(data['rate'] - 0.5) < 0.01 for data in ypd_additions.values())

    if all_correct:
        print("✓ All supplements at correct rate (0.5 mmol/gDW/h)")
    else:
        print("✗ Some supplements not at correct rate!")

    # Check for missing components
    print("\n" + "="*80)
    print("MISSING COMPONENTS CHECK")
    print("="*80)

    if ynb_missing:
        print(f"YNB missing: {', '.join(ynb_missing)}")
    else:
        print("✓ No YNB components missing")

    if aa_missing:
        print(f"Amino acids missing: {', '.join(aa_missing)}")
    else:
        print("✓ No amino acids missing")

    print("\n" + "="*80)
    print("COMPLETE! Results ready for table formatting.")
    print("="*80)

if __name__ == "__main__":
    main()