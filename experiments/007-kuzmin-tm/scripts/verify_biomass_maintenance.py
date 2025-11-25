#!/usr/bin/env python3
"""
Verify biomass and maintenance parameters in Yeast9 model.
As recommended by Vikas Upadhyay - ensures we're using updated biomass/GAM/NGAM.
"""

import cobra
import pandas as pd
import numpy as np
from cobra.io import load_model
import json
from datetime import datetime
import os

def check_biomass_reactions(model):
    """Check biomass reactions and their coefficients."""

    biomass_reactions = []

    # Look for biomass reactions
    for rxn in model.reactions:
        if 'biomass' in rxn.id.lower() or 'growth' in rxn.id.lower():
            biomass_reactions.append({
                'id': rxn.id,
                'name': rxn.name,
                'lower_bound': rxn.lower_bound,
                'upper_bound': rxn.upper_bound,
                'objective_coefficient': rxn.objective_coefficient,
                'num_metabolites': len(rxn.metabolites),
                'subsystem': rxn.subsystem,
                'is_objective': rxn.objective_coefficient != 0
            })

    return biomass_reactions

def check_maintenance_parameters(model):
    """Check GAM and NGAM parameters."""

    maintenance_info = {
        'GAM': None,
        'NGAM': None,
        'ATP_maintenance_reactions': []
    }

    # Look for ATP maintenance reactions (NGAM)
    for rxn in model.reactions:
        if 'atp' in rxn.id.lower() and 'maintenance' in rxn.id.lower():
            maintenance_info['ATP_maintenance_reactions'].append({
                'id': rxn.id,
                'name': rxn.name,
                'lower_bound': rxn.lower_bound,
                'upper_bound': rxn.upper_bound,
                'reaction': rxn.reaction
            })

        # Common NGAM reaction patterns
        if rxn.id in ['ATPM', 'NGAM', 'r_4041']:  # r_4041 is ATP maintenance in Yeast8
            maintenance_info['NGAM'] = {
                'id': rxn.id,
                'flux': rxn.lower_bound,
                'reaction': rxn.reaction
            }

    # Check GAM in biomass reaction
    biomass_rxns = [r for r in model.reactions if r.objective_coefficient != 0]
    if biomass_rxns:
        biomass_rxn = biomass_rxns[0]
        # Look for ATP coefficient in biomass (GAM)
        for met, coef in biomass_rxn.metabolites.items():
            if 'atp' in met.id.lower() and coef < 0:  # ATP consumed
                maintenance_info['GAM'] = {
                    'biomass_reaction': biomass_rxn.id,
                    'ATP_coefficient': abs(coef),
                    'metabolite_id': met.id
                }
                break

    return maintenance_info

def analyze_biomass_composition(model):
    """Analyze composition of biomass reaction."""

    biomass_rxns = [r for r in model.reactions if r.objective_coefficient != 0]

    if not biomass_rxns:
        return None

    biomass_rxn = biomass_rxns[0]
    composition = {
        'reaction_id': biomass_rxn.id,
        'metabolite_categories': {}
    }

    # Categorize metabolites
    categories = {
        'amino_acids': [],
        'nucleotides': [],
        'lipids': [],
        'carbohydrates': [],
        'cofactors': [],
        'energy': [],
        'other': []
    }

    for met, coef in biomass_rxn.metabolites.items():
        met_id = met.id.lower()
        met_name = met.name.lower()

        # Categorize based on metabolite ID/name
        if any(aa in met_id for aa in ['ala', 'arg', 'asn', 'asp', 'cys', 'gln', 'glu', 'gly',
                                        'his', 'ile', 'leu', 'lys', 'met', 'phe', 'pro', 'ser',
                                        'thr', 'trp', 'tyr', 'val']):
            categories['amino_acids'].append((met.id, coef))
        elif any(nt in met_id for nt in ['atp', 'gtp', 'ctp', 'utp', 'datp', 'dgtp', 'dctp', 'dttp']):
            categories['nucleotides'].append((met.id, coef))
        elif 'lipid' in met_name or 'fatty' in met_name or 'phosphatidyl' in met_name:
            categories['lipids'].append((met.id, coef))
        elif any(carb in met_id for carb in ['glucose', 'glycogen', 'trehalose', 'mannan']):
            categories['carbohydrates'].append((met.id, coef))
        elif any(cof in met_id for cof in ['nad', 'fad', 'coa', 'thf', 'sam']):
            categories['cofactors'].append((met.id, coef))
        elif met_id in ['atp_c', 'h2o_c', 'h_c', 'pi_c', 'adp_c']:
            categories['energy'].append((met.id, coef))
        else:
            categories['other'].append((met.id, coef))

    # Count metabolites in each category
    for cat, mets in categories.items():
        composition['metabolite_categories'][cat] = {
            'count': len(mets),
            'examples': mets[:3] if mets else []  # Show first 3 examples
        }

    composition['total_metabolites'] = len(biomass_rxn.metabolites)

    return composition

def compare_with_yeast8(model):
    """Compare with known Yeast8 parameters."""

    yeast8_params = {
        'NGAM': 1.0,  # mmol ATP/gDW/h
        'GAM': 59.276,  # mmol ATP/gDW (in biomass)
        'biomass_reaction': 'r_2111',  # Yeast8 biomass reaction
        'notes': 'Yeast8 standard parameters from Lu et al., 2019'
    }

    comparison = {
        'yeast8_reference': yeast8_params,
        'current_model': {},
        'differences': []
    }

    # Get current model parameters
    maintenance = check_maintenance_parameters(model)

    if maintenance['NGAM']:
        current_ngam = maintenance['NGAM']['flux']
        comparison['current_model']['NGAM'] = current_ngam
        if abs(current_ngam - yeast8_params['NGAM']) > 0.1:
            comparison['differences'].append(f"NGAM differs: {current_ngam} vs {yeast8_params['NGAM']}")

    if maintenance['GAM']:
        current_gam = maintenance['GAM']['ATP_coefficient']
        comparison['current_model']['GAM'] = current_gam
        if abs(current_gam - yeast8_params['GAM']) > 1:
            comparison['differences'].append(f"GAM differs: {current_gam} vs {yeast8_params['GAM']}")

    return comparison

def main():
    print("=" * 70)
    print("Biomass and Maintenance Parameters Verification")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load Yeast9 model
    print("Loading Yeast9 model...")
    model_path = "/home/michaelvolk/Documents/yeast-GEM/model/yeast-GEM.xml"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please ensure yeast-GEM is available")
        return

    model = load_model(model_path)
    print(f"Model loaded: {model.id}")
    print(f"Version: {model.version if hasattr(model, 'version') else 'Unknown'}\n")

    # Check biomass reactions
    print("1. BIOMASS REACTIONS")
    print("-" * 50)
    biomass_rxns = check_biomass_reactions(model)

    for rxn in biomass_rxns:
        print(f"\nReaction: {rxn['id']}")
        print(f"  Name: {rxn['name']}")
        print(f"  Is objective: {rxn['is_objective']}")
        print(f"  Bounds: [{rxn['lower_bound']}, {rxn['upper_bound']}]")
        print(f"  Metabolites: {rxn['num_metabolites']}")
        if rxn['subsystem']:
            print(f"  Subsystem: {rxn['subsystem']}")

    # Check maintenance parameters
    print("\n2. MAINTENANCE PARAMETERS")
    print("-" * 50)
    maintenance = check_maintenance_parameters(model)

    if maintenance['NGAM']:
        print(f"\nNGAM (Non-growth associated maintenance):")
        print(f"  Reaction: {maintenance['NGAM']['id']}")
        print(f"  Flux: {maintenance['NGAM']['flux']} mmol ATP/gDW/h")
        print(f"  Reaction string: {maintenance['NGAM']['reaction']}")
    else:
        print("\nNGAM: Not found")

    if maintenance['GAM']:
        print(f"\nGAM (Growth associated maintenance):")
        print(f"  In biomass reaction: {maintenance['GAM']['biomass_reaction']}")
        print(f"  ATP coefficient: {maintenance['GAM']['ATP_coefficient']} mmol ATP/gDW")
        print(f"  Metabolite ID: {maintenance['GAM']['metabolite_id']}")
    else:
        print("\nGAM: Not found in biomass reaction")

    # Analyze biomass composition
    print("\n3. BIOMASS COMPOSITION")
    print("-" * 50)
    composition = analyze_biomass_composition(model)

    if composition:
        print(f"\nBiomass reaction: {composition['reaction_id']}")
        print(f"Total metabolites: {composition['total_metabolites']}")
        print("\nMetabolite categories:")
        for cat, info in composition['metabolite_categories'].items():
            if info['count'] > 0:
                print(f"  {cat.replace('_', ' ').title()}: {info['count']} metabolites")
                for met_id, coef in info['examples']:
                    print(f"    - {met_id}: {coef:.4f}")

    # Compare with Yeast8
    print("\n4. COMPARISON WITH YEAST8")
    print("-" * 50)
    comparison = compare_with_yeast8(model)

    print(f"\nYeast8 reference parameters:")
    print(f"  NGAM: {comparison['yeast8_reference']['NGAM']} mmol ATP/gDW/h")
    print(f"  GAM: {comparison['yeast8_reference']['GAM']} mmol ATP/gDW")

    print(f"\nCurrent model parameters:")
    for param, value in comparison['current_model'].items():
        print(f"  {param}: {value}")

    if comparison['differences']:
        print(f"\nDifferences detected:")
        for diff in comparison['differences']:
            print(f"  ⚠ {diff}")
    else:
        print("\n✓ Parameters match Yeast8 reference")

    # Check for potential issues
    print("\n5. POTENTIAL ISSUES")
    print("-" * 50)

    issues = []

    # Check if multiple biomass reactions
    obj_rxns = [r for r in model.reactions if r.objective_coefficient != 0]
    if len(obj_rxns) > 1:
        issues.append(f"Multiple objective reactions found: {[r.id for r in obj_rxns]}")
    elif len(obj_rxns) == 0:
        issues.append("No objective reaction defined")

    # Check if NGAM is set
    if not maintenance['NGAM'] or maintenance['NGAM']['flux'] == 0:
        issues.append("NGAM not set or is zero - may affect baseline ATP requirements")

    # Check biomass bounds
    if biomass_rxns:
        for rxn in biomass_rxns:
            if rxn['is_objective']:
                if rxn['lower_bound'] != 0:
                    issues.append(f"Biomass reaction {rxn['id']} has non-zero lower bound: {rxn['lower_bound']}")
                if rxn['upper_bound'] <= 0:
                    issues.append(f"Biomass reaction {rxn['id']} has non-positive upper bound: {rxn['upper_bound']}")

    if issues:
        print("\nPotential issues detected:")
        for issue in issues:
            print(f"  ⚠ {issue}")
    else:
        print("\n✓ No major issues detected")

    # Save results
    output_dir = "/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/results/cobra-fba-growth"
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'model_id': model.id,
        'timestamp': datetime.now().isoformat(),
        'biomass_reactions': biomass_rxns,
        'maintenance': {
            'NGAM': maintenance['NGAM'],
            'GAM': maintenance['GAM'],
            'ATP_maintenance_reactions': maintenance['ATP_maintenance_reactions']
        },
        'composition': composition,
        'yeast8_comparison': comparison,
        'issues': issues
    }

    output_file = os.path.join(output_dir, "biomass_maintenance_verification.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    print("\n" + "=" * 70)
    print("Verification complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()