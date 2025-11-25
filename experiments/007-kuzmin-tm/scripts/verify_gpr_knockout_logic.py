#!/usr/bin/env python3
"""
Verify GPR knockout logic for complexes (AND rules) vs isoenzymes (OR rules).
As recommended by Vikas Upadhyay - ensures we're correctly handling gene-reaction rules.
"""

import cobra
import pandas as pd
import numpy as np
from cobra.io import load_model
import json
from datetime import datetime
import os

def test_gpr_logic(model):
    """Test specific examples of GPR rules to verify knockout logic."""

    test_cases = []

    # Find reactions with complex GPR rules
    for rxn in model.reactions:
        if rxn.gene_reaction_rule:
            gpr = rxn.gene_reaction_rule

            # Test case 1: Complex (AND rule) - all genes needed
            if ' and ' in gpr and ' or ' not in gpr:
                genes = [g.id for g in rxn.genes]
                if len(genes) >= 2:
                    test_cases.append({
                        'type': 'complex',
                        'reaction_id': rxn.id,
                        'gpr': gpr,
                        'genes': genes,
                        'expected': 'Should be disabled if ANY gene is knocked out'
                    })

            # Test case 2: Isoenzyme (OR rule) - alternative genes
            elif ' or ' in gpr and ' and ' not in gpr:
                genes = [g.id for g in rxn.genes]
                if len(genes) >= 2:
                    test_cases.append({
                        'type': 'isoenzyme',
                        'reaction_id': rxn.id,
                        'gpr': gpr,
                        'genes': genes,
                        'expected': 'Should be disabled only if ALL genes are knocked out'
                    })

            # Test case 3: Mixed (both AND and OR)
            elif ' and ' in gpr and ' or ' in gpr:
                genes = [g.id for g in rxn.genes]
                if len(genes) >= 3:
                    test_cases.append({
                        'type': 'mixed',
                        'reaction_id': rxn.id,
                        'gpr': gpr[:100] + '...' if len(gpr) > 100 else gpr,
                        'genes': genes[:5],  # Just show first 5 genes
                        'expected': 'Complex logic - needs careful handling'
                    })

    return test_cases[:20]  # Return first 20 test cases

def verify_knockout_behavior(model, test_case):
    """Verify a specific test case behaves correctly."""

    results = {}
    rxn_id = test_case['reaction_id']
    genes = test_case['genes']

    # Get wild-type flux bounds
    wt_rxn = model.reactions.get_by_id(rxn_id)
    wt_lower = wt_rxn.lower_bound
    wt_upper = wt_rxn.upper_bound
    results['wt_bounds'] = (wt_lower, wt_upper)

    if test_case['type'] == 'complex':
        # Test single gene knockout for complex
        if genes:
            with model as m:
                gene = m.genes.get_by_id(genes[0])
                gene.knock_out()
                ko_rxn = m.reactions.get_by_id(rxn_id)
                results['single_ko_bounds'] = (ko_rxn.lower_bound, ko_rxn.upper_bound)
                results['single_ko_disabled'] = (ko_rxn.lower_bound == 0 and ko_rxn.upper_bound == 0)

    elif test_case['type'] == 'isoenzyme':
        # Test single gene knockout for isoenzyme
        if genes:
            with model as m:
                gene = m.genes.get_by_id(genes[0])
                gene.knock_out()
                ko_rxn = m.reactions.get_by_id(rxn_id)
                results['single_ko_bounds'] = (ko_rxn.lower_bound, ko_rxn.upper_bound)
                results['single_ko_disabled'] = (ko_rxn.lower_bound == 0 and ko_rxn.upper_bound == 0)

            # Test all genes knockout
            if len(genes) >= 2:
                with model as m:
                    for gene_id in genes[:2]:  # Test with first 2 genes
                        if gene_id in m.genes:
                            gene = m.genes.get_by_id(gene_id)
                            gene.knock_out()
                    ko_rxn = m.reactions.get_by_id(rxn_id)
                    results['multi_ko_bounds'] = (ko_rxn.lower_bound, ko_rxn.upper_bound)
                    results['multi_ko_disabled'] = (ko_rxn.lower_bound == 0 and ko_rxn.upper_bound == 0)

    return results

def main():
    print("=" * 70)
    print("GPR Knockout Logic Verification")
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
    print(f"Reactions: {len(model.reactions)}")
    print(f"Genes: {len(model.genes)}\n")

    # Get test cases
    print("Identifying test cases...")
    test_cases = test_gpr_logic(model)

    # Verify each test case
    print("\nVerifying knockout behavior:\n")
    print("-" * 70)

    verification_results = []

    for i, test_case in enumerate(test_cases[:10], 1):  # Test first 10
        print(f"\nTest Case {i}:")
        print(f"  Type: {test_case['type'].upper()}")
        print(f"  Reaction: {test_case['reaction_id']}")
        print(f"  GPR: {test_case['gpr'][:80]}..." if len(test_case['gpr']) > 80 else f"  GPR: {test_case['gpr']}")
        print(f"  Genes involved: {len(test_case['genes'])}")
        print(f"  Expected: {test_case['expected']}")

        # Verify behavior
        results = verify_knockout_behavior(model, test_case)
        test_case['verification'] = results

        if test_case['type'] == 'complex':
            if 'single_ko_disabled' in results:
                status = "PASS" if results['single_ko_disabled'] else "FAIL"
                print(f"  Single KO disables reaction: {results['single_ko_disabled']} [{status}]")

        elif test_case['type'] == 'isoenzyme':
            if 'single_ko_disabled' in results:
                status = "PASS" if not results['single_ko_disabled'] else "FAIL"
                print(f"  Single KO preserves function: {not results['single_ko_disabled']} [{status}]")
            if 'multi_ko_disabled' in results:
                print(f"  Multi KO disables reaction: {results['multi_ko_disabled']}")

        verification_results.append(test_case)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    # Count reaction types
    complex_count = 0
    isoenzyme_count = 0
    mixed_count = 0

    for rxn in model.reactions:
        if rxn.gene_reaction_rule:
            gpr = rxn.gene_reaction_rule
            if ' and ' in gpr and ' or ' not in gpr:
                complex_count += 1
            elif ' or ' in gpr and ' and ' not in gpr:
                isoenzyme_count += 1
            elif ' and ' in gpr and ' or ' in gpr:
                mixed_count += 1

    print(f"\nReaction GPR types in model:")
    print(f"  Complexes (AND only): {complex_count}")
    print(f"  Isoenzymes (OR only): {isoenzyme_count}")
    print(f"  Mixed (AND + OR): {mixed_count}")
    print(f"  Total with GPR: {complex_count + isoenzyme_count + mixed_count}")

    # Save results
    output_dir = "/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/results/cobra-fba-growth"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "gpr_verification_results.json")
    with open(output_file, 'w') as f:
        # Convert to serializable format
        for result in verification_results:
            if 'verification' in result:
                for key, val in result['verification'].items():
                    if isinstance(val, tuple):
                        result['verification'][key] = list(val)
        json.dump(verification_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Check for potential issues
    print("\n" + "=" * 70)
    print("POTENTIAL ISSUES CHECK")
    print("=" * 70)

    # Look for reactions with many isoenzymes
    high_isoenzyme_rxns = []
    for rxn in model.reactions:
        if rxn.gene_reaction_rule and ' or ' in rxn.gene_reaction_rule:
            gene_count = len(rxn.genes)
            if gene_count > 5:
                high_isoenzyme_rxns.append((rxn.id, gene_count))

    if high_isoenzyme_rxns:
        print("\nReactions with many isoenzymes (>5 genes):")
        for rxn_id, count in sorted(high_isoenzyme_rxns, key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {rxn_id}: {count} genes")

    print("\n" + "=" * 70)
    print("Verification complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()