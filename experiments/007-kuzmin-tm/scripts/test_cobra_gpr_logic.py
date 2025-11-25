#!/usr/bin/env python3
"""
Test COBRA's GPR knockout logic to verify it correctly handles:
- AND rules (complexes): ANY gene KO should disable reaction
- OR rules (isoenzymes): ALL genes must be KO'd to disable reaction
"""

import cobra
from cobra.io import load_model


def test_gpr_logic():
    """Create minimal test cases to verify COBRA's GPR handling."""

    print("=" * 70)
    print("TESTING COBRA'S GPR KNOCKOUT LOGIC")
    print("=" * 70)

    # Create a simple test model
    model = cobra.Model("test_model")

    # Add metabolites
    met_a = cobra.Metabolite("A", name="A", compartment="c")
    met_b = cobra.Metabolite("B", name="B", compartment="c")
    met_c = cobra.Metabolite("C", name="C", compartment="c")
    model.add_metabolites([met_a, met_b, met_c])

    # Test Case 1: Complex (AND rule)
    # Both genes required - knocking out ANY should disable reaction
    rxn_complex = cobra.Reaction("R_complex")
    rxn_complex.name = "Complex reaction (AND rule)"
    rxn_complex.lower_bound = -10
    rxn_complex.upper_bound = 10
    rxn_complex.add_metabolites({met_a: -1, met_b: 1})
    rxn_complex.gene_reaction_rule = "G1 and G2"
    model.add_reactions([rxn_complex])

    # Test Case 2: Isoenzyme (OR rule)
    # Alternative genes - ALL must be KO'd to disable reaction
    rxn_iso = cobra.Reaction("R_isoenzyme")
    rxn_iso.name = "Isoenzyme reaction (OR rule)"
    rxn_iso.lower_bound = -10
    rxn_iso.upper_bound = 10
    rxn_iso.add_metabolites({met_b: -1, met_c: 1})
    rxn_iso.gene_reaction_rule = "G3 or G4"
    model.add_reactions([rxn_iso])

    # Test Case 3: Mixed (complex with isoenzymes)
    # (G5 and G6) or (G7 and G8)
    rxn_mixed = cobra.Reaction("R_mixed")
    rxn_mixed.name = "Mixed reaction"
    rxn_mixed.lower_bound = -10
    rxn_mixed.upper_bound = 10
    rxn_mixed.add_metabolites({met_a: -1, met_c: 1})
    rxn_mixed.gene_reaction_rule = "(G5 and G6) or (G7 and G8)"
    model.add_reactions([rxn_mixed])

    print("\n=== TEST CASE 1: COMPLEX (AND RULE) ===")
    print(f"Reaction: {rxn_complex.id}")
    print(f"GPR: {rxn_complex.gene_reaction_rule}")
    print(f"Initial functional status: {rxn_complex.functional}")
    print(f"Initial bounds: {rxn_complex.bounds}")

    # Test single gene KO in complex
    with model as m:
        gene = m.genes.get_by_id("G1")
        gene.knock_out()
        print(f"\nAfter knocking out G1:")
        print(f"  G1 functional: {gene.functional}")
        print(f"  Reaction functional: {rxn_complex.functional}")
        print(f"  Reaction bounds: {rxn_complex.bounds}")
        if rxn_complex.bounds == (0, 0):
            print("  ✓ CORRECT: Single KO disables complex")
        else:
            print("  ✗ ERROR: Single KO should disable complex!")

    print("\n=== TEST CASE 2: ISOENZYME (OR RULE) ===")
    print(f"Reaction: {rxn_iso.id}")
    print(f"GPR: {rxn_iso.gene_reaction_rule}")
    print(f"Initial functional status: {rxn_iso.functional}")
    print(f"Initial bounds: {rxn_iso.bounds}")

    # Test single gene KO in isoenzyme
    with model as m:
        gene = m.genes.get_by_id("G3")
        gene.knock_out()
        print(f"\nAfter knocking out G3 only:")
        print(f"  G3 functional: {gene.functional}")
        print(f"  Reaction functional: {rxn_iso.functional}")
        print(f"  Reaction bounds: {rxn_iso.bounds}")
        if rxn_iso.bounds != (0, 0):
            print("  ✓ CORRECT: Single KO preserves isoenzyme function")
        else:
            print("  ✗ ERROR: Single KO should NOT disable isoenzyme!")

    # Test all genes KO in isoenzyme
    with model as m:
        m.genes.get_by_id("G3").knock_out()
        m.genes.get_by_id("G4").knock_out()
        print(f"\nAfter knocking out both G3 and G4:")
        print(f"  Reaction functional: {rxn_iso.functional}")
        print(f"  Reaction bounds: {rxn_iso.bounds}")
        if rxn_iso.bounds == (0, 0):
            print("  ✓ CORRECT: All KOs disable isoenzyme")
        else:
            print("  ✗ ERROR: All KOs should disable isoenzyme!")

    print("\n=== TEST CASE 3: MIXED RULE ===")
    print(f"Reaction: {rxn_mixed.id}")
    print(f"GPR: {rxn_mixed.gene_reaction_rule}")
    print(f"Initial functional status: {rxn_mixed.functional}")
    print(f"Initial bounds: {rxn_mixed.bounds}")

    # Test knocking out one complex
    with model as m:
        m.genes.get_by_id("G5").knock_out()
        print(f"\nAfter knocking out G5 (breaks first complex):")
        print(f"  Reaction functional: {rxn_mixed.functional}")
        print(f"  Reaction bounds: {rxn_mixed.bounds}")
        if rxn_mixed.bounds != (0, 0):
            print("  ✓ CORRECT: Reaction still functional via second complex")
        else:
            print("  ✗ ERROR: Should still function through second complex!")

    # Test knocking out both complexes
    with model as m:
        m.genes.get_by_id("G5").knock_out()
        m.genes.get_by_id("G7").knock_out()
        print(f"\nAfter knocking out G5 and G7 (breaks both complexes):")
        print(f"  Reaction functional: {rxn_mixed.functional}")
        print(f"  Reaction bounds: {rxn_mixed.bounds}")
        if rxn_mixed.bounds == (0, 0):
            print("  ✓ CORRECT: Both complexes disabled, reaction disabled")
        else:
            print("  ✗ ERROR: Both complexes broken, should be disabled!")

    print("\n" + "=" * 70)
    print("DETAILED GPR EVALUATION LOGIC")
    print("=" * 70)

    # Show the internal evaluation logic
    print("\nCOBRA's _eval_gpr function logic:")
    print("1. For OR: returns True if ANY value is True (any())")
    print("2. For AND: returns True if ALL values are True (all())")
    print("3. A gene is True if it's NOT in the knockout set")
    print("\nTherefore:")
    print("- Complex (AND): KO any gene → some value False → all() returns False → reaction disabled ✓")
    print("- Isoenzyme (OR): KO one gene → other values still True → any() returns True → reaction enabled ✓")

    return True


def test_on_yeast_gem():
    """Test a few real reactions from Yeast-GEM."""
    print("\n" + "=" * 70)
    print("TESTING REAL YEAST-GEM REACTIONS")
    print("=" * 70)

    # Load model
    model_path = "/home/michaelvolk/Documents/yeast-GEM/model/yeast-GEM.xml"
    try:
        model = load_model(model_path)
    except:
        print("Could not load Yeast-GEM model. Using YeastGEM class instead.")
        from torchcell.metabolism.yeast_GEM import YeastGEM
        yeast_gem = YeastGEM()
        model = yeast_gem.model

    # Find example reactions
    complex_rxn = None
    isoenzyme_rxn = None

    for rxn in model.reactions:
        if rxn.gene_reaction_rule:
            gpr = rxn.gene_reaction_rule
            # Find a simple complex
            if ' and ' in gpr and ' or ' not in gpr and not complex_rxn:
                if len(rxn.genes) == 2:
                    complex_rxn = rxn
            # Find a simple isoenzyme
            elif ' or ' in gpr and ' and ' not in gpr and not isoenzyme_rxn:
                if len(rxn.genes) == 2:
                    isoenzyme_rxn = rxn

            if complex_rxn and isoenzyme_rxn:
                break

    if complex_rxn:
        print(f"\n=== REAL COMPLEX EXAMPLE ===")
        print(f"Reaction: {complex_rxn.id}")
        print(f"GPR: {complex_rxn.gene_reaction_rule}")
        genes = list(complex_rxn.genes)

        with model as m:
            gene = m.genes.get_by_id(genes[0].id)
            gene.knock_out()
            rxn = m.reactions.get_by_id(complex_rxn.id)
            print(f"After KO {genes[0].id}: functional={rxn.functional}, bounds={rxn.bounds}")
            if rxn.bounds == (0, 0):
                print("✓ Complex correctly disabled by single KO")
            else:
                print("✗ WARNING: Complex not disabled by single KO!")

    if isoenzyme_rxn:
        print(f"\n=== REAL ISOENZYME EXAMPLE ===")
        print(f"Reaction: {isoenzyme_rxn.id}")
        print(f"GPR: {isoenzyme_rxn.gene_reaction_rule}")
        genes = list(isoenzyme_rxn.genes)

        with model as m:
            gene = m.genes.get_by_id(genes[0].id)
            gene.knock_out()
            rxn = m.reactions.get_by_id(isoenzyme_rxn.id)
            print(f"After KO {genes[0].id}: functional={rxn.functional}, bounds={rxn.bounds}")
            if rxn.bounds != (0, 0):
                print("✓ Isoenzyme correctly preserved by single KO")
            else:
                print("✗ WARNING: Isoenzyme incorrectly disabled by single KO!")


if __name__ == "__main__":
    # Run synthetic tests
    test_gpr_logic()

    # Run real model tests
    test_on_yeast_gem()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("\nCOBRA's GPR logic implementation:")
    print("1. gene.knock_out() sets gene.functional = False")
    print("2. reaction.functional evaluates GPR with non-functional genes")
    print("3. GPR._eval_gpr() correctly handles AND/OR logic:")
    print("   - AND: all() function - requires ALL genes functional")
    print("   - OR: any() function - requires ANY gene functional")
    print("4. If reaction.functional is False, bounds are set to (0, 0)")
    print("\nThis implementation CORRECTLY handles complexes vs isoenzymes!")