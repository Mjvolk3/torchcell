# experiments/008-xue-ffa/scripts/identify_ffa_reactions
# [[experiments.008-xue-ffa.scripts.identify_ffa_reactions]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/identify_ffa_reactions
# Test file: experiments/008-xue-ffa/scripts/test_identify_ffa_reactions.py

import os
import os.path as osp
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
import networkx as nx
from collections import deque

# Load environment
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")

# Results directory
RESULTS_DIR = Path("/Users/michaelvolk/Documents/projects/torchcell/experiments/008-xue-ffa/results")
FFA_REACTIONS_DIR = RESULTS_DIR / "ffa_reactions"
os.makedirs(FFA_REACTIONS_DIR, exist_ok=True)


def find_ffa_metabolites(model):
    """
    Find the specific FFA metabolites we measured: C14:0, C16:0, C16:1, C18:0, C18:1

    Returns dict mapping our FFA names to metabolite IDs in the model
    """
    print("\n" + "="*80)
    print("SEARCHING FOR SPECIFIC FFA METABOLITES")
    print("="*80)

    target_ffas = {
        'C14:0': ['tetradecanoate', 'tetradecanoic', 'myristic', 'myristate', 'C14:0', '14:0'],
        'C16:0': ['hexadecanoate', 'hexadecanoic', 'palmitic', 'palmitate', 'C16:0', '16:0'],
        'C16:1': ['hexadecenoate', 'hexadecenoic', 'palmitoleic', 'palmitoleate', 'C16:1', '16:1'],
        'C18:0': ['octadecanoate', 'octadecanoic', 'stearic', 'stearate', 'C18:0', '18:0'],
        'C18:1': ['octadecenoate', 'octadecenoic', 'oleic', 'oleate', 'C18:1', '18:1'],
    }

    ffa_metabolites = {}

    for ffa_name, keywords in target_ffas.items():
        print(f"\nSearching for {ffa_name}...")
        found = []

        for met in model.metabolites:
            met_id_lower = met.id.lower()
            met_name_lower = met.name.lower() if met.name else ""

            # Check if any keyword matches
            for keyword in keywords:
                if keyword.lower() in met_id_lower or keyword.lower() in met_name_lower:
                    found.append({
                        'id': met.id,
                        'name': met.name,
                        'formula': met.formula,
                        'compartment': met.compartment
                    })
                    print(f"  Found: {met.id} - {met.name} ({met.formula}) [{met.compartment}]")
                    break

        if found:
            ffa_metabolites[ffa_name] = found
        else:
            print(f"  ⚠ WARNING: No metabolites found for {ffa_name}")

    return ffa_metabolites


def get_core_pathway_genes(genome):
    """Get the 14 core FFA pathway genes with systematic IDs."""
    ffa_pathway_genes_standard = {
        'ACC1',  # Acetyl-CoA carboxylase (rate limiting step)
        'FAS1', 'FAS2',  # Fatty acid synthase complex
        'ELO1', 'ELO2', 'ELO3',  # Fatty acid elongation
        'OLE1',  # Fatty acid desaturase (introduces double bonds)
        'FAA1', 'FAA2', 'FAA3', 'FAA4',  # Fatty acyl-CoA synthetases (activate FFAs)
        'POX1',  # Acyl-CoA oxidase (β-oxidation)
        'SLC1',  # 1-acyl-sn-glycerol-3-phosphate acyltransferase
    }

    print("\n" + "="*80)
    print("CONVERTING CORE PATHWAY GENES TO SYSTEMATIC IDS")
    print("="*80)

    gene_mapping = {}
    systematic_ids = set()

    for gene_name in sorted(ffa_pathway_genes_standard):
        if gene_name in genome.alias_to_systematic:
            sys_ids = genome.alias_to_systematic[gene_name]
            if isinstance(sys_ids, list) and len(sys_ids) > 0:
                sys_id = sys_ids[0]
                gene_mapping[gene_name] = sys_id
                systematic_ids.add(sys_id)
                print(f"  {gene_name} → {sys_id}")
            elif isinstance(sys_ids, str):
                gene_mapping[gene_name] = sys_ids
                systematic_ids.add(sys_ids)
                print(f"  {gene_name} → {sys_ids}")
        else:
            print(f"  {gene_name} → NOT FOUND in genome")

    return gene_mapping, systematic_ids


def find_reactions_by_genes(model, gene_ids, gene_mapping):
    """Find all reactions involving the core FFA pathway genes."""
    print("\n" + "="*80)
    print("FINDING REACTIONS WITH CORE FFA GENES")
    print("="*80)

    reactions_data = []

    for reaction in model.reactions:
        reaction_genes = {gene.id for gene in reaction.genes}
        matching_genes = reaction_genes & gene_ids

        if matching_genes:
            # Convert systematic IDs back to standard names
            std_names = []
            for sys_id in sorted(matching_genes):
                for std_name, mapped_id in gene_mapping.items():
                    if mapped_id == sys_id:
                        std_names.append(f"{std_name}({sys_id})")
                        break
                else:
                    std_names.append(sys_id)

            reactions_data.append({
                'reaction_id': reaction.id,
                'equation': reaction.reaction,
                'reversible': reaction.reversibility,
                'subsystem': reaction.subsystem,
                'genes': ', '.join(std_names),
                'gene_ids': ', '.join(sorted(matching_genes)),
                'num_metabolites': len(reaction.metabolites)
            })

    print(f"\nFound {len(reactions_data)} reactions involving core FFA genes")
    return reactions_data


def trace_pathway_from_ffas(model, ffa_metabolites, max_depth=10):
    """
    Trace backwards from FFA metabolites to find biosynthesis pathway.
    Uses BFS to find reactions producing FFAs and their precursors.
    """
    print("\n" + "="*80)
    print("TRACING BIOSYNTHESIS PATHWAY FROM FFAs")
    print("="*80)

    # Collect all FFA metabolite IDs we found
    ffa_met_ids = set()
    for ffa_name, mets in ffa_metabolites.items():
        for met in mets:
            ffa_met_ids.add(met['id'])

    print(f"\nStarting from {len(ffa_met_ids)} FFA metabolites")

    # Build reaction network: metabolite -> reactions that produce it
    met_to_producing_rxns = {}
    for reaction in model.reactions:
        for met, coef in reaction.metabolites.items():
            if coef > 0:  # Product
                if met.id not in met_to_producing_rxns:
                    met_to_producing_rxns[met.id] = []
                met_to_producing_rxns[met.id].append(reaction.id)

    # BFS backwards from FFAs
    visited_mets = set()
    visited_rxns = set()
    queue = deque([(met_id, 0) for met_id in ffa_met_ids])

    for met_id in ffa_met_ids:
        visited_mets.add(met_id)

    while queue:
        met_id, depth = queue.popleft()

        if depth >= max_depth:
            continue

        # Find reactions that produce this metabolite
        if met_id in met_to_producing_rxns:
            for rxn_id in met_to_producing_rxns[met_id]:
                if rxn_id in visited_rxns:
                    continue

                visited_rxns.add(rxn_id)
                rxn = model.reactions.get_by_id(rxn_id)

                # Add all reactants to queue
                for met, coef in rxn.metabolites.items():
                    if coef < 0 and met.id not in visited_mets:  # Reactant
                        visited_mets.add(met.id)
                        queue.append((met.id, depth + 1))

    print(f"Traced pathway: {len(visited_rxns)} reactions, {len(visited_mets)} metabolites")

    # Collect reaction details
    pathway_reactions = []
    for rxn_id in visited_rxns:
        rxn = model.reactions.get_by_id(rxn_id)
        reaction_genes = {gene.id for gene in rxn.genes}

        pathway_reactions.append({
            'reaction_id': rxn.id,
            'equation': rxn.reaction,
            'reversible': rxn.reversibility,
            'subsystem': rxn.subsystem,
            'genes': ', '.join(sorted(reaction_genes)) if reaction_genes else 'No genes',
            'num_metabolites': len(rxn.metabolites),
            'method': 'pathway_trace'
        })

    return pathway_reactions, visited_mets


def identify_ffa_reactions():
    """
    Focused FFA reaction identification:
    1. Find specific FFA metabolites (C14:0, C16:0, C16:1, C18:0, C18:1)
    2. Get reactions with core pathway genes
    3. Trace biosynthesis pathway from FFAs
    4. Compare and validate
    """
    print("="*80)
    print("FOCUSED FFA REACTION IDENTIFICATION")
    print("="*80)

    # Load genome for gene name conversion
    print("\nLoading genome...")
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False
    )

    # Load Yeast GEM
    print("Loading Yeast GEM model...")
    yeast_gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))
    model = yeast_gem.model

    print(f"Total reactions in model: {len(model.reactions)}")
    print(f"Total metabolites in model: {len(model.metabolites)}")
    print(f"Total genes in model: {len(model.genes)}")

    # Step 1: Find FFA metabolites
    ffa_metabolites = find_ffa_metabolites(model)

    # Step 2: Get core pathway genes
    gene_mapping, core_gene_ids = get_core_pathway_genes(genome)

    # Step 3: Find reactions with core genes
    core_gene_reactions = find_reactions_by_genes(model, core_gene_ids, gene_mapping)

    # Step 4: Trace pathway from FFAs
    pathway_reactions, pathway_metabolites = trace_pathway_from_ffas(model, ffa_metabolites, max_depth=15)

    # Compare the two approaches
    print("\n" + "="*80)
    print("COMPARING APPROACHES")
    print("="*80)

    core_rxn_ids = {r['reaction_id'] for r in core_gene_reactions}
    pathway_rxn_ids = {r['reaction_id'] for r in pathway_reactions}

    print(f"\nReactions with core genes: {len(core_rxn_ids)}")
    print(f"Reactions from pathway trace: {len(pathway_rxn_ids)}")
    print(f"Overlap: {len(core_rxn_ids & pathway_rxn_ids)}")
    print(f"Only in core genes: {len(core_rxn_ids - pathway_rxn_ids)}")
    print(f"Only in pathway trace: {len(pathway_rxn_ids - core_rxn_ids)}")

    # Combine both approaches - use core gene reactions as primary
    combined_reactions = core_gene_reactions.copy()

    # Add pathway reactions that involve core genes but were missed
    for rxn in pathway_reactions:
        if rxn['reaction_id'] not in core_rxn_ids:
            # Check if it involves any core genes
            if rxn['genes'] != 'No genes':
                gene_ids = set(rxn['genes'].split(', '))
                if gene_ids & core_gene_ids:
                    combined_reactions.append(rxn)

    # Create dataframes
    reactions_df = pd.DataFrame(combined_reactions)

    # Extract all genes and metabolites from these reactions
    all_genes = set()
    all_metabolites = set()

    for rxn_id in reactions_df['reaction_id']:
        rxn = model.reactions.get_by_id(rxn_id)
        for gene in rxn.genes:
            all_genes.add(gene.id)
        for met in rxn.metabolites:
            all_metabolites.add(met.id)

    # Create genes dataframe
    genes_data = []
    for gene_id in sorted(all_genes):
        if gene_id in model.genes:
            gene = model.genes.get_by_id(gene_id)
            # Find standard name
            std_name = gene_id
            for name, mapped_id in gene_mapping.items():
                if mapped_id == gene_id:
                    std_name = name
                    break

            genes_data.append({
                'gene_id': gene.id,
                'gene_name': gene.name if gene.name else gene.id,
                'standard_name': std_name,
                'is_core_pathway': gene.id in core_gene_ids,
            })
    genes_df = pd.DataFrame(genes_data)

    # Create metabolites dataframe
    metabolites_data = []
    for met_id in sorted(all_metabolites):
        if met_id in model.metabolites:
            met = model.metabolites.get_by_id(met_id)
            # Check if this is one of our target FFAs
            is_target_ffa = False
            for ffa_name, ffa_mets in ffa_metabolites.items():
                if any(m['id'] == met_id for m in ffa_mets):
                    is_target_ffa = True
                    break

            metabolites_data.append({
                'metabolite_id': met.id,
                'metabolite_name': met.name,
                'formula': met.formula,
                'compartment': met.compartment,
                'charge': met.charge,
                'is_target_ffa': is_target_ffa,
            })
    metabolites_df = pd.DataFrame(metabolites_data)

    # Print summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\nFocused FFA reactions identified: {len(reactions_df)}")
    pattern = r'\(Y'
    print(f"  With core pathway genes: {sum(reactions_df['genes'].str.contains(pattern, na=False))}")
    print(f"\nGenes involved: {len(genes_df)}")
    print(f"  Core pathway genes: {sum(genes_df['is_core_pathway'])}")
    print(f"  Associated genes: {sum(~genes_df['is_core_pathway'])}")
    print(f"\nMetabolites involved: {len(metabolites_df)}")
    print(f"  Target FFAs: {sum(metabolites_df['is_target_ffa'])}")

    # Print core pathway genes found
    print("\n" + "="*80)
    print("CORE PATHWAY GENES IN REACTIONS")
    print("="*80)
    core_genes_df = genes_df[genes_df['is_core_pathway']]
    for _, row in core_genes_df.iterrows():
        print(f"  {row['standard_name']}: {row['gene_id']} - {row['gene_name']}")

    # Print sample reactions
    print("\n" + "="*80)
    print("SAMPLE FFA REACTIONS (first 10)")
    print("="*80)
    for i, row in reactions_df.head(10).iterrows():
        print(f"\n{i+1}. {row['reaction_id']}")
        print(f"   Equation: {row['equation']}")
        print(f"   Subsystem: {row['subsystem']}")
        print(f"   Genes: {row['genes']}")

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    reactions_path = FFA_REACTIONS_DIR / "ffa_reactions_list.csv"
    reactions_df.to_csv(reactions_path, index=False)
    print(f"Saved reactions to: {reactions_path}")

    genes_path = FFA_REACTIONS_DIR / "ffa_genes_list.csv"
    genes_df.to_csv(genes_path, index=False)
    print(f"Saved genes to: {genes_path}")

    metabolites_path = FFA_REACTIONS_DIR / "ffa_metabolites_list.csv"
    metabolites_df.to_csv(metabolites_path, index=False)
    print(f"Saved metabolites to: {metabolites_path}")

    # Also save FFA metabolite mapping
    ffa_mapping = []
    for ffa_name, mets in ffa_metabolites.items():
        for met in mets:
            ffa_mapping.append({
                'ffa_type': ffa_name,
                'metabolite_id': met['id'],
                'metabolite_name': met['name'],
                'formula': met['formula'],
                'compartment': met['compartment']
            })
    ffa_mapping_df = pd.DataFrame(ffa_mapping)
    ffa_mapping_path = FFA_REACTIONS_DIR / "ffa_metabolite_mapping.csv"
    ffa_mapping_df.to_csv(ffa_mapping_path, index=False)
    print(f"Saved FFA metabolite mapping to: {ffa_mapping_path}")

    print("\n" + "="*80)
    print("FFA REACTION IDENTIFICATION COMPLETE!")
    print("="*80)

    return reactions_df, genes_df, metabolites_df, ffa_metabolites


if __name__ == "__main__":
    reactions_df, genes_df, metabolites_df, ffa_metabolites = identify_ffa_reactions()
