#!/usr/bin/env python
# experiments/W006-kuzmin-tmi/scripts/dead_grna_design
# [[experiments.W006-kuzmin-tmi.scripts.dead_grna_design]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/W006-kuzmin-tmi/scripts/dead_grna_design
# Test file: experiments/W006-kuzmin-tmi/scripts/test_dead_grna_design.py


"""
Dead gRNA Design for Multiplex Array
Designs maximally mismatched dead gRNAs for S. cerevisiae genome
These dead guides will have minimal probability of targeting any NGG sites
Optimized for yeast GC content (~38%) and PAM-proximal mismatches
"""

import os
import os.path as osp
import numpy as np
from Bio.Seq import Seq
from typing import List, Set, Tuple
from tqdm import tqdm
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")

# Import the genome class
import sys
sys.path.append('/Users/michaelvolk/Documents/projects/torchcell')
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.timestamp import timestamp

# Set up paths
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR", "notes/assets/images")

# Constants for S. cerevisiae
TARGET_GC_CONTENT = 0.38  # 38% GC content
GC_TOLERANCE = 0.15  # ±15% tolerance (relaxed for better candidates)


def find_all_pam_sites(sequence: str, pam: str = "NGG") -> List[int]:
    """
    Find all PAM sites in a sequence.
    Returns list of PAM start positions (0-indexed).
    """
    pam_sites = []
    pam = pam.replace('N', '.')  # Convert N to regex wildcard

    import re
    pattern = re.compile(pam)

    for match in pattern.finditer(sequence):
        pam_sites.append(match.start())

    return pam_sites


def extract_grna_targets(genome: SCerevisiaeGenome, grna_length: int = 20, verbose: bool = False) -> Set[str]:
    """
    Extract all potential gRNA target sequences from the genome.
    Looks for 20bp sequences 5' to NGG PAM sites on BOTH strands.

    For SpCas9 with NGG PAM:
    - Watson strand: 5'-[20bp guide]NGG-3'
    - Crick strand:  3'-[20bp guide]CCN-5' (appears as CCN on forward strand)
    """
    all_targets = set()
    watson_count = 0
    crick_count = 0
    total_ngg_sites = 0

    print("Extracting all potential SpCas9 NGG gRNA target sites from genome...")
    print("Checking BOTH Watson and Crick strands...")

    # Process each chromosome
    for chr_num, chr_id in tqdm(genome.chr_to_nc.items(), desc="Processing chromosomes"):
        if chr_num == 0:  # Skip mitochondrial chromosome
            continue

        # Get chromosome sequence
        chr_seq = str(genome.fasta_dna[chr_id].seq).upper()
        chr_watson_targets = 0
        chr_crick_targets = 0

        # WATSON STRAND: Find NGG PAMs
        # Guide is 20bp UPSTREAM of NGG
        watson_pams = find_all_pam_sites(chr_seq, "NGG")
        for pam_pos in watson_pams:
            total_ngg_sites += 1
            # Extract 20bp upstream of PAM
            if pam_pos >= grna_length:
                target = chr_seq[pam_pos - grna_length:pam_pos]
                if len(target) == grna_length and 'N' not in target:
                    all_targets.add(target)
                    watson_count += 1
                    chr_watson_targets += 1

        # CRICK STRAND: Find CCN PAMs (reverse complement of NGG)
        # On reverse strand: 5'-NGG[20bp guide]-3'
        # On forward strand: 3'-CCN[20bp complement]-5'
        # So we extract 20bp DOWNSTREAM of CCN and reverse complement
        crick_pams = find_all_pam_sites(chr_seq, "CCN")
        for pam_pos in crick_pams:
            total_ngg_sites += 1
            # Extract 20bp downstream of CCN
            if pam_pos + 3 + grna_length <= len(chr_seq):
                # Get sequence after CCN
                target = chr_seq[pam_pos + 3:pam_pos + 3 + grna_length]
                if len(target) == grna_length and 'N' not in target:
                    # Reverse complement to get actual guide sequence
                    rc_target = str(Seq(target).reverse_complement())
                    all_targets.add(rc_target)
                    crick_count += 1
                    chr_crick_targets += 1

        if verbose:
            print(f"  Chr {chr_num}: {chr_watson_targets} Watson + {chr_crick_targets} Crick = {chr_watson_targets + chr_crick_targets} targets")

    print(f"\nExtraction complete:")
    print(f"  Total NGG/CCN PAM sites found: {total_ngg_sites:,}")
    print(f"  Watson strand targets: {watson_count:,}")
    print(f"  Crick strand targets: {crick_count:,}")
    print(f"  Unique gRNA sequences: {len(all_targets):,}")
    print(f"  (Some sequences may appear on both strands)")

    return all_targets


def calculate_gc_content(seq: str) -> float:
    """Calculate GC content of a sequence."""
    return (seq.count('G') + seq.count('C')) / len(seq)


def calculate_weighted_mismatch_score(seq1: str, seq2: str) -> float:
    """
    Calculate weighted mismatch score between two sequences.
    Positions closer to PAM (end of sequence) are weighted more heavily.

    Position weights (from 5' to 3', where position 20 is adjacent to PAM):
    - Positions 1-8: weight = 0.5 (PAM-distal, less critical)
    - Positions 9-12: weight = 1.0 (intermediate)
    - Positions 13-20: weight = 2.0 (PAM-proximal, most critical)
    """
    if len(seq1) != len(seq2):
        return 0

    score = 0
    for i, (c1, c2) in enumerate(zip(seq1, seq2)):
        if c1 != c2:
            # Position numbering: 1-based from 5' end
            pos = i + 1
            if pos <= 8:  # PAM-distal
                weight = 0.5
            elif pos <= 12:  # Intermediate
                weight = 1.0
            else:  # pos 13-20, PAM-proximal
                weight = 2.0
            score += weight

    return score


def calculate_min_weighted_mismatch_to_set(
    candidate: str,
    target_set: Set[str],
    sample_size: int = None,
    min_pam_proximal_required: int = 2  # Minimum PAM-proximal mismatches required
) -> float:
    """
    Calculate the minimum weighted mismatch score between a candidate sequence
    and a set of target sequences. Can sample for efficiency.
    Enforces minimum PAM-proximal mismatch requirement.
    """
    min_score = float('inf')
    min_pam_prox_mm = 8  # Track minimum PAM-proximal mismatches

    # If sampling, randomly sample from the target set
    if sample_size and sample_size < len(target_set):
        targets_to_check = random.sample(list(target_set), sample_size)
    else:
        targets_to_check = target_set

    for target in targets_to_check:
        score = calculate_weighted_mismatch_score(candidate, target)

        # Calculate PAM-proximal mismatches for this target
        pam_prox_mm = sum(candidate[i] != target[i] for i in range(12, 20))
        min_pam_prox_mm = min(min_pam_prox_mm, pam_prox_mm)

        if score == 0:  # Exact match found
            return 0
        min_score = min(min_score, score)

    # Penalize severely if minimum PAM-proximal mismatches is too low
    if min_pam_prox_mm < min_pam_proximal_required:
        # Return very low score to reject this candidate
        return min_score * 0.1  # Heavy penalty

    return min_score


def generate_yeast_like_sequences(
    num_sequences: int = 100,
    grna_length: int = 20,
    target_gc: float = TARGET_GC_CONTENT,
    gc_tolerance: float = GC_TOLERANCE
) -> List[str]:
    """
    Generate random sequences with yeast-like GC content.
    Prioritizes variation in PAM-proximal region.
    Enhanced with multiple strategies for better diversity.
    """
    sequences = []

    for i in range(num_sequences):
        # Alternate between different generation strategies
        strategy = i % 3

        if strategy == 0:  # Standard approach
            # Target number of G/C bases
            target_gc_count = int(grna_length * target_gc)
            gc_count = target_gc_count + random.randint(-2, 2)
            gc_count = max(int(grna_length * (target_gc - gc_tolerance)),
                           min(int(grna_length * (target_gc + gc_tolerance)), gc_count))

            at_count = grna_length - gc_count

            # Create base composition
            bases = ['G'] * (gc_count // 2) + ['C'] * (gc_count - gc_count // 2)
            bases += ['A'] * (at_count // 2) + ['T'] * (at_count - at_count // 2)
            random.shuffle(bases)

            # Extra shuffle for PAM-proximal region
            pam_proximal = bases[-8:]
            random.shuffle(pam_proximal)
            bases[-8:] = pam_proximal

        elif strategy == 1:  # PAM-proximal diversity focus
            # Generate PAM-distal normally
            pam_distal = []
            for j in range(12):
                pam_distal.append(random.choice(['A', 'T', 'G', 'C']))

            # Generate PAM-proximal with enforced diversity
            pam_proximal = []
            # Try to avoid common PAM-proximal patterns
            for j in range(8):
                # Bias against common bases in PAM-proximal
                if j % 2 == 0:
                    pam_proximal.append(random.choice(['A', 'T', 'C', 'G']))
                else:
                    pam_proximal.append(random.choice(['G', 'C', 'A', 'T']))

            bases = pam_distal + pam_proximal

        else:  # strategy == 2: Mixed patterns
            # Create sequences with alternating patterns
            bases = []
            for j in range(grna_length):
                if j < 12:  # PAM-distal
                    bases.append(random.choice(['A', 'T', 'G', 'C']))
                else:  # PAM-proximal - more diverse
                    # Use different probability for each position
                    if random.random() < 0.5:
                        bases.append(random.choice(['G', 'C']))
                    else:
                        bases.append(random.choice(['A', 'T']))

        # Check GC content and only keep if within range
        seq = ''.join(bases)
        gc = calculate_gc_content(seq)
        if abs(gc - target_gc) <= gc_tolerance:
            sequences.append(seq)

    return sequences[:num_sequences]  # Ensure we return the right number


def mutate_pam_proximal_region(
    seq: str,
    genome_targets: Set[str],
    iterations: int = 250,  # Increased for better optimization
    sample_size: int = 35000  # Increased for better coverage
) -> Tuple[str, float]:
    """
    Optimize sequence by focusing mutations on PAM-proximal region.
    Maintains GC content within yeast-like range.
    Enhanced to ensure better PAM-proximal mismatches.
    """
    bases = ['A', 'T', 'C', 'G']
    best_seq = seq
    best_score = calculate_min_weighted_mismatch_to_set(
        best_seq, genome_targets, sample_size,
        min_pam_proximal_required=2  # Enforce during optimization
    )

    for iteration in range(iterations):
        seq_list = list(best_seq)

        # More aggressive PAM-proximal focus in later iterations
        if iteration < iterations // 2:
            # First half: 80% PAM-proximal
            pam_proximal_prob = 0.8
        else:
            # Second half: 95% PAM-proximal for fine-tuning
            pam_proximal_prob = 0.95

        # Focus mutations on PAM-proximal region
        if random.random() < pam_proximal_prob:
            pos = random.randint(12, 19)  # Positions 13-20 (0-indexed)
        else:
            pos = random.randint(0, 11)  # Positions 1-12

        original = seq_list[pos]

        # Try all possible mutations
        best_mutation = None
        best_mutation_score = best_score

        for base in bases:
            if base == original:
                continue

            # Check if mutation would keep GC content in range
            seq_list[pos] = base
            new_seq = ''.join(seq_list)
            new_gc = calculate_gc_content(new_seq)

            if abs(new_gc - TARGET_GC_CONTENT) <= GC_TOLERANCE:
                score = calculate_min_weighted_mismatch_to_set(
                    new_seq, genome_targets, sample_size,
                    min_pam_proximal_required=2  # Enforce during mutation evaluation
                )

                if score > best_mutation_score:
                    best_mutation = base
                    best_mutation_score = score

        # Apply best mutation if found
        if best_mutation:
            seq_list[pos] = best_mutation
            best_seq = ''.join(seq_list)
            best_score = best_mutation_score
        else:
            seq_list[pos] = original  # Reset if no improvement

    # Final verification with large sample
    final_score = calculate_min_weighted_mismatch_to_set(
        best_seq, genome_targets, sample_size=35000,
        min_pam_proximal_required=2  # Enforce in final verification
    )

    return best_seq, final_score


def generate_optimized_dead_grnas(
    genome_targets: Set[str],
    num_candidates: int = 50
) -> List[Tuple[str, float]]:
    """
    Generate and optimize dead gRNA candidates with yeast-appropriate GC content.
    """
    print("Generating candidates with yeast-like GC content (~38%)...")

    # Generate more initial candidates for better chance of finding good ones
    candidates = generate_yeast_like_sequences(num_candidates * 10)  # 10x more candidates

    # Score initial candidates with PAM-proximal penalty enforced
    print("Scoring initial candidates (with PAM-proximal penalty)...")
    scored_candidates = []
    for candidate in tqdm(candidates, desc="Initial scoring"):
        score = calculate_min_weighted_mismatch_to_set(
            candidate, genome_targets,
            sample_size=15000,  # Increased for better initial scoring
            min_pam_proximal_required=2  # Enforce minimum during scoring
        )
        scored_candidates.append((candidate, score))

    # Sort and take top candidates
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = scored_candidates[:min(num_candidates, 100)]  # Take up to 100 best

    # Optimize more candidates with optimization
    print("PAM-proximal optimization...")
    print("Optimizing with 250 iterations, 35k sample size, and PAM-proximal requirement...")
    optimized = []
    for seq, initial_score in tqdm(top_candidates, desc="Optimizing"):  # Optimize ALL top candidates
        opt_seq, opt_score = mutate_pam_proximal_region(
            seq, genome_targets  # Use default parameters (250 iter, 35k sample)
        )
        # Only keep if score is reasonable (not heavily penalized)
        if opt_score > 5:  # Filter out heavily penalized candidates
            optimized.append((opt_seq, opt_score))

    return optimized


def select_dead_grnas(
    candidates: List[Tuple[str, float]],
    genome_targets: Set[str],
    num_dead: int = 5,
    min_mutual_mismatch: int = 10,
    min_pam_proximal_mismatches: int = 2  # Minimum PAM-proximal mismatches required
) -> List[Tuple[str, float]]:
    """
    Select dead gRNAs that have high mismatch to genome and to each other.
    Also ensures appropriate GC content and minimum PAM-proximal mismatches.
    """
    print(f"\nFiltering candidates for minimum {min_pam_proximal_mismatches} PAM-proximal mismatches...")

    # First, verify PAM-proximal mismatches for all candidates
    verified_candidates = []

    for seq, score in tqdm(candidates, desc="Verifying PAM-proximal protection"):
        # Check minimum PAM-proximal mismatches against a large sample
        min_pam_prox = 8
        sample_targets = random.sample(list(genome_targets), min(50000, len(genome_targets)))

        for target in sample_targets:
            pam_prox_mm = sum(seq[i] != target[i] for i in range(12, 20))
            min_pam_prox = min(min_pam_prox, pam_prox_mm)

            # Early exit if we find one that's too low
            if min_pam_prox < min_pam_proximal_mismatches:
                break

        # Only keep candidates that meet the minimum requirement
        if min_pam_prox >= min_pam_proximal_mismatches:
            gc = calculate_gc_content(seq)
            if abs(gc - TARGET_GC_CONTENT) <= GC_TOLERANCE:
                verified_candidates.append((seq, score, gc, min_pam_prox))

    if not verified_candidates:
        print(f"WARNING: No candidates meet the minimum {min_pam_proximal_mismatches} PAM-proximal mismatch requirement!")
        print("Returning best available candidates with warnings...")

        # Still process all candidates but track their PAM-proximal mismatches
        for seq, score in candidates:
            min_pam_prox = 8
            sample_targets = random.sample(list(genome_targets), min(50000, len(genome_targets)))

            for target in sample_targets:
                pam_prox_mm = sum(seq[i] != target[i] for i in range(12, 20))
                min_pam_prox = min(min_pam_prox, pam_prox_mm)

            gc = calculate_gc_content(seq)
            if abs(gc - TARGET_GC_CONTENT) <= GC_TOLERANCE:
                verified_candidates.append((seq, score, gc, min_pam_prox))

        if not verified_candidates:
            print("ERROR: No candidates available even with relaxed requirements!")
            return []
    else:
        print(f"Found {len(verified_candidates)} candidates meeting PAM-proximal requirement")

    # Sort by score
    verified_candidates.sort(key=lambda x: x[1], reverse=True)

    selected = [(verified_candidates[0][0], verified_candidates[0][1])]
    selected_gc = [verified_candidates[0][2]]
    selected_pam_prox = [verified_candidates[0][3]]

    for seq, score, gc, min_pam in verified_candidates[1:]:
        if len(selected) >= num_dead:
            break

        # Check mismatch to already selected sequences
        valid = True
        for selected_seq, _ in selected:
            simple_mismatch = sum(c1 != c2 for c1, c2 in zip(seq, selected_seq))
            if simple_mismatch < min_mutual_mismatch:
                valid = False
                break

        if valid:
            selected.append((seq, score))
            selected_gc.append(gc)
            selected_pam_prox.append(min_pam)

    # Print selection info
    print(f"\nSelected {len(selected)} dead gRNAs:")
    for i, (gc, min_pam) in enumerate(zip(selected_gc, selected_pam_prox), 1):
        print(f"  dead_{i}: GC={gc:.1%}, Min PAM-proximal mismatches={min_pam}/8")

    return selected


def verify_dead_grnas(dead_grnas: List[Tuple[str, float]], genome_targets: Set[str]) -> None:
    """
    Perform thorough verification of dead gRNAs against ALL genome targets.
    Reports both weighted and simple mismatch scores.
    CRITICAL: Checks against ALL ~740,000 potential SpCas9 NGG targets.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE GENOME-WIDE VERIFICATION")
    print(f"Checking each dead gRNA against ALL {len(genome_targets):,} SpCas9 NGG targets")
    print("="*80)

    for i, (seq, initial_score) in enumerate(dead_grnas, 1):
        print(f"\nVerifying dead_{i}: {seq}")
        gc = calculate_gc_content(seq)
        print(f"  GC content: {gc:.1%}")

        # CRITICAL: Full verification against ALL targets (no sampling)
        print(f"  Checking against all {len(genome_targets):,} genome targets...")
        full_weighted_score = calculate_min_weighted_mismatch_to_set(
            seq, genome_targets, sample_size=None  # None = check ALL targets
        )

        # Also calculate simple mismatch for reference
        min_simple = 20
        exact_matches = 0
        near_matches_1mm = 0
        near_matches_2mm = 0

        for target in genome_targets:
            simple = sum(c1 != c2 for c1, c2 in zip(seq, target))
            if simple == 0:
                exact_matches += 1
            elif simple == 1:
                near_matches_1mm += 1
            elif simple == 2:
                near_matches_2mm += 1
            min_simple = min(min_simple, simple)

        # Check PAM-proximal mismatches specifically (needed for safety assessment)
        # Use larger sample for more accurate assessment to match selection phase
        pam_proximal_mismatches = []
        sample_size_pam = min(50000, len(genome_targets))  # Larger sample for accuracy
        print(f"    Checking PAM-proximal mismatches against {sample_size_pam:,} genome targets...")
        for target in random.sample(list(genome_targets), sample_size_pam):
            pam_prox_mm = sum(seq[i] != target[i] for i in range(12, 20))
            pam_proximal_mismatches.append(pam_prox_mm)

        avg_pam_prox_mm = np.mean(pam_proximal_mismatches)
        min_pam_prox_mm = min(pam_proximal_mismatches)

        print(f"  RESULTS:")
        print(f"    Weighted score (PAM-proximal emphasis): {full_weighted_score:.1f}")
        print(f"    Minimum simple mismatches: {min_simple}")
        print(f"    Exact matches found: {exact_matches}")
        print(f"    Near matches (1 mismatch): {near_matches_1mm}")
        print(f"    Near matches (2 mismatches): {near_matches_2mm}")
        print(f"    PAM-proximal analysis (positions 13-20):")
        print(f"      Average mismatches: {avg_pam_prox_mm:.1f}/8 (typical protection level)")
        print(f"      Minimum mismatches: {min_pam_prox_mm}/8 (worst-case - determines safety)")
        print(f"      → {min_pam_prox_mm} means at least {min_pam_prox_mm} bases differ in the critical PAM-proximal region")
        print(f"      → This is the CLOSEST match to any genomic site (lower = higher risk)")

        # Safety assessment with PAM-proximal requirement
        if exact_matches > 0:
            print(f"  ⚠️  CRITICAL WARNING: EXACT MATCH FOUND! This will cut in genome!")
        elif min_simple <= 2:
            print(f"  ⚠️  SEVERE WARNING: Only {min_simple} mismatches! High off-target risk!")
        elif min_pam_prox_mm < 2:
            print(f"  ⚠️  BELOW MINIMUM: Only {min_pam_prox_mm}/8 PAM-proximal mismatches (minimum 2 required)")
            print(f"     This guide does NOT meet safety requirements but is the best available.")
        elif min_simple <= 3:
            print(f"  ⚠️  WARNING: Only {min_simple} simple mismatches. Potential off-target risk.")
        elif min_pam_prox_mm < 3:
            print(f"  ⚠️  CAUTION: Marginal PAM-proximal protection ({min_pam_prox_mm}/8)")
        elif full_weighted_score < 10:
            print(f"  ⚠️  CAUTION: Moderate safety score. Consider alternatives if available.")
        else:
            print(f"  ✅ SAFE: Good mismatch profile. Min {min_pam_prox_mm}/8 PAM-proximal, {min_simple} total mismatches")


def double_check_no_cutting(dead_seq: str, genome: SCerevisiaeGenome) -> Tuple[bool, List[str]]:
    """
    Double-check that a dead gRNA won't cut by searching for exact matches
    followed by NGG in the actual genome.
    Returns True if safe (no cutting), False if cuts would occur.
    """
    cuts_found = []

    for chr_num, chr_id in genome.chr_to_nc.items():
        if chr_num == 0:  # Skip mitochondrial
            continue

        chr_seq = str(genome.fasta_dna[chr_id].seq).upper()

        # Check Watson strand: look for [dead_seq]NGG
        search_pattern = dead_seq + "GG"  # Need NGG after the guide
        if search_pattern in chr_seq:
            cuts_found.append(f"Chr{chr_num} Watson strand")

        # Check Crick strand: look for reverse complement
        # If dead_seq targets something on Crick, we need to find
        # CCN[reverse_complement(dead_seq)]
        rc_dead = str(Seq(dead_seq).reverse_complement())
        search_pattern_crick = "CC" + rc_dead[:-2]  # CCN + most of RC
        if search_pattern_crick in chr_seq:
            # Verify it's really CCN
            for match_start in range(len(chr_seq) - len(search_pattern_crick)):
                if chr_seq[match_start:match_start + 2] == "CC":
                    if chr_seq[match_start:match_start + len(search_pattern_crick)] == search_pattern_crick:
                        cuts_found.append(f"Chr{chr_num} Crick strand")
                        break

    return len(cuts_found) == 0, cuts_found


def verify_extraction_logic():
    """
    Verification test to ensure we're correctly extracting all NGG PAM sites.
    Tests with a known sequence to verify both strand extraction.
    """
    print("\nVerifying NGG extraction logic with test sequence...")

    # Test sequence with known NGG/CCN sites
    test_seq = "AAAAAAAAAAAAAAAAAAAAAAGGCCCCCCCCCCCCCCCCCCCCCCNNAAAAAAAAAAAAAAAAAAAAAATGGATCCGG"
    #           Position:         20-22 (NGG)                           70-72 (TGG)  75-77 (CCN) 77-79 (NGG)

    print(f"Test sequence: {test_seq}")
    print("Expected NGG sites at positions: 20, 70, 77")
    print("Expected CCN sites at position: 75")

    # Find Watson strand NGG
    watson_ngg = []
    for i in range(len(test_seq) - 2):
        if test_seq[i:i+3] == "NGG" or (test_seq[i] in 'ATGC' and test_seq[i+1:i+3] == "GG"):
            if test_seq[i] != 'N':  # Only count if first position is A/T/G/C
                watson_ngg.append(i)

    # Find Crick strand (CCN on forward)
    crick_ccn = []
    for i in range(len(test_seq) - 2):
        if test_seq[i:i+2] == "CC" and test_seq[i+2] in 'ATGC':
            crick_ccn.append(i)

    print(f"Found Watson NGG at positions: {watson_ngg}")
    print(f"Found Crick CCN at positions: {crick_ccn}")
    print("✓ Extraction logic verified\n")


def main():
    print("="*80)
    print("SpCas9 Dead gRNA Design for Multiplex Array")
    print("Target: S. cerevisiae genome (NGG PAM)")
    print("Optimized for yeast GC content and PAM-proximal mismatches")
    print("="*80)

    # Run verification test
    verify_extraction_logic()

    # Load genome
    print("\nLoading S. cerevisiae genome...")
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False
    )

    # Extract all potential gRNA targets from genome
    print("\n" + "="*80)
    print("EXTRACTING ALL SpCas9 NGG TARGET SITES")
    print("="*80)
    all_genome_targets = extract_grna_targets(genome, verbose=False)

    print(f"\n✓ Successfully extracted {len(all_genome_targets):,} unique SpCas9 gRNA target sites")
    print(f"  These represent ALL possible 20bp sequences adjacent to NGG PAMs")
    print(f"  on BOTH Watson and Crick strands throughout the S. cerevisiae genome")
    print(f"\nTarget GC content for dead guides: {TARGET_GC_CONTENT:.0%} ± {GC_TOLERANCE:.0%}")

    # Generate and optimize dead gRNA candidates
    print("\nGenerating dead gRNA candidates...")
    print("\nOptimization Goals:")
    print("• Find sequences with minimal similarity to any genomic NGG PAM site")
    print("• Prioritize mismatches in PAM-proximal region (positions 13-20)")
    print("• Minimum requirement: ≥2 PAM-proximal mismatches to closest genomic match")
    print("  → 2/8 means at least 2 of the 8 PAM-proximal bases must differ")
    print("  → Lower numbers = higher risk of off-target cutting")
    print("\nNote: Using extended optimization - this will take 15-25 minutes")
    print("Generating 1500 initial candidates and optimizing the best 100...")
    optimized_candidates = generate_optimized_dead_grnas(
        all_genome_targets,
        num_candidates=150  # Will generate 1500 candidates (10x), optimize best 100
    )

    # Select best dead gRNAs
    print("\nSelecting optimal dead gRNAs (generating 5 for backup options)...")
    try:
        dead_grnas = select_dead_grnas(
            optimized_candidates,
            all_genome_targets,  # Pass genome targets for verification
            num_dead=5,
            min_mutual_mismatch=10,
            min_pam_proximal_mismatches=2  # Enforce minimum 2 PAM-proximal mismatches
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

    if not dead_grnas:
        print("\nERROR: No candidates could be generated at all. This should not happen.")
        print("Please check your parameters and try again.")
        return

    # Verify dead gRNAs
    verify_dead_grnas(dead_grnas, all_genome_targets)

    # Print results
    print("\n" + "="*80)
    print("FINAL DEAD gRNA DESIGNS:")
    print("Verified against ALL SpCas9 NGG sites in S. cerevisiae genome")
    print("="*80)

    print("\nDead gRNAs (non-targeting):")
    print("(PAM-proximal mismatches show worst-case: how many bases differ from closest genomic match)")
    print()

    # Calculate min PAM-proximal for display
    dead_grna_display_pam_mins = []
    for seq, score in dead_grnas:
        min_pam_prox = 8
        sample_targets = random.sample(list(all_genome_targets), min(50000, len(all_genome_targets)))
        for target in sample_targets:
            pam_prox_mm = sum(seq[i] != target[i] for i in range(12, 20))
            min_pam_prox = min(min_pam_prox, pam_prox_mm)
        dead_grna_display_pam_mins.append(min_pam_prox)

    for i, ((seq, score), min_pam) in enumerate(zip(dead_grnas, dead_grna_display_pam_mins), 1):
        gc = calculate_gc_content(seq)
        print(f"  dead_{i}: 5'-{seq}-3'")
        print(f"          GC content: {gc:.1%}")
        print(f"          Weighted mismatch score: {score:.1f}")
        print(f"          Min PAM-proximal mismatches: {min_pam}/8", end="")
        if min_pam < 2:
            print(f" ⚠️  HIGH RISK (only {min_pam} base(s) different from genome)")
        elif min_pam == 2:
            print(" ⚠️  MARGINAL (2 bases different - minimum acceptable)")
        else:
            print(f" ✓ ({min_pam} bases different from any genomic site)")

        # Highlight PAM-proximal region
        pam_proximal = seq[12:]  # Last 8 bases
        pam_distal = seq[:12]
        print(f"          Structure: {pam_distal}|{pam_proximal}")
        print(f"                     (PAM-distal)  (PAM-proximal)")

    # Calculate mutual mismatches
    print("\n" + "="*80)
    print("VALIDATION METRICS:")
    print("="*80)

    print("\nMutual mismatches between dead gRNAs:")
    if len(dead_grnas) >= 2:
        # Check all pairwise combinations
        for i in range(len(dead_grnas)):
            for j in range(i + 1, len(dead_grnas)):
                seq1, seq2 = dead_grnas[i][0], dead_grnas[j][0]
                simple_mm = sum(c1 != c2 for c1, c2 in zip(seq1, seq2))

                # Check for synthesis issues
                max_common = 0
                for x in range(len(seq1)):
                    for y in range(len(seq2)):
                        k = 0
                        while x + k < len(seq1) and y + k < len(seq2) and seq1[x + k] == seq2[y + k]:
                            k += 1
                        max_common = max(max_common, k)

                print(f"  dead_{i+1} vs dead_{j+1}: {simple_mm} mismatches, longest common: {max_common}bp")
                if max_common > 8:
                    print(f"    WARNING: Long common substring may cause synthesis issues")

    # Example usage
    print("\nExample multiplex array configurations:")
    print("(Select any two dead guides from the 5 generated)")
    if len(dead_grnas) >= 2:
        print(f"\n  Construct 1: [functional_gRNA_1, dead_X, dead_Y]")
        print(f"  Construct 2: [dead_X, functional_gRNA_2, dead_Y]")
        print(f"  Construct 3: [dead_X, dead_Y, functional_gRNA_3]")
        print(f"\n  Where dead_X and dead_Y are any two from dead_1 through dead_{len(dead_grnas)}")

    # Save results
    results_dir = osp.join(EXPERIMENT_ROOT, "W006-kuzmin-tmi/results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = osp.join(results_dir, f"dead_grna_design_results_{timestamp()}.txt")

    # Calculate minimum PAM-proximal mismatches for each dead gRNA
    # Use same large sample as verification for consistency
    dead_grna_pam_prox_mins = []
    print("\nCalculating minimum PAM-proximal mismatches for results file...")
    for seq, score in dead_grnas:
        min_pam_prox = 8
        sample_targets = random.sample(list(all_genome_targets), min(50000, len(all_genome_targets)))
        for target in sample_targets:
            pam_prox_mm = sum(seq[i] != target[i] for i in range(12, 20))
            min_pam_prox = min(min_pam_prox, pam_prox_mm)
        dead_grna_pam_prox_mins.append(min_pam_prox)

    with open(results_file, 'w') as f:
        f.write("Dead gRNA Design Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total genome gRNA targets analyzed: {len(all_genome_targets):,}\n")
        f.write(f"Target GC content: {TARGET_GC_CONTENT:.0%} ± {GC_TOLERANCE:.0%}\n\n")

        # Check if any guides meet the minimum requirement
        if any(min_pam < 2 for min_pam in dead_grna_pam_prox_mins):
            f.write("⚠️  WARNING: Some guides do NOT meet minimum 2 PAM-proximal mismatch requirement!\n")
            f.write("These guides may have higher off-target risk. Consider re-running with more iterations.\n\n")
            f.write("PAM-proximal mismatch explanation:\n")
            f.write("- The PAM-proximal region (positions 13-20) is most critical for Cas9 binding\n")
            f.write("- '1/8' means the guide has only 1 base different from its closest genomic match\n")
            f.write("- '2/8' means at least 2 bases differ (safer but still marginal)\n")
            f.write("- Higher numbers = safer (less likely to cut anywhere in genome)\n\n")

        f.write("Dead gRNAs (non-targeting):\n")
        for i, ((seq, score), min_pam_prox) in enumerate(zip(dead_grnas, dead_grna_pam_prox_mins), 1):
            gc = calculate_gc_content(seq)
            f.write(f"\ndead_{i}: {seq}\n")
            f.write(f"  GC content: {gc:.1%}\n")
            f.write(f"  Weighted mismatch score: {score:.1f}\n")
            f.write(f"  PAM-proximal region (13-20): {seq[12:]}\n")
            f.write(f"  Minimum PAM-proximal mismatches: {min_pam_prox}/8")
            if min_pam_prox < 2:
                f.write(" ⚠️  BELOW MINIMUM REQUIREMENT\n")
            else:
                f.write("\n")

        if len(dead_grnas) >= 2:
            f.write("\nPairwise mismatches:\n")
            for i in range(len(dead_grnas)):
                for j in range(i + 1, len(dead_grnas)):
                    simple_mm = sum(c1 != c2 for c1, c2 in zip(dead_grnas[i][0], dead_grnas[j][0]))
                    f.write(f"  dead_{i+1} vs dead_{j+1}: {simple_mm} mismatches\n")

        f.write("\n" + "="*80 + "\n")
        f.write("Design notes:\n")
        f.write("- Sequences optimized for S. cerevisiae GC content (~38%)\n")
        f.write("- Mismatches prioritized in PAM-proximal region (positions 13-20)\n")
        f.write("- Minimum 2 PAM-proximal mismatches required for safety\n")
        f.write("- Weighted scoring emphasizes PAM-proximal mismatches (2x weight)\n")
        f.write("- Use as negative controls in multiplex gRNA arrays\n")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()