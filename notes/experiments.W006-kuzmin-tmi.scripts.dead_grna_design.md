---
id: oa8t4rj5l39iimp7gf2bkad
title: dead_grna_design
desc: ''
updated: 1759879067525
created: 1759693541158
---

## Dead SpCas9 NGG gRNA Design Algorithm for S. cerevisiae

This script generates non-targeting "dead" gRNAs for use as negative controls in multiplex CRISPR arrays. The algorithm ensures these sequences have minimal probability of cutting anywhere in the S. cerevisiae genome while maintaining synthesis feasibility.

**CRITICAL: The script verifies dead guides against ALL ~740,000 SpCas9 NGG target sites in the S. cerevisiae genome on BOTH Watson and Crick strands.**

## Key Design Principles

### Target Specifications

- **GC Content**: 38% ± 15% (relaxed tolerance for better PAM-proximal mismatches)
- **Length**: 20 base pairs (standard gRNA length)
- **Output**: 5 dead gRNA sequences (provides backup options for synthesis issues)
- **Mutual dissimilarity**: Minimum 10 mismatches between any two dead guides

## Detailed Generation Process

### 1. Initial Sequence Generation with Yeast-like GC Content

The algorithm generates random 20bp sequences matching yeast genomic composition:

```
Target: 38% GC content (±15% tolerance)
For 20bp sequence: ~8 GC bases, ~12 AT bases
Acceptable range: 23-53% GC content (5-11 GC bases out of 20)
```

**Process:**

- Creates base composition matching yeast (e.g., 4 G's, 4 C's, 6 A's, 6 T's)
- Randomly shuffles bases to create initial sequences
- Performs extra shuffling of the last 8 bases (PAM-proximal region) to increase variation where it matters most

### 2. Weighted Mismatch Scoring System

The algorithm uses position-sensitive scoring because Cas9 tolerance to mismatches varies along the guide:

```
5'[-----------|-------|--------]3'-NGG
  Pos 1-8    Pos 9-12  Pos 13-20  PAM
  Weight:0.5  Weight:1  Weight:2
  (tolerant) (moderate) (critical)
```

**Position weights:**

- **Positions 13-20** (PAM-proximal): Most critical for Cas9 binding. Mismatches here severely disrupt cutting. Weight = 2.0
- **Positions 9-12** (intermediate): Moderate sensitivity. Weight = 1.0
- **Positions 1-8** (PAM-distal): Cas9 tolerates mismatches better here. Weight = 0.5

This weighting system reflects empirical data showing that positions ~1-8 from the PAM are most mismatch-sensitive for target cleavage.

### 3. Optimization via PAM-Proximal Mutations

The optimization focuses mutations where they have the most impact:

```python
# 80% of mutations target positions 13-20
# 20% of mutations target other positions
```

**Optimization process:**

1. Takes a candidate sequence
2. Randomly selects positions to mutate (biased toward PAM-proximal)
3. Tries different bases while maintaining GC content (23-53%)
4. Keeps mutations that increase the weighted mismatch score
5. Iterates 30 times per candidate

### 4. Comprehensive Genome-wide SpCas9 Target Extraction

The script extracts ALL potential SpCas9 gRNA target sites from the S. cerevisiae genome:

**Dual-Strand Extraction Process:**

1. Processes all 16 nuclear chromosomes (excludes mitochondrial)
2. **Watson Strand NGG Sites:**
   - Finds all NGG PAM sites (where N = A/T/G/C)
   - Extracts 20bp sequence immediately upstream of each NGG
   - These represent forward strand targets
3. **Crick Strand NGG Sites:**
   - Finds all CCN PAM sites on forward strand (reverse complement of NGG)
   - Extracts 20bp sequence immediately downstream of each CCN
   - Reverse complements these sequences to get actual guide sequences
   - These represent reverse strand targets
4. **Result:** ~740,000 unique potential target sites covering BOTH strands

**Verification Features:**

- Reports total NGG/CCN PAM sites found
- Breaks down targets by Watson vs Crick strand
- Confirms extraction from both strands
- Validates that ALL SpCas9 sites are included

### 5. Sampling Strategy for Computational Efficiency

Since checking against 740,000+ genome targets is computationally expensive:

- **Initial scoring**: Samples 10,000 random genome targets for speed
- **Optimization**: Samples 25,000 targets per iteration for efficiency
- **Iterations**: 100 iterations per candidate (balanced for speed and quality)
- **Final verification**: Checks against ALL 740,000+ targets (NO SAMPLING)
- **Critical**: Final safety assessment ALWAYS uses complete genome-wide check

### 6. Selection Criteria

Final dead gRNAs must satisfy:

- GC content within 23-53% range (relaxed for better PAM-proximal protection)
- Maximum weighted mismatch score to genome
- At least 10 simple mismatches between each dead guide pair
- Minimal longest common substring (<8bp preferred to avoid synthesis issues)

## Example Result Analysis

For a typical dead gRNA like: `TATTAATTAATACGCTATCC`

- **GC content**: 30% (6 GC bases out of 20)
- **PAM-proximal region** (CGCTATCC): Contains variations that don't match common genomic patterns
- **PAM-distal region** (TATTAATTAATA): Can be more similar to genome since mismatches here are tolerated

## Comprehensive Genome-Wide Verification Process

The script performs multiple layers of verification to ensure absolute safety:

### Primary Verification (Against ALL Targets)

1. **Complete Genome Check**:
   - Verifies each dead gRNA against ALL ~740,000 SpCas9 NGG targets
   - NO SAMPLING - every single target is checked
   - Reports exact matches (must be 0 for safety)
   - Counts near matches with 1-2 mismatches

2. **Weighted Scoring**:
   - Calculates minimum weighted mismatch to any genome target
   - Emphasizes PAM-proximal mismatches (2x weight)

3. **Simple Mismatch Scoring**:
   - Reports unweighted mismatches for reference
   - Provides baseline safety metric

### Detailed Analysis

4. **PAM-Proximal Analysis**:
   - Specifically analyzes mismatches in critical positions 13-20
   - Reports minimum and average PAM-proximal mismatches

5. **Pairwise Comparison**:
   - Checks all 10 pairs of dead guides for mutual dissimilarity
   - Ensures minimum 10 mismatches between any pair

6. **Synthesis Check**:
   - Identifies long common substrings that could cause synthesis issues
   - Warns if common substrings exceed 8bp

### Verification Output

The verification explicitly reports:

- Total genome targets checked (should be ~740,000)
- Exact match count (MUST be 0)
- Near matches with 1-2 mismatches
- Safety classification based on comprehensive analysis
- Clear warnings if any potential cutting risk exists

## Safety Classifications

Based on comprehensive verification against ALL genome targets:

### Critical Failures

- **Exact matches found**: ⚠️ CRITICAL - Will definitely cut in genome (REJECT)
- **≤2 mismatches**: ⚠️ SEVERE WARNING - High off-target risk (REJECT)

### Risk Assessment (Weighted Score)

- **Score < 5**: ⚠️ WARNING - Low score, may have off-target effects
- **Score 5-10**: ⚠️ CAUTION - Moderate score, some off-target potential
- **Score > 10**: ✅ SAFE - High score, very low probability of cutting

### Additional Safety Metrics

- Minimum PAM-proximal mismatches must be ≥2
- Average PAM-proximal mismatches should be ≥5
- All pairwise dead gRNA mismatches must be ≥10

## Usage Notes

The script generates 5 dead gRNA sequences, allowing you to:

1. Choose the best pair based on your specific synthesis requirements
2. Have backup options if synthesis fails for certain sequences
3. Select guides with optimal mutual dissimilarity
4. Avoid sequences with problematic features (e.g., long homopolymer runs)

## Output Files

Results are saved to: `experiments/W006-kuzmin-tmi/results/dead_grna_design_results_[timestamp].txt`

The output includes:

- All 5 dead gRNA sequences verified against complete genome
- GC content for each sequence
- Weighted mismatch scores (PAM-proximal emphasis)
- Minimum PAM-proximal mismatches for each guide
- Simple mismatch scores (unweighted)
- PAM-proximal regions (positions 13-20) highlighted
- Complete pairwise mismatch matrix (all 10 combinations)
- Exact match and near-match counts
- Comprehensive safety assessment
- Design notes and parameters

## Critical Assurances

✅ **ALL SpCas9 NGG sites checked**: Every NGG PAM on both Watson and Crick strands
✅ **No sampling in final verification**: Complete genome-wide check for safety
✅ **Dual-strand coverage**: Both forward and reverse complement targets included
✅ **~740,000 targets verified**: Comprehensive coverage of S. cerevisiae genome
✅ **Multiple verification layers**: Weighted scoring, simple mismatches, and PAM-proximal analysis
✅ **Explicit safety warnings**: Clear indicators if any cutting risk exists
