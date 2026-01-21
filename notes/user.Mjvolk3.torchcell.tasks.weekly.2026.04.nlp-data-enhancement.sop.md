---
id: 9512m45535p1rdmxlbm7sj7
title: NLP Data Enhancement SOP
desc: 'Standard Operating Procedure for extracting dataset metadata from scientific papers'
updated: 1769024018731
created: 1768945362665
---

## 2026.01.20 - SOP: Extracting Dataset Metadata from Scientific Literature

### Purpose

This SOP defines the process for extracting dataset metadata (e.g., sample sizes, experimental conditions) from scientific publications and encoding it as traceable, reproducible code in dataset implementations.

**Intended for:** Human researchers and LLM agents performing dataset enhancement

**Future goal:** Codify as a Claude Code skill for automated extraction

### Overview

Many dataset fields cannot be derived from data tables alone and require information from paper text (methods, supplementary materials, figure legends). This SOP ensures extracted information is:

1. **Traceable:** Citation includes file path, line numbers, and verbatim quotes
2. **Reproducible:** Another researcher/agent can verify the extraction
3. **Maintainable:** Clear documentation enables future updates
4. **Auditable:** Evidence chain from paper → code is explicit

### Input Materials

**Required:**

- Dataset implementation file (e.g., `torchcell/datasets/scerevisiae/costanzo2016.py`)
- Schema definition file (`torchcell/datamodels/schema.py`)
- Primary publication (PDF + MMD formats)
- Supplementary materials (PDF + MMD formats)

**File organization:**

```bash
papers/
└── [publication-id]/
    ├── [publication-id].pdf          # Original PDF
    ├── [publication-id].mmd          # Mathpix markdown conversion
    ├── SI-[publication-id].pdf       # Supplementary info PDF
    └── SI-[publication-id].mmd       # Supplementary info MMD
```

**Example:**

```bash
papers/costanzoGlobalGeneticInteraction2016/
├── costanzoGlobalGeneticInteraction2016.pdf
├── costanzoGlobalGeneticInteraction2016.mmd
├── SI-costanzoGlobalGeneticInteraction2016.pdf
└── SI-costanzoGlobalGeneticInteraction2016.mmd
```

### Extraction Workflow

#### Phase 1: Identification

**Goal:** Determine what metadata needs extraction

**Process:**

1. **Read schema definition** to understand required fields

   ```python
   # Example: FitnessPhenotype requires n_samples
   class FitnessPhenotype(Phenotype, ModelStrict):
       fitness_se: float | None  # Requires n_samples to compute
       n_samples: int | None     # ← Must be extracted from paper
   ```

2. **Identify missing information** in data tables
   - Load raw dataset file (TSV, CSV, XLSX)
   - Check which schema fields are NOT in data columns
   - List fields requiring text extraction

3. **Categorize extraction targets:**
   - **Global constants:** Apply to all/most measurements (e.g., default n_samples)
   - **Conditional constants:** Vary by experimental condition (e.g., n_samples by temperature)
   - **Computed fields:** Derived from extracted + table data (e.g., SE = SD/√n)

#### Phase 2: Comprehensive Reading

**Goal:** Understand the full experimental design before targeted extraction

**Process:**

1. **Full paper read** (main text + SI)
   - Read both PDF and MMD versions
   - MMD is easier for grep/search but may have OCR errors
   - PDF is ground truth for ambiguous cases

2. **Document structure mapping:**
   - Methods section(s)
   - Supplementary methods
   - Figure legends and captions
   - Table notes and footnotes
   - Data availability statements

3. **Experimental design notes:**
   - How many replicates? (technical vs biological)
   - Are conditions uniform? (temperature, media, perturbation types)
   - Special cases? (controls, reference measurements, failed experiments)
   - Data processing? (outlier removal, normalization)

**⚠️ Critical:** Do not skip this phase. Targeted searches can miss important caveats that invalidate extraction.

#### Phase 3: Targeted Search

**Goal:** Find specific evidence for each metadata field

**Search strategy:**

1. **Keyword search in MMD files:**

   ```bash
   # Example: Finding sample sizes
   grep -in "replicate\|independent\|measurement\|n=\|duplicate\|triplicate" \
     papers/costanzoGlobalGeneticInteraction2016/*.mmd

   # Example: Finding specific experimental conditions
   grep -in "temperature.*26.*degree\|26.*°C\|26C" \
     papers/costanzoGlobalGeneticInteraction2016/*.mmd
   ```

2. **Context extraction:**
   - Note line numbers where evidence found
   - Extract 5-10 lines of surrounding context
   - Verify in PDF (MMD line numbers approximate PDF location)

3. **Priority sections for different metadata types:**

   | Metadata Type | Priority Sections |
   |---------------|-------------------|
   | Sample sizes (n) | Methods > Supplementary Methods > Figure legends |
   | Experimental conditions | Methods > Materials > Supplementary Tables |
   | Data processing | Methods > Data availability > Supplementary Methods |
   | Quality control | Methods > Supplementary Methods |
   | Outlier handling | Methods > Supplementary Methods |

4. **LLM-assisted extraction:**
   - Can use for initial extraction from long methods sections
   - MUST verify against paper text
   - MUST provide citations for verification

#### Phase 4: Evidence Documentation

**Goal:** Create traceable citations for extracted metadata

**Format:**

```python
# [Variable purpose/description]
# Quote: "[Exact text from paper, can span multiple lines.
#         Preserve line breaks for readability.]"
# Source: [filename].mmd, Lines [start]-[end]
# Verified: [filename].pdf, Page [N], [Section name]
# Date extracted: [YYYY-MM-DD]
[VARIABLE_NAME] = [value]
```

**Example:**

```python
# Default sample size for all fitness measurements
# Quote: "All fitness measurements represent the mean of at least
#         2 independent measurements performed on different days."
# Source: costanzoGlobalGeneticInteraction2016.mmd, Lines 1523-1524
# Verified: costanzoGlobalGeneticInteraction2016.pdf, Page 8, Methods section
# Date extracted: 2026-01-20
N_SAMPLES_DEFAULT = 2

# Wild-type control measurements at 26°C
# Quote: "Wild-type control measurements were performed in quadruplicate
#         for each 384-well array plate to establish baseline fitness."
# Source: SI-costanzoGlobalGeneticInteraction2016.mmd, Lines 234-236
# Verified: SI-costanzoGlobalGeneticInteraction2016.pdf, Page 12,
#           Supplementary Methods - Quality Control
# Date extracted: 2026-01-20
N_SAMPLES_WT_26C = 4

# Temperature-sensitive allele measurements
# Quote: "Temperature-sensitive allele strains were measured at both
#         permissive (26°C) and restrictive (30°C) temperatures in
#         biological triplicate."
# Source: costanzoGlobalGeneticInteraction2016.mmd, Lines 1678-1680
# Verified: costanzoGlobalGeneticInteraction2016.pdf, Page 9, Methods section
# Date extracted: 2026-01-20
N_SAMPLES_TSA = 3
```

**Requirements:**

- ✅ Verbatim quotes (use "..." for omissions if needed)
- ✅ Line numbers from MMD file (enables grep verification)
- ✅ PDF page and section (ground truth reference)
- ✅ Date of extraction (enables update tracking)
- ✅ Clear variable names (self-documenting code)
- ❌ No paraphrasing in quotes
- ❌ No assumed values without citation

#### Phase 5: Code Implementation

**Goal:** Integrate extracted metadata into dataset code

**Location in dataset file:**

```python
# 1. Place global constants at top of file, after imports
# 2. Group by category (default values, condition-specific, references)
# 3. Each constant has evidence documentation (Phase 4 format)

import logging
import pandas as pd
# ... other imports ...

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ============================================================================
# Sample Size Metadata - Extracted from Costanzo et al. 2016
# ============================================================================

# Default sample size for all fitness measurements
# Quote: "All fitness measurements represent the mean of at least
#         2 independent measurements performed on different days."
# Source: costanzoGlobalGeneticInteraction2016.mmd, Lines 1523-1524
# Verified: costanzoGlobalGeneticInteraction2016.pdf, Page 8, Methods
# Date extracted: 2026-01-20
N_SAMPLES_DEFAULT = 2

# [Additional constants with citations...]

# ============================================================================
# Dataset Classes
# ============================================================================

@register_dataset
class SmfCostanzo2016Dataset(ExperimentDataset):
    # ... implementation uses constants above ...
```

**Usage in processing methods:**

```python
def _process_data_item(self, row: pd.Series) -> dict:
    # ... existing code ...

    # Determine n_samples based on experimental conditions
    if row["Temperature"] == 26:
        if "tsa" in row["Array Strain ID"].lower():
            n_samples = N_SAMPLES_TSA
        else:
            n_samples = N_SAMPLES_DEFAULT
    elif row["Temperature"] == 30:
        n_samples = N_SAMPLES_DEFAULT
    else:
        # Unknown condition - log warning
        log.warning(f"Unknown temperature {row['Temperature']} for {row['Query Strain ID']}")
        n_samples = None

    # Compute derived fields
    fitness_std = row["Single mutant fitness stddev"]
    if fitness_std is not None and n_samples is not None and n_samples > 0:
        fitness_se = fitness_std / math.sqrt(n_samples)
    else:
        fitness_se = None

    # Create phenotype with extracted metadata
    phenotype = FitnessPhenotype(
        fitness=row["Single mutant fitness"],
        fitness_std=fitness_std,
        fitness_se=fitness_se,  # Derived from extracted n_samples
        n_samples=n_samples     # Extracted from paper
    )
```

#### Phase 6: Validation

**Goal:** Verify extraction correctness and code behavior

**Validation checklist:**

1. **Citation verification:**

   ```bash
   # Verify MMD line numbers
   sed -n '1523,1524p' papers/costanzoGlobalGeneticInteraction2016/costanzoGlobalGeneticInteraction2016.mmd

   # Should output the quoted text exactly
   ```

2. **PDF cross-check:**
   - Open PDF to cited page
   - Locate cited section
   - Confirm quote matches (accounting for OCR differences in MMD)

3. **Code correctness:**

   ```python
   # Test SE computation
   import math
   fitness_std = 0.1
   n_samples = 4
   expected_se = fitness_std / math.sqrt(n_samples)
   assert abs(expected_se - 0.05) < 1e-10
   ```

4. **Dataset loading:**

   ```python
   # Load small subset and verify fields present
   dataset = SmfCostanzo2016Dataset(root="...", subset_n=10)
   item = dataset[0]

   assert "n_samples" in item["experiment"]["phenotype"]
   assert "fitness_se" in item["experiment"]["phenotype"]
   assert item["experiment"]["phenotype"]["n_samples"] == 2  # or expected value
   ```

5. **Edge case testing:**
   - Missing fitness_std → fitness_se should be None
   - Zero or negative n_samples → handle gracefully
   - Unknown experimental conditions → log warning, set None

### Guidelines for LLM Agents

**When performing extraction:**

1. **Always read the entire paper first** (main + SI)
   - Do not skip to targeted search immediately
   - Understand full experimental context before extraction

2. **Search comprehensively:**
   - Use multiple keyword variants ("replicate" AND "measurement" AND "independent")
   - Check multiple sections (methods, SI, figure legends, table notes)
   - MMD files are easier to grep, but PDF is ground truth

3. **Provide complete citations:**
   - Exact quotes (no paraphrasing)
   - MMD line numbers (enables verification)
   - PDF page and section (ground truth)
   - Extraction date (enables tracking)

4. **Flag uncertainties:**
   - If text is ambiguous, state the ambiguity
   - If multiple interpretations exist, list them
   - If no evidence found, explicitly state this

5. **Consider special cases:**
   - Control measurements vs experimental measurements
   - Technical replicates vs biological replicates
   - Different conditions (temperature, media, perturbation types)
   - Failed/excluded measurements

6. **Validate extraction:**
   - Verify line numbers with grep
   - Check PDF for OCR errors in MMD
   - Ensure values are consistent across paper
   - Look for contradictions between main text and SI

### What to Ignore

**Do NOT extract if:**

1. **Not directly relevant to current schema fields**
   - Stay focused on specific metadata needs
   - Do not extract "potentially useful" information without clear purpose

2. **Redundant with data table columns**
   - If already in TSV/CSV/XLSX, use table data (primary source)
   - Only extract what's missing from tables

3. **Derived statistics you can compute**
   - Do not extract means if you have raw values
   - Do not extract SE if you can compute from SD and n

4. **Subjective interpretations**
   - Extract facts, not interpretations
   - If interpretation needed, document reasoning separately

5. **Author opinions or speculation**
   - Focus on methods and data, not discussion sections
   - Do not extract "possible explanations" or "future work"

### Quality Control

**Before considering extraction complete:**

- [ ] All required metadata fields have evidence citations
- [ ] All citations include exact quotes
- [ ] All citations include MMD line numbers
- [ ] All citations include PDF page and section
- [ ] All citations verified with grep and PDF check
- [ ] All global constants documented at top of dataset file
- [ ] All conditional logic documented with comments
- [ ] Edge cases handled (missing data, unknown conditions)
- [ ] Dataset loads successfully with new fields
- [ ] SE computations verified with test cases
- [ ] No assumed values without explicit citations

### Example: Complete Extraction

**Task:** Extract n_samples for Costanzo 2016 dataset

**Input materials:**

```bash
papers/costanzoGlobalGeneticInteraction2016/
├── costanzoGlobalGeneticInteraction2016.pdf
├── costanzoGlobalGeneticInteraction2016.mmd
├── SI-costanzoGlobalGeneticInteraction2016.pdf
└── SI-costanzoGlobalGeneticInteraction2016.mmd
```

**Phase 1 - Identification:**

- Schema requires: `n_samples: int | None`
- Data table has: fitness, fitness_std (but not n_samples)
- Need to extract: default n_samples, condition-specific values

**Phase 2 - Comprehensive reading:**

- Read full main text: Experiment uses SGA arrays, temperature conditions
- Read full SI: Details on quality control, replication strategy
- Note: Different conditions may have different n values

**Phase 3 - Targeted search:**

```bash
grep -in "replicate\|independent\|measurement" \
  papers/costanzoGlobalGeneticInteraction2016/*.mmd | head -20

# Found on line 1523: "at least 2 independent measurements"
# Found on line 234 of SI: "quadruplicate for control measurements"
```

**Phase 4 - Evidence documentation:**

```python
# Default sample size for single/double mutant fitness measurements
# Quote: "All fitness measurements represent the mean of at least
#         2 independent measurements performed on different days."
# Source: costanzoGlobalGeneticInteraction2016.mmd, Lines 1523-1524
# Verified: costanzoGlobalGeneticInteraction2016.pdf, Page 8, Methods section
# Date extracted: 2026-01-20
N_SAMPLES_DEFAULT = 2

# Wild-type reference measurements
# Quote: "Wild-type control measurements were performed in quadruplicate
#         for each 384-well array plate."
# Source: SI-costanzoGlobalGeneticInteraction2016.mmd, Lines 234-236
# Verified: SI-costanzoGlobalGeneticInteraction2016.pdf, Page 12,
#           Supplementary Methods
# Date extracted: 2026-01-20
N_SAMPLES_WT_CONTROL = 4
```

**Phase 5 - Implementation:**

- Add constants to top of `costanzo2016.py`
- Use in `_process_data_item()` methods
- Compute `fitness_se = fitness_std / sqrt(n_samples)`

**Phase 6 - Validation:**

```bash
# Verify line 1523
sed -n '1523p' papers/costanzoGlobalGeneticInteraction2016/costanzoGlobalGeneticInteraction2016.mmd
# Output matches quote ✓

# Test dataset
python -c "
from torchcell.datasets.scerevisiae.costanzo2016 import SmfCostanzo2016Dataset
dataset = SmfCostanzo2016Dataset(root='...', subset_n=10)
print(dataset[0]['experiment']['phenotype']['n_samples'])
# Output: 2 ✓
"
```

### Future: Structured Evidence Models

**Status:** Aspirational design for future implementation

**See also:** [[codify-nl-evidence|user.Mjvolk3.torchcell.tasks.weekly.2026.04.nlp-data-enhancement.codify-nl-evidence]]

#### Motivation

Current approach uses comments for evidence documentation. Future approach codifies evidence as structured data (pydantic models) that can be:

- Validated automatically
- Versioned independently
- Used to generate code
- Verified against source papers
- Compared across dataset versions

#### Pydantic Model Design

**Core model for LLM-based extraction:**

```python
# torchcell/datamodels/evidence.py
# FUTURE DESIGN - Not currently implemented
# This model specifies the structure for LLM output during metadata extraction

from pydantic import Field
from torchcell.datamodels.pydantic import ModelStrict

class EvidenceLocation(ModelStrict):
    """Location of evidence in source document."""
    file_path: str  # Relative to paper dir, e.g., "main.mmd" or "SI.mmd"
    line_range: str  # Format: "start-end" or "line" e.g., "1523-1524" or "1523"
    section: str | None = None  # Optional: "Methods", "Supplementary Methods", etc.

class PDFVerification(ModelStrict):
    """Cross-reference verification in PDF."""
    page: int
    section: str  # E.g., "Methods section", "Table 2 caption"
    verified: bool = True

class MetadataExtraction(ModelStrict):
    """Complete evidence record for a dataset metadata variable.

    This model defines the structure for LLM-based extraction of metadata
    from scientific papers. LLMs should output JSON matching this schema.

    Example usage:
        # LLM prompt includes this schema
        prompt = f'''
        Read the paper and extract sample size information.
        Output JSON matching this schema:
        {MetadataExtraction.model_json_schema()}
        '''

        # Validate LLM output
        extraction = MetadataExtraction.model_validate_json(llm_response)

        # Use in code generation
        code = f"{extraction.variable_name} = {extraction.variable_value}"
    """

    # REQUIRED: Variable being defined
    variable_name: str = Field(
        description="Python variable name, e.g., 'N_SAMPLES_DEFAULT'"
    )
    variable_value: int | float | str = Field(
        description="Value to assign to variable"
    )
    variable_description: str = Field(
        description="Brief description of what this variable represents"
    )

    # REQUIRED: Paper sources
    paper_dir: str = Field(
        description="Directory containing paper files, e.g., 'papers/costanzo2016'"
    )
    source_files: list[str] = Field(
        description="Files read during extraction, e.g., ['main.mmd', 'SI.mmd']"
    )

    # REQUIRED: Evidence text and locations
    evidence_quotes: list[str] = Field(
        description="Exact quotes from paper justifying the variable value. "
                    "Must be verbatim - no paraphrasing."
    )
    evidence_locations: list[EvidenceLocation] = Field(
        description="Location of each quote in source files. "
                    "Parallel to evidence_quotes."
    )

    # REQUIRED: Interpretation
    evidence_to_value_rationale: str = Field(
        description="Explanation of how the evidence implies this specific value. "
                    "E.g., 'Quote states at least 2 measurements, therefore minimum n=2'"
    )

    # OPTIONAL: Extraction methodology
    extraction_method: str = Field(
        default="llm",
        description="How evidence was extracted: 'llm', 'grep', 'manual', etc."
    )
    extraction_prompt: str | None = Field(
        default=None,
        description="Prompt used for LLM extraction (if applicable)"
    )

    # OPTIONAL: Quality indicators
    extraction_confidence: str | None = Field(
        default=None,
        description="Confidence level: 'high', 'medium', 'low'"
    )
    alternative_interpretations: list[str] | None = Field(
        default=None,
        description="Other possible values with rationales, if ambiguous"
    )

    # OPTIONAL: Verification
    pdf_verification: PDFVerification | None = Field(
        default=None,
        description="Cross-reference in PDF for validation"
    )

    # REQUIRED: Provenance
    extraction_date: str = Field(
        description="ISO date when extraction performed, e.g., '2026-01-20'"
    )
    extracted_by: str = Field(
        description="Agent/person who performed extraction, e.g., 'claude-sonnet-4.5'"
    )

    def to_citation_comment(self) -> str:
        """Generate citation comment for use in dataset code.

        Converts structured evidence back to the comment format
        currently used in dataset files.
        """
        lines = [f"# {self.variable_description}"]

        for quote, location in zip(self.evidence_quotes, self.evidence_locations):
            lines.append(f'# Quote: "{quote}"')
            location_str = f"{location.file_path}:{location.line_range}"
            if location.section:
                location_str += f", {location.section}"
            lines.append(f"# Source: {location_str}")

        if self.pdf_verification:
            lines.append(
                f"# Verified: PDF page {self.pdf_verification.page}, "
                f"{self.pdf_verification.section}"
            )

        lines.append(f"# Date extracted: {self.extraction_date}")
        lines.append(f"{self.variable_name} = {self.variable_value}")

        return "\n".join(lines)
```

#### Usage Example

**LLM Extraction with Structured Output:**

```python
# 1. Prepare LLM prompt with schema
from torchcell.datamodels.evidence import MetadataExtraction

schema = MetadataExtraction.model_json_schema()

prompt = f"""
Read the Costanzo 2016 paper (main text and SI) and extract information
about sample sizes for fitness measurements.

Output JSON matching this schema:
{json.dumps(schema, indent=2)}

Paper files available:
- papers/costanzo2016/main.mmd
- papers/costanzo2016/SI.mmd

Focus on finding:
- Default number of replicates for fitness measurements
- Wild-type control sample sizes
- Condition-specific variations (temperature, perturbation types)
"""

# 2. Get LLM response
llm_response = call_llm(prompt)

# 3. Validate with pydantic
extraction = MetadataExtraction.model_validate_json(llm_response)

# 4. Verify evidence locations
for quote, location in zip(extraction.evidence_quotes, extraction.evidence_locations):
    verify_quote_at_location(quote, location)

# 5. Generate code
citation_comment = extraction.to_citation_comment()
print(citation_comment)

# Output:
# # Default sample size for all fitness measurements
# # Quote: "All fitness measurements represent the mean of at least 2 independent measurements"
# # Source: main.mmd:1523-1524, Methods
# # Verified: PDF page 8, Methods section
# # Date extracted: 2026-01-20
# N_SAMPLES_DEFAULT = 2
```

**Storage Format:**

```bash
papers/costanzo2016/
├── main.pdf
├── main.mmd
├── SI.pdf
├── SI.mmd
└── evidence/
    ├── n_samples_default.json      # MetadataExtraction instance
    ├── n_samples_wt_26c.json
    └── n_samples_tsa.json
```

**Benefits Over Comment-Based Approach:**

1. **Validation:** Pydantic ensures all required fields present
2. **Versioning:** JSON diffs show exactly what changed
3. **Reproducibility:** Can re-verify locations programmatically
4. **LLM Integration:** Natural output format for LLM extractions
5. **Code Generation:** Comments generated from evidence, not manually written
6. **Testing:** Can check evidence validity in CI/CD

**Migration Path:**

- **Phase 1:** Use model to specify LLM output schema (current task)
- **Phase 2:** Store LLM outputs as JSON alongside code
- **Phase 3:** Build validation tools (verify locations still valid)
- **Phase 4:** Generate code from evidence files (evidence is source of truth)
- **Phase 5:** Convert existing comment-based extractions to structured evidence

**Not Required Now:**

This is a future design. For the current implementation of n_samples extraction:

- Continue using comment-based documentation
- Use the model as a guide for what information to include
- Consider LLM output, but validate manually
- Store evidence in comments until infrastructure built

### Future: Claude Code Skill

This SOP is designed to be converted into a Claude Code skill with the following invocation pattern:

```bash
/extract-paper-metadata \
  --dataset torchcell/datasets/scerevisiae/costanzo2016.py \
  --schema-field n_samples \
  --papers papers/costanzoGlobalGeneticInteraction2016/ \
  --output annotations
```

**Skill would:**

1. Read schema to understand field requirements
2. Read full paper (MMD + PDF)
3. Search for relevant evidence
4. Generate citations in required format
5. Produce code snippets for global constants
6. Validate citations (grep verification)
7. Run test loading of dataset
8. Output markdown report with all evidence

**Future extensions:**

- Multi-paper consensus extraction
- Automated contradiction detection
- Schema-aware hint generation
- Uncertainty quantification
- Version tracking for paper updates
