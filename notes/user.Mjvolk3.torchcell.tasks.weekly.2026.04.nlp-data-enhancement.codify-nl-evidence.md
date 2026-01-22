---
id: tlipzt3mmocb43rjk4e254x
title: codify-nl-evidence
desc: 'Codifying natural language evidence as structured data'
updated: 1769024015366
created: 1768947439725
---

## 2026.01.20 - Codifying Natural Language Evidence for Dataset Metadata

### The Vision: From Comments to Data

**Current state:**

```python
# Quote: "All fitness measurements represent the mean of at least
#         2 independent measurements"
# Source: costanzo2016.mmd, Lines 1523-1524
# Verified: costanzo2016.pdf, Page 8, Methods
# Date extracted: 2026-01-20
N_SAMPLES_DEFAULT = 2
```

**Future state:**

```python
# Evidence defined as structured data in separate file
# Can validate, version, diff, and regenerate code from evidence

from torchcell.datamodels.evidence import MetadataExtraction

EVIDENCE = MetadataExtraction(
    variable_name="N_SAMPLES_DEFAULT",
    variable_value=2,
    variable_description="Default sample size for all fitness measurements",
    paper_sources=["papers/costanzo2016/main.mmd", "papers/costanzo2016/SI.mmd"],
    evidence_quotes=[
        "All fitness measurements represent the mean of at least 2 independent measurements"
    ],
    evidence_locations=["main.mmd:1523-1524"],
    extraction_method="llm",
    extraction_prompt="Read methods sections and extract sample size information",
    evidence_to_value_rationale="Explicitly states 'at least 2 independent measurements', minimum n=2",
    pdf_verification={"page": 8, "section": "Methods"},
    extraction_date="2026-01-20",
    extracted_by="claude-sonnet-4.5"
)

# Code generation from evidence
N_SAMPLES_DEFAULT = EVIDENCE.variable_value  # Type-safe, traceable
```

### Why Structured Evidence?

**Problem with comment-based documentation:**

- ❌ Cannot programmatically verify evidence still exists at cited location
- ❌ Cannot detect when paper updates invalidate extraction
- ❌ Cannot diff evidence changes across dataset versions
- ❌ Cannot batch re-extract when methodology improves
- ❌ LLMs cannot easily consume/produce structured comments
- ❌ No type safety on variable values

**Benefits of pydantic-modeled evidence:**

- ✅ **Validation:** Ensure all required fields present (quotes, sources, dates)
- ✅ **Reproducibility:** Structured data enables automated verification
- ✅ **Versioning:** Git diffs show exactly what evidence changed
- ✅ **LLM Integration:** LLMs can output structured data directly
- ✅ **Provenance:** Full chain from paper → evidence → code
- ✅ **Testing:** Can verify evidence locations still valid
- ✅ **Migration:** Can regenerate code when schema changes

### Use Cases

**1. LLM-Based Extraction (Primary Use Case)**

```python
# Prompt LLM to output MetadataExtraction instances
prompt = """
Read the Costanzo 2016 paper and extract sample size information.
Output as MetadataExtraction JSON with fields:
- variable_name: Python variable name
- variable_value: Numeric value
- evidence_quotes: Exact quotes from paper
- evidence_locations: file:line format
- evidence_to_value_rationale: Why quote → value
"""

# LLM outputs structured JSON, we validate with pydantic
extraction = MetadataExtraction.model_validate_json(llm_response)

# Generate code from validated extraction
generate_dataset_code(extraction)
```

**2. Evidence Validation**

```python
# Check if evidence still exists at cited location
def validate_evidence(extraction: MetadataExtraction) -> bool:
    for quote, location in zip(extraction.evidence_quotes, extraction.evidence_locations):
        file_path, line_range = parse_location(location)
        actual_text = read_lines(file_path, line_range)
        if quote not in actual_text:
            log.warning(f"Evidence mismatch: {location}")
            return False
    return True

# Run validation on all extractions
for extraction in all_extractions:
    validate_evidence(extraction)
```

**3. Evidence Diffing**

```python
# Compare evidence across dataset versions
old_evidence = load_evidence("v1.0/costanzo2016_evidence.json")
new_evidence = load_evidence("v2.0/costanzo2016_evidence.json")

diff = compare_extractions(old_evidence, new_evidence)
# Output: "N_SAMPLES_DEFAULT changed from 2 to 4 because new quote found in SI"
```

**4. Batch Re-extraction**

```python
# When extraction methodology improves, re-extract all variables
papers = ["costanzo2016", "kuzmin2018", "kuzmin2020"]
schema_fields = ["n_samples", "temperature_conditions", "replicate_types"]

for paper in papers:
    for field in schema_fields:
        new_extraction = extract_with_improved_llm(paper, field)
        old_extraction = load_existing_extraction(paper, field)

        if new_extraction != old_extraction:
            show_diff_for_human_review(old_extraction, new_extraction)
```

### Design Principles

**1. LLM-First Design**

Since we're using LLMs for extraction, the model should be optimized for LLM output:

- Clear field names that LLMs understand
- Minimal required fields (LLMs can fail on complex schemas)
- Flexible evidence format (quotes can be short or long)
- Built-in rationale field (LLMs should explain their reasoning)

**2. Human-Readable**

Evidence files should be readable without code:

```json
{
  "variable_name": "N_SAMPLES_DEFAULT",
  "variable_value": 2,
  "evidence_quotes": [
    "All fitness measurements represent the mean of at least 2 independent measurements"
  ],
  "evidence_locations": ["main.mmd:1523-1524"],
  "extraction_date": "2026-01-20"
}
```

**3. Incremental Adoption**

Don't require rewriting all existing code:

- Start with new extractions using structured evidence
- Legacy comment-based extractions can coexist
- Gradually migrate as datasets are updated
- Tool to convert comments → structured evidence

**4. Verification Built-In**

Evidence should enable automated checking:

- Line number verification (does quote exist at location?)
- PDF cross-reference (optional but recommended)
- Multi-method agreement (grep + LLM should agree)
- Temporal tracking (when was this extracted?)

### Relationship to Current Workflow

**Current (comments in code):**

```python
# Manual process:
# 1. Human/LLM reads paper
# 2. Finds evidence
# 3. Writes comment with quote + location
# 4. Sets variable value
# 5. Git commit

# Problem: Evidence is unstructured, unvalidated
```

**Future (structured evidence):**

```python
# Semi-automated process:
# 1. LLM reads paper, outputs MetadataExtraction JSON
# 2. Pydantic validates structure
# 3. Automated checks verify locations
# 4. Human reviews evidence → value mapping
# 5. Code generator creates dataset code from evidence
# 6. Git commit of both evidence.json and generated code

# Benefit: Evidence is validated, versioned, reproducible
```

### Implementation Strategy

**Phase 1: Define Models** (Current Task)

- Create `MetadataExtraction` pydantic model
- Keep it simple - only essential fields
- Add to SOP as future direction
- Comment-out in actual code (not using yet)

**Phase 2: Pilot Extraction** (Next Task)

- Use model as LLM output schema for Costanzo 2016
- Validate LLMs can produce correct JSON
- Store evidence in `papers/costanzo2016/evidence.json`
- Still generate comment-based code (backward compatible)

**Phase 3: Validation Tools**

- Script to verify evidence locations
- Tool to diff evidence files
- Automated checks in CI/CD

**Phase 4: Code Generation**

- Generate dataset code from evidence files
- Evidence becomes source of truth
- Comments generated from evidence (not manually written)

**Phase 5: Full Migration**

- Convert existing comment-based extractions to structured evidence
- All datasets use evidence-driven workflow
- Claude Code skill for extraction

### Related Documents

- **SOP:** [[nlp-data-enhancement.sop|user.Mjvolk3.torchcell.tasks.weekly.2026.04.nlp-data-enhancement.sop]]
- **Implementation:** [[fitness-interaction-n_samples.wip|user.Mjvolk3.torchcell.tasks.weekly.2026.04.fitness-interaction-n_samples.wip]]
- **Motivation:** [[nlp-data-enhancement|user.Mjvolk3.torchcell.tasks.weekly.2026.04.nlp-data-enhancement]]

### Key Insight

The pydantic model isn't just documentation - it's a **contract between humans and LLMs**. When we ask an LLM to extract metadata, we're asking it to fill out a structured form. The model defines that form, making extraction:

- More reliable (validated structure)
- More reproducible (defined fields)
- More maintainable (versioned data)
- More auditable (evidence chain)

By codifying evidence, we transform "LLM extraction" from an art into an engineering discipline.
