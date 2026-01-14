---
id: n4id60ag1x5tx18ffw1wi4n
title: Add_frontmatter
desc: ''
updated: 1768421737263
created: 1768420886760
---

## 2026.01.14 - Enhanced Frontmatter Script

Enhanced `add_frontmatter.py` to support both Python and bash scripts with improved functionality:

### Key Changes

1. **Universal Shebang Preservation**
   - Detects shebangs (`#!`) in both `.py` and `.sh` files
   - Inserts frontmatter after shebang line to maintain script executability
   - Works for any scripting language (Python, bash, Perl, Ruby, etc.)

2. **Clean Note Naming**
   - Removes file extensions (`.py`, `.sh`) from dendron note names
   - Creates cleaner references: `[[experiments.014.scripts.analyze]]` instead of `[[experiments.014.scripts.analyze.py]]`

3. **Smart Test File Logic**
   - Only includes "Test file:" line for Python files in library directory (`torchcell/`)
   - Omits test file references for experiment scripts and bash files
   - Reduces clutter in frontmatter for non-testable scripts

### Example Outputs

**Bash script with shebang:**

```bash
#!/bin/bash
# experiments/014-genes-enriched-at-extreme-tmi/scripts/014-genes-enriched-at-extreme-tmi
# [[experiments.014-genes-enriched-at-extreme-tmi.scripts.014-genes-enriched-at-extreme-tmi]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/014-genes-enriched-at-extreme-tmi/scripts/014-genes-enriched-at-extreme-tmi
```

**Python library file:**

```python
# torchcell/datamodels/schema
# [[torchcell.datamodels.schema]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/schema
# Test file: tests/torchcell/datamodels/test_schema.py
```

**Python experiment script:**

```python
#!/usr/bin/env python
# experiments/014-genes-enriched-at-extreme-tmi/scripts/analyze_extreme_interactions
# [[experiments.014-genes-enriched-at-extreme-tmi.scripts.analyze_extreme_interactions]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/014-genes-enriched-at-extreme-tmi/scripts/analyze_extreme_interactions
```
