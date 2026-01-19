## For Local Configs

`/Users/michaelvolk/Documents/projects/torchcell/CLAUDE.local.md`

- Includes how to run local python

## Programming Guide

- Do NOT ever use fallback mechanisms unless we clearly tell you to. This means minimize try except blocks, unnecessary conditionals, etc.

## Dendron Paths

We use dendron paths in markdown files to link between project notes. These are very useful for giving additional context.

File of these patters:

[[2025.05.10 - Inspecting Data in GoGraph|dendron://torchcell/torchcell.models.dcell#20250510---inspecting-data-in-gograph]]

Can be found here:

notes/torchcell.models.dcell.md

**The General Pattern**

`notes/` dir is skipped in path description as it is default location

Dendron encode from `torchcell/torchcell.models.dcell` to `notes/torchcell.models.dcell.md`

When I tell you to write some output to a file that is in `notes/` then typially you just need to append or modify, we don't want you messing up dendron frontmatter.

### Creating New Dendron Notes

**For NEW notes in `notes/` directory, use `dendron-cli` instead of Write tool to ensure proper frontmatter:**

```bash
dendron-cli note write --fname "note.path.here"
```

**Pattern:**

- File path: `notes/note.path.here.md`
- Dendron fname: `note.path.here` (no `notes/` prefix, no `.md` extension)

**Examples:**

```bash
# Creates notes/torchcell.datamodels.expression-schema-discussion.md
dendron-cli note write --fname "torchcell.datamodels.expression-schema-discussion"

# Creates notes/experiments.012-sameith-kemmeren.analysis.md
dendron-cli note write --fname "experiments.012-sameith-kemmeren.analysis"
```

After creating with `dendron-cli`, you can append content using the Edit tool.

**Important:** Only use `dendron-cli` for NEW notes. Use Edit tool for modifying existing Dendron notes (append mode to preserve frontmatter).

### Dendron Note Organization - Date-Stamped Sections

**Pattern for organizing note content over time:**

Use H2 headings with date prefix to track changes and additions:

```markdown
## YYYY.MM.DD - Initial Analysis

Content from this date...

## YYYY.MM.DD - Updated Implementation Plan

Revised content from later date...

## YYYY.MM.DD - Final Results

Latest updates...
```

**Guidelines:**

- **Date format:** `YYYY.MM.DD` (e.g., `2026.01.19`)
- **Heading level:** Always use H2 (`##`) for date-stamped sections
- **Title:** Add descriptive title after date separator (` - `)
- **Scope:** All content below a dated H2 belongs to that date context until the next H2
- **Revisions:** If a note block needs significant revision, create a new dated H2 section rather than editing the old one (preserves history). We can remove as we see fit.
- **Chronology** Time goes from top to bottom. Newer notes should be below older notes.

**Example from actual notes:**

```markdown
## 2025.08.29 - Model Update for Torch Compile Optimization

### Problem Analysis - Torch Compile Optimization

The current DCell implementation experiences significant graph breaks...

### Implementation Plan - ModuleList Conversion

1. Phase 1: Create dcell_opt.py
2. Phase 2: Test torch.compile compatibility

## 2025.09.02 - Plan Revision

### Problem Analysis - Sequential Processing Bottleneck

After profiling, discovered sequential processing is the real issue...

### Implementation Plan - Parallel Stratum Processing

Updated approach based on new findings...
```

**Benefits:**

- ✓ Clear temporal tracking of note evolution
- ✓ Preserves historical context (don't overwrite old dates)
- ✓ Easy to find when changes were made
- ✓ Multiple H2 sections per note = multiple work sessions documented

## Structure of Python Program Outputs

`notes/assets/scripts/add_frontmatter.py` is used with vscode cmd to apply frontmatter to python files. This creates a linked dendron note file that I will use for natural language notes associated with this file. Here is an example.

Front matter:

```python
# experiments/013-uncharacterized-genes/scripts/triple_interaction_enrichment_of_uncharacterized_genes.py
# [[experiments.013-uncharacterized-genes.scripts.triple_interaction_enrichment_of_uncharacterized_genes]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/013-uncharacterized-genes/scripts/triple_interaction_enrichment_of_uncharacterized_genes
```

- python file: `experiments/013-uncharacterized-genes/scripts/triple_interaction_enrichment_of_uncharacterized_genes.py`
- image output: `notes/assets/images/013-uncharacterized-genes/gene_interaction_counts_2025-12-21-06-26-21.png`
  - Note that the path matches experimental dir, but is name such so we don't have to go back dirs in relative path for associated note file.
  - If we are outputting images, they must be references in some dendron note file. The most logical place to md image referens is in the note file that is associated with the python script.
- data output: `experiments/013-uncharacterized-genes/results`

```bash
michaelvolk@M1-MV-6 torchcell % tree experiments/013-uncharacterized-genes/results
experiments/013-uncharacterized-genes/results
├── dubious_genes.json
├── gene_interaction_counts.csv
├── only_dubious_genes.json
├── only_uncharacterized_genes.json
├── uncharacterized_genes.json
├── uncharacterized_triple_interactions.csv
└── union_all_genes.json

1 directory, 7 files
```

- note file: `notes/experiments.013-uncharacterized-genes.scripts.triple_interaction_enrichment_of_uncharacterized_genes.md`

Note files can then be linked to from our current weekly note where we track todos and progress. For example in.

`notes/user.Mjvolk3.torchcell.tasks.weekly.2026.02.md`

We have the following, which helps we track work easily.

```markdown
## 2026.01.06

- [x] notes on [[Triple_interaction_enrichment_of_uncharacterized_genes|experiments.013-uncharacterized-genes.scripts.triple_interaction_enrichment_of_uncharacterized_genes]]
```

### Markdown Formatting

All markdown files in `notes/` are automatically formatted on save using the markdownlint VSCode extension. Configuration is stored in `torchcell.code-workspace` (lines 79-98).

**Markdownlint Rules (from workspace config):**

- **MD007**: `{ "indent": 2 }` - Use 2-space indentation for nested lists
- **MD008**: `false` - Don't require consistent header style
- **MD013**: `false` - No line length limit
- **MD014**: `false` - Dollar signs in shell commands allowed
- **MD018**: `false` - No space required after hash on atx style headers
- **MD025**: `false` - Multiple top-level headings allowed
- **MD026**: `false` - Trailing punctuation in headings allowed
- **MD029**: `false` - Ordered list item prefix doesn't need to be ordered
- **MD033**: `false` - Inline HTML allowed
- **MD036**: `false` - Emphasis used instead of heading allowed
- **MD040**: `false` - Fenced code blocks without language allowed
- **MD045**: `false` - Images allowed without alt text
- **MD050**: `false` - Strong style doesn't need to be consistent

Write clean, well-structured markdown and VSCode will handle consistency automatically. Try to abide by these rules so we don't have to manually save to reformat.

### Saving Images in Python

- All images should be saved in `ASSET_IMAGES_DIR`
- Do this by using `load_dotenv`
- Time stamp the images with by using torchcell/timestamp.py

The common pattern is `(osp.join(ASSET_IMAGES_DIR, f"{title}_{timestamp()}.png"))`

- We do this to iterate on the plots until we are satisfied with them. Then we will typically ask for `timestamp` to be removed once we are satisfied.

The common pattern is `(osp.join(ASSET_IMAGES_DIR, f"{title}}.png"))`

## When Creating or Designing new Experiments

These env vars will be read in in python with:

```python
from dotenv import load_dotenv
```

These are the most relevant dirs. (These are just example paths, they can change so just read in the variable.):

```bash
DATA_ROOT="/scratch/projects/torchcell-scratch"
ASSET_IMAGES_DIR="/home/michaelvolk/Documents/projects/torchcell/notes/assets/images"
EXPERIMENT_ROOT="/home/michaelvolk/Documents/projects/torchcell/experiments"
```

**`DATA_ROOT`**

- Where all of the large data is saved becuase this dir is typically mapped to larger storage.
- Datasets for training
- Datasets used for inference
- Inference results too are often put here because the outputs can be very large.

**`ASSET_IMAGES_DIR`**

- Where we want to save all image output
- Often convenient if there are more than one image for that given experiment to put them into dir with something like `osp.join(ASSET_IMAGES_DIR, f"006-kuzmin-tmi-inferece_1_{timestamp}")`. Often we like to start by default putting time stamps on images and we iterate to a decent idea, then if we want things to stably overwrite we will ask to remove `timestamp` after. So add by default.

**`EXPERIMENT_ROOT`**

This is what a typical experiment looks like. There is configuration in `conf/` that parameterizes experiements written in `scripts`. Slurm scripts are also in scripts and they typically have matching names with their experiments. Inside naming is also typically a designation of which slurm machine or partition a given scripts belongs to. We try to copy slurm scripts are typically numbered in a fashion that matches yamls to have a very transparaent experiment ecosystem. If we are running sweeps with optuna or wandb. we break from this pattern because we only need one config specifying the script - so we can many outputs for one script instead of one-to-one. Sometimes we inspect models here and dump `profile_results/`. `slurm/outputs` is resereved for concatenated slurm output and error. `queries/` holds the queries that the experiment used for the study.

```bash
/home/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi
├── conf
├── profiling_results
├── queries
├── results
├── scripts
└── slurm

6 directories, 3 files
```

Here is more depth on a more recent experiment. We cannot show for previous becuase it is too deep.

```bash
(base) michaelvolk@gilahyper torchcell % tree /home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi                       16:57
/home/michaelvolk/Documents/projects/torchcell/experiments/010-kuzmin-tmi
├── conf
│   ├── default.yaml
│   ├── equivariant_cell_graph_transformer_cabbi_000_inference_r0.yaml
│   ├── equivariant_cell_graph_transformer_cabbi_000_inference_r1.yaml
│   ├── equivariant_cell_graph_transformer_cabbi_000_inference_r2.yaml
│   ├── equivariant_cell_graph_transformer_cabbi_000.yaml
│   ├── equivariant_cell_graph_transformer_cabbi_002.yaml
│   ├── equivariant_cell_graph_transformer_mmli_001.yaml
│   └── equivariant_cell_graph_transformer_mmli_003.yaml
├── queries
│   └── 001_small_build.cql
├── results
├── scripts
│   ├── equivariant_cell_graph_transformer_inference_1.py
│   ├── equivariant_cell_graph_transformer.py
│   ├── gh_equivariant_cell_graph_transformer_inference_1_E006_M01.slurm
│   ├── gh_equivariant_cell_graph_transformer_inference_1_E006_M02.slurm
│   ├── gh_equivariant_cell_graph_transformer_inference_1_E006_M03.slurm
│   ├── gh_select_12_and_24_gene_top_triples.slurm
│   ├── igb_cabbi_equivariant_cell_graph_transformer-ddp_000.slurm
│   ├── igb_cabbi_equivariant_cell_graph_transformer-ddp_002.slurm
│   ├── igb_mmli_equivariant_cell_graph_transformer-ddp_001.slurm
│   ├── igb_mmli_equivariant_cell_graph_transformer-ddp_003.slurm
│   ├── inference_dataset_1.py
│   ├── __init__.py
│   ├── __pycache__
│   │   └── inference_dataset_1.cpython-313.pyc
│   ├── query.py
│   └── select_12_and_24_genes_top_triples.py
└── slurm
    └── output
        ├── 010-inference-M01_686.out
        ├── 010-inference-M02_690.out
        ├── 010-inference-M02_696.out
        ├── 010-inference-M02_698.out
        ├── 010-inference-M03_693.out
        ├── 010-inference-M03_697.out
        ├── 010-inference-M03_699.out
        └── 010-select-12-genes_694.out
```

## Troubleshooting  Deep Learning Experiments

Before investigating complex distributed computing issues (DDP sync, TorchMetrics configuration, Lightning hooks), **verify your data paths first**.

Dendron note source: `notes/experiments.010-kuzmin-tmi.false-torchmetrics-bug-bc-wrong-dataset-path.md`

## Running Experiment Scripts

All experiment scripts should be run from the torchcell root directory:

```bash
# From /Users/michaelvolk/Documents/projects/torchcell/
python experiments/<experiment-id>/scripts/<script-name>.py
```

This ensures consistent path resolution for `DATA_ROOT`, `ASSET_IMAGES_DIR`, `EXPERIMENT_ROOT`.
