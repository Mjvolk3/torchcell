---
id: 5stszwhd7wffjmq7p0fmunu
title: Mermaid_diagram
desc: ''
updated: 1769044449949
created: 1769024922916
---

## Overview

`torchcell/ontology/mermaid_diagram.py` generates visual Mermaid diagrams from the BioCypher schema YAML to validate Biolink ontology mappings.

## Key Features

- **Auto-detection**: Only regenerates diagrams when schema changes
- **Three node types**: Biolink classes (blue), direct Biolink usage (green), inherited torchcell entities (orange)
- **Ontology validation**: Shows class inheritance (`is_a`) and predicate inheritance on edge labels
- **Two orientations**: Horizontal (RL) and vertical (BT)

## Generated Outputs

- [[Horizontal diagram|torchcell.ontology.mermaid_diagram.horizontal]] - Right-to-left layout
- [[Vertical diagram|torchcell.ontology.mermaid_diagram.vertical]] - Bottom-to-top layout

## Usage

```bash
make tc-onto-mermaid
```

## Design Decisions

- **Experiments as continuants**: Changed from `activity` (process) to `information content entity` (data record) to correctly model stored experimental observations
- **Dataset membership**: Uses `part of` predicate (not `has output`) since experiments are components of dataset collections
