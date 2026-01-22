# Makefile for torchcell development commands
# Run with: make <target>
# Note: Requires torchcell conda environment to be activated

.PHONY: tc-onto
tc-onto:
	@python torchcell/ontology/tc_ontology.py 2>&1 | grep -v "^INFO --"

.PHONY: tc-onto-expand
tc-onto-expand:
	@python torchcell/ontology/tc_ontology.py --expand 2>&1 | grep -v "^INFO --"

.PHONY: tc-onto-mermaid
tc-onto-mermaid:
	@python torchcell/ontology/mermaid_diagram.py

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make tc-onto         - Show schema → Biolink mappings (compact)"
	@echo "  make tc-onto-expand  - Show schema → Biolink mappings (detailed tree)"
	@echo "  make tc-onto-mermaid - Generate Mermaid diagrams from schema"
