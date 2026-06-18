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

# --- Manuscript (paper/nature-biotech) passthrough targets ---
.PHONY: paper paper-submission paper-editing paper-twocolumn paper-figproto paper-figlimits paper-figures paper-fig paper-flat paper-clean paper-sync paper-pull
paper:
	@$(MAKE) -C paper/nature-biotech paper
paper-submission:
	@$(MAKE) -C paper/nature-biotech submission
paper-editing:
	@$(MAKE) -C paper/nature-biotech editing
paper-twocolumn:
	@$(MAKE) -C paper/nature-biotech twocolumn
paper-figproto:
	@$(MAKE) -C paper/nature-biotech figproto
paper-figlimits:
	@$(MAKE) -C paper/nature-biotech figlimits
paper-figures:
	@$(MAKE) -C paper/nature-biotech figures
paper-fig:
	@$(MAKE) -C paper/nature-biotech fig
paper-flat:
	@$(MAKE) -C paper/nature-biotech flat
paper-clean:
	@$(MAKE) -C paper/nature-biotech clean
paper-sync:
	@bash paper/nature-biotech/sync-overleaf.sh
paper-pull:
	@bash paper/nature-biotech/paper-pull.sh

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make tc-onto         - Show schema → Biolink mappings (compact)"
	@echo "  make tc-onto-expand  - Show schema → Biolink mappings (detailed tree)"
	@echo "  make tc-onto-mermaid - Generate Mermaid diagrams from schema"
	@echo "  make paper           - Build submission + editing + twocolumn PDFs"
	@echo "  make paper-submission/-editing/-twocolumn/-figproto - one PDF"
	@echo "  make paper-fig       - Force re-render all figures from draw.io + size/scale check"
	@echo "  make paper-figlimits - Build the figure-sizing reference card for collaborators"
	@echo "  make paper-sync      - Publish the curated subset to Overleaf (workshop -> Overleaf)"
	@echo "  make paper-pull      - Merge collaborator Overleaf edits back into the workshop"
	@echo "  make paper-flat      - Flatten to single .tex for Springer submission"
	@echo "  make paper-clean     - Remove generated paper PDFs"
