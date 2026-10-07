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

# --- Tests and coverage (plan.test-suite-buildout.2026.09.25) ---
# Plain `pytest tests/torchcell` is the CI contract: every expensive bucket (gpu, slow,
# data, neo4j, network, wandb) is opt-in by flag in tests/conftest.py. The coverage
# tools are not in the conda env by default; the cov* targets say how to install them
# and never install anything themselves.
PYTHON ?= $(HOME)/miniconda3/envs/torchcell/bin/python
PYTEST := $(PYTHON) -m pytest
IMPORT_ALL := tests/torchcell/test_import_all.py
COV_DIR := .coverage-out
COV_TOOLS_HINT := "coverage / diff-cover missing: pip install -r env/requirements_test.txt"

.PHONY: test test-ci test-fast test-import test-quality paired-tests paired-tests-strict diff-cov legacy-check legacy-table cov cov-html cov-gaps
test:
	@$(PYTEST) tests/torchcell
# What the GitHub runner sees: an empty real DATA_ROOT instead of the dev box's.
test-ci:
	@DATA_ROOT=$$(mktemp -d) $(PYTEST) tests/torchcell
test-fast:
	@$(PYTEST) tests/torchcell -x -q --deselect $(IMPORT_ALL)
test-import:
	@$(PYTEST) $(IMPORT_ALL) -q --durations=10
test-quality:
	@$(PYTHON) scripts/test_quality_check.py tests
paired-tests:
	@$(PYTHON) scripts/check_paired_tests.py --base origin/main
paired-tests-strict:
	@$(PYTHON) scripts/check_paired_tests.py --base origin/main --strict
legacy-check:
	@$(PYTHON) scripts/legacy_partition.py --check
legacy-table:
	@$(PYTHON) scripts/legacy_partition.py --table
# Behavioral coverage (import-all deselected) is the reported number; the import-only
# run is a separate data file so scripts/coverage_gaps.py can show it as its own column.
cov:
	@$(PYTHON) -c "import coverage" 2>/dev/null || { echo $(COV_TOOLS_HINT); exit 1; }
	@mkdir -p $(COV_DIR)
	@$(PYTHON) -m coverage run --data-file=$(COV_DIR)/.coverage -m pytest tests/torchcell --deselect $(IMPORT_ALL)
	@$(PYTHON) -m coverage report --data-file=$(COV_DIR)/.coverage
	@$(PYTHON) -m coverage json --data-file=$(COV_DIR)/.coverage -o $(COV_DIR)/coverage.json
	@$(PYTHON) -m coverage xml --data-file=$(COV_DIR)/.coverage -o $(COV_DIR)/coverage.xml
cov-html: cov
	@$(PYTHON) -m coverage html --data-file=$(COV_DIR)/.coverage -d $(COV_DIR)/html
	@echo "open $(COV_DIR)/html/index.html"
cov-gaps: cov
	@if [ -f $(IMPORT_ALL) ]; then \
	  $(PYTHON) -m coverage run --data-file=$(COV_DIR)/.coverage.import -m pytest $(IMPORT_ALL) -q; \
	  $(PYTHON) -m coverage json --data-file=$(COV_DIR)/.coverage.import -o $(COV_DIR)/coverage-import.json; \
	  $(PYTHON) scripts/coverage_gaps.py --after $(COV_DIR)/coverage.json --import-only $(COV_DIR)/coverage-import.json; \
	else \
	  $(PYTHON) scripts/coverage_gaps.py --after $(COV_DIR)/coverage.json; \
	fi
diff-cov: cov
	@command -v $(HOME)/miniconda3/envs/torchcell/bin/diff-cover >/dev/null || { echo $(COV_TOOLS_HINT); exit 1; }
	@$(HOME)/miniconda3/envs/torchcell/bin/diff-cover $(COV_DIR)/coverage.xml --compare-branch=origin/main --fail-under=80 \
	  --exclude 'torchcell/legacy/*' 'torchcell/scratch/*' 'torchcell/experiments/*'

# --- Ops: served knowledge-graph releases on every host + service health ---
# `make ops` is the panel to glance at from GilaHyper: which release each host serves
# (version, release id, commit index, datasets, nodes, aliases, faults), whether the
# hosts are in sync and how far behind main they are, then health probes (Browser
# seed, tc-lit, merge-queue loop, slurm, disks, Radiant). Read-only. See scripts/ops.sh.
.PHONY: ops ops-fast ops-health ops-releases
ops:
	@bash scripts/ops.sh status
ops-fast:
	@OPS_HOSTS=gilahyper OPS_TIMEOUT_SECONDS=2 bash scripts/ops.sh status
ops-health:
	@bash scripts/ops.sh health
ops-releases:
	@bash scripts/ops.sh releases
ops-sync:
	@bash scripts/ops.sh sync

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

# --- Benchmark service (tc-bench): staging and production from one branch ---
# Two compose projects with separate databases, archives, secrets and ports. Staging is
# rebuilt from the current tree; production runs the exact image staging ran, never a
# rebuild. See docker-compose.tc-bench.yml and scripts/tc_bench_deploy.sh. DRY_RUN=1
# prints the commands only.
.PHONY: bench-redeploy bench-promote-prod site-build site-publish
bench-redeploy:
	@FORCE="$(FORCE)" DRY_RUN="$(DRY_RUN)" bash scripts/tc_bench_redeploy.sh
bench-promote-prod:
	@FORCE="$(FORCE)" CONFIRM="$(CONFIRM)" DRY_RUN="$(DRY_RUN)" STAGING_TAG="$(STAGING_TAG)" \
	  bash scripts/tc_bench_promote_prod.sh
# Build one tier's site into website/build-<tier>/ from website/site.<tier>.env.
site-build:
	@case "$(TIER)" in staging|prod) ;; *) echo "usage: make site-build TIER=staging|prod"; exit 2;; esac
	@test -f website/site.$(TIER).env || { echo "missing website/site.$(TIER).env (copy site.$(TIER).env.example)"; exit 2; }
	@cd website && set -a && . ./site.$(TIER).env && set +a && npx docusaurus build --out-dir build-$(TIER)
# Build one tier's site and copy it into the directory the reverse proxy serves
# (SITE_PUBLISH_DIR in website/site.<tier>.env). TC_SITE_BUILD_DIR moves the build off
# the repository disk; see scripts/tc_site_publish.sh.
site-publish:
	@bash scripts/tc_site_publish.sh "$(TIER)"

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make test            - pytest tests/torchcell (the CI contract; buckets opt-in by flag)"
	@echo "  make test-ci         - same under an empty DATA_ROOT, as the GitHub runner sees it"
	@echo "  make test-fast       - -x -q, import-all deselected"
	@echo "  make test-import     - the import-all smoke test with durations"
	@echo "  make test-quality    - anti-padding lint on tests/"
	@echo "  make paired-tests    - new torchcell modules ship a test (-strict: changed too)"
	@echo "  make legacy-check    - the live tree is closed under imports; legacy-table lists the cluster"
	@echo "  make cov / cov-html / cov-gaps / diff-cov - coverage report, HTML, campaign table, diff gate"
	@echo "  make tc-onto         - Show schema → Biolink mappings (compact)"
	@echo "  make tc-onto-expand  - Show schema → Biolink mappings (detailed tree)"
	@echo "  make tc-onto-mermaid - Generate Mermaid diagrams from schema"
	@echo "  make ops             - Served KG releases on every host + service health"
	@echo "  make ops-health      - Health probes only; make ops-releases - the table only"
	@echo "  make ops-fast        - GilaHyper only, 2 s probes: the panel in a few seconds, no radiant rows"
	@echo "  make bench-redeploy  - Rebuild the benchmark STAGING tier from this tree and verify it"
	@echo "  make bench-promote-prod CONFIRM=1 - Start PRODUCTION on the image staging ran (bare: plan only)"
	@echo "  make site-build TIER=staging|prod - Build that tier's website into website/build-<tier>/"
	@echo "  make site-publish TIER=staging|prod - Build it and copy it where the proxy (tc-proxy) serves it"
	@echo "  make paper           - Build submission + editing + twocolumn PDFs"
	@echo "  make paper-submission/-editing/-twocolumn/-figproto - one PDF"
	@echo "  make paper-fig       - Force re-render all figures from draw.io + size/scale check"
	@echo "  make paper-figlimits - Build the figure-sizing reference card for collaborators"
	@echo "  make paper-sync      - Publish the curated subset to Overleaf (workshop -> Overleaf)"
	@echo "  make paper-pull      - Merge collaborator Overleaf edits back into the workshop"
	@echo "  make paper-flat      - Flatten to single .tex for Springer submission"
	@echo "  make paper-clean     - Remove generated paper PDFs"
