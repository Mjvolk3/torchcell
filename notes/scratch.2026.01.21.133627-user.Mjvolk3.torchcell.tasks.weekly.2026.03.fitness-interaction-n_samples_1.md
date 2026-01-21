---
id: dy3r6jdox6rh7iyv8efz59f
title: fitness-interaction-n_samples_1
desc: ''
updated: 1769027020468
created: 1769024232718
---

# Scratch - Fitness Interaction n_samples Implementation

## Quick File Access

### Implementation Files

```bash
# Main implementation
/Users/michaelvolk/Documents/projects/torchcell.worktrees/fitness-interaction-n_samples_0/torchcell/datamodels/schema.py
/Users/michaelvolk/Documents/projects/torchcell.worktrees/fitness-interaction-n_samples_0/torchcell/datasets/scerevisiae/costanzo2016.py

# Worktree setup
/Users/michaelvolk/Documents/projects/torchcell.worktrees/fitness-interaction-n_samples_0/scripts/setup-worktree.sh
/Users/michaelvolk/Documents/projects/torchcell.worktrees/fitness-interaction-n_samples_0/.vscode/settings.json
/Users/michaelvolk/Documents/projects/torchcell.worktrees/fitness-interaction-n_samples_0/.env.vscode
```

### Documentation Files

```bash
# Planning and implementation
/Users/michaelvolk/Documents/projects/torchcell.worktrees/fitness-interaction-n_samples_0/notes/user.Mjvolk3.torchcell.tasks.weekly.2026.04.fitness-interaction-n_samples.wip.md

# Methodology
/Users/michaelvolk/Documents/projects/torchcell.worktrees/fitness-interaction-n_samples_0/notes/user.Mjvolk3.torchcell.tasks.weekly.2026.04.nlp-data-enhancement.md
/Users/michaelvolk/Documents/projects/torchcell.worktrees/fitness-interaction-n_samples_0/notes/user.Mjvolk3.torchcell.tasks.weekly.2026.04.nlp-data-enhancement.sop.md
/Users/michaelvolk/Documents/projects/torchcell.worktrees/fitness-interaction-n_samples_0/notes/user.Mjvolk3.torchcell.tasks.weekly.2026.04.nlp-data-enhancement.codify-nl-evidence.md

# Original task context
/Users/michaelvolk/Documents/projects/torchcell.worktrees/fitness-interaction-n_samples_0/notes/user.Mjvolk3.torchcell.tasks.weekly.2026.03.fitness-interaction-n_samples.md
```

## Status: Phase 1 Complete (Costanzo 2016)

### ✅ Completed

- Schema updated with `fitness_se` and `n_samples`
- Global n_samples constants added with paper citations
- SmfCostanzo2016Dataset updated
- DmfCostanzo2016Dataset updated
- All tests passing

### 🔧 Worktree Setup Issue

- Current worktree needs `.env.vscode` created for PYTHONPATH
- Plan: Start fresh with `fitness-interaction-n_samples_1` worktree
- Use updated `setup-worktree.sh` for clean setup

### 📋 Next Steps

1. Commit current work to branch
2. Create new worktree: `fitness-interaction-n_samples_1`
3. Run `./scripts/setup-worktree.sh`
4. Continue with Phase 2: Kuzmin 2018

## Key Insights from Implementation

**n_samples from Costanzo 2016:**

- Double mutant fitness: n=4 (technical replicates per screen)
- Query single mutant: n=68 (17 screens × 4 colonies)
- Array single mutant: n=1400 (350 screens × 4 colonies)

**SE vs SD impact:**

- Double mutants: SE ≈ SD/2 (less precise, n=4)
- Query SMF: SE ≈ SD/8.2 (4.1× more precise, n=68)
- Array SMF: SE ≈ SD/37.4 (18.7× more precise, n=1400)

This asymmetry is intentional - single mutants are measured with high precision because they're reused in thousands of interaction calculations.
