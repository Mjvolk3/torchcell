#!/bin/bash
# experiments/019-simb-multimodal/scripts/gh_pigment_transfer.slurm
# [[plan.cgt-metabolism.2026.07.25]]
# GilaHyper SINGLE-GPU run of the pigment metabolome-transfer contrast (Track A).
#
# Runs all four conditions sequentially in ONE job so the split and seed are byte-identical
# across them -- the 019 `_v2` lesson is that an unmatched split silently invalidates a
# Delta, and the Delta IS the result here:
#
#   A1 [betaxanthin]                 A2 [betaxanthin, mulleder19]    -> Delta_betaxanthin
#   A3 [beta_carotene]               A4 [beta_carotene, mulleder19]  -> Delta_beta_carotene
#
# HEADLINE: is Delta_betaxanthin > Delta_beta_carotene? Tyrosine (in Mulleder's 19) is
# betaxanthin's direct precursor and the Cachera cassette carries feedback-resistant
# ARO4^K229L / ARO7^G141S to push flux into it; beta-carotene's precursors (acetyl-CoA /
# GGPP) are not measured in Mulleder at all. So the negative arm is a designed control.
#
# ONE GPU on purpose: GPUs 1-3 are held by the concurrent 019 decoder sweep. The model is
# ~192 K params over ~4.9 k genotypes, so a single card is not the constraint.
#
# W&B OFFLINE: this is a correctness/fit validation run; the artifact is the results JSON.
# Sync later with the standard wandb_sync_agent_dirs pattern if the runs are worth keeping.
#
#   cd <worktree-root> && sbatch experiments/019-simb-multimodal/scripts/gh_pigment_transfer.slurm
#
#SBATCH -p main
#SBATCH --mem=64g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=019-pigment
#SBATCH --time=12:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output/%x_%j.out

set -euo pipefail

# Resolve the worktree root from where sbatch was submitted, so the branch's code is used
# and not the primary checkout's (the editable install otherwise wins -- see memory
# `worktree-build-needs-pythonpath`).
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell}"
mkdir -p /scratch/projects/torchcell-scratch/experiments/019-simb-multimodal/slurm/output

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export WANDB_MODE=offline

export PIGMENT_CONDITIONS="${PIGMENT_CONDITIONS:-A1,A2,A3,A4}"
export PIGMENT_MAX_EPOCHS="${PIGMENT_MAX_EPOCHS:-150}"

# --- Capacity / regularization -------------------------------------------------------
# Job 1325 (hidden 24, 3 layers, lr 1e-3, wd 1e-6) DIVERGED on A1: val loss rose
# monotonically 0.766 -> 2.053 over six epochs while TRAIN pearson reached 0.86 and val
# pearson collapsed to ~0.02. The pipeline is fine -- the model memorizes 4,235 training
# genotypes and generalizes nothing. That is the 019 `_v2` lesson repeating: on a few
# thousand samples this architecture needs to be small and decayed, not merely "not huge".
#
# Job 1326 (hidden 16, 2 layers, lr 3e-4, wd 1e-4) did NOT fix it either, and the agent's
# independent default-config run sat at val pearson ~-0.01 through epoch 15. So the problem
# is NOT capacity or step size -- it is STRUCTURAL, and `conf/cgt_embed_005.yaml` already
# names it:
#
#   "each gene is a FREE LEARNABLE VECTOR (6607 x d) with no content features. Each strain is
#    a single-gene deletion and the split is by strain, so a VAL gene has never appeared as a
#    perturbed gene during training. The cheapest solution is to write gene g's phenotype into
#    gene g's embedding -- perfect on train, undefined for an unseen gene."
#
# That is exactly train r=0.86 / val r~0.02. With `node_embeddings: []` a held-out gene's
# embedding never received a gradient, so generalization is impossible by construction and no
# amount of regularization recovers it. The fix is CONTENT features (so an unseen gene still
# has meaningful input) with the free identity table OFF (so the model cannot fall back on
# memorizing identities). Options, per the _005 sweep: codon_frequency (64d), calm (768d),
# fudt_upstream (768d), prot_T5_all (1024d), esm2_t33_650M_UR50D_all (1280d).
#
# CONTRACT: `run_pigment_conditions.py` reads the env var PIGMENT_OVERRIDES and splits it on
# COMMAS, then feeds the pieces to hydra `compose()`. It does NOT read sys.argv. Two traps,
# both of which cost a job here:
#   * job 1327 died with `OverrideParseException: mismatched input ' '` from passing a
#     space-containing value through `sbatch --export=...` (the shell strips the quotes and
#     the tail becomes a separate sbatch argument);
#   * job 1329 ran to completion but SILENTLY IGNORED the overrides -- they were passed as
#     space-separated positional args, which compose() never sees. The tell was a param count
#     (189,890) byte-identical to the run without them.
# So: comma-separated, exported as an env var, set HERE rather than via --export (which also
# splits on commas).
export PIGMENT_OVERRIDES="${PIGMENT_OVERRIDES:-cell_dataset.node_embeddings=[esm2_t33_650M_UR50D_all],model.learnable_embedding.enabled=false,model.hidden_channels=32,model.num_transformer_layers=2,model.dropout=0.1,regression_task.optimizer.lr=1e-3,regression_task.optimizer.weight_decay=1e-4}"

echo "=========================================================="
echo "host          : $(hostname)"
echo "job           : ${SLURM_JOB_ID:-none}  gpus=${CUDA_VISIBLE_DEVICES:-unset}"
echo "project root  : $PROJECT_ROOT"
echo "torchcell from: $(python -c 'import torchcell; print(torchcell.__file__)')"
echo "conditions    : $PIGMENT_CONDITIONS   max_epochs=$PIGMENT_MAX_EPOCHS"
echo "overrides     : $PIGMENT_OVERRIDES"
echo "=========================================================="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

# No positional overrides -- the driver reads PIGMENT_OVERRIDES from the environment.
python experiments/019-simb-multimodal/scripts/run_pigment_conditions.py

# Snapshot under the job id. The driver writes a fixed filename, so a concurrent
# interactive run of the same driver would otherwise clobber this job's report.
RESULTS=experiments/019-simb-multimodal/results
SNAPSHOT="$RESULTS/pigment_transfer_runs_slurm_${SLURM_JOB_ID:-manual}.json"
cp "$RESULTS/pigment_transfer_runs.json" "$SNAPSHOT"

echo "=========================================================="
echo "DONE. Report:   $RESULTS/pigment_transfer_runs.json"
echo "      Snapshot: $SNAPSHOT"
python - <<'PY'
import json, os
p = "experiments/019-simb-multimodal/results/pigment_transfer_runs.json"
d = json.load(open(p))
for name, v in d.get("conditions", {}).items():
    bits = [f"{name}: heads={v['active_heads']}"]
    for m in ("betaxanthin", "beta_carotene", "mulleder19"):
        pk, sp = v.get(f"{m}_pearson_peak"), v.get(f"{m}_spearman_peak")
        if pk is not None:
            bits.append(f"{m}: r={pk:.4f} rho={sp:.4f}" if sp is not None else f"{m}: r={pk:.4f}")
    print("  " + " | ".join(bits))
print("deltas:  ", json.dumps(d.get("deltas", {})))
print("headline:", json.dumps(d.get("headline", {})))
PY
echo "=========================================================="
