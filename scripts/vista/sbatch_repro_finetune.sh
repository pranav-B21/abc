#!/bin/bash
# Reproduce ONE published sim result end to end (Max's first check): finetune the
# released 200k multi-task DiT parent on a single task's sim episodes with the exact
# recipe behind every published per-task checkpoint (abc_sim/README.md § Finetuned
# per-task checkpoints): 25k steps, batch 90 per GPU on 2 GPUs (global 180), LR 1e-4,
# --flow.max-action-prefix 8, checkpoints every 5k. Then run
# sbatch_eval_run.sh on the checkpoints and compare with the published table.
#
# Vista has ONE GH200 per node, so 2 GPUs = 2 nodes (torchrun c10d rendezvous under srun).
# NODES=1 runs on one GPU instead, but that halves the global batch: not the published recipe.
#
# Submit from a LOGIN node, from the repo root:
#   sbatch scripts/vista/sbatch_repro_finetune.sh
#   TASK=pour sbatch scripts/vista/sbatch_repro_finetune.sh        # any --sim-data-list name
# Re-submitting the same TASK/TAG resumes from the run's last.pt (and the same wandb run).
#
#SBATCH -J abc-repro-ft
#SBATCH -p gh
#SBATCH -A ASC26032
#SBATCH -N 2
#SBATCH --ntasks-per-node 1
#SBATCH -t 12:00:00
#SBATCH -o /scratch/11138/pranavbelligundu/abc_runs/repro_ft_%j.log

set -euo pipefail
cd /work/11138/pranavbelligundu/vista/abc
source vista_env.sh

TASK=${TASK:-sim_put_the_plastic_bottles_in_the_bin}   # dataset name from prepare.py --sim-data-list
TAG=${TAG:-recipe}
TRAIN_STEPS=${TRAIN_STEPS:-25000}
RUN=repro_${TASK}_${TAG}
CACHE_ROOT=${CACHE_ROOT:-/scratch/11138/pranavbelligundu/abc_ft/$TASK}   # one task per cache root (sim_task preset)
OUT_DIR=/scratch/11138/pranavbelligundu/abc_runs/$RUN
mkdir -p "$CACHE_ROOT" "$OUT_DIR"

echo "[job] $(date -Is) nodes=$SLURM_JOB_NODELIST task=$TASK run=$RUN"

# 1. Data + parent (idempotent: prepare.py skips what is already present).
uv run prepare.py --sim-data "$TASK" --cache "$CACHE_ROOT" 2>&1 | tr '\r' '\n' | grep -vE '\[get\]|\[scan\]|ETA' | tail -4
uv run prepare.py --pretrained --cache "$CACHE_ROOT"       2>&1 | tr '\r' '\n' | grep -vE '\[get\]|\[scan\]|ETA' | tail -4

# 2. Resume if this run already has a checkpoint, else start from the parent.
if [ -f "$OUT_DIR/last.pt" ]; then
  START=(--resume-from "$OUT_DIR/last.pt"); echo "[resume] $OUT_DIR/last.pt"
else
  START=(--load-pretrained); echo "[start] from $CACHE_ROOT/abc_dit_xl_200k_model.pt"
fi

# 3. wandb: one run per RUN, resumed across re-submissions so the curve stays continuous.
export WANDB_PROJECT=abc-vista WANDB_NAME=$RUN WANDB_RUN_GROUP=repro-finetune
export WANDB_RUN_ID=$(echo "$RUN" | tr -c 'a-zA-Z0-9_\n-' '-') WANDB_RESUME=allow

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
NNODES=$SLURM_NNODES
echo "[launch] $NNODES node(s) x 1 GPU, rendezvous $HEAD_NODE:29500, global batch $((NNODES * 90))"

srun --ntasks-per-node 1 bash -c "
  source vista_env.sh
  uv run torchrun --nnodes $NNODES --nproc-per-node 1 \
    --rdzv-id $SLURM_JOB_ID --rdzv-backend c10d --rdzv-endpoint $HEAD_NODE:29500 \
    train.py \
      --cache-root $CACHE_ROOT --output-dir $OUT_DIR \
      ${START[*]} --mixture-preset sim_task \
      --flow.max-action-prefix 8 --train-steps $TRAIN_STEPS \
      --val-every 5000 --val-batches 40 --ckpt-every 5000 \
      --log-wandb --wandb-project abc-vista
"
echo "[job] done $(date -Is); checkpoints:"; ls -la "$OUT_DIR"
echo "next: RUN_DIR=$OUT_DIR TASK=$TASK sbatch scripts/vista/sbatch_eval_run.sh"
