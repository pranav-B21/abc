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
CKPT_EVERY=${CKPT_EVERY:-5000}
# INIT=pretrained: start from the 200k parent (--load-pretrained), the published recipe.
# INIT=scratch: no --load-pretrained, so the DiT starts from random weights and the vision
#   encoder from <cache-root>/dinov3_vitb16_pretrain_lvd1689m.pth if present, else random
#   (abc_minimal/train_loop.py:232). VISION=dinov3|random states which one is intended and is
#   checked against the cache root below, because train.py silently falls back to random.
#   Needs <cache-root>/norm_stats.json (make_matrix_roots.py writes the parent's stats there).
INIT=${INIT:-pretrained}
VISION=${VISION:-dinov3}
RUN=repro_${TASK}_${TAG}
CACHE_ROOT=${CACHE_ROOT:-/scratch/11138/pranavbelligundu/abc_ft/$TASK}   # one task per cache root (sim_task preset)
OUT_DIR=/scratch/11138/pranavbelligundu/abc_runs/$RUN
mkdir -p "$CACHE_ROOT" "$OUT_DIR"

echo "[job] $(date -Is) nodes=$SLURM_JOB_NODELIST task=$TASK run=$RUN"

# 1. Data + parent (idempotent: prepare.py skips what is already present).
# SKIP_PREPARE=1 for a data-scale subset root (make_data_subsets.py): prepare.py would
# download the FULL task into it, undoing the subset.
if [ "${SKIP_PREPARE:-0}" = 1 ]; then
  echo "[data] SKIP_PREPARE: $(ls "$CACHE_ROOT/train_sim" | wc -l) train episodes in $CACHE_ROOT"
  if [ "$INIT" = pretrained ]; then
    [ -f "$CACHE_ROOT/abc_dit_xl_200k_model.pt" ] || { echo "[fail] no parent in $CACHE_ROOT"; exit 1; }
  fi
else
  uv run prepare.py --sim-data "$TASK" --cache "$CACHE_ROOT" 2>&1 | tr '\r' '\n' | grep -vE '\[get\]|\[scan\]|ETA' | tail -4
  uv run prepare.py --pretrained --cache "$CACHE_ROOT"       2>&1 | tr '\r' '\n' | grep -vE '\[get\]|\[scan\]|ETA' | tail -4
fi

# 2. Resume if this run already has a checkpoint, else start from the parent (or from scratch).
DINO="$CACHE_ROOT/dinov3_vitb16_pretrain_lvd1689m.pth"
if [ "$INIT" = scratch ]; then
  [ -f "$CACHE_ROOT/norm_stats.json" ] || { echo "[fail] INIT=scratch needs $CACHE_ROOT/norm_stats.json"; exit 1; }
  case "$VISION" in
    dinov3) [ -f "$DINO" ] || { echo "[fail] VISION=dinov3 but no $DINO (train.py would use a random encoder)"; exit 1; } ;;
    random) [ ! -e "$DINO" ] || { echo "[fail] VISION=random but $DINO exists (train.py would load it)"; exit 1; } ;;
    *) echo "[fail] VISION must be dinov3 or random"; exit 1 ;;
  esac
elif [ "$INIT" != pretrained ]; then
  echo "[fail] INIT must be pretrained or scratch"; exit 1
fi

# 3. wandb: one run per RUN, resumed across re-submissions so the curve stays continuous.
export WANDB_PROJECT=abc-vista WANDB_NAME=$RUN WANDB_RUN_GROUP=${WANDB_GROUP:-repro-finetune}
export WANDB_RUN_ID=$(echo "$RUN" | tr -c 'a-zA-Z0-9_\n-' '-') WANDB_RESUME=allow

HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
NNODES=$SLURM_NNODES

# 4. Train, retrying inside this allocation. 2-node runs hang on a single all-reduce until the
# 10-min NCCL watchdog kills them with no prior warning (3 of 4 sweep jobs on 2026-10-08, 6 of 9
# matrix jobs on 2026-10-19), so each attempt resumes from last.pt (the newest N.pt, i.e. at most
# CKPT_EVERY steps lost) under a fresh rendezvous id and port.
MAX_TRIES=${MAX_TRIES:-4}
for try in $(seq 1 "$MAX_TRIES"); do
  if [ -f "$OUT_DIR/last.pt" ]; then
    START=(--resume-from "$OUT_DIR/last.pt"); echo "[resume] $OUT_DIR/last.pt"
  elif [ "$INIT" = pretrained ]; then
    START=(--load-pretrained); echo "[start] from $CACHE_ROOT/abc_dit_xl_200k_model.pt"
  else
    START=(); echo "[start] from scratch: random DiT, vision encoder = $VISION"
  fi
  PORT=$((29500 + try))
  echo "[launch] try $try/$MAX_TRIES: $NNODES node(s) x 1 GPU, rendezvous $HEAD_NODE:$PORT, global batch $((NNODES * 90))"
  if srun --ntasks-per-node 1 bash -c "
    source vista_env.sh
    uv run torchrun --nnodes $NNODES --nproc-per-node 1 \
      --rdzv-id ${SLURM_JOB_ID}_$try --rdzv-backend c10d --rdzv-endpoint $HEAD_NODE:$PORT \
      train.py \
        --cache-root $CACHE_ROOT --output-dir $OUT_DIR \
        ${START[*]} --mixture-preset sim_task \
        --flow.max-action-prefix 8 --train-steps $TRAIN_STEPS \
        --val-every $CKPT_EVERY --val-batches 40 --ckpt-every $CKPT_EVERY \
        --log-wandb --wandb-project abc-vista
  "; then break; fi
  echo "[retry] try $try failed $(date -Is)"
  sleep 30
done
[ -f "$OUT_DIR/$TRAIN_STEPS.pt" ] || { echo "[fail] no $OUT_DIR/$TRAIN_STEPS.pt after $MAX_TRIES tries"; exit 1; }
echo "[job] done $(date -Is); checkpoints:"; ls -la "$OUT_DIR"
echo "next: RUN_DIR=$OUT_DIR TASK=$TASK sbatch scripts/vista/sbatch_eval_run.sh"
