#!/bin/bash
# Evaluate every step checkpoint of one finetune run (from sbatch_repro_finetune.sh) under
# the published protocol (20 worlds x seeds 20260511/20260512, default RTC, 236 chunks,
# sequential CPU-MuJoCo physics, the trained prompt), then push the success-rate-vs-step
# curve and rollout videos to wandb (project abc-vista).
#
# Submit from a LOGIN node, from the repo root:
#   RUN_DIR=/scratch/11138/pranavbelligundu/abc_runs/repro_sim_put_the_plastic_bottles_in_the_bin_recipe \
#   TASK=sim_put_the_plastic_bottles_in_the_bin sbatch scripts/vista/sbatch_eval_run.sh
# STEPS="20000 25000" restricts which checkpoints are evaluated. Finished evals are skipped.
#
#SBATCH -J abc-eval-run
#SBATCH -p gh
#SBATCH -A ASC26032
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -o /scratch/11138/pranavbelligundu/abc_runs/eval_run_%j.log

set -uo pipefail
cd /work/11138/pranavbelligundu/vista/abc
source vista_env.sh

: "${RUN_DIR:?set RUN_DIR to a finetune output dir}"
TASK=${TASK:-sim_put_the_plastic_bottles_in_the_bin}
SEEDS=${SEEDS:-"20260511 20260512"}
NUM_WORLDS=${NUM_WORLDS:-20}
OUT_ROOT=${OUT_ROOT:-$RUN_DIR/eval}

# Dataset name -> catalogue --task and the prompt sim_224 finetunes train under
# (the dataset task_name with underscores as spaces; see abc_sim/README.md).
declare -A SIM_TASK=(
  [sim_put_the_plastic_bottles_in_the_bin]=put_plastic_bottles_in_bin
  [sim_load_the_plates_into_the_dish_rack]=load_plates_into_dish_rack
  [sim_pouring_beads]=pouring
)
EVAL_TASK=${EVAL_TASK:-${SIM_TASK[$TASK]:-$TASK}}
PROMPT=${PROMPT:-${TASK//_/ }}
LABEL=$(basename "$RUN_DIR")

STEPS=${STEPS:-$(ls "$RUN_DIR" | sed -nE 's/^([0-9]+)\.pt$/\1/p' | sort -n | tr '\n' ' ')}
echo "[job] $(hostname) $(date -Is) run=$LABEL task=$EVAL_TASK prompt='$PROMPT' steps=[$STEPS]"
mkdir -p "$OUT_ROOT"

for step in $STEPS; do
  ckpt="$RUN_DIR/$step.pt"
  [ -f "$ckpt" ] || { echo "[skip] no $ckpt"; continue; }
  for seed in $SEEDS; do
    out="$OUT_ROOT/${LABEL}_step${step}_seed${seed}"
    [ -f "$out/summary.json" ] && { echo "[skip] $out done"; continue; }
    echo "[eval] step=$step seed=$seed ($(date -Is))"
    args=(--checkpoint "$ckpt" --task "$EVAL_TASK" --prompt "$PROMPT"
          --num-worlds "$NUM_WORLDS" --seed "$seed" --output-dir "$out"
          --save-video --video-every-n-actions 15)
    uv run eval_policy.py "${args[@]}" \
      || { echo "[retry] --no-fast-inference"; uv run eval_policy.py "${args[@]}" --no-fast-inference; } \
      || echo "[fail] step=$step seed=$seed"
  done
done

uv run python scripts/vista/log_eval_wandb.py --root "$OUT_ROOT" \
  --name "eval-$LABEL" --group repro-finetune \
  || echo "[warn] wandb logging failed; results are still in $OUT_ROOT"
echo "[job] done $(date -Is)"
