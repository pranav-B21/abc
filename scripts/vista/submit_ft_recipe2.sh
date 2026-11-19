#!/bin/bash
# Fine-tuning recipe, round 2 (2026-11-19; Principled Latent Actions Stage 4).
#
# Round 1 (submit_ft_recipe.sh, RUNS.md 2026-11-19) on the 1.21 h subset: LR 1e-4 is worse than no fine-tuning
# from 5k on; LR 1e-5 ties no fine-tuning at 5k (32.4% vs 30.7%) then falls; vision LR x0.1 changes nothing.
# Within both runs val_loss was lowest where success was highest, but val ran only every 5k.
#
#   run            data      change from the published recipe        steps / ckpt   evaluated at
#   ft2_lr1e5_es   1.21 h    LR 1e-5                                  10k / 1k       every 1k (1k-10k)
#   ft2_lr3e6      1.21 h    LR 3e-6                                  25k / 5k       every 5k
#   ft2_lr1e5_3h   3.02 h    LR 1e-5                                  25k / 5k       5k, 15k, 25k
#   ft2_lr1e5_6h   6.05 h    LR 1e-5                                  25k / 5k       5k, 15k, 25k
#   ft2_lr1e5_12h  12.09 h   LR 1e-5                                  25k / 5k       5k, 15k, 25k
#
# Pre-registered stopping rule (fixed before any round-2 result): for each run, the checkpoint with the lowest
# val_loss (put_relative val_sim, logged at every checkpoint) is that recipe's result. Evaluating every
# ft2_lr1e5_es checkpoint on the 460 scenes also tests whether the rule picks well. The 3/6/12 h runs answer
# whether the gentler recipe lifts the whole data-scale curve (published recipe: 23.0 / 30.4 / 38.0%).
#
# Every eval uses the 460 matrix scenes (23 seeds x 20 worlds), one job per checkpoint, chained to training.
# Run from a LOGIN node, from the repo root:   bash scripts/vista/submit_ft_recipe2.sh
# Then:  uv run --no-sync python scripts/vista/compare_ft_recipes2.py
# ONLY="ft2_lr3e6" restricts to named entries; DRY_RUN=1 prints only. Re-running skips finished work.
# Disk: ~24 GB per checkpoint, ~720 GB in total; prune unselected checkpoints after the analysis.
set -euo pipefail
cd /work/11138/pranavbelligundu/vista/abc

FT=/scratch/11138/pranavbelligundu/abc_ft
RUNS=/scratch/11138/pranavbelligundu/abc_runs
DRY_RUN=${DRY_RUN:-0}
ONLY=${ONLY:-}
SEEDS=()
for s in $(seq 20260511 20 20260991); do
  [ "$s" = 20260631 ] || [ "$s" = 20260851 ] || SEEDS+=("$s")   # these two never generate a scene
done

want() { [ -z "$ONLY" ] || [[ " $ONLY " == *" $1 "* ]]; }
sub() {
  if [ "$DRY_RUN" = 1 ]; then echo "DRY-$RANDOM"; echo "  [dry] sbatch $*" >&2; return; fi
  local id; id=$(sbatch --parsable "$@" | tail -1)
  # On compute nodes TACC's sbatch is a stub that prints a notice and returns 0.
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "[fail] sbatch returned no job id; run this from a LOGIN node" >&2; exit 1; }
  echo "$id"
}

# name  cache root  steps  ckpt_every  eval steps (comma list)  extra train.py flags
RUNS_TABLE=(
  "ft2_lr1e5_es   put_relative_frac10  10000  1000  1000,2000,3000,4000,5000,6000,7000,8000,9000,10000  --optim.learning-rate 1e-5"
  "ft2_lr3e6      put_relative_frac10  25000  5000  5000,10000,15000,20000,25000                       --optim.learning-rate 3e-6"
  "ft2_lr1e5_3h   put_relative_frac25  25000  5000  5000,15000,25000                                   --optim.learning-rate 1e-5"
  "ft2_lr1e5_6h   put_relative_frac50  25000  5000  5000,15000,25000                                   --optim.learning-rate 1e-5"
  "ft2_lr1e5_12h  put_relative         25000  5000  5000,15000,25000                                   --optim.learning-rate 1e-5"
)

for row in "${RUNS_TABLE[@]}"; do
  read -r name root steps every evsteps extra <<<"$row"
  want "$name" || continue
  out=$RUNS/repro_put_relative_$name
  [ -d "$FT/$root/train_sim" ] || { echo "[fail] $FT/$root/train_sim missing"; exit 1; }
  dep=()
  if [ -f "$out/$steps.pt" ]; then
    echo "$name: already trained"
  else
    jid=$(TASK=put_relative TAG=$name CACHE_ROOT=$FT/$root SKIP_PREPARE=1 INIT=pretrained \
          TRAIN_STEPS=$steps CKPT_EVERY=$every WANDB_GROUP=ft-recipe2 EXTRA_ARGS="$extra" MAX_TRIES=8 \
          sub -J "abc-ft2-$name" -t 12:00:00 scripts/vista/sbatch_repro_finetune.sh)
    dep=(--dependency=afterok:$jid)
    echo "$name: train $jid ($root, $steps steps, ckpt every $every, $extra)"
  fi
  for step in ${evsteps//,/ }; do
    todo=0
    for s in "${SEEDS[@]}"; do
      [ -f "$out/eval/repro_put_relative_${name}_step${step}_seed$s/summary.json" ] || todo=$((todo + 1))
    done
    [ "$todo" -gt 0 ] || { echo "$name @$step: all 23 seeds done"; continue; }
    eid=$(RUN_DIR=$out TASK=put_relative STEPS=$step SEEDS="${SEEDS[*]}" \
          sub -J "abc-ft2-ev-$name-$((step / 1000))k" -t 08:00:00 "${dep[@]}" scripts/vista/sbatch_eval_run.sh)
    echo "$name @$step: eval $eid ($todo seeds)"
  done
done
