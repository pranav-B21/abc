#!/bin/bash
# First fine-tuning-recipe experiment (2026-11-02, Max: focus on better fine-tuning from the base model).
#
# Question: with little task data, can a gentler fine-tune stop the damage? On put_relative, the
# published recipe (LR 1e-4 for every parameter, 25k steps) on the 1.21 h subset scores 15.9% on
# 460 scenes, against 30.7% for the 200k parent with no fine-tuning (RUNS.md, 2026-10-31).
#
#   run                     data          change from the recipe
#   frac10 (exists)         1.21 h        none: the baseline
#   ft_lr1e5                1.21 h        --optim.learning-rate 1e-5 for every parameter
#                                         (the LR of ABC's real-robot fine-tunes, paper App. E)
#   ft_vis01                1.21 h        --optim.vision-lr-scale 0.1: DINOv3 at 1e-5, the rest at 1e-4
#                                         (the vision scale ABC used in pre-training)
#
# Every run keeps 25k steps with a checkpoint every 5k, so "fewer steps" is read off the checkpoints.
# Every checkpoint of all three runs is evaluated on the 460 matrix scenes (23 seeds x 20 worlds), one
# eval job per checkpoint chained after its training, so all points compare scene by scene against
# the parent and against each other.
#
# Run from a LOGIN node, from the repo root:   bash scripts/vista/submit_ft_recipe.sh
# Then:  uv run --no-project python scripts/vista/compare_ft_recipes.py
# ONLY="ft_lr1e5" restricts to named entries; DRY_RUN=1 prints only. Re-running skips finished work.
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
STEPS=(5000 10000 15000 20000 25000)

want() { [ -z "$ONLY" ] || [[ " $ONLY " == *" $1 "* ]]; }
sub() {
  if [ "$DRY_RUN" = 1 ]; then echo "DRY-$RANDOM"; echo "  [dry] sbatch $*" >&2; return; fi
  local id; id=$(sbatch --parsable "$@" | tail -1)
  # On compute nodes TACC's sbatch is a stub that prints a notice and returns 0.
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "[fail] sbatch returned no job id; run this from a LOGIN node" >&2; exit 1; }
  echo "$id"
}

# name  tag  extra train.py flags (quoted; "-" for none)
RUNS_TABLE=(
  "frac10    frac10    -"
  "ft_lr1e5  ft_lr1e5  --optim.learning-rate 1e-5"
  "ft_vis01  ft_vis01  --optim.vision-lr-scale 0.1"
)

for row in "${RUNS_TABLE[@]}"; do
  read -r name tag extra <<<"$row"
  want "$name" || continue
  [ "$extra" = "-" ] && extra=""
  out=$RUNS/repro_put_relative_$tag
  dep=()
  if [ -f "$out/25000.pt" ]; then
    echo "$name: already trained"
  else
    [ "$name" != frac10 ] || { echo "[fail] $out/25000.pt missing for the baseline"; exit 1; }
    jid=$(TASK=put_relative TAG=$tag CACHE_ROOT=$FT/put_relative_frac10 SKIP_PREPARE=1 INIT=pretrained \
          TRAIN_STEPS=25000 CKPT_EVERY=5000 WANDB_GROUP=ft-recipe EXTRA_ARGS="$extra" MAX_TRIES=8 \
          sub -J "abc-ftr-$name" -t 12:00:00 scripts/vista/sbatch_repro_finetune.sh)
    dep=(--dependency=afterok:$jid)
    echo "$name: train $jid ($extra)"
  fi
  for step in "${STEPS[@]}"; do
    todo=0
    for s in "${SEEDS[@]}"; do
      [ -f "$out/eval/repro_put_relative_${tag}_step${step}_seed$s/summary.json" ] || todo=$((todo + 1))
    done
    [ "$todo" -gt 0 ] || { echo "$name @$step: all 23 seeds done"; continue; }
    eid=$(RUN_DIR=$out TASK=put_relative STEPS=$step SEEDS="${SEEDS[*]}" \
          sub -J "abc-ftr-ev-$name-$((step / 1000))k" -t 08:00:00 "${dep[@]}" scripts/vista/sbatch_eval_run.sh)
    echo "$name @$step: eval $eid ($todo seeds)"
  done
done
