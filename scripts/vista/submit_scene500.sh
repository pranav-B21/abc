#!/bin/bash
# Evaluate every compared put_relative model on ~460 scenes instead of 100 (2026-10-26, extended
# 2026-10-28). The 100-scene matrix could not resolve gaps under ~13 points: our 12 h fine-tune
# tied the parent at 33/100, but on 460 scenes it is 38.0% vs 30.7% (p = 0.015).
#
# Scenes: seeds 20260511 + 20k for k = 0..24, spacing 20 so no world seed repeats. The first 5
# are the matrix seeds (already done, skipped by sbatch_eval_run.sh). 20260631 and 20260851 are
# left out: put_relative cannot place non-overlapping objects for them, for every model. That
# leaves 23 seeds x 20 worlds = 460 scenes. Each model's 18 new seeds go to two 1-node jobs.
#
# Models whose checkpoint does not exist yet (still training) are skipped with a note, and models
# with every seed done are not resubmitted, so re-running this script later fills in only the gaps.
#
# Run from a LOGIN node, from the repo root:   bash scripts/vista/submit_scene500.sh
# Then:  uv run --no-project --with matplotlib python scripts/vista/matrix_summary.py --out ...
# ONLY="parent ft12h" restricts to named entries; DRY_RUN=1 prints only.
set -euo pipefail
cd /work/11138/pranavbelligundu/vista/abc

RUNS=/scratch/11138/pranavbelligundu/abc_runs
DRY_RUN=${DRY_RUN:-0}
ONLY=${ONLY:-}
NEW=()
for s in $(seq 20260611 20 20260991); do
  [ "$s" = 20260631 ] || [ "$s" = 20260851 ] || NEW+=("$s")
done                                 # 18 seeds; the matrix used 20260511..20260591
HALF1="${NEW[*]:0:9}"
HALF2="${NEW[*]:9:9}"

want() { [ -z "$ONLY" ] || [[ " $ONLY " == *" $1 "* ]]; }
sub() {
  if [ "$DRY_RUN" = 1 ]; then echo "DRY-$RANDOM"; echo "  [dry] sbatch $*" >&2; return; fi
  local id; id=$(sbatch --parsable "$@" | tail -1)
  # On compute nodes TACC's sbatch is a stub that prints a notice and returns 0.
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "[fail] sbatch returned no job id; run this from a LOGIN node" >&2; exit 1; }
  echo "$id"
}

# short name  run dir  steps (space-separated, quoted)  wall time per job
MODELS=(
  "parent        put_relative_parent                  0                     06:00:00"
  "ft12h         repro_put_relative_frac100           25000                 06:00:00"
  "released      put_relative_released                25000                 06:00:00"
  "mt6pre        repro_mt6_pretrained                 25000                 06:00:00"
  "dino1         repro_put_relative_scratch_dino      25000                 06:00:00"
  "rand1         repro_put_relative_scratch_rand      25000                 06:00:00"
  "dinomt        repro_mt6_scratch_dino               25000                 06:00:00"
  "randmt        repro_mt6_scratch_rand               25000                 06:00:00"
  "frac10        repro_put_relative_frac10            25000                 06:00:00"
  "frac25        repro_put_relative_frac25            25000                 06:00:00"
  "frac50        repro_put_relative_frac50            25000                 06:00:00"
  "dino75k       repro_put_relative_scratch_dino_75k  25000,50000,75000     14:00:00"
)

for row in "${MODELS[@]}"; do
  read -r name run steps wall <<<"$row"
  want "$name" || continue
  steps=${steps//,/ }
  for step in $steps; do
    [ -f "$RUNS/$run/$step.pt" ] || { echo "$name: no $run/$step.pt yet (still training?); skipped, re-run later"; continue 2; }
  done
  for part in 1 2; do
    seeds=$HALF1; [ $part = 2 ] && seeds=$HALF2
    todo=0
    for step in $steps; do for s in $seeds; do
      [ -f "$RUNS/$run/eval/${run}_step${step}_seed$s/summary.json" ] || todo=$((todo + 1))
    done; done
    [ "$todo" -gt 0 ] || { echo "$name part $part: all done"; continue; }
    jid=$(RUN_DIR=$RUNS/$run TASK=put_relative STEPS="$steps" SEEDS="$seeds" \
          sub -J "abc-s500-$name-$part" -t "$wall" scripts/vista/sbatch_eval_run.sh)
    echo "$name part $part ($todo evals, seeds ${seeds%% *}..${seeds##* }): eval $jid"
  done
done
