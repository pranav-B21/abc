#!/bin/bash
# Fine-tuned vs not fine-tuned, on 500 scenes instead of 100 (2026-10-26).
# On the matrix's 100 scenes the 200k parent (no fine-tuning) and our 12 h put_relative fine-tune
# at 25k both score 33/100, too few scenes to see a gap under ~13 points. This evaluates both
# on 20 more seeds (400 more scenes), continuing the matrix's spacing of 20 so no world seed
# repeats: 20260511 + 20k for k = 0..24 (the first 5 are the matrix seeds, already done and
# skipped). Each model is split over two 1-node jobs of 10 seeds (~3 h each).
#
# Run from a LOGIN node, from the repo root:   bash scripts/vista/submit_scene500.sh
# Then:  uv run --no-project python scripts/vista/compare_scene500.py
# DRY_RUN=1 prints only.
set -euo pipefail
cd /work/11138/pranavbelligundu/vista/abc

RUNS=/scratch/11138/pranavbelligundu/abc_runs
DRY_RUN=${DRY_RUN:-0}
NEW=($(seq 20260611 20 20260991))   # 20 seeds; the matrix used 20260511..20260591
HALF1="${NEW[*]:0:10}"
HALF2="${NEW[*]:10:10}"

sub() {
  if [ "$DRY_RUN" = 1 ]; then echo "DRY-$RANDOM"; echo "  [dry] sbatch $*" >&2; return; fi
  sbatch --parsable "$@" | tail -1
}

# run dir  step  short name
for row in "put_relative_parent 0 parent" "repro_put_relative_frac100 25000 ft12h"; do
  read -r run step name <<<"$row"
  [ -f "$RUNS/$run/$step.pt" ] || { echo "[fail] missing $RUNS/$run/$step.pt"; exit 1; }
  for part in 1 2; do
    seeds=$HALF1; [ $part = 2 ] && seeds=$HALF2
    jid=$(RUN_DIR=$RUNS/$run TASK=put_relative STEPS=$step SEEDS="$seeds" \
          sub -J "abc-s500-$name-$part" -t 06:00:00 scripts/vista/sbatch_eval_run.sh)
    echo "$name part $part (seeds ${seeds%% *}..${seeds##* }): eval $jid"
  done
done
