#!/bin/bash
# Data-scale sweep on put_relative (a sim task the 200k parent never trained on: its sidecar's
# sim_prompt_map lists only bottles, sweep, mug-up, dishrack, mug-rack).
#
# Four finetunes of the 200k parent, identical recipe (25k steps, 2 nodes x 90 = global 180,
# prefix 8, ckpt every 5k), differing ONLY in train_sim/: nested 10/25/50/100% subsets by hours
# (scripts/vista/make_data_subsets.py, seed 0; val_sim and the parent are shared symlinks).
# Each finetune is chained into sbatch_eval_run.sh. One extra eval job scores two references:
# the parent itself (0% task data) and the released put_relative@25k finetune.
#
# Eval: 20 worlds x seeds 20260511 and 20260531 = 40 DISTINCT scenes (seeds 20 apart; the
# published 20260511/20260512 pair shares 19 of 20 scenes), the same 40 for every arm so
# arms can be compared scene by scene.
#
# Run from a LOGIN node, from the repo root, after building the subsets:
#   python3 scripts/vista/make_data_subsets.py --full-root $SCRATCH/abc_ft/put_relative
#   bash scripts/vista/submit_put_relative_sweep.sh
# FRACS="10 25" restricts the arms. Re-running resumes training / skips finished evals.
set -euo pipefail
cd /work/11138/pranavbelligundu/vista/abc

FT=/scratch/11138/pranavbelligundu/abc_ft
RUNS=/scratch/11138/pranavbelligundu/abc_runs
FRACS=${FRACS:-"10 25 50 100"}
export SEEDS="20260511 20260531"

for f in $FRACS; do
  if [ "$f" = 100 ]; then root=$FT/put_relative; else root=$FT/put_relative_frac$f; fi
  [ -d "$root/train_sim" ] || { echo "[fail] missing $root/train_sim: build subsets first"; exit 1; }
  jid=$(TASK=put_relative TAG=frac$f CACHE_ROOT=$root SKIP_PREPARE=1 \
        sbatch --parsable -J abc-pr-ft$f scripts/vista/sbatch_repro_finetune.sh | tail -1)
  eid=$(RUN_DIR=$RUNS/repro_put_relative_frac$f TASK=put_relative \
        sbatch --parsable -J abc-pr-ev$f --dependency=afterok:$jid scripts/vista/sbatch_eval_run.sh | tail -1)
  echo "frac$f: finetune $jid -> eval $eid"
done

# References, as run dirs of step-named symlinks so sbatch_eval_run.sh treats them like any run.
mkdir -p $RUNS/put_relative_parent $RUNS/put_relative_released
ln -sfn $FT/put_relative/abc_dit_xl_200k_model.pt $RUNS/put_relative_parent/0.pt
ln -sfn /scratch/11138/pranavbelligundu/abc_cache/finetuned_sim/put_relative/25000.pt $RUNS/put_relative_released/25000.pt
for ref in parent released; do
  rid=$(RUN_DIR=$RUNS/put_relative_$ref TASK=put_relative \
        sbatch --parsable -J abc-pr-$ref -t 02:00:00 scripts/vista/sbatch_eval_run.sh | tail -1)
  echo "$ref: eval $rid"
done
