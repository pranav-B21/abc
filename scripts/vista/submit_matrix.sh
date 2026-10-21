#!/bin/bash
# Max's 2x2 matrix (2026-10-14): pre-trained vs no pre-training x single-task vs multitask,
# all evaluated on put_relative, plus the put_relative data-scale sweep, on ONE fixed set of
# 100 distinct scenes (5 seeds x 20 worlds; seeds 20 apart so world seeds never overlap).
#
#   cell                          init                       data                      steps
#   pre-trained + single-task     200k parent                put_relative (12.1 h)     25k  (exists: frac100)
#   scratch(DINOv3) + single      DINOv3 vision, random DiT  put_relative              25k
#   scratch(random) + single      all random                 put_relative              25k
#   pre-trained + multitask       200k parent                mt6 (6 unseen tasks)      25k
#   scratch(DINOv3) + multitask   DINOv3 vision, random DiT  mt6                       25k
#   scratch(random) + multitask   all random                 mt6                       25k
#   75k check                     DINOv3 vision, random DiT  put_relative              75k (ckpt every 5k so a
#                                                                                          hang resumes; eval 25/50/75k)
#   data-scale sweep              200k parent                put_relative 10/25/50%    25k (resume, died 2026-10-08)
#   references (eval only)        parent zero-shot, released put_relative@25k
#
# Every cell uses the published fine-tune recipe (2 nodes x 90 = global batch 180, LR 1e-4,
# prefix 8) and differs only in init and train_sim/. Scratch cells normalize with the parent's
# stats (make_matrix_roots.py). Each training job is chained into its eval; a run whose final
# checkpoint already exists is not retrained, and finished evals are skipped, so re-running this
# script after a failure only resubmits what is missing.
#
# Prereqs (once): DINOv3 converted to $ABC_CACHE (convert_dinov3_hf.py), the 5 extra tasks
# downloaded to $SCRATCH/abc_ft/<task>, then:  python scripts/vista/make_matrix_roots.py
# Run from a LOGIN node, from the repo root:   bash scripts/vista/submit_matrix.sh
# ONLY="scratch_dino_single mt6_pretrained" restricts to named entries; DRY_RUN=1 prints only.
set -euo pipefail
cd /work/11138/pranavbelligundu/vista/abc

FT=/scratch/11138/pranavbelligundu/abc_ft
RUNS=/scratch/11138/pranavbelligundu/abc_runs
export SEEDS="20260511 20260531 20260551 20260571 20260591"
DRY_RUN=${DRY_RUN:-0}
ONLY=${ONLY:-}

want() { [ -z "$ONLY" ] || [[ " $ONLY " == *" $1 "* ]]; }
sub() {  # prints the job id (TACC's banner goes to stdout too, hence tail -1)
  if [ "$DRY_RUN" = 1 ]; then echo "DRY-$RANDOM"; echo "  [dry] sbatch $*" >&2; return; fi
  sbatch --parsable "$@" | tail -1
}

# name  task  tag  cache_root  init  vision  steps  ckpt_every
TRAIN=(
  "scratch_dino_single  put_relative  scratch_dino      put_relative_scratch_dino  scratch     dinov3  25000  5000"
  "scratch_rand_single  put_relative  scratch_rand      put_relative_scratch_rand  scratch     random  25000  5000"
  "mt6_pretrained       mt6           pretrained        mt6_pretrained             pretrained  -       25000  5000"
  "scratch_dino_mt6     mt6           scratch_dino      mt6_scratch_dino           scratch     dinov3  25000  5000"
  "scratch_rand_mt6     mt6           scratch_rand      mt6_scratch_rand           scratch     random  25000  5000"
  "scratch_dino_75k     put_relative  scratch_dino_75k  put_relative_scratch_dino  scratch     dinov3  75000  5000"
  "frac10               put_relative  frac10            put_relative_frac10        pretrained  -       25000  5000"
  "frac25               put_relative  frac25            put_relative_frac25        pretrained  -       25000  5000"
  "frac50               put_relative  frac50            put_relative_frac50        pretrained  -       25000  5000"
)

for row in "${TRAIN[@]}"; do
  read -r name task tag root init vision steps every <<<"$row"
  want "$name" || continue
  out=$RUNS/repro_${task}_${tag}
  [ -d "$FT/$root/train_sim" ] || { echo "[fail] $FT/$root/train_sim missing: run make_matrix_roots.py"; exit 1; }
  dep=()
  if [ -f "$out/$steps.pt" ]; then
    echo "$name: already trained ($out/$steps.pt)"
  else
    wall=12:00:00; [ "$steps" -gt 25000 ] && wall=20:00:00
    jid=$(TASK=$task TAG=$tag CACHE_ROOT=$FT/$root SKIP_PREPARE=1 INIT=$init VISION=${vision/-/dinov3} \
          TRAIN_STEPS=$steps CKPT_EVERY=$every WANDB_GROUP=matrix \
          sub -J "abc-mx-$name" -t "$wall" scripts/vista/sbatch_repro_finetune.sh)
    dep=(--dependency=afterok:$jid)
    echo "$name: train $jid"
  fi
  evsteps=""; [ "$steps" -gt 25000 ] && evsteps=$(seq 25000 25000 "$steps" | tr '\n' ' ')
  eid=$(RUN_DIR=$out TASK=put_relative STEPS="$evsteps" sub -J "abc-mx-ev-$name" "${dep[@]}" scripts/vista/sbatch_eval_run.sh)
  echo "$name: eval $eid"
done

# Already-trained models: re-evaluate on the 100 scenes (the 2 seeds already run are skipped).
for ref in repro_put_relative_frac100 put_relative_parent put_relative_released; do
  want "$ref" || continue
  [ -d "$RUNS/$ref" ] || { echo "[fail] missing $RUNS/$ref"; exit 1; }
  eid=$(RUN_DIR=$RUNS/$ref TASK=put_relative sub -J "abc-mx-ev-${ref#repro_}" scripts/vista/sbatch_eval_run.sh)
  echo "$ref: eval $eid"
done
