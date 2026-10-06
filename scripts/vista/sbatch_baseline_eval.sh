#!/bin/bash
# Reproduce the published per-task sim baselines on TACC Vista (one GH200).
#
# Protocol = abc_sim/README.md § Finetuned per-task checkpoints: the recommended
# released finetune per task, 20 randomized worlds x 2 seeds (20260511, 20260512),
# default RTC eval, default 236-chunk budget, sequential CPU-MuJoCo physics
# (no --parallel-worlds, which is a different engine and reads higher), and the
# exact prompt the checkpoint trained under (from the checkpoint manifest).
#
# Submit from a LOGIN node (sbatch is disabled on compute nodes), from the repo root:
#   sbatch scripts/vista/sbatch_baseline_eval.sh
# Override the task list or worlds:  TASKS="pour put_relative" NUM_WORLDS=50 sbatch ...
#
#SBATCH -J abc-baseline-eval
#SBATCH -p gh
#SBATCH -A ASC26032
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -o /scratch/11138/pranavbelligundu/abc_runs/baseline_eval_%j.log

set -uo pipefail
cd /work/11138/pranavbelligundu/vista/abc
source vista_env.sh

# manifest key | catalogue --task | published step | trained prompt
declare -A SIM_TASK=(
  [sim_put_the_plastic_bottles_in_the_bin]=put_plastic_bottles_in_bin
  [sim_load_the_plates_into_the_dish_rack]=load_plates_into_dish_rack
  [pour]=pouring
  [put_relative]=put_relative
)
declare -A STEP=(
  [sim_put_the_plastic_bottles_in_the_bin]=20000
  [sim_load_the_plates_into_the_dish_rack]=20000
  [pour]=25000
  [put_relative]=25000
)
declare -A PROMPT=(
  [sim_put_the_plastic_bottles_in_the_bin]="sim put the plastic bottles in the bin"
  [sim_load_the_plates_into_the_dish_rack]="sim load the plates into the dish rack"
  [pour]="sim pouring beads"
  [put_relative]="put relative"   # directive task: the env rewrites the live prompt each reset
)

TASKS=${TASKS:-"sim_put_the_plastic_bottles_in_the_bin sim_load_the_plates_into_the_dish_rack pour put_relative"}
SEEDS=${SEEDS:-"20260511 20260512"}
NUM_WORLDS=${NUM_WORLDS:-20}
OUT_ROOT=${OUT_ROOT:-/scratch/11138/pranavbelligundu/abc_runs/baseline_eval}
mkdir -p "$OUT_ROOT"

echo "[job] $(hostname) $(date -Is) tasks=[$TASKS] seeds=[$SEEDS] worlds=$NUM_WORLDS"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

for key in $TASKS; do
  ckpt="$ABC_CACHE/finetuned_sim/$key/${STEP[$key]}.pt"
  if [ ! -f "$ckpt" ]; then
    echo "[download] $key@${STEP[$key]}"
    uv run prepare.py --sim-checkpoint "$key@${STEP[$key]}" 2>&1 | tr '\r' '\n' | grep -vE '\[get\]|ETA' | tail -5
  fi
  [ -f "$ckpt" ] || { echo "[fail] missing $ckpt, skipping $key"; continue; }

  for seed in $SEEDS; do
    out="$OUT_ROOT/${key}_seed${seed}"
    if [ -f "$out/summary.json" ]; then echo "[skip] $out done"; continue; fi
    echo "[eval] $key seed=$seed -> $out  ($(date -Is))"
    args=(--checkpoint "$ckpt" --task "${SIM_TASK[$key]}" --prompt "${PROMPT[$key]}"
          --num-worlds "$NUM_WORLDS" --seed "$seed" --output-dir "$out"
          --save-video --video-every-n-actions 15)
    # --fast-inference (default: bf16 + torch.compile + CUDA graphs) is not yet verified on
    # aarch64; fall back to eager inference if it fails. Scores are unaffected either way.
    uv run eval_policy.py "${args[@]}" \
      || { echo "[retry] $key seed=$seed with --no-fast-inference"; uv run eval_policy.py "${args[@]}" --no-fast-inference; } \
      || echo "[fail] $key seed=$seed"
  done
done

# One table: per task, successes per seed vs. the published numbers.
uv run python - "$OUT_ROOT" <<'EOF'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
published = {  # abc_sim/README.md, seed A / seed B successes out of 20
    "sim_put_the_plastic_bottles_in_the_bin": "19 / 18",
    "sim_load_the_plates_into_the_dish_rack": "15 / 14 (12 on seed A under MuJoCo 3.8)",
    "pour": "13 / 16",
    "put_relative": "5 / 10",
}
print(f"\n{'task':42s} {'seed':>9s} {'success':>9s} {'max_prog':>9s}  published")
for d in sorted(root.glob("*_seed*")):
    f = d / "summary.json"
    if not f.exists():
        continue
    s = json.loads(f.read_text())
    key, seed = d.name.rsplit("_seed", 1)
    print(f"{key:42s} {seed:>9s} {s.get('num_success')}/{s.get('num_worlds', '?'):<6} "
          f"{s.get('mean_max_progress', float('nan')):9.2f}  {published.get(key, '')}")
EOF
uv run python scripts/vista/log_eval_wandb.py --root "$OUT_ROOT" \
  --name "baseline-eval-${SLURM_JOB_ID:-local}" --group baseline-released \
  || echo "[warn] wandb logging failed; results are still in $OUT_ROOT"
echo "[job] done $(date -Is)"
