# RUNS — ABC on TACC Vista

One entry per SLURM job or interactive run. Each entry gives the hostname, the absolute paths
of its outputs, what it ran and what came out. Chronological narrative is in `JOURNAL.md`.

All paths are on TACC Vista (`vista.tacc.utexas.edu`), allocation `ASC26032`, partition `gh`
(1× NVIDIA GH200 96 GB per node, aarch64). `$SCRATCH` = `/scratch/11138/pranavbelligundu`.
**`$SCRATCH` purges unaccessed files.** Copy anything that must survive.

Code: `/work/11138/pranavbelligundu/vista/abc`, fork `github.com/pranav-B21/abc`, branch
`vista-aarch64` (renamed `pranav_dev` 2026-10-18), commit `b214527` for the bottles runs; `c0ff182` for the put_relative sweep.

---

## Shared reference: model, data, code, settings

Every fine-tune and eval below uses this setup unless its entry says otherwise.

### Model

We did not design a model. We used ABC-DiT, the released policy from the ABC paper
(`amazon-far/abc`, arXiv 2606.27375), starting from its public multi-task checkpoint.

| | |
|---|---|
| Policy | ABC-DiT (diffusion transformer, "XL" size), 2,015,695,908 parameters |
| Starting checkpoint | `abc_dit_xl_200k_model.pt` (sha256 `79754ece…9f51ba7`), step 200k of a run configured for 500k |
| What the parent saw | mixture `xdof_csf_3_5k_may20_sim0514_hours_weighted`: 3,553.2 h real teleop + 98.47 h sim across 5 sim tasks (put-bottles, sweep paper scraps, turn mug, load dish rack, hang mug) |
| Transformer | hidden 1536, depth 32, 24 heads, MLP ratio 4 |
| Vision encoder | DINOv3 ViT-B/16 (768-dim, 12 layers), attention-pooled to 12 tokens per camera |
| Language encoder | CLIP text encoder, 512-dim task embedding from the prompt |
| Inputs | 3 cameras (top, left, right) at 224×168, plus 14-dim robot state |
| Output | chunk of 30 future actions, 14-dim (two YAM arms + grippers) |
| Objective | rectified flow matching; 10 denoising steps at inference |
| Conditioning | action-prefix conditioning, up to 8 already-committed actions |

Unverified: the exact per-arm split of the 14 dims, and whether actions are absolute joint
targets or deltas (see `abc_minimal/dataloader.py`).

### Code

Model, training loop and evaluator are upstream `amazon-far/abc`, unchanged. The fork adds 679
lines in 7 files (`git diff --stat upstream/main...HEAD`), none in model/train/eval code:

| File | Purpose |
|---|---|
| `pyproject.toml` | aarch64 pin `torchcodec==0.11.0+cu128` (upstream's `+cpu` has no ARM wheel) |
| `vista_env.sh` | conda-forge FFmpeg on PATH, caches on `$SCRATCH` |
| `scripts/vista/sbatch_repro_finetune.sh` | download data + parent, 2-node torchrun fine-tune |
| `scripts/vista/sbatch_eval_run.sh` | eval every `<step>.pt` of a run |
| `scripts/vista/sbatch_baseline_eval.sh` | eval the paper's released per-task checkpoints |
| `scripts/vista/log_eval_wandb.py` | push eval summaries + videos to wandb |
| `CLAUDE.md` | Vista setup notes |

Software: torch `2.11.0+cu128`, torchcodec `0.11.0+cu128`, mujoco `3.8.1`, warp-lang `1.13.0`,
mujoco-warp `3.8.0.3`, Python 3.12.15 (uv).

### Training settings (published per-task fine-tune recipe)

| Setting | Value |
|---|---|
| Start | `--load-pretrained` (parent weights, fresh optimizer) |
| Data | `--mixture-preset sim_task` (one task's `train_sim/`, weight 1.0) |
| Steps | 25,000 |
| Batch | 90/GPU × 2 GPUs (2 nodes) = 180 global |
| Optimizer | AdamW, β (0.9, 0.95), eps 1e-8, weight decay 0.01, grad clip 10 |
| LR | 1e-4, 1,000-step linear warmup, then constant (repo default) |
| Flow | max action prefix 8, state masked on 10% of samples |
| Precision | DINOv3 bf16 autocast, rest fp32; `torch.compile(fullgraph=True)` |
| Norm stats | embedded in the parent checkpoint (inputs scaled as in pretraining) |
| Checkpoints / val | every 5k steps; val on 40 batches (3,600 frames/rank) of `val_sim` |

Command (run on each node under `srun`):

    torchrun --nnodes 2 --nproc-per-node 1 --rdzv-backend c10d --rdzv-endpoint <head>:29500 \
      train.py --cache-root $SCRATCH/abc_ft/<task> --output-dir $SCRATCH/abc_runs/<run> \
      --load-pretrained --mixture-preset sim_task --flow.max-action-prefix 8 \
      --train-steps 25000 --val-every 5000 --val-batches 40 --ckpt-every 5000 \
      --log-wandb --wandb-project abc-vista

### Eval protocol (matches the paper's published per-task numbers)

| Setting | Value |
|---|---|
| Physics | sequential CPU MuJoCo (batched MJWarp `--parallel-worlds` reads higher; not used) |
| Inference | fast inference (bf16, torch.compile, CUDA graphs), RTC on (prefix 4, lead 4) |
| Execution | 15 of each 30-action chunk executed per replan |
| Episode length | 236 chunks |
| Scenes | 20 worlds per seed, seeds 20260511 and 20260512 |
| Prompt | the prompt the checkpoint trained under |
| Success | task evaluator's strict success (bottles: all bottles in bin) |
| Partial credit | `mean_max_progress` (bottles: fraction of bottles placed) |

**Seed overlap:** world *i* of seed *s* uses world seed *s + i*, so seeds 20260511 and
20260512 share 19 of 20 scenes. "40 trials" ≈ 21 distinct scenes, each run about twice. The
policy is stochastic, so outcomes on shared scenes differed on 6 of 19. Comparable to the
paper (same seeds), but the effective sample is smaller than 40.

Command:

    eval_policy.py --checkpoint <ckpt>.pt --task <sim task> --prompt "<training prompt>" \
      --num-worlds 20 --seed <seed> --output-dir <dir> --save-video --video-every-n-actions 15

Each eval dir holds `summary.json` (aggregate + per-world results incl. randomization) and
`world_000.mp4` … `world_019.mp4`.

### Data: `sim_put_the_plastic_bottles_in_the_bin` (v3)

| | |
|---|---|
| Source | `https://abc-data.timehorizons.org/dataset/sim_224/sim_put_the_plastic_bottles_in_the_bin_v3.tar` (sha256 `af7701e9…71fe5d5a`) |
| Size | 5,588 train / 287 val episodes, 5,304,463 frames, 49.12 h, 14.1 GB |
| Generated | scripted demos in the ABC MuJoCo sim, rendered with MJWarp; source batches `sim_tasks_20260522_…` and `sim_new_tasks_20260720_20260803_…` |
| Per episode | `combined_camera-images-rgb.mp4` (3 views), `states_actions.bin`, `scene_assembled.xml`, `randomization.json`, `initial_qpos.npy`, `scene_qpos.npy`, `episode_metadata.json` |
| Rate / resolution | 30 fps, 224×168 |
| Training prompt | `sim put the plastic bottles in the bin` |
| Usable training frames | 4,875,870 |
| Local copy | `/scratch/11138/pranavbelligundu/abc_ft/sim_put_the_plastic_bottles_in_the_bin/` (with the parent `.pt` + `.json` sidecar) |

All of the task's data was used; no filtering.

---

## smoke_eval — README smoke test
- **Host:** `c642-012.vista.tacc.utexas.edu` (idev), 2026-10-06 ~11:34 CDT
- **Ran:** `eval_policy.py --checkpoint $ABC_CACHE/bottles_75k.pt --num-worlds 1 --num-chunks 2 --no-fast-inference --save-video`
- **Output:** `/work/11138/pranavbelligundu/vista/abc/outputs/sim_eval_put_plastic_bottles_in_bin/`
- **Result:** pipeline runs end to end (~2.5 min); `success=False` expected at 2 chunks; all 3 cameras render.

## smoke_train — single-GPU training smoke test
- **Host:** `c642-012.vista.tacc.utexas.edu` (idev), 2026-10-06 ~12:12 CDT
- **Ran:** 300-step fine-tune of `bottles_75k.pt` on the bottles preview data, `torchrun --standalone --nproc-per-node 1 train.py --load-pretrained --pretrained-ckpt-name bottles_75k.pt`
- **Output:** `/scratch/11138/pranavbelligundu/abc_runs/smoke_train/`, log `/scratch/11138/pranavbelligundu/abc_runs/smoke_train.log`
- **Result:** 2.9–3.0 it/s at batch 90, peak 72.6 GiB, loss 0.046 → 0.019. Training works on one GH200.

## 1053205 — eval of the paper's released per-task checkpoints (baseline)
- **Host:** `c634-052`, 2026-10-06 20:40 → 22:09 CDT (1 h 29 m), COMPLETED
- **Script:** `scripts/vista/sbatch_baseline_eval.sh`
- **Checkpoints:** `/scratch/11138/pranavbelligundu/abc_cache/finetuned_sim/<task>/<step>.pt`, sha256-verified: bottles @20k, dish rack @20k, pour @25k, put_relative @25k
- **Output:** `/scratch/11138/pranavbelligundu/abc_runs/baseline_eval/<task>_seed<S>/`
- **Log:** `/scratch/11138/pranavbelligundu/abc_runs/baseline_eval_1053205.log`
- **wandb:** `abc-vista` run `tn2aookm` (group `baseline-released`)

| Task | Ours (A / B) | Published (A / B) | Mean max progress (A / B) |
|---|---|---|---|
| bottles → bin | 17 / 17 | 19 / 18 | 0.91 / 0.90 |
| plates → dish rack | 12 / 13 | 15 / 14 (12 on A under MuJoCo 3.8) | 0.82 / 0.85 |
| pour | 17 / 17 | 13 / 16 | 0.85 / 0.85 |
| put_relative | 10 / 9 | 5 / 10 | 0.76 / 0.70 |

**Verdict:** ours reads lower on two tasks and higher on two, with no consistent lean, and the
gaps are within 20-trial noise. Our sim scores the paper's checkpoints like the paper does.

## 1053206 — reproduction fine-tune, bottles (`repro_sim_put_the_plastic_bottles_in_the_bin_recipe`)
- **Hosts:** `c612-102`, `c641-152` (2 nodes), 2026-10-06 20:48 → 2026-10-07 00:06 CDT (3 h 18 m), COMPLETED
- **Script:** `scripts/vista/sbatch_repro_finetune.sh` (defaults: `TASK=sim_put_the_plastic_bottles_in_the_bin`, `TAG=recipe`)
- **Setup:** shared reference above (parent `abc_dit_xl_200k_model.pt`, bottles v3 data, published recipe)
- **Output:** `/scratch/11138/pranavbelligundu/abc_runs/repro_sim_put_the_plastic_bottles_in_the_bin_recipe/{5000,10000,15000,20000,25000,last}.pt` (24.2 GB each, incl. optimizer state)
- **Log:** `/scratch/11138/pranavbelligundu/abc_runs/repro_ft_1053206.log`
- **wandb:** `abc-vista` run `repro_sim_put_the_plastic_bottles_in_the_bin_recipe` (group `repro-finetune`)
- **Throughput:** 2.35 it/s, peak 72.6 GiB/GPU. First multi-node (c10d) run on Vista.

| Step | 5k | 10k | 15k | 20k | 25k |
|---|---|---|---|---|---|
| val recon error | 0.0901 | 0.0891 | 0.0836 | 0.0821 | 0.0753 |
| val loss | 0.0707 | 0.0681 | 0.0688 | 0.0686 | 0.0703 |

Train loss ~0.055 at start → ~0.043 at 25k.

## 1053225 — eval of every checkpoint of 1053206
- **Host:** `c637-131`, 2026-10-07 10:39 → 12:23 CDT (1 h 44 m), COMPLETED
- **Script:** `RUN_DIR=…/repro_sim_put_the_plastic_bottles_in_the_bin_recipe TASK=sim_put_the_plastic_bottles_in_the_bin sbatch scripts/vista/sbatch_eval_run.sh`
- **Eval:** `--task put_plastic_bottles_in_bin --prompt "sim put the plastic bottles in the bin"`, protocol above. Fast inference worked on all 10 evals (24–54 s warmup, two cold compiles ~500 s); the Triton "out of resource" lines in the log are rejected autotune candidates, not failures.
- **Output:** `/scratch/11138/pranavbelligundu/abc_runs/repro_sim_put_the_plastic_bottles_in_the_bin_recipe/eval/repro_sim_put_the_plastic_bottles_in_the_bin_recipe_step<N>_seed<S>/`
- **Log:** `/scratch/11138/pranavbelligundu/abc_runs/eval_run_1053225.log`
- **wandb:** `abc-vista` run `ifj5w53i` (success-rate-vs-step curve + videos)

| Step | Seed A | Seed B | Total /40 | Success | Mean max progress |
|---|---|---|---|---|---|
| 5k | 12 | 12 | 24 | 0.60 | 0.78 |
| 10k | 15 | 16 | 31 | 0.78 | 0.91 |
| 15k | 15 | 13 | 28 | 0.70 | 0.82 |
| 20k | 15 | 14 | 29 | 0.73 | 0.92 |
| 25k | 14 | 19 | 33 | 0.83 | 0.93 |

**Against the baseline** (released bottles @20k on our setup = 34/40, published 37/40), paired
per scene:

| | Both succeed | Only released | Only ours | McNemar exact p |
|---|---|---|---|---|
| ours 25k vs released | 27 | 7 | 6 | 1.0 |
| ours 20k vs released | 26 | 8 | 3 | 0.23 |

**Verdict:** the reproduction holds. Ours @25k is indistinguishable from the released
checkpoint. The 10k–20k dip is within noise (at 25k the two seeds alone differ by 5).

---

## put_relative data-scale sweep (submitted 2026-10-07)
Prepared by a parallel session on 2026-10-07; reviewed and committed as `c0ff182`. Checks: subsets
nested (10 ⊂ 25 ⊂ 50 ⊂ full, 287 / 733 / 1,462 / 2,921 episodes = 1.21 / 3.02 / 6.05 / 12.09 h),
val_sim (141 episodes) and parent shared by symlink, all scripts pass `bash -n`. The eval prompt
"put relative" is harmless: the env rewrites the live prompt to each scene's sampled directive
(`abc_minimal/sim_env.py:113`). Fixed 25k steps at every fraction means the 10% arm sees ~34 epochs;
the 5k-step checkpoints show whether it overfits.
- **Data:** `/scratch/11138/pranavbelligundu/abc_ft/put_relative/` (12.7 h, 2,921 train episodes) and nested subsets `/scratch/11138/pranavbelligundu/abc_ft/put_relative_frac{10,25,50}/` built by `scripts/vista/make_data_subsets.py`
- **Launcher:** `scripts/vista/submit_put_relative_sweep.sh` (4 fine-tunes, each chained to its eval; plus parent zero-shot and released-checkpoint reference evals). Eval seeds 20260511 / 20260531 = 40 distinct scenes.
- **Outputs (once run):** `/scratch/11138/pranavbelligundu/abc_runs/repro_put_relative_frac<F>/` and `/scratch/11138/pranavbelligundu/abc_runs/put_relative_<ref>/`
- **Submitted** 2026-10-07 from `login1.vista` via `bash scripts/vista/submit_put_relative_sweep.sh`:

| Arm | Fine-tune | Eval |
|---|---|---|
| 10% | 1056467 | 1056468 |
| 25% | 1056469 | 1056470 |
| 50% | 1056471 | 1056472 |
| 100% | 1056473 | 1056474 |
| parent (0%) | — | 1056475 |
| released @25k | — | 1056476 |

### Outcome (checked 2026-10-18)

**3 of 4 fine-tunes FAILED** on 2026-10-08 between 06:50 and 07:45 CDT, all with cross-node NCCL
all-reduce timeouts (600 s watchdog, then `RendezvousConnectionError` / SIGABRT). Three jobs on six
different nodes died in the same hour while frac100 survived, so this looks like a fabric/network
event, not a code bug. Their evals (1056468/70/72) never ran. All three are resumable from `last.pt`:

| Arm | Hosts | Died at step | Checkpoints on disk |
|---|---|---|---|
| 10% (1056467) | c620-021, c620-022 | ~9,700 | 5000, last (=5000) |
| 25% (1056469) | c613-112, c621-101 | ~16,420 | 5000, 10000, 15000, last (=15000) |
| 50% (1056471) | c621-122, c622-111 | ~8,340 | 5000, last (=5000) |

Resume (login node; re-submitting the same TAG resumes from `last.pt` and the same wandb run), then
chain each into `sbatch_eval_run.sh` with `SEEDS="20260511 20260531"`.

**Completed evals** (40 distinct scenes, seeds 20260511 / 20260531):

| Model | Seed A | Seed B | Total /40 | Mean max progress (A / B) |
|---|---|---|---|---|
| Parent, no put_relative training (0%) | 4 | 9 | 13 | 0.60 / 0.61 |
| Released put_relative @25k | 11 | 8 | 19 | 0.76 / 0.64 |
| Ours 100% @5k | 6 | 5 | 11 | 0.48 / 0.61 |
| Ours 100% @10k | 3 | 7 | 10 | 0.67 / 0.62 |
| Ours 100% @15k | 10 | 11 | 21 | 0.89 / 0.73 |
| Ours 100% @20k | 6 | 4 | 10 | 0.66 / 0.70 |
| Ours 100% @25k | 8 | 6 | 14 | 0.70 / 0.73 |

Logs: `repro_ft_1056473.log`, `eval_run_1056474.log` (100%), `eval_run_1056475.log` (parent),
`eval_run_1056476.log` (released).

**Read:** the parent already solves 13/40 with zero put_relative sim data, and full fine-tuning
lands at 10–21/40 depending on the checkpoint; the step-to-step swing (10 → 21 → 10) is as large
as the fine-tuning gain. So on put_relative most of the performance comes from pre-training, and
40 scenes cannot resolve the fine-tuning effect. Not yet tested statistically.
