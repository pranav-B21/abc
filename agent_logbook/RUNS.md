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

---

## 2×2 matrix: pre-training × multitask on put_relative (submitted 2026-10-18)
- **Launcher:** `bash scripts/vista/submit_matrix.sh` (login node). Re-running it skips finished training and evals.
- **Submitted** 2026-10-18 from `login2.vista`, commit `12af1b1`. Train → eval job IDs:
  scratch_dino_single 1079137→1079138, scratch_rand_single 1079139→1079140,
  mt6_pretrained 1079141→1079142, scratch_dino_mt6 1079143→1079144, scratch_rand_mt6 1079145→1079146,
  scratch_dino_75k 1079147→1079148, frac10 1079149→1079150, frac25 1079151→1079152,
  frac50 1079153→1079154; re-evals frac100 1079155, parent 1079156, released 1079157.
  Logs: `$SCRATCH/abc_runs/{repro_ft,eval_run}_<jobid>.log`.
- **Eval:** every model on the same 100 distinct put_relative scenes, seeds 20260511/31/51/71/91 × 20 worlds.
- **Summary:** `uv run --no-project --with matplotlib python scripts/vista/matrix_summary.py` → `/scratch/11138/pranavbelligundu/abc_runs/matrix_summary/` (table + `matrix_2x2.png`, `matrix_steps.png`, `data_scale.png`).

| Entry | Init | Vision encoder | Data (cache root under `$SCRATCH/abc_ft/`) | Steps / ckpt | Run dir |
|---|---|---|---|---|---|
| pre-trained single (exists) | 200k parent | parent | `put_relative` (12.09 h) | 25k / 5k | `repro_put_relative_frac100` |
| scratch_dino_single | random DiT | DINOv3 | `put_relative_scratch_dino` | 25k / 5k | `repro_put_relative_scratch_dino` |
| scratch_rand_single | random DiT | random | `put_relative_scratch_rand` | 25k / 5k | `repro_put_relative_scratch_rand` |
| mt6_pretrained | 200k parent | parent | `mt6_pretrained` (87.0 h) | 25k / 5k | `repro_mt6_pretrained` |
| scratch_dino_mt6 | random DiT | DINOv3 | `mt6_scratch_dino` | 25k / 5k | `repro_mt6_scratch_dino` |
| scratch_rand_mt6 | random DiT | random | `mt6_scratch_rand` | 25k / 5k | `repro_mt6_scratch_rand` |
| scratch_dino_75k | random DiT | DINOv3 | `put_relative_scratch_dino` | 75k / 25k | `repro_put_relative_scratch_dino_75k` |
| frac10/25/50 (resume) | 200k parent | parent | `put_relative_frac{10,25,50}` | 25k / 5k | `repro_put_relative_frac{10,25,50}` |
| refs (eval only) | — | — | — | — | `put_relative_parent`, `put_relative_released` |

mt6 = put_relative 12.09 h (13.9%), sim_pouring_beads 7.01 h, sim_inhand_transfer 4.95 h, grab_clutter
10.52 h, conveyor_pick 25.34 h, count_into_opaque_box 27.14 h; 19,093 usable episodes
(`$SCRATCH/abc_ft/mt6/composition.json`). Sampling is uniform over frames, i.e. hours-weighted.
Scratch roots use `$SCRATCH/abc_ft/matrix_norm_stats.json` (the parent's embedded stats) and, for
DINOv3, `$ABC_CACHE/dinov3_vitb16_pretrain_lvd1689m.pth` (converted from HF, see JOURNAL 2026-10-18).
Smoke tests on `c642-022` (300 / 100 / 60 steps): DINOv3 single, random single, random mt6 all train.
Expected disk: ~1.1 TB of new checkpoints.

### Outcome of the 2026-10-18 submission (checked 2026-10-21 on `c642-042`)
**3 of 9 training jobs finished; 6 died on 2026-10-19 between 02:37 and 05:41** from the same
2-node NCCL all-reduce hang as the 2026-10-08 sweep: training ran normally at ~2.3 it/s, then one
`ALLREDUCE` sat for 600 s and the watchdog killed both ranks, with nothing logged beforehand.
Their chained evals sat at `DependencyNeverSatisfied`. All 3 reference re-evals finished.

| Job | Run | Died at step | Saved checkpoints |
|---|---|---|---|
| 1079137 | scratch_dino_single | 5,460 | 5k |
| 1079139 | scratch_rand_single | 11,600 | 5k, 10k |
| 1079143 | scratch_dino_mt6 | 16,280 | 5k–15k |
| 1079145 | scratch_rand_mt6 | 22,960 | 5k–20k |
| 1079147 | scratch_dino_75k | 15,180 | **none** (ckpt every 25k, so all of it is lost) |
| 1079153 | frac50 | 10,660 | 5k, 10k (plus what the 10-08 run saved) |

**Fix (2026-10-21):** `sbatch_repro_finetune.sh` now retries inside its allocation (`MAX_TRIES=4`),
resuming from `last.pt` under a fresh rendezvous id and port, and only exits 0 once
`$TRAIN_STEPS.pt` exists. `submit_matrix.sh` now checkpoints the 75k run every 5k (eval only at
25k/50k/75k) and gives it 20 h. Resubmit (login node):
`scancel 1079138 1079140 1079144 1079146 1079148 1079154` then
`ONLY="scratch_dino_single scratch_rand_single scratch_dino_mt6 scratch_rand_mt6 scratch_dino_75k frac50" bash scripts/vista/submit_matrix.sh`.

**Resubmitted 2026-10-21** from `login1.vista` with the retry launcher (commit `d7765ed`); the old
chained evals were already gone from the queue. Train → eval: scratch_dino_single 1088031→1088032,
scratch_rand_single 1088033→1088034, scratch_dino_mt6 1088035→1088036, scratch_rand_mt6
1088037→1088038, scratch_dino_75k 1088039→1088040 (from step 0), frac50 1088041→1088042.

### Results so far (100 scenes, `matrix_summary/matrix_table.csv`)
Success / 100 at each checkpoint. Paired p = exact McNemar on the same 100 scenes.

| Model | 5k | 10k | 15k | 20k | 25k | 25k vs parent (paired) |
|---|---|---|---|---|---|---|
| Parent, zero-shot (no fine-tuning) | | | | | **33** | — |
| Released put_relative@25k | | | | | **39** | p = 0.45 (19 vs 25 discordant) |
| Pre-trained + single (frac100, 12.09 h) | 28 | 28 | 39 | 31 | **33** | p = 1.0 (23 vs 23) |
| Pre-trained + multitask (mt6, 87 h) | 23 | 31 | 23 | 25 | **32** | p = 1.0 (17 vs 16) |
| Pre-trained + single, frac25 (3.02 h) | 25 | 27 | 23 | 26 | **30** | p = 0.74 |
| Pre-trained + single, frac10 (1.21 h) | 21 | 21 | 16 | 23 | **20** | **p = 0.024, worse than the parent** |

- **Released checkpoint reproduces the README** (39/100 here vs 15/40 = 37.5% published).
- **Fine-tuning on put_relative does not measurably beat the zero-shot parent.** Our 12 h run ties it
  exactly, and even the released checkpoint's +6 points is not significant.
- **Multitask = single-task** for the pre-trained row (32 vs 33, p = 1.0).
- **Data scale:** 1.2 h → 20, 3 h → 30, 12 h → 33. The smallest subset is the only significant
  effect, and it is *negative* vs the parent: 1.2 h of task data makes the model worse.
- **Noise is large:** two models with the same 33% disagree on 46 of 100 scenes, and frac100
  moves 28 → 39 → 31 between neighbouring checkpoints. A 100-scene eval resolves roughly ±13 points
  (Wilson half-width at 33%) between models, so only big effects will be visible. The scratch
  cells are expected to be one.
