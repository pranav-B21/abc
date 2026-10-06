# CLAUDE.md

This file guides Claude Code (claude.ai/code) in this repository. It covers **this fork's** setup on
TACC Vista. Upstream's own docs remain authoritative for the code itself: `README.md` (setup,
training, eval), `abc_minimal/README.md` (training data, multi-node, conditioning) and
`abc_sim/README.md` (task catalogue, eval flags, published results). Don't copy them here. Link to them.

## What this repo is, and why it's here

A fork of [`amazon-far/abc`](https://github.com/amazon-far/abc). It holds ABC (*Scalable Behavior
Cloning with Open Data, Training, and Evaluation*, arXiv 2606.27375): the ABC-DiT and ABC-VLA
bimanual (YAM arms) behavior-cloning policies, plus `abc_sim`, a MuJoCo / MuJoCo-Warp simulator
with a scored task catalogue.

**Project goal (from Max):** use ABC's dataset and simulator as the evaluation environment.
RoboTwin was the original plan, but its SAPIEN sim needs RTX GPUs (x86 only), and those are scarce
on TACC. ABC's sim runs on GH200s.

- **Data:** about 3,000 h of real-robot data (`XDOF/ABC-130k`) and about 400 h of sim data (the
  24 tasks in `prepare.py --sim-data-list` add up to about 400 h).
- **Framing:** train on a fraction of the data, e.g. about 300 h (1/10 of the robot data), and check
  whether it matches the full model.
- **Deliverable:** training curves of **sim success rate against data scale** on ABC tasks.
- **Later:** a physical robot exists for real-world deployment (`deploy/`). Nothing on TACC
  touches it.

## Git

| Remote | URL | Use |
|---|---|---|
| `origin` | `github.com/pranav-B21/abc` | Our fork. Push here. |
| `upstream` | `github.com/amazon-far/abc` | Pull updates: `git fetch upstream && git merge upstream/main` |

Work happens on branch **`vista-aarch64`**. `main` tracks upstream untouched. The repo-local git
identity is `pranav-B21 <pranav.belligundu@gmail.com>`, set in `.git/config` because the account
has no global identity.

## Environment: TACC Vista specifics

Vista nodes have **one NVIDIA GH200 (96 GB) each, on aarch64 (ARM)**, running RHEL 9 (glibc
2.34) with driver 590 / CUDA 13.1. Upstream assumes x86, `apt` and a large `$HOME`. None of that
holds here.

**Always `source vista_env.sh` before any `uv run`** (from the repo root). It sets:

| Variable | Value | Why |
|---|---|---|
| `PATH` | `~/.local/bin` (uv), then the conda-forge ffmpeg **ahead of** `/usr/bin` | see the ffmpeg row below |
| `LD_LIBRARY_PATH` | `/work/11138/pranavbelligundu/vista/envs/ffmpeg/lib` | FFmpeg shared libs for `torchcodec` |
| `ABC_CACHE` | `/scratch/11138/pranavbelligundu/abc_cache` | Data and checkpoints (tens of GB) |
| `UV_CACHE_DIR`, `WARP_CACHE_PATH`, `HF_HOME` | under `/scratch/11138/pranavbelligundu/` | Keep `$HOME` (small quota) clean |
| `UV_LINK_MODE` | `copy` | `$WORK` and `$SCRATCH` are different filesystems, so uv can't hardlink |

### Changes from upstream (commit `9e06225`)

1. **`torchcodec` pinned per architecture** in `pyproject.toml`. Upstream pins
   `torchcodec==0.11.0+cpu`, which has **no aarch64 wheel**, so `uv sync` fails on Vista. The PyPI
   aarch64 wheel installs, but it links **CUDA 13** (`libnvrtc.so.13`, `libnppicc.so.13`) and
   fails to load next to torch `2.11.0+cu128`. The fix keeps x86_64 on `+cpu` and sends aarch64
   to `torchcodec==0.11.0+cu128` from the `pytorch-cu128` index (the same index as torch). If
   upstream bumps torch's CUDA, bump this pin to match.
2. **`vista_env.sh`**, described above.

### FFmpeg (installed out of tree)

The README says `sudo apt-get install ffmpeg`. On Vista:
- No sudo.
- `/usr/bin/ffmpeg` fails on compute nodes (`libunwind.so.8: cannot open shared object file`).
- Even when it loads, RHEL's build ships only the **libopenh264** H.264 decoder, which fails here
  ("Unable to create decoder").

So FFmpeg 7 comes from conda-forge, in a standalone prefix:

```bash
/work/11138/pranavbelligundu/vista/anaconda/bin/conda create -y -p /work/11138/pranavbelligundu/vista/envs/ffmpeg \
    -c conda-forge --override-channels "ffmpeg<8"
```

That prefix lives outside the repo (`vista/envs/ffmpeg`). Don't move it, because conda prefixes
are path-dependent. `torchcodec` uses its libraries. `abc_minimal/export_mcap.py` shells out to
`ffmpeg` for H.264, which is why it must come first on `PATH`.

### Python environment

`uv` 0.12 is in `~/.local/bin`, installed by the astral script. `.python-version` pins 3.12, and uv
downloaded its own CPython 3.12.15 (aarch64). The venv is `.venv/` (gitignored). To rebuild it:

```bash
source vista_env.sh && uv sync          # add --extra deploy only for real-robot work (not on TACC)
```

Verified versions: torch `2.11.0+cu128` (`cuda.is_available()` True), torchcodec `0.11.0+cu128`,
mujoco `3.8.1`, warp-lang `1.13.0` (it detects the GH200 as `sm_90`), mujoco-warp `3.8.0.3`.

## What has been downloaded

| What | Where | Size | How |
|---|---|---|---|
| `bottles_75k.pt`: 75k-step bottles-only DiT policy (vision backbone and norm stats included) | `$ABC_CACHE/` | 8.1 GB | `prepare.py --checkpoint` |
| `norm_stats.json`, bottles preview episodes (`train_real/`, `val_real/`, `train_sim/`, `val_sim/`) | `$ABC_CACHE/` | ~160 MB | same |
| All 37 sim asset packages (meshes, textures, MJCF) | `abc_sim/models/assets/` (gitignored, on `$WORK`) | 1.6 GB | `prepare.py --sim` |

**Not yet downloaded:**
- `assets_robocasa`: 2.8 GB, needed only by `inhand_transfer`. Get it with `--sim-robocasa`.
- The 200k multi-task parent (`--pretrained`, 8.1 GB).
- Any sim task's episodes (`--sim-data <task>`).
- The full bottles data (`--full`, 35 GB).
- DINOv3 weights.

`prepare.py --checkpoint` does **not** install the sim assets, despite what the README implies. A
fresh clone needs `prepare.py --sim`. Without it eval dies with
`Error opening file 'assets/i2rt_yam/assets/model2__17.stl'`.

## Verified so far (2026-10-06, node c642-012)

The README smoke test passes:

```bash
source vista_env.sh
uv run eval_policy.py --checkpoint "$ABC_CACHE/bottles_75k.pt" \
    --num-worlds 1 --num-chunks 2 --no-fast-inference --save-video --video-every-n-actions 15
```

It ran in about 2.5 min wall-clock (the first run also compiles MJWarp kernels into
`WARP_CACHE_PATH`). It writes `outputs/sim_eval_put_plastic_bottles_in_bin/{summary.json,world_000.mp4}`.
`success=False` is expected, since two chunks are too short to finish the task. A decoded frame
shows all three cameras (top, left, right) rendering correctly.

**Not yet verified on Vista:**
- The default `--fast-inference` (bf16, torch.compile and CUDA graphs).
- `--parallel-worlds` batched MJWarp eval.
- Any `train.py` run.
- Multi-node.

Treat each one as unknown until it has run here.

## How to run things

All commands assume `source vista_env.sh` and the repo root as working directory. `ABC_CACHE`
resolves to `$SCRATCH`, so write `$ABC_CACHE/...` instead of the README's `cache/...`.

### Downloading

```bash
uv run prepare.py --sim-data-list            # 24 sim tasks: size, episode counts, hours
uv run prepare.py --sim-data <task> [--cache DIR]   # one sim task's episodes and assets
uv run prepare.py --sim-checkpoint-list      # published per-task finetunes and their eval results
uv run prepare.py --sim-checkpoint <task>    # ~8 GB, sha256-verified; prints its eval command
uv run prepare.py --pretrained               # 200k multi-task DiT parent (finetuning start point)
```

Real-robot tasks beyond bottles come from the **gated** HF dataset `XDOF/ABC-130k`. Accept access on
the dataset page, `export HF_TOKEN=...`, then run `uv run scripts/export_hf_task.py --task <name>`.

### Evaluating (the deliverable's y-axis)

```bash
uv run eval_policy.py --checkpoint <ckpt.pt> --task <task> [--prompt "<training prompt>"] \
    --num-worlds 50 [--save-video --video-every-n-actions 15]
# -> outputs/sim_eval_<task>/summary.json : success_rate, num_success, mean_reward, mean_max_progress
```

Rules that matter for credible curves (details in `abc_sim/README.md` § Sim Eval):
- **`--num-worlds` ≥ 50** for any number you quote. At a true 10% success rate, the default 5 worlds
  usually reads 0.
- **Evaluate with the prompt the run trained under.** `train.py` prints it at startup. A finetune on
  `sim_224` episodes uses the dataset wording (e.g. `sim put the plastic bottles in the bin`), and the
  200k parent uses older wordings.
- **Don't compare across physics backends.** `--parallel-worlds` (MJWarp) reads 0.71–0.77 on
  put-bottles where sequential CPU MuJoCo reads 0.54 on the same scenes. Pick one backend for the
  whole sweep.
- **Fix `--num-chunks` across a sweep.** The default is 236. `lego_blocks_sorting` needs 400, and
  `ball_tray_balancing` is only comparable at equal chunk counts.
- Four tasks have no evaluator and always score 0: the three `put_markers_in_*_drawer` tasks and
  `build_wood_block_tower`. Don't use them for curves.
- Published per-task reference numbers (20 worlds × 2 seeds) are in `abc_sim/README.md`
  § Finetuned per-task checkpoints. Strong tasks: bottles 19/20, inhand-transfer 18/20,
  dishrack 15/20, pour 13–16/20. Several are near 0 (nuts_bolts, ball_tray, lego).

### Training and finetuning

```bash
# Per-task finetune from the 200k parent (the recipe behind every published sim checkpoint):
uv run prepare.py --sim-data <task> --cache $SCRATCH/abc_ft/<task>
uv run prepare.py --pretrained      --cache $SCRATCH/abc_ft/<task>
uv run torchrun --standalone --nproc-per-node 1 train.py \
    --cache-root $SCRATCH/abc_ft/<task> --load-pretrained --mixture-preset sim_task \
    --flow.max-action-prefix 8 --train-steps 25000 --val-every 5000 --val-batches 40 --ckpt-every 5000
# checkpoints -> <cache-root>/finetune_checkpoints/{5000,...,last}.pt ; --resume-from to continue
```

- **GPU count:** the reference recipes use 8 GPUs (from-scratch DiT) or 2 GPUs (finetune) at
  `--batch-size 90` per GPU. Vista has 1 GPU per node, so either run `--nproc-per-node 1` (same
  per-GPU batch, smaller global batch, more wall-clock) or go multi-node (below). A smaller global
  batch changes the optimization, so keep it **identical across every point of a data-scale sweep**.
- **DINOv3 trap:** from-scratch DiT training (without `--load-pretrained`) loads
  `$ABC_CACHE/dinov3_vitb16_pretrain_lvd1689m.pth`. **If that file is missing, it prints
  `no ... using random DINOv3` and trains anyway** with a randomly initialized vision backbone
  (`abc_minimal/train_loop.py:231`). `--load-pretrained` and `--resume-from` don't need the
  file. The weights are license-gated: the user has to accept Meta's terms for
  `facebook/dinov3-vitb16-pretrain-lvd1689m` on HF.
- **ABC-VLA:** finetuning (`prepare.py --vla-pretrained`, `--policy vla --load-pretrained`) needs no
  Gemma base. From-scratch VLA needs Google's `gemma_pytorch` Gemma-3-4B checkpoint from Kaggle. The
  HF format won't load.
- `--log-wandb` is off by default (project `minimal-abc`). Run `wandb login` once first.
- `uv run python train.py --help` lists every flag. The configs are dataclasses in
  `abc_minimal/config.py`.

### Data scaling (no built-in knob)

`TrainConfig` has **no data-fraction or max-episodes flag**. A component in
`MIXTURE_PRESETS` (`abc_minimal/config.py`) reads *every* episode under its `train_dir`. Weights
only change sampling proportions, not how much data exists. To train on X% of a task, build a
separate cache root whose `train_sim/` (or `train_real/`) holds only a subset of
`episode_<uuid>/` dirs. Symlinks avoid copying. Then point `--cache-root` at it, and keep
`val_*` and `norm_stats.json` identical across scales so the validation set is fixed. Choose the
subset with a fixed seed and nest the subsets (10% ⊂ 25% ⊂ 50% ⊂ 100%). Count each one in hours
from the frame counts, not in episodes, because episode lengths vary a lot by task.

### Viewers

`viz_policy.py` and `viz_episode.py` serve a viser page on `--port 8080` on the compute node. From a
laptop, tunnel through the login node, e.g. `ssh -L 8080:<compute-node>:8080 <user>@vista.tacc.utexas.edu`.

## SLURM / TACC rules

- **Submit `sbatch` from a login node.** It's disabled on compute nodes. Downloads (`prepare.py`)
  are fine on login nodes. Anything that touches CUDA needs a compute node (`idev` or `sbatch`).
- **Storage:** code on `$WORK` (this repo), data and checkpoints on `$SCRATCH`, nothing large in
  `$HOME`. `$SCRATCH` is purged when files go unaccessed, so re-run `prepare.py` if the cache
  disappears.
- **Multi-node:** see `abc_minimal/README.md` § Multi-node training. Upstream's recipe assumes
  node-local NVMe shards (`scripts/prepare_hf_shards.py`). On Vista `$SCRATCH` is shared, so a
  normal cache works, but the launch has to change from `torchrun --standalone` to a rendezvous
  across the SLURM allocation (`--nnodes $SLURM_NNODES --rdzv-backend c10d --rdzv-endpoint <head>:29500`
  under `srun`). That launcher hasn't been written yet.
- Compute nodes have internet access (git, HF, and the CLIP text weights fetched at runtime from
  `openaipublic.azureedge.net` all work).

## Gotchas

- `uv run` sometimes prints `Uninstalled 1 package / Installed 1 package`. That's uv re-syncing the
  editable `abc-minimal` package, and it's harmless.
- The `[warn] --save-video under --rtc` line is expected. Only the timing telemetry includes render
  time; scores are unaffected.
- `outputs/` (eval results) is gitignored and written inside the repo. Copy anything you want to keep.
