# JOURNAL — ABC on TACC Vista (Pranav)

Dated narrative of the work: what was tried, what was found, what was decided. Per-run
details (hosts, paths, numbers, settings) are in `RUNS.md`.

---

## 2026-10-06 — Port ABC to Vista and verify it runs

**Why ABC.** Max's longer-term goal is to use ABC's dataset and simulator as the evaluation
environment, plotting sim success rate against training-data scale. RoboTwin was the original
plan, but its SAPIEN sim needs x86 RTX GPUs, which are scarce on TACC. ABC's MuJoCo /
MuJoCo-Warp sim runs on Vista's GH200s. Max's first step is to show the paper's published sim
numbers reproduce here before anything else.

**Porting (commits `9e06225` … `b214527`, branch `vista-aarch64`).**
- `uv sync` failed on aarch64: upstream pins `torchcodec==0.11.0+cpu`, which has no ARM wheel,
  and the PyPI aarch64 wheel links CUDA 13 and won't load next to torch cu128. Pinned aarch64
  to `torchcodec==0.11.0+cu128` from the PyTorch index.
- System FFmpeg is unusable on compute nodes (`libunwind.so.8` missing; RHEL build has only
  the openh264 decoder). Installed FFmpeg 7 from conda-forge into
  `/work/11138/pranavbelligundu/vista/envs/ffmpeg`.
- Wrote `vista_env.sh` (FFmpeg on PATH, caches on `$SCRATCH`, uv thread cap on login nodes,
  HF token from `~/.hf_token`).
- Found that `prepare.py --checkpoint` does not install the sim assets; `prepare.py --sim` is
  required or eval dies on a missing mesh.

**Smoke tests on `c642-012`.** The README eval smoke test passes. A 300-step single-GPU
fine-tune runs at 2.9–3.0 it/s with 72.6 GiB peak, matching upstream's per-GPU H100 rate.

**Job scripts.** Wrote `scripts/vista/` (baseline eval, reproduction fine-tune, per-run eval,
wandb logger). The repro fine-tune uses 2 nodes because Vista has 1 GPU per node and the
published recipe uses 2 GPUs (global batch 180). Found TACC's `sbatch` prints a banner even
with `--parsable`; use `| tail -1` when chaining.

**Downloaded** the released per-task fine-tunes (bottles, dish rack, pour; put_relative
fetched by the job) and the 200k parent.

## 2026-10-06 evening → 2026-10-07 — Baseline eval and reproduction

**Baseline (job 1053205).** Ran the paper's four released per-task checkpoints under the
published protocol. Scores land within 20-trial noise of the paper, two tasks higher and two
lower, so the sim setup is consistent with theirs.

**Reproduction (jobs 1053206 → 1053225).** Fine-tuned the 200k parent on all of the bottles
sim data with the published recipe (25k steps, 2 nodes, 3.3 h), then evaluated every 5k
checkpoint. Success goes 0.60 → 0.83. At 25k we get 33/40 vs the released checkpoint's 34/40
on the same scenes; paired per scene, McNemar p = 1.0. **The reproduction holds.** Also
confirmed multi-node training and fast inference both work on Vista.

## 2026-10-07 — Two findings that change the sweep design

1. **The eval seeds overlap.** World seeds run `seed … seed+19`, so the paper's seeds 20260511
   and 20260512 share 19 of 20 scenes; "40 trials" is ~21 distinct scenes. Fine for comparing
   with the paper (same protocol), but understates noise. Decision: the sweep uses seeds 20
   apart (20260511 / 20260531) for 40 distinct scenes, held fixed across every arm.
2. **The 200k parent already trained on put-bottles.** Its metadata sidecar lists
   `put_plastic_bottles_in_bin` among its 5 sim tasks (98.47 h sim + 3,553 h real), and the
   `abc_sim` README reports the parent alone at 0.54 on it. Our fine-tune data comes from later
   generation batches than the parent's `sim0514` sim data, so the episodes are probably
   different, but it is the same scene and demonstrator. A data-scale sweep on bottles would
   measure topping up a task the model already does, not learning it from limited data.

**Decision.** No from-scratch training. Run the data-scale sweep on **put_relative**, which
the parent never saw in sim and which has headroom (released checkpoint 19/40 on our setup).
Nested 10 / 25 / 50 / 100% subsets by hours, fixed val set and norm stats, same recipe, with
the parent zero-shot as the 0% point and the released checkpoint as a reference.

**Status.** put_relative data downloaded and subsets built (`$SCRATCH/abc_ft/put_relative*`),
launcher `scripts/vista/submit_put_relative_sweep.sh` written (parallel session). Not yet
submitted, and the new scripts are uncommitted.

## 2026-10-08 → 2026-10-18 — Sweep partially ran; Max proposes a 2×2 matrix

**Sweep.** frac100, parent and released evals finished; the 10/25/50% fine-tunes died together
on 2026-10-08 from cross-node NCCL timeouts (resumable). The parent scores 13/40 on put_relative
with no put_relative data, and our full fine-tune 10–21/40 depending on step. Pre-training is
doing most of the work on this task. Details in `RUNS.md`.

**Max's to-do (2026-10-14).** A 2×2 matrix to separate pre-training from fine-tuning:
rows single-task vs multitask fine-tuning, columns pre-trained vs randomly initialized.
Pre-trained + single-task (what we have) is the expected best case and random-init +
multitask the expected worst. Goal: a fast closed-loop signal on whether pre-training is
necessary for single-task policies. Latent-action pre-training is one candidate method.
Also: share the fork with Max, and check in with the other lab contact about compute.

## Open items
- [x] Review and commit the sweep scripts (`c0ff182`).
- [x] Submit the sweep (jobs 1056467–1056476, 2026-10-07). Next: collect results into `RUNS.md`.
- [ ] 2×2 matrix: run random-init single-task, pre-trained multitask, random-init multitask; make a summary graph.
- [ ] Add Max as a collaborator on github.com/pranav-B21/abc.
- [ ] Check in with the other lab contact about staying on the current allocation.
- [ ] Resume the 3 failed sweep fine-tunes (10/25/50%, NCCL timeouts on 2026-10-08) and run their evals.
- [ ] Prune checkpoints in sweep runs (24 GB each, 145 GB per run) to the steps actually evaluated.
- [ ] Real-robot data-scale track (~300 h vs 3,000 h): blocked on access to gated `XDOF/ABC-130k`.
- [ ] Update `abc/CLAUDE.md` "Not yet verified": multi-node training and fast inference now verified.
- [ ] Back up the results in `/scratch/11138/pranavbelligundu/abc_runs/` that must survive; `$SCRATCH` purges unaccessed files. (This logbook lives in the repo at `abc/agent_logbook/`.)
