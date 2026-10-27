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

## 2026-10-18 — How the paper built "from scratch", and the matrix setup

**Paper (abc.bot/abc.pdf, §5.2 Fig. 13, App. E).** Max's column question was already asked on
real robots: four tasks (credit card, LEGO sort, pen cap, bottle cap), single-task policies that
"differ only in their initialization": scratch vs a 3.5K-h pre-train vs a 7K-h pre-train. More
pre-training data always wins. Bar values exist only in the chart. Their recipe is far bigger
than our sim recipe: 200k steps, batch 1,440, LR 1e-5, best of every-50k checkpoints picked by
eye. So our 25k-step random-init cells are a much weaker "scratch" than theirs.

**What "scratch" means in this codebase** (`abc_minimal/train_loop.py:206-243`):
- No `--load-pretrained` builds `DiTPolicy` with random weights, then loads DINOv3 into the
  vision encoder from `<cache-root>/dinov3_vitb16_pretrain_lvd1689m.pth` if the file exists.
- The CLIP **text** encoder is always the frozen pretrained OpenAI CLIP, in every condition
  (`CLIPTextEmbedder`, `.eval()` + `no_grad`), so even "all random" keeps a pretrained language
  encoder.
- The paper never says how its scratch arm initialized the vision encoder. Everything points
  to DINOv3: the code path above, the README telling you to fetch DINOv3 before training, and
  the parent's own vision encoder being DINOv3 (next point). Still an inference.

**DINOv3 weights.** HF access works (token in `~/.hf_token`). HF's file uses different key
names from the code, and the code loads with `strict=False`, so the raw file would have loaded
nothing. I wrote `scripts/vista/convert_dinov3_hf.py`. 162/175 converted tensors have cosine
≥ 0.99 with the 200k parent's vision encoder (max abs diff 0.006 on the patch embedding); the
other 13 are all-zero in both. So ABC's pre-training started from DINOv3 and moved it very
little.

**Smoke tests on `c642-022`.** From scratch with DINOv3: loads `missing=13` (buffers only),
loss 1.40 → 0.26 in 300 steps (the pre-trained model starts at 0.06). All random: prints
"using random DINOv3", loss 1.40 → 0.50 in 100 steps.

**Decisions (with Pranav).** Both no-pre-training variants (DINOv3 vision + random DiT, and all
random); multitask = put_relative + 5 tasks the parent never saw in sim (pour, inhand transfer,
grab clutter, conveyor pick, count into opaque box); 25k steps for every cell + one 75k
DINOv3/single-task check; every model evaluated on the same 100 distinct scenes (5 seeds × 20
worlds, seeds 20 apart); scratch cells normalize with the parent's stats; resume the 3 failed
sweep runs. Scripts: `make_matrix_roots.py`, `submit_matrix.sh`, `matrix_summary.py`;
`sbatch_repro_finetune.sh` gains `INIT=scratch VISION=dinov3|random CKPT_EVERY`.

## 2026-10-21: matrix half done, NCCL hangs again, and fine-tuning does not beat the parent

**What finished.** mt6_pretrained, frac10, frac25 trained and evaluated; parent, released and
frac100 re-evaluated on the 100 scenes. The other 6 training jobs (all 4 scratch cells, the 75k
check, frac50) died on 2026-10-19 from the same NCCL all-reduce hang as 10-08. That is now 9 of
13 two-node jobs, so it is not a one-off fabric event; treat it as a property of 2-node runs on
Vista. Rather than chase the cause, `sbatch_repro_finetune.sh` now retries in-allocation from
`last.pt` (up to 4 tries, fresh rendezvous each time). The 75k run checkpointed only every 25k
and lost all 15k steps; it now checkpoints every 5k. Needs a login node to resubmit (see RUNS.md).

**The finding.** On 100 scenes the zero-shot parent scores 33, the same as our 12 h put_relative
fine-tune (33, p = 1.0) and the 87 h multitask fine-tune (32, p = 1.0). The released checkpoint's
39 is not significantly above it either (p = 0.45). The released number itself reproduces the
README (37.5%). So on put_relative, everything the pre-trained row knows comes from pre-training,
and 25k steps of task data add nothing measurable to success (partial progress does rise, 0.62 → 0.71, p = 0.046, see RUNS.md); with only 1.2 h of it the model gets worse
(20, p = 0.024). The 40-scene sweep (parent 13/40, released 19/40) said the same thing more
noisily. Two consequences:
- put_relative is a weak task for a data-scale curve (the curve is flat from 3 h up). A task
  where the parent starts low would show scale effects better.
- The matrix's pre-training column is still informative: if the scratch cells come in well
  below 33, that gap *is* the value of pre-training, which is Max's question.

Open: put_relative is not one of the parent's 5 sim tasks, but the parent saw 3,553 h of real
teleop, which may contain put-relative-like behaviour. That would explain a 33% zero-shot rate.

## 2026-10-26: the 2×2 matrix is complete; pre-training is the whole effect

All four no-pre-training cells finished on the second submission. On the same 100 scenes at 25k:
pre-trained 33 / 32 (single / multi), no pre-training 11–13 in every cell, regardless of vision
init or task mix. Pre-training is worth ~20 points (p = 5e-4); 25k steps of training from scratch
on 12 h or 87 h of sim data gets about a third of what the 200k parent gets *with no fine-tuning
at all*. Multitask did nothing in any column. The DINOv3 encoder over a random one is at most a
few points (not significant at 25k; ~4–5 points pooled over checkpoints).

Read for Max: ABC's value on this task comes from its large-scale pre-training (3,553 h real +
98 h sim), not from the task data or from a pretrained vision encoder. The open caveat is
compute: scratch cells got 25k steps, the parent got 200k at a much larger batch. The 75k run
tests whether more steps close it; the flat 5k→25k scratch curves suggest not much.

The 75k and frac50 runs failed again, this time from a cluster-side stall followed by OOM on
every retry (leftover ranks held GPU memory). Fixed the retry to clean up and wait for memory.

## 2026-10-26: 500-scene check of fine-tuned vs not fine-tuned

On 100 scenes the parent and our 12 h fine-tune@25k tie on success (33 vs 33) while progress
favours the fine-tune (0.62 → 0.71, p = 0.047). With 100 scenes only gaps of ~13 points are
detectable, so Pranav chose to re-test just these two models on 500 scenes (step A of the plan;
5 training seeds were offered and declined for now). `submit_scene500.sh` adds 20 seeds
(20260611..20260991, spacing 20, no overlapping world seeds) as 4 one-node jobs;
`compare_scene500.py` gives the paired success and progress tests on the first 100 and on all.

- [x] Review and commit the sweep scripts (`c0ff182`).
- [x] Submit the sweep (jobs 1056467–1056476, 2026-10-07). Next: collect results into `RUNS.md`.
- [x] 2×2 matrix submitted 2026-10-18 (jobs 1079137–1079157). 3/9 trained; results in RUNS.md (2026-10-21).
- [x] Resubmitted the 6 hung runs 2026-10-21 (jobs 1088031–1088042). 4 matrix cells done 2026-10-23; matrix complete.
- [ ] Submit `submit_scene500.sh` (login node), then run `compare_scene500.py`.
- [x] Resubmitted scratch_dino_75k and frac50 2026-10-26 (jobs 1103215–1103218). Next: re-run `matrix_summary.py` when done.
- [x] Add Max as a collaborator on github.com/pranav-B21/abc (Pranav, by 2026-10-18).
- [ ] Check in with the other lab contact about staying on the current allocation.
- [ ] Resume the 3 failed sweep fine-tunes: frac10 and frac25 done 2026-10-19; frac50 hung again, in the resubmit.
- [ ] Prune checkpoints in sweep runs (24 GB each, 145 GB per run) to the steps actually evaluated.
- [ ] Real-robot data-scale track (~300 h vs 3,000 h): blocked on access to gated `XDOF/ABC-130k`.
- [x] Update `abc/CLAUDE.md` "Not yet verified" (2026-10-18).
- [ ] Back up the results in `/scratch/11138/pranavbelligundu/abc_runs/` that must survive; `$SCRATCH` purges unaccessed files. (This logbook lives in the repo at `abc/agent_logbook/`.)
