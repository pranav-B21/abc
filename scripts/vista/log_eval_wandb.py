"""Push sim-eval results (summary.json + rollout videos) to wandb.

eval_policy.py has no wandb hook; it writes <out>/summary.json and world_*.mp4. This
script walks a results root whose subdirectories are named

    <label>_seed<seed>                e.g. sim_put_the_plastic_bottles_in_the_bin_seed20260511
    <label>_step<step>_seed<seed>     e.g. repro_bottles_step20000_seed20260511

and logs one wandb run containing:
  - a table of every (label, step, seed) row with success_rate / num_success / progress,
  - per (label, step) seed-averaged success_rate and mean_max_progress, logged against
    `train_step` when steps are present, so a finetune's eval sweep plots as a curve,
  - up to --videos-per-eval rollout videos per eval directory.

Usage (on a node with internet; `wandb login` once beforehand):
    uv run python scripts/vista/log_eval_wandb.py --root $SCRATCH/abc_runs/baseline_eval \
        --name baseline-eval --group baseline
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import wandb

DIR_RE = re.compile(r"^(?P<label>.+?)(?:_step(?P<step>\d+))?_seed(?P<seed>\d+)$")


def collect(root: Path) -> list[dict]:
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        m = DIR_RE.match(d.name)
        f = d / "summary.json"
        if not m or not f.exists():
            continue
        s = json.loads(f.read_text())
        rows.append({
            "label": m["label"],
            "step": int(m["step"]) if m["step"] else None,
            "seed": int(m["seed"]),
            "task": s.get("task"),
            "prompt": s.get("prompt"),
            "num_worlds": s.get("num_worlds"),
            "num_success": s.get("num_success"),
            "success_rate": s.get("success_rate"),
            "mean_max_progress": s.get("mean_max_progress"),
            "mean_reward": s.get("mean_reward"),
            "checkpoint": s.get("checkpoint"),
            "dir": d,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--project", default="abc-vista")
    ap.add_argument("--name", default=None)
    ap.add_argument("--group", default=None)
    ap.add_argument("--videos-per-eval", type=int, default=2)
    args = ap.parse_args()

    rows = collect(args.root)
    if not rows:
        raise SystemExit(f"no <label>[_step<N>]_seed<S>/summary.json under {args.root}")

    run = wandb.init(project=args.project, name=args.name, group=args.group, job_type="sim-eval",
                     config={"root": str(args.root)})

    cols = ["label", "step", "seed", "task", "prompt", "num_worlds", "num_success",
            "success_rate", "mean_max_progress", "mean_reward", "checkpoint"]
    run.log({"eval/all_rows": wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in rows])})

    # Seed-averaged, per (label, step).
    agg: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        agg[(r["label"], r["step"])].append(r)
    summary = []
    for (label, step), rs in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        n = sum(r["num_worlds"] or 0 for r in rs)
        k = sum(r["num_success"] or 0 for r in rs)
        prog = sum(r["mean_max_progress"] or 0 for r in rs) / len(rs)
        summary.append([label, step, len(rs), k, n, k / n if n else None, prog])
    run.log({"eval/summary": wandb.Table(
        columns=["label", "step", "seeds", "successes", "worlds", "success_rate", "mean_max_progress"],
        data=summary)})

    # Curves: one metric per label, x-axis = training step (when present).
    run.define_metric("train_step")
    for label, step, _, _, _, sr, prog in summary:
        run.define_metric(f"{label}/*", step_metric="train_step")
        run.log({"train_step": step or 0, f"{label}/success_rate": sr, f"{label}/mean_max_progress": prog})

    if args.videos_per_eval > 0:
        for r in rows:
            vids = sorted(r["dir"].glob("world_*.mp4"))[: args.videos_per_eval]
            for v in vids:
                key = f"video/{r['label']}" + (f"_step{r['step']}" if r["step"] else "") + f"_seed{r['seed']}"
                run.log({f"{key}/{v.stem}": wandb.Video(str(v), format="mp4")})

    print(f"logged {len(rows)} eval dirs ({len(summary)} label/step groups) -> {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
