"""Bring every abc-vista W&B run in line with the naming and grouping scheme (2026-11-02).

Training runs (run ID = run dir) go in group `training` under the display name from wandb_names.sh;
eval runs (job_type sim-eval, label = the run dir above their results root) go in group `eval` as
`eval_<display name>` (`eval-<run dir>` when the map has no entry). An old group other than these two
is kept as a tag. Needed for jobs queued before the scheme existed: Slurm runs the copy of a batch
script taken at submission, so those jobs log under their old names and groups.

Dry run by default; --apply writes the changes.

    uv run --no-sync python scripts/vista/wandb_regroup.py [--apply]
"""

import argparse
import subprocess
from pathlib import Path

import wandb

PROJECT = "pranavbelligundu-the-university-of-texas-at-austin/abc-vista"
NAMES_SH = Path(__file__).with_name("wandb_names.sh")


def display_name(run_dir: str) -> str:
    out = subprocess.run(["bash", "-c", f'source "{NAMES_SH}"; wandb_display_name "$1"', "_", run_dir],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def target(run):
    """(name, group) the scheme wants for this run, or None if it can't tell."""
    if run.job_type == "sim-eval":
        root = (run.config or {}).get("root")
        if not root:
            return None
        if Path(root).name != "eval":  # not <run dir>/eval, e.g. the 2026-10-07 baseline eval: keep its name
            return run.name, "eval"
        label = Path(root).parent.name
        name = display_name(label)
        return (f"eval_{name}" if name != label else f"eval-{label}"), "eval"
    return display_name(run.id), "training"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write the changes (default: print them only)")
    args = ap.parse_args()

    changed = skipped = 0
    for run in wandb.Api(timeout=60).runs(PROJECT):
        want = target(run)
        if want is None:
            print(f"[skip] {run.id} ({run.name}): eval run without a config root")
            skipped += 1
            continue
        name, group = want
        tags = list(run.tags)
        if run.group and run.group not in ("training", "eval") and run.group not in tags:
            tags.append(run.group)
        if (run.name, run.group, sorted(run.tags)) == (name, group, sorted(tags)):
            continue
        print(f"{run.id}: {run.name!r} [{run.group}] -> {name!r} [{group}] tags={tags}")
        changed += 1
        if args.apply:
            run.name, run.group, run.tags = name, group, tags
            run.update()
    print(f"{changed} run(s) {'updated' if args.apply else 'to update (dry run; pass --apply)'}, {skipped} skipped")


if __name__ == "__main__":
    main()
