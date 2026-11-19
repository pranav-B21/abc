"""Analyze fine-tuning recipe round 2 (submit_ft_recipe2.sh).

For every round-2 run and evaluated checkpoint: success on the 460 matrix scenes (95% Wilson interval), progress,
val_loss at that checkpoint (from W&B), and exact McNemar against (a) the parent with no fine-tuning and (b) the
published recipe on the same data at the same step (frac10/25/50/100; for steps the published run never saved,
its nearest saved step). Then the pre-registered result per run: the checkpoint with the lowest val_loss, and
how far that pick falls short of the best checkpoint read on the test scenes.

    source vista_env.sh && uv run --no-sync python scripts/vista/compare_ft_recipes2.py
"""

from matrix_summary import RUNS, load_run, mcnemar_p, score

ENTITY = "pranavbelligundu-the-university-of-texas-at-austin/abc-vista"
PARENT = "put_relative_parent"
ROUND2 = {  # run dir -> (label, published-recipe run on the same data)
    "repro_put_relative_ft2_lr1e5_es": ("1.21 h, LR 1e-5, ckpt every 1k", "repro_put_relative_frac10"),
    "repro_put_relative_ft2_lr3e6": ("1.21 h, LR 3e-6", "repro_put_relative_frac10"),
    "repro_put_relative_ft2_lr1e5_3h": ("3.02 h, LR 1e-5", "repro_put_relative_frac25"),
    "repro_put_relative_ft2_lr1e5_6h": ("6.05 h, LR 1e-5", "repro_put_relative_frac50"),
    "repro_put_relative_ft2_lr1e5_12h": ("12.09 h, LR 1e-5", "repro_put_relative_frac100"),
}


def paired(ref, cur):
    keys = set(ref) & set(cur)
    only_ref = sum(ref[k][0] and not cur[k][0] for k in keys)
    only_cur = sum(cur[k][0] and not ref[k][0] for k in keys)
    return f"{only_cur - only_ref:+4d}/{len(keys)} p={mcnemar_p(only_ref, only_cur):.2g}"


def val_losses(run_id):
    try:
        import wandb
        r = wandb.Api(timeout=60).run(f"{ENTITY}/{run_id}")
        return {int(x["_step"]): float(x["val_loss"]) for x in r.scan_history(keys=["_step", "val_loss"])}
    except Exception as e:  # no W&B access: still print the success table
        print(f"  [warn] no val_loss for {run_id}: {type(e).__name__}")
        return {}


def main():
    parent = next(iter(load_run(RUNS / PARENT).values()))
    k, n, rate, lo, hi, prog = score(parent)
    print(f"parent, no fine-tuning: {k}/{n} = {rate:.1%} [{lo:.1%}, {hi:.1%}]  progress {prog:.3f}\n")
    for run, (label, base_run) in ROUND2.items():
        if not (RUNS / run / "eval").is_dir():
            print(f"{label}: no evals yet\n")
            continue
        base = load_run(RUNS / base_run)
        vl = val_losses(run)
        rows = []
        print(f"== {label}  (published recipe on this data: {base_run})")
        print(f"   {'step':>6s} {'success':>15s} {'95% CI':>16s} {'prog':>6s} {'val_loss':>9s}  "
              f"{'vs parent':>18s}  {'vs published':>18s}")
        for step, worlds in sorted(load_run(RUNS / run).items()):
            k, n, rate, lo, hi, prog = score(worlds)
            bstep = min(base, key=lambda s: abs(s - step)) if base else None
            vs_b = f"{paired(base[bstep], worlds)}@{bstep // 1000}k" if bstep is not None else "-"
            v = vl.get(step)
            rows.append((step, rate, v))
            print(f"   {step:6d} {k:5d}/{n:<3d} {rate:6.1%} [{lo:5.1%}, {hi:5.1%}] {prog:6.3f} "
                  f"{v if v is None else round(v, 4)!s:>9s}  {paired(parent, worlds):>18s}  {vs_b:>18s}")
        scored = [r for r in rows if r[2] is not None]
        if scored:
            pick = min(scored, key=lambda r: r[2])
            best = max(rows, key=lambda r: r[1])
            print(f"   pre-registered pick (lowest val_loss): step {pick[0]} -> {pick[1]:.1%}; "
                  f"best on test scenes: step {best[0]} -> {best[1]:.1%}")
        print()


if __name__ == "__main__":
    main()
