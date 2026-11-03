"""Compare fine-tuning recipes on the 1.21 h put_relative subset (submit_ft_recipe.sh).

For every run and checkpoint: success on the 460 matrix scenes with a 95% Wilson interval, mean
progress, and exact McNemar against (a) the 200k parent with no fine-tuning and (b) the published
recipe (frac10) at the same step. Scenes are paired on world seed; a row scores whatever subset is
present and says how many.

    uv run --no-project python scripts/vista/compare_ft_recipes.py
"""

from matrix_summary import RUNS, load_run, mcnemar_p, score

PARENT = "put_relative_parent"
BASE = "repro_put_relative_frac10"
RECIPES = {
    BASE: "published recipe (LR 1e-4, all params)",
    "repro_put_relative_ft_lr1e5": "LR 1e-5, all params",
    "repro_put_relative_ft_vis01": "LR 1e-4, vision 1e-5",
}


def paired(ref, cur):
    keys = set(ref) & set(cur)
    only_ref = sum(ref[k][0] and not cur[k][0] for k in keys)
    only_cur = sum(cur[k][0] and not ref[k][0] for k in keys)
    return f"{only_cur - only_ref:+4d} of {len(keys)}, p={mcnemar_p(only_ref, only_cur):.3g}"


def main():
    parent = next(iter(load_run(RUNS / PARENT).values()))
    k, n, rate, lo, hi, prog = score(parent)
    print(f"parent, no fine-tuning: {k}/{n} = {rate:.1%} [{lo:.1%}, {hi:.1%}]  progress {prog:.3f}\n")
    base = load_run(RUNS / BASE) if (RUNS / BASE / "eval").is_dir() else {}
    print(f"{'recipe':40s} {'step':>6s} {'success':>15s} {'95% CI':>16s} {'prog':>6s}  "
          f"{'vs parent':>22s}  {'vs published @ step':>22s}")
    for run, label in RECIPES.items():
        if not (RUNS / run / "eval").is_dir():
            print(f"{label:40s}   (no evals yet)")
            continue
        for step, worlds in sorted(load_run(RUNS / run).items()):
            k, n, rate, lo, hi, prog = score(worlds)
            vs_base = paired(base[step], worlds) if run != BASE and step in base else "-"
            print(f"{label:40s} {step:6d} {k:5d}/{n:<3d} {rate:6.1%} [{lo:5.1%}, {hi:5.1%}] {prog:6.3f}  "
                  f"{paired(parent, worlds):>22s}  {vs_base:>22s}")


if __name__ == "__main__":
    main()
