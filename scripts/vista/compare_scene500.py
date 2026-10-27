"""Fine-tuned vs not fine-tuned on put_relative, paired over up to 500 scenes (submit_scene500.sh).

Compares the 200k parent (no fine-tuning) with our 12 h fine-tune at 25k on every scene both
have been evaluated on: success (exact McNemar on discordant scenes) and per-scene max progress
(paired sign-flip permutation test). Also prints the matrix's first 100 scenes alone, so the
500-scene answer can be read against the earlier one.

    uv run --no-project python scripts/vista/compare_scene500.py
"""

import json
import math
import random
import re
from pathlib import Path

RUNS = Path("/scratch/11138/pranavbelligundu/abc_runs")
SEEDS = list(range(20260511, 20260992, 20))  # 25 seeds x 20 worlds = 500 scenes
MATRIX_SEEDS = SEEDS[:5]
MODELS = {"no fine-tuning (200k parent)": ("put_relative_parent", 0),
          "fine-tuned 12 h, step 25k": ("repro_put_relative_frac100", 25000)}


def load(run, step, seeds):
    out = {}
    for s in (RUNS / run / "eval").glob(f"*_step{step}_seed*/summary.json"):
        seed = int(re.search(r"_seed(\d+)$", s.parent.name)[1])
        if seed in seeds:
            for w in json.loads(s.read_text())["worlds"]:
                out[int(w["world_seed"])] = (bool(w["success"]), float(w["max_reward"]))
    return out


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def perm_p(d, iters=20000):
    rng = random.Random(0)
    obs = abs(sum(d))
    return sum(abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= obs - 1e-12
               for _ in range(iters)) / iters


def report(title, seeds):
    (na, (ra, sa)), (nb, (rb, sb)) = MODELS.items()
    A, B = load(ra, sa, seeds), load(rb, sb, seeds)
    ks = sorted(set(A) & set(B))
    n = len(ks)
    print(f"\n{title}: {n} scenes evaluated by both")
    if not n:
        return
    for name, M in ((na, A), (nb, B)):
        k = sum(M[s][0] for s in ks)
        lo, hi = wilson(k, n)
        prog = sum(M[s][1] for s in ks) / n
        print(f"  {name:30s} success {k:3d}/{n} = {k / n:.1%}  [{lo:.1%}, {hi:.1%}]  progress {prog:.3f}")
    b = sum(A[s][0] and not B[s][0] for s in ks)
    c = sum(B[s][0] and not A[s][0] for s in ks)
    d = [B[s][1] - A[s][1] for s in ks]
    print(f"  success: only no-fine-tuning {b}, only fine-tuned {c}, "
          f"difference {(c - b) / n:+.1%}, McNemar p = {mcnemar_p(b, c):.3g}")
    print(f"  progress: difference {sum(d) / n:+.3f}, fine-tuned better on {sum(x > 0 for x in d)} "
          f"/ worse on {sum(x < 0 for x in d)} scenes, permutation p = {perm_p(d):.3g}")


if __name__ == "__main__":
    report("Matrix scenes only (first 100)", set(MATRIX_SEEDS))
    report("All scenes", set(SEEDS))
