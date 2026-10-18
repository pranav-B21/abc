"""Summarize Max's 2x2 matrix and the put_relative data-scale sweep, with figures.

Reads every <run>/eval/<label>_step<N>_seed<S>/summary.json under abc_runs and scores each
(run, step) on the fixed scene set (5 seeds x 20 worlds = 100 scenes; scored on whatever subset
is present, and the table says how many). Outputs to --out:

    matrix_table.csv        every (cell, step): successes, n, rate, 95% Wilson CI, mean progress,
                            paired McNemar vs the pre-trained single-task cell at the same step
    matrix_2x2.png          the headline: success at the final step per cell, with CIs
    matrix_steps.png        success vs training step, one panel per initialization
    data_scale.png          put_relative success vs hours of task data (pre-trained, 25k)

    uv run --no-project --with matplotlib python scripts/vista/matrix_summary.py \
        --out /scratch/11138/pranavbelligundu/abc_runs/matrix_summary
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

RUNS = Path("/scratch/11138/pranavbelligundu/abc_runs")
SEEDS = [20260511, 20260531, 20260551, 20260571, 20260591]

# run dir -> (init, data); init in pretrained | scratch_dino | scratch_rand
CELLS = {
    "repro_put_relative_frac100": ("pretrained", "single"),
    "repro_put_relative_scratch_dino": ("scratch_dino", "single"),
    "repro_put_relative_scratch_rand": ("scratch_rand", "single"),
    "repro_mt6_pretrained": ("pretrained", "multi"),
    "repro_mt6_scratch_dino": ("scratch_dino", "multi"),
    "repro_mt6_scratch_rand": ("scratch_rand", "multi"),
}
EXTRA = {
    "repro_put_relative_scratch_dino_75k": "no pre-train (DINOv3), single, 75k",
    "put_relative_parent": "parent, zero-shot",
    "put_relative_released": "released put_relative @25k",
}
FRACS = {  # run dir -> hours of put_relative train data (subset.json / manifest)
    "repro_put_relative_frac10": 1.21,
    "repro_put_relative_frac25": 3.02,
    "repro_put_relative_frac50": 6.05,
    "repro_put_relative_frac100": 12.09,
}
INIT_LABEL = {
    "pretrained": "Pre-trained\n(ABC 200k parent)",
    "scratch_dino": "No pre-training\n(DINOv3 vision, random DiT)",
    "scratch_rand": "No pre-training\n(all random)",
}
DATA_LABEL = {"single": "Single-task (put_relative)", "multi": "Multitask (6 tasks)"}
# Validated default categorical slots 1-2 (dataviz palette.md), light surface.
COLOR = {"single": "#2a78d6", "multi": "#eb6834"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar_p(b, c):
    """Exact two-sided McNemar (binomial on discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def load_run(run: Path):
    """step -> {world_seed: (success, max_progress)} over the matrix seeds."""
    out = {}
    for s in sorted((run / "eval").glob("*/summary.json")):
        m = re.search(r"_step(\d+)_seed(\d+)$", s.parent.name)
        if not m or int(m[2]) not in SEEDS:
            continue
        step = int(m[1])
        for w in json.loads(s.read_text())["worlds"]:
            out.setdefault(step, {})[int(w["world_seed"])] = (
                bool(w["success"]),
                float(w["max_reward"]),  # per-world max progress; its mean is summary's mean_max_progress
            )
    return out


def score(worlds):
    k = sum(s for s, _ in worlds.values())
    n = len(worlds)
    lo, hi = wilson(k, n)
    prog = sum(p for _, p in worlds.values()) / n if n else math.nan
    return k, n, k / n if n else math.nan, lo, hi, prog


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=RUNS / "matrix_summary")
    ap.add_argument("--final-step", type=int, default=25000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    names = {**{k: f"{v[0]} / {v[1]}" for k, v in CELLS.items()}, **EXTRA,
             **{k: f"pre-trained, {h:g} h put_relative" for k, h in FRACS.items()}}
    data = {r: load_run(RUNS / r) for r in names if (RUNS / r / "eval").is_dir()}
    ref = data.get("repro_put_relative_frac100", {})

    rows = []
    for run, steps in data.items():
        for step, worlds in sorted(steps.items()):
            k, n, rate, lo, hi, prog = score(worlds)
            p = ""
            base = ref.get(step)
            if base and run != "repro_put_relative_frac100":
                common = set(base) & set(worlds)
                b = sum(base[w][0] and not worlds[w][0] for w in common)
                c = sum(worlds[w][0] and not base[w][0] for w in common)
                p = f"{mcnemar_p(b, c):.3g} (ref-only {b}, this-only {c}, n={len(common)})"
            rows.append({"run": run, "cell": names[run], "step": step, "successes": k, "n": n,
                         "rate": round(rate, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                         "mean_progress": round(prog, 3), "paired_vs_pretrained_single": p})
    with open(args.out / "matrix_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["run"])
        w.writeheader()
        w.writerows(rows)
    print(f"{'cell':45s} {'step':>6s} {'succ/n':>8s} {'rate':>6s} {'95% CI':>13s}  paired vs pre-trained single")
    for r in rows:
        print(f"{r['cell']:45s} {r['step']:6d} {r['successes']:3d}/{r['n']:<4d} {r['rate']:6.2f} "
              f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]  {r['paired_vs_pretrained_single']}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False})

    def style(ax):
        ax.set_ylim(0, 1)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)

    # 1. Headline 2x2 (+ the all-random variant): final-step success with Wilson CIs.
    fig, ax = plt.subplots(figsize=(8, 4.6))
    inits = list(INIT_LABEL)
    width = 0.34
    for j, kind in enumerate(("single", "multi")):
        for i, init in enumerate(inits):
            run = next((r for r, v in CELLS.items() if v == (init, kind)), None)
            worlds = data.get(run, {}).get(args.final_step)
            if not worlds:
                continue
            k, n, rate, lo, hi, _ = score(worlds)
            x = i + (j - 0.5) * (width + 0.03)
            ax.bar(x, rate, width, color=COLOR[kind], edgecolor=SURFACE, lw=2,
                   label=DATA_LABEL[kind] if i == 0 or not ax.get_legend_handles_labels()[1].count(DATA_LABEL[kind]) else None)
            ax.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]], color=INK, lw=1.2, capsize=3)
            ax.text(x, hi + 0.02, f"{k}/{n}", ha="center", va="bottom", fontsize=9, color=INK)
    for r, txt in (("put_relative_parent", "parent, no fine-tuning"), ("put_relative_released", "released checkpoint")):
        worlds = data.get(r, {})
        worlds = next(iter(worlds.values()), None)
        if worlds:
            rate = score(worlds)[2]
            ax.axhline(rate, color=INK2, lw=1, ls="--")
            ax.text(len(inits) - 0.42, rate + 0.01, f"{txt} ({rate:.2f})", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set_xlim(-0.6, len(inits) - 0.4)
    ax.set_xticks(range(len(inits)), [INIT_LABEL[i] for i in inits])
    ax.set_ylabel("put_relative success rate")
    ax.set_title(f"Pre-training x multitask, put_relative, step {args.final_step // 1000}k "
                 "(95% Wilson CI)", color=INK, fontsize=11, loc="left")
    style(ax)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out / "matrix_2x2.png", dpi=180)

    # 2. Success vs step, one panel per initialization.
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, init in zip(axes, inits):
        for kind in ("single", "multi"):
            run = next((r for r, v in CELLS.items() if v == (init, kind)), None)
            steps = data.get(run, {})
            if not steps:
                continue
            xs = sorted(steps)
            sc = [score(steps[s]) for s in xs]
            ax.plot([s / 1000 for s in xs], [c[2] for c in sc], color=COLOR[kind], lw=2, marker="o", ms=4,
                    label=DATA_LABEL[kind])
            ax.fill_between([s / 1000 for s in xs], [c[3] for c in sc], [c[4] for c in sc],
                            color=COLOR[kind], alpha=0.15, lw=0)
        if init == "scratch_dino" and "repro_put_relative_scratch_dino_75k" in data:
            steps = data["repro_put_relative_scratch_dino_75k"]
            xs = sorted(steps)
            ax.plot([s / 1000 for s in xs], [score(steps[s])[2] for s in xs], color=COLOR["single"], lw=2,
                    ls=":", marker="s", ms=4, label="Single-task, 75k run")
        ax.set_title(INIT_LABEL[init].replace("\n", " "), fontsize=9, color=INK, loc="left")
        ax.set_xlim(0, 77 if init == "scratch_dino" and "repro_put_relative_scratch_dino_75k" in data else 26)
        ax.set_xlabel("training step (k)")
        style(ax)
    axes[0].set_ylabel("put_relative success rate")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out / "matrix_steps.png", dpi=180)

    # 3. Data scale: pre-trained, single-task, final step, vs hours of put_relative data.
    pts = []
    for run, hours in FRACS.items():
        worlds = data.get(run, {}).get(args.final_step)
        if worlds:
            pts.append((hours, *score(worlds)))
    parent = next(iter(data.get("put_relative_parent", {}).values()), None)
    if pts:
        fig, ax = plt.subplots(figsize=(6, 4))
        pts.sort()
        hx = [p[0] for p in pts]
        ax.plot(hx, [p[3] for p in pts], color=COLOR["single"], lw=2, marker="o", ms=6, label="fine-tuned (25k steps)")
        ax.errorbar(hx, [p[3] for p in pts], yerr=[[p[3] - p[4] for p in pts], [p[5] - p[3] for p in pts]],
                    color=INK, lw=1, capsize=3, ls="none")
        for p in pts:
            ax.text(p[0], p[5] + 0.02, f"{p[1]}/{p[2]}", ha="center", fontsize=8, color=INK)
        if parent:
            rate = score(parent)[2]
            ax.axhline(rate, color=INK2, lw=1, ls="--")
            ax.text(16, rate + 0.01, f"parent, 0 h of task data ({rate:.2f})", ha="right", fontsize=8, color=INK2)
        ax.set_xscale("log")
        ax.set_xlim(0.9, 17)
        ax.minorticks_off()
        ax.set_xticks(hx, [f"{h:g} h" for h in hx])
        ax.set_xlabel("hours of put_relative sim data")
        ax.set_ylabel("put_relative success rate")
        ax.set_title("Data scale, pre-trained, single-task", fontsize=11, color=INK, loc="left")
        style(ax)
        fig.tight_layout()
        fig.savefig(args.out / "data_scale.png", dpi=180)
    print(f"wrote {args.out}/{{matrix_table.csv,matrix_2x2.png,matrix_steps.png,data_scale.png}}")


if __name__ == "__main__":
    main()
