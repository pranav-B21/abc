"""Build nested, hours-matched training subsets of one sim task for a data-scale sweep.

TrainConfig has no data-fraction knob: the sim_task preset reads every episode under
<cache-root>/train_sim/. So each fraction gets its own cache root whose train_sim/ holds
symlinks to a subset of the full root's episode dirs. Everything else is shared with the
full root (symlinked), so validation, the parent checkpoint and the norm stats it embeds
are identical at every scale.

Subsets are nested (10% c 25% c 50% c 100%): episodes are shuffled once with a fixed seed
and each fraction takes the shortest prefix of that order reaching its share of FRAMES
(= hours at 30 fps), not of episodes, since episode lengths vary.

    python3 scripts/vista/make_data_subsets.py --full-root $SCRATCH/abc_ft/put_relative \
        --fractions 0.1 0.25 0.5
    # -> $SCRATCH/abc_ft/put_relative_frac10/, ..._frac25/, ..._frac50/, each with subset.json
"""

import argparse
import json
import random
from pathlib import Path

FPS = 30
SHARED = ("val_sim", "abc_dit_xl_200k_model.pt", "abc_dit_xl_200k_model.json", "sim_224_manifest.json")


def episode_sizes(train_dir: Path) -> dict[str, int]:
    """Episode dir name -> states_actions.bin bytes (proportional to its frame count)."""
    return {
        p.name: (p / "states_actions.bin").stat().st_size
        for p in sorted(train_dir.iterdir())
        if (p / "states_actions.bin").exists()
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full-root", type=Path, required=True, help="cache root holding the full train_sim/")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames-total", type=int, help="override; default reads sim_224_manifest.json")
    args = ap.parse_args()

    full = args.full_root.resolve()
    task = full.name
    sizes = episode_sizes(full / "train_sim")
    total_bytes = sum(sizes.values())

    frames_total = args.frames_total
    if frames_total is None:
        manifest = json.loads((full / "sim_224_manifest.json").read_text())
        entry = manifest["tasks"][task]
        # The manifest counts train + val frames; scale train bytes by the per-frame width
        # recovered from the val split so hours are exact for the train split alone.
        val_bytes = sum(episode_sizes(full / "val_sim").values())
        frames_total = round(entry["frames"] * total_bytes / (total_bytes + val_bytes))
    bytes_per_frame = total_bytes / frames_total

    order = sorted(sizes)
    random.Random(args.seed).shuffle(order)

    for frac in sorted(args.fractions):
        target = frac * total_bytes
        picked, acc = [], 0
        for name in order:
            if acc >= target:
                break
            picked.append(name)
            acc += sizes[name]

        root = full.parent / f"{task}_frac{round(frac * 100)}"
        train = root / "train_sim"
        train.mkdir(parents=True, exist_ok=True)
        for stale in train.iterdir():  # rebuild cleanly; only ever symlinks here
            if stale.is_symlink():
                stale.unlink()
        for name in picked:
            (train / name).symlink_to(full / "train_sim" / name)
        for item in SHARED:
            link = root / item
            if (full / item).exists() and not link.exists():
                link.symlink_to((full / item).resolve() if (full / item).is_symlink() else full / item)

        frames = acc / bytes_per_frame
        summary = {
            "task": task,
            "full_root": str(full),
            "fraction_target": frac,
            "fraction_actual_frames": acc / total_bytes,
            "seed": args.seed,
            "episodes": len(picked),
            "episodes_full": len(sizes),
            "frames": round(frames),
            "hours": round(frames / FPS / 3600, 3),
            "hours_full": round(frames_total / FPS / 3600, 3),
            "episode_ids": picked,
        }
        (root / "subset.json").write_text(json.dumps(summary, indent=1))
        print(
            f"{root.name}: {len(picked)}/{len(sizes)} episodes, "
            f"{summary['hours']:.2f}/{summary['hours_full']:.2f} h ({summary['fraction_actual_frames']:.3f})"
        )


if __name__ == "__main__":
    main()
