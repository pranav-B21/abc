"""Build the cache roots for Max's 2x2 matrix (pre-trained vs no pre-training x single- vs multitask).

Every root is symlinks only, so the cells share episodes, validation and normalization:

    put_relative                  (exists)  single-task, pre-trained: parent + put_relative episodes
    put_relative_scratch_dino     single-task, no pre-training, DINOv3 vision encoder
    put_relative_scratch_rand     single-task, no pre-training, random vision encoder
    mt6                           train_sim/ = every episode of the 6 tasks (shared by the 3 below)
    mt6_pretrained                multitask, pre-trained
    mt6_scratch_dino              multitask, no pre-training, DINOv3 vision encoder
    mt6_scratch_rand              multitask, no pre-training, random vision encoder

- val_sim/ is put_relative's in every root, so validation loss is on the evaluation task.
- Scratch roots get norm_stats.json = the parent checkpoint's embedded stats, so inputs are scaled
  exactly as in the pre-trained cells (train.py needs the file when nothing is loaded).
- *_dino roots link $ABC_CACHE/dinov3_vitb16_pretrain_lvd1689m.pth (from convert_dinov3_hf.py);
  *_rand roots must not contain it, or train.py loads it.

    python scripts/vista/make_matrix_roots.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

FT = Path("/scratch/11138/pranavbelligundu/abc_ft")
ABC_CACHE = Path("/scratch/11138/pranavbelligundu/abc_cache")
DINO = "dinov3_vitb16_pretrain_lvd1689m.pth"
PARENT = "abc_dit_xl_200k_model.pt"
TARGET = "put_relative"
MT6 = [
    "put_relative",
    "sim_pouring_beads",
    "sim_inhand_transfer_the_item_to_other_side",
    "grab_clutter",
    "conveyor_pick",
    "count_into_opaque_box",
]
FPS = 30


def link(dst: Path, src: Path):
    src = src.resolve()
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src:
            return
        raise SystemExit(f"{dst} exists and is not a link to {src}; remove it first")
    dst.symlink_to(src)


def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return np.asarray(x).tolist()
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


def episode_frames(ep: Path) -> int:
    meta = json.loads((ep / "episode_metadata.json").read_text())
    return int(meta["num_steps"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()

    target = FT / TARGET
    parent = target / PARENT
    dino = ABC_CACHE / DINO
    for p in (target / "train_sim", target / "val_sim", parent, dino):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    # Parent's embedded normalization stats -> norm_stats.json for the scratch roots.
    stats_path = FT / "matrix_norm_stats.json"
    if not stats_path.exists():
        ck = torch.load(parent, map_location="cpu", mmap=True, weights_only=False)
        if "norm_stats" not in ck:
            raise SystemExit(f"{parent} embeds no norm_stats")
        stats_path.write_text(json.dumps({"norm_stats": to_jsonable(ck["norm_stats"])}))
        print(f"wrote {stats_path} (parent's embedded norm_stats)")

    # Multitask train_sim: every train episode of the 6 tasks, by symlink.
    mt_train = FT / "mt6" / "train_sim"
    mt_train.mkdir(parents=True, exist_ok=True)
    composition = {}
    for task in MT6:
        src = FT / task / "train_sim"
        if not src.is_dir():
            raise SystemExit(f"missing {src}: run prepare.py --sim-data {task} --cache {FT / task}")
        eps = sorted(p for p in src.iterdir() if (p / "states_actions.bin").exists())
        frames = 0
        for ep in eps:
            dst = mt_train / ep.name
            if dst.is_symlink() and dst.resolve() != ep.resolve():
                raise SystemExit(f"episode id collision: {ep.name}")
            link(dst, ep)
            frames += episode_frames(ep)
        composition[task] = {"episodes": len(eps), "hours": round(frames / FPS / 3600, 2)}
    total_h = sum(c["hours"] for c in composition.values())
    for c in composition.values():
        c["share"] = round(c["hours"] / total_h, 3)
    (FT / "mt6" / "composition.json").write_text(json.dumps(composition, indent=1))
    print(f"mt6: {sum(c['episodes'] for c in composition.values())} episodes, {total_h:.1f} h")
    for task, c in composition.items():
        print(f"  {task:45s} {c['episodes']:5d} eps {c['hours']:6.2f} h  {c['share']:.1%}")

    roots = {
        "put_relative_scratch_dino": (target / "train_sim", "scratch", True),
        "put_relative_scratch_rand": (target / "train_sim", "scratch", False),
        "mt6_pretrained": (mt_train, "pretrained", None),
        "mt6_scratch_dino": (mt_train, "scratch", True),
        "mt6_scratch_rand": (mt_train, "scratch", False),
    }
    for name, (train, init, with_dino) in roots.items():
        root = FT / name
        root.mkdir(exist_ok=True)
        link(root / "train_sim", train)
        link(root / "val_sim", target / "val_sim")
        link(root / "sim_224_manifest.json", target / "sim_224_manifest.json")
        if init == "pretrained":
            link(root / PARENT, parent)
            link(root / "abc_dit_xl_200k_model.json", target / "abc_dit_xl_200k_model.json")
        else:
            link(root / "norm_stats.json", stats_path)
            if with_dino:
                link(root / DINO, dino)
            elif (root / DINO).exists() or (root / DINO).is_symlink():
                raise SystemExit(f"{root / DINO} must not exist in a random-vision root")
        print(f"{name}: init={init} vision={'dinov3' if with_dino else ('parent' if with_dino is None else 'random')}")


if __name__ == "__main__":
    main()
