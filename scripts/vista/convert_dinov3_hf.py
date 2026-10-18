"""Convert the Hugging Face DINOv3 ViT-B/16 checkpoint to the key layout abc_minimal expects.

From-scratch DiT training (no --load-pretrained) loads
<cache-root>/dinov3_vitb16_pretrain_lvd1689m.pth into DinoVisionBackbone.dinov3_model with
strict=False (abc_minimal/train_loop.py:232). That code expects Meta's original key names
(blocks.N.attn.qkv, ls1.gamma, storage_tokens, ...). The HF repo
facebook/dinov3-vitb16-pretrain-lvd1689m ships only transformers-format model.safetensors
(layer.N.attention.q_proj, layer_scale1.lambda1, register_tokens, ...), so loading it as-is
matches ZERO keys and silently trains a random vision encoder. This script renames, fuses q/k/v
into qkv (DINOv3 has no key bias, so k's bias slot is zero), checks that every non-buffer key of
the model is filled with the right shape, and writes the .pth.

    python scripts/vista/convert_dinov3_hf.py \
        --hf $ABC_CACHE/dinov3_hf/model.safetensors \
        --out $ABC_CACHE/dinov3_vitb16_pretrain_lvd1689m.pth \
        [--compare-ckpt $SCRATCH/abc_ft/put_relative/abc_dit_xl_200k_model.pt]

--compare-ckpt reports per-tensor cosine similarity between the converted weights and the vision
encoder inside an ABC checkpoint. ABC's pretraining started from DINOv3 and trained the encoder at
0.1x LR, so a correct conversion is strongly correlated with it, while a wrong mapping is not.
"""

import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np
import torch

from abc_minimal.config import DiTConfig
from abc_minimal.dit import DinoVisionBackbone

# Buffers the model computes itself; Meta's .pth may or may not carry them.
MODEL_OWNED = re.compile(r"(^rope_embed\.periods$|\.attn\.qkv\.bias_mask$)")

DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": None}


def load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    """Minimal safetensors reader (the abc venv has no safetensors package)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    header.pop("__metadata__", None)
    out = {}
    for name, meta in header.items():
        start, end = meta["data_offsets"]
        raw = data[start:end]
        if meta["dtype"] == "BF16":
            t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float()
        else:
            t = torch.from_numpy(np.frombuffer(raw, dtype=DTYPES[meta["dtype"]]).copy()).float()
        out[name] = t.reshape(meta["shape"])
    return out


def convert(hf: dict[str, torch.Tensor], num_layers: int) -> dict[str, torch.Tensor]:
    sd = {
        "cls_token": hf["embeddings.cls_token"],
        "mask_token": hf["embeddings.mask_token"].reshape(1, -1),
        "storage_tokens": hf["embeddings.register_tokens"],
        "patch_embed.proj.weight": hf["embeddings.patch_embeddings.weight"],
        "patch_embed.proj.bias": hf["embeddings.patch_embeddings.bias"],
        "norm.weight": hf["norm.weight"],
        "norm.bias": hf["norm.bias"],
    }
    for i in range(num_layers):
        h, b = f"layer.{i}.", f"blocks.{i}."
        q, k, v = (hf[h + f"attention.{x}_proj.weight"] for x in "qkv")
        sd[b + "attn.qkv.weight"] = torch.cat([q, k, v], dim=0)
        qb = hf[h + "attention.q_proj.bias"]
        kb = hf.get(h + "attention.k_proj.bias", torch.zeros_like(qb))  # key_bias=False in DINOv3
        sd[b + "attn.qkv.bias"] = torch.cat([qb, kb, hf[h + "attention.v_proj.bias"]], dim=0)
        sd[b + "attn.proj.weight"] = hf[h + "attention.o_proj.weight"]
        sd[b + "attn.proj.bias"] = hf[h + "attention.o_proj.bias"]
        sd[b + "ls1.gamma"] = hf[h + "layer_scale1.lambda1"]
        sd[b + "ls2.gamma"] = hf[h + "layer_scale2.lambda1"]
        sd[b + "mlp.fc1.weight"] = hf[h + "mlp.up_proj.weight"]
        sd[b + "mlp.fc1.bias"] = hf[h + "mlp.up_proj.bias"]
        sd[b + "mlp.fc2.weight"] = hf[h + "mlp.down_proj.weight"]
        sd[b + "mlp.fc2.bias"] = hf[h + "mlp.down_proj.bias"]
        for n in ("norm1", "norm2"):
            sd[b + f"{n}.weight"] = hf[h + f"{n}.weight"]
            sd[b + f"{n}.bias"] = hf[h + f"{n}.bias"]
    return sd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", type=Path, required=True, help="HF model.safetensors")
    ap.add_argument("--out", type=Path, required=True, help="output .pth")
    ap.add_argument("--compare-ckpt", type=Path, help="ABC checkpoint whose vision encoder to compare against")
    args = ap.parse_args()

    hf = load_safetensors(args.hf)
    model = DinoVisionBackbone(DiTConfig()).dinov3_model
    expected = model.state_dict()
    num_layers = len({k.split(".")[1] for k in expected if k.startswith("blocks.")})
    sd = convert(hf, num_layers)

    used = len(sd)
    missing = [k for k in expected if k not in sd and not MODEL_OWNED.search(k)]
    extra = [k for k in sd if k not in expected]
    bad_shape = [k for k in sd if k in expected and tuple(sd[k].shape) != tuple(expected[k].shape)]
    if missing or extra or bad_shape:
        raise SystemExit(f"conversion incomplete: missing={missing} extra={extra} bad_shape={bad_shape}")
    # The exact load train_loop.py performs, so its printed counts are what training will see.
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"converted {used} tensors from {len(hf)} HF tensors; "
          f"load_state_dict missing={len(miss)} (model-owned buffers: {miss}) unexpected={len(unexp)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, args.out)
    print(f"wrote {args.out}")

    if args.compare_ckpt:
        ck = torch.load(args.compare_ckpt, map_location="cpu", mmap=True, weights_only=False)
        weights = ck.get("model", ck)
        prefix = "img_backbone.dinov3_model."
        rand = DinoVisionBackbone(DiTConfig()).dinov3_model.state_dict()
        rows = []
        for k, t in sd.items():
            ref = weights.get(prefix + k)
            if ref is None or t.numel() < 64:
                continue
            ref = ref.float().flatten()
            cos = torch.nn.functional.cosine_similarity(t.flatten(), ref, dim=0).item()
            cos_rand = torch.nn.functional.cosine_similarity(rand[k].float().flatten(), ref, dim=0).item()
            rows.append((k, cos, cos_rand))
        cos = np.array([r[1] for r in rows])
        cos_r = np.array([r[2] for r in rows])
        print(f"vs {args.compare_ckpt.name}: {len(rows)} tensors, cosine converted median "
              f"{np.median(cos):.3f} (min {cos.min():.3f}), random-init median {np.median(cos_r):.3f}")
        for k, c, cr in sorted(rows, key=lambda r: r[1])[:5]:
            print(f"  lowest: {k:40s} converted {c:.3f}  random {cr:.3f}")


if __name__ == "__main__":
    main()
