"""Encode the pinned official Qwen3.5-4B FFN matrices directly as block-scaled NVFP4."""

import hashlib
import json
import struct

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from thorustinfer_build.quantization.vq import rotate

REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SHARDS = {
    "model.safetensors-00001-of-00002.safetensors": "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
    "model.safetensors-00002-of-00002.safetensors": "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
}
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
_codes = np.arange(127)
_e, _m = _codes >> 3, _codes & 7
E4M3 = np.where(_e == 0, _m * 2.0**-9, (1 + _m / 8) * 2.0 ** (_e - 7)).astype(np.float32)


def nearest_even(x, levels):
    mid = (levels[:-1] + levels[1:]) * 0.5
    code = np.searchsorted(mid, x).astype(np.uint8)
    tie_up = (code % 2 == 1) & (code < len(mid)) & (x == mid[np.minimum(code, len(mid) - 1)])
    return code + tie_up


def checkpoint(directory):
    tensors = {}
    for shard, expected in SHARDS.items():
        path = directory / shard
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != expected:
                raise ValueError(f"Expected official {REVISION} shard {shard}")
            f.seek(0)
            length = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(length))
        tensors.update(
            {
                k: (path, 8 + length + v["data_offsets"][0], v["shape"], v["dtype"])
                for k, v in header.items()
                if k != "__metadata__"
            }
        )
    return tensors


def encode_tensor(record, signs, shape):
    file, offset, source_shape, dtype = record
    if source_shape != list(shape) or dtype != "BF16":
        raise ValueError("Unexpected official FFN geometry")
    original = np.memmap(file, mode="r", offset=offset, shape=shape, dtype="<u2")
    rotated = rotate((original.astype("<u4") << 16).view("<f4"), signs)
    global_scale = np.float32(np.max(np.abs(rotated)) / (6 * 448))
    if not np.isfinite(global_scale) or global_scale <= 0:
        raise ValueError("Invalid FFN tensor scale")
    n, k = shape
    weights = np.empty((n, k // 2), dtype=np.uint8)
    scales = np.empty((n, k // 16), dtype=np.uint8)
    for first in range(0, n, 128):
        block = rotated[first : first + 128].reshape(-1, k // 16, 16)
        scale = nearest_even(np.max(np.abs(block), axis=-1) / (6 * global_scale), E4M3)
        multiplier = E4M3[scale][..., None] * global_scale
        vals = np.divide(np.abs(block), multiplier, out=np.zeros_like(block), where=multiplier != 0)
        codes = nearest_even(vals, E2M1) | (np.signbit(block).astype("u1") << 3)
        weights[first : first + 128] = (codes[..., ::2] | (codes[..., 1::2] << 4)).reshape(
            -1, k // 2
        )
        scales[first : first + 128] = scale
    scales = scales.reshape(n // 128, 4, 32, k // 64, 4).transpose(0, 3, 2, 1, 4).copy()
    return weights.ravel(), scales.ravel(), np.array([global_scale], dtype="<f4")


def generate(source, e8p, output):
    tensors = checkpoint(source)
    index = json.loads((e8p / "cache/weights/model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    signs = {}
    for k in (2560, 9216):
        name = f"orinfer.rotation.k{k}.signs"
        with safe_open(e8p / "cache/weights" / index[name], framework="np") as f:
            signs[k] = f.get_tensor(name)
    for layer in range(32):
        arrays = {}
        for role in ("gate", "up", "down"):
            n, k = (2560, 9216) if role == "down" else (9216, 2560)
            matrix = tensors[f"model.language_model.layers.{layer}.mlp.{role}_proj.weight"]
            values = encode_tensor(matrix, signs[k], (n, k))
            for suffix, value in zip(("W", "S", "G"), values, strict=True):
                arrays[f"NVFP4.{layer}.{role}.{suffix}"] = value
        filename = f"nvfp4-layer-{layer:02}.safetensors"
        save_file(
            arrays,
            str(output / filename),
            metadata={f"orin.layout.{k}": "native_contiguous" for k in arrays},
        )
        print(f"Encoded FFN layer {layer + 1}/32", flush=True)
