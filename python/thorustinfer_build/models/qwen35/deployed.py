"""Adopt verified deployment shards without re-encoding their physical tensors."""

import hashlib
import math

import numpy as np
from safetensors.numpy import save_file

from thorustinfer_build.common.files import file_hash, write_json
from thorustinfer_build.packaging.publication import link_or_copy, source_path
from thorustinfer_build.weights.safetensors_source import local_header

from .storage import DTYPES, Weights


class DeployedWeights(Weights):
    def __init__(self, source, root, hashes):
        super().__init__(source / "cache/weights", root, packed_k=128)
        files = {p.name for p in self.source.glob("*.safetensors")}
        if set(hashes) != files or not set(self.index["weight_map"].values()) <= files:
            raise ValueError("Expected a SHA-256 for every deployment shard")
        for file in sorted(files):
            path = source_path(self.source, file)
            if file_hash(path) != hashes[file]:
                raise ValueError(f"Deployed shard identity changed: {file}")
            self.paths[file] = path
            self.headers[file] = local_header(path)

    def info(self, name):
        file = self.index["weight_map"][name]
        begin, header = self.headers[file]
        return self.paths[file], begin, header[name]

    def immutable(self, name, *, cast=None, shape=None):
        target = name if cast is None else f"Runtime.{name}.{cast}"
        if target in self.buffers:
            return target
        if target not in self.index["weight_map"]:
            return super().immutable(name, cast=cast, shape=shape)
        path, begin, info = self.info(target)
        dtype = DTYPES[info["dtype"]]
        if cast and np.dtype(cast).str != {"f16": "<f2", "f32": "<f4"}.get(dtype):
            raise ValueError(f"Unexpected deployed cast dtype: {target}")
        dims = info["shape"] if shape is None else list(shape)
        if math.prod(dims) != math.prod(info["shape"]):
            raise ValueError(f"Invalid deployed view: {target}")
        layout = self.headers[path.name][1]["__metadata__"][f"orin.layout.{target}"]
        if name.endswith(".packed") and (
            layout != "e8p-q4-k128-u32"
            or dtype != "u32"
            or len(dims) not in (3, 4)
            or dims[-1] != 16
        ):
            raise ValueError(f"Expected physical E8P K128 bank: {target}")
        lo, hi = info["data_offsets"]
        payload = np.memmap(path, mode="r", offset=begin + lo, dtype="u1", shape=hi - lo)
        self.buffers[target] = dict(
            name=target,
            dtype=dtype,
            shape=dims,
            layout=layout,
            alignment=256,
            access="read",
            data=dict(tensor=target, sha256=hashlib.sha256(payload).hexdigest()),
        )
        self.mapping[target] = path.name
        self.scopes[target] = "weights"
        return target

    def combined(self, name, inputs):
        if name in self.index["weight_map"]:
            return self.immutable(name)
        return super().combined(name, inputs)

    def finish(self):
        extras = {}
        derived = "derived.safetensors"
        while derived in self.paths:
            derived = "new-" + derived
        for name, data in self.extra.items():
            if name in self.index["weight_map"]:
                path, _, info = self.info(name)
                if (
                    list(data.shape) == info["shape"]
                    and DTYPES[info["dtype"]] == self.buffers[name]["dtype"]
                    and self.raw(name)[0] == data.tobytes()
                ):
                    self.mapping[name] = path.name
                    continue
            self.mapping[name] = derived
            extras[name] = data
        for filename, path in self.paths.items():
            link_or_copy(path, self.directory / filename)
        if extras:
            save_file(
                extras,
                str(self.directory / derived),
                metadata={f"orin.layout.{n}": self.buffers[n]["layout"] for n in extras},
            )
        total = sum(
            math.prod(b["shape"])
            * {"f16": 2, "f32": 4, "bf16": 2, "i8": 1, "i32": 4, "u32": 4, "u8": 1}[b["dtype"]]
            for b in self.buffers.values()
            if b["access"] == "read"
        )
        write_json(
            self.directory / "model.safetensors.index.json",
            dict(metadata=dict(total_size=total), weight_map=self.mapping),
        )
        return total
