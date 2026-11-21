"""Publish a content-addressed Thor W4A4 sibling of a validated SM110a E8P model."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from safetensors.numpy import load_file, save_file
from thorustinfer_build.models.qwen35 import thor_weights
from thorustinfer_build.models.qwen35.build import context_capacity

ABI = b"orinfer_nvfp4_ffn_v1"
POLICY = "nvfp4_ffn"


def json_bytes(x):
    return json.dumps(x, separators=(",", ":")).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def share(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def keep_referenced(items, references):
    """Compact a package table and update every live index into that table."""
    indices = sorted({owner[key] for owner, key in references})
    mapping = {index: position for position, index in enumerate(indices)}
    for owner, key in references:
        owner[key] = mapping[owner[key]]
    return [items[index] for index in indices]


def publish(e8p, package, encoded, output):
    model = output / "model"
    weights = model / "cache/weights"
    weights.mkdir(parents=True)
    desc = json.loads((e8p / "cache/model.json").read_bytes())
    data = (package / "package.json").read_bytes()
    if digest(data) != desc["execution_package"]:
        raise ValueError("Base model and execution package identities differ")
    pkg = json.loads(data)
    if pkg["target"] != "sm_110a" or pkg["compute_policy"] != "int8_quality":
        raise ValueError("Expected an SM110a E8P base")
    context_capacity(
        json.loads((e8p / "config.json").read_bytes()), desc["metadata"]["max_context"]
    )
    packet_rows = desc["metadata"]["chunk_tokens"]
    for f in e8p.iterdir():
        if f.is_file():
            share(f, model / f.name)
    base_weights = e8p / "cache/weights"
    for f in base_weights.glob("*.safetensors"):
        share(f, weights / f.name)
    index = json.loads((base_weights / "model.safetensors.index.json").read_bytes())

    def contract(name, dtype, shape, identity=None):
        c = dict(
            name=name,
            dtype=dtype,
            shape=shape,
            layout="native_contiguous",
            alignment=256,
            access="read" if identity else "read_write",
        )
        pkg["buffer_contracts"].append(c)
        desc["metadata"]["buffers"].append(dict(c, data=identity) if identity else c.copy())
        desc["buffer_scopes"][name] = "weights" if identity else "workspace"

    contract("NVFP4.Scratch", "u8", [64 << 20])
    for i in range(32):
        f = encoded / f"nvfp4-layer-{i:02}.safetensors"
        shard = load_file(str(f))
        for role in ("gate", "up", "down"):
            for key, length, typ in [
                ("W", 2560 * 9216 // 2, "u8"),
                ("S", 2560 * 9216 // 16, "u8"),
                ("G", 1, "f32"),
            ]:
                name = f"NVFP4.{i}.{role}.{key}"
                v = shard[name]
                if v.shape != (length,) or v.dtype != np.dtype("u1" if typ == "u8" else "<f4"):
                    raise ValueError(name)
                contract(name, typ, [length], dict(tensor=name, sha256=digest(v)))
                index["metadata"]["total_size"] += v.nbytes
                desc["metadata"]["weight_bytes"] += v.nbytes
                index["weight_map"][name] = f.name
        share(f, weights / f.name)
    (weights / "model.safetensors.index.json").write_bytes(json_bytes(index))
    pkg["compute_policy"] = desc["compute_policy"] = POLICY
    # Prefix snapshots enumerate kv_cache.buffers. Target-only packages must not
    # advertise the four draft pages, which no target program grows or maps.
    desc["metadata"].pop("mtp")
    draft_pages = {"MtpKPages", "MtpVPages", "MtpKPagesScale", "MtpVPagesScale"}
    for name in draft_pages:
        del desc["metadata"]["kv_cache"]["buffers"][name]

    desc["metadata"]["reset_buffers"] = [
        n for n in desc["metadata"]["reset_buffers"] if n not in draft_pages
    ]
    # Indirect packet kernels retain the common column ABI. Only their address
    # slots survive; no target operation accesses draft KV data.
    for table in (desc["metadata"]["buffers"], pkg["buffer_contracts"]):
        for buf in table:
            if buf["name"] in draft_pages:
                buf["shape"] = [1]
    desc["metadata"]["weight_scope"] += (
        "; 32 target FFNs use BF16-origin NVFP4 W4A4, per-16 E4M3 and FP32 tensor scales"
    )
    b = json.loads((package / "bindings.json").read_bytes())
    cat = pkg["catalog"]
    assets = load_file(str(package / "assets.safetensors"))

    def asset(data):
        h = digest(data)
        assets[h] = np.frombuffer(data, dtype="u1")
        for i, a in enumerate(cat["assets"]):
            if a["sha256"] == h:
                return i
        cat["assets"].append(dict(file="assets.safetensors", tensor=h, sha256=h))
        return len(cat["assets"]) - 1

    abi = asset(ABI)
    native = Path(__file__).resolve().parents[4] / "kernels/cuda/nvfp4"
    body = b"".join((native / file).read_bytes() for file in ("ffn.cu", "gemm.cuh", "fht.h"))
    implementation = asset(body)
    host_abi = asset((native / "abi.h").read_bytes())
    pkg["toolchain"].update(
        nvfp4_source=digest(body), nvfp4_abi=ABI.decode(), bf16_revision=thor_weights.REVISION
    )

    def role(name):
        if name not in b["roles"]:
            b["roles"].append(name)
        return {"buffer": b["roles"].index(name)}

    templates = {t["suffix"]: t for t in b["templates"] if t["prefix"] == "qwen_rows/layer"}
    upper = {
        z["layer"]: b["parameters"][z["parameters"]][3] for z in templates["/post/up"]["layers"]
    }
    b["templates"].remove(templates["/post/up"])
    b["templates"].remove(templates["/post/swiglu_quantize"])
    b["templates"] = [t for t in b["templates"] if not t["prefix"].startswith("qwen_rows/mtp/")]
    for name in ("gate", "down"):
        gate = name == "gate"
        operation = "gate_up" if gate else "down"
        t = templates["/post/" + name]
        t["suffix"] = "/post/" + operation
        readonly = [True, True, True, True, False, False, True, True] + (
            [True, True, True, False] if gate else []
        )
        params = ["buffer"] * len(readonly)
        params[6] = "i32"
        if gate:
            params[-1] = "buffer_slice"
        definition = len(cat["definitions"])
        cat["definitions"].append(
            dict(
                launch=dict(backend="thor_nvfp4", operation=operation),
                symbol=ABI.decode(),
                block=[1, 1, 1],
                shared_memory_bytes=0,
                cooperative=False,
                module=abi,
                source=implementation,
                host_abi=host_abi,
                parameters=params,
                readonly_arguments=readonly,
            )
        )
        choices = copy.deepcopy(b["choices"][t["choice"]])
        for z in t["layers"]:
            i = z["layer"]
            out = b["parameters"][z["parameters"]][3]

            def bank(r):
                return [role(f"NVFP4.{i}.{r}.{part}") for part in ("W", "S", "G")]

            data = [
                role("Norm" if gate else "RotationValues"),
                *bank(name),
                out,
                role("NVFP4.Scratch"),
                {"i32": packet_rows},
                role("orinfer.rotation.k2560.signs" if gate else "RotationMaxima"),
            ]
            if gate:
                data.extend([*bank("up"), upper[i]])
            for c in choices:
                rows = (
                    packet_rows
                    if not c["suffix"]
                    else int(c["suffix"].split("/")[-1].split("-")[0])
                )
                value = {"i32": rows}
                if value not in data:
                    data.append(value)
                args = list(range(len(readonly)))
                args[6] = data.index(value)
                c.update(definition=definition, arguments=args)
            z["parameters"] = len(b["parameters"])
            b["parameters"].append(data)
        t["choice"] = len(b["choices"])
        b["choices"].append(choices)
    b["choices"] = keep_referenced(b["choices"], [(t, "choice") for t in b["templates"]])
    b["parameters"] = keep_referenced(
        b["parameters"], [(z, "parameters") for t in b["templates"] for z in t["layers"]]
    )
    choices = [c for group in b["choices"] for c in group]
    cat["definitions"] = keep_referenced(cat["definitions"], [(c, "definition") for c in choices])
    cat["dynamic"] = keep_referenced(
        cat["dynamic"], [(c, "dynamic") for c in choices if c.get("dynamic") is not None]
    )
    cat["assets"] = keep_referenced(
        cat["assets"],
        [(d, key) for d in cat["definitions"] for key in ("module", "source", "host_abi")],
    )
    assets = {a["tensor"]: assets[a["tensor"]] for a in cat["assets"]}
    bindings = json_bytes(b)
    pkg["bindings"]["sha256"] = digest(bindings)
    raw = json_bytes(pkg)
    identity = digest(raw)
    out = output / "execution-packages" / identity
    out.mkdir(parents=True)
    for f in package.iterdir():
        if f.is_file() and f.name not in ("package.json", "bindings.json", "assets.safetensors"):
            share(f, out / f.name)
    (out / "package.json").write_bytes(raw)
    (out / "bindings.json").write_bytes(bindings)
    save_file(assets, str(out / "assets.safetensors"))
    desc["execution_package"] = identity
    (model / "cache/model.json").write_bytes(json_bytes(desc))
    return identity


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("bf16", "e8p-model", "e8p-package", "output"):
        ap.add_argument("--" + key, type=Path, required=True)
    a = ap.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise FileExistsError(a.output)
    with tempfile.TemporaryDirectory(dir=a.output.parent) as temp:
        tmp = Path(temp)
        encoded = tmp / "encoded"
        encoded.mkdir()
        thor_weights.generate(a.bf16, a.e8p_model, encoded)
        result = tmp / "publish"
        result.mkdir()
        identity = publish(a.e8p_model, a.e8p_package, encoded, result)
        result.rename(a.output)
    print(identity)


if __name__ == "__main__":
    main()
