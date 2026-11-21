"""Package a local DFlash, DFlash2 or DSpark checkpoint and its BF16 projections."""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from safetensors import safe_open
import tilelang

from kernels.cuda.model.block_draft import projection
from thorustinfer_build.common.files import file_hash
from thorustinfer_build.compiler.cuda import target
from thorustinfer_build.compiler.cuda.common import configure, export_kernel
from thorustinfer_build.contracts.abi import parse_host, validate_parameter_count


def verifier_identity(target_model):
    origin = json.loads((target_model / "config.json").read_bytes())["quantization_config"]
    return [origin["source"], origin["source_revision"]]


def build(source, target_model, output, cache):
    weights = source / "model.safetensors"
    with safe_open(weights, framework="pt") as tensors:
        shapes = {name: tensors.get_slice(name).get_shape() for name in tensors.keys()}
    configs = {}
    for name, shape in shapes.items():
        if (
            len(shape) != 2
            or not name.endswith(".weight")
            or name.startswith(("confidence_head.", "markov_head.markov_w1"))
        ):
            continue
        n, k = shape
        tag = f"projection-{n}-{k}"
        if tag in configs:
            continue
        work = cache / tag
        # TileLang's cache includes lowered code and compiler options.
        fn = projection(n, k)
        compiled = tilelang.compile(fn, out_idx=[], execution_backend="nvrtc", **target.options())
        export_kernel(compiled, work)
        (abi,) = parse_host((work / "host.txt").read_text())
        validate_parameter_count((work / "kernel.cu").read_text(), abi)
        cubin = work / "kernel.cubin"
        digest = file_hash(cubin)
        shutil.copyfile(cubin, output / (digest + ".cubin"))
        configs[tag] = dict(n=n, k=k, abi=abi, cubin=digest)
    shutil.copyfile(source / "config.json", output / "config.json")
    if (source / "LICENSE").is_file():
        shutil.copyfile(source / "LICENSE", output / "LICENSE")
    try:
        os.link(weights, output / weights.name)
    except OSError:
        shutil.copyfile(weights, output / weights.name)
    package = dict(
        schema=1,
        target=target.architecture(),
        verifier=verifier_identity(target_model),
        config_sha256=file_hash(source / "config.json"),
        weight_sha256=file_hash(weights),
        projections=list(configs.values()),
    )
    (output / "draft.json").write_text(json.dumps(package, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "target-model", "output", "cache"):
        p.add_argument("--" + arg, type=Path, required=True)
    a = p.parse_args()
    configure()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise FileExistsError(a.output)
    with tempfile.TemporaryDirectory(dir=a.output.parent) as stage:
        temp = Path(stage) / "package"
        temp.mkdir()
        build(a.source, a.target_model, temp, a.cache)
        temp.rename(a.output)


if __name__ == "__main__":
    main()
