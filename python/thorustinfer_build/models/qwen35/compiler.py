"""Reuse Flash Next's content cache and host-ABI emitter for native Qwen AOT."""

from thorustinfer_build.compiler.cuda import target as cuda_target
from thorustinfer_build.common.paths import ROOT


from thorustinfer_build.compiler.build_cache import BuildCache
from thorustinfer_build.compiler.host_compiler import Compiler
from thorustinfer_build.tuning.row_convergence import divide_rows
from thorustinfer_build.compiler.cuda.common import export_kernel
from thorustinfer_build.common.files import source_hash


class Aot:
    def __init__(self, root, output, cache, buffers, *, compile_lock):
        self.root, self.output, self.lock = root, output, compile_lock
        self.output.mkdir(parents=True, exist_ok=True)
        self.cache = BuildCache(cache)
        self.compiler = Compiler(root, root, output, buffers)
        self.kernels, self.dynamic = [], []
        self.objects = {}
        self.recipes = {}
        from thorustinfer_build.common.sources import source_closure

        sources = [
            ROOT / "tools/environments/cuda/uv.lock",
            *source_closure(
                [
                    "thorustinfer_build.models.qwen35.compiler",
                    "thorustinfer_build.models.qwen35.build",
                    "thorustinfer_build.models.nemotron_embed.build",
                    "thorustinfer_build.models.qwen_reranker.build",
                ]
            ),
        ]
        import torch
        import tilelang

        self.identity = dict(
            source=source_hash(ROOT, sources),
            tilelang=tilelang.__version__,
            cuda=torch.version.cuda,
            target=cuda_target.architecture(),
        )

    def emit(
        self,
        name,
        key,
        build,
        bindings,
        rows=4096,
        *,
        dynamic=True,
        row_divisor=1,
        row_multiplier=1,
        constants=None,
    ):
        self.recipes[name] = (key, build, bindings, rows, dynamic, row_divisor, row_multiplier)
        if (
            row_multiplier < 1
            or rows % row_multiplier
            or (row_multiplier != 1 and row_divisor != 1)
        ):
            raise ValueError("Invalid dynamic assignment row multiplier")
        if key not in self.objects:

            def construct(folder):
                with self.lock():
                    kernel = build()
                    export_kernel(kernel, folder)

            artifact, hit = self.cache.get(dict(**self.identity, operator=key), construct)
            self.objects[key] = artifact
            print("AOT", key, "cached" if hit else "compiled", flush=True)
        artifact = self.objects[key]
        kernel, launch = self.compiler.emit(
            dict(name=name),
            name,
            {
                k: (dict(kind="buffer", name=v) if isinstance(v, str) else v)
                for k, v in bindings.items()
            },
            rows // row_multiplier,
            multiplier=row_multiplier,
            key=key,
            artifact=artifact,
            constants=constants,
        )
        self.kernels.append(kernel)
        if dynamic:
            if row_divisor != 1:
                launch = divide_rows(launch, row_divisor)
                launch["capacity"] = rows * row_divisor
            self.dynamic.append(launch)
        return dict(kind="kernel", name=name)
