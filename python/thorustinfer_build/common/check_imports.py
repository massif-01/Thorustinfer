"""Check offline module references in the CUDA uv environment without using a GPU."""

import argparse
from importlib import import_module

from thorustinfer_build.common.paths import ROOT


def source_modules(root):
    modules = []
    for directory in ("python/thorustinfer_build", "kernels", "tools", "tests"):
        if not (root / directory).exists():
            continue
        for current, directories, files in (root / directory).walk():
            directories[:] = [
                name
                for name in directories
                if not name.startswith(".") and name not in {"__pycache__", "environments"}
            ]
            modules.extend(
                current / name
                for name in files
                if name.endswith(".py") and not name.startswith("test_") and name != "__init__.py"
            )
    return sorted(modules)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cuda"], default="cuda")
    args = parser.parse_args()
    root = ROOT
    modules = source_modules(root)
    failures = []
    imported = 0
    for path in modules:
        relative = path.relative_to(root)
        if relative.parts[0] == "python":
            relative = relative.relative_to("python")
        name = ".".join(relative.with_suffix("").parts)
        try:
            import_module(name)
            imported += 1
        except Exception as error:
            failures.append(f"{name}: {type(error).__name__}: {error}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Imported {imported} {args.backend} offline modules")


if __name__ == "__main__":
    main()
