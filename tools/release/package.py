"""Archive helpers for offline development assets."""

import lzma
from pathlib import Path
import tarfile
import sys

ROOT = Path(__file__).resolve().parents[2]
# Support direct script invocation before loading project modules.
sys.path.insert(0, str(ROOT))
from thorustinfer_build.packaging.publication import generated_outputs  # noqa: E402
from thorustinfer_build.common.files import file_hash  # noqa: E402


def archive(directory, output, epoch):
    omitted = generated_outputs(directory)
    with (
        output.open("xb") as destination,
        lzma.LZMAFile(destination, mode="wb", preset=6) as compressed,
    ):
        with tarfile.open(fileobj=compressed, mode="w") as tar:
            for path in [directory, *sorted(directory.rglob("*"))]:
                if str(path.relative_to(directory)) in omitted:
                    continue
                info = tar.gettarinfo(str(path), arcname=str(path.relative_to(directory.parent)))
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = epoch
                if info.isfile():
                    with path.open("rb") as stream:
                        tar.addfile(info, stream)
                else:
                    tar.addfile(info)


def verify(directory):
    checksums = directory / "SHA256SUMS"
    for line in checksums.read_text().splitlines():
        digest, name = line.split("  ", 1)
        if Path(name).name != name or file_hash(directory / name) != digest:
            raise ValueError(f"Release checksum mismatch: {name}")


def notices(metadata):
    text = ["License notices for the locked Cargo dependency graph.\n"]
    for package in sorted(metadata["packages"], key=lambda p: (p["name"], p["version"])):
        text.append(
            f"\n{'=' * 72}\n{package['name']} {package['version']}\nLicense: {package.get('license')}\nRepository: {package.get('repository')}\nAuthors: {', '.join(package['authors'])}\n"
        )
        root = Path(package["manifest_path"]).parent
        paths = {
            p
            for p in root.iterdir()
            if p.is_file() and p.name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))
        }
        if package.get("license_file"):
            paths.add(root / package["license_file"])
        for path in sorted(paths):
            text.append(f"\n--- {path.name} ---\n{path.read_text(errors='replace')}\n")
    return "".join(text)
