#!/bin/bash
# Build the independent engine from a source tree. No model, GPU lock or checkpoint is needed.
set -euo pipefail
if (( $# < 1 || $# > 2 )); then echo 'Usage: bash tools/thor/build.sh NEW_OUTPUT_DIRECTORY [CORRESPONDING_SOURCE_ARCHIVE]' >&2; exit 2; fi
root=$(cd "$(dirname "$0")/../.." && pwd)
out=$(realpath -m -- "$1")
source_archive=
if (( $# == 2 )); then source_archive=$(realpath -e -- "$2"); fi
: "${CUDA_HOME:?Set CUDA_HOME to the CUDA toolkit directory}"
: "${CUTLASS_ROOT:?Set CUTLASS_ROOT to the directory containing include/cutlass}"
mkdir -- "$out"
cd "$root"
python3 -m tools.dev.dependencies check
export CARGO_TARGET_DIR="$out/target"
cargo build --locked --offline --release --bin thorustinfer
exe="$CARGO_TARGET_DIR/release/thorustinfer"
version=$("$exe" --version)
[[ "$version" == 'thorustinfer '* ]]
bundle="thorustinfer-${version#thorustinfer }-linux-aarch64"
stage="$out/$bundle"
mkdir -p "$stage"/{bin,lib,licenses}
install -m 755 "$exe" "$stage/bin/thorustinfer"
bash tools/thor/build-provider.sh "$stage/lib/libthorustinfer_nvfp4_v1.so"
cp -- tools/thor/{README.md,install.sh} "$stage/"
cp -- COPYING LICENSE THIRD_PARTY_NOTICES.md tools/thor/licenses/* "$stage/licenses/"
cargo metadata --format-version 1 --locked --offline --filter-platform aarch64-unknown-linux-gnu > "$out/cargo-metadata.json"
PYTHONPATH="$root/python:$root" python3 - "$out/cargo-metadata.json" > "$stage/licenses/Rust-notices.txt" <<'PY'
import json, sys
from pathlib import Path
from tools.release.package import notices
print(notices(json.loads(Path(sys.argv[1]).read_text())))
PY
if [[ -n "$source_archive" ]]; then
    source_name="thorustinfer-${version#thorustinfer }-source.tar.xz"
    if [[ "$(basename -- "$source_archive")" != "$source_name" ]]; then
        echo "Expected corresponding source archive named $source_name" >&2
        exit 2
    fi
    (cd "$(dirname -- "$source_archive")" && sha256sum -- "$source_name") > "$stage/licenses/CORRESPONDING-SOURCE.sha256"
fi
(cd "$stage" && find bin lib licenses install.sh README.md -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
(cd "$out" && XZ_OPT="-3 -T2" tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -cJf "$bundle.tar.xz" "$bundle" && sha256sum "$bundle.tar.xz" > runtime.sha256)
