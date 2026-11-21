#!/bin/bash
set -euo pipefail
if (( $# > 1 )); then echo 'Usage: bash install.sh [INSTALL_DIRECTORY]' >&2; exit 2; fi
bundle=$(cd "$(dirname "$0")" && pwd)
prefix=$(realpath -m -- "${1:-$HOME/.local/opt/thorustinfer}")
if [[ -e "$prefix" ]]; then echo "thorustinfer: choose a new installation directory: $prefix" >&2; exit 1; fi
mkdir -p -- "$(dirname "$prefix")"
stage=$(mktemp -d -- "$(dirname "$prefix")/.thorustinfer-XXXXXX")
trap 'if [[ -d "$stage" ]]; then rm -rf -- "$stage"; fi' EXIT
(cd "$bundle" && sha256sum --quiet -c SHA256SUMS)
cp -a -- "$bundle"/{bin,lib,licenses,README.md} "$stage/"
mv -Tn -- "$stage" "$prefix"
if [[ -e "$stage" ]]; then echo "thorustinfer: installation directory was claimed: $prefix" >&2; exit 1; fi
printf 'Installed thorustinfer in %s\nRun: %s/bin/thorustinfer info\n' "$prefix" "$prefix"
