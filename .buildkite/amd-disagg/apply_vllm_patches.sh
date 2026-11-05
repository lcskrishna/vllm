#!/usr/bin/env bash
# Apply the vLLM source patches in a directory to the vllm package installed in
# a vllm-openai-rocm container, for fixes the image does not ship yet.
#
#   ./apply_vllm_patches.sh <patch-dir>             # as root, in the image
#   ./apply_vllm_patches.sh <patch-dir> --dry-run
#
# A patch the image already carries is skipped. One that applies neither way
# fails, so a run never serves an image that silently lacks a fix; if the
# nightly took a reworked version of it, serve the image as is with
# SKIP_VLLM_PATCHES=1.
#
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

PATCH_DIR="${1:-}"
[[ -d "$PATCH_DIR" ]] || die "usage: $0 <patch-dir> [--dry-run]"
DRY=0
[[ "${2:-}" == "--dry-run" ]] && DRY=1
[[ $DRY -eq 1 || $EUID -eq 0 ]] || die "must run as root inside the container"

ROOT="$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
echo "==> vllm $(python3 -c 'import vllm; print(vllm.__version__)') at ${ROOT}/vllm"
cd "$ROOT"

shopt -s nullglob
patches=("$PATCH_DIR"/*.patch)
[[ ${#patches[@]} -gt 0 ]] || { echo "==> no patches in $PATCH_DIR"; exit 0; }

for p in "${patches[@]}"; do
    name="$(basename "$p")"
    if patch -p1 --dry-run -R -f -s <"$p" >/dev/null 2>&1; then
        echo "==> $name: already in this image, skipped"
    elif patch -p1 --dry-run --forward -s <"$p" >/dev/null 2>&1; then
        if [[ $DRY -eq 1 ]]; then
            echo "==> $name: applies (dry run)"
            continue
        fi
        patch -p1 --forward -s <"$p"
        # Drop the bytecode of what changed rather than rely on its timestamps.
        for f in $(sed -n 's|^+++ b/||p' "$p"); do
            rm -f "$(dirname "$f")/__pycache__/$(basename "$f" .py)".*.pyc
        done
        echo "==> $name: applied"
    else
        die "$name applies neither forward nor in reverse to ${ROOT}/vllm; if this image already has the fix, set SKIP_VLLM_PATCHES=1"
    fi
done
