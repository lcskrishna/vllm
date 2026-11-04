#!/usr/bin/env bash
# Apply source patches to the vLLM package installed in this image, e.g. the
# vllm/ part of a PR the nightly does not have yet. run_xPyD_disagg.slurm runs
# this as root in a throwaway container and commits the result (VLLM_PATCHES).
#
#   ./apply_vllm_patches.sh patches/pr65098-moriio-remote-hosts.diff [...]
#
# Patches are `git diff` output against the vLLM repo root and may only touch
# vllm/ (tests/ and .buildkite/ are not installed). A patch that is already
# fully applied (the fix has landed in the image) is skipped; any other patch
# that does not apply exactly is an error.
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

[[ $# -gt 0 ]] || die "usage: $0 <patch>..."
[[ $EUID -eq 0 ]] || die "must run as root inside the container"

declare -a PATCHES=()
for p in "$@"; do
    [[ -f "$p" ]] || die "no such patch: $p"
    PATCHES+=("$(readlink -f "$p")")
done

# find_spec locates the package without running vllm/__init__.py.
SITE=$(python3 -c 'import importlib.util as u, os; print(os.path.dirname(u.find_spec("vllm").submodule_search_locations[0]))') \
    || die "vllm is not importable in this image"
cd "$SITE"
echo "==> vllm package root: $SITE"

if command -v git >/dev/null; then
    TOOL=git
    can_apply()   { git apply --check -p1 "$1"; }
    is_applied()  { git apply --check --reverse -p1 "$1"; }
    apply_patch() { git apply -p1 "$1"; }
elif command -v patch >/dev/null; then
    TOOL=patch
    can_apply()   { patch -p1 --forward --fuzz=0 --dry-run -s -i "$1" >/dev/null; }
    is_applied()  { patch -p1 --reverse --fuzz=0 --dry-run -s -i "$1" >/dev/null; }
    apply_patch() { patch -p1 --forward --fuzz=0 -s -i "$1"; }
else
    die "neither git nor patch is available in this image"
fi
echo "==> applying with $TOOL"

declare -a CHANGED=()
for p in "${PATCHES[@]}"; do
    mapfile -t files < <(sed -nE 's|^\+\+\+ b/(.*)$|\1|p' "$p")
    (( ${#files[@]} > 0 )) || die "$p: no files in patch"
    for f in "${files[@]}"; do
        [[ "$f" == vllm/* ]] || die "$p: touches $f, outside the installed vllm package"
    done
    if can_apply "$p" 2>/dev/null; then
        before=$(sha1sum "${files[@]}")
        apply_patch "$p"
        # git apply inside a work tree can skip paths and still succeed.
        after=$(sha1sum "${files[@]}")
        [[ $(comm -12 <(sort <<<"$before") <(sort <<<"$after")) == "" ]] \
            || die "$p: applied but some files are unchanged"
        echo "    applied $(basename "$p"): ${files[*]}"
        CHANGED+=("${files[@]}")
    elif is_applied "$p" 2>/dev/null; then
        echo "    skipped $(basename "$p"): already in the image"
    else
        can_apply "$p" || true
        die "$p does not apply to this image's vllm"
    fi
done

# Compile as root now: the serving container runs as the calling user and
# cannot write __pycache__ here.
if (( ${#CHANGED[@]} > 0 )); then
    python3 -m py_compile "${CHANGED[@]}"
    echo "==> verify"
    sha1sum "${CHANGED[@]}"
fi
