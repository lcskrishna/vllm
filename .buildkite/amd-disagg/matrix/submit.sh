#!/bin/bash
# submit.sh <env-file> <layout>: submit one run of the disagg model matrix and
# print "MX_JOB <job-id>". The env file picks the model (like dsr1_0528_env.sh);
# the layout picks the topology. Each job gets its own staged copy of the scripts.
# Optional overrides from the caller's env: KV_GIB (KV cache cap in GiB),
# MORI_IB_PATH_MTU, DRY_RUN=1, ACCURACY_LOG_SAMPLES=0 (matrix runs keep the
# gsm8k samples by default), VLLM_PATCHES (EP layouts default to the
# vllm-project/vllm#65098 MoRIIO patch; set it empty to serve the plain image),
# IMAGE / VLLM_ROUTER_IMAGE (default: the pinned digests below). MX_ID names the
# run in jobs.log, so a driver that lost the ssh session mid-submit can still
# find the job.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 <env-file> <1p1d-tp8|2p1d-tp8|1p1d-ep|2p1d-ep|1p2d-ep>" >&2; exit 2; }
MX_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$1"
case "$2" in
  1p1d-tp8) export NODES=2 xP=1 yD=1 WIDE_EP_MODE=0 ;;
  2p1d-tp8) export NODES=3 xP=2 yD=1 WIDE_EP_MODE=0 ;;
  1p1d-ep)  export NODES=2 xP=1 yD=1 WIDE_EP_MODE=1 ;;
  2p1d-ep)  export NODES=3 xP=2 yD=1 WIDE_EP_MODE=1 ;;
  1p2d-ep)  export NODES=3 xP=1 yD=2 WIDE_EP_MODE=1 ;;
  *) echo "unknown layout: $2" >&2; exit 2 ;;
esac
# The images the TP layouts and the first EP runs used (the nightly tags on
# 2026-11-04), so a nightly that moves mid-matrix cannot change two things at once.
export IMAGE="${IMAGE:-vllm/vllm-openai-rocm@sha256:2358d61b70974b700b74635953a9ab4646efa39d569ecc5c1eb1d5ab760c3466}"
export VLLM_ROUTER_IMAGE="${VLLM_ROUTER_IMAGE:-vllm/vllm-router@sha256:ff11b52acdd2112d2e569b6498e4b4d1fff63ce4454e0fb72ca2d9c2b7812c24}"
# A wide-EP role spans several nodes and vllm-router sends no per-node peer
# hosts; #65098 lets MoRIIO take them from the serve config instead.
if [[ "${WIDE_EP_MODE}" == 1 ]]; then
    export VLLM_PATCHES="${VLLM_PATCHES-patches/pr65098-moriio-remote-hosts.diff}"
fi
[[ -n "${KV_GIB:-}" ]] && export CLUSTER_SERVE_FLAGS="--kv-cache-memory-bytes $((KV_GIB << 30))"
export ACCURACY_LOG_SAMPLES="${ACCURACY_LOG_SAMPLES:-1}"
export BUILDKITE_COMMIT="mx-${MODEL_NAME}-$2-$(date +%m%d%H%M%S)"
cd ~/work/di-ci/vllm
out="$(bash .buildkite/amd-disagg/run-slurm-disagg-test.sh)" || true
echo "${out}"
job="$(sed -nE 's/^\[slurm-submit\] submitted job ([0-9]+)$/\1/p' <<< "${out}" | tail -n1)"
[[ -n "${job}" ]] || { echo "MX_SUBMIT_FAILED" >&2; exit 1; }
[[ -n "${MX_ID:-}" ]] && echo "${MX_ID} ${job} $(date -u +%FT%TZ) ${BUILDKITE_COMMIT}" >> "${MX_DIR}/jobs.log"
echo "MX_JOB ${job}"
