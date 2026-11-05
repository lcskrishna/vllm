#!/bin/bash
# submit.sh <layout>: submit one cell of the disagg model matrix and print
# "MX_JOB <job-id>". MODEL_NAME and MODEL_DIR (from the environment) pick the
# model, the layout picks the topology, and crusoe.env supplies the site
# settings. Each job gets its own staged copy of the scripts.
# Optional overrides from the caller's env: KV_GIB (KV cache cap in GiB),
# MORI_IB_PATH_MTU, DRY_RUN=1, ACCURACY_LOG_SAMPLES=0 (matrix runs keep the
# gsm8k samples by default), VLLM_PATCHES (EP layouts default to the
# vllm-project/vllm#65098 MoRIIO patch; set it empty to serve the plain image),
# IMAGE / VLLM_ROUTER_IMAGE (default: the launcher's nightly tags), and any
# crusoe.env setting. MX_ID names the run in jobs.log, so a driver that lost the
# ssh session mid-submit can still find the job.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: MODEL_NAME=... MODEL_DIR=... $0 <1p1d-tp8|2p1d-tp8|1p1d-ep|2p1d-ep|1p2d-ep>" >&2; exit 2; }
: "${MODEL_NAME:?set MODEL_NAME}" "${MODEL_DIR:?set MODEL_DIR}"
MX_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${MX_DIR}/crusoe.env"
case "$1" in
  1p1d-tp8) export NODES=2 xP=1 yD=1 WIDE_EP_MODE=0 ;;
  2p1d-tp8) export NODES=3 xP=2 yD=1 WIDE_EP_MODE=0 ;;
  1p1d-ep)  export NODES=2 xP=1 yD=1 WIDE_EP_MODE=1 ;;
  2p1d-ep)  export NODES=3 xP=2 yD=1 WIDE_EP_MODE=1 ;;
  1p2d-ep)  export NODES=3 xP=1 yD=2 WIDE_EP_MODE=1 ;;
  *) echo "unknown layout: $1" >&2; exit 2 ;;
esac
# A wide-EP role spans several nodes and vllm-router sends no per-node peer
# hosts; #65098 lets MoRIIO take them from the serve config instead.
if [[ "${WIDE_EP_MODE}" == 1 ]]; then
    export VLLM_PATCHES="${VLLM_PATCHES-patches/pr65098-moriio-remote-hosts.diff}"
fi
[[ -n "${KV_GIB:-}" ]] && export CLUSTER_SERVE_FLAGS="--kv-cache-memory-bytes $((KV_GIB << 30))"
export ACCURACY_LOG_SAMPLES="${ACCURACY_LOG_SAMPLES:-1}"
export BUILDKITE_COMMIT="mx-${MODEL_NAME}-$1-$(date +%m%d%H%M%S)"
cd "${MX_DIR}/../../.."
out="$(bash .buildkite/amd-disagg/run-slurm-disagg-test.sh)" || true
echo "${out}"
job="$(sed -nE 's/^\[slurm-submit\] submitted job ([0-9]+)$/\1/p' <<< "${out}" | tail -n1)"
[[ -n "${job}" ]] || { echo "MX_SUBMIT_FAILED" >&2; exit 1; }
[[ -n "${MX_ID:-}" ]] && echo "${MX_ID} ${job} $(date -u +%FT%TZ) ${BUILDKITE_COMMIT}" >> "${MX_DIR}/jobs.log"
echo "MX_JOB ${job}"
