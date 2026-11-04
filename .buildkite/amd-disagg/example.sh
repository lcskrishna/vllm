#!/bin/bash
# example.sh — one DeepSeek-R1-0528 gsm8k run on Crusoe Spur in the launcher's
# default layout (1P1D TP8). Run it on the Crusoe login node. To pin nodes, set
# e.g. NODELIST='crsuse2-m2m-v3-[081,082]'.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/matrix/crusoe.env"
export MODEL_NAME="${MODEL_NAME:-DeepSeek-R1-0528}"
export MODEL_DIR="${MODEL_DIR:-/shared_nfs/models/deepseek-ai}"
bash "${HERE}/run-slurm-disagg-test.sh"
