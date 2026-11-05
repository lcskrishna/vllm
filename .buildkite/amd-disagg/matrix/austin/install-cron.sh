#!/bin/bash
# install-cron.sh: add the nightly matrix run to this user's crontab at 01:37 IST
# (20:07 UTC, so this host's clock must be on UTC). Other crontab entries stay,
# and a crontab that already runs nightly.sh is left as it is.
set -euo pipefail
nightly="$(cd "$(dirname "$0")" && pwd)/nightly.sh"
work="${MX_WORK:-${HOME}/work/dist-inf-work}"
[[ -x "${nightly}" ]] || { echo "${nightly} is not executable" >&2; exit 1; }
[[ "$(date +%z)" == +0000 ]] || { echo "this host's clock is on $(date +%Z), not UTC; 20:07 would not be 01:37 IST" >&2; exit 1; }
line="7 20 * * * ${nightly} --scheduled >> ${work}/runs/cron.log 2>&1"
current="$(crontab -l 2>/dev/null || true)"
if grep -qF "${nightly}" <<< "${current}"; then
    echo "already installed:"
    grep -F "${nightly}" <<< "${current}"
    exit 0
fi
mkdir -p "${work}/runs"
{ [[ -z "${current}" ]] || printf '%s\n' "${current}"; echo "${line}"; } | crontab -
echo "installed: ${line}"
