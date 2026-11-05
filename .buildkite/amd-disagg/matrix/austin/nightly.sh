#!/bin/bash
# nightly.sh [--scheduled] [drive.py options]: one run of the disagg gsm8k matrix
# from the Austin node. cron starts it with --scheduled (see install-cron.sh).
#
# Updates the clone in $MX_WORK/vllm to the latest csrikris-crusoe, copies its
# tracked .buildkite/amd-disagg files to experiment-logs/<run> on Crusoe and runs
# drive.py against that copy. drive.log, state, summary.md and the job logs go to
# $MX_WORK/runs/<run>, and runs/latest points there. <run> is the IST date for a
# --scheduled run and <date>-<HHMM> otherwise. drive.py options (--models,
# --layouts, --cells, --image, --router-image, --ep-gate) pass through; its MX_*
# settings can be overridden from the environment. Nothing is deleted, here or
# on Crusoe.
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin

say() { echo "$(TZ=Asia/Kolkata date '+%F %T') nightly: $*"; }

main() {
    local -a argv=("$@")
    local work="${MX_WORK:-${HOME}/work/dist-inf-work}" branch=csrikris-crusoe
    local repo="${work}/vllm" runs="${work}/runs"
    local mx="${repo}/.buildkite/amd-disagg/matrix"
    local host="${MX_HOST:-${USER:-$(id -un)}@crs-spur.crusoe.amd.com}"
    local key="${MX_SSH_KEY:-${HOME}/.ssh/id_ed25519}"
    local scheduled=0
    if [[ "${1:-}" == --scheduled ]]; then
        scheduled=1
        shift
    fi

    mkdir -p "${runs}"
    if [[ -z "${MX_NIGHTLY_LOCKED:-}" ]]; then
        exec 9>> "${runs}/nightly.lock"
        flock -n 9 || { say "another nightly.sh holds ${runs}/nightly.lock; exiting"; exit 1; }
        export MX_NIGHTLY_LOCKED=1
    fi
    # git may rewrite this file; bash has already read all of main, and the
    # updated script takes over from here.
    if [[ -z "${MX_NIGHTLY_UPDATED:-}" ]]; then
        if git -C "${repo}" fetch -q --depth 1 origin "${branch}" && git -C "${repo}" reset -q --hard FETCH_HEAD; then
            say "code: $(git -C "${repo}" log -1 --format='%h %s')"
        else
            say "WARNING: could not update ${repo} to origin/${branch}; running $(git -C "${repo}" rev-parse --short HEAD) as it is"
        fi
        export MX_NIGHTLY_UPDATED=1
        exec bash "${mx}/austin/nightly.sh" "${argv[@]}"
    fi

    local day run cells
    day="$(TZ=Asia/Kolkata date +%F)"
    run="${day}"
    if [[ "${scheduled}" == 0 || -e "${runs}/${run}" ]]; then
        run="${day}-$(TZ=Asia/Kolkata date +%H%M)"
    fi
    [[ ! -e "${runs}/${run}" ]] || { say "${runs}/${run} already exists; try again in a minute"; exit 1; }
    cells="$(python3 "${mx}/drive.py" --list "$@")"
    [[ -n "${cells}" ]] || { say "no cell matches: $*"; exit 2; }

    local remote="${MX_REMOTE_BASE:-/shared_nfs/${host%@*}/experiment-logs}/${run}"
    local rsh="ssh -i ${key} -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"
    mkdir "${runs}/${run}"
    ln -sfn "${run}" "${runs}/latest"
    # Links, permissions and times only: Crusoe's groups differ from this host's.
    stage() {
        git -C "${repo}" ls-files -z .buildkite/amd-disagg \
            | rsync -lpt "$@" --from0 --files-from=- -e "${rsh}" "${repo}/" "${host}:${remote}/"
    }
    say "run ${run}: copying ${repo}/.buildkite/amd-disagg to ${host}:${remote}"
    stage
    local differ sha
    differ="$(stage -nci)"
    [[ -z "${differ}" ]] || { say "ERROR: the copy on Crusoe differs from ${repo}:"; echo "${differ}"; exit 1; }
    sha="$(git -C "${repo}" rev-parse --short HEAD)"

    export MX_HOST="${host}" MX_SSH_KEY="${key}" MX_REMOTE_DIR="${remote}" MX_STATE_DIR="${runs}/${run}"
    export MX_COMMIT="${sha} on ${branch}"
    say "run ${run}: $(wc -l <<< "${cells}") cells; driver log ${runs}/${run}/drive.log"
    # The 8 h limit keeps a hung driver from running into the next night. On
    # TERM the driver cancels its jobs in flight and writes summary.md.
    timeout -k 10m -s TERM "${MX_TIMEOUT:-8h}" python3 "${mx}/drive.py" "$@" >> "${runs}/${run}/drive.log" 2>&1 &
    local pid=$! rc
    trap 'kill -TERM "${pid}" 2>/dev/null || true' TERM INT HUP
    while true; do
        wait "${pid}" && rc=0 || rc=$?
        kill -0 "${pid}" 2>/dev/null || break
    done
    trap - TERM INT HUP
    say "run ${run} finished with exit code ${rc}; summary: ${runs}/${run}/summary.md"
    exit "${rc}"
}

main "$@"
