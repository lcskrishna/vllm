#!/bin/bash
# status.sh <job-id>...: one "MX_STATUS" line per matrix job. The verdict sources
# match run-slurm-disagg-test.sh's WAIT=1 poll: the .disagg_done sentinel (rank-0
# rc), then the gate line. class/note are only filled in once the job is over.
source "$(dirname "$0")/crusoe.env"
L="${LOG_ROOT}"
OOM='HSA_STATUS_ERROR_OUT_OF_RESOURCES|OutOfMemoryError|(CUDA|HIP) out of memory|0 bytes free|No available memory for the cache blocks'
first() { grep -m1 -ahoE ".{0,40}($1).{0,160}" "${@:2}" 2>/dev/null | head -n1 | tr -d '\r' | tr '\t|' '  '; }
for j in "$@"; do
    log="$L/vllm-disagg-pd-$j.log"; d="$L/$j"
    st=$(timeout 15 scontrol show job "$j" 2>/dev/null | grep -oE 'JobState=[A-Z_]+' | head -n1 | cut -d= -f2)
    rc=$(tr -dc 0-9 < "$d/.disagg_done" 2>/dev/null)
    gate=$(grep -ahoE '(PASS|FAIL): (exact_match=[0-9.]+|bench=[0-9]+/[0-9]+)' "$log" 2>/dev/null | tail -n1)
    nodes=$(grep -aoE '^SLURM_JOB_NODELIST: [^ ]+' "$log" 2>/dev/null | tail -n1 | cut -d' ' -f2)
    image=$(grep -m1 -aoE 'Digest: sha256:[0-9a-f]{12}' "$log" 2>/dev/null | cut -d: -f3)
    # VLLM_PATCHES baked into the run image, e.g. "+pr65098-moriio-remote-hosts".
    patches=$(grep -m1 -ahoE 'vllm patches: (building|reusing) .*' "$log" 2>/dev/null | grep -oE '\([^)]*\)$' | tr -d '()\r')
    for p in $patches; do image="${image:--}+$(basename "$p" .diff)"; done
    need=$(grep -aoE 'waiting on [0-9]+' "$log" 2>/dev/null | grep -oE '[0-9]+' | tail -n1)
    healthy=$(grep -acE '\] healthy: ' "$log" 2>/dev/null)
    # Seconds since lm_eval or vllm bench serve last wrote progress; "-" until
    # the workload has started. levels: benchmark levels that wrote a result.
    acc=$(ls -t "$d"/accuracy_*.log "$d"/bench_*/bench_*.log 2>/dev/null | head -n1)
    idle=-; [[ -n "$acc" ]] && idle=$(( $(date +%s) - $(stat -c %Y "$acc") ))
    levels=$(ls "$d"/bench_*/bench_*.json 2>/dev/null | wc -l)
    vllm=- cls=- note=-
    case "$st" in RUNNING|COMPLETING|PENDING|CONFIGURING|REQUEUED) over=0 ;; *) over=1 ;; esac
    [[ -n "$rc" ]] && over=1
    if (( over )); then
        logs=("$log" "$d"/*.log)
        vllm=$(grep -m1 -ahoE ' version [0-9]+\.[0-9]+\.[0-9][^ ]*' "$d"/prefill_*.log 2>/dev/null | head -n1 | tr -d '\r' | awk '{print $2}')
        if [[ "$rc" == 0 ]]; then
            cls=pass
        elif n=$(first 'PREFLIGHT-REJECTED: ' "${logs[@]}"); [[ -n "$n" ]]; then
            cls=preflight note=$n
        elif n=$(first 'vllm patches \(.*\) failed|does not apply to this image' "${logs[@]}"); [[ -n "$n" ]]; then
            cls=patch note=$n
        elif n=$(first 'KV cache is needed, which is larger than the available' "${logs[@]}"); [[ -n "$n" ]]; then
            cls=kvlen note=$n
        elif n=$(first "$OOM" "${logs[@]}"); [[ -n "$n" ]]; then
            cls=oom note=$n
        elif [[ "$gate" == "FAIL: bench"* ]]; then
            cls=bench note=$(grep -ahoE 'FAIL: bench=.{0,180}' "$log" | tail -n1 | tr -d '\r' | tr '\t|' '  ')
        elif [[ "$gate" == FAIL* ]]; then
            cls=accuracy note=$gate
        elif n=$(first 'remote blocks never arrived' "${logs[@]}"); [[ -n "$n" ]]; then
            cls=router note=$n
        elif n=$(first 'TIMEOUT waiting for|exited while waiting' "$log"); [[ -n "$n" ]]; then
            cls=bringup note=$n
            e=$(first '[A-Za-z_.]+(Error|Exception): ' "$d"/*.log); [[ -n "$e" ]] && note="$note; $e"
        elif n=$(first '[A-Za-z_.]+(Error|Exception): ' "$d"/*.log); [[ -n "$n" ]]; then
            cls=error note=$n
        else
            cls=unknown
        fi
    fi
    echo "MX_STATUS $j|state=${st:--}|rc=${rc:--}|gate=${gate:--}|healthy=${healthy:-0}/${need:--}|nodes=${nodes:--}|image=${image:--}|vllm=${vllm:--}|idle=$idle|levels=$levels|class=$cls|note=$note" | tr -d '\r'
done
